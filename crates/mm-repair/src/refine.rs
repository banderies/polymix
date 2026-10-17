//! Local refinement of a hole-fill patch: split long edges, Delaunay flips and relaxation,
//! touching only triangles of the patch (the boundary loop and the surrounding mesh stay
//! exactly as they were). Optionally interleaves fairing solves between refinement levels
//! (a cascadic coarse-to-fine scheme: each level warm-starts the next, so the final
//! fine-level solve needs few iterations).
//!
//! This plays the role of the refinement step in Liepa, "Filling Holes in Meshes"
//! (SGP 2003) and of geometry3Sharp's `SmoothedHoleFill` remesh pass.

use crate::laplacian::{cot, fair_vertices, fair_vertices_along, FairDirection, LaplacianWeights};
use crate::sparse::PcgParams;
use crate::util::{FxHashMap, FxHashSet};
use glam::DVec3;
use mm_core::mesh::{tri_edge_index, tri_other_vertex, Mesh};
use mm_core::INVALID;

/// Triangles and interior vertices created by a fill, plus a target edge length
/// ("scale attribute" in Liepa's terms) per vertex.
#[derive(Default)]
pub(crate) struct Patch {
    pub tris: FxHashSet<u32>,
    pub verts: Vec<u32>,
    vset: FxHashSet<u32>,
    size: FxHashMap<u32, f64>,
}

impl Patch {
    pub fn set_size(&mut self, v: u32, s: f64) {
        self.size.insert(v, s);
    }
    #[inline]
    fn size_of(&self, v: u32, default: f64) -> f64 {
        self.size.get(&v).copied().unwrap_or(default)
    }
    pub fn add_tri(&mut self, t: u32) {
        self.tris.insert(t);
    }
    pub fn add_vert(&mut self, v: u32) {
        if self.vset.insert(v) {
            self.verts.push(v);
        }
    }
    /// Edge with both triangles in the patch (so it may be split or flipped).
    #[inline]
    pub fn is_inner_edge(&self, m: &Mesh, e: u32) -> bool {
        let [t0, t1] = m.edge_triangles(e);
        t1 != INVALID && self.tris.contains(&t0) && self.tris.contains(&t1)
    }
    pub fn inner_edges(&self, m: &Mesh) -> Vec<u32> {
        let mut v = Vec::with_capacity(self.tris.len() * 3 / 2);
        for &t in &self.tris {
            for e in m.tri_edges(t) {
                if self.is_inner_edge(m, e) {
                    v.push(e);
                }
            }
        }
        v.sort_unstable();
        v.dedup();
        v
    }
    /// Triangles sorted by id (deterministic output order).
    pub fn sorted_tris(&self) -> Vec<u32> {
        let mut v: Vec<u32> = self.tris.iter().copied().collect();
        v.sort_unstable();
        v
    }
}

/// How interior patch vertices are relaxed.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Relax {
    /// Keep interior vertices on this plane.
    Plane { origin: DVec3, normal: DVec3 },
    /// Move within the local tangent plane (no shrinkage, keeps the faired shape).
    Tangential,
    /// Plain umbrella smoothing without fold checks (untangles fan fallbacks).
    Free,
}

/// Fairing solve interleaved with refinement levels.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Cascade {
    pub order: u32,
    pub weights: LaplacianWeights,
    /// Move vertices only along their normals (otherwise a full 3D solve).
    pub along_normals: bool,
}

pub(crate) struct RefineParams {
    /// Edge length used for vertices without a size, and the value sizes grade towards.
    pub target: f64,
    /// New vertices get `(1 − grading) · mean(endpoint sizes) + grading · target`:
    /// 0 interpolates the boundary density inwards (Liepa), > 0 grades towards `target`.
    pub grading: f64,
    pub relax: Relax,
    pub cascade: Option<Cascade>,
    pub max_levels: usize,
}

/// Relative tolerance of the per-level cascade solves.
const CASCADE_TOL: f64 = 1e-6;

/// Iteration budget for one fairing solve on `n` unknowns (a safety net: with the AMG
/// preconditioner used for large systems the solves converge in well under 100
/// iterations; small Jacobi-preconditioned systems get up to 2000).
pub(crate) fn solve_iteration_cap(n: usize) -> usize {
    (6_000_000 / n.max(1)).clamp(400, 2000)
}

/// Run one fairing solve of the given kind on the patch interior.
pub(crate) fn fair_patch(mesh: &mut Mesh, patch: &Patch, c: &Cascade, pcg: &PcgParams) -> Option<crate::sparse::PcgStats> {
    if c.along_normals {
        fair_vertices_along(mesh, &patch.verts, c.order, c.weights, FairDirection::VertexNormals, pcg).ok()
    } else {
        fair_vertices(mesh, &patch.verts, c.order, c.weights, pcg, None).ok()
    }
}

const RELAX_STEP: f64 = 0.6;

pub(crate) fn refine_patch(mesh: &mut Mesh, patch: &mut Patch, p: &RefineParams) {
    for _level in 0..p.max_levels {
        let n_split = split_pass(mesh, patch, p);
        flip_pass(mesh, patch, 4);
        relax_pass(mesh, patch, &p.relax, 2);
        flip_pass(mesh, patch, 2);
        // A level that grew the patch by less than 10% is close to the final size field:
        // finish splitting without re-fairing (the caller's final solve covers it).
        let significant = n_split * 10 > patch.verts.len() && n_split > 2;
        if let Some(c) = &p.cascade {
            if significant {
                let cap = solve_iteration_cap(patch.verts.len());
                let _ = fair_patch(mesh, patch, c, &PcgParams { tolerance: CASCADE_TOL, max_iterations: cap });
            }
        }
        if !significant {
            for _ in 0..8 {
                if split_pass(mesh, patch, p) == 0 {
                    break;
                }
            }
            break;
        }
    }
    for _ in 0..3 {
        flip_pass(mesh, patch, 3);
        relax_pass(mesh, patch, &p.relax, 2);
    }
    flip_pass(mesh, patch, 3);
}

fn split_pass(mesh: &mut Mesh, patch: &mut Patch, p: &RefineParams) -> usize {
    let relax = &p.relax;
    // Split when longer than 4/3 of the local size (the usual remeshing threshold).
    // In plane mode lengths are measured in the plane: loop vertices may sit off the plane,
    // and that offset must not count (splitting could never shorten it).
    let plane_n = if let Relax::Plane { normal, .. } = relax { Some(*normal) } else { None };
    let excess = |mesh: &Mesh, patch: &Patch, e: u32| -> f64 {
        let [a, b] = mesh.edge_vertices(e);
        let s = 0.5 * (patch.size_of(a, p.target) + patch.size_of(b, p.target));
        let d = mesh.vertex(a) - mesh.vertex(b);
        let len = match plane_n {
            Some(n) => (d - n * d.dot(n)).length(),
            None => d.length(),
        };
        len / (s * 4.0 / 3.0)
    };
    let mut cand: Vec<(f64, u32)> = patch
        .inner_edges(mesh)
        .into_iter()
        .filter_map(|e| {
            let x = excess(mesh, patch, e);
            (x > 1.0).then_some((x, e))
        })
        .collect();
    cand.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    let mut n = 0;
    for (_, e) in cand {
        if !mesh.is_edge(e) || !patch.is_inner_edge(mesh, e) || excess(mesh, patch, e) <= 1.0 {
            continue;
        }
        let [a, b] = mesh.edge_vertices(e);
        let s_mid = 0.5 * (patch.size_of(a, p.target) + patch.size_of(b, p.target));
        if let Ok(info) = mesh.split_edge(e) {
            patch.add_vert(info.v_new);
            patch.set_size(info.v_new, (1.0 - p.grading) * s_mid + p.grading * p.target);
            for t in info.t_new {
                if t != INVALID {
                    patch.add_tri(t);
                }
            }
            if let Relax::Plane { origin, normal } = relax {
                let q = mesh.vertex(info.v_new);
                mesh.set_vertex(info.v_new, q - *normal * (q - *origin).dot(*normal));
            }
            n += 1;
        }
    }
    n
}

/// Flip patch edges that violate the (intrinsic) Delaunay criterion, when the flip does
/// not fold the surface.
pub(crate) fn flip_pass(mesh: &mut Mesh, patch: &Patch, sweeps: usize) -> usize {
    let mut total = 0;
    for _ in 0..sweeps {
        let mut n = 0;
        for e in patch.inner_edges(mesh) {
            if patch.is_inner_edge(mesh, e) && should_flip(mesh, e) && mesh.flip_edge(e).is_ok() {
                n += 1;
            }
        }
        total += n;
        if n == 0 {
            break;
        }
    }
    total
}

fn should_flip(mesh: &Mesh, e: u32) -> bool {
    let [t0, t1] = mesh.edge_triangles(e);
    let [a, b] = mesh.edge_vertices(e);
    let tv0 = mesh.triangle(t0);
    let Some(j) = tri_edge_index(tv0, a, b) else { return false };
    let (p, q, c) = (tv0[j], tv0[(j + 1) % 3], tv0[(j + 2) % 3]);
    let d = tri_other_vertex(mesh.triangle(t1), a, b);
    if c == d || d == INVALID {
        return false;
    }
    let (pp, pq, pc, pd) = (mesh.vertex(p), mesh.vertex(q), mesh.vertex(c), mesh.vertex(d));
    if cot(pp - pc, pq - pc) + cot(pp - pd, pq - pd) >= -1e-6 {
        return false;
    }
    let n_old = ((pq - pp).cross(pc - pp) + (pp - pq).cross(pd - pq)).normalize_or_zero();
    let n1 = (pp - pc).cross(pd - pc).normalize_or_zero();
    let n2 = (pq - pd).cross(pc - pd).normalize_or_zero();
    n1.dot(n_old) > 0.05 && n2.dot(n_old) > 0.05 && n1.dot(n2) > 0.05
}

/// Would moving `v` to `target` keep every incident triangle's orientation?
fn move_keeps_orientation(mesh: &Mesh, v: u32, target: DVec3) -> bool {
    for t in mesh.vertex_triangles(v) {
        let tv = mesh.triangle(t);
        let p = mesh.tri_positions(t);
        let old = (p[1] - p[0]).cross(p[2] - p[0]);
        let mut q = p;
        for k in 0..3 {
            if tv[k] == v {
                q[k] = target;
            }
        }
        let new = (q[1] - q[0]).cross(q[2] - q[0]);
        if new.dot(old) <= 0.05 * old.length() * new.length() {
            return false;
        }
    }
    true
}

pub(crate) fn relax_pass(mesh: &mut Mesh, patch: &Patch, relax: &Relax, iters: usize) {
    for _ in 0..iters {
        for &v in &patch.verts {
            let p = mesh.vertex(v);
            let mut c = DVec3::ZERO;
            let mut k = 0usize;
            for nb in mesh.vertex_neighbors(v) {
                c += mesh.vertex(nb);
                k += 1;
            }
            if k == 0 {
                continue;
            }
            let d = (c / k as f64 - p) * RELAX_STEP;
            let target = match relax {
                Relax::Plane { origin, normal } => {
                    let q = p + d;
                    q - *normal * (q - *origin).dot(*normal)
                }
                Relax::Tangential => {
                    let n = mesh.vertex_normal(v);
                    p + (d - n * n.dot(d))
                }
                Relax::Free => p + d,
            };
            if !matches!(relax, Relax::Free) && !move_keeps_orientation(mesh, v, target) {
                continue;
            }
            mesh.set_vertex(v, target);
        }
    }
}
