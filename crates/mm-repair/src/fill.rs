//! Hole filling (Meshmixer Inspector fill types *Minimal*, *Flat* and *Smooth*).
//!
//! References: P. Liepa, "Filling Holes in Meshes", SGP 2003 (minimum-weight
//! triangulation with the (max dihedral angle, area) weight, patch refinement and fairing);
//! geometry3Sharp's `MinimalHoleFill` / `SmoothedHoleFill` / `PlanarHoleFiller`
//! (Boost License, R. Schmidt) for the overall structure. The code here is independent.
//!
//! Pipeline for one simple boundary loop:
//! 1. initial triangulation without new vertices: Liepa's O(n³) dynamic program (Minimal
//!    fills up to [`DP_MAX_LOOP`] vertices, and any loop of that size whose projection onto
//!    the best-fit plane is not simple), ear clipping in the best-fit plane when the
//!    projected loop is simple, otherwise a centroid fan (one new vertex). Chords that
//!    already exist as mesh edges are never used, so no non-manifold edge can appear.
//! 2. Refinement (Flat/Smooth, and Minimal when the fan fallback was needed): split patch
//!    edges longer than 4/3 of a per-vertex size field (loop vertices: their local edge
//!    length; new vertices: interpolated, Liepa-style, or graded towards an explicit
//!    target), Delaunay flips and relaxation. Only patch triangles are touched.
//! 3. Fairing. *Minimal*: none (the fan fallback is relaxed into a membrane so it is not a
//!    cone). *Flat*: interior vertices are kept in the best-fit plane during refinement,
//!    then their heights solve a membrane (harmonic) system with the loop fixed, using
//!    weights of the planar layout — exactly planar for planar loops, a minimal-surface-like
//!    sheet otherwise. *Smooth*: a thin-plate (bilaplacian, cotangent) system with the loop
//!    and the surrounding ring of original vertices fixed, so the patch continues the
//!    curvature around the hole; solves are interleaved with the refinement levels
//!    (cascadic coarse-to-fine) for speed on large holes.
//!
//! Loops that visit a vertex more than once (bowtie vertices) are split into simple
//! sub-loops which are filled independently. Every fill is atomic: on error the mesh is
//! left exactly as before.

use crate::laplacian::{fair_vertices_along, FairDirection, LaplacianWeights};
use crate::refine::{fair_patch, refine_patch, Cascade, Patch, Relax, RefineParams};
use crate::sparse::PcgParams;
use crate::util::{orient2, polygon_is_simple, signed_area2, FxHashMap, PlaneFrame};
use crate::{RepairError, Result};
use glam::{DVec2, DVec3};
use mm_core::mesh::Mesh;
use mm_core::topology::EdgeLoop;

/// Loops up to this many vertices are triangulated with Liepa's dynamic program.
pub const DP_MAX_LOOP: usize = 350;

/// Hole fill type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FillMode {
    /// Triangulate the loop without adding vertices (minimum dihedral angle, then area).
    Minimal,
    /// Planar-ish, refined to the target edge length (membrane surface).
    Flat,
    /// Refined and faired so it continues the surrounding curvature (thin plate).
    #[default]
    Smooth,
}

/// Parameters of [`fill_hole`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FillParams {
    /// Fill type.
    pub mode: FillMode,
    /// Edge length of the refined patch (Flat/Smooth). Default: mean loop edge length.
    pub target_edge_length: Option<f64>,
    /// Face group of the new triangles. Default: a new group (`max_group_id + 1`).
    pub group: Option<i32>,
    /// Laplacian weights for the Smooth fairing solve.
    pub smooth_weights: LaplacianWeights,
}

impl Default for FillParams {
    fn default() -> Self {
        Self { mode: FillMode::Smooth, target_edge_length: None, group: None, smooth_weights: LaplacianWeights::Cotangent }
    }
}

impl FillParams {
    /// Default parameters with the given mode.
    pub fn new(mode: FillMode) -> Self {
        Self { mode, ..Default::default() }
    }
}

/// Elements created by a fill.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FillResult {
    /// Triangles of the patch (sorted).
    pub new_triangles: Vec<u32>,
    /// Vertices created inside the patch.
    pub new_vertices: Vec<u32>,
}

/// Fill the hole bounded by `lp` (as returned by [`mm_core::topology::boundary_loops`]).
///
/// The loop is validated against the current mesh (every consecutive pair must be a
/// boundary edge); a reversed loop is accepted. On error the mesh is unchanged.
pub fn fill_hole(mesh: &mut Mesh, lp: &EdgeLoop, params: &FillParams) -> Result<FillResult> {
    let verts = oriented_loop_vertices(mesh, &lp.vertices)?;
    if let Some(l) = params.target_edge_length {
        if !(l > 0.0 && l.is_finite()) {
            return Err(RepairError::InvalidParameter("target_edge_length must be positive".into()));
        }
    }
    let group = params.group.unwrap_or_else(|| mesh.max_group_id() + 1);
    let cycles = split_simple_cycles(&verts);
    let mut result = FillResult::default();
    for cyc in &cycles {
        let mut part = FillResult::default();
        if let Err(e) = fill_simple_loop(mesh, cyc, params, group, &mut part) {
            // Roll back sub-loops filled so far (removing the triangles frees the new vertices).
            mesh.remove_triangles(result.new_triangles.iter().copied());
            return Err(e);
        }
        result.new_triangles.extend(part.new_triangles);
        result.new_vertices.extend(part.new_vertices);
    }
    Ok(result)
}

/// Validate a loop and return its vertices in boundary orientation (each `v[i] → v[i+1]`
/// as in the adjacent mesh triangle).
pub(crate) fn oriented_loop_vertices(mesh: &Mesh, v: &[u32]) -> Result<Vec<u32>> {
    let n = v.len();
    if n < 3 {
        return Err(RepairError::InvalidLoop(format!("loop has {n} vertices")));
    }
    let (mut fwd, mut rev) = (0usize, 0usize);
    for i in 0..n {
        let (a, b) = (v[i], v[(i + 1) % n]);
        if !mesh.is_vertex(a) {
            return Err(RepairError::InvalidLoop(format!("vertex {a} does not exist")));
        }
        let e = mesh.find_edge(a, b).ok_or_else(|| RepairError::InvalidLoop(format!("no edge between {a} and {b}")))?;
        if !mesh.is_boundary_edge(e) {
            return Err(RepairError::InvalidLoop(format!("edge {a}-{b} is not a boundary edge")));
        }
        if mesh.oriented_edge_vertices(e)[0] == a {
            fwd += 1;
        } else {
            rev += 1;
        }
    }
    let mut out = v.to_vec();
    if rev > fwd {
        out.reverse();
    }
    Ok(out)
}

/// Decompose a closed walk into simple cycles (preserving orientation).
pub(crate) fn split_simple_cycles(v: &[u32]) -> Vec<Vec<u32>> {
    let mut out = Vec::new();
    let mut stack: Vec<u32> = Vec::with_capacity(v.len());
    let mut pos: FxHashMap<u32, usize> = FxHashMap::default();
    for &x in v.iter().chain(std::iter::once(&v[0])) {
        if let Some(&i) = pos.get(&x) {
            let cyc: Vec<u32> = stack[i..].to_vec();
            for &y in &stack[i + 1..] {
                pos.remove(&y);
            }
            stack.truncate(i + 1);
            if cyc.len() >= 3 {
                out.push(cyc);
            }
        } else {
            pos.insert(x, stack.len());
            stack.push(x);
        }
    }
    out
}

fn fill_simple_loop(mesh: &mut Mesh, boundary: &[u32], params: &FillParams, group: i32, out: &mut FillResult) -> Result<()> {
    let n = boundary.len();
    // Fill triangles traverse the loop in reverse; `poly` is the loop in fill orientation.
    let poly: Vec<u32> = boundary.iter().rev().copied().collect();
    let pos: Vec<DVec3> = poly.iter().map(|&v| mesh.vertex(v)).collect();
    if n == 3 {
        if triangle_exists(mesh, poly[0], poly[1], poly[2]) {
            return Err(RepairError::IsolatedTriangle);
        }
        let t = mesh.append_triangle([poly[0], poly[1], poly[2]], group)?;
        out.new_triangles.push(t);
        return Ok(());
    }
    let perimeter: f64 = (0..n).map(|i| pos[i].distance(pos[(i + 1) % n])).sum();
    let target = params.target_edge_length.unwrap_or(perimeter / n as f64).max(1e-12);
    let frame = PlaneFrame::from_polygon(&pos);
    let q2: Vec<DVec2> = pos.iter().map(|&p| frame.project(p)).collect();
    let simple = polygon_is_simple(&q2) && signed_area2(&q2) > 0.0;

    let forbidden = |i: usize, j: usize| -> bool {
        let d = i.abs_diff(j);
        d != 1 && d != n - 1 && mesh.find_edge(poly[i], poly[j]).is_some()
    };
    let mut tris: Option<Vec<[usize; 3]>> = None;
    // Minimal fills keep their initial triangulation, so they get the O(n³) dynamic
    // program; Flat/Smooth patches are re-meshed and faired anyway, so a fold-free planar
    // ear clipping (O(n²)) is the better start whenever the projection is simple.
    let prefer_planar = simple && (params.mode != FillMode::Minimal || n > DP_MAX_LOOP);
    if prefer_planar {
        tris = ear_clip(&q2, &forbidden);
    }
    if tris.is_none() && n <= DP_MAX_LOOP {
        tris = dp_triangulate(mesh, &poly, &pos, &forbidden);
    }
    if tris.is_none() && simple && !prefer_planar {
        tris = ear_clip(&q2, &forbidden);
    }
    let fan = tris.is_none();

    let mut patch = Patch::default();
    let mut appended: Vec<u32> = Vec::new();
    let add = |mesh: &mut Mesh, tv: [u32; 3], appended: &mut Vec<u32>| -> Result<()> {
        let t = mesh.append_triangle(tv, group)?;
        appended.push(t);
        Ok(())
    };
    let res: Result<()> = (|| {
        if fan {
            let c = pos.iter().copied().sum::<DVec3>() / n as f64;
            let c = if params.mode != FillMode::Smooth && simple { frame.project_to_plane(c) } else { c };
            let vc = mesh.append_vertex(c);
            patch.add_vert(vc);
            for i in 0..n {
                add(mesh, [poly[i], poly[(i + 1) % n], vc], &mut appended)?;
            }
        } else {
            for &[a, b, c] in tris.as_ref().unwrap() {
                add(mesh, [poly[a], poly[b], poly[c]], &mut appended)?;
            }
        }
        Ok(())
    })();
    if let Err(e) = res {
        mesh.remove_triangles(appended.iter().copied());
        // An unused fan centre is isolated; removing triangles does not free it.
        for &v in &patch.verts {
            if mesh.is_vertex(v) && mesh.vertex_valence(v) == 0 {
                mesh.remove_isolated_vertices();
                break;
            }
        }
        return Err(e);
    }
    for &t in &appended {
        patch.add_tri(t);
    }
    // Per-vertex sizes: loop vertices keep their local edge length; new vertices
    // interpolate it (Liepa) or, with an explicit target, grade towards the target.
    for k in 0..n {
        let s = 0.5 * (pos[k].distance(pos[(k + 1) % n]) + pos[k].distance(pos[(k + n - 1) % n]));
        patch.set_size(poly[k], s);
    }
    for &v in &patch.verts.clone() {
        patch.set_size(v, target);
    }
    let grading = if params.target_edge_length.is_some() { 0.5 } else { 0.0 };

    let fine = PcgParams { tolerance: 1e-7, max_iterations: 4000 };
    let membrane_3d = Cascade { order: 1, weights: LaplacianWeights::Uniform, along_normals: false };
    let membrane = Cascade { order: 1, weights: LaplacianWeights::Cotangent, along_normals: true };
    let rp = |relax: Relax, cascade: Option<Cascade>, max_levels: usize| RefineParams { target, grading, relax, cascade, max_levels };
    match params.mode {
        FillMode::Minimal if !fan => {}
        FillMode::Minimal | FillMode::Flat if fan && !simple => {
            // Centroid fan over a loop without a simple projection: untangle with umbrella
            // relaxation and uniform membrane solves while refining, then settle on a
            // cotangent membrane (minimal-surface-like).
            refine_patch(mesh, &mut patch, &rp(Relax::Free, Some(membrane_3d), 40));
            let _ = fair_patch(mesh, &patch, &membrane, &fine);
        }
        FillMode::Minimal | FillMode::Flat => {
            if simple {
                refine_patch(mesh, &mut patch, &rp(Relax::Plane { origin: frame.origin, normal: frame.normal }, None, 40));
                // Membrane heights over the plane: exactly planar for planar loops.
                let _ = fair_vertices_along(mesh, &patch.verts, 1, LaplacianWeights::Cotangent, FairDirection::Fixed(frame.normal), &fine);
            } else {
                refine_patch(mesh, &mut patch, &rp(Relax::Tangential, None, 40));
                let _ = fair_patch(mesh, &patch, &membrane, &fine);
            }
        }
        FillMode::Smooth => {
            if fan {
                // Untangle the cone with umbrella relaxation and uniform membrane solves while
                // refining, so the thin-plate solve starts from a sensible surface.
                refine_patch(mesh, &mut patch, &rp(Relax::Free, Some(membrane_3d), 40));
            }
            // Full 3D thin-plate solves (cotangent weights of the current surface); the
            // size field keeps the patch density matched to the loop, so tangential drift is
            // negligible, and unlike normal-constrained solves this can also reshape a
            // membrane-like starting surface (fan fallback).
            let plate = Cascade { order: 2, weights: params.smooth_weights, along_normals: false };
            refine_patch(mesh, &mut patch, &rp(Relax::Tangential, Some(plate), 40));
            let cap = crate::refine::solve_iteration_cap(patch.verts.len());
            let _ = fair_patch(mesh, &patch, &plate, &PcgParams { tolerance: 1e-7, max_iterations: cap });
            crate::refine::flip_pass(mesh, &patch, 2);
        }
    }
    out.new_triangles.extend(patch.sorted_tris());
    out.new_vertices.extend(patch.verts.iter().copied());
    Ok(())
}

fn triangle_exists(mesh: &Mesh, a: u32, b: u32, c: u32) -> bool {
    mesh.find_edge(a, b).is_some_and(|e| {
        mesh.edge_triangles(e).iter().any(|&t| t != mm_core::INVALID && mesh.triangle(t).contains(&c))
    })
}

/// Liepa's dihedral-angle measure: chord distance between unit normals (monotone in the
/// angle, ≈ angle for small angles; 90° for degenerate triangles).
#[inline]
fn dihedral(a: DVec3, b: DVec3) -> f64 {
    (2.0 - 2.0 * a.dot(b)).max(0.0).sqrt()
}

/// Angle cost of a zero-area triangle (above any real dihedral angle, whose cost is ≤ 2).
const DEGENERATE_COST: f64 = 3.0;
/// Angle values closer than this are considered equal, and area decides.
const ANGLE_TOL: f64 = 1e-4;
/// Weight of the squared chord length in the secondary cost (breaks area ties in favour of
/// short diagonals, e.g. for planar loops).
const CHORD_WEIGHT: f64 = 0.01;

#[inline]
fn better(a: f64, ar: f64, best_a: f64, best_ar: f64) -> bool {
    if a < best_a - ANGLE_TOL {
        true
    } else if a > best_a + ANGLE_TOL {
        false
    } else {
        ar < best_ar
    }
}

/// Minimum-weight triangulation of polygon `poly` (fill orientation) by dynamic
/// programming over sub-polygons (Liepa 2003, after Barequet & Sharir). The weight of a
/// triangulation is (maximum dihedral angle between adjacent triangles, including the mesh
/// triangles across loop edges; total area), compared lexicographically.
fn dp_triangulate(mesh: &Mesh, poly: &[u32], pos: &[DVec3], forbidden: &dyn Fn(usize, usize) -> bool) -> Option<Vec<[usize; 3]>> {
    let n = poly.len();
    // Normals of the mesh triangles across each loop edge (k, k+1), as seen by the fill.
    let mut bn = vec![DVec3::ZERO; n];
    for k in 0..n {
        let (a, b) = (poly[k], poly[(k + 1) % n]);
        if let Some(e) = mesh.find_edge(a, b) {
            let t = mesh.edge_triangles(e)[0];
            let nrm = mesh.tri_normal(t);
            // A consistently oriented neighbour traverses b→a; if it has a→b, flip it.
            bn[k] = if mesh.oriented_edge_vertices(e)[0] == a { -nrm } else { nrm };
        }
    }
    let idx = |i: usize, j: usize| i * n + j;
    let mut ang = vec![f64::INFINITY; n * n];
    let mut area = vec![f64::INFINITY; n * n];
    let mut split = vec![u32::MAX; n * n];
    let mut tn = vec![DVec3::ZERO; n * n];
    for i in 0..n - 1 {
        ang[idx(i, i + 1)] = 0.0;
        area[idx(i, i + 1)] = 0.0;
        tn[idx(i, i + 1)] = bn[i];
    }
    let mut allowed = vec![true; n * n];
    for i in 0..n {
        for j in i + 2..n {
            if !(i == 0 && j == n - 1) && forbidden(i, j) {
                allowed[idx(i, j)] = false;
            }
        }
    }
    for len in 2..n {
        for i in 0..n - len {
            let j = i + len;
            let top = i == 0 && j == n - 1;
            if !allowed[idx(i, j)] {
                continue;
            }
            let (pi, pj) = (pos[i], pos[j]);
            let chord = if top { 0.0 } else { CHORD_WEIGHT * pi.distance_squared(pj) };
            let (mut best_a, mut best_ar, mut best_m, mut best_n) = (f64::INFINITY, f64::INFINITY, u32::MAX, DVec3::ZERO);
            for m in i + 1..j {
                let a1 = ang[idx(i, m)];
                let a2 = ang[idx(m, j)];
                if !a1.is_finite() || !a2.is_finite() {
                    continue;
                }
                let cr = (pos[m] - pi).cross(pj - pi);
                let lc = cr.length();
                let nt = if lc > 0.0 { cr / lc } else { DVec3::ZERO };
                let mut a = a1.max(a2).max(dihedral(nt, tn[idx(i, m)])).max(dihedral(nt, tn[idx(m, j)]));
                // Zero-area triangles (e.g. through coincident vertices) only as a last resort.
                let l2 = pi.distance_squared(pos[m]).max(pi.distance_squared(pj)).max(pos[m].distance_squared(pj));
                if lc <= 2e-6 * l2 {
                    a = a.max(DEGENERATE_COST);
                }
                if top {
                    a = a.max(dihedral(nt, bn[n - 1]));
                }
                let ar = area[idx(i, m)] + area[idx(m, j)] + 0.5 * lc + chord;
                if better(a, ar, best_a, best_ar) {
                    best_a = a;
                    best_ar = ar;
                    best_m = m as u32;
                    best_n = nt;
                }
            }
            if best_m != u32::MAX {
                ang[idx(i, j)] = best_a;
                area[idx(i, j)] = best_ar;
                split[idx(i, j)] = best_m;
                tn[idx(i, j)] = best_n;
            }
        }
    }
    if split[idx(0, n - 1)] == u32::MAX {
        return None;
    }
    let mut out = Vec::with_capacity(n - 2);
    let mut stack = vec![(0usize, n - 1)];
    while let Some((i, j)) = stack.pop() {
        if j < i + 2 {
            continue;
        }
        let m = split[idx(i, j)] as usize;
        out.push([i, m, j]);
        stack.push((i, m));
        stack.push((m, j));
    }
    Some(out)
}

/// Ear clipping of a simple counter-clockwise polygon, choosing the best-shaped ear first.
/// Diagonals for which `forbidden` returns true are never created. Returns `None` if the
/// polygon cannot be triangulated under these constraints.
pub(crate) fn ear_clip(q: &[DVec2], forbidden: &dyn Fn(usize, usize) -> bool) -> Option<Vec<[usize; 3]>> {
    let n = q.len();
    if n < 3 || signed_area2(q) <= 0.0 {
        return None;
    }
    let diag = {
        let mut lo = DVec2::splat(f64::INFINITY);
        let mut hi = DVec2::splat(f64::NEG_INFINITY);
        for p in q {
            lo = lo.min(*p);
            hi = hi.max(*p);
        }
        (hi - lo).length()
    };
    let eps = 1e-14 * diag * diag;
    let mut prev: Vec<usize> = (0..n).map(|i| (i + n - 1) % n).collect();
    let mut next: Vec<usize> = (0..n).map(|i| (i + 1) % n).collect();
    let mut alive = vec![true; n];
    let convex = |i: usize, prev: &[usize], next: &[usize]| orient2(q[prev[i]], q[i], q[next[i]]) > eps;
    let mut is_convex: Vec<bool> = (0..n).map(|i| convex(i, &prev, &next)).collect();
    let quality = |a: DVec2, b: DVec2, c: DVec2| {
        let ar = orient2(a, b, c);
        let s = a.distance_squared(b) + b.distance_squared(c) + c.distance_squared(a);
        if s > 0.0 {
            ar / s
        } else {
            0.0
        }
    };
    let ear_score = |i: usize, prev: &[usize], next: &[usize], is_convex: &[bool], alive: &[bool]| -> f64 {
        if !is_convex[i] {
            return f64::NEG_INFINITY;
        }
        let (a, c) = (prev[i], next[i]);
        if forbidden(a, c) {
            return f64::NEG_INFINITY;
        }
        let (pa, pb, pc) = (q[a], q[i], q[c]);
        let mut k = next[c];
        while k != a {
            if alive[k] && !is_convex[k] {
                let p = q[k];
                if orient2(pa, pb, p) >= -eps && orient2(pb, pc, p) >= -eps && orient2(pc, pa, p) >= -eps {
                    return f64::NEG_INFINITY;
                }
            }
            k = next[k];
        }
        quality(pa, pb, pc)
    };
    let mut score: Vec<f64> = (0..n).map(|i| ear_score(i, &prev, &next, &is_convex, &alive)).collect();
    let mut out = Vec::with_capacity(n - 2);
    let mut remaining = n;
    let mut start = 0usize;
    while remaining > 3 {
        let mut best = usize::MAX;
        let mut best_s = f64::NEG_INFINITY;
        let mut k = start;
        for _ in 0..remaining {
            if score[k] > best_s {
                best_s = score[k];
                best = k;
            }
            k = next[k];
        }
        if best == usize::MAX || !best_s.is_finite() {
            return None;
        }
        let (a, c) = (prev[best], next[best]);
        out.push([a, best, c]);
        alive[best] = false;
        next[a] = c;
        prev[c] = a;
        remaining -= 1;
        start = a;
        for v in [a, c] {
            is_convex[v] = convex(v, &prev, &next);
        }
        for v in [a, c] {
            score[v] = ear_score(v, &prev, &next, &is_convex, &alive);
        }
    }
    let a = start;
    let b = next[a];
    let c = next[b];
    out.push([a, b, c]);
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_cycle_split() {
        let c = split_simple_cycles(&[1, 2, 3, 9, 4, 5, 9, 6]);
        assert_eq!(c.len(), 2);
        assert_eq!(c[0], vec![9, 4, 5]);
        assert_eq!(c[1], vec![1, 2, 3, 9, 6]);
        let c = split_simple_cycles(&[1, 2, 3, 4]);
        assert_eq!(c, vec![vec![1, 2, 3, 4]]);
    }

    #[test]
    fn ear_clip_concave() {
        // An L-shaped polygon.
        let q = [
            DVec2::new(0.0, 0.0),
            DVec2::new(2.0, 0.0),
            DVec2::new(2.0, 1.0),
            DVec2::new(1.0, 1.0),
            DVec2::new(1.0, 2.0),
            DVec2::new(0.0, 2.0),
        ];
        let t = ear_clip(&q, &|_, _| false).unwrap();
        assert_eq!(t.len(), 4);
        let area: f64 = t.iter().map(|&[a, b, c]| 0.5 * orient2(q[a], q[b], q[c])).sum();
        assert!((area - 3.0).abs() < 1e-12);
        assert!(t.iter().all(|&[a, b, c]| orient2(q[a], q[b], q[c]) > 0.0));
    }
}
