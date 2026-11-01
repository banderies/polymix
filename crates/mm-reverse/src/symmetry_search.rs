// SPDX-License-Identifier: AGPL-3.0-only
//! The mirror plane a scan matches itself across — the search a new Mirror opens on. Ported
//! from ScanRuler (`src/core/symmetry.ts`, with the parts of `src/core/deviation/icp.ts`
//! (`sampleScan`, the point-to-plane ICP), `rigid.ts` and `surface.ts` (the scan prepared for
//! closest-point queries, pseudonormals and the facing-aware search) it runs on), © Stefan
//! Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! Reverse engineering models one half of a symmetric part and mirrors it, so the plane it is
//! mirrored about has to be found from the scan to a precision no hand-picked plane reaches.
//! The search is mirror registration: reflect a thinned sample of the scan through a candidate
//! plane, fit the reflection back onto the scan with point-to-plane ICP, and read the plane off
//! the result. Every sample x and its registered mirror image y are a pair the true plane
//! bisects — the plane's normal is the direction the pairs run in, and its point is where their
//! midpoints lie — so one pass turns a rough candidate into a good one, and a few passes settle
//! it.
//!
//! A scan is never complete: where it has a hole, the mirror image of the surface opposite
//! lands on nothing, and a fixture or a stamped number on one side has no counterpart at all.
//! Those pairs are left out the way ICP leaves them out — anything further from the scan than
//! three times the median distance is not a mirror image — so they neither steer the plane nor
//! inflate the residual; they show up as samples that did not match.
//!
//! Without a seed the candidates are the scan's three principal planes: for a mirror-symmetric
//! cloud the mirror normal is always one of the principal axes. The caller may add directions
//! of its own. The search runs in two stages: each candidate is first settled on a small sample
//! with a loose registration and judged there; only the winner, and a runner-up that is close,
//! go on to the full sample and the tight registration the reported plane needs.
//!
//! What comes back is a plane and how symmetric the part is about it: the RMS distance of the
//! matched mirrored samples from the scan, so an asymmetric part reports a large one instead of
//! a plane that looks authoritative (`symmetry_plane::symmetry_accepted` judges it).

use std::collections::HashMap;

use glam::DVec3;
use mm_core::bvh::Bvh;
use mm_core::geom::Aabb;
use mm_core::Mesh;

use crate::fit2d::{solve_linear, Mulberry32};
use crate::principal_frame::principal_frame;
use crate::symmetry_plane::SymmetryPlane;

// ---- the scan prepared for closest-point queries (surface.ts) ----

/// Where on a triangle the closest point landed (Ericson, Real-Time Collision Detection
/// §5.1.5): which normal is the one to read there depends on it.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Feature {
    Vertex(usize),
    Edge(usize, usize),
    Face,
}

/// The closest point on a triangle to `p`, and the feature of the triangle it belongs to — the
/// branch that fires is the classification, so the seam is exact.
fn closest_on_triangle(p: DVec3, a: DVec3, b: DVec3, c: DVec3) -> (DVec3, Feature) {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return (a, Feature::Vertex(0));
    }
    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return (b, Feature::Vertex(1));
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (a + ab * v, Feature::Edge(0, 1));
    }
    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return (c, Feature::Vertex(2));
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (a + ac * w, Feature::Edge(0, 2));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
        let w = (d4 - d3) / (d4 - d3 + (d5 - d6));
        return (b + (c - b) * w, Feature::Edge(1, 2));
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    (a + ab * v + ac * w, Feature::Face)
}

/// A closest point on the scan: how far, where, and the pseudonormal there.
#[derive(Debug, Clone, Copy)]
pub struct ClosestHit {
    pub distance: f64,
    pub point: DVec3,
    pub normal: DVec3,
}

/// A triangle mesh prepared for closest-point queries (ScanRuler's `NominalSurface`): the
/// distance from a BVH, the normal read off the pseudonormal of whichever vertex, edge or face
/// the closest point lands on, and a second query for the nearest surface that faces a given
/// way.
pub struct ScanSurface {
    positions: Vec<DVec3>,
    triangles: Vec<[u32; 3]>,
    /// The mesh the BVH is built over; each triangle's group is its index in `triangles`.
    mesh: Mesh,
    bvh: Bvh,
    face_normal: Vec<DVec3>,
    vertex_normal: Vec<DVec3>,
    edge_normal: HashMap<(u32, u32), DVec3>,
    /// Per BVH node, the box of its triangles' normals: a node none of whose triangles can
    /// face the way asked is not opened by the facing-aware search. It answers as a search
    /// of every triangle would; it only skips those that cannot win.
    node_normals: Vec<(DVec3, DVec3)>,
    pub bbox_diagonal: f64,
}

fn normalized_or_z(v: DVec3) -> DVec3 {
    let len = v.length();
    if len > 1e-20 {
        v / len
    } else {
        DVec3::Z
    }
}

fn edge_key(a: u32, b: u32) -> (u32, u32) {
    (a.min(b), a.max(b))
}

impl ScanSurface {
    /// The surface of a mesh; `None` without a triangle.
    pub fn new(positions: &[DVec3], triangles: &[[u32; 3]]) -> Option<ScanSurface> {
        let tris: Vec<[u32; 3]> = triangles.iter().copied().filter(|t| t.iter().all(|v| (*v as usize) < positions.len())).collect();
        if tris.is_empty() {
            return None;
        }
        let groups: Vec<i32> = (0..tris.len() as i32).collect();
        let (mesh, _) = Mesh::from_triangles(positions, &tris, Some(&groups));
        let bvh = Bvh::build(&mesh);
        let mut lo = DVec3::splat(f64::INFINITY);
        let mut hi = DVec3::splat(f64::NEG_INFINITY);
        for t in &tris {
            for v in t {
                let p = positions[*v as usize];
                lo = lo.min(p);
                hi = hi.max(p);
            }
        }
        let mut face_normal = Vec::with_capacity(tris.len());
        let mut vertex_normal = vec![DVec3::ZERO; positions.len()];
        let mut edge_normal: HashMap<(u32, u32), DVec3> = HashMap::with_capacity(tris.len() * 3 / 2);
        for t in &tris {
            let [a, b, c] = t.map(|v| positions[v as usize]);
            let n = normalized_or_z((b - a).cross(c - a));
            face_normal.push(n);
            // Vertex pseudonormal: weighted by the angle the face subtends there.
            for (k, (p, q, r)) in [(a, b, c), (b, a, c), (c, a, b)].into_iter().enumerate() {
                let (e1, e2) = (q - p, r - p);
                let (l1, l2) = (e1.length(), e2.length());
                if !(l1 > 1e-20 && l2 > 1e-20) {
                    continue;
                }
                let w = (e1.dot(e2) / (l1 * l2)).clamp(-1.0, 1.0).acos();
                vertex_normal[t[k] as usize] += n * w;
            }
            for (x, y) in [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])] {
                *edge_normal.entry(edge_key(x, y)).or_insert(DVec3::ZERO) += n;
            }
        }
        for n in vertex_normal.iter_mut() {
            *n = normalized_or_z(*n);
        }
        for n in edge_normal.values_mut() {
            *n = normalized_or_z(*n);
        }
        let node_normals = node_normal_boxes(&bvh, &mesh, &face_normal);
        Some(ScanSurface { positions: positions.to_vec(), triangles: tris, mesh, bvh, face_normal, vertex_normal, edge_normal, node_normals, bbox_diagonal: (hi - lo).length() })
    }

    /// The closest point on the surface to `p` within `max_distance`, or `None`.
    pub fn closest(&self, p: DVec3, max_distance: f64) -> Option<ClosestHit> {
        let hit = self.bvh.nearest(&self.mesh, p, max_distance)?;
        let f = self.mesh.tri_group(hit.tid);
        (f >= 0).then(|| self.fill_hit(p, f as usize))
    }

    /// The nearest surface within `max_distance` that faces roughly the way `n` does — whose
    /// face normal has `normal · n >= min_dot` — searched best-first, a box further away than
    /// the best triangle so far not opened.
    pub fn closest_facing(&self, p: DVec3, n: DVec3, min_dot: f64, max_distance: f64) -> Option<ClosestHit> {
        let nodes = &self.bvh.nodes;
        if nodes.is_empty() {
            return None;
        }
        let mut best_sq = if max_distance.is_finite() { max_distance * max_distance } else { f64::INFINITY };
        let mut best: Option<usize> = None;
        let mut stack: Vec<u32> = vec![0];
        let dist = |b: &Aabb| b.distance_squared(p);
        // The most any normal within a box of normals can lean towards `n`.
        let can_face = |(lo, hi): &(DVec3, DVec3)| (n.x * lo.x).max(n.x * hi.x) + (n.y * lo.y).max(n.y * hi.y) + (n.z * lo.z).max(n.z * hi.z) >= min_dot;
        while let Some(ni) = stack.pop() {
            let node = &nodes[ni as usize];
            if dist(&node.bounds) >= best_sq || !self.node_normals.get(ni as usize).is_none_or(can_face) {
                continue;
            }
            if node.is_leaf() {
                for &tid in self.bvh.leaf_tris(node) {
                    let f = self.mesh.tri_group(tid);
                    if f < 0 {
                        continue;
                    }
                    let f = f as usize;
                    if self.face_normal[f].dot(n) < min_dot {
                        continue;
                    }
                    let [a, b, c] = self.triangles[f].map(|v| self.positions[v as usize]);
                    let (q, _) = closest_on_triangle(p, a, b, c);
                    let d2 = q.distance_squared(p);
                    if d2 < best_sq {
                        best_sq = d2;
                        best = Some(f);
                    }
                }
            } else {
                let (l, r) = (node.start, node.start + 1);
                let (dl, dr) = (dist(&nodes[l as usize].bounds), dist(&nodes[r as usize].bounds));
                if dl < dr {
                    stack.push(r);
                    stack.push(l);
                } else {
                    stack.push(l);
                    stack.push(r);
                }
            }
        }
        best.map(|f| self.fill_hit(p, f))
    }

    /// The closest point on one known triangle, and the pseudonormal of whichever feature of
    /// it that point belongs to.
    fn fill_hit(&self, p: DVec3, f: usize) -> ClosestHit {
        let t = self.triangles[f];
        let [a, b, c] = t.map(|v| self.positions[v as usize]);
        let (q, feature) = closest_on_triangle(p, a, b, c);
        let normal = match feature {
            Feature::Vertex(k) => self.vertex_normal[t[k] as usize],
            Feature::Edge(i, j) => self.edge_normal.get(&edge_key(t[i], t[j])).copied().unwrap_or(self.face_normal[f]),
            Feature::Face => self.face_normal[f],
        };
        ClosestHit { distance: (p - q).length(), point: q, normal }
    }
}

/// The box of the face normals under each node of a BVH, children before their parents.
fn node_normal_boxes(bvh: &Bvh, mesh: &Mesh, face_normal: &[DVec3]) -> Vec<(DVec3, DVec3)> {
    let empty = (DVec3::splat(f64::INFINITY), DVec3::splat(f64::NEG_INFINITY));
    let nodes = &bvh.nodes;
    let mut out = vec![empty; nodes.len()];
    if nodes.is_empty() {
        return out;
    }
    // Post-order: a node is closed once both its children are.
    let mut stack: Vec<(u32, bool)> = vec![(0, false)];
    while let Some((ni, children_done)) = stack.pop() {
        let Some(node) = nodes.get(ni as usize) else { continue };
        if node.is_leaf() {
            let mut b = empty;
            for &tid in bvh.leaf_tris(node) {
                let f = mesh.tri_group(tid);
                if let Some(n) = (f >= 0).then(|| face_normal.get(f as usize)).flatten() {
                    b = (b.0.min(*n), b.1.max(*n));
                }
            }
            out[ni as usize] = b;
        } else if children_done {
            let (l, r) = (out.get(node.start as usize).copied().unwrap_or(empty), out.get(node.start as usize + 1).copied().unwrap_or(empty));
            out[ni as usize] = (l.0.min(r.0), l.1.max(r.1));
        } else {
            stack.push((ni, true));
            stack.push((node.start, false));
            stack.push((node.start + 1, false));
        }
    }
    out
}

/// The area-weighted normals of a mesh's vertices (ScanRuler's `computeVertexNormals`), the way
/// the scan's triangles wind.
pub fn vertex_normals(positions: &[DVec3], triangles: &[[u32; 3]]) -> Vec<DVec3> {
    let mut out = vec![DVec3::ZERO; positions.len()];
    for t in triangles {
        if t.iter().any(|v| *v as usize >= positions.len()) {
            continue;
        }
        let [a, b, c] = t.map(|v| positions[v as usize]);
        let n = (b - a).cross(c - a);
        for v in t {
            out[*v as usize] += n;
        }
    }
    for n in out.iter_mut() {
        let len = n.length();
        *n = if len > 0.0 { *n / len } else { DVec3::Z };
    }
    out
}

// ---- a rigid motion (rigid.ts) ----

/// A rigid body motion p ↦ R·p + t, R a proper rotation, row-major.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rigid {
    pub r: [f64; 9],
    pub t: DVec3,
}

impl Rigid {
    pub const IDENTITY: Rigid = Rigid { r: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], t: DVec3::ZERO };

    /// Rotation from a unit quaternion (w, x, y, z).
    fn from_quaternion(mut w: f64, mut x: f64, mut y: f64, mut z: f64) -> Rigid {
        let n = (w * w + x * x + y * y + z * z).sqrt();
        if !(n > 1e-20) {
            return Rigid::IDENTITY;
        }
        w /= n;
        x /= n;
        y /= n;
        z /= n;
        Rigid {
            r: [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - w * z),
                2.0 * (x * z + w * y),
                2.0 * (x * y + w * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - w * x),
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                1.0 - 2.0 * (x * x + y * y),
            ],
            t: DVec3::ZERO,
        }
    }

    /// The rotation whose axis-angle vector is `w` (‖w‖ the angle), plus `t`.
    fn from_twist(w: DVec3, t: DVec3) -> Rigid {
        let angle = w.length();
        let mut m = if angle > 1e-20 {
            let half = angle / 2.0;
            let s = half.sin() / angle;
            Rigid::from_quaternion(half.cos(), w.x * s, w.y * s, w.z * s)
        } else {
            Rigid::IDENTITY
        };
        m.t = t;
        m
    }

    pub fn apply(&self, p: DVec3) -> DVec3 {
        self.rotate(p) + self.t
    }

    /// A direction turned (the translation left out) — normals move this way.
    pub fn rotate(&self, p: DVec3) -> DVec3 {
        let r = &self.r;
        DVec3::new(r[0] * p.x + r[1] * p.y + r[2] * p.z, r[3] * p.x + r[4] * p.y + r[5] * p.z, r[6] * p.x + r[7] * p.y + r[8] * p.z)
    }

    /// `self ∘ b`: b first, then self.
    fn compose(&self, b: &Rigid) -> Rigid {
        let mut r = [0.0; 9];
        for i in 0..3 {
            for j in 0..3 {
                r[i * 3 + j] = self.r[i * 3] * b.r[j] + self.r[i * 3 + 1] * b.r[3 + j] + self.r[i * 3 + 2] * b.r[6 + j];
            }
        }
        Rigid { r, t: self.apply(b.t) }
    }

    /// The nearest proper rotation, by Newton iteration towards orthogonality: the product of
    /// many increments drifts, and an R no longer orthogonal quietly scales the part.
    fn reorthonormalized(&self) -> Rigid {
        let mut r = self.r;
        for _ in 0..12 {
            let det = r[0] * (r[4] * r[8] - r[5] * r[7]) - r[1] * (r[3] * r[8] - r[5] * r[6]) + r[2] * (r[3] * r[7] - r[4] * r[6]);
            if !(det.abs() > 1e-12) {
                break;
            }
            let inv = [
                (r[4] * r[8] - r[5] * r[7]) / det,
                (r[5] * r[6] - r[3] * r[8]) / det,
                (r[3] * r[7] - r[4] * r[6]) / det,
                (r[2] * r[7] - r[1] * r[8]) / det,
                (r[0] * r[8] - r[2] * r[6]) / det,
                (r[1] * r[6] - r[0] * r[7]) / det,
                (r[1] * r[5] - r[2] * r[4]) / det,
                (r[2] * r[3] - r[0] * r[5]) / det,
                (r[0] * r[4] - r[1] * r[3]) / det,
            ];
            let mut delta = 0.0;
            for i in 0..9 {
                let next = 0.5 * (r[i] + inv[i]);
                delta += (next - r[i]).abs();
                r[i] = next;
            }
            if delta < 1e-15 {
                break;
            }
        }
        Rigid { r, t: self.t }
    }
}

// ---- a thinned sample and the ICP (icp.ts) ----

/// A thinned copy of the scan, with its normals where the scan has them.
#[derive(Debug, Clone, Default)]
pub struct ScanSamples {
    pub xyz: Vec<DVec3>,
    pub normals: Option<Vec<DVec3>>,
}

impl ScanSamples {
    pub fn count(&self) -> usize {
        self.xyz.len()
    }
}

/// An evenly thinned sample of a scan's vertices: a deterministic stride with a random phase,
/// so repeat runs on the same scan give the same answer (ScanRuler's `sampleScan`).
pub fn sample_scan(positions: &[DVec3], normals: Option<&[DVec3]>, wanted: usize, seed: u32) -> ScanSamples {
    let total = positions.len();
    let count = wanted.min(total);
    if count == 0 {
        return ScanSamples::default();
    }
    let stride = total as f64 / count as f64;
    let phase = Mulberry32::new(seed).next_f64();
    let mut xyz = Vec::with_capacity(count);
    let mut nrm = normals.map(|_| Vec::with_capacity(count));
    for i in 0..count {
        let v = (total - 1).min(((i as f64 + phase) * stride).floor() as usize);
        xyz.push(positions[v]);
        if let (Some(out), Some(n)) = (nrm.as_mut(), normals) {
            out.push(n.get(v).copied().unwrap_or(DVec3::Z));
        }
    }
    ScanSamples { xyz, normals: nrm }
}

/// How an ICP runs.
#[derive(Debug, Clone, Copy)]
pub struct IcpOptions {
    pub max_iterations: usize,
    /// A pair further from the surface than this multiple of the iteration's median is dropped.
    pub reject_median_factor: f64,
    /// A pair whose two normals disagree by more than `acos(min_normal_dot)` is dropped.
    pub min_normal_dot: f64,
    /// Pair a sample with the nearest surface that faces its way rather than the nearest
    /// surface full stop (needs a finite `max_pair_distance` and the samples' normals).
    pub facing_search: bool,
    pub max_pair_distance: f64,
    /// Stop once a step moves the part less than this, mm.
    pub tolerance: f64,
}

/// What an ICP came to: the best-scoring pose and how well it fits.
#[derive(Debug, Clone, Copy)]
pub struct IcpResult {
    pub transform: Rigid,
    pub iterations: usize,
    pub rms: f64,
    pub mean_distance: f64,
    pub score: f64,
    pub matched: usize,
    pub sampled: usize,
    pub converged: bool,
}

/// Point-to-plane ICP fitting `samples` onto `surface` from `initial` (ScanRuler's `icp`). The
/// increment's rotation is taken about the correspondences' centroid, poses are compared on a
/// capped mean over every sample, and the best-scoring pose is the one returned.
pub fn icp(surface: &ScanSurface, samples: &ScanSamples, initial: Rigid, o: &IcpOptions) -> IcpResult {
    let diag = surface.bbox_diagonal;
    let facing = o.facing_search && samples.normals.is_some() && o.min_normal_dot > 0.0 && o.max_pair_distance.is_finite();
    let score_cap = diag * 0.05;
    let n = samples.count();
    let mut transform = initial;
    let mut best = initial;
    let (mut best_score, mut best_rms, mut best_mean, mut best_matched) = (f64::INFINITY, f64::INFINITY, f64::INFINITY, 0usize);
    let mut iterations = 0;
    let mut converged = false;
    let mut q = vec![DVec3::ZERO; n];
    let mut cp = vec![DVec3::ZERO; n];
    let mut cn = vec![DVec3::ZERO; n];
    let mut dist = vec![0.0f64; n];
    let mut ok = vec![false; n];
    let mut sorted: Vec<f64> = Vec::with_capacity(n);
    for iter in 0..o.max_iterations {
        iterations = iter + 1;
        sorted.clear();
        let mut score_sum = 0.0;
        // Each sample's closest surface, asked in parallel; the sums below run in order, as
        // one loop's would.
        let paired: Vec<(DVec3, Option<ClosestHit>)> = {
            use rayon::prelude::*;
            (0..n)
                .into_par_iter()
                .map(|i| {
                    let p = transform.apply(samples.xyz[i]);
                    // The sample's own normal, carried into the surface's frame before the
                    // query: a facing-aware search needs it too.
                    let sn = samples.normals.as_ref().map(|ns| transform.rotate(ns[i]));
                    // Nearest surface first, whatever it faces; only a sample whose nearest
                    // surface faces back at it pays for the facing-aware search.
                    let mut hit = surface.closest(p, o.max_pair_distance);
                    if let (Some(h), Some(sn)) = (hit, sn) {
                        if facing && sn.dot(h.normal) < o.min_normal_dot {
                            hit = surface.closest_facing(p, sn, o.min_normal_dot, o.max_pair_distance);
                        }
                    }
                    // Out of reach, or a pair whose two normals still disagree, counts against
                    // the pose as a miss does.
                    let hit = hit.filter(|h| h.distance <= o.max_pair_distance).filter(|h| sn.is_none_or(|sn| sn.dot(h.normal) >= o.min_normal_dot));
                    (p, hit)
                })
                .collect()
        };
        for (i, (p, hit)) in paired.into_iter().enumerate() {
            ok[i] = false;
            q[i] = p;
            let Some(h) = hit else {
                score_sum += score_cap;
                continue;
            };
            cp[i] = h.point;
            cn[i] = h.normal;
            dist[i] = h.distance;
            score_sum += if h.distance < score_cap { h.distance } else { score_cap };
            sorted.push(h.distance);
            ok[i] = true;
        }
        let score = score_sum / n.max(1) as f64;
        let found = sorted.len();
        if found < 3 {
            break;
        }
        sorted.sort_by(|a, b| a.total_cmp(b));
        let median = sorted[found >> 1];
        let cutoff = (median * o.reject_median_factor).max(diag * 1e-5);
        let mut c = DVec3::ZERO;
        let mut matched = 0usize;
        let mut sum_abs = 0.0;
        for i in 0..n {
            if !ok[i] {
                continue;
            }
            if dist[i] > cutoff {
                ok[i] = false;
                continue;
            }
            c += q[i];
            sum_abs += dist[i];
            matched += 1;
        }
        if matched < 6 {
            break;
        }
        c /= matched as f64;
        let mut ata = [0.0f64; 36];
        let mut atb = [0.0f64; 6];
        let mut sum_sq = 0.0;
        for i in 0..n {
            if !ok[i] {
                continue;
            }
            let qc = q[i] - c;
            let nn = cn[i];
            // The signed distance across the surface — the quantity driven to zero.
            let r = (q[i] - cp[i]).dot(nn);
            sum_sq += r * r;
            let w = qc.cross(nn);
            let row = [w.x, w.y, w.z, nn.x, nn.y, nn.z];
            for a in 0..6 {
                atb[a] -= row[a] * r;
                for b in a..6 {
                    ata[a * 6 + b] += row[a] * row[b];
                }
            }
        }
        for a in 0..6 {
            for b in 0..a {
                ata[a * 6 + b] = ata[b * 6 + a];
            }
        }
        let mean_distance = sum_abs / matched as f64;
        let rms = (sum_sq / matched as f64).sqrt();
        if score < best_score {
            best_score = score;
            best_rms = rms;
            best_mean = mean_distance;
            best_matched = matched;
            best = transform;
        }
        // A touch of Levenberg damping on the diagonal.
        let trace: f64 = (0..6).map(|a| ata[a * 6 + a]).sum();
        let lambda = (trace / 6.0) * 1e-9 + 1e-12;
        for a in 0..6 {
            ata[a * 6 + a] += lambda;
        }
        let Some(step) = solve_linear(6, &mut ata, &atb) else { break };
        let mut omega = DVec3::new(step[0], step[1], step[2]);
        let mut tau = DVec3::new(step[3], step[4], step[5]);
        // A wild step clamped rather than taken.
        let (angle, shift) = (omega.length(), tau.length());
        let (max_angle, max_shift) = (0.35, diag * 0.5);
        let scale = 1f64.min(if angle > max_angle { max_angle / angle } else { 1.0 }).min(if shift > max_shift { max_shift / shift } else { 1.0 });
        if scale < 1.0 {
            omega *= scale;
            tau *= scale;
        }
        // The increment is about the centroid: lifted back to the origin before composing.
        let local = Rigid::from_twist(omega, tau);
        let rc = local.rotate(c);
        let delta = Rigid { r: local.r, t: c + local.t - rc };
        transform = delta.compose(&transform).reorthonormalized();
        let moved = omega.length() * diag * 0.5 + tau.length();
        if moved < o.tolerance {
            converged = true;
            break;
        }
    }
    IcpResult {
        transform: best,
        iterations,
        rms: if best_rms.is_finite() { best_rms } else { 0.0 },
        mean_distance: if best_mean.is_finite() { best_mean } else { 0.0 },
        score: best_score,
        matched: best_matched,
        sampled: n,
        converged,
    }
}

// ---- the search (symmetry.ts) ----

/// A plane to start the search from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SeedPlane {
    pub normal: DVec3,
    pub point: DVec3,
}

/// How the search runs.
#[derive(Debug, Clone, Default)]
pub struct SymmetryOptions {
    /// Samples of the scan the search runs on (4000).
    pub samples: Option<usize>,
    /// Confine the samples to these vertices — a surface marked by hand.
    pub vertices: Option<Vec<u32>>,
    /// Reflect-register-bisect passes on the full sample (5).
    pub rounds: Option<usize>,
    /// Normals of further candidate planes, tried through the centroid beside the principal
    /// ones when no seed is given; one within 3° of a candidate there is dropped.
    pub directions: Vec<DVec3>,
}

/// Beyond this multiple of the median distance a mirrored sample has found no mirror image.
const REJECT_MEDIAN_FACTOR: f64 = 3.0;

/// The samples reflected through the plane, normals included.
fn reflect(s: &ScanSamples, n: DVec3, p: DVec3) -> ScanSamples {
    let pn = p.dot(n);
    let xyz = s.xyz.iter().map(|x| *x - n * (2.0 * (x.dot(n) - pn))).collect();
    let normals = s.normals.as_ref().map(|ns| ns.iter().map(|v| *v - n * (2.0 * v.dot(n))).collect());
    ScanSamples { xyz, normals }
}

/// Every point's distance to the scan, infinite where none lies within `reach`.
fn distances(surface: &ScanSurface, xyz: &[DVec3], reach: f64) -> Vec<f64> {
    use rayon::prelude::*;
    xyz.par_iter().map(|p| surface.closest(*p, reach).map_or(f64::INFINITY, |h| h.distance)).collect()
}

/// How far a mirrored sample may lie from the scan and still count as having found its mirror
/// image: three medians over the samples that found any surface — zero when none did.
fn match_threshold(d: &[f64]) -> f64 {
    let mut finite: Vec<f64> = d.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return 0.0;
    }
    finite.sort_by(|a, b| a.total_cmp(b));
    let median = finite[finite.len() >> 1];
    (REJECT_MEDIAN_FACTOR * median).max(1e-6)
}

/// The plane bisecting every kept sample and its registered mirror image: the normal along the
/// pairs (each taken the way round that agrees with `current`), the point at their midpoints.
fn bisector(x: &[DVec3], y: &[DVec3], keep: impl Fn(usize) -> bool, current: DVec3) -> Option<SeedPlane> {
    let mut s = DVec3::ZERO;
    let mut m = DVec3::ZERO;
    let mut kept = 0;
    for i in 0..x.len() {
        if !keep(i) {
            continue;
        }
        let d = y[i] - x[i];
        let sign = if d.dot(current) < 0.0 { -1.0 } else { 1.0 };
        s += d * sign;
        m += (x[i] + y[i]) / 2.0;
        kept += 1;
    }
    if kept < 20 {
        return None;
    }
    let len = s.length();
    if !(len > 0.0) || !len.is_finite() {
        return None;
    }
    Some(SeedPlane { normal: s / len, point: m / kept as f64 })
}

/// How symmetric the part is about a plane, nothing else moved: `rms` over the matched mirrored
/// samples, `capped` over all of them with every unmatched sample charged the full reach.
fn residual(surface: &ScanSurface, samples: &ScanSamples, n: DVec3, p: DVec3, reach: f64) -> (f64, usize, f64) {
    let mirrored = reflect(samples, n, p);
    let d = distances(surface, &mirrored.xyz, reach);
    let thr = match_threshold(&d);
    let (mut sum_sq, mut matched) = (0.0, 0usize);
    for v in &d {
        if *v <= thr {
            sum_sq += v * v;
            matched += 1;
        }
    }
    let count = mirrored.count();
    let unmatched = count - matched;
    let rms = if matched > 0 { (sum_sq / matched as f64).sqrt() } else { f64::INFINITY };
    let capped = ((sum_sq + unmatched as f64 * reach * reach) / count.max(1) as f64).sqrt();
    (rms, matched, capped)
}

/// The plane the scan is mirror-symmetric about (ScanRuler's `findSymmetryPlane`). `surface`
/// is the scan prepared for closest-point queries, `positions` its vertices and `normals`
/// theirs; `seed` a plane to start from, or `None` for the principal planes. `progress` hears
/// what the search is doing.
pub fn find_symmetry_plane(surface: &ScanSurface, positions: &[DVec3], normals: Option<&[DVec3]>, seed: Option<SeedPlane>, opts: &SymmetryOptions, progress: &dyn Fn(&str)) -> Result<SymmetryPlane, String> {
    let rounds = opts.rounds.unwrap_or(5);
    let (pos, nrm): (Vec<DVec3>, Option<Vec<DVec3>>) = match opts.vertices.as_ref().filter(|v| !v.is_empty()) {
        Some(v) => (v.iter().filter_map(|i| positions.get(*i as usize).copied()).collect(), normals.map(|ns| v.iter().map(|i| ns.get(*i as usize).copied().unwrap_or(DVec3::Z)).collect())),
        None => (positions.to_vec(), normals.map(|n| n.to_vec())),
    };
    let samples = sample_scan(&pos, nrm.as_deref(), opts.samples.unwrap_or(4000), 0x5eed);
    if samples.count() < 100 {
        return Err("Too few points to look for a symmetry.".into());
    }
    let diag = surface.bbox_diagonal;
    let reach = diag * 0.1;
    // The small sample the candidates are settled and judged on first.
    let coarse = if samples.count() > 1200 { Some(sample_scan(&pos, nrm.as_deref(), 600.max((samples.count() as f64 / 6.0).round() as usize), 0x5eed)) } else { None };
    let candidates: Vec<SeedPlane> = match seed {
        Some(s) => {
            let len = s.normal.length();
            if !(len > 1e-20) {
                return Err("The seed plane has no direction.".into());
            }
            vec![SeedPlane { normal: s.normal / len, point: s.point }]
        }
        None => {
            let frame = principal_frame(&samples.xyz);
            let mut out: Vec<SeedPlane> = frame.axes.iter().map(|a| SeedPlane { normal: *a, point: frame.centroid }).collect();
            let cos_same = (3f64).to_radians().cos();
            for d in &opts.directions {
                let len = d.length();
                if !(len > 1e-20) {
                    continue;
                }
                let normal = *d / len;
                if out.iter().any(|c| c.normal.dot(normal).abs() > cos_same) {
                    continue;
                }
                out.push(SeedPlane { normal, point: frame.centroid });
            }
            out
        }
    };
    // Passes of reflect, register, bisect from a starting plane; `fine` is the full sample and
    // the tight registration, without it the quick look.
    let settle = |s: &ScanSamples, start: SeedPlane, passes: usize, fine: bool, say: &dyn Fn(usize) -> String| -> SeedPlane {
        let (mut n, mut p) = (start.normal, start.point);
        let mut search = reach;
        for pass in 0..passes {
            progress(&say(pass));
            let mirrored = reflect(s, n, p);
            let res = icp(surface, &mirrored, Rigid::IDENTITY, &IcpOptions { max_iterations: if fine { 40 } else { 15 }, reject_median_factor: REJECT_MEDIAN_FACTOR, min_normal_dot: 0.5, facing_search: true, max_pair_distance: search, tolerance: diag * if fine { 1e-7 } else { 1e-5 } });
            // A mirror image that finds almost no surface is not a symmetry.
            if (res.matched as f64) < 0.2 * s.count() as f64 {
                break;
            }
            let moved: Vec<DVec3> = mirrored.xyz.iter().map(|x| res.transform.apply(*x)).collect();
            let d = distances(surface, &moved, search);
            let thr = match_threshold(&d);
            let Some(next) = bisector(&s.xyz, &moved, |i| d[i] <= thr, n) else { break };
            let turned = next.normal.dot(n).abs().min(1.0).acos();
            let shifted = (next.point - p).dot(next.normal).abs();
            n = next.normal;
            p = next.point;
            // What matched lay within `thr`; the next pass searches no further than a few
            // times that.
            search = reach.min((diag * 0.01).max(4.0 * thr));
            let settled = if fine { turned < 1e-7 && shifted < 1e-6 } else { turned < 1e-4 && shifted < diag * 1e-5 };
            if settled {
                break;
            }
        }
        SeedPlane { normal: n, point: p }
    };
    // Stage one: every candidate on the small sample, judged where it settles.
    const COARSE_PASSES: usize = 3;
    let total = candidates.len();
    let mut field: Vec<(SeedPlane, usize, f64)> = candidates.iter().enumerate().map(|(i, c)| (*c, i, 0.0)).collect();
    if let Some(coarse) = &coarse {
        for c in field.iter_mut() {
            let ci = c.1;
            c.0 = settle(coarse, c.0, COARSE_PASSES, false, &|pass| if seed.is_some() { format!("Finding the symmetry plane — a first look, pass {}…", pass + 1) } else { format!("Finding the symmetry plane — trying plane {} of {total}…", ci + 1) });
            c.2 = residual(surface, coarse, c.0.normal, c.0.point, reach).2;
        }
        // The winner goes on, and a runner-up too close to call on so few points.
        field.sort_by(|a, b| a.2.total_cmp(&b.2));
        let first = field.first().map_or(0.0, |f| f.2);
        field = field.into_iter().enumerate().filter(|(i, c)| *i == 0 || (*i == 1 && c.2 <= 1.25 * first)).map(|(_, c)| c).collect();
    }
    // Stage two: the full sample and the tight registration.
    let mut best: Option<(SymmetryPlane, f64)> = None;
    for (plane, ci, _) in field {
        let s = settle(&samples, plane, rounds, true, &|pass| format!("Finding the symmetry plane — refining, pass {}…", pass + 1));
        let (rms, matched, capped) = residual(surface, &samples, s.normal, s.point, reach);
        if best.as_ref().is_none_or(|b| capped < b.1) {
            best = Some((SymmetryPlane { normal: s.normal, point: s.point, rms, matched, sampled: samples.count(), candidate: if seed.is_some() { -1 } else { ci as i32 } }, capped));
        }
    }
    best.map(|b| b.0).ok_or_else(|| "No symmetry plane could be found.".into())
}

/// The search on a scan as the workspace holds it: its vertices and triangles, in the part's
/// frame — the surface prepared, the normals read off the triangles, the principal planes
/// tried, and beside them the coordinate planes.
///
/// ScanRuler's mesh worker hands the search the part's own face directions as further
/// candidates (`autoAlign(graph).axes`, the frame its flat faces are square to), which a lug or
/// a missing patch does not turn the way it turns the principal axes — on a part whose scatter
/// is nearly the same two ways the principal planes are any two square to each other, and the
/// search may settle on a diagonal symmetry of one block of it. The automatic alignment is not
/// ported; the coordinate axes stand in for its directions — on a part aligned first they are
/// the same — and a direction within 3° of a principal one is dropped, as the search drops any.
pub fn scan_symmetry(positions: &[DVec3], triangles: &[[u32; 3]], progress: &dyn Fn(&str)) -> Result<SymmetryPlane, String> {
    progress("Preparing the scan for the symmetry search…");
    let surface = ScanSurface::new(positions, triangles).ok_or("There is no scan to look for a symmetry in.")?;
    let normals = vertex_normals(positions, triangles);
    // The vertices the triangles use: a scan's dead slots and loose points are no surface.
    let mut used = vec![false; positions.len()];
    for t in triangles {
        for v in t {
            if let Some(u) = used.get_mut(*v as usize) {
                *u = true;
            }
        }
    }
    let vertices: Vec<u32> = (0..positions.len() as u32).filter(|v| used[*v as usize]).collect();
    find_symmetry_plane(&surface, positions, Some(&normals), None, &SymmetryOptions { vertices: Some(vertices), directions: vec![DVec3::X, DVec3::Y, DVec3::Z], ..Default::default() }, progress)
}

/// A coordinate plane as the seed of a search: the plane itself where it cuts through the part,
/// and its parallel through the part's centre where it does not (ScanRuler's `baseSeedPlane`).
pub fn base_seed_plane(normal: DVec3, center: DVec3, size: f64) -> SeedPlane {
    let off = center.dot(normal);
    let cuts = off.abs() <= size.max(0.0) / 2.0;
    SeedPlane { normal, point: if cuts { DVec3::ZERO } else { center } }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symmetry_plane::symmetry_accepted;

    /// A unit Gaussian from a uniform generator (Box–Muller), as the tests' helpers draw one.
    fn gaussian(rand: &mut Mulberry32) -> f64 {
        let mut u = 0.0;
        while u == 0.0 {
            u = rand.next_f64();
        }
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * rand.next_f64()).cos()
    }

    /// A box of `size` centred at the origin, each face a `grid` × `grid` grid, noise along each
    /// face's normal off its borders (the tests' `boxMesh`), as a soup.
    fn box_soup(size: f64, grid: usize, noise: f64, seed: u32) -> Vec<DVec3> {
        let mut rand = Mulberry32::new(seed);
        let h = size / 2.0;
        let faces: [([f64; 3], [f64; 3], [f64; 3]); 6] = [
            ([-h, -h, h], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            ([-h, -h, -h], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]),
            ([h, -h, -h], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
            ([-h, -h, -h], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
            ([-h, h, -h], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
            ([-h, -h, -h], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
        ];
        let mut tris = Vec::new();
        for (o, eu, ev) in faces {
            let (o, eu, ev) = (DVec3::from(o), DVec3::from(eu), DVec3::from(ev));
            let nrm = eu.cross(ev).normalize();
            let mut verts = vec![vec![DVec3::ZERO; grid + 1]; grid + 1];
            for i in 0..=grid {
                for j in 0..=grid {
                    let a = i as f64 / grid as f64 * size;
                    let b = j as f64 / grid as f64 * size;
                    let border = i == 0 || j == 0 || i == grid || j == grid;
                    let d = if noise > 0.0 && !border { gaussian(&mut rand) * noise } else { 0.0 };
                    verts[i][j] = o + eu * a + ev * b + nrm * d;
                }
            }
            for i in 0..grid {
                for j in 0..grid {
                    let (a, b, c, d) = (verts[i][j], verts[i + 1][j], verts[i + 1][j + 1], verts[i][j + 1]);
                    tris.extend([a, b, c, a, c, d]);
                }
            }
        }
        tris
    }

    /// A soup welded into vertices and triangles.
    fn weld(soup: &[DVec3]) -> (Vec<DVec3>, Vec<[u32; 3]>) {
        let mut map: HashMap<(i64, i64, i64), u32> = HashMap::new();
        let mut pos = Vec::new();
        let idx: Vec<u32> = soup
            .iter()
            .map(|p| {
                let key = ((p.x * 1e6).round() as i64, (p.y * 1e6).round() as i64, (p.z * 1e6).round() as i64);
                *map.entry(key).or_insert_with(|| {
                    pos.push(*p);
                    (pos.len() - 1) as u32
                })
            })
            .collect();
        (pos, idx.chunks(3).map(|c| [c[0], c[1], c[2]]).collect())
    }

    /// An L-shaped part: a 20 mm cube with a 10 mm cube grown off one side, off-centre in y —
    /// so the only mirror plane is z = `dz`. Lightly noisy, as a scan is.
    fn l_shape(dz: f64, bump_z: f64) -> (Vec<DVec3>, Vec<[u32; 3]>) {
        let mut soup: Vec<DVec3> = box_soup(20.0, 12, 0.01, 1).into_iter().map(|p| p + DVec3::new(0.0, 0.0, dz)).collect();
        soup.extend(box_soup(10.0, 8, 0.01, 2).into_iter().map(|p| p + DVec3::new(15.0, 5.0, dz + bump_z)));
        weld(&soup)
    }

    fn opts() -> SymmetryOptions {
        SymmetryOptions { samples: Some(1500), ..Default::default() }
    }

    fn run(g: &(Vec<DVec3>, Vec<[u32; 3]>), seed: Option<SeedPlane>, o: &SymmetryOptions) -> SymmetryPlane {
        let surface = ScanSurface::new(&g.0, &g.1).unwrap_or_else(|| panic!("a surface"));
        let normals = vertex_normals(&g.0, &g.1);
        find_symmetry_plane(&surface, &g.0, Some(&normals), seed, o, &|_| {}).unwrap_or_else(|e| panic!("{e}"))
    }

    #[test]
    fn finds_the_one_mirror_plane_of_an_l_shaped_part_from_its_principal_planes() {
        let r = run(&l_shape(0.0, 0.0), None, &opts());
        assert!(r.normal.z.abs() > 0.9999, "{r:?}");
        assert!(r.point.dot(r.normal).abs() < 0.02, "{r:?}");
        assert!(r.rms < 0.03, "{r:?}");
        assert!(r.matched as f64 > 0.6 * r.sampled as f64, "{r:?}");
        assert!(r.candidate >= 0);
        assert!(symmetry_accepted(&r));
    }

    #[test]
    fn settles_from_a_seed_that_is_tilted_and_off_the_plane_and_reports_it_as_such() {
        let t = 5f64.to_radians();
        let r = run(&l_shape(0.0, 0.0), Some(SeedPlane { normal: DVec3::new(0.0, t.sin(), t.cos()), point: DVec3::new(0.0, 0.0, 1.5) }), &opts());
        assert!(r.normal.z.abs() > 0.9999, "{r:?}");
        assert!(r.point.z.abs() < 0.02, "{r:?}");
        assert!(r.rms < 0.03, "{r:?}");
        assert_eq!(r.candidate, -1);
    }

    #[test]
    fn finds_the_plane_where_the_part_is_not_at_the_origin() {
        let r = run(&l_shape(7.0, 0.0), None, &opts());
        assert!(r.normal.z.abs() > 0.9999, "{r:?}");
        assert!((r.point.dot(r.normal).abs() - 7.0).abs() < 0.02, "{r:?}");
    }

    #[test]
    fn takes_the_callers_directions_as_further_candidates_and_drops_the_ones_it_already_has() {
        let r = run(&l_shape(0.0, 0.0), None, &SymmetryOptions { directions: vec![DVec3::Z], ..opts() });
        assert!(r.normal.z.abs() > 0.9999, "{r:?}");
        assert!(r.candidate < 3);
        assert!(r.rms < 0.03, "{r:?}");
    }

    #[test]
    fn says_so_when_the_part_is_not_symmetric() {
        // The small cube lifted 3 mm out of the plane: nothing mirrors cleanly.
        let r = run(&l_shape(0.0, 3.0), None, &opts());
        assert!(r.rms > 0.2, "{r:?}");
        assert!(!symmetry_accepted(&r));
    }

    #[test]
    fn finds_the_symmetry_of_the_marked_surface_alone_when_the_rest_is_left_out() {
        let g = l_shape(0.0, 3.0);
        let marked: Vec<u32> = (0..g.0.len() as u32).filter(|v| {
            let p = g.0[*v as usize];
            p.x.abs() <= 10.001 && p.y.abs() <= 10.001 && p.z.abs() <= 10.001
        }).collect();
        let inside: std::collections::HashSet<u32> = marked.iter().copied().collect();
        let cube: Vec<[u32; 3]> = g.1.iter().copied().filter(|t| t.iter().all(|v| inside.contains(v))).collect();
        let surface = ScanSurface::new(&g.0, &cube).unwrap_or_else(|| panic!("a surface"));
        let normals = vertex_normals(&g.0, &g.1);
        let r = find_symmetry_plane(&surface, &g.0, Some(&normals), None, &SymmetryOptions { vertices: Some(marked.clone()), ..opts() }, &|_| {}).unwrap_or_else(|e| panic!("{e}"));
        assert!(r.rms < 0.03, "{r:?}");
        // A cube's principal axes are any three square to each other — its scatter is the same
        // every way — so which of its nine mirror planes the search settles on turns on the
        // last bit of the noise: ScanRuler's own run lands on an axis plane, this one may land
        // on a diagonal one. Either is the cube's own.
        let n = r.normal.abs();
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let axis = n.max_element() > 0.9999;
        let diagonal = [n.x, n.y, n.z].iter().filter(|c| (**c - s).abs() < 1e-3).count() == 2;
        assert!(axis || diagonal, "{r:?}");
        assert!(r.point.dot(r.normal).abs() < 0.02, "{r:?}");
        assert!(r.sampled <= marked.len());
    }

    #[test]
    fn a_coordinate_plane_as_the_seed_is_the_plane_where_it_cuts_the_part_and_its_parallel_through_the_centre_where_not() {
        // A part of 40 across, centred 5 off the XY plane: the plane runs through it.
        assert_eq!(base_seed_plane(DVec3::Z, DVec3::new(10.0, 20.0, 5.0), 40.0), SeedPlane { normal: DVec3::Z, point: DVec3::ZERO });
        // The same part 300 up, fresh from the scanner: the plane misses it.
        assert_eq!(base_seed_plane(DVec3::Z, DVec3::new(10.0, 20.0, 300.0), 40.0), SeedPlane { normal: DVec3::Z, point: DVec3::new(10.0, 20.0, 300.0) });
    }

    #[test]
    fn a_part_that_scatters_alike_two_ways_finds_its_mirror_plane_among_the_coordinate_planes() {
        // A 40 mm block with a 20 mm one off its side: its scatter across y and z is nearly
        // the same, so its principal planes there are any two square to each other — and the
        // big block's diagonal symmetry lies near them. The coordinate planes standing in for
        // the part's face directions bring the true one, z = 5, into the search.
        let mut soup: Vec<DVec3> = box_soup(40.0, 24, 0.01, 1).into_iter().map(|p| p + DVec3::new(0.0, 0.0, 5.0)).collect();
        soup.extend(box_soup(20.0, 12, 0.01, 2).into_iter().map(|p| p + DVec3::new(30.0, 10.0, 5.0)));
        let g = weld(&soup);
        let r = scan_symmetry(&g.0, &g.1, &|_| {}).unwrap_or_else(|e| panic!("{e}"));
        assert!(r.normal.z.abs() > 0.9999, "{r:?}");
        assert!((r.point.dot(r.normal).abs() - 5.0).abs() < 0.02, "{r:?}");
        assert!(symmetry_accepted(&r));
    }

    #[test]
    fn a_scan_as_the_workspace_holds_it_is_searched_whole() {
        let g = l_shape(0.0, 0.0);
        let r = scan_symmetry(&g.0, &g.1, &|_| {}).unwrap_or_else(|e| panic!("{e}"));
        assert!(r.normal.z.abs() > 0.9999, "{r:?}");
        assert!(symmetry_accepted(&r));
        assert!(scan_symmetry(&[], &[], &|_| {}).is_err());
    }
}
