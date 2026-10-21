//! Vessel-aware helpers: estimate the local axis and cross-section of a tubular structure
//! at a clicked surface point, to "click a vessel and cut it square".
//!
//! Algorithm ([`estimate_vessel_cross_section`]):
//! 1. Find the surface point and normal under the click; estimate the radius from a ray
//!    cast through the vessel to the opposite wall (or use the caller's estimate).
//! 2. Iterate: gather the surface patch connected to the click within ~2 radii of the
//!    current axis point; the axis is the eigenvector of least variance of the
//!    area-weighted normal covariance (tube normals are perpendicular to the axis); the
//!    axis point is the least-squares intersection of the normal lines in the plane
//!    perpendicular to the axis, kept at the click's axial position; it is then replaced by
//!    the area centroid of the actual closed cross-section curve when one is found.
//! 3. The returned plane passes through the cross-section centroid, perpendicular to the
//!    axis, with its normal oriented away from the mesh's vertex centroid (towards the end
//!    of the branch, which is usually the side to trim).

use crate::cut::{compute_curves, curve_summary, point_segment_distance, snap_eps_with, CutContext};
use mm_core::linalg::{add_outer, sym_eigen3};
use crate::util::frame;
use crate::{EditError, EditResult};
use mm_core::bvh::RayHit;
use mm_core::geom::{closest_point_on_triangle, ray_triangle};
use mm_core::{Aabb, Bvh, DVec2, DVec3, Mesh, Plane, Ray};
use std::sync::{Mutex, PoisonError};

/// Local cross-section of a vessel.
#[derive(Clone, Copy, Debug)]
pub struct VesselCrossSection {
    /// Plane through `center`, normal = `axis`.
    pub plane: Plane,
    /// Cross-section centroid (on the vessel axis).
    pub center: DVec3,
    /// Unit axis direction (pointing away from the mesh centroid).
    pub axis: DVec3,
    /// Mean radius (equivalent radius of the cross-section when one was found).
    pub radius: f64,
    /// Area of the closed cross-section curve through the click, if found.
    pub area: Option<f64>,
}

/// Plane perpendicular to the local vessel axis through the cross-section centroid at
/// `point_on_surface`. `bvh` (if current) accelerates the point/ray queries;
/// `approx_radius` skips the ray-cast radius estimate.
pub fn suggest_cut_plane(
    mesh: &Mesh,
    bvh: Option<&Bvh>,
    point_on_surface: DVec3,
    approx_radius: Option<f64>,
) -> EditResult<Plane> {
    estimate_vessel_cross_section(mesh, bvh, point_on_surface, approx_radius).map(|x| x.plane)
}

/// Full estimate behind [`suggest_cut_plane`].
pub fn estimate_vessel_cross_section(
    mesh: &Mesh,
    bvh: Option<&Bvh>,
    point_on_surface: DVec3,
    approx_radius: Option<f64>,
) -> EditResult<VesselCrossSection> {
    let mut scratch = Scratch::take();
    let r = estimate(mesh, bvh, point_on_surface, approx_radius, &mut scratch);
    scratch.keep();
    r
}

/// Marks over a mesh's vertices and triangles and a buffer of distances, kept from call to
/// call, so each iteration of an estimate costs O(patch) rather than arrays as big as the mesh
/// made and cleared for every click (the patch of a finely meshed vessel holds tens of
/// thousands of vertices, too many to look up in maps quickly). A mark counts only with the
/// current stamp, which only increases; when it would wrap, every mark is cleared.
#[derive(Default)]
struct Scratch {
    stamp: u32,
    /// Per vertex: the stamp it was last gathered with, and its place in that patch.
    vert: Vec<(u32, u32)>,
    /// Per triangle: the stamp it was last taken with.
    tri: Vec<u32>,
    /// Signed distances by vertex for the cross-sections' contexts: all zeros between uses.
    sd: Vec<f64>,
    /// The mean of the last mesh's vertices, for its shape version (which no other shape has).
    centroid: Option<(u64, DVec3)>,
}

/// The scratch kept between estimates. At most two are kept (for estimates on two threads at
/// once), whichever threads ran them, so the memory held does not grow with the number of
/// threads that ever ran one.
static KEPT: Mutex<Vec<Scratch>> = Mutex::new(Vec::new());

impl Scratch {
    /// A kept scratch, or a new one.
    fn take() -> Self {
        KEPT.lock().unwrap_or_else(PoisonError::into_inner).pop().unwrap_or_default()
    }

    /// Keep this scratch for a later estimate (unless two are kept already).
    fn keep(self) {
        let mut kept = KEPT.lock().unwrap_or_else(PoisonError::into_inner);
        if kept.len() < 2 {
            kept.push(self);
        }
    }

    /// Room for every vertex and triangle of `mesh`. Scratch left far bigger by a bigger mesh
    /// starts again at this one's size, so a big mesh's marks are not held for good after it
    /// is closed.
    fn fit(&mut self, mesh: &Mesh) {
        if self.vert.len() > 4 * mesh.max_vid() + 4096 || self.tri.len() > 4 * mesh.max_tid() + 4096 {
            *self = Scratch::default();
        }
        if self.vert.len() < mesh.max_vid() {
            self.vert.resize(mesh.max_vid(), (0, 0));
        }
        if self.tri.len() < mesh.max_tid() {
            self.tri.resize(mesh.max_tid(), 0);
        }
    }

    /// A stamp no mark has yet.
    fn next(&mut self) -> u32 {
        if self.stamp == u32::MAX {
            self.vert.fill((0, 0));
            self.tri.fill(0);
            self.stamp = 0;
        }
        self.stamp += 1;
        self.stamp
    }

    /// The mean of `mesh`'s vertices: summed once per shape of the mesh, in vertex order (the
    /// same sum every time), rather than over the whole mesh on every click.
    fn centroid(&mut self, mesh: &Mesh) -> DVec3 {
        match self.centroid {
            Some((version, c)) if version == mesh.shape_version() => c,
            _ => {
                let nv = mesh.vertex_count().max(1) as f64;
                let c = mesh.vertex_ids().map(|v| mesh.vertex(v)).sum::<DVec3>() / nv;
                self.centroid = Some((mesh.shape_version(), c));
                c
            }
        }
    }
}

fn estimate(mesh: &Mesh, bvh: Option<&Bvh>, point_on_surface: DVec3, approx_radius: Option<f64>, scratch: &mut Scratch) -> EditResult<VesselCrossSection> {
    if mesh.triangle_count() == 0 {
        return Err(EditError::InvalidParameter("mesh is empty".into()));
    }
    let bvh = bvh.filter(|b| b.is_current(mesh) && !b.is_empty());
    let (t0, q) = nearest_point(mesh, bvh, point_on_surface);
    let tv = mesh.triangle(t0);
    let v0 = *tv
        .iter()
        .min_by(|&&a, &&b| {
            mesh.vertex(a).distance_squared(q).partial_cmp(&mesh.vertex(b).distance_squared(q)).unwrap()
        })
        .unwrap();
    let mut n0 = mesh.tri_normal(t0);
    if n0 == DVec3::ZERO {
        n0 = mesh.vertex_normal(v0);
    }
    if n0 == DVec3::ZERO {
        return Err(EditError::Failed("degenerate surface at the clicked point".into()));
    }
    let edges = mesh.vertex_edges(v0);
    let h = (edges.iter().map(|&e| mesh.edge_length(e)).sum::<f64>() / edges.len().max(1) as f64).max(1e-12);
    // 1e-9 of the bounding-box diagonal (from the tree's box when it holds every triangle).
    let eps = snap_eps_with(mesh, bvh);

    let mut r = match approx_radius {
        Some(r) if r.is_finite() && r > 0.0 => r,
        _ => {
            let min_t = 0.05 * h;
            let inward = first_hit(mesh, bvh, Ray::new(q - n0 * (1e-4 * h), -n0), min_t);
            match inward {
                Some(t) => 0.5 * t,
                None => match first_hit(mesh, bvh, Ray::new(q + n0 * (1e-4 * h), n0), min_t) {
                    // Inward-facing normals: the interior is on the normal side.
                    Some(t) => {
                        n0 = -n0;
                        0.5 * t
                    }
                    None => 5.0 * h,
                },
            }
        }
    };
    let mut c = q - n0 * r;
    let mut axis = DVec3::ZERO;
    let mut area = None;

    // Each iteration costs O(patch), not O(mesh): the marks and the distance buffer are
    // kept from call to call (see [`Scratch`]).
    scratch.fit(mesh);
    let mut sd_buf = std::mem::take(&mut scratch.sd);
    for it in 0..16 {
        let mut rho = (2.0 * r).max(3.0 * h);
        let mut patch = gather(mesh, v0, c, rho, scratch);
        for _ in 0..4 {
            if patch.len() >= 12 {
                break;
            }
            rho *= 1.5;
            patch = gather(mesh, v0, c, rho, scratch);
        }
        let stamp = scratch.stamp;
        let mut tris: Vec<u32> = Vec::new();
        for &v in &patch {
            for &e in mesh.vertex_edges(v) {
                for t in mesh.edge_triangles(e) {
                    if t != mm_core::INVALID && scratch.tri[t as usize] != stamp {
                        scratch.tri[t as usize] = stamp;
                        tris.push(t);
                    }
                }
            }
        }
        // Area-weighted normal and area of each patch vertex (by its place in the patch).
        let mut nacc = vec![DVec3::ZERO; patch.len()];
        let mut aacc = vec![0.0; patch.len()];
        for &t in &tris {
            let x = mesh.tri_cross(t);
            let a = x.length() / 6.0;
            for v in mesh.triangle(t) {
                let (s, i) = scratch.vert[v as usize];
                if s == stamp {
                    nacc[i as usize] += x;
                    aacc[i as usize] += a;
                }
            }
        }
        let nw: Vec<(DVec3, DVec3, f64)> = patch.iter().enumerate().map(|(i, &v)| (mesh.vertex(v), nacc[i].normalize_or_zero(), aacc[i])).collect();
        let mut m = [[0.0; 3]; 3];
        for &(_, nv, w) in &nw {
            add_outer(&mut m, nv, w);
        }
        let (_, vecs) = sym_eigen3(m);
        let mut a = vecs[0];
        if a == DVec3::ZERO {
            break;
        }
        if axis != DVec3::ZERO && a.dot(axis) < 0.0 {
            a = -a;
        }
        // Least-squares intersection of the normal lines, in the plane perpendicular to a.
        let (e1, e2) = frame(a);
        let (mut s00, mut s01, mut s11) = (0.0, 0.0, 0.0);
        let mut bv = DVec2::ZERO;
        for &(p, nv, w) in &nw {
            let m2 = DVec2::new(nv.dot(e1), nv.dot(e2));
            let l = m2.length();
            if l < 0.3 {
                continue;
            }
            let m2 = m2 / l;
            let d = p - c;
            let q2 = DVec2::new(d.dot(e1), d.dot(e2));
            let (p00, p01, p11) = (1.0 - m2.x * m2.x, -m2.x * m2.y, 1.0 - m2.y * m2.y);
            s00 += w * p00;
            s01 += w * p01;
            s11 += w * p11;
            bv += DVec2::new(p00 * q2.x + p01 * q2.y, p01 * q2.x + p11 * q2.y) * w;
        }
        let det = s00 * s11 - s01 * s01;
        let tr = s00 + s11;
        let x = if det > 1e-6 * tr * tr && tr > 0.0 {
            DVec2::new((s11 * bv.x - s01 * bv.y) / det, (s00 * bv.y - s01 * bv.x) / det)
        } else {
            DVec2::ZERO
        };
        let mut cn = c + e1 * x.x + e2 * x.y;
        cn += a * (q - cn).dot(a);
        let (mut rs, mut ws) = (0.0, 0.0);
        for &(p, _, w) in &nw {
            let d = p - cn;
            rs += w * (d - a * d.dot(a)).length();
            ws += w;
        }
        let mut rn = if ws > 0.0 { rs / ws } else { r };

        // Refine with the actual cross-section through the click.
        let plane = Plane::from_point_normal(cn, a);
        let ctx = CutContext::for_triangles_in(mesh, plane, eps, &tris, sd_buf);
        let curves = compute_curves(mesh, &ctx, Some(&tris));
        sd_buf = ctx.into_buffer();
        let best = curves
            .iter()
            .filter(|cv| cv.closed)
            .map(|cv| {
                let d = cv.segments.iter().map(|s| point_segment_distance(q, s[0], s[1])).fold(f64::INFINITY, f64::min);
                (d, cv)
            })
            .min_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
        area = None;
        if let Some((d, cv)) = best {
            let s = curve_summary(cv, a);
            if d < rn.max(h) && s.area > 0.0 && s.centroid.distance(cn) < 1.5 * rn.max(h) {
                cn = s.centroid;
                rn = (s.area / std::f64::consts::PI).sqrt();
                area = Some(s.area);
            }
        }
        let converged = it > 0 && a.dot(axis) > 1.0 - 1e-9 && cn.distance(c) < 1e-5 * rn.max(h);
        c = cn;
        r = rn.max(1e-12);
        axis = a;
        if converged {
            break;
        }
    }
    scratch.sd = sd_buf;
    if axis == DVec3::ZERO {
        return Err(EditError::Failed("could not estimate a vessel axis".into()));
    }
    let centroid = scratch.centroid(mesh);
    if (c - centroid).dot(axis) < 0.0 {
        axis = -axis;
    }
    Ok(VesselCrossSection { plane: Plane::from_point_normal(c, axis), center: c, axis, radius: r, area })
}

/// The nearest triangle to `p` and the nearest point on it; of triangles equally near, the one
/// with the lowest id. With a tree the same as the walk over all triangles below: every
/// triangle as near as the tree's answer has its box within that distance of `p`, so it is
/// among the triangles found there, and they are compared the same way.
fn nearest_point(mesh: &Mesh, bvh: Option<&Bvh>, p: DVec3) -> (u32, DVec3) {
    if let Some(b) = bvh {
        if let Some(hit) = b.nearest(mesh, p, f64::INFINITY) {
            // (Padded well past the rounding of the distance and of the triangles' boxes.)
            let r = hit.dist_sq.sqrt() * (1.0 + 1e-9) + 1e-9 * (1.0 + p.abs().max_element());
            let mut best = (hit.tid, hit.point, hit.dist_sq);
            b.query_aabb(&Aabb { min: p - DVec3::splat(r), max: p + DVec3::splat(r) }, |t| {
                let [a, b, c] = mesh.tri_positions(t);
                let (x, _) = closest_point_on_triangle(p, a, b, c);
                let d = x.distance_squared(p);
                if d < best.2 || (d == best.2 && t < best.0) {
                    best = (t, x, d);
                }
            });
            return (best.0, best.1);
        }
    }
    let mut best = (mm_core::INVALID, p, f64::INFINITY);
    for t in mesh.triangle_ids() {
        let [a, b, c] = mesh.tri_positions(t);
        let (x, _) = closest_point_on_triangle(p, a, b, c);
        let d = x.distance_squared(p);
        if d < best.2 {
            best = (t, x, d);
        }
    }
    (best.0, best.1)
}

fn first_hit(mesh: &Mesh, bvh: Option<&Bvh>, ray: Ray, min_t: f64) -> Option<f64> {
    let mut best: Option<f64> = None;
    let mut take = |t: f64| {
        if t > min_t && best.is_none_or(|b| t < b) {
            best = Some(t);
        }
    };
    match bvh {
        Some(b) => {
            let mut hits: Vec<RayHit> = Vec::new();
            b.raycast_all(mesh, &ray, &mut hits);
            hits.iter().for_each(|h| take(h.t));
        }
        None => {
            for t in mesh.triangle_ids() {
                let [a, b, c] = mesh.tri_positions(t);
                if let Some((th, _, _)) = ray_triangle(&ray, a, b, c) {
                    take(th);
                }
            }
        }
    }
    best
}

/// Vertices connected to `v0` through vertices within `rho` of `c` (v0 always included), in
/// the order found. Marks them in `scratch.vert` with a fresh stamp and their place in the
/// list.
fn gather(mesh: &Mesh, v0: u32, c: DVec3, rho: f64, scratch: &mut Scratch) -> Vec<u32> {
    let stamp = scratch.next();
    let r2 = rho * rho;
    scratch.vert[v0 as usize] = (stamp, 0);
    let mut out = vec![v0];
    let mut i = 0;
    while i < out.len() {
        let v = out[i];
        i += 1;
        for w in mesh.vertex_neighbors(v) {
            if scratch.vert[w as usize].0 != stamp && mesh.vertex(w).distance_squared(c) <= r2 {
                scratch.vert[w as usize] = (stamp, out.len() as u32);
                out.push(w);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_core::primitives;

    #[test]
    fn the_tree_finds_the_triangle_the_walk_over_all_finds() {
        // A grid (many triangles equally near its vertices and edges) and a tube, probed at
        // vertices, edge midpoints, centroids and points off the surface.
        let mut grid = primitives::plane_grid(4.0, 4.0, 16, 16);
        grid.transform(|p| DVec3::new(p.x, p.y, 0.2 * (p.x * 0.7).sin()));
        let path: Vec<DVec3> = (0..=30).map(|i| DVec3::new(0.0, 0.0, 0.3 * i as f64)).collect();
        let tube = primitives::tube_along(&path, &vec![1.0; path.len()], 16);
        for m in [grid, tube] {
            let bvh = Bvh::build(&m);
            let mut probes: Vec<DVec3> = m.vertex_ids().map(|v| m.vertex(v)).collect();
            for t in m.triangle_ids() {
                let [a, b, c] = m.tri_positions(t);
                probes.extend([(a + b) * 0.5, m.tri_centroid(t), m.tri_centroid(t) + m.tri_normal(t) * 0.3, c * 1.05]);
            }
            let mut ties = 0;
            for p in probes {
                let walked = nearest_point(&m, None, p);
                let treed = nearest_point(&m, Some(&bvh), p);
                assert_eq!(treed.0, walked.0, "at {p:?}");
                assert_eq!(treed.1.to_array().map(f64::to_bits), walked.1.to_array().map(f64::to_bits));
                ties += usize::from(bvh.nearest(&m, p, f64::INFINITY).is_some_and(|h| h.tid != walked.0));
            }
            assert!(ties > 0, "the plain tree query picks other triangles at ties");
        }
    }

    fn same(a: &VesselCrossSection, b: &VesselCrossSection) -> bool {
        a.center.to_array().map(f64::to_bits) == b.center.to_array().map(f64::to_bits)
            && a.axis.to_array().map(f64::to_bits) == b.axis.to_array().map(f64::to_bits)
            && (a.radius.to_bits(), a.area.map(f64::to_bits)) == (b.radius.to_bits(), b.area.map(f64::to_bits))
    }

    #[test]
    fn a_cross_section_with_a_tree_is_the_one_without() {
        let path: Vec<DVec3> = (0..=60).map(|i| DVec3::new(0.3 * (0.1 * i as f64).sin(), 0.0, 0.5 * i as f64)).collect();
        let m = primitives::tube_along(&path, &vec![2.0; path.len()], 32);
        let bvh = Bvh::build(&m);
        for v in m.vertex_ids().step_by(97) {
            let a = estimate_vessel_cross_section(&m, None, m.vertex(v), None).unwrap();
            let b = estimate_vessel_cross_section(&m, Some(&bvh), m.vertex(v), None).unwrap();
            assert!(same(&a, &b));
        }
    }

    #[test]
    fn kept_marks_change_nothing_and_survive_their_stamp_wrapping() {
        // A kept scratch, after other meshes (bigger and smaller) and with its stamp about to
        // wrap, gives what a fresh one gives.
        let path: Vec<DVec3> = (0..=40).map(|i| DVec3::new(0.0, 0.2 * (0.2 * i as f64).cos(), 0.5 * i as f64)).collect();
        let m = primitives::tube_along(&path, &vec![1.5; path.len()], 24);
        let big = primitives::icosphere(3.0, 5);
        let p = m.vertex(300);
        let fresh = estimate(&m, None, p, None, &mut Scratch::default()).unwrap();
        let mut kept = Scratch::default();
        estimate(&big, None, big.vertex(7), None, &mut kept).unwrap();
        assert!(kept.vert.len() >= big.max_vid());
        assert!(same(&estimate(&m, None, p, None, &mut kept).unwrap(), &fresh));
        assert_eq!((kept.vert.len(), kept.tri.len()), (m.max_vid(), m.max_tid()), "the big mesh's marks are let go");
        assert!(kept.sd.iter().all(|&x| x.to_bits() == 0), "the distances are cleared after");
        kept.stamp = u32::MAX - 2;
        assert!(same(&estimate(&m, None, p, None, &mut kept).unwrap(), &fresh));
        assert!(kept.stamp < 1000, "wrapped");
        assert!(same(&estimate(&m, None, p, None, &mut kept).unwrap(), &fresh));
        assert!(same(&estimate_vessel_cross_section(&m, None, p, None).unwrap(), &fresh));
        // The kept centroid is the one of the mesh at hand: with a part far below added, the
        // axis turns the other way (away from the centroid), with a scratch kept or fresh.
        let mut low = m.clone();
        let mut part = primitives::icosphere(2.0, 2);
        part.transform(|q| q - DVec3::Z * 200.0);
        low.append_mesh(&part, 3);
        let fresh_low = estimate(&low, None, p, None, &mut Scratch::default()).unwrap();
        assert!(fresh_low.axis.dot(fresh.axis) < 0.0);
        for (mesh, want) in [(&low, &fresh_low), (&m, &fresh), (&low, &fresh_low)] {
            assert!(same(&estimate(mesh, None, p, None, &mut kept).unwrap(), want));
            assert_eq!(kept.centroid.map(|c| c.0), Some(mesh.shape_version()));
        }
        // Moved: a new shape, a new centroid.
        let mut moved = m.clone();
        moved.transform(|q| q + DVec3::X);
        let fresh_moved = estimate(&moved, None, p + DVec3::X, None, &mut Scratch::default()).unwrap();
        assert!(same(&estimate(&moved, None, p + DVec3::X, None, &mut kept).unwrap(), &fresh_moved));
    }
}
