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

use crate::cut::{compute_curves, curve_summary, point_segment_distance, CutContext};
use mm_core::linalg::{add_outer, sym_eigen3};
use crate::util::frame;
use crate::{EditError, EditResult};
use mm_core::bvh::RayHit;
use mm_core::geom::{closest_point_on_triangle, ray_triangle};
use mm_core::{Bvh, DVec2, DVec3, Mesh, Plane, Ray};

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
    let diag = mesh.bounds().diagonal();

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
    let eps = 1e-9 * diag;
    // Stamp-based scratch so each iteration costs O(patch), not O(mesh).
    let mut scratch = Scratch {
        vmark: vec![0; mesh.max_vid()],
        tmark: vec![0; mesh.max_tid()],
        nacc: vec![DVec3::ZERO; mesh.max_vid()],
        aacc: vec![0.0; mesh.max_vid()],
        stamp: 0,
    };
    let mut sd_buf: Vec<f64> = Vec::new();

    for it in 0..16 {
        let mut rho = (2.0 * r).max(3.0 * h);
        let mut patch = gather(mesh, v0, c, rho, &mut scratch);
        for _ in 0..4 {
            if patch.len() >= 12 {
                break;
            }
            rho *= 1.5;
            patch = gather(mesh, v0, c, rho, &mut scratch);
        }
        let stamp = scratch.stamp;
        let mut tris: Vec<u32> = Vec::new();
        for &v in &patch {
            scratch.nacc[v as usize] = DVec3::ZERO;
            scratch.aacc[v as usize] = 0.0;
            for &e in mesh.vertex_edges(v) {
                for t in mesh.edge_triangles(e) {
                    if t != mm_core::INVALID && scratch.tmark[t as usize] != stamp {
                        scratch.tmark[t as usize] = stamp;
                        tris.push(t);
                    }
                }
            }
        }
        for &t in &tris {
            let x = mesh.tri_cross(t);
            let a = x.length() / 6.0;
            for v in mesh.triangle(t) {
                if scratch.vmark[v as usize] == stamp {
                    scratch.nacc[v as usize] += x;
                    scratch.aacc[v as usize] += a;
                }
            }
        }
        let nw: Vec<(DVec3, DVec3, f64)> = patch
            .iter()
            .map(|&v| (mesh.vertex(v), scratch.nacc[v as usize].normalize_or_zero(), scratch.aacc[v as usize]))
            .collect();
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
        let ctx = CutContext::for_triangles(mesh, plane, eps, &tris, std::mem::take(&mut sd_buf));
        let curves = compute_curves(mesh, &ctx, Some(&tris));
        sd_buf = ctx.sd;
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
    if axis == DVec3::ZERO {
        return Err(EditError::Failed("could not estimate a vessel axis".into()));
    }
    let nv = mesh.vertex_count().max(1) as f64;
    let centroid = mesh.vertex_ids().map(|v| mesh.vertex(v)).sum::<DVec3>() / nv;
    if (c - centroid).dot(axis) < 0.0 {
        axis = -axis;
    }
    Ok(VesselCrossSection { plane: Plane::from_point_normal(c, axis), center: c, axis, radius: r, area })
}

fn nearest_point(mesh: &Mesh, bvh: Option<&Bvh>, p: DVec3) -> (u32, DVec3) {
    if let Some(b) = bvh {
        if let Some(hit) = b.nearest(mesh, p, f64::INFINITY) {
            return (hit.tid, hit.point);
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

struct Scratch {
    vmark: Vec<u32>,
    tmark: Vec<u32>,
    nacc: Vec<DVec3>,
    aacc: Vec<f64>,
    stamp: u32,
}

/// Vertices connected to `v0` through vertices within `rho` of `c` (v0 always included).
/// Marks them in `scratch.vmark` with a fresh stamp.
fn gather(mesh: &Mesh, v0: u32, c: DVec3, rho: f64, scratch: &mut Scratch) -> Vec<u32> {
    scratch.stamp += 1;
    let stamp = scratch.stamp;
    let r2 = rho * rho;
    scratch.vmark[v0 as usize] = stamp;
    let mut out = vec![v0];
    let mut i = 0;
    while i < out.len() {
        let v = out[i];
        i += 1;
        for w in mesh.vertex_neighbors(v) {
            if scratch.vmark[w as usize] != stamp && mesh.vertex(w).distance_squared(c) <= r2 {
                scratch.vmark[w as usize] = stamp;
                out.push(w);
            }
        }
    }
    out
}
