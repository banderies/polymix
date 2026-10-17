//! Triangle/triangle intersection and mesh self-intersection detection.

use crate::bvh::Bvh;
use crate::geom::Aabb;
use crate::mesh::Mesh;
use glam::DVec3;
use rayon::prelude::*;

/// Möller's interval-overlap triangle/triangle intersection test ("A Fast Triangle-Triangle
/// Intersection Test", 1997). Coplanar triangles are tested with 2D edge/containment
/// checks. Touching at a single point counts as intersecting.
pub fn triangles_intersect(t1: [DVec3; 3], t2: [DVec3; 3]) -> bool {
    let scale = t1.iter().chain(t2.iter()).fold(0.0f64, |m, p| m.max(p.abs().max_element())).max(1.0);
    let eps = 1e-12 * scale;
    let n2 = (t2[1] - t2[0]).cross(t2[2] - t2[0]);
    let d2 = -n2.dot(t2[0]);
    let du = t1.map(|p| snap(n2.dot(p) + d2, eps * n2.length()));
    if du.iter().all(|&d| d > 0.0) || du.iter().all(|&d| d < 0.0) {
        return false;
    }
    let n1 = (t1[1] - t1[0]).cross(t1[2] - t1[0]);
    let d1 = -n1.dot(t1[0]);
    let dv = t2.map(|p| snap(n1.dot(p) + d1, eps * n1.length()));
    if dv.iter().all(|&d| d > 0.0) || dv.iter().all(|&d| d < 0.0) {
        return false;
    }
    if du.iter().all(|&d| d == 0.0) {
        return coplanar_intersect(n1, t1, t2);
    }
    let dir = n1.cross(n2);
    // Project onto the largest axis of the intersection line direction.
    let axis = dir.abs().max_position();
    let p1 = t1.map(|p| p[axis]);
    let p2 = t2.map(|p| p[axis]);
    let (Some(a), Some(b)) = (interval(p1, du), interval(p2, dv)) else {
        return false;
    };
    a.0.max(b.0) <= a.1.min(b.1)
}

#[inline]
fn snap(d: f64, eps: f64) -> f64 {
    if d.abs() < eps {
        0.0
    } else {
        d
    }
}

/// Interval of the triangle on the intersection line, from projected vertices `p` and
/// signed distances `d` to the other triangle's plane.
fn interval(p: [f64; 3], d: [f64; 3]) -> Option<(f64, f64)> {
    let mut ts = Vec::with_capacity(3);
    for i in 0..3 {
        let j = (i + 1) % 3;
        if d[i] == 0.0 {
            ts.push(p[i]);
        }
        if (d[i] > 0.0 && d[j] < 0.0) || (d[i] < 0.0 && d[j] > 0.0) {
            ts.push(p[i] + (p[j] - p[i]) * d[i] / (d[i] - d[j]));
        }
    }
    if ts.is_empty() {
        return None;
    }
    let lo = ts.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = ts.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    Some((lo, hi))
}

fn coplanar_intersect(n: DVec3, t1: [DVec3; 3], t2: [DVec3; 3]) -> bool {
    // Project onto the plane most aligned with the normal.
    let drop = n.abs().max_position();
    let (i, j) = match drop {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    };
    let a: Vec<(f64, f64)> = t1.iter().map(|p| (p[i], p[j])).collect();
    let b: Vec<(f64, f64)> = t2.iter().map(|p| (p[i], p[j])).collect();
    let orient = |p: (f64, f64), q: (f64, f64), r: (f64, f64)| (q.0 - p.0) * (r.1 - p.1) - (q.1 - p.1) * (r.0 - p.0);
    let on_seg = |p: (f64, f64), q: (f64, f64), r: (f64, f64)| {
        r.0 >= p.0.min(q.0) && r.0 <= p.0.max(q.0) && r.1 >= p.1.min(q.1) && r.1 <= p.1.max(q.1)
    };
    let seg = |p1: (f64, f64), p2: (f64, f64), q1: (f64, f64), q2: (f64, f64)| {
        let d1 = orient(q1, q2, p1);
        let d2 = orient(q1, q2, p2);
        let d3 = orient(p1, p2, q1);
        let d4 = orient(p1, p2, q2);
        if ((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0)) && ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0)) {
            return true;
        }
        // Touching / collinear cases: an endpoint lies on the other segment.
        (d1 == 0.0 && on_seg(q1, q2, p1)) || (d2 == 0.0 && on_seg(q1, q2, p2)) || (d3 == 0.0 && on_seg(p1, p2, q1)) || (d4 == 0.0 && on_seg(p1, p2, q2))
    };
    for k in 0..3 {
        for l in 0..3 {
            if seg(a[k], a[(k + 1) % 3], b[l], b[(l + 1) % 3]) {
                return true;
            }
        }
    }
    let inside = |p: (f64, f64), t: &[(f64, f64)]| {
        let s0 = orient(t[0], t[1], p);
        let s1 = orient(t[1], t[2], p);
        let s2 = orient(t[2], t[0], p);
        (s0 >= 0.0 && s1 >= 0.0 && s2 >= 0.0) || (s0 <= 0.0 && s1 <= 0.0 && s2 <= 0.0)
    };
    inside(a[0], &b) || inside(b[0], &a)
}

/// Pairs of non-adjacent triangles that intersect each other. Triangles sharing a vertex
/// are not tested against each other (fold-overs at a shared vertex are not reported).
pub fn self_intersections(mesh: &Mesh, bvh: &Bvh) -> Vec<(u32, u32)> {
    let tris: Vec<u32> = mesh.triangle_ids().collect();
    let mut pairs: Vec<(u32, u32)> = tris
        .par_iter()
        .flat_map_iter(|&t| {
            let p = mesh.tri_positions(t);
            let tv = mesh.triangle(t);
            let b = Aabb::from_points(p);
            let mut out = Vec::new();
            bvh.query_aabb(&b, |o| {
                if o <= t {
                    return;
                }
                let ov = mesh.triangle(o);
                if ov.iter().any(|v| tv.contains(v)) {
                    return;
                }
                if triangles_intersect(p, mesh.tri_positions(o)) {
                    out.push((t, o));
                }
            });
            out
        })
        .collect();
    pairs.sort_unstable();
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;

    #[test]
    fn basic_cases() {
        let t = [DVec3::ZERO, DVec3::X, DVec3::Y];
        // Piercing.
        assert!(triangles_intersect(t, [DVec3::new(0.2, 0.2, -1.0), DVec3::new(0.2, 0.2, 1.0), DVec3::new(0.3, 0.6, 0.0)]));
        // Parallel, offset.
        assert!(!triangles_intersect(t, t.map(|p| p + DVec3::Z * 0.1)));
        // Separated in plane.
        assert!(!triangles_intersect(t, t.map(|p| p + DVec3::X * 2.0)));
        // Coplanar overlap.
        assert!(triangles_intersect(t, t.map(|p| p + DVec3::new(0.2, 0.2, 0.0))));
        // Plane crossing but outside the triangle.
        assert!(!triangles_intersect(t, [DVec3::new(2.0, 2.0, -1.0), DVec3::new(2.0, 2.0, 1.0), DVec3::new(3.0, 2.0, 0.0)]));
    }

    #[test]
    fn clean_and_overlapping_spheres() {
        let a = primitives::icosphere(1.0, 3);
        let bvh = Bvh::build(&a);
        assert!(self_intersections(&a, &bvh).is_empty());
        let mut two = a.clone();
        let mut b = primitives::icosphere(1.0, 3);
        b.transform(|p| p + DVec3::X * 1.2);
        two.append_mesh(&b, 1);
        let bvh = Bvh::build(&two);
        let hits = self_intersections(&two, &bvh);
        assert!(hits.len() > 20, "{}", hits.len());
        // Every reported pair is between the two spheres.
        for (t, o) in hits {
            assert_ne!(two.tri_group(t), two.tri_group(o));
        }
    }
}
