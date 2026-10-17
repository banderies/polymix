//! Shared helpers: boundary-loop handling, polygon measures, rollback bookkeeping.

use crate::{EditError, EditResult};
use mm_core::mesh::tri_other_vertex;
use mm_core::topology::EdgeLoop;
use mm_core::{DVec2, DVec3, Mesh, Plane, INVALID};

/// True if triangle `tv` traverses the directed edge a→b.
#[inline]
pub(crate) fn has_directed_edge(tv: [u32; 3], a: u32, b: u32) -> bool {
    (tv[0] == a && tv[1] == b) || (tv[1] == a && tv[2] == b) || (tv[2] == a && tv[0] == b)
}

/// For a boundary edge between `a` and `b`: `Some(true)` if its triangle runs a→b,
/// `Some(false)` if it runs b→a, `None` if there is no such boundary edge.
pub(crate) fn boundary_edge_direction(mesh: &Mesh, a: u32, b: u32) -> Option<bool> {
    if !mesh.is_vertex(a) || !mesh.is_vertex(b) {
        return None;
    }
    let e = mesh.find_edge(a, b)?;
    if !mesh.is_boundary_edge(e) {
        return None;
    }
    let t = mesh.edge_triangles(e)[0];
    Some(has_directed_edge(mesh.triangle(t), a, b))
}

/// Validate a closed loop of boundary vertices (either orientation, optionally with the
/// first vertex repeated at the end) and return it in boundary orientation, i.e. edge
/// `i → i+1` runs the same way as in its adjacent triangle.
pub(crate) fn orient_boundary_loop(mesh: &Mesh, verts: &[u32]) -> EditResult<Vec<u32>> {
    let mut v: Vec<u32> = verts.to_vec();
    if v.len() >= 2 && v.first() == v.last() {
        v.pop();
    }
    if v.len() < 3 {
        return Err(EditError::InvalidLoop("a loop needs at least 3 vertices".into()));
    }
    let n = v.len();
    let (mut fwd, mut bwd) = (0usize, 0usize);
    for i in 0..n {
        match boundary_edge_direction(mesh, v[i], v[(i + 1) % n]) {
            Some(true) => fwd += 1,
            Some(false) => bwd += 1,
            None => {
                return Err(EditError::InvalidLoop(format!(
                    "vertices {} and {} are not joined by a boundary edge",
                    v[i],
                    v[(i + 1) % n]
                )))
            }
        }
    }
    if fwd > 0 && bwd > 0 {
        return Err(EditError::InvalidLoop("loop edges have inconsistent orientation".into()));
    }
    if bwd > 0 {
        v.reverse();
    }
    Ok(v)
}

/// Validate an open chain of boundary vertices (≥ 2) and return it in boundary orientation.
pub(crate) fn orient_boundary_span(mesh: &Mesh, verts: &[u32]) -> EditResult<Vec<u32>> {
    let mut v: Vec<u32> = verts.to_vec();
    if v.len() < 2 {
        return Err(EditError::InvalidLoop("a span needs at least 2 vertices".into()));
    }
    let (mut fwd, mut bwd) = (0usize, 0usize);
    for w in v.windows(2) {
        match boundary_edge_direction(mesh, w[0], w[1]) {
            Some(true) => fwd += 1,
            Some(false) => bwd += 1,
            None => {
                return Err(EditError::InvalidLoop(format!(
                    "vertices {} and {} are not joined by a boundary edge",
                    w[0], w[1]
                )))
            }
        }
    }
    if fwd > 0 && bwd > 0 {
        return Err(EditError::InvalidLoop("span edges have inconsistent orientation".into()));
    }
    if bwd > 0 {
        v.reverse();
    }
    Ok(v)
}

/// Rotate around `q` (inside the fan of `t_start`) to the next boundary edge leaving `q`.
fn next_boundary_edge(mesh: &Mesh, t_start: u32, q: u32) -> Option<(u32, u32)> {
    let mut t = t_start;
    for _ in 0..4096 {
        let tv = mesh.triangle(t);
        let j = tv.iter().position(|&v| v == q)?;
        let x = tv[(j + 1) % 3];
        let e = mesh.find_edge_from_tri(q, x, t)?;
        if mesh.is_boundary_edge(e) {
            return Some((e, x));
        }
        let n = mesh.edge_other_triangle(e, t);
        if n == INVALID || n == t_start {
            return None;
        }
        t = n;
    }
    None
}

/// Walk the boundary loop containing boundary edge `e0` (boundary orientation).
pub(crate) fn walk_boundary_loop(mesh: &Mesh, e0: u32) -> Option<EdgeLoop> {
    if !mesh.is_edge(e0) || !mesh.is_boundary_edge(e0) {
        return None;
    }
    let [p, q] = mesh.oriented_edge_vertices(e0);
    let mut lp = EdgeLoop { vertices: vec![p], edges: vec![e0] };
    let mut e = e0;
    let mut cur = q;
    for _ in 0..mesh.edge_count() + 1 {
        if cur == p {
            return Some(lp);
        }
        let t = mesh.edge_triangles(e)[0];
        let (ne, x) = next_boundary_edge(mesh, t, cur)?;
        if ne == e0 {
            return None;
        }
        lp.vertices.push(cur);
        lp.edges.push(ne);
        e = ne;
        cur = x;
    }
    None
}

pub(crate) fn positions(mesh: &Mesh, verts: &[u32]) -> Vec<DVec3> {
    verts.iter().map(|&v| mesh.vertex(v)).collect()
}

pub(crate) fn mean(pts: &[DVec3]) -> DVec3 {
    if pts.is_empty() {
        return DVec3::ZERO;
    }
    pts.iter().copied().sum::<DVec3>() / pts.len() as f64
}

/// Newell area vector of a closed polygon (|v| = area, direction by right-hand rule).
pub(crate) fn area_vector(pts: &[DVec3]) -> DVec3 {
    let c = mean(pts);
    let n = pts.len();
    let mut a = DVec3::ZERO;
    for i in 0..n {
        a += (pts[i] - c).cross(pts[(i + 1) % n] - c);
    }
    a * 0.5
}

/// Orthonormal (u, v) with u × v = n.
pub(crate) fn frame(n: DVec3) -> (DVec3, DVec3) {
    let n = n.normalize();
    let (u, _) = mm_core::geom::orthonormal_basis(n);
    let u = (u - n * u.dot(n)).normalize();
    (u, n.cross(u))
}

/// Signed area and area centroid of a 2D polygon (falls back to the vertex mean when the
/// area vanishes).
pub(crate) fn polygon_area_centroid_2d(pts: &[DVec2]) -> (f64, DVec2) {
    let n = pts.len();
    if n == 0 {
        return (0.0, DVec2::ZERO);
    }
    let o = pts[0];
    let mut a2 = 0.0;
    let mut c = DVec2::ZERO;
    for i in 0..n {
        let p = pts[i] - o;
        let q = pts[(i + 1) % n] - o;
        let cr = p.perp_dot(q);
        a2 += cr;
        c += (p + q) * cr;
    }
    let mean2 = pts.iter().copied().sum::<DVec2>() / n as f64;
    let scale: f64 = pts.iter().map(|p| (*p - mean2).length_squared()).sum::<f64>() / n as f64;
    if a2.abs() <= 1e-14 * scale.max(1e-300) {
        return (0.0, mean2);
    }
    (a2 * 0.5, o + c / (3.0 * a2))
}

/// Crossing-number point-in-polygon test.
pub(crate) fn point_in_polygon(poly: &[DVec2], p: DVec2) -> bool {
    let n = poly.len();
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (a, b) = (poly[i], poly[j]);
        if (a.y > p.y) != (b.y > p.y) {
            let x = a.x + (p.y - a.y) * (b.x - a.x) / (b.y - a.y);
            if p.x < x {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

#[inline]
pub(crate) fn smoothstep(x: f64) -> f64 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

/// A face-group id not used by any triangle.
pub(crate) fn new_group(mesh: &Mesh) -> i32 {
    if mesh.triangle_count() == 0 {
        return 1;
    }
    mesh.max_group_id().saturating_add(1)
}

pub(crate) fn normalize_plane(p: Plane) -> EditResult<Plane> {
    let l = p.normal.length();
    if !(l.is_finite() && l > 1e-300) || !p.d.is_finite() {
        return Err(EditError::InvalidParameter("plane normal must be non-zero and finite".into()));
    }
    Ok(Plane { normal: p.normal / l, d: p.d / l })
}

/// Live, de-duplicated triangle ids.
pub(crate) fn valid_unique_tris(mesh: &Mesh, tris: &[u32]) -> Vec<u32> {
    let mut seen = vec![false; mesh.max_tid()];
    let mut out = Vec::with_capacity(tris.len());
    for &t in tris {
        if mesh.is_triangle(t) && !seen[t as usize] {
            seen[t as usize] = true;
            out.push(t);
        }
    }
    out
}

/// Remove triangles added by a failed operation (vertices only they used are freed too).
pub(crate) fn rollback(mesh: &mut Mesh, tris: &[u32]) {
    mesh.remove_triangles(tris.iter().copied());
}

/// True if every edge of `tris` that has two triangles is traversed in opposite
/// directions by them.
pub(crate) fn locally_consistent(mesh: &Mesh, tris: &[u32]) -> bool {
    for &t in tris {
        if !mesh.is_triangle(t) {
            continue;
        }
        let tv = mesh.triangle(t);
        for (j, &e) in mesh.tri_edges(t).iter().enumerate() {
            let o = mesh.edge_other_triangle(e, t);
            if o == INVALID {
                continue;
            }
            let (a, b) = (tv[j], tv[(j + 1) % 3]);
            if !has_directed_edge(mesh.triangle(o), b, a) {
                return false;
            }
        }
    }
    true
}

/// Unit direction pointing away from the surface at a boundary loop given in boundary
/// orientation. Uses the loop's area vector (the cap of the loop faces `-area_vector` for an
/// outward-oriented surface) and flips it if the adjacent surface clearly lies in front.
pub(crate) fn loop_outward_direction(mesh: &Mesh, lp: &[u32]) -> Option<DVec3> {
    let pts = positions(mesh, lp);
    let av = area_vector(&pts);
    if av.length() <= 1e-300 {
        return None;
    }
    let d0 = -av.normalize();
    let c = mean(&pts);
    let n = lp.len();
    let (mut s, mut scale) = (0.0, 0.0);
    for i in 0..n {
        let (a, b) = (lp[i], lp[(i + 1) % n]);
        if let Some(e) = mesh.find_edge(a, b) {
            let t = mesh.edge_triangles(e)[0];
            let x = tri_other_vertex(mesh.triangle(t), a, b);
            if x != INVALID {
                let w = mesh.vertex(x) - c;
                s += w.dot(d0);
                scale += w.length();
            }
        }
    }
    if scale > 0.0 && s > 0.05 * scale {
        Some(-d0)
    } else {
        Some(d0)
    }
}

/// Resample a polyline (closed if `closed`) to `n` points evenly spaced by arc length,
/// starting at its first point.
pub(crate) fn resample_polyline(pts: &[DVec3], n: usize, closed: bool) -> Vec<DVec3> {
    let m = pts.len();
    if m == 0 || n == 0 {
        return Vec::new();
    }
    let segs = if closed { m } else { m - 1 };
    let mut cum = Vec::with_capacity(segs + 1);
    cum.push(0.0);
    for i in 0..segs {
        let l = pts[i].distance(pts[(i + 1) % m]);
        cum.push(cum[i] + l);
    }
    let total = *cum.last().unwrap();
    if total <= 0.0 || segs == 0 {
        return vec![pts[0]; n];
    }
    let mut out = Vec::with_capacity(n);
    let denom = if closed { n as f64 } else { (n.max(2) - 1) as f64 };
    let mut k = 0;
    for i in 0..n {
        let s = total * i as f64 / denom;
        while k + 1 < segs && cum[k + 1] < s {
            k += 1;
        }
        let l = cum[k + 1] - cum[k];
        let f = if l > 0.0 { ((s - cum[k]) / l).clamp(0.0, 1.0) } else { 0.0 };
        out.push(pts[k].lerp(pts[(k + 1) % m], f));
    }
    out
}

/// Append a band of triangles between `lower` (an existing boundary loop in boundary
/// orientation, or a previous ring) and `upper` (new ring with the same count, vertex `j`
/// above vertex `j`). Afterwards `upper` is a boundary loop with the same orientation as
/// `lower`. Each quad is split along its shorter diagonal.
pub(crate) fn stitch_band(
    mesh: &mut Mesh,
    lower: &[u32],
    upper: &[u32],
    group: i32,
    out: &mut Vec<u32>,
) -> EditResult<()> {
    let n = lower.len();
    debug_assert_eq!(n, upper.len());
    for j in 0..n {
        let (a0, a1) = (lower[j], lower[(j + 1) % n]);
        let (b0, b1) = (upper[j], upper[(j + 1) % n]);
        let d1 = mesh.vertex(a1).distance_squared(mesh.vertex(b0));
        let d2 = mesh.vertex(a0).distance_squared(mesh.vertex(b1));
        let tris = if d1 <= d2 { [[a1, a0, b0], [a1, b0, b1]] } else { [[a1, a0, b1], [a0, b0, b1]] };
        for tv in tris {
            out.push(mesh.append_triangle(tv, group)?);
        }
    }
    Ok(())
}
