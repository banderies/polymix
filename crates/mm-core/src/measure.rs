//! Global measurements: area, volume, centroid.

use crate::geom::Aabb;
use crate::mesh::Mesh;
use glam::DVec3;

pub fn surface_area(m: &Mesh) -> f64 {
    m.triangle_ids().map(|t| m.tri_area(t)).sum()
}

/// Signed volume via the divergence theorem (meaningful for closed meshes;
/// positive when outward-oriented).
pub fn volume(m: &Mesh) -> f64 {
    m.triangle_ids()
        .map(|t| {
            let [a, b, c] = m.tri_positions(t);
            a.dot(b.cross(c))
        })
        .sum::<f64>()
        / 6.0
}

/// Area-weighted centroid of the surface.
pub fn surface_centroid(m: &Mesh) -> DVec3 {
    let mut c = DVec3::ZERO;
    let mut w = 0.0;
    for t in m.triangle_ids() {
        let a = m.tri_area(t);
        c += m.tri_centroid(t) * a;
        w += a;
    }
    if w > 0.0 {
        c / w
    } else {
        DVec3::ZERO
    }
}

/// Centre of mass: of the enclosed solid for a closed mesh, of the surface otherwise (an open
/// or degenerate mesh has no well-defined volume).
pub fn center_of_mass(m: &Mesh) -> DVec3 {
    if m.is_closed() {
        center_of_mass_with(m, &m.bounds(), true)
    } else {
        surface_centroid(m)
    }
}

/// [`center_of_mass`] for a caller that already has the mesh's bounds and knows whether it is
/// closed (no boundary edges), so neither is computed again. Same result, bit for bit.
pub fn center_of_mass_with(m: &Mesh, bounds: &Aabb, closed: bool) -> DVec3 {
    if closed {
        // Sum of signed tetrahedra (o, a, b, c) about a nearby origin, for precision.
        let o = bounds.center();
        let (mut c, mut v) = (DVec3::ZERO, 0.0);
        for t in m.triangle_ids() {
            let [a, b, d] = m.tri_positions(t).map(|p| p - o);
            let vt = a.dot(b.cross(d)) / 6.0;
            c += (a + b + d) * (vt / 4.0);
            v += vt;
        }
        let scale = bounds.diagonal().powi(3);
        if v.abs() > 1e-9 * scale {
            return o + c / v;
        }
    }
    surface_centroid(m)
}

/// Summary statistics for display.
#[derive(Clone, Copy, Debug, Default)]
pub struct MeshStats {
    pub vertices: usize,
    pub triangles: usize,
    pub edges: usize,
    pub boundary_edges: usize,
    pub area: f64,
    pub volume: f64,
    pub bounds_min: DVec3,
    pub bounds_max: DVec3,
    pub avg_edge_length: f64,
}

pub fn stats(m: &Mesh) -> MeshStats {
    let b = m.bounds();
    MeshStats {
        vertices: m.vertex_count(),
        triangles: m.triangle_count(),
        edges: m.edge_count(),
        boundary_edges: m.boundary_edge_ids().count(),
        area: surface_area(m),
        volume: volume(m),
        bounds_min: b.min,
        bounds_max: b.max,
        avg_edge_length: m.average_edge_length(),
    }
}

/// Interior angles of triangle `t` (radians, in corner order).
pub fn triangle_angles(m: &Mesh, t: u32) -> [f64; 3] {
    let p = m.triangle(t).map(|v| m.vertex(v));
    [0, 1, 2].map(|i| (p[(i + 1) % 3] - p[i]).angle_between(p[(i + 2) % 3] - p[i]))
}

/// Aspect ratio of triangle `t`: circumradius over twice the inradius (1 for an equilateral
/// triangle, growing without bound for slivers).
pub fn triangle_aspect_ratio(m: &Mesh, t: u32) -> f64 {
    let p = m.triangle(t).map(|v| m.vertex(v));
    let (a, b, c) = (p[1].distance(p[2]), p[2].distance(p[0]), p[0].distance(p[1]));
    let area = 0.5 * (p[1] - p[0]).cross(p[2] - p[0]).length();
    if area <= 0.0 {
        return f64::INFINITY;
    }
    let s = 0.5 * (a + b + c);
    let inradius = area / s;
    let circumradius = a * b * c / (4.0 * area);
    circumradius / (2.0 * inradius)
}

/// Mixed Voronoi area per vertex (indexed by vertex id): each triangle's Voronoi regions
/// when it is non-obtuse, otherwise half its area at the obtuse corner and a quarter at the
/// others (Meyer et al.). The areas sum to the surface area.
pub fn mixed_areas(m: &Mesh) -> Vec<f64> {
    let mut area = vec![0.0; m.max_vid()];
    for t in m.triangle_ids() {
        let tv = m.triangle(t);
        let p = tv.map(|v| m.vertex(v));
        let a = triangle_angles(m, t);
        let ta = m.tri_area(t);
        let obtuse = a.iter().position(|&x| x > std::f64::consts::FRAC_PI_2);
        for j in 0..3 {
            area[tv[j] as usize] += match obtuse {
                Some(o) if o == j => ta / 2.0,
                Some(_) => ta / 4.0,
                None => {
                    // Edges from j to the other corners, weighted by the opposite cotangents.
                    let (k, l) = ((j + 1) % 3, (j + 2) % 3);
                    let cot = |x: f64| x.cos() / x.sin().max(1e-300);
                    (p[j].distance_squared(p[k]) * cot(a[l]) + p[j].distance_squared(p[l]) * cot(a[k])) / 8.0
                }
            };
        }
    }
    area
}

/// Discrete Gaussian curvature per vertex (indexed by vertex id): the angle deficit over the
/// mixed Voronoi area. Boundary vertices use the geodesic deficit (π − Σ angles).
pub fn gaussian_curvature(m: &Mesh) -> Vec<f64> {
    let mut angle = vec![0.0; m.max_vid()];
    let area = mixed_areas(m);
    for t in m.triangle_ids() {
        let tv = m.triangle(t);
        let a = triangle_angles(m, t);
        for j in 0..3 {
            angle[tv[j] as usize] += a[j];
        }
    }
    (0..m.max_vid() as u32)
        .map(|v| {
            if !m.is_vertex(v) || area[v as usize] <= 0.0 {
                return 0.0;
            }
            let full = if m.is_boundary_vertex(v) { std::f64::consts::PI } else { std::f64::consts::TAU };
            (full - angle[v as usize]) / area[v as usize]
        })
        .collect()
}

/// Discrete mean curvature per vertex (indexed by vertex id) from the cotangent Laplacian,
/// positive where the surface is convex (e.g. `1/r` on a sphere of radius `r`).
pub fn mean_curvature(m: &Mesh) -> Vec<f64> {
    let normals = m.compute_vertex_normals();
    let mut lap = vec![DVec3::ZERO; m.max_vid()];
    let area = mixed_areas(m);
    for t in m.triangle_ids() {
        let tv = m.triangle(t);
        let p = tv.map(|v| m.vertex(v));
        for j in 0..3 {
            let (i, k, o) = (j, (j + 1) % 3, (j + 2) % 3);
            // Cotangent of the angle at `o`, opposite edge (i, k).
            let (u, w) = (p[i] - p[o], p[k] - p[o]);
            let cot = u.dot(w) / u.cross(w).length().max(1e-300);
            lap[tv[i] as usize] += (p[k] - p[i]) * (0.5 * cot);
            lap[tv[k] as usize] += (p[i] - p[k]) * (0.5 * cot);
        }
    }
    (0..m.max_vid())
        .map(|v| {
            if area[v] <= 0.0 || !m.is_vertex(v as u32) || m.is_boundary_vertex(v as u32) {
                return 0.0;
            }
            // Δx = −2 H n; the Laplacian points into convex surfaces.
            -0.5 * (lap[v] / area[v]).dot(normals[v])
        })
        .collect()
}

/// Result of [`clearance`].
#[derive(Clone, Debug)]
pub struct Clearance {
    /// Distance from each vertex of the first mesh to the second (indexed by vertex id).
    pub dist_a: Vec<f64>,
    /// Distance from each vertex of the second mesh to the first, when computed both ways.
    pub dist_b: Option<Vec<f64>>,
    /// The smallest distance found (0 when the meshes intersect).
    pub min: f64,
    /// Closest points (on the first and the second mesh).
    pub closest: (DVec3, DVec3),
    pub intersecting: bool,
}

/// Clearance between two meshes: per-vertex distances to the other surface (one or both
/// ways), the closest pair of points, and whether the surfaces cross each other.
pub fn clearance(a: &Mesh, bvh_a: &crate::Bvh, b: &Mesh, bvh_b: &crate::Bvh, both_ways: bool) -> Clearance {
    use rayon::prelude::*;
    let one_way = |from: &Mesh, to: &Mesh, bvh: &crate::Bvh| -> (Vec<f64>, Option<(f64, DVec3, DVec3)>) {
        let ids: Vec<u32> = from.vertex_ids().collect();
        let hits: Vec<(u32, f64, DVec3)> = ids
            .par_iter()
            .filter_map(|&v| bvh.nearest(to, from.vertex(v), f64::INFINITY).map(|h| (v, h.dist_sq.sqrt(), h.point)))
            .collect();
        let mut d = vec![f64::INFINITY; from.max_vid()];
        let mut best: Option<(f64, DVec3, DVec3)> = None;
        for (v, dist, q) in hits {
            d[v as usize] = dist;
            if best.is_none_or(|b| dist < b.0) {
                best = Some((dist, from.vertex(v), q));
            }
        }
        (d, best)
    };
    let (dist_a, best_a) = one_way(a, b, bvh_b);
    let (dist_b, best_b) = if both_ways { let (d, x) = one_way(b, a, bvh_a); (Some(d), x) } else { (None, None) };
    let mut min = f64::INFINITY;
    let mut closest = (DVec3::ZERO, DVec3::ZERO);
    if let Some((d, p, q)) = best_a {
        (min, closest) = (d, (p, q));
    }
    if let Some((d, p, q)) = best_b.filter(|x| x.0 < min) {
        (min, closest) = (d, (q, p));
    }
    // Crossing surfaces: any triangle pair that intersects.
    let tris: Vec<u32> = a.triangle_ids().collect();
    let crossing = tris.par_iter().find_map_any(|&t| {
        let p = a.tri_positions(t);
        let mut found = None;
        bvh_b.query_aabb(&crate::Aabb::from_points(p), |o| {
            if found.is_none() && crate::intersect::triangles_intersect(p, b.tri_positions(o)) {
                found = Some(a.tri_centroid(t));
            }
        });
        found
    });
    if let Some(c) = crossing {
        min = 0.0;
        closest = (c, c);
    }
    Clearance { dist_a, dist_b, min, closest, intersecting: crossing.is_some() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;

    #[test]
    fn center_of_mass_of_solids_and_surfaces() {
        // A cube shifted off the origin: its centre.
        let mut cube = primitives::cube(2.0, 3);
        cube.transform(|p| p + DVec3::new(10.0, -4.0, 3.0));
        assert!((center_of_mass(&cube) - DVec3::new(10.0, -4.0, 3.0)).length() < 1e-9);
        // A solid cone (apex up, base at z = -1): the mass sits a quarter of the height above
        // the base, below the surface centroid.
        let pts: Vec<DVec3> = (0..64).map(|i| {
            let a = std::f64::consts::TAU * i as f64 / 64.0;
            DVec3::new(a.cos(), a.sin(), -1.0)
        }).collect();
        let mut pos = pts.clone();
        pos.push(DVec3::new(0.0, 0.0, 3.0));
        pos.push(DVec3::new(0.0, 0.0, -1.0));
        let (apex, base) = (64u32, 65u32);
        let mut tris = Vec::new();
        for i in 0..64u32 {
            let j = (i + 1) % 64;
            tris.push([i, j, apex]);
            tris.push([j, i, base]);
        }
        let cone = Mesh::from_triangles(&pos, &tris, None).0;
        assert!(cone.is_closed() && volume(&cone) > 0.0);
        let c = center_of_mass(&cone);
        assert!(c.truncate().length() < 1e-9 && (c.z - 0.0).abs() < 0.01, "{c:?}");
        // An open surface falls back to the surface centroid.
        let plane = primitives::plane_grid(2.0, 2.0, 4, 4);
        assert!(center_of_mass(&plane).length() < 1e-12);
    }

    /// `center_of_mass` as it was before `center_of_mass_with`: bounds computed twice, by
    /// walking the triangles.
    fn center_of_mass_before(m: &Mesh) -> DVec3 {
        let bounds = |m: &Mesh| Aabb::from_points(m.triangle_ids().flat_map(|t| m.tri_positions(t)));
        if m.is_closed() {
            let o = bounds(m).center();
            let (mut c, mut v) = (DVec3::ZERO, 0.0);
            for t in m.triangle_ids() {
                let [a, b, d] = m.tri_positions(t).map(|p| p - o);
                let vt = a.dot(b.cross(d)) / 6.0;
                c += (a + b + d) * (vt / 4.0);
                v += vt;
            }
            let scale = bounds(m).diagonal().powi(3);
            if v.abs() > 1e-9 * scale {
                return o + c / v;
            }
        }
        surface_centroid(m)
    }

    #[test]
    fn center_of_mass_with_matches_the_old_path_bit_for_bit() {
        let mut shifted = primitives::icosphere(1.3, 3);
        shifted.transform(|p| DVec3::new(p.x * 2.0 + 7.25, p.y - 3.5, p.z * 0.5 + 1e3));
        let mut open = primitives::cube(2.0, 4);
        let first: Vec<u32> = open.triangle_ids().take(3).collect();
        open.remove_triangles(first);
        let flat = primitives::plane_grid(3.0, 1.0, 5, 2);
        let tube = primitives::cylinder(0.4, 5.0, 16, 8, true);
        for (name, m) in [("shifted sphere", &shifted), ("open cube", &open), ("plane", &flat), ("capped tube", &tube), ("empty", &Mesh::new())] {
            let want = center_of_mass_before(m).to_array().map(f64::to_bits);
            assert_eq!(center_of_mass(m).to_array().map(f64::to_bits), want, "{name}");
            let closed = m.boundary_edge_ids().count() == 0;
            assert_eq!(center_of_mass_with(m, &m.bounds(), closed).to_array().map(f64::to_bits), want, "{name}");
        }
    }
}
