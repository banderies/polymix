//! Global measurements: area, volume, centroid.

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
