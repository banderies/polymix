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
