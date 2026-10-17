//! Whole-mesh transforms.

use crate::linalg::{add_outer, sym_eigen3};
use mm_core::{DMat3, DMat4, DVec3, Mesh};

/// Apply an affine transform. Mirroring transforms (negative determinant) also reverse the
/// triangle orientation so normals keep pointing outward.
pub fn transform_mesh(mesh: &mut Mesh, m: DMat4) {
    mesh.transform(|p| m.transform_point3(p));
    if DMat3::from_mat4(m).determinant() < 0.0 {
        mesh.reverse_orientation();
    }
}

/// Translate so the lowest point along `up` sits at 0 (Meshmixer "drop to ground").
/// Returns the translation applied.
pub fn drop_to_ground(mesh: &mut Mesh, up: DVec3) -> DVec3 {
    let up = up.normalize_or_zero();
    if up == DVec3::ZERO || mesh.vertex_count() == 0 {
        return DVec3::ZERO;
    }
    let lo = mesh.vertex_ids().map(|v| mesh.vertex(v).dot(up)).fold(f64::INFINITY, f64::min);
    let t = -up * lo;
    mesh.transform(|p| p + t);
    t
}

/// Translate the bounding-box centre to the origin. Returns the translation applied.
pub fn center_on_origin(mesh: &mut Mesh) -> DVec3 {
    let b = mesh.bounds();
    if b.is_empty() {
        return DVec3::ZERO;
    }
    let t = -b.center();
    mesh.transform(|p| p + t);
    t
}

/// Rotate (about the area-weighted surface centroid) so the principal axes of the surface
/// align with X (largest extent), Y, Z, and move the centroid to the origin. Returns the
/// applied transform (a proper rotation; orientation is preserved).
pub fn align_to_principal_axes(mesh: &mut Mesh) -> DMat4 {
    let c = mm_core::measure::surface_centroid(mesh);
    let mut cov = [[0.0; 3]; 3];
    for t in mesh.triangle_ids() {
        let a = mesh.tri_area(t);
        for p in mesh.tri_positions(t) {
            add_outer(&mut cov, p - c, a);
        }
    }
    let (_, v) = sym_eigen3(cov);
    let x = v[2];
    let y = v[1];
    let z = x.cross(y).normalize_or_zero();
    if x == DVec3::ZERO || z == DVec3::ZERO {
        return DMat4::IDENTITY;
    }
    // Rows are the new axes, so R * (p - c) expresses p in the principal frame.
    let r = DMat3::from_cols(x, y, z).transpose();
    let m = DMat4::from_mat3(r) * DMat4::from_translation(-c);
    transform_mesh(mesh, m);
    m
}
