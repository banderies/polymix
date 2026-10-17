//! Face-selection operations: discard, separate, duplicate, flip normals.

use crate::util::valid_unique_tris;
use mm_core::Mesh;

/// Delete the selected triangles (and vertices left unused). Returns the number removed.
pub fn discard(mesh: &mut Mesh, tris: &[u32]) -> usize {
    let t = valid_unique_tris(mesh, tris);
    mesh.remove_triangles(t.iter().copied());
    t.len()
}

/// Remove the selected triangles from `mesh` and return them as a new (compact) mesh.
pub fn separate(mesh: &mut Mesh, tris: &[u32]) -> Mesh {
    let t = valid_unique_tris(mesh, tris);
    let (sub, _) = mesh.submesh(&t);
    mesh.remove_triangles(t.iter().copied());
    sub
}

/// Copy of the selected triangles as a new (compact) mesh; `mesh` is unchanged.
pub fn duplicate(mesh: &Mesh, tris: &[u32]) -> Mesh {
    let t = valid_unique_tris(mesh, tris);
    mesh.submesh(&t).0
}

/// Reverse the orientation of the selected triangles. Flipping a proper subset of a
/// connected surface leaves its border inconsistently oriented, as in Meshmixer.
pub fn flip_normals(mesh: &mut Mesh, tris: &[u32]) {
    for t in valid_unique_tris(mesh, tris) {
        mesh.reverse_triangle(t);
    }
}
