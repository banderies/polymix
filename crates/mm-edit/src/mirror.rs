//! Mirror (Meshmixer *Edit > Mirror*): reflect a copy of the mesh across a plane and
//! optionally weld it to the original along the plane.

use crate::util::{normalize_plane, rollback};
use crate::{EditError, EditResult};
use mm_core::{Mesh, Plane};

/// Output of [`mirror`].
#[derive(Clone, Debug, Default)]
pub struct MirrorResult {
    /// Triangles of the mirrored copy.
    pub new_triangles: Vec<u32>,
    /// Number of seam vertices shared by the original and the copy.
    pub welded_vertices: usize,
}

/// Reflect a copy of the whole mesh across `plane` and append it. The copy's triangles are
/// reversed so it stays outward-oriented. With `weld`, boundary vertices lying on the plane
/// that have a boundary edge in the plane are shared by both halves, so e.g. a half-sphere
/// open at the plane becomes a closed sphere. Uses a tolerance of 1e-6 × the bounding-box
/// diagonal (see [`mirror_with_tolerance`]).
pub fn mirror(mesh: &mut Mesh, plane: Plane, weld: bool) -> EditResult<MirrorResult> {
    let tol = 1e-6 * mesh.bounds().diagonal();
    mirror_with_tolerance(mesh, plane, weld, tol)
}

/// [`mirror`] with an explicit on-plane distance tolerance for welding.
pub fn mirror_with_tolerance(mesh: &mut Mesh, plane: Plane, weld: bool, tol: f64) -> EditResult<MirrorResult> {
    let plane = normalize_plane(plane)?;
    let tol = tol.max(0.0);
    let n = plane.normal;
    let mut res = MirrorResult::default();
    let mut weldv = vec![false; mesh.max_vid()];
    if weld {
        let on = |m: &Mesh, v: u32| plane.signed_distance(m.vertex(v)).abs() <= tol;
        for e in mesh.boundary_edge_ids().collect::<Vec<_>>() {
            let [a, b] = mesh.edge_vertices(e);
            if on(mesh, a) && on(mesh, b) {
                weldv[a as usize] = true;
                weldv[b as usize] = true;
            }
        }
        for v in 0..weldv.len() as u32 {
            if weldv[v as usize] {
                mesh.set_vertex(v, plane.project(mesh.vertex(v)));
                res.welded_vertices += 1;
            }
        }
    }
    let tris: Vec<u32> = mesh.triangle_ids().collect();
    let mut vmap = vec![mm_core::INVALID; mesh.max_vid()];
    for v in mesh.vertex_ids().collect::<Vec<_>>() {
        vmap[v as usize] = if weldv[v as usize] {
            v
        } else {
            let p = mesh.vertex(v);
            mesh.append_vertex(p - n * (2.0 * plane.signed_distance(p)))
        };
    }
    for t in tris {
        let [a, b, c] = mesh.triangle(t);
        if weldv[a as usize] && weldv[b as usize] && weldv[c as usize] {
            // Lies in the plane: its mirror image would coincide with it.
            continue;
        }
        let tv = [vmap[a as usize], vmap[c as usize], vmap[b as usize]];
        match mesh.append_triangle(tv, mesh.tri_group(t)) {
            Ok(id) => res.new_triangles.push(id),
            Err(e) => {
                rollback(mesh, &res.new_triangles);
                mesh.remove_isolated_vertices();
                return Err(EditError::Failed(format!("mirror weld produced a non-manifold seam: {e}")));
            }
        }
    }
    mesh.remove_isolated_vertices();
    Ok(res)
}
