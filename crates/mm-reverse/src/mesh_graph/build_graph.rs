// SPDX-License-Identifier: AGPL-3.0-only
//! What a parsed file becomes before anything is measured on it, scan and reference alike:
//! one indexed mesh, welded and rid of degenerate triangles, with vertex adjacency for region
//! growing, and normals turned the right way out where that can be told. A broken file is
//! refused here with a message, rather than read past the end of a buffer further on.
//!
//! Ported from ScanRuler (`src/core/geometry/buildGraph.ts` and the `ParsedMesh` type of
//! `src/core/types.ts`), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only. The progress text
//! ScanRuler's worker posts while it welds is not ported; Polymix's own meshes come in
//! through `MeshGraph::from_mesh`.

use glam::DVec3;

use super::adjacency::build_adjacency;
use super::normals::{compute_vertex_normals, extent_where, orient_over};
use super::weld::{filter_degenerate_triangles, weld_triangle_soup};
use super::MeshGraph;

/// Why a mesh could not be read: the message for the user.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct GraphError(pub String);

/// How a parsed file holds its triangles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParsedKind {
    /// Three corners per triangle, each written out (an STL): welded here.
    Soup,
    /// Distinct vertices and an index buffer.
    Indexed,
}

/// A mesh as a file parser hands it on.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedMesh {
    pub kind: ParsedKind,
    pub positions: Vec<DVec3>,
    /// Three vertex ids per triangle; `None` for a soup (and an indexed file with none).
    pub indices: Option<Vec<u32>>,
    /// The file says its triangles face the way they are meant to, so the winding is kept as
    /// written instead of guessed.
    pub keep_winding: bool,
}

impl ParsedMesh {
    /// A triangle soup, three corners per triangle.
    pub fn soup(positions: Vec<DVec3>) -> Self {
        Self { kind: ParsedKind::Soup, positions, indices: None, keep_winding: false }
    }

    /// Distinct vertices and three vertex ids per triangle.
    pub fn indexed(positions: Vec<DVec3>, indices: Vec<u32>) -> Self {
        Self { kind: ParsedKind::Indexed, positions, indices: Some(indices), keep_winding: false }
    }
}

fn err(msg: &str) -> GraphError {
    GraphError(msg.to_owned())
}

/// The mesh graph of a parsed file: welded when it is a soup, its degenerate triangles
/// dropped, adjacency and area-weighted normals computed, the winding turned outward unless
/// the file settles it. Refuses incomplete, non-finite or dangling input, and a file with no
/// triangle left, with ScanRuler's messages.
pub fn build_mesh_graph(parsed: ParsedMesh) -> Result<MeshGraph, GraphError> {
    if parsed.positions.is_empty() || (parsed.kind == ParsedKind::Soup && !parsed.positions.len().is_multiple_of(3)) {
        return Err(err("The file contains incomplete vertex or triangle coordinates."));
    }
    if parsed.positions.iter().any(|p| !p.is_finite()) {
        return Err(err("The file contains non-finite vertex coordinates."));
    }
    if let Some(indices) = &parsed.indices {
        if !indices.len().is_multiple_of(3) {
            return Err(err("The file contains incomplete triangle indices."));
        }
        if indices.iter().any(|&i| i as usize >= parsed.positions.len()) {
            return Err(err("The file references vertices that do not exist."));
        }
    }

    let (positions, indices) = match parsed.kind {
        ParsedKind::Soup => weld_triangle_soup(&parsed.positions),
        ParsedKind::Indexed => (parsed.positions, parsed.indices.unwrap_or_default()),
    };
    assemble(positions, indices, parsed.keep_winding, |_| true)
}

/// The mesh graph of flat arrays as a parser fills them — three coordinates per vertex (per
/// soup corner), three vertex ids per triangle — with the same checks as
/// `build_mesh_graph`, the coordinate count among them.
pub fn build_mesh_graph_from_flat(kind: ParsedKind, positions: &[f32], indices: Option<&[u32]>, keep_winding: bool) -> Result<MeshGraph, GraphError> {
    if positions.is_empty() || !positions.len().is_multiple_of(3) || (kind == ParsedKind::Soup && !positions.len().is_multiple_of(9)) {
        return Err(err("The file contains incomplete vertex or triangle coordinates."));
    }
    let positions = positions.as_chunks::<3>().0.iter().map(|p| DVec3::new(f64::from(p[0]), f64::from(p[1]), f64::from(p[2]))).collect();
    build_mesh_graph(ParsedMesh { kind, positions, indices: indices.map(<[u32]>::to_vec), keep_winding })
}

/// Adjacency, normals and scale for welded positions and triangles; `live` says which vertex
/// slots hold a vertex (all of them for a parsed file).
pub(crate) fn assemble(positions: Vec<DVec3>, indices: Vec<u32>, keep_winding: bool, live: impl Fn(usize) -> bool) -> Result<MeshGraph, GraphError> {
    let mut indices = filter_degenerate_triangles(&indices);
    if indices.is_empty() {
        return Err(err("The file contains no usable triangles — point clouds are not supported yet."));
    }
    let vertex_count = positions.len();
    if indices.iter().any(|&i| i as usize >= vertex_count) {
        return Err(err("The file references vertices that do not exist."));
    }

    let adjacency = build_adjacency(&indices, vertex_count);
    let mut normals = compute_vertex_normals(&positions, &indices);
    if !keep_winding {
        orient_over(&positions, &live, &mut indices, &mut normals);
    }

    let bbox_diag = extent_where(&positions, &live).diagonal();
    let triangle_count = indices.len() / 3;
    Ok(MeshGraph {
        positions,
        indices,
        normals,
        adj_offsets: adjacency.offsets,
        adj_list: adjacency.list,
        vertex_count,
        triangle_count,
        bbox_diag,
    })
}

impl MeshGraph {
    /// The graph of a Polymix mesh. Vertex ids are the mesh's own — `positions` is indexed by
    /// them, removed vertices' slots included, which nothing is joined to — so a region found
    /// on the graph names the mesh's vertices. The triangles are the live ones in id order.
    /// The centroid and box that orient the winding and scale the fits are taken over the
    /// live vertices. `keep_winding` keeps the winding as the mesh has it; otherwise an
    /// inside-out mesh is turned outward in the graph's copy (see `orient_normals_outward`).
    pub fn from_mesh(mesh: &mm_core::Mesh, keep_winding: bool) -> Result<MeshGraph, GraphError> {
        let positions = mesh.positions().to_vec();
        if positions.iter().enumerate().any(|(v, p)| mesh.is_vertex(v as u32) && !p.is_finite()) {
            return Err(err("The file contains non-finite vertex coordinates."));
        }
        let mut indices = Vec::with_capacity(mesh.triangle_count() * 3);
        for t in mesh.triangle_ids() {
            indices.extend_from_slice(&mesh.triangle(t));
        }
        assemble(positions, indices, keep_winding, |v| mesh.is_vertex(v as u32))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refuses_broken_input_with_a_message() {
        let e = build_mesh_graph(ParsedMesh::soup(vec![])).unwrap_err();
        assert_eq!(e.0, "The file contains incomplete vertex or triangle coordinates.");
        let e = build_mesh_graph(ParsedMesh::soup(vec![DVec3::ZERO; 4])).unwrap_err();
        assert_eq!(e.0, "The file contains incomplete vertex or triangle coordinates.");
        let e = build_mesh_graph(ParsedMesh::soup(vec![DVec3::ZERO, DVec3::X, DVec3::new(f64::NAN, 0.0, 0.0)])).unwrap_err();
        assert_eq!(e.0, "The file contains non-finite vertex coordinates.");
        let e = build_mesh_graph(ParsedMesh::indexed(vec![DVec3::ZERO, DVec3::X, DVec3::Y], vec![0, 1])).unwrap_err();
        assert_eq!(e.0, "The file contains incomplete triangle indices.");
        let e = build_mesh_graph(ParsedMesh::indexed(vec![DVec3::ZERO, DVec3::X, DVec3::Y], vec![0, 1, 3])).unwrap_err();
        assert_eq!(e.0, "The file references vertices that do not exist.");
        let e = build_mesh_graph(ParsedMesh::indexed(vec![DVec3::ZERO, DVec3::X, DVec3::Y], vec![0, 1, 1])).unwrap_err();
        assert_eq!(e.0, "The file contains no usable triangles — point clouds are not supported yet.");
    }

    /// A tetrahedron wound inward is turned outward, normals and triangles together; wound
    /// outward, or kept as written, it is left alone.
    #[test]
    fn turns_an_inside_out_mesh_the_right_way_out() {
        let p = vec![DVec3::ZERO, DVec3::X, DVec3::Y, DVec3::Z];
        let outward: Vec<u32> = vec![0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3];
        let inward: Vec<u32> = outward.as_chunks::<3>().0.iter().flat_map(|t| [t[0], t[2], t[1]]).collect();
        let good = build_mesh_graph(ParsedMesh::indexed(p.clone(), outward.clone())).unwrap();
        assert_eq!(good.indices, outward);
        let fixed = build_mesh_graph(ParsedMesh::indexed(p.clone(), inward.clone())).unwrap();
        assert_eq!(fixed.indices, outward);
        for v in 0..4 {
            assert!((fixed.normals[v] - good.normals[v]).length() < 1e-12);
        }
        // The far corner's normal points away from the solid.
        assert!(good.normals[3].z > 0.0 && good.normals[1].x > 0.0);
        let kept = build_mesh_graph(ParsedMesh { keep_winding: true, ..ParsedMesh::indexed(p, inward.clone()) }).unwrap();
        assert_eq!(kept.indices, inward);
        assert!((good.bbox_diag - 3f64.sqrt()).abs() < 1e-12);
    }

    // normals.test.ts — 'scan normal orientation', on the tetrahedron and the open quad.
    mod scan_normal_orientation {
        use super::*;

        const TETRA_POSITIONS: [f32; 12] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        /// Outward winding: every face normal points away from the enclosed volume.
        const TETRA_OUTWARD: [u32; 12] = [0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3];
        /// Same faces, each reversed — an inside-out export.
        const TETRA_INWARD: [u32; 12] = [0, 1, 2, 0, 3, 1, 0, 2, 3, 1, 3, 2];

        /// Six times the signed volume enclosed by the winding — positive when the triangles
        /// wind outward.
        fn signed_volume6(positions: &[DVec3], indices: &[u32]) -> f64 {
            indices
                .as_chunks::<3>()
                .0
                .iter()
                .map(|t| {
                    let (a, b, c) = (positions[t[0] as usize], positions[t[1] as usize], positions[t[2] as usize]);
                    a.x * (b.y * c.z - b.z * c.y) + a.y * (b.z * c.x - b.x * c.z) + a.z * (b.x * c.y - b.y * c.x)
                })
                .sum()
        }

        fn tetra(indices: &[u32], keep_winding: bool) -> MeshGraph {
            build_mesh_graph_from_flat(ParsedKind::Indexed, &TETRA_POSITIONS, Some(indices), keep_winding).unwrap()
        }

        #[test]
        fn keeps_a_correctly_wound_closed_mesh_as_it_is() {
            // Vertex 3 sits at the +z apex, so its outward normal must have positive z.
            assert!(tetra(&TETRA_OUTWARD, false).normals[3].z > 0.0);
        }

        #[test]
        fn flips_the_normals_of_an_inside_out_closed_mesh() {
            assert!(tetra(&TETRA_INWARD, false).normals[3].z > 0.0);
        }

        #[test]
        fn reverses_the_winding_along_with_the_normals() {
            let g = tetra(&TETRA_INWARD, false);
            assert!(signed_volume6(&g.positions, &g.indices) > 0.0);
        }

        #[test]
        fn keeps_the_winding_of_a_mesh_whose_file_says_it_is_settled() {
            let g = tetra(&TETRA_INWARD, true);
            assert_eq!(g.indices, TETRA_INWARD);
            assert!(g.normals[3].z < 0.0);
        }

        #[test]
        fn leaves_an_open_sheet_exactly_as_wound() {
            // A flat quad wound so its normals face −z. There is no enclosed volume, so there
            // is nothing to decide by — the winding stands.
            let positions = [0.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0];
            let g = build_mesh_graph_from_flat(ParsedKind::Indexed, &positions, Some(&[0, 1, 2, 2, 1, 3]), false).unwrap();
            for v in 0..4 {
                assert!(g.normals[v].z < 0.0);
            }
        }
    }

    // inputValidation.test.ts — 'mesh input validation', the cases that reach buildMeshGraph.
    #[test]
    fn rejects_non_finite_coordinates_before_welding_or_normal_generation() {
        for coordinate in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let positions = [coordinate, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
            let e = build_mesh_graph_from_flat(ParsedKind::Soup, &positions, None, false).unwrap_err();
            assert!(e.0.contains("non-finite"), "{}", e.0);
        }
    }

    #[test]
    fn rejects_invalid_indices_even_when_their_triangle_is_degenerate() {
        let e = build_mesh_graph_from_flat(ParsedKind::Indexed, &[0.0; 9], Some(&[999, 999, 0]), false).unwrap_err();
        assert!(e.0.contains("vertices that do not exist"), "{}", e.0);
    }

    #[test]
    fn builds_from_flat_arrays_as_a_parser_fills_them() {
        // Two triangles of a square as a soup: the shared corners weld.
        let soup = [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0];
        let g = build_mesh_graph_from_flat(ParsedKind::Soup, &soup, None, false).unwrap();
        assert_eq!((g.vertex_count, g.triangle_count), (4, 2));
        assert_eq!(g.indices, vec![0, 1, 2, 0, 2, 3]);
        let e = build_mesh_graph_from_flat(ParsedKind::Soup, &soup[..12], None, false).unwrap_err();
        assert_eq!(e.0, "The file contains incomplete vertex or triangle coordinates.");
        let e = build_mesh_graph_from_flat(ParsedKind::Indexed, &soup[..8], Some(&[0, 1, 2]), false).unwrap_err();
        assert_eq!(e.0, "The file contains incomplete vertex or triangle coordinates.");
    }

    #[test]
    fn builds_from_a_polymix_mesh_with_its_own_vertex_ids() {
        let p = vec![DVec3::ZERO, DVec3::X, DVec3::Y, DVec3::Z];
        let tris = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]];
        let (mesh, _) = mm_core::Mesh::from_triangles(&p, &tris, None);
        let g = MeshGraph::from_mesh(&mesh, false).unwrap();
        assert_eq!(g.vertex_count, 4);
        assert_eq!(g.triangle_count, 4);
        assert_eq!(g.positions, p);
        let mut nb: Vec<u32> = g.neighbours(0).to_vec();
        nb.sort_unstable();
        nb.dedup();
        assert_eq!(nb, vec![1, 2, 3]);
    }
}
