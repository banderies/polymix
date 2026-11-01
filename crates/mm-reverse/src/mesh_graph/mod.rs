// SPDX-License-Identifier: AGPL-3.0-only
//! The mesh as the measuring core walks it: one indexed mesh, welded and rid of degenerate
//! triangles, with per-vertex normals turned the right way out and vertex adjacency for
//! region growing — and what is computed over it beside the fits: the mean curvature map, the
//! split of sharp edges for shading, the corner slots the mesh mode draws edges from, the
//! triangles of a marking and the connected pieces.
//!
//! Ported from ScanRuler's `src/core/geometry/` (`adjacency`, `normals`, `buildGraph`,
//! `weld`, `region`, `unionFind`, `curvature`, `crease`, `wireSlots`, `scratch`) and the
//! `MeshGraph` type of `src/core/types.ts`, © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.

// The walks test `!(x > limit)` where ScanRuler does, so that a NaN fails the test it would
// otherwise slip through; the negation is the point.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

pub mod adjacency;
pub mod build_graph;
pub mod crease;
pub mod curvature;
pub mod normals;
pub mod region;
pub mod scratch;
pub mod union_find;
pub mod weld;
pub mod wire_slots;

use glam::DVec3;

pub use adjacency::{build_adjacency, Adjacency};
pub use build_graph::{build_mesh_graph, build_mesh_graph_from_flat, GraphError, ParsedKind, ParsedMesh};
pub use crease::{
    coplanar_share, crease_angle_of, crease_budget, crease_for, crease_setting, looks_tessellated, scan_render, split_creases, CreaseMode, CreaseReport, CreaseSetting, CreaseSkip,
    CreaseSplit, ScanRender, CREASE_ANGLE_DEG, CREASE_ANGLE_MAX, CREASE_ANGLE_MIN, CREASE_ANGLE_STEP,
};
pub use curvature::{curvature_range, curvature_scale, mean_curvature, FieldScale, CURVATURE_SMOOTHING};
pub use normals::{compute_vertex_normals, extent_of, orient_normals_outward, tetra_volume6};
pub use region::triangles_within;
pub use scratch::Stamps;
pub use union_find::{find_root, vertex_components, UnionFind};
pub use weld::{filter_degenerate_triangles, weld_triangle_soup};
pub use wire_slots::{wire_conflicts, wire_slots, WIRE_SLOT_COUNT};

/// A welded triangle mesh with what every walk over it needs. Vertex `v`'s position and
/// normal are `positions[v]` and `normals[v]`, its neighbours
/// `adj_list[adj_offsets[v]..adj_offsets[v + 1]]` (a neighbour listed once per triangle the
/// two share); `indices` holds three vertex ids per triangle.
#[derive(Debug, Clone, PartialEq)]
pub struct MeshGraph {
    pub positions: Vec<DVec3>,
    pub indices: Vec<u32>,
    /// Unit area-weighted vertex normals, turned outward where that can be told.
    pub normals: Vec<DVec3>,
    pub adj_offsets: Vec<u32>,
    pub adj_list: Vec<u32>,
    pub vertex_count: usize,
    pub triangle_count: usize,
    /// The diagonal of the mesh's bounding box — the scale every "too large" test is read
    /// against.
    pub bbox_diag: f64,
}

impl MeshGraph {
    /// The neighbours of vertex `v`, each once per triangle the two share.
    #[inline]
    pub fn neighbours(&self, v: u32) -> &[u32] {
        let v = v as usize;
        &self.adj_list[self.adj_offsets[v] as usize..self.adj_offsets[v + 1] as usize]
    }

    /// Vertex `v`'s position.
    #[inline]
    pub fn position(&self, v: u32) -> DVec3 {
        self.positions[v as usize]
    }

    /// Vertex `v`'s normal.
    #[inline]
    pub fn normal(&self, v: u32) -> DVec3 {
        self.normals[v as usize]
    }
}
