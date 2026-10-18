//! `mm-solid`: Meshmixer-style "Make Solid" and related volumetric operations.
//!
//! The pipeline turns an arbitrary triangle soup (holes, self-intersections, overlapping
//! or duplicated pieces, inverted shells) into a closed, 2-manifold, consistently oriented
//! solid:
//!
//! 1. [`WindingNumber`] — hierarchical fast generalized winding numbers (Barill et al.
//!    2018) decide inside/outside robustly, even for open or self-intersecting input.
//! 2. [`field::MeshSdf`] + [`grid::build_field`] — a narrow-band signed distance field on a
//!    sparse grid of 8³ bricks. Exact unsigned distances (BVH nearest point) are computed
//!    only in bricks touched by the input triangles' padded bounding boxes; the sign comes
//!    from the winding number. Memory scales with the surface area, not the volume.
//! 3. [`extract::marching_tetrahedra`] — iso-surface extraction on the Kuhn
//!    (Freudenthal) 6-tetrahedra subdivision of the grid. The zero set of a piecewise
//!    linear function on a simplicial complex is always a closed 2-manifold, so the output
//!    is guaranteed watertight, edge-manifold and free of bowtie vertices, and it is
//!    oriented outward by construction.
//! 4. [`cleanup`] — link-condition-checked short-edge collapses and quality flips remove
//!    the slivers that marching tetrahedra produces, tiny shells are discarded, and in
//!    [`SolidMode::Accurate`] vertices are projected back onto the input surface.
//!
//! Higher-level operations: [`make_solid`], [`offset_solid`], [`hollow`] and
//! [`sdf_boolean`]. Custom implicit surfaces can be meshed with [`mesh_implicit`] or by
//! implementing [`FieldSource`].

// `!(x > 0.0)` is used on purpose so that NaN parameters are rejected; index loops are
// kept where they mirror tensor formulas.
#![allow(clippy::neg_cmp_op_on_partial_ord, clippy::needless_range_loop)]

pub mod blocky;
pub mod cleanup;
pub mod extract;
pub mod field;
pub mod grid;
mod hash;
pub mod pattern;
mod progress;
pub mod solid;
pub mod winding;

pub use field::{ImplicitField, MeshSdf};
pub use grid::{build_field, FieldSource, GridSpec, SparseField};
pub use solid::{
    cell_size_for_resolution, hollow, make_solid, mesh_implicit, offset_solid, sdf_boolean, transfer_face_groups, transfer_vertex_colors, BoolOp,
    MakeSolidParams, SolidMode,
};
pub use pattern::{make_pattern, Composition, Gradient, PatternParams, PatternType, Tiling};
pub use winding::{winding_number_exact, WindingNumber};

/// Errors produced by the solid operations.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum SolidError {
    /// The progress callback returned `false`.
    #[error("operation cancelled")]
    Cancelled,
    /// The input has no triangles (or only degenerate ones).
    #[error("input mesh is empty")]
    EmptyInput,
    /// A parameter is out of range.
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),
    /// The requested grid would exceed the supported size.
    #[error("grid too large: {0}")]
    GridTooLarge(String),
    /// An internal invariant was violated (this is a bug).
    #[error("internal error: {0}")]
    Internal(String),
}

/// Result type of this crate.
pub type Result<T> = std::result::Result<T, SolidError>;

/// Optional progress callback: receives a fraction in `[0, 1]`; returning `false` cancels
/// the operation, which then returns [`SolidError::Cancelled`]. It is always invoked on the
/// calling thread.
pub type Progress<'a> = Option<&'a dyn Fn(f32) -> bool>;
