//! Isotropic remeshing and quadric-error mesh simplification for [`mm_core::Mesh`].
//!
//! * [`remesh`] — isotropic (optionally curvature-adaptive) remesher, the equivalent of
//!   Meshmixer's *Edit > Remesh*. Split / collapse / flip / tangential-relax / project
//!   passes after Botsch & Kobbelt, "A Remeshing Approach to Multiresolution Modeling"
//!   (SGP 2004), structured like geometry3Sharp's `Remesher` (Boost License, R. Schmidt).
//! * [`reduce`] — Garland–Heckbert quadric-error edge-collapse simplification
//!   ("Surface Simplification Using Quadric Error Metrics", SIGGRAPH 1997), the
//!   equivalent of Meshmixer's *Reduce* / geometry3Sharp's `Reducer`.
//! * [`refine_uniform`], [`collapse_short_edges`], [`edge_length_stats`] — small helpers
//!   used by other tools.
//!
//! All operations work in place, keep the mesh valid ([`Mesh::check_validity`]) at every
//! step (so cancelling through the progress callback is always safe), and can be
//! restricted to a triangle region; triangles outside the region are left bit-identical.

mod cleanup;
mod constraints;
mod curves;
mod quadric;
mod reduce;
mod region;
mod remesh;
mod stats;
mod util;

pub use cleanup::{collapse_short_edges, refine_uniform, CleanupReport};
pub use reduce::{reduce, ReduceParams, ReduceReport, ReduceTarget};
pub use remesh::{remesh, BoundaryMode, GroupBorderSmoothing, PassTimings, RemeshParams, RemeshReport};
pub use stats::{edge_length_stats, edge_length_stats_in, triangle_quality, EdgeLengthStats};

#[cfg(doc)]
use mm_core::Mesh;

/// Progress callback: receives a fraction in `[0, 1]`; returning `false` cancels the
/// operation. A cancelled operation stops at the next safe point and leaves a valid mesh.
pub type ProgressFn<'a> = &'a dyn Fn(f32) -> bool;

/// Errors for invalid parameters.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum RemeshError {
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),
}
