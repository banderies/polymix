//! Hole filling, smoothing/fairing and automatic mesh repair — the Meshmixer *Inspector*
//! and *Smooth* tools for [`mm_core::Mesh`].
//!
//! * [`fill`] — [`fill_hole`] with [`FillMode::Minimal`] / [`FillMode::Flat`] /
//!   [`FillMode::Smooth`] (after Liepa, "Filling Holes in Meshes", SGP 2003).
//! * [`smooth`] — explicit Laplacian / Taubin smoothing ([`smooth_vertices`]) and implicit
//!   membrane / thin-plate fairing with fixed surroundings ([`fair_region`]).
//! * [`mod@inspect`] — defect analysis ([`inspect()`]), [`auto_repair`] and [`repair_single`].
//! * [`ops`] — the individual repair operations (crack welding, degenerate-triangle
//!   removal, bowtie splitting, small-component removal).
//! * [`sparse`] — CSR matrices and a Jacobi-preconditioned conjugate gradient solver.
//! * [`laplacian`] — Laplacian weight schemes.
//!
//! All edits keep the mesh valid ([`mm_core::Mesh::check_validity`]); operations that can fail are
//! atomic, and cancelling through a progress callback leaves a valid mesh.

#![warn(missing_docs)]

pub mod fill;
pub mod inspect;
pub mod laplacian;
pub mod ops;
pub mod smooth;
pub mod sparse;

mod refine;
mod util;

pub use fill::{fill_hole, FillMode, FillParams, FillResult};
pub use inspect::{
    auto_repair, inspect, repair_single, Defect, HoleInfo, InspectParams, InspectionReport, RepairParams, RepairReport,
    SingleRepair,
};
pub use laplacian::LaplacianWeights;
pub use ops::{close_cracks, close_cracks_in, CloseCracksReport, ComponentInfo};
pub use smooth::{
    fair_region, fair_region_with, smooth_vertices, smooth_vertices_with_progress, FairParams, SmoothMethod, SmoothParams,
};
pub use sparse::{CsrMatrix, PcgParams, PcgStats};

/// Progress callback: receives a fraction in `[0, 1]`; returning `false` requests
/// cancellation.
pub type Progress<'a> = Option<&'a dyn Fn(f32) -> bool>;

/// Errors of the repair operations.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum RepairError {
    /// The loop is not (or no longer) a closed chain of boundary edges of the mesh.
    #[error("invalid boundary loop: {0}")]
    InvalidLoop(String),
    /// The loop is the boundary of a lone triangle; filling it would duplicate it.
    #[error("the loop bounds an isolated triangle (filling it would duplicate the triangle)")]
    IsolatedTriangle,
    /// A parameter is out of range.
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),
    /// A local mesh edit was refused by `mm-core`.
    #[error("mesh edit failed: {0}")]
    Mesh(#[from] mm_core::MeshError),
    /// The linear solver broke down or produced non-finite values.
    #[error("linear solve failed: {0}")]
    Solver(String),
    /// The progress callback requested cancellation.
    #[error("operation cancelled")]
    Cancelled,
    /// The defect passed to [`repair_single`] is not present any more.
    #[error("defect no longer exists: {0}")]
    StaleDefect(String),
}

/// Result type of this crate.
pub type Result<T> = std::result::Result<T, RepairError>;
