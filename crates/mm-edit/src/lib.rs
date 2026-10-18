//! Modeling edits for Polymix (the Meshmixer "Edit" menu):
//!
//! * [`cut`] — plane cut (keep above / below / slice) with flat or remeshed caps, and the
//!   *local* plane cut that trims a single vessel branch, plus intersection-curve previews.
//! * [`vessel`] — estimate a vessel's local axis/cross-section from a click
//!   ([`vessel::suggest_cut_plane`]).
//! * [`flow`] — flow extensions: extrude a boundary loop straight out (optionally blending
//!   to a circular cross-section) and cap it.
//! * [`cap`] — planar capping of boundary loops, including nested loops (polygons with holes).
//! * [`extrude`] — extrude / offset a face selection, naive whole-surface offset.
//! * [`region`] — discard / separate / duplicate / flip a face selection.
//! * [`boolean`] — exact mesh booleans (union / difference / intersection).
//! * [`mod@mirror`], [`mod@bridge`] (bridge spans, join loops), [`smooth`] (smooth boundary),
//!   [`transform`] (affine transforms, drop to ground, principal-axis alignment).
//!
//! Every operation edits an [`mm_core::Mesh`] in place and keeps a manifold,
//! consistently oriented input manifold and consistently oriented.

mod triangulate;
mod util;

pub mod boolean;
pub mod bridge;
pub mod cap;
pub mod cut;
pub mod extrude;
pub mod face_groups;
pub mod fit;
pub mod flow;
pub mod mirror;
pub mod part;
pub mod region;
pub mod smooth;
pub mod soft;
pub mod stamp;
pub mod target;
pub mod transform;
pub mod tube;
pub mod vessel;

pub use boolean::{mesh_boolean, BooleanOp, BooleanResult};
pub use bridge::{bridge, join_loops, BridgeParams, BridgeResult};
pub use cap::{cap_planar_loops, CapResult};
pub use cut::{
    local_plane_cut, local_plane_cut_with, nearest_intersection_curve, plane_cut, plane_intersection_curves, CutFill,
    IntersectionCurve, KeepSide, LocalCutResult, PlaneCutParams, PlaneCutResult, Side,
};
pub use face_groups::{generate_face_groups, FaceGroupMode};
pub use fit::{convex_hull, fit_cylinder, fit_primitive, CylinderFit, FitResult, PrimitiveType};
pub use extrude::{
    extract_solid, extrude_selection, offset_selection, offset_selection_with, offset_surface, ExtrudeDirection, ExtrudeParams, ExtrudeResult,
};
pub use flow::{extrude_boundary_loop, ExtensionLength, FlowExtensionParams, FlowExtensionResult, Length};
pub use mirror::{mirror, mirror_with_tolerance, MirrorResult};
pub use part::{builtin_parts, drop_open_part, place_part, DropResult, Part, PartKind, Placement};
pub use region::{discard, duplicate, flip_normals, separate};
pub use smooth::smooth_boundary;
pub use soft::{soft_deform, soft_region, warp, Continuity, SoftKind, SoftRegion};
pub use stamp::{stamp_polygon, StampFrame, StampResult, Stencil};
pub use target::{attract_to_target, icp_align, project_to_target, AttractParams, IcpParams, IcpResult};
pub use tube::{add_tube, tube_mesh, tube_path, TubeEnd, TubeParams, TubePath};
pub use transform::{align_to_principal_axes, center_on_origin, drop_to_ground, transform_mesh};
pub use vessel::{estimate_vessel_cross_section, suggest_cut_plane, VesselCrossSection};

use mm_core::MeshError;

/// Errors reported by edit operations.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum EditError {
    /// No (live) triangles were selected.
    #[error("the selection is empty")]
    EmptySelection,
    /// The given vertices do not form a boundary loop / span.
    #[error("invalid boundary loop: {0}")]
    InvalidLoop(String),
    /// A parameter is out of range.
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),
    /// The plane misses the mesh.
    #[error("the plane does not intersect the mesh")]
    NoIntersection,
    /// A local cut curve that does not split the surface (e.g. around a handle).
    #[error("the chosen cut curve does not separate the surface into two pieces")]
    CurveDoesNotSeparate,
    /// A low-level mesh edit failed.
    #[error("mesh operation failed: {0}")]
    Mesh(#[from] MeshError),
    /// Any other failure.
    #[error("{0}")]
    Failed(String),
}

/// Result type of edit operations.
pub type EditResult<T> = Result<T, EditError>;

#[cfg(test)]
mod tests;
