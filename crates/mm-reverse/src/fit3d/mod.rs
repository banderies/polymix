// SPDX-License-Identifier: AGPL-3.0-only
//! The 3D primitive fits of the measuring core: the plane (total least squares, exact), the
//! sphere (Coope's algebraic fit polished geometrically, optionally with its diameter held),
//! the cylinder (axis from the normals, the circle across it, damped Gauss–Newton), the cone
//! (the same with the half-angle), the torus (the spine through the points moved along their
//! normals) and the circle through picked points. Each curved shape has a robust start
//! (LMedS over sampled candidates, `ransac`) and a Gaussian best fit with GOM-style "used
//! points" clipping (`clip`), and reports σ, the form error and the residual extremes the
//! same way.
//!
//! Positions and normals come as slices of `DVec3` indexed by vertex number (as
//! `mm_core::Mesh::positions` gives them), a patch or region as a slice of vertex numbers.
//! Where ScanRuler returns `null` the fits return `None`; where it throws a `FitError` with a
//! message for the user they return `Err(FitError)`.
//!
//! Ported from ScanRuler's `src/core/fit/` (`errors`, `linalg`, `ransac`, `clip`, `extents`,
//! `selection`, `seedOff`, `circle`, `plane`, `heightSurface`, `sphere`, `cylinder`, `cone`,
//! `torus`) and the types they use from `src/core/types.ts`, © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only. `circle2d` is `crate::fit2d::fit_circle_2d`, `FitError` and the seeded
//! generator `crate::fit2d::{FitError, Mulberry32}`.

// The fits test `!(x > limit)` throughout, as ScanRuler does, so that a NaN fails the test
// it would otherwise slip through; the negation is the point.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

pub mod circle;
pub mod clip;
pub mod cone;
pub mod cylinder;
pub mod extents;
pub mod fit_cone_from_seed;
pub mod fit_cylinder_from_seed;
pub mod fit_plane_from_seed;
pub mod fit_sphere_from_seed;
pub mod fit_torus_from_seed;
pub mod guide;
pub mod height_surface;
pub(crate) mod js_text;
pub mod linalg3;
pub mod plane;
pub mod ransac;
pub mod region_grow;
pub mod seed_off;
pub mod selection;
pub mod sphere;
pub mod torus;
pub mod types;

#[cfg(test)]
pub(crate) mod test_meshes;
#[cfg(test)]
pub(crate) mod testkit;

pub use circle::circle_from_points;
pub use clip::{clipped_refit, ClippedRefit};
pub use cone::{cone_axis_from_normals, cone_normal_align, cone_normal_signed, cone_residual, fit_cone_clipped, fit_cone_on_axis, ransac_cone, ClippedConeFit, RansacConeResult};
pub use cylinder::{
    axis_from_normals, cylinder_residual, fit_cylinder_clipped, fit_cylinder_on_axis, ransac_cylinder, refine_cylinder_geometric, split_two_cylinders, ClippedCylinderFit,
    RansacCylinderResult, TwoCylinderSplit,
};
pub use extents::{axial_extents, AxialExtents};
pub use fit_cone_from_seed::{fit_cone_from_seed, fit_cone_on_selection};
pub use fit_cylinder_from_seed::{fit_cylinder_from_seed, fit_cylinder_on_selection};
pub use fit_plane_from_seed::{fit_plane_from_seed, fit_plane_on_selection};
pub use fit_sphere_from_seed::{fit_sphere_from_seed, fit_sphere_on_selection};
pub use fit_torus_from_seed::{fit_torus_from_seed, fit_torus_on_selection, TORUS_MAX_R_DIAGS};
pub use guide::{
    axis_agrees, axis_off, guide_patch, guide_refusal, guided_estimate, guided_seeds, has_guide, in_box, judge_round, keep_where, radial_from, radial_off, radius_in_range, side_share,
    unit_axis, FitGuide, GuideBox, GuideSeen, Inliers, RadiusRange, Side, GUIDED_TRIES,
};
pub use height_surface::{fit_height, fit_surface_models, patch_frame, CurvatureShape, SurfaceCurvature, SurfaceFitResult, SurfaceFrame, SurfaceModel, SurfaceModels, SURFACE_MODELS};
pub use linalg3::{centroid_of, extent_of_indexed, normalize_or_z, project_onto_plane, rms_over, symmetric_eigen3, Eigen3, Extent};
pub use plane::{curved_beyond_plane, fit_plane_clipped, fit_plane_tls, plane_residual, ransac_plane, split_two_planes, ClippedPlaneFit, RansacPlaneResult, TwoPlaneSplit};
pub use region_grow::{
    collect_patch, flood_by_normal, grow_cone_region, grow_cylinder_region, grow_plane_region, grow_sphere_region, grow_torus_region, seeded_fit, stranded_seeds, GrowResult,
    SEED_PATCHES,
};
pub use ransac::{ransac_consensus, ransac_sphere, sphere_residual, RansacConsensus, RansacOptions, RansacResult, RansacShape};
pub use seed_off::{seed_off, SeedOff};
pub use selection::{require_selection, MIN_SELECTED};
pub use sphere::{fit_sphere_clipped, ClippedSphereFit};
pub use torus::{
    canonical_torus, fit_torus_clipped, ransac_torus, refine_torus_geometric, torus_coverage, torus_from_shift, torus_normal_align, torus_residual, TorusClippedFit, TorusCoverage,
    TorusRansac,
};
pub use types::{
    AxialWindow, CircleFit, Cone, ConeFit, Cylinder, CylinderFit, ElementKind, FitBase, FitData, FitMethodId, FitOutput, FitSettings, FitTold, HeldDiameter, LineFit, Plane, PlaneFit,
    PointFit, RegionFit, SecondSurface, Sphere, SphereFit, Torus, TorusFit,
};
