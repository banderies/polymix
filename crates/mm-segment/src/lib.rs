//! Segmentation of volumes for Polymix (no GUI): masks over a volume's voxel grid, the
//! operations that make and refine them (threshold, region growing, islands, margins,
//! smoothing, hole filling), Gaussian and joint smoothing, hollowing, segment statistics,
//! automatic thresholds, seeded segmentation (grow from seeds, watershed, fast marching),
//! filling between slices, shapes drawn on slices, level sets and guided vessel
//! segmentation, centerlines and cross-sections of a vessel, closed surface meshes of a mask
//! for the Mesh workspace, and label-map files (NIfTI, 3D Slicer `.seg.nrrd`).

pub mod autothreshold;
pub mod centerline;
pub mod grow;
pub mod interpolate;
pub mod io;
pub mod levelset;
pub mod mask;
pub mod mesh;
pub mod ops;
#[cfg(test)]
mod reference;
mod region;
pub mod shapes;
pub mod smooth;
pub mod stats;

pub use autothreshold::{auto_threshold, auto_threshold_values, AutoThreshold};
pub use centerline::{centerlines, cross_sections, stenosis, Branch, CenterlineParams, CenterlinePoint, Centerlines, CrossSection};
pub use grow::{fast_march, grow_from_seeds, watershed};
pub use interpolate::fill_between_slices;
pub use io::{write_nifti, write_seg_nrrd, SegmentInfo};
pub use levelset::{guided_vessel, level_set, level_set_with, FeatureMap, LevelSetParams};
pub use mask::{Connectivity, Mask};
pub use mesh::{mask_to_mesh, MeshParams};
pub use smooth::{gaussian, hollow, joint_smooth, HollowSide};
pub use stats::{segment_stats, SegmentStats};
