//! Segmentation of volumes for Polymix (no GUI): masks over a volume's voxel grid, the
//! operations that make and refine them (threshold, region growing, islands, margins,
//! smoothing, hole filling), closed surface meshes of a mask for the Mesh workspace, and
//! label-map files (NIfTI, 3D Slicer `.seg.nrrd`).

pub mod io;
pub mod mask;
pub mod mesh;
pub mod ops;

pub use io::{write_nifti, write_seg_nrrd, SegmentInfo};
pub use mask::{Connectivity, Mask};
pub use mesh::{mask_to_mesh, MeshParams};
