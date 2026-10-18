//! Volumetric image data for Polymix: find, read and organise DICOM, NIfTI, Analyze, NRRD,
//! MetaImage and MGH files, and stacks of 2D images. No GUI; the Volume workspace (`polymix-volume`) shows what this finds.

pub mod geometry;
pub mod model;
pub mod nifti;
pub mod nrrd;
pub mod metaimage;
pub mod mgh;
pub mod stack;
pub mod dicom;
pub mod jpegls;
pub mod organize;
pub mod scan;
pub mod slice;
pub mod load;
pub mod write;
