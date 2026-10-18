//! Volumetric image data for Polymix: find, read and organise DICOM, NIfTI, Analyze and
//! NRRD files. No GUI; the Volume workspace (`polymix-volume`) shows what this finds.

pub mod geometry;
pub mod model;
pub mod nifti;
pub mod nrrd;
pub mod dicom;
pub mod organize;
pub mod scan;
pub mod slice;
