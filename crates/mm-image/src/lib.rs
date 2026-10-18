//! Image preparation for Polymix (no GUI): image → image operations done before segmenting.
//! Images are `LoadedVolume`s; see [`image`] for the conventions every operation follows.

mod fft;
pub mod filter;
pub mod image;
pub mod n4;
pub mod register;
pub mod resample;

pub use image::{no_progress, Progress};
pub use n4::{auto_mask, n4, N4Output, N4Params};
pub use register::{register, write_itk_transform, Metric, Motion, RegisterParams, Registration};
pub use resample::{resample_onto, resample_to_spacing, Interpolation, Transform};
