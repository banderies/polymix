//! Image preparation for Polymix (no GUI): image → image operations done before segmenting,
//! starting with N4 bias field correction.

mod fft;
pub mod n4;

pub use n4::{auto_mask, n4, N4Output, N4Params};
