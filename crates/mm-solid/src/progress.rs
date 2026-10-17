//! Progress reporting / cancellation helper.

use crate::{Progress, Result, SolidError};

/// Maps a sub-task's local progress `[0, 1]` onto a slice `[lo, hi]` of the caller's range.
#[derive(Clone, Copy)]
pub(crate) struct Prog<'a> {
    cb: Progress<'a>,
    lo: f32,
    hi: f32,
}

impl<'a> Prog<'a> {
    pub fn new(cb: Progress<'a>) -> Self {
        Self { cb, lo: 0.0, hi: 1.0 }
    }
    pub fn none() -> Self {
        Self { cb: None, lo: 0.0, hi: 1.0 }
    }
    /// Sub-range `[a, b]` (fractions of this range).
    pub fn sub(&self, a: f32, b: f32) -> Prog<'a> {
        let w = self.hi - self.lo;
        Prog { cb: self.cb, lo: self.lo + w * a, hi: self.lo + w * b }
    }
    /// Report local progress `t`; returns `Err(Cancelled)` if the callback asks to stop.
    pub fn report(&self, t: f32) -> Result<()> {
        if let Some(f) = self.cb {
            let v = self.lo + (self.hi - self.lo) * t.clamp(0.0, 1.0);
            if !f(v) {
                return Err(SolidError::Cancelled);
            }
        }
        Ok(())
    }
}
