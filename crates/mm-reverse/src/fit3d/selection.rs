// SPDX-License-Identifier: AGPL-3.0-only
//! The hand-painted counterpart of the seed-and-grow pipeline: the user has already marked
//! which surface belongs to the element, so these fits take the selection exactly as given —
//! no local patch, no RANSAC hunt for the right surface, no region growing that could leak
//! somewhere else or stop short. Everything downstream (the clipped Gaussian best fit, the
//! reported sigma) stays the same, so a painted element and an auto-fitted one are the same
//! measurement.
//!
//! Ported from ScanRuler (`src/core/fit/selection.ts`), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only.

use crate::fit2d::FitError;

/// Below this a fit has nothing to work with — a handful of points can be matched exactly
/// by almost any surface.
pub const MIN_SELECTED: usize = 20;

/// Refuses a selection too small to fit, saying what to paint. `noun` names the surface
/// ("cylindrical surface", "sphere", …).
pub fn require_selection(selection: &[u32], noun: &str) -> Result<(), FitError> {
    if selection.len() < MIN_SELECTED {
        return Err(FitError::new(format!(
            "Paint more of the {noun} — a fit needs at least {MIN_SELECTED} marked points, and this selection has {}.",
            selection.len()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_what_to_paint_and_how_much() {
        let err = require_selection(&[1, 2, 3], "sphere").unwrap_err();
        assert_eq!(err.0, "Paint more of the sphere — a fit needs at least 20 marked points, and this selection has 3.");
        assert!(require_selection(&(0..20).collect::<Vec<u32>>(), "sphere").is_ok());
    }
}
