// SPDX-License-Identifier: AGPL-3.0-only
//! The round number a measured value was probably designed at. Ported from ScanRuler
//! (`plugins/reverse/core/nominal.ts`), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! A part is drawn at 6 mm and scanned at 5.98; a fillet is R2, not R1.97; a face is turned
//! 45°, not 44.8°. The sketch and the feature editors offer the rounding beside every
//! measured figure — one click to take it — and this says what that rounding is, and when the
//! value already is one so no chip is shown. The steps are the designer's, not the
//! metrologist's: half millimetres under 10 mm where small features are drawn to the half,
//! whole millimetres above, radii to the half whatever their size, angles to five degrees —
//! but a draft to the half degree (0.5, 1, 1.5, 2, 3°).

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NominalKind {
    Length,
    Radius,
    Angle,
    Draft,
}

/// How close a value has to be to its rounding to count as already nominal, in the value's
/// own unit — a hair under the tolerance of a good scan, so a value that was typed is not
/// offered to itself.
const NOMINAL_TOLERANCE: f64 = 0.02;

/// `Math.round` as JavaScript has it: halves round up, towards +∞.
fn js_round(x: f64) -> f64 {
    let f = x.floor();
    if x - f >= 0.5 {
        f + 1.0
    } else {
        f
    }
}

/// The step a value of this kind rounds to.
pub fn nominal_step(value: f64, kind: NominalKind) -> f64 {
    match kind {
        NominalKind::Angle => 5.0,
        NominalKind::Radius | NominalKind::Draft => 0.5,
        NominalKind::Length => {
            if value.abs() < 10.0 {
                0.5
            } else {
                1.0
            }
        }
    }
}

/// The nominal value nearest to `value`, or `None` when the value already is nominal (within
/// the tolerance) or is not a finite number. The sign is kept: an extrude reaching to −9.98 is
/// offered −10. Zero is nominal.
pub fn nominal_of(value: f64, kind: NominalKind) -> Option<f64> {
    if !value.is_finite() {
        return None;
    }
    let step = nominal_step(value, kind);
    let rounded = js_round(value / step) * step;
    // Away from a step boundary a length under 10 mm rounds to the half, but 9.8 mm rounds up
    // to 10 — a whole — and that is the number offered.
    let tidy = js_round(rounded * 1000.0) / 1000.0;
    // A hair of slack on the tolerance itself: 5.98 is 0.02 off 6 and is offered it, whatever
    // floating point makes of the subtraction.
    if (tidy - value).abs() < NOMINAL_TOLERANCE - 1e-9 {
        return None;
    }
    Some(if tidy == 0.0 { 0.0 } else { tidy })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offers_the_designers_round_figure_and_none_for_one_already_round() {
        assert_eq!(nominal_of(5.98, NominalKind::Length), Some(6.0));
        assert_eq!(nominal_of(9.8, NominalKind::Length), Some(10.0));
        assert_eq!(nominal_of(-9.98, NominalKind::Length), Some(-10.0));
        assert_eq!(nominal_of(6.0, NominalKind::Length), None);
        assert_eq!(nominal_of(1.97, NominalKind::Radius), Some(2.0));
        assert_eq!(nominal_of(44.8, NominalKind::Angle), Some(45.0));
        assert_eq!(nominal_of(f64::NAN, NominalKind::Angle), None);
    }

    // reNominal.test.ts, but for the chip's text (nominalText is the UI's and is not ported).

    #[test]
    fn rounds_lengths_to_the_half_millimetre_under_10_mm_and_the_millimetre_above() {
        assert_eq!(nominal_of(5.98, NominalKind::Length), Some(6.0));
        assert_eq!(nominal_of(6.27, NominalKind::Length), Some(6.5));
        assert_eq!(nominal_of(19.62, NominalKind::Length), Some(20.0));
        assert_eq!(nominal_of(20.4, NominalKind::Length), Some(20.0));
        assert_eq!(nominal_step(9.9, NominalKind::Length), 0.5);
        assert_eq!(nominal_step(10.1, NominalKind::Length), 1.0);
    }

    #[test]
    fn rounds_radii_to_the_half_whatever_their_size_and_angles_to_five_degrees() {
        assert_eq!(nominal_of(1.97, NominalKind::Radius), Some(2.0));
        assert_eq!(nominal_of(12.3, NominalKind::Radius), Some(12.5));
        assert_eq!(nominal_of(44.8, NominalKind::Angle), Some(45.0));
        assert_eq!(nominal_of(91.2, NominalKind::Angle), Some(90.0));
    }

    #[test]
    fn keeps_the_sign_an_extrude_reaching_behind_its_plane() {
        assert_eq!(nominal_of(-9.98, NominalKind::Length), Some(-10.0));
        assert_eq!(nominal_of(-0.3, NominalKind::Length), Some(-0.5));
    }

    #[test]
    fn offers_nothing_for_a_value_that_already_is_nominal() {
        assert_eq!(nominal_of(6.0, NominalKind::Length), None);
        assert_eq!(nominal_of(6.01, NominalKind::Length), None);
        assert_eq!(nominal_of(0.0, NominalKind::Length), None);
        assert_eq!(nominal_of(2.5, NominalKind::Radius), None);
        assert_eq!(nominal_of(90.0, NominalKind::Angle), None);
        assert_eq!(nominal_of(f64::NAN, NominalKind::Length), None);
    }

    #[test]
    fn a_value_just_past_the_tolerance_is_offered() {
        assert_eq!(nominal_of(6.03, NominalKind::Length), Some(6.0));
    }
}
