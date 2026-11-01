// SPDX-License-Identifier: AGPL-3.0-only
//! The units lengths are shown and typed in. Ported from ScanRuler (`src/core/units.ts`),
//! © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! Everything the workspace holds — stores, projects, every reading an agent is given — is in
//! millimetres and stays so; this is only the last step to the screen and the first from the
//! keyboard. Converting what is held would drift with every rounding and break every saved
//! project, so nothing here is ever written back except through [`from_display`], on a
//! number someone typed.
//!
//! Inches are decimal: no fractions, feet or mils. A site that asks for n decimals asks in
//! millimetres; in inches it gets one more, and never fewer than three, so 0.001 mm reads
//! 0.0001 in and 0.1 mm reads 0.004 in.

use serde::{Deserialize, Serialize};

use crate::js_number::{js_string, to_fixed, to_precision_number};

/// The units lengths are shown in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DisplayUnits {
    #[default]
    Mm,
    In,
}

impl DisplayUnits {
    /// The unit's id as stored: "mm" or "in".
    pub fn id(self) -> &'static str {
        match self {
            DisplayUnits::Mm => "mm",
            DisplayUnits::In => "in",
        }
    }

    /// The unit as a menu lists it.
    pub fn label(self) -> &'static str {
        match self {
            DisplayUnits::Mm => "Millimetres",
            DisplayUnits::In => "Inches",
        }
    }
}

/// The units on offer, in the order a menu lists them.
pub const DISPLAY_UNITS: [DisplayUnits; 2] = [DisplayUnits::Mm, DisplayUnits::In];

pub const MM_PER_INCH: f64 = 25.4;

/// A stored or received value as display units, or none when it is not one.
pub fn display_units_of(value: &str) -> Option<DisplayUnits> {
    match value {
        "mm" => Some(DisplayUnits::Mm),
        "in" => Some(DisplayUnits::In),
        _ => None,
    }
}

/// A length held in millimetres, in the units shown.
pub fn to_display(mm: f64, units: DisplayUnits) -> f64 {
    match units {
        DisplayUnits::In => mm / MM_PER_INCH,
        DisplayUnits::Mm => mm,
    }
}

/// A length typed in the units shown, in millimetres to hold.
pub fn from_display(v: f64, units: DisplayUnits) -> f64 {
    match units {
        DisplayUnits::In => v * MM_PER_INCH,
        DisplayUnits::Mm => v,
    }
}

/// The decimals a site asked for in millimetres, as shown in `units`.
pub fn length_decimals(mm_decimals: usize, units: DisplayUnits) -> usize {
    match units {
        DisplayUnits::In => (mm_decimals + 1).max(3),
        DisplayUnits::Mm => mm_decimals,
    }
}

/// The unit's own word, to a power for areas and volumes (`power` 1, 2 or 3).
pub fn unit_word(units: DisplayUnits, power: u8) -> String {
    let sup = match power {
        2 => "²",
        3 => "³",
        _ => "",
    };
    format!("{}{}", units.id(), sup)
}

/// How a figure is written beside its unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FmtOptions {
    /// Leave the unit word off, for a column that carries it in its header.
    pub bare: bool,
    /// A plus on a positive figure, the way a deviation is read.
    pub signed: bool,
}

/// Fixed decimals without the "-0.000" a tiny negative rounds to.
fn fixed(v: f64, decimals: usize, signed: bool) -> String {
    let text = to_fixed(v, decimals);
    if text.parse::<f64>().map(|n| n == 0.0).unwrap_or(false) {
        return to_fixed(0.0, decimals);
    }
    if signed && v > 0.0 {
        format!("+{text}")
    } else {
        text
    }
}

fn fmt(value: f64, units: DisplayUnits, mm_decimals: usize, power: u8, opts: FmtOptions) -> String {
    let scale = match units {
        DisplayUnits::In => MM_PER_INCH.powi(i32::from(power)),
        DisplayUnits::Mm => 1.0,
    };
    let text = fixed(value / scale, length_decimals(mm_decimals, units), opts.signed);
    if opts.bare {
        text
    } else {
        format!("{} {}", text, unit_word(units, power))
    }
}

/// "12.700 mm" / "0.5000 in" (ScanRuler's default is three millimetre decimals).
pub fn fmt_len(mm: f64, units: DisplayUnits, mm_decimals: usize, opts: FmtOptions) -> String {
    fmt(mm, units, mm_decimals, 1, opts)
}

/// "645.2 mm²" / "1.000 in²" (ScanRuler's default is one millimetre decimal).
pub fn fmt_area(mm2: f64, units: DisplayUnits, mm_decimals: usize, opts: FmtOptions) -> String {
    fmt(mm2, units, mm_decimals, 2, opts)
}

/// "16387.1 mm³" / "1.000 in³" (ScanRuler's default is one millimetre decimal).
pub fn fmt_vol(mm3: f64, units: DisplayUnits, mm_decimals: usize, opts: FmtOptions) -> String {
    fmt(mm3, units, mm_decimals, 3, opts)
}

/// The steps in use in millimetres, each with a round inch neighbour — not the converted
/// figure, which no one would type.
const INCH_STEPS: [(f64, f64); 9] = [(0.0001, 0.00001), (0.001, 0.0001), (0.01, 0.001), (0.05, 0.005), (0.1, 0.005), (0.5, 0.02), (1.0, 0.05), (5.0, 0.25), (10.0, 0.5)];

/// A step or preset for a length field in the units shown: a round figure in that unit. One
/// not in the table becomes its inch figure to one significant digit.
pub fn length_step(mm_step: f64, units: DisplayUnits) -> f64 {
    if units == DisplayUnits::Mm || !(mm_step > 0.0) || !mm_step.is_finite() {
        return mm_step;
    }
    if let Some(&(_, inch)) = INCH_STEPS.iter().find(|(mm, _)| *mm == mm_step) {
        return inch;
    }
    to_precision_number(mm_step / MM_PER_INCH, 1)
}

/// A length as the text of an input in `units`: rounded far enough below anything shown that
/// it reads back as typed, with no float noise ("0.49999999999999994"), and no trailing
/// zeros.
pub fn length_input_text(mm: f64, units: DisplayUnits) -> String {
    match units {
        DisplayUnits::Mm => js_string(mm),
        DisplayUnits::In => js_string(to_fixed(to_display(mm, units), 6).parse::<f64>().unwrap_or(f64::NAN)),
    }
}

/// A preset list for a length: one hand-picked list per unit, never one computed from the
/// other.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct LengthPresets {
    pub mm: Vec<f64>,
    #[serde(rename = "in")]
    pub inch: Vec<f64>,
}

/// The presets for `units`, as millimetres to hold.
pub fn presets_mm(presets: &LengthPresets, units: DisplayUnits) -> Vec<f64> {
    match units {
        DisplayUnits::In => presets.inch.iter().map(|v| v * MM_PER_INCH).collect(),
        DisplayUnits::Mm => presets.mm.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lengths_in_either_unit() {
        let plain = FmtOptions::default();
        assert_eq!(fmt_len(12.7, DisplayUnits::Mm, 3, plain), "12.700 mm");
        assert_eq!(fmt_len(12.7, DisplayUnits::In, 3, plain), "0.5000 in");
        assert_eq!(fmt_area(645.16, DisplayUnits::In, 1, plain), "1.000 in²");
        assert_eq!(fmt_vol(16387.064, DisplayUnits::Mm, 1, plain), "16387.1 mm³");
        assert_eq!(fmt_len(-0.0001, DisplayUnits::Mm, 3, plain), "0.000 mm");
        assert_eq!(fmt_len(0.25, DisplayUnits::Mm, 3, FmtOptions { bare: true, signed: true }), "+0.250");
        assert_eq!(length_decimals(1, DisplayUnits::In), 3);
        assert_eq!(length_decimals(3, DisplayUnits::In), 4);
    }

    // ---- ScanRuler's `tests/units.test.ts` (all but `lengthField`, which is the UI's) ------

    fn close(a: f64, b: f64, digits: i32) -> bool {
        // vitest's toBeCloseTo: |a − b| < 10^−digits / 2.
        (a - b).abs() < 10f64.powi(-digits) / 2.0
    }

    #[test]
    fn to_display_and_from_display_convert_and_round_trip() {
        assert_eq!(to_display(12.7, DisplayUnits::Mm), 12.7);
        assert_eq!(from_display(12.7, DisplayUnits::Mm), 12.7);
        assert_eq!(to_display(25.4, DisplayUnits::In), 1.0);
        assert_eq!(from_display(0.5, DisplayUnits::In), 12.7);
        assert_eq!(to_display(-50.8, DisplayUnits::In), -2.0);
        for mm in [0.0, 0.001, 0.1, 1.0, 3.175, 12.7, 99.99, -42.42, 12345.678] {
            assert!(close(from_display(to_display(mm, DisplayUnits::In), DisplayUnits::In), mm, 9), "{mm}");
        }
    }

    #[test]
    fn length_decimals_keep_millimetres_and_give_inches_one_more_at_least_three() {
        assert_eq!(length_decimals(0, DisplayUnits::Mm), 0);
        assert_eq!(length_decimals(3, DisplayUnits::Mm), 3);
        for (mm, inch) in [(0, 3), (1, 3), (2, 3), (3, 4), (4, 5)] {
            assert_eq!(length_decimals(mm, DisplayUnits::In), inch, "{mm}");
        }
    }

    #[test]
    fn fmt_len_as_scanruler_writes_it() {
        let plain = FmtOptions::default();
        let bare = FmtOptions { bare: true, signed: false };
        let signed = FmtOptions { bare: false, signed: true };
        assert_eq!(fmt_len(12.7, DisplayUnits::Mm, 3, plain), "12.700 mm");
        assert_eq!(fmt_len(12.7, DisplayUnits::In, 3, plain), "0.5000 in");
        assert_eq!(fmt_len(12.7, DisplayUnits::Mm, 1, plain), "12.7 mm");
        assert_eq!(fmt_len(12.7, DisplayUnits::In, 1, plain), "0.500 in");
        assert_eq!(fmt_len(25.4, DisplayUnits::In, 3, bare), "1.0000");
        assert_eq!(fmt_len(0.0, DisplayUnits::Mm, 3, plain), "0.000 mm");
        assert_eq!(fmt_len(0.0, DisplayUnits::In, 3, plain), "0.0000 in");
        assert_eq!(fmt_len(-0.0001, DisplayUnits::Mm, 3, plain), "0.000 mm");
        assert_eq!(fmt_len(-0.0001, DisplayUnits::In, 3, signed), "0.0000 in");
        assert_eq!(fmt_len(-1.5, DisplayUnits::Mm, 2, plain), "-1.50 mm");
        assert_eq!(fmt_len(-25.4, DisplayUnits::In, 2, plain), "-1.000 in");
        assert_eq!(fmt_len(0.05, DisplayUnits::Mm, 3, signed), "+0.050 mm");
        assert_eq!(fmt_len(-0.05, DisplayUnits::Mm, 3, signed), "-0.050 mm");
        assert_eq!(fmt_len(1234567.891, DisplayUnits::Mm, 2, plain), "1234567.89 mm");
        assert_eq!(fmt_len(25400.0, DisplayUnits::In, 0, plain), "1000.000 in");
    }

    #[test]
    fn fmt_area_and_fmt_vol_scale_by_the_square_and_the_cube() {
        let plain = FmtOptions::default();
        assert_eq!(fmt_area(MM_PER_INCH.powi(2), DisplayUnits::In, 1, plain), "1.000 in²");
        assert_eq!(fmt_area(100.0, DisplayUnits::Mm, 1, plain), "100.0 mm²");
        assert_eq!(fmt_vol(MM_PER_INCH.powi(3), DisplayUnits::In, 1, plain), "1.000 in³");
        assert_eq!(fmt_vol(1000.0, DisplayUnits::Mm, 0, plain), "1000 mm³");
        assert_eq!(fmt_vol(2.0 * MM_PER_INCH.powi(3), DisplayUnits::In, 2, FmtOptions { bare: true, signed: false }), "2.000");
        assert_eq!(unit_word(DisplayUnits::Mm, 1), "mm");
        assert_eq!(unit_word(DisplayUnits::In, 2), "in²");
        assert_eq!(unit_word(DisplayUnits::Mm, 3), "mm³");
    }

    #[test]
    fn length_step_gives_a_round_inch_neighbour_or_one_significant_figure() {
        assert_eq!(length_step(0.1, DisplayUnits::Mm), 0.1);
        for (mm, inch) in [(0.001, 0.0001), (0.01, 0.001), (0.05, 0.005), (0.1, 0.005), (1.0, 0.05), (5.0, 0.25)] {
            assert_eq!(length_step(mm, DisplayUnits::In), inch, "{mm}");
        }
        assert_eq!(length_step(2.0, DisplayUnits::In), 0.08);
        assert_eq!(length_step(25.0, DisplayUnits::In), 1.0);
        assert_eq!(length_step(0.0, DisplayUnits::In), 0.0);
    }

    #[test]
    fn length_input_text_presets_and_display_units_of() {
        assert_eq!(length_input_text(from_display(0.5, DisplayUnits::In), DisplayUnits::In), "0.5");
        assert_eq!(length_input_text(from_display(0.1234, DisplayUnits::In), DisplayUnits::In), "0.1234");
        assert_eq!(length_input_text(0.3, DisplayUnits::Mm), "0.3");
        let p = LengthPresets { mm: vec![0.1, 0.5], inch: vec![0.005, 0.02] };
        assert_eq!(presets_mm(&p, DisplayUnits::Mm), [0.1, 0.5]);
        assert_eq!(presets_mm(&p, DisplayUnits::In), [0.127, 0.508]);
        assert_eq!(display_units_of("in"), Some(DisplayUnits::In));
        assert_eq!(display_units_of("mm"), Some(DisplayUnits::Mm));
        assert_eq!(display_units_of("cm"), None);
    }

    #[test]
    fn steps_and_input_text() {
        assert_eq!(length_step(0.1, DisplayUnits::In), 0.005);
        assert_eq!(length_step(2.0, DisplayUnits::In), 0.08);
        assert_eq!(length_step(2.0, DisplayUnits::Mm), 2.0);
        assert_eq!(length_input_text(12.7, DisplayUnits::In), "0.5");
        assert_eq!(length_input_text(0.1 + 0.2, DisplayUnits::Mm), "0.30000000000000004");
        assert_eq!(display_units_of("in"), Some(DisplayUnits::In));
        assert_eq!(display_units_of("ft"), None);
    }
}
