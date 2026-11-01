// SPDX-License-Identifier: AGPL-3.0-only
//! The units lengths are shown and typed in — the part of them a thread's messages need.
//! Ported from ScanRuler (`src/core/units.ts`), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only.
//!
//! Everything the instrument holds — stores, projects, every reading an agent is given — is
//! in millimetres and stays so; this is only the last step to the screen and the first from
//! the keyboard. Converting what is held would drift with every rounding and break every
//! saved project, so nothing here is ever written back except through [`from_display`], on a
//! number someone typed.
//!
//! Inches are decimal: no fractions, feet or mils. A site that asks for n decimals asks in
//! millimetres; in inches it gets one more, and never fewer than three, so 0.001 mm reads
//! 0.0001 in and 0.1 mm reads 0.004 in.

use serde::{Deserialize, Serialize};

/// The units lengths are shown in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DisplayUnits {
    #[default]
    Mm,
    In,
}

impl DisplayUnits {
    /// Both, as the settings offer them.
    pub const ALL: [DisplayUnits; 2] = [DisplayUnits::Mm, DisplayUnits::In];

    /// What the settings call it.
    pub fn label(self) -> &'static str {
        match self {
            DisplayUnits::Mm => "Millimetres",
            DisplayUnits::In => "Inches",
        }
    }
}

pub const MM_PER_INCH: f64 = 25.4;

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
pub fn length_decimals(mm_decimals: i32, units: DisplayUnits) -> i32 {
    match units {
        DisplayUnits::In => (mm_decimals + 1).max(3),
        DisplayUnits::Mm => mm_decimals,
    }
}

/// The unit's own word, to a power for areas and volumes.
pub fn unit_word(units: DisplayUnits, power: u8) -> String {
    let word = match units {
        DisplayUnits::Mm => "mm",
        DisplayUnits::In => "in",
    };
    match power {
        2 => format!("{word}²"),
        3 => format!("{word}³"),
        _ => word.to_string(),
    }
}
