// SPDX-License-Identifier: AGPL-3.0-only
//! Heat-set threaded inserts, as the Hole feature sizes a hole for one: the CNC Kitchen
//! range, by the dimensions its tables give — the hole's diameter (D3), the insert's length
//! and the least wall (W) to leave round it. The hole is made a millimetre deeper than the
//! insert is long, for the plastic it pushes ahead of it. Ported from ScanRuler
//! (`plugins/reverse/core/inserts.ts`), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.

use serde::Serialize;

use crate::round::js_round;

/// An insert of the range.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct InsertSize {
    /// As the range names it: "M3", "M3 short".
    pub label: &'static str,
    /// The insert's length, mm.
    pub length: f64,
    /// The hole it is pressed into, mm.
    pub hole: f64,
    /// The least wall round the hole, mm.
    pub wall: f64,
}

pub const INSERT_SIZES: [InsertSize; 12] = [
    InsertSize { label: "M2", length: 3.0, hole: 3.2, wall: 1.3 },
    InsertSize { label: "M2.5", length: 4.0, hole: 4.0, wall: 1.6 },
    InsertSize { label: "M3", length: 5.7, hole: 4.0, wall: 1.6 },
    InsertSize { label: "M3 short", length: 3.0, hole: 4.0, wall: 1.6 },
    InsertSize { label: "M3 Voron", length: 4.0, hole: 4.4, wall: 1.3 },
    InsertSize { label: "M4", length: 8.1, hole: 5.6, wall: 2.1 },
    InsertSize { label: "M4 short", length: 4.0, hole: 5.6, wall: 2.1 },
    InsertSize { label: "M5", length: 9.5, hole: 6.4, wall: 2.6 },
    InsertSize { label: "M5 short", length: 5.8, hole: 6.4, wall: 2.6 },
    InsertSize { label: "M6", length: 12.7, hole: 8.0, wall: 3.3 },
    InsertSize { label: "M8", length: 12.7, hole: 9.7, wall: 3.3 },
    InsertSize { label: "1/4\"-20", length: 12.7, hole: 8.0, wall: 3.3 },
];

/// How much deeper than the insert the hole is made, mm.
const INSERT_EXTRA_DEPTH_MM: f64 = 1.0;

/// An insert by its label, case and spaces aside.
pub fn insert_size(label: &str) -> Option<&'static InsertSize> {
    let want = label.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase();
    INSERT_SIZES.iter().find(|s| s.label.to_lowercase() == want)
}

/// The hole an insert takes.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct InsertHole {
    pub radius: f64,
    pub depth: f64,
}

/// The hole an insert takes: its radius and its depth.
pub fn insert_hole(size: &InsertSize) -> InsertHole {
    InsertHole { radius: size.hole / 2.0, depth: js_round((size.length + INSERT_EXTRA_DEPTH_MM) * 100.0) / 100.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// reBodyTools.test.ts, "a hole drilled into a flat face": the insert's numbers (the
    /// drilling itself is the kernel's).
    #[test]
    fn sizes_a_hole_as_deep_as_an_insert_asks() {
        let m3 = insert_size("m3").map(insert_hole);
        assert_eq!(m3, Some(InsertHole { radius: 2.0, depth: 6.7 }));
    }

    #[test]
    fn finds_an_insert_by_its_label_case_and_spaces_aside() {
        assert_eq!(insert_size("  m3   SHORT ").map(|s| s.label), Some("M3 short"));
        assert_eq!(insert_size("1/4\"-20").map(|s| s.hole), Some(8.0));
        assert_eq!(insert_size("M7"), None);
        // Every label finds itself.
        for s in &INSERT_SIZES {
            assert_eq!(insert_size(s.label), Some(s));
        }
    }
}
