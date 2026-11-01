// SPDX-License-Identifier: AGPL-3.0-only
//! The scan's own symmetry plane as the plane a Mirror starts on. Ported from ScanRuler
//! (`plugins/reverse/core/symmetryPlane.ts`, with the acceptance test and its thresholds from
//! `src/core/symmetry.ts`), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! Half of a symmetric part is modelled and mirrored, and the plane it is mirrored about
//! decides whether the two halves meet: a hand-picked face or a coordinate plane is off by
//! whatever the alignment was off by, while the plane found by mirror registration is the
//! part's own. So the Mirror box opens with that plane already in — when the scan has one.
//! This file is the judgement and the conversion; the search itself runs where the scan is.

use glam::DVec3;
use serde::{Deserialize, Serialize};

use crate::frame::SectionFrame;
use crate::model::PlaneRef;

/// The RMS of the matched mirror images, mm, above which a found plane is no symmetry worth
/// defaulting to.
pub const SYMMETRY_MAX_RMS_MM: f64 = 0.1;

/// The share of the samples that must find their mirror image. A part symmetric in its body
/// but for a hook or a boss still passes; a plane that mirrors half the part into thin air
/// does not.
pub const SYMMETRY_MIN_MATCHED: f64 = 0.5;

/// The name a plane taken from the search carries in rows and boxes.
pub const SYMMETRY_LABEL: &str = "the scan’s symmetry plane";

/// What the mirror-registration search found.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SymmetryPlane {
    pub normal: DVec3,
    pub point: DVec3,
    /// RMS distance of the matched mirrored samples from the scan, in mm — zero for a perfectly
    /// symmetric part, and the σ the plane reports.
    pub rms: f64,
    /// How many samples found their mirror image on the scan, of how many.
    pub matched: usize,
    pub sampled: usize,
    /// Which candidate won: 0–2 a principal plane, 3 and up one of the directions the caller
    /// added, −1 a given seed.
    pub candidate: i32,
}

/// Whether a search's answer is a symmetry worth defaulting to.
pub fn symmetry_accepted(r: &SymmetryPlane) -> bool {
    r.rms.is_finite() && r.rms <= SYMMETRY_MAX_RMS_MM && r.sampled > 0 && r.matched as f64 / r.sampled as f64 >= SYMMETRY_MIN_MATCHED
}

/// The found plane as a reference a feature can stand on: a fitted plane, belonging to no
/// element, moving with the part — its sheet laid as a sketch's is, zeroed under the origin —
/// labelled `SYMMETRY_LABEL`. `None` when the search did not find a symmetry.
pub fn symmetry_plane_ref(r: &SymmetryPlane) -> Option<PlaneRef> {
    if !symmetry_accepted(r) {
        return None;
    }
    let frame = SectionFrame::sketch_on(r.point, r.normal, None)?;
    Some(PlaneRef::Fitted { frame, label: Some(SYMMETRY_LABEL.to_string()) })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn found() -> SymmetryPlane {
        SymmetryPlane { normal: DVec3::new(0.0, 0.0, 1.0), point: DVec3::new(1.0, 2.0, 3.0), rms: 0.02, matched: 3600, sampled: 4000, candidate: 0 }
    }

    #[test]
    fn takes_a_tight_mirror_image_over_most_of_the_samples() {
        assert!(symmetry_accepted(&found()));
        let r = symmetry_plane_ref(&found()).expect("a plane");
        let PlaneRef::Fitted { frame, label } = r else { panic!("a fitted plane") };
        // The plane through the found point, its sheet zeroed under the origin as a sketch's
        // is.
        assert_eq!(frame.origin + DVec3::ZERO, DVec3::new(0.0, 0.0, 3.0));
        assert_eq!(frame.normal, DVec3::new(0.0, 0.0, 1.0));
        assert_eq!(label.as_deref(), Some(SYMMETRY_LABEL));
    }

    #[test]
    fn refuses_a_loose_one_and_one_that_mirrors_half_the_part_into_thin_air() {
        assert_eq!(symmetry_plane_ref(&SymmetryPlane { rms: 0.4, ..found() }), None);
        assert_eq!(symmetry_plane_ref(&SymmetryPlane { matched: 1500, ..found() }), None);
        assert_eq!(symmetry_plane_ref(&SymmetryPlane { rms: f64::INFINITY, matched: 0, ..found() }), None);
    }
}
