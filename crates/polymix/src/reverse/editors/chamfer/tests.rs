// SPDX-License-Identifier: AGPL-3.0-only
//! The Chamfer box's numbers for its modes — the case of ScanRuler's `reChamferDelete.test.ts`
//! ("keeps a chamfer's second number for its mode alone, and flips it only with one"), ©
//! Stefan Hermann (CNC Kitchen), AGPL-3.0-only.

use mm_reverse::model::{BodyId, ChamferFeature, ChamferMode, Exprs};

use super::super::rnd::{forget, with_draft};
use super::{reopened, set_mode, write_mode};

#[test]
fn keeps_a_chamfers_second_number_for_its_mode_alone_and_flips_it_only_with_one() {
    forget();
    let mut x = ChamferFeature { body: BodyId::new(2), edges: vec![], edges_where: vec![], distance: 1.0, chain: true, skip_failing: false, mode: None, distance2: None, angle_deg: None, flip: false, exprs: Exprs::new() };
    assert_eq!(with_draft(|d| (d.distance2, d.angle_deg, d.flip)), (1.0, 45.0, false));
    // Equal: no mode, no flip, no second distance, whatever the box holds.
    with_draft(|d| {
        d.flip = true;
        d.distance2 = 3.0;
    });
    write_mode(&mut x);
    assert_eq!((x.mode, x.flip, x.distance2, x.angle_deg), (None, false, None, None));
    set_mode(&mut x, ChamferMode::TwoDistances);
    assert_eq!((x.mode, x.distance2, x.flip, x.angle_deg), (Some(ChamferMode::TwoDistances), Some(3.0), true, None));
    with_draft(|d| {
        d.flip = false;
        d.angle_deg = 30.0;
    });
    set_mode(&mut x, ChamferMode::DistanceAngle);
    assert_eq!((x.mode, x.angle_deg, x.distance2), (Some(ChamferMode::DistanceAngle), Some(30.0), None));
    // Reopened as it was saved.
    let saved = ChamferFeature { mode: Some(ChamferMode::TwoDistances), distance2: Some(3.0), angle_deg: None, flip: true, ..x.clone() };
    assert_eq!(reopened(&saved), (3.0, 45.0, true));
    let plain = ChamferFeature { mode: None, distance2: None, angle_deg: None, flip: false, distance: 2.0, ..x };
    assert_eq!(reopened(&plain), (2.0, 45.0, false));
    forget();
}
