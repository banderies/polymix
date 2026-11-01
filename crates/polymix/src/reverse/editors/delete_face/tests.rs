// SPDX-License-Identifier: AGPL-3.0-only
//! What a click on a face takes for a delete, and a second click letting all of it go — the
//! case of ScanRuler's `reChamferDelete.test.ts` ("the boxes at the store"), © Stefan Hermann
//! (CNC Kitchen), AGPL-3.0-only.

use glam::DVec3;
use mm_reverse::model::FaceRef;
use mm_reverse::topo::{FaceFingerprint, FaceKind, MatchTolerance};

use super::super::rnd::toggle_group;

fn face(hash: f64, kind: FaceKind) -> FaceRef {
    FaceFingerprint { kind, centroid: DVec3::new(0.0, 0.0, hash), dir: DVec3::Z, area: 10.0, radius: None, center: None, origin: 2 }
}

const TOL: MatchTolerance = MatchTolerance { position: 1e-4, angle: 1e-3, size: 1e-3 };

#[test]
fn takes_what_a_click_on_a_face_takes_and_lets_all_of_it_go_on_a_second_click() {
    let (c, p) = (FaceKind::Cylinder, FaceKind::Plane);
    let mut faces: Vec<FaceRef> = Vec::new();
    toggle_group(&mut faces, &face(5.0, c), vec![face(5.0, c), face(6.0, c), face(7.0, p)], &TOL);
    assert_eq!(faces.iter().map(|f| f.centroid.z).collect::<Vec<_>>(), vec![5.0, 6.0, 7.0]);
    toggle_group(&mut faces, &face(8.0, p), vec![face(8.0, p)], &TOL);
    assert_eq!(faces.iter().map(|f| f.centroid.z).collect::<Vec<_>>(), vec![5.0, 6.0, 7.0, 8.0]);
    toggle_group(&mut faces, &face(6.0, c), vec![face(6.0, c), face(5.0, c), face(7.0, p)], &TOL);
    assert_eq!(faces.iter().map(|f| f.centroid.z).collect::<Vec<_>>(), vec![8.0]);
}
