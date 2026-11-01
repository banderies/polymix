// SPDX-License-Identifier: AGPL-3.0-only
//! A chamfer by two distances, or by a distance and an angle, put to the kernel's one such
//! chamfer. Ported from ScanRuler (`plugins/reverse/core/chamferSides.ts`), © Stefan Hermann
//! (CNC Kitchen), AGPL-3.0-only.
//!
//! The kernel takes a distance on each edge's first face — the first of the solid's faces, in
//! the kernel's order, that the edge bounds — and the angle to that face. Which face that is
//! the operator has no say in, so what is asked of it is worked out from the chamfer wanted on
//! the reference face and the angle the two faces meet at.

use std::collections::HashMap;

use glam::DVec3;
use serde::{Deserialize, Serialize};

use crate::fillet_num::{js_num, js_round};

/// The chamfer as the feature asks for it, on each edge's reference face: two distances, the
/// first on that face — or a distance on it and the angle the chamfer makes with it, degrees.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "camelCase", rename_all_fields = "camelCase")]
pub enum SidedChamfer {
    TwoDistances { distance: f64, distance2: f64 },
    DistanceAngle { distance: f64, angle_deg: f64 },
}

/// What the kernel is asked at one edge: the distance on its first face, and the chamfer's
/// angle to that face, degrees.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AngledAsk {
    pub distance: f64,
    pub angle_deg: f64,
}

const DEG: f64 = 180.0 / std::f64::consts::PI;

/// The angle at an edge between its two faces, as a chamfer's cross-section sees it: between
/// the lines the faces leave the edge along, square to it — 90° at a block's edge, 120° at a
/// hexagon's. Read off the faces' outward normals there, it is the half turn less the angle
/// between them, for an edge into the body and one out of it alike.
pub fn face_angle(normals: [DVec3; 2]) -> f64 {
    let c = normals[0].dot(normals[1]).clamp(-1.0, 1.0);
    std::f64::consts::PI - c.acos()
}

/// A tenth of a degree, as the reasons write an angle.
fn tenth(x: f64) -> String {
    js_num(js_round(x * 10.0) / 10.0)
}

/// What the kernel is asked at an edge for the chamfer wanted there: measured from the edge's
/// first face — the reference face itself when `ref_is_first`, else the other — at the angle
/// the faces meet at. The triangle the chamfer cuts off has the faces' angle at the edge and
/// its two setbacks along the faces, so the one face's setback and angle follow from the
/// other's by the sines. A chamfer the kernel cannot be asked for from that face — its angle
/// there not under 90° — is a sentence saying so (the `Err`). `scale` sizes the distances (1
/// as asked).
pub fn angled_ask(want: SidedChamfer, ref_is_first: bool, normals: [DVec3; 2], scale: f64) -> Result<AngledAsk, String> {
    let phi = face_angle(normals);
    if !(phi > 1e-3 && phi < std::f64::consts::PI - 1e-3) {
        return Err("the faces at the edge run on into each other — there is no corner to chamfer.".to_string());
    }
    let (distance, angle) = match want {
        SidedChamfer::TwoDistances { distance: d1, distance2: d2 } => {
            let (a, b) = if ref_is_first { (d1, d2) } else { (d2, d1) };
            (a * scale, (b * phi.sin()).atan2(a - b * phi.cos()))
        }
        SidedChamfer::DistanceAngle { distance: d, angle_deg } => {
            let theta = angle_deg / DEG;
            if ref_is_first {
                (d * scale, theta)
            } else {
                // The setback on the other face, and the chamfer's angle to it: what is left
                // of the half turn.
                let far = std::f64::consts::PI - phi - theta;
                if !(far > 1e-6) {
                    return Err(format!("at {}° the chamfer does not reach the other face, which meets this one at {}°.", tenth(angle_deg), tenth(phi * DEG)));
                }
                ((d * scale * theta.sin()) / (phi + theta).sin(), far)
            }
        }
    };
    let angle_deg = angle * DEG;
    if !(angle_deg > 1e-6 && angle_deg < 90.0 - 1e-6) {
        return Err(format!(
            "the kernel measures this edge's chamfer from its other face, where it would stand at {}° — under 90° is all it takes there; flip the chamfer, or swap the distances.",
            tenth(angle_deg)
        ));
    }
    Ok(AngledAsk { distance, angle_deg })
}

/// The reference face of each edge, by the face hashes the edges have on either side (the
/// kernel's first face first): the face most of the edges share — a rim's top, a bore's end —
/// or, where the two count the same, the first; flipped, the other one.
pub fn reference_faces(sides: &[[u32; 2]], flip: bool) -> Vec<u32> {
    let mut count: HashMap<u32, usize> = HashMap::new();
    for s in sides {
        // Each face once per edge, as a set of the two would count it.
        *count.entry(s[0]).or_insert(0) += 1;
        if s[1] != s[0] {
            *count.entry(s[1]).or_insert(0) += 1;
        }
    }
    sides
        .iter()
        .map(|&[a, b]| {
            let ca = count.get(&a).copied().unwrap_or(0);
            let cb = count.get(&b).copied().unwrap_or(0);
            let reference = if cb > ca { b } else { a };
            if flip {
                if reference == a {
                    b
                } else {
                    a
                }
            } else {
                reference
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    //! Ported from ScanRuler `plugins/reverse/tests/reChamferDelete.test.ts`, "the sides of a
    //! chamfer" (the rest of that file builds bodies on the kernel).
    use super::*;

    fn close(a: f64, b: f64, digits: i32) -> bool {
        (a - b).abs() < 10f64.powi(-digits) / 2.0
    }

    #[test]
    fn reads_the_angle_the_faces_meet_at_off_their_normals_for_an_edge_out_of_the_body_and_one_into_it() {
        assert!(close(face_angle([DVec3::Z, DVec3::X]) * 180.0 / std::f64::consts::PI, 90.0, 9));
        // A hexagon's edge: the normals 60° apart.
        let a = face_angle([DVec3::X, DVec3::new(0.5, 3f64.sqrt() / 2.0, 0.0)]) * 180.0 / std::f64::consts::PI;
        assert!(close(a, 120.0, 9), "{a}");
    }

    #[test]
    fn asks_the_kernel_measuring_from_its_first_face_for_the_chamfer_wanted_on_the_reference_face() {
        let square = [DVec3::Z, DVec3::new(0.0, -1.0, 0.0)];
        let two = SidedChamfer::TwoDistances { distance: 1.0, distance2: 2.0 };
        let deg = |x: f64| x * 180.0 / std::f64::consts::PI;
        let r = angled_ask(two, true, square, 1.0).unwrap();
        assert_eq!(r.distance, 1.0);
        assert!(close(r.angle_deg, deg(2f64.atan()), 9));
        let r = angled_ask(two, false, square, 1.0).unwrap();
        assert_eq!(r.distance, 2.0);
        assert!(close(r.angle_deg, deg(0.5f64.atan()), 9));
        let angled = SidedChamfer::DistanceAngle { distance: 1.0, angle_deg: 30.0 };
        let r = angled_ask(angled, true, square, 1.0).unwrap();
        assert_eq!(r.distance, 1.0);
        assert!(close(r.angle_deg, 30.0, 9));
        // From the other face: its setback tan 30°, its angle what is left.
        let other = angled_ask(angled, false, square, 1.0).unwrap();
        assert!(close(other.distance, (std::f64::consts::PI / 6.0).tan(), 9));
        assert!(close(other.angle_deg, 60.0, 9));
        // Sized down for a diagnosis, the distances alike.
        assert!(close(angled_ask(two, false, square, 0.5).unwrap().distance, 1.0, 9));
        // A sharp wedge: the long side's angle on the short one's face is past what the kernel
        // takes.
        let wedge = [DVec3::Y, DVec3::new((std::f64::consts::PI / 3.0).sin(), -0.5, 0.0)];
        let refused = angled_ask(SidedChamfer::TwoDistances { distance: 1.0, distance2: 3.0 }, true, wedge, 1.0).unwrap_err();
        assert!(refused.contains("under 90°"), "{refused}");
    }

    #[test]
    fn reads_and_writes_the_chamfer_as_scanruler_s_feature_has_it() {
        // Not in ScanRuler's tests: the shapes `SidedChamfer` and `AngledAsk` have there.
        let want: SidedChamfer = serde_json::from_str(r#"{"mode":"distanceAngle","distance":1,"angleDeg":30}"#).unwrap();
        assert_eq!(want, SidedChamfer::DistanceAngle { distance: 1.0, angle_deg: 30.0 });
        let two: SidedChamfer = serde_json::from_str(r#"{"mode":"twoDistances","distance":1,"distance2":2}"#).unwrap();
        assert_eq!(two, SidedChamfer::TwoDistances { distance: 1.0, distance2: 2.0 });
        assert_eq!(serde_json::to_value(AngledAsk { distance: 1.0, angle_deg: 30.0 }).unwrap(), serde_json::json!({ "distance": 1.0, "angleDeg": 30.0 }));
    }

    #[test]
    fn takes_the_face_most_of_the_edges_share_as_their_reference_and_the_other_one_flipped() {
        // A rim: four edges, the top (1) shared by all — whichever the kernel lists first.
        let rim = [[1, 2], [3, 1], [1, 4], [5, 1]];
        assert_eq!(reference_faces(&rim, false), vec![1, 1, 1, 1]);
        assert_eq!(reference_faces(&rim, true), vec![2, 3, 4, 5]);
        // One edge alone: the kernel's first.
        assert_eq!(reference_faces(&[[7, 8]], false), vec![7]);
    }
}
