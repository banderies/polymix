// SPDX-License-Identifier: AGPL-3.0-only
//! What the Reverse Engineer draws into a picture of a section: the visible bodies cut by the
//! section's plane, each an outline under the body's name, a sheet's an open line drawn as a
//! surface's; and the sketches drawn in that section's plane, each piece labelled by its id.
//! Ported from ScanRuler (`plugins/reverse/app/reSectionLayer.ts`), © Stefan Hermann (CNC
//! Kitchen), AGPL-3.0-only.
//!
//! ScanRuler cuts the bodies with the kernel's own section sampled at 0.005 mm; here each body
//! is meshed at that chord and the mesh cut, which lies within the chord of the exact cut.

use glam::{DVec2, DVec3};
use serde_json::{json, Value};

use crate::frame::SectionFrame;
use crate::model::{Feature, FeatureKind};
use crate::slice::slice_mesh;

/// How finely a body's cut is sampled, mm.
pub const CUT_CHORD_MM: f64 = 0.005;
/// How near two ends lie to be one point of a loop, mm.
const JOIN_MM: f64 = 1e-4;
/// How far a sketch's normal may turn off the sheet's, radians, and still be drawn on it.
const SAME_PLANE_RAD: f64 = 1e-4;

/// Polylines joined end to end into the longest chains they make, each closed when it comes
/// back to its start.
pub fn join_polylines(lines: &[Vec<DVec2>]) -> Vec<(Vec<DVec2>, bool)> {
    let mut left: Vec<Vec<DVec2>> = lines.iter().filter(|l| l.len() >= 2).cloned().collect();
    let near = |a: DVec2, b: DVec2| a.distance(b) <= JOIN_MM;
    let mut out = Vec::new();
    while let Some(mut chain) = left.pop() {
        let mut grew = true;
        while grew && !near(chain[0], chain[chain.len() - 1]) {
            grew = false;
            for i in 0..left.len() {
                let l = &left[i];
                let (end, start) = (chain[chain.len() - 1], chain[0]);
                if near(end, l[0]) {
                    chain.extend_from_slice(&l[1..]);
                } else if near(end, l[l.len() - 1]) {
                    chain.extend(l.iter().rev().skip(1));
                } else if near(start, l[l.len() - 1]) {
                    let mut c = l[..l.len() - 1].to_vec();
                    c.extend_from_slice(&chain);
                    chain = c;
                } else if near(start, l[0]) {
                    let mut c: Vec<DVec2> = l.iter().rev().take(l.len() - 1).copied().collect();
                    c.extend_from_slice(&chain);
                    chain = c;
                } else {
                    continue;
                }
                left.remove(i);
                grew = true;
                break;
            }
        }
        let closed = chain.len() > 2 && near(chain[0], chain[chain.len() - 1]);
        out.push((chain, closed));
    }
    out
}

/// Whether a sketch's plane is the sheet's: its normal along the sheet's, either way round,
/// and its zero on the sheet's plane within `mm`.
pub fn lies_on_sheet(frame: &SectionFrame, sheet: &SectionFrame, mm: f64) -> bool {
    let (Some(n), Some(m)) = (sheet.normal.try_normalize(), frame.normal.try_normalize()) else { return false };
    if n.cross(m).length() > SAME_PLANE_RAD {
        return false;
    }
    (frame.origin - sheet.origin).dot(n).abs() <= mm
}

/// A body to cut: its name, its mesh (positions and triangles), whether a sheet.
pub struct LayerBody<'a> {
    pub name: &'a str,
    pub positions: &'a [DVec3],
    pub triangles: &'a [[u32; 3]],
    pub surface: bool,
}

/// The layer for a sheet: the bodies' outlines and the sketches' pieces, as ScanRuler's
/// section pictures take them (`{ outlines: [{label, points, closed, surface?}], pieces:
/// [{label, points}] }`), or `None` when there is nothing to draw.
pub fn section_layer(sheet: &SectionFrame, bodies: &[LayerBody], features: &[Feature], model_size: f64) -> Option<Value> {
    let mut outlines = Vec::new();
    for b in bodies {
        let cut = slice_mesh(b.positions, b.triangles, sheet.origin, sheet.normal, 0.0);
        let lines: Vec<Vec<DVec2>> = cut.project(sheet);
        for (points, closed) in join_polylines(&lines) {
            let mut o = json!({"label": b.name, "points": points.iter().map(|p| [p.x, p.y]).collect::<Vec<_>>(), "closed": closed});
            if b.surface {
                o["surface"] = json!(true);
            }
            outlines.push(o);
        }
    }
    let mm = (1e-6 * model_size).max(1e-3);
    let sketches: Vec<(&Feature, &crate::model::SketchFeature)> = features
        .iter()
        .filter_map(|f| match &f.kind {
            FeatureKind::Sketch(s) if lies_on_sheet(&s.frame, sheet, mm) => Some((f, s)),
            _ => None,
        })
        .collect();
    let several = sketches.len() > 1;
    let mut pieces = Vec::new();
    for (f, s) in &sketches {
        for e in s.sketch.entities.iter().filter(|e| !e.construction && !matches!(e.shape, crate::sketch::Shape::Point { .. })) {
            let pts: Vec<[f64; 2]> = crate::sketch::polyline(&e.shape, 64).iter().map(|q| sheet.to_sheet(s.frame.to_world(*q))).map(|p| [p.x, p.y]).collect();
            let label = if several { format!("{} {}", f.name, e.id) } else { e.id.to_string() };
            pieces.push(json!({"label": label, "points": pts}));
        }
    }
    (!outlines.is_empty() || !pieces.is_empty()).then(|| json!({"outlines": outlines, "pieces": pieces}))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Ported from ScanRuler's tests/reSectionWorld.test.ts (joinPolylines, liesOnSheet).
    #[test]
    fn joins_polylines_into_loops_either_way_round() {
        let a = vec![DVec2::new(0.0, 0.0), DVec2::new(1.0, 0.0)];
        let b = vec![DVec2::new(1.0, 1.0), DVec2::new(1.0, 0.0)];
        let c = vec![DVec2::new(1.0, 1.0), DVec2::new(0.0, 0.0)];
        let joined = join_polylines(&[a, b, c]);
        assert_eq!(joined.len(), 1);
        assert!(joined[0].1);
        assert_eq!(joined[0].0.len(), 4);
        let open = join_polylines(&[vec![DVec2::ZERO, DVec2::X], vec![DVec2::new(5.0, 5.0), DVec2::new(6.0, 5.0)]]);
        assert_eq!(open.len(), 2);
        assert!(open.iter().all(|o| !o.1));
    }

    #[test]
    fn a_sketch_lies_on_the_sheet_by_its_plane_either_way_round() {
        let sheet = SectionFrame { origin: DVec3::new(0.0, 0.0, 5.0), normal: DVec3::Z, u: DVec3::X, v: DVec3::Y };
        let flipped = SectionFrame { origin: DVec3::new(3.0, 1.0, 5.0), normal: -DVec3::Z, u: DVec3::X, v: -DVec3::Y };
        assert!(lies_on_sheet(&flipped, &sheet, 1e-3));
        assert!(!lies_on_sheet(&flipped.offset(0.01), &sheet, 1e-3));
        let turned = SectionFrame { normal: DVec3::X, ..sheet };
        assert!(!lies_on_sheet(&turned, &sheet, 1e-3));
    }

    #[test]
    fn a_cube_cut_through_its_middle_is_one_closed_outline() {
        let m = mm_core::primitives::cube(10.0, 2);
        let pos: Vec<DVec3> = m.positions().to_vec();
        let tris: Vec<[u32; 3]> = m.triangle_ids().map(|t| m.triangle(t)).collect();
        let sheet = SectionFrame { origin: DVec3::ZERO, normal: DVec3::Z, u: DVec3::X, v: DVec3::Y };
        let layer = section_layer(&sheet, &[LayerBody { name: "Solid 1", positions: &pos, triangles: &tris, surface: false }], &[], 10.0).unwrap();
        assert_eq!(layer["outlines"].as_array().unwrap().len(), 1);
        assert_eq!(layer["outlines"][0]["closed"], json!(true));
        assert_eq!(layer["outlines"][0]["label"], json!("Solid 1"));
        let away = SectionFrame { origin: DVec3::new(0.0, 0.0, 50.0), ..sheet };
        assert!(section_layer(&away, &[LayerBody { name: "Solid 1", positions: &pos, triangles: &tris, surface: false }], &[], 10.0).is_none());
    }
}
