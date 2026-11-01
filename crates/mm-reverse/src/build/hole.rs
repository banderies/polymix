// SPDX-License-Identifier: AGPL-3.0-only
//! The Hole feature's build: a round hole cut out of its body along an axis — a cylinder or
//! circle element's, a line element's, a round face's of the body itself — through everything
//! or to a depth each way along the axis from its point; or drilled square into a flat face of
//! the body from a point on it, into the body alone, to a depth. Ported from ScanRuler
//! (`plugins/reverse/core/resolve.ts` resolveHoleAxis and the hole step, `build.ts` the hole
//! step and tookSomething, `primitives.ts` cylinderSolid), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only, on the SolidCraft kernel.
//!
//! The radius and the depth are the feature's own: a hole sized for a heat-set insert holds the
//! insert's (`crate::inserts`), set by its box.

use glam::DVec3;
use solidcraft_kernel::{self as kernel, BoolOp};

use super::{diagonal_of, kerr, origins_after, sv3, volume, Builder};
use crate::model::{BodyId, ElementGeometry, Feature, FeatureKind, HoleAxis, HoleFeature, Part};
use crate::topo::{FaceFingerprint, FaceKind};

/// How far out of the face a hole drilled into one starts, mm: the cut does not end on the face.
pub const HOLE_LEAD_MM: f64 = 0.01;

/// Where a hole's axis runs: a point on it, its direction, the radius of what it was taken
/// from (a cylinder's, a circle's), and whether it goes one way only — into the body from a
/// flat face.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HoleAxisAt {
    pub origin: DVec3,
    pub dir: DVec3,
    pub radius: Option<f64>,
    pub one_way: bool,
}

/// A face fingerprint that names no face: what a new Hole or Thread holds until a face is
/// clicked (the model has no "none" for it, as ScanRuler's draft has).
pub fn unpicked_face() -> FaceFingerprint {
    FaceFingerprint { kind: FaceKind::Other, centroid: DVec3::ZERO, dir: DVec3::Z, area: 0.0, radius: None, center: None, origin: -1 }
}

/// Whether a fingerprint is [`unpicked_face`]'s: nothing clicked yet.
pub fn is_unpicked(f: &FaceFingerprint) -> bool {
    f.kind == FaceKind::Other && f.area == 0.0 && f.origin == -1 && f.radius.is_none() && f.center.is_none()
}

/// What a Hole with nothing to run along says, while its box waits for a click.
pub const HOLE_UNPICKED: &str = "Click a round face — or a flat one to drill into at its centre — or choose an element to drill along.";

/// Where a hole's axis runs, or why it cannot (ScanRuler's `resolveHoleAxis`): a cylinder or
/// circle element's axis and radius, a line element's axis, a cylindrical face's axis and
/// radius, or a flat face's inward normal from the point on it.
pub fn resolve_hole_axis(axis: &HoleAxis, part: &Part) -> Result<HoleAxisAt, String> {
    match axis {
        HoleAxis::Point { face, at } => {
            if face.kind != FaceKind::Plane {
                return Err("A hole is drilled into a flat face — the face picked is not one.".into());
            }
            Ok(HoleAxisAt { origin: *at, dir: -face.dir, radius: None, one_way: true })
        }
        HoleAxis::Face { face } => {
            if is_unpicked(face) {
                return Err(HOLE_UNPICKED.into());
            }
            match (face.kind, face.center) {
                (FaceKind::Cylinder, Some(center)) => Ok(HoleAxisAt { origin: center, dir: face.dir, radius: face.radius, one_way: false }),
                _ => Err("The face the hole follows is not a cylinder.".into()),
            }
        }
        HoleAxis::Element { id } => {
            let Some(e) = part.element(*id) else { return Err("The element the hole follows is gone.".into()) };
            match &e.geometry {
                ElementGeometry::Cylinder { center, axis, radius, .. } => Ok(HoleAxisAt { origin: *center, dir: *axis, radius: Some(*radius), one_way: false }),
                ElementGeometry::Circle { center, normal, radius } => Ok(HoleAxisAt { origin: *center, dir: *normal, radius: Some(*radius), one_way: false }),
                ElementGeometry::Line { center, dir, .. } => Ok(HoleAxisAt { origin: *center, dir: *dir, radius: None, one_way: false }),
                other => Err(format!("A {} has no axis to drill along.", other.kind())),
            }
        }
    }
}

/// Why a Hole cannot be resolved into a cut, read off the feature alone — or its axis.
pub fn resolve_hole(x: &HoleFeature, part: &Part) -> Result<HoleAxisAt, String> {
    let axis = resolve_hole_axis(&x.axis, part)?;
    if !(x.radius > 0.0) {
        return Err("The hole needs a radius.".into());
    }
    Ok(axis)
}

/// A solid cylinder about the axis through `origin`, from `from` to `to` millimetres along it
/// (ScanRuler's `cylinderSolid`).
pub(crate) fn cylinder_solid(origin: DVec3, axis: DVec3, radius: f64, from: f64, to: f64) -> Result<kernel::Body, String> {
    let n = axis.try_normalize().ok_or("The axis has no direction.")?;
    let (lo, hi) = (from.min(to), from.max(to));
    kernel::cylinder(sv3(origin + n * lo), sv3(n), radius, hi - lo).map_err(kerr)
}

/// The reach past a body that a through-cut needs (ScanRuler's `reachOf`).
pub(crate) fn reach_of(body: &kernel::Body) -> f64 {
    (diagonal_of(body) * 2.0).max(10.0)
}

/// A body's name as the body list gives it while the build runs: the name given it in the
/// part, else Solid n (Shell n for a surface) in the order of the features that made the bodies
/// standing — what a message names the body by (see `super::name_bodies`).
pub(crate) fn body_name(b: &Builder, part: &Part, id: BodyId) -> String {
    if let Some(n) = part.body_styles.get(&id).and_then(|s| s.name.clone()).filter(|n| !n.is_empty()) {
        return n;
    }
    let at = |x: BodyId| part.index_of(x.maker()).unwrap_or(usize::MAX);
    let mut ids = b.body_ids();
    ids.sort_by_key(|x| (at(*x), x.island));
    let (mut solids, mut shells) = (0, 0);
    let mut labelled: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for x in ids {
        let Some(s) = b.body(x) else { continue };
        let counted = if s.surface {
            shells += 1;
            format!("Shell {shells}")
        } else {
            solids += 1;
            format!("Solid {solids}")
        };
        let name = match &s.label {
            Some(l) => {
                let n = labelled.entry(l.clone()).or_insert(0);
                *n += 1;
                if *n == 1 {
                    l.clone()
                } else {
                    format!("{l} ({n})")
                }
            }
            None => counted,
        };
        if x == id {
            return name;
        }
    }
    id.to_string()
}

pub(crate) fn build(b: &mut Builder, part: &Part, _f: &Feature, x: &HoleFeature) -> Result<(), String> {
    let axis = resolve_hole(x, part)?;
    let (id, state) = b.body_for(x.body)?;
    let reach = reach_of(&state.shape);
    // From a flat face: into the body alone, the tool starting a hair out of the face so the
    // cut does not lie on it.
    let back = if axis.one_way {
        -HOLE_LEAD_MM
    } else {
        match x.depth {
            None => -reach,
            Some(d) => -d,
        }
    };
    let to = x.depth.unwrap_or(reach);
    let tool = cylinder_solid(axis.origin, axis.dir, x.radius, back, to)?;
    let was = volume(&state.shape).unwrap_or(0.0);
    let made = kernel::boolean(&state.shape, &tool, BoolOp::Cut).map_err(kerr)?;
    let left = made.as_ref().and_then(volume).unwrap_or(0.0);
    // A cut that meets the body nowhere — a hole drilled beside it or short of it — leaves the
    // body as it was, and a CAD cut with nothing to cut is an error, not a silent success.
    if (was - left).abs() <= 1e-7 * was.max(1.0) {
        return Err(format!("The hole meets nothing of {} — the cut took nothing out. Check where it stands against the body and how far it reaches.", body_name(b, part, id)));
    }
    let Some(made) = made.filter(|m| m.face_count() > 0) else { return Err("The cut left no closed solid (did it take the whole body?)".into()) };
    let tool_origins = b.own_origins(&tool);
    let origins = origins_after(&made, &[(&state.shape, &state.origins), (&tool, &tool_origins)], b.step);
    let ids = b.settle(id, made, origins, true)?;
    b.touched(ids);
    Ok(())
}

/// The hole feature a part's build would make of `x`, for a caller that holds the part: the
/// same resolve, for a box to say why it is not ready.
pub fn hole_problem(x: &HoleFeature, part: &Part) -> Option<String> {
    resolve_hole(x, part).err()
}

/// Whether a feature is a hole whose axis is still [`unpicked_face`]'s.
pub fn hole_unpicked(f: &FeatureKind) -> bool {
    matches!(f, FeatureKind::Hole(HoleFeature { axis: HoleAxis::Face { face }, .. }) if is_unpicked(face))
}

#[cfg(test)]
mod tests {
    //! ScanRuler's `reFeatures.test.ts` ("drills a hole along a cylinder element, and a hole
    //! along a round face of the body") and `reBodyTools.test.ts` ("a hole drilled into a flat
    //! face"), on the SolidCraft kernel.
    use std::f64::consts::PI;

    use glam::DVec2;

    use super::*;
    use crate::build::{build as build_part, default_regions, Built};
    use crate::frame::WorldAxis;
    use crate::inserts::{insert_hole, insert_size};
    use crate::model::{BodyOp, Element, ExtrudeExtent, ExtrudeFeature, SketchFeature};
    use crate::regions::find_regions;
    use crate::sketch::{Shape, Sketch};

    const BOX: f64 = 40.0 * 20.0 * 10.0;

    fn vol(built: &Built, i: usize) -> f64 {
        volume(&built.bodies[i].body).unwrap_or(f64::NAN)
    }

    fn sketch_of(part: &mut Part, s: Sketch) -> (crate::model::FeatureId, Vec<String>) {
        let regions = default_regions(&find_regions(&s));
        (part.add(FeatureKind::Sketch(SketchFeature::on_world(WorldAxis::Z, 0.0, s))), regions)
    }

    /// A 40 × 20 × 10 block: sketch 1, extrude 2.
    fn block() -> Part {
        let mut part = Part::default();
        let mut s = Sketch::default();
        s.add_rectangle(DVec2::new(0.0, 0.0), DVec2::new(40.0, 20.0));
        let (sk, regions) = sketch_of(&mut part, s);
        part.add(FeatureKind::Extrude(ExtrudeFeature::new(sk, regions, ExtrudeExtent::Distance { from: 0.0, to: 10.0 }, BodyOp::New)));
        part
    }

    fn hole(axis: HoleAxis, radius: f64, depth: Option<f64>) -> FeatureKind {
        FeatureKind::Hole(HoleFeature { body: BodyId::new(2), axis, radius, depth, insert: None, exprs: Default::default() })
    }

    fn pin() -> Element {
        Element { id: 10, name: "Cylinder 1".into(), geometry: ElementGeometry::Cylinder { center: DVec3::new(10.0, 10.0, 5.0), axis: DVec3::Z, radius: 3.0, length: 10.0, coverage: 360.0 }, stats: None }
    }

    #[test]
    fn drills_a_hole_along_a_cylinder_element_and_a_hole_along_a_round_face_of_the_body() {
        let mut part = block();
        part.elements.push(pin());
        part.add(hole(HoleAxis::Element { id: 10 }, 3.0, None));
        let built = build_part(&part);
        assert!(built.errors.is_empty(), "{:?}", built.errors);
        assert!((vol(&built, 0) - (BOX - PI * 9.0 * 10.0)).abs() < 1e-3 * BOX, "{}", vol(&built, 0));
        // The hole's wall names the hole's own axis: a second hole along it, wider, follows the
        // first.
        let wall = built.bodies[0].topology.faces.iter().find(|f| f.fingerprint.kind == FaceKind::Cylinder).map(|f| f.fingerprint).unwrap_or_else(unpicked_face);
        let c = wall.center.unwrap_or(DVec3::splat(f64::NAN));
        assert!((c.truncate() - DVec2::new(10.0, 10.0)).length() < 1e-6, "{c}");
        part.add(hole(HoleAxis::Face { face: wall }, 4.0, None));
        let wider = build_part(&part);
        assert!(wider.errors.is_empty(), "{:?}", wider.errors);
        assert!((vol(&wider, 0) - (BOX - PI * 16.0 * 10.0)).abs() < 1e-3 * BOX, "{}", vol(&wider, 0));
    }

    #[test]
    fn goes_into_the_body_alone_as_deep_as_an_insert_asks_from_the_faces_centre() {
        // A standoff: a Ø7 boss 8 tall.
        let mut part = Part::default();
        let mut s = Sketch::default();
        s.add_shape(Shape::Circle { c: DVec2::ZERO, r: 3.5 });
        let (sk, regions) = sketch_of(&mut part, s);
        part.add(FeatureKind::Extrude(ExtrudeFeature::new(sk, regions, ExtrudeExtent::Distance { from: 0.0, to: 8.0 }, BodyOp::New)));
        let boss = build_part(&part);
        let top = boss.bodies[0].topology.faces.iter().find(|f| f.fingerprint.kind == FaceKind::Plane && f.fingerprint.dir.z > 0.9).map(|f| f.fingerprint).unwrap_or_else(unpicked_face);
        let m3 = insert_size("m3").map(insert_hole).unwrap_or(crate::inserts::InsertHole { radius: 0.0, depth: 0.0 });
        assert_eq!((m3.radius, m3.depth), (2.0, 6.7));
        part.add(FeatureKind::Hole(HoleFeature { body: BodyId::new(2), axis: HoleAxis::Point { face: top, at: top.centroid }, radius: m3.radius, depth: Some(m3.depth), insert: Some("M3".into()), exprs: Default::default() }));
        let out = build_part(&part);
        assert!(out.errors.is_empty(), "{:?}", out.errors);
        let want = PI * (3.5f64.powi(2) * 8.0 - 2.0f64.powi(2) * 6.7);
        assert!((vol(&out, 0) - want).abs() < 1e-3 * want, "{} vs {want}", vol(&out, 0));
        // Down from the top only: the bottom stays closed.
        let bottom = out.bodies[0].topology.faces.iter().find(|f| f.fingerprint.kind == FaceKind::Plane && f.fingerprint.dir.z < -0.9).map(|f| f.fingerprint.area).unwrap_or(0.0);
        assert!((bottom - PI * 3.5f64.powi(2)).abs() < 2e-3 * bottom, "{bottom}");
    }

    #[test]
    fn a_hole_to_a_depth_goes_that_far_each_way_along_the_axis() {
        let mut part = block();
        part.elements.push(pin());
        part.add(hole(HoleAxis::Element { id: 10 }, 3.0, Some(2.0)));
        let built = build_part(&part);
        assert!(built.errors.is_empty(), "{:?}", built.errors);
        // From z 3 to z 7 about the element's middle at z 5.
        assert!((vol(&built, 0) - (BOX - PI * 9.0 * 4.0)).abs() < 1e-3 * BOX, "{}", vol(&built, 0));
    }

    #[test]
    fn fails_by_name_beside_the_body_on_a_wrong_face_and_on_a_gone_element() {
        let mut part = block();
        part.elements.push(pin());
        part.elements.push(Element { id: 11, name: "Cylinder 2".into(), geometry: ElementGeometry::Cylinder { center: DVec3::new(100.0, 10.0, 5.0), axis: DVec3::Z, radius: 3.0, length: 10.0, coverage: 360.0 }, stats: None });
        part.elements.push(Element { id: 12, name: "Sphere 1".into(), geometry: ElementGeometry::Sphere { center: DVec3::ZERO, radius: 3.0 }, stats: None });
        let beside = part.add(hole(HoleAxis::Element { id: 11 }, 3.0, None));
        let gone = part.add(hole(HoleAxis::Element { id: 99 }, 3.0, None));
        let ball = part.add(hole(HoleAxis::Element { id: 12 }, 3.0, None));
        let flat = FaceFingerprint { kind: FaceKind::Plane, centroid: DVec3::new(20.0, 10.0, 10.0), dir: DVec3::Z, area: 800.0, radius: None, center: None, origin: 2 };
        let not_round = part.add(hole(HoleAxis::Face { face: flat }, 3.0, None));
        let round = FaceFingerprint { kind: FaceKind::Cylinder, centroid: DVec3::ZERO, dir: DVec3::Z, area: 10.0, radius: Some(1.0), center: Some(DVec3::ZERO), origin: 2 };
        let not_flat = part.add(hole(HoleAxis::Point { face: round, at: DVec3::ZERO }, 3.0, Some(2.0)));
        let unpicked = part.add(hole(HoleAxis::Face { face: unpicked_face() }, 3.0, None));
        let no_radius = part.add(hole(HoleAxis::Element { id: 10 }, 0.0, None));
        let built = build_part(&part);
        let err = |id| built.errors.get(&id).cloned().unwrap_or_default();
        assert_eq!(err(beside), "The hole meets nothing of Solid 1 — the cut took nothing out. Check where it stands against the body and how far it reaches.");
        assert_eq!(err(gone), "The element the hole follows is gone.");
        assert_eq!(err(ball), "A sphere has no axis to drill along.");
        assert_eq!(err(not_round), "The face the hole follows is not a cylinder.");
        assert_eq!(err(not_flat), "A hole is drilled into a flat face — the face picked is not one.");
        assert_eq!(err(unpicked), HOLE_UNPICKED);
        assert_eq!(err(no_radius), "The hole needs a radius.");
        assert!((vol(&built, 0) - BOX).abs() < 1e-6);
    }
}
