// SPDX-License-Identifier: AGPL-3.0-only
//! The Revolve build's tests: ScanRuler's `reCoil.test.ts` (a revolve with a pitch) and the
//! revolve cases of `reBuild.test.ts`, © Stefan Hermann (CNC Kitchen), AGPL-3.0-only — on the
//! SolidCraft kernel.

use glam::{DVec2, DVec3};

use super::super::*;
use crate::frame::{SectionFrame, WorldAxis};
use crate::model::*;
use crate::regions::find_regions;
use crate::sketch::{Shape, Sketch};

/// The XZ plane, seen from −Y: the sheet's X is the part's X, its Y the part's Z — a profile drawn
/// here turns about Z, the way a lathe part's section is drawn.
fn xz() -> SectionFrame {
    SectionFrame { origin: DVec3::ZERO, normal: DVec3::new(0.0, -1.0, 0.0), u: DVec3::X, v: DVec3::Z }
}

fn on_frame(frame: SectionFrame, sketch: Sketch) -> FeatureKind {
    FeatureKind::Sketch(SketchFeature { plane: PlaneRef::Fitted { frame, label: None }, frame, sketch, exprs: Default::default(), image: None, inset: None })
}

/// A sketch on the XZ plane holding a rectangle from `a` to `b` on the sheet (feature 1).
fn profile(part: &mut Part, a: [f64; 2], b: [f64; 2]) -> (FeatureId, Vec<String>) {
    let mut s = Sketch::default();
    s.add_rectangle(DVec2::from(a), DVec2::from(b));
    let regions = find_regions(&s).into_iter().map(|r| r.key).collect();
    (part.add(on_frame(xz(), s)), regions)
}

fn coil(sketch: &(FeatureId, Vec<String>), angle: f64, pitch: f64, left: bool) -> FeatureKind {
    let mut r = RevolveFeature::new(sketch.0, sketch.1.clone(), AxisRef::World { axis: WorldAxis::Z }, angle, BodyOp::New);
    r.pitch = Some(pitch);
    r.left_handed = left;
    FeatureKind::Revolve(r)
}

/// A section of area A whose middle is R off the axis, screwed through `turns`: Pappus — the
/// climb along the axis adds nothing.
fn pappus(area: f64, radius: f64, turns: f64) -> f64 {
    area * 2.0 * std::f64::consts::PI * radius * turns
}

/// A body's box off its tessellation: tight, where the kernel's box holds the control points.
fn tight(b: &Body) -> (DVec3, DVec3) {
    let m = b.tessellate(0.005).expect("a mesh");
    m.positions.iter().fold((DVec3::splat(f64::INFINITY), DVec3::splat(f64::NEG_INFINITY)), |(lo, hi), p| (lo.min(dv3(*p)), hi.max(dv3(*p))))
}

fn body_of(built: &Built, f: FeatureId) -> &BuiltBody {
    built.body(BodyId::new(f)).unwrap_or_else(|| panic!("no body {f}: {:?}", built.errors))
}

#[test]
fn coils_on_past_a_full_turn_climbing_the_pitch_every_turn() {
    // 4 × 2 at 8..12 from the Z axis, 5 mm a turn, three turns.
    let mut part = Part::default();
    let s = profile(&mut part, [8.0, -1.0], [12.0, 1.0]);
    let f = part.add(coil(&s, 1080.0, 5.0, false));
    let built = build(&part);
    assert!(built.errors.is_empty(), "{:?}", built.errors);
    let b = body_of(&built, f);
    let v = volume(&b.body).unwrap_or(0.0);
    assert!((v - pappus(8.0, 10.0, 3.0)).abs() / pappus(8.0, 10.0, 3.0) < 1e-3, "{v}");
    // Each side of the section is one face however many turns: four sides and the two ends.
    assert_eq!(b.body.face_count(), 6);
    let (lo, hi) = tight(&b.body);
    assert!((lo.z + 1.0).abs() < 1e-2 && (hi.z - 16.0).abs() < 1e-2, "{lo} {hi}");
    assert!((hi.x - 12.0).abs() < 0.05, "{lo} {hi}");
}

#[test]
fn turns_as_a_right_hand_screw_or_a_left_hand_one_and_climbs_the_other_way_with_a_negative_pitch() {
    let run = |pitch: f64, left: bool| {
        let mut part = Part::default();
        let s = profile(&mut part, [8.0, -1.0], [12.0, 1.0]);
        let f = part.add(coil(&s, 720.0, pitch, left));
        let built = build(&part);
        assert!(built.errors.is_empty(), "{:?}", built.errors);
        body_of(&built, f).body.clone()
    };
    let (right, left, down) = (run(5.0, false), run(5.0, true), run(-5.0, false));
    // A quarter turn on, the profile's middle has climbed a quarter pitch: to +Y turning
    // counter-clockwise about +Z, to −Y the other hand.
    let at = |b: &Body, p: [f64; 3]| b.contains(sv3(DVec3::from(p)));
    assert!(at(&right, [0.0, 10.0, 1.25]));
    assert!(!at(&right, [0.0, -10.0, 1.25]));
    assert!(at(&left, [0.0, -10.0, 1.25]));
    assert!(!at(&left, [0.0, 10.0, 1.25]));
    let (vr, vl) = (volume(&right).unwrap_or(0.0), volume(&left).unwrap_or(1.0));
    assert!((vr - vl).abs() < 1e-3 * vr, "{vr} {vl}");
    // Climbing down, still right-handed: counter-clockwise seen down the way it climbs, which is
    // clockwise about +Z.
    assert!(at(&down, [0.0, -10.0, -1.25]));
    let (lo, hi) = tight(&down);
    assert!((hi.z - 1.0).abs() < 1e-2 && (lo.z + 11.0).abs() < 1e-2, "{lo} {hi}");
}

#[test]
fn runs_a_hole_in_the_profile_the_whole_way_along() {
    let mut part = Part::default();
    let mut s = Sketch::default();
    s.add_shape(Shape::Circle { c: DVec2::new(10.0, 0.0), r: 2.5 });
    s.add_shape(Shape::Circle { c: DVec2::new(10.0, 0.0), r: 1.5 });
    let tube = find_regions(&s).into_iter().find(|r| !r.holes.is_empty()).map(|r| r.key).expect("the tube's section");
    let sk = part.add(on_frame(xz(), s));
    let f = part.add(coil(&(sk, vec![tube]), 1080.0, 6.0, false));
    let built = build(&part);
    assert!(built.errors.is_empty(), "{:?}", built.errors);
    let want = pappus(std::f64::consts::PI * (2.5f64.powi(2) - 1.5f64.powi(2)), 10.0, 3.0);
    let v = volume(&body_of(&built, f).body).unwrap_or(0.0);
    assert!((v - want).abs() / want < 2e-3, "{v} {want}");
}

#[test]
fn cuts_a_thread_into_a_bar() {
    // A Ø10 bar up Z, and a 60° vee 0.9 deep reaching past its surface, 1.5 mm a turn. ScanRuler
    // cuts twelve turns into a bar 20 long; the kernel's boolean takes minutes over twelve helical
    // turns, so three are cut here into a bar 6 long — the same share of the bar taken.
    let mut part = Part::default();
    let mut bar = Sketch::default();
    bar.add_shape(Shape::Circle { c: DVec2::ZERO, r: 5.0 });
    let regions = find_regions(&bar).into_iter().map(|r| r.key).collect();
    let bs = part.add(FeatureKind::Sketch(SketchFeature::on_world(WorldAxis::Z, 0.0, bar)));
    part.add(FeatureKind::Extrude(ExtrudeFeature::new(bs, regions, ExtrudeExtent::Distance { from: 0.0, to: 6.0 }, BodyOp::New)));
    let mut vee = Sketch::default();
    let h = 0.65;
    let pts = [DVec2::new(5.3, 1.0 - h), DVec2::new(5.3, 1.0 + h), DVec2::new(5.3 - 1.2, 1.0)];
    for i in 0..3 {
        vee.add_shape(Shape::Line { a: pts[i], b: pts[(i + 1) % 3] });
    }
    let regions = find_regions(&vee).into_iter().map(|r| r.key).collect();
    let vs = part.add(on_frame(xz(), vee));
    let mut thread = RevolveFeature::new(vs, regions, AxisRef::World { axis: WorldAxis::Z }, 360.0 * 3.0, BodyOp::Cut);
    thread.pitch = Some(1.5);
    part.add(FeatureKind::Revolve(thread));
    let built = build(&part);
    assert!(built.errors.is_empty(), "{:?}", built.errors);
    assert_eq!(built.bodies.len(), 1);
    let cylinder = std::f64::consts::PI * 25.0 * 6.0;
    let v = volume(&built.bodies[0].body).unwrap_or(0.0);
    assert!(v < cylinder * 0.95 && v > cylinder * 0.7, "{v}");
}

#[test]
fn refuses_a_profile_that_would_run_into_its_next_turn_and_one_on_the_axis() {
    let message = |a: [f64; 2], b: [f64; 2], angle: f64, pitch: f64| {
        let mut part = Part::default();
        let s = profile(&mut part, a, b);
        let f = part.add(coil(&s, angle, pitch, false));
        build(&part).errors.get(&f).cloned()
    };
    let crowded = message([8.0, -2.0], [12.0, 2.0], 720.0, 3.0).unwrap_or_default();
    assert!(crowded.contains("reaches 4.000 mm along the axis — as far as the pitch of 3.000 mm or further"), "{crowded}");
    // Under one turn it never meets itself.
    assert_eq!(message([8.0, -2.0], [12.0, 2.0], 180.0, 3.0), None);
    let touching = message([0.0, -1.0], [4.0, 1.0], 720.0, 5.0).unwrap_or_default();
    assert!(touching.contains("touches or crosses the axis"), "{touching}");
    // Across it, the ends of every side well clear of it.
    let crossing = message([-1.0, -1.0], [3.0, 1.0], 720.0, 5.0).unwrap_or_default();
    assert!(crossing.contains("touches or crosses the axis"), "{crossing}");
}

#[test]
fn keeps_a_plain_revolve_to_a_full_turn() {
    let mut part = Part::default();
    let s = profile(&mut part, [8.0, -1.0], [12.0, 1.0]);
    let f = part.add(coil(&s, 720.0, 0.0, false));
    let built = build(&part);
    assert!(built.errors.get(&f).is_some_and(|m| m.contains("360° at most — give it a pitch")), "{:?}", built.errors);
    let mut part = Part::default();
    let s = profile(&mut part, [8.0, -1.0], [12.0, 1.0]);
    let f = part.add(coil(&s, 360.0, 0.0, false));
    let built = build(&part);
    assert!(built.errors.is_empty(), "{:?}", built.errors);
    let v = volume(&body_of(&built, f).body).unwrap_or(0.0);
    assert!((v - pappus(8.0, 10.0, 1.0)).abs() < 1e-3 * v, "{v}");
}

#[test]
fn names_a_profile_across_the_axis_and_an_axis_beside_the_plane() {
    // Reaching 1 mm over the Z axis.
    let mut part = Part::default();
    let s = profile(&mut part, [-1.0, -1.0], [3.0, 1.0]);
    let f = part.add(FeatureKind::Revolve(RevolveFeature::new(s.0, s.1, AxisRef::World { axis: WorldAxis::Z }, 360.0, BodyOp::New)));
    let m = build(&part).errors.get(&f).cloned().unwrap_or_default();
    assert!(m.starts_with("The profile crosses the revolve’s axis — it reaches 1 mm over it at (-1, 0, "), "{m}");
    // The X axis lies in the XZ plane; the Y axis pierces it outside the profile.
    let mut part = Part::default();
    let s = profile(&mut part, [8.0, 1.0], [12.0, 3.0]);
    let x = part.add(FeatureKind::Revolve(RevolveFeature::new(s.0, s.1.clone(), AxisRef::World { axis: WorldAxis::X }, 180.0, BodyOp::New)));
    let y = part.add(FeatureKind::Revolve(RevolveFeature::new(s.0, s.1, AxisRef::World { axis: WorldAxis::Y }, 180.0, BodyOp::New)));
    let built = build(&part);
    assert!(built.errors.get(&x).is_none(), "{:?}", built.errors);
    assert!(built.errors.get(&y).is_some_and(|m| m.starts_with("The axis does not lie in the sketch's plane")), "{:?}", built.errors);
}

#[test]
fn revolves_about_a_line_of_its_sketch_and_reads_the_ghosts_volume() {
    // A 5 × 10 rectangle with its left side on the sheet's Y axis, turned about that side.
    let mut part = Part::default();
    let mut s = Sketch::default();
    s.add_rectangle(DVec2::new(0.0, 0.0), DVec2::new(5.0, 10.0));
    let axis = s.entities.iter().find(|e| matches!(e.shape, Shape::Line { a, b } if a.x.abs() < 1e-9 && b.x.abs() < 1e-9)).map(|e| e.id).expect("the left side");
    let regions = find_regions(&s).into_iter().map(|r| r.key).collect();
    let sk = part.add(FeatureKind::Sketch(SketchFeature::on_world(WorldAxis::Z, 0.0, s)));
    let f = part.add(FeatureKind::Revolve(RevolveFeature::new(sk, regions, AxisRef::SketchLine { entity: axis }, 360.0, BodyOp::New)));
    let built = build(&part);
    assert!(built.errors.is_empty(), "{:?}", built.errors);
    let want = std::f64::consts::PI * 25.0 * 10.0;
    let v = volume(&body_of(&built, f).body).unwrap_or(0.0);
    assert!((v - want).abs() < 1e-3 * want, "{v}");
    assert!(built.ghost_volumes.get(&f).is_some_and(|g| (g - v).abs() < 1e-6 * want));
    // No axis chosen: said so.
    let mut part2 = part.clone();
    if let Some(FeatureKind::Revolve(r)) = part2.feature_mut(f).map(|x| &mut x.kind) {
        r.axis = crate::extrude_draft::no_axis();
    }
    assert_eq!(build(&part2).errors.get(&f).map(String::as_str), Some("Choose the axis the revolve turns about."));
}
