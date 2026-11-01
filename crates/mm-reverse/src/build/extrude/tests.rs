// SPDX-License-Identifier: AGPL-3.0-only
//! The Extrude build's tests: ScanRuler's `reTaper.test.ts` (an extrude with a taper),
//! `reExtrudeToFace.test.ts`, `reBuild.test.ts` (planes, tilted planes, a draft, a face pulled as
//! the profile), `reFeatures.test.ts` (to a face, to next, to last) and `reCommandsScan.test.ts`
//! (an oblique prism), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only — on the SolidCraft kernel.

use glam::{DVec2, DVec3};

use super::super::*;
use crate::extrude_draft::{no_plane, NO_BODY};
use crate::frame::{SectionFrame, WorldAxis};
use crate::model::*;
use crate::regions::find_regions;
use crate::sketch::{Shape, Sketch};
use crate::topo::FaceKind;

fn rad(d: f64) -> f64 {
    d.to_radians()
}

fn ok(built: &Built) {
    assert!(built.errors.is_empty(), "{:?}", built.errors);
}

fn vol_of(built: &Built, id: BodyId) -> f64 {
    built.body(id).and_then(|b| volume(&b.body)).unwrap_or(f64::NAN)
}

/// A sketch on a frame of its own.
fn on_frame(frame: SectionFrame, sketch: Sketch) -> FeatureKind {
    FeatureKind::Sketch(SketchFeature { plane: PlaneRef::Fitted { frame, label: None }, frame, sketch, exprs: Default::default(), image: None, inset: None })
}

/// A sketch on the XY plane at `z` with a `w` × `h` rectangle from `at`, and every region of it.
fn rect(part: &mut Part, w: f64, h: f64, at: [f64; 2], z: f64) -> (FeatureId, Vec<String>) {
    let mut s = Sketch::default();
    s.add_rectangle(DVec2::new(at[0], at[1]), DVec2::new(at[0] + w, at[1] + h));
    let regions = find_regions(&s).into_iter().map(|r| r.key).collect();
    (part.add(FeatureKind::Sketch(SketchFeature::on_world(WorldAxis::Z, z, s))), regions)
}

fn ext(sketch: &(FeatureId, Vec<String>), extent: ExtrudeExtent, draft: f64, op: BodyOp) -> FeatureKind {
    let mut e = ExtrudeFeature::new(sketch.0, sketch.1.clone(), extent, op);
    e.draft_deg = draft;
    FeatureKind::Extrude(e)
}

fn dist(from: f64, to: f64) -> ExtrudeExtent {
    ExtrudeExtent::Distance { from, to }
}

/// A 40 × 20 rectangle drafted `deg` over `h`: the frustum's volume.
fn frustum(h: f64, deg: f64, w: f64, d: f64) -> f64 {
    let t = h * rad(deg).tan();
    let top = (w - 2.0 * t) * (d - 2.0 * t);
    (h / 3.0) * (w * d + top + (w * d * top).sqrt())
}

/// The flat faces of a body square to Z, by their height: their areas.
fn caps_of(built: &Built, id: BodyId) -> Vec<(f64, f64)> {
    let Some(b) = built.body(id) else { return Vec::new() };
    b.topology.faces.iter().filter(|f| f.fingerprint.kind == FaceKind::Plane && f.fingerprint.dir.z.abs() > 0.999).map(|f| ((f.fingerprint.centroid.z * 1000.0).round() / 1000.0, f.fingerprint.area)).collect()
}

fn cap_at(built: &Built, id: BodyId, z: f64) -> Option<f64> {
    caps_of(built, id).into_iter().find(|(h, _)| (h - z).abs() < 1e-6).map(|c| c.1)
}

fn plane_element(part: &mut Part, id: ElementId, center: [f64; 3], normal: [f64; 3], basis_u: [f64; 3]) {
    let n = DVec3::from(normal).normalize();
    let u = DVec3::from(basis_u).normalize();
    part.elements.push(Element { id, name: format!("Plane {id}"), geometry: ElementGeometry::Plane { center: DVec3::from(center), normal: n, basis_u: u, basis_v: n.cross(u), extent_u: 80.0, extent_v: 80.0 }, stats: None });
}

fn plane_ref(id: ElementId) -> PlaneRef {
    PlaneRef::Plane { id, offset: None, up: None }
}

// ---- an extrude with a taper (reTaper.test.ts) ----

#[test]
fn a_taper_keeps_the_profile_at_the_sketch_plane_whichever_way_it_runs() {
    let run = |to: f64| {
        let mut part = Part::default();
        let s = rect(&mut part, 40.0, 20.0, [0.0, 0.0], 0.0);
        part.add(ext(&s, dist(0.0, to), 10.0, BodyOp::New));
        build(&part)
    };
    let (up, down) = (run(10.0), run(-10.0));
    ok(&down);
    let id = BodyId::new(2);
    let f = frustum(10.0, 10.0, 40.0, 20.0);
    assert!((vol_of(&down, id) - f).abs() / f < 1e-3, "{}", vol_of(&down, id));
    assert!((vol_of(&down, id) - vol_of(&up, id)).abs() < 1e-6 * f);
    // The full 40 × 20 at z = 0 both times, the narrow end away from it.
    assert!((cap_at(&down, id, 0.0).unwrap_or(0.0) - 800.0).abs() < 1e-3);
    assert!(cap_at(&down, id, -10.0).is_some_and(|a| a < 800.0));
    assert!((cap_at(&up, id, 0.0).unwrap_or(0.0) - 800.0).abs() < 1e-3);
}

#[test]
fn a_taper_leans_both_ways_from_the_sketch_plane_when_it_reaches_both_ways() {
    let run = |extent: ExtrudeExtent| {
        let mut part = Part::default();
        let s = rect(&mut part, 40.0, 20.0, [0.0, 0.0], 0.0);
        part.add(ext(&s, extent, 10.0, BodyOp::New));
        build(&part)
    };
    let id = BodyId::new(2);
    let sym = run(ExtrudeExtent::Symmetric { total: 20.0 });
    ok(&sym);
    assert_eq!(sym.bodies.len(), 1);
    let f10 = frustum(10.0, 10.0, 40.0, 20.0);
    assert!((vol_of(&sym, id) - 2.0 * f10).abs() / f10 < 2e-3, "{}", vol_of(&sym, id));
    // Widest at the sketch plane: as wide as the rectangle there.
    let bb = sym.bodies[0].body.outline_bounds();
    assert!((bb.max.x - bb.min.x - 40.0).abs() < 1e-4);
    let both = run(dist(-4.0, 10.0));
    ok(&both);
    assert!((vol_of(&both, id) - f10 - frustum(4.0, 10.0, 40.0, 20.0)).abs() / f10 < 2e-3);
}

#[test]
fn a_taper_starts_leaning_at_an_offset_start_and_leans_out_with_a_negative_draft() {
    let id = BodyId::new(2);
    let mut part = Part::default();
    let s = rect(&mut part, 40.0, 20.0, [0.0, 0.0], 0.0);
    part.add(ext(&s, dist(5.0, 15.0), 10.0, BodyOp::New));
    let off = build(&part);
    ok(&off);
    assert!((cap_at(&off, id, 5.0).unwrap_or(0.0) - 800.0).abs() < 1e-3);
    let f10 = frustum(10.0, 10.0, 40.0, 20.0);
    assert!((vol_of(&off, id) - f10).abs() / f10 < 1e-3);
    let mut part = Part::default();
    let s = rect(&mut part, 40.0, 20.0, [0.0, 0.0], 0.0);
    part.add(ext(&s, dist(0.0, 30.0), -10.0, BodyOp::New));
    let out = build(&part);
    ok(&out);
    let t = 30.0 * rad(10.0).tan();
    assert!((cap_at(&out, id, 30.0).unwrap_or(0.0) - (40.0 + 2.0 * t) * (20.0 + 2.0 * t)).abs() < 0.5);
}

#[test]
fn a_taper_reaches_a_plane_straight_or_tilted_leaning_all_the_way_to_it() {
    let id = BodyId::new(2);
    let mut part = Part::default();
    plane_element(&mut part, 10, [0.0, 0.0, 15.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
    let s = rect(&mut part, 40.0, 20.0, [-20.0, -10.0], 0.0);
    part.add(ext(&s, ExtrudeExtent::ToPlane { from: 0.0, plane: plane_ref(10) }, 10.0, BodyOp::New));
    let flat = build(&part);
    ok(&flat);
    let f15 = frustum(15.0, 10.0, 40.0, 20.0);
    assert!((vol_of(&flat, id) - f15).abs() / f15 < 2e-3, "{}", vol_of(&flat, id));
    assert!((flat.bodies[0].body.outline_bounds().max.z - 15.0).abs() < 1e-6, "{:?} {:?}", flat.bodies[0].body.outline_bounds(), caps_of(&flat, id));
    // Tilted 20° about Y and leaned out: the high side of the plane is further off than any
    // corner of the sketch reaches straight up.
    let mut part = Part::default();
    plane_element(&mut part, 11, [0.0, 0.0, 15.0], [rad(20.0).sin(), 0.0, rad(20.0).cos()], [rad(20.0).cos(), 0.0, -rad(20.0).sin()]);
    let s = rect(&mut part, 40.0, 20.0, [-20.0, -10.0], 0.0);
    part.add(ext(&s, ExtrudeExtent::ToPlane { from: 0.0, plane: plane_ref(11) }, -15.0, BodyOp::New));
    let tilted = build(&part);
    ok(&tilted);
    // The top is on the plane everywhere: no cap of the prism is left standing short of it.
    let caps: Vec<f64> = caps_of(&tilted, id).into_iter().map(|c| c.0).collect();
    assert_eq!(caps, vec![0.0]);
}

#[test]
fn a_taper_reaches_a_face_of_a_body_leaning_from_the_sketch_down_to_it() {
    let mut part = Part::default();
    let block = rect(&mut part, 40.0, 20.0, [0.0, 0.0], 0.0);
    part.add(ext(&block, dist(0.0, 10.0), 0.0, BodyOp::New));
    let first = build(&part);
    let top = first.bodies[0].topology.faces.iter().find(|f| f.fingerprint.kind == FaceKind::Plane && f.fingerprint.dir.z > 0.9 && (f.fingerprint.centroid.z - 10.0).abs() < 1e-6).map(|f| f.fingerprint).expect("the plate's top");
    // A 10 × 10 square 30 mm up, reaching down to the plate's top: 20 mm of frustum, the square
    // at the sketch and narrower at the plate.
    let square = rect(&mut part, 10.0, 10.0, [15.0, 5.0], 30.0);
    let post = part.add(ext(&square, ExtrudeExtent::ToFace { from: 0.0, body: BodyId::new(2), face: top }, 10.0, BodyOp::New));
    let built = build(&part);
    ok(&built);
    let id = BodyId::new(post);
    let f20 = frustum(20.0, 10.0, 10.0, 10.0);
    assert!((vol_of(&built, id) - f20).abs() / f20 < 2e-3, "{} {f20}", vol_of(&built, id));
    assert!((cap_at(&built, id, 30.0).unwrap_or(0.0) - 100.0).abs() < 1e-3);
    assert!(cap_at(&built, id, 10.0).is_some_and(|a| a < 100.0), "{:?}", caps_of(&built, id));
}

#[test]
fn refuses_a_taper_that_closes_the_profile_saying_where() {
    let mut part = Part::default();
    let s = rect(&mut part, 40.0, 20.0, [0.0, 0.0], 0.0);
    let f = part.add(ext(&s, dist(0.0, 120.0), 10.0, BodyOp::New));
    let built = build(&part);
    // The 20 mm side closes at 10 / tan 10° = 56.71 mm.
    let m = built.errors.get(&f).cloned().unwrap_or_default();
    assert!(m.contains("closes the profile 56.7") && m.contains("mm from where it starts, and the extrude runs 120.00 mm"), "{m}");
}

#[test]
fn refuses_a_taper_leaning_out_that_closes_a_hole_of_the_profile() {
    let run = |to: f64| {
        let mut part = Part::default();
        let mut s = Sketch::default();
        s.add_rectangle(DVec2::new(0.0, 0.0), DVec2::new(40.0, 20.0));
        s.add_shape(Shape::Circle { c: DVec2::new(20.0, 10.0), r: 3.0 });
        let holed = find_regions(&s).into_iter().find(|r| !r.holes.is_empty()).map(|r| r.key).expect("a region with a hole");
        let sk = part.add(FeatureKind::Sketch(SketchFeature::on_world(WorldAxis::Z, 0.0, s)));
        let f = part.add(ext(&(sk, vec![holed]), dist(0.0, to), -10.0, BodyOp::New));
        (f, build(&part))
    };
    let (f, built) = run(30.0);
    // A hole of radius 3 closes at 3 / tan 10° = 17.01 mm.
    let m = built.errors.get(&f).cloned().unwrap_or_default();
    assert!(m.contains("closes a hole or a notch of the profile 17.0"), "{m}");
    let (_, fine) = run(15.0);
    ok(&fine);
}

#[test]
fn cuts_through_all_with_a_taper_leaning_from_the_sketch() {
    let mut part = Part::default();
    let block = rect(&mut part, 40.0, 20.0, [0.0, 0.0], 0.0);
    part.add(ext(&block, dist(0.0, 10.0), 0.0, BodyOp::Join));
    let pocket = rect(&mut part, 10.0, 10.0, [15.0, 5.0], 10.0);
    part.add(ext(&pocket, ExtrudeExtent::ToLast { from: 0.0, reverse: true }, 10.0, BodyOp::Cut));
    let built = build(&part);
    ok(&built);
    // A pyramid's frustum out of the block: 10 × 10 at the top, narrower through to the bottom.
    let v = vol_of(&built, BodyId::new(2));
    assert!(v < 8000.0 - 10.0 * (10.0 - 2.0 * 10.0 * rad(10.0).tan()).powi(2), "{v}");
    assert!(v > 8000.0 - 1000.0, "{v}");
}

// ---- a draft, planes (reBuild.test.ts) ----

#[test]
fn extrudes_with_a_draft_the_sides_lean_in_and_a_negative_draft_leans_them_out() {
    let run = |draft: f64| {
        let mut part = Part::default();
        let s = rect(&mut part, 40.0, 20.0, [0.0, 0.0], 0.0);
        part.add(ext(&s, dist(0.0, 10.0), draft, BodyOp::New));
        build(&part)
    };
    let id = BodyId::new(2);
    let (straight, drafted, flared) = (run(0.0), run(10.0), run(-10.0));
    ok(&drafted);
    let d = 10.0 * rad(10.0).tan();
    let f = frustum(10.0, 10.0, 40.0, 20.0);
    assert!((vol_of(&drafted, id) - f).abs() / f < 1e-3);
    assert!(vol_of(&drafted, id) < vol_of(&straight, id));
    assert!(vol_of(&flared, id) > vol_of(&straight, id));
    // The top face sits where the draft leaves it, 10 mm up.
    let top = cap_at(&drafted, id, 10.0).unwrap_or(0.0);
    assert!((top - (40.0 - 2.0 * d) * (20.0 - 2.0 * d)).abs() / top < 1e-3);
}

#[test]
fn reaches_to_a_plane_and_between_two_planes() {
    let mut part = Part::default();
    plane_element(&mut part, 10, [0.0, 0.0, 15.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
    plane_element(&mut part, 11, [0.0, 0.0, -5.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
    let s = rect(&mut part, 40.0, 20.0, [0.0, 0.0], 0.0);
    let a = part.add(ext(&s, ExtrudeExtent::ToPlane { from: 0.0, plane: plane_ref(10) }, 0.0, BodyOp::New));
    let b = part.add(ext(&s, ExtrudeExtent::BetweenPlanes { a: plane_ref(11), b: plane_ref(10) }, 0.0, BodyOp::New));
    let built = build(&part);
    ok(&built);
    assert!((vol_of(&built, BodyId::new(a)) - 40.0 * 20.0 * 15.0).abs() < 1e-2);
    assert!((vol_of(&built, BodyId::new(b)) - 40.0 * 20.0 * 20.0).abs() < 1e-2);
    // A plane behind the sketch is reached the way it lies.
    let mut part = Part::default();
    plane_element(&mut part, 11, [0.0, 0.0, -5.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
    let s = rect(&mut part, 40.0, 20.0, [0.0, 0.0], 0.0);
    let c = part.add(ext(&s, ExtrudeExtent::ToPlane { from: 0.0, plane: plane_ref(11) }, 0.0, BodyOp::New));
    let built = build(&part);
    ok(&built);
    assert!((vol_of(&built, BodyId::new(c)) - 40.0 * 20.0 * 5.0).abs() < 1e-2);
    // Two planes both above the sketch: the slab between them.
    let mut part = Part::default();
    plane_element(&mut part, 10, [0.0, 0.0, 15.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
    plane_element(&mut part, 12, [0.0, 0.0, 5.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]);
    let s = rect(&mut part, 40.0, 20.0, [0.0, 0.0], 0.0);
    let d = part.add(ext(&s, ExtrudeExtent::BetweenPlanes { a: plane_ref(12), b: plane_ref(10) }, 0.0, BodyOp::New));
    let built = build(&part);
    ok(&built);
    assert!((vol_of(&built, BodyId::new(d)) - 40.0 * 20.0 * 10.0).abs() < 1e-2);
}

#[test]
fn trims_an_extrusion_to_a_tilted_plane_with_a_half_space() {
    let mut part = Part::default();
    let h = std::f64::consts::FRAC_1_SQRT_2;
    plane_element(&mut part, 12, [0.0, 0.0, 10.0], [h, 0.0, h], [h, 0.0, -h]);
    let s = rect(&mut part, 20.0, 20.0, [-10.0, -10.0], 0.0);
    let f = part.add(ext(&s, ExtrudeExtent::ToPlane { from: 0.0, plane: plane_ref(12) }, 0.0, BodyOp::New));
    let built = build(&part);
    ok(&built);
    // The plane x + z = 10 over the square −10..10: the height runs from 20 at x = −10 to 0 at
    // x = 10, so the volume is the square's area times the mean height of 10.
    assert!((vol_of(&built, BodyId::new(f)) - 20.0 * 20.0 * 10.0).abs() < 1e-2);
    assert!((built.bodies[0].body.outline_bounds().max.z - 20.0).abs() < 1e-4);
}

#[test]
fn an_extent_with_nothing_chosen_fails_saying_what_to_choose() {
    let mut part = Part::default();
    let s = rect(&mut part, 40.0, 20.0, [0.0, 0.0], 0.0);
    let a = part.add(ext(&s, ExtrudeExtent::ToPlane { from: 0.0, plane: no_plane() }, 0.0, BodyOp::New));
    let b = part.add(ext(&s, ExtrudeExtent::ToBody { from: 0.0, body: NO_BODY, reverse: false }, 0.0, BodyOp::New));
    let gone = part.add(ext(&s, ExtrudeExtent::ToPlane { from: 0.0, plane: plane_ref(99) }, 0.0, BodyOp::New));
    let built = build(&part);
    assert_eq!(built.errors.get(&a).map(String::as_str), Some("Choose the plane the extrude reaches to."));
    assert_eq!(built.errors.get(&b).map(String::as_str), Some("Choose the body the extrude reaches to."));
    assert!(built.errors.get(&gone).is_some_and(|m| m.starts_with("The plane this extrusion reaches to is gone")));
}

// ---- a face of a body pulled as the profile (reBuild.test.ts) ----

#[test]
fn pulls_a_face_as_it_stands_follows_it_when_an_earlier_feature_reshapes_it_and_fails_by_name_when_it_is_gone() {
    let block = |part: &mut Part, to: f64| {
        let s = rect(part, 40.0, 20.0, [0.0, 0.0], 0.0);
        part.add(ext(&s, dist(0.0, to), 0.0, BodyOp::New));
    };
    let mut first = Part::default();
    block(&mut first, 10.0);
    let built = build(&first);
    let top = built.bodies[0].topology.faces.iter().find(|f| f.fingerprint.kind == FaceKind::Plane && (f.fingerprint.centroid.z - 10.0).abs() < 1e-6).map(|f| f.fingerprint).expect("the top");
    let pull = |to: f64, op: BodyOp| {
        let mut e = ExtrudeFeature::new(0, Vec::new(), dist(0.0, to), op);
        e.face = Some(FaceProfile { body: BodyId::new(2), face: top });
        e.target = Some(BodyId::new(2));
        FeatureKind::Extrude(e)
    };
    // Out of the block: a boss the size of the face, joined.
    let mut out = first.clone();
    out.add(pull(5.0, BodyOp::Join));
    let built = build(&out);
    ok(&built);
    assert!((vol_of(&built, BodyId::new(2)) - 40.0 * 20.0 * 15.0).abs() < 1e-2);
    assert_eq!(built.bodies[0].body.face_count(), 6);
    // Into the block: a cut the size of the face.
    let mut into = first.clone();
    into.add(pull(-3.0, BodyOp::Cut));
    let built = build(&into);
    ok(&built);
    assert!((vol_of(&built, BodyId::new(2)) - 40.0 * 20.0 * 7.0).abs() < 1e-2);
    // The block made taller: the pull stands on the new top, not where the face was.
    let mut taller = Part::default();
    block(&mut taller, 12.0);
    taller.add(pull(5.0, BodyOp::Join));
    let built = build(&taller);
    ok(&built);
    assert!((vol_of(&built, BodyId::new(2)) - 40.0 * 20.0 * 17.0).abs() < 1e-2);
    // A hole cut through the block first: the top face has a hole, and so has what is pulled
    // from it — nothing was copied.
    let mut holed = Part::default();
    block(&mut holed, 10.0);
    let mut hole = Sketch::default();
    hole.add_shape(Shape::Circle { c: DVec2::new(10.0, 10.0), r: 3.0 });
    let regions = find_regions(&hole).into_iter().map(|r| r.key).collect();
    let hs = holed.add(FeatureKind::Sketch(SketchFeature::on_world(WorldAxis::Z, 0.0, hole)));
    let mut drill = ExtrudeFeature::new(hs, regions, dist(-1.0, 11.0), BodyOp::Cut);
    drill.target = Some(BodyId::new(2));
    holed.add(FeatureKind::Extrude(drill));
    holed.add(pull(5.0, BodyOp::Join));
    let built = build(&holed);
    ok(&built);
    let v = vol_of(&built, BodyId::new(2));
    assert!((v - (40.0 * 20.0 * 15.0 - std::f64::consts::PI * 9.0 * 15.0)).abs() < 0.2, "{v}");
    // The face gone — the block far taller — the pull fails by name.
    let mut gone = Part::default();
    block(&mut gone, 60.0);
    let p = gone.add(pull(5.0, BodyOp::Join));
    let built = build(&gone);
    assert!(built.errors.get(&p).is_some_and(|m| m.contains("face this extrude pulls is gone")), "{:?}", built.errors);
}

// ---- to a face, to next, to last (reFeatures.test.ts) ----

const BOX: f64 = 8000.0;

/// The 40 × 20 × 10 plate (feature 2), a sketch 30 up (feature 3) holding circles of r 4 at
/// `at`, and an extrude of them (feature 4) reaching `extent`; a join or a cut works on the plate.
fn plate_and(at: &[[f64; 2]], extent: ExtrudeExtent, op: BodyOp) -> (Built, FeatureId) {
    let mut part = Part::default();
    let s = rect(&mut part, 40.0, 20.0, [0.0, 0.0], 0.0);
    part.add(ext(&s, dist(0.0, 10.0), 0.0, BodyOp::New));
    let mut sk = Sketch::default();
    for c in at {
        sk.add_shape(Shape::Circle { c: DVec2::from(*c), r: 4.0 });
    }
    let regions = find_regions(&sk).into_iter().map(|r| r.key).collect();
    let over = part.add(FeatureKind::Sketch(SketchFeature::on_world(WorldAxis::Z, 30.0, sk)));
    let mut e = ExtrudeFeature::new(over, regions, extent, op);
    if op != BodyOp::New {
        e.target = Some(BodyId::new(2));
    }
    let f = part.add(FeatureKind::Extrude(e));
    (build(&part), f)
}

#[test]
fn extrudes_to_the_next_face_and_to_the_last_as_a_boss_and_as_a_cut() {
    let disc = std::f64::consts::PI * 16.0;
    let next = |reverse: bool| ExtrudeExtent::ToNext { from: 0.0, reverse };
    let last = ExtrudeExtent::ToLast { from: 0.0, reverse: true };
    let plate = BodyId::new(2);
    // The kernel measures a body off its tessellation: a round's chords read a hair low.
    let close = |a: f64, b: f64| (a - b).abs() < 2e-4 * b.abs().max(1.0);
    // To next, down onto the plate: a pin from the sketch to the plate's top.
    let (built, f) = plate_and(&[[20.0, 10.0]], next(true), BodyOp::Join);
    assert!(built.errors.get(&f).is_none(), "{:?}", built.errors);
    assert!(close(vol_of(&built, plate), BOX + disc * 20.0), "{}", vol_of(&built, plate));
    // The way the sketch looks there is nothing.
    let (built, f) = plate_and(&[[20.0, 10.0]], next(false), BodyOp::Join);
    assert!(built.errors.get(&f).is_some_and(|m| m.contains("reverse the direction")), "{:?}", built.errors);
    // Half off the plate, nothing stops the half that misses.
    let (built, f) = plate_and(&[[40.0, 10.0]], next(true), BodyOp::Join);
    assert!(built.errors.get(&f).is_some_and(|m| m.contains("does not cover the whole profile")), "{:?}", built.errors);
    // To last as a boss runs to the plate's far side; as a cut, through it.
    let (built, _) = plate_and(&[[20.0, 10.0]], last.clone(), BodyOp::Join);
    assert!(close(vol_of(&built, plate), BOX + disc * 20.0));
    let (built, _) = plate_and(&[[20.0, 10.0]], last, BodyOp::Cut);
    assert!(close(vol_of(&built, plate), BOX - disc * 10.0), "{}", vol_of(&built, plate));
    // To next as a cut goes through the first wall it meets.
    let (built, _) = plate_and(&[[20.0, 10.0]], next(true), BodyOp::Cut);
    assert!(close(vol_of(&built, plate), BOX - disc * 10.0));
    // A new body to next is measured against every body, and stands on the plate.
    let (built, f) = plate_and(&[[20.0, 10.0]], next(true), BodyOp::New);
    assert!(close(vol_of(&built, BodyId::new(f)), disc * 20.0));
    // Three circles, each to its own next face: three bosses, three holes.
    let three = [[8.0, 10.0], [20.0, 10.0], [32.0, 10.0]];
    let (built, f) = plate_and(&three, next(true), BodyOp::Join);
    assert!(built.errors.get(&f).is_none(), "{:?}", built.errors);
    assert_eq!(built.bodies.len(), 1);
    assert!(close(vol_of(&built, plate), BOX + 3.0 * disc * 20.0), "{} {:?}", vol_of(&built, plate), built.bodies.iter().map(|b| (b.id, volume(&b.body))).collect::<Vec<_>>());
    let (built, _) = plate_and(&three, next(true), BodyOp::Cut);
    assert!(close(vol_of(&built, plate), BOX - 3.0 * disc * 10.0));
    // One of them off the plate: refused, and the one named.
    let (built, f) = plate_and(&[[8.0, 10.0], [40.0, 10.0]], next(true), BodyOp::Join);
    let m = built.errors.get(&f).cloned().unwrap_or_default();
    assert!(m.starts_with("The profile's region about (40.00, 10.00, 30.00): The body does not cover the whole profile"), "{m}");
}

#[test]
fn extrudes_to_a_flat_face_of_a_body_as_to_its_plane() {
    let disc = std::f64::consts::PI * 16.0;
    let (built, _) = plate_and(&[[20.0, 10.0]], ExtrudeExtent::ToNext { from: 0.0, reverse: true }, BodyOp::Join);
    let top = built.bodies[0].topology.faces.iter().find(|f| f.fingerprint.kind == FaceKind::Plane && f.fingerprint.dir.z > 0.9 && (f.fingerprint.centroid.z - 10.0).abs() < 1e-6).map(|f| f.fingerprint).expect("the top");
    let (built, f) = plate_and(&[[20.0, 10.0]], ExtrudeExtent::ToFace { from: 0.0, body: BodyId::new(2), face: top }, BodyOp::Join);
    assert!(built.errors.get(&f).is_none(), "{:?}", built.errors);
    assert!((vol_of(&built, BodyId::new(2)) - (BOX + disc * 20.0)).abs() < 0.5);
}

// ---- up to a curved face, or a body (reExtrudeToFace.test.ts) ----

/// The box 40 × 20 × 20 from the origin (feature 2); the cylinder R30 about the axis through
/// (60, 10) along z, `height` tall from z = 0 (feature 4, named Cylinder).
fn box_and_cylinder(height: f64) -> Part {
    let mut part = Part::default();
    let b = rect(&mut part, 40.0, 20.0, [0.0, 0.0], 0.0);
    part.add_named(ext(&b, dist(0.0, 20.0), 0.0, BodyOp::New), "Box".into());
    let mut ring = Sketch::default();
    ring.add_shape(Shape::Circle { c: DVec2::new(60.0, 10.0), r: 30.0 });
    let regions = find_regions(&ring).into_iter().map(|r| r.key).collect();
    let rs = part.add(FeatureKind::Sketch(SketchFeature::on_world(WorldAxis::Z, 0.0, ring)));
    part.add_named(ext(&(rs, regions), dist(0.0, height), 0.0, BodyOp::New), "Cylinder".into());
    part
}

/// The side plane x = 0, its sheet's u along y and v along z: the profile y 2…18, z 2…18 (feature
/// 5), and a cut of the box (feature 6) reaching `extent`.
fn side_cut(part: &mut Part, extent: ExtrudeExtent) -> FeatureId {
    let yz = SectionFrame { origin: DVec3::ZERO, normal: DVec3::X, u: DVec3::Y, v: DVec3::Z };
    let mut s = Sketch::default();
    s.add_rectangle(DVec2::new(2.0, 2.0), DVec2::new(18.0, 18.0));
    let regions = find_regions(&s).into_iter().map(|r| r.key).collect();
    let side = part.add(on_frame(yz, s));
    let mut e = ExtrudeFeature::new(side, regions, extent, BodyOp::Cut);
    e.target = Some(BodyId::new(2));
    part.add(FeatureKind::Extrude(e))
}

/// The volume the cut takes: the profile run from x = 0 to the cylinder.
fn wanted(z0: f64, z1: f64) -> f64 {
    let n = 4000;
    let mut v = 0.0;
    for i in 0..n {
        let y = 2.0 + ((i as f64 + 0.5) * 16.0) / n as f64;
        v += (60.0 - (900.0 - (y - 10.0).powi(2)).sqrt()) * (16.0 / n as f64);
    }
    v * (z1 - z0)
}

fn cylinder_face(height: f64) -> FaceRef {
    let built = build(&box_and_cylinder(height));
    built.body(BodyId::new(4)).and_then(|b| b.topology.faces.iter().find(|f| f.fingerprint.kind == FaceKind::Cylinder).map(|f| f.fingerprint)).expect("the cylinder's face")
}

#[test]
fn cuts_a_box_up_to_a_cylinder_face_the_floor_its_arc() {
    let face = cylinder_face(20.0);
    let mut part = box_and_cylinder(20.0);
    let cut = side_cut(&mut part, ExtrudeExtent::ToFace { from: 0.0, body: BodyId::new(4), face });
    let built = build(&part);
    assert!(built.errors.get(&cut).is_none(), "{:?}", built.errors);
    let b = built.body(BodyId::new(2)).expect("the box");
    let v = volume(&b.body).unwrap_or(0.0);
    assert!((v - (16_000.0 - wanted(2.0, 18.0))).abs() < 0.5, "{v} {}", 16_000.0 - wanted(2.0, 18.0));
    // The pocket's floor lies on the cylinder; its walls are the profile's own, and nothing
    // stands past x = 31.1.
    let faces: Vec<FaceRef> = b.topology.faces.iter().map(|f| f.fingerprint).collect();
    // One cylinder face in Open CASCADE; the kernel keeps a cylinder's wall in two halves, and
    // the floor runs across where they meet.
    let floor: Vec<&FaceRef> = faces.iter().filter(|f| f.kind == FaceKind::Cylinder && f.origin == cut as i64).collect();
    assert!((1..=2).contains(&floor.len()), "{floor:?}");
    for f in floor {
        assert!(f.radius.is_some_and(|r| (r - 30.0).abs() < 1e-6));
    }
    let walls: Vec<&FaceRef> = faces.iter().filter(|f| f.origin == cut as i64 && f.kind == FaceKind::Plane).collect();
    assert_eq!(walls.len(), 4);
    for w in walls {
        assert!(w.centroid.x < 31.1);
    }
}

#[test]
fn cuts_the_same_up_to_the_cylinder_as_a_body() {
    let mut part = box_and_cylinder(20.0);
    let cut = side_cut(&mut part, ExtrudeExtent::ToBody { from: 0.0, body: BodyId::new(4), reverse: false });
    let built = build(&part);
    assert!(built.errors.get(&cut).is_none(), "{:?}", built.errors);
    let v = vol_of(&built, BodyId::new(2));
    assert!((v - (16_000.0 - wanted(2.0, 18.0))).abs() < 0.5, "{v}");
}

#[test]
fn refuses_a_profile_the_face_does_not_lie_under_saying_so() {
    // The cylinder 10 tall; the profile runs to z 18.
    let face = cylinder_face(10.0);
    let mut part = box_and_cylinder(10.0);
    let cut = side_cut(&mut part, ExtrudeExtent::ToFace { from: 0.0, body: BodyId::new(4), face });
    let built = build(&part);
    let m = built.errors.get(&cut).cloned().unwrap_or_default();
    assert!(m.contains("does not cover the whole profile"), "{m}");
    assert!(m.contains("cylinder face of Cylinder"), "{m}");
    // How far short, for a surface's margin to be grown by that and not by guess.
    assert!(m.contains("mm short along z — about ") && m.contains("% of its width"), "{m}");
    // The box is left as it was.
    assert!((vol_of(&built, BodyId::new(2)) - 16_000.0).abs() < 1e-3);
}

#[test]
fn refuses_a_body_that_lies_beside_the_profile() {
    let mut part = box_and_cylinder(20.0);
    let far = rect(&mut part, 20.0, 20.0, [30.0, 40.0], 0.0);
    let block = part.add_named(ext(&far, dist(0.0, 20.0), 0.0, BodyOp::New), "Far block".into());
    let cut = side_cut(&mut part, ExtrudeExtent::ToBody { from: 0.0, body: BodyId::new(block), reverse: false });
    let built = build(&part);
    assert!(built.errors.get(&cut).is_some_and(|m| m.contains("never meets Far block")), "{:?}", built.errors);
}

// ---- along a direction of its own (reCommandsScan.test.ts) ----

#[test]
fn extrudes_a_level_sketch_along_a_direction_of_its_own_an_oblique_prism_a_taper_leaning_about_it() {
    let lean = |direction: Option<DVec3>, draft: f64| {
        let mut part = Part::default();
        let s = rect(&mut part, 10.0, 10.0, [-5.0, -5.0], 0.0);
        let mut e = ExtrudeFeature::new(s.0, s.1, dist(0.0, 10.0), BodyOp::New);
        e.direction = direction;
        e.draft_deg = draft;
        let f = part.add(FeatureKind::Extrude(e));
        (build(&part), f)
    };
    let d = DVec3::new(0.2, 0.0, 1.0);
    let n = d.length();
    let up = 10.0 / n;
    let (built, f) = lean(Some(d / n), 0.0);
    assert!(built.errors.get(&f).is_none(), "{:?}", built.errors);
    let body = built.body(BodyId::new(f)).expect("the body");
    // Ten along the direction: as high as that over the plane, the base the square still, the
    // top moved over by the lean.
    assert!((volume(&body.body).unwrap_or(0.0) - 100.0 * up).abs() < 1e-3);
    let bb = body.body.outline_bounds();
    assert!((dv3(bb.min) - DVec3::new(-5.0, -5.0, 0.0)).length() < 1e-6, "{:?}", bb);
    assert!((dv3(bb.max) - DVec3::new(5.0 + 2.0 / n, 5.0, up)).length() < 1e-6, "{:?}", bb);
    let top = body.topology.faces.iter().find(|x| x.fingerprint.kind == FaceKind::Plane && x.fingerprint.dir.z > 0.9 && x.fingerprint.centroid.z > 1.0).map(|x| x.fingerprint).expect("the top");
    assert!((top.area - 100.0).abs() < 1e-6);
    assert!((top.centroid - DVec3::new(2.0 / n, 0.0, up)).length() < 1e-6);
    // Tapered 5° in: the top smaller, and every side leaning 5° off the direction — about it,
    // not about the plane's normal.
    let (built, f) = lean(Some(d / n), 5.0);
    assert!(built.errors.get(&f).is_none(), "{:?}", built.errors);
    let body = built.body(BodyId::new(f)).expect("the body");
    let top = body.topology.faces.iter().find(|x| x.fingerprint.kind == FaceKind::Plane && x.fingerprint.dir.z > 0.9 && x.fingerprint.centroid.z > 1.0).map(|x| x.fingerprint).expect("the top");
    assert!(top.area < 75.0, "{}", top.area);
    let sides: Vec<FaceRef> = body.topology.faces.iter().map(|x| x.fingerprint).filter(|x| x.kind == FaceKind::Plane && x.centroid.z > 1.0 && x.centroid.z < up - 1.0).collect();
    assert_eq!(sides.len(), 4);
    for s in sides {
        assert!((s.dir.dot(d) / n - rad(5.0).sin()).abs() < 1e-6, "{:?}", s.dir);
    }
    // Back to the plane's normal: the straight block.
    let (built, f) = lean(None, 0.0);
    assert!((vol_of(&built, BodyId::new(f)) - 1000.0).abs() < 1e-6);
    // A direction in the plane sweeps nothing, and says so.
    let (built, f) = lean(Some(DVec3::X), 0.0);
    assert!(built.errors.get(&f).is_some_and(|m| m.contains("along the sketch plane")));
}

#[test]
fn reads_the_ghosts_volume_off_the_solid_the_extrude_makes() {
    let mut part = Part::default();
    let block = rect(&mut part, 40.0, 20.0, [0.0, 0.0], 0.0);
    part.add(ext(&block, dist(0.0, 10.0), 0.0, BodyOp::New));
    let s = rect(&mut part, 10.0, 10.0, [15.0, 5.0], 10.0);
    let f = part.add(ext(&s, dist(-2.0, 5.0), 0.0, BodyOp::Join));
    let built = build(&part);
    ok(&built);
    assert!(built.ghost_volumes.get(&f).is_some_and(|v| (v - 700.0).abs() < 1e-6));
    assert!((vol_of(&built, BodyId::new(2)) - 8500.0).abs() < 1e-3);
}

#[test]
fn a_taper_with_a_symmetric_extent_or_a_plane_needs_no_start_at_the_sketch_plane() {
    // What was refused before (a taper had to start at the sketch plane) builds now.
    let mut part = Part::default();
    let s = rect(&mut part, 20.0, 20.0, [-10.0, -10.0], 0.0);
    let f = part.add(ext(&s, dist(-5.0, 5.0), 3.0, BodyOp::New));
    let built = build(&part);
    ok(&built);
    assert!(vol_of(&built, BodyId::new(f)) < 4000.0);
}

