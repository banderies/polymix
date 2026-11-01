// SPDX-License-Identifier: AGPL-3.0-only
//! Tests of Mirror, Combine, Cut by plane and Offset body — the cases of ScanRuler's
//! `reBodyTools.test.ts` (a body offset all round), `reAutoBodies.test.ts` (a Combine naming
//! a body left in pieces), `reSheetTrim.test.ts` and `reFeatures.test.ts`, © Stefan Hermann
//! (CNC Kitchen), AGPL-3.0-only.

use glam::{DVec2, DVec3};
use solidcraft_geom::{Vec2 as SVec2, Vec3 as SVec3};

use super::*;
use crate::frame::WorldAxis;
use crate::model::*;
use crate::regions::find_regions;
use crate::sketch::Sketch;

/// A sketch of rectangles on the XY plane, each [x0, y0, x1, y1] (reKit's `rects`).
fn rects(part: &mut Part, boxes: &[[f64; 4]]) -> (FeatureId, Vec<String>) {
    let mut s = Sketch::default();
    for [x0, y0, x1, y1] in boxes {
        s.add_rectangle(DVec2::new(*x0, *y0), DVec2::new(*x1, *y1));
    }
    let regions = find_regions(&s).iter().map(|r| r.key.clone()).collect();
    (part.add(FeatureKind::Sketch(SketchFeature::on_world(WorldAxis::Z, 0.0, s))), regions)
}

fn extrude(part: &mut Part, sketch: &(FeatureId, Vec<String>), from: f64, to: f64, op: BodyOp) -> FeatureId {
    part.add(FeatureKind::Extrude(ExtrudeFeature::new(sketch.0, sketch.1.clone(), ExtrudeExtent::Distance { from, to }, op)))
}

fn ids(built: &Built) -> Vec<BodyId> {
    built.bodies.iter().map(|b| b.id).collect()
}

fn volumes(built: &Built) -> Vec<f64> {
    built.bodies.iter().map(|b| (volume(&b.body).unwrap_or(f64::NAN) * 1000.0).round() / 1000.0).collect()
}

fn ok(built: &Built) {
    assert!(built.errors.is_empty(), "{:?}", built.errors);
}

// ---- a body offset all round (reBodyTools.test.ts) ----

#[test]
fn grows_a_box_by_a_distance_into_a_body_of_its_own_rounded_over_its_edges_the_box_kept() {
    let mut part = Part::default();
    let s = rects(&mut part, &[[0.0, 0.0, 10.0, 6.0]]);
    extrude(&mut part, &s, 0.0, 4.0, BodyOp::New);
    part.add(FeatureKind::OffsetBody(OffsetBodyFeature { body: BodyId::new(2), distance: 1.0, keep: true, exprs: Exprs::new() }));
    let built = build(&part);
    ok(&built);
    assert_eq!(ids(&built), vec![BodyId::new(2), BodyId::new(3)]);
    assert!((volumes(&built)[0] - 240.0).abs() < 1e-6);
    // A box grown by r: the box, its faces pushed out, quarter-cylinders along its edges and
    // eighth-spheres at its corners.
    let (a, b, c, r) = (10.0, 6.0, 4.0, 1.0);
    let want = a * b * c + 2.0 * r * (a * b + b * c + a * c) + std::f64::consts::PI * r * r * (a + b + c) + 4.0 / 3.0 * std::f64::consts::PI * r * r * r;
    // Measured on a fine tessellation: the kernel's volume is its mesh's, a chord short on
    // each round.
    let got = built.bodies[1].body.volume_at(1e-4).unwrap_or(0.0);
    assert!((got - want).abs() < 0.01, "{got} against {want}");
    assert!(built.bodies[1].body.validity().is_empty(), "{:?}", built.bodies[1].body.validity());
}

#[test]
fn shrinks_a_body_in_place_and_refuses_an_inward_offset_through_it() {
    let mut part = Part::default();
    let s = rects(&mut part, &[[0.0, 0.0, 10.0, 6.0]]);
    extrude(&mut part, &s, 0.0, 4.0, BodyOp::New);
    let o = part.add(FeatureKind::OffsetBody(OffsetBodyFeature { body: BodyId::new(2), distance: -1.0, keep: false, exprs: Exprs::new() }));
    let built = build(&part);
    ok(&built);
    assert_eq!(built.bodies.len(), 1);
    assert!((volumes(&built)[0] - 8.0 * 4.0 * 2.0).abs() < 1e-3, "{:?}", volumes(&built));
    if let Some(FeatureKind::OffsetBody(x)) = part.feature_mut(o).map(|f| &mut f.kind) {
        x.distance = -3.5;
    }
    let refused = build(&part);
    assert!(refused.errors.contains_key(&o), "{:?}", volumes(&refused));
}

#[test]
fn cut_from_a_case_leaves_a_pocket_with_the_clearance() {
    let mut part = Part::default();
    let case = rects(&mut part, &[[-5.0, -5.0, 25.0, 15.0]]);
    extrude(&mut part, &case, 0.0, 8.0, BodyOp::New);
    let inner = rects(&mut part, &[[0.0, 0.0, 10.0, 6.0]]);
    extrude(&mut part, &inner, 2.0, 12.0, BodyOp::New);
    part.add(FeatureKind::OffsetBody(OffsetBodyFeature { body: BodyId::new(4), distance: 0.5, keep: true, exprs: Exprs::new() }));
    part.add(FeatureKind::Boolean(BooleanFeature { op: BooleanOp::Cut, body: BodyId::new(2), tool: BodyId::new(5), keep: false }));
    let built = build(&part);
    ok(&built);
    // The keep-out went into the cut; the part and the case remain.
    assert_eq!(ids(&built), vec![BodyId::new(2), BodyId::new(4)]);
    // The part sits in its pocket half a millimetre clear of the case.
    let (case, inner) = (&built.bodies[0].body, &built.bodies[1].body);
    let mc = case.display_mesh(0.01).unwrap_or_default();
    let mi = inner.display_mesh(0.01).unwrap_or_default();
    let (dc, di) = (kernel::DistanceShape::new(case, &mc, 0.01), kernel::DistanceShape::new(inner, &mi, 0.01));
    let gap = kernel::body_distance(&di, &dc, f64::INFINITY).ok().flatten().map_or(f64::NAN, |d| d.distance);
    assert!((gap - 0.5).abs() < 1e-3, "{gap}");
}

#[test]
fn rounds_a_concave_edge_when_shrinking_and_a_round_face_grows_its_radius() {
    // An L in section: shrunk, the inside corner comes out round about the corner as it was.
    let mut part = Part::default();
    let s = rects(&mut part, &[[0.0, 0.0, 20.0, 10.0], [0.0, 10.0, 10.0, 20.0]]);
    extrude(&mut part, &s, 0.0, 5.0, BodyOp::Join);
    let built = build(&part);
    ok(&built);
    let l = volume(&built.bodies[0].body).unwrap_or(0.0);
    assert!((l - 1500.0).abs() < 1e-6, "{l}");
    part.add(FeatureKind::OffsetBody(OffsetBodyFeature { body: BodyId::new(2), distance: -1.0, keep: false, exprs: Exprs::new() }));
    let built = build(&part);
    ok(&built);
    // The sharp inner offset, 3 thick, and the quarter of a 1 mm round's square left at the
    // inside corner filled again.
    let sharp = (18.0 * 8.0 + 8.0 * 10.0) * 3.0;
    let filled = (1.0 - std::f64::consts::PI / 4.0) * 3.0;
    let got = volume(&built.bodies[0].body).unwrap_or(0.0);
    assert!((got - (sharp + filled)).abs() < 0.01, "{got} against {}", sharp + filled);
    // A cylinder grown by 1: radius and length up by 1 and 2, its rims rounded.
    let mut round = Part::default();
    let mut c = Sketch::default();
    c.add_shape(crate::sketch::Shape::Circle { c: DVec2::ZERO, r: 5.0 });
    let regions = find_regions(&c).iter().map(|r| r.key.clone()).collect();
    let sk = round.add(FeatureKind::Sketch(SketchFeature::on_world(WorldAxis::Z, 0.0, c)));
    round.add(FeatureKind::Extrude(ExtrudeFeature::new(sk, regions, ExtrudeExtent::Distance { from: 0.0, to: 10.0 }, BodyOp::New)));
    round.add(FeatureKind::OffsetBody(OffsetBodyFeature { body: BodyId::new(2), distance: 1.0, keep: false, exprs: Exprs::new() }));
    let built = build(&round);
    ok(&built);
    let pi = std::f64::consts::PI;
    // Pappus: the cylinder, its wall pushed out, the two caps pushed up and down, and the two
    // quarter-tori round the rims.
    let (r, h, d) = (5.0, 10.0, 1.0);
    let want = pi * (r + d) * (r + d) * h + 2.0 * pi * r * r * d + 2.0 * (pi * d * d / 4.0) * 2.0 * pi * (r + 4.0 * d / (3.0 * pi));
    let got = volume(&built.bodies[0].body).unwrap_or(0.0);
    assert!((got - want).abs() / want < 2e-3, "{got} against {want}");
}

#[test]
fn an_offset_needs_a_distance_and_a_solid() {
    let mut part = Part::default();
    let s = rects(&mut part, &[[0.0, 0.0, 10.0, 6.0]]);
    extrude(&mut part, &s, 0.0, 4.0, BodyOp::New);
    let o = part.add(FeatureKind::OffsetBody(OffsetBodyFeature { body: BodyId::new(2), distance: 0.0, keep: true, exprs: Exprs::new() }));
    assert_eq!(build(&part).errors.get(&o).map(String::as_str), Some("Offsetting: the offset needs a distance."));
}

// ---- a Combine naming a body left in pieces (reAutoBodies.test.ts) ----

fn combine(part: &mut Part, op: BooleanOp, body: BodyId, tool: BodyId, keep: bool) -> FeatureId {
    part.add(FeatureKind::Boolean(BooleanFeature { op, body, tool, keep }))
}

#[test]
fn a_combine_takes_every_piece_a_plug_patterned_into_copies_apart_cuts_all_its_windows() {
    let mut part = Part::default();
    let plate = rects(&mut part, &[[0.0, 0.0, 40.0, 40.0]]);
    extrude(&mut part, &plate, 0.0, 5.0, BodyOp::Join);
    let plug = rects(&mut part, &[[5.0, 5.0, 10.0, 10.0]]);
    extrude(&mut part, &plug, -1.0, 6.0, BodyOp::New);
    part.add(FeatureKind::Pattern(PatternFeature { body: BodyId::new(4), mode: PatternMode::Linear, axis: AxisRef::World { axis: WorldAxis::X }, count: 3, step: 10.0, axis2: None, count2: None, step2: None, exprs: Exprs::new() }));
    let patterned = build(&part);
    ok(&patterned);
    assert_eq!(patterned.bodies.len(), 4);
    let mut all = part.clone();
    combine(&mut all, BooleanOp::Cut, BodyId::new(2), BodyId::new(4), false);
    let r = build(&all);
    ok(&r);
    assert_eq!(ids(&r), vec![BodyId::new(2)]);
    assert_eq!(volumes(&r), vec![40.0 * 40.0 * 5.0 - 3.0 * 25.0 * 5.0]);
    // Naming one copy takes that copy alone.
    let copy = patterned.bodies.iter().map(|b| b.id).find(|id| id.feature == 5).unwrap_or_else(|| panic!("no copy of the pattern's"));
    let mut one = part.clone();
    combine(&mut one, BooleanOp::Cut, BodyId::new(2), copy, false);
    let r = build(&one);
    ok(&r);
    assert_eq!(r.bodies.len(), 3);
    assert!((volumes(&r)[0] - (40.0 * 40.0 * 5.0 - 25.0 * 5.0)).abs() < 1e-3);
}

#[test]
fn combines_two_profiles_extruded_together_one_moved_into_the_other_either_way_round() {
    let mut part = Part::default();
    let two = rects(&mut part, &[[0.0, 0.0, 20.0, 10.0], [22.0, 0.0, 32.0, 10.0]]);
    extrude(&mut part, &two, 0.0, 10.0, BodyOp::Join);
    let first = build(&part);
    ok(&first);
    let island = island_id(2, 1);
    assert_eq!(ids(&first), vec![BodyId::new(2), island]);
    // The smaller block moved 4 mm on: 2 mm into the larger.
    part.add(FeatureKind::MoveBody(MoveBodyFeature { bodies: vec![island], pivot: DVec3::ZERO, translate: DVec3::new(-4.0, 0.0, 0.0), rotate: DVec3::ZERO, copy: false, exprs: Exprs::new() }));
    let moved = build(&part);
    ok(&moved);
    assert_eq!(volumes(&moved), vec![2000.0, 1000.0]);
    for (body, tool) in [(island, BodyId::new(2)), (BodyId::new(2), island)] {
        let mut p = part.clone();
        combine(&mut p, BooleanOp::Join, body, tool, false);
        let r = build(&p);
        ok(&r);
        assert_eq!(ids(&r), vec![body]);
        assert!((volumes(&r)[0] - (2000.0 + 1000.0 - 200.0)).abs() < 1e-3, "{:?}", volumes(&r));
    }
}

#[test]
fn keeps_the_tool_as_a_body_of_its_own_when_asked_to() {
    let mut base = Part::default();
    let a = rects(&mut base, &[[0.0, 0.0, 20.0, 10.0]]);
    extrude(&mut base, &a, 0.0, 10.0, BodyOp::New);
    let b2 = rects(&mut base, &[[15.0, 0.0, 30.0, 10.0]]);
    extrude(&mut base, &b2, 0.0, 10.0, BodyOp::New);
    let run = |op, keep| {
        let mut p = base.clone();
        combine(&mut p, op, BodyId::new(2), BodyId::new(4), keep);
        build(&p)
    };
    let joined = run(BooleanOp::Join, true);
    ok(&joined);
    assert_eq!(ids(&joined), vec![BodyId::new(2), BodyId::new(4)]);
    assert_eq!(volumes(&joined), vec![3000.0, 1500.0]);
    assert_eq!(volumes(&run(BooleanOp::Cut, true)), vec![1500.0, 1500.0]);
    // Not kept, the tool goes into the result as before.
    assert_eq!(ids(&run(BooleanOp::Cut, false)), vec![BodyId::new(2)]);
}

#[test]
fn a_combine_whose_tool_stands_clear_of_the_body_fails_the_body_as_it_was() {
    let mut part = Part::default();
    let a = rects(&mut part, &[[0.0, 0.0, 10.0, 10.0]]);
    extrude(&mut part, &a, 0.0, 10.0, BodyOp::Join);
    let away = rects(&mut part, &[[50.0, 50.0, 60.0, 60.0]]);
    extrude(&mut part, &away, 0.0, 10.0, BodyOp::New);
    let c = combine(&mut part, BooleanOp::Cut, BodyId::new(2), BodyId::new(4), false);
    let r = build(&part);
    assert!(r.errors.get(&c).is_some_and(|e| e.contains("Extrude 2 meets nothing of Extrude 1")), "{:?}", r.errors);
    assert_eq!(volumes(&r), vec![1000.0, 1000.0]);
    // One that touches the body's face takes nothing out either.
    let mut touch = Part::default();
    let a = rects(&mut touch, &[[0.0, 0.0, 10.0, 10.0]]);
    extrude(&mut touch, &a, 0.0, 10.0, BodyOp::Join);
    let t = rects(&mut touch, &[[10.0, 0.0, 20.0, 10.0]]);
    extrude(&mut touch, &t, 0.0, 10.0, BodyOp::New);
    let c = combine(&mut touch, BooleanOp::Cut, BodyId::new(2), BodyId::new(4), false);
    assert!(build(&touch).errors.get(&c).is_some_and(|e| e.contains("meets nothing")));
    // A body combined with itself, and with a body that went into it already.
    let mut same = touch.clone();
    if let Some(FeatureKind::Boolean(x)) = same.feature_mut(c).map(|f| &mut f.kind) {
        x.tool = BodyId::new(2);
    }
    assert_eq!(build(&same).errors.get(&c).map(String::as_str), Some("The body this one is combined with has gone into it already."));
    let mut gone = Part::default();
    let a = rects(&mut gone, &[[0.0, 0.0, 10.0, 10.0]]);
    extrude(&mut gone, &a, 0.0, 10.0, BodyOp::New);
    let c = combine(&mut gone, BooleanOp::Join, BodyId::new(2), BodyId::new(9), false);
    assert_eq!(build(&gone).errors.get(&c).map(String::as_str), Some("The body this one is combined with is gone, or failed to build."));
}

#[test]
fn intersects_two_bodies_and_says_when_they_do_not_overlap() {
    let mut part = Part::default();
    let a = rects(&mut part, &[[0.0, 0.0, 20.0, 10.0]]);
    extrude(&mut part, &a, 0.0, 10.0, BodyOp::New);
    let b2 = rects(&mut part, &[[15.0, 0.0, 30.0, 10.0]]);
    extrude(&mut part, &b2, 0.0, 10.0, BodyOp::New);
    let mut both = part.clone();
    combine(&mut both, BooleanOp::Intersect, BodyId::new(2), BodyId::new(4), false);
    let r = build(&both);
    ok(&r);
    assert_eq!(volumes(&r), vec![500.0]);
    let mut apart = Part::default();
    let a = rects(&mut apart, &[[0.0, 0.0, 10.0, 10.0]]);
    extrude(&mut apart, &a, 0.0, 10.0, BodyOp::New);
    let b2 = rects(&mut apart, &[[50.0, 0.0, 60.0, 10.0]]);
    extrude(&mut apart, &b2, 0.0, 10.0, BodyOp::New);
    let c = combine(&mut apart, BooleanOp::Intersect, BodyId::new(2), BodyId::new(4), false);
    assert_eq!(build(&apart).errors.get(&c).map(String::as_str), Some("The bodies do not overlap."));
}

// ---- a sheet as the body (reSheetTrim.test.ts) ----

/// A sheet standing in the builder: a wall across the XZ plane at y = 5, from x −10 to 30 and z
/// −10 to 30 — what a Sheet feature makes.
fn with_sheet(b: &mut Builder, id: BodyId, y: f64) {
    // A line on the XY plane at y, swept up from z −10 to 30: the wall.
    let xy = solidcraft_geom::Plane::new(SVec3::new(0.0, 0.0, 0.0), SVec3::new(1.0, 0.0, 0.0), SVec3::new(0.0, 1.0, 0.0)).unwrap_or_else(|| panic!("a plane"));
    let line = vec![solidcraft_geom::Seg2::Line { a: SVec2::new(-10.0, y), b: SVec2::new(30.0, y) }];
    let sheet = kernel::extrude_sheet(&xy, &[line], -10.0, 30.0).unwrap_or_else(|e| panic!("{e}"));
    let mut state = BodyState::new(sheet.clone(), vec![-1; sheet.face_count()]);
    state.surface = true;
    b.set_body(id, state);
}

/// The block, a sheet put in beside it, and a Combine of the two played on.
fn sheet_combine(op: BooleanOp, keep: bool) -> Builder {
    let mut part = Part::default();
    let a = rects(&mut part, &[[0.0, 0.0, 20.0, 10.0]]);
    extrude(&mut part, &a, 0.0, 10.0, BodyOp::New);
    let mut b = play(&part, part.features.len());
    with_sheet(&mut b, BodyId::new(50), 5.0);
    let c = part.add(FeatureKind::Boolean(BooleanFeature { op, body: BodyId::new(50), tool: BodyId::new(2), keep }));
    let n = part.features.len();
    play_range(&mut b, &part, n - 1, n);
    assert!(b.out.errors.get(&c).is_none(), "{:?}", b.out.errors);
    b
}

fn sheet_area(b: &Builder, id: BodyId) -> f64 {
    let s = b.body(id).unwrap_or_else(|| panic!("no body {id}"));
    assert!(s.surface, "{id} is no sheet");
    kernel::measure(&s.shape).map(|m| m.area).unwrap_or(f64::NAN)
}

#[test]
fn a_sheet_intersected_with_a_solid_keeps_what_lies_inside_it_and_cut_what_lies_outside() {
    let inside = sheet_combine(BooleanOp::Intersect, false);
    // The wall across the block: 20 long, 10 high inside it.
    assert!((sheet_area(&inside, BodyId::new(50)) - 200.0).abs() < 1e-3, "{}", sheet_area(&inside, BodyId::new(50)));
    // The solid tool stays a body, to trim another sheet with.
    assert!(inside.body(BodyId::new(2)).is_some_and(|s| !s.surface));
    let outside = sheet_combine(BooleanOp::Cut, false);
    assert!((sheet_area(&outside, BodyId::new(50)) - (1600.0 - 200.0)).abs() < 1e-3, "{}", sheet_area(&outside, BodyId::new(50)));
}

#[test]
fn a_sheet_joins_sheets_only_and_a_solid_combines_with_solids_only() {
    let mut part = Part::default();
    let a = rects(&mut part, &[[0.0, 0.0, 20.0, 10.0]]);
    extrude(&mut part, &a, 0.0, 10.0, BodyOp::New);
    let mut b = play(&part, part.features.len());
    with_sheet(&mut b, BodyId::new(50), 5.0);
    let join = part.add(FeatureKind::Boolean(BooleanFeature { op: BooleanOp::Join, body: BodyId::new(50), tool: BodyId::new(2), keep: false }));
    let solid = part.add(FeatureKind::Boolean(BooleanFeature { op: BooleanOp::Cut, body: BodyId::new(2), tool: BodyId::new(50), keep: false }));
    let n = part.features.len();
    play_range(&mut b, &part, n - 2, n);
    assert!(b.out.errors.get(&join).is_some_and(|e| e.contains("a sheet joins sheets")), "{:?}", b.out.errors);
    assert!(b.out.errors.get(&solid).is_some_and(|e| e.contains("a solid is combined with solids")), "{:?}", b.out.errors);
}

// ---- mirror and cut by plane ----

#[test]
fn a_mirror_not_joined_is_a_body_of_its_own_and_a_joined_one_keeps_the_makers_of_the_faces_it_mirrors() {
    let mut part = Part::default();
    let a = rects(&mut part, &[[0.0, 0.0, 10.0, 6.0]]);
    extrude(&mut part, &a, 0.0, 4.0, BodyOp::New);
    let mut apart = part.clone();
    let m = apart.add(FeatureKind::Mirror(MirrorFeature { body: BodyId::new(2), plane: MirrorPlane::Plane(PlaneRef::world(WorldAxis::X, -1.0)), join: false, exprs: Exprs::new() }));
    let built = build(&apart);
    ok(&built);
    assert_eq!(ids(&built), vec![BodyId::new(2), BodyId::new(m)]);
    assert!(built.bodies[1].topology.faces.iter().all(|f| f.fingerprint.origin == m as i64));
    let mut joined = part.clone();
    joined.add(FeatureKind::Mirror(MirrorFeature { body: BodyId::new(2), plane: MirrorPlane::Plane(PlaneRef::world(WorldAxis::X, 0.0)), join: true, exprs: Exprs::new() }));
    let built = build(&joined);
    ok(&built);
    assert_eq!(volumes(&built), vec![480.0]);
    // The far ends, the block's and its mirror image's, are the extrude's.
    let ends: Vec<i64> = built.bodies[0].topology.faces.iter().filter(|f| f.fingerprint.dir.x.abs() > 0.9).map(|f| f.fingerprint.origin).collect();
    assert_eq!(ends, vec![2, 2]);
}

#[test]
fn a_cut_that_takes_the_whole_body_says_where_the_body_lies() {
    let mut part = Part::default();
    let a = rects(&mut part, &[[0.0, 0.0, 10.0, 10.0]]);
    extrude(&mut part, &a, 0.0, 5.0, BodyOp::New);
    let k = part.add(FeatureKind::CutPlane(CutPlaneFeature { body: BodyId::new(2), plane: PlaneRef::world(WorldAxis::Z, 9.8), keep: 1, exprs: Exprs::new() }));
    let built = build(&part);
    assert_eq!(
        built.errors.get(&k).map(String::as_str),
        Some("The cut takes all of Extrude 1 (body 2, x 0…10, y 0…10, z 0…5): it lies wholly on the side cut away (z < 9.8) — keep -1 keeps all of it, or move the plane.")
    );
    assert_eq!(volumes(&built), vec![500.0]);
}

#[test]
fn grows_an_l_rounding_its_convex_edges_and_leaving_its_inside_corner_sharp() {
    let mut part = Part::default();
    let s = rects(&mut part, &[[0.0, 0.0, 20.0, 10.0], [0.0, 10.0, 10.0, 20.0]]);
    extrude(&mut part, &s, 0.0, 5.0, BodyOp::Join);
    let o = part.add(FeatureKind::OffsetBody(OffsetBodyFeature { body: BodyId::new(2), distance: 1.0, keep: false, exprs: Exprs::new() }));
    let built = build(&part);
    ok(&built);
    let body = &built.bodies[0].body;
    assert!(body.validity().is_empty(), "{:?}", body.validity());
    // Slice by slice: the L's outline grown by ρ is 300 + 80 ρ + (5 π/4 − 1) ρ² — a quarter
    // disc at each of its five outside corners, and at the inside corner the two strips
    // overlapping in a ρ × ρ square, the faces there meeting again sharp. ρ is 1 through the
    // prism's 5 mm and √(1 − t²) at t past either end.
    let pi = std::f64::consts::PI;
    let grown = |rho: f64, rho2: f64| 300.0 + 80.0 * rho + (5.0 * pi / 4.0 - 1.0) * rho2;
    let want = 5.0 * grown(1.0, 1.0) + 2.0 * grown(pi / 4.0, 2.0 / 3.0);
    let got = body.volume_at(1e-4).unwrap_or(0.0);
    assert!((got - want).abs() < 0.05, "{got} against {want} ({:?})", built.errors.get(&o));
}

#[test]
fn grows_a_box_with_a_rounded_edge_its_round_growing_with_it() {
    let mut part = Part::default();
    let s = rects(&mut part, &[[0.0, 0.0, 10.0, 6.0]]);
    extrude(&mut part, &s, 0.0, 4.0, BodyOp::New);
    let built = build(&part);
    let edge = built.bodies[0].topology.edges.iter().find(|e| e.fingerprint.kind == crate::topo::EdgeKind::Line && (e.fingerprint.mid - DVec3::new(5.0, 6.0, 4.0)).length() < 1e-6).map(|e| e.fingerprint.clone()).unwrap_or_else(|| panic!("no top edge"));
    part.add(FeatureKind::Fillet(FilletFeature { body: BodyId::new(2), edges: vec![edge], edges_where: vec![], radius: 1.0, chain: false, skip_failing: false, variable: vec![], exprs: Exprs::new() }));
    let o = part.add(FeatureKind::OffsetBody(OffsetBodyFeature { body: BodyId::new(2), distance: 0.5, keep: true, exprs: Exprs::new() }));
    let built = build(&part);
    match built.errors.get(&o) {
        // Built: a valid solid, grown.
        None => {
            let b = &built.bodies[1].body;
            assert!(b.validity().is_empty(), "{:?}", b.validity());
            assert!(volume(b).unwrap_or(0.0) > volume(&built.bodies[0].body).unwrap_or(0.0) + 100.0);
        }
        // Refused: by name, the kernel's offset said to have failed or come out broken.
        Some(e) => assert!(e.starts_with("Offsetting: the kernel's offset of the body outward by 0.5 mm"), "{e}"),
    }
}

// ---- probes (review) ----

/// A wall across the XZ plane at y, from x `x0` to `x1` and z −10 to 30, as a sheet.
fn wall(x0: f64, x1: f64, y: f64) -> Body {
    let xy = solidcraft_geom::Plane::new(SVec3::new(0.0, 0.0, 0.0), SVec3::new(1.0, 0.0, 0.0), SVec3::new(0.0, 1.0, 0.0)).unwrap_or_else(|| panic!("a plane"));
    let line = vec![solidcraft_geom::Seg2::Line { a: SVec2::new(x0, y), b: SVec2::new(x1, y) }];
    kernel::extrude_sheet(&xy, &[line], -10.0, 30.0).unwrap_or_else(|e| panic!("{e}"))
}

fn put_sheet(b: &mut Builder, id: BodyId, sheet: Body) {
    let mut state = BodyState::new(sheet.clone(), vec![-1; sheet.face_count()]);
    state.surface = true;
    b.set_body(id, state);
}

#[test]
fn probe_sheet_join_apart_is_refused() {
    let mut part = Part::default();
    let a = rects(&mut part, &[[0.0, 0.0, 20.0, 10.0]]);
    extrude(&mut part, &a, 0.0, 10.0, BodyOp::New);
    let mut b = play(&part, part.features.len());
    put_sheet(&mut b, BodyId::new(50), wall(-10.0, 10.0, 5.0));
    put_sheet(&mut b, BodyId::new(51), wall(-10.0, 10.0, 25.0));
    let c = part.add(FeatureKind::Boolean(BooleanFeature { op: BooleanOp::Join, body: BodyId::new(50), tool: BodyId::new(51), keep: false }));
    let n = part.features.len();
    play_range(&mut b, &part, n - 1, n);
    eprintln!("apart join: {:?} bodies {:?}", b.out.errors.get(&c), b.body_ids());
}

#[test]
fn probe_sheet_cut_in_two_stays_one_body() {
    let mut part = Part::default();
    let a = rects(&mut part, &[[0.0, 0.0, 20.0, 10.0]]);
    extrude(&mut part, &a, -20.0, 40.0, BodyOp::New);
    let mut b = play(&part, part.features.len());
    put_sheet(&mut b, BodyId::new(50), wall(-10.0, 30.0, 5.0));
    let c = part.add(FeatureKind::Boolean(BooleanFeature { op: BooleanOp::Cut, body: BodyId::new(50), tool: BodyId::new(2), keep: false }));
    let n = part.features.len();
    play_range(&mut b, &part, n - 1, n);
    eprintln!("cut: {:?} warnings {:?} bodies {:?} pieces {}", b.out.errors.get(&c), b.warnings, b.body_ids(), b.body(BodyId::new(50)).map_or(0, |s| boolean::sheet_pieces(&s.shape)));
}

#[test]
fn probe_offset_sphere() {
    let s = kernel::sphere(SVec3::new(0.0, 0.0, 0.0), 5.0).unwrap_or_else(|e| panic!("{e}"));
    let r = offset_body::offset_solid(&s, 1.0);
    eprintln!("sphere: {:?}", r.as_ref().map(|b| volume(b)).map_err(|e| e.clone()));
    let c = kernel::cylinder(SVec3::new(0.0, 0.0, 0.0), SVec3::new(1.0, 1.0, 0.0).normalized().unwrap_or(SVec3::new(1.0, 0.0, 0.0)), 5.0, 10.0).unwrap_or_else(|e| panic!("{e}"));
    let r = offset_body::offset_solid(&c, 1.0);
    eprintln!("tilted cylinder: {:?}", r.as_ref().map(|b| volume(b)).map_err(|e| e.clone()));
}

#[test]
fn probe_join_extrude_partly_into_a_body() {
    let mut part = Part::default();
    let a = rects(&mut part, &[[0.0, 0.0, 10.0, 10.0]]);
    extrude(&mut part, &a, 0.0, 10.0, BodyOp::New);
    let t = rects(&mut part, &[[50.0, 0.0, 60.0, 10.0]]);
    extrude(&mut part, &t, 0.0, 10.0, BodyOp::New);
    // One profile into body 2, one standing apart.
    let two = rects(&mut part, &[[5.0, 0.0, 15.0, 10.0], [30.0, 0.0, 40.0, 10.0]]);
    extrude(&mut part, &two, 0.0, 10.0, BodyOp::Join);
    let built = build(&part);
    ok(&built);
    eprintln!("ids {:?} volumes {:?}", ids(&built), volumes(&built));
    let mut p = part.clone();
    let c = combine(&mut p, BooleanOp::Join, BodyId::new(4), BodyId::new(2), false);
    let r = build(&p);
    eprintln!("combine: {:?} ids {:?} volumes {:?}", r.errors.get(&c), ids(&r), volumes(&r));
}
