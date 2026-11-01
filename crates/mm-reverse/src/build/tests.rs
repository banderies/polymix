// SPDX-License-Identifier: AGPL-3.0-only
//! The build's tests: every kind built so far, on the kernel, and the faces and edges read off
//! the bodies with their makers — the cases of ScanRuler's `reTopo.test.ts` and
//! `reFeatures.test.ts`, © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.

use glam::{DVec2, DVec3};

use super::*;
use crate::frame::WorldAxis;
use crate::model::*;
use crate::pick_query::{StoredPick, StoredWhere};
use crate::regions::find_regions;
use crate::sketch::{Shape, Sketch};
use crate::topo::{EdgeKind, FaceKind};

fn near(a: DVec3, b: [f64; 3], tol: f64) -> bool {
    (a - DVec3::from(b)).length() < tol
}

/// A sketch on the XY plane at `z` with a rectangle from the origin, and its regions.
fn rect_sketch(part: &mut Part, x0: f64, y0: f64, w: f64, h: f64, z: f64) -> (FeatureId, Vec<String>) {
    let mut s = Sketch::default();
    s.add_rectangle(DVec2::new(x0, y0), DVec2::new(x0 + w, y0 + h));
    let regions = default_regions(&find_regions(&s));
    (part.add(FeatureKind::Sketch(SketchFeature::on_world(WorldAxis::Z, z, s))), regions)
}

fn circle_sketch(part: &mut Part, c: DVec2, r: f64, z: f64) -> (FeatureId, Vec<String>) {
    let mut s = Sketch::default();
    s.add_shape(Shape::Circle { c, r });
    let regions = default_regions(&find_regions(&s));
    (part.add(FeatureKind::Sketch(SketchFeature::on_world(WorldAxis::Z, z, s))), regions)
}

fn extrude(part: &mut Part, sketch: (FeatureId, Vec<String>), from: f64, to: f64, op: BodyOp) -> FeatureId {
    part.add(FeatureKind::Extrude(ExtrudeFeature::new(sketch.0, sketch.1, ExtrudeExtent::Distance { from, to }, op)))
}

/// A 40 × 20 × 10 block: sketch 1, extrude 2.
fn block() -> Part {
    let mut part = Part::default();
    let s = rect_sketch(&mut part, 0.0, 0.0, 40.0, 20.0, 0.0);
    extrude(&mut part, s, 0.0, 10.0, BodyOp::New);
    part
}

/// The block with a Ø6 hole through it at (10, 10): sketch 3, cut 4.
fn holed() -> Part {
    let mut part = block();
    let s = circle_sketch(&mut part, DVec2::new(10.0, 10.0), 3.0, 0.0);
    extrude(&mut part, s, -5.0, 15.0, BodyOp::Cut);
    part
}

fn vol(built: &Built, i: usize) -> f64 {
    volume(&built.bodies[i].body).unwrap_or(f64::NAN)
}

fn ok(built: &Built) {
    assert!(built.errors.is_empty(), "{:?}", built.errors);
}

#[test]
fn extrudes_a_rectangle_into_a_box_and_cuts_a_hole_through_it() {
    let mut part = Part::default();
    let s = rect_sketch(&mut part, -20.0, -10.0, 40.0, 20.0, 0.0);
    extrude(&mut part, s, 0.0, 10.0, BodyOp::Join);
    let built = build(&part);
    ok(&built);
    assert_eq!(built.bodies.len(), 1);
    assert_eq!(built.bodies[0].id, BodyId::new(2));
    assert_eq!(built.bodies[0].name, "Solid 1");
    assert!((vol(&built, 0) - 8000.0).abs() < 1.0);

    let h = circle_sketch(&mut part, DVec2::ZERO, 3.0, 10.0);
    part.add(FeatureKind::Extrude(ExtrudeFeature::new(h.0, h.1, ExtrudeExtent::ToLast { from: 0.0, reverse: true }, BodyOp::Cut)));
    let built = build(&part);
    ok(&built);
    let v = vol(&built, 0);
    assert!((v - (8000.0 - std::f64::consts::PI * 9.0 * 10.0)).abs() < 2.0, "{v}");
    assert_eq!(built.touched.get(&4), Some(&vec![BodyId::new(2)]));

    let step = export_step(&[("Solid 1", &built.bodies[0].body)]).unwrap_or_default();
    assert!(step.starts_with("ISO-10303-21;"));
    assert!(step.contains("CYLINDRICAL_SURFACE"));
}

#[test]
fn a_join_that_touches_nothing_is_a_body_of_its_own_and_a_cut_that_meets_nothing_fails() {
    let mut part = Part::default();
    let s = rect_sketch(&mut part, -5.0, -5.0, 10.0, 10.0, 0.0);
    extrude(&mut part, s.clone(), 0.0, 5.0, BodyOp::Join);
    extrude(&mut part, s.clone(), 20.0, 25.0, BodyOp::Join);
    let cut = extrude(&mut part, s.clone(), 40.0, 45.0, BodyOp::Cut);
    let built = build(&part);
    assert_eq!(built.bodies.iter().map(|b| b.id).collect::<Vec<_>>(), vec![BodyId::new(2), BodyId::new(3)]);
    assert_eq!(built.errors.get(&cut).map(String::as_str), Some("The cut meets no body — there is nothing for it to take out."));
    // A join bridging the two makes one body of all of them, under the oldest one's id.
    extrude(&mut part, s, 4.0, 21.0, BodyOp::Join);
    let built = build(&part);
    assert_eq!(built.bodies.iter().map(|b| b.id).collect::<Vec<_>>(), vec![BodyId::new(2)]);
    assert!((vol(&built, 0) - 2500.0).abs() < 1.0, "{}", vol(&built, 0));
}

#[test]
fn a_new_body_in_two_pieces_is_two_bodies_the_second_an_island_of_the_step() {
    let mut part = Part::default();
    let mut s = Sketch::default();
    s.add_rectangle(DVec2::new(0.0, 0.0), DVec2::new(10.0, 10.0));
    s.add_rectangle(DVec2::new(30.0, 0.0), DVec2::new(45.0, 10.0));
    let regions = default_regions(&find_regions(&s));
    assert_eq!(regions.len(), 2);
    let sk = part.add(FeatureKind::Sketch(SketchFeature::on_world(WorldAxis::Z, 0.0, s)));
    part.add(FeatureKind::Extrude(ExtrudeFeature::new(sk, regions, ExtrudeExtent::Symmetric { total: 4.0 }, BodyOp::New)));
    let built = build(&part);
    ok(&built);
    let ids: Vec<BodyId> = built.bodies.iter().map(|b| b.id).collect();
    assert_eq!(ids, vec![BodyId::new(2), island_id(2, 1)]);
    let names: Vec<&str> = built.bodies.iter().map(|b| b.name.as_str()).collect();
    assert_eq!(names, ["Solid 1", "Solid 2"]);
    assert!(built.warnings.get(&2).is_some_and(|w| w[0].contains("2 separate solids")));
    // A cut through the first piece's waist leaves it in two: the further one an island of
    // the cut.
    let waist = rect_sketch(&mut part, 4.0, -1.0, 2.0, 12.0, -5.0);
    extrude(&mut part, waist, 0.0, 10.0, BodyOp::Cut);
    let built = build(&part);
    ok(&built);
    let mut ids: Vec<BodyId> = built.bodies.iter().map(|b| b.id).collect();
    ids.sort();
    assert_eq!(ids, vec![BodyId::new(2), island_id(2, 1), BodyId::new(4)]);
}

#[test]
fn names_bodies_by_the_order_of_the_features_that_made_them() {
    // ScanRuler's reStore test: handed over out of order, named by the features' order, the
    // further bodies of a step after its first.
    let mut part = Part::default();
    for _ in 0..3 {
        part.add(FeatureKind::Sketch(SketchFeature::on_world(WorldAxis::Z, 0.0, Sketch::default())));
    }
    let body = |id: BodyId| BuiltBody { id, name: String::new(), body: kernel::box_solid(SVec3::new(0.0, 0.0, 0.0), SVec3::new(1.0, 1.0, 1.0)).unwrap_or_else(|e| panic!("{e}")), surface: false, label: None, topology: Arc::new(Topology::default()) };
    let mut bodies = vec![body(island_id(2, 1)), body(BodyId::new(3)), body(BodyId::new(2)), body(island_id(2, 2))];
    name_bodies(&mut bodies, &part);
    assert_eq!(bodies.iter().map(|b| b.name.as_str()).collect::<Vec<_>>(), ["Solid 2", "Solid 4", "Solid 1", "Solid 3"]);
    // A name given in the body styles is the body's; the count runs on under it.
    part.body_styles.insert(BodyId::new(3), BodyStyle { name: Some("Lid".into()), color: None });
    bodies[0].surface = true;
    bodies[3].label = Some("Header".into());
    name_bodies(&mut bodies, &part);
    assert_eq!(bodies.iter().map(|b| b.name.as_str()).collect::<Vec<_>>(), ["Shell 1", "Lid", "Solid 1", "Header"]);
}

#[test]
fn reads_a_box_as_six_outward_planes_and_twelve_lines_each_between_two_faces() {
    let built = build(&block());
    ok(&built);
    let t = &built.bodies[0].topology;
    assert_eq!(t.faces.len(), 6);
    assert!(t.faces.iter().all(|f| f.fingerprint.kind == FaceKind::Plane && f.full_turn.is_none()));
    let top = t.faces.iter().find(|f| near(f.fingerprint.centroid, [20.0, 10.0, 10.0], 1e-6)).unwrap_or_else(|| panic!("no top"));
    assert!(near(top.fingerprint.dir, [0.0, 0.0, 1.0], 1e-9));
    assert!((top.fingerprint.area - 800.0).abs() < 1e-6);
    let bottom = t.faces.iter().find(|f| near(f.fingerprint.centroid, [20.0, 10.0, 0.0], 1e-6)).unwrap_or_else(|| panic!("no bottom"));
    assert!(near(bottom.fingerprint.dir, [0.0, 0.0, -1.0], 1e-9));
    assert_eq!(t.edges.len(), 12);
    assert!(t.edges.iter().all(|e| e.fingerprint.kind == EdgeKind::Line && e.faces.len() == 2 && e.polyline.len() >= 2));
    let front = t.edges.iter().find(|e| near(e.fingerprint.mid, [20.0, 0.0, 10.0], 1e-6)).unwrap_or_else(|| panic!("no top front edge"));
    assert!(near(front.fingerprint.dir, [1.0, 0.0, 0.0], 1e-9));
    assert!((front.fingerprint.length - 40.0).abs() < 1e-6);
    // Each edge lies on both of its faces: their planes pass through its middle.
    for e in &t.edges {
        for f in &e.faces {
            let face = &t.faces[*f as usize].fingerprint;
            assert!((e.fingerprint.mid - face.centroid).dot(face.dir).abs() < 1e-6, "edge {} off face {}", e.hash, f);
        }
    }
    // Every face was made by the extrude, every edge too.
    assert!(t.faces.iter().all(|f| f.fingerprint.origin == 2));
    assert!(t.edges.iter().all(|e| e.fingerprint.origin == 2));
}

#[test]
fn reads_a_hole_as_a_cylinder_rimmed_by_circles_and_carries_each_faces_maker_through_the_cut() {
    let built = build(&holed());
    ok(&built);
    let t = &built.bodies[0].topology;
    // The kernel keeps a whole bore as faces on one cylinder (two halves), where Open
    // CASCADE has one: six planes and the bore.
    assert_eq!(t.faces.iter().filter(|f| f.fingerprint.kind == FaceKind::Plane).count(), 6);
    let walls: Vec<_> = t.faces.iter().filter(|f| f.fingerprint.kind == FaceKind::Cylinder).collect();
    assert!(!walls.is_empty() && walls.len() + 6 == t.faces.len());
    let area: f64 = walls.iter().map(|w| w.fingerprint.area).sum();
    assert!((area - 2.0 * std::f64::consts::PI * 3.0 * 10.0).abs() < 0.5, "{area}");
    for wall in &walls {
        assert!((wall.fingerprint.radius.unwrap_or(0.0) - 3.0).abs() < 1e-6);
        assert!(near(wall.fingerprint.dir, [0.0, 0.0, 1.0], 1e-9));
        assert!((wall.fingerprint.centroid.z - 5.0).abs() < 1e-2, "{:?}", wall.fingerprint.centroid);
        assert!(near(wall.fingerprint.center.unwrap_or_default(), [10.0, 10.0, 5.0], 1e-2));
        assert_eq!(wall.full_turn, Some(true));
        assert_eq!(wall.fingerprint.origin, 4);
    }
    let top = t.faces.iter().find(|f| f.fingerprint.kind == FaceKind::Plane && f.fingerprint.dir.z > 0.9).unwrap_or_else(|| panic!("no top"));
    assert_eq!(top.fingerprint.origin, 2);
    assert_eq!(t.faces.iter().filter(|f| f.fingerprint.origin == 2).count(), 6);
    let circles: Vec<_> = t.edges.iter().filter(|e| e.fingerprint.kind == EdgeKind::Circle).collect();
    assert!(circles.len() >= 2);
    assert!(circles.iter().all(|c| (c.fingerprint.radius.unwrap_or(0.0) - 3.0).abs() < 1e-6 && near(c.fingerprint.dir, [0.0, 0.0, 1.0], 1e-9)));
    let rims: f64 = circles.iter().map(|c| c.fingerprint.length).sum();
    assert!((rims - 2.0 * 2.0 * std::f64::consts::PI * 3.0).abs() < 1e-3, "{rims}");
    // The rim of the hole belongs to the cut; a box edge to the extrude.
    let rim = circles.iter().find(|e| e.fingerprint.mid.z > 9.0).unwrap_or_else(|| panic!("no rim"));
    assert_eq!(rim.fingerprint.origin, 4);
    let mut kinds: Vec<&str> = rim.fingerprint.faces.iter().map(|f| f.kind.name()).collect();
    kinds.sort();
    assert_eq!(kinds, ["cylinder", "plane"]);
    let line = t.edges.iter().find(|e| e.fingerprint.kind == EdgeKind::Line && e.fingerprint.faces.iter().all(|f| f.kind == FaceKind::Plane)).unwrap_or_else(|| panic!("no line"));
    assert_eq!(line.fingerprint.origin, 2);
}

/// The long top edge at the back of the holed block, as a pick stores it.
fn top_edge(built: &Built) -> EdgeRef {
    built.bodies[0].topology.edges.iter().find(|e| e.fingerprint.kind == EdgeKind::Line && e.fingerprint.length > 30.0 && e.fingerprint.mid.z > 9.0 && e.fingerprint.mid.y > 19.0).map(|e| e.fingerprint.clone()).unwrap_or_else(|| panic!("no top edge"))
}

fn fillet_kind(edges: Vec<EdgeRef>, radius: f64) -> FeatureKind {
    FeatureKind::Fillet(FilletFeature { body: BodyId::new(2), edges, edges_where: vec![], radius, chain: false, skip_failing: false, variable: vec![], exprs: Exprs::new() })
}

#[test]
fn rounds_a_picked_edge_tells_a_full_turn_from_a_fillets_part_of_one_and_names_the_fillet_its_maker() {
    let mut part = holed();
    let edge = top_edge(&build(&part));
    let f = part.add(fillet_kind(vec![edge], 2.0));
    let built = build(&part);
    ok(&built);
    let v = vol(&built, 0);
    let want = 8000.0 - std::f64::consts::PI * 9.0 * 10.0 - (4.0 - std::f64::consts::PI) * 40.0;
    assert!((v - want).abs() < 1.0, "{v} vs {want}");
    let rounds: Vec<_> = built.bodies[0].topology.faces.iter().filter(|x| x.fingerprint.kind == FaceKind::Cylinder).collect();
    let bores: Vec<_> = rounds.iter().filter(|x| (x.fingerprint.radius.unwrap_or(0.0) - 3.0).abs() < 1e-6).collect();
    assert!(!bores.is_empty() && bores.iter().all(|b| b.full_turn == Some(true) && b.fingerprint.origin == 4));
    let fillets: Vec<_> = rounds.iter().filter(|x| (x.fingerprint.radius.unwrap_or(0.0) - 2.0).abs() < 1e-6).collect();
    assert_eq!(fillets.len(), 1);
    let round = fillets[0];
    assert_eq!(round.full_turn, Some(false));
    assert_eq!(round.fingerprint.origin, f as i64);
    assert_eq!(built.touched.get(&f), Some(&vec![BodyId::new(2)]));
}

#[test]
fn finds_a_picked_face_and_its_edges_again_after_an_earlier_dimension_moved_and_names_a_lost_one() {
    let part = block();
    let first = build(&part);
    let t = &first.bodies[0].topology;
    let top = t.faces.iter().find(|f| f.fingerprint.dir.z > 0.9).map(|f| f.fingerprint).unwrap_or_else(|| panic!("no top"));
    let far = t.faces.iter().find(|f| f.fingerprint.dir.x > 0.9).map(|f| f.fingerprint).unwrap_or_else(|| panic!("no far side"));
    let top_edges: Vec<EdgeRef> = t.edges.iter().filter(|e| e.fingerprint.mid.z > 9.9).map(|e| e.fingerprint.clone()).collect();
    assert_eq!(top_edges.len(), 4);
    // The same body two millimetres longer: every pick lands again.
    let mut wider = Part::default();
    let s = rect_sketch(&mut wider, 0.0, 0.0, 42.0, 20.0, 0.0);
    extrude(&mut wider, s, 0.0, 10.0, BodyOp::New);
    let mut b = play(&wider, wider.features.len());
    let topo = b.body(BodyId::new(2)).map(|s| s.topology()).unwrap_or_default();
    let found = b.find_faces(BodyId::new(2), &[top], &[]).unwrap_or_default();
    assert!(near(topo.faces[found[0]].fingerprint.centroid, [21.0, 10.0, 10.0], 1e-6));
    let found = b.find_faces(BodyId::new(2), &[far], &[]).unwrap_or_default();
    assert!(near(topo.faces[found[0]].fingerprint.centroid, [42.0, 10.0, 5.0], 1e-6));
    let found = b.find_edges(BodyId::new(2), &top_edges, &[]).unwrap_or_default();
    assert_eq!(found.iter().collect::<std::collections::HashSet<_>>().len(), 4);
    assert!(found.iter().all(|i| (topo.edges[*i].fingerprint.mid.z - 10.0).abs() < 1e-6));
    // A face that is not there any more is reported, by what it was.
    let lost = FaceFingerprint { kind: FaceKind::Cylinder, centroid: DVec3::new(10.0, 10.0, 5.0), dir: DVec3::Z, area: 94.2, radius: Some(3.0), center: None, origin: 4 };
    assert_eq!(b.find_faces(BodyId::new(2), &[lost], &[]), Err("cylinder Ø6.0 at (10.0, 10.0, 5.0)".to_string()));
    let lost_edge = EdgeFingerprint { kind: EdgeKind::Circle, mid: DVec3::new(7.0, 10.0, 10.0), dir: DVec3::Z, length: 18.85, radius: Some(3.0), faces: vec![], origin: 4 };
    assert_eq!(b.find_edges(BodyId::new(2), &[lost_edge], &[]), Err("circle Ø6.0 at (7.0, 10.0, 10.0)".to_string()));
    assert_eq!(b.find_faces(BodyId::new(99), &[top], &[]), Err("the body is gone".to_string()));
}

#[test]
fn a_round_named_by_a_query_rounds_what_the_query_names_on_the_part_as_it_stands() {
    // Every top edge of the block, by a query: madeBy the extrude, along the top.
    let mut part = block();
    let first = build(&part);
    let t = &first.bodies[0].topology;
    let found: Vec<EdgeRef> = t.edges.iter().filter(|e| e.fingerprint.mid.z > 9.9).map(|e| e.fingerprint.clone()).collect();
    let query = StoredPick { query: StoredWhere { made_by: Some(2), bounds: Some(crate::pick_query::Bounds { min: DVec3::new(-100.0, -100.0, 9.9), max: DVec3::new(100.0, 100.0, 10.1) }), ..Default::default() }, found: found.clone() };
    let f = part.add(FeatureKind::Fillet(FilletFeature { body: BodyId::new(2), edges: found, edges_where: vec![query], radius: 1.0, chain: false, skip_failing: false, variable: vec![], exprs: Exprs::new() }));
    let built = build(&part);
    ok(&built);
    assert!(!built.warnings.contains_key(&f), "{:?}", built.warnings);
    let rounds = built.bodies[0].topology.faces.iter().filter(|x| x.fingerprint.kind == FaceKind::Cylinder).count();
    assert_eq!(rounds, 4);
    // A quarter round of radius 1 takes 1 − π/4 mm² off the section, along 120 mm of edge.
    let v = vol(&built, 0);
    assert!(v < 8000.0 - (1.0 - std::f64::consts::FRAC_PI_4) * 100.0, "{v}");
}

#[test]
fn chains_a_round_along_tangent_edges() {
    // A slot: two lines and two half circles. One straight edge picked, the chain takes the
    // rounds and the other line too.
    let mut part = Part::default();
    let mut s = Sketch::default();
    s.add_shape(Shape::Line { a: DVec2::new(0.0, 0.0), b: DVec2::new(20.0, 0.0) });
    s.add_shape(Shape::Arc { c: DVec2::new(20.0, 5.0), r: 5.0, start: -std::f64::consts::FRAC_PI_2, sweep: std::f64::consts::PI });
    s.add_shape(Shape::Line { a: DVec2::new(20.0, 10.0), b: DVec2::new(0.0, 10.0) });
    s.add_shape(Shape::Arc { c: DVec2::new(0.0, 5.0), r: 5.0, start: std::f64::consts::FRAC_PI_2, sweep: std::f64::consts::PI });
    let regions = default_regions(&find_regions(&s));
    let sk = part.add(FeatureKind::Sketch(SketchFeature::on_world(WorldAxis::Z, 0.0, s)));
    part.add(FeatureKind::Extrude(ExtrudeFeature::new(sk, regions, ExtrudeExtent::Distance { from: 0.0, to: 5.0 }, BodyOp::New)));
    let first = build(&part);
    ok(&first);
    let edge = first.bodies[0].topology.edges.iter().find(|e| e.fingerprint.kind == EdgeKind::Line && near(e.fingerprint.mid, [10.0, 0.0, 5.0], 1e-6)).map(|e| e.fingerprint.clone()).unwrap_or_else(|| panic!("no straight top edge"));
    let b = play(&part, part.features.len());
    let t = b.body(BodyId::new(2)).map(|s| s.topology()).unwrap_or_default();
    let seed = t.edges.iter().position(|e| near(e.fingerprint.mid, [10.0, 0.0, 5.0], 1e-6)).unwrap_or(0);
    let chain = super::fillet::chained(&t, &[seed], true);
    assert_eq!(chain.len(), 4, "the chain round the slot's top: {chain:?}");
    part.add(FeatureKind::Fillet(FilletFeature { body: BodyId::new(2), edges: vec![edge], edges_where: vec![], radius: 1.0, chain: true, skip_failing: false, variable: vec![], exprs: Exprs::new() }));
    let built = build(&part);
    ok(&built);
}

#[test]
fn bevels_an_edge_by_one_distance_and_by_two() {
    let mut part = block();
    let edge = top_edge(&build(&holed()));
    part.add(FeatureKind::Chamfer(ChamferFeature { body: BodyId::new(2), edges: vec![edge.clone()], edges_where: vec![], distance: 2.0, chain: false, skip_failing: false, mode: None, distance2: None, angle_deg: None, flip: false, exprs: Exprs::new() }));
    let built = build(&part);
    ok(&built);
    assert!((vol(&built, 0) - (8000.0 - 2.0 * 40.0)).abs() < 1e-3, "{}", vol(&built, 0));
    let mut two = block();
    two.add(FeatureKind::Chamfer(ChamferFeature { body: BodyId::new(2), edges: vec![edge], edges_where: vec![], distance: 2.0, chain: false, skip_failing: false, mode: Some(ChamferMode::TwoDistances), distance2: Some(4.0), angle_deg: None, flip: false, exprs: Exprs::new() }));
    let built = build(&two);
    ok(&built);
    assert!((vol(&built, 0) - (8000.0 - 4.0 * 40.0)).abs() < 1e-3, "{}", vol(&built, 0));
}

#[test]
fn hollows_a_block_open_at_its_top() {
    let mut part = block();
    let top = build(&part).bodies[0].topology.faces.iter().find(|f| f.fingerprint.dir.z > 0.9).map(|f| f.fingerprint).unwrap_or_else(|| panic!("no top"));
    part.add(FeatureKind::Shell(ShellFeature { body: BodyId::new(2), faces: vec![top], faces_where: vec![], thickness: 2.0, exprs: Exprs::new() }));
    let built = build(&part);
    ok(&built);
    let want = 8000.0 - 36.0 * 16.0 * 8.0;
    assert!((vol(&built, 0) - want).abs() < 1e-3, "{} vs {want}", vol(&built, 0));
}

#[test]
fn revolves_a_profile_about_a_coordinate_axis() {
    let mut part = Part::default();
    let mut s = Sketch::default();
    s.add_rectangle(DVec2::new(5.0, 0.0), DVec2::new(10.0, 20.0));
    let regions = default_regions(&find_regions(&s));
    let sk = part.add(FeatureKind::Sketch(SketchFeature::on_world(WorldAxis::Y, 0.0, s)));
    // The XZ plane's sheet runs along Z (u) and X (v): the Z axis lies in it.
    part.add(FeatureKind::Revolve(RevolveFeature::new(sk, regions.clone(), AxisRef::World { axis: WorldAxis::Z }, 360.0, BodyOp::Join)));
    let built = build(&part);
    ok(&built);
    assert_eq!(built.bodies.len(), 1);
    let v = vol(&built, 0);
    assert!(v > 100.0, "{v}");
    // A coil of a profile on its axis is refused, saying why (EXT: coils are built now).
    part.add(FeatureKind::Revolve(RevolveFeature { pitch: Some(2.0), ..RevolveFeature::new(sk, regions, AxisRef::World { axis: WorldAxis::Z }, 720.0, BodyOp::New) }));
    let built = build(&part);
    assert!(built.errors.get(&3).is_some_and(|e| e.contains("touches or crosses the axis")), "{:?}", built.errors);
}

#[test]
fn mirrors_a_body_about_a_coordinate_plane_and_about_its_own_face() {
    let mut part = block();
    part.add(FeatureKind::Mirror(MirrorFeature { body: BodyId::new(2), plane: MirrorPlane::Plane(PlaneRef::world(WorldAxis::X, 0.0)), join: false, exprs: Exprs::new() }));
    let built = build(&part);
    ok(&built);
    assert_eq!(built.bodies.iter().map(|b| b.id).collect::<Vec<_>>(), vec![BodyId::new(2), BodyId::new(3)]);
    let bb = built.bodies[1].body.outline_bounds();
    assert!((bb.min.x + 40.0).abs() < 1e-6 && bb.max.x.abs() < 1e-6);
    let mut joined = block();
    let far = build(&joined).bodies[0].topology.faces.iter().find(|f| f.fingerprint.dir.x > 0.9).map(|f| f.fingerprint).unwrap_or_else(|| panic!("no far side"));
    joined.add(FeatureKind::Mirror(MirrorFeature { body: BodyId::new(2), plane: MirrorPlane::Face { face: far }, join: true, exprs: Exprs::new() }));
    let built = build(&joined);
    ok(&built);
    assert_eq!(built.bodies.len(), 1);
    assert!((vol(&built, 0) - 16000.0).abs() < 1e-3);
}

#[test]
fn combines_two_bodies_cutting_the_tool_out_and_keeping_it_when_asked() {
    let mut part = block();
    let s = rect_sketch(&mut part, 30.0, 5.0, 20.0, 10.0, 0.0);
    extrude(&mut part, s, 0.0, 10.0, BodyOp::New);
    let c = part.add(FeatureKind::Boolean(BooleanFeature { op: BooleanOp::Cut, body: BodyId::new(2), tool: BodyId::new(4), keep: false }));
    let built = build(&part);
    ok(&built);
    assert_eq!(part.feature(c).map(|f| f.name.as_str()), Some("Combine 1"));
    assert_eq!(built.bodies.len(), 1);
    assert!((vol(&built, 0) - (8000.0 - 1000.0)).abs() < 1e-3);
    if let Some(FeatureKind::Boolean(x)) = part.feature_mut(c).map(|f| &mut f.kind) {
        x.keep = true;
        x.op = BooleanOp::Intersect;
    }
    let built = build(&part);
    ok(&built);
    assert_eq!(built.bodies.len(), 2);
    assert!((vol(&built, 0) - 1000.0).abs() < 1e-3);
}

#[test]
fn cuts_a_body_at_a_plane_keeping_either_side() {
    let mut part = block();
    let k = part.add(FeatureKind::CutPlane(CutPlaneFeature { body: BodyId::new(2), plane: PlaneRef::world(WorldAxis::Z, 4.0), keep: 1, exprs: Exprs::new() }));
    let built = build(&part);
    ok(&built);
    assert!((vol(&built, 0) - 4800.0).abs() < 1e-3, "{}", vol(&built, 0));
    if let Some(FeatureKind::CutPlane(x)) = part.feature_mut(k).map(|f| &mut f.kind) {
        x.keep = -1;
    }
    let built = build(&part);
    ok(&built);
    assert!((vol(&built, 0) - 3200.0).abs() < 1e-3);
    // The cut face is the cut's.
    let cap = built.bodies[0].topology.faces.iter().find(|f| f.fingerprint.dir.z > 0.9).unwrap_or_else(|| panic!("no cap"));
    assert_eq!(cap.fingerprint.origin, k as i64);
}

#[test]
fn moves_a_body_and_a_copy_of_it() {
    let mut part = block();
    let m = part.add(FeatureKind::MoveBody(MoveBodyFeature { bodies: vec![BodyId::new(2)], pivot: DVec3::ZERO, translate: DVec3::new(0.0, 0.0, 5.0), rotate: DVec3::new(0.0, 0.0, 90.0), copy: false, exprs: Exprs::new() }));
    let built = build(&part);
    ok(&built);
    let bb = built.bodies[0].body.outline_bounds();
    assert!((bb.min.x + 20.0).abs() < 1e-6 && (bb.max.y - 40.0).abs() < 1e-6 && (bb.min.z - 5.0).abs() < 1e-6, "{bb:?}");
    if let Some(FeatureKind::MoveBody(x)) = part.feature_mut(m).map(|f| &mut f.kind) {
        x.copy = true;
    }
    let built = build(&part);
    ok(&built);
    assert_eq!(built.bodies.iter().map(|b| b.id).collect::<Vec<_>>(), vec![BodyId::new(2), BodyId::new(m)]);
    // The matrix is ScanRuler's: turned about the pivot, then carried.
    let mm = move_matrix(DVec3::new(1.0, 0.0, 0.0), DVec3::new(0.0, 0.0, 2.0), DVec3::new(0.0, 0.0, 90.0));
    assert!((apply_matrix(&mm, DVec3::new(2.0, 0.0, 0.0)) - DVec3::new(1.0, 1.0, 2.0)).length() < 1e-12);
}

#[test]
fn patterns_a_body_in_a_row_and_round_an_axis() {
    let mut part = Part::default();
    let s = rect_sketch(&mut part, 0.0, 0.0, 5.0, 5.0, 0.0);
    extrude(&mut part, s, 0.0, 5.0, BodyOp::New);
    part.add(FeatureKind::Pattern(PatternFeature { body: BodyId::new(2), mode: PatternMode::Linear, axis: AxisRef::World { axis: WorldAxis::X }, count: 3, step: 10.0, axis2: Some(AxisRef::World { axis: WorldAxis::Y }), count2: Some(2), step2: Some(10.0), exprs: Exprs::new() }));
    let built = build(&part);
    ok(&built);
    assert_eq!(built.bodies.len(), 6);
    assert_eq!(built.bodies[0].id, BodyId::new(2));
    assert!(built.bodies.iter().skip(1).all(|b| b.id.feature == 3 || b.id.feature == 2));
    let total: f64 = built.bodies.iter().map(|b| volume(&b.body).unwrap_or(0.0)).sum();
    assert!((total - 6.0 * 125.0).abs() < 1e-3, "{total}");
    let mut round = Part::default();
    let s = rect_sketch(&mut round, 10.0, -2.0, 5.0, 4.0, 0.0);
    extrude(&mut round, s, 0.0, 3.0, BodyOp::New);
    round.add(FeatureKind::Pattern(PatternFeature { body: BodyId::new(2), mode: PatternMode::Circular, axis: AxisRef::World { axis: WorldAxis::Z }, count: 4, step: 360.0, axis2: None, count2: None, step2: None, exprs: Exprs::new() }));
    let built = build(&round);
    ok(&built);
    assert_eq!(built.bodies.len(), 4);
}

#[test]
fn extracts_elements_as_solids() {
    let mut part = Part::default();
    part.elements = vec![
        Element { id: 1, name: "Cylinder 1".into(), geometry: ElementGeometry::Cylinder { center: DVec3::new(0.0, 0.0, 5.0), axis: DVec3::Z, radius: 2.0, length: 10.0, coverage: 360.0 }, stats: None },
        Element { id: 2, name: "Plane 1".into(), geometry: ElementGeometry::Plane { center: DVec3::new(50.0, 0.0, 0.0), normal: DVec3::Z, basis_u: DVec3::X, basis_v: DVec3::Y, extent_u: 5.0, extent_v: 3.0 }, stats: None },
        Element { id: 3, name: "Sphere 1".into(), geometry: ElementGeometry::Sphere { center: DVec3::new(-50.0, 0.0, 0.0), radius: 4.0 }, stats: None },
        Element { id: 4, name: "Cone 1".into(), geometry: ElementGeometry::Cone { center: DVec3::new(0.0, 50.0, 0.0), axis: DVec3::Z, half_angle: 10.0, radius: 3.0, radius1: 2.0, radius2: 4.0, length: 6.0, coverage: 360.0 }, stats: None },
        Element { id: 5, name: "Point 1".into(), geometry: ElementGeometry::Point { center: DVec3::ZERO }, stats: None },
    ];
    for (element, thickness) in [(1, 0.0), (2, 2.0), (3, 0.0), (4, 0.0), (5, 0.0)] {
        part.add(FeatureKind::Extract(ExtractFeature { element: Some(element), section: None, thickness, op: BodyOp::New, target: None, exprs: Exprs::new() }));
    }
    let built = build(&part);
    let pi = std::f64::consts::PI;
    assert_eq!(built.errors.get(&5).map(String::as_str), Some("A point has no volume to make a body of."));
    assert_eq!(built.errors.len(), 1, "{:?}", built.errors);
    let v = |id: u32| built.body(BodyId::new(id)).and_then(|b| volume(&b.body)).unwrap_or(f64::NAN);
    assert!((v(1) - pi * 4.0 * 10.0).abs() < 0.05, "{}", v(1));
    assert!((v(2) - 10.0 * 6.0 * 2.0).abs() < 1e-3, "{}", v(2));
    assert!((v(3) - 4.0 / 3.0 * pi * 64.0).abs() < 0.5, "{}", v(3));
    assert!((v(4) - pi * 6.0 / 3.0 * (4.0 + 8.0 + 16.0)).abs() < 0.5, "{}", v(4));
}

#[test]
fn imports_a_step_file_the_part_holds_as_bodies_of_its_own() {
    let built = build(&block());
    let text = export_step(&[("Bracket", &built.bodies[0].body)]).unwrap_or_default();
    let mut part = Part::default();
    let key = step_file_key(&text);
    let file = StepFileRef { key: key.clone(), name: "bracket.step".into(), size: text.len() as u64 };
    part.add(FeatureKind::Import(ImportFeature { file: file.clone() }));
    let missing = build(&part);
    assert_eq!(missing.errors.get(&1).map(String::as_str), Some("The STEP file bracket.step is not in the part — import it again."));
    part.step_files.insert(key, StepFile { name: file.name, size: file.size, text });
    let built = build(&part);
    ok(&built);
    assert_eq!(built.bodies.len(), 1);
    assert_eq!(built.bodies[0].id, BodyId::new(1));
    assert_eq!(built.bodies[0].name, "Bracket");
    assert!((vol(&built, 0) - 8000.0).abs() < 1e-3);
}

#[test]
fn deletes_a_bevel_and_heals_the_block() {
    let mut part = block();
    let edge = top_edge(&build(&holed()));
    part.add(FeatureKind::Chamfer(ChamferFeature { body: BodyId::new(2), edges: vec![edge], edges_where: vec![], distance: 2.0, chain: false, skip_failing: false, mode: None, distance2: None, angle_deg: None, flip: false, exprs: Exprs::new() }));
    let bevelled = build(&part);
    ok(&bevelled);
    let bevel = bevelled.bodies[0].topology.faces.iter().find(|f| f.fingerprint.kind == FaceKind::Plane && f.fingerprint.dir.y > 0.5 && f.fingerprint.dir.z > 0.5).map(|f| f.fingerprint).unwrap_or_else(|| panic!("no bevel"));
    assert_eq!(bevel.origin, 3);
    part.add(FeatureKind::DeleteFace(DeleteFaceFeature { body: BodyId::new(2), faces: vec![bevel], faces_where: vec![] }));
    let built = build(&part);
    ok(&built);
    assert!((vol(&built, 0) - 8000.0).abs() < 1e-3, "{}", vol(&built, 0));
}

#[test]
fn a_kind_not_built_yet_fails_by_name_and_the_rest_build() {
    let mut part = block();
    let h = part.add(FeatureKind::Hole(HoleFeature { body: BodyId::new(2), axis: HoleAxis::Element { id: 1 }, radius: 1.0, depth: None, insert: None, exprs: Exprs::new() }));
    let built = build(&part);
    // ---- HOLE ---- The Hole is built now: the element it names is not in the part.
    assert_eq!(built.errors.get(&h).map(String::as_str), Some("The element the hole follows is gone."));
    assert_eq!(built.bodies.len(), 1);
    // A feature on a body that is gone says so.
    let mut gone = block();
    gone.add(FeatureKind::Shell(ShellFeature { body: BodyId::new(9), faces: vec![], faces_where: vec![], thickness: 1.0, exprs: Exprs::new() }));
    assert_eq!(build(&gone).errors.get(&3).map(String::as_str), Some("The body this feature works on is gone, or failed to build."));
}

#[test]
fn an_extrude_with_nothing_picked_or_its_sketch_below_it_says_why() {
    let mut part = block();
    if let Some(FeatureKind::Extrude(e)) = part.feature_mut(2).map(|f| &mut f.kind) {
        e.regions.clear();
    }
    assert_eq!(build(&part).errors.get(&2).map(String::as_str), Some("No profile picked — click the sketch's regions in the viewport."));
    let mut below = block();
    below.features.swap(0, 1);
    assert!(build(&below).errors.get(&2).is_some_and(|e| e.contains("stands after this extrude")));
    let mut suppressed = block();
    suppressed.features[0].flags.suppressed = true;
    assert!(build(&suppressed).errors.get(&2).is_some_and(|e| e.contains("is suppressed")));
    // A pick kept as given — every region — follows the sketch.
    let mut all = block();
    if let Some(FeatureKind::Extrude(e)) = all.feature_mut(2).map(|f| &mut f.kind) {
        e.region_pick = Some(RegionPick { given: RegionGiven::Word(RegionWord::All), keys: e.regions.clone() });
    }
    ok(&build(&all));
}

#[test]
fn a_sketch_on_a_bodys_face_follows_the_face() {
    let mut part = block();
    let top = build(&part).bodies[0].topology.faces.iter().find(|f| f.fingerprint.dir.z > 0.9).map(|f| f.fingerprint).unwrap_or_else(|| panic!("no top"));
    let plane = body_face_plane(BodyId::new(2), &top).unwrap_or_else(|| panic!("not flat"));
    let frame = plane.own_frame().unwrap_or_else(|| panic!("no frame"));
    let mut s = Sketch::default();
    s.add_shape(Shape::Circle { c: DVec2::new(10.0, 10.0), r: 2.0 });
    let regions = default_regions(&find_regions(&s));
    let sk = part.add(FeatureKind::Sketch(SketchFeature { plane, frame, sketch: s, exprs: Exprs::new(), image: None, inset: None }));
    part.add(FeatureKind::Extrude(ExtrudeFeature::new(sk, regions, ExtrudeExtent::Distance { from: 0.0, to: 5.0 }, BodyOp::Join)));
    // The block grown to 15 high: the boss stands on its new top.
    if let Some(FeatureKind::Extrude(e)) = part.feature_mut(2).map(|f| &mut f.kind) {
        e.extent = ExtrudeExtent::Distance { from: 0.0, to: 15.0 };
    }
    let built = build(&part);
    ok(&built);
    let bb = built.bodies[0].body.outline_bounds();
    assert!((bb.max.z - 20.0).abs() < 1e-6, "{bb:?}");
}

#[test]
fn mirror_matrix_reflects_across_the_plane() {
    let m = mirror_matrix(DVec3::new(0.0, 0.0, 5.0), DVec3::Z);
    assert!((apply_matrix(&m, DVec3::new(1.0, 2.0, 3.0)) - DVec3::new(1.0, 2.0, 7.0)).length() < 1e-12);
    let r = rotation_matrix(DVec3::new(1.0, 0.0, 0.0), DVec3::Z, std::f64::consts::FRAC_PI_2);
    assert!((apply_matrix(&r, DVec3::new(2.0, 0.0, 0.0)) - DVec3::new(1.0, 1.0, 0.0)).length() < 1e-12);
}

#[test]
fn a_body_mesh_names_its_drawn_edges_by_the_topologys_hashes() {
    let built = build(&holed());
    let mesh = body_mesh(&built.bodies[0].body).unwrap_or_default();
    assert!(!mesh.triangles.is_empty() && !mesh.edges.is_empty());
    assert_eq!(mesh.edges.len(), mesh.edge_ids.len());
    let t = &built.bodies[0].topology;
    for (pts, id) in mesh.edges.iter().zip(&mesh.edge_ids) {
        let e = &t.edges[*id as usize];
        let mid = pts[pts.len() / 2];
        assert!(e.polyline.windows(2).any(|w| dist_to_segment(mid, w[0], w[1]) < 0.1), "edge {id}");
    }
    assert_eq!(*mesh.tri_face.iter().max().unwrap_or(&0) as usize + 1, t.faces.len());
}

// ---- picks found again after an earlier feature changed ----

/// The block's extrude pulled to another height.
fn set_height(part: &mut Part, to: f64) {
    if let Some(FeatureKind::Extrude(e)) = part.feature_mut(2).map(|f| &mut f.kind) {
        e.extent = ExtrudeExtent::Distance { from: 0.0, to };
    }
}

/// The block's sketch drawn again as a rectangle of another size (the same pieces, so the same
/// region).
fn set_rect(part: &mut Part, w: f64, h: f64) {
    if let Some(s) = part.sketch_mut(1) {
        let mut sk = Sketch::default();
        sk.add_rectangle(DVec2::ZERO, DVec2::new(w, h));
        s.sketch = sk;
    }
}

/// An edge of a built body by where its middle lies.
fn edge_at(built: &Built, mid: [f64; 3]) -> EdgeRef {
    built.bodies[0].topology.edges.iter().find(|e| near(e.fingerprint.mid, mid, 1e-6)).map(|e| e.fingerprint.clone()).unwrap_or_else(|| panic!("no edge at {mid:?}"))
}

/// A face of a built body by which way it faces.
fn face_facing(built: &Built, dir: [f64; 3]) -> FaceRef {
    built.bodies[0].topology.faces.iter().find(|f| f.fingerprint.kind == FaceKind::Plane && near(f.fingerprint.dir, dir, 1e-9)).map(|f| f.fingerprint).unwrap_or_else(|| panic!("no face facing {dir:?}"))
}

/// The axes of the rounds of a built body of the given radius: (a point on the axis, its
/// direction).
fn rounds_of(built: &Built, radius: f64) -> Vec<(DVec3, DVec3)> {
    built.bodies[0].topology.faces.iter().filter(|f| f.fingerprint.kind == FaceKind::Cylinder && f.fingerprint.radius.is_some_and(|r| (r - radius).abs() < 1e-6)).filter_map(|f| f.fingerprint.center.map(|c| (c, f.fingerprint.dir))).collect()
}

#[test]
fn a_fillet_finds_its_edge_again_after_the_block_under_it_is_drawn_wider_and_taller() {
    // The block, its top front edge fingerprinted, a 2 mm round on it.
    let mut part = block();
    let edge = edge_at(&build(&part), [20.0, 0.0, 10.0]);
    assert_eq!(edge.origin, 2);
    let f = part.add(fillet_kind(vec![edge], 2.0));
    ok(&build(&part));
    // The sketch drawn 44 long and the extrude pulled to 12: the edge moved and grew, the
    // kernel's lists are made again — the round is still on the top front edge.
    set_rect(&mut part, 44.0, 20.0);
    set_height(&mut part, 12.0);
    let built = build(&part);
    ok(&built);
    assert!(!built.warnings.contains_key(&f), "{:?}", built.warnings);
    let rounds = rounds_of(&built, 2.0);
    assert_eq!(rounds.len(), 1, "{rounds:?}");
    let (on, along) = rounds[0];
    assert!(near(along, [1.0, 0.0, 0.0], 1e-9));
    assert!((on.y - 2.0).abs() < 1e-6 && (on.z - 10.0).abs() < 1e-6, "{on:?}");
    let want = 44.0 * 20.0 * 12.0 - (4.0 - std::f64::consts::PI) * 44.0;
    assert!((vol(&built, 0) - want).abs() < 1.0, "{} vs {want}", vol(&built, 0));
}

#[test]
fn a_fillet_takes_the_second_look_when_a_boss_is_pulled_from_7_to_17() {
    // ScanRuler's reTopo case: the top edge moves 10 mm, past the first look's reach, and is
    // found among the edges the same feature made, facing the same way — not the bottom edge
    // under it, and said on the row.
    let mut part = Part::default();
    let s = rect_sketch(&mut part, 0.0, 0.0, 20.0, 20.0, 0.0);
    extrude(&mut part, s, 0.0, 7.0, BodyOp::New);
    let edge = edge_at(&build(&part), [10.0, 0.0, 7.0]);
    let f = part.add(fillet_kind(vec![edge], 1.0));
    set_height(&mut part, 17.0);
    let built = build(&part);
    ok(&built);
    let rounds = rounds_of(&built, 1.0);
    assert_eq!(rounds.len(), 1);
    assert!((rounds[0].0.z - 16.0).abs() < 1e-6 && (rounds[0].0.y - 1.0).abs() < 1e-6, "{:?}", rounds[0]);
    assert!(built.warnings.get(&f).is_some_and(|w| w.iter().any(|l| l.contains("found again only loosely"))), "{:?}", built.warnings);
}

#[test]
fn a_fillet_whose_edge_an_earlier_change_took_away_fails_by_what_the_edge_was() {
    // The round is on the top back edge; a cut put in before it takes the back off the block,
    // and the edge its cut face leaves is the cut's, not the extrude's: the pick is gone, said
    // by what it was, and the rest of the part builds.
    let mut part = holed();
    let edge = top_edge(&build(&part));
    let f = part.add(fillet_kind(vec![edge], 2.0));
    ok(&build(&part));
    part.add(FeatureKind::CutPlane(CutPlaneFeature { body: BodyId::new(2), plane: PlaneRef::world(WorldAxis::Y, 15.0), keep: -1, exprs: Exprs::new() }));
    let cut = part.features.pop().unwrap_or_else(|| panic!("no cut"));
    let at = part.index_of(f).unwrap_or(0);
    part.features.insert(at, cut);
    let built = build(&part);
    let e = built.errors.get(&f).cloned().unwrap_or_default();
    assert_eq!(e, "The edge this fillet was on is gone (was: line 40.0 mm at (20.0, 20.0, 10.0)).");
    assert_eq!(built.errors.len(), 1, "{:?}", built.errors);
    assert!((vol(&built, 0) - (40.0 * 15.0 * 10.0 - std::f64::consts::PI * 9.0 * 10.0)).abs() < 1.0, "{}", vol(&built, 0));
}

#[test]
fn a_chamfer_a_shell_and_a_mirror_find_their_picks_again_on_a_block_drawn_otherwise() {
    let first = build(&block());
    let back = edge_at(&first, [20.0, 20.0, 10.0]);
    let top = face_facing(&first, [0.0, 0.0, 1.0]);
    let far = face_facing(&first, [1.0, 0.0, 0.0]);
    let redraw = |part: &mut Part| {
        set_rect(part, 44.0, 22.0);
        set_height(part, 11.0);
    };
    // The chamfer on the top back edge.
    let mut part = block();
    let c = part.add(FeatureKind::Chamfer(ChamferFeature { body: BodyId::new(2), edges: vec![back], edges_where: vec![], distance: 2.0, chain: false, skip_failing: false, mode: None, distance2: None, angle_deg: None, flip: false, exprs: Exprs::new() }));
    redraw(&mut part);
    let built = build(&part);
    ok(&built);
    assert!(!built.warnings.contains_key(&c), "{:?}", built.warnings);
    assert!((vol(&built, 0) - (44.0 * 22.0 * 11.0 - 2.0 * 44.0)).abs() < 1e-3, "{}", vol(&built, 0));
    let bevel = built.bodies[0].topology.faces.iter().find(|f| f.fingerprint.kind == FaceKind::Plane && f.fingerprint.dir.y > 0.5 && f.fingerprint.dir.z > 0.5).unwrap_or_else(|| panic!("no bevel"));
    assert!((bevel.fingerprint.centroid.y - 21.0).abs() < 1e-6 && (bevel.fingerprint.centroid.z - 10.0).abs() < 1e-6, "{:?}", bevel.fingerprint.centroid);
    // The shell open at the top.
    let mut part = block();
    part.add(FeatureKind::Shell(ShellFeature { body: BodyId::new(2), faces: vec![top], faces_where: vec![], thickness: 2.0, exprs: Exprs::new() }));
    redraw(&mut part);
    let built = build(&part);
    ok(&built);
    let want = 44.0 * 22.0 * 11.0 - 40.0 * 18.0 * 9.0;
    assert!((vol(&built, 0) - want).abs() < 1e-3, "{} vs {want}", vol(&built, 0));
    // The mirror about the far face, joined.
    let mut part = block();
    part.add(FeatureKind::Mirror(MirrorFeature { body: BodyId::new(2), plane: MirrorPlane::Face { face: far }, join: true, exprs: Exprs::new() }));
    redraw(&mut part);
    let built = build(&part);
    ok(&built);
    assert_eq!(built.bodies.len(), 1);
    let bb = built.bodies[0].body.outline_bounds();
    assert!((bb.max.x - 88.0).abs() < 1e-6 && bb.min.x.abs() < 1e-6, "{bb:?}");
    assert!((vol(&built, 0) - 2.0 * 44.0 * 22.0 * 11.0).abs() < 1e-3);
}

#[test]
fn a_combine_keeps_its_bodies_by_id_when_they_are_drawn_otherwise() {
    let mut part = block();
    let s = rect_sketch(&mut part, 30.0, 5.0, 20.0, 10.0, 0.0);
    extrude(&mut part, s, 0.0, 10.0, BodyOp::New);
    part.add(FeatureKind::Boolean(BooleanFeature { op: BooleanOp::Cut, body: BodyId::new(2), tool: BodyId::new(4), keep: false }));
    set_height(&mut part, 6.0);
    let built = build(&part);
    ok(&built);
    assert_eq!(built.bodies.iter().map(|b| b.id).collect::<Vec<_>>(), vec![BodyId::new(2)]);
    assert!((vol(&built, 0) - (40.0 * 20.0 * 6.0 - 10.0 * 10.0 * 6.0)).abs() < 1e-3, "{}", vol(&built, 0));
}

#[test]
fn a_face_a_join_merges_flush_with_the_body_stays_the_bodys() {
    // ScanRuler's propagate: the top of a block stays the block's when a boss is joined flush
    // with it; the boss's own side faces are the join's.
    let mut part = block();
    let s = rect_sketch(&mut part, 30.0, 5.0, 30.0, 10.0, 0.0);
    let j = extrude(&mut part, s, 0.0, 10.0, BodyOp::Join);
    let built = build(&part);
    ok(&built);
    assert_eq!(built.bodies.len(), 1);
    let t = &built.bodies[0].topology;
    let tops: Vec<_> = t.faces.iter().filter(|f| f.fingerprint.kind == FaceKind::Plane && f.fingerprint.dir.z > 0.9).collect();
    assert!(!tops.is_empty());
    // The top over the block's own part is the block's.
    let over_block = tops.iter().find(|f| f.fingerprint.centroid.x < 30.0).unwrap_or_else(|| panic!("no top over the block: {tops:?}"));
    assert_eq!(over_block.fingerprint.origin, 2, "{tops:?}");
    // The far end of the boss, out past the block, is the join's.
    let end = t.faces.iter().find(|f| f.fingerprint.kind == FaceKind::Plane && f.fingerprint.dir.x > 0.9).unwrap_or_else(|| panic!("no far end"));
    assert!((end.fingerprint.centroid.x - 60.0).abs() < 1e-6);
    assert_eq!(end.fingerprint.origin, j as i64);
}

#[test]
fn the_part_as_it_stood_before_a_feature_is_what_its_box_picks_on() {
    let mut part = block();
    let edge = edge_at(&build(&part), [20.0, 0.0, 10.0]);
    let f = part.add(fillet_kind(vec![edge], 2.0));
    part.add(FeatureKind::CutPlane(CutPlaneFeature { body: BodyId::new(2), plane: PlaneRef::world(WorldAxis::X, 30.0), keep: -1, exprs: Exprs::new() }));
    let (before, whole) = build_before(&part, f);
    ok(&before);
    ok(&whole);
    // Before the round: the block's six faces and twelve edges, the edge picked among them.
    assert_eq!(before.bodies.len(), 1);
    assert_eq!(before.bodies[0].topology.faces.len(), 6);
    assert!(before.bodies[0].topology.edges.iter().any(|e| near(e.fingerprint.mid, [20.0, 0.0, 10.0], 1e-6)));
    assert!(!before.touched.contains_key(&f));
    // The whole part: rounded and cut, as a build of it is.
    let again = build(&part);
    assert!((vol(&whole, 0) - vol(&again, 0)).abs() < 1e-9);
    assert_eq!(rounds_of(&whole, 2.0).len(), 1);
    assert!((vol(&whole, 0) - (30.0 * 20.0 * 10.0 - (4.0 - std::f64::consts::PI) * 30.0)).abs() < 1.0, "{}", vol(&whole, 0));
    // A feature not in the part: the whole part both times.
    let (all, _) = build_before(&part, 99);
    assert!((vol(&all, 0) - vol(&again, 0)).abs() < 1e-9);
}

#[test]
fn a_shell_with_no_face_open_leaves_the_body_closed_and_hollow() {
    let mut part = block();
    part.add(FeatureKind::Shell(ShellFeature { body: BodyId::new(2), faces: vec![], faces_where: vec![], thickness: 2.0, exprs: Exprs::new() }));
    let built = build(&part);
    ok(&built);
    assert!((vol(&built, 0) - (8000.0 - 36.0 * 16.0 * 6.0)).abs() < 1e-3, "{}", vol(&built, 0));
}
