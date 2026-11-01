// SPDX-License-Identifier: AGPL-3.0-only
//! Faces deleted and healed over — a round, a bore, a blind hole, a boss, every round of a block
//! — with what one click on a face takes: the cases of ScanRuler's `reChamferDelete.test.ts`
//! ("faces deleted and healed over"), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.

use glam::DVec2;

use crate::build::fillet::tests::{block, circle_sketch, edge_at, extrude, ok, plane, vol};
use crate::build::{body_mesh, build, Built};
use crate::face_group::{delete_face_group, edge_joins, FacedTriangles};
use crate::model::*;
use crate::topo::{EdgeKind, FaceInfo, FaceKind};

/// A block 10 on a side, from the origin.
fn cube() -> Part {
    block(10.0, 10.0, 10.0)
}

/// What a click on a face takes, read off the built body as the panel reads it.
fn group_at(built: &Built, face: u32, tangent: bool) -> Vec<FaceInfo> {
    let b = &built.bodies[0];
    let mesh = body_mesh(&b.body).unwrap_or_default();
    let joins = edge_joins(FacedTriangles { positions: &mesh.positions, normals: &mesh.normals, triangles: &mesh.triangles, tri_face: &mesh.tri_face }, &b.topology);
    delete_face_group(&b.topology, face, |e| joins.get(&e).copied(), tangent).into_iter().filter_map(|h| b.topology.face(h).cloned()).collect()
}

fn deleting(faces: &[FaceInfo]) -> FeatureKind {
    FeatureKind::DeleteFace(DeleteFaceFeature { body: BodyId::new(2), faces: faces.iter().map(|f| f.fingerprint).collect(), faces_where: vec![] })
}

fn fillet(edges: Vec<EdgeRef>, radius: f64) -> FeatureKind {
    FeatureKind::Fillet(FilletFeature { body: BodyId::new(2), edges, edges_where: vec![], radius, chain: true, skip_failing: false, variable: vec![], exprs: Exprs::new() })
}

#[test]
fn takes_a_round_off_a_block_the_faces_either_side_run_on_to_the_corner() {
    let edge = edge_at(&build(&cube()), [5.0, 0.0, 10.0]);
    let mut rounded = cube();
    rounded.add(fillet(vec![edge], 2.0));
    let r = build(&rounded);
    ok(&r);
    assert!(vol(&r) < 1000.0 - 0.5);
    let round = r.bodies[0].topology.faces.iter().find(|f| f.fingerprint.kind == FaceKind::Cylinder).cloned().unwrap_or_else(|| panic!("no round"));
    assert_eq!(group_at(&r, round.hash, true).iter().map(|f| f.hash).collect::<Vec<_>>(), vec![round.hash]);
    let mut out = rounded.clone();
    out.add(deleting(&[round]));
    let b = build(&out);
    ok(&b);
    assert!((vol(&b) - 1000.0).abs() < 1e-3, "{}", vol(&b));
    assert_eq!(b.bodies[0].body.face_count(), 6);
}

#[test]
fn takes_a_bore_off_whole_from_a_click_on_its_wall_and_a_blind_hole_with_its_floor() {
    for from in [-1.0, 5.0] {
        let mut holed = cube();
        let s = circle_sketch(&mut holed, DVec2::new(5.0, 5.0), 2.0, 0.0);
        extrude(&mut holed, s, from, 11.0, BodyOp::Cut);
        let b = build(&holed);
        ok(&b);
        let want = 1000.0 - std::f64::consts::PI * 4.0 * (10.0 - f64::max(0.0, from));
        assert!((vol(&b) - want).abs() < 0.1, "{} vs {want}", vol(&b));
        let t = &b.bodies[0].topology;
        let wall = t.faces.iter().find(|f| f.fingerprint.kind == FaceKind::Cylinder).cloned().unwrap_or_else(|| panic!("no bore"));
        let taken = group_at(&b, wall.hash, true);
        // The whole wall, its halves if it is split; a blind hole's floor.
        let walls = t.faces.iter().filter(|f| f.fingerprint.kind == FaceKind::Cylinder).count();
        assert_eq!(taken.iter().filter(|f| f.fingerprint.kind == FaceKind::Cylinder).count(), walls);
        assert_eq!(taken.iter().any(|f| f.fingerprint.kind == FaceKind::Plane), from > 0.0, "from {from}: {taken:?}");
        let mut out = holed.clone();
        out.add(deleting(&taken));
        let d = build(&out);
        ok(&d);
        assert!((vol(&d) - 1000.0).abs() < 1e-3, "from {from}: {}", vol(&d));
    }
}

#[test]
fn takes_a_boss_off_from_a_click_on_its_wall_the_round_at_its_foot_and_its_top_with_it() {
    let mut made = cube();
    if let Some(FeatureKind::Extrude(e)) = made.feature_mut(2).map(|f| &mut f.kind) {
        e.extent = ExtrudeExtent::Distance { from: 0.0, to: 5.0 };
    }
    let s = circle_sketch(&mut made, DVec2::new(5.0, 5.0), 2.0, 0.0);
    extrude(&mut made, s, 5.0, 10.0, BodyOp::Join);
    let first = build(&made);
    ok(&first);
    let foot = first.bodies[0].topology.edges.iter().find(|e| e.fingerprint.kind == EdgeKind::Circle && (e.fingerprint.mid.z - 5.0).abs() < 1e-6).map(|e| e.fingerprint.clone()).unwrap_or_else(|| panic!("no foot"));
    let mut rounded = made.clone();
    rounded.add(fillet(vec![foot], 1.0));
    let r = build(&rounded);
    ok(&r);
    let wall = r.bodies[0].topology.faces.iter().find(|f| f.fingerprint.kind == FaceKind::Cylinder).cloned().unwrap_or_else(|| panic!("no wall"));
    let taken = group_at(&r, wall.hash, true);
    assert!(taken.iter().any(|f| f.fingerprint.kind == FaceKind::Torus), "{taken:?}");
    assert!(taken.iter().any(|f| f.fingerprint.kind == FaceKind::Plane && (f.fingerprint.centroid.z - 10.0).abs() < 1e-6));
    // The plate's top, which the round runs into, stays.
    assert!(!taken.iter().any(|f| f.fingerprint.kind == FaceKind::Plane && (f.fingerprint.centroid.z - 5.0).abs() < 1e-6));
    let mut out = rounded.clone();
    out.add(deleting(&taken));
    let d = build(&out);
    ok(&d);
    assert!((vol(&d) - 500.0).abs() < 1e-3, "{}", vol(&d));
}

#[test]
fn takes_every_round_of_a_block_from_a_click_on_one_the_flats_left_and_heals_it_sharp() {
    let all: Vec<EdgeRef> = build(&cube()).bodies[0].topology.edges.iter().map(|e| e.fingerprint.clone()).collect();
    assert_eq!(all.len(), 12);
    let mut rounded = cube();
    rounded.add(fillet(all, 1.0));
    let r = build(&rounded);
    ok(&r);
    let rounds: Vec<u32> = r.bodies[0].topology.faces.iter().filter(|f| f.fingerprint.kind != FaceKind::Plane).map(|f| f.hash).collect();
    let one = r.bodies[0].topology.faces.iter().find(|f| f.fingerprint.kind == FaceKind::Cylinder).map(|f| f.hash).unwrap_or_else(|| panic!("no round"));
    let taken = group_at(&r, one, true);
    let mut got: Vec<u32> = taken.iter().map(|f| f.hash).collect();
    got.sort();
    let mut want = rounds.clone();
    want.sort();
    assert_eq!(got, want);
    // Off, the click takes the one round.
    assert_eq!(group_at(&r, one, false).iter().map(|f| f.hash).collect::<Vec<_>>(), vec![one]);
    let mut out = rounded.clone();
    out.add(deleting(&taken));
    let d = build(&out);
    ok(&d);
    assert!((vol(&d) - 1000.0).abs() < 1e-2, "{}", vol(&d));
}

#[test]
fn says_so_when_a_face_cannot_be_healed_over_and_names_a_face_that_is_gone() {
    let b = build(&cube());
    let top = plane(&b, [0.0, 0.0, 1.0]);
    let mut missing = cube();
    let m = missing.add(FeatureKind::DeleteFace(DeleteFaceFeature { body: BodyId::new(2), faces: vec![FaceRef { centroid: glam::DVec3::new(50.0, 50.0, 50.0), ..top }], faces_where: vec![] }));
    assert!(build(&missing).errors.get(&m).is_some_and(|e| e.contains("is gone")));
    let mut none = cube();
    let n = none.add(FeatureKind::DeleteFace(DeleteFaceFeature { body: BodyId::new(2), faces: vec![], faces_where: vec![] }));
    assert!(build(&none).errors.get(&n).is_some_and(|e| e.contains("Pick the faces")));
}
