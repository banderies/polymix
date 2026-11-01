// SPDX-License-Identifier: AGPL-3.0-only
//! A shell closed and hollow, opened at faces across a step, and on a body whose rounds are
//! no larger than the wall — the cases of ScanRuler's `reBuild.test.ts` ("a shell the thick
//! solid gives up on") and `reShellBudget.test.ts` ("how a shell was built"), © Stefan Hermann
//! (CNC Kitchen), AGPL-3.0-only.

use crate::build::build;
use crate::build::fillet::tests::{block, extrude, ok, plane, rect_sketch, vol};
use crate::model::*;
use crate::topo::FaceKind;

fn shell(faces: Vec<FaceRef>, thickness: f64) -> FeatureKind {
    FeatureKind::Shell(ShellFeature { body: BodyId::new(2), faces, faces_where: vec![], thickness, exprs: Exprs::new() })
}

/// A bar with a block on its end — nine flat faces.
fn bracket() -> Part {
    let mut part = Part::default();
    let bar = rect_sketch(&mut part, -48.65, 0.0, 94.7, 10.29, 0.0);
    extrude(&mut part, bar, 0.0, 7.46, BodyOp::New);
    let blk = rect_sketch(&mut part, 42.09, 0.0, 6.7, 14.11, 0.0);
    extrude(&mut part, blk, 0.0, 20.5, BodyOp::Join);
    part
}

fn face_where(part: &Part, test: impl Fn(&FaceRef) -> bool) -> FaceRef {
    let built = build(part);
    built.bodies[0].topology.faces.iter().map(|f| f.fingerprint).find(|f| f.kind == FaceKind::Plane && test(f)).unwrap_or_else(|| panic!("no such face"))
}

#[test]
fn leaves_a_body_with_no_face_open_closed_and_hollow_inside() {
    let mut part = block(40.0, 20.0, 10.0);
    let s = part.add(shell(vec![], 2.0));
    let built = build(&part);
    ok(&built);
    assert!(!built.warnings.contains_key(&s), "{:?}", built.warnings);
    assert!((vol(&built) - (8000.0 - 36.0 * 16.0 * 6.0)).abs() < 1e-3, "{}", vol(&built));
    // Two shells: the outside and the void.
    assert_eq!(built.bodies[0].body.shell_count(), 2);
    // A wall as thick as the body leaves nothing inside, and says so.
    let mut thick = block(40.0, 20.0, 10.0);
    let t = thick.add(shell(vec![], 5.0));
    let e = build(&thick).errors.get(&t).cloned().unwrap_or_default();
    assert!(e.starts_with("Shelling: "), "{e}");
}

#[test]
fn opens_the_front_the_bar_top_and_the_block_top_together() {
    let part = bracket();
    let was = vol(&build(&part));
    let front = face_where(&part, |f| f.dir.y < -0.9);
    let bar_top = face_where(&part, |f| f.dir.z > 0.9 && f.centroid.x < 0.0);
    let block_top = face_where(&part, |f| f.dir.z > 0.9 && f.centroid.x > 40.0);
    let mut shelled = part.clone();
    shelled.add(shell(vec![front, bar_top, block_top], 2.0));
    let built = build(&shelled);
    ok(&built);
    let v = vol(&built);
    // Three sides open of a 2 mm wall on a body this size: about half of it gone.
    assert!(v > 0.4 * was && v < 0.6 * was, "{v} of {was}");
    // The bar's top is open: what is left on its plane are the ends of the walls.
    let on_top: Vec<f64> = built.bodies[0].topology.faces.iter().filter(|f| f.fingerprint.kind == FaceKind::Plane && f.fingerprint.dir.z > 0.9 && (f.fingerprint.centroid.z - 7.46).abs() < 1e-3).map(|f| f.fingerprint.area).collect();
    assert!(!on_top.is_empty());
    assert!(on_top.iter().copied().fold(0.0, f64::max) < 0.3 * bar_top.area);
}

#[test]
fn opens_two_adjacent_faces_across_the_step_the_corner_between_them_gone() {
    let part = bracket();
    let was = vol(&build(&part));
    let front = face_where(&part, |f| f.dir.y < -0.9);
    let bottom = face_where(&part, |f| f.dir.z < -0.9);
    let mut shelled = part.clone();
    shelled.add(shell(vec![front, bottom], 2.0));
    let built = build(&shelled);
    ok(&built);
    // Nothing is left along the edge the two opened faces shared.
    let rod = built.bodies[0].topology.faces.iter().filter(|f| f.fingerprint.centroid.y < 2.0 + 1e-3 && f.fingerprint.centroid.z < 2.0 + 1e-3 && f.fingerprint.area > 1.0).count();
    assert_eq!(rod, 0);
    assert!(vol(&built) < was - 2.0 * (front.area + bottom.area) * 0.6);
}

#[test]
fn a_box_shelled_by_the_kernels_own_shell_says_nothing_of_how() {
    let part = block(40.0, 20.0, 10.0);
    let bottom = plane(&build(&part), [0.0, 0.0, -1.0]);
    let mut plain = part.clone();
    let s = plain.add(shell(vec![bottom], 1.0));
    let built = build(&plain);
    ok(&built);
    assert!(!built.warnings.contains_key(&s), "{:?}", built.warnings);
    assert!((vol(&built) - (8000.0 - 38.0 * 18.0 * 9.0)).abs() < 1e-3, "{}", vol(&built));
}

#[test]
fn shells_a_box_whose_rounds_are_no_larger_than_the_wall() {
    // Its four top edges rounded at 2, shelled open at the bottom at a wall of 3.
    let part = block(40.0, 20.0, 10.0);
    let first = build(&part);
    let tops: Vec<EdgeRef> = first.bodies[0].topology.edges.iter().filter(|e| (e.fingerprint.mid.z - 10.0).abs() < 1e-6).map(|e| e.fingerprint.clone()).collect();
    let bottom = plane(&first, [0.0, 0.0, -1.0]);
    let mut rounded = part.clone();
    rounded.add(FeatureKind::Fillet(FilletFeature { body: BodyId::new(2), edges: tops, edges_where: vec![], radius: 2.0, chain: false, skip_failing: false, variable: vec![], exprs: Exprs::new() }));
    let before = vol(&build(&rounded));
    let s = rounded.add(shell(vec![bottom], 3.0));
    let built = build(&rounded);
    ok(&built);
    // The outside as rounded, a 3 mm wall: what is left is the body less a core 34 × 14 × 7
    // (sharp inside where the rounds ran out) — or less, where the kernel's own shell rounds
    // the inside too.
    let v = vol(&built);
    assert!(v > before - 34.0 * 14.0 * 7.0 - 1.0 && v < before - 30.0 * 10.0 * 7.0, "{v}");
    if let Some(w) = built.warnings.get(&s) {
        let w = w.join(" ");
        assert!(w.starts_with("Built by cutting back, with booleans: 4 rounds no larger than the wall") && w.contains(" The kernel's thick solid gave up on it: "), "{w}");
    }
}

/// A plane face of a topology, and its four sides as edges.
fn face_with_sides(t: &mut crate::topo::Topology, at: glam::DVec3, w: f64, h: f64) {
    let hash = t.faces.len() as u32;
    let fingerprint = FaceRef { kind: FaceKind::Plane, centroid: at, dir: glam::DVec3::X, area: w * h, radius: None, center: None, origin: 2 };
    t.faces.push(crate::topo::FaceInfo { hash, fingerprint, full_turn: None, point: at });
    for length in [w, h, w, h] {
        let e = crate::topo::EdgeFingerprint { kind: crate::topo::EdgeKind::Line, mid: at, dir: glam::DVec3::Y, length, radius: None, faces: vec![], origin: 2 };
        t.edges.push(crate::topo::EdgeInfo { hash: t.edges.len() as u32, fingerprint: e, faces: vec![hash], ends: [at, at], end_tangents: [glam::DVec3::Y, glam::DVec3::Y], polyline: vec![at, at] });
    }
}

#[test]
fn warns_of_a_sliver_face_the_shell_made_by_its_index() {
    // The body's own face, and the shell's: a wall, and a step 5 µm wide beside it.
    let mut before = crate::topo::Topology::default();
    face_with_sides(&mut before, glam::DVec3::new(0.0, 10.0, 5.0), 20.0, 10.0);
    let mut after = before.clone();
    face_with_sides(&mut after, glam::DVec3::new(2.0, 10.0, 5.0), 16.0, 10.0);
    face_with_sides(&mut after, glam::DVec3::new(20.0, 20.0, 5.0), 0.005, 10.0);
    let w = super::slivers_left(&before, &after).unwrap_or_default();
    assert_eq!(w, "The shell left 1 sliver face, under 0.01 mm wide or 0.01 mm²: face 2 (0.05 mm² at (20.0, 20.0, 5.0)) — a round on the edges beside it will likely fail.");
    // The faces the body had already are not the shell's to warn of.
    assert_eq!(super::slivers_left(&after, &after), None);
}

/// The body a part's last feature left, with its faces and edges.
fn last_body(part: &Part) -> (solidcraft_kernel::Body, crate::topo::Topology) {
    let built = build(part);
    (built.bodies[0].body.clone(), (*built.bodies[0].topology).clone())
}

#[test]
fn cuts_back_a_wall_made_on_the_body_without_its_rounds_leaving_the_inside_sharp() {
    let part = block(40.0, 20.0, 10.0);
    let first = build(&part);
    let tops: Vec<EdgeRef> = first.bodies[0].topology.edges.iter().filter(|e| (e.fingerprint.mid.z - 10.0).abs() < 1e-6).map(|e| e.fingerprint.clone()).collect();
    let mut rounded = part.clone();
    rounded.add(FeatureKind::Fillet(FilletFeature { body: BodyId::new(2), edges: tops, edges_where: vec![], radius: 2.0, chain: false, skip_failing: false, variable: vec![], exprs: Exprs::new() }));
    let (body, t) = last_body(&rounded);
    let bottom = t.faces.iter().position(|f| f.fingerprint.kind == FaceKind::Plane && f.fingerprint.dir.z < -0.9).unwrap_or_else(|| panic!("no bottom"));
    match super::shell_sharp_inside(&body, &t, &[bottom], 3.0) {
        super::Way::Made(made, note) => {
            assert!(note.starts_with("Built by cutting back, with booleans: 4 rounds no larger than the wall taken off, the wall made on the body without them and cut back to its outside — the inside is left sharp there."), "{note}");
            // The outside as rounded, less a sharp core 34 × 14 × 7.
            let v = crate::build::volume(&made).unwrap_or(f64::NAN);
            let want = crate::build::volume(&body).unwrap_or(f64::NAN) - 34.0 * 14.0 * 7.0;
            assert!((v - want).abs() < 0.5, "{v} vs {want}");
        }
        super::Way::Why(w) => panic!("{w}"),
        super::Way::None => panic!("no rounds no larger than the wall"),
    }
    // No round no larger than the wall: the way does not apply.
    let (plain, pt) = last_body(&part);
    assert!(matches!(super::shell_sharp_inside(&plain, &pt, &[0], 3.0), super::Way::None));
}

#[test]
fn opens_a_bracket_the_long_way_by_offset_and_cut() {
    let part = bracket();
    let (body, t) = last_body(&part);
    let face = |test: &dyn Fn(&FaceRef) -> bool| t.faces.iter().position(|f| f.fingerprint.kind == FaceKind::Plane && test(&f.fingerprint)).unwrap_or_else(|| panic!("no such face"));
    let open = [face(&|f| f.dir.y < -0.9), face(&|f| f.dir.z > 0.9 && f.centroid.x < 0.0), face(&|f| f.dir.z > 0.9 && f.centroid.x > 40.0)];
    match super::shell_by_offset_cut(&body, &t, &open, 2.0) {
        super::Way::Made(made, _) => {
            let (v, was) = (crate::build::volume(&made).unwrap_or(f64::NAN), crate::build::volume(&body).unwrap_or(f64::NAN));
            assert!(v > 0.4 * was && v < 0.6 * was, "{v} of {was}");
        }
        super::Way::Why(w) => panic!("{w}"),
        super::Way::None => panic!("nothing to open"),
    }
    // Nothing to open: the way does not apply.
    assert!(matches!(super::shell_by_offset_cut(&body, &t, &[], 2.0), super::Way::None));
}

#[test]
fn says_the_reason_that_says_most() {
    let r = |why: &str, raw: bool| super::Reason { why: why.into(), raw };
    // A fact about the body before the kernel's words, those before a way that only gave up.
    assert_eq!(super::most_specific(&[r("the thick solid handed the body back without hollowing it.", false), r("BRep_Tool:: no parameter on edge", true), r("the wall is as thick as the body — nothing is left inside to take away.", false)]), "the wall is as thick as the body — nothing is left inside to take away");
    assert_eq!(super::most_specific(&[r("the thick solid handed the body back without hollowing it.", false), r("BRep_Tool:: no parameter on edge", true)]), "the kernel refused (BRep_Tool:: no parameter on edge)");
    assert_eq!(super::most_specific(&[r("the thick solid handed the body back without hollowing it.", false)]), "the thick solid handed the body back without hollowing it");
    // The core of a wall as thick as the body.
    let (body, t) = last_body(&block(40.0, 20.0, 10.0));
    let e = super::inner_core(&body, &t, 5.0).err().unwrap_or_default();
    assert!(e.starts_with("the wall is as thick as the body") || e.starts_with("the kernel's offset of the body inward by the wall"), "{e}");
    assert!(super::inner_core(&body, &t, 2.0).is_ok());
}

#[test]
fn makes_the_wall_under_a_round_face_by_thickening_it_into_the_body() {
    // A disc 10 in radius and 10 high: the wall under its round side is a ring 2 thick, inside
    // the body — what the long way cuts off over a round face opened, or leaves standing beside
    // one (ScanRuler's `wallUnder`, a face thickened into the body).
    let mut part = Part::default();
    let s = crate::build::fillet::tests::circle_sketch(&mut part, glam::DVec2::ZERO, 10.0, 0.0);
    extrude(&mut part, s, 0.0, 10.0, BodyOp::New);
    let (body, t) = last_body(&part);
    // The kernel makes the round side of halves: each one's wall is its half of the ring.
    let sides: Vec<usize> = t.faces.iter().enumerate().filter(|(_, f)| f.fingerprint.kind == FaceKind::Cylinder).map(|(i, _)| i).collect();
    assert!(!sides.is_empty(), "no round side");
    let ring = std::f64::consts::PI * (100.0 - 64.0) * 10.0;
    let mut core = body.clone();
    let mut v = 0.0;
    for side in &sides {
        let wall = super::wall_under(&body, &t, *side, 2.0).unwrap_or_else(|| panic!("no wall under the round side"));
        v += crate::build::volume(&wall).unwrap_or(f64::NAN);
        // It lies in the body: the body less the walls is the core within it.
        core = solidcraft_kernel::boolean(&core, &wall, solidcraft_kernel::BoolOp::Cut).ok().flatten().unwrap_or_else(|| panic!("nothing left"));
    }
    assert!((v - ring).abs() < 0.01 * ring, "{v} vs {ring}");
    let left = crate::build::volume(&core).unwrap_or(f64::NAN);
    assert!((left - std::f64::consts::PI * 64.0 * 10.0).abs() < 0.01 * ring, "{left}");
    // A flat face is pushed straight in.
    let top = t.faces.iter().position(|f| f.fingerprint.kind == FaceKind::Plane && f.fingerprint.dir.z > 0.9).unwrap_or_else(|| panic!("no top"));
    let slab = super::wall_under(&body, &t, top, 2.0).unwrap_or_else(|| panic!("no wall under the top"));
    assert!((crate::build::volume(&slab).unwrap_or(f64::NAN) - std::f64::consts::PI * 100.0 * 2.0).abs() < 0.5);
}
