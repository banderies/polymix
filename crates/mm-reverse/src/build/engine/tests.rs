// SPDX-License-Identifier: AGPL-3.0-only
//! The engine's tests: the cases of ScanRuler's `reBuildCache.test.ts`, `reBuildReuse.test.ts`,
//! `reParallel.test.ts`, `reSplitSays.test.ts`, `reCutElsewhere.test.ts` and
//! `reRoundDiagnose.test.ts`, © Stefan Hermann (CNC Kitchen), AGPL-3.0-only — a build taken up
//! from a kept one comes out as a build from nothing, and plays only what a change reached.

use glam::{DVec2, DVec3};

use super::*;
use crate::build::{build, default_regions, group_steps, split_words, volume, StepGroup};
use crate::frame::WorldAxis;
use crate::model::*;
use crate::regions::find_regions;
use crate::sketch::{Shape, Sketch};
use crate::topo::EdgeKind;

fn sketch_of(part: &mut Part, s: Sketch, z: f64) -> (FeatureId, Vec<String>) {
    let regions = default_regions(&find_regions(&s));
    (part.add(FeatureKind::Sketch(SketchFeature::on_world(WorldAxis::Z, z, s))), regions)
}

fn rect(part: &mut Part, x0: f64, y0: f64, x1: f64, y1: f64) -> (FeatureId, Vec<String>) {
    let mut s = Sketch::default();
    s.add_rectangle(DVec2::new(x0, y0), DVec2::new(x1, y1));
    sketch_of(part, s, 0.0)
}

fn circle(part: &mut Part, c: [f64; 2], r: f64) -> (FeatureId, Vec<String>) {
    let mut s = Sketch::default();
    s.add_shape(Shape::Circle { c: DVec2::from(c), r });
    sketch_of(part, s, 0.0)
}

fn outline(part: &mut Part, pts: &[[f64; 2]]) -> (FeatureId, Vec<String>) {
    let mut s = Sketch::default();
    for (i, a) in pts.iter().enumerate() {
        let b = pts[(i + 1) % pts.len()];
        s.add_shape(Shape::Line { a: DVec2::from(*a), b: DVec2::from(b) });
    }
    sketch_of(part, s, 0.0)
}

fn extrude(part: &mut Part, sketch: (FeatureId, Vec<String>), from: f64, to: f64, op: BodyOp) -> FeatureId {
    part.add(FeatureKind::Extrude(ExtrudeFeature::new(sketch.0, sketch.1, ExtrudeExtent::Distance { from, to }, op)))
}

fn ask(part: &Part) -> BuildAsk {
    BuildAsk { upto: part.features.len(), before: None, reuse: true, parallel: false }
}

fn run(e: &mut Engine, part: &Part) -> Built {
    e.build(part, &ask(part), &BuildControl::new()).map(|b| b.built).unwrap_or_default()
}

/// What a build came to, to hold two builds against each other: each body's id, name, faces
/// and volume to a thousandth, and what each feature said.
fn signature(b: &Built) -> (Vec<(BodyId, String, usize, i64)>, Vec<(FeatureId, String)>, Vec<(FeatureId, Vec<String>)>) {
    let bodies = b.bodies.iter().map(|x| (x.id, x.name.clone(), x.body.face_count(), (volume(&x.body).unwrap_or(0.0) * 1000.0).round() as i64)).collect();
    let mut errors: Vec<(FeatureId, String)> = b.errors.iter().map(|(k, v)| (*k, v.clone())).collect();
    errors.sort();
    let mut warnings: Vec<(FeatureId, Vec<String>)> = b.warnings.iter().map(|(k, v)| (*k, v.clone())).collect();
    warnings.sort();
    (bodies, errors, warnings)
}

/// The upright edge of a body at (x, y).
fn upright(b: &Built, id: BodyId, x: f64, y: f64) -> EdgeRef {
    let body = b.body(id).unwrap_or_else(|| panic!("no body {id}"));
    body.topology.edges.iter().find(|e| e.fingerprint.kind == EdgeKind::Line && e.fingerprint.dir.z.abs() > 0.99 && (e.fingerprint.mid.x - x).abs() < 1e-6 && (e.fingerprint.mid.y - y).abs() < 1e-6).map(|e| e.fingerprint.clone()).unwrap_or_else(|| panic!("no upright edge at ({x}, {y})"))
}

fn round(part: &mut Part, body: BodyId, edge: EdgeRef, radius: f64) -> FeatureId {
    part.add(FeatureKind::Fillet(FilletFeature { body, edges: vec![edge], edges_where: vec![], radius, chain: false, skip_failing: false, variable: vec![], exprs: Exprs::new() }))
}

fn set_radius(part: &mut Part, id: FeatureId, radius: f64) {
    if let Some(FeatureKind::Fillet(x)) = part.feature_mut(id).map(|f| &mut f.kind) {
        x.radius = radius;
    }
}

/// A plate 40 × 20 × 5 with a hole, then a round on two of its upright corners — six features.
fn plate() -> (Part, FeatureId) {
    let mut part = Part::default();
    let outline = rect(&mut part, 0.0, 0.0, 40.0, 20.0);
    let p = extrude(&mut part, outline, 0.0, 5.0, BodyOp::New);
    let hole = circle(&mut part, [20.0, 10.0], 3.0);
    extrude(&mut part, hole, -1.0, 6.0, BodyOp::Cut);
    let first = build(&part);
    let a = upright(&first, BodyId::new(p), 0.0, 0.0);
    let b = upright(&first, BodyId::new(p), 40.0, 0.0);
    round(&mut part, BodyId::new(p), a, 2.0);
    let last = round(&mut part, BodyId::new(p), b, 2.0);
    (part, last)
}

#[test]
fn plays_nothing_for_a_value_set_back_however_many_edits_came_between() {
    let (part, last) = plate();
    let n = part.features.len();
    let mut e = Engine::new();
    let first = run(&mut e, &part);
    assert!(first.errors.is_empty(), "{:?}", first.errors);
    assert_eq!((e.last_steps.recomputed, e.last_steps.restored, e.last_steps.of), (n, 0, n));
    for r in [2.5, 3.0, 3.5] {
        let mut p = part.clone();
        set_radius(&mut p, last, r);
        let built = run(&mut e, &p);
        assert_eq!((e.last_steps.recomputed, e.last_steps.restored), (1, n - 1), "radius {r}");
        assert_eq!(signature(&built), signature(&build(&p)), "radius {r}");
    }
    let back = run(&mut e, &part);
    assert_eq!((e.last_steps.recomputed, e.last_steps.restored), (0, n));
    assert_eq!(signature(&back), signature(&first));
    // The radius before it, too.
    let mut p = part.clone();
    set_radius(&mut p, last, 3.0);
    run(&mut e, &p);
    assert_eq!(e.last_steps.recomputed, 0);
}

#[test]
fn moves_the_rollback_bar_anywhere_and_back_as_a_state_looked_up() {
    let (part, _) = plate();
    let n = part.features.len();
    let mut e = Engine::new();
    run(&mut e, &part);
    for upto in (0..n).rev() {
        let built = e.build(&part, &BuildAsk { upto, ..ask(&part) }, &BuildControl::new()).map(|b| b.built).unwrap_or_default();
        assert_eq!((e.last_steps.recomputed, e.last_steps.restored, e.last_steps.of), (0, upto, upto));
        assert_eq!(signature(&built), signature(&crate::build::build_upto(&part, upto)), "rolled to {upto}");
    }
    run(&mut e, &part);
    assert_eq!(e.last_steps.recomputed, 0);
}

#[test]
fn hands_back_the_part_before_a_feature_from_the_same_build() {
    let (part, last) = plate();
    let mut e = Engine::new();
    let out = e.build(&part, &BuildAsk { before: Some(last), ..ask(&part) }, &BuildControl::new()).unwrap_or_else(|| panic!("stopped"));
    let (before, _) = crate::build::build_before(&part, last);
    assert_eq!(signature(out.before.as_ref().unwrap_or_else(|| panic!("no before"))), signature(&before));
    assert_eq!(signature(&out.built), signature(&build(&part)));
}

#[test]
fn keeps_what_it_holds_within_its_budget_the_build_standing_whatever_it_holds() {
    let (part, last) = plate();
    let n = part.features.len();
    let mut e = Engine::new();
    run(&mut e, &part);
    let one = e.cache_held();
    assert!(one.0 > 0);
    for r in [2.5, 3.0, 3.5, 4.0] {
        let mut p = part.clone();
        set_radius(&mut p, last, r);
        run(&mut e, &p);
    }
    let five = e.cache_held();
    assert!(five.0 > one.0);
    // A byte under it: the build standing — 3.5 again — is kept, the oldest used of the others
    // goes; the rest stay.
    e.budget = five.0 - 1;
    let mut p = part.clone();
    set_radius(&mut p, last, 3.5);
    run(&mut e, &p);
    assert_eq!(e.last_steps.recomputed, 0);
    assert!(e.cache_held().0 <= e.budget);
    e.budget = usize::MAX;
    for r in [3.0, 4.0] {
        let mut p = part.clone();
        set_radius(&mut p, last, r);
        run(&mut e, &p);
        assert_eq!(e.last_steps.recomputed, 0, "radius {r}");
    }
    // Nothing at all: the build standing alone.
    e.budget = 0;
    let mut p = part.clone();
    set_radius(&mut p, last, 4.0);
    run(&mut e, &p);
    assert_eq!(e.cache_held().2, n);
    run(&mut e, &part);
    assert_eq!((e.last_steps.recomputed, e.last_steps.restored), (1, n - 1));
}

/// A plate with a hole cut into it alone; a block of its own, rounded at a corner; and a boss
/// joined to whatever it meets — the block, but the join looks at every body to find that out.
fn tree(width: f64, corner: Option<EdgeRef>) -> Part {
    let mut part = Part::default();
    let outline = rect(&mut part, 0.0, 0.0, width, 20.0);
    let p = extrude(&mut part, outline, 0.0, 5.0, BodyOp::New);
    let hole = circle(&mut part, [20.0, 10.0], 3.0);
    extrude(&mut part, hole, -1.0, 6.0, BodyOp::Cut);
    let block = rect(&mut part, 60.0, 0.0, 70.0, 10.0);
    let b = extrude(&mut part, block, 0.0, 6.0, BodyOp::New);
    if let Some(edge) = corner {
        round(&mut part, BodyId::new(b), edge, 1.0);
    }
    let boss = rect(&mut part, 62.0, 2.0, 64.0, 4.0);
    extrude(&mut part, boss, 5.0, 8.0, BodyOp::Join);
    assert_eq!(p, 2);
    part
}

#[test]
fn takes_over_the_steps_past_a_change_that_the_change_did_not_reach_and_plays_those_it_did() {
    let plain = tree(40.0, None);
    let corner = upright(&build(&plain), BodyId::new(6), 60.0, 0.0);
    let first = tree(40.0, Some(corner.clone()));
    let mut e = Engine::new();
    run(&mut e, &first);
    let round_id = first.features.iter().find(|f| matches!(f.kind, FeatureKind::Fillet(_))).map(|f| f.id).unwrap_or(0);
    for width in [50.0, 60.0] {
        let wider = tree(width, Some(corner.clone()));
        let built = run(&mut e, &wider);
        assert!(built.errors.is_empty(), "{:?}", built.errors);
        let played: Vec<FeatureId> = e.last_steps.played.iter().map(|(id, _)| *id).collect();
        // The plate's sketch, its extrude and the cut play again, and the boss's join, which
        // read every body; the block and its round are taken over.
        assert!(!played.contains(&round_id), "{played:?}");
        assert!(!played.contains(&6), "{played:?}");
        assert!(played.contains(&1) && played.contains(&2) && played.contains(&4), "{played:?}");
        assert!(played.contains(&(round_id + 2)), "the join plays again: {played:?}");
        // The same tree built from nothing comes out the same.
        assert_eq!(signature(&built), signature(&build(&wider)), "width {width}");
    }
}

#[test]
fn a_build_without_reuse_plays_every_step_and_keeps_nothing() {
    let (part, _) = plate();
    let n = part.features.len();
    let mut e = Engine::new();
    let a = BuildAsk { reuse: false, ..ask(&part) };
    e.build(&part, &a, &BuildControl::new());
    e.build(&part, &a, &BuildControl::new());
    assert_eq!(e.last_steps.recomputed, n);
    run(&mut e, &part);
    assert_eq!(e.last_steps.recomputed, n);
}

#[test]
fn tells_steps_apart_that_json_alone_would_not_and_reads_the_sketches_a_step_names() {
    let key = |depth: f64| {
        let mut part = Part::default();
        part.add(FeatureKind::Shell(ShellFeature { body: BodyId::new(1), faces: vec![], faces_where: vec![], thickness: depth, exprs: Exprs::new() }));
        let steps: Vec<&Feature> = part.features.iter().collect();
        step_keys(&part, &steps)[0]
    };
    assert_ne!(key(f64::INFINITY), key(f64::NEG_INFINITY));
    assert_ne!(key(f64::INFINITY), key(f64::NAN));
    assert_eq!(key(1.0), key(1.0));
    // An extrusion reads its sketch: the sketch changed is the step changed, and the steps
    // before it are not.
    let (part, _) = plate();
    let steps: Vec<&Feature> = part.features.iter().collect();
    let keys = step_keys(&part, &steps);
    let mut moved = part.clone();
    if let Some(s) = moved.sketch_mut(3) {
        s.sketch.add_shape(Shape::Point { p: DVec2::ZERO });
    }
    let steps: Vec<&Feature> = moved.features.iter().collect();
    let other = step_keys(&moved, &steps);
    assert_eq!(other[0], keys[0]);
    assert_eq!(other[1], keys[1]);
    assert_ne!(other[2], keys[2]);
    assert_ne!(other[3], keys[3]);
    assert_eq!(other[4], keys[4]);
}

#[test]
fn says_which_features_changed_since_a_round_that_built_fails() {
    let (part, last) = plate();
    let mut e = Engine::new();
    run(&mut e, &part);
    // The plate made narrow: the far corner's round has lost its edge.
    let mut thin = part.clone();
    if let Some(s) = thin.sketch_mut(1) {
        s.sketch = Sketch::default();
        s.sketch.add_rectangle(DVec2::new(0.0, 0.0), DVec2::new(3.0, 20.0));
    }
    let regions = default_regions(&find_regions(&thin.sketch(1).map(|s| s.sketch.clone()).unwrap_or_default()));
    if let Some(FeatureKind::Extrude(x)) = thin.feature_mut(2).map(|f| &mut f.kind) {
        x.regions = regions;
    }
    let built = run(&mut e, &thin);
    let said = built.errors.get(&last).cloned().unwrap_or_default();
    assert!(said.ends_with("It built in the last build; changed before it since: Sketch 1, Extrude 1."), "{said}");
    // A build from nothing has no build before it to say so of.
    assert!(!build(&thin).errors.get(&last).is_some_and(|m| m.contains("last build")));
}

#[test]
fn stops_between_steps_when_asked() {
    let (part, _) = plate();
    let mut e = Engine::new();
    let control = BuildControl::new();
    control.stop();
    assert!(e.build(&part, &ask(&part), &control).is_none());
    assert!(control.in_play().is_none());
}

/// A plate and a block, each a body of its own and each rounded on an upright edge — two
/// groups with a round each.
fn two_parts(plate_width: f64, block_width: f64) -> Part {
    let mut part = Part::default();
    let p = rect(&mut part, 0.0, 0.0, plate_width, 20.0);
    let pe = extrude(&mut part, p, 0.0, 5.0, BodyOp::New);
    let b = rect(&mut part, 60.0, 0.0, 60.0 + block_width, 10.0);
    let be = extrude(&mut part, b, 0.0, 6.0, BodyOp::New);
    let first = build(&part);
    let a = upright(&first, BodyId::new(pe), 0.0, 0.0);
    let c = upright(&first, BodyId::new(be), 60.0, 0.0);
    round(&mut part, BodyId::new(pe), a, 1.0);
    round(&mut part, BodyId::new(be), c, 1.0);
    part
}

#[test]
fn puts_steps_on_bodies_of_their_own_apart_and_steps_that_read_every_body_with_all_of_them() {
    let part = two_parts(40.0, 10.0);
    let steps: Vec<&Feature> = part.features.iter().collect();
    let changed: Vec<bool> = steps.iter().map(|f| f.id == 2).collect();
    let g = group_steps(&part, &steps, &changed);
    let ids = |gr: &StepGroup| gr.steps.iter().map(|i| steps[*i].id).collect::<Vec<_>>();
    assert_eq!(g.groups.iter().map(ids).collect::<Vec<_>>(), vec![vec![1, 2, 5], vec![3, 4, 6]]);
    // The plate changed: the plate and its round to play, the block and its round not.
    assert_eq!(steps.iter().zip(&g.dirty).filter(|(_, d)| **d).map(|(f, _)| f.id).collect::<Vec<_>>(), vec![2, 5]);
    // A boss joined to whatever it meets reads every body: one group.
    let mut boss = part.clone();
    let s = rect(&mut boss, 1.0, 1.0, 2.0, 2.0);
    extrude(&mut boss, s, 0.0, 9.0, BodyOp::Join);
    let steps: Vec<&Feature> = boss.features.iter().collect();
    assert_eq!(group_steps(&boss, &steps, &vec![true; steps.len()]).groups.len(), 1);
    // A Combine of the two: one group; the block gone into the plate, a shell on it later is in
    // it too.
    let mut combined = part.clone();
    combined.add(FeatureKind::Boolean(BooleanFeature { op: BooleanOp::Join, body: BodyId::new(2), tool: BodyId::new(4), keep: false }));
    combined.add(FeatureKind::Shell(ShellFeature { body: BodyId::new(4), faces: vec![], faces_where: vec![], thickness: 1.0, exprs: Exprs::new() }));
    let steps: Vec<&Feature> = combined.features.iter().collect();
    assert_eq!(group_steps(&combined, &steps, &vec![true; steps.len()]).groups.len(), 1);
}

#[test]
fn builds_groups_on_helper_threads_and_comes_out_as_a_build_on_one() {
    let mut e = Engine::new();
    e.helpers = 1;
    e.share_min_ms = 0.0;
    let par = |part: &Part| BuildAsk { parallel: true, ..ask(part) };
    for (pw, bw) in [(40.0, 10.0), (50.0, 10.0), (60.0, 14.0), (50.0, 14.0)] {
        let part = two_parts(pw, bw);
        let built = e.build(&part, &par(&part), &BuildControl::new()).map(|b| b.built).unwrap_or_default();
        assert!(built.errors.is_empty(), "{:?}", built.errors);
        assert_eq!(signature(&built), signature(&build(&part)), "{pw} × {bw}");
    }
    // The plate of one build, the block of another: every step taken over, nothing played.
    let part = two_parts(40.0, 14.0);
    e.build(&part, &par(&part), &BuildControl::new());
    let part = two_parts(60.0, 10.0);
    e.build(&part, &par(&part), &BuildControl::new());
    assert_eq!(e.last_steps.recomputed, 0, "{:?}", e.last_steps);
}

#[test]
fn every_build_taken_up_comes_out_as_a_build_from_nothing() {
    // A run of edits, each held against a build from nothing.
    let (part, last) = plate();
    let mut e = Engine::new();
    let mut p = part.clone();
    let edits: Vec<Box<dyn Fn(&mut Part)>> = vec![
        Box::new(move |p: &mut Part| set_radius(p, last, 1.5)),
        Box::new(|p: &mut Part| {
            if let Some(f) = p.feature_mut(5) {
                f.flags.suppressed = true;
            }
        }),
        Box::new(|p: &mut Part| {
            if let Some(s) = p.sketch_mut(3) {
                s.sketch = Sketch::default();
                s.sketch.add_shape(Shape::Circle { c: DVec2::new(15.0, 8.0), r: 2.0 });
            }
            let regions = default_regions(&find_regions(&p.sketch(3).map(|s| s.sketch.clone()).unwrap_or_default()));
            if let Some(FeatureKind::Extrude(x)) = p.feature_mut(4).map(|f| &mut f.kind) {
                x.regions = regions;
            }
        }),
        Box::new(|p: &mut Part| {
            if let Some(f) = p.feature_mut(5) {
                f.flags.suppressed = false;
            }
        }),
        Box::new(|p: &mut Part| p.rolled_to = Some(3)),
        Box::new(|p: &mut Part| p.rolled_to = None),
    ];
    for (i, edit) in edits.iter().enumerate() {
        edit(&mut p);
        let upto = p.rolled_to.unwrap_or(p.features.len());
        let built = e.build(&p, &BuildAsk { upto, ..ask(&p) }, &BuildControl::new()).map(|b| b.built).unwrap_or_default();
        assert_eq!(signature(&built), signature(&crate::build::build(&p)), "edit {i}");
    }
}

/// Two legs 2 × 5 under a bar 10 × 2, and a lump on the bar over one leg — so the body's
/// centroid lies over that leg. Cut at y 5 below the bar, the legs are the same size.
fn legs(part: &mut Part, left: bool) -> (FeatureId, Vec<String>) {
    let pts: Vec<[f64; 2]> = if left {
        vec![[0.0, 0.0], [2.0, 0.0], [2.0, 5.0], [8.0, 5.0], [8.0, 0.0], [10.0, 0.0], [10.0, 7.0], [4.0, 7.0], [4.0, 9.0], [0.0, 9.0]]
    } else {
        vec![[0.0, 0.0], [2.0, 0.0], [2.0, 5.0], [8.0, 5.0], [8.0, 0.0], [10.0, 0.0], [10.0, 9.0], [6.0, 9.0], [6.0, 7.0], [0.0, 7.0]]
    };
    outline(part, &pts)
}

#[test]
fn a_cut_that_leaves_a_body_in_pieces_says_which_kept_the_id_and_where_each_lies() {
    for left in [true, false] {
        let mut part = Part::default();
        let s = legs(&mut part, left);
        let x = extrude(&mut part, s, 0.0, 3.0, BodyOp::New);
        let cut = part.add(FeatureKind::CutPlane(CutPlaneFeature { body: BodyId::new(x), plane: PlaneRef::world(WorldAxis::Y, 5.0), keep: -1, exprs: Exprs::new() }));
        let built = build(&part);
        assert!(built.errors.is_empty(), "{:?}", built.errors);
        let kept = built.body(BodyId::new(x)).unwrap_or_else(|| panic!("no kept body"));
        let bb = kept.body.outline_bounds();
        let mid = (bb.min.x + bb.max.x) / 2.0;
        assert!((mid - if left { 1.0 } else { 9.0 }).abs() < 1e-6, "{mid}");
        let words = built.warnings.get(&cut).map(|w| w.join(" ")).unwrap_or_default();
        let other = built.bodies.iter().find(|b| b.id != BodyId::new(x)).map(|b| split_words::id_words(b.id)).unwrap_or_default();
        let (k, o) = if left { ("0…2", "8…10") } else { ("8…10", "0…2") };
        assert!(words.contains(&format!("The cut at y 5 left Extrude 1 as two bodies: 2 keeps the id (x {k}, y 0…5, z 0…3; 30 mm³); {other} is new (x {o}")), "{words}");
        assert!(words.contains("the two the same size within 1 %, so the id stays with the piece nearest where the body stood"), "{words}");
    }
}

#[test]
fn a_cut_extent_run_on_that_nicks_a_far_corner_says_so() {
    // A plate 40 × 40 × 5, a wall at its far side and an arm from the wall's top back over the
    // plate to x `arm`, z 25…30; a bore Ø6 at (20, 20) cut from z −1 to `to`.
    let bored = |to: f64, arm: f64| -> (Built, FeatureId) {
        let mut part = Part::default();
        let plate = rect(&mut part, 0.0, 0.0, 40.0, 40.0);
        extrude(&mut part, plate, 0.0, 5.0, BodyOp::New);
        let wall = rect(&mut part, 35.0, 0.0, 40.0, 40.0);
        extrude(&mut part, wall, 0.0, 30.0, BodyOp::Join);
        let a = rect(&mut part, arm, 0.0, 40.0, 40.0);
        extrude(&mut part, a, 25.0, 30.0, BodyOp::Join);
        let bore = circle(&mut part, [20.0, 20.0], 3.0);
        let cut = extrude(&mut part, bore, -1.0, to, BodyOp::Cut);
        if let Some(f) = part.feature_mut(cut) {
            f.name = "Bore".into();
        }
        (build(&part), cut)
    };
    let (built, cut) = bored(40.0, 22.9);
    assert!(built.errors.is_empty(), "{:?}", built.errors);
    let words = built.warnings.get(&cut).map(|w| w.join(" ")).unwrap_or_default();
    assert!(words.starts_with("Bore also cut at (22."), "{words}");
    assert!(words.contains("27.5): ") && words.ends_with("22.5 mm along the R3 bore's axis — meant? Shorten the extent."), "{words}");
    // Cut short of the arm, or through an arm it goes all the way through: nothing said.
    let (built, cut) = bored(10.0, 22.9);
    assert!(!built.warnings.contains_key(&cut), "{:?}", built.warnings);
    let (built, cut) = bored(40.0, 10.0);
    assert!(!built.warnings.contains_key(&cut), "{:?}", built.warnings);
}

#[test]
fn a_round_larger_than_its_faces_says_what_size_it_takes() {
    let mut part = Part::default();
    let outline = rect(&mut part, 0.0, 0.0, 40.0, 20.0);
    let p = extrude(&mut part, outline, 0.0, 5.0, BodyOp::New);
    let edge = upright(&build(&part), BodyId::new(p), 0.0, 0.0);
    let r = round(&mut part, BodyId::new(p), edge, 30.0);
    let built = build(&part);
    let said = built.errors.get(&r).cloned().unwrap_or_default();
    assert!(said.starts_with("The edge does not round at 30 mm"), "{said}");
    assert!(said.contains("It rounds at up to") || said.contains("not round even at"), "{said}");
    assert!(built.failed_at.contains_key(&r));
    let _ = DVec3::ZERO;
}
