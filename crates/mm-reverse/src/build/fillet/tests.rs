// SPDX-License-Identifier: AGPL-3.0-only
//! The Fillet's build on the kernel — a round at a varying radius, on a tangent chain, of what
//! the kernel can, and one that takes the whole face between its edges — with the parts the
//! tests of the round, the chamfer, the shell and the face delete share. The cases of
//! ScanRuler's `reBodySteps.test.ts` (a varying round on a tangent chain) and
//! `kernelRounds.ts`, © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.

use glam::{DVec2, DVec3};

use crate::build::{build, default_regions, volume, Built};
use crate::frame::WorldAxis;
use crate::model::*;
use crate::regions::find_regions;
use crate::sketch::{Shape, Sketch};
use crate::topo::{EdgeKind, FaceKind};

/// A sketch on the XY plane at `z` with a rectangle from (x0, y0), and its regions.
pub(crate) fn rect_sketch(part: &mut Part, x0: f64, y0: f64, w: f64, h: f64, z: f64) -> (FeatureId, Vec<String>) {
    let mut s = Sketch::default();
    s.add_rectangle(DVec2::new(x0, y0), DVec2::new(x0 + w, y0 + h));
    let regions = default_regions(&find_regions(&s));
    (part.add(FeatureKind::Sketch(SketchFeature::on_world(WorldAxis::Z, z, s))), regions)
}

pub(crate) fn circle_sketch(part: &mut Part, c: DVec2, r: f64, z: f64) -> (FeatureId, Vec<String>) {
    let mut s = Sketch::default();
    s.add_shape(Shape::Circle { c, r });
    let regions = default_regions(&find_regions(&s));
    (part.add(FeatureKind::Sketch(SketchFeature::on_world(WorldAxis::Z, z, s))), regions)
}

pub(crate) fn extrude(part: &mut Part, sketch: (FeatureId, Vec<String>), from: f64, to: f64, op: BodyOp) -> FeatureId {
    part.add(FeatureKind::Extrude(ExtrudeFeature::new(sketch.0, sketch.1, ExtrudeExtent::Distance { from, to }, op)))
}

/// A block `w` × `h` × `d` from the origin: sketch 1, extrude 2.
pub(crate) fn block(w: f64, h: f64, d: f64) -> Part {
    let mut part = Part::default();
    let s = rect_sketch(&mut part, 0.0, 0.0, w, h, 0.0);
    extrude(&mut part, s, 0.0, d, BodyOp::New);
    part
}

pub(crate) fn vol(built: &Built) -> f64 {
    built.bodies.first().and_then(|b| volume(&b.body)).unwrap_or(f64::NAN)
}

pub(crate) fn ok(built: &Built) {
    assert!(built.errors.is_empty(), "{:?}", built.errors);
}

/// The straight edge whose middle lies at `mid`.
pub(crate) fn edge_at(built: &Built, mid: [f64; 3]) -> EdgeRef {
    built.bodies[0].topology.edges.iter().find(|e| (e.fingerprint.mid - DVec3::from(mid)).length() < 1e-6).map(|e| e.fingerprint.clone()).unwrap_or_else(|| panic!("no edge at {mid:?}"))
}

/// The flat face facing `dir`.
pub(crate) fn plane(built: &Built, dir: [f64; 3]) -> FaceRef {
    built.bodies[0].topology.faces.iter().find(|f| f.fingerprint.kind == FaceKind::Plane && (f.fingerprint.dir - DVec3::from(dir)).length() < 1e-6).map(|f| f.fingerprint).unwrap_or_else(|| panic!("no face facing {dir:?}"))
}

pub(crate) fn fillet(body: FeatureId, edges: Vec<EdgeRef>, radius: f64, chain: bool) -> FilletFeature {
    FilletFeature { body: BodyId::new(body), edges, edges_where: vec![], radius, chain, skip_failing: false, variable: vec![], exprs: Exprs::new() }
}

/// Two long sides joined by semicircular ends: the top edges are one tangent chain.
fn slot(length: f64, r: f64) -> Part {
    let mut part = Part::default();
    let mut s = Sketch::default();
    s.add_shape(Shape::Line { a: DVec2::new(0.0, -r), b: DVec2::new(length, -r) });
    s.add_shape(Shape::Arc { c: DVec2::new(length, 0.0), r, start: -std::f64::consts::FRAC_PI_2, sweep: std::f64::consts::PI });
    s.add_shape(Shape::Line { a: DVec2::new(length, r), b: DVec2::new(0.0, r) });
    s.add_shape(Shape::Arc { c: DVec2::new(0.0, 0.0), r, start: std::f64::consts::FRAC_PI_2, sweep: std::f64::consts::PI });
    let regions = default_regions(&find_regions(&s));
    let sk = part.add(FeatureKind::Sketch(SketchFeature::on_world(WorldAxis::Z, 0.0, s)));
    part.add(FeatureKind::Extrude(ExtrudeFeature::new(sk, regions, ExtrudeExtent::Distance { from: 0.0, to: 10.0 }, BodyOp::New)));
    part
}

#[test]
fn leaves_the_varying_edge_out_of_the_plain_edges_chain() {
    let part = slot(30.0, 5.0);
    let first = build(&part);
    ok(&first);
    let tops: Vec<_> = first.bodies[0].topology.edges.iter().filter(|e| e.fingerprint.kind == EdgeKind::Line && e.fingerprint.mid.z > 9.9).collect();
    let near = tops.iter().find(|e| e.fingerprint.mid.y < 0.0).map(|e| e.fingerprint.clone()).unwrap_or_else(|| panic!("no near side"));
    let far = tops.iter().find(|e| e.fingerprint.mid.y > 0.0).map(|e| e.fingerprint.clone()).unwrap_or_else(|| panic!("no far side"));
    // The near side picked, chained round the rim — through the far side too, which is
    // rounded at a radius of its own.
    let mut rounded = part.clone();
    let f = rounded.add(FeatureKind::Fillet(FilletFeature { variable: vec![VariableEdge { edge: far.clone(), radius: [1.0, 1.0] }], ..fillet(2, vec![near.clone()], 1.0, true) }));
    let out = build(&rounded);
    ok(&out);
    // The whole rim rounded at R1, as one chain of the one radius would.
    let pi = std::f64::consts::PI;
    let want = (30.0 * 10.0 + pi * 25.0) * 10.0 - (1.0 - pi / 4.0) * (60.0 + 2.0 * pi * 5.0);
    assert!((vol(&out) - want).abs() < 0.5, "{} vs {want}", vol(&out));
    let w = out.warnings.get(&f).map(|w| w.join(" ")).unwrap_or_default();
    assert!(w.contains("1 of the edges picked at the one radius lies on the tangent chain of an edge at a varying radius"), "{w}");
    // From R0.8 to R1.5 round the chain: the kernel takes a varying radius along one straight
    // edge between two flats, and says so by name.
    let mut growing = part.clone();
    let g = growing.add(FeatureKind::Fillet(FilletFeature { variable: vec![VariableEdge { edge: far, radius: [0.8, 1.5] }], ..fillet(2, vec![near], 1.0, true) }));
    let out = build(&growing);
    let e = out.errors.get(&g).cloned().unwrap_or_default();
    assert!(e.contains("a radius varying along a tangent chain"), "{e}");
}

#[test]
fn rounds_a_straight_edge_from_its_start_radius_to_its_end_radius() {
    let part = block(40.0, 20.0, 10.0);
    let first = build(&part);
    let edge = edge_at(&first, [20.0, 20.0, 10.0]);
    let start = first.bodies[0].topology.edges.iter().find(|e| e.fingerprint == edge).map(|e| e.ends[0]).unwrap_or_default();
    let mut rounded = part.clone();
    rounded.add(FeatureKind::Fillet(FilletFeature { edges: vec![], variable: vec![VariableEdge { edge, radius: [1.0, 3.0] }], ..fillet(2, vec![], 1.0, false) }));
    let out = build(&rounded);
    ok(&out);
    // The section's lost corner, (1 − π/4) r², with r running evenly from 1 to 3 along 40.
    let want = 8000.0 - (1.0 - std::f64::consts::FRAC_PI_4) * 40.0 * (1.0 + 3.0 + 9.0) / 3.0;
    assert!((vol(&out) - want).abs() < 0.2, "{} vs {want}", vol(&out));
    // The top keeps more of itself at the start, where the round is the smaller.
    let top = plane(&out, [0.0, 0.0, 1.0]);
    assert!((top.area - (800.0 - 40.0 * 2.0)).abs() < 1e-3, "{}", top.area);
    let towards_start = (top.centroid.x - 20.0) * (start.x - 20.0);
    assert!(towards_start > 0.0, "the top's middle at {} and the start at {}", top.centroid.x, start.x);
}

#[test]
fn rounds_what_holds_and_names_the_chains_left_out() {
    let part = block(40.0, 20.0, 10.0);
    let first = build(&part);
    // Each alone fits the back's 10 mm; both together overlap on it.
    let top = edge_at(&first, [20.0, 20.0, 10.0]);
    let bottom = edge_at(&first, [20.0, 20.0, 0.0]);
    let mut both = part.clone();
    let f = both.add(FeatureKind::Fillet(FilletFeature { skip_failing: true, ..fillet(2, vec![top.clone(), bottom.clone()], 9.0, false) }));
    let out = build(&both);
    ok(&out);
    let w = out.warnings.get(&f).map(|w| w.join(" ")).unwrap_or_default();
    assert!(w.starts_with("1 of the 2 chains left out — the kernel cannot round it with the rest: at (20.000, 20.000, 0.000)"), "{w}");
    let want = 8000.0 - (1.0 - std::f64::consts::FRAC_PI_4) * 81.0 * 40.0;
    assert!((vol(&out) - want).abs() < 0.5, "{} vs {want}", vol(&out));
    // None holds: said, and nothing rounded.
    let mut none = part.clone();
    let g = none.add(FeatureKind::Fillet(FilletFeature { skip_failing: true, ..fillet(2, vec![top, bottom], 25.0, false) }));
    let out = build(&none);
    assert_eq!(out.errors.get(&g).map(String::as_str), Some("None of the 2 chains rounds at 25 mm."));
    // Without it, the round fails whole.
    let mut strict = part.clone();
    let h = strict.add(FeatureKind::Fillet(fillet(2, vec![edge_at(&first, [20.0, 20.0, 10.0]), edge_at(&first, [20.0, 20.0, 0.0])], 9.0, false)));
    assert!(build(&strict).errors.get(&h).is_some_and(|e| e.starts_with("Filleting: ")));
}

#[test]
fn rounds_a_wall_whose_two_rounds_take_its_whole_top_into_a_half_round() {
    // A wall 5 thick, its two top edges rounded 2.5: one half round, no sliver between.
    let part = block(40.0, 5.0, 10.0);
    let first = build(&part);
    let a = edge_at(&first, [20.0, 0.0, 10.0]);
    let b = edge_at(&first, [20.0, 5.0, 10.0]);
    let mut rounded = part.clone();
    rounded.add(FeatureKind::Fillet(fillet(2, vec![a, b], 2.5, false)));
    let out = build(&rounded);
    ok(&out);
    let want = 2000.0 - 2.0 * (1.0 - std::f64::consts::FRAC_PI_4) * 6.25 * 40.0;
    assert!((vol(&out) - want).abs() < 0.2, "{} vs {want}", vol(&out));
    let thin = out.bodies[0].topology.faces.iter().filter(|f| f.fingerprint.area < 1e-3).count();
    assert_eq!(thin, 0, "a sliver face is left");
}

#[test]
fn a_varying_round_with_a_radius_of_none_fails_by_name() {
    let part = block(40.0, 20.0, 10.0);
    let edge = edge_at(&build(&part), [20.0, 20.0, 10.0]);
    let mut p = part.clone();
    let f = p.add(FeatureKind::Fillet(FilletFeature { variable: vec![VariableEdge { edge, radius: [0.0, 2.0] }], ..fillet(2, vec![], 1.0, false) }));
    assert_eq!(build(&p).errors.get(&f).map(String::as_str), Some("An edge rounded at a varying radius needs both its radii."));
}

#[test]
fn a_round_refused_at_the_whole_faces_width_is_made_a_hair_under_and_its_sliver_taken_off() {
    // The kernel's refusal at the exact size, as Open CASCADE refuses a half round: the round
    // is made a hair under it and the sliver of top it leaves closed over.
    let part = block(40.0, 5.0, 10.0);
    let built = build(&part);
    let body = built.bodies[0].body.clone();
    let t = &built.bodies[0].topology;
    let picks: Vec<usize> = t.edges.iter().enumerate().filter(|(_, e)| e.fingerprint.kind == EdgeKind::Line && (e.fingerprint.mid.z - 10.0).abs() < 1e-6 && e.fingerprint.length > 30.0).map(|(i, _)| i).collect();
    assert_eq!(picks.len(), 2);
    let points = super::edge_points(t, &picks);
    let origins = vec![2i64; body.face_count()];
    let made = super::full_width(&body, &origins, 3, 2.5, |r| {
        if (r - 2.5).abs() < 1e-12 {
            Err(solidcraft_kernel::KernelError::Failed("the face between would close up".into()))
        } else {
            solidcraft_kernel::fillet(&body, &points, r)
        }
    })
    .unwrap_or_else(|e| panic!("{e}"));
    assert!(super::closed_solid(&made.shape));
    let v = volume(&made.shape).unwrap_or(f64::NAN);
    let want = 2000.0 - 2.0 * (1.0 - std::f64::consts::FRAC_PI_4) * 6.25 * 40.0;
    assert!((v - want).abs() < 0.2, "{v} vs {want}");
    // No face as thin as the slack left: the sliver went.
    assert!(super::slivers(&made.shape, 8.0 * 2.5 * 1e-5).is_empty());
    // A refusal at every size stands as the kernel's.
    let none = super::full_width(&body, &origins, 3, 2.5, |_| Err(solidcraft_kernel::KernelError::Failed("no".into())));
    assert_eq!(none.err().as_deref(), Some("the operation failed: no"));
}
