// SPDX-License-Identifier: AGPL-3.0-only
//! A fit of a round's size from its edges' readings, a variable fillet's law along one edge, a
//! round read off a scan of a rounded block, and rounds placed solid by solid — after
//! ScanRuler's `reFit.ts`, `reFilletLaw.test.ts` and `reFeatureCommands.ts` (addOnSolids), ©
//! Stefan Hermann (CNC Kitchen), AGPL-3.0-only.

use glam::{DVec2, DVec3};

use super::*;
use crate::fillet_read::{read_edge_of, ScanSection};
use crate::pick_query::Bounds;

/// A block `w` × `h` × `d` from the origin: sketch 1, extrude 2.
fn block(w: f64, h: f64, d: f64) -> Part {
    let mut part = Part::default();
    let mut s = crate::sketch::Sketch::default();
    s.add_rectangle(DVec2::new(0.0, 0.0), DVec2::new(w, h));
    let regions = crate::build::default_regions(&crate::regions::find_regions(&s));
    let sk = part.add(FeatureKind::Sketch(crate::model::SketchFeature::on_world(crate::frame::WorldAxis::Z, 0.0, s)));
    part.add(FeatureKind::Extrude(crate::model::ExtrudeFeature::new(sk, regions, crate::model::ExtrudeExtent::Distance { from: 0.0, to: d }, crate::model::BodyOp::New)));
    part
}

fn edge_at(built: &crate::build::Built, mid: [f64; 3]) -> crate::model::EdgeRef {
    built.bodies[0].topology.edges.iter().find(|e| (e.fingerprint.mid - DVec3::from(mid)).length() < 1e-6).map(|e| e.fingerprint.clone()).unwrap_or_else(|| panic!("no edge at {mid:?}"))
}

fn fillet(body: FeatureId, edges: Vec<crate::model::EdgeRef>, radius: f64, chain: bool) -> crate::model::FilletFeature {
    crate::model::FilletFeature { body: BodyId::new(body), edges, edges_where: vec![], radius, chain, skip_failing: false, variable: vec![], exprs: crate::model::Exprs::new() }
}

fn reading(v: f64, s: f64) -> Reading {
    Reading { value: Some(v), sigma: Some(s), why: None }
}

#[test]
fn takes_the_median_of_the_edges_read_with_its_nominal_and_refuses_a_spread() {
    let fit = combined("fillet’s radius", "mm", &[reading(1.98, 0.02), reading(2.01, 0.02), reading(2.03, 0.03)], NominalKind::Radius).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(fit.value, 2.01);
    assert_eq!(fit.nominal, 2.0);
    assert_eq!(fit.off, 0.01);
    assert!(fit.sigma.is_some_and(|s| s > 0.0 && s < 0.02));
    let spread = combined("fillet’s radius", "mm", &[reading(2.0, 0.01), reading(3.0, 0.01)], NominalKind::Radius);
    assert!(spread.is_err_and(|e| e.starts_with("The fillet’s radius cannot be one number: the readings spread past two sigmas — #1 2 ± 0.01 mm; #2 3 ± 0.01 mm.")));
    let unread = combined("chamfer’s distance", "mm", &[reading(1.0, 0.01), Reading { value: None, sigma: None, why: Some("no round within 3 mm".into()) }], NominalKind::Length);
    assert!(unread.is_err_and(|e| e == "The chamfer’s distance cannot be fitted: not every one picked reads — #1 1 ± 0.01 mm; #2 no round within 3 mm. Pick only those that do, or give a number."));
}

#[test]
fn reads_one_radius_where_the_places_agree_a_line_where_they_lie_on_one_and_neither_for_a_v() {
    let at = |t: f64, r: f64| EdgeStation { t, at: DVec3::ZERO, radius: Some(r), sigma: Some(0.01) };
    let flat: Vec<EdgeStation> = (0..7).map(|i| at(i as f64 / 6.0, 2.0)).collect();
    assert_eq!(law_of(&flat, 2.0, None), (LawKind::One, [2.0, 2.0]));
    let line: Vec<EdgeStation> = (0..7).map(|i| at(i as f64 / 6.0, 1.0 + 2.0 * i as f64 / 6.0)).collect();
    let law = crate::fillet_read::linear_law(&line);
    assert_eq!(law_of(&line, 2.0, law.as_ref()).0, LawKind::Line);
    assert_eq!(law_of(&line, 2.0, law.as_ref()).1, [1.0, 3.0]);
    let v: Vec<EdgeStation> = (0..7).map(|i| at(i as f64 / 6.0, 1.0 + (i as f64 - 3.0).abs())).collect();
    let law = crate::fillet_read::linear_law(&v);
    assert_eq!(law_of(&v, 2.0, law.as_ref()).0, LawKind::Neither);
}

#[test]
fn fits_a_fillets_radius_off_a_scan_of_a_block_rounded_r3() {
    // The scan: the block rounded R3 on its back top edge, as the kernel meshes it.
    let mut rounded = block(40.0, 20.0, 10.0);
    let first = crate::build::build(&rounded);
    let edge = edge_at(&first, [20.0, 20.0, 10.0]);
    rounded.add(FeatureKind::Fillet(fillet(2, vec![edge], 3.0, false)));
    let scan = crate::build::build(&rounded);
    let mesh = crate::build::body_mesh(&scan.bodies[0].body).unwrap_or_default();
    let (positions, triangles) = crate::build::welded(&mesh.positions, &mesh.triangles, 1e-7);
    let section = ScanSection::new(positions, triangles);
    let size = section.size();
    let mut cut = |o: DVec3, n: DVec3, m: f64| section.cut(o, n, m);
    // The edge read on the sharp block, as a fit reads it.
    let t = &first.bodies[0].topology;
    let e = t.edges.iter().find(|e| (e.fingerprint.mid - DVec3::new(20.0, 20.0, 10.0)).length() < 1e-6).unwrap_or_else(|| panic!("no edge"));
    let fit = fit_edges(&mut cut, &[read_edge_of(e, t, None)], ReadShape::Round, None, size).unwrap_or_else(|e| panic!("{e}"));
    assert!((fit.value - 3.0).abs() < 0.05, "{fit:?}");
    assert_eq!(fit.nominal, 3.0);
    let law = fit_edge_law(&mut cut, &read_edge_of(e, t, None), 0, None, size).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(law.law.map(|l| (l[0] - l[1]).abs() < 1e-9), Some(true), "{law:?}");
}

#[test]
fn rounds_each_solid_a_feature_made_and_names_them_in_turn() {
    // One extrude, two blocks apart: two solids of feature 3.
    let mut part = Part::default();
    let mut s = crate::sketch::Sketch::default();
    s.add_rectangle(DVec2::new(0.0, 0.0), DVec2::new(10.0, 10.0));
    s.add_rectangle(DVec2::new(20.0, 0.0), DVec2::new(30.0, 10.0));
    let regions = crate::build::default_regions(&crate::regions::find_regions(&s));
    let sk = part.add(FeatureKind::Sketch(crate::model::SketchFeature::on_world(crate::frame::WorldAxis::Z, 0.0, s)));
    let ex = part.add(FeatureKind::Extrude(crate::model::ExtrudeFeature::new(sk, regions, crate::model::ExtrudeExtent::Distance { from: 0.0, to: 10.0 }, crate::model::BodyOp::New)));
    let built = crate::build::build(&part);
    assert_eq!(solids_of(&built, ex, None).len(), 2);
    // Every edge along the top.
    let top = Where { bounds: Some(Bounds { min: DVec3::new(-100.0, -100.0, 9.9), max: DVec3::new(100.0, 100.0, 10.1) }), ..Where::default() };
    let round = FeatureKind::Fillet(fillet(ex, vec![], 1.0, false));
    let on = rounds_on_solids(&part, ex, &round, &[top.clone()], None).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(on.made.len(), 2);
    assert!(on.made.iter().all(|m| m.2 == 4));
    let names: Vec<&str> = on.made.iter().filter_map(|m| on.part.feature(m.0)).map(|f| f.name.as_str()).collect();
    let of = part.feature(ex).map(|f| f.name.clone()).unwrap_or_default();
    assert_eq!(names, [format!("Fillet on {of} 1"), format!("Fillet on {of} 2")]);
    assert!(on.left_out.is_empty());
    let after = crate::build::build(&on.part);
    assert!(after.errors.is_empty(), "{:?}", after.errors);
    // A round that does not build on them: none is added.
    let big = FeatureKind::Fillet(fillet(ex, vec![], 8.0, false));
    let refused = rounds_on_solids(&part, ex, &big, &[top], Some("Big"));
    let e = refused.err().unwrap_or_default();
    assert!(e.starts_with(&format!("2 of the 2 fillets on {of} did not build, and none was added — Big 1 on ")), "{e}");
}

#[test]
fn refuses_a_name_a_feature_has_in_any_case_and_offers_a_free_one() {
    // Two blocks of one extrude, a feature already called "big 1": the rounds would take its
    // name, and none is added (ScanRuler's `requireFreeName`, `freeName`).
    let mut part = Part::default();
    let mut s = crate::sketch::Sketch::default();
    s.add_rectangle(DVec2::new(0.0, 0.0), DVec2::new(10.0, 10.0));
    s.add_rectangle(DVec2::new(20.0, 0.0), DVec2::new(30.0, 10.0));
    let regions = crate::build::default_regions(&crate::regions::find_regions(&s));
    let sk = part.add(FeatureKind::Sketch(crate::model::SketchFeature::on_world(crate::frame::WorldAxis::Z, 0.0, s)));
    let ex = part.add(FeatureKind::Extrude(crate::model::ExtrudeFeature::new(sk, regions, crate::model::ExtrudeExtent::Distance { from: 0.0, to: 10.0 }, crate::model::BodyOp::New)));
    let plane = crate::sketch::Sketch::default();
    let named = part.add_named(FeatureKind::Sketch(crate::model::SketchFeature::on_world(crate::frame::WorldAxis::Z, 20.0, plane)), "big 1".into());
    let top = Where { bounds: Some(Bounds { min: DVec3::new(-100.0, -100.0, 9.9), max: DVec3::new(100.0, 100.0, 10.1) }), ..Where::default() };
    let round = FeatureKind::Fillet(fillet(ex, vec![], 1.0, false));
    let e = rounds_on_solids(&part, ex, &round, &[top], Some("Big")).err().unwrap_or_default();
    assert_eq!(e, format!("Feature {named} is already named “big 1” — the commands find features by name, so each needs its own: “Big 2”, say."));
    // The body a feature worked on is one of its solids too.
    let built = crate::build::build(&part);
    assert_eq!(solids_of(&built, sk, Some(BodyId::new(ex))), vec![BodyId::new(ex)]);
}
