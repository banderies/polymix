// SPDX-License-Identifier: AGPL-3.0-only
//! A chamfer by two distances or by a distance and an angle, its reference face the one the
//! edges share or, flipped, the other — the cases of ScanRuler's `reChamferDelete.test.ts`
//! ("a chamfer by two distances or by a distance and an angle"), © Stefan Hermann (CNC
//! Kitchen), AGPL-3.0-only.

use crate::build::build;
use crate::build::fillet::tests::{block, edge_at, ok, plane, vol};
use crate::model::*;

/// A block 10 on a side, from the origin.
fn cube() -> Part {
    block(10.0, 10.0, 10.0)
}

fn chamfer(edges: Vec<EdgeRef>, mode: ChamferMode, distance2: Option<f64>, angle_deg: Option<f64>, flip: bool, chain: bool) -> FeatureKind {
    FeatureKind::Chamfer(ChamferFeature { body: BodyId::new(2), edges, edges_where: vec![], distance: 1.0, chain, skip_failing: false, mode: Some(mode), distance2, angle_deg, flip, exprs: Exprs::new() })
}

fn top_edges(part: &Part) -> Vec<EdgeRef> {
    let built = build(part);
    built.bodies[0].topology.edges.iter().filter(|e| e.fingerprint.kind == crate::topo::EdgeKind::Line && (e.fingerprint.mid.z - 10.0).abs() < 1e-6).map(|e| e.fingerprint.clone()).collect()
}

#[test]
fn cuts_one_face_back_by_the_first_distance_and_the_other_by_the_second_and_swaps_them_flipped() {
    let plain = build(&cube());
    let edge = edge_at(&plain, [5.0, 0.0, 10.0]);
    let mut once = cube();
    once.add(chamfer(vec![edge.clone()], ChamferMode::TwoDistances, Some(2.0), None, false, true));
    let b = build(&once);
    ok(&b);
    assert!((vol(&b) - (1000.0 - 0.5 * 1.0 * 2.0 * 10.0)).abs() < 1e-3, "{}", vol(&b));
    let areas = [plane(&b, [0.0, 0.0, 1.0]).area, plane(&b, [0.0, -1.0, 0.0]).area];
    let mut sorted = areas;
    sorted.sort_by(f64::total_cmp);
    assert!((sorted[0] - 80.0).abs() < 1e-3 && (sorted[1] - 90.0).abs() < 1e-3, "{areas:?}");
    let mut flipped = cube();
    flipped.add(chamfer(vec![edge], ChamferMode::TwoDistances, Some(2.0), None, true, true));
    let f = build(&flipped);
    ok(&f);
    assert!((plane(&f, [0.0, 0.0, 1.0]).area - areas[1]).abs() < 1e-3);
    assert!((plane(&f, [0.0, -1.0, 0.0]).area - areas[0]).abs() < 1e-3);
}

#[test]
fn chamfers_a_rim_from_the_top_it_shares_its_corners_met_as_one_chamfer() {
    let edges = top_edges(&cube());
    assert_eq!(edges.len(), 4);
    let mut part = cube();
    part.add(chamfer(edges, ChamferMode::TwoDistances, Some(2.0), None, false, false));
    let b = build(&part);
    ok(&b);
    // 1 in from every side on the top, 2 down every side.
    assert!((plane(&b, [0.0, 0.0, 1.0]).area - 64.0).abs() < 1e-3);
    for side in [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]] {
        assert!((plane(&b, side).area - 80.0).abs() < 1e-3, "{side:?}: {}", plane(&b, side).area);
    }
    // The four strips less the corners counted twice: ∫ (40w − 4w²) 2 dw.
    assert!((vol(&b) - (1000.0 - 2.0 * (20.0 - 4.0 / 3.0))).abs() < 1e-3, "{}", vol(&b));
}

#[test]
fn stands_at_its_angle_to_the_reference_face_the_top_or_each_side_flipped() {
    let edges = top_edges(&cube());
    let t = (std::f64::consts::PI / 6.0).tan();
    let mut on_top = cube();
    on_top.add(chamfer(edges.clone(), ChamferMode::DistanceAngle, None, Some(30.0), false, false));
    let b = build(&on_top);
    ok(&b);
    assert!((plane(&b, [0.0, 0.0, 1.0]).area - 64.0).abs() < 1e-3);
    assert!((plane(&b, [1.0, 0.0, 0.0]).area - 10.0 * (10.0 - t)).abs() < 1e-3);
    let mut on_sides = cube();
    on_sides.add(chamfer(edges, ChamferMode::DistanceAngle, None, Some(30.0), true, false));
    let b = build(&on_sides);
    ok(&b);
    assert!((plane(&b, [0.0, 0.0, 1.0]).area - (10.0 - 2.0 * t).powi(2)).abs() < 1e-3);
    assert!((plane(&b, [1.0, 0.0, 0.0]).area - 90.0).abs() < 1e-3);
}

#[test]
fn refuses_an_angle_of_90_or_more_and_a_second_distance_of_none() {
    let edge = edge_at(&build(&cube()), [5.0, 0.0, 10.0]);
    let mut steep = cube();
    let s = steep.add(chamfer(vec![edge.clone()], ChamferMode::DistanceAngle, None, Some(90.0), false, true));
    assert!(build(&steep).errors.get(&s).is_some_and(|e| e.contains("between 0° and 90°")));
    let mut none = cube();
    let n = none.add(chamfer(vec![edge], ChamferMode::TwoDistances, None, None, false, true));
    assert!(build(&none).errors.get(&n).is_some_and(|e| e.contains("second distance")));
}

#[test]
fn bevels_what_holds_and_names_the_chains_left_out() {
    let part = block(40.0, 20.0, 10.0);
    let first = build(&part);
    let top = edge_at(&first, [20.0, 20.0, 10.0]);
    let bottom = edge_at(&first, [20.0, 20.0, 0.0]);
    let mut both = part.clone();
    let f = both.add(FeatureKind::Chamfer(ChamferFeature { body: BodyId::new(2), edges: vec![top, bottom], edges_where: vec![], distance: 6.0, chain: false, skip_failing: true, mode: None, distance2: None, angle_deg: None, flip: false, exprs: Exprs::new() }));
    let out = build(&both);
    ok(&out);
    let w = out.warnings.get(&f).map(|w| w.join(" ")).unwrap_or_default();
    assert!(w.starts_with("1 of the 2 chains left out — the kernel cannot bevel it with the rest"), "{w}");
    assert!((vol(&out) - (8000.0 - 18.0 * 40.0)).abs() < 1e-2, "{}", vol(&out));
}

#[test]
fn refuses_by_name_an_edge_whose_faces_run_on_into_each_other() {
    // A round on the cube's front top edge: where it runs into the top there is no corner, and
    // a chamfer by two distances or an angle there says so (ScanRuler's `angledAsk`).
    let plain = build(&cube());
    let mut rounded = cube();
    rounded.add(FeatureKind::Fillet(crate::build::fillet::tests::fillet(2, vec![edge_at(&plain, [5.0, 0.0, 10.0])], 2.0, false)));
    let r = build(&rounded);
    ok(&r);
    let seam = r.bodies[0]
        .topology
        .edges
        .iter()
        .find(|e| e.fingerprint.kind == crate::topo::EdgeKind::Line && (e.fingerprint.mid - glam::DVec3::new(5.0, 2.0, 10.0)).length() < 1e-6)
        .map(|e| e.fingerprint.clone())
        .unwrap_or_else(|| panic!("no edge where the round meets the top"));
    for (mode, d2, angle) in [(ChamferMode::TwoDistances, Some(2.0), None), (ChamferMode::DistanceAngle, None, Some(30.0))] {
        let mut part = rounded.clone();
        let c = part.add(FeatureKind::Chamfer(ChamferFeature { body: BodyId::new(2), edges: vec![seam.clone()], edges_where: vec![], distance: 1.0, chain: false, skip_failing: false, mode: Some(mode), distance2: d2, angle_deg: angle, flip: false, exprs: Exprs::new() }));
        let e = build(&part).errors.get(&c).cloned().unwrap_or_default();
        assert!(e.starts_with("At line") && e.ends_with(": the faces at the edge run on into each other — there is no corner to chamfer."), "{mode:?}: {e}");
    }
}
