// SPDX-License-Identifier: AGPL-3.0-only
//! The Fillet box's picks between its modes — the case of ScanRuler's `reChamferDelete.test.ts`
//! ("the boxes at the store": edges clicked at a varying radius), © Stefan Hermann (CNC
//! Kitchen), AGPL-3.0-only.

use glam::DVec3;
use mm_reverse::model::{BodyId, EdgeRef, Exprs, FilletFeature};
use mm_reverse::topo::{EdgeFingerprint, EdgeKind, MatchTolerance};

use super::super::rnd::{forget, with_draft};
use super::{opens_variable, switch_mode, toggle_print};

fn edge(at: f64) -> EdgeRef {
    EdgeFingerprint { kind: EdgeKind::Line, mid: DVec3::new(at, 0.0, 10.0), dir: DVec3::X, length: 10.0, radius: None, faces: vec![], origin: 2 }
}

const TOL: MatchTolerance = MatchTolerance { position: 1e-4, angle: 1e-3, size: 1e-3 };

fn at(x: &FilletFeature) -> Vec<(f64, [f64; 2])> {
    x.variable.iter().map(|v| (v.edge.mid.x, v.radius)).collect()
}

#[test]
fn takes_edges_clicked_at_a_varying_radius_each_starting_as_the_last_and_carries_the_picks_between_the_modes() {
    forget();
    let body = BodyId::new(2);
    let mut x = FilletFeature { body, edges: vec![edge(11.0)], edges_where: vec![], radius: 2.0, chain: true, skip_failing: false, variable: vec![], exprs: Exprs::new() };
    // To Variable: the edge picked goes from the one radius at both ends.
    switch_mode(&mut x, true);
    assert!(with_draft(|d| d.variable));
    assert!(x.edges.is_empty());
    assert_eq!(at(&x), vec![(11.0, [2.0, 2.0])]);
    x.variable[0].radius = [1.0, 3.0];
    toggle_print(&mut x, body, edge(12.0), &TOL);
    assert_eq!(at(&x), vec![(11.0, [1.0, 3.0]), (12.0, [1.0, 3.0])]);
    // A picked one clicked again is let go.
    toggle_print(&mut x, body, edge(11.0), &TOL);
    assert_eq!(at(&x), vec![(12.0, [1.0, 3.0])]);
    assert!(x.edges.is_empty());
    // Back to Constant: the edges at the first one's start radius.
    switch_mode(&mut x, false);
    assert!(!with_draft(|d| d.variable));
    assert_eq!(x.edges.iter().map(|e| e.mid.x).collect::<Vec<_>>(), vec![12.0]);
    assert!(x.variable.is_empty());
    assert_eq!(x.radius, 1.0);
    assert!(with_draft(|d| d.radius_typed));
    // Committed in Variable and reopened on its varying edges alone: Variable.
    switch_mode(&mut x, true);
    assert!(opens_variable(&x));
    // Constant edges kept beside varying ones: it opens in Constant.
    x.edges.push(edge(13.0));
    assert!(!opens_variable(&x));
    forget();
}

#[test]
fn a_click_on_an_edge_of_another_body_starts_the_picks_again_there() {
    forget();
    let mut x = FilletFeature { body: BodyId::new(2), edges: vec![edge(1.0)], edges_where: vec![], radius: 1.0, chain: true, skip_failing: false, variable: vec![], exprs: Exprs::new() };
    toggle_print(&mut x, BodyId::new(5), edge(2.0), &TOL);
    assert_eq!(x.body, BodyId::new(5));
    assert_eq!(x.edges.iter().map(|e| e.mid.x).collect::<Vec<_>>(), vec![2.0]);
    forget();
}
