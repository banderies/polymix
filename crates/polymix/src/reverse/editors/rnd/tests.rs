// SPDX-License-Identifier: AGPL-3.0-only
//! What the Fillet, Chamfer, Shell and Delete face boxes share: a window over the bodies told
//! from a click and the body it goes into, what a box remembers let go when a box opens, and
//! the edges a fit reads, lost ones by their fingerprints — as ScanRuler's `reScene.ts`
//! (beginBodyWindow), `useReversePlugin.tsx` (onBodyWindowPick) and `reFit.ts` (edgesOf) do
//! them, © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.

use egui::pos2;
use glam::DVec3;
use mm_reverse::curved_edge::FacedMesh;
use mm_reverse::model::{BodyId, EdgeRef};
use mm_reverse::topo::{EdgeFingerprint, EdgeInfo, EdgeKind, Topology};

use super::{edge_reads, forget, most_on, window_moved, with_draft};

#[test]
fn tells_a_window_from_a_click_by_how_far_the_pointer_went_across_and_down_together() {
    // 4 across and 3 down: 5 px as the crow flies, 7 as the window counts — a window.
    assert!(window_moved(pos2(10.0, 10.0), pos2(14.0, 13.0)));
    assert!(window_moved(pos2(10.0, 10.0), pos2(3.0, 10.0)));
    // 6 px all told is still a click.
    assert!(!window_moved(pos2(10.0, 10.0), pos2(13.0, 13.0)));
    assert!(!window_moved(pos2(10.0, 10.0), pos2(10.0, 10.0)));
}

#[test]
fn takes_the_body_most_of_a_window_lies_on_the_first_met_where_two_tie() {
    let (a, b) = (BodyId::new(2), BodyId::new(5));
    assert_eq!(most_on(&[(a, 0), (b, 1), (b, 2), (a, 3)]), Some(a));
    assert_eq!(most_on(&[(b, 0), (a, 1), (a, 2), (b, 3)]), Some(b));
    assert_eq!(most_on(&[(b, 0), (a, 1), (a, 2)]), Some(a));
    assert_eq!(most_on(&[]), None);
}

#[test]
fn a_box_opened_starts_with_nothing_remembered_of_the_last() {
    forget();
    with_draft(|d| {
        d.variable = true;
        d.radius_typed = true;
        d.flip = true;
        d.distance2 = 3.0;
        d.tangent = false;
    });
    // What `ReverseWorkspace::open_feature` does for every box it opens.
    super::super::rnd_forget();
    with_draft(|d| {
        assert!(!d.variable && !d.radius_typed && !d.flip && d.tangent);
        assert_eq!((d.distance2, d.angle_deg), (1.0, 45.0));
    });
}

fn line(x: f64) -> EdgeRef {
    EdgeFingerprint { kind: EdgeKind::Line, mid: DVec3::new(x, 0.0, 10.0), dir: DVec3::X, length: 10.0, radius: None, faces: vec![], origin: 2 }
}

#[test]
fn reads_every_edge_picked_in_its_place_a_lost_one_off_its_fingerprint() {
    let edge = |hash: u32, x: f64| EdgeInfo { hash, fingerprint: line(x), faces: vec![], ends: [DVec3::new(x - 5.0, 0.0, 10.0), DVec3::new(x + 5.0, 0.0, 10.0)], end_tangents: [DVec3::X, DVec3::X], polyline: vec![DVec3::new(x - 5.0, 0.0, 10.0), DVec3::new(x + 5.0, 0.0, 10.0)] };
    let t = Topology { faces: vec![], edges: vec![edge(0, 5.0), edge(1, 25.0)] };
    let mesh = FacedMesh { face_hash: Some(Vec::new()), ..FacedMesh::default() };
    let picks = vec![line(5.0), line(15.0), line(25.0)];
    let found = |e: &EdgeRef| t.edges.iter().position(|x| x.fingerprint.mid == e.mid);
    let reads = edge_reads(&t, &mesh, &picks, found);
    // One a pick, in order: a varying edge's law lands on the edge it was read along.
    assert_eq!(reads.iter().map(|r| r.fingerprint().mid.x).collect::<Vec<_>>(), vec![5.0, 15.0, 25.0]);
    // Found, with the body's mesh to read an edge of any shape off; lost, the fingerprint alone.
    assert!(reads[0].mesh.is_some() && reads[2].mesh.is_some());
    assert!(reads[1].mesh.is_none() && reads[1].edge.ends.is_none());
}

#[test]
fn a_reading_that_came_in_while_no_box_was_drawn_is_not_waited_on() {
    forget();
    let (tx, rx) = std::sync::mpsc::channel();
    with_draft(|d| d.estimate_job = Some(rx));
    assert!(super::busy());
    // The worker done after the box was put away: taken in, kept for the box, and no longer
    // waited on — a script's idle step goes on.
    tx.send(Some((2.5, 40))).ok();
    assert!(!super::busy());
    assert_eq!(with_draft(|d| d.estimate_came), Some(Some((2.5, 40))));
    // A worker gone with nothing.
    let (tx, rx) = std::sync::mpsc::channel::<Option<(f64, usize)>>();
    with_draft(|d| d.estimate_job = Some(rx));
    drop(tx);
    assert!(!super::busy());
    forget();
}
