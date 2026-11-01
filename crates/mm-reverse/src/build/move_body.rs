// SPDX-License-Identifier: AGPL-3.0-only
//! The Move feature's build: bodies turned about a pivot — about X, then Y, then Z — and
//! carried, or copies of them moved and the bodies left as they were. Ported from ScanRuler
//! (`plugins/reverse/core/move.ts`, `resolve.ts` the moveBody step, `build.ts` the moveBody
//! step), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! Every body is moved before any is put back: a move the kernel refuses halfway leaves the
//! bodies as they were. A moved body's faces keep their makers (the kernel's history of the
//! placement carries them over), so later picks still find them; a copy is a body of the
//! move's own, every face of it the move's.

use solidcraft_kernel as kernel;

use super::{kerr, move_matrix, origins_after, BodyState, Builder};
use crate::model::{Feature, MoveBodyFeature, Part};

pub(crate) fn build(b: &mut Builder, _part: &Part, _f: &Feature, x: &MoveBodyFeature) -> Result<(), String> {
    if x.bodies.is_empty() {
        return Err("No body is picked to move.".into());
    }
    if !(x.pivot.is_finite() && x.translate.is_finite() && x.rotate.is_finite()) {
        return Err("The move needs a number for every distance and angle.".into());
    }
    let m = move_matrix(x.pivot, x.translate, x.rotate);
    // Each body once, by the id it stands under now.
    let mut ids = Vec::new();
    for body in &x.bodies {
        let id = b.resolve(*body);
        if !ids.contains(&id) {
            ids.push(id);
        }
    }
    let mut moved = Vec::new();
    for id in ids {
        let (id, state) = b.body_for(id).map_err(|_| "A body this move takes is gone, or failed to build.".to_string())?;
        let shape = kernel::transform_matrix(&state.shape, m).map_err(kerr)?;
        moved.push((id, state, shape));
    }
    let mut made = Vec::new();
    for (id, state, shape) in moved {
        if x.copy {
            let at = b.fresh_id()?;
            let own = b.own_origins(&shape);
            let settled = b.settle(at, shape, own, false)?;
            for c in &settled {
                if let Some(s) = b.body_state_mut(*c) {
                    s.surface = state.surface;
                    s.label = state.label.clone();
                }
            }
            made.extend(settled);
        } else {
            let origins = origins_after(&shape, &[(&state.shape, &state.origins)], b.step);
            let mut s = BodyState::new(shape, origins);
            s.surface = state.surface;
            s.label = state.label.clone();
            b.set_body(id, s);
            made.push(id);
        }
    }
    b.touched(made);
    Ok(())
}

#[cfg(test)]
mod tests {
    //! The parts of ScanRuler's `plugins/reverse/tests/reImportMove.test.ts` ("a Move") its build
    //! answers, on the SolidCraft kernel.
    use glam::{DVec2, DVec3};

    use crate::build::{build, volume};
    use crate::frame::WorldAxis;
    use crate::model::*;
    use crate::regions::find_regions;
    use crate::sketch::Sketch;

    /// A 40 × 20 × 10 block: sketch 1, extrude 2.
    fn block() -> Part {
        let mut part = Part::default();
        let mut s = Sketch::default();
        s.add_rectangle(DVec2::ZERO, DVec2::new(40.0, 20.0));
        let regions = crate::build::default_regions(&find_regions(&s));
        let sk = part.add(FeatureKind::Sketch(SketchFeature::on_world(WorldAxis::Z, 0.0, s)));
        part.add(FeatureKind::Extrude(ExtrudeFeature::new(sk, regions, ExtrudeExtent::Distance { from: 0.0, to: 10.0 }, BodyOp::New)));
        part
    }

    fn mover(bodies: Vec<BodyId>, translate: DVec3, rotate: DVec3, copy: bool) -> FeatureKind {
        FeatureKind::MoveBody(MoveBodyFeature { bodies, pivot: DVec3::new(20.0, 10.0, 5.0), translate, rotate, copy, exprs: Exprs::new() })
    }

    #[test]
    fn turns_and_carries_the_body_its_faces_keeping_their_maker() {
        let mut part = block();
        part.add(mover(vec![BodyId::new(2)], DVec3::new(0.0, 0.0, 30.0), DVec3::new(0.0, 0.0, 90.0), false));
        let built = build(&part);
        assert!(built.errors.is_empty(), "{:?}", built.errors);
        assert_eq!(built.bodies.len(), 1);
        let bb = built.bodies[0].body.outline_bounds();
        // Turned a quarter about the middle (20, 10): 20 × 40, then up 30.
        for (got, want) in [(bb.min.x, 10.0), (bb.max.x, 30.0), (bb.min.y, -10.0), (bb.max.y, 30.0), (bb.min.z, 30.0), (bb.max.z, 40.0)] {
            assert!((got - want).abs() < 1e-6, "{bb:?}");
        }
        assert!(built.bodies[0].topology.faces.iter().all(|f| f.fingerprint.origin == 2), "the extrude's faces stay the extrude's");
    }

    #[test]
    fn a_round_picked_on_the_moved_body_still_finds_its_edge_and_follows_the_box_made_taller() {
        // ScanRuler's "a Move": a 10 × 6 × 4 box turned a quarter about Z through (5, 3, 2) and
        // carried 100 along X; a fillet on its top edge picked as the panel picks it, and the box
        // made 5 tall above the move — the move and the round follow.
        let boxed = |height: f64| {
            let mut part = Part::default();
            let mut s = Sketch::default();
            s.add_rectangle(DVec2::ZERO, DVec2::new(10.0, 6.0));
            let regions = crate::build::default_regions(&find_regions(&s));
            let sk = part.add(FeatureKind::Sketch(SketchFeature::on_world(WorldAxis::Z, 0.0, s)));
            part.add(FeatureKind::Extrude(ExtrudeFeature::new(sk, regions, ExtrudeExtent::Distance { from: 0.0, to: height }, BodyOp::New)));
            part.add(FeatureKind::MoveBody(MoveBodyFeature { bodies: vec![BodyId::new(2)], pivot: DVec3::new(5.0, 3.0, 2.0), translate: DVec3::new(100.0, 0.0, 0.0), rotate: DVec3::new(0.0, 0.0, 90.0), copy: false, exprs: Exprs::new() }));
            part
        };
        let part = boxed(4.0);
        let built = build(&part);
        assert!(built.errors.is_empty(), "{:?}", built.errors);
        let b = &built.bodies[0];
        assert!((volume(&b.body).unwrap_or(0.0) - 240.0).abs() < 1e-6);
        let bb = b.body.outline_bounds();
        for (got, want) in [(bb.min.x, 102.0), (bb.min.y, -2.0), (bb.min.z, 0.0), (bb.max.x, 108.0), (bb.max.y, 8.0), (bb.max.z, 4.0)] {
            assert!((got - want).abs() < 1e-3, "{bb:?}");
        }
        let Some(edge) = b.topology.edges.iter().find(|e| e.fingerprint.kind == crate::topo::EdgeKind::Line && (e.fingerprint.mid.z - 4.0).abs() < 1e-6).map(|e| e.fingerprint.clone()) else { panic!("no top edge") };
        let fillet = |edge: EdgeRef| FeatureKind::Fillet(FilletFeature { body: BodyId::new(2), edges: vec![edge], edges_where: Vec::new(), radius: 1.0, chain: false, skip_failing: false, variable: Vec::new(), exprs: Exprs::new() });
        let mut rounded = part.clone();
        let f = rounded.add(fillet(edge.clone()));
        let built = build(&rounded);
        assert!(built.errors.is_empty(), "{:?}", built.errors);
        assert!(built.touched.contains_key(&f));
        assert!(volume(&built.bodies[0].body).unwrap_or(f64::INFINITY) < 240.0);
        let mut taller = boxed(5.0);
        taller.add(fillet(EdgeRef { mid: DVec3::new(edge.mid.x, edge.mid.y, 5.0), ..edge }));
        let built = build(&taller);
        assert!(built.errors.is_empty(), "{:?}", built.errors);
        assert!((built.bodies[0].body.outline_bounds().max.z - 5.0).abs() < 1e-3);
        assert!(volume(&built.bodies[0].body).unwrap_or(f64::INFINITY) < 300.0);
    }

    #[test]
    fn copies_the_body_stays_the_copy_a_body_of_the_moves_own_where_it_was_sent() {
        let mut part = block();
        let m = part.add(mover(vec![BodyId::new(2), BodyId::new(2)], DVec3::new(0.0, 50.0, 0.0), DVec3::ZERO, true));
        let built = build(&part);
        assert!(built.errors.is_empty(), "{:?}", built.errors);
        assert_eq!(built.bodies.iter().map(|b| b.id).collect::<Vec<_>>(), vec![BodyId::new(2), BodyId::new(m)], "one copy for the body named twice");
        let copy = &built.bodies[1];
        assert!((copy.body.outline_bounds().min.y - 50.0).abs() < 1e-6);
        assert!(copy.topology.faces.iter().all(|f| f.fingerprint.origin == m as i64), "a copy's faces are the move's");
        assert!((volume(&copy.body).unwrap_or(0.0) - 8000.0).abs() < 1e-3);
        assert_eq!(built.touched.get(&m), Some(&vec![BodyId::new(m)]));
    }

    #[test]
    fn fails_by_name_when_a_body_it_moves_is_gone_and_moves_nothing_then() {
        let mut part = block();
        let m = part.add(mover(vec![BodyId::new(2), BodyId::new(9)], DVec3::new(0.0, 0.0, 5.0), DVec3::ZERO, false));
        let built = build(&part);
        assert_eq!(built.errors.get(&m).map(String::as_str), Some("A body this move takes is gone, or failed to build."));
        assert!((built.bodies[0].body.outline_bounds().min.z).abs() < 1e-9, "the body stays where it was");
        let mut part = block();
        let m = part.add(mover(Vec::new(), DVec3::ZERO, DVec3::ZERO, false));
        assert_eq!(build(&part).errors.get(&m).map(String::as_str), Some("No body is picked to move."));
        let mut part = block();
        let m = part.add(mover(vec![BodyId::new(2)], DVec3::new(f64::NAN, 0.0, 0.0), DVec3::ZERO, false));
        assert_eq!(build(&part).errors.get(&m).map(String::as_str), Some("The move needs a number for every distance and angle."));
    }
}
