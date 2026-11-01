// SPDX-License-Identifier: AGPL-3.0-only
//! Moving the pieces of a sketch by hand: an end or a centre dragged to a new spot, a whole
//! piece carried along, and the sketch turned so that an edge the scanner left crooked lies
//! along the sheet. Ported from ScanRuler (`plugins/reverse/core/drag.ts`), © Stefan Hermann
//! (CNC Kitchen), AGPL-3.0-only.
//!
//! Pure geometry on one piece; the solver settles what the move does to the rest — but for a
//! shape defined in all but its place, which the hand carries whole (`rigid_cluster`).

use std::collections::HashSet;
use std::f64::consts::TAU;

use glam::DVec2;

use crate::sketch::{anchor_point, constraint_entities, Anchor, AnchorAt, Constraint, DistanceTo, Entity, EntityId, MirrorAxis, OnTarget, Rule, Shape, Sketch};
use crate::solver::{solve_sketch, SolveOptions};

/// An angle brought into (0, 2π] — an arc's sweep, which is never nothing.
fn positive_sweep(a: f64) -> f64 {
    let s = ((a % TAU) + TAU) % TAU;
    if s == 0.0 {
        TAU
    } else {
        s
    }
}

/// The piece with its anchor moved to `p`: a line's end goes there, an arc's end swings round
/// its centre to the angle of `p` — the radius stays, so the other end stays — and a centre
/// carries the piece.
pub fn move_anchor(e: &Entity, at: AnchorAt, p: DVec2) -> Entity {
    if let Shape::Spline(sp) = &e.shape {
        // A fit point goes where the hand is, the curve bending through it.
        let i = match at {
            AnchorAt::P(i) => crate::sketch_spline::spline_fit_point_index(sp, i as usize),
            _ => crate::sketch_spline::spline_point_index(sp, at),
        };
        let Some(i) = i else { return translate_entity(e, p - anchor_point(&e.shape, at)) };
        let mut moved = sp.clone();
        moved.pts[i] = p;
        return Entity { shape: Shape::Spline(moved), ..e.clone() };
    }
    match e.shape {
        Shape::Spline(_) => e.clone(),
        Shape::Line { a, b } => match at {
            AnchorAt::A => Entity { shape: Shape::Line { a: p, b }, ..e.clone() },
            AnchorAt::B => Entity { shape: Shape::Line { a, b: p }, ..e.clone() },
            AnchorAt::Center | AnchorAt::P(_) => translate_entity(e, p - (a + b) / 2.0),
        },
        Shape::Arc { c, r, start, sweep } => {
            if at == AnchorAt::Center {
                return translate_entity(e, p - c);
            }
            let angle = (p.y - c.y).atan2(p.x - c.x);
            if at == AnchorAt::A {
                let end = start + sweep;
                return Entity { shape: Shape::Arc { c, r, start: angle, sweep: positive_sweep(end - angle) }, ..e.clone() };
            }
            Entity { shape: Shape::Arc { c, r, start, sweep: positive_sweep(angle - start) }, ..e.clone() }
        }
        Shape::Circle { c, .. } => translate_entity(e, p - c),
        Shape::Point { .. } => Entity { shape: Shape::Point { p }, ..e.clone() },
    }
}

/// The piece carried by `delta`, slice points and all.
pub fn translate_entity(e: &Entity, delta: DVec2) -> Entity {
    let shape = match e.shape {
        Shape::Spline(ref sp) => Shape::Spline(crate::sketch_spline::SplineShape { pts: sp.pts.iter().map(|q| *q + delta).collect(), ..sp.clone() }),
        Shape::Line { a, b } => Shape::Line { a: a + delta, b: b + delta },
        Shape::Arc { c, r, start, sweep } => Shape::Arc { c: c + delta, r, start, sweep },
        Shape::Circle { c, r } => Shape::Circle { c: c + delta, r },
        Shape::Point { p } => Shape::Point { p: p + delta },
    };
    Entity { shape, points: e.points.iter().map(|q| *q + delta).collect(), ..e.clone() }
}

/// The piece without its slice points: what a dragged fitted piece becomes, since a hand that
/// moves it has overruled the scan.
pub fn detached(e: &Entity) -> Entity {
    Entity { points: Vec::new(), scatter: 0.0, ..e.clone() }
}

fn rotate(p: DVec2, c: f64, s: f64) -> DVec2 {
    DVec2::new(p.x * c - p.y * s, p.x * s + p.y * c)
}

/// The piece turned by `angle` radians about the sheet origin.
pub fn rotate_entity(e: &Entity, angle: f64) -> Entity {
    let (s, c) = angle.sin_cos();
    let shape = match e.shape {
        Shape::Spline(ref sp) => Shape::Spline(crate::sketch_spline::spline_placed(sp, |q| rotate(q, c, s), angle)),
        Shape::Line { a, b } => Shape::Line { a: rotate(a, c, s), b: rotate(b, c, s) },
        Shape::Arc { c: centre, r, start, sweep } => Shape::Arc { c: rotate(centre, c, s), r, start: start + angle, sweep },
        Shape::Circle { c: centre, r } => Shape::Circle { c: rotate(centre, c, s), r },
        Shape::Point { p } => Shape::Point { p: rotate(p, c, s) },
    };
    Entity { shape, points: e.points.iter().map(|q| rotate(*q, c, s)).collect(), ..e.clone() }
}

/// Turn the whole sketch in place by `angle` about the sheet origin — what aligning the sheet
/// to one of its lines does to the drawing, the frame turning the other way underneath.
pub fn rotate_sketch(sketch: &mut Sketch, angle: f64) {
    for e in &mut sketch.entities {
        *e = rotate_entity(e, angle);
    }
}

/// Whether a relation or a dimension ties its piece to the sheet itself — a coordinate plane's
/// line, an axis — rather than to another piece.
fn on_sheet(rule: &Rule) -> bool {
    match rule {
        Rule::PointOn { on, .. } | Rule::Collinear { on, .. } => matches!(on, OnTarget::Datum(_)),
        Rule::Symmetric { about, .. } => !matches!(about, MirrorAxis::Line(_)),
        Rule::Distance { q, .. } => matches!(q, DistanceTo::Datum(_)),
        _ => false,
    }
}

/// What the hand holds of a piece: one of its anchors, or the piece itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrabAt {
    Anchor(AnchorAt),
    Body,
}

/// The pieces a drag carries along whole, or `None` when the solver is to share the move out.
/// A shape its relations and dimensions leave nothing but its place on the sheet — a rectangle
/// with its width and height, not yet put anywhere — can follow the hand only whole: every
/// piece joined to the one in the hand, through relations and dimensions, is carried by the
/// hand's move as it is, rather than eased after it by the solver a step behind. Not a shape
/// held anywhere — a fixed, projected or fitted piece, a relation to a coordinate plane — nor
/// one with a freedom left besides its place, which the drag is there to use.
pub fn rigid_cluster(sketch: &Sketch, entity: EntityId, at: GrabAt) -> Option<Vec<EntityId>> {
    // The shape: what is joined to the piece in the hand.
    let mut shape: HashSet<EntityId> = HashSet::from([entity]);
    let mut grew = true;
    while grew {
        grew = false;
        for c in &sketch.constraints {
            let ids = constraint_entities(&c.rule);
            if !ids.iter().any(|i| shape.contains(i)) || ids.iter().all(|i| shape.contains(i)) {
                continue;
            }
            shape.extend(ids);
            grew = true;
        }
    }
    let pieces: Vec<&Entity> = sketch.entities.iter().filter(|e| shape.contains(&e.id)).collect();
    if pieces.iter().any(|e| e.fixed || e.projected || !e.points.is_empty()) {
        return None;
    }
    let own: Vec<Constraint> = sketch.constraints.iter().filter(|c| constraint_entities(&c.rule).iter().any(|i| shape.contains(i))).cloned().collect();
    if own.iter().any(|c| on_sheet(&c.rule)) {
        return None;
    }
    // Defined but for its place: held where the hand has it — the end or the centre at its
    // spot, a piece where it lies — nothing of it is left loose.
    let mut probe = Sketch {
        entities: pieces
            .iter()
            .map(|e| {
                let mut copy = (*e).clone();
                if e.id == entity && at == GrabAt::Body {
                    copy.fixed = true;
                }
                copy
            })
            .collect(),
        constraints: own,
        next_id: sketch.next_id,
    };
    if let GrabAt::Anchor(anchor) = at {
        let e = probe.entity(entity)?;
        let mut pin = Entity::new(Shape::Point { p: anchor_point(&e.shape, anchor) });
        pin.fixed = true;
        let pin = probe.add(pin);
        probe.constrain(Rule::Coincident { p: Anchor::new(entity, anchor), q: Anchor::new(pin, AnchorAt::Center) });
    }
    let loose: HashSet<EntityId> = solve_sketch(&mut probe, SolveOptions::default()).under_constrained.into_iter().collect();
    if pieces.iter().any(|e| loose.contains(&e.id)) {
        None
    } else {
        Some(pieces.iter().map(|e| e.id).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::SectionFrame;
    use crate::solver::Pull;
    use glam::DVec3;
    use std::f64::consts::PI;

    fn v(x: f64, y: f64) -> DVec2 {
        DVec2::new(x, y)
    }

    fn line_of(e: &Entity) -> (DVec2, DVec2) {
        match e.shape.clone() {
            Shape::Line { a, b } => (a, b),
            other => panic!("not a line: {other:?}"),
        }
    }

    fn close(a: f64, b: f64, digits: i32) -> bool {
        (a - b).abs() < 0.5 * 10f64.powi(-digits)
    }

    #[test]
    fn moves_a_line_end_carries_a_line_by_its_middle_and_moves_a_point() {
        let mut line = Entity::new(Shape::Line { a: v(0.0, 0.0), b: v(10.0, 0.0) }).fitted(vec![v(1.0, 0.0), v(2.0, 0.0)], 0.0);
        line.id = 1;
        assert_eq!(line_of(&move_anchor(&line, AnchorAt::B, v(10.0, 5.0))).1, v(10.0, 5.0));
        let carried = move_anchor(&line, AnchorAt::Center, v(7.0, 2.0));
        assert_eq!(line_of(&carried), (v(2.0, 2.0), v(12.0, 2.0)));
        assert_eq!(carried.points, vec![v(3.0, 2.0), v(4.0, 2.0)]);
        let point = Entity::new(Shape::Point { p: v(0.0, 0.0) });
        assert_eq!(move_anchor(&point, AnchorAt::Center, v(3.0, 4.0)).shape, Shape::Point { p: v(3.0, 4.0) });
    }

    #[test]
    fn swings_an_arc_end_round_its_centre_keeping_the_radius_and_the_other_end() {
        let arc = Entity::new(Shape::Arc { c: v(0.0, 0.0), r: 10.0, start: 0.0, sweep: PI / 2.0 });
        let Shape::Arc { r, start, sweep, .. } = move_anchor(&arc, AnchorAt::B, v(-7.0, 7.0)).shape else { panic!() };
        assert_eq!((r, start), (10.0, 0.0));
        assert!(close(sweep, (3.0 * PI) / 4.0, 9));
        let end_before = anchor_point(&arc.shape, AnchorAt::B);
        let other = move_anchor(&arc, AnchorAt::A, v(7.0, -7.0));
        let end_after = anchor_point(&other.shape, AnchorAt::B);
        assert!(close(end_after.x, end_before.x, 9) && close(end_after.y, end_before.y, 9));
        let Shape::Arc { start, sweep, .. } = other.shape else { panic!() };
        assert!(close(start, -PI / 4.0, 9));
        assert!(close(sweep, (3.0 * PI) / 4.0, 9));
    }

    #[test]
    fn carries_a_circle_and_detaches_a_fitted_piece_from_its_points() {
        let circle = translate_entity(&Entity::new(Shape::Circle { c: v(1.0, 1.0), r: 2.0 }).fitted(vec![v(3.0, 1.0)], 0.01), v(1.0, 1.0));
        assert_eq!(circle.shape, Shape::Circle { c: v(2.0, 2.0), r: 2.0 });
        assert_eq!(circle.points, vec![v(4.0, 2.0)]);
        let free = detached(&circle);
        assert!(free.points.is_empty());
        assert_eq!(free.scatter, 0.0);
        assert_eq!(free.shape, Shape::Circle { c: v(2.0, 2.0), r: 2.0 });
    }

    #[test]
    fn pulls_a_dragged_end_while_the_constraints_hold_a_horizontal_line_slides_not_tilts() {
        let mut sketch = Sketch::default();
        let id = sketch.add_shape(Shape::Line { a: v(0.0, 0.0), b: v(10.0, 0.0) });
        sketch.constrain(Rule::Horizontal { entity: id });
        // The hand pulls the end up and to the right.
        sketch.entities = sketch.entities.iter().map(|e| if e.id == id { move_anchor(e, AnchorAt::B, v(14.0, 3.0)) } else { e.clone() }).collect();
        let pull = [Pull { anchor: Anchor::new(id, AnchorAt::B), to: v(14.0, 3.0) }];
        let report = solve_sketch(&mut sketch, SolveOptions { pull: &pull, ..Default::default() });
        assert!(report.ok);
        let (a, b) = line_of(&sketch.entities[0]);
        assert!(close(b.y, a.y, 5));
        assert!(close(b.x, 14.0, 2));
    }

    #[test]
    fn carries_a_joined_neighbour_along_with_the_dragged_end() {
        let mut sketch = Sketch::default();
        let [bottom, right, ..] = sketch.add_rectangle(v(0.0, 0.0), v(10.0, 10.0));
        solve_sketch(&mut sketch, SolveOptions::default());
        // Drag the bottom-right corner outward; the right side's foot follows.
        sketch.entities = sketch.entities.iter().map(|e| if e.id == bottom { move_anchor(e, AnchorAt::B, v(14.0, 0.0)) } else { e.clone() }).collect();
        let pull = [Pull { anchor: Anchor::new(bottom, AnchorAt::B), to: v(14.0, 0.0) }];
        solve_sketch(&mut sketch, SolveOptions { pull: &pull, ..Default::default() });
        let b = line_of(sketch.entity(bottom).unwrap());
        let r = line_of(sketch.entity(right).unwrap());
        assert!(close(b.1.x, 14.0, 2));
        assert!(close(r.0.x, b.1.x, 5) && close(r.0.y, b.1.y, 5));
    }

    #[test]
    fn turns_every_piece_arcs_by_their_start_angle_points_and_slice_points_along() {
        let angle = PI / 2.0;
        let line = rotate_entity(&Entity::new(Shape::Line { a: v(1.0, 0.0), b: v(2.0, 0.0) }).fitted(vec![v(1.5, 0.0)], 0.0), angle);
        let (a, b) = line_of(&line);
        assert!(close(a.x, 0.0, 9) && close(a.y, 1.0, 9) && close(b.y, 2.0, 9));
        assert!(close(line.points[0].y, 1.5, 9));
        let arc = rotate_entity(&Entity::new(Shape::Arc { c: v(1.0, 0.0), r: 1.0, start: 0.0, sweep: 1.0 }), angle);
        let Shape::Arc { c, start, sweep, .. } = arc.shape else { panic!() };
        assert!(close(c.y, 1.0, 9));
        assert!(close(start, angle, 9));
        assert_eq!(sweep, 1.0);
    }

    #[test]
    fn leaves_every_point_where_it_was_on_the_part_once_the_frame_turns_the_other_way() {
        let frame = SectionFrame { origin: DVec3::new(5.0, 5.0, 5.0), normal: DVec3::Z, u: DVec3::X, v: DVec3::Y };
        let mut sketch = Sketch::default();
        // A line 30° off the sheet's X.
        let theta = PI / 6.0;
        sketch.add_shape(Shape::Line { a: v(0.0, 0.0), b: v(10.0 * theta.cos(), 10.0 * theta.sin()) });
        sketch.add_shape(Shape::Circle { c: v(3.0, 4.0), r: 1.0 });
        let lifted = |f: &SectionFrame, e: &Entity| match e.shape {
            Shape::Line { b, .. } => f.to_world(b),
            _ => f.to_world(e.shape.centre().unwrap_or_default()),
        };
        let before: Vec<DVec3> = sketch.entities.iter().map(|e| lifted(&frame, e)).collect();
        rotate_sketch(&mut sketch, -theta);
        let u = frame.u * theta.cos() + frame.v * theta.sin();
        let turned = SectionFrame { u, v: frame.normal.cross(u), ..frame };
        let (_, b) = line_of(&sketch.entities[0]);
        assert!(close(b.y, 0.0, 9) && close(b.x, 10.0, 9));
        let after: Vec<DVec3> = sketch.entities.iter().map(|e| lifted(&turned, e)).collect();
        for (p, q) in before.iter().zip(&after) {
            assert!((*p - *q).abs().max_element() < 5e-10);
        }
    }

    /// A 20 × 10 rectangle at (10, 10), drawn as the rectangle tool draws it.
    fn rectangle() -> (Sketch, [EntityId; 4]) {
        let mut s = Sketch::default();
        let ids = s.add_rectangle(v(10.0, 10.0), v(30.0, 20.0));
        (s, ids)
    }

    #[test]
    fn a_shape_defined_in_all_but_its_place_follows_the_hand_whole() {
        // Its width and height dimensioned, as the Dimension tool adds them.
        let (mut s, ids) = rectangle();
        s.constrain(Rule::Length { entity: ids[0], value: 20.0 });
        s.constrain(Rule::Length { entity: ids[1], value: 10.0 });
        assert_eq!(rigid_cluster(&s, ids[1], GrabAt::Anchor(AnchorAt::B)), Some(vec![1, 2, 3, 4]));
        assert_eq!(rigid_cluster(&s, ids[0], GrabAt::Body), Some(vec![1, 2, 3, 4]));
    }

    #[test]
    fn a_shape_held_where_its_corner_is_dropped_on_the_sheets_zero_is_carried_no_more() {
        let (mut s, ids) = rectangle();
        s.constrain(Rule::Length { entity: ids[0], value: 20.0 });
        s.constrain(Rule::Length { entity: ids[1], value: 10.0 });
        // Dropped on the zero, the corner is held there by the planes crossing at it.
        s.constrain(Rule::PointOn { p: Anchor::new(ids[0], AnchorAt::A), on: OnTarget::Datum(crate::datums::DatumPlane::Yz), bounded: false });
        s.constrain(Rule::PointOn { p: Anchor::new(ids[0], AnchorAt::A), on: OnTarget::Datum(crate::datums::DatumPlane::Zx), bounded: false });
        assert_eq!(rigid_cluster(&s, ids[1], GrabAt::Anchor(AnchorAt::B)), None);
    }

    #[test]
    fn a_shape_with_a_freedom_left_is_the_solvers() {
        let (s, ids) = rectangle();
        assert_eq!(rigid_cluster(&s, ids[1], GrabAt::Anchor(AnchorAt::B)), None);
    }

    #[test]
    fn a_fitted_or_fixed_shape_is_never_carried_whole() {
        let (mut s, ids) = rectangle();
        s.constrain(Rule::Length { entity: ids[0], value: 20.0 });
        s.constrain(Rule::Length { entity: ids[1], value: 10.0 });
        s.entity_mut(ids[2]).unwrap().fixed = true;
        assert_eq!(rigid_cluster(&s, ids[1], GrabAt::Anchor(AnchorAt::B)), None);
    }
}
