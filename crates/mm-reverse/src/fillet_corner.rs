// SPDX-License-Identifier: AGPL-3.0-only
//! A corner of the sketch made round, or cut off straight. Ported from ScanRuler
//! (`plugins/reverse/core/filletCorner.ts`), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! Two lines are cut back to where an arc of the asked radius touches them, the arc put
//! between, held to both with a coincidence and a tangency, and its radius made a dimension —
//! so the round is a number to type, as CAD's sketch fillet is. The opposite of
//! `sharpen_corner` (corners.rs), which reads the same four relations to take a round out
//! again.
//!
//! The two lines need not meet yet. A fit often leaves a corner as a short chord across the
//! round, or the operator has deleted that chord and the two ends stand off the corner: the
//! round then starts from where the ends are — a chord cut where the blend begins is as good
//! a first radius as the scan offers — and a piece that bridged the two ends is replaced by
//! the arc.
//!
//! A chamfer is the same cut with a straight piece put across (`chamfer_lines`).

use std::f64::consts::PI;

use glam::DVec2;
use serde::{Deserialize, Serialize};

use crate::corners::line_intersection;
use crate::fillet_num::js_num;
use crate::nominal::{nominal_of, NominalKind};
use crate::sketch::{Anchor, AnchorAt, ConstraintId, DimPlace, DistanceTo, Entity, EntityId, Rule, Shape, Sketch};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FilletResult {
    /// The arc put in, and the radius dimension on it.
    pub arc: EntityId,
    pub dimension: ConstraintId,
    pub radius: f64,
}

/// An end counts as at the corner within this, millimetres.
const AT_CORNER_MM: f64 = 1e-6;
/// The share of the shorter line a round with no radius to go by takes.
const DEFAULT_SHARE: f64 = 0.25;

/// `None` for a vector too short to have a direction, as ScanRuler's `normalize2` has it.
pub(crate) fn unit2(v: DVec2) -> Option<DVec2> {
    let l = v.length();
    if l > 1e-12 {
        Some(v / l)
    } else {
        None
    }
}

/// The other end of a line or an arc.
pub(crate) fn flip(at: AnchorAt) -> AnchorAt {
    if at == AnchorAt::A {
        AnchorAt::B
    } else {
        AnchorAt::A
    }
}

/// The anchors held coincident with `p`, directly.
pub(crate) fn joined_to(sketch: &Sketch, p: Anchor) -> Vec<Anchor> {
    let mut out = Vec::new();
    for c in &sketch.constraints {
        let Rule::Coincident { p: cp, q: cq } = c.rule else { continue };
        if cp == p {
            out.push(cq);
        } else if cq == p {
            out.push(cp);
        }
    }
    out
}

/// The piece whose two ends are joined to the two given ends — the chord a fit left across
/// the corner — or `None`.
fn bridge_between(sketch: &Sketch, p: Anchor, q: Anchor) -> Option<EntityId> {
    for j in joined_to(sketch, p) {
        if j.entity == q.entity || (j.at != AnchorAt::A && j.at != AnchorAt::B) {
            continue;
        }
        let other = Anchor::new(j.entity, flip(j.at));
        if joined_to(sketch, other).contains(&q) {
            return Some(j.entity);
        }
    }
    None
}

/// Take away the coincidence that joined two ends: the join that made the corner sharp goes,
/// and what is put in takes its place.
pub(crate) fn unjoin(sketch: &mut Sketch, a: Anchor, b: Anchor) {
    sketch.constraints.retain(|c| !matches!(c.rule, Rule::Coincident { p, q } if (p == a && q == b) || (p == b && q == a)));
}

/// The two lines that meet at an end — the end's own and the one joined to it there — or
/// `None` when the end is not a corner between two lines.
pub fn corner_lines_at(sketch: &Sketch, p: Anchor) -> Option<[EntityId; 2]> {
    let own = sketch.entity(p.entity)?;
    if !matches!(own.shape, Shape::Line { .. }) || p.at == AnchorAt::Center {
        return None;
    }
    for j in joined_to(sketch, p) {
        if let Some(e) = sketch.entity(j.entity) {
            if matches!(e.shape, Shape::Line { .. }) && e.id != own.id && j.at != AnchorAt::Center {
                return Some([own.id, e.id]);
            }
        }
    }
    None
}

/// A line seen from the corner: the end that is cut back, the way the line runs from the
/// corner, how far past the corner its near end stands, and how far its far end is.
#[derive(Debug, Clone, Copy)]
struct LineSeen {
    near: AnchorAt,
    dir: DVec2,
    off: f64,
    reach: f64,
}

fn line_ends(e: &Entity) -> Option<(DVec2, DVec2)> {
    match e.shape {
        Shape::Line { a, b } => Some((a, b)),
        _ => None,
    }
}

fn seen_line(a: DVec2, b: DVec2, x: DVec2) -> Option<LineSeen> {
    let near = if (a - x).length() <= (b - x).length() { AnchorAt::A } else { AnchorAt::B };
    let (near_p, far_p) = if near == AnchorAt::A { (a, b) } else { (b, a) };
    let dir = unit2(far_p - x)?;
    Some(LineSeen { near, dir, off: (near_p - x).dot(dir), reach: (far_p - x).length() })
}

/// Move one end of a line.
fn put(sketch: &mut Sketch, id: EntityId, at: AnchorAt, p: DVec2) {
    if let Some(e) = sketch.entity_mut(id) {
        if let Shape::Line { a, b } = &mut e.shape {
            if at == AnchorAt::A {
                *a = p;
            } else {
                *b = p;
            }
        }
    }
}

/// Round the corner between two lines. `radius` is the one asked for; without it the round
/// takes the radius the standing-off ends suggest, or a quarter of the shorter line. Returns
/// what was made, or in words why nothing was (the sketch then unchanged). `room` is how far
/// along each line the round may touch, where the caller knows better than the lines as they
/// now stand — a shape rounding all its corners measures it on the sides before the first one
/// was cut back.
pub fn fillet_lines(sketch: &mut Sketch, id_a: EntityId, id_b: EntityId, radius: Option<f64>, room: Option<f64>) -> Result<FilletResult, String> {
    let (la, lb) = match (sketch.entity(id_a), sketch.entity(id_b)) {
        (Some(la), Some(lb)) if la.id != lb.id => (la, lb),
        _ => return Err("A round goes between two lines.".to_string()),
    };
    let (Some((la_a, la_b)), Some((lb_a, lb_b))) = (line_ends(la), line_ends(lb)) else {
        return Err("A round goes between two lines.".to_string());
    };
    if la.projected || lb.projected {
        return Err("A projected piece stays as the part made it — it cannot be cut back for a round.".to_string());
    }
    let Some(x) = line_intersection((la_a, la_b), (lb_a, lb_b)) else {
        return Err("The two lines are parallel — no round fits between them.".to_string());
    };

    // Each line seen from the corner: the end that is cut back, the end that stays, and the
    // way the line runs from the corner.
    let (Some(a), Some(b)) = (seen_line(la_a, la_b, x), seen_line(lb_a, lb_b, x)) else {
        return Err("A line ends on the corner itself — there is nothing of it to round into.".to_string());
    };
    let half = a.dir.dot(b.dir).clamp(-1.0, 1.0).acos() / 2.0;
    let tan_half = half.tan();
    let bisector = unit2(a.dir + b.dir);
    let Some(bisector) = bisector.filter(|_| tan_half > 1e-6 && tan_half.is_finite()) else {
        return Err("The two lines run on in one line — there is no corner to round.".to_string());
    };

    // How far from the corner the round touches the lines.
    let fits = room.unwrap_or(0.95 * a.reach.min(b.reach));
    let touch = match radius {
        Some(radius) => {
            if !(radius > 0.0) {
                return Err("A round needs a radius above zero.".to_string());
            }
            let touch = radius / tan_half;
            if touch > fits {
                return Err(format!("A round of {} mm does not fit on these lines.", js_num(radius)));
            }
            touch
        }
        None => {
            let stand_off = (a.off.max(0.0) + b.off.max(0.0)) / 2.0;
            if stand_off > AT_CORNER_MM {
                stand_off.min(fits)
            } else {
                let guess = DEFAULT_SHARE * a.reach.min(b.reach) * tan_half;
                (nominal_of(guess, NominalKind::Radius).unwrap_or(guess) / tan_half).min(fits)
            }
        }
    };
    let r = touch * tan_half;

    let end_a = Anchor::new(id_a, a.near);
    let end_b = Anchor::new(id_b, b.near);
    if let Some(bridge) = bridge_between(sketch, end_a, end_b) {
        sketch.remove_entities(&[bridge]);
    }
    unjoin(sketch, end_a, end_b);

    let ta = x + a.dir * touch;
    let tb = x + b.dir * touch;
    let c = x + bisector * (r / half.sin());
    put(sketch, id_a, a.near, ta);
    put(sketch, id_b, b.near, tb);

    // Counter-clockwise the short way round, from whichever tangent point that starts at.
    let ang_a = (ta.y - c.y).atan2(ta.x - c.x);
    let ang_b = (tb.y - c.y).atan2(tb.x - c.x);
    let mut sweep = ang_b - ang_a;
    while sweep > PI {
        sweep -= 2.0 * PI;
    }
    while sweep <= -PI {
        sweep += 2.0 * PI;
    }
    let from_a = sweep > 0.0;
    let arc = sketch.add_shape(Shape::Arc { c, r, start: if from_a { ang_a } else { ang_b }, sweep: sweep.abs() });
    sketch.constrain(Rule::Coincident { p: end_a, q: Anchor::new(arc, if from_a { AnchorAt::A } else { AnchorAt::B }) });
    sketch.constrain(Rule::Coincident { p: end_b, q: Anchor::new(arc, if from_a { AnchorAt::B } else { AnchorAt::A }) });
    sketch.constrain(Rule::Tangent { a: id_a, b: arc });
    sketch.constrain(Rule::Tangent { a: id_b, b: arc });
    let dimension = sketch.constrain(Rule::Radius { entity: arc, value: r });
    Ok(FilletResult { arc, dimension, radius: r })
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ChamferResult {
    /// The line put across the corner, and the two dimensions that hold it: how far each of
    /// its ends stands off the other line of the corner.
    pub line: EntityId,
    pub dimensions: [ConstraintId; 2],
    /// How far along each line the corner was cut back.
    pub setback: f64,
}

/// Cut the corner between two lines off with a straight piece. `setback` is how far along
/// each line the cut starts; without it the standing-off ends say, or a quarter of the shorter
/// line. The piece is held by two dimensions, each from one of its ends to the other line of
/// the corner — on a square corner both read the setback itself, the 2 × 2 a drawing carries,
/// and typing over one makes the chamfer an uneven one. Returns what was made, or in words why
/// nothing was (the sketch then unchanged).
pub fn chamfer_lines(sketch: &mut Sketch, id_a: EntityId, id_b: EntityId, setback: Option<f64>) -> Result<ChamferResult, String> {
    let (la, lb) = match (sketch.entity(id_a), sketch.entity(id_b)) {
        (Some(la), Some(lb)) if la.id != lb.id => (la, lb),
        _ => return Err("A chamfer goes between two lines.".to_string()),
    };
    let (Some((la_a, la_b)), Some((lb_a, lb_b))) = (line_ends(la), line_ends(lb)) else {
        return Err("A chamfer goes between two lines.".to_string());
    };
    if la.projected || lb.projected {
        return Err("A projected piece stays as the part made it — it cannot be cut back for a chamfer.".to_string());
    }
    let Some(x) = line_intersection((la_a, la_b), (lb_a, lb_b)) else {
        return Err("The two lines are parallel — no chamfer fits between them.".to_string());
    };

    let (Some(a), Some(b)) = (seen_line(la_a, la_b, x), seen_line(lb_a, lb_b, x)) else {
        return Err("A line ends on the corner itself — there is nothing of it to cut back.".to_string());
    };
    let sin = (a.dir.x * b.dir.y - a.dir.y * b.dir.x).abs();
    if !(sin > 1e-6) {
        return Err("The two lines run on in one line — there is no corner to chamfer.".to_string());
    }

    let room = 0.95 * a.reach.min(b.reach);
    let cut = match setback {
        Some(setback) => {
            if !(setback > 0.0) {
                return Err("A chamfer needs a distance above zero.".to_string());
            }
            if setback > room {
                return Err(format!("A chamfer of {} mm does not fit on these lines.", js_num(setback)));
            }
            setback
        }
        None => {
            let stand_off = (a.off.max(0.0) + b.off.max(0.0)) / 2.0;
            if stand_off > AT_CORNER_MM {
                stand_off.min(room)
            } else {
                let guess = DEFAULT_SHARE * a.reach.min(b.reach);
                nominal_of(guess, NominalKind::Length).unwrap_or(guess).min(room)
            }
        }
    };

    let end_a = Anchor::new(id_a, a.near);
    let end_b = Anchor::new(id_b, b.near);
    if let Some(bridge) = bridge_between(sketch, end_a, end_b) {
        sketch.remove_entities(&[bridge]);
    }
    unjoin(sketch, end_a, end_b);

    let ta = x + a.dir * cut;
    let tb = x + b.dir * cut;
    put(sketch, id_a, a.near, ta);
    put(sketch, id_b, b.near, tb);

    let line = sketch.add_shape(Shape::Line { a: ta, b: tb });
    sketch.constrain(Rule::Coincident { p: end_a, q: Anchor::new(line, AnchorAt::A) });
    sketch.constrain(Rule::Coincident { p: end_b, q: Anchor::new(line, AnchorAt::B) });
    // Each distance is written outside the corner, clear of the pieces: its line stood off the
    // measured stretch, away from the other line's run.
    let outside = |sketch: &Sketch, from: DVec2, onto: EntityId, run: DVec2| -> Option<DimPlace> {
        let (oa, ob) = line_ends(sketch.entity(onto)?)?;
        let dir = unit2(ob - oa).unwrap_or(DVec2::X);
        let n = DVec2::new(-dir.y, dir.x);
        let d = unit2(n * -(from - oa).dot(n))?;
        let side = run.dot(DVec2::new(-d.y, d.x));
        let sign = if side > 0.0 {
            1.0
        } else if side < 0.0 {
            -1.0
        } else {
            1.0
        };
        Some(DimPlace { off: -sign * cut, along: 0.0 })
    };
    let first_place = outside(sketch, ta, id_b, b.dir);
    let first = sketch.constrain(Rule::Distance { p: Anchor::new(line, AnchorAt::A), q: DistanceTo::Line(id_b), value: cut * sin });
    if let Some(c) = sketch.constraint_mut(first) {
        c.place = first_place;
    }
    let second_place = outside(sketch, tb, id_a, a.dir);
    let second = sketch.constrain(Rule::Distance { p: Anchor::new(line, AnchorAt::B), q: DistanceTo::Line(id_a), value: cut * sin });
    if let Some(c) = sketch.constraint_mut(second) {
        c.place = second_place;
    }
    Ok(ChamferResult { line, dimensions: [first, second], setback: cut })
}

#[cfg(test)]
mod tests {
    //! Ported from ScanRuler `plugins/reverse/tests/reFilletCorner.test.ts` (filletLines,
    //! cornerLinesAt and the rounded corner the solve moves; the Fillet tool in the store is
    //! the app's) and the `chamferLines` part of `reDimLayout.test.ts`.
    use super::*;
    use crate::corners::sharpen_corner;
    use crate::sketch::{anchor_point, TOUCH_MM};
    use crate::solver::{solve_sketch, SolveOptions};

    fn close(a: f64, b: f64, digits: i32) -> bool {
        (a - b).abs() < 10f64.powi(-digits) / 2.0
    }

    /// Two lines meeting square at the origin, joined there.
    fn corner() -> (Sketch, EntityId, EntityId) {
        let mut sketch = Sketch::default();
        let a = sketch.add_shape(Shape::Line { a: DVec2::new(10.0, 0.0), b: DVec2::ZERO });
        let b = sketch.add_shape(Shape::Line { a: DVec2::ZERO, b: DVec2::new(0.0, 8.0) });
        sketch.constrain(Rule::Coincident { p: Anchor::new(a, AnchorAt::B), q: Anchor::new(b, AnchorAt::A) });
        (sketch, a, b)
    }

    fn line_of(sketch: &Sketch, id: EntityId) -> (DVec2, DVec2) {
        match sketch.entity(id).map(|e| e.shape.clone()) {
            Some(Shape::Line { a, b }) => (a, b),
            other => panic!("not a line: {other:?}"),
        }
    }

    fn arc_of(sketch: &Sketch, id: EntityId) -> (DVec2, f64, f64, f64) {
        match sketch.entity(id).map(|e| e.shape.clone()) {
            Some(Shape::Arc { c, r, start, sweep }) => (c, r, start, sweep),
            other => panic!("not an arc: {other:?}"),
        }
    }

    fn solve(sketch: &mut Sketch) -> bool {
        solve_sketch(sketch, SolveOptions::default()).ok
    }

    fn kind_name(rule: &Rule) -> &'static str {
        match rule {
            Rule::Coincident { .. } => "coincident",
            Rule::Radius { .. } => "radius",
            Rule::Tangent { .. } => "tangent",
            Rule::Distance { .. } => "distance",
            other => other.name(),
        }
    }

    #[test]
    fn cuts_both_lines_back_and_puts_a_tangent_arc_of_the_radius_between_them() {
        let (mut sketch, a, b) = corner();
        let made = fillet_lines(&mut sketch, a, b, Some(2.0), None).expect("a round");
        let (_, la_b) = line_of(&sketch, a);
        let (lb_a, _) = line_of(&sketch, b);
        let (c, r, _, sweep) = arc_of(&sketch, made.arc);
        assert!(close(la_b.x, 2.0, 2) && close(la_b.y, 0.0, 2));
        assert!(close(lb_a.x, 0.0, 2) && close(lb_a.y, 2.0, 2));
        assert!(close(c.x, 2.0, 2) && close(c.y, 2.0, 2));
        assert!(close(r, 2.0, 2));
        assert!(close(sweep, PI / 2.0, 2));
        // The arc's two ends are the two tangent points, whichever way it runs.
        let shape = sketch.entity(made.arc).map(|e| e.shape.clone()).expect("the arc");
        let ends = [anchor_point(&shape, AnchorAt::A), anchor_point(&shape, AnchorAt::B)];
        assert!(ends.iter().any(|p| (*p - DVec2::new(2.0, 0.0)).length() < 1e-9));
        assert!(ends.iter().any(|p| (*p - DVec2::new(0.0, 2.0)).length() < 1e-9));
        // The sharp join is gone; two joins, two tangencies and the radius hold it.
        let mut kinds: Vec<&str> = sketch.constraints.iter().map(|c| kind_name(&c.rule)).collect();
        kinds.sort();
        assert_eq!(kinds, vec!["coincident", "coincident", "radius", "tangent", "tangent"]);
        assert!(solve(&mut sketch));
    }

    #[test]
    fn follows_a_typed_radius_through_the_solver_and_sharpen_corner_takes_it_out_again() {
        let (mut sketch, a, b) = corner();
        let made = fillet_lines(&mut sketch, a, b, Some(2.0), None).expect("a round");
        if let Some(c) = sketch.constraint_mut(made.dimension) {
            c.rule.set_value(3.0);
        }
        assert!(solve(&mut sketch));
        assert!(close(arc_of(&sketch, made.arc).1, 3.0, 4));
        assert!(sharpen_corner(&mut sketch, made.arc));
        assert_eq!(sketch.entities.len(), 2);
    }

    #[test]
    fn takes_its_radius_from_ends_that_stand_off_the_corner_and_replaces_the_chord_between_them() {
        let mut sketch = Sketch::default();
        let a = sketch.add_shape(Shape::Line { a: DVec2::new(10.0, 0.0), b: DVec2::new(3.0, 0.0) });
        let b = sketch.add_shape(Shape::Line { a: DVec2::new(0.0, 3.0), b: DVec2::new(0.0, 8.0) });
        let chord = sketch.add_shape(Shape::Line { a: DVec2::new(3.0, 0.0), b: DVec2::new(0.0, 3.0) });
        sketch.constrain(Rule::Coincident { p: Anchor::new(a, AnchorAt::B), q: Anchor::new(chord, AnchorAt::A) });
        sketch.constrain(Rule::Coincident { p: Anchor::new(chord, AnchorAt::B), q: Anchor::new(b, AnchorAt::A) });
        let made = fillet_lines(&mut sketch, a, b, None, None).expect("a round");
        assert!(close(made.radius, 3.0, 2));
        assert!(sketch.entity(chord).is_none());
        assert_eq!(sketch.entities.len(), 3);
    }

    #[test]
    fn rounds_a_corner_that_is_not_square_on_the_inside_of_it() {
        let mut sketch = Sketch::default();
        let a = sketch.add_shape(Shape::Line { a: DVec2::ZERO, b: DVec2::new(10.0, 0.0) });
        let b = sketch.add_shape(Shape::Line { a: DVec2::ZERO, b: DVec2::new(10.0, 10.0) });
        let made = fillet_lines(&mut sketch, a, b, Some(1.0), None).expect("a round");
        let (c, _, _, sweep) = arc_of(&sketch, made.arc);
        // On the bisector at 22.5°, a radius above the first line.
        assert!(close(c.y, 1.0, 2));
        assert!(close(c.x, 1.0 / (PI / 8.0).tan(), 2));
        assert!(close(sweep, PI - PI / 4.0, 2));
    }

    #[test]
    fn refuses_in_words_and_changes_nothing_parallel_lines_a_round_too_large() {
        let mut sketch = Sketch::default();
        let a = sketch.add_shape(Shape::Line { a: DVec2::ZERO, b: DVec2::new(10.0, 0.0) });
        let b = sketch.add_shape(Shape::Line { a: DVec2::new(0.0, 5.0), b: DVec2::new(10.0, 5.0) });
        assert!(fillet_lines(&mut sketch, a, b, Some(1.0), None).is_err());
        let (mut c, ca, cb) = corner();
        let before = c.clone();
        assert!(fillet_lines(&mut c, ca, cb, Some(50.0), None).is_err());
        assert_eq!(c, before);
    }

    #[test]
    fn keeps_a_rectangle_one_closed_region() {
        // Adapted: ScanRuler counts the sketch's regions (regions.ts, which here rests on
        // SolidCraft's region finder). The test stands on what makes the one region: after the
        // round and the solve, every end of every piece is joined to another piece's end and
        // lies on it.
        let mut sketch = Sketch::default();
        let ids = sketch.add_rectangle(DVec2::ZERO, DVec2::new(20.0, 10.0));
        assert!(fillet_lines(&mut sketch, ids[0], ids[1], Some(2.0), None).is_ok());
        assert!(solve(&mut sketch));
        assert_eq!(sketch.entities.len(), 5);
        for e in &sketch.entities {
            for at in [AnchorAt::A, AnchorAt::B] {
                let end = Anchor::new(e.id, at);
                let joined = joined_to(&sketch, end);
                assert_eq!(joined.len(), 1, "{end:?} joined once");
                let (Some(p), Some(q)) = (sketch.anchor(end), sketch.anchor(joined[0])) else { panic!("both ends") };
                assert!(joined[0].entity != e.id && (p - q).length() < TOUCH_MM);
            }
        }
    }

    #[test]
    fn corner_lines_at_finds_the_two_lines_joined_at_an_end_and_nothing_at_a_free_end() {
        let (sketch, a, b) = corner();
        let mut found = corner_lines_at(&sketch, Anchor::new(a, AnchorAt::B)).expect("a corner");
        found.sort();
        let mut want = [a, b];
        want.sort();
        assert_eq!(found, want);
        assert!(corner_lines_at(&sketch, Anchor::new(a, AnchorAt::A)).is_none());
    }

    /// The worst angle, degrees, between a line and an arc where the two meet end to end —
    /// what a kink in the built face would be.
    fn worst_joint_deg(sketch: &Sketch) -> f64 {
        let mut worst: f64 = 0.0;
        for c in &sketch.constraints {
            let Rule::Coincident { p, q } = c.rule else { continue };
            let (Some(ep), Some(eq)) = (sketch.entity(p.entity), sketch.entity(q.entity)) else { continue };
            let pair = match (ep.shape.clone(), eq.shape.clone()) {
                (Shape::Line { a, b }, Shape::Arc { start, sweep, .. }) => Some((a, b, start, sweep, q.at)),
                (Shape::Arc { start, sweep, .. }, Shape::Line { a, b }) => Some((a, b, start, sweep, p.at)),
                _ => None,
            };
            let Some((la, lb, start, sweep, at)) = pair else { continue };
            let t = if at == AnchorAt::A { start } else { start + sweep };
            let d = lb - la;
            let cos = (t.cos() * d.x + t.sin() * d.y).abs() / d.length();
            worst = worst.max(cos.min(1.0).asin().to_degrees());
        }
        worst
    }

    #[test]
    fn a_rounded_corner_the_solve_moves_stays_tangent_to_the_micro_degree() {
        // A small rounded rectangle, its sides dimensioned to another size: the solve walks
        // every joint along its line. A tangency held as the line's distance from the centre
        // lets the joint slide off the tangent point by the square root of what is left — a
        // few hundredths of a degree.
        let mut sketch = Sketch::default();
        let sides = sketch.add_rectangle(DVec2::new(-2.95, -1.275), DVec2::new(2.95, 1.275));
        for i in 0..4 {
            fillet_lines(&mut sketch, sides[i], sides[(i + 1) % 4], Some(1.25), Some(1.275)).expect("a round");
        }
        sketch.constrain(Rule::Length { entity: sides[0], value: 3.9 });
        sketch.constrain(Rule::Length { entity: sides[1], value: 0.3 });
        assert!(solve(&mut sketch));
        let (a0, b0) = line_of(&sketch, sides[0]);
        assert!(close(b0.x - a0.x, 3.9, 6), "{}", b0.x - a0.x);
        let worst = worst_joint_deg(&sketch);
        assert!(worst < 1e-4, "{worst}");
    }

    fn dimension_value(sketch: &Sketch, id: ConstraintId) -> f64 {
        sketch.constraint(id).and_then(|c| c.rule.value()).unwrap_or(f64::NAN)
    }

    #[test]
    fn chamfer_lines_cuts_both_lines_back_and_puts_a_line_across_held_by_its_two_distances() {
        let (mut sketch, a, b) = corner();
        let made = chamfer_lines(&mut sketch, a, b, Some(2.0)).expect("a chamfer");
        assert_eq!(line_of(&sketch, a).1, DVec2::new(2.0, 0.0));
        assert_eq!(line_of(&sketch, b).0, DVec2::new(0.0, 2.0));
        let across = line_of(&sketch, made.line);
        assert_eq!(across.0, DVec2::new(2.0, 0.0));
        assert_eq!(across.1, DVec2::new(0.0, 2.0));
        assert_eq!(made.dimensions.map(|id| dimension_value(&sketch, id)), [2.0, 2.0]);
        assert!(solve(&mut sketch));
    }

    #[test]
    fn chamfer_lines_writes_each_distance_outside_the_corner() {
        // Not in ScanRuler's tests: `chamferLines`'s placing worked through by hand on the
        // square corner — each dimension's line stood off by the setback, away from the other
        // line's run.
        let (mut sketch, a, b) = corner();
        let made = chamfer_lines(&mut sketch, a, b, Some(2.0)).expect("a chamfer");
        let place = |id: ConstraintId| sketch.constraint(id).and_then(|c| c.place);
        assert_eq!(place(made.dimensions[0]), Some(DimPlace { off: 2.0, along: 0.0 }));
        assert_eq!(place(made.dimensions[1]), Some(DimPlace { off: -2.0, along: 0.0 }));
    }

    #[test]
    fn chamfer_lines_takes_a_new_distance_typed_over_one_of_the_two_as_an_uneven_chamfer() {
        let (mut sketch, a, b) = corner();
        let made = chamfer_lines(&mut sketch, a, b, Some(2.0)).expect("a chamfer");
        if let Some(c) = sketch.constraint_mut(made.dimensions[0]) {
            c.rule.set_value(3.0);
        }
        sketch.constrain(Rule::Horizontal { entity: a });
        sketch.constrain(Rule::Vertical { entity: b });
        assert!(solve(&mut sketch));
        let across = line_of(&sketch, made.line);
        let lb = line_of(&sketch, b);
        assert!(close((across.0.x - lb.0.x).abs(), 3.0, 4));
    }

    #[test]
    fn chamfer_lines_refuses_a_distance_that_does_not_fit_and_changes_nothing() {
        let (mut sketch, a, b) = corner();
        let before = sketch.clone();
        assert!(chamfer_lines(&mut sketch, a, b, Some(9.0)).is_err());
        assert_eq!(sketch, before);
    }

    #[test]
    fn chamfer_lines_is_taken_out_again_by_sharp_corner() {
        let (mut sketch, a, b) = corner();
        let made = chamfer_lines(&mut sketch, a, b, Some(2.0)).expect("a chamfer");
        assert!(sharpen_corner(&mut sketch, made.line));
        assert_eq!(sketch.entities.len(), 2);
        assert_eq!(line_of(&sketch, a).1, DVec2::ZERO);
        assert_eq!(sketch.constraints.iter().filter(|c| matches!(c.rule, Rule::Distance { .. })).count(), 0);
    }
}
