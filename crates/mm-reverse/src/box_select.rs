// SPDX-License-Identifier: AGPL-3.0-only
//! Which pieces of a sketch a selection box takes. Ported from ScanRuler
//! (`plugins/reverse/core/boxSelect.ts`), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! The two ways CAD sketchers read a box: a window, dragged rightward, takes what lies wholly
//! inside it; a crossing box, dragged leftward, takes whatever it touches. The box is the
//! screen rectangle laid on the sheet — four corners in sheet millimetres, in order round it —
//! so a sheet shown turned still gets a true answer. Exact for lines and arcs: an arc's
//! extremes towards each side of the box are its ends and the points of its circle square to
//! that side, and an arc that ends outside the box and touches it crosses one of its sides.

use std::f64::consts::TAU;

use glam::DVec2;

use crate::fit2d::unwind_angle;
use crate::sketch::{arc_point, Entity, EntityId, Shape, Sketch};

/// The pieces the box takes, by id, in the sketch's order.
pub fn pieces_in_box(sketch: &Sketch, corners: &[DVec2], crossing: bool) -> Vec<EntityId> {
    let Some(b) = convex(corners) else { return Vec::new() };
    sketch.entities.iter().filter(|e| if crossing { touches(e, &b) } else { inside(e, &b) }).map(|e| e.id).collect()
}

/// One side of the box: its inward unit normal and offset.
#[derive(Debug, Clone, Copy)]
struct Side {
    n: DVec2,
    d: f64,
}

/// A convex polygon turned counter-clockwise, with the inward normal of each side and its
/// offset.
struct SelectBox {
    corners: Vec<DVec2>,
    sides: Vec<Side>,
}

/// The box, or `None` for one with no area.
fn convex(corners: &[DVec2]) -> Option<SelectBox> {
    if corners.len() < 3 {
        return None;
    }
    let mut area = 0.0;
    for i in 0..corners.len() {
        let p = corners[i];
        let q = corners[(i + 1) % corners.len()];
        area += p.x * q.y - q.x * p.y;
    }
    if !(area.abs() > 1e-12) {
        return None;
    }
    // Counter-clockwise, so the inward normal is the left of every side.
    let ring: Vec<DVec2> = if area > 0.0 { corners.to_vec() } else { corners.iter().rev().copied().collect() };
    let sides = (0..ring.len())
        .map(|i| {
            let p = ring[i];
            let q = ring[(i + 1) % ring.len()];
            let n = DVec2::new(-(q.y - p.y), q.x - p.x);
            let l = n.x.hypot(n.y);
            let unit = n / l;
            Side { n: unit, d: unit.x * p.x + unit.y * p.y }
        })
        .collect();
    Some(SelectBox { corners: ring, sides })
}

/// A hair of slack, so a piece drawn on the box's edge counts as in it.
const EPS: f64 = 1e-9;

fn contains(b: &SelectBox, p: DVec2) -> bool {
    b.sides.iter().all(|s| s.n.x * p.x + s.n.y * p.y >= s.d - EPS)
}

/// An arc or a circle as a start and a sweep running counter-clockwise.
#[derive(Debug, Clone, Copy)]
struct Round {
    c: DVec2,
    r: f64,
    start: f64,
    sweep: f64,
}

fn round(e: &Entity) -> Option<Round> {
    match e.shape {
        Shape::Circle { c, r } => Some(Round { c, r, start: 0.0, sweep: TAU }),
        Shape::Arc { c, r, start, sweep } => Some(if sweep >= 0.0 { Round { c, r, start, sweep } } else { Round { c, r, start: start + sweep, sweep: -sweep } }),
        _ => None,
    }
}

fn on_sweep(a: &Round, angle: f64) -> bool {
    a.sweep >= TAU || unwind_angle(angle, a.start) - a.start <= a.sweep + EPS
}

fn inside(e: &Entity, b: &SelectBox) -> bool {
    if let Shape::Spline(_) = e.shape {
        // Finely sampled: a cubic bulges past its fit points by less than a sample's sag.
        return crate::sketch::polyline(&e.shape, 192).iter().all(|p| contains(b, *p));
    }
    match e.shape {
        Shape::Spline(_) => false,
        Shape::Point { p } => contains(b, p),
        Shape::Line { a, b: q } => contains(b, a) && contains(b, q),
        Shape::Arc { .. } | Shape::Circle { .. } => {
            let Some(a) = round(e) else { return false };
            // Each side in turn: the arc's nearest approach to it is one of its ends, or the
            // point of its circle that lies furthest out that way.
            let ends: Vec<DVec2> = if a.sweep >= TAU { Vec::new() } else { vec![arc_point(a.c, a.r, a.start), arc_point(a.c, a.r, a.start + a.sweep)] };
            b.sides.iter().all(|s| {
                let reach = |p: DVec2| s.n.x * p.x + s.n.y * p.y >= s.d - EPS;
                if !ends.iter().all(|p| reach(*p)) {
                    return false;
                }
                let out = (-s.n.y).atan2(-s.n.x);
                !on_sweep(&a, out) || reach(arc_point(a.c, a.r, out))
            })
        }
    }
}

fn touches(e: &Entity, b: &SelectBox) -> bool {
    if let Shape::Spline(_) = e.shape {
        let pts = crate::sketch::polyline(&e.shape, 192);
        if pts.iter().any(|p| contains(b, *p)) {
            return true;
        }
        return side_crossings(b).iter().any(|(p, q)| pts.windows(2).any(|w| segments_meet(w[0], w[1], *p, *q)));
    }
    match e.shape {
        Shape::Spline(_) => false,
        Shape::Point { p } => contains(b, p),
        Shape::Line { a, b: q } => contains(b, a) || contains(b, q) || side_crossings(b).iter().any(|(p0, p1)| segments_meet(a, q, *p0, *p1)),
        Shape::Arc { .. } | Shape::Circle { .. } => {
            let Some(a) = round(e) else { return false };
            // A piece with a point in the box touches it; one with none in it and still
            // touching it crosses a side.
            if contains(b, arc_point(a.c, a.r, a.start)) {
                return true;
            }
            if a.sweep < TAU && contains(b, arc_point(a.c, a.r, a.start + a.sweep)) {
                return true;
            }
            side_crossings(b).iter().any(|(p, q)| arc_meets_segment(&a, *p, *q))
        }
    }
}

fn side_crossings(b: &SelectBox) -> Vec<(DVec2, DVec2)> {
    (0..b.corners.len()).map(|i| (b.corners[i], b.corners[(i + 1) % b.corners.len()])).collect()
}

fn cross(o: DVec2, p: DVec2, q: DVec2) -> f64 {
    (p.x - o.x) * (q.y - o.y) - (p.y - o.y) * (q.x - o.x)
}

/// Whether two segments meet, ends and touching included.
fn segments_meet(a: DVec2, b: DVec2, p: DVec2, q: DVec2) -> bool {
    let d1 = cross(p, q, a);
    let d2 = cross(p, q, b);
    let d3 = cross(a, b, p);
    let d4 = cross(a, b, q);
    if ((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0)) && ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0)) {
        return true;
    }
    let within = |o: DVec2, s: DVec2, t: DVec2| o.x.min(s.x) - EPS <= t.x && t.x <= o.x.max(s.x) + EPS && o.y.min(s.y) - EPS <= t.y && t.y <= o.y.max(s.y) + EPS;
    (d1.abs() <= EPS && within(p, q, a)) || (d2.abs() <= EPS && within(p, q, b)) || (d3.abs() <= EPS && within(a, b, p)) || (d4.abs() <= EPS && within(a, b, q))
}

/// Whether an arc meets a segment: where the segment cuts the arc's circle, at a point the arc
/// runs through.
fn arc_meets_segment(a: &Round, p: DVec2, q: DVec2) -> bool {
    let d = q - p;
    let f = p - a.c;
    let aa = d.x * d.x + d.y * d.y;
    if aa == 0.0 {
        return false;
    }
    let bb = 2.0 * (f.x * d.x + f.y * d.y);
    let cc = f.x * f.x + f.y * f.y - a.r * a.r;
    let disc = bb * bb - 4.0 * aa * cc;
    if disc < 0.0 {
        return false;
    }
    let root = disc.sqrt();
    for t in [(-bb - root) / (2.0 * aa), (-bb + root) / (2.0 * aa)] {
        if t < -EPS || t > 1.0 + EPS {
            continue;
        }
        let at = DVec2::new(p.x + d.x * t, p.y + d.y * t);
        if on_sweep(a, (at.y - a.c.y).atan2(at.x - a.c.x)) {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn rect(x0: f64, y0: f64, x1: f64, y1: f64) -> Vec<DVec2> {
        vec![DVec2::new(x0, y0), DVec2::new(x1, y0), DVec2::new(x1, y1), DVec2::new(x0, y1)]
    }

    fn diamond(k: f64) -> Vec<DVec2> {
        vec![DVec2::new(5.0, 5.0 - k), DVec2::new(5.0 + k, 5.0), DVec2::new(5.0, 5.0 + k), DVec2::new(5.0 - k, 5.0)]
    }

    struct Scene {
        sketch: Sketch,
        inner: EntityId,
        across: EntityId,
        through: EntityId,
        away: EntityId,
        bulge: EntityId,
        reach: EntityId,
        ring: EntityId,
        round: EntityId,
        dot: EntityId,
    }

    fn scene() -> Scene {
        let mut sketch = Sketch::default();
        let line = |s: &mut Sketch, a: [f64; 2], b: [f64; 2]| s.add_shape(Shape::Line { a: DVec2::from(a), b: DVec2::from(b) });
        let inner = line(&mut sketch, [1.0, 1.0], [4.0, 4.0]);
        let across = line(&mut sketch, [5.0, 5.0], [20.0, 5.0]);
        let through = line(&mut sketch, [-5.0, 8.0], [15.0, 8.0]);
        let away = line(&mut sketch, [20.0, 20.0], [30.0, 20.0]);
        // Ends at (2, 2) and (8, 2), bulging down to y = -1: out of a window from y = 0,
        // wholly in one from y = -2.
        let bulge = sketch.add_shape(Shape::Arc { c: DVec2::new(5.0, 2.0), r: 3.0, start: PI, sweep: PI });
        // Ends far outside the box, its top reaching in at (5, 11.5).
        let reach = sketch.add_shape(Shape::Arc { c: DVec2::new(5.0, 20.0), r: 8.5, start: PI + 0.3, sweep: PI - 0.6 });
        let ring = sketch.add_shape(Shape::Circle { c: DVec2::new(5.0, 5.0), r: 2.0 });
        let round = sketch.add_shape(Shape::Circle { c: DVec2::new(5.0, 5.0), r: 50.0 });
        let dot = sketch.add_shape(Shape::Point { p: DVec2::new(9.0, 9.0) });
        Scene { sketch, inner, across, through, away, bulge, reach, ring, round, dot }
    }

    #[test]
    fn dragged_rightward_a_window_takes_only_what_lies_wholly_inside() {
        let s = scene();
        assert_eq!(pieces_in_box(&s.sketch, &rect(0.0, 0.0, 10.0, 12.0), false), vec![s.inner, s.ring, s.dot]);
        assert_eq!(pieces_in_box(&s.sketch, &rect(0.0, -2.0, 10.0, 12.0), false), vec![s.inner, s.bulge, s.ring, s.dot]);
    }

    #[test]
    fn dragged_leftward_a_crossing_box_takes_whatever_it_touches() {
        let s = scene();
        assert_eq!(pieces_in_box(&s.sketch, &rect(0.0, 0.0, 10.0, 12.0), true), vec![s.inner, s.across, s.through, s.bulge, s.reach, s.ring, s.dot]);
    }

    #[test]
    fn leaves_a_circle_the_box_lies_inside_of_which_it_does_not_touch() {
        let s = scene();
        assert!(!pieces_in_box(&s.sketch, &rect(0.0, 0.0, 10.0, 12.0), true).contains(&s.round));
        assert!(pieces_in_box(&s.sketch, &rect(-60.0, -60.0, 60.0, 60.0), false).contains(&s.round));
    }

    #[test]
    fn reads_a_turned_box_the_same_whichever_way_round_its_corners_run() {
        let s = scene();
        assert_eq!(pieces_in_box(&s.sketch, &rect(2.5, 2.5, 7.5, 7.5), false), vec![s.ring]);
        assert_eq!(pieces_in_box(&s.sketch, &diamond(2.5), false), Vec::<EntityId>::new());
        assert_eq!(pieces_in_box(&s.sketch, &diamond(3.0), false), vec![s.ring]);
        let reversed: Vec<DVec2> = diamond(3.0).into_iter().rev().collect();
        assert_eq!(pieces_in_box(&s.sketch, &reversed, false), vec![s.ring]);
        // Reaching 8.5 it holds the inner line, the bulge and the point too.
        assert_eq!(pieces_in_box(&s.sketch, &diamond(8.5), false), vec![s.inner, s.bulge, s.ring, s.dot]);
        // The inner line's end (4, 4) lies in the small diamond, the start of the line across
        // on its tip; the bulge stays below it.
        assert_eq!(pieces_in_box(&s.sketch, &diamond(2.5), true), vec![s.inner, s.across, s.ring]);
        assert!(!pieces_in_box(&s.sketch, &diamond(8.5), true).contains(&s.away));
    }

    #[test]
    fn takes_nothing_for_a_box_with_no_area() {
        let s = scene();
        assert_eq!(pieces_in_box(&s.sketch, &rect(3.0, 3.0, 3.0, 9.0), true), Vec::<EntityId>::new());
    }
}
