// SPDX-License-Identifier: AGPL-3.0-only
//! A sketch dimension as a drawing carries it, the way Onshape and Fusion put it on the sheet.
//! Ported from ScanRuler (`plugins/reverse/core/dimLayout.ts`), © Stefan Hermann
//! (CNC Kitchen), AGPL-3.0-only.
//!
//! Extension lines off the two measured points, a dimension line between them with a filled
//! arrow at each end, the number written along the line in a gap left for it; a radius as a
//! leader with its arrow on the rim; a diameter as a line through the centre, an arrow on the
//! rim at each end; an angle as an arc between the two lines. The number is where the operator
//! dragged it (`Constraint::place`), or where the layout puts it.
//!
//! Two steps, both pure. `dim_shape` reads the sketch: what is measured and where the number
//! sits, in sheet millimetres. `dim_strokes` (in `crate::dim_strokes`) turns that into lines
//! and arrowheads for one zoom — the arrows, the gaps and the overshoot are sized in screen
//! pixels, so the strokes are made again when the zoom changes and the shape is not.

use std::f64::consts::PI;

use glam::DVec2;

pub use crate::dim_strokes::{dim_strokes, ArcSpan, DimShape, DimStrokes};

use crate::datums::{trace_of, DatumTrace};
use crate::sketch::{anchor_point, Constraint, DimPlace, DistanceTo, EntityId, Rule, Shape, Sketch};

/// A unit vector, or `None` under 1e-12.
fn normalize2(v: DVec2) -> Option<DVec2> {
    let l = v.x.hypot(v.y);
    (l > 1e-12).then(|| v / l)
}

/// The left-hand perpendicular — a quarter turn counter-clockwise.
fn perp2(a: DVec2) -> DVec2 {
    DVec2::new(-a.y, a.x)
}

fn line_of(sketch: &Sketch, id: EntityId) -> Option<(DVec2, DVec2)> {
    match sketch.entity(id)?.shape {
        Shape::Line { a, b } => Some((a, b)),
        _ => None,
    }
}

/// The two points a length or a distance is measured between.
fn measured_points(rule: &Rule, sketch: &Sketch, datums: &[DatumTrace]) -> Option<(DVec2, DVec2)> {
    match *rule {
        Rule::Length { entity, .. } => line_of(sketch, entity),
        Rule::Distance { p, q, .. } => {
            let a = anchor_point(&sketch.entity(p.entity)?.shape, p.at);
            match q {
                DistanceTo::Datum(plane) => {
                    let trace = trace_of(datums, plane)?;
                    Some((a, trace.foot(a)))
                }
                DistanceTo::Line(l) => {
                    let (la, lb) = line_of(sketch, l)?;
                    let n = perp2(normalize2(lb - la).unwrap_or(DVec2::X));
                    Some((a, a - n * (a - la).dot(n)))
                }
                DistanceTo::Anchor(q) => Some((a, anchor_point(&sketch.entity(q.entity)?.shape, q.at))),
            }
        }
        _ => None,
    }
}

/// The angle's arc as it stands: from the first line's direction counter-clockwise by the
/// angle the solver reads, 0–180°, which ends on the second line's carrier — so the arc drawn
/// is the angle the number says.
struct AngleSector {
    vertex: DVec2,
    a0: f64,
    sweep: f64,
    lines: [[DVec2; 2]; 2],
    shorter: f64,
}

fn angle_sector(a: EntityId, b: EntityId, sketch: &Sketch) -> Option<AngleSector> {
    let (a_a, a_b) = line_of(sketch, a)?;
    let (b_a, b_b) = line_of(sketch, b)?;
    let da = normalize2(a_b - a_a)?;
    let db = normalize2(b_b - b_a)?;
    let cross = da.x * db.y - da.y * db.x;
    // The rays meet where the carriers cross, or at the first line's middle.
    let mut vertex = (a_a + a_b) / 2.0;
    if cross.abs() > 1e-9 {
        let w = b_a - a_a;
        vertex = a_a + da * ((w.x * db.y - w.y * db.x) / cross);
    }
    let turn = cross.atan2(da.dot(db));
    let sweep = ((turn % PI) + PI) % PI;
    Some(AngleSector {
        vertex,
        a0: da.y.atan2(da.x),
        sweep,
        lines: [[a_a, a_b], [b_a, b_b]],
        shorter: (a_b - a_a).x.hypot((a_b - a_a).y).min((b_b - b_a).x.hypot((b_b - b_a).y)),
    })
}

/// What a dimension measures and where its number sits. `lift` is how far the layout stands a
/// number off its piece when the operator has not placed it — sheet millimetres, a few percent
/// of the sketch. `None` when what the dimension hangs on is gone, or for a relation that is
/// no dimension.
pub fn dim_shape(c: &Constraint, sketch: &Sketch, datums: &[DatumTrace], lift: f64) -> Option<DimShape> {
    let place: Option<DimPlace> = c.place;
    match c.rule {
        Rule::Diameter { entity, .. } => {
            let (cc, r, angle) = match sketch.entity(entity)?.shape {
                Shape::Arc { c, r, start, sweep } => (c, r, start + sweep / 2.0),
                Shape::Circle { c, r } => (c, r, PI / 4.0),
                _ => return None,
            };
            Some(DimShape::Diameter { c: cc, r, angle: place.map_or(angle, |p| p.off), reach: place.map_or(r + 2.0 * lift, |p| p.along.max(0.0)) })
        }
        Rule::Radius { entity, .. } => {
            let (cc, r, span) = match sketch.entity(entity)?.shape {
                Shape::Arc { c, r, start, sweep } => (c, r, Some(ArcSpan { start, sweep })),
                Shape::Circle { c, r } => (c, r, None),
                _ => return None,
            };
            Some(DimShape::Radius {
                c: cc,
                r,
                angle: place.map_or_else(|| span.map_or(PI / 4.0, |s| s.start + s.sweep / 2.0), |p| p.off),
                reach: place.map_or(r + 2.0 * lift, |p| p.along.max(0.0)),
                span,
            })
        }
        Rule::Angle { a, b, .. } => {
            let s = angle_sector(a, b, sketch)?;
            let radius = match place {
                Some(p) => p.off.max(1e-6),
                None => lift.max((0.35 * s.shorter).min(4.0 * lift)),
            };
            Some(DimShape::Angle { vertex: s.vertex, a0: s.a0, sweep: s.sweep, radius, at: place.map_or(s.a0 + s.sweep / 2.0, |p| p.along), lines: s.lines })
        }
        Rule::Length { .. } | Rule::Distance { .. } => {
            let (p, q) = measured_points(&c.rule, sketch, datums)?;
            let d = normalize2(q - p)?;
            let n = perp2(d);
            // A line's length stands off the line, on the side away from the rest of the
            // sketch; a distance is drawn where it is measured until it is dragged aside.
            let mut off = 0.0;
            if let Some(place) = place {
                off = place.off;
            } else if matches!(c.rule, Rule::Length { .. }) {
                let middle = sketch.bounds().map_or(p, |(lo, hi)| (lo + hi) / 2.0);
                off = if (middle - p).dot(n) > 1e-9 { -lift } else { lift };
            }
            Some(DimShape::Linear { p, q, d, n, off, along: place.map_or(0.0, |p| p.along) })
        }
        _ => None,
    }
}

/// The place a number dragged to the sheet point `at` takes.
pub fn dim_place_at(c: &Constraint, sketch: &Sketch, at: DVec2, datums: &[DatumTrace]) -> Option<DimPlace> {
    let shape = dim_shape(c, sketch, datums, 1.0)?;
    let centre = match shape {
        DimShape::Linear { p, q, d, n, .. } => {
            let mid = (p + q) / 2.0;
            let w = at - mid;
            return Some(DimPlace { off: w.dot(n), along: w.dot(d) });
        }
        DimShape::Angle { vertex, .. } => vertex,
        DimShape::Radius { c, .. } | DimShape::Diameter { c, .. } => c,
    };
    let w = at - centre;
    let reach = w.x.hypot(w.y);
    if !(reach > 1e-9) {
        return None;
    }
    let angle = w.y.atan2(w.x);
    Some(match shape {
        DimShape::Angle { .. } => DimPlace { off: reach, along: angle },
        _ => DimPlace { off: angle, along: reach },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sketch::ConstraintId;

    fn dimension(s: &Sketch, id: ConstraintId) -> Constraint {
        s.constraint(id).cloned().unwrap()
    }

    fn placed(c: &Constraint, place: DimPlace) -> Constraint {
        Constraint { place: Some(place), ..c.clone() }
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 5e-3
    }

    fn line(s: &mut Sketch, a: [f64; 2], b: [f64; 2]) -> EntityId {
        s.add_shape(Shape::Line { a: DVec2::from(a), b: DVec2::from(b) })
    }

    #[test]
    fn stands_a_length_off_its_line_with_two_extension_lines_two_arrows_inside_and_a_gap() {
        let mut s = Sketch::default();
        let l = line(&mut s, [0.0, 0.0], [20.0, 0.0]);
        let id = s.constrain(Rule::Length { entity: l, value: 20.0 });
        let shape = dim_shape(&dimension(&s, id), &s, &[], 3.0).unwrap();
        let DimShape::Linear { off, .. } = shape else { panic!("not linear: {shape:?}") };
        assert!(close(off.abs(), 3.0));
        let st = dim_strokes(&shape, 0.05, 20.0);
        // Two extension lines and the dimension line in two parts.
        assert_eq!(st.lines.len(), 4);
        assert_eq!(st.arrows.len(), 2);
        // Inside arrows point away from each other.
        assert!(close(st.arrows[0].dir.x, -1.0));
        assert!(close(st.arrows[1].dir.x, 1.0));
        assert!(close(st.text.x, 10.0));
        assert!(close(st.text_angle, 0.0));
    }

    #[test]
    fn turns_the_arrows_in_from_outside_when_the_stretch_is_short_on_screen() {
        let mut s = Sketch::default();
        let l = line(&mut s, [0.0, 0.0], [1.0, 0.0]);
        let id = s.constrain(Rule::Length { entity: l, value: 1.0 });
        let st = dim_strokes(&dim_shape(&dimension(&s, id), &s, &[], 3.0).unwrap(), 0.1, 20.0);
        assert!(close(st.arrows[0].dir.x, 1.0));
        assert!(close(st.arrows[1].dir.x, -1.0));
    }

    #[test]
    fn keeps_the_number_readable_on_a_line_that_runs_down_the_sheet() {
        let mut s = Sketch::default();
        let l = line(&mut s, [0.0, 10.0], [-10.0, 0.0]);
        let id = s.constrain(Rule::Length { entity: l, value: 10f64.hypot(10.0) });
        let st = dim_strokes(&dim_shape(&dimension(&s, id), &s, &[], 2.0).unwrap(), 0.05, 20.0);
        assert!(close(st.text_angle, PI / 4.0));
    }

    #[test]
    fn puts_a_dragged_number_where_it_was_dropped_and_keeps_it_there_as_the_line_moves() {
        let mut s = Sketch::default();
        let l = line(&mut s, [0.0, 0.0], [20.0, 0.0]);
        let id = s.constrain(Rule::Length { entity: l, value: 20.0 });
        let c = dimension(&s, id);
        let place = dim_place_at(&c, &s, DVec2::new(14.0, -6.0), &[]).unwrap();
        let p = placed(&c, place);
        let st = dim_strokes(&dim_shape(&p, &s, &[], 3.0).unwrap(), 0.05, 20.0);
        assert!(close(st.text.x, 14.0) && close(st.text.y, -6.0));
        // The line moved up by five: the number goes with it.
        s.entity_mut(l).unwrap().shape = Shape::Line { a: DVec2::new(0.0, 5.0), b: DVec2::new(20.0, 5.0) };
        let moved = dim_strokes(&dim_shape(&p, &s, &[], 3.0).unwrap(), 0.05, 20.0);
        assert!(close(moved.text.y, -1.0));
    }

    #[test]
    fn carries_the_dimension_line_out_to_a_number_dragged_past_the_end() {
        let mut s = Sketch::default();
        let l = line(&mut s, [0.0, 0.0], [20.0, 0.0]);
        let id = s.constrain(Rule::Length { entity: l, value: 20.0 });
        let c = dimension(&s, id);
        let p = placed(&c, dim_place_at(&c, &s, DVec2::new(30.0, 4.0), &[]).unwrap());
        let st = dim_strokes(&dim_shape(&p, &s, &[], 3.0).unwrap(), 0.05, 20.0);
        let reach = st.lines.iter().flatten().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max);
        assert!(reach > 28.0 && reach < 30.0, "{reach}");
    }

    #[test]
    fn leads_a_radius_in_to_the_rim_from_a_number_outside_and_out_to_it_from_one_inside() {
        let mut s = Sketch::default();
        let circle = s.add_shape(Shape::Circle { c: DVec2::ZERO, r: 10.0 });
        let id = s.constrain(Rule::Radius { entity: circle, value: 10.0 });
        let c = dimension(&s, id);
        let outside = dim_strokes(&dim_shape(&placed(&c, dim_place_at(&c, &s, DVec2::new(20.0, 0.0), &[]).unwrap()), &s, &[], 1.0).unwrap(), 0.05, 30.0);
        assert!(close(outside.arrows[0].tip.x, 10.0));
        assert!(close(outside.arrows[0].dir.x, -1.0));
        let inside = dim_strokes(&dim_shape(&placed(&c, dim_place_at(&c, &s, DVec2::new(5.0, 0.0), &[]).unwrap()), &s, &[], 1.0).unwrap(), 0.05, 30.0);
        assert!(close(inside.arrows[0].dir.x, 1.0));
        assert!(close(inside.text.x, 5.0));
    }

    #[test]
    fn draws_a_diameter_rim_to_rim_through_the_centre_carried_out_to_a_number_outside() {
        let mut s = Sketch::default();
        let circle = s.add_shape(Shape::Circle { c: DVec2::ZERO, r: 10.0 });
        let id = s.constrain(Rule::Diameter { entity: circle, value: 20.0 });
        let c = dimension(&s, id);
        let outside = dim_strokes(&dim_shape(&placed(&c, dim_place_at(&c, &s, DVec2::new(20.0, 0.0), &[]).unwrap()), &s, &[], 1.0).unwrap(), 0.05, 30.0);
        // An arrow on the rim at each end, both pointing out at it.
        let mut tips: Vec<(f64, f64)> = outside.arrows.iter().map(|a| (a.tip.x, a.dir.x)).collect();
        tips.sort_by(|p, q| p.0.total_cmp(&q.0));
        assert!(close(tips[0].0, -10.0) && close(tips[1].0, 10.0));
        for (x, dir) in tips {
            assert_eq!(dir.signum(), x.signum());
        }
        // The line runs from the far rim on through the centre to the number.
        let xs: Vec<f64> = outside.lines.iter().flatten().map(|p| p.x).collect();
        assert!(close(xs.iter().copied().fold(f64::INFINITY, f64::min), -10.0));
        assert!(xs.iter().copied().fold(f64::NEG_INFINITY, f64::max) > 15.0);
        assert!(close(outside.text.x, 20.0));
        let inside = dim_strokes(&dim_shape(&placed(&c, dim_place_at(&c, &s, DVec2::new(0.0, 3.0), &[]).unwrap()), &s, &[], 1.0).unwrap(), 0.05, 30.0);
        assert!(close(inside.text.y, 3.0));
        let ys: Vec<f64> = inside.lines.iter().flatten().map(|p| p.y).collect();
        assert!(close(ys.iter().copied().fold(f64::NEG_INFINITY, f64::max), 10.0));
        assert!(close(ys.iter().copied().fold(f64::INFINITY, f64::min), -10.0));
    }

    #[test]
    fn draws_an_angle_as_the_arc_the_number_says_and_across_the_vertex_when_dragged_there() {
        let mut s = Sketch::default();
        let a = line(&mut s, [0.0, 0.0], [10.0, 0.0]);
        let b = line(&mut s, [0.0, 0.0], [0.0, 10.0]);
        let id = s.constrain(Rule::Angle { a, b, value: 90.0 });
        let c = dimension(&s, id);
        let shape = dim_shape(&c, &s, &[], 1.0).unwrap();
        let DimShape::Angle { sweep, .. } = shape else { panic!("not an angle: {shape:?}") };
        assert!(close(sweep, PI / 2.0));
        let st = dim_strokes(&shape, 0.02, 30.0);
        assert_eq!(st.arrows.len(), 2);
        assert!(st.text.x > 0.0 && st.text.y > 0.0);
        let across = dim_strokes(&dim_shape(&placed(&c, dim_place_at(&c, &s, DVec2::new(-4.0, -4.0), &[]).unwrap()), &s, &[], 1.0).unwrap(), 0.02, 30.0);
        assert!(close(across.text.x, -4.0) && close(across.text.y, -4.0));
        // Both arrow tips lie on the lines' carriers, on the far side.
        for arrow in &across.arrows {
            assert!(close(arrow.tip.x.abs().min(arrow.tip.y.abs()), 0.0));
            assert!(arrow.tip.x + arrow.tip.y < 0.0);
        }
    }
}
