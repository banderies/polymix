// SPDX-License-Identifier: AGPL-3.0-only
//! The sketch's Symmetric relation as CAD has it. Ported from ScanRuler
//! (`plugins/reverse/core/mirrorRelation.ts`), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only.
//!
//! Two pieces that are mirror images of each other about a line of the sketch — a centre
//! line, usually a construction one — or about one of the part's coordinate planes, on the
//! line it cuts the sheet in (`datums`). The solver's equation holds two anchors symmetric; a
//! piece is more than one anchor, so the relation between two pieces is the set of those
//! equations that pins one to the other's image: a line by both of its ends, a circle by its
//! centre, an arc by its ends and its centre. Which end mirrors which is read off where the
//! pieces lie — the pairing under which they are nearest to being mirror images already — so
//! that the relation moves them the least.

use glam::DVec2;
use serde::{Deserialize, Serialize};

use crate::datums::DatumTrace;
use crate::sketch::{anchor_point, Anchor, AnchorAt, Entity, Shape, ShapeKind};

/// A point's image in the line through `a` along the unit direction `d`.
pub fn mirror_point(p: DVec2, a: DVec2, d: DVec2) -> DVec2 {
    let v = p - a;
    let along = v.dot(d);
    DVec2::new(a.x + 2.0 * along * d.x - v.x, a.y + 2.0 * along * d.y - v.y)
}

/// What two pieces mirror about: a line of the sketch, or a coordinate plane's trace.
#[derive(Debug, Clone, Copy)]
pub enum MirrorLine<'a> {
    Piece(&'a Entity),
    Datum(&'a DatumTrace),
}

/// Two anchors held mirror images of each other.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SymmetricPair {
    pub p: Anchor,
    pub q: Anchor,
}

/// The word a piece's kind goes by in a message.
fn kind_word(k: ShapeKind) -> &'static str {
    match k {
        ShapeKind::Line => "line",
        ShapeKind::Arc => "arc",
        ShapeKind::Circle => "circle",
        ShapeKind::Point => "point",
        ShapeKind::Spline => "spline",
    }
}

/// A unit vector, or `None` under 1e-12.
fn normalize2(v: DVec2) -> Option<DVec2> {
    let l = v.x.hypot(v.y);
    (l > 1e-12).then(|| v / l)
}

/// The anchor pairs that make `a` and `b` mirror images about `axis` — a line of the sketch,
/// or a coordinate plane's trace in the sheet — or why there are none: the pieces are of
/// different kinds, the axis is one of them, or it has no length.
pub fn symmetric_pairs(a: &Entity, b: &Entity, axis: MirrorLine) -> Result<Vec<SymmetricPair>, String> {
    if let MirrorLine::Piece(axis) = axis {
        if axis.kind() != ShapeKind::Line {
            return Err("The third piece is what the two mirror about — it has to be a line.".into());
        }
        if axis.id == a.id || axis.id == b.id {
            return Err("The mirror line cannot be one of the two pieces.".into());
        }
    }
    if a.id == b.id {
        return Err("Symmetric takes two different pieces.".into());
    }
    let line = match axis {
        MirrorLine::Datum(t) => {
            let (p, d) = t.line();
            Some((p, Some(d)))
        }
        MirrorLine::Piece(e) => match e.shape {
            Shape::Line { a, b } => Some((a, normalize2(b - a))),
            _ => None,
        },
    };
    let Some((origin, Some(d))) = line else { return Err("The mirror line has no length.".into()) };
    if a.kind() != b.kind() {
        return Err(format!("A {} and a {} are not mirror images of each other — pick two pieces of one kind.", kind_word(a.kind()), kind_word(b.kind())));
    }
    let pair = |pa: AnchorAt, qa: AnchorAt| SymmetricPair { p: Anchor::new(a.id, pa), q: Anchor::new(b.id, qa) };
    if matches!(a.kind(), ShapeKind::Circle | ShapeKind::Point) {
        return Ok(vec![pair(AnchorAt::Center, AnchorAt::Center)]);
    }
    // A line or an arc: its two ends, paired the way that mirrors them best.
    let miss = |pa: AnchorAt, qa: AnchorAt| {
        let m = mirror_point(anchor_point(&a.shape, pa), origin, d) - anchor_point(&b.shape, qa);
        m.x.hypot(m.y)
    };
    let straight = miss(AnchorAt::A, AnchorAt::A) + miss(AnchorAt::B, AnchorAt::B) <= miss(AnchorAt::A, AnchorAt::B) + miss(AnchorAt::B, AnchorAt::A);
    let mut ends = if straight { vec![pair(AnchorAt::A, AnchorAt::A), pair(AnchorAt::B, AnchorAt::B)] } else { vec![pair(AnchorAt::A, AnchorAt::B), pair(AnchorAt::B, AnchorAt::A)] };
    if a.kind() == ShapeKind::Arc {
        ends.push(pair(AnchorAt::Center, AnchorAt::Center));
    }
    Ok(ends)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datums::{datum_traces, trace_of, DatumPlane};
    use crate::frame::SectionFrame;
    use crate::sketch::{MirrorAxis, Rule, Sketch};
    use crate::solver::{solve_sketch, SolveOptions};
    use glam::DVec3;

    fn line(s: &mut Sketch, a: [f64; 2], b: [f64; 2]) -> u32 {
        s.add_shape(Shape::Line { a: DVec2::from(a), b: DVec2::from(b) })
    }

    fn ent(s: &Sketch, id: u32) -> &Entity {
        s.entity(id).unwrap()
    }

    fn line_of(s: &Sketch, id: u32) -> (DVec2, DVec2) {
        match ent(s, id).shape.clone() {
            Shape::Line { a, b } => (a, b),
            other => panic!("not a line: {other:?}"),
        }
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 5e-7
    }

    #[test]
    fn reflects_across_a_line_through_a_point_along_a_direction() {
        assert_eq!(mirror_point(DVec2::new(3.0, 1.0), DVec2::ZERO, DVec2::new(0.0, 1.0)), DVec2::new(-3.0, 1.0));
        let p = mirror_point(DVec2::new(2.0, 0.0), DVec2::ZERO, DVec2::new(std::f64::consts::FRAC_1_SQRT_2, std::f64::consts::FRAC_1_SQRT_2));
        assert!(p.x.abs() < 5e-10 && (p.y - 2.0).abs() < 5e-10);
    }

    #[test]
    fn pairs_the_ends_of_two_lines_the_way_they_mirror_whichever_way_each_was_drawn() {
        let mut s = Sketch::default();
        let axis = s.add(Entity::new(Shape::Line { a: DVec2::new(0.0, -10.0), b: DVec2::new(0.0, 10.0) }).construction());
        let left = line(&mut s, [-5.0, 0.0], [-3.0, 4.0]);
        let same = line(&mut s, [5.2, 0.1], [3.1, 3.9]);
        let turned = line(&mut s, [3.1, 3.9], [5.2, 0.1]);
        let pair = |p: u32, pa: AnchorAt, q: u32, qa: AnchorAt| SymmetricPair { p: Anchor::new(p, pa), q: Anchor::new(q, qa) };
        assert_eq!(
            symmetric_pairs(ent(&s, left), ent(&s, same), MirrorLine::Piece(ent(&s, axis))),
            Ok(vec![pair(left, AnchorAt::A, same, AnchorAt::A), pair(left, AnchorAt::B, same, AnchorAt::B)])
        );
        assert_eq!(
            symmetric_pairs(ent(&s, left), ent(&s, turned), MirrorLine::Piece(ent(&s, axis))),
            Ok(vec![pair(left, AnchorAt::A, turned, AnchorAt::B), pair(left, AnchorAt::B, turned, AnchorAt::A)])
        );
    }

    #[test]
    fn holds_circles_by_their_centres_and_says_why_it_refuses_the_rest() {
        let mut s = Sketch::default();
        let axis = line(&mut s, [0.0, 0.0], [10.0, 0.0]);
        let c1 = s.add_shape(Shape::Circle { c: DVec2::new(2.0, 3.0), r: 1.0 });
        let c2 = s.add_shape(Shape::Circle { c: DVec2::new(2.0, -3.0), r: 1.0 });
        let l = line(&mut s, [1.0, 1.0], [2.0, 2.0]);
        assert_eq!(symmetric_pairs(ent(&s, c1), ent(&s, c2), MirrorLine::Piece(ent(&s, axis))), Ok(vec![SymmetricPair { p: Anchor::new(c1, AnchorAt::Center), q: Anchor::new(c2, AnchorAt::Center) }]));
        assert!(symmetric_pairs(ent(&s, c1), ent(&s, l), MirrorLine::Piece(ent(&s, axis))).unwrap_err().contains("one kind"));
        assert!(symmetric_pairs(ent(&s, c1), ent(&s, c2), MirrorLine::Piece(ent(&s, c1))).unwrap_err().contains("has to be a line"));
        assert!(symmetric_pairs(ent(&s, l), ent(&s, axis), MirrorLine::Piece(ent(&s, axis))).unwrap_err().contains("cannot be one of the two"));
    }

    #[test]
    fn is_what_the_solver_needs_to_make_one_line_the_mirror_image_of_the_other() {
        let mut s = Sketch::default();
        let mut axis_e = Entity::new(Shape::Line { a: DVec2::new(0.0, -10.0), b: DVec2::new(0.0, 10.0) }).construction();
        axis_e.fixed = true;
        let axis = s.add(axis_e);
        let mut left_e = Entity::new(Shape::Line { a: DVec2::new(-5.0, 0.0), b: DVec2::new(-3.0, 4.0) });
        left_e.fixed = true;
        let left = s.add(left_e);
        let right = line(&mut s, [6.0, 1.0], [2.0, 5.0]);
        let pairs = symmetric_pairs(ent(&s, left), ent(&s, right), MirrorLine::Piece(ent(&s, axis))).unwrap();
        for p in pairs {
            s.constrain(Rule::Symmetric { p: p.p, q: p.q, about: MirrorAxis::Line(axis) });
        }
        let report = solve_sketch(&mut s, SolveOptions::default());
        assert!(report.ok);
        let (a, b) = line_of(&s, right);
        assert!(close(a.x, 5.0) && close(a.y, 0.0) && close(b.x, 3.0) && close(b.y, 4.0), "{a} {b}");
    }

    #[test]
    fn takes_a_coordinate_planes_trace_as_the_mirror_line_and_the_solver_holds_it_there_from_the_frame() {
        // A sketch on the XY plane: the YZ plane cuts it in the line u = 0.
        let frame = SectionFrame { origin: DVec3::ZERO, normal: DVec3::Z, u: DVec3::X, v: DVec3::Y };
        let traces = datum_traces(&frame);
        let yz = trace_of(&traces, DatumPlane::Yz).unwrap();
        let mut s = Sketch::default();
        let mut left_e = Entity::new(Shape::Line { a: DVec2::new(-5.0, 0.0), b: DVec2::new(-3.0, 4.0) });
        left_e.fixed = true;
        let left = s.add(left_e);
        let right = line(&mut s, [2.0, 5.0], [6.0, 1.0]);
        let pairs = symmetric_pairs(ent(&s, left), ent(&s, right), MirrorLine::Datum(&yz)).unwrap();
        assert_eq!(pairs, vec![SymmetricPair { p: Anchor::new(left, AnchorAt::A), q: Anchor::new(right, AnchorAt::B) }, SymmetricPair { p: Anchor::new(left, AnchorAt::B), q: Anchor::new(right, AnchorAt::A) }]);
        for p in pairs {
            s.constrain(Rule::Symmetric { p: p.p, q: p.q, about: MirrorAxis::Datum(DatumPlane::Yz) });
        }
        // The trace is the solver's to be given; without it the relation holds nothing.
        let loose = solve_sketch(&mut s, SolveOptions::default());
        assert!(loose.ok);
        assert!(close(line_of(&s, right).1.x, 6.0));
        let report = solve_sketch(&mut s, SolveOptions { datums: &traces, ..Default::default() });
        assert!(report.ok);
        let (a, b) = line_of(&s, right);
        assert!(close(b.x, 5.0) && close(b.y, 0.0) && close(a.x, 3.0) && close(a.y, 4.0), "{a} {b}");
    }
}
