// SPDX-License-Identifier: AGPL-3.0-only
//! What the drawing tools of the sketch work out before a piece is made: the arc that leaves a
//! chain's last end tangent to the piece before it, the dimension two clicks of the Dimension
//! tool ask for, the relation two clicks of Coincident make, and the ways a rectangle, a
//! circle and an arc are drawn. Ported from ScanRuler (`plugins/reverse/core/draw.ts`),
//! © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! No store, no sheet — the caller passes what was clicked.

use std::f64::consts::{PI, TAU};

use glam::DVec2;
use serde::{Deserialize, Serialize};

use crate::datums::{trace_near, trace_of, DatumPlane, DatumTrace};
use crate::sketch::{anchor_point, anchors_of, measure_rule, nearest_on, Anchor, AnchorAt, DistanceTo, Entity, EntityId, OnTarget, Rule, Shape, ShapeKind, Sketch};
use crate::sketch_shapes::PolygonFit;

/// A unit vector, or `None` under 1e-12.
fn normalize2(v: DVec2) -> Option<DVec2> {
    let l = v.x.hypot(v.y);
    (l > 1e-12).then(|| v / l)
}

fn perp2(a: DVec2) -> DVec2 {
    DVec2::new(-a.y, a.x)
}

fn cross2(a: DVec2, b: DVec2) -> f64 {
    a.x * b.y - a.y * b.x
}

fn len2(a: DVec2) -> f64 {
    a.x.hypot(a.y)
}

/// The direction a piece runs on in past one of its ends — what a tangent arc begun there
/// leaves along. `None` for a piece without ends.
pub fn outward_tangent(e: &Entity, at: AnchorAt) -> Option<DVec2> {
    match e.shape {
        Shape::Line { a, b } => {
            let d = normalize2(b - a)?;
            if at == AnchorAt::Center {
                return None;
            }
            Some(if at == AnchorAt::B { d } else { d * -1.0 })
        }
        Shape::Arc { start, sweep, .. } => {
            if at == AnchorAt::Center {
                return None;
            }
            // Counter-clockwise past the end, clockwise back past the start.
            let theta = if at == AnchorAt::B { start + sweep } else { start };
            let ccw = DVec2::new(-theta.sin(), theta.cos());
            Some(if at == AnchorAt::B { ccw } else { ccw * -1.0 })
        }
        _ => None,
    }
}

/// An arc as a piece takes it, and which of its anchors lies at the first point given:
/// `starts_at_p` says whether anchor `A` lies at `p` (the arc runs counter-clockwise from it)
/// or at the other end.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ArcPlacement {
    pub c: DVec2,
    pub r: f64,
    pub start: f64,
    pub sweep: f64,
    pub starts_at_p: bool,
}

impl ArcPlacement {
    pub fn shape(&self) -> Shape {
        Shape::Arc { c: self.c, r: self.r, start: self.start, sweep: self.sweep }
    }
}

fn ccw(from: f64, to: f64) -> f64 {
    ((to - from) % TAU + TAU) % TAU
}

/// The arc from `p` to `q` that leaves `p` along the unit direction `t` — the one a line tool
/// draws when the hand presses on the chain's last end and drags. `None` when `q` lies
/// straight ahead or on `p`, where no arc is tangent.
pub fn tangent_arc(p: DVec2, t: DVec2, q: DVec2) -> Option<ArcPlacement> {
    let pq = q - p;
    let l = len2(pq);
    if !(l > 1e-9) {
        return None;
    }
    let n = perp2(t);
    let h = pq.dot(n);
    if h.abs() < 1e-6 * l {
        return None;
    }
    // The centre lies square to the tangent, as far off as puts q on the rim.
    let signed = (l * l) / (2.0 * h);
    let c = p + n * signed;
    let theta_p = (p.y - c.y).atan2(p.x - c.x);
    let theta_q = (q.y - c.y).atan2(q.x - c.x);
    // Centre on the left: the hand goes round counter-clockwise from p.
    Some(if signed > 0.0 {
        ArcPlacement { c, r: signed, start: theta_p, sweep: ccw(theta_p, theta_q), starts_at_p: true }
    } else {
        ArcPlacement { c, r: -signed, start: theta_q, sweep: ccw(theta_q, theta_p), starts_at_p: false }
    })
}

/// What the Dimension tool has under a click: a whole piece, a point — an end of a line or an
/// arc, a point piece — or one of the part's coordinate planes, clicked on the line it cuts
/// the sheet in.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DimPick {
    Piece(EntityId),
    Point(Anchor),
    Datum(DatumPlane),
}

/// The pick at a spot on the sheet: an end within reach wins over the piece it ends, as the
/// smaller target; a centre is taken through its piece; a coordinate plane's line is taken
/// last, where nothing of the sketch is.
pub fn dim_pick_at(sketch: &Sketch, spot: DVec2, radius: f64, datums: &[DatumTrace]) -> Option<DimPick> {
    let mut point: Option<Anchor> = None;
    let mut point_d = radius * 0.75;
    for e in &sketch.entities {
        if e.kind() == ShapeKind::Circle {
            continue;
        }
        for &at in anchors_of(&e.shape) {
            let d = len2(anchor_point(&e.shape, at) - spot);
            if d <= point_d {
                point_d = d;
                point = Some(Anchor::new(e.id, at));
            }
        }
    }
    if let Some(anchor) = point {
        return Some(DimPick::Point(anchor));
    }
    let mut piece: Option<EntityId> = None;
    let mut piece_d = radius;
    for e in &sketch.entities {
        let (_, dist) = nearest_on(&e.shape, spot);
        if dist <= piece_d {
            piece_d = dist;
            piece = Some(e.id);
        }
    }
    if let Some(id) = piece {
        return Some(DimPick::Piece(id));
    }
    trace_near(datums, spot, radius).map(|t| DimPick::Datum(t.plane))
}

/// What the Coincident relation has under a click: as the Dimension tool's pick, with a
/// circle's centre a point too — a centre is what a circle is held by.
pub fn rel_pick_at(sketch: &Sketch, spot: DVec2, radius: f64, datums: &[DatumTrace]) -> Option<DimPick> {
    let mut centre: Option<Anchor> = None;
    let mut centre_d = radius * 0.75;
    for e in &sketch.entities {
        let Shape::Circle { c, .. } = e.shape else { continue };
        let d = len2(c - spot);
        if d <= centre_d {
            centre_d = d;
            centre = Some(Anchor::new(e.id, AnchorAt::Center));
        }
    }
    let pick = dim_pick_at(sketch, spot, radius, datums);
    // A centre within reach wins over the rim it is inside of, as an end wins over its line.
    if let Some(centre) = centre {
        if !matches!(pick, Some(DimPick::Point(_))) {
            return Some(DimPick::Point(centre));
        }
    }
    pick
}

/// The relation two picks make coincident, as CAD reads the pair: two points meet; a point
/// lies on a line, on a round piece's rim or on a coordinate plane's line; a line lies along
/// another line or along a plane's line. A pair that makes none — two round pieces, two
/// planes, a round piece and a line — says why in a sentence.
pub fn coincident_for(sketch: &Sketch, first: DimPick, second: DimPick) -> Result<Rule, String> {
    let piece = |p: DimPick| match p {
        DimPick::Piece(id) => sketch.entity(id),
        _ => None,
    };
    let round = |e: Option<&Entity>| e.is_some_and(|e| e.shape.is_round());
    let same = |a: DimPick, b: DimPick| match (a, b) {
        (DimPick::Point(p), DimPick::Point(q)) => p.entity == q.entity,
        (DimPick::Piece(p), DimPick::Piece(q)) => p == q,
        (DimPick::Point(p), DimPick::Piece(q)) | (DimPick::Piece(q), DimPick::Point(p)) => p.entity == q,
        (DimPick::Datum(p), DimPick::Datum(q)) => p == q,
        _ => false,
    };
    if same(first, second) {
        return Err("Pick two different things to make coincident.".into());
    }
    // A point pick, whichever came first, and what it lies on.
    let point = match (first, second) {
        (DimPick::Point(p), other) => Some((p, other)),
        (other, DimPick::Point(p)) => Some((p, other)),
        _ => None,
    };
    if let Some((anchor, other)) = point {
        return match other {
            DimPick::Point(q) => Ok(Rule::Coincident { p: anchor, q }),
            DimPick::Datum(plane) => Ok(Rule::PointOn { p: anchor, on: OnTarget::Datum(plane), bounded: false }),
            DimPick::Piece(_) => {
                let Some(e) = piece(other) else { return Err("That piece is gone.".into()) };
                match e.shape {
                    Shape::Line { .. } => Ok(Rule::PointOn { p: anchor, on: OnTarget::Line(e.id), bounded: false }),
                    Shape::Arc { .. } | Shape::Circle { .. } | Shape::Spline(_) => Ok(Rule::PointOn { p: anchor, on: OnTarget::Curve(e.id), bounded: false }),
                    // A point piece: its one point.
                    Shape::Point { .. } => Ok(Rule::Coincident { p: anchor, q: Anchor::new(e.id, AnchorAt::Center) }),
                }
            }
        };
    }
    if matches!((first, second), (DimPick::Datum(_), DimPick::Datum(_))) {
        return Err("Two coordinate planes cannot be made coincident.".into());
    }
    // Two pieces, or a piece and a plane.
    let a = piece(first);
    let b = piece(second);
    let is_point = |e: Option<&Entity>| e.is_some_and(|e| e.kind() == ShapeKind::Point);
    if is_point(a) || is_point(b) {
        let (pt, onto) = if is_point(a) { (a, second) } else { (b, first) };
        let Some(pt) = pt else { return Err("That piece is gone.".into()) };
        return coincident_for(sketch, DimPick::Point(Anchor::new(pt.id, AnchorAt::Center)), onto);
    }
    let line = if a.is_some_and(|e| e.kind() == ShapeKind::Line) {
        a
    } else if b.is_some_and(|e| e.kind() == ShapeKind::Line) {
        b
    } else {
        None
    };
    let Some(line) = line else {
        return Err(if round(a) && round(b) { "Two round pieces are held by Tangent or Concentric, not Coincident.".into() } else { "Coincident takes a point, a line, or a coordinate plane’s line.".into() });
    };
    // The other pick: a plane, or the other piece.
    let line_is_a = a.is_some_and(|e| e.id == line.id) && matches!(first, DimPick::Piece(id) if id == line.id);
    let rest = if line_is_a { second } else { first };
    match rest {
        DimPick::Datum(plane) => Ok(Rule::Collinear { line: line.id, on: OnTarget::Datum(plane) }),
        DimPick::Piece(_) => {
            let Some(e) = piece(rest) else { return Err("That piece is gone.".into()) };
            match e.shape {
                Shape::Line { .. } => Ok(Rule::Collinear { line: line.id, on: OnTarget::Line(e.id) }),
                Shape::Arc { .. } | Shape::Circle { .. } => Err("A line and a round piece are held by Tangent, not Coincident — or pick an end of the line.".into()),
                Shape::Point { .. } => Err("Coincident takes a point, a line, or a coordinate plane’s line.".into()),
                Shape::Spline(_) => Err("A spline is joined by its ends or its fit points — pick one of those, then what it lies on.".into()),
            }
        }
        DimPick::Point(_) => Err("Coincident takes a point, a line, or a coordinate plane’s line.".into()),
    }
}

/// Two lines closer to parallel than this are dimensioned by the distance between them, not
/// by an angle of next to nothing.
const PARALLEL_DEG: f64 = 2.0;

fn measured(sketch: &Sketch, rule: Rule, datums: &[DatumTrace]) -> Option<f64> {
    measure_rule(sketch, &rule, datums)
}

/// The dimension the picks ask for, at the value measured now. One pick: a line's length, an
/// arc's radius, a circle's diameter. Two: between parallel lines their distance, between
/// other lines their angle, from a point or a round piece's centre to a line the perpendicular
/// distance, between two points or centres the distance; from a point, a centre or a line
/// lying parallel to it to a coordinate plane the distance to the plane. `None` when nothing
/// applies — a point alone, a plane alone.
pub fn dimension_for(sketch: &Sketch, first: DimPick, second: Option<DimPick>, datums: &[DatumTrace]) -> Option<Rule> {
    let piece = |p: DimPick| match p {
        DimPick::Piece(id) => sketch.entity(id),
        _ => None,
    };
    if matches!(first, DimPick::Datum(_)) || matches!(second, Some(DimPick::Datum(_))) {
        let (plane, other) = match (first, second) {
            (DimPick::Datum(plane), other) => (plane, other?),
            (other, Some(DimPick::Datum(plane))) => (plane, other),
            _ => return None,
        };
        if matches!(other, DimPick::Datum(_)) {
            return None;
        }
        let trace = trace_of(datums, plane)?;
        let p = match other {
            DimPick::Point(anchor) => anchor,
            _ => {
                let e = piece(other)?;
                match e.shape {
                    // A line is a distance from the plane only where it runs along it.
                    Shape::Line { a, b } => {
                        let d = normalize2(b - a)?;
                        if d.dot(trace.n).abs() > (PARALLEL_DEG * PI / 180.0).sin() {
                            return None;
                        }
                        Anchor::new(e.id, AnchorAt::A)
                    }
                    _ => Anchor::new(e.id, AnchorAt::Center),
                }
            }
        };
        let v = measured(sketch, Rule::Distance { p, q: DistanceTo::Datum(plane), value: 0.0 }, datums)?;
        return (v >= 1e-6).then_some(Rule::Distance { p, q: DistanceTo::Datum(plane), value: v });
    }
    // The pick as a point: itself, or a round piece's centre.
    let point_of = |p: DimPick| -> Option<Anchor> {
        match p {
            DimPick::Point(anchor) => Some(anchor),
            _ => {
                let e = piece(p)?;
                (e.kind() != ShapeKind::Line).then(|| Anchor::new(e.id, AnchorAt::Center))
            }
        }
    };
    let a = piece(first);
    let same_piece = matches!((first, second), (DimPick::Piece(x), Some(DimPick::Piece(y))) if x == y);
    let Some(second) = second.filter(|_| !same_piece) else {
        let a = a?;
        return match a.shape {
            Shape::Line { .. } => measured(sketch, Rule::Length { entity: a.id, value: 0.0 }, datums).map(|v| Rule::Length { entity: a.id, value: v }),
            // A full circle is a hole or a boss, given by its diameter as a drawing gives it;
            // an arc — a round corner, a slot's end — by its radius.
            Shape::Circle { r, .. } => Some(Rule::Diameter { entity: a.id, value: 2.0 * r }),
            Shape::Arc { r, .. } => Some(Rule::Radius { entity: a.id, value: r }),
            Shape::Point { .. } | Shape::Spline(_) => None,
        };
    };
    let b = piece(second);
    if let (Some(ea), Some(eb)) = (a, b) {
        if let (Shape::Line { a: a0, b: a1 }, Shape::Line { a: b0, b: b1 }) = (ea.shape.clone(), eb.shape.clone()) {
            let da = normalize2(a1 - a0)?;
            let db = normalize2(b1 - b0)?;
            if cross2(da, db).abs() < (PARALLEL_DEG * PI / 180.0).sin() {
                let p = Anchor::new(ea.id, AnchorAt::A);
                let v = measured(sketch, Rule::Distance { p, q: DistanceTo::Line(eb.id), value: 0.0 }, datums)?;
                return Some(Rule::Distance { p, q: DistanceTo::Line(eb.id), value: v });
            }
            let v = measured(sketch, Rule::Angle { a: ea.id, b: eb.id, value: 0.0 }, datums)?;
            return Some(Rule::Angle { a: ea.id, b: eb.id, value: v });
        }
    }
    let a_is_line = a.is_some_and(|e| e.kind() == ShapeKind::Line);
    let line = if a_is_line {
        a
    } else if b.is_some_and(|e| e.kind() == ShapeKind::Line) {
        b
    } else {
        None
    };
    if let Some(line) = line {
        let p = point_of(if a_is_line { second } else { first })?;
        // An end of the line itself is no distance from it.
        if p.entity == line.id {
            return None;
        }
        let v = measured(sketch, Rule::Distance { p, q: DistanceTo::Line(line.id), value: 0.0 }, datums)?;
        // A corner the line shares lies on it just the same.
        return (v >= 1e-6).then_some(Rule::Distance { p, q: DistanceTo::Line(line.id), value: v });
    }
    let p = point_of(first)?;
    let q = point_of(second)?;
    if p == q {
        return None;
    }
    let v = measured(sketch, Rule::Distance { p, q: DistanceTo::Anchor(q), value: 0.0 }, datums)?;
    (v >= 1e-9).then_some(Rule::Distance { p, q: DistanceTo::Anchor(q), value: v })
}

/// Whether a first pick is a dimension by itself the moment it is made — a round piece's
/// radius or diameter, a line's length — or waits for a second click, as a point and a
/// coordinate plane do.
pub fn dimension_is_immediate(sketch: &Sketch, pick: DimPick) -> bool {
    let DimPick::Piece(id) = pick else { return false };
    sketch.entity(id).is_some_and(|e| matches!(e.shape, Shape::Arc { .. } | Shape::Circle { .. } | Shape::Line { .. }))
}

// ---- the ways a rectangle, a circle and an arc are drawn ----

/// A rectangle from two opposite corners, from its centre and a corner, or from one side's two
/// ends and a click where the opposite side lies — the one way that draws it at an angle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RectangleMode {
    Corners,
    Center,
    ThreePoint,
}

/// A circle from its centre and a point on the rim, from the two ends of a diameter, or
/// through three points on the rim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CircleMode {
    Center,
    TwoPoint,
    ThreePoint,
}

/// An arc from its two ends and a point it passes through, or from its centre, its start and
/// where it ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ArcMode {
    ThreePoint,
    Center,
}

/// A polygon from its centre and a corner (inscribed in its circle), or from its centre and
/// the middle of a side (circumscribed — the circle's diameter the width across flats).
pub type PolygonMode = PolygonFit;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DrawModes {
    pub rectangle: RectangleMode,
    pub circle: CircleMode,
    pub arc: ArcMode,
    pub polygon: PolygonMode,
}

pub const DEFAULT_DRAW_MODES: DrawModes = DrawModes { rectangle: RectangleMode::Corners, circle: CircleMode::Center, arc: ArcMode::ThreePoint, polygon: PolygonFit::Inscribed };

impl Default for DrawModes {
    fn default() -> Self {
        DEFAULT_DRAW_MODES
    }
}

/// How many clicks a drawing tool takes, in the mode it is in.
pub fn draw_clicks(tool: &str, modes: &DrawModes) -> usize {
    match tool {
        "arc" => 3,
        "polygon" => 2,
        "rectangle" => {
            if modes.rectangle == RectangleMode::ThreePoint {
                3
            } else {
                2
            }
        }
        "circle" => {
            if modes.circle == CircleMode::ThreePoint {
                3
            } else {
                2
            }
        }
        _ => 2,
    }
}

const SAME: f64 = 1e-9;

/// A rectangle's four corners, in order round it, from the clicks of its mode; the first is
/// the first click's for the two-corner and three-point ways, and the clicked corner is the
/// third for the centre way. `None` when it would have no area.
pub fn rectangle_corners(mode: RectangleMode, clicks: &[DVec2]) -> Option<[DVec2; 4]> {
    let p = *clicks.first()?;
    let q = *clicks.get(1)?;
    if mode == RectangleMode::ThreePoint {
        let side = q - p;
        let l = len2(side);
        let r = *clicks.get(2)?;
        if l < SAME {
            return None;
        }
        let n = perp2(side * (1.0 / l));
        let h = (r - p).dot(n);
        if h.abs() < SAME {
            return None;
        }
        let across = n * h;
        return Some([p, q, q + across, p + across]);
    }
    // The centre way mirrors the clicked corner through the centre.
    let a = if mode == RectangleMode::Center { p * 2.0 - q } else { p };
    let b = q;
    if (a.x - b.x).abs() < SAME || (a.y - b.y).abs() < SAME {
        return None;
    }
    Some([a, DVec2::new(b.x, a.y), b, DVec2::new(a.x, b.y)])
}

/// A circle from the clicks of its mode, as its centre and radius, or `None` when three points
/// lie on a line or two coincide.
pub fn circle_from(mode: CircleMode, clicks: &[DVec2]) -> Option<(DVec2, f64)> {
    let p = *clicks.first()?;
    let q = *clicks.get(1)?;
    if mode == CircleMode::ThreePoint {
        return circle_through(p, q, *clicks.get(2)?);
    }
    let r = if mode == CircleMode::TwoPoint { len2(q - p) / 2.0 } else { len2(q - p) };
    if r < SAME {
        return None;
    }
    Some((if mode == CircleMode::TwoPoint { (p + q) * 0.5 } else { p }, r))
}

/// The circle through three points, or `None` when they lie on a line.
fn circle_through(p: DVec2, q: DVec2, s: DVec2) -> Option<(DVec2, f64)> {
    let d = 2.0 * cross2(q - p, s - p);
    if d.abs() < SAME {
        return None;
    }
    let qp = q - p;
    let sp = s - p;
    let q2 = qp.dot(qp);
    let s2 = sp.dot(sp);
    let c = DVec2::new(p.x + (sp.y * q2 - qp.y * s2) / d, p.y + (qp.x * s2 - sp.x * q2) / d);
    Some((c, len2(p - c)))
}

/// An arc from its centre, its start and a click where it ends: the circle through the start
/// about the centre, from the start round to the end's direction the shorter way — the way the
/// cursor went, for any arc up to a half circle; a longer one is drawn through three points.
/// `None` when the start is on the centre or the end on the start.
pub fn arc_about(c: DVec2, start: DVec2, end: DVec2) -> Option<ArcPlacement> {
    let r = len2(start - c);
    let towards = end - c;
    if r < SAME || len2(towards) < SAME {
        return None;
    }
    let from = (start.y - c.y).atan2(start.x - c.x);
    let to = towards.y.atan2(towards.x);
    let turn = ccw(from, to);
    if turn < 1e-9 || TAU - turn < 1e-9 {
        return None;
    }
    // Counter-clockwise the shorter way, the start is anchor A; clockwise the arc is the same
    // piece run from the end's side.
    Some(if turn <= PI {
        ArcPlacement { c, r, start: from, sweep: turn, starts_at_p: true }
    } else {
        ArcPlacement { c, r, start: to, sweep: TAU - turn, starts_at_p: false }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datums::datum_traces;
    use crate::frame::SectionFrame;
    use glam::DVec3;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 5e-3
    }

    fn v(x: f64, y: f64) -> DVec2 {
        DVec2::new(x, y)
    }

    #[test]
    fn the_tangent_arc_leaves_its_start_along_the_tangent_and_ends_on_the_point_turning_left() {
        let arc = tangent_arc(v(0.0, 0.0), v(1.0, 0.0), v(5.0, 5.0)).unwrap();
        assert!(arc.starts_at_p);
        assert!(close(arc.c.x, 0.0) && close(arc.c.y, 5.0));
        assert!(close(arc.r, 5.0));
        assert!(close(arc.sweep, PI / 2.0));
    }

    #[test]
    fn turning_right_it_is_stored_counter_clockwise_from_the_far_end() {
        let arc = tangent_arc(v(0.0, 0.0), v(1.0, 0.0), v(5.0, -5.0)).unwrap();
        assert!(!arc.starts_at_p);
        assert!(close(arc.c.y, -5.0));
        assert!(close(arc.sweep, PI / 2.0));
        // Anchor A is at the far end, B back at the start.
        assert!(close(arc.c.x + arc.r * arc.start.cos(), 5.0));
        assert!(close(arc.c.y + arc.r * (arc.start + arc.sweep).sin(), 0.0));
    }

    #[test]
    fn there_is_no_tangent_arc_straight_ahead() {
        assert!(tangent_arc(v(0.0, 0.0), v(1.0, 0.0), v(7.0, 0.0)).is_none());
    }

    #[test]
    fn an_arc_runs_on_counter_clockwise_past_its_end_clockwise_past_its_start() {
        let mut s = Sketch::default();
        let id = s.add_shape(Shape::Arc { c: DVec2::ZERO, r: 2.0, start: 0.0, sweep: PI / 2.0 });
        let arc = s.entity(id).unwrap();
        let end = outward_tangent(arc, AnchorAt::B).unwrap();
        assert!(close(end.x, -1.0) && close(end.y, 0.0));
        let start = outward_tangent(arc, AnchorAt::A).unwrap();
        assert!(close(start.x, 0.0) && close(start.y, -1.0));
    }

    struct Picks {
        sketch: Sketch,
        bottom: EntityId,
        top: EntityId,
        slanted: EntityId,
        hole: EntityId,
        round: EntityId,
    }

    fn picks() -> Picks {
        let mut sketch = Sketch::default();
        let bottom = sketch.add_shape(Shape::Line { a: v(0.0, 0.0), b: v(10.0, 0.0) });
        let top = sketch.add_shape(Shape::Line { a: v(0.0, 4.0), b: v(10.0, 4.05) });
        let slanted = sketch.add_shape(Shape::Line { a: v(20.0, 0.0), b: v(25.0, 5.0) });
        let hole = sketch.add_shape(Shape::Circle { c: v(5.0, 2.0), r: 1.0 });
        let round = sketch.add_shape(Shape::Arc { c: v(30.0, 0.0), r: 2.0, start: 0.0, sweep: PI / 2.0 });
        Picks { sketch, bottom, top, slanted, hole, round }
    }

    #[test]
    fn one_line_is_its_length_one_circle_its_diameter_one_arc_its_radius() {
        let p = picks();
        assert_eq!(dimension_for(&p.sketch, DimPick::Piece(p.bottom), None, &[]), Some(Rule::Length { entity: p.bottom, value: 10.0 }));
        assert!(matches!(dimension_for(&p.sketch, DimPick::Piece(p.bottom), Some(DimPick::Piece(p.bottom)), &[]), Some(Rule::Length { .. })));
        assert_eq!(dimension_for(&p.sketch, DimPick::Piece(p.hole), None, &[]), Some(Rule::Diameter { entity: p.hole, value: 2.0 }));
        assert_eq!(dimension_for(&p.sketch, DimPick::Piece(p.round), None, &[]), Some(Rule::Radius { entity: p.round, value: 2.0 }));
    }

    #[test]
    fn two_nearly_parallel_lines_are_the_distance_between_them_two_others_their_angle() {
        let p = picks();
        let d = dimension_for(&p.sketch, DimPick::Piece(p.bottom), Some(DimPick::Piece(p.top)), &[]).unwrap();
        let Rule::Distance { q: DistanceTo::Line(l), value, .. } = d else { panic!("{d:?}") };
        assert_eq!(l, p.top);
        assert!((value - 4.0).abs() < 0.05);
        let Some(Rule::Angle { value, .. }) = dimension_for(&p.sketch, DimPick::Piece(p.bottom), Some(DimPick::Piece(p.slanted)), &[]) else { panic!() };
        assert!((value - 45.0).abs() < 1e-9);
    }

    #[test]
    fn an_end_and_a_line_a_centre_and_a_line_two_ends() {
        let p = picks();
        let end = dim_pick_at(&p.sketch, v(25.1, 5.1), 0.5, &[]).unwrap();
        assert_eq!(end, DimPick::Point(Anchor::new(p.slanted, AnchorAt::B)));
        let Some(Rule::Distance { p: from, q: DistanceTo::Line(l), value }) = dimension_for(&p.sketch, end, Some(DimPick::Piece(p.bottom)), &[]) else { panic!() };
        assert_eq!((from, l), (Anchor::new(p.slanted, AnchorAt::B), p.bottom));
        assert!((value - 5.0).abs() < 1e-9);
        let Some(Rule::Distance { p: from, value, .. }) = dimension_for(&p.sketch, DimPick::Piece(p.bottom), Some(DimPick::Piece(p.hole)), &[]) else { panic!() };
        assert_eq!(from, Anchor::new(p.hole, AnchorAt::Center));
        assert!((value - 2.0).abs() < 1e-9);
        let corner = dim_pick_at(&p.sketch, v(0.0, 0.0), 0.5, &[]).unwrap();
        let Some(Rule::Distance { p: from, q: DistanceTo::Anchor(to), .. }) = dimension_for(&p.sketch, corner, Some(end), &[]) else { panic!() };
        assert_eq!((from, to), (Anchor::new(p.bottom, AnchorAt::A), Anchor::new(p.slanted, AnchorAt::B)));
    }

    #[test]
    fn a_lines_own_end_is_no_distance_from_it_and_a_point_alone_is_nothing() {
        let p = picks();
        let corner = dim_pick_at(&p.sketch, v(0.0, 0.0), 0.5, &[]).unwrap();
        assert_eq!(dimension_for(&p.sketch, corner, Some(DimPick::Piece(p.bottom)), &[]), None);
        assert_eq!(dimension_for(&p.sketch, corner, None, &[]), None);
    }

    #[test]
    fn the_middle_of_a_piece_picks_the_piece_its_end_the_point() {
        let p = picks();
        assert_eq!(dim_pick_at(&p.sketch, v(5.0, 0.1), 0.5, &[]), Some(DimPick::Piece(p.bottom)));
        assert_eq!(dim_pick_at(&p.sketch, v(50.0, 50.0), 0.5, &[]), None);
    }

    #[test]
    fn an_end_then_a_coordinate_planes_line_is_the_distance_to_the_plane() {
        // An L drawn by hand, its far end at (10, 8); the YZ plane cuts the sheet along u = 0.
        let mut s = Sketch::default();
        let first = s.add_shape(Shape::Line { a: v(0.0, 0.0), b: v(10.0, 0.0) });
        let second = s.add_shape(Shape::Line { a: v(10.0, 0.0), b: v(10.0, 8.0) });
        let traces = datum_traces(&SectionFrame { origin: DVec3::ZERO, normal: DVec3::Z, u: DVec3::X, v: DVec3::Y });
        let end = dim_pick_at(&s, v(10.0, 8.0), 0.5, &traces).unwrap();
        let plane = dim_pick_at(&s, v(0.1, 30.0), 0.5, &traces).unwrap();
        assert_eq!(plane, DimPick::Datum(DatumPlane::Yz));
        let Some(Rule::Distance { p, q: DistanceTo::Datum(DatumPlane::Yz), value }) = dimension_for(&s, end, Some(plane), &traces) else { panic!() };
        assert_eq!(p, Anchor::new(second, AnchorAt::B));
        assert!((value - 10.0).abs() < 1e-9);
        // An end, then a line, is the distance between them.
        let Some(Rule::Distance { q: DistanceTo::Line(l), value, .. }) = dimension_for(&s, end, dim_pick_at(&s, v(5.0, 0.0), 0.5, &traces), &traces) else { panic!() };
        assert_eq!(l, first);
        assert!((value - 8.0).abs() < 1e-9);
    }

    #[test]
    fn coincident_reads_the_pair_as_cad_does() {
        let p = picks();
        let end = DimPick::Point(Anchor::new(p.slanted, AnchorAt::B));
        assert_eq!(coincident_for(&p.sketch, end, DimPick::Piece(p.bottom)), Ok(Rule::PointOn { p: Anchor::new(p.slanted, AnchorAt::B), on: OnTarget::Line(p.bottom), bounded: false }));
        assert_eq!(coincident_for(&p.sketch, DimPick::Piece(p.hole), end), Ok(Rule::PointOn { p: Anchor::new(p.slanted, AnchorAt::B), on: OnTarget::Curve(p.hole), bounded: false }));
        assert_eq!(coincident_for(&p.sketch, DimPick::Piece(p.bottom), DimPick::Piece(p.top)), Ok(Rule::Collinear { line: p.bottom, on: OnTarget::Line(p.top) }));
        assert_eq!(coincident_for(&p.sketch, DimPick::Datum(DatumPlane::Zx), DimPick::Piece(p.top)), Ok(Rule::Collinear { line: p.top, on: OnTarget::Datum(DatumPlane::Zx) }));
        assert!(coincident_for(&p.sketch, DimPick::Piece(p.hole), DimPick::Piece(p.round)).unwrap_err().contains("Tangent or Concentric"));
        assert!(coincident_for(&p.sketch, DimPick::Piece(p.bottom), DimPick::Piece(p.round)).unwrap_err().contains("Tangent, not Coincident"));
        assert!(coincident_for(&p.sketch, DimPick::Datum(DatumPlane::Yz), DimPick::Datum(DatumPlane::Zx)).unwrap_err().contains("Two coordinate planes"));
        assert!(coincident_for(&p.sketch, DimPick::Piece(p.bottom), DimPick::Point(Anchor::new(p.bottom, AnchorAt::A))).unwrap_err().contains("two different"));
    }

    #[test]
    fn corners_circles_and_arcs_from_the_clicks_of_each_way() {
        assert_eq!(draw_clicks("rectangle", &DrawModes { rectangle: RectangleMode::ThreePoint, ..DEFAULT_DRAW_MODES }), 3);
        assert_eq!(draw_clicks("circle", &DrawModes { circle: CircleMode::TwoPoint, ..DEFAULT_DRAW_MODES }), 2);
        assert_eq!(rectangle_corners(RectangleMode::Center, &[v(5.0, 5.0), v(8.0, 7.0)]), Some([v(2.0, 3.0), v(8.0, 3.0), v(8.0, 7.0), v(2.0, 7.0)]));
        assert_eq!(rectangle_corners(RectangleMode::Corners, &[v(0.0, 0.0), v(0.0, 7.0)]), None);
        let turned = rectangle_corners(RectangleMode::ThreePoint, &[v(0.0, 0.0), v(6.0, 8.0), v(-4.0, 3.0)]).unwrap();
        assert!(close(turned[2].x, 2.0) && close(turned[2].y, 11.0));
        assert_eq!(rectangle_corners(RectangleMode::ThreePoint, &[v(0.0, 0.0), v(6.0, 8.0), v(3.0, 4.0)]), None);
        assert_eq!(circle_from(CircleMode::TwoPoint, &[v(0.0, 0.0), v(10.0, 0.0)]), Some((v(5.0, 0.0), 5.0)));
        let (c, r) = circle_from(CircleMode::ThreePoint, &[v(0.0, 0.0), v(10.0, 0.0), v(5.0, 5.0)]).unwrap();
        assert!(close(c.x, 5.0) && close(c.y, 0.0) && close(r, 5.0));
        assert_eq!(circle_from(CircleMode::ThreePoint, &[v(0.0, 0.0), v(5.0, 0.0), v(10.0, 0.0)]), None);
        let quarter = arc_about(v(0.0, 0.0), v(5.0, 0.0), v(0.0, 7.0)).unwrap();
        assert_eq!((quarter.r, quarter.start, quarter.starts_at_p), (5.0, 0.0, true));
        assert!(close(quarter.sweep, PI / 2.0));
        // The cursor clockwise of the start: the same quarter, run from its end.
        let back = arc_about(v(0.0, 0.0), v(5.0, 0.0), v(0.0, -7.0)).unwrap();
        assert!(!back.starts_at_p);
        assert!(close(back.start, -PI / 2.0));
        assert!(close(back.sweep, PI / 2.0));
    }
}
