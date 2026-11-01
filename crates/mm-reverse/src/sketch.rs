// SPDX-License-Identifier: AGPL-3.0-only
//! The sketch of the Reverse Engineer workspace: lines, arcs, circles and points on a plane,
//! the relations and dimensions between them, and the slice points a fitted piece was
//! measured on. Ported from ScanRuler (`plugins/reverse/core/sketch.ts`), © Stefan Hermann
//! (CNC Kitchen), AGPL-3.0-only.
//!
//! Everything is plain data in sheet millimetres — the (u, v) of the frame the sketch lies
//! on — so a sketch saves as JSON, solves in a few milliseconds (see `solver`) and is stood up
//! in the part through its frame. A piece is either *fitted*, with the slice points it was
//! segmented from kept beside it as soft data the solver keeps pulling it towards, or
//! *free*, drawn by hand with nothing behind it but a weak wish to stay where it was drawn.
//! Relations and typed dimensions are hard: the solver meets them exactly or reports that it
//! cannot.

use std::collections::HashMap;
use std::f64::consts::{PI, TAU};

use glam::DVec2;
use serde::{Deserialize, Serialize};

use crate::datums::DatumPlane;
use crate::fit2d::unwind_angle;

pub type EntityId = u32;
pub type ConstraintId = u32;

/// One of the two ends of a line, an arc or an open spline, the one point a point or a circle
/// stands for (its centre), or a spline's fit point by its place among them (`P(i)`) — an open
/// spline's first and last are its ends `A` and `B`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AnchorAt {
    A,
    B,
    Center,
    P(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Anchor {
    pub entity: EntityId,
    pub at: AnchorAt,
}

impl Anchor {
    pub fn new(entity: EntityId, at: AnchorAt) -> Self {
        Self { entity, at }
    }
}

/// The geometry of a piece. An arc runs counter-clockwise from `start` through `sweep`
/// radians; anchor `A` is at `start`, anchor `B` at `start + sweep`. A spline is the smooth
/// curve through its fit points (`crate::sketch_spline`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Shape {
    Line { a: DVec2, b: DVec2 },
    Arc { c: DVec2, r: f64, start: f64, sweep: f64 },
    Circle { c: DVec2, r: f64 },
    Point { p: DVec2 },
    Spline(crate::sketch_spline::SplineShape),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ShapeKind {
    Line,
    Arc,
    Circle,
    Point,
    Spline,
}

impl Shape {
    pub fn kind(&self) -> ShapeKind {
        match self {
            Shape::Line { .. } => ShapeKind::Line,
            Shape::Arc { .. } => ShapeKind::Arc,
            Shape::Circle { .. } => ShapeKind::Circle,
            Shape::Point { .. } => ShapeKind::Point,
            Shape::Spline(_) => ShapeKind::Spline,
        }
    }

    pub fn is_round(&self) -> bool {
        matches!(self, Shape::Arc { .. } | Shape::Circle { .. })
    }

    pub fn centre(&self) -> Option<DVec2> {
        match *self {
            Shape::Arc { c, .. } | Shape::Circle { c, .. } => Some(c),
            Shape::Point { p } => Some(p),
            Shape::Line { .. } | Shape::Spline(_) => None,
        }
    }

    /// The spline, when the piece is one.
    pub fn spline(&self) -> Option<&crate::sketch_spline::SplineShape> {
        match self {
            Shape::Spline(s) => Some(s),
            _ => None,
        }
    }

    pub fn radius(&self) -> Option<f64> {
        match *self {
            Shape::Arc { r, .. } | Shape::Circle { r, .. } => Some(r),
            _ => None,
        }
    }
}

/// A piece of the sketch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Entity {
    pub id: EntityId,
    pub shape: Shape,
    /// The slice points the piece was fitted to, or none for free geometry.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub points: Vec<DVec2>,
    /// The scatter of those points, millimetres — the weight the soft data carries.
    #[serde(default)]
    pub scatter: f64,
    /// Locked where it is.
    #[serde(default)]
    pub fixed: bool,
    /// Projected from a body's edge or another sketch: fixed where it landed.
    #[serde(default)]
    pub projected: bool,
    /// Where a projected piece was taken from, when it follows it (see `crate::project`): the
    /// build takes it again from there every time. A projection saved before projections
    /// followed the part has none, and stays where it landed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<ProjectionSource>,
    /// Drawn for construction only — a mirror line, an axis — and no part of the regions.
    #[serde(default)]
    pub construction: bool,
}

/// What a projected piece follows: a body's edge, by its fingerprint, or a piece of another
/// sketch, by its id. As ScanRuler's `ProjectionSource` (`{ body, edge }` or
/// `{ sketch, entity }`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ProjectionSource {
    Edge { body: crate::model::BodyId, edge: crate::topo::EdgeFingerprint },
    Sketch { sketch: crate::model::FeatureId, entity: EntityId },
}

impl Entity {
    pub fn new(shape: Shape) -> Self {
        Self { id: 0, shape, points: Vec::new(), scatter: 0.0, fixed: false, projected: false, source: None, construction: false }
    }

    pub fn construction(mut self) -> Self {
        self.construction = true;
        self
    }

    pub fn fitted(mut self, points: Vec<DVec2>, scatter: f64) -> Self {
        self.points = points;
        self.scatter = scatter;
        self
    }

    pub fn kind(&self) -> ShapeKind {
        self.shape.kind()
    }

    pub fn is_fitted(&self) -> bool {
        !self.points.is_empty()
    }
}

/// What a point or a line is held on: a line of the sketch, a round piece's rim, or a
/// coordinate plane's line in the sheet.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum OnTarget {
    Line(EntityId),
    Curve(EntityId),
    Datum(DatumPlane),
}

/// What a distance is taken to.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DistanceTo {
    Anchor(Anchor),
    Line(EntityId),
    Datum(DatumPlane),
}

/// What two anchors are mirrored about.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum MirrorAxis {
    X,
    Y,
    Line(EntityId),
    Datum(DatumPlane),
}

/// The relations and dimensions. A dimension is a relation with a value — the same hard
/// equation to the solver, shown differently by the panel.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Rule {
    Coincident { p: Anchor, q: Anchor },
    Horizontal { entity: EntityId },
    Vertical { entity: EntityId },
    Parallel { a: EntityId, b: EntityId },
    Perpendicular { a: EntityId, b: EntityId },
    /// A line tangent to an arc or a circle, or two of those tangent to each other.
    Tangent { a: EntityId, b: EntityId },
    Concentric { a: EntityId, b: EntityId },
    /// Two lines of one length, or two arcs or circles of one radius.
    Equal { a: EntityId, b: EntityId },
    /// An anchor held on a piece rather than at its end; `bounded` keeps it within the piece.
    PointOn { p: Anchor, on: OnTarget, bounded: bool },
    /// A line lying along another line or a coordinate plane's line (never a curve).
    Collinear { line: EntityId, on: OnTarget },
    Symmetric { p: Anchor, q: Anchor, about: MirrorAxis },
    Length { entity: EntityId, value: f64 },
    Distance { p: Anchor, q: DistanceTo, value: f64 },
    Radius { entity: EntityId, value: f64 },
    /// A circle's or an arc's diameter; the solver holds the radius at half of it.
    Diameter { entity: EntityId, value: f64 },
    /// Between two lines, degrees, 0–180 as the lines' directions read.
    Angle { a: EntityId, b: EntityId, value: f64 },
}

impl Rule {
    pub fn is_dimension(&self) -> bool {
        self.value().is_some()
    }

    pub fn value(&self) -> Option<f64> {
        match *self {
            Rule::Length { value, .. } | Rule::Distance { value, .. } | Rule::Radius { value, .. } | Rule::Diameter { value, .. } | Rule::Angle { value, .. } => Some(value),
            _ => None,
        }
    }

    pub fn set_value(&mut self, v: f64) {
        match self {
            Rule::Length { value, .. } | Rule::Distance { value, .. } | Rule::Radius { value, .. } | Rule::Diameter { value, .. } | Rule::Angle { value, .. } => *value = v,
            _ => {}
        }
    }

    /// The name the panel lists the rule by.
    pub fn name(&self) -> &'static str {
        match self {
            Rule::Coincident { .. } => "Coincident",
            Rule::Horizontal { .. } => "Horizontal",
            Rule::Vertical { .. } => "Vertical",
            Rule::Parallel { .. } => "Parallel",
            Rule::Perpendicular { .. } => "Perpendicular",
            Rule::Tangent { .. } => "Tangent",
            Rule::Concentric { .. } => "Concentric",
            Rule::Equal { .. } => "Equal",
            Rule::PointOn { .. } => "Point on",
            Rule::Collinear { .. } => "Collinear",
            Rule::Symmetric { .. } => "Symmetric",
            Rule::Length { .. } => "Length",
            Rule::Distance { .. } => "Distance",
            Rule::Radius { .. } => "Radius",
            Rule::Diameter { .. } => "Diameter",
            Rule::Angle { .. } => "Angle",
        }
    }
}

/// Where the operator dragged a dimension's number to, told against what it measures.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DimPlace {
    pub off: f64,
    pub along: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Constraint {
    pub id: ConstraintId,
    pub rule: Rule,
    /// Added by the sketcher on its own while a piece was drawn — shown as a chip the
    /// operator can take off.
    #[serde(default)]
    pub inferred: bool,
    /// The expression a dimension's value follows, if typed as one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expr: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub place: Option<DimPlace>,
}

/// The sketch as stored: its pieces and what holds them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sketch {
    pub entities: Vec<Entity>,
    pub constraints: Vec<Constraint>,
    pub next_id: u32,
}

impl Default for Sketch {
    fn default() -> Self {
        Self { entities: Vec::new(), constraints: Vec::new(), next_id: 1 }
    }
}

/// How far off level or upright a drawn line may be and still be meant so, degrees.
pub const INFER_AXIS_DEG: f64 = 3.0;
/// Ends this close are one corner — the distance a closed region is found across.
pub const TOUCH_MM: f64 = 2e-3;

/// The relation a line drawn from `a` to `b` was meant to have.
pub fn axis_relation_of(a: DVec2, b: DVec2, degrees: f64) -> Option<AxisRelation> {
    let d = b - a;
    let l = d.length();
    if !(l > 1e-9) {
        return None;
    }
    let limit = degrees.to_radians().sin();
    if d.y.abs() / l <= limit {
        Some(AxisRelation::Horizontal)
    } else if d.x.abs() / l <= limit {
        Some(AxisRelation::Vertical)
    } else {
        None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AxisRelation {
    Horizontal,
    Vertical,
}

impl AxisRelation {
    pub fn rule(self, entity: EntityId) -> Rule {
        match self {
            AxisRelation::Horizontal => Rule::Horizontal { entity },
            AxisRelation::Vertical => Rule::Vertical { entity },
        }
    }
}

// ---- geometry of a piece ----

pub fn arc_point(c: DVec2, r: f64, angle: f64) -> DVec2 {
    c + DVec2::new(angle.cos(), angle.sin()) * r
}

/// Where an anchor lies. A spline's `A`/`B` are an open curve's first and last fit points,
/// `P(i)` its fit point `i`; a spline has no centre, and what asks for one is given its middle.
pub fn anchor_point(shape: &Shape, at: AnchorAt) -> DVec2 {
    match shape {
        Shape::Line { a, b } => match at {
            AnchorAt::A => *a,
            AnchorAt::B => *b,
            AnchorAt::Center | AnchorAt::P(_) => (*a + *b) / 2.0,
        },
        Shape::Arc { c, r, start, sweep } => match at {
            AnchorAt::A => arc_point(*c, *r, *start),
            AnchorAt::B => arc_point(*c, *r, start + sweep),
            AnchorAt::Center | AnchorAt::P(_) => *c,
        },
        Shape::Circle { c, .. } => *c,
        Shape::Point { p } => *p,
        Shape::Spline(e) => {
            let i = match at {
                AnchorAt::P(i) => crate::sketch_spline::spline_fit_point_index(e, i as usize),
                _ => crate::sketch_spline::spline_point_index(e, at),
            };
            crate::sketch_spline::spline_anchor_point(e, i)
        }
    }
}

/// The anchors an entity offers. A spline's are its ends (an open one's `A` and `B`); its fit
/// points between are in [`all_anchors`].
pub fn anchors_of(shape: &Shape) -> &'static [AnchorAt] {
    match shape {
        Shape::Line { .. } | Shape::Arc { .. } => &[AnchorAt::A, AnchorAt::B],
        Shape::Spline(e) if crate::sketch_spline::is_closed(e) => &[],
        Shape::Spline(_) => &[AnchorAt::A, AnchorAt::B],
        _ => &[AnchorAt::Center],
    }
}

/// Every anchor an entity offers, a spline's fit points too (ScanRuler's `anchorsOf`): an open
/// curve's ends first, then the points between; a closed curve's from `P(0)`.
pub fn all_anchors(shape: &Shape) -> Vec<AnchorAt> {
    match shape {
        Shape::Spline(e) => crate::sketch_spline::spline_anchor_order(e).into_iter().map(|i| crate::sketch_spline::spline_anchor(e, i).unwrap_or(AnchorAt::P(i as u32))).collect(),
        _ => anchors_of(shape).to_vec(),
    }
}

/// The ends of a piece — a line's, an arc's or an open spline's two; none for anything else.
pub fn ends_of(shape: &Shape) -> &'static [AnchorAt] {
    match shape {
        Shape::Line { .. } | Shape::Arc { .. } => &[AnchorAt::A, AnchorAt::B],
        Shape::Spline(e) if crate::sketch_spline::spline_has_ends(e) => &[AnchorAt::A, AnchorAt::B],
        _ => &[],
    }
}

/// The angle of a point about a centre, from +u.
pub fn angle_of(c: DVec2, p: DVec2) -> f64 {
    (p.y - c.y).atan2(p.x - c.x)
}

/// The point on a piece closest to `p`, and its distance — for picking.
pub fn nearest_on(shape: &Shape, p: DVec2) -> (DVec2, f64) {
    if let Shape::Spline(e) = shape {
        return crate::sketch_spline::nearest_on_spline_piece(e, p);
    }
    match *shape {
        Shape::Spline(_) => (p, f64::INFINITY),
        Shape::Point { p: q } => (q, (p - q).length()),
        Shape::Line { a, b } => {
            let ab = b - a;
            let ll = ab.dot(ab);
            let t = if ll > 0.0 { (p - a).dot(ab) / ll } else { 0.0 };
            let q = if t < 0.0 {
                a
            } else if t > 1.0 {
                b
            } else {
                a + ab * t
            };
            (q, (p - q).length())
        }
        Shape::Circle { c, r } => {
            let d = (p - c).try_normalize().unwrap_or(DVec2::X);
            let q = c + d * r;
            (q, (p - q).length())
        }
        Shape::Arc { c, r, start, sweep } => {
            let angle = unwind_angle(angle_of(c, p), start);
            if angle - start <= sweep {
                let q = arc_point(c, r, angle);
                return (q, (p - q).length());
            }
            let (a, b) = (arc_point(c, r, start), arc_point(c, r, start + sweep));
            let (da, db) = ((p - a).length(), (p - b).length());
            if da <= db {
                (a, da)
            } else {
                (b, db)
            }
        }
    }
}

/// The polyline a piece draws as, `steps` segments per full turn of arc.
pub fn polyline(shape: &Shape, steps: usize) -> Vec<DVec2> {
    match *shape {
        Shape::Spline(ref e) => crate::sketch_spline::spline_piece_polyline(e, steps),
        Shape::Point { p } => vec![p],
        Shape::Line { a, b } => vec![a, b],
        Shape::Circle { c, r } => arc_polyline(c, r, 0.0, TAU, steps),
        Shape::Arc { c, r, start, sweep } => arc_polyline(c, r, start, sweep, steps),
    }
}

fn arc_polyline(c: DVec2, r: f64, start: f64, sweep: f64, steps: usize) -> Vec<DVec2> {
    let n = ((steps as f64 * sweep.abs() / TAU).ceil() as usize).max(2);
    (0..=n).map(|i| arc_point(c, r, start + sweep * i as f64 / n as f64)).collect()
}

/// The arc through three points — start `p`, end `q`, passing through `r` — as centre,
/// radius, start angle and sweep, and whether anchor `A` is at `p`. `None` when collinear.
pub fn arc_through(p: DVec2, q: DVec2, r: DVec2) -> Option<(DVec2, f64, f64, f64, bool)> {
    let (ax, ay, bx, by, cx, cy) = (p.x, p.y, q.x, q.y, r.x, r.y);
    let d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
    if d.abs() < 1e-9 {
        return None;
    }
    let (a2, b2, c2) = (ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy);
    let ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d;
    let uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d;
    let c = DVec2::new(ux, uy);
    let radius = (p - c).length();
    let (tp, tq, tr) = (angle_of(c, p), angle_of(c, q), angle_of(c, r));
    let ccw_pq = (tq - tp).rem_euclid(TAU);
    let ccw_pr = (tr - tp).rem_euclid(TAU);
    if ccw_pr <= ccw_pq {
        Some((c, radius, tp, ccw_pq, true))
    } else {
        Some((c, radius, tq, TAU - ccw_pq, false))
    }
}

impl Sketch {
    pub fn entity(&self, id: EntityId) -> Option<&Entity> {
        self.entities.iter().find(|e| e.id == id)
    }

    pub fn entity_mut(&mut self, id: EntityId) -> Option<&mut Entity> {
        self.entities.iter_mut().find(|e| e.id == id)
    }

    pub fn constraint(&self, id: ConstraintId) -> Option<&Constraint> {
        self.constraints.iter().find(|c| c.id == id)
    }

    pub fn constraint_mut(&mut self, id: ConstraintId) -> Option<&mut Constraint> {
        self.constraints.iter_mut().find(|c| c.id == id)
    }

    pub fn anchor(&self, a: Anchor) -> Option<DVec2> {
        self.entity(a.entity).map(|e| anchor_point(&e.shape, a.at))
    }

    /// Add an entity, handing back its id.
    pub fn add(&mut self, mut entity: Entity) -> EntityId {
        let id = self.next_id;
        self.next_id += 1;
        entity.id = id;
        self.entities.push(entity);
        id
    }

    pub fn add_shape(&mut self, shape: Shape) -> EntityId {
        self.add(Entity::new(shape))
    }

    pub fn constrain(&mut self, rule: Rule) -> ConstraintId {
        self.push_constraint(rule, false)
    }

    pub fn infer(&mut self, rule: Rule) -> ConstraintId {
        self.push_constraint(rule, true)
    }

    fn push_constraint(&mut self, rule: Rule, inferred: bool) -> ConstraintId {
        let id = self.next_id;
        self.next_id += 1;
        self.constraints.push(Constraint { id, rule, inferred, expr: None, place: None });
        id
    }

    /// Remove entities and every constraint that referred to them.
    pub fn remove_entities(&mut self, ids: &[EntityId]) {
        self.entities.retain(|e| !ids.contains(&e.id));
        self.constraints.retain(|c| !constraint_entities(&c.rule).iter().any(|id| ids.contains(id)));
    }

    pub fn remove_constraint(&mut self, id: ConstraintId) {
        self.constraints.retain(|c| c.id != id);
    }

    /// The extent of the sketch's pieces and their slice points.
    pub fn bounds(&self) -> Option<(DVec2, DVec2)> {
        let mut lo = DVec2::splat(f64::INFINITY);
        let mut hi = DVec2::splat(f64::NEG_INFINITY);
        for e in &self.entities {
            for p in polyline(&e.shape, 32).into_iter().chain(e.points.iter().copied()) {
                lo = lo.min(p);
                hi = hi.max(p);
            }
        }
        lo.x.is_finite().then_some((lo, hi))
    }

    /// A rectangle from two opposite corners: four lines, horizontal and vertical,
    /// coincident at the corners.
    pub fn add_rectangle(&mut self, p: DVec2, q: DVec2) -> [EntityId; 4] {
        let (lo, hi) = (p.min(q), p.max(q));
        let corners = [lo, DVec2::new(hi.x, lo.y), hi, DVec2::new(lo.x, hi.y)];
        let ids: Vec<EntityId> = (0..4).map(|i| self.add_shape(Shape::Line { a: corners[i], b: corners[(i + 1) % 4] })).collect();
        for i in 0..4 {
            self.constrain(Rule::Coincident { p: Anchor::new(ids[i], AnchorAt::B), q: Anchor::new(ids[(i + 1) % 4], AnchorAt::A) });
            let rule = if i % 2 == 0 { Rule::Horizontal { entity: ids[i] } } else { Rule::Vertical { entity: ids[i] } };
            self.infer(rule);
        }
        [ids[0], ids[1], ids[2], ids[3]]
    }

    /// A rectangle at any angle from its four corners in order round it.
    pub fn add_turned_rectangle(&mut self, corners: [DVec2; 4]) -> [EntityId; 4] {
        let ids: Vec<EntityId> = (0..4).map(|i| self.add_shape(Shape::Line { a: corners[i], b: corners[(i + 1) % 4] })).collect();
        for i in 0..4 {
            self.constrain(Rule::Coincident { p: Anchor::new(ids[i], AnchorAt::B), q: Anchor::new(ids[(i + 1) % 4], AnchorAt::A) });
        }
        if let Some(axis) = axis_relation_of(corners[0], corners[1], INFER_AXIS_DEG) {
            for (i, id) in ids.iter().enumerate() {
                let even = i % 2 == 0;
                let horizontal = even == (axis == AxisRelation::Horizontal);
                self.infer(if horizontal { Rule::Horizontal { entity: *id } } else { Rule::Vertical { entity: *id } });
            }
        } else {
            self.constrain(Rule::Parallel { a: ids[0], b: ids[2] });
            self.constrain(Rule::Parallel { a: ids[1], b: ids[3] });
            self.constrain(Rule::Perpendicular { a: ids[0], b: ids[1] });
        }
        [ids[0], ids[1], ids[2], ids[3]]
    }

    /// A regular polygon as CAD draws one: corners on a construction circle, sides equal.
    /// `at` is a corner (inscribed) or the middle of a side (circumscribed).
    pub fn add_polygon(&mut self, centre: DVec2, at: DVec2, sides: usize, circumscribed: bool) -> Option<Vec<EntityId>> {
        let corners = polygon_corners(centre, at, sides, circumscribed)?;
        let n = corners.len();
        let lines: Vec<EntityId> = (0..n).map(|i| self.add_shape(Shape::Line { a: corners[i], b: corners[(i + 1) % n] })).collect();
        let rim = self.add(Entity::new(Shape::Circle { c: centre, r: (corners[0] - centre).length() }).construction());
        let flats = circumscribed.then(|| self.add(Entity::new(Shape::Circle { c: centre, r: (at - centre).length() }).construction()));
        for i in 0..n {
            self.constrain(Rule::Coincident { p: Anchor::new(lines[i], AnchorAt::B), q: Anchor::new(lines[(i + 1) % n], AnchorAt::A) });
            self.constrain(Rule::PointOn { p: Anchor::new(lines[i], AnchorAt::A), on: OnTarget::Curve(rim), bounded: false });
            if i > 0 {
                self.constrain(Rule::Equal { a: lines[0], b: lines[i] });
            }
        }
        if let Some(flats) = flats {
            self.constrain(Rule::Concentric { a: flats, b: rim });
            self.constrain(Rule::Tangent { a: lines[0], b: flats });
        }
        Some(lines)
    }

    /// The relations a sketch drawn by hand would have been given as it was drawn, put on
    /// pieces handed in as coordinates: a coincidence where two ends meet (within
    /// `TOUCH_MM`, and not already joined through a chain of coincidences), and horizontal or
    /// vertical on a line lying exactly level or upright that has neither. Returns how many
    /// were added.
    pub fn infer_relations(&mut self) -> usize {
        let mut ends: Vec<(Anchor, DVec2)> = Vec::new();
        for e in &self.entities {
            for &at in ends_of(&e.shape) {
                ends.push((Anchor::new(e.id, at), anchor_point(&e.shape, at)));
            }
        }
        let mut parent: HashMap<Anchor, Anchor> = HashMap::new();
        fn find(parent: &mut HashMap<Anchor, Anchor>, k: Anchor) -> Anchor {
            let mut r = k;
            while let Some(&p) = parent.get(&r) {
                if p == r {
                    break;
                }
                r = p;
            }
            parent.insert(k, r);
            r
        }
        for c in &self.constraints {
            if let Rule::Coincident { p, q } = c.rule {
                let (rp, rq) = (find(&mut parent, p), find(&mut parent, q));
                parent.insert(rp, rq);
            }
        }
        let size = TOUCH_MM * 4.0;
        let cell = |p: DVec2| ((p.x / size).floor() as i64, (p.y / size).floor() as i64);
        let mut grid: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
        for (i, (_, p)) in ends.iter().enumerate() {
            grid.entry(cell(*p)).or_default().push(i);
        }
        let mut added = 0;
        let mut joins = Vec::new();
        for i in 0..ends.len() {
            let (ea, ep) = ends[i];
            let (cx, cy) = cell(ep);
            for dx in -1..=1 {
                for dy in -1..=1 {
                    let Some(list) = grid.get(&(cx + dx, cy + dy)) else { continue };
                    for &j in list {
                        if j <= i {
                            continue;
                        }
                        let (fa, fp) = ends[j];
                        if fa.entity == ea.entity || (ep - fp).length() > TOUCH_MM {
                            continue;
                        }
                        let (ra, rb) = (find(&mut parent, ea), find(&mut parent, fa));
                        if ra == rb {
                            continue;
                        }
                        joins.push((ea, fa));
                        parent.insert(ra, rb);
                    }
                }
            }
        }
        for (p, q) in joins {
            self.infer(Rule::Coincident { p, q });
            added += 1;
        }
        let lines: Vec<(EntityId, DVec2, DVec2)> = self
            .entities
            .iter()
            .filter(|e| !e.construction)
            .filter_map(|e| if let Shape::Line { a, b } = e.shape { Some((e.id, a, b)) } else { None })
            .collect();
        for (id, a, b) in lines {
            let has = self.constraints.iter().any(|c| matches!(c.rule, Rule::Horizontal { entity } | Rule::Vertical { entity } if entity == id));
            if has {
                continue;
            }
            if let Some(axis) = axis_relation_of(a, b, 1e-6) {
                self.infer(axis.rule(id));
                added += 1;
            }
        }
        added
    }
}

/// The ids a constraint refers to.
pub fn constraint_entities(rule: &Rule) -> Vec<EntityId> {
    match *rule {
        Rule::Coincident { p, q } => vec![p.entity, q.entity],
        Rule::Symmetric { p, q, about } => {
            let mut ids = vec![p.entity, q.entity];
            if let MirrorAxis::Line(l) = about {
                ids.push(l);
            }
            ids
        }
        Rule::Horizontal { entity } | Rule::Vertical { entity } | Rule::Length { entity, .. } | Rule::Radius { entity, .. } | Rule::Diameter { entity, .. } => vec![entity],
        Rule::Parallel { a, b } | Rule::Perpendicular { a, b } | Rule::Tangent { a, b } | Rule::Concentric { a, b } | Rule::Equal { a, b } | Rule::Angle { a, b, .. } => vec![a, b],
        Rule::Distance { p, q, .. } => match q {
            DistanceTo::Datum(_) => vec![p.entity],
            DistanceTo::Line(l) => vec![p.entity, l],
            DistanceTo::Anchor(q) => vec![p.entity, q.entity],
        },
        Rule::PointOn { p, on, .. } => match on {
            OnTarget::Datum(_) => vec![p.entity],
            OnTarget::Line(l) | OnTarget::Curve(l) => vec![p.entity, l],
        },
        Rule::Collinear { line, on } => match on {
            OnTarget::Datum(_) => vec![line],
            OnTarget::Line(l) | OnTarget::Curve(l) => vec![line, l],
        },
    }
}

/// Whether a relation of this kind can be put on the given pieces — the panel's enabling rule.
pub fn relation_takes(kind: &str, shapes: &[ShapeKind]) -> bool {
    let round = |k: &ShapeKind| matches!(k, ShapeKind::Arc | ShapeKind::Circle);
    // A spline is held tangent where one of its ends meets the other piece; the panel checks
    // for the joint (`crate::sketch_spline::spline_joint`).
    let curved = |k: &ShapeKind| round(k) || *k == ShapeKind::Spline;
    let lines = shapes.iter().all(|k| *k == ShapeKind::Line);
    match kind {
        "horizontal" | "vertical" => !shapes.is_empty() && lines,
        "parallel" | "perpendicular" | "angle" => shapes.len() == 2 && lines,
        "tangent" => shapes.len() == 2 && shapes.iter().any(curved) && shapes.iter().all(|k| *k == ShapeKind::Line || curved(k)) && !(shapes.contains(&ShapeKind::Circle) && shapes.contains(&ShapeKind::Spline)),
        "concentric" => shapes.len() == 2 && shapes.iter().all(round),
        "equal" => shapes.len() == 2 && (lines || shapes.iter().all(round)),
        "length" => shapes.len() == 1 && shapes[0] == ShapeKind::Line,
        "radius" | "diameter" => shapes.len() == 1 && round(&shapes[0]),
        _ => false,
    }
}

/// A regular polygon's corners counter-clockwise about `centre`: `at` is a corner for an
/// inscribed one, the middle of a side for a circumscribed one.
pub fn polygon_corners(centre: DVec2, at: DVec2, sides: usize, circumscribed: bool) -> Option<Vec<DVec2>> {
    if !(3..=64).contains(&sides) {
        return None;
    }
    let d = at - centre;
    let r = d.length();
    if !(r > 1e-9) {
        return None;
    }
    let half = PI / sides as f64;
    let radius = if circumscribed { r / half.cos() } else { r };
    let start = d.y.atan2(d.x) - if circumscribed { half } else { 0.0 };
    Some((0..sides).map(|i| arc_point(centre, radius, start + TAU * i as f64 / sides as f64)).collect())
}

/// The measured value a dimension would read on the sketch as it stands — the default when
/// one is added.
pub fn measure_rule(sketch: &Sketch, rule: &Rule, traces: &[crate::datums::DatumTrace]) -> Option<f64> {
    let shape = |id: EntityId| sketch.entity(id).map(|e| e.shape.clone());
    match *rule {
        Rule::Length { entity, .. } => match shape(entity)? {
            Shape::Line { a, b } => Some((b - a).length()),
            _ => None,
        },
        Rule::Radius { entity, .. } => shape(entity)?.radius(),
        Rule::Diameter { entity, .. } => shape(entity)?.radius().map(|r| 2.0 * r),
        Rule::Angle { a, b, .. } => {
            let (Shape::Line { a: a0, b: a1 }, Shape::Line { a: b0, b: b1 }) = (shape(a)?, shape(b)?) else { return None };
            let da = (a1 - a0).try_normalize()?;
            let db = (b1 - b0).try_normalize()?;
            let angle = (da.x * db.y - da.y * db.x).atan2(da.dot(db));
            Some(angle.rem_euclid(PI).to_degrees())
        }
        Rule::Distance { p, q, .. } => {
            let pp = sketch.anchor(p)?;
            match q {
                DistanceTo::Anchor(q) => Some((sketch.anchor(q)? - pp).length()),
                DistanceTo::Line(l) => {
                    let Shape::Line { a, b } = shape(l)? else { return None };
                    let d = (b - a).try_normalize()?;
                    Some((pp - a).dot(d.perp()).abs())
                }
                DistanceTo::Datum(plane) => crate::datums::trace_of(traces, plane).map(|t| t.offset(pp).abs()),
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axis_relations_within_three_degrees() {
        assert_eq!(axis_relation_of(DVec2::ZERO, DVec2::new(10.0, 0.5), INFER_AXIS_DEG), Some(AxisRelation::Horizontal));
        assert_eq!(axis_relation_of(DVec2::ZERO, DVec2::new(0.3, 10.0), INFER_AXIS_DEG), Some(AxisRelation::Vertical));
        assert_eq!(axis_relation_of(DVec2::ZERO, DVec2::new(10.0, 1.0), INFER_AXIS_DEG), None);
    }

    #[test]
    fn arc_through_three_points() {
        let (c, r, start, sweep, at_p) = arc_through(DVec2::new(1.0, 0.0), DVec2::new(-1.0, 0.0), DVec2::new(0.0, 1.0)).unwrap();
        assert!(c.length() < 1e-12 && (r - 1.0).abs() < 1e-12);
        assert!(at_p && start.abs() < 1e-12 && (sweep - PI).abs() < 1e-12);
    }

    #[test]
    fn inferred_relations_join_touching_ends() {
        let mut s = Sketch::default();
        s.add_shape(Shape::Line { a: DVec2::ZERO, b: DVec2::new(10.0, 0.0) });
        s.add_shape(Shape::Line { a: DVec2::new(10.0, 0.0), b: DVec2::new(10.0, 5.0) });
        assert_eq!(s.infer_relations(), 3);
        assert_eq!(s.infer_relations(), 0);
    }
}
