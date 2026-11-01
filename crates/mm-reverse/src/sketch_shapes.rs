// SPDX-License-Identifier: AGPL-3.0-only
//! The parts of the sketch model that `sketch` leaves to its callers: the regular polygon with
//! both of its construction circles handed back, the centre a centre-drawn rectangle carries,
//! the sketch pattern that repeats pieces with the relations they hold among themselves, the
//! relations a drawing implies told which pieces to touch, and the Fix shown as a relation.
//! Ported from ScanRuler (`plugins/reverse/core/sketch.ts`), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only.
//!
//! Everything here works on the `Sketch`, `Entity`, `Shape` and `Rule` of `crate::sketch`; it
//! adds pieces and relations the way `Sketch::add` and `Sketch::constrain` do, so ids keep
//! running over pieces and relations alike.

use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;

use glam::DVec2;
use serde::{Deserialize, Serialize};

use crate::sketch::{
    anchor_point, axis_relation_of, ends_of, polygon_corners, Anchor, AnchorAt, Constraint, ConstraintId, DistanceTo, Entity, EntityId, MirrorAxis, OnTarget, Rule, Shape, Sketch, TOUCH_MM,
};

/// The most sides a polygon is drawn with.
pub const POLYGON_MAX_SIDES: usize = 64;

/// How a polygon is sized against its construction circle: its corners on the circle
/// (inscribed), or its sides touching it (circumscribed) — the circle's diameter then the
/// width across flats, a hex nut's size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PolygonFit {
    Inscribed,
    Circumscribed,
}

/// What `add_polygon` made: the lines, from the corner clicked round (from the side clicked,
/// circumscribed); `circle`, the circle it is sized by — the corners' when inscribed, the inner
/// one when circumscribed; `rim`, the corners'.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolygonPieces {
    pub lines: Vec<EntityId>,
    pub circle: EntityId,
    pub rim: EntityId,
}

/// A relation added with the inferred flag as given — what `Sketch::constrain` and
/// `Sketch::infer` do, with the flag chosen by the caller.
fn push_rule(sketch: &mut Sketch, rule: Rule, inferred: bool) -> ConstraintId {
    let id = sketch.next_id;
    sketch.next_id += 1;
    sketch.constraints.push(Constraint { id, rule, inferred, expr: None, place: None });
    id
}

/// A line's unit direction, a to b — or `None` for a degenerate line (shorter than 1e-12).
pub fn line_dir(a: DVec2, b: DVec2) -> Option<DVec2> {
    let d = b - a;
    let l = d.x.hypot(d.y);
    if !(l > 1e-12) {
        return None;
    }
    Some(d / l)
}

/// A line's length.
pub fn line_length(a: DVec2, b: DVec2) -> f64 {
    (b.x - a.x).hypot(b.y - a.y)
}

/// A piece's Fix where the sketch shows its relations — a row in the list, a glyph on the
/// sheet, selected and removed as a relation is. The Fix itself is the piece's flag
/// (`fixed`), what the solver reads; it is shown under an id of its own that no relation has:
/// negative, from the piece's. A projected piece is fixed by the part, not by a Fix, and has
/// none.
pub fn fix_id(entity: EntityId) -> i64 {
    -1 - entity as i64
}

/// The piece whose Fix an id from `fix_id` stands for, or `None` for a relation's or a
/// dimension's own id.
pub fn fixed_entity_of(id: i64) -> Option<EntityId> {
    if id < 0 {
        EntityId::try_from(-1 - id).ok()
    } else {
        None
    }
}

/// The pieces fixed by a Fix, in the sketch's order.
pub fn fixed_pieces(sketch: &Sketch) -> Vec<&Entity> {
    sketch.entities.iter().filter(|e| e.fixed && !e.projected).collect()
}

/// A regular polygon as CAD draws one: its corners on a construction circle and its sides all
/// one length. Circumscribed, a second circle is drawn inside it, about the same centre and
/// touching a side — the width across flats, what it is then dimensioned by. (Every side held
/// tangent to that circle, as the obvious way has it, is singular where it is met: a tangency
/// and a side's middle on the circle say one thing to first order, and with equal sides alone
/// a polygon of an even count of them leans into alternating angles.) Its turn is left to a
/// relation or a dimension on a side. `None` with under three sides, over
/// `POLYGON_MAX_SIDES`, or no size.
pub fn add_polygon(sketch: &mut Sketch, centre: DVec2, at: DVec2, sides: usize, fit: PolygonFit) -> Option<PolygonPieces> {
    let corners = polygon_corners(centre, at, sides, fit == PolygonFit::Circumscribed)?;
    let n = corners.len();
    // The pieces first, sides then circles, so they take ids in a run.
    let lines: Vec<EntityId> = (0..n).map(|i| sketch.add(Entity::new(Shape::Line { a: corners[i], b: corners[(i + 1) % n] }))).collect();
    let rim = sketch.add(Entity::new(Shape::Circle { c: centre, r: (corners[0] - centre).length() }).construction());
    let flats = (fit == PolygonFit::Circumscribed).then(|| sketch.add(Entity::new(Shape::Circle { c: centre, r: (at - centre).length() }).construction()));
    for i in 0..n {
        sketch.constrain(Rule::Coincident { p: Anchor::new(lines[i], AnchorAt::B), q: Anchor::new(lines[(i + 1) % n], AnchorAt::A) });
        sketch.constrain(Rule::PointOn { p: Anchor::new(lines[i], AnchorAt::A), on: OnTarget::Curve(rim), bounded: false });
        if i > 0 {
            sketch.constrain(Rule::Equal { a: lines[0], b: lines[i] });
        }
    }
    let Some(flats) = flats else {
        return Some(PolygonPieces { lines, circle: rim, rim });
    };
    sketch.constrain(Rule::Concentric { a: flats, b: rim });
    sketch.constrain(Rule::Tangent { a: lines[0], b: flats });
    Some(PolygonPieces { lines, circle: flats, rim })
}

/// A rectangle's centre, as the construction a centre-drawn rectangle carries: its two
/// diagonals, end to corner, and a point where they cross — what the centre is held by and
/// dimensioned from. The rectangle's lines run round from `lines[0]`, each from its corner
/// (`A`). Returns the centre point's id.
pub fn add_rectangle_centre(sketch: &mut Sketch, lines: &[EntityId]) -> EntityId {
    let corner = |sketch: &Sketch, i: usize| -> DVec2 {
        match lines.get(i).and_then(|id| sketch.entity(*id)).map(|e| e.shape.clone()) {
            Some(Shape::Line { a, .. }) => a,
            _ => DVec2::ZERO,
        }
    };
    let line_at = |i: usize| lines.get(i).copied().unwrap_or(0);
    let mut diagonals = Vec::with_capacity(2);
    for (i, j) in [(0usize, 2usize), (1, 3)] {
        let (a, b) = (corner(sketch, i), corner(sketch, j));
        let id = sketch.add(Entity::new(Shape::Line { a, b }).construction());
        sketch.constrain(Rule::Coincident { p: Anchor::new(id, AnchorAt::A), q: Anchor::new(line_at(i), AnchorAt::A) });
        sketch.constrain(Rule::Coincident { p: Anchor::new(id, AnchorAt::B), q: Anchor::new(line_at(j), AnchorAt::A) });
        diagonals.push(id);
    }
    let a = corner(sketch, 0);
    let b = corner(sketch, 2);
    let centre = sketch.add(Entity::new(Shape::Point { p: (a + b) / 2.0 }).construction());
    for line in diagonals {
        sketch.constrain(Rule::PointOn { p: Anchor::new(centre, AnchorAt::Center), on: OnTarget::Line(line), bounded: false });
    }
    centre
}

/// How pieces are repeated: along one or two directions in steps — a row, a grid — or round a
/// centre.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SketchPatternSpec {
    Rectangular {
        /// The step from one copy to the next along the first direction.
        step: DVec2,
        /// How many there are in all along it, the originals counted.
        count: usize,
        /// A second direction, for a grid, and how many along it.
        step2: Option<DVec2>,
        count2: Option<usize>,
    },
    Circular {
        centre: DVec2,
        count: usize,
        /// The angle the copies span, degrees, counter-clockwise: 360 for a full turn, which
        /// then holds `count` evenly — one less spacing than an open span.
        angle_deg: f64,
    },
}

/// Where one copy goes: a turn about a centre, then a move.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Placement {
    shift: DVec2,
    turn: f64,
    about: DVec2,
}

/// Where every copy a pattern makes goes, the originals left out.
fn pattern_placements(spec: &SketchPatternSpec) -> Vec<Placement> {
    let mut out = Vec::new();
    match *spec {
        SketchPatternSpec::Rectangular { step, count, step2, count2 } => {
            let n1 = count.max(1);
            let n2 = if step2.is_some() { count2.unwrap_or(1).max(1) } else { 1 };
            for j in 0..n2 {
                for i in 0..n1 {
                    if i == 0 && j == 0 {
                        continue;
                    }
                    let v = step2.map_or(DVec2::ZERO, |s| s * j as f64);
                    out.push(Placement { shift: step * i as f64 + v, turn: 0.0, about: DVec2::ZERO });
                }
            }
        }
        SketchPatternSpec::Circular { centre, count, angle_deg } => {
            let n = count.max(1);
            let full = (angle_deg.abs() - 360.0).abs() < 1e-9;
            let step = if n > 1 { (angle_deg * PI) / 180.0 / if full { n as f64 } else { (n - 1) as f64 } } else { 0.0 };
            for i in 1..n {
                out.push(Placement { shift: DVec2::ZERO, turn: step * i as f64, about: centre });
            }
        }
    }
    out
}

/// A point carried by one of a pattern's placements.
fn place_point(p: DVec2, at: &Placement) -> DVec2 {
    let (sin, cos) = at.turn.sin_cos();
    let d = p - at.about;
    at.about + DVec2::new(d.x * cos - d.y * sin, d.x * sin + d.y * cos) + at.shift
}

/// A relation or a dimension naming other pieces: a copy of it with every piece it names put
/// through `id`.
pub fn renumbered(rule: &Rule, id: impl Fn(EntityId) -> EntityId) -> Rule {
    let anchor = |a: Anchor| Anchor { entity: id(a.entity), at: a.at };
    let on = |t: OnTarget| match t {
        OnTarget::Line(l) => OnTarget::Line(id(l)),
        OnTarget::Curve(c) => OnTarget::Curve(id(c)),
        OnTarget::Datum(d) => OnTarget::Datum(d),
    };
    match *rule {
        Rule::Coincident { p, q } => Rule::Coincident { p: anchor(p), q: anchor(q) },
        Rule::Horizontal { entity } => Rule::Horizontal { entity: id(entity) },
        Rule::Vertical { entity } => Rule::Vertical { entity: id(entity) },
        Rule::Parallel { a, b } => Rule::Parallel { a: id(a), b: id(b) },
        Rule::Perpendicular { a, b } => Rule::Perpendicular { a: id(a), b: id(b) },
        Rule::Tangent { a, b } => Rule::Tangent { a: id(a), b: id(b) },
        Rule::Concentric { a, b } => Rule::Concentric { a: id(a), b: id(b) },
        Rule::Equal { a, b } => Rule::Equal { a: id(a), b: id(b) },
        Rule::PointOn { p, on: t, bounded } => Rule::PointOn { p: anchor(p), on: on(t), bounded },
        Rule::Collinear { line, on: t } => Rule::Collinear { line: id(line), on: on(t) },
        Rule::Symmetric { p, q, about } => Rule::Symmetric {
            p: anchor(p),
            q: anchor(q),
            about: match about {
                MirrorAxis::Line(l) => MirrorAxis::Line(id(l)),
                other => other,
            },
        },
        Rule::Length { entity, value } => Rule::Length { entity: id(entity), value },
        Rule::Distance { p, q, value } => Rule::Distance {
            p: anchor(p),
            q: match q {
                DistanceTo::Anchor(a) => DistanceTo::Anchor(anchor(a)),
                DistanceTo::Line(l) => DistanceTo::Line(id(l)),
                DistanceTo::Datum(d) => DistanceTo::Datum(d),
            },
            value,
        },
        Rule::Radius { entity, value } => Rule::Radius { entity: id(entity), value },
        Rule::Diameter { entity, value } => Rule::Diameter { entity: id(entity), value },
        Rule::Angle { a, b, value } => Rule::Angle { a: id(a), b: id(b), value },
    }
}

/// Whether a relation ties its pieces to the sheet itself — a coordinate plane's line or a
/// sheet axis — rather than only to each other.
fn tied_to_sheet(rule: &Rule) -> bool {
    match rule {
        Rule::Distance { q: DistanceTo::Datum(_), .. } => true,
        Rule::PointOn { on: OnTarget::Datum(_), .. } | Rule::Collinear { on: OnTarget::Datum(_), .. } => true,
        Rule::Symmetric { about, .. } => !matches!(about, MirrorAxis::Line(_)),
        _ => false,
    }
}

/// Copies of the pieces given, as a pattern lays them: every copy a piece of its own, with
/// the relations and dimensions that hold the pieces among themselves copied onto it — a
/// copied rectangle is a rectangle of the same size, dimensioned the same, its dimensions
/// reading the same expressions. What ties a piece to anything outside the set (another
/// piece, a coordinate plane, a sheet axis, a Fix) stays with the original. Construction
/// pieces are copied as construction. Returns the ids of the new pieces, copy by copy, in the
/// order the pieces were given.
pub fn pattern_pieces(sketch: &mut Sketch, ids: &[EntityId], spec: &SketchPatternSpec) -> Vec<Vec<EntityId>> {
    let set: HashSet<EntityId> = ids.iter().copied().collect();
    let pieces: Vec<Entity> = sketch.entities.iter().filter(|e| set.contains(&e.id)).cloned().collect();
    if pieces.is_empty() {
        return Vec::new();
    }
    let inner: Vec<Constraint> = sketch
        .constraints
        .iter()
        .filter(|c| crate::sketch::constraint_entities(&c.rule).iter().all(|id| set.contains(id)) && !tied_to_sheet(&c.rule))
        .cloned()
        .collect();
    let mut copies = Vec::new();
    for at in pattern_placements(spec) {
        let place = |p: DVec2| place_point(p, &at);
        let mut map: HashMap<EntityId, EntityId> = HashMap::new();
        for e in &pieces {
            let shape = match e.shape {
                Shape::Line { a, b } => Shape::Line { a: place(a), b: place(b) },
                Shape::Arc { c, r, start, sweep } => Shape::Arc { c: place(c), r, start: start + at.turn, sweep },
                Shape::Circle { c, r } => Shape::Circle { c: place(c), r },
                Shape::Point { p } => Shape::Point { p: place(p) },
                Shape::Spline(ref sp) => Shape::Spline(crate::sketch_spline::spline_placed(sp, place, at.turn)),
            };
            let mut made = Entity::new(shape);
            made.construction = e.construction;
            map.insert(e.id, sketch.add(made));
        }
        let id = |old: EntityId| map.get(&old).copied().unwrap_or(old);
        for c in &inner {
            let next = sketch.next_id;
            sketch.next_id += 1;
            sketch.constraints.push(Constraint { id: next, rule: renumbered(&c.rule, id), inferred: c.inferred, expr: c.expr.clone(), place: None });
        }
        copies.push(pieces.iter().map(|e| id(e.id)).collect());
    }
    copies
}

/// The relations a sketch drawn by hand would have been given as it was drawn, put on pieces
/// handed in as coordinates: a coincidence where two ends meet — the ends of lines and arcs
/// within `TOUCH_MM` of each other, joined only where no chain of coincidences joins them
/// already — and horizontal or vertical on a line lying exactly level or upright that has
/// neither. With `pieces`, only what touches one of those. A line in `shaped` is a side of a
/// shape its own relations hold — a polygon's, a rectangle's — which a level of its own would
/// say twice. Marked inferred, as the panel marks its own. Returns how many were added.
pub fn infer_relations(sketch: &mut Sketch, pieces: Option<&HashSet<EntityId>>, shaped: Option<&HashSet<EntityId>>) -> usize {
    let mut ends: Vec<(Anchor, DVec2)> = Vec::new();
    for e in &sketch.entities {
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
    for c in &sketch.constraints {
        if let Rule::Coincident { p, q } = c.rule {
            let (rp, rq) = (find(&mut parent, p), find(&mut parent, q));
            parent.insert(rp, rq);
        }
    }
    // Ends bucketed by where they lie, so a sketch of thousands of pieces compares neighbours
    // only.
    let size = TOUCH_MM * 4.0;
    let cell = |p: DVec2| ((p.x / size).floor() as i64, (p.y / size).floor() as i64);
    let mut grid: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for (i, (_, p)) in ends.iter().enumerate() {
        grid.entry(cell(*p)).or_default().push(i);
    }
    let mut added = 0;
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
                    if fa.entity == ea.entity {
                        continue;
                    }
                    if let Some(only) = pieces {
                        if !only.contains(&ea.entity) && !only.contains(&fa.entity) {
                            continue;
                        }
                    }
                    if (ep - fp).x.hypot((ep - fp).y) > TOUCH_MM {
                        continue;
                    }
                    if find(&mut parent, ea) == find(&mut parent, fa) {
                        continue;
                    }
                    push_rule(sketch, Rule::Coincident { p: ea, q: fa }, true);
                    let (ra, rb) = (find(&mut parent, ea), find(&mut parent, fa));
                    parent.insert(ra, rb);
                    added += 1;
                }
            }
        }
    }
    let lines: Vec<(EntityId, DVec2, DVec2)> = sketch
        .entities
        .iter()
        .filter(|e| !e.construction && pieces.is_none_or(|p| p.contains(&e.id)) && !shaped.is_some_and(|s| s.contains(&e.id)))
        .filter_map(|e| if let Shape::Line { a, b } = e.shape { Some((e.id, a, b)) } else { None })
        .collect();
    for (id, a, b) in lines {
        let has = sketch.constraints.iter().any(|c| matches!(c.rule, Rule::Horizontal { entity } | Rule::Vertical { entity } if entity == id));
        if has {
            continue;
        }
        if let Some(axis) = axis_relation_of(a, b, 1e-6) {
            push_rule(sketch, axis.rule(id), true);
            added += 1;
        }
    }
    added
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datums::{datum_traces, DatumPlane};
    use crate::frame::SectionFrame;
    use crate::solver::{solve_sketch, SolveOptions};
    use glam::DVec3;

    fn near(a: DVec2, b: DVec2, digits: i32) {
        let tol = 0.5 * 10f64.powi(-digits);
        assert!((a.x - b.x).abs() < tol && (a.y - b.y).abs() < tol, "{a} vs {b}");
    }

    fn xy() -> SectionFrame {
        SectionFrame { origin: DVec3::ZERO, normal: DVec3::Z, u: DVec3::X, v: DVec3::Y }
    }

    fn line_of(s: &Sketch, id: EntityId) -> (DVec2, DVec2) {
        match s.entity(id).map(|e| e.shape.clone()) {
            Some(Shape::Line { a, b }) => (a, b),
            other => panic!("not a line: {other:?}"),
        }
    }

    #[test]
    fn the_regular_polygon_stays_regular_as_its_circle_is_dimensioned() {
        for fit in [PolygonFit::Inscribed, PolygonFit::Circumscribed] {
            let mut s = Sketch::default();
            let made = add_polygon(&mut s, DVec2::new(3.0, 4.0), DVec2::new(9.0, 4.0), 6, fit).unwrap();
            s.constrain(Rule::Radius { entity: made.circle, value: 8.0 });
            let report = solve_sketch(&mut s, SolveOptions::default());
            assert!(report.ok, "{fit:?} {report:?}");
            let want = if fit == PolygonFit::Inscribed { 8.0 } else { (2.0 * 8.0) / 3f64.sqrt() };
            for id in &made.lines {
                let (a, b) = line_of(&s, *id);
                assert!((line_length(a, b) - want).abs() < 5e-5, "{fit:?} side {}", line_length(a, b));
            }
        }
    }

    #[test]
    fn the_regular_polygon_is_fully_defined_by_its_centre_circle_and_one_sides_turn() {
        let traces = datum_traces(&xy());
        for fit in [PolygonFit::Inscribed, PolygonFit::Circumscribed] {
            for sides in [3, 5, 6, 8] {
                let mut s = Sketch::default();
                let PolygonPieces { lines, circle, .. } = add_polygon(&mut s, DVec2::ZERO, DVec2::new(5.0, 0.0), sides, fit).unwrap();
                s.constrain(Rule::PointOn { p: Anchor::new(circle, AnchorAt::Center), on: OnTarget::Datum(DatumPlane::Yz), bounded: false });
                s.constrain(Rule::PointOn { p: Anchor::new(circle, AnchorAt::Center), on: OnTarget::Datum(DatumPlane::Zx), bounded: false });
                s.constrain(Rule::Radius { entity: circle, value: 5.0 });
                let loose = solve_sketch(&mut s, SolveOptions { datums: &traces, ..Default::default() });
                assert!(loose.ok, "{fit:?} {sides}");
                assert_eq!(loose.redundant, 0, "{fit:?} {sides}");
                // The turn is all that is left: a corner on the X axis pins it.
                assert!(!loose.under_constrained.is_empty(), "{fit:?} {sides}");
                s.constrain(Rule::PointOn { p: Anchor::new(lines[0], AnchorAt::A), on: OnTarget::Datum(DatumPlane::Zx), bounded: false });
                let pinned = solve_sketch(&mut s, SolveOptions { datums: &traces, ..Default::default() });
                assert!(pinned.ok, "{fit:?} {sides}");
                assert_eq!(pinned.redundant, 0, "{fit:?} {sides}");
                assert_eq!(pinned.under_constrained, Vec::<EntityId>::new(), "{fit:?} {sides}");
            }
        }
    }

    #[test]
    fn the_pattern_lays_a_grid_of_copies_each_with_the_relations_and_dimensions_of_the_original() {
        let mut s = Sketch::default();
        let rect = s.add_rectangle(DVec2::ZERO, DVec2::new(4.0, 2.0));
        let dim = s.constrain(Rule::Length { entity: rect[0], value: 4.0 });
        if let Some(c) = s.constraint_mut(dim) {
            c.expr = Some("w".into());
        }
        let copies = pattern_pieces(&mut s, &rect, &SketchPatternSpec::Rectangular { step: DVec2::new(10.0, 0.0), count: 3, step2: Some(DVec2::new(0.0, 5.0)), count2: Some(2) });
        assert_eq!(copies.len(), 5);
        assert_eq!(s.entities.len(), 24);
        // Each copy is a rectangle: four corners, four axis relations, its length dimension
        // reading the same expression.
        for ids in &copies {
            let own: Vec<&Constraint> = s
                .constraints
                .iter()
                .filter(|c| match c.rule {
                    Rule::Coincident { p, .. } => ids.contains(&p.entity),
                    Rule::Horizontal { entity } | Rule::Vertical { entity } | Rule::Length { entity, .. } | Rule::Radius { entity, .. } | Rule::Diameter { entity, .. } => ids.contains(&entity),
                    _ => false,
                })
                .collect();
            assert_eq!(own.iter().filter(|c| matches!(c.rule, Rule::Coincident { .. })).count(), 4);
            assert_eq!(own.iter().filter(|c| matches!(c.rule, Rule::Horizontal { .. } | Rule::Vertical { .. })).count(), 4);
            let length = own.iter().find(|c| matches!(c.rule, Rule::Length { .. })).unwrap();
            assert_eq!(length.expr.as_deref(), Some("w"));
            assert_ne!(length.id, dim);
        }
        near(line_of(&s, copies[1][0]).0, DVec2::new(20.0, 0.0), 6);
        near(line_of(&s, copies[4][0]).0, DVec2::new(20.0, 5.0), 6);
        assert!(solve_sketch(&mut s, SolveOptions::default()).ok);
    }

    #[test]
    fn the_pattern_turns_copies_round_a_centre_spreading_a_full_turn_evenly() {
        let mut s = Sketch::default();
        let hole = s.add_shape(Shape::Circle { c: DVec2::new(10.0, 0.0), r: 1.5 });
        let copies = pattern_pieces(&mut s, &[hole], &SketchPatternSpec::Circular { centre: DVec2::ZERO, count: 4, angle_deg: 360.0 });
        assert_eq!(copies.len(), 3);
        near(s.entity(copies[0][0]).unwrap().shape.centre().unwrap(), DVec2::new(0.0, 10.0), 6);
        near(s.entity(copies[1][0]).unwrap().shape.centre().unwrap(), DVec2::new(-10.0, 0.0), 6);
        // An open span puts the last copy at its end.
        let mut t = Sketch::default();
        let p = t.add_shape(Shape::Point { p: DVec2::new(5.0, 0.0) });
        let open = pattern_pieces(&mut t, &[p], &SketchPatternSpec::Circular { centre: DVec2::ZERO, count: 3, angle_deg: 90.0 });
        near(anchor_point(&t.entity(open[1][0]).unwrap().shape, AnchorAt::Center), DVec2::new(0.0, 5.0), 6);
    }

    #[test]
    fn the_pattern_leaves_what_ties_the_pieces_to_the_axes_or_to_other_pieces_with_the_original() {
        let mut s = Sketch::default();
        let other = s.add_shape(Shape::Line { a: DVec2::new(0.0, -5.0), b: DVec2::new(10.0, -5.0) });
        let [bottom, ..] = s.add_rectangle(DVec2::ZERO, DVec2::new(4.0, 2.0));
        s.constrain(Rule::Distance { p: Anchor::new(bottom, AnchorAt::A), q: DistanceTo::Datum(DatumPlane::Yz), value: 0.0 });
        s.constrain(Rule::Parallel { a: bottom, b: other });
        let before = s.constraints.len();
        let ids: Vec<EntityId> = s.entities.iter().filter(|e| e.id != other).map(|e| e.id).collect();
        pattern_pieces(&mut s, &ids, &SketchPatternSpec::Rectangular { step: DVec2::new(8.0, 0.0), count: 2, step2: None, count2: None });
        // The copy has its rectangle's eight, not the distance or the parallel.
        assert_eq!(s.constraints.len() - before, 8);
    }

    #[test]
    fn inferred_relations_join_the_ends_that_meet_and_make_level_lines_horizontal_once() {
        let mut s = Sketch::default();
        let pts = [DVec2::new(0.0, 0.0), DVec2::new(10.0, 0.0), DVec2::new(10.0, 6.0), DVec2::new(3.0, 9.0)];
        for i in 0..pts.len() {
            s.add_shape(Shape::Line { a: pts[i], b: pts[(i + 1) % pts.len()] });
        }
        assert_eq!(infer_relations(&mut s, None, None), 4 + 2);
        assert_eq!(s.constraints.iter().filter(|c| matches!(c.rule, Rule::Coincident { .. })).count(), 4);
        assert_eq!(s.constraints.iter().filter(|c| matches!(c.rule, Rule::Horizontal { .. })).count(), 1);
        assert_eq!(s.constraints.iter().filter(|c| matches!(c.rule, Rule::Vertical { .. })).count(), 1);
        assert!(s.constraints.iter().all(|c| c.inferred));
        assert_eq!(infer_relations(&mut s, None, None), 0);
        assert!(solve_sketch(&mut s, SolveOptions::default()).ok);
    }

    #[test]
    fn inferred_relations_leave_what_a_chain_of_coincidences_joins_already_and_ends_apart() {
        let mut s = Sketch::default();
        let a = s.add_shape(Shape::Line { a: DVec2::ZERO, b: DVec2::new(5.0, 1.0) });
        let b = s.add_shape(Shape::Line { a: DVec2::new(5.0, 1.0), b: DVec2::new(9.0, 4.0) });
        let c = s.add_shape(Shape::Line { a: DVec2::new(5.0, 1.0), b: DVec2::new(1.0, 7.0) });
        s.constrain(Rule::Coincident { p: Anchor::new(a, AnchorAt::B), q: Anchor::new(b, AnchorAt::A) });
        s.constrain(Rule::Coincident { p: Anchor::new(b, AnchorAt::A), q: Anchor::new(c, AnchorAt::A) });
        s.add_shape(Shape::Line { a: DVec2::new(20.0, 0.0), b: DVec2::new(20.01, 3.0) });
        assert_eq!(infer_relations(&mut s, None, None), 0);
    }

    #[test]
    fn inferred_relations_touch_only_the_pieces_named() {
        let mut s = Sketch::default();
        s.add_rectangle(DVec2::ZERO, DVec2::new(4.0, 4.0));
        let loose = s.add_shape(Shape::Line { a: DVec2::new(4.0, 4.0), b: DVec2::new(9.0, 4.0) });
        let before = s.constraints.len();
        assert_eq!(infer_relations(&mut s, Some(&HashSet::from([loose])), None), 2);
        assert_eq!(s.constraints.len() - before, 2);
    }

    #[test]
    fn a_centre_drawn_rectangle_carries_its_centre_where_its_diagonals_cross() {
        let mut s = Sketch::default();
        let lines = s.add_rectangle(DVec2::new(-10.0, -6.0), DVec2::new(10.0, 6.0));
        let centre = add_rectangle_centre(&mut s, &lines);
        let e = s.entity(centre).unwrap();
        assert!(e.construction);
        near(anchor_point(&e.shape, AnchorAt::Center), DVec2::ZERO, 9);
        assert_eq!(s.entities.iter().filter(|e| e.construction && matches!(e.shape, Shape::Line { .. })).count(), 2);
        s.constrain(Rule::Length { entity: lines[0], value: 30.0 });
        assert!(solve_sketch(&mut s, SolveOptions::default()).ok);
        let p = anchor_point(&s.entity(centre).unwrap().shape, AnchorAt::Center);
        let (a, b) = line_of(&s, lines[0]);
        let (c, _) = line_of(&s, lines[2]);
        near(p, (a + c) / 2.0, 6);
        assert!((line_length(a, b) - 30.0).abs() < 1e-6);
    }

    #[test]
    fn a_fix_is_shown_under_an_id_no_relation_has() {
        assert_eq!(fix_id(7), -8);
        assert_eq!(fixed_entity_of(-8), Some(7));
        assert_eq!(fixed_entity_of(12), None);
        let mut s = Sketch::default();
        let a = s.add_shape(Shape::Point { p: DVec2::ZERO });
        let b = s.add_shape(Shape::Point { p: DVec2::ONE });
        s.entity_mut(a).unwrap().fixed = true;
        s.entity_mut(b).unwrap().fixed = true;
        s.entity_mut(b).unwrap().projected = true;
        assert_eq!(fixed_pieces(&s).iter().map(|e| e.id).collect::<Vec<_>>(), vec![a]);
    }

    #[test]
    fn the_equal_relation_takes_two_lines_or_two_round_pieces_and_nothing_mixed() {
        use crate::sketch::{relation_takes, ShapeKind};
        assert!(relation_takes("equal", &[ShapeKind::Line, ShapeKind::Line]));
        assert!(relation_takes("equal", &[ShapeKind::Circle, ShapeKind::Arc]));
        assert!(!relation_takes("equal", &[ShapeKind::Line, ShapeKind::Circle]));
    }

    #[test]
    fn the_equal_relation_makes_two_lines_one_length_and_two_circles_one_radius_a_dimension_on_one_driving_both() {
        let mut s = Sketch::default();
        let l1 = s.add_shape(Shape::Line { a: DVec2::new(0.0, 0.0), b: DVec2::new(10.0, 0.0) });
        let l2 = s.add_shape(Shape::Line { a: DVec2::new(0.0, 5.0), b: DVec2::new(4.0, 5.0) });
        let c1 = s.add_shape(Shape::Circle { c: DVec2::new(20.0, 0.0), r: 3.0 });
        let c2 = s.add_shape(Shape::Circle { c: DVec2::new(30.0, 0.0), r: 1.0 });
        s.constrain(Rule::Equal { a: l1, b: l2 });
        s.constrain(Rule::Equal { a: c1, b: c2 });
        s.constrain(Rule::Length { entity: l1, value: 12.0 });
        s.constrain(Rule::Radius { entity: c2, value: 2.5 });
        assert!(solve_sketch(&mut s, SolveOptions::default()).ok);
        let Some(Shape::Line { a, b }) = s.entity(l2).map(|e| e.shape.clone()) else { panic!("a line") };
        assert!((line_length(a, b) - 12.0).abs() < 5e-6);
        assert!((s.entity(c1).and_then(|e| e.shape.radius()).unwrap_or(f64::NAN) - 2.5).abs() < 5e-6);
    }

    #[test]
    fn the_regular_polygon_puts_an_inscribed_ones_corners_on_the_circle_through_the_corner_clicked() {
        let corners = polygon_corners(DVec2::ZERO, DVec2::new(10.0, 0.0), 6, false).expect("corners");
        assert_eq!(corners.len(), 6);
        near(corners[0], DVec2::new(10.0, 0.0), 6);
        for c in &corners {
            assert!((c.length() - 10.0).abs() < 5e-10);
        }
    }

    #[test]
    fn the_regular_polygon_sizes_a_circumscribed_one_across_its_flats() {
        let corners = polygon_corners(DVec2::ZERO, DVec2::new(6.5, 0.0), 6, true).expect("corners");
        // A hex 13 across flats: the side clicked runs upright through x = 6.5.
        assert_eq!(corners.iter().filter(|c| (c.x - 6.5).abs() < 1e-9).count(), 2);
        assert!((corners[0].length() - 6.5 / (PI / 6.0).cos()).abs() < 5e-10);
    }

    #[test]
    fn the_regular_polygon_refuses_under_three_sides_or_with_no_size() {
        assert_eq!(polygon_corners(DVec2::ZERO, DVec2::new(1.0, 0.0), 2, false), None);
        assert_eq!(polygon_corners(DVec2::ZERO, DVec2::ZERO, 6, false), None);
    }
}
