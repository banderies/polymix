// SPDX-License-Identifier: AGPL-3.0-only
//! Pieces of one sketch that nearly coincide with pieces of another on a parallel plane, and
//! the joins of one sketch that run on or turn a little. Ported from ScanRuler
//! (`plugins/reverse/core/sketchNearMiss.ts`), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! A wall fitted again in a second plan sketch, 0.07 mm off where the first has it, a round
//! drawn twice at R0.07 and R0.08: built, each is a step between the two features; said when
//! the second sketch is made, it is one piece projected, or one level fitted into the sketch
//! that has the wall already.
//!
//! The other sketch's pieces are laid on this sheet: on parallel planes u and v are the same,
//! whatever the distance between them.

use std::collections::HashSet;
use std::f64::consts::PI;

use glam::DVec2;

use crate::frame::SectionFrame;
use crate::model::{Feature, FeatureId, FeatureKind};
use crate::sketch::{arc_point, EntityId, Rule, Shape, ShapeKind, Sketch};

/// How near two ends lie to be one join, mm.
const JOIN_MM: f64 = 1e-3;
/// Two lines within this of one direction, degrees, are parallel.
const LINE_DEG: f64 = 0.5;
/// Of the shorter line, the share the two must run side by side.
const LINE_OVERLAP: f64 = 0.3;
/// Nearer than this is one piece — projected, or drawn on the other.
const SAME_MM: f64 = 1e-4;
/// A near miss is up to this far off, mm.
const NEAR_MM: f64 = 0.3;
/// Planes within this of parallel, degrees.
const PLANE_DEG: f64 = 0.5;
/// How many near misses an answer lists.
const LISTED: usize = 12;
/// How far off one line two lines may run and still be said to be one, as a sine: about 0.05°.
const ONE_LINE_SIN: f64 = 1e-3;
/// Under this turn, degrees, a join is a line drawn straight; over it, up to `KINK_MAX_DEG`, a
/// crease the fit left.
const KINK_MIN_DEG: f64 = 0.2;
/// Over this, a corner someone meant, degrees — as far as a round's failure names a shallow
/// crease.
const KINK_MAX_DEG: f64 = 5.0;

/// A sketch feature as these checks read it: its id and name, the frame its sheet lies on, and
/// its pieces.
#[derive(Debug, Clone, Copy)]
pub struct SheetSketch<'a> {
    pub id: FeatureId,
    pub name: &'a str,
    pub frame: SectionFrame,
    pub sketch: &'a Sketch,
}

impl<'a> SheetSketch<'a> {
    /// A feature of the part read as a sheet sketch; `None` for anything but a sketch.
    pub fn of(feature: &'a Feature) -> Option<Self> {
        match &feature.kind {
            FeatureKind::Sketch(s) => Some(Self { id: feature.id, name: &feature.name, frame: s.frame, sketch: &s.sketch }),
            _ => None,
        }
    }
}

/// A piece of this sketch that nearly coincides with one of another's.
#[derive(Debug, Clone, PartialEq)]
pub struct NearMiss {
    /// This sketch's piece, by id.
    pub piece: EntityId,
    /// The other sketch, by name.
    pub sketch: String,
    /// Its piece, by id.
    pub other: EntityId,
    /// How far apart the two lie, mm: a line's offset where they run side by side; for a round
    /// the centres' offset and the radii's difference together.
    pub off_mm: f64,
}

/// The near misses found: the nearest `LISTED`, and how many in all.
#[derive(Debug, Clone, PartialEq)]
pub struct NearMisses {
    pub misses: Vec<NearMiss>,
    pub found: usize,
}

/// `Math.round` as JavaScript has it: halves round up, towards +∞.
fn js_round(x: f64) -> f64 {
    let f = x.floor();
    if x - f >= 0.5 {
        f + 1.0
    } else {
        f
    }
}

fn round5(x: f64) -> f64 {
    js_round(x * 1e5) / 1e5
}

/// A sketch point on another sketch's sheet.
fn on_sheet(from: &SectionFrame, to: &SectionFrame, p: DVec2) -> DVec2 {
    let q = from.to_world(p) - to.origin;
    DVec2::new(q.dot(to.u), q.dot(to.v))
}

fn d2(a: DVec2, b: DVec2) -> f64 {
    (a.x - b.x).hypot(a.y - b.y)
}

/// How far apart two lines on one sheet lie where they run side by side, or `None` when they
/// are not near parallel or do not run side by side enough.
fn line_off(a: (DVec2, DVec2), b: (DVec2, DVec2)) -> Option<f64> {
    let (la, lb) = (d2(a.0, a.1), d2(b.0, b.1));
    if la < 1e-9 || lb < 1e-9 {
        return None;
    }
    let u = DVec2::new((a.1.x - a.0.x) / la, (a.1.y - a.0.y) / la);
    let w = DVec2::new((b.1.x - b.0.x) / lb, (b.1.y - b.0.y) / lb);
    let sin = (u.x * w.y - u.y * w.x).abs();
    if sin > ((LINE_DEG * PI) / 180.0).sin() {
        return None;
    }
    let along = |p: DVec2| (p.x - a.0.x) * u.x + (p.y - a.0.y) * u.y;
    let across = |p: DVec2| ((p.x - a.0.x) * -u.y + (p.y - a.0.y) * u.x).abs();
    let (t0, t1) = (along(b.0), along(b.1));
    let (lo, hi) = (t0.min(t1).max(0.0), la.min(t0.max(t1)));
    if hi - lo < LINE_OVERLAP * la.min(lb) {
        return None;
    }
    // Across, at both ends of the run side by side.
    let at = |t: f64| {
        let s = (t - t0) / (t1 - t0);
        DVec2::new(b.0.x + (b.1.x - b.0.x) * s, b.0.y + (b.1.y - b.0.y) * s)
    };
    Some((across(at(lo)) + across(at(hi))) / 2.0)
}

/// Whether two sketches lie on parallel planes.
fn parallel(a: &SectionFrame, b: &SectionFrame) -> bool {
    a.normal.cross(b.normal).length() <= ((PLANE_DEG * PI) / 180.0).sin()
}

/// The pieces of `sketch` that nearly coincide with a piece of another sketch on a parallel
/// plane, laid on this sheet: lines within half a degree of each other that run side by side
/// for 30 % of the shorter or more, 1e-4 < offset ≤ 0.3 mm; arcs and circles with centres
/// within 0.3 mm and radii within 0.3 mm of each other, not one. Construction pieces are left
/// out on both sides. The nearest first, and how many in all.
pub fn sketch_near_misses(sketch: &SheetSketch, others: &[SheetSketch]) -> NearMisses {
    let mut out: Vec<NearMiss> = Vec::new();
    let mine: Vec<_> = sketch.sketch.entities.iter().filter(|e| !e.construction).collect();
    for o in others {
        if o.id == sketch.id || !parallel(&sketch.frame, &o.frame) {
            continue;
        }
        for e in o.sketch.entities.iter().filter(|e| !e.construction) {
            match e.shape {
                Shape::Line { a, b } => {
                    let b = (on_sheet(&o.frame, &sketch.frame, a), on_sheet(&o.frame, &sketch.frame, b));
                    for m in &mine {
                        let Shape::Line { a: ma, b: mb } = m.shape else { continue };
                        if let Some(off) = line_off((ma, mb), b) {
                            if off > SAME_MM && off <= NEAR_MM {
                                out.push(NearMiss { piece: m.id, sketch: o.name.to_string(), other: e.id, off_mm: round5(off) });
                            }
                        }
                    }
                }
                Shape::Arc { c, r, .. } | Shape::Circle { c, r } => {
                    let c = on_sheet(&o.frame, &sketch.frame, c);
                    for m in &mine {
                        let (Shape::Arc { c: mc, r: mr, .. } | Shape::Circle { c: mc, r: mr }) = m.shape else { continue };
                        let (dc, dr) = (d2(mc, c), (mr - r).abs());
                        if dc > NEAR_MM || dr > NEAR_MM || (dc <= SAME_MM && dr <= SAME_MM) {
                            continue;
                        }
                        out.push(NearMiss { piece: m.id, sketch: o.name.to_string(), other: e.id, off_mm: round5(dc + dr) });
                    }
                }
                Shape::Point { .. } | Shape::Spline(_) => {}
            }
        }
    }
    out.sort_by(|a, b| a.off_mm.total_cmp(&b.off_mm));
    let found = out.len();
    out.truncate(LISTED);
    NearMisses { misses: out, found }
}

/// The two ends of an arc.
fn arc_ends(c: DVec2, r: f64, start: f64, sweep: f64) -> [DVec2; 2] {
    [arc_point(c, r, start), arc_point(c, r, start + sweep)]
}

/// The ends of a line or an arc; none for anything else.
fn piece_ends(shape: &Shape) -> Vec<DVec2> {
    match *shape {
        Shape::Line { a, b } => vec![a, b],
        Shape::Arc { c, r, start, sweep } => arc_ends(c, r, start, sweep).to_vec(),
        _ => Vec::new(),
    }
}

fn near(p: DVec2, q: DVec2) -> bool {
    d2(p, q) < JOIN_MM
}

/// Two lines of a sketch end to end on one line, and where they join.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CollinearJoin {
    pub pieces: [EntityId; 2],
    pub at: DVec2,
}

/// Lines of a sketch that lie end to end on one line, with nothing else meeting at the join (a
/// third piece there makes the join a junction, kept for the regions). Each is two faces of a
/// profile where the viewer, which hides a split of one surface, draws one.
pub fn collinear_pieces(sketch: &Sketch) -> Vec<CollinearJoin> {
    let drawn: Vec<_> = sketch.entities.iter().filter(|e| !e.construction).collect();
    let lines: Vec<(EntityId, DVec2, DVec2)> = drawn.iter().filter_map(|e| if let Shape::Line { a, b } = e.shape { Some((e.id, a, b)) } else { None }).collect();
    let ends_at = |shape: &Shape, p: DVec2| piece_ends(shape).iter().any(|q| near(*q, p));
    let mut out = Vec::new();
    for i in 0..lines.len() {
        for j in i + 1..lines.len() {
            let (a_id, a_a, a_b) = lines[i];
            let (b_id, b_a, b_b) = lines[j];
            for (p, far) in [(a_a, a_b), (a_b, a_a)] {
                let other = if near(b_a, p) {
                    b_b
                } else if near(b_b, p) {
                    b_a
                } else {
                    continue;
                };
                let u = far - p;
                let v = other - p;
                let (lu, lv) = (u.x.hypot(u.y), v.x.hypot(v.y));
                if lu < JOIN_MM || lv < JOIN_MM {
                    continue;
                }
                // Opposite ways from the join, on one line; a fold back is no run.
                let on_one = (u.x * v.y - u.y * v.x).abs() / (lu * lv) < ONE_LINE_SIN && u.x * v.x + u.y * v.y < 0.0;
                if !on_one {
                    continue;
                }
                if drawn.iter().any(|e| e.id != a_id && e.id != b_id && ends_at(&e.shape, p)) {
                    continue;
                }
                out.push(CollinearJoin { pieces: [a_id, b_id], at: DVec2::new(js_round(p.x * 1e4) / 1e4, js_round(p.y * 1e4) / 1e4) });
            }
        }
    }
    out
}

/// A join of two pieces that turns a little: which, by how much, where.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Kink {
    pub pieces: [EntityId; 2],
    pub deg: f64,
    pub at: DVec2,
}

/// The way a line or an arc leaves one of its ends (0 its first, 1 its last), along itself.
fn away_from(shape: &Shape, end: usize) -> DVec2 {
    match *shape {
        Shape::Line { a, b } => {
            let (p, far) = if end == 0 { (a, b) } else { (b, a) };
            let l = (far.x - p.x).hypot(far.y - p.y);
            let l = if l == 0.0 || l.is_nan() { 1.0 } else { l };
            DVec2::new((far.x - p.x) / l, (far.y - p.y) / l)
        }
        Shape::Arc { start, sweep, .. } => {
            let t = start + end as f64 * sweep;
            let s = if sweep == 0.0 || sweep.is_nan() { 1.0 } else { sweep };
            let way = (if end == 0 { 1.0 } else { -1.0 }) * js_sign(s);
            DVec2::new(-t.sin() * way, t.cos() * way)
        }
        _ => DVec2::ZERO,
    }
}

/// `Math.sign`: −1, 0 or 1, and NaN for NaN.
fn js_sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        x
    }
}

/// Joins of two lines or arcs of a sketch that turn between `KINK_MIN_DEG` and `KINK_MAX_DEG`,
/// with nothing else meeting there: nearly straight, and no tangent chain runs on across them,
/// so a fillet along the wall stops or fails there. An angle dimensioned between the two is
/// meant.
pub fn kinked_joins(sketch: &Sketch) -> Vec<Kink> {
    let drawn: Vec<_> = sketch.entities.iter().filter(|e| !e.construction).collect();
    let pieces: Vec<_> = drawn.iter().filter(|e| matches!(e.shape, Shape::Line { .. } | Shape::Arc { .. })).collect();
    let end_of = |shape: &Shape, end: usize| -> DVec2 { piece_ends(shape).get(end).copied().unwrap_or(DVec2::NAN) };
    let meets = |shape: &Shape, p: DVec2| piece_ends(shape).iter().any(|q| near(*q, p));
    let meant = |a: EntityId, b: EntityId| sketch.constraints.iter().any(|c| matches!(c.rule, Rule::Angle { a: x, b: y, .. } if (x == a && y == b) || (x == b && y == a)));
    let mut out = Vec::new();
    for i in 0..pieces.len() {
        for j in i + 1..pieces.len() {
            let (a, b) = (pieces[i], pieces[j]);
            for ea in [0usize, 1] {
                for eb in [0usize, 1] {
                    let p = end_of(&a.shape, ea);
                    if !near(p, end_of(&b.shape, eb)) {
                        continue;
                    }
                    let (u, v) = (away_from(&a.shape, ea), away_from(&b.shape, eb));
                    // Straight on, the two leave the join opposite ways.
                    let deg = ((u.x * v.y - u.y * v.x).abs().atan2(-(u.x * v.x + u.y * v.y)) * 180.0) / PI;
                    if deg < KINK_MIN_DEG || deg > KINK_MAX_DEG || meant(a.id, b.id) {
                        continue;
                    }
                    if drawn.iter().any(|e| e.id != a.id && e.id != b.id && meets(&e.shape, p)) {
                        continue;
                    }
                    out.push(Kink { pieces: [a.id, b.id], deg: js_round(deg * 100.0) / 100.0, at: DVec2::new(js_round(p.x * 1e4) / 1e4, js_round(p.y * 1e4) / 1e4) });
                }
            }
        }
    }
    out
}

/// A string as JSON writes it, quoted and escaped.
fn json_str(s: &str) -> String {
    serde_json::to_string(s).unwrap_or_else(|_| format!("\"{s}\""))
}

/// What a kink does and the fix, in one line, by the first of them; `None` with no kinks.
pub fn kink_fix(sketch: &SheetSketch, kinks: &[Kink]) -> Option<String> {
    let first = kinks.first()?;
    let [a, b] = first.pieces;
    let line = |id: EntityId| sketch.sketch.entity(id).map(|e| e.kind()) == Some(ShapeKind::Line);
    let straight = if line(a) && line(b) { format!("{{ kind: \"collinear\", line: {a}, on: {{ line: {b} }} }}") } else { format!("{{ kind: \"tangent\", a: {a}, b: {b} }}") };
    Some(format!(
        "A join that turns this little is a crease in the wall: a tangent chain stops at it, so a fillet along the wall breaks there (“a crease of {}°”). Round it — reverse.sketch.edit {{ sketch: {}, add: {{ rounds: [{{ a: {a}, b: {b}, radius }}] }} }}, with a radius you give, wider than the fillet to come: a join this shallow has no round for the scan to read. Or, meant straight, hold it: add.constraints {straight}. Leave it only if the crease is the part's.",
        first.deg,
        json_str(sketch.name)
    ))
}

/// The fix in one line: the other sketch's pieces projected into this one, or this level
/// fitted into the other sketch (fit.level, mm along its normal from its plane).
pub fn near_miss_fix(sketch: &SheetSketch, misses: &[NearMiss], others: &[SheetSketch]) -> String {
    let mut by_sketch: Vec<(String, Vec<EntityId>)> = Vec::new();
    for m in misses {
        match by_sketch.iter_mut().find(|(name, _)| *name == m.sketch) {
            Some((_, ids)) => {
                if !ids.contains(&m.other) {
                    ids.push(m.other);
                }
            }
            None => by_sketch.push((m.sketch.clone(), vec![m.other])),
        }
    }
    let join = |ids: &[EntityId]| ids.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ");
    let project = by_sketch.iter().map(|(name, pieces)| format!("{{ sketch: {}, pieces: [{}] }}", json_str(name), join(pieces))).collect::<Vec<_>>().join(", ");
    let mut seen = HashSet::new();
    let own: Vec<EntityId> = misses.iter().map(|m| m.piece).filter(|p| seen.insert(*p)).collect();
    let first = by_sketch.first().map_or("undefined", |(name, _)| name.as_str());
    let other = others.iter().find(|o| o.name == first);
    let fit = match other {
        Some(o) => {
            // Plus zero: a level of nothing reads 0, as JavaScript writes a negative zero.
            let level = js_round((sketch.frame.origin - o.frame.origin).dot(o.frame.normal) * 1000.0) / 1000.0 + 0.0;
            format!("reverse.sketch.edit {{ sketch: {}, add: {{ fit: {{ level: {} }} }} }}", json_str(o.name), level)
        }
        None => format!("fit.level into {first}"),
    };
    format!(
        "a piece drawn twice is a step when built: project the other sketch’s in place of yours (reverse.sketch.edit {{ sketch: {}, remove: {{ pieces: [{}] }}, add: {{ project: [{project}] }} }}), or fit this level into {first} instead ({fit})",
        json_str(sketch.name),
        join(&own)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sketch::{Constraint, Entity};
    use glam::DVec3;

    fn line(id: EntityId, a: [f64; 2], b: [f64; 2]) -> Entity {
        let mut e = Entity::new(Shape::Line { a: DVec2::from(a), b: DVec2::from(b) });
        e.id = id;
        e
    }

    fn sketch_of(entities: Vec<Entity>) -> Sketch {
        Sketch { entities, constraints: Vec::new(), next_id: 99 }
    }

    #[test]
    fn lines_end_to_end_not_those_with_a_third_piece_at_the_join_nor_a_fold() {
        assert_eq!(collinear_pieces(&sketch_of(vec![line(1, [0.0, 0.0], [5.0, 0.0]), line(2, [5.0, 0.0], [9.0, 0.0])])), vec![CollinearJoin { pieces: [1, 2], at: DVec2::new(5.0, 0.0) }]);
        assert_eq!(collinear_pieces(&sketch_of(vec![line(1, [0.0, 0.0], [5.0, 0.0]), line(2, [5.0, 0.0], [9.0, 0.0]), line(3, [5.0, 0.0], [5.0, 4.0])])), vec![]);
        assert_eq!(collinear_pieces(&sketch_of(vec![line(1, [0.0, 0.0], [5.0, 0.0]), line(2, [5.0, 0.0], [1.0, 0.0])])), vec![]);
        assert_eq!(collinear_pieces(&sketch_of(vec![line(1, [0.0, 0.0], [5.0, 0.0]), line(2, [5.0, 0.0], [9.0, 1.0])])), vec![]);
    }

    #[test]
    fn reads_joins_that_turn_0_2_to_5_degrees_line_to_line_and_line_to_arc() {
        let turn = |deg: f64| [10.0 + 10.0 * ((deg * PI) / 180.0).cos(), 10.0 * ((deg * PI) / 180.0).sin()];
        let kinked = kinked_joins(&sketch_of(vec![line(1, [0.0, 0.0], [10.0, 0.0]), line(2, [10.0, 0.0], turn(2.16))]));
        assert_eq!(kinked, vec![Kink { pieces: [1, 2], deg: 2.16, at: DVec2::new(10.0, 0.0) }]);
        assert_eq!(kinked_joins(&sketch_of(vec![line(1, [0.0, 0.0], [10.0, 0.0]), line(2, [10.0, 0.0], turn(0.1))])), vec![]);
        assert_eq!(kinked_joins(&sketch_of(vec![line(1, [0.0, 0.0], [10.0, 0.0]), line(2, [10.0, 0.0], turn(10.0))])), vec![]);
        assert_eq!(kinked_joins(&sketch_of(vec![line(1, [0.0, 0.0], [10.0, 0.0]), line(2, [10.0, 0.0], turn(3.0)), line(3, [10.0, 0.0], [10.0, 5.0])])), vec![]);
        let mut meant = sketch_of(vec![line(1, [0.0, 0.0], [10.0, 0.0]), line(2, [10.0, 0.0], turn(3.0))]);
        meant.next_id = 9;
        meant.constraints.push(Constraint { id: 5, rule: Rule::Angle { a: 1, b: 2, value: 3.0 }, inferred: false, expr: None, place: None });
        assert_eq!(kinked_joins(&meant), vec![]);
        // An arc leaving the line's end 3° off its tangent, whichever way it is drawn.
        let off = (3.0 * PI) / 180.0;
        let c = DVec2::new(10.0 - 5.0 * off.sin(), 5.0 * off.cos());
        let start = (-c.y).atan2(10.0 - c.x);
        let mut arc = Entity::new(Shape::Arc { c, r: 5.0, start, sweep: 1.0 });
        arc.id = 2;
        let found = kinked_joins(&sketch_of(vec![line(1, [0.0, 0.0], [10.0, 0.0]), arc]));
        assert_eq!(found.first().map(|k| (k.pieces, k.deg)), Some(([1, 2], 3.0)));
        // Tangent: no kink.
        let mut tangent = Entity::new(Shape::Arc { c: DVec2::new(10.0, 5.0), r: 5.0, start: -PI / 2.0, sweep: 1.0 });
        tangent.id = 2;
        assert_eq!(kinked_joins(&sketch_of(vec![line(1, [0.0, 0.0], [10.0, 0.0]), tangent])), vec![]);
    }

    /// Not a ScanRuler test: a wall fitted again 0.07 mm off in a plan sketch 5 mm up, a round
    /// drawn twice a hair apart, and what the fix says.
    #[test]
    fn finds_a_wall_and_a_round_drawn_twice_on_parallel_planes_and_says_the_fix() {
        let lower = sketch_of(vec![line(1, [0.0, 0.0], [20.0, 0.0]), {
            let mut c = Entity::new(Shape::Circle { c: DVec2::new(5.0, 5.0), r: 2.0 });
            c.id = 2;
            c
        }]);
        let upper = sketch_of(vec![line(7, [2.0, 0.07], [18.0, 0.07]), {
            let mut c = Entity::new(Shape::Circle { c: DVec2::new(5.05, 5.0), r: 2.02 });
            c.id = 8;
            c
        }]);
        let frame = |z: f64| SectionFrame { origin: DVec3::new(0.0, 0.0, z), normal: DVec3::Z, u: DVec3::X, v: DVec3::Y };
        let a = SheetSketch { id: 1, name: "Sketch 1", frame: frame(0.0), sketch: &lower };
        let b = SheetSketch { id: 2, name: "Sketch 2", frame: frame(5.0), sketch: &upper };
        let found = sketch_near_misses(&b, &[a, b]);
        assert_eq!(found.found, 2);
        assert_eq!(found.misses[0], NearMiss { piece: 7, sketch: "Sketch 1".to_string(), other: 1, off_mm: 0.07 });
        assert_eq!(found.misses[1].off_mm, 0.07);
        let fix = near_miss_fix(&b, &found.misses, &[a]);
        assert!(fix.contains("remove: { pieces: [7, 8] }, add: { project: [{ sketch: \"Sketch 1\", pieces: [1, 2] }] }"));
        assert!(fix.ends_with("(reverse.sketch.edit { sketch: \"Sketch 1\", add: { fit: { level: 5 } } })"));
    }
}
