// SPDX-License-Identifier: AGPL-3.0-only
//! Corners inferred, and a profile closed. Ported from ScanRuler
//! (`plugins/reverse/core/corners.ts` and `placeSegments` in `segment.ts`), © Stefan Hermann
//! (CNC Kitchen), AGPL-3.0-only.
//!
//! Fitted pieces never meet by themselves: every fit stops short of the blend, a sharp corner
//! needs its two lines carried on to their intersection, a fillet needs its arc's ends put
//! exactly on the sides it joins. So the pieces of a chain are walked in order, and wherever
//! two neighbouring ends lie within the gap a corner is allowed, they are brought to one
//! point and held there with a coincidence — plus a tangency where a line meets an arc it
//! runs into, or two arcs blend. What is left open is the operator's to close: Close bridges
//! the remaining gaps.

use std::f64::consts::TAU;

use glam::DVec2;

use crate::fit2d::unwind_angle;
use crate::segment::{Piece, SegmentedChain};
use crate::sketch::{anchor_point, angle_of, arc_point, constraint_entities, nearest_on, Anchor, AnchorAt, Entity, EntityId, Rule, Shape, Sketch, TOUCH_MM};

/// The gap two ends may stand apart and still be one corner, millimetres.
pub const CORNER_GAP_MM: f64 = 2.0;
/// How near a line has to run to tangent to an arc's circle, as a share of the corner gap.
const TANGENT_SHARE: f64 = 0.5;
/// And how near by angle; the miss the angle stands for is r·(1 − cos), never under twice the
/// pieces' scatter.
const TANGENT_DEG: f64 = 20.0;
/// Two lines meeting at under this angle are as good as parallel.
const PARALLEL_DEG: f64 = 5.0;
/// How far past its end a line may be carried to meet the other, in gaps.
const REACH_GAPS: f64 = 4.0;

fn tangent_slack(r: f64, scatter: f64, tol: f64) -> f64 {
    tol.min((r * (1.0 - TANGENT_DEG.to_radians().cos())).max(2.0 * scatter))
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CornerReport {
    /// How many corners were joined, and how many of those with a tangency.
    pub joined: usize,
    pub tangent: usize,
    /// How many gaps were left open — wider than the corner gap.
    pub open: usize,
}

/// A segment as it entered the sketch, and which of the entity's anchors stands at the run's
/// first and last point — what corner inference walks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlacedSegment {
    pub id: EntityId,
    pub from: AnchorAt,
    pub to: AnchorAt,
    /// This piece and the next make a corner whatever the gap between their ends.
    pub join_next: bool,
}

/// Add a segmented chain's pieces to the sketch as fitted entities, each with its slice points
/// behind it.
pub fn place_segments(sketch: &mut Sketch, cut: &SegmentedChain) -> Vec<PlacedSegment> {
    cut.segments
        .iter()
        .map(|seg| {
            let first = seg.points[0];
            let last = seg.points[seg.points.len() - 1];
            match seg.piece {
                Piece::Line(fit) => {
                    // The piece runs between the projections of the run's ends.
                    let project = |p: DVec2| fit.center + fit.dir * (p - fit.center).dot(fit.dir);
                    let id = sketch.add(Entity::new(Shape::Line { a: project(first), b: project(last) }).fitted(seg.points.clone(), cut.scatter));
                    PlacedSegment { id, from: AnchorAt::A, to: AnchorAt::B, join_next: seg.corner_after }
                }
                Piece::Arc(fit) => {
                    let ta = angle_of(fit.center, first);
                    let tb = angle_of(fit.center, last);
                    let tm = angle_of(fit.center, seg.points[seg.points.len() >> 1]);
                    // Which way round the run goes: counter-clockwise from its first point if the
                    // middle point lies on that way to the last.
                    let ccw_sweep = unwind_angle(tb, ta) - ta;
                    let ccw = unwind_angle(tm, ta) - ta <= ccw_sweep;
                    let (start, sweep) = if ccw { (ta, if ccw_sweep > 1e-9 { ccw_sweep } else { TAU }) } else { (tb, TAU - ccw_sweep) };
                    let id = sketch.add(Entity::new(Shape::Arc { c: fit.center, r: fit.radius, start, sweep }).fitted(seg.points.clone(), cut.scatter));
                    if ccw {
                        PlacedSegment { id, from: AnchorAt::A, to: AnchorAt::B, join_next: seg.corner_after }
                    } else {
                        PlacedSegment { id, from: AnchorAt::B, to: AnchorAt::A, join_next: seg.corner_after }
                    }
                }
            }
        })
        .collect()
}

/// `raw` (in [0, 2π)) or a full turn either side, whichever is nearest the sweep the arc had.
fn nearest_sweep(raw: f64, had: f64) -> f64 {
    [raw, raw + TAU, raw - TAU]
        .into_iter()
        .filter(|s| *s > 1e-6 && *s <= TAU + 1e-9)
        .min_by(|a, b| (a - had).abs().total_cmp(&(b - had).abs()))
        .unwrap_or(raw)
}

/// Move an entity's anchor to a point — a line's end outright, an arc's end to the angle the
/// point lies at, keeping the other end.
pub fn move_anchor(e: &mut Entity, at: AnchorAt, p: DVec2) {
    match &mut e.shape {
        Shape::Spline(sp) => {
            let i = match at {
                AnchorAt::P(i) => crate::sketch_spline::spline_fit_point_index(sp, i as usize),
                _ => crate::sketch_spline::spline_point_index(sp, at),
            };
            if let Some(q) = i.and_then(|i| sp.pts.get_mut(i)) {
                *q = p;
            }
        }
        Shape::Line { a, b } => {
            if at == AnchorAt::A {
                *a = p;
            } else {
                *b = p;
            }
        }
        Shape::Arc { c, start, sweep, .. } => {
            let angle = angle_of(*c, p);
            if at == AnchorAt::A {
                let end = *start + *sweep;
                let raw = unwind_angle(end, angle) - angle;
                *start = angle;
                *sweep = nearest_sweep(raw, *sweep);
            } else {
                let raw = unwind_angle(angle, *start) - *start;
                *sweep = nearest_sweep(raw, *sweep);
            }
        }
        Shape::Point { p: q } => *q = p,
        Shape::Circle { .. } => {}
    }
}

fn cross(a: DVec2, b: DVec2) -> f64 {
    a.x * b.y - a.y * b.x
}

/// Where two lines' carriers cross, or `None` when (nearly) parallel.
pub fn line_intersection(a: (DVec2, DVec2), b: (DVec2, DVec2)) -> Option<DVec2> {
    let da = a.1 - a.0;
    let db = b.1 - b.0;
    let denom = cross(da, db);
    let (la, lb) = (da.length(), db.length());
    if la < 1e-12 || lb < 1e-12 || denom.abs() / (la * lb) < PARALLEL_DEG.to_radians().sin() {
        return None;
    }
    let t = cross(b.0 - a.0, db) / denom;
    Some(a.0 + da * t)
}

fn line_circle(a: DVec2, b: DVec2, c: DVec2, r: f64) -> Vec<DVec2> {
    let Some(d) = (b - a).try_normalize() else { return Vec::new() };
    let off = (c - a).dot(d.perp());
    if off.abs() > r {
        return Vec::new();
    }
    let foot = a + d * (c - a).dot(d);
    let h = (r * r - off * off).max(0.0).sqrt();
    if h < 1e-9 {
        vec![foot]
    } else {
        vec![foot + d * h, foot - d * h]
    }
}

fn circle_circle(c1: DVec2, r1: f64, c2: DVec2, r2: f64) -> Vec<DVec2> {
    let d = (c2 - c1).length();
    if d < 1e-9 || d > r1 + r2 || d < (r1 - r2).abs() {
        return Vec::new();
    }
    let a = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d);
    let h = (r1 * r1 - a * a).max(0.0).sqrt();
    let u = (c2 - c1) / d;
    let foot = c1 + u * a;
    let n = u.perp();
    if h < 1e-9 {
        vec![foot]
    } else {
        vec![foot + n * h, foot - n * h]
    }
}

fn nearest_to(candidates: &[DVec2], p: DVec2) -> Option<DVec2> {
    candidates.iter().copied().min_by(|a, b| (*a - p).length().total_cmp(&(*b - p).length()))
}

/// Join two ends: bring them to one point — the intersection of their carriers where there is
/// one within reach, the tangent point where a line or an arc runs tangent into an arc, the
/// midpoint otherwise — and hold them with a coincidence, plus a tangency where one was
/// found. Returns whether a tangency was added.
pub fn join_ends(sketch: &mut Sketch, p: Anchor, q: Anchor, gap: f64) -> bool {
    let (Some(ea), Some(eb)) = (sketch.entity(p.entity).cloned(), sketch.entity(q.entity).cloned()) else { return false };
    let pa = anchor_point(&ea.shape, p.at);
    let pb = anchor_point(&eb.shape, q.at);
    let mid = (pa + pb) / 2.0;
    let reach = (REACH_GAPS * gap).max((pa - pb).length() * 2.0);
    let within = |x: Option<DVec2>| x.filter(|x| (*x - pa).length() <= reach && (*x - pb).length() <= reach);
    // Two lines running alongside each other never meet: held at one point, the solver could
    // only meet the coincidence where their carriers cross, far off, and would carry the
    // corner out there. (Not in ScanRuler, whose dense slices keep the riser between two such
    // lines as a piece of its own; a coarse mesh can lose it.) They are joined by a straight
    // bridge instead — the riser of the step they make.
    if let (Shape::Line { a: a0, b: a1 }, Shape::Line { a: b0, b: b1 }) = (ea.shape.clone(), eb.shape.clone()) {
        if line_intersection((a0, a1), (b0, b1)).is_none() && (pa - pb).length() > TOUCH_MM {
            let id = sketch.add_shape(Shape::Line { a: pa, b: pb });
            sketch.constrain(Rule::Coincident { p, q: Anchor::new(id, AnchorAt::A) });
            sketch.constrain(Rule::Coincident { p: Anchor::new(id, AnchorAt::B), q });
            return false;
        }
    }
    let mut meet = mid;
    let mut tangent = false;
    let tan_tol = gap * TANGENT_SHARE;
    let scatter = ea.scatter.max(eb.scatter);
    match (ea.shape.clone(), eb.shape.clone()) {
        (Shape::Line { a: a0, b: a1 }, Shape::Line { a: b0, b: b1 }) => {
            if let Some(x) = within(line_intersection((a0, a1), (b0, b1))) {
                meet = x;
            }
        }
        (Shape::Line { a, b }, Shape::Arc { c, r, .. }) | (Shape::Arc { c, r, .. }, Shape::Line { a, b }) => {
            let tangent_point = (b - a).try_normalize().and_then(|d| {
                let off = (c - a).dot(d.perp());
                ((off.abs() - r).abs() <= tangent_slack(r, scatter, tan_tol)).then(|| a + d * (c - a).dot(d))
            });
            if let Some(t) = within(tangent_point) {
                meet = t;
                tangent = true;
            } else if let Some(x) = within(nearest_to(&line_circle(a, b, c, r), mid)) {
                meet = x;
            }
        }
        (Shape::Arc { c: ca, r: ra, .. }, Shape::Arc { c: cb, r: rb, .. }) => {
            let d = (cb - ca).length();
            let t = (d >= 1e-9).then(|| {
                let u = (cb - ca) / d;
                let slack = tangent_slack(ra.min(rb), scatter, tan_tol);
                if (d - (ra + rb)).abs() <= slack {
                    Some(ca + u * ra)
                } else if (d - (ra - rb).abs()).abs() <= slack {
                    Some(if ra >= rb { ca + u * ra } else { ca - u * ra })
                } else {
                    None
                }
            });
            if let Some(t) = within(t.flatten()) {
                meet = t;
                tangent = true;
            } else if let Some(x) = within(nearest_to(&circle_circle(ca, ra, cb, rb), mid)) {
                meet = x;
            }
        }
        _ => {}
    }
    if let Some(e) = sketch.entity_mut(p.entity) {
        move_anchor(e, p.at, meet);
    }
    if let Some(e) = sketch.entity_mut(q.entity) {
        move_anchor(e, q.at, meet);
    }
    sketch.constrain(Rule::Coincident { p, q });
    if tangent {
        sketch.constrain(Rule::Tangent { a: p.entity, b: q.entity });
    }
    tangent
}

/// The pairs of ends that follow each other along the chain.
fn consecutive(order: &[PlacedSegment], closed: bool) -> Vec<(Anchor, Anchor, bool)> {
    let count = if closed { order.len() } else { order.len().saturating_sub(1) };
    if order.len() == 1 && !closed {
        return Vec::new();
    }
    (0..count)
        .map(|i| {
            let (cur, next) = (order[i], order[(i + 1) % order.len()]);
            (Anchor::new(cur.id, cur.to), Anchor::new(next.id, next.from), cur.join_next)
        })
        .collect()
}

fn gap_of(sketch: &Sketch, p: Anchor, q: Anchor) -> f64 {
    match (sketch.anchor(p), sketch.anchor(q)) {
        (Some(a), Some(b)) => (a - b).length(),
        _ => f64::INFINITY,
    }
}

fn already_joined(sketch: &Sketch, p: Anchor, q: Anchor) -> bool {
    sketch.constraints.iter().any(|c| matches!(c.rule, Rule::Coincident { p: a, q: b } if (a == p && b == q) || (a == q && b == p)))
}

/// Walk the chain's pieces and join every pair of neighbouring ends within `gap` millimetres
/// — and every pair the segmentation dropped a piece between. A single arc on a closed chain
/// is closed on itself into a circle.
pub fn infer_corners(sketch: &mut Sketch, order: &[PlacedSegment], closed: bool, gap: f64) -> CornerReport {
    let mut report = CornerReport::default();
    if order.len() == 1 && closed {
        if let Some(e) = sketch.entity_mut(order[0].id) {
            if let Shape::Arc { c, r, .. } = e.shape {
                e.shape = Shape::Circle { c, r };
                report.joined = 1;
            }
        }
        return report;
    }
    for (p, q, corner) in consecutive(order, closed) {
        if already_joined(sketch, p, q) {
            continue;
        }
        let apart = gap_of(sketch, p, q);
        if apart <= gap || (corner && apart.is_finite()) {
            if join_ends(sketch, p, q, gap.max(apart)) {
                report.tangent += 1;
            }
            report.joined += 1;
        } else {
            report.open += 1;
        }
    }
    report
}

/// Whether `x` lies past the given end of the line, looking out from the other end.
fn ahead(a: DVec2, b: DVec2, at: AnchorAt, x: DVec2) -> bool {
    let (from, to) = if at == AnchorAt::A { (b, a) } else { (a, b) };
    let d = to - from;
    let ll = d.dot(d);
    ll < 1e-12 || (x - from).dot(d) / ll >= 0.5
}

/// Close what corner inference left open: every remaining gap along the chain — and the gap
/// from an open chain's last end back to its first — is bridged. Two lines whose carriers
/// cross ahead of both ends are carried on to the crossing; anything else gets a straight
/// bridge. Returns how many gaps were closed.
pub fn close_chain(sketch: &mut Sketch, order: &[PlacedSegment], gap: f64) -> usize {
    if order.is_empty() {
        return 0;
    }
    let mut closed = 0;
    for (p, q, _) in consecutive(order, true) {
        if already_joined(sketch, p, q) {
            continue;
        }
        let (Some(ea), Some(eb)) = (sketch.entity(p.entity).cloned(), sketch.entity(q.entity).cloned()) else { continue };
        let pa = anchor_point(&ea.shape, p.at);
        let pb = anchor_point(&eb.shape, q.at);
        if (pa - pb).length() <= gap {
            join_ends(sketch, p, q, gap);
            closed += 1;
            continue;
        }
        if let (Shape::Line { a: a0, b: a1 }, Shape::Line { a: b0, b: b1 }) = (ea.shape, eb.shape) {
            if let Some(x) = line_intersection((a0, a1), (b0, b1)) {
                if ahead(a0, a1, p.at, x) && ahead(b0, b1, q.at, x) {
                    if let Some(e) = sketch.entity_mut(p.entity) {
                        move_anchor(e, p.at, x);
                    }
                    if let Some(e) = sketch.entity_mut(q.entity) {
                        move_anchor(e, q.at, x);
                    }
                    sketch.constrain(Rule::Coincident { p, q });
                    closed += 1;
                    continue;
                }
            }
        }
        let id = sketch.add_shape(Shape::Line { a: pa, b: pb });
        sketch.constrain(Rule::Coincident { p, q: Anchor::new(id, AnchorAt::A) });
        sketch.constrain(Rule::Coincident { p: Anchor::new(id, AnchorAt::B), q });
        closed += 1;
    }
    closed
}

/// Make a corner sharp: the fillet arc — or a chamfer's line — between two lines is removed
/// and the lines carried on to their intersection. False when the piece does not stand between
/// two lines that cross.
pub fn sharpen_corner(sketch: &mut Sketch, piece: EntityId) -> bool {
    let Some(e) = sketch.entity(piece) else { return false };
    if !matches!(e.shape, Shape::Arc { .. } | Shape::Line { .. }) {
        return false;
    }
    let neighbour = |at: AnchorAt| -> Option<(EntityId, AnchorAt, DVec2, DVec2)> {
        sketch.constraints.iter().find_map(|c| {
            let Rule::Coincident { p, q } = c.rule else { return None };
            let other = if p.entity == piece && p.at == at {
                q
            } else if q.entity == piece && q.at == at {
                p
            } else {
                return None;
            };
            match sketch.entity(other.entity)?.shape {
                Shape::Line { a, b } => Some((other.entity, other.at, a, b)),
                _ => None,
            }
        })
    };
    let (Some(na), Some(nb)) = (neighbour(AnchorAt::A), neighbour(AnchorAt::B)) else { return false };
    if na.0 == nb.0 {
        return false;
    }
    let Some(x) = line_intersection((na.2, na.3), (nb.2, nb.3)) else { return false };
    if let Some(e) = sketch.entity_mut(na.0) {
        move_anchor(e, na.1, x);
    }
    if let Some(e) = sketch.entity_mut(nb.0) {
        move_anchor(e, nb.1, x);
    }
    sketch.entities.retain(|e| e.id != piece);
    sketch.constraints.retain(|c| !constraint_entities(&c.rule).contains(&piece));
    sketch.constrain(Rule::Coincident { p: Anchor::new(na.0, na.1), q: Anchor::new(nb.0, nb.1) });
    true
}

/// An end of a sketch's outline that meets nothing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OpenEnd {
    pub piece: EntityId,
    pub end: AnchorAt,
    pub at: DVec2,
    /// The open end of another piece nearest it, and how far.
    pub nearest: Option<(EntityId, AnchorAt, f64)>,
}

/// The ends of a sketch's outline that meet nothing, the smallest gaps first — what a sketch
/// that closes no region is told.
pub fn open_ends(sketch: &Sketch) -> Vec<OpenEnd> {
    let outline: Vec<&Entity> = sketch.entities.iter().filter(|e| !e.construction && !matches!(e.shape, Shape::Point { .. })).collect();
    let mut ends = Vec::new();
    for e in &outline {
        if matches!(e.shape, Shape::Line { .. } | Shape::Arc { .. }) {
            for at in [AnchorAt::A, AnchorAt::B] {
                ends.push((e.id, at, anchor_point(&e.shape, at)));
            }
        }
    }
    let meets = |k: usize| {
        let (id, _, p) = ends[k];
        ends.iter().enumerate().any(|(j, q)| j != k && (q.2 - p).length() <= TOUCH_MM) || outline.iter().any(|e| e.id != id && nearest_on(&e.shape, p).1 <= TOUCH_MM)
    };
    let mut out = Vec::new();
    for k in 0..ends.len() {
        if meets(k) {
            continue;
        }
        let (piece, end, at) = ends[k];
        let mut nearest = None;
        let mut gap = f64::INFINITY;
        for (j, q) in ends.iter().enumerate() {
            if q.0 == piece {
                continue;
            }
            let d = (q.2 - at).length();
            if d > 1.0 && meets(j) {
                continue;
            }
            if d < gap {
                gap = d;
                nearest = Some((q.0, q.1, d));
            }
        }
        out.push(OpenEnd { piece, end, at, nearest });
    }
    out.sort_by(|a, b| a.nearest.map_or(f64::INFINITY, |n| n.2).total_cmp(&b.nearest.map_or(f64::INFINITY, |n| n.2)));
    out
}

/// What fitting one chain into the sketch did.
#[derive(Debug, Clone, PartialEq)]
pub struct ChainFit {
    pub order: Vec<PlacedSegment>,
    pub closed: bool,
    pub corners: CornerReport,
    pub lines: usize,
    pub arcs: usize,
}

/// Fit one slice chain into the sketch: cut it into lines and arcs, place them as fitted
/// pieces and (with `join_corners`) join their corners. The caller solves afterwards.
pub fn fit_chain(sketch: &mut Sketch, chain: &[DVec2], options: crate::segment::SegmentOptions, join_corners: bool) -> Result<ChainFit, crate::FitError> {
    let cut = crate::segment::segment_chain(chain, options)?;
    let order = place_segments(sketch, &cut);
    let corners = if join_corners { infer_corners(sketch, &order, cut.closed, CORNER_GAP_MM) } else { CornerReport::default() };
    let lines = cut.segments.iter().filter(|s| s.piece.is_line()).count();
    Ok(ChainFit { arcs: cut.segments.len() - lines, lines, order, closed: cut.closed, corners })
}

/// A point of an arc for drawing: `t` from 0 at anchor A to 1 at B.
pub fn arc_at(c: DVec2, r: f64, start: f64, sweep: f64, t: f64) -> DVec2 {
    arc_point(c, r, start + sweep * t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::segment::tests::rounded_rect_chain;
    use crate::segment::{segment_chain, SegmentOptions};
    use crate::solver::{solve_sketch, SolveOptions};

    #[test]
    fn joins_a_rounded_rectangle_at_eight_tangent_corners_into_one_solvable_loop() {
        let chain = rounded_rect_chain(60.0, 40.0, 5.0, 0.25, 0.03, 3);
        let cut = segment_chain(&chain, SegmentOptions::default()).unwrap();
        let mut s = Sketch::default();
        let placed = place_segments(&mut s, &cut);
        let report = infer_corners(&mut s, &placed, cut.closed, CORNER_GAP_MM);
        assert_eq!(report, CornerReport { joined: 8, tangent: 8, open: 0 });
        let solved = solve_sketch(&mut s, SolveOptions::default());
        assert!(solved.ok, "{solved:?}");
        assert!(solved.soft_rms < 0.08, "soft rms {}", solved.soft_rms);
        for c in &s.constraints {
            if let Rule::Coincident { p, q } = c.rule {
                assert!((s.anchor(p).unwrap() - s.anchor(q).unwrap()).length() < 1e-4);
            }
        }
    }

    #[test]
    fn reads_a_circle_as_one_arc_that_closes_into_a_circle() {
        let mut rand = crate::fit2d::Mulberry32::new(9);
        let mut chain: Vec<DVec2> = (0..360)
            .map(|i| {
                let a = (i as f64).to_radians();
                let r = 12.0 + rand.gaussian() * 0.02;
                DVec2::new(r * a.cos(), r * a.sin())
            })
            .collect();
        chain.push(chain[0]);
        let mut s = Sketch::default();
        let fit = fit_chain(&mut s, &chain, SegmentOptions::default(), true).unwrap();
        assert_eq!(fit.corners.joined, 1);
        let Shape::Circle { r, .. } = s.entities[0].shape else { panic!("not a circle: {:?}", s.entities[0].shape) };
        assert!((r - 12.0).abs() < 0.05);
    }

    #[test]
    fn makes_a_fillet_sharp_by_carrying_its_sides_to_the_corner() {
        let chain = rounded_rect_chain(60.0, 40.0, 5.0, 0.25, 0.02, 3);
        let mut s = Sketch::default();
        fit_chain(&mut s, &chain, SegmentOptions::default(), true).unwrap();
        let fillet = s.entities.iter().find(|e| matches!(e.shape, Shape::Arc { .. })).unwrap().id;
        assert!(sharpen_corner(&mut s, fillet));
        assert_eq!(s.entities.iter().filter(|e| matches!(e.shape, Shape::Arc { .. })).count(), 3);
        assert!(solve_sketch(&mut s, SolveOptions::default()).ok);
    }

    #[test]
    fn bridges_two_lines_running_alongside_each_other_instead_of_meeting_them_far_off() {
        let mut s = Sketch::default();
        // Out along y = 0, back along y = −1: the riser between them is missing.
        let pts_a: Vec<DVec2> = (0..20).map(|i| DVec2::new(-(i as f64) * 0.2, 0.001 * i as f64)).collect();
        let pts_b: Vec<DVec2> = (0..20).map(|i| DVec2::new(-4.0 + i as f64 * 0.2, -1.0)).collect();
        let a = s.add(Entity::new(Shape::Line { a: DVec2::ZERO, b: DVec2::new(-3.8, 0.02) }).fitted(pts_a, 0.01));
        let b = s.add(Entity::new(Shape::Line { a: DVec2::new(-4.0, -1.0), b: DVec2::new(-0.2, -1.0) }).fitted(pts_b, 0.01));
        let order = [PlacedSegment { id: a, from: AnchorAt::A, to: AnchorAt::B, join_next: false }, PlacedSegment { id: b, from: AnchorAt::A, to: AnchorAt::B, join_next: false }];
        let report = infer_corners(&mut s, &order, false, CORNER_GAP_MM);
        assert_eq!(report.joined, 1);
        assert_eq!(s.entities.len(), 3, "a riser bridges the two");
        let solved = solve_sketch(&mut s, SolveOptions::default());
        assert!(solved.ok);
        for e in &s.entities {
            let Shape::Line { a, b } = e.shape else { panic!() };
            assert!(a.x > -5.0 && b.x > -5.0, "nothing runs off: {a} {b}");
        }
    }

    #[test]
    fn carries_two_lines_on_to_their_corner_and_bridges_what_it_cannot_meet() {
        let mut s = Sketch::default();
        let a = s.add_shape(Shape::Line { a: DVec2::new(0.0, 0.0), b: DVec2::new(8.0, 0.0) });
        let b = s.add_shape(Shape::Line { a: DVec2::new(10.0, 3.0), b: DVec2::new(10.0, 10.0) });
        let order = [PlacedSegment { id: a, from: AnchorAt::A, to: AnchorAt::B, join_next: false }, PlacedSegment { id: b, from: AnchorAt::A, to: AnchorAt::B, join_next: false }];
        assert_eq!(close_chain(&mut s, &order, CORNER_GAP_MM), 2);
        assert_eq!(s.anchor(Anchor::new(a, AnchorAt::B)), Some(DVec2::new(10.0, 0.0)));
        assert_eq!(s.entities.len(), 3, "the far ends are bridged by a line");
    }
}
