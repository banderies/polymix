// SPDX-License-Identifier: AGPL-3.0-only
//! A corner of a fitted sketch made round where the scan is round. Ported from ScanRuler
//! (`plugins/reverse/core/roundCorner.ts`), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! An arc tangent to two pieces — lines or arcs, in any pair — the pieces cut back to where it
//! touches them, held to both by coincidences and tangencies. With a radius the round is that
//! radius, a dimension. Without one it is the scan's: the slice points about the corner that
//! the two pieces carried (a segmenter that missed the round merged them in) move onto the
//! arc, and the solver fits it to them with the tangencies held — "this corner is a tangent
//! blend", said once, and ScanRuler finds how large.
//!
//! `fillet_lines` (fillet_corner.rs) is the panel's sketch fillet for two lines, a radius
//! typed; this is the agent's, for the pieces a fit leaves, read off the scan.

use std::f64::consts::PI;

use glam::DVec2;
use serde::{Deserialize, Serialize};

use crate::fillet_corner::{flip, joined_to, unit2, unjoin};
use crate::fillet_num::to_precision;
use crate::fit2d::unwind_angle;
use crate::sketch::{anchor_point, angle_of, nearest_on, Anchor, AnchorAt, ConstraintId, Entity, EntityId, Rule, Shape, Sketch};

/// What the slice points said of a round put in without a radius.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Found {
    /// A round found among them.
    Read,
    /// The best fit at either end of the radii searched — the points hold no round, only a
    /// sharp corner or a stray.
    Limit,
    /// Too few points off the two pieces to search, the radius a guess from the pieces' size
    /// (ScanRuler's "none").
    #[serde(rename = "none")]
    Nothing,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RoundCorner {
    /// The arc put in, and the radius dimension on it when one was asked.
    pub arc: EntityId,
    pub dimension: Option<ConstraintId>,
    /// The radius it was put in at — the solve may move it.
    pub radius: f64,
    /// How many slice points moved onto the arc.
    pub points: usize,
    /// Without a radius given, what the points said of it.
    pub found: Option<Found>,
}

/// An end counts as joined to another within this, millimetres.
const JOIN_MM: f64 = 1e-6;

fn perp(v: DVec2) -> DVec2 {
    DVec2::new(-v.y, v.x)
}

/// A piece seen from the corner: the end that is cut back, and the way the piece runs away
/// from the corner there. The piece's geometry is as it stood when seen.
#[derive(Debug, Clone)]
struct Seen {
    id: EntityId,
    shape: Shape,
    points: Vec<DVec2>,
    near: AnchorAt,
    at: DVec2,
    dir: DVec2,
}

fn is_piece(e: &Entity) -> bool {
    matches!(e.shape, Shape::Line { .. } | Shape::Arc { .. })
}

fn seen(piece: &Entity, near: AnchorAt) -> Option<Seen> {
    let at = anchor_point(&piece.shape, near);
    let dir = match piece.shape {
        Shape::Line { .. } => unit2(anchor_point(&piece.shape, flip(near)) - at),
        // An arc runs counter-clockwise from a: at a it heads along the tangent ccw, at b back
        // clockwise.
        Shape::Arc { c, .. } => unit2(at - c).map(|radial| if near == AnchorAt::A { perp(radial) } else { -perp(radial) }),
        _ => None,
    }?;
    Some(Seen { id: piece.id, shape: piece.shape.clone(), points: piece.points.clone(), near, at, dir })
}

/// The ends of `a` and `b` that make the corner: the pair held coincident, else the nearest
/// pair.
fn corner_ends(sketch: &Sketch, a: &Entity, b: &Entity) -> (AnchorAt, AnchorAt) {
    for ea in [AnchorAt::A, AnchorAt::B] {
        for j in joined_to(sketch, Anchor::new(a.id, ea)) {
            if j.entity == b.id && (j.at == AnchorAt::A || j.at == AnchorAt::B) {
                return (ea, j.at);
            }
        }
    }
    let mut best = (AnchorAt::A, AnchorAt::A);
    let mut d = f64::INFINITY;
    for ea in [AnchorAt::A, AnchorAt::B] {
        for eb in [AnchorAt::A, AnchorAt::B] {
            let dd = (anchor_point(&a.shape, ea) - anchor_point(&b.shape, eb)).length();
            if dd < d {
                d = dd;
                best = (ea, eb);
            }
        }
    }
    best
}

/// The centre of the circle of radius r tangent to a piece's carrier on the corner's inside:
/// offset curves met. A line's offset is a line; an arc's a circle r nearer its centre or
/// farther, as the inside lies.
#[derive(Debug, Clone, Copy)]
enum Offset {
    Line { p: DVec2, d: DVec2 },
    Circle { c: DVec2, r: f64 },
}

fn offset_of(s: &Seen, other: &Seen, r: f64) -> Option<Offset> {
    // The inside of the corner from this piece: the side the other runs to.
    match s.shape {
        Shape::Line { .. } => {
            let n = perp(s.dir);
            let side = if n.dot(other.dir) >= 0.0 { 1.0 } else { -1.0 };
            Some(Offset::Line { p: s.at + n * (side * r), d: s.dir })
        }
        Shape::Arc { c, r: arc_r, .. } => {
            let toward_centre = unit2(c - s.at)?;
            let inward = toward_centre.dot(other.dir) >= 0.0;
            let big_r = if inward { arc_r - r } else { arc_r + r };
            if big_r > 0.0 {
                Some(Offset::Circle { c, r: big_r })
            } else {
                None
            }
        }
        _ => None,
    }
}

fn meet(a: Offset, b: Offset) -> Vec<DVec2> {
    match (a, b) {
        (Offset::Line { p: ap, d: ad }, Offset::Line { p: bp, d: bd }) => {
            let den = ad.x * bd.y - ad.y * bd.x;
            if den.abs() < 1e-12 {
                return Vec::new();
            }
            let t = ((bp.x - ap.x) * bd.y - (bp.y - ap.y) * bd.x) / den;
            vec![ap + ad * t]
        }
        (Offset::Circle { .. }, Offset::Line { .. }) => meet(b, a),
        (Offset::Line { p, d }, Offset::Circle { c, r }) => {
            let f = p - c;
            let bb = f.dot(d);
            let cc = f.dot(f) - r * r;
            let disc = bb * bb - cc;
            if disc < 0.0 {
                return Vec::new();
            }
            let q = disc.sqrt();
            vec![p + d * (-bb - q), p + d * (-bb + q)]
        }
        (Offset::Circle { c: ca, r: ra }, Offset::Circle { c: cb, r: rb }) => {
            let dv = cb - ca;
            let d = dv.length();
            if d < 1e-12 || d > ra + rb || d < (ra - rb).abs() {
                return Vec::new();
            }
            let x = (d * d + ra * ra - rb * rb) / (2.0 * d);
            let h = (ra * ra - x * x).max(0.0).sqrt();
            let u = dv / d;
            let m = ca + u * x;
            let n = perp(u);
            vec![m + n * h, m - n * h]
        }
    }
}

/// Where a circle centred `c` touches a piece's carrier.
fn touch_on(shape: &Shape, c: DVec2) -> DVec2 {
    match *shape {
        Shape::Line { a, b } => {
            let d = unit2(b - a).unwrap_or(DVec2::ZERO);
            a + d * (c - a).dot(d)
        }
        Shape::Arc { c: pc, r, .. } | Shape::Circle { c: pc, r } => {
            let u = unit2(c - pc).unwrap_or(DVec2::X);
            pc + u * r
        }
        Shape::Point { p } => p,
        Shape::Spline(ref sp) => crate::sketch_spline::nearest_on_spline_piece(sp, c).0,
    }
}

/// A round between two pieces: its centre and its two tangent points.
#[derive(Debug, Clone, Copy)]
struct Tangent {
    c: DVec2,
    ta: DVec2,
    tb: DVec2,
}

/// The round of radius r between the two pieces as they stand: its centre and the two
/// tangent points — the candidate nearest the corner, on the pieces' kept side of each
/// tangent point — or `None`.
fn tangent_circle(a: &Seen, b: &Seen, corner: DVec2, r: f64) -> Option<Tangent> {
    let oa = offset_of(a, b, r)?;
    let ob = offset_of(b, a, r)?;
    let mut best: Option<Tangent> = None;
    let mut d = f64::INFINITY;
    for c in meet(oa, ob) {
        let ta = touch_on(&a.shape, c);
        let tb = touch_on(&b.shape, c);
        let dd = (c - corner).length();
        if dd < d {
            d = dd;
            best = Some(Tangent { c, ta, tb });
        }
    }
    best
}

/// The corner itself: where the two pieces' ends meet, or the middle of the gap between them.
fn corner_of(a: &Seen, b: &Seen) -> DVec2 {
    (a.at + b.at) * 0.5
}

/// A piece cut back so that its near end lands at `p`.
fn trim_to(sketch: &mut Sketch, s: &Seen, p: DVec2) {
    let Some(e) = sketch.entity_mut(s.id) else { return };
    match &mut e.shape {
        Shape::Line { a, b } => {
            if s.near == AnchorAt::A {
                *a = p;
            } else {
                *b = p;
            }
        }
        Shape::Arc { c, start, sweep, .. } => {
            let end = *start + *sweep;
            let angle = angle_of(*c, p);
            if s.near == AnchorAt::A {
                let new_start = unwind_angle(angle, end - 2.0 * PI);
                *sweep = end - new_start;
                *start = new_start;
            } else {
                *sweep = unwind_angle(angle, *start) - *start;
            }
        }
        _ => {}
    }
}

/// The arc from tangent point ta to tb the short way round about c, as a sketch arc
/// (counter-clockwise from its a): start, sweep, and whether it starts at ta.
fn short_arc(c: DVec2, ta: DVec2, tb: DVec2) -> (f64, f64, bool) {
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
    (if from_a { ang_a } else { ang_b }, sweep.abs(), from_a)
}

/// How many radii the search tries, from the scatter (0.02 mm at least) to the room the
/// pieces leave, evenly on a log scale.
const RADIUS_STEPS: usize = 120;

/// The radius the slice points about the corner suggest: the points of either piece within
/// `window` of the corner that stand off their own piece by over three scatters — the round's
/// — and the tangent circle whose arc they lie nearest, searched over the radii that fit — and
/// whether the best lay at either end of the search, which is no reading.
fn radius_from_points(a: &Seen, b: &Seen, corner: DVec2, window: f64, scatter: f64) -> Option<(f64, bool)> {
    let mut pts: Vec<DVec2> = Vec::new();
    let mut off = 0;
    for s in [a, b] {
        for p in &s.points {
            if (*p - corner).length() > window {
                continue;
            }
            pts.push(*p);
            if nearest_on(&s.shape, *p).1 > 3.0 * scatter {
                off += 1;
            }
        }
    }
    if off < 3 {
        return None;
    }
    let cost = |r: f64| -> f64 {
        let Some(t) = tangent_circle(a, b, corner, r) else { return f64::INFINITY };
        let mut sum = 0.0;
        for p in &pts {
            let on_arc = ((*p - t.c).length() - r).abs();
            let on_a = dist_to_ray(*p, t.ta, a);
            let on_b = dist_to_ray(*p, t.tb, b);
            let d = on_arc.min(on_a).min(on_b).min(5.0 * scatter);
            sum += d * d;
        }
        sum
    };
    let mut best: Option<f64> = None;
    let mut at = usize::MAX;
    let mut best_cost = f64::INFINITY;
    let lo = scatter.max(0.02);
    let hi = window;
    for i in 0..=RADIUS_STEPS {
        let r = lo * (hi / lo).powf(i as f64 / RADIUS_STEPS as f64);
        let c = cost(r);
        if c < best_cost {
            best_cost = c;
            best = Some(r);
            at = i;
        }
    }
    best.map(|r| (r, at == 0 || at == RADIUS_STEPS))
}

/// Distance from p to a piece's kept part from tangent point t on.
fn dist_to_ray(p: DVec2, t: DVec2, s: &Seen) -> f64 {
    match s.shape {
        Shape::Line { .. } => {
            let along = (p - t).dot(s.dir).max(0.0);
            (p - (t + s.dir * along)).length()
        }
        Shape::Arc { c, r, .. } => ((p - c).length() - r).abs(),
        _ => f64::INFINITY,
    }
}

/// How far a piece reaches: a line's length, an arc's length.
fn reach_of(s: &Seen) -> f64 {
    match s.shape {
        Shape::Line { a, b } => (a - b).length(),
        Shape::Arc { r, sweep, .. } => r * sweep,
        _ => 0.0,
    }
}

/// Round the corner between pieces `id_a` and `id_b` of a sketch. Returns what was made, or in
/// words why nothing was. The sketch is changed in place; solve it after.
pub fn round_corner(sketch: &mut Sketch, id_a: EntityId, id_b: EntityId, radius: Option<f64>) -> Result<RoundCorner, String> {
    let (pa, pb) = match (sketch.entity(id_a), sketch.entity(id_b)) {
        (Some(pa), Some(pb)) if is_piece(pa) && is_piece(pb) && pa.id != pb.id => (pa.clone(), pb.clone()),
        _ => return Err(format!("A round goes between two lines or arcs — {id_a} and {id_b} are not.")),
    };
    if pa.projected || pb.projected {
        return Err("A projected piece stays as the part made it — it cannot be cut back for a round.".to_string());
    }
    if radius.is_some_and(|r| !(r > 0.0)) {
        return Err("A round needs a radius above zero.".to_string());
    }
    // Nothing of the scan to read a radius from: pieces drawn, or moved in by an inset, which
    // lets their points go (the scan's are the other skin's).
    if radius.is_none() && pa.points.is_empty() && pb.points.is_empty() {
        return Err(format!(
            "Pieces {id_a} and {id_b} carry no slice points — drawn, or moved by an inset, which lets them go — so the scan’s round is not there to read: give radius."
        ));
    }
    let (ea, eb) = corner_ends(sketch, &pa, &pb);
    let (Some(a), Some(b)) = (seen(&pa, ea), seen(&pb, eb)) else {
        return Err("A piece is too short to round into.".to_string());
    };
    let corner = corner_of(&a, &b);
    let room = 0.95 * reach_of(&a).min(reach_of(&b));
    let scatter = pa.scatter.max(pb.scatter).max(0.01);

    let mut found: Option<Found> = None;
    let r = match radius {
        Some(r) => r,
        None => {
            let read = radius_from_points(&a, &b, corner, room, scatter);
            found = Some(match read {
                Some((_, true)) => Found::Limit,
                Some((_, false)) => Found::Read,
                None => Found::Nothing,
            });
            match read {
                Some((r, _)) => r,
                None => {
                    // No round shows in the points: a quarter of the shorter piece, or the gap
                    // the ends stand apart by.
                    let gap = (a.at - b.at).length();
                    if gap > JOIN_MM {
                        gap / 2.0
                    } else {
                        0.25 * room
                    }
                }
            }
        }
    };
    let Some(t) = tangent_circle(&a, &b, corner, r) else {
        return Err(format!("No round of {} mm is tangent to both pieces here.", to_precision(r, 3)));
    };
    // The tangent points must lie on the pieces, not past their far ends.
    if (t.ta - corner).length() > room + (a.at - corner).length() || (t.tb - corner).length() > room + (b.at - corner).length() {
        return Err(format!("A round of {} mm does not fit on these pieces.", to_precision(r, 3)));
    }

    // The join that made the corner sharp goes, and a piece bridging the two ends (a chord a
    // fit left) — its points go to the round.
    let end_a = Anchor::new(pa.id, ea);
    let end_b = Anchor::new(pb.id, eb);
    let mut moved: Vec<DVec2> = Vec::new();
    for j in joined_to(sketch, end_a) {
        if j.entity == pb.id || (j.at != AnchorAt::A && j.at != AnchorAt::B) {
            continue;
        }
        let other = Anchor::new(j.entity, flip(j.at));
        if joined_to(sketch, other).contains(&end_b) {
            if let Some(bridge) = sketch.entity(j.entity) {
                moved.extend(bridge.points.iter().copied());
            }
            sketch.remove_entities(&[j.entity]);
            break;
        }
    }
    unjoin(sketch, end_a, end_b);
    trim_to(sketch, &a, t.ta);
    trim_to(sketch, &b, t.tb);
    let (start, sweep, from_a) = short_arc(t.c, t.ta, t.tb);
    let arc_id = sketch.add_shape(Shape::Arc { c: t.c, r, start, sweep });
    sketch.constrain(Rule::Coincident { p: end_a, q: Anchor::new(arc_id, if from_a { AnchorAt::A } else { AnchorAt::B }) });
    sketch.constrain(Rule::Coincident { p: end_b, q: Anchor::new(arc_id, if from_a { AnchorAt::B } else { AnchorAt::A }) });
    sketch.constrain(Rule::Tangent { a: pa.id, b: arc_id });
    sketch.constrain(Rule::Tangent { a: pb.id, b: arc_id });
    if let Some(radius) = radius {
        let dimension = sketch.constrain(Rule::Radius { entity: arc_id, value: radius });
        return Ok(RoundCorner { arc: arc_id, dimension: Some(dimension), radius, points: 0, found: None });
    }
    // The scan's round: the points nearer the arc than their own piece go to it, the bridge's
    // with them.
    if let Some(arc) = sketch.entity_mut(arc_id) {
        arc.points = moved;
        arc.scatter = scatter;
    }
    reassign_corner_points(sketch, pa.id, pb.id, arc_id);
    let points = sketch.entity(arc_id).map_or(0, |e| e.points.len());
    if points < 3 {
        // Too few points to fit: held at the radius found, said so.
        let dimension = sketch.constrain(Rule::Radius { entity: arc_id, value: r });
        return Ok(RoundCorner { arc: arc_id, dimension: Some(dimension), radius: r, points, found });
    }
    Ok(RoundCorner { arc: arc_id, dimension: None, radius: r, points, found })
}

/// Why a round's radius is no reading of the scan, in words — or `None` for one that is:
/// under three points on the arc, or a search that found its best at either end of the radii
/// it tried, or found nothing to search.
pub fn round_unread(points: usize, found: Option<Found>) -> Option<String> {
    let found = found?;
    let fix = "give radius, or re-fit the corner so its pieces carry its slice points";
    if points < 3 {
        return Some(format!("{points} point{} on the round — radius not read; {fix}", if points == 1 { "" } else { "s" }));
    }
    match found {
        Found::Limit => Some(format!("the points fit best at the end of the radii searched — radius not read; {fix}")),
        Found::Nothing => Some(format!("no round shows in the points about the corner — radius not read; {fix}")),
        Found::Read => None,
    }
}

/// The slice points of the two pieces and the round between them shared out again by which
/// each now lies nearest — after a solve has moved the tangent points. Returns how many
/// changed hands.
pub fn reassign_corner_points(sketch: &mut Sketch, id_a: EntityId, id_b: EntityId, arc_id: EntityId) -> usize {
    let pieces: Vec<(EntityId, Shape, Vec<DVec2>)> = [id_a, id_b, arc_id].iter().filter_map(|id| sketch.entity(*id)).filter(|e| is_piece(e)).map(|e| (e.id, e.shape.clone(), e.points.clone())).collect();
    if pieces.len() != 3 {
        return 0;
    }
    let mut out: Vec<(EntityId, Vec<DVec2>)> = Vec::new();
    for (id, _, _) in &pieces {
        if !out.iter().any(|(o, _)| o == id) {
            out.push((*id, Vec::new()));
        }
    }
    let mut changed = 0;
    for (from, _, points) in &pieces {
        for p in points {
            let mut best = *from;
            let mut d = f64::INFINITY;
            for (id, shape, _) in &pieces {
                let dd = nearest_on(shape, *p).1;
                if dd < d - 1e-9 {
                    d = dd;
                    best = *id;
                }
            }
            if best != *from {
                changed += 1;
            }
            if let Some((_, list)) = out.iter_mut().find(|(o, _)| *o == best) {
                list.push(*p);
            }
        }
    }
    for (id, pts) in out {
        if let Some(e) = sketch.entity_mut(id) {
            e.points = pts;
        }
    }
    changed
}

/// How well the slice points pin a round's radius with the round held tangent to its two
/// pieces as they stand: the points' scatter over the root of their summed squared
/// sensitivity to the radius, the sensitivity taken by building the tangent round a hair
/// larger and smaller. `None` where the round has too few points.
pub fn round_radius_sigma(sketch: &Sketch, id_a: EntityId, id_b: EntityId, arc_id: EntityId) -> Option<f64> {
    let pa = sketch.entity(id_a).filter(|e| is_piece(e))?;
    let pb = sketch.entity(id_b).filter(|e| is_piece(e))?;
    let arc = sketch.entity(arc_id)?;
    let Shape::Arc { c: arc_c, r: arc_r, .. } = arc.shape else { return None };
    let pts = &arc.points;
    if pts.len() < 3 {
        return None;
    }
    // The pieces seen from the round's ends.
    let ends = |piece: &Entity| -> AnchorAt {
        let a = (anchor_point(&piece.shape, AnchorAt::A) - arc_c).length() - arc_r;
        let b = (anchor_point(&piece.shape, AnchorAt::B) - arc_c).length() - arc_r;
        if a.abs() <= b.abs() {
            AnchorAt::A
        } else {
            AnchorAt::B
        }
    };
    let sa = seen(pa, ends(pa))?;
    let sb = seen(pb, ends(pb))?;
    let corner = corner_of(&sa, &sb);
    let h = (arc_r * 1e-3).max(1e-4);
    let up = tangent_circle(&sa, &sb, corner, arc_r + h)?;
    let down = tangent_circle(&sa, &sb, corner, arc_r - h)?;
    let mut jj = 0.0;
    let mut ss = 0.0;
    for p in pts {
        let res = |c: DVec2, r: f64| (*p - c).length() - r;
        let j = (res(up.c, arc_r + h) - res(down.c, arc_r - h)) / (2.0 * h);
        jj += j * j;
        let e = res(arc_c, arc_r);
        ss += e * e;
    }
    let n = pts.len() as f64;
    let scatter = (ss / (n - 1.0).max(1.0)).sqrt().max(arc.scatter);
    if jj > 1e-12 {
        Some(scatter / jj.sqrt())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    //! Ported from ScanRuler `plugins/reverse/tests/reRoundCorner.test.ts`, and the pure part
    //! of `reAnswersK3.test.ts`'s K5 (roundUnread).
    use super::*;
    use crate::fit2d::Mulberry32;
    use crate::solver::{solve_sketch, SolveOptions, SolveReport};

    const NOISE: f64 = 0.03;

    /// The tests' Gaussian draw, as ScanRuler's tests write it.
    fn gauss(rand: &mut Mulberry32) -> f64 {
        let u = rand.next_f64().max(1e-12);
        (-2.0 * u.ln()).sqrt() * (2.0 * PI * rand.next_f64()).cos()
    }

    /// Points along a polyline of exact samples, each moved off by noise along the local
    /// normal.
    fn noisy(rand: &mut Mulberry32, points: &[DVec2]) -> Vec<DVec2> {
        (0..points.len())
            .map(|i| {
                let q = points[(i + 1).min(points.len() - 1)];
                let o = points[i.saturating_sub(1)];
                let t = q - o;
                let l = t.length();
                let l = if l == 0.0 { 1.0 } else { l };
                let d = gauss(rand) * NOISE;
                DVec2::new(points[i].x - (t.y / l) * d, points[i].y + (t.x / l) * d)
            })
            .collect()
    }

    fn line(a: DVec2, b: DVec2) -> Vec<DVec2> {
        let step = 0.2;
        let n = crate::fillet_num::js_round((b - a).length() / step).max(1.0) as usize;
        (0..n).map(|i| a + (b - a) * (i as f64 / n as f64)).collect()
    }

    fn arc(c: DVec2, r: f64, from: f64, to: f64) -> Vec<DVec2> {
        let step = 0.2;
        let n = crate::fillet_num::js_round((to - from).abs() * r / step).max(2.0) as usize;
        (0..n)
            .map(|i| {
                let a = from + (to - from) * i as f64 / n as f64;
                c + DVec2::new(a.cos(), a.sin()) * r
            })
            .collect()
    }

    /// A sketch of two fitted pieces joined at their sharp corner, each carrying its slice
    /// points (the blend's among them).
    fn two_pieces(a: Shape, a_points: Vec<DVec2>, b: Shape, b_points: Vec<DVec2>, join: (AnchorAt, AnchorAt)) -> (Sketch, EntityId, EntityId) {
        let mut sketch = Sketch::default();
        let ia = sketch.add(Entity::new(a).fitted(a_points, NOISE));
        let ib = sketch.add(Entity::new(b).fitted(b_points, NOISE));
        sketch.constrain(Rule::Coincident { p: Anchor::new(ia, join.0), q: Anchor::new(ib, join.1) });
        (sketch, ia, ib)
    }

    struct Rounded {
        made: RoundCorner,
        report: SolveReport,
        r: f64,
        arc_points: usize,
        sigma: Option<f64>,
    }

    /// The round put in, solved, the points shared out again and solved once more — what the
    /// sketch edit command does.
    fn rounded(s: &mut (Sketch, EntityId, EntityId), radius: Option<f64>) -> Rounded {
        let made = round_corner(&mut s.0, s.1, s.2, radius).expect("a round");
        let mut report = solve_sketch(&mut s.0, SolveOptions::default());
        for _ in 0..3 {
            if reassign_corner_points(&mut s.0, s.1, s.2, made.arc) == 0 {
                break;
            }
            report = solve_sketch(&mut s.0, SolveOptions::default());
        }
        let arc = s.0.entity(made.arc).expect("the arc");
        let r = arc.shape.radius().unwrap_or(f64::NAN);
        let arc_points = arc.points.len();
        let sigma = round_radius_sigma(&s.0, s.1, s.2, made.arc);
        Rounded { made, report, r, arc_points, sigma }
    }

    #[test]
    fn a_corner_made_round_in_a_fitted_sketch() {
        // The three cases of the file share one generator, as the TS file's module-level
        // mulberry32(3) is shared by its tests run in order.
        let mut rand = Mulberry32::new(3);

        // Between two lines: an R2.9 blend the fit merged comes back within its sigma, the
        // lines still on their points. The corner at (20, 10): a wall along +x up to it, then
        // up along +y, rounded R2.9 — points of the round given half to each line.
        {
            let big_r = 2.9;
            let c = DVec2::new(20.0 - big_r, 10.0 - big_r);
            let along = noisy(&mut rand, &line(DVec2::new(0.0, 10.0), DVec2::new(20.0 - big_r, 10.0)));
            let round = noisy(&mut rand, &arc(c, big_r, PI / 2.0, 0.0));
            let up = noisy(&mut rand, &line(DVec2::new(20.0, 10.0 - big_r), DVec2::new(20.0, -10.0)));
            let half = round.len() / 2;
            let mut s = two_pieces(
                Shape::Line { a: DVec2::new(0.0, 10.0), b: DVec2::new(20.0, 10.0) },
                [along.clone(), round[..half].to_vec()].concat(),
                Shape::Line { a: DVec2::new(20.0, 10.0), b: DVec2::new(20.0, -10.0) },
                [round[half..].to_vec(), up].concat(),
                (AnchorAt::B, AnchorAt::A),
            );
            let out = rounded(&mut s, None);
            assert!(out.report.ok);
            assert!(out.made.dimension.is_none());
            let sigma = out.sigma.expect("a sigma");
            assert!(sigma < 0.1, "{sigma}");
            assert!((out.r - big_r).abs() < (2.0 * sigma).max(0.03), "R{} ± {sigma}", out.r);
            assert!(out.arc_points as f64 > round.len() as f64 * 0.6);
            let Some(Shape::Line { a: la_a, b: la_b }) = s.0.entity(s.1).map(|e| e.shape.clone()) else { panic!("a line") };
            assert!((la_a.y - 10.0).abs() < 0.02);
            assert!((la_b.x - (20.0 - out.r)).abs() < 0.02);
        }

        // A line into an arc, and two arcs: the blend found tangent to each. A line along y =
        // 0 up to (20, 0), then an R15 arc about (35, 0) rising from it square, blended R3:
        // the blend's centre 3 above the line and 18 from (35, 0).
        {
            let big_r = 3.0;
            let bx = 35.0 - (18.0f64 * 18.0 - big_r * big_r).sqrt();
            let blend_c = DVec2::new(bx, big_r);
            let to_arc = (big_r - 0.0).atan2(bx - 35.0);
            let flat = noisy(&mut rand, &line(DVec2::ZERO, DVec2::new(bx, 0.0)));
            let blend = noisy(&mut rand, &arc(blend_c, big_r, -PI / 2.0, (15.0 * to_arc.sin() - big_r).atan2(35.0 + 15.0 * to_arc.cos() - bx)));
            let big = noisy(&mut rand, &arc(DVec2::new(35.0, 0.0), 15.0, to_arc, PI / 2.0));
            let half = blend.len() / 2;
            let mut s = two_pieces(
                Shape::Line { a: DVec2::ZERO, b: DVec2::new(20.0, 0.0) },
                [flat, blend[..half].to_vec()].concat(),
                Shape::Arc { c: DVec2::new(35.0, 0.0), r: 15.0, start: PI / 2.0, sweep: PI / 2.0 },
                [blend[half..].to_vec(), big].concat(),
                (AnchorAt::B, AnchorAt::B),
            );
            let one = rounded(&mut s, None);
            assert!(one.report.ok);
            let sigma = one.sigma.unwrap_or(f64::NAN);
            assert!((one.r - big_r).abs() < (2.0 * sigma).max(0.05), "R{} ± {sigma}", one.r);

            let r2 = 2.5;
            // Two R15 circles about (−12, 0) and (12, 0), meeting at the origin; the valley
            // between their tops blended: the blend touches both from outside, its centre 17.5
            // from each, on x = 0.
            let by = (17.5f64 * 17.5 - 12.0 * 12.0).sqrt();
            let bc = DVec2::new(0.0, by);
            let tl = by.atan2(12.0);
            let tr = PI - tl;
            let left = noisy(&mut rand, &arc(DVec2::new(-12.0, 0.0), 15.0, PI / 2.0, tl));
            let blend2 = noisy(&mut rand, &arc(bc, r2, PI + tl, 2.0 * PI - tl));
            let right = noisy(&mut rand, &arc(DVec2::new(12.0, 0.0), 15.0, tr, PI / 2.0));
            let h2 = blend2.len() / 2;
            let meet_at = (15.0f64 * 15.0 - 12.0 * 12.0).sqrt().atan2(12.0);
            let mut s2 = two_pieces(
                Shape::Arc { c: DVec2::new(-12.0, 0.0), r: 15.0, start: meet_at, sweep: PI / 2.0 - meet_at },
                [left, blend2[..h2].to_vec()].concat(),
                Shape::Arc { c: DVec2::new(12.0, 0.0), r: 15.0, start: PI / 2.0, sweep: PI - meet_at - PI / 2.0 },
                [blend2[h2..].to_vec(), right].concat(),
                (AnchorAt::A, AnchorAt::B),
            );
            let two = rounded(&mut s2, None);
            assert!(two.report.ok);
            let sigma = two.sigma.unwrap_or(f64::NAN);
            assert!((two.r - r2).abs() < (2.0 * sigma).max(0.05), "R{} ± {sigma}", two.r);
        }

        // Holds a radius given, and says why where no round fits.
        {
            let mut s = two_pieces(
                Shape::Line { a: DVec2::new(0.0, 10.0), b: DVec2::new(20.0, 10.0) },
                Vec::new(),
                Shape::Line { a: DVec2::new(20.0, 10.0), b: DVec2::new(20.0, -10.0) },
                Vec::new(),
                (AnchorAt::B, AnchorAt::A),
            );
            let out = rounded(&mut s, Some(4.0));
            assert!(out.report.ok);
            assert!(out.made.dimension.is_some());
            assert!((out.r - 4.0).abs() < 0.5e-6);
            let mut t = two_pieces(
                Shape::Line { a: DVec2::new(0.0, 10.0), b: DVec2::new(20.0, 10.0) },
                Vec::new(),
                Shape::Line { a: DVec2::new(20.0, 10.0), b: DVec2::new(20.0, 8.0) },
                Vec::new(),
                (AnchorAt::B, AnchorAt::A),
            );
            let too_big = round_corner(&mut t.0, t.1, t.2, Some(5.0)).unwrap_err();
            assert!(too_big.contains("does not fit"), "{too_big}");
            let same = round_corner(&mut t.0, t.1, t.1, None).unwrap_err();
            assert!(same.contains("two lines or arcs"), "{same}");
        }
    }

    #[test]
    fn a_corner_whose_pieces_carry_no_points_about_it_r0_02_from_one_point_warns() {
        // The fan wheel's case: the pieces that carried the corner's round were taken out. The
        // two lines left run sharp into the corner, and the few points off them lie outside
        // it, where no round goes: the search's best is its smallest radius, R0.02.
        let mut sketch = Sketch::default();
        let along: Vec<DVec2> = (0..81).map(|i| DVec2::new(i as f64 * 0.25, 10.0)).collect();
        let down: Vec<DVec2> = (0..81).map(|i| DVec2::new(20.0, 10.0 - i as f64 * 0.25)).collect();
        let a = sketch.add(Entity::new(Shape::Line { a: DVec2::new(0.0, 10.0), b: DVec2::new(20.0, 10.0) }).fitted([along, vec![DVec2::new(20.4, 10.3), DVec2::new(20.3, 10.4)]].concat(), 0.01));
        let b = sketch.add(Entity::new(Shape::Line { a: DVec2::new(20.0, 10.0), b: DVec2::new(20.0, -10.0) }).fitted([down, vec![DVec2::new(20.35, 10.35)]].concat(), 0.01));
        sketch.constrain(Rule::Coincident { p: Anchor::new(a, AnchorAt::B), q: Anchor::new(b, AnchorAt::A) });
        let made = round_corner(&mut sketch, a, b, None).expect("a round");
        assert_eq!(made.found, Some(Found::Limit));
        assert!((made.radius - 0.02).abs() < 0.5e-6, "{}", made.radius);
        let warning = round_unread(made.points, made.found).expect("a warning");
        assert!(warning.contains("radius not read; give radius"), "{warning}");
        if made.points < 3 {
            assert!(warning.starts_with(&format!("{} point", made.points)), "{warning}");
        }
    }

    #[test]
    fn a_round_read_from_its_points_does_not_warn_one_given_never_does() {
        assert!(round_unread(12, Some(Found::Read)).is_none());
        assert!(round_unread(0, None).is_none());
        assert_eq!(
            round_unread(1, Some(Found::Read)).as_deref(),
            Some("1 point on the round — radius not read; give radius, or re-fit the corner so its pieces carry its slice points")
        );
    }
}
