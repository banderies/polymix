// SPDX-License-Identifier: AGPL-3.0-only
//! What a taper asks of the profile it leans. Ported from ScanRuler
//! (`plugins/reverse/core/taper.ts`), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! An extrude with a taper leans every side of its profile in, or out, as it goes, so the
//! section at any length is the profile's outline with every edge moved by that length times
//! the taper's tangent. The kernel's drafted prism keeps the profile's edges and leans each
//! one, whatever happens to the outline: past the length where an edge of the moved outline
//! shrinks to nothing, or runs into another, the solid it hands back crosses itself — a cone
//! run past its tip comes back as an hourglass, and the kernel calls it valid. So the outline
//! is moved here first, in the profile's own plane: the loops of the face's edges, each edge
//! sampled as a run of short segments, each segment moved along its normal, the corners where
//! the moved segments meet again.
//!
//! What is checked is the edges, not the segments. Where a round meets a side at an angle,
//! the leaned side runs over the round's first few segments and the round is trimmed back to
//! where the two meet again — as the kernel's leaned faces are — so a segment that gives way
//! at an edge's end is taken out and its neighbours meet past it. A length is refused only
//! where a whole edge has gone (a round past its radius, a side the others close over), where
//! an edge folds over in its middle (a bend in it tighter than the lean), or where two edges
//! that are not neighbours run into each other.

use std::collections::{HashMap, HashSet};

use glam::{DVec2, DVec3};
use serde::{Deserialize, Serialize};

use crate::js_fmt::to_fixed;

/// A closed loop as the face's edges run it: the corner points, the last joining the first,
/// and for each segment — from point i to the next — the edge it lies on, by a number the
/// face's loops share.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct EdgeLoop {
    pub points: Vec<DVec2>,
    pub edges: Vec<usize>,
}

/// A loop the taper is checked against: a plain closed loop of corner points (the last
/// joining the first), taken as a loop whose every segment is an edge of its own; or a loop
/// of the face's edges.
#[derive(Debug, Clone, PartialEq)]
pub enum TaperLoop {
    Plain(Vec<DVec2>),
    Edges(EdgeLoop),
}

impl From<Vec<DVec2>> for TaperLoop {
    fn from(points: Vec<DVec2>) -> Self {
        TaperLoop::Plain(points)
    }
}

impl From<EdgeLoop> for TaperLoop {
    fn from(l: EdgeLoop) -> Self {
        TaperLoop::Edges(l)
    }
}

/// What an edge of the face is read from: its sampled points and its two ends — the kernel's
/// edges as the kernel lists them.
#[derive(Debug, Clone, PartialEq)]
pub struct OutlineEdge {
    pub polyline: Vec<DVec3>,
    pub ends: [DVec3; 2],
}

/// The outline of a face that does not close, in words for the user.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("{0}")]
pub struct OutlineError(pub String);

/// The loops bounding a flat face, off its edges: each edge's points in the face's plane
/// through `origin` with the in-plane axes `u` and `v`, and the edges chained end to end into
/// loops. Each loop is run with the face on its left — the outer one counter-clockwise seen
/// down `u × v`, a hole clockwise. An edge whose end no other edge goes on from is an error:
/// a gap closed with a chord would check an outline the face does not have.
pub fn face_outline(edges: &[OutlineEdge], origin: DVec3, u: DVec3, v: DVec3) -> Result<Vec<EdgeLoop>, OutlineError> {
    let flat = |p: DVec3| -> DVec2 {
        let d = p - origin;
        DVec2::new(d.dot(u), d.dot(v))
    };
    let dist = |a: DVec2, b: DVec2| (a - b).length();
    // Each edge as its run of points, first to last, its ends the kernel's own.
    let mut runs: Vec<(usize, Vec<DVec2>)> = Vec::new();
    for (id, e) in edges.iter().enumerate() {
        if e.polyline.len() < 2 {
            continue;
        }
        let mut points: Vec<DVec2> = e.polyline.iter().map(|p| flat(*p)).collect();
        let a = flat(e.ends[0]);
        let b = flat(e.ends[1]);
        let last = points.len() - 1;
        if dist(points[0], b) + dist(points[last], a) < dist(points[0], a) + dist(points[last], b) {
            points.reverse();
        }
        points[0] = a;
        points[last] = b;
        runs.push((id, points));
    }
    let all: Vec<DVec2> = runs.iter().flat_map(|r| r.1.iter().copied()).collect();
    let span = span_of(&[&all]);
    let tol = 1e-3f64.max(span * 1e-5);
    let mut loops: Vec<EdgeLoop> = Vec::new();
    let mut used: HashSet<usize> = HashSet::new();
    let where_ = |p: DVec2| {
        let at = origin + u * p.x + v * p.y;
        format!("({}, {}, {})", to_fixed(at.x, 2), to_fixed(at.y, 2), to_fixed(at.z, 2))
    };
    for r in 0..runs.len() {
        if used.contains(&r) {
            continue;
        }
        used.insert(r);
        let mut lp = EdgeLoop::default();
        let take = |lp: &mut EdgeLoop, points: &[DVec2], id: usize| {
            for p in &points[..points.len() - 1] {
                lp.points.push(*p);
                lp.edges.push(id);
            }
        };
        take(&mut lp, &runs[r].1, runs[r].0);
        let start = runs[r].1[0];
        let mut end = runs[r].1[runs[r].1.len() - 1];
        while dist(end, start) > tol || lp.points.len() < 2 {
            // The edge that goes on from here: the one with an end nearest.
            let mut best: Option<usize> = None;
            let mut flip = false;
            let mut gap = f64::INFINITY;
            for (s, run) in runs.iter().enumerate() {
                if used.contains(&s) {
                    continue;
                }
                let p = &run.1;
                let d0 = dist(p[0], end);
                let d1 = dist(p[p.len() - 1], end);
                if d0 < gap {
                    (best, flip, gap) = (Some(s), false, d0);
                }
                if d1 < gap {
                    (best, flip, gap) = (Some(s), true, d1);
                }
            }
            let Some(best) = best.filter(|_| gap <= tol) else {
                return Err(OutlineError(format!(
                    "The profile's outline is broken at {}: no edge of the face goes on from there, so the taper cannot be checked against it.",
                    where_(end)
                )));
            };
            used.insert(best);
            let mut points = runs[best].1.clone();
            if flip {
                points.reverse();
            }
            take(&mut lp, &points, runs[best].0);
            end = points[points.len() - 1];
        }
        if lp.points.len() >= 3 {
            loops.push(lp);
        }
    }
    if loops.is_empty() {
        return Ok(loops);
    }
    // The outer loop is the one of most area; it runs counter-clockwise, the holes inside it
    // clockwise.
    let areas: Vec<f64> = loops.iter().map(|l| signed_area(&l.points)).collect();
    let mut outer = 0;
    for i in 1..loops.len() {
        if areas[i].abs() > areas[outer].abs() {
            outer = i;
        }
    }
    Ok(loops.into_iter().enumerate().map(|(i, l)| if (i == outer) == (areas[i] < 0.0) { reversed(&l) } else { l }).collect())
}

/// The area a closed polygon encloses by the shoelace sum, positive counter-clockwise.
pub(crate) fn signed_area(poly: &[DVec2]) -> f64 {
    let n = poly.len();
    let mut a = 0.0;
    for i in 0..n {
        a += poly[i].perp_dot(poly[(i + 1) % n]);
    }
    a / 2.0
}

/// The loop run the other way round, each segment still on its edge.
fn reversed(lp: &EdgeLoop) -> EdgeLoop {
    let n = lp.points.len();
    // Segment j of the reversed loop runs from point n−1−j to point n−2−j: the segment
    // n−2−j the other way.
    let mut points = lp.points.clone();
    points.reverse();
    EdgeLoop { points, edges: (0..n).map(|j| lp.edges[(2 * n - 2 - j) % n]).collect() }
}

/// How far across the loops reach — what the tolerances are taken from.
fn span_of(loops: &[&[DVec2]]) -> f64 {
    let mut lo = DVec2::splat(f64::INFINITY);
    let mut hi = DVec2::splat(f64::NEG_INFINITY);
    for l in loops {
        for p in l.iter() {
            lo = lo.min(*p);
            hi = hi.max(*p);
        }
    }
    if lo.x.is_finite() {
        (hi.x - lo.x).hypot(hi.y - lo.y)
    } else {
        0.0
    }
}

/// A loop without the segments too short to have a direction: the corner that repeats the
/// one before is dropped, with the segment ending there.
fn tidy(lp: &EdgeLoop, tol: f64) -> EdgeLoop {
    let mut points: Vec<DVec2> = Vec::new();
    let mut edges: Vec<usize> = Vec::new();
    for (i, p) in lp.points.iter().enumerate() {
        match points.last() {
            Some(last) if (*p - *last).length() <= tol => {
                if let Some(e) = edges.last_mut() {
                    *e = lp.edges[i];
                }
            }
            _ => {
                points.push(*p);
                edges.push(lp.edges[i]);
            }
        }
    }
    while points.len() > 1 && (points[0] - points[points.len() - 1]).length() <= tol {
        points.pop();
        edges.pop();
    }
    EdgeLoop { points, edges }
}

/// Where an outline breaks as it is moved: an edge gone or folded over, or two edges run into
/// each other — by the edges' numbers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InsetBreak {
    Vanishes { edge: usize },
    Folds { edge: usize },
    Crosses { edges: [usize; 2] },
}

impl InsetBreak {
    /// The edges named, one or two.
    pub fn edges(&self) -> Vec<usize> {
        match *self {
            InsetBreak::Vanishes { edge } | InsetBreak::Folds { edge } => vec![edge],
            InsetBreak::Crosses { edges } => edges.to_vec(),
        }
    }

    /// Why, as ScanRuler names it: `vanishes`, `folds` or `crosses`.
    pub fn why(&self) -> &'static str {
        match self {
            InsetBreak::Vanishes { .. } => "vanishes",
            InsetBreak::Folds { .. } => "folds",
            InsetBreak::Crosses { .. } => "crosses",
        }
    }
}

/// A corner of the moved loop: where it lies unmoved, and the way it goes per millimetre the
/// loop is moved.
#[derive(Debug, Clone, Copy)]
struct Corner {
    at: DVec2,
    mitre: DVec2,
}

fn normalize2(a: DVec2) -> Option<DVec2> {
    let l = a.length();
    if !(l > 1e-12) {
        None
    } else {
        Some(a / l)
    }
}

/// The loop with each segment moved `inset` to its left — into the face — and each corner
/// where two segments meet again, with the segment at an edge's end that gave way on the way
/// taken out and its neighbours made to meet past it, in the order they gave way. The moved
/// corners and each segment's edge, the segment from corner i on; or what broke.
fn inset_loop(lp: &EdgeLoop, inset: f64, tol: f64) -> Result<EdgeLoop, InsetBreak> {
    let points = &lp.points;
    let edges = &lp.edges;
    let n = points.len();
    let mut dir: Vec<DVec2> = Vec::with_capacity(n);
    for i in 0..n {
        match normalize2(points[(i + 1) % n] - points[i]) {
            Some(d) => dir.push(d),
            None => return Err(InsetBreak::Folds { edge: edges[i] }),
        }
    }
    let mut prev: Vec<usize> = (0..n).map(|i| (i + n - 1) % n).collect();
    let mut next: Vec<usize> = (0..n).map(|i| (i + 1) % n).collect();
    let mut left: HashMap<usize, usize> = HashMap::new();
    for e in edges {
        *left.entry(*e).or_insert(0) += 1;
    }
    // Two segments' moved lines meet off where they meet unmoved, along their normals' sum:
    // the mitre, as the leaned faces of a prism meet. Lines that run back on each other never
    // meet again.
    let mitre_of = |a: usize, b: usize| -> Option<DVec2> {
        let na = dir[a].perp();
        let nb = dir[b].perp();
        let denom = 1.0 + na.dot(nb);
        if denom < 1e-9 {
            return None;
        }
        Some((na + nb) / denom)
    };
    // The corner of each segment's start: the corner it shares with the one before.
    let mut start: Vec<Corner> = Vec::with_capacity(n);
    for i in 0..n {
        match mitre_of(prev[i], i) {
            Some(mitre) => start.push(Corner { at: points[i], mitre }),
            None => return Err(InsetBreak::Folds { edge: edges[i] }),
        }
    }
    // A segment's length along itself, moved all the way, and the share of the way at which
    // it gave way — none while it keeps its direction.
    let length_of = |i: usize, start: &[Corner], next: &[usize]| -> (f64, f64) {
        let a = start[i];
        let b = start[next[i]];
        let was = (b.at - a.at).dot(dir[i]);
        let now = was + inset * (b.mitre - a.mitre).dot(dir[i]);
        let gone = if now > tol {
            f64::INFINITY
        } else if was > 0.0 {
            was / (was - now)
        } else {
            0.0
        };
        (now, gone)
    };
    let mut alive = vec![true; n];
    let mut alive_count = n;
    loop {
        let mut first: Option<usize> = None;
        let mut when = f64::INFINITY;
        for i in 0..n {
            if !alive[i] {
                continue;
            }
            let (_, gone) = length_of(i, &start, &next);
            if gone < when {
                (first, when) = (Some(i), gone);
            }
        }
        let Some(first) = first else { break };
        let e = edges[first];
        let a = prev[first];
        let b = next[first];
        // A segment with its own edge on both sides gave way in the edge's middle: the edge
        // folds over itself there — or, all of it turned over at once, it is gone, as a round
        // past its radius.
        if edges[a] == e && edges[b] == e {
            let all_over = (0..n).filter(|&i| alive[i]).all(|i| edges[i] != e || length_of(i, &start, &next).0 <= tol);
            return Err(if all_over { InsetBreak::Vanishes { edge: e } } else { InsetBreak::Folds { edge: e } });
        }
        let count = left.get(&e).copied().unwrap_or(1) - 1;
        left.insert(e, count);
        if count == 0 || alive_count <= 3 {
            return Err(InsetBreak::Vanishes { edge: e });
        }
        alive[first] = false;
        alive_count -= 1;
        next[a] = b;
        prev[b] = a;
        // The neighbours meet where their lines do, unmoved; lines that run side by side
        // meet nowhere, unless they are one line.
        let mitre = mitre_of(a, b);
        let c = dir[a].perp_dot(dir[b]);
        let pa = points[a];
        let pb = start[b].at;
        let at = if c.abs() > 1e-12 {
            let s = (pb - pa).perp_dot(dir[b]) / c;
            pa + dir[a] * s
        } else if dir[a].perp_dot(pb - pa).abs() <= tol {
            pb
        } else {
            return Err(InsetBreak::Folds { edge: e });
        };
        let Some(mitre) = mitre else { return Err(InsetBreak::Folds { edge: e }) };
        start[b] = Corner { at, mitre };
    }
    let mut order: Vec<usize> = Vec::new();
    let Some(first) = (0..n).find(|&i| alive[i]) else { return Ok(EdgeLoop::default()) };
    let mut i = first;
    loop {
        order.push(i);
        i = next[i];
        if i == first {
            break;
        }
    }
    Ok(EdgeLoop { points: order.iter().map(|&i| start[i].at + start[i].mitre * inset).collect(), edges: order.iter().map(|&i| edges[i]).collect() })
}

#[derive(Debug, Clone, Copy)]
struct Segment {
    a: DVec2,
    b: DVec2,
    lp: usize,
    index: usize,
    count: usize,
    edge: usize,
    x0: f64,
    x1: f64,
    y0: f64,
    y1: f64,
}

/// Whether two segments meet — cross, touch or run over one another — within `tol`
/// millimetres.
fn segments_meet(s: &Segment, t: &Segment, tol: f64) -> bool {
    let side = |a: DVec2, b: DVec2, p: DVec2| -> f64 {
        let d = b - a;
        let l = d.length();
        if l > 0.0 {
            d.perp_dot(p - a) / l
        } else {
            (p - a).length()
        }
    };
    let within = |a: DVec2, b: DVec2, p: DVec2| -> bool {
        let d = b - a;
        let l2 = d.dot(d);
        if l2 == 0.0 {
            return (p - a).length() <= tol;
        }
        let t = (p - a).dot(d) / l2;
        let l = l2.sqrt();
        t >= -tol / l && t <= 1.0 + tol / l
    };
    let o1 = side(s.a, s.b, t.a);
    let o2 = side(s.a, s.b, t.b);
    let o3 = side(t.a, t.b, s.a);
    let o4 = side(t.a, t.b, s.b);
    if ((o1 > tol && o2 < -tol) || (o1 < -tol && o2 > tol)) && ((o3 > tol && o4 < -tol) || (o3 < -tol && o4 > tol)) {
        return true;
    }
    (o1.abs() <= tol && within(s.a, s.b, t.a)) || (o2.abs() <= tol && within(s.a, s.b, t.b)) || (o3.abs() <= tol && within(t.a, t.b, s.a)) || (o4.abs() <= tol && within(t.a, t.b, s.b))
}

/// The edges of two segments of the loops that meet though they are not neighbours; `None`
/// where none do. Of the pairs that meet, the two most nearly parallel are named — two sides
/// closing on each other, rather than one running into the other's end.
fn any_meet(loops: &[EdgeLoop], tol: f64) -> Option<[usize; 2]> {
    let mut segs: Vec<Segment> = Vec::new();
    for (li, l) in loops.iter().enumerate() {
        let n = l.points.len();
        for i in 0..n {
            let a = l.points[i];
            let b = l.points[(i + 1) % n];
            segs.push(Segment { a, b, lp: li, index: i, count: n, edge: l.edges[i], x0: a.x.min(b.x) - tol, x1: a.x.max(b.x) + tol, y0: a.y.min(b.y) - tol, y1: a.y.max(b.y) + tol });
        }
    }
    segs.sort_by(|p, q| p.x0.total_cmp(&q.x0));
    let mut met: Option<[usize; 2]> = None;
    let mut across = f64::INFINITY;
    for i in 0..segs.len() {
        let s = &segs[i];
        let mut j = i + 1;
        while j < segs.len() && segs[j].x0 <= s.x1 {
            let t = &segs[j];
            j += 1;
            if t.y0 > s.y1 || t.y1 < s.y0 {
                continue;
            }
            if s.lp == t.lp {
                let gap = s.index.abs_diff(t.index);
                if gap == 1 || gap == s.count - 1 {
                    continue;
                }
            }
            if !segments_meet(s, t, tol) {
                continue;
            }
            let sin = match (normalize2(s.b - s.a), normalize2(t.b - t.a)) {
                (Some(ds), Some(dt)) => ds.perp_dot(dt).abs(),
                _ => 1.0,
            };
            if sin < across {
                (met, across) = (Some([s.edge, t.edge]), sin);
            }
        }
    }
    met
}

/// Plain loops as loops of edges, each segment an edge of its own, numbered on across the
/// plain loops from 0.
fn as_edge_loops(loops: &[TaperLoop]) -> Vec<EdgeLoop> {
    let mut next = 0;
    loops
        .iter()
        .map(|l| match l {
            TaperLoop::Plain(points) => {
                let edges = (0..points.len()).map(|k| next + k).collect();
                next += points.len();
                EdgeLoop { points: points.clone(), edges }
            }
            TaperLoop::Edges(e) => e.clone(),
        })
        .collect()
}

fn edge_loops_break(edge_loops: &[EdgeLoop], inset: f64) -> Option<InsetBreak> {
    let span = span_of(&edge_loops.iter().map(|l| l.points.as_slice()).collect::<Vec<_>>());
    let first_edge = || edge_loops.first().and_then(|l| l.edges.first().copied()).unwrap_or(0);
    if !(span > 0.0) {
        return Some(InsetBreak::Vanishes { edge: first_edge() });
    }
    let tol = span * 1e-9;
    let clean: Vec<EdgeLoop> = edge_loops.iter().map(|l| tidy(l, tol)).filter(|l| l.points.len() >= 3).collect();
    if clean.is_empty() {
        return Some(InsetBreak::Vanishes { edge: first_edge() });
    }
    let mut moved: Vec<EdgeLoop> = Vec::new();
    for l in &clean {
        match inset_loop(l, inset, tol) {
            Ok(m) => moved.push(m),
            Err(b) => return Some(b),
        }
    }
    any_meet(&moved, tol).map(|edges| InsetBreak::Crosses { edges })
}

/// Where the outline breaks with every edge moved `inset` millimetres into the face — out of
/// it for a negative inset: an edge gone or folded over, or two edges that are not neighbours
/// run into each other; `None` where it keeps its shape. Loops as `face_outline` gives them,
/// the face on the left; a plain loop of points is one whose every segment is an edge.
pub fn inset_break(loops: &[TaperLoop], inset: f64) -> Option<InsetBreak> {
    edge_loops_break(&as_edge_loops(loops), inset)
}

/// Whether the outline keeps its shape moved `inset` — see `inset_break`.
pub fn inset_holds(loops: &[TaperLoop], inset: f64) -> bool {
    inset_break(loops, inset).is_none()
}

/// What broke where a taper ran short: the break, and the middle of each edge it names,
/// unmoved.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaperBreak {
    pub broke: InsetBreak,
    pub at: Vec<DVec2>,
}

/// How far a taper runs, and where the outline broke when it is short of the length asked.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaperReach {
    pub reach: f64,
    pub broke: Option<TaperBreak>,
}

/// How far a taper can run before the outline it moves breaks: `length` when it holds all the
/// way, else the length where it last holds, to a ten-thousandth of `length`, and what broke
/// just past it. `lean_in` is the tangent of the taper, positive where the sides lean in. An
/// outline that does not hold even unmoved is one the kernel made oddly, not one the taper
/// broke — it is let through.
pub fn taper_limit(loops: &[TaperLoop], lean_in: f64, length: f64) -> TaperReach {
    let edge_loops = as_edge_loops(loops);
    let holds = |inset: f64| edge_loops_break(&edge_loops, inset).is_none();
    if !(lean_in.abs() > 0.0) || holds(length * lean_in) || !holds(0.0) {
        return TaperReach { reach: length, broke: None };
    }
    let mut lo = 0.0;
    let mut hi = length;
    while hi - lo > length * 1e-4 {
        let m = (lo + hi) / 2.0;
        if holds(m * lean_in) {
            lo = m;
        } else {
            hi = m;
        }
    }
    match edge_loops_break(&edge_loops, hi * lean_in) {
        Some(broke) => TaperReach { reach: lo, broke: Some(TaperBreak { broke, at: broke.edges().iter().map(|e| middle_of(&edge_loops, *e)).collect() }) },
        None => TaperReach { reach: lo, broke: None },
    }
}

/// How far a taper can run before the outline breaks — see `taper_limit`.
pub fn taper_reach(loops: &[TaperLoop], lean_in: f64, length: f64) -> f64 {
    taper_limit(loops, lean_in, length).reach
}

/// The middle of an edge of the loops, unmoved: its middle segment's.
fn middle_of(loops: &[EdgeLoop], edge: usize) -> DVec2 {
    for l in loops {
        let on: Vec<usize> = l.edges.iter().enumerate().filter(|(_, e)| **e == edge).map(|(i, _)| i).collect();
        if on.is_empty() {
            continue;
        }
        let i = on[on.len() / 2];
        let a = l.points[i];
        let b = l.points[(i + 1) % l.points.len()];
        // An edge of an odd count of segments has its middle in one; of an even count, where
        // two meet.
        return if on.len() % 2 == 1 { (a + b) / 2.0 } else { a };
    }
    DVec2::ZERO
}

#[cfg(test)]
mod tests {
    //! Ported from ScanRuler's `plugins/reverse/tests/reTaper.test.ts` and
    //! `reTaperEdges.test.ts`: the outline a taper moves. The extrudes built through the
    //! kernel, and the face read off the kernel's own edges, are the app's to test.
    use super::*;
    use std::f64::consts::PI;

    fn p(x: f64, y: f64) -> DVec2 {
        DVec2::new(x, y)
    }

    fn rect(x0: f64, y0: f64, x1: f64, y1: f64) -> TaperLoop {
        TaperLoop::Plain(vec![p(x0, y0), p(x1, y0), p(x1, y1), p(x0, y1)])
    }

    fn ring_points(r: f64, n: usize) -> Vec<DVec2> {
        (0..n).map(|i| p(r * (2.0 * PI * i as f64 / n as f64).cos(), r * (2.0 * PI * i as f64 / n as f64).sin())).collect()
    }

    fn plain(points: &[(f64, f64)]) -> TaperLoop {
        TaperLoop::Plain(points.iter().map(|&(x, y)| p(x, y)).collect())
    }

    fn dumbbell() -> TaperLoop {
        plain(&[(0.0, 0.0), (20.0, 0.0), (20.0, 9.0), (30.0, 9.0), (30.0, 0.0), (50.0, 0.0), (50.0, 20.0), (30.0, 20.0), (30.0, 11.0), (20.0, 11.0), (20.0, 20.0), (0.0, 20.0)])
    }

    #[test]
    fn holds_a_rectangle_until_its_narrow_side_closes() {
        let r = [rect(0.0, 0.0, 40.0, 20.0)];
        assert!(inset_holds(&r, 0.0));
        assert!(inset_holds(&r, 9.9));
        assert!(!inset_holds(&r, 10.1));
        // Leaning out, a convex outline only grows.
        assert!(inset_holds(&r, -500.0));
    }

    #[test]
    fn catches_a_neck_that_pinches_off_though_no_edge_shrinks_away() {
        // Two 20 mm squares joined by a neck 2 mm wide: every edge keeps its length moved in
        // by 1.5, but the neck's two walls pass each other.
        assert!(inset_holds(&[dumbbell()], 0.9));
        assert!(!inset_holds(&[dumbbell()], 1.1));
    }

    #[test]
    fn catches_a_slot_or_a_hole_closing_as_the_sides_lean_out() {
        let slotted = plain(&[(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (11.0, 20.0), (11.0, 5.0), (9.0, 5.0), (9.0, 20.0), (0.0, 20.0)]);
        assert!(inset_holds(&[slotted.clone()], -0.9));
        assert!(!inset_holds(&[slotted], -1.1));
        // A plate with a round hole of radius 3, the hole run clockwise.
        let mut hole = ring_points(3.0, 64);
        hole.reverse();
        let plate = [rect(-20.0, -10.0, 20.0, 10.0), TaperLoop::Plain(hole)];
        assert!(inset_holds(&plate, -2.9));
        assert!(!inset_holds(&plate, -3.1));
        // Leaning in, the hole opens and the wall round it thins: 7 mm of it.
        assert!(inset_holds(&plate, 3.4));
        assert!(!inset_holds(&plate, 3.6));
    }

    #[test]
    fn closes_a_round_where_its_radius_runs_out() {
        assert!(inset_holds(&[TaperLoop::Plain(ring_points(5.0, 64))], 4.9));
        assert!(!inset_holds(&[TaperLoop::Plain(ring_points(5.0, 64))], 5.1));
    }

    #[test]
    fn finds_how_far_a_taper_can_run() {
        // 45°: the 20 mm side closes 10 mm out.
        assert!((taper_reach(&[rect(0.0, 0.0, 40.0, 20.0)], 1.0, 30.0) - 10.0).abs() < 0.005);
        assert_eq!(taper_reach(&[rect(0.0, 0.0, 40.0, 20.0)], 1.0, 8.0), 8.0);
        assert_eq!(taper_reach(&[rect(0.0, 0.0, 40.0, 20.0)], -1.0, 30.0), 30.0);
    }

    const R_TIP: f64 = 0.9;
    const TIP: f64 = PI * 2.0 / 3.0;
    const SAG: f64 = 0.6;

    /// A fan blade's plan, 40 mm between its tip centres: two long sides bowed out 0.6 mm,
    /// 2.8 mm apart at the middle, and an R0.9 round at each tip that meets the sides at about
    /// 26°, not tangent to them. Its pieces as arcs: centre, radius, start and sweep,
    /// counter-clockwise.
    fn blade_arcs() -> Vec<(DVec2, f64, f64, f64)> {
        let a = p(R_TIP * TIP.cos(), R_TIP * TIP.sin());
        let half = 20.0 - a.x;
        let r = (half * half + SAG * SAG) / (2.0 * SAG);
        let lean = (half / r).asin();
        vec![
            // The left tip, the bottom side, the right tip, the top side.
            (p(0.0, 0.0), R_TIP, TIP, TIP),
            (p(20.0, -a.y - SAG + r), r, -PI / 2.0 - lean, 2.0 * lean),
            (p(40.0, 0.0), R_TIP, -TIP / 2.0, TIP),
            (p(20.0, a.y + SAG - r), r, PI / 2.0 - lean, 2.0 * lean),
        ]
    }

    /// The blade's outline as its four edges, each arc in `n` segments.
    fn blade_loop(n: usize) -> EdgeLoop {
        let mut lp = EdgeLoop::default();
        for (e, (c, r, start, sweep)) in blade_arcs().into_iter().enumerate() {
            for i in 0..n {
                let t = start + sweep * i as f64 / n as f64;
                lp.points.push(c + p(r * t.cos(), r * t.sin()));
                lp.edges.push(e);
            }
        }
        lp
    }

    /// 3° over 12.6 mm: how far each side moves.
    fn inset() -> f64 {
        12.6 * 3f64.to_radians().tan()
    }

    #[test]
    fn trims_a_round_where_a_side_meets_it_at_an_angle_instead_of_refusing_its_first_segment() {
        let blade = blade_loop(24);
        assert!(inset_holds(&[blade.clone().into()], inset()));
        assert!(inset_holds(&[blade.clone().into()], -inset()));
        // The same points with every segment an edge of its own: the round's first segment
        // gives way at the corner, and that was a refusal.
        assert_eq!(inset_break(&[TaperLoop::Plain(blade.points)], inset()).map(|b| b.why()), Some("vanishes"));
    }

    #[test]
    fn still_refuses_where_a_whole_edge_vanishes() {
        // The tips close as the sides close in on them: each R0.9 round is trimmed shorter
        // until it is gone.
        let broke = inset_break(&[blade_loop(24).into()], 1.2);
        let Some(InsetBreak::Vanishes { edge }) = broke else { panic!("{broke:?}") };
        assert!([0, 2].contains(&edge));
        // A round of radius 5 as one edge, moved in past its radius.
        let ring = EdgeLoop { points: ring_points(5.0, 64), edges: vec![0; 64] };
        assert_eq!(inset_break(&[ring.clone().into()], 4.9), None);
        assert_eq!(inset_break(&[ring.into()], 5.1), Some(InsetBreak::Vanishes { edge: 0 }));
    }

    #[test]
    fn refuses_an_edge_that_folds_over_in_its_middle() {
        // A 20 × 10 plate whose top-right corner is one edge with the side and the top: a
        // straight run, a quarter round of radius 1, a straight run.
        let mut corner = vec![p(20.0, 0.0)];
        for i in 0..=16 {
            let t = (PI / 2.0) * (i as f64 / 16.0);
            corner.push(p(19.0 + t.cos(), 9.0 + t.sin()));
        }
        corner.push(p(0.0, 10.0));
        let mut points = vec![p(0.0, 0.0)];
        points.extend_from_slice(&corner[..corner.len() - 1]);
        points.push(p(0.0, 10.0));
        let mut edges = vec![0];
        edges.extend(std::iter::repeat_n(1, corner.len() - 1));
        edges.push(2);
        let plate = EdgeLoop { points, edges };
        assert_eq!(inset_break(&[plate.clone().into()], 0.9), None);
        assert_eq!(inset_break(&[plate.into()], 1.2), Some(InsetBreak::Folds { edge: 1 }));
    }

    #[test]
    fn refuses_two_edges_that_are_not_neighbours_running_into_each_other() {
        // Two 20 mm squares joined by a neck 2 mm wide: the neck's walls meet.
        assert_eq!(inset_break(&[dumbbell()], 0.9), None);
        let broke = inset_break(&[dumbbell()], 1.1);
        let Some(InsetBreak::Crosses { edges }) = broke else { panic!("{broke:?}") };
        // The neck's corners all meet at once: an edge from below it, from [20,0] round to
        // [30,9], and one from above, [30,20] round to [20,20].
        let (low, high) = (edges[0].min(edges[1]), edges[0].max(edges[1]));
        assert!([1, 2, 3].contains(&low), "{edges:?}");
        assert!([7, 8, 9].contains(&high), "{edges:?}");
    }

    /// ScanRuler's reTaper "reads a face's loops off the kernel's edges, the face on their
    /// left", with the edges the kernel lists for a 40 × 20 plate round a hole of radius 3
    /// given by hand: four lines, out of order and either way round, and the hole's circle
    /// sampled as a 64-gon whose two ends are one point — run counter-clockwise, so that the
    /// hole has to be turned round.
    #[test]
    fn reads_a_faces_loops_off_its_edges_the_face_on_their_left() {
        let at = |x: f64, y: f64| DVec3::new(x, y, 0.0);
        let line = |a: (f64, f64), b: (f64, f64)| OutlineEdge { polyline: vec![at(a.0, a.1), at(b.0, b.1)], ends: [at(a.0, a.1), at(b.0, b.1)] };
        let ring: Vec<DVec3> = (0..=64).map(|i| (2.0 * PI * i as f64) / 64.0).map(|t| at(20.0 + 3.0 * t.cos(), 10.0 + 3.0 * t.sin())).collect();
        let circle = OutlineEdge { ends: [ring[0], ring[64]], polyline: ring };
        let plate = vec![line((0.0, 0.0), (40.0, 0.0)), line((40.0, 20.0), (40.0, 0.0)), line((40.0, 20.0), (0.0, 20.0)), line((0.0, 0.0), (0.0, 20.0))];
        let mut with_hole = plate.clone();
        with_hole.insert(2, circle);
        let loops = face_outline(&with_hole, DVec3::ZERO, DVec3::X, DVec3::Y).expect("the plate closes");
        assert_eq!(loops.len(), 2);
        let mut areas: Vec<f64> = loops.iter().map(|l| signed_area(&l.points)).collect();
        areas.sort_by(|a, b| b.total_cmp(a));
        assert!((areas[0] - 800.0).abs() < 5e-4, "{areas:?}");
        assert!((areas[1] + PI * 9.0).abs() < 0.5, "{areas:?}");
        // The hole turned round keeps its one edge on every segment.
        let hole = loops.iter().find(|l| signed_area(&l.points) < 0.0).expect("a hole");
        assert!(hole.edges.iter().all(|&e| e == 2), "{:?}", hole.edges);
        // Seen from the other side, the same loop comes back the same way round, each segment
        // still on the edge it names.
        let flipped = face_outline(&plate, DVec3::ZERO, DVec3::Y, DVec3::X).expect("the plate closes");
        assert_eq!(flipped.len(), 1);
        assert!((signed_area(&flipped[0].points) - 800.0).abs() < 5e-4);
        let l = &flipped[0];
        for (j, &e) in l.edges.iter().enumerate() {
            let m = (l.points[j] + l.points[(j + 1) % l.points.len()]) / 2.0;
            let m3 = DVec3::Y * m.x + DVec3::X * m.y;
            let [a, b] = plate[e].ends;
            let off = (m3 - a).cross(b - a).length() / (b - a).length();
            assert!(off < 1e-9, "segment {j} is not on edge {e}");
        }
    }

    #[test]
    fn reads_the_loops_off_the_faces_edges_and_calls_a_gap_in_them_an_error() {
        let at = |x: f64, y: f64| DVec3::new(x, y, 0.0);
        let line = |a: (f64, f64), b: (f64, f64)| OutlineEdge { polyline: vec![at(a.0, a.1), at(b.0, b.1)], ends: [at(a.0, a.1), at(b.0, b.1)] };
        // A square, its edges listed out of order and two of them backwards.
        let square = vec![line((0.0, 0.0), (10.0, 0.0)), line((0.0, 10.0), (10.0, 10.0)), line((0.0, 10.0), (0.0, 0.0)), line((10.0, 10.0), (10.0, 0.0))];
        let loops = face_outline(&square, DVec3::ZERO, DVec3::X, DVec3::Y).expect("the square closes");
        assert_eq!(loops.len(), 1);
        assert_eq!(loops[0].points.len(), 4);
        assert_eq!(loops[0].edges.iter().collect::<HashSet<_>>().len(), 4);
        let as_loops: Vec<TaperLoop> = loops.into_iter().map(TaperLoop::from).collect();
        assert!(inset_holds(&as_loops, 4.9));
        // One side missing: no chord across the gap.
        let broken = face_outline(&square[1..], DVec3::ZERO, DVec3::X, DVec3::Y);
        let Err(e) = broken else { panic!("a gap was closed") };
        assert!(e.0.contains("outline is broken at ("), "{e}");
    }
}
