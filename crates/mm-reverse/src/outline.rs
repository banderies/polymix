// SPDX-License-Identifier: AGPL-3.0-only
//! The outline of a part seen square onto a plane: where a grid of rays meets the scan and
//! where it does not, the line between traced round by marching squares, and each loop read
//! as a CAD sketch would draw it — straight sides fitted as lines and met at their corners,
//! the rounds between them fitted as circles, a whole loop as one circle where it is round. A
//! flange's plan, a bezel's opening, a gasket's shape, read in one call instead of guessed
//! from a coarse height map. Ported from ScanRuler (`src/core/geometry/outline.ts`), © Stefan
//! Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! The traced line steps from cell to cell, so it is good to half a cell; the lines and
//! circles are least-squares fits over many of its points, and come out much closer than
//! that.

use std::collections::HashMap;

use glam::DVec2;

use crate::fit2d::fit_circle_2d;
use crate::js_fmt::js_round;
use crate::simplify::simplify_run;

/// A loop of the outline in the grid's own coordinates (u, v, mm), with whether it is a hole
/// in another and the area it closes.
#[derive(Debug, Clone, PartialEq)]
pub struct MaskLoop {
    pub points: Vec<DVec2>,
    pub hole: bool,
    pub area: f64,
}

/// The grid a mask lies on: cell (c, r) has its centre at u0 + (c + ½)·cell, v0 + (r + ½)·cell.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaskGrid {
    pub cols: usize,
    pub rows: usize,
    pub cell: f64,
    pub u0: f64,
    pub v0: f64,
}

/// The loops round the cells that are set in `mask` (row-major, row 0 at the low v), by
/// marching squares over the cell centres: each loop runs through the midpoints between a set
/// cell and an unset one. Two set cells that touch only at a corner are kept apart. Loops
/// closing less than `min_area` (mm²) are left out — specks of a ray that grazed an edge. The
/// largest first.
pub fn mask_loops(mask: &[u8], g: &MaskGrid, min_area: f64) -> Vec<MaskLoop> {
    let (cols, rows) = (g.cols as i64, g.rows as i64);
    let at = |c: i64, r: i64| c >= 0 && r >= 0 && c < cols && r < rows && mask[(r * cols + c) as usize] != 0;
    // A crossing's key: the cell edge it lies on, horizontal (between (c, r) and (c + 1, r))
    // or vertical (between (c, r) and (c, r + 1)), with the padding of one cell all round
    // counted in.
    let w = cols + 2;
    let h = rows + 2;
    let h_key = |c: i64, r: i64| (r + 1) * w + (c + 1);
    let v_key = |c: i64, r: i64| w * h + (r + 1) * w + (c + 1);
    let place = |k: i64| -> (f64, f64) {
        let vertical = k >= w * h;
        let i = if vertical { k - w * h } else { k };
        let c = (i % w) - 1;
        let r = i / w - 1;
        if vertical {
            (c as f64, r as f64 + 0.5)
        } else {
            (c as f64 + 0.5, r as f64)
        }
    };
    // The links, in the order the crossings were first met — the order the loops are walked
    // from.
    let mut next: HashMap<i64, Vec<i64>> = HashMap::new();
    let mut order: Vec<i64> = Vec::new();
    let mut link = |a: i64, b: i64| {
        for (x, y) in [(a, b), (b, a)] {
            let e = next.entry(x).or_insert_with(|| {
                order.push(x);
                Vec::new()
            });
            e.push(y);
        }
    };
    for r in -1..rows {
        for c in -1..cols {
            let (a, b, cc, d) = (at(c, r), at(c + 1, r), at(c + 1, r + 1), at(c, r + 1));
            let (bottom, top, left, right) = (h_key(c, r), h_key(c, r + 1), v_key(c, r), v_key(c + 1, r));
            let mut cross: Vec<i64> = Vec::with_capacity(4);
            if a != b {
                cross.push(bottom);
            }
            if b != cc {
                cross.push(right);
            }
            if d != cc {
                cross.push(top);
            }
            if a != d {
                cross.push(left);
            }
            if cross.len() == 2 {
                link(cross[0], cross[1]);
            } else if cross.len() == 4 {
                // A saddle: the two set corners cut off each on its own.
                if a {
                    link(bottom, left);
                    link(right, top);
                } else {
                    link(bottom, right);
                    link(top, left);
                }
            }
        }
    }
    let mut done: std::collections::HashSet<i64> = std::collections::HashSet::new();
    let mut loops: Vec<Vec<DVec2>> = Vec::new();
    for &start in &order {
        if done.contains(&start) {
            continue;
        }
        let mut keys = vec![start];
        done.insert(start);
        let mut prev = -1;
        let mut cur = start;
        loop {
            let Some(ns) = next.get(&cur) else { break };
            let to = if ns.first() != Some(&prev) { ns.first().copied() } else { ns.get(1).copied() };
            let Some(to) = to else { break };
            if to == start || done.contains(&to) {
                break;
            }
            done.insert(to);
            keys.push(to);
            prev = cur;
            cur = to;
        }
        if keys.len() < 3 {
            continue;
        }
        loops.push(
            keys.iter()
                .map(|&k| {
                    let (x, y) = place(k);
                    DVec2::new(g.u0 + (x + 0.5) * g.cell, g.v0 + (y + 0.5) * g.cell)
                })
                .collect(),
        );
    }
    let out: Vec<(Vec<DVec2>, f64)> = loops.into_iter().map(|p| {
        let a = signed_area(&p).abs();
        (p, a)
    }).filter(|l| l.1 >= min_area).collect();
    // A loop inside an odd number of others is a hole.
    let mut with_holes: Vec<MaskLoop> = out
        .iter()
        .enumerate()
        .map(|(i, (points, area))| {
            let around = out.iter().enumerate().filter(|(k, o)| *k != i && o.1 > *area && inside(points[0], &o.0)).count();
            MaskLoop { points: points.clone(), hole: around % 2 == 1, area: *area }
        })
        .collect();
    with_holes.sort_by(|a, b| b.area.total_cmp(&a.area));
    with_holes
}

fn signed_area(p: &[DVec2]) -> f64 {
    let n = p.len();
    let mut s = 0.0;
    let mut j = n.wrapping_sub(1);
    for i in 0..n {
        s += p[j].x * p[i].y - p[i].x * p[j].y;
        j = i;
    }
    s / 2.0
}

fn inside(q: DVec2, p: &[DVec2]) -> bool {
    let mut x = false;
    let n = p.len();
    let mut j = n.wrapping_sub(1);
    for i in 0..n {
        if (p[i].y > q.y) != (p[j].y > q.y) && q.x < ((p[j].x - p[i].x) * (q.y - p[i].y)) / (p[j].y - p[i].y) + p[i].x {
            x = !x;
        }
        j = i;
    }
    x
}

/// A round on a corner where two straight sides meet: the corner's index in the points.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShapeRound {
    pub at: usize,
    pub radius: f64,
    pub center: DVec2,
}

/// A side that is an arc, not a straight line: from the index of the point it starts at,
/// running to the next.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShapeArc {
    pub from: usize,
    pub radius: f64,
    pub center: DVec2,
}

/// A loop that is one circle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShapeCircle {
    pub center: DVec2,
    pub radius: f64,
    pub rms: f64,
}

/// A loop read as a sketch: its corners in order; each round on a corner where two straight
/// sides meet; each other side that is an arc; or one circle where the whole loop is round.
#[derive(Debug, Clone, PartialEq)]
pub struct LoopShape {
    pub points: Vec<DVec2>,
    pub rounds: Vec<ShapeRound>,
    pub arcs: Vec<ShapeArc>,
    pub circle: Option<ShapeCircle>,
    /// The worst distance of the traced loop from the shape, mm, and where on the loop it is.
    pub worst: f64,
    pub worst_at: Option<DVec2>,
}

#[derive(Debug, Clone, Copy)]
struct Circle {
    center: DVec2,
    radius: f64,
    rms: f64,
    max: f64,
}

fn circle_through(pts: &[DVec2]) -> Option<Circle> {
    if pts.len() < 3 {
        return None;
    }
    let (center, radius) = fit_circle_2d(pts)?;
    let mut s = 0.0;
    let mut max = 0.0f64;
    for p in pts {
        let d = ((*p - center).length() - radius).abs();
        s += d * d;
        max = max.max(d);
    }
    Some(Circle { center, radius, rms: (s / pts.len() as f64).sqrt(), max })
}

#[derive(Debug, Clone, Copy)]
struct Line {
    at: DVec2,
    dir: DVec2,
    rms: f64,
    max: f64,
}

/// The total-least-squares line through the points.
fn line_through(pts: &[DVec2]) -> Line {
    let n = pts.len() as f64;
    let m = pts.iter().fold(DVec2::ZERO, |s, p| s + *p) / n;
    let (mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0);
    for p in pts {
        let (x, y) = (p.x - m.x, p.y - m.y);
        sxx += x * x;
        sxy += x * y;
        syy += y * y;
    }
    let angle = 0.5 * (2.0 * sxy).atan2(sxx - syy);
    let dir = DVec2::new(angle.cos(), angle.sin());
    let mut s = 0.0;
    let mut max = 0.0f64;
    for p in pts {
        let d = (p.x - m.x) * dir.y - (p.y - m.y) * dir.x;
        s += d * d;
        max = max.max(d.abs());
    }
    Line { at: m, dir, rms: (s / n).sqrt(), max }
}

/// Where two lines meet; `None` when they run within a few degrees of parallel.
fn meet(a: &Line, b: &Line) -> Option<DVec2> {
    let den = a.dir.x * b.dir.y - a.dir.y * b.dir.x;
    if den.abs() < 4f64.to_radians().sin() {
        return None;
    }
    let t = ((b.at.x - a.at.x) * b.dir.y - (b.at.y - a.at.y) * b.dir.x) / den;
    Some(a.at + a.dir * t)
}

fn seg_dist(p: DVec2, a: DVec2, b: DVec2) -> f64 {
    let d = b - a;
    let l2 = d.dot(d);
    let t = if l2 > 0.0 { ((p - a).dot(d) / l2).clamp(0.0, 1.0) } else { 0.0 };
    (p.x - a.x - t * d.x).hypot(p.y - a.y - t * d.y)
}

/// A piece of the loop's reading, over the kept places `from` to `to` (a `from` below 0 runs
/// on round from the end).
#[derive(Debug, Clone, Copy)]
enum Piece {
    Line { line: Line, from: isize, to: isize },
    Arc { circle: Circle, from: isize, to: isize },
}

impl Piece {
    fn from(&self) -> isize {
        match *self {
            Piece::Line { from, .. } | Piece::Arc { from, .. } => from,
        }
    }
    fn to(&self) -> isize {
        match *self {
            Piece::Line { to, .. } | Piece::Arc { to, .. } => to,
        }
    }
    fn is_line(&self) -> bool {
        matches!(self, Piece::Line { .. })
    }
}

/// Where a line and an arc that runs on from it meet: the foot of the arc's centre on the
/// line, where the line is tangent to it within the tolerance; `None` otherwise.
fn touch(a: &Piece, b: &Piece, tolerance: f64) -> Option<DVec2> {
    let line = match (a, b) {
        (Piece::Line { line, .. }, _) | (_, Piece::Line { line, .. }) => line,
        _ => return None,
    };
    let arc = match (a, b) {
        (Piece::Arc { circle, .. }, _) | (_, Piece::Arc { circle, .. }) => circle,
        _ => return None,
    };
    let t = (arc.center - line.at).dot(line.dir);
    let foot = line.at + line.dir * t;
    if ((arc.center - foot).length() - arc.radius).abs() <= tolerance {
        Some(foot)
    } else {
        None
    }
}

/// The most thinned sides one line or arc may span.
const SPAN_MAX: usize = 40;

/// A traced loop read as a sketch, to within `tolerance` (mm): as few lines and arcs as follow
/// it that closely, a line taken over an arc that fits as well. The loop is thinned
/// (Ramer–Douglas–Peucker, to half the tolerance) to the places a piece may start, and the
/// fewest pieces are found over them (dynamic programming, from its sharpest corner round);
/// each piece is fitted over the traced points it spans. Two lines meet at a sharp corner; an
/// arc between two lines that meet is that corner's round — so a rounded rectangle comes back
/// as its four corners, each with its round. `cell` is the grid's, the size of the tracing's
/// steps.
pub fn loop_shape(traced: &[DVec2], tolerance: f64, cell: f64) -> LoopShape {
    let n = traced.len();
    let empty = |points: Vec<DVec2>, worst: f64| LoopShape { points, rounds: vec![], arcs: vec![], circle: None, worst, worst_at: None };
    if n == 0 {
        return empty(vec![], 0.0);
    }
    // The tracing's steps evened out a little: each point with its two neighbours.
    let lp: Vec<DVec2> = (0..n).map(|i| (traced[(i + n - 1) % n] + traced[i] * 2.0 + traced[(i + 1) % n]) / 4.0).collect();
    let fine = (tolerance / 2.0).max(0.3 * cell);
    if let Some(whole) = circle_through(&lp) {
        if whole.rms <= fine && whole.max <= tolerance {
            return LoopShape { points: vec![], rounds: vec![], arcs: vec![], circle: Some(ShapeCircle { center: whole.center, radius: whole.radius, rms: whole.rms }), worst: whole.max, worst_at: None };
        }
    }
    // Thinned, as a closed run from the point farthest from the first.
    let d0 = |i: usize| (lp[i] - lp[0]).length();
    let mut far = 0;
    for i in 1..n {
        if d0(i) > d0(far) {
            far = i;
        }
    }
    let flat: Vec<DVec2> = (0..=n).map(|i| lp[i % n]).collect();
    let mut thinned = simplify_run(&flat, 0, far + 1, fine);
    let second = simplify_run(&flat, far, n + 1, fine);
    if second.len() > 2 {
        thinned.extend_from_slice(&second[1..second.len() - 1]);
    }
    let m = thinned.len();
    if m < 3 {
        return empty(thinned.iter().map(|&i| lp[i]).collect(), 0.0);
    }
    // Started in the middle of the longest side, which is part of a line, so no corner and no
    // arc is cut in two where the loop closes: the places a piece may start, as running
    // indices from there once round.
    let end = |k: usize| if k + 1 < m { thinned[k + 1] } else { thinned[0] + n };
    let side_len = |k: usize| (lp[end(k) % n] - lp[thinned[k]]).length();
    let mut longest = 0;
    for k in 1..m {
        if side_len(k) > side_len(longest) {
            longest = k;
        }
    }
    let start = (js_round((thinned[longest] + end(longest)) as f64 / 2.0) as usize) % n;
    let mut mid: Vec<usize> = thinned.iter().map(|&i| if i <= start { i + n } else { i }).collect();
    mid.sort_unstable();
    let mut all = vec![start];
    all.extend(mid);
    all.push(start + n);
    let mut kept: Vec<usize> = Vec::with_capacity(all.len());
    for (k, &i) in all.iter().enumerate() {
        if k == 0 || i != all[k - 1] {
            kept.push(i);
        }
    }
    let big_m = kept.len() - 1;
    let points_of = |i: usize, j: usize| -> Vec<DVec2> { (kept[i]..=kept[j]).map(|k| lp[k % n]).collect() };
    let (mut lo, mut hi) = (DVec2::splat(f64::INFINITY), DVec2::splat(f64::NEG_INFINITY));
    for p in &lp {
        lo = lo.min(*p);
        hi = hi.max(*p);
    }
    let size = (hi.x - lo.x).max(hi.y - lo.y);
    // A line where one holds; an arc where one holds and a line does not, or where it follows
    // the points much more closely than the line does — a small round is within the tolerance
    // of a chamfer, and is still a round. An arc is held to the tracing's own scatter, a line
    // only to the tolerance: a gently curving side is drawn straight.
    let tight = (fine / 2.0).max(0.3 * cell);
    let piece_of = |i: usize, j: usize| -> Option<Piece> {
        let pts = points_of(i, j);
        let line = line_through(&pts);
        let straight = line.rms <= fine && line.max <= tolerance;
        if pts.len() >= 5 && !(straight && line.rms <= tight) {
            if let Some(circle) = circle_through(&pts) {
                if circle.rms <= tight && circle.max <= tolerance && circle.radius < size / 2.0 && (!straight || circle.rms < line.rms / 2.0) {
                    return Some(Piece::Arc { circle, from: i as isize, to: j as isize });
                }
            }
        }
        if straight {
            Some(Piece::Line { line, from: i as isize, to: j as isize })
        } else {
            None
        }
    };
    // The fewest pieces from the start to each place. Going back from a place, a piece that
    // fits neither way stops the search a little later: a longer one will not fit either.
    let mut best = vec![f64::INFINITY; big_m + 1];
    let mut back: Vec<Option<Piece>> = vec![None; big_m + 1];
    best[0] = 0.0;
    for j in 1..=big_m {
        let mut misses = 0;
        let mut i = j as isize - 1;
        while i >= (j as isize - SPAN_MAX as isize).max(0) && misses < 3 {
            let iu = i as usize;
            match piece_of(iu, j) {
                None => misses += 1,
                Some(p) => {
                    misses = 0;
                    if best[iu] + 1.0 < best[j] {
                        best[j] = best[iu] + 1.0;
                        back[j] = Some(p);
                    }
                }
            }
            i -= 1;
        }
    }
    let mut pieces: Vec<Piece> = Vec::new();
    let mut j = big_m;
    while j > 0 {
        let Some(p) = back[j] else { return empty(kept[..big_m].iter().map(|&k| lp[k % n]).collect(), fine) };
        pieces.insert(0, p);
        j = p.from() as usize;
    }
    // The start cut a line in two: one again, where one line holds them.
    if pieces.len() > 2 && pieces[0].is_line() && pieces[pieces.len() - 1].is_line() {
        let last = pieces[pieces.len() - 1];
        let mut pts = points_of(last.from() as usize, big_m);
        pts.extend(points_of(0, pieces[0].to() as usize));
        let line = line_through(&pts);
        if line.rms <= fine && line.max <= tolerance {
            pieces.pop();
            pieces[0] = Piece::Line { line, from: last.from() - big_m as isize, to: pieces[0].to() };
        }
    }
    // Each line fitted again over the middle of its points, away from its ends, where the
    // tracing has begun to turn into what comes next — kept where it still follows all of them
    // within the tolerance.
    for k in 0..pieces.len() {
        let Piece::Line { from, to, .. } = pieces[k] else { continue };
        let pts = if from < 0 {
            let mut v = points_of((from + big_m as isize) as usize, big_m);
            v.extend(points_of(0, to as usize));
            v
        } else {
            points_of(from as usize, to as usize)
        };
        let trim = (pts.len() as f64 * 0.2).floor() as usize;
        if pts.len() < 2 * trim + 5 {
            continue;
        }
        let line = line_through(&pts[trim..pts.len() - trim]);
        let off = pts.iter().map(|q| ((q.x - line.at.x) * line.dir.y - (q.y - line.at.y) * line.dir.x).abs()).fold(f64::NEG_INFINITY, f64::max);
        if off <= tolerance {
            pieces[k] = Piece::Line { line, from, to };
        }
    }
    // Each piece's start, in order: where two lines meet, a round's corner where the lines
    // either side of it meet, and anywhere else the place it starts from. A round is an arc
    // between two lines whose corner lies near it, outside the circle, about where the arc's
    // middle points — not a slot's end between two lines that never meet, nor a turn so far
    // round that the lines meet behind it.
    let pc = pieces.len() as isize;
    let at = |k: isize| pieces[(((k % pc) + pc) % pc) as usize];
    let round_on = |k: isize| -> Option<DVec2> {
        let (p, a, b) = (at(k), at(k - 1), at(k + 1));
        let (Piece::Arc { circle, from, to }, Piece::Line { line: la, .. }, Piece::Line { line: lb, .. }) = (p, a, b) else { return None };
        if pc < 3 {
            return None;
        }
        let corner = meet(&la, &lb)?;
        let c = circle.center;
        let out = (corner - c).length();
        let arc = points_of(from as usize, to as usize);
        let mid = arc[arc.len() >> 1];
        let toward = (mid - c).dot(corner - c) / ((mid - c).length() * out);
        if toward > 0.95 && out > circle.radius && out < 4.0 * circle.radius {
            Some(corner)
        } else {
            None
        }
    };
    let mut points: Vec<DVec2> = Vec::new();
    let mut rounds: Vec<ShapeRound> = Vec::new();
    let mut arcs: Vec<ShapeArc> = Vec::new();
    for k in 0..pc {
        let p = at(k);
        let corner = round_on(k);
        if let (Some(corner), Piece::Arc { circle, .. }) = (corner, p) {
            rounds.push(ShapeRound { at: points.len(), radius: circle.radius, center: circle.center });
            points.push(corner);
            continue;
        }
        // A line after a round starts at the round's corner, given already.
        if p.is_line() && round_on(k - 1).is_some() {
            continue;
        }
        let prev = at(k - 1);
        let met = match (&p, &prev) {
            (Piece::Line { line: lp_, .. }, Piece::Line { line: lq, .. }) => meet(lq, lp_),
            _ if p.is_line() != prev.is_line() => touch(&p, &prev, tolerance),
            _ => None,
        };
        if let Piece::Arc { circle, .. } = p {
            arcs.push(ShapeArc { from: points.len(), radius: circle.radius, center: circle.center });
        }
        let start_at = kept.get(((p.from() + big_m as isize) % big_m as isize) as usize).copied().unwrap_or(kept[0]);
        points.push(met.unwrap_or(lp[start_at % n]));
    }
    // How far the traced loop strays from the shape drawn: its sides, a round or an arc taken
    // as its circle.
    let mut worst = 0.0;
    let mut worst_at = None;
    for p in traced {
        let mut d = f64::INFINITY;
        for i in 0..points.len() {
            d = d.min(seg_dist(*p, points[i], points[(i + 1) % points.len()]));
        }
        for (c, r) in rounds.iter().map(|r| (r.center, r.radius)).chain(arcs.iter().map(|a| (a.center, a.radius))) {
            d = d.min(((*p - c).length() - r).abs());
        }
        if d > worst {
            worst = d;
            worst_at = Some(*p);
        }
    }
    LoopShape { points, rounds, arcs, circle: None, worst, worst_at }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A mask with a 10 × 6 block of cells set and a 2 × 2 hole in it.
    #[test]
    fn traces_a_block_and_its_hole() {
        let g = MaskGrid { cols: 14, rows: 10, cell: 1.0, u0: 0.0, v0: 0.0 };
        let mut mask = vec![0u8; g.cols * g.rows];
        for r in 2..8 {
            for c in 2..12 {
                mask[r * g.cols + c] = 1;
            }
        }
        for r in 4..6 {
            for c in 6..8 {
                mask[r * g.cols + c] = 0;
            }
        }
        let loops = mask_loops(&mask, &g, 0.0);
        assert_eq!(loops.len(), 2);
        assert!(!loops[0].hole && loops[1].hole);
        // Through the midpoints between set and unset centres: 10 × 6 and 2 × 2, less the four
        // corners cut across, an eighth of a cell each.
        assert!((loops[0].area - (60.0 - 0.5)).abs() < 1e-9, "{}", loops[0].area);
        assert!((loops[1].area - (4.0 - 0.5)).abs() < 1e-9, "{}", loops[1].area);
    }

    #[test]
    fn reads_a_rounded_rectangle_as_four_corners_with_their_rounds() {
        let (w, h, r) = (40.0, 20.0, 4.0);
        let mut pts = Vec::new();
        let corners = [(DVec2::new(w - r, r), -90.0f64), (DVec2::new(w - r, h - r), 0.0), (DVec2::new(r, h - r), 90.0), (DVec2::new(r, r), 180.0)];
        let sides = [(DVec2::new(r, 0.0), DVec2::new(w - r, 0.0)), (DVec2::new(w, r), DVec2::new(w, h - r)), (DVec2::new(w - r, h), DVec2::new(r, h)), (DVec2::new(0.0, h - r), DVec2::new(0.0, r))];
        for k in 0..4 {
            let (a, b) = sides[k];
            let count = ((b - a).length() / 0.1).round() as usize;
            for i in 0..count {
                pts.push(a + (b - a) * (i as f64 / count as f64));
            }
            let (c, s) = corners[k];
            for i in 0..63 {
                let t = (s + 90.0 * i as f64 / 63.0).to_radians();
                pts.push(c + DVec2::new(t.cos(), t.sin()) * r);
            }
        }
        let shape = loop_shape(&pts, 0.05, 0.05);
        assert_eq!(shape.points.len(), 4, "{shape:?}");
        assert_eq!(shape.rounds.len(), 4);
        for rd in &shape.rounds {
            assert!((rd.radius - r).abs() < 0.05, "{}", rd.radius);
        }
        assert!(shape.worst < 0.05);
    }
}
