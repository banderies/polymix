// SPDX-License-Identifier: AGPL-3.0-only
//! A fit that lands on what the sketch already holds. Ported from ScanRuler
//! (`plugins/reverse/core/shareFit.ts`), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! A sketch fitted at one level and then at another — the deck of a plate and the wing below
//! it — has walls that run through both: drawn twice, a hair apart, they leave small steps all
//! round the part. So before a level's outline is cut into pieces, every slice point within
//! `share` mm of a piece already there, running the piece's way, goes to that piece: that
//! stretch of wall is drawn already. On a fitted piece the points join its own, and the solve
//! refits it on the evidence of every level; a drawn or projected piece takes none. What is
//! left is fitted as open runs, and each run's ends land on the piece they reach — on its end,
//! or held on it within it (a T-junction, the solver's bounded point-on).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::f64::consts::PI;

use glam::DVec2;

use crate::chain_outline::chain_closed;
use crate::corners::PlacedSegment;
use crate::sketch::{anchor_point, ends_of, nearest_on, Anchor, AnchorAt, ConstraintId, Entity, EntityId, OnTarget, Rule, Shape, Sketch};

/// How near a slice point lies to a piece to be taken as that piece's wall, mm: steps under it
/// are one surface.
pub const SHARE_MM: f64 = 0.1;
/// How far a stretch of the outline may turn off the piece it lies along and still be that
/// piece's, degrees.
pub const SHARE_DEG: f64 = 5.0;
/// An end of a run this close to an end of the piece it reaches is put on that end (the sketch
/// input's snap distance).
const LAND_SNAP_MM: f64 = 0.05;
/// How far past a piece's end a run may end and still be landed on that end, mm: the run
/// leaves the piece where its points first stray from it, a slice spacing or two on.
const LAND_REACH_MM: f64 = 1.0;
/// The fewest points a run needs to be fitted (`runs_where`'s four).
const MIN_RUN_POINTS: usize = 4;
/// A stretch shorter than this along its piece is too short to tell its way: it lies across
/// the piece rather than along it.
const MIN_STRETCH_MM: f64 = 0.4;
/// How near the way a run arrives at a piece may lie to the piece's own way there and the end
/// still be held on it, degrees.
const LAND_GRAZE_DEG: f64 = 10.0;

/// Whether a piece is one a fit may land on: a line, an arc or a circle.
fn is_carrier(shape: &Shape) -> bool {
    matches!(shape, Shape::Line { .. } | Shape::Arc { .. } | Shape::Circle { .. })
}

/// The pieces a fit may land on: the sketch's lines, arcs and circles, but its construction
/// lines.
pub fn share_pieces(sketch: &Sketch) -> Vec<&Entity> {
    sketch.entities.iter().filter(|e| !e.construction && is_carrier(&e.shape)).collect()
}

/// Where a point's foot lies on a piece — on the piece itself, not its line or circle run on —
/// and how far off it the point is; `None` when the foot falls past an end.
fn foot_on(e: &Shape, p: DVec2) -> Option<f64> {
    match *e {
        Shape::Spline(_) => None,
        Shape::Line { a, b } => {
            let (dx, dy) = (b.x - a.x, b.y - a.y);
            let ll = dx * dx + dy * dy;
            if !(ll > 0.0) {
                return None;
            }
            let t = ((p.x - a.x) * dx + (p.y - a.y) * dy) / ll;
            if !(0.0..=1.0).contains(&t) {
                return None;
            }
            Some(((p.y - a.y) * dx - (p.x - a.x) * dy).abs() / ll.sqrt())
        }
        Shape::Circle { c, r } => Some(((p.x - c.x).hypot(p.y - c.y) - r).abs()),
        Shape::Arc { c, r, start, sweep } => {
            let off = ((p.x - c.x).hypot(p.y - c.y) - r).abs();
            let two = 2.0 * PI;
            let along = ((((p.y - c.y).atan2(p.x - c.x) - start) % two) + two) % two;
            (along <= sweep).then_some(off)
        }
        Shape::Point { .. } => None,
    }
}

/// The way a piece runs where a point's foot lies, a unit vector.
fn way_at(e: &Shape, p: DVec2) -> DVec2 {
    match *e {
        Shape::Line { a, b } => {
            let l = (b.x - a.x).hypot(b.y - a.y);
            let l = if l == 0.0 || l.is_nan() { 1.0 } else { l };
            DVec2::new((b.x - a.x) / l, (b.y - a.y) / l)
        }
        Shape::Arc { c, .. } | Shape::Circle { c, .. } => {
            let a = (p.y - c.y).atan2(p.x - c.x);
            DVec2::new(-a.sin(), a.cos())
        }
        Shape::Point { .. } | Shape::Spline(_) => DVec2::X,
    }
}

/// How far off a piece the points of a stretch lie, mm.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OffMm {
    pub mean: f64,
    pub max: f64,
    pub rms: f64,
}

/// A stretch of the outline taken as a piece's wall: the piece, its points, and how far off it
/// they lie.
#[derive(Debug, Clone, PartialEq)]
pub struct SharedStretch {
    pub piece: EntityId,
    pub points: Vec<DVec2>,
    pub off_mm: OffMm,
}

/// What is left of the outline: a run to fit, open, and the pieces its start and its end
/// reach — `None` at an open chain's own end.
#[derive(Debug, Clone, PartialEq)]
pub struct OpenRun {
    pub points: Vec<DVec2>,
    pub from: Option<EntityId>,
    pub to: Option<EntityId>,
}

/// A chain of the outline against the pieces already there: what of it is theirs, what is left
/// to fit, and whether none of it was theirs (whole: the chain is fitted as it is).
#[derive(Debug, Clone, PartialEq)]
pub struct ChainShare {
    pub shared: Vec<SharedStretch>,
    pub runs: Vec<OpenRun>,
    pub whole: bool,
    /// Points a few hundredths off a corner the sketch already has — two of its pieces meeting
    /// — or off the piece either side: that corner.
    pub corners: usize,
}

/// A run of one label along the chain: the piece (by its place among the pieces), or `None`,
/// and the points' indices.
struct Stretch {
    label: Option<usize>,
    idx: Vec<usize>,
}

/// The stretches of one label, in order — round a loop from a point where the label changes,
/// so none is cut in two.
fn stretches(label: &[Option<usize>], lp: bool) -> Vec<Stretch> {
    let n = label.len();
    let mut start = 0;
    if lp {
        match (0..n).find(|&i| label[i] != label[(i + n - 1) % n]) {
            Some(change) => start = change,
            None => return vec![Stretch { label: label.first().copied().flatten(), idx: (0..n).collect() }],
        }
    }
    let mut out: Vec<Stretch> = Vec::new();
    for k in 0..n {
        let i = (start + k) % n;
        match out.last_mut() {
            Some(last) if last.label == label[i] => last.idx.push(i),
            _ => out.push(Stretch { label: label[i], idx: vec![i] }),
        }
    }
    out
}

/// A chain of the outline shared out among the pieces already there. Each point goes to the
/// nearest piece whose foot it has within `share` mm; then each stretch of one piece is kept
/// only where it runs the piece's way (within `deg`, `SHARE_DEG` unless given) for
/// `MIN_STRETCH_MM` or more — a stretch across a piece is no wall of it. A run left between two
/// kept stretches whose points all lie within 3·share of the pieces either side is the corner
/// between them, drawn already (a scan's round of a few tenths at a sharp corner of the
/// sketch); so is one too short to fit. The rest are the runs to fit.
pub fn share_chain(chain: &[DVec2], pieces: &[&Entity], share: f64, deg: Option<f64>) -> ChainShare {
    let lp = chain_closed(chain);
    let pts: Vec<DVec2> = if lp { chain[..chain.len() - 1].to_vec() } else { chain.to_vec() };
    let n = pts.len();
    let none = || ChainShare { shared: Vec::new(), runs: Vec::new(), whole: true, corners: 0 };
    if n == 0 || pieces.is_empty() || !(share > 0.0) {
        return none();
    }
    let cos = ((deg.unwrap_or(SHARE_DEG) * PI) / 180.0).cos();
    // Each point's nearest piece within share, and how far off it.
    let mut near: Vec<Option<usize>> = vec![None; n];
    let mut near_off = vec![0.0; n];
    for i in 0..n {
        let mut best = None;
        let mut best_off = share;
        for (k, e) in pieces.iter().enumerate() {
            if let Some(off) = foot_on(&e.shape, pts[i]) {
                if off <= best_off {
                    best_off = off;
                    best = Some(k);
                }
            }
        }
        near[i] = best;
        near_off[i] = best_off;
    }
    // Then walked along the chain: a point stays with the piece the point before it went to
    // while it lies within 2·share of it, unless another lies nearer — so a wall a tenth off,
    // its points either side of share by the scan's noise, is one stretch, and a run leaves it
    // only where the outline clearly goes its own way. Round a loop twice, from a point that
    // knows its piece, so the first points have the walk behind them.
    let mut label: Vec<Option<usize>> = vec![None; n];
    let mut off = vec![0.0; n];
    let first = if lp { near.iter().position(|x| x.is_some()).unwrap_or(0) } else { 0 };
    let mut cur: Option<usize> = None;
    for k in 0..(if lp { 2 * n } else { n }) {
        let i = (first + k) % n;
        let stay = cur.and_then(|c| foot_on(&pieces[c].shape, pts[i]));
        match stay {
            Some(s) if s <= 2.0 * share && !(near[i].is_some() && near[i] != cur && near_off[i] < s) => {
                label[i] = cur;
                off[i] = s;
            }
            _ => {
                cur = near[i];
                label[i] = near[i];
                off[i] = near_off[i];
            }
        }
    }
    // A stretch kept only where it runs the piece's way, long enough to say.
    for s in stretches(&label, lp) {
        let Some(k) = s.label else { continue };
        let e = &pieces[k].shape;
        let (Some(&i0), Some(&i1)) = (s.idx.first(), s.idx.last()) else { continue };
        let (first, last) = (pts[i0], pts[i1]);
        let chord = last - first;
        let len = chord.x.hypot(chord.y);
        let mid = pts[s.idx[s.idx.len() >> 1]];
        let way = way_at(e, mid);
        let along = if len > 0.0 { (chord.x * way.x + chord.y * way.y).abs() / len } else { 0.0 };
        if s.idx.len() < 2 || len < MIN_STRETCH_MM.max(2.0 * share) || along < cos {
            for &i in &s.idx {
                label[i] = None;
            }
        }
    }
    let all = stretches(&label, lp);
    if all.iter().all(|s| s.label.is_none()) {
        return none();
    }
    let reach = 3.0 * share;
    let mut corners = 0;
    let mut runs: Vec<OpenRun> = Vec::new();
    for (k, s) in all.iter().enumerate() {
        if s.label.is_some() {
            continue;
        }
        let before = if k > 0 {
            all[k - 1].label
        } else if lp {
            all.last().and_then(|x| x.label)
        } else {
            None
        };
        let after = if k + 1 < all.len() {
            all[k + 1].label
        } else if lp {
            all.first().and_then(|x| x.label)
        } else {
            None
        };
        let sides: Vec<&Entity> = [before, after].iter().flatten().map(|&x| pieces[x]).collect();
        if sides.len() == 2 {
            let near = s.idx.iter().all(|&i| sides.iter().any(|e| nearest_on(&e.shape, pts[i]).1 <= reach));
            if near || s.idx.len() < MIN_RUN_POINTS {
                corners += s.idx.len();
                continue;
            }
        }
        runs.push(OpenRun { points: s.idx.iter().map(|&i| pts[i]).collect(), from: before.map(|x| pieces[x].id), to: after.map(|x| pieces[x].id) });
    }
    // The stretches' points by piece, in the order the pieces were first met.
    let mut by_piece: Vec<(EntityId, Vec<DVec2>, Vec<f64>)> = Vec::new();
    let mut slot: HashMap<EntityId, usize> = HashMap::new();
    for s in &all {
        let Some(k) = s.label else { continue };
        let id = pieces[k].id;
        let at = *slot.entry(id).or_insert_with(|| {
            by_piece.push((id, Vec::new(), Vec::new()));
            by_piece.len() - 1
        });
        for &i in &s.idx {
            by_piece[at].1.push(pts[i]);
            by_piece[at].2.push(off[i]);
        }
    }
    let shared = by_piece
        .into_iter()
        .map(|(piece, points, offs)| {
            let count = offs.len() as f64;
            let mean = offs.iter().sum::<f64>() / count;
            let max = offs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let rms = (offs.iter().map(|b| b * b).sum::<f64>() / count).sqrt();
            SharedStretch { piece, points, off_mm: OffMm { mean, max, rms } }
        })
        .collect();
    ChainShare { shared, runs, whole: false, corners }
}

/// A point's coordinates as a key, a negative zero read as zero.
fn key(p: DVec2) -> (u64, u64) {
    ((p.x + 0.0).to_bits(), (p.y + 0.0).to_bits())
}

/// The shared stretches' points put behind the fitted pieces they lie on, beside the points
/// they had: the solve then fits each such wall on every level. Its scatter — the weight its
/// points carry against the relations — is raised to how far the new points lay off it: a
/// level a few hundredths off is no noise a clean slice's few microns can hold the piece to,
/// and weighed as such it would pull against every relation on it. A drawn, fixed or projected
/// piece takes none, and a point the piece has already (the same chain fitted again) is not
/// taken twice. Returns the pieces that took points.
pub fn take_shared(sketch: &mut Sketch, shared: &[SharedStretch]) -> Vec<EntityId> {
    let mut took = Vec::new();
    for s in shared {
        let Some(e) = sketch.entity_mut(s.piece) else { continue };
        if e.fixed || e.projected || e.points.is_empty() {
            continue;
        }
        let had: HashSet<(u64, u64)> = e.points.iter().map(|p| key(*p)).collect();
        let fresh: Vec<DVec2> = s.points.iter().copied().filter(|p| !had.contains(&key(*p))).collect();
        if fresh.is_empty() {
            continue;
        }
        e.points.extend(fresh);
        e.scatter = e.scatter.max(s.off_mm.rms);
        took.push(e.id);
    }
    took
}

/// How a run's end was landed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Held {
    /// On the end of the piece it reached (coincident).
    End,
    /// Held on the piece within it (a bounded point-on).
    On,
    /// Held at first, and let go — the sketch did not solve with it held (`let_go`).
    Free,
}

/// Where a new run's end landed: on the end of the piece it reached (coincident), or held on
/// the piece within it (a bounded point-on) — and, where the run carries a line on in line past
/// that line's end, collinear with it.
#[derive(Debug, Clone, PartialEq)]
pub struct Landing {
    pub end: Anchor,
    pub on: EntityId,
    pub held: Held,
    pub at: DVec2,
    pub collinear: bool,
    /// The relations that hold it, by id.
    pub constraints: Vec<ConstraintId>,
}

/// A new run's end landed on the piece it reached: within `LAND_SNAP_MM` of one of that
/// piece's ends, or with its foot past one by up to `LAND_REACH_MM`, it is made coincident with
/// that end — a line's end moved there first; with its foot on the piece, it is held on it by a
/// point-on that keeps it within the piece (bounded). A line run on past a line's end in line
/// with it — its far end within `share` of that line — is held collinear with it too: one wall,
/// drawn on. `None` where the end lies too far past the piece to land.
pub fn land_end(sketch: &mut Sketch, end: Anchor, on: EntityId, share: f64) -> Option<Landing> {
    let e = sketch.entity(end.entity)?.shape.clone();
    let t = sketch.entity(on)?;
    let (t_id, t_shape) = (t.id, t.shape.clone());
    if !is_carrier(&t_shape) {
        return None;
    }
    let p = anchor_point(&e, end.at);
    // The nearest of the piece's ends, the first of two as near.
    let mut nearest: Option<(AnchorAt, DVec2, f64)> = None;
    for &at in ends_of(&t_shape) {
        let q = anchor_point(&t_shape, at);
        let d = (q.x - p.x).hypot(q.y - p.y);
        if nearest.is_none_or(|(_, _, best)| d < best) {
            nearest = Some((at, q, d));
        }
    }
    let on_piece = foot_on(&t_shape, p).is_some();
    if let Some((at, q, d)) = nearest {
        if d <= LAND_SNAP_MM || (!on_piece && d <= LAND_REACH_MM) {
            if let Some(Entity { shape: Shape::Line { a, b }, .. }) = sketch.entity_mut(end.entity) {
                match end.at {
                    AnchorAt::A => *a = q,
                    AnchorAt::B => *b = q,
                    AnchorAt::Center | AnchorAt::P(_) => {}
                }
            }
            let mut landing = Landing { end, on, held: Held::End, at: q, collinear: false, constraints: vec![sketch.constrain(Rule::Coincident { p: end, q: Anchor::new(t_id, at) })] };
            let moved = sketch.entity(end.entity).map(|x| x.shape.clone());
            if let (Some(Shape::Line { a: la, b: lb }), Shape::Line { a: ma, b: mb }) = (moved, t_shape) {
                if in_line((la, lb), (ma, mb), share) {
                    landing.constraints.push(sketch.constrain(Rule::Collinear { line: end.entity, on: OnTarget::Line(t_id) }));
                    landing.collinear = true;
                }
            }
            return Some(landing);
        }
    }
    // An end that comes along the piece rather than to it — the run's way within
    // LAND_GRAZE_DEG of the piece's there — is no junction: where it meets the piece is as good
    // as anywhere along it.
    if !on_piece || grazes(&e, end.at, &t_shape, p) {
        return None;
    }
    let target = if matches!(t_shape, Shape::Line { .. }) { OnTarget::Line(t_id) } else { OnTarget::Curve(t_id) };
    let cid = sketch.constrain(Rule::PointOn { p: end, on: target, bounded: true });
    Some(Landing { end, on, held: Held::On, at: p, collinear: false, constraints: vec![cid] })
}

/// Whether a piece's end arrives along `t` at `p` rather than across it.
fn grazes(e: &Shape, at: AnchorAt, t: &Shape, p: DVec2) -> bool {
    let way = match *e {
        Shape::Line { a, b } => b - a,
        Shape::Arc { start, sweep, .. } => {
            let a = if at == AnchorAt::A { start } else { start + sweep };
            DVec2::new(-a.sin(), a.cos())
        }
        _ => return false,
    };
    let len = way.x.hypot(way.y);
    if !(len > 0.0) {
        return false;
    }
    let there = way_at(t, p);
    (way.x * there.x + way.y * there.y).abs() / len >= ((LAND_GRAZE_DEG * PI) / 180.0).cos()
}

/// A fit's landings let go where the sketch does not solve with them held: while the solve
/// leaves relations unmet and some landing's relations are among them, the landing whose
/// relations miss the most is taken out (held: free) and the sketch solved again — at most
/// `rounds` times (ScanRuler's default is 8). A near miss the fit could not tell from a
/// junction then leaves an end open, which the answer shows, rather than the whole fit refused.
/// `solve` solves the sketch and says what was unmet, by relation id.
pub fn let_go(sketch: &mut Sketch, landings: &mut [Landing], mut solve: impl FnMut(&mut Sketch) -> Option<HashMap<ConstraintId, f64>>, rounds: usize) {
    for _ in 0..rounds {
        let Some(unmet) = solve(sketch) else { return };
        if unmet.is_empty() {
            return;
        }
        let mut worst: Option<(usize, f64)> = None;
        for (i, l) in landings.iter().enumerate() {
            if l.held == Held::Free || !l.constraints.iter().any(|id| unmet.contains_key(id)) {
                continue;
            }
            let miss = l.constraints.iter().map(|id| unmet.get(id).copied().unwrap_or(0.0)).fold(f64::NEG_INFINITY, f64::max);
            if worst.is_none_or(|(_, m)| miss > m) {
                worst = Some((i, miss));
            }
        }
        let Some((i, _)) = worst else { return };
        let gone = landings[i].constraints.clone();
        sketch.constraints.retain(|c| !gone.contains(&c.id));
        landings[i].held = Held::Free;
    }
}

/// Whether a line lies along another's line: running its way within `SHARE_DEG`, both its ends
/// within `share` of it.
fn in_line(l: (DVec2, DVec2), m: (DVec2, DVec2), share: f64) -> bool {
    let (dx, dy) = (m.1.x - m.0.x, m.1.y - m.0.y);
    let ll = dx.hypot(dy);
    if !(ll > 0.0) {
        return false;
    }
    let off_of = |p: DVec2| ((p.y - m.0.y) * dx - (p.x - m.0.x) * dy).abs() / ll;
    let (lx, ly) = (l.1.x - l.0.x, l.1.y - l.0.y);
    let len_l = lx.hypot(ly);
    if !(len_l > 0.0) {
        return false;
    }
    let along = (lx * dx + ly * dy).abs() / (len_l * ll);
    along >= ((SHARE_DEG * PI) / 180.0).cos() && off_of(l.0) <= share && off_of(l.1) <= share
}

/// A point-on's anchor where it lies past the piece it is held on — on the line or circle the
/// piece runs on, beyond its end — and how far, mm; `None` on the piece, or held on anything
/// but a line or an arc.
pub fn past_piece(sketch: &Sketch, p: Anchor, on: OnTarget) -> Option<f64> {
    let id = match on {
        OnTarget::Line(l) | OnTarget::Curve(l) => l,
        OnTarget::Datum(_) => return None,
    };
    let t = sketch.entity(id)?.shape.clone();
    let e = sketch.entity(p.entity)?.shape.clone();
    if !matches!(t, Shape::Line { .. } | Shape::Arc { .. }) {
        return None;
    }
    let pp = anchor_point(&e, p.at);
    if foot_on(&t, pp).is_some() {
        return None;
    }
    // Past an end: how far along the line or round the circle from it.
    match t {
        Shape::Line { a, b } => {
            let (dx, dy) = (b.x - a.x, b.y - a.y);
            let ll = dx.hypot(dy);
            if !(ll > 0.0) {
                return None;
            }
            let s = ((pp.x - a.x) * dx + (pp.y - a.y) * dy) / ll;
            Some(if s < 0.0 { -s } else { s - ll })
        }
        Shape::Arc { c, r, start, sweep } => {
            let two = 2.0 * PI;
            let along = ((((pp.y - c.y).atan2(pp.x - c.x) - start) % two) + two) % two;
            Some((along - sweep).min(two - along) * r)
        }
        _ => None,
    }
}

/// One run of a chain a shared fit fitted: its pieces in order, or why it was not; how many
/// points it had; and the pieces its start and its end reached.
#[derive(Debug, Clone, PartialEq)]
pub struct RunReport {
    pub order: Option<Vec<PlacedSegment>>,
    pub points: usize,
    pub skipped: Option<String>,
    pub from: Option<EntityId>,
    pub to: Option<EntityId>,
}

/// A stretch a shared fit gave a piece, summed up.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SharedSummary {
    pub piece: EntityId,
    pub points: usize,
    pub off_mean: f64,
    pub off_max: f64,
}

/// Where a run's end landed, and which chain and run it was.
#[derive(Debug, Clone, PartialEq)]
pub struct Junction {
    pub landing: Landing,
    pub chain: usize,
    pub run: usize,
}

/// What a fit that shared its chains with the sketch's pieces did: by the index of each chain
/// it shared, the runs fitted — each with its pieces in order, or why it was not — and the
/// pieces the run's start and end reached; the stretches shared, piece by piece; where the
/// runs' ends landed; and how many points were corners already drawn.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ShareReport {
    pub chains: BTreeMap<usize, Vec<RunReport>>,
    pub shared: Vec<SharedSummary>,
    pub junctions: Vec<Junction>,
    pub corners: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::{solve_sketch, SolveOptions};

    /// A seeded noise in [-1, 1].
    fn noise_of(seed: i64) -> impl FnMut() -> f64 {
        let mut s = seed;
        move || {
            s = (s * 16807) % 2147483647;
            (s as f64 / 2147483647.0 - 0.5) * 2.0
        }
    }

    /// A closed outline through the corners, a point every `step` mm pushed off it by up to
    /// `noise` square to the side — closed back on its first point, as a section's loop is.
    fn outline(corners: &[[f64; 2]], step: f64, noise: f64, seed: i64) -> Vec<DVec2> {
        let mut rnd = noise_of(seed);
        let mut out: Vec<DVec2> = Vec::new();
        for (i, a) in corners.iter().enumerate() {
            let b = corners[(i + 1) % corners.len()];
            let len = (b[0] - a[0]).hypot(b[1] - a[1]);
            let n = [-(b[1] - a[1]) / len, (b[0] - a[0]) / len];
            let k = ((len / step) + 0.5).floor() as usize;
            for j in 0..k {
                let d = noise * rnd();
                out.push(DVec2::new(a[0] + ((b[0] - a[0]) * j as f64) / k as f64 + n[0] * d, a[1] + ((b[1] - a[1]) * j as f64) / k as f64 + n[1] * d));
            }
        }
        out.push(out[0]);
        out
    }

    /// A rectangle of four lines fitted to their sides' points, joined at the corners.
    fn fitted_rectangle(lo: [f64; 2], hi: [f64; 2]) -> (Sketch, Vec<EntityId>) {
        let mut sketch = Sketch::default();
        let corners = [lo, [hi[0], lo[1]], hi, [lo[0], hi[1]]];
        let sides: Vec<EntityId> = (0..4)
            .map(|i| {
                let (a, b) = (corners[i], corners[(i + 1) % 4]);
                let points: Vec<DVec2> = (0..21).map(|k| DVec2::new(a[0] + ((b[0] - a[0]) * k as f64) / 20.0, a[1] + ((b[1] - a[1]) * k as f64) / 20.0)).collect();
                sketch.add(Entity::new(Shape::Line { a: DVec2::from(a), b: DVec2::from(b) }).fitted(points, 0.01))
            })
            .collect();
        for i in 0..4 {
            sketch.constrain(Rule::Coincident { p: Anchor::new(sides[i], AnchorAt::B), q: Anchor::new(sides[(i + 1) % 4], AnchorAt::A) });
        }
        (sketch, sides)
    }

    fn sorted(mut v: Vec<EntityId>) -> Vec<EntityId> {
        v.sort_unstable();
        v
    }

    #[test]
    fn takes_the_walls_the_deck_has_and_leaves_the_wings_as_one_run_between_the_decks_bottom_and_top() {
        let (mut sketch, sides) = fitted_rectangle([0.0, 0.0], [20.0, 10.0]);
        let (bottom, top, left) = (sides[0], sides[2], sides[3]);
        // The level below: the deck's outline and a wing out to x 30, its walls 0.03 mm off
        // the deck's.
        let chain = outline(&[[0.03, 0.03], [30.0, 0.03], [30.0, 10.03], [0.03, 10.03]], 0.2, 0.01, 5);
        let split = share_chain(&chain, &share_pieces(&sketch), 0.1, None);
        assert!(!split.whole);
        assert_eq!(sorted(split.shared.iter().map(|s| s.piece).collect()), sorted(vec![bottom, top, left]));
        for s in &split.shared {
            assert!(s.off_mm.mean > 0.015);
            assert!(s.off_mm.max < 0.05);
        }
        assert_eq!(split.runs.len(), 1);
        let run = &split.runs[0];
        assert_eq!(run.from, Some(bottom));
        assert_eq!(run.to, Some(top));
        assert!(run.points.iter().map(|p| p.x).fold(f64::INFINITY, f64::min) > 19.9);
        // The fitted pieces take the points beside their own; once only.
        assert_eq!(sorted(take_shared(&mut sketch, &split.shared)), sorted(vec![bottom, top, left]));
        assert_eq!(take_shared(&mut sketch, &split.shared), Vec::<EntityId>::new());
        assert!(sketch.entity(bottom).map_or(0, |e| e.points.len()) > 21 + 90);
    }

    #[test]
    fn leaves_a_scans_round_at_a_corner_the_sketch_has_sharp_and_a_wall_0_3_mm_off_is_not_the_same_wall() {
        let (sketch, _) = fitted_rectangle([0.0, 0.0], [20.0, 10.0]);
        // The same outline, its corners rounded R1 — 0.29 mm in off the sharp corner at most:
        // no run is left.
        let mut round: Vec<DVec2> = Vec::new();
        let r = 1.0;
        let corners: [([f64; 2], f64); 4] = [([20.0 - r, r], -PI / 2.0), ([20.0 - r, 10.0 - r], 0.0), ([r, 10.0 - r], PI / 2.0), ([r, r], PI)];
        for (i, (c, a0)) in corners.iter().enumerate() {
            for k in 0..=8 {
                let t = a0 + (k as f64 / 8.0) * (PI / 2.0);
                round.push(DVec2::new(c[0] + r * t.cos(), c[1] + r * t.sin()));
            }
            let (n, n0) = corners[(i + 1) % 4];
            let from = round[round.len() - 1];
            let to = DVec2::new(n[0] + r * n0.cos(), n[1] + r * n0.sin());
            for k in 1..50 {
                round.push(DVec2::new(from.x + ((to.x - from.x) * k as f64) / 50.0, from.y + ((to.y - from.y) * k as f64) / 50.0));
            }
        }
        round.push(round[0]);
        let split = share_chain(&round, &share_pieces(&sketch), 0.1, None);
        assert_eq!(split.runs, Vec::new());
        assert!(split.corners > 0);
        // A wall standing 0.3 mm off: past share, fitted again.
        let off = outline(&[[0.0, -0.3], [20.0, -0.3], [20.0, 10.0], [0.0, 10.0]], 0.2, 0.005, 5);
        let apart = share_chain(&off, &share_pieces(&sketch), 0.1, None);
        assert!(!apart.runs.is_empty());
        assert!(!apart.shared.iter().any(|s| s.piece == 1));
    }

    #[test]
    fn lands_a_runs_end_on_the_pieces_end_in_line_where_it_carries_the_line_on_or_within_the_piece() {
        let (mut sketch, sides) = fitted_rectangle([0.0, 0.0], [20.0, 10.0]);
        let bottom = sides[0];
        // Carried on: a line from x 20.2 on along y 0 lands on the corner and holds collinear.
        let on = sketch.add_shape(Shape::Line { a: DVec2::new(20.2, 0.01), b: DVec2::new(30.0, 0.01) });
        let landing = land_end(&mut sketch, Anchor::new(on, AnchorAt::A), bottom, 0.1).expect("lands");
        assert_eq!((landing.held, landing.on, landing.at, landing.collinear), (Held::End, bottom, DVec2::new(20.0, 0.0), true));
        assert!(matches!(sketch.entity(on).map(|e| e.shape.clone()), Some(Shape::Line { a, .. }) if a == DVec2::new(20.0, 0.0)));
        // A T: a wall up from the bottom at x 12 is held on it, within it.
        let wall = sketch.add_shape(Shape::Line { a: DVec2::new(12.0, 0.08), b: DVec2::new(12.0, 10.0) });
        let landing = land_end(&mut sketch, Anchor::new(wall, AnchorAt::A), bottom, 0.1).expect("lands");
        assert_eq!((landing.held, landing.on), (Held::On, bottom));
        assert_eq!(sketch.constraints.last().map(|c| c.rule.clone()), Some(Rule::PointOn { p: Anchor::new(wall, AnchorAt::A), on: OnTarget::Line(bottom), bounded: true }));
    }

    /// A carrier from x 0 to 10 and a wall standing on it at x 8; the carrier then dimensioned
    /// 5 long.
    fn tee(bounded: bool) -> (crate::solver::SolveReport, (DVec2, DVec2), DVec2, Option<f64>) {
        let mut sketch = Sketch::default();
        let carrier = sketch.add_shape(Shape::Line { a: DVec2::new(0.0, 0.0), b: DVec2::new(10.0, 0.0) });
        let wall = sketch.add_shape(Shape::Line { a: DVec2::new(8.0, 0.0), b: DVec2::new(8.0, 5.0) });
        sketch.constrain(Rule::Horizontal { entity: carrier });
        sketch.constrain(Rule::PointOn { p: Anchor::new(wall, AnchorAt::A), on: OnTarget::Line(carrier), bounded });
        sketch.constrain(Rule::Length { entity: carrier, value: 5.0 });
        let report = solve_sketch(&mut sketch, SolveOptions::default());
        let c = match sketch.entity(carrier).map(|e| e.shape.clone()) {
            Some(Shape::Line { a, b }) => (a, b),
            _ => (DVec2::NAN, DVec2::NAN),
        };
        let end = sketch.anchor(Anchor::new(wall, AnchorAt::A)).unwrap_or(DVec2::NAN);
        let past = past_piece(&sketch, Anchor::new(wall, AnchorAt::A), OnTarget::Line(carrier));
        (report, c, end, past)
    }

    #[test]
    fn keeps_the_walls_foot_on_the_shortened_carrier_the_solve_met() {
        let (report, c, end, past) = tee(true);
        assert!(report.ok);
        assert!(((c.1 - c.0).length() - 5.0).abs() < 5e-7);
        assert!(end.x <= c.0.x.max(c.1.x) + 1e-6);
        assert!(end.x >= c.0.x.min(c.1.x) - 1e-6);
        assert_eq!(past, None);
    }

    #[test]
    fn leaves_a_plain_point_on_on_the_line_past_the_carriers_end_as_before_and_says_how_far() {
        let (report, c, end, past) = tee(false);
        assert!(report.ok);
        assert!(end.x > c.0.x.max(c.1.x) + 0.4);
        assert!(past.is_some_and(|p| p > 0.4));
    }

    #[test]
    fn counts_the_bound_in_neither_the_equations_nor_the_redundant_ones() {
        let plain = tee(false).0;
        let held = tee(true).0;
        assert_eq!(held.equations, plain.equations);
        assert_eq!(held.redundant, plain.redundant);
    }

    /// Not a ScanRuler test: a landing whose relation the solve cannot meet is let go.
    #[test]
    fn lets_go_the_landing_whose_relation_misses_most() {
        let mut sketch = Sketch::default();
        let a = sketch.add_shape(Shape::Line { a: DVec2::ZERO, b: DVec2::new(10.0, 0.0) });
        let b = sketch.add_shape(Shape::Line { a: DVec2::new(10.0, 0.0), b: DVec2::new(10.0, 5.0) });
        let held = sketch.constrain(Rule::Coincident { p: Anchor::new(a, AnchorAt::B), q: Anchor::new(b, AnchorAt::A) });
        let mut landings = vec![Landing { end: Anchor::new(b, AnchorAt::A), on: a, held: Held::End, at: DVec2::new(10.0, 0.0), collinear: false, constraints: vec![held] }];
        let mut calls = 0;
        let_go(
            &mut sketch,
            &mut landings,
            |s| {
                calls += 1;
                let unmet: HashMap<ConstraintId, f64> = s.constraints.iter().map(|c| (c.id, 1.0)).collect();
                Some(unmet)
            },
            8,
        );
        assert_eq!(landings[0].held, Held::Free);
        assert!(sketch.constraints.is_empty());
        assert_eq!(calls, 2);
    }
}
