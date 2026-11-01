// SPDX-License-Identifier: AGPL-3.0-only
//! A slice chain cut into lines and arcs — the mesh sketch's first move. Ported from
//! ScanRuler (`plugins/reverse/core/segment.ts`), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only.
//!
//! A section through a part comes back as one polyline per outline, a point every fraction
//! of a millimetre, noisy as the scan is; the sketch wants a dozen pieces with the slice
//! points kept behind each one as the soft data the solver pulls it towards.
//!
//! The cut is made in passes. Ramer–Douglas–Peucker at the noise tolerance marks the corners
//! and lays chords along the curves; then neighbouring runs are merged, best first, as long
//! as one line — or, failing that, one arc — still explains the joined points. The chords of
//! a fillet merge into its arc, the pieces of a straight side into its line, and a corner
//! stays a corner because no one piece fits across it. The boundaries between a line and an
//! arc are slid to where the residuals say; what is left under a short length is asked about
//! (the dissolve pass); and a last look at every piece asks whether the other kind explains
//! it better.
//!
//! What "explains" means is three things: the σ of the run; the worst stretch — the mean
//! miss over a few points in a row — which a long run cannot average away; and the heading:
//! the chain has to come into a piece and leave it along the piece. An arc is held to the
//! noise tolerance and a line to a wave tolerance several times that (`detail_levels`),
//! because a scanned wall bows and wanders and a round does not.

use std::collections::HashMap;
use std::f64::consts::PI;

use glam::DVec2;

use crate::chain::view_chain;
use crate::fit2d::{fit_arc_points, fit_circle_points, fit_line_points, ArcFit, FitError, LineFit};
use crate::simplify::simplify_run;

/// What a piece of a chain is.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Piece {
    Line(LineFit),
    Arc(ArcFit),
}

impl Piece {
    pub fn is_line(&self) -> bool {
        matches!(self, Piece::Line(_))
    }

    pub fn is_arc(&self) -> bool {
        matches!(self, Piece::Arc(_))
    }

    pub fn sigma(&self) -> f64 {
        match self {
            Piece::Line(l) => l.sigma,
            Piece::Arc(a) => a.sigma,
        }
    }

    pub fn kind(&self) -> PieceKind {
        match self {
            Piece::Line(_) => PieceKind::Line,
            Piece::Arc(_) => PieceKind::Arc,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PieceKind {
    Line,
    Arc,
}

/// One piece of the chain, over the run of chain points `[lo, hi]` (indices may run past the
/// end of a closed chain and wrap).
#[derive(Debug, Clone, PartialEq)]
pub struct ChainSegment {
    pub piece: Piece,
    pub lo: isize,
    pub hi: isize,
    pub points: Vec<DVec2>,
    /// A piece was dropped between this one and the next — a blob in the corner — and the
    /// two make the corner between them.
    pub corner_after: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SegmentedChain {
    pub segments: Vec<ChainSegment>,
    /// The chain as segmented: a loop's closing duplicate dropped.
    pub points: Vec<DVec2>,
    pub closed: bool,
    /// The chain's scatter and point spacing, millimetres.
    pub scatter: f64,
    pub spacing: f64,
    /// The tolerance the cut was made at.
    pub tolerance: f64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SegmentOptions {
    /// Millimetres a piece may miss its points by; the chain's scatter says otherwise. A
    /// typed tolerance is taken as it stands — `detail` scales only the one read off the chain.
    pub tolerance: Option<f64>,
    /// How fine the cut is, 0 (coarse: few, long pieces) to 1 (fine: every small feature its
    /// own piece).
    pub detail: Option<f64>,
    /// Pieces of fewer points are folded into a neighbour.
    pub min_points: Option<usize>,
    /// A piece shorter than this, millimetres, that only cuts a corner its neighbours make by
    /// themselves is dropped; a multiple of the tolerance otherwise.
    pub min_length: Option<f64>,
}

/// How many scatters the default tolerance is.
const TOLERANCE_SCATTERS: f64 = 3.0;
/// The tolerance floor as a share of the point spacing.
const TOLERANCE_SPACING: f64 = 0.6;
const fn deg(d: f64) -> f64 {
    d * PI / 180.0
}
/// An arc has to cover this much to be an arc rather than a bowed line.
const MIN_SWEEP: f64 = deg(6.0);
/// A line is made an arc when a circle explains it this much better, at a radius under this
/// many line lengths.
const LINE_ARC_F_CRIT: f64 = 16.0;
const LINE_ARC_RADIUS_LENGTHS: f64 = 20.0;
/// The F test makes an arc of a line only when the arc turns at least this far…
const LINE_ARC_MIN_SWEEP: f64 = deg(20.0);
/// …unless it bows by this many noise tolerances: a millimetre's bow under 0.04 mm of noise
/// is a curve the part was drawn with.
const CURVE_BULGES: f64 = 5.0;
/// How far a straight side may wander, in tolerances: at the middle of the detail setting,
/// and at its coarse end — once at the fine end.
const WAVE_MIDDLE: f64 = 3.0;
const WAVE_COARSE: f64 = 5.0;
/// A line has its whole waviness from this many wave tolerances of length up, and the noise
/// tolerance alone at none.
const WAVE_LENGTHS: f64 = 20.0;
/// How far the chain's heading at either end of a run may be off the tangent of the arc
/// fitted to it.
const ARC_END_HEADING: f64 = deg(20.0);
/// A turn under this is no turn, when an arc's two halves are asked whether they turn the
/// same way.
const ARC_TURN_NOISE: f64 = deg(8.0);
const ARC_TURN_ASKED: f64 = deg(170.0);
const ARC_TURN_SLACK: f64 = deg(35.0);
/// The same for a line.
const LINE_END_HEADING: f64 = deg(15.0);
/// A piece is judged over this many points in a row.
const DEPART_WINDOW: usize = 5;
/// An arc may leave its points by this many tolerances over such a stretch.
const ARC_DEPART: f64 = 2.5;
/// A piece under this many wave tolerances long is looked at as a possible corner-cutter.
const SHORT_WAVES: f64 = 5.0;
/// An arc that turns less than this has to bulge `SHALLOW_BULGES` wave tolerances to be an
/// arc: under that it is a wall that bows.
const SHALLOW_SWEEP: f64 = deg(30.0);
const SHALLOW_BULGES: f64 = 3.0;
/// A piece no line and no arc explains is asked about up to this many short lengths.
const UNEXPLAINED_SHORTS: f64 = 3.0;
/// An arc that turns this far is a fillet however short it is — given the points to say so.
const SHORT_ARC_SWEEP: f64 = deg(50.0);
const SHORT_ARC_POINTS: isize = 8;
/// — and a radius of at least this many wave tolerances.
const SHORT_ARC_WAVES: f64 = 3.0;

/// Two lines under this angle have no corner between them.
fn meet_parallel() -> f64 {
    deg(8.0).sin()
}

/// What a piece is held to: an arc to the noise, a line to the waviness.
#[derive(Debug, Clone, Copy)]
struct Tolerances {
    line: f64,
    arc: f64,
}

const ANYTHING: Tolerances = Tolerances { line: f64::INFINITY, arc: f64::INFINITY };

/// The three levels a detail setting gives at a noise tolerance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetailLevels {
    /// How far a line may wander.
    pub wave: f64,
    /// How far an arc may miss.
    pub arc: f64,
    /// Pieces under this length are asked about.
    pub short: f64,
}

/// What a detail setting means at a noise tolerance. A scanned wall is not straight: sheet
/// metal bows, a print sags, a blob in a corner lifts the last millimetres of a side. So an
/// arc is held to the noise tolerance and a line to `wave`, the detail setting's multiple of
/// it; pieces under `short` are the ones the dissolve pass asks about.
pub fn detail_levels(tolerance: f64, detail: f64) -> DetailLevels {
    let d = if detail.is_finite() { detail } else { 0.5 }.clamp(0.0, 1.0);
    let factor = if d >= 0.5 { WAVE_MIDDLE.powf(2.0 * (1.0 - d)) } else { WAVE_MIDDLE + (WAVE_COARSE - WAVE_MIDDLE) * (1.0 - 2.0 * d) };
    let wave = tolerance * factor;
    DetailLevels { wave, arc: tolerance, short: SHORT_WAVES * wave }
}

/// The default tolerance of a chain of this scatter and spacing.
pub fn default_tolerance(scatter: f64, spacing: f64) -> f64 {
    (TOLERANCE_SCATTERS * scatter).max(TOLERANCE_SPACING * spacing)
}

fn line_of(points: &[DVec2]) -> Option<LineFit> {
    if points.len() >= 2 {
        fit_line_points(points).ok()
    } else {
        None
    }
}

fn arc_of(points: &[DVec2]) -> Option<ArcFit> {
    if points.len() < 5 {
        return None;
    }
    let arc = fit_arc_points(points).ok()?;
    if !(arc.sweep >= MIN_SWEEP) || !arc.radius.is_finite() {
        return None;
    }
    Some(arc)
}

/// How far a piece leaves its points: the largest mean signed miss over any `DEPART_WINDOW`
/// points in a row. A long side averages away anything short in its σ; a stretch of points
/// all off to one side is another feature, however long the rest of the run.
fn worst_miss(points: &[DVec2], piece: &Piece) -> f64 {
    let n = points.len();
    if n == 0 {
        return 0.0;
    }
    let w = DEPART_WINDOW.min(n);
    let miss: Vec<f64> = match piece {
        Piece::Line(l) => points.iter().map(|p| (p.x - l.center.x) * -l.dir.y + (p.y - l.center.y) * l.dir.x).collect(),
        Piece::Arc(a) => points.iter().map(|p| (*p - a.center).length() - a.radius).collect(),
    };
    let mut sum = 0.0;
    let mut worst = 0.0f64;
    for i in 0..n {
        sum += miss[i];
        if i >= w {
            sum -= miss[i - w];
        }
        if i + 1 >= w && sum.abs() > worst * w as f64 {
            worst = sum.abs() / w as f64;
        }
    }
    worst
}

fn turn(a: DVec2, b: DVec2) -> f64 {
    (a.x * b.y - a.y * b.x).atan2(a.x * b.x + a.y * b.y)
}

/// Whether the chain comes into an arc and leaves it along the arc. A circle fits things
/// that are no arc well enough to pass on its σ — the riser of a small step, a fillet with a
/// stretch of its side on board — and both give themselves away at their ends, where the
/// chain heads one way and the circle another.
fn runs_along(points: &[DVec2], arc: &ArcFit) -> bool {
    let n = points.len();
    let k = (n / 4).clamp(1, 4);
    let off = |a: DVec2, b: DVec2| -> f64 {
        let chord = b - a;
        let radial = (a + b) / 2.0 - arc.center;
        let (lc, lr) = (chord.length(), radial.length());
        if lc < 1e-12 || lr < 1e-12 {
            return 0.0;
        }
        // The tangent is square to the radius: the chord's share along the radius is the sine
        // of how far off it heads.
        (chord.dot(radial).abs() / (lc * lr)).min(1.0).asin()
    };
    if off(points[0], points[k]) > ARC_END_HEADING || off(points[n - 1 - k], points[n - 1]) > ARC_END_HEADING {
        return false;
    }
    // And it turns one way throughout: an S turns into its middle one way and out of it the
    // other. Turns under a few degrees are noise.
    let mid = n >> 1;
    let head = points[k] - points[0];
    let middle = points[(mid + k).min(n - 1)] - points[mid.saturating_sub(k)];
    let tail = points[n - 1] - points[n - 1 - k];
    let into = turn(head, middle);
    let out = turn(middle, tail);
    if into.abs() > ARC_TURN_NOISE && out.abs() > ARC_TURN_NOISE && into * out < 0.0 {
        return false;
    }
    // And by as much as the arc says. Past half a turn the chords cannot say.
    let claimed = arc.sweep * (1.0 - k as f64 / n as f64);
    if claimed > ARC_TURN_ASKED {
        return true;
    }
    (turn(head, tail).abs() - claimed).abs() <= ARC_TURN_SLACK
}

/// Whether the chain comes into a line and leaves it along the line: a wall that wanders
/// keeps its heading within a few degrees, and a side that has started round a corner has
/// not.
fn heads_along(points: &[DVec2], line: &LineFit) -> bool {
    let n = points.len();
    let k = (n / 4).clamp(1, 4);
    let off = |a: DVec2, b: DVec2| -> f64 {
        let chord = b - a;
        let lc = chord.length();
        if lc < 1e-12 {
            return 0.0;
        }
        ((chord.x * -line.dir.y + chord.y * line.dir.x).abs() / lc).min(1.0).asin()
    };
    off(points[0], points[k]) <= LINE_END_HEADING && off(points[n - 1 - k], points[n - 1]) <= LINE_END_HEADING
}

/// Whether the run goes along its line one way, end to end: its first and last points lie
/// at least half the line's length apart along it. A strip narrower than a wall may wander
/// — a section grazing a face, a lip cut lengthwise — runs out and back within a line's
/// tolerance, and taken as one line it would be placed from its first point to its last,
/// which stand side by side. (Not in ScanRuler, whose scans seldom cut so; a mesh from CAD,
/// with exact planes, does.)
fn runs_one_way(points: &[DVec2], line: &LineFit) -> bool {
    let n = points.len();
    let along = (points[n - 1] - points[0]).dot(line.dir).abs();
    along >= 0.5 * line.length
}

/// A piece and how well it sits: the larger of its σ and its worst stretch as a share of
/// what it is allowed — under one is a fit.
#[derive(Debug, Clone, Copy)]
struct Judged {
    piece: Piece,
    score: f64,
}

/// The better piece for a run: a line if one will do, an arc if not, `None` if neither. A
/// line's σ is held to half its waviness.
fn judged_piece_of(points: &[DVec2], tol: Tolerances) -> Option<Judged> {
    if let Some(line) = line_of(points) {
        let piece = Piece::Line(line);
        if tol.line == f64::INFINITY {
            return Some(Judged { piece, score: 0.0 });
        }
        let allowed = tol.arc + (tol.line - tol.arc) * (line.length / (WAVE_LENGTHS * tol.line)).min(1.0);
        if line.sigma <= allowed / 2.0 && heads_along(points, &line) && runs_one_way(points, &line) {
            let worst = worst_miss(points, &piece);
            if worst <= allowed {
                return Some(Judged { piece, score: ((2.0 * line.sigma) / allowed).max(worst / allowed) });
            }
        }
    }
    if let Some(arc) = arc_of(points) {
        let piece = Piece::Arc(arc);
        if tol.arc == f64::INFINITY {
            return Some(Judged { piece, score: 0.0 });
        }
        if arc.sigma <= tol.arc && runs_along(points, &arc) {
            let worst = worst_miss(points, &piece);
            if worst <= ARC_DEPART * tol.arc {
                return Some(Judged { piece, score: (arc.sigma / tol.arc).max(worst / (ARC_DEPART * tol.arc)) });
            }
        }
    }
    None
}

fn piece_of(points: &[DVec2], tol: Tolerances) -> Option<Piece> {
    judged_piece_of(points, tol).map(|j| j.piece)
}

/// The fit of a run as one kind, whatever its residual.
fn fit_as(points: &[DVec2], kind: PieceKind) -> Option<Piece> {
    match kind {
        PieceKind::Line => line_of(points).map(Piece::Line),
        PieceKind::Arc => arc_of(points).map(Piece::Arc),
    }
}

/// How far an arc bows off its chord.
fn bulge_of(arc: &ArcFit) -> f64 {
    arc.radius * (1.0 - (arc.sweep / 2.0).cos())
}

/// What a run is: a line if one explains it at the tolerance — unless a circle of a few line
/// lengths explains it significantly better (an F test), which gives away a short line on a
/// tight arc — else an arc, else whichever fits tighter.
fn classify(pts: &[DVec2], tol: Tolerances, scatter: f64) -> Option<Piece> {
    let mut piece = piece_of(pts, tol).or_else(|| piece_of(pts, ANYTHING))?;
    if let Piece::Line(line) = piece {
        if pts.len() >= 6 {
            if let Ok(circle) = fit_circle_points(pts) {
                let noise = circle.sigma.max(0.5 * scatter);
                let gain = pts.len() as f64 * (line.sigma.powi(2) - circle.sigma.powi(2));
                if gain > LINE_ARC_F_CRIT * noise * noise && circle.radius < LINE_ARC_RADIUS_LENGTHS * line.length {
                    if let Some(arc) = arc_of(pts) {
                        if (arc.sweep >= LINE_ARC_MIN_SWEEP || bulge_of(&arc) >= CURVE_BULGES * tol.arc) && runs_along(pts, &arc) {
                            piece = Piece::Arc(arc);
                        }
                    }
                }
            }
        }
    }
    // A shallow arc that bulges no more than a wall may wander is the wall.
    if let Piece::Arc(arc) = piece {
        if arc.sweep < SHALLOW_SWEEP && tol.line.is_finite() {
            let bulge = bulge_of(&arc);
            if bulge < (SHALLOW_BULGES * tol.line).min(CURVE_BULGES * tol.arc) {
                if let Some(line) = line_of(pts) {
                    piece = Piece::Line(line);
                }
            }
        }
    }
    Some(piece)
}

fn nearest_of(candidates: &[DVec2], near: DVec2) -> Option<DVec2> {
    candidates.iter().copied().min_by(|a, b| (*a - near).length().total_cmp(&(*b - near).length()))
}

/// Where two pieces' carriers cross nearest `near` — two lines at their intersection, a line
/// and a circle or two circles at the nearer of their two crossings, or where they come
/// closest when they just miss. `None` for two lines that run alongside each other.
pub fn meet_of(a: &Piece, b: &Piece, near: DVec2) -> Option<DVec2> {
    match (a, b) {
        (Piece::Line(la), Piece::Line(lb)) => {
            let (da, db) = (la.dir, lb.dir);
            let denom = da.x * db.y - da.y * db.x;
            if denom.abs() < meet_parallel() {
                return None;
            }
            let w = lb.center - la.center;
            let t = (w.x * db.y - w.y * db.x) / denom;
            Some(la.center + da * t)
        }
        (Piece::Arc(aa), Piece::Arc(ab)) => {
            let d = (ab.center - aa.center).length();
            if d < 1e-9 {
                return None;
            }
            let u = (ab.center - aa.center) / d;
            let (r1, r2) = (aa.radius, ab.radius);
            let along = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d);
            let h2 = r1 * r1 - along * along;
            let foot = aa.center + u * along;
            if h2 <= 0.0 {
                return Some(foot);
            }
            let h = h2.sqrt();
            let perp = DVec2::new(-u.y, u.x);
            nearest_of(&[foot + perp * h, foot - perp * h], near)
        }
        (Piece::Line(line), Piece::Arc(arc)) | (Piece::Arc(arc), Piece::Line(line)) => {
            let w = arc.center - line.center;
            let t = w.dot(line.dir);
            let foot = line.center + line.dir * t;
            let off = (arc.center - foot).length();
            if off >= arc.radius {
                return Some(foot);
            }
            let h = (arc.radius * arc.radius - off * off).sqrt();
            nearest_of(&[foot + line.dir * h, foot - line.dir * h], near)
        }
    }
}

/// A run of chain points `[lo, hi]`, with what the passes learnt about it.
#[derive(Debug, Clone, Copy, Default)]
struct Run {
    lo: isize,
    hi: isize,
    /// Looked at by the dissolve pass and found to be a feature of its own.
    kept: bool,
    /// A piece was dropped between this run and the next: the two make the corner between
    /// them, however far apart their ends stand.
    corner_after: bool,
    /// A cluster of short runs kept as the one feature it is.
    cluster: bool,
}

fn span(lo: isize, hi: isize) -> Run {
    Run { lo, hi, ..Run::default() }
}

fn ceil_div(a: isize, n: isize) -> isize {
    if a >= 0 {
        (a + n - 1) / n
    } else {
        -((-a) / n)
    }
}

/// The working state of one `segment_chain`.
struct Cutter {
    points: Vec<DVec2>,
    n: isize,
    closed: bool,
    tol: Tolerances,
    scatter: f64,
    min_points: isize,
    min_length: f64,
    runs: Vec<Run>,
    /// Points taken out of the fitting: a blob in a corner, a burr on a side.
    masked: Vec<bool>,
    fits: HashMap<(isize, isize), Option<Judged>>,
}

impl Cutter {
    fn modn(&self, i: isize) -> usize {
        i.rem_euclid(self.n) as usize
    }

    fn at(&self, i: isize) -> DVec2 {
        self.points[self.modn(i)]
    }

    fn points_of(&self, r: Run) -> Vec<DVec2> {
        (r.lo..=r.hi).filter(|&i| !self.masked[self.modn(i)]).map(|i| self.at(i)).collect()
    }

    /// `b` as it follows `a` round the chain: shifted by whole turns until it starts at or
    /// after `a`'s end.
    fn following(&self, a: Run, b: Run) -> Run {
        let k = ceil_div(a.hi - b.lo, self.n);
        span(b.lo + k * self.n, b.hi + k * self.n)
    }

    fn union_of(&self, a: Run, b: Run) -> Run {
        let f = self.following(a, b);
        let mut lo = a.lo;
        let mut hi = f.hi;
        // Never more than once round.
        if hi - lo > self.n {
            hi = lo + self.n;
        }
        while lo >= self.n {
            lo -= self.n;
            hi -= self.n;
        }
        span(lo, hi)
    }

    fn piece_over(&mut self, r: Run) -> Option<Judged> {
        let key = (r.lo, r.hi);
        if let Some(j) = self.fits.get(&key) {
            return *j;
        }
        let judged = judged_piece_of(&self.points_of(r), self.tol);
        self.fits.insert(key, judged);
        judged
    }

    fn classify_run(&self, r: Run) -> Option<Piece> {
        classify(&self.points_of(r), self.tol, self.scatter)
    }

    /// Run `i` and the `count` after it made one, in the list.
    fn merge_at(&mut self, i: usize, count: usize) {
        let len = self.runs.len();
        let mut union = self.runs[i];
        for k in 1..=count {
            union = self.union_of(union, self.runs[(i + k) % len]);
        }
        if self.runs[(i + count) % len].corner_after {
            union.corner_after = true;
        }
        self.runs[i] = union;
        let mut drop: Vec<usize> = (1..=count).map(|k| (i + k) % len).collect();
        drop.sort_unstable_by(|a, b| b.cmp(a));
        drop.dedup();
        for j in drop {
            self.runs.remove(j);
        }
    }

    /// Merge neighbours while one piece still explains them, best first: the pair whose
    /// union sits best within what it is allowed merges, so a fillet's chords gather into
    /// their arc before a side swallows one of them.
    fn merge_all(&mut self) {
        loop {
            if self.runs.len() <= 1 {
                break;
            }
            let len = self.runs.len();
            let pair_count = if self.closed { len } else { len - 1 };
            let mut best = None;
            let mut best_score = f64::INFINITY;
            for i in 0..pair_count {
                if self.closed && len == 2 && i == 1 {
                    break;
                }
                let u = self.union_of(self.runs[i], self.runs[(i + 1) % len]);
                if let Some(j) = self.piece_over(u) {
                    if j.score < best_score {
                        best_score = j.score;
                        best = Some(i);
                    }
                }
            }
            match best {
                Some(i) => self.merge_at(i, 1),
                None => break,
            }
        }
    }

    fn kind_of(&self, r: Run) -> PieceKind {
        self.classify_run(r).map_or(PieceKind::Line, |p| p.kind())
    }

    /// RDP puts a break where the chord bends, which on a fillet is a few points into the
    /// straight side; a piece fitted with a neighbour's points in it is biased. So every
    /// boundary between a line and an arc is slid to the split at which the two pieces, each
    /// refitted on its own side, explain their points best.
    fn settle_boundaries(&mut self) {
        for _pass in 0..3 {
            let mut moved = false;
            let pair_count = if self.closed { self.runs.len() } else { self.runs.len() - 1 };
            let mut i = 0;
            while i < pair_count && self.runs.len() > 1 {
                let j = (i + 1) % self.runs.len();
                let a = self.runs[i];
                let b = self.following(a, self.runs[j]);
                i += 1;
                if b.lo != a.hi {
                    continue;
                }
                let reach = (a.hi - a.lo).min(b.hi - b.lo).min(16);
                if reach < 2 {
                    continue;
                }
                let kind_a = self.kind_of(a);
                let kind_b = self.kind_of(b);
                if kind_a == kind_b {
                    continue;
                }
                let mut best_k = a.hi;
                let mut best_cost = f64::INFINITY;
                for k in a.hi - reach..=a.hi + reach {
                    if k - a.lo + 1 < self.min_points || b.hi - k + 1 < self.min_points {
                        continue;
                    }
                    let left = fit_as(&self.points_of(span(a.lo, k)), kind_a);
                    let right = fit_as(&self.points_of(span(k, b.hi)), kind_b);
                    let (Some(left), Some(right)) = (left, right) else { continue };
                    let cost = left.sigma().powi(2) * (k - a.lo + 1) as f64 + right.sigma().powi(2) * (b.hi - k + 1) as f64;
                    if cost < best_cost {
                        best_cost = cost;
                        best_k = k;
                    }
                }
                if best_k != a.hi {
                    moved = true;
                    self.runs[i - 1] = span(a.lo, best_k);
                    let shift = b.lo - self.runs[j].lo;
                    self.runs[j] = span(best_k - shift, b.hi - shift);
                }
            }
            if !moved {
                break;
            }
        }
    }

    fn path_length(&self, r: Run) -> f64 {
        (r.lo..r.hi).map(|i| (self.at(i + 1) - self.at(i)).length()).sum()
    }

    fn is_short(&mut self, r: Run) -> bool {
        if r.kept {
            return false;
        }
        let count = r.hi - r.lo + 1;
        if count < self.min_points {
            return true;
        }
        let length = self.path_length(r);
        if length >= UNEXPLAINED_SHORTS * self.min_length {
            return false;
        }
        // Unexplained: what is left of a corner once its sides are fitted.
        if self.piece_over(r).is_none() {
            return true;
        }
        if let Some(Piece::Arc(arc)) = self.classify_run(r) {
            // A round the size of what a wall may wander by is a lump, or a corner the
            // scanner rounded, whatever a circle through it says.
            if arc.radius < SHORT_ARC_WAVES * self.tol.line {
                return true;
            }
            if length >= self.min_length {
                return false;
            }
            return !(arc.sweep >= SHORT_ARC_SWEEP && count >= SHORT_ARC_POINTS);
        }
        length < self.min_length
    }

    /// Short pieces: noise, a burr, a blob, or a feature. What is left under the short length
    /// comes in clusters, and a cluster is asked about as one thing against the long pieces
    /// either side of it. Noise its neighbours explain with it on board: the lot is one piece.
    /// A burr on a side: the neighbours are one piece with its points left out. A blob in a
    /// corner: the neighbours carried on meet beside it, so they make the corner and the
    /// cluster goes. Anything else — a step's riser — is a feature and stays as one piece.
    fn dissolve_shorts(&mut self) {
        let n = self.n;
        let mut guard = 0;
        while guard < 4 * n && self.runs.len() > 1 {
            guard += 1;
            let count = self.runs.len();
            let runs = self.runs.clone();
            let short: Vec<bool> = runs.iter().map(|r| self.is_short(*r)).collect();
            if !short.iter().any(|s| *s) {
                break;
            }
            if short.iter().all(|s| *s) {
                for r in &mut self.runs {
                    r.kept = true;
                }
                break;
            }
            // The first cluster: from a short run whose predecessor is long, on through the
            // short runs after it.
            let mut first = None;
            for k in 0..count {
                let before = if self.closed { short[(k + count - 1) % count] } else { k > 0 && short[k - 1] };
                if short[k] && !before {
                    first = Some(k);
                    break;
                }
            }
            let Some(first) = first else { break };
            let mut size = 1;
            while size < count && short[(first + size) % count] && (self.closed || first + size < count) {
                size += 1;
            }
            let members: Vec<Run> = (0..size).map(|k| self.runs[(first + k) % count]).collect();
            let has_prev = self.closed || first > 0;
            let has_next = self.closed || first + size < count;
            let p = (first + count - 1) % count;
            let q = (first + size) % count;
            let mut cluster = members[0];
            for m in &members[1..] {
                cluster = self.union_of(cluster, *m);
            }
            let length: f64 = members.iter().map(|r| self.path_length(*r)).sum();

            if self.closed && p == q && length < UNEXPLAINED_SHORTS * self.min_length && matches!(self.classify_run(self.runs[p]), Some(Piece::Arc(_))) {
                // A short piece between the two ends of one arc: a flat the mesh's facets cut
                // into a bore or a boss. It goes, and the arc runs round on its own to close
                // into a circle. (ScanRuler keeps it; its dense scans seldom make one.)
                for k in cluster.lo + 1..cluster.hi {
                    let m = self.modn(k);
                    self.masked[m] = true;
                }
                let mut gone: Vec<usize> = (0..size).map(|k| (first + k) % count).collect();
                gone.sort_unstable_by(|x, y| y.cmp(x));
                for k in gone {
                    self.runs.remove(k);
                }
                self.fits.clear();
                continue;
            }
            if length >= UNEXPLAINED_SHORTS * self.min_length || (self.closed && p == q) {
                self.keep(first, size, cluster);
                continue;
            }
            if !has_prev || !has_next {
                // At an open chain's end: fold it in if the neighbour takes it, drop it if it
                // is no longer than a stub, keep it otherwise.
                let j = if has_next { q } else { p };
                let joined = if has_next { self.union_of(cluster, self.runs[j]) } else { self.union_of(self.runs[j], cluster) };
                if self.piece_over(joined).is_some() {
                    self.merge_at(if has_next { first } else { j }, size);
                } else if length < self.min_length {
                    let lo = cluster.lo + isize::from(has_prev);
                    let hi = cluster.hi - isize::from(has_next);
                    for k in lo..=hi {
                        let m = self.modn(k);
                        self.masked[m] = true;
                    }
                    self.runs.drain(first..first + size);
                    self.fits.clear();
                } else {
                    self.keep(first, size, cluster);
                }
                continue;
            }
            let a = self.runs[p];
            let b = self.runs[q];
            let all = self.union_of(self.union_of(a, cluster), b);
            if self.piece_over(all).is_some() {
                self.merge_at(p, size + 1);
                continue;
            }
            let inner: Vec<usize> = (cluster.lo + 1..cluster.hi).map(|k| self.modn(k)).filter(|&k| !self.masked[k]).collect();
            for &k in &inner {
                self.masked[k] = true;
            }
            self.fits.clear();
            if self.piece_over(all).is_some() {
                self.merge_at(p, size + 1);
                continue;
            }
            let pa = self.classify_run(a);
            let pb = self.classify_run(b);
            // Two lines make a corner; a line and a round, or two rounds, cross wherever their
            // circles happen to, and what lies between those is more often the small blend
            // of an S-bend than a blob.
            let meet = match (pa, pb) {
                (Some(pa @ Piece::Line(_)), Some(pb @ Piece::Line(_))) => meet_of(&pa, &pb, self.at((cluster.lo + cluster.hi).div_euclid(2))),
                _ => None,
            };
            let reach = length.max(self.min_length);
            if let Some(meet) = meet {
                if (meet - self.at(cluster.lo)).length() <= reach && (meet - self.at(cluster.hi)).length() <= reach {
                    self.runs[p].corner_after = true;
                    let mut gone: Vec<usize> = (0..size).map(|k| (first + k) % count).collect();
                    gone.sort_unstable_by(|x, y| y.cmp(x));
                    for k in gone {
                        self.runs.remove(k);
                    }
                    continue;
                }
            }
            for &k in &inner {
                self.masked[k] = false;
            }
            self.fits.clear();
            self.keep(first, size, cluster);
        }
    }

    /// The cluster made the one piece it is, and left alone from here on.
    fn keep(&mut self, first: usize, size: usize, cluster: Run) {
        if size == 1 {
            self.runs[first].kept = true;
            return;
        }
        self.merge_at(first, size - 1);
        if let Some(merged) = self.runs.iter_mut().find(|r| r.lo == cluster.lo && r.hi == cluster.hi) {
            merged.kept = true;
            merged.cluster = true;
        } else {
            let runs = self.runs.clone();
            for (i, r) in runs.iter().enumerate() {
                if self.is_short(*r) {
                    self.runs[i].kept = true;
                }
            }
        }
    }
}

/// Cut a chain into lines and arcs. Fails on a chain too short to cut at all.
pub fn segment_chain(chain: &[DVec2], opts: SegmentOptions) -> Result<SegmentedChain, FitError> {
    let view = view_chain(chain);
    let (spacing, scatter, closed) = (view.spacing, view.scatter, view.closed);
    let points = view.points;
    let n = points.len();
    if n < 4 {
        return Err(FitError::new("The chain has too few points to fit anything to."));
    }
    let tolerance = opts.tolerance.unwrap_or_else(|| default_tolerance(scatter, spacing));
    let levels = detail_levels(tolerance, opts.detail.unwrap_or(0.5));
    let tol = Tolerances { line: levels.wave, arc: levels.arc };
    let min_points = opts.min_points.unwrap_or(4).max(3) as isize;
    let min_length = opts.min_length.unwrap_or(levels.short);

    // ---- the corners and chords: RDP at the tolerance ----
    let breaks: Vec<usize> = if closed {
        // A loop has no ends for RDP to hang from: the point farthest from the first gives
        // two, and each half is simplified on its own.
        let mut far = 1;
        let mut far_d = -1.0;
        for (i, p) in points.iter().enumerate().skip(1) {
            let d = (*p - points[0]).length();
            if d > far_d {
                far_d = d;
                far = i;
            }
        }
        let first = simplify_run(&points, 0, far + 1, tolerance);
        let second = simplify_run(&points, far, n, tolerance);
        let mut b = first;
        b.extend_from_slice(&second[1..]);
        b.push(n); // n stands for the closing return to 0
        b
    } else {
        simplify_run(&points, 0, n, tolerance)
    };
    let runs: Vec<Run> = breaks.windows(2).map(|w| span(w[0] as isize, w[1] as isize)).collect();

    let mut cut = Cutter {
        points,
        n: n as isize,
        closed,
        tol,
        scatter,
        min_points,
        min_length,
        runs,
        masked: vec![false; n],
        fits: HashMap::new(),
    };
    cut.merge_all();
    cut.settle_boundaries();
    cut.dissolve_shorts();

    // ---- name each run ----
    let mut segments = Vec::new();
    for r in &cut.runs {
        let mut pts = cut.points_of(*r);
        // A kept cluster is a riser with the scanner's rounding at both ends: its middle half
        // is the riser, and what the line is fitted to and held to.
        if r.cluster && pts.len() >= 8 {
            let q = pts.len() >> 2;
            pts = pts[q..pts.len() - q].to_vec();
        }
        let line = if r.cluster { line_of(&pts) } else { None };
        let piece = match line {
            Some(l) => Some(Piece::Line(l)),
            None => classify(&pts, tol, scatter),
        };
        let Some(piece) = piece else { continue };
        segments.push(ChainSegment { piece, lo: r.lo, hi: r.hi, points: pts, corner_after: r.corner_after });
    }
    if segments.is_empty() {
        return Err(FitError::new("No line or arc could be fitted to the chain."));
    }
    Ok(SegmentedChain { segments, points: cut.points, closed, scatter, spacing, tolerance })
}

/// How many places along a chain an arc may end at, at most, when it is cut into a given
/// number of arcs.
const ARC_BREAKS: usize = 160;

/// A chain as `count` arcs end to end: the breaks are where the arcs together sit closest on
/// the points. For a curve the cut makes otherwise — two tangent arcs of a near radius that
/// one arc within the noise would swallow, a long shallow curve.
pub fn arcs_of_chain(chain: &[DVec2], count: usize, tolerance: Option<f64>) -> Result<SegmentedChain, FitError> {
    let view = view_chain(chain);
    let points = view.points;
    let n = points.len();
    let pieces = count.max(1);
    if n < 5 * pieces {
        return Err(FitError::new(format!(
            "The chain has {n} points — too few for {pieces} arc{}, five each at least.",
            if pieces == 1 { "" } else { "s" }
        )));
    }
    let tolerance = tolerance.unwrap_or_else(|| default_tolerance(view.scatter, view.spacing));
    let stride = n.div_ceil(ARC_BREAKS).max(1);
    let mut at: Vec<usize> = (0..n - 1).step_by(stride).collect();
    at.push(n - 1);
    let m = at.len();
    let mut costs: HashMap<(usize, usize), Option<(f64, ArcFit)>> = HashMap::new();
    let mut arc_cost = |a: usize, b: usize| -> Option<(f64, ArcFit)> {
        *costs.entry((a, b)).or_insert_with(|| {
            let run = &points[at[a]..=at[b]];
            let fit = arc_of(run)?;
            let cost = run.iter().map(|p| ((*p - fit.center).length() - fit.radius).powi(2)).sum();
            Some((cost, fit))
        })
    };
    let mut best = vec![vec![f64::INFINITY; m]; pieces];
    let mut from = vec![vec![usize::MAX; m]; pieces];
    for k in 0..pieces {
        for b in 1..m {
            if k == 0 {
                best[0][b] = arc_cost(0, b).map_or(f64::INFINITY, |c| c.0);
                continue;
            }
            for a in k..b {
                if !best[k - 1][a].is_finite() {
                    continue;
                }
                let Some((c, _)) = arc_cost(a, b) else { continue };
                let total = best[k - 1][a] + c;
                if total < best[k][b] {
                    best[k][b] = total;
                    from[k][b] = a;
                }
            }
        }
    }
    if !best[pieces - 1][m - 1].is_finite() {
        return Err(FitError::new(format!("The chain cannot be cut into {pieces} arcs — at least one stretch of it is straight or too short.")));
    }
    let mut ends = vec![m - 1];
    for k in (1..pieces).rev() {
        ends.insert(0, from[k][ends[0]]);
    }
    ends.insert(0, 0);
    let mut segments = Vec::new();
    for k in 0..pieces {
        let (lo, hi) = (at[ends[k]], at[ends[k + 1]]);
        let (_, fit) = arc_cost(ends[k], ends[k + 1]).expect("cost was finite");
        segments.push(ChainSegment { piece: Piece::Arc(fit), lo: lo as isize, hi: hi as isize, points: points[lo..=hi].to_vec(), corner_after: false });
    }
    Ok(SegmentedChain { segments, points, closed: false, scatter: view.scatter, spacing: view.spacing, tolerance })
}

/// How many places along a chain a piece may end at, at most, when it is cut into given kinds.
const KIND_BREAKS: usize = 100;
/// The most points a piece's cost is read over while the breaks are sought.
const KIND_COST_POINTS: usize = 200;

/// A chain as the pieces `kinds` names, end to end and in that order — the lines and arcs of
/// another station, so the stations of a loft come out piece for piece alike. `closed`: the
/// chain is a loop starting at its first point, and the last piece comes back to it. Returns
/// the cut and its cost (the sum of squared misses, mm²).
pub fn kinds_of_chain(chain: &[DVec2], kinds: &[PieceKind], tolerance: Option<f64>, closed: bool) -> Result<(SegmentedChain, f64), FitError> {
    let view = view_chain(chain);
    let mut points = view.points.clone();
    if closed {
        points.push(view.points[0]);
    }
    let n = points.len();
    let pieces = kinds.len();
    if pieces == 0 {
        return Err(FitError::new("There are no pieces to cut the chain into."));
    }
    let fewest = |k: PieceKind| if k == PieceKind::Arc { 5 } else { 3 };
    let names = || kinds.iter().map(|k| if *k == PieceKind::Arc { "arc" } else { "line" }).collect::<Vec<_>>().join(", ");
    let needed = kinds.iter().fold(1, |sum, k| sum + fewest(*k) - 1);
    if n < needed {
        return Err(FitError::new(format!("The chain has {n} points — too few for {pieces} pieces ({}).", names())));
    }
    let tolerance = tolerance.unwrap_or_else(|| default_tolerance(view.scatter, view.spacing));
    let stride = (n - 1).div_ceil(KIND_BREAKS).max(1);
    let mut at: Vec<usize> = (0..n - 1).step_by(stride).collect();
    at.push(n - 1);
    let m = at.len();
    let mut costs: HashMap<(usize, usize, PieceKind), f64> = HashMap::new();
    let mut cost_of = |a: usize, b: usize, kind: PieceKind| -> f64 {
        *costs.entry((a, b, kind)).or_insert_with(|| {
            let count = at[b] - at[a] + 1;
            if count < fewest(kind) {
                return f64::INFINITY;
            }
            let step = count.div_ceil(KIND_COST_POINTS).max(1);
            let mut run = Vec::new();
            let mut last = usize::MAX;
            let mut i = at[a];
            while i <= at[b] {
                run.push(points[i]);
                last = i;
                i += step;
            }
            if last != at[b] {
                run.push(points[at[b]]);
            }
            let sigma = match kind {
                PieceKind::Arc => fit_arc_points(&run).ok().filter(|f| f.radius.is_finite()).map(|f| f.sigma),
                PieceKind::Line => fit_line_points(&run).ok().map(|f| f.sigma),
            };
            match sigma {
                Some(s) if s.is_finite() => s * s * count as f64,
                _ => f64::INFINITY,
            }
        })
    };
    let mut best = vec![vec![f64::INFINITY; m]; pieces];
    let mut from = vec![vec![usize::MAX; m]; pieces];
    for k in 0..pieces {
        for b in 1..m {
            if k == 0 {
                best[0][b] = cost_of(0, b, kinds[0]);
                continue;
            }
            for a in k..b {
                if !best[k - 1][a].is_finite() {
                    continue;
                }
                let total = best[k - 1][a] + cost_of(a, b, kinds[k]);
                if total < best[k][b] {
                    best[k][b] = total;
                    from[k][b] = a;
                }
            }
        }
    }
    if !best[pieces - 1][m - 1].is_finite() {
        return Err(FitError::new(format!("The chain cannot be cut into {} — a stretch of it is too short for its piece, or no piece fits it.", names())));
    }
    let mut ends = vec![m - 1];
    for k in (1..pieces).rev() {
        ends.insert(0, from[k][ends[0]]);
    }
    ends.insert(0, 0);
    let mut segments = Vec::new();
    let mut cost = 0.0;
    for k in 0..pieces {
        let (lo, hi) = (at[ends[k]], at[ends[k + 1]]);
        let run = points[lo..=hi].to_vec();
        let piece = match kinds[k] {
            PieceKind::Arc => fit_arc_points(&run).ok().map(Piece::Arc),
            PieceKind::Line => fit_line_points(&run).ok().map(Piece::Line),
        }
        .ok_or_else(|| FitError::new(format!("Piece {} ({}) does not fit its stretch of the chain.", k + 1, if kinds[k] == PieceKind::Arc { "arc" } else { "line" })))?;
        cost += piece.sigma().powi(2) * run.len() as f64;
        segments.push(ChainSegment { piece, lo: lo as isize, hi: hi as isize, points: run, corner_after: false });
    }
    Ok((SegmentedChain { segments, points, closed, scatter: view.scatter, spacing: view.spacing, tolerance }, cost))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::fit2d::Mulberry32;

    /// A rounded rectangle traced as a slice would be, with Gaussian noise across it.
    pub(crate) fn rounded_rect_chain(w: f64, h: f64, r: f64, step: f64, noise: f64, seed: u32) -> Vec<DVec2> {
        let mut rand = Mulberry32::new(seed);
        let mut pts = Vec::new();
        let corners = [(DVec2::new(w - r, r), -PI / 2.0), (DVec2::new(w - r, h - r), 0.0), (DVec2::new(r, h - r), PI / 2.0), (DVec2::new(r, r), PI)];
        let straights = [
            (DVec2::new(r, 0.0), DVec2::new(w - r, 0.0)),
            (DVec2::new(w, r), DVec2::new(w, h - r)),
            (DVec2::new(w - r, h), DVec2::new(r, h)),
            (DVec2::new(0.0, h - r), DVec2::new(0.0, r)),
        ];
        for side in 0..4 {
            let (a, b) = straights[side];
            let len = (b - a).length();
            let count = ((len / step).round() as usize).max(1);
            let normal = DVec2::new(-(b.y - a.y) / len, (b.x - a.x) / len);
            for i in 0..count {
                let t = i as f64 / count as f64;
                let e = rand.gaussian() * noise;
                pts.push(a + (b - a) * t + normal * e);
            }
            let (c, start) = corners[side];
            let m = (((r * PI) / 2.0 / step).round() as usize).max(2);
            for i in 0..m {
                let ang = start + (PI / 2.0) * i as f64 / m as f64;
                let rr = r + rand.gaussian() * noise;
                pts.push(c + DVec2::new(ang.cos(), ang.sin()) * rr);
            }
        }
        pts.push(pts[0]);
        pts
    }

    #[test]
    fn cuts_a_rounded_rectangle_into_four_lines_and_four_arcs() {
        let chain = rounded_rect_chain(60.0, 40.0, 5.0, 0.25, 0.03, 3);
        let cut = segment_chain(&chain, SegmentOptions::default()).unwrap();
        assert!(cut.closed);
        let lines = cut.segments.iter().filter(|s| s.piece.is_line()).count();
        let arcs: Vec<_> = cut.segments.iter().filter_map(|s| if let Piece::Arc(a) = s.piece { Some(a) } else { None }).collect();
        assert_eq!(lines, 4, "{:#?}", cut.segments.iter().map(|s| s.piece.kind()).collect::<Vec<_>>());
        assert_eq!(arcs.len(), 4);
        for a in arcs {
            assert!((a.radius - 5.0).abs() < 0.5, "radius {}", a.radius);
        }
        for i in 0..cut.segments.len() {
            assert_ne!(cut.segments[i].piece.kind(), cut.segments[(i + 1) % cut.segments.len()].piece.kind());
        }
    }

    #[test]
    fn cuts_a_sharp_rectangle_into_four_lines() {
        let chain = rounded_rect_chain(50.0, 30.0, 0.001, 0.25, 0.02, 3);
        let cut = segment_chain(&chain, SegmentOptions::default()).unwrap();
        assert_eq!(cut.segments.iter().filter(|s| s.piece.is_line()).count(), 4);
        assert_eq!(cut.segments.len(), 4);
    }

    #[test]
    fn reads_a_circle_as_one_arc() {
        let mut rand = Mulberry32::new(9);
        let mut chain: Vec<DVec2> = (0..360)
            .map(|i| {
                let a = (i as f64).to_radians();
                let r = 12.0 + rand.gaussian() * 0.02;
                DVec2::new(r * a.cos(), r * a.sin())
            })
            .collect();
        chain.push(chain[0]);
        let cut = segment_chain(&chain, SegmentOptions::default()).unwrap();
        assert_eq!(cut.segments.len(), 1);
        assert!(cut.segments[0].piece.is_arc());
    }

    #[test]
    fn a_bore_with_a_facet_flat_is_one_arc() {
        // A circle of radius 4 with a 1.5 mm chord cut across it, as a coarse mesh's facets cut
        // a bore.
        let mut chain = Vec::new();
        let flat_half = (0.75f64 / 4.0).asin();
        let (a0, a1) = (flat_half, std::f64::consts::TAU - flat_half);
        for i in 0..=160 {
            let a = a0 + (a1 - a0) * i as f64 / 160.0;
            chain.push(DVec2::new(4.0 * a.cos(), 4.0 * a.sin()));
        }
        let (p, q) = (chain[chain.len() - 1], chain[0]);
        for i in 1..6 {
            chain.push(p + (q - p) * (i as f64 / 6.0));
        }
        chain.push(chain[0]);
        let cut = segment_chain(&chain, SegmentOptions::default()).unwrap();
        assert_eq!(cut.segments.len(), 1, "{:?}", cut.segments.iter().map(|s| s.piece.kind()).collect::<Vec<_>>());
        assert!(cut.segments[0].piece.is_arc());
    }

    #[test]
    fn takes_a_coarser_tolerance_as_fewer_longer_pieces() {
        let chain = rounded_rect_chain(60.0, 40.0, 1.2, 0.25, 0.03, 3);
        let fine = segment_chain(&chain, SegmentOptions { tolerance: Some(0.05), ..Default::default() }).unwrap();
        let coarse = segment_chain(&chain, SegmentOptions { tolerance: Some(1.0), ..Default::default() }).unwrap();
        assert!(coarse.segments.len() < fine.segments.len());
        assert_eq!(coarse.segments.len(), 4);
    }

    #[test]
    fn arcs_of_chain_splits_two_radii() {
        let mut chain = Vec::new();
        for i in 0..=90 {
            let a = (i as f64).to_radians();
            chain.push(DVec2::new(10.0 * a.cos(), 10.0 * a.sin()));
        }
        // A second arc of radius 20 tangent at (0, 10), centre (0, -10).
        for i in 1..=60 {
            let a = (90.0 + i as f64 * 0.5).to_radians();
            chain.push(DVec2::new(20.0 * a.cos(), -10.0 + 20.0 * a.sin()));
        }
        let cut = arcs_of_chain(&chain, 2, None).unwrap();
        let radii: Vec<f64> = cut.segments.iter().map(|s| if let Piece::Arc(a) = s.piece { a.radius } else { 0.0 }).collect();
        assert!((radii[0] - 10.0).abs() < 0.2 && (radii[1] - 20.0).abs() < 0.4, "{radii:?}");
    }

    #[test]
    fn kinds_of_chain_follows_the_kinds() {
        let mut chain: Vec<DVec2> = (0..=40).map(|i| DVec2::new(i as f64 * 0.5, 0.0)).collect();
        for i in 1..=30 {
            let a = (-90.0 + i as f64 * 3.0).to_radians();
            chain.push(DVec2::new(20.0 + 5.0 * a.cos(), 5.0 + 5.0 * a.sin()));
        }
        let (cut, cost) = kinds_of_chain(&chain, &[PieceKind::Line, PieceKind::Arc], None, false).unwrap();
        assert!(cut.segments[0].piece.is_line() && cut.segments[1].piece.is_arc());
        assert!(cost < 1e-3, "cost {cost}");
    }
}
