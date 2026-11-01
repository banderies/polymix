// SPDX-License-Identifier: AGPL-3.0-only
//! A chain of regular teeth read for the design behind it — a Fresnel's rings on a section
//! through its axis, a rack, a knurl, a ribbed face. Ported from ScanRuler
//! (`plugins/reverse/core/teeth.ts`), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! Fitted piece by piece, teeth come out as dozens of short lines and arcs, broken where the
//! scan rounds their tips; what was drawn is less: two envelopes the tips and the roots lie
//! on, one pitch, and where in each tooth its steep side stands.
//!
//! The chain's trend comes first — the line or arc it follows overall — and each point's
//! height off it. The tips and the roots are the extremes of that height, taken with a
//! hysteresis of about a third of the teeth's depth so the scan's noise makes none; each is
//! then set at the corner its two sides meet at, the middle of each side fitted as a line,
//! for a scan rounds a sharp tip. The tips (crests: the extremes out of the material, where
//! the cut says which side that is) and the roots (troughs) are fitted with an envelope each,
//! a line or an arc. The pitch is the spacing that lays the corners on a regular comb, read
//! along whichever measure keeps them most regular: the sheet's u or v — a Fresnel's rings
//! stand at one radial pitch, which on a section through its axis is u — or along the trend,
//! as a rack's teeth and a gear's do. The profile is that comb laid on the envelopes: each
//! corner at its regular place, on its envelope.

use std::cmp::Ordering;
use std::f64::consts::PI;

use glam::DVec2;
use serde::Serialize;

use crate::chain::view_chain;
use crate::fit2d::{fit_circle_2d, solve_linear, FitError};

/// A line or an arc the chain, its tips or its roots follow, and how its points sit on it
/// (rms and worst, mm).
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum Envelope {
    /// An arc. Its `radius_sigma` is the fit's own uncertainty in its radius, mm: a short arc
    /// of a large circle lays its points well for many radii, its centre moving with them.
    #[serde(rename_all = "camelCase")]
    Arc { center: DVec2, radius: f64, radius_sigma: f64, rms: f64, worst: f64 },
    /// A line through `point` along `dir`, its + u way.
    Line { point: DVec2, dir: DVec2, rms: f64, worst: f64 },
}

impl Envelope {
    /// Whether it is an arc.
    pub fn is_arc(&self) -> bool {
        matches!(self, Envelope::Arc { .. })
    }

    fn same_kind(&self, other: &Envelope) -> bool {
        self.is_arc() == other.is_arc()
    }
}

/// What the pitch is measured along: the sheet's u or v, or the trend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum PitchAlong {
    U,
    V,
    Trend,
}

/// What a corner of a tooth is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ToothCorner {
    Crest,
    Trough,
}

/// How the crests were told from the troughs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum CrestsBy {
    /// By the material: the extremes out of it.
    Material,
    /// By side — away from the trend's centre, or on its left for a straight one — where the
    /// cut does not say.
    Side,
}

/// An envelope and the corners it was fitted to.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CornerEnvelope {
    #[serde(flatten)]
    pub envelope: Envelope,
    pub corners: Vec<DVec2>,
}

/// How far each trough lies off the crests' envelope, mm.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct TeethDepth {
    pub mean: f64,
    pub min: f64,
    pub max: f64,
}

/// How the chain's points between the profile's ends sit on it.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct TeethFit {
    pub points: usize,
    pub rms: f64,
    pub worst: f64,
}

/// A chain of teeth read for its design.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TeethRead {
    /// The line or arc the chain follows overall.
    pub trend: Envelope,
    /// The tips' envelope and the corners it was fitted to.
    pub crests: CornerEnvelope,
    /// The roots'.
    pub troughs: CornerEnvelope,
    /// How the crests were told from the troughs: by the material (the extremes out of it),
    /// or by side — away from the trend's centre, or on its left for a straight one — where
    /// the cut does not say.
    pub crests_by: CrestsBy,
    pub pitch: f64,
    pub along: PitchAlong,
    /// How far the corners lie off the regular comb along the pitch, rms mm.
    pub spread: f64,
    /// Along the pitch, from each crest to the next trough.
    pub crest_to_trough: f64,
    /// The shorter and the longer part of a tooth along the pitch.
    pub riser: f64,
    pub facet: f64,
    /// How far each trough lies off the crests' envelope, mm.
    pub depth: TeethDepth,
    /// The profile: the corners in order along the pitch, each at its regular place on its
    /// envelope — and what each is.
    pub profile: Vec<DVec2>,
    pub kinds: Vec<ToothCorner>,
    /// The pitch measure of the profile's first corner.
    pub start: f64,
    /// How the chain's points between the profile's ends sit on it.
    pub fit: TeethFit,
}

/// What the cut says about a chain of teeth.
#[derive(Debug, Clone, Copy, Default)]
pub struct TeethOptions<'a> {
    /// Each point's way out of the material, where the cut says.
    pub outward: Option<&'a [Option<DVec2>]>,
}

/// At least this many crests and troughs make a chain of teeth.
const TEETH_MIN: usize = 3;

/// An extreme must stand this share of the teeth's depth (the heights' 10–90 % spread) past
/// the last one, and this many of the chain's scatters at least, to count.
const HYSTERESIS: f64 = 0.35;

const NOISE_STEPS: f64 = 4.0;

/// The share of each side left out at its corners before its middle is fitted as a line:
/// where the scan rounds the tip.
const SIDE_TRIM: f64 = 0.2;

/// An arc is taken for a trend or an envelope where its rms is under this share of a line's,
/// and its radius under this.
const ARC_GAIN: f64 = 0.5;

const ARC_RADIUS_MAX: f64 = 1e5;

/// Two sides meeting more nearly parallel than this make no corner.
fn corner_sin() -> f64 {
    ((5.0 * PI) / 180.0).sin()
}

/// A measure along a curved trend is taken over u or v only where it keeps the corners this
/// much more regular.
const ALONG_GAIN: f64 = 0.8;

/// JavaScript's `Math.round`, halves up.
fn js_round(v: f64) -> f64 {
    crate::round::js_round(v)
}

/// The best line through points (total least squares), with how they sit on it.
#[derive(Debug, Clone, Copy)]
struct LineThrough {
    point: DVec2,
    dir: DVec2,
    rms: f64,
    worst: f64,
}

/// The best line through points (total least squares), with how they sit on it; its
/// direction runs + u (or + v, square to u). `None` for fewer than two distinct points.
fn line_through(points: &[DVec2]) -> Option<LineThrough> {
    let n = points.len();
    if n < 2 {
        return None;
    }
    let nf = n as f64;
    let (mut mx, mut my) = (0.0, 0.0);
    for p in points {
        mx += p.x / nf;
        my += p.y / nf;
    }
    let (mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0);
    for p in points {
        let x = p.x - mx;
        let y = p.y - my;
        sxx += x * x;
        sxy += x * y;
        syy += y * y;
    }
    if !(sxx + syy > 0.0) {
        return None;
    }
    let t = 0.5 * (2.0 * sxy).atan2(sxx - syy);
    let mut dir = DVec2::new(t.cos(), t.sin());
    if dir.x < -1e-12 || (dir.x.abs() <= 1e-12 && dir.y < 0.0) {
        dir = -dir;
    }
    let (mut ss, mut worst) = (0.0, 0.0f64);
    for p in points {
        let d = (dir.x * (p.y - my) - dir.y * (p.x - mx)).abs();
        ss += d * d;
        worst = worst.max(d);
    }
    Some(LineThrough { point: DVec2::new(mx, my), dir, rms: (ss / nf).sqrt(), worst })
}

/// The line or arc points follow: an arc where it fits them well past a line (ARC_GAIN), a
/// line otherwise.
pub fn envelope_of(points: &[DVec2]) -> Result<Envelope, FitError> {
    let line = line_through(points).ok_or_else(|| FitError::new("The points coincide — they make no line or arc."))?;
    let circle = if points.len() >= 4 { fit_circle_2d(points) } else { None };
    if let Some((center, r)) = circle {
        if r.is_finite() && r < ARC_RADIUS_MAX {
            let (mut ss, mut worst) = (0.0, 0.0f64);
            for p in points {
                let d = ((p.x - center.x).hypot(p.y - center.y) - r).abs();
                ss += d * d;
                worst = worst.max(d);
            }
            let rms = (ss / points.len() as f64).sqrt();
            if rms < ARC_GAIN * line.rms {
                return Ok(Envelope::Arc { center, radius: r, radius_sigma: radius_sigma_of(points, center, rms), rms, worst });
            }
        }
    }
    Ok(Envelope::Line { point: line.point, dir: line.dir, rms: line.rms, worst: line.worst })
}

/// The standard error of a fitted circle's radius: from the fit's Jacobian in its centre and
/// radius, the points' scatter about it taken as their noise.
fn radius_sigma_of(points: &[DVec2], center: DVec2, rms: f64) -> f64 {
    let n = points.len();
    if n <= 3 {
        return f64::INFINITY;
    }
    let mut jtj = [0.0f64; 9];
    for p in points {
        let mut d = (p.x - center.x).hypot(p.y - center.y);
        if d == 0.0 || d.is_nan() {
            d = 1.0;
        }
        let row = [-(p.x - center.x) / d, -(p.y - center.y) / d, -1.0];
        for a in 0..3 {
            for b in 0..3 {
                jtj[a * 3 + b] += row[a] * row[b];
            }
        }
    }
    let Some(col) = solve_linear(3, &mut jtj, &[0.0, 0.0, 1.0]) else { return f64::INFINITY };
    if !(col[2] > 0.0) {
        return f64::INFINITY;
    }
    rms * (n as f64 / (n as f64 - 3.0)).sqrt() * col[2].sqrt()
}

/// How far a point lies off an envelope, signed: out from an arc's centre, to the left of a
/// line's direction.
pub fn off_envelope(e: &Envelope, p: DVec2) -> f64 {
    match *e {
        Envelope::Arc { center, radius, .. } => (p.x - center.x).hypot(p.y - center.y) - radius,
        Envelope::Line { point, dir, .. } => dir.x * (p.y - point.y) - dir.y * (p.x - point.x),
    }
}

/// The way a point's height off an envelope counts positive.
fn up_from(e: &Envelope, p: DVec2) -> DVec2 {
    match *e {
        Envelope::Line { dir, .. } => DVec2::new(-dir.y, dir.x),
        Envelope::Arc { center, .. } => {
            let mut d = (p.x - center.x).hypot(p.y - center.y);
            if d == 0.0 || d.is_nan() {
                d = 1.0;
            }
            DVec2::new((p.x - center.x) / d, (p.y - center.y) / d)
        }
    }
}

/// Where along a trend a point lies, mm: along a line from its point, round an arc
/// counter-clockwise, the angle unwound near `near` (radians). Returns the measure and the
/// angle.
fn along_trend(e: &Envelope, p: DVec2, near: f64) -> (f64, f64) {
    match *e {
        Envelope::Line { point, dir, .. } => (dir.x * (p.x - point.x) + dir.y * (p.y - point.y), 0.0),
        Envelope::Arc { center, radius, .. } => {
            let mut a = (p.y - center.y).atan2(p.x - center.x);
            // A NaN angle would never settle; it is left as it is.
            if a.is_finite() && near.is_finite() {
                while a - near > PI {
                    a -= 2.0 * PI;
                }
                while a - near < -PI {
                    a += 2.0 * PI;
                }
            }
            (radius * a, a)
        }
    }
}

/// Where two lines cross, as a point; `None` for lines near parallel.
fn crossing(p: DVec2, d: DVec2, q: DVec2, e: DVec2) -> Option<DVec2> {
    let den = d.x * e.y - d.y * e.x;
    if den.abs() < corner_sin() {
        return None;
    }
    let t = ((q.x - p.x) * e.y - (q.y - p.y) * e.x) / den;
    Some(DVec2::new(p.x + t * d.x, p.y + t * d.y))
}

/// Where a line through `p` along the unit `d` meets an envelope, the crossing nearest
/// `near`; `None` where it misses it.
fn on_envelope(e: &Envelope, p: DVec2, d: DVec2, near: DVec2) -> Option<DVec2> {
    match *e {
        Envelope::Line { point, dir, .. } => crossing(p, d, point, dir),
        Envelope::Arc { center, radius, .. } => {
            let w = p - center;
            let b = d.x * w.x + d.y * w.y;
            let disc = b * b - (w.x * w.x + w.y * w.y - radius * radius);
            if disc < 0.0 {
                return None;
            }
            let hits = [-b - disc.sqrt(), -b + disc.sqrt()].map(|t| DVec2::new(p.x + t * d.x, p.y + t * d.y));
            let far = |q: DVec2| (q.x - near.x).hypot(q.y - near.y);
            Some(if far(hits[0]) <= far(hits[1]) { hits[0] } else { hits[1] })
        }
    }
}

/// The tooth's corner at an extreme: where the lines through the middles of its two sides
/// cross, or the extreme itself where they make none near it.
fn corner_at(chain: &[DVec2], before: usize, at: usize, after: usize) -> DVec2 {
    let side = |from: usize, to: usize, trim_from: bool, trim_to: bool| -> Option<LineThrough> {
        let n = to.saturating_sub(from) as f64;
        let a = from + if trim_from { (SIDE_TRIM * n).ceil() as usize } else { 0 };
        let b = to.saturating_sub(if trim_to { (SIDE_TRIM * n).ceil() as usize } else { 0 });
        let end = (b + 1).min(chain.len());
        if a >= end {
            return None;
        }
        line_through(&chain[a..end])
    };
    let left = side(before, at, before > 0, true);
    let right = side(at, after, true, after + 1 < chain.len());
    let p = chain[at];
    let (Some(left), Some(right)) = (left, right) else { return p };
    let Some(q) = crossing(left.point, left.dir, right.point, right.dir) else { return p };
    // A corner the sides make far from the extreme is no corner of it.
    let to_before = (chain[at].x - chain[before].x).hypot(chain[at].y - chain[before].y);
    let to_after = (chain[after].x - chain[at].x).hypot(chain[after].y - chain[at].y);
    let reach = 0.5 * to_before.min(to_after);
    if (q.x - p.x).hypot(q.y - p.y) <= reach {
        q
    } else {
        p
    }
}

/// The share at `f` of sorted values.
fn quantile(sorted: &[f64], f: f64) -> f64 {
    let last = sorted.len().saturating_sub(1);
    let i = js_round(f * last as f64).max(0.0) as usize;
    sorted.get(i.min(last)).copied().unwrap_or(f64::NAN)
}

/// The distance from a point to a polyline.
fn off_polyline(line: &[DVec2], p: DVec2) -> f64 {
    let mut best = f64::INFINITY;
    for w in line.windows(2) {
        let (a, b) = (w[0], w[1]);
        let ab = b - a;
        let l2 = ab.x * ab.x + ab.y * ab.y;
        let t = if l2 > 0.0 { (((p.x - a.x) * ab.x + (p.y - a.y) * ab.y) / l2).clamp(0.0, 1.0) } else { 0.0 };
        best = best.min((p.x - a.x - t * ab.x).hypot(p.y - a.y - t * ab.y));
    }
    best
}

/// An extreme of the chain's height off its trend.
#[derive(Debug, Clone, Copy)]
struct Extreme {
    i: usize,
    kind: ToothCorner,
}

/// An extreme set at its tooth's corner.
#[derive(Debug, Clone, Copy)]
struct Corner {
    i: usize,
    kind: ToothCorner,
    at: DVec2,
}

/// A corner on the comb: which corner, its measure, and the tooth it belongs to.
#[derive(Debug, Clone, Copy)]
struct Row {
    corner: usize,
    s: f64,
    k: f64,
}

/// The regular comb along one measure.
#[derive(Debug, Clone)]
struct Comb {
    along: PitchAlong,
    rows: Vec<Row>,
    a0: f64,
    r: f64,
    p: f64,
    spread: f64,
    irregular: f64,
}

impl Comb {
    fn place(&self, kind: ToothCorner, k: f64) -> f64 {
        self.a0 + k * self.p + if kind == ToothCorner::Trough { self.r } else { 0.0 }
    }
}

/// A chain of regular teeth read as its crests, troughs, envelopes and pitch, and the regular
/// profile they make — see the head of this file. Fails where the chain has fewer than three
/// crests and three troughs about its trend.
pub fn read_teeth(chain0: &[DVec2], opts: TeethOptions<'_>) -> Result<TeethRead, FitError> {
    let closed = chain0.len() > 2 && chain0.first().map(|p| (p.x, p.y)) == chain0.last().map(|p| (p.x, p.y));
    let chain = if closed { &chain0[..chain0.len() - 1] } else { chain0 };
    let n = chain.len();
    if n < 4 * TEETH_MIN {
        return Err(FitError::new(format!("The chain has {n} points — too few for teeth.")));
    }
    let trend = envelope_of(chain)?;

    // Each point's height off the trend, crests positive: out of the material where the cut
    // says which way that is.
    let mut toward = 0.0;
    let mut told = 0usize;
    if let Some(out) = opts.outward {
        for (i, &p) in chain.iter().enumerate() {
            let Some(Some(o)) = out.get(i) else { continue };
            let up = up_from(&trend, p);
            toward += o.x * up.x + o.y * up.y;
            told += 1;
        }
    }
    let by_material = told > 0 && (toward / told as f64).abs() > 0.05;
    let sign = if by_material && toward < 0.0 { -1.0 } else { 1.0 };
    let g: Vec<f64> = chain.iter().map(|&p| sign * off_envelope(&trend, p)).collect();

    // The extremes, with a hysteresis of a third of the depth.
    let mut sorted = g.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let depth = quantile(&sorted, 0.9) - quantile(&sorted, 0.1);
    let delta = (HYSTERESIS * depth).max(NOISE_STEPS * view_chain(chain).scatter);
    let mut found: Vec<Extreme> = Vec::new();
    let mut mode = 0i8;
    let mut hi = 0usize;
    let mut lo = 0usize;
    for i in 1..n {
        if mode == 0 {
            if g[i] > g[hi] {
                hi = i;
            }
            if g[i] < g[lo] {
                lo = i;
            }
            if g[hi] - g[lo] > delta {
                if hi > lo {
                    found.push(Extreme { i: lo, kind: ToothCorner::Trough });
                    mode = 1;
                } else {
                    found.push(Extreme { i: hi, kind: ToothCorner::Crest });
                    mode = -1;
                }
            }
        } else if mode == 1 {
            if g[i] > g[hi] {
                hi = i;
            } else if g[i] < g[hi] - delta {
                found.push(Extreme { i: hi, kind: ToothCorner::Crest });
                mode = -1;
                lo = i;
            }
        } else if g[i] < g[lo] {
            lo = i;
        } else if g[i] > g[lo] + delta {
            found.push(Extreme { i: lo, kind: ToothCorner::Trough });
            mode = 1;
            hi = i;
        }
    }
    // An end of the chain is no corner.
    let ext: Vec<Extreme> = found.into_iter().filter(|e| e.i > 1 && e.i + 2 < n).collect();
    let crest_count = ext.iter().filter(|e| e.kind == ToothCorner::Crest).count();
    let trough_count = ext.len() - crest_count;
    if crest_count < TEETH_MIN || trough_count < TEETH_MIN {
        return Err(FitError::new(format!(
            "Only {crest_count} crests and {trough_count} troughs stand out of the chain’s trend — teeth take {TEETH_MIN} of each at least; give within round one band of teeth."
        )));
    }

    // Each at the corner its sides meet at.
    let corners: Vec<Corner> = ext
        .iter()
        .enumerate()
        .map(|(k, e)| {
            let before = if k > 0 { ext[k - 1].i } else { 0 };
            let after = ext.get(k + 1).map_or(n - 1, |x| x.i);
            Corner { i: e.i, kind: e.kind, at: corner_at(chain, before, e.i, after) }
        })
        .collect();
    let corners_of = |kind: ToothCorner| corners.iter().filter(|c| c.kind == kind).map(|c| c.at).collect::<Vec<_>>();
    let crest_env = envelope_of(&corners_of(ToothCorner::Crest))?;
    let trough_env = envelope_of(&corners_of(ToothCorner::Trough))?;

    // The measure along the trend at each corner, unwound along the chain — along the crests'
    // envelope where it is the trend's kind: the tips lay the pitch line truer than the
    // chain's lopsided teeth do.
    let pitch_line = if crest_env.same_kind(&trend) { crest_env } else { trend };
    let mut near = along_trend(&pitch_line, chain[0], 0.0).1;
    let mut angle_at: Vec<f64> = Vec::with_capacity(n);
    for &p in chain {
        near = along_trend(&pitch_line, p, near).1;
        angle_at.push(near);
    }
    let mut measures: Vec<PitchAlong> = vec![PitchAlong::Trend];
    if trend.is_arc() {
        measures.push(PitchAlong::U);
        measures.push(PitchAlong::V);
    }
    let measure = |along: PitchAlong, c: &Corner| -> f64 {
        match along {
            PitchAlong::Trend => along_trend(&pitch_line, c.at, angle_at[c.i]).0,
            PitchAlong::U => c.at.x,
            PitchAlong::V => c.at.y,
        }
    };

    // The regular comb along each measure: crest k at a + k·p, the trough after it at
    // a + r + k·p, least squares.
    let combs: Vec<Option<Comb>> = measures
        .iter()
        .map(|&along| {
            let mut order: Vec<(usize, f64)> = corners.iter().enumerate().map(|(j, c)| (j, measure(along, c))).collect();
            // A stable sort, as JavaScript's is.
            order.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            let mut k: f64 = if corners[order[0].0].kind == ToothCorner::Crest { 0.0 } else { -1.0 };
            let rows: Vec<Row> = order
                .iter()
                .enumerate()
                .map(|(j, &(corner, s))| {
                    if j > 0 && corners[corner].kind == ToothCorner::Crest {
                        k += 1.0;
                    }
                    Row { corner, s, k }
                })
                .collect();
            let mut ata = [0.0f64; 9];
            let mut atb = [0.0f64; 3];
            for row in &rows {
                let a_row = [1.0, if corners[row.corner].kind == ToothCorner::Trough { 1.0 } else { 0.0 }, row.k];
                for a in 0..3 {
                    atb[a] += a_row[a] * row.s;
                    for b in 0..3 {
                        ata[a * 3 + b] += a_row[a] * a_row[b];
                    }
                }
            }
            let x = solve_linear(3, &mut ata, &atb)?;
            if !(x[2] > 0.0) {
                return None;
            }
            let mut comb = Comb { along, rows, a0: x[0], r: x[1], p: x[2], spread: 0.0, irregular: 0.0 };
            let mut ss = 0.0;
            for row in &comb.rows {
                ss += (row.s - comb.place(corners[row.corner].kind, row.k)).powi(2);
            }
            comb.spread = (ss / comb.rows.len() as f64).sqrt();
            comb.irregular = comb.spread / comb.p;
            Some(comb)
        })
        .collect();
    let fits: Vec<Comb> = combs.into_iter().flatten().collect();
    if fits.is_empty() {
        return Err(FitError::new("The crests and troughs make no regular comb along the chain."));
    }
    // Along the trend, unless u or v keeps the corners clearly more regular.
    let along_trend_comb = fits.iter().position(|c| c.along == PitchAlong::Trend);
    let mut best = 0usize;
    for (j, c) in fits.iter().enumerate().skip(1) {
        if c.irregular < fits[best].irregular {
            best = j;
        }
    }
    let chosen = match along_trend_comb {
        Some(t) if best != t && !(fits[best].irregular < ALONG_GAIN * fits[t].irregular) => t,
        _ => best,
    };
    let comb = &fits[chosen];

    // The profile: each corner at its regular place, on its envelope — where the line square
    // to the measure there meets it.
    let station = |s: f64| -> (DVec2, DVec2) {
        match comb.along {
            PitchAlong::U => (DVec2::new(s, 0.0), DVec2::new(0.0, 1.0)),
            PitchAlong::V => (DVec2::new(0.0, s), DVec2::new(1.0, 0.0)),
            PitchAlong::Trend => match pitch_line {
                Envelope::Line { point, dir, .. } => (DVec2::new(point.x + s * dir.x, point.y + s * dir.y), DVec2::new(-dir.y, dir.x)),
                Envelope::Arc { center, radius, .. } => {
                    let a = s / radius;
                    (center, DVec2::new(a.cos(), a.sin()))
                }
            },
        }
    };
    let mut profile: Vec<DVec2> = Vec::with_capacity(comb.rows.len());
    let mut kinds: Vec<ToothCorner> = Vec::with_capacity(comb.rows.len());
    for row in &comb.rows {
        let c = &corners[row.corner];
        let (p, d) = station(comb.place(c.kind, row.k));
        let env = if c.kind == ToothCorner::Crest { &crest_env } else { &trough_env };
        profile.push(on_envelope(env, p, d, c.at).unwrap_or(c.at));
        kinds.push(c.kind);
    }

    let troughs_off: Vec<f64> = corners.iter().filter(|c| c.kind == ToothCorner::Trough).map(|c| off_envelope(&crest_env, c.at).abs()).collect();
    let r = ((comb.r % comb.p) + comb.p) % comb.p;
    // The chain's points between the profile's first and last corner.
    let from = ext.iter().map(|e| e.i).min().unwrap_or(0);
    let to = ext.iter().map(|e| e.i).max().unwrap_or(0);
    let mut ss = 0.0;
    let mut worst = 0.0f64;
    for &p in &chain[from..=to] {
        let d = off_polyline(&profile, p);
        ss += d * d;
        worst = worst.max(d);
    }
    let first = &comb.rows[0];
    Ok(TeethRead {
        trend,
        crests: CornerEnvelope { envelope: crest_env, corners: corners_of(ToothCorner::Crest) },
        troughs: CornerEnvelope { envelope: trough_env, corners: corners_of(ToothCorner::Trough) },
        crests_by: if by_material { CrestsBy::Material } else { CrestsBy::Side },
        pitch: comb.p,
        along: comb.along,
        spread: comb.spread,
        crest_to_trough: r,
        riser: r.min(comb.p - r),
        facet: r.max(comb.p - r),
        depth: TeethDepth {
            mean: troughs_off.iter().sum::<f64>() / troughs_off.len() as f64,
            min: troughs_off.iter().copied().fold(f64::INFINITY, f64::min),
            max: troughs_off.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        },
        profile,
        kinds,
        start: comb.place(corners[first.corner].kind, first.k),
        fit: TeethFit { points: to - from + 1, rms: (ss / (to - from + 1) as f64).sqrt(), worst },
    })
}

#[cfg(test)]
mod tests {
    //! reTeeth.test.ts: the lamp's Fresnel as its section through the axis gave it — crests
    //! on an R92.21 envelope, troughs on an R92.70 one, from r0 15.92 at a 2.47 radial pitch
    //! with a 0.85 riser, 21 teeth — clean, with scan noise and with its tips rounded as a
    //! scan rounds them, and a straight rack.
    use super::*;
    use crate::fit2d::Mulberry32;

    /// The lamp's Fresnel: u the radius, v along the axis.
    const CREST_R: f64 = 92.21;
    const TROUGH_R: f64 = 92.7;
    const R0: f64 = 15.92;
    const PITCH: f64 = 2.47;
    const RISER: f64 = 0.85;
    const TEETH: usize = 21;

    // The envelopes' centres on the axis, as the run's profile has them: a crest at u 10.98,
    // v −45.3139 and a trough at u 11.83, v −45.8421.
    fn crest_v0() -> f64 {
        -45.3139 + (CREST_R * CREST_R - 10.98 * 10.98).sqrt()
    }
    fn trough_v0() -> f64 {
        -45.8421 + (TROUGH_R * TROUGH_R - 11.83 * 11.83).sqrt()
    }
    fn crest_at(u: f64) -> DVec2 {
        DVec2::new(u, crest_v0() - (CREST_R * CREST_R - u * u).sqrt())
    }
    fn trough_at(u: f64) -> DVec2 {
        DVec2::new(u, trough_v0() - (TROUGH_R * TROUGH_R - u * u).sqrt())
    }

    /// The teeth's corners from the trough before the first crest to the crest after the last
    /// trough, in order out along u.
    fn corners() -> Vec<DVec2> {
        let mut out = vec![trough_at(R0 - PITCH + RISER)];
        for k in 0..TEETH {
            let u = R0 + k as f64 * PITCH;
            out.push(crest_at(u));
            out.push(trough_at(u + RISER));
        }
        out.push(crest_at(R0 + TEETH as f64 * PITCH));
        out
    }

    /// An open polyline with its sides cut into steps no longer than `step`, its last point
    /// kept.
    fn along(poly: &[DVec2], step: f64) -> Vec<DVec2> {
        let mut out = Vec::new();
        for w in poly.windows(2) {
            let (a, b) = (w[0], w[1]);
            let k = ((b.x - a.x).hypot(b.y - a.y) / step).ceil().max(1.0) as usize;
            for t in 0..k {
                let t = t as f64;
                let kf = k as f64;
                out.push(DVec2::new(a.x + ((b.x - a.x) * t) / kf, a.y + ((b.y - a.y) * t) / kf));
            }
        }
        if let Some(&last) = poly.last() {
            out.push(last);
        }
        out
    }

    /// The profile as a section gives it: from the middle of the facet before the first crest
    /// to the middle of the facet after the last trough, a point every `step` mm, each moved
    /// by noise; with `round`, each corner cut by a chord `round` mm back along both sides, as
    /// a scan blunts a tip.
    fn sawtooth(step: f64, noise: f64, round: f64) -> Vec<DVec2> {
        let mut c = corners();
        let mid = |a: DVec2, b: DVec2| DVec2::new((a.x + b.x) / 2.0, (a.y + b.y) / 2.0);
        let len = c.len();
        let mut trimmed = vec![mid(c[0], c[1])];
        trimmed.extend_from_slice(&c[1..len - 1]);
        trimmed.push(mid(c[len - 2], c[len - 1]));
        c = trimmed;
        if round > 0.0 {
            let mut cut = vec![c[0]];
            let back = |p: DVec2, q: DVec2| {
                let d = (q.x - p.x).hypot(q.y - p.y);
                DVec2::new(p.x + ((q.x - p.x) * round) / d, p.y + ((q.y - p.y) * round) / d)
            };
            for i in 1..c.len() - 1 {
                cut.push(back(c[i], c[i - 1]));
                cut.push(back(c[i], c[i + 1]));
            }
            cut.push(c[c.len() - 1]);
            c = cut;
        }
        let mut rand = Mulberry32::new(5);
        along(&c, step)
            .into_iter()
            .map(|p| {
                let x = p.x + rand.gaussian() * noise;
                let y = p.y + rand.gaussian() * noise;
                DVec2::new(x, y)
            })
            .collect()
    }

    #[track_caller]
    fn near(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol, "{a} against {b}");
    }

    fn arc_of(e: &Envelope) -> (DVec2, f64, f64) {
        match *e {
            Envelope::Arc { center, radius, radius_sigma, .. } => (center, radius, radius_sigma),
            Envelope::Line { .. } => panic!("the envelope is a line, not an arc"),
        }
    }

    #[test]
    fn reads_the_fresnels_envelopes_its_radial_pitch_riser_and_first_crest() -> Result<(), FitError> {
        let chain = sawtooth(0.05, 0.0, 0.0);
        // The material lies below the teeth: out of it is +v.
        let outward: Vec<Option<DVec2>> = chain.iter().map(|_| Some(DVec2::new(0.0, 1.0))).collect();
        let t = read_teeth(&chain, TeethOptions { outward: Some(&outward) })?;
        assert_eq!(t.crests_by, CrestsBy::Material);
        assert_eq!(t.along, PitchAlong::U);
        assert!(t.crests.envelope.is_arc());
        assert!(t.troughs.envelope.is_arc());
        let (crest_center, crest_r, _) = arc_of(&t.crests.envelope);
        let (_, trough_r, _) = arc_of(&t.troughs.envelope);
        near(crest_r, CREST_R, 0.01);
        near(trough_r, TROUGH_R, 0.01);
        near(crest_center.x, 0.0, 0.01);
        near(t.pitch, PITCH, 1e-3);
        near(t.riser, RISER, 1e-3);
        near(t.facet, PITCH - RISER, 1e-3);
        near(t.crest_to_trough, RISER, 1e-3);
        near(t.start, R0, 2e-3);
        assert_eq!(t.profile.len(), 42);
        assert_eq!(t.kinds[0], ToothCorner::Crest);
        assert_eq!(t.kinds[41], ToothCorner::Trough);
        // Each corner of the profile on the design's.
        let want = corners();
        let want = &want[1..want.len() - 1];
        for (i, (p, w)) in t.profile.iter().zip(want).enumerate() {
            assert!((*p - *w).length() < 2e-3, "corner {i}: {p} against {w}");
        }
        assert!(t.fit.rms < 1e-3);
        near(t.depth.mean, 0.6, 0.1);
        Ok(())
    }

    #[test]
    fn reads_them_through_the_scans_noise_and_its_rounded_tips() -> Result<(), FitError> {
        let chain = sawtooth(0.08, 0.02, 0.12);
        let outward: Vec<Option<DVec2>> = chain.iter().map(|_| Some(DVec2::new(0.0, 1.0))).collect();
        let t = read_teeth(&chain, TeethOptions { outward: Some(&outward) })?;
        assert_eq!(t.along, PitchAlong::U);
        assert!(t.crests.envelope.is_arc() && t.troughs.envelope.is_arc(), "envelopes are arcs");
        let (_, crest_r, crest_sigma) = arc_of(&t.crests.envelope);
        let (_, trough_r, _) = arc_of(&t.troughs.envelope);
        near(crest_r, CREST_R, 0.25);
        near(trough_r, TROUGH_R, 0.25);
        // The radius is the least sure of it: its sigma says by how much.
        assert!((crest_r - CREST_R).abs() < 3.0 * crest_sigma);
        assert!(crest_sigma < 0.3);
        near(t.pitch, PITCH, 0.005);
        near(t.riser, RISER, 0.02);
        near(t.start, R0, 0.03);
        assert_eq!(t.profile.len(), 42);
        assert!(t.fit.rms < 0.05);
        Ok(())
    }

    #[test]
    fn tells_crests_by_the_material_else_by_side_and_refuses_a_chain_with_no_teeth() -> Result<(), FitError> {
        let chain = sawtooth(0.05, 0.0, 0.0);
        // The material above the teeth instead: the tips are the other way.
        let outward: Vec<Option<DVec2>> = chain.iter().map(|_| Some(DVec2::new(0.0, -1.0))).collect();
        let down = read_teeth(&chain, TeethOptions { outward: Some(&outward) })?;
        assert_eq!(down.crests_by, CrestsBy::Material);
        near(arc_of(&down.crests.envelope).1, TROUGH_R, 0.01);
        for c in &down.crests.corners {
            near(c.y, trough_at(c.x).y, 1e-3);
        }
        // Told nothing: the crests are those away from the trend's centre — below it here —
        // and the profile is the same.
        let side = read_teeth(&chain, TeethOptions::default())?;
        assert_eq!(side.crests_by, CrestsBy::Side);
        near(arc_of(&side.crests.envelope).1, TROUGH_R, 0.01);
        assert_eq!(side.profile.len(), 42);
        let flat = read_teeth(&along(&[DVec2::new(0.0, 0.0), DVec2::new(10.0, 0.0), DVec2::new(10.0, 5.0)], 0.1), TeethOptions::default());
        let message = flat.err().map(|e| e.to_string()).unwrap_or_default();
        assert!(message.contains("teeth take 3 of each"), "{message}");
        Ok(())
    }

    /// The reading as the command answers it: the envelopes tagged by kind, their corners
    /// beside them.
    #[test]
    fn answers_in_the_shape_the_command_gives() -> Result<(), Box<dyn std::error::Error>> {
        let chain = sawtooth(0.05, 0.0, 0.0);
        let outward: Vec<Option<DVec2>> = chain.iter().map(|_| Some(DVec2::new(0.0, 1.0))).collect();
        let t = serde_json::to_value(read_teeth(&chain, TeethOptions { outward: Some(&outward) })?)?;
        assert_eq!(t["crests"]["kind"], serde_json::json!("arc"));
        assert!(t["crests"]["radiusSigma"].is_number());
        assert_eq!(t["crests"]["corners"].as_array().map(Vec::len), Some(21));
        assert_eq!(t["crestsBy"], serde_json::json!("material"));
        assert_eq!(t["along"], serde_json::json!("u"));
        assert_eq!(t["kinds"][0], serde_json::json!("crest"));
        Ok(())
    }

    #[test]
    fn reads_a_straight_rack_along_its_trend() -> Result<(), FitError> {
        // Teeth 2 apart on a line at 30°, 0.8 deep, their steep side 0.5 long.
        let a = PI / 6.0;
        let d = DVec2::new(a.cos(), a.sin());
        let nrm = DVec2::new(-d.y, d.x);
        let at = |s: f64, h: f64| DVec2::new(5.0 + s * d.x + h * nrm.x, 3.0 + s * d.y + h * nrm.y);
        let mut c = vec![at(-0.7, 0.4)];
        for k in 0..8 {
            let k = k as f64;
            c.push(at(2.0 * k, 0.8));
            c.push(at(2.0 * k + 0.5, 0.0));
        }
        c.push(at(16.0 - 0.7, 0.4));
        let t = read_teeth(&along(&c, 0.05), TeethOptions::default())?;
        assert_eq!(t.along, PitchAlong::Trend);
        assert!(!t.crests.envelope.is_arc());
        near(t.pitch, 2.0, 1e-3);
        near(t.riser, 0.5, 1e-3);
        near(t.depth.mean, 0.8, 1e-3);
        assert_eq!(t.profile.len(), 16);
        Ok(())
    }
}
