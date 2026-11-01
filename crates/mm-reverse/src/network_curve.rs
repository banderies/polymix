// SPDX-License-Identifier: AGPL-3.0-only
//! A curve of a patch network: its trace along the scan and the smooth curve fitted to it.
//!
//! Ported from ScanRuler (`plugins/reverse/core/networkCurve.ts`), © Stefan Hermann
//! (CNC Kitchen), AGPL-3.0-only.
//!
//! The trace is the scan path ([`crate::scan_path`]) between each two of the curve's points
//! in turn (its end node, its grips, its other end node), so it lies on the scan and follows
//! its noise. The curve is a clamped cubic B-spline with uniform knots
//! ([`crate::network_spline`]), fitted to the trace by least squares at its arc-length
//! parameter: through its two ends exactly (the nodes, which the curves meeting there share)
//! and kept fair by a slight term on the poles' second differences, which holds a span with
//! no points in it between its neighbours; in a network, its smoothness fairs the curve as
//! well ([`fit_curve`]'s `fairing`). Where the curve goes on smoothly into another at a node
//! (a regular node, PN8), the end's tangent is given and its second pole held on that line,
//! so the two meet tangent.
//!
//! Its span count is its strip's (the network fit, whose Auto starts every strip at four and
//! grows it where a patch misses the scan). A curve that is no cell's whole side takes
//! [`auto_spans`]: the trace's length over eight times the scan's scatter along it, between
//! 2 and 20, the rule Fit Surface sizes its net by. The scatter is read off a pilot fit's
//! residual, as Fit Surface reads it: what six spans cannot follow is noise, or a shape that
//! wants more spans.

use glam::DVec3;
use serde::{Deserialize, Serialize};

use crate::linalg::{cholesky_solve, solve_linear};
use crate::network_spline::{basis_at, curve_derivative, curve_point, divided_weights, greville_of, DEGREE};
use crate::scan_path::{trace_path, PathWay, ScanPaths};

/// The fewest and most spans a curve, a strip, takes.
pub const SPANS_MIN: usize = 2;
pub const SPANS_MAX: usize = 20;
/// How far Auto lets a patch stray from the scan beyond its noise before its strips get more
/// spans, where a network gives no tolerance, mm (the network fit).
pub const DEFAULT_TOLERANCE: f64 = 0.01;
/// The pilot fit's spans, for Auto's scatter.
const PILOT_SPANS: usize = 6;
/// The fairing term's weight per pole, as a share of the data's. A circle's poles have
/// second differences of millimetres, so the term bends any arc inward: at 1e-3 it took a
/// 270° arc of 20 mm radius 19 µm in at eight spans; at 1e-6, under a tenth of a micron. A
/// span no point falls in is still held between its neighbours.
const FAIRING: f64 = 1e-6;
/// The scatter Auto never reads below, mm.
const SCATTER_FLOOR: f64 = 1e-3;
/// The rounds of parameter correction a curve's fit takes.
const CURVE_ROUNDS: usize = 0;

/// A run of a trace between two of its curve's points: a section or the edges' fallback,
/// with the triangles its ends lie in.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TraceRun {
    pub way: PathWay,
    pub from: usize,
    pub to: usize,
    pub ends: [i32; 2],
}

/// A curve's trace: the scan paths between its points, end to end.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct CurveTrace {
    /// x, y, z per point.
    pub points: Vec<f32>,
    /// Where each point lies on the mesh, two numbers a point (`ScanPath::on`).
    pub on: Vec<i32>,
    /// The arc length at each point from the start, mm.
    pub at: Vec<f64>,
    pub length: f64,
    /// Where each of the curve's own points is in `points`: the first end, the grips, the
    /// other end.
    pub grips: Vec<usize>,
    /// The runs between the points, each a section or the edges' fallback, with the
    /// triangles their ends lie in.
    pub runs: Vec<TraceRun>,
}

/// The points a curve is fitted to: a trace, or any polyline with its arc length.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct CurvePoints {
    /// x, y, z per point.
    pub points: Vec<f32>,
    /// The arc length at each point from the start, mm.
    pub at: Vec<f64>,
    pub length: f64,
}

/// A borrowed view of the points a curve is fitted to: a trace's, or a polyline's.
#[derive(Debug, Clone, Copy)]
pub struct PointsRef<'a> {
    pub points: &'a [f32],
    pub at: &'a [f64],
    pub length: f64,
}

impl<'a> PointsRef<'a> {
    /// How many points.
    pub fn count(&self) -> usize {
        self.points.len() / 3
    }

    /// Point i.
    pub fn point(&self, i: usize) -> DVec3 {
        DVec3::new(self.points[i * 3] as f64, self.points[i * 3 + 1] as f64, self.points[i * 3 + 2] as f64)
    }
}

impl<'a> From<&'a CurveTrace> for PointsRef<'a> {
    fn from(t: &'a CurveTrace) -> Self {
        PointsRef { points: &t.points, at: &t.at, length: t.length }
    }
}

impl<'a> From<&'a CurvePoints> for PointsRef<'a> {
    fn from(t: &'a CurvePoints) -> Self {
        PointsRef { points: &t.points, at: &t.at, length: t.length }
    }
}

impl CurveTrace {
    pub fn as_points(&self) -> PointsRef<'_> {
        self.into()
    }

    /// Point i.
    pub fn point(&self, i: usize) -> DVec3 {
        self.as_points().point(i)
    }

    /// Sets point i (as ScanRuler's `points.set(p, i * 3)`).
    pub fn set_point(&mut self, i: usize, p: DVec3) {
        if i * 3 + 2 < self.points.len() {
            self.points[i * 3] = p.x as f32;
            self.points[i * 3 + 1] = p.y as f32;
            self.points[i * 3 + 2] = p.z as f32;
        }
    }
}

impl CurvePoints {
    pub fn as_points(&self) -> PointsRef<'_> {
        self.into()
    }

    /// A polyline as points with their arc length.
    pub fn from_polyline(points: Vec<f32>) -> Self {
        let at = arc_lengths(&points);
        let length = at.last().copied().unwrap_or(0.0);
        CurvePoints { points, at, length }
    }
}

/// The arc length at each point of a polyline (x, y, z per point), from the first.
pub(crate) fn arc_lengths(xyz: &[f32]) -> Vec<f64> {
    let count = xyz.len() / 3;
    let mut at = vec![0.0; count];
    for i in 1..count {
        let d = |k: usize| xyz[i * 3 + k] as f64 - xyz[i * 3 - 3 + k] as f64;
        at[i] = at[i - 1] + (d(0) * d(0) + d(1) * d(1) + d(2) * d(2)).sqrt();
    }
    at
}

/// The trace through `points`, each taken to the scan, in order.
pub fn trace_curve(scan: &ScanPaths, points: &[DVec3]) -> CurveTrace {
    let mut xyz: Vec<f32> = Vec::new();
    let mut on: Vec<i32> = Vec::new();
    let mut grips: Vec<usize> = Vec::new();
    let mut runs: Vec<TraceRun> = Vec::new();
    for k in 0..points.len().saturating_sub(1) {
        let path = trace_path(scan, points[k], points[k + 1], None);
        let n = path.points.len() / 3;
        if n == 0 {
            continue;
        }
        // Each run starts where the last ended: its first point is dropped.
        let skip = if xyz.is_empty() { 0 } else { 1 };
        let from = xyz.len() / 3 - skip;
        if skip == 0 {
            grips.push(0);
        }
        for i in skip..n {
            xyz.extend_from_slice(&path.points[i * 3..i * 3 + 3]);
            on.extend_from_slice(&path.on[i * 2..i * 2 + 2]);
        }
        grips.push(xyz.len() / 3 - 1);
        runs.push(TraceRun { way: path.way, from, to: xyz.len() / 3 - 1, ends: path.ends });
    }
    if xyz.is_empty() && !points.is_empty() {
        if let Some(c) = scan.closest(points[0]) {
            xyz.extend_from_slice(&[c.point.x as f32, c.point.y as f32, c.point.z as f32]);
            on.extend_from_slice(&[-1, c.triangle as i32]);
            grips.push(0);
        }
    }
    let at = arc_lengths(&xyz);
    let length = at.last().copied().unwrap_or(0.0);
    CurveTrace { points: xyz, on, at, length, grips, runs }
}

/// A curve fitted to a trace.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CurveFit {
    pub spans: usize,
    /// spans + 3 poles, x, y, z each.
    pub poles: Vec<f64>,
    /// The trace's distances from the curve at their parameters, mm.
    pub rms: f64,
    pub max: f64,
}

/// What a fit holds besides its data: the tangent at either end, a unit direction pointing
/// along the curve, from the start and into the end.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct CurveEnds {
    pub start_tangent: Option<DVec3>,
    pub end_tangent: Option<DVec3>,
}

/// The clamped cubic of `spans` spans through the trace's two ends, fitted to the rest of it
/// (see the module documentation). `fairing` is a network's smoothness on the curve (the
/// network fit): the weight of ∫ |C‴|² ds against the points, mm⁵; the third derivative, as
/// a patch's, which bends a curve's arc less than the second would.
// CURVE_ROUNDS is ScanRuler's own, at none: its rounds' loop is kept, empty.
#[allow(clippy::reversed_empty_ranges)]
pub fn fit_curve(trace: PointsRef<'_>, spans: usize, ends: &CurveEnds, fairing: f64) -> CurveFit {
    let count = trace.count();
    let n = spans.max(SPANS_MIN) + DEGREE;
    let s = n - DEGREE;
    let length = if trace.length > 0.0 { trace.length } else { 1.0 };
    let first = if count > 0 { trace.point(0) } else { DVec3::ZERO };
    let last = if count > 0 { trace.point(count - 1) } else { first };
    // Each pole coordinate as a constant and at most one unknown: the ends fixed, a held
    // end's second pole the end plus a free distance along its tangent, the rest free.
    let mut base = vec![0.0f64; n * 3];
    let mut unknown = vec![-1i64; n * 3];
    let mut coef = vec![0.0f64; n * 3];
    let mut k = 0usize;
    for c in 0..3 {
        base[c] = first[c];
        base[(n - 1) * 3 + c] = last[c];
    }
    let start_held = ends.start_tangent.map(|_| {
        k += 1;
        k - 1
    });
    let end_held = ends.end_tangent.map(|_| {
        k += 1;
        k - 1
    });
    for i in 1..n - 1 {
        for c in 0..3 {
            let at = i * 3 + c;
            match (i, ends.start_tangent, ends.end_tangent, start_held, end_held) {
                (1, Some(tangent), _, Some(held), _) => {
                    base[at] = first[c];
                    unknown[at] = held as i64;
                    coef[at] = tangent[c];
                }
                (i, _, Some(tangent), _, Some(held)) if i == n - 2 => {
                    base[at] = last[c];
                    unknown[at] = held as i64;
                    coef[at] = -tangent[c];
                }
                _ => {
                    unknown[at] = k as i64;
                    k += 1;
                    coef[at] = 1.0;
                }
            }
        }
    }
    // The poles fitted with the points at parameters t.
    let solve_at = |t: &[f64]| -> (Vec<f64>, Vec<f64>) {
        let mut ata = vec![0.0f64; k * k];
        let mut atb = vec![0.0f64; k];
        let mut zs: Vec<usize> = Vec::with_capacity(8);
        let mut ws: Vec<f64> = Vec::with_capacity(8);
        // A row Σ w[m] · pole(idx[m], c) = rhs, weighted by `weight`.
        let mut add_row = |row_idx: &[usize], row_w: &[f64], c: usize, rhs: f64, weight: f64| {
            let mut r = rhs;
            zs.clear();
            ws.clear();
            for m in 0..row_idx.len() {
                let at = row_idx[m] * 3 + c;
                r -= row_w[m] * base[at];
                if unknown[at] >= 0 {
                    zs.push(unknown[at] as usize);
                    ws.push(row_w[m] * coef[at]);
                }
            }
            for a in 0..zs.len() {
                atb[zs[a]] += weight * ws[a] * r;
                for b in 0..zs.len() {
                    ata[zs[a] * k + zs[b]] += weight * ws[a] * ws[b];
                }
            }
        };
        let mut basis = [0.0f64; 4];
        let mut row_idx: Vec<usize> = Vec::with_capacity(4);
        let mut row_w: Vec<f64> = Vec::with_capacity(4);
        for p in 0..count {
            let span = basis_at(s, t[p], &mut basis);
            row_idx.clear();
            row_w.clear();
            for (i, w) in basis.iter().enumerate() {
                row_idx.push(span - DEGREE + i);
                row_w.push(*w);
            }
            for c in 0..3 {
                add_row(&row_idx, &row_w, c, trace.points[p * 3 + c] as f64, 1.0);
            }
        }
        // The fairing: each pole's second difference, weighed as a share of the points a
        // pole carries.
        let fair = FAIRING * (count as f64 / n as f64).max(1.0);
        for i in 1..n - 1 {
            for c in 0..3 {
                add_row(&[i - 1, i, i + 1], &[1.0, -2.0, 1.0], c, 0.0, fair);
            }
        }
        // The smoothness: third divided differences over the Greville abscissae, each over
        // its share of the curve; in [0, 1] a third derivative stands for one along the curve
        // times its length cubed.
        if fairing > 0.0 {
            let g = greville_of(s);
            let weight = fairing / ((n - 1) as f64 * length.powf(5.0));
            for i in 0..n.saturating_sub(3) {
                let w = divided_weights(&g, i, 3);
                let idx = [i, i + 1, i + 2, i + 3];
                for c in 0..3 {
                    add_row(&idx, &w, c, 0.0, weight);
                }
            }
        }
        let mut diag = 0.0;
        for i in 0..k {
            diag += ata[i * k + i];
        }
        for i in 0..k {
            ata[i * k + i] += 1e-12 * (diag / (k.max(1) as f64)).max(1e-12);
        }
        let z = cholesky_solve(k, &ata, &atb, 1e-12)
            .or_else(|| solve_linear(k, &mut ata.clone(), &atb))
            .unwrap_or_else(|| vec![0.0; k]);
        let mut poles = vec![0.0f64; n * 3];
        for at in 0..n * 3 {
            poles[at] = base[at] + if unknown[at] >= 0 { coef[at] * z[unknown[at] as usize] } else { 0.0 };
        }
        (poles, z)
    };
    // The points start at their arc length's share; after each fit every one but the ends is
    // taken to the parameter of its nearest point on the curve (Gauss–Newton), and the curve
    // fitted again. A uniform cubic does not run at even speed (round a circle its clamped
    // ends run slower) and held to the arc length it bends off the points to keep up: 19 µm
    // off a 270° arc of 20 mm radius at eight spans. Two rounds take that to under a micron.
    let mut t: Vec<f64> = (0..count).map(|p| trace.at.get(p).copied().unwrap_or(0.0) / length).collect();
    let mut solved = solve_at(&t);
    for _ in 0..CURVE_ROUNDS {
        project_onto(&solved.0, s, trace.points, &mut t);
        solved = solve_at(&t);
    }
    let (poles, z) = solved;
    // A held end's pole the wrong side of the end (the tangent given the wrong way, or a
    // curve too short for it) falls back to the free fit.
    let start_wrong = start_held.is_some_and(|h| z[h] < 0.0);
    let end_wrong = end_held.is_some_and(|h| z[h] < 0.0);
    if start_wrong || end_wrong {
        let ends = CurveEnds {
            start_tangent: if start_wrong { None } else { ends.start_tangent },
            end_tangent: if end_wrong { None } else { ends.end_tangent },
        };
        return fit_curve(trace, spans, &ends, fairing);
    }
    project_onto(&poles, s, trace.points, &mut t);
    let mut sq = 0.0;
    let mut max = 0.0f64;
    for p in 0..count {
        let q = curve_point(&poles, s, t[p]);
        let d = (q - trace.point(p)).length();
        sq += d * d;
        if d > max {
            max = d;
        }
    }
    let rms = if count > 0 { (sq / count as f64).sqrt() } else { 0.0 };
    CurveFit { spans: s, poles, rms, max }
}

/// Each point but the two ends taken to the parameter of its nearest point on the curve: a
/// few Gauss–Newton steps from where it was, in [0, 1].
fn project_onto(poles: &[f64], spans: usize, points: &[f32], t: &mut [f64]) {
    if t.len() < 2 {
        return;
    }
    for p in 1..t.len() - 1 {
        let mut tt = t[p];
        let x = DVec3::new(points[p * 3] as f64, points[p * 3 + 1] as f64, points[p * 3 + 2] as f64);
        for _ in 0..3 {
            let q = curve_point(poles, spans, tt);
            let d = curve_derivative(poles, spans, tt);
            let dd = d.dot(d);
            if !(dd > 0.0) {
                break;
            }
            let g = (x - q).dot(d);
            tt = (tt + g / dd).clamp(0.0, 1.0);
        }
        t[p] = tt;
    }
}

/// Auto's span count for a trace: its length over eight times its scatter (a pilot fit's
/// RMS), between SPANS_MIN and SPANS_MAX.
pub fn auto_spans(trace: PointsRef<'_>) -> usize {
    if !(trace.length > 0.0) {
        return SPANS_MIN;
    }
    let scatter = fit_curve(trace, PILOT_SPANS, &CurveEnds::default(), 0.0).rms;
    let spans = (trace.length / (8.0 * scatter.max(SCATTER_FLOOR))).round();
    if spans.is_nan() {
        return SPANS_MIN;
    }
    spans.clamp(SPANS_MIN as f64, SPANS_MAX as f64) as usize
}

/// The provisional fit's span a this many of the scan's edges along the trace
/// ([`fit_provisional`]).
const PROVISIONAL_EDGES: f64 = 6.0;

/// A curve fitted for the directions it leaves its ends by (what orders the curves round a
/// node and pairs those going straight on through it) before its strip says its spans:
/// Auto, but no finer than a span every PROVISIONAL_EDGES of the scan's edges (`spacing`,
/// mm). A trace has a point or two an edge, and a span with fewer than a handful of them
/// leaves its end's tangent to the noise: a 20 mm spoke on a 1.1 mm mesh at Auto's 20 spans
/// left its centre 9° out of the surface.
pub fn fit_provisional(trace: PointsRef<'_>, spacing: f64) -> CurveFit {
    let dense = if spacing > 0.0 {
        let d = (trace.length / (PROVISIONAL_EDGES * spacing)).floor();
        if d.is_nan() {
            SPANS_MIN
        } else {
            (d.max(SPANS_MIN as f64)).min(1e9) as usize
        }
    } else {
        SPANS_MAX
    };
    fit_curve(trace, auto_spans(trace).min(dense), &CurveEnds::default(), 0.0)
}

/// How far along a trace the way it leaves an end is read: this many of the scan's edges,
/// and never more than this share of the trace.
const LEAVING_EDGES: f64 = 10.0;
const LEAVING_SHARE: f64 = 0.3;

/// The way a trace leaves its first end (0) or its last (1), a unit vector pointing along it
/// away from the end: a quadratic through the end fitted by least squares to the points
/// within LEAVING_EDGES of the scan's edges (`spacing`, mm), at least 2 mm, of it. A fitted
/// curve's own end tangent is read off its first span alone, a few noisy points at Auto's
/// spans: the four curves meeting a 30° crease's two ends came out at 10° and 20° where the
/// scan's halves stand at 15°, and the patches' corners twisted the fold from 26° to 35°
/// along the crease.
pub fn leaving_direction(trace: PointsRef<'_>, end: u8, spacing: f64) -> DVec3 {
    let n = trace.count();
    if n < 2 {
        return DVec3::X;
    }
    let reach = (LEAVING_EDGES * spacing).max(2.0).min(LEAVING_SHARE * trace.length);
    let i0 = if end == 0 { 0 } else { n - 1 };
    let p0 = trace.point(i0);
    // Σ over the points of (s, s²) against their offsets from the end.
    let (mut ss, mut s3, mut s4) = (0.0f64, 0.0f64, 0.0f64);
    let mut b1 = DVec3::ZERO;
    let mut b2 = DVec3::ZERO;
    let mut used = 0;
    let order: Box<dyn Iterator<Item = usize>> = if end == 0 { Box::new(1..n) } else { Box::new((0..n - 1).rev()) };
    for i in order {
        let s = (trace.at[i] - trace.at[i0]).abs();
        if s > reach && used >= 2 {
            break;
        }
        if !(s > 0.0) {
            continue;
        }
        let d = trace.point(i) - p0;
        b1 += s * d;
        b2 += s * s * d;
        ss += s * s;
        s3 += s * s * s;
        s4 += s * s * s * s;
        used += 1;
    }
    let det = ss * s4 - s3 * s3;
    let mut dir = DVec3::ZERO;
    for k in 0..3 {
        dir[k] = if det > 1e-18 * (ss * s4).max(1.0) {
            (b1[k] * s4 - b2[k] * s3) / det
        } else if ss > 0.0 {
            b1[k] / ss
        } else {
            0.0
        };
    }
    let l = dir.length();
    if !(l > 0.0) {
        let q = if end == 0 { 1 } else { n - 2 };
        let d = trace.point(q) - p0;
        let ld = d.length();
        let ld = if ld != 0.0 && !ld.is_nan() { ld } else { 1.0 };
        return d / ld;
    }
    dir / l
}

/// The unit tangent of a fitted curve at its start (`end` 0, along it) or its end (`end` 1,
/// into it).
pub fn end_tangent(fit: &CurveFit, end: u8) -> DVec3 {
    let d = curve_derivative(&fit.poles, fit.spans, if end == 0 { 0.0 } else { 1.0 });
    let l = d.length();
    let l = if l != 0.0 && !l.is_nan() { l } else { 1.0 };
    d / l
}

/// The fitted curve sampled at `count` evenly spaced parameters, flat.
pub fn sample_curve(fit: &CurveFit, count: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(count * 3);
    for i in 0..count {
        let q = curve_point(&fit.poles, fit.spans, if count > 1 { i as f64 / (count - 1) as f64 } else { 0.0 });
        out.extend_from_slice(&[q.x, q.y, q.z]);
    }
    out
}

/// Traces end to end, each forwards or backwards (`reversed`), as one: a cell's side made of
/// more than one curve. Each starts where the one before ended.
pub fn join_traces(parts: &[(PointsRef<'_>, bool)]) -> CurvePoints {
    let mut xyz: Vec<f32> = Vec::new();
    for (trace, reversed) in parts {
        let n = trace.count();
        for k in 0..n {
            let i = if *reversed { n - 1 - k } else { k };
            if !xyz.is_empty() && k == 0 {
                continue;
            }
            xyz.extend_from_slice(&trace.points[i * 3..i * 3 + 3]);
        }
    }
    CurvePoints::from_polyline(xyz)
}

/// A fitted curve as points along it, `per_span` a span, with their arc length: to fit
/// another curve to.
pub fn curve_samples(fit: &CurveFit, per_span: usize) -> CurvePoints {
    let count = fit.spans * per_span + 1;
    let mut points = Vec::with_capacity(count * 3);
    for i in 0..count {
        let q = curve_point(&fit.poles, fit.spans, if count > 1 { i as f64 / (count - 1) as f64 } else { 0.0 });
        points.extend_from_slice(&[q.x as f32, q.y as f32, q.z as f32]);
    }
    CurvePoints::from_polyline(points)
}

/// A curve's poles in the other order: the same curve run backwards.
pub fn reversed_poles(poles: &[f64]) -> Vec<f64> {
    poles.chunks_exact(3).rev().flatten().copied().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arc(r: f64, deg: f64, count: usize) -> CurvePoints {
        let mut pts = Vec::new();
        for i in 0..count {
            let a = deg.to_radians() * i as f64 / (count - 1) as f64;
            pts.extend_from_slice(&[(r * a.cos()) as f32, (r * a.sin()) as f32, 0.0]);
        }
        CurvePoints::from_polyline(pts)
    }

    #[test]
    fn fits_an_arc_through_its_ends() {
        let p = arc(20.0, 270.0, 400);
        let fit = fit_curve(p.as_points(), 8, &CurveEnds::default(), 0.0);
        assert_eq!(fit.spans, 8);
        assert_eq!(fit.poles.len(), 11 * 3);
        // Through the ends exactly.
        assert!((curve_point(&fit.poles, 8, 0.0) - p.as_points().point(0)).length() < 1e-6);
        assert!((curve_point(&fit.poles, 8, 1.0) - p.as_points().point(399)).length() < 1e-6);
        // Within a few microns of a 20 mm arc.
        assert!(fit.max < 0.03, "max {}", fit.max);
        let t = end_tangent(&fit, 0);
        assert!((t - DVec3::Y).length() < 0.05);
        let l = leaving_direction(p.as_points(), 1, 0.5);
        assert!((l - DVec3::new(-1.0, 0.0, 0.0)).length() < 0.05, "{l}");
    }

    #[test]
    fn holds_a_given_end_tangent() {
        let p = arc(20.0, 90.0, 100);
        let along = DVec3::new(0.2, 1.0, 0.0).normalize();
        let fit = fit_curve(p.as_points(), 4, &CurveEnds { start_tangent: Some(along), end_tangent: None }, 0.0);
        assert!((end_tangent(&fit, 0) - along).length() < 1e-9);
        // The wrong way round falls back to the free fit.
        let wrong = fit_curve(p.as_points(), 4, &CurveEnds { start_tangent: Some(-along), end_tangent: None }, 0.0);
        let free = fit_curve(p.as_points(), 4, &CurveEnds::default(), 0.0);
        assert_eq!(wrong, free);
    }

    #[test]
    fn auto_spans_reads_the_noise() {
        // A straight line with no noise: the floor rules, length / (8 · 0.001) past 20.
        let p = CurvePoints::from_polyline((0..50).flat_map(|i| [i as f32, 0.0, 0.0]).collect());
        assert_eq!(auto_spans(p.as_points()), SPANS_MAX);
        let short = CurvePoints::from_polyline(vec![0.0, 0.0, 0.0, 0.005, 0.0, 0.0]);
        assert_eq!(auto_spans(short.as_points()), SPANS_MIN);
        let joined = join_traces(&[(p.as_points(), false), (p.as_points(), true)]);
        assert_eq!(joined.points.len() / 3, 99);
        assert!((joined.length - 98.0).abs() < 1e-9);
        assert_eq!(reversed_poles(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), vec![4.0, 5.0, 6.0, 1.0, 2.0, 3.0]);
    }
}
