// SPDX-License-Identifier: AGPL-3.0-only
//! The sketch's spline: a smooth curve through the fit points it was drawn by. Ported from
//! ScanRuler (`plugins/reverse/core/sketchSpline.ts`; the spline parts of
//! `plugins/reverse/core/sketch.ts` — `SketchSpline`, `splinePointIndex`, `splineAnchor`,
//! `anchorPoint`, `anchorsOf`, `endsOf`, `nearestOnEntity`, `entityPolyline`, `splineJoint`;
//! and `liftSpline` / `liftPiece` of `plugins/reverse/core/profiles.ts` with the `LoopPiece`
//! spline of `kernelTypes.ts`), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! It is the interpolating spline the 2D workspace draws (`flat_spline`): cubic Hermite
//! between the points on chord-length parameters, C2 through every point, with natural ends,
//! or an end held to a direction where a tangent relation holds it. A closed one wraps round,
//! smooth at its first point too.
//!
//! Along the curve a place is one parameter t: segment k runs from t = k to t = k + 1, so an
//! open curve of n points runs 0 to n − 1 and a closed one 0 to n, and on round again past n.
//! The kernel is handed the curve itself, not samples of it: a stretch as cubic Béziers on
//! their chord-length spans, joined into one B-spline with simple knots — exact, the curve
//! being C2 at every inner point, and so one smooth edge to offset, shell and round against
//! rather than a chain of patches.

use std::ops::{Add, Mul};

use glam::{DVec2, DVec3};
use serde::{Deserialize, Serialize};

use crate::flat_spline::{eval_spline, eval_spline_derivative, fit_spline_points, FlatSplineFit, HANDLE_REACH};
use crate::frame::SectionFrame;
use crate::sketch::{AnchorAt, Constraint, EntityId, Rule};

/// The samples per segment the nearest-point search starts from by default.
pub const NEAREST_STEPS: usize = 16;

/// What a spline is drawn by: the sketch piece's own fields. `pts` are the fit points in
/// order, the curve smooth through every one and back round to the first when `closed`.
/// `start_dir` and `end_dir` are the directions the curve leaves its first point and arrives
/// at its last, radians — set while a tangent relation holds that end, natural otherwise (the
/// curve runs on straight past it); the solver finds them.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SplineShape {
    pub pts: Vec<DVec2>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub closed: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start_dir: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end_dir: Option<f64>,
}

impl SplineShape {
    /// An open spline through `pts`, its ends natural.
    pub fn open(pts: Vec<DVec2>) -> Self {
        Self { pts, ..Self::default() }
    }

    /// A spline through `pts` and back round to the first.
    pub fn closed(pts: Vec<DVec2>) -> Self {
        Self { pts, closed: true, ..Self::default() }
    }
}

/// A cubic Bézier: its four poles.
pub type Bezier = [DVec2; 4];

/// A cubic B-spline: its poles, its distinct knots and their multiplicities — clamped, four
/// at either end — the form OCCT and STEP take. In the sheet the poles are `DVec2`; lifted
/// into the part (`lift_bspline`) they are `DVec3`, what a profile loop's spline piece is.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BSpline<P = DVec2> {
    pub poles: Vec<P>,
    pub knots: Vec<f64>,
    pub mults: Vec<usize>,
}

/// A pole of a B-spline in any dimension: what the Bézier chain is joined over.
pub trait Pole: Copy + Add<Output = Self> + Mul<f64, Output = Self> {
    const DIM: usize;
    fn coord(self, i: usize) -> f64;
}

impl Pole for DVec2 {
    const DIM: usize = 2;
    fn coord(self, i: usize) -> f64 {
        if i == 0 {
            self.x
        } else {
            self.y
        }
    }
}

impl Pole for DVec3 {
    const DIM: usize = 3;
    fn coord(self, i: usize) -> f64 {
        match i {
            0 => self.x,
            1 => self.y,
            _ => self.z,
        }
    }
}

/// One sample of the curve: its parameter and where it is.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SplineSample {
    pub t: f64,
    pub p: DVec2,
}

/// The point of a curve nearest a spot: its parameter, where it is and how far.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NearestOnSpline {
    pub t: f64,
    pub point: DVec2,
    pub dist: f64,
}

/// A stretch of the curve as Béziers in the direction asked, each with the span of
/// chord-length parameter it covers.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StretchBeziers {
    pub pieces: Vec<Bezier>,
    pub spans: Vec<f64>,
}

/// A spline's soft data: the slice points along it and their scatter about it, mm.
#[derive(Debug, Clone, PartialEq)]
pub struct SplineSoftPoints {
    pub points: Vec<DVec2>,
    pub scatter: f64,
}

/// How many segments the curve has: one fewer than its points, open; as many, closed.
pub fn spline_segments_of(e: &SplineShape) -> usize {
    if is_closed(e) {
        e.pts.len()
    } else {
        e.pts.len().saturating_sub(1)
    }
}

/// Whether the curve closes: asked so, with three points at least.
pub fn is_closed(e: &SplineShape) -> bool {
    e.closed && e.pts.len() >= 3
}

fn segments_of_fit(fit: &FlatSplineFit) -> usize {
    if fit.closed {
        fit.points.len()
    } else {
        fit.points.len().saturating_sub(1)
    }
}

fn hypot(d: DVec2) -> f64 {
    d.x.hypot(d.y)
}

/// The curve through the piece's points: tangents solved, an end held to its direction where
/// one is set. Two points on one spot — a solve passing through it, a hand that clicked
/// twice — still give a curve, its tangents read off the neighbours, rather than none.
pub fn spline_curve(e: &SplineShape) -> FlatSplineFit {
    let pts = &e.pts;
    let n = pts.len();
    let closed = is_closed(e);
    let chord = |i: usize| hypot(pts[(i + 1) % n] - pts[i]);
    let mut handles: Vec<Option<DVec2>> = vec![None; n];
    if !closed && n >= 2 {
        // A handle reaches a third of the chord beside it, and is then the curve's
        // derivative of unit length — the speed of a chord-length curve.
        if let Some(dir) = e.start_dir {
            let r = HANDLE_REACH * chord(0);
            handles[0] = Some(DVec2::new(dir.cos() * r, dir.sin() * r));
        }
        if let Some(dir) = e.end_dir {
            let r = HANDLE_REACH * chord(n - 2);
            handles[n - 1] = Some(DVec2::new(dir.cos() * r, dir.sin() * r));
        }
    }
    if n >= 2 {
        if let Ok(fit) = fit_spline_points(pts, &handles, closed) {
            return fit;
        }
        // Below: a curve that still evaluates.
    }
    let tangents = (0..n)
        .map(|i| {
            let before = pts[if closed { (i + n - 1) % n } else { i.saturating_sub(1) }];
            let after = pts[if closed { (i + 1) % n } else { (i + 1).min(n - 1) }];
            let d = after - before;
            let l = hypot(d);
            if l > 1e-12 {
                DVec2::new(d.x / l, d.y / l)
            } else {
                DVec2::X
            }
        })
        .collect();
    let mut length = 0.0;
    for i in 0..spline_segments_of(e) {
        length += chord(i);
    }
    FlatSplineFit { points: pts.clone(), tangents, fixed: vec![false; n], closed, length, sigma: 0.0, used_points: n }
}

/// The segment and its own parameter at t — a closed curve's t taken round, an open one's
/// held to its ends.
fn locate(fit: &FlatSplineFit, t: f64) -> (usize, f64) {
    let s = segments_of_fit(fit);
    if s == 0 {
        return (0, 0.0);
    }
    let sf = s as f64;
    let w = if fit.closed { ((t % sf) + sf) % sf } else { t.max(0.0).min(sf) };
    let k = w.floor().min(sf - 1.0);
    (k as usize, w - k)
}

/// The point at t.
pub fn spline_at(fit: &FlatSplineFit, t: f64) -> DVec2 {
    if segments_of_fit(fit) == 0 {
        return fit.points.first().copied().unwrap_or(DVec2::ZERO);
    }
    let (k, u) = locate(fit, t);
    eval_spline(fit, k, u)
}

/// The derivative at t, per unit of t — the way the curve runs on there.
pub fn spline_deriv_at(fit: &FlatSplineFit, t: f64) -> DVec2 {
    if segments_of_fit(fit) == 0 {
        return DVec2::X;
    }
    let (k, u) = locate(fit, t);
    eval_spline_derivative(fit, k, u)
}

/// The unit direction the curve runs at t, or +X where it stands still.
pub fn spline_dir_at(fit: &FlatSplineFit, t: f64) -> DVec2 {
    let d = spline_deriv_at(fit, t);
    let l = hypot(d);
    if l > 1e-12 {
        DVec2::new(d.x / l, d.y / l)
    } else {
        DVec2::X
    }
}

/// The parameter at the curve's end: 0 at its start, the segment count at its end.
pub fn spline_end(fit: &FlatSplineFit) -> f64 {
    segments_of_fit(fit) as f64
}

/// The curve sampled, `per_segment` steps a segment (16 by default), with the parameter of
/// every sample — a closed one back at its start.
pub fn spline_samples(fit: &FlatSplineFit, per_segment: usize) -> Vec<SplineSample> {
    let s = segments_of_fit(fit);
    if s == 0 {
        return vec![SplineSample { t: 0.0, p: fit.points.first().copied().unwrap_or(DVec2::ZERO) }];
    }
    let per = per_segment as f64;
    let mut out = Vec::with_capacity(s * per_segment + 1);
    for k in 0..s {
        for i in (if k == 0 { 0 } else { 1 })..=per_segment {
            out.push(SplineSample { t: k as f64 + i as f64 / per, p: eval_spline(fit, k, i as f64 / per) });
        }
    }
    out
}

/// The point of the curve nearest `p`: its parameter, where it is and how far. The samples
/// (`per_segment` a segment, 16 by default) find the stretch, a golden-section search closes
/// on it — to a hundred-millionth of a segment.
pub fn nearest_on_spline(fit: &FlatSplineFit, p: DVec2, per_segment: usize) -> NearestOnSpline {
    let s = segments_of_fit(fit);
    let d2 = |t: f64| {
        let q = spline_at(fit, t);
        let (dx, dy) = (q.x - p.x, q.y - p.y);
        dx * dx + dy * dy
    };
    if s == 0 {
        return NearestOnSpline { t: 0.0, point: spline_at(fit, 0.0), dist: d2(0.0).sqrt() };
    }
    let mut best = 0.0;
    let mut best_d = f64::INFINITY;
    for sample in spline_samples(fit, per_segment) {
        let d = d2(sample.t);
        if d < best_d {
            best_d = d;
            best = sample.t;
        }
    }
    let step = 1.0 / per_segment as f64;
    let mut lo = best - step;
    let mut hi = best + step;
    let sf = s as f64;
    if !fit.closed {
        lo = lo.max(0.0);
        hi = hi.min(sf);
    }
    let g = (5f64.sqrt() - 1.0) / 2.0;
    let mut x1 = hi - g * (hi - lo);
    let mut x2 = lo + g * (hi - lo);
    let mut f1 = d2(x1);
    let mut f2 = d2(x2);
    for _ in 0..40 {
        if f1 < f2 {
            hi = x2;
            x2 = x1;
            f2 = f1;
            x1 = hi - g * (hi - lo);
            f1 = d2(x1);
        } else {
            lo = x1;
            x1 = x2;
            f1 = f2;
            x2 = lo + g * (hi - lo);
            f2 = d2(x2);
        }
    }
    let mut t = (lo + hi) / 2.0;
    // The samples' own ends are candidates too: an open curve's nearest point may be its
    // tip, which the bracket reaches only from one side.
    if d2(best) < d2(t) {
        t = best;
    }
    if fit.closed {
        t = ((t % sf) + sf) % sf;
    }
    let point = spline_at(fit, t);
    NearestOnSpline { t, point, dist: hypot(point - p) }
}

// ---- the curve as the kernel takes it ----

fn lerp(a: DVec2, b: DVec2, u: f64) -> DVec2 {
    DVec2::new(a.x + (b.x - a.x) * u, a.y + (b.y - a.y) * u)
}

/// Segment k as a Bézier, and the chord its parameter spans.
fn segment_bezier(fit: &FlatSplineFit, k: usize) -> (Bezier, f64) {
    let n = fit.points.len().max(1);
    let get = |v: &[DVec2], i: usize| v.get(i).copied().unwrap_or(DVec2::ZERO);
    let p0 = get(&fit.points, k % n);
    let p1 = get(&fit.points, (k + 1) % n);
    let h = hypot(p1 - p0);
    let m0 = get(&fit.tangents, k % n);
    let m1 = get(&fit.tangents, (k + 1) % n);
    (
        [
            p0,
            DVec2::new(p0.x + (h * m0.x) / 3.0, p0.y + (h * m0.y) / 3.0),
            DVec2::new(p1.x - (h * m1.x) / 3.0, p1.y - (h * m1.y) / 3.0),
            p1,
        ],
        h,
    )
}

/// The part of a Bézier from u0 to u1, by de Casteljau.
fn sub_bezier(b: Bezier, u0: f64, u1: f64) -> Bezier {
    let split = |c: Bezier, u: f64| -> (Bezier, Bezier) {
        let a = lerp(c[0], c[1], u);
        let m = lerp(c[1], c[2], u);
        let d = lerp(c[2], c[3], u);
        let e = lerp(a, m, u);
        let f = lerp(m, d, u);
        let g = lerp(e, f, u);
        ([c[0], a, e, g], [g, f, d, c[3]])
    };
    let mut c = b;
    if u1 < 1.0 {
        c = split(c, u1).0;
    }
    if u0 > 0.0 {
        c = split(c, u0 / u1).1;
    }
    c
}

/// The stretch of the curve from t0 to t1 as Béziers in that direction, each with the span of
/// chord-length parameter it covers. A closed curve's stretch may run on past its first point
/// (t past the segment count). Spans of no length — two points on one spot — are left out. An
/// open curve's ends hold the stretch, an endless one included; a `NaN` end gives nothing, as
/// does an endless stretch round a closed curve (where ScanRuler would never return).
pub fn spline_stretch_beziers(fit: &FlatSplineFit, t0: f64, t1: f64) -> StretchBeziers {
    let s = segments_of_fit(fit);
    let mut out = StretchBeziers::default();
    if s == 0 || t0.is_nan() || t1.is_nan() {
        return out;
    }
    let sf = s as f64;
    let mut lo = t0.min(t1);
    let mut hi = t0.max(t1);
    if !fit.closed {
        lo = lo.max(0.0);
        hi = hi.min(sf);
    }
    if !lo.is_finite() || !hi.is_finite() {
        return out;
    }
    let first = (lo + 1e-12).floor() as i64;
    let last = (hi - 1e-12).ceil() as i64 - 1;
    let si = s as i64;
    for j in first..=last {
        let jf = j as f64;
        let u0 = (lo - jf).max(0.0);
        let u1 = (hi - jf).min(1.0);
        if u1 - u0 < 1e-12 {
            continue;
        }
        let k = if fit.closed { ((j % si) + si) % si } else { j.clamp(0, si - 1) } as usize;
        let (bezier, h) = segment_bezier(fit, k);
        let span = (u1 - u0) * h;
        if !(span > 1e-12) {
            continue;
        }
        out.pieces.push(sub_bezier(bezier, u0, u1));
        out.spans.push(span);
    }
    if t0 > t1 {
        out.pieces.reverse();
        out.spans.reverse();
        for p in &mut out.pieces {
            *p = [p[3], p[2], p[1], p[0]];
        }
    }
    out
}

fn mix<P: Pole>(a: P, wa: f64, b: P, wb: f64) -> P {
    a * wa + b * wb
}

/// Béziers end to end, on the spans given, as one cubic B-spline. Where the chain is C2 —
/// every stretch of a spline is — the poles are the de Boor points behind it and the knots
/// simple: each Bézier's two inner poles lie on a leg of the de Boor polygon, a third and two
/// thirds of the way on equal spans and as the spans either side weigh them otherwise, so
/// each Bézier gives back its leg's two points. A chain that is not C2 — checked by laying the
/// Béziers again from the points found — is handed over as it is, its inner knots of
/// multiplicity three: exact still, only not smooth in its knots. `None` for an empty chain,
/// or one whose spans do not match its pieces one for one.
pub fn bezier_chain_to_bspline<P: Pole>(pieces: &[[P; 4]], spans: &[f64]) -> Option<BSpline<P>> {
    let m = pieces.len();
    if m == 0 || spans.len() != m {
        return None;
    }
    let mut knots = vec![0.0];
    for s in spans {
        let last = knots[knots.len() - 1];
        knots.push(last + s);
    }
    let span = |k: isize| if k < 0 || k as usize >= m { 0.0 } else { spans[k as usize] };
    // Leg k + 1 → k + 2 of the de Boor polygon carries Bézier k's inner poles:
    //   b1 = ((Δk + Δk+1) d1 + Δk−1 d2) / Σ,  b2 = (Δk+1 d1 + (Δk−1 + Δk) d2) / Σ.
    let weights = |k: isize| {
        let sum = span(k - 1) + span(k) + span(k + 1);
        ((span(k) + span(k + 1)) / sum, span(k - 1) / sum, span(k + 1) / sum, (span(k - 1) + span(k)) / sum)
    };
    let mut de: Vec<P> = Vec::with_capacity(m + 3);
    de.push(pieces[0][0]);
    for k in 0..m {
        let (a, b, c, d) = weights(k as isize);
        let det = a * d - b * c;
        let b1 = pieces[k][1];
        let b2 = pieces[k][2];
        de.push(mix(b1, d / det, b2, -b / det));
        if k == m - 1 {
            de.push(mix(b2, a / det, b1, -c / det));
        }
    }
    de.push(pieces[m - 1][3]);
    // Laid again from the points found, the Béziers have to come back.
    let mut size = 1.0f64;
    for p in pieces {
        for q in p {
            for i in 0..P::DIM {
                size = size.max(q.coord(i).abs());
            }
        }
    }
    let tol = 1e-9 * size;
    let mut smooth = true;
    for k in 0..m {
        if !smooth {
            break;
        }
        let ki = k as isize;
        let (a, b, c, d) = weights(ki);
        let b1 = mix(de[k + 1], a, de[k + 2], b);
        let b2 = mix(de[k + 1], c, de[k + 2], d);
        let b0 = if k == 0 {
            de[0]
        } else {
            let both = span(ki - 1) + span(ki);
            mix(pieces[k - 1][2], span(ki) / both, pieces[k][1], span(ki - 1) / both)
        };
        for i in 0..P::DIM {
            if (b1.coord(i) - pieces[k][1].coord(i)).abs() > tol || (b2.coord(i) - pieces[k][2].coord(i)).abs() > tol || (b0.coord(i) - pieces[k][0].coord(i)).abs() > tol {
                smooth = false;
            }
        }
    }
    if smooth {
        let mults = (0..knots.len()).map(|i| if i == 0 || i == m { 4 } else { 1 }).collect();
        return Some(BSpline { poles: de, knots, mults });
    }
    let mut poles = vec![pieces[0][0]];
    for p in pieces {
        poles.extend([p[1], p[2], p[3]]);
    }
    let mults = (0..knots.len()).map(|i| if i == 0 || i == m { 4 } else { 3 }).collect();
    Some(BSpline { poles, knots, mults })
}

/// The stretch of the curve from t0 to t1 as one B-spline, `None` where it has no length.
pub fn spline_stretch(fit: &FlatSplineFit, t0: f64, t1: f64) -> Option<BSpline> {
    let StretchBeziers { pieces, spans } = spline_stretch_beziers(fit, t0, t1);
    if pieces.is_empty() {
        return None;
    }
    bezier_chain_to_bspline(&pieces, &spans)
}

/// A clamped cubic B-spline's point at parameter t, by de Boor — the knots as `knots` and
/// `mults` give them. `None` for a B-spline with too few poles or knots to be one.
pub fn bspline_at<P: Pole>(b: &BSpline<P>, t: f64) -> Option<P> {
    let knots: Vec<f64> = b.mults.iter().zip(&b.knots).flat_map(|(&m, &k)| std::iter::repeat_n(k, m)).collect();
    let p = 3;
    let n = b.poles.len();
    if n < p + 1 || knots.len() < n + p + 1 {
        return None;
    }
    let mut k = p;
    while k < n - 1 && t >= knots[k + 1] {
        k += 1;
    }
    let mut d: Vec<P> = b.poles[k - p..=k].to_vec();
    for r in 1..=p {
        for j in (r..=p).rev() {
            let i = k - p + j;
            let den = knots[i + p - r + 1] - knots[i];
            let a = if den > 0.0 { (t - knots[i]) / den } else { 0.0 };
            d[j] = d[j - 1] * (1.0 - a) + d[j] * a;
        }
    }
    Some(d[p])
}

/// A B-spline sampled at `count` + 1 points from its start to its end.
pub fn bspline_samples<P: Pole>(b: &BSpline<P>, count: usize) -> Vec<P> {
    let (Some(&t0), Some(&t1)) = (b.knots.first(), b.knots.last()) else { return Vec::new() };
    (0..=count).filter_map(|i| bspline_at(b, t0 + ((t1 - t0) * i as f64) / count as f64)).collect()
}

/// A B-spline in the sheet lifted into the part through the sketch's frame. Lifted pole by
/// pole it is the curve lifted: the map is affine.
pub fn lift_bspline(frame: &SectionFrame, b: &BSpline) -> BSpline<DVec3> {
    BSpline { poles: b.poles.iter().map(|p| frame.to_world(*p)).collect(), knots: b.knots.clone(), mults: b.mults.clone() }
}

/// A whole spline of the sketch, or the stretch of it from `t0` (0 when `None`) to `t1` (its
/// end when `None`), lifted into the part as the clamped cubic B-spline a profile loop's
/// spline piece is — `None` where it has no length.
pub fn lift_spline(frame: &SectionFrame, e: &SplineShape, t0: Option<f64>, t1: Option<f64>) -> Option<BSpline<DVec3>> {
    let fit = spline_curve(e);
    let stretch = spline_stretch(&fit, t0.unwrap_or(0.0), t1.unwrap_or_else(|| spline_end(&fit)))?;
    Some(lift_bspline(frame, &stretch))
}

// ---- fitting to the slice ----

/// A spline given the slice points along it as its soft data — what the solver then pulls it
/// onto, its fit points free to move as far as its relations let them — or `None` when none
/// lie along it. They are taken within a twentieth of its length, a millimetre to five: a
/// curve drawn by hand over the slice lies that close, and a neighbouring wall further off.
pub fn fit_spline_to_points(e: &SplineShape, slice: &[DVec2]) -> Option<SplineSoftPoints> {
    let fit = spline_curve(e);
    let reach = (0.05 * fit.length).clamp(1.0, 5.0);
    let points = points_along(&fit, slice, reach);
    if points.len() < e.pts.len().max(4) {
        return None;
    }
    let scatter = spline_scatter(e, &points).max(0.01);
    Some(SplineSoftPoints { points, scatter })
}

/// The RMS of a spline's slice points off the curve, millimetres.
pub fn spline_scatter(e: &SplineShape, points: &[DVec2]) -> f64 {
    if points.is_empty() {
        return 0.0;
    }
    let fit = spline_curve(e);
    let mut sum = 0.0;
    for p in points {
        let d = nearest_on_spline(&fit, *p, 8).dist;
        sum += d * d;
    }
    (sum / points.len() as f64).sqrt()
}

/// Where a spline of `count` fit points starts along a chain of slice points: evenly spaced
/// by length along it, end to end on an open chain, round it on a closed one — the solver
/// then settles the curve onto the points between. `None` for a chain too short to space
/// them on.
pub fn fit_points_along(chain: &[DVec2], count: usize, closed: bool) -> Option<Vec<DVec2>> {
    let mut pts: Vec<DVec2> = if closed && chain.len() > 1 && hypot(chain[0] - chain[chain.len() - 1]) < 1e-9 { chain[..chain.len() - 1].to_vec() } else { chain.to_vec() };
    if closed {
        let first = *pts.first()?;
        pts.push(first);
    }
    let mut run = vec![0.0];
    for i in 1..pts.len() {
        run.push(run[i - 1] + hypot(pts[i] - pts[i - 1]));
    }
    let total = run[run.len() - 1];
    if !(total > 1e-9) {
        return None;
    }
    let n = count.max(if closed { 3 } else { 2 });
    let steps = if closed { n } else { n - 1 };
    let mut out = Vec::with_capacity(n);
    let mut j = 0;
    for k in 0..n {
        let at = (total * k as f64) / steps as f64;
        while j < run.len() - 2 && run[j + 1] < at {
            j += 1;
        }
        let span = run[j + 1] - run[j];
        let u = if span > 0.0 { (at - run[j]) / span } else { 0.0 };
        out.push(DVec2::new(pts[j].x + (pts[j + 1].x - pts[j].x) * u, pts[j].y + (pts[j + 1].y - pts[j].y) * u));
    }
    Some(out)
}

/// The slice points that lie along the curve: within `reach` of it, and beside it rather than
/// past an open curve's tips — what a spline fitted to the slice is pulled to.
pub fn points_along(fit: &FlatSplineFit, points: &[DVec2], reach: f64) -> Vec<DVec2> {
    let s = segments_of_fit(fit) as f64;
    let (mut x0, mut y0, mut x1, mut y1) = (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    for p in &fit.points {
        x0 = x0.min(p.x);
        y0 = y0.min(p.y);
        x1 = x1.max(p.x);
        y1 = y1.max(p.y);
    }
    let pad = reach + 0.5 * fit.length;
    let (x0, y0, x1, y1) = (x0 - pad, y0 - pad, x1 + pad, y1 + pad);
    let mut out = Vec::new();
    for &p in points {
        if p.x < x0 || p.x > x1 || p.y < y0 || p.y > y1 {
            continue;
        }
        let near = nearest_on_spline(fit, p, 8);
        if near.dist > reach {
            continue;
        }
        if !fit.closed && (near.t < 1e-3 || near.t > s - 1e-3) {
            continue;
        }
        out.push(p);
    }
    out
}

// ---- the spline as a piece of the sketch ----

/// Which of a spline's fit points an end anchor names: `A` the first of an open curve, `B`
/// its last; a closed curve has no ends, and `Center` names no fit point.
pub fn spline_point_index(e: &SplineShape, at: AnchorAt) -> Option<usize> {
    let n = e.pts.len();
    if n == 0 || is_closed(e) {
        return None;
    }
    match at {
        AnchorAt::A => Some(0),
        AnchorAt::B => Some(n - 1),
        AnchorAt::Center | AnchorAt::P(_) => None,
    }
}

/// Which fit point the anchor `p{i}` names: any of a closed curve's, an open one's between its
/// ends (its first and last are named `A` and `B`); `None` for one it does not have.
pub fn spline_fit_point_index(e: &SplineShape, i: usize) -> Option<usize> {
    let n = e.pts.len();
    if i >= n {
        return None;
    }
    (is_closed(e) || (i > 0 && i < n - 1)).then_some(i)
}

/// The end anchor fit point `i` is named by — `A` for an open curve's first, `B` for its
/// last — or `None` where it is named by its place, `p{i}`.
pub fn spline_anchor(e: &SplineShape, i: usize) -> Option<AnchorAt> {
    if is_closed(e) {
        return None;
    }
    if i == 0 {
        Some(AnchorAt::A)
    } else if i + 1 == e.pts.len() {
        Some(AnchorAt::B)
    } else {
        None
    }
}

/// Where a spline's anchor lies: the fit point it names (from `spline_point_index` or
/// `spline_fit_point_index`), or — a spline has no centre — its middle for what asks for one.
pub fn spline_anchor_point(e: &SplineShape, index: Option<usize>) -> DVec2 {
    if let Some(p) = index.and_then(|i| e.pts.get(i)) {
        return *p;
    }
    let fit = spline_curve(e);
    spline_at(&fit, spline_end(&fit) / 2.0)
}

/// The fit points a spline offers as anchors, in the order the sketch lists them: an open
/// curve's ends first (`A`, `B`), then the points between (`p1`, `p2`, …); a closed curve's
/// from `p0`.
pub fn spline_anchor_order(e: &SplineShape) -> Vec<usize> {
    let n = e.pts.len();
    if is_closed(e) {
        return (0..n).collect();
    }
    if n == 0 {
        return Vec::new();
    }
    let mut out = vec![0, n - 1];
    out.extend(1..n.saturating_sub(1));
    out
}

/// Whether a spline has ends where one piece of a chain meets the next: an open one's two.
pub fn spline_has_ends(e: &SplineShape) -> bool {
    !is_closed(e)
}

/// The point on a spline closest to `p`, and its distance — for picking.
pub fn nearest_on_spline_piece(e: &SplineShape, p: DVec2) -> (DVec2, f64) {
    let near = nearest_on_spline(&spline_curve(e), p, NEAREST_STEPS);
    (near.point, near.dist)
}

/// The polyline a spline draws as, for `steps` segments per full turn of arc (96 by default
/// for the other pieces): a sixth of that per segment, four at least.
pub fn spline_piece_polyline(e: &SplineShape, steps: usize) -> Vec<DVec2> {
    let per = ((steps as f64 / 6.0) + 0.5).floor().max(4.0) as usize;
    spline_samples(&spline_curve(e), per).into_iter().map(|s| s.p).collect()
}

/// A copy of a spline as a pattern lays it (`patternPieces`' spline case): every fit point put
/// through `place`, and the end directions a tangent holds turned by `turn`, the angle `place`
/// turns the sheet by. An end nothing holds stays natural; a closed curve stays closed.
pub fn spline_placed(e: &SplineShape, place: impl Fn(DVec2) -> DVec2, turn: f64) -> SplineShape {
    SplineShape { pts: e.pts.iter().map(|p| place(*p)).collect(), closed: e.closed, start_dir: e.start_dir.map(|d| d + turn), end_dir: e.end_dir.map(|d| d + turn) }
}

/// Where a coincidence joins an end of a spline to another piece.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplineJoint {
    /// The spline's end there: `A` or `B`.
    pub end: AnchorAt,
    /// The other piece's anchor there.
    pub at: AnchorAt,
}

/// Where a coincidence joins an end of a spline to another piece: the spline's end and the
/// other's anchor, or `None`. A spline is held tangent there and nowhere else — along its
/// length it has no one direction to hold.
pub fn spline_joint(constraints: &[Constraint], spline: EntityId, other: EntityId) -> Option<SplineJoint> {
    for c in constraints {
        let Rule::Coincident { p, q } = c.rule else { continue };
        for (mine, theirs) in [(p, q), (q, p)] {
            if mine.entity == spline && theirs.entity == other && matches!(mine.at, AnchorAt::A | AnchorAt::B) && theirs.at != AnchorAt::Center {
                return Some(SplineJoint { end: mine.at, at: theirs.at });
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sketch::Anchor;
    use std::f64::consts::PI;

    fn v(x: f64, y: f64) -> DVec2 {
        DVec2::new(x, y)
    }

    fn near(a: DVec2, b: DVec2) -> f64 {
        (a.x - b.x).hypot(a.y - b.y)
    }

    /// A closed spline's points: `n` round a circle of radius `r`, the first at `from`
    /// radians.
    fn ring(n: usize, r: f64, from: f64) -> Vec<DVec2> {
        (0..n).map(|i| v(r * (from + (2.0 * PI * i as f64) / n as f64).cos(), r * (from + (2.0 * PI * i as f64) / n as f64).sin())).collect()
    }

    fn open() -> FlatSplineFit {
        spline_curve(&SplineShape::open(vec![v(0.0, 0.0), v(10.0, 8.0), v(25.0, 6.0), v(32.0, -4.0), v(45.0, 2.0)]))
    }

    #[test]
    fn a_stretch_is_one_cubic_bspline_with_simple_knots_on_the_curve_to_a_nanometre() {
        let open = open();
        let b = spline_stretch(&open, 0.3, 3.6).unwrap();
        assert_eq!(b.mults[0], 4);
        assert_eq!(b.mults[b.mults.len() - 1], 4);
        assert!(b.mults[1..b.mults.len() - 1].iter().all(|m| *m == 1));
        assert_eq!(b.poles.len(), b.knots.len() + 2);
        assert!(near(b.poles[0], spline_at(&open, 0.3)) < 1e-12);
        assert!(near(b.poles[b.poles.len() - 1], spline_at(&open, 3.6)) < 1e-12);
        for p in bspline_samples(&b, 400) {
            assert!(nearest_on_spline(&open, p, NEAREST_STEPS).dist < 1e-8);
        }
    }

    #[test]
    fn runs_the_way_it_is_asked_from_t1_back_to_t0() {
        let open = open();
        let b = spline_stretch(&open, 3.6, 0.3).unwrap();
        assert!(near(b.poles[0], spline_at(&open, 3.6)) < 1e-12);
        assert!(near(bspline_at(&b, b.knots[b.knots.len() - 1]).unwrap(), spline_at(&open, 0.3)) < 1e-8);
    }

    #[test]
    fn a_closed_splines_stretch_runs_on_over_its_first_point() {
        let closed = spline_curve(&SplineShape::closed(ring(5, 20.0, 0.0)));
        let b = spline_stretch(&closed, 4.5, 5.5).unwrap();
        assert!(b.mults[1..b.mults.len() - 1].iter().all(|m| *m == 1));
        assert!(near(b.poles[b.poles.len() - 1], spline_at(&closed, 0.5)) < 1e-12);
        for p in bspline_samples(&b, 100) {
            assert!(nearest_on_spline(&closed, p, NEAREST_STEPS).dist < 1e-8);
        }
    }

    // Beyond the ScanRuler tests: the lift into the part and the chain that is not C2.

    #[test]
    fn a_spline_lifts_pole_for_pole_through_its_frame() {
        let frame = SectionFrame { origin: DVec3::new(1.0, 2.0, 3.0), normal: DVec3::X, u: DVec3::Y, v: DVec3::Z };
        let e = SplineShape::open(vec![v(0.0, 0.0), v(10.0, 8.0), v(25.0, 6.0)]);
        let lifted = lift_spline(&frame, &e, None, None).unwrap();
        let flat = spline_stretch(&spline_curve(&e), 0.0, 2.0).unwrap();
        assert_eq!(lifted.knots, flat.knots);
        assert_eq!(lifted.mults, flat.mults);
        for (p, q) in lifted.poles.iter().zip(&flat.poles) {
            assert!((*p - DVec3::new(1.0, 2.0 + q.x, 3.0 + q.y)).length() < 1e-12);
        }
        let end = bspline_at(&lifted, lifted.knots[lifted.knots.len() - 1]).unwrap();
        assert!((end - DVec3::new(1.0, 27.0, 9.0)).length() < 1e-9);
    }

    #[test]
    fn a_chain_with_a_kink_keeps_its_bezier_poles_on_triple_knots() {
        let a = [v(0.0, 0.0), v(1.0, 0.0), v(2.0, 0.0), v(3.0, 0.0)];
        let b = [v(3.0, 0.0), v(3.0, 1.0), v(3.0, 2.0), v(3.0, 3.0)];
        let s = bezier_chain_to_bspline(&[a, b], &[3.0, 3.0]).unwrap();
        assert_eq!(s.mults, vec![4, 3, 4]);
        assert_eq!(s.poles.len(), 7);
        let corner = bspline_at(&s, 3.0).unwrap();
        assert!(near(corner, v(3.0, 0.0)) < 1e-12);
        assert!(bezier_chain_to_bspline::<DVec2>(&[], &[]).is_none());
    }

    #[test]
    fn an_open_curves_ends_hold_a_stretch_asked_past_them_even_an_endless_one() {
        let open = open();
        let whole = spline_stretch(&open, 0.0, spline_end(&open)).unwrap();
        assert_eq!(spline_stretch(&open, f64::NEG_INFINITY, f64::INFINITY), Some(whole.clone()));
        assert_eq!(spline_stretch(&open, -2.0, 9.0), Some(whole));
        let back = spline_stretch(&open, f64::INFINITY, 0.3).unwrap();
        assert!(near(back.poles[0], spline_at(&open, spline_end(&open))) < 1e-12);
        assert!(spline_stretch(&open, f64::NAN, 2.0).is_none());
        let closed = spline_curve(&SplineShape::closed(ring(5, 20.0, 0.0)));
        assert!(spline_stretch(&closed, 0.0, f64::INFINITY).is_none());
    }

    #[test]
    fn a_placed_copy_turns_its_held_ends_with_its_points() {
        let e = SplineShape { pts: vec![v(0.0, 0.0), v(10.0, 8.0), v(25.0, 6.0)], closed: false, start_dir: Some(0.4), end_dir: None };
        let turn = 0.7f64;
        let (c, s) = (turn.cos(), turn.sin());
        let place = |p: DVec2| v(5.0 + p.x * c - p.y * s, -3.0 + p.x * s + p.y * c);
        let copy = spline_placed(&e, place, turn);
        assert_eq!(copy.start_dir, Some(0.4 + turn));
        assert_eq!(copy.end_dir, None);
        let (a, b) = (spline_curve(&e), spline_curve(&copy));
        for i in 0..=20 {
            let t = i as f64 / 10.0;
            assert!(near(place(spline_at(&a, t)), spline_at(&b, t)) < 1e-9);
        }
    }

    #[test]
    fn a_fit_point_is_named_by_its_end_or_its_place() {
        let e = SplineShape::open(vec![v(0.0, 0.0), v(10.0, 5.0), v(20.0, 0.0)]);
        assert_eq!(spline_anchor_order(&e), vec![0, 2, 1]);
        assert_eq!(spline_anchor(&e, 0), Some(AnchorAt::A));
        assert_eq!(spline_anchor(&e, 2), Some(AnchorAt::B));
        assert_eq!(spline_anchor(&e, 1), None);
        assert_eq!(spline_point_index(&e, AnchorAt::B), Some(2));
        assert_eq!(spline_fit_point_index(&e, 1), Some(1));
        assert_eq!(spline_fit_point_index(&e, 2), None);
        assert_eq!(spline_anchor_point(&e, spline_fit_point_index(&e, 1)), v(10.0, 5.0));
        let ring = SplineShape::closed(ring(4, 10.0, 0.0));
        assert_eq!(spline_point_index(&ring, AnchorAt::A), None);
        assert_eq!(spline_fit_point_index(&ring, 0), Some(0));
        assert!(!spline_has_ends(&ring));
        let constraints = vec![Constraint {
            id: 9,
            rule: Rule::Coincident { p: Anchor::new(4, AnchorAt::B), q: Anchor::new(7, AnchorAt::A) },
            inferred: false,
            expr: None,
            place: None,
        }];
        assert_eq!(spline_joint(&constraints, 7, 4), Some(SplineJoint { end: AnchorAt::A, at: AnchorAt::B }));
        assert_eq!(spline_joint(&constraints, 4, 7), Some(SplineJoint { end: AnchorAt::B, at: AnchorAt::A }));
        assert_eq!(spline_joint(&constraints, 7, 5), None);
    }
}
