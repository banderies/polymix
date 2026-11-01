// SPDX-License-Identifier: AGPL-3.0-only
//! The fit-point spline: the curve a CAD sketch draws through the points clicked, with a
//! tangent handle at every point to bend it by. Ported from ScanRuler
//! (`src/core/flat/spline.ts`, the `FlatSplineFit` type of `src/core/flat/types.ts`),
//! © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! Cubic Hermite segments between consecutive points, parameterised by chord length, and the
//! tangents solved so the curve is C2 wherever no handle has been set — the same
//! interpolating spline a sketcher's "fit point spline" is — with natural ends on an open
//! curve (no curvature at the tips) and a periodic wrap on a closed one. A handle that has
//! been dragged fixes the derivative at its point; the rest of the tangents are re-solved
//! around it, so the curve stays as smooth as it can be everywhere the hand has not been.
//!
//! The handle the stage draws is the derivative scaled to a third of the mean chord to the
//! neighbours: an automatic handle then looks like a Bézier control handle would, and a
//! dragged one is what the curve leaves the point along, at the pull the drag gave it.
//! Everything here is in document units; the store keeps handles in image pixels beside the
//! picks, like every source, and converts on the way in.

use glam::DVec2;
use serde::{Deserialize, Serialize};

use crate::fit2d::{solve_linear, FitError};

/// How far a handle reaches, as a fraction of the mean chord to the neighbours.
pub const HANDLE_REACH: f64 = 1.0 / 3.0;

/// The samples per segment the stage draws and hit-tests the curve with.
pub const POLYLINE_STEPS: usize = 24;

/// A free curve through fit points — the profile of a cam, a fillet that is no arc, the
/// outline of a lever. Cubic Hermite between consecutive points, parameterised by chord
/// length, so a segment from `points[k]` to `points[k + 1]` runs at the derivatives
/// `tangents[k]` and `tangents[k + 1]` per unit of parameter. The tangents are solved for C2
/// continuity wherever nothing set them by hand — `fixed[i]` says which were — with natural
/// ends on an open curve and a periodic wrap on a closed one, whose last segment returns to
/// `points[0]`. `length` is the arc length of the whole curve. `sigma` and `used_points` are
/// what every flat fit reports: picked geometry has no residuals and carries a zero σ.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FlatSplineFit {
    pub points: Vec<DVec2>,
    pub tangents: Vec<DVec2>,
    pub fixed: Vec<bool>,
    pub closed: bool,
    pub length: f64,
    pub sigma: f64,
    pub used_points: usize,
}

/// Which end of a tangent handle is meant: `A` is the end the curve leaves the point along,
/// `B` the one it arrives from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HandleEnd {
    A,
    B,
}

/// One handle as drawn: the point it belongs to, the end the curve leaves along (`a`) and the
/// one it arrives from (`b`), and whether it was set by hand.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SplineHandle {
    pub at: DVec2,
    pub a: DVec2,
    pub b: DVec2,
    pub fixed: bool,
}

/// The curve as cubic Béziers — the form SVG, STEP and CAD kernels all take exactly:
/// 3s + 1 poles for s segments, segment k running through poles 3k to 3k + 3, and the
/// chord-length parameter each segment starts at, plus the end — s + 1 knots.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SplineBezierForm {
    pub poles: Vec<DVec2>,
    pub knots: Vec<f64>,
}

/// Which segment of the curve passes nearest a spot, and how far away it runs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NearestSegment {
    pub segment: usize,
    pub dist: f64,
}

/// How many segments a curve through `point_count` points has: one fewer, open; as many,
/// closed.
pub fn spline_segment_count(point_count: usize, closed: bool) -> usize {
    if closed {
        point_count
    } else {
        point_count.saturating_sub(1)
    }
}

/// A point or tangent by index, or zero where a stored fit is short of it.
fn at(v: &[DVec2], i: usize) -> DVec2 {
    v.get(i).copied().unwrap_or(DVec2::ZERO)
}

/// Chord length of segment k.
fn chords(points: &[DVec2], closed: bool) -> Vec<f64> {
    let n = points.len();
    (0..spline_segment_count(n, closed))
        .map(|k| {
            let a = points[k];
            let b = points[(k + 1) % n];
            (b.x - a.x).hypot(b.y - a.y)
        })
        .collect()
}

/// The handle reach at point i: a third of the mean of its chords — the one chord at the tip
/// of an open curve.
fn reaches(points: &[DVec2], closed: bool, h: &[f64]) -> Vec<f64> {
    let n = points.len();
    (0..n)
        .map(|i| {
            let before = if closed {
                h.get((i + n - 1) % n).copied()
            } else if i > 0 {
                h.get(i - 1).copied()
            } else {
                None
            };
            let after = if closed {
                h.get(i % n).copied()
            } else if i + 1 < n {
                h.get(i).copied()
            } else {
                None
            };
            let mean = match (before, after) {
                (Some(b), Some(a)) => (b + a) / 2.0,
                (Some(b), None) => b,
                (None, Some(a)) => a,
                (None, None) => 1.0,
            };
            mean * HANDLE_REACH
        })
        .collect()
}

/// Solve the spline through `points` with the handles that were set. A handle is the offset
/// (document units) from its point to the end the curve leaves along, or `None` to let the
/// tangent there be solved; a list shorter than the points leaves the rest free. Fails, with
/// a user-facing message, when the points cannot carry a curve.
pub fn fit_spline_points(points: &[DVec2], handles: &[Option<DVec2>], closed: bool) -> Result<FlatSplineFit, FitError> {
    let n = points.len();
    if n < 2 {
        return Err(FitError::new("A spline needs at least two points."));
    }
    if closed && n < 3 {
        return Err(FitError::new("A closed spline needs at least three points."));
    }
    let h = chords(points, closed);
    if h.iter().any(|c| !(*c > 1e-9)) {
        return Err(FitError::new("Two consecutive points coincide — move one of them apart."));
    }
    let reach = reaches(points, closed, &h);
    let handle_at = |i: usize| handles.get(i).copied().flatten();

    // One row per point, for the derivative there. A free interior point asks for equal
    // second derivatives on both sides; a free tip of an open curve for none; a fixed point
    // simply states its derivative.
    let mut a = vec![0.0; n * n];
    let mut bx = vec![0.0; n];
    let mut by = vec![0.0; n];
    let segments = spline_segment_count(n, closed);
    for i in 0..n {
        if let Some(handle) = handle_at(i) {
            a[i * n + i] = 1.0;
            bx[i] = handle.x / reach[i];
            by[i] = handle.y / reach[i];
            continue;
        }
        let h_prev = if closed {
            h[(i + n - 1) % n]
        } else if i > 0 {
            h[i - 1]
        } else {
            0.0
        };
        let h_next = if closed {
            h[i % segments]
        } else if i + 1 < n {
            h[i]
        } else {
            0.0
        };
        if !closed && i == 0 {
            let next = i + 1;
            a[i * n + i] = 2.0;
            a[i * n + next] = 1.0;
            bx[i] = (3.0 * (points[next].x - points[i].x)) / h_next;
            by[i] = (3.0 * (points[next].y - points[i].y)) / h_next;
            continue;
        }
        if !closed && i == n - 1 {
            let prev = i - 1;
            a[i * n + prev] = 1.0;
            a[i * n + i] = 2.0;
            bx[i] = (3.0 * (points[i].x - points[prev].x)) / h_prev;
            by[i] = (3.0 * (points[i].y - points[prev].y)) / h_prev;
            continue;
        }
        let prev = if closed { (i + n - 1) % n } else { i - 1 };
        let next = if closed { (i + 1) % n } else { i + 1 };
        a[i * n + prev] += 1.0 / h_prev;
        a[i * n + i] += 2.0 * (1.0 / h_prev + 1.0 / h_next);
        a[i * n + next] += 1.0 / h_next;
        bx[i] = 3.0 * ((points[i].x - points[prev].x) / (h_prev * h_prev) + (points[next].x - points[i].x) / (h_next * h_next));
        by[i] = 3.0 * ((points[i].y - points[prev].y) / (h_prev * h_prev) + (points[next].y - points[i].y) / (h_next * h_next));
    }
    // The system is diagonally dominant and small — a few dozen points at the most — so a
    // dense solve, twice, is well within a pointer move.
    let mx = solve_linear(n, &mut a.clone(), &bx);
    let my = solve_linear(n, &mut a.clone(), &by);
    let (Some(mx), Some(my)) = (mx, my) else {
        return Err(FitError::new("Couldn't lay a curve through these points."));
    };

    let mut fit = FlatSplineFit {
        points: points.to_vec(),
        tangents: (0..n).map(|i| DVec2::new(mx[i], my[i])).collect(),
        fixed: (0..n).map(|i| handle_at(i).is_some()).collect(),
        closed,
        length: 0.0,
        sigma: 0.0,
        used_points: n,
    };
    fit.length = arc_length(&fit);
    Ok(fit)
}

/// The two points and two derivatives of segment k, and its chord.
fn segment_ends(fit: &FlatSplineFit, k: usize) -> (DVec2, DVec2, DVec2, DVec2, f64) {
    let n = fit.points.len().max(1);
    let p0 = at(&fit.points, k % n);
    let p1 = at(&fit.points, (k + 1) % n);
    let h = (p1.x - p0.x).hypot(p1.y - p0.y);
    let m0 = at(&fit.tangents, k % n);
    let m1 = at(&fit.tangents, (k + 1) % n);
    (p0, p1, m0, m1, h)
}

/// The point on segment k at u in [0, 1].
pub fn eval_spline(fit: &FlatSplineFit, k: usize, u: f64) -> DVec2 {
    let (p0, p1, m0, m1, h) = segment_ends(fit, k);
    let u2 = u * u;
    let u3 = u2 * u;
    let c0 = 2.0 * u3 - 3.0 * u2 + 1.0;
    let c1 = (u3 - 2.0 * u2 + u) * h;
    let c2 = -2.0 * u3 + 3.0 * u2;
    let c3 = (u3 - u2) * h;
    DVec2::new(
        c0 * p0.x + c1 * m0.x + c2 * p1.x + c3 * m1.x,
        c0 * p0.y + c1 * m0.y + c2 * p1.y + c3 * m1.y,
    )
}

/// The derivative on segment k with respect to u.
pub fn eval_spline_derivative(fit: &FlatSplineFit, k: usize, u: f64) -> DVec2 {
    let (p0, p1, m0, m1, h) = segment_ends(fit, k);
    let u2 = u * u;
    let c0 = 6.0 * u2 - 6.0 * u;
    let c1 = (3.0 * u2 - 4.0 * u + 1.0) * h;
    let c2 = -6.0 * u2 + 6.0 * u;
    let c3 = (3.0 * u2 - 2.0 * u) * h;
    DVec2::new(
        c0 * p0.x + c1 * m0.x + c2 * p1.x + c3 * m1.x,
        c0 * p0.y + c1 * m0.y + c2 * p1.y + c3 * m1.y,
    )
}

// Five-point Gauss–Legendre on [-1, 1]; four panels of it per segment integrate the speed
// of a cubic to well past any scanner's resolution.
const GAUSS_X: [f64; 5] = [0.0, 0.5384693101056831, -0.5384693101056831, 0.906179845938664, -0.906179845938664];
const GAUSS_W: [f64; 5] = [0.5688888888888889, 0.4786286704993665, 0.4786286704993665, 0.2369268850561891, 0.2369268850561891];
const PANELS: usize = 4;

fn arc_length(fit: &FlatSplineFit) -> f64 {
    let mut total = 0.0;
    for k in 0..spline_segment_count(fit.points.len(), fit.closed) {
        for panel in 0..PANELS {
            let u0 = panel as f64 / PANELS as f64;
            let half = 1.0 / (2.0 * PANELS as f64);
            let mut sum = 0.0;
            for g in 0..GAUSS_X.len() {
                let d = eval_spline_derivative(fit, k, u0 + half * (1.0 + GAUSS_X[g]));
                sum += GAUSS_W[g] * d.x.hypot(d.y);
            }
            total += sum * half;
        }
    }
    total
}

/// The curve sampled for drawing and hit-testing: `per_segment` steps on each segment (the
/// stage uses `POLYLINE_STEPS`), the first point once, a closed curve back at its start.
pub fn spline_polyline(fit: &FlatSplineFit, per_segment: usize) -> Vec<DVec2> {
    let Some(&first) = fit.points.first() else { return Vec::new() };
    let mut out = vec![first];
    for k in 0..spline_segment_count(fit.points.len(), fit.closed) {
        for i in 1..=per_segment {
            out.push(eval_spline(fit, k, i as f64 / per_segment as f64));
        }
    }
    out
}

/// A spot on the curve to hang its label from: halfway along the samples.
pub fn spline_midpoint(fit: &FlatSplineFit) -> DVec2 {
    let pts = spline_polyline(fit, 8);
    pts.get(pts.len() / 2).copied().unwrap_or(DVec2::ZERO)
}

/// The handles as the stage draws them, one per point.
pub fn spline_handles(fit: &FlatSplineFit) -> Vec<SplineHandle> {
    let h = chords(&fit.points, fit.closed);
    let reach = reaches(&fit.points, fit.closed, &h);
    fit.points
        .iter()
        .enumerate()
        .map(|(i, &p)| {
            let m = at(&fit.tangents, i);
            let dx = m.x * reach[i];
            let dy = m.y * reach[i];
            SplineHandle { at: p, a: DVec2::new(p.x + dx, p.y + dy), b: DVec2::new(p.x - dx, p.y - dy), fixed: fit.fixed.get(i).copied().unwrap_or(false) }
        })
        .collect()
}

/// The curve as cubic Béziers on chord-length knots.
pub fn spline_bezier_form(fit: &FlatSplineFit) -> SplineBezierForm {
    let n = fit.points.len();
    let Some(&first) = fit.points.first() else { return SplineBezierForm { poles: Vec::new(), knots: vec![0.0] } };
    let mut poles = vec![first];
    let mut knots = vec![0.0];
    for k in 0..spline_segment_count(n, fit.closed) {
        let (p0, p1, m0, m1, h) = segment_ends(fit, k);
        poles.push(DVec2::new(p0.x + (h * m0.x) / 3.0, p0.y + (h * m0.y) / 3.0));
        poles.push(DVec2::new(p1.x - (h * m1.x) / 3.0, p1.y - (h * m1.y) / 3.0));
        poles.push(p1);
        let last = knots[knots.len() - 1];
        knots.push(last + h);
    }
    SplineBezierForm { poles, knots }
}

/// Which segment of the curve passes nearest `p`, and how far away it runs — what a click on
/// the curve inserts a point into: before the segment's end point, at index `segment + 1`.
/// `per_segment` is the sampling (the stage uses `POLYLINE_STEPS`).
pub fn nearest_spline_segment(fit: &FlatSplineFit, p: DVec2, per_segment: usize) -> NearestSegment {
    let pts = spline_polyline(fit, per_segment);
    let mut best = f64::INFINITY;
    let mut segment = 0;
    for i in 0..pts.len().saturating_sub(1) {
        let a = pts[i];
        let b = pts[i + 1];
        let dx = b.x - a.x;
        let dy = b.y - a.y;
        let len2 = dx * dx + dy * dy;
        let t = if len2 > 0.0 { (((p.x - a.x) * dx + (p.y - a.y) * dy) / len2).clamp(0.0, 1.0) } else { 0.0 };
        let d = (p.x - (a.x + t * dx)).hypot(p.y - (a.y + t * dy));
        if d < best {
            best = d;
            segment = i / per_segment.max(1);
        }
    }
    NearestSegment { segment, dist: best }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(x: f64, y: f64) -> DVec2 {
        DVec2::new(x, y)
    }

    /// vitest's `toBeCloseTo(expected, digits)`.
    fn close(a: f64, b: f64, digits: i32) -> bool {
        (a - b).abs() < 10f64.powi(-digits) / 2.0
    }

    fn square() -> Vec<DVec2> {
        vec![v(0.0, 0.0), v(10.0, 0.0), v(10.0, 10.0), v(0.0, 10.0)]
    }

    fn wave() -> Vec<DVec2> {
        vec![v(0.0, 0.0), v(10.0, 8.0), v(20.0, -3.0), v(35.0, 5.0)]
    }

    fn none(n: usize) -> Vec<Option<DVec2>> {
        vec![None; n]
    }

    /// Second derivative on segment k with respect to the chord-length parameter, by central
    /// difference — what C2 continuity compares.
    fn second_derivative(fit: &FlatSplineFit, k: usize, u: f64) -> DVec2 {
        let n = fit.points.len();
        let a = fit.points[k];
        let b = fit.points[(k + 1) % n];
        let h = (b.x - a.x).hypot(b.y - a.y);
        let e = 1e-4;
        let p0 = eval_spline(fit, k, u - e);
        let p1 = eval_spline(fit, k, u);
        let p2 = eval_spline(fit, k, u + e);
        v((p0.x - 2.0 * p1.x + p2.x) / (e * e * h * h), (p0.y - 2.0 * p1.y + p2.y) / (e * e * h * h))
    }

    #[test]
    fn runs_through_every_point_in_order_and_counts_its_segments() {
        let fit = fit_spline_points(&wave(), &none(4), false).unwrap();
        assert_eq!(fit.points, wave());
        assert_eq!(fit.fixed, vec![false, false, false, false]);
        assert_eq!(fit.used_points, 4);
        assert_eq!(spline_segment_count(4, false), 3);
        for k in 0..3 {
            assert_eq!(eval_spline(&fit, k, 0.0), wave()[k]);
            let end = eval_spline(&fit, k, 1.0);
            assert!(close(end.x, wave()[k + 1].x, 9));
            assert!(close(end.y, wave()[k + 1].y, 9));
        }
    }

    #[test]
    fn is_a_straight_segment_through_two_points_at_their_distance() {
        let fit = fit_spline_points(&[v(1.0, 1.0), v(4.0, 5.0)], &none(2), false).unwrap();
        assert!(close(fit.length, 5.0, 9));
        let mid = eval_spline(&fit, 0, 0.5);
        assert!(close(mid.x, 2.5, 9));
        assert!(close(mid.y, 3.0, 9));
    }

    #[test]
    fn is_c2_at_every_free_interior_point_and_has_no_curvature_at_open_ends() {
        let fit = fit_spline_points(&wave(), &none(4), false).unwrap();
        for k in [1, 2] {
            let before = second_derivative(&fit, k - 1, 1.0 - 1e-4);
            let after = second_derivative(&fit, k, 1e-4);
            assert!(close(before.x, after.x, 3));
            assert!(close(before.y, after.y, 3));
        }
        let start = second_derivative(&fit, 0, 1e-4);
        let end = second_derivative(&fit, 2, 1.0 - 1e-4);
        assert!(start.x.hypot(start.y) < 1e-2);
        assert!(end.x.hypot(end.y) < 1e-2);
    }

    #[test]
    fn closes_on_itself_with_c2_continuity_across_the_seam() {
        let fit = fit_spline_points(&square(), &none(4), true).unwrap();
        assert!(fit.closed);
        assert_eq!(spline_segment_count(4, true), 4);
        let back = eval_spline(&fit, 3, 1.0);
        assert!(close(back.x, 0.0, 9));
        assert!(close(back.y, 0.0, 9));
        // The seam is a free point like any other.
        let before = second_derivative(&fit, 3, 1.0 - 1e-4);
        let after = second_derivative(&fit, 0, 1e-4);
        assert!(close(before.x, after.x, 3));
        assert!(close(before.y, after.y, 3));
        // A square's rounded ring is symmetric: the same tangent speed at every corner, and
        // the polyline returns to its start.
        let speeds: Vec<f64> = fit.tangents.iter().map(|t| t.x.hypot(t.y)).collect();
        for s in &speeds {
            assert!(close(*s, speeds[0], 9));
        }
        let pts = spline_polyline(&fit, POLYLINE_STEPS);
        assert_eq!(pts[0], v(0.0, 0.0));
        assert!(close(pts[pts.len() - 1].x, 0.0, 9));
        assert!(close(pts[pts.len() - 1].y, 0.0, 9));
    }

    #[test]
    fn approximates_a_circle_through_points_on_it_to_a_fraction_of_a_percent() {
        // Twelve points round a Ø 20 circle, closed: the length is close to the
        // circumference and every sample within a hair of the radius.
        let pts: Vec<DVec2> = (0..12).map(|i| v(10.0 * (i as f64 * std::f64::consts::PI / 6.0).cos(), 10.0 * (i as f64 * std::f64::consts::PI / 6.0).sin())).collect();
        let fit = fit_spline_points(&pts, &none(12), true).unwrap();
        let circumference = 20.0 * std::f64::consts::PI;
        assert!((fit.length - circumference).abs() / circumference < 0.002);
        for p in spline_polyline(&fit, POLYLINE_STEPS) {
            assert!(close(p.x.hypot(p.y), 10.0, 1));
        }
        // Open, through four points 40° apart on a 120° arc — what the e2e run clicks along
        // the disc — the natural ends straighten the tips a little and the length falls a
        // hair short of the arc's, within a percent.
        let arc: Vec<DVec2> = [100.0f64, 140.0, 180.0, 220.0].iter().map(|deg| v(10.0 * deg.to_radians().cos(), 10.0 * deg.to_radians().sin())).collect();
        let open = fit_spline_points(&arc, &none(4), false).unwrap();
        let arc_length = (10.0 * 120.0 * std::f64::consts::PI) / 180.0;
        assert!(open.length < arc_length);
        assert!((open.length - arc_length).abs() / arc_length < 0.01);
    }

    #[test]
    fn honours_a_handle_set_by_hand_and_re_solves_the_rest_around_it() {
        // The middle of three points on a line, with its tangent turned straight up: the
        // curve leaves the point along the handle, at the handle's pull.
        let pts = [v(0.0, 0.0), v(10.0, 0.0), v(20.0, 0.0)];
        let straight = fit_spline_points(&pts, &none(3), false).unwrap();
        assert!(close(straight.length, 20.0, 9));
        let bent = fit_spline_points(&pts, &[None, Some(v(0.0, 4.0)), None], false).unwrap();
        assert_eq!(bent.fixed, vec![false, true, false]);
        // The mean chord at the middle point is 10, the reach a third of it: a handle of 4
        // along +Y is a derivative of 4 / (10 / 3) = 1.2 along +Y.
        assert!(close(bent.tangents[1].x, 0.0, 9));
        assert!(close(bent.tangents[1].y, 4.0 / (10.0 * HANDLE_REACH), 9));
        assert!(bent.length > straight.length);
        // The handles read back exactly what was set, symmetric about the point.
        let handles = spline_handles(&bent);
        assert_eq!(handles[1].at, v(10.0, 0.0));
        assert!(close(handles[1].a.x, 10.0, 9));
        assert!(close(handles[1].a.y, 4.0, 9));
        assert!(close(handles[1].b.x, 10.0, 9));
        assert!(close(handles[1].b.y, -4.0, 9));
        assert!(handles[1].fixed);
        assert!(!handles[0].fixed);
        // An automatic handle on the unbent line reaches a third of the way to the
        // neighbour, along it; once the middle is bent the ends re-solve and their handles
        // follow.
        let plain = spline_handles(&straight);
        assert!(close(plain[0].a.x, 10.0 * HANDLE_REACH, 9));
        assert!(close(plain[0].a.y, 0.0, 9));
        assert!(close(plain[2].b.x, 20.0 - 10.0 * HANDLE_REACH, 9));
        assert!(handles[0].a.y < 0.0);
        // The free points are still C2 with the fixed one in place.
        let before = second_derivative(&bent, 0, 1.0 - 1e-4);
        let after = second_derivative(&bent, 1, 1e-4);
        assert!((before.x - after.x).hypot(before.y - after.y) > 0.0);
    }

    #[test]
    fn refuses_what_cannot_carry_a_curve_with_a_reason() {
        let one = fit_spline_points(&[v(0.0, 0.0)], &none(1), false).unwrap_err();
        assert!(one.0.contains("at least two"));
        let two_closed = fit_spline_points(&[v(0.0, 0.0), v(5.0, 5.0)], &none(2), true).unwrap_err();
        assert!(two_closed.0.contains("closed spline needs at least three"));
        let doubled = fit_spline_points(&[v(0.0, 0.0), v(0.0, 0.0), v(5.0, 5.0)], &none(3), false).unwrap_err();
        assert!(doubled.0.contains("coincide"));
        // Closed, the seam is a segment too: the last point may not sit on the first.
        let seam = fit_spline_points(&[v(0.0, 0.0), v(5.0, 5.0), v(0.0, 0.0)], &none(3), true).unwrap_err();
        assert!(seam.0.contains("coincide"));
    }

    #[test]
    fn the_bezier_form_has_three_poles_per_segment_on_chord_length_knots_and_draws_the_same_curve() {
        let fit = fit_spline_points(&wave(), &none(4), false).unwrap();
        let SplineBezierForm { poles, knots } = spline_bezier_form(&fit);
        assert_eq!(poles.len(), 10);
        assert_eq!(knots.len(), 4);
        assert_eq!(knots[0], 0.0);
        assert!(close(knots[1], 10f64.hypot(8.0), 9));
        assert_eq!(poles[0], wave()[0]);
        assert_eq!(poles[9], wave()[3]);
        for k in 0..3 {
            let (b0, b1, b2, b3) = (poles[3 * k], poles[3 * k + 1], poles[3 * k + 2], poles[3 * k + 3]);
            for u in [0.25f64, 0.5, 0.8] {
                let w = 1.0 - u;
                let x = w.powi(3) * b0.x + 3.0 * w * w * u * b1.x + 3.0 * w * u * u * b2.x + u.powi(3) * b3.x;
                let y = w.powi(3) * b0.y + 3.0 * w * w * u * b1.y + 3.0 * w * u * u * b2.y + u.powi(3) * b3.y;
                let p = eval_spline(&fit, k, u);
                assert!(close(x, p.x, 9));
                assert!(close(y, p.y, 9));
            }
        }
    }

    #[test]
    fn the_bezier_form_of_a_closed_curve_ends_where_it_began() {
        let SplineBezierForm { poles, knots } = spline_bezier_form(&fit_spline_points(&square(), &none(4), true).unwrap());
        assert_eq!(poles.len(), 13);
        assert_eq!(knots.len(), 5);
        assert_eq!(poles[12], v(0.0, 0.0));
    }

    #[test]
    fn the_hit_test_finds_the_segment_a_spot_lies_nearest_and_how_far_off_it_is() {
        let fit = fit_spline_points(&wave(), &none(4), false).unwrap();
        // Right on the second segment's midpoint.
        let on = eval_spline(&fit, 1, 0.5);
        let hit = nearest_spline_segment(&fit, on, POLYLINE_STEPS);
        assert_eq!(hit.segment, 1);
        assert!(hit.dist < 0.05);
        // Well away from the curve: still a nearest segment, but far.
        let off = nearest_spline_segment(&fit, v(15.0, 40.0), POLYLINE_STEPS);
        assert!(off.dist > 30.0);
        // A spot beside the last segment inserts before the last point.
        let late = eval_spline(&fit, 2, 0.7);
        assert_eq!(nearest_spline_segment(&fit, v(late.x, late.y + 0.2), POLYLINE_STEPS).segment, 2);
    }

    #[test]
    fn hangs_the_label_halfway_along() {
        let fit = fit_spline_points(&[v(0.0, 0.0), v(10.0, 0.0)], &none(2), false).unwrap();
        let mid = spline_midpoint(&fit);
        assert!(close(mid.x, 5.0, 9));
        assert!(close(mid.y, 0.0, 9));
    }
}
