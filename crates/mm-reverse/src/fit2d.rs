// SPDX-License-Identifier: AGPL-3.0-only
//! The flat fits: line, circle and arc through 2D points, whether those points were clicked
//! by hand or collected off a slice chain. Ported from ScanRuler (`src/core/flat/fit.ts`,
//! `src/core/fit/circle2d.ts`, `src/core/fit/linalg.ts`), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only.
//!
//! The circle kernel is an algebraic (Coope) start refined by the orthogonal-distance
//! fixed-point iteration; a line is the total-least-squares direction of the 2×2 covariance.
//! Each fit reports σ (the RMS residual) and the form error (peak to peak).

use glam::DVec2;

/// A fit that could not be made, with a message meant for the user.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("{0}")]
pub struct FitError(pub String);

impl FitError {
    pub fn new(msg: impl Into<String>) -> Self {
        Self(msg.into())
    }
}

/// A straight edge. The line the math uses is infinite; `center`, `dir` and `length`
/// describe the measured segment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LineFit {
    pub center: DVec2,
    pub dir: DVec2,
    pub length: f64,
    pub sigma: f64,
    pub used_points: usize,
    pub form_error: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CircleFit {
    pub center: DVec2,
    pub radius: f64,
    pub sigma: f64,
    pub used_points: usize,
    pub form_error: f64,
}

/// A partial circle: the fit is the full circle's, the angles record which part of it was
/// measured. Counter-clockwise from `start`, radians from +X; `sweep` is positive and at
/// most a full turn.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArcFit {
    pub center: DVec2,
    pub radius: f64,
    pub start: f64,
    pub sweep: f64,
    pub sigma: f64,
    pub used_points: usize,
    pub form_error: f64,
}

/// Gaussian elimination with partial pivoting on a row-major `n×n` system. `a` is
/// overwritten. `None` when the system is singular to working precision.
pub fn solve_linear(n: usize, a: &mut [f64], b: &[f64]) -> Option<Vec<f64>> {
    let max_abs = a[..n * n].iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let tiny = max_abs.max(1.0) * 1e-13;
    let mut x = b[..n].to_vec();
    for col in 0..n {
        let mut pivot_row = col;
        let mut best = a[col * n + col].abs();
        for r in col + 1..n {
            let v = a[r * n + col].abs();
            if v > best {
                best = v;
                pivot_row = r;
            }
        }
        if best < tiny {
            return None;
        }
        if pivot_row != col {
            for c in col..n {
                a.swap(col * n + c, pivot_row * n + c);
            }
            x.swap(col, pivot_row);
        }
        let piv = a[col * n + col];
        for r in col + 1..n {
            let f = a[r * n + col] / piv;
            if f == 0.0 {
                continue;
            }
            for c in col..n {
                a[r * n + c] -= f * a[col * n + c];
            }
            x[r] -= f * x[col];
        }
    }
    for row in (0..n).rev() {
        let mut s = x[row];
        for c in row + 1..n {
            s -= a[row * n + c] * x[c];
        }
        x[row] = s / a[row * n + row];
    }
    Some(x)
}

/// Mulberry32: the small seeded generator ScanRuler uses wherever a fit samples at random,
/// so a refit of the same points gives the same answer.
#[derive(Debug, Clone)]
pub struct Mulberry32(u32);

impl Mulberry32 {
    pub fn new(seed: u32) -> Self {
        Self(seed)
    }

    /// A uniform number in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x6d2b_79f5);
        let a = self.0;
        let mut t = (a ^ (a >> 15)).wrapping_mul(1 | a);
        t = (t.wrapping_add((t ^ (t >> 7)).wrapping_mul(61 | t))) ^ t;
        ((t ^ (t >> 14)) as f64) / 4_294_967_296.0
    }

    /// A standard normal number (Box–Muller), as the tests' `gaussian` draws it.
    pub fn gaussian(&mut self) -> f64 {
        let mut u = 0.0;
        while u == 0.0 {
            u = self.next_f64();
        }
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * self.next_f64()).cos()
    }

    /// An index in `0..n`.
    pub fn index(&mut self, n: usize) -> usize {
        ((self.next_f64() * n as f64) as usize).min(n.saturating_sub(1))
    }
}

/// Best-fit circle through 2D points: centre and radius. `None` when the points do not
/// determine a circle.
pub fn fit_circle_2d(points: &[DVec2]) -> Option<(DVec2, f64)> {
    let n = points.len();
    if n < 3 {
        return None;
    }
    // Work about the centroid: the normal equations carry fourth powers of the coordinates,
    // and a short arc far from the origin would cancel them to round-off.
    let mean = points.iter().copied().sum::<DVec2>() / n as f64;
    let pu: Vec<f64> = points.iter().map(|p| p.x - mean.x).collect();
    let pv: Vec<f64> = points.iter().map(|p| p.y - mean.y).collect();

    let (mut suu, mut suv, mut svv, mut su, mut sv, mut sub, mut svb, mut sb) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for i in 0..n {
        let b = pu[i] * pu[i] + pv[i] * pv[i];
        suu += pu[i] * pu[i];
        suv += pu[i] * pv[i];
        svv += pv[i] * pv[i];
        su += pu[i];
        sv += pv[i];
        sub += pu[i] * b;
        svb += pv[i] * b;
        sb += b;
    }
    let mut a = [4.0 * suu, 4.0 * suv, 2.0 * su, 4.0 * suv, 4.0 * svv, 2.0 * sv, 2.0 * su, 2.0 * sv, n as f64];
    let sol = solve_linear(3, &mut a, &[2.0 * sub, 2.0 * svb, sb])?;
    let mut cu = sol[0];
    let mut cv = sol[1];
    let r2 = sol[2] + cu * cu + cv * cv;
    if !(r2 > 0.0) || !r2.is_finite() {
        return None;
    }
    let mut r = r2.sqrt();

    // Orthogonal-distance refinement.
    for _ in 0..100 {
        let (mut sd, mut sxu, mut sxv, mut m) = (0.0, 0.0, 0.0, 0usize);
        for i in 0..n {
            let du = pu[i] - cu;
            let dv = pv[i] - cv;
            let dist = du.hypot(dv);
            if dist < 1e-12 {
                continue;
            }
            sd += dist;
            sxu += du / dist;
            sxv += dv / dist;
            m += 1;
        }
        if m == 0 {
            break;
        }
        let r_new = sd / m as f64;
        let nu = su / n as f64 - r_new * (sxu / m as f64);
        let nv = sv / n as f64 - r_new * (sxv / m as f64);
        let moved = (nu - cu).hypot(nv - cv);
        cu = nu;
        cv = nv;
        r = r_new;
        if moved < 1e-10 * r.max(1.0) {
            break;
        }
    }
    if !cu.is_finite() || !cv.is_finite() || !r.is_finite() || !(r > 0.0) {
        return None;
    }
    Some((DVec2::new(cu + mean.x, cv + mean.y), r))
}

/// Total-least-squares line through two or more points. σ is the RMS perpendicular residual,
/// the form error its peak-to-peak span; the segment covers the points' extent along the
/// line, and the direction always points to +X (or +Y when upright) so a refit agrees.
pub fn fit_line_points(points: &[DVec2]) -> Result<LineFit, FitError> {
    let n = points.len();
    if n < 2 {
        return Err(FitError::new("A line needs at least two points."));
    }
    let mean = points.iter().copied().sum::<DVec2>() / n as f64;
    let (mut sxx, mut sxy, mut syy, mut spread) = (0.0, 0.0, 0.0, 0.0f64);
    for p in points {
        let d = *p - mean;
        sxx += d.x * d.x;
        sxy += d.x * d.y;
        syy += d.y * d.y;
        spread = spread.max(d.length_squared());
    }
    if !(spread > 0.0) {
        return Err(FitError::new("The points coincide — they do not define a line."));
    }
    let theta = 0.5 * (2.0 * sxy).atan2(sxx - syy);
    let mut dir = DVec2::new(theta.cos(), theta.sin());
    if dir.x < 0.0 || (dir.x == 0.0 && dir.y < 0.0) {
        dir = -dir;
    }
    let (mut t_min, mut t_max, mut e_min, mut e_max, mut sum_sq) = (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY, 0.0);
    for p in points {
        let d = *p - mean;
        let t = d.dot(dir);
        let e = -d.x * dir.y + d.y * dir.x;
        t_min = t_min.min(t);
        t_max = t_max.max(t);
        e_min = e_min.min(e);
        e_max = e_max.max(e);
        sum_sq += e * e;
    }
    let t_mid = (t_min + t_max) / 2.0;
    Ok(LineFit {
        center: mean + dir * t_mid,
        dir,
        length: t_max - t_min,
        sigma: (sum_sq / n as f64).sqrt(),
        used_points: n,
        form_error: e_max - e_min,
    })
}

fn circle_with_residuals(points: &[DVec2], what: &str) -> Result<(DVec2, f64, f64, f64), FitError> {
    if points.len() < 3 {
        return Err(FitError::new(format!("A {what} needs at least three points.")));
    }
    let (center, r) = fit_circle_2d(points).ok_or_else(|| FitError::new(format!("Couldn't fit a {what} through these points.")))?;
    let (mut sum_sq, mut e_min, mut e_max) = (0.0, f64::INFINITY, f64::NEG_INFINITY);
    for p in points {
        let e = (*p - center).length() - r;
        e_min = e_min.min(e);
        e_max = e_max.max(e);
        sum_sq += e * e;
    }
    Ok((center, r, (sum_sq / points.len() as f64).sqrt(), e_max - e_min))
}

pub fn fit_circle_points(points: &[DVec2]) -> Result<CircleFit, FitError> {
    let (center, radius, sigma, form_error) = circle_with_residuals(points, "circle")?;
    Ok(CircleFit { center, radius, sigma, used_points: points.len(), form_error })
}

/// The circle fit plus the angular extent the points cover: their angles about the centre
/// with the largest gap between neighbours left out.
pub fn fit_arc_points(points: &[DVec2]) -> Result<ArcFit, FitError> {
    let (center, radius, sigma, form_error) = circle_with_residuals(points, "arc")?;
    let mut angles: Vec<f64> = points.iter().map(|p| (p.y - center.y).atan2(p.x - center.x)).collect();
    angles.sort_by(|a, b| a.total_cmp(b));
    let last = angles.len() - 1;
    let mut gap_at = last;
    let mut gap = angles[0] + std::f64::consts::TAU - angles[last];
    for i in 1..angles.len() {
        if angles[i] - angles[i - 1] > gap {
            gap = angles[i] - angles[i - 1];
            gap_at = i - 1;
        }
    }
    let start = angles[(gap_at + 1) % angles.len()];
    Ok(ArcFit { center, radius, start, sweep: std::f64::consts::TAU - gap, sigma, used_points: points.len(), form_error })
}

const RANSAC_ROUNDS: usize = 96;
/// Inliers lie within this many robust sigmas (1.4826·MAD) of the model.
const INLIER_BAND: f64 = 3.0;

/// LMedS consensus: the model with the least median absolute residual over random minimal
/// samples, and which points lie within a robust band of it.
fn consensus(points: &[DVec2], mut residuals_of: impl FnMut(&mut Mulberry32) -> Option<Vec<f64>>) -> Vec<bool> {
    let mut rand = Mulberry32::new(7);
    let mut best_median = f64::INFINITY;
    let mut best: Option<Vec<f64>> = None;
    for _ in 0..RANSAC_ROUNDS {
        let Some(residuals) = residuals_of(&mut rand) else { continue };
        let mut sorted: Vec<f64> = residuals.iter().map(|r| r.abs()).collect();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let median = sorted[sorted.len() / 2];
        if median < best_median {
            best_median = median;
            best = Some(residuals);
        }
    }
    let Some(best) = best else { return vec![true; points.len()] };
    // A clean set has a near-zero MAD; the floor keeps sub-pixel scatter from being culled.
    let band = (INLIER_BAND * 1.4826 * best_median).max(0.35);
    best.iter().map(|r| r.abs() <= band).collect()
}

fn inliers(points: &[DVec2], keep: &[bool]) -> Vec<DVec2> {
    points.iter().zip(keep).filter(|(_, k)| **k).map(|(p, _)| *p).collect()
}

/// A line through region-collected points: consensus first, TLS on what agrees.
pub fn fit_line_region(points: &[DVec2]) -> Result<LineFit, FitError> {
    let few = || FitError::new("The region caught too few edge points for a line.");
    if points.len() < 8 {
        return Err(few());
    }
    let keep = consensus(points, |rand| {
        let a = points[rand.index(points.len())];
        let b = points[rand.index(points.len())];
        let d = b - a;
        let len = d.length();
        if len < 1e-9 {
            return None;
        }
        Some(points.iter().map(|p| ((p.x - a.x) * d.y - (p.y - a.y) * d.x) / len).collect())
    });
    let kept = inliers(points, &keep);
    if kept.len() < 8 {
        return Err(few());
    }
    fit_line_points(&kept)
}

fn circle_residuals(points: &[DVec2], rand: &mut Mulberry32) -> Option<Vec<f64>> {
    let sample = [points[rand.index(points.len())], points[rand.index(points.len())], points[rand.index(points.len())]];
    let (c, r) = fit_circle_2d(&sample)?;
    Some(points.iter().map(|p| (*p - c).length() - r).collect())
}

/// A circle through region-collected points, consensus first.
pub fn fit_circle_region(points: &[DVec2]) -> Result<CircleFit, FitError> {
    let few = || FitError::new("The region caught too few edge points for a circle.");
    if points.len() < 12 {
        return Err(few());
    }
    let keep = consensus(points, |rand| circle_residuals(points, rand));
    let kept = inliers(points, &keep);
    if kept.len() < 12 {
        return Err(few());
    }
    fit_circle_points(&kept)
}

/// An arc through region-collected points, consensus first.
pub fn fit_arc_region(points: &[DVec2]) -> Result<ArcFit, FitError> {
    let few = || FitError::new("The region caught too few edge points for an arc.");
    if points.len() < 12 {
        return Err(few());
    }
    let keep = consensus(points, |rand| circle_residuals(points, rand));
    let kept = inliers(points, &keep);
    if kept.len() < 12 {
        return Err(few());
    }
    fit_arc_points(&kept)
}

/// `angle` unwound to lie in `[start, start + 2π)`.
pub fn unwind_angle(angle: f64, start: f64) -> f64 {
    let two = std::f64::consts::TAU;
    let mut a = (angle - start) % two;
    if a < 0.0 {
        a += two;
    }
    start + a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mulberry_matches_the_javascript_sequence() {
        // mulberry32(7)'s first three draws in JavaScript (node).
        let mut r = Mulberry32::new(7);
        for want in [0.011704753153026, 0.061958257574588, 0.976907632779330] {
            assert!((r.next_f64() - want).abs() < 1e-14);
        }
    }

    #[test]
    fn circle_through_an_arc_far_from_the_origin() {
        let pts: Vec<DVec2> = (0..20)
            .map(|i| {
                let a = 0.3 + i as f64 * 0.02;
                DVec2::new(5000.0 + 7.0 * a.cos(), -3000.0 + 7.0 * a.sin())
            })
            .collect();
        let arc = fit_arc_points(&pts).unwrap();
        assert!((arc.radius - 7.0).abs() < 1e-6);
        assert!((arc.center - DVec2::new(5000.0, -3000.0)).length() < 1e-6);
        assert!((arc.start - 0.3).abs() < 1e-9);
        assert!((arc.sweep - 0.38).abs() < 1e-9);
    }

    #[test]
    fn line_direction_and_extent() {
        let pts = [DVec2::new(1.0, 1.0), DVec2::new(3.0, 3.0), DVec2::new(2.0, 2.0)];
        let l = fit_line_points(&pts).unwrap();
        assert!((l.length - 8f64.sqrt()).abs() < 1e-12);
        assert!(l.sigma < 1e-12);
        assert!(l.dir.x > 0.0);
        assert!((l.center - DVec2::new(2.0, 2.0)).length() < 1e-12);
    }

    #[test]
    fn region_line_ignores_strays() {
        let mut pts: Vec<DVec2> = (0..40).map(|i| DVec2::new(i as f64, 0.5 * i as f64)).collect();
        pts.extend((0..5).map(|i| DVec2::new(i as f64 * 3.0, 30.0)));
        let l = fit_line_region(&pts).unwrap();
        assert!(l.sigma < 1e-9);
        assert_eq!(l.used_points, 40);
    }
}
