// SPDX-License-Identifier: AGPL-3.0-only
//! Circle through three or more picked points — GOM's "3-point circle", generalised: three
//! points give the exact circle, more give the Gaussian best fit. Small point counts, so it
//! runs wherever it is asked; the patch pipeline never sees a circle.
//!
//! Ported from ScanRuler (`src/core/fit/circle.ts`; its 2D kernel, `circle2d.ts`, is
//! `crate::fit2d::fit_circle_2d`), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.

use glam::{DVec2, DVec3};

use super::linalg3::{eigen_of_scatter, hypot2, js_max, ortho_basis};
use super::types::{CircleFit, FitBase};
use crate::fit2d::{fit_circle_2d, FitError};

/// The best-fit circle through the given points.
///
/// The plane is the total-least-squares plane of the points (exact for three); the circle
/// is fitted in it — algebraic (Coope) start, orthogonal-distance refinement, the same 2D
/// fit the cylinder pipeline uses. σ is the RMS 3D distance of the points to the circle
/// curve, the form error the peak-to-peak in-plane radial residual (the circularity of the
/// picks).
///
/// Fails with a user-facing message on degenerate input.
pub fn circle_from_points(points: &[DVec3]) -> Result<CircleFit, FitError> {
    let n = points.len();
    if n < 3 {
        return Err(FitError::new("A circle needs at least three points."));
    }

    // Centroid and covariance — the TLS plane normal is the direction the points scatter
    // least along.
    let nf = n as f64;
    let (mut mx, mut my, mut mz) = (0.0, 0.0, 0.0);
    for p in points {
        mx += p.x;
        my += p.y;
        mz += p.z;
    }
    mx /= nf;
    my /= nf;
    mz /= nf;
    let (mut cxx, mut cxy, mut cxz, mut cyy, mut cyz, mut czz) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let mut spread = 0.0f64;
    for p in points {
        let (x, y, z) = (p.x - mx, p.y - my, p.z - mz);
        cxx += x * x;
        cxy += x * y;
        cxz += x * z;
        cyy += y * y;
        cyz += y * z;
        czz += z * z;
        spread = js_max(spread, x * x + y * y + z * z);
    }
    if !(spread > 0.0) {
        return Err(FitError::new("The points coincide — they do not define a circle."));
    }
    let eig = eigen_of_scatter(cxx, cxy, cxz, cyy, cyz, czz);
    // Points on a line scatter along one direction only — the two smaller eigenvalues both
    // vanish and no plane (or circle) is defined.
    if !(eig.values[1] > 1e-10 * spread) {
        return Err(FitError::new("The points lie on a line — they do not define a circle."));
    }
    let mut normal = eig.vectors[0];
    let (u, v) = ortho_basis(normal);

    // Project onto the plane through the centroid.
    let mut projected = Vec::with_capacity(n);
    let mut heights = Vec::with_capacity(n);
    for p in points {
        let (x, y, z) = (p.x - mx, p.y - my, p.z - mz);
        projected.push(DVec2::new(x * u.x + y * u.y + z * u.z, x * v.x + y * v.y + z * v.z));
        heights.push(x * normal.x + y * normal.y + z * normal.z);
    }

    // The circle itself is the shared 2D kernel's job.
    let (c, r) = fit_circle_2d(&projected).ok_or_else(|| FitError::new("Couldn't fit a circle through these points."))?;
    let (cu, cv) = (c.x, c.y);

    // Which way the normal points is arbitrary for a TLS plane; pick the same side every
    // time so re-picking the same feature gives the same element.
    if normal.z < 0.0 || (normal.z == 0.0 && (normal.y < 0.0 || (normal.y == 0.0 && normal.x < 0.0))) {
        normal = -normal;
    }

    // Residuals: radial in the circle's own plane for the form error, full 3D distance to
    // the curve for σ.
    let mut sum_sq = 0.0;
    let mut min_r = f64::INFINITY;
    let mut max_r = f64::NEG_INFINITY;
    for (q, h) in projected.iter().zip(&heights) {
        let er = hypot2(q.x - cu, q.y - cv) - r;
        if er < min_r {
            min_r = er;
        }
        if er > max_r {
            max_r = er;
        }
        sum_sq += er * er + h * h;
    }

    Ok(CircleFit {
        base: FitBase {
            sigma: (sum_sq / nf).sqrt(),
            used_points: n,
            region_size: n,
            form_error: Some(max_r - min_r),
            residual_min: Some(min_r),
            residual_max: Some(max_r),
        },
        center: DVec3::new(mx + cu * u.x + cv * v.x, my + cu * u.y + cv * v.y, mz + cu * u.z + cv * v.z),
        normal,
        radius: r,
    })
}

#[cfg(test)]
mod tests {
    // circleFit.test.ts. Its 'moves rigidly with transformFit' case tests the alignment
    // module's transform of a fit, not the fit, and is not here.
    use super::*;
    use crate::fit2d::Mulberry32;

    /// Points on a circle of the given pose, optionally perturbed.
    fn circle_points(center: [f64; 3], normal: [f64; 3], radius: f64, angles: &[f64], jitter: f64, seed: u32) -> Vec<DVec3> {
        let nl = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        let n = DVec3::new(normal[0] / nl, normal[1] / nl, normal[2] / nl);
        // Any basis across the normal.
        let pick = if n.x.abs() < 0.9 { DVec3::X } else { DVec3::Y };
        let u0 = DVec3::new(pick.y * n.z - pick.z * n.y, pick.z * n.x - pick.x * n.z, pick.x * n.y - pick.y * n.x);
        let ul = u0.length();
        let u = u0 / ul;
        let v = DVec3::new(n.y * u.z - n.z * u.y, n.z * u.x - n.x * u.z, n.x * u.y - n.y * u.x);
        let mut rand = Mulberry32::new(seed);
        angles
            .iter()
            .map(|&a| {
                let r = radius + if jitter != 0.0 { (rand.next_f64() - 0.5) * 2.0 * jitter } else { 0.0 };
                let h = if jitter != 0.0 { (rand.next_f64() - 0.5) * 2.0 * jitter } else { 0.0 };
                DVec3::new(
                    center[0] + r * (a.cos() * u.x + a.sin() * v.x) + h * n.x,
                    center[1] + r * (a.cos() * u.y + a.sin() * v.y) + h * n.y,
                    center[2] + r * (a.cos() * u.z + a.sin() * v.z) + h * n.z,
                )
            })
            .collect()
    }

    /// `expect(a).toBeCloseTo(b, digits)`.
    fn close(a: f64, b: f64, digits: i32) -> bool {
        (a - b).abs() < 10f64.powi(-digits) / 2.0
    }

    #[test]
    fn recovers_the_exact_circle_through_three_points() {
        let fit = circle_from_points(&circle_points([10.0, -4.0, 2.0], [0.0, 0.0, 1.0], 6.5, &[0.3, 1.9, 4.1], 0.0, 7)).unwrap();
        assert!(close(fit.radius, 6.5, 9));
        assert!(close(fit.center.x, 10.0, 9));
        assert!(close(fit.center.y, -4.0, 9));
        assert!(close(fit.center.z, 2.0, 9));
        assert!(close(fit.normal.z.abs(), 1.0, 9));
        assert!(fit.base.sigma < 1e-9);
        assert_eq!(fit.base.used_points, 3);
    }

    #[test]
    fn fits_a_tilted_circle_from_many_noisy_points() {
        let angles: Vec<f64> = (0..24).map(|i| (i as f64 / 24.0) * 2.0 * std::f64::consts::PI).collect();
        let fit = circle_from_points(&circle_points([3.0, 8.0, -5.0], [1.0, 2.0, 2.0], 12.0, &angles, 0.02, 7)).unwrap();
        assert!(close(fit.radius, 12.0, 1));
        assert!(close(fit.center.x, 3.0, 1));
        assert!(close(fit.center.y, 8.0, 1));
        assert!(close(fit.center.z, -5.0, 1));
        // Same line as [1,2,2]/3, either way round.
        let d = (fit.normal.x * (1.0 / 3.0) + fit.normal.y * (2.0 / 3.0) + fit.normal.z * (2.0 / 3.0)).abs();
        assert!(d > 0.999);
        assert!(fit.base.sigma < 0.05);
        assert!(fit.base.sigma > 0.0);
        // Peak-to-peak of a ±0.02 mm radial jitter.
        let form = fit.base.form_error.unwrap();
        assert!(form > 0.01);
        assert!(form < 0.09);
    }

    #[test]
    fn works_on_a_partial_arc() {
        let angles: Vec<f64> = (0..9).map(|i| 0.2 + (i as f64 / 8.0) * 1.1).collect();
        let fit = circle_from_points(&circle_points([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 40.0, &angles, 0.005, 7)).unwrap();
        assert!(close(fit.radius, 40.0, 0));
    }

    #[test]
    fn refuses_fewer_than_three_points() {
        let err = circle_from_points(&[DVec3::ZERO, DVec3::X]).unwrap_err();
        assert!(err.0.contains("three points"));
    }

    #[test]
    fn refuses_collinear_points() {
        let pts: Vec<DVec3> = (0..4).map(|i| DVec3::new(i as f64, 0.0, 0.0)).collect();
        assert!(circle_from_points(&pts).unwrap_err().0.contains("line"));
    }

    #[test]
    fn refuses_coincident_points() {
        assert!(circle_from_points(&[DVec3::new(1.0, 2.0, 3.0); 3]).is_err());
    }

    #[test]
    fn orients_the_normal_the_same_way_whatever_the_pick_order() {
        let pts = circle_points([0.0, 0.0, 0.0], [0.0, 0.0, -1.0], 5.0, &[0.1, 2.0, 4.4], 0.0, 7);
        let a = circle_from_points(&pts).unwrap();
        let reversed: Vec<DVec3> = pts.iter().rev().copied().collect();
        let b = circle_from_points(&reversed).unwrap();
        assert!(close(a.normal.x, b.normal.x, 12));
        assert!(close(a.normal.y, b.normal.y, 12));
        assert!(close(a.normal.z, b.normal.z, 12));
        // The deterministic flip puts it on the +Z side here.
        assert!(a.normal.z > 0.0);
    }

    // circleFit.test.ts — 'fitCircle2d': the shared 2D kernel, which is
    // `crate::fit2d::fit_circle_2d` here.
    #[test]
    fn fits_a_short_arc_far_from_the_origin_as_image_pixel_coordinates_are() {
        // 40 points over 76° of a radius-30 circle centred at (2480, 230): the kind of arc an
        // edge chain on a 600 dpi scan hands over.
        let pts: Vec<DVec2> = (0..40).map(|i| DVec2::new(2480.0 + 30.0 * (i as f64 / 30.0).cos(), 230.0 + 30.0 * (i as f64 / 30.0).sin())).collect();
        let (c, r) = fit_circle_2d(&pts).expect("a circle");
        assert!(close(c.x, 2480.0, 6));
        assert!(close(c.y, 230.0, 6));
        assert!(close(r, 30.0, 6));
    }
}
