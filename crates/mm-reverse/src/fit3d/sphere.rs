// SPDX-License-Identifier: AGPL-3.0-only
//! The sphere's best fit: Coope's algebraic fit, linear and dependable, as a start, then the
//! orthogonal-distance fixed-point iteration that makes it the Gaussian best fit, once per
//! round of the "used points" clipping. The robust estimate a click starts from lives in
//! `ransac`.
//!
//! A ball whose diameter is known — a gauge ball, a ballbar's — can be fitted with it held:
//! the free fit is the start, and only the centre moves. On a partial scan that is what keeps
//! the centre where it belongs, since there the radius and the centre's place along the
//! cap's axis trade off against each other, and a surface that reads a little off towards
//! the grazing rim moves both of them several times as far as it is off.
//!
//! Ported from ScanRuler (`src/core/fit/sphere.ts`), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only.

use glam::DVec3;

use super::clip::clipped_refit;
use super::linalg3::{centroid_of, hypot3, js_max};
use super::ransac::sphere_residual;
use super::types::Sphere;
use crate::fit2d::solve_linear;

/// Linear (algebraic) sphere fit after Coope: with q = p − centroid, solve 2q·a + t = |q|²
/// in least squares for the centre offset a and t = r² − |a|². Fast and dependable as an
/// initial estimate.
fn fit_sphere_algebraic(positions: &[DVec3], idx: &[u32]) -> Option<Sphere> {
    let n = idx.len();
    if n < 4 {
        return None;
    }
    let m = centroid_of(positions, idx);

    let (mut sxx, mut sxy, mut sxz, mut syy, mut syz, mut szz) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let (mut sx, mut sy, mut sz) = (0.0, 0.0, 0.0);
    let (mut sxb, mut syb, mut szb, mut sb) = (0.0, 0.0, 0.0, 0.0);
    for &i in idx {
        let q = positions[i as usize] - m;
        let b = q.x * q.x + q.y * q.y + q.z * q.z;
        sxx += q.x * q.x;
        sxy += q.x * q.y;
        sxz += q.x * q.z;
        syy += q.y * q.y;
        syz += q.y * q.z;
        szz += q.z * q.z;
        sx += q.x;
        sy += q.y;
        sz += q.z;
        sxb += q.x * b;
        syb += q.y * b;
        szb += q.z * b;
        sb += b;
    }

    #[rustfmt::skip]
    let mut a = [
        4.0 * sxx, 4.0 * sxy, 4.0 * sxz, 2.0 * sx,
        4.0 * sxy, 4.0 * syy, 4.0 * syz, 2.0 * sy,
        4.0 * sxz, 4.0 * syz, 4.0 * szz, 2.0 * sz,
        2.0 * sx, 2.0 * sy, 2.0 * sz, n as f64,
    ];
    let rhs = [2.0 * sxb, 2.0 * syb, 2.0 * szb, sb];
    let u = solve_linear(4, &mut a, &rhs)?;

    let r2 = u[3] + u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
    if !(r2 > 0.0) || !r2.is_finite() {
        return None;
    }
    Some(Sphere { center: DVec3::new(m.x + u[0], m.y + u[1], m.z + u[2]), r: r2.sqrt() })
}

/// Geometric (orthogonal-distance) refinement minimising Σ(dᵢ − r)² by the standard
/// fixed-point iteration c ← x̄ − r̄·ū. This is the Gaussian best fit.
fn refine_sphere_geometric(positions: &[DVec3], idx: &[u32], init: Sphere, max_iter: usize) -> Sphere {
    let n = idx.len();
    if n == 0 {
        return init;
    }
    let mean = centroid_of(positions, idx);

    let (mut cx, mut cy, mut cz, mut r) = (init.center.x, init.center.y, init.center.z, init.r);
    for _ in 0..max_iter {
        let (mut sd, mut sux, mut suy, mut suz, mut m) = (0.0, 0.0, 0.0, 0.0, 0usize);
        for &i in idx {
            let p = positions[i as usize];
            let dx = p.x - cx;
            let dy = p.y - cy;
            let dz = p.z - cz;
            let d = (dx * dx + dy * dy + dz * dz).sqrt();
            if d < 1e-12 {
                continue;
            }
            sd += d;
            sux += dx / d;
            suy += dy / d;
            suz += dz / d;
            m += 1;
        }
        if m == 0 {
            break;
        }
        let mf = m as f64;
        let r_new = sd / mf;
        let ncx = mean.x - r_new * (sux / mf);
        let ncy = mean.y - r_new * (suy / mf);
        let ncz = mean.z - r_new * (suz / mf);
        let moved = ((ncx - cx).powi(2) + (ncy - cy).powi(2) + (ncz - cz).powi(2)).sqrt();
        cx = ncx;
        cy = ncy;
        cz = ncz;
        r = r_new;
        if moved < 1e-10 * js_max(1.0, r) {
            break;
        }
    }
    if !cx.is_finite() || !r.is_finite() {
        return init;
    }
    Sphere { center: DVec3::new(cx, cy, cz), r }
}

/// The centre of the sphere of radius `r` that fits best — Gauss–Newton on Σ(|p − c| − r)²
/// over the centre alone, from `start`.
fn fit_sphere_centre(positions: &[DVec3], idx: &[u32], r: f64, start: Sphere, max_iter: usize) -> Option<Sphere> {
    let (mut cx, mut cy, mut cz) = (start.center.x, start.center.y, start.center.z);
    for _ in 0..max_iter {
        let (mut axx, mut axy, mut axz, mut ayy, mut ayz, mut azz) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let (mut bx, mut by, mut bz) = (0.0, 0.0, 0.0);
        for &i in idx {
            let p = positions[i as usize];
            let dx = p.x - cx;
            let dy = p.y - cy;
            let dz = p.z - cz;
            let d = (dx * dx + dy * dy + dz * dz).sqrt();
            if d < 1e-12 {
                continue;
            }
            let (ux, uy, uz) = (dx / d, dy / d, dz / d);
            let e = d - r;
            axx += ux * ux;
            axy += ux * uy;
            axz += ux * uz;
            ayy += uy * uy;
            ayz += uy * uz;
            azz += uz * uz;
            bx += ux * e;
            by += uy * e;
            bz += uz * e;
        }
        let step = solve_linear(3, &mut [axx, axy, axz, axy, ayy, ayz, axz, ayz, azz], &[bx, by, bz])?;
        cx += step[0];
        cy += step[1];
        cz += step[2];
        if hypot3(step[0], step[1], step[2]) < 1e-10 * js_max(1.0, r) {
            break;
        }
    }
    if cx.is_finite() && cy.is_finite() && cz.is_finite() {
        Some(Sphere { center: DVec3::new(cx, cy, cz), r })
    } else {
        None
    }
}

/// A clipped sphere fit.
#[derive(Debug, Clone, PartialEq)]
pub struct ClippedSphereFit {
    pub sphere: Sphere,
    /// RMS of radial residuals over the used points.
    pub sigma: f64,
    /// Peak-to-peak radial residual — the surface's sphericity.
    pub span: f64,
    /// The signed radial extremes `span` is the difference of: how far inside and outside
    /// the fitted sphere the kept points reach.
    pub min: f64,
    pub max: f64,
    pub used: Vec<u32>,
    /// With the radius held: the sphere the same used points give when it is free — how far
    /// its radius is off the held one says how far the scan sits off the ball.
    pub free: Option<Sphere>,
}

/// The free fit of one round: algebraic, polished geometrically.
fn free_fit(positions: &[DVec3], used: &[u32]) -> Option<Sphere> {
    let alg = fit_sphere_algebraic(positions, used)?;
    Some(refine_sphere_geometric(positions, used, alg, 150))
}

/// Gaussian best fit with GOM-style "used points" clipping (see `clipped_refit`), each
/// round an algebraic fit polished geometrically — and, with `held` (a radius), the centre
/// then fitted with that radius.
pub fn fit_sphere_clipped(positions: &[DVec3], idx: &[u32], k: f64, held: Option<f64>) -> Option<ClippedSphereFit> {
    let r = clipped_refit(
        positions,
        idx,
        k,
        |used| {
            let free = free_fit(positions, used)?;
            match held {
                Some(radius) => fit_sphere_centre(positions, used, radius, free, 50),
                None => Some(free),
            }
        },
        sphere_residual,
    )?;
    let free = if held.is_some() { free_fit(positions, &r.used) } else { None };
    Some(ClippedSphereFit { sphere: r.model, sigma: r.sigma, span: r.span, min: r.min, max: r.max, used: r.used, free })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit2d::Mulberry32;
    use crate::fit3d::testkit::{all_indices, icosphere, p32, rad, sample_sphere};

    const CENTER: [f64; 3] = [10.0, -5.0, 3.0];
    const R: f64 = 12.7;

    // sphereFit.test.ts — 'Gaussian sphere fit'.
    #[test]
    fn recovers_a_full_noisy_sphere_to_micron_level() {
        let pts = sample_sphere(30_000, CENTER, R, 0.02, 42, None);
        let fit = fit_sphere_clipped(&pts, &all_indices(30_000), 3.0, None).expect("a fit");
        assert!((fit.sphere.r - R).abs() < 0.005);
        assert!((fit.sphere.center.x - CENTER[0]).abs() < 0.005);
        assert!((fit.sphere.center.y - CENTER[1]).abs() < 0.005);
        assert!((fit.sphere.center.z - CENTER[2]).abs() < 0.005);
        assert!(fit.sigma > 0.015);
        assert!(fit.sigma < 0.025);
    }

    #[test]
    fn recovers_a_60_degree_cap() {
        let pts = sample_sphere(20_000, CENTER, R, 0.02, 43, Some(rad(60.0)));
        let fit = fit_sphere_clipped(&pts, &all_indices(20_000), 3.0, None).expect("a fit");
        assert!((fit.sphere.r - R).abs() < 0.02);
        assert!((fit.sphere.center.z - CENTER[2]).abs() < 0.02);
    }

    #[test]
    fn sigma_clipping_sheds_sparse_gross_outliers() {
        let mut pts = sample_sphere(20_000, CENTER, R, 0.02, 44, None);
        // Push 2% of points 0.5 mm outward — dust / reconstruction spikes.
        let mut rand = Mulberry32::new(99);
        let c = DVec3::from_array(CENTER);
        for _ in 0..400 {
            let k = rand.index(20_000);
            let p = pts[k];
            let d = p - c;
            let len = (d.x * d.x + d.y * d.y + d.z * d.z).sqrt();
            pts[k] = p32(p.x + (d.x / len) * 0.5, p.y + (d.y / len) * 0.5, p.z + (d.z / len) * 0.5);
        }
        let clipped = fit_sphere_clipped(&pts, &all_indices(20_000), 3.0, None).expect("a fit");
        assert!((clipped.sphere.r - R).abs() < 0.01);
        assert!(clipped.used.len() < 20_000);
    }

    /// sphereKnownDiameter.test.ts's `cap`: a cap of half-angle `cap_deg` round +Z on a ball
    /// of radius R centred at the origin, on a regular grid, reading `rim_mm` outside the ball
    /// at its rim and rising to it quadratically from the pole — a scanner that loses
    /// accuracy as the surface turns away from it.
    fn cap(cap_deg: f64, rim_mm: f64) -> Vec<DVec3> {
        let (rings, round) = (60, 120);
        let mut out = Vec::with_capacity(rings * round);
        for i in 1..=rings {
            let t = i as f64 / rings as f64;
            let a = rad(cap_deg) * t;
            let r = R + rim_mm * t * t;
            for j in 0..round {
                let ph = (2.0 * std::f64::consts::PI * j as f64) / round as f64;
                out.push(p32(r * a.sin() * ph.cos(), r * a.sin() * ph.sin(), r * a.cos()));
            }
        }
        out
    }

    // sphereKnownDiameter.test.ts — the two tests that run on fitSphereClipped itself.
    #[test]
    fn keeps_the_centre_of_a_30_degree_cap_where_the_free_fit_lets_it_run() {
        let pts = cap(30.0, 0.01);
        let idx = all_indices(pts.len());
        let free = fit_sphere_clipped(&pts, &idx, 0.0, None).expect("a free fit");
        let held = fit_sphere_clipped(&pts, &idx, 0.0, Some(R)).expect("a held fit");
        // The free fit trades the rim's 10 µm for a ball 0.15 mm too large, its centre 75 µm
        // back along the axis.
        assert!(free.sphere.center.z.abs() > 0.05);
        assert_eq!(held.sphere.r, R);
        assert!(held.sphere.center.length() < 0.005);
        // And it says what the points give free: the same fit as above.
        let held_free = held.free.expect("the free sphere beside the held one");
        assert!((held_free.r - free.sphere.r).abs() < 0.5e-6);
        assert!(free.free.is_none());
    }

    #[test]
    fn changes_nothing_on_a_whole_ball_scanned_true() {
        let (positions, _) = icosphere(4, 5.0, 0.0, 7);
        let idx = all_indices(positions.len());
        let free = fit_sphere_clipped(&positions, &idx, 3.0, None).expect("a free fit");
        let held = fit_sphere_clipped(&positions, &idx, 3.0, Some(free.sphere.r)).expect("a held fit");
        assert!((held.sphere.center - free.sphere.center).length() < 1e-6);
    }
}
