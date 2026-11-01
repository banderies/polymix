// SPDX-License-Identifier: AGPL-3.0-only
//! The robust first guess every fit from a click starts with. A patch round a click can take
//! in a neighbouring face, which would drag a least-squares fit off the surface clicked, so
//! candidates are scored by median residual (LMedS — RANSAC's sampling with no threshold to
//! tune) and only the winner's consensus set goes on to the clipped best fit. Each shape
//! brings its own candidates; the sphere's live here. The seeded generator that makes fits
//! repeatable is `crate::fit2d::Mulberry32`.
//!
//! Ported from ScanRuler (`src/core/fit/ransac.ts`), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only.

use glam::DVec3;

use super::linalg3::{extent_of_indexed, js_max};
use super::sphere::fit_sphere_clipped;
use super::types::Sphere;
use crate::fit2d::{solve_linear, Mulberry32};

/// The seed every robust estimate uses unless told otherwise.
pub const DEFAULT_SEED: u32 = 0x5eed;

/// How many candidates a robust estimate samples, and the seed it samples them with.
/// `None` takes each estimator's default (256 candidates for most, 320 for the torus; seed
/// `0x5eed`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RansacOptions {
    pub iterations: Option<usize>,
    pub seed: Option<u32>,
}

/// The winner of an LMedS search and the points that agree with it.
#[derive(Debug, Clone, PartialEq)]
pub struct RansacConsensus<M> {
    pub model: M,
    pub inliers: Vec<u32>,
    /// Robust noise estimate (1.4826 · median absolute residual).
    pub sigma_est: f64,
}

/// What a shape brings to `ransac_consensus`: an optional deterministic candidate scored
/// before the sampled ones, the candidate generator (handed the shared generator on every
/// call), and the signed residual.
pub struct RansacShape<M, G, R> {
    pub initial: Option<M>,
    pub generate: G,
    pub residual: R,
}

/// Shared scaffolding for the robust shape estimators (LMedS — no threshold parameter
/// needed): every candidate is scored by its median absolute residual over an
/// at-most-512-point strided subset of the patch, and the winner's consensus set is
/// everything within 3·1.4826·median of it, floored at a hundred-thousandth of the patch
/// diagonal so a perfectly clean patch does not gate on zero. What a candidate is and how
/// one is sampled stays with each shape; `shape` receives the patch's bounding-box diagonal
/// (for scale sanity checks) and returns the candidate generator, an optional deterministic
/// candidate to score before the sampled ones, and the signed residual. The caller refines
/// the consensus set itself — plane, sphere and cylinder each have their own refinement.
pub fn ransac_consensus<M, G, R>(
    positions: &[DVec3],
    patch: &[u32],
    opts: RansacOptions,
    shape: impl FnOnce(f64) -> RansacShape<M, G, R>,
) -> Option<RansacConsensus<M>>
where
    G: FnMut(&mut Mulberry32) -> Option<M>,
    R: Fn(&M, DVec3) -> f64,
{
    let n = patch.len();
    if n < 30 {
        return None;
    }
    let iterations = opts.iterations.unwrap_or(256);
    let mut rand = Mulberry32::new(opts.seed.unwrap_or(DEFAULT_SEED));

    let diag = extent_of_indexed(positions, patch).diagonal();
    if !(diag > 0.0) {
        return None;
    }

    let RansacShape { initial, mut generate, residual } = shape(diag);

    let score_n = n.min(512);
    let stride = n as f64 / score_n as f64;
    let subset: Vec<u32> = (0..score_n).map(|i| patch[(i as f64 * stride).floor() as usize]).collect();

    // One scratch buffer serves every candidate: the residuals are not needed after the
    // median is read, so it can be sorted in place instead of copied.
    let mut resid = vec![0.0f64; score_n];
    let mut median_residual = |m: &M| -> f64 {
        for (r, &i) in resid.iter_mut().zip(&subset) {
            *r = residual(m, positions[i as usize]).abs();
        }
        resid.sort_unstable_by(|a, b| a.total_cmp(b));
        resid[score_n >> 1]
    };

    let mut best_median = f64::INFINITY;
    let mut best: Option<M> = None;

    if let Some(initial) = initial {
        let med = median_residual(&initial);
        if med < best_median {
            best_median = med;
            best = Some(initial);
        }
    }

    for _ in 0..iterations {
        let Some(m) = generate(&mut rand) else { continue };
        let med = median_residual(&m);
        if med < best_median {
            best_median = med;
            best = Some(m);
        }
    }

    let best = best?;
    let sigma_est = 1.4826 * best_median;
    let thr = js_max(3.0 * sigma_est, diag * 1e-5);

    let inliers: Vec<u32> = patch.iter().copied().filter(|&i| residual(&best, positions[i as usize]).abs() <= thr).collect();
    if inliers.len() < 30 {
        return None;
    }
    Some(RansacConsensus { model: best, inliers, sigma_est })
}

/// Exact sphere through four points (`None` when they are near coplanar).
fn sphere_from4(positions: &[DVec3], i0: u32, i1: u32, i2: u32, i3: u32) -> Option<Sphere> {
    let p0 = positions[i0 as usize];
    let mut a = [0.0f64; 9];
    let mut b = [0.0f64; 3];
    for (r, id) in [i1, i2, i3].into_iter().enumerate() {
        let q = positions[id as usize] - p0;
        a[r * 3] = 2.0 * q.x;
        a[r * 3 + 1] = 2.0 * q.y;
        a[r * 3 + 2] = 2.0 * q.z;
        b[r] = q.x * q.x + q.y * q.y + q.z * q.z;
    }
    let c = solve_linear(3, &mut a, &b)?;
    let r = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
    if !r.is_finite() || r <= 0.0 {
        return None;
    }
    Some(Sphere { center: DVec3::new(p0.x + c[0], p0.y + c[1], p0.z + c[2]), r })
}

/// A robust sphere estimate and the points it rests on.
#[derive(Debug, Clone, PartialEq)]
pub struct RansacResult {
    pub sphere: Sphere,
    pub inliers: Vec<u32>,
    /// Robust noise estimate (1.4826 · median absolute residual), or the refit's σ when
    /// larger.
    pub sigma: f64,
}

/// The signed distance of a point from a sphere's surface (positive outside).
pub fn sphere_residual(s: &Sphere, p: DVec3) -> f64 {
    let d = p - s.center;
    (d.x * d.x + d.y * d.y + d.z * d.z).sqrt() - s.r
}

/// Robust sphere estimate on a local patch: sample four-point candidate spheres, score by
/// median absolute residual (LMedS — no threshold parameter needed), then refine on the
/// consensus set with a 3-sigma clipped Gaussian fit.
pub fn ransac_sphere(positions: &[DVec3], patch: &[u32], opts: RansacOptions) -> Option<RansacResult> {
    let core = ransac_consensus(positions, patch, opts, |diag| {
        let n = patch.len();
        RansacShape {
            initial: None,
            generate: move |rand: &mut Mulberry32| {
                let i0 = patch[rand.index(n)];
                let i1 = patch[rand.index(n)];
                let i2 = patch[rand.index(n)];
                let i3 = patch[rand.index(n)];
                if i0 == i1 || i0 == i2 || i0 == i3 || i1 == i2 || i1 == i3 || i2 == i3 {
                    return None;
                }
                let s = sphere_from4(positions, i0, i1, i2, i3)?;
                // A candidate wildly out of scale with the local patch is never the sphere
                // the user clicked.
                if s.r > diag * 200.0 || s.r < diag * 0.02 {
                    return None;
                }
                Some(s)
            },
            residual: sphere_residual,
        }
    })?;

    let refined = fit_sphere_clipped(positions, &core.inliers, 3.0, None)?;
    Some(RansacResult { sphere: refined.sphere, inliers: refined.used, sigma: js_max(js_max(refined.sigma, core.sigma_est), 1e-9) })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit3d::testkit::{all_indices, p32, rad, sample_sphere};

    const CENTER: [f64; 3] = [10.0, -5.0, 3.0];
    const R: f64 = 12.7;

    // sphereFit.test.ts — 'RANSAC sphere estimate'.
    #[test]
    fn finds_the_sphere_despite_30_percent_structured_outliers() {
        let mut all = sample_sphere(3_000, CENTER, R, 0.02, 45, Some(rad(70.0)));
        // Fake "rod" contamination: a dense line of points leaving the sphere.
        let mut rand = Mulberry32::new(7);
        for _ in 0..1_300 {
            let t = rand.next_f64() * 20.0;
            let x = CENTER[0] + (rand.next_f64() - 0.5) * 2.0;
            let y = CENTER[1] + (rand.next_f64() - 0.5) * 2.0;
            all.push(p32(x, y, CENTER[2] + R + t));
        }
        let res = ransac_sphere(&all, &all_indices(4_300), RansacOptions { seed: Some(1), ..Default::default() }).expect("a sphere");
        assert!((res.sphere.r - R).abs() < 0.1);
        assert!((res.sphere.center.x - CENTER[0]).abs() < 0.1);
    }
}
