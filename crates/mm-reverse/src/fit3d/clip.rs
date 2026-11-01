// SPDX-License-Identifier: AGPL-3.0-only
//! The "used points" clipping every fitted shape goes through: fit, drop the points further
//! out than k·sigma, fit again, until the set stops changing — the Gaussian best fit with
//! GOM-style outlier rejection, written once so the plane, sphere, cylinder, cone and torus
//! all report sigma and form deviation the same way. A shape brings only its own fit and its
//! signed residual.
//!
//! Ported from ScanRuler (`src/core/fit/clip.ts`), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only.

use glam::DVec3;

/// One clipped best fit: the model, and what its residuals say over the points it kept.
#[derive(Debug, Clone, PartialEq)]
pub struct ClippedRefit<M> {
    pub model: M,
    /// RMS of the residuals over the used points.
    pub sigma: f64,
    /// Peak-to-peak residual over the used points — the GD&T form deviation (flatness,
    /// cylindricity, sphericity) of the surface the fit kept.
    pub span: f64,
    /// The signed residual extremes `span` is the difference of, in the model's own
    /// residual convention — along the normal for a plane, radially out for a cylinder,
    /// sphere or cone: how far the kept surface reaches below and above the fit.
    pub min: f64,
    pub max: f64,
    /// The points the fit kept (vertex indices).
    pub used: Vec<u32>,
}

/// Gaussian best fit with GOM-style "used points" clipping, shared by the plane, sphere,
/// cylinder, cone and torus fits: fit, discard residuals beyond k·sigma, refit, until the
/// point set is stable. k = 0 means use all points.
///
/// `fit` returning `None` (a degenerate model) ends the loop with the last good round's
/// result. Non-finite residuals — a point sitting on a cylinder's axis, say — are left out
/// of sigma and dropped by the clip.
pub fn clipped_refit<M>(
    positions: &[DVec3],
    idx: &[u32],
    k: f64,
    mut fit: impl FnMut(&[u32]) -> Option<M>,
    residual: impl Fn(&M, DVec3) -> f64,
) -> Option<ClippedRefit<M>> {
    let mut used: Vec<u32> = idx.to_vec();
    let mut result: Option<ClippedRefit<M>> = None;

    for _iter in 0..12 {
        let Some(model) = fit(&used) else { return result };

        let mut sum_sq = 0.0;
        let mut m = 0usize;
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        let mut res = Vec::with_capacity(used.len());
        for &i in &used {
            let e = residual(&model, positions[i as usize]);
            res.push(e);
            if !e.is_finite() {
                continue;
            }
            sum_sq += e * e;
            if e < lo {
                lo = e;
            }
            if e > hi {
                hi = e;
            }
            m += 1;
        }
        if m == 0 {
            return result;
        }
        let sigma = (sum_sq / m as f64).sqrt();

        if k <= 0.0 || sigma < 1e-9 {
            return Some(ClippedRefit { model, sigma, span: hi - lo, min: lo, max: hi, used });
        }
        let thr = k * sigma;
        let keep = res.iter().filter(|e| e.abs() <= thr).count();
        if keep == used.len() || keep < 10 {
            return Some(ClippedRefit { model, sigma, span: hi - lo, min: lo, max: hi, used });
        }

        let next: Vec<u32> = used.iter().zip(&res).filter(|(_, e)| e.abs() <= thr).map(|(&i, _)| i).collect();
        let kept = std::mem::replace(&mut used, next);
        result = Some(ClippedRefit { model, sigma, span: hi - lo, min: lo, max: hi, used: kept });
    }
    result
}
