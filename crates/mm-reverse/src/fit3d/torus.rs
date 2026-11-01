// SPDX-License-Identifier: AGPL-3.0-only
//! The torus: a tube of radius r round a circle of radius R — a fillet's round, a bend, an
//! O-ring seat. What a fillet-shaped region of a scan is, once the plane, cylinder, cone and
//! sphere are ruled out.
//!
//! The estimate rests on one fact about a torus: every surface normal passes through the
//! spine circle, a tube radius away. So for the right r the points moved inward along their
//! normals by r all lie on that circle, and a circle through them gives the axis, the centre
//! and R at once. The tube radius is one unknown, searched over a range of scales, each
//! candidate scored by the median torus residual (LMedS, as the other shapes are) — convex
//! tubes and concave fillets alike, since a fillet's spine lies the other way along the
//! normals and the search tries both signs. Damped Gauss–Newton over all seven parameters
//! polishes the winner, the axis direction stepped by two tilt angles so it never leaves
//! the unit sphere, as the cone's is.
//!
//! Ported from ScanRuler (`src/core/fit/torus.ts`), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only.

use std::collections::HashSet;

use glam::{DVec2, DVec3};

use super::clip::clipped_refit;
use super::linalg3::{eigen_of_scatter, hypot2, hypot3, js_hypot, js_max, js_round, ortho_basis, rms_over};
use super::ransac::{ransac_consensus, RansacOptions, RansacShape};
use super::types::Torus;
use crate::fit2d::{fit_circle_2d, solve_linear, Mulberry32};

/// The signed distance from the torus surface: positive outside the tube.
pub fn torus_residual(t: &Torus, p: DVec3) -> f64 {
    let d = p - t.center;
    let a = d.x * t.axis.x + d.y * t.axis.y + d.z * t.axis.z;
    let (rx, ry, rz) = (d.x - a * t.axis.x, d.y - a * t.axis.y, d.z - a * t.axis.z);
    let rho = (rx * rx + ry * ry + rz * rz).sqrt();
    // On the axis the nearest spine point is any of them; the distance to the tube is still
    // well defined.
    ((rho - t.major).powi(2) + a * a).sqrt() - t.minor
}

/// |cos| between a vertex normal and the torus's surface normal at the point — the direction
/// from the nearest spine point.
pub fn torus_normal_align(t: &Torus, p: DVec3, n: DVec3) -> f64 {
    let d = p - t.center;
    let a = d.x * t.axis.x + d.y * t.axis.y + d.z * t.axis.z;
    let (mut rx, mut ry, mut rz) = (d.x - a * t.axis.x, d.y - a * t.axis.y, d.z - a * t.axis.z);
    let rho = (rx * rx + ry * ry + rz * rz).sqrt();
    if rho < 1e-12 {
        return 0.0;
    }
    rx /= rho;
    ry /= rho;
    rz /= rho;
    // From the spine point to the surface point.
    let (sx, sy, sz) = (d.x - t.major * rx, d.y - t.major * ry, d.z - t.major * rz);
    let len = (sx * sx + sy * sy + sz * sz).sqrt();
    if len < 1e-12 {
        return 0.0;
    }
    ((sx * n.x + sy * n.y + sz * n.z) / len).abs()
}

fn rms(positions: &[DVec3], idx: &[u32], t: &Torus) -> f64 {
    rms_over(positions, idx, |p| torus_residual(t, p))
}

/// The torus whose spine passes through the points moved `shift` along their normals — a
/// circle fitted through them: the total-least-squares plane, then the 2D circle in it.
/// `None` when they do not make a circle (a straight tube, or too few). `shift` is signed:
/// negative for a concave tube, whose spine lies against the normals.
pub fn torus_from_shift(positions: &[DVec3], normals: &[DVec3], idx: &[u32], shift: f64) -> Option<Torus> {
    let n = idx.len();
    if n < 4 {
        return None;
    }
    let shifted: Vec<DVec3> = idx
        .iter()
        .map(|&i| {
            let (p, nrm) = (positions[i as usize], normals[i as usize]);
            DVec3::new(p.x - shift * nrm.x, p.y - shift * nrm.y, p.z - shift * nrm.z)
        })
        .collect();
    let nf = n as f64;
    let (mut mx, mut my, mut mz) = (0.0, 0.0, 0.0);
    for s in &shifted {
        mx += s.x;
        my += s.y;
        mz += s.z;
    }
    mx /= nf;
    my /= nf;
    mz /= nf;
    let (mut cxx, mut cxy, mut cxz, mut cyy, mut cyz, mut czz) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for s in &shifted {
        let (x, y, z) = (s.x - mx, s.y - my, s.z - mz);
        cxx += x * x;
        cxy += x * y;
        cxz += x * z;
        cyy += y * y;
        cyz += y * z;
        czz += z * z;
    }
    let eig = eigen_of_scatter(cxx, cxy, cxz, cyy, cyz, czz);
    if !(eig.values[1] > 1e-9 * js_max(eig.values[2], 1e-12)) {
        return None;
    }
    let axis = eig.vectors[0];
    let (u, v) = ortho_basis(axis);
    let projected: Vec<DVec2> = shifted
        .iter()
        .map(|s| {
            let (x, y, z) = (s.x - mx, s.y - my, s.z - mz);
            DVec2::new(x * u.x + y * u.y + z * u.z, x * v.x + y * v.y + z * v.z)
        })
        .collect();
    let (c, r) = fit_circle_2d(&projected)?;
    if !(r > 0.0) || !r.is_finite() {
        return None;
    }
    Some(Torus {
        center: DVec3::new(mx + c.x * u.x + c.y * v.x, my + c.x * u.y + c.y * v.y, mz + c.x * u.z + c.y * v.z),
        axis,
        major: r,
        minor: shift.abs(),
    })
}

/// The seven parameters of a refinement step: the centre's move, two tilts of the axis in
/// the plane square to it, R's and r's change.
fn apply(base: &Torus, p: &[f64]) -> Torus {
    let (u, v) = ortho_basis(base.axis);
    let mut ax = base.axis.x + p[3] * u.x + p[4] * v.x;
    let mut ay = base.axis.y + p[3] * u.y + p[4] * v.y;
    let mut az = base.axis.z + p[3] * u.z + p[4] * v.z;
    let mut len = hypot3(ax, ay, az);
    if len == 0.0 || len.is_nan() {
        len = 1.0;
    }
    ax /= len;
    ay /= len;
    az /= len;
    Torus {
        center: DVec3::new(base.center.x + p[0], base.center.y + p[1], base.center.z + p[2]),
        axis: DVec3::new(ax, ay, az),
        major: base.major + p[5],
        minor: base.minor + p[6],
    }
}

/// Geometric refinement of all seven parameters by damped Gauss–Newton with a numerical
/// Jacobian: the centre, two tilts of the axis in the plane square to it, R and r.
/// ScanRuler's default for `max_iter` is 40.
pub fn refine_torus_geometric(positions: &[DVec3], idx: &[u32], init: Torus, max_iter: usize) -> Torus {
    const P: usize = 7;
    if idx.len() < 8 {
        return init;
    }
    let mut t = init;
    let mut cost = rms(positions, idx, &t);
    if !cost.is_finite() {
        return init;
    }
    let mut lambda = 1e-6;
    let scale = js_max(js_max(t.major, t.minor), 1e-3);
    for _iter in 0..max_iter {
        let mut jtj = [0.0f64; P * P];
        let mut jtf = [0.0f64; P];
        let mut row = [0.0f64; P];
        let h = 1e-6 * scale;
        let delta_of = |a: usize| if (3..=4).contains(&a) { h / scale } else { h };
        // The Jacobian by central differences, parameter by parameter, over every point:
        // seven perturbed models per iteration.
        let mut plus = Vec::with_capacity(P);
        let mut minus = Vec::with_capacity(P);
        for a in 0..P {
            let mut step = [0.0f64; P];
            step[a] = delta_of(a);
            plus.push(apply(&t, &step));
            step[a] = -step[a];
            minus.push(apply(&t, &step));
        }
        for &i in idx {
            let p = positions[i as usize];
            let e = torus_residual(&t, p);
            if !e.is_finite() {
                continue;
            }
            for a in 0..P {
                row[a] = (torus_residual(&plus[a], p) - torus_residual(&minus[a], p)) / (2.0 * delta_of(a));
            }
            for a in 0..P {
                jtf[a] += row[a] * e;
                for b in a..P {
                    jtj[a * P + b] += row[a] * row[b];
                }
            }
        }
        for a in 0..P {
            for b in 0..a {
                jtj[a * P + b] = jtj[b * P + a];
            }
        }
        let mut improved = false;
        let mut attempt = 0;
        while attempt < 8 && !improved {
            attempt += 1;
            let mut damped = jtj;
            for a in 0..P {
                damped[a * P + a] *= 1.0 + lambda;
            }
            let rhs = jtf.map(|x| -x);
            let Some(delta) = solve_linear(P, &mut damped, &rhs) else {
                lambda *= 10.0;
                continue;
            };
            let next = apply(&t, &delta);
            if !(next.major > 0.0) || !(next.minor > 0.0) {
                lambda *= 10.0;
                continue;
            }
            let next_cost = rms(positions, idx, &next);
            if next_cost < cost {
                let moved = js_hypot(&[delta[0], delta[1], delta[2], delta[5], delta[6]]) + hypot2(delta[3], delta[4]) * scale;
                t = next;
                cost = next_cost;
                lambda = (lambda / 10.0).max(1e-9);
                improved = true;
                if moved < 1e-9 * scale {
                    return t;
                }
            } else {
                lambda *= 10.0;
            }
        }
        if !improved {
            break;
        }
    }
    t
}

/// A clipped torus fit.
#[derive(Debug, Clone, PartialEq)]
pub struct TorusClippedFit {
    pub torus: Torus,
    /// RMS of the residuals over the used points.
    pub sigma: f64,
    /// Peak-to-peak residual over the used points — the tube's form deviation.
    pub span: f64,
    /// The signed residual extremes `span` is the difference of: how far inside and outside
    /// the fitted tube the kept points reach.
    pub min: f64,
    pub max: f64,
    pub used: Vec<u32>,
}

/// Gaussian best fit with GOM-style "used points" clipping (see `clipped_refit`), each
/// round the geometric refinement from the last.
pub fn fit_torus_clipped(positions: &[DVec3], idx: &[u32], k: f64, init: Torus) -> Option<TorusClippedFit> {
    let mut current = init;
    let r = clipped_refit(
        positions,
        idx,
        k,
        |used| {
            let next = refine_torus_geometric(positions, used, current, 40);
            if !(next.major > 0.0) || !(next.minor > 0.0) {
                return None;
            }
            current = next;
            Some(next)
        },
        torus_residual,
    )?;
    Some(TorusClippedFit { torus: r.model, sigma: r.sigma, span: r.span, min: r.min, max: r.max, used: r.used })
}

/// A robust torus estimate and the points it rests on.
#[derive(Debug, Clone, PartialEq)]
pub struct TorusRansac {
    pub torus: Torus,
    pub inliers: Vec<u32>,
    /// Robust noise estimate (1.4826 · median absolute residual), or the refit's σ when
    /// larger.
    pub sigma: f64,
}

/// Robust torus estimate on a local patch: the tube radius searched over a range of scales
/// with either sign, each a spine circle through a random subset of the shifted points,
/// scored by median residual; the winner refined on its consensus set. 320 candidates
/// unless `opts` says otherwise.
pub fn ransac_torus(positions: &[DVec3], normals: &[DVec3], patch: &[u32], opts: RansacOptions) -> Option<TorusRansac> {
    let opts = RansacOptions { iterations: Some(opts.iterations.unwrap_or(320)), seed: opts.seed };
    let core = ransac_consensus(positions, patch, opts, |diag| {
        let n = patch.len();
        let subset = n.min(48);
        let mut call = 0usize;
        RansacShape {
            initial: None,
            generate: move |rand: &mut Mulberry32| {
                // The tube radius on a log scale from a fiftieth to twice the patch's
                // diagonal, the sign alternating; the points a random subset so no one
                // stretch of the patch decides.
                let k = call;
                call += 1;
                let frac = ((k >> 1) % 40) as f64 / 39.0;
                let sign = if k & 1 == 1 { -1.0 } else { 1.0 };
                let shift = sign * diag * 0.02 * 100f64.powf(frac);
                // The circle is a least-squares fit, so its points have to be the round's
                // alone: the patch comes in breadth-first order from the click, and its first
                // few hundred vertices are the surface the click was on — a random pick among
                // those, not a stride across the whole patch, which on a fillet would take the
                // flats along.
                let local = n.min((subset * 2).max(300));
                let idx: Vec<u32> = (0..subset).map(|_| patch[rand.index(local)]).collect();
                let t = torus_from_shift(positions, normals, &idx, shift)?;
                if !(t.major > 1e-6) || t.major > diag * 50.0 {
                    return None;
                }
                // Polished on its own points before it is scored: the shift grid is coarse
                // and the vertex normals of a mesh are averaged, so the raw circle sits a few
                // percent off, and a candidate that close to the round scores a median the
                // flats beside it cannot match.
                let polished = refine_torus_geometric(positions, &idx, t, 8);
                Some(if polished.major > 1e-6 && polished.major <= diag * 50.0 { polished } else { t })
            },
            residual: torus_residual,
        }
    })?;
    let refined = fit_torus_clipped(positions, &core.inliers, 3.0, core.model)?;
    Some(TorusRansac { torus: refined.torus, inliers: refined.used, sigma: js_max(js_max(refined.sigma, core.sigma_est), 1e-9) })
}

/// The torus with its axis pointing the way the majority of the region's normals lean, so
/// two fits of the same round agree in sign.
pub fn canonical_torus(t: Torus, normals: &[DVec3], idx: &[u32]) -> Torus {
    let mut s = 0.0;
    for &i in idx {
        let n = normals[i as usize];
        s += n.x * t.axis.x + n.y * t.axis.y + n.z * t.axis.z;
    }
    if s < 0.0 {
        Torus { axis: -t.axis, ..t }
    } else {
        t
    }
}

/// How much of the tube's cross section a region wraps and how far round the spine it
/// reaches, in degrees.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TorusCoverage {
    pub tube_deg: f64,
    pub spine_deg: f64,
}

/// How much of the tube's cross section the region wraps, in degrees, and how far round the
/// spine it reaches — for the readout. Both counted in 5° bins.
pub fn torus_coverage(t: &Torus, positions: &[DVec3], idx: &[u32]) -> TorusCoverage {
    let (u, v) = ortho_basis(t.axis);
    let mut spine = HashSet::new();
    let mut tube = HashSet::new();
    for &i in idx {
        let d = positions[i as usize] - t.center;
        let a = d.x * t.axis.x + d.y * t.axis.y + d.z * t.axis.z;
        let pu = d.x * u.x + d.y * u.y + d.z * u.z;
        let pv = d.x * v.x + d.y * v.y + d.z * v.z;
        let rho = hypot2(pu, pv);
        spine.insert(js_round((pv.atan2(pu) * 180.0) / std::f64::consts::PI / 5.0) as i64);
        tube.insert(js_round((a.atan2(rho - t.major) * 180.0) / std::f64::consts::PI / 5.0) as i64);
    }
    TorusCoverage { tube_deg: (tube.len() as f64 * 5.0).min(360.0), spine_deg: (spine.len() as f64 * 5.0).min(360.0) }
}

#[cfg(test)]
mod tests {
    // torusFit.test.ts — 'the torus fit'. Its other two blocks run the seed-to-torus
    // pipeline on a mesh graph (fitTorusFromSeed), which is not part of these fits.
    use super::*;
    use crate::fit3d::testkit::{p32, unit};

    const PI: f64 = std::f64::consts::PI;

    fn basis(d: DVec3) -> (DVec3, DVec3) {
        let helper = if d.x.abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
        let u = unit([d.y * helper[2] - d.z * helper[1], d.z * helper[0] - d.x * helper[2], d.x * helper[1] - d.y * helper[0]]);
        let v = unit([d.y * u.z - d.z * u.y, d.z * u.x - d.x * u.z, d.x * u.y - d.y * u.x]);
        (u, v)
    }

    /// Points on a torus with exact outward normals, over a stretch of the spine
    /// (`spine_arc`) and of the tube (`tube_from` … `tube_to`), noisy along the normal.
    #[allow(clippy::too_many_arguments)]
    fn sample_torus(n: usize, t: &Torus, noise: f64, spine_arc: f64, tube_from: f64, tube_to: f64, seed: u32) -> (Vec<DVec3>, Vec<DVec3>) {
        let mut rand = Mulberry32::new(seed);
        let d = unit(t.axis.to_array());
        let (u, v) = basis(d);
        let mut positions = Vec::with_capacity(n);
        let mut normals = Vec::with_capacity(n);
        for _ in 0..n {
            let a = (rand.next_f64() - 0.5) * spine_arc;
            let b = tube_from + rand.next_f64() * (tube_to - tube_from);
            let rx = a.cos() * u.x + a.sin() * v.x;
            let ry = a.cos() * u.y + a.sin() * v.y;
            let rz = a.cos() * u.z + a.sin() * v.z;
            let nx = b.cos() * rx + b.sin() * d.x;
            let ny = b.cos() * ry + b.sin() * d.y;
            let nz = b.cos() * rz + b.sin() * d.z;
            let h = t.minor + rand.gaussian() * noise;
            positions.push(p32(t.center.x + t.major * rx + h * nx, t.center.y + t.major * ry + h * ny, t.center.z + t.major * rz + h * nz));
            normals.push(p32(nx, ny, nz));
        }
        (positions, normals)
    }

    fn all(n: usize) -> Vec<u32> {
        (0..n as u32).collect()
    }

    /// Off-axis on purpose, and off the origin.
    fn truth() -> Torus {
        Torus { center: DVec3::new(3.0, -2.0, 5.0), axis: unit([0.3, 0.5, 0.8]), major: 20.0, minor: 4.0 }
    }

    fn agrees(t: &Torus, truth: &Torus, tol: f64) {
        assert!((t.major - truth.major).abs() < tol, "R {} against {}", t.major, truth.major);
        assert!((t.minor - truth.minor).abs() < tol, "r {} against {}", t.minor, truth.minor);
        assert!((t.axis.dot(truth.axis).abs() - 1.0).abs() < 1e-4);
        assert!((t.center - truth.center).length() < tol);
    }

    #[test]
    fn reads_a_signed_distance_and_a_spine_through_the_shifted_points_recovers_the_torus_exactly() {
        let truth = truth();
        let (positions, normals) = sample_torus(600, &truth, 0.0, 2.0 * PI, -PI, PI, 5);
        for p in &positions[..20] {
            assert!(torus_residual(&truth, *p).abs() < 1e-4);
        }
        // A point a millimetre out along the normal reads +1.
        let p = positions[0] + normals[0];
        assert!((torus_residual(&truth, p) - 1.0).abs() < 0.5e-3);
        let t = torus_from_shift(&positions, &normals, &all(600), truth.minor).expect("a torus");
        agrees(&t, &truth, 1e-3);
        // The wrong shift is not the torus: the residual says so.
        let wrong = torus_from_shift(&positions, &normals, &all(600), truth.minor * 2.0).expect("a torus");
        let worst = positions.iter().map(|&p| torus_residual(&wrong, p).abs()).fold(0.0, f64::max);
        assert!(worst > 1.0);
    }

    #[test]
    fn refines_a_rough_start_to_the_noise_on_a_whole_ring_and_the_robust_estimate_needs_no_start() {
        let truth = truth();
        let (positions, normals) = sample_torus(4000, &truth, 0.02, 2.0 * PI, -PI, PI, 5);
        let rough = Torus {
            center: truth.center + DVec3::new(0.5, 0.0, 0.0),
            axis: unit([truth.axis.x + 0.05, truth.axis.y - 0.04, truth.axis.z]),
            major: truth.major * 1.05,
            minor: truth.minor * 0.9,
        };
        let refined = refine_torus_geometric(&positions, &all(4000), rough, 40);
        agrees(&refined, &truth, 0.01);
        let clipped = fit_torus_clipped(&positions, &all(4000), 3.0, rough).expect("a fit");
        assert!(clipped.sigma < 0.025);
        agrees(&clipped.torus, &truth, 0.01);
        let robust = ransac_torus(&positions, &normals, &all(4000), RansacOptions { seed: Some(3), ..Default::default() }).expect("a torus");
        agrees(&robust.torus, &truth, 0.02);
    }

    #[test]
    fn finds_a_fillets_quarter_round_from_a_stretch_of_it_a_concave_one_too() {
        // A quarter of the tube over a sixth of the spine: a fillet on the edge of a round
        // boss.
        let truth = truth();
        let (positions, normals) = sample_torus(3000, &truth, 0.01, PI / 3.0, 0.0, PI / 2.0, 5);
        let opts = RansacOptions { seed: Some(9), ..Default::default() };
        let robust = ransac_torus(&positions, &normals, &all(3000), opts).expect("a torus");
        agrees(&robust.torus, &truth, 0.05);
        // The same round scanned from inside: the normals point into the tube.
        let inward: Vec<DVec3> = normals.iter().map(|n| -*n).collect();
        let concave = ransac_torus(&positions, &inward, &all(3000), opts).expect("a torus");
        agrees(&concave.torus, &truth, 0.05);
    }

    // Not in ScanRuler's tests: the readout helpers on a quarter round.
    #[test]
    fn coverage_and_canonical_axis_of_a_quarter_round() {
        let truth = truth();
        let (positions, normals) = sample_torus(3000, &truth, 0.0, PI / 3.0, 0.0, PI / 2.0, 5);
        let cover = torus_coverage(&truth, &positions, &all(3000));
        assert!(cover.spine_deg > 55.0 && cover.spine_deg < 70.0);
        assert!(cover.tube_deg > 85.0 && cover.tube_deg < 100.0);
        let flipped = Torus { axis: -truth.axis, ..truth };
        assert_eq!(canonical_torus(flipped, &normals, &all(3000)).axis, truth.axis);
        assert!((torus_normal_align(&truth, positions[0], normals[0]) - 1.0).abs() < 1e-6);
    }
}
