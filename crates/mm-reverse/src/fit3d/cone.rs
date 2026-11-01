// SPDX-License-Identifier: AGPL-3.0-only
//! The cone: a surface of revolution whose radius changes steadily along the axis. It is
//! described by a point on the axis and the radius there, not by the apex, so nothing blows
//! up as the half-angle goes to zero and the cone becomes a cylinder. The fit follows the
//! cylinder's — an axis from the normals (a cone's all meet the axis at one angle, so their
//! tips lie in a plane), a circle across it, the taper read off a line of radius against
//! axial position — with three-point-and-normal candidates scored by median residual
//! (LMedS) and damped Gauss–Newton over all six parameters to finish.
//!
//! Ported from ScanRuler (`src/core/fit/cone.ts`), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only.

use glam::DVec3;

use super::clip::clipped_refit;
use super::linalg3::{centroid_of, eigen_of_scatter, hypot2, hypot3, js_max, js_min, normalize_or_z, ortho_basis, project_onto_plane, rms_over};
use super::ransac::{ransac_consensus, RansacOptions, RansacShape};
use super::types::Cone;
use crate::fit2d::{fit_circle_2d, solve_linear, Mulberry32};

/// A point read against a cone: its orthogonal distance from the surface (positive
/// outside), the unit radial direction, the axial position and the radial distance there.
#[derive(Debug, Clone, Copy)]
struct Slant {
    e: f64,
    w: DVec3,
    t: f64,
    rho: f64,
}

/// Orthogonal distance of a point from the cone surface (positive outside), plus the unit
/// radial direction, axial position and radial distance there. The cone is anchored at a
/// point on the axis — not at the apex — so nothing blows up as the half-angle approaches
/// zero and the shape becomes a cylinder. In the (t, rho) half-plane the surface is the line
/// rho = r + t·tan(phi), and the point-to-line distance is cos(phi)·(rho − r) − sin(phi)·t.
/// NaN on the axis, where the radial direction is undefined.
fn slant(c: &Cone, p: DVec3) -> Slant {
    let q = p - c.point;
    let t = q.x * c.axis.x + q.y * c.axis.y + q.z * c.axis.z;
    let w = DVec3::new(q.x - t * c.axis.x, q.y - t * c.axis.y, q.z - t * c.axis.z);
    let rho = (w.x * w.x + w.y * w.y + w.z * w.z).sqrt();
    if !(rho > 1e-12) {
        return Slant { e: f64::NAN, w: DVec3::ZERO, t, rho };
    }
    let e = c.phi.cos() * (rho - c.r) - c.phi.sin() * t;
    Slant { e, w: DVec3::new(w.x / rho, w.y / rho, w.z / rho), t, rho }
}

/// The signed orthogonal distance of a point from the cone surface (positive outside); NaN
/// on the axis.
pub fn cone_residual(c: &Cone, p: DVec3) -> f64 {
    slant(c, p).e
}

/// |cos| between a vertex normal and the cone's surface normal at that point — the
/// membership test region growing uses to stay on the conical face.
pub fn cone_normal_align(c: &Cone, p: DVec3, n: DVec3) -> f64 {
    cone_normal_signed(c, p, n).abs()
}

/// The same cosine signed: positive where the vertex normal points away from the axis (a
/// pin's tip), negative toward it (a countersink).
pub fn cone_normal_signed(c: &Cone, p: DVec3, n: DVec3) -> f64 {
    let Slant { w, .. } = slant(c, p);
    if w == DVec3::ZERO {
        return 0.0;
    }
    let cos = c.phi.cos();
    let sin = c.phi.sin();
    // Surface normal: the radial direction tilted back by the half-angle.
    (cos * w.x - sin * c.axis.x) * n.x + (cos * w.y - sin * c.axis.y) * n.y + (cos * w.z - sin * c.axis.z) * n.z
}

fn rms(positions: &[DVec3], idx: &[u32], c: &Cone) -> f64 {
    rms_over(positions, idx, |p| slant(c, p).e)
}

/// The radius grows along the +axis by convention; a fit that walked the half-angle
/// negative is the same cone with the axis reversed.
fn canonical(c: Cone) -> Cone {
    if c.phi >= 0.0 {
        return c;
    }
    Cone { axis: -c.axis, phi: -c.phi, ..c }
}

/// Axis direction from surface normals. A cone's normals all make the same angle with its
/// axis, so their tips lie in one plane — the plane's normal is the axis. (For a cylinder
/// that plane passes through the origin: the same estimate, so this degrades gracefully
/// toward zero half-angle.)
pub fn cone_axis_from_normals(normals: &[DVec3], idx: &[u32]) -> Option<DVec3> {
    let n = idx.len();
    if n < 3 {
        return None;
    }
    let mean = centroid_of(normals, idx);
    let (mut cxx, mut cxy, mut cxz, mut cyy, mut cyz, mut czz) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for &i in idx {
        let q = normals[i as usize] - mean;
        cxx += q.x * q.x;
        cxy += q.x * q.y;
        cxz += q.x * q.z;
        cyy += q.y * q.y;
        cyz += q.y * q.z;
        czz += q.z * q.z;
    }
    let eig = eigen_of_scatter(cxx, cxy, cxz, cyy, cyz, czz);
    // A patch that curves too little (an almost-flat sliver) leaves the plane through the
    // normal tips undetermined — the axis would be a coin toss.
    if !(eig.values[1] > 1e-6 * n as f64) {
        return None;
    }
    Some(eig.vectors[0])
}

/// Best cone for a fixed axis direction: place the axis with the shared 2D circle kernel in
/// the plane across it (a cone projects to an annulus whose middle circle is centred on the
/// axis), then read the taper straight off a line fit of radial distance against axial
/// position.
pub fn fit_cone_on_axis(positions: &[DVec3], idx: &[u32], axis: DVec3) -> Option<Cone> {
    let n = idx.len();
    if n < 6 {
        return None;
    }
    let d = normalize_or_z(axis);
    let (u, v) = ortho_basis(d);
    let centre = centroid_of(positions, idx);

    // Project onto the (u, v) plane through the centroid.
    let projected = project_onto_plane(positions, idx, centre, u, v);
    let (c, _) = fit_circle_2d(&projected)?;
    let (cu, cv) = (c.x, c.y);

    // rho against t is a straight line on a cone: rho = r + t·tan(phi).
    let (mut st, mut stt, mut sr, mut str) = (0.0, 0.0, 0.0, 0.0);
    for (k, &i) in idx.iter().enumerate() {
        let q = positions[i as usize] - centre;
        let t = q.x * d.x + q.y * d.y + q.z * d.z;
        let rho = hypot2(projected[k].x - cu, projected[k].y - cv);
        st += t;
        stt += t * t;
        sr += rho;
        str += t * rho;
    }
    let nf = n as f64;
    let det = nf * stt - st * st;
    // All points at one axial position — a circle slice — cannot say how the radius changes
    // along the axis; call it straight and let refinement decide.
    let mut tan = 0.0;
    let mut r = sr / nf;
    if det > 1e-12 * js_max(1.0, stt) * nf {
        tan = (nf * str - st * sr) / det;
        r = (sr - tan * st) / nf;
    }
    if !r.is_finite() || !(r > 0.0) || !tan.is_finite() {
        return None;
    }

    Some(canonical(Cone {
        point: DVec3::new(centre.x + cu * u.x + cv * v.x, centre.y + cu * u.y + cv * v.y, centre.z + cu * u.z + cv * v.z),
        axis: d,
        r,
        phi: tan.atan(),
    }))
}

/// Geometric (orthogonal-distance) refinement of all six cone degrees of freedom by damped
/// Gauss–Newton — the cylinder's five parameters plus the half-angle. The axis point slides
/// in the plane it is normal to and the direction is perturbed by two angles in that same
/// plane, exactly as for the cylinder, so the parametrisation never leaves the unit sphere
/// and the slide-along-the-axis degeneracy is simply not present.
fn refine_cone_geometric(positions: &[DVec3], idx: &[u32], init: Cone, max_iter: usize) -> Cone {
    if idx.len() < 7 {
        return init;
    }
    let mut c = init;
    let mut cost = rms(positions, idx, &c);
    if !cost.is_finite() {
        return init;
    }
    let mut lambda = 1e-6;

    for _iter in 0..max_iter {
        let (u, v) = ortho_basis(c.axis);
        let cos = c.phi.cos();
        let sin = c.phi.sin();
        let mut jtj = [0.0f64; 36];
        let mut jtf = [0.0f64; 6];
        let mut row = [0.0f64; 6];
        let mut m = 0usize;

        for &i in idx {
            let Slant { e, w, t, rho } = slant(&c, positions[i as usize]);
            if !e.is_finite() {
                continue;
            }
            let su = w.dot(u);
            let sv = w.dot(v);
            let tilt = t * cos + rho * sin;
            row[0] = -cos * su;
            row[1] = -cos * sv;
            row[2] = -tilt * su;
            row[3] = -tilt * sv;
            row[4] = -cos;
            row[5] = -(sin * (rho - c.r) + cos * t);
            for a in 0..6 {
                jtf[a] += row[a] * e;
                for b in a..6 {
                    jtj[a * 6 + b] += row[a] * row[b];
                }
            }
            m += 1;
        }
        if m < 7 {
            break;
        }
        for a in 0..6 {
            for b in 0..a {
                jtj[a * 6 + b] = jtj[b * 6 + a];
            }
        }

        let mut applied = false;
        let mut converged = false;
        for _attempt in 0..6 {
            let mut mat = jtj;
            for a in 0..6 {
                mat[a * 6 + a] *= 1.0 + lambda;
            }
            let rhs = jtf.map(|x| -x);
            let Some(step) = solve_linear(6, &mut mat, &rhs) else {
                lambda *= 10.0;
                continue;
            };
            let axis = normalize_or_z(DVec3::new(
                c.axis.x + step[2] * u.x + step[3] * v.x,
                c.axis.y + step[2] * u.y + step[3] * v.y,
                c.axis.z + step[2] * u.z + step[3] * v.z,
            ));
            let cand = Cone {
                point: DVec3::new(
                    c.point.x + step[0] * u.x + step[1] * v.x,
                    c.point.y + step[0] * u.y + step[1] * v.y,
                    c.point.z + step[0] * u.z + step[1] * v.z,
                ),
                axis,
                r: c.r + step[4],
                phi: c.phi + step[5],
            };
            // Past about 85° the shape is a flat face and the parametrisation loses its
            // meaning; a negative radius means the step overshot the apex.
            if !(cand.r > 0.0) || !cand.point.x.is_finite() || !(cand.phi.abs() < 1.48) {
                lambda *= 10.0;
                continue;
            }
            let cand_cost = rms(positions, idx, &cand);
            if cand_cost.is_finite() && cand_cost <= cost {
                converged = cost - cand_cost < 1e-12 * js_max(cost, cand.r);
                c = cand;
                cost = cand_cost;
                lambda = (lambda * 0.3).max(1e-9);
                applied = true;
                break;
            }
            lambda *= 10.0;
        }
        if !applied || converged {
            break;
        }
    }
    canonical(c)
}

/// A clipped cone fit.
#[derive(Debug, Clone, PartialEq)]
pub struct ClippedConeFit {
    pub cone: Cone,
    /// RMS of the slant-normal residuals over the used points.
    pub sigma: f64,
    /// Peak-to-peak residual — the surface's conicity.
    pub span: f64,
    /// The signed extremes `span` is the difference of: how far inside and outside the
    /// fitted wall the kept points reach, along the slant normal.
    pub min: f64,
    pub max: f64,
    pub used: Vec<u32>,
}

/// Gaussian best fit with GOM-style "used points" clipping (see `clipped_refit`). Each round
/// re-seeds on the previous round's axis and then refines geometrically, so the axis carries
/// over between clipping rounds instead of restarting from `init`.
pub fn fit_cone_clipped(positions: &[DVec3], idx: &[u32], init: Cone, k: f64) -> Option<ClippedConeFit> {
    let mut start = init;
    let r = clipped_refit(
        positions,
        idx,
        k,
        |used| {
            let seeded = fit_cone_on_axis(positions, used, start.axis).unwrap_or(start);
            let c = refine_cone_geometric(positions, used, seeded, 60);
            if !(c.r > 0.0) || !c.r.is_finite() || !c.phi.is_finite() {
                return None;
            }
            start = c;
            Some(c)
        },
        cone_residual,
    )?;
    Some(ClippedConeFit { cone: r.model, sigma: r.sigma, span: r.span, min: r.min, max: r.max, used: r.used })
}

/// A robust cone estimate and the points it rests on.
#[derive(Debug, Clone, PartialEq)]
pub struct RansacConeResult {
    pub cone: Cone,
    pub inliers: Vec<u32>,
    /// Robust noise estimate (1.4826 · median absolute residual), or the refit's σ when
    /// larger.
    pub sigma: f64,
}

/// Cone through three points with known surface normals: the apex is where the three
/// tangent planes meet, the axis is normal to the plane through the three unit
/// apex-to-point directions (they all make the half-angle with the axis, so they lie on one
/// circle of directions), and the half-angle is the mean of those three angles.
fn cone_from3(positions: &[DVec3], normals: &[DVec3], i0: u32, i1: u32, i2: u32) -> Option<Cone> {
    let ids = [i0, i1, i2];
    let mut a = [0.0f64; 9];
    let mut b = [0.0f64; 3];
    for (r, &id) in ids.iter().enumerate() {
        let (n, p) = (normals[id as usize], positions[id as usize]);
        a[r * 3] = n.x;
        a[r * 3 + 1] = n.y;
        a[r * 3 + 2] = n.z;
        b[r] = n.x * p.x + n.y * p.y + n.z * p.z;
    }
    let apex = solve_linear(3, &mut a, &b)?;
    let apex = DVec3::new(apex[0], apex[1], apex[2]);

    let mut dirs = [DVec3::ZERO; 3];
    for (k, &id) in ids.iter().enumerate() {
        let d = positions[id as usize] - apex;
        let len = hypot3(d.x, d.y, d.z);
        if !(len > 1e-9) {
            return None;
        }
        dirs[k] = DVec3::new(d.x / len, d.y / len, d.z / len);
    }
    let axis_raw = (dirs[1] - dirs[0]).cross(dirs[2] - dirs[0]);
    // Directions too alike leave the circle through them — the axis — anywhere.
    if !(hypot3(axis_raw.x, axis_raw.y, axis_raw.z) > 0.02) {
        return None;
    }
    let mut d = normalize_or_z(axis_raw);
    let mut dot_sum = dirs[0].dot(d);
    dot_sum += dirs[1].dot(d);
    dot_sum += dirs[2].dot(d);
    if dot_sum < 0.0 {
        d = -d;
    }
    let phi = js_min(1.0, dot_sum.abs() / 3.0).acos();
    if !(phi > 0.005) || !(phi < 1.48) {
        return None;
    }

    // Anchor the axis at the samples' mean axial position, where the region is.
    let mut t_mean = 0.0;
    for &id in &ids {
        let q = positions[id as usize] - apex;
        t_mean += q.x * d.x + q.y * d.y + q.z * d.z;
    }
    t_mean /= 3.0;
    let r = t_mean * phi.tan();
    if !(r > 0.0) || !r.is_finite() {
        return None;
    }
    Some(Cone { point: DVec3::new(apex.x + t_mean * d.x, apex.y + t_mean * d.y, apex.z + t_mean * d.z), axis: d, r, phi })
}

/// Robust cone estimate on a local patch: score the normal-plane axis plus a batch of
/// three-point-with-normals candidates by median absolute residual (LMedS), then refine on
/// the consensus set.
pub fn ransac_cone(positions: &[DVec3], normals: &[DVec3], patch: &[u32], opts: RansacOptions) -> Option<RansacConeResult> {
    let core = ransac_consensus(positions, patch, opts, |diag| {
        let n = patch.len();
        // A radius wildly out of scale with the local patch is really a plane, never the cone
        // the user clicked.
        let plausible = move |c: &Cone| c.r > diag * 0.005 && c.r <= diag * 60.0 && c.phi.abs() < 1.48;
        let from_normals = cone_axis_from_normals(normals, patch).and_then(|axis| fit_cone_on_axis(positions, patch, axis));
        RansacShape {
            initial: from_normals.filter(|c| plausible(c)),
            generate: move |rand: &mut Mulberry32| {
                let i0 = patch[rand.index(n)];
                let i1 = patch[rand.index(n)];
                let i2 = patch[rand.index(n)];
                if i0 == i1 || i0 == i2 || i1 == i2 {
                    return None;
                }
                cone_from3(positions, normals, i0, i1, i2).filter(|c| plausible(c))
            },
            residual: cone_residual,
        }
    })?;

    let refined = fit_cone_clipped(positions, &core.inliers, core.model, 3.0)?;
    Some(RansacConeResult { cone: refined.cone, inliers: refined.used, sigma: js_max(js_max(refined.sigma, core.sigma_est), 1e-9) })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit3d::testkit::{all_indices, p32, rad, sample_cone, unit, v32};

    const POINT: [f64; 3] = [4.0, -3.0, 2.0];
    // Off-axis on purpose, so an axis-aligned bug cannot pass by accident.
    const AXIS: [f64; 3] = [0.4, 0.6, 0.6928];
    const R: f64 = 6.35;
    const LENGTH: f64 = 30.0;
    const TAU: f64 = 2.0 * std::f64::consts::PI;

    fn phi() -> f64 {
        rad(15.0)
    }

    /// Exact outward normals for points sampled on the cone wall.
    fn wall_normals(points: &[DVec3], point: [f64; 3], axis: [f64; 3], phi: f64) -> Vec<DVec3> {
        let d = unit(axis);
        let (cos, sin) = (phi.cos(), phi.sin());
        let p0 = DVec3::from_array(point);
        points
            .iter()
            .map(|&p| {
                let q = p - p0;
                let t = q.x * d.x + q.y * d.y + q.z * d.z;
                let w = DVec3::new(q.x - t * d.x, q.y - t * d.y, q.z - t * d.z);
                let l = (w.x * w.x + w.y * w.y + w.z * w.z).sqrt();
                p32((cos * w.x) / l - sin * d.x, (cos * w.y) / l - sin * d.y, (cos * w.z) / l - sin * d.z)
            })
            .collect()
    }

    /// Distance from the true axis to the fitted one, measured at the sample's centre.
    fn axis_offset(c: &Cone) -> f64 {
        let q = DVec3::from_array(POINT) - c.point;
        let t = q.dot(c.axis);
        (q - c.axis * t).length()
    }

    /// The fitted surface radius at the true anchor's axial position.
    fn radius_at(c: &Cone) -> f64 {
        let t = (DVec3::from_array(POINT) - c.point).dot(c.axis);
        c.r + t * c.phi.tan()
    }

    fn tilt(c: &Cone) -> f64 {
        c.axis.dot(unit(AXIS)).abs().min(1.0).acos().to_degrees()
    }

    // coneFit.test.ts — 'Gaussian cone fit'.
    #[test]
    fn recovers_radius_half_angle_and_axis_from_a_10_degree_off_starting_guess() {
        let pts = sample_cone(20_000, POINT, AXIS, R, phi(), LENGTH, 0.02, 42, TAU);
        let bad = unit([AXIS[0] + 0.18, AXIS[1] - 0.1, AXIS[2]]);
        let idx = all_indices(20_000);
        let init = fit_cone_on_axis(&pts, &idx, bad).expect("a start");
        let fit = fit_cone_clipped(&pts, &idx, init, 3.0).expect("a fit");
        assert!((radius_at(&fit.cone) - R).abs() < 0.005);
        assert!((fit.cone.phi - phi()).abs() < 0.001);
        assert!(tilt(&fit.cone) < 0.05);
        assert!(axis_offset(&fit.cone) < 0.005);
        assert!(fit.sigma > 0.015);
        assert!(fit.sigma < 0.025);
    }

    #[test]
    fn orients_the_axis_toward_the_growing_radius() {
        let pts = sample_cone(10_000, POINT, AXIS, R, phi(), LENGTH, 0.01, 42, TAU);
        let idx = all_indices(10_000);
        let init = fit_cone_on_axis(&pts, &idx, unit([-AXIS[0], -AXIS[1], -AXIS[2]])).expect("a start");
        let fit = fit_cone_clipped(&pts, &idx, init, 3.0).expect("a fit");
        assert!(fit.cone.phi > 0.0);
        assert!(fit.cone.axis.dot(unit(AXIS)) > 0.999);
    }

    #[test]
    fn recovers_a_140_degree_arc() {
        let pts = sample_cone(20_000, POINT, AXIS, R, phi(), LENGTH, 0.02, 43, rad(140.0));
        let idx = all_indices(20_000);
        let init = fit_cone_on_axis(&pts, &idx, unit([AXIS[0] + 0.1, AXIS[1], AXIS[2]])).expect("a start");
        let fit = fit_cone_clipped(&pts, &idx, init, 3.0).expect("a fit");
        assert!((radius_at(&fit.cone) - R).abs() < 0.05);
        assert!((fit.cone.phi - phi()).abs() < 0.01);
        assert!(tilt(&fit.cone) < 0.3);
        assert!(axis_offset(&fit.cone) < 0.05);
    }

    #[test]
    fn degrades_to_a_cylinder_zero_taper_comes_back_as_zero_half_angle() {
        // A cone fit on cylinder data must not invent a taper.
        let pts = sample_cone(20_000, POINT, AXIS, R, 0.0, LENGTH, 0.02, 46, TAU);
        let idx = all_indices(20_000);
        let init = fit_cone_on_axis(&pts, &idx, DVec3::from_array(AXIS)).expect("a start");
        let fit = fit_cone_clipped(&pts, &idx, init, 3.0).expect("a fit");
        assert!(fit.cone.phi.abs() < 0.002);
        assert!((radius_at(&fit.cone) - R).abs() < 0.01);
    }

    #[test]
    fn sigma_clipping_sheds_sparse_gross_outliers() {
        let mut pts = sample_cone(20_000, POINT, AXIS, R, phi(), LENGTH, 0.02, 44, TAU);
        let normals = wall_normals(&pts, POINT, AXIS, phi());
        let mut rand = Mulberry32::new(99);
        for _ in 0..400 {
            let j = rand.index(20_000);
            let (p, n) = (pts[j], normals[j]);
            pts[j] = p32(p.x + n.x * 0.5, p.y + n.y * 0.5, p.z + n.z * 0.5);
        }
        let idx = all_indices(20_000);
        let init = fit_cone_on_axis(&pts, &idx, DVec3::from_array(AXIS)).expect("a start");
        let clipped = fit_cone_clipped(&pts, &idx, init, 3.0).expect("a fit");
        assert!((radius_at(&clipped.cone) - R).abs() < 0.01);
        assert!(clipped.used.len() < 20_000);
    }

    // coneFit.test.ts — 'RANSAC cone estimate'.
    #[test]
    fn finds_the_cone_despite_30_percent_structured_outliers() {
        let mut all = sample_cone(3_000, POINT, AXIS, R, phi(), LENGTH, 0.02, 45, rad(200.0));
        let mut all_n = wall_normals(&all, POINT, AXIS, phi());
        // Fake shoulder contamination: a flat disc of points off the end.
        let mut rand = Mulberry32::new(7);
        let d = unit(AXIS);
        for _ in 0..1_300 {
            let x = POINT[0] + (rand.next_f64() - 0.5) * 20.0 + d.x * (LENGTH / 2.0);
            let y = POINT[1] + (rand.next_f64() - 0.5) * 20.0 + d.y * (LENGTH / 2.0);
            let z = POINT[2] + (rand.next_f64() - 0.5) * 20.0 + d.z * (LENGTH / 2.0);
            all.push(p32(x, y, z));
            all_n.push(v32(d));
        }
        let res = ransac_cone(&all, &all_n, &all_indices(4_300), RansacOptions { seed: Some(1), ..Default::default() }).expect("a cone");
        assert!((radius_at(&res.cone) - R).abs() < 0.1);
        assert!((res.cone.phi - phi()).abs() < 0.02);
        assert!(tilt(&res.cone) < 1.0);
        assert!(axis_offset(&res.cone) < 0.1);
        // The disc must not have been swept into the consensus set — bar the few of its
        // points that happen to sit on the ring where it crosses the wall.
        let from_disc = res.inliers.iter().filter(|&&i| i >= 3_000).count();
        assert!((from_disc as f64) < 0.03 * res.inliers.len() as f64);
    }

    #[test]
    fn rejects_a_patch_with_no_curvature_to_work_with() {
        // A 4-degree sliver of a huge cone is indistinguishable from a plane.
        let sliver = sample_cone(2_000, POINT, AXIS, 500.0, phi(), 20.0, 0.02, 8, rad(4.0));
        let normals = wall_normals(&sliver, POINT, AXIS, phi());
        let res = ransac_cone(&sliver, &normals, &all_indices(2_000), RansacOptions { seed: Some(2), ..Default::default() });
        assert!(res.is_none());
    }

    // coneNormalAlign / coneNormalSigned: a pin's tip faces away from the axis, a
    // countersink towards it.
    #[test]
    fn signed_normal_tells_a_pin_from_a_countersink() {
        let c = Cone { point: DVec3::ZERO, axis: DVec3::Z, r: 5.0, phi: rad(45.0) };
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let p = DVec3::new(5.0, 0.0, 0.0);
        assert!((cone_normal_signed(&c, p, DVec3::new(s, 0.0, -s)) - 1.0).abs() < 1e-12);
        assert!((cone_normal_signed(&c, p, DVec3::new(-s, 0.0, s)) + 1.0).abs() < 1e-12);
        assert!((cone_normal_align(&c, p, DVec3::new(-s, 0.0, s)) - 1.0).abs() < 1e-12);
        assert_eq!(cone_normal_signed(&c, DVec3::new(0.0, 0.0, 3.0), DVec3::X), 0.0);
    }
}
