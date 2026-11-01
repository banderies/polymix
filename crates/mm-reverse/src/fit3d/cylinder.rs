// SPDX-License-Identifier: AGPL-3.0-only
//! The cylinder, found from its normals first: every normal of a cylinder is square to its
//! axis, so the axis is the direction the normal cloud scatters least along, and the
//! position and radius are then a circle fit in the plane across it. Candidates from two
//! points with their normals compete with that estimate by median residual (LMedS); damped
//! Gauss–Newton refines all five degrees of freedom, the axis tilted by two angles so it
//! stays a unit vector and the axis point never wanders along the axis itself.
//!
//! Ported from ScanRuler (`src/core/fit/cylinder.ts`), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only.

use glam::DVec3;

use super::clip::clipped_refit;
use super::linalg3::{centroid_of, eigen_of_scatter, hypot2, hypot3, js_max, js_min, normalize_or_z, ortho_basis, project_onto_plane, rms_over};
use super::plane::{curved_beyond_plane, strided};
use super::ransac::{ransac_consensus, RansacOptions, RansacShape};
use super::types::Cylinder;
use crate::fit2d::{fit_circle_2d, solve_linear, Mulberry32};

/// A point read against a cylinder: its distance from the surface (positive outside), the
/// unit radial direction there and its axial position.
#[derive(Debug, Clone, Copy)]
struct Radial {
    e: f64,
    w: DVec3,
    t: f64,
}

/// Distance of a point from the cylinder surface (positive outside), and the unit radial
/// direction there. NaN on the axis, where the radial direction is undefined.
fn radial(c: &Cylinder, p: DVec3) -> Radial {
    let q = p - c.point;
    let t = q.x * c.axis.x + q.y * c.axis.y + q.z * c.axis.z;
    let w = DVec3::new(q.x - t * c.axis.x, q.y - t * c.axis.y, q.z - t * c.axis.z);
    let rho = (w.x * w.x + w.y * w.y + w.z * w.z).sqrt();
    if !(rho > 1e-12) {
        return Radial { e: f64::NAN, w: DVec3::ZERO, t };
    }
    Radial { e: rho - c.r, w: DVec3::new(w.x / rho, w.y / rho, w.z / rho), t }
}

/// The signed distance of a point from the cylinder surface (positive outside); NaN on the
/// axis.
pub fn cylinder_residual(c: &Cylinder, p: DVec3) -> f64 {
    radial(c, p).e
}

fn rms(positions: &[DVec3], idx: &[u32], c: &Cylinder) -> f64 {
    rms_over(positions, idx, |p| radial(c, p).e)
}

/// Axis direction from surface normals. Every normal of a cylinder is perpendicular to its
/// axis, so the axis is the direction the normal cloud scatters least along — a far better
/// starting guess than anything derived from the point positions alone.
pub fn axis_from_normals(normals: &[DVec3], idx: &[u32]) -> Option<DVec3> {
    let n = idx.len();
    if n < 3 {
        return None;
    }
    let (mut cxx, mut cxy, mut cxz, mut cyy, mut cyz, mut czz) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for &i in idx {
        let v = normals[i as usize];
        cxx += v.x * v.x;
        cxy += v.x * v.y;
        cxz += v.x * v.z;
        cyy += v.y * v.y;
        cyz += v.y * v.z;
        czz += v.z * v.z;
    }
    let eig = eigen_of_scatter(cxx, cxy, cxz, cyy, cyz, czz);
    // A patch that curves too little (an almost-flat sliver) leaves the two smallest
    // directions indistinguishable — the axis would be a coin toss.
    if !(eig.values[1] > 1e-6 * n as f64) {
        return None;
    }
    Some(eig.vectors[0])
}

/// Best circle in the plane perpendicular to a fixed axis — that is the cylinder's position
/// and radius for that direction. Algebraic (Coope) fit followed by the 2D form of the
/// orthogonal-distance fixed-point iteration.
pub fn fit_cylinder_on_axis(positions: &[DVec3], idx: &[u32], axis: DVec3) -> Option<Cylinder> {
    if idx.len() < 5 {
        return None;
    }
    let d = normalize_or_z(axis);
    let (u, v) = ortho_basis(d);
    let centre = centroid_of(positions, idx);

    // Project onto the (u, v) plane through the centroid; the circle itself is the shared
    // 2D kernel's job.
    let projected = project_onto_plane(positions, idx, centre, u, v);
    let (c, r) = fit_circle_2d(&projected)?;
    let (cu, cv) = (c.x, c.y);

    Some(Cylinder {
        point: DVec3::new(centre.x + cu * u.x + cv * v.x, centre.y + cu * u.y + cv * v.y, centre.z + cu * u.z + cv * v.z),
        axis: d,
        r,
    })
}

/// Geometric (orthogonal-distance) refinement of all five cylinder degrees of freedom by
/// damped Gauss–Newton. The axis direction is perturbed inside the plane it is normal to
/// (two angles), which keeps it a unit vector without a constraint equation, and the axis
/// point slides in the same plane — the two degrees of freedom that would otherwise just
/// walk along the axis are simply not in the parametrisation. ScanRuler's default for
/// `max_iter` is 60.
pub fn refine_cylinder_geometric(positions: &[DVec3], idx: &[u32], init: Cylinder, max_iter: usize) -> Cylinder {
    if idx.len() < 6 {
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
        let mut jtj = [0.0f64; 25];
        let mut jtf = [0.0f64; 5];
        let mut row = [0.0f64; 5];
        let mut m = 0usize;

        for &i in idx {
            let Radial { e, w, t } = radial(&c, positions[i as usize]);
            if !e.is_finite() {
                continue;
            }
            let su = w.dot(u);
            let sv = w.dot(v);
            row[0] = -su;
            row[1] = -sv;
            row[2] = -t * su;
            row[3] = -t * sv;
            row[4] = -1.0;
            for a in 0..5 {
                jtf[a] += row[a] * e;
                for b in a..5 {
                    jtj[a * 5 + b] += row[a] * row[b];
                }
            }
            m += 1;
        }
        if m < 6 {
            break;
        }
        for a in 0..5 {
            for b in 0..a {
                jtj[a * 5 + b] = jtj[b * 5 + a];
            }
        }

        let mut applied = false;
        let mut converged = false;
        for _attempt in 0..6 {
            let mut mat = jtj;
            for a in 0..5 {
                mat[a * 5 + a] *= 1.0 + lambda;
            }
            let rhs = jtf.map(|x| -x);
            let Some(step) = solve_linear(5, &mut mat, &rhs) else {
                lambda *= 10.0;
                continue;
            };
            let axis = normalize_or_z(DVec3::new(
                c.axis.x + step[2] * u.x + step[3] * v.x,
                c.axis.y + step[2] * u.y + step[3] * v.y,
                c.axis.z + step[2] * u.z + step[3] * v.z,
            ));
            let cand = Cylinder {
                point: DVec3::new(
                    c.point.x + step[0] * u.x + step[1] * v.x,
                    c.point.y + step[0] * u.y + step[1] * v.y,
                    c.point.z + step[0] * u.z + step[1] * v.z,
                ),
                axis,
                r: c.r + step[4],
            };
            if !(cand.r > 0.0) || !cand.point.x.is_finite() {
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
    c
}

/// A clipped cylinder fit.
#[derive(Debug, Clone, PartialEq)]
pub struct ClippedCylinderFit {
    pub cylinder: Cylinder,
    /// RMS of the radial residuals over the used points.
    pub sigma: f64,
    /// Peak-to-peak radial residual — the surface's cylindricity.
    pub span: f64,
    /// The signed radial extremes `span` is the difference of: how far inside and outside
    /// the fitted wall the kept points reach — a bore's tightest and widest spot, doubled,
    /// are the min and max diameter.
    pub min: f64,
    pub max: f64,
    pub used: Vec<u32>,
}

/// Gaussian best fit with GOM-style "used points" clipping (see `clipped_refit`). Each round
/// re-fits the circle on the previous round's axis and then refines geometrically, so the
/// axis carries over between clipping rounds instead of restarting from `init`.
pub fn fit_cylinder_clipped(positions: &[DVec3], idx: &[u32], init: Cylinder, k: f64) -> Option<ClippedCylinderFit> {
    let mut start = init;
    let r = clipped_refit(
        positions,
        idx,
        k,
        |used| {
            let seeded = fit_cylinder_on_axis(positions, used, start.axis).unwrap_or(start);
            let c = refine_cylinder_geometric(positions, used, seeded, 60);
            if !(c.r > 0.0) || !c.r.is_finite() {
                return None;
            }
            start = c;
            Some(c)
        },
        cylinder_residual,
    )?;
    Some(ClippedCylinderFit { cylinder: r.model, sigma: r.sigma, span: r.span, min: r.min, max: r.max, used: r.used })
}

/// A robust cylinder estimate and the points it rests on.
#[derive(Debug, Clone, PartialEq)]
pub struct RansacCylinderResult {
    pub cylinder: Cylinder,
    pub inliers: Vec<u32>,
    /// Robust noise estimate (1.4826 · median absolute residual), or the refit's σ when
    /// larger.
    pub sigma: f64,
}

/// Cylinder through two points with known surface normals: the axis is perpendicular to
/// both normals, and in the plane across it the axis sits where the two (projected) normal
/// lines meet.
fn cylinder_from2(positions: &[DVec3], normals: &[DVec3], i0: u32, i1: u32) -> Option<Cylinder> {
    let (pa, pb) = (positions[i0 as usize], positions[i1 as usize]);
    let (n0, n1) = (normals[i0 as usize], normals[i1 as usize]);
    let axis_raw = n0.cross(n1);
    let len = hypot3(axis_raw.x, axis_raw.y, axis_raw.z);
    // Near-parallel normals say nothing about where the axis runs.
    if !(len > 0.08) {
        return None;
    }
    let d = normalize_or_z(axis_raw);
    let (u, v) = ortho_basis(d);

    let (p0u, p0v) = (pa.dot(u), pa.dot(v));
    let (p1u, p1v) = (pb.dot(u), pb.dot(v));
    let (n0u, n0v) = (n0.dot(u), n0.dot(v));
    let (n1u, n1v) = (n1.dot(u), n1.dot(v));

    // Intersect p0 + s·n0 with p1 + t·n1 in the (u, v) plane.
    let det = n0u * -n1v - -n1u * n0v;
    if det.abs() < 1e-9 {
        return None;
    }
    let du = p1u - p0u;
    let dv = p1v - p0v;
    let s = (du * -n1v - -n1u * dv) / det;
    let cu = p0u + s * n0u;
    let cv = p0v + s * n0v;
    let r = hypot2(cu - p0u, cv - p0v);
    if !(r > 0.0) || !r.is_finite() {
        return None;
    }

    // Any point with the right (u, v) offset lies on the axis; anchor it at the first sample
    // so the axis passes near the patch.
    let along_u = cu - p0u;
    let along_v = cv - p0v;
    Some(Cylinder {
        point: DVec3::new(pa.x + along_u * u.x + along_v * v.x, pa.y + along_u * u.y + along_v * v.y, pa.z + along_u * u.z + along_v * v.z),
        axis: d,
        r,
    })
}

/// Robust cylinder estimate on a local patch: score the normal-covariance axis plus a batch
/// of two-point-with-normals candidates by median absolute residual (LMedS), then refine on
/// the consensus set.
pub fn ransac_cylinder(positions: &[DVec3], normals: &[DVec3], patch: &[u32], opts: RansacOptions) -> Option<RansacCylinderResult> {
    let core = ransac_consensus(positions, patch, opts, |diag| {
        let n = patch.len();
        // A radius wildly out of scale with the local patch is really a plane, never the
        // cylinder the user clicked.
        let plausible = move |c: &Cylinder| c.r > diag * 0.01 && c.r <= diag * 60.0;
        let from_normals = axis_from_normals(normals, patch).and_then(|axis| fit_cylinder_on_axis(positions, patch, axis));
        RansacShape {
            initial: from_normals.filter(|c| plausible(c)),
            generate: move |rand: &mut Mulberry32| {
                let i0 = patch[rand.index(n)];
                let i1 = patch[rand.index(n)];
                if i0 == i1 {
                    return None;
                }
                cylinder_from2(positions, normals, i0, i1).filter(|c| plausible(c))
            },
            residual: cylinder_residual,
        }
    })?;

    let refined = fit_cylinder_clipped(positions, &core.inliers, core.model, 3.0)?;
    // A plane fits a patch with no curvature as well — a flat face whose normals scatter, as
    // a scan's do, still yields candidates, and the radius cap above is all that holds them.
    if !curved_beyond_plane(positions, &refined.used, rms(positions, &refined.used, &refined.cylinder)) {
        return None;
    }
    Some(RansacCylinderResult { cylinder: refined.cylinder, inliers: refined.used, sigma: js_max(js_max(refined.sigma, core.sigma_est), 1e-9) })
}

// ---- One cylinder, or two ------------------------------------------------------------

/// Two cylinders are taken over one when they leave this share of its RMS or less, part by
/// this many times their own RMS over most of the region, and the smaller holds this share
/// of the points — as `split_two_planes` has it.
const TWO_CYL_OVER_ONE: f64 = 0.75;
const TWO_CYL_PARTED: f64 = 6.0;
const TWO_CYL_SHARE: f64 = 0.08;
/// How many points the split is searched on, and how many rounds of assigning each to the
/// nearer cylinder and refitting both it runs.
const TWO_CYL_SAMPLE: usize = 4000;
const TWO_CYL_ROUNDS: usize = 6;

/// A region read as two cylinders.
#[derive(Debug, Clone, PartialEq)]
pub struct TwoCylinderSplit {
    /// The two cylinders, the larger part first.
    pub cylinders: [Cylinder; 2],
    /// Each cylinder's points, as indices into the `idx` given.
    pub members: [Vec<u32>; 2],
    /// RMS of each point off the nearer of the two, and off the one.
    pub rms_two: f64,
    pub rms_one: f64,
    /// How far apart the two lie over most of the region (the 90th percentile, at its
    /// points), mm.
    pub parted: f64,
}

/// A cylinder on some points: fresh, from the axis their normals give (or the start's), else
/// refined from the start.
fn cylinder_on(positions: &[DVec3], normals: &[DVec3], idx: &[u32], start: Cylinder, fresh: bool) -> Option<Cylinder> {
    let axis = (if fresh { axis_from_normals(normals, idx) } else { None }).unwrap_or(start.axis);
    let seeded = if fresh { fit_cylinder_on_axis(positions, idx, axis)? } else { start };
    let c = refine_cylinder_geometric(positions, idx, seeded, 15);
    (c.r > 0.0 && c.r.is_finite() && c.point.x.is_finite()).then_some(c)
}

/// The `q` quantile of the values, as the sorted copy's element at ⌊q·(n − 1)⌋.
fn quantile(a: &[f64], q: f64) -> f64 {
    let mut s = a.to_vec();
    s.sort_unstable_by(|x, y| x.total_cmp(y));
    s[(q * (s.len() - 1) as f64).floor() as usize]
}

/// Whether a region fitted as one cylinder is two: an opening whose lip leans while its wall
/// stands, or whose one side stands and the other leans, grows as one surface and fits as
/// one cylinder tilted between them, its σ saying neither. So the points are split into two
/// cylinders, each point to the nearer, refitted until the split stands, from several
/// starts: bands along the axis (a lip above a wall), halves across it (one side against the
/// other), and by how far the normals lean off square to the axis. The split is kept when
/// the two lie clearly nearer to the points than the one, part by far more than the points
/// scatter off them, both hold a real share, and a radius that changes smoothly along the
/// axis — a cone, a barrel — does no better. `None` when the region is one surface.
pub fn split_two_cylinders(positions: &[DVec3], normals: &[DVec3], idx: &[u32], one: &Cylinder) -> Option<TwoCylinderSplit> {
    if idx.len() < 120 {
        return None;
    }
    let sample = strided(idx, TWO_CYL_SAMPLE);
    let n = sample.len();
    let (u, v) = ortho_basis(one.axis);
    let mut t = vec![0.0; n];
    let mut cu = vec![0.0; n];
    let mut cv = vec![0.0; n];
    let mut lean = vec![0.0; n];
    for (i, &s) in sample.iter().enumerate() {
        let r = radial(one, positions[s as usize]);
        t[i] = r.t;
        cu[i] = r.w.dot(u);
        cv[i] = r.w.dot(v);
        lean[i] = normals[s as usize].dot(one.axis).abs();
    }
    let mut starts: Vec<Vec<u8>> = Vec::new();
    for q in [0.3, 0.5, 0.7] {
        let cut = quantile(&t, q);
        starts.push(t.iter().map(|&x| u8::from(x > cut)).collect());
    }
    for k in 0..4 {
        let a = (k as f64 * std::f64::consts::PI) / 4.0;
        let (du, dv) = (a.cos(), a.sin());
        starts.push((0..n).map(|i| u8::from(cu[i] * du + cv[i] * dv > 0.0)).collect());
    }
    let lean_cut = quantile(&lean, 0.5);
    starts.push(lean.iter().map(|&x| u8::from(x > lean_cut)).collect());

    let mut best: Option<([Cylinder; 2], f64)> = None;
    for mut side in starts {
        let mut cyl: Option<[Cylinder; 2]> = None;
        for _round in 0..TWO_CYL_ROUNDS {
            let mut parts: [Vec<u32>; 2] = [Vec::new(), Vec::new()];
            for (i, &s) in side.iter().enumerate() {
                parts[s as usize].push(sample[i]);
            }
            if parts[0].len() < 30 || parts[1].len() < 30 {
                cyl = None;
                break;
            }
            let fresh = cyl.is_none();
            let (s0, s1) = match cyl {
                Some([a, b]) => (a, b),
                None => (*one, *one),
            };
            let c0 = cylinder_on(positions, normals, &parts[0], s0, fresh);
            let c1 = cylinder_on(positions, normals, &parts[1], s1, fresh);
            let (Some(c0), Some(c1)) = (c0, c1) else {
                cyl = None;
                break;
            };
            cyl = Some([c0, c1]);
            let mut moved = 0usize;
            for (i, &s) in sample.iter().enumerate() {
                let p = positions[s as usize];
                let next = u8::from(radial(&c1, p).e.abs() < radial(&c0, p).e.abs());
                if next != side[i] {
                    moved += 1;
                }
                side[i] = next;
            }
            if moved == 0 {
                break;
            }
        }
        let Some(cyl) = cyl else { continue };
        let mut ss = 0.0;
        for &s in sample.iter() {
            let p = positions[s as usize];
            ss += js_min(radial(&cyl[0], p).e.abs(), radial(&cyl[1], p).e.abs()).powi(2);
        }
        let rms = (ss / n as f64).sqrt();
        if rms.is_finite() && best.as_ref().is_none_or(|b| rms < b.1) {
            best = Some((cyl, rms));
        }
    }
    let ([c0, c1], _) = best?;

    // Every point to the nearer, the RMS over all, how far apart the two lie at each point,
    // and the one's residual with a radius that changes smoothly along the axis taken out:
    // e ≈ a + b t + c t², least squares.
    let total = idx.len();
    let mut side = vec![0u8; total];
    let mut apart = vec![0.0f32; total];
    let mut e_one = vec![0.0f64; total];
    let mut t_one = vec![0.0f64; total];
    let mut m = [0.0f64; 9];
    let mut rhs = [0.0f64; 3];
    let (mut ss_one, mut ss_two, mut count1) = (0.0, 0.0, 0usize);
    for (i, &vi) in idx.iter().enumerate() {
        let p = positions[vi as usize];
        let e0 = radial(&c0, p).e;
        let e1 = radial(&c1, p).e;
        let r = radial(one, p);
        if !e0.is_finite() || !e1.is_finite() || !r.e.is_finite() {
            continue;
        }
        if e1.abs() < e0.abs() {
            side[i] = 1;
            count1 += 1;
        }
        apart[i] = (e0 - e1).abs() as f32;
        ss_two += js_min(e0 * e0, e1 * e1);
        ss_one += r.e * r.e;
        e_one[i] = r.e;
        t_one[i] = r.t;
        let row = [1.0, r.t, r.t * r.t];
        for a in 0..3 {
            rhs[a] += row[a] * r.e;
            for b in 0..3 {
                m[a * 3 + b] += row[a] * row[b];
            }
        }
    }
    let rms_one = (ss_one / total as f64).sqrt();
    let rms_two = (ss_two / total as f64).sqrt();
    if !(rms_one > 0.0) || !(rms_two < TWO_CYL_OVER_ONE * rms_one) {
        return None;
    }
    if (count1.min(total - count1) as f64) < (30.0f64).max(TWO_CYL_SHARE * total as f64) {
        return None;
    }
    apart.sort_unstable_by(|a, b| a.total_cmp(b));
    let parted = f64::from(apart[(0.9 * (total - 1) as f64).floor() as usize]);
    if !(parted >= TWO_CYL_PARTED * rms_two) {
        return None;
    }
    if let Some(coef) = solve_linear(3, &mut m, &rhs) {
        let mut ss_smooth = 0.0;
        for i in 0..total {
            ss_smooth += (e_one[i] - coef[0] - coef[1] * t_one[i] - coef[2] * t_one[i] * t_one[i]).powi(2);
        }
        if !(rms_two < TWO_CYL_OVER_ONE * (ss_smooth / total as f64).sqrt()) {
            return None;
        }
    }

    let mut a = Vec::new();
    let mut b = Vec::new();
    for (i, &s) in side.iter().enumerate() {
        if s == 1 {
            b.push(i as u32);
        } else {
            a.push(i as u32);
        }
    }
    let big = a.len() >= b.len();
    Some(TwoCylinderSplit {
        cylinders: if big { [c0, c1] } else { [c1, c0] },
        members: if big { [a, b] } else { [b, a] },
        rms_two,
        rms_one,
        parted,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit3d::testkit::{all_indices, helper_basis, p32, rad, sample_cylinder, sample_plane, unit, v32};

    const POINT: [f64; 3] = [4.0, -3.0, 2.0];
    // Off-axis on purpose, so an axis-aligned bug cannot pass by accident.
    const AXIS: [f64; 3] = [0.4, 0.6, 0.6928];
    const R: f64 = 6.35;
    const LENGTH: f64 = 30.0;
    const TAU: f64 = 2.0 * std::f64::consts::PI;

    /// Exact outward normals for points sampled on the cylinder wall.
    fn wall_normals(points: &[DVec3], point: [f64; 3], axis: [f64; 3]) -> Vec<DVec3> {
        let d = unit(axis);
        let p0 = DVec3::from_array(point);
        points
            .iter()
            .map(|&p| {
                let q = p - p0;
                let t = q.x * d.x + q.y * d.y + q.z * d.z;
                let w = DVec3::new(q.x - t * d.x, q.y - t * d.y, q.z - t * d.z);
                let l = (w.x * w.x + w.y * w.y + w.z * w.z).sqrt();
                p32(w.x / l, w.y / l, w.z / l)
            })
            .collect()
    }

    /// Distance from the true axis to the fitted one, measured at the sample's centre — the
    /// number that actually matters for a downstream measurement.
    fn axis_offset(c: &Cylinder) -> f64 {
        let q = DVec3::from_array(POINT) - c.point;
        let t = q.dot(c.axis);
        (q - c.axis * t).length()
    }

    fn tilt(c: &Cylinder) -> f64 {
        let cosine = c.axis.dot(unit(AXIS)).abs().min(1.0);
        cosine.acos().to_degrees()
    }

    // cylinderFit.test.ts — 'Gaussian cylinder fit'.
    #[test]
    fn recovers_radius_and_axis_from_a_10_degree_off_starting_guess() {
        let pts = sample_cylinder(20_000, POINT, AXIS, R, LENGTH, 0.02, 42, TAU);
        // Tilt the initial axis 10° and shift it 1 mm sideways.
        let bad = unit([AXIS[0] + 0.18, AXIS[1] - 0.1, AXIS[2]]);
        let idx = all_indices(20_000);
        let init = fit_cylinder_on_axis(&pts, &idx, bad).expect("a start");
        let fit = fit_cylinder_clipped(&pts, &idx, init, 3.0).expect("a fit");
        assert!((fit.cylinder.r - R).abs() < 0.005);
        assert!(tilt(&fit.cylinder) < 0.05);
        assert!(axis_offset(&fit.cylinder) < 0.005);
        assert!(fit.sigma > 0.015);
        assert!(fit.sigma < 0.025);
    }

    #[test]
    fn recovers_a_140_degree_arc() {
        let pts = sample_cylinder(20_000, POINT, AXIS, R, LENGTH, 0.02, 43, rad(140.0));
        let idx = all_indices(20_000);
        let init = fit_cylinder_on_axis(&pts, &idx, unit([AXIS[0] + 0.1, AXIS[1], AXIS[2]])).expect("a start");
        let fit = fit_cylinder_clipped(&pts, &idx, init, 3.0).expect("a fit");
        assert!((fit.cylinder.r - R).abs() < 0.05);
        assert!(tilt(&fit.cylinder) < 0.3);
        assert!(axis_offset(&fit.cylinder) < 0.05);
    }

    #[test]
    fn sigma_clipping_sheds_sparse_gross_outliers() {
        let mut pts = sample_cylinder(20_000, POINT, AXIS, R, LENGTH, 0.02, 44, TAU);
        let normals = wall_normals(&pts, POINT, AXIS);
        let mut rand = Mulberry32::new(99);
        for _ in 0..400 {
            let j = rand.index(20_000);
            let (p, n) = (pts[j], normals[j]);
            pts[j] = p32(p.x + n.x * 0.5, p.y + n.y * 0.5, p.z + n.z * 0.5);
        }
        let idx = all_indices(20_000);
        let init = fit_cylinder_on_axis(&pts, &idx, DVec3::from_array(AXIS)).expect("a start");
        let clipped = fit_cylinder_clipped(&pts, &idx, init, 3.0).expect("a fit");
        assert!((clipped.cylinder.r - R).abs() < 0.01);
        assert!(clipped.used.len() < 20_000);
    }

    // cylinderFit.test.ts — 'RANSAC cylinder estimate'.
    #[test]
    fn finds_the_cylinder_despite_30_percent_structured_outliers() {
        let mut all = sample_cylinder(3_000, POINT, AXIS, R, LENGTH, 0.02, 45, rad(200.0));
        let mut all_n = wall_normals(&all, POINT, AXIS);
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
        let res = ransac_cylinder(&all, &all_n, &all_indices(4_300), RansacOptions { seed: Some(1), ..Default::default() }).expect("a cylinder");
        assert!((res.cylinder.r - R).abs() < 0.1);
        assert!(tilt(&res.cylinder) < 1.0);
        assert!(axis_offset(&res.cylinder) < 0.1);
        // The disc must not have been swept into the consensus set — bar the few of its
        // points that happen to sit on the ring where it crosses the wall.
        let from_disc = res.inliers.iter().filter(|&&i| i >= 3_000).count();
        assert!((from_disc as f64) < 0.03 * res.inliers.len() as f64);
    }

    #[test]
    fn finds_a_large_cylinder_from_a_few_degrees_of_it() {
        // 4° of a 500 mm radius: the arc rises 0.3 mm off its chord, fifteen times the noise
        // — curved, however little of the turn it is.
        let sliver = sample_cylinder(2_000, POINT, AXIS, 500.0, 20.0, 0.02, 8, rad(4.0));
        let normals = wall_normals(&sliver, POINT, AXIS);
        let res = ransac_cylinder(&sliver, &normals, &all_indices(2_000), RansacOptions { seed: Some(2), ..Default::default() }).expect("a cylinder");
        assert!((res.cylinder.r - 500.0).abs() < 5.0);
        assert!(tilt(&res.cylinder) < 0.5);
    }

    /// Unit normals each turned off its own by a random few degrees.
    fn scattered(normals: &[DVec3], degrees: f64) -> Vec<DVec3> {
        let mut rand = Mulberry32::new(5);
        let s = rad(degrees);
        normals
            .iter()
            .map(|n| {
                let x = n.x + rand.gaussian() * s;
                let y = n.y + rand.gaussian() * s;
                let z = n.z + rand.gaussian() * s;
                v32(unit([x, y, z]))
            })
            .collect()
    }

    #[test]
    fn rejects_a_patch_with_no_curvature_to_work_with() {
        // A flat patch whose normals scatter by a few degrees, as a scan's do: pairs of them
        // still cross, and without a check every candidate is a cylinder as large as the
        // radius cap lets it be.
        let normal = unit([0.3, -0.5, 0.8]);
        let flat = sample_plane(2_000, POINT, normal.to_array(), 30.0, 0.02, 11);
        let flat_normals = vec![v32(normal); flat.len()];
        let opts = RansacOptions { seed: Some(2), ..Default::default() };
        assert!(ransac_cylinder(&flat, &scattered(&flat_normals, 3.0), &all_indices(2_000), opts).is_none());
        // 2° of a 500 mm radius rises 76 µm off its chord, under noise of 0.3 mm.
        let sliver = sample_cylinder(2_000, POINT, AXIS, 500.0, 20.0, 0.3, 8, rad(2.0));
        assert!(ransac_cylinder(&sliver, &wall_normals(&sliver, POINT, AXIS), &all_indices(2_000), opts).is_none());
    }

    /// Points on a lobed wall about the test axis: r(θ) = R + amp·cos(lobes·θ), no noise — a
    /// surface that reaches exactly `amp` inside and outside the round cylinder it averages
    /// to. The angles are laid out evenly (only the axial position is random), so the lobes
    /// cancel exactly and the sample hits both the peaks and the flats.
    fn lobed_wall(n: usize, lobes: f64, amp: f64) -> Vec<DVec3> {
        let mut rand = Mulberry32::new(7);
        let d = unit(AXIS);
        let h = if d.x.abs() < 0.9 { DVec3::X } else { DVec3::Y };
        let u0 = DVec3::new(d.y * h.z - d.z * h.y, d.z * h.x - d.x * h.z, d.x * h.y - d.y * h.x);
        let u = unit(u0.to_array());
        let v = DVec3::new(d.y * u.z - d.z * u.y, d.z * u.x - d.x * u.z, d.x * u.y - d.y * u.x);
        (0..n)
            .map(|i| {
                let ang = (i as f64 / n as f64) * TAU;
                let t = (rand.next_f64() - 0.5) * LENGTH;
                let r = R + amp * (lobes * ang).cos();
                let (cu, cv) = (ang.cos() * r, ang.sin() * r);
                p32(
                    POINT[0] + cu * u.x + cv * v.x + t * d.x,
                    POINT[1] + cu * u.y + cv * v.y + t * d.y,
                    POINT[2] + cu * u.z + cv * v.z + t * d.z,
                )
            })
            .collect()
    }

    // cylinderFit.test.ts — 'radial extremes'.
    #[test]
    fn reports_how_far_a_lobed_wall_reaches_inside_and_outside_the_fit() {
        // Three lobes average out, so the Gaussian radius is R, and the wall reaches 50 µm
        // inside it at the flats and 50 µm outside at the lobes — the two numbers the min
        // and max diameter are read from.
        let (n, amp) = (6000, 0.05);
        let pts = lobed_wall(n, 3.0, amp);
        let init = Cylinder { point: DVec3::from_array(POINT), axis: unit(AXIS), r: R };
        let fit = fit_cylinder_clipped(&pts, &all_indices(n), init, 0.0).expect("a fit");
        assert!((fit.cylinder.r - R).abs() < 5e-4);
        assert!((fit.min + amp).abs() < 5e-4);
        assert!((fit.max - amp).abs() < 5e-4);
        assert!((fit.span - (fit.max - fit.min)).abs() < 0.5e-12);
        assert_eq!(fit.used.len(), n);
    }

    // cylinderTwoSurfaces.test.ts, adapted: ScanRuler runs these regions through its
    // segmentation (`secondCylinder`, on a welded mesh graph with averaged vertex normals);
    // here the same bands go to `split_two_cylinders` directly, as the grid points with their
    // exact normals, against the one cylinder the whole region fits.

    /// A band of a cylinder of radius `r` about `axis` through `base`, `length` long, its
    /// radius growing by `flare` per mm along it, as a grid with a little radial noise
    /// (the test's Park–Miller generator), and the exact outward normals.
    #[allow(clippy::too_many_arguments)]
    fn band(base: [f64; 3], axis: [f64; 3], length: f64, seed: u64, flare: f64, r: f64, pts: &mut Vec<DVec3>, nrm: &mut Vec<DVec3>) {
        let a = unit(axis);
        let (u, v) = helper_basis(a);
        let mut s = seed;
        let mut rnd = || {
            s = (s * 16807) % 2147483647;
            (s as f64 / 2147483647.0 - 0.5) * 2.0
        };
        let (nu, nv) = (240usize, (length / 0.5).round() as usize);
        let slope = flare.atan();
        for i in 0..=nv {
            let t = (length * i as f64) / nv as f64;
            for k in 0..nu {
                let q = (TAU * k as f64) / nu as f64;
                let rr = r + flare * t + 0.02 * rnd();
                let radial = u * q.cos() + v * q.sin();
                pts.push(v32(DVec3::from_array(base) + a * t + radial * rr));
                nrm.push(v32(radial * slope.cos() - a * slope.sin()));
            }
        }
    }

    fn one_cylinder(pts: &[DVec3], nrm: &[DVec3]) -> (Vec<u32>, Cylinder) {
        let all = all_indices(pts.len());
        let axis = axis_from_normals(nrm, &all).expect("an axis");
        let init = fit_cylinder_on_axis(pts, &all, axis).expect("a start");
        let fit = fit_cylinder_clipped(pts, &all, init, 3.0).expect("a fit");
        (all, fit.cylinder)
    }

    #[test]
    fn splits_an_opening_whose_upper_half_leans_from_its_lower() {
        let tilt = rad(22.0);
        let (mut pts, mut nrm) = (Vec::new(), Vec::new());
        band([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 20.0, 3, 0.0, 40.0, &mut pts, &mut nrm);
        band([0.0, 0.0, 20.0], [0.0, tilt.sin(), tilt.cos()], 20.0, 5, 0.0, 40.0, &mut pts, &mut nrm);
        let (all, one) = one_cylinder(&pts, &nrm);
        let split = split_two_cylinders(&pts, &nrm, &all, &one).expect("two cylinders");
        let share = split.members[1].len() as f64 / all.len() as f64;
        assert!(share > 0.4 && share <= 0.5);
        let [a, b] = split.cylinders;
        let angle = a.axis.dot(b.axis).abs().min(1.0).acos().to_degrees();
        assert!(angle > 21.0 && angle < 23.0);
        assert!((a.r - 40.0).abs() < 0.2);
        assert!((b.r - 40.0).abs() < 0.2);
    }

    #[test]
    fn reads_a_plain_cylinder_and_a_drafted_one_as_one_surface() {
        let (mut pts, mut nrm) = (Vec::new(), Vec::new());
        band([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 40.0, 7, 0.0, 40.0, &mut pts, &mut nrm);
        let (all, one) = one_cylinder(&pts, &nrm);
        assert!(split_two_cylinders(&pts, &nrm, &all, &one).is_none());
        // 1° of draft: the radius grows 0.7 mm over the 40.
        let (mut pts, mut nrm) = (Vec::new(), Vec::new());
        band([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 40.0, 9, rad(1.0).tan(), 40.0, &mut pts, &mut nrm);
        let (all, one) = one_cylinder(&pts, &nrm);
        assert!(split_two_cylinders(&pts, &nrm, &all, &one).is_none());
    }
}
