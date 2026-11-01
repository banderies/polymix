// SPDX-License-Identifier: AGPL-3.0-only
//! The plane — unlike the curved shapes, its Gaussian best fit is exact: the centroid, and
//! as normal the direction the points scatter least along, with no iteration. Its robust
//! start scores three-point planes by median residual (LMedS), so a patch straddling an edge
//! settles on the face holding most of it, where a least-squares plane would be dragged
//! towards the other face. Settling on it says nothing of the other face, though, so a grown
//! region can be split into two planes to see whether it holds two (`split_two_planes`).
//!
//! Ported from ScanRuler (`src/core/fit/plane.ts`), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only.

use std::borrow::Cow;

use glam::DVec3;

use super::clip::clipped_refit;
use super::height_surface::{fit_height, patch_frame, SurfaceModel};
use super::linalg3::{centroid_of, eigen_of_scatter, js_max, js_min, ortho_basis, rms_over};
use super::ransac::{ransac_consensus, RansacOptions, RansacShape};
use super::types::Plane;
use crate::fit2d::Mulberry32;

/// Signed distance of a point from the plane (positive on the normal side).
pub fn plane_residual(p: &Plane, x: DVec3) -> f64 {
    p.normal.x * x.x + p.normal.y * x.y + p.normal.z * x.z - p.d
}

/// Total-least-squares plane: the centroid, plus the direction the points scatter least
/// along as the normal. Exact — this *is* the Gaussian best fit for a plane, so no iterative
/// refinement step is needed.
pub fn fit_plane_tls(positions: &[DVec3], idx: &[u32]) -> Option<Plane> {
    if idx.len() < 3 {
        return None;
    }
    let m = centroid_of(positions, idx);

    let (mut cxx, mut cxy, mut cxz, mut cyy, mut cyz, mut czz) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for &i in idx {
        let q = positions[i as usize] - m;
        cxx += q.x * q.x;
        cxy += q.x * q.y;
        cxz += q.x * q.z;
        cyy += q.y * q.y;
        cyz += q.y * q.z;
        czz += q.z * q.z;
    }

    let eig = eigen_of_scatter(cxx, cxy, cxz, cyy, cyz, czz);
    // Collinear points leave the normal free to spin around the line they lie on: the two
    // smallest scatter directions are then equally small.
    if !(eig.values[1] > 1e-12 * eig.values[2]) || !(eig.values[2] > 0.0) {
        return None;
    }
    let n = eig.vectors[0];
    if !n.x.is_finite() {
        return None;
    }
    Some(Plane { normal: n, d: n.x * m.x + n.y * m.y + n.z * m.z })
}

/// How much nearer than the best plane a curved fit has to lie to its points for the
/// curvature to be there at all, as the plane's RMS over the fit's. On a flat patch the
/// curved shape's spare parameters buy next to nothing, and the ratio is 1; a cylinder whose
/// arc rises 1.5σ off its chord scores about 1.1.
const CURVED_OVER_FLAT: f64 = 1.1;

/// Whether a curved fit tells its points from a plane: whether it lies clearly nearer to
/// them than the best plane through them does. A patch it does not has no curvature to work
/// with, and the radius it was fitted is the noise's.
pub fn curved_beyond_plane(positions: &[DVec3], idx: &[u32], curved_rms: f64) -> bool {
    let Some(plane) = fit_plane_tls(positions, idx) else { return true };
    rms_over(positions, idx, |p| plane_residual(&plane, p)) >= CURVED_OVER_FLAT * curved_rms
}

/// A clipped plane fit.
#[derive(Debug, Clone, PartialEq)]
pub struct ClippedPlaneFit {
    pub plane: Plane,
    /// RMS of the point-to-plane distances over the used points.
    pub sigma: f64,
    /// Peak-to-peak distance over the used points — the patch's flatness.
    pub span: f64,
    /// The signed extremes `span` is the difference of: how far below and above the fit the
    /// kept points reach, along the fitted normal.
    pub min: f64,
    pub max: f64,
    pub used: Vec<u32>,
}

/// Gaussian best fit with GOM-style "used points" clipping (see `clipped_refit`), each
/// round an exact TLS fit.
pub fn fit_plane_clipped(positions: &[DVec3], idx: &[u32], k: f64) -> Option<ClippedPlaneFit> {
    let r = clipped_refit(positions, idx, k, |used| fit_plane_tls(positions, used), plane_residual)?;
    Some(ClippedPlaneFit { plane: r.model, sigma: r.sigma, span: r.span, min: r.min, max: r.max, used: r.used })
}

/// How much nearer two planes have to lie to a region's points than one does, as their RMS
/// over the one plane's, for the region to be read as two surfaces. Noise alone, split down
/// the middle into two planes either side of it, comes to about 0.6 — two planes for one
/// surface buy that much from nothing.
const TWO_OVER_ONE: f64 = 0.75;
/// So the two planes have to part as well: over most of the region (the 90th percentile)
/// lie this many times their RMS apart, at least. Noise split in two gives planes about
/// three times their RMS apart everywhere, smoothed noise in blobs not much more; a crease
/// parts them more and more away from the line they meet on, a step by its height.
const PARTED_RMS: f64 = 6.0;
/// The share of the points the smaller of the two has to hold, at least — a few strays off a
/// face are outliers, not a second face.
const SECOND_SHARE: f64 = 0.08;
/// Assignment rounds the two-plane split runs from each start, on at most this many of the
/// points.
const SPLIT_ROUNDS: usize = 12;
const SPLIT_SAMPLE: usize = 20_000;
/// How many points the quadric is fitted on.
const QUADRIC_SAMPLE: usize = 4000;

/// A region read as two planes.
#[derive(Debug, Clone, PartialEq)]
pub struct TwoPlaneSplit {
    /// The two planes, the larger part first.
    pub planes: [Plane; 2],
    /// Each plane's points, as indices into the `idx` given (not vertex numbers).
    pub members: [Vec<u32>; 2],
    /// RMS of each point off the nearer of the two, and off the one plane.
    pub rms_two: f64,
    pub rms_one: f64,
    /// How far apart the two planes lie over most of the region (the 90th percentile, at
    /// its points), mm.
    pub parted: f64,
}

/// Every k-th of the points, to at most `most` of them.
pub(crate) fn strided(idx: &[u32], most: usize) -> Cow<'_, [u32]> {
    if idx.len() <= most {
        return Cow::Borrowed(idx);
    }
    let step = idx.len() as f64 / most as f64;
    Cow::Owned((0..most).map(|i| idx[(i as f64 * step).floor() as usize]).collect())
}

/// Whether a region fitted as one plane is two: a seed near a shallow crease grows across
/// it, and a step lower than the band joins its two levels. A plane through such a region
/// settles between the two, or — robustly, as `ransac_plane` does — on the larger, and its σ
/// alone says neither. So the points are split into two planes, each point to the nearer,
/// refitted until the split stands, from two starts: by which side of the one plane a point
/// lies (a step) and by which way its normal leans across it (a crease). The split is kept
/// when the two planes lie clearly nearer to the points than the one, part by far more than
/// the points scatter off them, both hold a real share of the points, and a quadric does no
/// better — a surface that bows is curved, not two. `None` when the region is one surface.
/// `normals` are the vertex normals beside `positions`.
pub fn split_two_planes(positions: &[DVec3], normals: &[DVec3], idx: &[u32]) -> Option<TwoPlaneSplit> {
    if idx.len() < 60 {
        return None;
    }
    let sample = strided(idx, SPLIT_SAMPLE);
    let n = sample.len();
    let one = fit_plane_tls(positions, &sample)?;
    let residual: Vec<f64> = sample.iter().map(|&i| plane_residual(&one, positions[i as usize])).collect();

    // The way the normals lean across the one plane: the in-plane direction their
    // tangential parts scatter along most.
    let (u, v) = ortho_basis(one.normal);
    let mut tu = vec![0.0; n];
    let mut tv = vec![0.0; n];
    let (mut mu, mut mv) = (0.0, 0.0);
    for (i, &s) in sample.iter().enumerate() {
        let nrm = normals[s as usize];
        tu[i] = nrm.dot(u);
        tv[i] = nrm.dot(v);
        mu += tu[i];
        mv += tv[i];
    }
    mu /= n as f64;
    mv /= n as f64;
    let (mut suu, mut suv, mut svv) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let (a, b) = (tu[i] - mu, tv[i] - mv);
        suu += a * a;
        suv += a * b;
        svv += b * b;
    }
    let angle = 0.5 * (2.0 * suv).atan2(suu - svv);
    let (lu, lv) = (angle.cos(), angle.sin());

    let starts: [Vec<u8>; 2] = [
        residual.iter().map(|&r| u8::from(r > 0.0)).collect(),
        (0..n).map(|i| u8::from((tu[i] - mu) * lu + (tv[i] - mv) * lv > 0.0)).collect(),
    ];
    let mut best: Option<([Plane; 2], f64)> = None;
    for side in starts {
        if let Some((_, planes, rms)) = refine_split(positions, &sample, side) {
            if best.as_ref().is_none_or(|b| rms < b.1) {
                best = Some((planes, rms));
            }
        }
    }
    let ([p0, p1], _) = best?;

    // Every point of the region to the nearer plane, the RMS over all, and how far apart
    // the planes lie at each point — measured along the one plane's normal with the two
    // turned the same way.
    let total = idx.len();
    let mut side = vec![0u8; total];
    let mut apart = vec![0.0f32; total];
    let turn = if p0.normal.dot(p1.normal) >= 0.0 { 1.0 } else { -1.0 };
    let (mut ss_one, mut ss_two, mut count1) = (0.0, 0.0, 0usize);
    for (i, &vi) in idx.iter().enumerate() {
        let p = positions[vi as usize];
        let s0 = plane_residual(&p0, p);
        let s1 = turn * plane_residual(&p1, p);
        let r0 = s0.abs();
        let r1 = s1.abs();
        if r1 < r0 {
            side[i] = 1;
            count1 += 1;
        }
        apart[i] = (s0 - s1).abs() as f32;
        ss_two += js_min(r0, r1).powi(2);
        ss_one += plane_residual(&one, p).powi(2);
    }
    let rms_one = (ss_one / total as f64).sqrt();
    let rms_two = (ss_two / total as f64).sqrt();
    if !(rms_one > 0.0) || !(rms_two < TWO_OVER_ONE * rms_one) {
        return None;
    }
    if (count1.min(total - count1) as f64) < (30.0f64).max(SECOND_SHARE * total as f64) {
        return None;
    }

    apart.sort_unstable_by(|a, b| a.total_cmp(b));
    let parted = f64::from(apart[(0.9 * (total - 1) as f64).floor() as usize]);
    if !(parted >= PARTED_RMS * rms_two) {
        return None;
    }

    // A bowed surface splits into two planes well too; a quadric over the region tells it
    // from a crease, which no quadric follows as closely as two planes do.
    let few = strided(idx, QUADRIC_SAMPLE);
    let packed: Vec<DVec3> = few.iter().map(|&i| positions[i as usize]).collect();
    let mut ss_few = 0.0;
    for &p in &packed {
        ss_few += js_min(plane_residual(&p0, p).abs(), plane_residual(&p1, p).abs()).powi(2);
    }
    if let Some(frame) = patch_frame(&packed) {
        if let Some(quadric) = fit_height(&packed, &frame, SurfaceModel::Quadric) {
            if quadric.rms <= (ss_few / few.len() as f64).sqrt() {
                return None;
            }
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
    Some(TwoPlaneSplit {
        planes: if big { [p0, p1] } else { [p1, p0] },
        members: if big { [a, b] } else { [b, a] },
        rms_two,
        rms_one,
        parted,
    })
}

/// Two planes from a first split of the points (side 0 or 1 each), each point then moved to
/// the nearer plane and both refitted, until no point moves. `None` when a side empties.
fn refine_split(positions: &[DVec3], idx: &[u32], start: Vec<u8>) -> Option<(Vec<u8>, [Plane; 2], f64)> {
    let n = idx.len();
    let mut side = start;
    let mut planes: Option<[Plane; 2]> = None;
    let mut rms = f64::INFINITY;
    for _round in 0..SPLIT_ROUNDS {
        let mut groups: [Vec<u32>; 2] = [Vec::new(), Vec::new()];
        for (i, &s) in side.iter().enumerate() {
            groups[s as usize].push(idx[i]);
        }
        if groups[0].len() < 3 || groups[1].len() < 3 {
            return None;
        }
        let p0 = fit_plane_tls(positions, &groups[0])?;
        let p1 = fit_plane_tls(positions, &groups[1])?;
        planes = Some([p0, p1]);
        let mut moved = 0usize;
        let mut ss = 0.0;
        for (i, &vi) in idx.iter().enumerate() {
            let p = positions[vi as usize];
            let r0 = plane_residual(&p0, p).abs();
            let r1 = plane_residual(&p1, p).abs();
            let s = u8::from(r1 < r0);
            if s != side[i] {
                side[i] = s;
                moved += 1;
            }
            ss += js_min(r0, r1).powi(2);
        }
        rms = (ss / n as f64).sqrt();
        if moved == 0 {
            break;
        }
    }
    planes.map(|planes| (side, planes, rms))
}

/// A robust plane estimate and the points it rests on.
#[derive(Debug, Clone, PartialEq)]
pub struct RansacPlaneResult {
    pub plane: Plane,
    pub inliers: Vec<u32>,
    /// Robust noise estimate (1.4826 · median absolute residual), or the refit's σ when
    /// larger.
    pub sigma: f64,
}

/// Robust plane estimate on a local patch: sample three-point candidate planes, score by
/// median absolute residual (LMedS — no threshold parameter needed), then refine on the
/// consensus set with a 3-sigma clipped Gaussian fit. A patch straddling an edge would drag
/// a plain least-squares plane off the surface the user clicked; the median score ignores
/// the smaller side.
pub fn ransac_plane(positions: &[DVec3], patch: &[u32], opts: RansacOptions) -> Option<RansacPlaneResult> {
    let core = ransac_consensus(positions, patch, opts, |diag| {
        let n = patch.len();
        // Three sampled points must span a real triangle to define a plane; the area of a
        // sliver is what makes the normal noise-dominated.
        let min_area = diag * diag * 1e-4;
        RansacShape {
            initial: None,
            generate: move |rand: &mut Mulberry32| {
                let i0 = patch[rand.index(n)];
                let i1 = patch[rand.index(n)];
                let i2 = patch[rand.index(n)];
                if i0 == i1 || i0 == i2 || i1 == i2 {
                    return None;
                }
                let a = positions[i0 as usize];
                let u = positions[i1 as usize] - a;
                let v = positions[i2 as usize] - a;
                let nrm = u.cross(v);
                let len = (nrm.x * nrm.x + nrm.y * nrm.y + nrm.z * nrm.z).sqrt();
                if !(len > min_area) {
                    return None;
                }
                let nrm = DVec3::new(nrm.x / len, nrm.y / len, nrm.z / len);
                Some(Plane { normal: nrm, d: nrm.x * a.x + nrm.y * a.y + nrm.z * a.z })
            },
            residual: plane_residual,
        }
    })?;

    let refined = fit_plane_clipped(positions, &core.inliers, 3.0)?;
    Some(RansacPlaneResult { plane: refined.plane, inliers: refined.used, sigma: js_max(js_max(refined.sigma, core.sigma_est), 1e-9) })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit3d::testkit::{all_indices, p32, sample_plane, unit};

    const POINT: [f64; 3] = [3.0, -2.0, 7.0];
    // Deliberately off-axis, so an axis-aligned bug cannot pass by accident.
    const NORMAL: [f64; 3] = [0.3, -0.5, 0.81];

    fn point() -> DVec3 {
        DVec3::from_array(POINT)
    }

    // planeFit.test.ts — 'Gaussian plane fit'.
    #[test]
    fn recovers_a_noisy_plane_to_micron_level() {
        let pts = sample_plane(20_000, POINT, NORMAL, 40.0, 0.02, 42);
        let fit = fit_plane_clipped(&pts, &all_indices(20_000), 3.0).expect("a fit");
        let align = fit.plane.normal.dot(unit(NORMAL)).abs();
        assert!(1.0 - align < 1e-6);
        assert!(plane_residual(&fit.plane, point()).abs() < 0.002);
        assert!(fit.sigma > 0.015);
        assert!(fit.sigma < 0.025);
        // Peak-to-peak of the kept points: wider than one sigma, no wider than the full
        // ±3-sigma clip window.
        assert!(fit.span > fit.sigma);
        assert!(fit.span <= 6.0 * fit.sigma + 1e-9);
    }

    #[test]
    fn reports_how_far_the_kept_points_reach_below_and_above_the_fit() {
        // A clean plane plus one point 30 µm above it and one 20 µm below: the extremes are
        // those two, read against the fitted normal whichever way the eigenvector happened
        // to point.
        let n = 2000;
        let mut pts = sample_plane(n, POINT, NORMAL, 40.0, 0.0, 42);
        let nn = unit(NORMAL);
        let c = point();
        pts.push(p32(c.x + 0.03 * nn.x, c.y + 0.03 * nn.y, c.z + 0.03 * nn.z));
        pts.push(p32(c.x - 0.02 * nn.x, c.y - 0.02 * nn.y, c.z - 0.02 * nn.z));
        let fit = fit_plane_clipped(&pts, &all_indices(n + 2), 0.0).expect("a fit");
        let facing = fit.plane.normal.dot(nn) > 0.0;
        let (below, above) = if facing { (fit.min, fit.max) } else { (-fit.max, -fit.min) };
        assert!((above - 0.03).abs() < 2e-4);
        assert!((below + 0.02).abs() < 2e-4);
        assert!((fit.span - (fit.max - fit.min)).abs() < 0.5e-12);
    }

    #[test]
    fn refuses_a_collinear_point_set() {
        let pts: Vec<DVec3> = (0..30).map(|i| p32(i as f64 * 0.5, i as f64 * 0.25, -(i as f64) * 0.1)).collect();
        assert!(fit_plane_tls(&pts, &all_indices(30)).is_none());
    }

    #[test]
    fn sigma_clipping_sheds_sparse_gross_outliers() {
        let mut pts = sample_plane(20_000, POINT, NORMAL, 40.0, 0.02, 44);
        let nn = unit(NORMAL);
        // Lift 2% of the points 0.5 mm off the surface — dust / spikes.
        let mut rand = Mulberry32::new(99);
        for _ in 0..400 {
            let j = rand.index(20_000);
            let p = pts[j];
            pts[j] = p32(p.x + nn.x * 0.5, p.y + nn.y * 0.5, p.z + nn.z * 0.5);
        }
        let all = fit_plane_clipped(&pts, &all_indices(20_000), 0.0).expect("a fit");
        let clipped = fit_plane_clipped(&pts, &all_indices(20_000), 3.0).expect("a fit");
        assert!(clipped.used.len() < 20_000);
        assert!(clipped.sigma < all.sigma / 3.0);
        assert!(plane_residual(&clipped.plane, point()).abs() < 0.005);
    }

    // planeFit.test.ts — 'RANSAC plane estimate'.
    #[test]
    fn locks_onto_the_clicked_face_when_the_patch_straddles_an_edge() {
        // 3,000 points on the wanted face, 2,000 on a perpendicular neighbour — a plain
        // least-squares plane would tilt into the corner between them.
        let mut all = sample_plane(3_000, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 20.0, 0.02, 45);
        all.extend(sample_plane(2_000, [10.0, 0.0, 5.0], [1.0, 0.0, 0.0], 10.0, 0.02, 46));
        let res = ransac_plane(&all, &all_indices(5_000), RansacOptions { seed: Some(3), ..Default::default() }).expect("a plane");
        assert!(res.plane.normal.z.abs() > 0.9999);
        assert!(res.plane.d.abs() < 0.005);
        assert!(res.sigma < 0.05);
    }

    // Not in ScanRuler's tests (it checks the split through its segmentation's region
    // report): the split on point sets made for it. A shallow crease is two planes; a flat
    // noisy patch and a bowed cap are one surface each.
    fn with_normals(points: Vec<DVec3>, normal_at: impl Fn(DVec3) -> DVec3) -> (Vec<DVec3>, Vec<DVec3>) {
        let normals = points.iter().map(|&p| normal_at(p)).collect();
        (points, normals)
    }

    #[test]
    fn splits_a_shallow_crease_into_its_two_faces() {
        // Two faces meeting along the y axis at 8° either side of flat, 30 mm each way.
        let tilt = 8f64.to_radians();
        let mut rand = Mulberry32::new(5);
        let mut pts = Vec::new();
        for _ in 0..6_000 {
            let x = (rand.next_f64() - 0.5) * 60.0;
            let y = (rand.next_f64() - 0.5) * 30.0;
            let z = x.abs() * tilt.tan() + rand.gaussian() * 0.01;
            pts.push(p32(x, y, z));
        }
        let (positions, normals) = with_normals(pts, |p| if p.x < 0.0 { DVec3::new(tilt.sin(), 0.0, tilt.cos()) } else { DVec3::new(-tilt.sin(), 0.0, tilt.cos()) });
        let split = split_two_planes(&positions, &normals, &all_indices(positions.len())).expect("two planes");
        assert!(split.rms_two < 0.02);
        for plane in split.planes {
            assert!((plane.normal.z.abs() - tilt.cos()).abs() < 1e-3);
        }
        let share = split.members[1].len() as f64 / positions.len() as f64;
        assert!(share > 0.45 && share <= 0.5);
    }

    #[test]
    fn reads_a_flat_patch_and_a_bowed_cap_as_one_surface() {
        let flat = sample_plane(6_000, POINT, NORMAL, 40.0, 0.02, 11);
        let n = unit(NORMAL);
        let (positions, normals) = with_normals(flat, |_| n);
        assert!(split_two_planes(&positions, &normals, &all_indices(positions.len())).is_none());

        let mut rand = Mulberry32::new(6);
        let r = 200.0;
        let mut cap = Vec::new();
        for _ in 0..6_000 {
            let x = (rand.next_f64() - 0.5) * 60.0;
            let y = (rand.next_f64() - 0.5) * 60.0;
            cap.push(p32(x, y, (r * r - x * x - y * y).sqrt() - r + rand.gaussian() * 0.005));
        }
        let (positions, normals) = with_normals(cap, |p| (p - DVec3::new(0.0, 0.0, -r)).normalize());
        assert!(split_two_planes(&positions, &normals, &all_indices(positions.len())).is_none());
    }
}
