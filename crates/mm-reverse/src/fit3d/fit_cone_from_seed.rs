// SPDX-License-Identifier: AGPL-3.0-only
//! A cone from a click or from a painted selection, ready to be an element: the cylinder's
//! seed-and-grow pipeline with the cone's estimators in it, the checks that turn down a flat
//! or a stray patch passing for a taper, and the report — the radius at the middle and at
//! both ends read off the taper, the half-angle in degrees, the arc of the surface the scan
//! covers.
//!
//! Ported from ScanRuler (`src/core/fit/fitConeFromSeed.ts`), © Stefan Hermann
//! (CNC Kitchen), AGPL-3.0-only.

use std::cell::RefCell;

use glam::DVec3;

use super::cone::{cone_axis_from_normals, cone_residual, fit_cone_clipped, fit_cone_on_axis, ransac_cone, ClippedConeFit};
use super::extents::axial_extents;
use super::guide::{axis_agrees, axis_off, guide_refusal, guided_estimate, guided_seeds, has_guide, judge_round, FitGuide, GuideSeen};
use super::linalg3::{hypot3, js_max};
use super::ransac::{RansacOptions, DEFAULT_SEED};
use super::region_grow::{grow_cone_region, on_graph, seeded_fit};
use super::selection::require_selection;
use super::types::{AxialWindow, Cone, ConeFit, FitBase, FitSettings, RegionFit};
use crate::fit2d::FitError;
use crate::mesh_graph::MeshGraph;

/// The reported geometry of a cone: everything of `ConeFit` but the fit's numbers.
#[derive(Debug, Clone, Copy, PartialEq)]
struct ConeReport {
    center: DVec3,
    axis: DVec3,
    half_angle: f64,
    radius: f64,
    radius1: f64,
    radius2: f64,
    length: f64,
    coverage: f64,
}

/// The fitted extent turned into the reported geometry: the axis point moves to the middle of
/// the surface, and the radii at the middle and both ends are read off the taper. `None` when
/// the extent is degenerate or the surface sits on the wrong side of the apex.
fn cone_report(positions: &[DVec3], region: &[u32], c: &Cone) -> Option<ConeReport> {
    let ext = axial_extents(positions, region, c.point, c.axis);
    let length = ext.max_t - ext.min_t;
    if !(length > 0.0) {
        return None;
    }
    let tan = c.phi.tan();
    let mid = (ext.min_t + ext.max_t) / 2.0;
    let radius = c.r + mid * tan;
    if !(radius > 0.0) || !radius.is_finite() {
        return None;
    }
    Some(ConeReport {
        center: DVec3::new(c.point.x + mid * c.axis.x, c.point.y + mid * c.axis.y, c.point.z + mid * c.axis.z),
        axis: c.axis,
        half_angle: (c.phi * 180.0) / std::f64::consts::PI,
        radius,
        // The small end can graze the apex; a hair of numerical overshoot past it is not a
        // negative radius.
        radius1: js_max(0.0, c.r + ext.min_t * tan),
        radius2: c.r + ext.max_t * tan,
        length,
        coverage: ext.coverage,
    })
}

/// The cone's outward surface normal at a point, for a guide's side.
fn cone_outward(c: Cone) -> impl Fn(DVec3) -> Option<DVec3> {
    let (cos, sin) = (c.phi.cos(), c.phi.sin());
    move |p| {
        let (qx, qy, qz) = (p.x - c.point.x, p.y - c.point.y, p.z - c.point.z);
        let t = qx * c.axis.x + qy * c.axis.y + qz * c.axis.z;
        let (wx, wy, wz) = (qx - t * c.axis.x, qy - t * c.axis.y, qz - t * c.axis.z);
        let rho = hypot3(wx, wy, wz);
        if !(rho > 1e-12) {
            return None;
        }
        Some(DVec3::new((cos * wx) / rho - sin * c.axis.x, (cos * wy) / rho - sin * c.axis.y, (cos * wz) / rho - sin * c.axis.z))
    }
}

/// Full auto-fit pipeline from a single user click — the cylinder's pipeline with the cone
/// estimators plugged in:
/// 1. a breadth-first patch around the seed,
/// 2. robust LMedS cone estimate on the patch (normal-plane axis plus
///    three-point-with-normals candidates),
/// 3. model-guided region growing across the whole conical surface,
/// 4. final Gaussian best fit with the user's "used points" sigma preset.
///
/// Retried with a larger patch when the local neighbourhood is too small or too flat to pin
/// the axis down. The window is a cylinder's and is not used.
pub fn fit_cone_from_seed(g: &MeshGraph, seeds: &[u32], settings: &FitSettings, _window: Option<&AxialWindow>, guide: Option<&FitGuide>) -> Result<RegionFit<ConeFit>, FitError> {
    let seeds = on_graph(g, seeds);
    let guided = has_guide(guide);
    let seen: RefCell<Vec<GuideSeen>> = RefCell::new(Vec::new());
    let fit = seeded_fit(
        g,
        &seeds,
        |patch, seed| {
            let opts = RansacOptions { seed: Some(seed), ..RansacOptions::default() };
            match guided {
                Some(guided) => guided_estimate(
                    g,
                    patch,
                    guided,
                    |p| ransac_cone(&g.positions, &g.normals, p, opts),
                    |c| {
                        let k = c.cone;
                        if !axis_agrees(guided, k.axis) {
                            Some(axis_off(guided, k.r, k.axis, k.point, c.inliers.len()))
                        } else {
                            judge_round(g, guided, k.r, &c.inliers, cone_outward(k), true)
                        }
                    },
                    |c, v| cone_residual(&c.cone, g.positions[v as usize]).abs() <= js_max(3.0 * c.sigma, 0.01 * c.cone.r),
                    &mut seen.borrow_mut(),
                ),
                None => ransac_cone(&g.positions, &g.normals, patch, opts),
            }
        },
        |cand| {
            let from = match guided {
                Some(guided) => guided_seeds(g, &cand.inliers, guided, cone_outward(cand.cone)),
                None => seeds.clone(),
            };
            if from.is_empty() {
                return None;
            }
            grow_cone_region(g, &from, cand.cone, cand.sigma, cand.inliers.len(), guided)
        },
        |grown| {
            let Some(fin) = fit_cone_clipped(&g.positions, &grown.region, grown.model, settings.k()) else { return Ok(None) };
            let c = fin.cone;
            if !c.r.is_finite() || c.r <= 0.0 {
                return Ok(None);
            }
            // Steeper than ~83° is a face, not a taper — the plane tool's territory.
            if !(c.phi.abs() < 1.45) {
                return Ok(None);
            }
            let Some(report) = cone_report(&g.positions, &grown.region, &c) else { return Ok(None) };
            // Out of scale with the model, or noisier than 3% of the mid radius: this was a
            // flat patch or a junk region, not a cone.
            if report.radius2 > 0.8 * g.bbox_diag {
                return Ok(None);
            }
            if fin.sigma > 0.03 * report.radius {
                return Ok(None);
            }
            // Under ~30° of arc the circle through the points is barely curved, so the radius
            // and the axis position are guesses.
            if !(report.coverage >= 29.0) {
                return Ok(None);
            }
            if let Some(guided) = guided {
                let no = if !axis_agrees(guided, report.axis) {
                    Some(axis_off(guided, report.radius, report.axis, report.center, grown.region.len()))
                } else {
                    judge_round(g, guided, report.radius, &grown.region, cone_outward(c), false)
                };
                if let Some(no) = no {
                    seen.borrow_mut().push(no);
                    return Ok(None);
                }
            }
            Ok(Some(cone_output(&report, &fin, grown.region)))
        },
    )?;
    if let Some(fit) = fit {
        return Ok(fit);
    }
    if let Some(guided) = guided {
        return Err(FitError::new(guide_refusal("cone", guided, &seen.borrow())));
    }
    Err(FitError::new("Couldn't fit a cone at this point — try clicking in the middle of the tapered surface, away from its edges."))
}

/// Best-fit cone on a hand-painted selection. The axis still has to be guessed before the
/// six-parameter fit can be refined: the normal-plane axis is the cheap and accurate answer
/// for a clean selection, and a robust LMedS estimate takes over when the marked surface is
/// noisy enough that the normals alone cannot pin the direction down.
pub fn fit_cone_on_selection(g: &MeshGraph, selection: &[u32], settings: &FitSettings) -> Result<RegionFit<ConeFit>, FitError> {
    let selection = on_graph(g, selection);
    require_selection(&selection, "conical surface")?;

    let mut init = cone_axis_from_normals(&g.normals, &selection).and_then(|axis| fit_cone_on_axis(&g.positions, &selection, axis));
    if init.is_none() {
        init = ransac_cone(&g.positions, &g.normals, &selection, RansacOptions { seed: Some(DEFAULT_SEED), ..RansacOptions::default() }).map(|r| r.cone);
    }
    let init = init.ok_or_else(|| FitError::new("Couldn't find an axis in the marked surface — it curves too little to be told from a plane. Mark more of the way around the cone."))?;

    let fin = fit_cone_clipped(&g.positions, &selection, init, settings.k())
        .filter(|f| f.cone.r.is_finite() && f.cone.r > 0.0 && f.cone.phi.abs() < 1.45)
        .ok_or_else(|| FitError::new("Couldn't fit a cone to the marked surface."))?;
    let report = cone_report(&g.positions, &selection, &fin.cone).ok_or_else(|| FitError::new("Couldn't measure the extent of the marked surface."))?;
    Ok(cone_output(&report, &fin, selection))
}

/// The cone fitted to `region`, as an element takes it.
fn cone_output(report: &ConeReport, fin: &ClippedConeFit, region: Vec<u32>) -> RegionFit<ConeFit> {
    let fit = ConeFit {
        base: FitBase {
            sigma: fin.sigma,
            used_points: fin.used.len(),
            region_size: region.len(),
            form_error: Some(fin.span),
            residual_min: Some(fin.min),
            residual_max: Some(fin.max),
        },
        center: report.center,
        axis: report.axis,
        half_angle: report.half_angle,
        radius: report.radius,
        radius1: report.radius1,
        radius2: report.radius2,
        length: report.length,
        coverage: report.coverage,
    };
    RegionFit::new(fit, region)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit3d::testkit::{cone_mesh, graph_soup, seed_near};
    use crate::fit3d::types::FitMethodId;

    const SETTINGS: FitSettings = FitSettings { method: FitMethodId::Gaussian, sigma: 3, known_diameter: None };

    const R1: f64 = 4.0;
    const R2: f64 = 10.0;
    const LENGTH: f64 = 40.0;

    fn frustum() -> MeshGraph {
        graph_soup(cone_mesh(R1, R2, LENGTH, 64, 24, 0.01, 11).0)
    }

    #[test]
    fn grows_along_the_tapered_wall_without_climbing_onto_the_end_caps() {
        let half_angle = ((R2 - R1) / LENGTH).atan() * 180.0 / std::f64::consts::PI;
        let graph = frustum();
        let seed = seed_near(&graph, [(R1 + R2) / 2.0, 0.0, 0.0]);
        let out = fit_cone_from_seed(&graph, &[seed], &SETTINGS, None, None).unwrap();

        assert!((out.half_angle - half_angle).abs() < 0.1);
        assert!((out.radius - (R1 + R2) / 2.0).abs() < 0.05);
        // The axis points the way the radius grows: +Z.
        assert!(out.axis.z > 0.9999);
        assert!(out.center.x.hypot(out.center.y) < 0.02);
        assert!(out.base.sigma < 0.02);
        assert!(out.coverage > 352.0);
        // Stopping short of the rims caps the measured length and the end radii.
        assert!(out.length < LENGTH);
        assert!(out.length > LENGTH - 4.0 * (LENGTH / 24.0));
        assert!(out.radius1 > R1 - 0.05);
        assert!(out.radius2 < R2 + 0.05);
        assert!(out.radius1 < out.radius2);
    }

    #[test]
    fn fails_on_the_flat_end_cap_instead_of_jumping_to_the_wall() {
        let graph = frustum();
        let cap_seed = seed_near(&graph, [0.0, 0.0, LENGTH / 2.0]);
        let e = fit_cone_from_seed(&graph, &[cap_seed], &SETTINGS, None, None).unwrap_err();
        assert!(e.0.to_lowercase().contains("cone"));
    }

    #[test]
    fn recovers_half_angle_and_axis_from_a_marked_120_degree_band_of_the_wall() {
        let tan = (R2 - R1) / LENGTH;
        let graph = frustum();
        let selection: Vec<u32> = (0..graph.vertex_count as u32)
            .filter(|&v| {
                let p = graph.positions[v as usize];
                let rho = p.x.hypot(p.y);
                (rho - (R1 + (p.z + LENGTH / 2.0) * tan)).abs() <= 0.2 && p.z.abs() <= LENGTH / 4.0 && p.y.atan2(p.x).abs() < std::f64::consts::PI / 3.0
            })
            .collect();
        let out = fit_cone_on_selection(&graph, &selection, &SETTINGS).unwrap();

        assert!((out.half_angle - tan.atan() * 180.0 / std::f64::consts::PI).abs() < 0.2);
        assert!((out.radius - (R1 + R2) / 2.0).abs() < 0.05);
        assert!(out.axis.z > 0.999);
        assert!(out.center.x.hypot(out.center.y) < 0.05);
        // The marked band is half the shaft, and reaches ~120° around it.
        assert!((out.length - LENGTH / 2.0).abs() < 1.0);
        assert!(out.coverage > 110.0);
        assert!(out.coverage < 130.0);
        assert_eq!(out.base.region_size, selection.len());
    }
}
