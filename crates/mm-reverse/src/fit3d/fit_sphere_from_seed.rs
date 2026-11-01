// SPDX-License-Identifier: AGPL-3.0-only
//! A sphere from a click or from a painted selection, ready to be an element. The clicked
//! path turns down what only passes for a ball — a flat, a rod, a stray region — by its size
//! against the part's and its noise against its radius; a painted selection is taken as
//! given. Either fits with a known diameter held when the settings carry one, and turns it
//! down when the surface plainly is not a ball of that size.
//!
//! Ported from ScanRuler (`src/core/fit/fitSphereFromSeed.ts`), © Stefan Hermann
//! (CNC Kitchen), AGPL-3.0-only.

use std::cell::RefCell;

use super::guide::{guide_refusal, guided_estimate, guided_seeds, has_guide, judge_round, radial_from, FitGuide, GuideSeen};
use super::js_text::to_fixed;
use super::linalg3::{hypot3, js_max};
use super::ransac::{ransac_sphere, RansacOptions};
use super::region_grow::{grow_sphere_region, on_graph, seeded_fit};
use super::selection::require_selection;
use super::sphere::{fit_sphere_clipped, ClippedSphereFit};
use super::types::{AxialWindow, FitBase, FitSettings, HeldDiameter, RegionFit, SphereFit};
use crate::fit2d::FitError;
use crate::mesh_graph::MeshGraph;

/// Full auto-fit pipeline from a single user click:
/// 1. a breadth-first patch around the seed,
/// 2. robust RANSAC/LMedS sphere estimate on the patch,
/// 3. model-guided region growing across the whole sphere surface,
/// 4. final Gaussian best fit with the user's "used points" sigma preset.
///
/// Retried with a larger patch when the local neighbourhood is too small or too flat to pin
/// the sphere down. The window is a cylinder's and is not used.
pub fn fit_sphere_from_seed(g: &MeshGraph, seeds: &[u32], settings: &FitSettings, _window: Option<&AxialWindow>, guide: Option<&FitGuide>) -> Result<RegionFit<SphereFit>, FitError> {
    let seeds = on_graph(g, seeds);
    let guided = has_guide(guide);
    let seen: RefCell<Vec<GuideSeen>> = RefCell::new(Vec::new());
    let mut held: Vec<ClippedSphereFit> = Vec::new();
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
                    |p| ransac_sphere(&g.positions, p, opts),
                    |c| judge_round(g, guided, c.sphere.r, &c.inliers, radial_from(c.sphere.center), true),
                    |c, v| {
                        let s = c.sphere;
                        let q = g.positions[v as usize];
                        let d = hypot3(q.x - s.center.x, q.y - s.center.y, q.z - s.center.z);
                        (d - s.r).abs() <= js_max(3.0 * c.sigma, 0.01 * s.r)
                    },
                    &mut seen.borrow_mut(),
                ),
                None => ransac_sphere(&g.positions, patch, opts),
            }
        },
        |cand| {
            let from = match guided {
                Some(guided) => guided_seeds(g, &cand.inliers, guided, radial_from(cand.sphere.center)),
                None => seeds.clone(),
            };
            if from.is_empty() {
                return None;
            }
            grow_sphere_region(g, &from, cand.sphere, cand.sigma, cand.inliers.len(), guided)
        },
        |grown| {
            let Some(fin) = fit_sphere_clipped(&g.positions, &grown.region, settings.k(), held_radius(settings)) else { return Ok(None) };
            let sphere = fin.sphere;
            if !sphere.r.is_finite() || sphere.r <= 0.0 {
                return Ok(None);
            }
            // A known diameter far off what the surface fits freely is said as such, before
            // its residuals turn it down as no ball at all.
            if held_too_far(&fin) {
                held.push(fin);
                return Ok(None);
            }
            // A "sphere" larger than the model or noisier than 3% of its radius is a misfit
            // (flat area, rod, junk region) — try a larger patch or fail.
            if sphere.r > 0.8 * g.bbox_diag {
                return Ok(None);
            }
            if fin.sigma > 0.03 * sphere.r {
                return Ok(None);
            }
            if let Some(guided) = guided {
                if let Some(no) = judge_round(g, guided, sphere.r, &grown.region, radial_from(sphere.center), false) {
                    seen.borrow_mut().push(no);
                    return Ok(None);
                }
            }
            Ok(Some(sphere_output(&fin, grown.region)))
        },
    )?;
    if let Some(fit) = fit {
        return Ok(fit);
    }
    if let Some(last) = held.last() {
        return Err(FitError::new(held_refusal(last)));
    }
    if let Some(guided) = guided {
        return Err(FitError::new(guide_refusal("sphere", guided, &seen.borrow())));
    }
    Err(FitError::new("Couldn't fit a sphere at this point — try clicking nearer the middle of a sphere."))
}

/// Best-fit sphere on a hand-painted selection — the marked points as given, with the user's
/// outlier cut-off and nothing else in the way.
pub fn fit_sphere_on_selection(g: &MeshGraph, selection: &[u32], settings: &FitSettings) -> Result<RegionFit<SphereFit>, FitError> {
    let selection = on_graph(g, selection);
    require_selection(&selection, "sphere")?;
    let fin = fit_sphere_clipped(&g.positions, &selection, settings.k(), held_radius(settings))
        .filter(|f| f.sphere.r.is_finite() && f.sphere.r > 0.0)
        .ok_or_else(|| FitError::new("Couldn't fit a sphere to the marked surface — it curves too little to place a centre. Mark more of the ball."))?;
    if held_too_far(&fin) {
        return Err(FitError::new(held_refusal(&fin)));
    }
    Ok(sphere_output(&fin, selection))
}

/// The radius the settings hold, if they hold one.
fn held_radius(settings: &FitSettings) -> Option<f64> {
    settings.known_diameter.filter(|d| d.is_finite() && *d > 0.0).map(|d| d / 2.0)
}

/// A known diameter further off the one the surface fits freely than this — a millimetre,
/// or a tenth of it on a large ball — is the wrong ball or a typo, not a scan reading a
/// little off.
fn held_limit(diameter: f64) -> f64 {
    js_max(1.0, 0.1 * diameter)
}

fn held_too_far(fin: &ClippedSphereFit) -> bool {
    match fin.free {
        None => false,
        Some(free) => (2.0 * free.r - 2.0 * fin.sphere.r).abs() > held_limit(2.0 * fin.sphere.r),
    }
}

fn held_refusal(fin: &ClippedSphereFit) -> String {
    let known = 2.0 * fin.sphere.r;
    let free = fin.free.map_or(f64::NAN, |f| 2.0 * f.r);
    format!(
        "The surface here fits a sphere of Ø {} mm, too far from the known Ø {} mm to hold it — check the diameter, or that this is the ball meant.",
        to_fixed(free, 3),
        to_fixed(known, 3)
    )
}

/// The sphere fitted to `region`, as an element takes it.
fn sphere_output(fin: &ClippedSphereFit, region: Vec<u32>) -> RegionFit<SphereFit> {
    let sphere = fin.sphere;
    let held = fin.free.map(|free| HeldDiameter {
        free_diameter: 2.0 * free.r,
        shift: hypot3(free.center.x - sphere.center.x, free.center.y - sphere.center.y, free.center.z - sphere.center.z),
    });
    let fit = SphereFit {
        base: FitBase {
            sigma: fin.sigma,
            used_points: fin.used.len(),
            region_size: region.len(),
            form_error: Some(fin.span),
            residual_min: Some(fin.min),
            residual_max: Some(fin.max),
        },
        center: sphere.center,
        radius: sphere.r,
        held,
    };
    RegionFit::new(fit, region)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit3d::guide::{GuideBox, RadiusRange, Side};
    use crate::fit3d::test_meshes::cup_mesh;
    use crate::fit3d::testkit::{all_indices, graph_indexed, graph_soup, icosphere, seed_near};
    use crate::fit3d::types::FitMethodId;
    use glam::DVec3;

    const SETTINGS: FitSettings = FitSettings { method: FitMethodId::Gaussian, sigma: 3, known_diameter: None };

    fn close(a: f64, b: f64, digits: i32) -> bool {
        (a - b).abs() < 10f64.powi(-digits) / 2.0
    }

    #[test]
    fn grows_from_one_vertex_to_the_whole_sphere_and_nails_the_radius() {
        let r = 10.0;
        let graph = graph_indexed(icosphere(5, r, 0.005, 7)); // 10,242 vertices
        let out = fit_sphere_from_seed(&graph, &[0], &SETTINGS, None, None).unwrap();
        assert!((out.radius - r).abs() < 0.01);
        assert!(out.center.length() < 0.01);
        assert!(out.base.region_size as f64 > graph.vertex_count as f64 * 0.95);
        assert!(out.base.sigma < 0.01);
    }

    #[test]
    fn recovers_the_centre_from_a_marked_cap() {
        let r = 10.0;
        let graph = graph_indexed(icosphere(4, r, 0.005, 7));
        let selection: Vec<u32> = (0..graph.vertex_count as u32).filter(|&v| graph.positions[v as usize].z > r * 0.5).collect();
        let out = fit_sphere_on_selection(&graph, &selection, &SETTINGS).unwrap();
        assert!((out.radius - r).abs() < 0.05);
        assert!(out.center.length() < 0.05);
        assert_eq!(out.base.region_size, selection.len());
    }

    fn ball() -> MeshGraph {
        graph_indexed(icosphere(4, 5.0, 0.0, 7))
    }

    #[test]
    fn holds_it_from_a_click_and_from_a_marked_surface_and_reports_the_free_diameter_and_the_shift() {
        let g = ball();
        let top: Vec<u32> = (0..g.vertex_count as u32).filter(|&v| g.positions[v as usize].z > 2.0).collect();
        let marked = fit_sphere_on_selection(&g, &top, &FitSettings { known_diameter: Some(10.02), ..SETTINGS }).unwrap();
        assert_eq!(marked.radius, 5.01);
        let held = marked.held.unwrap();
        assert!(held.free_diameter > 9.9);
        assert!(held.free_diameter < 10.01);
        assert!(held.shift > 0.0);
        // The bigger ball held on the same cap sits further back from it.
        assert!(marked.center.z < 0.0);

        let clicked = fit_sphere_from_seed(&g, &[top[0]], &FitSettings { known_diameter: Some(10.0), ..SETTINGS }, None, None).unwrap();
        assert_eq!(clicked.radius, 5.0);
        assert!(clicked.held.is_some());
        assert!(fit_sphere_from_seed(&g, &[top[0]], &SETTINGS, None, None).unwrap().held.is_none());
    }

    #[test]
    fn refuses_a_diameter_the_surface_plainly_is_not() {
        let g = ball();
        let all = all_indices(g.vertex_count);
        let wrong = FitSettings { known_diameter: Some(14.0), ..SETTINGS };
        let e = fit_sphere_on_selection(&g, &all, &wrong).unwrap_err();
        assert!(e.0.contains("known Ø 14.000"), "{}", e.0);
        let e = fit_sphere_from_seed(&g, &[0], &wrong, None, None).unwrap_err();
        assert!(e.0.contains("known Ø 14.000"), "{}", e.0);
        // Within a millimetre it is held, however far the scan reads off.
        assert_eq!(fit_sphere_on_selection(&g, &all, &FitSettings { known_diameter: Some(10.8), ..SETTINGS }).unwrap().radius, 5.4);
    }

    fn cup() -> (MeshGraph, u32) {
        let graph = graph_soup(cup_mesh(6.5, 10.0, 0.004));
        let seed = seed_near(&graph, [8.25, 0.0, 0.0]);
        (graph, seed)
    }

    #[test]
    fn takes_the_cups_inside_from_a_seed_on_its_rim() {
        let (graph, seed) = cup();
        let inside = fit_sphere_from_seed(&graph, &[seed], &SETTINGS, None, Some(&FitGuide { side: Some(Side::Inside), ..FitGuide::default() })).unwrap();
        assert!(close(inside.radius, 6.5, 1), "{}", inside.radius);
    }

    #[test]
    fn and_its_outside_when_asked_for_that() {
        let (graph, seed) = cup();
        let outside = fit_sphere_from_seed(&graph, &[seed], &SETTINGS, None, Some(&FitGuide { side: Some(Side::Outside), ..FitGuide::default() })).unwrap();
        assert!(close(outside.radius, 10.0, 1), "{}", outside.radius);
    }

    #[test]
    fn takes_the_size_asked_for_over_the_other() {
        let (graph, seed) = cup();
        let small = FitGuide { radius: Some(RadiusRange { min: None, max: Some(8.0) }), ..FitGuide::default() };
        let large = FitGuide { radius: Some(RadiusRange { min: Some(8.0), max: None }), ..FitGuide::default() };
        assert!(close(fit_sphere_from_seed(&graph, &[seed], &SETTINGS, None, Some(&small)).unwrap().radius, 6.5, 1));
        assert!(close(fit_sphere_from_seed(&graph, &[seed], &SETTINGS, None, Some(&large)).unwrap().radius, 10.0, 1));
    }

    #[test]
    fn refuses_a_size_nothing_there_has_naming_what_it_found() {
        let (graph, seed) = cup();
        let guide = FitGuide { radius: Some(RadiusRange { min: Some(20.0), max: Some(30.0) }), ..FitGuide::default() };
        let e = fit_sphere_from_seed(&graph, &[seed], &SETTINGS, None, Some(&guide)).unwrap_err();
        let at = e.0.find("radius R20–R30").expect("the range is named");
        assert!(e.0[at..].contains("Found R"), "{}", e.0);
    }

    #[test]
    fn keeps_the_region_inside_a_box() {
        let (graph, seed) = cup();
        let guide = FitGuide { side: Some(Side::Inside), within: Some(GuideBox { min: DVec3::new(0.0, -20.0, -20.0), max: DVec3::new(20.0, 20.0, 1.0) }), ..FitGuide::default() };
        let out = fit_sphere_from_seed(&graph, &[seed], &SETTINGS, None, Some(&guide)).unwrap();
        assert!(close(out.radius, 6.5, 1));
        for &v in &out.region {
            assert!(graph.positions[v as usize].x >= 0.0);
        }
    }
}
