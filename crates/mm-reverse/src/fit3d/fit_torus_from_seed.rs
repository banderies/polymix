// SPDX-License-Identifier: AGPL-3.0-only
//! A torus from a click on a round or from a painted selection, ready to be an element.
//! Beyond fit quality the click itself has to lie on the torus and face the way its tube does
//! there: a click on the flat beside a round finds the round in its patch, and a torus leaning
//! on the flat's first rows passes on residual alone — the flat's own normal does not. A
//! spindle torus, tube wider than spine, is never taken for a scanned round.
//!
//! Ported from ScanRuler (`src/core/fit/fitTorusFromSeed.ts`), © Stefan Hermann
//! (CNC Kitchen), AGPL-3.0-only.

use std::cell::RefCell;

use super::guide::{guide_patch, FitGuide};
use super::js_text::js_number;
use super::linalg3::{js_max, js_round};
use super::ransac::RansacOptions;
use super::region_grow::{grow_torus_region, on_graph, seeded_fit};
use super::selection::require_selection;
use super::torus::{canonical_torus, fit_torus_clipped, ransac_torus, torus_coverage, torus_normal_align, torus_residual, TorusClippedFit};
use super::types::{AxialWindow, FitBase, FitSettings, RegionFit, Torus, TorusFit};
use crate::fit2d::FitError;
use crate::mesh_graph::MeshGraph;

/// The fit as an element: the axis pointing with the region's normals, the coverage read off
/// the used points.
fn output(g: &MeshGraph, t: Torus, fin: &TorusClippedFit, region: Vec<u32>) -> RegionFit<TorusFit> {
    let torus = canonical_torus(t, &g.normals, &fin.used);
    let cover = torus_coverage(&torus, &g.positions, &fin.used);
    let fit = TorusFit {
        base: FitBase {
            sigma: fin.sigma,
            used_points: fin.used.len(),
            region_size: region.len(),
            form_error: Some(fin.span),
            residual_min: Some(fin.min),
            residual_max: Some(fin.max),
        },
        center: torus.center,
        axis: torus.axis,
        major_radius: torus.major,
        minor_radius: torus.minor,
        tube_coverage: cover.tube_deg,
        spine_coverage: cover.spine_deg,
    };
    RegionFit::new(fit, region)
}

/// How far the spine may reach, in scan diagonals: a large shroud or a bowed panel is a torus
/// whose spine lies well outside the part (R592 on a part whose diagonal is a fraction of
/// that), while past a few diagonals a patch curves too little the long way for R to mean
/// anything.
pub const TORUS_MAX_R_DIAGS: f64 = 5.0;

/// Why a torus is not taken for a scanned round, and whether that is its size.
struct Implausible {
    why: String,
    size: bool,
}

/// A number as the refusals write it: whole from 100 up, to two places below.
fn round(x: f64) -> String {
    js_number(if x.abs() >= 100.0 { js_round(x) } else { js_round(x * 100.0) / 100.0 })
}

/// Why a torus is not taken for a scanned round, or `None` when it may be: one whose tube is
/// wider than its spine is a spindle, whose two sheets cross, and a scan of a round never is
/// one; a spine or a tube far larger than the scan is a surface too flat to place them.
fn implausible(t: &Torus, g: &MeshGraph) -> Option<Implausible> {
    if !t.major.is_finite() || !t.minor.is_finite() || !(t.minor > 0.0) {
        return Some(Implausible { why: "no torus".into(), size: false });
    }
    if !(t.major > t.minor * 0.999) {
        return Some(Implausible { why: format!("R {} is no larger than r {} — a spindle, not a round", round(t.major), round(t.minor)), size: false });
    }
    let max_r = TORUS_MAX_R_DIAGS * g.bbox_diag;
    if !(t.major < max_r) {
        return Some(Implausible {
            why: format!(
                "R {} is past the limit of {} × the scan's diagonal ({} mm) — the surface bends too little the long way to place the spine; fit a cylinder or a B-spline surface",
                round(t.major),
                js_number(TORUS_MAX_R_DIAGS),
                round(max_r)
            ),
            size: true,
        });
    }
    if !(t.minor < 0.5 * g.bbox_diag) {
        return Some(Implausible { why: format!("r {} is past the limit of half the scan's diagonal ({} mm)", round(t.minor), round(0.5 * g.bbox_diag)), size: true });
    }
    None
}

/// The seed's normal may lean this far from the tube's (32°) — a scan's normals scatter, and
/// the grow allows the same.
fn seed_cos_max() -> f64 {
    (32.0 * std::f64::consts::PI / 180.0).cos()
}

/// Whether the clicked vertices lie on the torus and face with it.
fn on_it(g: &MeshGraph, seeds: &[u32], t: &Torus, sigma: f64) -> bool {
    let band = js_max(3.0 * sigma, 0.02 * t.minor);
    let cos_max = seed_cos_max();
    for &v in seeds {
        let (p, n) = (g.positions[v as usize], g.normals[v as usize]);
        if torus_residual(t, p).abs() > band {
            return false;
        }
        if torus_normal_align(t, p, n) < cos_max {
            return false;
        }
    }
    true
}

/// Full auto-fit pipeline from a single click on a round: a local patch, the robust torus
/// estimate on it, model-guided growing across the whole round, the clipped best fit. The
/// patch grows when the local neighbourhood is too small or too flat to place the spine. A
/// torus takes a box alone of the guide; the window is a cylinder's and is not used.
pub fn fit_torus_from_seed(g: &MeshGraph, seeds: &[u32], settings: &FitSettings, _window: Option<&AxialWindow>, guide: Option<&FitGuide>) -> Result<RegionFit<TorusFit>, FitError> {
    let seeds = on_graph(g, seeds);
    let within = guide.and_then(|g| g.within).map(|w| FitGuide { within: Some(w), ..FitGuide::default() });
    // The last torus turned down for its size, to name the limit in the refusal.
    let refused: RefCell<Option<String>> = RefCell::new(None);
    let plausible = |t: &Torus| match implausible(t, g) {
        Some(no) => {
            if no.size {
                *refused.borrow_mut() = Some(no.why);
            }
            false
        }
        None => true,
    };
    let fit = seeded_fit(
        g,
        &seeds,
        |patch, seed| {
            let opts = RansacOptions { seed: Some(seed), ..RansacOptions::default() };
            let cand = match &within {
                Some(w) => ransac_torus(&g.positions, &g.normals, &guide_patch(g, patch, w), opts),
                None => ransac_torus(&g.positions, &g.normals, patch, opts),
            }?;
            plausible(&cand.torus).then_some(cand)
        },
        |cand| {
            let from: &[u32] = if within.is_some() { &cand.inliers } else { &seeds };
            grow_torus_region(g, from, cand.torus, cand.sigma, cand.inliers.len(), within.as_ref())
        },
        |grown| {
            let Some(fin) = fit_torus_clipped(&g.positions, &grown.region, settings.k(), grown.model) else { return Ok(None) };
            if !plausible(&fin.torus) {
                return Ok(None);
            }
            // Noisier than a few percent of the tube radius it is not the round the click was
            // on — a flat, a cylinder, a corner.
            if fin.sigma > 0.05 * fin.torus.minor {
                return Ok(None);
            }
            // And the click itself has to lie on it, facing the way the tube faces there: a
            // click on the flat beside a round finds the round in its patch and a torus that
            // leans on the flat's first rows, which the residual alone lets through — the
            // flat's own normal does not.
            if !on_it(g, &seeds, &fin.torus, fin.sigma) {
                return Ok(None);
            }
            Ok(Some(output(g, fin.torus, &fin, grown.region)))
        },
    )?;
    if let Some(fit) = fit {
        return Ok(fit);
    }
    if let Some(why) = refused.into_inner() {
        return Err(FitError::new(format!("Couldn't fit a torus at this point: the one found was turned down — {why}.")));
    }
    Err(FitError::new("Couldn't fit a torus at this point — try clicking on the middle of a round or a bend."))
}

/// Best-fit torus on a hand-painted selection: the marked points as given, the robust
/// estimate on them for a start, then the clipped fit.
pub fn fit_torus_on_selection(g: &MeshGraph, selection: &[u32], settings: &FitSettings) -> Result<RegionFit<TorusFit>, FitError> {
    let selection = on_graph(g, selection);
    require_selection(&selection, "round")?;
    let seed = selection.first().copied().unwrap_or(1);
    let cand = ransac_torus(&g.positions, &g.normals, &selection, RansacOptions { seed: Some(seed), ..RansacOptions::default() });
    let fin = cand.and_then(|c| fit_torus_clipped(&g.positions, &selection, settings.k(), c.torus));
    let no = fin.as_ref().and_then(|f| implausible(&f.torus, g));
    if let Some(no) = &no {
        if no.size {
            return Err(FitError::new(format!("Couldn't fit a torus to the marked surface: {}.", no.why)));
        }
    }
    match fin {
        Some(fin) if no.is_none() => Ok(output(g, fin.torus, &fin, selection)),
        _ => Err(FitError::new("Couldn't fit a torus to the marked surface — it has to curve two ways, like a fillet's round or a bend. Mark more of it.")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit3d::testkit::{graph_soup, p32};
    use crate::fit3d::types::FitMethodId;
    use glam::DVec3;

    const SETTINGS: FitSettings = FitSettings { method: FitMethodId::Gaussian, sigma: 3, known_diameter: None };

    fn push(tris: &mut Vec<DVec3>, ps: &[[f64; 3]]) {
        for p in ps {
            tris.push(p32(p[0], p[1], p[2]));
        }
    }

    /// A mesh of a ring with a flat either side of its round — a fillet between two faces, as
    /// a scan of a boss's edge has it — as a triangle grid.
    fn fillet_mesh(r_major: f64, r: f64, radial: usize, tube: i64) -> Vec<DVec3> {
        let ring = |a: f64, b: f64| -> [f64; 3] {
            // A quarter tube from the top flat (b = 0, pointing up) to the side (b = π/2,
            // pointing out), plus flats beyond either end.
            let rho = r_major + r * b.sin();
            let z = r * b.cos();
            [rho * a.cos(), rho * a.sin(), z]
        };
        let mut rows: Vec<Vec<[f64; 3]>> = Vec::new();
        for j in -6..=tube + 6 {
            let row = (0..radial)
                .map(|k| {
                    let a = (k as f64 / radial as f64) * 2.0 * std::f64::consts::PI;
                    if j < 0 {
                        let rr = r_major - (0 - j) as f64 * (r / 4.0);
                        [rr * a.cos(), rr * a.sin(), r]
                    } else if j > tube {
                        [(r_major + r) * a.cos(), (r_major + r) * a.sin(), -((j - tube) as f64) * (r / 4.0)]
                    } else {
                        ring(a, (std::f64::consts::PI / 2.0) * (j as f64 / tube as f64))
                    }
                })
                .collect();
            rows.push(row);
        }
        let mut tris = Vec::new();
        for j in 0..rows.len() - 1 {
            for k in 0..radial {
                let k2 = (k + 1) % radial;
                let (a, b, c, d) = (rows[j][k], rows[j][k2], rows[j + 1][k2], rows[j + 1][k]);
                push(&mut tris, &[a, b, c, a, c, d]);
            }
        }
        tris
    }

    mod fillet {
        use super::*;

        const R: f64 = 25.0;
        const SMALL_R: f64 = 3.0;

        fn graph() -> MeshGraph {
            graph_soup(fillet_mesh(R, SMALL_R, 120, 16))
        }

        fn seed_near(g: &MeshGraph, x: f64, y: f64, z: f64) -> u32 {
            crate::fit3d::testkit::seed_near(g, [x, y, z])
        }

        #[test]
        fn grows_round_the_ring_from_a_click_on_the_round_and_stops_at_the_flats() {
            let graph = graph();
            let mid = std::f64::consts::PI / 4.0;
            let rho = R + SMALL_R * mid.sin();
            let seed = seed_near(&graph, rho * 0.3f64.cos(), rho * 0.3f64.sin(), SMALL_R * mid.cos());
            let out = fit_torus_from_seed(&graph, &[seed], &SETTINGS, None, None).unwrap();
            assert!((out.major_radius - R).abs() < 0.05, "{}", out.major_radius);
            assert!((out.minor_radius - SMALL_R).abs() < 0.05, "{}", out.minor_radius);
            assert!(out.axis.z.abs() > 0.9999);
            assert!(out.center.length() < 0.05);
            assert!(out.base.sigma < 0.01);
            assert!(out.spine_coverage > 350.0);
            // A quarter of the tube, and no flat in the region.
            assert!(out.tube_coverage < 120.0);
            assert!(out.tube_coverage > 60.0);
            for &v in &out.region {
                let p = graph.positions[v as usize];
                assert!((p.x.hypot(p.y) - R).abs() <= SMALL_R + 0.05 && p.z <= SMALL_R + 0.05);
            }
            // A marked selection of the same round gives the same torus.
            let sel = fit_torus_on_selection(&graph, &out.region, &SETTINGS).unwrap();
            assert!((sel.minor_radius - SMALL_R).abs() < 0.05);
        }

        #[test]
        fn fails_on_the_flat_instead_of_inventing_a_round() {
            let graph = graph();
            let seed = seed_near(&graph, (R - SMALL_R) * 1f64.cos(), (R - SMALL_R) * 1f64.sin(), SMALL_R);
            let e = fit_torus_from_seed(&graph, &[seed], &SETTINGS, None, None).unwrap_err();
            assert!(e.0.to_lowercase().contains("torus"));
        }
    }

    /// A patch of a large torus's outer side as a triangle grid: `spine_deg` of the spine,
    /// `tube_deg` of the tube about its outermost line — a shroud or a bowed panel, whose
    /// spine lies far outside the part.
    fn torus_patch_mesh(r_major: f64, r: f64, spine_deg: f64, tube_deg: f64, nu: usize, nv: usize) -> Vec<DVec3> {
        let at = |a: f64, b: f64| -> [f64; 3] { [(r_major + r * b.cos()) * a.cos(), (r_major + r * b.cos()) * a.sin(), r * b.sin()] };
        let mut tris = Vec::new();
        let (aa, bb) = ((spine_deg * std::f64::consts::PI) / 180.0, (tube_deg * std::f64::consts::PI) / 180.0);
        for i in 0..nu {
            for j in 0..nv {
                let a0 = -aa / 2.0 + (aa * i as f64) / nu as f64;
                let a1 = -aa / 2.0 + (aa * (i + 1) as f64) / nu as f64;
                let b0 = -bb / 2.0 + (bb * j as f64) / nv as f64;
                let b1 = -bb / 2.0 + (bb * (j + 1) as f64) / nv as f64;
                let (p, q, s, t) = (at(a0, b0), at(a1, b0), at(a1, b1), at(a0, b1));
                push(&mut tris, &[p, q, s, p, s, t]);
            }
        }
        tris
    }

    #[test]
    fn fits_a_shroud_whose_spine_lies_well_past_the_part() {
        // R600 / r75 on a part a fraction of that across.
        let graph = graph_soup(torus_patch_mesh(600.0, 75.0, 40.0, 120.0, 140, 60));
        // R is past the old limit of 0.8 × the scan's diagonal.
        assert!(600.0 > 0.8 * graph.bbox_diag);
        let seed = crate::fit3d::testkit::seed_near(&graph, [675.0, 0.0, 0.0]);
        let out = fit_torus_from_seed(&graph, &[seed], &SETTINGS, None, None).unwrap();
        assert!((out.major_radius - 600.0).abs() < 1.0, "{}", out.major_radius);
        assert!((out.minor_radius - 75.0).abs() < 0.5, "{}", out.minor_radius);
        assert!(out.axis.z.abs() > 0.999);
    }

    #[test]
    fn names_the_limit_when_the_spine_lies_too_far_out() {
        // A small patch of the same torus: R600 is past 5 × its diagonal.
        let graph = graph_soup(torus_patch_mesh(600.0, 75.0, 6.0, 40.0, 60, 40));
        assert!(600.0 > 5.0 * graph.bbox_diag);
        let all: Vec<u32> = (0..graph.vertex_count as u32).collect();
        let e = fit_torus_on_selection(&graph, &all, &SETTINGS).unwrap_err();
        assert!(e.0.contains("limit of 5 × the scan's diagonal"), "{}", e.0);
    }

    #[test]
    fn writes_sizes_as_the_refusals_have_them() {
        assert_eq!(round(592.4), "592");
        assert_eq!(round(7.456), "7.46");
        assert_eq!(round(2.0), "2");
    }
}
