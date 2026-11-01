// SPDX-License-Identifier: AGPL-3.0-only
//! A cylinder from a click or from a painted selection, ready to be an element: patch,
//! robust estimate, a region grown along the whole wall, the clipped best fit, and the checks
//! that turn down a flat or a stray region passing for one. The fit can also be confined to a
//! span along the axis; the reported length then stays the whole surface's, so the span is a
//! fixed distance in from the feature's ends rather than creeping inward with every refit,
//! while axis, radius and sigma come from the kept surface alone.
//!
//! Ported from ScanRuler (`src/core/fit/fitCylinderFromSeed.ts`), © Stefan Hermann
//! (CNC Kitchen), AGPL-3.0-only.

use std::cell::RefCell;

use glam::DVec3;

use super::cylinder::{axis_from_normals, cylinder_residual, fit_cylinder_clipped, fit_cylinder_on_axis, ransac_cylinder, ClippedCylinderFit};
use super::extents::axial_extents;
use super::guide::{axis_agrees, axis_off, guide_refusal, guided_estimate, guided_seeds, has_guide, judge_round, keep_where, radial_off, unique_by_last, FitGuide, GuideSeen};
use super::js_text::{js_number, to_fixed};
use super::linalg3::{js_max, js_min, js_round};
use super::ransac::{RansacOptions, DEFAULT_SEED};
use super::region_grow::{grow_cylinder_region, on_graph, seeded_fit};
use super::selection::{require_selection, MIN_SELECTED};
use super::types::{AxialWindow, Cylinder, CylinderFit, FitBase, FitSettings, FitTold, RegionFit, SecondSurface};
use crate::fit2d::{solve_linear, FitError};
use crate::mesh_graph::MeshGraph;

type CylinderResult = RegionFit<CylinderFit>;

/// The fit as it is reported: the axis, the radius and the residual numbers off the finished
/// best fit, the arc and the point counts off the surface it ran on (`used`), the middle and
/// the length off how far `extent` reaches along the axis (`None`: `used` itself). The two
/// surfaces are the same one unless the fit was confined to a window — then the length still
/// says how much of the feature the scan covered, while everything else is the kept surface's
/// own.
fn cylinder_result(positions: &[DVec3], fin: &ClippedCylinderFit, used: Vec<u32>, extent: Option<&[u32]>) -> Option<CylinderResult> {
    let c = fin.cylinder;
    let on_used = axial_extents(positions, &used, c.point, c.axis);
    let span = match extent {
        None => on_used,
        Some(extent) => axial_extents(positions, extent, c.point, c.axis),
    };
    let length = span.max_t - span.min_t;
    if !(length > 0.0) {
        return None;
    }
    let mid = (span.min_t + span.max_t) / 2.0;
    let fit = CylinderFit {
        base: FitBase {
            sigma: fin.sigma,
            used_points: fin.used.len(),
            region_size: used.len(),
            form_error: Some(fin.span),
            residual_min: Some(fin.min),
            residual_max: Some(fin.max),
        },
        center: DVec3::new(c.point.x + mid * c.axis.x, c.point.y + mid * c.axis.y, c.point.z + mid * c.axis.z),
        axis: c.axis,
        radius: c.r,
        length,
        coverage: on_used.coverage,
    };
    Some(RegionFit::new(fit, used))
}

/// The fit again, on only the surface inside the drawn span.
///
/// The span is measured off the fit already in hand — its two ends, moved by the window — so
/// what is kept is exactly what the user sees kept when an end grip is pulled in. The best
/// fit then starts from that fit and runs on the kept vertices alone. What comes back keeps
/// the length of the whole surface (measured along the new axis), so the window stays a
/// fixed pair of numbers off the feature's ends rather than creeping in with every re-fit,
/// while the axis, the radius, the sigma and the point counts are the kept surface's own.
///
/// A window that only reaches outward takes nothing away, and the fit stands.
fn fit_inside_window(g: &MeshGraph, full: CylinderResult, window: &AxialWindow, settings: &FitSettings) -> Result<CylinderResult, FitError> {
    if !(window.start < 0.0) && !(window.end < 0.0) {
        return Ok(full);
    }
    let region = &full.region;
    let (a, c) = (full.axis, full.center);
    let half = full.length / 2.0;
    let lo = -half - window.start;
    let hi = half + window.end;

    let kept: Vec<u32> = region
        .iter()
        .copied()
        .filter(|&v| {
            let p = g.positions[v as usize];
            let t = (p.x - c.x) * a.x + (p.y - c.y) * a.y + (p.z - c.z) * a.z;
            t >= lo && t <= hi
        })
        .collect();
    if kept.len() < MIN_SELECTED {
        return Err(FitError::new(format!(
            "Only {} of the {} points lie inside the span — a fit needs at least {MIN_SELECTED}. Pull the ends back out, or switch the option off.",
            kept.len(),
            region.len()
        )));
    }

    let init = Cylinder { point: c, axis: a, r: full.radius };
    let fin = fit_cylinder_clipped(&g.positions, &kept, init, settings.k())
        .filter(|f| f.cylinder.r.is_finite() && f.cylinder.r > 0.0)
        .ok_or_else(|| FitError::new("Couldn't fit a cylinder to the surface inside the span."))?;
    cylinder_result(&g.positions, &fin, kept, Some(region)).ok_or_else(|| FitError::new("Couldn't measure the length of the surface inside the span."))
}

/// Why a surface the search found was turned down.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Why {
    Noise,
    Arc,
    Scale,
}

/// A surface the search found and turned down, with what it saw of it.
#[derive(Debug, Clone, Copy, PartialEq)]
struct TurnedDown {
    radius: f64,
    sigma: f64,
    /// How far the region it grew reaches along the axis, mm.
    length: f64,
    coverage: f64,
    why: Why,
}

/// What a surface turned down was, in the words a refusal lists it with — the numbers first,
/// then what they suggest.
fn turned_down_as(s: &TurnedDown) -> String {
    let over = format!("σ {} over {} mm along the axis", to_fixed(s.sigma, 2), to_fixed(s.length, 1));
    match s.why {
        Why::Noise => format!("{over} — two surfaces? give within, or seed nearer one end"),
        Why::Arc => format!("{over}, {}° of arc — too little of the round to pin it; seed where more of it was scanned", js_number(js_round(s.coverage))),
        Why::Scale => "out of scale with the part — a flat, not a round".to_owned(),
    }
}

/// What the checks every surface passes made of a grown region: a fit that stands, or what
/// was seen of one turned down.
enum Judged {
    Stands(Box<(CylinderResult, ClippedCylinderFit)>),
    Turned(TurnedDown),
}

/// Full auto-fit pipeline from a single user click:
/// 1. a breadth-first patch around the seed,
/// 2. robust LMedS cylinder estimate on the patch (normal-covariance axis plus
///    two-point-with-normals candidates),
/// 3. model-guided region growing across the whole cylindrical surface,
/// 4. final Gaussian best fit with the user's "used points" sigma preset.
///
/// Retried with a larger patch when the local neighbourhood is too small or too flat to pin
/// the axis down.
///
/// A region noisier than the radius allows has most often run on past a step — a bore half a
/// millimetre wider below a flange: the band is refitted from each round's σ, so it widens
/// with the region and takes both. Such a region is parted along the axis where two cylinders
/// fit it best (see `part_along`), the part the seed is on fitted on its own, and the other
/// answered as `told.second`. What is turned down on the way is listed in the refusal when
/// nothing stands.
pub fn fit_cylinder_from_seed(g: &MeshGraph, seeds: &[u32], settings: &FitSettings, window: Option<&AxialWindow>, guide: Option<&FitGuide>) -> Result<CylinderResult, FitError> {
    let seeds = on_graph(g, seeds);
    let guided = has_guide(guide);
    let seen: RefCell<Vec<GuideSeen>> = RefCell::new(Vec::new());
    let mut turned: Vec<TurnedDown> = Vec::new();
    let radial = |c: &Cylinder| radial_off(c.point, c.axis);

    // The best fit on a grown region with the checks every surface passes, or what was seen
    // when it is turned down.
    let judge = |region: &[u32], model: Cylinder| -> Option<Judged> {
        let fin = fit_cylinder_clipped(&g.positions, region, model, settings.k())?;
        let c = fin.cylinder;
        if !c.r.is_finite() || c.r <= 0.0 {
            return None;
        }
        let span = axial_extents(&g.positions, region, c.point, c.axis);
        let seen_as = |why: Why| Judged::Turned(TurnedDown { radius: c.r, sigma: fin.sigma, length: span.max_t - span.min_t, coverage: span.coverage, why });
        // Out of scale with the model, or noisier than 3% of the radius: this was a flat
        // patch or a junk region, not a cylinder.
        if c.r > 0.8 * g.bbox_diag {
            return Some(seen_as(Why::Scale));
        }
        if fin.sigma > 0.03 * c.r {
            return Some(seen_as(Why::Noise));
        }
        // Under ~30° of arc the circle through the points is barely curved, so the radius and
        // the axis position are guesses.
        match cylinder_result(&g.positions, &fin, region.to_vec(), None) {
            Some(result) if result.coverage >= 29.0 => Some(Judged::Stands(Box::new((result, fin)))),
            _ => Some(seen_as(Why::Arc)),
        }
    };

    // The guide's verdict on a fit that passed the checks: `None` when it stands.
    let guide_says = |result: &CylinderResult, c: &Cylinder, region: &[u32]| -> Option<GuideSeen> {
        let guided = guided?;
        if !axis_agrees(guided, result.axis) {
            Some(axis_off(guided, c.r, result.axis, result.center, region.len()))
        } else {
            judge_round(g, guided, c.r, region, radial(c), false)
        }
    };

    // A region that read as two surfaces, parted along the axis: the seed's part fitted and
    // checked, moved in from the cut toward the seed while its fit is still too noisy — the
    // step's own rounding, a wall that tapers into it — with the other part as
    // `told.second`.
    let parted = |wide: &[u32], model: Cylinder, first: &TurnedDown| -> Option<CylinderResult> {
        let seed = *seeds.first()?;
        let part = part_along(&g.positions, wide, model, first.sigma, seed)?;
        for &cut in &part.cuts {
            let near = part.near_at(cut);
            if near.len() < MIN_SELECTED {
                break;
            }
            let Some(Judged::Stands(stands)) = judge(&near, model) else { continue };
            let (mut r, fin) = *stands;
            if guide_says(&r, &fin.cylinder, &near).is_some() {
                return None;
            }
            let ends = axial_extents(&g.positions, &part.far, r.center, r.axis);
            let second = SecondSurface { radius: part.far_fit.cylinder.r, band: [ends.min_t, ends.max_t], share: part.far.len() as f64 / wide.len() as f64 };
            let note = format!(
                "Grown freely the surface read σ {} over {} mm — two surfaces along the axis. Parted where two cylinders fit it best: R{} over {} mm where the seed is (σ {}); R{} from {} to {} mm along the axis from its centre (second). Give within to fit the other.",
                to_fixed(first.sigma, 2),
                to_fixed(first.length, 1),
                to_fixed(r.radius, 3),
                to_fixed(r.length, 1),
                to_fixed(r.base.sigma, 3),
                to_fixed(second.radius, 3),
                to_fixed(second.band[0], 1),
                to_fixed(second.band[1], 1)
            );
            r.told = Some(FitTold { note, second: Some(second) });
            return Some(r);
        }
        None
    };

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
                    |p| ransac_cylinder(&g.positions, &g.normals, p, opts),
                    |c| {
                        let cy = c.cylinder;
                        if !axis_agrees(guided, cy.axis) {
                            Some(axis_off(guided, cy.r, cy.axis, cy.point, c.inliers.len()))
                        } else {
                            judge_round(g, guided, cy.r, &c.inliers, radial(&cy), true)
                        }
                    },
                    |c, v| {
                        let e = cylinder_residual(&c.cylinder, g.positions[v as usize]);
                        e.abs() <= js_max(3.0 * c.sigma, 0.01 * c.cylinder.r)
                    },
                    &mut seen.borrow_mut(),
                ),
                None => ransac_cylinder(&g.positions, &g.normals, patch, opts),
            }
        },
        |cand| {
            let from = match guided {
                Some(guided) => guided_seeds(g, &cand.inliers, guided, radial(&cand.cylinder)),
                None => seeds.clone(),
            };
            if from.is_empty() {
                return None;
            }
            grow_cylinder_region(g, &from, cand.cylinder, cand.sigma, cand.inliers.len(), guided)
        },
        |grown| {
            let Some(j) = judge(&grown.region, grown.model) else { return Ok(None) };
            let result = match j {
                Judged::Turned(t) => {
                    let result = if t.why == Why::Noise { parted(&grown.region, grown.model, &t) } else { None };
                    match result {
                        Some(r) => r,
                        None => {
                            // Remembered for the refusal — with a guide in the guide's list.
                            turned.push(t);
                            if guided.is_some() {
                                seen.borrow_mut().push(GuideSeen::because(t.radius, turned_down_as(&t)));
                            }
                            return Ok(None);
                        }
                    }
                }
                Judged::Stands(stands) => {
                    let (result, fin) = *stands;
                    if let Some(no) = guide_says(&result, &fin.cylinder, &grown.region) {
                        seen.borrow_mut().push(no);
                        return Ok(None);
                    }
                    result
                }
            };

            // The surface has been found; confining the fit to part of it is a separate
            // matter, and one a bigger patch could not help with.
            let Some(window) = window else { return Ok(Some(result)) };
            let told = result.told.clone();
            let mut inside = fit_inside_window(g, result, window, settings)?;
            if told.is_some() {
                inside.told = told;
            }
            Ok(Some(inside))
        },
    )?;
    if let Some(fit) = fit {
        return Ok(fit);
    }
    if let Some(guided) = guided {
        return Err(FitError::new(guide_refusal("cylinder", guided, &seen.borrow())));
    }
    // The rounds first, then the flats; one line per size.
    let mut found = unique_by_last(&turned, |t| to_fixed(t.radius, 2));
    found.sort_by_key(|t| t.why == Why::Scale);
    found.truncate(3);
    let said = if found.is_empty() {
        String::new()
    } else {
        format!(" Found {}.", found.iter().map(|t| format!("R{} ({})", to_fixed(t.radius, 2), turned_down_as(t))).collect::<Vec<_>>().join(", "))
    };
    Err(FitError::new(format!("Couldn't fit a cylinder at this point.{said} Try clicking on a clearly curved part of the cylindrical surface.")))
}

/// The cuts along the axis a region is tried parted at: this many over its length, but never
/// closer than this share of the radius.
const PART_CUTS: f64 = 64.0;
const PART_STEP_OF_R: f64 = 0.05;
/// Points each side of a cut is fitted on during the search, at most; the part kept is
/// fitted on all of its own.
const PART_SAMPLE: usize = 3000;
/// Two parts are taken over one when they leave this share of its σ or less, and of the σ
/// left once a radius changing smoothly along the axis is taken out.
const PART_GAIN: f64 = 0.7;

/// A region parted along the axis: the cuts to try from the best one toward the seed, the
/// other part at the best cut, and its fit.
struct PartAlong<'a> {
    cuts: Vec<f64>,
    far: Vec<u32>,
    far_fit: ClippedCylinderFit,
    region: &'a [u32],
    positions: &'a [DVec3],
    model: Cylinder,
    toward: f64,
}

impl PartAlong<'_> {
    fn along(&self, v: u32) -> f64 {
        along(self.positions, &self.model, v)
    }

    /// The seed's part at a cut.
    fn near_at(&self, cut: f64) -> Vec<u32> {
        keep_where(self.region, |v| self.toward * (self.along(v) - cut) > 0.0)
    }
}

/// How far vertex `v` lies along the cylinder's axis from its point.
fn along(positions: &[DVec3], c: &Cylinder, v: u32) -> f64 {
    let p = positions[v as usize];
    (p.x - c.point.x) * c.axis.x + (p.y - c.point.y) * c.axis.y + (p.z - c.point.z) * c.axis.z
}

/// A region that is two cylinders one after the other along the axis — a bore and a
/// counterbore, a bore that steps wider into a boss — parted at the cut where a cylinder
/// fitted on each side leaves the least scatter. Unlike `split_two_cylinders`, which gives
/// each point to the nearer of two cylinders, this holds when the wider one is off centre and
/// its wall meets the narrower one's down one side. `None` when no cut leaves under
/// `PART_GAIN` of the one cylinder's σ.
///
/// `cuts` runs from the best cut toward the seed, a step at a time, and `near_at` gives the
/// seed's part at one — for the caller to take the first that fits as one surface; `far` is
/// the other part at the best cut, with its fit.
fn part_along<'a>(positions: &'a [DVec3], region: &'a [u32], model: Cylinder, sigma_one: f64, seed: u32) -> Option<PartAlong<'a>> {
    let c = model;
    let t: Vec<f64> = region.iter().map(|&v| along(positions, &c, v)).collect();
    let ts = along(positions, &c, seed);
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for &x in &t {
        if x < lo {
            lo = x;
        }
        if x > hi {
            hi = x;
        }
    }
    let step = js_max((hi - lo) / PART_CUTS, PART_STEP_OF_R * c.r);
    let stride = (region.len() / PART_SAMPLE).max(1);
    let sides = |cut: f64, every: usize| -> (Vec<u32>, Vec<u32>) {
        let (mut near, mut far) = (Vec::new(), Vec::new());
        let up = ts > cut;
        for i in (0..region.len()).step_by(every) {
            if (t[i] > cut) == up {
                near.push(region[i]);
            } else {
                far.push(region[i]);
            }
        }
        (near, far)
    };
    let mut best: Option<(f64, f64)> = None;
    let mut cut = lo + 2.0 * step;
    while cut <= hi - 2.0 * step {
        let (near, far) = sides(cut, stride);
        if near.len() >= 30 && far.len() >= 30 {
            if let (Some(a), Some(b)) = (fit_cylinder_clipped(positions, &near, c, 3.0), fit_cylinder_clipped(positions, &far, c, 3.0)) {
                let (nn, nf) = (near.len() as f64, far.len() as f64);
                let rms = ((a.sigma.powi(2) * nn + b.sigma.powi(2) * nf) / (nn + nf)).sqrt();
                if rms.is_finite() && best.is_none_or(|(_, best_rms)| rms < best_rms) {
                    best = Some((cut, rms));
                }
            }
        }
        // A step too small to move the cut would never end the walk.
        if cut + step == cut {
            break;
        }
        cut += step;
    }
    // A radius that changes smoothly along the axis — a taper, a cone fitted as a cylinder —
    // parts as well as a step does; a quadratic in t takes that out, and the parts must beat
    // it too.
    let mut m = [0.0f64; 9];
    let mut rhs = [0.0f64; 3];
    let mut e = vec![0.0f64; region.len()];
    for i in 0..region.len() {
        e[i] = cylinder_residual(&c, positions[region[i] as usize]);
        let row = [1.0, t[i], t[i] * t[i]];
        for a in 0..3 {
            rhs[a] += row[a] * e[i];
            for b in 0..3 {
                m[a * 3 + b] += row[a] * row[b];
            }
        }
    }
    let k = solve_linear(3, &mut m, &rhs);
    let smooth = match &k {
        Some(k) => {
            let mut ss = 0.0;
            for i in 0..region.len() {
                ss += (e[i] - k[0] - k[1] * t[i] - k[2] * t[i] * t[i]).powi(2);
            }
            (ss / region.len() as f64).sqrt()
        }
        None => sigma_one,
    };
    let (best_cut, best_rms) = best?;
    if !(best_rms < PART_GAIN * js_min(sigma_one, smooth)) {
        return None;
    }
    let (_, far) = sides(best_cut, 1);
    let far_fit = (if far.len() >= MIN_SELECTED { fit_cylinder_clipped(positions, &far, c, 3.0) } else { None })?;
    let toward = if ts > best_cut { 1.0 } else { -1.0 };
    let mut cuts = Vec::new();
    let mut cut = best_cut;
    while toward * (ts - cut) > step {
        cuts.push(cut);
        if cut + toward * step == cut {
            break;
        }
        cut += toward * step;
    }
    Some(PartAlong { cuts, far, far_fit, region, positions, model, toward })
}

/// Best-fit cylinder on a hand-painted selection. The axis still has to be guessed before the
/// five-parameter fit can be refined: the normal-covariance axis is the cheap and accurate
/// answer for a clean selection, and a robust LMedS estimate takes over when the marked
/// surface is noisy enough that the normals alone cannot pin the direction down.
pub fn fit_cylinder_on_selection(g: &MeshGraph, selection: &[u32], settings: &FitSettings, window: Option<&AxialWindow>) -> Result<CylinderResult, FitError> {
    let selection = on_graph(g, selection);
    require_selection(&selection, "cylindrical surface")?;

    let mut init = axis_from_normals(&g.normals, &selection).and_then(|axis| fit_cylinder_on_axis(&g.positions, &selection, axis));
    if init.is_none() {
        init = ransac_cylinder(&g.positions, &g.normals, &selection, RansacOptions { seed: Some(DEFAULT_SEED), ..RansacOptions::default() }).map(|r| r.cylinder);
    }
    let init = init.ok_or_else(|| {
        FitError::new("Couldn't find an axis in the marked surface — it curves too little to be told from a plane. Mark more of the way around the cylinder.")
    })?;

    let fin = fit_cylinder_clipped(&g.positions, &selection, init, settings.k())
        .filter(|f| f.cylinder.r.is_finite() && f.cylinder.r > 0.0)
        .ok_or_else(|| FitError::new("Couldn't fit a cylinder to the marked surface."))?;
    let result = cylinder_result(&g.positions, &fin, selection, None).ok_or_else(|| FitError::new("Couldn't measure the length of the marked surface."))?;
    match window {
        Some(window) => fit_inside_window(g, result, window, settings),
        None => Ok(result),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit2d::Mulberry32;
    use crate::fit3d::guide::{RadiusRange, Side};
    use crate::fit3d::test_meshes::{plated_hole, stepped_bore, SteppedBore};
    use crate::fit3d::testkit::{cylinder_mesh, graph_soup, p32, seed_near};
    use crate::fit3d::types::FitMethodId;

    const SETTINGS: FitSettings = FitSettings { method: FitMethodId::Gaussian, sigma: 3, known_diameter: None };

    fn close(a: f64, b: f64, digits: i32) -> bool {
        (a - b).abs() < 10f64.powi(-digits) / 2.0
    }

    /// What follows two ASCII digits at the start of `s`.
    fn two_digits(s: &str) -> Option<&str> {
        let b = s.as_bytes();
        (b.len() >= 2 && b[0].is_ascii_digit() && b[1].is_ascii_digit()).then(|| &s[2..])
    }

    /// What follows `\d+\.\d` at the start of `s`.
    fn one_decimal(s: &str) -> Option<&str> {
        let int = s.bytes().take_while(u8::is_ascii_digit).count();
        let b = s.as_bytes();
        (int > 0 && b.get(int) == Some(&b'.') && b.get(int + 1).is_some_and(u8::is_ascii_digit)).then(|| &s[int + 2..])
    }

    mod shaft {
        use super::*;

        const R: f64 = 8.0;
        const LENGTH: f64 = 40.0;
        const RADIAL: usize = 64;
        const AXIAL: usize = 24;

        fn graph() -> MeshGraph {
            graph_soup(cylinder_mesh(R, LENGTH, RADIAL, AXIAL, 0.01, 11).0)
        }

        #[test]
        fn grows_along_the_wall_without_climbing_onto_the_end_caps() {
            let graph = graph();
            let seed = seed_near(&graph, [R, 0.0, 0.0]);
            let out = fit_cylinder_from_seed(&graph, &[seed], &SETTINGS, None, None).unwrap();

            assert!((out.radius - R).abs() < 0.01);
            assert!(out.axis.z.abs() > 0.9999);
            assert!(out.center.x.hypot(out.center.y) < 0.01);
            assert!(out.base.sigma < 0.02);
            // A 64-facet ring leaves 5.6° between neighbouring vertices, so a fully wrapped
            // region reads as 360° minus one of those gaps.
            assert!(out.coverage > 352.0);

            // The wall is (AXIAL + 1) rings; the two rims are shared with the caps, so their
            // averaged normals put them out of the region — everything else is in, and
            // nothing from the caps is.
            let wall_rings = AXIAL + 1;
            assert!(out.base.region_size > (wall_rings - 4) * RADIAL);
            assert!(out.base.region_size <= (wall_rings - 2) * RADIAL);
            // Stopping short of the rims also caps the measured length.
            assert!(out.length < LENGTH);
            assert!(out.length > LENGTH - 4.0 * (LENGTH / AXIAL as f64));
        }

        #[test]
        fn fails_on_the_flat_end_cap_instead_of_jumping_to_the_wall() {
            let graph = graph();
            let cap_seed = seed_near(&graph, [0.0, 0.0, LENGTH / 2.0]);
            let e = fit_cylinder_from_seed(&graph, &[cap_seed], &SETTINGS, None, None).unwrap_err();
            assert!(e.0.to_lowercase().contains("cylinder"));
        }
    }

    /// A surface of revolution about +Z, radius `r_of(z)`, rings `h` apart, with Gaussian
    /// radial noise. Open at both ends.
    fn revolve(r_of: impl Fn(f64) -> f64, z_lo: f64, z_hi: f64, h: f64, radial: usize, noise: f64) -> Vec<DVec3> {
        let mut rand = Mulberry32::new(9);
        let rings = ((z_hi - z_lo) / h).round() as usize;
        let grid: Vec<Vec<[f64; 3]>> = (0..=rings)
            .map(|j| {
                let z = z_lo + j as f64 * h;
                (0..radial)
                    .map(|k| {
                        let a = (k as f64 / radial as f64) * 2.0 * std::f64::consts::PI;
                        let r = r_of(z) + rand.gaussian() * noise;
                        [a.cos() * r, a.sin() * r, z]
                    })
                    .collect()
            })
            .collect();
        let mut tris = Vec::new();
        for j in 0..rings {
            for k in 0..radial {
                let k2 = (k + 1) % radial;
                let (a, b, c, d) = (grid[j][k], grid[j][k2], grid[j + 1][k2], grid[j + 1][k]);
                for p in [a, b, c, a, c, d] {
                    tris.push(p32(p[0], p[1], p[2]));
                }
            }
        }
        tris
    }

    #[test]
    fn a_shaft_with_rounded_ends_is_measured_on_the_straight_wall() {
        const NOISE: f64 = 0.01;
        const H: f64 = 0.25;
        const R: f64 = 8.0;
        const HALF: f64 = 3.0;
        const EDGE: f64 = 1.5; // rounding radius at each end
        let r_of = |z: f64| {
            let e = (z.abs() - (HALF - EDGE)).max(0.0);
            R - (EDGE - (EDGE * EDGE - e * e).sqrt())
        };
        // Rings land on the wall's end and every 0.25 mm into the roll-off, the same offsets
        // as the face's.
        let graph = graph_soup(revolve(r_of, -(HALF - EDGE) - 1.25, HALF - EDGE + 1.25, H, 64, NOISE));
        let out = fit_cylinder_from_seed(&graph, &[seed_near(&graph, [R, 0.0, 0.0])], &SETTINGS, None, None).unwrap();

        assert!(out.axis.z.abs() > 0.9999);
        // The first ring of each roll-off alone would take 2 µm off the radius.
        assert!((out.radius - R).abs() < 0.001, "{}", out.radius);
        for &v in &out.region {
            assert!(graph.positions[v as usize].z.abs() <= HALF - EDGE + 1e-6);
        }
        let wall_rings = (2.0 * (HALF - EDGE)) / H + 1.0;
        assert!(out.base.region_size as f64 > 0.75 * wall_rings * 64.0);
        assert!(out.coverage > 352.0);
    }

    #[test]
    fn recovers_radius_and_axis_from_a_marked_120_degree_band_of_the_wall() {
        const R: f64 = 8.0;
        const LENGTH: f64 = 40.0;
        let graph = graph_soup(cylinder_mesh(R, LENGTH, 64, 24, 0.01, 11).0);
        let selection: Vec<u32> = (0..graph.vertex_count as u32)
            .filter(|&v| {
                let p = graph.positions[v as usize];
                let rho = p.x.hypot(p.y);
                (rho - R).abs() <= 0.2 && p.z.abs() <= LENGTH / 4.0 && p.y.atan2(p.x).abs() < std::f64::consts::PI / 3.0
            })
            .collect();
        let out = fit_cylinder_on_selection(&graph, &selection, &SETTINGS, None).unwrap();

        assert!((out.radius - R).abs() < 0.02);
        assert!(out.axis.z.abs() > 0.999);
        assert!(out.center.x.hypot(out.center.y) < 0.05);
        // The marked band is half the shaft, and reaches ~120° around it.
        assert!((out.length - LENGTH / 2.0).abs() < 1.0);
        assert!(out.coverage > 110.0);
        assert!(out.coverage < 130.0);
        assert_eq!(out.base.region_size, selection.len());
    }

    /// A cylinder fit confined to the drawn span: the surface past an end that was pulled in
    /// stays out of the best fit, and only out of the best fit — the length reported still
    /// says how far the scan reached.
    mod window {
        use super::*;

        const R: f64 = 8.0;
        const LENGTH: f64 = 40.0;
        /// The top of the bore came out badly: over the last 8 mm the wall flares outward, up
        /// to 0.06 mm at the rim on a 0.02 mm noise floor — the kind of rounded-off mouth a
        /// scanner leaves on a hole. Mild on purpose: a gross flare is thrown out by the sigma
        /// clipping on its own, and it is the one that is not that bends a fit.
        const NOISE: f64 = 0.02;
        const FLARE_FROM: f64 = LENGTH / 2.0 - 8.0;
        const FLARE: f64 = 0.06;

        fn flared_cylinder() -> MeshGraph {
            let (mut soup, _) = cylinder_mesh(R, LENGTH, 64, 24, NOISE, 11);
            for p in &mut soup {
                let r = p.x.hypot(p.y);
                if p.z <= FLARE_FROM || r < R * 0.5 {
                    continue;
                }
                let k = 1.0 + (FLARE * (p.z - FLARE_FROM)) / (LENGTH / 2.0 - FLARE_FROM) / R;
                // In place in a Float32Array: each scaled coordinate is stored in single
                // precision.
                *p = p32(p.x * k, p.y * k, p.z);
            }
            graph_soup(soup)
        }

        /// Every wall vertex — the flared ones included — but neither end cap.
        fn wall(g: &MeshGraph) -> Vec<u32> {
            (0..g.vertex_count as u32)
                .filter(|&v| {
                    let p = g.positions[v as usize];
                    p.x.hypot(p.y) > R * 0.5 && p.z.abs() < LENGTH / 2.0 - 1e-3
                })
                .collect()
        }

        struct Setup {
            graph: MeshGraph,
            selection: Vec<u32>,
            full: CylinderResult,
            wall_span: f64,
        }

        fn setup() -> Setup {
            let graph = flared_cylinder();
            let selection = wall(&graph);
            let full = fit_cylinder_on_selection(&graph, &selection, &SETTINGS, None).unwrap();
            // How far along Z the marked wall reaches — the rim rows belong to the caps and
            // are not in it, so this is a little short of the rod itself.
            let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
            for &v in &selection {
                let z = graph.positions[v as usize].z;
                lo = lo.min(z);
                hi = hi.max(z);
            }
            Setup { graph, selection, full, wall_span: hi - lo }
        }

        /// The window that pulls the flared end in by `mm`: which of the two sides that is
        /// depends on which way the fit happened to point its axis.
        fn off_top(s: &Setup, mm: f64) -> AxialWindow {
            if s.full.axis.z > 0.0 {
                AxialWindow { start: 0.0, end: -mm }
            } else {
                AxialWindow { start: -mm, end: 0.0 }
            }
        }

        #[test]
        fn leaves_the_surface_past_a_pulled_in_end_out_of_the_fit() {
            let s = setup();
            let out = fit_cylinder_on_selection(&s.graph, &s.selection, &SETTINGS, Some(&off_top(&s, 8.0))).unwrap();
            let count_below = s.selection.iter().filter(|&&v| s.graph.positions[v as usize].z < LENGTH / 2.0 - 8.0).count();
            // Exactly the vertices inside the span, give or take the one row of the mesh a
            // slightly tilted axis can move across the cut.
            assert!(out.base.region_size.abs_diff(count_below) <= 64);
            assert_eq!(out.region.len(), out.base.region_size);
            assert!(out.base.used_points <= out.base.region_size);
            assert!(out.base.region_size < s.selection.len());
        }

        #[test]
        fn finds_the_true_diameter_once_the_flare_is_out_of_the_way() {
            let s = setup();
            let out = fit_cylinder_on_selection(&s.graph, &s.selection, &SETTINGS, Some(&off_top(&s, 8.0))).unwrap();
            // The flare pulls the whole-surface fit a few microns wide; inside the span the
            // radius comes back to within the noise.
            assert!((s.full.radius - R).abs() > 0.003, "{}", s.full.radius);
            assert!((out.radius - R).abs() < 0.001, "{}", out.radius);
            assert!(out.base.sigma < s.full.base.sigma);
        }

        #[test]
        fn still_reports_the_length_the_scan_reached() {
            let s = setup();
            let out = fit_cylinder_on_selection(&s.graph, &s.selection, &SETTINGS, Some(&off_top(&s, 8.0))).unwrap();
            assert!((out.length - s.wall_span).abs() < 0.05);
            assert!((out.length - s.full.length).abs() < 0.05);
            assert!(out.center.z.abs() < 0.1);
            assert!(out.coverage > 350.0);
        }

        #[test]
        fn takes_nothing_away_when_the_span_only_reaches_outward() {
            let s = setup();
            let out = fit_cylinder_on_selection(&s.graph, &s.selection, &SETTINGS, Some(&AxialWindow { start: 3.0, end: 5.0 })).unwrap();
            assert_eq!(out.radius, s.full.radius);
            assert_eq!(out.base.region_size, s.full.base.region_size);
            assert_eq!(out.length, s.full.length);
        }

        #[test]
        fn refuses_a_span_with_too_little_surface_left_inside_it() {
            let s = setup();
            let e = fit_cylinder_on_selection(&s.graph, &s.selection, &SETTINGS, Some(&AxialWindow { start: -LENGTH, end: -LENGTH })).unwrap_err();
            assert!(e.0.contains("inside the span"), "{}", e.0);
        }

        #[test]
        fn applies_to_a_fit_grown_from_a_click_the_same_way() {
            let s = setup();
            let graph = &s.graph;
            // A seed on the wall, well away from the flare.
            let seed = (0..graph.vertex_count as u32)
                .find(|&v| {
                    let p = graph.positions[v as usize];
                    p.x.hypot(p.y) > R * 0.5 && p.z.abs() < 0.9
                })
                .expect("a seed on the wall");
            let grown = fit_cylinder_from_seed(graph, &[seed], &SETTINGS, None, None).unwrap();
            let window = if grown.axis.z > 0.0 { AxialWindow { start: 0.0, end: -8.0 } } else { AxialWindow { start: -8.0, end: 0.0 } };
            let out = fit_cylinder_from_seed(graph, &[seed], &SETTINGS, Some(&window), None).unwrap();
            assert!(out.base.region_size < grown.base.region_size);
            assert!((grown.radius - R).abs() > 0.003, "{}", grown.radius);
            assert!((out.radius - R).abs() < 0.001, "{}", out.radius);
            assert!((out.length - grown.length).abs() < 0.05);
        }
    }

    /// A fit told the axis: a seed at a hole's middle snaps to its mouth, between the wall and
    /// the face.
    mod told_the_axis {
        use super::*;

        fn setup() -> (MeshGraph, u32) {
            let graph = graph_soup(plated_hole(3.0, 30.0, 6.0, 0.004));
            // The hole's middle at the top face: the nearest vertex is on its mouth.
            let seed = seed_near(&graph, [0.0, 0.0, 3.0]);
            (graph, seed)
        }

        fn axis(a: [f64; 3]) -> FitGuide {
            FitGuide { axis: Some(DVec3::from_array(a)), ..FitGuide::default() }
        }

        #[test]
        fn finds_the_bore_from_its_mouth() {
            let (graph, seed) = setup();
            let out = fit_cylinder_from_seed(&graph, &[seed], &SETTINGS, None, Some(&axis([0.0, 0.0, 1.0]))).unwrap();
            assert!(close(out.radius, 3.0, 1), "{}", out.radius);
            assert!(out.axis.z.abs() > 0.999);
        }

        #[test]
        fn with_the_side_as_well() {
            let (graph, seed) = setup();
            let guide = FitGuide { side: Some(Side::Inside), ..axis([0.0, 0.0, 1.0]) };
            let out = fit_cylinder_from_seed(&graph, &[seed], &SETTINGS, None, Some(&guide)).unwrap();
            assert!(close(out.radius, 3.0, 1), "{}", out.radius);
        }

        #[test]
        fn refuses_a_bore_on_another_axis() {
            let (graph, seed) = setup();
            let e = fit_cylinder_from_seed(&graph, &[seed], &SETTINGS, None, Some(&axis([1.0, 0.0, 0.0]))).unwrap_err();
            assert!(e.0.contains("No cylinder there"), "{}", e.0);
        }

        #[test]
        fn names_the_axis_it_turned_down_where_it_runs_and_how_far_off_to_be_asked_for_after_all() {
            let (graph, seed) = setup();
            let said = fit_cylinder_from_seed(&graph, &[seed], &SETTINGS, None, Some(&axis([0.0, 0.5, 1.0]))).unwrap_err().0;
            // R3.dd (axis off by d.d°: axis [a, b, c] through [...], n points)
            let at = said.find(" (axis off by ").expect("an axis turned down");
            // R3\.\d\d right before it.
            let r = said[..at].rsplit_once("R3.").map(|(_, d)| d).unwrap_or_else(|| panic!("{said}"));
            assert!(r.len() == 2 && r.chars().all(|c| c.is_ascii_digit()), "{said}");
            let after = &said[at + " (axis off by ".len()..];
            // \d+\.\d°: axis [-?0(\.\d+)?, -?0(\.\d+)?, -?(1|0\.99\d*)] through [.+], \d+ points)
            let (deg, after) = after.split_once("°: axis [").unwrap_or_else(|| panic!("{said}"));
            let (whole, tenth) = deg.split_once('.').unwrap_or_else(|| panic!("{said}"));
            assert!(!whole.is_empty() && whole.chars().all(|c| c.is_ascii_digit()) && tenth.len() == 1 && tenth.chars().all(|c| c.is_ascii_digit()), "{said}");
            let (axis_text, after) = after.split_once(']').unwrap_or_else(|| panic!("{said}"));
            let comps: Vec<&str> = axis_text.split(", ").collect();
            assert_eq!(comps.len(), 3, "{said}");
            let small = |s: &str| {
                let s = s.strip_prefix('-').unwrap_or(s);
                s == "0" || s.strip_prefix("0.").is_some_and(|d| !d.is_empty() && d.chars().all(|c| c.is_ascii_digit()))
            };
            let unit = |s: &str| {
                let s = s.strip_prefix('-').unwrap_or(s);
                s == "1" || s.strip_prefix("0.99").is_some_and(|d| d.chars().all(|c| c.is_ascii_digit()))
            };
            assert!(small(comps[0]) && small(comps[1]) && unit(comps[2]), "{said}");
            let (centre, after) = after.strip_prefix(" through [").and_then(|s| s.split_once("], ")).unwrap_or_else(|| panic!("{said}"));
            assert!(!centre.is_empty(), "{said}");
            let (count, _) = after.split_once(" points)").unwrap_or_else(|| panic!("{said}"));
            assert!(!count.is_empty() && count.chars().all(|c| c.is_ascii_digit()), "{said}");
            // The bore stands on Z; the axis given leans 26.6° off it.
            assert!(said.contains("axis off by 26."), "{said}");
            assert!(said.contains("give that axis or drop axis"), "{said}");
        }
    }

    /// A bore that steps wider below a flange, fitted from a seed on the narrow part: the band
    /// stops at a square shoulder; down a drafted step it runs on into the wide part, the one
    /// fit is too noisy to stand, and the region is parted along the axis — the narrow part
    /// answered, the wide one as `second`, whether the wide one is on the same axis or off it.
    /// A bore that is noisy all the way is not parted, and its refusal says what it saw.
    mod stepped {
        use super::*;

        fn bore(noise: f64) -> SteppedBore {
            SteppedBore { noise, ..SteppedBore::new(4.5, 9.0, 4.95, 16.0) }
        }

        #[test]
        fn fits_the_narrow_part_from_a_seed_on_it_the_band_stopping_at_the_shoulder() {
            let g = graph_soup(stepped_bore(bore(0.023)));
            let out = fit_cylinder_from_seed(&g, &[seed_near(&g, [4.5, 0.0, -3.0])], &SETTINGS, None, None).unwrap();
            assert!((out.radius - 4.5).abs() < 0.02, "{}", out.radius);
            assert!((out.length - 9.0).abs() < 0.75, "{}", out.length);
            assert!(out.axis.z.abs() > 0.999);
        }

        #[test]
        fn parts_a_region_that_ran_on_down_a_drafted_step_and_names_the_wider_part() {
            // Noisier walls widen the band until it runs down the 1.2 mm draft into the wider
            // part, and the one fit is too noisy to stand.
            let g = graph_soup(stepped_bore(SteppedBore { taper: 1.2, ..bore(0.08) }));
            let out = fit_cylinder_from_seed(&g, &[seed_near(&g, [4.5, 0.0, -3.0])], &SETTINGS, None, None).unwrap();
            assert!((out.radius - 4.5).abs() < 0.02, "{}", out.radius);
            assert!((out.length - 9.0).abs() < 0.75, "{}", out.length);
            let told = out.told.clone().expect("told of the second surface");
            // ^Grown freely the surface read σ 0\.\d\d over \d+\.\d mm — two surfaces along the
            // axis\. Parted
            let rest = told.note.strip_prefix("Grown freely the surface read σ 0.").unwrap_or_else(|| panic!("{}", told.note));
            let rest = two_digits(rest).and_then(|r| r.strip_prefix(" over ")).and_then(one_decimal).unwrap_or_else(|| panic!("{}", told.note));
            assert!(rest.starts_with(" mm — two surfaces along the axis. Parted"), "{}", told.note);
            let second = told.second.expect("a second surface");
            assert!((second.radius - 4.95).abs() < 0.03, "{}", second.radius);
            // The wider part lies below this one, along the axis from its centre.
            let down = -out.axis.z.signum();
            assert!(down * (second.band[0] + second.band[1]) / 2.0 > 4.5);
            assert!(second.share > 0.4);
        }

        #[test]
        fn holds_when_the_wider_part_is_off_the_axis_its_wall_meeting_the_narrow_ones_down_one_side() {
            let g = graph_soup(stepped_bore(SteppedBore { offset: 0.3, taper: 1.2, ..bore(0.06) }));
            let out = fit_cylinder_from_seed(&g, &[seed_near(&g, [-4.5, 0.0, -4.0])], &SETTINGS, None, None).unwrap();
            assert!((out.radius - 4.5).abs() < 0.02, "{}", out.radius);
            assert!((out.length - 9.0).abs() < 1.0, "{}", out.length);
            assert!((out.told.as_ref().and_then(|t| t.second).expect("a second surface").radius - 4.95).abs() < 0.03);
        }

        #[test]
        fn leaves_a_plain_bore_as_it_was() {
            let g = graph_soup(stepped_bore(SteppedBore { noise: 0.023, ..SteppedBore::new(4.5, 12.0, 4.5, 12.0) }));
            let out = fit_cylinder_from_seed(&g, &[seed_near(&g, [4.5, 0.0, -3.0])], &SETTINGS, None, None).unwrap();
            assert!((out.radius - 4.5).abs() < 0.02);
            assert!(out.told.is_none());
        }

        fn noisy() -> MeshGraph {
            graph_soup(stepped_bore(SteppedBore { noise: 0.17, ..SteppedBore::new(4.5, 12.0, 4.5, 12.0) }))
        }

        /// `R4\.\d\d \(σ 0\.\d\d over \d+\.\d mm along the axis` at `text`'s start; what
        /// follows it.
        fn reads_found_bore(text: &str) -> Option<&str> {
            let rest = two_digits(text.strip_prefix("R4.")?)?.strip_prefix(" (σ 0.")?;
            one_decimal(two_digits(rest)?.strip_prefix(" over ")?)?.strip_prefix(" mm along the axis")
        }

        #[test]
        fn refuses_a_bore_too_noisy_all_along_with_the_size_and_sigma_it_saw() {
            let g = noisy();
            let e = fit_cylinder_from_seed(&g, &[seed_near(&g, [4.5, 0.0, -6.0])], &SETTINGS, None, None).unwrap_err();
            let head = "Couldn't fit a cylinder at this point. Found ";
            assert!(e.0.starts_with(head), "{}", e.0);
            let rest = reads_found_bore(&e.0[head.len()..]).unwrap_or_else(|| panic!("{}", e.0));
            assert!(rest.starts_with(" — two surfaces? give within, or seed nearer one end)"), "{}", e.0);
        }

        #[test]
        fn lists_the_same_under_a_guide_those_of_the_size_asked_for_first() {
            let g = noisy();
            let guide = FitGuide { radius: Some(RadiusRange { min: Some(3.5), max: Some(5.5) }), ..FitGuide::default() };
            let e = fit_cylinder_from_seed(&g, &[seed_near(&g, [4.5, 0.0, -6.0])], &SETTINGS, None, Some(&guide)).unwrap_err();
            let mark = "radius R3.5–R5.5). Found ";
            let at = e.0.find(mark).unwrap_or_else(|| panic!("{}", e.0));
            assert!(reads_found_bore(&e.0[at + mark.len()..]).is_some(), "{}", e.0);
        }
    }
}
