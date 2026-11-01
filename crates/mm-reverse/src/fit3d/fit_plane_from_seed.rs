// SPDX-License-Identifier: AGPL-3.0-only
//! A plane from a click or from a painted selection, ready to be an element. A bare plane has
//! no side and no size, so both are added: the normal is turned by a vote of the surface
//! normals to point out of the part, and the element is drawn as the smallest-area rectangle
//! round the region in the plane — the principal axes would box a square patch in
//! diagonally, well past its edges.
//!
//! Ported from ScanRuler (`src/core/fit/fitPlaneFromSeed.ts`), © Stefan Hermann
//! (CNC Kitchen), AGPL-3.0-only.

use glam::DVec3;

use super::guide::{guide_patch, FitGuide};
use super::linalg3::{js_max, ortho_basis};
use super::plane::{fit_plane_clipped, ransac_plane, ClippedPlaneFit};
use super::ransac::RansacOptions;
use super::region_grow::{grow_plane_region, on_graph, seeded_fit};
use super::selection::require_selection;
use super::types::{AxialWindow, FitBase, FitSettings, Plane, PlaneFit, RegionFit};
use crate::fit2d::FitError;
use crate::mesh_graph::MeshGraph;

/// In-plane rotations tried when boxing the patch in, over a quarter turn — past that the
/// box repeats.
const FRAME_STEPS: usize = 45;
const FRAME_SAMPLES: usize = 4096;

/// The rectangle a plane element is drawn as.
struct PatchExtents {
    center: DVec3,
    basis_u: DVec3,
    basis_v: DVec3,
    extent_u: f64,
    extent_v: f64,
}

/// The tightest rectangle around the measured patch, in the plane itself, so the element is
/// drawn as the piece of surface it was fitted to. The obvious choice — the patch's principal
/// axes — is arbitrary for a square patch and then boxes it in diagonally, sticking well out
/// past its real edges; the smallest-area frame is stable whatever the shape.
fn patch_extents(positions: &[DVec3], region: &[u32], p: &Plane) -> Option<PatchExtents> {
    let n = region.len();
    let (mut mx, mut my, mut mz) = (0.0, 0.0, 0.0);
    for &v in region {
        let q = positions[v as usize];
        mx += q.x;
        my += q.y;
        mz += q.z;
    }
    mx /= n as f64;
    my /= n as f64;
    mz /= n as f64;

    let (u0, v0) = ortho_basis(p.normal);
    let step = n.div_ceil(FRAME_SAMPLES).max(1);
    let mut sa: Vec<f64> = Vec::new();
    let mut sb: Vec<f64> = Vec::new();
    for &v in region.iter().step_by(step) {
        let q = positions[v as usize];
        let (x, y, z) = (q.x - mx, q.y - my, q.z - mz);
        sa.push(x * u0.x + y * u0.y + z * u0.z);
        sb.push(x * v0.x + y * v0.y + z * v0.z);
    }
    if sa.len() < 3 {
        return None;
    }

    let mut best_area = f64::INFINITY;
    let mut best_angle = 0.0;
    for t in 0..FRAME_STEPS {
        let angle = (t as f64 / FRAME_STEPS as f64) * (std::f64::consts::PI / 2.0);
        let (c, s) = (angle.cos(), angle.sin());
        let (mut lo1, mut hi1, mut lo2, mut hi2) = (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
        for (a, b) in sa.iter().zip(&sb) {
            let q1 = a * c + b * s;
            let q2 = -a * s + b * c;
            if q1 < lo1 {
                lo1 = q1;
            }
            if q1 > hi1 {
                hi1 = q1;
            }
            if q2 < lo2 {
                lo2 = q2;
            }
            if q2 > hi2 {
                hi2 = q2;
            }
        }
        let area = (hi1 - lo1) * (hi2 - lo2);
        if area < best_area {
            best_area = area;
            best_angle = angle;
        }
    }

    let (ca, sn) = (best_angle.cos(), best_angle.sin());
    let basis_u = DVec3::new(u0.x * ca + v0.x * sn, u0.y * ca + v0.y * sn, u0.z * ca + v0.z * sn);
    let basis_v = DVec3::new(-u0.x * sn + v0.x * ca, -u0.y * sn + v0.y * ca, -u0.z * sn + v0.z * ca);

    let (mut min_u, mut max_u, mut min_v, mut max_v) = (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
    for &v in region {
        let q = positions[v as usize];
        let (x, y, z) = (q.x - mx, q.y - my, q.z - mz);
        let pu = x * basis_u.x + y * basis_u.y + z * basis_u.z;
        let pv = x * basis_v.x + y * basis_v.y + z * basis_v.z;
        if pu < min_u {
            min_u = pu;
        }
        if pu > max_u {
            max_u = pu;
        }
        if pv < min_v {
            min_v = pv;
        }
        if pv > max_v {
            max_v = pv;
        }
    }
    if !min_u.is_finite() || !min_v.is_finite() {
        return None;
    }

    let cu = (min_u + max_u) / 2.0;
    let cv = (min_v + max_v) / 2.0;
    let cx = mx + cu * basis_u.x + cv * basis_v.x;
    let cy = my + cu * basis_u.y + cv * basis_v.y;
    let cz = mz + cu * basis_u.z + cv * basis_v.z;
    // Pull the middle of the patch onto the fitted plane itself.
    let nn = p.normal;
    let off = nn.x * cx + nn.y * cy + nn.z * cz - p.d;
    Some(PatchExtents {
        center: DVec3::new(cx - off * nn.x, cy - off * nn.y, cz - off * nn.z),
        basis_u,
        basis_v,
        extent_u: (max_u - min_u) / 2.0,
        extent_v: (max_v - min_v) / 2.0,
    })
}

/// Orient the plane normal the way the surface faces, so the reported normal points out of
/// the part rather than into it.
fn orient_to_surface(p: Plane, normals: &[DVec3], region: &[u32]) -> Plane {
    let mut vote = 0.0;
    for &v in region {
        let n = normals[v as usize];
        vote += p.normal.x * n.x + p.normal.y * n.y + p.normal.z * n.z;
    }
    if vote >= 0.0 {
        return p;
    }
    Plane { normal: -p.normal, d: -p.d }
}

/// The residual extremes as they read against the reported normal. Turning the plane to face
/// the surface turns every residual with it, so the point furthest below the fit becomes the
/// one furthest above.
fn oriented_extremes(fin: &ClippedPlaneFit, plane: &Plane) -> (f64, f64) {
    let same = fin.plane.normal.x * plane.normal.x + fin.plane.normal.y * plane.normal.y + fin.plane.normal.z * plane.normal.z > 0.0;
    if same {
        (fin.min, fin.max)
    } else {
        (-fin.max, -fin.min)
    }
}

/// The plane fitted to `region`, as an element takes it.
fn plane_output(fin: &ClippedPlaneFit, plane: &Plane, ext: &PatchExtents, region: Vec<u32>) -> RegionFit<PlaneFit> {
    let (residual_min, residual_max) = oriented_extremes(fin, plane);
    let fit = PlaneFit {
        base: FitBase {
            sigma: fin.sigma,
            used_points: fin.used.len(),
            region_size: region.len(),
            form_error: Some(fin.span),
            residual_min: Some(residual_min),
            residual_max: Some(residual_max),
        },
        center: ext.center,
        normal: plane.normal,
        basis_u: ext.basis_u,
        basis_v: ext.basis_v,
        extent_u: ext.extent_u,
        extent_v: ext.extent_v,
    };
    RegionFit::new(fit, region)
}

/// Full auto-fit pipeline from a single user click:
/// 1. a breadth-first patch around the seed,
/// 2. robust RANSAC/LMedS plane estimate on the patch,
/// 3. model-guided region growing across the whole flat surface,
/// 4. final Gaussian best fit with the user's "used points" sigma preset.
///
/// Retried with a larger patch when the local neighbourhood is too small to separate the
/// surface from its neighbours. A plane takes a box alone of the guide: it has no side, size
/// or axis. The window is a cylinder's and is not used.
pub fn fit_plane_from_seed(g: &MeshGraph, seeds: &[u32], settings: &FitSettings, _window: Option<&AxialWindow>, guide: Option<&FitGuide>) -> Result<RegionFit<PlaneFit>, FitError> {
    let seeds = on_graph(g, seeds);
    let within = guide.and_then(|g| g.within).map(|w| FitGuide { within: Some(w), ..FitGuide::default() });
    let fit = seeded_fit(
        g,
        &seeds,
        |patch, seed| {
            let opts = RansacOptions { seed: Some(seed), ..RansacOptions::default() };
            match &within {
                Some(w) => ransac_plane(&g.positions, &guide_patch(g, patch, w), opts),
                None => ransac_plane(&g.positions, patch, opts),
            }
        },
        |cand| {
            let from: &[u32] = if within.is_some() { &cand.inliers } else { &seeds };
            grow_plane_region(g, from, cand.plane, cand.sigma, cand.inliers.len(), within.as_ref())
        },
        |grown| {
            let Some(fin) = fit_plane_clipped(&g.positions, &grown.region, settings.k()) else { return Ok(None) };
            let plane = orient_to_surface(fin.plane, &g.normals, &grown.region);
            let Some(ext) = patch_extents(&g.positions, &grown.region, &plane) else { return Ok(None) };
            // A "plane" rougher than a tenth of its own size is a curved or broken surface,
            // not a flat one.
            if !(fin.sigma < 0.1 * js_max(ext.extent_u, ext.extent_v)) {
                return Ok(None);
            }
            Ok(Some(plane_output(&fin, &plane, &ext, grown.region)))
        },
    )?;
    fit.ok_or_else(|| FitError::new("Couldn't fit a plane at this point — try clicking in the middle of a flat surface, away from edges."))
}

/// Best-fit plane on a hand-painted selection: steps 4 and 5 of the pipeline above, on
/// exactly the marked points. The roughness check the seeded path applies is left out on
/// purpose — there it guards against a region that grew onto the wrong surface, and here the
/// surface is the user's own choice.
pub fn fit_plane_on_selection(g: &MeshGraph, selection: &[u32], settings: &FitSettings) -> Result<RegionFit<PlaneFit>, FitError> {
    let selection = on_graph(g, selection);
    require_selection(&selection, "surface")?;
    let fin = fit_plane_clipped(&g.positions, &selection, settings.k())
        .ok_or_else(|| FitError::new("Couldn't fit a plane to the marked surface — the points lie along a line rather than spanning an area."))?;
    let plane = orient_to_surface(fin.plane, &g.normals, &selection);
    let ext = patch_extents(&g.positions, &selection, &plane).ok_or_else(|| FitError::new("Couldn't measure the extent of the marked surface — try marking a wider patch."))?;
    Ok(plane_output(&fin, &plane, &ext, selection))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit2d::Mulberry32;
    use crate::fit3d::types::FitMethodId;
    use crate::fit3d::testkit::{box_mesh, graph_soup, p32, seed_near};

    const SETTINGS: FitSettings = FitSettings { method: FitMethodId::Gaussian, sigma: 3, known_diameter: None };

    const SIZE: f64 = 40.0;
    const GRID: usize = 20;

    fn box_graph() -> MeshGraph {
        graph_soup(box_mesh(SIZE, GRID, 0.02, 13))
    }

    #[test]
    fn grows_across_one_face_and_stops_at_its_edges() {
        let graph = box_graph();
        let seed = seed_near(&graph, [0.0, 0.0, SIZE / 2.0]);
        let out = fit_plane_from_seed(&graph, &[seed], &SETTINGS, None, None).unwrap();
        assert!(out.normal.z.abs() > 0.999);
        // The normal is oriented the way the surface faces, i.e. outward.
        assert!(out.normal.z > 0.0);
        assert!((out.center.z - SIZE / 2.0).abs() < 0.01);
        assert!(out.base.sigma > 0.012);
        assert!(out.base.sigma < 0.03);
        // One face is (GRID + 1)² vertices, of which the border ring is shared with the
        // neighbouring faces and so is left out.
        assert!(out.base.region_size > (GRID - 2).pow(2));
        assert!(out.base.region_size < (GRID + 1).pow(2));
        assert!(2.0 * out.extent_u < SIZE);
        assert!(2.0 * out.extent_u > SIZE * 0.8);
    }

    #[test]
    fn measures_the_same_distance_between_opposite_faces_from_either_side() {
        let graph = box_graph();
        let top = fit_plane_from_seed(&graph, &[seed_near(&graph, [0.0, 0.0, SIZE / 2.0])], &SETTINGS, None, None).unwrap();
        let bottom = fit_plane_from_seed(&graph, &[seed_near(&graph, [0.0, 0.0, -SIZE / 2.0])], &SETTINGS, None, None).unwrap();
        let gap = (bottom.center - top.center).dot(top.normal).abs();
        assert!((gap - SIZE).abs() < 0.01);
        // Opposite faces of a box: normals antiparallel to within a thousandth.
        assert!(top.normal.dot(bottom.normal) < -0.9999);
    }

    /// A height field z(x, y) over a square grid of pitch `h`, with Gaussian noise along Z;
    /// cells whose corner the field leaves undefined are skipped.
    fn height_field(half: f64, h: f64, z_of: impl Fn(f64, f64) -> Option<f64>, noise: f64) -> Vec<DVec3> {
        let mut rand = Mulberry32::new(7);
        let n = ((2.0 * half) / h).round() as usize;
        let grid: Vec<Vec<Option<[f64; 3]>>> = (0..=n)
            .map(|i| {
                (0..=n)
                    .map(|j| {
                        let x = -half + i as f64 * h;
                        let y = -half + j as f64 * h;
                        z_of(x, y).map(|z| [x, y, z + rand.gaussian() * noise])
                    })
                    .collect()
            })
            .collect();
        let mut tris = Vec::new();
        for i in 0..n {
            for j in 0..n {
                if let (Some(a), Some(b), Some(c), Some(d)) = (grid[i][j], grid[i + 1][j], grid[i + 1][j + 1], grid[i][j + 1]) {
                    for p in [a, b, c, a, c, d] {
                        tris.push(p32(p[0], p[1], p[2]));
                    }
                }
            }
        }
        tris
    }

    /// The rounding a scanner leaves at every edge: the surface rolls off over a radius, and
    /// the first ring or two of the roll-off sit close enough to the surface, at a normal
    /// tilted little enough, to pass the growing tests. The peel takes those rings back off.
    #[test]
    fn a_face_with_rounded_edges_is_measured_on_the_flat_not_pulled_into_the_part() {
        const NOISE: f64 = 0.01;
        const H: f64 = 0.25;
        const FLAT: f64 = 3.0; // half-size of the flat top
        const R: f64 = 1.5; // rounding radius
        // Rolls off along a quarter circle past the flat; the outer rings sit 0.021 and
        // 0.086 mm below the top, at 10° and 20° — the first is inside a 3.5σ band and the
        // 25° angle, the second is not.
        let top = |x: f64, y: f64| -> Option<f64> {
            let e = (x.abs() - FLAT).max(0.0).hypot((y.abs() - FLAT).max(0.0));
            if e >= 0.9 * R {
                return None;
            }
            Some(-(R - (R * R - e * e).sqrt()))
        };
        let graph = graph_soup(height_field(FLAT + R, H, top, NOISE));
        let out = fit_plane_from_seed(&graph, &[seed_near(&graph, [0.0, 0.0, 0.0])], &SETTINGS, None, None).unwrap();

        assert!(out.normal.z.abs() > 0.9999);
        // Unbiased: the flat is at z = 0 to within the noise on its mean; the first roll-off
        // ring alone would pull it 3 µm into the part.
        assert!(out.center.z.abs() < 0.0015, "{}", out.center.z);
        // Nothing of the roll-off is in the region…
        for &v in &out.region {
            let p = graph.positions[v as usize];
            assert!(p.x.abs().max(p.y.abs()) <= FLAT + 1e-6);
        }
        // …and most of the flat still is.
        let flat_count = (2.0 * FLAT / H + 1.0).powi(2);
        assert!(out.base.region_size as f64 > 0.75 * flat_count);
    }

    /// The vertices a brush stroke would have left behind, stood in for by a predicate over
    /// the scan's own vertices.
    fn mark(g: &MeshGraph, keep: impl Fn(DVec3) -> bool) -> Vec<u32> {
        (0..g.vertex_count as u32).filter(|&v| keep(g.positions[v as usize])).collect()
    }

    const MSIZE: f64 = 20.0;

    #[test]
    fn measures_only_the_marked_half_of_a_face_not_the_whole_face() {
        let graph = graph_soup(box_mesh(MSIZE, 20, 0.01, 13));
        let selection = mark(&graph, |p| p.z > MSIZE / 2.0 - 0.1 && p.x <= 0.001);
        let out = fit_plane_on_selection(&graph, &selection, &SETTINGS).unwrap();

        assert!(out.normal.z.abs() > 0.999);
        assert!((out.center.z - MSIZE / 2.0).abs() < 0.02);
        // Half of the face in x, all of it in y — the marked patch, not the face a grown
        // region would have covered.
        assert!((out.center.x + MSIZE / 4.0).abs() < 0.3);
        assert!(out.center.y.abs() < 0.3);
        let mut extents = [out.extent_u, out.extent_v];
        extents.sort_by(f64::total_cmp);
        assert!((extents[0] - MSIZE / 4.0).abs() < 0.3);
        assert!((extents[1] - MSIZE / 2.0).abs() < 0.3);
        assert_eq!(out.base.region_size, selection.len());
        assert!(out.base.sigma < 0.03);
    }

    #[test]
    fn reads_the_residual_extremes_against_the_outward_normal() {
        // The top face, on every point: the extremes are exactly the furthest the marked
        // vertices lie below and above the reported plane — with the sign following the
        // normal that was turned to face out of the part.
        let graph = graph_soup(box_mesh(MSIZE, 20, 0.01, 13));
        let selection = mark(&graph, |p| p.z > MSIZE / 2.0 - 0.1);
        let out = fit_plane_on_selection(&graph, &selection, &FitSettings { sigma: 0, ..SETTINGS }).unwrap();
        assert!(out.normal.z > 0.999);
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for &v in &selection {
            let e = (graph.positions[v as usize] - out.center).dot(out.normal);
            lo = lo.min(e);
            hi = hi.max(e);
        }
        let close = |a: f64, b: f64| (a - b).abs() < 0.5e-6;
        let base = out.base;
        assert!(close(base.residual_min.unwrap(), lo));
        assert!(close(base.residual_max.unwrap(), hi));
        assert!(close(base.form_error.unwrap(), hi - lo));
        assert!(base.residual_min.unwrap() < 0.0);
        assert!(base.residual_max.unwrap() > 0.0);
    }

    #[test]
    fn refuses_a_marking_too_small_to_fit_anything_to() {
        let graph = graph_soup(box_mesh(MSIZE, 20, 0.01, 13));
        let selection = mark(&graph, |p| p.z > MSIZE / 2.0 - 0.1 && p.x > 9.0 && p.y > 9.0);
        assert!(!selection.is_empty());
        assert!(selection.len() < 20);
        assert!(fit_plane_on_selection(&graph, &selection, &SETTINGS).is_err());
    }
}
