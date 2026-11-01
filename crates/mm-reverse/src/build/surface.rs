// SPDX-License-Identifier: AGPL-3.0-only
//! The Fit surface feature's build: a surface fitted through a region of the scan — a body of
//! its own, marked as a surface. Freeform, a degree-3 B-spline net (Auto's size from the
//! region's extent and noise, or the one typed), carried past the region by the share asked
//! (Expand by) and held fair (Smoothness), robust where asked; or a plane or a cylinder held to
//! what the part already has (Auto Constrain) and drawn as an exact face round the points. Its
//! report says σ, the worst points, the net, and how far the margin stands off the scan past
//! the region. Ported from ScanRuler (`plugins/reverse/core/build.ts` surfaceOf, faceThrough,
//! pastRegionOf, primitiveSurface, nearScan; `resolve.ts` the surface's checks;
//! `features.ts` constrainRefs; `buildSteps.ts` SURFACE_MIN_REGION; `buildTypes.ts`
//! SURFACE_TRIANGLES), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only, on the SolidCraft kernel.
//! The freeform face is the fit's own control net (see `surf_kernel::fit_face`), not a surface
//! through a grid sampled off it.

use std::sync::Arc;

use glam::DVec3;
use solidcraft_geom::{Plane, Seg2, Vec2 as SVec2};
use solidcraft_kernel::{self as kernel, Body};

use super::surf::{BuildScan, SurfaceFitReport};
use super::surf_kernel::{fit_face, surface_area, surface_through};
use super::{kerr, mesh_deflection, polygon_region, sv3, BodyState, Builder};
use crate::auto_constrain::{constrain_cylinder, constrain_plane, cylinder_patch, plane_patch, ConstrainRef, RefKind};
use crate::model::{BodyId, ConstrainMode, ElementGeometry, Feature, FeatureKind, Part, RegionStale, SurfaceFeature, SurfaceShape, TrimPrimitive};
use crate::surface_fit::{fit_surface_reported, grid_area, grid_of, js_num, mesh_triangles, Net, RegionFacing, SurfaceFitParams, SurfaceFitResult, GRID_AREA_SLACK, GRID_PER_SPAN, GRID_RETRIES, MESH_ANGLE};
use crate::surface_region::{past_region, PAST_REGION_MM};

/// The fewest points a surface is fitted through.
pub const SURFACE_MIN_REGION: usize = 16;

/// The most triangles a fitted freeform surface may mesh to at the build's chord, the finest
/// it is meshed to — read off the fit before the kernel meshes it.
pub const SURFACE_TRIANGLES: f64 = 300_000.0;

pub(crate) fn build(b: &mut Builder, part: &Part, f: &Feature, x: &SurfaceFeature) -> Result<(), String> {
    if x.region.len() < SURFACE_MIN_REGION {
        return Err(format!("The region holds {} points; a surface needs at least {SURFACE_MIN_REGION}.", x.region.len()));
    }
    // Its box moved with the parameters and its points were not read in it again: fitted, the
    // surface would be the old box's.
    match &x.region_stale {
        Some(RegionStale::Yes(true)) => return Err("The box its region is read in moved with the parameters, and the points in it were not read again yet — the box reads them over the scan.".into()),
        Some(RegionStale::Why(why)) => return Err(format!("The box its region is read in moved with the parameters, and the points in it were not read again: {why} — the box reads them over the scan.")),
        _ => {}
    }
    let scan = b.scan_for("fit a surface through")?;
    let positions = scan.positions();
    let mut xyz = Vec::with_capacity(x.region.len());
    for v in &x.region {
        match positions.get(*v as usize) {
            Some(p) => xyz.push(*p),
            None => return Err("The region names a vertex the scan does not have — it was marked on another scan.".into()),
        }
    }
    let id = BodyId::new(b.step);
    let (face, report, primitive, fit) = match x.shape.unwrap_or(SurfaceShape::Freeform) {
        SurfaceShape::Freeform => {
            let params = fit_params(x);
            let facing = x.region_box.as_ref().and_then(|r| r.normal).map(|n| RegionFacing { dir: n.dir, within_deg: n.within_deg });
            let got = match fit_surface_reported(&xyz, &params, facing.as_ref()) {
                Ok(r) => r,
                Err(e) => {
                    // A fit refused for points that are not its surface's names them.
                    if let Some(strays) = e.strays() {
                        b.report().strays = strays.to_vec();
                    }
                    return Err(e.to_string());
                }
            };
            let face = face_through(&got.fit)?;
            let past = past_region_of(b, &scan, &got.fit);
            let report = SurfaceFitReport {
                rms: got.fit.rms,
                max: got.fit.max,
                net: (got.fit.net.u, got.fit.net.v),
                points: x.region.len(),
                worst: got.worst.clone(),
                dropped: params.robust.then_some(got.dropped),
                strays: got.strays.clone().unwrap_or_default(),
                past_region: past,
                ..Default::default()
            };
            (face, report, None, Some(got.fit))
        }
        shape => {
            let (face, report, primitive) = primitive_surface(b, part, f, x, shape, &xyz)?;
            (face, report, Some(primitive), None)
        }
    };
    let own = b.own_origins(&face);
    let mut state = BodyState::new(face, own);
    state.surface = true;
    let at = b.fresh_id()?;
    b.set_body(at, state);
    if let Some(p) = primitive {
        b.surf.fitted.insert(at, p);
    }
    if let Some(fit) = fit {
        b.surf.fits.insert(at, Arc::new(fit));
    }
    debug_assert_eq!(at, id);
    b.touched(vec![at]);
    b.report().surface = Some(report);
    Ok(())
}

/// The fit's parameters as the fit takes them.
pub fn fit_params(x: &SurfaceFeature) -> SurfaceFitParams {
    SurfaceFitParams {
        net: x.fit.net.map(|n| Net { u: n.u as usize, v: n.v as usize }),
        degree: x.fit.degree,
        expand_pct: x.fit.expand_pct,
        smoothness: x.fit.smoothness,
        robust: x.fit.robust.unwrap_or(false),
        normal: x.fit.normal,
    }
}

/// The kernel's face through a fitted surface, checked before anything meshes it (ScanRuler's
/// `faceThrough`): that it follows the fit — its area within GRID_AREA_SLACK of the grid's, a
/// grid tried at the denser spacings when not — and that it meshes to no more than
/// SURFACE_TRIANGLES at the finest chord it is meshed to.
pub(crate) fn face_through(fit: &SurfaceFitResult) -> Result<Body, String> {
    let (mut area, mut off) = (0.0, 0.0);
    for (k, per_span) in std::iter::once(GRID_PER_SPAN).chain(GRID_RETRIES).enumerate() {
        let grid = grid_of(fit, per_span);
        // The fit's own net first; a surface through the grid after.
        let face = if k == 0 { fit_face(fit)? } else { surface_through(&grid)? };
        area = grid_area(&grid);
        off = surface_area(&face) / area;
        if !(off > GRID_AREA_SLACK) {
            let (mut lo, mut hi) = (DVec3::splat(f64::INFINITY), DVec3::splat(f64::NEG_INFINITY));
            for p in &grid.points {
                lo = lo.min(*p);
                hi = hi.max(*p);
            }
            let chord = mesh_deflection((hi - lo).length());
            let triangles = mesh_triangles(fit, chord, MESH_ANGLE);
            if triangles > SURFACE_TRIANGLES {
                return Err(format!(
                    "The surface would mesh to about {} k triangles at its {} mm chord, more than {} k — it bends too often for its size. Fit a coarser net, or smooth it more.",
                    (triangles / 1000.0).round(),
                    js_num(chord),
                    js_num(SURFACE_TRIANGLES / 1000.0)
                ));
            }
            return Ok(face);
        }
    }
    Err(format!(
        "The kernel's surface through the fit does not follow it: {} the fit's {} mm² of area, at every grid it was given. Fit another net, or keep the region to one surface.",
        if off > 100.0 { format!("{} times", off.round()) } else { format!("{} % more than", ((off - 1.0) * 100.0).round()) },
        area.round()
    ))
}

/// How far a fitted surface stands off the scan past its region, and where (ScanRuler's
/// `pastRegionOf`) — said on the row past PAST_REGION_MM: a split by it carries its margin
/// into the part.
fn past_region_of(b: &mut Builder, scan: &BuildScan, fit: &SurfaceFitResult) -> Option<(f64, DVec3)> {
    let extent = (fit.u_range[1] - fit.u_range[0]).max(fit.v_range[1] - fit.v_range[0]);
    let reach = (0.25 * extent).max(1.0);
    let past = past_region(fit, |p| scan.closest(p, reach))?;
    let off = (past.off * 1000.0).round() / 1000.0;
    let at = (past.at * 100.0).round() / 100.0;
    if past.off > PAST_REGION_MM {
        b.warn(format!(
            "Past its region the surface stands up to {:.2} mm off the scan, at ({}, {}, {}) — its margin follows the fit, not the scan, and a split by it carries that into the part. Take less margin (Expand by), mark the region out to where the surface has to go, or split within the region (split within).",
            past.off,
            js_num(at.x),
            js_num(at.y),
            js_num(at.z)
        ));
    }
    Some((off, at))
}

/// What a plane or a cylinder fitted in the workspace can be held to, as far as the part
/// knows it (ScanRuler's `constrainRefs`): the part's coordinate planes, and the measured
/// planes, cylinders and cones. The surfaces fitted before one are the build's to add.
pub fn constrain_refs(part: &Part) -> Vec<ConstrainRef> {
    let mut refs: Vec<ConstrainRef> = [("datum:xy", "XY plane", DVec3::Z), ("datum:yz", "YZ plane", DVec3::X), ("datum:zx", "XZ plane", DVec3::Y)]
        .into_iter()
        .map(|(key, name, normal)| ConstrainRef { key: key.into(), kind: RefKind::Plane, name: name.into(), origin: DVec3::ZERO, dir: normal })
        .collect();
    for e in &part.elements {
        match &e.geometry {
            ElementGeometry::Plane { center, normal, .. } => refs.push(ConstrainRef { key: format!("element:{}", e.id), kind: RefKind::Plane, name: e.name.clone(), origin: *center, dir: *normal }),
            ElementGeometry::Cylinder { center, axis, .. } | ElementGeometry::Cone { center, axis, .. } => {
                refs.push(ConstrainRef { key: format!("element:{}", e.id), kind: RefKind::Axis, name: e.name.clone(), origin: *center, dir: *axis })
            }
            _ => {}
        }
    }
    refs
}

/// A plane or a cylinder through the region, held to the references — the part's own and the
/// planes and cylinders the surfaces before it came out as — and drawn as an exact face round
/// where the points are (ScanRuler's `primitiveSurface`). What was fitted is handed back beside
/// the face: a later fit is held to it, and a trim cuts with the whole of it.
fn primitive_surface(b: &Builder, part: &Part, f: &Feature, x: &SurfaceFeature, shape: SurfaceShape, xyz: &[DVec3]) -> Result<(Body, SurfaceFitReport, TrimPrimitive), String> {
    let mut refs = constrain_refs(part);
    let at = part.index_of(f.id).unwrap_or(part.features.len());
    for s in &part.features[..at.min(part.features.len())] {
        let FeatureKind::Surface(sf) = &s.kind else { continue };
        if sf.shape.unwrap_or(SurfaceShape::Freeform) == SurfaceShape::Freeform {
            continue;
        }
        match b.surf.fitted.get(&BodyId::new(s.id)) {
            Some(TrimPrimitive::Plane { origin, normal, .. }) => refs.push(ConstrainRef { key: format!("surface:{}", s.id), kind: RefKind::Plane, name: s.name.clone(), origin: *origin, dir: *normal }),
            Some(TrimPrimitive::Cylinder { origin, axis, .. }) => refs.push(ConstrainRef { key: format!("surface:{}", s.id), kind: RefKind::Axis, name: s.name.clone(), origin: *origin, dir: *axis }),
            _ => {}
        }
    }
    let mode = x.constrain.clone().unwrap_or(ConstrainMode::Auto);
    let points = xyz.len();
    if shape == SurfaceShape::Plane {
        let fit = constrain_plane(xyz, &refs, &mode).ok_or("The region has no plane through it — its points lie in a line.")?;
        let patch = plane_patch(xyz, fit.origin, fit.normal, x.fit.expand_pct);
        let plane = Plane::new(sv3(patch.corner), sv3(patch.u), sv3(patch.v)).ok_or("The fitted plane has no direction.")?;
        let region = polygon_region(&[glam::DVec2::new(0.0, 0.0), glam::DVec2::new(patch.width, 0.0), glam::DVec2::new(patch.width, patch.height), glam::DVec2::new(0.0, patch.height)]);
        let face = kernel::patch_region(&plane, &region).map_err(kerr)?;
        let report = SurfaceFitReport { rms: fit.rms, max: fit.max, net: (0, 0), points, shape: Some(SurfaceShape::Plane), rms_free: Some(fit.rms_free), relations: fit.relations.clone(), ..Default::default() };
        return Ok((face, report, TrimPrimitive::Plane { origin: fit.origin, normal: fit.normal, basis_u: patch.u, basis_v: patch.v }));
    }
    let fit = constrain_cylinder(xyz, &refs, &mode).ok_or("No cylinder was found through the region — mark more of the round, or fit it as a freeform surface.")?;
    let patch = cylinder_patch(xyz, fit.origin, fit.axis, x.fit.expand_pct);
    // The plane square to the axis through its origin, its u and v the patch's (u × v = axis).
    let plane = Plane::new(sv3(fit.origin), sv3(patch.u), sv3(patch.v)).ok_or("The fitted cylinder has no axis.")?;
    let chain: Vec<Seg2> = if patch.full {
        solidcraft_geom::Loop2::circle(SVec2::new(0.0, 0.0), fit.radius).segs
    } else {
        vec![Seg2::Arc { center: SVec2::new(0.0, 0.0), radius: fit.radius, start: patch.start, sweep: patch.sweep }]
    };
    let flip = plane.normal().dot(sv3(fit.axis)) < 0.0;
    let (from, to) = if flip { (-patch.to, -patch.from) } else { (patch.from, patch.to) };
    let face = kernel::extrude_sheet(&plane, &[chain], from, to).map_err(kerr)?;
    let report = SurfaceFitReport { rms: fit.rms, max: fit.max, net: (0, 0), points, shape: Some(SurfaceShape::Cylinder), radius: Some(fit.radius), rms_free: Some(fit.rms_free), relations: fit.relations.clone(), ..Default::default() };
    Ok((face, report, TrimPrimitive::Cylinder { origin: fit.origin, axis: fit.axis, radius: fit.radius, length: patch.to - patch.from }))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::build::{build_scanned, BuildScan};
    use crate::model::{NetSize, SurfaceFitParams as ModelFit};
    use std::sync::Arc;

    /// A scan of z = height(x, y) over x, y in ±half, as a grid of triangles (reThicken's
    /// `patchScan`), and its every vertex as the region.
    pub(crate) fn patch_scan(half: f64, height: impl Fn(f64, f64) -> f64) -> (Vec<DVec3>, Vec<[u32; 3]>, Vec<u32>) {
        let n = 25;
        let mut positions = Vec::new();
        let mut tris = Vec::new();
        for i in 0..n {
            for j in 0..n {
                let x = -half + (2.0 * half * i as f64) / (n - 1) as f64;
                let y = -half + (2.0 * half * j as f64) / (n - 1) as f64;
                positions.push(DVec3::new(x as f32 as f64, y as f32 as f64, height(x, y) as f32 as f64));
            }
        }
        for i in 0..n - 1 {
            for j in 0..n - 1 {
                let a = (i * n + j) as u32;
                let nn = n as u32;
                tris.push([a, a + nn, a + nn + 1]);
                tris.push([a, a + nn + 1, a + 1]);
            }
        }
        let region = (0..(n * n) as u32).collect();
        (positions, tris, region)
    }

    pub(crate) fn surface(region: Vec<u32>, net: u32, expand: f64) -> FeatureKind {
        FeatureKind::Surface(SurfaceFeature {
            region,
            fit: ModelFit { net: Some(NetSize { u: net, v: net }), degree: 3, expand_pct: expand, smoothness: 0.01, robust: None, normal: None },
            shape: None,
            constrain: None,
            region_box: None,
            region_stale: None,
            exprs: Default::default(),
        })
    }

    /// reThicken's flange: a saddle-ish freeform patch 30 mm across.
    pub(crate) fn flange(x: f64, y: f64) -> f64 {
        3.0 + 0.03 * x * x + 0.01 * x * y - 0.008 * y * y
    }

    #[test]
    fn fits_a_freeform_skin_through_a_region_and_says_how_well() {
        let (pos, tris, region) = patch_scan(15.0, flange);
        let scan = Arc::new(BuildScan::new(pos, tris));
        let mut part = Part::default();
        let id = part.add(surface(region, 5, 5.0));
        let built = build_scanned(&part, Some(scan));
        assert!(built.errors.is_empty(), "{:?}", built.errors);
        let b = &built.bodies[0];
        assert!(b.surface);
        assert_eq!(b.body.face_count(), 1);
        let r = built.surf.get(&id).and_then(|r| r.surface.clone()).unwrap();
        assert_eq!(r.net, (5, 5));
        assert_eq!(r.points, 625);
        assert!(r.rms < 0.05, "{}", r.rms);
        assert!(r.shape.is_none());
        // With no scan it fails, saying why.
        let built = crate::build::build(&part);
        assert_eq!(built.errors.get(&id).map(String::as_str), Some("The scan is not here to fit a surface through."));
    }

    #[test]
    fn fits_a_plane_held_to_the_xy_plane_and_a_cylinder() {
        // A plane tilted 0.2° off flat, and a half cylinder of R 10 about X.
        let (pos, tris, region) = patch_scan(15.0, |x, _| 5.0 + x * 0.2f64.to_radians().tan());
        let scan = Arc::new(BuildScan::new(pos, tris));
        let mut part = Part::default();
        let mut k = surface(region, 5, 10.0);
        if let FeatureKind::Surface(s) = &mut k {
            s.shape = Some(SurfaceShape::Plane);
        }
        let id = part.add(k);
        let built = build_scanned(&part, Some(scan.clone()));
        assert!(built.errors.is_empty(), "{:?}", built.errors);
        let r = built.surf.get(&id).and_then(|r| r.surface.clone()).unwrap();
        assert_eq!(r.shape, Some(SurfaceShape::Plane));
        assert_eq!(r.relations.len(), 1, "{:?}", r.relations);
        assert_eq!(r.relations[0].reference, "datum:xy");
        assert!(r.rms_free.unwrap() < r.rms);
        // Held to nothing, it is the free fit.
        if let Some(FeatureKind::Surface(s)) = part.feature_mut(id).map(|f| &mut f.kind) {
            s.constrain = Some(ConstrainMode::Free);
        }
        let built = build_scanned(&part, Some(scan));
        let r = built.surf.get(&id).and_then(|r| r.surface.clone()).unwrap();
        assert!(r.relations.is_empty());

        let mut pos = Vec::new();
        let mut tris = Vec::new();
        let (na, nx) = (24u32, 11u32);
        for i in 0..=na {
            let a = std::f64::consts::PI * i as f64 / na as f64;
            for j in 0..nx {
                pos.push(DVec3::new(-10.0 + 2.0 * j as f64, 10.0 * a.cos(), 10.0 * a.sin()));
            }
        }
        for i in 0..na {
            for j in 0..nx - 1 {
                let a = i * nx + j;
                tris.push([a, a + nx, a + nx + 1]);
                tris.push([a, a + nx + 1, a + 1]);
            }
        }
        let region: Vec<u32> = (0..pos.len() as u32).collect();
        let scan = Arc::new(BuildScan::new(pos, tris));
        let mut part = Part::default();
        let mut k = surface(region, 5, 10.0);
        if let FeatureKind::Surface(s) = &mut k {
            s.shape = Some(SurfaceShape::Cylinder);
        }
        let id = part.add(k);
        let built = build_scanned(&part, Some(scan));
        assert!(built.errors.is_empty(), "{:?}", built.errors);
        let r = built.surf.get(&id).and_then(|r| r.surface.clone()).unwrap();
        assert!((r.radius.unwrap() - 10.0).abs() < 1e-3, "{:?}", r.radius);
        assert!(built.bodies[0].surface);
    }

    #[test]
    fn refuses_a_region_too_small_or_named_on_another_scan() {
        let (pos, tris, _) = patch_scan(15.0, flange);
        let scan = Arc::new(BuildScan::new(pos, tris));
        let mut part = Part::default();
        let a = part.add(surface((0..10).collect(), 5, 5.0));
        let b = part.add(surface((620..640).collect(), 5, 5.0));
        let built = build_scanned(&part, Some(scan));
        assert_eq!(built.errors.get(&a).map(String::as_str), Some("The region holds 10 points; a surface needs at least 16."));
        assert_eq!(built.errors.get(&b).map(String::as_str), Some("The region names a vertex the scan does not have — it was marked on another scan."));
    }

    // reSurfaceRunaway.test.ts: the kernel's side of a fitted surface.
    mod runaway {
        use super::*;
        use crate::surface_fit::{fit_surface, Net, PULL};

        /// Points of z = f(x, y) over w × h at a step, with a little noise.
        fn patch(f: impl Fn(f64, f64) -> f64, w: f64, h: f64, step: f64) -> Vec<DVec3> {
            let mut pts = Vec::new();
            let mut x = -w / 2.0;
            while x <= w / 2.0 + 1e-9 {
                let mut y = -h / 2.0;
                while y <= h / 2.0 + 1e-9 {
                    pts.push(DVec3::new(x, y, f(x, y) + 0.01 * (7.0 * x + 3.0 * y).sin()));
                    y += step;
                }
                x += step;
            }
            pts
        }

        /// A surface feature fitted through every point, the points a scan of their own.
        fn fitted(xyz: Vec<DVec3>, net: u32, smoothness: f64, expand: f64) -> (crate::build::Built, u32) {
            let region: Vec<u32> = (0..xyz.len() as u32).collect();
            let scan = Arc::new(BuildScan::new(xyz, Vec::new()));
            let mut part = Part::default();
            let mut k = surface(region, net, expand);
            if let FeatureKind::Surface(s) = &mut k {
                s.fit.smoothness = smoothness;
            }
            let id = part.add(k);
            (build_scanned(&part, Some(scan)), id)
        }

        // ScanRuler holds the count within a factor of two of Open CASCADE's mesher; SolidCraft's
        // lays a third or so of what that one does at a chord, so here the count is held to an
        // upper bound of what the kernel makes — the budget it guards refuses no surface the
        // kernel would mesh within it — and to within ten times it.
        #[test]
        fn counts_the_triangles_the_mesher_makes_within_a_factor_of_two() {
            let cases: [(Vec<DVec3>, usize); 3] = [
                (patch(|x, _| (900.0 - x * x).sqrt() - 30.0, 40.0, 20.0, 0.5), 8),
                (patch(|x, y| (x * x - y * y) / 80.0, 40.0, 40.0, 1.0), 10),
                (patch(|x, y| (x * x + y * y) / 400.0, 200.0, 200.0, 4.0), 12),
            ];
            for (xyz, n) in cases {
                let params = crate::surface_fit::SurfaceFitParams { net: Some(Net { u: n, v: n }), smoothness: 0.001, ..Default::default() };
                let fit = fit_surface(&xyz, &params, Some(0.01), PULL).unwrap();
                let face = fit_face(&fit).unwrap();
                for chord in [0.05, 0.02] {
                    let made = face.display_mesh(chord).unwrap().triangles.len() as f64;
                    let said = mesh_triangles(&fit, chord, MESH_ANGLE);
                    assert!(made / said > 0.1 && made / said < 2.0, "net {n} chord {chord}: {made} against {said}");
                }
            }
        }

        #[test]
        fn refuses_a_surface_that_would_mesh_to_too_many_triangles_before_it_is_meshed() {
            // An egg crate a metre across, 600 mm deep, the net following it.
            let xyz = patch(|x, y| 300.0 * (x / 51.0).sin() * (y / 51.0).sin(), 1000.0, 1000.0, 10.0);
            let t = std::time::Instant::now();
            let (built, id) = fitted(xyz, 20, 0.0, 0.0);
            let e = built.errors.get(&id).cloned().unwrap_or_default();
            assert!(e.starts_with("The surface would mesh to about ") && e.contains(" k triangles at its 0.05 mm chord"), "{e}");
            // Refused before anything meshes it: in seconds (ScanRuler's ten, here with room
            // for a machine under load).
            assert!(t.elapsed().as_secs_f64() < 30.0, "{:?}", t.elapsed());
        }

        #[test]
        fn draws_a_fitted_surface_to_a_chord_after_its_size() {
            let (built, id) = fitted(patch(|x, y| (x * x + y * y) / 2000.0, 300.0, 300.0, 6.0), 8, 0.01, 35.0);
            assert!(built.errors.is_empty(), "{:?}", built.errors);
            let b = built.bodies.iter().find(|b| b.id == BodyId::new(id)).unwrap();
            // About 400 mm across with its margin: drawn to 0.16 mm, not 0.05.
            let m = crate::build::body_mesh(&b.body).unwrap();
            assert!(m.triangles.len() < 3000, "{}", m.triangles.len());
        }
    }
}
