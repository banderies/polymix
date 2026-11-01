// SPDX-License-Identifier: AGPL-3.0-only
//! A sloped or gently curved patch of scan measured as a surface over its own plane. Ported
//! from ScanRuler (`src/core/fit/heightSurface.ts`), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only.
//!
//! The best plane through the points sets a frame — origin at their centroid, w along its
//! normal, u and v in it — and the patch is fitted as a height over (u, v), from a plane
//! (h = a + b·u + c·v) through a bilinear patch (+ d·u·v, a hyperbolic paraboloid — a
//! twisted ruled side) to a full quadric (+ d·u² + e·u·v + f·v², a dome, a bowl, a saddle, a
//! cylinder's strip). Each is told by its RMS and worst miss, so the simplest that holds is
//! plain; the quadric also by its principal curvatures and their directions at the patch's
//! middle. Least squares on the heights, the frame's tilt taken out first — the patch's
//! slope is in the frame, not in the coefficients, so a steep side fits as well as a flat
//! one.

use std::collections::BTreeMap;

use glam::DVec3;
use serde::{Deserialize, Serialize};

use crate::linalg::solve_linear;
use crate::principal_frame::{cross, js_hypot3, ortho_basis, symmetric_eigen3};

/// The models a patch is fitted with, simplest first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SurfaceModel {
    Plane,
    Bilinear,
    Quadric,
}

/// Every model, simplest first.
pub const SURFACE_MODELS: [SurfaceModel; 3] = [SurfaceModel::Plane, SurfaceModel::Bilinear, SurfaceModel::Quadric];

impl SurfaceModel {
    /// The names of h(u, v)'s coefficients, in the order of its terms.
    pub fn names(self) -> &'static [&'static str] {
        match self {
            SurfaceModel::Plane => &["a", "b", "c"],
            SurfaceModel::Bilinear => &["a", "b", "c", "d"],
            SurfaceModel::Quadric => &["a", "b", "c", "d", "e", "f"],
        }
    }
}

/// The patch's own frame: a point of the surface is origin + u·basis_u + v·basis_v +
/// h(u, v)·normal.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SurfaceFrame {
    pub origin: DVec3,
    pub basis_u: DVec3,
    pub basis_v: DVec3,
    pub normal: DVec3,
}

/// What a quadric's principal curvatures say of its shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CurvatureShape {
    Flat,
    Cylinder,
    Dome,
    Saddle,
}

/// The quadric's principal curvatures at the patch's middle (1/mm, `k1` the larger in size),
/// their radii and directions in the world.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Curvature {
    pub k1: f64,
    pub k2: f64,
    pub r1: Option<f64>,
    pub r2: Option<f64>,
    pub axis1: DVec3,
    pub axis2: DVec3,
    pub shape: CurvatureShape,
}

/// One model fitted to a patch (ScanRuler's heightSurface `SurfaceFitResult`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HeightFit {
    pub model: SurfaceModel,
    /// h(u, v)'s coefficients, by name: a, b (u), c (v), then d (uv) for the bilinear; d (u²),
    /// e (uv), f (v²) for the quadric.
    pub coefficients: BTreeMap<String, f64>,
    /// The points' RMS distance off the surface along the normal, and the worst, mm.
    pub rms: f64,
    pub worst: f64,
    /// The quadric's principal curvatures.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub curvature: Option<Curvature>,
}

/// The nullable normalise of ScanRuler's `vec.ts`: None under 1e-12, its length
/// `Math.hypot`'s.
fn normalize(a: DVec3) -> Option<DVec3> {
    let l = js_hypot3(a);
    if l < 1e-12 {
        return None;
    }
    Some(DVec3::new(a.x / l, a.y / l, a.z / l))
}

/// The points' frame: their centroid, and the plane they scatter least off. None for too
/// few or for points in a line.
pub fn patch_frame(points: &[DVec3]) -> Option<SurfaceFrame> {
    let n = points.len();
    if n < 3 {
        return None;
    }
    let (mut mx, mut my, mut mz) = (0.0, 0.0, 0.0);
    for p in points {
        mx += p.x;
        my += p.y;
        mz += p.z;
    }
    let nf = n as f64;
    mx /= nf;
    my /= nf;
    mz /= nf;
    let (mut xx, mut xy, mut xz, mut yy, mut yz, mut zz) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for p in points {
        let (x, y, z) = (p.x - mx, p.y - my, p.z - mz);
        xx += x * x;
        xy += x * y;
        xz += x * z;
        yy += y * y;
        yz += y * z;
        zz += z * z;
    }
    let e = symmetric_eigen3(&[xx, xy, xz, xy, yy, yz, xz, yz, zz]);
    if !(e.values[1] > 1e-12 * e.values[2].max(1e-30)) {
        return None;
    }
    let normal = normalize(e.vectors[0])?;
    let (basis_u, basis_v) = ortho_basis(normal);
    Some(SurfaceFrame { origin: DVec3::new(mx, my, mz), basis_u, basis_v, normal })
}

/// The terms of a model at (u, v), in the order of its coefficients.
fn terms(model: SurfaceModel, u: f64, v: f64) -> Vec<f64> {
    match model {
        SurfaceModel::Plane => vec![1.0, u, v],
        SurfaceModel::Bilinear => vec![1.0, u, v, u * v],
        SurfaceModel::Quadric => vec![1.0, u, v, u * u, u * v, v * v],
    }
}

/// One model fitted to the points in the frame. None where the points do not pin its
/// coefficients (all on a line across the patch).
pub fn fit_height(points: &[DVec3], frame: &SurfaceFrame, model: SurfaceModel) -> Option<HeightFit> {
    let n = points.len();
    let k = model.names().len();
    if n < k + 1 {
        return None;
    }
    let mut ata = vec![0.0; k * k];
    let mut atb = vec![0.0; k];
    let mut local = Vec::with_capacity(n);
    for p in points {
        let d = *p - frame.origin;
        let u = d.dot(frame.basis_u);
        let v = d.dot(frame.basis_v);
        let h = d.dot(frame.normal);
        local.push((u, v, h));
        let t = terms(model, u, v);
        for r in 0..k {
            atb[r] += t[r] * h;
            for c in 0..k {
                ata[r * k + c] += t[r] * t[c];
            }
        }
    }
    let x = solve_linear(k, &mut ata, &atb)?;
    if !x.iter().all(|v| v.is_finite()) {
        return None;
    }
    let mut sum = 0.0;
    let mut worst = 0.0f64;
    for &(u, v, hh) in &local {
        let t = terms(model, u, v);
        let mut h = 0.0;
        for r in 0..k {
            h += t[r] * x[r];
        }
        let e = hh - h;
        sum += e * e;
        worst = worst.max(e.abs());
    }
    let coefficients = model.names().iter().zip(&x).map(|(name, v)| (name.to_string(), *v)).collect();
    let curvature = if model == SurfaceModel::Quadric { Some(curvature_at(frame, x[1], x[2], x[3], x[4], x[5])) } else { None };
    Some(HeightFit { model, coefficients, rms: (sum / n as f64).sqrt(), worst, curvature })
}

/// The principal curvatures of h = a + b·u + c·v + d·u² + e·u·v + f·v² at (0, 0): the shape
/// operator, the second fundamental form over the first, its eigenvalues and eigenvectors
/// carried into the world.
fn curvature_at(frame: &SurfaceFrame, b: f64, c: f64, d: f64, e: f64, f: f64) -> Curvature {
    let w = (1.0 + b * b + c * c).sqrt();
    // First form E F G, second L M N.
    let (fe, ff, fg) = (1.0 + b * b, b * c, 1.0 + c * c);
    let (l, m, nn) = ((2.0 * d) / w, e / w, (2.0 * f) / w);
    // S = I⁻¹ II, a 2 × 2.
    let det = fe * fg - ff * ff;
    let s11 = (fg * l - ff * m) / det;
    let s12 = (fg * m - ff * nn) / det;
    let s21 = (fe * m - ff * l) / det;
    let s22 = (fe * nn - ff * m) / det;
    let tr = s11 + s22;
    let dt = s11 * s22 - s12 * s21;
    let disc = ((tr * tr) / 4.0 - dt).max(0.0).sqrt();
    let mut ka = tr / 2.0 + disc;
    let mut kb = tr / 2.0 - disc;
    if kb.abs() > ka.abs() {
        std::mem::swap(&mut ka, &mut kb);
    }
    // An eigenvector of S for k, in (u, v), carried onto the surface: x_u = basis_u +
    // b·normal, x_v = basis_v + c·normal.
    let xu = frame.basis_u + frame.normal * b;
    let xv = frame.basis_v + frame.normal * c;
    let dir = |k: f64| -> DVec3 {
        let p: (f64, f64) = if s12.abs() > 1e-12 {
            (s12, k - s11)
        } else if s21.abs() > 1e-12 {
            (k - s22, s21)
        } else if k == ka {
            if (s11 - ka).abs() < (s22 - ka).abs() {
                (1.0, 0.0)
            } else {
                (0.0, 1.0)
            }
        } else if (s11 - kb).abs() < (s22 - kb).abs() {
            (1.0, 0.0)
        } else {
            (0.0, 1.0)
        };
        normalize(xu * p.0 + xv * p.1).unwrap_or(frame.basis_u)
    };
    let axis1 = dir(ka);
    // The second square to the first in the tangent plane, as a symmetric operator's are.
    let tangent_normal = normalize(cross(xu, xv)).unwrap_or(frame.normal);
    let axis2 = normalize(cross(tangent_normal, axis1)).unwrap_or_else(|| dir(kb));
    let tiny = 1e-5;
    let shape = if ka.abs() < tiny {
        CurvatureShape::Flat
    } else if kb.abs() < tiny {
        CurvatureShape::Cylinder
    } else if ka * kb > 0.0 {
        CurvatureShape::Dome
    } else {
        CurvatureShape::Saddle
    };
    Curvature {
        k1: ka,
        k2: kb,
        r1: if ka.abs() > tiny { Some(1.0 / ka) } else { None },
        r2: if kb.abs() > tiny { Some(1.0 / kb) } else { None },
        axis1,
        axis2,
        shape,
    }
}

/// How much worse than the closest a simpler model may fit and still be the one taken: the
/// RMS within this share of the lowest.
const SIMPLEST_SHARE: f64 = 1.1;

/// Every model asked for, fitted in one frame, and the simplest that holds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SurfaceModels {
    pub frame: SurfaceFrame,
    pub fits: Vec<HeightFit>,
    pub best: Option<SurfaceModel>,
}

/// Every model asked for fitted, and the simplest that holds — whose RMS is within a tenth
/// of the lowest. None for points that have no plane (too few, or on a line).
pub fn fit_surface_models(points: &[DVec3], models: &[SurfaceModel]) -> Option<SurfaceModels> {
    let frame = patch_frame(points)?;
    let fits: Vec<HeightFit> = models.iter().filter_map(|&m| fit_height(points, &frame, m)).collect();
    if fits.is_empty() {
        return Some(SurfaceModels { frame, fits, best: None });
    }
    let lowest = fits.iter().map(|f| f.rms).fold(f64::INFINITY, f64::min);
    let best = SURFACE_MODELS
        .iter()
        .copied()
        .filter(|m| fits.iter().any(|f| f.model == *m))
        .find(|m| fits.iter().find(|f| f.model == *m).is_some_and(|f| f.rms <= SIMPLEST_SHARE * lowest + 1e-9));
    Some(SurfaceModels { frame, fits, best })
}

// Ported from ScanRuler's tests/heightSurface.test.ts: a tilted plane, a twisted ruled face,
// a sphere's cap and a cylinder's strip, each sampled with noise — the simplest model that
// holds named, the quadric's curvatures read back, and a patch turned in space read the
// same.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit2d::Mulberry32;

    /// The tests' shared generator: one mulberry32(21) for the whole file, as the module
    /// level `rand` of the TypeScript test is, its Gaussian as `gauss` draws it.
    struct Gauss(Mulberry32);
    impl Gauss {
        fn next(&mut self) -> f64 {
            let a = self.0.next_f64().max(1e-12);
            (-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * self.0.next_f64()).cos()
        }
    }

    /// Points z = f(x, y) over a square of side 2·half, `n` a side, with noise along z.
    fn patch(g: &mut Gauss, f: impl Fn(f64, f64) -> f64, half: f64, n: usize, noise: f64) -> Vec<DVec3> {
        let mut out = Vec::new();
        for i in 0..n {
            for j in 0..n {
                let x = -half + (2.0 * half * i as f64) / (n - 1) as f64;
                let y = -half + (2.0 * half * j as f64) / (n - 1) as f64;
                out.push(DVec3::new(x, y, f(x, y) + g.next() * noise));
            }
        }
        out
    }

    /// Points turned about x by `deg` and moved.
    fn turned(points: &[DVec3], deg: f64, by: DVec3) -> Vec<DVec3> {
        let (s, c) = deg.to_radians().sin_cos();
        points.iter().map(|p| DVec3::new(p.x + by.x, c * p.y - s * p.z + by.y, s * p.y + c * p.z + by.z)).collect()
    }

    fn rms_of(r: &SurfaceModels, m: SurfaceModel) -> f64 {
        r.fits.iter().find(|f| f.model == m).map(|f| f.rms).unwrap_or(f64::NAN)
    }

    fn quadric(r: &SurfaceModels) -> Curvature {
        r.fits.iter().find(|f| f.model == SurfaceModel::Quadric).and_then(|f| f.curvature).unwrap()
    }

    // The TypeScript tests share one generator across the file, in order; so do these, run
    // as one test in the file's order.
    #[test]
    fn a_patch_fitted_as_a_surface() {
        let mut g = Gauss(Mulberry32::new(21));

        // A tilted plane is a plane, its slope in the frame.
        let r = fit_surface_models(&patch(&mut g, |x, y| 0.2 * x - 0.1 * y + 3.0, 10.0, 41, 0.01), &SURFACE_MODELS).unwrap();
        assert_eq!(r.best, Some(SurfaceModel::Plane));
        assert!(rms_of(&r, SurfaceModel::Plane) < 0.015);
        let n = r.frame.normal;
        let want = DVec3::new(-0.2, 0.1, 1.0).normalize();
        assert!(n.dot(want).abs() > 0.9999);

        // A twisted ruled face is bilinear, not a plane and not more.
        let r = fit_surface_models(&patch(&mut g, |x, y| 0.01 * x * y, 10.0, 41, 0.01), &SURFACE_MODELS).unwrap();
        assert!(rms_of(&r, SurfaceModel::Plane) > 0.2);
        assert!(rms_of(&r, SurfaceModel::Bilinear) < 0.015);
        assert_eq!(r.best, Some(SurfaceModel::Bilinear));

        // A sphere's cap is a quadric dome of its radius; a cylinder's strip a quadric
        // cylinder along its axis.
        let sphere = fit_surface_models(&patch(&mut g, |x, y| (50.0 * 50.0 - x * x - y * y).sqrt(), 10.0, 41, 0.005), &SURFACE_MODELS).unwrap();
        assert_eq!(sphere.best, Some(SurfaceModel::Quadric));
        let c = quadric(&sphere);
        assert_eq!(c.shape, CurvatureShape::Dome);
        assert!((c.r1.unwrap().abs() - 50.0).abs() < 2.0);
        assert!((c.r2.unwrap().abs() - 50.0).abs() < 2.0);
        let cyl = fit_surface_models(&patch(&mut g, |x, _| (30.0 * 30.0 - x * x).sqrt(), 8.0, 41, 0.005), &SURFACE_MODELS).unwrap();
        assert_eq!(cyl.best, Some(SurfaceModel::Quadric));
        let k = quadric(&cyl);
        assert_eq!(k.shape, CurvatureShape::Cylinder);
        assert!((k.r1.unwrap().abs() - 30.0).abs() < 1.0);
        // The round runs across x; the straight direction is the axis, y.
        assert!(k.axis2.y.abs() > 0.99);

        // Reads a patch turned in space the same.
        let flat = fit_surface_models(&patch(&mut g, |x, y| (50.0 * 50.0 - x * x - y * y).sqrt(), 10.0, 41, 0.005), &SURFACE_MODELS).unwrap();
        let tilted_points = turned(&patch(&mut g, |x, y| (50.0 * 50.0 - x * x - y * y).sqrt(), 10.0, 41, 0.005), 40.0, DVec3::new(5.0, -3.0, 12.0));
        let tilt = fit_surface_models(&tilted_points, &SURFACE_MODELS).unwrap();
        let a = quadric(&flat);
        let b = quadric(&tilt);
        assert!((a.r1.unwrap().abs() - b.r1.unwrap().abs()).abs() < 1.0);
        assert_eq!(tilt.best, Some(SurfaceModel::Quadric));

        // Points on a line have no surface.
        let line: Vec<DVec3> = (0..4).map(|i| DVec3::splat(i as f64)).collect();
        assert!(fit_surface_models(&line, &SURFACE_MODELS).is_none());
    }
}
