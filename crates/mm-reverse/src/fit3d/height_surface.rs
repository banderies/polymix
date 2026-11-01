// SPDX-License-Identifier: AGPL-3.0-only
//! A sloped or gently curved patch of scan measured as a surface over its own plane: the
//! best plane through the points sets a frame — origin at their centroid, w along its
//! normal, u and v in it — and the patch is fitted as a height over (u, v), from a plane
//! (h = a + b·u + c·v) through a bilinear patch (+ d·u·v, a hyperbolic paraboloid — a
//! twisted ruled side) to a full quadric (+ d·u² + e·u·v + f·v², a dome, a bowl, a saddle, a
//! cylinder's strip). Each is told by its RMS and worst miss, so the simplest that holds is
//! plain; the quadric also by its principal curvatures and their directions at the patch's
//! middle. Least squares on the heights, the frame's tilt taken out first — the patch's
//! slope is in the frame, not in the coefficients, so a steep side fits as well as a flat
//! one.
//!
//! The plane split (`plane::split_two_planes`) asks the quadric whether a region that splits
//! into two planes merely bows.
//!
//! Ported from ScanRuler (`src/core/fit/heightSurface.ts`), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only.

use glam::DVec3;
use serde::{Deserialize, Serialize};

use super::linalg3::{eigen_of_scatter, js_max, js_min, normalize, ortho_basis};
use crate::fit2d::solve_linear;

/// The height models, simplest first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SurfaceModel {
    Plane,
    Bilinear,
    Quadric,
}

pub const SURFACE_MODELS: [SurfaceModel; 3] = [SurfaceModel::Plane, SurfaceModel::Bilinear, SurfaceModel::Quadric];

impl SurfaceModel {
    /// The names of h(u, v)'s coefficients, in order: a, b (u), c (v), then d (uv) for the
    /// bilinear; d (u²), e (uv), f (v²) for the quadric.
    pub fn coefficient_names(self) -> &'static [&'static str] {
        match self {
            SurfaceModel::Plane => &["a", "b", "c"],
            SurfaceModel::Bilinear => &["a", "b", "c", "d"],
            SurfaceModel::Quadric => &["a", "b", "c", "d", "e", "f"],
        }
    }

    /// The terms of the model at (u, v), in the order of its coefficients.
    fn terms(self, u: f64, v: f64) -> Vec<f64> {
        match self {
            SurfaceModel::Plane => vec![1.0, u, v],
            SurfaceModel::Bilinear => vec![1.0, u, v, u * v],
            SurfaceModel::Quadric => vec![1.0, u, v, u * u, u * v, v * v],
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

/// What a quadric's curvature at the patch's middle says it is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CurvatureShape {
    Flat,
    Cylinder,
    Dome,
    Saddle,
}

/// The quadric's principal curvatures at the patch's middle (1/mm, `k1` the larger in
/// size), their radii (none where the curvature vanishes) and directions in the world.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SurfaceCurvature {
    pub k1: f64,
    pub k2: f64,
    pub r1: Option<f64>,
    pub r2: Option<f64>,
    pub axis1: DVec3,
    pub axis2: DVec3,
    pub shape: CurvatureShape,
}

/// One model fitted to a patch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SurfaceFitResult {
    pub model: SurfaceModel,
    /// h(u, v)'s coefficients in the order `model.coefficient_names()` gives.
    pub coefficients: Vec<f64>,
    /// The points' RMS distance off the surface along the normal, and the worst, mm.
    pub rms: f64,
    pub worst: f64,
    /// The quadric's principal curvatures; `None` for the other models.
    pub curvature: Option<SurfaceCurvature>,
}

impl SurfaceFitResult {
    /// A coefficient by its name (a … f).
    pub fn coefficient(&self, name: &str) -> Option<f64> {
        self.model.coefficient_names().iter().position(|n| *n == name).map(|i| self.coefficients[i])
    }
}

/// The points' frame: their centroid, and the plane they scatter least off. `None` for too
/// few or for points in a line.
pub fn patch_frame(points: &[DVec3]) -> Option<SurfaceFrame> {
    let n = points.len();
    if n < 3 {
        return None;
    }
    let nf = n as f64;
    let (mut mx, mut my, mut mz) = (0.0, 0.0, 0.0);
    for p in points {
        mx += p.x;
        my += p.y;
        mz += p.z;
    }
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
    let eig = eigen_of_scatter(xx, xy, xz, yy, yz, zz);
    if !(eig.values[1] > 1e-12 * js_max(eig.values[2], 1e-30)) {
        return None;
    }
    let normal = normalize(eig.vectors[0])?;
    let (basis_u, basis_v) = ortho_basis(normal);
    Some(SurfaceFrame { origin: DVec3::new(mx, my, mz), basis_u, basis_v, normal })
}

/// One model fitted to the points in the frame. `None` where the points do not pin its
/// coefficients (all on a line across the patch).
pub fn fit_height(points: &[DVec3], frame: &SurfaceFrame, model: SurfaceModel) -> Option<SurfaceFitResult> {
    let n = points.len();
    let k = model.coefficient_names().len();
    if n < k + 1 {
        return None;
    }
    let mut ata = vec![0.0; k * k];
    let mut atb = vec![0.0; k];
    let mut local = Vec::with_capacity(n);
    for p in points {
        let d = *p - frame.origin;
        let (u, v, h) = (d.dot(frame.basis_u), d.dot(frame.basis_v), d.dot(frame.normal));
        local.push((u, v, h));
        let t = model.terms(u, v);
        for r in 0..k {
            atb[r] += t[r] * h;
            for c in 0..k {
                ata[r * k + c] += t[r] * t[c];
            }
        }
    }
    let x = solve_linear(k, &mut ata, &atb)?;
    if !x.iter().all(|c| c.is_finite()) {
        return None;
    }
    let mut sum = 0.0;
    let mut worst = 0.0f64;
    for &(u, v, h) in &local {
        let t = model.terms(u, v);
        let mut fitted = 0.0;
        for r in 0..k {
            fitted += t[r] * x[r];
        }
        let e = h - fitted;
        sum += e * e;
        worst = js_max(worst, e.abs());
    }
    let curvature = if model == SurfaceModel::Quadric { Some(curvature_at(frame, x[1], x[2], x[3], x[4], x[5])) } else { None };
    Some(SurfaceFitResult { model, coefficients: x, rms: (sum / n as f64).sqrt(), worst, curvature })
}

/// The principal curvatures of h = a + b·u + c·v + d·u² + e·u·v + f·v² at (0, 0): the shape
/// operator, the second fundamental form over the first, its eigenvalues and eigenvectors
/// carried into the world.
fn curvature_at(frame: &SurfaceFrame, b: f64, c: f64, d: f64, e: f64, f: f64) -> SurfaceCurvature {
    let w = (1.0 + b * b + c * c).sqrt();
    // First form E F G, second L M N.
    let (ee, ff, gg) = (1.0 + b * b, b * c, 1.0 + c * c);
    let (ll, mm, nn) = ((2.0 * d) / w, e / w, (2.0 * f) / w);
    // S = I⁻¹ II, a 2 × 2.
    let det = ee * gg - ff * ff;
    let (s11, s12) = ((gg * ll - ff * mm) / det, (gg * mm - ff * nn) / det);
    let (s21, s22) = ((ee * mm - ff * ll) / det, (ee * nn - ff * mm) / det);
    let tr = s11 + s22;
    let dt = s11 * s22 - s12 * s21;
    let disc = js_max(0.0, (tr * tr) / 4.0 - dt).sqrt();
    let (mut ka, mut kb) = (tr / 2.0 + disc, tr / 2.0 - disc);
    if kb.abs() > ka.abs() {
        std::mem::swap(&mut ka, &mut kb);
    }
    let xu = frame.basis_u + frame.normal * b;
    let xv = frame.basis_v + frame.normal * c;
    // An eigenvector of S for k, in (u, v), carried onto the surface: x_u = basis_u +
    // b·normal, x_v = basis_v + c·normal.
    let dir = |k: f64| -> DVec3 {
        let p: [f64; 2] = if s12.abs() > 1e-12 {
            [s12, k - s11]
        } else if s21.abs() > 1e-12 {
            [k - s22, s21]
        } else if k == ka {
            if (s11 - ka).abs() < (s22 - ka).abs() {
                [1.0, 0.0]
            } else {
                [0.0, 1.0]
            }
        } else if (s11 - kb).abs() < (s22 - kb).abs() {
            [1.0, 0.0]
        } else {
            [0.0, 1.0]
        };
        normalize(xu * p[0] + xv * p[1]).unwrap_or(frame.basis_u)
    };
    let axis1 = dir(ka);
    // The second square to the first in the tangent plane, as a symmetric operator's are.
    let tangent_normal = normalize(xu.cross(xv)).unwrap_or(frame.normal);
    let axis2 = normalize(tangent_normal.cross(axis1)).unwrap_or_else(|| dir(kb));
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
    SurfaceCurvature {
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

/// Every model asked for, fitted, and the simplest that holds.
#[derive(Debug, Clone, PartialEq)]
pub struct SurfaceModels {
    pub frame: SurfaceFrame,
    pub fits: Vec<SurfaceFitResult>,
    pub best: Option<SurfaceModel>,
}

impl SurfaceModels {
    /// The fit of one model, when it was asked for and could be made.
    pub fn fit(&self, model: SurfaceModel) -> Option<&SurfaceFitResult> {
        self.fits.iter().find(|f| f.model == model)
    }
}

/// Every model asked for fitted, and the simplest that holds — whose RMS is within a tenth
/// of the lowest.
pub fn fit_surface_models(points: &[DVec3], models: &[SurfaceModel]) -> Option<SurfaceModels> {
    let frame = patch_frame(points)?;
    let fits: Vec<SurfaceFitResult> = models.iter().filter_map(|&m| fit_height(points, &frame, m)).collect();
    if fits.is_empty() {
        return Some(SurfaceModels { frame, fits, best: None });
    }
    let lowest = fits.iter().map(|f| f.rms).fold(f64::INFINITY, js_min);
    let best = SURFACE_MODELS
        .iter()
        .copied()
        .filter(|m| fits.iter().any(|f| f.model == *m))
        .find(|m| fits.iter().find(|f| f.model == *m).is_some_and(|f| f.rms <= SIMPLEST_SHARE * lowest + 1e-9));
    Some(SurfaceModels { frame, fits, best })
}

#[cfg(test)]
mod tests {
    // heightSurface.test.ts. Its noise comes from one generator, mulberry32(21), shared by
    // the tests in file order; each test here starts the generator where ScanRuler's run
    // has it, so every patch carries the same noise.
    use super::*;
    use crate::fit2d::Mulberry32;

    const SIDE: usize = 41;

    struct Noise(Mulberry32);

    impl Noise {
        /// The generator after `patches` patches of 41 × 41 points have drawn from it.
        fn after(patches: usize) -> Self {
            let mut rand = Mulberry32::new(21);
            for _ in 0..patches * SIDE * SIDE * 2 {
                rand.next_f64();
            }
            Noise(rand)
        }

        fn gauss(&mut self) -> f64 {
            (-2.0 * self.0.next_f64().max(1e-12).ln()).sqrt() * (2.0 * std::f64::consts::PI * self.0.next_f64()).cos()
        }

        /// Points z = f(x, y) over a square of side 2·half, `n` a side, with noise along z.
        fn patch(&mut self, f: impl Fn(f64, f64) -> f64, half: f64, n: usize, noise: f64) -> Vec<DVec3> {
            let mut out = Vec::with_capacity(n * n);
            for i in 0..n {
                for j in 0..n {
                    let x = -half + (2.0 * half * i as f64) / (n - 1) as f64;
                    let y = -half + (2.0 * half * j as f64) / (n - 1) as f64;
                    out.push(DVec3::new(x, y, f(x, y) + self.gauss() * noise));
                }
            }
            out
        }
    }

    /// Points turned about x by `deg` and moved.
    fn turned(points: &[DVec3], deg: f64, by: [f64; 3]) -> Vec<DVec3> {
        let (c, s) = (((deg * std::f64::consts::PI) / 180.0).cos(), ((deg * std::f64::consts::PI) / 180.0).sin());
        points.iter().map(|p| DVec3::new(p.x + by[0], c * p.y - s * p.z + by[1], s * p.y + c * p.z + by[2])).collect()
    }

    fn rms_of(r: &SurfaceModels, m: SurfaceModel) -> f64 {
        r.fit(m).map_or(f64::NAN, |f| f.rms)
    }

    #[test]
    fn a_tilted_plane_is_a_plane_its_slope_in_the_frame() {
        let mut noise = Noise::after(0);
        let r = fit_surface_models(&noise.patch(|x, y| 0.2 * x - 0.1 * y + 3.0, 10.0, SIDE, 0.01), &SURFACE_MODELS).unwrap();
        assert_eq!(r.best, Some(SurfaceModel::Plane));
        assert!(rms_of(&r, SurfaceModel::Plane) < 0.015);
        let n = r.frame.normal;
        let l = (0.2f64 * 0.2 + 0.1 * 0.1 + 1.0).sqrt();
        let want = DVec3::new(-0.2 / l, 0.1 / l, 1.0 / l);
        assert!(n.dot(want).abs() > 0.9999);
    }

    #[test]
    fn a_twisted_ruled_face_is_bilinear_not_a_plane_and_not_more() {
        let mut noise = Noise::after(1);
        let r = fit_surface_models(&noise.patch(|x, y| 0.01 * x * y, 10.0, SIDE, 0.01), &SURFACE_MODELS).unwrap();
        assert!(rms_of(&r, SurfaceModel::Plane) > 0.2);
        assert!(rms_of(&r, SurfaceModel::Bilinear) < 0.015);
        assert_eq!(r.best, Some(SurfaceModel::Bilinear));
    }

    #[test]
    fn a_spheres_cap_is_a_dome_and_a_cylinders_strip_a_cylinder_along_its_axis() {
        let mut noise = Noise::after(2);
        let sphere = fit_surface_models(&noise.patch(|x, y| (50.0f64 * 50.0 - x * x - y * y).sqrt(), 10.0, SIDE, 0.005), &SURFACE_MODELS).unwrap();
        assert_eq!(sphere.best, Some(SurfaceModel::Quadric));
        let c = sphere.fit(SurfaceModel::Quadric).and_then(|f| f.curvature).unwrap();
        assert_eq!(c.shape, CurvatureShape::Dome);
        assert!((c.r1.unwrap().abs() - 50.0).abs() < 2.0);
        assert!((c.r2.unwrap().abs() - 50.0).abs() < 2.0);
        let cyl = fit_surface_models(&noise.patch(|x, _| (30.0f64 * 30.0 - x * x).sqrt(), 8.0, SIDE, 0.005), &SURFACE_MODELS).unwrap();
        assert_eq!(cyl.best, Some(SurfaceModel::Quadric));
        let k = cyl.fit(SurfaceModel::Quadric).and_then(|f| f.curvature).unwrap();
        assert_eq!(k.shape, CurvatureShape::Cylinder);
        assert!((k.r1.unwrap().abs() - 30.0).abs() < 1.0);
        // The round runs across x; the straight direction is the axis, y.
        assert!(k.axis2.y.abs() > 0.99);
    }

    #[test]
    fn reads_a_patch_turned_in_space_the_same() {
        let mut noise = Noise::after(4);
        let dome = |x: f64, y: f64| (50.0f64 * 50.0 - x * x - y * y).sqrt();
        let flat = fit_surface_models(&noise.patch(dome, 10.0, SIDE, 0.005), &SURFACE_MODELS).unwrap();
        let tilted = turned(&noise.patch(dome, 10.0, SIDE, 0.005), 40.0, [5.0, -3.0, 12.0]);
        let tilt = fit_surface_models(&tilted, &SURFACE_MODELS).unwrap();
        let a = flat.fit(SurfaceModel::Quadric).and_then(|f| f.curvature).unwrap();
        let b = tilt.fit(SurfaceModel::Quadric).and_then(|f| f.curvature).unwrap();
        assert!((a.r1.unwrap().abs() - b.r1.unwrap().abs()).abs() < 1.0);
        assert_eq!(tilt.best, Some(SurfaceModel::Quadric));
    }

    #[test]
    fn points_on_a_line_have_no_surface() {
        let line: Vec<DVec3> = (0..4).map(|i| DVec3::splat(i as f64)).collect();
        assert!(fit_surface_models(&line, &SURFACE_MODELS).is_none());
    }
}
