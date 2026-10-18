//! Resampling: values between voxels (nearest, trilinear, Catmull-Rom cubic, Lanczos-3
//! windowed sinc), a new voxel size over the same extent, and one image onto another's grid
//! through an affine map between their worlds (as after registration).
//!
//! Points within half a voxel of the grid count as inside (the edge values extend that far);
//! further out a sample takes the `outside` value.

use crate::image::{check, on_grid, Progress};
use anyhow::{bail, Result};
use glam::{DMat4, DVec3, DVec4};
use mm_volume::geometry::Geometry;
use mm_volume::load::LoadedVolume;
use rayon::prelude::*;

/// Largest image made by resampling.
const MAX_VOXELS: usize = 600_000_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interpolation {
    /// The nearest voxel (labels, masks).
    Nearest,
    /// Trilinear.
    Linear,
    /// Catmull-Rom cubic (interpolating, so no prefilter; sharper than linear, may overshoot a
    /// little at edges).
    Cubic,
    /// Lanczos-3 windowed sinc (sharpest; slowest).
    Sinc,
}

impl Interpolation {
    pub const ALL: [Interpolation; 4] = [Interpolation::Nearest, Interpolation::Linear, Interpolation::Cubic, Interpolation::Sinc];

    pub fn label(self) -> &'static str {
        match self {
            Interpolation::Nearest => "Nearest",
            Interpolation::Linear => "Linear",
            Interpolation::Cubic => "Cubic",
            Interpolation::Sinc => "Windowed sinc",
        }
    }

    /// Kernel half-width in voxels.
    fn radius(self) -> i64 {
        match self {
            Interpolation::Nearest => 0,
            Interpolation::Linear => 1,
            Interpolation::Cubic => 2,
            Interpolation::Sinc => 3,
        }
    }

    fn weight(self, x: f64) -> f64 {
        let a = x.abs();
        match self {
            Interpolation::Nearest => (a < 0.5) as i32 as f64,
            Interpolation::Linear => (1.0 - a).max(0.0),
            Interpolation::Cubic => {
                // Keys' cubic with a = -0.5 (Catmull-Rom).
                if a < 1.0 {
                    1.5 * a * a * a - 2.5 * a * a + 1.0
                } else if a < 2.0 {
                    -0.5 * a * a * a + 2.5 * a * a - 4.0 * a + 2.0
                } else {
                    0.0
                }
            }
            Interpolation::Sinc => {
                if a < 1e-9 {
                    1.0
                } else if a < 3.0 {
                    let px = std::f64::consts::PI * a;
                    3.0 * px.sin() * (px / 3.0).sin() / (px * px)
                } else {
                    0.0
                }
            }
        }
    }
}

/// An affine map between RAS-millimetre worlds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform {
    pub matrix: DMat4,
}

impl Transform {
    pub fn identity() -> Self {
        Self { matrix: DMat4::IDENTITY }
    }

    pub fn translation(t: DVec3) -> Self {
        Self { matrix: DMat4::from_translation(t) }
    }

    pub fn inverse(&self) -> Self {
        Self { matrix: self.matrix.inverse() }
    }

    /// `self` after `first`: a point goes through `first`, then `self`.
    pub fn after(&self, first: &Transform) -> Self {
        Self { matrix: self.matrix * first.matrix }
    }

    pub fn apply(&self, p: DVec3) -> DVec3 {
        self.matrix.transform_point3(p)
    }
}

/// Up to six kernel taps along one axis: first index and weights (normalised to sum 1).
fn taps(x: f64, n: usize, interp: Interpolation) -> (i64, [f64; 6], usize) {
    let mut w = [0.0; 6];
    if interp == Interpolation::Nearest {
        let i = (x.round() as i64).clamp(0, n as i64 - 1);
        w[0] = 1.0;
        return (i, w, 1);
    }
    let r = interp.radius();
    let base = x.floor() as i64 - r + 1;
    let count = (2 * r) as usize;
    let mut sum = 0.0;
    for (k, wk) in w.iter_mut().enumerate().take(count) {
        *wk = interp.weight(x - (base + k as i64) as f64);
        sum += *wk;
    }
    if sum.abs() > 1e-12 {
        for wk in w.iter_mut().take(count) {
            *wk /= sum;
        }
    }
    (base, w, count)
}

/// Value of `vol` at continuous voxel coordinates `ijk` (outside → `outside`).
pub fn sample(vol: &LoadedVolume, ijk: DVec3, interp: Interpolation, outside: f32) -> f32 {
    let d = vol.dims;
    if (0..3).any(|a| !(ijk[a] >= -0.5 && ijk[a] <= d[a] as f64 - 0.5)) {
        return outside;
    }
    let (bx, wx, cx) = taps(ijk.x, d[0], interp);
    let (by, wy, cy) = taps(ijk.y, d[1], interp);
    let (bz, wz, cz) = taps(ijk.z, d[2], interp);
    let clamp = |i: i64, n: usize| i.clamp(0, n as i64 - 1) as usize;
    let mut acc = 0.0f64;
    for (c, &wkz) in wz.iter().enumerate().take(cz) {
        if wkz == 0.0 {
            continue;
        }
        let k = clamp(bz + c as i64, d[2]);
        for (b, &wky) in wy.iter().enumerate().take(cy) {
            if wky == 0.0 {
                continue;
            }
            let row = (k * d[1] + clamp(by + b as i64, d[1])) * d[0];
            let mut line = 0.0;
            for (a, &wkx) in wx.iter().enumerate().take(cx) {
                line += wkx * vol.values[row + clamp(bx + a as i64, d[0])] as f64;
            }
            acc += wkz * wky * line;
        }
    }
    acc as f32
}

/// Fill `dims` voxels of `grid` from `src`, mapping each output voxel to `src` voxel
/// coordinates with `to_src` (output voxel → source voxel).
fn fill(src: &LoadedVolume, dims: [usize; 3], to_src: DMat4, interp: Interpolation, outside: f32, progress: Progress) -> Result<Vec<f32>> {
    let n = dims[0] * dims[1] * dims[2];
    if n > MAX_VOXELS {
        bail!("the result would have {} million voxels (the limit is {} million)", n / 1_000_000, MAX_VOXELS / 1_000_000);
    }
    let plane = dims[0] * dims[1];
    let mut out = vec![0.0f32; n];
    let done = std::sync::atomic::AtomicUsize::new(0);
    let stop = std::sync::atomic::AtomicBool::new(false);
    out.par_chunks_mut(plane.max(1)).enumerate().for_each(|(k, p)| {
        if stop.load(std::sync::atomic::Ordering::Relaxed) {
            return;
        }
        for j in 0..dims[1] {
            for i in 0..dims[0] {
                let v = to_src * DVec4::new(i as f64, j as f64, k as f64, 1.0);
                p[j * dims[0] + i] = sample(src, v.truncate(), interp, outside);
            }
        }
        let d = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if !progress(d as f32 / dims[2] as f32) {
            stop.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    });
    if stop.load(std::sync::atomic::Ordering::Relaxed) {
        bail!("cancelled");
    }
    Ok(out)
}

/// Resample to a new voxel size over the same extent and orientation (dims = extent /
/// spacing, rounded; the new grid is centred on the old one). When shrinking, the image is
/// first smoothed (Gaussian, σ = ½·√(f² − 1) voxels for a size ratio f) so detail finer than
/// the new voxels does not alias.
pub fn resample_to_spacing(vol: &LoadedVolume, spacing: [f64; 3], interp: Interpolation, progress: Progress) -> Result<LoadedVolume> {
    if spacing.iter().any(|&s| !(s > 0.0 && s.is_finite())) {
        bail!("voxel sizes must be positive");
    }
    let old = vol.geometry.spacing;
    let ratio = [0, 1, 2].map(|a| spacing[a] / old[a]);
    let dims = [0, 1, 2].map(|a| ((vol.dims[a] as f64 / ratio[a]).round() as usize).max(1));
    if dims.iter().product::<usize>() > MAX_VOXELS {
        bail!("the result would have {} million voxels (the limit is {} million)", dims.iter().product::<usize>() / 1_000_000, MAX_VOXELS / 1_000_000);
    }
    // Anti-aliasing when shrinking (not for nearest: labels must stay labels).
    let sigma = ratio.map(|f| if f > 1.01 && interp != Interpolation::Nearest { 0.5 * (f * f - 1.0).sqrt() } else { 0.0 });
    let smoothed;
    let src = if sigma.iter().any(|&s| s > 0.1) {
        let v = crate::filter::gaussian(&vol.values, vol.dims, sigma, &|f| progress(0.3 * f))?;
        smoothed = crate::image::with_values(vol, &vol.name, v);
        &smoothed
    } else {
        vol
    };
    // New voxel i ↔ old index c + (i − c') · ratio, centres c and c' of the two grids.
    let c = [0, 1, 2].map(|a| (vol.dims[a] as f64 - 1.0) * 0.5);
    let c2 = [0, 1, 2].map(|a| (dims[a] as f64 - 1.0) * 0.5);
    let new_to_old = DMat4::from_translation(DVec3::new(c[0] - c2[0] * ratio[0], c[1] - c2[1] * ratio[1], c[2] - c2[2] * ratio[2])) * DMat4::from_scale(DVec3::new(ratio[0], ratio[1], ratio[2]));
    let values = fill(src, dims, new_to_old, interp, 0.0, &|f| progress(0.3 + 0.7 * f))?;
    let affine = vol.geometry.matrix() * new_to_old;
    let geometry = Geometry::from_affine([dims[0], dims[1], dims[2], 1], affine);
    check(progress, 1.0)?;
    let mut out = on_grid(&vol.name, geometry, values);
    out.invert = vol.invert;
    out.file_window = vol.file_window;
    Ok(out)
}

/// Resample `moving` onto the grid `grid` (its `dims[0..3]` and affine): output voxel x takes
/// the moving value at world point `transform.matrix · world(x)` (fixed world → moving world).
pub fn resample_onto(moving: &LoadedVolume, grid: &Geometry, transform: &Transform, interp: Interpolation, outside: f32, progress: Progress) -> Result<LoadedVolume> {
    let dims = [grid.dims[0], grid.dims[1], grid.dims[2]];
    let to_src = moving.geometry.matrix().inverse() * transform.matrix * grid.matrix();
    let values = fill(moving, dims, to_src, interp, outside, progress)?;
    let mut out = on_grid(&moving.name, grid.clone(), values);
    out.invert = moving.invert;
    out.file_window = moving.file_window;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::no_progress;

    fn image(dims: [usize; 3], spacing: [f64; 3], f: impl Fn(DVec3) -> f64) -> LoadedVolume {
        let axes = [DVec3::new(0.8, 0.6, 0.0), DVec3::new(-0.6, 0.8, 0.0), DVec3::Z];
        let g = Geometry::from_axes([dims[0], dims[1], dims[2], 1], spacing, axes, DVec3::new(-20.0, 5.0, 3.0));
        let mut values = Vec::with_capacity(dims.iter().product());
        for k in 0..dims[2] {
            for j in 0..dims[1] {
                for i in 0..dims[0] {
                    values.push(f(g.world(DVec3::new(i as f64, j as f64, k as f64))) as f32);
                }
            }
        }
        LoadedVolume::from_values("t", dims, values, g)
    }

    /// A smooth function of world position.
    fn smooth(p: DVec3) -> f64 {
        100.0 + 30.0 * (p.x / 15.0).sin() + 20.0 * (p.y / 11.0).cos() + 10.0 * (p.z / 9.0).sin()
    }

    #[test]
    fn constants_stay_constant() {
        let v = image([12, 10, 8], [1.0, 1.5, 2.0], |_| 7.0);
        for interp in Interpolation::ALL {
            let r = resample_to_spacing(&v, [0.7, 0.9, 1.3], interp, no_progress()).unwrap();
            assert!(r.values.iter().all(|&x| (x - 7.0).abs() < 1e-4), "{interp:?}");
        }
    }

    #[test]
    fn up_and_down_sampling_follow_a_smooth_function() {
        let v = image([40, 36, 30], [1.0, 1.0, 1.5], smooth);
        for interp in Interpolation::ALL {
            for spacing in [[0.5, 0.6, 0.8], [2.0, 2.0, 2.5]] {
                let r = resample_to_spacing(&v, spacing, interp, no_progress()).unwrap();
                // Compare inside (away from the edges, where the edge extension differs).
                let mut worst = 0.0f64;
                for k in 0..r.dims[2] {
                    for j in 0..r.dims[1] {
                        for i in 0..r.dims[0] {
                            let ijk = DVec3::new(i as f64, j as f64, k as f64);
                            let p = r.geometry.world(ijk);
                            let back = v.geometry.matrix().inverse().transform_point3(p);
                            if (0..3).all(|a| back[a] > 3.0 && back[a] < v.dims[a] as f64 - 4.0) {
                                worst = worst.max((r.get(i, j, k) as f64 - smooth(p)).abs());
                            }
                        }
                    }
                }
                let limit = match (interp, spacing[0] < 1.0) {
                    (Interpolation::Nearest, _) => 9.0,
                    (_, true) => 1.0,
                    // Shrinking smooths a little first (anti-aliasing).
                    (_, false) => 5.0,
                };
                assert!(worst < limit, "{interp:?} {spacing:?}: worst error {worst:.3}");
            }
        }
        // The extent is kept.
        let r = resample_to_spacing(&v, [0.5, 0.5, 0.5], Interpolation::Linear, no_progress()).unwrap();
        assert_eq!(r.dims, [80, 72, 90]);
        assert_eq!(r.geometry.spacing.map(|s| (s * 1e6).round() / 1e6), [0.5, 0.5, 0.5]);
        assert!(resample_to_spacing(&v, [0.0, 1.0, 1.0], Interpolation::Linear, no_progress()).is_err());
        assert!(resample_to_spacing(&v, [0.001, 0.001, 0.001], Interpolation::Linear, no_progress()).is_err(), "size cap");
    }

    #[test]
    fn onto_a_grid_through_a_transform() {
        let v = image([30, 28, 20], [1.0, 1.0, 1.5], smooth);
        for interp in Interpolation::ALL {
            let same = resample_onto(&v, &v.geometry, &Transform::identity(), interp, 0.0, no_progress()).unwrap();
            let err = same.values.iter().zip(&v.values).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            assert!(err < 1e-3, "{interp:?} identity: {err}");
        }
        // Moving content: output voxel x shows the moving image at world(x) + t.
        let t = DVec3::new(3.0, -2.0, 1.5);
        let r = resample_onto(&v, &v.geometry, &Transform::translation(t), Interpolation::Cubic, -1.0, no_progress()).unwrap();
        let p = v.geometry.world(DVec3::new(14.0, 13.0, 10.0));
        let got = r.get(14, 13, 10) as f64;
        assert!((got - smooth(p + t)).abs() < 0.5, "{got} vs {}", smooth(p + t));
        // Far outside: the outside value.
        let far = resample_onto(&v, &v.geometry, &Transform::translation(DVec3::splat(500.0)), Interpolation::Linear, -1.0, no_progress()).unwrap();
        assert!(far.values.iter().all(|&x| x == -1.0));
        // Composition and inverse.
        let a = Transform::translation(t);
        assert!((a.inverse().after(&a).apply(p) - p).length() < 1e-9);
        assert!(!resample_onto(&v, &v.geometry, &Transform::identity(), Interpolation::Linear, 0.0, &|_| false).map(|_| ()).is_ok());
    }
}
