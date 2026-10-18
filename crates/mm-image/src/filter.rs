//! Separable Gaussian smoothing, with the standard deviation in millimetres (so anisotropic
//! voxels are smoothed evenly in space). The edges are extended with their own values.

use crate::image::Progress;
use mm_volume::load::LoadedVolume;
use rayon::prelude::*;

/// Smooth `values` (a `dims` grid) with per-axis standard deviations in voxels. Axes with σ
/// below 0.1 voxel are left alone.
pub fn gaussian(values: &[f32], dims: [usize; 3], sigma_vox: [f64; 3], progress: Progress) -> anyhow::Result<Vec<f32>> {
    let [nx, ny, _] = dims;
    let plane = nx * ny;
    let mut out = values.to_vec();
    for axis in 0..3 {
        let s = sigma_vox[axis];
        let len = dims[axis];
        if s < 0.1 || len < 2 {
            continue;
        }
        let r = (3.0 * s).ceil() as i64;
        let w: Vec<f32> = (-r..=r).map(|x| (-((x * x) as f64) / (2.0 * s * s)).exp() as f32).collect();
        let total: f32 = w.iter().sum();
        let w: Vec<f32> = w.iter().map(|x| x / total).collect();
        let stride = [1, nx, plane][axis];
        let starts: Vec<usize> = match axis {
            0 => (0..dims[1] * dims[2]).map(|l| l * nx).collect(),
            1 => (0..nx * dims[2]).map(|l| (l / nx) * plane + l % nx).collect(),
            _ => (0..plane).collect(),
        };
        let src = out.clone();
        let lines: Vec<(usize, Vec<f32>)> = starts
            .par_iter()
            .map(|&start| {
                let line: Vec<f32> = (0..len)
                    .map(|x| {
                        let mut acc = 0.0;
                        for (t, &wt) in w.iter().enumerate() {
                            let y = (x as i64 + t as i64 - r).clamp(0, len as i64 - 1) as usize;
                            acc += wt * src[start + y * stride];
                        }
                        acc
                    })
                    .collect();
                (start, line)
            })
            .collect();
        for (start, line) in lines {
            for (x, v) in line.into_iter().enumerate() {
                out[start + x * stride] = v;
            }
        }
        crate::image::check(progress, (axis + 1) as f32 / 3.0)?;
    }
    Ok(out)
}

/// Smooth an image with a Gaussian of standard deviation `sigma_mm`.
pub fn gaussian_mm(vol: &LoadedVolume, sigma_mm: f64, progress: Progress) -> anyhow::Result<Vec<f32>> {
    let s = vol.geometry.spacing.map(|sp| sigma_mm / sp.max(1e-9));
    gaussian(&vol.values, vol.dims, s, progress)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::no_progress;

    #[test]
    fn spreads_an_impulse_by_sigma() {
        let dims = [41, 3, 3];
        let mut v = vec![0.0f32; 41 * 9];
        v[4 * 41 + 20] = 1.0; // the middle of the centre line
        let out = gaussian(&v, dims, [3.0, 0.0, 0.0], no_progress()).unwrap();
        let line = &out[4 * 41..5 * 41];
        let sum: f32 = line.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
        let var: f32 = line.iter().enumerate().map(|(x, &w)| w * (x as f32 - 20.0).powi(2)).sum();
        assert!((var.sqrt() - 3.0).abs() < 0.05, "σ {}", var.sqrt());
        // A constant stays constant.
        let c = gaussian(&vec![5.0; 41 * 9], dims, [2.0, 1.0, 1.0], no_progress()).unwrap();
        assert!(c.iter().all(|&x| (x - 5.0).abs() < 1e-4));
    }
}
