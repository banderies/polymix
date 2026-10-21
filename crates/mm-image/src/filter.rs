//! Separable Gaussian smoothing, with the standard deviation in millimetres (so anisotropic
//! voxels are smoothed evenly in space). The edges are extended with their own values.

use crate::image::Progress;
use mm_volume::load::LoadedVolume;
use rayon::prelude::*;

/// Smooth `values` (a `dims` grid) with per-axis standard deviations in voxels. Axes with σ
/// below 0.1 voxel are left alone.
///
/// Every voxel of a pass is its taps' weighted values added from 0 in tap order, the edges
/// clamped. The passes go row by row: along x within the row, along y and z by adding whole
/// rows of the neighbouring lines, which keeps the per-voxel order and lets the additions run
/// side by side. Two buffers take turns; the first pass reads `values` directly.
pub fn gaussian(values: &[f32], dims: [usize; 3], sigma_vox: [f64; 3], progress: Progress) -> anyhow::Result<Vec<f32>> {
    let [nx, ny, nz] = dims;
    // The last pass's output (none yet: still `values`) and a buffer for the next pass.
    let mut cur: Option<Vec<f32>> = None;
    let mut spare: Option<Vec<f32>> = None;
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
        let r = r as usize;
        let src: &[f32] = cur.as_deref().unwrap_or(values);
        let mut dst = spare.take().unwrap_or_else(|| vec![0.0f32; values.len()]);
        let row_of = |k: usize, j: usize| &src[(k * ny + j) * nx..][..nx];
        let near = |c: usize, t: usize, n: usize| (c + t).saturating_sub(r).min(n - 1);
        dst.par_chunks_mut(nx.max(1)).enumerate().for_each(|(row, o)| {
            let (j, k) = (row % ny, row / ny);
            match axis {
                0 => smooth_line(row_of(k, j), &w, r, o),
                1 => add_rows(o, &w, |t| row_of(k, near(j, t, ny))),
                _ => add_rows(o, &w, |t| row_of(near(k, t, nz), j)),
            }
        });
        spare = cur.replace(dst);
        crate::image::check(progress, (axis + 1) as f32 / 3.0)?;
    }
    Ok(cur.unwrap_or_else(|| values.to_vec()))
}

/// `out[x] = Σ_t w[t] · line[x + t − r]` (indices clamped to the line), summed from 0 in tap
/// order.
fn smooth_line(line: &[f32], w: &[f32], r: usize, out: &mut [f32]) {
    let n = line.len();
    out.fill(0.0);
    for (t, &wt) in w.iter().enumerate() {
        // x + t − r falls inside the line for x in lo..hi; before it the first value repeats,
        // after it the last.
        let lo = r.saturating_sub(t).min(n);
        let hi = (n + r).saturating_sub(t).min(n).max(lo);
        for o in &mut out[..lo] {
            *o += wt * line[0];
        }
        if lo < hi {
            for (o, &v) in out[lo..hi].iter_mut().zip(&line[lo + t - r..hi + t - r]) {
                *o += wt * v;
            }
        }
        for o in &mut out[hi..] {
            *o += wt * line[n - 1];
        }
    }
}

/// `out = Σ_t w[t] · row(t)`, summed from 0 in tap order.
fn add_rows<'a>(out: &mut [f32], w: &[f32], row: impl Fn(usize) -> &'a [f32]) {
    out.fill(0.0);
    for (t, &wt) in w.iter().enumerate() {
        for (o, &v) in out.iter_mut().zip(row(t)) {
            *o += wt * v;
        }
    }
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

    /// The filter written plainly: each voxel of each pass on its own, its taps added from 0 in
    /// order.
    fn reference(values: &[f32], dims: [usize; 3], sigma_vox: [f64; 3]) -> Vec<f32> {
        let stride = [1, dims[0], dims[0] * dims[1]];
        let mut out = values.to_vec();
        for axis in 0..3 {
            let (s, len) = (sigma_vox[axis], dims[axis]);
            if s < 0.1 || len < 2 {
                continue;
            }
            let r = (3.0 * s).ceil() as i64;
            let w: Vec<f32> = (-r..=r).map(|x| (-((x * x) as f64) / (2.0 * s * s)).exp() as f32).collect();
            let total: f32 = w.iter().sum();
            let w: Vec<f32> = w.iter().map(|x| x / total).collect();
            let src = out.clone();
            for (q, o) in out.iter_mut().enumerate() {
                let x = (q / stride[axis]) % len;
                let start = q - x * stride[axis];
                let mut acc = 0.0f32;
                for (t, &wt) in w.iter().enumerate() {
                    acc += wt * src[start + (x as i64 + t as i64 - r).clamp(0, len as i64 - 1) as usize * stride[axis]];
                }
                *o = acc;
            }
        }
        out
    }

    #[test]
    fn matches_the_plain_filter_bit_for_bit() {
        let mut seed = 99u64;
        for dims in [[23, 17, 11], [2, 3, 2], [1, 7, 5], [64, 1, 3], [5, 4, 40]] {
            let n = dims.iter().product();
            let v: Vec<f32> = (0..n)
                .map(|q| {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    // Some exact zeros of both signs among noisy values.
                    if q % 13 == 0 { -0.0 } else if q % 17 == 0 { 0.0 } else { ((seed >> 40) as f32 / 65536.0 - 128.0) * 3.7 }
                })
                .collect();
            for s in [[1.0, 1.0, 1.0], [0.5, 2.5, 0.0], [0.0, 0.0, 4.0], [3.0, 0.05, 1.3], [0.0; 3], [7.0, 7.0, 7.0]] {
                let a = gaussian(&v, dims, s, no_progress()).unwrap();
                let b = reference(&v, dims, s);
                assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()), "{dims:?} σ {s:?}");
            }
        }
        // Cancelling stops it.
        assert!(gaussian(&[0.0; 8], [2, 2, 2], [1.0; 3], &|_| false).is_err());
    }
}
