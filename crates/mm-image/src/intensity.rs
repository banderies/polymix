//! Intensity normalisation, so thresholds carry over between scans: rescale to a range,
//! clip to percentiles, z-score, and histogram matching to a reference image (piecewise-linear
//! between percentiles, as Nyúl & Udupa's standardisation); and subtraction of two images on
//! the same grid (post − pre contrast).

use anyhow::{bail, Result};
use mm_volume::load::LoadedVolume;
use rayon::prelude::*;

/// Values where `mask` is set (all when `None`), finite only, sorted.
fn sorted_values(v: &[f32], mask: Option<&[u8]>) -> Vec<f32> {
    let mut s: Vec<f32> = match mask {
        Some(m) => v.par_iter().zip(m).filter(|(x, &m)| m != 0 && x.is_finite()).map(|(&x, _)| x).collect(),
        None => v.par_iter().copied().filter(|x| x.is_finite()).collect(),
    };
    s.par_sort_unstable_by(f32::total_cmp);
    s
}

/// The `p`-th percentile (0–100) of sorted values.
fn percentile(sorted: &[f32], p: f64) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let at = (p.clamp(0.0, 100.0) / 100.0) * (sorted.len() - 1) as f64;
    let (i, f) = (at.floor() as usize, at - at.floor());
    let j = (i + 1).min(sorted.len() - 1);
    (sorted[i] as f64 * (1.0 - f) + sorted[j] as f64 * f) as f32
}

/// Map the values' `p_lo`–`p_hi` percentiles (of the masked voxels) linearly onto `lo`–`hi`;
/// with `clip`, values beyond those percentiles are clamped.
pub fn rescale(vol: &LoadedVolume, mask: Option<&[u8]>, p_lo: f64, p_hi: f64, lo: f32, hi: f32, clip: bool) -> Result<Vec<f32>> {
    let s = sorted_values(&vol.values, mask);
    if s.is_empty() {
        bail!("no values to rescale");
    }
    let (a, b) = (percentile(&s, p_lo), percentile(&s, p_hi));
    if b <= a {
        bail!("the image is constant over the chosen range");
    }
    let k = (hi - lo) / (b - a);
    Ok(vol
        .values
        .par_iter()
        .map(|&x| {
            let x = if clip { x.clamp(a, b) } else { x };
            lo + (x - a) * k
        })
        .collect())
}

/// Clamp values to the `p_lo`–`p_hi` percentiles (of the masked voxels), keeping their scale.
pub fn clip_percentiles(vol: &LoadedVolume, mask: Option<&[u8]>, p_lo: f64, p_hi: f64) -> Result<Vec<f32>> {
    let s = sorted_values(&vol.values, mask);
    if s.is_empty() {
        bail!("no values to clip");
    }
    let (a, b) = (percentile(&s, p_lo), percentile(&s, p_hi));
    Ok(vol.values.par_iter().map(|&x| x.clamp(a, b.max(a))).collect())
}

/// (value − mean) / standard deviation, both taken over the masked voxels.
pub fn zscore(vol: &LoadedVolume, mask: Option<&[u8]>) -> Result<Vec<f32>> {
    let s = sorted_values(&vol.values, mask);
    if s.len() < 2 {
        bail!("too few values for a z-score");
    }
    let n = s.len() as f64;
    let mean = s.iter().map(|&x| x as f64).sum::<f64>() / n;
    let sd = (s.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    if sd <= 0.0 {
        bail!("the image is constant");
    }
    Ok(vol.values.par_iter().map(|&x| ((x as f64 - mean) / sd) as f32).collect())
}

/// Percentiles used as landmarks for histogram matching.
const LANDMARKS: [f64; 13] = [0.5, 2.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 98.0, 99.5];

/// Remap `vol`'s intensities so its histogram (over `mask`) matches `reference`'s (over
/// `ref_mask`): piecewise-linear between matching percentiles, extended linearly beyond.
pub fn match_histogram(vol: &LoadedVolume, mask: Option<&[u8]>, reference: &LoadedVolume, ref_mask: Option<&[u8]>) -> Result<Vec<f32>> {
    let a = sorted_values(&vol.values, mask);
    let b = sorted_values(&reference.values, ref_mask);
    if a.len() < 2 || b.len() < 2 {
        bail!("too few values to match histograms");
    }
    let src: Vec<f32> = LANDMARKS.iter().map(|&p| percentile(&a, p)).collect();
    let dst: Vec<f32> = LANDMARKS.iter().map(|&p| percentile(&b, p)).collect();
    // Drop repeated source landmarks (flat histograms), keeping the mapping monotonic.
    let mut pts: Vec<(f32, f32)> = Vec::new();
    for (&s, &d) in src.iter().zip(&dst) {
        if pts.last().is_none_or(|&(ps, _)| s > ps) {
            pts.push((s, d.max(pts.last().map(|p| p.1).unwrap_or(f32::NEG_INFINITY))));
        }
    }
    if pts.len() < 2 {
        bail!("the image is constant over its mask");
    }
    let map = |x: f32| -> f32 {
        let k = pts.partition_point(|&(s, _)| s <= x).clamp(1, pts.len() - 1);
        let ((s0, d0), (s1, d1)) = (pts[k - 1], pts[k]);
        d0 + (x - s0) * (d1 - d0) / (s1 - s0)
    };
    Ok(vol.values.par_iter().map(|&x| map(x)).collect())
}

/// `a − b`, voxel by voxel (both on the same grid).
pub fn subtract(a: &LoadedVolume, b: &LoadedVolume) -> Result<Vec<f32>> {
    if a.dims != b.dims {
        bail!("the images are on different grids (resample or register one onto the other first)");
    }
    Ok(a.values.par_iter().zip(&b.values).map(|(x, y)| x - y).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec3;
    use mm_volume::geometry::Geometry;

    fn image(values: Vec<f32>) -> LoadedVolume {
        let n = values.len();
        LoadedVolume::from_values("t", [n, 1, 1], values, Geometry::from_axes([n, 1, 1, 1], [1.0; 3], [DVec3::X, DVec3::Y, DVec3::Z], DVec3::ZERO))
    }

    #[test]
    fn rescale_clip_and_zscore() {
        let v = image((0..=100).map(|x| x as f32).collect());
        let r = rescale(&v, None, 0.0, 100.0, 0.0, 1.0, false).unwrap();
        assert!((r[50] - 0.5).abs() < 1e-6 && r[100] == 1.0);
        let r = rescale(&v, None, 10.0, 90.0, 0.0, 1.0, true).unwrap();
        assert_eq!((r[0], r[100]), (0.0, 1.0));
        let c = clip_percentiles(&v, None, 5.0, 95.0).unwrap();
        assert_eq!((c[0], c[50], c[100]), (5.0, 50.0, 95.0));
        let z = zscore(&v, None).unwrap();
        assert!(z[50].abs() < 1e-6 && (z[100] - 1.7069).abs() < 1e-3, "{}", z[100]);
        // Masked statistics.
        let mask: Vec<u8> = (0..=100).map(|x| (x >= 50) as u8).collect();
        let z = zscore(&v, Some(&mask)).unwrap();
        assert!(z[75].abs() < 1e-6);
        assert!(zscore(&image(vec![3.0; 5]), None).is_err());
    }

    #[test]
    fn histogram_matching_and_subtraction() {
        // A scan whose intensities are a monotonic distortion of the reference's.
        let reference = image((0..1000).map(|x| (x % 200) as f32).collect());
        let distorted = image(reference.values.iter().map(|&x| 3.0 * x.sqrt() * 10.0 + 7.0).collect());
        let m = match_histogram(&distorted, None, &reference, None).unwrap();
        let err = m.iter().zip(&reference.values).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(err < 4.0, "max error {err}");
        let d = subtract(&reference, &image(vec![1.0; 1000])).unwrap();
        assert_eq!(d[5], 4.0);
        assert!(subtract(&reference, &image(vec![1.0; 10])).is_err());
    }
}
