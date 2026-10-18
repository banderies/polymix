//! Rigid and affine registration of two scans of a patient (TOF-MRA ↔ T1, pre ↔ post contrast):
//! the transform from the fixed image's world to the moving image's world that best lines up
//! their intensities.
//!
//! - Start: the moving image's centre of intensity mass placed on the fixed one's (foreground
//!   above half of Otsu's threshold; geometric centres when there is no foreground).
//! - Pyramid: block-averaged and lightly smoothed copies, coarse to fine (halving voxels per
//!   level), so large misalignments are found before details.
//! - Metric on a fixed, seeded set of points in the fixed image's foreground: normalised
//!   cross-correlation (same contrast) or Mattes-style mutual information (different
//!   contrasts; 48-bin joint histogram with partial-volume binning so it changes smoothly),
//!   moving values by trilinear interpolation. Higher is better.
//! - Optimiser: regular-step gradient ascent with central-difference gradients, step halving
//!   on failure. Parameters are scaled so a unit step moves points about 1 mm: rotations, log
//!   scales and shears by the sample radius, translations in mm. Rotations and the affine
//!   part act about the fixed image's centre of mass.

use crate::image::{check, Progress};
use crate::resample::Transform;
use anyhow::{bail, Result};
use glam::{DMat3, DMat4, DVec3, DVec4};
use mm_volume::geometry::Geometry;
use mm_volume::load::LoadedVolume;
use rayon::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Motion {
    /// Rotation and translation.
    Rigid,
    /// Also scaling and shearing.
    Affine,
}

impl Motion {
    pub fn label(self) -> &'static str {
        match self {
            Motion::Rigid => "Rigid",
            Motion::Affine => "Affine",
        }
    }

    fn params(self) -> usize {
        match self {
            Motion::Rigid => 6,
            Motion::Affine => 12,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metric {
    /// Mattes-style mutual information: images of different contrasts.
    MutualInformation,
    /// Normalised cross-correlation: images of the same contrast.
    Correlation,
}

impl Metric {
    pub fn label(self) -> &'static str {
        match self {
            Metric::MutualInformation => "Mutual information",
            Metric::Correlation => "Correlation",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RegisterParams {
    pub motion: Motion,
    pub metric: Metric,
    /// Pyramid levels (each halves the voxels per axis).
    pub levels: usize,
    /// Optimiser iterations per level.
    pub iterations: usize,
    /// Points sampled per level.
    pub samples: usize,
}

impl Default for RegisterParams {
    fn default() -> Self {
        Self { motion: Motion::Rigid, metric: Metric::MutualInformation, levels: 3, iterations: 100, samples: 20_000 }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Registration {
    /// Fixed world → moving world (what `resample_onto` takes to put the moving image on the
    /// fixed grid).
    pub transform: Transform,
    /// The metric with the images as they are (identity) and after registration, on the
    /// finest level.
    pub metric_before: f64,
    pub metric_after: f64,
    /// Optimiser iterations run at each level, coarse to fine.
    pub iterations: Vec<usize>,
}

/// One pyramid level of an image: values with world → voxel and voxel → world maps.
struct Level {
    dims: [usize; 3],
    values: Vec<f32>,
    to_voxel: DMat4,
    to_world: DMat4,
}

impl Level {
    fn new(vol: &LoadedVolume, f: usize, progress: Progress) -> Result<Self> {
        let g = &vol.geometry;
        if f <= 1 {
            return Ok(Self { dims: vol.dims, values: vol.values.clone(), to_voxel: g.matrix().inverse(), to_world: g.matrix() });
        }
        let d = vol.dims.map(|n| n.div_ceil(f));
        let mut values = vec![0.0f32; d[0] * d[1] * d[2]];
        values.par_chunks_mut(d[0] * d[1]).enumerate().for_each(|(z, p)| {
            for y in 0..d[1] {
                for x in 0..d[0] {
                    let (mut sum, mut n) = (0.0f64, 0usize);
                    for k in z * f..((z + 1) * f).min(vol.dims[2]) {
                        for j in y * f..((y + 1) * f).min(vol.dims[1]) {
                            for i in x * f..((x + 1) * f).min(vol.dims[0]) {
                                sum += vol.get(i, j, k) as f64;
                                n += 1;
                            }
                        }
                    }
                    p[y * d[0] + x] = (sum / n.max(1) as f64) as f32;
                }
            }
        });
        let values = crate::filter::gaussian(&values, d, [0.5; 3], progress)?;
        // Level voxel v is the centre of block v: original index f·v + (f − 1)/2.
        let half = (f as f64 - 1.0) * 0.5;
        let to_world = g.matrix() * DMat4::from_translation(DVec3::splat(half)) * DMat4::from_scale(DVec3::splat(f as f64));
        Ok(Self { dims: d, values, to_voxel: to_world.inverse(), to_world })
    }

    /// Trilinear value at continuous voxel coordinates; `None` outside the grid.
    fn linear(&self, v: DVec3) -> Option<f32> {
        let d = self.dims;
        if (0..3).any(|a| !(v[a] >= 0.0 && v[a] <= d[a] as f64 - 1.0)) {
            return None;
        }
        let i = (v.x.floor() as usize).min(d[0].saturating_sub(2));
        let j = (v.y.floor() as usize).min(d[1].saturating_sub(2));
        let k = (v.z.floor() as usize).min(d[2].saturating_sub(2));
        let (fx, fy, fz) = ((v.x - i as f64) as f32, (v.y - j as f64) as f32, (v.z - k as f64) as f32);
        let (i1, j1, k1) = ((i + 1).min(d[0] - 1), (j + 1).min(d[1] - 1), (k + 1).min(d[2] - 1));
        let at = |i: usize, j: usize, k: usize| self.values[(k * d[1] + j) * d[0] + i];
        let l = |a: f32, b: f32, t: f32| a + (b - a) * t;
        let c00 = l(at(i, j, k), at(i1, j, k), fx);
        let c10 = l(at(i, j1, k), at(i1, j1, k), fx);
        let c01 = l(at(i, j, k1), at(i1, j, k1), fx);
        let c11 = l(at(i, j1, k1), at(i1, j1, k1), fx);
        Some(l(l(c00, c10, fy), l(c01, c11, fy), fz))
    }
}

/// Deterministic random numbers (SplitMix64).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Voxels above half of Otsu's threshold (all voxels when that finds nothing).
fn foreground(values: &[f32]) -> Vec<usize> {
    let step = (values.len() / 1_000_000).max(1);
    let sample: Vec<f32> = values.iter().step_by(step).copied().collect();
    let t = mm_segment::ops::otsu(&sample).map(|t| t * 0.5);
    let fg: Vec<usize> = match t {
        Some(t) => (0..values.len()).filter(|&n| values[n] > t).collect(),
        None => Vec::new(),
    };
    if fg.len() < 16 { (0..values.len()).collect() } else { fg }
}

/// Intensity-weighted centre of the foreground, in world coordinates.
fn centre_of_mass(vol: &LoadedVolume) -> DVec3 {
    let fg = foreground(&vol.values);
    let (mut sum, mut w) = (DVec3::ZERO, 0.0f64);
    let plane = vol.dims[0] * vol.dims[1];
    for &n in &fg {
        let x = vol.values[n].max(0.0) as f64;
        let ijk = DVec3::new((n % vol.dims[0]) as f64, ((n / vol.dims[0]) % vol.dims[1]) as f64, (n / plane) as f64);
        sum += ijk * x;
        w += x;
    }
    let ijk = if w > 0.0 { sum / w } else { DVec3::new(vol.dims[0] as f64 - 1.0, vol.dims[1] as f64 - 1.0, vol.dims[2] as f64 - 1.0) * 0.5 };
    vol.geometry.world(ijk)
}

/// Robust intensity range (0.5th–99.5th percentile) of some values.
fn range(values: impl Iterator<Item = f32>) -> (f32, f32) {
    let mut v: Vec<f32> = values.filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return (0.0, 1.0);
    }
    v.sort_by(f32::total_cmp);
    let q = |p: f64| v[((v.len() - 1) as f64 * p).round() as usize];
    let (lo, hi) = (q(0.005), q(0.995));
    if hi > lo { (lo, hi) } else { (lo, lo + 1.0) }
}

const BINS: usize = 48;

/// The sampled points of one level and what is needed to score a transform there.
struct Problem<'a> {
    points: Vec<DVec3>,
    fixed: Vec<f32>,
    /// Fixed histogram bin of each point (mutual information).
    fixed_bin: Vec<u16>,
    moving: &'a Level,
    moving_range: (f32, f32),
    metric: Metric,
}

impl Problem<'_> {
    /// Score of `t` (fixed world → moving world); very low when the images barely overlap.
    fn score(&self, t: &DMat4) -> f64 {
        let to_vox = self.moving.to_voxel * *t;
        let n = self.points.len();
        let min_overlap = (n / 4).max(8);
        match self.metric {
            Metric::Correlation => {
                let s = self
                    .points
                    .par_iter()
                    .zip(&self.fixed)
                    .with_min_len(1024)
                    .fold(
                        || [0.0f64; 6],
                        |mut a, (p, &f)| {
                            if let Some(m) = self.moving.linear((to_vox * p.extend(1.0)).truncate()) {
                                let (f, m) = (f as f64, m as f64);
                                a[0] += 1.0;
                                a[1] += f;
                                a[2] += m;
                                a[3] += f * f;
                                a[4] += m * m;
                                a[5] += f * m;
                            }
                            a
                        },
                    )
                    .reduce(|| [0.0; 6], |a, b| [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3], a[4] + b[4], a[5] + b[5]]);
                if (s[0] as usize) < min_overlap {
                    return -1e9;
                }
                let c = s[0];
                let (vf, vm) = (s[3] - s[1] * s[1] / c, s[4] - s[2] * s[2] / c);
                if vf <= 1e-12 || vm <= 1e-12 {
                    return -1.0;
                }
                (s[5] - s[1] * s[2] / c) / (vf * vm).sqrt()
            }
            Metric::MutualInformation => {
                let (lo, hi) = self.moving_range;
                let scale = (BINS - 1) as f32 / (hi - lo);
                let joint = self
                    .points
                    .par_iter()
                    .zip(&self.fixed_bin)
                    .with_min_len(1024)
                    .fold(
                        || vec![0.0f64; BINS * BINS],
                        |mut h, (p, &fb)| {
                            if let Some(m) = self.moving.linear((to_vox * p.extend(1.0)).truncate()) {
                                let c = ((m - lo) * scale).clamp(0.0, (BINS - 1) as f32);
                                let b = (c.floor() as usize).min(BINS - 2);
                                let w = (c - b as f32) as f64;
                                h[fb as usize * BINS + b] += 1.0 - w;
                                h[fb as usize * BINS + b + 1] += w;
                            }
                            h
                        },
                    )
                    .reduce(|| vec![0.0f64; BINS * BINS], |mut a, b| {
                        a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
                        a
                    });
                let total: f64 = joint.iter().sum();
                if (total as usize) < min_overlap {
                    return -1e9;
                }
                let mut pf = [0.0f64; BINS];
                let mut pm = [0.0f64; BINS];
                for a in 0..BINS {
                    for b in 0..BINS {
                        let v = joint[a * BINS + b] / total;
                        pf[a] += v;
                        pm[b] += v;
                    }
                }
                let mut mi = 0.0;
                for a in 0..BINS {
                    for b in 0..BINS {
                        let v = joint[a * BINS + b] / total;
                        if v > 0.0 {
                            mi += v * (v / (pf[a] * pm[b])).ln();
                        }
                    }
                }
                mi
            }
        }
    }
}

/// The transform for parameters `x` (actual units): rotations (rad), translation (mm), log
/// scales and shears, acting about `centre`, after the initial shift `shift`.
fn transform_of(x: &[f64], centre: DVec3, shift: DVec3) -> DMat4 {
    let r = DMat3::from_rotation_z(x[2]) * DMat3::from_rotation_y(x[1]) * DMat3::from_rotation_x(x[0]);
    let a = if x.len() >= 12 {
        let s = DMat3::from_diagonal(DVec3::new(x[6].exp(), x[7].exp(), x[8].exp()));
        let shear = DMat3::from_cols(DVec3::X, DVec3::new(x[9], 1.0, 0.0), DVec3::new(x[10], x[11], 1.0));
        r * shear * s
    } else {
        r
    };
    let t = DVec3::new(x[3], x[4], x[5]);
    // p ↦ A (p − c) + c + shift + t
    let offset = centre + shift + t - a * centre;
    DMat4::from_cols(a.x_axis.extend(0.0), a.y_axis.extend(0.0), a.z_axis.extend(0.0), DVec4::new(offset.x, offset.y, offset.z, 1.0))
}

/// Register `moving` to `fixed`. `progress` gets the fraction done; returning `false` cancels.
pub fn register(fixed: &LoadedVolume, moving: &LoadedVolume, p: &RegisterParams, progress: Progress) -> Result<Registration> {
    if fixed.values.is_empty() || moving.values.is_empty() {
        bail!("an image is empty");
    }
    let levels = p.levels.clamp(1, 6);
    let np = p.motion.params();
    let centre = centre_of_mass(fixed);
    let shift = centre_of_mass(moving) - centre;
    let mut x = vec![0.0f64; np];
    let mut runs = Vec::new();
    let mut before = 0.0;
    let mut after = 0.0;
    let mut seed = Rng(0x5eed_1234_abcd_0001);
    for level in 0..levels {
        let f = 1usize << (levels - 1 - level);
        // Keep at least 8 voxels per axis at the coarsest level.
        let f = (0..=f.trailing_zeros()).map(|e| f >> e).find(|&g| fixed.dims.iter().chain(&moving.dims).all(|&d| d.div_ceil(g) >= 8)).unwrap_or(1);
        let lp = |a: f32| move |t: f32| progress((level as f32 + a * t) / levels as f32);
        let fl = Level::new(fixed, f, &lp(0.05))?;
        let ml = Level::new(moving, f, &lp(0.1))?;
        check(progress, (level as f32 + 0.1) / levels as f32)?;
        // Sample points: random foreground voxels of the fixed level, jittered inside the voxel.
        let fg = foreground(&fl.values);
        let count = p.samples.max(100).min(fg.len() * 4);
        let plane = fl.dims[0] * fl.dims[1];
        let mut points = Vec::with_capacity(count);
        let mut fixed_values = Vec::with_capacity(count);
        for _ in 0..count {
            let n = fg[(seed.next() % fg.len() as u64) as usize];
            let ijk = DVec3::new((n % fl.dims[0]) as f64, ((n / fl.dims[0]) % fl.dims[1]) as f64, (n / plane) as f64);
            let jitter = DVec3::new(seed.unit() - 0.5, seed.unit() - 0.5, seed.unit() - 0.5) * 0.8;
            let v = (ijk + jitter).clamp(DVec3::ZERO, DVec3::new(fl.dims[0] as f64 - 1.0, fl.dims[1] as f64 - 1.0, fl.dims[2] as f64 - 1.0));
            if let Some(val) = fl.linear(v) {
                points.push(fl.to_world.transform_point3(v));
                fixed_values.push(val);
            }
        }
        let (flo, fhi) = range(fixed_values.iter().copied());
        let fixed_bin = fixed_values.iter().map(|&v| (((v - flo) / (fhi - flo) * (BINS - 1) as f32).round().clamp(0.0, (BINS - 1) as f32)) as u16).collect();
        let step = (ml.values.len() / 200_000).max(1);
        let moving_range = range(ml.values.iter().step_by(step).copied());
        let problem = Problem { points, fixed: fixed_values, fixed_bin, moving: &ml, moving_range, metric: p.metric };
        // Parameter scales: a unit (scaled) step moves points about 1 mm.
        let radius = {
            let n = problem.points.len().max(1) as f64;
            (problem.points.iter().map(|q| (*q - centre).length_squared()).sum::<f64>() / n).sqrt().max(1.0)
        };
        let scales: Vec<f64> = (0..np).map(|k| if (3..6).contains(&k) { 1.0 } else { radius }).collect();
        let score = |x: &[f64]| problem.score(&transform_of(x, centre, shift));
        if level + 1 == levels {
            before = problem.score(&DMat4::IDENTITY);
        }
        // Gradient ascent in scaled coordinates along conjugate directions (Polak–Ribière),
        // the step growing after a success and halving after a failure.
        let voxel = fl.to_world.x_axis.truncate().length().min(fl.to_world.y_axis.truncate().length()).min(fl.to_world.z_axis.truncate().length());
        let mut step = 2.0 * voxel;
        let min_step = 0.0005 * voxel.max(0.05);
        let h = (0.25 * voxel).max(0.05);
        let mut best = score(&x);
        let mut it = 0;
        let mut prev_grad: Option<Vec<f64>> = None;
        let mut dir = vec![0.0; np];
        while it < p.iterations && step > min_step {
            it += 1;
            // Gradient with respect to the scaled parameters.
            let grad: Vec<f64> = (0..np)
                .map(|k| {
                    let mut a = x.clone();
                    let mut b = x.clone();
                    a[k] += h / scales[k];
                    b[k] -= h / scales[k];
                    (score(&a) - score(&b)) / (2.0 * h)
                })
                .collect();
            let gg: f64 = grad.iter().map(|g| g * g).sum();
            if gg <= 0.0 || !gg.is_finite() {
                break;
            }
            let beta = match &prev_grad {
                Some(pg) => {
                    let pgg: f64 = pg.iter().map(|g| g * g).sum();
                    (grad.iter().zip(pg).map(|(g, q)| g * (g - q)).sum::<f64>() / pgg).max(0.0)
                }
                None => 0.0,
            };
            for k in 0..np {
                dir[k] = grad[k] + beta * dir[k];
            }
            // Not an ascent direction any more: restart along the gradient.
            if dir.iter().zip(&grad).map(|(d, g)| d * g).sum::<f64>() <= 0.0 {
                dir.clone_from(&grad);
            }
            prev_grad = Some(grad);
            let norm = dir.iter().map(|d| d * d).sum::<f64>().sqrt();
            let cand: Vec<f64> = (0..np).map(|k| x[k] + step * dir[k] / norm / scales[k]).collect();
            let s = score(&cand);
            if s > best {
                x = cand;
                best = s;
                step = (step * 1.5).min(4.0 * voxel);
            } else {
                step *= 0.5;
                // Start the conjugate directions over from the gradient.
                prev_grad = None;
                dir.iter_mut().for_each(|d| *d = 0.0);
            }
            if it % 4 == 0 {
                check(progress, (level as f32 + 0.1 + 0.9 * it as f32 / p.iterations as f32) / levels as f32)?;
            }
        }
        runs.push(it);
        if level + 1 == levels {
            after = best;
        }
    }
    check(progress, 1.0)?;
    Ok(Registration { transform: Transform { matrix: transform_of(&x, centre, shift) }, metric_before: before, metric_after: after, iterations: runs })
}

/// Save a transform for 3D Slicer / ITK (`.tfm`): an `AffineTransform_double_3_3` in ITK's
/// LPS coordinates, in ITK's resampling direction — it maps points of the fixed image to the
/// moving image (as `Registration::transform` does), which is how ITK and Slicer's BRAINSFit
/// store registration results: applied to the moving image in Slicer, it lines it up with the
/// fixed one.
pub fn write_itk_transform(path: &std::path::Path, t: &Transform) -> Result<()> {
    let flip = DMat4::from_scale(DVec3::new(-1.0, -1.0, 1.0));
    let m = flip * t.matrix * flip;
    let a = [m.x_axis, m.y_axis, m.z_axis];
    let mut params = Vec::new();
    for row in 0..3 {
        for col in a.iter() {
            params.push(col[row]);
        }
    }
    params.extend([m.w_axis.x, m.w_axis.y, m.w_axis.z]);
    let p: Vec<String> = params.iter().map(|v| format!("{v}")).collect();
    let text = format!("#Insight Transform File V1.0\n#Transform 0\nTransform: AffineTransform_double_3_3\nParameters: {}\nFixedParameters: 0 0 0\n", p.join(" "));
    std::fs::write(path, text)?;
    Ok(())
}

/// The grid of `vol` (for resampling onto it).
pub fn grid_of(vol: &LoadedVolume) -> Geometry {
    let mut g = vol.geometry.clone();
    g.dims = [vol.dims[0], vol.dims[1], vol.dims[2], 1];
    g
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::no_progress;
    use crate::resample::{resample_onto, Interpolation};

    /// Soft inside-ness of an ellipsoid (1 inside, 0 outside, a few-mm ramp at the surface so
    /// 2 mm voxels sample it without aliasing).
    fn soft(p: DVec3, c: DVec3, r: DVec3) -> f64 {
        let d = ((p - c) / r).length();
        let dist_mm = (d - 1.0) * r.min_element();
        1.0 / (1.0 + (dist_mm / 1.2).exp())
    }

    /// A head-like phantom: tissue intensities `t` for (head, inner, small structures A and
    /// B), given by world position. The small structures sit asymmetrically in all three
    /// directions so every rotation and scaling shows.
    fn phantom(p: DVec3, t: [f64; 4]) -> f64 {
        let mut v = 0.0;
        v += (t[0] - v) * soft(p, DVec3::ZERO, DVec3::new(45.0, 55.0, 40.0));
        v += (t[1] - v) * soft(p, DVec3::new(5.0, 8.0, 3.0), DVec3::new(25.0, 30.0, 20.0));
        for c in [DVec3::new(-20.0, -25.0, 10.0), DVec3::new(25.0, 30.0, 18.0), DVec3::new(-28.0, 20.0, -20.0)] {
            v += (t[2] - v) * soft(p, c, DVec3::splat(8.0));
        }
        for (c, r) in [(DVec3::new(18.0, -10.0, -18.0), DVec3::new(7.0, 12.0, 6.0)), (DVec3::new(-5.0, -38.0, -12.0), DVec3::new(10.0, 5.0, 7.0)), (DVec3::new(30.0, 5.0, 25.0), DVec3::new(5.0, 6.0, 9.0))] {
            v += (t[3] - v) * soft(p, c, r);
        }
        v
    }

    fn image(dims: [usize; 3], spacing: f64, origin: DVec3, f: impl Fn(DVec3) -> f64 + Sync) -> LoadedVolume {
        let g = Geometry::from_axes([dims[0], dims[1], dims[2], 1], [spacing; 3], [DVec3::X, DVec3::Y, DVec3::Z], origin);
        let plane = dims[0] * dims[1];
        let values: Vec<f32> = (0..plane * dims[2])
            .into_par_iter()
            .map(|n| {
                let p = g.world(DVec3::new((n % dims[0]) as f64, ((n / dims[0]) % dims[1]) as f64, (n / plane) as f64));
                // A little deterministic noise.
                let noise = ((n as u64).wrapping_mul(2654435761) % 1000) as f64 / 1000.0 - 0.5;
                (f(p) + 4.0 * noise) as f32
            })
            .collect();
        LoadedVolume::from_values("phantom", dims, values, g)
    }

    /// Largest distance between `a · b` and the identity over points within 50 mm.
    fn error_mm(m: DMat4) -> f64 {
        let mut worst = 0.0f64;
        for x in [-50.0, 0.0, 50.0] {
            for y in [-50.0, 0.0, 50.0] {
                for z in [-40.0, 0.0, 40.0] {
                    let p = DVec3::new(x, y, z);
                    worst = worst.max((m.transform_point3(p) - p).length());
                }
            }
        }
        worst
    }

    /// The moving image shows the phantom moved by `g`: its value at q is phantom(g(q)), so
    /// the registration should find g⁻¹.
    fn moved(g: DMat4, t: [f64; 4]) -> LoadedVolume {
        image([60, 70, 52], 2.0, DVec3::new(-58.0, -68.0, -50.0), move |q| phantom(g.transform_point3(q), t))
    }

    fn rigid(deg: f64, axis: DVec3, t: DVec3) -> DMat4 {
        DMat4::from_translation(t) * DMat4::from_axis_angle(axis.normalize(), deg.to_radians())
    }

    const T1: [f64; 4] = [300.0, 600.0, 900.0, 450.0];

    #[test]
    fn rigid_with_correlation() {
        let fixed = image([64, 72, 56], 2.0, DVec3::new(-62.0, -70.0, -54.0), |p| phantom(p, T1));
        let g = rigid(8.0, DVec3::new(0.3, 0.2, 1.0), DVec3::new(10.0, -6.0, 4.0));
        let moving = moved(g, T1);
        let start = std::time::Instant::now();
        let r = register(&fixed, &moving, &RegisterParams { metric: Metric::Correlation, ..Default::default() }, no_progress()).unwrap();
        let err = error_mm(g * r.transform.matrix);
        eprintln!("rigid / correlation: {:.3} mm, {:.2?}, iterations {:?}, metric {:.4} → {:.4}", err, start.elapsed(), r.iterations, r.metric_before, r.metric_after);
        assert!(err < 0.6, "error {err:.3} mm");
        assert!(r.metric_after > r.metric_before);
        // The moving image resampled onto the fixed grid lines up.
        let on = resample_onto(&moving, &grid_of(&fixed), &r.transform, Interpolation::Linear, 0.0, no_progress()).unwrap();
        let diff: f64 = on.values.iter().zip(&fixed.values).map(|(a, b)| (a - b).abs() as f64).sum::<f64>() / on.values.len() as f64;
        assert!(diff < 15.0, "mean difference {diff:.2}");
    }

    #[test]
    fn rigid_with_mutual_information_across_contrasts() {
        let fixed = image([64, 72, 56], 2.0, DVec3::new(-62.0, -70.0, -54.0), |p| phantom(p, T1));
        let g = rigid(-7.0, DVec3::new(1.0, -0.4, 0.2), DVec3::new(-8.0, 9.0, -5.0));
        // A different contrast, not monotonic in the first one (background stays dark).
        let moving = moved(g, [700.0, 200.0, 450.0, 950.0]);
        let r = register(&fixed, &moving, &RegisterParams { metric: Metric::MutualInformation, ..Default::default() }, no_progress()).unwrap();
        let err = error_mm(g * r.transform.matrix);
        eprintln!("rigid / MI: {:.3} mm, iterations {:?}, metric {:.4} → {:.4}", err, r.iterations, r.metric_before, r.metric_after);
        assert!(err < 0.8, "error {err:.3} mm");
    }

    #[test]
    fn affine_recovers_anisotropic_scaling() {
        let fixed = image([64, 72, 56], 2.0, DVec3::new(-62.0, -70.0, -54.0), |p| phantom(p, T1));
        let g = DMat4::from_translation(DVec3::new(4.0, -3.0, 2.0)) * DMat4::from_scale(DVec3::new(1.08, 0.94, 1.03)) * DMat4::from_rotation_z(5f64.to_radians());
        let moving = moved(g, T1);
        let rigid_only = register(&fixed, &moving, &RegisterParams { metric: Metric::Correlation, ..Default::default() }, no_progress()).unwrap();
        let r = register(&fixed, &moving, &RegisterParams { motion: Motion::Affine, metric: Metric::Correlation, iterations: 200, ..Default::default() }, no_progress()).unwrap();
        let (e_rigid, e_affine) = (error_mm(g * rigid_only.transform.matrix), error_mm(g * r.transform.matrix));
        eprintln!("affine: rigid {e_rigid:.3} mm, affine {e_affine:.3} mm, iterations {:?}", r.iterations);
        assert!(e_affine < 1.0 && e_affine < e_rigid / 3.0, "rigid {e_rigid:.3}, affine {e_affine:.3}");
    }

    #[test]
    fn itk_transform_file_in_lps() {
        let dir = std::env::temp_dir().join(format!("mmi-tfm-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t.tfm");
        write_itk_transform(&p, &Transform::translation(DVec3::new(10.0, 20.0, 30.0))).unwrap();
        let text = std::fs::read_to_string(&p).unwrap();
        assert!(text.starts_with("#Insight Transform File V1.0"));
        assert!(text.contains("Transform: AffineTransform_double_3_3"));
        let params: Vec<f64> = text.lines().find_map(|l| l.strip_prefix("Parameters: ")).unwrap().split_whitespace().map(|v| v.parse().unwrap()).collect();
        assert_eq!(params, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, -10.0, -20.0, 30.0]);
        // A rotation about z keeps its sense in LPS (x and y both flip).
        write_itk_transform(&p, &Transform { matrix: DMat4::from_rotation_z(0.3) }).unwrap();
        let text = std::fs::read_to_string(&p).unwrap();
        let params: Vec<f64> = text.lines().find_map(|l| l.strip_prefix("Parameters: ")).unwrap().split_whitespace().map(|v| v.parse().unwrap()).collect();
        assert!((params[1] - (-(0.3f64).sin())).abs() < 1e-12 && (params[3] - 0.3f64.sin()).abs() < 1e-12);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Timing on a 256³ pair: `cargo test -p mm-image --release -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn timing_256() {
        let fixed = image([256, 256, 256], 1.0, DVec3::splat(-127.5), |p| phantom(p, T1));
        let g = rigid(6.0, DVec3::new(0.2, 1.0, 0.3), DVec3::new(7.0, -5.0, 3.0));
        for (metric, t) in [(Metric::Correlation, T1), (Metric::MutualInformation, [700.0, 200.0, 450.0, 950.0])] {
            let moving = image([256, 256, 256], 1.0, DVec3::splat(-127.5), move |q| phantom(g.transform_point3(q), t));
            let start = std::time::Instant::now();
            let r = register(&fixed, &moving, &RegisterParams { metric, ..Default::default() }, no_progress()).unwrap();
            eprintln!("256³ {metric:?}: {:.2?}, error {:.3} mm", start.elapsed(), error_mm(g * r.transform.matrix));
        }
    }
}
