//! N4 bias field correction: removes the smooth shading MR receive coils leave across an image
//! (Tustison et al., "N4ITK: Improved N3 bias correction", IEEE TMI 29(6), 2010; the method of
//! ITK's N4BiasFieldCorrectionImageFilter and ANTs, written here from the paper).
//!
//! The shading is a smooth multiplicative field, so in the log domain it adds. Each iteration:
//! 1. sharpens the histogram of the current corrected log image: the histogram is taken as the
//!    true one blurred by a Gaussian (the bias), deconvolved with a Wiener filter, and every
//!    voxel moves to its expected true value (Sled's N3 step);
//! 2. fits a smooth cubic B-spline field to what the sharpening moved, and adds it to the bias
//!    estimate.
//!
//! Levels double the B-spline control points (finer shading); a level ends early once an
//! iteration changes the field by less than the convergence threshold. Fitting runs on a copy
//! shrunk by block averaging; the field is then enlarged smoothly to the full grid.

use crate::fft::fft;
use anyhow::{bail, Result};
use mm_segment::{ops, Connectivity, Mask};
use mm_volume::load::LoadedVolume;
use rayon::prelude::*;

#[derive(Clone, Debug, PartialEq)]
pub struct N4Params {
    /// Voxels averaged per axis before fitting (speed; the field is smooth anyway).
    pub shrink: usize,
    /// Iterations at each level (the B-spline mesh doubles from one level to the next).
    pub iterations: Vec<usize>,
    /// A level stops once the field changes by less than this (coefficient of variation).
    pub convergence: f64,
    /// B-spline control point spacing at the first level (mm).
    pub spline_distance_mm: f64,
    /// Histogram bins for sharpening.
    pub bins: usize,
    /// Assumed width of the bias's blur of the log histogram (full width at half maximum).
    pub fwhm: f64,
    /// Wiener filter noise.
    pub wiener_noise: f64,
}

impl Default for N4Params {
    fn default() -> Self {
        Self { shrink: 4, iterations: vec![50, 50, 50, 50], convergence: 0.001, spline_distance_mm: 200.0, bins: 200, fwhm: 0.15, wiener_noise: 0.01 }
    }
}

pub struct N4Output {
    /// The image divided by the bias field.
    pub corrected: Vec<f32>,
    /// The multiplicative bias field (geometric mean 1 over the mask).
    pub bias: Vec<f32>,
    /// The voxels the field was fitted to.
    pub mask: Vec<u8>,
    /// Iterations run at each level.
    pub iterations: Vec<usize>,
}

/// The object in an image (head, phantom): above half of Otsu's threshold (so shaded edges
/// stay in), the largest connected part with its holes filled.
pub fn auto_mask(vol: &LoadedVolume) -> Vec<u8> {
    let step = (vol.values.len() / 2_000_000).max(1);
    // The whole image: the background (noise, or zeros when skull-stripped) is one class.
    let sample: Vec<f32> = vol.values.iter().step_by(step).copied().filter(|v| v.is_finite()).collect();
    let Some(t) = ops::otsu(&sample) else { return vol.values.iter().map(|&v| (v > 0.0) as u8).collect() };
    let t = t * 0.5;
    let mut m = Mask { dims: vol.dims, data: vol.values.par_iter().map(|&v| (v > t) as u8).collect() };
    ops::keep_largest(&mut m, Connectivity::Six);
    ops::fill_holes(&mut m);
    m.data
}

/// B-spline basis along one axis of the shrunk grid: per sample, its first control point and
/// the four cubic weights.
struct Axis {
    controls: usize,
    first: Vec<usize>,
    w: Vec<[f64; 4]>,
}

impl Axis {
    fn new(n: usize, spans: usize) -> Self {
        let spans = spans.max(1);
        let (mut first, mut w) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for i in 0..n {
            let t = if n > 1 { i as f64 * spans as f64 / (n - 1) as f64 } else { 0.0 };
            let j = (t.floor() as usize).min(spans - 1);
            let f = t - j as f64;
            let g = 1.0 - f;
            first.push(j);
            w.push([g * g * g / 6.0, (3.0 * f * f * f - 6.0 * f * f + 4.0) / 6.0, (-3.0 * f * f * f + 3.0 * f * f + 3.0 * f + 1.0) / 6.0, f * f * f / 6.0]);
        }
        Self { controls: spans + 3, first, w }
    }
}

/// `Σ_x data(x) · Π_axes factor(weights)` for every control point, one axis at a time.
fn contract(data: &[f64], n: [usize; 3], ax: &[Axis; 3], factor: impl Fn(&[f64; 4]) -> [f64; 4]) -> Vec<f64> {
    let [nx, ny, nz] = n;
    let [cx, cy, cz] = [ax[0].controls, ax[1].controls, ax[2].controls];
    let fx: Vec<[f64; 4]> = ax[0].w.iter().map(&factor).collect();
    let fy: Vec<[f64; 4]> = ax[1].w.iter().map(&factor).collect();
    let fz: Vec<[f64; 4]> = ax[2].w.iter().map(&factor).collect();
    // Along x: (cx, ny, nz).
    let mut t1 = vec![0.0; cx * ny * nz];
    t1.par_chunks_mut(cx * ny).enumerate().for_each(|(z, out)| {
        for y in 0..ny {
            let row = &data[(z * ny + y) * nx..][..nx];
            let o = &mut out[y * cx..][..cx];
            for (x, &d) in row.iter().enumerate() {
                if d != 0.0 {
                    for a in 0..4 {
                        o[ax[0].first[x] + a] += d * fx[x][a];
                    }
                }
            }
        }
    });
    // Along y: (cx, cy, nz).
    let mut t2 = vec![0.0; cx * cy * nz];
    t2.par_chunks_mut(cx * cy).enumerate().for_each(|(z, out)| {
        for y in 0..ny {
            let src = &t1[(z * ny + y) * cx..][..cx];
            for a in 0..4 {
                let o = &mut out[(ax[1].first[y] + a) * cx..][..cx];
                for (kx, &s) in src.iter().enumerate() {
                    o[kx] += s * fy[y][a];
                }
            }
        }
    });
    // Along z: (cx, cy, cz).
    let mut t3 = vec![0.0; cx * cy * cz];
    for z in 0..nz {
        let src = &t2[z * cx * cy..][..cx * cy];
        for a in 0..4 {
            let o = &mut t3[(ax[2].first[z] + a) * cx * cy..][..cx * cy];
            for (k, &s) in src.iter().enumerate() {
                o[k] += s * fz[z][a];
            }
        }
    }
    t3
}

/// The B-spline with control values `ctrl` at every sample of the grid.
fn evaluate(ctrl: &[f64], n: [usize; 3], ax: &[Axis; 3]) -> Vec<f64> {
    let [nx, ny, nz] = n;
    let [cx, cy, _] = [ax[0].controls, ax[1].controls, ax[2].controls];
    // Along z: (cx, cy, nz).
    let mut e1 = vec![0.0; cx * cy * nz];
    e1.par_chunks_mut(cx * cy).enumerate().for_each(|(z, out)| {
        for a in 0..4 {
            let src = &ctrl[(ax[2].first[z] + a) * cx * cy..][..cx * cy];
            let w = ax[2].w[z][a];
            for (o, &s) in out.iter_mut().zip(src) {
                *o += s * w;
            }
        }
    });
    // Along y: (cx, ny, nz).
    let mut e2 = vec![0.0; cx * ny * nz];
    e2.par_chunks_mut(cx * ny).enumerate().for_each(|(z, out)| {
        for y in 0..ny {
            for a in 0..4 {
                let src = &e1[(z * cy + ax[1].first[y] + a) * cx..][..cx];
                let w = ax[1].w[y][a];
                for (o, &s) in out[y * cx..][..cx].iter_mut().zip(src) {
                    *o += s * w;
                }
            }
        }
    });
    // Along x: (nx, ny, nz).
    let mut f = vec![0.0; nx * ny * nz];
    f.par_chunks_mut(nx * ny).enumerate().for_each(|(z, out)| {
        for y in 0..ny {
            let src = &e2[(z * ny + y) * cx..][..cx];
            for x in 0..nx {
                let w = &ax[0].w[x];
                let j = ax[0].first[x];
                out[y * nx + x] = src[j] * w[0] + src[j + 1] * w[1] + src[j + 2] * w[2] + src[j + 3] * w[3];
            }
        }
    });
    f
}

/// A smooth B-spline approximation of `r` where `m` is set (one level of multilevel B-spline
/// approximation, Lee, Wolberg & Shin 1997), evaluated on the whole grid.
#[cfg(test)]
fn fit(r: &[f64], m: &[f64], n: [usize; 3], ax: &[Axis; 3]) -> Vec<f64> {
    let rm: Vec<f64> = r.iter().zip(m).map(|(a, b)| a * b).collect();
    fit_weighted(&rm, &fit_denominator(m, n, ax), n, ax)
}

/// The denominator of [`fit`]'s control values: it depends only on the mask and the mesh, so
/// a level makes it once for all its iterations.
fn fit_denominator(m: &[f64], n: [usize; 3], ax: &[Axis; 3]) -> Vec<f64> {
    contract(m, n, ax, |w| w.map(|x| x * x))
}

/// [`fit`] from `rm` (the residual times the mask) and the denominator.
fn fit_weighted(rm: &[f64], den: &[f64], n: [usize; 3], ax: &[Axis; 3]) -> Vec<f64> {
    let num = contract(rm, n, ax, |w| {
        let s: f64 = w.iter().map(|x| x * x).sum();
        w.map(|x| x * x * x / s)
    });
    let ctrl: Vec<f64> = num.iter().zip(den).map(|(&a, &b)| if b > 1e-12 { a / b } else { 0.0 }).collect();
    evaluate(&ctrl, n, ax)
}

/// The values a sharpened histogram moves its values to: linear between the expected true
/// values of its bins (no bins: values stay).
struct Sharpening {
    lo: f64,
    slope: f64,
    expect: Vec<f64>,
}

impl Sharpening {
    fn at(&self, x: f64) -> f64 {
        let bins = self.expect.len();
        if bins == 0 {
            return x;
        }
        let c = ((x - self.lo) / self.slope).clamp(0.0, (bins - 1) as f64);
        let i = (c.floor() as usize).min(bins - 2);
        let f = c - i as f64;
        self.expect[i] * (1.0 - f) + self.expect[i + 1] * f
    }
}

/// Sharpen the histogram of `u` (values where `m` is set, from `lo` to `hi`): each value
/// moves to its expected value under the deconvolved histogram. The histogram is summed in
/// voxel order.
fn sharpen(u: &[f64], m: &[f64], (lo, hi): (f64, f64), p: &N4Params) -> Sharpening {
    let bins = p.bins.max(2);
    if hi - lo <= 1e-9 {
        return Sharpening { lo, slope: 1.0, expect: Vec::new() };
    }
    let slope = (hi - lo) / (bins - 1) as f64;
    let mut hist = vec![0.0; bins];
    for (&x, &w) in u.iter().zip(m) {
        if w > 0.0 {
            let c = (x - lo) / slope;
            let i = (c.floor() as usize).min(bins - 1);
            let f = c - i as f64;
            hist[i] += w * (1.0 - f);
            if i + 1 < bins {
                hist[i + 1] += w * f;
            }
        }
    }
    let pad = (bins.next_power_of_two()) * 2;
    let offset = (pad - bins) / 2;
    // The Gaussian blur (in bins), wrapped around 0.
    let fwhm = p.fwhm / slope;
    let exp_factor = 4.0 * std::f64::consts::LN_2 / (fwhm * fwhm);
    let scale = 2.0 * (std::f64::consts::LN_2 / std::f64::consts::PI).sqrt() / fwhm;
    let mut fr = vec![0.0; pad];
    fr[0] = scale;
    for k in 1..=pad / 2 {
        let g = scale * (-((k * k) as f64) * exp_factor).exp();
        fr[k] = g;
        fr[pad - k] = g;
    }
    let mut fi = vec![0.0; pad];
    fft(&mut fr, &mut fi, false);
    // Wiener deconvolution of the histogram.
    let (mut vr, mut vi) = (vec![0.0; pad], vec![0.0; pad]);
    vr[offset..offset + bins].copy_from_slice(&hist);
    fft(&mut vr, &mut vi, false);
    for k in 0..pad {
        let d = fr[k] * fr[k] + fi[k] * fi[k] + p.wiener_noise;
        // V · conj(F) / d
        let (a, b) = (vr[k] * fr[k] + vi[k] * fi[k], vi[k] * fr[k] - vr[k] * fi[k]);
        (vr[k], vi[k]) = (a / d, b / d);
    }
    fft(&mut vr, &mut vi, true);
    let est: Vec<f64> = vr.iter().map(|&x| x.max(0.0)).collect();
    // E[true | observed] = (x · U) ⊛ F / U ⊛ F.
    let convolve = |v: &[f64]| -> Vec<f64> {
        let (mut r, mut i) = (v.to_vec(), vec![0.0; pad]);
        fft(&mut r, &mut i, false);
        for k in 0..pad {
            (r[k], i[k]) = (r[k] * fr[k] - i[k] * fi[k], r[k] * fi[k] + i[k] * fr[k]);
        }
        fft(&mut r, &mut i, true);
        r
    };
    let weighted: Vec<f64> = est.iter().enumerate().map(|(k, &e)| e * (lo + (k as f64 - offset as f64) * slope)).collect();
    let num = convolve(&weighted);
    let den = convolve(&est);
    let expect: Vec<f64> = (0..bins).map(|b| {
        let (n, d) = (num[b + offset], den[b + offset]);
        if d.abs() > 1e-12 { n / d } else { lo + b as f64 * slope }
    }).collect();
    Sharpening { lo, slope, expect }
}

/// Correct `vol` for intensity shading. `mask`: the voxels to fit to (`None`: [`auto_mask`]).
/// `progress` gets the fraction done; returning `false` cancels.
pub fn n4(vol: &LoadedVolume, mask: Option<&[u8]>, p: &N4Params, progress: &(dyn Fn(f32) -> bool + Sync)) -> Result<N4Output> {
    let dims = vol.dims;
    // The automatic mask, made here, goes into the output as it is.
    let auto = match mask {
        Some(m) if m.len() == vol.values.len() => None,
        Some(_) => bail!("the mask does not match the image"),
        None => Some(auto_mask(vol)),
    };
    let given = mask;
    let mask: &[u8] = auto.as_deref().or(given).unwrap_or_default();
    let s = p.shrink.max(1);
    let sd = dims.map(|d| d.div_ceil(s));
    let n = sd[0] * sd[1] * sd[2];
    // Block means of the positive voxels in the mask; the log image is fitted where they exist.
    let mut v = vec![0.0f64; n];
    let mut m = vec![0.0f64; n];
    v.par_chunks_mut(sd[0] * sd[1]).zip(m.par_chunks_mut(sd[0] * sd[1])).enumerate().for_each(|(zs, (vp, mp))| {
        for ys in 0..sd[1] {
            for xs in 0..sd[0] {
                let (mut sum, mut count) = (0.0, 0usize);
                for k in zs * s..((zs + 1) * s).min(dims[2]) {
                    for j in ys * s..((ys + 1) * s).min(dims[1]) {
                        for i in xs * s..((xs + 1) * s).min(dims[0]) {
                            let q = vol.index(i, j, k);
                            let x = vol.values[q];
                            if mask[q] != 0 && x > 0.0 && x.is_finite() {
                                sum += x as f64;
                                count += 1;
                            }
                        }
                    }
                }
                if count > 0 {
                    vp[ys * sd[0] + xs] = (sum / count as f64).ln();
                    mp[ys * sd[0] + xs] = 1.0;
                }
            }
        }
    });
    let fitted: usize = m.iter().filter(|&&w| w > 0.0).count();
    if fitted < 64 {
        bail!("too few voxels in the mask to estimate the bias field");
    }
    let spacing = vol.geometry.spacing;
    let spans0: [usize; 3] = [0, 1, 2].map(|a| ((sd[a] as f64 * spacing[a] * s as f64) / p.spline_distance_mm.max(1.0)).round().max(1.0) as usize);
    let total: usize = p.iterations.iter().sum::<usize>().max(1);
    let mut done = 0usize;
    let mut log_bias = vec![0.0f64; n];
    let mut runs = Vec::new();
    // Per iteration: the corrected log image, the residual of its sharpening times the mask,
    // and the field's change as a factor (where the mask is set).
    let (mut u, mut rm, mut change) = (vec![0.0f64; n], vec![0.0f64; n], vec![0.0f64; n]);
    for (level, &iters) in p.iterations.iter().enumerate() {
        let spans = spans0.map(|x| x << level);
        let ax = [Axis::new(sd[0], spans[0]), Axis::new(sd[1], spans[1]), Axis::new(sd[2], spans[2])];
        let den = if iters > 0 { fit_denominator(&m, sd, &ax) } else { Vec::new() };
        let mut run = 0;
        for _ in 0..iters {
            // The corrected log image and its range over the mask (minimum and maximum are
            // exact, so the order they are found in does not matter).
            let range = u
                .par_iter_mut()
                .zip(&v)
                .zip(&log_bias)
                .zip(&m)
                .map(|(((o, a), b), &w)| {
                    *o = a - b;
                    if w > 0.0 { (*o, *o) } else { (f64::INFINITY, f64::NEG_INFINITY) }
                })
                .reduce(|| (f64::INFINITY, f64::NEG_INFINITY), |(a, b), (c, d)| (a.min(c), b.max(d)));
            let sharp = sharpen(&u, &m, range, p);
            rm.par_iter_mut().zip(&u).zip(&m).for_each(|((o, &a), &w)| *o = (if w > 0.0 { a - sharp.at(a) } else { 0.0 }) * w);
            let delta = fit_weighted(&rm, &den, sd, &ax);
            change.par_iter_mut().zip(&mut log_bias).zip(&delta).zip(&m).for_each(|(((c, b), &d), &w)| {
                if w > 0.0 {
                    *c = d.exp();
                }
                *b += d;
            });
            // Coefficient of variation of the field's change over the mask (summed in order).
            let masked = || change.iter().zip(&m).filter(|(_, &w)| w > 0.0).map(|(c, _)| c);
            let mean = masked().sum::<f64>() / fitted as f64;
            let var = masked().map(|c| (c - mean) * (c - mean)).sum::<f64>() / fitted as f64;
            run += 1;
            done += 1;
            if !progress(0.9 * done as f32 / total as f32) {
                bail!("cancelled");
            }
            if var.sqrt() / mean < p.convergence {
                break;
            }
        }
        done += iters - run;
        runs.push(run);
    }
    // Geometric mean 1 over the mask, so intensities keep their scale.
    let mean = log_bias.iter().zip(&m).filter(|(_, &w)| w > 0.0).map(|(b, _)| b).sum::<f64>() / fitted as f64;
    log_bias.iter_mut().for_each(|b| *b -= mean);
    // Enlarge: shrunk sample (i, j, k) is the centre of its block.
    let half = (s as f64 - 1.0) * 0.5;
    let coord = |i: usize, a: usize| ((i as f64 - half) / s as f64).clamp(0.0, (sd[a] - 1) as f64);
    let at = |x: usize, y: usize, z: usize| log_bias[(z * sd[1] + y) * sd[0] + x];
    // The x samples and weight of every column, the same for every row.
    let columns: Vec<(usize, usize, f64)> = (0..dims[0])
        .map(|i| {
            let tx = coord(i, 0);
            let x0 = (tx.floor() as usize).min(sd[0].saturating_sub(2));
            (x0, (x0 + 1).min(sd[0] - 1), tx - x0 as f64)
        })
        .collect();
    let mut bias = vec![0.0f32; vol.values.len()];
    bias.par_chunks_mut((dims[0] * dims[1]).max(1)).enumerate().for_each(|(k, plane)| {
        let tz = coord(k, 2);
        let z0 = (tz.floor() as usize).min(sd[2].saturating_sub(2));
        let z1 = (z0 + 1).min(sd[2] - 1);
        let fz = tz - z0 as f64;
        for j in 0..dims[1] {
            let ty = coord(j, 1);
            let y0 = (ty.floor() as usize).min(sd[1].saturating_sub(2));
            let y1 = (y0 + 1).min(sd[1] - 1);
            let fy = ty - y0 as f64;
            for (i, &(x0, x1, fx)) in columns.iter().enumerate() {
                let l = |a: f64, b: f64, t: f64| a + (b - a) * t;
                let c00 = l(at(x0, y0, z0), at(x1, y0, z0), fx);
                let c10 = l(at(x0, y1, z0), at(x1, y1, z0), fx);
                let c01 = l(at(x0, y0, z1), at(x1, y0, z1), fx);
                let c11 = l(at(x0, y1, z1), at(x1, y1, z1), fx);
                plane[j * dims[0] + i] = l(l(c00, c10, fy), l(c01, c11, fy), fz).exp() as f32;
            }
        }
    });
    progress(0.97);
    let corrected = vol.values.par_iter().zip(&bias).map(|(&x, &b)| x / b).collect();
    let mask = match auto {
        Some(m) => m,
        None => given.unwrap_or_default().to_vec(),
    };
    Ok(N4Output { corrected, bias, mask, iterations: runs })
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec3;
    use mm_volume::geometry::Geometry;

    /// A head-like phantom: three nested tissues (300, 600, 900) under a smooth shading of
    /// about ±40 %, with a little noise. Returns the image, the true bias and the tissue labels.
    fn phantom() -> (LoadedVolume, Vec<f64>, Vec<u8>) {
        let dims = [56, 64, 48];
        let spacing = 2.0;
        let c = DVec3::new(28.0, 32.0, 24.0) * spacing;
        let (mut values, mut bias, mut labels) = (Vec::new(), Vec::new(), Vec::new());
        let mut seed = 12345u64;
        for k in 0..dims[2] {
            for j in 0..dims[1] {
                for i in 0..dims[0] {
                    let p = DVec3::new(i as f64, j as f64, k as f64) * spacing - c;
                    let r = (p / DVec3::new(50.0, 58.0, 44.0)).length();
                    let (tissue, label) = if r < 0.45 { (900.0, 3) } else if r < 0.75 { (600.0, 2) } else if r < 1.0 { (300.0, 1) } else { (0.0, 0) };
                    let b = (0.30 * p.x / 50.0 - 0.20 * p.y / 60.0 + 0.25 * (p.z / 45.0).powi(2) - 0.1).exp();
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    let noise = 1.0 + 0.02 * (((seed >> 33) as f64 / (1u64 << 31) as f64) - 0.5);
                    values.push((tissue * b * noise) as f32);
                    bias.push(b);
                    labels.push(label);
                }
            }
        }
        let g = Geometry::from_axes([dims[0], dims[1], dims[2], 1], [spacing; 3], [DVec3::X, DVec3::Y, DVec3::Z], DVec3::ZERO);
        (LoadedVolume::from_values("phantom", dims, values, g), bias, labels)
    }

    /// Coefficient of variation of `v` within tissue `label`.
    fn cv(v: &[f32], labels: &[u8], label: u8) -> f64 {
        let x: Vec<f64> = v.iter().zip(labels).filter(|(_, &l)| l == label).map(|(&a, _)| a as f64).collect();
        let mean = x.iter().sum::<f64>() / x.len() as f64;
        (x.iter().map(|a| (a - mean) * (a - mean)).sum::<f64>() / x.len() as f64).sqrt() / mean
    }

    #[test]
    fn removes_smooth_shading() {
        let (vol, truth, labels) = phantom();
        let p = N4Params { shrink: 2, iterations: vec![50, 50, 50], spline_distance_mm: 120.0, ..Default::default() };
        let out = n4(&vol, None, &p, &|_| true).unwrap();
        // The automatic mask finds the phantom.
        let inside = labels.iter().filter(|&&l| l > 0).count();
        let masked = out.mask.iter().filter(|&&m| m > 0).count();
        assert!((masked as f64 / inside as f64 - 1.0).abs() < 0.1, "{masked} vs {inside}");
        for t in 1..=3 {
            let (before, after) = (cv(&vol.values, &labels, t), cv(&out.corrected, &labels, t));
            assert!(after < before * 0.4 && after < 0.06, "tissue {t}: CV {before:.3} → {after:.3}");
        }
        // The field matches the true one (up to scale) inside the phantom.
        let pairs: Vec<(f64, f64)> = out.bias.iter().zip(&truth).zip(&labels).filter(|(_, &l)| l > 0).map(|((&b, &t), _)| ((b as f64).ln(), t.ln())).collect();
        let n = pairs.len() as f64;
        let (ma, mb) = pairs.iter().fold((0.0, 0.0), |(a, b), (x, y)| (a + x / n, b + y / n));
        let (sab, saa, sbb) = pairs.iter().fold((0.0, 0.0, 0.0), |(c, d, e), (x, y)| (c + (x - ma) * (y - mb), d + (x - ma) * (x - ma), e + (y - mb) * (y - mb)));
        let corr = sab / (saa * sbb).sqrt();
        assert!(corr > 0.95, "correlation with the true field {corr:.3}");
        assert!(out.iterations.iter().all(|&i| i >= 1));
        // Cancelling stops it.
        assert!(n4(&vol, None, &p, &|_| false).is_err());
    }

    #[test]
    fn the_result_does_not_depend_on_the_threads() {
        let (vol, _, _) = phantom();
        let p = N4Params { shrink: 2, iterations: vec![20, 20], spline_distance_mm: 120.0, convergence: 0.0, ..Default::default() };
        let many = n4(&vol, None, &p, &|_| true).unwrap();
        let one = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap().install(|| n4(&vol, None, &p, &|_| true).unwrap());
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&many.corrected), bits(&one.corrected));
        assert_eq!(bits(&many.bias), bits(&one.bias));
        assert_eq!((many.mask, many.iterations), (one.mask, one.iterations));
    }

    #[test]
    fn bspline_fit_reproduces_a_smooth_field() {
        let n = [20, 16, 12];
        let ax = [Axis::new(n[0], 3), Axis::new(n[1], 3), Axis::new(n[2], 2)];
        let r: Vec<f64> = (0..n[0] * n[1] * n[2]).map(|q| {
            let (x, y, z) = ((q % 20) as f64 / 19.0, ((q / 20) % 16) as f64 / 15.0, (q / 320) as f64 / 11.0);
            0.5 * x - 0.3 * y * y + 0.2 * z
        }).collect();
        let m = vec![1.0; r.len()];
        // Fitting the residual again and again (as N4's iterations do) converges to the field.
        let mut f = vec![0.0; r.len()];
        let mut errors = Vec::new();
        for _ in 0..200 {
            let res: Vec<f64> = r.iter().zip(&f).map(|(a, b)| a - b).collect();
            let d = fit(&res, &m, n, &ax);
            f.iter_mut().zip(d).for_each(|(a, b)| *a += b);
            errors.push(r.iter().zip(&f).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max));
        }
        assert!(errors.windows(2).all(|w| w[1] <= w[0] + 1e-12), "the error never grows");
        assert!(errors[199] < 0.02 && errors[199] < errors[0] / 10.0, "max error {}", errors[199]);
    }
}
