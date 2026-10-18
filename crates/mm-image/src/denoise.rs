//! Noise reduction: median, Perona–Malik gradient anisotropic diffusion, curvature
//! anisotropic diffusion (the modified curvature diffusion equation of Whitaker & Xue, as in
//! ITK), and non-local means (Buades et al. 2005), with a robust noise estimate.
//!
//! The diffusion filters follow ITK's conventions: the edge threshold is `conductance` times
//! the image's root-mean-square gradient magnitude (recomputed every iteration), and the time
//! step is 1/16 for unit voxels, scaled for the actual spacing so the explicit scheme stays
//! stable. Every filter treats the grid's edges as reflecting (no flux across them).

use crate::image::{check, Progress};
use mm_volume::load::LoadedVolume;
use rayon::prelude::*;

/// Index helper for a grid.
#[derive(Clone, Copy)]
struct Grid {
    n: [usize; 3],
}

impl Grid {
    fn idx(&self, i: usize, j: usize, k: usize) -> usize {
        (k * self.n[1] + j) * self.n[0] + i
    }

    /// Index of (i, j, k) moved by `d` along `axis`, clamped to the grid.
    fn step(&self, ijk: [usize; 3], axis: usize, d: i64) -> usize {
        let mut c = ijk;
        c[axis] = (c[axis] as i64 + d).clamp(0, self.n[axis] as i64 - 1) as usize;
        self.idx(c[0], c[1], c[2])
    }
}

/// Median over a cube of (2r + 1)³ voxels (clamped at the edges).
pub fn median(vol: &LoadedVolume, radius: usize, progress: Progress) -> anyhow::Result<Vec<f32>> {
    if radius == 0 {
        return Ok(vol.values.clone());
    }
    let g = Grid { n: vol.dims };
    let [nx, ny, nz] = vol.dims;
    let r = radius as i64;
    let done = std::sync::atomic::AtomicUsize::new(0);
    let cancelled = std::sync::atomic::AtomicBool::new(false);
    let mut out = vec![0.0f32; vol.values.len()];
    out.par_chunks_mut(nx * ny).enumerate().for_each(|(k, plane)| {
        if cancelled.load(std::sync::atomic::Ordering::Relaxed) {
            return;
        }
        let mut buf = Vec::with_capacity(((2 * r + 1) as usize).pow(3));
        for j in 0..ny {
            for i in 0..nx {
                buf.clear();
                for dz in -r..=r {
                    let kk = (k as i64 + dz).clamp(0, nz as i64 - 1) as usize;
                    for dy in -r..=r {
                        let jj = (j as i64 + dy).clamp(0, ny as i64 - 1) as usize;
                        let row = g.idx(0, jj, kk);
                        for dx in -r..=r {
                            let ii = (i as i64 + dx).clamp(0, nx as i64 - 1) as usize;
                            buf.push(vol.values[row + ii]);
                        }
                    }
                }
                let mid = buf.len() / 2;
                let (_, m, _) = buf.select_nth_unstable_by(mid, f32::total_cmp);
                plane[j * nx + i] = *m;
            }
        }
        let d = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if !progress(d as f32 / nz as f32) {
            cancelled.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    });
    check(progress, 1.0).and_then(|_| if cancelled.into_inner() { anyhow::bail!("cancelled") } else { Ok(out) })
}

/// Stable explicit time step for diffusion on voxels of this spacing (1/16 for unit voxels, as
/// ITK uses).
fn time_step(spacing: [f64; 3]) -> f64 {
    3.0 / (16.0 * spacing.iter().map(|h| 1.0 / (h * h)).sum::<f64>())
}

/// Mean squared gradient magnitude (central differences, mm).
fn mean_gradient_sq(v: &[f32], g: Grid, h: [f64; 3]) -> f64 {
    let [nx, ny, nz] = g.n;
    let total: f64 = (0..nz)
        .into_par_iter()
        .map(|k| {
            let mut s = 0.0;
            for j in 0..ny {
                for i in 0..nx {
                    let c = [i, j, k];
                    for a in 0..3 {
                        let d = (v[g.step(c, a, 1)] - v[g.step(c, a, -1)]) as f64 / (2.0 * h[a]);
                        s += d * d;
                    }
                }
            }
            s
        })
        .sum();
    total / v.len().max(1) as f64
}

/// Perona–Malik gradient anisotropic diffusion (edge-preserving smoothing). Fluxes across
/// differences much larger than `conductance` × the image's RMS gradient magnitude are cut
/// (conduction `exp(-(d / K)²)` with `K² = 2 · conductance² · ⟨|∇I|²⟩`, as in ITK); typical
/// `conductance` 0.5–3, `iterations` 5–20.
pub fn gradient_diffusion(vol: &LoadedVolume, iterations: usize, conductance: f64, progress: Progress) -> anyhow::Result<Vec<f32>> {
    let g = Grid { n: vol.dims };
    let h = vol.geometry.spacing;
    let dt = time_step(h);
    let [nx, ny, _] = g.n;
    let mut v = vol.values.clone();
    for it in 0..iterations {
        let k2 = 2.0 * conductance * conductance * mean_gradient_sq(&v, g, h);
        if k2 <= 0.0 {
            break;
        }
        let mut next = vec![0.0f32; v.len()];
        next.par_chunks_mut(nx * ny).enumerate().for_each(|(k, plane)| {
            for j in 0..ny {
                for i in 0..nx {
                    let c = [i, j, k];
                    let x = v[g.idx(i, j, k)] as f64;
                    let mut change = 0.0;
                    for a in 0..3 {
                        let fwd = (v[g.step(c, a, 1)] as f64 - x) / h[a];
                        let bwd = (x - v[g.step(c, a, -1)] as f64) / h[a];
                        let cf = (-fwd * fwd / k2).exp();
                        let cb = (-bwd * bwd / k2).exp();
                        change += (cf * fwd - cb * bwd) / h[a];
                    }
                    plane[j * nx + i] = (x + dt * change) as f32;
                }
            }
        });
        v = next;
        check(progress, (it + 1) as f32 / iterations.max(1) as f32)?;
    }
    Ok(v)
}

/// Curvature anisotropic diffusion: the modified curvature diffusion equation
/// `I_t = |∇I| ∇·(c(|∇I|) ∇I / |∇I|)` (Whitaker & Xue 2001, ITK's
/// CurvatureAnisotropicDiffusionImageFilter), with the same conductance as
/// [`gradient_diffusion`]. It smooths along edges rather than only up to them, leaving fewer
/// blotches; typical `conductance` 0.5–3, `iterations` 5–20.
pub fn curvature_diffusion(vol: &LoadedVolume, iterations: usize, conductance: f64, progress: Progress) -> anyhow::Result<Vec<f32>> {
    let g = Grid { n: vol.dims };
    let h = vol.geometry.spacing;
    let dt = time_step(h);
    let [nx, ny, _] = g.n;
    let mut v = vol.values.clone();
    for it in 0..iterations {
        let k2 = 2.0 * conductance * conductance * mean_gradient_sq(&v, g, h);
        if k2 <= 0.0 {
            break;
        }
        let mut next = vec![0.0f32; v.len()];
        next.par_chunks_mut(nx * ny).enumerate().for_each(|(k, plane)| {
            let val = |q: usize| v[q] as f64;
            for j in 0..ny {
                for i in 0..nx {
                    let c = [i, j, k];
                    let x = val(g.idx(i, j, k));
                    let mut speed = 0.0;
                    let mut fwd = [0.0; 3];
                    let mut bwd = [0.0; 3];
                    for a in 0..3 {
                        let pa = g.step(c, a, 1);
                        let ma = g.step(c, a, -1);
                        fwd[a] = (val(pa) - x) / h[a];
                        bwd[a] = (x - val(ma)) / h[a];
                        // Gradient at the half-way points along `a`: the difference along `a`,
                        // and along the other axes the average of the central differences at
                        // both ends.
                        let (mut gf, mut gb) = (fwd[a] * fwd[a], bwd[a] * bwd[a]);
                        let mut cpa = c;
                        cpa[a] = (c[a] + 1).min(g.n[a] - 1);
                        let mut cma = c;
                        cma[a] = c[a].saturating_sub(1);
                        for b in (0..3).filter(|&b| b != a) {
                            let here = (val(g.step(c, b, 1)) - val(g.step(c, b, -1))) / (2.0 * h[b]);
                            let ahead = (val(g.step(cpa, b, 1)) - val(g.step(cpa, b, -1))) / (2.0 * h[b]);
                            let behind = (val(g.step(cma, b, 1)) - val(g.step(cma, b, -1))) / (2.0 * h[b]);
                            gf += 0.25 * (here + ahead).powi(2);
                            gb += 0.25 * (here + behind).powi(2);
                        }
                        let (mf, mb) = (gf.sqrt(), gb.sqrt());
                        let ff = if mf > 1e-12 { (-gf / k2).exp() * fwd[a] / mf } else { 0.0 };
                        let fb = if mb > 1e-12 { (-gb / k2).exp() * bwd[a] / mb } else { 0.0 };
                        speed += (ff - fb) / h[a];
                    }
                    // |∇I| by upwind differences in the direction of the motion.
                    let mut mag = 0.0;
                    for a in 0..3 {
                        mag += if speed > 0.0 { bwd[a].min(0.0).powi(2) + fwd[a].max(0.0).powi(2) } else { bwd[a].max(0.0).powi(2) + fwd[a].min(0.0).powi(2) };
                    }
                    plane[j * nx + i] = (x + dt * mag.sqrt() * speed) as f32;
                }
            }
        });
        v = next;
        check(progress, (it + 1) as f32 / iterations.max(1) as f32)?;
    }
    Ok(v)
}

/// Noise standard deviation: the robust spread (median absolute deviation) of the response to
/// Immerkær's noise-only mask, slice by slice in-plane, ignoring flat regions (such as the
/// zeros around a skull-stripped brain). Exact for Gaussian noise on a smooth image.
pub fn noise_sigma(vol: &LoadedVolume) -> f64 {
    let [nx, ny, nz] = vol.dims;
    if nx < 3 || ny < 3 {
        return 0.0;
    }
    // The mask [1 -2 1; -2 4 -2; 1 -2 1] cancels planes and has norm 6.
    let step = ((nx * ny * nz) / 2_000_000).max(1);
    let mut r: Vec<f64> = (0..nz)
        .into_par_iter()
        .flat_map_iter(|k| {
            let v = &vol.values[k * nx * ny..(k + 1) * nx * ny];
            (1..ny - 1).flat_map(move |j| (1..nx - 1).map(move |i| (i, j))).step_by(step).filter_map(move |(i, j)| {
                let at = |di: i64, dj: i64| v[((j as i64 + dj) as usize) * nx + (i as i64 + di) as usize] as f64;
                let n = [at(-1, -1), at(0, -1), at(1, -1), at(-1, 0), at(0, 0), at(1, 0), at(-1, 1), at(0, 1), at(1, 1)];
                if n.iter().all(|&x| x == n[0]) {
                    return None;
                }
                Some(n[0] - 2.0 * n[1] + n[2] - 2.0 * n[3] + 4.0 * n[4] - 2.0 * n[5] + n[6] - 2.0 * n[7] + n[8])
            })
        })
        .collect();
    if r.is_empty() {
        return 0.0;
    }
    let mid = r.len() / 2;
    let med = *r.select_nth_unstable_by(mid, f64::total_cmp).1;
    let mut dev: Vec<f64> = r.iter().map(|x| (x - med).abs()).collect();
    let mad = *dev.select_nth_unstable_by(mid, f64::total_cmp).1;
    1.4826 * mad / 6.0
}

/// Sum of `v` over a box of (2r + 1) samples along each axis, edges clamped, into `out`
/// (`tmp` is scratch of the same size): rows and columns plane by plane, then across planes.
fn box_sum(v: &[f32], n: [usize; 3], r: usize, tmp: &mut [f32], out: &mut [f32]) {
    let [nx, ny, nz] = n;
    let plane = nx * ny;
    let r = r as i64;
    tmp.par_chunks_mut(plane).enumerate().for_each(|(k, t)| {
        let src = &v[k * plane..(k + 1) * plane];
        let mut rows = vec![0.0f32; plane];
        for j in 0..ny {
            let line = &src[j * nx..(j + 1) * nx];
            let at = |x: i64| line[x.clamp(0, nx as i64 - 1) as usize];
            let mut acc: f32 = (-r..=r).map(at).sum();
            for x in 0..nx as i64 {
                rows[j * nx + x as usize] = acc;
                acc += at(x + r + 1) - at(x - r);
            }
        }
        for i in 0..nx {
            let at = |y: i64| rows[y.clamp(0, ny as i64 - 1) as usize * nx + i];
            let mut acc: f32 = (-r..=r).map(at).sum();
            for y in 0..ny as i64 {
                t[y as usize * nx + i] = acc;
                acc += at(y + r + 1) - at(y - r);
            }
        }
    });
    out.par_chunks_mut(plane).enumerate().for_each(|(k, o)| {
        o.fill(0.0);
        for dz in -r..=r {
            let kk = (k as i64 + dz).clamp(0, nz as i64 - 1) as usize;
            for (a, &b) in o.iter_mut().zip(&tmp[kk * plane..(kk + 1) * plane]) {
                *a += b;
            }
        }
    });
}

/// Non-local means: every voxel becomes a weighted average of the voxels within `search`
/// voxels whose surrounding patches (radius `patch`) look alike. Weights are
/// `exp(-max(d² − 2σ², 0) / h²)` for the mean squared patch difference `d²`, noise σ from
/// [`noise_sigma`] and `h = strength · σ` (about 1: a good start; larger smooths more).
///
/// Speed: patch distances for each search offset come from one box filter over the whole
/// image (Darbon et al. 2008), and each offset serves both voxels of a pair. Time grows with
/// (2 · search + 1)³; search 2–3 suits most scans.
pub fn non_local_means(vol: &LoadedVolume, search: usize, patch: usize, strength: f64, progress: Progress) -> anyhow::Result<Vec<f32>> {
    let sigma = noise_sigma(vol);
    if sigma <= 0.0 || search == 0 {
        return Ok(vol.values.clone());
    }
    let n = vol.dims;
    let [nx, ny, nz] = n;
    let plane = nx * ny;
    let h2 = ((strength * sigma) as f32).powi(2).max(1e-12);
    let bias = (2.0 * sigma * sigma) as f32;
    let patch_count = ((2 * patch + 1) as f32).powi(3);
    let v = &vol.values;
    let mut num = vec![0.0f32; v.len()];
    let mut den = vec![0.0f32; v.len()];
    let mut wmax = vec![0.0f32; v.len()];
    // Half of the offsets: the other half are the same pairs seen from the other voxel.
    let s = search as i64;
    let offsets: Vec<[i64; 3]> = (-s..=s)
        .flat_map(|dz| (-s..=s).flat_map(move |dy| (-s..=s).map(move |dx| [dx, dy, dz])))
        .filter(|&[dx, dy, dz]| dz > 0 || (dz == 0 && (dy > 0 || (dy == 0 && dx > 0))))
        .collect();
    let mut sq = vec![0.0f32; v.len()];
    let mut tmp = vec![0.0f32; v.len()];
    let mut d = vec![0.0f32; v.len()];
    for (count, &[dx, dy, dz]) in offsets.iter().enumerate() {
        let shift = dz * plane as i64 + dy * nx as i64 + dx;
        let inside = |i: usize, j: usize, k: usize| {
            let (a, b, c) = (i as i64 + dx, j as i64 + dy, k as i64 + dz);
            a >= 0 && b >= 0 && c >= 0 && (a as usize) < nx && (b as usize) < ny && (c as usize) < nz
        };
        // Squared differences to the shifted image, then their patch sums.
        sq.par_chunks_mut(plane).enumerate().for_each(|(k, p)| {
            for j in 0..ny {
                for i in 0..nx {
                    let q = (k * ny + j) * nx + i;
                    p[j * nx + i] = if inside(i, j, k) {
                        let e = v[q] - v[(q as i64 + shift) as usize];
                        e * e
                    } else {
                        0.0
                    };
                }
            }
        });
        box_sum(&sq, n, patch, &mut tmp, &mut d);
        // Weights of the pairs (x, x + offset), kept in `d`.
        d.par_chunks_mut(plane).enumerate().for_each(|(k, p)| {
            for j in 0..ny {
                for i in 0..nx {
                    let w = &mut p[j * nx + i];
                    *w = if inside(i, j, k) { (-((*w / patch_count - bias).max(0.0)) / h2).exp() } else { 0.0 };
                }
            }
        });
        // x takes from x + offset, and x + offset from x (each voxel written by one thread).
        num.par_chunks_mut(plane)
            .zip(den.par_chunks_mut(plane))
            .zip(wmax.par_chunks_mut(plane))
            .enumerate()
            .for_each(|(k, ((np, dp), mp))| {
                for j in 0..ny {
                    for i in 0..nx {
                        let q = (k * ny + j) * nx + i;
                        let o = j * nx + i;
                        let w = d[q];
                        if w > 0.0 {
                            np[o] += w * v[(q as i64 + shift) as usize];
                            dp[o] += w;
                            mp[o] = mp[o].max(w);
                        }
                        let (a, b, c) = (i as i64 - dx, j as i64 - dy, k as i64 - dz);
                        if a >= 0 && b >= 0 && c >= 0 && (a as usize) < nx && (b as usize) < ny && (c as usize) < nz {
                            let src = (q as i64 - shift) as usize;
                            let w = d[src];
                            if w > 0.0 {
                                np[o] += w * v[src];
                                dp[o] += w;
                                mp[o] = mp[o].max(w);
                            }
                        }
                    }
                }
            });
        check(progress, (count + 1) as f32 / offsets.len() as f32)?;
    }
    // The voxel itself counts as much as its best match.
    Ok((0..v.len())
        .into_par_iter()
        .map(|q| {
            let w0 = if wmax[q] > 0.0 { wmax[q] } else { 1.0 };
            (num[q] + w0 * v[q]) / (den[q] + w0)
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::no_progress;
    use glam::DVec3;
    use mm_volume::geometry::Geometry;

    /// Normal random numbers (Box–Muller on a small LCG), reproducible.
    struct Rng(u64);

    impl Rng {
        fn uniform(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        }

        fn normal(&mut self) -> f64 {
            let (a, b) = (self.uniform(), self.uniform());
            (-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()
        }
    }

    /// A piecewise-constant phantom: a block of 200 in 100, split by a step edge at x = 24 (the
    /// left part is 100, the right 200), plus a bright ball; returns (clean, noisy).
    fn phantom(sigma: f64) -> (LoadedVolume, LoadedVolume) {
        let dims = [48, 40, 32];
        let mut clean = Vec::new();
        for k in 0..dims[2] {
            for j in 0..dims[1] {
                for i in 0..dims[0] {
                    let ball = (DVec3::new(i as f64, j as f64, k as f64) - DVec3::new(12.0, 20.0, 16.0)).length() < 6.0;
                    clean.push(if ball { 300.0 } else if i >= 24 { 200.0 } else { 100.0 });
                }
            }
        }
        let mut rng = Rng(7);
        let noisy: Vec<f32> = clean.iter().map(|&c: &f32| c + (sigma * rng.normal()) as f32).collect();
        let g = Geometry::from_axes([dims[0], dims[1], dims[2], 1], [1.0, 1.0, 1.2], [DVec3::X, DVec3::Y, DVec3::Z], DVec3::ZERO);
        (LoadedVolume::from_values("clean", dims, clean, g.clone()), LoadedVolume::from_values("noisy", dims, noisy, g))
    }

    fn rms(a: &[f32], b: &[f32]) -> f64 {
        (a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum::<f64>() / a.len() as f64).sqrt()
    }

    /// Mean step across the edge at x = 24, two voxels from it on each side, away from the ball.
    fn edge_contrast(v: &[f32], vol: &LoadedVolume) -> f64 {
        let mut s = 0.0;
        let mut n = 0.0;
        for k in 4..28 {
            for j in 30..38 {
                s += (v[vol.index(26, j, k)] - v[vol.index(21, j, k)]) as f64;
                n += 1.0;
            }
        }
        s / n
    }

    #[test]
    fn noise_estimate() {
        for sigma in [5.0, 20.0] {
            let (_, noisy) = phantom(sigma);
            let est = noise_sigma(&noisy);
            assert!((est / sigma - 1.0).abs() < 0.15, "σ {sigma}: estimated {est:.2}");
        }
        // A flat image has none, and the zeros around a skull-stripped image are ignored.
        let (clean, mut noisy) = phantom(10.0);
        let flat = LoadedVolume::from_values("flat", clean.dims, vec![7.0; clean.values.len()], clean.geometry.clone());
        assert_eq!(noise_sigma(&flat), 0.0);
        for k in 0..noisy.dims[2] {
            for j in 0..noisy.dims[1] {
                for i in 0..noisy.dims[0] / 3 {
                    let q = noisy.index(i, j, k);
                    noisy.values[q] = 0.0;
                }
            }
        }
        let est = noise_sigma(&noisy);
        assert!((est / 10.0 - 1.0).abs() < 0.15, "with a zero background: {est:.2}");
    }

    #[test]
    fn every_filter_reduces_noise_and_keeps_edges() {
        let (clean, noisy) = phantom(20.0);
        let before = rms(&noisy.values, &clean.values);
        let contrast = edge_contrast(&clean.values, &clean);
        let runs: [(&str, Vec<f32>, f64, f64); 4] = [
            ("median", median(&noisy, 1, no_progress()).unwrap(), 0.6, 0.85),
            ("gradient diffusion", gradient_diffusion(&noisy, 10, 1.0, no_progress()).unwrap(), 0.5, 0.9),
            ("curvature diffusion", curvature_diffusion(&noisy, 10, 1.0, no_progress()).unwrap(), 0.5, 0.9),
            ("non-local means", non_local_means(&noisy, 2, 1, 1.0, no_progress()).unwrap(), 0.4, 0.9),
        ];
        for (name, out, max_ratio, min_contrast) in runs {
            let after = rms(&out, &clean.values);
            let kept = edge_contrast(&out, &clean) / contrast;
            eprintln!("{name}: RMS error {before:.1} → {after:.1} ({:.0} %), edge contrast kept {:.0} %", 100.0 * after / before, 100.0 * kept);
            assert!(after < before * max_ratio, "{name}: RMS {before:.1} → {after:.1}");
            assert!(kept > min_contrast, "{name}: edge contrast kept {kept:.2}");
        }
    }

    #[test]
    fn median_removes_salt_and_pepper() {
        let (clean, _) = phantom(0.0);
        let mut sp = clean.clone();
        let mut rng = Rng(3);
        for v in sp.values.iter_mut() {
            let u = rng.uniform();
            if u < 0.03 {
                *v = 0.0;
            } else if u < 0.06 {
                *v = 1000.0;
            }
        }
        let out = median(&sp, 1, no_progress()).unwrap();
        // Away from the edges of the phantom's regions, the clean values come back exactly.
        let wrong = out.iter().zip(&clean.values).filter(|(a, b)| (*a - *b).abs() > 1e-3).count();
        assert!(wrong < out.len() / 100, "{wrong} voxels still wrong");
    }

    #[test]
    fn constants_stay_and_cancelling_stops() {
        let (clean, noisy) = phantom(10.0);
        let flat = LoadedVolume::from_values("flat", clean.dims, vec![42.0; clean.values.len()], clean.geometry.clone());
        for out in [
            median(&flat, 1, no_progress()).unwrap(),
            gradient_diffusion(&flat, 3, 1.0, no_progress()).unwrap(),
            curvature_diffusion(&flat, 3, 1.0, no_progress()).unwrap(),
            non_local_means(&flat, 1, 1, 1.0, no_progress()).unwrap(),
        ] {
            assert!(out.iter().all(|&x| (x - 42.0).abs() < 1e-4));
        }
        let stop: Progress = &|_| false;
        assert!(median(&noisy, 1, stop).is_err());
        assert!(gradient_diffusion(&noisy, 3, 1.0, stop).is_err());
        assert!(curvature_diffusion(&noisy, 3, 1.0, stop).is_err());
        assert!(non_local_means(&noisy, 1, 1, 1.0, stop).is_err());
    }
}