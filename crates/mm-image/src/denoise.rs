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

/// Median over a cube of (2r + 1)³ voxels (clamped at the edges), in the total order of
/// `f32::total_cmp`.
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
    // The rows of a voxel's cube, plane by plane (edges clamped).
    let rows_of = |j: usize, k: usize| {
        (-r..=r).flat_map(move |dz| {
            let kk = (k as i64 + dz).clamp(0, nz as i64 - 1) as usize;
            (-r..=r).map(move |dy| g.idx(0, (j as i64 + dy).clamp(0, ny as i64 - 1) as usize, kk))
        })
    };
    out.par_chunks_mut((nx * ny).max(1)).enumerate().for_each(|(k, plane)| {
        if cancelled.load(std::sync::atomic::Ordering::Relaxed) {
            return;
        }
        if radius == 1 {
            // The 9 rows as total-order keys, one key of edge padding on each side.
            let mut keys = vec![0i32; 9 * (nx + 2)];
            let mut cube = [[0i32; MEDIAN_LANES]; 27];
            for j in 0..ny {
                for (row, src) in keys.chunks_mut(nx + 2).zip(rows_of(j, k)) {
                    let line = &vol.values[src..][..nx];
                    row[0] = total_key(line[0]);
                    row[nx + 1] = total_key(line[nx - 1]);
                    row[1..=nx].iter_mut().zip(line).for_each(|(o, &x)| *o = total_key(x));
                }
                for i0 in (0..nx).step_by(MEDIAN_LANES) {
                    let lanes = (nx - i0).min(MEDIAN_LANES);
                    for (row, c) in keys.chunks(nx + 2).zip(cube.chunks_mut(3)) {
                        for (dx, e) in c.iter_mut().enumerate() {
                            e[..lanes].copy_from_slice(&row[i0 + dx..][..lanes]);
                        }
                    }
                    for &(a, b) in median27_network() {
                        let (low, high) = cube.split_at_mut(b as usize);
                        for (x, y) in low[a as usize].iter_mut().zip(high[0].iter_mut()) {
                            (*x, *y) = ((*x).min(*y), (*x).max(*y));
                        }
                    }
                    for (o, &m) in plane[j * nx + i0..][..lanes].iter_mut().zip(&cube[13]) {
                        *o = from_total_key(m);
                    }
                }
            }
        } else {
            let mut buf = Vec::with_capacity(((2 * r + 1) as usize).pow(3));
            let mut rows = Vec::with_capacity(((2 * r + 1) as usize).pow(2));
            for j in 0..ny {
                rows.clear();
                rows.extend(rows_of(j, k));
                for i in 0..nx {
                    buf.clear();
                    if i as i64 >= r && i as i64 + r < nx as i64 {
                        for &row in &rows {
                            buf.extend_from_slice(&vol.values[row + i - radius..][..2 * radius + 1]);
                        }
                    } else {
                        for &row in &rows {
                            buf.extend((-r..=r).map(|dx| vol.values[row + (i as i64 + dx).clamp(0, nx as i64 - 1) as usize]));
                        }
                    }
                    let mid = buf.len() / 2;
                    let (_, m, _) = buf.select_nth_unstable_by(mid, f32::total_cmp);
                    plane[j * nx + i] = *m;
                }
            }
        }
        let d = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if !progress(d as f32 / nz as f32) {
            cancelled.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    });
    check(progress, 1.0).and_then(|_| if cancelled.into_inner() { anyhow::bail!("cancelled") } else { Ok(out) })
}

/// Voxels whose 3 × 3 × 3 medians are found side by side.
const MEDIAN_LANES: usize = 8;

/// `x` as an integer that orders as `f32::total_cmp` orders the floats.
fn total_key(x: f32) -> i32 {
    let b = x.to_bits() as i32;
    b ^ (((b >> 31) as u32) >> 1) as i32
}

/// The float of a [`total_key`].
fn from_total_key(k: i32) -> f32 {
    f32::from_bits((k ^ (((k >> 31) as u32) >> 1) as i32) as u32)
}

/// Compare-exchanges (lower index gets the smaller) that bring the median of 27 values to
/// index 13: Batcher's odd–even merge sort of 32, without the exchanges with the 5 positions
/// past 27 (which would hold +∞ and never move) or that cannot reach index 13.
fn median27_network() -> &'static [(u8, u8)] {
    static NETWORK: std::sync::OnceLock<Vec<(u8, u8)>> = std::sync::OnceLock::new();
    NETWORK.get_or_init(|| {
        fn merge(lo: usize, hi: usize, r: usize, out: &mut Vec<(u8, u8)>) {
            let step = r * 2;
            if step < hi - lo {
                merge(lo, hi, step, out);
                merge(lo + r, hi, step, out);
                out.extend((lo + r..hi - r).step_by(step).map(|i| (i as u8, (i + r) as u8)));
            } else {
                out.push((lo as u8, (lo + r) as u8));
            }
        }
        fn sort(lo: usize, hi: usize, out: &mut Vec<(u8, u8)>) {
            if hi > lo {
                let mid = lo + (hi - lo) / 2;
                sort(lo, mid, out);
                sort(mid + 1, hi, out);
                merge(lo, hi, 1, out);
            }
        }
        let mut all = Vec::new();
        sort(0, 31, &mut all);
        let mut needed = [false; 32];
        needed[13] = true;
        let mut kept: Vec<(u8, u8)> = Vec::new();
        for &(a, b) in all.iter().rev().filter(|&&(_, b)| b < 27) {
            if needed[a as usize] || needed[b as usize] {
                needed[a as usize] = true;
                needed[b as usize] = true;
                kept.push((a, b));
            }
        }
        kept.reverse();
        kept
    })
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
    let mut v = vol.values.clone();
    let mut next = vec![0.0f32; v.len()];
    for it in 0..iterations {
        let k2 = 2.0 * conductance * conductance * mean_gradient_sq(&v, g, h);
        if k2 <= 0.0 {
            break;
        }
        gradient_step(&v, &mut next, g, h, k2, dt);
        std::mem::swap(&mut v, &mut next);
        check(progress, (it + 1) as f32 / iterations.max(1) as f32)?;
    }
    Ok(v)
}

/// Planes per task of a diffusion step: each task also makes the faces just below its first
/// plane, so runs of several planes share that work.
fn plane_run(nz: usize) -> usize {
    (nz / (2 * rayon::current_num_threads())).clamp(2, 8)
}

/// The flux of gradient diffusion through the face from a voxel of value `x` to its
/// neighbour of value `y` along an axis of spacing `h`.
fn conduction_flux(x: f64, y: f64, h: f64, k2: f64) -> f64 {
    let d = (y - x) / h;
    (-d * d / k2).exp() * d
}

/// One explicit step of gradient diffusion from `v` into `next`. A voxel's change is
/// `Σ_a (flux out − flux in) / h_a`, and the flux through a face is the same number for the
/// voxels on both sides, so each face's flux is made once: along x and y while going through
/// a plane row by row, along z for a whole plane ahead. At the grid's edges a voxel's missing
/// face is its own clamped one, as the per-voxel formula gives.
fn gradient_step(v: &[f32], next: &mut [f32], g: Grid, h: [f64; 3], k2: f64, dt: f64) {
    let [nx, ny, nz] = g.n;
    let plane = nx * ny;
    let run = plane_run(nz);
    let val = |q: usize| v[q] as f64;
    // Fluxes through the z faces above the voxels of plane k (from each voxel to the one above).
    let z_faces = |k: usize, f: &mut [f64]| {
        let up = if k + 1 < nz { plane } else { 0 };
        for (o, q) in f.iter_mut().zip(k * plane..) {
            *o = conduction_flux(val(q), val(q + up), h[2], k2);
        }
    };
    next.par_chunks_mut((plane * run).max(1)).enumerate().for_each(|(r, out)| {
        let k0 = r * run;
        let (mut fz_below, mut fz_above) = (vec![0.0f64; plane], vec![0.0f64; plane]);
        let (mut fy_below, mut fy_above) = (vec![0.0f64; nx], vec![0.0f64; nx]);
        let mut fx = vec![0.0f64; nx];
        if k0 > 0 {
            z_faces(k0 - 1, &mut fz_below);
        }
        for (dk, o) in out.chunks_mut(plane).enumerate() {
            let k = k0 + dk;
            z_faces(k, &mut fz_above);
            for j in 0..ny {
                let row = (k * ny + j) * nx;
                let next_row = if j + 1 < ny { row + nx } else { row };
                for i in 0..nx {
                    fx[i] = conduction_flux(val(row + i), val(row + (i + 1).min(nx - 1)), h[0], k2);
                    fy_above[i] = conduction_flux(val(row + i), val(next_row + i), h[1], k2);
                }
                for i in 0..nx {
                    let q = row + i;
                    let x = val(q);
                    let below = [
                        if i > 0 { fx[i - 1] } else { conduction_flux(x, x, h[0], k2) },
                        if j > 0 { fy_below[i] } else { conduction_flux(x, x, h[1], k2) },
                        if k > 0 { fz_below[j * nx + i] } else { conduction_flux(x, x, h[2], k2) },
                    ];
                    let above = [fx[i], fy_above[i], fz_above[j * nx + i]];
                    let mut change = 0.0;
                    for a in 0..3 {
                        change += (above[a] - below[a]) / h[a];
                    }
                    o[j * nx + i] = (x + dt * change) as f32;
                }
                std::mem::swap(&mut fy_below, &mut fy_above);
            }
            std::mem::swap(&mut fz_below, &mut fz_above);
        }
    });
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
    let mut v = vol.values.clone();
    let mut next = vec![0.0f32; v.len()];
    for it in 0..iterations {
        let k2 = 2.0 * conductance * conductance * mean_gradient_sq(&v, g, h);
        if k2 <= 0.0 {
            break;
        }
        curvature_step(&v, &mut next, g, h, k2, dt);
        std::mem::swap(&mut v, &mut next);
        check(progress, (it + 1) as f32 / iterations.max(1) as f32)?;
    }
    Ok(v)
}

/// Central differences (mm) along x, y and z at every voxel of plane `k`.
fn central_differences(v: &[f32], g: Grid, h: [f64; 3], k: usize, d: &mut [[f64; 3]]) {
    let [nx, ny, _] = g.n;
    let val = |q: usize| v[q] as f64;
    for j in 0..ny {
        for i in 0..nx {
            let c = [i, j, k];
            let e = &mut d[j * nx + i];
            for b in 0..3 {
                e[b] = (val(g.step(c, b, 1)) - val(g.step(c, b, -1))) / (2.0 * h[b]);
            }
        }
    }
}

/// The flux of curvature diffusion through the face along axis `a` from a voxel (value `x`,
/// central differences `dx`) to its neighbour (`y`, `dy`), and the difference `(y − x) / h`
/// across the face (the one voxel's forward, the other's backward difference). The gradient
/// at the face is that difference along `a` and, along the other axes, the average of the
/// central differences at both ends.
fn curvature_flux(a: usize, x: f64, y: f64, dx: &[f64; 3], dy: &[f64; 3], h: [f64; 3], k2: f64) -> (f64, f64) {
    let fwd = (y - x) / h[a];
    let mut gf = fwd * fwd;
    for b in (0..3).filter(|&b| b != a) {
        gf += 0.25 * (dx[b] + dy[b]).powi(2);
    }
    let mf = gf.sqrt();
    (if mf > 1e-12 { (-gf / k2).exp() * fwd / mf } else { 0.0 }, fwd)
}

/// One explicit step of curvature diffusion from `v` into `next`. As in [`gradient_step`],
/// each face's flux is made once and serves the voxels on both sides; the central differences
/// it needs are made once per voxel, a plane ahead of the plane being updated.
fn curvature_step(v: &[f32], next: &mut [f32], g: Grid, h: [f64; 3], k2: f64, dt: f64) {
    let [nx, ny, nz] = g.n;
    let plane = nx * ny;
    let run = plane_run(nz);
    let val = |q: usize| v[q] as f64;
    // Fluxes through the z faces above plane k, from the central differences of plane k and
    // of the plane above (or of plane k again at the top).
    let z_faces = |k: usize, d: &[[f64; 3]], d_up: &[[f64; 3]], f: &mut [(f64, f64)]| {
        let up = if k + 1 < nz { plane } else { 0 };
        for (n, o) in f.iter_mut().enumerate() {
            let q = k * plane + n;
            *o = curvature_flux(2, val(q), val(q + up), &d[n], &d_up[n], h, k2);
        }
    };
    next.par_chunks_mut((plane * run).max(1)).enumerate().for_each(|(r, out)| {
        let k0 = r * run;
        let (mut d_here, mut d_up) = (vec![[0.0f64; 3]; plane], vec![[0.0f64; 3]; plane]);
        // (flux, difference) of the faces.
        let (mut fz_below, mut fz_above) = (vec![(0.0f64, 0.0f64); plane], vec![(0.0f64, 0.0f64); plane]);
        let (mut fy_below, mut fy_above) = (vec![(0.0f64, 0.0f64); nx], vec![(0.0f64, 0.0f64); nx]);
        let mut fx = vec![(0.0f64, 0.0f64); nx];
        central_differences(v, g, h, k0, &mut d_here);
        if k0 > 0 {
            central_differences(v, g, h, k0 - 1, &mut d_up);
            z_faces(k0 - 1, &d_up, &d_here, &mut fz_below);
        }
        for (dk, o) in out.chunks_mut(plane).enumerate() {
            let k = k0 + dk;
            if k + 1 < nz {
                central_differences(v, g, h, k + 1, &mut d_up);
                z_faces(k, &d_here, &d_up, &mut fz_above);
            } else {
                z_faces(k, &d_here, &d_here, &mut fz_above);
            }
            for j in 0..ny {
                let row = (k * ny + j) * nx;
                let (d_row, d_next_row) = (&d_here[j * nx..][..nx], &d_here[(j + 1).min(ny - 1) * nx..][..nx]);
                let next_row = if j + 1 < ny { row + nx } else { row };
                for i in 0..nx {
                    let i1 = (i + 1).min(nx - 1);
                    fx[i] = curvature_flux(0, val(row + i), val(row + i1), &d_row[i], &d_row[i1], h, k2);
                    fy_above[i] = curvature_flux(1, val(row + i), val(next_row + i), &d_row[i], &d_next_row[i], h, k2);
                }
                for i in 0..nx {
                    let q = row + i;
                    let x = val(q);
                    let own = |a: usize| curvature_flux(a, x, x, &d_row[i], &d_row[i], h, k2);
                    let below = [
                        if i > 0 { fx[i - 1] } else { own(0) },
                        if j > 0 { fy_below[i] } else { own(1) },
                        if k > 0 { fz_below[j * nx + i] } else { own(2) },
                    ];
                    let above = [fx[i], fy_above[i], fz_above[j * nx + i]];
                    let mut speed = 0.0;
                    let mut fwd = [0.0; 3];
                    let mut bwd = [0.0; 3];
                    for a in 0..3 {
                        fwd[a] = above[a].1;
                        bwd[a] = below[a].1;
                        speed += (above[a].0 - below[a].0) / h[a];
                    }
                    // |∇I| by upwind differences in the direction of the motion.
                    let mut mag = 0.0;
                    for a in 0..3 {
                        mag += if speed > 0.0 { bwd[a].min(0.0).powi(2) + fwd[a].max(0.0).powi(2) } else { bwd[a].max(0.0).powi(2) + fwd[a].min(0.0).powi(2) };
                    }
                    o[j * nx + i] = (x + dt * mag.sqrt() * speed) as f32;
                }
                std::mem::swap(&mut fy_below, &mut fy_above);
            }
            std::mem::swap(&mut fz_below, &mut fz_above);
            std::mem::swap(&mut d_here, &mut d_up);
        }
    });
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
    let r = patch as i64;
    // Patch sums along x and y of each plane, then the pair weights.
    let mut tmp = vec![0.0f32; v.len()];
    let mut d = vec![0.0f32; v.len()];
    for (count, &[dx, dy, dz]) in offsets.iter().enumerate() {
        let shift = dz * plane as i64 + dy * nx as i64 + dx;
        // Along each axis, the voxels x whose partner x + offset is inside the grid, and those
        // whose partner x − offset is.
        let span = |o: i64, len: usize| (-o).clamp(0, len as i64) as usize..(len as i64 - o).clamp(0, len as i64) as usize;
        let ahead = [span(dx, nx), span(dy, ny), span(dz, nz)];
        let behind = [span(-dx, nx), span(-dy, ny), span(-dz, nz)];
        // Squared differences to the shifted image (0 where the partner is outside), summed
        // over the patch (a box of 2·patch + 1 samples per axis, edges clamped) along rows and
        // then columns, plane by plane; across planes below.
        tmp.par_chunks_mut(plane).enumerate().for_each_init(
            || (vec![0.0f32; plane], vec![0.0f32; nx], vec![0.0f32; nx]),
            |(rows, sq, acc), (k, t)| {
                if !ahead[2].contains(&k) {
                    t.fill(0.0);
                    return;
                }
                for j in 0..ny {
                    let out = &mut rows[j * nx..][..nx];
                    if !ahead[1].contains(&j) {
                        out.fill(0.0);
                        continue;
                    }
                    let q0 = (k * ny + j) * nx;
                    sq.fill(0.0);
                    for i in ahead[0].clone() {
                        let e = v[q0 + i] - v[(q0 as i64 + i as i64 + shift) as usize];
                        sq[i] = e * e;
                    }
                    let at = |x: i64| sq[x.clamp(0, nx as i64 - 1) as usize];
                    let mut acc: f32 = (-r..=r).map(at).sum();
                    for (x, o) in out.iter_mut().enumerate() {
                        *o = acc;
                        acc += at(x as i64 + r + 1) - at(x as i64 - r);
                    }
                }
                // Every column's running sum at once, a row at a time (each column adds the same
                // values in the same order as on its own: from −0, as `sum` starts).
                let row = |y: i64| &rows[y.clamp(0, ny as i64 - 1) as usize * nx..][..nx];
                acc.fill(-0.0);
                for y in -r..=r {
                    acc.iter_mut().zip(row(y)).for_each(|(a, &b)| *a += b);
                }
                for y in 0..ny {
                    t[y * nx..][..nx].copy_from_slice(acc);
                    let (add, sub) = (row(y as i64 + r + 1), row(y as i64 - r));
                    acc.iter_mut().zip(add).zip(sub).for_each(|((a, &b), &c)| *a += b - c);
                }
            },
        );
        // Patch sums across planes, and from them the weights of the pairs (x, x + offset)
        // (0 where the partner is outside).
        d.par_chunks_mut(plane).enumerate().for_each(|(k, o)| {
            if !ahead[2].contains(&k) {
                o.fill(0.0);
                return;
            }
            for j in 0..ny {
                let row = &mut o[j * nx..][..nx];
                row.fill(0.0);
                if !ahead[1].contains(&j) {
                    continue;
                }
                for dz in -r..=r {
                    let kk = (k as i64 + dz).clamp(0, nz as i64 - 1) as usize;
                    for (a, &b) in row.iter_mut().zip(&tmp[(kk * ny + j) * nx..][..nx]) {
                        *a += b;
                    }
                }
                for (i, w) in row.iter_mut().enumerate() {
                    *w = if ahead[0].contains(&i) {
                        // Patches closer than the noise allows weigh exactly exp(0) = 1.
                        let excess = (*w / patch_count - bias).max(0.0);
                        if excess == 0.0 { 1.0 } else { (-excess / h2).exp() }
                    } else {
                        0.0
                    };
                }
            }
        });
        // x takes from x + offset, and x + offset from x (each voxel written by one thread).
        num.par_chunks_mut(plane)
            .zip(den.par_chunks_mut(plane))
            .zip(wmax.par_chunks_mut(plane))
            .enumerate()
            .for_each(|(k, ((np, dp), mp))| {
                // Add the pairs of weight `w[i]` with partner value `x[i]` to the voxels of
                // row `j` from column `lo` on (pairs of weight 0 add nothing).
                let mut add = |j: usize, lo: usize, w: &[f32], x: &[f32]| {
                    let o = j * nx + lo..j * nx + lo + w.len();
                    for ((((n, d), m), &w), &x) in np[o.clone()].iter_mut().zip(&mut dp[o.clone()]).zip(&mut mp[o]).zip(w).zip(x) {
                        let take = w > 0.0;
                        *n = if take { *n + w * x } else { *n };
                        *d = if take { *d + w } else { *d };
                        *m = if take { m.max(w) } else { *m };
                    }
                };
                for j in 0..ny {
                    let q0 = (k * ny + j) * nx;
                    // The pairs (x, x + offset), where the partner is inside (elsewhere `d`
                    // is 0) …
                    if ahead[2].contains(&k) && ahead[1].contains(&j) && !ahead[0].is_empty() {
                        let (lo, len) = (ahead[0].start, ahead[0].len());
                        let partner = (q0 as i64 + lo as i64 + shift) as usize;
                        add(j, lo, &d[q0 + lo..][..len], &v[partner..][..len]);
                    }
                    // … then the pairs (x − offset, x), after them for every voxel as before.
                    if behind[2].contains(&k) && behind[1].contains(&j) && !behind[0].is_empty() {
                        let (lo, len) = (behind[0].start, behind[0].len());
                        let src = (q0 as i64 + lo as i64 - shift) as usize;
                        add(j, lo, &d[src..][..len], &v[src..][..len]);
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

    /// Noisy values on `dims` with flat patches, exact zeros and a step, plus a NaN and an
    /// infinity when `odd` (whose neighbourhoods must come out as the per-voxel formulas give).
    fn rough(dims: [usize; 3], odd: bool) -> Vec<f32> {
        let mut rng = Rng(11);
        let n = dims.iter().product::<usize>();
        let mut v: Vec<f32> = (0..n).map(|q| if q % 9 < 2 { 50.0 } else if q % 9 == 2 { 0.0 } else { (if q % dims[0] > dims[0] / 2 { 200.0 } else { 100.0 }) + (20.0 * rng.normal()) as f32 }).collect();
        if odd && n > 8 {
            v[n / 3] = f32::NAN;
            v[n / 2] = f32::INFINITY;
        }
        v
    }

    fn same_bits(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()))
    }

    /// The gradient diffusion step written per voxel, every flux made on its own.
    fn reference_gradient_step(v: &[f32], g: Grid, h: [f64; 3], k2: f64, dt: f64) -> Vec<f32> {
        let [nx, ny, nz] = g.n;
        let mut next = vec![0.0f32; v.len()];
        for k in 0..nz {
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
                    next[g.idx(i, j, k)] = (x + dt * change) as f32;
                }
            }
        }
        next
    }

    /// The curvature diffusion step written per voxel, every flux and central difference made
    /// on its own.
    fn reference_curvature_step(v: &[f32], g: Grid, h: [f64; 3], k2: f64, dt: f64) -> Vec<f32> {
        let [nx, ny, nz] = g.n;
        let val = |q: usize| v[q] as f64;
        let mut next = vec![0.0f32; v.len()];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let c = [i, j, k];
                    let x = val(g.idx(i, j, k));
                    let mut speed = 0.0;
                    let mut fwd = [0.0; 3];
                    let mut bwd = [0.0; 3];
                    for a in 0..3 {
                        fwd[a] = (val(g.step(c, a, 1)) - x) / h[a];
                        bwd[a] = (x - val(g.step(c, a, -1))) / h[a];
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
                    let mut mag = 0.0;
                    for a in 0..3 {
                        mag += if speed > 0.0 { bwd[a].min(0.0).powi(2) + fwd[a].max(0.0).powi(2) } else { bwd[a].max(0.0).powi(2) + fwd[a].min(0.0).powi(2) };
                    }
                    next[g.idx(i, j, k)] = (x + dt * mag.sqrt() * speed) as f32;
                }
            }
        }
        next
    }

    #[test]
    fn diffusion_steps_match_the_per_voxel_formulas_bit_for_bit() {
        for (dims, h) in [([37, 29, 23], [0.47, 0.47, 1.0]), ([16, 1, 9], [1.0, 1.0, 1.0]), ([2, 3, 2], [0.8, 1.3, 2.0]), ([1, 1, 1], [1.0; 3]), ([5, 6, 40], [1.0, 0.7, 0.9])] {
            let g = Grid { n: dims };
            for odd in [false, true] {
                let v = rough(dims, odd);
                let dt = time_step(h);
                for k2 in [30.0, 3000.0] {
                    let mut next = vec![0.0f32; v.len()];
                    gradient_step(&v, &mut next, g, h, k2, dt);
                    assert!(same_bits(&next, &reference_gradient_step(&v, g, h, k2, dt)), "gradient {dims:?} odd {odd}");
                    curvature_step(&v, &mut next, g, h, k2, dt);
                    assert!(same_bits(&next, &reference_curvature_step(&v, g, h, k2, dt)), "curvature {dims:?} odd {odd}");
                }
            }
        }
        // Whole runs: the steps in turn with the per-iteration conductance.
        let dims = [21, 18, 13];
        let vol = LoadedVolume::from_values("r", dims, rough(dims, false), Geometry::from_axes([21, 18, 13, 1], [0.6, 0.6, 1.4], [DVec3::X, DVec3::Y, DVec3::Z], DVec3::ZERO));
        let (g, h) = (Grid { n: dims }, vol.geometry.spacing);
        let (mut a, mut b) = (vol.values.clone(), vol.values.clone());
        for _ in 0..3 {
            a = reference_gradient_step(&a, g, h, 2.0 * 1.5 * 1.5 * mean_gradient_sq(&a, g, h), time_step(h));
            b = reference_curvature_step(&b, g, h, 2.0 * 1.5 * 1.5 * mean_gradient_sq(&b, g, h), time_step(h));
        }
        assert!(same_bits(&gradient_diffusion(&vol, 3, 1.5, no_progress()).unwrap(), &a));
        assert!(same_bits(&curvature_diffusion(&vol, 3, 1.5, no_progress()).unwrap(), &b));
    }

    /// Sum of `v` over a box of (2r + 1) samples along each axis, edges clamped: rows and
    /// columns plane by plane, then across planes.
    fn reference_box_sum(v: &[f32], n: [usize; 3], r: usize) -> Vec<f32> {
        let [nx, ny, nz] = n;
        let plane = nx * ny;
        let r = r as i64;
        let mut tmp = vec![0.0f32; v.len()];
        for k in 0..nz {
            let mut rows = vec![0.0f32; plane];
            for j in 0..ny {
                let at = |x: i64| v[k * plane + j * nx + x.clamp(0, nx as i64 - 1) as usize];
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
                    tmp[k * plane + y as usize * nx + i] = acc;
                    acc += at(y + r + 1) - at(y - r);
                }
            }
        }
        let mut out = vec![0.0f32; v.len()];
        for k in 0..nz {
            for dz in -r..=r {
                let kk = (k as i64 + dz).clamp(0, nz as i64 - 1) as usize;
                for q in 0..plane {
                    out[k * plane + q] += tmp[kk * plane + q];
                }
            }
        }
        out
    }

    /// Non-local means written plainly: per offset, the squared differences of the whole
    /// image, their box sums, the weights, then both voxels of every pair.
    fn reference_non_local_means(vol: &LoadedVolume, search: usize, patch: usize, strength: f64) -> Vec<f32> {
        let sigma = noise_sigma(vol);
        if sigma <= 0.0 || search == 0 {
            return vol.values.clone();
        }
        let [nx, ny, nz] = vol.dims;
        let h2 = ((strength * sigma) as f32).powi(2).max(1e-12);
        let bias = (2.0 * sigma * sigma) as f32;
        let patch_count = ((2 * patch + 1) as f32).powi(3);
        let v = &vol.values;
        let (mut num, mut den, mut wmax) = (vec![0.0f32; v.len()], vec![0.0f32; v.len()], vec![0.0f32; v.len()]);
        let s = search as i64;
        let ijk = |q: usize| [(q % nx) as i64, ((q / nx) % ny) as i64, (q / (nx * ny)) as i64];
        let inside = |p: [i64; 3]| p[0] >= 0 && p[1] >= 0 && p[2] >= 0 && (p[0] as usize) < nx && (p[1] as usize) < ny && (p[2] as usize) < nz;
        for dz in -s..=s {
            for dy in -s..=s {
                for dx in -s..=s {
                    if !(dz > 0 || (dz == 0 && (dy > 0 || (dy == 0 && dx > 0)))) {
                        continue;
                    }
                    let shift = (dz * ny as i64 + dy) * nx as i64 + dx;
                    let ahead = |q: usize| inside([ijk(q)[0] + dx, ijk(q)[1] + dy, ijk(q)[2] + dz]);
                    let sq: Vec<f32> = (0..v.len())
                        .map(|q| {
                            if ahead(q) {
                                let e = v[q] - v[(q as i64 + shift) as usize];
                                e * e
                            } else {
                                0.0
                            }
                        })
                        .collect();
                    let d: Vec<f32> = reference_box_sum(&sq, vol.dims, patch)
                        .iter()
                        .enumerate()
                        .map(|(q, &w)| if ahead(q) { (-((w / patch_count - bias).max(0.0)) / h2).exp() } else { 0.0 })
                        .collect();
                    for q in 0..v.len() {
                        let w = d[q];
                        if w > 0.0 {
                            num[q] += w * v[(q as i64 + shift) as usize];
                            den[q] += w;
                            wmax[q] = wmax[q].max(w);
                        }
                        let c = ijk(q);
                        if inside([c[0] - dx, c[1] - dy, c[2] - dz]) {
                            let src = (q as i64 - shift) as usize;
                            let w = d[src];
                            if w > 0.0 {
                                num[q] += w * v[src];
                                den[q] += w;
                                wmax[q] = wmax[q].max(w);
                            }
                        }
                    }
                }
            }
        }
        (0..v.len())
            .map(|q| {
                let w0 = if wmax[q] > 0.0 { wmax[q] } else { 1.0 };
                (num[q] + w0 * v[q]) / (den[q] + w0)
            })
            .collect()
    }

    #[test]
    fn non_local_means_matches_the_plain_filter_bit_for_bit() {
        let (_, noisy) = phantom(15.0);
        let small = |dims: [usize; 3]| LoadedVolume::from_values("s", dims, rough(dims, false), Geometry::from_axes([dims[0], dims[1], dims[2], 1], [1.0; 3], [DVec3::X, DVec3::Y, DVec3::Z], DVec3::ZERO));
        let cases = [(noisy, 2, 1, 1.0), (small([13, 11, 9]), 1, 2, 0.7), (small([4, 3, 5]), 3, 1, 1.3), (small([9, 4, 6]), 2, 0, 1.0), (small([3, 3, 3]), 4, 3, 2.0)];
        for (vol, search, patch, strength) in cases {
            assert!(noise_sigma(&vol) > 0.0, "{:?} would be returned as it is", vol.dims);
            let a = non_local_means(&vol, search, patch, strength, no_progress()).unwrap();
            let b = reference_non_local_means(&vol, search, patch, strength);
            assert!(same_bits(&a, &b), "{:?} search {search} patch {patch}", vol.dims);
        }
    }

    /// The median written plainly: each voxel's cube gathered and the middle one selected.
    fn reference_median(vol: &LoadedVolume, radius: usize) -> Vec<f32> {
        let [nx, ny, nz] = vol.dims;
        let r = radius as i64;
        let c = |x: i64, n: usize| x.clamp(0, n as i64 - 1) as usize;
        (0..vol.values.len())
            .map(|q| {
                let (i, j, k) = ((q % nx) as i64, ((q / nx) % ny) as i64, (q / (nx * ny)) as i64);
                let mut buf = Vec::new();
                for dz in -r..=r {
                    for dy in -r..=r {
                        for dx in -r..=r {
                            buf.push(vol.get(c(i + dx, nx), c(j + dy, ny), c(k + dz, nz)));
                        }
                    }
                }
                let mid = buf.len() / 2;
                *buf.select_nth_unstable_by(mid, f32::total_cmp).1
            })
            .collect()
    }

    #[test]
    fn median_network_finds_the_middle_of_27() {
        let mut rng = Rng(5);
        for _ in 0..20_000 {
            // Few distinct values too, so ties are tried.
            let spread = if rng.uniform() < 0.5 { 4.0 } else { 1e6 };
            let mut v: [i32; 27] = std::array::from_fn(|_| (rng.uniform() * spread) as i32 - (spread / 2.0) as i32);
            let mut sorted = v;
            sorted.sort();
            for &(a, b) in median27_network() {
                let (x, y) = (v[a as usize], v[b as usize]);
                v[a as usize] = x.min(y);
                v[b as usize] = x.max(y);
            }
            assert_eq!(v[13], sorted[13]);
        }
        for x in [0.0f32, -0.0, 1.5, -2.25, f32::INFINITY, f32::NEG_INFINITY, f32::NAN, -f32::NAN, f32::MIN_POSITIVE] {
            assert_eq!(from_total_key(total_key(x)).to_bits(), x.to_bits());
            assert_eq!(total_key(x).cmp(&total_key(1.0)), x.total_cmp(&1.0));
        }
    }

    #[test]
    fn median_matches_the_plain_filter_bit_for_bit() {
        for dims in [[19, 13, 7], [1, 1, 1], [2, 5, 3], [9, 1, 4], [17, 3, 2]] {
            let mut v = rough(dims, true);
            // Signed zeros and infinities, whose order total_cmp fixes.
            for (q, x) in v.iter_mut().enumerate() {
                match q % 29 {
                    0 => *x = -0.0,
                    1 => *x = 0.0,
                    2 => *x = f32::NEG_INFINITY,
                    3 => *x = -f32::NAN,
                    _ => {}
                }
            }
            let vol = LoadedVolume::from_values("m", dims, v, Geometry::from_axes([dims[0], dims[1], dims[2], 1], [1.0; 3], [DVec3::X, DVec3::Y, DVec3::Z], DVec3::ZERO));
            for radius in [1, 2] {
                let a = median(&vol, radius, no_progress()).unwrap();
                let b = reference_median(&vol, radius);
                assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()), "{dims:?} radius {radius}");
            }
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