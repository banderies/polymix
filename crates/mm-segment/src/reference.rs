//! The straightforward forms of operations this crate does faster, kept for tests: the faster
//! forms must give the same results to the bit (or the same masks) on any input.

use crate::mask::{Connectivity, Mask};
use mm_volume::load::LoadedVolume;
use rayon::prelude::*;

/// Neighbours of voxel `n` inside the grid.
fn neighbours(dims: [usize; 3], n: usize, offsets: &[[i64; 3]], mut f: impl FnMut(usize)) {
    let plane = dims[0] * dims[1];
    let (i, j, k) = ((n % dims[0]) as i64, ((n / dims[0]) % dims[1]) as i64, (n / plane) as i64);
    for o in offsets {
        let (a, b, c) = (i + o[0], j + o[1], k + o[2]);
        if a >= 0 && b >= 0 && c >= 0 && (a as usize) < dims[0] && (b as usize) < dims[1] && (c as usize) < dims[2] {
            f((c as usize * dims[1] + b as usize) * dims[0] + a as usize);
        }
    }
}

/// The region connected to `seed` whose voxels pass `inside`, by a depth-first search.
pub(crate) fn flood(dims: [usize; 3], seed: usize, conn: Connectivity, inside: impl Fn(usize) -> bool) -> Mask {
    let mut out = Mask::new(dims);
    if seed >= out.len() || !inside(seed) {
        return out;
    }
    let offsets = conn.offsets();
    let mut stack = vec![seed];
    out.data[seed] = 1;
    while let Some(n) = stack.pop() {
        neighbours(dims, n, &offsets, |m| {
            if out.data[m] == 0 && inside(m) {
                out.data[m] = 1;
                stack.push(m);
            }
        });
    }
    out
}

pub(crate) fn region_grow(vol: &LoadedVolume, seed: [usize; 3], lo: f32, hi: f32, within: Option<&Mask>, conn: Connectivity) -> Mask {
    let s = vol.index(seed[0].min(vol.dims[0] - 1), seed[1].min(vol.dims[1] - 1), seed[2].min(vol.dims[2] - 1));
    flood(vol.dims, s, conn, |n| {
        let v = vol.values[n];
        v >= lo && v <= hi && within.is_none_or(|w| w.data[n] != 0)
    })
}

pub(crate) fn components(mask: &Mask, conn: Connectivity) -> (Vec<u32>, Vec<usize>) {
    let offsets = conn.offsets();
    let mut label = vec![0u32; mask.len()];
    let mut sizes = vec![0usize];
    let mut stack = Vec::new();
    for start in 0..mask.len() {
        if mask.data[start] == 0 || label[start] != 0 {
            continue;
        }
        let id = sizes.len() as u32;
        let mut size = 1;
        label[start] = id;
        stack.push(start);
        while let Some(n) = stack.pop() {
            neighbours(mask.dims, n, &offsets, |m| {
                if mask.data[m] != 0 && label[m] == 0 {
                    label[m] = id;
                    size += 1;
                    stack.push(m);
                }
            });
        }
        sizes.push(size);
    }
    (label, sizes)
}

pub(crate) fn keep_largest(mask: &mut Mask, conn: Connectivity) {
    let (label, sizes) = components(mask, conn);
    let Some(best) = (1..sizes.len()).max_by_key(|&c| sizes[c]) else { return };
    mask.data.par_iter_mut().zip(&label).for_each(|(m, &l)| *m = (l == best as u32) as u8);
}

pub(crate) fn island_at(mask: &mut Mask, seed: [usize; 3], keep: bool, conn: Connectivity) {
    let s = mask.index(seed[0], seed[1], seed[2]);
    let island = flood(mask.dims, s, conn, |n| mask.data[n] != 0);
    if keep {
        *mask = island;
    } else {
        mask.subtract(&island);
    }
}

pub(crate) fn fill_holes(mask: &mut Mask) {
    let dims = mask.dims;
    let offsets = Connectivity::Six.offsets();
    let mut outside = vec![0u8; mask.len()];
    let mut stack = Vec::new();
    for n in 0..mask.len() {
        let [i, j, k] = mask.ijk(n);
        let edge = i == 0 || j == 0 || k == 0 || i + 1 == dims[0] || j + 1 == dims[1] || k + 1 == dims[2];
        if edge && mask.data[n] == 0 && outside[n] == 0 {
            outside[n] = 1;
            stack.push(n);
            while let Some(p) = stack.pop() {
                neighbours(dims, p, &offsets, |m| {
                    if mask.data[m] == 0 && outside[m] == 0 {
                        outside[m] = 1;
                        stack.push(m);
                    }
                });
            }
        }
    }
    mask.data.par_iter_mut().zip(&outside).for_each(|(m, &o)| *m = (o == 0) as u8);
}

fn dt1d(f: &[f64], h: f64, out: &mut [f64]) {
    let n = f.len();
    let sites: Vec<usize> = (0..n).filter(|&q| f[q].is_finite()).collect();
    if sites.is_empty() {
        out.iter_mut().for_each(|o| *o = f64::INFINITY);
        return;
    }
    let mut v: Vec<usize> = Vec::with_capacity(sites.len());
    let mut z: Vec<f64> = Vec::with_capacity(sites.len() + 1);
    let x = |q: usize| q as f64 * h;
    for &q in &sites {
        let mut s;
        loop {
            match v.last() {
                None => {
                    s = f64::NEG_INFINITY;
                    break;
                }
                Some(&p) => {
                    s = ((f[q] + x(q) * x(q)) - (f[p] + x(p) * x(p))) / (2.0 * (x(q) - x(p)));
                    if s <= *z.last().unwrap() {
                        v.pop();
                        z.pop();
                    } else {
                        break;
                    }
                }
            }
        }
        v.push(q);
        z.push(s);
    }
    let mut k = 0;
    for (q, o) in out.iter_mut().enumerate() {
        while k + 1 < v.len() && z[k + 1] < x(q) {
            k += 1;
        }
        let d = x(q) - x(v[k]);
        *o = d * d + f[v[k]];
    }
}

pub(crate) fn distance_sq(dims: [usize; 3], site: &[u8], spacing: [f64; 3]) -> Vec<f64> {
    let [nx, ny, nz] = dims;
    let plane = nx * ny;
    let mut d: Vec<f64> = site.par_iter().map(|&s| if s != 0 { 0.0 } else { f64::INFINITY }).collect();
    d.par_chunks_mut(plane).for_each(|p| {
        let mut out = vec![0.0; nx.max(ny)];
        for j in 0..ny {
            let row = &mut p[j * nx..(j + 1) * nx];
            dt1d(row, spacing[0], &mut out[..nx]);
            row.copy_from_slice(&out[..nx]);
        }
        let mut col = vec![0.0; ny];
        for i in 0..nx {
            for j in 0..ny {
                col[j] = p[j * nx + i];
            }
            dt1d(&col, spacing[1], &mut out[..ny]);
            for j in 0..ny {
                p[j * nx + i] = out[j];
            }
        }
    });
    if nz > 1 {
        let cols: Vec<(usize, Vec<f64>)> = (0..ny)
            .into_par_iter()
            .flat_map_iter(|j| {
                let d = &d;
                (0..nx).map(move |i| {
                    let col: Vec<f64> = (0..nz).map(|k| d[k * plane + j * nx + i]).collect();
                    let mut out = vec![0.0; nz];
                    dt1d(&col, spacing[2], &mut out);
                    (j * nx + i, out)
                })
            })
            .collect();
        for (at, col) in cols {
            for (k, v) in col.into_iter().enumerate() {
                d[k * plane + at] = v;
            }
        }
    }
    d
}

pub(crate) fn grow(mask: &mut Mask, mm: f64, spacing: [f64; 3]) {
    if mm <= 0.0 {
        return;
    }
    let d = distance_sq(mask.dims, &mask.data, spacing);
    let r2 = mm * mm + 1e-9;
    mask.data.par_iter_mut().zip(&d).for_each(|(m, &x)| *m = (x <= r2) as u8);
}

pub(crate) fn shrink(mask: &mut Mask, mm: f64, spacing: [f64; 3]) {
    if mm <= 0.0 {
        return;
    }
    let outside: Vec<u8> = mask.data.par_iter().map(|&m| (m == 0) as u8).collect();
    let d = distance_sq(mask.dims, &outside, spacing);
    let r2 = mm * mm + 1e-9;
    mask.data.par_iter_mut().zip(&d).for_each(|(m, &x)| *m = (*m != 0 && x > r2) as u8);
}

pub(crate) fn majority(mask: &mut Mask, r: usize) {
    if r == 0 {
        return;
    }
    let [nx, ny, nz] = mask.dims;
    let plane = nx * ny;
    let box_sum = |src: &[u16], len: usize, at: &(dyn Fn(usize, usize) -> usize + Sync), lines: usize| -> Vec<(usize, u16)> {
        (0..lines)
            .into_par_iter()
            .flat_map_iter(|l| {
                let mut acc = 0u32;
                let mut out = Vec::with_capacity(len);
                for x in 0..(len + r) {
                    if x < len {
                        acc += src[at(l, x)] as u32;
                    }
                    if x > 2 * r {
                        acc -= src[at(l, x - 2 * r - 1)] as u32;
                    }
                    if x >= r {
                        out.push((at(l, x - r), acc as u16));
                    }
                }
                out
            })
            .collect()
    };
    let mut s: Vec<u16> = mask.data.iter().map(|&v| v as u16).collect();
    type Pass = (usize, usize, Box<dyn Fn(usize, usize) -> usize + Sync>);
    let passes: [Pass; 3] = [
        (nx, ny * nz, Box::new(move |l, x| l * nx + x)),
        (ny, nx * nz, Box::new(move |l, y| (l / nx) * plane + y * nx + l % nx)),
        (nz, plane, Box::new(move |l, z| z * plane + l)),
    ];
    for (len, lines, at) in passes.iter() {
        let sums = box_sum(&s, *len, at.as_ref(), *lines);
        for (n, v) in sums {
            s[n] = v;
        }
    }
    let full = ((2 * r + 1) as u32).pow(3);
    mask.data.par_iter_mut().zip(&s).for_each(|(m, &c)| *m = (c as u32 * 2 > full) as u8);
}

/// mesh.rs's Gaussian: unnormalised f32 weights, each sum from +0 divided by their total.
pub(crate) fn mesh_gaussian(values: &mut [f32], dims: [usize; 3], sigma: [f64; 3]) {
    let [nx, ny, nz] = dims;
    let plane = nx * ny;
    for axis in 0..3 {
        let s = sigma[axis];
        let len = dims[axis];
        if s < 0.2 || len < 2 {
            continue;
        }
        let r = (3.0 * s).ceil() as i64;
        let w: Vec<f32> = (-r..=r).map(|x| (-(x * x) as f64 / (2.0 * s * s)).exp() as f32).collect();
        let total: f32 = w.iter().sum();
        let src = values.to_vec();
        let (stride, lines): (usize, Vec<usize>) = match axis {
            0 => (1, (0..ny * nz).map(|l| l * nx).collect()),
            1 => (nx, (0..nx * nz).map(|l| (l / nx) * plane + l % nx).collect()),
            _ => (plane, (0..plane).collect()),
        };
        let out: Vec<(usize, Vec<f32>)> = lines
            .par_iter()
            .map(|&start| {
                let line: Vec<f32> = (0..len)
                    .map(|x| {
                        let mut acc = 0.0;
                        for (t, &wt) in w.iter().enumerate() {
                            let y = (x as i64 + t as i64 - r).clamp(0, len as i64 - 1) as usize;
                            acc += wt * src[start + y * stride];
                        }
                        acc / total
                    })
                    .collect();
                (start, line)
            })
            .collect();
        for (start, line) in out {
            for (x, v) in line.into_iter().enumerate() {
                values[start + x * stride] = v;
            }
        }
    }
}

/// The per-voxel blur of grow.rs (`levelset` = false) and levelset.rs (true), which differ in
/// when an axis is left alone.
pub(crate) fn voxel_blur(values: &mut [f32], dims: [usize; 3], sigma: [f64; 3], levelset: bool) {
    let strides = [1, dims[0], dims[0] * dims[1]];
    for ((&s, &len), &stride) in sigma.iter().zip(&dims).zip(&strides) {
        let r = ((3.0 * s).ceil() as usize).min(2 * len);
        let skip = if levelset { s.is_nan() || s < 0.05 || len < 2 } else { s <= 0.0 || r == 0 || len < 2 };
        if skip {
            continue;
        }
        let w: Vec<f64> = (0..=2 * r).map(|t| (-((t as f64 - r as f64).powi(2)) / (2.0 * s * s)).exp()).collect();
        let total: f64 = w.iter().sum();
        let w: Vec<f32> = w.iter().map(|x| (x / total) as f32).collect();
        let src = values.to_vec();
        values.par_iter_mut().enumerate().for_each(|(n, v)| {
            let x = (n / stride) % len;
            let base = n - x * stride;
            *v = w.iter().enumerate().map(|(t, &wt)| wt * src[base + (x + t).saturating_sub(r).min(len - 1) * stride]).sum();
        });
    }
}

/// The per-voxel gradient magnitude of grow.rs and levelset.rs.
pub(crate) fn voxel_gradient(values: &[f32], dims: [usize; 3], spacing: [f64; 3]) -> Vec<f32> {
    let strides = [1, dims[0], dims[0] * dims[1]];
    (0..values.len())
        .into_par_iter()
        .map(|n| {
            let mut g2 = 0.0f32;
            for ((&len, &s), &h) in dims.iter().zip(&strides).zip(&spacing) {
                let x = (n / s) % len;
                let (lo, hi) = (if x > 0 { n - s } else { n }, if x + 1 < len { n + s } else { n });
                if lo != hi {
                    let d = (values[hi] - values[lo]) / (((hi - lo) / s) as f64 * h) as f32;
                    g2 += d * d;
                }
            }
            g2.sqrt()
        })
        .collect()
}

/// smooth.rs's blur as it was.
pub(crate) fn smooth_blur(values: &mut [f32], dims: [usize; 3], sigma: [f64; 3]) {
    let kernel = |s: f64| -> Vec<f32> {
        let r = (3.0 * s).ceil() as i64;
        let w: Vec<f64> = (-r..=r).map(|x| (-((x * x) as f64) / (2.0 * s * s)).exp()).collect();
        let total: f64 = w.iter().sum();
        w.iter().map(|&v| (v / total) as f32).collect()
    };
    let [nx, ny, nz] = dims;
    if values.is_empty() {
        return;
    }
    let mut src = vec![0f32; values.len()];
    for axis in 0..3 {
        if sigma[axis].is_nan() || sigma[axis] < 0.05 || dims[axis] < 2 {
            continue;
        }
        let w = kernel(sigma[axis].min(dims[axis] as f64));
        let r = w.len() / 2;
        src.copy_from_slice(values);
        if axis == 0 {
            values.par_chunks_mut(nx).zip(src.par_chunks(nx)).for_each(|(out, line)| {
                for (x, o) in out.iter_mut().enumerate() {
                    *o = w.iter().enumerate().map(|(t, &wt)| wt * line[(x + t).saturating_sub(r).min(nx - 1)]).sum();
                }
            });
        } else {
            let (len, step) = if axis == 1 { (ny, 1) } else { (nz, ny) };
            values.par_chunks_mut(nx).enumerate().for_each(|(l, out)| {
                let p = if axis == 1 { l % ny } else { l / ny };
                let base = l - p * step;
                out.fill(0.0);
                for (t, &wt) in w.iter().enumerate() {
                    let q = base + (p + t).saturating_sub(r).min(len - 1) * step;
                    for (o, &v) in out.iter_mut().zip(&src[q * nx..(q + 1) * nx]) {
                        *o += wt * v;
                    }
                }
            });
        }
    }
}

/// Deterministic noise in [0, 1) from a number (the SplitMix64 finaliser).
pub(crate) fn unit(n: u64) -> f64 {
    let mut x = n.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    (x >> 11) as f64 / (1u64 << 53) as f64
}

/// A mask of `dims` with a few balls and speckle, from `seed`: balls of 1.5 to 6 voxels
/// placed anywhere (often cut by the grid's edges), and about one voxel in `speckle` set alone.
pub(crate) fn blobs(dims: [usize; 3], seed: u64, balls: usize, speckle: u64) -> Mask {
    let mut m = Mask::new(dims);
    let mut k = seed.wrapping_mul(1000);
    let mut next = || {
        k += 1;
        unit(k)
    };
    let shapes: Vec<([f64; 3], f64)> = (0..balls).map(|_| ([0, 1, 2].map(|a| next() * dims[a] as f64), 1.5 + 4.5 * next())).collect();
    for n in 0..m.len() {
        let p = m.ijk(n).map(|x| x as f64);
        let inside = shapes.iter().any(|(c, r)| (0..3).map(|a| (p[a] - c[a]).powi(2)).sum::<f64>() <= r * r);
        let lone = speckle > 0 && unit(seed ^ (n as u64).wrapping_mul(7919)) < 1.0 / speckle as f64;
        m.data[n] = (inside || lone) as u8;
    }
    m
}

/// Grid sizes for equality tests: thin, flat, odd and single-voxel ones among them.
pub(crate) const DIMS: [[usize; 3]; 9] = [[1, 1, 1], [1, 7, 1], [9, 1, 1], [12, 9, 1], [1, 6, 11], [7, 1, 9], [17, 13, 11], [24, 20, 16], [31, 27, 23]];

/// Spacings for equality tests, including ones whose multiples round unevenly.
pub(crate) const SPACINGS: [[f64; 3]; 4] = [[1.0, 1.0, 1.0], [0.7, 0.43, 1.3], [0.48828125, 0.48828125, 0.625], [0.48828124999999994, 0.4882812500000001, 0.7000000000000001]];
