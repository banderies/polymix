//! Segmentation operations on masks: threshold, region growing (flood fill within an
//! intensity range), islands (connected components), margins (grow / shrink by a distance in
//! millimetres through exact Euclidean distance transforms), opening / closing, majority
//! smoothing and hole filling.

use crate::mask::{Connectivity, Mask};
use mm_volume::load::LoadedVolume;
use rayon::prelude::*;

/// Voxels with values in `[lo, hi]`.
pub fn threshold(vol: &LoadedVolume, lo: f32, hi: f32) -> Mask {
    Mask { dims: vol.dims, data: vol.values.par_iter().map(|&v| (v >= lo && v <= hi) as u8).collect() }
}

/// Otsu's threshold: the value splitting the finite `values` into two classes with the most
/// variance between them (over a 256-bin histogram). `None` when they are all equal.
pub fn otsu(values: &[f32]) -> Option<f32> {
    let (lo, hi) = values.iter().filter(|v| v.is_finite()).fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
    if hi <= lo {
        return None;
    }
    const BINS: usize = 256;
    let scale = (BINS - 1) as f64 / (hi - lo) as f64;
    let mut hist = [0u64; BINS];
    for &v in values.iter().filter(|v| v.is_finite()) {
        hist[((v - lo) as f64 * scale) as usize] += 1;
    }
    let total: u64 = hist.iter().sum();
    let sum: f64 = hist.iter().enumerate().map(|(k, &c)| k as f64 * c as f64).sum();
    // The best split, and how far it stays best (empty bins between the classes).
    let (mut w0, mut sum0, mut best, mut first, mut last) = (0u64, 0.0f64, -1.0f64, 0usize, 0usize);
    for (k, &c) in hist.iter().enumerate().take(BINS - 1) {
        w0 += c;
        sum0 += k as f64 * c as f64;
        let w1 = total - w0;
        if w0 == 0 || w1 == 0 {
            continue;
        }
        let (m0, m1) = (sum0 / w0 as f64, (sum - sum0) / w1 as f64);
        let between = w0 as f64 * w1 as f64 * (m0 - m1) * (m0 - m1);
        if between > best * (1.0 + 1e-12) {
            (best, first, last) = (between, k, k);
        } else if between >= best * (1.0 - 1e-12) && last + 1 == k {
            last = k;
        }
    }
    // After bin `first`, or in the middle of a gap of empty bins.
    Some(lo + ((first + last) as f64 * 0.5 + 1.0) as f32 / scale as f32)
}

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

/// The region connected to `seed` whose voxels pass `inside` (by index), as a mask.
fn flood(dims: [usize; 3], seed: usize, conn: Connectivity, inside: impl Fn(usize) -> bool) -> Mask {
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

/// Region growing: the voxels connected to `seed` with values in `[lo, hi]` (and inside
/// `within`, if given). Empty when the seed itself is out of range.
pub fn region_grow(vol: &LoadedVolume, seed: [usize; 3], lo: f32, hi: f32, within: Option<&Mask>, conn: Connectivity) -> Mask {
    let s = vol.index(seed[0].min(vol.dims[0] - 1), seed[1].min(vol.dims[1] - 1), seed[2].min(vol.dims[2] - 1));
    flood(vol.dims, s, conn, |n| {
        let v = vol.values[n];
        v >= lo && v <= hi && within.is_none_or(|w| w.data[n] != 0)
    })
}

/// Connected components of a mask: a component number per voxel (0 = not in the mask,
/// components from 1) and the voxel count of each (index 0 unused).
pub fn components(mask: &Mask, conn: Connectivity) -> (Vec<u32>, Vec<usize>) {
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

/// Keep only the largest island.
pub fn keep_largest(mask: &mut Mask, conn: Connectivity) {
    let (label, sizes) = components(mask, conn);
    let Some(best) = (1..sizes.len()).max_by_key(|&c| sizes[c]) else { return };
    mask.data.par_iter_mut().zip(&label).for_each(|(m, &l)| *m = (l == best as u32) as u8);
}

/// Remove islands smaller than `min_voxels`; returns how many were removed.
pub fn remove_small(mask: &mut Mask, min_voxels: usize, conn: Connectivity) -> usize {
    let (label, sizes) = components(mask, conn);
    let removed = sizes.iter().skip(1).filter(|&&s| s < min_voxels).count();
    mask.data.par_iter_mut().zip(&label).for_each(|(m, &l)| {
        if l != 0 && sizes[l as usize] < min_voxels {
            *m = 0;
        }
    });
    removed
}

/// Keep (or, with `keep = false`, remove) the island containing `seed`.
pub fn island_at(mask: &mut Mask, seed: [usize; 3], keep: bool, conn: Connectivity) {
    let s = mask.index(seed[0], seed[1], seed[2]);
    let island = flood(mask.dims, s, conn, |n| mask.data[n] != 0);
    if keep {
        *mask = island;
    } else {
        mask.subtract(&island);
    }
}

/// Fill cavities: background regions not connected to the edge of the grid.
pub fn fill_holes(mask: &mut Mask) {
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

/// 1D squared distance transform (Felzenszwalb & Huttenlocher) of `f` sampled `h` apart;
/// infinite entries are not sites.
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

/// Squared Euclidean distance (mm²) from every voxel to the nearest voxel where `site` is
/// set, for voxels `spacing` millimetres apart. Infinite when there are no sites.
pub fn distance_sq(dims: [usize; 3], site: &[u8], spacing: [f64; 3]) -> Vec<f64> {
    let [nx, ny, nz] = dims;
    let plane = nx * ny;
    let mut d: Vec<f64> = site.par_iter().map(|&s| if s != 0 { 0.0 } else { f64::INFINITY }).collect();
    // Along i and j, plane by plane.
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
    // Along k, row by row.
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

/// Grow the mask by `mm` (every voxel within that distance of it joins).
pub fn grow(mask: &mut Mask, mm: f64, spacing: [f64; 3]) {
    if mm <= 0.0 {
        return;
    }
    let d = distance_sq(mask.dims, &mask.data, spacing);
    let r2 = mm * mm + 1e-9;
    mask.data.par_iter_mut().zip(&d).for_each(|(m, &x)| *m = (x <= r2) as u8);
}

/// Shrink the mask by `mm` (voxels within that distance of the outside leave).
pub fn shrink(mask: &mut Mask, mm: f64, spacing: [f64; 3]) {
    if mm <= 0.0 {
        return;
    }
    let outside: Vec<u8> = mask.data.par_iter().map(|&m| (m == 0) as u8).collect();
    let d = distance_sq(mask.dims, &outside, spacing);
    let r2 = mm * mm + 1e-9;
    mask.data.par_iter_mut().zip(&d).for_each(|(m, &x)| *m = (*m != 0 && x > r2) as u8);
}

/// Remove details thinner than about `2 · mm` (shrink, then grow).
pub fn open(mask: &mut Mask, mm: f64, spacing: [f64; 3]) {
    shrink(mask, mm, spacing);
    grow(mask, mm, spacing);
}

/// Close gaps narrower than about `2 · mm` (grow, then shrink).
pub fn close(mask: &mut Mask, mm: f64, spacing: [f64; 3]) {
    grow(mask, mm, spacing);
    shrink(mask, mm, spacing);
}

/// Majority (binary median) filter over a box of `2 · r + 1` voxels per side.
pub fn majority(mask: &mut Mask, r: usize) {
    if r == 0 {
        return;
    }
    let [nx, ny, nz] = mask.dims;
    let plane = nx * ny;
    // Box sums, one axis at a time.
    let box_sum = |src: &[u16], len: usize, at: &(dyn Fn(usize, usize) -> usize + Sync), lines: usize| -> Vec<(usize, u16)> {
        (0..lines)
            .into_par_iter()
            .flat_map_iter(|l| {
                let mut acc = 0u32;
                let mut out = Vec::with_capacity(len);
                // Running window [x - r, x + r].
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
    // (line length, lines, index of position x on line l).
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

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec3;
    use mm_volume::geometry::Geometry;

    fn ball_volume(n: usize, r: f64) -> LoadedVolume {
        let c = (n as f64 - 1.0) * 0.5;
        let mut values = Vec::with_capacity(n * n * n);
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let d = DVec3::new(i as f64 - c, j as f64 - c, k as f64 - c).length();
                    values.push(if d <= r { 100.0 } else { 0.0 });
                }
            }
        }
        LoadedVolume::from_values("ball", [n; 3], values, Geometry::from_axes([n, n, n, 1], [1.0; 3], [DVec3::X, DVec3::Y, DVec3::Z], DVec3::ZERO))
    }

    #[test]
    fn threshold_grow_and_islands() {
        let mut vol = ball_volume(21, 5.0);
        // A second, separate blob of 2 voxels.
        let a = vol.index(1, 1, 1);
        let b = vol.index(2, 1, 1);
        vol.values[a] = 100.0;
        vol.values[b] = 100.0;
        let m = threshold(&vol, 50.0, 200.0);
        let ball = m.count() - 2;
        assert!((ball as f64 - 4.0 / 3.0 * std::f64::consts::PI * 125.0).abs() < 60.0, "{ball}");
        let g = region_grow(&vol, [10, 10, 10], 50.0, 200.0, None, Connectivity::Six);
        assert_eq!(g.count(), ball);
        assert_eq!(region_grow(&vol, [0, 0, 0], 50.0, 200.0, None, Connectivity::Six).count(), 0);
        let (_, sizes) = components(&m, Connectivity::Six);
        assert_eq!(sizes.len(), 3);
        let mut k = m.clone();
        keep_largest(&mut k, Connectivity::Six);
        assert_eq!(k.count(), ball);
        let mut r = m.clone();
        assert_eq!(remove_small(&mut r, 3, Connectivity::Six), 1);
        assert_eq!(r.count(), ball);
        let mut i = m.clone();
        island_at(&mut i, [1, 1, 1], false, Connectivity::Six);
        assert_eq!(i.count(), ball);
        island_at(&mut i, [1, 1, 1], true, Connectivity::Six);
        assert_eq!(i.count(), 0);
    }

    #[test]
    fn otsu_splits_two_classes() {
        let mut v: Vec<f32> = (0..1000).map(|k| 20.0 + (k % 7) as f32).collect();
        v.extend((0..400).map(|k| 200.0 + (k % 11) as f32));
        let t = otsu(&v).unwrap();
        assert!(v.iter().all(|&x| (x < t) == (x < 100.0)), "{t}");
        assert!(t > 60.0 && t < 160.0, "the middle of the gap: {t}");
        assert_eq!(otsu(&[3.0; 10]), None);
    }

    #[test]
    fn distances_match_brute_force() {
        let dims = [7, 5, 4];
        let spacing = [0.5, 1.0, 2.0];
        let mut site = vec![0u8; 140];
        for n in [3usize, 40, 101, 139] {
            site[n] = 1;
        }
        let d = distance_sq(dims, &site, spacing);
        let pos = |n: usize| DVec3::new((n % 7) as f64 * 0.5, ((n / 7) % 5) as f64, (n / 35) as f64 * 2.0);
        for (n, &dn) in d.iter().enumerate() {
            let best = (0..140).filter(|&m| site[m] != 0).map(|m| (pos(n) - pos(m)).length_squared()).fold(f64::INFINITY, f64::min);
            assert!((dn - best).abs() < 1e-9, "{n}: {dn} vs {best}");
        }
        assert!(distance_sq(dims, &[0; 140], spacing).iter().all(|d| d.is_infinite()));
    }

    #[test]
    fn margins_holes_and_smoothing() {
        let vol = ball_volume(31, 6.0);
        let m = threshold(&vol, 50.0, 200.0);
        let n0 = m.count();
        let mut g = m.clone();
        grow(&mut g, 2.0, [1.0; 3]);
        let mut s = g.clone();
        shrink(&mut s, 2.0, [1.0; 3]);
        // Growing by 2 mm adds every voxel within 2 mm of the ball (brute force).
        let on: Vec<usize> = (0..m.len()).filter(|&n| m.data[n] != 0).collect();
        let p = |n: usize| { let [i, j, k] = m.ijk(n); DVec3::new(i as f64, j as f64, k as f64) };
        let brute = (0..m.len()).filter(|&n| on.iter().any(|&q| (p(n) - p(q)).length_squared() <= 4.0 + 1e-9)).count();
        assert_eq!(g.count(), brute);
        assert!((s.count() as i64 - n0 as i64).abs() < (n0 / 20) as i64, "{} vs {n0}", s.count());
        // A hollow ball fills.
        let mut hollow = g.clone();
        hollow.subtract(&m);
        fill_holes(&mut hollow);
        assert_eq!(hollow.count(), g.count());
        // Majority removes a lone voxel and keeps the ball.
        let mut noisy = m.clone();
        noisy.set(1, 1, 1, true);
        majority(&mut noisy, 1);
        assert!(!noisy.get(1, 1, 1));
        assert!(noisy.get(15, 15, 15));
        assert!((noisy.count() as i64 - n0 as i64).abs() < (n0 / 10) as i64);
        // Opening removes a thin spike, closing fills a 1-voxel gap.
        let mut spiky = m.clone();
        for i in 0..10 {
            spiky.set(i, 15, 15, true);
        }
        open(&mut spiky, 1.5, [1.0; 3]);
        assert!(!spiky.get(2, 15, 15));
        let mut gap = m.clone();
        gap.set(15, 15, 15, false);
        close(&mut gap, 1.5, [1.0; 3]);
        assert!(gap.get(15, 15, 15));
    }
}
