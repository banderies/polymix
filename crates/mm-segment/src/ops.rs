//! Segmentation operations on masks: threshold, region growing (flood fill within an
//! intensity range), islands (connected components), margins (grow / shrink by a distance in
//! millimetres through exact Euclidean distance transforms), opening / closing, majority
//! smoothing and hole filling.
//!
//! Most of them only change voxels near the mask, and work on its bounding box grown by their
//! reach. Connected regions are filled a run of voxels along i at a time (scanline filling),
//! which visits each voxel a few times instead of testing its neighbours one by one.

use crate::mask::{Connectivity, Mask};
use crate::region::Region;
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

/// The rows next to a row of voxels, as (Δj, Δk), and whether a run's neighbours in them reach
/// a voxel past its ends (diagonally, with 26-connectivity).
fn row_steps(conn: Connectivity) -> (&'static [(isize, isize)], bool) {
    match conn {
        Connectivity::Six => (&[(0, -1), (-1, 0), (1, 0), (0, 1)], false),
        Connectivity::TwentySix => (&[(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)], true),
    }
}

/// Scanline flood fill of a grid of `dims` voxels: every voxel connected (by `conn`) to voxel
/// `seed` through voxels that `open` admits (by index) is passed to `mark` once, and `mark`
/// must make `open` refuse it from then on. Each run of open voxels along i is taken whole, and
/// the rows next to it are scanned for the runs it reaches, one of whose voxels goes on the
/// stack (`stack` is scratch).
fn fill_runs<S: ?Sized>(dims: [usize; 3], seed: [usize; 3], conn: Connectivity, s: &mut S, open: impl Fn(&S, usize) -> bool, mut mark: impl FnMut(&mut S, usize), stack: &mut Vec<[u32; 3]>) {
    let [bx, by, bz] = dims;
    let start = |j: usize, k: usize| (k * by + j) * bx;
    let (steps, wide) = row_steps(conn);
    stack.clear();
    stack.push(seed.map(|x| x as u32));
    while let Some([x, j, k]) = stack.pop() {
        let (x, j, k) = (x as usize, j as usize, k as usize);
        let row = start(j, k);
        if !open(s, row + x) {
            continue;
        }
        let (mut a, mut e) = (x, x);
        while a > 0 && open(s, row + a - 1) {
            a -= 1;
        }
        while e + 1 < bx && open(s, row + e + 1) {
            e += 1;
        }
        for q in a..=e {
            mark(s, row + q);
        }
        let (lo, hi) = if wide { (a.saturating_sub(1), (e + 1).min(bx - 1)) } else { (a, e) };
        for &(dj, dk) in steps {
            let (Some(nj), Some(nk)) = (j.checked_add_signed(dj), k.checked_add_signed(dk)) else { continue };
            if nj >= by || nk >= bz {
                continue;
            }
            let next = start(nj, nk);
            let mut run = false;
            for q in lo..=hi {
                let o = open(s, next + q);
                if o && !run {
                    stack.push([q as u32, nj as u32, nk as u32]);
                }
                run = o;
            }
        }
    }
}

/// The region connected to `seed` whose voxels pass `inside` (by index), as a mask.
fn flood(dims: [usize; 3], seed: usize, conn: Connectivity, inside: impl Fn(usize) -> bool) -> Mask {
    let mut out = Mask::new(dims);
    if seed >= out.len() || !inside(seed) {
        return out;
    }
    let g = out.ijk(seed);
    fill_runs(dims, g, conn, &mut out.data, |o, n| o[n] == 0 && inside(n), |o, n| o[n] = 1, &mut Vec::new());
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

/// The first voxel at or after `n` where `data` is not 0, skipping zeros eight at a time.
fn next_set(data: &[u8], mut n: usize) -> Option<usize> {
    while let Some(w) = data.get(n..n + 8) {
        if u64::from_ne_bytes(w.try_into().unwrap()) != 0 {
            break;
        }
        n += 8;
    }
    data.get(n..)?.iter().position(|&v| v != 0).map(|p| p + n)
}

/// Connected components of a mask: a component number per voxel (0 = not in the mask,
/// components from 1) and the voxel count of each (index 0 unused). Components are numbered in
/// the order of their first voxel.
pub fn components(mask: &Mask, conn: Connectivity) -> (Vec<u32>, Vec<usize>) {
    let mut label = vec![0u32; mask.len()];
    let mut sizes = vec![0usize];
    let mut stack = Vec::new();
    // The labels, the component's number and its size so far.
    let mut st = (label.as_mut_slice(), 0u32, 0usize);
    let mut at = 0;
    while let Some(n) = next_set(&mask.data, at) {
        at = n + 1;
        if st.0[n] != 0 {
            continue;
        }
        (st.1, st.2) = (sizes.len() as u32, 0);
        let mark = |s: &mut (&mut [u32], u32, usize), q: usize| {
            s.0[q] = s.1;
            s.2 += 1;
        };
        fill_runs(mask.dims, mask.ijk(n), conn, &mut st, |s, q| mask.data[q] != 0 && s.0[q] == 0, mark, &mut stack);
        sizes.push(st.2);
    }
    (label, sizes)
}

/// States of voxels while islands or the outside are found: not reached (or not in the mask),
/// in the mask, reached, and in the island kept.
const EMPTY: u8 = 0;
const SET: u8 = 1;
const REACHED: u8 = 2;
const KEPT: u8 = 3;

/// Fill from voxel `seed` of a grid of `dims` through voxels of state `from`, setting them to
/// `to`.
fn fill_state(st: &mut [u8], dims: [usize; 3], seed: [usize; 3], conn: Connectivity, from: u8, to: u8, stack: &mut Vec<[u32; 3]>) {
    fill_runs(dims, seed, conn, st, |s, n| s[n] == from, |s, n| s[n] = to, stack);
}

/// Keep only the largest island (of those as large, the one whose first voxel comes last).
pub fn keep_largest(mask: &mut Mask, conn: Connectivity) {
    let mut st = vec![EMPTY; mask.len()];
    let mut stack = Vec::new();
    // The voxels of the largest island so far are marked `kept`, those of the island being
    // measured `other`, and the rest of the mask, once reached, REACHED. When the island
    // measured is the larger, the two marks change places (and the island it beat is marked
    // REACHED), else it is marked REACHED: one large island among small ones is filled once.
    let (mut kept, mut other) = (KEPT, KEPT + 1);
    // The size of the largest island so far and a voxel of it.
    let mut best: Option<(usize, [usize; 3])> = None;
    let mut at = 0;
    while let Some(n) = next_set(&mask.data, at) {
        at = n + 1;
        if st[n] != EMPTY {
            continue;
        }
        let g = mask.ijk(n);
        let mut s = (st.as_mut_slice(), 0usize);
        let mark = |s: &mut (&mut [u8], usize), q: usize| {
            s.0[q] = other;
            s.1 += 1;
        };
        fill_runs(mask.dims, g, conn, &mut s, |s, q| mask.data[q] != 0 && s.0[q] == EMPTY, mark, &mut stack);
        let size = s.1;
        if best.is_none_or(|(most, _)| size >= most) {
            if let Some((_, beaten)) = best {
                fill_state(&mut st, mask.dims, beaten, conn, kept, REACHED, &mut stack);
            }
            best = Some((size, g));
            (kept, other) = (other, kept);
        } else {
            fill_state(&mut st, mask.dims, g, conn, other, REACHED, &mut stack);
        }
    }
    if best.is_some() {
        mask.data.par_iter_mut().zip(&st).for_each(|(m, &s)| *m = (s == kept) as u8);
    }
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
    let Some((lo, hi)) = mask.bounds() else { return };
    // Past the mask's bounding box all is background reaching the grid's edge; so is the ring of
    // voxels just round the box, and the background of the box reaches the edge exactly where it
    // reaches the box's faces (the ring, or the grid's edge where the box meets it).
    let b = Region::around(lo, hi, [1; 3], mask.dims);
    let d = b.dims;
    let mut st = b.gather(&mask.data, mask.dims, |m| if m != 0 { SET } else { EMPTY });
    let mut stack = Vec::new();
    for r in 0..d[1] * d[2] {
        let (y, z) = (r % d[1], r / d[1]);
        let face = y == 0 || z == 0 || y + 1 == d[1] || z + 1 == d[2];
        let xs: &mut dyn Iterator<Item = usize> = if face { &mut (0..d[0]) } else { &mut [0, d[0] - 1].into_iter() };
        for x in xs {
            if st[r * d[0] + x] == EMPTY {
                fill_state(&mut st, d, [x, y, z], Connectivity::Six, EMPTY, REACHED, &mut stack);
            }
        }
    }
    b.write(&mut mask.data, mask.dims, |q| st[q] != REACHED);
}

/// Scratch of [`dt1d`]: the sites of the parabolas of the lower envelope, and where each
/// parabola starts to be the lowest.
#[derive(Default)]
struct Envelope {
    v: Vec<usize>,
    z: Vec<f64>,
}

/// 1D squared distance transform (Felzenszwalb & Huttenlocher) of `f` sampled `h` apart, its
/// first sample being number `off` of the line; infinite entries are not sites.
fn dt1d(f: &[f64], h: f64, off: usize, out: &mut [f64], env: &mut Envelope) {
    let Envelope { v, z } = env;
    v.clear();
    z.clear();
    let x = |q: usize| (q + off) as f64 * h;
    for (q, &fq) in f.iter().enumerate() {
        if !fq.is_finite() {
            continue;
        }
        let mut s;
        loop {
            match v.last() {
                None => {
                    s = f64::NEG_INFINITY;
                    break;
                }
                Some(&p) => {
                    s = ((fq + x(q) * x(q)) - (f[p] + x(p) * x(p))) / (2.0 * (x(q) - x(p)));
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
    if v.is_empty() {
        out.iter_mut().for_each(|o| *o = f64::INFINITY);
        return;
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
    distance_sq_at(dims, site, spacing, [0; 3])
}

/// [`distance_sq`] of a box of `dims` voxels whose first voxel is voxel `origin` of a larger
/// grid, with coordinates measured from the grid's first voxel: where the box holds every site
/// of the grid along each line through it, the distances are the grid's to the bit.
pub(crate) fn distance_sq_at(dims: [usize; 3], site: &[u8], spacing: [f64; 3], origin: [usize; 3]) -> Vec<f64> {
    let [nx, ny, nz] = dims;
    let mut d: Vec<f64> = site.par_iter().map(|&s| if s != 0 { 0.0 } else { f64::INFINITY }).collect();
    if d.is_empty() {
        return d;
    }
    // Along i and j, plane by plane.
    d.par_chunks_mut(nx * ny).for_each_init(
        || (Envelope::default(), vec![0.0; nx.max(ny)], vec![0.0; ny]),
        |(env, out, col), p| {
            for row in p.chunks_mut(nx) {
                dt1d(row, spacing[0], origin[0], &mut out[..nx], env);
                row.copy_from_slice(&out[..nx]);
            }
            for i in 0..nx {
                for (j, c) in col.iter_mut().enumerate() {
                    *c = p[j * nx + i];
                }
                dt1d(col, spacing[1], origin[1], &mut out[..ny], env);
                for (j, &o) in out[..ny].iter().enumerate() {
                    p[j * nx + i] = o;
                }
            }
        },
    );
    // Along k: the rows of one j in every plane at a time, copied into a tile of their own.
    if nz > 1 {
        Region::all(dims).rows_by_j(&mut d, dims).into_par_iter().for_each_init(
            || (Envelope::default(), vec![0.0; nz * nx], vec![0.0; nz], vec![0.0; nz]),
            |(env, tile, col, out), mut rows| {
                for (row, t) in rows.iter().zip(tile.chunks_mut(nx)) {
                    t.copy_from_slice(row);
                }
                for i in 0..nx {
                    for (k, c) in col.iter_mut().enumerate() {
                        *c = tile[k * nx + i];
                    }
                    dt1d(col, spacing[2], origin[2], out, env);
                    for (k, &o) in out.iter().enumerate() {
                        tile[k * nx + i] = o;
                    }
                }
                for (row, t) in rows.iter_mut().zip(tile.chunks(nx)) {
                    row.copy_from_slice(t);
                }
            },
        );
    }
    d
}

/// Whether every spacing is a positive number: distances then grow away from a voxel, and a
/// margin's reach is a number of voxels along each axis.
fn even_spacing(spacing: [f64; 3]) -> bool {
    spacing.iter().all(|&s| s.is_finite() && s > 0.0)
}

/// The distances of the voxels of box `b` of the mask's grid to the voxels where `site` of
/// the mask is (a grid's worth when the box is the whole grid).
fn box_distances(mask: &Mask, b: &Region, spacing: [f64; 3], site: impl Fn(u8) -> u8 + Sync) -> Vec<f64> {
    let s = b.gather(&mask.data, mask.dims, site);
    distance_sq_at(b.dims, &s, spacing, b.lo)
}

/// Grow the mask by `mm` (every voxel within that distance of it joins).
pub fn grow(mask: &mut Mask, mm: f64, spacing: [f64; 3]) {
    if mm <= 0.0 {
        return;
    }
    let r2 = mm * mm + 1e-9;
    let Some((lo, hi)) = mask.bounds() else {
        // Nothing is near an empty mask; only an infinite margin takes in the infinitely far.
        if f64::INFINITY <= r2 {
            mask.data.par_iter_mut().for_each(|m| *m = 1);
        }
        return;
    };
    // Beyond the margin's reach and a voxel from the mask's box nothing joins, and in that box
    // (which holds every site) the distances are the whole grid's.
    let b = match even_spacing(spacing) {
        true => Region::around(lo, hi, spacing.map(|s| ((r2.sqrt() / s).ceil() as usize).saturating_add(1)), mask.dims),
        false => Region::all(mask.dims),
    };
    let d = box_distances(mask, &b, spacing, |m| m);
    let dx = b.dims[0];
    b.par_rows_mut(&mut mask.data, mask.dims).for_each(|(r, row)| {
        for (m, &x) in row.iter_mut().zip(&d[r * dx..(r + 1) * dx]) {
            *m = (x <= r2) as u8;
        }
    });
}

/// Shrink the mask by `mm` (voxels within that distance of the outside leave).
pub fn shrink(mask: &mut Mask, mm: f64, spacing: [f64; 3]) {
    if mm <= 0.0 {
        return;
    }
    let Some((lo, hi)) = mask.bounds() else { return };
    // Every voxel past the mask's box is outside it: the ring of voxels round the box holds the
    // nearest outside voxel of each voxel in the box, and the distances there are the grid's.
    let b = match even_spacing(spacing) {
        true => Region::around(lo, hi, [1; 3], mask.dims),
        false => Region::all(mask.dims),
    };
    let d = box_distances(mask, &b, spacing, |m| (m == 0) as u8);
    let r2 = mm * mm + 1e-9;
    let dx = b.dims[0];
    b.par_rows_mut(&mut mask.data, mask.dims).for_each(|(r, row)| {
        for (m, &x) in row.iter_mut().zip(&d[r * dx..(r + 1) * dx]) {
            *m = (*m != 0 && x > r2) as u8;
        }
    });
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

/// Box sums held between the passes of [`majority`]: as narrow as the largest sum allows.
trait Count: Copy + Default + Send + Sync {
    fn new(v: u64) -> Self;
    fn get(self) -> u64;
}

impl Count for u16 {
    fn new(v: u64) -> Self {
        v as u16
    }
    fn get(self) -> u64 {
        self as u64
    }
}

impl Count for u32 {
    fn new(v: u64) -> Self {
        v as u32
    }
    fn get(self) -> u64 {
        self as u64
    }
}

impl Count for u64 {
    fn new(v: u64) -> Self {
        v
    }
    fn get(self) -> u64 {
        self
    }
}

/// Majority (binary median) filter over a box of `2 · r + 1` voxels per side.
pub fn majority(mask: &mut Mask, r: usize) {
    if r == 0 {
        return;
    }
    // Beyond `r` voxels from the mask's box a box sees none of it, and stays empty.
    let Some((lo, hi)) = mask.bounds() else { return };
    let b = Region::around(lo, hi, [r; 3], mask.dims);
    let w = r.saturating_mul(2).saturating_add(1) as u64;
    // The sums over i and j of the mask's values (bytes) in a box.
    let most = 255 * w.min(b.dims[0] as u64) * w.min(b.dims[1] as u64);
    if most <= u16::MAX as u64 {
        majority_in::<u16>(mask, &b, r);
    } else if most <= u32::MAX as u64 {
        majority_in::<u32>(mask, &b, r);
    } else {
        majority_in::<u64>(mask, &b, r);
    }
}

/// Sums over a window of `r` before to `r` after of a line of values (0 past its ends), one
/// step at a time: what enters the window is added and what leaves it taken away.
fn running_sums(line: &[u8], r: usize, out: &mut [u32]) {
    let n = line.len();
    let mut s: u32 = line[..=r.min(n - 1)].iter().map(|&v| v as u32).sum();
    for (x, o) in out.iter_mut().enumerate() {
        *o = s;
        let enter = x.saturating_add(r).saturating_add(1);
        if enter < n {
            s += line[enter] as u32;
        }
        if x >= r {
            s -= line[x - r] as u32;
        }
    }
}

/// [`running_sums`] of `len` rows at once (`row(q)` the q-th, all as long as `acc`): `emit`
/// gets each row's number and its window's sums, in order.
fn running_rows<'a, V: Copy + 'a>(len: usize, r: usize, acc: &mut [u64], row: impl Fn(usize) -> &'a [V], get: impl Fn(V) -> u64, mut emit: impl FnMut(usize, &[u64])) {
    acc.fill(0);
    for q in 0..=r.min(len - 1) {
        for (a, &v) in acc.iter_mut().zip(row(q)) {
            *a += get(v);
        }
    }
    for p in 0..len {
        emit(p, acc);
        let enter = p.saturating_add(r).saturating_add(1);
        if enter < len {
            for (a, &v) in acc.iter_mut().zip(row(enter)) {
                *a += get(v);
            }
        }
        if p >= r {
            for (a, &v) in acc.iter_mut().zip(row(p - r)) {
                *a -= get(v);
            }
        }
    }
}

/// [`majority`] over the region `b` of the mask's grid (the voxels outside it stay as they
/// are): box sums along i with a running window, then along j and k with running sums of
/// whole rows. Past the grid's edges the mask counts as empty.
fn majority_in<T: Count>(mask: &mut Mask, b: &Region, r: usize) {
    let [bx, by, bz] = b.dims;
    let full = (r.saturating_mul(2).saturating_add(1) as u64).saturating_pow(3);
    let (data, dims) = (&mask.data, mask.dims);
    // Along i and j, plane by plane.
    let mut sums = vec![T::default(); b.len()];
    sums.par_chunks_mut(bx * by).enumerate().for_each_init(
        || (vec![0u32; bx * by], vec![0u64; bx]),
        |(along_i, acc), (z, plane)| {
            for (y, out) in along_i.chunks_mut(bx).enumerate() {
                let at = b.row_start(z * by + y, dims);
                running_sums(&data[at..at + bx], r, out);
            }
            let along_i = &*along_i;
            running_rows(by, r, acc, |q| &along_i[q * bx..(q + 1) * bx], |v| v as u64, |y, s| {
                for (o, &a) in plane[y * bx..(y + 1) * bx].iter_mut().zip(s) {
                    *o = T::new(a);
                }
            });
        },
    );
    // Along k, into the mask: the rows of one j in every plane at a time.
    let sums = &sums;
    b.rows_by_j(&mut mask.data, dims).into_par_iter().enumerate().for_each_init(
        || vec![0u64; bx],
        |acc, (y, mut rows)| {
            running_rows(bz, r, acc, |q| &sums[(q * by + y) * bx..(q * by + y + 1) * bx], T::get, |z, s| {
                for (m, &a) in rows[z].iter_mut().zip(s) {
                    *m = (a.saturating_mul(2) > full) as u8;
                }
            });
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference::{self, blobs, DIMS, SPACINGS};
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
        let p = |n: usize| {
            let [i, j, k] = m.ijk(n);
            DVec3::new(i as f64, j as f64, k as f64)
        };
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

    /// Masks for the equality tests: blobs with speckle on every test grid, an empty one and a
    /// full one.
    fn masks() -> Vec<Mask> {
        let mut out = Vec::new();
        for (s, &dims) in DIMS.iter().enumerate() {
            for seed in 0..3u64 {
                out.push(blobs(dims, 100 * s as u64 + seed, 1 + seed as usize * 2, [0, 40, 7][seed as usize]));
            }
            out.push(Mask::new(dims));
            out.push(Mask { dims, data: vec![1; dims.iter().product()] });
        }
        out
    }

    #[test]
    fn distances_are_the_old_ones_to_the_bit() {
        for m in masks() {
            for sp in SPACINGS {
                let (a, b) = (reference::distance_sq(m.dims, &m.data, sp), distance_sq(m.dims, &m.data, sp));
                assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()), "{:?} {sp:?}", m.dims);
            }
        }
    }

    #[test]
    fn margins_are_the_old_ones() {
        let odd = [[0.5, 0.0, 1.0], [f64::NAN, 1.0, 1.0], [-0.5, 0.7, 1.0], [f64::INFINITY, 1.0, 1.0], [1e-7, 1.0, 1.0]];
        for m in masks() {
            for sp in SPACINGS.iter().chain(&odd) {
                for mm in [0.43, 1.3, 2.1, 5.0, 1e-6, 0.0, f64::NAN, f64::INFINITY] {
                    let (mut a, mut b) = (m.clone(), m.clone());
                    reference::grow(&mut a, mm, *sp);
                    grow(&mut b, mm, *sp);
                    assert_eq!(a, b, "grow {:?} {sp:?} {mm}", m.dims);
                    let (mut a, mut b) = (m.clone(), m.clone());
                    reference::shrink(&mut a, mm, *sp);
                    shrink(&mut b, mm, *sp);
                    assert_eq!(a, b, "shrink {:?} {sp:?} {mm}", m.dims);
                }
            }
        }
        // Small segments in a big grid: their margin is worked out on their box.
        let mut m = Mask::new([120, 90, 70]);
        for (i, j, k) in [(60, 40, 30), (61, 40, 30), (10, 80, 5), (119, 0, 69)] {
            m.set(i, j, k, true);
        }
        for sp in SPACINGS {
            for mm in [0.5, 1.0, 2.5, 6.0] {
                let (mut a, mut b) = (m.clone(), m.clone());
                reference::grow(&mut a, mm, sp);
                grow(&mut b, mm, sp);
                assert_eq!(a, b);
                reference::shrink(&mut a, mm * 0.3, sp);
                shrink(&mut b, mm * 0.3, sp);
                assert_eq!(a, b);
            }
        }
    }

    #[test]
    fn majority_is_the_old_one_below_a_radius_of_20() {
        for m in masks() {
            for r in [1, 2, 3, 4, 5, 7, 19] {
                let (mut a, mut b) = (m.clone(), m.clone());
                reference::majority(&mut a, r);
                majority(&mut b, r);
                assert_eq!(a, b, "{:?} r {r}", m.dims);
            }
        }
        // Values other than 0 and 1 count as they are, as they did.
        let mut m = blobs([20, 18, 16], 5, 3, 9);
        m.data.iter_mut().enumerate().for_each(|(n, v)| *v *= 1 + (n % 3) as u8);
        for r in [1, 2, 3] {
            let (mut a, mut b) = (m.clone(), m.clone());
            reference::majority(&mut a, r);
            majority(&mut b, r);
            assert_eq!(a, b, "r {r}");
        }
    }

    #[test]
    fn majority_counts_past_65535() {
        // A full 50³ grid: at radius 20 the boxes round its middle are more than half full (the
        // old 16-bit sums wrapped round past 41³ voxels there and emptied it).
        let full = Mask { dims: [50; 3], data: vec![1; 50 * 50 * 50] };
        let (mut m, mut old) = (full.clone(), full.clone());
        majority(&mut m, 20);
        reference::majority(&mut old, 20);
        assert!(m.get(25, 25, 25) && !old.get(25, 25, 25));
        assert!((0..m.len()).all(|n| m.ijk(n).iter().any(|x| !(15..35).contains(x)) || m.data[n] == 1));
        // A brute-force count at a few voxels of a blob.
        let m = blobs([45, 44, 43], 3, 6, 5);
        let mut g = m.clone();
        majority(&mut g, 21);
        for n in (0..m.len()).step_by(997) {
            let p = m.ijk(n);
            let mut c = 0u64;
            for k in p[2].saturating_sub(21)..=(p[2] + 21).min(42) {
                for j in p[1].saturating_sub(21)..=(p[1] + 21).min(43) {
                    for i in p[0].saturating_sub(21)..=(p[0] + 21).min(44) {
                        c += m.get(i, j, k) as u64;
                    }
                }
            }
            assert_eq!(g.data[n], (2 * c > 43u64.pow(3)) as u8);
        }
    }

    #[test]
    fn islands_and_holes_are_the_old_ones() {
        let mut ms = masks();
        // Ties: islands of the same size, the larger first.
        let mut t = Mask::new([12, 10, 8]);
        for (i, j, k) in [(1, 1, 1), (2, 1, 1), (8, 1, 1), (9, 1, 1), (5, 8, 6), (5, 8, 7)] {
            t.set(i, j, k, true);
        }
        ms.push(t);
        // Shells with cavities, touching the grid's edges.
        let mut s = Mask::new([16, 15, 14]);
        for n in 0..s.len() {
            let p = s.ijk(n).map(|x| x as f64);
            let d = ((p[0] - 7.0).powi(2) + (p[1] - 7.0).powi(2) + (p[2] - 6.5).powi(2)).sqrt();
            s.data[n] = ((4.0..=6.5).contains(&d) || (p[0] == 0.0 && d < 9.0)) as u8;
        }
        ms.push(s);
        for m in ms {
            for conn in [Connectivity::Six, Connectivity::TwentySix] {
                assert_eq!(reference::components(&m, conn), components(&m, conn), "{:?}", m.dims);
                let (mut a, mut b) = (m.clone(), m.clone());
                reference::keep_largest(&mut a, conn);
                keep_largest(&mut b, conn);
                assert_eq!(a, b, "keep_largest {:?} {conn:?}", m.dims);
                for n in (0..m.len()).step_by(m.len() / 7 + 1).chain([m.len() - 1]) {
                    let seed = m.ijk(n);
                    for keep in [true, false] {
                        let (mut a, mut b) = (m.clone(), m.clone());
                        reference::island_at(&mut a, seed, keep, conn);
                        island_at(&mut b, seed, keep, conn);
                        assert_eq!(a, b, "island_at {:?} {seed:?} {keep} {conn:?}", m.dims);
                    }
                }
            }
            let (mut a, mut b) = (m.clone(), m.clone());
            reference::fill_holes(&mut a);
            fill_holes(&mut b);
            assert_eq!(a, b, "fill_holes {:?}", m.dims);
        }
        // Seeds out of the grid, and values other than 0 and 1.
        let mut m = blobs([9, 8, 7], 11, 2, 5);
        m.data.iter_mut().enumerate().for_each(|(n, v)| *v *= 1 + (n % 3) as u8);
        for (seed, keep) in [([20, 0, 0], true), ([20, 0, 0], false), ([4, 4, 3], false), ([4, 4, 3], true)] {
            let (mut a, mut b) = (m.clone(), m.clone());
            reference::island_at(&mut a, seed, keep, Connectivity::Six);
            island_at(&mut b, seed, keep, Connectivity::Six);
            assert_eq!(a, b);
        }
        let (mut a, mut b) = (m.clone(), m.clone());
        reference::fill_holes(&mut a);
        fill_holes(&mut b);
        assert_eq!(a, b);
    }

    #[test]
    fn region_growing_is_the_old_one() {
        for (s, &dims) in DIMS.iter().enumerate() {
            let n = dims.iter().product::<usize>();
            let values: Vec<f32> = (0..n).map(|q| (reference::unit(q as u64 * 31 + s as u64) * 100.0) as f32).collect();
            let vol = LoadedVolume::from_values("t", dims, values, Geometry::from_axes([dims[0], dims[1], dims[2], 1], [1.0; 3], [DVec3::X, DVec3::Y, DVec3::Z], DVec3::ZERO));
            let within = blobs(dims, s as u64, 3, 3);
            for q in (0..n).step_by(n / 5 + 1) {
                let seed = [q % dims[0], (q / dims[0]) % dims[1], q / (dims[0] * dims[1])];
                for conn in [Connectivity::Six, Connectivity::TwentySix] {
                    for (lo, hi) in [(30.0, 100.0), (0.0, 55.0)] {
                        for w in [None, Some(&within)] {
                            assert_eq!(reference::region_grow(&vol, seed, lo, hi, w, conn), region_grow(&vol, seed, lo, hi, w, conn), "{dims:?} {seed:?}");
                        }
                    }
                }
            }
        }
    }
}
