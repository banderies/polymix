//! Fill between slices: carry a segmentation drawn on some slices across the empty slices
//! between them, as 3D Slicer's "Fill between slices" effect does.
//!
//! Shape-based interpolation (Raya & Udupa 1990; Herman, Zheng & Bucholtz 1992): along the
//! chosen voxel axis, a segment's *key slices* are the slices holding any of its voxels. Each
//! slice between two consecutive key slices takes the segment where the blend of the two key
//! slices' signed distance maps, weighted by the slice's position between them, is inside
//! (≤ 0). The distances are in millimetres within the slice, from exact Euclidean distance
//! transforms (Felzenszwalb & Huttenlocher 2012, [`ops::distance_sq`](crate::ops::distance_sq))
//! of the region and of its background, with the boundary halfway between the pixel centres on
//! either side of it. Slices before the first and after the last key slice are left alone.
//!
//! Plain blending of distance maps loses regions that move: two discs that only partly overlap
//! blend into a waist, and two that do not overlap vanish in between. So each key slice is
//! first split into 8-connected regions and the regions are matched across the gap, and each
//! match is blended with its centroids aligned, in the spirit of matching the slices before
//! interpolating (Goshtasby, Turner & Ackerman 1992): each side's distance map is moved by
//! whole pixels so that its centroid sits on (the pixel nearest) the centroid interpolated for
//! the slice, the two sides staying aligned to within half a pixel. Regions that overlap the
//! other key slice's regions are matched with them (joined through any overlap, so a branching
//! vessel blends as one group); regions overlapping nothing pair with the nearest such region
//! on the other key slice, nearest pairs first; a region still alone shrinks uniformly and is
//! gone just before halfway.
//!
//! Filling only adds voxels that were empty: key slices and other segments' voxels are kept,
//! and where two segments reach the same empty voxel, the one deeper inside it (the more
//! negative blended distance) takes it.
//!
//! - S. P. Raya, J. K. Udupa, "Shape-based interpolation of multidimensional objects", IEEE
//!   Trans. Med. Imaging 9(1), 32–42 (1990).
//! - G. T. Herman, J. Zheng, C. A. Bucholtz, "Shape-based interpolation", IEEE Comput. Graph.
//!   Appl. 12(3), 69–79 (1992).
//! - A. Goshtasby, D. A. Turner, L. V. Ackerman, "Matching of tomographic slices for
//!   interpolation", IEEE Trans. Med. Imaging 11(4), 507–516 (1992).
//! - P. F. Felzenszwalb, D. P. Huttenlocher, "Distance transforms of sampled functions",
//!   Theory of Computing 8, 415–428 (2012).

use crate::mask::{Connectivity, Mask};
use crate::ops::{components, distance_sq};
use rayon::prelude::*;

/// Fill the empty slices between segmented slices of `segments` along voxel axis `axis` (0, 1
/// or 2) of the label map `labels` (`dims` voxels, `spacing` mm apart, i fastest). Returns the
/// voxels added for each of `segments` in order (label 0 and repeats skipped), as (label, voxel
/// indices in increasing order). Only voxels that are 0 in `labels` are added; the input is not
/// modified.
pub fn fill_between_slices(labels: &[u8], dims: [usize; 3], spacing: [f64; 3], axis: usize, segments: &[u8]) -> Vec<(u8, Vec<usize>)> {
    assert!(axis < 3, "no voxel axis {axis}");
    assert_eq!(labels.len(), dims[0] * dims[1] * dims[2], "label map of the wrong size");
    let mut segs: Vec<u8> = Vec::new();
    for &l in segments {
        if l != 0 && !segs.contains(&l) {
            segs.push(l);
        }
    }
    let mut out: Vec<(u8, Vec<usize>)> = segs.iter().map(|&l| (l, Vec::new())).collect();
    if labels.is_empty() || segs.is_empty() {
        return out;
    }
    let pl = Planes::new(dims, spacing, axis);
    // The labels each slice holds, as 256-bit sets, gathered plane by plane of the volume.
    let held = labels
        .par_chunks(dims[0] * dims[1])
        .enumerate()
        .fold(
            || vec![[0u64; 4]; pl.n],
            |mut acc, (k, p)| {
                for (n, &l) in p.iter().enumerate().filter(|(_, &l)| l != 0) {
                    acc[[n % dims[0], n / dims[0], k][axis]][l as usize >> 6] |= 1 << (l & 63);
                }
                acc
            },
        )
        .reduce(
            || vec![[0u64; 4]; pl.n],
            |mut a, b| {
                a.iter_mut().zip(b).for_each(|(x, y)| (0..4).for_each(|w| x[w] |= y[w]));
                a
            },
        );
    // Each gap between consecutive key slices of a segment, filled on its own.
    let gaps: Vec<(u8, usize, usize)> = segs
        .iter()
        .flat_map(|&l| {
            let keys: Vec<usize> = (0..pl.n).filter(|&s| (held[s][l as usize >> 6] >> (l & 63)) & 1 != 0).collect();
            keys.windows(2).filter(|w| w[1] > w[0] + 1).map(|w| (l, w[0], w[1])).collect::<Vec<_>>()
        })
        .collect();
    let mut claims: Vec<(usize, f32, u8)> =
        gaps.par_iter().flat_map_iter(|&(l, s0, s1)| fill_gap(labels, &pl, l, s0, s1).into_iter().map(move |(n, d)| (n, d, l))).collect();
    // Each voxel to the deepest claim on it (a segment claims a voxel at most once).
    claims.par_sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)).then(a.2.cmp(&b.2)));
    claims.dedup_by_key(|c| c.0);
    let mut slot = [0usize; 256];
    for (k, &l) in segs.iter().enumerate() {
        slot[l as usize] = k;
    }
    for (n, _, l) in claims {
        out[slot[l as usize]].1.push(n);
    }
    out
}

/// The slices across one voxel axis: slice `s`, pixel (`x`, `y`) along the other two axes in
/// order, `spacing` mm apart.
struct Planes {
    n: usize,
    nu: usize,
    nv: usize,
    /// Index steps along the axis, `x` and `y`.
    stride: [usize; 3],
    spacing: [f64; 2],
}

impl Planes {
    fn new(dims: [usize; 3], spacing: [f64; 3], axis: usize) -> Self {
        let st = [1, dims[0], dims[0] * dims[1]];
        let (u, v) = match axis {
            0 => (1, 2),
            1 => (0, 2),
            _ => (0, 1),
        };
        Self { n: dims[axis], nu: dims[u], nv: dims[v], stride: [st[axis], st[u], st[v]], spacing: [spacing[u], spacing[v]] }
    }

    /// Voxel index of pixel (`x`, `y`) of slice `s`.
    fn at(&self, s: usize, x: usize, y: usize) -> usize {
        s * self.stride[0] + x * self.stride[1] + y * self.stride[2]
    }

    /// Pixels of slice `s` that are `label`, row by row.
    fn mask(&self, labels: &[u8], s: usize, label: u8) -> Vec<u8> {
        (0..self.nv).flat_map(|y| (0..self.nu).map(move |x| (labels[self.at(s, x, y)] == label) as u8)).collect()
    }
}

/// The voxels the slices strictly between key slices `s0` and `s1` of `label` take, with their
/// blended distances.
fn fill_gap(labels: &[u8], pl: &Planes, label: u8, s0: usize, s1: usize) -> Vec<(usize, f32)> {
    let regions = regions(pl.mask(labels, s0, label), pl.mask(labels, s1, label), pl);
    (s0 + 1..s1).into_par_iter().flat_map_iter(|s| blend(&regions, (s - s0) as f64 / (s1 - s0) as f64, s, labels, pl)).collect()
}

/// Pixel count, coordinate sums and bounding box of some pixels of a slice.
#[derive(Clone, Copy)]
struct Stats {
    n: f64,
    sum: [f64; 2],
    lo: [i64; 2],
    hi: [i64; 2],
}

impl Stats {
    const NONE: Self = Self { n: 0.0, sum: [0.0; 2], lo: [i64::MAX; 2], hi: [i64::MIN; 2] };

    fn pixel(x: usize, y: usize) -> Self {
        let p = [x as i64, y as i64];
        Self { n: 1.0, sum: [x as f64, y as f64], lo: p, hi: p }
    }

    fn add(&mut self, o: &Self) {
        self.n += o.n;
        for d in 0..2 {
            self.sum[d] += o.sum[d];
            self.lo[d] = self.lo[d].min(o.lo[d]);
            self.hi[d] = self.hi[d].max(o.hi[d]);
        }
    }

    fn centre(&self) -> [f64; 2] {
        [self.sum[0] / self.n, self.sum[1] / self.n]
    }
}

fn find(parent: &mut [usize], mut k: usize) -> usize {
    while parent[k] != k {
        parent[k] = parent[parent[k]];
        k = parent[k];
    }
    k
}

fn join(parent: &mut [usize], p: usize, q: usize) {
    let (p, q) = (find(parent, p), find(parent, q));
    parent[p.max(q)] = p.min(q);
}

/// Pixel totals of each group (at its root node) on the first and on the second key slice,
/// where nodes from `na` on are the second slice's regions.
fn totals(parent: &mut [usize], stats: &[Stats], na: usize) -> Vec<[Stats; 2]> {
    let mut out = vec![[Stats::NONE; 2]; stats.len()];
    for (k, st) in stats.iter().enumerate() {
        out[find(parent, k)][(k >= na) as usize].add(st);
    }
    out
}

/// Signed distance map (mm, negative inside) of one key slice's part of a region, over a
/// window of the slice (`lo` its first pixel, `w` × `h` pixels) with two pixels of background
/// all round (pixels beyond the slice are background).
struct Field {
    phi: Vec<f64>,
    lo: [i64; 2],
    w: usize,
    h: usize,
    /// What to blend with when the region has no partner: it then shrinks uniformly and is gone
    /// just before halfway.
    end: f64,
}

impl Field {
    /// The map of the slice's pixels (by index) that are `inside`, over pixels `lo` to `hi`.
    fn new(lo: [i64; 2], hi: [i64; 2], pl: &Planes, inside: impl Fn(usize) -> bool) -> Self {
        let (w, h) = ((hi[0] - lo[0] + 1) as usize, (hi[1] - lo[1] + 1) as usize);
        let m: Vec<u8> = (0..w * h)
            .map(|q| {
                let (x, y) = (lo[0] + (q % w) as i64, lo[1] + (q / w) as i64);
                (x >= 0 && y >= 0 && (x as usize) < pl.nu && (y as usize) < pl.nv && inside(y as usize * pl.nu + x as usize)) as u8
            })
            .collect();
        let out: Vec<u8> = m.iter().map(|&v| 1 - v).collect();
        let sp = [pl.spacing[0], pl.spacing[1], 1.0];
        let (to_in, to_out) = (distance_sq([w, h, 1], &m, sp), distance_sq([w, h, 1], &out, sp));
        // The boundary lies halfway between pixel centres either side of it.
        let half = 0.5 * pl.spacing[0].min(pl.spacing[1]);
        let phi: Vec<f64> = m.iter().zip(to_in.iter().zip(&to_out)).map(|(&v, (&a, &b))| if v != 0 { half - b.sqrt() } else { a.sqrt() - half }).collect();
        let end = half - phi.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        Self { phi, lo, w, h, end }
    }

    /// The distance at pixel (`x`, `y`) of the slice, within the window.
    fn at(&self, x: i64, y: i64) -> f64 {
        self.phi[(y - self.lo[1]) as usize * self.w + (x - self.lo[0]) as usize]
    }
}

/// A group of matched regions to blend: its parts on the first and second key slices (at
/// least one present), the move of its centroid from the first to the second (pixels), and
/// that move in whole pixels.
struct Region {
    a: Option<Field>,
    b: Option<Field>,
    shift: [f64; 2],
    whole: [i64; 2],
}

/// The window of a group's distance maps, in the first key slice's pixels: the bounding boxes
/// of both parts (the second's moved back by `whole`), with two pixels to spare.
fn window(ga: &Stats, gb: &Stats, whole: [i64; 2]) -> ([i64; 2], [i64; 2]) {
    let (mut lo, mut hi) = ([i64::MAX; 2], [i64::MIN; 2]);
    for (st, back) in [(ga, [0; 2]), (gb, whole)] {
        if st.n > 0.0 {
            for d in 0..2 {
                lo[d] = lo[d].min(st.lo[d] - back[d]);
                hi[d] = hi[d].max(st.hi[d] - back[d]);
            }
        }
    }
    ([lo[0] - 2, lo[1] - 2], [hi[0] + 2, hi[1] + 2])
}

/// Match the regions of key slices `a` and `b` (masks of one segment, row by row) and make
/// their distance maps.
fn regions(a: Vec<u8>, b: Vec<u8>, pl: &Planes) -> Vec<Region> {
    // 8-connected regions of each slice (26-connectivity within a single slice).
    let dims = [pl.nu, pl.nv, 1];
    let (la, sa) = components(&Mask { dims, data: a }, Connectivity::TwentySix);
    let (lb, sb) = components(&Mask { dims, data: b }, Connectivity::TwentySix);
    let na = sa.len() - 1;
    let nodes = na + sb.len() - 1;
    // Node c - 1 is region c of the first slice, na + c - 1 region c of the second; regions
    // that overlap are joined.
    let node = |l: &[u32], n: usize, first: usize| (l[n] != 0).then(|| first + l[n] as usize - 1);
    let mut parent: Vec<usize> = (0..nodes).collect();
    let mut stats = vec![Stats::NONE; nodes];
    for n in 0..la.len() {
        let (p, q) = (node(&la, n, 0), node(&lb, n, na));
        for k in [p, q].into_iter().flatten() {
            stats[k].add(&Stats::pixel(n % pl.nu, n / pl.nu));
        }
        if let (Some(p), Some(q)) = (p, q) {
            join(&mut parent, p, q);
        }
    }
    // Groups on one slice only pair with the nearest such group on the other, nearest first.
    let sides = totals(&mut parent, &stats, na);
    let lone = |s: usize| -> Vec<usize> { (0..nodes).filter(|&r| sides[r][s].n > 0.0 && sides[r][1 - s].n == 0.0).collect() };
    let mm = |p: [f64; 2], q: [f64; 2]| ((p[0] - q[0]) * pl.spacing[0]).hypot((p[1] - q[1]) * pl.spacing[1]);
    let (lone_a, lone_b) = (lone(0), lone(1));
    let mut pairs: Vec<(f64, usize, usize)> = Vec::with_capacity(lone_a.len() * lone_b.len());
    for &p in &lone_a {
        for &q in &lone_b {
            pairs.push((mm(sides[p][0].centre(), sides[q][1].centre()), p, q));
        }
    }
    pairs.sort_by(|x, y| x.0.total_cmp(&y.0));
    let mut used = vec![false; nodes];
    for (_, p, q) in pairs {
        if !used[p] && !used[q] {
            (used[p], used[q]) = (true, true);
            join(&mut parent, p, q);
        }
    }
    let sides = totals(&mut parent, &stats, na);
    let root: Vec<usize> = (0..nodes).map(|k| find(&mut parent, k)).collect();
    (0..nodes)
        .into_par_iter()
        .filter(|&r| root[r] == r)
        .map(|r| {
            let [ga, gb] = sides[r];
            let shift = if ga.n > 0.0 && gb.n > 0.0 {
                let (p, q) = (ga.centre(), gb.centre());
                [q[0] - p[0], q[1] - p[1]]
            } else {
                [0.0; 2]
            };
            let whole = [shift[0].round() as i64, shift[1].round() as i64];
            let (lo, hi) = window(&ga, &gb, whole);
            let a = (ga.n > 0.0).then(|| Field::new(lo, hi, pl, |n| la[n] != 0 && root[la[n] as usize - 1] == r));
            let (lo, hi) = ([lo[0] + whole[0], lo[1] + whole[1]], [hi[0] + whole[0], hi[1] + whole[1]]);
            let b = (gb.n > 0.0).then(|| Field::new(lo, hi, pl, |n| lb[n] != 0 && root[na + lb[n] as usize - 1] == r));
            Region { a, b, shift, whole }
        })
        .collect()
}

/// The empty voxels of slice `s`, a fraction `t` of the way from the first key slice to the
/// second, that the regions take, with the deepest blended distance on each.
fn blend(regions: &[Region], t: f64, s: usize, labels: &[u8], pl: &Planes) -> Vec<(usize, f32)> {
    let nu = pl.nu;
    let mut best = vec![f64::INFINITY; nu * pl.nv];
    let (mut lo, mut hi) = ([usize::MAX; 2], [0usize; 2]);
    for r in regions {
        // Whole-pixel moves of the two parts onto the interpolated centroid (`whole` apart, so
        // they stay aligned to within half a pixel).
        let oa = [(t * r.shift[0]).round() as i64, (t * r.shift[1]).round() as i64];
        let ob = [oa[0] - r.whole[0], oa[1] - r.whole[1]];
        // Where the region can be: its window, carried along.
        let (f, o) = match (&r.a, &r.b) {
            (Some(a), _) => (a, oa),
            (None, Some(b)) => (b, ob),
            (None, None) => continue,
        };
        let (x0, y0) = ((f.lo[0] + o[0]).max(0), (f.lo[1] + o[1]).max(0));
        let (x1, y1) = ((f.lo[0] + o[0] + f.w as i64 - 1).min(nu as i64 - 1), (f.lo[1] + o[1] + f.h as i64 - 1).min(pl.nv as i64 - 1));
        if x0 > x1 || y0 > y1 {
            continue;
        }
        for y in y0..=y1 {
            for x in x0..=x1 {
                if labels[pl.at(s, x as usize, y as usize)] != 0 {
                    continue;
                }
                let d = match (&r.a, &r.b) {
                    (Some(a), Some(b)) => (1.0 - t) * a.at(x - oa[0], y - oa[1]) + t * b.at(x - ob[0], y - ob[1]),
                    (Some(a), None) => (1.0 - t) * a.at(x, y) + t * a.end,
                    (None, Some(b)) => (1.0 - t) * b.end + t * b.at(x, y),
                    (None, None) => unreachable!(),
                };
                let n = y as usize * nu + x as usize;
                best[n] = best[n].min(d);
            }
        }
        let (x0, x1, y0, y1) = (x0 as usize, x1 as usize, y0 as usize, y1 as usize);
        (lo, hi) = ([lo[0].min(x0), lo[1].min(y0)], [hi[0].max(x1), hi[1].max(y1)]);
    }
    let mut out = Vec::new();
    for y in lo[1]..=hi[1] {
        for x in lo[0]..=hi[0] {
            let d = best[y * nu + x];
            if d <= 0.0 {
                out.push((pl.at(s, x, y), d as f32));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Paint a disc of `label` (radius `r` mm about pixel `c`) on slice `s` of `pl`.
    fn disc(labels: &mut [u8], pl: &Planes, s: usize, c: [f64; 2], r: f64, label: u8) {
        for y in 0..pl.nv {
            for x in 0..pl.nu {
                if ((x as f64 - c[0]) * pl.spacing[0]).hypot((y as f64 - c[1]) * pl.spacing[1]) <= r {
                    labels[pl.at(s, x, y)] = label;
                }
            }
        }
    }

    /// The label map with the added voxels written in (each must have been empty).
    fn apply(labels: &[u8], added: &[(u8, Vec<usize>)]) -> Vec<u8> {
        let mut out = labels.to_vec();
        for (l, v) in added {
            for &n in v {
                assert_eq!(out[n], 0, "voxel {n} was not empty");
                out[n] = *l;
            }
        }
        out
    }

    /// Pixel count and centroid (pixels) of `label` on slice `s`.
    fn area(labels: &[u8], pl: &Planes, s: usize, label: u8) -> (usize, [f64; 2]) {
        let m = pl.mask(labels, s, label);
        let mut st = Stats::NONE;
        for (n, _) in m.iter().enumerate().filter(|(_, &v)| v != 0) {
            st.add(&Stats::pixel(n % pl.nu, n / pl.nu));
        }
        (st.n as usize, if st.n > 0.0 { st.centre() } else { [f64::NAN; 2] })
    }

    /// Discs of radius `r0` mm on slice 0 and `r1` mm on slice 10 across `axis` (48 × 52
    /// pixels in-plane, `pixel` mm, slices 2 mm apart), filled: the areas of slices 0..=10 in
    /// mm², checking the key slices are kept.
    fn discs_across(axis: usize, pixel: f64, r0: f64, r1: f64) -> Vec<f64> {
        let (mut dims, mut spacing) = ([0; 3], [pixel; 3]);
        let (u, v) = [(1, 2), (0, 2), (0, 1)][axis];
        (dims[axis], dims[u], dims[v], spacing[axis]) = (11, 48, 52, 2.0);
        let pl = Planes::new(dims, spacing, axis);
        let mut labels = vec![0u8; dims.iter().product()];
        let c = [24.0, 26.0];
        disc(&mut labels, &pl, 0, c, r0, 3);
        disc(&mut labels, &pl, 10, c, r1, 3);
        let added = fill_between_slices(&labels, dims, spacing, axis, &[3]);
        assert_eq!(added.len(), 1);
        let filled = apply(&labels, &added);
        for s in [0, 10] {
            assert_eq!(pl.mask(&filled, s, 3), pl.mask(&labels, s, 3), "key slice {s} changed");
        }
        (0..11).map(|s| area(&filled, &pl, s, 3).0 as f64 * pixel * pixel).collect()
    }

    #[test]
    fn interpolate_growing_disc() {
        let a = discs_across(2, 1.0, 4.0, 8.0);
        assert!((a[5] - PI * 36.0).abs() < 0.1 * PI * 36.0, "middle {} mm²", a[5]);
        // Growing between the key slices (the first and last steps may round to the keys).
        let grows = |a: &[f64]| a[1..10].windows(2).all(|w| w[1] > w[0]) && a[0] <= a[1] && a[9] <= a[10];
        assert!(grows(&a), "not growing: {a:?}");
        // Pixels of 0.5 mm, slices 2 mm apart: the same radii in mm.
        let b = discs_across(2, 0.5, 4.0, 8.0);
        assert!((b[5] - PI * 36.0).abs() < 0.1 * PI * 36.0, "anisotropic middle {} mm²", b[5]);
        assert!(grows(&b), "not growing: {b:?}");
        // Across the other axes, slice for slice the same.
        for axis in [0, 1] {
            assert_eq!(discs_across(axis, 1.0, 4.0, 8.0), a, "axis {axis}");
            assert_eq!(discs_across(axis, 0.5, 4.0, 8.0), b, "axis {axis}");
        }
    }

    #[test]
    fn interpolate_single_slice_adds_nothing() {
        let dims = [32, 32, 12];
        let pl = Planes::new(dims, [1.0; 3], 2);
        let mut labels = vec![0u8; dims.iter().product()];
        disc(&mut labels, &pl, 3, [16.0, 16.0], 6.0, 1);
        let added = fill_between_slices(&labels, dims, [1.0; 3], 2, &[1, 0, 2, 1]);
        assert_eq!(added, vec![(1, vec![]), (2, vec![])]);
    }

    #[test]
    fn interpolate_moving_regions_keep_their_size() {
        let dims = [64, 44, 9];
        let pl = Planes::new(dims, [1.0; 3], 2);
        // A disc moving sideways by 2.5 diameters: plain blending would lose it in between.
        let mut labels = vec![0u8; dims.iter().product()];
        disc(&mut labels, &pl, 0, [12.0, 20.0], 5.0, 1);
        disc(&mut labels, &pl, 8, [37.0, 20.0], 5.0, 1);
        let full = area(&labels, &pl, 0, 1).0 as f64;
        let filled = apply(&labels, &fill_between_slices(&labels, dims, [1.0; 3], 2, &[1]));
        for s in 1..8 {
            let (n, c) = area(&filled, &pl, s, 1);
            let want = [12.0 + 25.0 * s as f64 / 8.0, 20.0];
            assert!((n as f64 - full).abs() < 0.1 * full, "slice {s}: {n} vs {full}");
            assert!((c[0] - want[0]).abs() <= 0.5 && (c[1] - want[1]).abs() <= 0.5, "slice {s}: {c:?} vs {want:?}");
        }
        // Two discs, each moving by 1.2 radii (overlapping their partners) in opposite
        // directions: no waist, each about halfway.
        let mut labels = vec![0u8; dims.iter().product()];
        disc(&mut labels, &pl, 0, [12.0, 12.0], 5.0, 1);
        disc(&mut labels, &pl, 0, [44.0, 30.0], 5.0, 1);
        disc(&mut labels, &pl, 8, [18.0, 12.0], 5.0, 1);
        disc(&mut labels, &pl, 8, [38.0, 30.0], 5.0, 1);
        let filled = apply(&labels, &fill_between_slices(&labels, dims, [1.0; 3], 2, &[1]));
        let mid = Mask { dims: [64, 44, 1], data: pl.mask(&filled, 4, 1) };
        let (comp, sizes) = components(&mid, Connectivity::TwentySix);
        assert_eq!(sizes.len(), 3, "two regions in the middle");
        for (c, want) in [(1u32, [15.0, 12.0]), (2, [41.0, 30.0])] {
            let mut st = Stats::NONE;
            for n in (0..comp.len()).filter(|&n| comp[n] == c) {
                st.add(&Stats::pixel(n % 64, n / 64));
            }
            let at = st.centre();
            assert!((st.n - full).abs() < 0.1 * full, "region {c}: {} vs {full}", st.n);
            assert!((at[0] - want[0]).abs() < 0.5 && (at[1] - want[1]).abs() < 0.5, "region {c}: {at:?} vs {want:?}");
        }
    }

    #[test]
    fn interpolate_region_without_partner_ends_before_halfway() {
        let dims = [56, 44, 9];
        let pl = Planes::new(dims, [1.0; 3], 2);
        let mut labels = vec![0u8; dims.iter().product()];
        disc(&mut labels, &pl, 0, [12.0, 12.0], 5.0, 1);
        disc(&mut labels, &pl, 0, [40.0, 30.0], 5.0, 1);
        disc(&mut labels, &pl, 8, [12.0, 12.0], 5.0, 1);
        let filled = apply(&labels, &fill_between_slices(&labels, dims, [1.0; 3], 2, &[1]));
        let near = |s: usize| (30..=50).flat_map(|x| (20..=40).map(move |y| (x, y))).filter(|&(x, y)| filled[pl.at(s, x, y)] == 1).count();
        let at = |s: usize| (0..=24).flat_map(|x| (0..=24).map(move |y| (x, y))).filter(|&(x, y)| filled[pl.at(s, x, y)] == 1).count();
        assert!(near(1) > near(2) && near(2) > near(3) && near(3) > 0, "{} {} {}", near(1), near(2), near(3));
        assert!((4..8).all(|s| near(s) == 0));
        // The disc on both slices is carried unchanged.
        assert!((1..8).all(|s| at(s) == at(0)));
    }

    #[test]
    fn interpolate_segments_share_by_depth() {
        let dims = [48, 32, 10];
        let pl = Planes::new(dims, [1.0; 3], 2);
        let mut labels = vec![0u8; dims.iter().product()];
        for s in [0, 8] {
            disc(&mut labels, &pl, s, [16.0, 16.0], 8.0, 1);
        }
        for s in [1, 9] {
            disc(&mut labels, &pl, s, [30.0, 16.0], 8.0, 2);
        }
        let added = fill_between_slices(&labels, dims, [1.0; 3], 2, &[1, 2]);
        assert_eq!(added.iter().map(|a| a.0).collect::<Vec<_>>(), vec![1, 2]);
        assert!(added.iter().all(|(_, v)| v.windows(2).all(|w| w[0] < w[1])));
        // No voxel of either segment is overwritten (apply checks), each fills its own gap.
        let filled = apply(&labels, &added);
        assert_eq!(filled[pl.at(1, 16, 16)], 1);
        assert_eq!(filled[pl.at(1, 30, 16)], 2);
        assert_eq!(filled[pl.at(8, 16, 16)], 1);
        assert_eq!(filled[pl.at(8, 30, 16)], 2);
        assert_eq!(filled[pl.at(9, 16, 16)], 0);
        // Where both reach, the nearer centre (deeper inside) wins.
        for s in 2..8 {
            for x in 16..=22 {
                assert_eq!(filled[pl.at(s, x, 16)], 1, "slice {s}, x {x}");
            }
            for x in 24..=30 {
                assert_eq!(filled[pl.at(s, x, 16)], 2, "slice {s}, x {x}");
            }
        }
    }

    #[test]
    fn interpolate_large_volume_quickly() {
        let dims = [256, 256, 100];
        let pl = Planes::new(dims, [0.8, 0.8, 1.5], 2);
        let mut labels = vec![0u8; dims.iter().product()];
        for s in (0..100).filter(|s| s % 10 == 0 || s % 10 == 5) {
            let f = s as f64;
            disc(&mut labels, &pl, s, [128.0 + 40.0 * (f / 15.0).cos(), 128.0 + 30.0 * (f / 20.0).sin()], 30.0 + 15.0 * (f / 10.0).sin(), 1);
            disc(&mut labels, &pl, s, [40.0 + f, 200.0], 8.0, 2);
        }
        let t = std::time::Instant::now();
        let added = fill_between_slices(&labels, dims, [0.8, 0.8, 1.5], 2, &[1, 2]);
        let secs = t.elapsed().as_secs_f64();
        let n: usize = added.iter().map(|a| a.1.len()).sum();
        eprintln!("fill between slices, 256 × 256 × 100: {n} voxels added in {:.0} ms", secs * 1e3);
        assert!(n > 0);
        assert!(secs < if cfg!(debug_assertions) { 60.0 } else { 3.0 }, "{secs} s");
    }
}
