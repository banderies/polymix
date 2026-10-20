//! Seeded segmentation, as 3D Slicer's Grow from seeds, Watershed and Fast marching effects.
//! All three spread labelled seeds through a priority queue of voxels (a binary heap, first
//! in first out among equal priorities so that even plateaus are shared out by distance):
//!
//! - **Grow from seeds** ([`grow_from_seeds`]) is GrowCut (Vezhnevets & Konouchine 2005) in
//!   the shortest-path form of Fast GrowCut (Zhu, Kolesov, Gao, Kikinis & Tannenbaum 2014):
//!   rather than iterating the cellular automaton until no cell changes, every voxel takes
//!   the label of the seed joined to it by the cheapest path, found for all labels at once
//!   by one multi-source Dijkstra search. A step from voxel p to its neighbour q costs
//!   `|I(p) − I(q)| / (Imax − Imin)` (the volume's intensity range) plus `locality · |pq|`
//!   (step length in mm), so a label crosses even regions almost for free and stalls at an
//!   edge, where another label's front arrives more cheaply.
//! - **Watershed** ([`watershed`]) is marker-controlled flooding (Beucher & Meyer 1993) by
//!   Meyer's priority-queue algorithm (1991). The image is smoothed with a Gaussian whose
//!   standard deviation is the object scale (mm), its gradient magnitude taken by central
//!   differences, and that relief flooded from the markers in increasing order of gradient:
//!   each voxel takes the label of the neighbour that reached it first, so the labels meet on
//!   the gradient's crests, the objects' edges. No watershed lines are left between them.
//! - **Fast marching** ([`fast_march`]) solves the eikonal equation `|∇T| F = 1` outward
//!   from a segment with Sethian's fast marching method (1996; first-order upwind scheme on
//!   the 6-neighbour stencil, not a Dijkstra approximation, so fronts stay round), with a
//!   speed `F = exp(−((I − μ) / σ)² / 2)` from the mean μ and standard deviation σ of the
//!   segment's intensities, and keeps the voxels reached first until the segment has the
//!   target size.
//!
//! Grow from seeds and Watershed work within a region of interest (a mask, or the bounding
//! box of the seeds grown by a margin) and leave the rest of the label map as it was. All
//! three use 6-connectivity.
//!
//! References:
//! - V. Vezhnevets, V. Konouchine, "GrowCut: interactive multi-label N-D image segmentation
//!   by cellular automata", Proc. Graphicon 2005, pp. 150–156.
//! - L. Zhu, I. Kolesov, Y. Gao, R. Kikinis, A. Tannenbaum, "An effective interactive
//!   medical image segmentation method using Fast GrowCut", MICCAI 2014 Workshop on
//!   Interactive Medical Image Computing.
//! - S. Beucher, F. Meyer, "The morphological approach to segmentation: the watershed
//!   transformation", in E. R. Dougherty (ed.), Mathematical Morphology in Image Processing,
//!   Marcel Dekker, 1993, pp. 433–481.
//! - F. Meyer, "Un algorithme optimal de ligne de partage des eaux", 8e congrès AFCET de
//!   reconnaissance des formes et intelligence artificielle, Lyon, 1991.
//! - J. A. Sethian, "A fast marching level set method for monotonically advancing fronts",
//!   Proc. Natl. Acad. Sci. USA 93 (4), 1996, pp. 1591–1595.

use crate::mask::{Connectivity, Mask};
use mm_volume::load::LoadedVolume;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Smallest standard deviation of the fast-marching speed, as a fraction of the volume's
/// intensity range (a seed of equal voxels would otherwise admit nothing else).
const SD_FLOOR: f64 = 0.02;
/// Smallest fast-marching speed (reached beyond about 11.7 σ from the mean): voxels unlike
/// the seed are still reached, nearest first, and arrival times stay finite in `f32`.
const MIN_SPEED: f64 = 1e-30;

/// Each voxel of the region of interest takes the label of the seed it is reached from most
/// cheaply (multi-label Dijkstra); seeds are the non-zero voxels of `seeds`.
///
/// A step between neighbours costs their intensity difference over the volume's intensity
/// range, plus `locality` (≥ 0) per millimetre, which favours nearer seeds (0 = intensity
/// only). The region of interest is `roi` or, without one, the bounding box of the seeds
/// grown by `margin_mm`; voxels outside it keep their value in `seeds`, and voxels in it that
/// no seed reaches stay 0. Seeds keep their labels (seeds outside a `roi` mask still grow into
/// it). Give at least two labels, the object and what surrounds it, as Slicer asks: with one,
/// every voxel of the region connected to the seeds takes it.
pub fn grow_from_seeds(vol: &LoadedVolume, seeds: &[u8], roi: Option<&Mask>, margin_mm: f64, locality: f64) -> Vec<u8> {
    assert_eq!(seeds.len(), vol.values.len(), "label map and volume of different sizes");
    let mut out = seeds.to_vec();
    let Some(inner) = roi_box(vol, seeds, roi, margin_mm) else { return out };
    let Work { b, values, mut labels, allowed } = Work::new(vol, seeds, inner, [0; 3], roi);
    let scale = intensity_scale(vol);
    let locality = locality.max(0.0) as f32;
    let stencil = Stencil::new(b.dims, Connectivity::Six, spacing(vol));
    let mut dist = vec![f32::INFINITY; b.len()];
    let mut queue = Queue::default();
    for (n, &l) in labels.iter().enumerate() {
        if l != 0 {
            dist[n] = 0.0;
            queue.push(0.0, n);
        }
    }
    while let Some((d, p)) = queue.pop() {
        if d > dist[p] {
            continue; // reached more cheaply since
        }
        let (v, l) = (values[p], labels[p]);
        stencil.each(p, |q, len| {
            if allowed[q] == 0 {
                return;
            }
            let c = (values[q] - v).abs() * scale;
            let dq = d + if c.is_finite() { c } else { 1.0 } + locality * len;
            if dq < dist[q] {
                dist[q] = dq;
                labels[q] = l;
                queue.push(dq, q);
            }
        });
    }
    b.write(&mut out, vol.dims, |n| (allowed[n] != 0 && labels[n] != 0).then_some(labels[n]));
    out
}

/// Marker-based watershed: the gradient magnitude of the image smoothed with a Gaussian of
/// standard deviation `object_scale_mm` (0 = none) is flooded from the markers (the non-zero
/// voxels of `markers`) in increasing order, each voxel taking the label of the neighbour that
/// reached it first. The region of interest is `roi` or the bounding box of the markers grown
/// by `margin_mm`, as for [`grow_from_seeds`]; voxels outside it keep their value in
/// `markers`. A larger object scale ignores more noise and fine detail, but rounds corners
/// and moves the edges of small convex objects inwards a little (about scale² / radius).
pub fn watershed(vol: &LoadedVolume, markers: &[u8], object_scale_mm: f64, roi: Option<&Mask>, margin_mm: f64) -> Vec<u8> {
    assert_eq!(markers.len(), vol.values.len(), "label map and volume of different sizes");
    let mut out = markers.to_vec();
    let Some(inner) = roi_box(vol, markers, roi, margin_mm) else { return out };
    let sp = spacing(vol);
    let sigma = sp.map(|s| object_scale_mm.max(0.0) / s);
    // Room around the region for the Gaussian and the central differences.
    let pad = sigma.map(|s| ((3.0 * s).ceil() as usize).saturating_add(1));
    let Work { b, mut values, mut labels, allowed } = Work::new(vol, markers, inner, pad, roi);
    blur(&mut values, b.dims, sigma);
    let grad = gradient_magnitude(&values, b.dims, sp);
    drop(values);
    let stencil = Stencil::new(b.dims, Connectivity::Six, sp);
    let mut queue = Queue::default();
    // The markers come out first, and put their neighbours in at their gradient.
    for (n, &l) in labels.iter().enumerate() {
        if l != 0 {
            queue.push(f32::NEG_INFINITY, n);
        }
    }
    while let Some((_, p)) = queue.pop() {
        let l = labels[p];
        stencil.each(p, |q, _| {
            if allowed[q] != 0 && labels[q] == 0 {
                labels[q] = l;
                queue.push(grad[q], q);
            }
        });
    }
    b.write(&mut out, vol.dims, |n| (allowed[n] != 0 && labels[n] != 0).then_some(labels[n]));
    out
}

/// Grow `seed` by fast marching until it has `target_voxels` voxels (counting its own), or
/// until nothing connected to it is left. The front moves at speed
/// `exp(−((I − μ) / σ)² / 2)` (at least 1e-30), μ and σ being the mean and standard deviation
/// of the intensities in `seed` (σ at least 2 % of the volume's range), so voxels like the
/// seed's are taken first, nearest first. An empty seed, or one already at the target, comes
/// back unchanged.
pub fn fast_march(vol: &LoadedVolume, seed: &Mask, target_voxels: usize) -> Mask {
    assert_eq!(seed.dims, vol.dims, "mask and volume of different sizes");
    let mut out = seed.clone();
    let start: Vec<usize> = (0..seed.len()).into_par_iter().filter(|&n| seed.data[n] != 0).collect();
    let mut count = start.len();
    if count == 0 || count >= target_voxels {
        return out;
    }
    // Intensity statistics of the seed.
    let inside: Vec<f64> = start.iter().map(|&n| vol.values[n] as f64).filter(|v| v.is_finite()).collect();
    let k = inside.len().max(1) as f64;
    let mean = inside.iter().sum::<f64>() / k;
    let var = inside.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / k;
    let sd = var.sqrt().max(SD_FLOOR * (vol.range.1 - vol.range.0) as f64).max(1e-12);
    let speed = |n: usize| {
        let z = (vol.values[n] as f64 - mean) / sd;
        (-0.5 * z * z).exp().max(MIN_SPEED) // NaN → MIN_SPEED
    };
    let dims = vol.dims;
    let sp = spacing(vol);
    let strides = [1, dims[0], dims[0] * dims[1]];
    // Arrival time at `n` from its known neighbours: the first-order upwind solution of
    // Σ ((T − a_i) / h_i)² = 1 / F², taking the axes in increasing order of their earliest
    // known neighbour a_i while the solution stays above the next one.
    let arrival = |n: usize, t: &[f32], known: &[u8]| -> f32 {
        let mut near = [(0.0f64, 0.0f64); 3];
        let mut c = 0;
        for ((&len, &s), &h) in dims.iter().zip(&strides).zip(&sp) {
            let x = (n / s) % len;
            let mut a = f32::INFINITY;
            if x > 0 && known[n - s] != 0 {
                a = t[n - s];
            }
            if x + 1 < len && known[n + s] != 0 {
                a = a.min(t[n + s]);
            }
            if a.is_finite() {
                near[c] = (a as f64, h);
                c += 1;
            }
        }
        near[..c].sort_by(|x, y| x.0.total_cmp(&y.0));
        let f = speed(n);
        let rhs = 1.0 / (f * f);
        let (mut sa, mut sb, mut sc, mut time) = (0.0, 0.0, 0.0, f64::INFINITY);
        for &(a, h) in &near[..c] {
            if time <= a {
                break;
            }
            let w = 1.0 / (h * h);
            (sa, sb, sc) = (sa + w, sb + w * a, sc + w * a * a);
            let disc = sb * sb - sa * (sc - rhs);
            if disc < 0.0 {
                break;
            }
            time = (sb + disc.sqrt()) / sa;
        }
        time as f32
    };
    let stencil = Stencil::new(dims, Connectivity::Six, sp);
    // Update the arrival times of the unknown neighbours of `p`.
    let relax = |p: usize, t: &mut [f32], known: &[u8], queue: &mut Queue| {
        stencil.each(p, |q, _| {
            if known[q] == 0 {
                let tq = arrival(q, t, known);
                if tq < t[q] {
                    t[q] = tq;
                    queue.push(tq, q);
                }
            }
        });
    };
    let mut t = vec![f32::INFINITY; seed.len()];
    for &n in &start {
        t[n] = 0.0;
    }
    let mut queue = Queue::default();
    for &n in &start {
        relax(n, &mut t, &out.data, &mut queue);
    }
    while let Some((time, p)) = queue.pop() {
        if out.data[p] != 0 || time > t[p] {
            continue;
        }
        out.data[p] = 1;
        count += 1;
        if count >= target_voxels {
            break;
        }
        relax(p, &mut t, &out.data, &mut queue);
    }
    out
}

/// A voxel waiting in a [`Queue`].
#[derive(Clone, Copy)]
struct Item {
    key: f32,
    seq: u32,
    at: u32,
}

impl Ord for Item {
    /// Reversed (`BinaryHeap` pops the greatest): smallest key, then earliest in.
    fn cmp(&self, o: &Self) -> Ordering {
        o.key.total_cmp(&self.key).then(o.seq.cmp(&self.seq))
    }
}

impl PartialOrd for Item {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

impl PartialEq for Item {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o) == Ordering::Equal
    }
}

impl Eq for Item {}

/// Voxels by smallest key, first in first out among equal keys (the order number wraps
/// after 4 billion pushes, which only affects ties).
#[derive(Default)]
struct Queue {
    heap: BinaryHeap<Item>,
    seq: u32,
}

impl Queue {
    fn push(&mut self, key: f32, at: usize) {
        let at = u32::try_from(at).expect("more than 4 billion voxels");
        self.heap.push(Item { key, seq: self.seq, at });
        self.seq = self.seq.wrapping_add(1);
    }

    fn pop(&mut self) -> Option<(f32, usize)> {
        self.heap.pop().map(|i| (i.key, i.at as usize))
    }
}

/// The neighbour steps within a box of `dims` voxels: offset, index step and length (mm).
struct Stencil {
    dims: [usize; 3],
    steps: Vec<([i64; 3], isize, f32)>,
}

impl Stencil {
    fn new(dims: [usize; 3], conn: Connectivity, spacing: [f64; 3]) -> Self {
        let steps = conn
            .offsets()
            .into_iter()
            .map(|o| {
                let d = o[0] + o[1] * dims[0] as i64 + o[2] * (dims[0] * dims[1]) as i64;
                let len = (0..3).map(|a| (o[a] as f64 * spacing[a]).powi(2)).sum::<f64>().sqrt();
                (o, d as isize, len as f32)
            })
            .collect();
        Self { dims, steps }
    }

    /// `f(neighbour, step length)` for the neighbours of `n` inside the box.
    #[inline]
    fn each(&self, n: usize, mut f: impl FnMut(usize, f32)) {
        let [nx, ny, nz] = self.dims.map(|d| d as i64);
        let (i, j, k) = (n as i64 % nx, (n as i64 / nx) % ny, n as i64 / (nx * ny));
        for &(o, d, len) in &self.steps {
            let (a, b, c) = (i + o[0], j + o[1], k + o[2]);
            if a >= 0 && b >= 0 && c >= 0 && a < nx && b < ny && c < nz {
                f(n.wrapping_add_signed(d), len);
            }
        }
    }
}

/// A box of the voxel grid: its first voxel and its size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Region {
    lo: [usize; 3],
    dims: [usize; 3],
}

impl Region {
    fn len(&self) -> usize {
        self.dims.iter().product()
    }

    /// Grid voxel of the box's voxel `n`.
    fn grid(&self, n: usize) -> [usize; 3] {
        let [nx, ny, _] = self.dims;
        [n % nx + self.lo[0], (n / nx) % ny + self.lo[1], n / (nx * ny) + self.lo[2]]
    }

    fn contains(&self, g: [usize; 3]) -> bool {
        (0..3).all(|a| g[a] >= self.lo[a] && g[a] - self.lo[a] < self.dims[a])
    }

    /// Grown by `pad` voxels along each axis, within a grid of `grid` voxels.
    fn grown(&self, pad: [usize; 3], grid: [usize; 3]) -> Region {
        let lo = [0, 1, 2].map(|a| self.lo[a].saturating_sub(pad[a]));
        let hi = [0, 1, 2].map(|a| (self.lo[a] + self.dims[a]).saturating_add(pad[a]).min(grid[a]));
        Region { lo, dims: [0, 1, 2].map(|a| hi[a] - lo[a]) }
    }

    fn union(&self, o: &Region) -> Region {
        let lo = [0, 1, 2].map(|a| self.lo[a].min(o.lo[a]));
        let hi = [0, 1, 2].map(|a| (self.lo[a] + self.dims[a]).max(o.lo[a] + o.dims[a]));
        Region { lo, dims: [0, 1, 2].map(|a| hi[a] - lo[a]) }
    }

    /// Write `f(n)`, where it is some, for the box's voxels `n` into a grid label map.
    fn write(&self, out: &mut [u8], grid: [usize; 3], f: impl Fn(usize) -> Option<u8> + Sync) {
        let [bx, by, bz] = self.dims;
        out.par_chunks_mut((grid[0] * grid[1]).max(1)).enumerate().skip(self.lo[2]).take(bz).for_each(|(k, plane)| {
            for j in 0..by {
                for i in 0..bx {
                    if let Some(l) = f(((k - self.lo[2]) * by + j) * bx + i) {
                        plane[(j + self.lo[1]) * grid[0] + i + self.lo[0]] = l;
                    }
                }
            }
        });
    }
}

/// The bounding box of the non-zero voxels of `data`, if any.
fn bounding_box(dims: [usize; 3], data: &[u8]) -> Option<Region> {
    data.par_chunks((dims[0] * dims[1]).max(1))
        .enumerate()
        .filter_map(|(k, plane)| {
            let mut b: Option<Region> = None;
            for (n, _) in plane.iter().enumerate().filter(|(_, &v)| v != 0) {
                let r = Region { lo: [n % dims[0], n / dims[0], k], dims: [1; 3] };
                b = Some(b.map_or(r, |b| b.union(&r)));
            }
            b
        })
        .reduce_with(|a, b| a.union(&b))
}

/// Where seeded growth works: the bounding box of the seeds and `roi`, or, without `roi`, of
/// the seeds grown by `margin_mm` (within the grid). `None` without seeds.
fn roi_box(vol: &LoadedVolume, seeds: &[u8], roi: Option<&Mask>, margin_mm: f64) -> Option<Region> {
    let b = bounding_box(vol.dims, seeds)?;
    Some(match roi {
        Some(m) => {
            assert_eq!(m.dims, vol.dims, "mask and volume of different sizes");
            bounding_box(m.dims, &m.data).map_or(b, |r| b.union(&r))
        }
        None => b.grown(spacing(vol).map(|s| (margin_mm.max(0.0) / s).ceil() as usize), vol.dims),
    })
}

/// The voxels of a box: intensities, labels, and whether each is in the region of interest.
struct Work {
    b: Region,
    values: Vec<f32>,
    labels: Vec<u8>,
    allowed: Vec<u8>,
}

impl Work {
    /// The box `inner` (the region of interest's, all of it allowed unless `roi` is given)
    /// grown by `pad` voxels, which are not.
    fn new(vol: &LoadedVolume, seeds: &[u8], inner: Region, pad: [usize; 3], roi: Option<&Mask>) -> Self {
        let dims = vol.dims;
        let b = inner.grown(pad, dims);
        let at = |n: usize| {
            let g = b.grid(n);
            (g, (g[2] * dims[1] + g[1]) * dims[0] + g[0])
        };
        let values = (0..b.len()).into_par_iter().map(|n| vol.values[at(n).1]).collect();
        let labels = (0..b.len()).into_par_iter().map(|n| seeds[at(n).1]).collect();
        let allowed = (0..b.len())
            .into_par_iter()
            .map(|n| {
                let (g, i) = at(n);
                (inner.contains(g) && roi.is_none_or(|m| m.data[i] != 0)) as u8
            })
            .collect();
        Self { b, values, labels, allowed }
    }
}

/// Voxel spacing in mm (1 where the file's is unusable).
fn spacing(vol: &LoadedVolume) -> [f64; 3] {
    vol.geometry.spacing.map(|s| if s > 0.0 && s.is_finite() { s } else { 1.0 })
}

/// One over the volume's intensity range (1 when it is empty).
fn intensity_scale(vol: &LoadedVolume) -> f32 {
    let w = vol.range.1 - vol.range.0;
    if w > 0.0 && w.is_finite() {
        1.0 / w
    } else {
        1.0
    }
}

/// Separable Gaussian blur of a box of `dims` voxels, with standard deviation `sigma` voxels
/// along each axis (0 = none); the ends of each line are repeated.
fn blur(values: &mut [f32], dims: [usize; 3], sigma: [f64; 3]) {
    let strides = [1, dims[0], dims[0] * dims[1]];
    for ((&s, &len), &stride) in sigma.iter().zip(&dims).zip(&strides) {
        // Wider than the line itself changes nothing more.
        let r = ((3.0 * s).ceil() as usize).min(2 * len);
        if s <= 0.0 || r == 0 || len < 2 {
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

/// Gradient magnitude (per mm) of a box of `dims` voxels, by central differences (one-sided
/// at the ends of lines).
fn gradient_magnitude(values: &[f32], dims: [usize; 3], spacing: [f64; 3]) -> Vec<f32> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec3;
    use mm_volume::geometry::Geometry;
    use std::time::Instant;

    /// Deterministic noise in [-1, 1) from a voxel index (the SplitMix64 finaliser).
    fn noise(n: usize) -> f32 {
        let mut x = (n as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^= x >> 31;
        (x >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }

    fn volume(dims: [usize; 3], values: Vec<f32>) -> LoadedVolume {
        LoadedVolume::from_values("test", dims, values, Geometry::from_axes([dims[0], dims[1], dims[2], 1], [1.0; 3], [DVec3::X, DVec3::Y, DVec3::Z], DVec3::ZERO))
    }

    /// A bright ball (100) of radius `r` voxels around voxel (n/2, n/2, n/2) of an n³ grid, on
    /// a dark background (0), with ±4 noise; and the ball's voxels.
    fn ball(n: usize, r: f64) -> (LoadedVolume, Mask) {
        let c = (n / 2) as f64;
        let mut truth = Mask::new([n; 3]);
        for idx in 0..truth.len() {
            let [i, j, k] = truth.ijk(idx);
            truth.data[idx] = (DVec3::new(i as f64 - c, j as f64 - c, k as f64 - c).length() <= r) as u8;
        }
        let values = truth.data.iter().enumerate().map(|(idx, &t)| 100.0 * t as f32 + 4.0 * noise(idx)).collect();
        (volume([n; 3], values), truth)
    }

    /// Label 1 in a 3³ cube at the centre, label 2 on the faces of the box one voxel in from
    /// the grid's (as a stroke around the object).
    fn seeds(n: usize) -> Vec<u8> {
        let c = n / 2;
        (0..n * n * n)
            .map(|idx| {
                let p = [idx % n, (idx / n) % n, idx / (n * n)];
                if p.iter().all(|&x| x.abs_diff(c) <= 1) {
                    1
                } else if p.iter().all(|&x| (1..=n - 2).contains(&x)) && p.iter().any(|&x| x == 1 || x == n - 2) {
                    2
                } else {
                    0
                }
            })
            .collect()
    }

    fn dice(a: &Mask, b: &Mask) -> f64 {
        let both = a.data.iter().zip(&b.data).filter(|(&x, &y)| x != 0 && y != 0).count();
        2.0 * both as f64 / (a.count() + b.count()) as f64
    }

    #[test]
    fn grow_from_seeds_finds_the_ball() {
        let (vol, truth) = ball(40, 11.0);
        let s = seeds(40);
        for locality in [0.0, 0.01, 0.05] {
            let out = grow_from_seeds(&vol, &s, None, 5.0, locality);
            let d = dice(&Mask::from_labels(vol.dims, &out, 1), &truth);
            assert!(d > 0.95, "locality {locality}: Dice {d}");
            // The box of the seeds grown by 5 mm is the whole grid.
            assert!(out.iter().all(|&l| l == 1 || l == 2));
            assert!(s.iter().zip(&out).all(|(&a, &b)| a == 0 || a == b), "seeds keep their labels");
        }
    }

    #[test]
    fn grow_from_seeds_keeps_to_the_region() {
        let (vol, truth) = ball(40, 11.0);
        let s = seeds(40);
        // A mask: the lower half.
        let mut half = Mask::new(vol.dims);
        half.data.iter_mut().enumerate().for_each(|(n, m)| *m = (n / 1600 < 20) as u8);
        let out = grow_from_seeds(&vol, &s, Some(&half), 0.0, 0.0);
        for n in 0..out.len() {
            if half.data[n] == 0 {
                assert_eq!(out[n], s[n]);
            }
        }
        let mut got = Mask::from_labels(vol.dims, &out, 1);
        got.intersect(&half);
        let mut want = truth.clone();
        want.intersect(&half);
        assert!(dice(&got, &want) > 0.95);
        // A margin box: a seed in the ball, one just outside it, 2 mm around them.
        let mut s = vec![0u8; out.len()];
        s[vol.index(20, 20, 20)] = 1;
        s[vol.index(20, 20, 34)] = 2;
        let out = grow_from_seeds(&vol, &s, None, 2.0, 0.0);
        let inner = Region { lo: [18, 18, 18], dims: [5, 5, 19] };
        for (n, &l) in out.iter().enumerate() {
            let g = [n % 40, (n / 40) % 40, n / 1600];
            assert_eq!(l != 0, inner.contains(g), "{g:?}");
            if inner.contains(g) {
                assert_eq!(l == 1, truth.data[n] != 0, "{g:?}");
            }
        }
        // One label takes the whole region.
        s[vol.index(20, 20, 34)] = 0;
        let out = grow_from_seeds(&vol, &s, None, 3.0, 0.0);
        assert_eq!(out.iter().filter(|&&l| l == 1).count(), 7 * 7 * 7);
    }

    #[test]
    fn watershed_finds_the_ball() {
        let (vol, truth) = ball(40, 11.0);
        let s = seeds(40);
        for scale in [0.0, 1.0, 2.0] {
            let out = watershed(&vol, &s, scale, None, 5.0);
            let got = Mask::from_labels(vol.dims, &out, 1);
            let d = dice(&got, &truth);
            if scale <= 1.0 {
                assert!(d > 0.95, "object scale {scale}: Dice {d}");
            } else {
                // Smoothing moves the crest of a convex edge inwards (σ² / r ≈ 0.4 voxel here).
                assert!(d > 0.9, "object scale {scale}: Dice {d}");
                assert!(got.data.iter().zip(&truth.data).all(|(&g, &t)| g == 0 || t != 0));
            }
            assert!(out.iter().all(|&l| l == 1 || l == 2));
        }
        assert_eq!(watershed(&vol, &s, 0.0, None, 5.0), {
            let mut exact = vec![2u8; s.len()];
            truth.write_label(&mut exact, 1, true);
            exact
        });
        // Within a mask.
        let mut half = Mask::new(vol.dims);
        half.data.iter_mut().enumerate().for_each(|(n, m)| *m = (n % 40 < 20) as u8);
        let out = watershed(&vol, &s, 1.0, Some(&half), 0.0);
        assert!((0..out.len()).all(|n| half.data[n] != 0 || out[n] == s[n]));
        let mut got = Mask::from_labels(vol.dims, &out, 1);
        got.intersect(&half);
        let mut want = truth.clone();
        want.intersect(&half);
        assert!(dice(&got, &want) > 0.95);
    }

    #[test]
    fn fast_march_fills_the_ball_first() {
        let (vol, truth) = ball(40, 10.0);
        let mut seed = Mask::new(vol.dims);
        for k in 19..=21 {
            for j in 19..=21 {
                for i in 19..=21 {
                    seed.set(i, j, k, true);
                }
            }
        }
        let whole = truth.count();
        let half = fast_march(&vol, &seed, whole / 2);
        assert_eq!(half.count(), whole / 2);
        assert!(half.data.iter().zip(&truth.data).all(|(&h, &t)| h == 0 || t != 0), "stays in the ball");
        assert!(seed.data.iter().zip(&half.data).all(|(&s, &h)| s == 0 || h != 0), "keeps the seed");
        let more = fast_march(&vol, &seed, whole + 300);
        assert_eq!(more.count(), whole + 300);
        assert!(truth.data.iter().zip(&more.data).all(|(&t, &m)| t == 0 || m != 0), "fills the ball before leaking");
        // Already big enough.
        assert_eq!(fast_march(&vol, &seed, 5), seed);
    }

    #[test]
    fn fast_march_fronts_are_round() {
        // On an even image the front is a sphere (Dijkstra on 6 neighbours would give an
        // octahedron, Dice ≈ 0.8 with the ball).
        let n = 41;
        let vol = volume([n; 3], vec![7.0; n * n * n]);
        let mut seed = Mask::new(vol.dims);
        seed.set(20, 20, 20, true);
        let mut sphere = Mask::new(vol.dims);
        for idx in 0..sphere.len() {
            let [i, j, k] = sphere.ijk(idx);
            sphere.data[idx] = (DVec3::new(i as f64 - 20.0, j as f64 - 20.0, k as f64 - 20.0).length() <= 12.0) as u8;
        }
        let grown = fast_march(&vol, &seed, sphere.count());
        let d = dice(&grown, &sphere);
        assert!(d > 0.95, "Dice {d}");
    }

    #[test]
    fn empty_and_tiny_inputs() {
        let (vol, _) = ball(12, 3.0);
        let none = vec![0u8; vol.values.len()];
        assert_eq!(grow_from_seeds(&vol, &none, None, 5.0, 0.0), none);
        assert_eq!(watershed(&vol, &none, 1.0, None, 5.0), none);
        assert_eq!(fast_march(&vol, &Mask::new(vol.dims), 100).count(), 0);
        let empty = Mask::new(vol.dims);
        assert_eq!(grow_from_seeds(&vol, &none, Some(&empty), 0.0, 0.0), none);
        let one = volume([1; 3], vec![5.0]);
        assert_eq!(grow_from_seeds(&one, &[0], None, 5.0, 1.0), vec![0]);
        assert_eq!(grow_from_seeds(&one, &[3], None, 5.0, 1.0), vec![3]);
        assert_eq!(watershed(&one, &[0], 2.0, None, 5.0), vec![0]);
        assert_eq!(watershed(&one, &[3], 2.0, None, 5.0), vec![3]);
        let seed = Mask { dims: [1; 3], data: vec![1] };
        assert_eq!(fast_march(&one, &seed, 10), seed);
        assert_eq!(fast_march(&one, &Mask::new([1; 3]), 10).count(), 0);
    }

    #[test]
    #[cfg_attr(debug_assertions, ignore = "timing: run with --release")]
    fn grow_from_seeds_is_fast() {
        let (vol, truth) = ball(128, 40.0);
        let s = seeds(128);
        let t0 = Instant::now();
        let out = grow_from_seeds(&vol, &s, None, 5.0, 0.0);
        let grow = t0.elapsed().as_secs_f64();
        let t0 = Instant::now();
        let ws = watershed(&vol, &s, 1.0, None, 5.0);
        let flood = t0.elapsed().as_secs_f64();
        let mut seed = Mask::new(vol.dims);
        seed.set(64, 64, 64, true);
        let t0 = Instant::now();
        let fm = fast_march(&vol, &seed, truth.count());
        let march = t0.elapsed().as_secs_f64();
        eprintln!("128³: grow from seeds {grow:.3} s, watershed {flood:.3} s, fast marching {march:.3} s");
        assert!(dice(&Mask::from_labels(vol.dims, &out, 1), &truth) > 0.95);
        assert!(dice(&Mask::from_labels(vol.dims, &ws, 1), &truth) > 0.95);
        assert_eq!(fm, truth);
        assert!(grow < 5.0, "{grow} s");
    }
}
