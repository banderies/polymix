//! Vessel segmentation with level sets: a segment grown (or shrunk) to the walls of a vessel by
//! a geodesic active contour, and a vessel segmented from points clicked along it.
//!
//! - **Geodesic active contour** ([`level_set`]; Caselles, Kimmel & Sapiro 1997). The surface
//!   of the segment is the zero level of φ, a signed distance in mm (negative inside), and moves
//!   along its outward normal n at the speed `α·P − g·β·κ − γ·∇g·n`, that is
//!   `∂φ/∂t = −α·P·|∇φ| + g·β·κ·|∇φ| + γ·∇g·∇φ`:
//!   - P, the region speed, is 1 inside the structure's intensity range and falls smoothly
//!     (a cubic step) to −1 a little outside it, so the front runs through the vessel and backs
//!     off from everything else; a feature image s in [0, 1] (a vesselness, say) can stand in
//!     for it as `P = 2·s − 1`. It takes the place of the constant balloon of the paper, and
//!     stops the front by itself: the balloon is not slowed by g, which in a vessel a few voxels
//!     across is small everywhere and would stall it;
//!   - g = 1 / (1 + |∇(G_σ * I)|² / k²), the edge-stopping function, is small on the edges of
//!     the image smoothed by a Gaussian of σ. The last two terms are the geodesic ones, the
//!     descent of the surface's area weighted by g: smoothing by the mean curvature κ (the sum
//!     of the principal curvatures) where there is no edge, and advection into the valley of g
//!     (the edge) from either side. k is taken from the image: 2.5 times the median gradient
//!     magnitude (noise), or a tenth of the 99th percentile (an edge) in an image without noise.
//!     κ is limited to that of a sphere one voxel in radius, so that a seed one voxel thick, where
//!     φ has a minimum and κ no meaning, is not smoothed away (at the default weights).
//!
//!   Derivatives are taken in millimetres, so anisotropic voxels need no resampling; the
//!   curvature and advection weights are per voxel of the finest spacing, so that they mean the
//!   same at any resolution. The scheme is explicit: upwind (Godunov) differences for the
//!   balloon and advection terms, central differences for the curvature, and a time step from
//!   the stability limits of both, voxel by voxel. Only a narrow band around the front is
//!   updated (Adalsteinsson & Sethian 1995; Whitaker 1998), within the bounding box of the
//!   starting segment grown by the distance limit. φ is reinitialised to a signed distance
//!   whenever the zero crossings may have moved two voxels, and at least every 20 iterations:
//!   the voxels next to the front take their distance to the nearest crossing along the axes
//!   (linear interpolation, sub-voxel; see `Field::crossing` for why not to a plane through
//!   them), and the rest of the band, layer by layer outward, the first-order solution of the
//!   eikonal equation |∇φ| = 1 from the layers before (as fast marching does, Sethian 1996, in
//!   the order of the layers rather than of the distances). The evolution stops when the zero
//!   crossings stay put (hardly a voxel changes sides, and they move less than a 500th of a
//!   voxel per iteration) twice running.
//! - **Guided vessel** ([`guided_vessel`]): the minimal path (Deschamps & Cohen 2001) between
//!   successive clicked points through a cost that is low deep inside the intensity range and
//!   high outside it (so the path keeps to the middle of the vessel, as the centrelines of
//!   Antiga et al. 2008), found by Dijkstra's algorithm on the 26-neighbour grid in a box around
//!   the two points; a thin tube (half the radius hint) along the joined path; and the level set
//!   grown from that tube, at most three radii from the path, out to the vessel's wall.
//!
//! References:
//! - V. Caselles, R. Kimmel, G. Sapiro, "Geodesic active contours", Int. J. Comput. Vision 22
//!   (1), 1997, pp. 61–79.
//! - D. Adalsteinsson, J. A. Sethian, "A fast level set method for propagating interfaces",
//!   J. Comput. Phys. 118 (2), 1995, pp. 269–277.
//! - R. T. Whitaker, "A level-set approach to 3D reconstruction from range data", Int. J.
//!   Comput. Vision 29 (3), 1998, pp. 203–231.
//! - J. A. Sethian, "A fast marching level set method for monotonically advancing fronts",
//!   Proc. Natl. Acad. Sci. USA 93 (4), 1996, pp. 1591–1595.
//! - T. Deschamps, L. D. Cohen, "Fast extraction of minimal paths in 3D images and
//!   applications to virtual endoscopy", Med. Image Anal. 5 (4), 2001, pp. 281–299.
//! - L. Antiga, M. Piccinelli, L. Botti, B. Ene-Iordache, A. Remuzzi, D. A. Steinman, "An
//!   image-based modeling framework for patient-specific computational hemodynamics", Med.
//!   Biol. Eng. Comput. 46 (11), 2008, pp. 1097–1112.

use crate::mask::{Connectivity, Mask};
use crate::ops::distance_sq;
use anyhow::{bail, Result};
use mm_volume::load::LoadedVolume;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Parameters of [`level_set`] (and of the level set of [`guided_vessel`]). The weights are
/// relative: propagation is in mm per unit of time, curvature and advection per voxel of the
/// finest spacing (so 0.3 means the same at any resolution).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LevelSetParams {
    /// Intensity range of the structure (values inside it propagate the front outward, outside it push back).
    pub range: (f32, f32),
    /// Weights: propagation (balloon), curvature (smoothness), advection (pull to edges).
    pub propagation: f64,
    pub curvature: f64,
    pub advection: f64,
    /// Smoothing (σ, mm) of the image before the edge map / gradient is taken.
    pub edge_sigma_mm: f64,
    pub iterations: usize,
    /// The front never moves farther than this from the initial mask (mm); 0 = no limit.
    pub max_distance_mm: f64,
}

impl Default for LevelSetParams {
    /// Contrast-filled vessels in CTA (Hounsfield units, calcium left out); for MRA, set the
    /// range from the image. The weights suit both: smoothing light enough for vessels a few
    /// voxels across, edges of the image smoothed by half a millimetre, a front that stops
    /// within 10 mm of the start.
    fn default() -> Self {
        Self { range: (200.0, 800.0), propagation: 1.0, curvature: 0.3, advection: 1.0, edge_sigma_mm: 0.5, iterations: 300, max_distance_mm: 10.0 }
    }
}

/// Width of the region speed's fall outside the range, as a fraction of the range's width
/// (kept between 0.2 % and 5 % of the volume's intensity range).
const RAMP: f32 = 0.1;
/// Reinitialise φ at least this often (iterations).
const REINIT_EVERY: usize = 20;
/// The front has stopped when, between reinitialisations, its zero crossings moved less than
/// this (voxels of the finest spacing per iteration, on average) and no more than one voxel in
/// ten thousand next to it changed sides, `STILL_FOR` times in a row.
const STILL: f32 = 2e-3;
const STILL_FOR: usize = 2;
/// Cost per mm of a path through voxels far outside the range (1 at the middle of the ramp;
/// inside the range it is at most about 0.44, less the deeper the voxel).
const OUTSIDE_COST: f32 = 10.0;
/// Cost per mm of a path deep inside a range wider than the box (keeps it short there).
const BASE_COST: f32 = 1e-3;
/// Share of [`guided_vessel`]'s progress taken by the paths.
const PATH_SHARE: f32 = 0.2;

/// Geodesic active contour from the segment `init`: its surface moves out through voxels in
/// `p.range` (or where `speed`, a feature image of the volume's size with values in [0, 1], is
/// above one half), back from the others, and settles on the edges of the image. Works on the
/// bounding box of `init` grown by `p.max_distance_mm` (the whole volume without a limit),
/// with about 25 bytes a voxel of it. `progress` gets the fraction done every iteration;
/// returning `false` cancels (an error). An empty `init` is an error; a segment that shrinks
/// away comes back empty (a seed a voxel or two across can, with a curvature weight above about
/// one half).
pub fn level_set(vol: &LoadedVolume, init: &Mask, speed: Option<&[f32]>, p: &LevelSetParams, progress: &dyn Fn(f32) -> bool) -> Result<Mask> {
    if init.dims != vol.dims || init.len() != vol.values.len() {
        bail!("mask and volume of different sizes");
    }
    if speed.is_some_and(|s| s.len() != vol.values.len()) {
        bail!("speed image and volume of different sizes");
    }
    let Some((lo, hi)) = init.bounds() else { bail!("the starting segment is empty") };
    if !progress(0.0) {
        bail!("cancelled");
    }
    let h = spacing(vol);
    let (hmin, hmax) = (h.iter().copied().fold(f64::INFINITY, f64::min), h.iter().copied().fold(0.0, f64::max));
    let (inner, outer) = band_widths(hmin, hmax);
    let sigma = if p.edge_sigma_mm > 0.0 { p.edge_sigma_mm } else { 0.0 };
    let limit = (p.max_distance_mm.is_finite() && p.max_distance_mm > 0.0).then_some(p.max_distance_mm);
    // Room for the front, its band and the Gaussian.
    let b = match limit {
        Some(d) => Region::around(lo, hi, h.map(|s| ((d + outer as f64 + 3.0 * sigma) / s).ceil() as usize + 2), vol.dims),
        None => Region { lo: [0; 3], dims: vol.dims },
    };
    if b.len() > u32::MAX as usize {
        bail!("the region is too large ({} million voxels)", b.len() / 1_000_000);
    }
    let floor_value = vol.range.0;
    let values = b.gather(&vol.values, vol.dims, |v| if v.is_finite() { v } else { floor_value });
    let region_speed: Vec<f32> = match speed {
        Some(s) => b.gather(s, vol.dims, |x| if x.is_finite() { (2.0 * x - 1.0).clamp(-1.0, 1.0) } else { -1.0 }),
        None => {
            let w = ramp_width(p.range, vol);
            b.gather(&vol.values, vol.dims, |v| range_speed(v, p.range, w))
        }
    };
    let g = edge_map(values, b.dims, h, sigma);
    let mut phi = b.gather(&init.data, vol.dims, |m| if m != 0 { -1.0f32 } else { 1.0 });
    // φ may not fall below (distance from the start) − limit: the front stays within it.
    let floor = limit.map(|d| {
        let start = b.gather(&init.data, vol.dims, |m| m);
        distance_sq(b.dims, &start, h).into_par_iter().map(|d2| (d2.sqrt() - d) as f32).collect()
    });
    let field = Field {
        dims: b.dims,
        strides: [1, b.dims[0], b.dims[0] * b.dims[1]],
        h: h.map(|x| x as f32),
        hmin: hmin as f32,
        hmax: hmax as f32,
        inv_h2: h.iter().map(|&x| 1.0 / (x * x)).sum::<f64>() as f32,
        inner,
        outer,
        speed: region_speed,
        g,
        floor,
        alpha: p.propagation as f32,
        beta: (p.curvature.max(0.0) * hmin) as f32,
        gamma: (p.advection.max(0.0) * hmin) as f32,
        kmax: (2.0 / hmin) as f32,
    };
    field.evolve(&mut phi, p.iterations, progress)?;
    let mut out = Mask::new(vol.dims);
    b.write(&mut out.data, vol.dims, |n| phi[n] < 0.0);
    Ok(out)
}

/// A vessel from `points` clicked along it (voxel coordinates, in order): the minimal paths
/// between successive points through a cost low deep inside `range`, a tube of half
/// `radius_mm` (a rough radius of the vessel) along them, and the level set of `p` grown from
/// that tube with the region speed of `range` (which replaces `p.range`), no farther than three
/// radii from the path (or `p.max_distance_mm` from the tube, if less). `progress` gets the
/// fraction done; returning `false` cancels (an error).
pub fn guided_vessel(vol: &LoadedVolume, points: &[[f64; 3]], range: (f32, f32), radius_mm: f64, p: &LevelSetParams, progress: &dyn Fn(f32) -> bool) -> Result<Mask> {
    if points.is_empty() {
        bail!("no points along the vessel");
    }
    if !(radius_mm.is_finite() && radius_mm > 0.0) {
        bail!("the vessel radius must be positive");
    }
    if vol.values.is_empty() || vol.values.len() != vol.voxel_count() {
        bail!("empty volume");
    }
    let dims = vol.dims;
    let voxels = points
        .iter()
        .map(|q| {
            if !q.iter().all(|x| x.is_finite()) {
                bail!("a point is not a number");
            }
            Ok([0, 1, 2].map(|a| q[a].round().clamp(0.0, (dims[a] - 1) as f64) as usize))
        })
        .collect::<Result<Vec<[usize; 3]>>>()?;
    if !progress(0.0) {
        bail!("cancelled");
    }
    let mut path = vec![voxels[0]];
    let legs = voxels.len() - 1;
    for (l, w) in voxels.windows(2).enumerate() {
        path.extend(minimal_path(vol, w[0], w[1], range, radius_mm).into_iter().skip(1));
        if !progress(PATH_SHARE * (l + 1) as f32 / legs as f32) {
            bail!("cancelled");
        }
    }
    let core = 0.5 * radius_mm;
    let init = tube(dims, &path, core, spacing(vol));
    let reach = 3.0 * radius_mm - core;
    let max_distance_mm = if p.max_distance_mm > 0.0 { p.max_distance_mm.min(reach) } else { reach };
    let q = LevelSetParams { range, max_distance_mm, ..*p };
    level_set(vol, &init, None, &q, &|f| progress(PATH_SHARE + (1.0 - PATH_SHARE) * f))
}

/// Voxel spacing in mm (1 where the file's is unusable).
fn spacing(vol: &LoadedVolume) -> [f64; 3] {
    vol.geometry.spacing.map(|s| if s > 0.0 && s.is_finite() { s } else { 1.0 })
}

/// Half-widths (mm) of the voxels updated around the front, and of the band whose distances
/// are kept (the updated voxels' neighbours, diagonal ones too, all in it).
fn band_widths(hmin: f64, hmax: f64) -> (f32, f32) {
    let inner = hmax + 2.0 * hmin;
    (inner as f32, (inner + 1.5 * hmax) as f32)
}

/// Width of the region speed's ramp for `range` in `vol`.
fn ramp_width(range: (f32, f32), vol: &LoadedVolume) -> f32 {
    let full = (vol.range.1 - vol.range.0).abs();
    let w = RAMP * (range.1 - range.0).abs();
    let w = if w.is_nan() { 0.05 * full } else { w.clamp(0.002 * full, 0.05 * full) };
    w.max(1e-6)
}

/// Region speed of value `v`: 1 in `range`, falling smoothly to −1 at `w` outside it (and
/// beyond); −1 for NaN.
fn range_speed(v: f32, range: (f32, f32), w: f32) -> f32 {
    let (lo, hi) = (range.0.min(range.1), range.0.max(range.1));
    let out = if v.is_nan() {
        1.0
    } else if v < lo {
        (lo - v) / w
    } else if v > hi {
        (v - hi) / w
    } else {
        return 1.0;
    };
    let s = out.min(1.0);
    1.0 - 2.0 * s * s * (3.0 - 2.0 * s)
}

/// A box of the voxel grid: its first voxel and its size.
#[derive(Clone, Copy, Debug)]
struct Region {
    lo: [usize; 3],
    dims: [usize; 3],
}

impl Region {
    /// The voxels from `lo` to `hi` (inclusive) and `pad` more along each axis, within a grid
    /// of `full`.
    fn around(lo: [usize; 3], hi: [usize; 3], pad: [usize; 3], full: [usize; 3]) -> Self {
        let a = [0, 1, 2].map(|x| lo[x].saturating_sub(pad[x]));
        let b = [0, 1, 2].map(|x| hi[x].saturating_add(pad[x]).min(full[x] - 1));
        Self { lo: a, dims: [0, 1, 2].map(|x| b[x] - a[x] + 1) }
    }

    fn len(&self) -> usize {
        self.dims.iter().product()
    }

    /// The box's index of grid voxel `g` (inside the box).
    fn local(&self, g: [usize; 3]) -> usize {
        ((g[2] - self.lo[2]) * self.dims[1] + g[1] - self.lo[1]) * self.dims[0] + g[0] - self.lo[0]
    }

    /// Grid voxel of the box's voxel `n`.
    fn grid(&self, n: usize) -> [usize; 3] {
        let [nx, ny, _] = self.dims;
        [n % nx + self.lo[0], (n / nx) % ny + self.lo[1], n / (nx * ny) + self.lo[2]]
    }

    /// `f` of each of the box's voxels of `data` (a grid of `full`), as a grid of its own.
    fn gather<T: Copy + Sync, U: Send>(&self, data: &[T], full: [usize; 3], f: impl Fn(T) -> U + Sync) -> Vec<U> {
        let [nx, ny, _] = self.dims;
        let f = &f;
        (0..ny * self.dims[2])
            .into_par_iter()
            .flat_map_iter(|r| {
                let at = ((self.lo[2] + r / ny) * full[1] + self.lo[1] + r % ny) * full[0] + self.lo[0];
                data[at..at + nx].iter().map(move |&v| f(v))
            })
            .collect()
    }

    /// Set the voxels of a grid mask `out` (of `full`) in the box to `f` of the box's index.
    fn write(&self, out: &mut [u8], full: [usize; 3], f: impl Fn(usize) -> bool + Sync) {
        let [bx, by, bz] = self.dims;
        out.par_chunks_mut((full[0] * full[1]).max(1)).enumerate().skip(self.lo[2]).take(bz).for_each(|(k, plane)| {
            for j in 0..by {
                let row = &mut plane[(j + self.lo[1]) * full[0] + self.lo[0]..][..bx];
                for (i, m) in row.iter_mut().enumerate() {
                    *m = f(((k - self.lo[2]) * by + j) * bx + i) as u8;
                }
            }
        });
    }
}

/// Separable Gaussian blur of a box of `dims` voxels, with standard deviation `sigma` voxels
/// along each axis (an axis is left alone below a twentieth of a voxel); the ends of each line
/// are repeated.
fn blur(values: &mut [f32], dims: [usize; 3], sigma: [f64; 3]) {
    let strides = [1, dims[0], dims[0] * dims[1]];
    let mut src = vec![0f32; values.len()];
    for ((&s, &len), &stride) in sigma.iter().zip(&dims).zip(&strides) {
        if s.is_nan() || s < 0.05 || len < 2 {
            continue;
        }
        // Wider than the line itself changes nothing more.
        let r = ((3.0 * s).ceil() as usize).min(2 * len);
        let w: Vec<f64> = (0..=2 * r).map(|t| (-((t as f64 - r as f64).powi(2)) / (2.0 * s * s)).exp()).collect();
        let total: f64 = w.iter().sum();
        let w: Vec<f32> = w.iter().map(|x| (x / total) as f32).collect();
        src.copy_from_slice(values);
        values.par_iter_mut().enumerate().for_each(|(n, v)| {
            let x = (n / stride) % len;
            let base = n - x * stride;
            *v = w.iter().enumerate().map(|(t, &wt)| wt * src[base + (x + t).saturating_sub(r).min(len - 1) * stride]).sum();
        });
    }
}

/// The edge-stopping function g = 1 / (1 + |∇(G_σ * I)|² / k²) of a box of `dims` intensities
/// `h` mm apart, gradients by central differences (one-sided at the box's faces); k is 2.5
/// times the median gradient magnitude in the box, or a tenth of its 99th percentile if that
/// is more. 1 everywhere in an even box.
fn edge_map(mut values: Vec<f32>, dims: [usize; 3], h: [f64; 3], sigma_mm: f64) -> Vec<f32> {
    blur(&mut values, dims, h.map(|s| sigma_mm / s));
    let strides = [1, dims[0], dims[0] * dims[1]];
    let grad: Vec<f32> = (0..values.len())
        .into_par_iter()
        .map(|n| {
            let mut g2 = 0.0f32;
            for ((&len, &s), &hh) in dims.iter().zip(&strides).zip(&h) {
                let x = (n / s) % len;
                let (lo, hi) = (if x > 0 { n - s } else { n }, if x + 1 < len { n + s } else { n });
                if lo != hi {
                    let d = (values[hi] - values[lo]) / (((hi - lo) / s) as f64 * hh) as f32;
                    g2 += d * d;
                }
            }
            g2.sqrt()
        })
        .collect();
    drop(values);
    // Percentiles of at most a million evenly spread values.
    let mut sample: Vec<f32> = grad.iter().step_by((grad.len() / 1_000_000).max(1)).copied().collect();
    sample.par_sort_unstable_by(f32::total_cmp);
    let at = |q: f64| sample[((sample.len() - 1) as f64 * q).round() as usize];
    let k = (2.5 * at(0.5)).max(0.1 * at(0.99));
    if !(k > 0.0 && k.is_finite()) {
        return vec![1.0; grad.len()];
    }
    grad.into_par_iter().map(|x| 1.0 / (1.0 + (x / k) * (x / k))).collect()
}

/// Voxels around the front: those whose distance is kept, those updated (a subset; the voxels
/// next to the front first), and those next to the front with their signed distance to it.
struct Band {
    all: Vec<u32>,
    active: Vec<u32>,
    front: Vec<(u32, f32)>,
}

/// States of voxels while φ is reinitialised: not reached, in the layer being solved, solved.
const FAR: u8 = 0;
const TRIAL: u8 = 1;
const KNOWN: u8 = 2;

/// What moves the front, over a box of voxels.
struct Field {
    dims: [usize; 3],
    strides: [usize; 3],
    /// Spacing (mm), its smallest and largest, and Σ 1/h².
    h: [f32; 3],
    hmin: f32,
    hmax: f32,
    inv_h2: f32,
    /// Half-widths (mm) of the updated voxels and of the band.
    inner: f32,
    outer: f32,
    /// Region speed P in [−1, 1].
    speed: Vec<f32>,
    /// Edge-stopping function g in (0, 1].
    g: Vec<f32>,
    /// Smallest φ allowed (the distance limit), if any.
    floor: Option<Vec<f32>>,
    /// Propagation weight; curvature and advection weights times the finest spacing.
    alpha: f32,
    beta: f32,
    gamma: f32,
    /// Largest mean curvature taken (1/mm): a sphere's of one voxel radius.
    kmax: f32,
}

impl Field {
    /// The voxels before and after `n` along each axis (`n` itself at the box's faces).
    #[inline]
    fn around(&self, n: usize) -> ([usize; 3], [usize; 3]) {
        let mut lo = [n; 3];
        let mut hi = [n; 3];
        for (a, (l, u)) in lo.iter_mut().zip(hi.iter_mut()).enumerate() {
            let s = self.strides[a];
            let x = (n / s) % self.dims[a];
            if x > 0 {
                *l = n - s;
            }
            if x + 1 < self.dims[a] {
                *u = n + s;
            }
        }
        (lo, hi)
    }

    /// ∂φ/∂t at `n`, and the inverse of the largest stable time step there.
    fn rate(&self, phi: &[f32], n: usize) -> (f32, f32) {
        let (lo, hi) = self.around(n);
        let f0 = phi[n];
        // One-sided differences back and forward, central first and second differences, the
        // central difference's span (mm), and the gradient of g.
        let mut back = [0f32; 3];
        let mut fwd = [0f32; 3];
        let mut d1 = [0f32; 3];
        let mut d2 = [0f32; 3];
        let mut span = [0f32; 3];
        let mut dg = [0f32; 3];
        for a in 0..3 {
            let h = self.h[a];
            let (fm, fp) = (phi[lo[a]], phi[hi[a]]);
            back[a] = (f0 - fm) / h;
            fwd[a] = (fp - f0) / h;
            d2[a] = (fp - 2.0 * f0 + fm) / (h * h);
            span[a] = ((hi[a] - lo[a]) / self.strides[a]) as f32 * h;
            if span[a] > 0.0 {
                d1[a] = (fp - fm) / span[a];
                dg[a] = (self.g[hi[a]] - self.g[lo[a]]) / span[a];
            }
        }
        let mixed = |a: usize, b: usize| -> f32 {
            if span[a] == 0.0 || span[b] == 0.0 {
                return 0.0;
            }
            (phi[hi[a] + hi[b] - n] - phi[hi[a] + lo[b] - n] - phi[lo[a] + hi[b] - n] + phi[lo[a] + lo[b] - n]) / (span[a] * span[b])
        };
        // κ |∇φ|, κ = div(∇φ / |∇φ|) limited to ± kmax: it vanishes with the gradient at a
        // minimum of φ (a seed one voxel thick), where the formula itself is undefined.
        let [x, y, z] = d1;
        let grad2 = x * x + y * y + z * z;
        let curvature = if grad2 > 1e-12 {
            let norm = grad2.sqrt();
            let kappa = (d2[0] * (y * y + z * z) + d2[1] * (x * x + z * z) + d2[2] * (x * x + y * y) - 2.0 * (x * y * mixed(0, 1) + x * z * mixed(0, 2) + y * z * mixed(1, 2))) / (grad2 * norm);
            kappa.clamp(-self.kmax, self.kmax) * norm
        } else {
            0.0
        };
        let g0 = self.g[n];
        // Balloon: |∇φ| upwind of the front's motion.
        let f = self.alpha * self.speed[n];
        let mut up = 0.0f32;
        for (&b, &u) in back.iter().zip(&fwd) {
            let (b, u) = if f > 0.0 { (b.max(0.0), u.min(0.0)) } else { (b.min(0.0), u.max(0.0)) };
            up += b * b + u * u;
        }
        // Advection along −∇g, upwind.
        let (mut adv, mut adv_abs) = (0.0f32, 0.0f32);
        for a in 0..3 {
            let v = -self.gamma * dg[a];
            adv += v * if v > 0.0 { back[a] } else { fwd[a] };
            adv_abs += v.abs();
        }
        let rate = -f * up.sqrt() + self.beta * g0 * curvature - adv;
        let stiff = 2.0 * (f.abs() + adv_abs) / self.hmin + 2.0 * self.beta * g0 * self.inv_h2;
        (rate, stiff)
    }

    /// Evolve φ for at most `iterations` steps (fewer when the front stops).
    fn evolve(&self, phi: &mut [f32], iterations: usize, progress: &dyn Fn(f32) -> bool) -> Result<()> {
        let mut state = vec![FAR; phi.len()];
        let mut band = self.reinit(phi, None, &mut state);
        // Signed distance to the front of the voxels next to it at the last reinitialisation.
        let mut last = vec![f32::NAN; phi.len()];
        for &(n, d) in &band.front {
            last[n as usize] = d;
        }
        // The change of φ in the last step (updated voxels only).
        let mut delta = vec![0f32; phi.len()];
        let (mut moved, mut since, mut still) = (0f32, 0usize, 0usize);
        for it in 0..iterations {
            if !progress(it as f32 / iterations as f32) {
                bail!("cancelled");
            }
            if band.front.is_empty() {
                break; // gone, or filling the box
            }
            let rates: Vec<(f32, f32)> = band.active.par_iter().map(|&n| self.rate(phi, n as usize)).collect();
            let stiffest = rates.iter().fold(0.0f32, |m, r| m.max(r.1));
            if !(stiffest > 0.0 && stiffest.is_finite()) {
                break; // nothing drives it
            }
            let dt = 1.0 / stiffest;
            for (&n, &(r, _)) in band.active.iter().zip(&rates) {
                let n = n as usize;
                let mut v = phi[n] + dt * r;
                if let Some(f) = &self.floor {
                    v = v.max(f[n]);
                }
                delta[n] = v - phi[n];
                phi[n] = v;
            }
            moved += self.front_step(phi, &delta, &band.active);
            since += 1;
            // Before the front can leave the updated voxels.
            if moved < self.inner - self.hmax && since < REINIT_EVERY {
                continue;
            }
            let before = std::mem::take(&mut band.front);
            band = self.reinit(phi, Some(&band.all), &mut state);
            // How far the front moved since the last time: the mean change per iteration of the
            // distances next to it (a voxel newly next to it counting as a voxel's move), and
            // how many voxels changed sides.
            let (mut total, mut flips) = (0f32, 0usize);
            for &(n, d) in &band.front {
                let was = last[n as usize];
                if was.is_nan() || (was < 0.0) != (d < 0.0) {
                    flips += 1;
                }
                total += if was.is_nan() { self.hmin } else { (d - was).abs() };
            }
            let shift = total / band.front.len().max(1) as f32 / since as f32;
            for (n, _) in before {
                last[n as usize] = f32::NAN;
            }
            for &(n, d) in &band.front {
                last[n as usize] = d;
            }
            (moved, since) = (0.0, 0);
            still = if shift < STILL * self.hmin && flips <= band.front.len() / 10_000 { still + 1 } else { 0 };
            if still >= STILL_FOR {
                break;
            }
        }
        if !progress(1.0) {
            bail!("cancelled");
        }
        Ok(())
    }

    /// How far (mm) the front moved in the last step at most: the shift of the zero crossings
    /// of φ between 6-neighbours, from φ before and after the step on either side of each (both
    /// updated, being next to the front; a crossing new to the pair counts from the end it came
    /// past). Unlike the change of φ itself, it is nought where φ only grows steeper across a
    /// front that stays put.
    fn front_step(&self, phi: &[f32], delta: &[f32], active: &[u32]) -> f32 {
        active
            .par_chunks(4096)
            .map(|chunk| {
                let mut most = 0f32;
                for &n in chunk {
                    let n = n as usize;
                    let f0 = phi[n];
                    if f0 >= 0.0 {
                        continue;
                    }
                    let (lo, hi) = self.around(n);
                    for (a, &h) in self.h.iter().enumerate() {
                        for q in [lo[a], hi[a]] {
                            let fq = phi[q];
                            if q != n && fq >= 0.0 {
                                let t = f0 / (f0 - fq);
                                let (b0, bq) = (f0 - delta[n], fq - delta[q]);
                                let was = if b0 == bq { 0.5 } else { (b0 / (b0 - bq)).clamp(0.0, 1.0) };
                                most = most.max((t - was).abs() * h);
                            }
                        }
                    }
                }
                most
            })
            .reduce(|| 0.0, f32::max)
    }

    /// Distance (mm) from voxel `n` to the front along the axes, if a 6-neighbour is on the
    /// other side of it: the nearest zero crossing of φ (linear interpolation) on the six lines
    /// to its neighbours. Along a plane this overestimates the distance by the same factor on
    /// both sides of the front, so the crossings stay where they were; the distance to a plane
    /// through the crossings would, but would also wear away anything one or two voxels thick
    /// a little at each reinitialisation.
    fn crossing(&self, phi: &[f32], n: usize) -> Option<f32> {
        let f0 = phi[n];
        let inside = f0 < 0.0;
        let (lo, hi) = self.around(n);
        let mut nearest = f32::INFINITY;
        for (a, &h) in self.h.iter().enumerate() {
            for q in [lo[a], hi[a]] {
                let fq = phi[q];
                if q != n && (fq < 0.0) != inside {
                    nearest = nearest.min((f0 / (f0 - fq) * h).max(1e-4 * h));
                }
            }
        }
        nearest.is_finite().then_some(nearest)
    }

    /// Distance at `n` from its known neighbours: the first-order upwind solution of |∇T| = 1,
    /// taking the axes in increasing order of their nearest known neighbour while the solution
    /// stays above it.
    fn eikonal(&self, n: usize, phi: &[f32], state: &[u8]) -> f32 {
        let (lo, hi) = self.around(n);
        let mut near = [(0f32, 0f32); 3];
        let mut c = 0;
        for (a, &h) in self.h.iter().enumerate() {
            let mut v = f32::INFINITY;
            for q in [lo[a], hi[a]] {
                if q != n && state[q] == KNOWN {
                    v = v.min(phi[q].abs());
                }
            }
            if v.is_finite() {
                near[c] = (v, h);
                c += 1;
            }
        }
        near[..c].sort_by(|x, y| x.0.total_cmp(&y.0));
        let (mut sa, mut sb, mut sc, mut t) = (0f32, 0f32, 0f32, f32::INFINITY);
        for &(v, h) in &near[..c] {
            if t <= v {
                break;
            }
            let w = 1.0 / (h * h);
            (sa, sb, sc) = (sa + w, sb + w * v, sc + w * v * v);
            let disc = sb * sb - sa * (sc - 1.0);
            if disc < 0.0 {
                break;
            }
            t = (sb + disc.sqrt()) / sa;
        }
        t
    }

    /// Make φ a signed distance (mm) again within the band, keeping its zero crossings, and
    /// find the new band. The front lies among `previous` (the last band), or anywhere at the
    /// start. The voxels next to it take their distance from [`Field::crossing`]; the others,
    /// layer after layer of 6-neighbours outward, solve the eikonal equation from the layers
    /// before them and then once more with their own layer's values, until a layer lies wholly
    /// beyond the band (the order of fast marching only roughly, but in parallel and without a
    /// heap). Voxels left out of the band keep their sign. `state` is all `FAR`, and stays so.
    fn reinit(&self, phi: &mut [f32], previous: Option<&[u32]>, state: &mut [u8]) -> Band {
        let signed = |n: usize| self.crossing(phi, n).map(|d| (n as u32, if phi[n] < 0.0 { -d } else { d }));
        let front: Vec<(u32, f32)> = match previous {
            Some(p) => p.par_iter().filter_map(|&n| signed(n as usize)).collect(),
            None => (0..phi.len()).into_par_iter().filter_map(signed).collect(),
        };
        let far = self.outer + self.hmax;
        let clear = |v: &mut f32| *v = if *v < 0.0 { -far } else { far };
        match previous {
            Some(p) => p.iter().for_each(|&n| clear(&mut phi[n as usize])),
            None => phi.par_iter_mut().for_each(clear),
        }
        let mut band = Band { all: Vec::with_capacity(front.len() * 8), active: Vec::with_capacity(front.len() * 6), front: Vec::new() };
        let mut touched: Vec<u32> = Vec::with_capacity(front.len() * 8);
        for &(n, d) in &front {
            phi[n as usize] = d;
            state[n as usize] = KNOWN;
            touched.push(n);
            band.all.push(n);
            band.active.push(n);
        }
        let mut layer: Vec<u32> = front.iter().map(|f| f.0).collect();
        loop {
            let mut next = Vec::with_capacity(layer.len() + layer.len() / 4);
            for &n in &layer {
                let (lo, hi) = self.around(n as usize);
                for m in lo.into_iter().chain(hi) {
                    if state[m] == FAR {
                        state[m] = TRIAL;
                        next.push(m as u32);
                    }
                }
            }
            if next.is_empty() {
                break;
            }
            // Twice: from the layers before, then with the layer's own values too.
            for _ in 0..2 {
                let t: Vec<f32> = next.par_iter().map(|&m| self.eikonal(m as usize, phi, state)).collect();
                for (&m, &t) in next.iter().zip(&t) {
                    let m = m as usize;
                    let t = if state[m] == KNOWN { t.min(phi[m].abs()) } else { t };
                    phi[m] = if phi[m] < 0.0 { -t } else { t };
                }
                for &m in &next {
                    state[m as usize] = KNOWN;
                }
            }
            touched.extend_from_slice(&next);
            next.retain(|&m| phi[m as usize].abs() <= self.outer);
            for &m in &next {
                band.all.push(m);
                if phi[m as usize].abs() <= self.inner {
                    band.active.push(m);
                }
            }
            layer = next;
        }
        for n in touched {
            state[n as usize] = FAR;
        }
        band.front = front;
        band
    }

}

/// A voxel waiting in a [`Queue`].
#[derive(Clone, Copy)]
struct Item {
    key: f32,
    at: u32,
}

impl Ord for Item {
    /// Reversed (`BinaryHeap` pops the greatest): smallest key first.
    fn cmp(&self, o: &Self) -> Ordering {
        o.key.total_cmp(&self.key).then(o.at.cmp(&self.at))
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

/// Voxels by smallest key.
#[derive(Default)]
struct Queue {
    heap: BinaryHeap<Item>,
}

impl Queue {
    fn push(&mut self, key: f32, at: usize) {
        self.heap.push(Item { key, at: at as u32 });
    }

    fn pop(&mut self) -> Option<(f32, usize)> {
        self.heap.pop().map(|i| (i.key, i.at as usize))
    }
}

/// The cheapest 26-connected path of voxels from `a` to `b` (both included), in a box around
/// them grown by three radii or a third of their distance. A millimetre costs
/// `(h / (d + h/2))²` (plus a little) inside `range`, d being the depth (mm) from the nearest
/// voxel outside it and h the finest spacing, so the path keeps to the middle of the vessel;
/// outside it costs from 1 to [`OUTSIDE_COST`], the more the farther the value is from the
/// range.
fn minimal_path(vol: &LoadedVolume, a: [usize; 3], b: [usize; 3], range: (f32, f32), radius_mm: f64) -> Vec<[usize; 3]> {
    if a == b {
        return vec![a];
    }
    let h = spacing(vol);
    let (hmin, hmax) = (h.iter().copied().fold(f64::INFINITY, f64::min), h.iter().copied().fold(0.0, f64::max));
    let length = (0..3).map(|x| ((a[x] as f64 - b[x] as f64) * h[x]).powi(2)).sum::<f64>().sqrt();
    let margin = (3.0 * radius_mm).max(length / 3.0) + 2.0 * hmax;
    let lo = [0, 1, 2].map(|x| a[x].min(b[x]));
    let hi = [0, 1, 2].map(|x| a[x].max(b[x]));
    let r = Region::around(lo, hi, h.map(|s| (margin / s).ceil() as usize), vol.dims);
    let w = ramp_width(range, vol);
    let speed = r.gather(&vol.values, vol.dims, |v| range_speed(v, range, w));
    let outside: Vec<u8> = speed.par_iter().map(|&s| (s <= 0.0) as u8).collect();
    let depth = distance_sq(r.dims, &outside, h);
    let hm = hmin as f32;
    let cost: Vec<f32> = speed
        .par_iter()
        .zip(&depth)
        .map(|(&s, &d2)| if s <= 0.0 { 1.0 - (OUTSIDE_COST - 1.0) * s } else { BASE_COST + (hm / (d2.sqrt() as f32 + 0.5 * hm)).powi(2) })
        .collect();
    drop(depth);
    // Dijkstra from a until b comes out.
    let [nx, ny, nz] = r.dims.map(|d| d as i64);
    let steps: Vec<([i64; 3], i64, f32)> = Connectivity::TwentySix
        .offsets()
        .into_iter()
        .map(|o| (o, o[0] + o[1] * nx + o[2] * nx * ny, (0..3).map(|x| (o[x] as f64 * h[x]).powi(2)).sum::<f64>().sqrt() as f32))
        .collect();
    let (start, goal) = (r.local(a), r.local(b));
    let mut dist = vec![f32::INFINITY; r.len()];
    let mut from = vec![u32::MAX; r.len()];
    let mut queue = Queue::default();
    dist[start] = 0.0;
    queue.push(0.0, start);
    while let Some((d, p)) = queue.pop() {
        if d > dist[p] {
            continue;
        }
        if p == goal {
            break;
        }
        let (i, j, k) = (p as i64 % nx, (p as i64 / nx) % ny, p as i64 / (nx * ny));
        for &(o, step, len) in &steps {
            let (x, y, z) = (i + o[0], j + o[1], k + o[2]);
            if x < 0 || y < 0 || z < 0 || x >= nx || y >= ny || z >= nz {
                continue;
            }
            let q = (p as i64 + step) as usize;
            let dq = d + 0.5 * (cost[p] + cost[q]) * len;
            if dq < dist[q] {
                dist[q] = dq;
                from[q] = p as u32;
                queue.push(dq, q);
            }
        }
    }
    let mut path = vec![b];
    let mut p = goal;
    while p != start {
        p = from[p] as usize;
        path.push(r.grid(p));
    }
    path.reverse();
    path
}

/// The voxels within `radius_mm` of the voxels of `path`, in a grid of `dims` voxels `h` mm apart.
fn tube(dims: [usize; 3], path: &[[usize; 3]], radius_mm: f64, h: [f64; 3]) -> Mask {
    let lo = path.iter().fold([usize::MAX; 3], |m, p| [0, 1, 2].map(|x| m[x].min(p[x])));
    let hi = path.iter().fold([0; 3], |m, p| [0, 1, 2].map(|x| m[x].max(p[x])));
    let r = Region::around(lo, hi, h.map(|s| (radius_mm / s).ceil() as usize + 1), dims);
    let mut sites = vec![0u8; r.len()];
    for &p in path {
        sites[r.local(p)] = 1;
    }
    let d2 = distance_sq(r.dims, &sites, h);
    let lim = radius_mm * radius_mm + 1e-9;
    let mut out = Mask::new(dims);
    r.write(&mut out.data, dims, |n| d2[n] <= lim);
    out
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

    fn volume(dims: [usize; 3], spacing: [f64; 3], values: Vec<f32>) -> LoadedVolume {
        LoadedVolume::from_values("test", dims, values, Geometry::from_axes([dims[0], dims[1], dims[2], 1], spacing, [DVec3::X, DVec3::Y, DVec3::Z], DVec3::ZERO))
    }

    /// The voxels whose centres (in mm) pass `inside`.
    fn shape(dims: [usize; 3], spacing: [f64; 3], inside: impl Fn(DVec3) -> bool) -> Mask {
        let mut m = Mask::new(dims);
        for n in 0..m.len() {
            let [i, j, k] = m.ijk(n);
            m.data[n] = inside(DVec3::new(i as f64 * spacing[0], j as f64 * spacing[1], k as f64 * spacing[2])) as u8;
        }
        m
    }

    /// 100 on `truth`, 0 elsewhere, plus noise of ± `amp`.
    fn image(truth: &Mask, spacing: [f64; 3], amp: f32) -> LoadedVolume {
        let values = truth.data.iter().enumerate().map(|(n, &t)| 100.0 * t as f32 + amp * noise(n)).collect();
        volume(truth.dims, spacing, values)
    }

    fn dice(a: &Mask, b: &Mask) -> f64 {
        let both = a.data.iter().zip(&b.data).filter(|(&x, &y)| x != 0 && y != 0).count();
        2.0 * both as f64 / (a.count() + b.count()) as f64
    }

    /// Grown (or, with `shrink`, shrunk) by one voxel across its faces.
    fn one_voxel(m: &Mask, shrink: bool) -> Mask {
        let offsets = Connectivity::Six.offsets();
        let mut out = m.clone();
        for n in 0..m.len() {
            let [i, j, k] = m.ijk(n);
            for o in &offsets {
                let (a, b, c) = (i as i64 + o[0], j as i64 + o[1], k as i64 + o[2]);
                let inside = a >= 0 && b >= 0 && c >= 0 && (a as usize) < m.dims[0] && (b as usize) < m.dims[1] && (c as usize) < m.dims[2];
                let v = inside && m.get(a as usize, b as usize, c as usize);
                if shrink && !v {
                    out.data[n] = 0;
                } else if !shrink && v {
                    out.data[n] = 1;
                }
            }
        }
        out
    }

    /// Every voxel of `got` is within a voxel of `truth`, and every voxel more than a voxel
    /// inside `truth` is in `got`, except in the slices along axis `axis` outside `keep`.
    fn within_a_voxel(got: &Mask, truth: &Mask, axis: usize, keep: std::ops::Range<usize>) {
        let (outer, inner) = (one_voxel(truth, false), one_voxel(truth, true));
        for n in 0..got.len() {
            let p = got.ijk(n);
            assert!(got.data[n] == 0 || outer.data[n] != 0, "leaked at {p:?}");
            if keep.contains(&p[axis]) {
                assert!(inner.data[n] == 0 || got.data[n] != 0, "short of the wall at {p:?}");
            }
        }
    }

    fn params(range: (f32, f32)) -> LevelSetParams {
        LevelSetParams { range, iterations: 600, max_distance_mm: 0.0, ..Default::default() }
    }

    #[test]
    fn grows_along_a_cylinder_to_its_wall() {
        let dims = [40, 40, 48];
        let sp = [1.0; 3];
        let truth = shape(dims, sp, |p| (p.x - 20.0).hypot(p.y - 19.6) <= 4.0);
        let vol = image(&truth, sp, 25.0);
        let seed = shape(dims, sp, |p| p.distance(DVec3::new(20.0, 20.0, 24.0)) <= 1.5);
        let steps = std::cell::Cell::new(0);
        let t0 = Instant::now();
        let out = level_set(&vol, &seed, None, &params((50.0, 1000.0)), &|_| {
            steps.set(steps.get() + 1);
            true
        })
        .unwrap();
        eprintln!("cylinder: {:.3} s, {} iterations, Dice {:.3}", t0.elapsed().as_secs_f64(), steps.get() - 2, dice(&out, &truth));
        within_a_voxel(&out, &truth, 2, 3..45);
        assert!(dice(&out, &truth) > 0.95);
        assert!(steps.get() < 400, "stops when the front does");
        // With a distance limit the front stops short of the ends.
        let p = LevelSetParams { max_distance_mm: 8.0, ..params((50.0, 1000.0)) };
        let out = level_set(&vol, &seed, None, &p, &|_| true).unwrap();
        within_a_voxel(&out, &truth, 2, 17..32);
        assert!(out.data.iter().enumerate().all(|(n, &m)| m == 0 || out.ijk(n)[2].abs_diff(24) <= 10));
    }

    #[test]
    fn follows_a_thin_vessel_from_a_thin_seed() {
        // 1.6 voxels in radius, partial-volume edges; the seed is three voxels in a row.
        let dims = [64, 24, 24];
        let sp = [1.0; 3];
        let depth = |p: DVec3| (p.y - 12.0).hypot(p.z - 12.3) - 1.6;
        let truth = shape(dims, sp, |p| depth(p) <= 0.0 && (4.0..60.0).contains(&p.x));
        let values = (0..truth.len())
            .map(|n| {
                let [i, j, k] = truth.ijk(n);
                let p = DVec3::new(i as f64, j as f64, k as f64);
                let frac = if (4..60).contains(&i) { (0.5 - depth(p)).clamp(0.0, 1.0) } else { 0.0 };
                100.0 * frac as f32 + 10.0 * noise(n)
            })
            .collect();
        let vol = volume(dims, sp, values);
        let mut seed = Mask::new(dims);
        for i in 31..34 {
            seed.set(i, 12, 12, true);
        }
        let out = level_set(&vol, &seed, None, &params((50.0, 1000.0)), &|_| true).unwrap();
        eprintln!("thin vessel: Dice {:.3}", dice(&out, &truth));
        within_a_voxel(&out, &truth, 0, 5..59);
        assert!(dice(&out, &truth) > 0.85);
    }

    #[test]
    fn anisotropic_voxels() {
        // Along i, 3.2 mm across: 6 voxels in j, 2 in k each side of the axis.
        let dims = [64, 32, 12];
        let sp = [0.5, 0.5, 1.5];
        let truth = shape(dims, sp, |p| (p.y - 8.0).hypot(p.z - 8.25) <= 3.2);
        let vol = image(&truth, sp, 10.0);
        let seed = shape(dims, sp, |p| p.distance(DVec3::new(16.0, 8.0, 8.25)) <= 1.2);
        let out = level_set(&vol, &seed, None, &params((50.0, 1000.0)), &|_| true).unwrap();
        eprintln!("anisotropic: Dice {:.3}", dice(&out, &truth));
        within_a_voxel(&out, &truth, 0, 4..60);
        assert!(dice(&out, &truth) > 0.9);
    }

    #[test]
    fn a_speed_image_replaces_the_range() {
        let dims = [32, 32, 32];
        let sp = [1.0; 3];
        let truth = shape(dims, sp, |p| (p.x - 16.0).hypot(p.z - 15.5) <= 4.5);
        let vol = image(&truth, sp, 10.0);
        let seed = shape(dims, sp, |p| p.distance(DVec3::new(16.0, 16.0, 16.0)) <= 2.0);
        // Nothing is in this range: the seed shrinks away.
        let wrong = params((500.0, 600.0));
        assert_eq!(level_set(&vol, &seed, None, &wrong, &|_| true).unwrap().count(), 0);
        let speed: Vec<f32> = truth.data.iter().map(|&t| t as f32).collect();
        let out = level_set(&vol, &seed, Some(&speed), &wrong, &|_| true).unwrap();
        within_a_voxel(&out, &truth, 1, 2..30);
        assert!(dice(&out, &truth) > 0.9);
    }

    /// A quarter of a torus (major radius `major`, tube radius 3 voxels) around (2, 2) in the
    /// plane k = 8.
    fn torus(dims: [usize; 3], major: f64) -> Mask {
        shape(dims, [1.0; 3], |p| {
            let (x, y) = (p.x - 2.0, p.y - 2.0);
            x >= 0.0 && y >= 0.0 && (x.hypot(y) - major).hypot(p.z - 8.0) <= 3.0
        })
    }

    #[test]
    fn guided_vessel_follows_the_clicked_tube() {
        let dims = [52, 52, 17];
        // The vessel, and one beside it, 3 voxels from its wall.
        let (truth, other) = (torus(dims, 44.0), torus(dims, 35.0));
        let mut both = truth.clone();
        both.union(&other);
        let vol = image(&both, [1.0; 3], 10.0);
        let points: Vec<[f64; 3]> = [6.0f64, 32.0, 58.0, 84.0].iter().map(|a| [2.0 + 44.0 * a.to_radians().cos(), 2.0 + 44.0 * a.to_radians().sin(), 8.0]).collect();
        let t0 = Instant::now();
        let out = guided_vessel(&vol, &points, (50.0, 1000.0), 3.0, &LevelSetParams::default(), &|_| true).unwrap();
        let d = dice(&out, &truth);
        eprintln!("guided: {:.3} s, Dice {d:.3}", t0.elapsed().as_secs_f64());
        assert!(d > 0.8, "Dice {d}");
        let mut shared = out.clone();
        shared.intersect(&other);
        assert_eq!(shared.count(), 0, "the tube beside it");
        within_a_voxel(&out, &truth, 2, 0..0);
        // Two points close together, and one alone.
        let out = guided_vessel(&vol, &points[1..2], (50.0, 1000.0), 3.0, &LevelSetParams::default(), &|_| true).unwrap();
        assert!(out.count() > 100);
        let mut shared = out.clone();
        shared.intersect(&other);
        assert_eq!(shared.count(), 0);
    }

    #[test]
    fn cancel_and_empty_inputs() {
        let dims = [20, 20, 20];
        let truth = shape(dims, [1.0; 3], |p| (p.x - 10.0).hypot(p.y - 10.0) <= 3.0);
        let vol = image(&truth, [1.0; 3], 5.0);
        let seed = shape(dims, [1.0; 3], |p| p.distance(DVec3::splat(10.0)) <= 1.5);
        let p = params((50.0, 1000.0));
        assert!(level_set(&vol, &seed, None, &p, &|_| false).is_err());
        assert!(level_set(&vol, &seed, None, &p, &|f| f < 0.1).is_err());
        assert!(level_set(&vol, &Mask::new(dims), None, &p, &|_| true).is_err());
        assert!(level_set(&vol, &Mask::new([2, 2, 2]), None, &p, &|_| true).is_err());
        assert!(level_set(&vol, &seed, Some(&[0.5; 3]), &p, &|_| true).is_err());
        let line = [[10.0, 10.0, 3.0], [10.0, 10.0, 16.0]];
        assert!(guided_vessel(&vol, &line, (50.0, 1000.0), 3.0, &p, &|_| false).is_err());
        assert!(guided_vessel(&vol, &line, (50.0, 1000.0), 3.0, &p, &|f| f < 0.5).is_err());
        assert!(guided_vessel(&vol, &[], (50.0, 1000.0), 3.0, &p, &|_| true).is_err());
        assert!(guided_vessel(&vol, &line, (50.0, 1000.0), 0.0, &p, &|_| true).is_err());
        assert!(guided_vessel(&vol, &[[f64::NAN; 3]], (50.0, 1000.0), 3.0, &p, &|_| true).is_err());
        // No iterations: the start as it was; a one-voxel volume.
        let none = LevelSetParams { iterations: 0, ..p };
        assert_eq!(level_set(&vol, &seed, None, &none, &|_| true).unwrap(), seed);
        let one = volume([1; 3], [1.0; 3], vec![100.0]);
        let all = Mask { dims: [1; 3], data: vec![1] };
        assert_eq!(level_set(&one, &all, None, &p, &|_| true).unwrap(), all);
        assert_eq!(guided_vessel(&one, &[[0.0; 3]], (50.0, 1000.0), 1.0, &p, &|_| true).unwrap(), all);
    }

    #[test]
    #[ignore = "timing at an MRA's size (about 400 MB): run with --release --ignored"]
    fn realistic_size() {
        // An MRA-like grid: 512×512×160 voxels of 0.4×0.4×0.8 mm; a vessel tree in mm.
        let dims = [512usize, 512, 160];
        let sp = [0.4, 0.4, 0.8];
        let seg = |p: DVec3, a: DVec3, b: DVec3| {
            let t = ((p - a).dot(b - a) / (b - a).length_squared()).clamp(0.0, 1.0);
            p.distance(a + (b - a) * t)
        };
        let segs = [
            (DVec3::new(20.0, 100.0, 64.0), DVec3::new(100.0, 100.0, 64.0), 2.5),
            (DVec3::new(100.0, 100.0, 64.0), DVec3::new(180.0, 120.0, 70.0), 2.0),
            (DVec3::new(100.0, 100.0, 64.0), DVec3::new(140.0, 40.0, 50.0), 1.5),
            (DVec3::new(140.0, 40.0, 50.0), DVec3::new(190.0, 20.0, 60.0), 1.2),
            (DVec3::new(60.0, 100.0, 64.0), DVec3::new(60.0, 180.0, 90.0), 1.0),
        ];
        let depth = |p: DVec3| segs.iter().map(|&(a, b, r)| seg(p, a, b) - r).fold(f64::INFINITY, f64::min);
        let n = dims[0] * dims[1] * dims[2];
        let t0 = Instant::now();
        let (values, truth): (Vec<f32>, Vec<u8>) = (0..n)
            .into_par_iter()
            .map(|q| {
                let p = DVec3::new((q % dims[0]) as f64 * sp[0], ((q / dims[0]) % dims[1]) as f64 * sp[1], (q / (dims[0] * dims[1])) as f64 * sp[2]);
                let d = depth(p);
                let frac = (0.5 - d / 0.5).clamp(0.0, 1.0) as f32;
                (100.0 + 300.0 * frac + 40.0 * noise(q), (d <= 0.0) as u8)
            })
            .unzip();
        let truth = Mask { dims, data: truth };
        let vol = volume(dims, sp, values);
        eprintln!("made in {:.2} s, {} vessel voxels", t0.elapsed().as_secs_f64(), truth.count());
        let range = (250.0, 2000.0);
        let mm = |x: f64, y: f64, z: f64| [x / sp[0], y / sp[1], z / sp[2]];
        // Guided along the trunk and the second branch (about 160 mm).
        let points = [mm(25.0, 100.0, 64.0), mm(70.0, 100.0, 64.0), mm(120.0, 105.0, 65.5), mm(175.0, 119.0, 69.6)];
        let p = LevelSetParams { range, ..Default::default() };
        let t0 = Instant::now();
        let out = guided_vessel(&vol, &points, range, 2.0, &p, &|_| true).unwrap();
        let mut hit = out.clone();
        hit.intersect(&truth);
        eprintln!("guided 4 points: {:.2} s, {} voxels, {:.3} in the tree", t0.elapsed().as_secs_f64(), out.count(), hit.count() as f64 / out.count() as f64);
        // From the whole tree shrunk by 0.8 mm, limit 10 mm.
        let mut rough = truth.clone();
        crate::ops::shrink(&mut rough, 0.8, sp);
        let t0 = Instant::now();
        let out = level_set(&vol, &rough, None, &p, &|_| true).unwrap();
        eprintln!("rough tree, limit 10 mm: {:.2} s, Dice {:.3}", t0.elapsed().as_secs_f64(), dice(&out, &truth));
        // From a seed, limit 60 mm.
        let mut seed = Mask::new(dims);
        let c = mm(100.0, 100.0, 64.0).map(|x| x.round() as usize);
        for k in c[2] - 1..=c[2] + 1 {
            for j in c[1] - 2..=c[1] + 2 {
                for i in c[0] - 2..=c[0] + 2 {
                    seed.set(i, j, k, true);
                }
            }
        }
        let steps = std::cell::Cell::new(0);
        let q = LevelSetParams { max_distance_mm: 60.0, iterations: 2000, ..p };
        let t0 = Instant::now();
        let out = level_set(&vol, &seed, None, &q, &|_| {
            steps.set(steps.get() + 1);
            true
        })
        .unwrap();
        eprintln!("seed, limit 60 mm: {:.2} s, {} iterations, Dice {:.3}", t0.elapsed().as_secs_f64(), steps.get() - 2, dice(&out, &truth));
    }

    /// A small vessel tree in a 128×128×64 volume: a trunk along i with two branches.
    fn tree(dims: [usize; 3]) -> Mask {
        let seg = |p: DVec3, a: DVec3, b: DVec3| {
            let t = ((p - a).dot(b - a) / (b - a).length_squared()).clamp(0.0, 1.0);
            p.distance(a + (b - a) * t)
        };
        let s = dims[0] as f64 / 128.0;
        shape(dims, [1.0; 3], |p| {
            let c = DVec3::new(64.0, 64.0, 32.0) * s;
            seg(p, DVec3::new(4.0, 64.0, 32.0) * s, DVec3::new(124.0, 64.0, 32.0) * s) <= 5.0 * s
                || seg(p, c, DVec3::new(100.0, 120.0, 50.0) * s) <= 3.5 * s
                || seg(p, c, DVec3::new(30.0, 8.0, 12.0) * s) <= 2.5 * s
        })
    }

    #[test]
    #[cfg_attr(debug_assertions, ignore = "timing: run with --release")]
    fn level_set_is_fast() {
        let dims = [128, 128, 64];
        let truth = tree(dims);
        let vol = image(&truth, [1.0; 3], 15.0);
        let seed = shape(dims, [1.0; 3], |p| p.distance(DVec3::new(64.0, 64.0, 32.0)) <= 2.0);
        // The whole volume, a fixed number of iterations.
        for iterations in [100, 300] {
            let p = LevelSetParams { range: (50.0, 1000.0), iterations, max_distance_mm: 0.0, ..Default::default() };
            let t0 = Instant::now();
            let out = level_set(&vol, &seed, None, &p, &|_| true).unwrap();
            eprintln!("128×128×64, {iterations} iterations: {:.3} s, Dice {:.3}", t0.elapsed().as_secs_f64(), dice(&out, &truth));
        }
        let p = LevelSetParams { range: (50.0, 1000.0), iterations: 1000, max_distance_mm: 0.0, ..Default::default() };
        let steps = std::cell::Cell::new(0);
        let t0 = Instant::now();
        let out = level_set(&vol, &seed, None, &p, &|_| {
            steps.set(steps.get() + 1);
            true
        })
        .unwrap();
        let secs = t0.elapsed().as_secs_f64();
        let d = dice(&out, &truth);
        eprintln!("128×128×64, until it stops: {secs:.3} s, {} iterations, Dice {d:.3}", steps.get() - 2);
        assert!(d > 0.95, "Dice {d}");
        // Generous: other tests and programs share the processor.
        assert!(secs < 10.0, "{secs} s");
        // From a rough segment (the tree shrunk by 1.5 mm) with a limit of 5 mm.
        let mut rough = truth.clone();
        crate::ops::shrink(&mut rough, 1.5, [1.0; 3]);
        let p = LevelSetParams { range: (50.0, 1000.0), max_distance_mm: 5.0, ..Default::default() };
        let t0 = Instant::now();
        let out = level_set(&vol, &rough, None, &p, &|_| true).unwrap();
        eprintln!("128×128×64 from a rough segment: {:.3} s, Dice {:.3}", t0.elapsed().as_secs_f64(), dice(&out, &truth));
        // Guided along the trunk and one branch, in a 256×256×128 volume.
        let dims = [256, 256, 128];
        let truth = tree(dims);
        let vol = image(&truth, [1.0; 3], 15.0);
        let points = [[16.0, 128.0, 64.0], [80.0, 128.0, 64.0], [164.0, 184.0, 82.0], [196.0, 234.0, 98.0]];
        let t0 = Instant::now();
        let out = guided_vessel(&vol, &points, (50.0, 1000.0), 4.0, &LevelSetParams::default(), &|_| true).unwrap();
        let mut hit = out.clone();
        hit.intersect(&truth);
        eprintln!("guided, 256×256×128: {:.3} s, {} voxels, {:.3} of them in the tree", t0.elapsed().as_secs_f64(), out.count(), hit.count() as f64 / out.count() as f64);
        assert!(hit.count() as f64 > 0.95 * out.count() as f64);
    }
}
