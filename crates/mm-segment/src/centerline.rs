//! Centerlines and cross-sections of a segmented vessel.
//!
//! The centerline tree follows TEASAR (Sato et al. 2000): every voxel's distance from the
//! boundary, a shortest-path tree from a root under a step cost that rises steeply toward the
//! wall, then, again and again, the farthest voxel not yet covered traced back along that tree
//! to the centerline found so far, with the voxels within a multiple of the local radius of
//! the new path covered, until every voxel is. Paths shorter than a minimum are dropped, ends
//! that only run from the middle of a vessel out to its tip are cut, and the paths are
//! smoothed to sub-voxel coordinates. Cross-sections sample the mask in planes perpendicular
//! to a branch.

use crate::mask::{Connectivity, Mask};
use crate::ops::distance_sq;
use anyhow::{bail, Result};
use glam::{DMat4, DVec2, DVec3};
use mm_volume::geometry::Geometry;
use rayon::prelude::*;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CenterlinePoint {
    /// Continuous voxel coordinates (i, j, k) of the mask's grid.
    pub voxel: [f64; 3],
    /// RAS position, mm.
    pub ras: [f64; 3],
    /// Radius of the largest inscribed sphere, mm.
    pub radius_mm: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Branch {
    /// From the root (branch 0) or the junction with the parent, to the tip.
    pub points: Vec<CenterlinePoint>,
    /// The branch it leaves (its first point lies on it).
    pub parent: Option<usize>,
    pub length_mm: f64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Centerlines {
    /// Parents come before their children; branch 0 starts at the root.
    pub branches: Vec<Branch>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CenterlineParams {
    /// Branches shorter than this (mm) are dropped (default 5).
    pub min_branch_mm: f64,
    /// Voxels within this × the local radius of a path are covered by it (default 1.75, at
    /// least 1).
    pub cover: f64,
    /// Points the centerline should start from / reach (voxel coordinates), e.g. the vessel's
    /// ends; empty = found automatically. The first is the root; each is moved to the nearest
    /// voxel of the vessel (the first picks the connected piece, the others snap to it).
    pub ends: Vec<[f64; 3]>,
    /// Smoothing of the paths, mm (Gaussian σ along the path; 0 = none; default 1.5). About
    /// 1.5 voxels or more takes the voxel staircase out of the paths and their lengths.
    pub smoothing_mm: f64,
}

impl Default for CenterlineParams {
    fn default() -> Self {
        Self { min_branch_mm: 5.0, cover: 1.75, ends: Vec::new(), smoothing_mm: 1.5 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CrossSection {
    /// Along the branch from its first point, mm.
    pub distance_mm: f64,
    pub ras: [f64; 3],
    /// Unit tangent of the branch (RAS), the plane's normal.
    pub normal: [f64; 3],
    pub area_mm2: f64,
    /// Diameter of a circle of the same area.
    pub equivalent_diameter_mm: f64,
    /// Smallest and largest caliper width of the section.
    pub min_diameter_mm: f64,
    pub max_diameter_mm: f64,
    /// Inscribed-sphere radius of the centerline here (interpolated between its points).
    pub radius_mm: f64,
}

const NONE: u32 = u32::MAX;

/// Trilinear value at continuous voxel coordinates of a 0 / 1 grid given by `on`.
fn trilinear(p: DVec3, on: impl Fn(i64, i64, i64) -> bool) -> f64 {
    let f = p.floor();
    let t = p - f;
    let (i, j, k) = (f.x as i64, f.y as i64, f.z as i64);
    let mut v = 0.0;
    for c in 0..8 {
        let (a, b, d) = (c & 1, (c >> 1) & 1, (c >> 2) & 1);
        if on(i + a, j + b, k + d) {
            let wx = if a == 1 { t.x } else { 1.0 - t.x };
            let wy = if b == 1 { t.y } else { 1.0 - t.y };
            let wz = if d == 1 { t.z } else { 1.0 - t.z };
            v += wx * wy * wz;
        }
    }
    v
}

fn mask_on(mask: &Mask) -> impl Fn(i64, i64, i64) -> bool + '_ {
    move |i, j, k| {
        let [nx, ny, nz] = mask.dims;
        i >= 0 && j >= 0 && k >= 0 && (i as usize) < nx && (j as usize) < ny && (k as usize) < nz && mask.data[mask.index(i as usize, j as usize, k as usize)] != 0
    }
}

/// One connected piece of the mask on its bounding box plus a voxel all round, as a graph of
/// its voxels ("nodes").
struct Region {
    /// Full-grid voxel index of the box's voxel (0, 0, 0).
    origin: [i64; 3],
    dims: [usize; 3],
    spacing: [f64; 3],
    /// Node of each box voxel, `NONE` outside the piece.
    node: Vec<u32>,
    /// Box index of each node.
    at: Vec<usize>,
    /// Distance of each node from the nearest voxel outside the piece, mm.
    dbf: Vec<f64>,
}

impl Region {
    fn xyz(&self, c: usize) -> [usize; 3] {
        [c % self.dims[0], (c / self.dims[0]) % self.dims[1], c / (self.dims[0] * self.dims[1])]
    }

    fn scale(&self) -> DVec3 {
        DVec3::from(self.spacing)
    }

    /// Box coordinates of node `n` in mm (for distances).
    fn mm(&self, n: u32) -> DVec3 {
        let [x, y, z] = self.xyz(self.at[n as usize]);
        DVec3::new(x as f64, y as f64, z as f64) * self.scale()
    }

    /// Full-grid voxel coordinates of node `n`.
    fn voxel(&self, n: u32) -> DVec3 {
        let [x, y, z] = self.xyz(self.at[n as usize]);
        DVec3::new((x as i64 + self.origin[0]) as f64, (y as i64 + self.origin[1]) as f64, (z as i64 + self.origin[2]) as f64)
    }

    fn to_box(&self, p: DVec3) -> DVec3 {
        p - DVec3::new(self.origin[0] as f64, self.origin[1] as f64, self.origin[2] as f64)
    }

    /// Whether box voxel (x, y, z) is in the piece (false outside the box).
    fn is_in(&self, x: i64, y: i64, z: i64) -> bool {
        let [nx, ny, nz] = self.dims;
        x >= 0 && y >= 0 && z >= 0 && (x as usize) < nx && (y as usize) < ny && (z as usize) < nz && self.node[(z as usize * ny + y as usize) * nx + x as usize] != NONE
    }

    /// Trilinear value of the piece (0 / 1) at full-grid voxel coordinates.
    fn sample(&self, p: DVec3) -> f64 {
        trilinear(self.to_box(p), |x, y, z| self.is_in(x, y, z))
    }

    /// The node nearest to full-grid voxel coordinates `p` (in mm).
    fn nearest(&self, p: DVec3) -> u32 {
        let q = self.to_box(p) * self.scale();
        (0..self.at.len() as u32).into_par_iter().map(|n| ((self.mm(n) - q).length_squared(), n)).min_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1))).map_or(0, |(_, n)| n)
    }

    /// Radius (mm) of the largest sphere about `p` (full-grid voxel coordinates) inside the
    /// piece: the distance to the nearest voxel centre outside it. On round vessels this is
    /// within a few hundredths of a voxel of the true radius on average (the half-way surface
    /// between inside and outside voxels would give about 0.4 voxel less, as it dips between
    /// the voxels of the staircase).
    fn radius(&self, p: DVec3) -> f64 {
        let s = self.scale();
        let c = self.to_box(p);
        let q = [0, 1, 2].map(|a| (c[a].round() as i64).clamp(0, self.dims[a] as i64 - 1));
        let qv = DVec3::new(q[0] as f64, q[1] as f64, q[2] as f64);
        let depth = if self.is_in(q[0], q[1], q[2]) { self.dbf[self.node[(q[2] as usize * self.dims[1] + q[1] as usize) * self.dims[0] + q[0] as usize] as usize] } else { 0.0 };
        // The nearest outside voxel of `q` is `depth` away.
        let bound = depth + ((c - qv) * s).length();
        let h = [0, 1, 2].map(|a| (bound / self.spacing[a]).ceil() as i64 + 1);
        let mut best = bound * bound;
        for z in (q[2] - h[2]).max(0)..=(q[2] + h[2]).min(self.dims[2] as i64 - 1) {
            for y in (q[1] - h[1]).max(0)..=(q[1] + h[1]).min(self.dims[1] as i64 - 1) {
                for x in (q[0] - h[0]).max(0)..=(q[0] + h[0]).min(self.dims[0] as i64 - 1) {
                    if !self.is_in(x, y, z) {
                        best = best.min(((DVec3::new(x as f64, y as f64, z as f64) - c) * s).length_squared());
                    }
                }
            }
        }
        best.sqrt()
    }
}

/// Box-index offsets and lengths (mm) of the 26 neighbours of a voxel in a box of `dims`.
fn steps(dims: [usize; 3], spacing: [f64; 3]) -> Vec<(isize, f64)> {
    let [nx, ny, _] = dims;
    Connectivity::TwentySix
        .offsets()
        .iter()
        .map(|o| (((o[2] * ny as i64 + o[1]) * nx as i64 + o[0]) as isize, (DVec3::new(o[0] as f64, o[1] as f64, o[2] as f64) * DVec3::from(spacing)).length()))
        .collect()
}

/// 26-connected pieces of the set voxels of a box whose edge voxels are all clear: a piece
/// number per voxel (0 = clear, pieces from 1) and the voxel count of each (index 0 unused).
fn pieces(dims: [usize; 3], data: &[u8]) -> (Vec<u32>, Vec<usize>) {
    let steps = steps(dims, [1.0; 3]);
    let mut label = vec![0u32; data.len()];
    let mut sizes = vec![0usize];
    let mut stack = Vec::new();
    for start in 0..data.len() {
        if data[start] == 0 || label[start] != 0 {
            continue;
        }
        let id = sizes.len() as u32;
        let mut size = 0;
        label[start] = id;
        stack.push(start);
        while let Some(c) = stack.pop() {
            size += 1;
            for &(d, _) in &steps {
                let m = (c as isize + d) as usize;
                if data[m] != 0 && label[m] == 0 {
                    label[m] = id;
                    stack.push(m);
                }
            }
        }
        sizes.push(size);
    }
    (label, sizes)
}

/// The 26-connected piece of `mask` containing the voxel nearest `root` (the largest piece
/// without one), with each voxel's distance from the boundary.
fn region(mask: &Mask, spacing: [f64; 3], root: Option<DVec3>) -> Result<Region> {
    let Some((lo, hi)) = mask.bounds() else { bail!("the segment is empty") };
    let origin = [0, 1, 2].map(|a| lo[a] as i64 - 1);
    let dims = [0, 1, 2].map(|a| hi[a] - lo[a] + 3);
    let n = dims[0] * dims[1] * dims[2];
    if n > 1_500_000_000 {
        bail!("the segment spans too many voxels ({} million)", n / 1_000_000);
    }
    // The bounding box, one clear voxel wider all round.
    let mut crop = vec![0u8; n];
    crop.par_chunks_mut(dims[0] * dims[1]).enumerate().for_each(|(z, p)| {
        if z == 0 || z + 1 == dims[2] {
            return;
        }
        for y in 1..dims[1] - 1 {
            let from = mask.index(lo[0], lo[1] + y - 1, lo[2] + z - 1);
            for (x, &v) in mask.data[from..from + dims[0] - 2].iter().enumerate() {
                p[y * dims[0] + x + 1] = (v != 0) as u8;
            }
        }
    });
    let (label, sizes) = pieces(dims, &crop);
    let pick = match root {
        Some(e) => {
            let q = (e - DVec3::new(origin[0] as f64, origin[1] as f64, origin[2] as f64)) * DVec3::from(spacing);
            let at = |c: usize| DVec3::new((c % dims[0]) as f64, ((c / dims[0]) % dims[1]) as f64, (c / (dims[0] * dims[1])) as f64) * DVec3::from(spacing);
            let best = (0..n).into_par_iter().filter(|&c| label[c] != 0).map(|c| ((at(c) - q).length_squared(), c)).min_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            label[best.map_or(0, |(_, c)| c)]
        }
        None => (1..sizes.len()).max_by_key(|&c| sizes[c]).unwrap_or(0) as u32,
    };
    let at: Vec<usize> = (0..n).into_par_iter().filter(|&c| label[c] == pick).collect();
    if at.len() >= NONE as usize {
        bail!("the segment has too many voxels");
    }
    let mut node = vec![NONE; n];
    for (i, &c) in at.iter().enumerate() {
        node[c] = i as u32;
    }
    let outside: Vec<u8> = label.par_iter().map(|&l| (l != pick) as u8).collect();
    drop(label);
    let d2 = distance_sq(dims, &outside, spacing);
    let dbf = at.par_iter().map(|&c| d2[c].sqrt()).collect();
    Ok(Region { origin, dims, spacing, node, at, dbf })
}

/// Shortest paths from `start` between 26-neighbouring nodes, a step of `len` mm from `a` to
/// `b` costing `cost(a, b, len)`: the previous node on the path to each node and the path's
/// length in mm. `tick` gets the fraction of nodes done; returning `false` cancels.
fn shortest_paths(r: &Region, start: u32, cost: impl Fn(usize, usize, f64) -> f64, tick: &dyn Fn(f64) -> bool) -> Result<(Vec<u32>, Vec<f64>)> {
    let steps = steps(r.dims, r.spacing);
    let count = r.at.len();
    let mut dist = vec![f64::INFINITY; count];
    let mut prev = vec![NONE; count];
    let mut len = vec![0.0; count];
    let mut done = vec![false; count];
    let mut heap = BinaryHeap::new();
    dist[start as usize] = 0.0;
    heap.push(Reverse((0u64, start)));
    let mut settled = 0usize;
    // Non-negative floats order as their bits.
    while let Some(Reverse((bits, a))) = heap.pop() {
        let a = a as usize;
        if done[a] {
            continue;
        }
        done[a] = true;
        settled += 1;
        if settled.is_multiple_of(65536) && !tick(settled as f64 / count as f64) {
            bail!("cancelled");
        }
        let da = f64::from_bits(bits);
        let c = r.at[a];
        // Nodes are never on the box's edge, so every neighbour is in the box.
        for &(delta, l) in &steps {
            let b = r.node[(c as isize + delta) as usize];
            if b == NONE || done[b as usize] {
                continue;
            }
            let b = b as usize;
            let nd = da + cost(a, b, l);
            if nd < dist[b] {
                dist[b] = nd;
                prev[b] = a as u32;
                len[b] = len[a] + l;
                heap.push(Reverse((nd.to_bits(), b as u32)));
            }
        }
    }
    Ok((prev, len))
}

/// Mark the nodes within `radius` mm of node `n`; returns how many were not marked yet.
fn cover(r: &Region, n: u32, radius: f64, covered: &mut [bool]) -> usize {
    let [x, y, z] = r.xyz(r.at[n as usize]).map(|v| v as i64);
    let [sx, sy, sz] = r.spacing;
    let hz = (radius / sz).floor() as i64;
    let mut fresh = 0;
    // Row by row over the ball's extent.
    for k in (z - hz).max(0)..=(z + hz).min(r.dims[2] as i64 - 1) {
        let left_z = radius * radius - ((k - z) as f64 * sz).powi(2);
        let hy = (left_z.max(0.0).sqrt() / sy).floor() as i64;
        for j in (y - hy).max(0)..=(y + hy).min(r.dims[1] as i64 - 1) {
            let left = left_z - ((j - y) as f64 * sy).powi(2);
            if left < 0.0 {
                continue;
            }
            let hx = (left.sqrt() / sx).floor() as i64;
            let row = (k as usize * r.dims[1] + j as usize) * r.dims[0];
            for i in (x - hx).max(0)..=(x + hx).min(r.dims[0] as i64 - 1) {
                let m = r.node[row + i as usize];
                if m != NONE && !covered[m as usize] {
                    covered[m as usize] = true;
                    fresh += 1;
                }
            }
        }
    }
    fresh
}

/// How many points of `path` to keep so that its tip, where it only runs from the middle of
/// a vessel into its end, is cut: back to the innermost point whose inscribed sphere, grown
/// by √2, still reaches the tip. On a flat-ended tube that is a radius from the end (the
/// sphere touching the end reaches its rim), on a round end about 0.4 radius short of it.
fn trim_tip(r: &Region, path: &[u32]) -> usize {
    let n = path.len();
    let tip = r.mm(path[n - 1]);
    let reach = 4.0 * path.iter().map(|&v| r.dbf[v as usize]).fold(0.0, f64::max);
    let (mut keep, mut arc) = (n, 0.0);
    for j in (0..n - 1).rev() {
        arc += (r.mm(path[j]) - r.mm(path[j + 1])).length();
        if arc > reach {
            break;
        }
        if (r.mm(path[j]) - tip).length() <= std::f64::consts::SQRT_2 * r.dbf[path[j] as usize] {
            keep = j + 1;
        }
    }
    keep
}

fn path_mm(r: &Region, path: &[u32]) -> f64 {
    path.windows(2).map(|w| (r.mm(w[1]) - r.mm(w[0])).length()).sum()
}

/// Gaussian smoothing over arc length (σ = `sigma` mm) with a window that narrows
/// symmetrically toward the ends, which stay where they are; a point that would leave the
/// piece moves only as far as it stays inside.
fn smooth(r: &Region, pts: &mut [DVec3], sigma: f64) {
    let n = pts.len();
    if sigma.is_nan() || sigma <= 0.0 || n < 3 {
        return;
    }
    let s = r.scale();
    let mut arc = vec![0.0; n];
    for i in 1..n {
        arc[i] = arc[i - 1] + ((pts[i] - pts[i - 1]) * s).length();
    }
    let total = arc[n - 1];
    let src = pts.to_vec();
    pts[1..n - 1].par_iter_mut().enumerate().for_each(|(i, q)| {
        let i = i + 1;
        let half = (3.0 * sigma).min(arc[i]).min(total - arc[i]);
        if half <= 0.0 {
            return;
        }
        let mut first = i;
        while first > 0 && arc[i] - arc[first - 1] <= half {
            first -= 1;
        }
        let (mut acc, mut weight) = (DVec3::ZERO, 0.0);
        for j in first..n {
            let d = arc[j] - arc[i];
            if d > half {
                break;
            }
            let w = (-0.5 * d * d / (sigma * sigma)).exp();
            acc += src[j] * w;
            weight += w;
        }
        let shift = acc / weight - src[i];
        let mut t = 1.0;
        while t > 1.0 / 64.0 && r.sample(src[i] + shift * t) < 0.5 {
            t *= 0.5;
        }
        if r.sample(src[i] + shift * t) >= 0.5 {
            *q = src[i] + shift * t;
        }
    });
}

/// A traced path: nodes from its root or junction to its tip.
struct Path {
    nodes: Vec<u32>,
    /// Parent branch and the index of the junction among its nodes.
    parent: Option<(usize, usize)>,
    /// Exact coordinates asked for the first / last point (ends given by the caller).
    start_at: Option<DVec3>,
    tip_at: Option<DVec3>,
}

/// The centerline tree of the vessel in `mask` (the largest 26-connected piece, or the one
/// nearest the first of `p.ends`). `progress` gets the fraction done; returning `false`
/// cancels (the result is then an error). An empty mask is an error; a piece too short for
/// `p.min_branch_mm` gives no branches.
///
/// The root is the first end, or else the voxel farthest (along the vessel) from the deepest
/// one. Paths follow shortest paths from the root where a step of length `l` between voxels
/// at depths `d₁`, `d₂` (distance from the boundary) costs `l · ((D/d₁)⁴ + (D/d₂)⁴) / 2`,
/// `D` the largest depth: steps near the wall cost far more than steps through the middle,
/// in thin vessels as in thick ones. The other ends are reached first (farthest first), then
/// the farthest voxel not yet covered, until every voxel is covered. Automatic ends are cut
/// back to the innermost point whose inscribed sphere, grown by √2, still reaches them: a
/// radius short of a flat vessel end, about 0.4 radius short of a round one. Ends given by
/// the caller are kept exactly (where they lie inside the vessel).
pub fn centerlines(mask: &Mask, geometry: &Geometry, p: &CenterlineParams, progress: &dyn Fn(f32) -> bool) -> Result<Centerlines> {
    let spacing = geometry.spacing;
    if spacing.iter().any(|s| !(s.is_finite() && *s > 0.0)) {
        bail!("invalid voxel spacing {spacing:?}");
    }
    if mask.dims.iter().product::<usize>() != mask.data.len() {
        bail!("the mask's size does not match its dimensions");
    }
    let check = |f: f64| if progress(f as f32) { Ok(()) } else { bail!("cancelled") };
    check(0.0)?;
    let ends: Vec<DVec3> = p.ends.iter().map(|&e| DVec3::from(e)).collect();
    let r = region(mask, spacing, ends.first().copied())?;
    let count = r.at.len();
    check(0.15)?;
    let end_nodes: Vec<u32> = ends.iter().map(|&e| r.nearest(e)).collect();
    let root = match end_nodes.first() {
        Some(&n) => n,
        None => {
            let deepest = (0..count).max_by(|&a, &b| r.dbf[a].total_cmp(&r.dbf[b]).then(b.cmp(&a))).unwrap_or(0) as u32;
            let (_, len) = shortest_paths(&r, deepest, |_, _, l| l, &|f| progress((0.15 + 0.15 * f) as f32))?;
            (0..count).max_by(|&a, &b| len[a].total_cmp(&len[b]).then(b.cmp(&a))).unwrap_or(0) as u32
        }
    };
    check(0.3)?;
    let deepest = r.dbf.iter().copied().fold(0.0, f64::max);
    let w: Vec<f64> = r.dbf.par_iter().map(|&d| (deepest / d).powi(4)).collect();
    let (prev, len) = shortest_paths(&r, root, |a, b, l| 0.5 * l * (w[a] + w[b]), &|f| progress((0.3 + 0.3 * f) as f32))?;
    check(0.6)?;

    // Targets: the given ends (farthest first), then the farthest voxels not yet covered.
    let mut wanted: Vec<(u32, DVec3)> = end_nodes.iter().copied().zip(ends.iter().copied()).skip(1).filter(|&(n, _)| n != root).collect();
    wanted.sort_by(|a, b| len[b.0 as usize].total_cmp(&len[a.0 as usize]).then(a.0.cmp(&b.0)));
    wanted.dedup_by_key(|w| w.0);
    let mut wanted = wanted.into_iter();
    let mut order: Vec<u32> = (0..count as u32).collect();
    order.par_sort_unstable_by(|&a, &b| len[b as usize].total_cmp(&len[a as usize]).then(a.cmp(&b)));
    let mut next = 0;
    let mut covered = vec![false; count];
    let mut n_covered = 0;
    let scale = p.cover.max(1.0);
    // Branch and point index of the voxels of kept paths (cut ends map to the nearest kept point).
    let mut on_tree = vec![(NONE, 0u32); count];
    let mut paths: Vec<Path> = Vec::new();
    for iteration in 0usize.. {
        let (target, given) = match wanted.next() {
            Some((n, e)) => (n, Some(e)),
            None => {
                while next < count && covered[order[next] as usize] {
                    next += 1;
                }
                if next == count {
                    break;
                }
                (order[next], None)
            }
        };
        if iteration % 16 == 15 {
            check(0.6 + 0.35 * n_covered as f64 / count as f64)?;
        }
        // Back along the shortest-path tree to the centerline so far (or to the root).
        let mut raw = vec![target];
        let mut v = target;
        while on_tree[v as usize].0 == NONE && prev[v as usize] != NONE {
            v = prev[v as usize];
            raw.push(v);
        }
        raw.reverse();
        for &v in &raw {
            n_covered += cover(&r, v, scale * r.dbf[v as usize], &mut covered);
        }
        if raw.len() < 2 {
            continue;
        }
        let first = paths.is_empty();
        let mut nodes = raw.clone();
        let parent = if first {
            None
        } else {
            let (b, k) = on_tree[raw[0] as usize];
            nodes[0] = paths[b as usize].nodes[k as usize];
            Some((b as usize, k as usize))
        };
        if given.is_none() {
            nodes.truncate(trim_tip(&r, &nodes));
        }
        let mut cut = 0;
        if first && ends.is_empty() {
            nodes.reverse();
            let keep = trim_tip(&r, &nodes);
            cut = nodes.len() - keep;
            nodes.truncate(keep);
            nodes.reverse();
        }
        if given.is_none() && (nodes.len() < 2 || path_mm(&r, &nodes) < p.min_branch_mm) {
            if first {
                // The longest path is too short: no branches.
                break;
            }
            continue;
        }
        let b = paths.len() as u32;
        let last = nodes.len() - 1;
        for (i, &v) in raw.iter().enumerate() {
            if on_tree[v as usize].0 == NONE {
                on_tree[v as usize] = (b, i.saturating_sub(cut).min(last) as u32);
            }
        }
        let start_at = if first { ends.first().copied() } else { None };
        paths.push(Path { nodes, parent, start_at, tip_at: given });
    }
    check(0.95)?;

    let mut branches: Vec<Branch> = Vec::with_capacity(paths.len());
    for path in &paths {
        let mut pts: Vec<DVec3> = path.nodes.iter().map(|&n| r.voxel(n)).collect();
        if let Some((b, k)) = path.parent {
            pts[0] = DVec3::from(branches[b].points[k].voxel);
        }
        // Given ends inside the vessel are kept exactly.
        if let Some(e) = path.start_at.filter(|&e| r.sample(e) >= 0.5) {
            pts[0] = e;
        }
        if let Some(e) = path.tip_at.filter(|&e| r.sample(e) >= 0.5) {
            *pts.last_mut().unwrap() = e;
        }
        smooth(&r, &mut pts, p.smoothing_mm);
        let mut points: Vec<CenterlinePoint> = pts.par_iter().map(|&v| CenterlinePoint { voxel: v.to_array(), ras: geometry.world(v).to_array(), radius_mm: r.radius(v) }).collect();
        if let Some((b, k)) = path.parent {
            points[0] = branches[b].points[k];
        }
        let length_mm = points.windows(2).map(|w| (DVec3::from(w[1].ras) - DVec3::from(w[0].ras)).length()).sum();
        branches.push(Branch { points, parent: path.parent.map(|(b, _)| b), length_mm });
    }
    check(1.0)?;
    Ok(Centerlines { branches })
}

/// Convex hull of points, counter-clockwise (Andrew's monotone chain).
fn hull(mut pts: Vec<DVec2>) -> Vec<DVec2> {
    pts.sort_by(|a, b| a.x.total_cmp(&b.x).then(a.y.total_cmp(&b.y)));
    pts.dedup();
    if pts.len() < 3 {
        return pts;
    }
    let turn = |o: DVec2, a: DVec2, b: DVec2| (a - o).perp_dot(b - o);
    let mut h: Vec<DVec2> = Vec::with_capacity(2 * pts.len());
    for &q in &pts {
        while h.len() >= 2 && turn(h[h.len() - 2], h[h.len() - 1], q) <= 0.0 {
            h.pop();
        }
        h.push(q);
    }
    let lower = h.len() + 1;
    for &q in pts.iter().rev().skip(1) {
        while h.len() >= lower && turn(h[h.len() - 2], h[h.len() - 1], q) <= 0.0 {
            h.pop();
        }
        h.push(q);
    }
    h.pop();
    h
}

/// Smallest and largest caliper width of a convex polygon over all directions (rotating
/// calipers): the smallest is across one of its edges, the largest is its diameter.
fn widths(h: &[DVec2]) -> (f64, f64) {
    let max = h.iter().enumerate().flat_map(|(i, a)| h[i + 1..].iter().map(move |b| (*a - *b).length())).fold(0.0, f64::max);
    if h.len() < 3 {
        return (0.0, max);
    }
    let min = (0..h.len())
        .map(|i| {
            let (a, b) = (h[i], h[(i + 1) % h.len()]);
            let e = (b - a).normalize_or_zero();
            h.iter().map(|&q| e.perp_dot(q - a).abs()).fold(0.0, f64::max)
        })
        .fold(f64::INFINITY, f64::min);
    (min, max)
}

/// Area (mm²) and smallest / largest width (mm) of the piece of `mask` around `centre` (RAS)
/// in the plane through it perpendicular to `normal`, sampled on square cells `cell` mm wide
/// out to `limit` mm (trilinear, inside at 0.5 and above; 4-connected to the centre's cell, or
/// to the nearest inside cell within `seek` mm when the centre is not inside).
fn section(mask: &Mask, to_voxel: &DMat4, centre: DVec3, normal: DVec3, cell: f64, limit: f64, seek: f64) -> (f64, f64, f64) {
    let (u, v) = normal.any_orthonormal_pair();
    let c = to_voxel.transform_point3(centre);
    let (du, dv) = (to_voxel.transform_vector3(u * cell), to_voxel.transform_vector3(v * cell));
    // Cell (a, b) spans [a, a + 1] × [b, b + 1] cells from the centre and is sampled in its
    // middle, so on a grid-aligned plane no sample falls exactly half-way between voxels.
    let g = (limit / cell).ceil() as i64;
    let size = (2 * g) as usize;
    let on = mask_on(mask);
    let inside = |a: i64, b: i64| trilinear(c + du * (a as f64 + 0.5) + dv * (b as f64 + 0.5), &on) >= 0.5;
    // 0 not seen, 1 in the piece, 2 outside.
    let mut state = vec![0u8; size * size];
    let cell_at = |a: i64, b: i64| (b + g) as usize * size + (a + g) as usize;
    let within = |a: i64, b: i64| a >= -g && a < g && b >= -g && b < g;
    // The cells around the centre, else the nearest inside cell within `seek` (by rings).
    let seed = (0..((seek / cell).ceil() as i64).clamp(1, g)).find_map(|ring| {
        let mut best: Option<(i64, i64, i64)> = None;
        for b in -ring - 1..=ring {
            for a in -ring - 1..=ring {
                let (x, y) = (2 * a + 1, 2 * b + 1);
                if x.abs().max(y.abs()) == 2 * ring + 1 && inside(a, b) && best.is_none_or(|(_, _, d)| x * x + y * y < d) {
                    best = Some((a, b, x * x + y * y));
                }
            }
        }
        best
    });
    let Some((a0, b0, _)) = seed else { return (0.0, 0.0, 0.0) };
    let mut rows = vec![(i64::MAX, i64::MIN); size];
    let mut stack = vec![(a0, b0)];
    state[cell_at(a0, b0)] = 1;
    let mut cells = 0usize;
    while let Some((a, b)) = stack.pop() {
        cells += 1;
        let row = &mut rows[(b + g) as usize];
        *row = (row.0.min(a), row.1.max(a));
        for (x, y) in [(a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1)] {
            if !within(x, y) || state[cell_at(x, y)] != 0 {
                continue;
            }
            if inside(x, y) {
                state[cell_at(x, y)] = 1;
                stack.push((x, y));
            } else {
                state[cell_at(x, y)] = 2;
            }
        }
    }
    // The hull of the cells' squares: the outer corners of each row's end cells.
    let mut corners = Vec::new();
    for (k, &(lo, hi)) in rows.iter().enumerate() {
        if lo > hi {
            continue;
        }
        let b = k as f64 - g as f64;
        for a in [lo as f64, hi as f64 + 1.0] {
            corners.push(DVec2::new(a, b) * cell);
            corners.push(DVec2::new(a, b + 1.0) * cell);
        }
    }
    let (min, max) = widths(&hull(corners));
    (cells as f64 * cell * cell, min, max)
}

/// Cross-sections of the vessel in `mask` every `step_mm` along `branch` (from its first
/// point; the smallest voxel size if `step_mm` is not positive). Each is the plane through
/// the point perpendicular to the branch's tangent (its chord over about a radius either
/// side), with the mask sampled trilinearly (inside at 0.5) on square cells a quarter of the
/// smallest voxel size wide. Only the piece connected to the centerline counts, so other
/// vessels crossing the plane do not; it is cut off at 6 radii (at least the radius + 5 mm)
/// from the centre, where a plane runs along a vessel rather than across it.
pub fn cross_sections(mask: &Mask, geometry: &Geometry, branch: &Branch, step_mm: f64) -> Vec<CrossSection> {
    let pts: Vec<DVec3> = branch.points.iter().map(|p| DVec3::from(p.ras)).collect();
    if pts.len() < 2 || mask.dims.iter().product::<usize>() != mask.data.len() {
        return Vec::new();
    }
    let mut arc = vec![0.0; pts.len()];
    for i in 1..pts.len() {
        arc[i] = arc[i - 1] + (pts[i] - pts[i - 1]).length();
    }
    let total = arc[pts.len() - 1];
    let smallest = geometry.spacing.iter().copied().fold(f64::INFINITY, f64::min);
    let largest = geometry.spacing.iter().copied().fold(0.0, f64::max);
    let step = if step_mm > 0.0 { step_mm } else { smallest };
    let cell = 0.25 * smallest;
    let to_voxel = geometry.matrix().inverse();
    // Position and radius at arc length `s`.
    let at = |s: f64| -> (DVec3, f64) {
        let s = s.clamp(0.0, total);
        let i = arc.partition_point(|&a| a <= s).clamp(1, pts.len() - 1);
        let span = arc[i] - arc[i - 1];
        let t = if span > 0.0 { (s - arc[i - 1]) / span } else { 0.0 };
        (pts[i - 1].lerp(pts[i], t), branch.points[i - 1].radius_mm * (1.0 - t) + branch.points[i].radius_mm * t)
    };
    let count = (total / step + 1e-9).floor() as usize + 1;
    (0..count)
        .into_par_iter()
        .map(|k| {
            let s = k as f64 * step;
            let (centre, radius) = at(s);
            // Tangent over about a radius either side.
            let h = radius.max(2.0 * smallest);
            let fallback = (pts[pts.len() - 1] - pts[0]).normalize_or(DVec3::Z);
            let normal = (at(s + h).0 - at(s - h).0).normalize_or(fallback);
            let limit = (6.0 * radius).max(radius + 5.0);
            let (area, min, max) = section(mask, &to_voxel, centre, normal, cell, limit, largest);
            CrossSection {
                distance_mm: s,
                ras: centre.to_array(),
                normal: normal.to_array(),
                area_mm2: area,
                equivalent_diameter_mm: 2.0 * (area / std::f64::consts::PI).sqrt(),
                min_diameter_mm: min,
                max_diameter_mm: max,
                radius_mm: radius,
            }
        })
        .collect()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) }
}

/// The narrowest section (by equivalent diameter) and its narrowing in percent,
/// `100 · (1 − d / d_ref)`. The reference `d_ref` is the median equivalent diameter of the
/// sections outside the narrowing, the narrowing being the run of consecutive sections around
/// the narrowest one that are below 90% of the reference (starting from the median of all
/// sections and repeated until it settles). Empty sections are ignored. `None` with fewer
/// than 3 sections, or when the narrowing takes them all.
pub fn stenosis(sections: &[CrossSection]) -> Option<(usize, f64)> {
    let d: Vec<(usize, f64)> = sections.iter().enumerate().map(|(i, s)| (i, s.equivalent_diameter_mm)).filter(|&(_, d)| d > 0.0).collect();
    if d.len() < 3 {
        return None;
    }
    let narrowest = (0..d.len()).min_by(|&a, &b| d[a].1.total_cmp(&d[b].1))?;
    let mut reference = median(d.iter().map(|x| x.1).collect());
    for _ in 0..10 {
        let (mut lo, mut hi) = (narrowest, narrowest);
        while lo > 0 && d[lo - 1].1 < 0.9 * reference {
            lo -= 1;
        }
        while hi + 1 < d.len() && d[hi + 1].1 < 0.9 * reference {
            hi += 1;
        }
        let outside: Vec<f64> = d.iter().enumerate().filter(|&(k, _)| k < lo || k > hi).map(|(_, x)| x.1).collect();
        if outside.is_empty() {
            return None;
        }
        let next = median(outside);
        let settled = (next - reference).abs() <= 1e-9 * reference;
        reference = next;
        if settled {
            break;
        }
    }
    Some((d[narrowest].0, (100.0 * (1.0 - d[narrowest].1 / reference)).max(0.0)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;
    use std::time::Instant;

    fn grid(dims: [usize; 3], spacing: [f64; 3]) -> Geometry {
        Geometry::from_axes([dims[0], dims[1], dims[2], 1], spacing, [DVec3::X, DVec3::Y, DVec3::Z], DVec3::ZERO)
    }

    /// The voxels whose centres (in mm) pass `inside`.
    fn shape(dims: [usize; 3], spacing: [f64; 3], inside: impl Fn(DVec3) -> bool + Sync) -> Mask {
        let mut m = Mask::new(dims);
        let s = DVec3::from(spacing);
        m.data.par_iter_mut().enumerate().for_each(|(n, v)| {
            let p = DVec3::new((n % dims[0]) as f64, ((n / dims[0]) % dims[1]) as f64, (n / (dims[0] * dims[1])) as f64) * s;
            *v = inside(p) as u8;
        });
        m
    }

    /// Distance from `p` to the segment `a`–`b`.
    fn to_segment(p: DVec3, a: DVec3, b: DVec3) -> f64 {
        let t = ((p - a).dot(b - a) / (b - a).length_squared()).clamp(0.0, 1.0);
        (p - a.lerp(b, t)).length()
    }

    fn to_polyline(p: DVec3, line: &[DVec3]) -> f64 {
        line.windows(2).map(|w| to_segment(p, w[0], w[1])).fold(f64::INFINITY, f64::min)
    }

    fn run(m: &Mask, g: &Geometry, p: &CenterlineParams) -> Centerlines {
        centerlines(m, g, p, &|_| true).unwrap()
    }

    /// Sections at least a radius away from both ends of the branch.
    fn middle(sections: &[CrossSection], length: f64, r: f64) -> Vec<CrossSection> {
        sections.iter().filter(|s| s.distance_mm > r && s.distance_mm < length - r).copied().collect()
    }

    #[test]
    fn straight_cylinder() {
        let dims = [80, 25, 25];
        let m = shape(dims, [1.0; 3], |p| p.x >= 10.0 && p.x <= 69.0 && (p.y - 12.0).powi(2) + (p.z - 12.0).powi(2) <= 25.0);
        let g = grid(dims, [1.0; 3]);
        let c = run(&m, &g, &CenterlineParams::default());
        assert_eq!(c.branches.len(), 1);
        let b = &c.branches[0];
        assert_eq!(b.parent, None);
        for q in &b.points {
            let off = DVec3::new(0.0, q.voxel[1] - 12.0, q.voxel[2] - 12.0).length();
            assert!(off <= 0.5, "{:?} is {off} off the axis", q.voxel);
        }
        // 60 voxels long, less about a radius at each end.
        assert!((b.length_mm - 50.0).abs() < 4.0, "length {}", b.length_mm);
        let inner: Vec<&CenterlinePoint> = b.points.iter().filter(|q| q.voxel[0] > 20.0 && q.voxel[0] < 60.0).collect();
        let mean_r = inner.iter().map(|q| q.radius_mm).sum::<f64>() / inner.len() as f64;
        assert!((mean_r - 5.0).abs() < 0.3, "radius {mean_r}");
        assert_eq!(b.points[0].ras, b.points[0].voxel);
        let s = cross_sections(&m, &g, b, 2.0);
        assert!(s.len() >= 20);
        for x in middle(&s, b.length_mm, 5.0) {
            assert!((x.area_mm2 / (PI * 25.0) - 1.0).abs() < 0.05, "area {}", x.area_mm2);
            // The voxelised disk spans 11 voxels along the grid axes (its 0.5 level is half a
            // voxel past the last voxel centre), a little less diagonally.
            assert!((x.min_diameter_mm - 10.0).abs() < 1.2 && (x.max_diameter_mm - 10.0).abs() < 1.2, "{} {}", x.min_diameter_mm, x.max_diameter_mm);
            assert!(x.min_diameter_mm <= x.equivalent_diameter_mm + 0.5 && x.max_diameter_mm >= x.equivalent_diameter_mm);
            assert!(DVec3::from(x.normal).x.abs() > 0.999);
        }
    }

    #[test]
    fn finds_a_narrowing() {
        let dims = [100, 29, 29];
        // Radius 6, narrowing to 3 at x = 50 over 24 voxels.
        let radius = |x: f64| if (x - 50.0).abs() < 12.0 { 6.0 - 1.5 * (1.0 + (PI * (x - 50.0) / 12.0).cos()) } else { 6.0 };
        let m = shape(dims, [1.0; 3], |p| p.x >= 8.0 && p.x <= 91.0 && (p.y - 14.0).powi(2) + (p.z - 14.0).powi(2) <= radius(p.x).powi(2));
        let g = grid(dims, [1.0; 3]);
        let c = run(&m, &g, &CenterlineParams::default());
        assert_eq!(c.branches.len(), 1);
        let s = cross_sections(&m, &g, &c.branches[0], 1.0);
        let (i, pct) = stenosis(&s).unwrap();
        assert!((s[i].ras[0] - 50.0).abs() <= 2.0, "at {:?}", s[i].ras);
        assert!((pct - 50.0).abs() < 8.0, "{pct}%");
        // A plain cylinder has none to speak of.
        let plain = shape(dims, [1.0; 3], |p| p.x >= 8.0 && p.x <= 91.0 && (p.y - 14.0).powi(2) + (p.z - 14.0).powi(2) <= 36.0);
        let c = run(&plain, &g, &CenterlineParams::default());
        let s = cross_sections(&plain, &g, &c.branches[0], 1.0);
        assert!(stenosis(&s).unwrap().1 < 8.0);
        assert_eq!(stenosis(&s[..2]), None);
    }

    fn y_shape() -> (Mask, Geometry, [DVec3; 4]) {
        let dims = [72, 21, 84];
        let (base, fork, left, right) = (DVec3::new(36.0, 10.0, 6.0), DVec3::new(36.0, 10.0, 40.0), DVec3::new(14.0, 10.0, 76.0), DVec3::new(58.0, 10.0, 76.0));
        let m = shape(dims, [1.0; 3], |p| to_segment(p, base, fork).min(to_segment(p, fork, left)).min(to_segment(p, fork, right)) <= 4.0);
        (m, grid(dims, [1.0; 3]), [base, fork, left, right])
    }

    #[test]
    fn y_branches() {
        let (m, g, [base, fork, left, right]) = y_shape();
        let c = run(&m, &g, &CenterlineParams::default());
        assert!((2..=3).contains(&c.branches.len()), "{} branches", c.branches.len());
        assert_eq!(c.branches[0].parent, None);
        for (i, b) in c.branches.iter().enumerate().skip(1) {
            let p = b.parent.unwrap();
            assert!(p < i);
            // Starts on its parent.
            assert!(c.branches[p].points.iter().any(|q| q.voxel == b.points[0].voxel));
        }
        // On the axes, but for the fork, where the middle of the union rises between the arms.
        for b in &c.branches {
            for q in &b.points {
                let v = DVec3::from(q.voxel);
                let d = to_polyline(v, &[base, fork, left]).min(to_segment(v, fork, right));
                assert!(d < if (v - fork).length() < 8.0 { 2.5 } else { 0.75 }, "{v} is {d} off the axes");
                // Smoothing kept it inside.
                assert!(trilinear(v, mask_on(&m)) >= 0.5, "{v} left the vessel");
            }
        }
        // With the ends given, the paths start and end there.
        let p = CenterlineParams { ends: vec![base.to_array(), left.to_array(), right.to_array()], ..Default::default() };
        let c = run(&m, &g, &p);
        assert_eq!(c.branches.len(), 2);
        assert_eq!((c.branches[0].parent, c.branches[1].parent), (None, Some(0)));
        assert_eq!(c.branches[0].points[0].voxel, base.to_array());
        let tips: Vec<[f64; 3]> = c.branches.iter().map(|b| b.points.last().unwrap().voxel).collect();
        assert!(tips.contains(&left.to_array()) && tips.contains(&right.to_array()), "{tips:?}");
        // The side branch leaves near the fork.
        assert!((DVec3::from(c.branches[1].points[0].voxel) - fork).length() < 5.0);
        // Only the given ends: none dropped, none added.
        let p = CenterlineParams { min_branch_mm: f64::INFINITY, ..p };
        assert_eq!(run(&m, &g, &p).branches.len(), 2);
    }

    #[test]
    fn anisotropic_cylinder() {
        let spacing = [0.5, 0.5, 1.5];
        let dims = [104, 34, 30];
        // 50 mm long, radius 5 mm, along a diagonal of the x–z plane (mm).
        let (a, b) = (DVec3::new(6.0, 8.0, 6.0), DVec3::new(46.0, 8.0, 36.0));
        let axis = (b - a).normalize();
        let m = shape(dims, spacing, |p| {
            let t = (p - a).dot(axis);
            (0.0..=50.0).contains(&t) && (p - a - axis * t).length() <= 5.0
        });
        let g = grid(dims, spacing);
        let c = run(&m, &g, &CenterlineParams::default());
        assert_eq!(c.branches.len(), 1);
        let br = &c.branches[0];
        for q in &br.points {
            let d = to_segment(DVec3::from(q.ras), a, b);
            assert!(d < 0.5, "{:?} is {d} mm off the axis", q.ras);
        }
        assert!((br.length_mm - 40.0).abs() < 4.0, "length {}", br.length_mm);
        let inner: Vec<f64> = br.points.iter().filter(|q| (DVec3::from(q.ras) - a).dot(axis) > 15.0 && (DVec3::from(q.ras) - a).dot(axis) < 35.0).map(|q| q.radius_mm).collect();
        let mean_r = inner.iter().sum::<f64>() / inner.len() as f64;
        assert!((mean_r - 5.0).abs() < 0.5, "radius {mean_r}");
        let s = cross_sections(&m, &g, br, 2.0);
        for x in middle(&s, br.length_mm, 5.0) {
            assert!((x.area_mm2 / (PI * 25.0) - 1.0).abs() < 0.06, "area {}", x.area_mm2);
            assert!((x.min_diameter_mm - 10.0).abs() < 1.2 && (x.max_diameter_mm - 10.0).abs() < 1.2, "{} {}", x.min_diameter_mm, x.max_diameter_mm);
            assert!(DVec3::from(x.normal).dot(axis).abs() > 0.995);
        }
        // The same grid turned and moved in the world: the same lengths and sections.
        let turned = Geometry::from_axes([dims[0], dims[1], dims[2], 1], spacing, [DVec3::new(0.8, 0.6, 0.0), DVec3::new(-0.6, 0.8, 0.0), DVec3::Z], DVec3::new(-20.0, 7.0, 3.0));
        let c2 = run(&m, &turned, &CenterlineParams::default());
        let br2 = &c2.branches[0];
        assert!((br2.length_mm - br.length_mm).abs() < 1e-9);
        assert_eq!(br2.points[3].voxel, br.points[3].voxel);
        let s2 = cross_sections(&m, &turned, br2, 2.0);
        assert_eq!(s2.len(), s.len());
        for (x, y) in s.iter().zip(&s2) {
            assert!((x.area_mm2 / y.area_mm2 - 1.0).abs() < 0.02, "{} vs {}", x.area_mm2, y.area_mm2);
            assert!((x.max_diameter_mm - y.max_diameter_mm).abs() < 0.3);
        }
    }

    #[test]
    fn a_neighbouring_tube_is_kept_apart() {
        let dims = [70, 32, 25];
        // Radius 5 at y = 12, radius 4 at y = 23: one empty voxel between them.
        let m = shape(dims, [1.0; 3], |p| p.x >= 8.0 && p.x <= 61.0 && ((p.y - 12.0).powi(2) + (p.z - 12.0).powi(2) <= 25.0 || (p.y - 23.0).powi(2) + (p.z - 12.0).powi(2) <= 16.0));
        assert!(m.get(30, 17, 12) && !m.get(30, 18, 12) && m.get(30, 19, 12));
        let g = grid(dims, [1.0; 3]);
        let c = run(&m, &g, &CenterlineParams::default());
        assert_eq!(c.branches.len(), 1);
        let b = &c.branches[0];
        assert!(b.points.iter().all(|q| (q.voxel[1] - 12.0).abs() < 0.6));
        for x in middle(&cross_sections(&m, &g, b, 2.0), b.length_mm, 5.0) {
            assert!((x.area_mm2 / (PI * 25.0) - 1.0).abs() < 0.05, "area {}", x.area_mm2);
            assert!(x.max_diameter_mm < 11.5, "{}", x.max_diameter_mm);
        }
        // An end in the other tube picks it.
        let p = CenterlineParams { ends: vec![[9.0, 23.0, 12.0]], ..Default::default() };
        let c = run(&m, &g, &p);
        assert_eq!(c.branches.len(), 1);
        assert!(c.branches[0].points.iter().all(|q| (q.voxel[1] - 23.0).abs() < 0.6));
        assert!((c.branches[0].points[2].radius_mm - 4.0).abs() < 0.6);
    }

    #[test]
    fn tortuous_tube_keeps_to_the_middle() {
        // A helix of radius 12 and pitch 30, two turns, tube radius 3.5.
        let axis: Vec<DVec3> = (0..=400).map(|k| { let t = k as f64 / 400.0 * 4.0 * PI; DVec3::new(20.0 + 12.0 * t.cos(), 20.0 + 12.0 * t.sin(), 8.0 + 30.0 * t / (2.0 * PI)) }).collect();
        let dims = [41, 41, 77];
        let m = shape(dims, [1.0; 3], |p| to_polyline(p, &axis) <= 3.5);
        let g = grid(dims, [1.0; 3]);
        let c = run(&m, &g, &CenterlineParams::default());
        assert_eq!(c.branches.len(), 1);
        let b = &c.branches[0];
        let worst = b.points.iter().map(|q| to_polyline(DVec3::from(q.voxel), &axis)).fold(0.0, f64::max);
        assert!(worst < 0.5, "{worst} off the axis");
        // The ends are round, so the branch reaches about to the ends of the axis; the voxel
        // staircase is smoothed out of the length.
        let true_len: f64 = axis.windows(2).map(|w| (w[1] - w[0]).length()).sum();
        assert!((b.length_mm - true_len).abs() < 0.02 * true_len, "{} vs {true_len}", b.length_mm);
        // Unsmoothed, the staircase makes it longer.
        let raw = run(&m, &g, &CenterlineParams { smoothing_mm: 0.0, ..Default::default() });
        assert!(raw.branches[0].length_mm > 1.05 * true_len);
    }

    #[test]
    fn empty_and_cancelled() {
        let g = grid([10, 10, 10], [1.0; 3]);
        let empty = Mask::new([10, 10, 10]);
        assert!(centerlines(&empty, &g, &CenterlineParams::default(), &|_| true).is_err());
        // A lone voxel or a small blob has no branch long enough.
        let mut blob = Mask::new([10, 10, 10]);
        blob.set(5, 5, 5, true);
        assert!(run(&blob, &g, &CenterlineParams::default()).branches.is_empty());
        let ball = shape([10, 10, 10], [1.0; 3], |p| (p - DVec3::splat(5.0)).length() <= 2.0);
        assert!(run(&ball, &g, &CenterlineParams::default()).branches.is_empty());
        assert!(cross_sections(&empty, &g, &Branch { points: Vec::new(), parent: None, length_mm: 0.0 }, 1.0).is_empty());
        let (m, g, _) = y_shape();
        let err = centerlines(&m, &g, &CenterlineParams::default(), &|_| false).unwrap_err();
        assert!(err.to_string().contains("cancelled"));
        // Cancelled part-way.
        let calls = std::sync::atomic::AtomicUsize::new(0);
        assert!(centerlines(&m, &g, &CenterlineParams::default(), &|_| calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 3).is_err());
    }

    /// A tube of radius `r` around `axis` in a grid of `n`³ voxels, each voxel's radius
    /// jittered by up to ±`jitter` (a rough wall), stamped segment by segment.
    fn stamped_tube(n: usize, axis: &[DVec3], r: f64, jitter: f64) -> Mask {
        let mut m = Mask::new([n; 3]);
        for w in axis.windows(2) {
            let (lo, hi) = (w[0].min(w[1]) - r - 2.0, w[0].max(w[1]) + r + 2.0);
            for k in lo.z as usize..=hi.z as usize {
                for j in lo.y as usize..=hi.y as usize {
                    for i in lo.x as usize..=hi.x as usize {
                        let hash = ((m.index(i, j, k) as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 11) as f64 / (1u64 << 53) as f64;
                        if to_segment(DVec3::new(i as f64, j as f64, k as f64), w[0], w[1]) <= r + jitter * (2.0 * hash - 1.0) {
                            m.set(i, j, k, true);
                        }
                    }
                }
            }
        }
        m
    }

    /// Timings on realistic cases: `cargo test -p mm-segment --release -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn timing_tortuous_tubes_in_a_big_grid() {
        let n = 256;
        let g = grid([n; 3], [1.0; 3]);
        // Spanning 200 voxels along x, winding in y and z.
        let wavy = |x0: f64, len: f64, ay: f64, ly: f64, az: f64, lz: f64| -> Vec<DVec3> {
            (0..=800).map(|k| { let x = x0 + len * k as f64 / 800.0; DVec3::new(x, 128.0 + ay * (2.0 * PI * x / ly).sin(), 128.0 + az * (2.0 * PI * x / lz).sin()) }).collect()
        };
        let cases = [("radius 4", wavy(28.0, 200.0, 20.0, 80.0, 15.0, 55.0), 4.0, 0.0), ("radius 6, rough wall (±0.75)", wavy(40.0, 176.0, 30.0, 120.0, 25.0, 90.0), 6.0, 0.75), ("radius 20", wavy(40.0, 176.0, 25.0, 200.0, 20.0, 160.0), 20.0, 0.0)];
        for (name, axis, r, jitter) in cases {
            let m = stamped_tube(n, &axis, r, jitter);
            let t = Instant::now();
            let c = run(&m, &g, &CenterlineParams::default());
            let took = t.elapsed();
            let b = &c.branches[0];
            let worst = b.points.iter().map(|q| to_polyline(DVec3::from(q.voxel), &axis)).fold(0.0, f64::max);
            let t = Instant::now();
            let s = cross_sections(&m, &g, b, 0.5);
            let areas: Vec<f64> = middle(&s, b.length_mm, r).iter().map(|x| x.area_mm2).collect();
            println!(
                "{name}: {} voxels; centerlines {took:?}: {} branch(es), {:.0} mm, at most {worst:.2} voxel off the axis; {} cross-sections {:?}, area {:.0}–{:.0} mm² (π r² = {:.0})",
                m.count(),
                c.branches.len(),
                b.length_mm,
                s.len(),
                t.elapsed(),
                areas.iter().copied().fold(f64::INFINITY, f64::min),
                areas.iter().copied().fold(0.0, f64::max),
                PI * r * r
            );
            assert_eq!(c.branches.len(), 1);
        }
    }
}
