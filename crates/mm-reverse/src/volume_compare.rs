// SPDX-License-Identifier: AGPL-3.0-only
//! The bodies and the scan compared as solids. Ported from ScanRuler
//! (`src/core/deviation/volume.ts`, `src/core/deviation/winding.ts`,
//! `src/commands/volumeAnswer.ts`, `plugins/reverse/app/reVolumeCompare.ts`), © Stefan
//! Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! On a grid over the part each cell is asked whether it lies inside the bodies and whether
//! inside the scan, and the cells where the answers differ, grouped into connected regions, are
//! material the bodies have and the part has not (excess) or the part has and the bodies have
//! not (missing) — so many cubic millimetres, so deep, wherever they sit. Inside the bodies is
//! exact where they are closed: a column of cells along Z counts the crossings below each cell.
//! Inside the scan is its winding number — about 1 inside and 0 outside, easing towards ½ near
//! its holes — read off a tree of the scan's triangles (Barnes–Hut, to the second moment).
//! What stays near ½ is unsure: told apart, never called an error. A cell within the tolerance
//! of the scan's surface is the scan's noise, not an error.
//!
//! Where ScanRuler closes the scan's holes with the reference's own triangles before reading
//! the winding number (the patch, read off the map on the reference and the scan's open rims),
//! this port reads the scan as it is — ScanRuler's own `farPatch: false` with no map on the
//! reference: across a hole in the scan the cells come out unsure rather than closed. The
//! answer says so (`patch`).

use std::collections::HashMap;

use glam::DVec3;
use mm_core::bvh::Bvh;
use mm_core::Mesh;
use rayon::prelude::*;
use serde_json::{json, Value};

use crate::compare::{Surface, SurfaceBody};

// ---- the winding tree (winding.ts) ----

/// The trees' β by default (the volume compare uses [`DEFAULT_VOLUME_BETA`]).
#[cfg(test)]
const DEFAULT_BETA: f64 = 2.0;
const LEAF_SIZE: usize = 8;
const FOUR_PI: f64 = 4.0 * std::f64::consts::PI;

/// The generalised winding number of a triangle soup at any point, by a tree whose far nodes
/// are read by their moments.
pub struct WindingTree {
    beta2: f64,
    /// Per node: centre (3), radius, area-weighted normal sum (3).
    node: Vec<[f64; 7]>,
    moment: Vec<[f64; 9]>,
    moment2: Vec<[f64; 18]>,
    right: Vec<i32>,
    first: Vec<u32>,
    count: Vec<u8>,
    corners: Vec<[DVec3; 3]>,
}

fn solid_angle(c: &[DVec3; 3], p: DVec3) -> f64 {
    let (a, b, cc) = (c[0] - p, c[1] - p, c[2] - p);
    let (la, lb, lc) = (a.length(), b.length(), cc.length());
    let det = a.dot(b.cross(cc));
    let den = la * lb * lc + a.dot(b) * lc + b.dot(cc) * la + cc.dot(a) * lb;
    det.atan2(den)
}

impl WindingTree {
    pub fn new(positions: &[DVec3], triangles: &[[u32; 3]], beta: f64) -> WindingTree {
        let mut keep = Vec::new();
        let mut g = Vec::new();
        let mut an = Vec::new();
        for (t, tri) in triangles.iter().enumerate() {
            let [a, b, c] = tri.map(|v| positions[v as usize]);
            let n = (b - a).cross(c - a) / 2.0;
            if n == DVec3::ZERO {
                continue;
            }
            keep.push(t);
            an.push(n);
            g.push((a + b + c) / 3.0);
        }
        let kept = keep.len();
        let mut order: Vec<u32> = (0..kept as u32).collect();
        let mut tree = WindingTree { beta2: beta * beta, node: Vec::new(), moment: Vec::new(), moment2: Vec::new(), right: Vec::new(), first: Vec::new(), count: Vec::new(), corners: Vec::new() };
        let corner = |t: usize, k: usize| positions[triangles[t][k] as usize];
        let mut work: Vec<(usize, usize, i32)> = vec![(0, kept, -1)];
        while let Some((start, end, parent)) = work.pop() {
            let id = tree.node.len();
            tree.node.push([0.0; 7]);
            tree.moment.push([0.0; 9]);
            tree.moment2.push([0.0; 18]);
            tree.right.push(-1);
            tree.first.push(0);
            tree.count.push(0);
            if parent >= 0 {
                tree.right[parent as usize] = id as i32;
            }
            // The node's summary.
            let (mut area, mut c, mut nsum) = (0.0, DVec3::ZERO, DVec3::ZERO);
            for &i in &order[start..end] {
                let a = an[i as usize].length();
                area += a;
                c += g[i as usize] * a;
                nsum += an[i as usize];
            }
            if area > 0.0 {
                c /= area;
            }
            let mut r2: f64 = 0.0;
            for &i in &order[start..end] {
                for k in 0..3 {
                    r2 = r2.max((corner(keep[i as usize], k) - c).length_squared());
                }
            }
            tree.node[id] = [c.x, c.y, c.z, r2.sqrt(), nsum.x, nsum.y, nsum.z];
            let mut m1 = [0.0; 9];
            for &i in &order[start..end] {
                let d = g[i as usize] - c;
                let n = an[i as usize];
                for j in 0..3 {
                    for k in 0..3 {
                        m1[j * 3 + k] += d[j] * n[k];
                    }
                }
            }
            tree.moment[id] = m1;
            let mut m2 = [0.0; 18];
            for &i in &order[start..end] {
                let t = keep[i as usize];
                let y = [corner(t, 0) - c, corner(t, 1) - c, corner(t, 2) - c];
                let s = y[0] + y[1] + y[2];
                let pair = |a: usize, b: usize| (y[0][a] * y[0][b] + y[1][a] * y[1][b] + y[2][a] * y[2][b] + s[a] * s[b]) / 12.0;
                let s6 = [pair(0, 0), pair(0, 1), pair(0, 2), pair(1, 1), pair(1, 2), pair(2, 2)];
                let n = an[i as usize];
                for p in 0..6 {
                    for j in 0..3 {
                        m2[p * 3 + j] += s6[p] * n[j];
                    }
                }
            }
            tree.moment2[id] = m2;
            if end - start <= LEAF_SIZE {
                tree.right[id] = -1;
                tree.first[id] = start as u32;
                tree.count[id] = (end - start) as u8;
                continue;
            }
            let (mut lo, mut hi) = (DVec3::splat(f64::INFINITY), DVec3::splat(f64::NEG_INFINITY));
            for &i in &order[start..end] {
                lo = lo.min(g[i as usize]);
                hi = hi.max(g[i as usize]);
            }
            let e = hi - lo;
            let axis = if e.x >= e.y && e.x >= e.z { 0 } else if e.y >= e.z { 1 } else { 2 };
            let mid = (start + end) / 2;
            order[start..end].select_nth_unstable_by(mid - start, |a, b| g[*a as usize][axis].total_cmp(&g[*b as usize][axis]));
            // Left first (the node after this one), then right.
            work.push((mid, end, id as i32));
            work.push((start, mid, -1));
        }
        tree.corners = order.iter().map(|i| {
            let t = keep[*i as usize];
            [corner(t, 0), corner(t, 1), corner(t, 2)]
        }).collect();
        tree
    }

    /// The winding number at a point.
    pub fn at(&self, p: DVec3) -> f64 {
        if self.node.is_empty() {
            return 0.0;
        }
        let mut sum = 0.0;
        let mut stack: Vec<usize> = Vec::with_capacity(64);
        stack.push(0);
        while let Some(id) = stack.pop() {
            let nd = &self.node[id];
            let r = DVec3::new(nd[0] - p.x, nd[1] - p.y, nd[2] - p.z);
            let d2 = r.length_squared();
            if d2 > self.beta2 * nd[3] * nd[3] {
                let d = d2.sqrt();
                let inv3 = 1.0 / (d2 * d);
                sum += (nd[4] * r.x + nd[5] * r.y + nd[6] * r.z) * inv3;
                let m = &self.moment[id];
                let trace = m[0] + m[4] + m[8];
                let rmr = r.x * (m[0] * r.x + m[1] * r.y + m[2] * r.z) + r.y * (m[3] * r.x + m[4] * r.y + m[5] * r.z) + r.z * (m[6] * r.x + m[7] * r.y + m[8] * r.z);
                sum += trace * inv3 - 3.0 * rmr * inv3 / d2;
                let q = &self.moment2[id];
                let (xx, xy, xz, yy, yz, zz) = (0, 3, 6, 9, 12, 15);
                let ux = q[xx] + q[xy + 1] + q[xz + 2];
                let uy = q[xy] + q[yy + 1] + q[yz + 2];
                let uz = q[xz] + q[yz + 1] + q[zz + 2];
                let vx = q[xx] + q[yy] + q[zz];
                let vy = q[xx + 1] + q[yy + 1] + q[zz + 1];
                let vz = q[xx + 2] + q[yy + 2] + q[zz + 2];
                let rq = |o: usize| r.x * q[o] + r.y * q[o + 1] + r.z * q[o + 2];
                let rrr = r.x * r.x * rq(xx) + r.y * r.y * rq(yy) + r.z * r.z * rq(zz) + 2.0 * (r.x * r.y * rq(xy) + r.x * r.z * rq(xz) + r.y * r.z * rq(yz));
                let inv5 = inv3 / d2;
                sum += 0.5 * (-3.0 * inv5 * (2.0 * (r.x * ux + r.y * uy + r.z * uz) + r.x * vx + r.y * vy + r.z * vz) + 15.0 * inv5 * rrr / d2);
                continue;
            }
            let child = self.right[id];
            if child >= 0 {
                stack.push(child as usize);
                stack.push(id + 1);
                continue;
            }
            let s = self.first[id] as usize;
            for t in s..s + self.count[id] as usize {
                sum += 2.0 * solid_angle(&self.corners[t], p);
            }
        }
        sum / FOUR_PI
    }

    /// The winding number summed over every triangle, exactly.
    pub fn exact(&self, p: DVec3) -> f64 {
        self.corners.iter().map(|c| 2.0 * solid_angle(c, p)).sum::<f64>() / FOUR_PI
    }
}

// ---- the volume error (volume.ts) ----

const COARSE: usize = 4;
pub const MAX_CELLS: usize = 16_000_000;
const CELLS_ALONG: f64 = 250.0;
pub const MIN_REGION_CELLS: usize = 8;
const NEAR_HALF: f64 = 0.4;
const DEFAULT_UNSURE: [f64; 2] = [0.3, 0.7];
const PAD_TOLERANCES: f64 = 10.0;
const SCAN_REACH: f64 = 0.25;
pub const DEFAULT_VOLUME_BETA: f64 = 1.5;
const F_SCAN: u8 = 4;
const F_BAND: u8 = 8;

/// The cell for a tolerance on a reference this long.
pub fn default_cell(tolerance: f64, longest: f64) -> f64 {
    (tolerance / 2.0).max(longest / CELLS_ALONG)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VolumeKind {
    Excess,
    Missing,
    Unsure,
}

impl VolumeKind {
    pub fn name(self) -> &'static str {
        match self {
            VolumeKind::Excess => "excess",
            VolumeKind::Missing => "missing",
            VolumeKind::Unsure => "unsure",
        }
    }
}

#[derive(Debug, Clone)]
pub struct VolumeRegion {
    pub kind: VolumeKind,
    pub cells: usize,
    pub volume: f64,
    pub max_depth: f64,
    pub thickness: f64,
    pub min: DVec3,
    pub max: DVec3,
    pub centroid: DVec3,
    pub deepest: DVec3,
    /// The reference triangle nearest the deepest point, by its index in the meshes given.
    pub reference_triangle: Option<usize>,
    pub evidence: f64,
    pub confidence: f64,
}

#[derive(Debug, Clone)]
pub struct VolumeDeviation {
    pub cell: f64,
    pub smallest_seen: f64,
    pub model_volume: f64,
    pub reference_open: Option<(usize, DVec3, DVec3)>,
    pub totals: [(f64, f64); 3],
    pub specks: f64,
    pub regions: Vec<VolumeRegion>,
    pub scan_sign: f64,
    pub seconds: f64,
}

/// What a volume compare is asked.
#[derive(Debug, Clone, Default)]
pub struct VolumeOptions {
    pub tolerance: f64,
    pub cell: Option<f64>,
    pub within: Option<(DVec3, DVec3)>,
}

fn box_of(p: &[DVec3]) -> (DVec3, DVec3) {
    p.iter().fold((DVec3::splat(f64::INFINITY), DVec3::splat(f64::NEG_INFINITY)), |(lo, hi), q| (lo.min(*q), hi.max(*q)))
}

/// 1 for each cell whose centre lies inside the reference, by the crossings of each column of
/// cells along Z; the columns through a hole in the reference read from those beside them.
fn model_inside(p: &[DVec3], tris: &[[u32; 3]], o: DVec3, h: f64, nx: usize, ny: usize, nz: usize) -> (Vec<u8>, Option<(usize, DVec3, DVec3)>) {
    let (ex, ey) = (h * 1.37e-5, h * 2.71e-5);
    let columns = nx * ny;
    let mut hits: Vec<Vec<(f64, i32)>> = vec![Vec::new(); columns];
    for t in tris {
        let [a, b, c] = t.map(|v| p[v as usize]);
        let area = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
        if area == 0.0 {
            continue;
        }
        let i0 = (((a.x.min(b.x).min(c.x) - o.x - ex) / h - 0.5).ceil()).max(0.0) as i64;
        let i1 = (((a.x.max(b.x).max(c.x) - o.x - ex) / h - 0.5).floor()).min(nx as f64 - 1.0) as i64;
        let j0 = (((a.y.min(b.y).min(c.y) - o.y - ey) / h - 0.5).ceil()).max(0.0) as i64;
        let j1 = (((a.y.max(b.y).max(c.y) - o.y - ey) / h - 0.5).floor()).min(ny as f64 - 1.0) as i64;
        if i0 > i1 || j0 > j1 {
            continue;
        }
        let sign = if area > 0.0 { -1 } else { 1 };
        let inv = 1.0 / area;
        for j in j0..=j1 {
            let y = o.y + (j as f64 + 0.5) * h + ey;
            for i in i0..=i1 {
                let x = o.x + (i as f64 + 0.5) * h + ex;
                let wa = ((b.x - x) * (c.y - y) - (b.y - y) * (c.x - x)) * inv;
                let wb = ((c.x - x) * (a.y - y) - (c.y - y) * (a.x - x)) * inv;
                let wc = 1.0 - wa - wb;
                if wa < 0.0 || wb < 0.0 || wc < 0.0 {
                    continue;
                }
                hits[i as usize + j as usize * nx].push((wa * a.z + wb * b.z + wc * c.z, sign));
            }
        }
    }
    let mut inside = vec![0u8; columns * nz];
    let mut open = vec![false; columns];
    let mut open_list = Vec::new();
    let (mut z_lo, mut z_hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for (col, list) in hits.iter_mut().enumerate() {
        if list.is_empty() {
            continue;
        }
        let total: i32 = list.iter().map(|x| x.1).sum();
        if total != 0 {
            open[col] = true;
            open_list.push(col);
            for (z, _) in list.iter() {
                z_lo = z_lo.min(*z);
                z_hi = z_hi.max(*z);
            }
        }
        list.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut q = 0;
        let mut sum = if total < 0 { -total } else { 0 };
        for k in 0..nz {
            let z = o.z + (k as f64 + 0.5) * h;
            while q < list.len() && list[q].0 < z {
                sum += list[q].1;
                q += 1;
            }
            if sum > 0 {
                inside[col + k * columns] = 1;
            }
        }
    }
    if open_list.is_empty() {
        return (inside, None);
    }
    for &col in &open_list {
        let (i, j) = ((col % nx) as i64, (col / nx) as i64);
        let mut ring = Vec::new();
        let mut r = 1i64;
        while ring.is_empty() && r < nx.max(ny) as i64 {
            for dj in -r..=r {
                for di in -r..=r {
                    if di.abs().max(dj.abs()) != r {
                        continue;
                    }
                    let (ii, jj) = (i + di, j + dj);
                    if ii < 0 || jj < 0 || ii >= nx as i64 || jj >= ny as i64 {
                        continue;
                    }
                    let c = ii as usize + jj as usize * nx;
                    if !open[c] && !hits[c].is_empty() {
                        ring.push(c);
                    }
                }
            }
            r += 1;
        }
        for k in 0..nz {
            let says: usize = ring.iter().map(|c| inside[c + k * columns] as usize).sum();
            if 2 * says <= ring.len() {
                inside[col + k * columns] = 0;
            }
        }
    }
    let (mut ilo, mut ihi, mut jlo, mut jhi) = (nx, 0, ny, 0);
    for &col in &open_list {
        let (i, j) = (col % nx, col / nx);
        ilo = ilo.min(i);
        ihi = ihi.max(i);
        jlo = jlo.min(j);
        jhi = jhi.max(j);
    }
    (inside, Some((open_list.len(), DVec3::new(o.x + ilo as f64 * h, o.y + jlo as f64 * h, z_lo), DVec3::new(o.x + (ihi + 1) as f64 * h, o.y + (jhi + 1) as f64 * h, z_hi))))
}

fn cell_index(p: DVec3, o: DVec3, h: f64, n: [usize; 3]) -> Option<usize> {
    let (i, j, k) = (((p.x - o.x) / h).floor(), ((p.y - o.y) / h).floor(), ((p.z - o.z) / h).floor());
    if i < 0.0 || j < 0.0 || k < 0.0 || i >= n[0] as f64 || j >= n[1] as f64 || k >= n[2] as f64 {
        return None;
    }
    Some(i as usize + n[0] * (j as usize + n[1] * k as usize))
}

/// Sets `bit` on every cell a mesh's triangles pass through.
fn stamp(flags: &mut [u8], bit: u8, p: &[DVec3], tris: &[[u32; 3]], o: DVec3, h: f64, n: [usize; 3]) {
    let mut mark = |q: DVec3| {
        if let Some(c) = cell_index(q, o, h, n) {
            flags[c] |= bit;
        }
    };
    for t in tris {
        let c = t.map(|v| p[v as usize]);
        let l = [(c[1] - c[0]).length(), (c[2] - c[1]).length(), (c[0] - c[2]).length()];
        let longest = l[0].max(l[1]).max(l[2]);
        if longest <= h {
            mark(c[0]);
            mark(c[1]);
            mark(c[2]);
            mark((c[0] + c[1] + c[2]) / 3.0);
            continue;
        }
        let first = if longest == l[0] { 0 } else if longest == l[1] { 1 } else { 2 };
        let (a, b, f) = (c[first], c[(first + 1) % 3], c[(first + 2) % 3]);
        let (u, v) = (b - a, f - a);
        let area2 = u.cross(v).length();
        let rows = ((area2 / longest / (h / 2.0)).ceil() as usize).max(1);
        for r in 0..=rows {
            let t = r as f64 / rows as f64;
            let s = a + v * t;
            let e = b + (f - b) * t;
            let steps = ((((1.0 - t) * longest) / (h / 2.0)).ceil() as usize).max(1);
            for q in 0..=steps {
                mark(s + (e - s) * (q as f64 / steps as f64));
            }
        }
    }
}

/// The bodies (`reference`) and the scan compared as solids.
pub fn volume_deviation(reference: (&[DVec3], &[[u32; 3]]), scan: (&[DVec3], &[[u32; 3]]), o: &VolumeOptions) -> Result<VolumeDeviation, String> {
    let started = std::time::Instant::now();
    let t_tol = o.tolerance;
    let (rp, rt) = reference;
    let (sp, st) = scan;
    if rt.is_empty() {
        return Err("There is no solid body to compare — build one first.".into());
    }
    if st.is_empty() {
        return Err("There is no scan to compare the bodies with.".into());
    }
    let tree = WindingTree::new(sp, st, DEFAULT_VOLUME_BETA);
    let (scan_mesh, _) = Mesh::from_triangles(sp, st, None);
    let scan_bvh = Bvh::build(&scan_mesh);
    let ref_mesh = crate::build::BodyMesh { positions: rp.to_vec(), triangles: rt.to_vec(), tri_face: vec![0; rt.len()], ..Default::default() };
    let ref_surface = Surface::of_bodies(&[SurfaceBody { tag: 0, mesh: &ref_mesh, sheet: false }]).ok_or("The bodies have no surface to compare.")?;
    // The grid.
    let (rlo, rhi) = box_of(rp);
    let ext = rhi - rlo;
    let longest = ext.x.max(ext.y).max(ext.z);
    let (slo, shi) = box_of(sp);
    let reach = SCAN_REACH * longest;
    let mut lo = DVec3::ZERO;
    let mut hi = DVec3::ZERO;
    for k in 0..3 {
        lo[k] = (rlo[k] - PAD_TOLERANCES * t_tol).min(slo[k].max(rlo[k] - reach));
        hi[k] = (rhi[k] + PAD_TOLERANCES * t_tol).max(shi[k].min(rhi[k] + reach));
        if let Some((wlo, whi)) = o.within {
            lo[k] = lo[k].max(wlo[k]);
            hi[k] = hi[k].min(whi[k]);
        }
    }
    let mut h = o.cell.unwrap_or_else(|| default_cell(t_tol, longest));
    let cells_at = |s: f64| (0..3).map(|k| ((hi[k] - lo[k]) / s).ceil().max(1.0)).product::<f64>();
    while cells_at(h) > MAX_CELLS as f64 {
        h *= 1.05;
    }
    let nx = ((hi.x - lo.x) / h).ceil().max(1.0) as usize;
    let ny = ((hi.y - lo.y) / h).ceil().max(1.0) as usize;
    let nz = ((hi.z - lo.z) / h).ceil().max(1.0) as usize;
    let nxy = nx * ny;
    let n = nxy * nz;
    let dims = [nx, ny, nz];
    let org = lo;
    let centre = |i: usize, j: usize, k: usize| org + DVec3::new((i as f64 + 0.5) * h, (j as f64 + 0.5) * h, (k as f64 + 0.5) * h);
    // Inside the reference.
    let (model, reference_open) = model_inside(rp, rt, org, h, nx, ny, nz);
    let model_cells: usize = model.iter().map(|m| *m as usize).sum();
    // Which way the scan is wound: the sign of its winding number over cells deep inside.
    let scan_sign = {
        let mut deep = Vec::new();
        for c in 0..n {
            if model[c] == 0 {
                continue;
            }
            let (i, j, k) = (c % nx, (c / nx) % ny, c / nxy);
            if i < 2 || j < 2 || k < 2 || i >= nx - 2 || j >= ny - 2 || k >= nz - 2 {
                continue;
            }
            if model[c - 2] != 0 && model[c + 2] != 0 && model[c - 2 * nx] != 0 && model[c + 2 * nx] != 0 && model[c - 2 * nxy] != 0 && model[c + 2 * nxy] != 0 {
                deep.push(c);
            }
        }
        let cells: Vec<usize> = if deep.is_empty() { (0..n).filter(|c| model[*c] != 0).collect() } else { deep };
        let step = (cells.len() / 64).max(1);
        let sum: f64 = cells.iter().step_by(step).map(|c| tree.at(centre(c % nx, (c / nx) % ny, c / nxy))).sum();
        if sum < 0.0 {
            -1.0
        } else {
            1.0
        }
    };
    let winding = |p: DVec3| scan_sign * tree.at(p);
    // Coarse.
    let (cx, cy, cz) = (nx.div_ceil(COARSE), ny.div_ceil(COARSE), nz.div_ceil(COARSE));
    let cn = cx * cy * cz;
    let half = COARSE as f64 / 2.0 * h;
    let coarse_at = |b: usize| {
        let (i, j, k) = (b % cx, (b / cx) % cy, b / (cx * cy));
        org + DVec3::new((i * COARSE) as f64 * h + half, (j * COARSE) as f64 * h + half, (k * COARSE) as f64 * h + half)
    };
    let coarse: Vec<f64> = (0..cn).into_par_iter().map(|b| winding(coarse_at(b))).collect();
    // Where to read again: where a scan triangle crosses, near ½, and round the rims.
    let mut refine = vec![0u8; cn];
    let s = COARSE as f64 * h;
    let cdims = [cx, cy, cz];
    for t in st {
        let c = t.map(|v| sp[v as usize]);
        let (blo, bhi) = (c[0].min(c[1]).min(c[2]), c[0].max(c[1]).max(c[2]));
        let mut r0 = [0usize; 3];
        let mut r1 = [0usize; 3];
        let mut outside = false;
        for k in 0..3 {
            let a = ((blo[k] - org[k]) / s).floor().max(0.0);
            let b = ((bhi[k] - org[k]) / s).floor().min(cdims[k] as f64 - 1.0);
            if a > b {
                outside = true;
                break;
            }
            r0[k] = a as usize;
            r1[k] = b as usize;
        }
        if outside {
            continue;
        }
        for k in r0[2]..=r1[2] {
            for j in r0[1]..=r1[1] {
                for i in r0[0]..=r1[0] {
                    refine[i + cx * (j + cy * k)] = 1;
                }
            }
        }
    }
    let near_half: Vec<bool> = coarse.iter().map(|w| (w - 0.5).abs() < NEAR_HALF).collect();
    for k in 0..cz {
        for j in 0..cy {
            for i in 0..cx {
                if !near_half[i + cx * (j + cy * k)] {
                    continue;
                }
                for dk in k.saturating_sub(1)..=(k + 1).min(cz - 1) {
                    for dj in j.saturating_sub(1)..=(j + 1).min(cy - 1) {
                        for di in i.saturating_sub(1)..=(i + 1).min(cx - 1) {
                            refine[di + cx * (dj + cy * dk)] = 1;
                        }
                    }
                }
            }
        }
    }
    let mut flags = vec![0u8; n];
    stamp(&mut flags, F_SCAN, sp, st, org, h, dims);
    for c in 0..n {
        if flags[c] & F_SCAN == 0 {
            continue;
        }
        let (i, j, k) = (c % nx, (c / nx) % ny, c / nxy);
        flags[c] |= F_BAND;
        if i > 0 {
            flags[c - 1] |= F_BAND;
        }
        if i < nx - 1 {
            flags[c + 1] |= F_BAND;
        }
        if j > 0 {
            flags[c - nx] |= F_BAND;
        }
        if j < ny - 1 {
            flags[c + nx] |= F_BAND;
        }
        if k > 0 {
            flags[c - nxy] |= F_BAND;
        }
        if k < nz - 1 {
            flags[c + nxy] |= F_BAND;
        }
    }
    // The coarse cells near the scan's rims are read cell by cell.
    let mut full = vec![0u8; cn];
    {
        let mut sides: HashMap<(u32, u32), u32> = HashMap::new();
        for t in st {
            for q in 0..3 {
                let (a, b) = (t[q], t[(q + 1) % 3]);
                *sides.entry((a.min(b), a.max(b))).or_insert(0) += 1;
            }
        }
        for ((a, b), cnt) in sides {
            if cnt != 1 {
                continue;
            }
            for v in [a, b] {
                let p = sp[v as usize];
                let r = |k: usize, d: f64| ((p[k] + d - org[k]) / s).floor();
                let (i0, i1) = (r(0, -1.5 * h).max(0.0) as usize, r(0, 1.5 * h).min(cx as f64 - 1.0));
                let (j0, j1) = (r(1, -1.5 * h).max(0.0) as usize, r(1, 1.5 * h).min(cy as f64 - 1.0));
                let (k0, k1) = (r(2, -1.5 * h).max(0.0) as usize, r(2, 1.5 * h).min(cz as f64 - 1.0));
                if i1 < 0.0 || j1 < 0.0 || k1 < 0.0 {
                    continue;
                }
                for k in k0..=k1 as usize {
                    for j in j0..=j1 as usize {
                        for i in i0..=i1 as usize {
                            full[i + cx * (j + cy * k)] = 1;
                        }
                    }
                }
            }
        }
    }
    // Fine: per coarse cell, its cells' winding numbers.
    let [unsure_lo, unsure_hi] = DEFAULT_UNSURE;
    let hh = COARSE / 2;
    let blocks: Vec<Vec<(usize, f64)>> = (0..cn)
        .into_par_iter()
        .map(|b| {
            let (bi, bj, bk) = (b % cx, (b / cx) % cy, b / (cx * cy));
            let (i0, j0, k0) = (bi * COARSE, bj * COARSE, bk * COARSE);
            let (si, sj, sk) = (COARSE.min(nx - i0), COARSE.min(ny - j0), COARSE.min(nz - k0));
            let cell_at = |li: usize, lj: usize, lk: usize| i0 + li + nx * (j0 + lj + ny * (k0 + lk));
            let read = |li: usize, lj: usize, lk: usize| winding(centre(i0 + li, j0 + lj, k0 + lk));
            let mut local = [coarse[b]; COARSE * COARSE * COARSE];
            if refine[b] == 0 {
            } else if full[b] != 0 {
                for lk in 0..sk {
                    for lj in 0..sj {
                        for li in 0..si {
                            local[li + COARSE * (lj + COARSE * lk)] = read(li, lj, lk);
                        }
                    }
                }
            } else {
                let mut group = [-1i8; COARSE * COARSE * COARSE];
                let mut crossed = false;
                for lk in 0..sk {
                    for lj in 0..sj {
                        for li in 0..si {
                            let l = li + COARSE * (lj + COARSE * lk);
                            if flags[cell_at(li, lj, lk)] & F_BAND != 0 {
                                local[l] = read(li, lj, lk);
                                group[l] = 0;
                                crossed = true;
                            }
                        }
                    }
                }
                let octant = |i: usize, j: usize, k: usize| (i >= hh) as usize + 2 * (j >= hh) as usize + 4 * (k >= hh) as usize;
                let mut stack = Vec::new();
                for lk in 0..sk {
                    for lj in 0..sj {
                        for li in 0..si {
                            let l0 = li + COARSE * (lj + COARSE * lk);
                            if group[l0] != -1 {
                                continue;
                            }
                            let value = read(li, lj, lk);
                            let corner = if crossed { None } else { Some(octant(li, lj, lk)) };
                            stack.push(l0);
                            group[l0] = 1;
                            while let Some(l) = stack.pop() {
                                local[l] = value;
                                let (a, bb, c) = (l % COARSE, (l / COARSE) % COARSE, l / (COARSE * COARSE));
                                for s2 in 0..6 {
                                    let (ni, nj, nk) = (a as i64 + [-1, 1, 0, 0, 0, 0][s2], bb as i64 + [0, 0, -1, 1, 0, 0][s2], c as i64 + [0, 0, 0, 0, -1, 1][s2]);
                                    if ni < 0 || nj < 0 || nk < 0 || ni >= si as i64 || nj >= sj as i64 || nk >= sk as i64 {
                                        continue;
                                    }
                                    let (ni, nj, nk) = (ni as usize, nj as usize, nk as usize);
                                    let m = ni + COARSE * (nj + COARSE * nk);
                                    if group[m] != -1 {
                                        continue;
                                    }
                                    if corner.is_some_and(|q| octant(ni, nj, nk) != q) {
                                        continue;
                                    }
                                    group[m] = 1;
                                    stack.push(m);
                                }
                            }
                        }
                    }
                }
            }
            let mut out = Vec::with_capacity(si * sj * sk);
            for lk in 0..sk {
                for lj in 0..sj {
                    for li in 0..si {
                        out.push((cell_at(li, lj, lk), local[li + COARSE * (lj + COARSE * lk)]));
                    }
                }
            }
            out
        })
        .collect();
    let mut kind = vec![0u8; n];
    let mut wq = vec![0u8; n];
    for list in blocks {
        for (c, w) in list {
            wq[c] = ((w * 200.0).round() + 25.0).clamp(0.0, 255.0) as u8;
            if w >= unsure_lo && w <= unsure_hi {
                kind[c] = 3;
            } else if (w > unsure_hi) as u8 != model[c] {
                kind[c] = if model[c] != 0 { 1 } else { 2 };
            }
        }
    }
    // Within the tolerance of the scan's surface: noise.
    let drop: Vec<usize> = (0..n)
        .into_par_iter()
        .filter(|c| kind[*c] != 0)
        .filter(|c| scan_bvh.nearest(&scan_mesh, centre(c % nx, (c / nx) % ny, c / nxy), t_tol).is_some_and(|h| h.dist_sq.sqrt() <= t_tol))
        .collect();
    for c in drop {
        kind[c] = 0;
    }
    // Regions.
    const VISITED: u8 = 0x80;
    let cell_volume = h * h * h;
    let mut found: Vec<(u8, Vec<usize>)> = Vec::new();
    let mut specks = 0.0;
    for start in 0..n {
        let kd = kind[start];
        if kd == 0 || kd & VISITED != 0 {
            continue;
        }
        let mut queue = vec![start];
        kind[start] = kd | VISITED;
        let mut head = 0;
        while head < queue.len() {
            let c = queue[head];
            head += 1;
            let (i, j, k) = (c % nx, (c / nx) % ny, c / nxy);
            let nb = [(i > 0).then(|| c - 1), (i < nx - 1).then(|| c + 1), (j > 0).then(|| c - nx), (j < ny - 1).then(|| c + nx), (k > 0).then(|| c - nxy), (k < nz - 1).then(|| c + nxy)];
            for d in nb.into_iter().flatten() {
                if kind[d] != kd {
                    continue;
                }
                kind[d] = kd | VISITED;
                queue.push(d);
            }
        }
        if queue.len() < MIN_REGION_CELLS {
            specks += queue.len() as f64 * cell_volume;
        } else {
            found.push((kd, queue));
        }
    }
    let mut regions = Vec::new();
    for (kd, cells) in &found {
        let mark = kd | VISITED | 0x40;
        for c in cells {
            kind[*c] = mark;
        }
        let (mut sum, mut conf) = (DVec3::ZERO, 0.0);
        let (mut bmin, mut bmax) = (DVec3::splat(f64::INFINITY), DVec3::splat(f64::NEG_INFINITY));
        let (mut max_depth, mut deepest) = (-1.0f64, cells[0]);
        let (mut model_faces, mut other_faces) = (0usize, 0usize);
        for &c in cells {
            let (i, j, k) = (c % nx, (c / nx) % ny, c / nxy);
            let x = centre(i, j, k);
            sum += x;
            bmin = bmin.min(x);
            bmax = bmax.max(x);
            conf += (((wq[c] as f64 - 25.0) / 200.0 - 0.5).abs() * 2.0).min(1.0);
            let depth = scan_bvh.nearest(&scan_mesh, x, f64::INFINITY).map_or(0.0, |h| h.dist_sq.sqrt());
            if depth > max_depth {
                max_depth = depth;
                deepest = c;
            }
            let nb = [(i > 0).then(|| c - 1), (i < nx - 1).then(|| c + 1), (j > 0).then(|| c - nx), (j < ny - 1).then(|| c + nx), (k > 0).then(|| c - nxy), (k < nz - 1).then(|| c + nxy)];
            for d in nb {
                if d.is_some_and(|d| kind[d] == mark) {
                    continue;
                }
                match d {
                    Some(d) if model[d] != model[c] => model_faces += 1,
                    _ => other_faces += 1,
                }
            }
        }
        for c in cells {
            kind[*c] = kd | VISITED;
        }
        let deep = centre(deepest % nx, (deepest / nx) % ny, deepest / nxy);
        let near = ref_surface.closest(deep, f64::INFINITY);
        let faces = if model_faces.min(other_faces) > 0 { model_faces.min(other_faces) } else { model_faces.max(other_faces) };
        let volume = cells.len() as f64 * cell_volume;
        regions.push(VolumeRegion {
            kind: [VolumeKind::Excess, VolumeKind::Missing, VolumeKind::Unsure][(*kd as usize).clamp(1, 3) - 1],
            cells: cells.len(),
            volume,
            max_depth,
            thickness: if faces > 0 { volume / (faces as f64 * h * h) } else { 0.0 },
            min: bmin - DVec3::splat(h / 2.0),
            max: bmax + DVec3::splat(h / 2.0),
            centroid: sum / cells.len() as f64,
            deepest: deep,
            reference_triangle: near.map(|n| n.tri),
            // With no patch, every side the scan bounds is the real scan's.
            evidence: 1.0,
            confidence: conf / cells.len() as f64,
        });
    }
    regions.sort_by(|a, b| b.volume.total_cmp(&a.volume));
    let model_volume = model_cells as f64 * cell_volume;
    let mut totals = [(0.0, 0.0); 3];
    for (k, kind) in [VolumeKind::Excess, VolumeKind::Missing, VolumeKind::Unsure].iter().enumerate() {
        let v: f64 = regions.iter().filter(|r| r.kind == *kind).map(|r| r.volume).sum();
        totals[k] = (v, if model_volume > 0.0 { 100.0 * v / model_volume } else { 0.0 });
    }
    Ok(VolumeDeviation { cell: h, smallest_seen: 2.0 * h, model_volume, reference_open, totals, specks, regions, scan_sign, seconds: started.elapsed().as_secs_f64() })
}

// ---- the answer (volumeAnswer.ts) ----

pub const VOLUME_REGIONS: usize = 8;
pub const MAX_REGIONS: usize = 50;

fn r1(v: f64) -> f64 {
    (v * 10.0).round() / 10.0
}
fn r2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}
fn r3(v: f64) -> f64 {
    (v * 1000.0).round() / 1000.0
}
fn v3(p: DVec3) -> Value {
    json!([r3(p.x), r3(p.y), r3(p.z)])
}

/// A region as an answer says it.
pub fn region_out(r: &VolumeRegion, cell: f64) -> Value {
    json!({
        "kind": r.kind.name(),
        "volume": r1(r.volume),
        "maxDepth": r2(r.max_depth),
        "thickness": r2(r.thickness),
        "shape": if r.thickness < 3.0 * cell { "layer" } else { "lump" },
        "centroid": v3(r.centroid),
        "deepest": v3(r.deepest),
        "box": {"min": v3(r.min), "max": v3(r.max)},
        "evidence": r2(r.evidence),
        "confidence": r2(r.confidence),
    })
}

/// The figures of a measurement, its largest `regions` regions named (the unsure only from 64
/// cells), the kinds given — each with the body face nearest its deepest point as `name_face`
/// says it.
pub fn volume_out(v: &VolumeDeviation, tolerance: f64, regions: usize, kinds: &[VolumeKind], name_face: &dyn Fn(Option<usize>) -> (Value, Value)) -> Value {
    let listed: Vec<&VolumeRegion> = v.regions.iter().filter(|r| kinds.contains(&r.kind) && (r.kind != VolumeKind::Unsure || r.cells >= 64)).take(regions).collect();
    let mut out = json!({
        "tolerance": tolerance,
        "cell": r3(v.cell),
        "smallestSeen": r3(v.smallest_seen),
        "modelVolume": r1(v.model_volume),
        "totals": {
            "excess": {"volume": r1(v.totals[0].0), "share": r2(v.totals[0].1)},
            "missing": {"volume": r1(v.totals[1].0), "share": r2(v.totals[1].1)},
            "unsure": {"volume": r1(v.totals[2].0), "share": r2(v.totals[2].1)},
        },
        "found": v.regions.len(),
        "regions": listed.iter().map(|r| {
            let mut row = region_out(r, v.cell);
            let (body, face) = name_face(r.reference_triangle);
            row["body"] = body;
            row["face"] = face;
            row
        }).collect::<Vec<_>>(),
        "seconds": r1(v.seconds),
        "patch": "the scan read as it is: its holes are not closed with the bodies' own faces, so cells across a hole read unsure",
    });
    if let Some((cols, lo, hi)) = v.reference_open {
        out["referenceOpen"] = json!({"columns": cols, "box": {"min": v3(lo), "max": v3(hi)}});
        out["warnings"] = json!([format!("The reference's mesh has a hole over {cols} column{} of cells, in ({})–({}): read from the columns beside it, so no lump stands on it, and no finer than that there.", if cols == 1 { "" } else { "s" }, [lo.x, lo.y, lo.z].map(|x| crate::js_fmt::num(r3(x))).join(", "), [hi.x, hi.y, hi.z].map(|x| crate::js_fmt::num(r3(x))).join(", "))]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cube(size: f64, at: DVec3, n: usize) -> (Vec<DVec3>, Vec<[u32; 3]>) {
        let m = mm_core::primitives::cube(size, n);
        let p = m.positions().iter().map(|q| *q + at).collect();
        let t = m.triangle_ids().map(|t| m.triangle(t)).collect();
        (p, t)
    }

    // Ported from ScanRuler's tests (winding.test.ts): inside a closed surface the winding
    // number is 1, outside 0, and the tree agrees with the exact sum.
    #[test]
    fn the_winding_number_is_one_inside_and_nought_outside() {
        let (p, t) = cube(10.0, DVec3::ZERO, 6);
        let tree = WindingTree::new(&p, &t, DEFAULT_BETA);
        // One whole turn inside, its sign the winding's; none outside.
        assert!((tree.exact(DVec3::ZERO).abs() - 1.0).abs() < 1e-9);
        assert!((tree.at(DVec3::ZERO).abs() - 1.0).abs() < 2e-2, "{}", tree.at(DVec3::ZERO));
        assert!(tree.at(DVec3::new(30.0, 0.0, 0.0)).abs() < 1e-3);
        for q in [DVec3::new(4.0, 1.0, -2.0), DVec3::new(6.0, 0.5, 0.5), DVec3::new(20.0, 20.0, 1.0)] {
            assert!((tree.at(q) - tree.exact(q)).abs() < 2e-2, "{q}");
        }
    }

    // Ported from ScanRuler's tests/reVolumeCompare.test.ts: a box modelled a millimetre too
    // tall over a scan of the true box reads that slab as excess, and a box modelled too
    // short reads it missing.
    #[test]
    fn a_body_too_tall_reads_the_slab_as_excess_and_too_short_as_missing() {
        let (sp, st) = cube(20.0, DVec3::ZERO, 8);
        // The body: 20 × 20 × 22, its top 2 mm above the scan's.
        let (mut bp, bt) = cube(20.0, DVec3::ZERO, 2);
        for p in &mut bp {
            if p.z > 0.0 {
                p.z += 2.0;
            }
        }
        let v = volume_deviation((&bp, &bt), (&sp, &st), &VolumeOptions { tolerance: 0.1, cell: Some(0.25), within: None }).unwrap();
        let excess = v.totals[0].0;
        assert!((excess - 800.0).abs() < 80.0, "excess {excess}");
        assert!(v.totals[1].0 < 20.0, "missing {}", v.totals[1].0);
        assert_eq!(v.regions[0].kind, VolumeKind::Excess);
        assert!((v.regions[0].max_depth - 2.0).abs() < 0.3);
        assert!(v.reference_open.is_none());
        // And the other way round.
        let (mut short, st2) = cube(20.0, DVec3::ZERO, 2);
        for p in &mut short {
            if p.z > 0.0 {
                p.z -= 2.0;
            }
        }
        let v = volume_deviation((&short, &st2), (&sp, &st), &VolumeOptions { tolerance: 0.1, cell: Some(0.25), within: None }).unwrap();
        assert!((v.totals[1].0 - 800.0).abs() < 80.0, "missing {}", v.totals[1].0);
        assert!(v.totals[0].0 < 20.0);
        let out = volume_out(&v, 0.1, VOLUME_REGIONS, &[VolumeKind::Excess, VolumeKind::Missing, VolumeKind::Unsure], &|_| (Value::Null, Value::Null));
        assert_eq!(out["regions"][0]["kind"], json!("missing"));
        assert_eq!(out["regions"][0]["shape"], json!("lump"));
    }
}
