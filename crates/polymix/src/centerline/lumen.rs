//! Centerlines through the lumen, and a check that a centerline stays inside it.
//!
//! [`super::compute`]'s rings follow the surface, and where a vessel is not a simple tube (a
//! confluence such as the torcula, a fenestration or septum, a tight bend) a ring's centroid
//! can lie in the tissue between channels. These centerlines are found from the inside
//! instead, through the centres of the largest spheres inside the vessel, as VMTK's run:
//!
//! 1. The lumen (the surface with its open ends capped) is voxelized row by row from the
//!    signed crossings of a ray along each row: a ray grazing the surface crosses it twice in
//!    opposite senses, which cancel, and a ray through an edge or vertex crosses it once. Rows
//!    through a hole the capping left are decided by the winding number.
//! 2. Every lumen voxel gets its distance to the vessel wall, measured to the surface (not the
//!    grid). Neighbouring voxels are linked only where the segment between them is inside, so
//!    a septum or a fenestration's bridge thinner than a voxel still parts its two sides.
//! 3. TEASAR (Sato et al. 2000) on that graph from the inlet: shortest paths under a step cost
//!    that rises steeply toward the wall, first to each vessel end, then to the farthest voxels
//!    not yet covered. Where a new branch's tip reaches, through the lumen, a part of the tree
//!    far from it along the tree (the other channel of a fenestration), the two are joined
//!    into a loop; other short side branches (bumps of the wall) are dropped.
//! 4. Each branch is smoothed and each point moved, within its cross-section, to where the
//!    wall is farthest. A point only moves within the largest ball about it that the surface
//!    does not enter, and a segment between two points is kept only where it is proven inside
//!    (their balls overlap along it, or a ray along it meets no surface); otherwise the points
//!    go back to the voxel path, whose steps were proven so. Points are thinned to the spacing
//!    the same way.
//! 5. [`check`] then tests every point (winding number) and segment (ray) against the
//!    surface, whatever made them.

use super::{Branch, CappedCache, Centerline, CenterlineNode, Item};
use anyhow::{anyhow, bail};
use glam::DVec3;
use mm_core::{Bvh, Mesh, Ray};
use rayon::prelude::*;
use std::collections::{BinaryHeap, HashMap};
use std::f64::consts::PI;
use std::sync::OnceLock;

const NONE: u32 = u32::MAX;

/// Voxel neighbour offsets (i, j, k), ordered so that the opposite of offset `d` is `25 - d`.
const OFFSETS: [[i64; 3]; 26] = {
    let mut o = [[0i64; 3]; 26];
    let (mut n, mut c) = (0, 0i64);
    while c < 27 {
        if c != 13 {
            o[n] = [c % 3 - 1, (c / 3) % 3 - 1, c / 9 - 1];
            n += 1;
        }
        c += 1;
    }
    o
};

/// A vessel end, or a point picked on the wall near the inlet.
#[derive(Clone, Copy, Debug)]
pub struct LumenEnd {
    pub at: DVec3,
    /// The vessel's radius there, mm (sets the grid's resolution and the spacing).
    pub radius: f64,
    /// `at` is the centre of an opening (an open end or a flat end cap), which the centerline
    /// runs to exactly; otherwise it is a point on the wall, and the centerline starts on the
    /// axis inside.
    pub opening: bool,
}

#[derive(Clone, Debug)]
pub struct LumenInput {
    pub inlet: LumenEnd,
    /// The other vessel ends, each reached by a branch.
    pub outlets: Vec<LumenEnd>,
    /// Face groups of flat end caps: not wall, so radii are measured to the wall only.
    pub cap_groups: Vec<i32>,
}

#[derive(Clone, Copy, Debug)]
pub struct LumenParams {
    /// Spacing of the points along the centerline, mm (<= 0: automatic, a fifth of the
    /// inlet's radius). Bends keep more points.
    pub spacing: f64,
    /// Blind side branches running on for less than this multiple of their radius beyond the
    /// vessel they leave (no further than a bump on its wall) are dropped.
    pub prune_factor: f64,
    /// Voxel size, mm (<= 0: automatic, from the narrowest end's radius).
    pub voxel: f64,
    /// Also follow blind side branches (vessels without an open end), as when there are no
    /// other ends. Otherwise the tree joins the ends, with loops round fenestrations.
    pub blind_branches: bool,
}

impl Default for LumenParams {
    fn default() -> Self {
        Self { spacing: 0.0, prune_factor: 2.5, voxel: 0.0, blind_branches: false }
    }
}

/// What [`check`] found.
#[derive(Clone, Debug, Default)]
pub struct LumenCheck {
    pub points: usize,
    pub segments: usize,
    /// Points outside the lumen.
    pub outside: Vec<DVec3>,
    /// Midpoints of the segments that cross the surface.
    pub crossing: Vec<DVec3>,
    /// Smallest distance from a point inside to the vessel wall (the surface less its end
    /// caps), mm.
    pub min_clearance: f64,
}

impl LumenCheck {
    pub fn passed(&self) -> bool {
        self.outside.is_empty() && self.crossing.is_empty()
    }
}

/// A centerline through the lumen, with its check.
#[derive(Clone, Debug)]
pub struct LumenCenterline {
    pub centerline: Centerline,
    pub check: LumenCheck,
    /// Outlets (indices into [`LumenInput::outlets`]) the lumen does not connect to the inlet.
    pub unreached: Vec<usize>,
    /// Loops closed around fenestrations.
    pub loops: usize,
    /// Voxel size used, mm.
    pub voxel: f64,
}

/// The lumen's closed surface (open ends capped) and the vessel wall (the surface less any
/// end caps), with their trees.
struct Surfaces<'a> {
    closed: &'a Mesh,
    closed_bvh: &'a Bvh,
    wall: &'a Mesh,
    wall_bvh: Bvh,
}

impl Surfaces<'_> {
    /// Distance to the vessel wall.
    fn wall(&self, p: DVec3) -> f64 {
        self.wall_bvh.nearest(self.wall, p, f64::INFINITY).map_or(0.0, |h| h.dist_sq.sqrt())
    }

    fn wall_point(&self, p: DVec3) -> Option<DVec3> {
        self.wall_bvh.nearest(self.wall, p, f64::INFINITY).map(|h| h.point)
    }

    /// Distance to the closed surface (caps included) of a point `wall` from the wall.
    fn clearance(&self, p: DVec3, wall: f64) -> f64 {
        self.closed_bvh.nearest(self.closed, p, wall).map_or(wall, |h| h.dist_sq.sqrt().min(wall))
    }

    /// Whether the segment from `a` to `b` meets the closed surface (not counting where it
    /// touches it at an end, as at an opening's centre).
    fn blocked(&self, a: DVec3, b: DVec3) -> bool {
        segment_blocked(self.closed, self.closed_bvh, a, b)
    }

    /// Whether the segment between two points inside the lumen, `ca` and `cb` from the
    /// surface, is inside: their balls overlap along it, or a ray along it meets nothing.
    fn proven(&self, a: DVec3, ca: f64, b: DVec3, cb: f64) -> bool {
        ca + cb > a.distance(b) * (1.0 + 1e-9) || !self.blocked(a, b)
    }
}

fn segment_blocked(mesh: &Mesh, bvh: &Bvh, a: DVec3, b: DVec3) -> bool {
    let d = b - a;
    let len = d.length();
    len > 0.0 && bvh.raycast_range(mesh, &Ray::new(a, d), len * 1e-7, len * (1.0 - 1e-7), |_| true).is_some()
}

#[derive(Clone, Copy)]
struct Span {
    i0: u32,
    /// Last voxel (inclusive).
    i1: u32,
    /// Id of voxel `i0`.
    first: u32,
}

/// The lumen's voxels, row by row (rows along x).
struct Grid {
    /// Corner of voxel (0, 0, 0).
    lo: DVec3,
    v: f64,
    dims: [usize; 3],
    /// Spans of row `j + k·ny`: `spans[rows[r]..rows[r + 1]]`.
    rows: Vec<u32>,
    spans: Vec<Span>,
    ijk: Vec<[u32; 3]>,
}

impl Grid {
    fn len(&self) -> usize {
        self.ijk.len()
    }

    fn centre(&self, a: u32) -> DVec3 {
        let [i, j, k] = self.ijk[a as usize];
        self.lo + DVec3::new(i as f64 + 0.5, j as f64 + 0.5, k as f64 + 0.5) * self.v
    }

    /// The lumen voxel at (i, j, k), or `NONE`.
    fn id(&self, i: i64, j: i64, k: i64) -> u32 {
        let [nx, ny, nz] = self.dims;
        if i < 0 || j < 0 || k < 0 || i >= nx as i64 || j >= ny as i64 || k >= nz as i64 {
            return NONE;
        }
        let r = j as usize + k as usize * ny;
        let i = i as u32;
        for s in &self.spans[self.rows[r] as usize..self.rows[r + 1] as usize] {
            if i < s.i0 {
                break;
            }
            if i <= s.i1 {
                return s.first + (i - s.i0);
            }
        }
        NONE
    }

    fn neighbour(&self, a: u32, d: usize) -> u32 {
        let [i, j, k] = self.ijk[a as usize];
        let o = OFFSETS[d];
        self.id(i as i64 + o[0], j as i64 + o[1], k as i64 + o[2])
    }

    /// The lumen voxels whose centres lie within `r` of `p`.
    fn ball(&self, p: DVec3, r: f64, mut f: impl FnMut(u32)) {
        let c = (p - self.lo) / self.v - DVec3::splat(0.5);
        let rv = r / self.v;
        let [nx, ny, nz] = self.dims.map(|d| d as i64);
        let (k0, k1) = (((c.z - rv).ceil() as i64).max(0), ((c.z + rv).floor() as i64).min(nz - 1));
        for k in k0..=k1 {
            let left_z = rv * rv - (k as f64 - c.z).powi(2);
            if left_z < 0.0 {
                continue;
            }
            let hy = left_z.sqrt();
            let (j0, j1) = (((c.y - hy).ceil() as i64).max(0), ((c.y + hy).floor() as i64).min(ny - 1));
            for j in j0..=j1 {
                let left = left_z - (j as f64 - c.y).powi(2);
                if left < 0.0 {
                    continue;
                }
                let hx = left.sqrt();
                let (i0, i1) = (((c.x - hx).ceil() as i64).max(0), ((c.x + hx).floor() as i64).min(nx - 1));
                if i1 < i0 {
                    continue;
                }
                let row = j as usize + k as usize * ny as usize;
                for s in &self.spans[self.rows[row] as usize..self.rows[row + 1] as usize] {
                    let (a, b) = ((s.i0 as i64).max(i0), (s.i1 as i64).min(i1));
                    for i in a..=b {
                        f(s.first + (i - s.i0 as i64) as u32);
                    }
                }
            }
        }
    }
}

/// The voxels inside the closed `surface` on a grid of `v` mm voxels, voxel (0, 0, 0)'s corner
/// at `lo` (outside the surface's bounds).
fn voxelize(surface: &Mesh, bvh: &Bvh, lo: DVec3, v: f64, dims: [usize; 3], winding: &OnceLock<mm_solid::WindingNumber>, going: &dyn Fn(f32) -> bool) -> Option<Grid> {
    let [nx, ny, _] = dims;
    let total = ny * dims[2];
    let x0 = lo.x - v;
    let index = |x: f64| (x - lo.x) / v - 0.5;
    let row = |hits: &mut Vec<mm_core::bvh::RayHit>, r: usize| -> Vec<(u32, u32)> {
            let (j, k) = (r % ny, r / ny);
            // Nudged off the grid, so a ray does not run exactly along an edge.
            let (y, z) = (lo.y + (j as f64 + 0.500_37) * v, lo.z + (k as f64 + 0.500_61) * v);
            bvh.raycast_all(surface, &Ray::new(DVec3::new(x0, y, z), DVec3::X), hits);
            if hits.is_empty() {
                return Vec::new();
            }
            // Crossings along the ray, +1 entering and −1 leaving (by the triangle's facing).
            let mut ev: Vec<(f64, i32)> = hits
                .iter()
                .filter_map(|h| {
                    let s = surface.tri_cross(h.tid).x;
                    (s != 0.0).then_some((x0 + h.t, if s < 0.0 { 1 } else { -1 }))
                })
                .collect();
            ev.sort_by(|a, b| a.0.total_cmp(&b.0));
            // Crossings at one point (the triangles around an edge or vertex the ray passes
            // through) count once, or not at all where they cancel (a graze).
            let mut merged: Vec<(f64, i32)> = Vec::with_capacity(ev.len());
            let mut a = 0;
            while a < ev.len() {
                let (mut b, mut sum) = (a, 0);
                while b < ev.len() && ev[b].0 - ev[a].0 <= 1e-9 * (1.0 + ev[a].0.abs()) {
                    sum += ev[b].1;
                    b += 1;
                }
                if sum != 0 {
                    merged.push((ev[a].0, sum.signum()));
                }
                a = b;
            }
            let mut spans: Vec<(u32, u32)> = Vec::new();
            let mut push = |i0: f64, i1: f64| {
                let (i0, i1) = (i0.ceil().max(0.0), i1.floor().min(nx as f64 - 1.0));
                if i1 >= i0 {
                    let (i0, i1) = (i0 as u32, i1 as u32);
                    match spans.last_mut() {
                        Some(last) if last.1 + 1 >= i0 => last.1 = last.1.max(i1),
                        _ => spans.push((i0, i1)),
                    }
                }
            };
            if merged.iter().map(|e| e.1).sum::<i32>() == 0 {
                let mut s = 0;
                for w in 0..merged.len().saturating_sub(1) {
                    s += merged[w].1;
                    if s != 0 {
                        push(index(merged[w].0), index(merged[w + 1].0));
                    }
                }
            } else if let (Some(first), Some(last)) = (merged.first(), merged.last()) {
                // The ray passes through a hole: its voxels are decided by the winding number.
                let wn = winding.get_or_init(|| mm_solid::WindingNumber::new(surface));
                let (i0, i1) = (index(first.0).ceil().max(0.0) as i64, (index(last.0).floor() as i64).min(nx as i64 - 1));
                for i in i0..=i1 {
                    let c = DVec3::new(lo.x + (i as f64 + 0.5) * v, y, z);
                    if wn.eval(c).abs() >= 0.5 {
                        push(i as f64, i as f64);
                    }
                }
            }
            spans
    };
    let mut rows: Vec<Vec<(u32, u32)>> = Vec::with_capacity(total);
    for start in (0..total).step_by(16384) {
        if !going(start as f32 / total as f32) {
            return None;
        }
        rows.par_extend((start..(start + 16384).min(total)).into_par_iter().map_init(Vec::new, |hits, r| row(hits, r)));
    }
    let mut grid = Grid { lo, v, dims, rows: Vec::with_capacity(rows.len() + 1), spans: Vec::new(), ijk: Vec::new() };
    grid.rows.push(0);
    for (r, row) in rows.iter().enumerate() {
        let (j, k) = ((r % ny) as u32, (r / ny) as u32);
        for &(i0, i1) in row {
            grid.spans.push(Span { i0, i1, first: grid.ijk.len() as u32 });
            grid.ijk.extend((i0..=i1).map(|i| [i, j, k]));
        }
        grid.rows.push(grid.spans.len() as u32);
    }
    Some(grid)
}

/// The 26-neighbour steps between lumen voxels whose segment is inside: a bit per offset.
fn link(grid: &Grid, clear: &[f64], s: &Surfaces) -> Vec<u32> {
    let lens: [f64; 26] = OFFSETS.map(|o| DVec3::new(o[0] as f64, o[1] as f64, o[2] as f64).length() * grid.v);
    let n = grid.len();
    let one_way: Vec<u32> = (0..n as u32)
        .into_par_iter()
        .map(|a| {
            let mut m = 0u32;
            for (d, len) in lens.iter().enumerate() {
                let b = grid.neighbour(a, d);
                if b != NONE && (clear[a as usize] + clear[b as usize] > len * (1.0 + 1e-9) || !s.blocked(grid.centre(a), grid.centre(b))) {
                    m |= 1 << d;
                }
            }
            m
        })
        .collect();
    // Both ways or neither.
    (0..n as u32)
        .into_par_iter()
        .map(|a| {
            let mut m = one_way[a as usize];
            for d in 0..26 {
                if m & (1 << d) != 0 && one_way[grid.neighbour(a, d) as usize] & (1 << (25 - d)) == 0 {
                    m &= !(1 << d);
                }
            }
            m
        })
        .collect()
}

/// The linked neighbours of voxel `a` and the step lengths to them.
fn steps<'g>(grid: &'g Grid, links: &'g [u32], a: u32) -> impl Iterator<Item = (u32, f64)> + 'g {
    let m = links[a as usize];
    (0..26).filter(move |d| m & (1 << d) != 0).map(move |d| {
        let o = OFFSETS[d];
        (grid.neighbour(a, d), DVec3::new(o[0] as f64, o[1] as f64, o[2] as f64).length() * grid.v)
    })
}

/// The voxels linked to `root`.
fn component(grid: &Grid, links: &[u32], root: u32) -> Vec<bool> {
    let mut inside = vec![false; grid.len()];
    inside[root as usize] = true;
    let mut stack = vec![root];
    while let Some(a) = stack.pop() {
        for (b, _) in steps(grid, links, a) {
            if !inside[b as usize] {
                inside[b as usize] = true;
                stack.push(b);
            }
        }
    }
    inside
}

/// The voxel to start or end at for `end`: of the voxels near it (within its radius, at least
/// two voxels; further if there are none), the nearest among those at least half as far from
/// the wall as the farthest, so it is not a stray voxel by the wall.
fn anchor(grid: &Grid, wall: &[f64], end: &LumenEnd, allowed: impl Fn(u32) -> bool) -> Option<u32> {
    for reach in [1.0, 2.0, 4.0] {
        let r = (end.radius.max(0.0) * reach).max(2.0 * grid.v * reach);
        let mut near = Vec::new();
        grid.ball(end.at, r, |a| {
            if allowed(a) {
                near.push(a);
            }
        });
        let Some(deepest) = near.iter().map(|&a| wall[a as usize]).reduce(f64::max) else { continue };
        let best = near.into_iter().filter(|&a| wall[a as usize] >= 0.5 * deepest).min_by(|&a, &b| grid.centre(a).distance(end.at).total_cmp(&grid.centre(b).distance(end.at)).then(a.cmp(&b)));
        if best.is_some() {
            return best;
        }
    }
    None
}

/// Where a path from `from` (inside) ends at the opening `end`: its centre, or, where the
/// straight way there meets the cap first (a cap not quite flat), that point if it is near it.
fn opening_end(s: &Surfaces, from: DVec3, end: &LumenEnd) -> Option<DVec3> {
    let d = end.at - from;
    let len = d.length();
    if len == 0.0 {
        return Some(end.at);
    }
    match s.closed_bvh.raycast_range(s.closed, &Ray::new(from, d), len * 1e-7, len * (1.0 - 1e-7), |_| true) {
        None => Some(end.at),
        Some(h) => (h.point.distance(end.at) <= 0.25 * end.radius).then_some(h.point),
    }
}

/// Mark the voxels within `r` of voxel `a`.
fn cover(grid: &Grid, a: u32, r: f64, covered: &mut [bool]) {
    grid.ball(grid.centre(a), r, |b| covered[b as usize] = true);
}

/// How many voxels of `path` to keep so that its tip, where it only runs from the middle of a
/// vessel out to its wall, is cut: back to the innermost point whose inscribed ball, grown by
/// √2, still reaches the tip.
fn trim_tip(grid: &Grid, wall: &[f64], path: &[u32]) -> usize {
    let n = path.len();
    let tip = grid.centre(path[n - 1]);
    let reach = 4.0 * path.iter().map(|&a| wall[a as usize]).fold(0.0, f64::max);
    let (mut keep, mut arc) = (n, 0.0);
    for j in (0..n - 1).rev() {
        arc += grid.centre(path[j]).distance(grid.centre(path[j + 1]));
        if arc > reach {
            break;
        }
        if grid.centre(path[j]).distance(tip) <= std::f64::consts::SQRT_2 * wall[path[j] as usize] {
            keep = j + 1;
        }
    }
    keep
}

/// Where a voxel lies on the tree: a kept voxel of a path, or one cut from its start (`head`)
/// or tip (`tail`), by index.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Part {
    Node,
    Head,
    Tail,
}

/// A path traced through the voxels.
struct Path {
    nodes: Vec<u32>,
    /// Voxels cut from its start (from the one next to `nodes[0]` outward) and its tip.
    head: Vec<u32>,
    tail: Vec<u32>,
    /// The path and node its first node lies on.
    parent: Option<(usize, usize)>,
    /// Arc length at each node, and from the root to its first node, along the tree.
    arc: Vec<f64>,
    offset: f64,
    start_at: Option<DVec3>,
    tip_at: Option<DVec3>,
    /// The tip joins this path's node (a loop around a fenestration).
    join: Option<(usize, usize)>,
}

/// Arc length at each voxel of a path.
fn arcs(grid: &Grid, nodes: &[u32]) -> Vec<f64> {
    std::iter::once(0.0)
        .chain(nodes.windows(2).scan(0.0, |s, w| {
            *s += grid.centre(w[0]).distance(grid.centre(w[1]));
            Some(*s)
        }))
        .collect()
}

/// Distance along the tree (paths and their first-node attachments, not the loops) between
/// node `i` of path `p` and node `j` of path `q`.
fn tree_distance(paths: &[Path], (p, i): (usize, usize), (q, j): (usize, usize)) -> f64 {
    let at = |p: usize, i: usize| paths[p].offset + paths[p].arc[i];
    let mut up = vec![(p, i)];
    let mut c = p;
    while let Some(parent) = paths[c].parent {
        up.push(parent);
        c = parent.0;
    }
    let (mut c, mut cj) = (q, j);
    loop {
        if let Some(&(_, ci)) = up.iter().find(|u| u.0 == c) {
            return at(p, i) + at(q, j) - 2.0 * at(c, ci.min(cj));
        }
        let Some((pp, k)) = paths[c].parent else { return f64::INFINITY };
        (c, cj) = (pp, k);
    }
}

/// The voxels along the tree (kept paths and their first-node attachments) from node `a` to
/// node `b`, each a (path, node index).
fn tree_path(paths: &[Path], a: (usize, usize), b: (usize, usize)) -> Vec<u32> {
    let up = |mut x: (usize, usize)| {
        let mut chain = vec![x];
        while let Some(parent) = paths[x.0].parent {
            chain.push(parent);
            x = parent;
        }
        chain
    };
    let (ua, ub) = (up(a), up(b));
    let Some((ia, ib)) = ua.iter().enumerate().find_map(|(k, x)| ub.iter().position(|y| y.0 == x.0).map(|l| (k, l))) else { return Vec::new() };
    let (c, ca, cb) = (ua[ia].0, ua[ia].1, ub[ib].1);
    let mut out = Vec::new();
    // Up from `a` (each path's first node is its parent's node, which follows), along the
    // common path, then down to `b`.
    for &(q, i) in &ua[..ia] {
        out.extend((1..=i).rev().map(|k| paths[q].nodes[k]));
    }
    if ca <= cb {
        out.extend((ca..=cb).map(|k| paths[c].nodes[k]));
    } else {
        out.extend((cb..=ca).rev().map(|k| paths[c].nodes[k]));
    }
    for &(q, i) in ub[..ib].iter().rev() {
        out.extend((1..=i).map(|k| paths[q].nodes[k]));
    }
    out
}

/// Reusable state of the local searches for loops.
struct Local {
    cost: Vec<f64>,
    arc: Vec<f64>,
    prev: Vec<u32>,
    stamp: Vec<u32>,
    generation: u32,
}

/// Where a search from a tip meets a traced path: a kept one (voxel on the tree), or one put
/// aside (path, node index).
#[derive(Clone, Copy, Debug)]
enum Hit {
    Tree(u32, u32, Part),
    Aside(u32, u32),
}

enum Verdict {
    Accept,
    Skip,
    Stop,
}

struct Teasar<'a> {
    grid: &'a Grid,
    links: &'a [u32],
    wall: &'a [f64],
    s: &'a Surfaces<'a>,
    /// Step weight of each voxel, rising steeply toward the wall.
    weight: Vec<f64>,
    covered: Vec<bool>,
    on_tree: Vec<(u32, u32, Part)>,
    paths: Vec<Path>,
    /// Automatic paths not kept (yet): a later tip may close a loop through one, which then
    /// joins the tree.
    aside: Vec<Path>,
    on_aside: Vec<(u32, u32)>,
    loops: usize,
}

impl Teasar<'_> {
    fn step_cost(&self, a: u32, b: u32, len: f64) -> f64 {
        0.5 * len * (self.weight[a as usize] + self.weight[b as usize])
    }

    /// The voxels from the node a voxel on the tree belongs to out to that voxel (just the
    /// node for a kept voxel), with the node.
    fn reach_out(&self, (p, idx, part): (u32, u32, Part)) -> (Vec<u32>, (usize, usize)) {
        let path = &self.paths[p as usize];
        let (p, idx) = (p as usize, idx as usize);
        match part {
            Part::Node => (vec![path.nodes[idx]], (p, idx)),
            Part::Head => (std::iter::once(path.nodes[0]).chain(path.head[..=idx].iter().copied()).collect(), (p, 0)),
            Part::Tail => (std::iter::once(*path.nodes.last().unwrap()).chain(path.tail[..=idx].iter().copied()).collect(), (p, path.nodes.len() - 1)),
        }
    }

    fn mark(&mut self, p: usize) -> Vec<u32> {
        let path = &self.paths[p];
        let mut fresh = Vec::new();
        let all = path.nodes.iter().enumerate().map(|(i, &a)| (a, i, Part::Node)).chain(path.head.iter().enumerate().map(|(i, &a)| (a, i, Part::Head))).chain(path.tail.iter().enumerate().map(|(i, &a)| (a, i, Part::Tail)));
        let all: Vec<_> = all.collect();
        for (a, i, part) in all {
            if self.on_tree[a as usize].0 == NONE {
                self.on_tree[a as usize] = (p as u32, i as u32, part);
                fresh.push(a);
            }
        }
        fresh
    }

    /// Put the last path aside (its voxels in `fresh` leave the tree).
    fn put_aside(&mut self, fresh: Vec<u32>) {
        for a in fresh {
            self.on_tree[a as usize] = (NONE, 0, Part::Node);
        }
        let path = self.paths.pop().expect("a path");
        let g = self.aside.len() as u32;
        for (i, &a) in path.nodes.iter().enumerate() {
            if self.on_aside[a as usize].0 == NONE && self.on_tree[a as usize].0 == NONE {
                self.on_aside[a as usize] = (g, i as u32);
            }
        }
        self.aside.push(path);
    }

    /// The tree node a hit leads to, and the distance along the tree to it from there.
    fn landing(&self, hit: Hit) -> Option<((usize, usize), f64)> {
        match hit {
            Hit::Tree(p, idx, part) => Some((self.reach_out((p, idx, part)).1, 0.0)),
            Hit::Aside(g, idx) => {
                let path = &self.aside[g as usize];
                ((idx as usize) < path.nodes.len()).then(|| (path.parent.expect("a branch"), path.arc[idx as usize]))
            }
        }
    }

    /// From the tip of path `own`, through the lumen within `reach` mm (not through traced
    /// paths), the ways to traced paths far from the tip along the tree, cheapest first, to
    /// `judge` (with the voxels from the tip to the hit), until it accepts one or stops.
    fn search(&self, local: &mut Local, own: usize, reach: f64, mut judge: impl FnMut(&[u32], Hit) -> Verdict) -> Option<(Vec<u32>, Hit)> {
        let tip = *self.paths[own].nodes.last()?;
        let tip_node = (own, self.paths[own].nodes.len() - 1);
        local.generation += 1;
        let g = local.generation;
        let mut heap = BinaryHeap::new();
        let set = |local: &mut Local, a: u32, c: f64, arc: f64, prev: u32| {
            local.stamp[a as usize] = g;
            local.cost[a as usize] = c;
            local.arc[a as usize] = arc;
            local.prev[a as usize] = prev;
        };
        set(local, tip, 0.0, 0.0, NONE);
        heap.push(Item(0.0, tip));
        let r_tip = self.wall[tip as usize];
        while let Some(Item(c, a)) = heap.pop() {
            if c > local.cost[a as usize] {
                continue;
            }
            if a != tip {
                let t = self.on_tree[a as usize];
                let hit = if t.0 != NONE {
                    (t.0 as usize != own).then_some(Hit::Tree(t.0, t.1, t.2))
                } else if self.on_aside[a as usize].0 != NONE {
                    Some(Hit::Aside(self.on_aside[a as usize].0, self.on_aside[a as usize].1))
                } else {
                    None
                };
                if t.0 != NONE || hit.is_some() {
                    if let Some(((node, extra), hit)) = hit.and_then(|h| self.landing(h).map(|l| (l, h))) {
                        if tree_distance(&self.paths, tip_node, node) + extra >= 2.0 * local.arc[a as usize] + 2.0 * r_tip {
                            let mut way = vec![a];
                            while let Some(p) = way.last().map(|&l| local.prev[l as usize]).filter(|&p| p != NONE) {
                                way.push(p);
                            }
                            way.reverse();
                            match judge(&way, hit) {
                                Verdict::Accept => return Some((way, hit)),
                                Verdict::Stop => return None,
                                Verdict::Skip => {}
                            }
                        }
                    }
                    // Not through traced paths.
                    continue;
                }
            }
            for (b, len) in steps(self.grid, self.links, a) {
                let arc = local.arc[a as usize] + len;
                if arc > reach {
                    continue;
                }
                let nc = c + self.step_cost(a, b, len);
                if local.stamp[b as usize] != g || nc < local.cost[b as usize] {
                    set(local, b, nc, arc, a);
                    heap.push(Item(nc, b));
                }
            }
        }
        None
    }

    /// The channel a loop from path `own`'s tip along `way` to `hit` would add: its voxels from
    /// `own`'s first node to the tree node it lands on, and that node.
    fn channel(&self, own: usize, way: &[u32], hit: Hit) -> Option<(Vec<u32>, (usize, usize))> {
        let mut c: Vec<u32> = self.paths[own].nodes.iter().chain(&way[1..]).copied().collect();
        let node = match hit {
            Hit::Tree(p, idx, part) => {
                let (lead, node) = self.reach_out((p, idx, part));
                c.extend(lead.iter().rev().skip(1));
                node
            }
            Hit::Aside(g, idx) => {
                let path = &self.aside[g as usize];
                c.extend(path.nodes[..idx as usize].iter().rev());
                path.parent?
            }
        };
        Some((c, node))
    }

    /// Whether a channel (voxels from tree node `from` to tree node `to`) is a second way
    /// round tissue, as around a fenestration's bridge, big enough to count: between the
    /// junctions' spheres it is no narrower than a quarter of the vessel there and at least
    /// as long as it is wide, and a chord across the loop it makes with the tree meets the
    /// wall. (A flat vessel's cross-section gives ways round with nothing in between; a
    /// window in a septum, short or narrow ones.)
    fn round_tissue(&self, channel: &[u32], from: (usize, usize), to: (usize, usize), v: f64) -> bool {
        let g = self.grid;
        let (ja, jb) = (channel[0], *channel.last().unwrap());
        let (ra, rb) = (self.wall[ja as usize], self.wall[jb as usize]);
        let is_inner = |a: u32| g.centre(a).distance(g.centre(ja)) > ra && g.centre(a).distance(g.centre(jb)) > rb;
        let mut inner: Vec<f64> = channel.iter().filter(|&&a| is_inner(a)).map(|&a| self.wall[a as usize]).collect();
        if inner.is_empty() {
            return false;
        }
        inner.sort_by(f64::total_cmp);
        let inner_len: f64 = channel.windows(2).filter(|w| is_inner(w[0]) && is_inner(w[1])).map(|w| g.centre(w[0]).distance(g.centre(w[1]))).sum();
        if inner[0] < (0.25 * ra.min(rb)).max(1.5 * v) || inner_len < 2.0 * inner[inner.len() / 2] {
            return false;
        }
        let back = tree_path(&self.paths, from, to);
        if back.len() < 2 {
            return false;
        }
        let at = |voxels: &[u32], f: f64| {
            let arc = arcs(g, voxels);
            let want = f * arc[arc.len() - 1];
            let k = arc.partition_point(|&x| x < want).min(voxels.len() - 1);
            g.centre(voxels[k])
        };
        [0.25, 0.5, 0.75].iter().any(|&f| self.s.blocked(at(channel, f), at(&back, f)))
    }

    /// Close a loop from the last path's tip if one goes round tissue (see `round_tissue`):
    /// the path then runs on to the tree, and a path put aside that it meets joins the tree.
    fn close_loop(&mut self, local: &mut Local, v: f64) -> bool {
        let own = self.paths.len() - 1;
        let tip = *self.paths[own].nodes.last().unwrap();
        let reach = 8.0 * self.wall[tip as usize] + 4.0 * v;
        let from = self.paths[own].parent.expect("a branch");
        let mut tries = 0;
        let found = self.search(local, own, reach, |way, hit| {
            tries += 1;
            if tries > 8 {
                return Verdict::Stop;
            }
            match self.channel(own, way, hit) {
                Some((c, to)) if self.round_tissue(&c, from, to, v) => Verdict::Accept,
                _ => Verdict::Skip,
            }
        });
        let Some((way, hit)) = found else { return false };
        // Its cut tip leaves the tree: it runs on instead.
        for a in std::mem::take(&mut self.paths[own].tail) {
            if self.on_tree[a as usize].0 == own as u32 && self.on_tree[a as usize].2 == Part::Tail {
                self.on_tree[a as usize] = (NONE, 0, Part::Node);
            }
        }
        let mut extra: Vec<u32> = way[1..].to_vec();
        let join = match hit {
            Hit::Tree(p, idx, part) => {
                let (lead, node) = self.reach_out((p, idx, part));
                extra.extend(lead.iter().rev().skip(1));
                node
            }
            Hit::Aside(g, idx) => {
                // The path put aside joins the tree, up to where this one meets it.
                let mut other = std::mem::replace(&mut self.aside[g as usize], Path { nodes: Vec::new(), head: Vec::new(), tail: Vec::new(), parent: None, arc: Vec::new(), offset: 0.0, start_at: None, tip_at: None, join: None });
                other.nodes.truncate(idx as usize + 1);
                other.tail.clear();
                other.arc.truncate(idx as usize + 1);
                self.paths.push(other);
                let q = self.paths.len() - 1;
                for &a in &self.paths[q].nodes {
                    cover(self.grid, a, COVER * self.wall[a as usize], &mut self.covered);
                }
                self.mark(q);
                (q, idx as usize)
            }
        };
        for &a in &extra {
            cover(self.grid, a, COVER * self.wall[a as usize], &mut self.covered);
        }
        let path = &mut self.paths[own];
        path.nodes.extend(extra);
        path.join = Some(join);
        path.arc = arcs(self.grid, &path.nodes);
        self.mark(own);
        self.loops += 1;
        true
    }

    /// How much of the last path to keep (nodes) if it is a vessel of its own (a blind side
    /// branch): at least a fifth as wide as the vessel it leaves (not a spike of the wall),
    /// tube-shaped beyond that vessel's sphere (not the corner of a flat vessel's
    /// cross-section), and leading away from the tree: its tip, cut back where it narrows
    /// into a crevice, is at least `prune` of that vessel's radii from the rest of the tree,
    /// and `prune` of its own beyond that vessel's sphere (a corner running alongside its
    /// vessel stays near it).
    fn side_branch(&self, prune: f64, v: f64) -> Option<usize> {
        let g = self.grid;
        let own = self.paths.len() - 1;
        let nodes = &self.paths[own].nodes;
        let j = g.centre(nodes[0]);
        let rj = self.wall[nodes[0] as usize];
        let first = nodes.iter().position(|&a| g.centre(a).distance(j) > rj)?;
        let out = &nodes[first..];
        if out.len() < 3 {
            return None;
        }
        let mut radii: Vec<f64> = out.iter().map(|&a| self.wall[a as usize]).collect();
        radii.sort_by(f64::total_cmp);
        let r = radii[radii.len() / 2];
        if r < (0.2 * rj).max(2.0 * v) {
            return None;
        }
        let keep = nodes.len() - nodes.iter().rev().take_while(|&&a| self.wall[a as usize] < 0.5 * r).count();
        if keep <= first + 1 {
            return None;
        }
        let tip = g.centre(nodes[keep - 1]);
        let away = self.paths[..own].iter().flat_map(|p| p.nodes.iter()).map(|&a| g.centre(a).distance(tip)).fold(f64::INFINITY, f64::min);
        if away < (prune * rj).max(rj + prune * r) {
            return None;
        }
        let m = out.len() / 2;
        let t = (g.centre(out[m + 1]) - g.centre(out[m - 1])).normalize_or_zero();
        tubular(self.s, g.centre(out[m]), t, r).then_some(keep)
    }

    /// Keep the last path's first `keep` nodes (its voxels in `fresh` were just marked).
    fn cut_tip(&mut self, fresh: Vec<u32>, keep: usize) {
        for a in fresh {
            self.on_tree[a as usize] = (NONE, 0, Part::Node);
        }
        let p = self.paths.len() - 1;
        let path = &mut self.paths[p];
        let mut cut = path.nodes.split_off(keep);
        cut.append(&mut path.tail);
        path.tail = cut;
        path.arc.truncate(keep);
        self.mark(p);
    }
}

/// How far the voxels within a multiple of the wall's distance from a path are covered by it.
const COVER: f64 = 1.75;

/// Whether the lumen's cross-section at `p` square to `t` is tube-shaped: its widest width
/// across `p` (of four directions) is at most three times its narrowest.
fn tubular(s: &Surfaces, p: DVec3, t: DVec3, r: f64) -> bool {
    let u = t.any_orthonormal_vector();
    let w = t.cross(u);
    let far = 40.0 * r;
    let reach = |d: DVec3| s.closed_bvh.raycast_range(s.closed, &Ray::new(p, d), 0.0, far, |_| true).map_or(far, |h| h.t);
    let widths: Vec<f64> = [0.0, 0.25, 0.5, 0.75].iter().map(|&k| {
        let a = PI * k;
        let d = u * a.cos() + w * a.sin();
        reach(d) + reach(-d)
    }).collect();
    let (lo, hi) = widths.iter().fold((f64::INFINITY, 0.0f64), |(l, h), &x| (l.min(x), h.max(x)));
    hi <= 3.0 * lo
}

/// Shortest paths from `root` under the step weights: each voxel's previous one and the
/// path's length in mm. `None` once `going` returns false.
fn shortest_paths(grid: &Grid, links: &[u32], root: u32, weight: &[f64], going: &dyn Fn(f32) -> bool, count: usize) -> Option<(Vec<u32>, Vec<f64>)> {
    let n = grid.len();
    let mut cost = vec![f64::INFINITY; n];
    let mut prev = vec![NONE; n];
    let mut len = vec![0.0; n];
    let mut heap = BinaryHeap::new();
    cost[root as usize] = 0.0;
    heap.push(Item(0.0, root));
    let mut settled = 0usize;
    while let Some(Item(c, a)) = heap.pop() {
        if c > cost[a as usize] {
            continue;
        }
        settled += 1;
        if settled.is_multiple_of(65536) && !going(settled as f32 / count.max(1) as f32) {
            return None;
        }
        for (b, l) in steps(grid, links, a) {
            let nc = c + 0.5 * l * (weight[a as usize] + weight[b as usize]);
            if nc < cost[b as usize] {
                cost[b as usize] = nc;
                prev[b as usize] = a;
                len[b as usize] = len[a as usize] + l;
                heap.push(Item(nc, b));
            }
        }
    }
    Some((prev, len))
}

/// Gaussian smoothing over arc length (σ given per point), the window narrowing
/// symmetrically toward the ends, which stay.
fn gauss(pts: &[DVec3], sigma: impl Fn(usize) -> f64 + Sync) -> Vec<DVec3> {
    let n = pts.len();
    let mut arc = vec![0.0; n];
    for i in 1..n {
        arc[i] = arc[i - 1] + pts[i].distance(pts[i - 1]);
    }
    let total = arc[n - 1];
    (0..n)
        .into_par_iter()
        .map(|i| {
            let s = sigma(i);
            let half = (3.0 * s).min(arc[i]).min(total - arc[i]);
            if i == 0 || i + 1 == n || half <= 0.0 || s <= 0.0 {
                return pts[i];
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
                let w = (-0.5 * d * d / (s * s)).exp();
                acc += pts[j] * w;
                weight += w;
            }
            acc / weight
        })
        .collect()
}

/// `b`, or the point toward it from `a` no further than `r` from `a`.
fn within(a: DVec3, b: DVec3, r: f64) -> DVec3 {
    let d = b - a;
    let l = d.length();
    if l <= r.max(0.0) {
        b
    } else {
        a + d * (r.max(0.0) / l)
    }
}

/// `p` (inside the lumen) moved within the plane square to `t` toward where the wall is
/// farthest, in steps that stay within the largest ball about it that the surface does not
/// enter, so it stays inside.
fn centre(s: &Surfaces, mut p: DVec3, t: DVec3, v: f64) -> DVec3 {
    let mut d = s.wall(p);
    let mut c = s.clearance(p, d);
    let mut step = v;
    for _ in 0..40 {
        let Some(q) = s.wall_point(p) else { break };
        let away = p - q;
        let g = (away - t * away.dot(t)).normalize_or_zero();
        if g == DVec3::ZERO {
            break;
        }
        if step < 0.9 * c {
            let trial = p + g * step;
            let dt = s.wall(trial);
            if dt > d {
                (p, d) = (trial, dt);
                c = s.clearance(p, d);
                continue;
            }
        }
        step *= 0.5;
        if step < 0.01 * v {
            break;
        }
    }
    p
}

/// A chain's points from its voxel path (`raw`, each `raw_clear` from the surface; the two
/// ends stay): smoothed, centred in their cross-sections, every segment proven inside, and
/// thinned to `spacing` where the straight segments between the points kept are proven
/// inside too and stay near the curve. With each point's distance to the wall.
fn refine(s: &Surfaces, v: f64, spacing: f64, raw: &[DVec3], raw_wall: &[f64], raw_clear: &[f64]) -> Vec<(DVec3, f64)> {
    let n = raw.len();
    if n < 3 {
        return raw.iter().zip(raw_wall).map(|(&p, &w)| (p, w)).collect();
    }
    let ends = |i: usize| i == 0 || i + 1 == n;
    // The voxel staircase out, over about a quarter of the radius.
    let smooth = gauss(raw, |i| (1.5 * v).max(0.25 * raw_wall[i]));
    let p: Vec<DVec3> = (0..n).map(|i| if ends(i) { raw[i] } else { within(raw[i], smooth[i], 0.9 * raw_clear[i]) }).collect();
    let centred: Vec<DVec3> = (0..n).into_par_iter().map(|i| if ends(i) { p[i] } else { centre(s, p[i], (p[i + 1] - p[i - 1]).normalize_or_zero(), v) }).collect();
    let clear_at = |q: DVec3| s.clearance(q, s.wall(q));
    let c_clear: Vec<f64> = centred.par_iter().map(|&q| clear_at(q)).collect();
    let calm = gauss(&centred, |_| 1.5 * v);
    let mut q: Vec<DVec3> = (0..n).map(|i| if ends(i) { centred[i] } else { within(centred[i], calm[i], 0.9 * c_clear[i]) }).collect();
    let mut cq: Vec<f64> = q.par_iter().enumerate().map(|(i, &x)| if ends(i) { raw_clear[i] } else { clear_at(x) }).collect();
    // Every segment proven inside, or its moved points back on the voxel path.
    let mut moved: Vec<bool> = (0..n).map(|i| !ends(i)).collect();
    loop {
        let bad: Vec<usize> = (0..n - 1).into_par_iter().filter(|&i| !s.proven(q[i], cq[i], q[i + 1], cq[i + 1])).collect();
        let mut changed = false;
        for i in bad {
            for j in [i, i + 1] {
                if moved[j] {
                    (q[j], cq[j], moved[j]) = (raw[j], raw_clear[j], false);
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    let wall: Vec<f64> = q.par_iter().map(|&x| s.wall(x)).collect();
    // Thinned: from each point kept, the farthest within the spacing whose chord stays near
    // the points between and is proven inside.
    let mut arc = vec![0.0; n];
    for i in 1..n {
        arc[i] = arc[i - 1] + q[i].distance(q[i - 1]);
    }
    let mut keep = vec![0];
    let mut a = 0;
    while a + 1 < n {
        let mut best = a + 1;
        for b in a + 2..n {
            if arc[b] - arc[a] > spacing {
                break;
            }
            let chord = q[b] - q[a];
            let near = (a + 1..b).all(|i| {
                let t = ((q[i] - q[a]).dot(chord) / chord.length_squared().max(1e-300)).clamp(0.0, 1.0);
                q[i].distance(q[a] + chord * t) <= (0.25 * v).max(0.05 * wall[i])
            });
            if !near || !s.proven(q[a], cq[a], q[b], cq[b]) {
                break;
            }
            best = b;
        }
        keep.push(best);
        a = best;
    }
    keep.into_iter().map(|i| (q[i], wall[i])).collect()
}

/// The centerline tree of the lumen of `mesh` from `input.inlet` to every outlet and side
/// branch (see the module). Its open ends are capped with `cache` (or a capping kept there is
/// used). `progress` gets the fraction done; `Ok(None)` once it returns false.
pub fn compute_lumen(mesh: &Mesh, cache: &CappedCache, input: &LumenInput, params: &LumenParams, progress: Option<&dyn Fn(f32) -> bool>) -> anyhow::Result<Option<LumenCenterline>> {
    let going = |x: f32| progress.is_none_or(|p| p(x));
    macro_rules! at {
        ($x:expr) => {
            if !going($x) {
                return Ok(None);
            }
        };
    }
    if mesh.triangle_count() == 0 {
        bail!("the model is empty");
    }
    // The closed surface: the mesh with its open ends capped (or, where they cannot be, as it
    // is: rows through its holes are then decided by the winding number).
    let capped = cache.get(mesh, &|x| going(0.15 * x));
    at!(0.15);
    let fallback;
    let (closed, closed_bvh) = match &capped {
        Some(c) => (&c.mesh, &c.bvh),
        None => {
            fallback = Bvh::build(mesh);
            (mesh, &fallback)
        }
    };
    let wall_tris: Vec<u32> = mesh.triangle_ids().filter(|&t| !input.cap_groups.contains(&mesh.tri_group(t))).collect();
    if wall_tris.is_empty() {
        bail!("the model has no wall besides its end caps");
    }
    let wall_bvh = if wall_tris.len() == mesh.triangle_count() { Bvh::build(mesh) } else { Bvh::build_subset(mesh, &wall_tris) };
    let s = Surfaces { closed, closed_bvh, wall: mesh, wall_bvh };
    at!(0.2);

    // Voxels: a sixth of the narrowest end's radius, coarser where the lumen would take more
    // than about eight million of them.
    let bounds = closed.bounds();
    let size = bounds.max - bounds.min;
    let r_ref = std::iter::once(&input.inlet).chain(&input.outlets).map(|e| e.radius).filter(|r| r.is_finite() && *r > 0.0).fold(f64::INFINITY, f64::min);
    let r_ref = if r_ref.is_finite() { r_ref } else { 0.05 * size.min_element() };
    let volume = mm_core::measure::volume(closed).abs().min(size.x * size.y * size.z);
    let mut v = if params.voxel > 0.0 { params.voxel } else { r_ref / 6.0 };
    v = v.max((volume / 8.0e6).cbrt()).max((size.y * size.z / 4.0e6).sqrt()).max(size.max_element() * 1e-4);
    let lo = bounds.min - DVec3::splat(v);
    let dims = [0, 1, 2].map(|a| ((size[a] + 2.0 * v) / v).ceil() as usize + 1);
    let winding = OnceLock::new();
    let Some(grid) = voxelize(closed, closed_bvh, lo, v, dims, &winding, &|x| going(0.2 + 0.1 * x)) else { return Ok(None) };
    if grid.len() == 0 {
        bail!("found no inside to the model (is it a closed surface, or one with open vessel ends?)");
    }
    let n = grid.len();
    let wall: Vec<f64> = (0..n as u32).into_par_iter().map(|a| s.wall(grid.centre(a))).collect();
    at!(0.4);
    let clear: Vec<f64> = (0..n as u32).into_par_iter().map(|a| s.clearance(grid.centre(a), wall[a as usize])).collect();
    at!(0.45);
    let links = link(&grid, &clear, &s);
    at!(0.5);

    let root = anchor(&grid, &wall, &input.inlet, |_| true).ok_or_else(|| anyhow!("the inlet is not at the model's inside"))?;
    let inside = component(&grid, &links, root);
    let count = inside.iter().filter(|&&x| x).count();
    let deepest = (0..n).filter(|&a| inside[a]).map(|a| wall[a]).fold(0.0, f64::max);
    let floor = 0.25 * v;
    let weight: Vec<f64> = wall.par_iter().map(|&d| (deepest / d.max(floor)).powi(4)).collect();
    let Some((prev, len)) = shortest_paths(&grid, &links, root, &weight, &|x| going(0.5 + 0.15 * x), count) else { return Ok(None) };
    at!(0.65);

    // Targets: the outlets (farthest first), then the farthest voxels not yet covered.
    let mut unreached = Vec::new();
    let mut wanted: Vec<(u32, usize)> = Vec::new();
    for (o, end) in input.outlets.iter().enumerate() {
        match anchor(&grid, &wall, end, |a| inside[a as usize]) {
            Some(a) if a != root => wanted.push((a, o)),
            Some(_) => {}
            None => unreached.push(o),
        }
    }
    wanted.sort_by(|a, b| len[b.0 as usize].total_cmp(&len[a.0 as usize]).then(a.0.cmp(&b.0)));
    wanted.dedup_by_key(|w| w.0);
    let mut wanted = wanted.into_iter();
    let mut order: Vec<u32> = (0..n as u32).filter(|&a| inside[a as usize]).collect();
    order.par_sort_unstable_by(|&a, &b| len[b as usize].total_cmp(&len[a as usize]).then(a.cmp(&b)));
    let mut t = Teasar { grid: &grid, links: &links, wall: &wall, s: &s, weight, covered: vec![false; n], on_tree: vec![(NONE, 0, Part::Node); n], paths: Vec::new(), aside: Vec::new(), on_aside: vec![(NONE, 0); n], loops: 0 };
    let mut local = Local { cost: vec![0.0; n], arc: vec![0.0; n], prev: vec![NONE; n], stamp: vec![0; n], generation: 0 };
    let mut next = 0;
    let blind = params.blind_branches || input.outlets.is_empty();
    for iteration in 0usize.. {
        let (target, outlet) = match wanted.next() {
            Some((a, o)) => (a, Some(o)),
            None => {
                while next < order.len() && t.covered[order[next] as usize] {
                    next += 1;
                }
                if next == order.len() {
                    break;
                }
                (order[next], None)
            }
        };
        if iteration % 16 == 15 {
            at!(0.65 + 0.2 * next as f32 / order.len().max(1) as f32);
        }
        // Back along the shortest paths to the tree (or the root).
        let mut raw = vec![target];
        let mut a = target;
        while t.on_tree[a as usize].0 == NONE && prev[a as usize] != NONE {
            a = prev[a as usize];
            raw.push(a);
        }
        raw.reverse();
        for &a in &raw {
            cover(&grid, a, COVER * wall[a as usize], &mut t.covered);
        }
        t.covered[target as usize] = true;
        if raw.len() < 2 {
            continue;
        }
        let first = t.paths.is_empty();
        let (mut nodes, parent) = if first {
            (raw, None)
        } else {
            // From the tree's node (out along any voxels cut from its path) onward.
            let (lead, node) = t.reach_out(t.on_tree[raw[0] as usize]);
            (lead.into_iter().chain(raw[1..].iter().copied()).collect::<Vec<u32>>(), Some(node))
        };
        let mut tail = Vec::new();
        if outlet.is_none() {
            let keep = trim_tip(&grid, &wall, &nodes);
            tail = nodes.split_off(keep);
        }
        let mut head = Vec::new();
        if first && !input.inlet.opening {
            nodes.reverse();
            let keep = trim_tip(&grid, &wall, &nodes);
            head = nodes.split_off(keep);
            nodes.reverse();
        }
        if nodes.len() < 2 {
            if first {
                bail!("the lumen is too small for a centerline at this resolution ({v:.3} mm voxels)");
            }
            continue;
        }
        let start_at = if first && input.inlet.opening { opening_end(&s, grid.centre(nodes[0]), &input.inlet) } else { None };
        let tip_at = outlet.and_then(|o| opening_end(&s, grid.centre(*nodes.last().unwrap()), &input.outlets[o]));
        let arc = arcs(&grid, &nodes);
        let offset = parent.map_or(0.0, |(q, k): (usize, usize)| t.paths[q].offset + t.paths[q].arc[k]);
        let path = Path { nodes, head, tail, parent, arc, offset, start_at, tip_at, join: None };
        let p = t.paths.len();
        t.paths.push(path);
        let fresh = t.mark(p);
        if outlet.is_some() || first {
            continue;
        }
        // A tip meeting the tree again round tissue closes a loop (a fenestration); a blind
        // side branch is kept when asked for (or the vessel has no other ends). The rest
        // (bumps of the wall, corners of flat cross-sections, a channel's first half) are put
        // aside, for a later tip to close a loop through.
        if t.close_loop(&mut local, v) {
            continue;
        }
        if let Some(keep) = if blind { t.side_branch(params.prune_factor, v) } else { None } {
            t.cut_tip(fresh, keep);
            continue;
        }
        t.put_aside(fresh);
    }
    at!(0.85);
    if t.paths.is_empty() {
        bail!("found no centerline from the inlet");
    }
    let loops = t.loops;
    let paths = t.paths;

    // The graph of the paths: voxels (shared where paths meet) and the exact ends.
    let mut vertex_of: HashMap<u32, usize> = HashMap::new();
    let mut vpos: Vec<DVec3> = Vec::new();
    let mut vwall: Vec<f64> = Vec::new();
    let mut vclear: Vec<f64> = Vec::new();
    let mut adj: Vec<Vec<usize>> = Vec::new();
    let mut vertex = |a: u32, vpos: &mut Vec<DVec3>, vwall: &mut Vec<f64>, vclear: &mut Vec<f64>, adj: &mut Vec<Vec<usize>>| {
        *vertex_of.entry(a).or_insert_with(|| {
            vpos.push(grid.centre(a));
            vwall.push(wall[a as usize]);
            vclear.push(clear[a as usize]);
            adj.push(Vec::new());
            vpos.len() - 1
        })
    };
    let mut root_vertex = None;
    // The ends of paths at openings, on their caps.
    let mut ends_at: Vec<DVec3> = Vec::new();
    for path in &paths {
        let mut seq = Vec::new();
        let mut exact = |at: DVec3, vpos: &mut Vec<DVec3>, vwall: &mut Vec<f64>, vclear: &mut Vec<f64>, adj: &mut Vec<Vec<usize>>| {
            ends_at.push(at);
            vpos.push(at);
            vwall.push(s.wall(at));
            vclear.push(0.0);
            adj.push(Vec::new());
            vpos.len() - 1
        };
        if let Some(at) = path.start_at {
            seq.push(exact(at, &mut vpos, &mut vwall, &mut vclear, &mut adj));
        }
        for &a in &path.nodes {
            seq.push(vertex(a, &mut vpos, &mut vwall, &mut vclear, &mut adj));
        }
        if let Some(at) = path.tip_at {
            seq.push(exact(at, &mut vpos, &mut vwall, &mut vclear, &mut adj));
        }
        root_vertex.get_or_insert(seq[0]);
        for w in seq.windows(2) {
            if w[0] != w[1] && !adj[w[0]].contains(&w[1]) {
                adj[w[0]].push(w[1]);
                adj[w[1]].push(w[0]);
            }
        }
    }
    let root_vertex = root_vertex.unwrap();

    // Chains between junctions and ends, from the inlet outward.
    struct Chain {
        verts: Vec<usize>,
        parent: Option<usize>,
        joins: Option<usize>,
    }
    let breakpoint = |x: usize| x == root_vertex || adj[x].len() != 2;
    let mut reached_by: Vec<Option<Option<usize>>> = vec![None; vpos.len()];
    let mut walked: std::collections::HashSet<(usize, usize)> = Default::default();
    let mut chains: Vec<Chain> = Vec::new();
    let mut queue = std::collections::VecDeque::from([root_vertex]);
    reached_by[root_vertex] = Some(None);
    while let Some(u) = queue.pop_front() {
        for &w in &adj[u] {
            if walked.contains(&(u.min(w), u.max(w))) {
                continue;
            }
            walked.insert((u.min(w), u.max(w)));
            let mut verts = vec![u, w];
            let (mut before, mut cur) = (u, w);
            while !breakpoint(cur) {
                let Some(&next) = adj[cur].iter().find(|&&x| x != before) else { break };
                if walked.contains(&(cur.min(next), cur.max(next))) {
                    break;
                }
                walked.insert((cur.min(next), cur.max(next)));
                verts.push(next);
                (before, cur) = (cur, next);
            }
            let id = chains.len();
            let parent = reached_by[u].flatten();
            let joins = match reached_by[cur] {
                Some(by) => by,
                None => {
                    reached_by[cur] = Some(Some(id));
                    queue.push_back(cur);
                    None
                }
            };
            chains.push(Chain { verts, parent, joins });
        }
    }
    at!(0.9);

    // Each chain's points: smoothed, centred, proven inside, thinned.
    let spacing = if params.spacing > 0.0 { params.spacing } else { (0.2 * input.inlet.radius).max(v) };
    let refined: Vec<Vec<(DVec3, f64)>> = chains
        .par_iter()
        .map(|c| {
            let raw: Vec<DVec3> = c.verts.iter().map(|&x| vpos[x]).collect();
            let rw: Vec<f64> = c.verts.iter().map(|&x| vwall[x]).collect();
            let rc: Vec<f64> = c.verts.iter().map(|&x| vclear[x]).collect();
            refine(&s, v, spacing, &raw, &rw, &rc)
        })
        .collect();
    at!(0.97);

    let junction = |x: usize| adj[x].len() >= 3;
    let mut cl = Centerline::default();
    let mut node_of: HashMap<usize, usize> = HashMap::new();
    for (c, pts) in chains.iter().zip(&refined) {
        let m = pts.len();
        let mut ids = Vec::with_capacity(m);
        for (i, &(p, w)) in pts.iter().enumerate() {
            let shared = if i == 0 { Some(c.verts[0]) } else if i + 1 == m { c.verts.last().copied() } else { None };
            let mut add = || {
                cl.nodes.push(CenterlineNode { position: p, radius: w, band: 0 });
                cl.nodes.len() - 1
            };
            ids.push(match shared {
                Some(x) => *node_of.entry(x).or_insert_with(add),
                None => add(),
            });
        }
        for w in ids.windows(2) {
            cl.edges.push((w[0], w[1]));
        }
        let length = pts.windows(2).map(|w| w[0].0.distance(w[1].0)).sum();
        // Radii of the junctions' spheres, which span several branches, are left out.
        let from = usize::from(junction(c.verts[0]) && m > 2);
        let to = m - usize::from(junction(*c.verts.last().unwrap()) && m > 2 + from);
        let rs: Vec<f64> = pts[from..to].iter().map(|p| p.1).collect();
        cl.branches.push(Branch {
            nodes: ids,
            length,
            mean_radius: rs.iter().sum::<f64>() / rs.len().max(1) as f64,
            min_radius: rs.iter().copied().fold(f64::INFINITY, f64::min),
            max_radius: rs.iter().copied().fold(0.0, f64::max),
            parent: c.parent,
            joins: c.joins,
        });
    }
    let wn = mm_solid::WindingNumber::new(closed);
    let check = check_with(closed, closed_bvh, (s.wall, &s.wall_bvh), &wn, &cl, &ends_at);
    at!(1.0);
    Ok(Some(LumenCenterline { centerline: cl, check, unreached, loops, voxel: v }))
}

/// Test every point of `cl` and every segment between its neighbouring points against the
/// lumen of `mesh`, with its open ends capped by `cache` (see [`LumenCheck`]): a point is
/// inside where the surface's winding number about it is at least ½ (the centres of the
/// vessel's openings, `openings`, which lie on their caps, aside); a segment crosses the wall
/// where a ray along it meets the surface. `None` once `going` returns false.
#[cfg_attr(not(test), allow(dead_code))]
pub fn check(mesh: &Mesh, cache: &CappedCache, cl: &Centerline, openings: &[DVec3], going: &dyn Fn(f32) -> bool) -> Option<LumenCheck> {
    let capped = cache.get(mesh, going);
    if !going(0.0) {
        return None;
    }
    let wall_bvh = Bvh::build(mesh);
    let (closed, bvh) = match &capped {
        Some(c) => (&c.mesh, &c.bvh),
        None => (mesh, &wall_bvh),
    };
    let wn = mm_solid::WindingNumber::new(closed);
    Some(check_with(closed, bvh, (mesh, &wall_bvh), &wn, cl, openings))
}

fn check_with(closed: &Mesh, bvh: &Bvh, (wall, wall_bvh): (&Mesh, &Bvh), wn: &mm_solid::WindingNumber, cl: &Centerline, openings: &[DVec3]) -> LumenCheck {
    let used: Vec<usize> = {
        let mut u: Vec<usize> = cl.branches.iter().flat_map(|b| b.nodes.iter().copied()).collect();
        u.sort_unstable();
        u.dedup();
        u
    };
    let b = closed.bounds();
    let tolerance = 1e-9 * (1.0 + b.max.abs().max(b.min.abs()).max_element());
    let opening = |p: DVec3| openings.iter().any(|o| o.distance(p) <= tolerance);
    let tested: Vec<(DVec3, bool, f64)> = used
        .par_iter()
        .map(|&i| {
            let p = cl.nodes[i].position;
            if opening(p) {
                return (p, true, f64::INFINITY);
            }
            let inside = wn.eval(p).abs() >= 0.5;
            (p, inside, wall_bvh.nearest(wall, p, f64::INFINITY).map_or(0.0, |h| h.dist_sq.sqrt()))
        })
        .collect();
    let segments: Vec<(usize, usize)> = cl.branches.iter().flat_map(|b| b.nodes.windows(2).map(|w| (w[0], w[1]))).collect();
    let crossing: Vec<DVec3> = segments
        .par_iter()
        .filter_map(|&(a, b)| {
            let (p, q) = (cl.nodes[a].position, cl.nodes[b].position);
            segment_blocked(closed, bvh, p, q).then(|| 0.5 * (p + q))
        })
        .collect();
    LumenCheck {
        points: used.len(),
        segments: segments.len(),
        outside: tested.iter().filter(|t| !t.1).map(|t| t.0).collect(),
        crossing,
        min_clearance: tested.iter().filter(|t| t.1).map(|t| t.2).fold(f64::INFINITY, f64::min),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_core::{primitives, Plane};

    /// The planar ends of `mesh` as openings.
    fn openings(mesh: &Mesh) -> Vec<LumenEnd> {
        crate::tools::vessel::find_vessel_ends(mesh).iter().filter(|e| e.planar).map(|e| LumenEnd { at: e.center, radius: 0.5 * e.diameter, opening: true }).collect()
    }

    /// The centerline from the end nearest `inlet` to the others.
    fn run(mesh: &Mesh, inlet: DVec3) -> LumenCenterline {
        let mut ends = openings(mesh);
        let i = (0..ends.len()).min_by(|&a, &b| ends[a].at.distance(inlet).total_cmp(&ends[b].at.distance(inlet))).expect("an end");
        let inlet = ends.remove(i);
        compute_lumen(mesh, &CappedCache::default(), &LumenInput { inlet, outlets: ends, cap_groups: Vec::new() }, &LumenParams::default(), None).unwrap().unwrap()
    }

    /// The rings' centerline from the end nearest `inlet`, checked the same way.
    fn rings_checked(mesh: &Mesh, inlet: DVec3) -> LumenCheck {
        let lp = mm_core::topology::boundary_loops(mesh).into_iter().min_by(|a, b| a.centroid(mesh).distance(inlet).total_cmp(&b.centroid(mesh).distance(inlet))).unwrap();
        let r = lp.positions(mesh).iter().map(|p| p.distance(lp.centroid(mesh))).sum::<f64>() / lp.len() as f64;
        let cl = super::super::compute(mesh, &lp.vertices, r, &Default::default());
        let ends: Vec<DVec3> = openings(mesh).iter().map(|e| e.at).collect();
        check(mesh, &CappedCache::default(), &cl, &ends, &|_| true).unwrap()
    }

    /// Union of tubes along `paths` (radius `r` each) as one surface, opened square to the
    /// planes given (point, outward normal: the part beyond is cut off).
    fn vessel(paths: &[(Vec<DVec3>, f64)], cuts: &[(DVec3, DVec3)]) -> Mesh {
        let mut raw = Mesh::new();
        for (p, r) in paths {
            raw.append_mesh(&primitives::tube_along(p, &vec![*r; p.len()], 32), 0);
        }
        let params = mm_solid::MakeSolidParams::from_sliders(&raw, 0.5, 0.5);
        let mut m = mm_solid::make_solid(&raw, &params, None).unwrap();
        for &(at, n) in cuts {
            mm_edit::plane_cut(&mut m, Plane::from_point_normal(at, n), &mm_edit::PlaneCutParams { keep: mm_edit::KeepSide::Below, fill: mm_edit::CutFill::None, ..Default::default() }).unwrap();
        }
        m
    }

    fn line(a: DVec3, b: DVec3, steps: usize) -> Vec<DVec3> {
        (0..=steps).map(|i| a.lerp(b, i as f64 / steps as f64)).collect()
    }

    fn all_points(cl: &Centerline) -> Vec<DVec3> {
        cl.branches.iter().flat_map(|b| b.nodes.iter().map(|&i| cl.nodes[i].position)).collect()
    }

    #[test]
    fn a_straight_tube_has_its_axis_and_radius() {
        let path: Vec<DVec3> = (0..=40).map(|i| DVec3::new(0.0, 0.0, i as f64 * 0.5)).collect();
        let tube = primitives::tube_along(&path, &vec![1.0; path.len()], 32);
        let r = run(&tube, DVec3::ZERO);
        assert!(r.check.passed(), "{:?}", r.check);
        assert_eq!(r.centerline.branches.len(), 1, "{:?}", r.centerline.branches);
        let b = &r.centerline.branches[0];
        assert!((b.length - 20.0).abs() < 0.05, "length {}", b.length);
        // The 32-gon's inscribed circle.
        let inradius = (PI / 32.0).cos();
        for &i in &b.nodes {
            let n = r.centerline.nodes[i];
            assert!(n.position.truncate().length() < 0.02, "off the axis: {:?}", n.position);
            assert!((n.radius - inradius).abs() < 0.01, "radius {} at z = {:.2}", n.radius, n.position.z);
        }
    }

    #[test]
    fn a_fenestration_is_a_loop_inside_both_channels() {
        // A vessel (radius 1.5) along z whose middle splits around a bridge of tissue: the
        // second channel bows out to x = 5 and rejoins, leaving a hole about 18 mm long.
        let side: Vec<DVec3> = (0..=60).map(|i| DVec3::new(5.0 * (PI * i as f64 / 60.0).sin(), 0.0, 15.0 + 0.5 * i as f64)).collect();
        let m = vessel(&[(line(DVec3::ZERO, DVec3::Z * 60.0, 60), 1.5), (side, 1.5)], &[(DVec3::Z * 2.0, -DVec3::Z), (DVec3::Z * 58.0, DVec3::Z)]);
        let r = run(&m, DVec3::ZERO);
        assert!(r.check.passed(), "{:?}", r.check);
        assert_eq!(r.loops, 1, "branches {:?}", r.centerline.branches.iter().map(|b| (b.length, b.parent, b.joins)).collect::<Vec<_>>());
        let pts = all_points(&r.centerline);
        // Down the middle of the other channel too, and never in the bridge between them.
        assert!(pts.iter().any(|p| (p.z - 30.0).abs() < 1.0 && (p.x - 5.0).abs() < 0.3), "no point in the side channel");
        for p in &pts {
            let bridge = p.z > 22.0 && p.z < 38.0 && p.x > 1.6 && p.x < 5.0 * (PI * (p.z - 15.0) / 30.0).sin() - 1.6;
            assert!(!bridge, "a point in the bridge: {p:?}");
        }
        assert_eq!(r.centerline.branches.iter().filter(|b| b.joins.is_some()).count(), 1);
        println!("rings on the fenestration: {:?}", rings_checked(&m, DVec3::ZERO));
    }

    #[test]
    fn a_septum_thinner_than_a_voxel_is_not_crossed() {
        // A hairpin: two legs (radius 1.5) 0.1 mm apart up to z = 15, splaying out and joined
        // by a loop at the top. The way from one end to the other is all the way round.
        let mut path: Vec<DVec3> = Vec::new();
        let leg = |s: f64, z: f64| DVec3::new(s * (1.55 + 3.45 * ((z - 15.0).max(0.0) / 7.0).powi(2).min(1.0)), 0.0, z);
        path.extend((0..=44).map(|i| leg(-1.0, 0.5 * i as f64)));
        path.extend((1..48).map(|i| {
            let a = PI - PI * i as f64 / 48.0;
            DVec3::new(5.0 * a.cos(), 0.0, 22.0 + 5.0 * a.sin())
        }));
        path.extend((0..=44).rev().map(|i| leg(1.0, 0.5 * i as f64)));
        let tube = primitives::tube_along(&path, &vec![1.5; path.len()], 32);
        let r = run(&tube, DVec3::new(-1.55, 0.0, 0.0));
        assert!(r.voxel > 0.1, "the septum must be thinner than a voxel: {}", r.voxel);
        assert!(r.check.passed(), "{:?}", r.check);
        assert_eq!(r.centerline.branches.len(), 1, "{:?}", r.centerline.branches);
        assert!(all_points(&r.centerline).iter().any(|p| p.z > 26.5), "it did not go round");
        let along: f64 = path.windows(2).map(|w| w[0].distance(w[1])).sum();
        assert!((r.centerline.branches[0].length - along).abs() < 0.03 * along, "length {} of {along}", r.centerline.branches[0].length);
    }

    #[test]
    fn a_confluence_stays_inside() {
        // Torcula-like: four sinuses into a chamber, one (the straight sinus) narrower.
        let chamber = primitives::uv_sphere(4.5, 48, 24);
        let mut paths = vec![(line(DVec3::ZERO, DVec3::Z * 30.0, 30), 2.5), (line(DVec3::ZERO, DVec3::X * 30.0, 30), 2.5), (line(DVec3::ZERO, -DVec3::X * 30.0, 30), 2.2), (line(DVec3::ZERO, DVec3::new(0.0, 25.0, 6.0), 26), 1.6)];
        // The left transverse sinus leaves low, so the junction is not symmetric.
        paths[2].0.iter_mut().for_each(|p| p.z -= 0.06 * p.x.abs());
        let mut raw = Mesh::new();
        raw.append_mesh(&chamber, 0);
        for (p, r) in &paths {
            raw.append_mesh(&primitives::tube_along(p, &vec![*r; p.len()], 32), 0);
        }
        let params = mm_solid::MakeSolidParams::from_sliders(&raw, 0.5, 0.5);
        let mut m = mm_solid::make_solid(&raw, &params, None).unwrap();
        for (at, n) in [(DVec3::Z * 28.0, DVec3::Z), (DVec3::X * 28.0, DVec3::X), (-DVec3::X * 28.0, -DVec3::X), (DVec3::new(0.0, 23.0, 5.5), DVec3::new(0.0, 25.0, 6.0).normalize())] {
            mm_edit::plane_cut(&mut m, Plane::from_point_normal(at, n), &mm_edit::PlaneCutParams { keep: mm_edit::KeepSide::Below, fill: mm_edit::CutFill::None, ..Default::default() }).unwrap();
        }
        let r = run(&m, DVec3::X * 28.0);
        assert!(r.check.passed(), "{:?}", r.check);
        assert!(r.unreached.is_empty());
        assert!(r.centerline.branches.len() >= 4, "{:?}", r.centerline.branches.iter().map(|b| (b.length, b.parent)).collect::<Vec<_>>());
        println!("rings on the confluence: {:?}", rings_checked(&m, DVec3::X * 28.0));
    }

    #[test]
    fn a_tight_bend_keeps_to_its_axis() {
        // A U-turn whose bend radius is 1.6 times the vessel's.
        let (r, bend) = (2.0, 3.2);
        let mut axis: Vec<DVec3> = line(DVec3::new(-bend, 0.0, -15.0), DVec3::new(-bend, 0.0, 0.0), 30);
        axis.extend((1..64).map(|i| {
            let a = PI - PI * i as f64 / 64.0;
            DVec3::new(bend * a.cos(), 0.0, bend * a.sin())
        }));
        axis.extend(line(DVec3::new(bend, 0.0, 0.0), DVec3::new(bend, 0.0, -15.0), 30));
        let tube = primitives::tube_along(&axis, &vec![r; axis.len()], 48);
        let res = run(&tube, axis[0]);
        assert!(res.check.passed(), "{:?}", res.check);
        // Each point on the bend's axis (the inner wall is nearer, so the largest sphere sits
        // a little outward of it; at most a tenth of the radius here).
        let off = |p: DVec3| if p.z > 0.0 { (DVec3::new(p.x, 0.0, p.z).length() - bend).abs().max(p.y.abs()) } else { (p.x.abs() - bend).abs().max(p.y.abs()) };
        for p in all_points(&res.centerline) {
            assert!(off(p) < 0.1 * r, "{:.3} off the axis at {p:?}", off(p));
        }
    }

    #[test]
    fn a_closed_model_from_a_picked_point_has_its_branches() {
        // A closed Y (no open ends): the inlet picked on the trunk's wall near its bottom.
        let left: Vec<DVec3> = (0..=20).map(|i| DVec3::new(-(i as f64) * 0.6, 0.0, 20.0 + i as f64 * 0.8)).collect();
        let right: Vec<DVec3> = (0..=20).map(|i| DVec3::new(i as f64 * 0.6, 0.0, 20.0 + i as f64 * 0.8)).collect();
        let m = vessel(&[(line(DVec3::ZERO, DVec3::Z * 20.0, 20), 2.0), (left, 1.5), (right, 1.5)], &[]);
        assert!(mm_core::topology::boundary_loops(&m).is_empty());
        let inlet = LumenEnd { at: DVec3::new(2.0, 0.0, 3.0), radius: 2.0, opening: false };
        let r = compute_lumen(&m, &CappedCache::default(), &LumenInput { inlet, outlets: Vec::new(), cap_groups: Vec::new() }, &LumenParams::default(), None).unwrap().unwrap();
        assert!(r.check.passed(), "{:?}", r.check);
        let cl = &r.centerline;
        assert_eq!(cl.branches.len(), 3, "{:?}", cl.branches.iter().map(|b| (b.length, b.parent)).collect::<Vec<_>>());
        // It starts on the trunk's axis, not at the wall, and reaches up both limbs.
        let start = cl.nodes[cl.branches[0].nodes[0]].position;
        assert!(start.truncate().length() < 0.3 && start.z < 6.0, "starts at {start:?}");
        let tips: Vec<DVec3> = cl.branches[1..].iter().map(|b| cl.nodes[*b.nodes.last().unwrap()].position).collect();
        assert!(tips.iter().any(|p| p.x < -8.0 && p.z > 30.0) && tips.iter().any(|p| p.x > 8.0 && p.z > 30.0), "tips {tips:?}");
    }

    #[test]
    fn the_check_finds_a_centerline_outside() {
        let path: Vec<DVec3> = (0..=20).map(|i| DVec3::new(0.0, 0.0, i as f64)).collect();
        let tube = primitives::tube_along(&path, &vec![1.0; path.len()], 32);
        let mut cl = Centerline::default();
        for (i, p) in [DVec3::new(0.0, 0.0, 0.0), DVec3::new(0.0, 0.0, 5.0), DVec3::new(3.0, 0.0, 10.0), DVec3::new(0.0, 0.0, 15.0), DVec3::new(0.0, 0.0, 20.0)].into_iter().enumerate() {
            cl.nodes.push(CenterlineNode { position: p, radius: 1.0, band: i });
        }
        cl.branches.push(Branch { nodes: (0..5).collect(), length: 0.0, mean_radius: 1.0, min_radius: 1.0, max_radius: 1.0, parent: None, joins: None });
        let k = check(&tube, &CappedCache::default(), &cl, &[DVec3::ZERO, DVec3::Z * 20.0], &|_| true).unwrap();
        assert_eq!((k.points, k.segments), (5, 4));
        assert_eq!(k.outside, vec![DVec3::new(3.0, 0.0, 10.0)]);
        assert_eq!(k.crossing.len(), 2);
        assert!(!k.passed());
    }

    /// The centerline of a mesh of one's own, against the rings': `POLYMIX_CENTERLINE_MESH` (a
    /// mesh file), `POLYMIX_CENTERLINE_INLET` (x,y,z near the inlet end; else the widest end),
    /// `POLYMIX_CENTERLINE_OUT` (a .vtk to save the centerline to), `POLYMIX_CENTERLINE_BLIND`
    /// (set: blind branches too). Run with
    /// `cargo test --release -p polymix centerline::lumen::tests::a_mesh -- --ignored --nocapture`.
    #[test]
    #[ignore = "needs POLYMIX_CENTERLINE_MESH"]
    fn a_mesh() {
        let path = std::env::var("POLYMIX_CENTERLINE_MESH").expect("POLYMIX_CENTERLINE_MESH");
        let mesh = mm_io::read_mesh(std::path::Path::new(&path)).unwrap().mesh;
        let mut ends = openings(&mesh);
        let near: Option<DVec3> = std::env::var("POLYMIX_CENTERLINE_INLET").ok().map(|s| {
            let v: Vec<f64> = s.split(',').map(|x| x.trim().parse().unwrap()).collect();
            DVec3::new(v[0], v[1], v[2])
        });
        let i = match near {
            Some(p) => (0..ends.len()).min_by(|&a, &b| ends[a].at.distance(p).total_cmp(&ends[b].at.distance(p))),
            None => (0..ends.len()).max_by(|&a, &b| ends[a].radius.total_cmp(&ends[b].radius)),
        }
        .expect("an end");
        let inlet = ends.remove(i);
        println!("{} triangles; inlet at {:?} (radius {:.2}), {} other ends", mesh.triangle_count(), inlet.at, inlet.radius, ends.len());
        let t = std::time::Instant::now();
        let rings = rings_checked(&mesh, inlet.at);
        println!("rings: {} of {} points outside, {} of {} segments cross the wall ({:.1} s)", rings.outside.len(), rings.points, rings.crossing.len(), rings.segments, t.elapsed().as_secs_f64());
        for p in rings.outside.iter().take(10) {
            println!("  outside: {p:.2?}");
        }
        let t = std::time::Instant::now();
        let params = LumenParams { blind_branches: std::env::var("POLYMIX_CENTERLINE_BLIND").is_ok(), ..Default::default() };
        let r = compute_lumen(&mesh, &CappedCache::default(), &LumenInput { inlet, outlets: ends, cap_groups: Vec::new() }, &params, None).unwrap().unwrap();
        println!("lumen: {:.1} s, {:.3} mm voxels, {} branches, {} loops, unreached {:?}", t.elapsed().as_secs_f64(), r.voxel, r.centerline.branches.len(), r.loops, r.unreached);
        println!("lumen check: {} of {} points outside, {} of {} segments cross the wall, closest {:.3} mm", r.check.outside.len(), r.check.points, r.check.crossing.len(), r.check.segments, r.check.min_clearance);
        for (i, b) in r.centerline.branches.iter().enumerate() {
            println!("  {}: {:.1} mm, ⌀ {:.2} ({:.2}..{:.2}), parent {:?}, joins {:?}", i + 1, b.length, 2.0 * b.mean_radius, 2.0 * b.min_radius, 2.0 * b.max_radius, b.parent.map(|p| p + 1), b.joins.map(|j| j + 1));
        }
        // The points nearest the surface.
        let capped = CappedCache::default();
        let closed = capped.get(&mesh, &|_| true).unwrap();
        let mut near: Vec<(f64, usize, usize, f64)> = r.centerline.branches.iter().enumerate().flat_map(|(bi, b)| b.nodes.iter().enumerate().map(move |(k, &i)| (bi, k, i))).map(|(bi, k, i)| {
            let p = r.centerline.nodes[i].position;
            (closed.bvh.nearest(&closed.mesh, p, f64::INFINITY).unwrap().dist_sq.sqrt(), bi + 1, k, r.centerline.nodes[i].radius)
        }).collect();
        near.sort_by(|a, b| a.0.total_cmp(&b.0));
        for (d, b, k, radius) in near.iter().take(6) {
            println!("  {d:.3} mm from the surface: branch {b} point {k} (radius {radius:.3})");
        }
        if let Ok(out) = std::env::var("POLYMIX_CENTERLINE_OUT") {
            std::fs::write(out, r.centerline.to_vtk()).unwrap();
        }
        assert!(r.check.passed());
    }
}
