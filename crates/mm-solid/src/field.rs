//! Field sources: the signed distance to a triangle soup (sign from the generalized
//! winding number), user implicit functions, and CSG / hollowing combinators.

use crate::grid::{lidx, FieldSource, GridSpec, BRICK, BRICK_VOL};
use crate::winding::WindingNumber;
use glam::DVec3;
use mm_core::geom::closest_point_on_triangle;
use mm_core::{Aabb, Bvh, Mesh, INVALID};
use smallvec::SmallVec;
use rayon::prelude::*;
use std::borrow::Cow;
use std::sync::atomic::{AtomicU64, Ordering};

/// Slope of the winding-number modulation: `g(w) = clamp(SHARP·(τ − |w|), −1, 1)`.
const SHARP: f64 = 4.0;

/// Winding-number accuracy parameter used for fields. The field only needs the winding
/// number for its sign and for the smooth modulation across holes (saturating 0.25 away
/// from the threshold), so β = 1.5 (max error ≈ 2e-2 next to the surface, ≈ 2× faster than
/// β = 2) is ample.
pub const FIELD_BETA: f64 = 1.5;

/// Default narrow-band margin beyond |offset|, in cells. Every lattice edge with a sign
/// change has both endpoints within √3 cells of the zero set, so 2 cells suffices.
pub const DEFAULT_BAND_CELLS: f64 = 2.0;

/// Thin-feature thickening data for [`MeshSdf`] (see `MakeSolidParams::min_thickness`).
#[derive(Clone, Debug)]
pub(crate) struct ThinShell {
    /// Triangles to thicken and their target half-thickness.
    bvh: Bvh,
    radius: Vec<f64>,
    /// Per-BVH-node maximum radius.
    node_rmax: Vec<f64>,
}

/// Narrow-band signed distance to a triangle mesh/soup.
///
/// `f(p) = g(|w(p)|) · d(p) − offset`, where `d` is the exact unsigned distance to the
/// triangles (computed up to `band`, clamped beyond), `w` the fast winding number, and
/// `g(w) = clamp(4·(τ − w), −1, 1)` the inside/outside indicator (τ = winding threshold).
/// Away from holes `|w|` is ≈0 or ≈1 and `f` is the exact signed distance; across holes
/// `g` interpolates smoothly, so the zero set caps holes along the `|w| = τ` surface
/// instead of producing a staircase. Using `|w|` makes inverted shells count as solid.
pub struct MeshSdf<'a> {
    mesh: &'a Mesh,
    bvh: Cow<'a, Bvh>,
    wn: Cow<'a, WindingNumber>,
    tris: Vec<u32>,
    band: f64,
    offset: f64,
    tau: f64,
    far: f64,
    thin: Option<ThinShell>,
    /// The mesh is closed and consistently oriented, so its winding number is an integer
    /// that only changes across the surface (see `eval_brick`).
    piecewise_constant: bool,
}

impl<'a> MeshSdf<'a> {
    /// SDF of `mesh` for a lattice with `cell_size`, offset by `offset` (positive grows),
    /// with inside defined by `|winding| ≥ winding_threshold`.
    pub fn new(mesh: &'a Mesh, cell_size: f64, offset: f64, winding_threshold: f64) -> Self {
        Self::with_band(mesh, offset, winding_threshold, offset.abs() + DEFAULT_BAND_CELLS * cell_size)
    }

    /// As [`MeshSdf::new`] with an explicit band: exact distances are computed up to
    /// `band` (must exceed `|offset|`); the field is clamped to `±(band − |offset|)`.
    pub fn with_band(mesh: &'a Mesh, offset: f64, winding_threshold: f64, band: f64) -> Self {
        let bvh = Bvh::build(mesh);
        let wn = WindingNumber::with_beta(mesh, FIELD_BETA);
        Self::from_parts(mesh, bvh, wn, offset, winding_threshold, band)
    }

    /// Build from a prebuilt BVH and winding-number tree of `mesh`.
    pub fn from_parts(mesh: &'a Mesh, bvh: Bvh, wn: WindingNumber, offset: f64, winding_threshold: f64, band: f64) -> Self {
        Self::from_cows(mesh, Cow::Owned(bvh), Cow::Owned(wn), offset, winding_threshold, band)
    }

    /// As [`MeshSdf::from_parts`], borrowing the trees (built over all triangles of `mesh`;
    /// the winding number with the accuracy the field should use, normally [`FIELD_BETA`]).
    pub fn from_refs(mesh: &'a Mesh, bvh: &'a Bvh, wn: &'a WindingNumber, offset: f64, winding_threshold: f64, band: f64) -> Self {
        Self::from_cows(mesh, Cow::Borrowed(bvh), Cow::Borrowed(wn), offset, winding_threshold, band)
    }

    fn from_cows(mesh: &'a Mesh, bvh: Cow<'a, Bvh>, wn: Cow<'a, WindingNumber>, offset: f64, winding_threshold: f64, band: f64) -> Self {
        let band = band.max(offset.abs() * 1.0001 + 1e-12);
        let far = band - offset.abs();
        let tris = mesh.triangle_ids().collect();
        // (The winding number leaves out triangles with a non-finite corner.)
        let piecewise_constant = mesh.is_closed() && mesh.is_consistently_oriented() && mesh.vertex_ids().all(|v| mesh.vertex(v).is_finite());
        Self { mesh, bvh, wn, tris, band, offset, tau: winding_threshold, far, thin: None, piecewise_constant }
    }

    /// The winding-number evaluator.
    pub fn winding(&self) -> &WindingNumber {
        &self.wn
    }
    /// BVH of the input mesh.
    pub fn bvh(&self) -> &Bvh {
        &self.bvh
    }
    /// The input mesh.
    pub fn mesh(&self) -> &Mesh {
        self.mesh
    }
    /// Exact-distance band width.
    pub fn band(&self) -> f64 {
        self.band
    }

    /// Unsigned distance to the input, clamped to `band`.
    #[inline]
    pub fn unsigned_distance(&self, p: DVec3) -> f64 {
        self.nearest_hinted(p, INVALID).0
    }

    /// Distance to the input (clamped to `band`) and the nearest triangle (or `INVALID`),
    /// using triangle `hint` (e.g. the previous sample's answer) as the initial bound.
    fn nearest_hinted(&self, p: DVec3, hint: u32) -> (f64, u32) {
        let mut best2 = self.band * self.band;
        let mut best_t = INVALID;
        if hint != INVALID {
            let [a, b, c] = self.mesh.tri_positions(hint);
            let d2 = closest_point_on_triangle(p, a, b, c).0.distance_squared(p);
            if d2 < best2 {
                best2 = d2;
                best_t = hint;
            }
        }
        let nodes = &self.bvh.nodes;
        if nodes.is_empty() {
            return (self.band, INVALID);
        }
        let mut stack: SmallVec<[u32; 64]> = SmallVec::new();
        stack.push(0);
        while let Some(ni) = stack.pop() {
            let node = &nodes[ni as usize];
            if node.bounds.distance_squared(p) >= best2 {
                continue;
            }
            if node.is_leaf() {
                for &t in self.bvh.leaf_tris(node) {
                    let [a, b, c] = self.mesh.tri_positions(t);
                    let d2 = closest_point_on_triangle(p, a, b, c).0.distance_squared(p);
                    if d2 < best2 {
                        best2 = d2;
                        best_t = t;
                    }
                }
            } else {
                let (l, r) = (node.start, node.start + 1);
                let dl = nodes[l as usize].bounds.distance_squared(p);
                let dr = nodes[r as usize].bounds.distance_squared(p);
                if dl < dr {
                    stack.push(r);
                    stack.push(l);
                } else {
                    stack.push(l);
                    stack.push(r);
                }
            }
        }
        if best_t == INVALID {
            (self.band, INVALID)
        } else {
            (best2.sqrt().min(self.band), best_t)
        }
    }

    #[inline]
    fn value(&self, p: DVec3, d: f64) -> f64 {
        let mut f = self.indicator(p) * d;
        if self.thin.is_some() {
            f = f.min(self.thin_distance(p));
        }
        (f - self.offset).clamp(-self.far, self.far)
    }

    /// Inside/outside indicator in [−1, 1] (−1 inside).
    #[inline]
    pub fn indicator(&self, p: DVec3) -> f64 {
        ((self.tau - self.wn.eval(p).abs()) * SHARP).clamp(-1.0, 1.0)
    }

    /// Enable min-thickness handling: points within `radius[i]` of triangle `tris[i]`
    /// count as inside.
    pub(crate) fn set_thin_shell(&mut self, tris: &[u32], radius: Vec<f64>) {
        if tris.is_empty() {
            self.thin = None;
            return;
        }
        let bvh = Bvh::build_subset(self.mesh, tris);
        // Map each BVH leaf slot back to its radius (bvh.tris is a permutation of `tris`).
        let mut rmap = crate::hash::FastMap::default();
        for (i, &t) in tris.iter().enumerate() {
            rmap.insert(t, radius[i]);
        }
        let leaf_r: Vec<f64> = bvh.tris.iter().map(|t| rmap[t]).collect();
        let mut node_rmax = vec![0.0; bvh.nodes.len()];
        // Children always have larger indices than their parent: fill bottom-up.
        for i in (0..bvh.nodes.len()).rev() {
            let n = &bvh.nodes[i];
            node_rmax[i] = if n.is_leaf() {
                leaf_r[n.start as usize..(n.start + n.count) as usize].iter().copied().fold(0.0, f64::max)
            } else {
                node_rmax[n.start as usize].max(node_rmax[n.start as usize + 1])
            };
        }
        let max_r = node_rmax.first().copied().unwrap_or(0.0);
        self.thin = Some(ThinShell { bvh, radius: leaf_r, node_rmax });
        // Keep the original margin beyond the thickened zero set.
        self.band += max_r;
        self.far = self.band - self.offset.abs();
    }

    /// min over thickened triangles of (distance − radius), or +∞.
    fn thin_distance(&self, p: DVec3) -> f64 {
        let Some(th) = &self.thin else { return f64::INFINITY };
        let mut best = self.band;
        let mut stack: [u32; 96] = [0; 96];
        let mut sp = 1;
        while sp > 0 {
            sp -= 1;
            let ni = stack[sp] as usize;
            let node = &th.bvh.nodes[ni];
            let lb = node.bounds.distance_squared(p).sqrt() - th.node_rmax[ni];
            if lb >= best {
                continue;
            }
            if node.is_leaf() {
                for slot in node.start as usize..(node.start + node.count) as usize {
                    let t = th.bvh.tris[slot];
                    let [a, b, c] = self.mesh.tri_positions(t);
                    let (q, _) = closest_point_on_triangle(p, a, b, c);
                    let v = q.distance(p) - th.radius[slot];
                    if v < best {
                        best = v;
                    }
                }
            } else if sp + 2 <= stack.len() {
                stack[sp] = node.start;
                stack[sp + 1] = node.start + 1;
                sp += 2;
            }
        }
        best
    }
}

impl FieldSource for MeshSdf<'_> {
    fn eval(&self, p: DVec3) -> f64 {
        self.value(p, self.unsigned_distance(p))
    }

    fn eval_brick(&self, origin: DVec3, cell_size: f64, out: &mut [f64; BRICK_VOL]) {
        // Serpentine order keeps consecutive samples adjacent, so the previous nearest
        // triangle is a tight initial bound.
        if self.thin.is_some() {
            let mut hint = INVALID;
            for z in 0..8 {
                for yy in 0..8 {
                    let y = if z % 2 == 0 { yy } else { 7 - yy };
                    for xx in 0..8 {
                        let x = if (z * 8 + yy) % 2 == 0 { xx } else { 7 - xx };
                        let p = origin + DVec3::new(x as f64, y as f64, z as f64) * cell_size;
                        let (d, t) = self.nearest_hinted(p, hint);
                        if t != INVALID {
                            hint = t;
                        }
                        out[lidx(x, y, z)] = self.value(p, d);
                    }
                }
            }
            return;
        }
        // Without thickening the value is `clamp(g·d − offset)`, and two shortcuts give the
        // values `value` would for less work:
        // - The sign comes first. Deep inside (g = −1) a grow of at least `far` clamps the
        //   value to −far whatever the distance (likewise outside for a shrink), so the
        //   nearest-point query is skipped there.
        // - On a closed, consistently oriented mesh the winding number only changes across
        //   the surface. When the previous sample's empty ball (radius `d`) reaches this one
        //   and its sign was well settled, this sample has the same sign. A skipped sample
        //   has no distance, so the one after it evaluates its winding number again.
        let far = self.far;
        let settled_inside = self.offset >= far;
        let settled_outside = -self.offset >= far;
        let reach = cell_size * (1.0 + 1e-9);
        let mut hint = INVALID;
        // Distance and unclamped sign value of the previous sample, when it was computed.
        let mut prev: Option<(f64, f64)> = None;
        for z in 0..8 {
            for yy in 0..8 {
                let y = if z % 2 == 0 { yy } else { 7 - yy };
                for xx in 0..8 {
                    let x = if (z * 8 + yy) % 2 == 0 { xx } else { 7 - xx };
                    let p = origin + DVec3::new(x as f64, y as f64, z as f64) * cell_size;
                    let u = match prev {
                        Some((d, u)) if self.piecewise_constant && d > reach && u.abs() >= 1.4 => u,
                        _ => (self.tau - self.wn.eval(p).abs()) * SHARP,
                    };
                    let g = u.clamp(-1.0, 1.0);
                    if (g == -1.0 && settled_inside) || (g == 1.0 && settled_outside) {
                        out[lidx(x, y, z)] = g * far;
                        prev = None;
                        continue;
                    }
                    let (d, t) = self.nearest_hinted(p, hint);
                    if t != INVALID {
                        hint = t;
                    }
                    prev = Some((d, u));
                    out[lidx(x, y, z)] = (g * d - self.offset).clamp(-far, far);
                }
            }
        }
    }

    fn far(&self) -> f64 {
        self.far
    }

    fn seed_bricks(&self, grid: &GridSpec) -> Vec<[u32; 3]> {
        rasterize_band(self.mesh, &self.tris, grid, self.band)
    }

    fn uniform_sign(&self, c: DVec3, r: f64) -> Option<bool> {
        let reach = (2.0 * r).max(r + self.band);
        if self.bvh.nearest(self.mesh, c, reach).is_some() {
            return None;
        }
        let g = (self.tau - self.wn.eval(c).abs()) * SHARP;
        if g >= 1.0 {
            Some(false)
        } else if g <= -1.0 {
            Some(true)
        } else {
            None
        }
    }
}

/// Largest brick grid (in bricks) whose seeds are collected in a bitset (128 MiB); beyond
/// it the candidates are listed and sorted instead.
const SEED_BITSET_MAX: u64 = 1 << 30;

/// The bricks the box of triangle `p` padded by `band` covers, as inclusive ranges.
#[inline]
fn brick_range(p: &[DVec3; 3], grid: &GridSpec, band: f64) -> Option<([u32; 3], [u32; 3])> {
    let bsize = BRICK as f64 * grid.cell_size;
    let nb = grid.bricks;
    let bb = Aabb::from_points(*p).expanded(band);
    let mut lo = [0u32; 3];
    let mut hi = [0u32; 3];
    let mut empty = false;
    for a in 0..3 {
        // A brick's samples span [8b·h, (8b+7)·h] relative to the origin.
        let l = ((bb.min[a] - grid.origin[a]) / bsize - 7.0 / 8.0).ceil().max(0.0);
        let u = ((bb.max[a] - grid.origin[a]) / bsize).floor().min(nb[a] as f64 - 1.0);
        if u < l {
            empty = true;
        }
        lo[a] = l as u32;
        hi[a] = u.max(0.0) as u32;
    }
    (!empty).then_some((lo, hi))
}

/// Bricks whose sample box comes within `band` of any of the triangles, sorted (by x, then
/// y, then z) and without repeats.
pub(crate) fn rasterize_band(mesh: &Mesh, tris: &[u32], grid: &GridSpec, band: f64) -> Vec<[u32; 3]> {
    let nb = grid.bricks;
    let total = nb.iter().map(|&n| n as u64).product::<u64>();
    if total > SEED_BITSET_MAX {
        return rasterize_band_listed(mesh, tris, grid, band);
    }
    let rb = grid.brick_radius();
    // One bit per brick, numbered so that the set bits read in order come out sorted as
    // `[x, y, z]` arrays. A brick that another triangle already claimed skips the distance
    // test, so each brick is tested about once rather than once per nearby triangle.
    let (ny, nz) = (nb[1] as usize, nb[2] as usize);
    let bit = |b: [u32; 3]| (b[0] as usize * ny + b[1] as usize) * nz + b[2] as usize;
    let words: Vec<AtomicU64> = (0..(total as usize).div_ceil(64)).map(|_| AtomicU64::new(0)).collect();
    tris.par_iter().for_each(|&t| {
        let p = mesh.tri_positions(t);
        let Some((lo, hi)) = brick_range(&p, grid, band) else { return };
        let count = (hi[0] - lo[0] + 1) as u64 * (hi[1] - lo[1] + 1) as u64 * (hi[2] - lo[2] + 1) as u64;
        for z in lo[2]..=hi[2] {
            for y in lo[1]..=hi[1] {
                for x in lo[0]..=hi[0] {
                    let b = [x, y, z];
                    let i = bit(b);
                    let (word, mask) = (&words[i >> 6], 1u64 << (i & 63));
                    if word.load(Ordering::Relaxed) & mask != 0 {
                        continue;
                    }
                    if count > 8 {
                        let c = grid.brick_center(b);
                        let (q, _) = closest_point_on_triangle(c, p[0], p[1], p[2]);
                        if q.distance(c) > band + rb {
                            continue;
                        }
                    }
                    word.fetch_or(mask, Ordering::Relaxed);
                }
            }
        }
    });
    let yz = ny * nz;
    words
        .par_iter()
        .enumerate()
        .flat_map_iter(|(w, word)| {
            let mut bits = word.load(Ordering::Relaxed);
            let mut out = Vec::new();
            while bits != 0 {
                let i = w * 64 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                out.push([(i / yz) as u32, (i % yz / nz) as u32, (i % nz) as u32]);
            }
            out.into_iter()
        })
        .collect()
}

/// [`rasterize_band`] for very large grids: every candidate listed, then sorted.
fn rasterize_band_listed(mesh: &Mesh, tris: &[u32], grid: &GridSpec, band: f64) -> Vec<[u32; 3]> {
    let rb = grid.brick_radius();
    let mut keys: Vec<[u32; 3]> = tris
        .par_iter()
        .flat_map_iter(|&t| {
            let p = mesh.tri_positions(t);
            let mut out = Vec::new();
            if let Some((lo, hi)) = brick_range(&p, grid, band) {
                let count = (hi[0] - lo[0] + 1) as u64 * (hi[1] - lo[1] + 1) as u64 * (hi[2] - lo[2] + 1) as u64;
                for z in lo[2]..=hi[2] {
                    for y in lo[1]..=hi[1] {
                        for x in lo[0]..=hi[0] {
                            let b = [x, y, z];
                            if count > 8 {
                                let c = grid.brick_center(b);
                                let (q, _) = closest_point_on_triangle(c, p[0], p[1], p[2]);
                                if q.distance(c) > band + rb {
                                    continue;
                                }
                            }
                            out.push(b);
                        }
                    }
                }
            }
            out.into_iter()
        })
        .collect();
    keys.par_sort_unstable();
    keys.dedup();
    keys
}

/// A user implicit function `f` (negative inside) with a Lipschitz bound, used to skip
/// regions that cannot contain the zero set.
pub struct ImplicitField<F: Fn(DVec3) -> f64 + Sync> {
    f: F,
    lipschitz: f64,
    far: f64,
}

impl<F: Fn(DVec3) -> f64 + Sync> ImplicitField<F> {
    /// `lipschitz` bounds |∇f| (1 for a true distance field); values are clamped to
    /// `±far`. Larger `lipschitz` values are safer but slower.
    pub fn new(f: F, lipschitz: f64, far: f64) -> Self {
        Self { f, lipschitz: lipschitz.max(1e-12), far }
    }
}

impl<F: Fn(DVec3) -> f64 + Sync> FieldSource for ImplicitField<F> {
    fn eval(&self, p: DVec3) -> f64 {
        let v = (self.f)(p);
        if v.is_nan() {
            self.far
        } else {
            v.clamp(-self.far, self.far)
        }
    }
    fn far(&self) -> f64 {
        self.far
    }
    fn seed_bricks(&self, grid: &GridSpec) -> Vec<[u32; 3]> {
        let nb = grid.bricks;
        let reach = self.lipschitz * (grid.brick_radius() + 3f64.sqrt() * grid.cell_size);
        let all: Vec<[u32; 3]> = (0..nb[2])
            .flat_map(|z| (0..nb[1]).flat_map(move |y| (0..nb[0]).map(move |x| [x, y, z])))
            .collect();
        all.into_par_iter().filter(|&b| (self.f)(grid.brick_center(b)).abs() <= reach).collect()
    }
    fn uniform_sign(&self, c: DVec3, r: f64) -> Option<bool> {
        let v = (self.f)(c);
        (v.abs() > self.lipschitz * r).then_some(v < 0.0)
    }
}

/// Boolean operation for [`crate::sdf_boolean`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BoolOp {
    /// A ∪ B
    #[default]
    Union,
    /// A − B
    Difference,
    /// A ∩ B
    Intersection,
}

/// CSG combination of two fields (min/max of signed distances).
pub(crate) struct BoolField<A, B> {
    pub a: A,
    pub b: B,
    pub op: BoolOp,
}

impl<A: FieldSource, B: FieldSource> FieldSource for BoolField<A, B> {
    fn eval(&self, p: DVec3) -> f64 {
        let far = self.far();
        let fa = self.a.eval(p);
        let v = match self.op {
            BoolOp::Union => {
                if fa <= -far {
                    return -far;
                }
                fa.min(self.b.eval(p))
            }
            BoolOp::Intersection => {
                if fa >= far {
                    return far;
                }
                fa.max(self.b.eval(p))
            }
            BoolOp::Difference => {
                if fa >= far {
                    return far;
                }
                fa.max(-self.b.eval(p))
            }
        };
        v.clamp(-far, far)
    }
    fn far(&self) -> f64 {
        self.a.far().min(self.b.far())
    }
    fn seed_bricks(&self, grid: &GridSpec) -> Vec<[u32; 3]> {
        let mut s = self.a.seed_bricks(grid);
        s.extend(self.b.seed_bricks(grid));
        s
    }
    fn uniform_sign(&self, c: DVec3, r: f64) -> Option<bool> {
        let a = self.a.uniform_sign(c, r);
        let b = self.b.uniform_sign(c, r);
        match self.op {
            BoolOp::Union => match (a, b) {
                (Some(true), _) | (_, Some(true)) => Some(true),
                (Some(false), Some(false)) => Some(false),
                _ => None,
            },
            BoolOp::Intersection => match (a, b) {
                (Some(false), _) | (_, Some(false)) => Some(false),
                (Some(true), Some(true)) => Some(true),
                _ => None,
            },
            BoolOp::Difference => match (a, b) {
                (Some(false), _) | (_, Some(true)) => Some(false),
                (Some(true), Some(false)) => Some(true),
                _ => None,
            },
        }
    }
}

/// Shell of thickness `t` inside the base solid: `max(f, −f − t)`.
pub(crate) struct HollowField<A> {
    pub base: A,
    pub thickness: f64,
    pub far: f64,
}

impl<A: FieldSource> FieldSource for HollowField<A> {
    fn eval(&self, p: DVec3) -> f64 {
        let f = self.base.eval(p);
        f.max(-f - self.thickness).clamp(-self.far, self.far)
    }
    fn far(&self) -> f64 {
        self.far
    }
    fn seed_bricks(&self, grid: &GridSpec) -> Vec<[u32; 3]> {
        self.base.seed_bricks(grid)
    }
    fn uniform_sign(&self, c: DVec3, r: f64) -> Option<bool> {
        // Uniform regions of the base are either outside or deep inside (hollowed out).
        self.base.uniform_sign(c, r).map(|_| false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_core::primitives;

    /// `eval_brick` as it was: every sample's nearest point (hinted by the previous one's),
    /// then `value`.
    fn reference_brick(sdf: &MeshSdf, origin: DVec3, h: f64) -> [f64; BRICK_VOL] {
        let mut out = [0.0; BRICK_VOL];
        let mut hint = INVALID;
        for z in 0..8 {
            for yy in 0..8 {
                let y = if z % 2 == 0 { yy } else { 7 - yy };
                for xx in 0..8 {
                    let x = if (z * 8 + yy) % 2 == 0 { xx } else { 7 - xx };
                    let p = origin + DVec3::new(x as f64, y as f64, z as f64) * h;
                    let (d, t) = sdf.nearest_hinted(p, hint);
                    if t != INVALID {
                        hint = t;
                    }
                    out[lidx(x, y, z)] = sdf.value(p, d);
                }
            }
        }
        out
    }

    /// Every brick of the grid around `m` evaluated both ways, bit for bit.
    fn check_bricks(m: &Mesh, h: f64, offsets: &[f64]) {
        for &offset in offsets {
            let sdf = MeshSdf::new(m, h, offset, 0.5);
            let grid = GridSpec::covering(&m.bounds(), h, sdf.band() + 2.0 * h).unwrap();
            let mut bricks = Vec::new();
            for z in 0..grid.bricks[2] {
                for y in 0..grid.bricks[1] {
                    for x in 0..grid.bricks[0] {
                        bricks.push([x, y, z]);
                    }
                }
            }
            let differ = bricks
                .par_iter()
                .filter(|&&b| {
                    let o = grid.brick_origin(b);
                    let mut fast = [0.0; BRICK_VOL];
                    sdf.eval_brick(o, h, &mut fast);
                    let slow = reference_brick(&sdf, o, h);
                    fast.iter().zip(&slow).any(|(a, b)| a.to_bits() != b.to_bits())
                })
                .count();
            assert_eq!(differ, 0, "offset {offset}: {differ} of {} bricks differ", bricks.len());
        }
    }

    #[test]
    fn brick_shortcuts_give_the_same_values() {
        // Closed and consistently oriented: both shortcuts.
        let sphere = primitives::icosphere(1.0, 4);
        check_bricks(&sphere, 0.05, &[0.0, 0.3, -0.3, 0.12, -0.08]);
        // Closed and overlapping itself (winding number 2 in the overlap).
        let mut two = primitives::icosphere(1.0, 3);
        let mut other = primitives::icosphere(0.8, 3);
        other.transform(|p| p + DVec3::new(0.7, 0.2, 0.1));
        two.append_mesh(&other, 1);
        check_bricks(&two, 0.05, &[0.0, 0.25, -0.25]);
        // A patch turned inside out: no sign carried.
        let mut flipped = primitives::icosphere(1.0, 4);
        for t in flipped.triangle_ids().collect::<Vec<_>>() {
            if flipped.tri_centroid(t).z > 0.6 {
                flipped.reverse_triangle(t);
            }
        }
        check_bricks(&flipped, 0.05, &[0.0, 0.3, -0.3]);
        // Open: no sign carried.
        let keep: Vec<u32> = sphere.triangle_ids().filter(|&t| sphere.tri_centroid(t).z < 0.7).collect();
        let (open, _) = sphere.submesh(&keep);
        check_bricks(&open, 0.05, &[0.0, 0.3, -0.3]);
    }

    #[test]
    fn seed_bitset_matches_the_sorted_list() {
        let mut seed = 0x1234_5678_9abc_def1u64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        let sphere = primitives::icosphere(1.0, 4);
        let mut soup = Mesh::new();
        for _ in 0..400 {
            let c = DVec3::new(rnd(), rnd(), rnd()) * 3.0 - DVec3::splat(1.5);
            let v: Vec<u32> = (0..3).map(|_| soup.append_vertex(c + (DVec3::new(rnd(), rnd(), rnd()) - DVec3::splat(0.5)) * 0.6)).collect();
            let _ = soup.append_triangle([v[0], v[1], v[2]], 0);
        }
        for (m, h) in [(&sphere, 0.05), (&sphere, 0.013), (&soup, 0.04)] {
            let tris: Vec<u32> = m.triangle_ids().collect();
            for band in [0.0, 0.1, 0.5] {
                let full = GridSpec::covering(&m.bounds(), h, band + 2.0 * h).unwrap();
                // Also a grid over only part of the mesh.
                let part = GridSpec::new(full.origin + DVec3::splat(0.4), h, full.bricks.map(|b| b / 2 + 1));
                for grid in [full, part] {
                    assert_eq!(rasterize_band(m, &tris, &grid, band), rasterize_band_listed(m, &tris, &grid, band), "h {h} band {band}");
                }
            }
        }
    }
}
