//! Sparse blocked scalar grids and the generic narrow-band field builder.
//!
//! Samples live on a regular lattice `origin + h·(i, j, k)`, grouped into 8³ bricks.
//! Only *active* bricks store values (quantized to `i16` relative to the field's clamp
//! value [`FieldSource::far`]); every other brick has a uniform inside/outside state kept
//! in a one-bit-per-brick table. The field is therefore defined at every sample of the
//! grid, which is what makes the extracted surface watertight by construction.
//!
//! [`build_field`] decides which bricks to activate:
//! 1. *seed* bricks supplied by the source (for meshes: bricks touched by each triangle's
//!    bounding box padded by the narrow-band width);
//! 2. a hierarchical classification of the rest of the volume (blocks of 16³ bricks
//!    subdivided until the source can prove a uniform sign, single bricks by their centre);
//! 3. propagation: an inactive brick is activated when its uniform state disagrees with an
//!    adjacent active brick's samples or with a neighbouring brick's state. This follows
//!    sign changes that leave the narrow band (e.g. winding-number caps over large holes)
//!    and guarantees the invariant used by extraction: every sample within Chebyshev
//!    distance 1 of an inactive brick has that brick's sign.

use crate::hash::FastMap;
use crate::progress::Prog;
use crate::{Progress, Result, SolidError};
use glam::DVec3;
use mm_core::Aabb;
use rayon::prelude::*;

/// Samples per brick edge.
pub const BRICK: u32 = 8;
pub(crate) const BRICK_VOL: usize = 512;
/// Quantized value of the clamp magnitude.
pub(crate) const QMAX: i16 = i16::MAX;
/// Largest supported number of samples along one axis.
pub const MAX_SAMPLES_PER_AXIS: u32 = 16384;
const KEY_MASK: u64 = (1 << 21) - 1;

#[inline]
pub(crate) fn key(b: [u32; 3]) -> u64 {
    b[0] as u64 | ((b[1] as u64) << 21) | ((b[2] as u64) << 42)
}

#[inline]
pub(crate) fn unkey(k: u64) -> [u32; 3] {
    [(k & KEY_MASK) as u32, ((k >> 21) & KEY_MASK) as u32, ((k >> 42) & KEY_MASK) as u32]
}

/// Index of local sample (x, y, z) inside a brick.
#[inline]
pub(crate) fn lidx(x: usize, y: usize, z: usize) -> usize {
    x + 8 * y + 64 * z
}

#[inline]
pub(crate) fn quantize(f: f64, far: f64) -> i16 {
    let q = (f / far * QMAX as f64).round().clamp(-(QMAX as f64), QMAX as f64) as i16;
    // Zero is never stored: exact zeros count as outside, keeping the PL zero set generic.
    if q == 0 {
        1
    } else {
        q
    }
}

/// Lattice description: sample `(i, j, k)` is at `origin + cell_size·(i, j, k)`;
/// there are `bricks[a] · 8` samples along axis `a`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GridSpec {
    /// Position of sample (0, 0, 0).
    pub origin: DVec3,
    /// Sample spacing.
    pub cell_size: f64,
    /// Number of 8³ bricks along each axis.
    pub bricks: [u32; 3],
}

impl GridSpec {
    /// Grid with the given origin, cell size and brick counts.
    pub fn new(origin: DVec3, cell_size: f64, bricks: [u32; 3]) -> Self {
        Self { origin, cell_size, bricks }
    }

    /// Grid covering `bounds` expanded by `pad`, centred on the bounds.
    pub fn covering(bounds: &Aabb, cell_size: f64, pad: f64) -> Result<Self> {
        if !(cell_size > 0.0) || !cell_size.is_finite() {
            return Err(SolidError::InvalidParameter(format!("cell size {cell_size}")));
        }
        if bounds.is_empty() || !bounds.min.is_finite() || !bounds.max.is_finite() {
            return Err(SolidError::EmptyInput);
        }
        let ext = bounds.extents() + DVec3::splat(2.0 * pad.max(0.0));
        let mut bricks = [0u32; 3];
        for a in 0..3 {
            let samples = (ext[a] / cell_size).ceil() + 2.0;
            if samples > MAX_SAMPLES_PER_AXIS as f64 {
                return Err(SolidError::GridTooLarge(format!(
                    "{samples:.0} samples along axis {a} (max {MAX_SAMPLES_PER_AXIS}); increase the cell size"
                )));
            }
            bricks[a] = (samples as u32).max(2).div_ceil(BRICK);
        }
        let span = DVec3::new(bricks[0] as f64, bricks[1] as f64, bricks[2] as f64) * (BRICK as f64) - DVec3::ONE;
        let origin = bounds.center() - span * cell_size * 0.5;
        Ok(Self { origin, cell_size, bricks })
    }

    /// Samples along each axis.
    pub fn samples(&self) -> [u32; 3] {
        self.bricks.map(|b| b * BRICK)
    }

    /// Total number of bricks (active or not).
    pub fn brick_count(&self) -> u64 {
        self.bricks.iter().map(|&b| b as u64).product()
    }

    /// Position of sample `i`.
    #[inline]
    pub fn sample_point(&self, i: [u32; 3]) -> DVec3 {
        self.origin + DVec3::new(i[0] as f64, i[1] as f64, i[2] as f64) * self.cell_size
    }

    /// Position of the first sample of brick `b`.
    #[inline]
    pub fn brick_origin(&self, b: [u32; 3]) -> DVec3 {
        self.origin + DVec3::new(b[0] as f64, b[1] as f64, b[2] as f64) * (BRICK as f64 * self.cell_size)
    }

    /// Centre of the box spanned by brick `b`'s samples.
    #[inline]
    pub fn brick_center(&self, b: [u32; 3]) -> DVec3 {
        self.brick_origin(b) + DVec3::splat(3.5 * self.cell_size)
    }

    /// Half-diagonal of the box spanned by one brick's samples.
    #[inline]
    pub fn brick_radius(&self) -> f64 {
        3.5 * 3f64.sqrt() * self.cell_size
    }

    /// Brick containing world position `p` (may be out of range).
    #[inline]
    pub fn brick_of(&self, p: DVec3) -> [i64; 3] {
        let q = (p - self.origin) / (BRICK as f64 * self.cell_size);
        [q.x.floor() as i64, q.y.floor() as i64, q.z.floor() as i64]
    }

    #[inline]
    pub(crate) fn in_range(&self, b: [i64; 3]) -> bool {
        (0..3).all(|a| b[a] >= 0 && b[a] < self.bricks[a] as i64)
    }
}

/// One bit per brick, stored in two levels: 16³-brick top blocks are either uniform
/// (no storage) or hold a 4096-bit table. Memory therefore scales with the number of
/// mixed top blocks (near the surface), not with the volume.
#[derive(Clone, Debug)]
pub(crate) struct BitGrid {
    dims: [u32; 3],
    tdims: [u32; 3],
    tops: Vec<Top>,
}

#[derive(Clone, Debug)]
enum Top {
    Uniform(bool),
    Bits(Box<[u64; 64]>),
}

const TOP_SHIFT: u32 = 4;
const TOP: u32 = 1 << TOP_SHIFT;

impl BitGrid {
    pub fn new(dims: [u32; 3]) -> Self {
        let tdims = dims.map(|d| d.div_ceil(TOP).max(1));
        let n = tdims.iter().map(|&d| d as usize).product::<usize>();
        Self { dims, tdims, tops: vec![Top::Uniform(false); n] }
    }
    #[inline]
    fn top_index(&self, b: [u32; 3]) -> usize {
        let t = b.map(|x| (x >> TOP_SHIFT) as usize);
        t[0] + self.tdims[0] as usize * (t[1] + self.tdims[1] as usize * t[2])
    }
    #[inline]
    fn local(b: [u32; 3]) -> usize {
        let m = TOP - 1;
        ((b[0] & m) + TOP * ((b[1] & m) + TOP * (b[2] & m))) as usize
    }
    /// Out-of-range coordinates read as `false`.
    #[inline]
    pub fn get(&self, b: [i64; 3]) -> bool {
        if (0..3).any(|a| b[a] < 0 || b[a] >= self.dims[a] as i64) {
            return false;
        }
        let b = [b[0] as u32, b[1] as u32, b[2] as u32];
        match &self.tops[self.top_index(b)] {
            Top::Uniform(v) => *v,
            Top::Bits(bits) => {
                let l = Self::local(b);
                (bits[l >> 6] >> (l & 63)) & 1 == 1
            }
        }
    }
    #[inline]
    pub fn set(&mut self, b: [u32; 3]) {
        let ti = self.top_index(b);
        let l = Self::local(b);
        let top = &mut self.tops[ti];
        if let Top::Uniform(false) = top {
            *top = Top::Bits(Box::new([0; 64]));
        }
        if let Top::Bits(bits) = top {
            bits[l >> 6] |= 1 << (l & 63);
        }
    }
    #[inline]
    pub fn clear(&mut self, b: [u32; 3]) {
        let ti = self.top_index(b);
        let l = Self::local(b);
        let top = &mut self.tops[ti];
        if let Top::Uniform(true) = top {
            *top = Top::Bits(Box::new([u64::MAX; 64]));
        }
        if let Top::Bits(bits) = top {
            bits[l >> 6] &= !(1 << (l & 63));
        }
    }
    /// Set every brick of the cube `[o, o + s)³` (clipped to the grid).
    pub fn set_block(&mut self, o: [u32; 3], s: u32) {
        if s == TOP && o.iter().all(|&x| x % TOP == 0) {
            let ti = self.top_index(o);
            self.tops[ti] = Top::Uniform(true);
            return;
        }
        for z in o[2]..(o[2] + s).min(self.dims[2]) {
            for y in o[1]..(o[1] + s).min(self.dims[1]) {
                for x in o[0]..(o[0] + s).min(self.dims[0]) {
                    self.set([x, y, z]);
                }
            }
        }
    }
    pub fn bytes(&self) -> usize {
        self.tops.len() * std::mem::size_of::<Top>()
            + self.tops.iter().filter(|t| matches!(t, Top::Bits(_))).count() * 512
    }
}

/// State of a brick.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Slot {
    Active(u32),
    Uniform(bool),
}

/// Sparse brick grid of quantized field values (negative = inside).
#[derive(Clone, Debug)]
pub struct SparseField {
    grid: GridSpec,
    far: f64,
    index: FastMap<u64, u32>,
    coords: Vec<[u32; 3]>,
    values: Vec<[i16; BRICK_VOL]>,
    background: BitGrid,
}

impl SparseField {
    /// Lattice description.
    pub fn grid(&self) -> &GridSpec {
        &self.grid
    }
    /// Clamp magnitude: stored values lie in `[-far, far]`.
    pub fn far(&self) -> f64 {
        self.far
    }
    /// Number of bricks holding explicit values.
    pub fn active_bricks(&self) -> usize {
        self.coords.len()
    }
    /// Approximate heap memory, in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.values.len() * std::mem::size_of::<[i16; BRICK_VOL]>()
            + self.coords.len() * (12 + 16 + 4)
            + self.background.bytes()
    }
    /// Coordinates of the active bricks.
    pub fn active_brick_coords(&self) -> &[[u32; 3]] {
        &self.coords
    }

    #[inline]
    pub(crate) fn slot(&self, b: [i64; 3]) -> Slot {
        if !self.grid.in_range(b) {
            return Slot::Uniform(false);
        }
        let k = key([b[0] as u32, b[1] as u32, b[2] as u32]);
        match self.index.get(&k) {
            Some(&i) => Slot::Active(i),
            None => Slot::Uniform(self.background.get(b)),
        }
    }

    #[inline]
    pub(crate) fn brick_values(&self, i: u32) -> &[i16; BRICK_VOL] {
        &self.values[i as usize]
    }

    #[inline]
    pub(crate) fn brick_coord(&self, i: u32) -> [u32; 3] {
        self.coords[i as usize]
    }

    /// Quantized value at sample `i` (out-of-range samples are outside).
    #[inline]
    pub(crate) fn sample_q(&self, i: [i64; 3]) -> i16 {
        let b = [i[0].div_euclid(8), i[1].div_euclid(8), i[2].div_euclid(8)];
        match self.slot(b) {
            Slot::Active(s) => {
                let l = lidx(i[0].rem_euclid(8) as usize, i[1].rem_euclid(8) as usize, i[2].rem_euclid(8) as usize);
                self.values[s as usize][l]
            }
            Slot::Uniform(inside) => {
                if inside {
                    -QMAX
                } else {
                    QMAX
                }
            }
        }
    }

    /// Field value at sample `i` (negative inside).
    pub fn sample(&self, i: [i64; 3]) -> f64 {
        self.sample_q(i) as f64 * self.far / QMAX as f64
    }

    /// Trilinearly interpolated field value at a world position.
    pub fn eval_trilinear(&self, p: DVec3) -> f64 {
        let q = (p - self.grid.origin) / self.grid.cell_size;
        let f = q.floor();
        let t = q - f;
        let i0 = [f.x as i64, f.y as i64, f.z as i64];
        let mut acc = 0.0;
        for c in 0..8 {
            let o = [(c & 1) as i64, ((c >> 1) & 1) as i64, ((c >> 2) & 1) as i64];
            let w = (if o[0] == 1 { t.x } else { 1.0 - t.x })
                * (if o[1] == 1 { t.y } else { 1.0 - t.y })
                * (if o[2] == 1 { t.z } else { 1.0 - t.z });
            if w != 0.0 {
                acc += w * self.sample([i0[0] + o[0], i0[1] + o[1], i0[2] + o[2]]);
            }
        }
        acc
    }

    /// Copy the 9³ samples `8b + [0, 9)³` into `out` (index `x + 9y + 81z`).
    pub(crate) fn gather9(&self, b: [u32; 3], out: &mut [i16; 729]) {
        for o in 0..8u32 {
            let ox = (o & 1) as usize;
            let oy = ((o >> 1) & 1) as usize;
            let oz = ((o >> 2) & 1) as usize;
            let nb = [b[0] as i64 + ox as i64, b[1] as i64 + oy as i64, b[2] as i64 + oz as i64];
            let xs = if ox == 0 { 0..8 } else { 0..1 };
            let ys = if oy == 0 { 0..8 } else { 0..1 };
            let zs = if oz == 0 { 0..8 } else { 0..1 };
            match self.slot(nb) {
                Slot::Active(s) => {
                    let v = &self.values[s as usize];
                    for z in zs {
                        for y in ys.clone() {
                            for x in xs.clone() {
                                out[(x + 8 * ox) + 9 * (y + 8 * oy) + 81 * (z + 8 * oz)] = v[lidx(x, y, z)];
                            }
                        }
                    }
                }
                Slot::Uniform(inside) => {
                    let q = if inside { -QMAX } else { QMAX };
                    for z in zs {
                        for y in ys.clone() {
                            for x in xs.clone() {
                                out[(x + 8 * ox) + 9 * (y + 8 * oy) + 81 * (z + 8 * oz)] = q;
                            }
                        }
                    }
                }
            }
        }
    }
}

/// A scalar field that can be sampled into a [`SparseField`] (negative = inside).
pub trait FieldSource: Sync {
    /// Field value at `p`, clamped to `[-far, far]`. Must be exact (or as accurate as
    /// possible) wherever `|f| < far`.
    fn eval(&self, p: DVec3) -> f64;
    /// Clamp magnitude.
    fn far(&self) -> f64;
    /// Bricks that must hold explicit values (typically every brick within the narrow band
    /// of the zero set). Out-of-range coordinates are ignored.
    fn seed_bricks(&self, grid: &GridSpec) -> Vec<[u32; 3]>;
    /// `Some(inside)` if the field certainly has a single sign (and |f| ≥ far) in the ball
    /// `(center, radius)`; `None` if unknown. Used to classify large empty regions without
    /// sampling them.
    fn uniform_sign(&self, center: DVec3, radius: f64) -> Option<bool> {
        let _ = (center, radius);
        None
    }
    /// Evaluate all 512 samples of a brick whose first sample is at `origin`. The default
    /// calls [`FieldSource::eval`] per sample.
    fn eval_brick(&self, origin: DVec3, cell_size: f64, out: &mut [f64; BRICK_VOL]) {
        for z in 0..8 {
            for y in 0..8 {
                for x in 0..8 {
                    out[lidx(x, y, z)] = self.eval(origin + DVec3::new(x as f64, y as f64, z as f64) * cell_size);
                }
            }
        }
    }
}

/// Offsets of the 26 neighbours, with the local sample indices of the layer facing each.
struct Layers {
    offsets: Vec<[i64; 3]>,
    layers: Vec<Vec<u16>>,
}

impl Layers {
    fn new() -> Self {
        let mut offsets = Vec::with_capacity(26);
        let mut layers = Vec::with_capacity(26);
        for oz in -1i64..=1 {
            for oy in -1i64..=1 {
                for ox in -1i64..=1 {
                    if ox == 0 && oy == 0 && oz == 0 {
                        continue;
                    }
                    let o = [ox, oy, oz];
                    let mut l = Vec::new();
                    for z in 0..8 {
                        for y in 0..8 {
                            for x in 0..8 {
                                let c = [x as i64, y as i64, z as i64];
                                let ok = (0..3).all(|a| match o[a] {
                                    1 => c[a] == 7,
                                    -1 => c[a] == 0,
                                    _ => true,
                                });
                                if ok {
                                    l.push(lidx(x, y, z) as u16);
                                }
                            }
                        }
                    }
                    offsets.push(o);
                    layers.push(l);
                }
            }
        }
        Self { offsets, layers }
    }
}

enum Cls {
    Single([u32; 3], bool),
    Uniform([u32; 3], u32, bool),
    Split([u32; 3], u32),
}

const TOP_BLOCK: u32 = TOP;
const EVAL_CHUNK: usize = 1024;

/// Sample `src` on `grid` into a sparse field (see the module docs for the algorithm).
pub fn build_field<S: FieldSource + ?Sized>(src: &S, grid: GridSpec, progress: Progress) -> Result<SparseField> {
    build_field_with(src, grid, &Prog::new(progress))
}

pub(crate) fn build_field_with<S: FieldSource + ?Sized>(src: &S, grid: GridSpec, prog: &Prog) -> Result<SparseField> {
    let far = src.far();
    if !(far > 0.0) || !far.is_finite() {
        return Err(SolidError::InvalidParameter(format!("field clamp value {far}")));
    }
    let nb = grid.bricks;
    let h = grid.cell_size;

    // 1. Seeds.
    let mut seeds: Vec<u64> = src
        .seed_bricks(&grid)
        .into_iter()
        .filter(|b| (0..3).all(|a| b[a] < nb[a]))
        .map(key)
        .collect();
    seeds.par_sort_unstable();
    seeds.dedup();
    let mut active_bits = BitGrid::new(nb);
    for &k in &seeds {
        active_bits.set(unkey(k));
    }
    prog.report(0.02)?;

    // 2. Hierarchical background classification.
    let mut background = BitGrid::new(nb);
    let mut singles: Vec<[u32; 3]> = Vec::new();
    let top = TOP_BLOCK.min(nb.iter().copied().max().unwrap_or(1).next_power_of_two());
    let mut pending: Vec<([u32; 3], u32)> = Vec::new();
    for z in (0..nb[2]).step_by(top as usize) {
        for y in (0..nb[1]).step_by(top as usize) {
            for x in (0..nb[0]).step_by(top as usize) {
                pending.push(([x, y, z], top));
            }
        }
    }
    let at_boundary = |o: [u32; 3], s: u32| (0..3).any(|a| o[a] == 0 || o[a] + s >= nb[a]);
    let mut boundary_inside: Vec<u64> = Vec::new();
    let mut uniform_blocks: Vec<([u32; 3], u32, bool)> = Vec::new();
    while !pending.is_empty() {
        let res: Vec<Cls> = pending
            .par_iter()
            .map(|&(o, s)| {
                if s == 1 {
                    if active_bits.get([o[0] as i64, o[1] as i64, o[2] as i64]) {
                        return Cls::Single(o, false);
                    }
                    return Cls::Single(o, src.eval(grid.brick_center(o)) < 0.0);
                }
                let c = grid.brick_origin(o) + DVec3::splat((BRICK * s - 1) as f64 * 0.5 * h);
                let r = (BRICK * s - 1) as f64 * 0.5 * h * 3f64.sqrt();
                match src.uniform_sign(c, r) {
                    // Inside blocks touching the domain boundary are resolved per brick.
                    Some(true) if at_boundary(o, s) => Cls::Split(o, s),
                    Some(inside) => Cls::Uniform(o, s, inside),
                    None => Cls::Split(o, s),
                }
            })
            .collect();
        let mut next = Vec::new();
        for r in res {
            match r {
                Cls::Single(o, inside) => {
                    if inside {
                        background.set(o);
                        if at_boundary(o, 1) {
                            boundary_inside.push(key(o));
                        }
                    }
                    singles.push(o);
                }
                Cls::Uniform(o, s, inside) => {
                    if inside {
                        background.set_block(o, s);
                    }
                    uniform_blocks.push((o, s, inside));
                }
                Cls::Split(o, s) => {
                    let hs = s / 2;
                    for c in 0..8u32 {
                        let co = [o[0] + (c & 1) * hs, o[1] + ((c >> 1) & 1) * hs, o[2] + ((c >> 2) & 1) * hs];
                        if (0..3).all(|a| co[a] < nb[a]) {
                            next.push((co, hs));
                        }
                    }
                }
            }
        }
        pending = next;
        prog.report(0.05)?;
    }

    // 2b. `uniform_sign` is a heuristic: a uniform block bordering an inactive brick of the
    // other sign would break the extraction invariant (and nothing else would notice when
    // the neighbour is another uniform block), so such blocks are re-classified per brick.
    loop {
        let bg = &background;
        let ab = &active_bits;
        let conflict: Vec<bool> = uniform_blocks
            .par_iter()
            .map(|&(o, s, inside)| {
                let lo = o.map(|x| x as i64 - 1);
                let hi = [o[0] + s, o[1] + s, o[2] + s].map(|x| x as i64);
                for z in lo[2]..=hi[2] {
                    for y in lo[1]..=hi[1] {
                        let interior_row = z > lo[2] && z < hi[2] && y > lo[1] && y < hi[1];
                        let mut x = lo[0];
                        while x <= hi[0] {
                            let n = [x, y, z];
                            if grid.in_range(n) && !ab.get(n) && bg.get(n) != inside {
                                return true;
                            }
                            // Skip the block interior.
                            x = if interior_row && x == lo[0] { hi[0] } else { x + 1 };
                        }
                    }
                }
                false
            })
            .collect();
        if !conflict.iter().any(|&c| c) {
            break;
        }
        let mut refine: Vec<[u32; 3]> = Vec::new();
        let mut keep = Vec::with_capacity(uniform_blocks.len());
        for (blk, c) in uniform_blocks.into_iter().zip(conflict) {
            if c {
                let (o, s, _) = blk;
                for z in o[2]..(o[2] + s).min(nb[2]) {
                    for y in o[1]..(o[1] + s).min(nb[1]) {
                        for x in o[0]..(o[0] + s).min(nb[0]) {
                            refine.push([x, y, z]);
                        }
                    }
                }
            } else {
                keep.push(blk);
            }
        }
        uniform_blocks = keep;
        let signs: Vec<bool> = refine.par_iter().map(|&o| src.eval(grid.brick_center(o)) < 0.0).collect();
        for (o, inside) in refine.into_iter().zip(signs) {
            if inside {
                background.set(o);
                if at_boundary(o, 1) {
                    boundary_inside.push(key(o));
                }
            } else {
                background.clear(o);
            }
            singles.push(o);
        }
    }
    drop(uniform_blocks);

    // 3. Individually classified bricks that disagree with an inactive neighbour.
    let layers = Layers::new();
    let mut extra: Vec<u64> = singles
        .par_iter()
        .filter(|&&o| {
            let b = [o[0] as i64, o[1] as i64, o[2] as i64];
            if active_bits.get(b) {
                return false;
            }
            let s = background.get(b);
            layers.offsets.iter().any(|d| {
                let n = [b[0] + d[0], b[1] + d[1], b[2] + d[2]];
                grid.in_range(n) && !active_bits.get(n) && background.get(n) != s
            })
        })
        .map(|&o| key(o))
        .collect();
    extra.extend(boundary_inside);
    seeds.extend(extra);
    seeds.par_sort_unstable();
    seeds.dedup();
    drop(singles);

    // 4. Evaluate active bricks, then propagate until consistent.
    let mut field = SparseField {
        grid,
        far,
        index: FastMap::default(),
        coords: Vec::new(),
        values: Vec::new(),
        background,
    };
    field.index.reserve(seeds.len() * 2);
    let initial = seeds.len().max(1) as f32;
    let mut batch = seeds;
    let mut checked = 0usize;
    loop {
        for k in &batch {
            active_bits.set(unkey(*k));
        }
        for chunk in batch.chunks(EVAL_CHUNK) {
            let vals: Vec<[i16; BRICK_VOL]> = chunk.par_iter().map(|&k| eval_brick(src, &field.grid, unkey(k), far)).collect();
            for (&k, v) in chunk.iter().zip(vals) {
                field.index.insert(k, field.coords.len() as u32);
                field.coords.push(unkey(k));
                field.values.push(v);
            }
            // Propagation adds bricks beyond the initial estimate; approach 1 asymptotically.
            let done = field.coords.len() as f32;
            prog.report(0.05 + 0.95 * done / (initial + 0.05 * done).max(done))?;
        }
        let fref = &field;
        let ab = &active_bits;
        let mut new: Vec<u64> = (checked..field.coords.len())
            .into_par_iter()
            .flat_map_iter(|i| {
                let c = fref.coords[i];
                let v = &fref.values[i];
                let b = [c[0] as i64, c[1] as i64, c[2] as i64];
                let mut out: Vec<u64> = Vec::new();
                for (d, layer) in layers.offsets.iter().zip(&layers.layers) {
                    let n = [b[0] + d[0], b[1] + d[1], b[2] + d[2]];
                    if !fref.grid.in_range(n) || ab.get(n) {
                        continue;
                    }
                    let inside = fref.background.get(n);
                    if layer.iter().any(|&l| (v[l as usize] < 0) != inside) {
                        out.push(key([n[0] as u32, n[1] as u32, n[2] as u32]));
                    }
                }
                out.into_iter()
            })
            .collect();
        checked = field.coords.len();
        new.par_sort_unstable();
        new.dedup();
        if new.is_empty() {
            break;
        }
        batch = new;
    }
    prog.report(1.0)?;
    Ok(field)
}

fn eval_brick<S: FieldSource + ?Sized>(src: &S, grid: &GridSpec, b: [u32; 3], far: f64) -> [i16; BRICK_VOL] {
    let mut f = [0.0f64; BRICK_VOL];
    src.eval_brick(grid.brick_origin(b), grid.cell_size, &mut f);
    let n = grid.samples();
    let mut out = [0i16; BRICK_VOL];
    for z in 0..8u32 {
        for y in 0..8u32 {
            for x in 0..8u32 {
                let g = [b[0] * 8 + x, b[1] * 8 + y, b[2] * 8 + z];
                let l = lidx(x as usize, y as usize, z as usize);
                // The outermost sample layer is forced outside so the surface always closes.
                out[l] = if (0..3).any(|a| g[a] == 0 || g[a] + 1 >= n[a]) { QMAX } else { quantize(f[l], far) };
            }
        }
    }
    out
}
