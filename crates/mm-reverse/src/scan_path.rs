// SPDX-License-Identifier: AGPL-3.0-only
//! A path along the scan's surface between two points on it.
//!
//! Ported from ScanRuler (`src/core/geometry/scanPath.ts`), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only.
//!
//! The path is a plane section: the plane through the two points that holds the scan's mean
//! normal there, cut through the triangles in a tube round the segment between them, and of
//! the section's chains the piece that runs from the one point to the other. On a smooth
//! surface that is the curve a person means by "from here to there along the part": a
//! straight line on a plane, an arc of a circle across a cylinder, a great circle on a
//! sphere. And it is cheap: only the triangles in the tube are cut. Each point of the section
//! lies on a triangle edge, and the path says which, so what is on either side of it can be
//! told by the triangles.
//!
//! Where the section does not join the two points (a hole in the scan between them, a fold
//! the plane leaves by another sheet) the path falls back to the shortest way along the
//! mesh's edges (A*, the edges' lengths their weights), which goes round a hole rather than
//! across it.
//!
//! The scan is prepared once ([`ScanPaths`]): its triangles in a grid by place, for the tube
//! and the nearest point; the vertices' adjacency, which of them are joined at all and the
//! search's arrays for the fallback, made the first time it is needed. The viewport traces a
//! path to the cursor with it, a worker with the same points gets the same path.
//!
//! Positions are kept as `f32`, as ScanRuler keeps them in a `Float32Array`, and every sum is
//! made in `f64` from them, so the numbers are the ones ScanRuler computes.

use std::any::{Any, TypeId};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use glam::DVec3;

use crate::adjacency::{build_adjacency, Adjacency};

/// How a point of a path lies on the mesh, as its `on` pair says: on the edge between
/// vertices i < j, at vertex v (v, v), or inside triangle t (-1, t), which is what a path's
/// own ends are.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathPlace {
    Edge(u32, u32),
    Vertex(u32),
    Triangle(u32),
}

/// Which way a path was found: a plane section, or the fallback along the mesh's edges.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PathWay {
    Section,
    Edges,
}

/// A path along the scan.
#[derive(Debug, Clone, PartialEq)]
pub struct ScanPath {
    /// x, y, z per point, from the first point to the second.
    pub points: Vec<f32>,
    /// Two numbers per point: where it lies on the mesh (see [`PathPlace`] and [`place_of`]).
    pub on: Vec<i32>,
    /// Along the path, mm.
    pub length: f64,
    /// A plane section, or the fallback along the mesh's edges.
    pub way: PathWay,
    /// The triangles the two ends lie in; -1 for none (an empty scan).
    pub ends: [i32; 2],
}

/// The nearest point of the scan to a point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScanPoint {
    pub point: DVec3,
    pub triangle: u32,
    /// The triangle's own normal.
    pub normal: DVec3,
    pub distance: f64,
}

/// How many triangles a grid cell holds on average, about.
const PER_CELL: f64 = 2.0;
/// The most cells the grid is allowed, whatever the triangles.
const MAX_CELLS: f64 = 2_000_000.0;
/// The tube's radius about the segment between the points: this share of the segment's
/// length, and so many edges more; enough for the arc of a quarter of a cylinder between
/// two points on it.
const TUBE_SHARE: f64 = 0.75;
const TUBE_EDGES: f64 = 3.0;
/// The radius the scan's normal at a point is averaged over, in edges.
const NORMAL_EDGES: f64 = 3.0;
/// How far a path's end may lie off the section, in edges, for the section to be taken as
/// passing through it.
const END_SLACK_EDGES: f64 = 0.5;

/// The stamps that keep a grid query from giving a triangle twice.
struct Stamps {
    stamp: Vec<u32>,
    query: u32,
}

/// What [`ScanPaths::edge_route`] keeps between searches: per vertex, its piece of the mesh,
/// the way's length to it, the vertex it was come to from, its stamp (seen or done with, by
/// which search) and the vertex taken up when it was last looked at, by count; the heap, and
/// the stamps' counters.
struct EdgeSearch {
    piece: Vec<i32>,
    g: Vec<f64>,
    back: Vec<i32>,
    state: Vec<u32>,
    mark: Vec<u32>,
    heap: Heap,
    query: u32,
    expansion: u32,
}

/// A scan prepared for paths: see the module documentation.
///
/// The queries take `&self`: the grid's stamps and the edge search's arrays sit behind
/// locks, made the first time they are needed, so a prepared scan can be shared.
pub struct ScanPaths {
    positions: Vec<f32>,
    indices: Vec<u32>,
    triangle_count: usize,
    /// The mesh's mean edge, mm: what the tolerances count in.
    edge: f64,
    min: [f64; 3],
    cell: f64,
    dims: [usize; 3],
    offsets: Vec<u32>,
    list: Vec<u32>,
    /// Per cell, the largest side of the boxes of the triangles in it: how far past the cell
    /// they reach; per column of cells along z, and per slab of them across x, the largest of
    /// their cells'.
    reach: Vec<f32>,
    column_reach: Vec<f32>,
    slab_reach: Vec<f32>,
    stamps: Mutex<Stamps>,
    adjacency: OnceLock<Adjacency>,
    search: Mutex<Option<EdgeSearch>>,
    /// What other modules keep for this scan, made once: ScanRuler keeps these in
    /// `WeakMap`s keyed by the scan (the cells' incidence, the network reads' cache).
    memo: Mutex<HashMap<TypeId, Arc<dyn Any + Send + Sync>>>,
}

impl std::fmt::Debug for ScanPaths {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScanPaths")
            .field("vertices", &(self.positions.len() / 3))
            .field("triangles", &self.triangle_count)
            .field("edge", &self.edge)
            .field("dims", &self.dims)
            .finish()
    }
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl ScanPaths {
    /// The scan prepared: `positions` x, y, z per vertex, `indices` three a triangle. A
    /// triangle naming a vertex past the positions is left out (ScanRuler's typed arrays
    /// never hold one), and so are indices past the last whole triangle.
    pub fn new(positions: Vec<f32>, indices: Vec<u32>) -> Self {
        let vertex_count = positions.len() / 3;
        let mut indices = indices;
        indices.truncate(indices.len() / 3 * 3);
        if indices.iter().any(|&v| v as usize >= vertex_count) {
            indices = indices.chunks_exact(3).filter(|t| t.iter().all(|&v| (v as usize) < vertex_count)).flatten().copied().collect();
        }
        let triangle_count = indices.len() / 3;
        let p = &positions;
        let mut min = [f64::INFINITY; 3];
        let mut max = [f64::NEG_INFINITY; 3];
        for i in (0..vertex_count * 3).step_by(3) {
            for k in 0..3 {
                let x = p[i + k] as f64;
                if x < min[k] {
                    min[k] = x;
                }
                if x > max[k] {
                    max[k] = x;
                }
            }
        }
        if !(min[0] <= max[0]) {
            min = [0.0; 3];
            max = [0.0; 3];
        }
        let valid = |t: usize| indices[t * 3..t * 3 + 3].iter().all(|&v| (v as usize) < vertex_count);
        let at = |v: u32, k: usize| p[v as usize * 3 + k] as f64;
        // The mean edge, from a sample of the triangles.
        let step = (triangle_count / 5000).max(1);
        let mut sum = 0.0;
        let mut count = 0usize;
        let mut t = 0;
        while t < triangle_count {
            if valid(t) {
                let (a, b, c) = (indices[t * 3], indices[t * 3 + 1], indices[t * 3 + 2]);
                let d = |u: u32, w: u32| {
                    hypot3(at(w, 0) - at(u, 0), at(w, 1) - at(u, 1), at(w, 2) - at(u, 2))
                };
                sum += d(a, b) + d(b, c) + d(c, a);
                count += 3;
            }
            t += step;
        }
        let edge = if count > 0 { sum / count as f64 } else { 1.0 };
        // Cells a couple of edges across, as many as the cap allows.
        let span = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
        let mut cell = (edge * PER_CELL.sqrt()).max(1e-6);
        let cells_at = |h: f64| {
            ((span[0] / h).floor() + 1.0) * ((span[1] / h).floor() + 1.0) * ((span[2] / h).floor() + 1.0)
        };
        while cells_at(cell) > MAX_CELLS {
            cell *= 1.25;
        }
        let dims = [
            (span[0] / cell).floor() as usize + 1,
            (span[1] / cell).floor() as usize + 1,
            (span[2] / cell).floor() as usize + 1,
        ];
        let cells = dims[0] * dims[1] * dims[2];
        // Two passes: count each triangle into the cells its box touches, then place it, and
        // note how far past each cell its triangles reach, a hair over so that the float's
        // rounding never takes it under (triangles_crossing counts on it).
        let box_of = |t: usize| -> Option<([usize; 6], f64)> {
            if !valid(t) {
                return None;
            }
            let (a, b, c) = (indices[t * 3], indices[t * 3 + 1], indices[t * 3 + 2]);
            let mut out = [0usize; 6];
            let mut side = 0.0f64;
            for k in 0..3 {
                let lo = at(a, k).min(at(b, k)).min(at(c, k));
                let hi = at(a, k).max(at(b, k)).max(at(c, k));
                let l = ((lo - min[k]) / cell).floor();
                let h = ((hi - min[k]) / cell).floor();
                if l.is_nan() || h.is_nan() {
                    return None;
                }
                out[k] = (l.max(0.0) as usize).min(dims[k] - 1);
                out[k + 3] = (h.max(0.0) as usize).min(dims[k] - 1);
                side = side.max(hi - lo);
            }
            Some((out, side))
        };
        let cell_id = |x: usize, y: usize, z: usize| (x * dims[1] + y) * dims[2] + z;
        let mut offsets = vec![0u32; cells + 1];
        for t in 0..triangle_count {
            let Some((b, _)) = box_of(t) else { continue };
            for x in b[0]..=b[3] {
                for y in b[1]..=b[4] {
                    for z in b[2]..=b[5] {
                        offsets[cell_id(x, y, z) + 1] += 1;
                    }
                }
            }
        }
        for i in 0..cells {
            offsets[i + 1] += offsets[i];
        }
        let mut list = vec![0u32; offsets[cells] as usize];
        let mut cursor: Vec<u32> = offsets[..cells].to_vec();
        let mut reach = vec![0f32; cells];
        for t in 0..triangle_count {
            let Some((b, side)) = box_of(t) else { continue };
            let side = side * (1.0 + 1e-6);
            for x in b[0]..=b[3] {
                for y in b[1]..=b[4] {
                    for z in b[2]..=b[5] {
                        let id = cell_id(x, y, z);
                        list[cursor[id] as usize] = t as u32;
                        cursor[id] += 1;
                        if side > reach[id] as f64 {
                            reach[id] = side as f32;
                        }
                    }
                }
            }
        }
        let [nx, ny, nz] = dims;
        let mut column_reach = vec![0f32; nx * ny];
        let mut slab_reach = vec![0f32; nx];
        for x in 0..nx {
            for y in 0..ny {
                let col = x * ny + y;
                let mut r = 0f32;
                for z in 0..nz {
                    r = r.max(reach[col * nz + z]);
                }
                column_reach[col] = r;
                slab_reach[x] = slab_reach[x].max(r);
            }
        }
        ScanPaths {
            stamps: Mutex::new(Stamps { stamp: vec![0; triangle_count], query: 0 }),
            positions,
            indices,
            triangle_count,
            edge,
            min,
            cell,
            dims,
            offsets,
            list,
            reach,
            column_reach,
            slab_reach,
            adjacency: OnceLock::new(),
            search: Mutex::new(None),
            memo: Mutex::new(HashMap::new()),
        }
    }

    /// A Polymix mesh prepared for paths: its live vertices and triangles, compacted (the
    /// mesh's free lists left out), in the order the mesh lists them.
    pub fn from_mesh(mesh: &mm_core::mesh::Mesh) -> Self {
        let mut local = vec![u32::MAX; mesh.max_vid()];
        let mut positions = Vec::with_capacity(mesh.vertex_count() * 3);
        for (n, v) in mesh.vertex_ids().enumerate() {
            local[v as usize] = n as u32;
            let p = mesh.vertex(v);
            positions.extend_from_slice(&[p.x as f32, p.y as f32, p.z as f32]);
        }
        let mut indices = Vec::with_capacity(mesh.triangle_count() * 3);
        for t in mesh.triangle_ids() {
            let tv = mesh.triangle(t);
            if tv.iter().all(|&v| (v as usize) < local.len() && local[v as usize] != u32::MAX) {
                indices.extend(tv.iter().map(|&v| local[v as usize]));
            }
        }
        ScanPaths::new(positions, indices)
    }

    /// x, y, z per vertex.
    pub fn positions(&self) -> &[f32] {
        &self.positions
    }

    /// Three vertices a triangle.
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    pub fn triangle_count(&self) -> usize {
        self.triangle_count
    }

    pub fn vertex_count(&self) -> usize {
        self.positions.len() / 3
    }

    /// The mesh's mean edge, mm: what the tolerances count in.
    pub fn edge(&self) -> f64 {
        self.edge
    }

    /// Vertex `v`'s position.
    pub fn vertex(&self, v: u32) -> DVec3 {
        let i = v as usize * 3;
        DVec3::new(self.positions[i] as f64, self.positions[i + 1] as f64, self.positions[i + 2] as f64)
    }

    /// Triangle `t`'s three vertices.
    pub fn triangle(&self, t: u32) -> [u32; 3] {
        let i = t as usize * 3;
        [self.indices[i], self.indices[i + 1], self.indices[i + 2]]
    }

    /// What a module keeps for this scan, made by `make` the first time it is asked for
    /// (by its type); as ScanRuler's `WeakMap`s keyed by the scan.
    pub fn memo<T: Any + Send + Sync>(&self, make: impl FnOnce(&ScanPaths) -> T) -> Arc<T> {
        let key = TypeId::of::<T>();
        if let Some(had) = lock(&self.memo).get(&key) {
            if let Ok(t) = had.clone().downcast::<T>() {
                return t;
            }
        }
        let made: Arc<T> = Arc::new(make(self));
        let mut map = lock(&self.memo);
        let entry = map.entry(key).or_insert_with(|| made.clone() as Arc<dyn Any + Send + Sync>);
        entry.clone().downcast::<T>().unwrap_or(made)
    }

    /// The cell range [c0, c1] the box [lo, hi] covers, each way.
    fn cell_range(&self, lo: DVec3, hi: DVec3) -> ([i64; 3], [i64; 3]) {
        let mut c0 = [0i64; 3];
        let mut c1 = [0i64; 3];
        for k in 0..3 {
            let a = ((lo[k] - self.min[k]) / self.cell).floor();
            let b = ((hi[k] - self.min[k]) / self.cell).floor();
            // A NaN gives no cells, as the loops over it in ScanRuler run none.
            if a.is_nan() || b.is_nan() {
                return ([0; 3], [-1; 3]);
            }
            c0[k] = a.clamp(-1.0, 4e18).max(0.0) as i64;
            c1[k] = b.clamp(-1.0, 4e18).min((self.dims[k] - 1) as f64) as i64;
        }
        (c0, c1)
    }

    fn next_query(stamps: &mut Stamps) -> u32 {
        stamps.query = stamps.query.wrapping_add(1);
        if stamps.query == u32::MAX {
            stamps.stamp.iter_mut().for_each(|s| *s = 0);
            stamps.query = 1;
        }
        stamps.query
    }

    /// Every triangle whose box may meet the box [lo, hi], each once.
    pub fn triangles_in(&self, lo: DVec3, hi: DVec3) -> Vec<u32> {
        let mut out = Vec::new();
        let mut guard = lock(&self.stamps);
        let stamps = &mut *guard;
        let q = Self::next_query(stamps);
        let (c0, c1) = self.cell_range(lo, hi);
        let (ny, nz) = (self.dims[1] as i64, self.dims[2] as i64);
        for x in c0[0]..=c1[0] {
            for y in c0[1]..=c1[1] {
                for z in c0[2]..=c1[2] {
                    let id = ((x * ny + y) * nz + z) as usize;
                    for i in self.offsets[id]..self.offsets[id + 1] {
                        let t = self.list[i as usize];
                        if stamps.stamp[t as usize] == q {
                            continue;
                        }
                        stamps.stamp[t as usize] = q;
                        out.push(t);
                    }
                }
            }
        }
        out
    }

    /// The triangles of [`triangles_in`](Self::triangles_in)`(lo, hi)` that the plane
    /// m·x = d0 crosses (a corner on either side of it, a corner on it counted on the side m
    /// points to), each once and in the order `triangles_in` gives them; found from the cells
    /// the plane passes near, not from every cell of the box. A box n cells wide has n³ of
    /// them, and a plane comes near some n² of those.
    ///
    /// A triangle in a cell overlaps it, and one the plane crosses has a point on the plane;
    /// so that point lies inside the cell grown each way by the triangle's box's largest
    /// side, which is at most the cell's reach. The plane then passes within
    /// (cell/2 + reach)·(|mx| + |my| + |mz|) of the cell's centre, and a cell farther off
    /// holds none of the triangles wanted. The cells are visited x, y, z, as `triangles_in`
    /// visits them, the z steps of each column worked out from the column's reach, the y
    /// steps of each slab from the slab's. Every cell a crossed triangle is in passes, so
    /// each comes out at the first of its cells in the box, as from `triangles_in`.
    pub fn triangles_crossing(&self, lo: DVec3, hi: DVec3, m: DVec3, d0: f64) -> Vec<u32> {
        let mut out = Vec::new();
        let mut guard = lock(&self.stamps);
        let stamps = &mut *guard;
        let q = Self::next_query(stamps);
        let p = &self.positions;
        let idx = &self.indices;
        let (min, cell) = (self.min, self.cell);
        let ny = self.dims[1];
        let nz = self.dims[2];
        let (c0, c1) = self.cell_range(lo, hi);
        let (mx, my, mz) = (m.x, m.y, m.z);
        let l1 = mx.abs() + my.abs() + mz.abs();
        // Half a cell, and a thousandth of one more for the rounding in the cells' indices
        // and in the sums along m.
        let half = cell * (0.5 + 1e-3);
        // m·x along z at the box's first and last cells, for each slab's y steps.
        let z_a = mz * (min[2] + (c0[2] as f64 + 0.5) * cell);
        let z_b = mz * (min[2] + (c1[2] as f64 + 0.5) * cell);
        let z_lo = z_a.min(z_b);
        let z_hi = z_a.max(z_b);
        let side = |v: u32| {
            let i = v as usize * 3;
            mx * p[i] as f64 + my * p[i + 1] as f64 + mz * p[i + 2] as f64 - d0 >= 0.0
        };
        for x in c0[0]..=c1[0] {
            let slab = self.slab_reach[x as usize] as f64;
            if !(slab > 0.0) {
                continue;
            }
            let fx = mx * (min[0] + (x as f64 + 0.5) * cell) - d0;
            let slab_far = (half + slab) * l1;
            let Some((y0, y1)) = steps_within(fx + my * (min[1] + 0.5 * cell), my * cell, -slab_far - z_hi, slab_far - z_lo, c0[1], c1[1]) else {
                continue;
            };
            for y in y0..=y1 {
                let col = x as usize * ny + y as usize;
                let column = self.column_reach[col] as f64;
                if !(column > 0.0) {
                    continue;
                }
                let fy = fx + my * (min[1] + (y as f64 + 0.5) * cell);
                let column_far = (half + column) * l1;
                let Some((z0, z1)) = steps_within(fy + mz * (min[2] + 0.5 * cell), mz * cell, -column_far, column_far, c0[2], c1[2]) else {
                    continue;
                };
                for z in z0..=z1 {
                    let id = col * nz + z as usize;
                    let start = self.offsets[id];
                    let end = self.offsets[id + 1];
                    if start == end || (fy + mz * (min[2] + (z as f64 + 0.5) * cell)).abs() > (half + self.reach[id] as f64) * l1 {
                        continue;
                    }
                    for i in start..end {
                        let t = self.list[i as usize];
                        if stamps.stamp[t as usize] == q {
                            continue;
                        }
                        stamps.stamp[t as usize] = q;
                        // The sign test section_path makes, the sums the same way round.
                        let (a, b, c) = (idx[t as usize * 3], idx[t as usize * 3 + 1], idx[t as usize * 3 + 2]);
                        let (sa, sb, sc) = (side(a), side(b), side(c));
                        if sa != sb || sb != sc {
                            out.push(t);
                        }
                    }
                }
            }
        }
        out
    }

    /// The nearest point of the scan to `p`, or `None` on an empty scan.
    pub fn closest(&self, p: DVec3) -> Option<ScanPoint> {
        if self.triangle_count == 0 || self.list.is_empty() {
            return None;
        }
        let span = self.dims[0].max(self.dims[1]).max(self.dims[2]) as f64 * self.cell;
        let mut near = DVec3::ZERO;
        let mut r = self.cell;
        loop {
            let mut best = f64::INFINITY;
            let mut best_t: Option<u32> = None;
            let mut at = DVec3::ZERO;
            let rr = DVec3::splat(r);
            for t in self.triangles_in(p - rr, p + rr) {
                let d = self.closest_on_triangle(t, p, &mut near);
                if d < best {
                    best = d;
                    best_t = Some(t);
                    at = near;
                }
            }
            // A triangle nearer than r lies in the cube searched; past the whole grid there
            // is no more to search.
            if (best_t.is_some() && best <= r) || r > 2.0 * span + self.distance_to_grid(p) || !r.is_finite() {
                let t = best_t?;
                return Some(ScanPoint { point: at, triangle: t, normal: self.triangle_normal(t), distance: best });
            }
            r *= 2.0;
        }
    }

    /// The nearest point of the scan to `p`, taken a hair inside its triangle where it lies
    /// on the triangle's edge or corner, so that a point on the scan lies in one triangle,
    /// and every path that ends at it ends in that one. A hair is a ten-thousandth of the
    /// triangle's size: tens of nanometres on a scan.
    pub fn closest_inside(&self, p: DVec3) -> Option<ScanPoint> {
        let c = self.closest(p)?;
        let t = c.triangle;
        let [ia, ib, id] = self.triangle(t);
        let (a, b, d) = (self.vertex(ia), self.vertex(ib), self.vertex(id));
        // Barycentric coordinates by areas along the triangle's normal.
        let n = self.triangle_cross(t);
        let nn = n.dot(n);
        if !(nn > 0.0) {
            return Some(c);
        }
        let w = |u: DVec3, v: DVec3| (v - u).cross(c.point - u).dot(n) / nn;
        let bary = [w(b, d), w(d, a), w(a, b)];
        let hair = 1e-4;
        if bary[0].min(bary[1]).min(bary[2]) >= hair {
            return Some(c);
        }
        let held = [bary[0].max(hair), bary[1].max(hair), bary[2].max(hair)];
        let sum = held[0] + held[1] + held[2];
        let point = (held[0] * a + held[1] * b + held[2] * d) / sum;
        Some(ScanPoint { point, distance: (point - p).length(), ..c })
    }

    /// The scan's normal about a point on it: the triangles' normals within a few edges, by
    /// area.
    pub fn normal_at(&self, p: DVec3, triangle: u32) -> DVec3 {
        let r = NORMAL_EDGES * self.edge;
        let mut n = DVec3::ZERO;
        let mut q = DVec3::ZERO;
        let rr = DVec3::splat(r);
        for t in self.triangles_in(p - rr, p + rr) {
            if self.closest_on_triangle(t, p, &mut q) > r {
                continue;
            }
            n += self.triangle_cross(t);
        }
        let l = n.length();
        if l > 0.0 {
            n / l
        } else {
            self.triangle_normal(triangle)
        }
    }

    /// The vertices' adjacency (CSR), made once.
    pub fn neighbours(&self) -> &Adjacency {
        self.adjacency.get_or_init(|| build_adjacency(&self.indices, self.positions.len() / 3))
    }

    /// The vertices of the shortest way along the mesh's edges from vertex `from` to vertex
    /// `to`, both included, or `None` where no edges join them: told at once, from the pieces
    /// of the mesh, rather than by searching the whole of from's piece.
    ///
    /// A*: the vertices are taken in the order of the way to them so far plus the straight
    /// distance on to `to`. That distance is never more than the way left along the edges,
    /// so the first way to reach `to` is a shortest one; and the search covers an ellipse
    /// round the two vertices instead of Dijkstra's disk of everything within the way's
    /// length of `from`. The straight distance is taken a hair under, so that the rounding
    /// never puts it past the way along the edges. The ways' lengths are the edges' summed
    /// from `from` as Dijkstra summed them, so the shortest comes out the same to the last
    /// bit; where two ways to a vertex tie, the one through the vertex nearer `from` is kept,
    /// as Dijkstra, taking the nearer first, kept it. Only where those two are as near as
    /// each other can the way differ from Dijkstra's, and then it is as short.
    pub fn edge_route(&self, from: u32, to: u32) -> Option<Vec<u32>> {
        let count = self.positions.len() / 3;
        if from as usize >= count || to as usize >= count {
            return None;
        }
        let adjacency = self.neighbours();
        let mut guard = lock(&self.search);
        let s = guard.get_or_insert_with(|| self.edge_search(adjacency));
        let (from, to) = (from as usize, to as usize);
        if s.piece[from] != s.piece[to] {
            return None;
        }
        if from == to {
            return Some(vec![from as u32]);
        }
        let p = &self.positions;
        let pos = |v: usize, k: usize| p[v * 3 + k] as f64;
        // Two stamps a search: seen, so that g holds, and done with.
        if s.query >= 0xffff_fff0 {
            s.state.iter_mut().for_each(|x| *x = 0);
            s.query = 0;
        }
        s.query += 2;
        let seen = s.query;
        let done = s.query + 1;
        let (tx, ty, tz) = (pos(to, 0), pos(to, 1), pos(to, 2));
        let hair = 1.0 - 1e-9;
        s.heap.clear();
        s.g[from] = 0.0;
        s.state[from] = seen;
        let (fx, fy, fz) = (pos(from, 0) - tx, pos(from, 1) - ty, pos(from, 2) - tz);
        s.heap.push(from as i32, hair * (fx * fx + fy * fy + fz * fz).sqrt());
        let mut found = false;
        while s.heap.size > 0 {
            let v = s.heap.pop() as usize;
            if s.state[v] == done {
                continue;
            }
            s.state[v] = done;
            if v == to {
                found = true;
                break;
            }
            if s.expansion == u32::MAX {
                s.mark.iter_mut().for_each(|x| *x = 0);
                s.expansion = 0;
            }
            s.expansion += 1;
            let e = s.expansion;
            let gv = s.g[v];
            let (vx, vy, vz) = (pos(v, 0), pos(v, 1), pos(v, 2));
            for &w in adjacency.neighbours(v) {
                let w = w as usize;
                // An edge is listed once for each triangle on it: one look will do.
                if s.mark[w] == e {
                    continue;
                }
                s.mark[w] = e;
                let known = s.state[w] >= seen;
                // No way through v comes to w sooner than one already has.
                if known && s.g[w] <= gv {
                    continue;
                }
                let (wx, wy, wz) = (pos(w, 0), pos(w, 1), pos(w, 2));
                let nd = gv + hypot3(wx - vx, wy - vy, wz - vz);
                if !known || nd < s.g[w] {
                    s.g[w] = nd;
                    s.back[w] = v as i32;
                    // A vertex done with is taken up again if a way to it turns out shorter,
                    // which, with the hair, does not happen.
                    s.state[w] = seen;
                    let (dx, dy, dz) = (wx - tx, wy - ty, wz - tz);
                    s.heap.push(w as i32, nd + hair * (dx * dx + dy * dy + dz * dz).sqrt());
                } else if nd == s.g[w] && s.back[w] >= 0 && gv < s.g[s.back[w] as usize] {
                    s.back[w] = v as i32;
                }
            }
        }
        if !found {
            return None;
        }
        let mut route = Vec::new();
        let mut v = to;
        loop {
            route.push(v as u32);
            if v == from || route.len() > count {
                break;
            }
            let b = s.back[v];
            if b < 0 {
                return None;
            }
            v = b as usize;
        }
        route.reverse();
        Some(route)
    }

    /// What the edge search needs made now rather than at the first search that falls back
    /// to the edges (a tenth of a second on two million triangles), for a caller with time in
    /// hand, such as one that has just prepared the scan and traces paths under a hand next.
    pub fn prepare_edge_routes(&self) {
        let adjacency = self.neighbours();
        let mut guard = lock(&self.search);
        if guard.is_none() {
            *guard = Some(self.edge_search(adjacency));
        }
    }

    /// The arrays the edge search keeps between searches, and the pieces of the mesh (the
    /// vertices edges join, labelled), made once.
    fn edge_search(&self, adjacency: &Adjacency) -> EdgeSearch {
        let count = self.positions.len() / 3;
        let mut piece = vec![-1i32; count];
        let mut queue = vec![0i32; count];
        let mut pieces = 0i32;
        for start in 0..count {
            if piece[start] >= 0 {
                continue;
            }
            piece[start] = pieces;
            let mut head = 0;
            let mut tail = 0;
            queue[tail] = start as i32;
            tail += 1;
            while head < tail {
                let v = queue[head] as usize;
                head += 1;
                for &w in adjacency.neighbours(v) {
                    if piece[w as usize] < 0 {
                        piece[w as usize] = pieces;
                        queue[tail] = w as i32;
                        tail += 1;
                    }
                }
            }
            pieces += 1;
        }
        // The queue, done with, serves as the way back.
        EdgeSearch {
            piece,
            g: vec![0.0; count],
            back: queue,
            state: vec![0; count],
            mark: vec![0; count],
            heap: Heap::new(),
            query: 0,
            expansion: 0,
        }
    }

    /// Triangle t's unit normal; +z for a triangle with no area.
    pub fn triangle_normal(&self, t: u32) -> DVec3 {
        let n = self.triangle_cross(t);
        let l = n.length();
        if l > 0.0 {
            n / l
        } else {
            DVec3::Z
        }
    }

    /// Twice the triangle's area along its normal.
    pub fn triangle_cross(&self, t: u32) -> DVec3 {
        let [a, b, c] = self.triangle(t);
        let (a, b, c) = (self.vertex(a), self.vertex(b), self.vertex(c));
        let (u, v) = (b - a, c - a);
        DVec3::new(u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x)
    }

    /// How far `p` is from triangle t, its nearest point written to `out`.
    pub fn closest_on_triangle(&self, t: u32, p: DVec3, out: &mut DVec3) -> f64 {
        let [a, b, c] = self.triangle(t);
        *out = closest_on_triangle(p, self.vertex(a), self.vertex(b), self.vertex(c));
        (*out - p).length()
    }

    fn distance_to_grid(&self, p: DVec3) -> f64 {
        let mut d = 0.0f64;
        for k in 0..3 {
            let lo = self.min[k];
            let hi = self.min[k] + self.dims[k] as f64 * self.cell;
            d = d.max(lo - p[k]).max(p[k] - hi);
        }
        d.max(0.0)
    }
}

/// √(x² + y² + z²).
pub(crate) fn hypot3(x: f64, y: f64, z: f64) -> f64 {
    (x * x + y * y + z * z).sqrt()
}

/// Of the whole numbers k0 … k1, the first and the last k for which at + step·k may lie in
/// [a, b], rounded outwards; `None` where there are none.
fn steps_within(at: f64, step: f64, a: f64, b: f64, k0: i64, k1: i64) -> Option<(i64, i64)> {
    if step == 0.0 {
        let all = at >= a && at <= b;
        return if all && k0 <= k1 { Some((k0, k1)) } else { None };
    }
    let u = (a - at) / step;
    let v = (b - at) / step;
    if u.is_nan() || v.is_nan() {
        return None;
    }
    let lo = (u.min(v)).floor().max(k0 as f64);
    let hi = (u.max(v)).ceil().min(k1 as f64);
    if lo > hi {
        return None;
    }
    Some((lo as i64, hi as i64))
}

/// The nearest point of triangle abc to p (Ericson, Real-Time Collision Detection §5.1.5).
pub fn closest_on_triangle(p: DVec3, a: DVec3, b: DVec3, c: DVec3) -> DVec3 {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return a + v * ab;
    }
    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return a + w * ac;
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
        let w = (d4 - d3) / (d4 - d3 + (d5 - d6));
        return b + w * (c - b);
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    a + ab * v + ac * w
}

/// Where a point of a path lies, read from its `on` pair.
pub fn place_of(on: &[i32], i: usize) -> PathPlace {
    let a = on[i * 2];
    let b = on[i * 2 + 1];
    if a < 0 {
        return PathPlace::Triangle(b.max(0) as u32);
    }
    if a == b {
        PathPlace::Vertex(a as u32)
    } else {
        PathPlace::Edge(a as u32, b as u32)
    }
}

/// The path along the scan from `a` to `b`, both on or near it: the plane section between
/// them, or the shortest way along the edges where the section does not join them (see the
/// module documentation). `normal` is the direction the plane holds; absent, the scan's mean
/// normal at the two points. Each point is taken to the scan first, inside its triangle
/// ([`ScanPaths::closest_inside`]): paths that meet at a point all end in its one triangle,
/// through its edges, so the edges they cross close round it.
pub fn trace_path(scan: &ScanPaths, a: DVec3, b: DVec3, normal: Option<DVec3>) -> ScanPath {
    trace_path_by(scan, a, b, normal, &|lo, hi, m, d0| scan.triangles_crossing(lo, hi, m, d0))
}

/// [`trace_path`] with the triangles a section is cut from found by `crossing` (the tests
/// set it against every triangle of the box).
fn trace_path_by(scan: &ScanPaths, a: DVec3, b: DVec3, normal: Option<DVec3>, crossing: &dyn Fn(DVec3, DVec3, DVec3, f64) -> Vec<u32>) -> ScanPath {
    let (Some(ca), Some(cb)) = (scan.closest_inside(a), scan.closest_inside(b)) else {
        return ScanPath { points: Vec::new(), on: Vec::new(), length: 0.0, way: PathWay::Section, ends: [-1, -1] };
    };
    let pa = ca.point;
    let pb = cb.point;
    let d = pb - pa;
    let dl = d.length();
    if ca.triangle == cb.triangle || dl < 1e-9 {
        return ScanPath {
            points: [pa, pb].iter().flat_map(|p| [p.x as f32, p.y as f32, p.z as f32]).collect(),
            on: vec![-1, ca.triangle as i32, -1, cb.triangle as i32],
            length: dl,
            way: PathWay::Section,
            ends: [ca.triangle as i32, cb.triangle as i32],
        };
    }
    let n = match normal {
        Some(n) => n,
        None => {
            let na = scan.normal_at(pa, ca.triangle);
            let nb = scan.normal_at(pb, cb.triangle);
            let n = na + nb;
            if n.length() < 1e-6 {
                na
            } else {
                n
            }
        }
    };
    // The plane through both points holding the normal: square to d × n.
    let m = DVec3::new(d.y * n.z - d.z * n.y, d.z * n.x - d.x * n.z, d.x * n.y - d.y * n.x);
    let ml = m.length();
    let nl = n.length();
    // The normal along the segment (a jump straight off the surface) holds no plane; nor
    // does a plane all but square to the surface at an end.
    let section = if ml > 1e-3 * dl * nl {
        section_path(scan, pa, pb, ca.triangle, cb.triangle, m / ml, dl, crossing)
    } else {
        None
    };
    section.unwrap_or_else(|| edge_path(scan, pa, pb, ca.triangle, cb.triangle))
}

/// The plane section's piece from pa to pb, or `None` where it does not join them inside the
/// tube.
#[allow(clippy::too_many_arguments)]
fn section_path(scan: &ScanPaths, pa: DVec3, pb: DVec3, ta: u32, tb: u32, m: DVec3, dl: f64, crossing: &dyn Fn(DVec3, DVec3, DVec3, f64) -> Vec<u32>) -> Option<ScanPath> {
    let p = scan.positions();
    let idx = scan.indices();
    let radius = TUBE_SHARE * dl + TUBE_EDGES * scan.edge();
    let lo = pa.min(pb) - DVec3::splat(radius);
    let hi = pa.max(pb) + DVec3::splat(radius);
    let d0 = m.x * pa.x + m.y * pa.y + m.z * pa.z;
    let pos = |v: usize, k: usize| p[v * 3 + k] as f64;
    let dist = |v: usize| m.x * pos(v, 0) + m.y * pos(v, 1) + m.z * pos(v, 2) - d0;
    let (ax, ay, az) = (pb.x - pa.x, pb.y - pa.y, pb.z - pa.z);
    let aa = ax * ax + ay * ay + az * az;
    // How far a point is from the segment pa–pb.
    let off_segment = |x: f64, y: f64, z: f64| {
        let t = (((x - pa.x) * ax + (y - pa.y) * ay + (z - pa.z) * az) / aa).clamp(0.0, 1.0);
        hypot3(x - pa.x - t * ax, y - pa.y - t * ay, z - pa.z - t * az)
    };
    // The crossing points, one per mesh edge (the lower vertex first, so the same edge gives
    // the same point whichever triangle asks) and a vertex on the plane taken to its positive
    // side, so every crossing is on an edge and a triangle crossed is crossed on two of them.
    let vertex_count = (p.len() / 3) as u64;
    let mut point_of: HashMap<u64, usize> = HashMap::new();
    let mut pts: Vec<f64> = Vec::new();
    let mut edge_of: Vec<i32> = Vec::new();
    let mut cross = |i: usize, j: usize| -> usize {
        let lo2 = i.min(j);
        let hi2 = i.max(j);
        let key = lo2 as u64 * vertex_count + hi2 as u64;
        if let Some(&had) = point_of.get(&key) {
            return had;
        }
        let di = dist(lo2);
        let dj = dist(hi2);
        let t = di / (di - dj);
        let at = pts.len() / 3;
        for k in 0..3 {
            pts.push(pos(lo2, k) + t * (pos(hi2, k) - pos(lo2, k)));
        }
        edge_of.push(lo2 as i32);
        edge_of.push(hi2 as i32);
        point_of.insert(key, at);
        at
    };
    let mut seg_a: Vec<usize> = Vec::new();
    let mut seg_b: Vec<usize> = Vec::new();
    let mut seg_tri: Vec<u32> = Vec::new();
    // Only the tube's triangles the plane crosses, from the cells it passes near: a box
    // round a long path holds much of the scan, the plane cuts a sliver of it. They come in
    // the box's order, so where the order tells (an edge more than two triangles share, two
    // segments equally near an end) the path is the one the whole box gives.
    for t in crossing(lo, hi, m, d0) {
        let (a, b, c) = (idx[t as usize * 3] as usize, idx[t as usize * 3 + 1] as usize, idx[t as usize * 3 + 2] as usize);
        let (sa, sb, sc) = (dist(a) >= 0.0, dist(b) >= 0.0, dist(c) >= 0.0);
        if sa == sb && sb == sc {
            continue;
        }
        let cx = (pos(a, 0) + pos(b, 0) + pos(c, 0)) / 3.0;
        let cy = (pos(a, 1) + pos(b, 1) + pos(c, 1)) / 3.0;
        let cz = (pos(a, 2) + pos(b, 2) + pos(c, 2)) / 3.0;
        if off_segment(cx, cy, cz) > radius {
            continue;
        }
        let mut ends: Vec<usize> = Vec::with_capacity(2);
        if sa != sb {
            ends.push(cross(a, b));
        }
        if sb != sc {
            ends.push(cross(b, c));
        }
        if sc != sa {
            ends.push(cross(c, a));
        }
        if ends.len() < 2 {
            continue;
        }
        seg_a.push(ends[0]);
        seg_b.push(ends[1]);
        seg_tri.push(t);
    }
    if seg_a.is_empty() {
        return None;
    }
    let count = pts.len() / 3;
    let mut segs_at: Vec<Vec<usize>> = vec![Vec::new(); count];
    for s in 0..seg_a.len() {
        segs_at[seg_a[s]].push(s);
        segs_at[seg_b[s]].push(s);
    }
    // The segments the two ends lie on: their own triangles', or the nearest.
    let slack = END_SLACK_EDGES * scan.edge();
    let segment_at = |q: DVec3, tri: u32| -> Option<usize> {
        let mut best = None;
        let mut best_d = f64::INFINITY;
        for s in 0..seg_a.len() {
            if seg_tri[s] == tri {
                return Some(s);
            }
            let d = point_segment(q, &pts, seg_a[s], seg_b[s]);
            if d < best_d {
                best_d = d;
                best = Some(s);
            }
        }
        if best_d <= slack {
            best
        } else {
            None
        }
    };
    let sa = segment_at(pa, ta)?;
    let sb = segment_at(pb, tb)?;
    // Walk the chain from pa's segment each way; the walk that comes to pb's is the path,
    // the shorter where both do (a closed section).
    let walk = |first: usize| -> Option<Vec<usize>> {
        let mut out = Vec::new();
        let mut seen: HashSet<usize> = HashSet::new();
        seen.insert(sa);
        let mut point = first;
        let mut seg = sa;
        loop {
            if seg == sb {
                return Some(out);
            }
            out.push(point);
            let next = segs_at[point].iter().copied().find(|&s| s != seg && !seen.contains(&s))?;
            seen.insert(next);
            seg = next;
            point = if seg_a[seg] == point { seg_b[seg] } else { seg_a[seg] };
        }
    };
    let pt = |i: usize| DVec3::new(pts[i * 3], pts[i * 3 + 1], pts[i * 3 + 2]);
    let length_of = |route: &[usize]| {
        let mut l = 0.0;
        let mut prev = pa;
        for &i in route {
            let q = pt(i);
            l += (q - prev).length();
            prev = q;
        }
        l + (pb - prev).length()
    };
    let route = if sa == sb {
        Vec::new()
    } else {
        let one = walk(seg_a[sa]);
        let other = walk(seg_b[sa]);
        match (one, other) {
            (Some(one), Some(other)) => {
                if length_of(&one) <= length_of(&other) {
                    one
                } else {
                    other
                }
            }
            (Some(one), None) => one,
            (None, Some(other)) => other,
            (None, None) => return None,
        }
    };
    let mut points = Vec::with_capacity((route.len() + 2) * 3);
    let mut on = Vec::with_capacity((route.len() + 2) * 2);
    points.extend_from_slice(&[pa.x as f32, pa.y as f32, pa.z as f32]);
    on.extend_from_slice(&[-1, ta as i32]);
    for &i in &route {
        points.extend_from_slice(&[pts[i * 3] as f32, pts[i * 3 + 1] as f32, pts[i * 3 + 2] as f32]);
        on.extend_from_slice(&[edge_of[i * 2], edge_of[i * 2 + 1]]);
    }
    points.extend_from_slice(&[pb.x as f32, pb.y as f32, pb.z as f32]);
    on.extend_from_slice(&[-1, tb as i32]);
    Some(ScanPath { points, on, length: length_of(&route), way: PathWay::Section, ends: [ta as i32, tb as i32] })
}

/// How far q is from the segment between points i and j of pts.
fn point_segment(q: DVec3, pts: &[f64], i: usize, j: usize) -> f64 {
    let (x0, y0, z0) = (pts[i * 3], pts[i * 3 + 1], pts[i * 3 + 2]);
    let (dx, dy, dz) = (pts[j * 3] - x0, pts[j * 3 + 1] - y0, pts[j * 3 + 2] - z0);
    let dd = dx * dx + dy * dy + dz * dz;
    let t = if dd > 0.0 { (((q.x - x0) * dx + (q.y - y0) * dy + (q.z - z0) * dz) / dd).clamp(0.0, 1.0) } else { 0.0 };
    hypot3(q.x - x0 - t * dx, q.y - y0 - t * dy, q.z - z0 - t * dz)
}

/// The shortest way along the mesh's edges from the nearest corner of ta to the nearest of
/// tb, the two ends added ([`ScanPaths::edge_route`]). The ends lie in their triangles; the
/// corners are where the path joins the edges. With no way between them, the straight
/// segment, marked as edges, is all there is.
fn edge_path(scan: &ScanPaths, pa: DVec3, pb: DVec3, ta: u32, tb: u32) -> ScanPath {
    let corner = |q: DVec3, t: u32| {
        let tv = scan.triangle(t);
        let mut best = tv[0];
        let mut best_d = f64::INFINITY;
        for &v in &tv {
            let d = (scan.vertex(v) - q).length();
            if d < best_d {
                best_d = d;
                best = v;
            }
        }
        best
    };
    let route = scan.edge_route(corner(pa, ta), corner(pb, tb)).unwrap_or_default();
    let mut points = Vec::with_capacity((route.len() + 2) * 3);
    let mut on = Vec::with_capacity((route.len() + 2) * 2);
    points.extend_from_slice(&[pa.x as f32, pa.y as f32, pa.z as f32]);
    on.extend_from_slice(&[-1, ta as i32]);
    let mut length = 0.0;
    let mut prev = pa;
    for &v in &route {
        let i = v as usize * 3;
        let pos = scan.positions();
        points.extend_from_slice(&[pos[i], pos[i + 1], pos[i + 2]]);
        on.extend_from_slice(&[v as i32, v as i32]);
        let q = scan.vertex(v);
        length += (q - prev).length();
        prev = q;
    }
    points.extend_from_slice(&[pb.x as f32, pb.y as f32, pb.z as f32]);
    on.extend_from_slice(&[-1, tb as i32]);
    length += (pb - prev).length();
    ScanPath { points, on, length, way: PathWay::Edges, ends: [ta as i32, tb as i32] }
}

/// A binary min-heap of (vertex, key), grown as it fills.
struct Heap {
    ids: Vec<i32>,
    keys: Vec<f64>,
    size: usize,
}

impl Heap {
    fn new() -> Self {
        Heap { ids: vec![0; 1024], keys: vec![0.0; 1024], size: 0 }
    }

    fn clear(&mut self) {
        self.size = 0;
    }

    fn push(&mut self, id: i32, key: f64) {
        if self.size == self.ids.len() {
            let n = self.size * 2;
            self.ids.resize(n, 0);
            self.keys.resize(n, 0.0);
        }
        let mut i = self.size;
        self.size += 1;
        while i > 0 {
            let parent = (i - 1) >> 1;
            if self.keys[parent] <= key {
                break;
            }
            self.ids[i] = self.ids[parent];
            self.keys[i] = self.keys[parent];
            i = parent;
        }
        self.ids[i] = id;
        self.keys[i] = key;
    }

    /// The vertex with the least key, taken off.
    fn pop(&mut self) -> i32 {
        let top = self.ids[0];
        self.size -= 1;
        let n = self.size;
        if n > 0 {
            let last_id = self.ids[n];
            let last_key = self.keys[n];
            let mut i = 0;
            loop {
                let l = 2 * i + 1;
                if l >= n {
                    break;
                }
                let r = l + 1;
                let c = if r < n && self.keys[r] < self.keys[l] { r } else { l };
                if self.keys[c] >= last_key {
                    break;
                }
                self.ids[i] = self.ids[c];
                self.keys[i] = self.keys[c];
                i = c;
            }
            self.ids[i] = last_id;
            self.keys[i] = last_key;
        }
        top
    }
}

/// Ported from ScanRuler's `tests/scanPath.test.ts`: a path along the scan. Across a cylinder
/// it is the arc of the circle round it, over a sphere the great circle, each point on the
/// surface and on a triangle edge; across a hole in a sheet the section breaks and the path
/// goes round the hole along the edges. The triangles a section is cut from are found from
/// the cells the plane passes near, and they are the ones, in the order, that every triangle
/// of the box would give.
#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use super::*;
    use crate::fit2d::Mulberry32;
    use crate::network_meshes::{icosphere, TestMesh};

    /// A grid of (i, j) over [0, 1]² mapped by `at`, two triangles a cell wound
    /// counter-clockwise in (u, v); cells for which `hole` says so left out. `wrap` joins the
    /// last column to the first.
    fn grid_mesh(nu: usize, nv: usize, at: &dyn Fn(f64, f64) -> DVec3, wrap: bool, hole: Option<&dyn Fn(f64, f64) -> bool>) -> TestMesh {
        let cols = if wrap { nu } else { nu + 1 };
        let mut positions = Vec::new();
        for j in 0..=nv {
            for i in 0..cols {
                let p = at(i as f64 / nu as f64, j as f64 / nv as f64);
                positions.extend_from_slice(&[p.x as f32, p.y as f32, p.z as f32]);
            }
        }
        let id = |i: usize, j: usize| (j * cols + if wrap { i % nu } else { i }) as u32;
        let mut indices = Vec::new();
        for j in 0..nv {
            for i in 0..nu {
                if hole.is_some_and(|h| h((i as f64 + 0.5) / nu as f64, (j as f64 + 0.5) / nv as f64)) {
                    continue;
                }
                indices.extend_from_slice(&[id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j), id(i + 1, j + 1), id(i, j + 1)]);
            }
        }
        TestMesh { positions, indices }
    }

    fn scan_of(m: &TestMesh) -> ScanPaths {
        ScanPaths::new(m.positions.clone(), m.indices.clone())
    }

    /// The triangles of the box the plane crosses, as the section took them before: every
    /// triangle of the box, those with all three corners on one side left out.
    fn crossing_in_box(scan: &ScanPaths, lo: DVec3, hi: DVec3, m: DVec3, d0: f64) -> Vec<u32> {
        let p = scan.positions();
        let side = |v: u32| {
            let i = v as usize * 3;
            m.x * p[i] as f64 + m.y * p[i + 1] as f64 + m.z * p[i + 2] as f64 - d0 >= 0.0
        };
        scan.triangles_in(lo, hi)
            .into_iter()
            .filter(|&t| {
                let [a, b, c] = scan.triangle(t);
                let (a, b, c) = (side(a), side(b), side(c));
                a != b || b != c
            })
            .collect()
    }

    /// The path a scan whose sections are cut from every triangle of the box traces.
    fn box_path(scan: &ScanPaths, a: DVec3, b: DVec3, normal: Option<DVec3>) -> ScanPath {
        trace_path_by(scan, a, b, normal, &|lo, hi, m, d0| crossing_in_box(scan, lo, hi, m, d0))
    }

    /// The test meshes: a noisy curved sheet, a noisy ball, a bent sheet with a round hole,
    /// and a sheet with a fin standing on a row of its edges (edges three triangles share,
    /// where the order the triangles come in tells which way a section goes on).
    fn test_meshes() -> Vec<(&'static str, TestMesh)> {
        // The sheet draws its noise point by point, in the grid's order.
        let rand = std::cell::RefCell::new(Mulberry32::new(5));
        let sheet = grid_mesh(90, 70, &|u, v| DVec3::new(120.0 * u, 90.0 * v, 15.0 * (3.0 * u).sin() * (2.0 * v).cos() + 0.05 * rand.borrow_mut().gaussian()), false, None);
        let ball = icosphere(4, 30.0, 0.05, 7);
        let holed = grid_mesh(60, 60, &|u, v| DVec3::new(100.0 * u, 100.0 * v, 8.0 * (u - 0.5).powi(2)), false, Some(&|u, v| (u - 0.5).hypot(v - 0.5) < 0.15));
        let flat = grid_mesh(40, 40, &|u, v| DVec3::new(100.0 * u, 100.0 * v, 0.0), false, None);
        let base = (flat.positions.len() / 3) as u32;
        let mut fin_positions = flat.positions.clone();
        let mut fin_indices = flat.indices.clone();
        for i in 0..=40 {
            fin_positions.extend_from_slice(&[(2.5 * i as f64) as f32, 50.0, 10.0]);
        }
        for i in 0..40u32 {
            let b0 = 20 * 41 + i;
            let t0 = base + i;
            fin_indices.extend_from_slice(&[b0, b0 + 1, t0 + 1, b0, t0 + 1, t0]);
        }
        vec![("curved sheet", sheet), ("ball", ball), ("holed sheet", holed), ("finned sheet", TestMesh { positions: fin_positions, indices: fin_indices })]
    }

    /// A point on triangle t, at random.
    fn point_on(mesh: &TestMesh, t: usize, rand: &mut Mulberry32) -> DVec3 {
        let mut u = rand.next_f64();
        let mut v = rand.next_f64();
        if u + v > 1.0 {
            (u, v) = (1.0 - u, 1.0 - v);
        }
        let corner = |k: usize| mesh.indices[t * 3 + k] as usize * 3;
        let (a, b, c) = (corner(0), corner(1), corner(2));
        let p = &mesh.positions;
        let at = |k: usize| (1.0 - u - v) * p[a + k] as f64 + u * p[b + k] as f64 + v * p[c + k] as f64;
        DVec3::new(at(0), at(1), at(2))
    }

    fn random_unit(rand: &mut Mulberry32) -> DVec3 {
        let v = DVec3::new(rand.gaussian(), rand.gaussian(), rand.gaussian());
        let l = hypot3(v.x, v.y, v.z);
        v / l
    }

    /// The edge search as it was: Dijkstra from `from`, the distances and the way back in
    /// maps, a binary heap on plain arrays; the vertices of the way, or none where there is
    /// none, and the distances it came to.
    fn dijkstra_route(scan: &ScanPaths, from: u32, to: u32) -> (Vec<u32>, HashMap<u32, f64>) {
        let p = scan.positions();
        let adj = scan.neighbours();
        let mut ids: Vec<u32> = Vec::new();
        let mut keys: Vec<f64> = Vec::new();
        let push = |ids: &mut Vec<u32>, keys: &mut Vec<f64>, id: u32, key: f64| {
            let mut i = ids.len();
            ids.push(id);
            keys.push(key);
            while i > 0 {
                let parent = (i - 1) >> 1;
                if keys[parent] <= key {
                    break;
                }
                ids[i] = ids[parent];
                keys[i] = keys[parent];
                i = parent;
            }
            ids[i] = id;
            keys[i] = key;
        };
        let pop = |ids: &mut Vec<u32>, keys: &mut Vec<f64>| -> (u32, f64) {
            let top = (ids[0], keys[0]);
            let last_id = ids.pop().unwrap_or(0);
            let last_key = keys.pop().unwrap_or(0.0);
            let n = ids.len();
            if n > 0 {
                let mut i = 0;
                loop {
                    let l = 2 * i + 1;
                    if l >= n {
                        break;
                    }
                    let r = l + 1;
                    let c = if r < n && keys[r] < keys[l] { r } else { l };
                    if keys[c] >= last_key {
                        break;
                    }
                    ids[i] = ids[c];
                    keys[i] = keys[c];
                    i = c;
                }
                ids[i] = last_id;
                keys[i] = last_key;
            }
            top
        };
        let pos = |v: u32, k: usize| p[v as usize * 3 + k] as f64;
        let mut dist: HashMap<u32, f64> = HashMap::from([(from, 0.0)]);
        let mut back: HashMap<u32, u32> = HashMap::new();
        push(&mut ids, &mut keys, from, 0.0);
        let mut found = from == to;
        while !found && !ids.is_empty() {
            let (v, dv) = pop(&mut ids, &mut keys);
            if dv > *dist.get(&v).unwrap_or(&f64::INFINITY) {
                continue;
            }
            if v == to {
                found = true;
                break;
            }
            for &w in adj.neighbours(v as usize) {
                let nd = dv + hypot3(pos(w, 0) - pos(v, 0), pos(w, 1) - pos(v, 1), pos(w, 2) - pos(v, 2));
                if nd < *dist.get(&w).unwrap_or(&f64::INFINITY) {
                    dist.insert(w, nd);
                    back.insert(w, v);
                    push(&mut ids, &mut keys, w, nd);
                }
            }
        }
        let mut route = Vec::new();
        if found {
            let mut v = to;
            loop {
                route.push(v);
                if v == from {
                    break;
                }
                v = back[&v];
            }
            route.reverse();
        }
        (route, dist)
    }

    /// A way's length, its edges summed from its start as the search sums them.
    fn route_length(scan: &ScanPaths, route: &[u32]) -> f64 {
        let p = scan.positions();
        let pos = |v: u32, k: usize| p[v as usize * 3 + k] as f64;
        let mut l = 0.0;
        for i in 1..route.len() {
            let (v, w) = (route[i - 1], route[i]);
            l += hypot3(pos(w, 0) - pos(v, 0), pos(w, 1) - pos(v, 1), pos(w, 2) - pos(v, 2));
        }
        l
    }

    fn close_to(a: f64, b: f64, digits: i32) -> bool {
        (a - b).abs() < 10f64.powi(-digits) / 2.0
    }

    fn point(path: &ScanPath, i: usize) -> DVec3 {
        DVec3::new(path.points[i * 3] as f64, path.points[i * 3 + 1] as f64, path.points[i * 3 + 2] as f64)
    }

    #[test]
    fn runs_round_a_cylinder_along_the_circle() {
        let r = 20.0;
        let mesh = grid_mesh(120, 40, &|u, v| DVec3::new(r * (2.0 * PI * u).cos(), r * (2.0 * PI * u).sin(), -20.0 + 40.0 * v), true, None);
        let scan = scan_of(&mesh);
        let path = trace_path(&scan, DVec3::new(r, 0.0, 3.3), DVec3::new(0.0, r, 3.3), None);
        assert_eq!(path.way, PathWay::Section);
        // The chords' sag at 3° a facet: the arc is about 0.006 % shorter.
        assert!(close_to(path.length, (PI / 2.0) * r, 1), "length {}", path.length);
        let n = path.points.len() / 3;
        for i in 0..n {
            let p = point(&path, i);
            assert!((p.x.hypot(p.y) - r).abs() < 0.02);
            // The ends are taken a hair inside their triangles: 0.1 µm here.
            assert!(close_to(p.z, 3.3, 3));
        }
        // Every point between the ends lies on an edge.
        for i in 1..n - 1 {
            assert!(matches!(place_of(&path.on, i), PathPlace::Edge(..)));
        }
        assert_eq!(place_of(&path.on, 0), PathPlace::Triangle(path.ends[0] as u32));
        // About one crossing per facet the arc passes.
        assert!(n > 25);
    }

    #[test]
    fn goes_over_a_sphere_on_the_great_circle() {
        let r = 30.0;
        let mesh = icosphere(5, r, 0.0, 7);
        let scan = scan_of(&mesh);
        let angle = (100.0 * PI) / 180.0;
        let b = DVec3::new(r * angle.cos(), r * angle.sin() * 0.6, r * angle.sin() * 0.8);
        let path = trace_path(&scan, DVec3::new(r, 0.0, 0.0), b, None);
        assert_eq!(path.way, PathWay::Section);
        assert!(path.length / (r * angle) > 0.995);
        assert!(path.length / (r * angle) < 1.001);
        // On the plane through the centre and both points.
        let n = path.points.len() / 3;
        let (pa, pb) = (point(&path, 0), point(&path, n - 1));
        let normal = DVec3::new(pa.y * pb.z - pa.z * pb.y, pa.z * pb.x - pa.x * pb.z, pa.x * pb.y - pa.y * pb.x);
        let l = normal.length();
        for i in 0..n {
            assert!((point(&path, i).dot(normal) / l).abs() < 0.05);
        }
    }

    #[test]
    fn goes_round_a_hole_along_the_edges_where_the_section_breaks() {
        // A 100 × 100 sheet with a 30 mm square hole in the middle; the two points either
        // side of it on a line through it.
        let mesh = grid_mesh(50, 50, &|u, v| DVec3::new(100.0 * u, 100.0 * v, 0.0), false, Some(&|u, v| (u - 0.5).abs() < 0.15 && (v - 0.5).abs() < 0.15));
        let scan = scan_of(&mesh);
        let path = trace_path(&scan, DVec3::new(20.0, 50.3, 0.0), DVec3::new(80.0, 50.3, 0.0), None);
        assert_eq!(path.way, PathWay::Edges);
        // The hole's rim is 14 mm out from its middle; the path may walk it but nothing
        // inside.
        let n = path.points.len() / 3;
        for i in 0..n {
            let p = point(&path, i);
            assert!(!((p.x - 50.0).abs() < 13.9 && (p.y - 50.0).abs() < 13.9));
        }
        // Round the hole: longer than across it, within the detour's corner.
        assert!(path.length > 60.0);
        assert!(path.length < 60.0 + 2.0 * 15.0 + 10.0);
        for i in 1..n - 1 {
            assert!(matches!(place_of(&path.on, i), PathPlace::Vertex(_)));
        }
        // Beside the hole the section joins them.
        let beside = trace_path(&scan, DVec3::new(20.0, 10.3, 0.0), DVec3::new(80.0, 10.3, 0.0), None);
        assert_eq!(beside.way, PathWay::Section);
        assert!(close_to(beside.length, 60.0, 6), "length {}", beside.length);
    }

    #[test]
    fn takes_points_off_the_surface_onto_it_and_finds_the_nearest_point() {
        let r = 20.0;
        let mesh = grid_mesh(90, 30, &|u, v| DVec3::new(r * (2.0 * PI * u).cos(), r * (2.0 * PI * u).sin(), 30.0 * v), true, None);
        let scan = scan_of(&mesh);
        let near = scan.closest(DVec3::new(25.0, 0.2, 10.0)).expect("a point");
        assert!(close_to(near.point.x.hypot(near.point.y), r, 1));
        assert!(close_to(near.distance, 5.0, 1));
        let path = trace_path(&scan, DVec3::new(r + 0.5, 0.0, 10.0), DVec3::new(0.0, r + 0.5, 10.0), None);
        let p = point(&path, 0);
        assert!(close_to(p.x.hypot(p.y), r, 1));
    }

    #[test]
    fn finds_the_triangles_a_plane_crosses_from_the_cells_near_it_as_every_triangle_of_the_box_gives_them() {
        let mut rand = Mulberry32::new(11);
        // Planes square to the axes too, through vertices (corners on the plane), and boxes
        // running off the grid.
        let axes = [DVec3::X, DVec3::Y, DVec3::Z, DVec3::new(0.0, -1.0, 0.0), DVec3::new(0.6, 0.0, -0.8)];
        for (name, mesh) in test_meshes() {
            let scan = scan_of(&mesh);
            let vertices = mesh.positions.len() / 3;
            let mut crossed = 0;
            for i in 0..200 {
                let v = (rand.next_f64() * vertices as f64).floor() as usize;
                let at = DVec3::new(mesh.positions[v * 3] as f64, mesh.positions[v * 3 + 1] as f64, mesh.positions[v * 3 + 2] as f64);
                let r = 0.5 + 60.0 * rand.next_f64().powi(2);
                let lo = DVec3::new(at.x - r * (0.5 + rand.next_f64()), at.y - r * (0.5 + rand.next_f64()), at.z - r * (0.5 + rand.next_f64()));
                let hi = DVec3::new(at.x + r * (0.5 + rand.next_f64()), at.y + r * (0.5 + rand.next_f64()), at.z + r * (0.5 + rand.next_f64()));
                let m = if i % 4 == 0 { axes[(i / 4) % axes.len()] } else { random_unit(&mut rand) };
                let through = if i % 3 == 0 {
                    at
                } else {
                    let t = (rand.next_f64() * scan.triangle_count() as f64).floor() as usize;
                    point_on(&mesh, t, &mut rand)
                };
                let d0 = m.x * through.x + m.y * through.y + m.z * through.z;
                let want = crossing_in_box(&scan, lo, hi, m, d0);
                assert_eq!(scan.triangles_crossing(lo, hi, m, d0), want, "{name}, query {i}");
                crossed += want.len();
            }
            assert!(crossed > 1000, "{name}: {crossed}");
        }
    }

    #[test]
    fn traces_the_same_paths_as_when_every_triangle_of_the_box_was_cut() {
        let mut rand = Mulberry32::new(23);
        for (name, mesh) in test_meshes() {
            let scan = scan_of(&mesh);
            let boxed = scan_of(&mesh);
            let mut sections = 0;
            for i in 0..150 {
                let t = (rand.next_f64() * scan.triangle_count() as f64).floor() as usize;
                let mut a = point_on(&mesh, t, &mut rand);
                let t = (rand.next_f64() * scan.triangle_count() as f64).floor() as usize;
                let b = point_on(&mesh, t, &mut rand);
                // Off the surface a little, as the cursor is, and now and then a plane held by
                // a normal given.
                let off = random_unit(&mut rand);
                for k in 0..3 {
                    a[k] += 0.3 * rand.next_f64() * off[k];
                }
                let normal = if i % 5 == 0 { Some(random_unit(&mut rand)) } else { None };
                let got = trace_path(&scan, a, b, normal);
                let want = box_path(&boxed, a, b, normal);
                assert_eq!(got.way, want.way, "{name}, path {i}");
                assert_eq!(got.ends, want.ends, "{name}, path {i}");
                assert_eq!(got.length, want.length, "{name}, path {i}");
                assert_eq!(got.points, want.points, "{name}, path {i}");
                assert_eq!(got.on, want.on, "{name}, path {i}");
                if got.way == PathWay::Section {
                    sections += 1;
                }
            }
            assert!(sections > 30, "{name}: {sections}");
        }
    }

    #[test]
    fn finds_the_shortest_way_along_the_edges_as_dijkstra_found_it() {
        // As long to the last bit, and the same way but where a vertex is come to as soon
        // through two vertices themselves as far from the start, which Dijkstra settled by
        // the order its heap held them in. A flat, even grid is full of those; a noisy or
        // curved mesh has none.
        let mut rand = Mulberry32::new(31);
        let mut meshes = test_meshes();
        // Two pieces, nothing joining them: no way, told without a search.
        let ball = icosphere(2, 10.0, 0.0, 7);
        let far = grid_mesh(10, 10, &|u, v| DVec3::new(50.0 + 20.0 * u, 20.0 * v, 0.0), false, None);
        let n = (ball.positions.len() / 3) as u32;
        let mut positions = ball.positions.clone();
        positions.extend_from_slice(&far.positions);
        let mut indices = ball.indices.clone();
        indices.extend(far.indices.iter().map(|v| v + n));
        meshes.push(("ball and sheet", TestMesh { positions, indices }));
        for (name, mesh) in meshes {
            let scan = scan_of(&mesh);
            let vertices = mesh.positions.len() / 3;
            let mut other = 0;
            let mut none = 0;
            for i in 0..300 {
                let from = (rand.next_f64() * vertices as f64).floor() as u32;
                let to = if i % 10 == 0 { from } else { (rand.next_f64() * vertices as f64).floor() as u32 };
                let (want, dist) = dijkstra_route(&scan, from, to);
                let got = scan.edge_route(from, to);
                // The holed sheet keeps the vertices of its hole, joined to nothing.
                if want.is_empty() {
                    assert!(got.is_none(), "{name}, {from} → {to}");
                    none += 1;
                    continue;
                }
                let route = got.unwrap_or_else(|| panic!("{name}, {from} → {to}: no way"));
                assert_eq!(route_length(&scan, &route), route_length(&scan, &want), "{name}, {from} → {to}");
                if route == want {
                    continue;
                }
                other += 1;
                // From the end back, the first vertex the two ways come to from different
                // vertices: those are as far from the start.
                let mut k = 1;
                while route[route.len() - k] == want[want.len() - k] {
                    k += 1;
                }
                let (a, b) = (route[route.len() - k], want[want.len() - k]);
                assert_eq!(dist.get(&a), dist.get(&b), "{name}, {from} → {to}");
            }
            if name == "curved sheet" || name == "ball" || name == "holed sheet" {
                assert_eq!(other, 0, "{name}");
            }
            if name == "ball and sheet" {
                assert!(none > 100, "{none}");
            }
        }
    }
}
