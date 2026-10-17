//! Quadric-error edge-collapse simplification (Meshmixer *Reduce*).
//!
//! Garland & Heckbert, "Surface Simplification Using Quadric Error Metrics"
//! (SIGGRAPH 1997), organised like geometry3Sharp's `Reducer` (Boost License):
//!
//! * per-vertex quadrics = sum of area-weighted plane quadrics of incident triangles;
//!   free boundary edges and preserved face-group borders add perpendicular "penalty"
//!   planes so those curves keep their shape;
//! * candidate collapses live in a binary heap with lazy invalidation (per-edge version
//!   stamps); after a collapse only the edges around the surviving vertex are re-scored;
//! * the new position minimises the summed quadric (3x3 solve), falling back to the best
//!   point on the edge when the system is ill-conditioned or the solution runs away;
//! * collapses that break the link condition, flip / over-rotate a normal or create
//!   low-quality triangles are rejected, so the mesh stays valid and closed meshes stay
//!   closed.
//!
//! **Error metric.** Quadric cost is a sum over the merged neighbourhood of
//! `area x squared distance to the triangle planes` (units length⁴): good for ordering
//! collapses, but not a distance. [`ReduceTarget::MaxDeviation`] therefore measures
//! geometry directly: samples of the original surface (every movable vertex, every edge
//! midpoint and every triangle centroid) are tracked on result triangles, and a collapse
//! is accepted only if every sample of the affected fan stays within `0.95 x tolerance`
//! of the new fan and the new vertex stays within the tolerance of the original surface.
//! This bounds the one-sided Hausdorff distance original → result at the samples. Between
//! samples the distance can peak inside an original triangle that straddles a concave
//! crease of the result (distance to a piecewise-planar surface is not convex); the 5%
//! margin covers this on smooth surfaces (no excess measured in the tests), while on a
//! raw voxel staircase with a tolerance below the voxel size isolated points exceeded the
//! tolerance by up to ~20%.

use crate::constraints::*;
use crate::quadric::Quadric;
use crate::region::Region;
use crate::util::*;
use crate::{ProgressFn, RemeshError};
use mm_core::geom::closest_point_on_triangle;
use mm_core::{Bvh, DVec3, Mesh, INVALID};
use rayon::prelude::*;
use smallvec::SmallVec;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::time::{Duration, Instant};

/// Weight of boundary / group-border penalty planes relative to the area-weighted
/// surface quadrics (scaled by squared edge length).
const PENALTY_WEIGHT: f64 = 10.0;

/// Tracked samples must stay within this fraction of the deviation tolerance, leaving
/// headroom for the surface between samples.
const SAMPLE_MARGIN: f64 = 0.95;

/// What [`reduce`] aims for. In region mode counts refer to the region's triangles.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ReduceTarget {
    /// Stop once at most this many triangles remain.
    TriangleCount(usize),
    /// Keep this percentage `(0, 100]` of the current triangle count.
    Percentage(f64),
    /// Simplify as far as possible while the original surface stays within this
    /// distance of the result (sample-based bound with a 5% margin, see the module docs).
    MaxDeviation(f64),
}

/// Parameters for [`reduce`].
#[derive(Clone, Debug)]
pub struct ReduceParams {
    pub target: ReduceTarget,
    /// Keep boundary vertices exactly (never moved or removed). When false, boundaries
    /// are simplified too, held in shape by penalty quadrics.
    pub preserve_boundary: bool,
    /// Face-group borders become curves: their vertices are only removed along the
    /// border and border junctions are fixed.
    pub preserve_group_boundaries: bool,
    /// Only simplify these triangles; the region border is fixed and everything outside
    /// is left bit-identical. New region ids are returned in the report.
    pub region: Option<Vec<u32>>,
    /// Reject collapses whose new worst triangle is below this quality (see
    /// [`crate::triangle_quality`]) and worse than the current worst.
    pub min_triangle_quality: f64,
    /// Reject collapses rotating any triangle normal by more than this many degrees.
    /// Normal flips (> 90°) are always rejected.
    pub max_normal_change_deg: f64,
}

impl Default for ReduceParams {
    fn default() -> Self {
        Self {
            target: ReduceTarget::Percentage(50.0),
            preserve_boundary: true,
            preserve_group_boundaries: true,
            region: None,
            min_triangle_quality: 0.05,
            max_normal_change_deg: 60.0,
        }
    }
}

impl ReduceParams {
    pub fn with_target(target: ReduceTarget) -> Self {
        Self { target, ..Default::default() }
    }
}

/// Result of [`reduce`].
#[derive(Clone, Debug, Default)]
pub struct ReduceReport {
    pub initial_triangles: usize,
    pub final_triangles: usize,
    pub collapses: usize,
    /// For [`ReduceTarget::MaxDeviation`]: largest distance from a tracked sample of the
    /// original surface to the result triangle it is assigned to (an upper bound on its
    /// distance to the result; at most `0.95 x tolerance`).
    pub max_deviation: Option<f64>,
    /// New ids of the region triangles (only when a region was given).
    pub region_triangles: Option<Vec<u32>>,
    pub cancelled: bool,
    pub elapsed: Duration,
}

/// Quadric-error simplification in place. See the module docs.
pub fn reduce(mesh: &mut Mesh, params: &ReduceParams, progress: Option<ProgressFn>) -> Result<ReduceReport, RemeshError> {
    let start = Instant::now();
    match params.target {
        ReduceTarget::Percentage(p) if !(p > 0.0 && p <= 100.0) => {
            return Err(RemeshError::InvalidParameter("percentage must be in (0, 100]".into()))
        }
        ReduceTarget::MaxDeviation(d) if !(d.is_finite() && d > 0.0) => {
            return Err(RemeshError::InvalidParameter("max deviation must be positive".into()))
        }
        _ => {}
    }
    if !params.min_triangle_quality.is_finite() || !params.max_normal_change_deg.is_finite() {
        return Err(RemeshError::InvalidParameter("quality / normal thresholds must be finite".into()));
    }
    let mut prog = Progress::new(progress);
    let mut r = Reducer::new(mesh, params);
    let initial = r.count;
    let target = match params.target {
        ReduceTarget::TriangleCount(n) => n,
        ReduceTarget::Percentage(p) => ((initial as f64) * p / 100.0).round() as usize,
        ReduceTarget::MaxDeviation(_) => 0,
    };
    if initial > target {
        r.run(target, &mut prog);
    }
    if prog.ok() {
        prog.report(1.0);
    }
    let region_triangles = if r.region.enabled() { Some(r.region.triangles(r.mesh)) } else { None };
    Ok(ReduceReport {
        initial_triangles: initial,
        final_triangles: r.count,
        collapses: r.collapses,
        max_deviation: r.dev.as_ref().map(|d| d.measured_max(r.mesh)),
        region_triangles,
        cancelled: prog.cancelled,
        elapsed: start.elapsed(),
    })
}

#[derive(Clone, Copy)]
struct Cand {
    cost: f32,
    keep: u32,
    rem: u32,
    pos: DVec3,
}

/// Heap key: f32 cost bits (order-preserving for non-negative floats) above the edge id.
/// Popping re-evaluates the edge; a key whose cost no longer matches is stale.
#[inline]
fn heap_key(cost: f32, e: u32) -> Reverse<u64> {
    Reverse(((cost.to_bits() as u64) << 32) | e as u64)
}

struct Reducer<'m> {
    mesh: &'m mut Mesh,
    region: Region,
    vc: Vec<VCons>,
    ec: Vec<ECons>,
    q: Vec<Quadric>,
    /// Cost bits of the newest heap key per edge (older keys are stale).
    key_cost: Vec<u32>,
    heap: BinaryHeap<Reverse<u64>>,
    min_dot: f64,
    min_q: f64,
    dev: Option<Deviation>,
    count: usize,
    collapses: usize,
    stamp: Stamp,
}

impl<'m> Reducer<'m> {
    fn new(mesh: &'m mut Mesh, p: &ReduceParams) -> Self {
        let mut region = Region::new(mesh, p.region.as_deref());
        let mut stamp = Stamp::default();
        let edges = region.edges(mesh, &mut stamp);
        let verts = region.vertices(mesh, &mut stamp);
        let opts = ConstraintOptions {
            boundary: if p.preserve_boundary { BoundaryRule::Fixed } else { BoundaryRule::Free },
            sharp_deg: None,
            groups: p.preserve_group_boundaries,
            corner_deg: None,
            smooth_groups: None,
        };
        let cons = build_constraints(mesh, &region, &edges, &verts, &opts);
        let count = region.triangle_count(mesh);

        let m: &Mesh = mesh;
        let tq: Vec<Quadric> = (0..m.max_tid() as u32)
            .into_par_iter()
            .map(|t| {
                if !m.is_triangle(t) {
                    return Quadric::default();
                }
                let [a, b, c] = m.tri_positions(t);
                let cr = cross3(a, b, c);
                let l = cr.length();
                if l > 0.0 {
                    Quadric::from_point_normal(a, cr / l, 0.5 * l)
                } else {
                    Quadric::default()
                }
            })
            .collect();
        let ecr = &cons.e;
        let free_boundary = !p.preserve_boundary;
        let q: Vec<Quadric> = (0..m.max_vid() as u32)
            .into_par_iter()
            .map(|v| {
                let mut s = Quadric::default();
                if !m.is_vertex(v) {
                    return s;
                }
                for t in m.vertex_triangles(v) {
                    s.add(&tq[t as usize]);
                }
                for &e in m.vertex_edges(v) {
                    let bnd = m.is_boundary_edge(e);
                    let feature = ecr.get(e as usize).is_some_and(|c| c.kind == E_FEATURE);
                    if !(feature || (bnd && free_boundary)) {
                        continue;
                    }
                    let [a, b] = m.edge_vertices(e);
                    let (pa, pb) = (m.vertex(a), m.vertex(b));
                    let d = pb - pa;
                    let w = PENALTY_WEIGHT * d.length_squared();
                    for t in m.edge_triangles(e) {
                        if t == INVALID {
                            continue;
                        }
                        let n = d.cross(m.tri_normal(t)).normalize_or_zero();
                        if n != DVec3::ZERO {
                            s.add(&Quadric::from_point_normal(pa, n, w));
                        }
                    }
                }
                s
            })
            .collect();

        let dev = match p.target {
            ReduceTarget::MaxDeviation(tol) => Some(Deviation::new(m, &mut region, &cons.v, &verts, tol)),
            _ => None,
        };
        let ne = m.max_eid();
        Reducer {
            mesh,
            region,
            vc: cons.v,
            ec: cons.e,
            q,
            key_cost: vec![u32::MAX; ne],
            heap: BinaryHeap::new(),
            min_dot: p.max_normal_change_deg.clamp(0.0, 180.0).to_radians().cos().max(0.0),
            min_q: p.min_triangle_quality.clamp(0.0, 1.0),
            dev,
            count,
            collapses: 0,
            stamp,
        }
    }

    fn evaluate(&self, e: u32) -> Option<Cand> {
        let m = &*self.mesh;
        if !m.is_edge(e) {
            return None;
        }
        let c = self.ec[e as usize];
        if c.kind == E_FIXED {
            return None;
        }
        let [a, b] = m.edge_vertices(e);
        let ra = removable(self.vc[a as usize], c);
        let rb = removable(self.vc[b as usize], c);
        if !ra && !rb {
            return None;
        }
        let q = self.q[a as usize].sum(&self.q[b as usize]);
        let (pa, pb) = (m.vertex(a), m.vertex(b));
        let (keep, rem, pos) = if ra && rb {
            let pos = if c.kind == E_FEATURE { q.min_on_segment(pa, pb) } else { optimal_position(&q, pa, pb) };
            (a, b, pos)
        } else if ra {
            (b, a, pb)
        } else {
            (a, b, pa)
        };
        // Tiny length term (same length^4 units as the cost) orders zero-cost collapses in
        // flat regions shortest-first instead of by edge id.
        let l2 = pa.distance_squared(pb);
        let cost = q.eval(pos) + 1e-6 * l2 * l2;
        Some(Cand { cost: cost as f32, keep, rem, pos })
    }

    fn push(&mut self, e: u32) {
        match self.evaluate(e) {
            Some(c) => {
                self.key_cost[e as usize] = c.cost.to_bits();
                self.heap.push(heap_key(c.cost, e));
            }
            None => self.key_cost[e as usize] = u32::MAX,
        }
    }

    fn fill_heap(&mut self) {
        let edges = self.region.edges(self.mesh, &mut self.stamp);
        let this = &*self;
        let keys: Vec<Reverse<u64>> =
            edges.par_iter().filter_map(|&e| this.evaluate(e).map(|c| heap_key(c.cost, e))).collect();
        self.key_cost.iter_mut().for_each(|k| *k = u32::MAX);
        for k in &keys {
            self.key_cost[(k.0 & 0xffff_ffff) as usize] = (k.0 >> 32) as u32;
        }
        self.heap = BinaryHeap::from(keys);
    }

    fn run(&mut self, target: usize, prog: &mut Progress) {
        let initial = self.count;
        let span = (initial - target).max(1) as f32;
        for _round in 0..3 {
            self.fill_heap();
            let heap0 = self.heap.len().max(1) as f32;
            let before = self.collapses;
            let mut pops = 0usize;
            while self.count > target {
                let Some(ent) = self.heap.pop() else { break };
                pops += 1;
                if pops.is_multiple_of(4096) {
                    let f = if self.dev.is_some() {
                        1.0 - self.heap.len() as f32 / heap0
                    } else {
                        (initial - self.count) as f32 / span
                    };
                    if !prog.report(f) {
                        return;
                    }
                }
                let e = (ent.0 & 0xffff_ffff) as u32;
                if self.key_cost[e as usize] != (ent.0 >> 32) as u32 || !self.mesh.is_edge(e) {
                    continue; // stale: a fresher key for this edge is in the heap
                }
                self.key_cost[e as usize] = u32::MAX;
                let Some(c) = self.evaluate(e) else { continue };
                self.try_collapse(e, c);
            }
            if self.count <= target || self.collapses == before || self.dev.is_some() {
                break;
            }
        }
    }

    fn try_collapse(&mut self, e: u32, c: Cand) -> bool {
        let m = &*self.mesh;
        let Cand { keep, rem, pos, .. } = c;
        if m.can_collapse(keep, rem).is_err() {
            return false;
        }
        let [t0, t1] = m.edge_triangles(e);
        for o in m.edge_opposite_vertices(e) {
            if o == INVALID {
                continue;
            }
            if let Some(x) = m.find_edge(rem, o) {
                if self.ec[x as usize].kind != E_NONE {
                    return false;
                }
            }
        }
        if !collapse_shape_ok(m, keep, rem, pos, t0, t1, self.min_dot, self.min_q) {
            return false;
        }
        if let Some(dev) = self.dev.as_mut() {
            if !dev.check(m, keep, rem, pos, t0, t1) {
                return false;
            }
        }
        let removed = if t1 == INVALID { 1 } else { 2 };
        if self.mesh.collapse_edge(keep, rem).is_err() {
            return false;
        }
        self.mesh.set_vertex(keep, pos);
        let qr = self.q[rem as usize];
        self.q[keep as usize].add(&qr);
        if let Some(dev) = self.dev.as_mut() {
            dev.commit();
        }
        self.count -= removed;
        self.collapses += 1;
        let around: SmallVec<[u32; 16]> = self.mesh.vertex_edges(keep).iter().copied().collect();
        for e2 in around {
            self.push(e2);
        }
        true
    }
}

/// Quadric minimiser near edge (pa, pb): the 3x3 solve when well conditioned and not
/// running away from the edge, else the best of {segment minimiser, midpoint, ends}.
fn optimal_position(q: &Quadric, pa: DVec3, pb: DVec3) -> DVec3 {
    let mid = (pa + pb) * 0.5;
    let len = pa.distance(pb);
    if let Some(x) = q.solve() {
        if x.distance(mid) <= 2.0 * len {
            return x;
        }
    }
    let mut best = q.min_on_segment(pa, pb);
    let mut bc = q.eval(best);
    for p in [mid, pa, pb] {
        let c = q.eval(p);
        if c < bc - 1e-15 * bc.abs() {
            best = p;
            bc = c;
        }
    }
    best
}

/// Tracks samples of the original surface on the simplified surface for
/// [`ReduceTarget::MaxDeviation`].
///
/// Each sample sits in the list of one current triangle; each triangle keeps an upper
/// bound on the distance of its samples. Moving one vertex of a triangle by `δ` moves
/// every point of the triangle by at most `δ`, so a surviving fan triangle whose
/// `bound + δ` stays within tolerance is accepted without touching its samples; only
/// samples of the two removed triangles and of triangles whose bound would exceed the
/// tolerance are measured exactly (and re-homed onto the nearest fan triangle).
struct Deviation {
    tol: f64,
    reference: Mesh,
    bvh: Bvh,
    pts: Vec<DVec3>,
    next: Vec<u32>,
    head: Vec<u32>,
    bound: Vec<f64>,
    /// Staged by `check`, applied by `commit`.
    pending_pts: Vec<(u32, u32)>,
    pending_clear: SmallVec<[u32; 32]>,
    pending_bounds: SmallVec<[(u32, f64); 32]>,
}

impl Deviation {
    fn new(m: &Mesh, region: &mut Region, vc: &[VCons], verts: &[u32], tol: f64) -> Self {
        let reference = if region.enabled() { m.submesh(&region.triangles_with_collar(m)).0 } else { m.clone() };
        let bvh = Bvh::build(&reference);
        let mut d = Deviation {
            tol,
            reference,
            bvh,
            pts: Vec::new(),
            next: Vec::new(),
            head: vec![INVALID; m.max_tid()],
            bound: vec![0.0; m.max_tid()],
            pending_pts: Vec::new(),
            pending_clear: SmallVec::new(),
            pending_bounds: SmallVec::new(),
        };
        for &v in verts {
            if vc[v as usize].kind == V_FIXED {
                continue;
            }
            let Some(&t) = m.vertex_triangles(v).first() else { continue };
            d.add_point(m.vertex(v), t);
        }
        // Interior samples: distance to a piecewise-planar surface is not convex, so
        // vertices alone under-sample the original triangles.
        let tris = region.triangles(m);
        let mut es = Stamp::default();
        es.begin(m.max_eid());
        for &t in &tris {
            d.add_point(m.tri_centroid(t), t);
            for e in m.tri_edges(t) {
                if es.mark(e) {
                    d.add_point(m.edge_midpoint(e), t);
                }
            }
        }
        d
    }

    fn add_point(&mut self, p: DVec3, t: u32) {
        let i = self.pts.len() as u32;
        self.pts.push(p);
        self.next.push(self.head[t as usize]);
        self.head[t as usize] = i;
    }

    /// Would collapsing `rem` into `keep` at `pos` keep every sample within tolerance?
    /// On success the changes are staged for [`Deviation::commit`].
    fn check(&mut self, m: &Mesh, keep: u32, rem: u32, pos: DVec3, t0: u32, t1: u32) -> bool {
        if self.bvh.nearest(&self.reference, pos, self.tol).is_none() {
            return false;
        }
        let tol = SAMPLE_MARGIN * self.tol;
        let (dk, dr) = (pos.distance(m.vertex(keep)), pos.distance(m.vertex(rem)));
        let mut old_fan: SmallVec<[u32; 32]> = SmallVec::new();
        for v in [rem, keep] {
            for t in m.vertex_triangles(v) {
                if !old_fan.contains(&t) {
                    old_fan.push(t);
                }
            }
        }
        // Surviving triangles with their new corners and the displacement of the
        // corner that moves.
        let mut new_fan: SmallVec<[(u32, [DVec3; 3], f64); 32]> = SmallVec::new();
        for &t in &old_fan {
            if t == t0 || t == t1 {
                continue;
            }
            let tv = m.triangle(t);
            let mut p = [m.vertex(tv[0]), m.vertex(tv[1]), m.vertex(tv[2])];
            let mut delta = 0.0;
            for k in 0..3 {
                if tv[k] == rem {
                    p[k] = pos;
                    delta = dr;
                } else if tv[k] == keep {
                    p[k] = pos;
                    delta = dk;
                }
            }
            new_fan.push((t, p, delta));
        }
        if new_fan.is_empty() {
            return false;
        }
        // Try the likely triangles first (any within tolerance is acceptable), then scan
        // the whole fan for the nearest.
        let place = |x: DVec3, first: &[usize]| -> Option<(usize, f64)> {
            for &k in first {
                let (_, [a, b, c], _) = new_fan[k];
                let d = closest_point_on_triangle(x, a, b, c).0.distance(x);
                if d <= tol {
                    return Some((k, d));
                }
            }
            let mut best = f64::INFINITY;
            let mut bk = 0;
            for (k, &(_, [a, b, c], _)) in new_fan.iter().enumerate() {
                let d2 = closest_point_on_triangle(x, a, b, c).0.distance_squared(x);
                if d2 < best {
                    best = d2;
                    bk = k;
                }
            }
            let d = best.sqrt();
            (d <= tol).then_some((bk, d))
        };
        self.pending_pts.clear();
        self.pending_clear.clear();
        self.pending_bounds.clear();
        // New bound per fan triangle; fast-path triangles start from bound + delta.
        let mut nb: SmallVec<[f64; 32]> = SmallVec::new();
        let mut lists: SmallVec<[(u32, SmallVec<[usize; 3]>); 32]> = SmallVec::new();
        for (k, &(t, _, delta)) in new_fan.iter().enumerate() {
            let b = self.bound[t as usize] + delta;
            if b <= tol {
                nb.push(b);
            } else {
                nb.push(0.0);
                lists.push((t, smallvec::smallvec![k]));
            }
        }
        for t in [t0, t1] {
            if t != INVALID {
                // Samples of a removed triangle mostly land across its surviving edges.
                let first = m
                    .tri_neighbors(t)
                    .iter()
                    .filter_map(|&n| new_fan.iter().position(|&(nt, _, _)| nt == n))
                    .collect();
                lists.push((t, first));
            }
        }
        for (t, first) in &lists {
            let t = *t;
            let mut i = self.head[t as usize];
            while i != INVALID {
                let Some((k, d)) = place(self.pts[i as usize], first) else { return false };
                self.pending_pts.push((i, new_fan[k].0));
                nb[k] = nb[k].max(d);
                i = self.next[i as usize];
            }
            self.pending_clear.push(t);
        }
        for (k, &(t, _, _)) in new_fan.iter().enumerate() {
            self.pending_bounds.push((t, nb[k]));
        }
        true
    }

    fn commit(&mut self) {
        for &t in &self.pending_clear {
            self.head[t as usize] = INVALID;
        }
        for &(i, t) in &self.pending_pts {
            self.next[i as usize] = self.head[t as usize];
            self.head[t as usize] = i;
        }
        for &(t, b) in &self.pending_bounds {
            self.bound[t as usize] = b;
        }
    }

    /// Largest distance from a sample to the triangle it is assigned to (an upper bound
    /// on its distance to the simplified surface).
    fn measured_max(&self, m: &Mesh) -> f64 {
        (0..m.max_tid() as u32)
            .into_par_iter()
            .filter(|&t| m.is_triangle(t))
            .map(|t| {
                let [a, b, c] = m.tri_positions(t);
                let mut mx = 0.0f64;
                let mut i = self.head[t as usize];
                while i != INVALID {
                    let x = self.pts[i as usize];
                    mx = mx.max(closest_point_on_triangle(x, a, b, c).0.distance(x));
                    i = self.next[i as usize];
                }
                mx
            })
            .reduce(|| 0.0, f64::max)
    }
}
