//! Isotropic remeshing (Meshmixer *Remesh*).
//!
//! The algorithm follows Botsch & Kobbelt, "A Remeshing Approach to Multiresolution
//! Modeling" (SGP 2004); the constraint handling (fixed / curve-constrained vertices,
//! unflippable / uncollapsible feature edges, projection targets) mirrors
//! geometry3Sharp's `Remesher` + `MeshConstraints` (Boost License, R. Schmidt).
//!
//! One iteration = split long edges (> 4/3·L), collapse short edges (< 4/5·L),
//! valence-improving flips, tangential relaxation, projection onto the input surface.

use crate::constraints::*;
use crate::curves::{CurveIndex, Seg};
use crate::region::Region;
use crate::stats::{edge_length_stats_in, EdgeLengthStats};
use crate::util::*;
use crate::{ProgressFn, RemeshError};
use mm_core::mesh::EdgeSplitInfo;
use mm_core::{Bvh, DVec3, Mesh, INVALID};
use rayon::prelude::*;
use std::collections::BinaryHeap;
use std::time::{Duration, Instant};

/// Split edges longer than this multiple of the local target length.
pub(crate) const MAX_FACTOR: f64 = 4.0 / 3.0;
/// Collapse edges shorter than this multiple of the local target length.
pub(crate) const MIN_FACTOR: f64 = 4.0 / 5.0;

/// How mesh-boundary edges are treated by [`remesh`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BoundaryMode {
    /// Boundary vertices and edges are kept exactly as they are.
    #[default]
    Fixed,
    /// Boundary edges may be split and collapsed and boundary vertices may slide, but
    /// they always stay on the original boundary polyline (Meshmixer "Refined Boundary").
    ConstrainToBoundary,
    /// Boundary vertices are relaxed along the boundary and projected onto the input
    /// surface like interior vertices; the boundary curve may shrink slightly.
    Free,
}

/// Parameters for [`remesh`].
#[derive(Clone, Debug)]
pub struct RemeshParams {
    /// Target edge length in mesh units (mm for Slicer exports). A value `<= 0` (or
    /// non-finite) means "use the current mean edge length of the region".
    pub target_edge_length: f64,
    /// Number of split/collapse/flip/smooth/project iterations.
    pub iterations: usize,
    /// Tangential relaxation speed in `[0, 1]` (0 disables smoothing).
    pub smoothing: f64,
    pub boundary: BoundaryMode,
    /// Keep edges whose dihedral angle exceeds this many degrees as feature curves
    /// (also: feature curves turning by more than this angle get fixed corners).
    pub preserve_sharp_edges: Option<f64>,
    /// Keep face-group borders as feature curves (vertices on them only slide along them).
    pub preserve_group_boundaries: bool,
    /// Curvature-adaptive sizing: the target varies between 0.5x and 2x
    /// `target_edge_length`, shorter where the surface is more curved.
    pub adaptive: bool,
    /// Project vertices back onto the input surface after every iteration.
    pub project_to_original: bool,
    /// Only remesh these triangles. The region border is kept fixed and everything
    /// outside is left bit-identical. The new region ids are returned in the report.
    pub region: Option<Vec<u32>>,
    /// Smooth face-group borders instead of keeping them exactly (needs
    /// `preserve_group_boundaries`; Meshmixer "Smooth Group Boundaries" / Smooth Boundary).
    pub smooth_group_boundaries: Option<GroupBorderSmoothing>,
}

/// How [`RemeshParams::smooth_group_boundaries`] smooths face-group border curves.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroupBorderSmoothing {
    /// Per pass, border vertices move this fraction (0..1) toward the midpoint of their two
    /// neighbours along the border.
    pub strength: f64,
    /// Per pass, they are pulled back toward the original border by this fraction (0..1);
    /// higher keeps more of the original shape and stops loops from shrinking.
    pub anchor: f64,
    /// Only smooth borders between a group `< split` and a group `>= split` (other group
    /// borders stay exact). `None` smooths every group border.
    pub split: Option<i32>,
}

impl Default for RemeshParams {
    fn default() -> Self {
        Self {
            target_edge_length: 0.0,
            iterations: 10,
            smoothing: 0.5,
            boundary: BoundaryMode::Fixed,
            preserve_sharp_edges: None,
            preserve_group_boundaries: true,
            adaptive: false,
            project_to_original: true,
            region: None,
            smooth_group_boundaries: None,
        }
    }
}

impl RemeshParams {
    /// Default parameters with the given target edge length.
    pub fn with_target(target_edge_length: f64) -> Self {
        Self { target_edge_length, ..Default::default() }
    }
}

/// Result of [`remesh`].
#[derive(Clone, Debug, Default)]
pub struct RemeshReport {
    /// The target edge length actually used (after "auto" resolution).
    pub target_edge_length: f64,
    /// Iterations fully completed.
    pub iterations: usize,
    pub splits: usize,
    pub collapses: usize,
    pub flips: usize,
    /// Triangles in the region (or mesh) before and after.
    pub initial_triangles: usize,
    pub final_triangles: usize,
    pub final_vertices: usize,
    /// Edge lengths of the remeshed region (or the whole mesh).
    pub edge_lengths: EdgeLengthStats,
    /// New ids of the region triangles (only when a region was given).
    pub region_triangles: Option<Vec<u32>>,
    pub cancelled: bool,
    pub elapsed: Duration,
    /// Time spent in each pass type, summed over iterations.
    pub timings: PassTimings,
}

/// Cumulative wall time per pass type.
#[derive(Clone, Copy, Debug, Default)]
pub struct PassTimings {
    /// Constraint classification, reference copy, BVH and sizing field.
    pub setup: Duration,
    pub split: Duration,
    pub collapse: Duration,
    pub flip: Duration,
    pub smooth_project: Duration,
}

/// Isotropic remeshing in place. See the module documentation for the algorithm.
///
/// Every intermediate state is a valid mesh, so cancelling through `progress` is safe.
pub fn remesh(mesh: &mut Mesh, params: &RemeshParams, progress: Option<ProgressFn>) -> Result<RemeshReport, RemeshError> {
    let start = Instant::now();
    if !params.smoothing.is_finite() {
        return Err(RemeshError::InvalidParameter("smoothing must be finite".into()));
    }
    if let Some(a) = params.preserve_sharp_edges {
        if !(a.is_finite() && a > 0.0 && a < 180.0) {
            return Err(RemeshError::InvalidParameter("crease angle must be in (0, 180) degrees".into()));
        }
    }
    let setup = Setup {
        region: params.region.as_deref(),
        constraints: ConstraintOptions {
            boundary: match params.boundary {
                BoundaryMode::Fixed => BoundaryRule::Fixed,
                BoundaryMode::ConstrainToBoundary => BoundaryRule::Curve,
                BoundaryMode::Free => BoundaryRule::Free,
            },
            sharp_deg: params.preserve_sharp_edges,
            groups: params.preserve_group_boundaries,
            corner_deg: params.preserve_sharp_edges,
            smooth_groups: params.smooth_group_boundaries.filter(|_| params.preserve_group_boundaries).map(|g| g.split),
        },
        base: params.target_edge_length,
        adaptive: params.adaptive,
        project: params.project_to_original,
        smoothing: params.smoothing.clamp(0.0, 1.0),
        free_boundary: params.boundary == BoundaryMode::Free,
        group_smoothing: params
            .smooth_group_boundaries
            .filter(|_| params.preserve_group_boundaries)
            .map(|g| (g.strength.clamp(0.0, 1.0), g.anchor.clamp(0.0, 1.0))),
    };
    let mut r = Remesher::new(mesh, setup);
    r.timings.setup = start.elapsed();
    let initial_triangles = r.region.triangle_count(r.mesh);
    let mut prog = Progress::new(progress);
    let mut done = 0;
    if initial_triangles > 0 {
        done = r.run(params.iterations, &mut prog);
    }
    if prog.ok() {
        prog.report(1.0);
    }
    let tris = r.region.triangles(r.mesh);
    let edge_lengths = edge_length_stats_in(r.mesh, &tris);
    Ok(RemeshReport {
        target_edge_length: r.base,
        iterations: done,
        splits: r.splits,
        collapses: r.collapses,
        flips: r.flips,
        initial_triangles,
        final_triangles: tris.len(),
        final_vertices: r.mesh.vertex_count(),
        edge_lengths,
        region_triangles: if r.region.enabled() { Some(tris) } else { None },
        cancelled: prog.cancelled,
        elapsed: start.elapsed(),
        timings: r.timings,
    })
}

/// Configuration for [`Remesher::new`] (shared with the cleanup helpers).
pub(crate) struct Setup<'a> {
    pub region: Option<&'a [u32]>,
    pub constraints: ConstraintOptions,
    /// Target edge length; `<= 0` = mean edge length.
    pub base: f64,
    pub adaptive: bool,
    pub project: bool,
    pub smoothing: f64,
    pub free_boundary: bool,
    /// (strength, anchor) for smoothing the flagged group-border curves.
    pub group_smoothing: Option<(f64, f64)>,
}

/// Frozen copy of the input surface used for projection and the sizing field.
pub(crate) struct Reference {
    mesh: Mesh,
    bvh: Bvh,
    /// Sizing field per reference vertex.
    size: Vec<f64>,
}

pub(crate) struct Remesher<'m> {
    pub mesh: &'m mut Mesh,
    pub region: Region,
    pub vc: Vec<VCons>,
    pub ec: Vec<ECons>,
    /// Local target edge length per vertex.
    pub vsize: Vec<f64>,
    pub base: f64,
    pub adaptive: bool,
    pub free_boundary: bool,
    pub smoothing: f64,
    pub project: bool,
    pub curves: CurveIndex,
    /// Per curve: smooth it (group border selected for smoothing).
    pub curve_smooth: Vec<bool>,
    pub group_smoothing: Option<(f64, f64)>,
    pub reference: Option<Reference>,
    /// Reject collapses that create edges longer than `MAX_FACTOR` x target.
    pub check_collapse_len: bool,
    pub collapse_min_quality: f64,
    pub min_normal_dot: f64,
    pub stamp: Stamp,
    pub splits: usize,
    pub collapses: usize,
    pub flips: usize,
    pub timings: PassTimings,
}

impl<'m> Remesher<'m> {
    pub fn new(mesh: &'m mut Mesh, s: Setup) -> Self {
        let mut region = Region::new(mesh, s.region);
        let mut stamp = Stamp::default();
        let edges = region.edges(mesh, &mut stamp);
        let verts = region.vertices(mesh, &mut stamp);
        let cons = build_constraints(mesh, &region, &edges, &verts, &s.constraints);
        let base = if s.base.is_finite() && s.base > 0.0 {
            s.base
        } else if edges.is_empty() {
            1.0
        } else {
            edges.iter().map(|&e| mesh.edge_length(e)).sum::<f64>() / edges.len() as f64
        };
        let base = if base > 0.0 { base } else { 1.0 };

        let mut vsize = vec![base; mesh.max_vid()];
        let reference = if s.project || s.adaptive {
            let (rmesh, vmap) = if region.enabled() {
                let tris = region.triangles_with_collar(mesh);
                let (m, vmap) = mesh.submesh(&tris);
                (m, Some(vmap))
            } else {
                (mesh.clone(), None)
            };
            let bvh = Bvh::build(&rmesh);
            let size = if s.adaptive { compute_sizing(&rmesh, base) } else { Vec::new() };
            if s.adaptive {
                for &v in &verts {
                    let rv = vmap.as_ref().map_or(v, |m| m[v as usize]);
                    if rv != INVALID {
                        vsize[v as usize] = size[rv as usize];
                    }
                }
            }
            Some(Reference { mesh: rmesh, bvh, size })
        } else {
            None
        };

        let mut segs = Vec::new();
        for &e in &edges {
            let c = cons.e[e as usize];
            if c.kind == E_FEATURE {
                let [a, b] = mesh.edge_vertices(e);
                segs.push(Seg { a: mesh.vertex(a), b: mesh.vertex(b), sa: vsize[a as usize], sb: vsize[b as usize], curve: c.curve });
            }
        }
        let curves = CurveIndex::new(segs, cons.curve_count);
        let curve_smooth = cons.smooth;

        Remesher {
            mesh,
            region,
            vc: cons.v,
            ec: cons.e,
            vsize,
            base,
            adaptive: s.adaptive,
            free_boundary: s.free_boundary,
            smoothing: s.smoothing,
            project: s.project,
            curves,
            curve_smooth,
            group_smoothing: s.group_smoothing,
            reference,
            check_collapse_len: true,
            collapse_min_quality: 0.1,
            min_normal_dot: 0.17, // ~80 degrees
            stamp,
            splits: 0,
            collapses: 0,
            flips: 0,
            timings: PassTimings::default(),
        }
    }

    /// Run full iterations; returns the number completed.
    pub fn run(&mut self, iterations: usize, prog: &mut Progress) -> usize {
        let n = iterations.max(1) as f32;
        let mut done = 0;
        for it in 0..iterations {
            let f = |k: f32| (it as f32 + k / 4.0) / n;
            if !prog.report(f(0.0)) {
                break;
            }
            let t = Instant::now();
            self.split_pass(MAX_FACTOR, prog, f(0.0));
            self.timings.split += t.elapsed();
            if !prog.ok() {
                break;
            }
            let t = Instant::now();
            self.collapse_pass(MIN_FACTOR, prog, f(1.0));
            self.timings.collapse += t.elapsed();
            if !prog.ok() || !prog.report(f(2.0)) {
                break;
            }
            let t = Instant::now();
            self.flip_pass(prog, f(2.0));
            self.timings.flip += t.elapsed();
            if !prog.ok() || !prog.report(f(3.0)) {
                break;
            }
            let t = Instant::now();
            self.smooth_project_pass();
            self.timings.smooth_project += t.elapsed();
            done += 1;
        }
        done
    }

    fn grow(&mut self) {
        let nv = self.mesh.max_vid();
        if self.vc.len() < nv {
            self.vc.resize(nv, VCons::FREE);
            self.vsize.resize(nv, self.base);
        }
        let ne = self.mesh.max_eid();
        if self.ec.len() < ne {
            self.ec.resize(ne, ECons::NONE);
        }
    }

    #[inline]
    fn h(&self, v: u32) -> f64 {
        self.vsize[v as usize]
    }

    #[inline]
    fn edge_target(&self, e: u32) -> f64 {
        let [a, b] = self.mesh.edge_vertices(e);
        0.5 * (self.h(a) + self.h(b))
    }

    #[inline]
    fn split_ratio(&self, e: u32, factor: f64) -> f64 {
        if self.ec[e as usize].kind == E_FIXED {
            return 0.0;
        }
        self.mesh.edge_length(e) / (factor * self.edge_target(e))
    }

    // ------------------------------------------------------------------
    // Split
    // ------------------------------------------------------------------

    /// Split splittable edges longer than `factor` x target, most-overlong first. Edges
    /// created by a split are re-queued while still too long, so one pass refines a
    /// coarse mesh all the way down (except where `split_deferred` holds back).
    pub fn split_pass(&mut self, factor: f64, prog: &mut Progress, frac: f32) {
        let edges = self.region.edges(self.mesh, &mut self.stamp);
        let mut heap: BinaryHeap<(OrdF64, u32)> = edges
            .iter()
            .filter_map(|&e| {
                let r = self.split_ratio(e, factor);
                (r > 1.0).then_some((OrdF64(r), e))
            })
            .collect();
        let mut ops = 0usize;
        while let Some((OrdF64(r), e)) = heap.pop() {
            if !self.mesh.is_edge(e) {
                continue;
            }
            let cur = self.split_ratio(e, factor);
            if cur <= 1.0 || (cur - r).abs() > 1e-12 * r {
                continue;
            }
            if self.split_deferred(e, factor) {
                continue;
            }
            let len = self.mesh.edge_length(e);
            let Some(info) = self.split(e) else { continue };
            // Only re-queue edges that got strictly shorter: next to fixed long edges,
            // splits can otherwise cycle on collinear vertices forever.
            for ne in [info.e_split, info.e_new_b, info.e_new_opp[0], info.e_new_opp[1]] {
                if ne != INVALID {
                    let r2 = self.split_ratio(ne, factor);
                    if r2 > 1.0 && self.mesh.edge_length(ne) < 0.99 * len {
                        heap.push((OrdF64(r2), ne));
                    }
                }
            }
            ops += 1;
            if ops.is_multiple_of(16384) && !prog.report(frac) {
                break;
            }
        }
    }

    /// Longest-edge-first rule (Rivara): when an adjacent triangle still has an edge more
    /// in need of splitting than `e`, that edge is unsplittable (fixed) or deferred, and
    /// splitting `e` would only push the new vertex towards it, cascading into slivers.
    fn split_deferred(&self, e: u32, factor: f64) -> bool {
        let m = &*self.mesh;
        let r = m.edge_length(e) / (factor * self.edge_target(e));
        m.edge_triangles(e).iter().any(|&t| {
            t != INVALID
                && m.tri_edges(t)
                    .iter()
                    .any(|&f| f != e && m.edge_length(f) / (factor * self.edge_target(f)) > r * (1.0 + 1e-9))
        })
    }

    /// Split edge `e` at its midpoint, propagating constraints, sizes and region flags.
    pub fn split(&mut self, e: u32) -> Option<EdgeSplitInfo> {
        let c = self.ec[e as usize];
        if c.kind == E_FIXED {
            return None;
        }
        let [a, b] = self.mesh.edge_vertices(e);
        let info = self.mesh.split_edge(e).ok()?;
        self.grow();
        let v = info.v_new as usize;
        self.vc[v] = if c.kind == E_FEATURE { VCons::on_curve(c.curve) } else { VCons::FREE };
        self.vsize[v] = 0.5 * (self.vsize[a as usize] + self.vsize[b as usize]);
        self.ec[info.e_new_b as usize] = c;
        for k in 0..2 {
            if info.e_new_opp[k] != INVALID {
                self.ec[info.e_new_opp[k] as usize] = ECons::NONE;
            }
            if info.t_new[k] != INVALID {
                self.region.add(info.t_new[k]);
            }
        }
        self.splits += 1;
        Some(info)
    }

    // ------------------------------------------------------------------
    // Collapse
    // ------------------------------------------------------------------

    /// Collapse edges shorter than `factor` x target, shortest first (one sweep).
    pub fn collapse_pass(&mut self, factor: f64, prog: &mut Progress, frac: f32) -> usize {
        let edges = self.region.edges(self.mesh, &mut self.stamp);
        let mut cands: Vec<(f64, u32)> = edges
            .iter()
            .filter_map(|&e| {
                if self.ec[e as usize].kind == E_FIXED {
                    return None;
                }
                let r = self.mesh.edge_length(e) / (factor * self.edge_target(e));
                (r < 1.0).then_some((r, e))
            })
            .collect();
        cands.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        let mut n = 0;
        for (i, &(_, e)) in cands.iter().enumerate() {
            if !self.mesh.is_edge(e) {
                continue;
            }
            if self.mesh.edge_length(e) >= factor * self.edge_target(e) {
                continue;
            }
            if self.try_collapse(e) {
                n += 1;
            }
            if i % 16384 == 16383 && !prog.report(frac) {
                break;
            }
        }
        n
    }

    /// Collapse edge `e` if constraints, topology and geometry allow it.
    pub fn try_collapse(&mut self, e: u32) -> bool {
        let mesh = &*self.mesh;
        let c = self.ec[e as usize];
        if c.kind == E_FIXED {
            return false;
        }
        let [a, b] = mesh.edge_vertices(e);
        let ra = removable(self.vc[a as usize], c);
        let rb = removable(self.vc[b as usize], c);
        if !ra && !rb {
            return false;
        }
        let pa = mesh.vertex(a);
        let pb = mesh.vertex(b);
        let mid = (pa + pb) * 0.5;
        let (keep, rem, pos) = if ra && rb {
            if c.kind == E_FEATURE {
                (a, b, mid)
            } else {
                let be = mesh.is_boundary_edge(e);
                let ba = !be && mesh.is_boundary_vertex(a);
                let bb = !be && mesh.is_boundary_vertex(b);
                match (ba, bb) {
                    (true, false) => (a, b, pa),
                    (false, true) => (b, a, pb),
                    _ => (a, b, mid),
                }
            }
        } else if ra {
            (b, a, pb)
        } else {
            (a, b, pa)
        };
        // Cheapest tests first: most candidates fail the edge-length rule.
        if !self.collapse_lengths_ok(keep, rem, pos) || mesh.can_collapse(keep, rem).is_err() {
            return false;
        }
        let [t0, t1] = mesh.edge_triangles(e);
        for o in mesh.edge_opposite_vertices(e) {
            if o == INVALID {
                continue;
            }
            if let Some(x) = mesh.find_edge(rem, o) {
                if self.ec[x as usize].kind != E_NONE {
                    return false;
                }
            }
        }
        if !collapse_shape_ok(mesh, keep, rem, pos, t0, t1, self.min_normal_dot, self.collapse_min_quality) {
            return false;
        }
        let new_size = if ra && rb { 0.5 * (self.h(a) + self.h(b)) } else { self.h(keep) };
        if self.mesh.collapse_edge(keep, rem).is_err() {
            return false;
        }
        self.mesh.set_vertex(keep, pos);
        self.vsize[keep as usize] = new_size;
        self.collapses += 1;
        true
    }

    /// Botsch–Kobbelt rule: the collapse must not create edges longer than
    /// `MAX_FACTOR` x target.
    fn collapse_lengths_ok(&self, keep: u32, rem: u32, pos: DVec3) -> bool {
        if !self.check_collapse_len {
            return true;
        }
        let mesh = &*self.mesh;
        let hk = self.h(keep);
        let ok = |v: u32, skip: u32| {
            mesh.vertex_neighbors(v).all(|x| {
                if x == skip {
                    return true;
                }
                let lim = MAX_FACTOR * 0.5 * (hk + self.h(x));
                pos.distance_squared(mesh.vertex(x)) <= lim * lim
            })
        };
        ok(rem, keep) && (mesh.vertex(keep) == pos || ok(keep, rem))
    }

    // ------------------------------------------------------------------
    // Flip
    // ------------------------------------------------------------------

    /// One sweep of valence-improving edge flips.
    pub fn flip_pass(&mut self, prog: &mut Progress, frac: f32) {
        let edges = self.region.edges(self.mesh, &mut self.stamp);
        for (i, &e) in edges.iter().enumerate() {
            if !self.mesh.is_edge(e) || self.ec[e as usize].kind != E_NONE || self.mesh.is_boundary_edge(e) {
                continue;
            }
            if self.flip_improves_valence(e) && self.flip_geometry_ok(e) && self.mesh.flip_edge(e).is_ok() {
                self.flips += 1;
            }
            if i % 65536 == 65535 && !prog.report(frac) {
                break;
            }
        }
    }

    fn flip_improves_valence(&self, e: u32) -> bool {
        let mesh = &*self.mesh;
        let [a, b] = mesh.edge_vertices(e);
        let [c, d] = mesh.edge_opposite_vertices(e);
        if d == INVALID {
            return false;
        }
        let info = |v: u32| {
            let bnd = mesh.is_boundary_vertex(v);
            (mesh.vertex_valence(v) as i64, if bnd { 4i64 } else { 6i64 }, bnd)
        };
        let (va, ta, ba) = info(a);
        let (vb, tb, bb) = info(b);
        let (vc, tc, _) = info(c);
        let (vd, td, _) = info(d);
        // Never reduce a vertex below valence 3 (interior) / 2 (boundary).
        if va - 1 < if ba { 2 } else { 3 } || vb - 1 < if bb { 2 } else { 3 } {
            return false;
        }
        let sq = |x: i64| x * x;
        let before = sq(va - ta) + sq(vb - tb) + sq(vc - tc) + sq(vd - td);
        let after = sq(va - 1 - ta) + sq(vb - 1 - tb) + sq(vc + 1 - tc) + sq(vd + 1 - td);
        after < before
    }

    fn flip_geometry_ok(&self, e: u32) -> bool {
        let mesh = &*self.mesh;
        let [p, q] = mesh.oriented_edge_vertices(e);
        let [c, d] = mesh.edge_opposite_vertices(e);
        if d == INVALID {
            return false;
        }
        let (pp, pq, pc, pd) = (mesh.vertex(p), mesh.vertex(q), mesh.vertex(c), mesh.vertex(d));
        // Old: (p, q, c), (q, p, d). New (as mm-core builds them): (c, p, d), (d, q, c).
        let qo = quality(pp, pq, pc).min(quality(pq, pp, pd));
        let qn = quality(pc, pp, pd).min(quality(pd, pq, pc));
        if qn < 1e-4 || qn < 0.3 * qo {
            return false;
        }
        let n0 = cross3(pp, pq, pc).normalize_or_zero();
        let n1 = cross3(pq, pp, pd).normalize_or_zero();
        let m0 = cross3(pc, pp, pd).normalize_or_zero();
        let m1 = cross3(pd, pq, pc).normalize_or_zero();
        let avg = (n0 + n1).normalize_or_zero();
        if m0.dot(avg) < 0.1 || m1.dot(avg) < 0.1 {
            return false;
        }
        // Do not increase the bending across the edge (beyond a flat-ish allowance).
        let old_cos = n0.dot(n1);
        let new_cos = m0.dot(m1);
        new_cos >= old_cos.min(0.9) - 1e-9
    }

    // ------------------------------------------------------------------
    // Tangential relaxation + projection
    // ------------------------------------------------------------------

    /// Jacobi-style tangential smoothing followed by projection, in parallel. Moves that
    /// would invert a triangle are reverted.
    pub fn smooth_project_pass(&mut self) {
        let verts: Vec<u32> = {
            let all = self.region.vertices(self.mesh, &mut self.stamp);
            all.into_iter().filter(|&v| self.vc[v as usize].kind != V_FIXED).collect()
        };
        if verts.is_empty() {
            return;
        }
        let mesh: &Mesh = self.mesh;
        let ctx = SmoothCtx {
            mesh,
            vc: &self.vc,
            ec: &self.ec,
            vsize: &self.vsize,
            curves: &self.curves,
            curve_smooth: &self.curve_smooth,
            group_smoothing: self.group_smoothing,
            reference: if self.project || self.adaptive { self.reference.as_ref() } else { None },
            project: self.project,
            lambda: self.smoothing,
            adaptive: self.adaptive,
            free_boundary: self.free_boundary,
        };
        let upd: Vec<(DVec3, f64)> = verts.par_iter().map(|&v| ctx.update(v)).collect();
        let old = mesh.positions();
        let mut np = old.to_vec();
        for (i, &v) in verts.iter().enumerate() {
            np[v as usize] = upd[i].0;
        }
        // Reverting a vertex can invalidate a neighbour's move, so iterate; each round
        // reverts at least one vertex. Fall back to reverting everything if it drags on.
        let mut clean = false;
        for _ in 0..32 {
            let bad: Vec<u32> = verts
                .par_iter()
                .copied()
                .filter(|&v| np[v as usize] != old[v as usize] && fan_degrades(mesh, v, old, &np))
                .collect();
            if bad.is_empty() {
                clean = true;
                break;
            }
            for v in bad {
                np[v as usize] = old[v as usize];
            }
        }
        if !clean {
            for &v in &verts {
                np[v as usize] = old[v as usize];
            }
        }
        let kept: Vec<bool> = verts.iter().enumerate().map(|(i, &v)| np[v as usize] == upd[i].0).collect();
        let pos = self.mesh.positions_mut();
        for &v in &verts {
            pos[v as usize] = np[v as usize];
        }
        self.mesh.bump_shape_version();
        if self.adaptive {
            for (i, &v) in verts.iter().enumerate() {
                if kept[i] {
                    self.vsize[v as usize] = upd[i].1;
                }
            }
        }
    }
}

/// True if moving to `np` inverts or nearly degenerates a triangle around `v`.
fn fan_degrades(mesh: &Mesh, v: u32, old: &[DVec3], np: &[DVec3]) -> bool {
    for t in mesh.vertex_triangles(v) {
        let [a, b, c] = mesh.triangle(t);
        let (oa, ob, oc) = (old[a as usize], old[b as usize], old[c as usize]);
        let (na, nb, nc) = (np[a as usize], np[b as usize], np[c as usize]);
        let qo = quality(oa, ob, oc);
        let qn = quality(na, nb, nc);
        if qo > 1e-6 && cross3(oa, ob, oc).dot(cross3(na, nb, nc)) <= 0.0 {
            return true;
        }
        if qn < 0.01 && qn < 0.25 * qo {
            return true;
        }
    }
    false
}

struct SmoothCtx<'a> {
    mesh: &'a Mesh,
    vc: &'a [VCons],
    ec: &'a [ECons],
    vsize: &'a [f64],
    curves: &'a CurveIndex,
    curve_smooth: &'a [bool],
    group_smoothing: Option<(f64, f64)>,
    reference: Option<&'a Reference>,
    project: bool,
    lambda: f64,
    adaptive: bool,
    free_boundary: bool,
}

impl SmoothCtx<'_> {
    /// New (position, size) for vertex `v`.
    fn update(&self, v: u32) -> (DVec3, f64) {
        let m = self.mesh;
        let p = m.vertex(v);
        let h = self.vsize[v as usize];
        let c = self.vc[v as usize];
        if c.kind == V_CURVE {
            let mut nb = [DVec3::ZERO; 2];
            let mut k = 0;
            for &e in m.vertex_edges(v) {
                let ce = self.ec[e as usize];
                if ce.kind == E_FEATURE && ce.curve == c.curve {
                    if k < 2 {
                        nb[k] = m.vertex(m.edge_other_vertex(e, v));
                    }
                    k += 1;
                }
            }
            if k != 2 {
                return (p, h);
            }
            if let (Some((strength, anchor)), true) = (self.group_smoothing, self.curve_smooth.get(c.curve as usize).copied().unwrap_or(false)) {
                // Smoothed group border: toward the neighbours' midpoint, back toward the
                // original border by `anchor`, then onto the surface.
                let mut ps = p + ((nb[0] + nb[1]) * 0.5 - p) * strength;
                if let Some((q, _)) = self.curves.nearest(c.curve, ps) {
                    ps += (q - ps) * anchor;
                }
                if let Some(r) = self.reference {
                    if let Some(hit) = r.bvh.nearest(&r.mesh, ps, 4.0 * h).or_else(|| r.bvh.nearest(&r.mesh, ps, f64::INFINITY)) {
                        return (hit.point, h);
                    }
                }
                return (ps, h);
            }
            let t = (nb[1] - nb[0]).normalize_or_zero();
            let target = (nb[0] + nb[1]) * 0.5;
            let ps = p + t * (self.lambda * t.dot(target - p));
            return match self.curves.nearest(c.curve, ps) {
                Some((q, s)) => (q, if self.adaptive { s } else { h }),
                None => (ps, h),
            };
        }
        let n = m.vertex_normal(v);
        let ps = if self.free_boundary && m.is_boundary_vertex(v) {
            let mut sum = DVec3::ZERO;
            let mut k = 0;
            for &e in m.vertex_edges(v) {
                if m.is_boundary_edge(e) {
                    sum += m.vertex(m.edge_other_vertex(e, v));
                    k += 1;
                }
            }
            if k == 2 {
                p + (sum * 0.5 - p) * self.lambda
            } else {
                p
            }
        } else {
            let mut sum = DVec3::ZERO;
            let mut w = 0.0;
            for u in m.vertex_neighbors(v) {
                let wu = if self.adaptive { 1.0 / self.vsize[u as usize].max(1e-300) } else { 1.0 };
                sum += m.vertex(u) * wu;
                w += wu;
            }
            if w <= 0.0 {
                return (p, h);
            }
            let d = sum / w - p;
            let d = d - n * n.dot(d);
            p + d * self.lambda
        };
        if let (true, Some(r)) = (self.project, self.reference) {
            let hit = r
                .bvh
                .nearest(&r.mesh, ps, 4.0 * h)
                .or_else(|| r.bvh.nearest(&r.mesh, ps, f64::INFINITY));
            if let Some(hit) = hit {
                if n != DVec3::ZERO && r.mesh.tri_normal(hit.tid).dot(n) < 0.0 {
                    return (p, h);
                }
                let s = if self.adaptive {
                    let [i, j, k] = r.mesh.triangle(hit.tid);
                    r.size[i as usize] * hit.bary.x + r.size[j as usize] * hit.bary.y + r.size[k as usize] * hit.bary.z
                } else {
                    h
                };
                return (hit.point, s);
            }
        }
        (ps, h)
    }
}

/// Curvature-based sizing field on `m`: `base * sqrt(k_ref / k)` clamped to
/// `[0.5, 2] * base`, where `k` is a smoothed max normal curvature estimate per vertex and
/// `k_ref` its median (chord-error-like scaling: edge length ~ curvature^-1/2).
fn compute_sizing(m: &Mesh, base: f64) -> Vec<f64> {
    let n = m.max_vid();
    let normals = m.compute_vertex_normals();
    let mut k: Vec<f64> = (0..n as u32)
        .into_par_iter()
        .map(|v| {
            if !m.is_vertex(v) {
                return 0.0;
            }
            let p = m.vertex(v);
            let nv = normals[v as usize];
            let mut mx = 0.0f64;
            for u in m.vertex_neighbors(v) {
                let d = p - m.vertex(u);
                let l2 = d.length_squared();
                if l2 > 1e-300 {
                    mx = mx.max((nv - normals[u as usize]).dot(d).abs() / l2);
                }
            }
            mx
        })
        .collect();
    for _ in 0..3 {
        let kk = &k;
        k = (0..n as u32)
            .into_par_iter()
            .map(|v| {
                if !m.is_vertex(v) {
                    return 0.0;
                }
                let mut s = kk[v as usize];
                let mut c = 1.0;
                for u in m.vertex_neighbors(v) {
                    s += kk[u as usize];
                    c += 1.0;
                }
                s / c
            })
            .collect();
    }
    let mut vals: Vec<f64> = m.vertex_ids().map(|v| k[v as usize]).collect();
    if vals.is_empty() {
        return vec![base; n];
    }
    let mid = vals.len() / 2;
    vals.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    let diag = m.bounds().diagonal().max(1e-12);
    let kref = vals[mid].max(1.0 / diag);
    k.iter().map(|&kv| base * (kref / kv.max(1e-300)).sqrt().clamp(0.5, 2.0)).collect()
}
