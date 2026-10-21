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
use crate::region::{IdVals, Region, PAR_MIN_LEN};
use crate::stats::{edge_length_stats_in, edge_length_stats_of_triangles, EdgeLengthStats};
use crate::util::*;
use crate::{ProgressFn, RemeshError};
use mm_core::mesh::EdgeSplitInfo;
use mm_core::{Bvh, DVec3, Mesh, INVALID};
use rayon::prelude::*;
use smallvec::SmallVec;
use std::borrow::Cow;
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
    /// Curvature-adaptive sizing: the target varies with curvature around
    /// `target_edge_length`, shorter where the surface is more curved (see `adaptivity`).
    pub adaptive: bool,
    /// Strength of adaptive sizing in `[0, 1]`: 0.5 varies the target between 0.5× and 2×,
    /// 1 between 0.25× and 4×, 0 is uniform.
    pub adaptivity: f64,
    /// Explicit target edge length per input vertex (indexed by vertex id); overrides
    /// `adaptive` sizing. Used for smooth transitions around a remeshed selection.
    pub vertex_sizes: Option<Vec<f64>>,
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
            adaptivity: 0.5,
            vertex_sizes: None,
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
    remesh_with_reference(mesh, params, None, progress)
}

/// [`remesh`] with the projection reference already at hand: `reference` is a mesh with the
/// same content as `mesh` (typically the mesh it was cloned from, which keeps the versions
/// and id ranges) and a fresh [`Bvh::build`] of it. That saves the copy and the tree build of
/// the setup. The reference is used only without a region and only while its versions and id
/// ranges match `mesh` and the tree is current and covers every triangle; otherwise the setup
/// builds its own, as [`remesh`] does.
///
/// The result is the same as [`remesh`]'s only if the tree is exactly what `Bvh::build` makes
/// of the reference: not refit after moving vertices (a refit tree keeps the old layout, and
/// nearest-point ties can then resolve to another triangle) and not a subset (rejected). The
/// versions cannot tell a refit tree apart, so the caller must not pass one. Neither may
/// `reference` have been changed through `positions_mut` without a version bump since.
///
/// A cancel during the setup returns a cancelled report with the mesh unchanged.
pub fn remesh_with_reference(
    mesh: &mut Mesh,
    params: &RemeshParams,
    reference: Option<(&Mesh, &Bvh)>,
    progress: Option<ProgressFn>,
) -> Result<RemeshReport, RemeshError> {
    let start = Instant::now();
    if !params.smoothing.is_finite() {
        return Err(RemeshError::InvalidParameter("smoothing must be finite".into()));
    }
    if let Some(a) = params.preserve_sharp_edges {
        if !(a.is_finite() && a > 0.0 && a < 180.0) {
            return Err(RemeshError::InvalidParameter("crease angle must be in (0, 180) degrees".into()));
        }
    }
    let setup = Setup::of(params);
    let mut prog = Progress::new(progress);
    let reference = reference.filter(|&r| params.region.is_none() && usable_reference(mesh, r));
    let Some(mut r) = Remesher::with_reference(&mut *mesh, setup, reference, &mut prog) else {
        // Cancelled during the setup, before anything changed.
        let mut region = Region::new(mesh, params.region.as_deref());
        let n = region.triangle_count(mesh);
        return Ok(RemeshReport {
            target_edge_length: params.target_edge_length,
            initial_triangles: n,
            final_triangles: n,
            final_vertices: mesh.vertex_count(),
            region_triangles: if region.enabled() { Some(region.triangles(mesh)) } else { None },
            cancelled: true,
            elapsed: start.elapsed(),
            timings: PassTimings { setup: start.elapsed(), ..Default::default() },
            ..Default::default()
        });
    };
    r.timings.setup = start.elapsed();
    let initial_triangles = r.region.triangle_count(r.mesh);
    let mut done = 0;
    if initial_triangles > 0 {
        done = r.run(params.iterations, &mut prog);
    }
    if prog.ok() {
        prog.report(1.0);
    }
    let (final_triangles, edge_lengths, region_triangles) = if r.region.enabled() {
        let tris = r.region.triangles(r.mesh);
        (tris.len(), edge_length_stats_in(r.mesh, &tris), Some(tris))
    } else {
        (r.mesh.triangle_count(), edge_length_stats_of_triangles(r.mesh), None)
    };
    Ok(RemeshReport {
        target_edge_length: r.base,
        iterations: done,
        splits: r.splits,
        collapses: r.collapses,
        flips: r.flips,
        initial_triangles,
        final_triangles,
        final_vertices: r.mesh.vertex_count(),
        edge_lengths,
        region_triangles,
        cancelled: prog.cancelled,
        elapsed: start.elapsed(),
        timings: r.timings,
    })
}

/// Can `(m, bvh)` stand in for the projection reference the setup of `mesh` builds? `m` must
/// hold the same content (see [`same_content`]) and `bvh` be current for it and cover all its
/// triangles (not a [`Bvh::build_subset`]). A refit tree passes these checks but gives other
/// nearest-point ties, so callers pass only fresh [`Bvh::build`]s.
pub(crate) fn usable_reference(mesh: &Mesh, (m, bvh): (&Mesh, &Bvh)) -> bool {
    same_content(m, mesh) && bvh.is_current(m) && bvh.can_refit(m) && bvh.tris.len() == m.triangle_count()
}

/// True if `a` and `b` are the same mesh state: equal versions (unique per edit) and id
/// ranges, as for a mesh and its unedited clone.
pub(crate) fn same_content(a: &Mesh, b: &Mesh) -> bool {
    a.shape_version() == b.shape_version()
        && a.topology_version() == b.topology_version()
        && a.max_vid() == b.max_vid()
        && a.max_tid() == b.max_tid()
        && a.max_eid() == b.max_eid()
        && a.vertex_count() == b.vertex_count()
        && a.triangle_count() == b.triangle_count()
}

/// Configuration for [`Remesher::new`] (shared with the cleanup helpers).
pub(crate) struct Setup<'a> {
    pub region: Option<&'a [u32]>,
    pub constraints: ConstraintOptions,
    /// Target edge length; `<= 0` = mean edge length.
    pub base: f64,
    pub adaptive: bool,
    /// Adaptive sizing strength (see `RemeshParams::adaptivity`).
    pub adaptivity: f64,
    /// Explicit per-input-vertex sizes (see `RemeshParams::vertex_sizes`).
    pub sizes: Option<&'a [f64]>,
    pub project: bool,
    pub smoothing: f64,
    pub free_boundary: bool,
    /// (strength, anchor) for smoothing the flagged group-border curves.
    pub group_smoothing: Option<(f64, f64)>,
}

impl<'a> Setup<'a> {
    /// The setup of [`remesh`] with `params`.
    fn of(params: &'a RemeshParams) -> Self {
        Setup {
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
            adaptivity: params.adaptivity.clamp(0.0, 1.0),
            sizes: params.vertex_sizes.as_deref(),
            project: params.project_to_original,
            smoothing: params.smoothing.clamp(0.0, 1.0),
            free_boundary: params.boundary == BoundaryMode::Free,
            group_smoothing: params
                .smooth_group_boundaries
                .filter(|_| params.preserve_group_boundaries)
                .map(|g| (g.strength.clamp(0.0, 1.0), g.anchor.clamp(0.0, 1.0))),
        }
    }
}

/// Frozen copy of the input surface used for projection and the sizing field (borrowed
/// when the caller already has one, see [`remesh_with_reference`]).
pub(crate) struct Reference<'a> {
    mesh: Cow<'a, Mesh>,
    bvh: Cow<'a, Bvh>,
    /// Sizing field per reference vertex.
    size: Vec<f64>,
}

/// Target edge length of vertex `v`: unset entries are 0 and stand for `base`. Sizes are
/// always positive, and the zero default lets a small region keep only the sizes it sets.
/// `MAPS` says whether the sizes are in a hash map (see [`IdVals::at`]).
#[inline]
fn size_at<const MAPS: bool>(vsize: &IdVals<f64>, base: f64, v: u32) -> f64 {
    let s = vsize.at::<MAPS>(v);
    if s != 0.0 {
        s
    } else {
        base
    }
}

/// Collapse candidates tested together, in parallel, by the collapse sweep. Larger windows
/// mean fewer waits for the worker threads but more candidates tested again: in the first
/// sweep of a 1M-triangle remesh about 6% with 4096, 11% with 16384.
const COLLAPSE_WINDOW: usize = 4096;

/// Sweeps with fewer candidates run serially. The windows pay off on long sweeps (a whole
/// large mesh); on a dab or a selection they cost more than they save, most of all when a
/// coarsening sweep collapses most of its candidates and so tests most of them again (a
/// 30k-triangle reduce dab: 10% slower with windows from 4096 candidates on).
const COLLAPSE_SERIAL_BELOW: usize = 16 * COLLAPSE_WINDOW;

/// A collapse as the sweep decides it: `rem` merges into `keep`, which moves to `pos` and
/// gets target size `size`.
#[derive(Clone, Copy)]
struct Collapse {
    keep: u32,
    rem: u32,
    pos: DVec3,
    size: f64,
}

/// What the collapse sweep does with one candidate.
#[derive(Clone, Copy)]
enum Planned {
    /// Gone or no longer short: passed over (without a progress report).
    Skip,
    /// Tested and rejected.
    Reject,
    Collapse(Collapse),
}

pub(crate) struct Remesher<'m> {
    pub mesh: &'m mut Mesh,
    pub region: Region,
    pub vc: ConsVec<VCons>,
    pub ec: ConsVec<ECons>,
    /// Local target edge length per vertex (0 = `base`, see `size_at`).
    pub vsize: IdVals<f64>,
    pub base: f64,
    pub adaptive: bool,
    pub free_boundary: bool,
    pub smoothing: f64,
    pub project: bool,
    pub curves: CurveIndex,
    /// Per curve: smooth it (group border selected for smoothing).
    pub curve_smooth: Vec<bool>,
    pub group_smoothing: Option<(f64, f64)>,
    pub reference: Option<Reference<'m>>,
    /// Reject collapses that create edges longer than `MAX_FACTOR` x target.
    pub check_collapse_len: bool,
    pub collapse_min_quality: f64,
    pub min_normal_dot: f64,
    pub splits: usize,
    pub collapses: usize,
    pub flips: usize,
    pub timings: PassTimings,
    /// Scratch positions of the smoothing pass, kept between passes so that a region pass
    /// only refreshes the region's vertices.
    np: IdVals<DVec3>,
}

impl<'m> Remesher<'m> {
    pub fn new(mesh: &'m mut Mesh, s: Setup) -> Self {
        Self::with_reference(mesh, s, None, &mut Progress::new(None)).expect("a setup without a progress callback is never cancelled")
    }

    /// [`Remesher::new`] that borrows a ready projection reference (checked by the caller,
    /// see [`remesh_with_reference`]) and checks `prog` between the setup steps. `None`
    /// means cancelled; the mesh is unchanged then.
    pub fn with_reference(mesh: &'m mut Mesh, s: Setup, given: Option<(&'m Mesh, &'m Bvh)>, prog: &mut Progress) -> Option<Self> {
        let mut region = Region::new(mesh, s.region);
        let edges = region.edges(mesh);
        let verts = region.vertices(mesh);
        let cons = build_constraints(mesh, &region, &edges, &verts, &s.constraints);
        if !prog.report(0.0) {
            return None;
        }
        let base = if s.base.is_finite() && s.base > 0.0 {
            s.base
        } else if edges.is_empty() {
            1.0
        } else {
            edges.iter().map(|&e| mesh.edge_length(e)).sum::<f64>() / edges.len() as f64
        };
        let base = if base > 0.0 { base } else { 1.0 };

        let mut vsize = IdVals::new(region.is_sparse(), with_headroom(mesh.max_vid()));
        // Explicit sizes behave like adaptive sizing with a given field.
        let adaptive = s.adaptive || s.sizes.is_some();
        let reference = if s.project || adaptive {
            let (rmesh, bvh, vmap, collar) = if region.enabled() {
                let tris = region.triangles_with_collar(mesh);
                let (m, vmap) = mesh.submesh(&tris);
                let bvh = Bvh::build(&m);
                (Cow::Owned(m), Cow::Owned(bvh), Some(vmap), tris)
            } else if let Some((m, bvh)) = given {
                (Cow::Borrowed(m), Cow::Borrowed(bvh), None, Vec::new())
            } else {
                let m = mesh.clone();
                let bvh = Bvh::build(&m);
                (Cow::Owned(m), Cow::Owned(bvh), None, Vec::new())
            };
            if !prog.report(0.0) {
                return None;
            }
            let size = match s.sizes {
                Some(input) => {
                    // Reference vertex -> input vertex, then its given size.
                    let mut out = vec![base; rmesh.max_vid()];
                    let mut put = |v: u32| {
                        let rv = vmap.as_ref().map_or(v, |m| m.get(v as usize).copied().unwrap_or(INVALID));
                        if rv != INVALID && (rv as usize) < out.len() {
                            out[rv as usize] = input.get(v as usize).copied().filter(|x| x.is_finite() && *x > 0.0).unwrap_or(base);
                        }
                    };
                    if vmap.is_some() {
                        // Only the vertices of the reference triangles map anywhere.
                        for &t in &collar {
                            mesh.triangle(t).into_iter().for_each(&mut put);
                        }
                    } else {
                        mesh.vertex_ids().for_each(put);
                    }
                    out
                }
                None if s.adaptive => compute_sizing(&rmesh, base, s.adaptivity),
                None => Vec::new(),
            };
            if adaptive {
                for &v in &verts {
                    let rv = vmap.as_ref().map_or(v, |m| m[v as usize]);
                    if rv != INVALID {
                        vsize.set(v, size[rv as usize]);
                    }
                }
            }
            Some(Reference { mesh: rmesh, bvh, size })
        } else {
            None
        };

        let mut segs = Vec::new();
        for &e in &edges {
            let c = cons.e.get(e);
            if c.kind == E_FEATURE {
                let [a, b] = mesh.edge_vertices(e);
                let (sa, sb) = (size_at::<true>(&vsize, base, a), size_at::<true>(&vsize, base, b));
                segs.push(Seg { a: mesh.vertex(a), b: mesh.vertex(b), sa, sb, curve: c.curve });
            }
        }
        let curves = CurveIndex::new(segs, cons.curve_count);
        let curve_smooth = cons.smooth;

        Some(Remesher {
            mesh,
            region,
            vc: cons.v,
            ec: cons.e,
            vsize,
            base,
            adaptive,
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
            splits: 0,
            collapses: 0,
            flips: 0,
            timings: PassTimings::default(),
            np: IdVals::from_vec(Vec::new()),
        })
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

    /// After a split: when splits made a small region a good part of the mesh, its per-id
    /// state moves from hash maps into arrays (see [`Region::densify_if_large`]).
    fn grow(&mut self) {
        if self.region.densify_if_large(self.mesh) {
            let (nv, ne) = (with_headroom(self.mesh.max_vid()), with_headroom(self.mesh.max_eid()));
            self.vc.densify(nv);
            self.ec.densify(ne);
            self.vsize.densify(nv);
        }
    }

    // The passes are compiled once per storage of the per-id state: `MAPS` is true when it
    // lives in hash maps (a small region, see `IdVals::at`). Each pass dispatches on
    // `self.region.is_sparse()`, which holds exactly when the state is in hash maps. A split
    // can move it into arrays (`grow`) in the middle of a pass compiled for maps, which still
    // reads it correctly; it never moves the other way.

    #[inline]
    fn h<const MAPS: bool>(&self, v: u32) -> f64 {
        size_at::<MAPS>(&self.vsize, self.base, v)
    }

    #[inline]
    fn edge_target<const MAPS: bool>(&self, e: u32) -> f64 {
        let [a, b] = self.mesh.edge_vertices(e);
        0.5 * (self.h::<MAPS>(a) + self.h::<MAPS>(b))
    }

    #[inline]
    fn split_ratio<const MAPS: bool>(&self, e: u32, factor: f64) -> f64 {
        if self.ec.at::<MAPS>(e).kind == E_FIXED {
            return 0.0;
        }
        self.mesh.edge_length(e) / (factor * self.edge_target::<MAPS>(e))
    }

    // ------------------------------------------------------------------
    // Split
    // ------------------------------------------------------------------

    /// Split splittable edges longer than `factor` x target, most-overlong first. Edges
    /// created by a split are re-queued while still too long, so one pass refines a
    /// coarse mesh all the way down (except where `split_deferred` holds back).
    pub fn split_pass(&mut self, factor: f64, prog: &mut Progress, frac: f32) {
        if self.region.is_sparse() {
            self.split_pass_in::<true>(factor, prog, frac)
        } else {
            self.split_pass_in::<false>(factor, prog, frac)
        }
    }

    fn split_pass_in<const MAPS: bool>(&mut self, factor: f64, prog: &mut Progress, frac: f32) {
        let edges = self.region.edges(self.mesh);
        let this = &*self;
        // Scored in parallel but collected in edge order, so the heap is the one a serial
        // collect builds.
        let cands: Vec<(OrdF64, u32)> = edges
            .par_iter()
            .with_min_len(PAR_MIN_LEN)
            .filter_map(|&e| {
                let r = this.split_ratio::<MAPS>(e, factor);
                (r > 1.0).then_some((OrdF64(r), e))
            })
            .collect();
        let mut heap = BinaryHeap::from(cands);
        let mut ops = 0usize;
        while let Some((OrdF64(r), e)) = heap.pop() {
            if !self.mesh.is_edge(e) {
                continue;
            }
            let cur = self.split_ratio::<MAPS>(e, factor);
            if cur <= 1.0 || (cur - r).abs() > 1e-12 * r {
                continue;
            }
            if self.split_deferred::<MAPS>(e, factor) {
                continue;
            }
            let len = self.mesh.edge_length(e);
            let Some(info) = self.split::<MAPS>(e) else { continue };
            // Only re-queue edges that got strictly shorter: next to fixed long edges,
            // splits can otherwise cycle on collinear vertices forever.
            for ne in [info.e_split, info.e_new_b, info.e_new_opp[0], info.e_new_opp[1]] {
                if ne != INVALID {
                    let r2 = self.split_ratio::<MAPS>(ne, factor);
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
    fn split_deferred<const MAPS: bool>(&self, e: u32, factor: f64) -> bool {
        let m = &*self.mesh;
        let r = m.edge_length(e) / (factor * self.edge_target::<MAPS>(e));
        m.edge_triangles(e).iter().any(|&t| {
            t != INVALID
                && m.tri_edges(t)
                    .iter()
                    .any(|&f| f != e && m.edge_length(f) / (factor * self.edge_target::<MAPS>(f)) > r * (1.0 + 1e-9))
        })
    }

    /// Split edge `e` at its midpoint, propagating constraints, sizes and region flags.
    fn split<const MAPS: bool>(&mut self, e: u32) -> Option<EdgeSplitInfo> {
        let c = self.ec.at::<MAPS>(e);
        if c.kind == E_FIXED {
            return None;
        }
        let [a, b] = self.mesh.edge_vertices(e);
        let info = self.mesh.split_edge(e).ok()?;
        self.grow();
        let v = info.v_new;
        self.vc.set(v, if c.kind == E_FEATURE { VCons::on_curve(c.curve) } else { VCons::FREE });
        let size = 0.5 * (self.h::<MAPS>(a) + self.h::<MAPS>(b));
        self.vsize.set(v, size);
        self.ec.set(info.e_new_b, c);
        for k in 0..2 {
            if info.e_new_opp[k] != INVALID {
                self.ec.set(info.e_new_opp[k], ECons::NONE);
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
        self.collapse_sweep(factor, prog, frac, COLLAPSE_WINDOW, true, COLLAPSE_SERIAL_BELOW)
    }

    /// [`Remesher::collapse_pass`] with windows of `window` candidates (0: serial), going
    /// serial for the rest of the sweep when the windows wait for threads if `fall_back`.
    /// Fewer than `serial_below` candidates are swept serially from the start.
    fn collapse_sweep(&mut self, factor: f64, prog: &mut Progress, frac: f32, window: usize, fall_back: bool, serial_below: usize) -> usize {
        if self.region.is_sparse() {
            self.collapse_sweep_in::<true>(factor, prog, frac, window, fall_back, serial_below)
        } else {
            self.collapse_sweep_in::<false>(factor, prog, frac, window, fall_back, serial_below)
        }
    }

    fn collapse_sweep_in<const MAPS: bool>(
        &mut self,
        factor: f64,
        prog: &mut Progress,
        frac: f32,
        window: usize,
        fall_back: bool,
        serial_below: usize,
    ) -> usize {
        let edges = self.region.edges(self.mesh);
        let this = &*self;
        // Collected in parallel in edge order, then the same serial sort as before.
        let mut cands: Vec<(f64, u32)> = edges
            .par_iter()
            .with_min_len(PAR_MIN_LEN)
            .filter_map(|&e| {
                if this.ec.at::<MAPS>(e).kind == E_FIXED {
                    return None;
                }
                let r = this.mesh.edge_length(e) / (factor * this.edge_target::<MAPS>(e));
                (r < 1.0).then_some((r, e))
            })
            .collect();
        cands.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        // The sweep goes through the candidates in order, each one tested against the mesh as
        // the earlier collapses left it. A test reads only the edge's two vertices and their
        // neighbours (positions, sizes, edges and the triangles around them). A collapse moves
        // the kept vertex and changes the edges and triangles of the removed vertex and of its
        // neighbours (the kept one among them), nothing else. So each window of candidates is
        // tested in parallel against the mesh at the window's start, and the serial commit
        // trusts a test unless an earlier collapse of the window marked one of the vertices it
        // read; then it tests again. The collapses are exactly those of a serial sweep, for any
        // split into windows.
        let mut parallel = window > 0 && cands.len() >= serial_below;
        let mut marks = IdVals::<u32>::new(self.region.is_sparse(), if parallel { self.mesh.max_vid() } else { 0 });
        let mut n = 0;
        let (mut start, mut epoch) = (0, 0u32);
        while start < cands.len() {
            let end = if parallel { (start + window).min(cands.len()) } else { cands.len() };
            let batch = &cands[start..end];
            let this = &*self;
            let t = Instant::now();
            let planned: Option<Vec<([u32; 2], Planned)>> = parallel.then(|| {
                batch
                    .par_iter()
                    .map(|&(_, e)| {
                        let ends = if this.mesh.is_edge(e) { this.mesh.edge_vertices(e) } else { [INVALID; 2] };
                        (ends, this.plan_collapse::<MAPS>(e, factor))
                    })
                    .collect()
            });
            let testing = t.elapsed();
            let t = Instant::now();
            epoch += 1;
            for (j, &(_, e)) in batch.iter().enumerate() {
                let plan = match &planned {
                    Some(planned) => {
                        let (ends, plan) = planned[j];
                        // An edge dead at the window's start stays dead (collapses create nothing).
                        let stale = ends[0] != INVALID && {
                            let m = &*self.mesh;
                            let marked = |v: u32| marks.at::<MAPS>(v) == epoch;
                            ends.iter().any(|&v| marked(v) || m.vertex_neighbors(v).any(marked))
                        };
                        if stale {
                            self.plan_collapse::<MAPS>(e, factor)
                        } else {
                            plan
                        }
                    }
                    None => self.plan_collapse::<MAPS>(e, factor),
                };
                match plan {
                    Planned::Skip => continue,
                    Planned::Reject => {}
                    Planned::Collapse(c) => {
                        // The removed vertex's neighbours (the kept one and the two opposite
                        // ones among them) get edges re-pointed, added or dropped.
                        let changed: SmallVec<[u32; 16]> =
                            if parallel { self.mesh.vertex_neighbors(c.rem).collect() } else { SmallVec::new() };
                        if self.commit_collapse(c) {
                            n += 1;
                            if parallel {
                                marks.set_at::<MAPS>(c.rem, epoch);
                                for x in changed {
                                    marks.set_at::<MAPS>(x, epoch);
                                }
                            }
                        }
                    }
                }
                if (start + j) % 16384 == 16383 && !prog.report(frac) {
                    return n;
                }
            }
            // On a busy machine each window waits for the worker threads to be scheduled,
            // which shows as testing slower than committing; then the rest of the sweep runs
            // serially, as that wait would cost more than the parallel tests save.
            parallel &= !fall_back || testing <= t.elapsed();
            start = end;
        }
        n
    }

    /// What the collapse sweep does with candidate `e`, without changing anything.
    fn plan_collapse<const MAPS: bool>(&self, e: u32, factor: f64) -> Planned {
        if !self.mesh.is_edge(e) || self.mesh.edge_length(e) >= factor * self.edge_target::<MAPS>(e) {
            return Planned::Skip;
        }
        match self.collapse_plan::<MAPS>(e) {
            Some(c) => Planned::Collapse(c),
            None => Planned::Reject,
        }
    }

    /// How edge `e` would collapse, if constraints, topology and geometry allow it.
    fn collapse_plan<const MAPS: bool>(&self, e: u32) -> Option<Collapse> {
        let mesh = &*self.mesh;
        let c = self.ec.at::<MAPS>(e);
        if c.kind == E_FIXED {
            return None;
        }
        let [a, b] = mesh.edge_vertices(e);
        let ra = removable(self.vc.at::<MAPS>(a), c);
        let rb = removable(self.vc.at::<MAPS>(b), c);
        if !ra && !rb {
            return None;
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
        if !self.collapse_lengths_ok::<MAPS>(keep, rem, pos) || mesh.can_collapse(keep, rem).is_err() {
            return None;
        }
        let [t0, t1] = mesh.edge_triangles(e);
        for o in mesh.edge_opposite_vertices(e) {
            if o == INVALID {
                continue;
            }
            if let Some(x) = mesh.find_edge(rem, o) {
                if self.ec.at::<MAPS>(x).kind != E_NONE {
                    return None;
                }
            }
        }
        if !collapse_shape_ok(mesh, keep, rem, pos, t0, t1, self.min_normal_dot, self.collapse_min_quality) {
            return None;
        }
        let size = if ra && rb { 0.5 * (self.h::<MAPS>(a) + self.h::<MAPS>(b)) } else { self.h::<MAPS>(keep) };
        Some(Collapse { keep, rem, pos, size })
    }

    fn commit_collapse(&mut self, c: Collapse) -> bool {
        if self.mesh.collapse_edge(c.keep, c.rem).is_err() {
            return false;
        }
        self.mesh.set_vertex(c.keep, c.pos);
        self.vsize.set(c.keep, c.size);
        self.collapses += 1;
        true
    }

    /// Botsch–Kobbelt rule: the collapse must not create edges longer than
    /// `MAX_FACTOR` x target.
    fn collapse_lengths_ok<const MAPS: bool>(&self, keep: u32, rem: u32, pos: DVec3) -> bool {
        if !self.check_collapse_len {
            return true;
        }
        let mesh = &*self.mesh;
        let hk = self.h::<MAPS>(keep);
        let ok = |v: u32, skip: u32| {
            mesh.vertex_neighbors(v).all(|x| {
                if x == skip {
                    return true;
                }
                let lim = MAX_FACTOR * 0.5 * (hk + self.h::<MAPS>(x));
                pos.distance_squared(mesh.vertex(x)) <= lim * lim
            })
        };
        ok(rem, keep) && (mesh.vertex(keep) == pos || ok(keep, rem))
    }

    // ------------------------------------------------------------------
    // Flip
    // ------------------------------------------------------------------

    /// One sweep of valence-improving edge flips, in edge order.
    ///
    /// Whether an edge wants a flip depends only on its two triangles (their corners and
    /// positions), the valences of those four corners and whether they lie on the boundary.
    /// No vertex moves during the sweep and a flip never changes which vertices lie on the
    /// boundary, so every edge is tested once up front, in parallel, and the sweep re-tests
    /// only the edges that passed then or whose triangles or corners an earlier flip of the
    /// sweep touched. That makes exactly the flips of testing every edge in turn.
    pub fn flip_pass(&mut self, prog: &mut Progress, frac: f32) {
        if self.region.is_sparse() {
            self.flip_pass_in::<true>(prog, frac)
        } else {
            self.flip_pass_in::<false>(prog, frac)
        }
    }

    fn flip_pass_in<const MAPS: bool>(&mut self, prog: &mut Progress, frac: f32) {
        let edges = self.region.edges(self.mesh);
        let m: &Mesh = self.mesh;
        // Boundary flags per vertex, unless the region is a small part of a large mesh.
        let bnd: Vec<bool> = if self.region.enabled() && edges.len() * 16 < m.max_eid() {
            Vec::new()
        } else {
            (0..m.max_vid() as u32).into_par_iter().with_min_len(PAR_MIN_LEN).map(|v| m.is_vertex(v) && m.is_boundary_vertex(v)).collect()
        };
        let bnd = (!bnd.is_empty()).then_some(&bnd[..]);
        let this = &*self;
        let wanted: Vec<bool> = edges.par_iter().with_min_len(PAR_MIN_LEN).map(|&e| this.flip_wanted::<MAPS>(e, bnd)).collect();
        let mut touched = IdVals::<bool>::new(self.region.is_sparse(), self.mesh.max_eid());
        for (i, &e) in edges.iter().enumerate() {
            if wanted[i] || touched.at::<MAPS>(e) {
                if self.flip_wanted::<MAPS>(e, bnd) {
                    if let Ok(info) = self.mesh.flip_edge(e) {
                        self.flips += 1;
                        // The flip changed the valences of these four vertices and the corners
                        // of the edge's two triangles, which lie around them: re-test every
                        // edge of their triangles.
                        let m = &*self.mesh;
                        for v in [info.v_old[0], info.v_old[1], info.v_new[0], info.v_new[1]] {
                            for &x in m.vertex_edges(v) {
                                for t in m.edge_triangles(x) {
                                    if t != INVALID {
                                        for y in m.tri_edges(t) {
                                            touched.set_at::<MAPS>(y, true);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                debug_assert!(!self.flip_wanted::<MAPS>(e, bnd), "edge {e} skipped but wants a flip");
            }
            if i % 65536 == 65535 && !prog.report(frac) {
                break;
            }
        }
    }

    /// Would the sweep flip edge `e` (before `Mesh::flip_edge`'s own checks)? `bnd` holds
    /// the boundary flag per vertex when the caller computed them.
    ///
    /// Kept out of line with its two tests inlined, as the compiler lays it out without the
    /// per-mode copies; calling the tests instead slows the flip pass down by several percent.
    #[inline(never)]
    fn flip_wanted<const MAPS: bool>(&self, e: u32, bnd: Option<&[bool]>) -> bool {
        let m = &*self.mesh;
        m.is_edge(e)
            && self.ec.at::<MAPS>(e).kind == E_NONE
            && !m.is_boundary_edge(e)
            && self.flip_improves_valence(e, bnd)
            && self.flip_geometry_ok(e)
    }

    #[inline(always)]
    fn flip_improves_valence(&self, e: u32, bnd: Option<&[bool]>) -> bool {
        let mesh = &*self.mesh;
        let [a, b] = mesh.edge_vertices(e);
        let [c, d] = mesh.edge_opposite_vertices(e);
        if d == INVALID {
            return false;
        }
        let info = |v: u32| {
            let bnd = match bnd {
                Some(flags) => flags[v as usize],
                None => mesh.is_boundary_vertex(v),
            };
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

    #[inline(always)]
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
        if self.region.is_sparse() {
            self.smooth_project_pass_in::<true>()
        } else {
            self.smooth_project_pass_in::<false>()
        }
    }

    fn smooth_project_pass_in<const MAPS: bool>(&mut self) {
        let all = self.region.vertices(self.mesh);
        let verts: Vec<u32> = all.iter().copied().filter(|&v| self.vc.at::<MAPS>(v).kind != V_FIXED).collect();
        if verts.is_empty() {
            return;
        }
        let region_tris = if self.region.enabled() { self.region.triangles(self.mesh) } else { Vec::new() };
        let sparse = self.region.is_sparse();
        let mesh: &Mesh = self.mesh;
        let old = mesh.positions();
        // `np` must hold the current position of every vertex a moving fan reads. The
        // triangles of a movable vertex all lie in the region, so in region mode refreshing
        // the region's vertices is enough (in a hash map for a small region).
        let np = &mut self.np;
        if self.region.enabled() {
            if !sparse {
                // An array over every vertex, sized once (the values move over if the region
                // has grown out of hash maps).
                np.densify(old.len());
            } else if np.is_sparse() {
                np.clear();
            } else {
                *np = IdVals::new(true, 0);
            }
            for &v in &all {
                np.set(v, old[v as usize]);
            }
        } else {
            np.copy_from(old);
        }
        let ctx = SmoothCtx {
            mesh,
            vc: &self.vc,
            ec: &self.ec,
            vsize: &self.vsize,
            base: self.base,
            curves: &self.curves,
            curve_smooth: &self.curve_smooth,
            group_smoothing: self.group_smoothing,
            reference: if self.project || self.adaptive { self.reference.as_ref() } else { None },
            project: self.project,
            lambda: self.smoothing,
            adaptive: self.adaptive,
            free_boundary: self.free_boundary,
        };
        // Unlike the scans, the updates stay fully parallel even for a few hundred vertices:
        // each projects onto the reference, which outweighs handing work to the pool (a
        // serial update made remesh dabs 10-30% slower).
        let upd: Vec<(DVec3, f64)> = verts.par_iter().map(|&v| ctx.update::<MAPS>(v)).collect();
        for (i, &v) in verts.iter().enumerate() {
            np.set(v, upd[i].0);
        }
        // Reverting a vertex can invalidate a neighbour's move, so iterate; each round
        // reverts at least one vertex. Fall back to reverting everything if it drags on.
        //
        // A move is bad when a triangle of the vertex's fan degrades, which depends on that
        // triangle's corners only, so each triangle is tested once (not once per corner) and
        // only the triangles of reverted vertices are tested again. After the first round
        // only the neighbours of the vertices just reverted can turn bad. The movable
        // vertices' triangles all lie in the region, so only those need testing.
        let mut tri_bad: IdVals<bool> = if self.region.enabled() {
            let cur = &*np;
            let flags: Vec<bool> = region_tris.par_iter().with_min_len(PAR_MIN_LEN).map(|&t| tri_degrades::<MAPS>(mesh, t, old, cur)).collect();
            let mut out = IdVals::new(sparse, mesh.max_tid());
            for (&t, &f) in region_tris.iter().zip(&flags) {
                if f {
                    out.set(t, true);
                }
            }
            out
        } else {
            let cur = &*np;
            IdVals::from_vec(
                (0..mesh.max_tid() as u32)
                    .into_par_iter()
                    .with_min_len(PAR_MIN_LEN)
                    .map(|t| mesh.is_triangle(t) && tri_degrades::<MAPS>(mesh, t, old, cur))
                    .collect(),
            )
        };
        let fan_bad = |v: u32, tri_bad: &IdVals<bool>| {
            mesh.vertex_edges(v).iter().any(|&e| mesh.edge_triangles(e).iter().any(|&t| t != INVALID && tri_bad.at::<MAPS>(t)))
        };
        let mut clean = false;
        let mut check: Vec<u32> = Vec::new();
        for round in 0..32 {
            let todo: &[u32] = if round == 0 { &verts } else { &check };
            let cur = &*np;
            let flags = &tri_bad;
            let bad: Vec<u32> =
                todo.par_iter().with_min_len(PAR_MIN_LEN).copied().filter(|&v| cur.at::<MAPS>(v) != old[v as usize] && fan_bad(v, flags)).collect();
            if bad.is_empty() {
                clean = true;
                break;
            }
            for &v in &bad {
                np.set(v, old[v as usize]);
            }
            for &v in &bad {
                for t in mesh.vertex_triangles(v) {
                    tri_bad.set(t, tri_degrades::<MAPS>(mesh, t, old, np));
                }
            }
            check.clear();
            for &v in &bad {
                check.extend(mesh.vertex_neighbors(v));
            }
            check.sort_unstable();
            check.dedup();
        }
        if !clean {
            for &v in &verts {
                np.set(v, old[v as usize]);
            }
        }
        let kept: Vec<bool> = verts.iter().enumerate().map(|(i, &v)| np.get(v) == upd[i].0).collect();
        let pos = self.mesh.positions_mut();
        for &v in &verts {
            pos[v as usize] = self.np.get(v);
        }
        self.mesh.bump_shape_version();
        if self.adaptive {
            for (i, &v) in verts.iter().enumerate() {
                if kept[i] {
                    self.vsize.set(v, upd[i].1);
                }
            }
        }
    }
}

/// True if moving the corners from `old` to `np` inverts or nearly degenerates triangle `t`.
#[inline]
fn tri_degrades<const MAPS: bool>(mesh: &Mesh, t: u32, old: &[DVec3], np: &IdVals<DVec3>) -> bool {
    let [a, b, c] = mesh.triangle(t);
    let (oa, ob, oc) = (old[a as usize], old[b as usize], old[c as usize]);
    let (na, nb, nc) = (np.at::<MAPS>(a), np.at::<MAPS>(b), np.at::<MAPS>(c));
    let qo = quality(oa, ob, oc);
    let qn = quality(na, nb, nc);
    if qo > 1e-6 && cross3(oa, ob, oc).dot(cross3(na, nb, nc)) <= 0.0 {
        return true;
    }
    qn < 0.01 && qn < 0.25 * qo
}

struct SmoothCtx<'a> {
    mesh: &'a Mesh,
    vc: &'a ConsVec<VCons>,
    ec: &'a ConsVec<ECons>,
    vsize: &'a IdVals<f64>,
    base: f64,
    curves: &'a CurveIndex,
    curve_smooth: &'a [bool],
    group_smoothing: Option<(f64, f64)>,
    reference: Option<&'a Reference<'a>>,
    project: bool,
    lambda: f64,
    adaptive: bool,
    free_boundary: bool,
}

impl SmoothCtx<'_> {
    /// New (position, size) for vertex `v`.
    fn update<const MAPS: bool>(&self, v: u32) -> (DVec3, f64) {
        let m = self.mesh;
        let p = m.vertex(v);
        let h = size_at::<MAPS>(self.vsize, self.base, v);
        let c = self.vc.at::<MAPS>(v);
        if c.kind == V_CURVE {
            let mut nb = [DVec3::ZERO; 2];
            let mut k = 0;
            for &e in m.vertex_edges(v) {
                let ce = self.ec.at::<MAPS>(e);
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
                let wu = if self.adaptive { 1.0 / size_at::<MAPS>(self.vsize, self.base, u).max(1e-300) } else { 1.0 };
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
fn compute_sizing(m: &Mesh, base: f64, adaptivity: f64) -> Vec<f64> {
    let n = m.max_vid();
    let normals = m.compute_vertex_normals();
    let mut k: Vec<f64> = (0..n as u32)
        .into_par_iter()
        .with_min_len(PAR_MIN_LEN)
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
            .with_min_len(PAR_MIN_LEN)
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
    // adaptivity 0.5: sqrt(kref / k) within [0.5, 2]; 1: kref / k within [0.25, 4].
    let (exp, range) = (adaptivity, 2f64.powf(4.0 * adaptivity));
    k.iter().map(|&kv| base * (kref / kv.max(1e-300)).powf(exp).clamp(1.0 / range.sqrt(), range.sqrt())).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_core::primitives;
    use std::cell::Cell;

    type RawState = (Vec<(u32, [u64; 3], Vec<u32>)>, Vec<(u32, [u32; 3], [u32; 3], i32)>, Vec<(u32, [u32; 2], [u32; 2])>);

    /// Everything about the live elements, by id (including the order of the edge lists).
    fn raw(m: &Mesh) -> RawState {
        let v = m
            .vertex_ids()
            .map(|v| {
                let p = m.vertex(v);
                (v, [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()], m.vertex_edges(v).to_vec())
            })
            .collect();
        let t = m.triangle_ids().map(|t| (t, m.triangle(t), m.tri_edges(t), m.tri_group(t))).collect();
        let e = m.edge_ids().map(|e| (e, m.edge_vertices(e), m.edge_triangles(e))).collect();
        (v, t, e)
    }

    fn noisy(mut m: Mesh, amp: f64, seed: u64) -> Mesh {
        let mut s = seed;
        let mut rnd = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
        };
        for v in m.vertex_ids().collect::<Vec<_>>() {
            let p = m.vertex(v);
            m.set_vertex(v, p + DVec3::new(rnd(), rnd(), rnd()) * amp);
        }
        m
    }

    /// Meshes and parameters covering closed and open surfaces, feature curves, regions and
    /// adaptive sizing, with enough noise for smoothing moves to be reverted.
    fn cases() -> Vec<(Mesh, RemeshParams)> {
        let sphere = noisy(primitives::icosphere(1.0, 4), 0.02, 1);
        let mut grouped = noisy(primitives::cylinder(1.0, 3.0, 40, 24, false), 0.02, 2);
        for t in grouped.triangle_ids().collect::<Vec<_>>() {
            let g = (grouped.tri_centroid(t).x > 0.3) as i32;
            grouped.set_tri_group(t, g);
        }
        let grid = noisy(primitives::plane_grid(4.0, 4.0, 40, 40), 0.04, 3);
        // In-plane jitter of nearly half an edge: smoothing moves collide, and reverting one
        // vertex makes a neighbour's move bad in the next round.
        let mut rough = primitives::plane_grid(4.0, 4.0, 40, 40);
        let mut s = 5u64;
        for v in rough.vertex_ids().collect::<Vec<_>>() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let a = ((s >> 11) as f64) / ((1u64 << 53) as f64) * std::f64::consts::TAU;
            let p = rough.vertex(v);
            rough.set_vertex(v, p + DVec3::new(a.cos(), a.sin(), 0.0) * 0.045);
        }
        let avg = |m: &Mesh| m.average_edge_length();
        let region: Vec<u32> = sphere.triangle_ids().filter(|&t| sphere.tri_centroid(t).z > -0.2).collect();
        let small: Vec<u32> = sphere.triangle_ids().filter(|&t| sphere.tri_centroid(t).z > 0.9).collect();
        vec![
            (sphere.clone(), RemeshParams { smoothing: 1.0, ..RemeshParams::with_target(0.8 * avg(&sphere)) }),
            (sphere.clone(), RemeshParams { adaptive: true, adaptivity: 0.8, ..RemeshParams::with_target(avg(&sphere)) }),
            (sphere.clone(), RemeshParams { region: Some(region), smoothing: 1.0, ..RemeshParams::with_target(0.6 * avg(&sphere)) }),
            (sphere.clone(), RemeshParams { region: Some(small), ..RemeshParams::with_target(0.6 * avg(&sphere)) }),
            (
                grouped.clone(),
                RemeshParams {
                    boundary: BoundaryMode::ConstrainToBoundary,
                    preserve_sharp_edges: Some(40.0),
                    smooth_group_boundaries: Some(GroupBorderSmoothing { strength: 0.5, anchor: 0.1, split: None }),
                    ..RemeshParams::with_target(1.3 * avg(&grouped))
                },
            ),
            (grid.clone(), RemeshParams { boundary: BoundaryMode::Free, smoothing: 1.0, ..RemeshParams::with_target(0.7 * avg(&grid)) }),
            (grid.clone(), RemeshParams { project_to_original: false, smoothing: 1.0, ..RemeshParams::with_target(1.5 * avg(&grid)) }),
            (rough.clone(), RemeshParams { project_to_original: false, smoothing: 1.0, ..RemeshParams::with_target(avg(&rough)) }),
            (rough.clone(), RemeshParams { smoothing: 1.0, ..RemeshParams::with_target(0.9 * avg(&rough)) }),
        ]
    }

    impl Remesher<'_> {
        /// The collapse sweep as it was: every candidate tested and collapsed in turn.
        fn collapse_pass_serial(&mut self, factor: f64) -> usize {
            let edges = self.region.edges(self.mesh);
            let mut cands: Vec<(f64, u32)> = edges
                .iter()
                .filter_map(|&e| {
                    if self.ec.get(e).kind == E_FIXED {
                        return None;
                    }
                    let r = self.mesh.edge_length(e) / (factor * self.edge_target::<true>(e));
                    (r < 1.0).then_some((r, e))
                })
                .collect();
            cands.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
            let mut n = 0;
            for &(_, e) in &cands {
                if !self.mesh.is_edge(e) || self.mesh.edge_length(e) >= factor * self.edge_target::<true>(e) {
                    continue;
                }
                if let Some(c) = self.collapse_plan::<true>(e) {
                    n += self.commit_collapse(c) as usize;
                }
            }
            n
        }

        /// The flip sweep as it was: every edge tested in turn.
        fn flip_pass_every_edge(&mut self) {
            let edges = self.region.edges(self.mesh);
            for &e in &edges {
                if !self.mesh.is_edge(e) || self.ec.get(e).kind != E_NONE || self.mesh.is_boundary_edge(e) {
                    continue;
                }
                if self.flip_improves_valence(e, None) && self.flip_geometry_ok(e) && self.mesh.flip_edge(e).is_ok() {
                    self.flips += 1;
                }
            }
        }

        /// The smoothing pass as it was: a fresh copy of every position, and every vertex's
        /// whole fan tested again each round. Returns the number of rounds that reverted.
        fn smooth_project_pass_every_fan(&mut self) -> usize {
            let fan_degrades =
                |mesh: &Mesh, v: u32, old: &[DVec3], np: &IdVals<DVec3>| mesh.vertex_triangles(v).iter().any(|&t| tri_degrades::<true>(mesh, t, old, np));
            let verts: Vec<u32> = {
                let all = self.region.vertices(self.mesh);
                all.into_iter().filter(|&v| self.vc.get(v).kind != V_FIXED).collect()
            };
            if verts.is_empty() {
                return 0;
            }
            let mesh: &Mesh = self.mesh;
            let ctx = SmoothCtx {
                mesh,
                vc: &self.vc,
                ec: &self.ec,
                vsize: &self.vsize,
                base: self.base,
                curves: &self.curves,
                curve_smooth: &self.curve_smooth,
                group_smoothing: self.group_smoothing,
                reference: if self.project || self.adaptive { self.reference.as_ref() } else { None },
                project: self.project,
                lambda: self.smoothing,
                adaptive: self.adaptive,
                free_boundary: self.free_boundary,
            };
            let upd: Vec<(DVec3, f64)> = verts.par_iter().map(|&v| ctx.update::<true>(v)).collect();
            let old = mesh.positions();
            let mut np = IdVals::from_vec(old.to_vec());
            for (i, &v) in verts.iter().enumerate() {
                np.set(v, upd[i].0);
            }
            let mut clean = false;
            let mut rounds = 0;
            for _ in 0..32 {
                let bad: Vec<u32> =
                    verts.iter().copied().filter(|&v| np.get(v) != old[v as usize] && fan_degrades(mesh, v, old, &np)).collect();
                if bad.is_empty() {
                    clean = true;
                    break;
                }
                rounds += 1;
                for v in bad {
                    np.set(v, old[v as usize]);
                }
            }
            if !clean {
                for &v in &verts {
                    np.set(v, old[v as usize]);
                }
            }
            let kept: Vec<bool> = verts.iter().enumerate().map(|(i, &v)| np.get(v) == upd[i].0).collect();
            let pos = self.mesh.positions_mut();
            for &v in &verts {
                pos[v as usize] = np.get(v);
            }
            self.mesh.bump_shape_version();
            if self.adaptive {
                for (i, &v) in verts.iter().enumerate() {
                    if kept[i] {
                        self.vsize.set(v, upd[i].1);
                    }
                }
            }
            rounds
        }
    }

    #[test]
    fn passes_match_the_serial_sweeps() {
        let mut multi_round = 0;
        for (k, (mesh, params)) in cases().into_iter().enumerate() {
            let (mut a, mut b) = (mesh.clone(), mesh.clone());
            let mut ra = Remesher::new(&mut a, Setup::of(&params));
            let mut rb = Remesher::new(&mut b, Setup::of(&params));
            let mut prog = Progress::new(None);
            for it in 0..5 {
                for r in [&mut ra, &mut rb] {
                    r.split_pass(MAX_FACTOR, &mut prog, 0.0);
                }
                // Windows of several sizes, and none: the same collapses for any split. Many
                // small windows run on one thread (the tests do not depend on timing), so that
                // a busy machine does not slow the test down.
                let window = [16, 128, 1000, 4096, 0][it];
                let n = if window < 1000 {
                    let one = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
                    one.install(|| ra.collapse_sweep(MIN_FACTOR, &mut Progress::new(None), 0.0, window, false, 0))
                } else {
                    ra.collapse_sweep(MIN_FACTOR, &mut prog, 0.0, window, false, 0)
                };
                assert_eq!(n, rb.collapse_pass_serial(MIN_FACTOR), "case {k} iteration {it}: collapses");
                assert!(raw(ra.mesh) == raw(rb.mesh), "case {k} iteration {it}: collapse pass differs");
                assert!(ra.vsize == rb.vsize, "case {k} iteration {it}: sizes differ after collapsing");
                ra.flip_pass(&mut prog, 0.0);
                rb.flip_pass_every_edge();
                assert_eq!(ra.flips, rb.flips, "case {k} iteration {it}: flips");
                assert!(raw(ra.mesh) == raw(rb.mesh), "case {k} iteration {it}: flip pass differs");
                ra.smooth_project_pass();
                if rb.smooth_project_pass_every_fan() > 1 {
                    multi_round += 1;
                }
                assert!(raw(ra.mesh) == raw(rb.mesh), "case {k} iteration {it}: smoothing pass differs");
                assert!(ra.vsize == rb.vsize, "case {k} iteration {it}: sizes differ");
            }
            assert!(ra.flips > 0, "case {k} made no flips");
        }
        assert!(multi_round > 0, "no smoothing pass needed a second round of reverts");
    }

    #[test]
    fn a_given_reference_gives_the_same_remesh() {
        let src = noisy(primitives::icosphere(1.0, 4), 0.01, 7);
        let bvh = Bvh::build(&src);
        let h = src.average_edge_length();
        let sizes: Vec<f64> = (0..src.max_vid()).map(|v| h * (0.7 + 0.1 * (v % 5) as f64)).collect();
        let mut other = src.clone();
        other.set_vertex(0, other.vertex(0) * 1.2);
        let other_bvh = Bvh::build(&other);
        for params in [
            RemeshParams { iterations: 4, ..RemeshParams::with_target(0.8 * h) },
            RemeshParams { iterations: 3, adaptive: true, ..RemeshParams::with_target(h) },
            RemeshParams { iterations: 3, vertex_sizes: Some(sizes.clone()), ..RemeshParams::with_target(h) },
        ] {
            let mut a = src.clone();
            let ra = remesh(&mut a, &params, None).unwrap();
            let mut b = src.clone();
            let rb = remesh_with_reference(&mut b, &params, Some((&src, &bvh)), None).unwrap();
            assert!(raw(&a) == raw(&b), "a given reference changed the result");
            assert_eq!((ra.splits, ra.collapses, ra.flips, ra.final_triangles), (rb.splits, rb.collapses, rb.flips, rb.final_triangles));
            assert_eq!(ra.edge_lengths, rb.edge_lengths);
            // A reference of another state is not used (it would project onto other positions).
            let mut c = src.clone();
            remesh_with_reference(&mut c, &params, Some((&other, &other_bvh)), None).unwrap();
            assert!(raw(&a) == raw(&c), "a stale reference was used");
            // And the matching one is borrowed, not copied.
            let mut d = src.clone();
            let r = Remesher::with_reference(&mut d, Setup::of(&params), Some((&src, &bvh)), &mut Progress::new(None)).unwrap();
            assert!(matches!(r.reference.as_ref().map(|x| &x.mesh), Some(Cow::Borrowed(_))));
        }
        // A tree over only some of the triangles is not taken for the reference's own.
        let some: Vec<u32> = src.triangle_ids().step_by(2).collect();
        assert!(usable_reference(&src, (&src, &bvh)));
        assert!(!usable_reference(&src, (&src, &Bvh::build_subset(&src, &some))));
        assert!(!usable_reference(&src, (&other, &other_bvh)));
        // Reduce to a deviation bound: the same with a given reference.
        let p = crate::ReduceParams::with_target(crate::ReduceTarget::MaxDeviation(0.01));
        let mut a = src.clone();
        let ra = crate::reduce(&mut a, &p, None).unwrap();
        let mut b = src.clone();
        let rb = crate::reduce::reduce_with_reference(&mut b, &p, Some((&src, &bvh)), None).unwrap();
        assert!(raw(&a) == raw(&b));
        assert_eq!(ra.max_deviation, rb.max_deviation);
        let mut c = src.clone();
        crate::reduce::reduce_with_reference(&mut c, &p, Some((&other, &other_bvh)), None).unwrap();
        assert!(raw(&a) == raw(&c), "a stale reference was used");
    }

    /// Runs `f` with the per-id state of every region kept as its size picks, all in hash
    /// maps, and all in arrays: the results must agree.
    fn same_in_maps_and_arrays<R: PartialEq>(what: &str, f: impl Fn() -> R) {
        let run = |force: Option<bool>| {
            crate::region::FORCE_SPARSE.with(|c| c.set(force));
            let r = f();
            crate::region::FORCE_SPARSE.with(|c| c.set(None));
            r
        };
        let auto = run(None);
        assert!(run(Some(true)) == auto, "{what}: hash maps give another result");
        assert!(run(Some(false)) == auto, "{what}: arrays give another result");
    }

    /// A noisy icosphere with two face groups.
    fn grouped_sphere(level: usize, noise: f64) -> Mesh {
        let mut m = noisy(primitives::icosphere(1.0, level), noise, 11);
        for t in m.triangle_ids().collect::<Vec<_>>() {
            let g = (m.tri_centroid(t).x > 0.1) as i32;
            m.set_tri_group(t, g);
        }
        m
    }

    #[test]
    fn region_state_in_maps_or_arrays_gives_the_same_results() {
        let coarse = grouped_sphere(4, 0.02);
        let fine = grouped_sphere(6, 0.005);
        let cap = |m: &Mesh, z: f64| -> Vec<u32> { m.triangle_ids().filter(|&t| m.tri_centroid(t).z > z).collect() };
        // A cap small enough for hash maps when chosen by size, a small region in arrays (it
        // still builds its lists from its own triangles) and a large one.
        let tiny = cap(&fine, 0.997);
        assert!(Region::new(&fine, Some(&tiny)).is_sparse() && !Region::new(&coarse, Some(&cap(&coarse, 0.9))).is_sparse());
        for (base, region) in [(&fine, tiny.clone()), (&coarse, cap(&coarse, 0.9)), (&coarse, cap(&coarse, -0.2))] {
            let h = base.average_edge_length();
            let sizes: Vec<f64> = (0..base.max_vid()).map(|v| h * (0.6 + 0.1 * (v % 4) as f64)).collect();
            for (k, params) in [
                RemeshParams::with_target(0.8 * h),
                // Fine enough for the small region to grow past the switch to arrays.
                RemeshParams { iterations: 4, ..RemeshParams::with_target(0.3 * h) },
                RemeshParams { adaptive: true, adaptivity: 0.8, ..RemeshParams::with_target(h) },
                RemeshParams { vertex_sizes: Some(sizes.clone()), ..RemeshParams::with_target(h) },
                RemeshParams {
                    boundary: BoundaryMode::ConstrainToBoundary,
                    preserve_sharp_edges: Some(30.0),
                    smooth_group_boundaries: Some(GroupBorderSmoothing { strength: 0.5, anchor: 0.1, split: None }),
                    ..RemeshParams::with_target(0.7 * h)
                },
                RemeshParams { boundary: BoundaryMode::Free, project_to_original: false, smoothing: 1.0, ..RemeshParams::with_target(1.4 * h) },
            ]
            .into_iter()
            .enumerate()
            {
                same_in_maps_and_arrays(&format!("remesh {k} on {} triangles", region.len()), || {
                    let mut m = base.clone();
                    let r = remesh(&mut m, &RemeshParams { region: Some(region.clone()), ..params.clone() }, None).unwrap();
                    (raw(&m), r.splits, r.collapses, r.flips, r.final_triangles, r.final_vertices, r.edge_lengths, r.region_triangles)
                });
            }
            // Refine and collapse as the density brushes run them.
            same_in_maps_and_arrays("refine and collapse", || {
                let mut m = base.clone();
                let a = crate::refine_uniform(&mut m, 0.3 * h, Some(&region), None).unwrap();
                let tris = a.region_triangles.clone().unwrap();
                let c = crate::collapse_short_edges(&mut m, 0.5 * h, Some(&tris), None).unwrap();
                (raw(&m), a.splits, c.collapses, c.region_triangles)
            });
            for (k, p) in [
                crate::ReduceParams { target: crate::ReduceTarget::Percentage(60.0), max_normal_change_deg: 6.0, ..Default::default() },
                crate::ReduceParams::with_target(crate::ReduceTarget::Percentage(20.0)),
                crate::ReduceParams { uniform: true, preserve_boundary: false, ..crate::ReduceParams::with_target(crate::ReduceTarget::Percentage(30.0)) },
                crate::ReduceParams::with_target(crate::ReduceTarget::MaxDeviation(0.01)),
            ]
            .into_iter()
            .enumerate()
            {
                same_in_maps_and_arrays(&format!("reduce {k} on {} triangles", region.len()), || {
                    let mut m = base.clone();
                    let r = crate::reduce(&mut m, &crate::ReduceParams { region: Some(region.clone()), ..p.clone() }, None).unwrap();
                    (raw(&m), r.collapses, r.final_triangles, r.max_deviation, r.region_triangles)
                });
            }
        }
        // The tiny region does start in hash maps and switches to arrays as it grows.
        let mut m = fine.clone();
        let params = RemeshParams { region: Some(tiny), ..RemeshParams::with_target(0.3 * fine.average_edge_length()) };
        let mut r = Remesher::new(&mut m, Setup::of(&params));
        assert!(r.region.is_sparse() && r.vsize.is_sparse());
        r.split_pass(MAX_FACTOR, &mut Progress::new(None), 0.0);
        assert!(!r.region.is_sparse() && !r.vsize.is_sparse());
    }

    #[test]
    fn state_moving_into_arrays_after_smoothing_gives_the_same_remesh() {
        // The region of the test above switches during its first split pass, before smoothing
        // has filled `np`. Here the switch comes after one or more full iterations, with `np`
        // still a hash map, and the remesh must carry on exactly as if it had stayed in maps.
        let base = grouped_sphere(4, 0.02);
        let h = base.average_edge_length();
        let region: Vec<u32> = base.triangle_ids().filter(|&t| base.tri_centroid(t).z > 0.3).collect();
        for params in [
            RemeshParams { iterations: 4, ..RemeshParams::with_target(0.8 * h) },
            RemeshParams { iterations: 4, adaptive: true, adaptivity: 0.8, ..RemeshParams::with_target(h) },
        ] {
            let params = RemeshParams { region: Some(region.clone()), ..params };
            let force = |f: Option<bool>| crate::region::FORCE_SPARSE.with(|c| c.set(f));
            let run = |switch_before: Option<usize>| {
                let mut m = base.clone();
                force(Some(true));
                let mut r = Remesher::new(&mut m, Setup::of(&params));
                let mut prog = Progress::new(None);
                for it in 0..params.iterations {
                    if switch_before == Some(it) {
                        force(Some(false));
                        r.grow();
                        assert!(!r.region.is_sparse() && !r.vsize.is_sparse() && r.np.is_sparse(), "iteration {it}: no switch");
                    }
                    assert_eq!(r.run(1, &mut prog), 1);
                }
                force(None);
                let sizes: Vec<u64> = (0..r.mesh.max_vid() as u32).map(|v| r.vsize.get(v).to_bits()).collect();
                (raw(r.mesh), r.splits, r.collapses, r.flips, sizes)
            };
            let maps = run(None);
            assert!(maps.1 > 0 && maps.2 > 0 && maps.3 > 0);
            for k in [1, 2, 3] {
                assert!(run(Some(k)) == maps, "switching before iteration {k} changed the result (adaptive: {})", params.adaptive);
            }
        }
    }

    #[test]
    fn cancelling_during_setup_leaves_the_mesh_unchanged() {
        let src = noisy(primitives::icosphere(1.0, 4), 0.01, 9);
        let region: Vec<u32> = src.triangle_ids().filter(|&t| src.tri_centroid(t).y > 0.0).collect();
        for region in [None, Some(region)] {
            // Remesh: the first progress call comes after the constraints, the second after
            // the reference. Reduce: before and after its setup.
            for stop_at in [1, 2] {
                let calls = Cell::new(0);
                let cb = |_f: f32| {
                    calls.set(calls.get() + 1);
                    calls.get() < stop_at
                };
                let mut m = src.clone();
                let params = RemeshParams { region: region.clone(), ..RemeshParams::with_target(0.5 * src.average_edge_length()) };
                let rep = remesh(&mut m, &params, Some(&cb)).unwrap();
                assert!(rep.cancelled && rep.iterations == 0 && rep.splits == 0);
                assert_eq!(calls.get(), stop_at);
                assert!(raw(&m) == raw(&src), "cancelled setup changed the mesh");
                assert_eq!(rep.initial_triangles, region.as_ref().map_or(src.triangle_count(), |r| r.len()));
                assert_eq!(rep.region_triangles, region.clone());

                let calls = Cell::new(0);
                let cb = |_f: f32| {
                    calls.set(calls.get() + 1);
                    calls.get() < stop_at
                };
                let mut m = src.clone();
                let p = crate::ReduceParams { region: region.clone(), ..crate::ReduceParams::with_target(crate::ReduceTarget::MaxDeviation(0.01)) };
                let rep = crate::reduce(&mut m, &p, Some(&cb)).unwrap();
                assert!(rep.cancelled && rep.collapses == 0);
                assert_eq!(calls.get(), stop_at);
                assert!(raw(&m) == raw(&src), "cancelled reduce setup changed the mesh");
                assert_eq!(rep.region_triangles, region.clone());
            }
        }
    }
}
