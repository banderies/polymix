//! Plane cuts (Meshmixer *Edit > Plane Cut*) and the local, single-branch cut used to trim
//! vessels.
//!
//! Algorithm: signed distances are computed for every vertex; distances at round-off level
//! count as zero, and an endpoint of a crossing edge whose intersection point lies within
//! `snap_tolerance × edge length` of it is moved onto the plane instead of creating a
//! sliver. Remaining crossing edges are split exactly at the plane, so afterwards no
//! triangle straddles the plane and every triangle has a side (coplanar triangles take the
//! side their normal faces away from). Triangles on the discarded side are deleted (or, for
//! [`KeepSide::Both`], re-attached to duplicated seam vertices) and the new boundary loops
//! are capped with [`crate::cap`].
//!
//! The local cut first groups the plane/mesh intersection into connected curves (segments
//! linked through shared crossing edges or on-plane vertices), picks the curve nearest to a
//! seed point, splits only that curve, and deletes only what a flood fill from the
//! discarded side — bounded by that curve — reaches.

use crate::cap::cap_oriented_loops;
use crate::util::*;
use crate::{EditError, EditResult};
use mm_core::hash::{FastMap, FastSet};
use mm_core::{Aabb, Bvh, DVec2, DVec3, Mesh, Plane, INVALID};
use smallvec::SmallVec;
use std::collections::{HashMap, HashSet};

/// One side of a cutting plane; `Above` is the side the plane normal points to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Side {
    /// Positive signed distance.
    Above,
    /// Negative signed distance.
    Below,
}

impl Side {
    /// The other side.
    pub fn opposite(self) -> Side {
        match self {
            Side::Above => Side::Below,
            Side::Below => Side::Above,
        }
    }
}

/// What to keep after a cut.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeepSide {
    /// Keep the part on the side the plane normal points to.
    Above,
    /// Keep the part behind the plane.
    Below,
    /// Slice: keep both parts, separated along the plane (seam vertices are duplicated).
    Both,
}

/// How to close the boundary loops created by a cut.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CutFill {
    /// Leave the cut open.
    None,
    /// Planar constrained-Delaunay triangulation using only the loop vertices.
    Flat,
    /// Planar triangulation refined so cap edges are roughly `target_edge_length`.
    FlatRemeshed {
        /// Desired cap edge length (mesh units).
        target_edge_length: f64,
    },
}

/// Parameters for [`plane_cut`] / [`local_plane_cut_with`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlaneCutParams {
    /// Which part(s) to keep.
    pub keep: KeepSide,
    /// How to close the new boundary loops.
    pub fill: CutFill,
    /// Sliver prevention, as a fraction of edge length (0 ≤ f < 0.5): when the plane meets
    /// an edge within `f × |edge|` of an endpoint, that endpoint is moved onto the plane
    /// instead of splitting the edge. Distances below 1e-9 × the bounding-box diagonal
    /// always count as "on the plane". Default 0.01.
    pub snap_tolerance: f64,
}

impl Default for PlaneCutParams {
    fn default() -> Self {
        Self { keep: KeepSide::Above, fill: CutFill::Flat, snap_tolerance: 0.01 }
    }
}

/// Outcome of a plane cut.
#[derive(Clone, Debug, Default)]
pub struct PlaneCutResult {
    /// Boundary loops created by the cut (vertex ids in boundary orientation, i.e. as in the
    /// adjacent triangles). With [`KeepSide::Both`] every cut curve yields one loop per piece.
    /// A loop may also contain pre-existing boundary edges when the plane crossed an open
    /// boundary; such loops are not planar and are never capped.
    pub cut_loops: Vec<Vec<u32>>,
    /// The piece each loop in `cut_loops` bounds.
    pub loop_sides: Vec<Side>,
    /// Whether each loop in `cut_loops` was capped.
    pub loop_capped: Vec<bool>,
    /// Triangles added by capping.
    pub cap_triangles: Vec<u32>,
    /// Face group of each cap region.
    pub cap_groups: Vec<i32>,
    /// Number of triangles deleted.
    pub removed_triangles: usize,
    /// Vertices inserted on crossing edges.
    pub split_vertices: Vec<u32>,
    /// Existing vertices moved onto the plane to avoid slivers.
    pub snapped_vertices: Vec<u32>,
}

/// Outcome of [`local_plane_cut`].
#[derive(Clone, Debug, Default)]
pub struct LocalCutResult {
    /// Loops, caps and counts, as for a global cut.
    pub cut: PlaneCutResult,
    /// Whether the chosen intersection curve was closed (only closed curves are capped).
    pub curve_closed: bool,
    /// Distance from the seed point to the chosen curve.
    pub curve_distance: f64,
}

/// A connected component of the plane/mesh intersection (for previews and picking).
#[derive(Clone, Debug, Default)]
pub struct IntersectionCurve {
    /// Unordered line segments; consistently oriented along the curve.
    pub segments: Vec<[DVec3; 2]>,
    /// True if the curve has no open ends (it does not run into a mesh boundary).
    pub closed: bool,
    /// Total segment length.
    pub length: f64,
    /// Enclosed area (closed curves; 0 otherwise).
    pub area: f64,
    /// Area centroid for closed curves, length-weighted centroid otherwise.
    pub centroid: DVec3,
}

impl IntersectionCurve {
    /// Distance from `p` to the curve.
    pub fn distance(&self, p: DVec3) -> f64 {
        self.segments.iter().map(|s| point_segment_distance(p, s[0], s[1])).fold(f64::INFINITY, f64::min)
    }
    /// Diameter of the circle with the same area.
    pub fn equivalent_diameter(&self) -> f64 {
        2.0 * (self.area / std::f64::consts::PI).sqrt()
    }

    /// The segments in 2D coordinates of their plane (`normal`), about `origin`.
    fn flat(&self, origin: DVec3, normal: DVec3) -> (Vec<[DVec2; 2]>, DVec3, DVec3) {
        let (u, v) = frame(normal);
        let to2 = |p: DVec3| DVec2::new((p - origin).dot(u), (p - origin).dot(v));
        (self.segments.iter().map(|s| [to2(s[0]), to2(s[1])]).collect(), u, v)
    }

    /// Whether `p` (on the curve's plane, whose normal is `normal`) lies inside the curve
    /// (its winding number is not zero; false for open curves).
    pub fn encloses(&self, p: DVec3, normal: DVec3) -> bool {
        self.closed && winding(&self.flat(p, normal).0, DVec2::ZERO) != 0
    }

    /// The centre of the largest circle inside the closed curve (its pole of inaccessibility)
    /// and that circle's radius: a point well inside even a crescent, whose centroid is not.
    /// `None` for open curves.
    pub fn inscribed_centre(&self, normal: DVec3) -> Option<(DVec3, f64)> {
        if !self.closed || self.segments.is_empty() {
            return None;
        }
        let origin = self.segments[0][0];
        let (segs, u, v) = self.flat(origin, normal);
        let (lo, hi) = segs.iter().flatten().fold((DVec2::INFINITY, DVec2::NEG_INFINITY), |(a, b), &p| (a.min(p), b.max(p)));
        let clearance = |q: DVec2| -> f64 {
            if winding(&segs, q) == 0 {
                return -1.0;
            }
            segs.iter().map(|s| point_segment_distance_2d(q, s[0], s[1])).fold(f64::INFINITY, f64::min)
        };
        // A grid over the curve's box, then finer grids around the best point.
        let n = 24;
        let mut cell = (hi - lo) / n as f64;
        let mut best = (-1.0, lo);
        for i in 0..=n {
            for j in 0..=n {
                let q = lo + DVec2::new(i as f64 * cell.x, j as f64 * cell.y);
                let c = clearance(q);
                if c > best.0 {
                    best = (c, q);
                }
            }
        }
        if best.0 <= 0.0 {
            return None;
        }
        for _ in 0..3 {
            let centre = best.1;
            for i in -3..=3 {
                for j in -3..=3 {
                    let q = centre + DVec2::new(i as f64 * cell.x, j as f64 * cell.y) / 3.0;
                    let c = clearance(q);
                    if c > best.0 {
                        best = (c, q);
                    }
                }
            }
            cell /= 3.0;
        }
        Some((origin + u * best.1.x + v * best.1.y, best.0))
    }
}

/// Winding number of the closed polygon `segs` (oriented segments, in any order) about `q`.
fn winding(segs: &[[DVec2; 2]], q: DVec2) -> i32 {
    let turn: f64 = segs.iter().map(|s| (s[0] - q).angle_to(s[1] - q)).sum();
    (turn / std::f64::consts::TAU).round() as i32
}

fn point_segment_distance_2d(p: DVec2, a: DVec2, b: DVec2) -> f64 {
    let ab = b - a;
    let l2 = ab.length_squared();
    let t = if l2 > 0.0 { ((p - a).dot(ab) / l2).clamp(0.0, 1.0) } else { 0.0 };
    p.distance(a + ab * t)
}

pub(crate) fn point_segment_distance(p: DVec3, a: DVec3, b: DVec3) -> f64 {
    let ab = b - a;
    let l2 = ab.length_squared();
    let t = if l2 > 0.0 { ((p - a).dot(ab) / l2).clamp(0.0, 1.0) } else { 0.0 };
    p.distance(a + ab * t)
}

// ----------------------------------------------------------------------------------------
// Signed-distance bookkeeping
// ----------------------------------------------------------------------------------------

pub(crate) struct CutContext {
    pub plane: Plane,
    /// Signed distance by vertex id; exactly 0.0 for vertices treated as on the plane (and for
    /// those a context made for a few triangles did not set). Empty for a context made
    /// [`CutContext::for_triangles_local`], which keeps them in `local`.
    pub sd: Vec<f64>,
    /// The signed distances of a few triangles' vertices (unset ones read as 0.0), so a
    /// context near a click costs as much as the triangles there, not the whole mesh.
    local: Option<FastMap<u32, f64>>,
    /// The entries of `sd` set by [`CutContext::for_triangles_in`], to clear after.
    touched: Vec<u32>,
}

impl CutContext {
    pub fn new(mesh: &Mesh, plane: Plane, eps: f64) -> Self {
        let mut sd = vec![0.0; mesh.max_vid()];
        for v in mesh.vertex_ids() {
            let d = plane.signed_distance(mesh.vertex(v));
            sd[v as usize] = if d.abs() <= eps { 0.0 } else { d };
        }
        Self { plane, sd, local: None, touched: Vec::new() }
    }

    /// Every vertex of `tris` and of their edge neighbours (enough for [`compute_curves`]
    /// restricted to `tris`), with repeats.
    fn each_vertex_near(mesh: &Mesh, tris: &[u32], mut f: impl FnMut(u32)) {
        for &t in tris {
            for e in mesh.tri_edges(t) {
                for o in mesh.edge_triangles(e) {
                    if o != INVALID {
                        mesh.triangle(o).into_iter().for_each(&mut f);
                    }
                }
            }
        }
    }

    /// Context valid only for `tris` and their edge neighbours (enough for
    /// [`compute_curves`] restricted to `tris`); `sd` is a reusable buffer (of any contents: it
    /// is cleared first; see [`CutContext::for_triangles_in`] for one kept all zeros).
    // (Kept for callers that clear their buffer here; the crate's own now keep it zeroed.)
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn for_triangles(mesh: &Mesh, plane: Plane, eps: f64, tris: &[u32], mut sd: Vec<f64>) -> Self {
        sd.clear();
        sd.resize(mesh.max_vid(), 0.0);
        Self::for_triangles_in(mesh, plane, eps, tris, sd)
    }

    /// [`CutContext::for_triangles`] with its distances in a small map rather than a buffer as
    /// long as the mesh: for a few triangles.
    pub fn for_triangles_local(mesh: &Mesh, plane: Plane, eps: f64, tris: &[u32]) -> Self {
        let mut local: FastMap<u32, f64> = FastMap::default();
        Self::each_vertex_near(mesh, tris, |v| {
            local.entry(v).or_insert_with(|| {
                let d = plane.signed_distance(mesh.vertex(v));
                if d.abs() <= eps { 0.0 } else { d }
            });
        });
        Self { plane, sd: Vec::new(), local: Some(local), touched: Vec::new() }
    }

    /// [`CutContext::for_triangles`] with `sd` a buffer of zeros (of any length: it grows to
    /// the mesh's vertices), so many triangles are looked up as fast as in a whole-mesh
    /// context without clearing a buffer as long as the mesh each time; only their entries
    /// are set, and [`CutContext::into_buffer`] clears just those, for the buffer's next use.
    pub fn for_triangles_in(mesh: &Mesh, plane: Plane, eps: f64, tris: &[u32], mut sd: Vec<f64>) -> Self {
        if sd.len() < mesh.max_vid() {
            sd.resize(mesh.max_vid(), 0.0);
        }
        // Entries are 0.0 until set, and set ones are 0.0 only on the plane: a vertex met again
        // with its distance set is skipped (each is met many times, through every triangle
        // around it), and one met at 0.0 is worked out (again) and noted for clearing (so the
        // list holds each once, bar a few on-plane vertices met again).
        let mut touched = Vec::new();
        Self::each_vertex_near(mesh, tris, |v| {
            let slot = &mut sd[v as usize];
            if *slot != 0.0 {
                return;
            }
            touched.push(v);
            let d = plane.signed_distance(mesh.vertex(v));
            *slot = if d.abs() <= eps { 0.0 } else { d };
        });
        Self { plane, sd, local: None, touched }
    }

    /// The buffer of a context made [`CutContext::for_triangles`] or
    /// [`CutContext::for_triangles_in`], all zeros again.
    pub fn into_buffer(mut self) -> Vec<f64> {
        for &v in &self.touched {
            self.sd[v as usize] = 0.0;
        }
        self.sd
    }

    /// Signed distance of `v` (0.0 where it is on the plane, or unknown to this context).
    #[inline]
    fn dist(&self, v: u32) -> f64 {
        match &self.local {
            Some(m) => m.get(&v).copied().unwrap_or(0.0),
            None => self.sd.get(v as usize).copied().unwrap_or(0.0),
        }
    }

    /// Put `v` on the plane in this context.
    fn set_on_plane(&mut self, v: u32) {
        match &mut self.local {
            Some(m) => {
                m.insert(v, 0.0);
            }
            None => {
                if self.sd.len() <= v as usize {
                    self.sd.resize(v as usize + 1, 0.0);
                }
                self.sd[v as usize] = 0.0;
            }
        }
    }

    #[inline]
    pub fn sign(&self, v: u32) -> i8 {
        let d = self.dist(v);
        if d > 0.0 {
            1
        } else if d < 0.0 {
            -1
        } else {
            0
        }
    }

    #[inline]
    pub fn crosses(&self, mesh: &Mesh, e: u32) -> bool {
        let [a, b] = mesh.edge_vertices(e);
        self.sign(a) * self.sign(b) < 0
    }

    /// Parameter of the plane crossing along `edge_vertices(e)[0] → [1]`.
    #[inline]
    pub fn param(&self, mesh: &Mesh, e: u32) -> f64 {
        let [a, b] = mesh.edge_vertices(e);
        let (da, db) = (self.dist(a), self.dist(b));
        (da / (da - db)).clamp(0.0, 1.0)
    }

    pub fn crossing_point(&self, mesh: &Mesh, e: u32) -> DVec3 {
        let [a, b] = mesh.edge_vertices(e);
        let p = mesh.vertex(a).lerp(mesh.vertex(b), self.param(mesh, e));
        self.plane.project(p)
    }

    pub fn snap(&mut self, mesh: &mut Mesh, v: u32) {
        let p = self.plane.project(mesh.vertex(v));
        mesh.set_vertex(v, p);
        self.set_on_plane(v);
    }

    /// Split crossing edge `e` at the plane; returns the new (on-plane) vertex.
    pub fn split(&mut self, mesh: &mut Mesh, e: u32) -> EditResult<u32> {
        let s = self.param(mesh, e);
        let info = mesh.split_edge_at(e, s)?;
        let v = info.v_new;
        let p = self.plane.project(mesh.vertex(v));
        mesh.set_vertex(v, p);
        self.set_on_plane(v);
        Ok(v)
    }

    /// Side of a triangle that does not straddle the plane.
    pub fn tri_side(&self, mesh: &Mesh, t: u32) -> Side {
        let tv = mesh.triangle(t);
        let s = [self.sign(tv[0]), self.sign(tv[1]), self.sign(tv[2])];
        if s.iter().any(|&x| x > 0) {
            Side::Above
        } else if s.iter().any(|&x| x < 0) {
            Side::Below
        } else if mesh.tri_cross(t).dot(self.plane.normal) > 0.0 {
            // Coplanar and facing +n: it bounds material below the plane.
            Side::Below
        } else {
            Side::Above
        }
    }
}

fn snap_eps(mesh: &Mesh) -> f64 {
    1e-9 * mesh.bounds().diagonal()
}

/// The distance within which the cuts and intersection curves count a vertex as on the plane:
/// 1e-9 of the mesh's bounding-box diagonal. Ask once for many curves through one mesh (see
/// [`local_intersection_curves_with`]).
pub fn snap_distance(mesh: &Mesh) -> f64 {
    snap_eps(mesh)
}

/// [`snap_distance`], from `bvh`'s box when it holds every triangle of `mesh` (current) and the
/// mesh's own box is not cached: the same box (the same coordinates, so the same diagonal),
/// without a walk over the mesh.
pub(crate) fn snap_eps_with(mesh: &Mesh, bvh: Option<&Bvh>) -> f64 {
    match bvh {
        Some(b) if b.tris.len() == mesh.triangle_count() && b.is_current(mesh) && mesh.cached_bounds().is_none() => 1e-9 * b.bounds().diagonal(),
        _ => snap_eps(mesh),
    }
}

// ----------------------------------------------------------------------------------------
// Intersection curves
// ----------------------------------------------------------------------------------------

pub(crate) struct Curve {
    pub crossing_edges: Vec<u32>,
    pub plane_vertices: Vec<u32>,
    pub segments: Vec<[DVec3; 2]>,
    pub closed: bool,
}

const VKEY: u64 = 1 << 32;

fn uf_find(p: &mut [usize], mut x: usize) -> usize {
    while p[x] != x {
        p[x] = p[p[x]];
        x = p[x];
    }
    x
}

/// Connected components of the plane/mesh intersection (optionally restricted to a subset
/// of triangles). Does not modify the mesh.
pub(crate) fn compute_curves(mesh: &Mesh, ctx: &CutContext, subset: Option<&[u32]>) -> Vec<Curve> {
    let n = ctx.plane.normal;
    let mut index: FastMap<u64, usize> = FastMap::default();
    let mut keys: Vec<u64> = Vec::new();
    let mut segs: Vec<(usize, usize, [DVec3; 2])> = Vec::new();
    let mut inplane: FastSet<u32> = FastSet::default();

    let elem = |k: u64, index: &mut FastMap<u64, usize>, keys: &mut Vec<u64>| -> usize {
        *index.entry(k).or_insert_with(|| {
            keys.push(k);
            keys.len() - 1
        })
    };
    let orient = |t: u32, a: DVec3, b: DVec3| -> [DVec3; 2] {
        let d = mesh.tri_cross(t).cross(n);
        if (b - a).dot(d) < 0.0 {
            [b, a]
        } else {
            [a, b]
        }
    };

    let mut visit = |t: u32| {
        let tv = mesh.triangle(t);
        let s = [ctx.sign(tv[0]), ctx.sign(tv[1]), ctx.sign(tv[2])];
        let (mn, mx) = (s[0].min(s[1]).min(s[2]), s[0].max(s[1]).max(s[2]));
        if mn > 0 || mx < 0 {
            return;
        }
        let te = mesh.tri_edges(t);
        let zeros = s.iter().filter(|&&x| x == 0).count();
        if zeros >= 2 {
            for j in 0..3 {
                if s[j] == 0 && s[(j + 1) % 3] == 0 {
                    inplane.insert(te[j]);
                }
            }
            return;
        }
        let mut found: SmallVec<[(u64, DVec3); 3]> = SmallVec::new();
        for j in 0..3 {
            if s[j] == 0 {
                found.push((VKEY | tv[j] as u64, mesh.vertex(tv[j])));
            }
            if s[j] * s[(j + 1) % 3] < 0 {
                found.push((te[j] as u64, ctx.crossing_point(mesh, te[j])));
            }
        }
        if found.len() == 2 {
            let ia = elem(found[0].0, &mut index, &mut keys);
            let ib = elem(found[1].0, &mut index, &mut keys);
            segs.push((ia, ib, orient(t, found[0].1, found[1].1)));
        }
    };
    match subset {
        Some(ts) => ts.iter().filter(|&&t| mesh.is_triangle(t)).for_each(|&t| visit(t)),
        None => mesh.triangle_ids().for_each(&mut visit),
    }

    // Edges lying in the plane are part of the curve when their triangles are on opposite sides.
    let mut inplane: Vec<u32> = inplane.into_iter().collect();
    inplane.sort_unstable();
    for e in inplane {
        let [t0, t1] = mesh.edge_triangles(e);
        if t1 == INVALID || ctx.tri_side(mesh, t0) == ctx.tri_side(mesh, t1) {
            continue;
        }
        let [a, b] = mesh.edge_vertices(e);
        let ia = elem(VKEY | a as u64, &mut index, &mut keys);
        let ib = elem(VKEY | b as u64, &mut index, &mut keys);
        let t = if mesh.tri_cross(t0).cross(n).length_squared() > 0.0 { t0 } else { t1 };
        segs.push((ia, ib, orient(t, mesh.vertex(a), mesh.vertex(b))));
    }

    let ne = keys.len();
    let mut parent: Vec<usize> = (0..ne).collect();
    let mut degree = vec![0u32; ne];
    for &(a, b, _) in &segs {
        degree[a] += 1;
        degree[b] += 1;
        let (ra, rb) = (uf_find(&mut parent, a), uf_find(&mut parent, b));
        if ra != rb {
            parent[ra] = rb;
        }
    }
    let mut comp_of_root: Vec<usize> = vec![usize::MAX; ne];
    let mut curves: Vec<Curve> = Vec::new();
    for i in 0..ne {
        let r = uf_find(&mut parent, i);
        if comp_of_root[r] == usize::MAX {
            comp_of_root[r] = curves.len();
            curves.push(Curve { crossing_edges: Vec::new(), plane_vertices: Vec::new(), segments: Vec::new(), closed: true });
        }
        let ci = comp_of_root[r];
        let k = keys[i];
        if k & VKEY != 0 {
            curves[ci].plane_vertices.push((k & 0xffff_ffff) as u32);
        } else {
            curves[ci].crossing_edges.push(k as u32);
        }
        if !degree[i].is_multiple_of(2) {
            curves[ci].closed = false;
        }
    }
    for &(a, _, s) in &segs {
        let r = uf_find(&mut parent, a);
        let ci = comp_of_root[r];
        curves[ci].segments.push(s);
    }
    curves.retain(|c| !c.segments.is_empty());
    curves
}

pub(crate) fn curve_summary(c: &Curve, normal: DVec3) -> IntersectionCurve {
    let length: f64 = c.segments.iter().map(|s| s[0].distance(s[1])).sum();
    let (u, v) = frame(normal);
    let o = c.segments.first().map(|s| s[0]).unwrap_or(DVec3::ZERO);
    let to2 = |p: DVec3| {
        let d = p - o;
        DVec2::new(d.dot(u), d.dot(v))
    };
    let mut a2 = 0.0;
    let mut c2 = DVec2::ZERO;
    let mut lc = DVec3::ZERO;
    for s in &c.segments {
        let (p, q) = (to2(s[0]), to2(s[1]));
        let cr = p.perp_dot(q);
        a2 += cr;
        c2 += (p + q) * cr;
        lc += (s[0] + s[1]) * 0.5 * s[0].distance(s[1]);
    }
    let lcent = if length > 0.0 { lc / length } else { o };
    let (area, centroid) = if c.closed && a2.abs() > 1e-12 * length * length {
        let k = c2 / (3.0 * a2);
        // Lift back to 3D on the plane through the segments.
        let off = (lcent - o).dot(normal);
        ((a2 * 0.5).abs(), o + u * k.x + v * k.y + normal * off)
    } else {
        (0.0, lcent)
    };
    IntersectionCurve { segments: c.segments.clone(), closed: c.closed, length, area, centroid }
}

/// All connected intersection curves of `plane` with the mesh (for previews).
pub fn plane_intersection_curves(mesh: &Mesh, plane: Plane) -> Vec<IntersectionCurve> {
    let Ok(plane) = normalize_plane(plane) else { return Vec::new() };
    let ctx = CutContext::new(mesh, plane, snap_eps(mesh));
    compute_curves(mesh, &ctx, None).iter().map(|c| curve_summary(c, plane.normal)).collect()
}

/// [`plane_intersection_curves`] near `point` only: the curves where `plane` cuts the
/// triangles within `reach` of it (cheap on a big mesh; a curve the reach cuts short is open).
/// `bvh` must be current for `mesh`, or built from a subset of it holding those triangles.
pub fn local_intersection_curves(mesh: &Mesh, bvh: &Bvh, plane: Plane, point: DVec3, reach: f64) -> Vec<IntersectionCurve> {
    local_curves(mesh, bvh, plane, point, reach, None)
}

/// [`local_intersection_curves`] with the on-plane distance given (`snap`, from
/// [`snap_distance`]), for many curves through one mesh.
pub fn local_intersection_curves_with(mesh: &Mesh, bvh: &Bvh, plane: Plane, point: DVec3, reach: f64, snap: f64) -> Vec<IntersectionCurve> {
    local_curves(mesh, bvh, plane, point, reach, Some(snap))
}

fn local_curves(mesh: &Mesh, bvh: &Bvh, plane: Plane, point: DVec3, reach: f64, snap: Option<f64>) -> Vec<IntersectionCurve> {
    let Ok(plane) = normalize_plane(plane) else { return Vec::new() };
    let mut tris = Vec::new();
    bvh.query_aabb(&Aabb::from_points([point - DVec3::splat(reach), point + DVec3::splat(reach)]), |t| {
        let d = mesh.triangle(t).map(|v| plane.signed_distance(mesh.vertex(v)));
        if d.iter().any(|&x| x <= 0.0) && d.iter().any(|&x| x >= 0.0) {
            tris.push(t);
        }
    });
    if tris.is_empty() {
        return Vec::new();
    }
    let eps = snap.unwrap_or_else(|| snap_eps_with(mesh, Some(bvh)));
    let ctx = CutContext::for_triangles_local(mesh, plane, eps, &tris);
    compute_curves(mesh, &ctx, Some(&tris)).iter().map(|c| curve_summary(c, plane.normal)).collect()
}

/// The intersection curve nearest to `seed` (what [`local_plane_cut`] would cut).
pub fn nearest_intersection_curve(mesh: &Mesh, plane: Plane, seed: DVec3) -> Option<IntersectionCurve> {
    plane_intersection_curves(mesh, plane)
        .into_iter()
        .map(|c| (c.distance(seed), c))
        .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
        .map(|x| x.1)
}

// ----------------------------------------------------------------------------------------
// Shared cut steps
// ----------------------------------------------------------------------------------------

/// Boundary loops through the given (now boundary) edges, with a flag telling whether the
/// loop consists only of those edges.
fn collect_loops(mesh: &Mesh, cut_edges: &[u32]) -> Vec<(Vec<u32>, bool)> {
    let cut: HashSet<u32> = cut_edges.iter().copied().collect();
    let mut seen: HashSet<u32> = HashSet::new();
    let mut out = Vec::new();
    for &e in cut_edges {
        if seen.contains(&e) || !mesh.is_edge(e) || !mesh.is_boundary_edge(e) {
            continue;
        }
        match walk_boundary_loop(mesh, e) {
            Some(lp) => {
                let all_cut = lp.edges.iter().all(|x| cut.contains(x));
                seen.extend(lp.edges.iter().copied());
                out.push((lp.vertices, all_cut));
            }
            None => {
                seen.insert(e);
            }
        }
    }
    out
}

fn finish_side(
    mesh: &mut Mesh,
    loops: Vec<(Vec<u32>, bool)>,
    side: Side,
    cap_normal: DVec3,
    fill: CutFill,
    res: &mut PlaneCutResult,
) {
    let fill_on = !matches!(fill, CutFill::None);
    let to_cap: Vec<usize> = (0..loops.len()).filter(|&i| loops[i].1 && fill_on).collect();
    let cap_loops: Vec<Vec<u32>> = to_cap.iter().map(|&i| loops[i].0.clone()).collect();
    let cr = cap_oriented_loops(mesh, &cap_loops, Some(cap_normal), fill);
    for (i, (lp, _)) in loops.into_iter().enumerate() {
        let capped = match to_cap.iter().position(|&k| k == i) {
            Some(k) => !cr.failed_loops.contains(&k),
            None => false,
        };
        res.cut_loops.push(lp);
        res.loop_sides.push(side);
        res.loop_capped.push(capped);
    }
    res.cap_triangles.extend(cr.triangles);
    res.cap_groups.extend(cr.groups);
}

/// Re-create `tris` on private copies of the vertices in `verts` (separating them from the
/// rest of the mesh along those vertices). Returns the old→new vertex map.
pub(crate) fn detach(mesh: &mut Mesh, tris: &[u32], verts: &HashSet<u32>) -> EditResult<HashMap<u32, u32>> {
    let recs: Vec<([u32; 3], i32)> =
        tris.iter().filter(|&&t| mesh.is_triangle(t)).map(|&t| (mesh.triangle(t), mesh.tri_group(t))).collect();
    for &t in tris {
        if mesh.is_triangle(t) {
            mesh.remove_triangle(t, false)?;
        }
    }
    let mut vmap: HashMap<u32, u32> = HashMap::new();
    for (tv, g) in recs {
        let mut m = tv;
        for x in m.iter_mut() {
            if verts.contains(x) {
                *x = match vmap.get(x) {
                    Some(&y) => y,
                    None => {
                        let y = mesh.append_vertex(mesh.vertex(*x));
                        vmap.insert(*x, y);
                        y
                    }
                };
            }
        }
        mesh.append_triangle(m, g)?;
    }
    Ok(vmap)
}

fn check_fill(fill: CutFill) -> EditResult<()> {
    if let CutFill::FlatRemeshed { target_edge_length } = fill {
        if !(target_edge_length.is_finite() && target_edge_length > 0.0) {
            return Err(EditError::InvalidParameter("target_edge_length must be positive".into()));
        }
    }
    Ok(())
}

fn side_index(s: Side) -> u8 {
    match s {
        Side::Above => 0,
        Side::Below => 1,
    }
}

// ----------------------------------------------------------------------------------------
// Global plane cut
// ----------------------------------------------------------------------------------------

/// Cut the whole mesh with a plane (Meshmixer *Plane Cut*). See the module docs.
pub fn plane_cut(mesh: &mut Mesh, plane: Plane, params: &PlaneCutParams) -> EditResult<PlaneCutResult> {
    let plane = normalize_plane(plane)?;
    check_fill(params.fill)?;
    let mut res = PlaneCutResult::default();
    if mesh.triangle_count() == 0 {
        return Ok(res);
    }
    let n = plane.normal;
    let mut ctx = CutContext::new(mesh, plane, snap_eps(mesh));

    // Vertices within round-off of the plane: put them exactly on it.
    for v in 0..mesh.max_vid() as u32 {
        if mesh.is_vertex(v) && ctx.sd[v as usize] == 0.0 && plane.signed_distance(mesh.vertex(v)) != 0.0 {
            mesh.set_vertex(v, plane.project(mesh.vertex(v)));
        }
    }
    let crossing: Vec<u32> = mesh.edge_ids().filter(|&e| ctx.crosses(mesh, e)).collect();
    let f = params.snap_tolerance.clamp(0.0, 0.49);
    if f > 0.0 {
        let mut snap = Vec::new();
        for &e in &crossing {
            let s = ctx.param(mesh, e);
            let [a, b] = mesh.edge_vertices(e);
            if s < f {
                snap.push(a);
            } else if s > 1.0 - f {
                snap.push(b);
            }
        }
        for v in snap {
            if ctx.sd[v as usize] != 0.0 {
                ctx.snap(mesh, v);
                res.snapped_vertices.push(v);
            }
        }
    }
    for e in crossing {
        if ctx.crosses(mesh, e) {
            res.split_vertices.push(ctx.split(mesh, e)?);
        }
    }

    let mut side = vec![0u8; mesh.max_tid()];
    for t in mesh.triangle_ids() {
        side[t as usize] = side_index(ctx.tri_side(mesh, t));
    }

    match params.keep {
        KeepSide::Above | KeepSide::Below => {
            let keep = if params.keep == KeepSide::Above { Side::Above } else { Side::Below };
            let k = side_index(keep);
            let discard: Vec<u32> = mesh.triangle_ids().filter(|&t| side[t as usize] != k).collect();
            let mut cut_edges = Vec::new();
            for &t in &discard {
                for e in mesh.tri_edges(t) {
                    let o = mesh.edge_other_triangle(e, t);
                    if o != INVALID && side[o as usize] == k {
                        cut_edges.push(e);
                    }
                }
            }
            res.removed_triangles = discard.len();
            mesh.remove_triangles(discard);
            let loops = collect_loops(mesh, &cut_edges);
            let cap_n = if keep == Side::Above { -n } else { n };
            finish_side(mesh, loops, keep, cap_n, params.fill, &mut res);
        }
        KeepSide::Both => {
            let below = side_index(Side::Below);
            let mut pairs: Vec<[u32; 2]> = Vec::new();
            let mut seam: HashSet<u32> = HashSet::new();
            for t in mesh.triangle_ids() {
                if side[t as usize] != below {
                    continue;
                }
                for e in mesh.tri_edges(t) {
                    let o = mesh.edge_other_triangle(e, t);
                    if o != INVALID && side[o as usize] != below {
                        let ev = mesh.edge_vertices(e);
                        pairs.push(ev);
                        seam.extend(ev);
                    }
                }
            }
            let mut moved: HashSet<u32> = HashSet::new();
            for &v in &seam {
                for t in mesh.vertex_triangles(v) {
                    if side[t as usize] == below {
                        moved.insert(t);
                    }
                }
            }
            let mut moved: Vec<u32> = moved.into_iter().collect();
            moved.sort_unstable();
            let vmap = detach(mesh, &moved, &seam)?;
            let above_edges: Vec<u32> = pairs.iter().filter_map(|&[a, b]| mesh.find_edge(a, b)).collect();
            let below_edges: Vec<u32> = pairs
                .iter()
                .filter_map(|&[a, b]| mesh.find_edge(*vmap.get(&a)?, *vmap.get(&b)?))
                .collect();
            let la = collect_loops(mesh, &above_edges);
            finish_side(mesh, la, Side::Above, -n, params.fill, &mut res);
            let lb = collect_loops(mesh, &below_edges);
            finish_side(mesh, lb, Side::Below, n, params.fill, &mut res);
        }
    }
    Ok(res)
}

// ----------------------------------------------------------------------------------------
// Local plane cut
// ----------------------------------------------------------------------------------------

/// Cut only the intersection curve nearest to `seed` (e.g. the vessel branch the user
/// clicked) and remove only the piece that curve separates on the discarded side; other
/// parts of the mesh crossing the same plane are untouched. See [`local_plane_cut_with`].
pub fn local_plane_cut(
    mesh: &mut Mesh,
    plane: Plane,
    seed: DVec3,
    keep: KeepSide,
    fill: CutFill,
) -> EditResult<LocalCutResult> {
    local_plane_cut_with(mesh, plane, seed, &PlaneCutParams { keep, fill, ..Default::default() })
}

/// [`local_plane_cut`] with full parameters.
///
/// With `KeepSide::Above` the piece behind the curve (on the plane's negative side, as seen
/// from the curve) is deleted; `Below` deletes the piece in front; `Both` detaches the
/// pieces along the curve. If the chosen curve is open (it runs into a mesh boundary) it
/// can still separate the surface, but it is not capped. Errors with
/// [`EditError::CurveDoesNotSeparate`] if the surface stays connected around the curve
/// (e.g. a cut around a handle); in that case the curve vertices have already been
/// inserted, but the geometry is unchanged apart from sub-tolerance snapping.
pub fn local_plane_cut_with(
    mesh: &mut Mesh,
    plane: Plane,
    seed: DVec3,
    params: &PlaneCutParams,
) -> EditResult<LocalCutResult> {
    let plane = normalize_plane(plane)?;
    check_fill(params.fill)?;
    if mesh.triangle_count() == 0 {
        return Err(EditError::NoIntersection);
    }
    let n = plane.normal;
    let mut ctx = CutContext::new(mesh, plane, snap_eps(mesh));
    let curves = compute_curves(mesh, &ctx, None);
    let (ci, dist) = curves
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let d = c.segments.iter().map(|s| point_segment_distance(seed, s[0], s[1])).fold(f64::INFINITY, f64::min);
            (i, d)
        })
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .ok_or(EditError::NoIntersection)?;
    let mut out = LocalCutResult { curve_closed: curves[ci].closed, curve_distance: dist, ..Default::default() };
    let res = &mut out.cut;

    let mut edge_curve: HashMap<u32, usize> = HashMap::new();
    for (i, c) in curves.iter().enumerate() {
        for &e in &c.crossing_edges {
            edge_curve.insert(e, i);
        }
    }
    let curve = &curves[ci];
    let mut cv: HashSet<u32> = HashSet::new();
    for &v in &curve.plane_vertices {
        let p = mesh.vertex(v);
        if plane.signed_distance(p) != 0.0 {
            mesh.set_vertex(v, plane.project(p));
        }
        cv.insert(v);
    }
    let f = params.snap_tolerance.clamp(0.0, 0.49);
    if f > 0.0 {
        let mut snap = Vec::new();
        for &e in &curve.crossing_edges {
            let s = ctx.param(mesh, e);
            let [a, b] = mesh.edge_vertices(e);
            let cand = if s < f {
                a
            } else if s > 1.0 - f {
                b
            } else {
                continue;
            };
            // Only snap if every crossing edge at the vertex belongs to this curve.
            let exclusive = mesh
                .vertex_edges(cand)
                .iter()
                .all(|&x| !ctx.crosses(mesh, x) || edge_curve.get(&x) == Some(&ci));
            if exclusive {
                snap.push(cand);
            }
        }
        for v in snap {
            if ctx.sd[v as usize] != 0.0 {
                ctx.snap(mesh, v);
                res.snapped_vertices.push(v);
                cv.insert(v);
            }
        }
    }
    for &e in &curve.crossing_edges {
        if mesh.is_edge(e) && ctx.crosses(mesh, e) {
            let v = ctx.split(mesh, e)?;
            res.split_vertices.push(v);
            cv.insert(v);
        }
    }

    // Curve edges: on-plane edges between curve vertices separating opposite sides.
    let mut curve_edges: HashSet<u32> = HashSet::new();
    for &v in &cv {
        for &e in mesh.vertex_edges(v) {
            let w = mesh.edge_other_vertex(e, v);
            if w > v && cv.contains(&w) {
                let [t0, t1] = mesh.edge_triangles(e);
                if t1 != INVALID && ctx.tri_side(mesh, t0) != ctx.tri_side(mesh, t1) {
                    curve_edges.insert(e);
                }
            }
        }
    }
    if curve_edges.is_empty() {
        return Err(EditError::NoIntersection);
    }
    let mut curve_list: Vec<u32> = curve_edges.iter().copied().collect();
    curve_list.sort_unstable();

    // For `Both`, the piece behind the curve is the one detached onto new seam vertices.
    let discard_side = if params.keep == KeepSide::Below { Side::Above } else { Side::Below };
    let keep_side = discard_side.opposite();
    let seeds: Vec<u32> = curve_list
        .iter()
        .flat_map(|&e| mesh.edge_triangles(e))
        .filter(|&t| t != INVALID && ctx.tri_side(mesh, t) == discard_side)
        .collect();
    let mut in_fill = vec![false; mesh.max_tid()];
    let mut fill_tris = Vec::new();
    let mut stack = Vec::new();
    for &s in &seeds {
        if !in_fill[s as usize] {
            in_fill[s as usize] = true;
            stack.push(s);
        }
    }
    while let Some(t) = stack.pop() {
        fill_tris.push(t);
        for e in mesh.tri_edges(t) {
            if curve_edges.contains(&e) {
                continue;
            }
            let o = mesh.edge_other_triangle(e, t);
            if o != INVALID && !in_fill[o as usize] {
                in_fill[o as usize] = true;
                stack.push(o);
            }
        }
    }
    for &e in &curve_list {
        let [t0, t1] = mesh.edge_triangles(e);
        if in_fill[t0 as usize] && in_fill[t1 as usize] {
            return Err(EditError::CurveDoesNotSeparate);
        }
    }

    let cap_n = |s: Side| if s == Side::Above { -n } else { n };
    match params.keep {
        KeepSide::Above | KeepSide::Below => {
            res.removed_triangles = fill_tris.len();
            mesh.remove_triangles(fill_tris);
            let loops = collect_loops(mesh, &curve_list);
            finish_side(mesh, loops, keep_side, cap_n(keep_side), params.fill, res);
        }
        KeepSide::Both => {
            let seam: HashSet<u32> = curve_list.iter().flat_map(|&e| mesh.edge_vertices(e)).collect();
            let mut moved: Vec<u32> = Vec::new();
            let mut mark: HashSet<u32> = HashSet::new();
            for &v in &seam {
                for t in mesh.vertex_triangles(v) {
                    if in_fill[t as usize] && mark.insert(t) {
                        moved.push(t);
                    }
                }
            }
            moved.sort_unstable();
            let pairs: Vec<[u32; 2]> = curve_list.iter().map(|&e| mesh.edge_vertices(e)).collect();
            let vmap = detach(mesh, &moved, &seam)?;
            let kept_edges: Vec<u32> = pairs.iter().filter_map(|&[a, b]| mesh.find_edge(a, b)).collect();
            let moved_edges: Vec<u32> = pairs
                .iter()
                .filter_map(|&[a, b]| mesh.find_edge(*vmap.get(&a)?, *vmap.get(&b)?))
                .collect();
            let la = collect_loops(mesh, &kept_edges);
            finish_side(mesh, la, keep_side, cap_n(keep_side), params.fill, res);
            let lb = collect_loops(mesh, &moved_edges);
            finish_side(mesh, lb, discard_side, cap_n(discard_side), params.fill, res);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_core::primitives;

    fn same_curves(a: &[Curve], b: &[Curve]) -> bool {
        let bits = |s: &[[DVec3; 2]]| s.iter().map(|x| x.map(|p| p.to_array().map(f64::to_bits))).collect::<Vec<_>>();
        a.len() == b.len()
            && a.iter().zip(b).all(|(x, y)| x.crossing_edges == y.crossing_edges && x.plane_vertices == y.plane_vertices && x.closed == y.closed && bits(&x.segments) == bits(&y.segments))
    }

    /// Two wavy tubes side by side, with planes through vertices (on-plane corners) and between.
    fn tubes() -> Mesh {
        let mut m = Mesh::new();
        for k in 0..2 {
            let path: Vec<DVec3> = (0..=60).map(|i| DVec3::new(4.0 * k as f64 + 0.5 * (0.2 * i as f64).sin(), 0.0, 0.25 * i as f64)).collect();
            m.append_mesh(&primitives::tube_along(&path, &vec![1.0 + 0.2 * k as f64; path.len()], 24), 0);
        }
        m
    }

    #[test]
    fn contexts_for_a_few_triangles_give_the_whole_contexts_curves() {
        let m = tubes();
        let eps = snap_eps(&m);
        let all: Vec<u32> = m.triangle_ids().collect();
        let mut buffer = Vec::new();
        for plane in [Plane::from_point_normal(m.vertex(100), DVec3::Z), Plane::from_point_normal(DVec3::new(0.3, 0.0, 7.1), DVec3::new(0.2, 0.1, 1.0)), Plane::from_point_normal(m.vertex(500), DVec3::X)] {
            let full = CutContext::new(&m, plane, eps);
            assert!(same_curves(&compute_curves(&m, &full, None), &compute_curves(&m, &full, Some(&all))));
            // A few triangles: a context for them alone, in a map or in a reused buffer (kept
            // all zeros, or of any contents), gives what the whole one gives, and the buffer
            // is all zeros again after.
            let near: Vec<u32> = all.iter().copied().filter(|&t| m.tri_centroid(t).distance(DVec3::new(0.0, 0.0, 7.0)) < 3.0).collect();
            let want = compute_curves(&m, &full, Some(&near));
            let local = CutContext::for_triangles_local(&m, plane, eps, &near);
            assert!(same_curves(&compute_curves(&m, &local, Some(&near)), &want));
            let reused = CutContext::for_triangles_in(&m, plane, eps, &near, std::mem::take(&mut buffer));
            assert!(same_curves(&compute_curves(&m, &reused, Some(&near)), &want));
            buffer = reused.into_buffer();
            assert!(buffer.len() == m.max_vid() && buffer.iter().all(|&x| x.to_bits() == 0));
            let cleared = CutContext::for_triangles(&m, plane, eps, &near, vec![1.0; 7]);
            assert!(same_curves(&compute_curves(&m, &cleared, Some(&near)), &want));
            assert!(cleared.into_buffer().iter().all(|&x| x.to_bits() == 0));
        }
    }

    #[test]
    fn a_full_tree_gives_the_snap_distance() {
        let mut m = tubes();
        let bvh = Bvh::build(&m);
        // Not cached yet: from the tree's box; then from the mesh's own.
        m.bump_shape_version();
        let bvh2 = Bvh::build(&m);
        assert!(m.cached_bounds().is_none());
        assert_eq!(snap_eps_with(&m, Some(&bvh2)).to_bits(), (1e-9 * m.bounds().diagonal()).to_bits());
        assert_eq!(snap_eps_with(&m, Some(&bvh2)).to_bits(), snap_distance(&m).to_bits());
        // A stale tree, or one over some triangles, is not used.
        assert_eq!(snap_eps_with(&m, Some(&bvh)).to_bits(), snap_distance(&m).to_bits());
        let some: Vec<u32> = m.triangle_ids().take(10).collect();
        assert_eq!(snap_eps_with(&m, Some(&Bvh::build_subset(&m, &some))).to_bits(), snap_distance(&m).to_bits());
    }
}
