//! Dynamic indexed triangle mesh with explicit edge topology.
//!
//! The design follows the "vertex/edge/triangle with free lists" approach used by
//! geometry3Sharp's `DMesh3` (Boost License, R. Schmidt): every element lives in a flat
//! array, deleted slots are recycled through a free list, and each edge stores its two
//! vertices and (up to) two adjacent triangles. Edges can therefore never be non-manifold;
//! the builder in [`Mesh::from_triangles`] splits vertices apart when the input contains
//! non-manifold edges so that the rest of the system only ever sees manifold edges
//! (non-manifold *vertices*, i.e. "bowties", are representable and detectable).
//!
//! Supported local edits: [`Mesh::split_edge`], [`Mesh::flip_edge`],
//! [`Mesh::collapse_edge`], [`Mesh::poke_triangle`], [`Mesh::remove_triangle`].

use glam::DVec3;
use smallvec::SmallVec;

/// Sentinel index for "no element".
pub const INVALID: u32 = u32::MAX;

static VERSION_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// Globally unique version stamp. Versions are unique across *all* meshes, so caches keyed
/// on a version (GPU buffers, BVHs) can never confuse two different meshes, e.g. a fresh
/// copy that received the same number of edits as another one.
#[inline]
fn next_version() -> u64 {
    VERSION_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum MeshError {
    #[error("index is not a valid vertex")]
    NotAVertex,
    #[error("index is not a valid triangle")]
    NotATriangle,
    #[error("index is not a valid edge")]
    NotAnEdge,
    #[error("operation would create a non-manifold edge")]
    NonManifold,
    #[error("triangle already exists")]
    DuplicateTriangle,
    #[error("triangle has repeated vertices")]
    Degenerate,
    #[error("operation not allowed on a boundary edge")]
    IsBoundaryEdge,
    #[error("edge flip would create an edge that already exists")]
    FlippedEdgeExists,
    #[error("collapse would violate the link condition")]
    InvalidNeighbourhood,
    #[error("collapse would degenerate a tetrahedron")]
    CollapseTetrahedron,
    #[error("adjacent triangles have inconsistent orientation")]
    InconsistentOrientation,
    #[error("mesh topology is broken")]
    BrokenTopology,
}

pub type MeshResult<T> = Result<T, MeshError>;

/// Result of [`Mesh::split_edge`].
#[derive(Clone, Copy, Debug)]
pub struct EdgeSplitInfo {
    /// The edge that was split; it now connects `v_a` and `v_new`.
    pub e_split: u32,
    pub v_a: u32,
    pub v_b: u32,
    pub v_new: u32,
    /// New edge connecting `v_new` and `v_b`.
    pub e_new_b: u32,
    /// New edges from `v_new` to the opposite vertex of each original triangle
    /// (second is INVALID for boundary edges).
    pub e_new_opp: [u32; 2],
    /// The original triangles (kept) and the newly created triangles.
    pub t_orig: [u32; 2],
    pub t_new: [u32; 2],
    pub v_opp: [u32; 2],
}

/// Result of [`Mesh::flip_edge`].
#[derive(Clone, Copy, Debug)]
pub struct EdgeFlipInfo {
    pub e: u32,
    pub v_old: [u32; 2],
    pub v_new: [u32; 2],
    pub t: [u32; 2],
}

/// Result of [`Mesh::collapse_edge`].
#[derive(Clone, Copy, Debug)]
pub struct EdgeCollapseInfo {
    pub v_kept: u32,
    pub v_removed: u32,
    pub is_boundary: bool,
    pub e_collapsed: u32,
    pub t_removed: [u32; 2],
    pub e_removed: [u32; 2],
    pub e_kept: [u32; 2],
}

/// Result of [`Mesh::poke_triangle`].
#[derive(Clone, Copy, Debug)]
pub struct PokeInfo {
    pub v_new: u32,
    pub t_orig: u32,
    pub t_new: [u32; 2],
}

type EdgeList = SmallVec<[u32; 8]>;

#[derive(Clone, Default)]
pub struct Mesh {
    vpos: Vec<DVec3>,
    vedges: Vec<EdgeList>,
    vlive: Vec<bool>,
    vfree: Vec<u32>,
    nv: usize,

    tv: Vec<[u32; 3]>,
    te: Vec<[u32; 3]>,
    tgroup: Vec<i32>,
    tlive: Vec<bool>,
    tfree: Vec<u32>,
    nt: usize,

    ev: Vec<[u32; 2]>,
    et: Vec<[u32; 2]>,
    elive: Vec<bool>,
    efree: Vec<u32>,
    ne: usize,

    /// Incremented on any topology change.
    topo_version: u64,
    /// Incremented on any vertex position change (and on topology changes).
    shape_version: u64,
    /// Incremented when a triangle's face group changes (only meaningful together with the
    /// shape version: a different mesh always has a different shape version).
    group_version: u64,
    /// Optional per-vertex colours (RGB in 0..1, parallel to the positions).
    vcolor: Option<Vec<[f32; 3]>>,
    /// Incremented when colours are set, enabled or cleared (like `group_version`).
    color_version: u64,
}

/// Colour given to new vertices of a coloured mesh when nothing better is known.
pub const DEFAULT_VERTEX_COLOR: [f32; 3] = [0.8, 0.8, 0.8];

fn lerp3(a: [f32; 3], b: [f32; 3], s: f32) -> [f32; 3] {
    [0, 1, 2].map(|i| a[i] + (b[i] - a[i]) * s)
}

impl std::fmt::Debug for Mesh {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mesh")
            .field("vertices", &self.nv)
            .field("triangles", &self.nt)
            .field("edges", &self.ne)
            .finish()
    }
}

#[inline]
fn sorted2(a: u32, b: u32) -> [u32; 2] {
    if a < b {
        [a, b]
    } else {
        [b, a]
    }
}

/// Given a triangle's vertices and two of them, return the third.
#[inline]
pub fn tri_other_vertex(tv: [u32; 3], a: u32, b: u32) -> u32 {
    for &v in &tv {
        if v != a && v != b {
            return v;
        }
    }
    INVALID
}

/// Index `j` such that `(tv[j], tv[j+1])` is the edge `{a, b}` in either orientation.
#[inline]
pub fn tri_edge_index(tv: [u32; 3], a: u32, b: u32) -> Option<usize> {
    (0..3).find(|&j| {
        let p = tv[j];
        let q = tv[(j + 1) % 3];
        (p == a && q == b) || (p == b && q == a)
    })
}

impl Mesh {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(nv: usize, nt: usize) -> Self {
        let mut m = Self::default();
        m.vpos.reserve(nv);
        m.vedges.reserve(nv);
        m.vlive.reserve(nv);
        m.tv.reserve(nt);
        m.te.reserve(nt);
        m.tgroup.reserve(nt);
        m.tlive.reserve(nt);
        let ne = nt * 3 / 2 + 16;
        m.ev.reserve(ne);
        m.et.reserve(ne);
        m.elive.reserve(ne);
        m
    }

    // ------------------------------------------------------------------
    // Counts, validity, accessors
    // ------------------------------------------------------------------

    pub fn vertex_count(&self) -> usize {
        self.nv
    }
    pub fn triangle_count(&self) -> usize {
        self.nt
    }
    pub fn edge_count(&self) -> usize {
        self.ne
    }
    /// Upper bound (exclusive) on vertex indices.
    pub fn max_vid(&self) -> usize {
        self.vpos.len()
    }
    /// Upper bound (exclusive) on triangle indices.
    pub fn max_tid(&self) -> usize {
        self.tv.len()
    }
    /// Upper bound (exclusive) on edge indices.
    pub fn max_eid(&self) -> usize {
        self.ev.len()
    }
    pub fn topology_version(&self) -> u64 {
        self.topo_version
    }
    pub fn shape_version(&self) -> u64 {
        self.shape_version
    }
    /// Mark the mesh as changed (e.g. after bulk edits through `vertices_mut`).
    pub fn bump_shape_version(&mut self) {
        self.shape_version = next_version();
    }

    #[inline]
    pub fn is_vertex(&self, v: u32) -> bool {
        (v as usize) < self.vlive.len() && self.vlive[v as usize]
    }
    #[inline]
    pub fn is_triangle(&self, t: u32) -> bool {
        (t as usize) < self.tlive.len() && self.tlive[t as usize]
    }
    #[inline]
    pub fn is_edge(&self, e: u32) -> bool {
        (e as usize) < self.elive.len() && self.elive[e as usize]
    }

    #[inline]
    pub fn vertex(&self, v: u32) -> DVec3 {
        self.vpos[v as usize]
    }
    #[inline]
    pub fn set_vertex(&mut self, v: u32, p: DVec3) {
        self.vpos[v as usize] = p;
        self.shape_version = next_version();
    }
    /// Raw position array (indexed by vertex id; dead slots hold stale data).
    pub fn positions(&self) -> &[DVec3] {
        &self.vpos
    }
    /// Mutable raw position array. Call [`Mesh::bump_shape_version`] afterwards.
    pub fn positions_mut(&mut self) -> &mut [DVec3] {
        &mut self.vpos
    }

    #[inline]
    pub fn triangle(&self, t: u32) -> [u32; 3] {
        self.tv[t as usize]
    }
    #[inline]
    pub fn tri_edges(&self, t: u32) -> [u32; 3] {
        self.te[t as usize]
    }
    #[inline]
    pub fn tri_group(&self, t: u32) -> i32 {
        self.tgroup[t as usize]
    }
    #[inline]
    pub fn set_tri_group(&mut self, t: u32, g: i32) {
        self.tgroup[t as usize] = g;
        self.group_version = self.group_version.wrapping_add(1);
    }
    pub fn group_version(&self) -> u64 {
        self.group_version
    }
    pub fn max_group_id(&self) -> i32 {
        self.triangle_ids().map(|t| self.tgroup[t as usize]).max().unwrap_or(0)
    }
    #[inline]
    pub fn edge_vertices(&self, e: u32) -> [u32; 2] {
        self.ev[e as usize]
    }
    #[inline]
    pub fn edge_triangles(&self, e: u32) -> [u32; 2] {
        self.et[e as usize]
    }
    #[inline]
    pub fn vertex_edges(&self, v: u32) -> &[u32] {
        &self.vedges[v as usize]
    }
    #[inline]
    pub fn vertex_valence(&self, v: u32) -> usize {
        self.vedges[v as usize].len()
    }
    #[inline]
    pub fn edge_other_vertex(&self, e: u32, v: u32) -> u32 {
        let [a, b] = self.ev[e as usize];
        if a == v {
            b
        } else if b == v {
            a
        } else {
            INVALID
        }
    }
    #[inline]
    pub fn edge_other_triangle(&self, e: u32, t: u32) -> u32 {
        let [a, b] = self.et[e as usize];
        if a == t {
            b
        } else if b == t {
            a
        } else {
            INVALID
        }
    }
    #[inline]
    pub fn is_boundary_edge(&self, e: u32) -> bool {
        self.et[e as usize][1] == INVALID
    }
    pub fn is_boundary_vertex(&self, v: u32) -> bool {
        self.vedges[v as usize].iter().any(|&e| self.is_boundary_edge(e))
    }
    pub fn is_boundary_triangle(&self, t: u32) -> bool {
        self.te[t as usize].iter().any(|&e| self.is_boundary_edge(e))
    }
    pub fn has_boundary(&self) -> bool {
        self.edge_ids().any(|e| self.is_boundary_edge(e))
    }
    pub fn is_closed(&self) -> bool {
        !self.has_boundary()
    }

    pub fn vertex_ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.vlive.iter().enumerate().filter(|(_, &l)| l).map(|(i, _)| i as u32)
    }
    pub fn triangle_ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.tlive.iter().enumerate().filter(|(_, &l)| l).map(|(i, _)| i as u32)
    }
    pub fn edge_ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.elive.iter().enumerate().filter(|(_, &l)| l).map(|(i, _)| i as u32)
    }
    pub fn boundary_edge_ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.edge_ids().filter(|&e| self.is_boundary_edge(e))
    }

    /// Find the edge between vertices `a` and `b`.
    pub fn find_edge(&self, a: u32, b: u32) -> Option<u32> {
        let (s, o) = if self.vedges[a as usize].len() <= self.vedges[b as usize].len() {
            (a, b)
        } else {
            (b, a)
        };
        let key = sorted2(a, b);
        let _ = o;
        self.vedges[s as usize].iter().copied().find(|&e| self.ev[e as usize] == key)
    }

    /// Find the edge `{a, b}` among the edges of triangle `t`.
    pub fn find_edge_from_tri(&self, a: u32, b: u32, t: u32) -> Option<u32> {
        let key = sorted2(a, b);
        self.te[t as usize].iter().copied().find(|&e| self.ev[e as usize] == key)
    }

    /// Vertices opposite edge `e` in its (up to two) triangles.
    pub fn edge_opposite_vertices(&self, e: u32) -> [u32; 2] {
        let [a, b] = self.ev[e as usize];
        let [t0, t1] = self.et[e as usize];
        let c = tri_other_vertex(self.tv[t0 as usize], a, b);
        let d = if t1 != INVALID { tri_other_vertex(self.tv[t1 as usize], a, b) } else { INVALID };
        [c, d]
    }

    /// Edge `e`'s vertices oriented as they appear in its first triangle.
    pub fn oriented_edge_vertices(&self, e: u32) -> [u32; 2] {
        let [a, b] = self.ev[e as usize];
        let t = self.et[e as usize][0];
        let tv = self.tv[t as usize];
        let j = tri_edge_index(tv, a, b).expect("edge not in its triangle");
        [tv[j], tv[(j + 1) % 3]]
    }

    /// One-ring neighbour vertices of `v` (unordered).
    pub fn vertex_neighbors(&self, v: u32) -> impl Iterator<Item = u32> + '_ {
        self.vedges[v as usize].iter().map(move |&e| self.edge_other_vertex(e, v))
    }

    /// Triangles incident to `v` (unordered, unique).
    pub fn vertex_triangles(&self, v: u32) -> SmallVec<[u32; 12]> {
        let mut out: SmallVec<[u32; 12]> = SmallVec::new();
        for &e in self.vedges[v as usize].iter() {
            for &t in self.et[e as usize].iter() {
                if t != INVALID && !out.contains(&t) {
                    out.push(t);
                }
            }
        }
        out
    }

    /// Triangles sharing an edge with `t` (INVALID for boundary edges).
    pub fn tri_neighbors(&self, t: u32) -> [u32; 3] {
        let te = self.te[t as usize];
        [
            self.edge_other_triangle(te[0], t),
            self.edge_other_triangle(te[1], t),
            self.edge_other_triangle(te[2], t),
        ]
    }

    /// True if the triangles around `v` do not form a single fan (a "bowtie" vertex).
    pub fn is_bowtie_vertex(&self, v: u32) -> bool {
        let tris = self.vertex_triangles(v);
        if tris.is_empty() {
            return false;
        }
        // Flood fill across edges incident to v.
        let mut seen: SmallVec<[u32; 12]> = SmallVec::new();
        let mut stack: SmallVec<[u32; 12]> = SmallVec::new();
        stack.push(tris[0]);
        seen.push(tris[0]);
        while let Some(t) = stack.pop() {
            for &e in self.te[t as usize].iter() {
                let [a, b] = self.ev[e as usize];
                if a != v && b != v {
                    continue;
                }
                let o = self.edge_other_triangle(e, t);
                if o != INVALID && !seen.contains(&o) {
                    seen.push(o);
                    stack.push(o);
                }
            }
        }
        seen.len() != tris.len()
    }

    // ------------------------------------------------------------------
    // Geometry helpers
    // ------------------------------------------------------------------

    #[inline]
    pub fn tri_positions(&self, t: u32) -> [DVec3; 3] {
        let [a, b, c] = self.tv[t as usize];
        [self.vpos[a as usize], self.vpos[b as usize], self.vpos[c as usize]]
    }
    /// Unnormalized normal (length = 2 * area).
    #[inline]
    pub fn tri_cross(&self, t: u32) -> DVec3 {
        let [a, b, c] = self.tri_positions(t);
        (b - a).cross(c - a)
    }
    #[inline]
    pub fn tri_normal(&self, t: u32) -> DVec3 {
        self.tri_cross(t).normalize_or_zero()
    }
    #[inline]
    pub fn tri_area(&self, t: u32) -> f64 {
        0.5 * self.tri_cross(t).length()
    }
    #[inline]
    pub fn tri_centroid(&self, t: u32) -> DVec3 {
        let [a, b, c] = self.tri_positions(t);
        (a + b + c) / 3.0
    }
    #[inline]
    pub fn edge_length(&self, e: u32) -> f64 {
        let [a, b] = self.ev[e as usize];
        self.vpos[a as usize].distance(self.vpos[b as usize])
    }
    #[inline]
    pub fn edge_midpoint(&self, e: u32) -> DVec3 {
        let [a, b] = self.ev[e as usize];
        (self.vpos[a as usize] + self.vpos[b as usize]) * 0.5
    }

    /// Area-weighted vertex normal.
    pub fn vertex_normal(&self, v: u32) -> DVec3 {
        let mut n = DVec3::ZERO;
        for t in self.vertex_triangles(v) {
            n += self.tri_cross(t);
        }
        n.normalize_or_zero()
    }

    /// Area-weighted normals for all vertices, indexed by vertex id.
    pub fn compute_vertex_normals(&self) -> Vec<DVec3> {
        let mut n = vec![DVec3::ZERO; self.max_vid()];
        for t in self.triangle_ids() {
            let c = self.tri_cross(t);
            for &v in &self.tv[t as usize] {
                n[v as usize] += c;
            }
        }
        for x in n.iter_mut() {
            *x = x.normalize_or_zero();
        }
        n
    }

    /// Axis-aligned bounds of live vertices that are referenced by triangles.
    pub fn bounds(&self) -> crate::Aabb {
        let mut b = crate::Aabb::EMPTY;
        for t in self.triangle_ids() {
            for p in self.tri_positions(t) {
                b.include(p);
            }
        }
        b
    }

    pub fn average_edge_length(&self) -> f64 {
        if self.ne == 0 {
            return 0.0;
        }
        self.edge_ids().map(|e| self.edge_length(e)).sum::<f64>() / self.ne as f64
    }

    // ------------------------------------------------------------------
    // Allocation helpers
    // ------------------------------------------------------------------

    fn alloc_vertex(&mut self, p: DVec3) -> u32 {
        self.nv += 1;
        if let Some(v) = self.vfree.pop() {
            let i = v as usize;
            self.vpos[i] = p;
            self.vedges[i].clear();
            self.vlive[i] = true;
            if let Some(c) = &mut self.vcolor {
                c[i] = DEFAULT_VERTEX_COLOR;
            }
            v
        } else {
            self.vpos.push(p);
            self.vedges.push(EdgeList::new());
            self.vlive.push(true);
            if let Some(c) = &mut self.vcolor {
                c.push(DEFAULT_VERTEX_COLOR);
            }
            (self.vpos.len() - 1) as u32
        }
    }

    fn free_vertex(&mut self, v: u32) {
        let i = v as usize;
        debug_assert!(self.vlive[i]);
        self.vlive[i] = false;
        self.vedges[i] = EdgeList::new();
        self.vfree.push(v);
        self.nv -= 1;
    }

    fn alloc_triangle(&mut self, tv: [u32; 3], te: [u32; 3], g: i32) -> u32 {
        self.nt += 1;
        if let Some(t) = self.tfree.pop() {
            let i = t as usize;
            self.tv[i] = tv;
            self.te[i] = te;
            self.tgroup[i] = g;
            self.tlive[i] = true;
            t
        } else {
            self.tv.push(tv);
            self.te.push(te);
            self.tgroup.push(g);
            self.tlive.push(true);
            (self.tv.len() - 1) as u32
        }
    }

    fn free_triangle(&mut self, t: u32) {
        debug_assert!(self.tlive[t as usize]);
        self.tlive[t as usize] = false;
        self.tfree.push(t);
        self.nt -= 1;
    }

    /// Allocate an edge and register it with both vertices.
    fn alloc_edge(&mut self, a: u32, b: u32, t0: u32, t1: u32) -> u32 {
        self.ne += 1;
        let e = if let Some(e) = self.efree.pop() {
            let i = e as usize;
            self.ev[i] = sorted2(a, b);
            self.et[i] = [t0, t1];
            self.elive[i] = true;
            e
        } else {
            self.ev.push(sorted2(a, b));
            self.et.push([t0, t1]);
            self.elive.push(true);
            (self.ev.len() - 1) as u32
        };
        self.vedges[a as usize].push(e);
        self.vedges[b as usize].push(e);
        e
    }

    /// Free an edge (caller is responsible for vertex edge lists).
    fn free_edge(&mut self, e: u32) {
        debug_assert!(self.elive[e as usize]);
        self.elive[e as usize] = false;
        self.efree.push(e);
        self.ne -= 1;
    }

    fn remove_vertex_edge(&mut self, v: u32, e: u32) -> bool {
        let list = &mut self.vedges[v as usize];
        if let Some(i) = list.iter().position(|&x| x == e) {
            list.swap_remove(i);
            true
        } else {
            false
        }
    }

    /// Replace triangle `t_old` with `t_new` in edge `e`, keeping slot 0 valid.
    fn replace_edge_triangle(&mut self, e: u32, t_old: u32, t_new: u32) -> bool {
        let et = &mut self.et[e as usize];
        if et[0] == t_old {
            if t_new == INVALID {
                et[0] = et[1];
                et[1] = INVALID;
            } else {
                et[0] = t_new;
            }
            true
        } else if et[1] == t_old {
            et[1] = t_new;
            true
        } else {
            false
        }
    }

    fn replace_tri_edge(&mut self, t: u32, e_old: u32, e_new: u32) -> bool {
        for x in self.te[t as usize].iter_mut() {
            if *x == e_old {
                *x = e_new;
                return true;
            }
        }
        false
    }

    // ------------------------------------------------------------------
    // Construction
    // ------------------------------------------------------------------

    pub fn append_vertex(&mut self, p: DVec3) -> u32 {
        self.shape_version = next_version();
        self.alloc_vertex(p)
    }

    /// Append a triangle. Fails if it would create a non-manifold edge or duplicate an
    /// existing triangle.
    pub fn append_triangle(&mut self, tv: [u32; 3], group: i32) -> MeshResult<u32> {
        for &v in &tv {
            if !self.is_vertex(v) {
                return Err(MeshError::NotAVertex);
            }
        }
        if tv[0] == tv[1] || tv[1] == tv[2] || tv[2] == tv[0] {
            return Err(MeshError::Degenerate);
        }
        let mut found = [INVALID; 3];
        for j in 0..3 {
            if let Some(e) = self.find_edge(tv[j], tv[(j + 1) % 3]) {
                if self.et[e as usize][1] != INVALID {
                    return Err(MeshError::NonManifold);
                }
                found[j] = e;
            }
        }
        if found.iter().all(|&e| e != INVALID) {
            // All three edges exist; reject if they already bound a common triangle.
            let t0 = self.et[found[0] as usize][0];
            if self.et[found[1] as usize][0] == t0 && self.et[found[2] as usize][0] == t0 {
                return Err(MeshError::DuplicateTriangle);
            }
        }
        let t = self.alloc_triangle(tv, [INVALID; 3], group);
        let mut te = [INVALID; 3];
        for j in 0..3 {
            te[j] = if found[j] != INVALID {
                self.et[found[j] as usize][1] = t;
                found[j]
            } else {
                self.alloc_edge(tv[j], tv[(j + 1) % 3], t, INVALID)
            };
        }
        self.te[t as usize] = te;
        self.topo_version = next_version();
        self.shape_version = next_version();
        Ok(t)
    }

    /// Build a mesh from a triangle soup with shared vertex indices.
    ///
    /// Triangles that would produce non-manifold edges get private copies of their
    /// vertices (so the edge becomes a pair of coincident boundary edges, exactly like
    /// geometry3Sharp's builder). Degenerate and duplicate triangles are skipped.
    /// Returns the mesh and the number of triangles that needed special handling.
    pub fn from_triangles(positions: &[DVec3], tris: &[[u32; 3]], groups: Option<&[i32]>) -> (Mesh, BuildReport) {
        let mut m = Mesh::with_capacity(positions.len(), tris.len());
        for &p in positions {
            m.alloc_vertex(p);
        }
        let mut report = BuildReport::default();
        for (i, &t) in tris.iter().enumerate() {
            let g = groups.map(|g| g[i]).unwrap_or(0);
            if t.iter().any(|&v| v as usize >= positions.len()) {
                report.invalid_index += 1;
                continue;
            }
            match m.append_triangle(t, g) {
                Ok(_) => {}
                Err(MeshError::Degenerate) => report.degenerate += 1,
                Err(MeshError::DuplicateTriangle) => report.duplicate += 1,
                Err(MeshError::NonManifold) => {
                    let nt = [
                        m.alloc_vertex(positions[t[0] as usize]),
                        m.alloc_vertex(positions[t[1] as usize]),
                        m.alloc_vertex(positions[t[2] as usize]),
                    ];
                    m.append_triangle(nt, g).expect("fresh vertices cannot be non-manifold");
                    report.nonmanifold_split += 1;
                }
                Err(e) => panic!("unexpected error building mesh: {e}"),
            }
        }
        // Remove vertices not referenced by any triangle.
        for v in 0..m.max_vid() as u32 {
            if m.vlive[v as usize] && m.vedges[v as usize].is_empty() {
                m.free_vertex(v);
                report.unreferenced_vertices += 1;
            }
        }
        m.topo_version = next_version();
        m.shape_version = next_version();
        (m, report)
    }

    // ------------------------------------------------------------------
    // Removal
    // ------------------------------------------------------------------

    /// Remove a triangle; edges left without triangles are deleted, and optionally
    /// vertices left without edges are deleted too.
    pub fn remove_triangle(&mut self, t: u32, remove_isolated_vertices: bool) -> MeshResult<()> {
        if !self.is_triangle(t) {
            return Err(MeshError::NotATriangle);
        }
        let te = self.te[t as usize];
        let tv = self.tv[t as usize];
        for &e in &te {
            self.replace_edge_triangle(e, t, INVALID);
            if self.et[e as usize][0] == INVALID {
                let [a, b] = self.ev[e as usize];
                self.remove_vertex_edge(a, e);
                self.remove_vertex_edge(b, e);
                self.free_edge(e);
            }
        }
        self.free_triangle(t);
        if remove_isolated_vertices {
            for &v in &tv {
                if self.vlive[v as usize] && self.vedges[v as usize].is_empty() {
                    self.free_vertex(v);
                }
            }
        }
        self.topo_version = next_version();
        self.shape_version = next_version();
        Ok(())
    }

    /// Remove a set of triangles and then any isolated vertices.
    pub fn remove_triangles(&mut self, tris: impl IntoIterator<Item = u32>) {
        let mut touched = Vec::new();
        for t in tris {
            if self.is_triangle(t) {
                touched.extend_from_slice(&self.tv[t as usize]);
                let _ = self.remove_triangle(t, false);
            }
        }
        for v in touched {
            if self.vlive[v as usize] && self.vedges[v as usize].is_empty() {
                self.free_vertex(v);
            }
        }
    }

    /// Remove any live vertex that has no incident edges.
    pub fn remove_isolated_vertices(&mut self) -> usize {
        let mut n = 0;
        for v in 0..self.max_vid() as u32 {
            if self.vlive[v as usize] && self.vedges[v as usize].is_empty() {
                self.free_vertex(v);
                n += 1;
            }
        }
        n
    }

    /// Reverse the orientation of a triangle.
    pub fn reverse_triangle(&mut self, t: u32) {
        let tv = &mut self.tv[t as usize];
        tv.swap(1, 2);
        let te = &mut self.te[t as usize];
        // edges were (01, 12, 20); after swapping v1/v2 vertex order is (0,2,1):
        // edges become (02, 21, 10) = (old e2, old e1, old e0)
        te.swap(0, 2);
        self.topo_version = next_version();
        self.shape_version = next_version();
    }

    /// Reverse the orientation of every triangle.
    pub fn reverse_orientation(&mut self) {
        for t in 0..self.max_tid() as u32 {
            if self.tlive[t as usize] {
                self.reverse_triangle(t);
            }
        }
    }

    // ------------------------------------------------------------------
    // Local topology edits
    // ------------------------------------------------------------------

    /// Split edge `e` at parameter `s` along (a,b) where `[a, b] = edge_vertices(e)`.
    pub fn split_edge_at(&mut self, e: u32, s: f64) -> MeshResult<EdgeSplitInfo> {
        if !self.is_edge(e) {
            return Err(MeshError::NotAnEdge);
        }
        let [a, b] = self.ev[e as usize];
        let et = self.et[e as usize];
        let p = self.vpos[a as usize].lerp(self.vpos[b as usize], s);
        let f = self.alloc_vertex(p);
        if let Some(c) = &mut self.vcolor {
            c[f as usize] = lerp3(c[a as usize], c[b as usize], s as f32);
        }

        // Edge e becomes (a, f); new edge (f, b).
        self.remove_vertex_edge(b, e);
        self.ev[e as usize] = sorted2(a, f);
        self.vedges[f as usize].push(e);
        let e_fb = self.alloc_edge(f, b, INVALID, INVALID);

        let mut info = EdgeSplitInfo {
            e_split: e,
            v_a: a,
            v_b: b,
            v_new: f,
            e_new_b: e_fb,
            e_new_opp: [INVALID; 2],
            t_orig: et,
            t_new: [INVALID; 2],
            v_opp: [INVALID; 2],
        };
        // Reset edge triangle lists; rebuilt below.
        self.et[e as usize] = [INVALID; 2];

        for (k, &t) in et.iter().enumerate() {
            if t == INVALID {
                continue;
            }
            let tv = self.tv[t as usize];
            let j = tri_edge_index(tv, a, b).ok_or(MeshError::BrokenTopology)?;
            let p_ = tv[j];
            let q_ = tv[(j + 1) % 3];
            let r_ = tv[(j + 2) % 3];
            info.v_opp[k] = r_;
            let te = self.te[t as usize];
            let e_qr = te[(j + 1) % 3];
            let e_rp = te[(j + 2) % 3];
            let g = self.tgroup[t as usize];

            // t: (p, f, r)   t2: (f, q, r)
            let t2 = self.alloc_triangle([f, q_, r_], [INVALID; 3], g);
            let e_fr = self.alloc_edge(f, r_, t, t2);
            let (e_pf, e_fq) = if p_ == a { (e, e_fb) } else { (e_fb, e) };
            self.tv[t as usize] = [p_, f, r_];
            self.te[t as usize] = [e_pf, e_fr, e_rp];
            self.te[t2 as usize] = [e_fq, e_qr, e_fr];
            self.replace_edge_triangle(e_qr, t, t2);
            // add t to e_pf and t2 to e_fq
            Self::push_edge_tri(&mut self.et[e_pf as usize], t);
            Self::push_edge_tri(&mut self.et[e_fq as usize], t2);
            info.t_new[k] = t2;
            info.e_new_opp[k] = e_fr;
        }
        self.topo_version = next_version();
        self.shape_version = next_version();
        Ok(info)
    }

    #[inline]
    fn push_edge_tri(et: &mut [u32; 2], t: u32) {
        if et[0] == INVALID {
            et[0] = t;
        } else {
            debug_assert!(et[1] == INVALID);
            et[1] = t;
        }
    }

    /// Split an edge at its midpoint.
    pub fn split_edge(&mut self, e: u32) -> MeshResult<EdgeSplitInfo> {
        self.split_edge_at(e, 0.5)
    }

    /// Rotate interior edge `e` inside the quad formed by its two triangles.
    pub fn flip_edge(&mut self, e: u32) -> MeshResult<EdgeFlipInfo> {
        if !self.is_edge(e) {
            return Err(MeshError::NotAnEdge);
        }
        let [t0, t1] = self.et[e as usize];
        if t1 == INVALID {
            return Err(MeshError::IsBoundaryEdge);
        }
        let [a, b] = self.ev[e as usize];
        let tv0 = self.tv[t0 as usize];
        let tv1 = self.tv[t1 as usize];
        let j0 = tri_edge_index(tv0, a, b).ok_or(MeshError::BrokenTopology)?;
        let j1 = tri_edge_index(tv1, a, b).ok_or(MeshError::BrokenTopology)?;
        let p = tv0[j0];
        let q = tv0[(j0 + 1) % 3];
        let c = tv0[(j0 + 2) % 3];
        if tv1[j1] != q || tv1[(j1 + 1) % 3] != p {
            return Err(MeshError::InconsistentOrientation);
        }
        let d = tv1[(j1 + 2) % 3];
        if c == d {
            return Err(MeshError::DuplicateTriangle);
        }
        if self.find_edge(c, d).is_some() {
            return Err(MeshError::FlippedEdgeExists);
        }
        let te0 = self.te[t0 as usize];
        let te1 = self.te[t1 as usize];
        let e_qc = te0[(j0 + 1) % 3];
        let e_cp = te0[(j0 + 2) % 3];
        let e_pd = te1[(j1 + 1) % 3];
        let e_dq = te1[(j1 + 2) % 3];

        // t0' = (c, p, d), t1' = (d, q, c)
        self.tv[t0 as usize] = [c, p, d];
        self.te[t0 as usize] = [e_cp, e_pd, e];
        self.tv[t1 as usize] = [d, q, c];
        self.te[t1 as usize] = [e_dq, e_qc, e];
        self.replace_edge_triangle(e_pd, t1, t0);
        self.replace_edge_triangle(e_qc, t0, t1);

        self.remove_vertex_edge(a, e);
        self.remove_vertex_edge(b, e);
        self.ev[e as usize] = sorted2(c, d);
        self.vedges[c as usize].push(e);
        self.vedges[d as usize].push(e);
        self.topo_version = next_version();
        self.shape_version = next_version();
        Ok(EdgeFlipInfo { e, v_old: [a, b], v_new: [c, d], t: [t0, t1] })
    }

    /// Check whether collapsing `v_remove` into `v_keep` is topologically valid.
    pub fn can_collapse(&self, v_keep: u32, v_remove: u32) -> MeshResult<()> {
        self.collapse_check(v_keep, v_remove).map(|_| ())
    }

    #[allow(clippy::type_complexity)]
    fn collapse_check(&self, b: u32, a: u32) -> MeshResult<(u32, [u32; 2], u32, u32, bool)> {
        if !self.is_vertex(a) || !self.is_vertex(b) {
            return Err(MeshError::NotAVertex);
        }
        let eab = self.find_edge(a, b).ok_or(MeshError::NotAnEdge)?;
        let [t0, t1] = self.et[eab as usize];
        let c = tri_other_vertex(self.tv[t0 as usize], a, b);
        let boundary = t1 == INVALID;
        let d = if boundary { INVALID } else { tri_other_vertex(self.tv[t1 as usize], a, b) };
        if c == d {
            return Err(MeshError::DuplicateTriangle);
        }
        // Link condition: common neighbours of a and b must be exactly {c, d}.
        for &ea in self.vedges[a as usize].iter() {
            let x = self.edge_other_vertex(ea, a);
            if x == b || x == c || x == d {
                continue;
            }
            if self.find_edge(b, x).is_some() {
                return Err(MeshError::InvalidNeighbourhood);
            }
        }
        // Interior edge between two boundary vertices: collapse would pinch the surface.
        if !boundary && self.is_boundary_vertex(a) && self.is_boundary_vertex(b) {
            return Err(MeshError::InvalidNeighbourhood);
        }
        // Tetrahedron / folded configurations: triangles (a,c,d) and (b,c,d) both exist.
        if !boundary {
            if let Some(ecd) = self.find_edge(c, d) {
                let [s0, s1] = self.et[ecd as usize];
                if s1 != INVALID {
                    let has = |t: u32, v: u32| self.tv[t as usize].contains(&v);
                    if (has(s0, a) && has(s1, b)) || (has(s0, b) && has(s1, a)) {
                        return Err(MeshError::CollapseTetrahedron);
                    }
                }
            }
        }
        // Collapsing must not leave an edge with no triangles: if both other edges of a
        // removed triangle are boundary edges, the triangle is a dangling "ear".
        let eac = self.find_edge_from_tri(a, c, t0).ok_or(MeshError::BrokenTopology)?;
        let ebc = self.find_edge_from_tri(b, c, t0).ok_or(MeshError::BrokenTopology)?;
        if self.is_boundary_edge(eac) && self.is_boundary_edge(ebc) {
            return Err(MeshError::InvalidNeighbourhood);
        }
        if !boundary {
            let ead = self.find_edge_from_tri(a, d, t1).ok_or(MeshError::BrokenTopology)?;
            let ebd = self.find_edge_from_tri(b, d, t1).ok_or(MeshError::BrokenTopology)?;
            if self.is_boundary_edge(ead) && self.is_boundary_edge(ebd) {
                return Err(MeshError::InvalidNeighbourhood);
            }
        }
        Ok((eab, [t0, t1], c, d, boundary))
    }

    /// Collapse edge (v_keep, v_remove), deleting `v_remove`. `v_keep` keeps its position;
    /// callers typically set it afterwards (e.g. to the edge midpoint).
    pub fn collapse_edge(&mut self, v_keep: u32, v_remove: u32) -> MeshResult<EdgeCollapseInfo> {
        let b = v_keep;
        let a = v_remove;
        let (eab, [t0, t1], c, d, boundary) = self.collapse_check(b, a)?;

        let eac = self.find_edge_from_tri(a, c, t0).unwrap();
        let ebc = self.find_edge_from_tri(b, c, t0).unwrap();
        let tac = self.edge_other_triangle(eac, t0);
        let (ead, ebd, tad) = if !boundary {
            let ead = self.find_edge_from_tri(a, d, t1).unwrap();
            let ebd = self.find_edge_from_tri(b, d, t1).unwrap();
            (ead, ebd, self.edge_other_triangle(ead, t1))
        } else {
            (INVALID, INVALID, INVALID)
        };

        // 1) Re-point triangles of a (other than t0/t1) at b.
        let tris_a = self.vertex_triangles(a);
        for &t in tris_a.iter() {
            if t == t0 || t == t1 {
                continue;
            }
            for x in self.tv[t as usize].iter_mut() {
                if *x == a {
                    *x = b;
                }
            }
        }
        // 2) Re-point edges of a (other than ab, ac, ad) at b.
        let edges_a: EdgeList = self.vedges[a as usize].clone();
        for &e in edges_a.iter() {
            if e == eab || e == eac || e == ead {
                continue;
            }
            let o = self.edge_other_vertex(e, a);
            self.ev[e as usize] = sorted2(b, o);
            self.vedges[b as usize].push(e);
        }
        // 3) Detach removed edges from their surviving endpoints.
        self.remove_vertex_edge(b, eab);
        self.remove_vertex_edge(c, eac);
        if !boundary {
            self.remove_vertex_edge(d, ead);
        }
        // 4) Stitch across the removed triangles.
        self.replace_edge_triangle(ebc, t0, tac);
        if tac != INVALID {
            self.replace_tri_edge(tac, eac, ebc);
        }
        if !boundary {
            self.replace_edge_triangle(ebd, t1, tad);
            if tad != INVALID {
                self.replace_tri_edge(tad, ead, ebd);
            }
        }
        // 5) Free removed elements.
        self.free_triangle(t0);
        self.free_edge(eab);
        self.free_edge(eac);
        if !boundary {
            self.free_triangle(t1);
            self.free_edge(ead);
        }
        self.free_vertex(a);

        self.topo_version = next_version();
        self.shape_version = next_version();
        Ok(EdgeCollapseInfo {
            v_kept: b,
            v_removed: a,
            is_boundary: boundary,
            e_collapsed: eab,
            t_removed: [t0, t1],
            e_removed: [eac, ead],
            e_kept: [ebc, ebd],
        })
    }

    /// Insert a vertex inside triangle `t` at barycentric coordinates `bary`,
    /// replacing `t` with three triangles.
    pub fn poke_triangle(&mut self, t: u32, bary: DVec3) -> MeshResult<PokeInfo> {
        if !self.is_triangle(t) {
            return Err(MeshError::NotATriangle);
        }
        let [a, b, c] = self.tv[t as usize];
        let [e_ab, e_bc, e_ca] = self.te[t as usize];
        let p = self.vpos[a as usize] * bary.x + self.vpos[b as usize] * bary.y + self.vpos[c as usize] * bary.z;
        let g = self.tgroup[t as usize];
        let f = self.alloc_vertex(p);
        if let Some(col) = &mut self.vcolor {
            let (ca, cb, cc) = (col[a as usize], col[b as usize], col[c as usize]);
            col[f as usize] = [0, 1, 2].map(|i| ca[i] * bary.x as f32 + cb[i] * bary.y as f32 + cc[i] * bary.z as f32);
        }
        // t: (a, b, f), t1: (b, c, f), t2: (c, a, f)
        let t1 = self.alloc_triangle([b, c, f], [INVALID; 3], g);
        let t2 = self.alloc_triangle([c, a, f], [INVALID; 3], g);
        let e_af = self.alloc_edge(a, f, t2, t);
        let e_bf = self.alloc_edge(b, f, t, t1);
        let e_cf = self.alloc_edge(c, f, t1, t2);
        self.tv[t as usize] = [a, b, f];
        self.te[t as usize] = [e_ab, e_bf, e_af];
        self.te[t1 as usize] = [e_bc, e_cf, e_bf];
        self.te[t2 as usize] = [e_ca, e_af, e_cf];
        self.replace_edge_triangle(e_bc, t, t1);
        self.replace_edge_triangle(e_ca, t, t2);
        self.topo_version = next_version();
        self.shape_version = next_version();
        Ok(PokeInfo { v_new: f, t_orig: t, t_new: [t1, t2] })
    }

    /// Detach the given triangles (all incident to `v`) onto a new vertex at the same
    /// position. Edges shared between a detached and a kept triangle are duplicated, so they
    /// become a pair of boundary edges. Typical use: splitting a bowtie (non-manifold)
    /// vertex by passing one of its triangle fans. Triangle ids are preserved.
    pub fn split_vertex(&mut self, v: u32, tris: &[u32]) -> MeshResult<u32> {
        if !self.is_vertex(v) {
            return Err(MeshError::NotAVertex);
        }
        for &t in tris {
            if !self.is_triangle(t) || !self.tv[t as usize].contains(&v) {
                return Err(MeshError::NotATriangle);
            }
        }
        let in_set = |t: u32| t != INVALID && tris.contains(&t);
        let nv = self.alloc_vertex(self.vpos[v as usize]);
        if let Some(c) = &mut self.vcolor {
            c[nv as usize] = c[v as usize];
        }
        let edges: EdgeList = self.vedges[v as usize].clone();
        for e in edges {
            let [t0, t1] = self.et[e as usize];
            let (i0, i1) = (in_set(t0), in_set(t1));
            let x = self.edge_other_vertex(e, v);
            if i0 && (t1 == INVALID || i1) {
                // Entirely detached: move the edge to the new vertex.
                self.remove_vertex_edge(v, e);
                self.ev[e as usize] = sorted2(nv, x);
                self.vedges[nv as usize].push(e);
            } else if i0 || i1 {
                // Shared by a detached and a kept triangle: give the detached one a copy.
                let (t_det, t_keep) = if i0 { (t0, t1) } else { (t1, t0) };
                self.et[e as usize] = [t_keep, INVALID];
                let ne = self.alloc_edge(nv, x, t_det, INVALID);
                self.replace_tri_edge(t_det, e, ne);
            }
        }
        for &t in tris {
            for x in self.tv[t as usize].iter_mut() {
                if *x == v {
                    *x = nv;
                }
            }
        }
        if self.vedges[v as usize].is_empty() {
            self.free_vertex(v);
        }
        self.topo_version = next_version();
        self.shape_version = next_version();
        Ok(nv)
    }

    /// Split every bowtie vertex so each triangle fan gets its own vertex.
    /// Returns the number of vertices created.
    pub fn split_bowtie_vertices(&mut self) -> usize {
        let mut created = 0;
        let candidates: Vec<u32> = self.vertex_ids().filter(|&v| self.is_bowtie_vertex(v)).collect();
        for v in candidates {
            // Peel off fans until a single one remains.
            while self.is_vertex(v) && self.is_bowtie_vertex(v) {
                let tris = self.vertex_triangles(v);
                let fan = self.triangle_fan(v, tris[0]);
                if fan.len() == tris.len() || self.split_vertex(v, &fan).is_err() {
                    break;
                }
                created += 1;
            }
        }
        created
    }

    /// Triangles around `v` reachable from `t` through edges incident to `v`.
    pub fn triangle_fan(&self, v: u32, t: u32) -> Vec<u32> {
        let mut fan = vec![t];
        let mut k = 0;
        while k < fan.len() {
            let cur = fan[k];
            k += 1;
            for &e in self.te[cur as usize].iter() {
                if !self.ev[e as usize].contains(&v) {
                    continue;
                }
                let o = self.edge_other_triangle(e, cur);
                if o != INVALID && !fan.contains(&o) {
                    fan.push(o);
                }
            }
        }
        fan
    }

    /// Remove a single vertex that has no incident edges.
    pub fn remove_isolated_vertex(&mut self, v: u32) -> MeshResult<()> {
        if !self.is_vertex(v) {
            return Err(MeshError::NotAVertex);
        }
        if !self.vedges[v as usize].is_empty() {
            return Err(MeshError::InvalidNeighbourhood);
        }
        self.free_vertex(v);
        Ok(())
    }

    // ------------------------------------------------------------------
    // Bulk operations
    // ------------------------------------------------------------------

    /// Append all triangles of `other` into this mesh (vertices are copied, no welding).
    /// Returns the vertex map from `other` ids to new ids.
    pub fn append_mesh(&mut self, other: &Mesh, group_offset: i32) -> Vec<u32> {
        let mut vmap = vec![INVALID; other.max_vid()];
        if other.vcolor.is_some() {
            self.enable_vertex_colors();
        }
        for v in other.vertex_ids() {
            vmap[v as usize] = self.alloc_vertex(other.vertex(v));
            if let (Some(c), Some(oc)) = (&mut self.vcolor, &other.vcolor) {
                c[vmap[v as usize] as usize] = oc[v as usize];
            }
        }
        for t in other.triangle_ids() {
            let [a, b, c] = other.triangle(t);
            let tv = [vmap[a as usize], vmap[b as usize], vmap[c as usize]];
            self.append_triangle(tv, other.tri_group(t) + group_offset)
                .expect("copying a valid mesh cannot fail");
        }
        self.topo_version = next_version();
        self.shape_version = next_version();
        vmap
    }

    /// Copy of the mesh restricted to the given triangles.
    pub fn submesh(&self, tris: &[u32]) -> (Mesh, Vec<u32>) {
        let mut vmap = vec![INVALID; self.max_vid()];
        let mut m = Mesh::with_capacity(tris.len(), tris.len());
        if self.vcolor.is_some() {
            m.enable_vertex_colors();
        }
        for &t in tris {
            let mut tv = [INVALID; 3];
            for (k, &v) in self.tv[t as usize].iter().enumerate() {
                if vmap[v as usize] == INVALID {
                    vmap[v as usize] = m.alloc_vertex(self.vpos[v as usize]);
                    if let (Some(c), Some(sc)) = (&mut m.vcolor, &self.vcolor) {
                        c[vmap[v as usize] as usize] = sc[v as usize];
                    }
                }
                tv[k] = vmap[v as usize];
            }
            m.append_triangle(tv, self.tgroup[t as usize]).expect("submesh of valid mesh");
        }
        (m, vmap)
    }

    /// Compact copy: contiguous vertex and triangle ids. Returns (mesh, old->new vertex map).
    pub fn compacted(&self) -> (Mesh, Vec<u32>) {
        let tris: Vec<u32> = self.triangle_ids().collect();
        self.submesh(&tris)
    }

    /// Whether the mesh carries per-vertex colours.
    pub fn has_vertex_colors(&self) -> bool {
        self.vcolor.is_some()
    }
    /// Colour of vertex `v` (when the mesh has colours).
    pub fn vertex_color(&self, v: u32) -> Option<[f32; 3]> {
        self.vcolor.as_ref().map(|c| c[v as usize])
    }
    /// Set the colour of `v`, giving the mesh colours (default grey) if it had none.
    pub fn set_vertex_color(&mut self, v: u32, c: [f32; 3]) {
        self.enable_vertex_colors();
        if let Some(col) = &mut self.vcolor {
            col[v as usize] = c;
        }
        self.color_version = self.color_version.wrapping_add(1);
    }
    pub fn color_version(&self) -> u64 {
        self.color_version
    }
    /// Give the mesh per-vertex colours (all [`DEFAULT_VERTEX_COLOR`]) if it has none.
    pub fn enable_vertex_colors(&mut self) {
        if self.vcolor.is_none() {
            self.vcolor = Some(vec![DEFAULT_VERTEX_COLOR; self.vpos.len()]);
            self.color_version = self.color_version.wrapping_add(1);
        }
    }
    pub fn clear_vertex_colors(&mut self) {
        self.vcolor = None;
        self.color_version = self.color_version.wrapping_add(1);
    }
    /// Colour at barycentric coordinates `bary` of triangle `t`.
    pub fn color_at(&self, t: u32, bary: DVec3) -> Option<[f32; 3]> {
        let c = self.vcolor.as_ref()?;
        let [a, b, d] = self.tv[t as usize].map(|v| c[v as usize]);
        Some([0, 1, 2].map(|i| a[i] * bary.x as f32 + b[i] * bary.y as f32 + d[i] * bary.z as f32))
    }
    /// Vertex colours in the order of [`Mesh::to_arrays`]' positions.
    pub fn colors_compact(&self) -> Option<Vec<[f32; 3]>> {
        let c = self.vcolor.as_ref()?;
        Some(self.vertex_ids().map(|v| c[v as usize]).collect())
    }

    /// Flat arrays (compacted) suitable for export / GPU upload.
    pub fn to_arrays(&self) -> (Vec<DVec3>, Vec<[u32; 3]>, Vec<i32>) {
        let mut vmap = vec![INVALID; self.max_vid()];
        let mut pos = Vec::with_capacity(self.nv);
        for v in self.vertex_ids() {
            vmap[v as usize] = pos.len() as u32;
            pos.push(self.vpos[v as usize]);
        }
        let mut tris = Vec::with_capacity(self.nt);
        let mut groups = Vec::with_capacity(self.nt);
        for t in self.triangle_ids() {
            let [a, b, c] = self.tv[t as usize];
            tris.push([vmap[a as usize], vmap[b as usize], vmap[c as usize]]);
            groups.push(self.tgroup[t as usize]);
        }
        (pos, tris, groups)
    }

    /// Apply a function to every live vertex position.
    pub fn transform(&mut self, f: impl Fn(DVec3) -> DVec3) {
        for i in 0..self.vpos.len() {
            if self.vlive[i] {
                self.vpos[i] = f(self.vpos[i]);
            }
        }
        self.shape_version = next_version();
    }

    // ------------------------------------------------------------------
    // Validation (used heavily by tests)
    // ------------------------------------------------------------------

    /// Exhaustively verify internal consistency.
    pub fn check_validity(&self) -> Result<(), String> {
        let mut nv = 0;
        let mut nt = 0;
        let mut ne = 0;
        for v in 0..self.max_vid() as u32 {
            if !self.vlive[v as usize] {
                continue;
            }
            nv += 1;
            for &e in self.vedges[v as usize].iter() {
                if !self.is_edge(e) {
                    return Err(format!("vertex {v} lists dead edge {e}"));
                }
                if !self.ev[e as usize].contains(&v) {
                    return Err(format!("vertex {v} lists edge {e} not containing it"));
                }
            }
            let mut sorted: Vec<u32> = self.vedges[v as usize].to_vec();
            sorted.sort_unstable();
            sorted.dedup();
            if sorted.len() != self.vedges[v as usize].len() {
                return Err(format!("vertex {v} has duplicate edges"));
            }
        }
        for e in 0..self.max_eid() as u32 {
            if !self.elive[e as usize] {
                continue;
            }
            ne += 1;
            let [a, b] = self.ev[e as usize];
            if a >= b {
                return Err(format!("edge {e} vertices not sorted"));
            }
            if !self.is_vertex(a) || !self.is_vertex(b) {
                return Err(format!("edge {e} has dead vertex"));
            }
            if !self.vedges[a as usize].contains(&e) || !self.vedges[b as usize].contains(&e) {
                return Err(format!("edge {e} missing from vertex lists"));
            }
            let [t0, t1] = self.et[e as usize];
            if !self.is_triangle(t0) {
                return Err(format!("edge {e} slot0 invalid triangle {t0}"));
            }
            if t1 != INVALID && !self.is_triangle(t1) {
                return Err(format!("edge {e} slot1 invalid triangle {t1}"));
            }
            if t0 == t1 {
                return Err(format!("edge {e} has same triangle twice"));
            }
            for &t in &[t0, t1] {
                if t == INVALID {
                    continue;
                }
                if !self.te[t as usize].contains(&e) {
                    return Err(format!("edge {e} triangle {t} does not reference it"));
                }
                if tri_edge_index(self.tv[t as usize], a, b).is_none() {
                    return Err(format!("edge {e} vertices not in triangle {t}"));
                }
            }
            // No duplicate edges between a and b.
            let dup = self.vedges[a as usize]
                .iter()
                .filter(|&&x| self.ev[x as usize] == [a, b])
                .count();
            if dup != 1 {
                return Err(format!("duplicate edge between {a} and {b}"));
            }
        }
        for t in 0..self.max_tid() as u32 {
            if !self.tlive[t as usize] {
                continue;
            }
            nt += 1;
            let tv = self.tv[t as usize];
            let te = self.te[t as usize];
            for j in 0..3 {
                if !self.is_vertex(tv[j]) {
                    return Err(format!("triangle {t} has dead vertex"));
                }
                let e = te[j];
                if !self.is_edge(e) {
                    return Err(format!("triangle {t} has dead edge {e}"));
                }
                if self.ev[e as usize] != sorted2(tv[j], tv[(j + 1) % 3]) {
                    return Err(format!("triangle {t} edge {j} mismatch"));
                }
                if !self.et[e as usize].contains(&t) {
                    return Err(format!("triangle {t} edge {e} does not list it"));
                }
            }
            if tv[0] == tv[1] || tv[1] == tv[2] || tv[0] == tv[2] {
                return Err(format!("triangle {t} degenerate"));
            }
        }
        if nv != self.nv || nt != self.nt || ne != self.ne {
            return Err(format!(
                "count mismatch: v {nv}/{} t {nt}/{} e {ne}/{}",
                self.nv, self.nt, self.ne
            ));
        }
        Ok(())
    }

    /// True if every interior edge is shared by triangles with opposite winding.
    pub fn is_consistently_oriented(&self) -> bool {
        self.edge_ids().all(|e| {
            let [t0, t1] = self.et[e as usize];
            if t1 == INVALID {
                return true;
            }
            let [a, b] = self.ev[e as usize];
            let tv0 = self.tv[t0 as usize];
            let tv1 = self.tv[t1 as usize];
            let j0 = tri_edge_index(tv0, a, b).unwrap();
            let j1 = tri_edge_index(tv1, a, b).unwrap();
            tv0[j0] == tv1[(j1 + 1) % 3]
        })
    }
}

/// Statistics from [`Mesh::from_triangles`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BuildReport {
    pub degenerate: usize,
    pub duplicate: usize,
    pub nonmanifold_split: usize,
    pub invalid_index: usize,
    pub unreferenced_vertices: usize,
}
