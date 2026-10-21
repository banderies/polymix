//! Optional triangle region that restricts an operation.
//!
//! Outside triangles are never modified: every vertex touching an outside triangle and
//! every edge on the region border is constrained as fixed by the callers, so local
//! edits only ever touch region triangles. Triangles created inside the region (by
//! splits) are added; triangles removed by collapses are dropped lazily.
//!
//! The edge and vertex lists come in increasing id order. A region that covers a good part
//! of the mesh gets them from a parallel scan over all ids; a small one (a sculpt dab on a
//! large mesh) from its own triangles, so that its cost does not grow with the mesh. For the
//! same reason the per-id state of a small region (here and in the remesher and reducer)
//! lives in hash maps rather than arrays over every id, see [`IdVals`].

use crate::constraints::with_headroom;
use crate::util::Stamp;
use mm_core::hash::{fast_set_with_capacity, FastMap, FastSet};
use mm_core::{Mesh, INVALID};
use rayon::prelude::*;

/// Shortest piece of a list that rayon is given: shorter lists (dab-sized ones) are scanned
/// serially. Handing them to the worker threads costs more than the scan, and on the UI
/// thread it means waiting for those threads to be scheduled, which on a busy machine takes
/// milliseconds.
pub(crate) const PAR_MIN_LEN: usize = 2048;

#[cfg(test)]
thread_local! {
    /// Lets tests force the storage of the regions they create: `Some(true)` hash maps,
    /// `Some(false)` arrays (the results are the same either way).
    pub(crate) static FORCE_SPARSE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Should the per-id state of a region of `n` triangles on `mesh` live in hash maps?
///
/// A hash-map access costs several array accesses, and a pass makes many per triangle, so
/// maps only win while the arrays over every id cost more to clear than that. Measured on
/// 1M and 3M-triangle vessels, the two break even at about 1/900 and 1/500 of the mesh for
/// one-pass edits (density dabs, refine, reduce) and a little lower for multi-pass remeshes,
/// so normal brush dabs (about a thousand triangles) stay in maps and selections move to
/// arrays.
fn sparse_for(n: usize, mesh: &Mesh) -> bool {
    #[cfg(test)]
    if let Some(s) = FORCE_SPARSE.with(|f| f.get()) {
        return s;
    }
    n * 512 < mesh.max_tid()
}

/// Values per vertex, edge or triangle id, all at the default (zero) until set. Over a whole
/// mesh or a large region they live in an array over every id. For a small region on a large
/// mesh they live in a hash map of the ids set: an array over every id is not free even when
/// zeroed, as the allocator zeroes the large blocks it reuses, so its cost would grow with
/// the mesh. Both give the same values.
///
/// In hash-map mode the array stays empty, so a read or write in array mode costs what an
/// array access costs (a bounds check) and only misses go on to the map. Hot loops are also
/// compiled once per mode (see [`IdVals::at`]).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct IdVals<T> {
    dense: Vec<T>,
    sparse: Option<FastMap<u32, T>>,
}

impl<T: Copy + Default> IdVals<T> {
    /// Values for ids below `n` (more can be set later), in a hash map if `sparse`.
    pub fn new(sparse: bool, n: usize) -> Self {
        if sparse {
            Self { dense: Vec::new(), sparse: Some(FastMap::default()) }
        } else {
            Self::from_vec(vec![T::default(); n])
        }
    }

    /// An array of values by id.
    pub fn from_vec(dense: Vec<T>) -> Self {
        Self { dense, sparse: None }
    }

    pub fn is_sparse(&self) -> bool {
        self.sparse.is_some()
    }

    #[inline]
    pub fn get(&self, i: u32) -> T {
        match self.dense.get(i as usize) {
            Some(&x) => x,
            None => self.get_missed(i),
        }
    }

    #[inline(never)]
    fn get_missed(&self, i: u32) -> T {
        self.sparse.as_ref().and_then(|m| m.get(&i).copied()).unwrap_or_default()
    }

    /// [`IdVals::get`] for code compiled for one mode: `MAPS` must say whether the values
    /// are in a hash map. In array mode it is the plain array read. A possible call into the
    /// hash-map path makes small hot functions save registers on every call, which costs a
    /// whole-mesh remesh a few percent, so the passes are compiled once per mode instead.
    #[inline(always)]
    pub fn at<const MAPS: bool>(&self, i: u32) -> T {
        if MAPS {
            self.get(i)
        } else {
            debug_assert!(self.sparse.is_none(), "array read of values in a hash map");
            self.dense.get(i as usize).copied().unwrap_or_default()
        }
    }

    /// [`IdVals::set`] for code compiled for one mode (see [`IdVals::at`]); in array mode
    /// `i` must be inside the array (as for marks over ids that exist).
    #[inline(always)]
    pub fn set_at<const MAPS: bool>(&mut self, i: u32, x: T) {
        if MAPS {
            self.set(i, x)
        } else {
            debug_assert!(self.sparse.is_none(), "array write of values in a hash map");
            self.dense[i as usize] = x;
        }
    }

    #[inline]
    pub fn set(&mut self, i: u32, x: T) {
        match self.dense.get_mut(i as usize) {
            Some(slot) => *slot = x,
            None => self.set_missed(i, x),
        }
    }

    #[inline(never)]
    fn set_missed(&mut self, i: u32, x: T) {
        match &mut self.sparse {
            Some(m) => {
                m.insert(i, x);
            }
            None => {
                let k = i as usize;
                self.dense.resize(with_headroom(k + 1), T::default());
                self.dense[k] = x;
            }
        }
    }

    /// Move the values into an array over ids below `n` (and any id set). An array already
    /// shorter than `n` is lengthened in one step, rather than id by id as later writes
    /// would.
    pub fn densify(&mut self, n: usize) {
        if let Some(m) = self.sparse.take() {
            self.dense = vec![T::default(); n];
            // Each id is written once, so the order of the walk does not matter.
            for (i, x) in m {
                self.set(i, x);
            }
        } else if self.dense.len() < n {
            self.dense.resize(n, T::default());
        }
    }

    /// Drop every value, keeping the mode and the allocations.
    pub fn clear(&mut self) {
        match &mut self.sparse {
            Some(m) => m.clear(),
            None => self.dense.fill(T::default()),
        }
    }

    /// Make this an array holding `src` (reusing the allocation).
    pub fn copy_from(&mut self, src: &[T]) {
        self.sparse = None;
        self.dense.clear();
        self.dense.extend_from_slice(src);
    }
}

pub(crate) struct Region {
    enabled: bool,
    tin: IdVals<bool>,
    tris: Vec<u32>,
    /// Vertex marks for the walks of a region in arrays (see [`Region::vertices`]).
    seen: Stamp,
}

impl Region {
    pub fn new(mesh: &Mesh, region: Option<&[u32]>) -> Self {
        match region {
            None => Self { enabled: false, tin: IdVals::from_vec(Vec::new()), tris: Vec::new(), seen: Stamp::default() },
            Some(r) => {
                // With room for the triangles splits add.
                let mut tin = IdVals::<bool>::new(sparse_for(r.len(), mesh), with_headroom(mesh.max_tid()));
                let mut tris = Vec::with_capacity(r.len());
                for &t in r {
                    if mesh.is_triangle(t) && !tin.get(t) {
                        tin.set(t, true);
                        tris.push(t);
                    }
                }
                tris.sort_unstable();
                Self { enabled: true, tin, tris, seen: Stamp::default() }
            }
        }
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// True when the region's per-id state lives in hash maps (see [`IdVals`]); the
    /// remesher and reducer keep theirs the same way.
    #[inline]
    pub fn is_sparse(&self) -> bool {
        self.tin.is_sparse()
    }

    /// A small region that grew to a good part of the mesh (by splits) moves its flags into
    /// an array, which beats a hash map then; returns true when it did, so the caller can do
    /// the same with its own state. Triangles not yet refreshed out count too, so the switch
    /// comes a bit early.
    pub fn densify_if_large(&mut self, mesh: &Mesh) -> bool {
        if !self.tin.is_sparse() || sparse_for(self.tris.len() / 2, mesh) {
            return false;
        }
        self.tin.densify(with_headroom(mesh.max_tid()));
        true
    }

    /// True if `t` (a live triangle) belongs to the region.
    #[inline]
    pub fn contains(&self, t: u32) -> bool {
        !self.enabled || self.tin.get(t)
    }

    /// [`Region::contains`] compiled for one mode (`MAPS == self.is_sparse()`, see [`IdVals::at`]).
    #[inline(always)]
    pub fn contains_in<const MAPS: bool>(&self, t: u32) -> bool {
        !self.enabled || self.tin.at::<MAPS>(t)
    }

    /// Register a triangle created inside the region.
    #[inline]
    pub fn add(&mut self, t: u32) {
        if !self.enabled {
            return;
        }
        self.tin.set(t, true);
        self.tris.push(t);
    }

    /// Drop dead ids and duplicates from the triangle list.
    pub fn refresh(&mut self, mesh: &Mesh) {
        if !self.enabled {
            return;
        }
        let tin = &self.tin;
        self.tris.retain(|&t| mesh.is_triangle(t) && tin.get(t));
        self.tris.sort_unstable();
        self.tris.dedup();
    }

    /// Live region triangles (all live triangles when no region is set).
    pub fn triangles(&mut self, mesh: &Mesh) -> Vec<u32> {
        if !self.enabled {
            return mesh.triangle_ids().collect();
        }
        self.refresh(mesh);
        self.tris.clone()
    }

    pub fn triangle_count(&mut self, mesh: &Mesh) -> usize {
        if !self.enabled {
            return mesh.triangle_count();
        }
        self.refresh(mesh);
        self.tris.len()
    }

    /// True when the region is small next to the mesh, so walking its own triangles beats a
    /// scan over every id (always with flags in a hash map). Both ways give the same lists.
    #[inline]
    fn is_small(&self, mesh: &Mesh) -> bool {
        self.tris.len() * 16 < mesh.max_tid() || self.tin.is_sparse()
    }

    /// Edges of region triangles (all live edges when no region is set), sorted.
    ///
    /// A live edge's triangles are live, and the live triangles flagged in `tin` are exactly
    /// the region (ids are only flagged when added, and only region triangles are created),
    /// so the scan keeps the same edges as a walk over the region triangles.
    pub fn edges(&mut self, mesh: &Mesh) -> Vec<u32> {
        if !self.enabled {
            (0..mesh.max_eid() as u32).into_par_iter().with_min_len(PAR_MIN_LEN).filter(|&e| mesh.is_edge(e)).collect()
        } else if self.is_small(mesh) {
            self.edges_walk(mesh)
        } else {
            self.edges_scan(mesh)
        }
    }

    fn edges_walk(&mut self, mesh: &Mesh) -> Vec<u32> {
        self.refresh(mesh);
        // Each edge once, from the lower-id region triangle on it: the other triangle of an
        // edge is live, so its flag says whether it is in the region. This sorts half the
        // list that a sort with deduplication would, and needs no marks over every id.
        let tin = &self.tin;
        let mut out = Vec::with_capacity(self.tris.len() * 3 / 2 + 8);
        for &t in &self.tris {
            for e in mesh.tri_edges(t) {
                let o = mesh.edge_other_triangle(e, t);
                if o == INVALID || o > t || !tin.get(o) {
                    out.push(e);
                }
            }
        }
        out.sort_unstable();
        out
    }

    fn edges_scan(&self, mesh: &Mesh) -> Vec<u32> {
        let tin = &self.tin;
        let inside = |t: u32| tin.get(t);
        (0..mesh.max_eid() as u32)
            .into_par_iter()
            .with_min_len(PAR_MIN_LEN)
            .filter(|&e| {
                if !mesh.is_edge(e) {
                    return false;
                }
                let [t0, t1] = mesh.edge_triangles(e);
                inside(t0) || inside(t1)
            })
            .collect()
    }

    /// Vertices of region triangles (all live vertices when no region is set), sorted.
    pub fn vertices(&mut self, mesh: &Mesh) -> Vec<u32> {
        if !self.enabled {
            (0..mesh.max_vid() as u32).into_par_iter().with_min_len(PAR_MIN_LEN).filter(|&v| mesh.is_vertex(v)).collect()
        } else if self.is_small(mesh) {
            self.vertices_walk(mesh)
        } else {
            self.vertices_scan(mesh)
        }
    }

    fn vertices_walk(&mut self, mesh: &Mesh) -> Vec<u32> {
        self.refresh(mesh);
        let mut out: Vec<u32>;
        if self.tin.is_sparse() {
            out = self.tris.iter().flat_map(|&t| mesh.triangle(t)).collect();
            out.sort_unstable();
            out.dedup();
        } else {
            // A vertex is a corner of about six triangles, so dropping repeats with marks
            // first sorts a sixth of the list. The marks are over every vertex id, which a
            // region with its state in arrays pays for anyway (once, they are reused).
            let seen = &mut self.seen;
            seen.begin(mesh.max_vid());
            out = Vec::with_capacity(self.tris.len() / 2 + 8);
            for &t in &self.tris {
                for v in mesh.triangle(t) {
                    if seen.mark(v) {
                        out.push(v);
                    }
                }
            }
            out.sort_unstable();
        }
        out
    }

    fn vertices_scan(&self, mesh: &Mesh) -> Vec<u32> {
        let tin = &self.tin;
        let inside = |t: u32| tin.get(t);
        // A live vertex's triangles are the triangles of its edges.
        (0..mesh.max_vid() as u32)
            .into_par_iter()
            .with_min_len(PAR_MIN_LEN)
            .filter(|&v| {
                mesh.is_vertex(v)
                    && mesh.vertex_edges(v).iter().any(|&e| {
                        let [t0, t1] = mesh.edge_triangles(e);
                        inside(t0) || inside(t1)
                    })
            })
            .collect()
    }

    /// Region triangles plus every outside triangle sharing a vertex with the region
    /// (used to build the projection reference near the region border).
    pub fn triangles_with_collar(&mut self, mesh: &Mesh) -> Vec<u32> {
        let mut tris = self.triangles(mesh);
        if !self.enabled {
            return tris;
        }
        let small = self.is_small(mesh);
        let mut seen_t = Seen::new(small, mesh.max_tid(), 2 * tris.len());
        let mut seen_v = Seen::new(small, mesh.max_vid(), tris.len());
        for &t in &tris {
            seen_t.insert(t);
        }
        let mut extra = Vec::new();
        for &t in &tris {
            for v in mesh.triangle(t) {
                if !seen_v.insert(v) {
                    continue;
                }
                for n in mesh.vertex_triangles(v) {
                    if seen_t.insert(n) {
                        extra.push(n);
                    }
                }
            }
        }
        tris.extend(extra);
        tris
    }
}

/// Ids seen so far: marks over every id for a large region, a hash set for a small one (the
/// same answers either way).
enum Seen {
    Marks(Stamp),
    Set(FastSet<u32>),
}

impl Seen {
    fn new(small: bool, ids: usize, expected: usize) -> Self {
        if small {
            Seen::Set(fast_set_with_capacity(expected))
        } else {
            let mut st = Stamp::default();
            st.begin(ids);
            Seen::Marks(st)
        }
    }

    /// True if `i` was not seen before.
    fn insert(&mut self, i: u32) -> bool {
        match self {
            Seen::Marks(st) => st.mark(i),
            Seen::Set(set) => set.insert(i),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_core::primitives;

    #[test]
    fn values_in_maps_and_arrays_agree() {
        // The same writes to a hash map, an array, and an array that starts short, against a
        // plain reference; reads of ids never set (and past every end) give the default.
        let mut s = 7u64;
        let mut rnd = |n: u64| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 33) % n
        };
        let mut maps = IdVals::<u64>::new(true, 0);
        let mut arrays = IdVals::<u64>::new(false, 500);
        let mut short = IdVals::<u64>::new(false, 3);
        let mut want = vec![0u64; 5000];
        for k in 0..3000 {
            let (i, x) = (rnd(5000) as u32, rnd(4));
            for v in [&mut maps, &mut arrays, &mut short] {
                v.set(i, x);
            }
            want[i as usize] = x;
            if k == 1500 {
                // Moving a map into an array keeps every value.
                let mut moved = maps.clone();
                moved.densify(100);
                assert!(!moved.is_sparse());
                assert!((0..6000).all(|i| moved.get(i) == maps.get(i)));
            }
        }
        assert!(maps.is_sparse() && !arrays.is_sparse());
        for i in (0..6000).chain([u32::MAX - 1, u32::MAX]) {
            let w = want.get(i as usize).copied().unwrap_or(0);
            assert_eq!((maps.get(i), maps.at::<true>(i)), (w, w), "id {i}");
            assert_eq!((arrays.get(i), arrays.at::<false>(i), arrays.at::<true>(i)), (w, w, w), "id {i}");
            assert_eq!(short.get(i), w, "id {i}");
        }
        maps.clear();
        assert!(maps.is_sparse() && (0..6000).all(|i| maps.get(i) == 0));
        arrays.copy_from(&[5, 6, 7]);
        assert!(!arrays.is_sparse() && arrays.get(2) == 7 && arrays.get(3) == 0);
    }

    /// The old list builders: a walk over the refreshed region triangles with id marks.
    fn marked(r: &mut Region, mesh: &Mesh, ids: impl Fn(u32) -> [u32; 3], n: usize) -> Vec<u32> {
        r.refresh(mesh);
        let mut stamp = Stamp::default();
        stamp.begin(n);
        let mut out = Vec::new();
        for &t in &r.tris {
            for x in ids(t) {
                if stamp.mark(x) {
                    out.push(x);
                }
            }
        }
        out.sort_unstable();
        out
    }

    #[test]
    fn scans_and_walks_give_the_same_lists() {
        let base = primitives::icosphere(1.0, 4);
        // The small region with its flags in an array (as its size picks) and in a hash map.
        for (zmin, force) in [(0.2, None), (0.97, None), (0.97, Some(true))] {
            let mut m = base.clone();
            let tris: Vec<u32> = m.triangle_ids().filter(|&t| m.tri_centroid(t).z > zmin).collect();
            FORCE_SPARSE.with(|f| f.set(force));
            let mut r = Region::new(&m, Some(&tris));
            FORCE_SPARSE.with(|f| f.set(None));
            assert_eq!(r.is_sparse(), force == Some(true));
            // Edit the region the way a remesh does: splits register new triangles, collapses
            // leave dead ids (still flagged) behind, and later splits reuse them.
            for (k, &t) in tris.iter().enumerate() {
                if !m.is_triangle(t) {
                    continue;
                }
                let e = m.tri_edges(t)[k % 3];
                let [t0, t1] = m.edge_triangles(e);
                if !(r.contains(t0) && r.contains(t1)) {
                    continue;
                }
                let [a, b] = m.edge_vertices(e);
                let interior = |v: u32| m.vertex_triangles(v).iter().all(|&x| r.contains(x));
                if k % 3 == 0 && interior(a) && interior(b) && m.can_collapse(a, b).is_ok() {
                    m.collapse_edge(a, b).unwrap();
                } else if let Ok(info) = m.split_edge(e) {
                    for nt in info.t_new {
                        r.add(nt);
                    }
                }
            }
            m.check_validity().unwrap();
            // Both ways of building the lists are in use.
            assert_eq!(r.is_small(&m), zmin > 0.5);
            let edges = marked(&mut r, &m, |t| m.tri_edges(t), m.max_eid());
            let verts = marked(&mut r, &m, |t| m.triangle(t), m.max_vid());
            assert!(!edges.is_empty() && !verts.is_empty());
            assert_eq!(r.edges_scan(&m), edges);
            assert_eq!(r.edges_walk(&m), edges);
            assert_eq!(r.edges(&m), edges);
            assert_eq!(r.vertices_scan(&m), verts);
            assert_eq!(r.vertices_walk(&m), verts);
            assert_eq!(r.vertices(&m), verts);
            // The collar, against the old walk with marks over every id.
            let mut want = r.triangles(&m);
            let (mut st, mut vs) = (Stamp::default(), Stamp::default());
            st.begin(m.max_tid());
            vs.begin(m.max_vid());
            for &t in &want {
                st.mark(t);
            }
            let mut extra = Vec::new();
            for &t in &want {
                for v in m.triangle(t) {
                    if vs.mark(v) {
                        extra.extend(m.vertex_triangles(v).into_iter().filter(|&n| st.mark(n)));
                    }
                }
            }
            want.extend(extra);
            assert_eq!(r.triangles_with_collar(&m), want);
        }
        // Without a region: every live id.
        let mut m = base.clone();
        let v = m.vertex_ids().nth(10).unwrap();
        let u = m.vertex_neighbors(v).next().unwrap();
        m.collapse_edge(u, v).unwrap();
        let mut r = Region::new(&m, None);
        assert_eq!(r.edges(&m), m.edge_ids().collect::<Vec<_>>());
        assert_eq!(r.vertices(&m), m.vertex_ids().collect::<Vec<_>>());
    }
}
