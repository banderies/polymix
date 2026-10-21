//! Bounding-volume hierarchy over the triangles of a [`Mesh`], for ray casting,
//! nearest-point and range queries.
//!
//! The tree stores triangle ids only; queries take the mesh by reference, so the caller
//! must rebuild the tree after editing the mesh (compare [`Bvh::shape_version`] with
//! [`Mesh::shape_version`]).

use crate::geom::{closest_point_on_triangle, ray_triangle, Aabb, Ray};
use crate::mesh::{Mesh, INVALID};
use glam::DVec3;
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

const LEAF_SIZE: usize = 4;

/// Builds over at least this many triangles take the parallel path. Below it the serial
/// build takes under about 50 ms, and on a busy machine the parallel one was often slower
/// there: the UI thread hands the build to the pool and waits for pool threads to wake.
const PAR_BUILD: usize = 256 * 1024;
/// Within a parallel build, nodes over at least this many triangles fold their bounds in
/// parallel and build their two halves on two threads; smaller subtrees stay on one thread.
const PAR_NODE: usize = 16 * 1024;
/// Trees with at least this many nodes (about 650k triangles) refit in parallel. A smaller
/// tree refits serially in under 2 ms; the parallel refit gained less than 2x there even at
/// best, while on a busy machine the UI thread, which hands the work to the pool and waits,
/// can wait far longer for pool threads to wake.
const PAR_REFIT: usize = 512 * 1024;
/// Within a parallel refit, subtrees of at least this many nodes split across two threads and
/// smaller ones stay on one, so a refit makes a few large tasks rather than many small ones.
const PAR_REFIT_SPLIT: usize = 48 * 1024;

/// The stack of nodes still to visit in a query.
trait NodeStack {
    /// A stack holding the root.
    fn root() -> Self;
    fn push(&mut self, node: u32);
    fn pop(&mut self) -> Option<u32>;
}

impl NodeStack for Vec<u32> {
    #[inline]
    fn root() -> Self {
        let mut s = Vec::with_capacity(64);
        s.push(0);
        s
    }
    #[inline]
    fn push(&mut self, node: u32) {
        Vec::push(self, node);
    }
    #[inline]
    fn pop(&mut self) -> Option<u32> {
        Vec::pop(self)
    }
}

/// A stack in a fixed array, so a query allocates nothing. A query holds at most one node
/// per level plus one, and the builders' trees have at most 31 levels: they halve the
/// triangles (at most 2^32) until a leaf holds 4. So this serves every tree from a builder;
/// any other tree gets a `Vec` (see [`Bvh::fits_array_stack`]). Measured on a 1.3M tree, it
/// is faster than the `Vec` (whose allocation costs most in parallel scans), and a
/// `SmallVec` was slower than both.
struct ArrayStack {
    slots: [u32; 64],
    len: usize,
}

impl NodeStack for ArrayStack {
    #[inline]
    fn root() -> Self {
        ArrayStack { slots: [0; 64], len: 1 }
    }
    #[inline]
    fn push(&mut self, node: u32) {
        self.slots[self.len] = node;
        self.len += 1;
    }
    #[inline]
    fn pop(&mut self) -> Option<u32> {
        if self.len == 0 {
            None
        } else {
            self.len -= 1;
            Some(self.slots[self.len])
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BvhNode {
    pub bounds: Aabb,
    /// For interior nodes: index of the left child (right child is `left + 1`).
    /// For leaves: start into [`Bvh::tris`].
    pub start: u32,
    /// Number of triangles for a leaf; 0 for interior nodes.
    pub count: u32,
}

impl BvhNode {
    #[inline]
    pub fn is_leaf(&self) -> bool {
        self.count > 0
    }
}

#[derive(Clone, Debug, Default)]
pub struct Bvh {
    pub nodes: Vec<BvhNode>,
    pub tris: Vec<u32>,
    shape_version: u64,
    topo_version: u64,
    /// What [`Bvh::refit_vertices`] needs to find a triangle's leaf and a node's parent. Made
    /// on first use and shared by copies: refits never change the shape of the tree.
    links: Option<Arc<RefitLinks>>,
    /// The node count a builder gave the tree (0 if none did). While `nodes` still has that
    /// many, the tree is the builder's, shallow enough for [`ArrayStack`].
    built_nodes: usize,
}

/// The leaf of each triangle and the parent of each node.
#[derive(Debug)]
struct RefitLinks {
    /// Leaf node of each triangle id; `INVALID` for triangles outside the tree.
    leaf_of: Vec<u32>,
    /// Parent of each node; `INVALID` for the root.
    parent: Vec<u32>,
    /// Some triangle is in more than one leaf (a subset build given a repeated id), and
    /// `leaf_of` names only one of them, so refits must redo the whole tree.
    repeats: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct RayHit {
    pub tid: u32,
    pub t: f64,
    pub point: DVec3,
    /// Barycentric weights of the triangle's vertices.
    pub bary: DVec3,
}

#[derive(Clone, Copy, Debug)]
pub struct NearestHit {
    pub tid: u32,
    pub point: DVec3,
    pub dist_sq: f64,
    pub bary: DVec3,
}

struct BuildItem {
    tid: u32,
    centroid: DVec3,
    bounds: Aabb,
}

/// The box around the items and the box around their centroids, folded in slice order.
fn fold_items(items: &[BuildItem]) -> (Aabb, Aabb) {
    let mut bounds = Aabb::EMPTY;
    let mut cb = Aabb::EMPTY;
    for it in items.iter() {
        bounds = bounds.union(&it.bounds);
        cb.include(it.centroid);
    }
    (bounds, cb)
}

/// Node count of the subtree over `n` items, for every size the parallel builder meets.
/// A node splits `n` into `n / 2` and `n - n / 2`, so each level has at most two sizes.
struct NodeCounts(crate::hash::FastMap<usize, usize>);

impl NodeCounts {
    fn new(n: usize) -> Self {
        let mut memo = crate::hash::FastMap::default();
        Self::fill(n, &mut memo);
        Self(memo)
    }
    fn fill(n: usize, memo: &mut crate::hash::FastMap<usize, usize>) -> usize {
        if n <= LEAF_SIZE {
            return 1;
        }
        if let Some(&c) = memo.get(&n) {
            return c;
        }
        let c = 1 + Self::fill(n / 2, memo) + Self::fill(n - n / 2, memo);
        memo.insert(n, c);
        c
    }
    fn get(&self, n: usize) -> usize {
        if n <= LEAF_SIZE {
            1
        } else {
            self.0[&n]
        }
    }
}

/// Builds the subtree over `items` (which start at `lo` in the whole item array), laid out
/// exactly as the serial builder lays it out: the node's children at `base` and `base + 1`,
/// then the right child's descendants, then the left child's (the serial builder pops the
/// right child first). `below` holds those descendants, from index `base` on. Returns the
/// node itself, for the caller to store.
///
/// Each node is made as in the serial builder: its boxes are folded over its items in the
/// order they have when the node is reached, then the items are split around the same
/// median. Nodes of `par` items or more fold in chunks, in parallel. That gives the very
/// same boxes, because `Aabb::union` keeps, on each axis, the last of the values that tie
/// for the minimum (or the maximum), however the fold is grouped. This holds for every value
/// except NaN, which the caller rules out first.
fn build_node(items: &mut [BuildItem], lo: usize, below: &mut [BvhNode], base: u32, counts: &NodeCounts, par: usize) -> BvhNode {
    let n = items.len();
    let (bounds, cb) = if n >= par {
        let parts: Vec<(Aabb, Aabb)> = items.par_chunks((par / 4).max(1)).map(fold_items).collect();
        parts.iter().fold((Aabb::EMPTY, Aabb::EMPTY), |(b, c), (pb, pc)| (b.union(pb), c.union(pc)))
    } else {
        fold_items(items)
    };
    if n <= LEAF_SIZE {
        return BvhNode { bounds, start: lo as u32, count: n as u32 };
    }
    let ext = cb.extents();
    let axis = if ext.x >= ext.y && ext.x >= ext.z { 0 } else if ext.y >= ext.z { 1 } else { 2 };
    let mid = n / 2;
    items.select_nth_unstable_by(mid, |a, b| a.centroid[axis].partial_cmp(&b.centroid[axis]).unwrap_or(std::cmp::Ordering::Equal));
    let (left, right) = items.split_at_mut(mid);
    let right_below = counts.get(n - mid) - 1;
    let (kids, rest) = below.split_at_mut(2);
    let (rest_right, rest_left) = rest.split_at_mut(right_below);
    let (base_right, base_left) = (base + 2, base + 2 + right_below as u32);
    let (l, r) = if n >= par {
        rayon::join(
            || build_node(left, lo, rest_left, base_left, counts, par),
            || build_node(right, lo + mid, rest_right, base_right, counts, par),
        )
    } else {
        (build_node(left, lo, rest_left, base_left, counts, par), build_node(right, lo + mid, rest_right, base_right, counts, par))
    };
    kids[0] = l;
    kids[1] = r;
    BvhNode { bounds, start: base, count: 0 }
}

/// The box around the corners of `tris`, in order. Always inlined: as a call it made the
/// serial refit a fifth slower.
#[inline(always)]
fn leaf_bounds(mesh: &Mesh, tris: &[u32]) -> Aabb {
    let mut b = Aabb::EMPTY;
    for &t in tris {
        for p in mesh.tri_positions(t) {
            b.include(p);
        }
    }
    b
}

/// Refits the two children at `below[0]` and `below[1]` (indices `base` and `base + 1`) and
/// everything under them, which `below` holds as the builders lay it out: the two children,
/// then the right child's descendants, then the left child's. Subtrees of `split` nodes or
/// more split across threads. Each box comes out as in the serial refit (a leaf from its
/// corners, an interior node as the union of its children's boxes), only the order in which
/// nodes are reached differs. Returns false if the tree is not laid out that way, or deeper
/// than any the builders make (`depth` levels are left to go), with the boxes partly redone.
fn refit_below(below: &mut [BvhNode], base: u32, mesh: &Mesh, tris: &[u32], depth: u32, split: usize) -> bool {
    if below.len() < 2 || depth == 0 {
        return false;
    }
    let (kids, rest) = below.split_at_mut(2);
    let (l, r) = (kids[0], kids[1]);
    let first = base as usize + 2;
    // How many of `rest` are the right child's descendants; the left child's follow.
    let right = if l.is_leaf() { rest.len() } else { (l.start as usize).wrapping_sub(first) };
    let laid_out = right <= rest.len()
        && if r.is_leaf() { right == 0 } else { r.start as usize == first && right >= 2 }
        && (l.is_leaf() || rest.len() - right >= 2);
    if !laid_out {
        return false;
    }
    let big = rest.len() >= split;
    let (rest_r, rest_l) = rest.split_at_mut(right);
    let fit = |n: BvhNode, below: &mut [BvhNode], base: usize| -> Option<Aabb> {
        if n.is_leaf() {
            Some(leaf_bounds(mesh, &tris[n.start as usize..(n.start + n.count) as usize]))
        } else if refit_below(below, base as u32, mesh, tris, depth - 1, split) {
            Some(below[0].bounds.union(&below[1].bounds))
        } else {
            None
        }
    };
    let (bl, br) = if big {
        rayon::join(|| fit(l, rest_l, first + right), || fit(r, rest_r, first))
    } else {
        (fit(l, rest_l, first + right), fit(r, rest_r, first))
    };
    match (bl, br) {
        (Some(bl), Some(br)) => {
            kids[0].bounds = bl;
            kids[1].bounds = br;
            true
        }
        _ => false,
    }
}

impl Bvh {
    /// Build over all triangles of the mesh.
    pub fn build(mesh: &Mesh) -> Self {
        let tris: Vec<u32> = mesh.triangle_ids().collect();
        Self::build_subset(mesh, &tris)
    }

    /// Build over a subset of triangles.
    ///
    /// Large builds run on the thread pool and give the same tree, bit for bit, as the
    /// serial build.
    pub fn build_subset(mesh: &Mesh, tids: &[u32]) -> Self {
        Self::build_subset_with(mesh, tids, PAR_BUILD, PAR_NODE)
    }

    /// [`Bvh::build_subset`] with its thresholds: parallel from `par_build` triangles on, and
    /// within that, parallel nodes from `par_node` triangles on.
    fn build_subset_with(mesh: &Mesh, tids: &[u32], par_build: usize, par_node: usize) -> Self {
        let item = |&t: &u32| {
            let b = Aabb::from_points(mesh.tri_positions(t));
            BuildItem { tid: t, centroid: b.center(), bounds: b }
        };
        let mut bvh = Bvh {
            nodes: Vec::new(),
            tris: Vec::new(),
            shape_version: mesh.shape_version(),
            topo_version: mesh.topology_version(),
            links: None,
            built_nodes: 0,
        };
        if tids.len() < par_build.max(1) {
            bvh.build_serial(tids.iter().map(item).collect());
            return bvh;
        }
        // The whole build is one task for the pool. Called from outside it (the UI thread),
        // each parallel step would otherwise hand its work over and wait for a pool thread to
        // wake; inside, the steps share out their work without waiting.
        rayon::scope(|_| {
            // A NaN would make the folds depend on their grouping (see build_node), so a mesh
            // with NaN coordinates takes the serial path.
            let nan = AtomicBool::new(false);
            let mut items: Vec<BuildItem> = tids
                .par_iter()
                .map(|t| {
                    let it = item(t);
                    if it.bounds.min.is_nan() || it.bounds.max.is_nan() || it.centroid.is_nan() {
                        nan.store(true, Ordering::Relaxed);
                    }
                    it
                })
                .collect();
            if nan.into_inner() {
                bvh.build_serial(items);
                return;
            }
            let counts = NodeCounts::new(items.len());
            let mut nodes = vec![BvhNode { bounds: Aabb::EMPTY, start: 0, count: 0 }; counts.get(items.len())];
            let (root, below) = nodes.split_at_mut(1);
            root[0] = build_node(&mut items, 0, below, 1, &counts, par_node);
            bvh.nodes = nodes;
            bvh.tris = items.par_iter().map(|it| it.tid).collect();
            bvh.built_nodes = bvh.nodes.len();
        });
        bvh
    }

    /// The serial build, with an explicit stack of (node index, range).
    fn build_serial(&mut self, mut items: Vec<BuildItem>) {
        self.nodes = Vec::with_capacity(2 * items.len() / LEAF_SIZE + 1);
        self.built_nodes = 0;
        if items.is_empty() {
            return;
        }
        self.nodes.push(BvhNode { bounds: Aabb::EMPTY, start: 0, count: 0 });
        let mut stack = vec![(0usize, 0usize, items.len())];
        while let Some((ni, lo, hi)) = stack.pop() {
            let slice = &mut items[lo..hi];
            let (bounds, cb) = fold_items(slice);
            let n = hi - lo;
            if n <= LEAF_SIZE {
                self.nodes[ni] = BvhNode { bounds, start: lo as u32, count: n as u32 };
                continue;
            }
            let ext = cb.extents();
            let axis = if ext.x >= ext.y && ext.x >= ext.z { 0 } else if ext.y >= ext.z { 1 } else { 2 };
            let mid = n / 2;
            slice.select_nth_unstable_by(mid, |a, b| a.centroid[axis].partial_cmp(&b.centroid[axis]).unwrap_or(std::cmp::Ordering::Equal));
            let left = self.nodes.len();
            self.nodes.push(BvhNode { bounds: Aabb::EMPTY, start: 0, count: 0 });
            self.nodes.push(BvhNode { bounds: Aabb::EMPTY, start: 0, count: 0 });
            self.nodes[ni] = BvhNode { bounds, start: left as u32, count: 0 };
            stack.push((left, lo, lo + mid));
            stack.push((left + 1, lo + mid, hi));
        }
        self.tris = items.iter().map(|it| it.tid).collect();
        self.built_nodes = self.nodes.len();
    }

    /// True if queries can use an [`ArrayStack`]: the tree is a builder's (as far as can be
    /// told: its node count is still the one the builder gave it).
    #[inline]
    fn fits_array_stack(&self) -> bool {
        self.built_nodes == self.nodes.len()
    }

    pub fn shape_version(&self) -> u64 {
        self.shape_version
    }
    pub fn is_current(&self, mesh: &Mesh) -> bool {
        self.shape_version == mesh.shape_version()
    }

    /// True if only vertex positions changed since the build, so [`Bvh::refit`] suffices.
    pub fn can_refit(&self, mesh: &Mesh) -> bool {
        self.topo_version == mesh.topology_version()
    }

    /// Recompute node bounds after vertices moved (topology unchanged). Large trees split
    /// the work across threads, with the same result.
    pub fn refit(&mut self, mesh: &Mesh) {
        self.refit_with(mesh, PAR_REFIT, PAR_REFIT_SPLIT);
    }

    /// [`Bvh::refit`] with its thresholds: parallel for trees of `par_refit` nodes or more,
    /// splitting subtrees of `split` nodes or more.
    fn refit_with(&mut self, mesh: &Mesh, par_refit: usize, split: usize) {
        if !(self.nodes.len() >= par_refit && self.refit_parallel(mesh, split)) {
            self.refit_sweep(mesh);
        }
        self.shape_version = mesh.shape_version();
    }

    /// The serial refit. Children always have larger indices than their parent, so a reverse
    /// sweep is bottom-up.
    fn refit_sweep(&mut self, mesh: &Mesh) {
        for i in (0..self.nodes.len()).rev() {
            let n = self.nodes[i];
            let b = if n.is_leaf() {
                leaf_bounds(mesh, &self.tris[n.start as usize..(n.start + n.count) as usize])
            } else {
                self.nodes[n.start as usize].bounds.union(&self.nodes[n.start as usize + 1].bounds)
            };
            self.nodes[i].bounds = b;
        }
    }

    /// The refit as a split recursion over the builders' layout (see `refit_below`). False if
    /// the tree is laid out otherwise; the serial sweep then redoes every box.
    fn refit_parallel(&mut self, mesh: &Mesh, split: usize) -> bool {
        let Bvh { nodes, tris, .. } = self;
        let Some(root) = nodes.first().copied() else { return true };
        if root.is_leaf() || root.start != 1 {
            return false;
        }
        let (head, below) = nodes.split_at_mut(1);
        if !refit_below(below, 1, mesh, tris, 64, split) {
            return false;
        }
        head[0].bounds = below[0].bounds.union(&below[1].bounds);
        true
    }

    /// Refit after only `verts` moved: recomputes the leaves that hold their triangles and
    /// those leaves' ancestors, and leaves the rest of the tree alone. Gives what
    /// [`Bvh::refit`] gives, provided the tree was current before the moves, the topology is
    /// unchanged ([`Bvh::can_refit`]) and `verts` holds every vertex moved since (repeats and
    /// dead vertices are fine). The caller keeps that chain; nothing here can check it.
    ///
    /// The first call on a tree from a build refits it whole. Where +0.0 and -0.0 tie for an
    /// extreme, or a NaN is in play, a build's boxes can differ in those bits from a refit's;
    /// once every box is a refit's, redoing only the changed ones gives what a full refit
    /// gives, bit for bit, as the chain of full refits did before.
    pub fn refit_vertices(&mut self, mesh: &Mesh, verts: &[u32]) {
        debug_assert!(self.can_refit(mesh), "refit_vertices after a topology change");
        if self.nodes.is_empty() {
            self.shape_version = mesh.shape_version();
            return;
        }
        let Some(links) = self.current_links() else {
            self.make_refit_links();
            self.refit(mesh);
            return;
        };
        if links.repeats {
            self.refit(mesh);
            return;
        }
        let mut dirty: Vec<u32> = Vec::with_capacity(verts.len() * 2);
        for &v in verts {
            if !mesh.is_vertex(v) {
                continue;
            }
            for &e in mesh.vertex_edges(v) {
                for t in mesh.edge_triangles(e) {
                    if let Some(&leaf) = links.leaf_of.get(t as usize) {
                        if leaf != INVALID {
                            dirty.push(leaf);
                        }
                    }
                }
            }
        }
        dirty.sort_unstable();
        dirty.dedup();
        // A stroke over much of the mesh: the full refit gives the same, sooner.
        if dirty.len() > self.nodes.len() / 16 {
            self.refit(mesh);
            return;
        }
        let mut up = std::collections::BinaryHeap::with_capacity(dirty.len());
        for &leaf in &dirty {
            let n = self.nodes[leaf as usize];
            self.nodes[leaf as usize].bounds = leaf_bounds(mesh, &self.tris[n.start as usize..(n.start + n.count) as usize]);
            up.push(links.parent[leaf as usize]);
        }
        // Highest index first. A node's children have larger indices, so when a node comes
        // up, every changed node below it is already redone (and pushed it again, hence the
        // repeats to skip).
        let mut last = INVALID;
        while let Some(i) = up.pop() {
            if i == INVALID || i == last {
                continue;
            }
            last = i;
            let n = self.nodes[i as usize];
            self.nodes[i as usize].bounds = self.nodes[n.start as usize].bounds.union(&self.nodes[n.start as usize + 1].bounds);
            up.push(links.parent[i as usize]);
        }
        self.shape_version = mesh.shape_version();
    }

    /// The links for [`Bvh::refit_vertices`], if the tree has them for its present nodes.
    fn current_links(&self) -> Option<Arc<RefitLinks>> {
        self.links.as_ref().filter(|links| links.parent.len() == self.nodes.len()).cloned()
    }

    /// Makes the links for [`Bvh::refit_vertices`].
    fn make_refit_links(&mut self) {
        let max_tid = self.tris.iter().max().map_or(0, |&t| t as usize + 1);
        let mut leaf_of = vec![INVALID; max_tid];
        let mut parent = vec![INVALID; self.nodes.len()];
        let mut repeats = false;
        for (i, n) in self.nodes.iter().enumerate() {
            if n.is_leaf() {
                for &t in self.leaf_tris(n) {
                    repeats |= std::mem::replace(&mut leaf_of[t as usize], i as u32) != INVALID;
                }
            } else {
                parent[n.start as usize] = i as u32;
                parent[n.start as usize + 1] = i as u32;
            }
        }
        self.links = Some(Arc::new(RefitLinks { leaf_of, parent, repeats }));
    }
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
    pub fn bounds(&self) -> Aabb {
        self.nodes.first().map(|n| n.bounds).unwrap_or(Aabb::EMPTY)
    }
    #[inline]
    pub fn leaf_tris(&self, node: &BvhNode) -> &[u32] {
        &self.tris[node.start as usize..(node.start + node.count) as usize]
    }

    /// Closest ray hit, optionally filtered.
    pub fn raycast_filtered(&self, mesh: &Mesh, ray: &Ray, max_t: f64, filter: impl Fn(u32) -> bool) -> Option<RayHit> {
        if self.fits_array_stack() {
            self.raycast_filtered_in::<ArrayStack>(mesh, ray, max_t, filter)
        } else {
            self.raycast_filtered_in::<Vec<u32>>(mesh, ray, max_t, filter)
        }
    }

    fn raycast_filtered_in<S: NodeStack>(&self, mesh: &Mesh, ray: &Ray, max_t: f64, filter: impl Fn(u32) -> bool) -> Option<RayHit> {
        if self.nodes.is_empty() {
            return None;
        }
        let inv = DVec3::ONE / ray.dir;
        let mut best: Option<RayHit> = None;
        let mut best_t = max_t;
        let mut stack = S::root();
        while let Some(ni) = stack.pop() {
            let node = &self.nodes[ni as usize];
            if node.bounds.ray_hit(ray.origin, inv, best_t).is_none() {
                continue;
            }
            if node.is_leaf() {
                for &t in self.leaf_tris(node) {
                    if !filter(t) {
                        continue;
                    }
                    let [a, b, c] = mesh.tri_positions(t);
                    if let Some((th, u, v)) = ray_triangle(ray, a, b, c) {
                        if th < best_t {
                            best_t = th;
                            best = Some(RayHit { tid: t, t: th, point: ray.at(th), bary: DVec3::new(1.0 - u - v, u, v) });
                        }
                    }
                }
            } else {
                let l = node.start;
                let r = l + 1;
                let dl = self.nodes[l as usize].bounds.ray_hit(ray.origin, inv, best_t);
                let dr = self.nodes[r as usize].bounds.ray_hit(ray.origin, inv, best_t);
                match (dl, dr) {
                    (Some(a), Some(b)) => {
                        if a < b {
                            stack.push(r);
                            stack.push(l);
                        } else {
                            stack.push(l);
                            stack.push(r);
                        }
                    }
                    (Some(_), None) => stack.push(l),
                    (None, Some(_)) => stack.push(r),
                    _ => {}
                }
            }
        }
        best
    }

    /// Closest hit with `t` in `[t_min, t_max]`.
    pub fn raycast_range(&self, mesh: &Mesh, ray: &Ray, t_min: f64, t_max: f64, filter: impl Fn(u32) -> bool) -> Option<RayHit> {
        let shifted = Ray { origin: ray.at(t_min), dir: ray.dir };
        self.raycast_filtered(mesh, &shifted, t_max - t_min, filter).map(|mut h| {
            h.t += t_min;
            h
        })
    }

    pub fn raycast(&self, mesh: &Mesh, ray: &Ray) -> Option<RayHit> {
        self.raycast_filtered(mesh, ray, f64::INFINITY, |_| true)
    }

    /// All ray/triangle intersections (unsorted), used for parity tests.
    pub fn raycast_all(&self, mesh: &Mesh, ray: &Ray, out: &mut Vec<RayHit>) {
        if self.fits_array_stack() {
            self.raycast_all_in::<ArrayStack>(mesh, ray, out)
        } else {
            self.raycast_all_in::<Vec<u32>>(mesh, ray, out)
        }
    }

    fn raycast_all_in<S: NodeStack>(&self, mesh: &Mesh, ray: &Ray, out: &mut Vec<RayHit>) {
        out.clear();
        if self.nodes.is_empty() {
            return;
        }
        let inv = DVec3::ONE / ray.dir;
        let mut stack = S::root();
        while let Some(ni) = stack.pop() {
            let node = &self.nodes[ni as usize];
            if node.bounds.ray_hit(ray.origin, inv, f64::INFINITY).is_none() {
                continue;
            }
            if node.is_leaf() {
                for &t in self.leaf_tris(node) {
                    let [a, b, c] = mesh.tri_positions(t);
                    if let Some((th, u, v)) = ray_triangle(ray, a, b, c) {
                        out.push(RayHit { tid: t, t: th, point: ray.at(th), bary: DVec3::new(1.0 - u - v, u, v) });
                    }
                }
            } else {
                stack.push(node.start);
                stack.push(node.start + 1);
            }
        }
    }

    /// Nearest point on the mesh to `p`, searching within `max_dist`.
    pub fn nearest(&self, mesh: &Mesh, p: DVec3, max_dist: f64) -> Option<NearestHit> {
        if self.fits_array_stack() {
            self.nearest_in::<ArrayStack>(mesh, p, max_dist)
        } else {
            self.nearest_in::<Vec<u32>>(mesh, p, max_dist)
        }
    }

    fn nearest_in<S: NodeStack>(&self, mesh: &Mesh, p: DVec3, max_dist: f64) -> Option<NearestHit> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut best_d2 = if max_dist.is_finite() { max_dist * max_dist } else { f64::INFINITY };
        let mut best: Option<NearestHit> = None;
        let mut stack = S::root();
        while let Some(ni) = stack.pop() {
            let node = &self.nodes[ni as usize];
            if node.bounds.distance_squared(p) > best_d2 {
                continue;
            }
            if node.is_leaf() {
                for &t in self.leaf_tris(node) {
                    let [a, b, c] = mesh.tri_positions(t);
                    let (q, bary) = closest_point_on_triangle(p, a, b, c);
                    let d2 = q.distance_squared(p);
                    if d2 < best_d2 {
                        best_d2 = d2;
                        best = Some(NearestHit { tid: t, point: q, dist_sq: d2, bary });
                    }
                }
            } else {
                let l = node.start;
                let r = l + 1;
                let dl = self.nodes[l as usize].bounds.distance_squared(p);
                let dr = self.nodes[r as usize].bounds.distance_squared(p);
                if dl < dr {
                    stack.push(r);
                    stack.push(l);
                } else {
                    stack.push(l);
                    stack.push(r);
                }
            }
        }
        best
    }

    /// Triangles whose bounding boxes intersect `query`.
    pub fn query_aabb(&self, query: &Aabb, f: impl FnMut(u32)) {
        if self.fits_array_stack() {
            self.query_aabb_in::<ArrayStack>(query, f)
        } else {
            self.query_aabb_in::<Vec<u32>>(query, f)
        }
    }

    fn query_aabb_in<S: NodeStack>(&self, query: &Aabb, mut f: impl FnMut(u32)) {
        if self.nodes.is_empty() {
            return;
        }
        let mut stack = S::root();
        while let Some(ni) = stack.pop() {
            let node = &self.nodes[ni as usize];
            if !node.bounds.intersects(query) {
                continue;
            }
            if node.is_leaf() {
                for &t in self.leaf_tris(node) {
                    f(t);
                }
            } else {
                stack.push(node.start);
                stack.push(node.start + 1);
            }
        }
    }

    /// Triangles that come within distance `r` of `center`.
    pub fn query_sphere(&self, mesh: &Mesh, center: DVec3, r: f64, mut f: impl FnMut(u32)) {
        let r2 = r * r;
        let q = Aabb { min: center - DVec3::splat(r), max: center + DVec3::splat(r) };
        self.query_aabb(&q, |t| {
            let [a, b, c] = mesh.tri_positions(t);
            let (p, _) = closest_point_on_triangle(center, a, b, c);
            if p.distance_squared(center) <= r2 {
                f(t);
            }
        });
    }

    /// Point-in-solid test by ray parity, voting over three axis-aligned rays to resist
    /// grazing hits and small holes.
    pub fn is_inside(&self, mesh: &Mesh, p: DVec3) -> bool {
        let dirs = [
            DVec3::new(0.5773, 0.5774, 0.5775),
            DVec3::new(-0.6123, 0.3536, -0.7072),
            DVec3::new(0.1, -0.9899, 0.1001),
        ];
        let mut votes = 0;
        let mut hits = Vec::new();
        for d in dirs {
            self.raycast_all(mesh, &Ray::new(p, d), &mut hits);
            if hits.len() % 2 == 1 {
                votes += 1;
            }
        }
        votes >= 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;

    /// Tiny deterministic PRNG so tests are reproducible without extra dependencies.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            self.0 >> 33
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
        fn unit(&mut self) -> f64 {
            self.next() as f64 / (1u64 << 31) as f64 - 0.5
        }
    }

    fn bits(b: &Aabb) -> [u64; 6] {
        let (lo, hi) = (b.min.to_array(), b.max.to_array());
        [lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]].map(f64::to_bits)
    }

    /// Same nodes (boxes bit for bit), same triangle order, same versions.
    fn assert_same_tree(a: &Bvh, b: &Bvh, what: &str) {
        assert_eq!(a.nodes.len(), b.nodes.len(), "{what}: node count");
        for (i, (x, y)) in a.nodes.iter().zip(&b.nodes).enumerate() {
            assert_eq!((x.start, x.count), (y.start, y.count), "{what}: node {i}");
            assert_eq!(bits(&x.bounds), bits(&y.bounds), "{what}: box of node {i}");
        }
        assert_eq!(a.tris, b.tris, "{what}: triangle order");
        assert_eq!((a.shape_version, a.topo_version), (b.shape_version, b.topo_version), "{what}: versions");
    }

    /// The layout the refit sweep and other bottom-up passes rely on.
    fn assert_layout(b: &Bvh, n_tris: usize) {
        let mut seen = vec![false; b.tris.len()];
        for (i, n) in b.nodes.iter().enumerate() {
            if n.is_leaf() {
                assert!(n.count as usize <= LEAF_SIZE);
                for s in n.start..n.start + n.count {
                    assert!(!std::mem::replace(&mut seen[s as usize], true), "slot {s} in two leaves");
                }
            } else {
                assert!(n.start as usize > i && (n.start as usize + 1) < b.nodes.len(), "children after node {i}");
            }
        }
        assert!(seen.iter().all(|&s| s));
        assert_eq!(b.tris.len(), n_tris);
    }

    /// The serial refit as it was: one reverse sweep over leaves and interior alike.
    fn refit_serial(b: &mut Bvh, mesh: &Mesh) {
        for i in (0..b.nodes.len()).rev() {
            let n = b.nodes[i];
            b.nodes[i].bounds = if n.is_leaf() {
                leaf_bounds(mesh, &b.tris[n.start as usize..(n.start + n.count) as usize])
            } else {
                b.nodes[n.start as usize].bounds.union(&b.nodes[n.start as usize + 1].bounds)
            };
        }
        b.shape_version = mesh.shape_version();
    }

    fn noisy(mut m: Mesh, amp: f64, seed: u64) -> Mesh {
        let mut rng = Lcg(seed);
        for p in m.positions_mut() {
            *p += DVec3::new(rng.unit(), rng.unit(), rng.unit()) * amp;
        }
        m.bump_shape_version();
        m
    }

    /// A flat grid where z is +0.0 at some corners and -0.0 at others, so both ends of the z
    /// range tie between the two zeros; plus x and y values that repeat across rows.
    fn signed_zero_grid() -> Mesh {
        let mut m = primitives::plane_grid(2.0, 1.0, 120, 70);
        for (i, p) in m.positions_mut().iter_mut().enumerate() {
            if i % 3 == 1 {
                p.z = -0.0;
            }
        }
        m.bump_shape_version();
        m
    }

    fn test_meshes() -> Vec<(&'static str, Mesh)> {
        let mut nan = noisy(primitives::icosphere(1.0, 4), 0.01, 5);
        nan.set_vertex(17, DVec3::new(f64::NAN, 0.2, 0.3));
        nan.set_vertex(301, DVec3::new(0.1, f64::NAN, f64::NAN));
        vec![
            ("signed-zero grid", signed_zero_grid()),
            ("icosphere", primitives::icosphere(1.0, 5)),
            ("noisy icosphere", noisy(primitives::icosphere(1.0, 4), 0.01, 3)),
            ("cube", primitives::cube(2.0, 30)),
            ("open cylinder", noisy(primitives::cylinder(0.5, 3.0, 64, 40, false), 0.003, 9)),
            ("NaN corners", nan),
        ]
    }

    #[test]
    fn parallel_build_matches_serial() {
        for (name, m) in test_meshes() {
            let all: Vec<u32> = m.triangle_ids().collect();
            // A subset, in a scrambled order: the order of the ids decides the item order.
            let mut rng = Lcg(11);
            let mut some: Vec<u32> = all.iter().copied().filter(|t| t % 5 != 2).collect();
            for i in (1..some.len()).rev() {
                some.swap(i, rng.below(i + 1));
            }
            for (what, tids) in [("all", &all), ("subset", &some)] {
                let serial = Bvh::build_subset_with(&m, tids, usize::MAX, usize::MAX);
                assert_layout(&serial, tids.len());
                // Parallel all the way down to nodes of 64 items and folds of 16.
                for par_node in [64, 1000, 1 << 20] {
                    let par = Bvh::build_subset_with(&m, tids, 0, par_node);
                    assert_same_tree(&par, &serial, &format!("{name} {what} par_node {par_node}"));
                }
                assert_same_tree(&Bvh::build_subset(&m, tids), &serial, &format!("{name} {what} default"));
                let built = Bvh::build_subset(&m, tids);
                assert!(built.is_current(&m) && built.can_refit(&m));
            }
        }
        // Tiny and empty inputs.
        let m = primitives::icosphere(1.0, 1);
        for k in [0, 1, 3, 4, 5, 9, 17] {
            let tids: Vec<u32> = m.triangle_ids().take(k).collect();
            let serial = Bvh::build_subset_with(&m, &tids, usize::MAX, usize::MAX);
            assert_same_tree(&Bvh::build_subset_with(&m, &tids, 0, 2), &serial, &format!("{k} triangles"));
            assert_eq!(serial.nodes.is_empty(), k == 0);
        }
    }

    #[test]
    fn large_parallel_build_matches_serial() {
        // Above the default thresholds, so the real parallel path runs with its own sizes.
        let m = noisy(primitives::icosphere(1.0, 7), 0.001, 21);
        assert!(m.triangle_count() >= PAR_BUILD && m.triangle_count() >= 4 * PAR_NODE);
        let tids: Vec<u32> = m.triangle_ids().collect();
        let serial = Bvh::build_subset_with(&m, &tids, usize::MAX, usize::MAX);
        assert_same_tree(&Bvh::build(&m), &serial, "noisy icosphere 7");
        // Tied centroids, with the default node size (the grid is under the build size).
        let grid = primitives::plane_grid(4.0, 4.0, 200, 200);
        let tids: Vec<u32> = grid.triangle_ids().collect();
        let serial = Bvh::build_subset_with(&grid, &tids, usize::MAX, usize::MAX);
        assert_same_tree(&Bvh::build_subset_with(&grid, &tids, 0, PAR_NODE), &serial, "grid 200");
        assert_same_tree(&Bvh::build(&grid), &serial, "grid 200 default");
    }

    #[test]
    fn large_parallel_refit_matches_serial() {
        // Split as the default refit splits large trees (this one is under its size).
        let m0 = noisy(primitives::icosphere(1.0, 7), 0.0005, 4);
        let tree = Bvh::build(&m0);
        assert!(tree.nodes.len() >= 4 * PAR_REFIT_SPLIT);
        assert!(tree.clone().refit_parallel(&m0, PAR_REFIT_SPLIT));
        let mut m = m0.clone();
        let mut rng = Lcg(19);
        for _ in 0..m.max_vid() / 4 {
            let v = rng.below(m.max_vid()) as u32;
            let p = m.vertex(v) + DVec3::new(rng.unit(), rng.unit(), rng.unit()) * 0.01;
            m.set_vertex(v, p);
        }
        m.set_vertex(5, DVec3::new(-0.0, 0.0, f64::NAN));
        let mut a = tree.clone();
        a.refit_with(&m, 0, PAR_REFIT_SPLIT);
        let mut b = tree.clone();
        refit_serial(&mut b, &m);
        assert_same_tree(&a, &b, "noisy icosphere 7");
    }

    #[test]
    fn parallel_refit_matches_serial() {
        for (name, m0) in test_meshes() {
            let tree = Bvh::build(&m0);
            let mut m = m0.clone();
            let mut rng = Lcg(5);
            for round in 0..4 {
                for _ in 0..m.max_vid() / 3 {
                    let v = rng.below(m.max_vid()) as u32;
                    let p = m.vertex(v) + DVec3::new(rng.unit(), rng.unit(), rng.unit()) * 0.2;
                    m.set_vertex(v, p);
                }
                if round == 2 {
                    m.set_vertex(3, DVec3::new(-0.0, f64::NAN, 0.0));
                }
                let mut b = tree.clone();
                refit_serial(&mut b, &m);
                // Parallel down to subtrees of 64 nodes, then with larger splits, then with
                // the defaults (which these small trees refit serially).
                for split in [64, 1000, 1 << 20] {
                    let mut a = tree.clone();
                    a.refit_with(&m, 0, split);
                    assert_same_tree(&a, &b, &format!("{name} round {round} split {split}"));
                    assert!(a.is_current(&m));
                }
                let mut a = tree.clone();
                a.refit(&m);
                assert_same_tree(&a, &b, &format!("{name} round {round} default"));
            }
        }
        let big = primitives::icosphere(1.0, 5);
        let mut tree = Bvh::build(&big);
        assert!(tree.refit_parallel(&big, 64), "the builders' layout takes the parallel path");
        // The same tree numbered breadth first: still parents before children and siblings
        // side by side, but subtrees no longer in one run, so the refit must notice and sweep.
        let mut order = vec![0usize];
        let mut k = 0;
        while k < order.len() {
            let n = tree.nodes[order[k]];
            if !n.is_leaf() {
                order.extend([n.start as usize, n.start as usize + 1]);
            }
            k += 1;
        }
        let mut new_index = vec![0u32; order.len()];
        for (new, &old) in order.iter().enumerate() {
            new_index[old] = new as u32;
        }
        tree.nodes = order
            .iter()
            .map(|&old| {
                let n = tree.nodes[old];
                if n.is_leaf() {
                    n
                } else {
                    BvhNode { start: new_index[n.start as usize], ..n }
                }
            })
            .collect();
        let mut m = big.clone();
        m.transform(|p| p * 1.5 + DVec3::X);
        let mut serial = tree.clone();
        refit_serial(&mut serial, &m);
        assert!(!tree.clone().refit_parallel(&m, 64));
        tree.refit_with(&m, 0, 64);
        assert_same_tree(&tree, &serial, "breadth-first layout");
        assert_eq!(tree.nearest(&m, DVec3::new(3.0, 0.1, 0.2), f64::INFINITY).map(|h| h.tid), Bvh::build(&m).nearest(&m, DVec3::new(3.0, 0.1, 0.2), f64::INFINITY).map(|h| h.tid));
    }

    #[test]
    fn array_and_vec_stacks_give_the_same_answers() {
        let m = noisy(primitives::icosphere(1.0, 4), 0.01, 8);
        let bvh = Bvh::build(&m);
        assert!(bvh.fits_array_stack());
        let mut rng = Lcg(3);
        let (mut a, mut b) = (Vec::new(), Vec::new());
        let key = |h: &Option<NearestHit>| h.map(|h| (h.tid, h.dist_sq.to_bits(), h.point.to_array().map(f64::to_bits)));
        let rkey = |h: &Option<RayHit>| h.map(|h| (h.tid, h.t.to_bits()));
        for _ in 0..2000 {
            let p = DVec3::new(rng.unit(), rng.unit(), rng.unit()) * 3.0;
            let d = DVec3::new(rng.unit(), rng.unit(), rng.unit());
            assert_eq!(key(&bvh.nearest_in::<ArrayStack>(&m, p, 0.7)), key(&bvh.nearest_in::<Vec<u32>>(&m, p, 0.7)));
            assert_eq!(key(&bvh.nearest(&m, p, f64::INFINITY)), key(&bvh.nearest_in::<Vec<u32>>(&m, p, f64::INFINITY)));
            let ray = Ray::new(p, d);
            let odd = |t: u32| !t.is_multiple_of(3);
            assert_eq!(rkey(&bvh.raycast_filtered_in::<ArrayStack>(&m, &ray, 5.0, odd)), rkey(&bvh.raycast_filtered_in::<Vec<u32>>(&m, &ray, 5.0, odd)));
            bvh.raycast_all_in::<ArrayStack>(&m, &ray, &mut a);
            bvh.raycast_all_in::<Vec<u32>>(&m, &ray, &mut b);
            assert_eq!(a.iter().map(|h| (h.tid, h.t.to_bits())).collect::<Vec<_>>(), b.iter().map(|h| (h.tid, h.t.to_bits())).collect::<Vec<_>>());
            let q = Aabb { min: p, max: p + DVec3::splat(0.4) };
            let (mut qa, mut qb) = (Vec::new(), Vec::new());
            bvh.query_aabb_in::<ArrayStack>(&q, |t| qa.push(t));
            bvh.query_aabb_in::<Vec<u32>>(&q, |t| qb.push(t));
            assert_eq!(qa, qb);
        }
    }

    #[test]
    fn deep_hand_made_tree_uses_a_growing_stack() {
        // A chain 150 levels deep (each interior node: a leaf and the rest), far deeper than
        // any builder's tree; queries must still work, on a stack that grows.
        let m = primitives::plane_grid(15.0, 10.0, 15, 10);
        let tids: Vec<u32> = m.triangle_ids().take(150).collect();
        let mut bvh = Bvh::build_subset(&m, &tids);
        let mut nodes = Vec::new();
        for (k, _) in tids.iter().enumerate() {
            let i = nodes.len();
            if k + 1 == tids.len() {
                nodes.push(BvhNode { bounds: Aabb::EMPTY, start: k as u32, count: 1 });
            } else {
                nodes.push(BvhNode { bounds: Aabb::EMPTY, start: i as u32 + 1, count: 0 });
                nodes.push(BvhNode { bounds: Aabb::EMPTY, start: k as u32, count: 1 });
            }
        }
        bvh.nodes = nodes;
        bvh.tris = tids.clone();
        assert!(!bvh.fits_array_stack());
        bvh.refit(&m);
        let mut all = Vec::new();
        bvh.query_aabb(&m.bounds(), |t| all.push(t));
        all.sort_unstable();
        assert_eq!(all, tids);
        let far = DVec3::new(100.0, 100.0, 1.0);
        let want = tids.iter().map(|&t| m.tri_centroid(t).distance(far)).fold(f64::INFINITY, f64::min);
        assert!(bvh.nearest(&m, far, f64::INFINITY).unwrap().point.distance(far) <= want);
        let hit = bvh.raycast(&m, &Ray::new(m.tri_centroid(tids[140]) + DVec3::Z, -DVec3::Z)).unwrap();
        assert_eq!(hit.tid, tids[140]);
    }

    #[test]
    fn refit_vertices_matches_full_refit() {
        for (name, m0) in test_meshes() {
            for (what, ids) in [("all", 0), ("subset", 1), ("repeated ids", 2)] {
                // From a build, as the scene has it, and from a tree a full refit left.
                for from_refit in [false, true] {
                    let mut m = m0.clone();
                    let mut tids: Vec<u32> = m.triangle_ids().filter(|t| ids != 1 || t % 3 != 0).collect();
                    if ids == 2 {
                        // A subset build given some ids twice: those triangles sit in two
                        // leaves, and both must follow their corners.
                        tids.extend(m.triangle_ids().step_by(7).take(40));
                    }
                    let mut inc = Bvh::build_subset(&m, &tids);
                    if from_refit {
                        inc.refit(&m);
                    }
                    // The chain of full refits each step, as the scene did before.
                    let mut full = inc.clone();
                    let mut rng = Lcg(77);
                    for step in 0..60 {
                        // A dab: the vertices near one, some listed twice, plus a dead id.
                        let c = m.vertex(rng.below(m.max_vid()) as u32);
                        let r = if step % 10 == 9 { 10.0 } else { 0.05 + 0.1 * rng.unit().abs() };
                        let mut moved: Vec<u32> = m.vertex_ids().filter(|&v| m.vertex(v).distance(c) < r).collect();
                        let dirs: Vec<DVec3> = moved.iter().map(|_| DVec3::new(rng.unit(), rng.unit(), rng.unit()) * 0.02).collect();
                        let pos = m.positions_mut();
                        for (&v, d) in moved.iter().zip(&dirs) {
                            pos[v as usize] += *d;
                        }
                        if step % 7 == 3 {
                            // Exact zeros and a NaN, so ties and NaN pass through the leaves too.
                            if let Some(&v) = moved.first() {
                                pos[v as usize].z = if step % 2 == 0 { -0.0 } else { f64::NAN };
                            }
                        }
                        m.bump_shape_version();
                        if moved.len() > 2 {
                            moved.push(moved[1]);
                        }
                        moved.push(u32::MAX - 1);
                        inc.refit_vertices(&m, &moved);
                        full.refit(&m);
                        let label = format!("{name} {what} from refit {from_refit} step {step}");
                        assert_same_tree(&inc, &full, &label);
                        assert!(inc.is_current(&m));
                        assert_eq!(inc.links.as_ref().map(|l| l.repeats), Some(ids == 2), "{label}");
                    }
                }
            }
        }
        // A copy taken before the first refit keeps its own boxes, and has no links.
        let mut m = noisy(primitives::icosphere(1.0, 4), 0.01, 3);
        let mut inc = Bvh::build(&m);
        let keep = inc.clone();
        let moved: Vec<u32> = (0..m.max_vid() as u32).step_by(97).collect();
        for &v in &moved {
            m.set_vertex(v, m.vertex(v) * 1.3);
        }
        inc.refit_vertices(&m, &moved);
        let mut full = keep.clone();
        full.refit(&m);
        assert_same_tree(&inc, &full, "from a build");
        assert!(!keep.is_current(&m));
        assert!(keep.links.is_none() && inc.links.is_some());
        // Copies made after it share the links, and refit incrementally from there.
        let mut copy = inc.clone();
        assert!(Arc::ptr_eq(copy.links.as_ref().unwrap(), inc.links.as_ref().unwrap()));
        m.set_vertex(10, m.vertex(10) * 0.9);
        copy.refit_vertices(&m, &[10]);
        full.refit(&m);
        assert_same_tree(&copy, &full, "a copy");
    }

    #[test]
    fn first_refit_vertices_matches_full_refits_on_ties() {
        // Built boxes where +0.0 and -0.0 tie can keep the other zero than a refit's; the first
        // incremental refit redoes the whole tree, so nothing of the build's choice survives.
        let mut m = signed_zero_grid();
        let tree = Bvh::build(&m);
        let mut refit = tree.clone();
        refit.refit(&m);
        let differs = tree.nodes.iter().zip(&refit.nodes).any(|(a, b)| bits(&a.bounds) != bits(&b.bounds));
        let mut inc = tree.clone();
        let mut full = tree.clone();
        m.set_vertex(0, m.vertex(0) + DVec3::X * 0.001);
        inc.refit_vertices(&m, &[0]);
        full.refit(&m);
        assert_same_tree(&inc, &full, "signed-zero grid, first dab");
        // The case the test is about must be present: the build and the refit disagree.
        assert!(differs, "the grid no longer has signed-zero ties between build and refit");
    }
}
