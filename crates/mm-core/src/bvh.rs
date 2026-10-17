//! Bounding-volume hierarchy over the triangles of a [`Mesh`], for ray casting,
//! nearest-point and range queries.
//!
//! The tree stores triangle ids only; queries take the mesh by reference, so the caller
//! must rebuild the tree after editing the mesh (compare [`Bvh::shape_version`] with
//! [`Mesh::shape_version`]).

use crate::geom::{closest_point_on_triangle, ray_triangle, Aabb, Ray};
use crate::mesh::Mesh;
use glam::DVec3;

const LEAF_SIZE: usize = 4;

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

impl Bvh {
    /// Build over all triangles of the mesh.
    pub fn build(mesh: &Mesh) -> Self {
        let tris: Vec<u32> = mesh.triangle_ids().collect();
        Self::build_subset(mesh, &tris)
    }

    /// Build over a subset of triangles.
    pub fn build_subset(mesh: &Mesh, tids: &[u32]) -> Self {
        let mut items: Vec<BuildItem> = tids
            .iter()
            .map(|&t| {
                let p = mesh.tri_positions(t);
                let b = Aabb::from_points(p);
                BuildItem { tid: t, centroid: b.center(), bounds: b }
            })
            .collect();
        let mut bvh = Bvh {
            nodes: Vec::with_capacity(2 * items.len() / LEAF_SIZE + 1),
            tris: Vec::new(),
            shape_version: mesh.shape_version(),
            topo_version: mesh.topology_version(),
        };
        if items.is_empty() {
            return bvh;
        }
        bvh.nodes.push(BvhNode { bounds: Aabb::EMPTY, start: 0, count: 0 });
        // Iterative build with an explicit stack of (node index, range).
        let mut stack = vec![(0usize, 0usize, items.len())];
        while let Some((ni, lo, hi)) = stack.pop() {
            let slice = &mut items[lo..hi];
            let mut bounds = Aabb::EMPTY;
            let mut cb = Aabb::EMPTY;
            for it in slice.iter() {
                bounds = bounds.union(&it.bounds);
                cb.include(it.centroid);
            }
            let n = hi - lo;
            if n <= LEAF_SIZE {
                bvh.nodes[ni] = BvhNode { bounds, start: lo as u32, count: n as u32 };
                continue;
            }
            let ext = cb.extents();
            let axis = if ext.x >= ext.y && ext.x >= ext.z { 0 } else if ext.y >= ext.z { 1 } else { 2 };
            let mid = n / 2;
            slice.select_nth_unstable_by(mid, |a, b| a.centroid[axis].partial_cmp(&b.centroid[axis]).unwrap_or(std::cmp::Ordering::Equal));
            let left = bvh.nodes.len();
            bvh.nodes.push(BvhNode { bounds: Aabb::EMPTY, start: 0, count: 0 });
            bvh.nodes.push(BvhNode { bounds: Aabb::EMPTY, start: 0, count: 0 });
            bvh.nodes[ni] = BvhNode { bounds, start: left as u32, count: 0 };
            stack.push((left, lo, lo + mid));
            stack.push((left + 1, lo + mid, hi));
        }
        bvh.tris = items.iter().map(|it| it.tid).collect();
        bvh
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

    /// Recompute node bounds after vertices moved (topology unchanged). Children always
    /// have larger indices than their parent, so a reverse sweep is bottom-up.
    pub fn refit(&mut self, mesh: &Mesh) {
        for i in (0..self.nodes.len()).rev() {
            let n = self.nodes[i];
            let b = if n.is_leaf() {
                let mut b = Aabb::EMPTY;
                for &t in &self.tris[n.start as usize..(n.start + n.count) as usize] {
                    for p in mesh.tri_positions(t) {
                        b.include(p);
                    }
                }
                b
            } else {
                self.nodes[n.start as usize].bounds.union(&self.nodes[n.start as usize + 1].bounds)
            };
            self.nodes[i].bounds = b;
        }
        self.shape_version = mesh.shape_version();
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
        if self.nodes.is_empty() {
            return None;
        }
        let inv = DVec3::ONE / ray.dir;
        let mut best: Option<RayHit> = None;
        let mut best_t = max_t;
        let mut stack: Vec<u32> = Vec::with_capacity(64);
        stack.push(0);
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
        out.clear();
        if self.nodes.is_empty() {
            return;
        }
        let inv = DVec3::ONE / ray.dir;
        let mut stack: Vec<u32> = vec![0];
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
        if self.nodes.is_empty() {
            return None;
        }
        let mut best_d2 = if max_dist.is_finite() { max_dist * max_dist } else { f64::INFINITY };
        let mut best: Option<NearestHit> = None;
        let mut stack: Vec<u32> = Vec::with_capacity(64);
        stack.push(0);
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
    pub fn query_aabb(&self, query: &Aabb, mut f: impl FnMut(u32)) {
        if self.nodes.is_empty() {
            return;
        }
        let mut stack: Vec<u32> = vec![0];
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
