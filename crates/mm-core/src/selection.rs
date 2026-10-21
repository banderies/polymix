//! Triangle selections and the region operations Meshmixer exposes on them
//! (expand/contract, connected, invert, boundary optimisation, brush painting).

use crate::bvh::Bvh;
use crate::hash::{FastMap, FastSet};
use crate::mesh::{Mesh, INVALID};
use glam::DVec3;

/// A set of triangles stored as a dense bitmap over triangle ids.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TriSelection {
    bits: Vec<bool>,
    count: usize,
}

impl TriSelection {
    pub fn new(m: &Mesh) -> Self {
        Self { bits: vec![false; m.max_tid()], count: 0 }
    }
    pub fn from_tris(m: &Mesh, tris: impl IntoIterator<Item = u32>) -> Self {
        let mut s = Self::new(m);
        for t in tris {
            s.insert(t);
        }
        s
    }
    pub fn all(m: &Mesh) -> Self {
        Self::from_tris(m, m.triangle_ids())
    }
    /// Resize for a mesh whose triangle id range may have grown, dropping dead ids.
    pub fn sync(&mut self, m: &Mesh) {
        if self.count == 0 {
            // Nothing selected, so nothing to drop: only the size changes.
            self.grow_to(m.max_tid());
            return;
        }
        self.bits.resize(m.max_tid(), false);
        self.count = 0;
        for t in 0..self.bits.len() {
            if self.bits[t] && !m.is_triangle(t as u32) {
                self.bits[t] = false;
            }
            if self.bits[t] {
                self.count += 1;
            }
        }
    }

    /// Size the bitmap for `max_tid` triangle slots (usually growing it), without checking
    /// which triangles are live. For an empty selection this is all [`TriSelection::sync`]
    /// does; with triangles selected, `sync` also drops the dead ones.
    pub fn grow_to(&mut self, max_tid: usize) {
        if max_tid < self.bits.len() {
            self.count -= self.bits[max_tid..].iter().filter(|&&b| b).count();
        }
        self.bits.resize(max_tid, false);
    }
    #[inline]
    pub fn contains(&self, t: u32) -> bool {
        self.bits.get(t as usize).copied().unwrap_or(false)
    }
    #[inline]
    pub fn insert(&mut self, t: u32) -> bool {
        let i = t as usize;
        if i >= self.bits.len() {
            self.bits.resize(i + 1, false);
        }
        if !self.bits[i] {
            self.bits[i] = true;
            self.count += 1;
            true
        } else {
            false
        }
    }
    #[inline]
    pub fn remove(&mut self, t: u32) -> bool {
        let i = t as usize;
        if i < self.bits.len() && self.bits[i] {
            self.bits[i] = false;
            self.count -= 1;
            true
        } else {
            false
        }
    }
    pub fn clear(&mut self) {
        self.bits.iter_mut().for_each(|b| *b = false);
        self.count = 0;
    }
    pub fn len(&self) -> usize {
        self.count
    }
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
    pub fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        self.bits.iter().enumerate().filter(|(_, &b)| b).map(|(i, _)| i as u32)
    }
    pub fn to_vec(&self) -> Vec<u32> {
        self.iter().collect()
    }
    pub fn bits(&self) -> &[bool] {
        &self.bits
    }

    pub fn invert(&mut self, m: &Mesh) {
        self.bits.resize(m.max_tid(), false);
        self.count = 0;
        for t in 0..self.bits.len() {
            self.bits[t] = m.is_triangle(t as u32) && !self.bits[t];
            if self.bits[t] {
                self.count += 1;
            }
        }
    }

    /// Vertices touched by selected triangles.
    pub fn vertices(&self, m: &Mesh) -> Vec<u32> {
        let mut mark = vec![false; m.max_vid()];
        let mut out = Vec::new();
        for t in self.iter() {
            for v in m.triangle(t) {
                if !mark[v as usize] {
                    mark[v as usize] = true;
                    out.push(v);
                }
            }
        }
        out
    }

    /// Edges on the border of the selection (one side selected, the other not or none).
    pub fn boundary_edges(&self, m: &Mesh) -> Vec<u32> {
        let mut out = Vec::new();
        for t in self.iter() {
            for e in m.tri_edges(t) {
                let o = m.edge_other_triangle(e, t);
                if o == INVALID || !self.contains(o) {
                    out.push(e);
                }
            }
        }
        out
    }

    /// Vertices on the border between selected and unselected triangles
    /// (mesh boundary vertices of the selection included).
    pub fn boundary_vertices(&self, m: &Mesh) -> Vec<u32> {
        let mut mark = vec![false; m.max_vid()];
        let mut out = Vec::new();
        for e in self.boundary_edges(m) {
            for v in m.edge_vertices(e) {
                if !mark[v as usize] {
                    mark[v as usize] = true;
                    out.push(v);
                }
            }
        }
        out
    }

    /// Grow by one ring of vertex-adjacent triangles.
    pub fn expand(&mut self, m: &Mesh) {
        let verts = self.vertices(m);
        for v in verts {
            for t in m.vertex_triangles(v) {
                self.insert(t);
            }
        }
    }

    /// Shrink by removing triangles touching the selection border.
    pub fn contract(&mut self, m: &Mesh) {
        let border = self.boundary_vertices(m);
        let mut remove = Vec::new();
        for v in border {
            for t in m.vertex_triangles(v) {
                if self.contains(t) {
                    remove.push(t);
                }
            }
        }
        for t in remove {
            self.remove(t);
        }
    }

    /// Expand to every triangle edge-connected to the current selection.
    pub fn expand_to_connected(&mut self, m: &Mesh) {
        let seeds = self.to_vec();
        for t in crate::topology::flood_fill_triangles(m, &seeds, |_| true) {
            self.insert(t);
        }
    }

    /// Expand to whole face groups touched by the selection.
    /// Grow through edge-connected triangles that share the group of the triangle they are
    /// reached from (Meshmixer "Expand to Groups").
    pub fn expand_to_connected_groups(&mut self, m: &Mesh) {
        let mut stack: Vec<u32> = self.iter().collect();
        while let Some(t) = stack.pop() {
            let g = m.tri_group(t);
            for n in m.tri_neighbors(t) {
                if n != INVALID && !self.contains(n) && m.tri_group(n) == g {
                    self.insert(n);
                    stack.push(n);
                }
            }
        }
    }

    /// Add every triangle whose group occurs in the selection, connected or not
    /// (Meshmixer "Expand to Same Groups").
    pub fn expand_to_groups(&mut self, m: &Mesh) {
        let mut groups: Vec<i32> = self.iter().map(|t| m.tri_group(t)).collect();
        groups.sort_unstable();
        groups.dedup();
        for t in m.triangle_ids() {
            if groups.binary_search(&m.tri_group(t)).is_ok() {
                self.insert(t);
            }
        }
    }

    /// Smooth a jagged selection border: repeatedly add unselected triangles with at least
    /// two selected neighbours and drop selected triangles with at most one.
    pub fn optimize_boundary(&mut self, m: &Mesh, iterations: usize) {
        for _ in 0..iterations {
            let mut changed = false;
            let border = self.boundary_edges(m);
            let mut cand: Vec<u32> = Vec::new();
            for e in border {
                for t in m.edge_triangles(e) {
                    if t != INVALID {
                        cand.push(t);
                    }
                }
            }
            cand.sort_unstable();
            cand.dedup();
            let mut add = Vec::new();
            let mut del = Vec::new();
            for t in cand {
                let n_sel = m.tri_neighbors(t).iter().filter(|&&n| n != INVALID && self.contains(n)).count();
                if !self.contains(t) && n_sel >= 2 {
                    add.push(t);
                } else if self.contains(t) && n_sel <= 1 {
                    del.push(t);
                }
            }
            for t in add {
                changed |= self.insert(t);
            }
            for t in del {
                changed |= self.remove(t);
            }
            if !changed {
                break;
            }
        }
    }

    /// Paint (or erase, with `deselect`) one brush stamp. `allow` filters candidate
    /// triangles (back faces, interior faces); the seed triangle is always eligible.
    /// Returns the number of triangles changed.
    pub fn paint_brush(&mut self, m: &Mesh, bvh: &Bvh, stamp: &BrushStamp, allow: impl Fn(u32) -> bool, deselect: bool) -> usize {
        let mut n = 0;
        for t in brush_region(m, bvh, stamp, allow) {
            let changed = if deselect { self.remove(t) } else { self.insert(t) };
            if changed {
                n += 1;
            }
        }
        n
    }
}

/// The triangles one brush stamp covers, in the order they are found.
fn brush_region(m: &Mesh, bvh: &Bvh, stamp: &BrushStamp, allow: impl Fn(u32) -> bool) -> Vec<u32> {
    let BrushStamp { shape, seed, center, normal, radius, crease_cos } = *stamp;
    match shape {
        BrushShape::Sphere => {
            let mut out = Vec::new();
            bvh.query_sphere(m, center, radius, |t| {
                if t == seed || allow(t) {
                    out.push(t);
                }
            });
            if !out.contains(&seed) {
                out.push(seed);
            }
            out
        }
        BrushShape::SphereDisc => {
            // Connected part of the sphere within a slab around the tangent plane.
            let half = radius * 0.35;
            let mut inside = FastSet::default();
            bvh.query_sphere(m, center, radius, |t| {
                if allow(t) && (m.tri_centroid(t) - center).dot(normal).abs() <= half {
                    inside.insert(t);
                }
            });
            inside.insert(seed);
            flood_across(m, seed, |t| inside.contains(&t), crease_cos, &mut FastSet::default())
        }
        BrushShape::Unwrap => geodesic_disc(m, seed, center, radius, &allow, crease_cos),
    }
}

/// Magic wand: the faces edge-connected to `seed` whose normals are within `max_angle`
/// (radians) of the seed's normal, not crossing creases sharper than `crease_cos`.
pub fn magic_wand(m: &Mesh, seed: u32, max_angle: f64, crease_cos: Option<f64>) -> Vec<u32> {
    let n0 = m.tri_normal(seed);
    let c = max_angle.cos();
    // The wand can take a whole part, so it marks a bitmap over all triangles rather than
    // hashing each one.
    let mut seen = vec![false; m.max_tid()];
    flood_across(m, seed, |t| m.tri_normal(t).dot(n0) >= c, crease_cos, &mut seen)
}

/// One brush stamp for [`TriSelection::paint_brush`].
#[derive(Clone, Copy, Debug)]
pub struct BrushStamp {
    pub shape: BrushShape,
    /// Triangle under the brush centre.
    pub seed: u32,
    pub center: DVec3,
    /// Surface normal at the centre.
    pub normal: DVec3,
    pub radius: f64,
    /// Do not spread across edges sharper than this (cosine of the angle between the
    /// faces' normals); `None` = no limit. Not used by the Sphere brush.
    pub crease_cos: Option<f64>,
}

/// Is the edge between adjacent triangles `a` and `b` sharper than `crease_cos` allows?
pub fn is_crease(m: &Mesh, a: u32, b: u32, crease_cos: Option<f64>) -> bool {
    crease_cos.is_some_and(|c| m.tri_normal(a).dot(m.tri_normal(b)) < c)
}

/// The triangles a flood has taken: a bitmap over all triangle ids for floods that may
/// cover much of the mesh, a hash set for the small ones of a brush stamp.
trait Taken {
    fn has(&self, t: u32) -> bool;
    fn take(&mut self, t: u32);
}

impl Taken for Vec<bool> {
    #[inline]
    fn has(&self, t: u32) -> bool {
        self[t as usize]
    }
    #[inline]
    fn take(&mut self, t: u32) {
        self[t as usize] = true;
    }
}

impl Taken for FastSet<u32> {
    #[inline]
    fn has(&self, t: u32) -> bool {
        self.contains(&t)
    }
    #[inline]
    fn take(&mut self, t: u32) {
        self.insert(t);
    }
}

/// Edge-connected flood from `seed` through triangles accepted by `accept`, not crossing
/// creases, in the order the triangles are reached. `seen` starts empty.
fn flood_across(m: &Mesh, seed: u32, accept: impl Fn(u32) -> bool, crease_cos: Option<f64>, seen: &mut impl Taken) -> Vec<u32> {
    seen.take(seed);
    let mut out = vec![seed];
    let mut stack = vec![seed];
    while let Some(t) = stack.pop() {
        for n in m.tri_neighbors(t) {
            if n != INVALID && !seen.has(n) && accept(n) && !is_crease(m, t, n, crease_cos) {
                seen.take(n);
                out.push(n);
                stack.push(n);
            }
        }
    }
    out
}

/// Shape of the selection brush (Meshmixer's Brush Mode).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrushShape {
    /// Everything inside the 3D sphere.
    Sphere,
    /// The connected part of the sphere near the surface under the brush.
    SphereDisc,
    /// A disc measured along the surface (geodesic distance), never jumping across gaps.
    Unwrap,
}

/// Triangles within geodesic distance `radius` of `center` (on triangle `seed`): surface
/// distances to the vertices first (bounded Dijkstra), then an edge-connected flood from the
/// seed through triangles whose corners average within the radius, accepted by `allow`, and
/// not across creases.
fn geodesic_disc(m: &Mesh, seed: u32, center: DVec3, radius: f64, allow: &impl Fn(u32) -> bool, crease_cos: Option<f64>) -> Vec<u32> {
    use std::collections::BinaryHeap;
    #[derive(PartialEq)]
    struct Item(f64, u32);
    impl Eq for Item {}
    impl PartialOrd for Item {
        fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(o))
        }
    }
    impl Ord for Item {
        fn cmp(&self, o: &Self) -> std::cmp::Ordering {
            o.0.total_cmp(&self.0)
        }
    }
    // Looked up only, never walked, so its order cannot show.
    let mut dist: FastMap<u32, f64> = FastMap::default();
    let mut heap = BinaryHeap::new();
    for v in m.triangle(seed) {
        let d = m.vertex(v).distance(center);
        dist.insert(v, d);
        heap.push(Item(d, v));
    }
    while let Some(Item(d, v)) = heap.pop() {
        if d > dist.get(&v).copied().unwrap_or(f64::INFINITY) || d > radius {
            continue;
        }
        let p = m.vertex(v);
        for u in m.vertex_neighbors(v) {
            let nd = d + p.distance(m.vertex(u));
            if nd < dist.get(&u).copied().unwrap_or(f64::INFINITY) {
                dist.insert(u, nd);
                heap.push(Item(nd, u));
            }
        }
    }
    let within = |t: u32| {
        let tv = m.triangle(t);
        tv.iter().all(|u| dist.contains_key(u)) && tv.iter().map(|u| dist[u]).sum::<f64>() / 3.0 <= radius
    };
    flood_across(m, seed, |t| within(t) && allow(t), crease_cos, &mut FastSet::default())
}

#[cfg(test)]
mod tests {
    use crate::primitives;

    #[test]
    fn brush_shapes() {
        // A folded strip: two parallel plates 0.3 apart, joined at one end (like a thin wall).
        let mut m = primitives::plane_grid(4.0, 4.0, 40, 40);
        let mut b = primitives::plane_grid(4.0, 4.0, 40, 40);
        b.transform(|p| p + DVec3::new(0.0, 0.0, 0.3));
        m.append_mesh(&b, 0);
        let bvh = Bvh::build(&m);
        let center = DVec3::ZERO;
        let seed = bvh.nearest(&m, center, 1.0).unwrap().tid;
        let stamp = |shape: BrushShape| BrushStamp { shape, seed, center, normal: DVec3::Z, radius: 0.5, crease_cos: None };
        let count = |shape: BrushShape| {
            let mut s = TriSelection::new(&m);
            s.paint_brush(&m, &bvh, &stamp(shape), |_| true, false);
            s
        };
        let sphere = count(BrushShape::Sphere);
        let disc = count(BrushShape::SphereDisc);
        let unwrap = count(BrushShape::Unwrap);
        // The sphere reaches the other plate; the disc and the geodesic brush do not.
        assert!(sphere.iter().any(|t| m.tri_centroid(t).z > 0.1));
        assert!(disc.iter().all(|t| m.tri_centroid(t).z < 0.1));
        assert!(unwrap.iter().all(|t| m.tri_centroid(t).z < 0.1));
        // The geodesic disc is round with the given radius.
        assert!(unwrap.iter().all(|t| m.tri_centroid(t).length() < 0.55));
        assert!(unwrap.len() > 50);
        // Filters apply (except to the seed).
        let mut f = TriSelection::new(&m);
        f.paint_brush(&m, &bvh, &stamp(BrushShape::Unwrap), |t| m.tri_centroid(t).x < 0.0, false);
        assert!(f.iter().all(|t| t == seed || m.tri_centroid(t).x < 0.0));
    }
    use super::*;

    #[test]
    fn connected_vs_same_group_expansion() {
        // Two separate patches of group 1 on a grid; start from one triangle of the first.
        let mut m = crate::primitives::plane_grid(10.0, 2.0, 20, 4);
        let ids: Vec<u32> = m.triangle_ids().collect();
        for &t in &ids {
            let x = m.tri_centroid(t).x;
            if (-4.0..-2.0).contains(&x) || (2.0..4.0).contains(&x) {
                m.set_tri_group(t, 1);
            }
        }
        let seed = ids.iter().copied().find(|&t| (-4.0..-2.0).contains(&m.tri_centroid(t).x)).unwrap();
        let mut a = TriSelection::new(&m);
        a.insert(seed);
        a.expand_to_connected_groups(&m);
        assert!(a.iter().all(|t| m.tri_centroid(t).x < 0.0 && m.tri_group(t) == 1));
        let mut b = TriSelection::new(&m);
        b.insert(seed);
        b.expand_to_groups(&m);
        assert_eq!(b.len(), 2 * a.len());
    }

    #[test]
    fn expand_contract_invert() {
        let m = primitives::icosphere(1.0, 2);
        let mut s = TriSelection::new(&m);
        s.insert(0);
        s.expand(&m);
        assert!(s.len() > 1);
        let n = s.len();
        s.expand(&m);
        assert!(s.len() > n);
        s.contract(&m);
        assert_eq!(s.len(), n);
        s.invert(&m);
        assert_eq!(s.len(), m.triangle_count() - n);
        s.expand_to_connected(&m);
        assert_eq!(s.len(), m.triangle_count());
    }

    #[test]
    fn brush_stops_at_creases() {
        // A cube: the brush at a face centre with a large radius spills over the edges unless
        // a crease limit is set.
        let m = primitives::cube(2.0, 8);
        let bvh = Bvh::build(&m);
        let hit = bvh.raycast(&m, &crate::geom::Ray::new(DVec3::new(0.2, 0.1, 5.0), -DVec3::Z)).unwrap();
        for shape in [BrushShape::Unwrap, BrushShape::SphereDisc] {
            let base = BrushStamp { shape, seed: hit.tid, center: hit.point, normal: DVec3::Z, radius: 1.6, crease_cos: None };
            let mut free = TriSelection::new(&m);
            free.paint_brush(&m, &bvh, &base, |_| true, false);
            let mut limited = TriSelection::new(&m);
            limited.paint_brush(&m, &bvh, &BrushStamp { crease_cos: Some(30f64.to_radians().cos()), ..base }, |_| true, false);
            assert!(limited.iter().all(|t| m.tri_normal(t).z > 0.99), "{shape:?} stays on the top face");
            if shape == BrushShape::Unwrap {
                assert!(free.iter().any(|t| m.tri_normal(t).z < 0.5), "without a limit it spills over");
            }
        }
    }

    #[test]
    fn magic_wand_regions() {
        let cube = primitives::cube(2.0, 4);
        let top = cube.triangle_ids().find(|&t| cube.tri_normal(t).z > 0.99).unwrap();
        let face = magic_wand(&cube, top, 20f64.to_radians(), None);
        assert_eq!(face.len(), 32);
        assert!(face.iter().all(|&t| cube.tri_normal(t).z > 0.99));
        let sphere = primitives::icosphere(1.0, 4);
        let pole = sphere.triangle_ids().max_by(|&a, &b| sphere.tri_centroid(a).z.total_cmp(&sphere.tri_centroid(b).z)).unwrap();
        let cap = magic_wand(&sphere, pole, 30f64.to_radians(), None);
        assert!(cap.iter().all(|&t| sphere.tri_centroid(t).z > 30f64.to_radians().cos() - 0.05));
        assert!(cap.len() > 50);
    }

    #[test]
    fn paint_selects_local_patch() {
        let m = primitives::icosphere(1.0, 3);
        let bvh = Bvh::build(&m);
        let hit = bvh.raycast(&m, &crate::geom::Ray::new(DVec3::new(0.0, 0.0, 5.0), -DVec3::Z)).unwrap();
        let mut s = TriSelection::new(&m);
        let stamp = BrushStamp { shape: BrushShape::SphereDisc, seed: hit.tid, center: hit.point, normal: DVec3::Z, radius: 0.3, crease_cos: None };
        s.paint_brush(&m, &bvh, &stamp, |_| true, false);
        assert!(s.len() > 3 && s.len() < m.triangle_count() / 4);
        for t in s.iter() {
            assert!(m.tri_centroid(t).z > 0.5);
        }
    }

    /// The flood as it was, with a std `HashSet`.
    fn flood_std(m: &Mesh, seed: u32, accept: impl Fn(u32) -> bool, crease_cos: Option<f64>) -> Vec<u32> {
        let mut seen = std::collections::HashSet::from([seed]);
        let mut out = vec![seed];
        let mut stack = vec![seed];
        while let Some(t) = stack.pop() {
            for n in m.tri_neighbors(t) {
                if n != INVALID && !seen.contains(&n) && accept(n) && !is_crease(m, t, n, crease_cos) {
                    seen.insert(n);
                    out.push(n);
                    stack.push(n);
                }
            }
        }
        out
    }

    /// The brush region as it was, with std hash sets and maps.
    fn region_std(m: &Mesh, bvh: &Bvh, stamp: &BrushStamp, allow: impl Fn(u32) -> bool) -> Vec<u32> {
        let BrushStamp { shape, seed, center, normal, radius, crease_cos } = *stamp;
        match shape {
            BrushShape::Sphere => brush_region(m, bvh, stamp, allow),
            BrushShape::SphereDisc => {
                let half = radius * 0.35;
                let mut inside = std::collections::HashSet::new();
                bvh.query_sphere(m, center, radius, |t| {
                    if allow(t) && (m.tri_centroid(t) - center).dot(normal).abs() <= half {
                        inside.insert(t);
                    }
                });
                inside.insert(seed);
                flood_std(m, seed, |t| inside.contains(&t), crease_cos)
            }
            BrushShape::Unwrap => {
                #[derive(PartialEq)]
                struct Item(f64, u32);
                impl Eq for Item {}
                impl PartialOrd for Item {
                    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
                        Some(self.cmp(o))
                    }
                }
                impl Ord for Item {
                    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
                        o.0.total_cmp(&self.0)
                    }
                }
                let mut dist: std::collections::HashMap<u32, f64> = std::collections::HashMap::new();
                let mut heap = std::collections::BinaryHeap::new();
                for v in m.triangle(seed) {
                    let d = m.vertex(v).distance(center);
                    dist.insert(v, d);
                    heap.push(Item(d, v));
                }
                while let Some(Item(d, v)) = heap.pop() {
                    if d > dist.get(&v).copied().unwrap_or(f64::INFINITY) || d > radius {
                        continue;
                    }
                    let p = m.vertex(v);
                    for u in m.vertex_neighbors(v) {
                        let nd = d + p.distance(m.vertex(u));
                        if nd < dist.get(&u).copied().unwrap_or(f64::INFINITY) {
                            dist.insert(u, nd);
                            heap.push(Item(nd, u));
                        }
                    }
                }
                let within = |t: u32| {
                    let tv = m.triangle(t);
                    tv.iter().all(|u| dist.contains_key(u)) && tv.iter().map(|u| dist[u]).sum::<f64>() / 3.0 <= radius
                };
                flood_std(m, seed, |t| within(t) && allow(t), crease_cos)
            }
        }
    }

    #[test]
    fn floods_unchanged_by_their_sets() {
        // A cube (creases), a lumpy sphere and a folded double plate.
        let mut lumpy = primitives::icosphere(1.0, 4);
        lumpy.transform(|p| p * (1.0 + 0.2 * (5.0 * p.x).sin() * (4.0 * p.y).cos()));
        let mut plates = primitives::plane_grid(4.0, 4.0, 40, 40);
        let mut b = primitives::plane_grid(4.0, 4.0, 40, 40);
        b.transform(|p| p + DVec3::new(0.0, 0.0, 0.3));
        plates.append_mesh(&b, 0);
        for m in [primitives::cube(2.0, 10), lumpy, plates] {
            let bvh = Bvh::build(&m);
            let ids: Vec<u32> = m.triangle_ids().collect();
            let diag = m.bounds().diagonal();
            for (k, &seed) in ids.iter().step_by(ids.len() / 23).enumerate() {
                let crease = [None, Some(20f64.to_radians().cos()), Some(60f64.to_radians().cos())][k % 3];
                for angle in [10.0f64, 45.0, 180.0] {
                    let n0 = m.tri_normal(seed);
                    let c = angle.to_radians().cos();
                    assert_eq!(magic_wand(&m, seed, angle.to_radians(), crease), flood_std(&m, seed, |t| m.tri_normal(t).dot(n0) >= c, crease));
                }
                let center = m.tri_centroid(seed);
                for shape in [BrushShape::Sphere, BrushShape::SphereDisc, BrushShape::Unwrap] {
                    for r in [0.02, 0.1, 0.3] {
                        let stamp = BrushStamp { shape, seed, center, normal: m.tri_normal(seed), radius: r * diag, crease_cos: crease };
                        let allow = |t: u32| t % 7 != 3;
                        assert_eq!(brush_region(&m, &bvh, &stamp, allow), region_std(&m, &bvh, &stamp, allow), "{shape:?} r {r}");
                    }
                }
            }
        }
    }

    #[test]
    fn grow_to_is_sync_for_empty_selections() {
        let mut m = primitives::icosphere(1.0, 2);
        let mut a = TriSelection::new(&m);
        let mut b = TriSelection::new(&m);
        // Grow (new triangles), then shrink (a smaller mesh in its place).
        for e in m.edge_ids().step_by(5).collect::<Vec<_>>() {
            let _ = m.split_edge(e);
        }
        a.sync(&m);
        b.grow_to(m.max_tid());
        assert_eq!(a, b);
        let small = primitives::icosphere(1.0, 1);
        a.sync(&small);
        b.grow_to(small.max_tid());
        assert_eq!(a, b);
        assert_eq!(b.bits().len(), small.max_tid());
        // With triangles selected, the count follows what is cut off.
        let mut s = TriSelection::all(&m);
        s.grow_to(100);
        assert_eq!(s.len(), 100);
        assert_eq!(s.iter().count(), 100);
        s.grow_to(m.max_tid());
        assert_eq!(s.len(), 100);
        // sync's shortcut for empty selections leaves the same state as its full walk.
        let mut full = TriSelection::all(&m);
        full.clear();
        let mut fresh = TriSelection::new(&primitives::icosphere(1.0, 3));
        full.sync(&small);
        fresh.sync(&small);
        assert_eq!(full, fresh);
    }
}