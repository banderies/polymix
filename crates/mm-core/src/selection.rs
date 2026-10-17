//! Triangle selections and the region operations Meshmixer exposes on them
//! (expand/contract, connected, invert, boundary optimisation, brush painting).

use crate::bvh::Bvh;
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

    /// Select triangles within `radius` of `center` that are connected (through such
    /// triangles) to `seed`. This is the surface "sphere brush" used by the select tool.
    pub fn paint_sphere(&mut self, m: &Mesh, bvh: &Bvh, seed: u32, center: DVec3, radius: f64, deselect: bool) -> usize {
        let mut inside = std::collections::HashSet::new();
        bvh.query_sphere(m, center, radius, |t| {
            inside.insert(t);
        });
        if !inside.contains(&seed) {
            inside.insert(seed);
        }
        let region = crate::topology::flood_fill_triangles(m, &[seed], |t| inside.contains(&t));
        let mut n = 0;
        for t in region {
            let changed = if deselect { self.remove(t) } else { self.insert(t) };
            if changed {
                n += 1;
            }
        }
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;

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
    fn paint_selects_local_patch() {
        let m = primitives::icosphere(1.0, 3);
        let bvh = Bvh::build(&m);
        let hit = bvh.raycast(&m, &crate::geom::Ray::new(DVec3::new(0.0, 0.0, 5.0), -DVec3::Z)).unwrap();
        let mut s = TriSelection::new(&m);
        s.paint_sphere(&m, &bvh, hit.tid, hit.point, 0.3, false);
        assert!(s.len() > 3 && s.len() < m.triangle_count() / 4);
        for t in s.iter() {
            assert!(m.tri_centroid(t).z > 0.5);
        }
    }
}
