//! Optional triangle region that restricts an operation.
//!
//! Outside triangles are never modified: every vertex touching an outside triangle and
//! every edge on the region border is constrained as fixed by the callers, so local
//! edits only ever touch region triangles. Triangles created inside the region (by
//! splits) are added; triangles removed by collapses are dropped lazily.

use crate::util::Stamp;
use mm_core::Mesh;

pub(crate) struct Region {
    enabled: bool,
    tin: Vec<bool>,
    tris: Vec<u32>,
}

impl Region {
    pub fn new(mesh: &Mesh, region: Option<&[u32]>) -> Self {
        match region {
            None => Self { enabled: false, tin: Vec::new(), tris: Vec::new() },
            Some(r) => {
                let mut tin = vec![false; mesh.max_tid()];
                let mut tris = Vec::with_capacity(r.len());
                for &t in r {
                    if mesh.is_triangle(t) && !tin[t as usize] {
                        tin[t as usize] = true;
                        tris.push(t);
                    }
                }
                tris.sort_unstable();
                Self { enabled: true, tin, tris }
            }
        }
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// True if `t` (a live triangle) belongs to the region.
    #[inline]
    pub fn contains(&self, t: u32) -> bool {
        !self.enabled || self.tin.get(t as usize).copied().unwrap_or(false)
    }

    /// Register a triangle created inside the region.
    #[inline]
    pub fn add(&mut self, t: u32) {
        if !self.enabled {
            return;
        }
        let i = t as usize;
        if i >= self.tin.len() {
            self.tin.resize(i + 1, false);
        }
        self.tin[i] = true;
        self.tris.push(t);
    }

    /// Drop dead ids and duplicates from the triangle list.
    pub fn refresh(&mut self, mesh: &Mesh) {
        if !self.enabled {
            return;
        }
        let tin = &self.tin;
        self.tris.retain(|&t| mesh.is_triangle(t) && tin[t as usize]);
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

    /// Edges of region triangles (all live edges when no region is set).
    pub fn edges(&mut self, mesh: &Mesh, stamp: &mut Stamp) -> Vec<u32> {
        if !self.enabled {
            return mesh.edge_ids().collect();
        }
        self.refresh(mesh);
        stamp.begin(mesh.max_eid());
        let mut out = Vec::with_capacity(self.tris.len() * 3 / 2 + 8);
        for &t in &self.tris {
            for e in mesh.tri_edges(t) {
                if stamp.mark(e) {
                    out.push(e);
                }
            }
        }
        out.sort_unstable();
        out
    }

    /// Vertices of region triangles (all live vertices when no region is set).
    pub fn vertices(&mut self, mesh: &Mesh, stamp: &mut Stamp) -> Vec<u32> {
        if !self.enabled {
            return mesh.vertex_ids().collect();
        }
        self.refresh(mesh);
        stamp.begin(mesh.max_vid());
        let mut out = Vec::with_capacity(self.tris.len() / 2 + 8);
        for &t in &self.tris {
            for v in mesh.triangle(t) {
                if stamp.mark(v) {
                    out.push(v);
                }
            }
        }
        out.sort_unstable();
        out
    }

    /// Region triangles plus every outside triangle sharing a vertex with the region
    /// (used to build the projection reference near the region border).
    pub fn triangles_with_collar(&mut self, mesh: &Mesh) -> Vec<u32> {
        let mut tris = self.triangles(mesh);
        if !self.enabled {
            return tris;
        }
        let mut st = Stamp::default();
        st.begin(mesh.max_tid());
        for &t in &tris {
            st.mark(t);
        }
        let mut extra = Vec::new();
        let mut vs = Stamp::default();
        vs.begin(mesh.max_vid());
        for &t in &tris {
            for v in mesh.triangle(t) {
                if !vs.mark(v) {
                    continue;
                }
                for n in mesh.vertex_triangles(v) {
                    if st.mark(n) {
                        extra.push(n);
                    }
                }
            }
        }
        tris.extend(extra);
        tris
    }
}
