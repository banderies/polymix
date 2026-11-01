// SPDX-License-Identifier: AGPL-3.0-only
//! Vertex-to-vertex adjacency in compressed rows (offsets and a flat neighbour list).
//!
//! Ported from ScanRuler (`src/core/geometry/adjacency.ts`), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only.

/// A mesh's vertex adjacency (CSR): vertex `v`'s neighbours are
/// `list[offsets[v]..offsets[v + 1]]`.
///
/// Neighbours may appear more than once (once per shared triangle); a breadth-first search's
/// visited checks make that harmless, and removing the duplicates is not worth a pass.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Adjacency {
    pub offsets: Vec<u32>,
    pub list: Vec<u32>,
}

impl Adjacency {
    /// Vertex `v`'s neighbours, each once per triangle they share.
    pub fn neighbours(&self, v: usize) -> &[u32] {
        &self.list[self.offsets[v] as usize..self.offsets[v + 1] as usize]
    }
}

/// The adjacency of the triangles `indices` (three a triangle) over `vertex_count` vertices.
/// An index at or past `vertex_count` is left out.
pub fn build_adjacency(indices: &[u32], vertex_count: usize) -> Adjacency {
    let mut offsets = vec![0u32; vertex_count + 1];
    let tris = indices.len() / 3;
    let valid = |t: usize| indices[t * 3..t * 3 + 3].iter().all(|&v| (v as usize) < vertex_count);
    for t in 0..tris {
        if !valid(t) {
            continue;
        }
        for k in 0..3 {
            offsets[indices[t * 3 + k] as usize + 1] += 2;
        }
    }
    for v in 0..vertex_count {
        offsets[v + 1] += offsets[v];
    }
    let mut list = vec![0u32; offsets[vertex_count] as usize];
    let mut cursor: Vec<u32> = offsets[..vertex_count].to_vec();
    for t in 0..tris {
        if !valid(t) {
            continue;
        }
        let a = indices[t * 3];
        let b = indices[t * 3 + 1];
        let c = indices[t * 3 + 2];
        let mut put = |v: u32, w: u32| {
            list[cursor[v as usize] as usize] = w;
            cursor[v as usize] += 1;
        };
        put(a, b);
        put(a, c);
        put(b, a);
        put(b, c);
        put(c, a);
        put(c, b);
    }
    Adjacency { offsets, list }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_triangles_share_an_edge() {
        let adj = build_adjacency(&[0, 1, 2, 0, 2, 3], 4);
        assert_eq!(adj.offsets, vec![0, 4, 6, 10, 12]);
        assert_eq!(adj.neighbours(0), &[1, 2, 2, 3]);
        assert_eq!(adj.neighbours(2), &[0, 1, 0, 3]);
    }
}
