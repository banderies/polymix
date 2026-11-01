// SPDX-License-Identifier: AGPL-3.0-only
//! Vertex-to-vertex adjacency of a triangle mesh in CSR form.
//!
//! Ported from ScanRuler (`src/core/geometry/adjacency.ts`), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only.

/// Vertex-to-vertex adjacency in CSR form: the neighbours of `v` are
/// `list[offsets[v]..offsets[v + 1]]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Adjacency {
    pub offsets: Vec<u32>,
    pub list: Vec<u32>,
}

/// Vertex-to-vertex adjacency in CSR form (offsets plus a flat neighbour list). Neighbours
/// may appear more than once (once per shared triangle); the visited checks of a
/// breadth-first walk make that harmless and deduplication is not worth a pass. `indices`
/// holds three vertex ids per triangle, each below `vertex_count`.
pub fn build_adjacency(indices: &[u32], vertex_count: usize) -> Adjacency {
    let mut offsets = vec![0u32; vertex_count + 1];
    for tri in indices.as_chunks::<3>().0 {
        offsets[tri[0] as usize + 1] += 2;
        offsets[tri[1] as usize + 1] += 2;
        offsets[tri[2] as usize + 1] += 2;
    }
    for v in 0..vertex_count {
        offsets[v + 1] += offsets[v];
    }

    let mut list = vec![0u32; offsets[vertex_count] as usize];
    let mut cursor: Vec<u32> = offsets[..vertex_count].to_vec();
    let mut put = |list: &mut Vec<u32>, at: u32, w: u32| {
        let c = &mut cursor[at as usize];
        list[*c as usize] = w;
        *c += 1;
    };
    for tri in indices.as_chunks::<3>().0 {
        let (a, b, c) = (tri[0], tri[1], tri[2]);
        put(&mut list, a, b);
        put(&mut list, a, c);
        put(&mut list, b, a);
        put(&mut list, b, c);
        put(&mut list, c, a);
        put(&mut list, c, b);
    }
    Adjacency { offsets, list }
}
