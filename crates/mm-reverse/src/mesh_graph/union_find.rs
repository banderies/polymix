// SPDX-License-Identifier: AGPL-3.0-only
//! Disjoint sets, the one way every grouping pass here keeps them: a parent per member, the
//! path halved on every find so the trees stay shallow without a rank. Which member ends up
//! the root is up to the order of the joins; callers that use the root for more than a key
//! say how they join.
//!
//! Ported from ScanRuler (`src/core/geometry/unionFind.ts`), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only.

/// The root of `i`'s set, halving the path on the way up.
pub fn find_root(parent: &mut [u32], mut i: usize) -> usize {
    while parent[i] as usize != i {
        parent[i] = parent[parent[i] as usize];
        i = parent[i] as usize;
    }
    i
}

/// Disjoint sets over the members 0 … n − 1.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnionFind {
    pub parent: Vec<u32>,
}

impl UnionFind {
    /// The members 0 … n − 1, each a set of its own.
    pub fn new(n: usize) -> Self {
        Self { parent: (0..n as u32).collect() }
    }

    /// The root of `i`'s set.
    pub fn find(&mut self, i: usize) -> usize {
        find_root(&mut self.parent, i)
    }

    /// Join the sets of `a` and `b`: `a`'s root goes under `b`'s.
    pub fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra != rb {
            self.parent[ra] = rb as u32;
        }
    }
}

/// The connected pieces of a triangle mesh: each triangle joins its three corners, the
/// second's and third's roots going under the first's.
pub fn vertex_components(indices: &[u32], vertex_count: usize) -> UnionFind {
    let mut sets = UnionFind::new(vertex_count);
    for tri in indices.as_chunks::<3>().0 {
        let a = sets.find(tri[0] as usize);
        for &corner in &tri[1..] {
            let b = sets.find(corner as usize);
            if b != a {
                sets.parent[b] = a as u32;
            }
        }
    }
    sets
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joins_the_corners_of_each_triangle_into_one_piece() {
        // Two separate triangles and a lone vertex.
        let mut sets = vertex_components(&[0, 1, 2, 3, 4, 5], 7);
        assert_eq!(sets.find(1), sets.find(2));
        assert_eq!(sets.find(0), sets.find(2));
        assert_eq!(sets.find(3), sets.find(5));
        assert_ne!(sets.find(0), sets.find(3));
        assert_eq!(sets.find(6), 6);
        // The second's and third's roots go under the first's.
        assert_eq!(sets.find(1), 0);
        let mut u = UnionFind::new(3);
        u.union(0, 2);
        assert_eq!(u.find(0), 2);
    }
}
