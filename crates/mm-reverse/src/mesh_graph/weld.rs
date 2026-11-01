// SPDX-License-Identifier: AGPL-3.0-only
//! A triangle soup welded into an indexed mesh, and the triangles welding leaves with a
//! corner repeated dropped.
//!
//! Ported from ScanRuler (`src/core/geometry/weld.ts`), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only. ScanRuler matches the float bit patterns through an open-addressing table
//! of its own; here the standard hash map does the matching. The vertices are numbered in the
//! order they are first met either way, so the welded mesh is the same.

use std::collections::HashMap;

use glam::DVec3;

/// Merge the duplicated vertices of a triangle soup (three corners per triangle) into an
/// indexed mesh. Vertices are matched on their exact bit patterns — a binary STL repeats
/// vertices bit-identically — and numbered in the order they are first met. Returns the
/// distinct positions and one index per soup corner.
pub fn weld_triangle_soup(soup: &[DVec3]) -> (Vec<DVec3>, Vec<u32>) {
    let mut table: HashMap<[u64; 3], u32> = HashMap::with_capacity(soup.len());
    let mut positions = Vec::new();
    let mut indices = Vec::with_capacity(soup.len());
    for p in soup {
        let key = [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()];
        let id = *table.entry(key).or_insert_with(|| {
            positions.push(*p);
            positions.len() as u32 - 1
        });
        indices.push(id);
    }
    (positions, indices)
}

/// Drop triangles that reference the same vertex twice (zero-area after welding).
pub fn filter_degenerate_triangles(indices: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(indices.len());
    for tri in indices.as_chunks::<3>().0 {
        let (a, b, c) = (tri[0], tri[1], tri[2]);
        if a == b || b == c || a == c {
            continue;
        }
        out.extend_from_slice(tri);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn welds_shared_corners_in_the_order_they_are_met() {
        let a = DVec3::new(0.0, 0.0, 0.0);
        let b = DVec3::new(1.0, 0.0, 0.0);
        let c = DVec3::new(0.0, 1.0, 0.0);
        let d = DVec3::new(1.0, 1.0, 0.0);
        let (positions, indices) = weld_triangle_soup(&[a, b, c, b, d, c]);
        assert_eq!(positions, vec![a, b, c, d]);
        assert_eq!(indices, vec![0, 1, 2, 1, 3, 2]);
        // A negative zero is another bit pattern, and another vertex.
        let (p2, _) = weld_triangle_soup(&[a, DVec3::new(-0.0, 0.0, 0.0)]);
        assert_eq!(p2.len(), 2);
        assert_eq!(filter_degenerate_triangles(&[0, 1, 2, 1, 1, 2, 3, 4, 3]), vec![0, 1, 2]);
    }
}
