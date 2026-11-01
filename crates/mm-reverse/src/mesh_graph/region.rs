// SPDX-License-Identifier: AGPL-3.0-only
//! The triangles a marked surface consists of: the ones with all three corners in the
//! marking. That is the rule the tint on the part follows, so what a search is confined to is
//! exactly what is coloured — a triangle with one corner outside the marking is bare on
//! screen and out of the search alike.
//!
//! Ported from ScanRuler (`src/core/geometry/region.ts`), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only.

/// The triangle index reduced to the triangles whose three corners are all among `vertices`,
/// in their original order. A marked vertex beyond `vertex_count` is ignored rather than read
/// past.
pub fn triangles_within(indices: &[u32], vertices: &[u32], vertex_count: usize) -> Vec<u32> {
    let mut inside = vec![false; vertex_count];
    for &v in vertices {
        if (v as usize) < vertex_count {
            inside[v as usize] = true;
        }
    }
    let marked = |v: u32| inside.get(v as usize).copied().unwrap_or(false);
    let mut out = Vec::new();
    for tri in indices.as_chunks::<3>().0 {
        if marked(tri[0]) && marked(tri[1]) && marked(tri[2]) {
            out.extend_from_slice(tri);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // Two triangles sharing the edge 1–2: (0,1,2) and (1,3,2).
    const INDICES: [u32; 6] = [0, 1, 2, 1, 3, 2];

    #[test]
    fn keeps_a_triangle_only_when_all_three_corners_are_marked() {
        assert_eq!(triangles_within(&INDICES, &[0, 1, 2], 4), vec![0, 1, 2]);
        assert_eq!(triangles_within(&INDICES, &[1, 2, 3], 4), vec![1, 3, 2]);
        assert_eq!(triangles_within(&INDICES, &[0, 1, 2, 3], 4), vec![0, 1, 2, 1, 3, 2]);
        // Two corners are not enough: the tint would leave the triangle bare.
        assert!(triangles_within(&INDICES, &[1, 2], 4).is_empty());
    }

    #[test]
    fn ignores_a_vertex_beyond_the_mesh_rather_than_reading_past_it() {
        assert_eq!(triangles_within(&INDICES, &[0, 1, 2, 99], 4), vec![0, 1, 2]);
    }
}
