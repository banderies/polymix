// SPDX-License-Identifier: AGPL-3.0-only
//! The per-vertex "corner slot" a viewport's mesh mode is drawn from.
//!
//! Drawing the triangle edges of a million-triangle scan as lines means a second index
//! buffer three times the size of the first and a second pass over the whole part every
//! frame — and at any distance where the triangles are smaller than a pixel, a solid block of
//! line colour. The viewport draws the edges inside the surface shader instead: a fragment
//! that knows how far it is from the nearest edge of its own triangle can darken itself
//! there, in the same pass that shades it.
//!
//! What a fragment gets for free is its barycentric position, *if* the three corners of every
//! triangle carry three different one-hot attributes. Indexed geometry shares vertices
//! between triangles, so the attribute has to be chosen per vertex such that no triangle sees
//! the same value twice — a proper colouring of the vertex graph. Eight colours ("slots") are
//! more than a scan's vertices ever need in practice: a greedy pass in vertex order hands
//! each vertex the lowest slot none of its already-coloured neighbours holds, and only a
//! vertex whose coloured neighbours already cover all eight is forced to share. Where one is,
//! the triangles around it lose an edge or two, not the picture; see `wire_conflicts` for the
//! count.
//!
//! Ported from ScanRuler (`src/core/geometry/wireSlots.ts`), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only.

/// How many distinct slots there are. Eight is what two vec4 varyings carry.
pub const WIRE_SLOT_COUNT: usize = 8;

const UNSET: u8 = 255;

/// A slot per vertex, 0–7, such that (nearly) every triangle's three corners differ.
/// Adjacency in CSR form, as `build_adjacency` makes it; neighbours listed more than once are
/// harmless.
pub fn wire_slots(adj_offsets: &[u32], adj_list: &[u32], vertex_count: usize) -> Vec<u8> {
    let mut slots = vec![UNSET; vertex_count];
    // Which slots this vertex's coloured neighbours hold, as a stamp per slot rather than a
    // set that would have to be cleared per vertex.
    let mut stamped = [0u32; WIRE_SLOT_COUNT];
    let mut counts = [0u32; WIRE_SLOT_COUNT];
    for v in 0..vertex_count {
        let stamp = v as u32 + 1;
        let around = &adj_list[adj_offsets[v] as usize..adj_offsets[v + 1] as usize];
        for &w in around {
            let s = slots[w as usize];
            if s != UNSET {
                stamped[s as usize] = stamp;
            }
        }
        let mut pick = UNSET;
        for (s, &mark) in stamped.iter().enumerate() {
            if mark != stamp {
                pick = s as u8;
                break;
            }
        }
        if pick == UNSET {
            // Every slot is taken around this vertex. Share the one fewest neighbours hold,
            // so the fewest triangles lose an edge.
            counts.fill(0);
            for &w in around {
                let s = slots[w as usize];
                if s != UNSET {
                    counts[s as usize] += 1;
                }
            }
            pick = 0;
            for s in 1..WIRE_SLOT_COUNT {
                if counts[s] < counts[pick as usize] {
                    pick = s as u8;
                }
            }
        }
        slots[v] = pick;
    }
    slots
}

/// How many triangles have two corners in the same slot — the ones the shader cannot draw
/// every edge of. Zero on any ordinary mesh; a diagnostic, not something the viewport needs.
pub fn wire_conflicts(indices: &[u32], slots: &[u8]) -> usize {
    indices
        .as_chunks::<3>().0.iter()
        .filter(|t| {
            let (a, b, c) = (slots[t[0] as usize], slots[t[1] as usize], slots[t[2] as usize]);
            a == b || b == c || a == c
        })
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit3d::testkit::icosphere;
    use crate::mesh_graph::adjacency::build_adjacency;

    fn flat(faces: &[[u32; 3]]) -> Vec<u32> {
        faces.iter().flatten().copied().collect()
    }

    #[test]
    fn gives_every_triangle_of_a_closed_mesh_three_distinct_corners() {
        let (positions, faces) = icosphere(4, 10.0, 0.0, 7);
        let indices = flat(&faces);
        let n = positions.len();
        let adj = build_adjacency(&indices, n);
        let slots = wire_slots(&adj.offsets, &adj.list, n);
        assert_eq!(slots.len(), n);
        assert!(slots.iter().all(|&s| (s as usize) < WIRE_SLOT_COUNT));
        assert_eq!(wire_conflicts(&indices, &slots), 0);
    }

    #[test]
    fn copes_with_an_odd_fan_where_three_colours_would_not_do() {
        // A hub with seven spokes: around an odd ring, alternating two colours meets itself,
        // so at least one more is needed somewhere.
        let n = 7u32;
        let indices: Vec<u32> = (0..n).flat_map(|i| [0, 1 + i, 1 + (i + 1) % n]).collect();
        let adj = build_adjacency(&indices, n as usize + 1);
        let slots = wire_slots(&adj.offsets, &adj.list, n as usize + 1);
        assert_eq!(wire_conflicts(&indices, &slots), 0);
    }

    #[test]
    fn shares_a_slot_rather_than_failing_when_all_eight_are_taken() {
        // Eight mutually adjacent vertices take all eight slots; a ninth joined to every one
        // of them has nowhere to go and must still get a slot.
        let k = WIRE_SLOT_COUNT;
        let adj: Vec<Vec<u32>> = (0..=k).map(|a| (0..=k).filter(|&b| b != a).map(|b| b as u32).collect()).collect();
        let mut offsets = vec![0u32; k + 2];
        for v in 0..=k {
            offsets[v + 1] = offsets[v] + adj[v].len() as u32;
        }
        let list: Vec<u32> = adj.concat();
        let slots = wire_slots(&offsets, &list, k + 1);
        for (v, &s) in slots.iter().enumerate().take(k) {
            assert_eq!(s as usize, v);
        }
        assert!((slots[k] as usize) < k);
    }
}
