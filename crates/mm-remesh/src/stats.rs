//! Edge-length and triangle-quality statistics.

use crate::util::{quality, Stamp};
use mm_core::hash::{fast_set_with_capacity, FastSet};
use mm_core::{Mesh, INVALID};
use rayon::prelude::*;

/// Summary of edge lengths.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EdgeLengthStats {
    pub count: usize,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    /// Standard deviation of the edge lengths.
    pub std_dev: f64,
}

impl EdgeLengthStats {
    fn from_lengths(it: impl Iterator<Item = f64>) -> Self {
        let mut n = 0usize;
        let mut min = f64::INFINITY;
        let mut max = 0.0f64;
        let mut sum = 0.0;
        let mut sum2 = 0.0;
        for l in it {
            n += 1;
            min = min.min(l);
            max = max.max(l);
            sum += l;
            sum2 += l * l;
        }
        if n == 0 {
            return Self::default();
        }
        let mean = sum / n as f64;
        let var = (sum2 / n as f64 - mean * mean).max(0.0);
        Self { count: n, min, max, mean, std_dev: var.sqrt() }
    }
}

/// Edge-length statistics over all edges of the mesh.
pub fn edge_length_stats(mesh: &Mesh) -> EdgeLengthStats {
    EdgeLengthStats::from_lengths(mesh.edge_ids().map(|e| mesh.edge_length(e)))
}

/// Edge-length statistics over the edges of the given triangles (dead ids are ignored).
pub fn edge_length_stats_in(mesh: &Mesh, tris: &[u32]) -> EdgeLengthStats {
    // A few triangles of a large mesh (a brush dab) are deduplicated with a hash set rather
    // than marks over every edge id.
    let edges = first_seen_edges(mesh, tris, tris.len() * 16 < mesh.max_tid());
    EdgeLengthStats::from_lengths(edges.into_iter().map(|e| mesh.edge_length(e)))
}

/// Edges of the live triangles in `tris`, in the order they first appear; `small` picks a
/// hash set over id marks for the deduplication (same list either way).
fn first_seen_edges(mesh: &Mesh, tris: &[u32], small: bool) -> Vec<u32> {
    let mut edges = Vec::new();
    let live = tris.iter().copied().filter(|&t| mesh.is_triangle(t));
    if small {
        let mut seen: FastSet<u32> = fast_set_with_capacity(tris.len() * 2);
        for t in live {
            edges.extend(mesh.tri_edges(t).into_iter().filter(|&e| seen.insert(e)));
        }
    } else {
        let mut st = Stamp::default();
        st.begin(mesh.max_eid());
        for t in live {
            edges.extend(mesh.tri_edges(t).into_iter().filter(|&e| st.mark(e)));
        }
    }
    edges
}

/// [`edge_length_stats_in`] over all live triangles in id order, with the lengths gathered
/// in parallel: an edge first appears at the lower id of its two triangles, so the lengths
/// come in the same order and give the same sums.
pub(crate) fn edge_length_stats_of_triangles(mesh: &Mesh) -> EdgeLengthStats {
    let lengths: Vec<f64> = (0..mesh.max_tid() as u32)
        .into_par_iter()
        .with_min_len(crate::region::PAR_MIN_LEN)
        .filter(|&t| mesh.is_triangle(t))
        .flat_map_iter(|t| {
            mesh.tri_edges(t)
                .into_iter()
                .filter(move |&e| {
                    let o = mesh.edge_other_triangle(e, t);
                    o == INVALID || o > t || !mesh.is_triangle(o)
                })
                .map(|e| mesh.edge_length(e))
        })
        .collect();
    EdgeLengthStats::from_lengths(lengths.into_iter())
}

/// Triangle shape quality in `[0, 1]`: `4·√3·area / Σ edge²` (1 = equilateral).
pub fn triangle_quality(mesh: &Mesh, t: u32) -> f64 {
    let [a, b, c] = mesh.tri_positions(t);
    quality(a, b, c)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_core::{primitives, DVec3};

    fn bits(s: EdgeLengthStats) -> [u64; 5] {
        [s.count as u64, s.min.to_bits(), s.max.to_bits(), s.mean.to_bits(), s.std_dev.to_bits()]
    }

    #[test]
    fn stats_paths_agree_to_the_bit() {
        // An open, edited mesh: dead ids, reused ids and boundary edges.
        let mut m = primitives::plane_grid(3.0, 2.0, 30, 20);
        let mut k = 0u32;
        for e in m.edge_ids().collect::<Vec<_>>() {
            k += 1;
            if !m.is_edge(e) {
                continue;
            }
            let [a, b] = m.edge_vertices(e);
            if k.is_multiple_of(7) && m.can_collapse(a, b).is_ok() {
                m.collapse_edge(a, b).unwrap();
            } else if k.is_multiple_of(5) {
                m.split_edge(e).unwrap();
            } else if k.is_multiple_of(3) {
                let _ = m.flip_edge(e);
            }
        }
        m.check_validity().unwrap();
        for (i, v) in m.vertex_ids().collect::<Vec<_>>().into_iter().enumerate() {
            let p = m.vertex(v);
            m.set_vertex(v, p + DVec3::new(0.0, 0.0, 1e-3 * (i % 11) as f64));
        }
        let all: Vec<u32> = m.triangle_ids().collect();
        let some: Vec<u32> = all.iter().copied().filter(|t| t % 3 != 1).chain([u32::MAX - 1]).collect();
        for tris in [&all, &some] {
            assert_eq!(first_seen_edges(&m, tris, true), first_seen_edges(&m, tris, false));
        }
        let want = bits(EdgeLengthStats::from_lengths(first_seen_edges(&m, &all, false).into_iter().map(|e| m.edge_length(e))));
        assert_eq!(bits(edge_length_stats_of_triangles(&m)), want);
        assert_eq!(bits(edge_length_stats_in(&m, &all)), want);
    }
}
