//! Edge-length and triangle-quality statistics.

use crate::util::{quality, Stamp};
use mm_core::Mesh;

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
    let mut st = Stamp::default();
    st.begin(mesh.max_eid());
    let mut edges = Vec::new();
    for &t in tris {
        if !mesh.is_triangle(t) {
            continue;
        }
        for e in mesh.tri_edges(t) {
            if st.mark(e) {
                edges.push(e);
            }
        }
    }
    EdgeLengthStats::from_lengths(edges.into_iter().map(|e| mesh.edge_length(e)))
}

/// Triangle shape quality in `[0, 1]`: `4·√3·area / Σ edge²` (1 = equilateral).
pub fn triangle_quality(mesh: &Mesh, t: u32) -> f64 {
    let [a, b, c] = mesh.tri_positions(t);
    quality(a, b, c)
}
