// SPDX-License-Identifier: AGPL-3.0-only
//! Mean curvature at every vertex of a scan, for colouring it: where a flat turns into a
//! round and a round into a freeform blend is where the borders of surface regions are put,
//! and on a bare grey scan that line is guessed.
//!
//! The reading is the cotangent Laplacian's — the mesh's own discrete mean curvature normal,
//! Δx = −2H·n, over each vertex's own share of the surface, which holds on an uneven
//! triangulation where an average of the normal curvatures towards each neighbour does not —
//! signed by the vertex normal the graph already turned outward: positive on a convex round,
//! negative in a fillet's hollow, in 1/mm (a cylinder of radius R reads 1/2R, a sphere 1/R).
//!
//! A scan's noise is larger than its curvature at the scale of one triangle: 0.02 mm of
//! scatter on 0.3 mm edges reads as ±0.4 /mm, a round of 2 mm as 0.25. But the noise is a
//! vertex pushed out between neighbours pushed in — it all but cancels over a patch — so the
//! raw reading is averaged over the neighbours a few times, and what is left is the shape.
//!
//! Ported from ScanRuler (`src/core/geometry/curvature.ts`), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only. The values are kept in single precision as ScanRuler keeps them (a colour
//! map's input); `FieldScale` is the part of ScanRuler's colour-map module the scale needs,
//! not the colour maps themselves.

use super::MeshGraph;
use crate::fit3d::linalg3::{js_max, js_min};

/// Passes of neighbour averaging over the raw reading. Four reach about two rings, enough to
/// put a scan's scatter under a 2 mm round without washing the round into the flats beside
/// it.
pub const CURVATURE_SMOOTHING: usize = 4;

/// Mean curvature per vertex, 1/mm, convex positive; NaN on the rim of an open scan, where
/// the surface has no other side to be curved against, and on a vertex no triangle with an
/// area uses. `smoothing` passes of neighbour averaging follow the raw reading
/// (`CURVATURE_SMOOTHING` is ScanRuler's default).
pub fn mean_curvature(graph: &MeshGraph, smoothing: usize) -> Vec<f32> {
    let p = &graph.positions;
    let vertex_count = graph.vertex_count;
    // Σ (cot α + cot β)(x_w − x_v) over a vertex's edges, gathered a triangle at a time:
    // each corner's cotangent belongs to the edge across from it.
    let mut lap = vec![0.0f64; vertex_count * 3];
    let mut area = vec![0.0f64; vertex_count];
    for tri in graph.indices.as_chunks::<3>().0 {
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let (a, b, c) = (i0 * 3, i1 * 3, i2 * 3);
        let (pa, pb, pc) = (p[i0], p[i1], p[i2]);
        let (abx, aby, abz) = (pb.x - pa.x, pb.y - pa.y, pb.z - pa.z);
        let (acx, acy, acz) = (pc.x - pa.x, pc.y - pa.y, pc.z - pa.z);
        let (bcx, bcy, bcz) = (pc.x - pb.x, pc.y - pb.y, pc.z - pb.z);
        let nx = aby * acz - abz * acy;
        let ny = abz * acx - abx * acz;
        let nz = abx * acy - aby * acx;
        let twice = (nx * nx + ny * ny + nz * nz).sqrt();
        if !(twice > 1e-20) {
            continue;
        }
        // A sliver's cotangents run to thousands and one of them would own the vertex; held
        // to ±20 (an angle of 3°) it is one poor triangle among six.
        let cot = |dot: f64| js_max(-20.0, js_min(20.0, dot / twice));
        let cot_a = cot(abx * acx + aby * acy + abz * acz);
        let cot_b = cot(-(abx * bcx + aby * bcy + abz * bcz));
        let cot_c = cot(acx * bcx + acy * bcy + acz * bcz);
        // Edge bc lies across from a, ca from b, ab from c.
        lap[b] += cot_a * bcx;
        lap[b + 1] += cot_a * bcy;
        lap[b + 2] += cot_a * bcz;
        lap[c] -= cot_a * bcx;
        lap[c + 1] -= cot_a * bcy;
        lap[c + 2] -= cot_a * bcz;
        lap[a] += cot_b * acx;
        lap[a + 1] += cot_b * acy;
        lap[a + 2] += cot_b * acz;
        lap[c] -= cot_b * acx;
        lap[c + 1] -= cot_b * acy;
        lap[c + 2] -= cot_b * acz;
        lap[a] += cot_c * abx;
        lap[a + 1] += cot_c * aby;
        lap[a + 2] += cot_c * abz;
        lap[b] -= cot_c * abx;
        lap[b + 1] -= cot_c * aby;
        lap[b + 2] -= cot_c * abz;
        // Each corner's share of the triangle is what lies nearer to it than to the other two
        // (Meyer et al.'s mixed area) — a third each reads a ball 14 % too round at a vertex
        // five triangles meet at. The Voronoi share of an obtuse triangle falls outside it, so
        // that one is split by halves.
        let d_a = abx * acx + aby * acy + abz * acz;
        let d_b = -(abx * bcx + aby * bcy + abz * bcz);
        let d_c = acx * bcx + acy * bcy + acz * bcz;
        if d_a < 0.0 || d_b < 0.0 || d_c < 0.0 {
            let quarter = twice / 8.0;
            area[i0] += if d_a < 0.0 { 2.0 * quarter } else { quarter };
            area[i1] += if d_b < 0.0 { 2.0 * quarter } else { quarter };
            area[i2] += if d_c < 0.0 { 2.0 * quarter } else { quarter };
        } else {
            let ab2 = abx * abx + aby * aby + abz * abz;
            let ac2 = acx * acx + acy * acy + acz * acz;
            let bc2 = bcx * bcx + bcy * bcy + bcz * bcz;
            area[i0] += (ab2 * d_c + ac2 * d_b) / (8.0 * twice);
            area[i1] += (ab2 * d_c + bc2 * d_a) / (8.0 * twice);
            area[i2] += (ac2 * d_b + bc2 * d_a) / (8.0 * twice);
        }
    }

    let mut values = vec![0.0f32; vertex_count];
    for v in 0..vertex_count {
        // The adjacency lists a neighbour once per shared triangle: twice inside the
        // surface, once along its rim — so the ids of an inner vertex's neighbours cancel
        // under exclusive-or, and a rim vertex's do not.
        let x = graph.neighbours(v as u32).iter().fold(0u32, |acc, &w| acc ^ w);
        if x != 0 || !(area[v] > 1e-20) {
            values[v] = f32::NAN;
            continue;
        }
        let n = graph.normals[v];
        let along = lap[v * 3] * n.x + lap[v * 3 + 1] * n.y + lap[v * 3 + 2] * n.z;
        // Δx = Σ / 2A and H = −Δx·n / 2.
        values[v] = (-along / (4.0 * area[v])) as f32;
    }

    for _ in 0..smoothing {
        let mut next = vec![0.0f32; vertex_count];
        for v in 0..vertex_count {
            if values[v].is_nan() {
                next[v] = f32::NAN;
                continue;
            }
            let mut sum = f64::from(values[v]);
            let mut n = 1.0f64;
            for &w in graph.neighbours(v as u32) {
                let w = values[w as usize];
                if w.is_nan() {
                    continue;
                }
                sum += f64::from(w);
                n += 1.0;
            }
            next[v] = (sum / n) as f32;
        }
        values = next;
    }
    values
}

/// The scale a curvature map is read at: the curvature nine vertices in ten stay under, so
/// the flats and the gentle blends spread over the ramp and the sharp edges run off its ends
/// into the caps — an edge drawn dark is an edge found. Never under the curvature of a round
/// as large as the part (`size`, its bounding diagonal), which on a block of flats would
/// stretch the noise over the whole ramp.
pub fn curvature_range(values: &[f32], size: f64) -> f64 {
    // A sample is enough for a percentile, and a sort of two million is not what a key in the
    // view bar should cost.
    let step = (values.len() / 50_000).max(1);
    let mut sample: Vec<f64> = values.iter().step_by(step).map(|&x| f64::from(x).abs()).filter(|x| !x.is_nan()).collect();
    let floor = if size > 0.0 { 2.0 / size } else { 1e-3 };
    if sample.is_empty() {
        return floor;
    }
    sample.sort_by(f64::total_cmp);
    let at = (sample.len() - 1).min((sample.len() as f64 * 0.9).floor() as usize);
    js_max(floor, sample[at])
}

/// How a scalar field is read as colour: what the ends of the ramp mean, which way round it
/// runs, and what counts as a measurement at all.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FieldScale {
    /// Value at the bottom of the legend.
    pub low: f64,
    /// Value at the top.
    pub high: f64,
    /// Number of discrete colour bands, or `None` for a continuous ramp.
    pub bands: Option<u32>,
    /// Outside this window there is no measurement — bare material.
    pub valid_min: f64,
    pub valid_max: f64,
    /// Run the ramp from its high end down instead of from its low end up.
    pub reversed: bool,
}

/// How a curvature map is read as colour: about zero like a deviation, the flats in the
/// middle of the ramp, a convex round towards its high end and a hollow towards its low,
/// whatever is sharper than the range in the caps. The rim of an open scan is NaN and stays
/// bare.
pub fn curvature_scale(range: f64) -> FieldScale {
    FieldScale { low: -range, high: range, bands: None, valid_min: f64::NEG_INFINITY, valid_max: f64::INFINITY, reversed: false }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit3d::testkit::{box_mesh, cylinder_mesh, graph_indexed, graph_soup, icosphere, p32};

    fn median(values: &[f64]) -> f64 {
        let mut s = values.to_vec();
        s.sort_by(f64::total_cmp);
        s[s.len() / 2]
    }

    fn close(a: f64, b: f64, digits: i32) -> bool {
        (a - b).abs() < 10f64.powi(-digits) / 2.0
    }

    #[test]
    fn reads_one_over_r_on_a_ball_convex_positive_without_smoothing() {
        let g = graph_indexed(icosphere(4, 10.0, 0.0, 7));
        let h = mean_curvature(&g, 0);
        assert_eq!(h.len(), g.vertex_count);
        for (v, &k) in h.iter().enumerate() {
            assert!(close(f64::from(k), 0.1, 2), "{v}: {k}");
        }
    }

    #[test]
    fn reads_one_over_2r_on_a_cylinder_wall_and_nothing_on_its_caps() {
        let (soup, _) = cylinder_mesh(5.0, 30.0, 64, 24, 0.0, 11);
        let g = graph_soup(soup);
        let h = mean_curvature(&g, 0);
        let mut wall = Vec::new();
        let mut cap = Vec::new();
        for (p, &k) in g.positions.iter().zip(&h) {
            let r = p.x.hypot(p.y);
            if p.z.abs() < 13.0 && (r - 5.0).abs() < 1e-3 {
                wall.push(f64::from(k));
            }
            if (p.z.abs() - 15.0).abs() < 1e-3 && r < 4.0 {
                cap.push(f64::from(k));
            }
        }
        assert!(wall.len() > 100);
        for k in wall {
            assert!(close(k, 0.1, 2), "{k}");
        }
        for k in cap {
            assert!(k.abs() < 1e-6, "{k}");
        }
    }

    #[test]
    fn is_negative_inside_a_hollow_the_same_ball_with_its_normals_turned_in() {
        let mut g = graph_indexed(icosphere(3, 10.0, 0.0, 7));
        for n in &mut g.normals {
            *n = -*n;
        }
        let h = mean_curvature(&g, 0);
        assert!(close(f64::from(h[0]), -0.1, 2));
    }

    #[test]
    fn finds_a_noisy_ball_under_its_noise_once_smoothed_and_not_before() {
        // 0.02 mm of scatter on edges of about 0.7 mm: the raw reading is thrown by more than
        // the ball's own 0.1 /mm, the smoothed one is not.
        let g = graph_indexed(icosphere(5, 10.0, 0.02, 7));
        let raw: Vec<f64> = mean_curvature(&g, 0).iter().map(|&x| f64::from(x)).collect();
        let smooth: Vec<f64> = mean_curvature(&g, CURVATURE_SMOOTHING).iter().map(|&x| f64::from(x)).collect();
        let spread = |xs: &[f64]| median(&xs.iter().map(|x| (x - 0.1).abs()).collect::<Vec<_>>());
        assert!(spread(&raw) > 0.03);
        assert!(spread(&smooth) < 0.01);
        assert!(close(median(&smooth), 0.1, 2));
    }

    #[test]
    fn leaves_the_rim_of_an_open_surface_unread() {
        // A flat square of four triangles round a middle vertex: the middle is inside, the
        // four corners are rim.
        let positions = vec![p32(0.0, 0.0, 0.0), p32(1.0, 0.0, 0.0), p32(1.0, 1.0, 0.0), p32(0.0, 1.0, 0.0), p32(0.5, 0.5, 0.0)];
        let faces = vec![[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]];
        let g = graph_indexed((positions, faces));
        let h = mean_curvature(&g, 2);
        assert!(close(f64::from(h[4]), 0.0, 9));
        assert!(h[..4].iter().all(|k| k.is_nan()));
    }

    #[test]
    fn scales_a_block_of_flats_by_the_part_not_by_its_noise() {
        let g = graph_soup(box_mesh(40.0, 20, 0.005, 13));
        let h = mean_curvature(&g, CURVATURE_SMOOTHING);
        let size = 40.0 * 3f64.sqrt();
        assert!(curvature_range(&h, size) >= 2.0 / size);
        // A ball's own curvature is what nine vertices in ten stay under.
        let b = graph_indexed(icosphere(4, 10.0, 0.0, 7));
        assert!(close(curvature_range(&mean_curvature(&b, CURVATURE_SMOOTHING), 20.0), 0.1, 2));
        let scale = curvature_scale(0.25);
        assert_eq!((scale.low, scale.high, scale.bands), (-0.25, 0.25, None));
    }
}
