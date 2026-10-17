mod common;

use common::*;
use mm_core::{primitives, Bvh, DVec3};
use mm_remesh::*;

#[test]
fn edge_stats_known_values() {
    let m = primitives::plane_grid(1.0, 1.0, 1, 1);
    let st = edge_length_stats(&m);
    assert_eq!(st.count, 5);
    assert!((st.min - 1.0).abs() < 1e-12);
    assert!((st.max - 2f64.sqrt()).abs() < 1e-12);
    assert!((st.mean - (4.0 + 2f64.sqrt()) / 5.0).abs() < 1e-12);
    let tris: Vec<u32> = m.triangle_ids().take(1).collect();
    assert_eq!(edge_length_stats_in(&m, &tris).count, 3);
    let q = triangle_quality(&m, tris[0]);
    assert!((q - 3f64.sqrt() / 2.0).abs() < 1e-12, "right isoceles quality {q}");
}

#[test]
fn refine_uniform_splits_long_edges_without_moving_the_surface() {
    let mut m = primitives::icosphere(1.0, 2);
    let orig = m.clone();
    let rep = refine_uniform(&mut m, 0.1, None, None).unwrap();
    assert_valid(&m);
    assert!(m.is_closed());
    assert!(rep.splits > 0);
    assert!(edge_length_stats(&m).max <= 0.1);
    for v in orig.vertex_ids() {
        assert_eq!(m.vertex(v), orig.vertex(v));
    }
    let bvh = Bvh::build(&orig);
    for v in m.vertex_ids() {
        assert!(dist_to(&orig, &bvh, m.vertex(v)) < 1e-12);
    }
}

#[test]
fn refine_uniform_region() {
    let mut m = primitives::plane_grid(2.0, 2.0, 8, 8);
    let region: Vec<u32> = m.triangle_ids().filter(|&t| m.tri_centroid(t).x < 0.0).collect();
    let before = snapshot_outside(&m, &region);
    let rep = refine_uniform(&mut m, 0.05, Some(&region), None).unwrap();
    assert_valid(&m);
    let nr = rep.region_triangles.unwrap();
    assert!(nr.len() > 10 * region.len());
    assert_eq!(snapshot_outside(&m, &nr), before);
    // Edges farther than one border-edge length (0.25) from the fixed border (x = 0)
    // are all short; closer ones grade towards the long border edges.
    let inr = tri_set(&m, &nr);
    let mut checked = 0;
    for e in m.edge_ids() {
        let [t0, _] = m.edge_triangles(e);
        if inr[t0 as usize] && m.edge_midpoint(e).x < -0.25 {
            assert!(m.edge_length(e) <= 0.05 + 1e-12, "{}", m.edge_length(e));
            checked += 1;
        }
    }
    assert!(checked > 1000);
    // No sliver cascade against the fixed border.
    assert!(min_quality(&m) > 0.5, "min quality {}", min_quality(&m));
}

#[test]
fn collapse_short_edges_cleans_tiny_edges() {
    let mut m = primitives::plane_grid(1.0, 1.0, 10, 10);
    let segs = boundary_segments(&m);
    let mut rng = Lcg(5);
    for _ in 0..150 {
        let e = loop {
            let e = rng.below(m.max_eid()) as u32;
            if m.is_edge(e) {
                break e;
            }
        };
        let s = [1e-6, 1e-3, 0.995][rng.below(3)];
        m.split_edge_at(e, s).unwrap();
    }
    assert_valid(&m);
    let min_len = 0.01;
    let short_before = m.edge_ids().filter(|&e| m.edge_length(e) < min_len).count();
    let rep = collapse_short_edges(&mut m, min_len, None, None).unwrap();
    assert_valid(&m);
    let short_after = m.edge_ids().filter(|&e| m.edge_length(e) < min_len).count();
    println!("short {short_before} -> {short_after}, {rep:?}");
    assert!(short_before > 50);
    assert!(short_after * 20 <= short_before, "{short_after} short edges remain");
    for v in boundary_vertices(&m) {
        assert!(dist_to_segments(m.vertex(v), &segs) < 1e-12);
    }
    for v in m.vertex_ids() {
        assert_eq!(m.vertex(v).z, 0.0);
    }
    // Corners of the square survive (they are the only vertices with two boundary
    // edges at 90 degrees and a collapse along the boundary keeps them on the boundary).
    let area: f64 = m.triangle_ids().map(|t| m.tri_area(t)).sum();
    assert!((area - 1.0).abs() < 1e-3, "area {area}");
    let _ = DVec3::ZERO;
}

#[test]
fn cleanup_parameters_validated() {
    let mut m = primitives::icosphere(1.0, 1);
    assert!(refine_uniform(&mut m, 0.0, None, None).is_err());
    assert!(collapse_short_edges(&mut m, f64::NAN, None, None).is_err());
}
