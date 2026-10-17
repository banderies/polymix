mod common;

use common::*;
use mm_core::{measure, primitives, Bvh, DVec3, Mesh};
use mm_remesh::*;
use std::cell::Cell;

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs()
}

#[test]
fn icosphere_hits_target_and_stays_on_sphere() {
    let mut m = primitives::icosphere(1.0, 3);
    let v0 = measure::volume(&m);
    let target = 0.08;
    let rep = remesh(&mut m, &RemeshParams::with_target(target), None).unwrap();
    assert_valid(&m);
    assert!(m.is_closed());
    let st = edge_length_stats(&m);
    println!("{rep:?}\nmean q {:.3} min q {:.3}", mean_quality(&m), min_quality(&m));
    assert!(rel(st.mean, target) < 0.15, "mean edge {} vs {target}", st.mean);
    let max_rad = m.vertex_ids().map(|v| (m.vertex(v).length() - 1.0).abs()).fold(0.0, f64::max);
    assert!(max_rad < 0.01, "radial deviation {max_rad}");
    assert!(rel(measure::volume(&m), v0) < 0.01, "volume {} vs {v0}", measure::volume(&m));
    assert!(mean_quality(&m) > 0.85);
    assert!(min_quality(&m) > 0.3);
    assert_eq!(rep.final_triangles, m.triangle_count());
    assert!(rep.edge_lengths.count > 0);
}

#[test]
fn coarse_uv_sphere_is_regularised() {
    let mut m = primitives::uv_sphere(1.0, 16, 8);
    let orig = m.clone();
    let v0 = measure::volume(&m);
    let target = 0.12;
    remesh(&mut m, &RemeshParams::with_target(target), None).unwrap();
    assert_valid(&m);
    assert!(m.is_closed());
    let st = edge_length_stats(&m);
    println!("{st:?} mean q {:.3} min q {:.3}", mean_quality(&m), min_quality(&m));
    assert!(rel(st.mean, target) < 0.15, "mean edge {}", st.mean);
    let bvh = Bvh::build(&orig);
    let max_d = m.vertex_ids().map(|v| dist_to(&orig, &bvh, m.vertex(v))).fold(0.0, f64::max);
    assert!(max_d < 1e-9, "vertices must lie on the input surface ({max_d})");
    let max_rad = m.vertex_ids().map(|v| (m.vertex(v).length() - 1.0).abs()).fold(0.0, f64::max);
    assert!(max_rad < 0.06, "radial deviation {max_rad}");
    assert!(rel(measure::volume(&m), v0) < 0.02, "volume {} vs {v0}", measure::volume(&m));
    // The 16-valent poles must be gone.
    let max_val = m.vertex_ids().map(|v| m.vertex_valence(v)).max().unwrap();
    assert!(max_val <= 9, "max valence {max_val}");
    assert!(mean_quality(&m) > 0.8);
}

#[test]
fn auto_target_uses_mean_edge_length() {
    let mut m = primitives::icosphere(2.0, 3);
    let mean0 = edge_length_stats(&m).mean;
    let rep = remesh(&mut m, &RemeshParams::default(), None).unwrap();
    assert_valid(&m);
    assert!((rep.target_edge_length - mean0).abs() < 1e-12);
    assert!(rel(rep.edge_lengths.mean, mean0) < 0.15);
}

#[test]
fn open_cylinder_fixed_boundary() {
    let mut m = primitives::cylinder(1.0, 3.0, 24, 6, false);
    let bverts: Vec<(u32, DVec3)> = boundary_vertices(&m).into_iter().map(|v| (v, m.vertex(v))).collect();
    let nbe = m.boundary_edge_ids().count();
    let rep = remesh(&mut m, &RemeshParams { boundary: BoundaryMode::Fixed, ..RemeshParams::with_target(0.15) }, None).unwrap();
    assert_valid(&m);
    println!("{rep:?}");
    for &(v, p) in &bverts {
        assert!(m.is_vertex(v));
        assert_eq!(m.vertex(v), p, "fixed boundary vertex moved");
    }
    assert_eq!(boundary_vertices(&m).len(), bverts.len());
    assert_eq!(m.boundary_edge_ids().count(), nbe);
    // Interior refined and kept on the cylinder, without slivers along the fixed rim.
    assert!(m.triangle_count() > 600);
    assert!(min_quality(&m) > 0.4, "min quality {}", min_quality(&m));
    for v in m.vertex_ids() {
        let p = m.vertex(v);
        let r = (p.x * p.x + p.y * p.y).sqrt();
        assert!(r <= 1.0 + 1e-9 && r > 0.98, "r {r}");
    }
}

#[test]
fn open_cylinder_constrained_boundary() {
    let mut m = primitives::cylinder(1.0, 3.0, 24, 6, false);
    let segs = boundary_segments(&m);
    let nbe = m.boundary_edge_ids().count();
    let target = 0.12;
    let rep = remesh(&mut m, &RemeshParams { boundary: BoundaryMode::ConstrainToBoundary, ..RemeshParams::with_target(target) }, None)
        .unwrap();
    assert_valid(&m);
    println!("{rep:?}");
    let bv = boundary_vertices(&m);
    for &v in &bv {
        let d = dist_to_segments(m.vertex(v), &segs);
        assert!(d < 1e-9, "boundary vertex left the boundary curve by {d}");
        let z = m.vertex(v).z;
        assert!(z.abs() < 1e-9 || (z - 3.0).abs() < 1e-9);
    }
    assert!(m.boundary_edge_ids().count() > nbe, "boundary should be refined");
    assert_eq!(mm_core::topology::boundary_loops(&m).len(), 2);
    let st = edge_length_stats(&m);
    assert!(rel(st.mean, target) < 0.15, "mean {}", st.mean);
    // boundary edge lengths are regularised too
    let bl: Vec<f64> = m.boundary_edge_ids().map(|e| m.edge_length(e)).collect();
    let bmean = bl.iter().sum::<f64>() / bl.len() as f64;
    assert!(rel(bmean, target) < 0.3, "boundary mean {bmean}");
}

#[test]
fn open_cylinder_free_boundary() {
    let mut m = primitives::cylinder(1.0, 3.0, 24, 6, false);
    let orig = m.clone();
    remesh(&mut m, &RemeshParams { boundary: BoundaryMode::Free, ..RemeshParams::with_target(0.15) }, None).unwrap();
    assert_valid(&m);
    assert_eq!(mm_core::topology::boundary_loops(&m).len(), 2);
    let bvh = Bvh::build(&orig);
    for v in m.vertex_ids() {
        assert!(dist_to(&orig, &bvh, m.vertex(v)) < 1e-6);
    }
}

#[test]
fn cube_keeps_sharp_edges_and_corners() {
    let mut m = primitives::cube(2.0, 3);
    let rep = remesh(
        &mut m,
        &RemeshParams { preserve_sharp_edges: Some(30.0), ..RemeshParams::with_target(0.2) },
        None,
    )
    .unwrap();
    assert_valid(&m);
    println!("{rep:?} mean q {:.3} min q {:.3}", mean_quality(&m), min_quality(&m));
    assert!(m.triangle_count() > 500);
    // Every triangle still lies in one face plane.
    for t in m.triangle_ids() {
        let n = m.tri_normal(t);
        assert!(n.abs().max_element() > 1.0 - 1e-9, "triangle {t} straddles a crease: {n:?}");
    }
    // All vertices on the cube surface.
    for v in m.vertex_ids() {
        let p = m.vertex(v);
        assert!((p.abs().max_element() - 1.0).abs() < 1e-9, "{p:?}");
    }
    // Corners present.
    for sx in [-1.0, 1.0] {
        for sy in [-1.0, 1.0] {
            for sz in [-1.0, 1.0] {
                let c = DVec3::new(sx, sy, sz);
                assert!(m.vertex_ids().any(|v| m.vertex(v) == c), "corner {c:?} lost");
            }
        }
    }
    assert!((measure::volume(&m) - 8.0).abs() < 1e-9);
    let st = edge_length_stats(&m);
    assert!(rel(st.mean, 0.2) < 0.15, "mean {}", st.mean);
}

#[test]
fn cube_without_feature_preservation_still_valid() {
    let mut m = primitives::cube(2.0, 3);
    remesh(&mut m, &RemeshParams::with_target(0.2), None).unwrap();
    assert_valid(&m);
    assert!(m.is_closed());
}

#[test]
fn region_remesh_leaves_outside_bit_identical() {
    let mut m = primitives::icosphere(1.0, 4);
    let region: Vec<u32> = m.triangle_ids().filter(|&t| m.tri_centroid(t).z > 0.3).collect();
    let before = snapshot_outside(&m, &region);
    let mean0 = edge_length_stats(&m).mean;
    let target = 0.5 * mean0;
    let rep = remesh(&mut m, &RemeshParams { region: Some(region.clone()), ..RemeshParams::with_target(target) }, None).unwrap();
    assert_valid(&m);
    assert!(m.is_closed());
    let new_region = rep.region_triangles.clone().unwrap();
    let inr = tri_set(&m, &new_region);
    let outside: Vec<u32> = m.triangle_ids().filter(|&t| !inr[t as usize]).collect();
    assert_eq!(outside.len() + new_region.len(), m.triangle_count());
    let after = snapshot_outside(&m, &new_region);
    assert_eq!(before, after, "outside triangles changed");
    assert!(new_region.len() > 2 * region.len(), "region should be refined: {} -> {}", region.len(), new_region.len());
    let st = edge_length_stats_in(&m, &new_region);
    println!("{rep:?}");
    // Border edges keep the old length, so allow a bit more slack.
    assert!(rel(st.mean, target) < 0.2, "region mean {} target {target}", st.mean);
    for &t in &new_region {
        for v in m.triangle(t) {
            assert!((m.vertex(v).length() - 1.0).abs() < 0.01);
        }
    }
    let qmin = new_region.iter().map(|&t| triangle_quality(&m, t)).fold(1.0, f64::min);
    assert!(qmin > 0.4, "region min quality {qmin}");
}

#[test]
fn region_coarsening_and_open_region() {
    // Coarsen a patch of a plane grid that touches the mesh boundary.
    let mut m = primitives::plane_grid(4.0, 4.0, 40, 40);
    let region: Vec<u32> = m.triangle_ids().filter(|&t| m.tri_centroid(t).x > 0.5).collect();
    let before = snapshot_outside(&m, &region);
    let rep = remesh(
        &mut m,
        &RemeshParams { region: Some(region.clone()), boundary: BoundaryMode::ConstrainToBoundary, ..RemeshParams::with_target(0.3) },
        None,
    )
    .unwrap();
    assert_valid(&m);
    let new_region = rep.region_triangles.unwrap();
    assert!(new_region.len() < region.len() / 2);
    assert_eq!(snapshot_outside(&m, &new_region), before);
    for v in m.vertex_ids() {
        assert!(m.vertex(v).z.abs() < 1e-12);
    }
}

#[test]
fn group_boundaries_are_preserved() {
    let mut m = primitives::icosphere(1.0, 3);
    let tris: Vec<u32> = m.triangle_ids().collect();
    for &t in &tris {
        let g = if m.tri_centroid(t).x > 0.1 { 1 } else { 0 };
        m.set_tri_group(t, g);
    }
    let border = |m: &Mesh| -> Vec<(DVec3, DVec3)> {
        m.edge_ids()
            .filter(|&e| {
                let [t0, t1] = m.edge_triangles(e);
                m.tri_group(t0) != m.tri_group(t1)
            })
            .map(|e| {
                let [a, b] = m.edge_vertices(e);
                (m.vertex(a), m.vertex(b))
            })
            .collect()
    };
    let segs = border(&m);
    assert!(!segs.is_empty());
    remesh(&mut m, &RemeshParams { preserve_group_boundaries: true, ..RemeshParams::with_target(0.07) }, None).unwrap();
    assert_valid(&m);
    let after = border(&m);
    assert!(after.len() > segs.len(), "group border should be refined too");
    for (a, b) in after {
        assert!(dist_to_segments(a, &segs) < 1e-9 && dist_to_segments(b, &segs) < 1e-9);
    }
    // Each group stays one connected patch.
    for g in [0, 1] {
        let gt: Vec<u32> = m.triangle_ids().filter(|&t| m.tri_group(t) == g).collect();
        let comp = mm_core::topology::flood_fill_triangles(&m, &gt[..1], |t| m.tri_group(t) == g);
        assert_eq!(comp.len(), gt.len());
    }
}

#[test]
fn adaptive_sizing_follows_curvature() {
    // A tube that narrows from radius 1 to 0.25.
    let n = 20;
    let path: Vec<DVec3> = (0..=n).map(|i| DVec3::new(0.0, 0.0, 6.0 * i as f64 / n as f64)).collect();
    let radii: Vec<f64> = (0..=n).map(|i| 1.0 - 0.75 * i as f64 / n as f64).collect();
    let mut m = primitives::tube_along(&path, &radii, 48);
    let rep = remesh(
        &mut m,
        &RemeshParams { adaptive: true, boundary: BoundaryMode::ConstrainToBoundary, ..RemeshParams::with_target(0.2) },
        None,
    )
    .unwrap();
    assert_valid(&m);
    println!("{rep:?}");
    let mean_len = |zlo: f64, zhi: f64| {
        let ls: Vec<f64> = m
            .edge_ids()
            .filter(|&e| {
                let z = m.edge_midpoint(e).z;
                z > zlo && z < zhi
            })
            .map(|e| m.edge_length(e))
            .collect();
        ls.iter().sum::<f64>() / ls.len() as f64
    };
    let thick = mean_len(0.5, 1.5);
    let thin = mean_len(4.5, 5.5);
    println!("thick {thick} thin {thin}");
    assert!(thin < 0.75 * thick, "thin end should be finer: thick {thick} thin {thin}");
    assert!(thick > 0.2 * 0.5 && thick < 0.2 * 2.5);
}

#[test]
fn no_projection_and_no_smoothing_modes() {
    let mut m = primitives::icosphere(1.0, 2);
    remesh(&mut m, &RemeshParams { project_to_original: false, ..RemeshParams::with_target(0.15) }, None).unwrap();
    assert_valid(&m);
    let mut m = primitives::icosphere(1.0, 2);
    remesh(&mut m, &RemeshParams { smoothing: 0.0, ..RemeshParams::with_target(0.15) }, None).unwrap();
    assert_valid(&m);
}

#[test]
fn cancellation_leaves_valid_mesh() {
    let mut m = primitives::icosphere(1.0, 4);
    let calls = Cell::new(0);
    let cb = |_f: f32| {
        calls.set(calls.get() + 1);
        calls.get() < 3
    };
    let rep = remesh(&mut m, &RemeshParams::with_target(0.03), Some(&cb)).unwrap();
    assert!(rep.cancelled);
    assert!(rep.iterations < 10);
    assert_valid(&m);
    assert!(m.is_closed());
}

#[test]
fn progress_is_monotone_and_reaches_one() {
    let mut m = primitives::icosphere(1.0, 3);
    let last = Cell::new(-1.0f32);
    let cb = |f: f32| {
        assert!(f >= last.get() - 1e-6, "progress went backwards");
        last.set(f);
        true
    };
    remesh(&mut m, &RemeshParams { iterations: 3, ..RemeshParams::with_target(0.1) }, Some(&cb)).unwrap();
    assert_eq!(last.get(), 1.0);
}

#[test]
fn invalid_parameters_rejected() {
    let mut m = primitives::icosphere(1.0, 1);
    assert!(remesh(&mut m, &RemeshParams { smoothing: f64::NAN, ..Default::default() }, None).is_err());
    assert!(remesh(&mut m, &RemeshParams { preserve_sharp_edges: Some(-1.0), ..Default::default() }, None).is_err());
    // Empty region is a no-op.
    let rep = remesh(&mut m, &RemeshParams { region: Some(vec![]), ..Default::default() }, None).unwrap();
    assert_eq!(rep.region_triangles, Some(vec![]));
    assert_valid(&m);
}
