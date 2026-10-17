mod common;

use common::*;
use mm_core::{measure, primitives, DVec3, Mesh};
use mm_remesh::*;
use std::cell::Cell;

fn euler(m: &Mesh) -> i64 {
    m.vertex_count() as i64 - m.edge_count() as i64 + m.triangle_count() as i64
}

fn bumpy_grid(n: usize) -> Mesh {
    let mut m = primitives::plane_grid(2.0, 2.0, n, n);
    m.transform(|p| DVec3::new(p.x, p.y, 0.1 * (3.0 * p.x).sin() * (3.0 * p.y).cos()));
    m
}

#[test]
fn icosphere_to_triangle_count() {
    let mut m = primitives::icosphere(1.0, 5);
    let v0 = measure::volume(&m);
    let rep = reduce(&mut m, &ReduceParams::with_target(ReduceTarget::TriangleCount(2000)), None).unwrap();
    println!("{rep:?} minq {:.3} meanq {:.3}", min_quality(&m), mean_quality(&m));
    assert_valid(&m);
    assert!(m.is_closed());
    assert_eq!(euler(&m), 2);
    assert!(m.triangle_count() <= 2000 && m.triangle_count() >= 1998, "{}", m.triangle_count());
    assert_eq!(rep.final_triangles, m.triangle_count());
    assert!(((measure::volume(&m) - v0) / v0).abs() < 0.01, "volume {} vs {v0}", measure::volume(&m));
    let max_rad = m.vertex_ids().map(|v| (m.vertex(v).length() - 1.0).abs()).fold(0.0, f64::max);
    assert!(max_rad < 0.01, "radial {max_rad}");
    assert!(min_quality(&m) > 0.05);
    assert!(mean_quality(&m) > 0.6);
}

#[test]
fn percentage_target() {
    let mut m = primitives::icosphere(1.0, 4);
    let n0 = m.triangle_count();
    let rep = reduce(&mut m, &ReduceParams::with_target(ReduceTarget::Percentage(25.0)), None).unwrap();
    assert_valid(&m);
    let want = n0 / 4;
    assert!((m.triangle_count() as i64 - want as i64).abs() <= 2, "{} vs {want}", m.triangle_count());
    assert_eq!(rep.initial_triangles, n0);
    assert!(m.is_closed());
}

#[test]
fn aggressive_reduction_stays_valid_and_closed() {
    let mut m = primitives::icosphere(1.0, 4);
    reduce(&mut m, &ReduceParams::with_target(ReduceTarget::TriangleCount(40)), None).unwrap();
    assert_valid(&m);
    assert!(m.is_closed());
    assert_eq!(euler(&m), 2);
    assert!(m.triangle_count() <= 60, "{}", m.triangle_count());
    assert!(measure::volume(&m) > 2.0);
}

#[test]
fn preserve_boundary_keeps_boundary_exactly() {
    let mut m = bumpy_grid(40);
    let bverts: Vec<(u32, DVec3)> = boundary_vertices(&m).into_iter().map(|v| (v, m.vertex(v))).collect();
    let nbe = m.boundary_edge_ids().count();
    let target = m.triangle_count() / 5;
    let rep = reduce(
        &mut m,
        &ReduceParams { preserve_boundary: true, ..ReduceParams::with_target(ReduceTarget::TriangleCount(target)) },
        None,
    )
    .unwrap();
    println!("{rep:?}");
    assert_valid(&m);
    for &(v, p) in &bverts {
        assert!(m.is_vertex(v));
        assert_eq!(m.vertex(v), p);
    }
    assert_eq!(m.boundary_edge_ids().count(), nbe);
    assert!(m.triangle_count() <= target + 1 && m.triangle_count() + 2 >= target, "{}", m.triangle_count());
}

#[test]
fn free_boundary_is_simplified_but_keeps_its_shape() {
    let mut m = primitives::cylinder(1.0, 2.0, 64, 16, false);
    let segs = boundary_segments(&m);
    let nbe = m.boundary_edge_ids().count();
    reduce(
        &mut m,
        &ReduceParams { preserve_boundary: false, ..ReduceParams::with_target(ReduceTarget::TriangleCount(100)) },
        None,
    )
    .unwrap();
    assert_valid(&m);
    assert!(m.boundary_edge_ids().count() < nbe, "boundary should be simplified");
    assert_eq!(mm_core::topology::boundary_loops(&m).len(), 2);
    for v in boundary_vertices(&m) {
        let p = m.vertex(v);
        assert!(p.z.abs() < 1e-6 || (p.z - 2.0).abs() < 1e-6, "boundary left its plane: {p:?}");
        assert!(dist_to_segments(p, &segs) < 0.02, "boundary vertex moved off the rim: {p:?}");
    }
}

#[test]
fn max_deviation_is_respected_on_sphere() {
    let mut m = primitives::icosphere(1.0, 5);
    let orig = m.clone();
    let tol = 0.005;
    let rep = reduce(&mut m, &ReduceParams::with_target(ReduceTarget::MaxDeviation(tol)), None).unwrap();
    println!("{rep:?}");
    assert_valid(&m);
    assert!(m.is_closed());
    assert!(m.triangle_count() < orig.triangle_count() / 4, "too little reduction: {}", m.triangle_count());
    let fwd = hausdorff_one_sided(&orig, &m, 2, 1);
    let bwd = hausdorff_one_sided(&m, &orig, 4, 2);
    println!("forward {fwd} backward {bwd}");
    assert!(fwd <= tol, "original -> reduced {fwd}");
    assert!(bwd <= tol * 1.05, "reduced -> original {bwd}");
    assert!(rep.max_deviation.unwrap() <= tol * 0.95 + 1e-12);
}

#[test]
fn max_deviation_on_open_bumpy_surface() {
    let mut m = bumpy_grid(60);
    let orig = m.clone();
    let tol = 0.002;
    let rep = reduce(
        &mut m,
        &ReduceParams { preserve_boundary: false, ..ReduceParams::with_target(ReduceTarget::MaxDeviation(tol)) },
        None,
    )
    .unwrap();
    println!("{rep:?}");
    assert_valid(&m);
    assert!(m.triangle_count() < orig.triangle_count() / 3, "{}", m.triangle_count());
    let fwd = hausdorff_one_sided(&orig, &m, 2, 3);
    let bwd = hausdorff_one_sided(&m, &orig, 4, 4);
    println!("forward {fwd} backward {bwd}");
    assert!(fwd <= tol, "original -> reduced {fwd}");
    assert!(bwd <= tol * 1.05, "reduced -> original {bwd}");
}

#[test]
fn region_reduce_leaves_outside_bit_identical() {
    let mut m = primitives::icosphere(1.0, 4);
    let region: Vec<u32> = m.triangle_ids().filter(|&t| m.tri_centroid(t).z > 0.0).collect();
    let before = snapshot_outside(&m, &region);
    let rep = reduce(
        &mut m,
        &ReduceParams { region: Some(region.clone()), ..ReduceParams::with_target(ReduceTarget::Percentage(30.0)) },
        None,
    )
    .unwrap();
    assert_valid(&m);
    assert!(m.is_closed());
    let new_region = rep.region_triangles.clone().unwrap();
    tri_set(&m, &new_region);
    assert_eq!(snapshot_outside(&m, &new_region), before);
    let want = (region.len() as f64 * 0.3).round() as i64;
    assert!((new_region.len() as i64 - want).abs() <= 2, "{} vs {want}", new_region.len());
    assert_eq!(rep.final_triangles, new_region.len());
}

#[test]
fn group_boundaries_survive_reduction() {
    let mut m = primitives::icosphere(1.0, 4);
    for t in m.triangle_ids().collect::<Vec<_>>() {
        let c = m.tri_centroid(t);
        m.set_tri_group(t, if c.x + 0.3 * c.y > 0.1 { 1 } else { 0 });
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
    reduce(&mut m, &ReduceParams::with_target(ReduceTarget::Percentage(15.0)), None).unwrap();
    assert_valid(&m);
    let after = border(&m);
    assert!(!after.is_empty() && after.len() < segs.len());
    for (a, b) in after {
        assert!(dist_to_segments(a, &segs) < 1e-9 && dist_to_segments(b, &segs) < 1e-9);
    }
    for g in [0, 1] {
        let gt: Vec<u32> = m.triangle_ids().filter(|&t| m.tri_group(t) == g).collect();
        let comp = mm_core::topology::flood_fill_triangles(&m, &gt[..1], |t| m.tri_group(t) == g);
        assert_eq!(comp.len(), gt.len());
    }
}

#[test]
fn flat_regions_reduce_to_few_triangles() {
    let mut m = primitives::cube(2.0, 10);
    reduce(&mut m, &ReduceParams::with_target(ReduceTarget::TriangleCount(12)), None).unwrap();
    assert_valid(&m);
    assert!(m.is_closed());
    assert!(m.triangle_count() <= 40, "{}", m.triangle_count());
    assert!((measure::volume(&m) - 8.0).abs() < 1e-6, "volume {}", measure::volume(&m));
}

#[test]
fn cancellation_leaves_valid_mesh() {
    let mut m = primitives::icosphere(1.0, 6);
    let calls = Cell::new(0);
    let cb = |_f: f32| {
        calls.set(calls.get() + 1);
        calls.get() < 3
    };
    let rep = reduce(&mut m, &ReduceParams::with_target(ReduceTarget::TriangleCount(100)), Some(&cb)).unwrap();
    assert!(rep.cancelled);
    assert!(m.triangle_count() > 100);
    assert_valid(&m);
    assert!(m.is_closed());
}

#[test]
fn invalid_parameters_rejected() {
    let mut m = primitives::icosphere(1.0, 1);
    assert!(reduce(&mut m, &ReduceParams::with_target(ReduceTarget::Percentage(0.0)), None).is_err());
    assert!(reduce(&mut m, &ReduceParams::with_target(ReduceTarget::Percentage(150.0)), None).is_err());
    assert!(reduce(&mut m, &ReduceParams::with_target(ReduceTarget::MaxDeviation(-1.0)), None).is_err());
    let n = m.triangle_count();
    let rep = reduce(&mut m, &ReduceParams::with_target(ReduceTarget::TriangleCount(n + 10)), None).unwrap();
    assert_eq!(rep.collapses, 0);
    assert_valid(&m);
}
