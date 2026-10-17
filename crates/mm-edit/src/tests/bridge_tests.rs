use super::*;
use crate::*;
use mm_core::DVec3;

fn two_tubes(n1: usize, n2: usize, gap: f64, tilt: f64) -> (Mesh, Vec<u32>, Vec<u32>) {
    let mut m = primitives::cylinder(0.5, 1.0, n1, 3, false);
    let mut t2 = primitives::cylinder(0.55, 1.0, n2, 3, false);
    let rot = mm_core::DQuat::from_rotation_y(tilt);
    t2.transform(|p| rot * p + DVec3::new(0.2, 0.1, 1.0 + gap));
    m.append_mesh(&t2, 1);
    let loops = boundary_loops(&m);
    let top = loops.iter().find(|l| l.vertices.iter().all(|&v| (m.vertex(v).z - 1.0).abs() < 1e-9)).unwrap();
    let mut rest: Vec<_> =
        loops.iter().filter(|l| l.vertices.iter().all(|&v| m.vertex(v).z > 1.0 + 1e-9)).cloned().collect();
    rest.sort_by(|a, b| a.centroid(&m).z.partial_cmp(&b.centroid(&m).z).unwrap());
    (m.clone(), top.vertices.clone(), rest[0].vertices.clone())
}

#[test]
fn join_loops_with_different_counts() {
    for (segments, smooth) in [(1usize, false), (0, true), (5, false)] {
        let (mut m, a, b) = two_tubes(16, 24, 0.8, 0.0);
        let r = join_loops(&mut m, &a, &b, &BridgeParams { segments, smooth }).unwrap();
        assert_valid(&m);
        assert_eq!(n_components(&m), 1);
        assert_eq!(n_loops(&m), 2);
        assert!(!r.triangles.is_empty());
        if segments == 5 {
            assert_eq!(r.new_vertices.len(), 4 * 24);
        }
        // Close the ends: the joined tube must enclose positive volume.
        let loops: Vec<Vec<u32>> = boundary_loops(&m).into_iter().map(|l| l.vertices).collect();
        cap_planar_loops(&mut m, &loops, None, CutFill::Flat).unwrap();
        assert_valid(&m);
        assert!(m.is_closed());
        assert!(volume(&m) > 0.0);
    }
    // Tilted second tube with a smooth bridge, loops passed reversed.
    let (mut m, a, b) = two_tubes(20, 12, 1.5, 0.5);
    let rb: Vec<u32> = b.iter().rev().copied().collect();
    join_loops(&mut m, &a, &rb, &BridgeParams { segments: 0, smooth: true }).unwrap();
    assert_valid(&m);
    assert_eq!(n_loops(&m), 2);
}

#[test]
fn bridge_spans_between_grids() {
    let mut m = primitives::plane_grid(1.0, 1.0, 4, 4);
    m.transform(|p| p - DVec3::X);
    let mut g2 = primitives::plane_grid(1.0, 1.2, 6, 6);
    g2.transform(|p| p + DVec3::X);
    m.append_mesh(&g2, 1);
    let mut a: Vec<u32> = m.vertex_ids().filter(|&v| (m.vertex(v).x + 0.5).abs() < 1e-9).collect();
    let mut b: Vec<u32> = m.vertex_ids().filter(|&v| (m.vertex(v).x - 0.5).abs() < 1e-9).collect();
    a.sort_by(|&x, &y| m.vertex(x).y.partial_cmp(&m.vertex(y).y).unwrap());
    b.sort_by(|&x, &y| m.vertex(y).y.partial_cmp(&m.vertex(x).y).unwrap());
    for segs in [1usize, 0] {
        let mut mm = m.clone();
        let r = bridge(&mut mm, &a, &b, &BridgeParams { segments: segs, smooth: false }).unwrap();
        assert_valid(&mm);
        assert_eq!(n_components(&mm), 1);
        assert_eq!(n_loops(&mm), 1);
        for &t in &r.triangles {
            assert!(mm.tri_normal(t).dot(DVec3::Z) > 0.999);
        }
    }
    // Overlapping spans are rejected.
    assert!(bridge(&mut m.clone(), &a, &a, &BridgeParams::default()).is_err());
}

#[test]
fn bridge_across_a_hole_splits_it() {
    // Square hole in a grid: bridge the two opposite sides of the hole.
    let mut m = primitives::plane_grid(3.0, 3.0, 6, 6);
    let hole: Vec<u32> = m
        .triangle_ids()
        .filter(|&t| {
            let c = m.tri_centroid(t);
            c.x.abs() < 1.0 && c.y.abs() < 0.5
        })
        .collect();
    discard(&mut m, &hole);
    assert_valid(&m);
    let near = |v: u32, x: f64| (m.vertex(v).x - x).abs() < 1e-9 && m.vertex(v).y.abs() < 0.5 + 1e-9;
    let inner = boundary_loops(&m).into_iter().find(|l| l.vertices.iter().all(|&v| m.vertex(v).x.abs() < 1.5 - 1e-9)).unwrap();
    let mut left: Vec<u32> = inner.vertices.iter().copied().filter(|&v| near(v, -1.0)).collect();
    let mut right: Vec<u32> = inner.vertices.iter().copied().filter(|&v| near(v, 1.0)).collect();
    left.sort_by(|&x, &y| m.vertex(x).y.partial_cmp(&m.vertex(y).y).unwrap());
    right.sort_by(|&x, &y| m.vertex(x).y.partial_cmp(&m.vertex(y).y).unwrap());
    let loops_before = n_loops(&m);
    // Bridge the middle parts of the hole's left and right sides (single edges).
    let l = vec![left[0], left[1]];
    let r = vec![right[1], right[0]];
    bridge(&mut m, &l, &r, &BridgeParams { segments: 4, smooth: false }).unwrap();
    assert_valid(&m);
    assert_eq!(n_loops(&m), loops_before + 1);
}
