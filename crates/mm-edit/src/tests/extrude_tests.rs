use super::*;
use crate::*;
use mm_core::DVec3;

#[test]
fn extrude_sphere_patch() {
    let mut m = primitives::icosphere(1.0, 3);
    let v0 = volume(&m);
    let sel: Vec<u32> = m.triangle_ids().filter(|&t| m.tri_centroid(t).z > 0.7).collect();
    let r = extrude_selection(&mut m, &sel, &ExtrudeParams { distance: 0.2, ..Default::default() }).unwrap();
    assert_valid(&m);
    assert!(m.is_closed());
    assert!(!r.side_tris.is_empty());
    assert_eq!(r.new_region_tris.len(), sel.len());
    assert!(volume(&m) > v0 + 0.01);
    for &t in &r.new_region_tris {
        for v in m.triangle(t) {
            assert!((m.vertex(v).length() - 1.2).abs() < 0.02);
        }
    }
    // Side walls face outward (away from the z axis).
    for &t in &r.side_tris {
        let c = m.tri_centroid(t);
        assert!(m.tri_normal(t).dot(DVec3::new(c.x, c.y, 0.0)) > 0.0);
    }
    // Inward extrusion (pocket) with several rows.
    let mut p = primitives::icosphere(1.0, 3);
    let sel: Vec<u32> = p.triangle_ids().filter(|&t| p.tri_centroid(t).z > 0.7).collect();
    let r = extrude_selection(&mut p, &sel, &ExtrudeParams { distance: -0.3, side_segments: 3, ..Default::default() })
        .unwrap();
    assert_valid(&p);
    assert!(p.is_closed());
    assert!(volume(&p) < v0);
    assert_eq!(r.side_tris.len() % 3, 0);
}

#[test]
fn extrude_band_with_two_border_loops() {
    let mut m = primitives::icosphere(1.0, 3);
    let sel: Vec<u32> = m
        .triangle_ids()
        .filter(|&t| {
            let z = m.tri_centroid(t).z;
            z > 0.1 && z < 0.5
        })
        .collect();
    let v0 = volume(&m);
    extrude_selection(
        &mut m,
        &sel,
        &ExtrudeParams { distance: 0.15, direction: ExtrudeDirection::Normal, side_segments: 2, harden_sides: true },
    )
    .unwrap();
    assert_valid(&m);
    assert!(m.is_closed());
    assert!(volume(&m) > v0);
}

#[test]
fn extrude_grid_patch_touching_boundary() {
    let mut m = primitives::plane_grid(2.0, 2.0, 10, 10);
    let sel: Vec<u32> = m.triangle_ids().filter(|&t| m.tri_centroid(t).x > 0.45).collect();
    let r = extrude_selection(
        &mut m,
        &sel,
        &ExtrudeParams { distance: 0.3, direction: ExtrudeDirection::Fixed(DVec3::Z * 7.0), ..Default::default() },
    )
    .unwrap();
    assert_valid(&m);
    assert_eq!(n_loops(&m), 1);
    assert_eq!(n_components(&m), 1);
    for &t in &r.new_region_tris {
        for v in m.triangle(t) {
            assert!((m.vertex(v).z - 0.3).abs() < 1e-12);
        }
    }
    // The wall is a single strip from the bottom boundary to the top boundary.
    assert_eq!(r.side_tris.len(), 2 * 10);
}

#[test]
fn offset_selection_and_surface() {
    let mut m = primitives::icosphere(1.0, 3);
    let sel: Vec<u32> = m.triangle_ids().filter(|&t| m.tri_centroid(t).z > 0.5).collect();
    let border: Vec<u32> = mm_core::TriSelection::from_tris(&m, sel.iter().copied()).boundary_vertices(&m);
    let before: Vec<DVec3> = border.iter().map(|&v| m.vertex(v)).collect();
    let n = offset_selection(&mut m, &sel, 0.1, 0.3).unwrap();
    assert!(n > 0);
    assert_valid(&m);
    for (i, &v) in border.iter().enumerate() {
        assert_eq!(m.vertex(v), before[i], "border stays fixed");
    }
    let top = m.vertex_ids().map(|v| m.vertex(v).z).fold(f64::NEG_INFINITY, f64::max);
    assert!((top - 1.1).abs() < 0.01);

    let mut s = primitives::icosphere(1.0, 2);
    offset_surface(&mut s, 0.25);
    assert_valid(&s);
    for v in s.vertex_ids() {
        assert!((s.vertex(v).length() - 1.25).abs() < 1e-3);
    }
}

#[test]
fn random_extrudes_and_extensions() {
    let mut seed = 99u64;
    let mut rnd = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    for i in 0..30 {
        let mut m = primitives::icosphere(1.0, 3);
        let ids: Vec<u32> = m.triangle_ids().collect();
        let t0 = ids[(rnd() * ids.len() as f64) as usize % ids.len()];
        let c = m.tri_centroid(t0);
        let r = 0.2 + 0.8 * rnd();
        // Region: triangles within r of the seed centroid (may be multiply connected).
        let sel: Vec<u32> = ids.iter().copied().filter(|&t| m.tri_centroid(t).distance(c) < r).collect();
        let dist = if i % 2 == 0 { 0.05 + 0.3 * rnd() } else { -(0.05 + 0.2 * rnd()) };
        let dir = if i % 3 == 0 { ExtrudeDirection::Fixed(c) } else { ExtrudeDirection::Normal };
        let p = ExtrudeParams { distance: dist, direction: dir, side_segments: 1 + i % 3, harden_sides: i % 2 == 0 };
        extrude_selection(&mut m, &sel, &p).unwrap();
        assert_valid(&m);
        assert!(m.is_closed());
    }
    for i in 0..20 {
        let mut m = primitives::cylinder(0.5, 3.0, 16 + i, 10, false);
        let n = DVec3::new(rnd() - 0.5, rnd() - 0.5, 1.5).normalize();
        let plane = mm_core::Plane::from_point_normal(DVec3::new(0.0, 0.0, 1.0 + rnd()), n);
        let r = plane_cut(&mut m, plane, &PlaneCutParams { keep: KeepSide::Below, fill: CutFill::None, ..Default::default() })
            .unwrap();
        let lp = r.cut_loops.iter().find(|l| l.iter().all(|&v| plane.signed_distance(m.vertex(v)).abs() < 1e-9)).unwrap();
        let fe = extrude_boundary_loop(
            &mut m,
            lp,
            &FlowExtensionParams {
                length: ExtensionLength::Diameters(1.0 + 3.0 * rnd()),
                make_circular: i % 2 == 0,
                cap: true,
                cap_remesh: i % 3 != 0,
                ..Default::default()
            },
        )
        .unwrap();
        assert_valid(&m);
        assert!(fe.capped);
        assert!(fe.direction.dot(n) > 0.999999, "extension follows the cut plane normal outward");
        let rest: Vec<Vec<u32>> = boundary_loops(&m).into_iter().map(|l| l.vertices).collect();
        cap_planar_loops(&mut m, &rest, None, CutFill::Flat).unwrap();
        assert_valid(&m);
        assert!(m.is_closed());
        assert!(volume(&m) > 0.0);
    }
}
