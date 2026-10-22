use super::*;
use crate::*;
use mm_core::{DVec3, Plane};

/// Open tube along z from 0 to 2, trimmed at z = 1.5 (uncapped): returns mesh and cut loop.
fn cut_tube(slices: usize, xscale: f64) -> (Mesh, Vec<u32>) {
    let mut m = primitives::cylinder(0.5, 2.0, slices, 8, false);
    m.transform(|p| DVec3::new(p.x * xscale, p.y, p.z));
    let r = plane_cut(
        &mut m,
        Plane::from_point_normal(DVec3::new(0.0, 0.0, 1.5), DVec3::Z),
        &PlaneCutParams { keep: KeepSide::Below, fill: CutFill::None, ..Default::default() },
    )
    .unwrap();
    assert_valid(&m);
    assert_eq!(r.cut_loops.len(), 1);
    (m, r.cut_loops[0].clone())
}

#[test]
fn flow_extension_on_cut_tube() {
    let (mut m, lp) = cut_tube(24, 1.0);
    let params = FlowExtensionParams {
        length: ExtensionLength::Diameters(2.0),
        cap: true,
        cap_remesh: false,
        ..Default::default()
    };
    // Loop given in reverse order must work too.
    let rev: Vec<u32> = lp.iter().rev().copied().collect();
    let r = extrude_boundary_loop(&mut m, &rev, &params).unwrap();
    assert_valid(&m);
    assert!(r.capped);
    assert_eq!(n_loops(&m), 1, "only the far (z = 0) end stays open");
    assert!(r.direction.dot(DVec3::Z) > 0.999999, "direction must point out of the tube");
    let d_eq = r.equivalent_diameter;
    let poly_area = 0.5 * 24.0 * 0.25 * (TAU / 24.0).sin();
    assert!((d_eq - 2.0 * (poly_area / PI).sqrt()).abs() < 1e-9);
    assert!((r.length - 2.0 * d_eq).abs() < 1e-12);
    let zmax = m.vertex_ids().map(|v| m.vertex(v).z).fold(f64::NEG_INFINITY, f64::max);
    assert!((zmax - (1.5 + 2.0 * d_eq)).abs() < 1e-9);
    for &v in &r.end_loop {
        assert!((m.vertex(v).z - zmax).abs() < 1e-9);
    }
    for &t in &r.cap_triangles {
        assert!(m.tri_normal(t).dot(DVec3::Z) > 0.999);
    }
    // The wall and the cap (the outlet) are face groups of their own, new to the mesh.
    let cap_group = r.cap_group.expect("capped");
    assert!(r.wall_group != cap_group && r.wall_group > 0 && cap_group > 0);
    assert!(r.wall_triangles.iter().all(|&t| m.tri_group(t) == r.wall_group));
    assert!(r.cap_triangles.iter().all(|&t| m.tri_group(t) == cap_group));
    assert_eq!(m.triangle_ids().filter(|&t| m.tri_group(t) == r.wall_group).count(), r.wall_triangles.len());
    // Close the other end and check the volume of the extended cylinder.
    let other: Vec<Vec<u32>> = boundary_loops(&m).into_iter().map(|l| l.vertices).collect();
    cap_planar_loops(&mut m, &other, None, CutFill::Flat).unwrap();
    assert_valid(&m);
    assert!(m.is_closed());
    assert!((volume(&m) - poly_area * (1.5 + 2.0 * d_eq)).abs() < 1e-9);
}

#[test]
fn flow_extension_absolute_rings_and_remeshed_cap() {
    let (mut m, lp) = cut_tube(32, 1.0);
    let r = extrude_boundary_loop(
        &mut m,
        &lp,
        &FlowExtensionParams { length: ExtensionLength::Absolute(3.0), segments: 7, cap: true, ..Default::default() },
    )
    .unwrap();
    assert_valid(&m);
    assert_eq!(r.rings.len(), 7);
    assert!(r.capped);
    assert!(r.cap_triangles.len() > 30, "remeshed cap has interior vertices");
    assert!((r.end_center.z - 4.5).abs() < 1e-9);
}

#[test]
fn flow_extension_blends_to_circle() {
    let (mut m, lp) = cut_tube(40, 2.0); // elliptical cross-section
    let r = extrude_boundary_loop(
        &mut m,
        &lp,
        &FlowExtensionParams {
            length: ExtensionLength::Diameters(3.0),
            make_circular: true,
            transition: 0.6,
            cap: true,
            ..Default::default()
        },
    )
    .unwrap();
    assert_valid(&m);
    let area = PI * (r.equivalent_diameter * 0.5).powi(2);
    let radius = (area / PI).sqrt();
    for &v in &r.end_loop {
        let p = m.vertex(v);
        let d = DVec3::new(p.x, p.y, 0.0).length();
        assert!((d - radius).abs() < 1e-9, "end ring is a circle of equal area");
    }
    // Rings stay ordered: no wall triangle is degenerate or inverted relative to its neighbours.
    for &t in &r.wall_triangles {
        assert!(m.tri_area(t) > 1e-6);
    }
}

#[test]
fn flow_extension_custom_direction_and_errors() {
    let (mut m, lp) = cut_tube(24, 1.0);
    let dir = DVec3::new(0.3, 0.0, 1.0).normalize();
    let r = extrude_boundary_loop(
        &mut m,
        &lp,
        &FlowExtensionParams { length: ExtensionLength::Absolute(1.0), direction: Some(dir), ..Default::default() },
    )
    .unwrap();
    assert_valid(&m);
    assert!((r.direction - dir).length() < 1e-12);
    let bad = extrude_boundary_loop(&mut m, &[0, 1, 2], &FlowExtensionParams::default());
    assert!(bad.is_err());
}
