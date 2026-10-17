use super::*;
use crate::*;
use mm_core::{DMat4, DVec3, Plane};

#[test]
fn mirror_half_sphere_with_weld() {
    let base = primitives::icosphere(1.0, 3);
    let plane = Plane::from_point_normal(DVec3::ZERO, DVec3::Z);
    let mut half = base.clone();
    plane_cut(&mut half, plane, &PlaneCutParams { keep: KeepSide::Above, fill: CutFill::None, ..Default::default() })
        .unwrap();
    let mut capped = base.clone();
    plane_cut(&mut capped, plane, &PlaneCutParams { keep: KeepSide::Above, fill: CutFill::Flat, ..Default::default() })
        .unwrap();
    let half_vol = volume(&capped);
    let mut m = half.clone();
    let r = mirror(&mut m, plane, true).unwrap();
    assert_valid(&m);
    assert!(m.is_closed());
    assert_eq!(n_components(&m), 1);
    assert!(r.welded_vertices > 0);
    assert!((volume(&m) - 2.0 * half_vol).abs() < 1e-9);
    assert!(volume(&m) > 0.0);

    // Without weld: two open halves.
    let mut w = half.clone();
    mirror(&mut w, plane, false).unwrap();
    assert_valid(&w);
    assert_eq!(n_components(&w), 2);
    assert_eq!(n_loops(&w), 2);

    // Mirror of a closed mesh across a distant plane: a reflected, outward-facing copy.
    let mut s = primitives::icosphere(1.0, 2);
    let v = volume(&s);
    mirror(&mut s, Plane::from_point_normal(DVec3::new(3.0, 0.0, 0.0), DVec3::X), true).unwrap();
    assert_valid(&s);
    assert!((volume(&s) - 2.0 * v).abs() < 1e-9);
}

#[test]
fn transforms_keep_orientation() {
    let mut m = primitives::icosphere(1.0, 2);
    let v = volume(&m);
    transform_mesh(&mut m, DMat4::from_scale(DVec3::new(-1.0, 2.0, 1.0)));
    assert_valid(&m);
    assert!((volume(&m) - 2.0 * v).abs() < 1e-9, "mirroring scale keeps positive volume");
    let t = drop_to_ground(&mut m, DVec3::Z);
    assert!((t.z - 1.0).abs() < 1e-12);
    let zmin = m.vertex_ids().map(|x| m.vertex(x).z).fold(f64::INFINITY, f64::min);
    assert!(zmin.abs() < 1e-12);
    center_on_origin(&mut m);
    assert!(m.bounds().center().length() < 1e-12);

    let mut c = primitives::cylinder(0.3, 4.0, 16, 8, true);
    transform_mesh(&mut c, DMat4::from_rotation_x(0.7) * DMat4::from_rotation_z(0.3));
    let vc = volume(&c);
    align_to_principal_axes(&mut c);
    assert_valid(&c);
    assert!((volume(&c) - vc).abs() < 1e-9);
    let ext = c.bounds().extents();
    assert!((ext.x - 4.0).abs() < 1e-6, "long axis on X: {ext:?}");
}

#[test]
fn region_operations() {
    let mut m = primitives::icosphere(1.0, 2);
    let n = m.triangle_count();
    let sel: Vec<u32> = m.triangle_ids().filter(|&t| m.tri_centroid(t).z > 0.5).collect();
    let d = duplicate(&m, &sel);
    assert_valid(&d);
    assert_eq!(d.triangle_count(), sel.len());
    assert_eq!(m.triangle_count(), n);
    let mut m2 = m.clone();
    let s = separate(&mut m2, &sel);
    assert_valid(&s);
    assert_valid(&m2);
    assert_eq!(s.triangle_count() + m2.triangle_count(), n);
    let k = discard(&mut m, &sel);
    assert_eq!(k, sel.len());
    assert_valid(&m);
    let mut f = primitives::icosphere(1.0, 2);
    let all: Vec<u32> = f.triangle_ids().collect();
    let v = volume(&f);
    flip_normals(&mut f, &all);
    assert_valid(&f);
    assert!((volume(&f) + v).abs() < 1e-12);
}

#[test]
fn smooth_boundary_straightens_ragged_loop() {
    let mut m = primitives::cylinder(0.5, 2.0, 32, 8, false);
    let lp = boundary_loops(&m).into_iter().find(|l| m.vertex(l.vertices[0]).z > 1.0).unwrap();
    for (i, &v) in lp.vertices.iter().enumerate() {
        let mut p = m.vertex(v);
        p.z += if i % 2 == 0 { 0.08 } else { -0.08 };
        m.set_vertex(v, p);
    }
    let rough = |m: &Mesh| {
        let z: Vec<f64> = lp.vertices.iter().map(|&v| m.vertex(v).z).collect();
        let mean = z.iter().sum::<f64>() / z.len() as f64;
        z.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / z.len() as f64
    };
    let r0 = rough(&m);
    smooth_boundary(&mut m, &lp.vertices, 10).unwrap();
    assert_valid(&m);
    assert!(rough(&m) < 0.05 * r0, "{} vs {}", rough(&m), r0);
    // Stays on the cylinder (tangent-plane motion) and does not shrink much.
    for &v in &lp.vertices {
        let p = m.vertex(v);
        assert!(((p.x * p.x + p.y * p.y).sqrt() - 0.5).abs() < 0.02);
    }
}
