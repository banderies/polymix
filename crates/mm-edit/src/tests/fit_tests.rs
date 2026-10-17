//! Fit Primitive: each fit recovers the shape it was given.

use super::*;
use crate::fit::*;

fn all(m: &Mesh) -> Vec<u32> {
    m.triangle_ids().collect()
}

#[test]
fn cylinder_patch_gives_radius_and_axis() {
    // Half of a tube of radius 2 along a tilted axis.
    let mut tube = primitives::cylinder(2.0, 10.0, 48, 20, false);
    let q = mm_core::DQuat::from_euler(glam::EulerRot::XYZ, 0.4, 0.3, 0.0);
    tube.transform(|p| q * p + DVec3::new(1.0, -2.0, 3.0));
    let sel: Vec<u32> = tube.triangle_ids().filter(|&t| (q.inverse() * (tube.tri_centroid(t) - DVec3::new(1.0, -2.0, 3.0))).x > 0.0).collect();
    let r = fit_primitive(&tube, &sel, PrimitiveType::Cylinder, 32).unwrap();
    assert_valid(&r.mesh);
    assert!(r.mesh.is_closed());
    // Every fitted vertex lies on the original cylinder (radius 2 about the axis), caps aside.
    let axis = q * DVec3::Z;
    let on = r.mesh.vertex_ids().filter(|&v| {
        let d = r.mesh.vertex(v) - DVec3::new(1.0, -2.0, 3.0);
        (d - axis * d.dot(axis)).length() > 1.0
    });
    for v in on {
        let d = r.mesh.vertex(v) - DVec3::new(1.0, -2.0, 3.0);
        let rad = (d - axis * d.dot(axis)).length();
        assert!((rad - 2.0).abs() < 0.02, "radius {rad}");
    }
    assert!(r.description.contains("r 1.99") || r.description.contains("r 2.00"), "{}", r.description);
}

#[test]
fn spheres_and_planes() {
    let mut s = primitives::icosphere(1.5, 4);
    s.transform(|p| p + DVec3::new(0.5, 0.0, -1.0));
    let cap: Vec<u32> = s.triangle_ids().filter(|&t| s.tri_centroid(t).z > -0.2).collect();
    let best = fit_primitive(&s, &cap, PrimitiveType::SphereBestFit, 24).unwrap();
    let b = best.mesh.bounds();
    assert!((b.center() - DVec3::new(0.5, 0.0, -1.0)).length() < 0.02 && (b.extents().x - 3.0).abs() < 0.03, "{}", best.description);
    // A flat patch: square / rectangle / disc lie in its plane, facing its way.
    let plane = primitives::plane_grid(4.0, 2.0, 16, 8);
    for kind in [PrimitiveType::Square, PrimitiveType::Rectangle, PrimitiveType::DiscFromPlane, PrimitiveType::HeightField] {
        let r = fit_primitive(&plane, &all(&plane), kind, 16).unwrap();
        assert_valid(&r.mesh);
        assert!(r.mesh.vertex_ids().all(|v| r.mesh.vertex(v).z.abs() < 1e-9), "{kind:?} leaves the plane");
        assert!(r.mesh.triangle_ids().all(|t| r.mesh.tri_normal(t).z > 0.99), "{kind:?} faces the wrong way");
        let e = r.mesh.bounds().extents();
        match kind {
            PrimitiveType::Square => assert!((e.x - 4.0).abs() < 1e-9 && (e.y - 4.0).abs() < 1e-9),
            PrimitiveType::Rectangle | PrimitiveType::HeightField => assert!((e.x - 4.0).abs() < 1e-6 && (e.y - 2.0).abs() < 1e-6, "{kind:?} {e}"),
            _ => {}
        }
    }
}

#[test]
fn hull_and_sweep_are_closed() {
    let mut blob = primitives::icosphere(1.0, 2);
    blob.transform(|p| DVec3::new(p.x * 2.0, p.y, p.z * 0.5 + p.x * p.x * 0.2));
    let hull = fit_primitive(&blob, &all(&blob), PrimitiveType::ConvexHull, 32).unwrap();
    assert_valid(&hull.mesh);
    assert!(hull.mesh.is_closed());
    // Contains every input point.
    let w = mm_solid::WindingNumber::new(&hull.mesh);
    assert!(blob.vertex_ids().all(|v| w.eval(blob.vertex(v) * 0.999) > 0.5));
    let cube = primitives::cube(1.0, 3);
    let h = convex_hull(&cube.vertex_ids().map(|v| cube.vertex(v)).collect::<Vec<_>>()).unwrap();
    assert!((volume(&h) - 1.0).abs() < 1e-9, "cube hull volume {}", volume(&h));
    let sweep = fit_primitive(&blob, &all(&blob), PrimitiveType::LinearSweep, 32).unwrap();
    assert_valid(&sweep.mesh);
    assert!(sweep.mesh.is_closed() && volume(&sweep.mesh) > 0.0);
    let sp = fit_primitive(&blob, &all(&blob), PrimitiveType::SphereFromPlane, 16).unwrap();
    assert_valid(&sp.mesh);
}
