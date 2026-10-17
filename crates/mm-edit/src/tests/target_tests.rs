//! Target-surface operations: projection, attraction, ICP registration.

use super::*;
use crate::target::*;

#[test]
fn attract_wraps_a_patch_onto_a_sphere() {
    let target = primitives::icosphere(1.0, 4);
    let bvh = mm_core::Bvh::build(&target);
    let mut m = primitives::icosphere(1.2, 3);
    let sel: Vec<u32> = m.triangle_ids().filter(|&t| m.tri_centroid(t).z > 0.6).collect();
    let moved = attract_to_target(&mut m, &sel, &target, &bvh, &AttractParams { offset: 0.05, ..Default::default() });
    assert!(!moved.is_empty());
    m.check_validity().unwrap();
    for &v in &moved {
        assert!((m.vertex(v).length() - 1.05).abs() < 0.01, "{}", m.vertex(v).length());
    }
    // Unselected vertices stay.
    assert!(m.vertex_ids().filter(|v| !moved.contains(v)).all(|v| (m.vertex(v).length() - 1.2).abs() < 1e-9));
}

#[test]
fn normal_threshold_skips_facing_away() {
    let target = primitives::plane_grid(4.0, 4.0, 8, 8);
    let bvh = mm_core::Bvh::build(&target);
    // A small sphere above a +Z-facing plane: only vertices whose normal is within 60° of +Z
    // (the top cap) are projected.
    let mut m = primitives::icosphere(0.5, 2);
    m.transform(|p| p + DVec3::Z);
    let verts: Vec<u32> = m.vertex_ids().collect();
    let up = verts.iter().filter(|&&v| m.vertex(v).z > 1.0 + 0.25).count();
    let moved = project_to_target(&mut m, &verts, &target, &bvh, 0.0, Some(60.0));
    assert!(moved > 0 && moved <= up + 4, "{moved} vs {up}");
}

#[test]
fn icp_recovers_a_rigid_motion() {
    // An asymmetric target so the fit is unique.
    let mut target = primitives::cube(1.0, 6);
    target.transform(|p| DVec3::new(2.0 * p.x, p.y, 0.6 * p.z));
    let bvh = mm_core::Bvh::build(&target);
    let q = mm_core::DQuat::from_euler(glam::EulerRot::XYZ, 0.15, -0.1, 0.2);
    let shift = DVec3::new(0.12, -0.08, 0.05);
    let pts: Vec<DVec3> = target.vertex_ids().map(|v| q * target.vertex(v) + shift).collect();
    let r = icp_align(&pts, &target, &bvh, &IcpParams::default());
    let err = pts.iter().zip(target.vertex_ids()).map(|(&p, v)| r.apply(p).distance(target.vertex(v))).fold(0.0, f64::max);
    assert!(err < 1e-3, "max error {err} after {} iterations, rms {}", r.iterations, r.rms);
    let only_t = icp_align(&pts, &target, &bvh, &IcpParams { rotate: false, ..Default::default() });
    assert_eq!(only_t.rotation, mm_core::DQuat::IDENTITY);
}
