use super::*;
use crate::*;
use mm_core::{Bvh, DVec3, Plane};

#[test]
fn local_cut_trims_only_the_clicked_leg() {
    let base = u_tube(3.0, 3.0, 0.4, 24);
    assert_valid(&base);
    assert_eq!(n_loops(&base), 2);
    let plane = Plane::from_point_normal(DVec3::new(0.0, 0.0, 1.1), DVec3::Z);
    let leg_b: Vec<DVec3> = {
        let mut v: Vec<DVec3> =
            base.vertex_ids().map(|v| base.vertex(v)).filter(|p| p.x > 1.5 && p.z < 1.5).collect();
        v.sort_by(|a, b| (a.x, a.y, a.z).partial_cmp(&(b.x, b.y, b.z)).unwrap());
        v
    };

    let mut m = base.clone();
    let r = local_plane_cut(&mut m, plane, DVec3::new(0.4, 0.0, 1.0), KeepSide::Above, CutFill::Flat).unwrap();
    assert_valid(&m);
    assert!(r.curve_closed);
    assert!(r.curve_distance < 0.15);
    assert_eq!(r.cut.cut_loops.len(), 1);
    assert!(r.cut.loop_capped[0]);
    assert_eq!(n_loops(&m), 1, "only the untouched leg's open end remains");
    // Leg A ends at the plane; leg B is untouched.
    for v in m.vertex_ids() {
        let p = m.vertex(v);
        if p.x < 1.5 {
            assert!(p.z > 1.1 - 1e-9);
        }
    }
    let mut after: Vec<DVec3> = m.vertex_ids().map(|v| m.vertex(v)).filter(|p| p.x > 1.5 && p.z < 1.5).collect();
    after.sort_by(|a, b| (a.x, a.y, a.z).partial_cmp(&(b.x, b.y, b.z)).unwrap());
    assert_eq!(after, leg_b);
    for &t in &r.cut.cap_triangles {
        assert!(m.tri_normal(t).dot(DVec3::Z) < -0.999);
    }

    // A global cut would cut both legs.
    let mut g = base.clone();
    let rg = plane_cut(&mut g, plane, &PlaneCutParams { keep: KeepSide::Above, ..Default::default() }).unwrap();
    assert_eq!(rg.cut_loops.len(), 2);
}

#[test]
fn local_cut_keep_below_removes_the_rest() {
    // Keeping the lower stub of leg A deletes everything connected above the curve,
    // including leg B (it is reached through the arc).
    let base = u_tube(3.0, 3.0, 0.4, 24);
    let mut m = base.clone();
    let plane = Plane::from_point_normal(DVec3::new(0.0, 0.0, 1.1), DVec3::Z);
    let r = local_plane_cut(&mut m, plane, DVec3::new(-0.4, 0.0, 1.1), KeepSide::Below, CutFill::Flat).unwrap();
    assert_valid(&m);
    assert_eq!(n_components(&m), 1);
    assert!(m.vertex_ids().all(|v| m.vertex(v).x < 0.5 && m.vertex(v).z < 1.1 + 1e-9));
    assert!(r.cut.loop_capped[0]);
}

#[test]
fn local_cut_on_parallel_tubes_and_slice() {
    let mut base = primitives::cylinder(0.5, 2.0, 20, 6, false);
    let mut other = primitives::cylinder(0.5, 2.0, 28, 6, false);
    other.transform(|p| p + DVec3::new(3.0, 0.0, 0.0));
    base.append_mesh(&other, 1);
    let plane = Plane::from_point_normal(DVec3::new(0.0, 0.0, 1.13), DVec3::new(0.05, 0.0, 1.0));
    let mut m = base.clone();
    let r = local_plane_cut(&mut m, plane, DVec3::new(3.5, 0.0, 1.2), KeepSide::Below, CutFill::Flat).unwrap();
    assert_valid(&m);
    assert!(r.cut.loop_capped[0]);
    // Tube A untouched.
    let a_before = base.vertex_ids().filter(|&v| base.vertex(v).x < 1.5).count();
    let a_after = m.vertex_ids().filter(|&v| m.vertex(v).x < 1.5).count();
    assert_eq!(a_before, a_after);
    assert!(m.vertex_ids().all(|v| m.vertex(v).x < 1.5 || plane.signed_distance(m.vertex(v)) < 1e-9));
    assert_eq!(n_loops(&m), 3);

    // Slice (keep both) on tube A only.
    let mut s = base.clone();
    let rs = local_plane_cut(&mut s, plane, DVec3::new(-0.5, 0.0, 1.1), KeepSide::Both, CutFill::Flat).unwrap();
    assert_valid(&s);
    assert_eq!(n_components(&s), 3);
    assert_eq!(rs.cut.cut_loops.len(), 2);
    assert!(rs.cut.loop_capped.iter().all(|&c| c));
    assert_eq!(n_loops(&s), 4);
    assert!(volume(&s).is_finite());
}

#[test]
fn local_cut_errors() {
    let mut m = primitives::icosphere(1.0, 2);
    let e = local_plane_cut(
        &mut m,
        Plane::from_point_normal(DVec3::new(0.0, 0.0, 3.0), DVec3::Z),
        DVec3::ZERO,
        KeepSide::Above,
        CutFill::Flat,
    );
    assert!(matches!(e, Err(EditError::NoIntersection)));
    // A cut around the tube of a torus does not separate it.
    let mut t = torus(2.0, 0.5, 48, 24);
    let e = local_plane_cut(
        &mut t,
        Plane::from_point_normal(DVec3::ZERO, DVec3::Y),
        DVec3::new(2.5, 0.0, 0.0),
        KeepSide::Above,
        CutFill::Flat,
    );
    assert!(matches!(e, Err(EditError::CurveDoesNotSeparate)), "{e:?}");
    assert_valid(&t);
}

fn tube_point(center: DVec3, axis: DVec3, radius: f64) -> DVec3 {
    let (u, _) = mm_core::geom::orthonormal_basis(axis);
    center + u * radius
}

#[test]
fn suggest_plane_straight_tube() {
    let axis = DVec3::new(1.0, 2.0, 3.0).normalize();
    let p0 = DVec3::new(0.5, -1.0, 2.0);
    let path: Vec<DVec3> = (0..=40).map(|i| p0 + axis * (10.0 * i as f64 / 40.0)).collect();
    let m = primitives::tube_along(&path, &vec![0.5; path.len()], 32);
    let mid = p0 + axis * 4.3;
    let click = tube_point(mid, axis, 0.5);
    let bvh = Bvh::build(&m);
    for b in [None, Some(&bvh)] {
        let x = estimate_vessel_cross_section(&m, b, click, None).unwrap();
        assert!(x.axis.dot(axis).abs() > 0.9999, "axis {:?}", x.axis);
        assert!(x.center.distance(mid) < 0.02, "center {:?} vs {mid:?}", x.center);
        assert!((x.radius - 0.5).abs() < 0.02);
        assert!(x.area.is_some());
        let plane = suggest_cut_plane(&m, b, click, Some(0.5)).unwrap();
        assert!(plane.normal.dot(axis).abs() > 0.9999);
        assert!(plane.signed_distance(mid).abs() < 0.02);
    }
}

#[test]
fn suggest_plane_curved_tube_then_local_cut() {
    let bend = 5.0;
    let path: Vec<DVec3> = (0..=90)
        .map(|i| {
            let a = PI * i as f64 / 90.0;
            DVec3::new(bend * a.cos(), bend * a.sin(), 0.0)
        })
        .collect();
    let m0 = primitives::tube_along(&path, &vec![0.4; path.len()], 24);
    let a = 0.37 * PI;
    let center = DVec3::new(bend * a.cos(), bend * a.sin(), 0.0);
    let tangent = DVec3::new(-a.sin(), a.cos(), 0.0);
    let click = center + DVec3::new(a.cos(), a.sin(), 0.0) * 0.4;
    let x = estimate_vessel_cross_section(&m0, None, click, None).unwrap();
    assert!(x.axis.dot(tangent).abs() > 0.998, "axis {:?} vs {tangent:?}", x.axis);
    assert!(x.center.distance(center) < 0.05);

    // Click-to-cut: square cut through the bend with a local cut.
    let mut m = m0.clone();
    let r = local_plane_cut(&mut m, x.plane, click, KeepSide::Below, CutFill::Flat).unwrap();
    assert_valid(&m);
    assert!(r.cut.loop_capped[0]);
    assert_eq!(n_loops(&m), 1);
}

/// Y-shaped vessel built with this crate's own tools: a trunk whose top opening is split in
/// two by a bridge (the carina), each half joined to a tilted branch tube.
pub(crate) fn y_vessel() -> Mesh {
    let slices = 24;
    let mut m = primitives::cylinder(0.5, 2.0, slices, 10, false);
    let top: Vec<u32> = (0..slices).map(|j| (10 * slices + j) as u32).collect();
    for &v in &top {
        assert!((m.vertex(v).z - 2.0).abs() < 1e-12);
    }
    // Spans across the opening at y = ±0.5 (angles 90°-105° and 255°-270°).
    bridge(&mut m, &[top[6], top[7]], &[top[18], top[17]], &BridgeParams { segments: 0, smooth: false }).unwrap();
    assert_valid(&m);
    let mut halves: Vec<Vec<u32>> = boundary_loops(&m)
        .into_iter()
        .filter(|l| l.vertices.iter().all(|&v| (m.vertex(v).z - 2.0).abs() < 1e-9))
        .map(|l| l.vertices)
        .collect();
    assert_eq!(halves.len(), 2);
    halves.sort_by(|a, b| {
        let ca = a.iter().map(|&v| m.vertex(v).x).sum::<f64>();
        let cb = b.iter().map(|&v| m.vertex(v).x).sum::<f64>();
        ca.partial_cmp(&cb).unwrap()
    });
    for (half, sx) in halves.iter().zip([-1.0, 1.0]) {
        let th: f64 = 0.5;
        let dir = DVec3::new(sx * th.sin(), 0.0, th.cos());
        let mut b = primitives::cylinder(0.3, 3.0, 20, 12, false);
        let rot = mm_core::DQuat::from_rotation_arc(DVec3::Z, dir);
        let start = DVec3::new(sx * 0.45, 0.0, 2.6);
        b.transform(|p| start + rot * p);
        let before = m.max_vid() as u32;
        m.append_mesh(&b, 1);
        let bottom: Vec<u32> = boundary_loops(&m)
            .into_iter()
            .filter(|l| l.vertices.iter().all(|&v| v >= before))
            .min_by(|x, y| x.centroid(&m).z.partial_cmp(&y.centroid(&m).z).unwrap())
            .unwrap()
            .vertices;
        join_loops(&mut m, half, &bottom, &BridgeParams { segments: 0, smooth: true }).unwrap();
        assert_valid(&m);
    }
    m
}

#[test]
fn y_vessel_click_to_cut_one_branch() {
    let base = y_vessel();
    assert_valid(&base);
    assert_eq!(n_components(&base), 1);
    assert_eq!(n_loops(&base), 3, "trunk inlet + two branch outlets");
    // Close the inlet/outlets to verify the junction encloses a positive volume.
    let mut closed = base.clone();
    let loops: Vec<Vec<u32>> = boundary_loops(&closed).into_iter().map(|l| l.vertices).collect();
    cap_planar_loops(&mut closed, &loops, None, CutFill::Flat).unwrap();
    assert_valid(&closed);
    assert!(closed.is_closed() && volume(&closed) > 0.0);

    // Click on the right branch, halfway up.
    let th: f64 = 0.5;
    let dir = DVec3::new(th.sin(), 0.0, th.cos());
    let center = DVec3::new(0.45, 0.0, 2.6) + dir * 1.5;
    let click = center + DVec3::new(0.0, 0.3, 0.0);
    let bvh = mm_core::Bvh::build(&base);
    let x = estimate_vessel_cross_section(&base, Some(&bvh), click, None).unwrap();
    assert!(x.axis.dot(dir) > 0.995, "axis {:?} should point up the branch {dir:?}", x.axis);
    assert!(x.center.distance(center) < 0.05);
    assert!((x.radius - 0.3).abs() < 0.02);

    let left_before: Vec<DVec3> = base.vertex_ids().map(|v| base.vertex(v)).filter(|p| p.x < -0.2).collect();
    let mut m = base.clone();
    let r = local_plane_cut(&mut m, x.plane, click, KeepSide::Below, CutFill::FlatRemeshed { target_edge_length: 0.1 })
        .unwrap();
    assert_valid(&m);
    assert!(r.cut.loop_capped[0]);
    assert_eq!(n_loops(&m), 2, "right outlet replaced by a cap");
    let left_after: Vec<DVec3> = m.vertex_ids().map(|v| m.vertex(v)).filter(|p| p.x < -0.2).collect();
    assert_eq!(left_before.len(), left_after.len(), "left branch untouched");
    assert!(m.vertex_ids().all(|v| x.plane.signed_distance(m.vertex(v)) < 1e-9 || m.vertex(v).x < 0.3));

    // A horizontal plane crossing both branches: the local cut takes only the clicked one.
    let plane = Plane::from_point_normal(DVec3::new(0.0, 0.0, 4.0), DVec3::Z);
    assert_eq!(plane_intersection_curves(&base, plane).len(), 2);
    let mut h = base.clone();
    let seed = DVec3::new(-1.1, 0.0, 4.0);
    local_plane_cut(&mut h, plane, seed, KeepSide::Below, CutFill::Flat).unwrap();
    assert_valid(&h);
    assert!(h.vertex_ids().any(|v| h.vertex(v).z > 4.5 && h.vertex(v).x > 0.0), "right branch kept");
    assert!(h.vertex_ids().all(|v| h.vertex(v).z < 4.0 + 1e-9 || h.vertex(v).x > 0.0), "left branch trimmed");

    // Then add a flow extension to the left outlet stub after an uncapped cut.
    let mut f = base.clone();
    let rr = local_plane_cut(&mut f, plane, seed, KeepSide::Below, CutFill::None).unwrap();
    let fe = extrude_boundary_loop(
        &mut f,
        &rr.cut.cut_loops[0],
        &FlowExtensionParams { length: ExtensionLength::Diameters(5.0), make_circular: true, cap: true, ..Default::default() },
    )
    .unwrap();
    assert_valid(&f);
    assert!(fe.capped && fe.direction.z > 0.999);
}
