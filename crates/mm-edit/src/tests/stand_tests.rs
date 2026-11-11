use super::*;
use crate::boolean::{mesh_boolean, BooleanOp};
use crate::stand::*;
use crate::thicken::{thicken, ThickenParams};
use mm_core::{Bvh, DVec2};
use std::collections::HashSet;

fn base_at(z: f64) -> BaseFrame {
    BaseFrame::new(DVec3::new(0.0, 0.0, z), DVec3::Z, DVec3::X)
}

fn square(width: f64, thickness: f64) -> BaseParams {
    BaseParams { shape: BaseShape::Rectangle, width, depth: width, thickness, corner_radius: 4.0 }
}

/// A triangle by its corners' exact positions, whichever corner comes first.
fn key(m: &Mesh, t: u32) -> [u64; 9] {
    let mut c: Vec<[u64; 3]> = m.triangle(t).iter().map(|&v| m.vertex(v).to_array().map(f64::to_bits)).collect();
    c.sort();
    [c[0][0], c[0][1], c[0][2], c[1][0], c[1][1], c[1][2], c[2][0], c[2][1], c[2][2]]
}

#[test]
fn bases_are_closed_slabs_under_their_plane() {
    let frame = BaseFrame::new(DVec3::new(1.0, 2.0, -10.0), DVec3::new(0.1, -0.2, 1.0), DVec3::new(1.0, 1.0, 0.0));
    for (shape, rc) in [(BaseShape::Rectangle, 8.0), (BaseShape::Rectangle, 0.0), (BaseShape::Ellipse, 0.0)] {
        let p = BaseParams { shape, width: 80.0, depth: 50.0, thickness: 4.0, corner_radius: rc };
        let m = base_mesh(&frame, &p, 3.0).unwrap();
        assert_valid(&m);
        assert!(m.is_closed(), "{shape:?}");
        let area = match shape {
            BaseShape::Rectangle => 80.0 * 50.0 - (4.0 - PI) * rc * rc,
            BaseShape::Ellipse => PI * 40.0 * 25.0,
        };
        let want = area * 4.0;
        assert!((volume(&m) - want).abs() < 0.01 * want, "{shape:?}: volume {:.0} vs {want:.0}", volume(&m));
        for v in m.vertex_ids() {
            let q = frame.local(m.vertex(v));
            assert!(q.z > -4.0 - 1e-6 && q.z < 1e-6, "{shape:?}: a vertex off the slab at height {}", q.z);
            assert!(p.outline_distance(DVec2::new(q.x, q.y)) < 1e-6, "{shape:?}: a vertex outside the outline");
        }
        // Remeshed to about the edge asked for: the top is not one fan of slivers.
        let top = m.triangle_ids().filter(|&t| m.tri_normal(t).dot(frame.up) > 0.999).count();
        assert!(top > 200, "{shape:?}: {top} top triangles");
        assert!(m.triangle_ids().all(|t| t < m.max_tid() as u32 && m.tri_group(t) == 0));
    }
    assert!(base_mesh(&frame, &BaseParams { thickness: 0.0, ..Default::default() }, 3.0).is_err());
}

#[test]
fn a_leg_holds_a_ball_up_off_the_base() {
    let ball = primitives::icosphere(10.0, 4);
    let bvh = Bvh::build(&ball);
    for up in [DVec3::Z, DVec3::new(0.3, -0.1, 1.0).normalize()] {
        let base = BaseFrame::new(-up * 22.0, up, DVec3::X);
        let base_params = square(40.0, 5.0);
        let p = LegParams { diameter: 3.0, root_diameter: 7.0, foot_diameter: 7.0, root_flare: 4.0, foot_flare: 4.0, bend: 0.5 };
        let plan = plan_leg(&ball, &bvh, -up * 12.0, base.center, &base, 5.0, &p).unwrap();
        assert!(plan.root.distance(-up * 10.0) < 0.1, "the root is moved onto the surface");
        assert!(plan.axis.dot(-up) > 0.99, "it faces down");
        assert_eq!((plan.overhang, plan.buried, plan.clearance), (0.0, false, f64::INFINITY));
        // A solid part: the far side is a diameter away, so the root sinks a quarter of its
        // radius in.
        assert!((plan.wall.unwrap() - 20.0).abs() < 0.1);
        assert!((plan.embed - 0.875).abs() < 1e-9, "embed {}", plan.embed);
        let leg = leg_mesh(&plan, None).unwrap();
        assert_valid(&leg);
        assert!(leg.is_closed());
        // The column is its diameter halfway down.
        let mid: Vec<f64> = leg.vertex_ids().map(|v| leg.vertex(v)).filter(|q| (q.dot(up) + 16.0).abs() < 1.0).map(|q| (q - up * q.dot(up)).length()).collect();
        assert!(!mid.is_empty() && mid.iter().all(|r| (r - 1.5).abs() < 0.15), "column radii {:?}", (mid.iter().copied().fold(9.0, f64::min), mid.iter().copied().fold(0.0, f64::max)));
        // It reaches into the base, and into the ball by `embed` below its surface all over
        // the root's footprint, no further.
        let lo = leg.vertex_ids().map(|v| leg.vertex(v).dot(up)).fold(f64::INFINITY, f64::min);
        assert!((lo + 24.5).abs() < 0.2, "the foot's plug ends at {lo}");
        let deepest = leg.vertex_ids().map(|v| leg.vertex(v).length()).fold(f64::INFINITY, f64::min);
        let tol = 0.5 * plan.cell_size();
        assert!((deepest - (10.0 - plan.embed)).abs() < tol + 0.05, "the root's plug is {:.3} deep", 10.0 - deepest);
        let rim: Vec<f64> = leg.vertex_ids().map(|v| leg.vertex(v)).filter(|q| q.length() < 10.0 && (q - up * q.dot(up)).length() > 3.0).map(|q| 10.0 - q.length()).collect();
        assert!(!rim.is_empty() && rim.iter().all(|d| *d < plan.embed + tol + 0.05), "under the footprint's rim it follows the surface");

        let stand = join_stand(&base_mesh(&base, &base_params, 2.0).unwrap(), &[&leg], None).unwrap();
        assert_eq!(stand.max_group_id(), 1, "base and leg groups");
        let all = mesh_boolean(&ball, &stand, BooleanOp::Union).unwrap().mesh;
        assert_valid(&all);
        assert!(all.is_closed());
        assert_eq!(connected_components(&all).len(), 1, "ball, leg and base are one piece");
        assert!(volume(&all) > volume(&ball) + 40.0 * 40.0 * 5.0 * 0.99);
    }
}

/// An open tube along x of radius 3 thickened outward into a 1.5 mm wall: a printed vessel.
fn vessel() -> Mesh {
    let points: Vec<DVec3> = (0..=160).map(|k| DVec3::new(-20.0 + 0.25 * k as f64, 0.0, 0.0)).collect();
    let tube = primitives::tube_along(&points, &vec![3.0; points.len()], 48);
    thicken(&tube, &ThickenParams { thickness: 1.5, ..Default::default() }, None).unwrap().mesh
}

#[test]
fn legs_on_a_hollow_vessel_change_only_its_outer_surface() {
    let shell = vessel();
    let bvh = Bvh::build(&shell);
    let radial = |q: DVec3| (q.y * q.y + q.z * q.z).sqrt();
    // The lumen's surface: the tube thickened (at radius 3), away from the open ends.
    let lumen: Vec<[u64; 9]> =
        shell.triangle_ids().filter(|&t| shell.triangle(t).iter().all(|&v| radial(shell.vertex(v)) < 3.01 && shell.vertex(v).x.abs() < 19.0)).map(|t| key(&shell, t)).collect();
    assert!(lumen.len() > 1000);
    let base = base_at(-15.0);
    let base_mesh = base_mesh(&base, &BaseParams { width: 60.0, depth: 40.0, ..square(60.0, 4.0) }, 2.0).unwrap();
    let s = std::f64::consts::FRAC_1_SQRT_2;
    // Under the vessel, a root wider than the vessel, on its side, and halfway round.
    for (at, root_diameter) in [(DVec3::new(0.0, 0.0, -6.0), 6.0), (DVec3::new(-8.0, 0.0, -6.0), 12.0), (DVec3::new(8.0, 6.0, 0.0), 6.0), (DVec3::new(4.0, 6.0 * s, -6.0 * s), 6.0)] {
        let p = LegParams { diameter: 2.5, root_diameter, foot_diameter: 6.0, root_flare: 4.0, foot_flare: 3.0, bend: 0.5 };
        let foot = default_foot(&shell, &bvh, at, &base, &BaseParams { width: 60.0, depth: 40.0, ..square(60.0, 4.0) }, root_diameter);
        let plan = plan_leg(&shell, &bvh, at, foot, &base, 4.0, &p).unwrap();
        let what = format!("root at {at} ⌀ {root_diameter}");
        assert!((radial(plan.root) - 4.5).abs() < 0.05, "{what}: root on the outer wall");
        // The wall straight down the axis (square to it at the root's centre).
        let wall = plan.wall.unwrap();
        assert!((wall - 1.5).abs() < 0.05, "{what}: wall {wall}");
        assert!(plan.embed <= PLUG_SHARE * wall + 1e-9 && plan.embed > 0.5 * PLUG_SHARE * wall, "{what}: embed {}", plan.embed);
        assert_eq!(plan.overhang > 0.1, root_diameter > 9.0, "{what}: overhang {}", plan.overhang);
        assert!(!plan.buried && plan.clearance > 1.0, "{what}");
        let leg = leg_mesh(&plan, None).unwrap();
        assert_valid(&leg);
        assert!(leg.is_closed(), "{what}");
        // Nothing of the leg comes near the lumen: it stops well inside the wall.
        let deepest = leg.vertex_ids().map(|v| leg.vertex(v)).filter(|q| q.x.abs() < 19.0).map(radial).fold(f64::INFINITY, f64::min);
        assert!(deepest > 3.0 + 0.3, "{what}: a leg vertex {deepest} from the axis");

        let stand = join_stand(&base_mesh, &[&leg], None).unwrap();
        let all = mesh_boolean(&shell, &stand, BooleanOp::Union).unwrap().mesh;
        assert_valid(&all);
        assert!(all.is_closed(), "{what}");
        assert_eq!(connected_components(&all).len(), 1, "{what}: one piece");
        // The lumen is untouched: every triangle of its surface is still there as it was,
        // and nothing new lies inside it.
        let kept: HashSet<[u64; 9]> = all.triangle_ids().map(|t| key(&all, t)).collect();
        let lost = lumen.iter().filter(|k| !kept.contains(*k)).count();
        assert_eq!(lost, 0, "{what}: {lost} lumen triangles changed");
        let inside = all.vertex_ids().map(|v| all.vertex(v)).filter(|q| q.x.abs() < 19.0 && radial(*q) < 2.99).count();
        assert_eq!(inside, 0, "{what}");
    }
}

#[test]
fn short_legs_shorten_their_flares_and_roots_must_be_above_the_base() {
    let ball = primitives::icosphere(10.0, 4);
    let bvh = Bvh::build(&ball);
    let p = LegParams { root_flare: 4.0, foot_flare: 4.0, ..Default::default() };
    // 5 mm below the ball: the flares (8 mm together) would meet.
    let plan = plan_leg(&ball, &bvh, DVec3::new(0.0, 0.0, -10.0), DVec3::new(0.0, 0.0, -15.0), &base_at(-15.0), 3.0, &p).unwrap();
    assert!((plan.root_flare + plan.foot_flare - 4.0).abs() < 1e-9, "{} + {}", plan.root_flare, plan.foot_flare);
    assert!(leg_mesh(&plan, None).unwrap().is_closed());
    // A base cutting through the ball: the root is not above it.
    assert!(plan_leg(&ball, &bvh, DVec3::new(0.0, 0.0, -10.0), DVec3::ZERO, &base_at(-8.0), 3.0, &p).is_err());
    assert!(plan_leg(&ball, &bvh, DVec3::new(0.0, 0.0, -10.0), DVec3::ZERO, &base_at(-20.0), 3.0, &LegParams { diameter: 0.0, ..p }).is_err());
}

#[test]
fn a_column_running_into_the_object_is_reported() {
    // A small ball hangs between the big ball's underside and the base.
    let mut two = primitives::icosphere(6.0, 3);
    let mut small = primitives::icosphere(2.0, 3);
    small.transform(|q| q + DVec3::new(0.0, 0.0, -14.0));
    two.append_mesh(&small, 1);
    let bvh = Bvh::build(&two);
    let p = LegParams { bend: 0.0, ..Default::default() };
    let plan = plan_leg(&two, &bvh, DVec3::new(0.0, 0.0, -6.0), DVec3::new(0.0, 0.0, -25.0), &base_at(-25.0), 3.0, &p).unwrap();
    assert!(plan.clearance < -1.0, "clearance {}", plan.clearance);
    let clear = plan_leg(&two, &bvh, DVec3::new(0.0, 0.0, -6.0), DVec3::new(9.0, 0.0, -25.0), &base_at(-25.0), 3.0, &LegParams { root_flare: 2.0, ..p });
    assert!(clear.is_ok());
}

#[test]
fn legs_apart_join_the_base_in_one_union_and_keep_their_groups() {
    let ball = primitives::icosphere(10.0, 4);
    let bvh = Bvh::build(&ball);
    let base = base_at(-22.0);
    let bp = square(50.0, 4.0);
    let p = LegParams { diameter: 2.0, root_diameter: 4.0, foot_diameter: 4.0, root_flare: 2.0, foot_flare: 2.0, bend: 0.0 };
    // Two legs far apart, and a third beside the first (their boxes meet: a union of its own).
    let roots = [DVec3::new(-5.0, 0.0, -8.66), DVec3::new(5.0, 0.0, -8.66), DVec3::new(-3.0, 2.5, -9.2)];
    let legs: Vec<Mesh> = roots
        .iter()
        .map(|&r| {
            let r = r.normalize() * 10.0;
            let plan = plan_leg(&ball, &bvh, r, base.project(r), &base, bp.thickness, &p).unwrap();
            leg_mesh(&plan, None).unwrap()
        })
        .collect();
    assert!(legs[0].bounds().intersects(&legs[2].bounds()) && !legs[0].bounds().intersects(&legs[1].bounds()));
    let base_m = base_mesh(&base, &bp, 2.0).unwrap();
    let refs: Vec<&Mesh> = legs.iter().collect();
    let calls = std::cell::Cell::new(Vec::new());
    let report = |f: f32| {
        let mut v = calls.take();
        v.push(f);
        calls.set(v);
        true
    };
    let stand = join_stand(&base_m, &refs, Some(&report)).unwrap();
    assert_eq!(calls.take(), vec![0.0, 0.5], "two unions");
    assert_valid(&stand);
    assert!(stand.is_closed());
    let mut groups: Vec<i32> = stand.triangle_ids().map(|t| stand.tri_group(t)).collect();
    groups.sort();
    groups.dedup();
    assert_eq!(groups, vec![0, 1, 2, 3], "the base 0, leg k k + 1");
    // As a union per leg makes it.
    let mut one_by_one = base_m.clone();
    for leg in &legs {
        one_by_one = mesh_boolean(&one_by_one, leg, BooleanOp::Union).unwrap().mesh;
    }
    assert!((volume(&stand) - volume(&one_by_one)).abs() < 1e-6 * volume(&stand));
    // Each leg's triangles keep its group: leg 2's are below the ball's right side.
    let right = stand.triangle_ids().filter(|&t| stand.tri_group(t) == 2).all(|t| stand.tri_centroid(t).x > 0.0);
    assert!(right, "leg 2 where it was made");
    // Stopped when told to.
    assert!(join_stand(&base_m, &refs, Some(&|_| false)).is_err());
}
