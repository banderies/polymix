use super::*;
use crate::boolean::{mesh_boolean, BooleanOp};
use crate::pad::*;
use crate::stand::PLUG_SHARE;
use crate::thicken::{thicken, ThickenParams};
use mm_core::Bvh;
use std::collections::HashSet;

/// A triangle by its corners' exact positions, whichever corner comes first.
fn key(m: &Mesh, t: u32) -> [u64; 9] {
    let mut c: Vec<[u64; 3]> = m.triangle(t).iter().map(|&v| m.vertex(v).to_array().map(f64::to_bits)).collect();
    c.sort();
    [c[0][0], c[0][1], c[0][2], c[1][0], c[1][1], c[1][2], c[2][0], c[2][1], c[2][2]]
}

/// An open tube along x of radius 3 about (y, z) = `at`, thickened outward into a 1.5 mm
/// wall: a printed vessel.
fn vessel_at(y: f64, z: f64) -> Mesh {
    let points: Vec<DVec3> = (0..=160).map(|k| DVec3::new(-20.0 + 0.25 * k as f64, y, z)).collect();
    let tube = primitives::tube_along(&points, &vec![3.0; points.len()], 48);
    thicken(&tube, &ThickenParams { thickness: 1.5, ..Default::default() }, None).unwrap().mesh
}

/// The pad's top face's triangles (facing along the axis, at the top's height).
fn top_faces(m: &Mesh, plan: &PadPlan) -> Vec<u32> {
    let h = plan.top_center().dot(plan.axis);
    m.triangle_ids().filter(|&t| m.tri_normal(t).dot(plan.axis) > 0.9999 && (m.tri_centroid(t).dot(plan.axis) - h).abs() < 1e-9).collect()
}

#[test]
fn a_pad_on_a_ball_is_a_closed_block_with_a_flat_top() {
    let ball = primitives::icosphere(10.0, 4);
    let bvh = Bvh::build(&ball);
    let p = PadParams { length: 8.0, width: 4.0, height: 3.0, draft_deg: 20.0, ..Default::default() };
    let plan = plan_pad(&ball, &bvh, DVec3::new(0.0, 0.0, 12.0), 0.0, &p).unwrap();
    assert!(plan.center.distance(DVec3::new(0.0, 0.0, 10.0)) < 0.05, "the centre is moved onto the surface");
    assert!(plan.axis.dot(DVec3::Z) > 0.999);
    assert!(!plan.follows_surface, "a ball runs no way in particular");
    assert!(plan.along.dot(DVec3::X).abs() > 0.999, "so the line runs along a world axis: {}", plan.along);
    assert!(plan.ok() && plan.overhang == 0.0 && plan.wall.is_none_or(|w| w > 15.0), "{plan:?}");
    assert!(plan.lift <= 0.0 && plan.lift > -0.5, "resting on the top of the ball: lift {}", plan.lift);
    let m = pad_mesh(&plan).unwrap();
    assert_valid(&m);
    assert!(m.is_closed());
    // A flat top as big as asked for, `height` above the base.
    let (tl, tw) = p.top();
    assert!((tl - (8.0 + 6.0 * 20f64.to_radians().tan())).abs() < 1e-12);
    let top = top_faces(&m, &plan);
    let area: f64 = top.iter().map(|&t| m.tri_area(t)).sum();
    assert!((area - tl * tw).abs() < 1e-6 * tl * tw, "top area {area} vs {}", tl * tw);
    let top_z = 10.0 + plan.lift + 3.0;
    for &t in &top {
        for v in m.triangle(t) {
            assert!((m.vertex(v).z - top_z).abs() < 1e-9);
        }
    }
    assert!(m.vertex_ids().all(|v| m.vertex(v).z <= top_z + 1e-9));
    // The underside lies inside the ball, just under its surface.
    let under: Vec<DVec3> = m.vertex_ids().map(|v| m.vertex(v)).filter(|q| q.z < 10.0 + plan.lift - 1e-6).collect();
    let deepest = under.iter().map(|q| q.length()).fold(f64::INFINITY, f64::min);
    assert!(deepest > 9.0 && deepest < 10.0, "{deepest}");
    let all = mesh_boolean(&ball, &m, BooleanOp::Union).unwrap().mesh;
    assert_valid(&all);
    assert!(all.is_closed());
    assert_eq!(connected_components(&all).len(), 1, "one piece");
    assert!(volume(&all) > volume(&ball) + 8.0 * 4.0 * 3.0);
}

#[test]
fn a_pad_on_a_hollow_vessel_changes_only_its_outer_surface() {
    let shell = vessel_at(0.0, 0.0);
    let bvh = Bvh::build(&shell);
    let radial = |q: DVec3| (q.y * q.y + q.z * q.z).sqrt();
    let lumen: Vec<[u64; 9]> =
        shell.triangle_ids().filter(|&t| shell.triangle(t).iter().all(|&v| radial(shell.vertex(v)) < 3.01 && shell.vertex(v).x.abs() < 19.0)).map(|t| key(&shell, t)).collect();
    assert!(lumen.len() > 1000);
    // Along the vessel, across it, turned 45° on its side, and wider than the vessel.
    let cases = [
        (DVec3::new(2.0, 0.0, 6.0), 0.0, PadParams { length: 12.0, width: 5.0, height: 4.0, draft_deg: 30.0, ..Default::default() }),
        (DVec3::new(-3.0, 0.0, 6.0), 90.0, PadParams { length: 6.0, width: 4.0, height: 3.0, draft_deg: 45.0, ..Default::default() }),
        (DVec3::new(0.0, 6.0, 0.0), 45.0, PadParams { length: 8.0, width: 3.0, height: 5.0, draft_deg: 10.0, ..Default::default() }),
        (DVec3::new(4.0, 0.0, -6.0), 0.0, PadParams { length: 10.0, width: 12.0, height: 3.0, draft_deg: 0.0, ..Default::default() }),
    ];
    for (at, angle, p) in cases {
        let what = format!("pad at {at}, {angle}°, {p:?}");
        let plan = plan_pad(&shell, &bvh, at, angle, &p).unwrap();
        assert!((radial(plan.center) - 4.5).abs() < 0.05, "{what}: on the outer wall");
        assert!(plan.follows_surface && plan.reference.dot(DVec3::X).abs() > 0.99, "{what}: measured from the vessel's length: {}", plan.reference);
        let turned = plan.along.dot(DVec3::X).abs();
        assert!((turned - angle.to_radians().cos().abs()).abs() < 0.02, "{what}: along {}", plan.along);
        let wall = plan.wall.unwrap();
        assert!((wall - 1.5).abs() < 0.05, "{what}: wall {wall}");
        assert!(plan.embed <= PLUG_SHARE * wall + 1e-9 && plan.embed > 0.0, "{what}: embed {}", plan.embed);
        assert!(plan.ok(), "{what}: {plan:?}");
        assert_eq!(plan.overhang, 0.0, "{what}: the contact is all on the vessel");
        let (cl, cw) = plan.contact.size();
        assert!(cl <= p.length + 1e-9 && cw <= p.width + 1e-9, "{what}: contact {cl} × {cw}");
        let pad = pad_mesh(&plan).unwrap();
        assert_valid(&pad);
        assert!(pad.is_closed(), "{what}");
        let deepest = pad.vertex_ids().map(|v| pad.vertex(v)).map(radial).fold(f64::INFINITY, f64::min);
        assert!(deepest > 3.0 + 0.3, "{what}: a pad vertex {deepest} from the axis");

        let all = mesh_boolean(&shell, &pad, BooleanOp::Union).unwrap().mesh;
        assert_valid(&all);
        assert!(all.is_closed(), "{what}");
        assert_eq!(connected_components(&all).len(), 1, "{what}: one piece");
        let kept: HashSet<[u64; 9]> = all.triangle_ids().map(|t| key(&all, t)).collect();
        let lost = lumen.iter().filter(|k| !kept.contains(*k)).count();
        assert_eq!(lost, 0, "{what}: {lost} lumen triangles changed");
        let inside = all.vertex_ids().map(|v| all.vertex(v)).filter(|q| q.x.abs() < 19.0 && radial(*q) < 2.99).count();
        assert_eq!(inside, 0, "{what}");
        // The top face is there, flat, as big as asked for.
        let (tl, tw) = p.top();
        let area: f64 = top_faces(&all, &plan).iter().map(|&t| all.tri_area(t)).sum();
        assert!((area - tl * tw).abs() < 1e-3 * tl * tw, "{what}: top area {area} vs {}", tl * tw);
    }
}

#[test]
fn a_square_pad_joins_a_vessel_only_where_it_faces_the_pad() {
    // A pad as wide as it is long, wider than the vessel (outer radius 4.5): it joins the
    // vessel only over its crown, and leans out from there to the square top.
    let shell = vessel_at(0.0, 0.0);
    let bvh = Bvh::build(&shell);
    let p = PadParams { length: 12.0, width: 12.0, height: 4.0, draft_deg: 20.0, ..Default::default() };
    let plan = plan_pad(&shell, &bvh, DVec3::new(0.0, 0.0, 6.0), 0.0, &p).unwrap();
    assert!(plan.ok() && plan.overhang == 0.0 && plan.cut_back(), "{plan:?}");
    let (cl, cw) = plan.contact.size();
    assert!((cl - 12.0).abs() < 1e-9, "the whole length along the vessel: {cl}");
    // Up to where the wall turns 45° away: 2 · 4.5 · sin 45°.
    let crown = 9.0 * 45f64.to_radians().sin();
    assert!(cw <= crown + 1e-6 && cw > crown - 0.4, "across: {cw} vs {crown}");
    assert!((plan.contact.y0 + plan.contact.y1).abs() < 0.2, "centred on the crown: {:?}", plan.contact);
    let pad = pad_mesh(&plan).unwrap();
    assert_valid(&pad);
    assert!(pad.is_closed());
    // Nothing of it hangs down the vessel's sides: below the contact's plane it stays over
    // the contact, no lower than the wall 45° round from the top, less the plug.
    // (In the pad's frame: across the line, and up its axis from the centre.)
    for q in pad.vertex_ids().map(|v| pad.vertex(v) - plan.center) {
        let (y, z) = (q.dot(plan.across), q.dot(plan.axis));
        if z < plan.lift - 1e-9 {
            assert!(y >= plan.contact.y0 - 1e-9 && y <= plan.contact.y1 + 1e-9, "{q}");
            assert!(z > -4.5 * (1.0 - 45f64.to_radians().cos()) - PLUG_SHARE * 1.5 - 0.05, "{q}");
        }
    }
    // The top is the square asked for.
    let (tl, tw) = p.top();
    assert!((tl - tw).abs() < 1e-12);
    let area: f64 = top_faces(&pad, &plan).iter().map(|&t| pad.tri_area(t)).sum();
    assert!((area - tl * tw).abs() < 1e-6 * tl * tw, "top area {area} vs {}", tl * tw);
    let all = mesh_boolean(&shell, &pad, BooleanOp::Union).unwrap().mesh;
    assert_valid(&all);
    assert!(all.is_closed());
    assert_eq!(connected_components(&all).len(), 1, "one piece");

    // Up to 90°: the vessel's whole width (but no more).
    let wide = PadParams { contact_deg: 90.0, ..p };
    let plan = plan_pad(&shell, &bvh, DVec3::new(0.0, 0.0, 6.0), 0.0, &wide).unwrap();
    let (_, cw) = plan.contact.size();
    assert!(cw > 8.0 && cw <= 9.0, "{cw}");
    assert!(plan.ok() && plan.overhang == 0.0, "{plan:?}");
    // Narrower than the crown: as asked.
    let narrow = PadParams { width: 4.0, ..p };
    let plan = plan_pad(&shell, &bvh, DVec3::new(0.0, 0.0, 6.0), 0.0, &narrow).unwrap();
    assert!(!plan.cut_back() && plan.contact.size() == (12.0, 4.0), "{:?}", plan.contact);
}

#[test]
fn a_branch_over_or_beside_the_pad_stops_it() {
    let p = PadParams { length: 10.0, width: 5.0, height: 8.0, draft_deg: 30.0, ..Default::default() };
    // A second vessel right over the first: the ray down to the centre meets it first.
    let mut over = vessel_at(0.0, 0.0);
    over.append_mesh(&vessel_at(0.0, 12.0), 10);
    // One above and beside it: the pad's flared side runs into it.
    let mut beside = vessel_at(0.0, 0.0);
    beside.append_mesh(&vessel_at(7.0, 9.0), 10);
    for (what, m) in [("over", over), ("beside", beside)] {
        let bvh = Bvh::build(&m);
        let plan = plan_pad(&m, &bvh, DVec3::new(0.0, 0.0, 5.0), 0.0, &p).unwrap();
        assert!((plan.center.z - 4.5).abs() < 0.05, "{what}: on the first vessel");
        assert!(!plan.ok(), "{what}: {plan:?}");
    }
    // Lower and narrower, the pad clears the one beside.
    let mut beside = vessel_at(0.0, 0.0);
    beside.append_mesh(&vessel_at(7.0, 9.0), 10);
    let bvh = Bvh::build(&beside);
    let low = PadParams { width: 4.0, height: 2.0, draft_deg: 10.0, ..p };
    let plan = plan_pad(&beside, &bvh, DVec3::new(0.0, 0.0, 5.0), 0.0, &low).unwrap();
    assert!(plan.ok() && plan.clearance > 0.0, "{plan:?}");
}

#[test]
fn a_pad_on_an_inside_curve_rests_on_its_highest_points() {
    // A torus's inner rim (radius 16, facing the hole) curves up toward the ends of a line
    // laid along it: the base rests on the line's ends, above the centre.
    let t = torus(20.0, 4.0, 160, 48);
    let bvh = Bvh::build(&t);
    let p = PadParams { length: 16.0, width: 3.0, height: 3.0, draft_deg: 15.0, ..Default::default() };
    let plan = plan_pad(&t, &bvh, DVec3::new(14.0, 0.0, 0.0), 0.0, &p).unwrap();
    assert!(plan.axis.dot(-DVec3::X) > 0.99, "{}", plan.axis);
    assert!(plan.follows_surface && plan.along.dot(DVec3::Y).abs() > 0.99, "along the rim: {}", plan.along);
    let rise = 16.0 - (16.0f64 * 16.0 - 8.0 * 8.0).sqrt();
    assert!((plan.lift - rise).abs() < 0.3, "lift {} vs {rise}", plan.lift);
    assert!(plan.ok(), "{plan:?}");
    let pad = pad_mesh(&plan).unwrap();
    assert_valid(&pad);
    assert!(pad.is_closed());
    let all = mesh_boolean(&t, &pad, BooleanOp::Union).unwrap().mesh;
    assert_valid(&all);
    assert!(all.is_closed());
    assert_eq!(connected_components(&all).len(), 1);
}

#[test]
fn pad_sizes_are_checked() {
    let ball = primitives::icosphere(10.0, 3);
    let bvh = Bvh::build(&ball);
    let at = DVec3::new(0.0, 0.0, 12.0);
    let ok = PadParams::default();
    assert!(plan_pad(&ball, &bvh, at, 0.0, &ok).is_ok());
    for bad in [PadParams { width: 0.0, ..ok }, PadParams { height: -1.0, ..ok }, PadParams { draft_deg: 80.0, ..ok }, PadParams { contact_deg: 0.0, ..ok }, PadParams { contact_deg: 95.0, ..ok }, PadParams { length: 10.0, width: 4.0, height: 6.0, draft_deg: -30.0, ..Default::default() }] {
        assert!(plan_pad(&ball, &bvh, at, 0.0, &bad).is_err(), "{bad:?}");
    }
    assert!(plan_pad(&ball, &bvh, at, f64::NAN, &ok).is_err());
    // Narrowing, but not to nothing: a smaller top.
    let narrow = PadParams { length: 10.0, width: 4.0, height: 2.0, draft_deg: -20.0, ..Default::default() };
    let plan = plan_pad(&ball, &bvh, at, 0.0, &narrow).unwrap();
    let m = pad_mesh(&plan).unwrap();
    assert_valid(&m);
    assert!(m.is_closed());
    assert!(narrow.top().1 < 4.0 && narrow.top().1 > 2.0);
}
