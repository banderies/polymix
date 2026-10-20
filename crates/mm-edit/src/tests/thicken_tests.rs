use super::*;
use crate::thicken::*;

/// Pairs of triangles cutting through each other (neighbours sharing a vertex aside).
fn crossings(m: &Mesh) -> usize {
    let bvh = mm_core::Bvh::build(m);
    let mut n = 0;
    for t in m.triangle_ids() {
        let (tv, tp) = (m.triangle(t), m.tri_positions(t));
        bvh.query_aabb(&mm_core::Aabb::from_points(tp), |o| {
            if o > t && !m.triangle(o).iter().any(|v| tv.contains(v)) && mm_core::intersect::triangles_intersect(tp, m.tri_positions(o)) {
                n += 1;
            }
        });
    }
    n
}

/// An open tube of radius `r` along `path` (sampled every 0.25), its ends cut square.
fn tube(path: impl Fn(f64) -> DVec3, len: f64, r: f64) -> Mesh {
    let n = (len / 0.25).round() as usize;
    let points: Vec<DVec3> = (0..=n).map(|k| path(len * k as f64 / n as f64)).collect();
    primitives::tube_along(&points, &vec![r; points.len()], 48)
}

fn straight(z: f64) -> DVec3 {
    DVec3::new(0.0, 0.0, z)
}

/// Radial distances from the z axis of the vertices of group `g` between z = 5 and 25.
fn radii(m: &Mesh, pick: impl Fn(i32) -> bool) -> (f64, f64) {
    let r: Vec<f64> = m
        .triangle_ids()
        .filter(|&t| pick(m.tri_group(t)))
        .flat_map(|t| m.triangle(t))
        .map(|v| m.vertex(v))
        .filter(|p| (5.0..25.0).contains(&p.z))
        .map(|p| p.truncate().length())
        .collect();
    (r.iter().copied().fold(f64::INFINITY, f64::min), r.iter().copied().fold(0.0, f64::max))
}

#[test]
fn a_tube_thickens_into_a_wall_each_way() {
    // Radius 3, 30 long, open at both ends; a 2 mm wall outward, inward and centred.
    let m0 = tube(straight, 30.0, 3.0);
    for (side, inner, outer) in [(ThickenSide::Outward, 3.0, 5.0), (ThickenSide::Inward, 1.0, 3.0), (ThickenSide::Centered, 2.0, 4.0)] {
        let res = thicken(&m0, &ThickenParams { thickness: 2.0, side, ..Default::default() }, None).unwrap();
        let m = &res.mesh;
        assert_valid(m);
        assert!(m.is_closed(), "{side:?}: closed");
        assert_eq!(crossings(m), 0, "{side:?}");
        assert_eq!(res.openings.len(), 2);
        // The wall between the two radii (the tube's facets lie a little inside its radius).
        let want = PI * (outer * outer - inner * inner) * 30.0;
        assert!((volume(m) - want).abs() < 0.03 * want, "{side:?}: volume {:.0} vs {want:.0}", volume(m));
        let new_side = |g: i32| g == res.wall_group;
        let (lo, hi) = radii(m, new_side);
        let at = if side == ThickenSide::Outward { outer } else { inner };
        if side != ThickenSide::Centered {
            assert!(lo > at - 0.1 && hi < at + 0.1, "{side:?}: new side at {lo:.3}..{hi:.3}, want {at}");
        }
        assert!((res.measured[1] - 2.0).abs() < 0.1 && res.measured[0] > 1.8, "{side:?}: measured {:?}", res.measured);
        // Flat rims just in front of the end planes (a twentieth of the thickness).
        let rim: Vec<DVec3> = m.triangle_ids().filter(|&t| m.tri_group(t) == res.rim_group).flat_map(|t| m.tri_positions(t)).collect();
        let off = rim.iter().map(|p| (p.z + 0.1).abs().min((p.z - 30.1).abs())).fold(0.0, f64::max);
        assert!(!rim.is_empty() && off < 1e-6, "{side:?}: rims at the ends ({off:.2e} off)");
        // The kept surface is the input, untouched.
        if side != ThickenSide::Centered {
            let (lo, hi) = radii(m, |g| g < res.wall_group);
            let keep = if side == ThickenSide::Outward { inner } else { outer };
            assert!(lo > keep * (1.0 - 1e-9) * (PI / 48.0).cos() && hi < keep + 1e-9, "{side:?}: kept side {lo:.4}..{hi:.4}");
        }
    }
}

#[test]
fn a_tight_bend_thickens_without_folding() {
    // Radius 3 bent round a radius of 4.5: the inside of the bend is 1.5 from its centre, so
    // a 2 mm wall moved out along the normals would cross over itself there.
    let bend = |s: f64| {
        let a = s / 4.5;
        DVec3::new(4.5 * a.cos(), 4.5 * a.sin(), 0.0)
    };
    let m0 = tube(bend, 4.5 * 2.5, 3.0);
    let res = thicken(&m0, &ThickenParams { thickness: 2.0, ..Default::default() }, None).unwrap();
    assert_valid(&res.mesh);
    assert!(res.mesh.is_closed());
    assert_eq!(crossings(&res.mesh), 0);
    assert!(res.measured[0] > 1.7, "measured {:?}", res.measured);
}

#[test]
fn an_opening_too_narrow_for_an_inward_wall_is_reported() {
    // Radius 1.5: a 2 mm wall inside it closes the lumen up.
    let m0 = tube(straight, 30.0, 1.5);
    let e = thicken(&m0, &ThickenParams { thickness: 2.0, side: ThickenSide::Inward, ..Default::default() }, None).unwrap_err();
    assert!(e.to_string().contains("too narrow"), "{e}");
    // Outward it is fine.
    let res = thicken(&m0, &ThickenParams { thickness: 2.0, ..Default::default() }, None).unwrap();
    assert!(res.mesh.is_closed());
}

#[test]
fn a_closed_surface_gets_a_hollow_wall() {
    // A sphere of radius 5 thickened outward by 1: a closed shell with a closed cavity.
    let m0 = primitives::icosphere(5.0, 4);
    let res = thicken(&m0, &ThickenParams { thickness: 1.0, ..Default::default() }, None).unwrap();
    assert_valid(&res.mesh);
    assert!(res.mesh.is_closed() && res.openings.is_empty());
    assert_eq!(connected_components(&res.mesh).len(), 2);
    let want = 4.0 / 3.0 * PI * (6f64.powi(3) - 5f64.powi(3));
    assert!((volume(&res.mesh) - want).abs() < 0.03 * want, "volume {:.0} vs {want:.0}", volume(&res.mesh));
}
