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
        // Flat rims just in front of the end planes (a twentieth of the thickness), a group each.
        assert_eq!(res.rim_groups.len(), 2, "{side:?}");
        assert_ne!(res.rim_groups[0], res.rim_groups[1], "{side:?}: a group per rim");
        for (opening, &g) in res.openings.iter().zip(&res.rim_groups) {
            assert!(g != res.wall_group && g > 0, "{side:?}: rim group {g}");
            let rim: Vec<DVec3> = m.triangle_ids().filter(|&t| m.tri_group(t) == g).flat_map(|t| m.tri_positions(t)).collect();
            let end = if opening.centre.z < 15.0 { -0.1 } else { 30.1 };
            let off = rim.iter().map(|p| (p.z - end).abs()).fold(0.0, f64::max);
            assert!(!rim.is_empty() && off < 1e-6, "{side:?}: the rim of the opening at z {:.1} lies at its end ({off:.2e} off)", opening.centre.z);
        }
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

#[test]
fn a_barb_on_the_wall_stays_sharp_and_the_lumen_keeps_its_shape() {
    // Radius 3, 30 long, a 2 mm wall outward: a barb 1 mm wider than the 10 mm outside at the
    // top end, 2 mm back from the rim, ramping over 1.866 (15°).
    let m0 = tube(straight, 30.0, 3.0);
    let top = openings(&m0).iter().position(|(_, o)| o.normal.z > 0.9).unwrap();
    let barb = crate::Barb { width: 1.0, setback: 2.0, length: 0.5 / 15f64.to_radians().tan() };
    let res = thicken(&m0, &ThickenParams { thickness: 2.0, barbs: vec![(top, barb)], ..Default::default() }, None).unwrap();
    let m = &res.mesh;
    assert_valid(m);
    assert!(m.is_closed());
    assert_eq!(crossings(m), 0);
    assert_eq!(res.barb_groups.len(), 1);
    let g = res.barb_groups[0];
    let barb_points: Vec<DVec3> = m.triangle_ids().filter(|&t| m.tri_group(t) == g).flat_map(|t| m.triangle(t)).map(|v| m.vertex(v)).collect();
    assert!(!barb_points.is_empty(), "the barb has a group of its own");
    // The crest: a ring exactly 5.5 off the axis, where it was put (the rim stands a
    // twentieth of the wall in front of the opening), and nothing of the wall wider.
    let crest_z = 30.0 + 0.1 - 2.0 - barb.length;
    let crest: Vec<&DVec3> = barb_points.iter().filter(|p| (p.truncate().length() - 5.5).abs() < 1e-9).collect();
    assert!(crest.len() >= 48, "{} crest points", crest.len());
    assert!(crest.iter().all(|p| (p.z - crest_z).abs() < 1e-9));
    assert!(m.vertex_ids().all(|v| m.vertex(v).truncate().length() < 5.5 + 1e-9));
    // The back face, square to the axis toward the vessel, from the outside (about 5) out to
    // the crest.
    let back: f64 = m.triangle_ids().filter(|&t| m.tri_group(t) == g && m.tri_normal(t).z < -0.999).map(|t| m.tri_area(t)).sum();
    let want = PI * (5.5f64.powi(2) - 5.0f64.powi(2));
    assert!((back - want).abs() < 0.05 * want, "back face {back:.3} vs {want:.3}");
    // The lumen is the surface itself, untouched.
    let lumen: Vec<f64> = m.vertex_ids().map(|v| m.vertex(v)).filter(|p| p.truncate().length() < 4.0 && p.z > 0.5 && p.z < 29.5).map(|p| p.truncate().length()).collect();
    assert!(!lumen.is_empty() && lumen.iter().all(|r| (r - 3.0).abs() < 1e-9));

    // Inward and centred, the barb stands on the wall's outside there; at the end itself (no
    // setback) it is cut flat with the rim.
    for (side, outside) in [(ThickenSide::Inward, 3.0), (ThickenSide::Centered, 4.0)] {
        let barb = crate::Barb { setback: 0.0, ..barb };
        let res = thicken(&m0, &ThickenParams { thickness: 2.0, side, barbs: vec![(top, barb)], ..Default::default() }, None).unwrap();
        let m = &res.mesh;
        assert_valid(m);
        assert!(m.is_closed(), "{side:?}");
        assert_eq!(crossings(m), 0, "{side:?}");
        let crest: Vec<DVec3> = m.vertex_ids().map(|v| m.vertex(v)).filter(|p| (p.truncate().length() - (outside + 0.5)).abs() < 1e-9).collect();
        assert!(crest.len() >= 48 && crest.iter().all(|p| (p.z - (30.1 - barb.length)).abs() < 1e-9), "{side:?}");
        let zmax = m.vertex_ids().map(|v| m.vertex(v).z).fold(f64::MIN, f64::max);
        assert!((zmax - 30.1).abs() < 1e-9, "{side:?}: the rim cuts the barb flat ({zmax})");
    }

    // A barb needs a straight round tube behind its end: not one widening there.
    let n = 120;
    let points: Vec<DVec3> = (0..=n).map(|k| straight(30.0 * k as f64 / n as f64)).collect();
    let radii: Vec<f64> = points.iter().map(|p| if p.z > 27.0 { 3.0 + (p.z - 27.0) / 3.0 } else { 3.0 }).collect();
    let flared = primitives::tube_along(&points, &radii, 48);
    let top = openings(&flared).iter().position(|(_, o)| o.normal.z > 0.9).unwrap();
    let e = thicken(&flared, &ThickenParams { thickness: 2.0, barbs: vec![(top, barb)], ..Default::default() }, None).unwrap_err();
    assert!(e.to_string().contains("straight round tube"), "{e}");
    // Nor on an end that is not there, nor two on one end.
    assert!(thicken(&m0, &ThickenParams { barbs: vec![(7, barb)], ..Default::default() }, None).is_err());
    assert!(thicken(&m0, &ThickenParams { barbs: vec![(0, barb), (0, barb)], ..Default::default() }, None).is_err());
}
