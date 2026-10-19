use super::*;
use crate::*;
use mm_core::DVec3;

/// Two open tubes along z with a gap, the second shifted sideways by `offset`: returns the
/// mesh, the first tube's top loop and the second tube's bottom loop.
fn gap_tubes(n1: usize, n2: usize, gap: f64, offset: f64) -> (Mesh, Vec<u32>, Vec<u32>) {
    let mut m = primitives::cylinder(0.5, 2.0, n1, 6, false);
    let mut t2 = primitives::cylinder(0.6, 2.0, n2, 6, false);
    t2.transform(|p| p + DVec3::new(offset, 0.0, 2.0 + gap));
    m.append_mesh(&t2, 1);
    let loops = boundary_loops(&m);
    let at = |z: f64| loops.iter().find(|l| l.vertices.iter().all(|&v| (m.vertex(v).z - z).abs() < 1e-9)).unwrap().vertices.clone();
    (m.clone(), at(2.0), at(2.0 + gap))
}

fn ring_centre(m: &Mesh, verts: &[u32]) -> DVec3 {
    verts.iter().map(|&v| m.vertex(v)).sum::<DVec3>() / verts.len() as f64
}

fn perimeter(m: &Mesh, lp: &[u32]) -> f64 {
    (0..lp.len()).map(|i| m.vertex(lp[i]).distance(m.vertex(lp[(i + 1) % lp.len()]))).sum()
}

#[test]
fn loft_shapes_and_ends_make_valid_tubes() {
    use EndContinuity::*;
    let cases = [
        (LoftShape::Minimal, G1, G1),
        (LoftShape::Straight, G1, G1),
        (LoftShape::Smooth, G1, G1),
        (LoftShape::Smooth, G0, G1),
        (LoftShape::Smooth, G1, G0),
        (LoftShape::Smooth, G0, G0),
    ];
    for (shape, ea, eb) in cases {
        let (mut m, a, b) = gap_tubes(16, 24, 1.5, 0.8);
        let params = LoftParams { shape, ends: [LoftEnd { continuity: ea, tension: 1.0, ..Default::default() }, LoftEnd { continuity: eb, tension: 1.0, ..Default::default() }], ..Default::default() };
        let r = loft(&mut m, &a, &b, &params).unwrap();
        assert_valid(&m);
        assert_eq!(connected_components(&m).len(), 1, "{shape:?} {ea:?} {eb:?}");
        assert_eq!(boundary_loops(&m).len(), 2);
        assert_eq!(r.new_vertices.is_empty(), shape == LoftShape::Minimal);
        let loops: Vec<Vec<u32>> = boundary_loops(&m).into_iter().map(|l| l.vertices).collect();
        cap_planar_loops(&mut m, &loops, None, CutFill::Flat).unwrap();
        assert!(m.is_closed() && volume(&m) > 0.0, "{shape:?} {ea:?} {eb:?}");
    }
}

#[test]
fn g1_ends_leave_along_the_wall() {
    // The second tube is shifted sideways: a straight loft leaves the first tube at an angle,
    // a G1 end leaves along the tube's axis (z).
    let lean = |shape, continuity| {
        let (mut m, a, b) = gap_tubes(24, 24, 2.0, 1.2);
        let h = 0.1;
        // Rows are added in order, each with one point per perimeter / edge length.
        let n = (perimeter(&m, &a).max(perimeter(&m, &b)) / h).round() as usize;
        let params = LoftParams { shape, ends: [LoftEnd { continuity, tension: 1.0, ..Default::default() }; 2], edge_length: Some(h), ..Default::default() };
        let r = loft(&mut m, &a, &b, &params).unwrap();
        let d = ring_centre(&m, &r.new_vertices[..n]) - ring_centre(&m, &a);
        d.normalize().dot(DVec3::Z).acos().to_degrees()
    };
    let straight = lean(LoftShape::Straight, EndContinuity::G1);
    let g0 = lean(LoftShape::Smooth, EndContinuity::G0);
    let g1 = lean(LoftShape::Smooth, EndContinuity::G1);
    assert!(straight > 25.0, "straight leaves at an angle ({straight:.1}°)");
    assert!(g1 < 6.0, "G1 leaves along the wall ({g1:.1}°)");
    assert!(g0 > g1 + 10.0, "G0 is free to turn ({g0:.1}° vs {g1:.1}°)");
}

#[test]
fn loft_turns_between_openings_facing_the_same_way() {
    // Two tubes side by side, both open at the top: a G1 loft arches over between them.
    let mut m = primitives::cylinder(0.5, 2.0, 20, 4, false);
    let mut t2 = primitives::cylinder(0.5, 2.0, 20, 4, false);
    t2.transform(|p| p + DVec3::new(3.0, 0.0, 0.0));
    m.append_mesh(&t2, 1);
    let tops: Vec<Vec<u32>> = boundary_loops(&m).into_iter().filter(|l| l.vertices.iter().all(|&v| (m.vertex(v).z - 2.0).abs() < 1e-9)).map(|l| l.vertices).collect();
    assert_eq!(tops.len(), 2);
    let arch = |tension: f64| {
        let mut m = m.clone();
        let params = LoftParams { ends: [LoftEnd { continuity: EndContinuity::G1, tension, ..Default::default() }; 2], ..Default::default() };
        let r = loft(&mut m, &tops[0], &tops[1], &params).unwrap();
        assert_valid(&m);
        assert_eq!(connected_components(&m).len(), 1);
        r.new_vertices.iter().map(|&v| m.vertex(v).z).fold(f64::MIN, f64::max)
    };
    let (low, high) = (arch(1.0), arch(2.0));
    assert!(low > 2.6, "the tube arches up ({low:.2})");
    assert!(high > low + 0.3, "more tension, higher arch ({high:.2} vs {low:.2})");
}

#[test]
fn loft_twist_and_bulge() {
    let (base, a, b) = gap_tubes(20, 20, 2.0, 0.0);
    let mid_radius = |params: &LoftParams| {
        let mut m = base.clone();
        let r = loft(&mut m, &a, &b, params).unwrap();
        assert_valid(&m);
        let mid: Vec<u32> = r.new_vertices.iter().copied().filter(|&v| (m.vertex(v).z - 3.0).abs() < 0.15).collect();
        let c = ring_centre(&m, &mid);
        mid.iter().map(|&v| m.vertex(v).distance(c)).sum::<f64>() / mid.len() as f64
    };
    let plain = mid_radius(&LoftParams { shape: LoftShape::Straight, ..Default::default() });
    let wide = mid_radius(&LoftParams { shape: LoftShape::Straight, bulge: 0.5, ..Default::default() });
    let twisted = mid_radius(&LoftParams { shape: LoftShape::Straight, twist_deg: 90.0, ..Default::default() });
    assert!((plain - 0.55).abs() < 0.03, "halfway between radii 0.5 and 0.6 ({plain:.3})");
    assert!((wide / plain - 1.5).abs() < 0.05, "bulge 0.5 is 50 % wider ({:.3})", wide / plain);
    assert!(twisted < plain - 0.05, "a twist pulls the middle in ({twisted:.3})");
    let mut m = base.clone();
    assert!(loft(&mut m, &a, &b, &LoftParams { bulge: -1.0, ..Default::default() }).is_err());
}

/// An arc-shaped vessel (radius `big` in the xy-plane, tube radius `small`) with the section
/// between `gap.0` and `gap.1` degrees missing: the mesh and the two openings facing the gap.
fn arc_with_gap(big: f64, small: f64, gap: (f64, f64)) -> (Mesh, Vec<u32>, Vec<u32>) {
    let arc = |a0: f64, a1: f64| {
        let path: Vec<DVec3> = (0..=40).map(|k| {
            let a = (a0 + (a1 - a0) * k as f64 / 40.0).to_radians();
            DVec3::new(big * a.cos(), big * a.sin(), 0.0)
        }).collect();
        primitives::tube_along(&path, &vec![small; path.len()], 32)
    };
    let mut m = arc(0.0, gap.0);
    m.append_mesh(&arc(gap.1, gap.0 + gap.1), 1);
    let angle = |l: &mm_core::topology::EdgeLoop| { let c = l.centroid(&m); c.y.atan2(c.x).to_degrees() };
    let loops = boundary_loops(&m);
    let near = |deg: f64| loops.iter().min_by(|a, b| (angle(a) - deg).abs().total_cmp(&(angle(b) - deg).abs())).unwrap().vertices.clone();
    (m.clone(), near(gap.0), near(gap.1))
}

#[test]
fn g2_ends_follow_the_wall_bend() {
    let (big, small) = (6.0, 0.6);
    let (base, a, b) = arc_with_gap(big, small, (50.0, 90.0));
    // How far the new wall strays from the ideal torus the vessel follows.
    let stray = |c: EndContinuity| {
        let mut m = base.clone();
        let params = LoftParams { ends: [LoftEnd { continuity: c, tension: 1.0, ..Default::default() }; 2], edge_length: Some(0.1), ..Default::default() };
        let r = loft(&mut m, &a, &b, &params).unwrap();
        assert_valid(&m);
        r.new_vertices.iter().map(|&v| {
            let p = m.vertex(v);
            let d = ((p.x * p.x + p.y * p.y).sqrt() - big).hypot(p.z);
            (d - small).abs()
        }).fold(0.0, f64::max)
    };
    let (g1, g2) = (stray(EndContinuity::G1), stray(EndContinuity::G2));
    // The baseline is the mesh's own polygonal cross-section (0.003 for 32 sides).
    assert!(g2 < 0.005, "G2 follows the torus ({g2:.4})");
    assert!(g1 < 0.012 && g2 < g1, "G2 beats G1 on a bend ({g2:.4} vs {g1:.4})");
}

#[test]
fn loft_mixed_ends_round_and_fair() {
    use EndContinuity::*;
    // Every mix of end settings, with and without fairing, gives a valid closed tube.
    for (ea, eb) in [(G2, G2), (G0, G2), (G2, G1), (G0, G0)] {
        for fair in [0.0, 1.0] {
            let (mut m, a, b) = gap_tubes(16, 24, 1.5, 0.8);
            let params = LoftParams { ends: [LoftEnd { continuity: ea, tension: 1.0, ..Default::default() }, LoftEnd { continuity: eb, tension: 1.2, ..Default::default() }], fair, ..Default::default() };
            loft(&mut m, &a, &b, &params).unwrap();
            assert_valid(&m);
            assert_eq!(boundary_loops(&m).len(), 2, "{ea:?} {eb:?} fair {fair}");
            let loops: Vec<Vec<u32>> = boundary_loops(&m).into_iter().map(|l| l.vertices).collect();
            cap_planar_loops(&mut m, &loops, None, CutFill::Flat).unwrap();
            assert!(m.is_closed() && volume(&m) > 0.0, "{ea:?} {eb:?} fair {fair}");
        }
    }
}

#[test]
fn loft_round_makes_the_middle_circular() {
    // Elliptical openings: Round pulls the middle cross-section to a circle.
    let (mut base, a, b) = gap_tubes(32, 32, 2.0, 0.0);
    base.transform(|p| DVec3::new(p.x * 1.6, p.y, p.z));
    let spread = |round: f64| {
        let mut m = base.clone();
        let r = loft(&mut m, &a, &b, &LoftParams { shape: LoftShape::Straight, round, ..Default::default() }).unwrap();
        assert_valid(&m);
        let mid: Vec<u32> = r.new_vertices.iter().copied().filter(|&v| (m.vertex(v).z - 3.0).abs() < 0.08).collect();
        let c = ring_centre(&m, &mid);
        let d: Vec<f64> = mid.iter().map(|&v| (m.vertex(v) - c).truncate().length()).collect();
        let mean = d.iter().sum::<f64>() / d.len() as f64;
        d.iter().map(|x| (x - mean).abs()).fold(0.0, f64::max) / mean
    };
    let (oval, round) = (spread(0.0), spread(1.0));
    assert!(oval > 0.2, "the openings are oval ({oval:.3})");
    assert!(round < 0.02, "Round 100 % gives a circle ({round:.3})");
}

#[test]
fn faired_g1_tube_keeps_its_ends_and_size() {
    // Fairing (bi-Laplacian here) keeps G1 ends leaving along the wall and does not pinch.
    let (mut m, a, b) = gap_tubes(24, 24, 2.0, 1.2);
    let h = 0.1;
    let n = (perimeter(&m, &a).max(perimeter(&m, &b)) / h).round() as usize;
    let params = LoftParams { edge_length: Some(h), fair: 1.0, ..Default::default() };
    let r = loft(&mut m, &a, &b, &params).unwrap();
    assert_valid(&m);
    let d = ring_centre(&m, &r.new_vertices[..n]) - ring_centre(&m, &a);
    let lean = d.normalize().dot(DVec3::Z).acos().to_degrees();
    let rows = r.new_vertices.len() / n;
    let mid = &r.new_vertices[(rows / 2) * n..(rows / 2 + 1) * n];
    let c = ring_centre(&m, mid);
    let radius = mid.iter().map(|&v| m.vertex(v).distance(c)).sum::<f64>() / n as f64;
    assert!(lean < 6.0, "G1 kept ({lean:.1}°)");
    assert!(radius > 0.45, "no pinch ({radius:.3}, ends 0.5 and 0.6)");
}

#[test]
fn normal_ends_stand_up_from_a_surface() {
    // Two holes in the top of a sphere. Leaving out of the openings, the tube arches over the
    // surface like a handle; leaving along the wall (which runs across a hole in a surface,
    // not away from it) it dives into the sphere.
    let mut base = primitives::icosphere(2.0, 4);
    let centres = [DVec3::new(-0.9, 0.0, 1.8).normalize() * 2.0, DVec3::new(0.9, 0.0, 1.8).normalize() * 2.0];
    let holes: Vec<u32> = base.triangle_ids().filter(|&t| centres.iter().any(|c| base.tri_centroid(t).distance(*c) < 0.4)).collect();
    base.remove_triangles(holes);
    let loops: Vec<Vec<u32>> = boundary_loops(&base).into_iter().map(|l| l.vertices).collect();
    assert_eq!(loops.len(), 2);
    let lowest = |direction: EndDirection| {
        let mut m = base.clone();
        let params = LoftParams { ends: [LoftEnd { direction, ..Default::default() }; 2], ..Default::default() };
        let r = loft(&mut m, &loops[0], &loops[1], &params).unwrap();
        assert_valid(&m);
        assert!(m.is_closed());
        r.new_vertices.iter().map(|&v| m.vertex(v).length()).fold(f64::MAX, f64::min)
    };
    let (normal, wall) = (lowest(EndDirection::Normal), lowest(EndDirection::Wall));
    assert!(normal > 2.0, "out of the openings: above the surface ({normal:.3})");
    assert!(wall < 1.9, "along the wall: into the sphere ({wall:.3})");
}

/// A lumpy tube along z with its middle cut out along jagged lines, as a brush selection
/// leaves it, and the wall flaring just before the lower cut: returns the mesh and the two
/// openings of the gap.
fn ragged_gap() -> (Mesh, Vec<u32>, Vec<u32>) {
    let path: Vec<DVec3> = (0..=80).map(|k| DVec3::new(0.0, 0.0, k as f64 * 0.25)).collect();
    let radii: Vec<f64> = path.iter().map(|p| 1.0 + 0.05 * (p.z * 2.3).sin() + 0.35 * (-((p.z - 8.6) / 0.6).powi(2)).exp()).collect();
    let mut m = primitives::tube_along(&path, &radii, 40);
    let jag = |p: DVec3, k: f64| 0.35 * (p.y.atan2(p.x) * k).sin() * (p.y.atan2(p.x) * 3.0).cos();
    let cut: Vec<u32> = m.triangle_ids().filter(|&t| {
        let c = m.tri_centroid(t);
        c.z > 9.0 + jag(c, 7.0) && c.z < 11.0 + jag(c, 5.0)
    }).collect();
    m.remove_triangles(cut);
    let mut gap: Vec<Vec<u32>> = boundary_loops(&m).into_iter().map(|l| l.vertices).filter(|l| l.iter().all(|&v| (5.0..15.0).contains(&m.vertex(v).z))).collect();
    assert_eq!(gap.len(), 2);
    let (b, a) = (gap.pop().unwrap(), gap.pop().unwrap());
    (m, a, b)
}

/// Sharpest bend (degrees) between neighbouring triangles of `tris`.
fn sharpest_bend(m: &Mesh, tris: &[u32]) -> f64 {
    let set: std::collections::HashSet<u32> = tris.iter().copied().collect();
    m.edge_ids()
        .filter_map(|e| {
            let [s, t] = m.edge_triangles(e);
            (t != mm_core::INVALID && set.contains(&s) && set.contains(&t)).then(|| m.tri_normal(s).angle_between(m.tri_normal(t)).to_degrees())
        })
        .fold(0.0, f64::max)
}

#[test]
fn context_and_blend_join_ragged_openings_without_folds() {
    // Ragged openings fold the edge-only loft (each point's direction comes from one row of
    // triangles, and a notch's edges point round the tube, not along it). With the vessel's
    // axis behind each opening steering the tube, tidied openings and a smoothed, remeshed
    // join, the tube neither folds nor crosses the openings' notches.
    let (mut m, a, b) = ragged_gap();
    let params = LoftParams { ends: [LoftEnd { context: 0.5, ..Default::default() }; 2], blend: 0.5, remesh: true, ..Default::default() };
    let r = loft(&mut m, &a, &b, &params).unwrap();
    assert_valid(&m);
    assert_eq!(boundary_loops(&m).len(), 2, "only the tube's far ends stay open");
    let bend = sharpest_bend(&m, &r.triangles);
    assert!(bend < 40.0, "no folds or creases ({bend:.1}°)");
    // The tube follows the vessel (along z), not the flare just below the gap.
    let off_axis = r.new_vertices.iter().map(|&v| m.vertex(v).truncate().length()).fold(0.0, f64::max);
    assert!(off_axis < 1.45, "stays about the axis ({off_axis:.2})");
    // Without the vessel behind the openings, the same openings fold the tube.
    let (mut m, a, b) = ragged_gap();
    let r = loft(&mut m, &a, &b, &LoftParams::default()).unwrap();
    assert!(sharpest_bend(&m, &r.triangles) > 90.0);
}

#[test]
fn axis_fit_follows_a_bent_vessel() {
    // A tube bent along an arc of radius 10: the axis fitted behind its open end leaves along
    // the arc's tangent there and bends with curvature 1 / 10, whatever the end's own last
    // row of triangles does.
    let arc: Vec<DVec3> = (0..=40).map(|k| {
        let a = k as f64 / 40.0 * 1.2;
        DVec3::new(10.0 * a.cos(), 10.0 * a.sin(), 0.0)
    }).collect();
    let m = primitives::tube_along(&arc, &[1.0; 41], 32);
    let end = boundary_loops(&m).into_iter().map(|l| l.vertices).find(|l| l.iter().all(|&v| m.vertex(v).y > 5.0)).unwrap();
    let axis = super::super::loft::tests_support::vessel_axis(&m, &end, 8.0).unwrap();
    let tangent = DVec3::new(-(1.2f64).sin(), 1.2f64.cos(), 0.0);
    assert!(axis.0.angle_between(tangent).to_degrees() < 3.0, "direction {:?}", axis.0);
    assert!((axis.1 - 0.1).abs() < 0.02, "curvature {:.3}", axis.1);
}

/// A tube of radius 1 bent along an arc of radius 10 about z (0.0 to 1.4 rad), with the
/// stretch from 0.6 to 0.8 rad cut out: the mesh and the gap's two openings.
fn arc_gap() -> (Mesh, Vec<u32>, Vec<u32>) {
    let at = |a: f64| DVec3::new(10.0 * a.cos(), 10.0 * a.sin(), 0.0);
    let piece = |a0: f64, a1: f64| {
        let path: Vec<DVec3> = (0..=30).map(|k| at(a0 + (a1 - a0) * k as f64 / 30.0)).collect();
        primitives::tube_along(&path, &[1.0; 31], 32)
    };
    let mut m = piece(0.0, 0.6);
    m.append_mesh(&piece(0.8, 1.4), 1);
    let gap: Vec<Vec<u32>> = boundary_loops(&m).into_iter().map(|l| l.vertices).filter(|l| {
        let c = ring_centre(&m, l);
        let a = c.y.atan2(c.x);
        (0.5..0.9).contains(&a)
    }).collect();
    assert_eq!(gap.len(), 2);
    (m, gap[0].clone(), gap[1].clone())
}

/// Largest distance of the tube's new points from the arc of radius 10 (the true axis),
/// less the tube's radius 1.
fn off_arc(m: &Mesh, verts: &[u32]) -> f64 {
    verts.iter().map(|&v| {
        let p = m.vertex(v);
        let axis = DVec3::new(p.x, p.y, 0.0).normalize() * 10.0;
        (p.distance(axis) - 1.0).abs()
    }).fold(0.0, f64::max)
}

#[test]
fn sweep_follows_a_bent_vessel() {
    // Along the axis, the tube continues the arc across the gap — with the axes fitted to
    // the wall, or to the vessels' centerlines given as guides.
    let (mut m, a, b) = arc_gap();
    let params = LoftParams { shape: LoftShape::Sweep, ends: [LoftEnd { context: 0.5, ..Default::default() }; 2], ..Default::default() };
    let r = loft(&mut m, &a, &b, &params).unwrap();
    assert_valid(&m);
    let fitted = off_arc(&m, &r.new_vertices);
    assert!(fitted < 0.05, "on the vessel's wall ({fitted:.3})");
    let (mut m, a, b) = arc_gap();
    let guide = |a0: f64, step: f64| AxisGuide {
        points: (0..20).map(|k| {
            let t = a0 + step * k as f64;
            DVec3::new(10.0 * t.cos(), 10.0 * t.sin(), 0.0)
        }).collect(),
        radii: vec![1.0; 20],
    };
    // Which opening is at the lower angle decides which way each guide runs.
    let lower_first = { let c = ring_centre(&m, &a); c.y.atan2(c.x) < 0.7 };
    let (ga, gb) = if lower_first { (guide(0.6, -0.03), guide(0.8, 0.03)) } else { (guide(0.8, 0.03), guide(0.6, -0.03)) };
    let r = loft_guided(&mut m, &a, &b, &params, [Some(&ga), Some(&gb)]).unwrap();
    assert_valid(&m);
    let guided = off_arc(&m, &r.new_vertices);
    assert!(guided < 0.05, "guided by the centerlines ({guided:.3})");
    // Without a vessel behind an opening there is no axis to follow.
    let mut sphere = primitives::icosphere(2.0, 3);
    let caps: Vec<u32> = sphere.triangle_ids().filter(|&t| sphere.tri_centroid(t).z.abs() > 1.7).collect();
    sphere.remove_triangles(caps);
    let holes: Vec<Vec<u32>> = boundary_loops(&sphere).into_iter().map(|l| l.vertices).collect();
    let flat = loft(&mut sphere, &holes[0], &holes[1], &LoftParams { shape: LoftShape::Sweep, ..Default::default() });
    assert!(flat.is_err() || sphere.check_validity().is_ok());
}

#[test]
fn sweep_morphs_without_pinching_turned_openings() {
    // Two coaxial tubes, the second turned a quarter round against the first (twist 90°):
    // columns blended straight across cut the corner (the middle shrinks to about 70 % of
    // the radius); the sweep turns them round the axis and keeps the radius.
    let middle_radius = |shape: LoftShape| {
        let (mut m, a, b) = gap_tubes(24, 24, 2.0, 0.0);
        let params = LoftParams { shape, ends: [LoftEnd { context: 0.5, ..Default::default() }; 2], twist_deg: 90.0, edge_length: Some(0.1), ..Default::default() };
        let r = loft(&mut m, &a, &b, &params).unwrap();
        assert_valid(&m);
        r.new_vertices.iter().map(|&v| m.vertex(v)).filter(|p| (p.z - 3.0).abs() < 0.06).map(|p| p.truncate().length()).fold(f64::MAX, f64::min)
    };
    let (smooth, sweep) = (middle_radius(LoftShape::Smooth), middle_radius(LoftShape::Sweep));
    assert!(smooth < 0.45, "straight blending pinches ({smooth:.3}; ends 0.5 and 0.6)");
    assert!(sweep > 0.5, "the sweep keeps the size ({sweep:.3})");
}
