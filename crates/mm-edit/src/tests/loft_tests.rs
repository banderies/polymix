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
        let params = LoftParams { shape, ends: [LoftEnd { continuity: ea, tension: 1.0 }, LoftEnd { continuity: eb, tension: 1.0 }], ..Default::default() };
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
        let params = LoftParams { shape, ends: [LoftEnd { continuity, tension: 1.0 }; 2], edge_length: Some(h), ..Default::default() };
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
        let params = LoftParams { ends: [LoftEnd { continuity: EndContinuity::G1, tension }; 2], ..Default::default() };
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
