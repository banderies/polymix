use super::*;
use crate::stent::*;
use crate::{mesh_boolean, BooleanOp, EndContinuity};

/// A vessel along `axis(s)` for s in 0..`len` with radius `radius(s)`: the open tube and its
/// centerline.
fn vessel(axis: impl Fn(f64) -> DVec3, radius: impl Fn(f64) -> f64, len: f64, slices: usize) -> (Mesh, VesselPath) {
    let n = (len / 0.25).round() as usize;
    let points: Vec<DVec3> = (0..=n).map(|k| axis(len * k as f64 / n as f64)).collect();
    let radii: Vec<f64> = (0..=n).map(|k| radius(len * k as f64 / n as f64)).collect();
    (primitives::tube_along(&points, &radii, slices), VesselPath { points, radii })
}

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

fn straight(z: f64) -> DVec3 {
    DVec3::new(0.0, 0.0, z)
}

/// A straight vessel of radius 2 along z (0..60), narrowed to 1 at z = 30.
fn stenosed() -> (Mesh, VesselPath) {
    vessel(straight, |z| 2.0 - (-((z - 30.0) / 3.0).powi(2)).exp(), 60.0, 48)
}

fn sections_at(path: &VesselPath, a: f64, b: f64) -> [StentSection; 2] {
    // Sections at arc lengths `a` and `b` of a path sampled every 0.25.
    let at = |s: f64| {
        let i = (s / 0.25).round() as usize;
        StentSection { center: path.points[i], normal: (path.points[i + 1] - path.points[i - 1]).normalize() }
    };
    [at(a), at(b)]
}

const ARC: f64 = 20.0;

fn arc(s: f64) -> DVec3 {
    DVec3::new(ARC * (s / ARC).cos(), ARC * (s / ARC).sin(), 0.0)
}

fn bare(diameter: f64, length: f64) -> StentParams {
    StentParams { diameter, length, straightness: 0.0, stiffness: 1.0, remesh: false, ..Default::default() }
}

#[test]
fn stent_is_centred_on_the_sections() {
    // A 30 mm stent over 20 mm of vessel reaches 5 mm past each section; a 14 mm one stops
    // 3 mm short of each.
    let (_, path) = stenosed();
    let pl = place_stent(&path, sections_at(&path, 20.0, 40.0), &bare(4.0, 30.0)).unwrap();
    assert!((pl.centerline_length - 20.0).abs() < 0.05 && (pl.path_length - 20.0).abs() < 0.05, "{} {}", pl.centerline_length, pl.path_length);
    let [a, b] = pl.overhang();
    assert!((a - 5.0).abs() < 0.05 && (b - 5.0).abs() < 0.05, "{a} {b}");
    assert!(pl.point(0.0).distance(straight(15.0)) < 0.05 && pl.point(30.0).distance(straight(45.0)) < 0.05);
    // Picked in either order.
    let [s0, s1] = sections_at(&path, 20.0, 40.0);
    let pl = place_stent(&path, [s1, s0], &bare(4.0, 14.0)).unwrap();
    let [a, b] = pl.overhang();
    assert!((a + 3.0).abs() < 0.05 && (b + 3.0).abs() < 0.05, "{a} {b}");
}

#[test]
fn bare_stent_opens_the_narrowing_and_nothing_else() {
    let (m0, path) = stenosed();
    let sections = sections_at(&path, 20.0, 40.0);
    let radius = |p: DVec3| p.truncate().length();
    // Stiffness 1: a round lumen of the stent's radius from 15 to 45; past the end blend
    // (2) nothing moves, and nothing moves along the vessel.
    let mut m = m0.clone();
    let params = bare(4.0, 30.0);
    let pl = place_stent(&path, sections, &params).unwrap();
    let rep = apply_stent(&mut m, &pl, &params, None).unwrap();
    assert_valid(&m);
    assert!((rep.narrowest[0] - 1.0).abs() < 0.05 && (rep.narrowest[1] - 2.0).abs() < 0.05, "{:?}", rep.narrowest);
    for v in m.vertex_ids() {
        let (p, q) = (m0.vertex(v), m.vertex(v));
        assert!((p.z - q.z).abs() < 1e-9, "moved along the vessel at z = {}", p.z);
        if (15.5..44.5).contains(&p.z) {
            assert!(radius(q) > 1.98 && radius(q) < 2.03, "radius {} at z = {}", radius(q), p.z);
        }
        if !(12.5..47.5).contains(&p.z) {
            assert!(p.distance(q) < 1e-12, "moved at z = {}", p.z);
        }
    }
    // Stiffness ½ opens the narrowing half way; stiffness 0 and following the vessel change
    // nothing.
    let mut m = m0.clone();
    let half = StentParams { stiffness: 0.5, ..params };
    let rep = apply_stent(&mut m, &pl, &half, None).unwrap();
    assert!((rep.narrowest[1] - 1.5).abs() < 0.03, "{:?}", rep.narrowest);
    let mut m = m0.clone();
    let none = StentParams { stiffness: 0.0, ..params };
    let rep = apply_stent(&mut m, &place_stent(&path, sections, &none).unwrap(), &none, None).unwrap();
    assert_eq!(rep.moved, 0);
    // Remeshed, the lumen stays open and the mesh even.
    let mut m = m0.clone();
    let remeshed = StentParams { remesh: true, ..params };
    apply_stent(&mut m, &pl, &remeshed, None).unwrap();
    assert_valid(&m);
    let narrowest = m.vertex_ids().map(|v| m.vertex(v)).filter(|p| (16.0..44.0).contains(&p.z)).map(radius).fold(f64::INFINITY, f64::min);
    assert!(narrowest > 1.95, "remeshed lumen {narrowest}");
    assert_eq!(boundary_loops(&m).len(), 2);
}

#[test]
fn smoothest_path_between_sections_on_an_arc_is_the_arc() {
    // Sections square to an arc: the G1 / G2 path of least bending is the arc itself, G0 the
    // chord.
    let (_, path) = vessel(arc, |_| 1.0, 30.0, 24);
    let sections = sections_at(&path, 7.0, 23.0);
    for continuity in [EndContinuity::G1, EndContinuity::G2] {
        let params = StentParams { diameter: 2.0, length: 16.0, straightness: 1.0, continuity, ..bare(2.0, 16.0) };
        let pl = place_stent(&path, sections, &params).unwrap();
        let off = (0..pl.s.len()).filter(|&i| (0.0..=16.0).contains(&pl.s[i])).map(|i| (pl.axis[i].length() - ARC).abs()).fold(0.0, f64::max);
        assert!(off < 0.03, "{continuity:?}: {off}");
        assert!((pl.path_length - 16.0).abs() < 0.05, "{continuity:?}: {}", pl.path_length);
        let t = pl.tangent(pl.sections[0]);
        assert!(t.dot(sections[0].normal).abs() > 0.999, "{continuity:?} leaves along the normal");
    }
    let params = StentParams { straightness: 1.0, continuity: EndContinuity::G0, ..bare(2.0, 16.0) };
    let pl = place_stent(&path, sections, &params).unwrap();
    let chord = sections[0].center.distance(sections[1].center);
    assert!((pl.path_length - chord).abs() < 1e-6, "{} vs {chord}", pl.path_length);
    // The stent keeps its length, so on the shorter chord it reaches further past each end.
    let [a, b] = pl.overhang();
    assert!((a - 0.5 * (16.0 - chord)).abs() < 0.02 && (a - b).abs() < 1e-6, "{a} {b}");
}

#[test]
fn straightness_moves_a_bent_vessel_onto_the_stent() {
    let (m0, path) = vessel(arc, |_| 1.0, 30.0, 24);
    let sections = sections_at(&path, 7.0, 23.0);
    let middle: Vec<u32> = m0.vertex_ids().filter(|&v| (m0.vertex(v).y.atan2(m0.vertex(v).x) - 0.75).abs() < 0.012).collect();
    let centre = |m: &Mesh| middle.iter().map(|&v| m.vertex(v)).sum::<DVec3>() / middle.len() as f64;
    let sagitta = ARC * (1.0 - 0.4f64.cos());
    for (w, want) in [(1.0, sagitta), (0.5, 0.5 * sagitta)] {
        let mut m = m0.clone();
        let params = StentParams { straightness: w, continuity: EndContinuity::G0, ..bare(2.0, 16.0) };
        let pl = place_stent(&path, sections, &params).unwrap();
        let rep = apply_stent(&mut m, &pl, &params, None).unwrap();
        assert_valid(&m);
        assert!(!rep.outside);
        assert_eq!(crossings(&m), 0);
        let moved = ARC - centre(&m).length();
        assert!((moved - want).abs() < 0.12 * sagitta, "straightness {w}: moved {moved:.3} of {want:.3}");
        // Moved as a whole: the ring stays round about the stent's axis.
        let c = centre(&m);
        let radii: Vec<f64> = middle.iter().map(|&v| m.vertex(v).distance(c)).collect();
        let (lo, hi) = (radii.iter().copied().fold(f64::INFINITY, f64::min), radii.iter().copied().fold(0.0, f64::max));
        assert!(lo > 0.93 && hi < 1.07, "ring radii {lo:.3}..{hi:.3}");
    }
}

#[test]
fn floating_stent_end_leaves_a_wide_lumen_alone() {
    // The stent reaches from a vessel of its own size into a wide one (an "elephant trunk"):
    // straightened, it carries the narrow stretch with it but not the wide lumen it floats
    // in, whose wall it never touches (the straight stent's tip ends up 3.6 off the
    // centerline there).
    let width = |s: f64| 1.0 + 5.0 * smoothstep((s - 18.0) / 4.0);
    let (m0, path) = vessel(arc, width, 32.0, 32);
    let sections = sections_at(&path, 6.0, 16.0);
    let mut m = m0.clone();
    let params = StentParams { straightness: 1.0, continuity: EndContinuity::G0, ..bare(2.0, 26.0) };
    let pl = place_stent(&path, sections, &params).unwrap();
    apply_stent(&mut m, &pl, &params, None).unwrap();
    assert_valid(&m);
    assert_eq!(crossings(&m), 0);
    let along = |p: DVec3| p.y.atan2(p.x) * ARC;
    let (mut narrow, mut wide): (f64, f64) = (0.0, 0.0);
    for v in m.vertex_ids() {
        let (p, d) = (m0.vertex(v), m0.vertex(v).distance(m.vertex(v)));
        if (9.0..13.0).contains(&along(p)) {
            narrow = narrow.max(d);
        }
        if along(p) > 23.5 {
            wide = wide.max(d);
        }
    }
    assert!(narrow > 0.3, "the narrow stretch straightens ({narrow:.3})");
    assert!(wide < 0.05, "the wide lumen stays put ({wide:.3})");
}

fn smoothstep(x: f64) -> f64 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

#[test]
fn covered_stent_lines_a_dilated_stretch() {
    // A fusiform dilation (radius 2 widening to 4): covered, the lumen is the stent's tube.
    let (mut m, path) = vessel(straight, |z| 2.0 + 2.0 * (-((z - 30.0) / 5.0).powi(2)).exp(), 60.0, 48);
    let params = StentParams { covered: true, remesh: true, ..bare(4.0, 30.0) };
    let pl = place_stent(&path, sections_at(&path, 20.0, 40.0), &params).unwrap();
    let rep = apply_stent(&mut m, &pl, &params, None).unwrap();
    assert_valid(&m);
    assert!(rep.removed_triangles > 0);
    assert_eq!(crossings(&m), 0);
    assert_eq!(boundary_loops(&m).len(), 2, "only the vessel's own ends stay open");
    let widest = m.vertex_ids().map(|v| m.vertex(v)).filter(|p| (16.0..44.0).contains(&p.z)).map(|p| p.truncate().length()).fold(0.0, f64::max);
    assert!(widest < 2.1, "lined to the stent ({widest:.3})");
    let widest_outside = m.vertex_ids().map(|v| m.vertex(v)).filter(|p| p.z < 12.0 || p.z > 48.0).map(|p| p.truncate().length()).fold(0.0, f64::max);
    assert!(widest_outside < 2.1 && widest_outside > 1.9);
}

/// A closed vessel along z (radius 2, 0..60) with a side branch (radius 1) along x at z = 30.
fn branched() -> (Mesh, VesselPath) {
    let main = primitives::cylinder(2.0, 60.0, 48, 120, true);
    let mut side = primitives::cylinder(1.0, 10.0, 24, 20, true);
    side.transform(|p| DVec3::new(p.z, p.y, 30.0 - p.x));
    let m = mesh_boolean(&main, &side, BooleanOp::Union).unwrap().mesh;
    let points: Vec<DVec3> = (0..=240).map(|k| straight(k as f64 * 0.25)).collect();
    (m, VesselPath { radii: vec![2.0; points.len()], points })
}

#[test]
fn bare_stent_keeps_a_side_branch_open() {
    let (mut m, path) = branched();
    let params = StentParams { remesh: true, ..bare(4.4, 30.0) };
    let pl = place_stent(&path, sections_at(&path, 20.0, 40.0), &params).unwrap();
    apply_stent(&mut m, &pl, &params, None).unwrap();
    assert_valid(&m);
    assert!(m.is_closed());
    assert_eq!(crossings(&m), 0);
    assert!(m.vertex_ids().any(|v| m.vertex(v).x > 9.0), "the branch is still there");
}

#[test]
fn covered_stent_seals_off_a_side_branch() {
    let (mut m, path) = branched();
    let params = StentParams { covered: true, remesh: true, ..bare(4.0, 30.0) };
    let pl = place_stent(&path, sections_at(&path, 20.0, 40.0), &params).unwrap();
    let rep = apply_stent(&mut m, &pl, &params, None).unwrap();
    assert_valid(&m);
    assert!(m.is_closed(), "sealed, not left open");
    assert_eq!(crossings(&m), 0);
    assert!(rep.removed_triangles > 0);
    let reach = m.vertex_ids().map(|v| m.vertex(v).x).fold(f64::NEG_INFINITY, f64::max);
    assert!(reach < 2.2, "the branch is gone ({reach:.2})");
    assert_eq!(connected_components(&m).len(), 1);
}


