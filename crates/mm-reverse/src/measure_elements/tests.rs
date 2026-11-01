// SPDX-License-Identifier: AGPL-3.0-only
//! Ported from ScanRuler's `tests/construct.test.ts`, `tests/residualRange.test.ts`,
//! `tests/unitsCore.test.ts` (element text) and `plugins/reverse/tests/reFeatures.test.ts`
//! (coordinate plane offers), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only; the fits, names,
//! sections and drawings are tests of this port.

use glam::DVec3;

use super::*;
use crate::fit3d::testkit::{box_mesh, cylinder_mesh, graph_soup, seed_near};
use crate::fit3d::{CylinderFit, PlaneFit, SphereFit};
use crate::model::{Element, Part};

const SIZE: f64 = 100.0;

fn close(a: f64, b: f64, digits: i32) -> bool {
    (a - b).abs() < 10f64.powi(-digits) / 2.0
}

fn stats() -> FitBase {
    FitBase::default()
}

fn point(c: [f64; 3]) -> FitData {
    FitData::Point(PointFit { base: stats(), center: DVec3::from_array(c) })
}

fn sphere(c: [f64; 3], radius: f64) -> FitData {
    FitData::Sphere(SphereFit { base: FitBase { sigma: 0.001, used_points: 500, region_size: 600, ..FitBase::default() }, center: DVec3::from_array(c), radius, held: None })
}

fn cylinder(c: [f64; 3], axis: [f64; 3], length: f64) -> FitData {
    FitData::Cylinder(CylinderFit { base: FitBase { sigma: 0.001, used_points: 500, region_size: 600, ..FitBase::default() }, center: DVec3::from_array(c), axis: DVec3::from_array(axis), radius: 4.0, length, coverage: 360.0 })
}

fn plane(c: [f64; 3], n: [f64; 3], u: [f64; 3], v: [f64; 3]) -> FitData {
    FitData::Plane(PlaneFit {
        base: FitBase { sigma: 0.001, used_points: 500, region_size: 600, ..FitBase::default() },
        center: DVec3::from_array(c),
        normal: DVec3::from_array(n),
        basis_u: DVec3::from_array(u),
        basis_v: DVec3::from_array(v),
        extent_u: 10.0,
        extent_v: 10.0,
    })
}

fn line(c: [f64; 3], d: [f64; 3]) -> FitData {
    FitData::Line(LineFit { base: stats(), center: DVec3::from_array(c), dir: DVec3::from_array(d), length: 10.0 })
}

fn eval(method: &str, refs: &[FitData], params: &[f64]) -> Result<FitData, String> {
    evaluate_construction(method, refs, params, SIZE)
}

fn center(f: &FitData) -> DVec3 {
    f.center()
}

// ---- construct.test.ts ----

#[test]
fn builds_a_point_from_coordinates() {
    let p = eval("point-coords", &[], &[1.0, 2.0, 3.0]).unwrap();
    assert_eq!(p.kind(), ElementKind::Point);
    assert_eq!(center(&p), DVec3::new(1.0, 2.0, 3.0));
}

#[test]
fn builds_the_midpoint_of_two_sphere_centers() {
    let p = eval("point-midpoint", &[sphere([0.0, 0.0, 0.0], 5.0), sphere([10.0, 4.0, -2.0], 5.0)], &[]).unwrap();
    assert_eq!(center(&p), DVec3::new(5.0, 2.0, -1.0));
}

#[test]
fn pierces_a_plane_with_a_line() {
    let p = eval("point-line-plane", &[line([2.0, 3.0, 0.0], [0.0, 0.0, 1.0]), plane([0.0, 0.0, 7.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])], &[]).unwrap();
    let c = center(&p);
    assert!(close(c.x, 2.0, 9) && close(c.y, 3.0, 9) && close(c.z, 7.0, 9));
}

#[test]
fn pierces_a_tilted_plane_with_a_cylinder_axis_beyond_the_measured_extents() {
    let s = std::f64::consts::FRAC_1_SQRT_2;
    let p = eval("point-line-plane", &[cylinder([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 10.0), plane([20.0, 0.0, -20.0], [s, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, s])], &[]).unwrap();
    let c = center(&p);
    assert!(close(c.x, 0.0, 9) && close(c.y, 0.0, 9) && close(c.z, 0.0, 9));
}

#[test]
fn rejects_a_line_parallel_to_the_plane() {
    assert!(eval("point-line-plane", &[line([0.0, 0.0, 4.0], [1.0, 0.0, 0.0]), plane([0.0; 3], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])], &[]).is_err());
}

#[test]
fn builds_a_line_through_two_points() {
    let FitData::Line(l) = eval("line-two-points", &[point([0.0, 0.0, 0.0]), point([0.0, 0.0, 8.0])], &[]).unwrap() else { panic!("not a line") };
    assert!(close(l.dir.z, 1.0, 9));
    assert!(close(l.length, 8.0, 9));
    assert_eq!(l.center, DVec3::new(0.0, 0.0, 4.0));
}

#[test]
fn rejects_coincident_points() {
    assert!(eval("line-two-points", &[point([1.0, 1.0, 1.0]), point([1.0, 1.0, 1.0])], &[]).is_err());
}

#[test]
fn extracts_a_cylinder_axis() {
    let FitData::Line(l) = eval("line-axis", &[cylinder([1.0, 2.0, 3.0], [0.0, 1.0, 0.0], 30.0)], &[]).unwrap() else { panic!("not a line") };
    assert_eq!(l.center, DVec3::new(1.0, 2.0, 3.0));
    assert_eq!(l.dir, DVec3::new(0.0, 1.0, 0.0));
    assert!(close(l.length, 30.0, 9));
}

#[test]
fn intersects_two_perpendicular_planes_along_the_expected_direction() {
    let floor = plane([0.0, 0.0, 2.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    let wall = plane([3.0, 0.0, 10.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
    let FitData::Line(l) = eval("line-plane-plane", &[floor, wall], &[]).unwrap() else { panic!("not a line") };
    assert!(close(l.dir.y.abs(), 1.0, 9));
    assert!(close(l.center.x, 3.0, 9));
    assert!(close(l.center.z, 2.0, 9));
}

#[test]
fn rejects_parallel_planes() {
    let a = plane([0.0; 3], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    let b = plane([0.0, 0.0, 5.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    assert!(eval("line-plane-plane", &[a, b], &[]).is_err());
}

#[test]
fn spans_a_plane_through_three_points() {
    let FitData::Plane(p) = eval("plane-three-points", &[point([0.0, 0.0, 5.0]), point([10.0, 0.0, 5.0]), point([0.0, 10.0, 5.0])], &[]).unwrap() else { panic!("not a plane") };
    assert!(close(p.normal.z.abs(), 1.0, 9));
    assert!(close(p.center.z, 5.0, 9));
}

#[test]
fn rejects_collinear_points() {
    assert!(eval("plane-three-points", &[point([0.0; 3]), point([1.0, 0.0, 0.0]), point([2.0, 0.0, 0.0])], &[]).is_err());
}

#[test]
fn offsets_a_plane_along_its_normal() {
    let src = plane([0.0, 0.0, 2.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    let FitData::Plane(p) = eval("plane-offset", &[src], &[3.5]).unwrap() else { panic!("not a plane") };
    assert!(close(p.center.z, 5.5, 9));
    assert_eq!(p.normal, DVec3::Z);
    assert_eq!(p.base.used_points, 0);
}

#[test]
fn builds_the_midplane_of_two_opposing_faces() {
    let top = plane([0.0, 0.0, 10.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    let bottom = plane([0.0; 3], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]);
    let FitData::Plane(p) = eval("plane-midplane", &[top, bottom], &[]).unwrap() else { panic!("not a plane") };
    assert!(close(p.center.z, 5.0, 9));
    assert!(close(p.normal.z.abs(), 1.0, 9));
}

#[test]
fn refuses_a_midplane_between_perpendicular_planes() {
    let upright = plane([0.0, 0.0, 10.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    let side = plane([5.0, 0.0, 5.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
    assert!(eval("plane-midplane", &[upright, side], &[]).unwrap_err().contains("perpendicular"));
}

#[test]
fn builds_a_plane_from_coordinates_with_a_normalized_normal() {
    let FitData::Plane(p) = eval("plane-coords", &[], &[0.0, 0.0, 2.0, 1.0, 2.0, 3.0]).unwrap() else { panic!("not a plane") };
    assert_eq!(p.normal, DVec3::Z);
    assert_eq!(p.center, DVec3::new(1.0, 2.0, 3.0));
    assert!(p.extent_u > 0.0);
}

#[test]
fn rebuilds_a_centroid_point_from_its_carried_coordinates() {
    let r = eval("point-centroid", &[], &[12.5, -3.0, 7.25]).unwrap();
    assert_eq!(r.kind(), ElementKind::Point);
    assert_eq!(center(&r), DVec3::new(12.5, -3.0, 7.25));
    assert!(eval("point-centroid", &[], &[f64::NAN; 3]).unwrap_err().contains("not been measured"));
}

#[test]
fn rebuilds_a_symmetry_plane_with_the_mirror_residual_as_its_sigma() {
    let FitData::Plane(p) = eval("plane-symmetry", &[], &[0.0, 0.0, 2.0, 1.0, 2.0, 3.0, 0.0123, 3900.0, 4000.0]).unwrap() else { panic!("not a plane") };
    assert_eq!(p.normal, DVec3::Z);
    assert_eq!(p.center, DVec3::new(1.0, 2.0, 3.0));
    assert!(close(p.base.sigma, 0.0123, 9));
    assert_eq!(p.base.used_points, 3900);
    assert_eq!(p.base.region_size, 4000);
    assert!(close(p.extent_u, SIZE * 0.3, 9));
    assert!(eval("plane-symmetry", &[], &[f64::NAN; 9]).unwrap_err().contains("Find the symmetry plane"));
}

fn floor() -> FitData {
    plane([0.0; 3], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
}

#[test]
fn intersects_a_perpendicular_cylinder_with_a_plane() {
    let FitData::Circle(c) = eval("circle-plane-cylinder", &[floor(), cylinder([2.0, 3.0, 10.0], [0.0, 0.0, 1.0], 20.0)], &[]).unwrap() else { panic!("not a circle") };
    assert!(close(c.center.x, 2.0, 9) && close(c.center.y, 3.0, 9) && close(c.center.z, 0.0, 9));
    assert!(close(c.radius, 4.0, 9));
    assert!(close(c.normal.z.abs(), 1.0, 9));
}

#[test]
fn accepts_a_slightly_leaning_cylinder_and_keeps_its_radius() {
    let a = (2.0f64).to_radians().sin();
    let FitData::Circle(c) = eval("circle-plane-cylinder", &[floor(), cylinder([0.0, 0.0, 10.0], [a, 0.0, (1.0 - a * a).sqrt()], 20.0)], &[]).unwrap() else { panic!("not a circle") };
    assert!(close(c.radius, 4.0, 9));
}

#[test]
fn refuses_a_cylinder_leaning_past_the_ellipse_threshold() {
    let s = std::f64::consts::FRAC_1_SQRT_2;
    let e = eval("circle-plane-cylinder", &[floor(), cylinder([0.0, 0.0, 10.0], [s, 0.0, s], 20.0)], &[]).unwrap_err();
    assert!(e.contains("ellipse"), "{e}");
    assert!(e.contains("45.0°"), "{e}");
}

#[test]
fn refuses_a_cylinder_parallel_to_the_plane() {
    assert!(eval("circle-plane-cylinder", &[floor(), cylinder([0.0, 0.0, 10.0], [1.0, 0.0, 0.0], 20.0)], &[]).is_err());
}

#[test]
fn intersects_a_sphere_with_a_plane() {
    let FitData::Circle(c) = eval("circle-plane-sphere", &[floor(), sphere([1.0, 2.0, 3.0], 5.0)], &[]).unwrap() else { panic!("not a circle") };
    assert!(close(c.center.x, 1.0, 9) && close(c.center.y, 2.0, 9) && close(c.center.z, 0.0, 9));
    assert!(close(c.radius, 4.0, 9));
}

#[test]
fn refuses_a_plane_that_misses_the_sphere() {
    assert!(eval("circle-plane-sphere", &[floor(), sphere([0.0, 0.0, 9.0], 5.0)], &[]).unwrap_err().contains("misses"));
}

#[test]
fn builds_a_circle_from_coordinates() {
    let FitData::Circle(c) = eval("circle-coords", &[], &[12.0, 0.0, 0.0, 2.0, 5.0, 6.0, 7.0]).unwrap() else { panic!("not a circle") };
    assert!(close(c.radius, 6.0, 9));
    assert_eq!(c.normal, DVec3::Z);
    assert_eq!(c.center, DVec3::new(5.0, 6.0, 7.0));
}

#[test]
fn refuses_a_zero_or_negative_diameter() {
    assert!(eval("circle-coords", &[], &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]).is_err());
}

#[test]
fn uses_circle_centers_as_points_in_other_constructions() {
    let a = eval("circle-coords", &[], &[10.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]).unwrap();
    let b = eval("circle-coords", &[], &[10.0, 0.0, 0.0, 1.0, 8.0, 0.0, 0.0]).unwrap();
    let FitData::Line(l) = eval("line-two-points", &[a, b], &[]).unwrap() else { panic!("not a line") };
    assert!(close(l.length, 8.0, 9));
    assert!(close(l.dir.x.abs(), 1.0, 9));
}

// ---- residualRange.test.ts and unitsCore.test.ts ----

fn bore() -> FitData {
    FitData::Cylinder(CylinderFit {
        base: FitBase { sigma: 0.0288, used_points: 2075, region_size: 2075, form_error: Some(0.1486), residual_min: Some(-0.074), residual_max: Some(0.0746) },
        center: DVec3::ZERO,
        axis: DVec3::Z,
        radius: 4.1585,
        length: 22.042,
        coverage: 358.0,
    })
}

fn ball() -> FitData {
    FitData::Sphere(SphereFit { base: FitBase { sigma: 0.004, used_points: 9000, region_size: 9200, form_error: Some(0.03), residual_min: Some(-0.012), residual_max: Some(0.018) }, center: DVec3::new(1.0, 2.0, 3.0), radius: 12.7, held: None })
}

fn face() -> FitData {
    FitData::Plane(PlaneFit {
        base: FitBase { sigma: 0.0081, used_points: 4000, region_size: 4100, form_error: Some(0.05), residual_min: Some(-0.021), residual_max: Some(0.029) },
        center: DVec3::ZERO,
        normal: DVec3::Z,
        basis_u: DVec3::X,
        basis_v: DVec3::Y,
        extent_u: 10.0,
        extent_v: 8.0,
    })
}

#[test]
fn the_diameter_range_is_the_gaussian_radius_moved_to_the_radial_extremes_doubled() {
    let (min, max) = diameter_range(&bore()).unwrap();
    assert!(close(min, 2.0 * (4.1585 - 0.074), 9));
    assert!(close(max, 2.0 * (4.1585 + 0.0746), 9));
    assert_eq!(format_diameter_range(&bore(), DisplayUnits::Mm).as_deref(), Some("Ø min 8.169 · max 8.466 mm"));
}

#[test]
fn the_diameter_range_is_absent_for_a_plane_and_for_a_fit_without_residuals() {
    assert!(diameter_range(&face()).is_none());
    let mut b = ball();
    if let FitData::Sphere(s) = &mut b {
        s.base.residual_min = None;
        s.base.residual_max = None;
    }
    assert!(diameter_range(&b).is_none());
    assert!(format_diameter_range(&b, DisplayUnits::Mm).is_none());
}

#[test]
fn the_detail_line_carries_the_min_and_max_diameter_beside_the_form_error() {
    let line = format_detail(&bore(), DisplayUnits::Mm);
    assert!(line.contains("cylindricity 0.1486 mm"), "{line}");
    assert!(line.contains("Ø min 8.169 · max 8.466 mm"), "{line}");
    assert!(line.contains("length 22.042 mm"), "{line}");
    assert!(line.contains("2,075 of 2,075 points"), "{line}");
    assert!(format_detail(&ball(), DisplayUnits::Mm).contains("Ø min 25.376 · max 25.436 mm"));
    assert!(!format_detail(&face(), DisplayUnits::Mm).contains('Ø'));
}

#[test]
fn writes_the_headline_in_inches_four_decimals_for_three() {
    let mut inch_ball = ball();
    if let FitData::Sphere(s) = &mut inch_ball {
        s.radius = 12.7;
    }
    assert_eq!(format_primary(&inch_ball, DisplayUnits::In), "Ø 1.0000 in");
    assert_eq!(format_primary(&inch_ball, DisplayUnits::Mm), "Ø 25.400 mm");
    let mut b = bore();
    if let FitData::Cylinder(c) = &mut b {
        c.radius = 6.35;
        c.length = 50.8;
        c.base.sigma = 0.0254;
    }
    assert_eq!(format_primary(&b, DisplayUnits::In), "Ø 0.5000 in");
    let detail = format_detail(&b, DisplayUnits::In);
    assert!(detail.contains("σ 0.00100 in"), "{detail}");
    assert!(detail.contains("length 2.0000 in"), "{detail}");
    assert!(!detail.contains(" mm"), "{detail}");
}

#[test]
fn a_plane_reads_its_sigma_a_point_its_place_and_a_line_nothing() {
    assert_eq!(format_primary(&face(), DisplayUnits::Mm), "σ 0.0081 mm");
    assert_eq!(format_primary(&point([1.04, 2.0, -3.0]), DisplayUnits::Mm), "(1.0, 2.0, -3.0)");
    assert_eq!(format_primary(&line([0.0; 3], [0.0, 0.0, 1.0]), DisplayUnits::Mm), "");
    assert_eq!(format_detail(&line([0.0; 3], [0.0, 0.0, 1.0]), DisplayUnits::Mm), "direction (0.0000, 0.0000, 1.0000)");
    // A constructed plane has no sigma to show.
    let FitData::Plane(p) = face() else { unreachable!() };
    let made = FitData::Plane(PlaneFit { base: FitBase::default(), ..p });
    assert_eq!(format_primary(&made, DisplayUnits::Mm), "");
    assert_eq!(format_detail(&made, DisplayUnits::Mm), "20.00 × 16.00 mm patch");
}

// ---- reFeatures.test.ts: the coordinate planes on offer ----

#[test]
fn offers_the_coordinate_planes_at_zero_once_aligned_and_through_the_centre_with_a_scan() {
    let c = Some(DVec3::new(10.0, 20.0, 30.0));
    let unaligned = coordinate_plane_offers(c, false);
    assert_eq!(unaligned.iter().map(|o| o.label.as_str()).collect::<Vec<_>>(), ["YZ plane through the centre", "XZ plane through the centre", "XY plane through the centre"]);
    assert_eq!(unaligned.iter().map(|o| crate::model::plane_offset(&o.plane)).collect::<Vec<_>>(), [10.0, 20.0, 30.0]);
    let aligned = coordinate_plane_offers(c, true);
    assert_eq!(aligned.iter().map(|o| o.label.as_str()).collect::<Vec<_>>(), ["YZ plane", "XZ plane", "XY plane", "YZ plane through the centre", "XZ plane through the centre", "XY plane through the centre"]);
    assert_eq!(aligned.iter().map(|o| crate::model::plane_offset(&o.plane)).collect::<Vec<_>>(), [0.0, 0.0, 0.0, 10.0, 20.0, 30.0]);
    let keys: std::collections::HashSet<&str> = aligned.iter().map(|o| o.key.as_str()).collect();
    assert_eq!(keys.len(), 6);
    assert_eq!(coordinate_plane_offers(None, true).iter().map(|o| o.label.as_str()).collect::<Vec<_>>(), ["YZ plane", "XZ plane", "XY plane"]);
}

#[test]
fn keys_a_coordinate_plane_by_its_axis_and_its_place_to_the_micron() {
    assert_eq!(world_plane_key(WorldAxis::Z, 0.0), "world:z:0");
    assert_eq!(world_plane_key(WorldAxis::X, 12.34567), "world:x:12.346");
    assert_eq!(world_plane_key(WorldAxis::Y, -0.5), "world:y:-0.5");
    assert_eq!(plane_key(&PlaneRef::world(WorldAxis::Y, -0.5)), "world:y:-0.5");
    assert_eq!(plane_key(&PlaneRef::Plane { id: 4, offset: Some(2.0), up: None }), "plane:4");
}

// ---- this port: the part's list, sections, offers, fits and drawings ----

fn part_with(elements: Vec<Element>) -> Part {
    Part { elements, ..Part::default() }
}

#[test]
fn numbers_names_and_ids_follow_the_highest_in_the_part() {
    let mut part = Part::default();
    assert_eq!(next_element_name(&part, "Plane"), "Plane 1");
    let (g, s) = geometry_of(&face());
    let name = next_element_name(&part, "Plane");
    let a = add_element(&mut part, name, g.clone(), s.clone());
    let name = next_element_name(&part, "Plane");
    let b = add_element(&mut part, name, g.clone(), s);
    let name = next_element_name(&part, "Cylinder");
    let c = add_element(&mut part, name, g, None);
    assert_eq!((a, b, c), (1, 2, 3));
    assert_eq!(part.elements.iter().map(|e| e.name.as_str()).collect::<Vec<_>>(), ["Plane 1", "Plane 2", "Cylinder 1"]);
    // A "Planet" is no plane.
    part.elements[0].name = "Planet 9".into();
    assert_eq!(next_element_name(&part, "Plane"), "Plane 3");
    assert_eq!(element_color(1), 0x1877c0);
    assert_eq!(element_color(9), 0x1877c0);
    assert_eq!(element_color(2), 0xe8590c);
}

#[test]
fn an_element_keeps_its_fit_and_gives_it_back() {
    for fit in [bore(), ball(), face(), point([1.0, 2.0, 3.0]), line([0.0; 3], [0.0, 1.0, 0.0])] {
        let (g, s) = geometry_of(&fit);
        let e = Element { id: 1, name: "x".into(), geometry: g, stats: s };
        assert_eq!(fit_of(&e, SIZE), fit);
        assert_eq!(element_kind(&e.geometry), Some(fit.kind()));
    }
}

#[test]
fn a_section_stands_for_its_plane_and_is_offered_with_the_planes() {
    let mut part = part_with(Vec::new());
    let (g, s) = geometry_of(&bore());
    let cyl = add_element(&mut part, "Cylinder 1".into(), g, s);
    let frame = section_frame(SectionAlong::Element(cyl), 2.5, &part).unwrap();
    assert!((frame.normal - DVec3::Z).length() < 1e-12);
    assert!((frame.origin - DVec3::new(0.0, 0.0, 2.5)).length() < 1e-12);
    let world = section_frame(SectionAlong::World(WorldAxis::X), -4.0, &part).unwrap();
    assert!((world.origin - DVec3::new(-4.0, 0.0, 0.0)).length() < 1e-12);
    let sec = add_element(&mut part, "Section 1".into(), ElementGeometry::Section { frame }, None);
    let (pg, ps) = geometry_of(&face());
    add_element(&mut part, "Plane 1".into(), pg, ps);
    let offers = plane_offers(&part, Some(DVec3::ONE), false);
    assert_eq!(offers.iter().map(|o| (o.group, o.label.as_str())).collect::<Vec<_>>(), [
        ("Sections", "Section 1"),
        ("Planes", "Plane 1"),
        ("Coordinate planes", "YZ plane through the centre"),
        ("Coordinate planes", "XZ plane through the centre"),
        ("Coordinate planes", "XY plane through the centre"),
    ]);
    assert_eq!(offers[0].plane, PlaneRef::Section { id: sec, offset: None, up: None });
    // A sketch on the section lies on it.
    let resolved = crate::build::resolve_plane_frame(&offers[0].plane, &part).unwrap();
    assert!((resolved.normal - DVec3::Z).length() < 1e-12 && (resolved.origin.z - 2.5).abs() < 1e-12);
    assert_eq!(describe_cut(Some("Cylinder 1"), 2.5, DisplayUnits::Mm), "along Cylinder 1, +2.500 mm");
    assert_eq!(describe_cut(None, -1.0, DisplayUnits::Mm), "−1.000 mm from a deleted element");
    // The section takes a plane's slot in a construction.
    let fit = fit_of(part.element(sec).unwrap(), SIZE);
    assert!(ref_plane(&fit).is_some());
}

#[test]
fn the_axes_a_radial_plane_turns_about_are_the_axis_elements_then_the_coordinate_axes() {
    let mut part = Part::default();
    let (g, s) = geometry_of(&bore());
    add_element(&mut part, "Cylinder 1".into(), g, s);
    let (pg, ps) = geometry_of(&face());
    add_element(&mut part, "Plane 1".into(), pg, ps);
    let axes = radial_axis_offers(&part);
    assert_eq!(axes.iter().map(|a| a.1.as_str()).collect::<Vec<_>>(), ["Cylinder 1", "the X axis", "the Y axis", "the Z axis"]);
    // A plane through the cylinder's axis stands on it.
    let plane = PlaneRef::Radial { axis: axes[0].0, angle_deg: 30.0, offset: None };
    let f = crate::build::resolve_plane_frame(&plane, &part).unwrap();
    assert!(f.normal.dot(DVec3::Z).abs() < 1e-12);
    assert!((f.v - DVec3::Z).length() < 1e-12);
}

#[test]
fn a_fitted_plane_is_sketched_on_a_hair_inside_its_face() {
    let FitData::Plane(p) = face() else { unreachable!() };
    let plane = fitted_plane(&PlaneFit { center: DVec3::new(0.0, 0.0, 5.0), ..p }).unwrap();
    let PlaneRef::Fitted { frame, .. } = &plane else { panic!("not a fitted plane") };
    assert!((frame.origin.z - 5.0).abs() < 1e-12);
    let cut = slice_frame(&plane, frame);
    assert!((cut.origin.z - (5.0 - crate::model::FACE_SLICE_INSET_MM)).abs() < 1e-12);
    let world = PlaneRef::world(WorldAxis::Z, 5.0);
    let wf = world.own_frame().unwrap();
    assert_eq!(slice_frame(&world, &wf), wf);
}

#[test]
fn fits_a_plane_and_a_cylinder_from_a_click_and_from_a_marked_surface() {
    let g = graph_soup(box_mesh(20.0, 20, 0.01, 13));
    let seed = seed_near(&g, [0.0, 0.0, 10.0]);
    let out = fit_from_seeds(&g, ElementKind::Plane, &[seed], &FitSettings::default()).unwrap();
    let FitData::Plane(p) = out.fit else { panic!("not a plane") };
    assert!(p.normal.z.abs() > 0.999, "{:?}", p.normal);
    assert!((p.center.z - 10.0).abs() < 0.01);
    assert!(!out.region.is_empty());
    let marked = fit_on_marked(&g, ElementKind::Plane, &out.region, &FitSettings::default()).unwrap();
    assert!(matches!(marked.fit, FitData::Plane(_)));
    assert!(fit_from_seeds(&g, ElementKind::Point, &[seed], &FitSettings::default()).is_err());

    let (soup, _) = cylinder_mesh(5.0, 30.0, 48, 12, 0.01, 2);
    let g = graph_soup(soup);
    let seed = seed_near(&g, [5.0, 0.0, 0.0]);
    let out = fit_from_seeds(&g, ElementKind::Cylinder, &[seed], &FitSettings::default()).unwrap();
    let FitData::Cylinder(c) = out.fit else { panic!("not a cylinder") };
    assert!((c.radius - 5.0).abs() < 0.05, "{}", c.radius);
    let (geometry, stats) = geometry_of(&out.fit);
    assert!(stats.as_ref().is_some_and(|s| s.used_points > 0));
    assert!(format_primary(&fit_of(&Element { id: 1, name: "c".into(), geometry, stats }, SIZE), DisplayUnits::Mm).starts_with("Ø 10.0"));
}

#[test]
fn picks_a_point_and_a_circle_through_three_or_more() {
    assert_eq!(picked_point(DVec3::ONE).kind(), ElementKind::Point);
    let FitData::Circle(c) = circle_through(&[DVec3::new(5.0, 0.0, 1.0), DVec3::new(0.0, 5.0, 1.0), DVec3::new(-5.0, 0.0, 1.0)]).unwrap() else { panic!("not a circle") };
    assert!((c.radius - 5.0).abs() < 1e-9);
    assert!(circle_through(&[DVec3::ZERO, DVec3::X]).is_err());
}

#[test]
fn every_kind_opens_on_a_method_of_its_own_and_its_slots_take_their_roles() {
    for k in ELEMENT_KINDS {
        let m = creation_method(k.kind, default_method(k.kind)).unwrap();
        assert_eq!(m.kind, k.kind);
        assert!(!methods_for_kind(k.kind).is_empty());
    }
    assert_eq!(kind_info(ElementKind::Torus).noun, "round");
    let circle = creation_method(ElementKind::Circle, "circle-plane-cylinder").unwrap();
    assert!(slot_takes(&circle.slots[1], ElementKind::Cylinder));
    assert!(!slot_takes(&circle.slots[1], ElementKind::Cone));
    assert!(provides_role(ElementKind::Circle, RefRole::Point) && provides_role(ElementKind::Circle, RefRole::Axis));
}

#[test]
fn draws_every_element_with_lines_and_the_surfaces_with_triangles() {
    let mut part = Part::default();
    for fit in [bore(), ball(), face(), point([0.0; 3]), line([0.0; 3], [1.0, 0.0, 0.0])] {
        let (g, s) = geometry_of(&fit);
        add_element(&mut part, "e".into(), g, s);
    }
    for e in &part.elements {
        let wires = element_wires(&e.geometry, SIZE);
        assert!(!wires.is_empty() && wires.iter().all(|w| w.len() >= 2 && w.iter().all(|p| p.is_finite())));
        let (pts, tris) = element_surface(&e.geometry, SIZE);
        assert!(tris.iter().all(|t| t.iter().all(|i| (*i as usize) < pts.len())));
    }
    // A cylinder's wall stands at its radius.
    let (pts, tris) = element_surface(&part.elements[0].geometry, SIZE);
    assert!(!tris.is_empty());
    assert!(pts.iter().all(|p| (p.truncate().length() - 4.1585).abs() < 1e-9));
}

#[test]
fn maps_a_fit_from_the_scene_into_the_part() {
    let flip = |p: DVec3| DVec3::new(p.x, -p.z, p.y);
    let FitData::Cylinder(c) = map_fit(&bore(), flip) else { panic!("not a cylinder") };
    assert!((c.axis - DVec3::new(0.0, -1.0, 0.0)).length() < 1e-12);
    assert_eq!(c.radius, 4.1585);
}
