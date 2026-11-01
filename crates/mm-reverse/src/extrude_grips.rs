// SPDX-License-Identifier: AGPL-3.0-only
//! What an extrude or a revolve being set up wears in the viewport besides its ghost: the grips
//! that drag its numbers. An extrude wears an arrow on each face the distance sets — pull the
//! far one and the body grows — and one on the far edge that leans the sides; a revolve wears a
//! ring about its axis whose turn is the angle, the way CAD puts a manipulator on the feature
//! rather than a number in a box alone. Geometry only, in the part's coordinates; the view draws
//! it. Ported from ScanRuler (`plugins/reverse/core/preview.ts`, the extrude and revolve parts),
//! © Stefan Hermann (CNC Kitchen), AGPL-3.0-only. The grips of the features on a body (a round's,
//! a shell's, a moved face's, a Move's gizmo) are ported with their features.

use glam::{DVec2, DVec3};
use serde::{Deserialize, Serialize};

use crate::frame::SectionFrame;
use crate::model::{ExtrudeExtent, FULL_TURN_DEG};
use crate::regions::Region;

/// Which number a grip drags: an extrude's start or end, its taper, a revolve's angle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum GripSide {
    ExtrudeFrom,
    ExtrudeTo,
    Taper,
    Turn,
}

/// An arrow is dragged along its direction and reports millimetres; a ring turns about it and
/// reports degrees.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum GripShape {
    Arrow,
    Ring,
}

/// A grip a feature wears.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FeatureGrip {
    pub side: GripSide,
    pub shape: GripShape,
    pub position: DVec3,
    pub dir: DVec3,
    /// An arrow's length, or a ring's radius: millimetres, or pixels on the screen with
    /// `pixels`.
    pub size: f64,
    /// Held at its size on the screen whatever the zoom.
    pub pixels: bool,
}

/// How long a feature's arrow is on the screen, in pixels, at any zoom: enough to take hold of,
/// and on a small face or a fine edge it leaves what it stands on in view.
pub const ARROW_PX: f64 = 44.0;

/// The steepest taper the box and the grip take, degrees either way.
pub const MAX_TAPER_DEG: f64 = 85.0;

/// A feature's arrow at `position` along `dir`, held on the screen.
fn arrow_grip(side: GripSide, position: DVec3, dir: DVec3, size: f64) -> FeatureGrip {
    FeatureGrip { side, shape: GripShape::Arrow, position, dir, size, pixels: true }
}

/// The area-weighted middle of a polygon (ScanRuler's `polygonCentroid2`), or the mean of its
/// points for a sliver with no area, or the origin for none.
pub fn polygon_centroid(poly: &[DVec2]) -> DVec2 {
    let n = poly.len();
    let (mut a, mut cx, mut cy) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let p0 = poly[i];
        let p1 = poly[(i + 1) % n];
        let f = p0.x * p1.y - p1.x * p0.y;
        a += f;
        cx += (p0.x + p1.x) * f;
        cy += (p0.y + p1.y) * f;
    }
    if a.abs() < 1e-12 {
        if n == 0 {
            return DVec2::ZERO;
        }
        let m = poly.iter().fold(DVec2::ZERO, |s, p| s + *p);
        return m / n as f64;
    }
    DVec2::new(cx / (3.0 * a), cy / (3.0 * a))
}

/// A region's outer loop as the polygon its grips are read off.
pub fn outer_polygon(r: &Region) -> Vec<DVec2> {
    r.region.outer.polyline(0.01).iter().map(|v| DVec2::new(v.x, v.y)).collect()
}

/// Where the picked profiles sit, on the sheet: the area-weighted middle of their outer loops,
/// or the sheet origin with nothing picked.
pub fn regions_centroid(regions: &[&Region]) -> DVec2 {
    let (mut x, mut y, mut w) = (0.0, 0.0, 0.0);
    for r in regions {
        let c = polygon_centroid(&outer_polygon(r));
        let a = r.area.max(1e-9);
        x += c.x * a;
        y += c.y * a;
        w += a;
    }
    if w > 0.0 {
        DVec2::new(x / w, y / w)
    } else {
        DVec2::ZERO
    }
}

/// The grips of an extrude: an arrow on the start face pointing back along the normal (or the
/// way it is swept, `along`), one on the end face pointing forward, each through the profiles'
/// centroid. An extent limited by a plane has no number on that side and no grip there; a
/// symmetric one's two arrows both move its total.
pub fn extrude_grips(frame: &SectionFrame, regions: &[&Region], extent: &ExtrudeExtent, along: Option<DVec3>) -> Vec<FeatureGrip> {
    if regions.is_empty() {
        return Vec::new();
    }
    extrude_grips_at(frame.to_world(regions_centroid(regions)), along.unwrap_or(frame.normal), extent)
}

/// The grips of an extrude through a given centre along a given normal — a face pulled
/// straight has no regions, only its centroid and normal.
pub fn extrude_grips_at(centre: DVec3, normal: DVec3, extent: &ExtrudeExtent) -> Vec<FeatureGrip> {
    let n = normal;
    let back = -n;
    let arrow = |side: GripSide, offset: f64, dir: DVec3| arrow_grip(side, centre + n * offset, dir, ARROW_PX);
    match extent {
        // The lower face wears the back-pointing arrow whichever number it is, so pulling an
        // arrow outward always grows the body.
        ExtrudeExtent::Distance { from, to } => {
            if to >= from {
                vec![arrow(GripSide::ExtrudeFrom, *from, back), arrow(GripSide::ExtrudeTo, *to, n)]
            } else {
                vec![arrow(GripSide::ExtrudeFrom, *from, n), arrow(GripSide::ExtrudeTo, *to, back)]
            }
        }
        ExtrudeExtent::Symmetric { total } => vec![arrow(GripSide::ExtrudeFrom, -total / 2.0, back), arrow(GripSide::ExtrudeTo, total / 2.0, n)],
        ExtrudeExtent::ToPlane { from, .. } | ExtrudeExtent::ToFace { from, .. } | ExtrudeExtent::ToBody { from, .. } => vec![arrow(GripSide::ExtrudeFrom, *from, back)],
        // The start's arrow points away from where the body goes.
        ExtrudeExtent::ToNext { from, reverse } | ExtrudeExtent::ToLast { from, reverse } => vec![arrow(GripSide::ExtrudeFrom, *from, if *reverse { n } else { back })],
        ExtrudeExtent::BetweenPlanes { .. } => Vec::new(),
    }
}

/// Where an extrude's taper leans from and the end it leans furthest to, as offsets along the
/// normal: from the sketch plane, or the end of the extent nearest it, to the end further from
/// there. `None` for an extent with an end found on the part, which has no number to put the
/// grip at.
pub fn taper_span(extent: &ExtrudeExtent) -> Option<(f64, f64)> {
    match extent {
        ExtrudeExtent::Symmetric { total } => Some((0.0, total / 2.0)),
        ExtrudeExtent::Distance { from, to } => {
            let lo = from.min(*to);
            let hi = from.max(*to);
            let start = lo.max(0.0).min(hi);
            // The far end; of two as far, To.
            let end = if (to - start).abs() >= (from - start).abs() { *to } else { *from };
            Some((start, end))
        }
        _ => None,
    }
}

/// The taper grip of an extrude: an arrow on the far edge of the body, pointing out of it —
/// pull it out and the sides lean out, push it in and they lean in. It stands where the outline
/// reaches furthest along the sheet's X, half way along the side there, carried to the far end
/// and out by the lean, so it rides the ghost's edge. `taper_deg` is the taper as the box shows
/// it, positive leaning out. `None` with no profile, or an extent with no far end to stand on.
pub fn taper_grip(frame: &SectionFrame, regions: &[&Region], extent: &ExtrudeExtent, taper_deg: f64) -> Option<FeatureGrip> {
    let (start, end) = taper_span(extent)?;
    if regions.is_empty() {
        return None;
    }
    let length = (end - start).abs();
    if !(length > 1e-6) {
        return None;
    }
    let polys: Vec<Vec<DVec2>> = regions.iter().map(|r| outer_polygon(r)).collect();
    let max = polys.iter().flatten().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        return None;
    }
    let tol = 1e-6 * max.abs().max(1.0);
    let (mut v0, mut v1) = (f64::INFINITY, f64::NEG_INFINITY);
    for p in polys.iter().flatten() {
        if p.x < max - tol {
            continue;
        }
        v0 = v0.min(p.y);
        v1 = v1.max(p.y);
    }
    let edge = frame.to_world(DVec2::new(max, (v0 + v1) / 2.0));
    let lean = taper_deg.to_radians().tan();
    let position = edge + frame.normal * end + frame.u * (length * lean);
    Some(arrow_grip(GripSide::Taper, position, frame.u, ARROW_PX * 0.8))
}

/// The taper a drag of the taper grip makes: `delta` millimetres pulled out along the arrow
/// since the drag began, `length` the extrude's run from where the taper starts to the grip —
/// the edge moves out by the pull, and the taper is the angle that puts it there. Degrees,
/// positive leaning out, to a tenth, within [`MAX_TAPER_DEG`].
pub fn drag_taper(start_deg: f64, length: f64, delta: f64) -> f64 {
    if !(length > 1e-9) {
        return start_deg;
    }
    let out = length * start_deg.to_radians().tan() + delta;
    let deg = (out.atan2(length).to_degrees() * 10.0).round() / 10.0;
    // Back where it started from a lean, the rounding makes −0 of it: none.
    if deg == 0.0 {
        0.0
    } else {
        deg.clamp(-MAX_TAPER_DEG, MAX_TAPER_DEG)
    }
}

/// The extent an extrude grip's drag makes: `delta` millimetres pulled along the arrow since the
/// drag began, applied to the extent it began with. Outward is always growth, whichever way the
/// arrow points.
pub fn drag_extrude_extent(start: &ExtrudeExtent, side: GripSide, delta: f64) -> ExtrudeExtent {
    let mut out = start.clone();
    match &mut out {
        ExtrudeExtent::Distance { from, to } => {
            let forward = *to >= *from;
            match side {
                GripSide::ExtrudeTo => *to += if forward { delta } else { -delta },
                GripSide::ExtrudeFrom => *from -= if forward { delta } else { -delta },
                _ => {}
            }
        }
        // Either arrow pulled out by d grows the whole by 2d — the other side follows, that
        // being what symmetric means.
        ExtrudeExtent::Symmetric { total } => *total = (*total + 2.0 * delta).max(0.01),
        ExtrudeExtent::ToPlane { from, .. } | ExtrudeExtent::ToFace { from, .. } | ExtrudeExtent::ToBody { from, .. } => {
            if side == GripSide::ExtrudeFrom {
                *from -= delta;
            }
        }
        ExtrudeExtent::ToNext { from, reverse } | ExtrudeExtent::ToLast { from, reverse } => {
            if side == GripSide::ExtrudeFrom {
                *from -= if *reverse { -delta } else { delta };
            }
        }
        ExtrudeExtent::BetweenPlanes { .. } => {}
    }
    out
}

/// Which end of an extrude a point clicked on the part sets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtrudeEnd {
    From,
    To,
}

/// The extent an extrude has once one of its ends is put through a point clicked on the part:
/// `height` is the point's height over the sketch plane along its normal, negative behind it —
/// so a click behind the plane turns the extrude round, and a click for each end makes one that
/// reaches both ways. A symmetric extent reaches the point with either half. An end found on the
/// part — a plane, a face, the next one — keeps that end and takes the point for its start only.
/// `None` when the point gives the extent nothing: a body of no height, an end that is not a
/// number.
pub fn extent_through_point(extent: &ExtrudeExtent, end: ExtrudeEnd, height: f64) -> Option<ExtrudeExtent> {
    if !height.is_finite() {
        return None;
    }
    let h = crate::js_number::js_round(height * 100.0) / 100.0;
    match extent {
        ExtrudeExtent::Distance { from, to } => {
            let (from, to) = if end == ExtrudeEnd::To { (*from, h) } else { (h, *to) };
            ((to - from).abs() >= 1e-9).then_some(ExtrudeExtent::Distance { from, to })
        }
        ExtrudeExtent::Symmetric { .. } => (h.abs() >= 1e-9).then_some(ExtrudeExtent::Symmetric { total: h.abs() * 2.0 }),
        ExtrudeExtent::ToPlane { .. } | ExtrudeExtent::ToFace { .. } | ExtrudeExtent::ToBody { .. } | ExtrudeExtent::ToNext { .. } | ExtrudeExtent::ToLast { .. } => {
            if end != ExtrudeEnd::From {
                return None;
            }
            let mut out = extent.clone();
            match &mut out {
                ExtrudeExtent::ToPlane { from, .. } | ExtrudeExtent::ToFace { from, .. } | ExtrudeExtent::ToBody { from, .. } | ExtrudeExtent::ToNext { from, .. } | ExtrudeExtent::ToLast { from, .. } => *from = h,
                _ => {}
            }
            Some(out)
        }
        ExtrudeExtent::BetweenPlanes { .. } => None,
    }
}

/// A straight extrude's ghost, stretched instead of built again: the affine map that carries a
/// prism reaching from `from0` to `to0` along `normal` onto the one from `from1` to `to1` —
/// heights over the sketch plane scaled and shifted, nothing moved across it — as a
/// column-major 4 × 4. What a grip's drag shows between the kernel's answers. `None` for a prism
/// of no height, which has nothing to stretch.
pub fn prism_stretch(origin: DVec3, normal: DVec3, from0: f64, to0: f64, from1: f64, to1: f64) -> Option<[f64; 16]> {
    let span = to0 - from0;
    if span.abs() < 1e-9 {
        return None;
    }
    let s = (to1 - from1) / span;
    // p' = p + n · ((s − 1)·h + from1 − s·from0), with h = (p − origin) · n.
    let shift = from1 - s * from0 - (s - 1.0) * origin.dot(normal);
    let mut m = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    let n = normal.to_array();
    for r in 0..3 {
        for c in 0..3 {
            m[c * 4 + r] += (s - 1.0) * n[r] * n[c];
        }
        m[12 + r] = shift * n[r];
    }
    Some(m)
}

/// A column-major 4 × 4 on a point.
pub fn apply_column_major(m: &[f64; 16], p: DVec3) -> DVec3 {
    DVec3::new(m[0] * p.x + m[4] * p.y + m[8] * p.z + m[12], m[1] * p.x + m[5] * p.y + m[9] * p.z + m[13], m[2] * p.x + m[6] * p.y + m[10] * p.z + m[14])
}

/// The grip of a revolve: a ring about the axis, through the plane of the profiles' centroid,
/// as wide as the centroid's distance from the axis so the ring runs through the body being
/// turned. `None` when the centroid lies on the axis — a profile revolving about its own middle
/// is not a solid.
pub fn revolve_grip(frame: &SectionFrame, regions: &[&Region], axis_origin: DVec3, axis_dir: DVec3) -> Option<FeatureGrip> {
    if regions.is_empty() {
        return None;
    }
    let centre = frame.to_world(regions_centroid(regions));
    let dir = axis_dir.try_normalize()?;
    let along = (centre - axis_origin).dot(dir);
    let foot = axis_origin + dir * along;
    let radius = (centre - foot).length();
    if radius < 1e-6 {
        return None;
    }
    Some(FeatureGrip { side: GripSide::Turn, shape: GripShape::Ring, position: foot, dir, size: radius, pixels: false })
}

/// The angle a revolve grip's drag makes: degrees turned about the axis since the drag began,
/// applied to the angle it began with, kept positive and within `max_deg` — a full turn for a
/// plain revolve, as many as a coil takes for one with a pitch, the ring wound round and round.
pub fn drag_revolve_angle(start_deg: f64, delta_deg: f64, max_deg: f64) -> f64 {
    let a = start_deg + delta_deg;
    max_deg.min((crate::js_number::js_round(a * 10.0) / 10.0).max(1.0))
}

/// [`drag_revolve_angle`] for a plain revolve, within a full turn.
pub fn drag_revolve_angle_plain(start_deg: f64, delta_deg: f64) -> f64 {
    drag_revolve_angle(start_deg, delta_deg, FULL_TURN_DEG)
}

#[cfg(test)]
mod tests {
    // ScanRuler's `rePreview.test.ts`.
    use super::*;
    use crate::model::{MAX_COIL_DEG, PlaneRef};
    use crate::regions::find_regions;
    use crate::sketch::Sketch;

    fn xy() -> SectionFrame {
        SectionFrame { origin: DVec3::ZERO, normal: DVec3::Z, u: DVec3::X, v: DVec3::Y }
    }

    /// A frame standing at z = 5 with its normal down the part's Y axis.
    fn tilted() -> SectionFrame {
        SectionFrame { origin: DVec3::new(1.0, 2.0, 5.0), normal: DVec3::Y, u: DVec3::X, v: DVec3::new(0.0, 0.0, -1.0) }
    }

    fn rectangle_regions(w: f64, h: f64, at: [f64; 2]) -> Vec<Region> {
        let mut s = Sketch::default();
        s.add_rectangle(DVec2::new(at[0], at[1]), DVec2::new(at[0] + w, at[1] + h));
        find_regions(&s)
    }

    fn refs(r: &[Region]) -> Vec<&Region> {
        r.iter().collect()
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    fn plane(id: u32) -> PlaneRef {
        PlaneRef::Plane { id, offset: None, up: None }
    }

    #[test]
    fn the_centroid_is_the_middle_of_a_rectangle_and_the_area_weighted_middle_of_several() {
        assert_eq!(polygon_centroid(&[DVec2::new(0.0, 0.0), DVec2::new(4.0, 0.0), DVec2::new(4.0, 2.0), DVec2::new(0.0, 2.0)]), DVec2::new(2.0, 1.0));
        let one = rectangle_regions(4.0, 2.0, [0.0, 0.0]);
        let c = regions_centroid(&refs(&one));
        assert!(close(c.x, 2.0) && close(c.y, 1.0));
        // A 4×2 at the origin and a 2×2 at x = 10..12: areas 8 and 4.
        let mut both = one.clone();
        both.extend(rectangle_regions(2.0, 2.0, [10.0, 0.0]));
        assert!(close(regions_centroid(&refs(&both)).x, (2.0 * 8.0 + 11.0 * 4.0) / 12.0));
    }

    #[test]
    fn the_centroid_falls_back_to_the_vertex_mean_for_a_sliver_with_no_area() {
        assert_eq!(polygon_centroid(&[DVec2::new(0.0, 0.0), DVec2::new(2.0, 0.0), DVec2::new(4.0, 0.0)]), DVec2::new(2.0, 0.0));
        assert_eq!(polygon_centroid(&[]), DVec2::ZERO);
    }

    #[test]
    fn puts_an_arrow_on_each_face_of_a_distance_through_the_centroid_pointing_outward() {
        let regions = rectangle_regions(40.0, 20.0, [0.0, 0.0]);
        let grips = extrude_grips(&xy(), &refs(&regions), &ExtrudeExtent::Distance { from: -2.0, to: 10.0 }, None);
        assert_eq!(grips.iter().map(|g| g.side).collect::<Vec<_>>(), vec![GripSide::ExtrudeFrom, GripSide::ExtrudeTo]);
        let (from, to) = (grips[0], grips[1]);
        assert!((from.position - DVec3::new(20.0, 10.0, -2.0)).length() < 1e-9);
        assert_eq!(from.dir, DVec3::new(0.0, 0.0, -1.0));
        assert!((to.position - DVec3::new(20.0, 10.0, 10.0)).length() < 1e-9);
        assert_eq!(to.dir, DVec3::Z);
        assert_eq!(from.shape, GripShape::Arrow);
        // Held at a size on the screen, not on the part.
        assert!(from.pixels);
        assert_eq!(from.size, ARROW_PX);
        let small = rectangle_regions(0.5, 0.5, [0.0, 0.0]);
        assert_eq!(extrude_grips(&xy(), &refs(&small), &ExtrudeExtent::Distance { from: 0.0, to: 1.0 }, None)[1].size, ARROW_PX);
    }

    #[test]
    fn the_arrows_follow_the_frame_along_its_normal_from_its_origin() {
        let regions = rectangle_regions(40.0, 20.0, [0.0, 0.0]);
        let g = extrude_grips(&tilted(), &refs(&regions), &ExtrudeExtent::Distance { from: 0.0, to: 6.0 }, None);
        // Sheet (20, 10) lifts to origin + 20·U + 10·V = (21, 2, −5).
        assert!((g[0].position - DVec3::new(21.0, 2.0, -5.0)).length() < 1e-9);
        assert!((g[1].position - DVec3::new(21.0, 8.0, -5.0)).length() < 1e-9);
        assert_eq!(g[1].dir, DVec3::Y);
    }

    #[test]
    fn turns_the_arrows_round_when_the_distance_runs_backwards() {
        let regions = rectangle_regions(40.0, 20.0, [0.0, 0.0]);
        let g = extrude_grips(&xy(), &refs(&regions), &ExtrudeExtent::Distance { from: 10.0, to: -2.0 }, None);
        assert!(close(g[0].position.z, 10.0) && close(g[0].dir.z, 1.0));
        assert!(close(g[1].position.z, -2.0) && close(g[1].dir.z, -1.0));
    }

    #[test]
    fn gives_a_symmetric_extent_two_arrows_a_to_plane_one_between_planes_none() {
        let regions = rectangle_regions(40.0, 20.0, [0.0, 0.0]);
        let r = refs(&regions);
        assert_eq!(extrude_grips(&xy(), &r, &ExtrudeExtent::Symmetric { total: 8.0 }, None).iter().map(|g| g.position.z).collect::<Vec<_>>(), vec![-4.0, 4.0]);
        assert_eq!(extrude_grips(&xy(), &r, &ExtrudeExtent::ToPlane { from: 1.0, plane: plane(1) }, None).iter().map(|g| g.side).collect::<Vec<_>>(), vec![GripSide::ExtrudeFrom]);
        assert!(extrude_grips(&xy(), &r, &ExtrudeExtent::BetweenPlanes { a: plane(1), b: plane(2) }, None).is_empty());
        assert!(extrude_grips(&xy(), &[], &ExtrudeExtent::Distance { from: 0.0, to: 1.0 }, None).is_empty());
    }

    #[test]
    fn dragging_pulls_the_end_out_and_the_start_back_growing_the_body_either_way() {
        let start = ExtrudeExtent::Distance { from: 0.0, to: 10.0 };
        assert_eq!(drag_extrude_extent(&start, GripSide::ExtrudeTo, 5.0), ExtrudeExtent::Distance { from: 0.0, to: 15.0 });
        assert_eq!(drag_extrude_extent(&start, GripSide::ExtrudeFrom, 5.0), ExtrudeExtent::Distance { from: -5.0, to: 10.0 });
        // Pushed in past the other face, the numbers simply cross.
        assert_eq!(drag_extrude_extent(&start, GripSide::ExtrudeTo, -12.0), ExtrudeExtent::Distance { from: 0.0, to: -2.0 });
    }

    #[test]
    fn dragging_reads_a_backwards_distance_the_same_way() {
        let start = ExtrudeExtent::Distance { from: 10.0, to: -2.0 };
        assert_eq!(drag_extrude_extent(&start, GripSide::ExtrudeTo, 3.0), ExtrudeExtent::Distance { from: 10.0, to: -5.0 });
        assert_eq!(drag_extrude_extent(&start, GripSide::ExtrudeFrom, 3.0), ExtrudeExtent::Distance { from: 13.0, to: -2.0 });
    }

    #[test]
    fn dragging_grows_a_symmetric_extent_by_twice_the_pull_and_keeps_it_positive() {
        assert_eq!(drag_extrude_extent(&ExtrudeExtent::Symmetric { total: 10.0 }, GripSide::ExtrudeTo, 2.0), ExtrudeExtent::Symmetric { total: 14.0 });
        assert_eq!(drag_extrude_extent(&ExtrudeExtent::Symmetric { total: 10.0 }, GripSide::ExtrudeFrom, 2.0), ExtrudeExtent::Symmetric { total: 14.0 });
        match drag_extrude_extent(&ExtrudeExtent::Symmetric { total: 10.0 }, GripSide::ExtrudeTo, -8.0) {
            ExtrudeExtent::Symmetric { total } => assert!(close(total, 0.01)),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn dragging_moves_only_the_start_of_a_to_plane_extent_and_nothing_of_one_between_planes() {
        let to_plane = ExtrudeExtent::ToPlane { from: 1.0, plane: plane(1) };
        assert_eq!(drag_extrude_extent(&to_plane, GripSide::ExtrudeFrom, 2.0), ExtrudeExtent::ToPlane { from: -1.0, plane: plane(1) });
        assert_eq!(drag_extrude_extent(&to_plane, GripSide::ExtrudeTo, 2.0), to_plane);
        let between = ExtrudeExtent::BetweenPlanes { a: plane(1), b: plane(2) };
        assert_eq!(drag_extrude_extent(&between, GripSide::ExtrudeFrom, 2.0), between);
    }

    #[test]
    fn an_extent_through_a_point_puts_the_end_asked_for_at_the_points_height() {
        let start = ExtrudeExtent::Distance { from: 0.0, to: 10.0 };
        assert_eq!(extent_through_point(&start, ExtrudeEnd::To, 24.638), Some(ExtrudeExtent::Distance { from: 0.0, to: 24.64 }));
        // Behind the sketch plane: the extrude turns round.
        assert_eq!(extent_through_point(&start, ExtrudeEnd::To, -8.0), Some(ExtrudeExtent::Distance { from: 0.0, to: -8.0 }));
        // One click for each end: a body that reaches both ways.
        let both = extent_through_point(&extent_through_point(&start, ExtrudeEnd::To, 12.0).unwrap_or(start.clone()), ExtrudeEnd::From, -7.5);
        assert_eq!(both, Some(ExtrudeExtent::Distance { from: -7.5, to: 12.0 }));
    }

    #[test]
    fn an_extent_through_a_point_refuses_a_body_of_no_height_and_an_end_that_is_not_a_number() {
        let start = ExtrudeExtent::Distance { from: 0.0, to: 10.0 };
        assert_eq!(extent_through_point(&start, ExtrudeEnd::To, 0.001), None);
        assert_eq!(extent_through_point(&start, ExtrudeEnd::To, f64::NAN), None);
        assert_eq!(extent_through_point(&ExtrudeExtent::ToNext { from: 0.0, reverse: false }, ExtrudeEnd::To, 5.0), None);
        assert_eq!(extent_through_point(&ExtrudeExtent::ToNext { from: 0.0, reverse: false }, ExtrudeEnd::From, 5.0), Some(ExtrudeExtent::ToNext { from: 5.0, reverse: false }));
    }

    #[test]
    fn an_extent_through_a_point_reaches_it_with_either_half_of_a_symmetric_one() {
        assert_eq!(extent_through_point(&ExtrudeExtent::Symmetric { total: 10.0 }, ExtrudeEnd::To, -6.0), Some(ExtrudeExtent::Symmetric { total: 12.0 }));
        assert_eq!(extent_through_point(&ExtrudeExtent::Symmetric { total: 10.0 }, ExtrudeEnd::To, 0.0), None);
    }

    #[test]
    fn a_stretch_carries_the_heights_from_one_from_and_to_onto_another_and_nothing_across_the_plane() {
        let t = tilted();
        let m = prism_stretch(t.origin, t.normal, 0.0, 10.0, -5.0, 25.0).expect("a stretch");
        let foot = apply_column_major(&m, DVec3::new(7.0, 2.0, -3.0));
        let top = apply_column_major(&m, DVec3::new(7.0, 12.0, -3.0));
        let mid = apply_column_major(&m, DVec3::new(-1.0, 7.0, 4.0));
        assert!(close(foot.x, 7.0) && close(foot.y, 2.0 - 5.0) && close(foot.z, -3.0));
        assert!(close(top.y, 2.0 + 25.0));
        assert!(close(mid.x, -1.0) && close(mid.y, 2.0 + 10.0) && close(mid.z, 4.0));
    }

    #[test]
    fn a_stretch_turns_the_prism_over_and_has_nothing_to_stretch_of_none() {
        let m = prism_stretch(DVec3::ZERO, DVec3::Z, 0.0, 10.0, 0.0, -4.0).expect("a stretch");
        assert!(close(apply_column_major(&m, DVec3::new(1.0, 1.0, 10.0)).z, -4.0));
        assert!(close(apply_column_major(&m, DVec3::new(1.0, 1.0, 0.0)).z, 0.0));
        assert!(prism_stretch(DVec3::ZERO, DVec3::Z, 3.0, 3.0, 0.0, 5.0).is_none());
    }

    #[test]
    fn the_revolve_grip_rings_the_axis_through_the_profile_at_the_centroids_distance() {
        // A 5 × 10 rectangle beside the sheet's Y axis, revolved about that axis.
        let regions = rectangle_regions(5.0, 10.0, [0.0, 0.0]);
        let ring = revolve_grip(&xy(), &refs(&regions), DVec3::ZERO, DVec3::Y).expect("a ring");
        assert_eq!(ring.shape, GripShape::Ring);
        assert_eq!(ring.side, GripSide::Turn);
        assert!((ring.position - DVec3::new(0.0, 5.0, 0.0)).length() < 1e-9);
        assert!(close(ring.size, 2.5));
        assert_eq!(ring.dir, DVec3::Y);
    }

    #[test]
    fn the_revolve_grip_is_none_with_nothing_picked_a_bad_axis_or_the_centroid_on_the_axis() {
        let regions = rectangle_regions(4.0, 4.0, [-2.0, -2.0]);
        assert!(revolve_grip(&xy(), &[], DVec3::ZERO, DVec3::Y).is_none());
        assert!(revolve_grip(&xy(), &refs(&regions), DVec3::ZERO, DVec3::ZERO).is_none());
        assert!(revolve_grip(&xy(), &refs(&regions), DVec3::ZERO, DVec3::Y).is_none());
    }

    #[test]
    fn the_ring_turns_the_angle_by_the_degrees_dragged_within_a_full_turn() {
        assert_eq!(drag_revolve_angle_plain(90.0, 45.0), 135.0);
        assert_eq!(drag_revolve_angle_plain(300.0, 90.0), 360.0);
        assert_eq!(drag_revolve_angle_plain(10.0, -30.0), 1.0);
        assert_eq!(drag_revolve_angle_plain(90.0, 0.04), 90.0);
    }

    #[test]
    fn the_ring_winds_a_coil_on_past_a_full_turn() {
        assert_eq!(drag_revolve_angle(1080.0, 400.0, MAX_COIL_DEG), 1480.0);
        assert_eq!(drag_revolve_angle(300.0, 90.0, MAX_COIL_DEG), 390.0);
        assert_eq!(drag_revolve_angle(10.0, -30.0, MAX_COIL_DEG), 1.0);
    }

    #[test]
    fn the_taper_leans_from_the_sketch_plane_or_the_end_nearest_it_to_the_end_further_away() {
        let d = |from: f64, to: f64| ExtrudeExtent::Distance { from, to };
        assert_eq!(taper_span(&d(0.0, 10.0)), Some((0.0, 10.0)));
        assert_eq!(taper_span(&d(-4.0, 10.0)), Some((0.0, 10.0)));
        assert_eq!(taper_span(&d(-12.0, 10.0)), Some((0.0, -12.0)));
        assert_eq!(taper_span(&d(5.0, 15.0)), Some((5.0, 15.0)));
        assert_eq!(taper_span(&d(0.0, -8.0)), Some((0.0, -8.0)));
        assert_eq!(taper_span(&ExtrudeExtent::Symmetric { total: 20.0 }), Some((0.0, 10.0)));
        assert_eq!(taper_span(&ExtrudeExtent::ToNext { from: 0.0, reverse: false }), None);
    }

    #[test]
    fn the_taper_grip_rides_the_far_edge_half_way_along_the_furthest_side_out_by_the_lean() {
        let regions = rectangle_regions(40.0, 20.0, [0.0, 0.0]);
        let g = taper_grip(&xy(), &refs(&regions), &ExtrudeExtent::Distance { from: 0.0, to: 10.0 }, 10.0).expect("a grip");
        assert_eq!(g.side, GripSide::Taper);
        assert_eq!(g.shape, GripShape::Arrow);
        assert_eq!(g.dir, DVec3::X);
        assert!(close(g.position.x, 40.0 + 10.0 * 10f64.to_radians().tan()));
        assert!(close(g.position.y, 10.0));
        assert!(close(g.position.z, 10.0));
        // Leaning in, it stands in from the side; an extrude reaching down, on the far end
        // below the plane.
        let small = rectangle_regions(4.0, 2.0, [0.0, 0.0]);
        let down = taper_grip(&tilted(), &refs(&small), &ExtrudeExtent::Distance { from: 0.0, to: -6.0 }, -20.0).expect("a grip");
        assert!(close(down.position.x, 1.0 + 4.0 - 6.0 * 20f64.to_radians().tan()));
        assert!(close(down.position.y, 2.0 - 6.0));
        assert!(close(down.position.z, 5.0 - 1.0));
        assert!(taper_grip(&xy(), &[], &ExtrudeExtent::Distance { from: 0.0, to: 10.0 }, 0.0).is_none());
        assert!(taper_grip(&xy(), &refs(&small), &ExtrudeExtent::ToLast { from: 0.0, reverse: false }, 0.0).is_none());
    }

    #[test]
    fn a_pull_on_the_taper_grip_is_the_angle_that_puts_the_edge_there() {
        assert_eq!(drag_taper(0.0, 10.0, 10.0), 45.0);
        assert_eq!(drag_taper(45.0, 10.0, -10.0), 0.0);
        assert_eq!(drag_taper(0.0, 10.0, -10.0), -45.0);
        assert_eq!(drag_taper(0.0, 10.0, 1e9), MAX_TAPER_DEG);
        assert_eq!(drag_taper(12.0, 0.0, 5.0), 12.0);
    }
}
