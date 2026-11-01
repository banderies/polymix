// SPDX-License-Identifier: AGPL-3.0-only
//! An Extrude or a Revolve as its box holds it before it is made: when it is ready to make, what
//! a click on the part does to its extent, the result a pulled face follows, a coil's starting
//! pitch and turns, the angle the box takes, and the regions a feature takes as they were asked
//! for — every one, the largest, by key or by a point inside. Ported from ScanRuler
//! (`plugins/reverse/state/reDrafts.ts` extrudeDraftReady, extrudePickEnd, revolveDraftReady,
//! revolveAngle, startingPitch; `state/reSketchDraft.ts` faceExtrudeOp; `state/reSolidSlice.ts`
//! extrudeToPoint, setRevolveCoil; `state/reFeatureHelpers.ts` autoRegions, everyRegion, coilOf;
//! `app/reCommandCore.ts` regionKeys, regionPickOf), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only.
//!
//! ScanRuler's draft holds an extent's plane, face or body as nothing (`null`, a body of −1)
//! until one is chosen; the feature here is the draft, so "nothing chosen" is a reference that
//! names nothing — [`no_plane`], [`NO_BODY`], [`no_axis`] — which a box is not ready with and the
//! build refuses by name.

use glam::{DVec2, DVec3};

use crate::display_units::{fmt_len, DisplayUnits, FmtOptions};
use crate::extrude_grips::{extent_through_point, ExtrudeEnd};
use crate::frame::SectionFrame;
use crate::model::{extent_ends_on_part, revolve_max_deg, AxisRef, BodyId, BodyOp, ElementId, ExtrudeExtent, ExtrudeFeature, PlaneRef, RegionAsk, RegionGiven, RegionPick, RegionWord, RevolveFeature, FULL_TURN_DEG};
use crate::nominal::{nominal_step, NominalKind};
use crate::profile_axis::{lift_profiles, profile_span};
use crate::regions::{region_at, Region};

/// An element id no element has: the plane or axis a box has not been given yet.
pub const NO_ELEMENT: ElementId = ElementId::MAX;

/// A body id no body has (feature ids start at 1): the body or face a box has not been given.
pub const NO_BODY: BodyId = BodyId { feature: 0, island: 0 };

/// The plane an extent has before one is chosen.
pub fn no_plane() -> PlaneRef {
    PlaneRef::Plane { id: NO_ELEMENT, offset: None, up: None }
}

/// Whether a plane has been chosen.
pub fn plane_chosen(p: &PlaneRef) -> bool {
    !matches!(p, PlaneRef::Plane { id, .. } if *id == NO_ELEMENT)
}

/// The axis a revolve has before one is chosen.
pub fn no_axis() -> AxisRef {
    AxisRef::Element { id: NO_ELEMENT }
}

/// Whether a revolve's axis has been chosen.
pub fn axis_chosen(a: &AxisRef) -> bool {
    !matches!(a, AxisRef::Element { id } if *id == NO_ELEMENT)
}

/// What an extent still lacks: a plane, a face or a body to reach to — `None` when it names
/// everything it needs.
pub fn extent_lacks(e: &ExtrudeExtent) -> Option<&'static str> {
    match e {
        ExtrudeExtent::ToPlane { plane, .. } if !plane_chosen(plane) => Some("Choose the plane the extrude reaches to."),
        ExtrudeExtent::ToFace { body, .. } if *body == NO_BODY => Some("Click the face of a body the extrude reaches to."),
        ExtrudeExtent::ToBody { body, .. } if *body == NO_BODY => Some("Choose the body the extrude reaches to."),
        ExtrudeExtent::BetweenPlanes { a, b } if !plane_chosen(a) || !plane_chosen(b) => Some("Choose both planes the extrude runs between."),
        _ => None,
    }
}

/// Whether an extrude's box can make it: a sketch's regions or a pulled face, and an extent with
/// everything it names (ScanRuler's `extrudeDraftReady`). The sentence the agent is answered with
/// when it is not.
pub fn extrude_ready(e: &ExtrudeFeature) -> Result<(), String> {
    let profile = e.face.is_some() || (e.sketch_id != 0 && !e.regions.is_empty());
    if profile && extent_lacks(&e.extent).is_none() {
        Ok(())
    } else {
        Err("The extrude needs a profile — a sketch’s regions or a face — and an extent with everything it names.".into())
    }
}

/// Whether a revolve's box can make it: a profile, an axis, and an angle within what it may turn
/// (ScanRuler's `revolveDraftReady`), or why not.
pub fn revolve_ready(r: &RevolveFeature) -> Result<(), String> {
    let pitch = r.pitch.unwrap_or(0.0);
    if r.sketch_id != 0 && !r.regions.is_empty() && axis_chosen(&r.axis) && r.angle_deg > 0.0 && r.angle_deg <= revolve_max_deg(Some(pitch)) {
        Ok(())
    } else if r.angle_deg > revolve_max_deg(Some(pitch)) {
        Err(format!("A revolve turns {FULL_TURN_DEG}° at most — give it a pitch to coil past a whole turn."))
    } else {
        Err("The revolve needs a profile, an axis and an angle.".into())
    }
}

/// The angle a revolve's box takes: positive, at most a full turn without a pitch and a thousand
/// with one, to a tenth of a degree.
pub fn revolve_angle(deg: f64, pitch: f64) -> f64 {
    revolve_max_deg(Some(pitch)).min((crate::js_number::js_round(deg.abs() * 10.0) / 10.0).max(0.1))
}

/// A coil's pitch to start with: half again as far as the picked profiles reach along the axis,
/// rounded up — the turns clear of each other with room between — or 5 mm while there is no axis
/// or profile to read.
pub fn starting_pitch(frame: &SectionFrame, regions: &[&Region], axis: Option<(DVec3, DVec3)>) -> f64 {
    const FALLBACK: f64 = 5.0;
    let Some((origin, dir)) = axis else { return FALLBACK };
    let Some(dir) = dir.try_normalize() else { return FALLBACK };
    if regions.is_empty() {
        return FALLBACK;
    }
    let profiles = lift_profiles(frame, &regions.iter().map(|r| r.region.clone()).collect::<Vec<_>>());
    let (min, max) = profile_span(&profiles, origin, dir);
    let reach = max - min;
    if !(reach > 1e-6) {
        return FALLBACK;
    }
    let step = nominal_step(reach * 1.5, NominalKind::Length);
    ((reach * 1.5) / step - 1e-9).ceil() * step
}

/// A revolve made a coil, or plain again (ScanRuler's `setRevolveCoil`): ticked, it takes the
/// starting pitch and three turns where it turned a whole one; let go, its pitch goes and its
/// angle stops at a full turn.
pub fn set_coil(r: &mut RevolveFeature, on: bool, pitch: impl FnOnce() -> f64) {
    let coil = r.pitch.is_some_and(|p| p != 0.0 && p.is_finite());
    if on == coil {
        return;
    }
    if !on {
        r.pitch = None;
        r.left_handed = false;
        r.angle_deg = r.angle_deg.min(FULL_TURN_DEG);
        return;
    }
    r.pitch = Some(pitch());
    if r.angle_deg >= FULL_TURN_DEG {
        r.angle_deg = 3.0 * FULL_TURN_DEG;
    }
}

/// The result an extrude pulled from a face has while it still follows the direction: outward
/// joins the body, inward cuts it.
pub fn face_extrude_op(extent: &ExtrudeExtent) -> BodyOp {
    let outward = match extent {
        ExtrudeExtent::Distance { from, to } => to + from >= 0.0,
        _ => true,
    };
    if outward {
        BodyOp::Join
    } else {
        BodyOp::Cut
    }
}

/// The end of the open extrude a click on the part sets: To for a point in front of the sketch
/// plane, From for one behind it — the body runs from the lowest point clicked to the highest, in
/// whichever order they are clicked. An extent that ends on the part has only its start to give.
pub fn extrude_pick_end(extent: &ExtrudeExtent, height: f64) -> ExtrudeEnd {
    if extent_ends_on_part(extent) || height < 0.0 {
        ExtrudeEnd::From
    } else {
        ExtrudeEnd::To
    }
}

/// An extrude's end put through a point clicked on the part (ScanRuler's `extrudeToPoint`): the
/// plane the distance is measured from is `origin`/`normal` — the sketch's, or a pulled face's;
/// swept askew along `direction`, the offset counts along the sweep. The extent it comes to and
/// what the status line says, or why the click gave nothing.
pub fn extrude_to_point(extent: &ExtrudeExtent, origin: DVec3, normal: DVec3, direction: Option<DVec3>, point: DVec3) -> Result<(ExtrudeExtent, String), String> {
    let steep = direction.map_or(1.0, |d| {
        let c = d.dot(normal);
        if c == 0.0 {
            1.0
        } else {
            c
        }
    });
    let height = (point - origin).dot(normal) / steep;
    let end = extrude_pick_end(extent, height);
    let Some(next) = extent_through_point(extent, end, height) else {
        if matches!(extent, ExtrudeExtent::BetweenPlanes { .. }) || (extent_ends_on_part(extent) && end == ExtrudeEnd::To) {
            return Err("This extent ends on the part already — set Extent to Distance to reach a clicked point.".into());
        }
        return Err("That point lies on the other end — the body would have no height.".into());
    };
    let h = crate::js_number::js_round(height * 100.0) / 100.0;
    let said = match &next {
        ExtrudeExtent::Symmetric { total } => format!("Total {} — each half reaches the point clicked.", fmt_len(*total, DisplayUnits::Mm, 2, FmtOptions::default())),
        _ => format!("{} = {}, the height of the point clicked{}.", if end == ExtrudeEnd::To { "To" } else { "From" }, fmt_len(h, DisplayUnits::Mm, 2, FmtOptions::default()), if h < 0.0 { " — behind the sketch plane" } else { "" }),
    };
    Ok((next, said))
}

/// The regions an Extrude or a Revolve opens with when nothing chose them: the sketch's one
/// region when it has exactly one, else none — the operator clicks those wanted.
pub fn auto_regions(regions: &[Region]) -> Vec<String> {
    if regions.len() == 1 {
        vec![regions[0].key.clone()]
    } else {
        Vec::new()
    }
}

/// Every closed region of a sketch — what the sketch's own Extrude and Revolve keys open with:
/// the sketch just drawn is the profile, all of it, and a click on a region lets that one go.
pub fn every_region(regions: &[Region]) -> Vec<String> {
    regions.iter().map(|r| r.key.clone()).collect()
}

/// The regions of a sketch a feature is given as they were asked for (ScanRuler's `regionKeys`):
/// none asked takes the one region of a sketch that has one; "all", "largest", keys and points
/// inside regions otherwise — each refused, naming the regions there are, where it names none.
pub fn region_keys(sketch_name: &str, regions: &[Region], asked: Option<&RegionGiven>) -> Result<Vec<String>, String> {
    let listed = || regions.iter().map(|r| format!("{} ({} mm², inside at [{:.3}, {:.3}], {} hole{})", r.key, crate::js_fmt::to_fixed(r.area, 3), r.region.interior_point().x, r.region.interior_point().y, r.holes.len(), if r.holes.len() == 1 { "" } else { "s" })).collect::<Vec<_>>().join("; ");
    if regions.is_empty() {
        return Err(format!("{sketch_name} has no closed region — its pieces have to close a loop, their ends coincident."));
    }
    let Some(asked) = asked else {
        if regions.len() == 1 {
            return Ok(vec![regions[0].key.clone()]);
        }
        return Err(format!("{sketch_name} has {} closed regions — say which in regions (\"all\", \"largest\", their keys, or a point inside each): {}", regions.len(), listed()));
    };
    match asked {
        RegionGiven::Word(RegionWord::All) => Ok(regions.iter().map(|r| r.key.clone()).collect()),
        RegionGiven::Word(RegionWord::Largest) => Ok(regions.iter().fold(&regions[0], |a, b| if b.area > a.area { b } else { a }).key.clone()).map(|k| vec![k]),
        RegionGiven::List(list) => list
            .iter()
            .map(|p| match p {
                RegionAsk::Key(k) => {
                    if regions.iter().any(|r| &r.key == k) {
                        Ok(k.clone())
                    } else {
                        Err(format!("{sketch_name} has no region {k} — it has {}", listed()))
                    }
                }
                RegionAsk::At { at } => region_at(regions, *at).map(|r| r.key.clone()).ok_or_else(|| format!("No region of {sketch_name} holds the point [{}, {}] — it has {}", crate::js_number::js_string(at.x), crate::js_number::js_string(at.y), listed())),
            })
            .collect(),
    }
}

/// A region pick as the feature keeps it, to be taken again as given when its sketch is edited:
/// "all", "largest", or a list with a point in it, with the keys it came to. `None` for keys
/// alone, or nothing given: the keys are the pick, matched again by their pieces and by place.
pub fn region_pick_of(asked: Option<&RegionGiven>, keys: &[String]) -> Option<RegionPick> {
    match asked? {
        RegionGiven::Word(w) => Some(RegionPick { given: RegionGiven::Word(*w), keys: keys.to_vec() }),
        RegionGiven::List(list) if list.iter().all(|p| matches!(p, RegionAsk::Key(_))) => None,
        RegionGiven::List(list) => Some(RegionPick { given: RegionGiven::List(list.clone()), keys: keys.to_vec() }),
    }
}

/// A region pick read from words as a script writes it: `all`, `largest`, or keys and points
/// inside regions written `@u,v` (a key holds commas of its own), separated by `;`.
pub fn region_given_of(text: &str) -> Option<RegionGiven> {
    let t = text.trim();
    match t {
        "" => None,
        "all" => Some(RegionGiven::Word(RegionWord::All)),
        "largest" => Some(RegionGiven::Word(RegionWord::Largest)),
        _ => Some(RegionGiven::List(
            t.split(';')
                .filter(|p| !p.trim().is_empty())
                .map(|p| {
                    let p = p.trim();
                    let point = p.strip_prefix('@').and_then(|uv| {
                        let (u, v) = uv.split_once(',')?;
                        Some(DVec2::new(u.trim().parse().ok()?, v.trim().parse().ok()?))
                    });
                    match point {
                        Some(at) => RegionAsk::At { at },
                        None => RegionAsk::Key(p.to_string()),
                    }
                })
                .collect(),
        )),
    }
}

#[cfg(test)]
mod tests {
    // ScanRuler's `reStore.test.ts` (an extrude's end through a point on the part; a coil's
    // start) and `reFeatureDraft.test.ts` (a face pulled as the profile), the parts that are no
    // store's.
    use super::*;
    use crate::regions::find_regions;
    use crate::sketch::{Shape, Sketch};

    fn square_regions() -> Vec<Region> {
        let mut s = Sketch::default();
        s.add_rectangle(DVec2::new(0.0, 0.0), DVec2::new(10.0, 10.0));
        find_regions(&s)
    }

    #[test]
    fn puts_to_at_a_point_in_front_of_the_sketch_plane_and_from_at_one_behind_it_in_either_order() {
        let at = |e: &ExtrudeExtent, p: [f64; 3]| extrude_to_point(e, DVec3::ZERO, DVec3::Z, None, DVec3::from(p));
        let e = ExtrudeExtent::Distance { from: 0.0, to: 10.0 };
        let (e, said) = at(&e, [3.0, 3.0, 17.456]).expect("To");
        assert!(said.starts_with("To = 17.46"), "{said}");
        assert_eq!(e, ExtrudeExtent::Distance { from: 0.0, to: 17.46 });
        let (e, said) = at(&e, [50.0, -20.0, -6.0]).expect("From");
        assert!(said.starts_with("From = -6.00") && said.contains("behind the sketch plane"), "{said}");
        assert_eq!(e, ExtrudeExtent::Distance { from: -6.0, to: 17.46 });
        // The bottom first, then the top: the same body.
        let e = ExtrudeExtent::Distance { from: 0.0, to: 10.0 };
        let (e, said) = at(&e, [0.0, 0.0, -6.0]).expect("From");
        assert!(said.starts_with("From = -6.00"), "{said}");
        let (e, said) = at(&e, [0.0, 0.0, 17.456]).expect("To");
        assert!(said.starts_with("To = 17.46"), "{said}");
        assert_eq!(e, ExtrudeExtent::Distance { from: -6.0, to: 17.46 });
    }

    #[test]
    fn says_why_a_click_gave_nothing_and_sets_the_start_of_an_extent_that_ends_on_the_part() {
        let at = |e: &ExtrudeExtent, p: [f64; 3]| extrude_to_point(e, DVec3::ZERO, DVec3::Z, None, DVec3::from(p));
        let e = ExtrudeExtent::Distance { from: 0.0, to: 10.0 };
        assert!(at(&e, [0.0, 0.0, 0.0]).err().is_some_and(|m| m.contains("no height")));
        let last = ExtrudeExtent::ToLast { from: 0.0, reverse: false };
        assert!(at(&last, [0.0, 0.0, 5.0]).is_ok_and(|(_, s)| s.starts_with("From = 5.00")));
        assert!(at(&last, [0.0, 0.0, -5.0]).is_ok_and(|(_, s)| s.starts_with("From = -5.00")));
        let sym = ExtrudeExtent::Symmetric { total: 10.0 };
        assert!(at(&sym, [0.0, 0.0, 25.4]).is_ok_and(|(e, s)| s.starts_with("Total 50.80 mm — each half") && e == ExtrudeExtent::Symmetric { total: 50.8 }));
        // Swept askew, the offset counts along the sweep.
        let tilt = DVec3::new(1.0, 0.0, 1.0).normalize();
        let (e, _) = extrude_to_point(&ExtrudeExtent::Distance { from: 0.0, to: 1.0 }, DVec3::ZERO, DVec3::Z, Some(tilt), DVec3::new(0.0, 0.0, 10.0)).expect("To");
        assert_eq!(e, ExtrudeExtent::Distance { from: 0.0, to: 14.14 });
    }

    #[test]
    fn a_pulled_face_joins_going_out_and_cuts_going_in() {
        assert_eq!(face_extrude_op(&ExtrudeExtent::Distance { from: 0.0, to: 5.0 }), BodyOp::Join);
        assert_eq!(face_extrude_op(&ExtrudeExtent::Distance { from: 0.0, to: -3.0 }), BodyOp::Cut);
        // A click on the part measures from the face's plane.
        let (_, said) = extrude_to_point(&ExtrudeExtent::Distance { from: 0.0, to: 5.0 }, DVec3::new(20.0, 10.0, 10.0), DVec3::Z, None, DVec3::new(5.0, 5.0, 14.0)).expect("To");
        assert!(said.starts_with("To = 4.00 mm"), "{said}");
    }

    #[test]
    fn a_coil_starts_with_a_pitch_half_again_the_profiles_reach_and_three_turns() {
        // A 2 × 2 square off the Y axis, on the XY plane: 2 mm along Y, 3 mm a turn.
        let mut s = Sketch::default();
        s.add_rectangle(DVec2::new(5.0, 0.0), DVec2::new(7.0, 2.0));
        let regions = find_regions(&s);
        let frame = SectionFrame { origin: DVec3::ZERO, normal: DVec3::Z, u: DVec3::X, v: DVec3::Y };
        let refs: Vec<&Region> = regions.iter().collect();
        assert_eq!(starting_pitch(&frame, &refs, Some((DVec3::ZERO, DVec3::Y))), 3.0);
        assert_eq!(starting_pitch(&frame, &refs, None), 5.0);
        let mut r = RevolveFeature::new(1, vec![regions[0].key.clone()], AxisRef::World { axis: crate::frame::WorldAxis::Y }, 360.0, BodyOp::Join);
        set_coil(&mut r, true, || starting_pitch(&frame, &refs, Some((DVec3::ZERO, DVec3::Y))));
        assert_eq!((r.pitch, r.angle_deg), (Some(3.0), 1080.0));
        // Made plain again it stops at a full turn and keeps no pitch.
        set_coil(&mut r, false, || 0.0);
        assert_eq!((r.pitch, r.angle_deg, r.left_handed), (None, 360.0, false));
    }

    #[test]
    fn keeps_a_plain_revolve_to_a_full_turn_however_the_angle_is_given() {
        assert_eq!(revolve_angle(900.0, 0.0), 360.0);
        assert_eq!(revolve_angle(1800.0, 3.0), 1800.0);
        assert_eq!(revolve_angle(-90.04, 0.0), 90.0);
        assert_eq!(revolve_angle(0.0, 0.0), 0.1);
    }

    #[test]
    fn an_extrude_is_ready_with_a_profile_and_an_extent_that_names_all_it_needs() {
        let regions = square_regions();
        let mut e = ExtrudeFeature::new(1, auto_regions(&regions), ExtrudeExtent::Distance { from: 0.0, to: 10.0 }, BodyOp::Join);
        assert!(extrude_ready(&e).is_ok());
        e.extent = ExtrudeExtent::ToPlane { from: 0.0, plane: no_plane() };
        assert!(extrude_ready(&e).is_err());
        e.extent = ExtrudeExtent::ToBody { from: 0.0, body: NO_BODY, reverse: false };
        assert!(extrude_ready(&e).is_err());
        e.extent = ExtrudeExtent::ToBody { from: 0.0, body: BodyId::new(2), reverse: false };
        assert!(extrude_ready(&e).is_ok());
        e.regions.clear();
        assert!(extrude_ready(&e).is_err());
        let mut r = RevolveFeature::new(1, auto_regions(&regions), no_axis(), 360.0, BodyOp::Join);
        assert_eq!(revolve_ready(&r), Err("The revolve needs a profile, an axis and an angle.".into()));
        r.axis = AxisRef::World { axis: crate::frame::WorldAxis::Y };
        assert!(revolve_ready(&r).is_ok());
        r.angle_deg = 720.0;
        assert!(revolve_ready(&r).is_err_and(|m| m.contains("360° at most")));
    }

    #[test]
    fn takes_regions_as_they_were_asked_for() {
        let mut s = Sketch::default();
        s.add_rectangle(DVec2::new(0.0, 0.0), DVec2::new(40.0, 20.0));
        s.add_shape(Shape::Circle { c: DVec2::new(20.0, 10.0), r: 4.0 });
        let regions = find_regions(&s);
        assert!(region_keys("Sketch 1", &regions, None).is_err_and(|m| m.starts_with("Sketch 1 has 2 closed regions — say which")));
        assert_eq!(region_keys("Sketch 1", &regions, region_given_of("all").as_ref()), Ok(every_region(&regions)));
        assert_eq!(region_keys("Sketch 1", &regions, region_given_of("largest").as_ref()), Ok(vec![regions[0].key.clone()]));
        assert_eq!(region_keys("Sketch 1", &regions, region_given_of("@20,10").as_ref()), Ok(vec![regions[1].key.clone()]));
        assert!(region_keys("Sketch 1", &regions, region_given_of("@90,90").as_ref()).is_err_and(|m| m.starts_with("No region of Sketch 1 holds the point [90, 90]")));
        assert!(region_keys("Sketch 1", &regions, region_given_of("nope").as_ref()).is_err_and(|m| m.starts_with("Sketch 1 has no region nope")));
        // Kept as given where by more than keys.
        let all = region_given_of("all");
        assert_eq!(region_pick_of(all.as_ref(), &every_region(&regions)).map(|p| p.given), all);
        assert_eq!(region_pick_of(region_given_of(&regions[0].key).as_ref(), &[regions[0].key.clone()]), None);
        assert!(region_pick_of(region_given_of("@2,2").as_ref(), &[regions[0].key.clone()]).is_some());
        assert_eq!(auto_regions(&regions), Vec::<String>::new());
        assert_eq!(auto_regions(&square_regions()).len(), 1);
    }
}
