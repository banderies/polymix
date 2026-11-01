// SPDX-License-Identifier: AGPL-3.0-only
//! Script steps for the Extrude and the Revolve boxes, so they can be driven without a hand on
//! the mouse (tests, GUI checks): open a box, set its extent, taper, direction, axis, coil and
//! result as the box's own fields do, drag a grip by an amount, click a height on the part, pull a
//! face, read the end off the scan, and make or put away the feature.
//!
//! - `extrude-box[:<regions>]`, `revolve-box[:<regions>]` — regions `all`, `largest`, keys, or
//!   `@u,v` points, separated by `;`, kept as given
//! - `extent:distance:<from>:<to>`, `extent:symmetric:<total>`, `extent:plane:<from>:<x|y|z>:<offset>`,
//!   `extent:between:<axis>:<offset>:<axis>:<offset>`, `extent:face:<from>:<x>:<y>:<z>` (the face
//!   nearest the point), `extent:body:<from>:<feature>[:rev]`, `extent:next:<from>[:rev]`,
//!   `extent:last:<from>[:rev]`
//! - `taper:<deg>` (positive leans out), `direction:<x>:<y>:<z>` or `direction:normal`,
//!   `op:join|cut|new`, `flip`, `point:<x>:<y>:<z>` (a click on the part at that point)
//! - `pull:<x>:<y>:<z>` — an Extrude box pulling the flat face nearest the point (`pull:top`, the
//!   highest face looking up)
//! - `axis:<x|y|z>` or `axis:line:<id>[:<sketch>]`, `angle:<deg>`, `coil:on|off`, `pitch:<mm>`,
//!   `turns:<n>`, `left:on|off`
//! - `grip:<from|to|taper|turn>:<pull>` — a grip dragged by that pull (mm, or degrees)
//! - `toscan[:<near>]` — the extrude's end read off the scan
//! - `create`, `cancel`, `boxlog` — the box's words and state to the log

use glam::DVec3;
use mm_reverse::extrude_draft::{region_given_of, region_keys, region_pick_of};
use mm_reverse::extrude_grips::{GripSide, MAX_TAPER_DEG};
use mm_reverse::frame::WorldAxis;
use mm_reverse::model::{AxisRef, BodyId, BodyOp, ExtrudeExtent, FeatureKind, PlaneRef};

use super::ext_fields::{with_box, GripDrag};
use super::{extrude, revolve};
use crate::reverse::{Mode, ReverseWorkspace};

fn axis_of(s: Option<&&str>) -> Option<WorldAxis> {
    match s.copied() {
        Some("x") => Some(WorldAxis::X),
        Some("y") => Some(WorldAxis::Y),
        Some("z") => Some(WorldAxis::Z),
        _ => None,
    }
}

/// A step for the Extrude or Revolve box; `false` for a step that is none of these.
pub(crate) fn step(ws: &mut ReverseWorkspace, cmd: &str, args: &[&str]) -> bool {
    let num = |i: usize| args.get(i).and_then(|v| v.parse::<f64>().ok());
    let point = |i: usize| Some(DVec3::new(num(i)?, num(i + 1)?, num(i + 2)?));
    match cmd {
        "extrude-box" | "revolve-box" => {
            let given = args.first().and_then(|t| region_given_of(&args[0..].join(":")).filter(|_| !t.is_empty()));
            let sketch = ws.last_sketch();
            let (regions, kept) = match (&given, sketch) {
                (Some(g), Some(s)) => {
                    let name = ws.part.feature(s).map_or(String::new(), |f| f.name.clone());
                    match region_keys(&name, &extrude::regions_of(ws, s), Some(g)) {
                        Ok(k) => {
                            let kept = region_pick_of(Some(g), &k);
                            (Some(k), kept)
                        }
                        Err(e) => {
                            ws.status = e;
                            log::info!("reverse box: {}", ws.status);
                            return true;
                        }
                    }
                }
                _ => (None, None),
            };
            if cmd == "extrude-box" {
                extrude::open(ws, None, regions);
            } else {
                revolve::open(ws, None, regions, None);
            }
            if let Mode::Feature(edit) = &mut ws.mode {
                match &mut edit.kind {
                    FeatureKind::Extrude(e) => e.region_pick = kept,
                    FeatureKind::Revolve(r) => r.region_pick = kept,
                    _ => {}
                }
            }
        }
        "pull" => {
            // `pull:top`: the highest face of the bodies looking up.
            let top = || {
                let mut best: Option<(f64, DVec3)> = None;
                for b in ws.pick_set() {
                    for f in &b.topology.faces {
                        if f.fingerprint.dir.z > 0.99 && best.is_none_or(|x| f.fingerprint.centroid.z > x.0) {
                            best = Some((f.fingerprint.centroid.z, f.point));
                        }
                    }
                }
                best.map(|b| b.1)
            };
            let Some(at) = (if args.first() == Some(&"top") { top() } else { point(0) }) else { return true };
            match extrude::face_near(ws, at) {
                Some((body, face)) => {
                    if let Err(e) = extrude::open_face(ws, body, face) {
                        ws.status = e;
                    }
                }
                None => ws.status = "There is no face there to pull.".into(),
            }
        }
        "create" => ws.commit_feature(),
        "cancel" => ws.close_mode(),
        "boxlog" => {
            let Mode::Feature(edit) = &ws.mode else {
                log::info!("reverse box: none open; status: {}", ws.status);
                return true;
            };
            let words = super::box_words(ws, edit);
            let id = ws.preview_feature;
            let json = serde_json::to_string(&edit.kind).unwrap_or_default();
            log::info!(
                "reverse box: {} | {} | ready {} | ghost {:?} | error {:?} | grips {} | {json}",
                words.as_ref().map_or("", |w| w.title.as_str()),
                words.as_ref().map_or("", |w| w.create.as_str()),
                words.as_ref().is_some_and(|w| w.ready),
                id.and_then(|i| ws.built.ghost_volumes.get(&i)),
                id.and_then(|i| ws.built.errors.get(&i)),
                match &edit.kind {
                    FeatureKind::Extrude(e) => extrude::grips_of(ws, e).len(),
                    FeatureKind::Revolve(r) => revolve::grips_of(ws, r).len(),
                    _ => 0,
                }
            );
        }
        "toscan" => {
            let near = num(0);
            let mut mode = std::mem::replace(&mut ws.mode, Mode::Idle);
            if let Mode::Feature(edit) = &mut mode {
                if let FeatureKind::Extrude(e) = &mut edit.kind {
                    ws.status = match extrude::fit_end_to_scan(ws, e, near) {
                        Ok(s) | Err(s) => s,
                    };
                    log::info!("reverse toscan: {}", ws.status);
                }
            }
            ws.mode = mode;
            ws.request_build();
        }
        "point" => {
            let Some(at) = point(0) else { return true };
            let mut mode = std::mem::replace(&mut ws.mode, Mode::Idle);
            if let Mode::Feature(edit) = &mut mode {
                if let FeatureKind::Extrude(e) = &mut edit.kind {
                    // As a click on the part there: only a Distance or a Symmetric extent takes it.
                    if !extrude::click_point(ws, e, at) && !extrude::takes_point(&e.extent) {
                        log::info!("reverse point: this extent takes no point on the part");
                    }
                }
            }
            ws.mode = mode;
            ws.request_build();
        }
        "grip" if matches!(&ws.mode, Mode::Feature(edit) if matches!(edit.kind, FeatureKind::Revolve(_))) => turn_ring(ws, num(1).unwrap_or(0.0)),
        "extent" | "taper" | "direction" | "op" | "flip" | "grip" => {
            // The box's feature taken out while the workspace is read, and put back.
            let mut mode = std::mem::replace(&mut ws.mode, Mode::Idle);
            if let Mode::Feature(edit) = &mut mode {
                if let FeatureKind::Extrude(e) = &mut edit.kind {
                    extrude_step(ws, cmd, args, e);
                }
            }
            ws.mode = mode;
            ws.request_build();
        }
        "axis" | "angle" | "coil" | "pitch" | "turns" | "left" => {
            let mut mode = std::mem::replace(&mut ws.mode, Mode::Idle);
            if let Mode::Feature(edit) = &mut mode {
                if let FeatureKind::Revolve(r) = &mut edit.kind {
                    revolve_step(ws, cmd, args, r);
                }
            }
            ws.mode = mode;
            ws.request_build();
        }
        _ => return false,
    }
    true
}

/// A step on the open Extrude box's feature.
fn extrude_step(ws: &mut ReverseWorkspace, cmd: &str, args: &[&str], e: &mut mm_reverse::model::ExtrudeFeature) {
    let num = |i: usize| args.get(i).and_then(|v| v.parse::<f64>().ok());
    let point = |i: usize| Some(DVec3::new(num(i)?, num(i + 1)?, num(i + 2)?));
    match cmd {
        "extent" => {
            let rev = args.iter().any(|a| *a == "rev");
            let from = num(1).unwrap_or(0.0);
            let next = match args.first().copied() {
                Some("distance") => Some(ExtrudeExtent::Distance { from: num(1).unwrap_or(0.0), to: num(2).unwrap_or(10.0) }),
                Some("symmetric") => Some(ExtrudeExtent::Symmetric { total: num(1).unwrap_or(10.0) }),
                Some("plane") => axis_of(args.get(2)).map(|a| ExtrudeExtent::ToPlane { from, plane: PlaneRef::world(a, num(3).unwrap_or(0.0)) }),
                Some("between") => match (axis_of(args.get(1)), axis_of(args.get(3))) {
                    (Some(a), Some(b)) => Some(ExtrudeExtent::BetweenPlanes { a: PlaneRef::world(a, num(2).unwrap_or(0.0)), b: PlaneRef::world(b, num(4).unwrap_or(0.0)) }),
                    _ => None,
                },
                Some("next") => Some(ExtrudeExtent::ToNext { from, reverse: rev }),
                Some("last") => Some(ExtrudeExtent::ToLast { from, reverse: rev }),
                Some("body") => num(2).map(|f| ExtrudeExtent::ToBody { from, body: BodyId::new(f as u32), reverse: rev }),
                Some("face") => point(2).and_then(|p| extrude::face_near(ws, p)).and_then(|(body, i)| ws.pick_view(body).and_then(|v| v.topology.faces.get(i)).map(|f| ExtrudeExtent::ToFace { from, body, face: f.fingerprint })),
                Some(k) => Some(extrude::extent_of_kind(k)),
                None => None,
            };
            if let Some(x) = next {
                e.extent = x;
            }
        }
        "taper" => {
            // As the box's field takes it: within ±85°.
            let t = num(0).unwrap_or(0.0).clamp(-MAX_TAPER_DEG, MAX_TAPER_DEG);
            e.draft_deg = if t == 0.0 { 0.0 } else { -t };
        }
        "direction" => e.direction = if args.first() == Some(&"normal") { None } else { point(0).and_then(|d| d.try_normalize()) },
        "op" => {
            let op = match args.first().copied() {
                Some("cut") => BodyOp::Cut,
                Some("new") => BodyOp::New,
                _ => BodyOp::Join,
            };
            extrude::set_op(e, op);
        }
        "flip" => extrude::flip(e),
        _ => {
            let side = match args.first().copied() {
                Some("from") => GripSide::ExtrudeFrom,
                Some("taper") => GripSide::Taper,
                _ => GripSide::ExtrudeTo,
            };
            match extrude::grips_of(ws, e).into_iter().find(|g| g.side == side) {
                Some(g) => {
                    let drag = extrude::drag_of(e, g, 0.0);
                    extrude::apply_drag(e, &drag, num(1).unwrap_or(0.0));
                }
                None => ws.status = "The extrude wears no such grip now.".into(),
            }
        }
    }
}

/// A step on the open Revolve box's feature.
fn revolve_step(ws: &mut ReverseWorkspace, cmd: &str, args: &[&str], r: &mut mm_reverse::model::RevolveFeature) {
    let num = |i: usize| args.get(i).and_then(|v| v.parse::<f64>().ok());
    match cmd {
        "axis" => match (args.first().copied(), num(1)) {
            // A line of the box's sketch, or of the sketch named after it — as a click on it.
            (Some("line"), Some(id)) => {
                let sketch = num(2).map_or(r.sketch_id, |s| s as u32);
                revolve::take_line(ws, r, sketch, id as u32);
            }
            _ => r.axis = axis_of(args.first()).map_or(r.axis, |axis| AxisRef::World { axis }),
        },
        "angle" => r.angle_deg = mm_reverse::extrude_draft::revolve_angle(num(0).unwrap_or(360.0), r.pitch.unwrap_or(0.0)),
        "coil" => {
            let frame = ws.sketch_frame(r.sketch_id);
            let regions = extrude::regions_of(ws, r.sketch_id);
            let picked: Vec<&mm_reverse::regions::Region> = regions.iter().filter(|x| r.regions.contains(&x.key)).collect();
            let along = revolve::axis_in_part(ws, r.sketch_id, &r.axis);
            let pitch = frame.map_or(5.0, |f| mm_reverse::extrude_draft::starting_pitch(&f, &picked, along));
            mm_reverse::extrude_draft::set_coil(r, args.first() != Some(&"off"), || pitch);
        }
        "pitch" => {
            if let Some(p) = num(0).filter(|p| *p != 0.0) {
                r.pitch = Some(p);
            }
        }
        "turns" => r.angle_deg = mm_reverse::extrude_draft::revolve_angle(num(0).unwrap_or(1.0) * 360.0, r.pitch.unwrap_or(0.0)),
        _ => r.left_handed = args.first() != Some(&"off"),
    }
}

/// A revolve's ring dragged by `deg` in a script.
pub(crate) fn turn_ring(ws: &mut ReverseWorkspace, deg: f64) {
    let mut mode = std::mem::replace(&mut ws.mode, Mode::Idle);
    if let Mode::Feature(edit) = &mut mode {
        if let FeatureKind::Revolve(r) = &mut edit.kind {
            if let Some(g) = revolve::grips_of(ws, r).first().copied() {
                let drag = GripDrag { side: GripSide::Turn, grip: g, t0: 0.0, last: 0.0, wound: 0.0, extent: None, taper_deg: 0.0, taper_run: 0.0, angle_deg: r.angle_deg };
                revolve::apply_drag(r, &drag, deg);
            }
        }
    }
    ws.mode = mode;
    with_box(|b| b.owed = false);
    ws.request_build();
}

#[cfg(test)]
mod tests {
    // The box's own rules, as ScanRuler's store tests read them (`reStore.test.ts`,
    // `reFeatureDraft.test.ts`): Reverse direction, the result turning a cut over, a pulled
    // face's result following the way it is pulled, the grips' drags and typed values, and a
    // coil named apart from the revolves.
    use super::super::ext_fields::{reset_box, with_box};
    use super::super::{add_new, extrude, not_ready};
    use mm_reverse::extrude_grips::{FeatureGrip, GripShape, GripSide};
    use mm_reverse::frame::WorldAxis;
    use mm_reverse::model::{AxisRef, BodyId, BodyOp, ExtrudeExtent, ExtrudeFeature, FaceProfile, FeatureKind, Part, RevolveFeature};
    use mm_reverse::topo::FaceKind;

    fn distance(from: f64, to: f64) -> ExtrudeFeature {
        ExtrudeFeature::new(1, vec!["1,2,3,4".into()], ExtrudeExtent::Distance { from, to }, BodyOp::Join)
    }

    #[test]
    fn reverse_direction_turns_a_distance_over_and_looks_the_other_way_for_an_end_on_the_part() {
        reset_box();
        let mut e = distance(-2.0, 10.0);
        extrude::flip(&mut e);
        assert_eq!(e.extent, ExtrudeExtent::Distance { from: 2.0, to: -10.0 });
        e.extent = ExtrudeExtent::ToNext { from: 0.0, reverse: false };
        extrude::flip(&mut e);
        assert_eq!(e.extent, ExtrudeExtent::ToNext { from: 0.0, reverse: true });
        // Going to a cut turns the extrude over, so it cuts into what it meets.
        let mut e = distance(0.0, 10.0);
        extrude::set_op(&mut e, BodyOp::Cut);
        assert_eq!((e.op, e.extent.clone()), (BodyOp::Cut, ExtrudeExtent::Distance { from: 0.0, to: -10.0 }));
        extrude::set_op(&mut e, BodyOp::New);
        assert_eq!(e.extent, ExtrudeExtent::Distance { from: 0.0, to: 10.0 });
    }

    #[test]
    fn a_pulled_face_joins_going_out_and_cuts_going_in_until_a_result_is_chosen_by_hand() {
        reset_box();
        with_box(|b| b.op_auto = true);
        let face = mm_reverse::model::FaceRef { kind: FaceKind::Plane, centroid: glam::DVec3::new(20.0, 10.0, 10.0), dir: glam::DVec3::Z, area: 800.0, radius: None, center: None, origin: 2 };
        let mut e = ExtrudeFeature::new(0, Vec::new(), ExtrudeExtent::Distance { from: 0.0, to: 5.0 }, BodyOp::Join);
        e.face = Some(FaceProfile { body: BodyId::new(2), face });
        extrude::set_grip_value(&mut e, GripSide::ExtrudeTo, -3.0);
        assert_eq!(e.op, BodyOp::Cut);
        extrude::flip(&mut e);
        assert_eq!(e.op, BodyOp::Join);
        // A result chosen by hand stays chosen.
        extrude::set_op(&mut e, BodyOp::New);
        extrude::set_grip_value(&mut e, GripSide::ExtrudeTo, -3.0);
        assert_eq!(e.op, BodyOp::New);
    }

    #[test]
    fn a_drag_pulls_the_extent_it_began_with_to_whole_hundredths_and_lets_a_parameter_go() {
        reset_box();
        let mut e = distance(0.0, 10.0);
        e.exprs.insert("extent.to", "h");
        let grip = FeatureGrip { side: GripSide::ExtrudeTo, shape: GripShape::Arrow, position: glam::DVec3::ZERO, dir: glam::DVec3::Z, size: 44.0, pixels: true };
        let drag = extrude::drag_of(&e, grip, 0.0);
        extrude::apply_drag(&mut e, &drag, 2.3456);
        assert_eq!(e.extent, ExtrudeExtent::Distance { from: 0.0, to: 12.35 });
        assert!(!e.exprs.contains("extent.to"));
        // The taper's arrow: the far edge pulled out by the run is 45°.
        let taper = FeatureGrip { side: GripSide::Taper, ..grip };
        let drag = extrude::drag_of(&e, taper, 0.0);
        extrude::apply_drag(&mut e, &drag, 12.35);
        assert_eq!(e.draft_deg, -45.0);
        // Typed on a symmetric extent's grip: the whole thickness.
        e.extent = ExtrudeExtent::Symmetric { total: 4.0 };
        extrude::set_grip_value(&mut e, GripSide::ExtrudeFrom, -7.0);
        assert_eq!(e.extent, ExtrudeExtent::Symmetric { total: 7.0 });
    }

    #[test]
    fn only_an_extrude_by_numbers_takes_a_point_clicked_on_the_part() {
        // ScanRuler arms the part's point pick for a Distance or a Symmetric extent alone; an end
        // found on the part leaves a click on it to nothing.
        assert!(extrude::takes_point(&ExtrudeExtent::Distance { from: 0.0, to: 10.0 }));
        assert!(extrude::takes_point(&ExtrudeExtent::Symmetric { total: 10.0 }));
        for kind in ["toPlane", "toFace", "toBody", "toNext", "toLast", "betweenPlanes"] {
            assert!(!extrude::takes_point(&extrude::extent_of_kind(kind)), "{kind}");
        }
    }

    #[test]
    fn a_round_value_chip_reads_as_the_round_number_with_its_unit() {
        use super::super::ext_fields::nominal_text;
        use mm_reverse::nominal::NominalKind;
        assert_eq!(nominal_text(15.0, NominalKind::Angle), "15°");
        assert_eq!(nominal_text(12.0, NominalKind::Length), "12 mm");
        assert_eq!(nominal_text(12.5, NominalKind::Length), "12.5 mm");
        assert_eq!(nominal_text(1.5, NominalKind::Draft), "1.5°");
    }

    #[test]
    fn a_coil_is_named_apart_from_the_revolves_and_a_box_not_ready_says_why() {
        let mut part = Part::default();
        let mut coil = RevolveFeature::new(1, vec!["k".into()], AxisRef::World { axis: WorldAxis::Z }, 1080.0, BodyOp::New);
        coil.pitch = Some(3.0);
        let c = add_new(&mut part, FeatureKind::Revolve(coil));
        let r = add_new(&mut part, FeatureKind::Revolve(RevolveFeature::new(1, vec!["k".into()], AxisRef::World { axis: WorldAxis::Z }, 360.0, BodyOp::New)));
        assert_eq!(part.feature(c).map(|f| f.name.as_str()), Some("Coil 1"));
        assert_eq!(part.feature(r).map(|f| f.name.as_str()), Some("Revolve 1"));
        let none = RevolveFeature::new(1, vec!["k".into()], mm_reverse::extrude_draft::no_axis(), 360.0, BodyOp::New);
        assert_eq!(not_ready(&FeatureKind::Revolve(none)).as_deref(), Some("The revolve needs a profile, an axis and an angle."));
        let mut e = distance(0.0, 10.0);
        e.extent = extrude::extent_of_kind("toPlane");
        assert!(not_ready(&FeatureKind::Extrude(e)).is_some());
    }
}
