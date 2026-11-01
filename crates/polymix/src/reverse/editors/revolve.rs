// SPDX-License-Identifier: AGPL-3.0-only
//! The Revolve feature's box: the sketch and its profiles, the axis — a line of the sketch
//! clicked on the part or chosen, a measured line, cylinder, cone or circle, or a coordinate axis
//! — the angle, or a coil's pitch, turns, height and hand, and what it does with the bodies it
//! meets; the preview's volume, the ⓘ cards, and the ring about the axis that drags the angle.
//! Ported from ScanRuler (`plugins/reverse/ui/RevolveEditor.tsx`, `state/reSolidSlice.ts` the
//! revolve), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.

use egui::{Pos2, Stroke};
use glam::DVec3;
use mm_reverse::extrude_draft::{auto_regions, axis_chosen, no_axis, revolve_angle, revolve_ready, set_coil, starting_pitch};
use mm_reverse::extrude_grips::{drag_revolve_angle, revolve_grip, FeatureGrip, GripSide};
use mm_reverse::frame::WorldAxis;
use mm_reverse::model::{revolve_max_deg, AxisRef, BodyOp, ElementGeometry, FeatureId, FeatureKind, RevolveFeature, FULL_TURN_DEG};
use mm_reverse::nominal::NominalKind;
use mm_reverse::regions::Region;
use mm_reverse::sketch::Shape;

use super::ext_fields::{self as fields, info_dot, num, preview_line, profiles_field, ref_picker, with_box, GripDrag, GripField, Hint, Unit};
use super::extrude::{region_under, regions_of, toggle_region};
use crate::reverse::{PickWhat, ReverseWorkspace};
use crate::tools::Viewport;

pub(crate) const FIRST_PICK: PickWhat = PickWhat::Regions;

/// Revolve stands by the sketch in the panel, not among these buttons.
#[allow(dead_code)]
pub(crate) fn new_feature(_ws: &ReverseWorkspace) -> Result<FeatureKind, String> {
    Err("Revolve starts from a sketch.".into())
}

/// Open a Revolve box on a sketch — the last one, unless one is named — with the regions given,
/// or its one region where it has exactly one; about the axis given, or none until one is chosen;
/// a full turn, no pitch, Join.
pub(crate) fn open(ws: &mut ReverseWorkspace, sketch: Option<FeatureId>, regions: Option<Vec<String>>, axis: Option<AxisRef>) {
    let Some(sketch) = sketch.or_else(|| ws.last_sketch()) else {
        ws.status = "Revolve takes a sketch: make one first (New sketch).".into();
        return;
    };
    let regions = regions.unwrap_or_else(|| auto_regions(&regions_of(ws, sketch)));
    fields::reset_box();
    ws.open_feature(FeatureKind::Revolve(RevolveFeature::new(sketch, regions, axis.unwrap_or_else(no_axis), FULL_TURN_DEG, BodyOp::Join)), None);
}

/// The box's title, its Create key, and whether it is ready.
pub(crate) fn box_words(ws: &ReverseWorkspace, editing: Option<FeatureId>, r: &RevolveFeature) -> super::BoxWords {
    let name = editing.and_then(|id| ws.part.feature(id)).map(|f| f.name.clone());
    let coil = is_coil(r);
    super::BoxWords {
        title: match &name {
            Some(n) => format!("Edit {n}"),
            None if coil => "New coil".into(),
            None => "New revolve".into(),
        },
        create: match (&name, coil) {
            (Some(_), _) => "Save changes".into(),
            (None, true) => "Create coil".into(),
            (None, false) => "Create revolve".into(),
        },
        tip: "Enter or a middle click does the same",
        ready: revolve_ready(r).is_ok(),
        own_preview: true,
        create_now: false,
    }
}

fn is_coil(r: &RevolveFeature) -> bool {
    r.pitch.is_some_and(|p| p != 0.0 && p.is_finite())
}

/// The axes a revolve can turn about: the sketch's lines, the measured lines, cylinders, cones
/// and circles, and the coordinate axes — by key, label, group and reference.
fn axis_options(ws: &ReverseWorkspace, sketch: FeatureId) -> Vec<(String, String, &'static str, AxisRef)> {
    let mut out = Vec::new();
    if let (Some(s), Some(f)) = (ws.part.sketch(sketch), ws.part.feature(sketch)) {
        for e in &s.sketch.entities {
            if let Shape::Line { .. } = e.shape {
                out.push((format!("sketch:{}", e.id), format!("{} line {}{}", f.name, e.id, if e.construction { " (construction)" } else { "" }), "Sketch lines", AxisRef::SketchLine { entity: e.id }));
            }
        }
    }
    for el in &ws.part.elements {
        if matches!(el.geometry, ElementGeometry::Line { .. } | ElementGeometry::Cylinder { .. } | ElementGeometry::Cone { .. } | ElementGeometry::Circle { .. }) {
            out.push((format!("element:{}", el.id), el.name.clone(), "Elements", AxisRef::Element { id: el.id }));
        }
    }
    for axis in WorldAxis::ALL {
        out.push((format!("world:{axis:?}"), format!("{axis:?} axis"), "Coordinate axes", AxisRef::World { axis }));
    }
    out
}

fn axis_key(a: &AxisRef) -> Option<String> {
    if !axis_chosen(a) {
        return None;
    }
    Some(match a {
        AxisRef::SketchLine { entity } => format!("sketch:{entity}"),
        AxisRef::Element { id } => format!("element:{id}"),
        AxisRef::World { axis } => format!("world:{axis:?}"),
    })
}

/// An axis in the part: a sketch line lifted through the sketch's frame, or what the reference
/// names.
pub(crate) fn axis_in_part(ws: &ReverseWorkspace, sketch: FeatureId, a: &AxisRef) -> Option<(DVec3, DVec3)> {
    if !axis_chosen(a) {
        return None;
    }
    match a {
        AxisRef::SketchLine { entity } => {
            let frame = ws.sketch_frame(sketch)?;
            match ws.part.sketch(sketch)?.sketch.entity(*entity).map(|e| e.shape.clone()) {
                Some(Shape::Line { a, b }) => {
                    let (pa, pb) = (frame.to_world(a), frame.to_world(b));
                    Some((pa, (pb - pa).try_normalize()?))
                }
                _ => None,
            }
        }
        other => mm_reverse::build::resolve_axis(other, None, &ws.part),
    }
}

fn axis_hint(ws: &ReverseWorkspace, sketch: FeatureId, a: &AxisRef, name: &str) -> Option<Hint> {
    let (origin, dir) = axis_in_part(ws, sketch, a)?;
    Some(Hint::Axis { origin, dir, name: name.to_string() })
}

/// Lets a parameter go from the slots a typed or dragged number stands in for.
fn unlink(r: &mut RevolveFeature, paths: &[&str]) {
    for p in paths {
        r.exprs.remove(p);
    }
}

pub(crate) fn ui(ws: &mut ReverseWorkspace, ui: &mut egui::Ui, r: &mut RevolveFeature, pick: &mut PickWhat) -> bool {
    fields::box_for(ws, "revolve");
    *pick = PickWhat::Regions;
    let mut changed = false;
    // Saved from its box, a feature joins or cuts every body it meets (ScanRuler's
    // `withoutTarget`).
    if r.target.take().is_some() {
        changed = true;
    }
    // The sketch: another chosen takes its own regions, and lets the axis go.
    let sketches: Vec<(FeatureId, String)> = ws.part.features.iter().filter(|f| matches!(f.kind, FeatureKind::Sketch(_))).map(|f| (f.id, f.name.clone())).collect();
    ui.horizontal(|ui| {
        ui.label("Sketch");
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let name = sketches.iter().find(|s| s.0 == r.sketch_id).map_or_else(|| "Select…".to_string(), |s| s.1.clone());
            egui::ComboBox::from_id_salt("re_revolve_sketch").selected_text(name).width(170.0).show_ui(ui, |ui| {
                for (id, n) in &sketches {
                    if ui.selectable_label(r.sketch_id == *id, n).clicked() && r.sketch_id != *id {
                        r.sketch_id = *id;
                        r.regions = auto_regions(&regions_of(ws, *id));
                        r.region_pick = None;
                        r.axis = no_axis();
                        changed = true;
                    }
                }
            });
        });
    });
    changed |= profiles_field(ui, ws, r.sketch_id, &mut r.regions, &mut r.region_pick);
    // The axis.
    let options = axis_options(ws, r.sketch_id);
    ui.horizontal(|ui| {
        ui.label("Axis");
        info_dot(ui, ws, "axis", "Axis", &["Click a line of the sketch on the part, or a line, cylinder, cone or circle element, or choose here. A construction line drawn down the middle of the profile is the usual axis."]);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let listed: Vec<(String, String, &'static str)> = options.iter().map(|o| (o.0.clone(), o.1.clone(), o.2)).collect();
            let chosen = axis_key(&r.axis);
            let (picked, hovered) = ref_picker(ui, "re_revolve_axis", chosen.as_deref(), "Click in the viewport, or choose…", &listed);
            if let Some(o) = picked.and_then(|k| options.iter().find(|o| o.0 == k)) {
                r.axis = o.3;
                changed = true;
            }
            let hover = hovered.and_then(|k| options.iter().find(|o| o.0 == k)).and_then(|o| axis_hint(ws, r.sketch_id, &o.3, &o.1));
            let shown = axis_key(&r.axis).and_then(|k| options.iter().find(|o| o.0 == k)).and_then(|o| axis_hint(ws, r.sketch_id, &o.3, &o.1));
            with_box(|b| {
                b.hover_hint = hover;
                b.chosen_hints = shown.into_iter().collect();
            });
        });
    });
    preview_line(ui, ws, revolve_ready(r).is_ok());
    // A coil, or a plain revolve.
    let mut coil = is_coil(r);
    ui.horizontal(|ui| {
        if ui.checkbox(&mut coil, "Coil — climb along the axis").changed() {
            let frame = ws.sketch_frame(r.sketch_id);
            let regions = regions_of(ws, r.sketch_id);
            let picked: Vec<&Region> = regions.iter().filter(|x| r.regions.contains(&x.key)).collect();
            let axis = axis_in_part(ws, r.sketch_id, &r.axis);
            set_coil(r, coil, || frame.map_or(5.0, |f| starting_pitch(&f, &picked, axis)));
            changed = true;
        }
        info_dot(
            ui,
            ws,
            "coil",
            "Coil",
            &[
                "The profile turns about the axis and climbs along it by the pitch every turn: a spring, a thread, a worm. The angle runs on past 360° — as many turns as you like; type them as turns or as a height, or wind the ring round and round.",
                "The profile has to reach less far along the axis than the pitch, or each turn would run into the next. For a thread, draw the groove's section on a plane through the bar's axis, reaching a little past the bar, and set Result to Cut.",
            ],
        );
    });
    let pitch = r.pitch.unwrap_or(0.0);
    let mut angle = r.angle_deg;
    let angle_expr = r.exprs.get("angleDeg").map(str::to_string);
    let (_, c) = num(
        ui,
        ws,
        "Angle",
        &mut angle,
        Unit::Deg,
        Some(NominalKind::Angle),
        Some(if is_coil(r) { "Degrees turned — any number of turns: 1080° is three. The ring about the axis drags it, round and round." } else { "360° for a whole body of revolution — a full turn at most; tick Coil to turn on past it. The ring about the axis drags it." }),
        angle_expr.as_deref(),
        Some(0.1..=revolve_max_deg(Some(pitch))),
    );
    if c {
        r.angle_deg = revolve_angle(angle, pitch);
        changed = true;
    }
    if is_coil(r) {
        let mut p = pitch;
        let (_, c) = num(ui, ws, "Pitch", &mut p, Unit::Mm, None, Some("How far the profile climbs along the axis each turn — a thread's pitch, a spring's from one coil to the next. Negative climbs the other way along the axis, which Reverse direction does too."), r.exprs.get("pitch"), None);
        if c && p != 0.0 {
            r.pitch = Some(p);
            changed = true;
        }
        let mut turns = (r.angle_deg / 360.0 * 1e4).round() / 1e4;
        let (_, c) = num(ui, ws, "Turns", &mut turns, Unit::Turns, None, Some("How many times the profile goes round — the angle over 360°. A fraction is fine."), r.exprs.get("turns"), None);
        if c {
            r.angle_deg = revolve_angle(turns * 360.0, pitch);
            changed = true;
        }
        let mut height = (pitch.abs() * (r.angle_deg / 360.0) * 1e4).round() / 1e4;
        let (_, c) = num(ui, ws, "Height", &mut height, Unit::Mm, None, Some("How far the profile climbs along the axis from start to end — the turns times the pitch. Typed, it sets the turns."), r.exprs.get("height"), None);
        if c && pitch != 0.0 {
            r.angle_deg = revolve_angle((height.abs() / pitch.abs()) * 360.0, pitch);
            changed = true;
        }
        changed |= ui.checkbox(&mut r.left_handed, "Left-handed").changed();
        if ui.button("⇅ Reverse direction").on_hover_text("Climb the other way along the axis — the hand stays as it is").clicked() {
            r.pitch = Some(-pitch);
            changed = true;
        }
    }
    // The result.
    let can_cut = super::extrude::can_cut(ws);
    ui.horizontal(|ui| {
        ui.label("Result");
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            for (o, tip) in [
                (BodyOp::New, "A body of its own, whatever it overlaps — a tool to cut the scan with in Mesh Tools"),
                (BodyOp::Cut, "Taken out of every body it goes into; a body it parts is a body a piece"),
                (BodyOp::Join, "Made one with every body it touches; a body of its own where it touches none, one body of all where it bridges several"),
            ] {
                let enabled = o != BodyOp::Cut || can_cut || r.op == BodyOp::Cut;
                if ui.add_enabled(enabled, egui::Button::selectable(r.op == o, o.name())).on_hover_text(tip).clicked() && r.op != o {
                    r.op = o;
                    changed = true;
                }
            }
        });
    });
    if let Some((side, v)) = fields::grip_field(ui.ctx(), |side| (side == GripSide::Turn).then_some(r.angle_deg)) {
        if side == GripSide::Turn {
            r.angle_deg = revolve_angle(v, r.pitch.unwrap_or(0.0));
            unlink(r, &["angleDeg", "turns", "height"]);
            changed = true;
        }
    }
    changed
}

// ---- the part ----

/// The line of a sketch shown on the part under a screen point, within reach: the box's sketch
/// first, then any other shown — as the part draws them, each a line to click.
fn line_under(ws: &ReverseWorkspace, vp: Viewport, p: Pos2, first: FeatureId) -> Option<(FeatureId, u32)> {
    let mut order: Vec<FeatureId> = vec![first];
    order.extend(ws.part.features.iter().rev().filter(|f| f.visible() && !f.suppressed() && f.id != first && matches!(f.kind, FeatureKind::Sketch(_))).map(|f| f.id));
    for sketch in order {
        let (Some(frame), Some(s)) = (ws.sketch_frame(sketch), ws.part.sketch(sketch)) else { continue };
        let mut best: Option<(u32, f32)> = None;
        for e in &s.sketch.entities {
            let Shape::Line { a, b } = e.shape else { continue };
            let (Some(pa), Some(pb)) = (ws.project(vp, frame.to_world(a)), ws.project(vp, frame.to_world(b))) else { continue };
            let ab = pb - pa;
            let t = if ab.length_sq() > 0.0 { ((p - pa).dot(ab) / ab.length_sq()).clamp(0.0, 1.0) } else { 0.0 };
            let d = (pa + ab * t - p).length();
            if d <= 6.0 && best.is_none_or(|x| d < x.1) {
                best = Some((e.id, d));
            }
        }
        if let Some((id, _)) = best {
            return Some((sketch, id));
        }
    }
    None
}

/// A line of a sketch clicked for the axis (ScanRuler's `setRevolve({ sketchId, axis })`): a line
/// of another sketch takes that sketch, with its own regions as a new box takes them.
pub(crate) fn take_line(ws: &ReverseWorkspace, r: &mut RevolveFeature, sketch: FeatureId, entity: u32) {
    if sketch != r.sketch_id {
        r.sketch_id = sketch;
        r.regions = auto_regions(&regions_of(ws, sketch));
        r.region_pick = None;
    }
    r.axis = AxisRef::SketchLine { entity };
}

pub(crate) fn pick(ws: &mut ReverseWorkspace, vp: Viewport, p: Pos2, clicked: bool, r: &mut RevolveFeature, _pick: &mut PickWhat) -> bool {
    // A line of a sketch takes the click as the axis.
    let line = line_under(ws, vp, p, r.sketch_id);
    with_box(|b| b.hover_line = line);
    if let Some((sketch, entity)) = line {
        ws.view.hover_region = None;
        if clicked {
            take_line(ws, r, sketch, entity);
            return true;
        }
        return false;
    }
    // A region of a sketch, through the part — of another sketch, that sketch taken with it.
    let region = region_under(ws, vp, p, r.sketch_id);
    ws.view.hover_region = region.as_ref().filter(|x| x.0 == r.sketch_id).map(|x| x.1.clone());
    if let (true, Some((of, key))) = (clicked, region) {
        toggle_region(&mut r.sketch_id, &mut r.regions, &mut r.region_pick, of, key);
        return true;
    }
    false
}

pub(crate) fn paint(ws: &ReverseWorkspace, painter: &egui::Painter, vp: Viewport, r: &RevolveFeature) {
    fields::paint_region_faces(ws, painter, vp, r.sketch_id, &r.regions);
    fields::paint_box_hint(ws, painter, vp);
    // The sketch line under the pointer, white and a little wider.
    if let Some((frame, s, id)) = with_box(|b| b.hover_line).and_then(|(sketch, id)| Some((ws.sketch_frame(sketch)?, ws.part.sketch(sketch)?, id))) {
        if let Some(Shape::Line { a, b }) = s.sketch.entity(id).map(|e| e.shape.clone()) {
            if let (Some(pa), Some(pb)) = (ws.project(vp, frame.to_world(a)), ws.project(vp, frame.to_world(b))) {
                painter.line_segment([pa, pb], Stroke::new(3.0, egui::Color32::WHITE));
            }
        }
    }
    let grips = grips_of(ws, r);
    fields::paint_grips(ws, painter, vp, &grips, r.op == BodyOp::Cut);
}

/// The ring the revolve wears about its axis, once it has one.
pub(crate) fn grips_of(ws: &ReverseWorkspace, r: &RevolveFeature) -> Vec<FeatureGrip> {
    let (Some(frame), Some((o, d))) = (ws.sketch_frame(r.sketch_id), axis_in_part(ws, r.sketch_id, &r.axis)) else { return Vec::new() };
    let regions = regions_of(ws, r.sketch_id);
    let picked: Vec<&Region> = regions.iter().filter(|x| r.regions.contains(&x.key)).collect();
    revolve_grip(&frame, &picked, o, d).into_iter().collect()
}

/// A drag of the ring: the angle it began with turned by the degrees wound since, round and
/// round for a coil.
pub(crate) fn apply_drag(r: &mut RevolveFeature, drag: &GripDrag, delta_deg: f64) {
    r.angle_deg = drag_revolve_angle(drag.angle_deg, delta_deg, revolve_max_deg(r.pitch));
    unlink(r, &["angleDeg", "turns", "height"]);
}

/// The pointer on the ring: lit under it, dragged round to turn the angle — a build asked for
/// one at a time — and its field taking typing once let go.
pub(crate) fn grip_input(ws: &mut ReverseWorkspace, vp: Viewport, pos: Option<Pos2>, pressed: bool, down: bool, released: bool, r: &mut RevolveFeature) -> Option<bool> {
    let grips = grips_of(ws, r);
    if let Some(mut drag) = with_box(|b| b.drag.take()) {
        if down && !released {
            if let Some(delta) = pos.and_then(|p| fields::drag_delta(ws, vp, p, &mut drag)) {
                let before = r.angle_deg;
                apply_drag(r, &drag, delta);
                if r.angle_deg != before {
                    with_box(|b| b.owed = true);
                }
            }
            with_box(|b| b.drag = Some(drag));
            return Some(with_box(|b| {
                let go = b.owed && !ws.building();
                if go {
                    b.owed = false;
                }
                go
            }));
        }
        let v = r.angle_deg;
        with_box(|b| {
            b.owed = false;
            b.field = Some(GripField { side: GripSide::Turn, text: fields::field_words(v, GripSide::Turn).0, typing: true, error: None });
        });
        return Some(true);
    }
    let screen = fields::on_screen(ws, vp, &grips);
    let hover = pos.and_then(|p| fields::grip_under(&screen, p));
    with_box(|b| b.hover_grip = hover.map(|i| grips[i].side));
    let (i, p) = (hover?, pos?);
    if pressed {
        let grip = grips[i];
        let t0 = fields::drag_start(ws, vp, p, &grip)?;
        let drag = GripDrag { side: grip.side, grip, t0, last: t0, wound: 0.0, extent: None, taper_deg: 0.0, taper_run: 0.0, angle_deg: r.angle_deg };
        with_box(|b| {
            b.field = Some(GripField { side: grip.side, text: String::new(), typing: false, error: None });
            b.drag = Some(drag);
        });
    }
    Some(false)
}
