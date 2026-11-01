// SPDX-License-Identifier: AGPL-3.0-only
//! The Extrude feature's box: the sketch and its profiles — or a flat face of a body pulled as it
//! stands — how far the extrusion reaches (a distance, a plane, a face, a body, the next face, the
//! last, between two planes, symmetric), its taper, the way it is swept, and what it does with the
//! bodies it meets; the preview's volume, the ⓘ cards, and the arrows on the part that drag its
//! numbers. A click on a region of the sketch picks it; a click elsewhere on the part puts an end
//! at that height, or — reaching to a face — takes the face. Ported from ScanRuler
//! (`plugins/reverse/ui/ExtrudeEditor.tsx`, `ui/DirectionField.tsx`, `ui/PlaneSelect.tsx`,
//! `state/reSolidSlice.ts` the extrude, `state/reGripSlice.ts`), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only.

use egui::Pos2;
use glam::{DVec2, DVec3};
use mm_reverse::extrude_draft::{auto_regions, extent_lacks, extrude_ready, extrude_to_point, face_extrude_op, no_plane, plane_chosen, NO_BODY};
use mm_reverse::extrude_grips::{drag_extrude_extent, drag_taper, extrude_grips, extrude_grips_at, taper_grip, taper_span, FeatureGrip, GripSide, MAX_TAPER_DEG};
use mm_reverse::frame::{SectionFrame, WorldAxis};
use mm_reverse::model::{AxisRef, BodyId, BodyOp, ElementGeometry, ExtrudeExtent, ExtrudeFeature, FaceProfile, FaceRef, FeatureId, FeatureKind, PlaneRef, MIN_SWEEP_RAD};
use mm_reverse::nominal::NominalKind;
use mm_reverse::regions::{region_at, Region};
use mm_reverse::topo::{describe_face, FaceKind};

use super::ext_fields::{self as fields, hint, info_dot, num, preview_line, profiles_field, ref_picker, with_box, BoxState, GripDrag, GripField, Hint, Unit};
use super::{body_name, paint_faces};
use crate::reverse::{PickWhat, ReverseWorkspace};
use crate::tools::Viewport;

pub(crate) const FIRST_PICK: PickWhat = PickWhat::Regions;

/// Extrude stands by the sketch in the panel, not among these buttons.
#[allow(dead_code)]
pub(crate) fn new_feature(_ws: &ReverseWorkspace) -> Result<FeatureKind, String> {
    Err("Extrude starts from a sketch.".into())
}

// ---- opening ----

/// Open an Extrude box on a sketch — the last one, unless one is named — with the regions given,
/// or its one region where it has exactly one: Distance from 0 to 10 mm, no taper, Join.
pub(crate) fn open(ws: &mut ReverseWorkspace, sketch: Option<FeatureId>, regions: Option<Vec<String>>) {
    let Some(sketch) = sketch.or_else(|| ws.last_sketch()) else {
        ws.status = "Extrude takes a sketch: make one first (New sketch).".into();
        return;
    };
    let regions = regions.unwrap_or_else(|| auto_regions(&regions_of(ws, sketch)));
    fields::reset_box();
    ws.open_feature(FeatureKind::Extrude(ExtrudeFeature::new(sketch, regions, ExtrudeExtent::Distance { from: 0.0, to: 10.0 }, BodyOp::Join)), None);
    ws.status = "Click the part to put the extrude's end at that height, or type the distances.".into();
}

/// Open an Extrude box pulling a flat face of a body as it stands (ScanRuler's `extrudeFace`):
/// no sketch is made; out of the body it joins, into it it cuts, until a result is chosen by
/// hand. The face is read off the body as the box's picks see it.
pub(crate) fn open_face(ws: &mut ReverseWorkspace, body: BodyId, face: usize) -> Result<(), String> {
    let print = ws.pick_view(body).and_then(|b| b.topology.faces.get(face)).map(|f| f.fingerprint).ok_or("There is no such face to pull.")?;
    if print.kind != FaceKind::Plane {
        return Err("Only a flat face can be pulled.".into());
    }
    let mut e = ExtrudeFeature::new(0, Vec::new(), ExtrudeExtent::Distance { from: 0.0, to: 5.0 }, BodyOp::Join);
    e.face = Some(FaceProfile { body, face: print });
    fields::reset_box();
    ws.open_feature(FeatureKind::Extrude(e), None);
    fields::box_for(ws, "extrude");
    with_box(|b| b.op_auto = true);
    ws.status = "Pull this face: drag the arrow out to add to the body, in to cut into it.".into();
    Ok(())
}

/// A sketch's closed regions as last built, or read off the sketch as it is.
pub(crate) fn regions_of(ws: &ReverseWorkspace, sketch: FeatureId) -> Vec<Region> {
    match ws.built.sketches.get(&sketch) {
        Some(b) => b.regions.clone(),
        None => ws.part.sketch(sketch).map_or(Vec::new(), |s| mm_reverse::regions::find_regions(&s.sketch)),
    }
}

// ---- the box ----

/// The box's title, its Create key, and whether it is ready.
pub(crate) fn box_words(ws: &ReverseWorkspace, editing: Option<FeatureId>, e: &ExtrudeFeature) -> super::BoxWords {
    let name = editing.and_then(|id| ws.part.feature(id)).map(|f| f.name.clone());
    super::BoxWords {
        title: name.as_ref().map_or_else(|| "New extrude".to_string(), |n| format!("Edit {n}")),
        create: if name.is_some() { "Save changes".into() } else { "Create extrude".into() },
        tip: "Enter or a middle click does the same",
        ready: extrude_ready(e).is_ok(),
        own_preview: true,
        create_now: with_box(|b| std::mem::take(&mut b.create_now)),
    }
}

const EXTENTS: [(&str, &str); 8] = [
    ("distance", "Distance"),
    ("toPlane", "To a plane"),
    ("toFace", "To a face of a body"),
    ("toBody", "To a body"),
    ("toNext", "To next"),
    ("toLast", "To last / through all"),
    ("betweenPlanes", "Between two planes"),
    ("symmetric", "Symmetric"),
];

fn extent_key(e: &ExtrudeExtent) -> &'static str {
    match e {
        ExtrudeExtent::Distance { .. } => "distance",
        ExtrudeExtent::ToPlane { .. } => "toPlane",
        ExtrudeExtent::ToFace { .. } => "toFace",
        ExtrudeExtent::ToBody { .. } => "toBody",
        ExtrudeExtent::ToNext { .. } => "toNext",
        ExtrudeExtent::ToLast { .. } => "toLast",
        ExtrudeExtent::BetweenPlanes { .. } => "betweenPlanes",
        ExtrudeExtent::Symmetric { .. } => "symmetric",
    }
}

/// A face fingerprint that names no face: what To a face holds until one is clicked.
fn no_face() -> FaceRef {
    FaceRef { kind: FaceKind::Plane, centroid: DVec3::ZERO, dir: DVec3::Z, area: 0.0, radius: None, center: None, origin: -1 }
}

/// The extent a kind switched to starts with: a distance from 0 to 10 mm, a symmetric 10 mm, the
/// rest with nothing chosen.
pub(crate) fn extent_of_kind(kind: &str) -> ExtrudeExtent {
    match kind {
        "symmetric" => ExtrudeExtent::Symmetric { total: 10.0 },
        "toPlane" => ExtrudeExtent::ToPlane { from: 0.0, plane: no_plane() },
        "toFace" => ExtrudeExtent::ToFace { from: 0.0, body: NO_BODY, face: no_face() },
        "toBody" => ExtrudeExtent::ToBody { from: 0.0, body: NO_BODY, reverse: false },
        "toNext" => ExtrudeExtent::ToNext { from: 0.0, reverse: false },
        "toLast" => ExtrudeExtent::ToLast { from: 0.0, reverse: false },
        "betweenPlanes" => ExtrudeExtent::BetweenPlanes { a: no_plane(), b: no_plane() },
        _ => ExtrudeExtent::Distance { from: 0.0, to: 10.0 },
    }
}

/// Reverse direction (ScanRuler's `flipExtrude`): to next, to last and to a body look the other
/// way; a distance's From and To change sign — and a pulled face's result follows.
pub(crate) fn flip(e: &mut ExtrudeFeature) {
    let neg = |v: f64| if v == 0.0 { 0.0 } else { -v };
    match &mut e.extent {
        ExtrudeExtent::ToNext { reverse, .. } | ExtrudeExtent::ToLast { reverse, .. } | ExtrudeExtent::ToBody { reverse, .. } => *reverse = !*reverse,
        ExtrudeExtent::Distance { from, to } => {
            (*from, *to) = (neg(*from), neg(*to));
            follow_face(e);
        }
        _ => {}
    }
}

/// A pulled face's result while it follows the way the face is pulled.
fn follow_face(e: &mut ExtrudeFeature) {
    if e.face.is_some() && with_box(|b| b.op_auto) {
        e.op = face_extrude_op(&e.extent);
    }
}

/// The result chosen (ScanRuler's `setExtrudeOp`): by hand it stops following a pulled face's
/// way, and going to or from a cut turns the extrude over, so a cut goes into what it meets.
pub(crate) fn set_op(e: &mut ExtrudeFeature, op: BodyOp) {
    if e.op == op {
        return;
    }
    let was_cut = e.op == BodyOp::Cut;
    e.op = op;
    with_box(|b| b.op_auto = false);
    if (op == BodyOp::Cut) != was_cut {
        flip(e);
    }
}

/// The plane the extrude's offsets count from: the sketch's, or the pulled face's.
fn base_plane(ws: &ReverseWorkspace, e: &ExtrudeFeature) -> Option<(DVec3, DVec3)> {
    match &e.face {
        Some(f) => Some((f.face.centroid, f.face.dir.normalize_or_zero())),
        None => ws.sketch_frame(e.sketch_id).map(|f| (f.origin, f.normal)),
    }
}

/// The planes an extrusion can reach to: the sections, the measured planes, and the coordinate
/// planes at zero and, with a scan, through its centre — by key, label and group.
fn plane_options(ws: &ReverseWorkspace) -> Vec<(String, String, &'static str, PlaneRef)> {
    let mut out = Vec::new();
    for el in &ws.part.elements {
        if let ElementGeometry::Section { .. } = el.geometry {
            out.push((format!("section:{}", el.id), el.name.clone(), "Sections", PlaneRef::Section { id: el.id, offset: None, up: None }));
        }
    }
    for el in &ws.part.elements {
        if let ElementGeometry::Plane { .. } = el.geometry {
            out.push((format!("plane:{}", el.id), el.name.clone(), "Planes", PlaneRef::Plane { id: el.id, offset: None, up: None }));
        }
    }
    // In ScanRuler's order (`WORLD_AXES`): the YZ, XZ and XY planes.
    for axis in WorldAxis::ALL {
        out.push((world_key(axis, 0.0), axis.plane_name().to_string(), "Coordinate planes", PlaneRef::world(axis, 0.0)));
    }
    if let Some(c) = ws.scan_centre() {
        for axis in WorldAxis::ALL {
            let off = c.dot(axis.dir());
            out.push((world_key(axis, off), format!("{} through the centre", axis.plane_name()), "Coordinate planes", PlaneRef::world(axis, off)));
        }
    }
    out
}

fn world_key(axis: WorldAxis, offset: f64) -> String {
    format!("world:{axis:?}:{}", (offset * 1000.0).round() / 1000.0)
}

fn plane_key(p: &PlaneRef) -> Option<String> {
    if !plane_chosen(p) {
        return None;
    }
    Some(match p {
        PlaneRef::World { axis, offset, .. } => world_key(*axis, *offset),
        PlaneRef::Section { id, .. } => format!("section:{id}"),
        PlaneRef::Plane { id, .. } => format!("plane:{id}"),
        _ => "other".into(),
    })
}

/// A plane as a hint draws it.
fn plane_hint(ws: &ReverseWorkspace, p: &PlaneRef, name: &str) -> Option<Hint> {
    let f = ws.plane_frame(p)?;
    Some(Hint::Plane { origin: f.origin, normal: f.normal, name: name.to_string() })
}

/// A plane picker: the list of planes, the one under the pointer drawn on the part, the chosen
/// one too while the box is open. Returns whether the plane changed.
fn plane_select(ui: &mut egui::Ui, ws: &ReverseWorkspace, salt: &str, plane: &mut PlaneRef) -> bool {
    let mut options = plane_options(ws);
    // A plane given otherwise (a script, a project) is listed as it reads.
    if let Some(k) = plane_key(plane).filter(|k| !options.iter().any(|o| &o.0 == k)) {
        options.push((k, mm_reverse::build::describe_plane_ref(plane, &ws.part), "Other planes", plane.clone()));
    }
    let listed: Vec<(String, String, &'static str)> = options.iter().map(|o| (o.0.clone(), o.1.clone(), o.2)).collect();
    let chosen = plane_key(plane);
    let (picked, hovered) = ref_picker(ui, salt, chosen.as_deref(), "Choose the plane…", &listed);
    let hover = hovered.and_then(|k| options.iter().find(|o| o.0 == k)).and_then(|o| plane_hint(ws, &o.3, &o.1));
    let shown = chosen.as_ref().and_then(|k| options.iter().find(|o| &o.0 == k)).and_then(|o| plane_hint(ws, &o.3, &o.1));
    with_box(|b| {
        if hover.is_some() || b.hover_hint.as_ref().is_some_and(|h| matches!(h, Hint::Plane { .. })) {
            b.hover_hint = hover;
        }
        b.chosen_hints.extend(shown);
    });
    match picked.and_then(|k| options.into_iter().find(|o| o.0 == k)) {
        Some(o) => {
            *plane = o.3;
            true
        }
        None => false,
    }
}

/// Element axes and the coordinate axes the extrude can be swept along, by key, label and way.
fn direction_offers(ws: &ReverseWorkspace) -> Vec<(String, String, DVec3)> {
    let mut out = Vec::new();
    for el in &ws.part.elements {
        if matches!(el.geometry, ElementGeometry::Line { .. } | ElementGeometry::Cylinder { .. } | ElementGeometry::Cone { .. }) {
            if let Some((_, d)) = mm_reverse::build::resolve_axis(&AxisRef::Element { id: el.id }, None, &ws.part) {
                out.push((format!("element:{}", el.id), el.name.clone(), d.normalize_or_zero()));
            }
        }
    }
    for axis in WorldAxis::ALL {
        out.push((format!("world:{axis:?}"), format!("{axis:?} axis"), axis.dir()));
    }
    out
}

fn same_way(a: DVec3, b: DVec3) -> bool {
    (a.dot(b).abs() - 1.0).abs() < 1e-9
}

/// The way the profile is swept: the sketch plane's normal, an axis read once and kept as a
/// direction turned to the plane's front, or three numbers typed — refused where it lies in the
/// plane. Returns whether it changed.
fn direction_field(ui: &mut egui::Ui, ws: &ReverseWorkspace, e: &mut ExtrudeFeature) -> bool {
    let normal = base_plane(ws, e).map(|p| p.1);
    let offers = direction_offers(ws);
    let along = |d: DVec3| normal.is_some_and(|n| d.dot(n).abs() >= MIN_SWEEP_RAD.sin());
    let facing = |d: DVec3| if normal.is_some_and(|n| d.dot(n) < 0.0) { -d } else { d };
    let typing = with_box(|b| b.typing_direction);
    let chosen = match e.direction {
        Some(d) => offers.iter().find(|o| same_way(o.2, d)).map_or("typed".to_string(), |o| o.0.clone()),
        None if typing => "typed".to_string(),
        None => "normal".to_string(),
    };
    let lean = match (e.direction, normal) {
        (Some(d), Some(n)) => d.dot(n).abs().min(1.0).acos().to_degrees(),
        _ => 0.0,
    };
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label("Direction");
        info_dot(
            ui,
            ws,
            "direction",
            "The way the profile is swept",
            &[
                "Along the sketch plane's normal, as an extrude starts — or askew of it: along a measured line or axis, a coordinate axis, or three numbers typed. The ends stay in planes parallel to the sketch's, the distances count along the direction and a taper leans the sides about it — a leaning boss on a level sketch, with no tilted sketch plane to make.",
                "An axis is read once and kept as a direction, turned to the plane's front; one lying in the sketch plane sweeps nothing and is not offered.",
            ],
        );
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let label = match chosen.as_str() {
                "normal" => "The sketch plane's normal".to_string(),
                "typed" => "Typed…".to_string(),
                k => offers.iter().find(|o| o.0 == k).map_or(k.to_string(), |o| o.1.clone()),
            };
            egui::ComboBox::from_id_salt("re_extrude_direction").selected_text(label).width(170.0).show_ui(ui, |ui| {
                if ui.selectable_label(chosen == "normal", "The sketch plane's normal").clicked() {
                    e.direction = None;
                    with_box(|b| b.typing_direction = false);
                    changed = true;
                }
                for (key, label, d) in &offers {
                    if ui.add_enabled(along(*d), egui::Button::selectable(chosen == *key, label.as_str())).clicked() {
                        e.direction = Some(facing(*d));
                        with_box(|b| b.typing_direction = false);
                        changed = true;
                    }
                }
                if ui.selectable_label(chosen == "typed", "Typed…").clicked() {
                    with_box(|b| b.typing_direction = true);
                }
            });
        });
    });
    if chosen == "typed" {
        ui.horizontal(|ui| {
            ui.label(if lean > 0.0 { format!("{}° off the normal", (lean * 10.0).round() / 10.0) } else { "x, y, z".to_string() });
            let shown = e.direction.or(normal).unwrap_or(DVec3::Z);
            let mut v = [shown.x, shown.y, shown.z];
            for c in &mut v {
                *c = (*c * 1e4).round() / 1e4;
            }
            let mut moved = false;
            for c in &mut v {
                moved |= ui.add(egui::DragValue::new(c).speed(0.01).max_decimals(4)).changed();
            }
            if moved {
                if let Some(next) = DVec3::from(v).try_normalize().filter(|d| along(*d)) {
                    e.direction = Some(facing(next));
                    changed = true;
                }
            }
        });
    } else if e.direction.is_some() {
        hint(ui, &format!("{}° off the sketch plane's normal.", (lean * 10.0).round() / 10.0));
    }
    changed
}

/// Whether there is a body for a cut to go into: a solid one made before this feature — not the
/// box's own ghost, nor a sheet or a fitted surface, which have no inside to take from.
pub(crate) fn can_cut(ws: &ReverseWorkspace) -> bool {
    ws.pick_set().iter().filter(|b| Some(b.id.maker()) != ws.preview_feature).any(|b| ws.built.body(b.id).is_none_or(|x| !x.surface))
}

/// The result: Join, Cut or New body, each with what it does.
fn op_field(ui: &mut egui::Ui, ws: &ReverseWorkspace, e: &mut ExtrudeFeature) -> bool {
    let can_cut = can_cut(ws);
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label("Result");
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let mut op = e.op;
            for (o, tip) in [
                (BodyOp::New, "A body of its own, whatever it overlaps — a tool to cut the scan with in Mesh Tools"),
                (BodyOp::Cut, "Taken out of every body it goes into; a body it parts is a body a piece"),
                (BodyOp::Join, "Made one with every body it touches; a body of its own where it touches none, one body of all where it bridges several"),
            ] {
                let enabled = o != BodyOp::Cut || can_cut || e.op == BodyOp::Cut;
                if ui.add_enabled(enabled, egui::Button::selectable(op == o, o.name())).on_hover_text(tip).clicked() {
                    op = o;
                }
            }
            if op != e.op {
                set_op(e, op);
                changed = true;
            }
        });
    });
    changed
}

/// Lets a parameter go from the slots a typed or dragged number stands in for.
fn unlink(e: &mut ExtrudeFeature, paths: &[&str]) {
    for p in paths {
        e.exprs.remove(p);
    }
}

fn from_field(ui: &mut egui::Ui, ws: &ReverseWorkspace, e: &mut ExtrudeFeature, hint_text: &str) -> bool {
    let expr = e.exprs.get("extent.from").map(str::to_string);
    let mut changed = false;
    if let ExtrudeExtent::ToPlane { from, .. } | ExtrudeExtent::ToFace { from, .. } | ExtrudeExtent::ToBody { from, .. } | ExtrudeExtent::ToNext { from, .. } | ExtrudeExtent::ToLast { from, .. } | ExtrudeExtent::Distance { from, .. } = &mut e.extent {
        let (_, c) = num(ui, ws, "From", from, Unit::Mm, Some(NominalKind::Length), Some(hint_text), expr.as_deref(), None);
        changed = c;
    }
    if changed && expr.is_some() {
        unlink(e, &["extent.from"]);
    }
    changed
}

pub(crate) fn ui(ws: &mut ReverseWorkspace, ui: &mut egui::Ui, e: &mut ExtrudeFeature, pick: &mut PickWhat) -> bool {
    fields::box_for(ws, "extrude");
    // The chosen planes are told again by their pickers below, this frame.
    with_box(|b| b.chosen_hints.clear());
    let mut changed = false;
    // Saved from its box, a feature joins or cuts every body it meets: the one body it was saved
    // to work on goes (ScanRuler's `withoutTarget`; the box's draft holds none, nor its ghost).
    if e.target.take().is_some() {
        changed = true;
    }
    // The profile: a face pulled, or a sketch's regions.
    match &e.face {
        Some(f) => {
            ui.horizontal(|ui| {
                ui.label("Profile");
                info_dot(ui, ws, "face", "A face pulled", &["The face itself is the profile: it is found again on the body whenever this extrude is built, and pulled as it stands then — an earlier feature that reshapes the face reshapes what is pulled from it. Nothing is copied into a sketch."]);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.strong(format!("{} of {}", describe_face(&f.face), body_name(ws, f.body)));
                });
            });
        }
        None => {
            let sketches: Vec<(FeatureId, String)> = ws.part.features.iter().filter(|f| matches!(f.kind, FeatureKind::Sketch(_))).map(|f| (f.id, f.name.clone())).collect();
            ui.horizontal(|ui| {
                ui.label("Sketch");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let name = sketches.iter().find(|s| s.0 == e.sketch_id).map_or_else(|| "Select…".to_string(), |s| s.1.clone());
                    egui::ComboBox::from_id_salt("re_extrude_sketch").selected_text(name).width(170.0).show_ui(ui, |ui| {
                        for (id, n) in &sketches {
                            if ui.selectable_label(e.sketch_id == *id, n).clicked() && e.sketch_id != *id {
                                e.sketch_id = *id;
                                e.regions = auto_regions(&regions_of(ws, *id));
                                e.region_pick = None;
                                changed = true;
                            }
                        }
                    });
                });
            });
            changed |= profiles_field(ui, ws, e.sketch_id, &mut e.regions, &mut e.region_pick);
        }
    }
    preview_line(ui, ws, extrude_ready(e).is_ok());
    // Extent.
    ui.horizontal(|ui| {
        ui.label("Extent");
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let now = extent_key(&e.extent);
            let label = EXTENTS.iter().find(|x| x.0 == now).map_or("", |x| x.1);
            egui::ComboBox::from_id_salt("re_extrude_extent").selected_text(label).width(170.0).show_ui(ui, |ui| {
                for (key, label) in EXTENTS {
                    if ui.selectable_label(now == key, label).clicked() && now != key {
                        e.extent = extent_of_kind(key);
                        changed = true;
                    }
                }
            });
        });
    });
    // Reverse direction.
    let (flips, tip) = match &e.extent {
        ExtrudeExtent::Distance { .. } => (true, "Extrude the other way along the sketch plane’s normal — From and To change sign, and the ghost turns over"),
        ExtrudeExtent::ToNext { .. } | ExtrudeExtent::ToLast { .. } | ExtrudeExtent::ToBody { .. } => (true, "Look for the body the other way along the sketch plane’s normal"),
        ExtrudeExtent::Symmetric { .. } => (false, "A symmetric extrude has no direction to reverse"),
        _ => (false, "An extrude limited by a plane reaches the way the plane lies"),
    };
    let r = ui.add_enabled(flips, egui::Button::new("⇅ Reverse direction"));
    let r = if flips { r.on_hover_text(tip) } else { r.on_disabled_hover_text(tip) };
    if r.clicked() {
        flip(e);
        changed = true;
    }
    *pick = PickWhat::Regions;
    let op = e.op;
    match &mut e.extent {
        ExtrudeExtent::Distance { from, to } => {
            let to_expr = e.exprs.get("extent.to").map(str::to_string);
            let (to_resp, c) = num(ui, ws, "To", to, Unit::Mm, Some(NominalKind::Length), Some("How far along the sketch plane's normal the body ends — in front of the plane. The arrow on that face drags it; a click on the part in front of the plane sets it."), to_expr.as_deref(), None);
            changed |= c;
            // To takes the keys as the box opens, its number selected to be typed over, and again
            // once a profile is picked — never from another field the operator is in.
            let at = (ws.preview_feature, e.regions.len());
            if with_box(|b| b.focused_for != Some(at)) && ui.ctx().memory(|m| m.focused().is_none()) {
                to_resp.request_focus();
                with_box(|b| b.focused_for = Some(at));
            }
            if to_resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                with_box(|b| b.create_now = true);
            }
            let from_expr = e.exprs.get("extent.from").map(str::to_string);
            let (_, c) = num(ui, ws, "From", from, Unit::Mm, Some(NominalKind::Length), Some("How far along the normal the body starts — 0 is the plane itself, negative is behind it. The arrow on that face drags it; a click on the part behind the plane sets it."), from_expr.as_deref(), None);
            changed |= c;
            ui.horizontal_wrapped(|ui| {
                hint(ui, "Or click the part: in front of the plane sets To, behind it From.");
                info_dot(
                    ui,
                    ws,
                    "point",
                    "Up to a point on the part",
                    &[
                        "Click the scan, a body or a body's corner in the viewport: a point in front of the sketch plane puts To at its height over the plane, a point behind it puts From there. A click on the part's top and one on its bottom make the body that spans them, in either order. For a part lying wholly behind the plane, put To at 0 by hand — a click behind the plane never moves it.",
                        "The height is read off the point once and kept as a number, which the field and the arrow go on changing. A sketch region under the cursor still takes the click as a profile — turn the part so the point is clear of it.",
                    ],
                );
            });
            *pick = PickWhat::Height;
        }
        ExtrudeExtent::Symmetric { total } => {
            let expr = e.exprs.get("extent.total").map(str::to_string);
            let (_, c) = num(ui, ws, "Total", total, Unit::Mm, Some(NominalKind::Length), Some("The whole thickness, half on each side of the sketch plane; either arrow drags it"), expr.as_deref(), Some(0.001..=1e5));
            changed |= c;
            hint(ui, "Or click a point on the part: each half reaches it.");
            *pick = PickWhat::Height;
        }
        _ => {}
    }
    match e.extent.clone() {
        ExtrudeExtent::ToPlane { plane, .. } => {
            changed |= from_field(ui, ws, e, "How far along the normal the body starts");
            let mut p = plane;
            ui.horizontal(|ui| {
                ui.label("Up to");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if plane_select(ui, ws, "re_extrude_plane", &mut p) {
                        if let ExtrudeExtent::ToPlane { plane, .. } = &mut e.extent {
                            *plane = p.clone();
                        }
                        changed = true;
                    }
                });
            });
        }
        ExtrudeExtent::ToFace { body, face, .. } => {
            changed |= from_field(ui, ws, e, "How far along the normal the body starts");
            ui.horizontal(|ui| {
                ui.label("Up to");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if body == NO_BODY {
                        ui.weak("Click a face of a body");
                    } else {
                        ui.strong(describe_face(&face));
                    }
                });
            });
            *pick = PickWhat::Faces;
        }
        ExtrudeExtent::ToBody { body, .. } => {
            changed |= from_field(ui, ws, e, "How far along the normal the body starts");
            let own = ws.preview_feature;
            let bodies: Vec<(BodyId, String)> = ws.pick_set().iter().filter(|b| Some(b.id.maker()) != own).map(|b| (b.id, b.name.clone())).collect();
            ui.horizontal(|ui| {
                ui.label("Up to");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let shown = if body == NO_BODY { "Choose the body…".to_string() } else { body_name(ws, body) };
                    egui::ComboBox::from_id_salt("re_extrude_to_body").selected_text(shown).width(170.0).show_ui(ui, |ui| {
                        for (id, name) in &bodies {
                            if ui.selectable_label(body == *id, name).clicked() {
                                if let ExtrudeExtent::ToBody { body, .. } = &mut e.extent {
                                    *body = *id;
                                }
                                changed = true;
                            }
                        }
                    });
                });
            });
            hint(ui, "Runs until the profile meets the body’s surface — a solid’s skin or a sheet — the way the body lies; the body has to lie under the whole profile.");
        }
        ExtrudeExtent::ToNext { .. } | ExtrudeExtent::ToLast { .. } => {
            changed |= from_field(ui, ws, e, "How far along the normal the body starts");
            let next = matches!(e.extent, ExtrudeExtent::ToNext { .. });
            let mut words = match (next, op == BodyOp::Cut) {
                (true, true) => "Cuts through the first wall of the body the profile meets, and stops where it comes out.".to_string(),
                (true, false) => "Runs until the profile meets the body — the whole profile has to land on it. Reverse direction looks the other way.".to_string(),
                (false, true) => "Cuts through all of the body that way.".to_string(),
                (false, false) => "Runs to the far end of the body that way.".to_string(),
            };
            if op == BodyOp::New {
                words.push_str(" Measured against every body there is.");
            }
            hint(ui, &words);
        }
        ExtrudeExtent::BetweenPlanes { a, b } => {
            let (mut pa, mut pb) = (a, b);
            for (label, salt, p) in [("From plane", "re_extrude_plane_a", &mut pa), ("To plane", "re_extrude_plane_b", &mut pb)] {
                ui.horizontal(|ui| {
                    ui.label(label);
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        changed |= plane_select(ui, ws, salt, p);
                    });
                });
            }
            e.extent = ExtrudeExtent::BetweenPlanes { a: pa, b: pb };
        }
        _ => {}
    }
    // The taper, shown the way Fusion shows it: positive leans the sides out. The feature keeps
    // a mould's draft, positive leaning in — the same number turned over.
    let mut taper = if e.draft_deg == 0.0 { 0.0 } else { -e.draft_deg };
    let taper_expr = e.exprs.get("taper").map(str::to_string);
    let (_, c) = num(
        ui,
        ws,
        "Taper",
        &mut taper,
        Unit::Deg,
        Some(NominalKind::Angle),
        Some("Degrees the sides lean as the extrude goes away from the sketch, as Fusion's taper angle: positive leans them out, negative leans them in — a moulded boss's draft. 0 for straight sides. The sketch's outline stays the section at the sketch plane, or at the end of the extent nearest it, and an extrude reaching both ways leans both ways from there. The arrow on the far edge drags it. A taper that would close the profile before the end is refused, saying where it closes."),
        taper_expr.as_deref(),
        Some(-MAX_TAPER_DEG..=MAX_TAPER_DEG),
    );
    if c {
        let t = taper.clamp(-MAX_TAPER_DEG, MAX_TAPER_DEG);
        e.draft_deg = if t == 0.0 { 0.0 } else { -t };
        changed = true;
    }
    changed |= direction_field(ui, ws, e);
    changed |= op_field(ui, ws, e);
    if changed {
        follow_face(e);
    }
    // The field on a grip once it has been dragged.
    if let Some((side, v)) = fields::grip_field(ui.ctx(), |side| grip_value(e, side)) {
        set_grip_value(e, side, v);
        changed = true;
    }
    changed
}

// ---- the part ----

/// The region of a sketch shown on the part under a screen point: the box's sketch first, then
/// any other shown — clicking one of another sketch takes that sketch.
pub(crate) fn region_under(ws: &ReverseWorkspace, vp: Viewport, p: Pos2, first: FeatureId) -> Option<(FeatureId, String)> {
    let mut order: Vec<FeatureId> = vec![first];
    order.extend(ws.part.features.iter().rev().filter(|f| f.visible() && !f.suppressed() && f.id != first && matches!(f.kind, FeatureKind::Sketch(_))).map(|f| f.id));
    for id in order {
        let Some(frame) = ws.sketch_frame(id) else { continue };
        let Some(sheet) = ws.sheet_point(vp, p, &frame) else { continue };
        let regions = ws.built.sketches.get(&id).map(|b| b.regions.as_slice()).unwrap_or(&[]);
        if let Some(r) = region_at(regions, sheet) {
            return Some((id, r.key.clone()));
        }
    }
    None
}

/// A region clicked: toggled in the profiles, or — of another sketch — that sketch taken with it
/// alone (ScanRuler's `toggleDraftRegion`).
pub(crate) fn toggle_region(sketch: &mut FeatureId, regions: &mut Vec<String>, region_pick: &mut Option<mm_reverse::model::RegionPick>, of: FeatureId, key: String) {
    if *sketch != of {
        *sketch = of;
        *regions = vec![key];
    } else if let Some(i) = regions.iter().position(|k| *k == key) {
        regions.remove(i);
    } else {
        regions.push(key);
    }
    *region_pick = None;
}

pub(crate) fn pick(ws: &mut ReverseWorkspace, vp: Viewport, p: Pos2, clicked: bool, e: &mut ExtrudeFeature, pick: &mut PickWhat) -> bool {
    // A region of a sketch takes the click, through the part.
    if e.face.is_none() {
        let region = region_under(ws, vp, p, e.sketch_id);
        ws.view.hover_region = region.as_ref().filter(|r| r.0 == e.sketch_id).map(|r| r.1.clone());
        if let Some((of, key)) = region {
            if clicked {
                toggle_region(&mut e.sketch_id, &mut e.regions, &mut e.region_pick, of, key);
                return true;
            }
            return false;
        }
    }
    // Reaching to a face: a face of a body takes the click.
    if *pick == PickWhat::Faces {
        if let Some(crate::reverse::view::Hit::Body { body, face, .. }) = ws.pick_hit(vp, p) {
            ws.view.hover_face = Some((body, face));
            if clicked {
                if let Some(print) = ws.pick_view(body).and_then(|v| v.topology.faces.get(face)).map(|f| f.fingerprint) {
                    if let ExtrudeExtent::ToFace { body: b, face: f, .. } = &mut e.extent {
                        *b = body;
                        *f = print;
                    }
                    ws.status = format!("The extrude reaches to {}.", describe_face(&print));
                    return true;
                }
            }
        }
        return false;
    }
    // Elsewhere on the part: an end at that height — a body's corner where one is under the
    // pointer. Only an extrude by numbers takes a point (ScanRuler's `extrudeTakesPoint`: a
    // Distance or a Symmetric extent); an end found on the part leaves the click alone.
    if clicked && *pick == PickWhat::Height {
        let at = corner_under(ws, vp, p).or_else(|| ws.pick_hit(vp, p).map(|h| h.point()));
        if let Some(at) = at {
            return click_point(ws, e, at);
        }
    }
    false
}

/// Whether an extent takes a point clicked on the part for one of its numbers: a Distance or a
/// Symmetric one.
pub(crate) fn takes_point(extent: &ExtrudeExtent) -> bool {
    matches!(extent, ExtrudeExtent::Distance { .. } | ExtrudeExtent::Symmetric { .. })
}

/// A point on the part clicked while the box is open (ScanRuler's `extrudeToPoint`): the end in
/// play put through it, measured from the sketch's plane or the pulled face's, a pulled face's
/// result following, the expressions behind the numbers let go; the status line says what came
/// of it. Whether the extrude changed.
pub(crate) fn click_point(ws: &mut ReverseWorkspace, e: &mut ExtrudeFeature, at: DVec3) -> bool {
    if !takes_point(&e.extent) {
        return false;
    }
    let Some((origin, normal)) = base_plane(ws, e) else {
        ws.status = "Choose the sketch first — the distance is measured from its plane.".into();
        return false;
    };
    match extrude_to_point(&e.extent, origin, normal, e.direction, at) {
        Ok((extent, said)) => {
            e.extent = extent;
            follow_face(e);
            unlink(e, &["extent.from", "extent.to", "extent.total"]);
            ws.status = said;
            true
        }
        Err(why) => {
            ws.status = why;
            false
        }
    }
}

/// A corner of a body of the pick set within reach of a screen point: the end of an edge.
fn corner_under(ws: &ReverseWorkspace, vp: Viewport, p: Pos2) -> Option<DVec3> {
    let mut best: Option<(DVec3, f32)> = None;
    for b in ws.pick_set().iter().filter(|b| !ws.hidden_bodies.contains(&b.id)) {
        for e in &b.topology.edges {
            for q in e.ends {
                let Some(s) = ws.project(vp, q) else { continue };
                let d = (s - p).length();
                if d <= 8.0 && best.is_none_or(|x| d < x.1) {
                    best = Some((q, d));
                }
            }
        }
    }
    best.map(|b| b.0)
}

pub(crate) fn paint(ws: &ReverseWorkspace, painter: &egui::Painter, vp: Viewport, e: &ExtrudeFeature) {
    if e.face.is_none() {
        fields::paint_region_faces(ws, painter, vp, e.sketch_id, &e.regions);
    }
    fields::paint_box_hint(ws, painter, vp);
    if let ExtrudeExtent::ToFace { body, face, .. } = &e.extent {
        if *body != NO_BODY {
            paint_faces(ws, painter, vp, *body, std::slice::from_ref(face));
        }
    }
    let grips = grips_of(ws, e);
    fields::paint_grips(ws, painter, vp, &grips, e.op == BodyOp::Cut);
    // While a grip is dragged, the profile's outline at the ends it has come to, at once.
    if with_box(|b| b.drag.is_some()) || ws.building() {
        if let Some((loops, along, from, to)) = stretch_of(ws, e) {
            fields::paint_stretch(ws, painter, vp, &loops, along, from, to, e.op == BodyOp::Cut);
        }
    }
}

/// The picked profiles' outlines in the part, the way they are swept, and the ends of the extent
/// where it has numbers.
fn stretch_of(ws: &ReverseWorkspace, e: &ExtrudeFeature) -> Option<(Vec<Vec<DVec3>>, DVec3, f64, f64)> {
    // A tapered one is not stretched: its sides would lean the wrong amount.
    if e.draft_deg != 0.0 {
        return None;
    }
    let (from, to) = match e.extent {
        ExtrudeExtent::Distance { from, to } => (from, to),
        ExtrudeExtent::Symmetric { total } => (-total / 2.0, total / 2.0),
        _ => return None,
    };
    let frame = ws.sketch_frame(e.sketch_id).filter(|_| e.face.is_none())?;
    let along = e.direction.and_then(|d| d.try_normalize()).unwrap_or(frame.normal);
    let loops = regions_of(ws, e.sketch_id).iter().filter(|r| e.regions.contains(&r.key)).flat_map(|r| r.outline(0.1)).map(|l| l.into_iter().map(|q| frame.to_world(q)).collect()).collect();
    Some((loops, along, from, to))
}

// ---- grips ----

/// The frame a face pulled stands in: at its centroid, along its outward normal.
fn face_frame(f: &FaceProfile) -> Option<SectionFrame> {
    SectionFrame::sketch_on(f.face.centroid, f.face.dir.try_normalize()?, None)
}

/// The grips the extrude wears: an arrow on each face its numbers set, and one on the far edge
/// for the taper — not when it is swept askew, or pulled from a face.
pub(crate) fn grips_of(ws: &ReverseWorkspace, e: &ExtrudeFeature) -> Vec<FeatureGrip> {
    if extrude_ready(e).is_err() && extent_lacks(&e.extent).is_none() {
        return Vec::new();
    }
    if let Some(f) = &e.face {
        let Some(frame) = face_frame(f) else { return Vec::new() };
        return extrude_grips_at(frame.origin, e.direction.and_then(|d| d.try_normalize()).unwrap_or(frame.normal), &e.extent);
    }
    let Some(frame) = ws.sketch_frame(e.sketch_id) else { return Vec::new() };
    let regions = regions_of(ws, e.sketch_id);
    let picked: Vec<&Region> = regions.iter().filter(|r| e.regions.contains(&r.key)).collect();
    let along = e.direction.and_then(|d| d.try_normalize());
    let mut grips = extrude_grips(&frame, &picked, &e.extent, along);
    // Swept a way of its own, the taper's grip stays away: it stands on a side square to the
    // plane (ScanRuler hides it whenever a direction is given, the normal's own included).
    if e.direction.is_none() {
        if let Some(g) = taper_grip(&frame, &picked, &e.extent, if e.draft_deg == 0.0 { 0.0 } else { -e.draft_deg }) {
            grips.push(g);
        }
    }
    grips
}

/// The number a grip drags, as its field reads it.
fn grip_value(e: &ExtrudeFeature, side: GripSide) -> Option<f64> {
    match (side, &e.extent) {
        (GripSide::Taper, _) => Some(if e.draft_deg == 0.0 { 0.0 } else { -e.draft_deg }),
        (GripSide::ExtrudeTo, ExtrudeExtent::Distance { to, .. }) => Some(*to),
        (GripSide::ExtrudeFrom, ExtrudeExtent::Distance { from, .. }) => Some(*from),
        (GripSide::ExtrudeTo | GripSide::ExtrudeFrom, ExtrudeExtent::Symmetric { total }) => Some(*total),
        (GripSide::ExtrudeFrom, ExtrudeExtent::ToPlane { from, .. } | ExtrudeExtent::ToFace { from, .. } | ExtrudeExtent::ToBody { from, .. } | ExtrudeExtent::ToNext { from, .. } | ExtrudeExtent::ToLast { from, .. }) => Some(*from),
        _ => None,
    }
}

/// The slots a grip stands in for, whose expressions a drag or a typed number lets go.
fn grip_slots(e: &ExtrudeFeature, side: GripSide) -> Vec<&'static str> {
    match side {
        GripSide::Taper => vec!["taper"],
        _ if matches!(e.extent, ExtrudeExtent::Symmetric { .. }) => vec!["extent.total"],
        GripSide::ExtrudeFrom => vec!["extent.from"],
        _ => vec!["extent.to"],
    }
}

/// A number typed on a grip's field (ScanRuler's `setGripValue`): it stands on its own, the
/// expression behind it let go.
pub(crate) fn set_grip_value(e: &mut ExtrudeFeature, side: GripSide, value: f64) {
    if !value.is_finite() {
        return;
    }
    let slots = grip_slots(e, side);
    if side == GripSide::Taper {
        let t = value.clamp(-MAX_TAPER_DEG, MAX_TAPER_DEG);
        e.draft_deg = if t == 0.0 { 0.0 } else { -t };
    } else {
        match &mut e.extent {
            ExtrudeExtent::Distance { from, to } => {
                if side == GripSide::ExtrudeFrom {
                    *from = value;
                } else {
                    *to = value;
                }
            }
            ExtrudeExtent::Symmetric { total } => *total = value.abs(),
            ExtrudeExtent::ToPlane { from, .. } | ExtrudeExtent::ToFace { from, .. } | ExtrudeExtent::ToBody { from, .. } | ExtrudeExtent::ToNext { from, .. } | ExtrudeExtent::ToLast { from, .. } if side == GripSide::ExtrudeFrom => *from = value,
            _ => return,
        }
        follow_face(e);
    }
    unlink(e, &slots);
}

/// A grip dragged `delta` (millimetres along its arrow) from where the drag began (ScanRuler's
/// `gripDrag`): the extent it began with pulled, whole hundredths; the taper the pull puts the
/// far edge at.
pub(crate) fn apply_drag(e: &mut ExtrudeFeature, drag: &GripDrag, delta: f64) {
    if drag.side == GripSide::Taper {
        let t = drag_taper(drag.taper_deg, drag.taper_run, delta);
        e.draft_deg = if t == 0.0 { 0.0 } else { -t };
        unlink(e, &["taper"]);
        return;
    }
    let Some(start) = &drag.extent else { return };
    let tidy = |v: f64| (v * 100.0).round() / 100.0;
    let mut next = drag_extrude_extent(start, drag.side, delta);
    match &mut next {
        ExtrudeExtent::Distance { from, to } => {
            *from = tidy(*from);
            *to = tidy(*to);
        }
        ExtrudeExtent::Symmetric { total } => *total = tidy(*total),
        ExtrudeExtent::ToPlane { from, .. } | ExtrudeExtent::ToFace { from, .. } | ExtrudeExtent::ToBody { from, .. } | ExtrudeExtent::ToNext { from, .. } | ExtrudeExtent::ToLast { from, .. } => *from = tidy(*from),
        ExtrudeExtent::BetweenPlanes { .. } => {}
    }
    let slots = grip_slots(e, drag.side);
    e.extent = next;
    follow_face(e);
    unlink(e, &slots);
}

/// What a drag begins with: the grip, the extent and the taper as they stand.
pub(crate) fn drag_of(e: &ExtrudeFeature, grip: FeatureGrip, t0: f64) -> GripDrag {
    let taper = if e.draft_deg == 0.0 { 0.0 } else { -e.draft_deg };
    let run = taper_span(&e.extent).map_or(0.0, |(s, end)| (end - s).abs());
    GripDrag { side: grip.side, grip, t0, last: t0, wound: 0.0, extent: Some(e.extent.clone()), taper_deg: taper, taper_run: run, angle_deg: 0.0 }
}

/// The pointer on the grips: under it one is lit; pressed on one, a drag begins; dragged, the
/// number follows — a build asked for one at a time; let go, the grip's field takes typing.
/// `None` where the grips leave the pointer to the box's other picks.
pub(crate) fn grip_input(ws: &mut ReverseWorkspace, vp: Viewport, pos: Option<Pos2>, pressed: bool, down: bool, released: bool, e: &mut ExtrudeFeature) -> Option<bool> {
    let grips = grips_of(ws, e);
    if let Some(mut drag) = with_box(|b| b.drag.take()) {
        if down && !released {
            if let Some(delta) = pos.and_then(|p| fields::drag_delta(ws, vp, p, &mut drag)) {
                let before = e.clone();
                apply_drag(e, &drag, delta);
                if *e != before {
                    with_box(|b| b.owed = true);
                }
            }
            with_box(|b| b.drag = Some(drag));
            // One build out at a time: the next is asked for once the last has come in.
            return Some(with_box(|b| {
                let go = b.owed && !ws.building();
                if go {
                    b.owed = false;
                }
                go
            }));
        }
        // Let go: the field takes typing; the last of the drag built.
        let v = grip_value(e, drag.side).unwrap_or(0.0);
        with_box(|b| {
            b.owed = false;
            b.field = Some(GripField { side: drag.side, text: fields::field_words(v, drag.side).0, typing: true, error: None });
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
        let drag = drag_of(e, grip, t0);
        with_box(|b| {
            b.field = Some(GripField { side: grip.side, text: String::new(), typing: false, error: None });
            b.drag = Some(drag);
        });
    }
    Some(false)
}

/// What the status line says while the box is ready and waiting.
#[allow(dead_code)]
pub(crate) fn waiting(e: &ExtrudeFeature) -> Option<&'static str> {
    extent_lacks(&e.extent)
}

/// The scan's points near an extrude's prism with their normals, and the extrude's end read off
/// them ("To scan", ScanRuler's `fitExtrudeEnd`): kept as a distance, as one typed. What the
/// status line says of it.
pub(crate) fn fit_end_to_scan(ws: &mut ReverseWorkspace, e: &mut ExtrudeFeature, near: Option<f64>) -> Result<String, String> {
    use mm_reverse::extrude_scan::{end_box, fit_extrude_end, EndAsk, EndRegion};
    let frame = ws.sketch_frame(e.sketch_id).filter(|_| e.face.is_none()).ok_or("An extrude to the scan needs its sketch.")?;
    let regions: Vec<EndRegion> = regions_of(ws, e.sketch_id)
        .iter()
        .filter(|r| e.regions.contains(&r.key))
        .map(|r| {
            let loops = r.outline(0.05);
            EndRegion { outer: loops.first().cloned().unwrap_or_default(), holes: loops.into_iter().skip(1).collect() }
        })
        .collect();
    if regions.is_empty() {
        return Err("The sketch has none of the regions picked.".into());
    }
    let dir = e.direction.and_then(|d| d.try_normalize()).unwrap_or(frame.normal);
    let (from, reverse) = match &e.extent {
        ExtrudeExtent::Distance { from, to } => (*from, *to < *from),
        _ => (0.0, false),
    };
    let size = 2.0 * fields::model_extent(ws).1;
    let (lo, hi) = end_box(&frame, dir, &regions, from, size);
    let parts = ws.scan_parts();
    let (mut points, mut normals) = (Vec::new(), Vec::new());
    for p in parts.iter() {
        let mut acc = vec![DVec3::ZERO; p.positions.len()];
        for t in &p.triangles {
            let [a, b, c] = t.map(|i| p.positions.get(i as usize).copied().unwrap_or_default());
            let n = (b - a).cross(c - a);
            for i in t {
                if let Some(s) = acc.get_mut(*i as usize) {
                    *s += n;
                }
            }
        }
        for v in &p.live_vertices {
            let q = p.positions[*v as usize];
            if q.cmplt(lo).any() || q.cmpgt(hi).any() {
                continue;
            }
            if let Some(n) = acc[*v as usize].try_normalize() {
                points.push(q);
                normals.push(n);
            }
        }
    }
    let r = fit_extrude_end(&frame, dir, &regions, &points, &normals, EndAsk { from, reverse, cut: e.op == BodyOp::Cut, near })?;
    e.extent = ExtrudeExtent::Distance { from, to: r.to };
    unlink(e, &["extent.to"]);
    Ok(format!(
        "To = {} mm, read off {} scan points (σ {}); the nominal is {}{}.",
        mm_reverse::js_number::to_fixed(r.to, 3),
        r.points,
        r.sigma.map_or("–".into(), |s| mm_reverse::js_number::to_fixed(s, 4)),
        r.nominal,
        r.sigmas.map_or(String::new(), |s| format!(", {} sigma off", mm_reverse::js_number::to_fixed(s, 1)))
    ))
}

/// A face's index on a body of the pick set nearest a point, for a script.
pub(crate) fn face_near(ws: &ReverseWorkspace, at: DVec3) -> Option<(BodyId, usize)> {
    let mut best: Option<(f64, BodyId, usize)> = None;
    for b in ws.pick_set() {
        for (i, f) in b.topology.faces.iter().enumerate() {
            let d = (f.point - at).length().min((f.fingerprint.centroid - at).length());
            if best.is_none_or(|x| d < x.0) {
                best = Some((d, b.id, i));
            }
        }
    }
    best.map(|b| (b.1, b.2))
}

/// The state the box keeps, for a test.
#[allow(dead_code)]
pub(crate) fn box_state() -> BoxState {
    with_box(|b| b.clone())
}

/// A point on the sheet lifted into the part through a frame.
#[allow(dead_code)]
pub(crate) fn lift(frame: &SectionFrame, p: DVec2) -> DVec3 {
    frame.to_world(p)
}
