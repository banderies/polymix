// SPDX-License-Identifier: AGPL-3.0-only
//! The Hole feature's box: the body; what the hole runs along — a cylinder, circle or line
//! element chosen from the list (its axis drawn on the part while the pointer is over it), a
//! round face of the body clicked (its axis, and its radius to start), or a flat face clicked
//! to drill into at its centre, square to it; a heat-set insert to size the hole for; the
//! diameter; through all, or a depth. Ported from ScanRuler (`plugins/reverse/ui/
//! FeatureEditor.tsx` the hole case, `state/reFeatureSlice.ts` the hole draft and its face
//! picks, `app/useReversePlugin.tsx` the element pick), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only.

use std::cell::Cell;

use egui::{Pos2, RichText, Stroke};
use glam::DVec3;
use mm_reverse::build::{is_unpicked, resolve_hole_axis, unpicked_face};
use mm_reverse::inserts::{insert_hole, insert_size, INSERT_SIZES};
use mm_reverse::length_fields::len_bare;
use mm_reverse::model::{BodyId, ElementGeometry, ElementId, Exprs, FaceRef, FeatureKind, HoleAxis, HoleFeature, Part};
use mm_reverse::nominal::NominalKind;
use mm_reverse::topo::{describe_face, FaceKind};

use super::thread::{bare3, label, last_solid, mm, num, parameter_table, select, HINT, PICKED, SELECT_WIDTH, UNITS};
use crate::reverse::view::Hit;
use crate::reverse::{PickWhat, ReverseWorkspace};
use crate::tools::Viewport;

pub(crate) const READY: bool = true;
pub(crate) const BUTTON: (&str, &str) = ("Hole", "A hole through the body along a cylinder or circle element, or a round face of the body");
pub(crate) const FIRST_PICK: PickWhat = PickWhat::Faces;

/// The radius a new hole starts at, mm: Ø6.
const START_RADIUS: f64 = 3.0;
/// The depth a hole takes when Through all is let go, or a flat face is clicked, mm.
const START_DEPTH: f64 = 10.0;

thread_local! {
    /// The element under the pointer in the Along list, drawn on the part.
    static HOVERED: Cell<Option<ElementId>> = const { Cell::new(None) };
}

pub(crate) fn new_feature(ws: &ReverseWorkspace) -> Result<FeatureKind, String> {
    let body = last_solid(ws).ok_or("A hole is drilled into a body: make a body first.")?;
    Ok(FeatureKind::Hole(HoleFeature { body, axis: HoleAxis::Face { face: unpicked_face() }, radius: START_RADIUS, depth: None, insert: None, exprs: Exprs::new() }))
}

/// A hole along a measured element's axis, its radius the element's: what the chip on a
/// cylinder or circle element opens ("Drill a hole through the body along {element}").
#[allow(dead_code)]
pub(crate) fn new_along_element(ws: &ReverseWorkspace, id: ElementId) -> Result<FeatureKind, String> {
    let FeatureKind::Hole(mut x) = new_feature(ws)? else { return Err("No hole.".into()) };
    take_element(&ws.part, &mut x, id)?;
    Ok(FeatureKind::Hole(x))
}

/// A hole on a face of a body, as a click on it takes it: what the chip on a round face opens
/// ("A hole along this face’s axis").
#[allow(dead_code)]
pub(crate) fn new_on_face(ws: &ReverseWorkspace, body: BodyId, face: FaceRef) -> Result<FeatureKind, String> {
    let FeatureKind::Hole(mut x) = new_feature(ws)? else { return Err("No hole.".into()) };
    x.body = body;
    take_face(&mut x, face);
    Ok(FeatureKind::Hole(x))
}

/// A measured element taken for the hole's axis (ScanRuler's `elementPick` into a hole): a
/// cylinder's or a circle's sets the diameter, a line's keeps it; any other has no axis.
pub(crate) fn take_element(part: &Part, x: &mut HoleFeature, id: ElementId) -> Result<(), String> {
    let e = part.element(id).ok_or("The element is gone.")?;
    match e.geometry {
        ElementGeometry::Cylinder { radius, .. } | ElementGeometry::Circle { radius, .. } => {
            x.axis = HoleAxis::Element { id };
            x.radius = radius;
            Ok(())
        }
        ElementGeometry::Line { .. } => {
            x.axis = HoleAxis::Element { id };
            Ok(())
        }
        _ => Err(format!("{} has no axis to drill along — pick a cylinder, a circle or a line.", e.name)),
    }
}

/// A face of the body taken for the hole (ScanRuler's `toggleDraftFace` for a hole): a flat face
/// is drilled into at its centre — a boss's top, a pad — to a depth; a round one along its axis,
/// its radius taken while the hole is at the size it started at and sized for no insert.
/// Returns whether the face was taken.
pub(crate) fn take_face(x: &mut HoleFeature, face: FaceRef) -> bool {
    match face.kind {
        FaceKind::Plane => {
            x.axis = HoleAxis::Point { face, at: face.centroid };
            x.depth = Some(x.depth.unwrap_or(START_DEPTH));
            true
        }
        FaceKind::Cylinder => {
            if x.radius == START_RADIUS && x.insert.is_none() {
                if let Some(r) = face.radius {
                    x.radius = r;
                }
            }
            x.axis = HoleAxis::Face { face };
            true
        }
        _ => false,
    }
}

/// Whether the box has everything it needs to be created (ScanRuler's `featureDraftReady`).
pub(crate) fn ready(part: &Part, x: &HoleFeature) -> bool {
    resolve_hole_axis(&x.axis, part).is_ok() && x.radius > 0.0 && x.depth.is_none_or(|d| d > 0.0)
}

/// The elements a hole runs along: the cylinders, circles and lines.
fn axis_elements(part: &Part) -> Vec<(ElementId, String)> {
    part.elements.iter().filter(|e| matches!(e.geometry, ElementGeometry::Cylinder { .. } | ElementGeometry::Circle { .. } | ElementGeometry::Line { .. })).map(|e| (e.id, e.name.clone())).collect()
}

pub(crate) fn ui(ws: &mut ReverseWorkspace, ui: &mut egui::Ui, x: &mut HoleFeature, pick: &mut PickWhat) -> bool {
    let mut changed = false;
    let id = egui::Id::NULL.with("re_hole_box");
    HOVERED.with(|h| h.set(None));
    // 1. The body.
    let bodies: Vec<(BodyId, String)> = ws.pick_set().iter().map(|b| (b.id, b.name.clone())).collect();
    let shown = bodies.iter().find(|b| b.0 == x.body).map_or("Select…".to_string(), |b| b.1.clone());
    let options: Vec<(String, Option<String>)> = bodies.iter().map(|b| (b.1.clone(), None)).collect();
    if let Some(i) = select(ui, id.with("body"), "Body", None, &shown, &options, None) {
        if bodies[i].0 != x.body {
            x.body = bodies[i].0;
            changed = true;
        }
    }
    // 2. What it runs along.
    let elements = axis_elements(&ws.part);
    let chosen = match x.axis {
        HoleAxis::Element { id } => elements.iter().find(|e| e.0 == id).map(|e| e.1.clone()),
        _ => None,
    };
    let placeholder = match &x.axis {
        HoleAxis::Face { face } if !is_unpicked(face) => describe_face(face),
        HoleAxis::Point { face, .. } => format!("{}, at its centre", describe_face(face)),
        _ => "Click a round face — or a flat one to drill into at its centre — or choose…".to_string(),
    };
    let mut over = None;
    let mut take = None;
    ui.horizontal(|ui| {
        label(ui, "Along", None);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let text = match &chosen {
                Some(c) => RichText::new(c.clone()),
                None => RichText::new(placeholder.clone()).weak(),
            };
            let r = egui::ComboBox::from_id_salt(id.with("along")).selected_text(text).width(SELECT_WIDTH).truncate().show_ui(ui, |ui| {
                if elements.is_empty() {
                    ui.weak("Nothing to choose from");
                    return;
                }
                ui.label(RichText::new("ELEMENTS").weak().small());
                for (eid, name) in &elements {
                    let r = ui.selectable_label(chosen.as_deref() == Some(name.as_str()) && matches!(x.axis, HoleAxis::Element { id } if id == *eid), name);
                    if r.hovered() {
                        over = Some(*eid);
                    }
                    if r.clicked() {
                        take = Some(*eid);
                    }
                }
            });
            r.response.on_hover_text(chosen.clone().unwrap_or_else(|| placeholder.clone()));
        });
    });
    HOVERED.with(|h| h.set(over));
    if let Some(eid) = take {
        if let Err(e) = take_element(&ws.part, x, eid) {
            ws.status = e;
        }
        changed = true;
    }
    // 3. The insert.
    let mut options: Vec<(String, Option<String>)> = vec![("None — a plain hole".to_string(), None)];
    options.extend(INSERT_SIZES.iter().map(|z| (format!("{} — Ø {} × {}", z.label, bare3(z.hole), bare3(z.length)), None)));
    let insert = x.insert.as_deref().and_then(insert_size);
    let shown = insert.map_or_else(|| options[0].0.clone(), |z| format!("{} — Ø {} × {}", z.label, bare3(z.hole), bare3(z.length)));
    if let Some(i) = select(ui, id.with("insert"), "Insert", Some("Size the hole for a heat-set insert: its hole diameter, and a millimetre deeper than the insert is long"), &shown, &options, None) {
        match i.checked_sub(1).and_then(|k| INSERT_SIZES.get(k)) {
            Some(z) => {
                let h = insert_hole(z);
                x.insert = Some(z.label.to_string());
                x.radius = h.radius;
                x.depth = Some(h.depth);
                x.exprs.remove("diameter");
                x.exprs.remove("depth");
            }
            None => x.insert = None,
        }
        changed = true;
    }
    // 4. The diameter.
    let scope = parameter_table(&ws.part);
    let nominal = if x.insert.is_some() { None } else { Some(NominalKind::Length) };
    if let Some(v) = num(ui, &scope, id.with("diameter"), "Diameter", Some("The hole's diameter — the element's, to start"), 2.0 * x.radius, "diameter", &mut x.exprs, nominal) {
        x.radius = v.abs() / 2.0;
        x.insert = None;
        changed = true;
    }
    // 5. Through all.
    let mut through = x.depth.is_none();
    if ui.checkbox(&mut through, RichText::new("Through all").weak()).changed() {
        x.depth = if through { None } else { Some(START_DEPTH) };
        if through {
            x.exprs.remove("depth");
        }
        x.insert = None;
        changed = true;
    }
    // 6. The depth.
    if let Some(depth) = x.depth {
        let hint_text = if matches!(x.axis, HoleAxis::Point { .. }) { "How far into the body from the face" } else { "How far each way along the axis from its point" };
        if let Some(v) = num(ui, &scope, id.with("depth"), "Depth", Some(hint_text), depth, "depth", &mut x.exprs, nominal) {
            x.depth = Some(v.abs());
            x.insert = None;
            changed = true;
        }
    }
    // 7. The wall an insert wants.
    if let Some(z) = x.insert.as_deref().and_then(insert_size) {
        super::thread::hint(ui, format!("Leave at least {} of wall round it — a boss of Ø {} or more.", mm(z.wall), len_bare(z.hole + 2.0 * z.wall, UNITS, 1)));
    }
    *pick = PickWhat::Faces;
    changed
}

/// A click on a face of the box's body: a flat face is drilled into at its centre, a round one
/// along its axis; the face under the cursor is lit.
pub(crate) fn pick(ws: &mut ReverseWorkspace, vp: Viewport, p: Pos2, clicked: bool, x: &mut HoleFeature, _pick: &mut PickWhat) -> bool {
    let Some(Hit::Body { body: b, face, .. }) = ws.pick_hit(vp, p) else { return false };
    if b != x.body {
        return false;
    }
    ws.view.hover_face = Some((b, face));
    if !clicked {
        return false;
    }
    let Some(print) = ws.pick_view(b).and_then(|v| v.topology.faces.get(face)).map(|f| f.fingerprint) else { return false };
    take_face(x, print)
}

/// A line along an axis through a point, drawn on the part.
fn axis_line(ws: &ReverseWorkspace, painter: &egui::Painter, vp: Viewport, from: DVec3, to: DVec3, stroke: Stroke, dashed: bool) {
    let n = 32;
    let pts: Vec<Pos2> = (0..=n).filter_map(|k| ws.project(vp, from + (to - from) * (k as f64 / n as f64))).collect();
    if pts.len() < 2 {
        return;
    }
    if dashed {
        painter.add(egui::Shape::dashed_line(&pts, stroke, 8.0, 5.0));
    } else {
        painter.add(egui::Shape::line(pts, stroke));
    }
}

/// A circle about an axis, drawn on the part.
fn circle(ws: &ReverseWorkspace, painter: &egui::Painter, vp: Viewport, center: DVec3, axis: DVec3, r: f64, stroke: Stroke) {
    let (u, v) = mm_reverse::linalg::ortho_basis(axis.normalize_or_zero());
    let pts: Vec<Pos2> = (0..=48)
        .filter_map(|k| {
            let a = std::f64::consts::TAU * k as f64 / 48.0;
            ws.project(vp, center + (u * a.cos() + v * a.sin()) * r)
        })
        .collect();
    if pts.len() >= 2 {
        painter.add(egui::Shape::line(pts, stroke));
    }
}

/// The element under the pointer in the Along list, its axis drawn in the hint colour with its
/// name; the hole's own axis and its circle where it starts, in the picked colour.
pub(crate) fn paint(ws: &ReverseWorkspace, painter: &egui::Painter, vp: Viewport, x: &HoleFeature) {
    let size = ws.pick_view(x.body).map_or(50.0, |b| b.mesh.bounds().diagonal().max(10.0));
    if let Some(e) = HOVERED.with(|h| h.get()).and_then(|id| ws.part.element(id)) {
        if let Ok(a) = resolve_hole_axis(&HoleAxis::Element { id: e.id }, &ws.part) {
            let d = a.dir.normalize_or_zero();
            let half = match e.geometry {
                ElementGeometry::Cylinder { length, .. } | ElementGeometry::Line { length, .. } => (length / 2.0).max(size * 0.3),
                _ => size * 0.3,
            };
            axis_line(ws, painter, vp, a.origin - d * half, a.origin + d * half, Stroke::new(2.0, HINT), true);
            if let Some(q) = ws.project(vp, a.origin + d * half) {
                painter.text(q + egui::vec2(6.0, -6.0), egui::Align2::LEFT_BOTTOM, &e.name, egui::FontId::proportional(13.0), HINT);
            }
        }
    }
    let Ok(a) = resolve_hole_axis(&x.axis, &ws.part) else { return };
    let d = a.dir.normalize_or_zero();
    let (from, to) = if a.one_way {
        (a.origin, a.origin + d * x.depth.unwrap_or(START_DEPTH))
    } else {
        match x.depth {
            Some(depth) => (a.origin - d * depth, a.origin + d * depth),
            None => (a.origin - d * size, a.origin + d * size),
        }
    };
    axis_line(ws, painter, vp, from, to, Stroke::new(1.5, PICKED), true);
    if x.radius > 0.0 {
        circle(ws, painter, vp, a.origin, d, x.radius, Stroke::new(2.0, PICKED));
    }
    if let Some(q) = ws.project(vp, a.origin) {
        painter.circle_filled(q, 4.0, PICKED);
    }
}

// ---- the script's step ----

/// The script's `hole` step: a Hole on the face nearest a point — a round face along its axis,
/// a flat one drilled into at its centre — or along an element (`hole:element:<id>`), with what
/// the words after it ask — `d=` (the diameter), `depth=`, `through`, `insert=<label>` —
/// created, or left open with `open`. Returns what it did, for the log.
pub(crate) fn script_step(ws: &mut ReverseWorkspace, args: &[&str]) -> String {
    let Ok(FeatureKind::Hole(mut x)) = new_feature(ws) else { return "hole step: no body".into() };
    let rest: &[&str] = if args.first() == Some(&"element") {
        let Some(eid) = args.get(1).and_then(|v| v.parse::<ElementId>().ok()) else { return "hole step: hole:element:<id>".into() };
        if let Err(e) = take_element(&ws.part, &mut x, eid) {
            return format!("hole step: {e}");
        }
        &args[2..]
    } else {
        let n = |i: usize| args.get(i).and_then(|v| v.parse::<f64>().ok());
        let (Some(px), Some(py), Some(pz)) = (n(0), n(1), n(2)) else { return "hole step: give a point on the face, hole:x:y:z[:…], or hole:element:<id>".into() };
        let Some((body, face)) = super::thread::face_near(ws, DVec3::new(px, py, pz)) else { return "hole step: no body near the point".into() };
        let Some(print) = ws.pick_view(body).and_then(|v| v.topology.faces.get(face)).map(|f| f.fingerprint) else { return "hole step: no face there".into() };
        x.body = body;
        if !take_face(&mut x, print) {
            return format!("hole step: the face there is a {} — a hole takes a round face or a flat one", print.kind.name());
        }
        &args[3..]
    };
    let mut open = false;
    for a in rest {
        let (k, v) = a.split_once('=').unwrap_or((a, ""));
        match (k, v.parse::<f64>().ok()) {
            ("d", Some(d)) => {
                x.radius = d / 2.0;
                x.insert = None;
            }
            ("depth", Some(d)) => x.depth = Some(d),
            ("through", _) => x.depth = None,
            ("insert", _) => {
                if let Some(z) = insert_size(&v.replace('_', " ")) {
                    let h = insert_hole(z);
                    x.insert = Some(z.label.to_string());
                    x.radius = h.radius;
                    x.depth = Some(h.depth);
                }
            }
            ("open", _) => open = true,
            _ => {}
        }
    }
    let ok = ready(&ws.part, &x);
    let what = format!("Ø{} {}", bare3(2.0 * x.radius), x.depth.map_or("through all".to_string(), |d| format!("{} deep", bare3(d))));
    ws.open_feature(FeatureKind::Hole(x), None);
    if open {
        return format!("hole box open: {what}");
    }
    if !ok {
        return format!("hole not ready: {what}");
    }
    ws.commit_feature();
    format!("hole {what} created")
}

#[cfg(test)]
mod tests {
    //! ScanRuler's `reFeatureDraft.test.ts`: a hole takes a round face of the body for its axis
    //! and its radius, or a flat one to drill into at its centre; it is ready once it has what it
    //! runs along.
    use super::*;
    use mm_reverse::model::Element;
    use mm_reverse::topo::FaceFingerprint;

    fn hole() -> HoleFeature {
        HoleFeature { body: BodyId::new(2), axis: HoleAxis::Face { face: unpicked_face() }, radius: START_RADIUS, depth: None, insert: None, exprs: Exprs::new() }
    }

    #[test]
    fn takes_a_round_face_for_its_axis_and_radius_or_a_flat_one_to_drill_into_at_its_centre() {
        let part = Part::default();
        let mut x = hole();
        assert!(!ready(&part, &x));
        let flat = FaceFingerprint { kind: FaceKind::Plane, centroid: DVec3::new(1.0, 2.0, 3.0), dir: DVec3::Z, area: 100.0, radius: None, center: None, origin: 2 };
        assert!(take_face(&mut x, flat));
        assert_eq!(x.axis, HoleAxis::Point { face: flat, at: flat.centroid });
        assert_eq!(x.depth, Some(10.0));
        assert!(ready(&part, &x));
        let round = FaceFingerprint { kind: FaceKind::Cylinder, centroid: DVec3::new(10.0, 10.0, 5.0), dir: DVec3::Z, area: 94.0, radius: Some(3.5), center: Some(DVec3::new(10.0, 10.0, 5.0)), origin: 2 };
        assert!(take_face(&mut x, round));
        assert!(matches!(x.axis, HoleAxis::Face { .. }));
        assert_eq!(x.radius, 3.5);
        // Sized another way already, a round face keeps the size.
        let mut sized = HoleFeature { radius: 2.0, ..hole() };
        take_face(&mut sized, round);
        assert_eq!(sized.radius, 2.0);
        let mut insert = HoleFeature { insert: Some("M3".into()), ..hole() };
        take_face(&mut insert, round);
        assert_eq!(insert.radius, START_RADIUS);
    }

    #[test]
    fn takes_a_cylinder_or_circle_element_with_its_radius_a_line_without_and_nothing_else() {
        let mut part = Part::default();
        part.elements.push(Element { id: 4, name: "Cylinder 1".into(), geometry: ElementGeometry::Cylinder { center: DVec3::ZERO, axis: DVec3::Z, radius: 2.5, length: 8.0, coverage: 360.0 }, stats: None });
        part.elements.push(Element { id: 5, name: "Line 1".into(), geometry: ElementGeometry::Line { center: DVec3::ZERO, dir: DVec3::X, length: 8.0 }, stats: None });
        part.elements.push(Element { id: 6, name: "Sphere 1".into(), geometry: ElementGeometry::Sphere { center: DVec3::ZERO, radius: 2.0 }, stats: None });
        let mut x = hole();
        assert!(take_element(&part, &mut x, 4).is_ok());
        assert_eq!((x.axis.clone(), x.radius), (HoleAxis::Element { id: 4 }, 2.5));
        assert!(ready(&part, &x));
        assert!(take_element(&part, &mut x, 5).is_ok());
        assert_eq!((x.axis.clone(), x.radius), (HoleAxis::Element { id: 5 }, 2.5));
        assert_eq!(take_element(&part, &mut x, 6), Err("Sphere 1 has no axis to drill along — pick a cylinder, a circle or a line.".into()));
        assert_eq!(axis_elements(&part).iter().map(|e| e.0).collect::<Vec<_>>(), vec![4, 5]);
        // Not ready with no radius, or a depth of nothing.
        assert!(!ready(&part, &HoleFeature { radius: 0.0, ..x.clone() }));
        assert!(!ready(&part, &HoleFeature { depth: Some(0.0), ..x }));
    }
}
