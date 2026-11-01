// SPDX-License-Identifier: AGPL-3.0-only
//! The Draft feature's box: the body, the neutral plane (a flat face of the body clicked, or a
//! plane chosen), the faces to lean (each kept as its fingerprint) and the angle; groups of
//! faces at angles of their own, set through the commands, said under it. The first flat face
//! clicked is the neutral plane; the faces clicked after it are leaned, a second click letting
//! one go. Also the few fields the Move face box shares with it: a number with its nominal chip
//! and its ⓘ card, the body list and the plane list.
//!
//! Ported from ScanRuler (`plugins/reverse/ui/FeatureEditor.tsx` the draft fields,
//! `featureFields.tsx` Num, `PlaneSelect.tsx`; `state/reFeatureSlice.ts` toggleDraftFace),
//! © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.

use std::cell::RefCell;

use egui::{Color32, Pos2, RichText, Stroke};
use mm_reverse::build::describe_plane_ref;
use mm_reverse::draft_fit::{fit_draft_faces, DraftFaceIn, DRAFT_ANGLE};
use mm_reverse::display_units::{fmt_len, DisplayUnits, FmtOptions};
use mm_reverse::frame::WorldAxis;
use mm_reverse::model::{BodyId, DraftFeature, ElementGeometry, Exprs, FaceRef, FeatureId, FeatureKind, MirrorPlane, PlaneRef};
use mm_reverse::nominal::{nominal_of, NominalKind};
use mm_reverse::snap_face::{halves_of, one_surface};
use mm_reverse::topo::{describe_face, match_face, match_tolerance, FaceKind};

use super::paint_faces;
use crate::reverse::sketcher::plural;
use crate::reverse::{Mode, PickWhat, ReverseWorkspace};
use crate::tools::Viewport;

pub(crate) const READY: bool = true;
pub(crate) const BUTTON: (&str, &str) = ("Draft", "Lean faces of the body about a neutral plane — a moulding's draft, put on after the fact: click the flat face that stays, then the faces to lean");
pub(crate) const FIRST_PICK: PickWhat = PickWhat::Faces;

/// The angle a new draft starts at, degrees.
const DEFAULT_DRAFT_DEG: f64 = 2.0;
/// The colour a plane is hinted in on the part (§6.9).
const PLANE_HINT: Color32 = Color32::from_rgb(0xff, 0xb0, 0x20);

/// What the box keeps that the feature does not: whether a neutral plane is set yet (a new
/// draft holds a placeholder until one is clicked or chosen).
struct DraftBox {
    frame: u64,
    key: Option<FeatureId>,
    neutral_set: bool,
    /// Set up by a script step for the box it opened: the next frame's box takes it as its own.
    adopt: bool,
}

thread_local! {
    static BOX: RefCell<Option<DraftBox>> = const { RefCell::new(None) };
}

/// The box being edited's own key: the feature it edits, or none for a new one.
pub(super) fn box_key(ws: &ReverseWorkspace) -> Option<FeatureId> {
    ws.preview_feature.filter(|id| ws.part.feature(*id).is_some())
}

/// Whether the box state kept for `frame` and `key` is still the box's: shown the frame before,
/// for the same feature. A box closed and another opened, or another feature opened, starts
/// afresh.
pub(super) fn same_box(was_frame: u64, was_key: Option<FeatureId>, frame: u64, key: Option<FeatureId>) -> bool {
    frame <= was_frame + 1 && was_key == key
}

fn with_box<R>(ws: &ReverseWorkspace, frame: u64, f: impl FnOnce(&mut DraftBox) -> R) -> R {
    let key = box_key(ws);
    BOX.with(|b| {
        let mut b = b.borrow_mut();
        let fresh = !b.as_ref().is_some_and(|s| s.adopt || same_box(s.frame, s.key, frame, key));
        if fresh {
            *b = Some(DraftBox { frame, key, neutral_set: key.is_some(), adopt: false });
        }
        let s = b.get_or_insert(DraftBox { frame, key, neutral_set: key.is_some(), adopt: false });
        s.frame = frame;
        s.key = key;
        s.adopt = false;
        f(s)
    })
}

/// Whether the open draft's neutral plane is set, as the box last knew it.
fn neutral_set() -> bool {
    BOX.with(|b| b.borrow().as_ref().is_some_and(|s| s.neutral_set))
}

fn set_neutral(set: bool) {
    BOX.with(|b| {
        if let Some(s) = b.borrow_mut().as_mut() {
            s.neutral_set = set;
        }
    });
}

/// The body a new feature on a body starts on: the last solid body, else the last body.
pub(super) fn default_body(ws: &ReverseWorkspace) -> Option<BodyId> {
    ws.built.bodies.iter().rev().find(|b| !b.surface).or(ws.built.bodies.last()).map(|b| b.id).or_else(|| ws.bodies.last().map(|b| b.id))
}

pub(crate) fn new_feature(ws: &ReverseWorkspace) -> Result<FeatureKind, String> {
    let body = default_body(ws).ok_or("A draft leans faces of a body: make a body first.")?;
    Ok(FeatureKind::Draft(DraftFeature { body, neutral: MirrorPlane::Plane(PlaneRef::world(WorldAxis::Z, 0.0)), faces: Vec::new(), faces_where: Vec::new(), angle_deg: DEFAULT_DRAFT_DEG, groups: Vec::new(), exprs: Exprs::new() }))
}

// ---- fields the boxes share ----

/// An ⓘ beside a label, its card on hover: a title and a paragraph.
pub(super) fn info_dot(ui: &mut egui::Ui, title: &str, text: &str) {
    ui.label(RichText::new("ⓘ").weak()).on_hover_ui(|ui| {
        ui.set_max_width(320.0);
        ui.strong(title);
        ui.label(text);
    });
}

/// A nominal value as its chip writes it.
fn nominal_text(v: f64, kind: NominalKind) -> String {
    match kind {
        NominalKind::Angle => format!("{}°", mm_reverse::js_number::to_fixed(v, 0)),
        NominalKind::Draft => format!("{}°", mm_reverse::js_number::to_fixed(v, if v % 1.0 == 0.0 { 0 } else { 1 })),
        _ => fmt_len(v, DisplayUnits::Mm, if v % 1.0 == 0.0 { 0 } else { 1 }, FmtOptions::default()),
    }
}

/// A number of a feature, as the boxes give one: its label and ⓘ card, the round number it
/// was probably meant as (one click takes it), and the field. Returns whether it changed.
pub(super) fn num_field(ui: &mut egui::Ui, label: &str, value: &mut f64, unit: &str, nominal: Option<NominalKind>, hint: &str) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(label);
        if !hint.is_empty() {
            info_dot(ui, label, hint);
        }
        if let Some(n) = nominal.and_then(|k| nominal_of(*value, k).map(|n| (n, k))) {
            if ui.small_button(format!("→ {}", nominal_text(n.0, n.1))).on_hover_text("The round number this was probably meant as — click to take it").clicked() {
                *value = n.0;
                changed = true;
            }
        }
        let speed = if unit == "°" { 0.1 } else { 0.05 };
        changed |= ui.add(egui::DragValue::new(value).speed(speed).max_decimals(4).suffix(format!(" {unit}"))).changed();
    });
    changed
}

/// The body a feature works on, as a list of every body; changing it returns true.
pub(super) fn body_select(ui: &mut egui::Ui, ws: &ReverseWorkspace, body: &mut BodyId) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label("Body");
        let name = ws.built.body(*body).map_or_else(|| super::body_name(ws, *body), |b| b.name.clone());
        egui::ComboBox::from_id_salt("re_feature_body").selected_text(name).show_ui(ui, |ui| {
            for b in &ws.built.bodies {
                if ui.selectable_label(b.id == *body, &b.name).clicked() && b.id != *body {
                    *body = b.id;
                    changed = true;
                }
            }
        });
    });
    changed
}

/// The planes a feature can take: the sections, the measured planes and the coordinate planes
/// (at zero, and through the scan's centre where there is a scan) — each with its list label
/// and group.
fn plane_options(ws: &ReverseWorkspace) -> Vec<(String, &'static str, PlaneRef)> {
    let mut out = Vec::new();
    for e in &ws.part.elements {
        if let ElementGeometry::Section { .. } = e.geometry {
            out.push((e.name.clone(), "Sections", PlaneRef::Section { id: e.id, offset: None, up: None }));
        }
    }
    for e in &ws.part.elements {
        if let ElementGeometry::Plane { .. } = e.geometry {
            out.push((e.name.clone(), "Planes", PlaneRef::Plane { id: e.id, offset: None, up: None }));
        }
    }
    for axis in [WorldAxis::Z, WorldAxis::Y, WorldAxis::X] {
        out.push((axis.plane_name().to_string(), "Coordinate planes", PlaneRef::world(axis, 0.0)));
    }
    if let Some(c) = ws.scan_centre() {
        for axis in [WorldAxis::Z, WorldAxis::Y, WorldAxis::X] {
            let o = (c.dot(axis.dir()) * 1000.0).round() / 1000.0;
            out.push((format!("{} through the centre", axis.plane_name()), "Coordinate planes", PlaneRef::world(axis, o)));
        }
    }
    out
}

/// The plane picker: a list of the planes on offer under their groups, the placeholder shown
/// while none of them is the value. Returns the plane chosen.
pub(super) fn plane_select(ui: &mut egui::Ui, ws: &ReverseWorkspace, id: &str, value: Option<&PlaneRef>, placeholder: &str) -> Option<PlaneRef> {
    let options = plane_options(ws);
    let shown = value.and_then(|v| options.iter().find(|o| &o.2 == v)).map_or_else(|| placeholder.to_string(), |o| o.0.clone());
    let mut chosen = None;
    egui::ComboBox::from_id_salt(id).selected_text(shown).width(200.0).show_ui(ui, |ui| {
        let mut group = "";
        for (label, g, r) in &options {
            if *g != group {
                group = g;
                ui.label(RichText::new(*g).weak().small());
            }
            if ui.selectable_label(value == Some(r), label).clicked() {
                chosen = Some(r.clone());
            }
        }
    });
    chosen
}

/// The "Clear" key beside a pick count.
pub(super) fn clear_key(ui: &mut egui::Ui) -> bool {
    ui.small_button("Clear").on_hover_text("Let every picked one go and pick again").clicked()
}

// ---- the box ----

pub(crate) fn ui(ws: &mut ReverseWorkspace, ui: &mut egui::Ui, x: &mut DraftFeature, pick: &mut PickWhat) -> bool {
    let frame = ui.ctx().cumulative_frame_nr();
    let set = with_box(ws, frame, |s| s.neutral_set);
    let mut changed = false;
    if body_select(ui, ws, &mut x.body) {
        set_neutral(false);
        x.faces.clear();
        x.faces_where.clear();
        changed = true;
    }
    ui.horizontal(|ui| {
        ui.label("Neutral");
        info_dot(ui, "The neutral plane", "Where the leaned faces keep their place: they turn about the line in which they meet this plane. Click a flat face of the body — the face the part stands on in its mould, usually — or choose a plane. A positive angle leans the faces in as they go away from it.");
        let placeholder = match (&x.neutral, set) {
            (MirrorPlane::Face { face }, true) => describe_face(face),
            (MirrorPlane::Plane(PlaneRef::Fitted { label, .. }), true) => label.clone().unwrap_or_else(|| "a fitted plane".into()),
            (MirrorPlane::Plane(p), true) => describe_plane_ref(p, &ws.part),
            _ => "Click a flat face of the body, or choose…".into(),
        };
        let value = match (&x.neutral, set) {
            (MirrorPlane::Plane(p), true) => Some(p),
            _ => None,
        };
        if let Some(p) = plane_select(ui, ws, "re_draft_neutral", value, &placeholder) {
            x.neutral = MirrorPlane::Plane(p);
            set_neutral(true);
            changed = true;
        }
    });
    if neutral_set() && ui.button("Pick another neutral face").on_hover_text("Pick the neutral plane again — the next flat face clicked is it").clicked() {
        set_neutral(false);
        changed = true;
    }
    ui.horizontal_wrapped(|ui| {
        ui.label("Faces");
        let n = x.faces.len();
        ui.label(if !neutral_set() {
            "First the neutral face".to_string()
        } else if n == 0 {
            "Click the faces of the body to lean".to_string()
        } else {
            format!("{n} face{} picked", plural(n))
        });
        if n > 0 && clear_key(ui) {
            x.faces.clear();
            x.faces_where.clear();
            changed = true;
        }
    });
    let was = x.angle_deg;
    if num_field(ui, "Angle", &mut x.angle_deg, "°", Some(NominalKind::Angle), "Degrees the faces lean in as they go away from the neutral plane; negative leans them out") {
        if x.angle_deg != was {
            x.exprs.remove("angleDeg");
        }
        changed = true;
    }
    if !x.groups.is_empty() {
        let said: Vec<String> = x.groups.iter().map(|g| format!("{} face{} at {}°", g.faces.len(), plural(g.faces.len()), mm_reverse::round::number_text(mm_reverse::round::round_to(g.angle_deg, 2)))).collect();
        ui.weak(format!("Then {}.", said.join(", "))).on_hover_text("More faces at angles of their own, set through the commands (groups); kept as they are when this box is applied");
    }
    *pick = PickWhat::Faces;
    changed
}

/// A click on a face of a body: the first flat face (or one of another body) is the neutral
/// plane; later ones are leaned, a second click letting one go. The face under the cursor is
/// lit.
pub(crate) fn pick(ws: &mut ReverseWorkspace, vp: Viewport, p: Pos2, clicked: bool, x: &mut DraftFeature, _pick: &mut PickWhat) -> bool {
    let Some(crate::reverse::view::Hit::Body { body, face, .. }) = ws.pick_hit(vp, p) else { return false };
    ws.view.hover_face = Some((body, face));
    if !clicked {
        return false;
    }
    let Some(view) = ws.pick_view(body) else { return false };
    let Some(print) = view.topology.faces.get(face).map(|f| f.fingerprint) else { return false };
    let is_it = picked_as(view, face);
    toggle_face(x, body, print, neutral_set(), is_it)
}

/// Whether a face kept as a pick is the face `face` of a body as shown: the face itself, or the
/// other half of it (the kernel keeps a whole bore as two) — not merely a face on its surface.
pub(super) fn picked_as(view: &crate::reverse::BodyView, face: usize) -> impl Fn(&FaceRef) -> bool {
    let halves = halves_of(&view.topology, face);
    let prints = view.topology.face_prints();
    let tol = match_tolerance(view.mesh.bounds().diagonal());
    move |f: &FaceRef| match_face(f, &prints, &tol).is_some_and(|i| halves.contains(&i))
}

/// What a click on a face does to a draft (ScanRuler's `toggleDraftFace`): with no neutral
/// plane, or on another body, a flat face is the neutral plane (the faces kept on the same
/// body); otherwise the face is leaned, or let go when it is picked already (`is_it` says
/// whether a pick is the face clicked). Returns whether the draft changed.
fn toggle_face(x: &mut DraftFeature, body: BodyId, print: FaceRef, set: bool, is_it: impl Fn(&FaceRef) -> bool) -> bool {
    if !set || x.body != body {
        if print.kind != FaceKind::Plane {
            return false;
        }
        if x.body != body {
            x.faces.clear();
            x.faces_where.clear();
        }
        x.body = body;
        x.neutral = MirrorPlane::Face { face: print };
        set_neutral(true);
        return true;
    }
    match x.faces.iter().position(is_it) {
        Some(k) => {
            x.faces.remove(k);
        }
        None => x.faces.push(print),
    }
    true
}

pub(crate) fn paint(ws: &ReverseWorkspace, painter: &egui::Painter, vp: Viewport, x: &DraftFeature) {
    paint_faces(ws, painter, vp, x.body, &x.faces);
    // The neutral face, outlined in the colour a plane is hinted in.
    if let (MirrorPlane::Face { face }, true) = (&x.neutral, neutral_set()) {
        if let Some(b) = ws.pick_view(x.body) {
            let tol = match_tolerance(b.mesh.bounds().diagonal());
            if let Some(i) = match_face(face, &b.topology.face_prints(), &tol) {
                for e in b.topology.edges.iter().filter(|e| e.faces.contains(&(i as u32))) {
                    let pts: Vec<Pos2> = e.polyline.iter().filter_map(|q| ws.project(vp, *q)).collect();
                    if pts.len() >= 2 {
                        painter.add(egui::Shape::line(pts, Stroke::new(2.5, PLANE_HINT)));
                    }
                }
            }
        }
    }
    // Where the draft failed: the faces its diagnosis names, outlined red.
    if let Some(lines) = ws.preview_feature.and_then(|id| ws.built.failed_where.get(&id)) {
        for l in lines {
            let pts: Vec<Pos2> = l.iter().filter_map(|q| ws.project(vp, *q)).collect();
            if pts.len() >= 2 {
                painter.add(egui::Shape::line(pts, Stroke::new(3.0, Color32::from_rgb(0xe0, 0x56, 0x4f))));
            }
        }
    }
}

// ---- what Create waits for ----

/// Why the draft cannot be created yet (ScanRuler's `featureDraftReady`): no neutral plane, no
/// face to lean, or an angle that is none or not under 89°; `None` when it can.
fn unready(x: &DraftFeature, neutral: bool) -> Option<&'static str> {
    let bad = |deg: f64| !(deg.abs() > 1e-9) || !(deg.abs() < 89.0);
    if !neutral {
        return Some("Create waits for the neutral plane: click a flat face of the body, or choose a plane.");
    }
    if x.faces.is_empty() || x.groups.iter().any(|g| g.faces.is_empty()) {
        return Some("Create waits for the faces to lean: click them on the body.");
    }
    if bad(x.angle_deg) || x.groups.iter().any(|g| bad(g.angle_deg)) {
        return Some("Create waits for an angle: one that is not zero, and under 89°.");
    }
    None
}

/// Why the open Draft box's Create does nothing yet, if it does not.
pub(super) fn not_ready(x: &DraftFeature) -> Option<&'static str> {
    unready(x, neutral_set())
}

// ---- script steps (tests) ----

/// The faces of a body facing along a direction word (`+x`, `-z`, …), or a round one (`cyl`).
pub(super) fn faces_facing(ws: &ReverseWorkspace, body: BodyId, word: &str) -> Vec<FaceRef> {
    let Some(b) = ws.pick_view(body).or_else(|| ws.body_view(body)) else { return Vec::new() };
    let dir = match word {
        "+x" | "x" => Some(glam::DVec3::X),
        "-x" => Some(-glam::DVec3::X),
        "+y" | "y" => Some(glam::DVec3::Y),
        "-y" => Some(-glam::DVec3::Y),
        "+z" | "z" => Some(glam::DVec3::Z),
        "-z" => Some(-glam::DVec3::Z),
        _ => None,
    };
    let mut out: Vec<FaceRef> = Vec::new();
    for f in &b.topology.faces {
        let fp = f.fingerprint;
        let takes = match dir {
            Some(d) => fp.kind == FaceKind::Plane && fp.dir.dot(d) > 0.99,
            None => word == "cyl" && fp.kind == FaceKind::Cylinder && f.full_turn != Some(false),
        };
        if takes && !out.iter().any(|o| one_surface(o, &fp)) {
            out.push(fp);
        }
    }
    // The largest first: a block's top before a step's.
    out.sort_by(|a, b| b.area.total_cmp(&a.area));
    out
}

/// `draft:<angle>:<neutral face>:<face>,<face>…[:open]` — a draft on the last solid body, the
/// neutral plane the largest face facing one way and the faces the largest facing each other
/// way given, created at once (`open`: its box left open).
pub(super) fn script(ws: &mut ReverseWorkspace, args: &[&str]) {
    let Some(body) = default_body(ws) else {
        ws.status = "draft step: there is no body.".into();
        return;
    };
    let angle = args.first().and_then(|a| a.parse::<f64>().ok()).unwrap_or(DEFAULT_DRAFT_DEG);
    let Some(neutral) = args.get(1).and_then(|w| faces_facing(ws, body, w).into_iter().next()) else {
        ws.status = "draft step: no neutral face facing that way.".into();
        return;
    };
    let faces: Vec<FaceRef> = args.get(2).map(|list| list.split(',').filter_map(|w| faces_facing(ws, body, w).into_iter().next()).collect()).unwrap_or_default();
    start_about(ws, body, neutral, faces, angle);
    if args.get(3) != Some(&"open") {
        ws.commit_feature();
    }
}

/// A Draft box opened on a body about a flat face of it as the neutral plane, with the faces to
/// lean and the angle given (ScanRuler's `startFeature('draft', { body, neutral })` — the Draft
/// chip on a body face starts it with no face to lean and the default angle).
pub(super) fn start_about(ws: &mut ReverseWorkspace, body: BodyId, neutral: FaceRef, faces: Vec<FaceRef>, angle_deg: f64) {
    // The box's state first: the build the box asks for as it opens reads it.
    BOX.with(|b| *b.borrow_mut() = Some(DraftBox { frame: 0, key: None, neutral_set: true, adopt: true }));
    ws.open_feature(FeatureKind::Draft(DraftFeature { body, neutral: MirrorPlane::Face { face: neutral }, faces, faces_where: Vec::new(), angle_deg, groups: Vec::new(), exprs: Exprs::new() }), None);
}

/// The Draft chip on a flat face of a body: a Draft about it, the faces to lean clicked next.
pub(super) fn start_on_face(ws: &mut ReverseWorkspace, body: BodyId, neutral: FaceRef) {
    start_about(ws, body, neutral, Vec::new(), DEFAULT_DRAFT_DEG);
    ws.status = "Click the faces of the body to lean.".into();
}

/// The open Draft's angles read off the scan (ScanRuler's `angleDeg: "fit"`, and each group's
/// `groups.<i>.angleDeg: "fit"`): each face's lean fitted inside its outline, the median of a
/// group's faces taken, the readings said on the status line.
pub(super) fn fit_script(ws: &mut ReverseWorkspace) {
    let Mode::Feature(edit) = &ws.mode else { return };
    let FeatureKind::Draft(x) = &edit.kind else { return };
    let x = x.clone();
    let pull = match &x.neutral {
        MirrorPlane::Face { face } => face.dir.try_normalize().map(|n| (face.centroid, n)),
        MirrorPlane::Plane(p) => mm_reverse::build::resolve_plane(p, &ws.part),
    };
    let Some(pull) = pull.filter(|_| neutral_set()) else {
        ws.status = "A fitted draft needs its neutral plane.".into();
        return;
    };
    // Each face with its triangles off the body's mesh, its outline.
    let faces_in = |faces: &[FaceRef]| -> Vec<DraftFaceIn> {
        faces
            .iter()
            .map(|f| {
                let triangles = ws.pick_view(x.body).and_then(|b| {
                    let i = match_face(f, &b.topology.face_prints(), &match_tolerance(b.mesh.bounds().diagonal()))? as u32;
                    let m = &b.part_mesh;
                    let tris: Vec<[glam::DVec3; 3]> = m.triangles.iter().zip(&m.tri_face).filter(|(_, t)| **t == i).map(|(t, _)| t.map(|v| m.positions.get(v as usize).copied().unwrap_or_default())).collect();
                    (!tris.is_empty()).then_some(tris)
                });
                DraftFaceIn { fingerprint: *f, triangles }
            })
            .collect()
    };
    let main = faces_in(&x.faces);
    let groups: Vec<Vec<DraftFaceIn>> = x.groups.iter().map(|g| faces_in(&g.faces)).collect();
    // The scan's points and their normals in the part's coordinates.
    let flip = ws.flip;
    let out = |p: glam::DVec3| if flip { crate::app::flip_zy_out(p) } else { p };
    let mut points: Vec<(glam::DVec3, glam::DVec3)> = Vec::new();
    for o in ws.scan.iter().filter(|o| o.visible) {
        let normals = o.mesh.compute_vertex_normals();
        for v in o.mesh.vertex_ids() {
            points.push((out(o.mesh.vertex(v)), out(normals.get(v as usize).copied().unwrap_or_default())));
        }
    }
    let scan_in = |lo: glam::DVec3, hi: glam::DVec3, limit: usize| {
        let inside: Vec<&(glam::DVec3, glam::DVec3)> = points.iter().filter(|(p, _)| p.cmpge(lo).all() && p.cmple(hi).all()).collect();
        let step = inside.len().div_ceil(limit.max(1)).max(1);
        inside.into_iter().step_by(step).map(|(p, n)| (*p, *n)).unzip()
    };
    let said = |r: &mm_reverse::draft_fit::FitReading| {
        let coverage = r.coverage.map_or(String::new(), |c| format!(", the scan spanning {} % of the faces' height", mm_reverse::round::js_round(c * 100.0)));
        format!("{}° ± {}° (nominal {}°) over {} points{coverage}.{}", mm_reverse::round::number_text(r.value), mm_reverse::round::number_text(r.sigma), mm_reverse::round::number_text(r.nominal), r.points, r.warning.as_ref().map_or(String::new(), |w| format!(" {w}")))
    };
    let mut fitted: Vec<(Option<usize>, f64)> = Vec::new();
    let mut lines: Vec<String> = Vec::new();
    match fit_draft_faces(&main, pull, DRAFT_ANGLE, &scan_in) {
        Err(e) => lines.push(e),
        Ok(r) => {
            lines.push(format!("Draft angle fitted: {}", said(&r)));
            fitted.push((None, r.value));
        }
    }
    for (i, g) in groups.iter().enumerate() {
        match fit_draft_faces(g, pull, &format!("angle of group {}", i + 1), &scan_in) {
            Err(e) => lines.push(e),
            Ok(r) => {
                lines.push(format!("Group {} fitted: {}", i + 1, said(&r)));
                fitted.push((Some(i), r.value));
            }
        }
    }
    ws.status = lines.join(" ");
    if !fitted.is_empty() {
        if let Mode::Feature(edit) = &mut ws.mode {
            if let FeatureKind::Draft(d) = &mut edit.kind {
                for (which, v) in fitted {
                    match which {
                        None => {
                            d.angle_deg = v;
                            d.exprs.remove("angleDeg");
                        }
                        Some(i) => {
                            if let Some(g) = d.groups.get_mut(i) {
                                g.angle_deg = v;
                            }
                        }
                    }
                }
            }
        }
        ws.request_build();
    }
    log::info!("reverse draft fit: {}", ws.status);
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec3;

    fn face(dir: DVec3, kind: FaceKind) -> FaceRef {
        FaceRef { kind, centroid: dir * 5.0, dir, area: 50.0, radius: (kind == FaceKind::Cylinder).then_some(3.0), center: (kind == FaceKind::Cylinder).then_some(DVec3::ZERO), origin: 2 }
    }

    fn draft() -> DraftFeature {
        DraftFeature { body: BodyId::new(2), neutral: MirrorPlane::Plane(PlaneRef::world(WorldAxis::Z, 0.0)), faces: Vec::new(), faces_where: Vec::new(), angle_deg: DEFAULT_DRAFT_DEG, groups: Vec::new(), exprs: Exprs::new() }
    }

    #[test]
    fn takes_the_first_flat_face_as_its_neutral_plane_and_the_faces_after_it_as_the_ones_leaned() {
        // ScanRuler's reFeatureDraft: 'a draft takes the first flat face as its neutral plane'.
        let mut d = draft();
        let round = face(DVec3::Z, FaceKind::Cylinder);
        let same = |p: FaceRef| move |f: &FaceRef| *f == p;
        // A round face cannot be the neutral plane.
        assert!(!toggle_face(&mut d, BodyId::new(2), round, false, same(round)));
        let bottom = face(-DVec3::Z, FaceKind::Plane);
        assert!(toggle_face(&mut d, BodyId::new(2), bottom, false, same(bottom)));
        assert!(matches!(d.neutral, MirrorPlane::Face { .. }));
        let side = face(DVec3::X, FaceKind::Plane);
        assert!(toggle_face(&mut d, BodyId::new(2), side, true, same(side)));
        assert!(toggle_face(&mut d, BodyId::new(2), round, true, same(round)));
        assert_eq!(d.faces.len(), 2);
        // A second wall on the plane of the first, apart from it, is a face of its own.
        let beside = FaceRef { centroid: side.centroid + DVec3::Y * 30.0, ..side };
        assert!(toggle_face(&mut d, BodyId::new(2), beside, true, same(beside)));
        assert_eq!(d.faces.len(), 3);
        assert!(toggle_face(&mut d, BodyId::new(2), beside, true, same(beside)));
        // Clicked again, a face is let go.
        assert!(toggle_face(&mut d, BodyId::new(2), round, true, same(round)));
        assert_eq!(d.faces.len(), 1);
        assert_eq!(d.faces[0].dir, DVec3::X);
        // A face of another body starts again on it, as its neutral plane.
        let other = face(DVec3::Z, FaceKind::Plane);
        assert!(toggle_face(&mut d, BodyId::new(5), other, true, same(other)));
        assert_eq!(d.body, BodyId::new(5));
        assert!(d.faces.is_empty());
    }

    #[test]
    fn create_waits_for_the_neutral_plane_the_faces_and_an_angle_under_89() {
        // ScanRuler's featureDraftReady for a draft.
        let mut d = draft();
        d.faces.push(face(DVec3::X, FaceKind::Plane));
        assert!(unready(&d, false).is_some_and(|w| w.contains("neutral")));
        assert_eq!(unready(&d, true), None);
        d.angle_deg = 89.0;
        assert!(unready(&d, true).is_some_and(|w| w.contains("89")));
        d.angle_deg = -3.0;
        assert_eq!(unready(&d, true), None);
        d.groups.push(mm_reverse::model::DraftGroup { faces: Vec::new(), angle_deg: 2.0 });
        assert!(unready(&d, true).is_some());
        d.groups.clear();
        d.faces.clear();
        assert!(unready(&d, true).is_some_and(|w| w.contains("faces")));
    }

    #[test]
    fn a_box_is_its_own_while_it_is_shown_frame_after_frame_for_one_feature() {
        assert!(same_box(10, None, 11, None));
        assert!(!same_box(10, None, 13, None));
        assert!(!same_box(10, Some(3), 11, Some(4)));
        assert!(same_box(10, Some(3), 10, Some(3)));
    }
}
