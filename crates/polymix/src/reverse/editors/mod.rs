// SPDX-License-Identifier: AGPL-3.0-only
//! The feature boxes of the Reverse Engineer workspace, one file per feature kind: each kind's
//! editor (what its box shows), what a click in the viewport picks for it, what it paints over
//! the part, the feature a button in the panel starts, and the button itself. A port of the
//! feature boxes of ScanRuler's Reverse engineer workspace (`plugins/reverse/ui/`), © Stefan
//! Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! This module only dispatches by kind and holds what several editors share (the sketch and
//! region lines, the result line, picking and painting edges, faces and regions); a kind's own
//! editor lives in `editors/<kind>.rs` and is changed there alone. A kind whose editor is not
//! built yet says so in one line, and its button in the panel waits under "Not built yet".

mod boolean;
mod chamfer;
mod cut_plane;
mod delete_face;
mod draft;
mod extract;
mod extrude;
mod fillet;
mod hole;
mod import;
mod loft;
mod loft_kit;
mod loft_stations;
mod mirror;
mod move_body;
mod move_face;
mod network;
mod offset_body;
mod pattern;
mod revolve;
mod sheet;
mod shell;
mod split;
mod surface;
mod sweep;
mod thicken;
mod thread;
mod trim;
// ---- SURF ----
pub(crate) mod surf_common;
mod surf_script;

use egui::{Color32, Pos2, Stroke};
use glam::DVec3;
use mm_reverse::model::{BodyId, BodyOp, EdgeRef, FaceRef, FeatureId, FeatureKind, RegionPick};
use mm_reverse::topo::{edge_distance, face_distance, match_edge, match_face, match_tolerance, MatchTolerance};

use super::sketcher::colors;
use super::view::PICKED;
use super::{FeatureEdit, PickWhat, ReverseWorkspace};
use crate::tools::Viewport;

/// The kinds a button in the panel starts, in the order they stand there (Extrude and Revolve
/// stand first, by the sketch).
pub(crate) const NEW_FROM_PANEL: [&str; 24] = [
    "fillet", "chamfer", "shell", "mirror", "boolean", "cutPlane", "hole", "thread", "pattern", "moveBody", "draft", "moveFace", "deleteFace", "offsetBody", "extract", "import", "sheet", "loft", "sweep", "thicken", "split", "surface", "network", "trim",
];

/// A kind's button in the panel: its label and what it does.
pub(crate) fn button(kind: &str) -> (&'static str, &'static str) {
    match kind {
        "fillet" => fillet::BUTTON,
        "chamfer" => chamfer::BUTTON,
        "shell" => shell::BUTTON,
        "mirror" => mirror::BUTTON,
        "boolean" => boolean::BUTTON,
        "cutPlane" => cut_plane::BUTTON,
        "hole" => hole::BUTTON,
        "thread" => thread::BUTTON,
        "pattern" => pattern::BUTTON,
        "moveBody" => move_body::BUTTON,
        "draft" => draft::BUTTON,
        "moveFace" => move_face::BUTTON,
        "deleteFace" => delete_face::BUTTON,
        "offsetBody" => offset_body::BUTTON,
        "extract" => extract::BUTTON,
        "import" => import::BUTTON,
        "sheet" => sheet::BUTTON,
        "loft" => loft::BUTTON,
        "sweep" => sweep::BUTTON,
        "thicken" => thicken::BUTTON,
        "split" => split::BUTTON,
        "surface" => surface::BUTTON,
        "network" => network::BUTTON,
        "trim" => trim::BUTTON,
        _ => ("?", ""),
    }
}

/// Whether a kind's editor is built: its button stands with the others.
pub(crate) fn ready(kind: &str) -> bool {
    match kind {
        "fillet" => fillet::READY,
        "chamfer" => chamfer::READY,
        "shell" => shell::READY,
        "mirror" => mirror::READY,
        "boolean" => boolean::READY,
        "cutPlane" => cut_plane::READY,
        "hole" => hole::READY,
        "thread" => thread::READY,
        "pattern" => pattern::READY,
        "moveBody" => move_body::READY,
        "draft" => draft::READY,
        "moveFace" => move_face::READY,
        "deleteFace" => delete_face::READY,
        "offsetBody" => offset_body::READY,
        "extract" => extract::READY,
        "import" => import::READY,
        "sheet" => sheet::READY,
        "loft" => loft::READY,
        "sweep" => sweep::READY,
        "thicken" => thicken::READY,
        "split" => split::READY,
        "surface" => surface::READY,
        "network" => network::READY,
        "trim" => trim::READY,
        _ => false,
    }
}

/// The feature a kind's button starts, as the part stands — or why it cannot start one now.
pub(crate) fn new_feature(kind: &str, ws: &ReverseWorkspace) -> Result<FeatureKind, String> {
    match kind {
        "fillet" => fillet::new_feature(ws),
        "chamfer" => chamfer::new_feature(ws),
        "shell" => shell::new_feature(ws),
        "mirror" => mirror::new_feature(ws),
        "boolean" => boolean::new_feature(ws),
        "cutPlane" => cut_plane::new_feature(ws),
        "hole" => hole::new_feature(ws),
        "thread" => thread::new_feature(ws),
        "pattern" => pattern::new_feature(ws),
        "moveBody" => move_body::new_feature(ws),
        "draft" => draft::new_feature(ws),
        "moveFace" => move_face::new_feature(ws),
        "deleteFace" => delete_face::new_feature(ws),
        "offsetBody" => offset_body::new_feature(ws),
        "extract" => extract::new_feature(ws),
        "import" => import::new_feature(ws),
        "sheet" => sheet::new_feature(ws),
        "loft" => loft::new_feature(ws),
        "sweep" => sweep::new_feature(ws),
        "thicken" => thicken::new_feature(ws),
        "split" => split::new_feature(ws),
        "surface" => surface::new_feature(ws),
        "network" => network::new_feature(ws),
        "trim" => trim::new_feature(ws),
        other => Err(format!("No feature is called {other}.")),
    }
}

/// What a click picks first when a kind's box opens.
pub(crate) fn first_pick(kind: &FeatureKind) -> PickWhat {
    match kind {
        FeatureKind::Sketch(_) => PickWhat::Nothing,
        FeatureKind::Extrude(_) => extrude::FIRST_PICK,
        FeatureKind::Revolve(_) => revolve::FIRST_PICK,
        FeatureKind::Extract(_) => extract::FIRST_PICK,
        FeatureKind::Sheet(_) => sheet::FIRST_PICK,
        FeatureKind::Loft(_) => loft::FIRST_PICK,
        FeatureKind::Sweep(_) => sweep::FIRST_PICK,
        FeatureKind::Thicken(_) => thicken::FIRST_PICK,
        FeatureKind::Surface(_) => surface::FIRST_PICK,
        FeatureKind::Network(_) => network::FIRST_PICK,
        FeatureKind::Trim(_) => trim::FIRST_PICK,
        FeatureKind::Import(_) => import::FIRST_PICK,
        FeatureKind::MoveBody(_) => move_body::FIRST_PICK,
        FeatureKind::Shell(_) => shell::FIRST_PICK,
        FeatureKind::Fillet(_) => fillet::FIRST_PICK,
        FeatureKind::Chamfer(_) => chamfer::FIRST_PICK,
        FeatureKind::Hole(_) => hole::FIRST_PICK,
        FeatureKind::Thread(_) => thread::FIRST_PICK,
        FeatureKind::CutPlane(_) => cut_plane::FIRST_PICK,
        FeatureKind::Mirror(_) => mirror::FIRST_PICK,
        FeatureKind::Boolean(_) => boolean::FIRST_PICK,
        FeatureKind::Split(_) => split::FIRST_PICK,
        FeatureKind::Pattern(_) => pattern::FIRST_PICK,
        FeatureKind::Draft(_) => draft::FIRST_PICK,
        FeatureKind::MoveFace(_) => move_face::FIRST_PICK,
        FeatureKind::DeleteFace(_) => delete_face::FIRST_PICK,
        FeatureKind::OffsetBody(_) => offset_body::FIRST_PICK,
    }
}

/// The open feature's editor, in the panel. Returns whether the feature changed.
pub(crate) fn ui(ws: &mut ReverseWorkspace, ui: &mut egui::Ui, edit: &mut FeatureEdit) -> bool {
    let pick = &mut edit.pick;
    match &mut edit.kind {
        FeatureKind::Sketch(_) => false,
        FeatureKind::Extrude(x) => extrude::ui(ws, ui, x, pick),
        FeatureKind::Revolve(x) => revolve::ui(ws, ui, x, pick),
        FeatureKind::Extract(x) => extract::ui(ws, ui, x, pick),
        FeatureKind::Sheet(x) => sheet::ui(ws, ui, x, pick),
        FeatureKind::Loft(x) => loft::ui(ws, ui, x, pick),
        FeatureKind::Sweep(x) => sweep::ui(ws, ui, x, pick),
        FeatureKind::Thicken(x) => thicken::ui(ws, ui, x, pick),
        FeatureKind::Surface(x) => surface::ui(ws, ui, x, pick),
        FeatureKind::Network(x) => network::ui(ws, ui, x, pick),
        FeatureKind::Trim(x) => trim::ui(ws, ui, x, pick),
        FeatureKind::Import(x) => import::ui(ws, ui, x, pick),
        FeatureKind::MoveBody(x) => move_body::ui(ws, ui, x, pick),
        FeatureKind::Shell(x) => shell::ui(ws, ui, x, pick),
        FeatureKind::Fillet(x) => fillet::ui(ws, ui, x, pick),
        FeatureKind::Chamfer(x) => chamfer::ui(ws, ui, x, pick),
        FeatureKind::Hole(x) => hole::ui(ws, ui, x, pick),
        FeatureKind::Thread(x) => thread::ui(ws, ui, x, pick),
        FeatureKind::CutPlane(x) => cut_plane::ui(ws, ui, x, pick),
        FeatureKind::Mirror(x) => mirror::ui(ws, ui, x, pick),
        FeatureKind::Boolean(x) => boolean::ui(ws, ui, x, pick),
        FeatureKind::Split(x) => split::ui(ws, ui, x, pick),
        FeatureKind::Pattern(x) => pattern::ui(ws, ui, x, pick),
        FeatureKind::Draft(x) => draft::ui(ws, ui, x, pick),
        FeatureKind::MoveFace(x) => move_face::ui(ws, ui, x, pick),
        FeatureKind::DeleteFace(x) => delete_face::ui(ws, ui, x, pick),
        FeatureKind::OffsetBody(x) => offset_body::ui(ws, ui, x, pick),
    }
}

/// A click (or the hover) in the viewport while a feature box is open, by the kind's editor.
/// Returns whether the feature changed.
pub(crate) fn pick(ws: &mut ReverseWorkspace, vp: Viewport, p: Pos2, clicked: bool, edit: &mut FeatureEdit) -> bool {
    let pick = &mut edit.pick;
    match &mut edit.kind {
        FeatureKind::Sketch(_) => false,
        FeatureKind::Extrude(x) => extrude::pick(ws, vp, p, clicked, x, pick),
        FeatureKind::Revolve(x) => revolve::pick(ws, vp, p, clicked, x, pick),
        FeatureKind::Extract(x) => extract::pick(ws, vp, p, clicked, x, pick),
        FeatureKind::Sheet(x) => sheet::pick(ws, vp, p, clicked, x, pick),
        FeatureKind::Loft(x) => loft::pick(ws, vp, p, clicked, x, pick),
        FeatureKind::Sweep(x) => sweep::pick(ws, vp, p, clicked, x, pick),
        FeatureKind::Thicken(x) => thicken::pick(ws, vp, p, clicked, x, pick),
        FeatureKind::Surface(x) => surface::pick(ws, vp, p, clicked, x, pick),
        FeatureKind::Network(x) => network::pick(ws, vp, p, clicked, x, pick),
        FeatureKind::Trim(x) => trim::pick(ws, vp, p, clicked, x, pick),
        FeatureKind::Import(x) => import::pick(ws, vp, p, clicked, x, pick),
        FeatureKind::MoveBody(x) => move_body::pick(ws, vp, p, clicked, x, pick),
        FeatureKind::Shell(x) => shell::pick(ws, vp, p, clicked, x, pick),
        FeatureKind::Fillet(x) => fillet::pick(ws, vp, p, clicked, x, pick),
        FeatureKind::Chamfer(x) => chamfer::pick(ws, vp, p, clicked, x, pick),
        FeatureKind::Hole(x) => hole::pick(ws, vp, p, clicked, x, pick),
        FeatureKind::Thread(x) => thread::pick(ws, vp, p, clicked, x, pick),
        FeatureKind::CutPlane(x) => cut_plane::pick(ws, vp, p, clicked, x, pick),
        FeatureKind::Mirror(x) => mirror::pick(ws, vp, p, clicked, x, pick),
        FeatureKind::Boolean(x) => boolean::pick(ws, vp, p, clicked, x, pick),
        FeatureKind::Split(x) => split::pick(ws, vp, p, clicked, x, pick),
        FeatureKind::Pattern(x) => pattern::pick(ws, vp, p, clicked, x, pick),
        FeatureKind::Draft(x) => draft::pick(ws, vp, p, clicked, x, pick),
        FeatureKind::MoveFace(x) => move_face::pick(ws, vp, p, clicked, x, pick),
        FeatureKind::DeleteFace(x) => delete_face::pick(ws, vp, p, clicked, x, pick),
        FeatureKind::OffsetBody(x) => offset_body::pick(ws, vp, p, clicked, x, pick),
    }
}

/// What the open feature box paints over the part: its picks, by the kind's editor.
pub(crate) fn paint(ws: &ReverseWorkspace, painter: &egui::Painter, vp: Viewport, edit: &FeatureEdit) {
    match &edit.kind {
        FeatureKind::Sketch(_) => {}
        FeatureKind::Extrude(x) => extrude::paint(ws, painter, vp, x),
        FeatureKind::Revolve(x) => revolve::paint(ws, painter, vp, x),
        FeatureKind::Extract(x) => extract::paint(ws, painter, vp, x),
        FeatureKind::Sheet(x) => sheet::paint(ws, painter, vp, x),
        FeatureKind::Loft(x) => loft::paint(ws, painter, vp, x),
        FeatureKind::Sweep(x) => sweep::paint(ws, painter, vp, x),
        FeatureKind::Thicken(x) => thicken::paint(ws, painter, vp, x),
        FeatureKind::Surface(x) => surface::paint(ws, painter, vp, x),
        FeatureKind::Network(x) => network::paint(ws, painter, vp, x),
        FeatureKind::Trim(x) => trim::paint(ws, painter, vp, x),
        FeatureKind::Import(x) => import::paint(ws, painter, vp, x),
        FeatureKind::MoveBody(x) => move_body::paint(ws, painter, vp, x),
        FeatureKind::Shell(x) => shell::paint(ws, painter, vp, x),
        FeatureKind::Fillet(x) => fillet::paint(ws, painter, vp, x),
        FeatureKind::Chamfer(x) => chamfer::paint(ws, painter, vp, x),
        FeatureKind::Hole(x) => hole::paint(ws, painter, vp, x),
        FeatureKind::Thread(x) => thread::paint(ws, painter, vp, x),
        FeatureKind::CutPlane(x) => cut_plane::paint(ws, painter, vp, x),
        FeatureKind::Mirror(x) => mirror::paint(ws, painter, vp, x),
        FeatureKind::Boolean(x) => boolean::paint(ws, painter, vp, x),
        FeatureKind::Split(x) => split::paint(ws, painter, vp, x),
        FeatureKind::Pattern(x) => pattern::paint(ws, painter, vp, x),
        FeatureKind::Draft(x) => draft::paint(ws, painter, vp, x),
        FeatureKind::MoveFace(x) => move_face::paint(ws, painter, vp, x),
        FeatureKind::DeleteFace(x) => delete_face::paint(ws, painter, vp, x),
        FeatureKind::OffsetBody(x) => offset_body::paint(ws, painter, vp, x),
    }
}

/// Whether the open feature cuts: its preview drawn in the cut's colour.
pub(crate) fn cuts(kind: &FeatureKind) -> bool {
    let op = match kind {
        FeatureKind::Extrude(x) => Some(x.op),
        FeatureKind::Revolve(x) => Some(x.op),
        FeatureKind::Extract(x) => Some(x.op),
        FeatureKind::Loft(x) => Some(x.op),
        FeatureKind::Sweep(x) => Some(x.op),
        FeatureKind::Thicken(x) => Some(x.op),
        _ => None,
    };
    op == Some(BodyOp::Cut)
}

// ---- what the editors share ----

/// A body's name in the list — or as it stood before the open feature, where the feature takes
/// it (a Combine's tool) — or its id where it is not standing.
pub(crate) fn body_name(ws: &ReverseWorkspace, id: BodyId) -> String {
    ws.body_view(id).or_else(|| ws.pick_view(id)).map_or_else(|| format!("{id} (gone)"), |b| b.name.clone())
}

/// The first body shown, for a new feature on a body.
#[allow(dead_code)] // the body boxes now pick their own (RND, BOOL)
pub(crate) fn first_body(ws: &ReverseWorkspace) -> Option<BodyId> {
    ws.bodies.first().map(|b| b.id)
}

/// A sketch to take regions from; a change of sketch takes the new one's regions.
#[allow(dead_code)] // the boxes that pick a sketch's regions; Extrude and Revolve have their own (EXT)
pub(crate) fn sketch_choice(ui: &mut egui::Ui, ws: &ReverseWorkspace, sketch: &mut FeatureId, regions: &mut Vec<String>, region_pick: &mut Option<RegionPick>) -> bool {
    let sketches: Vec<(FeatureId, String)> = ws.part.features.iter().filter(|f| matches!(f.kind, FeatureKind::Sketch(_))).map(|f| (f.id, f.name.clone())).collect();
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label("Sketch");
        let name = sketches.iter().find(|s| s.0 == *sketch).map_or("(gone)".to_string(), |s| s.1.clone());
        egui::ComboBox::from_id_salt("re_feature_sketch").selected_text(name).show_ui(ui, |ui| {
            for (id, n) in &sketches {
                if ui.selectable_value(sketch, *id, n).changed() {
                    changed = true;
                }
            }
        });
    });
    if changed {
        *regions = ws.default_regions(*sketch);
        *region_pick = None;
    }
    changed
}

/// How many of a sketch's regions are picked, with a button to take every one that is no hole.
#[allow(dead_code)] // the boxes that pick a sketch's regions; Extrude and Revolve have their own (EXT)
pub(crate) fn regions_line(ui: &mut egui::Ui, ws: &ReverseWorkspace, sketch: FeatureId, regions: &mut Vec<String>, region_pick: &mut Option<RegionPick>) -> bool {
    let total = ws.built.sketches.get(&sketch).map_or(0, |s| s.regions.len());
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(format!("Regions: {} of {total}", regions.len()));
        if ui.small_button("All").on_hover_text("Every region that is no hole of another").clicked() {
            *regions = ws.default_regions(sketch);
            *region_pick = None;
            changed = true;
        }
    });
    ui.weak("Click regions on the sketch to pick them; an inner loop left unpicked becomes a hole.");
    changed
}

/// The result of a solid feature: joined, cut, or a body of its own.
#[allow(dead_code)] // the boxes that pick a sketch's regions; Extrude and Revolve have their own (EXT)
pub(crate) fn op_line(ui: &mut egui::Ui, op: &mut BodyOp) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        for o in BodyOp::ALL {
            changed |= ui.selectable_value(op, o, o.name()).changed();
        }
    });
    changed
}

/// The body a feature works on, named, with what a click on the part does.
#[allow(dead_code)] // the body boxes now show their own (RND, BOOL)
pub(crate) fn body_line(ui: &mut egui::Ui, ws: &ReverseWorkspace, body: BodyId, hint: &str) {
    ui.horizontal(|ui| {
        ui.label("Body");
        ui.strong(body_name(ws, body)).on_hover_text(body.to_string());
    });
    ui.weak(hint);
}

/// A click on a region of a sketch picks it, a second lets it go; the region under the cursor
/// is lit. A pick by hand lets a pick kept as given go.
#[allow(dead_code)] // the boxes that pick a sketch's regions; Extrude and Revolve have their own (EXT)
pub(crate) fn pick_region(ws: &mut ReverseWorkspace, vp: Viewport, p: Pos2, clicked: bool, sketch: FeatureId, regions: &mut Vec<String>, region_pick: &mut Option<RegionPick>) -> bool {
    let region = ws.region_of_sketch_at(vp, p, sketch);
    ws.view.hover_region = region.clone();
    let (true, Some(key)) = (clicked, region) else { return false };
    match regions.iter().position(|k| *k == key) {
        Some(i) => {
            regions.remove(i);
        }
        None => regions.push(key),
    }
    *region_pick = None;
    true
}

/// A sketch's picked regions and the one under the cursor, outlined on its plane.
#[allow(dead_code)] // the boxes that pick a sketch's regions; Extrude and Revolve have their own (EXT)
pub(crate) fn paint_regions(ws: &ReverseWorkspace, painter: &egui::Painter, vp: Viewport, sketch: FeatureId, picked: &[String]) {
    let Some(frame) = ws.sketch_frame(sketch) else { return };
    let Some(built) = ws.built.sketches.get(&sketch) else { return };
    for r in &built.regions {
        let chosen = picked.contains(&r.key);
        let hovered = ws.view.hover_region.as_deref() == Some(r.key.as_str());
        if !chosen && !hovered {
            continue;
        }
        let color = if hovered { colors::HOVER } else { PICKED };
        for l in r.outline(0.05) {
            let pts: Vec<Pos2> = l.iter().filter_map(|q| ws.project(vp, frame.to_world(*q))).collect();
            if pts.len() >= 2 {
                painter.add(egui::Shape::closed_line(pts, Stroke::new(if hovered { 3.0 } else { 2.0 }, color)));
            }
        }
    }
}

/// How closely two fingerprints of one face or edge, read off two builds of the same body,
/// agree: a hair of the part.
fn same_tolerance(ws: &ReverseWorkspace, body: BodyId) -> MatchTolerance {
    let size = ws.pick_view(body).map_or(100.0, |b| {
        let bb = b.mesh.bounds();
        if bb.is_empty() {
            100.0
        } else {
            bb.diagonal()
        }
    });
    MatchTolerance { position: 1e-4 * size.max(1.0), angle: 1e-3, size: 1e-3 }
}

/// A click on an edge picks it as its fingerprint, a second click lets it go; an edge of another
/// body starts the picks again on that body. The edge under the cursor is lit. Edges are picked
/// on the body as it stood before the feature (see `ReverseWorkspace::pick_set`): an edge the
/// preview has rounded away is still there to let go of, and no edge the feature itself makes
/// can be picked.
pub(crate) fn pick_edge(ws: &mut ReverseWorkspace, vp: Viewport, p: Pos2, clicked: bool, body: &mut BodyId, edges: &mut Vec<EdgeRef>) -> bool {
    let edge = ws.edge_at(vp, p);
    ws.view.hover_edge = edge;
    let (true, Some((b, i))) = (clicked, edge) else { return false };
    let Some(print) = ws.pick_view(b).and_then(|v| v.topology.edges.get(i)).map(|e| e.fingerprint.clone()) else { return false };
    if b != *body {
        *body = b;
        edges.clear();
    }
    let tol = same_tolerance(ws, b);
    match edges.iter().position(|e| edge_distance(e, &print, &tol).is_some()) {
        Some(k) => {
            edges.remove(k);
        }
        None => edges.push(print),
    }
    true
}

/// A click on a face picks it as its fingerprint, a second click lets it go; a face of another
/// body starts the picks again on that body. The face under the cursor is lit. Faces are picked
/// on the body as it stood before the feature, as edges are.
pub(crate) fn pick_face(ws: &mut ReverseWorkspace, vp: Viewport, p: Pos2, clicked: bool, body: &mut BodyId, faces: &mut Vec<FaceRef>) -> bool {
    let Some(super::view::Hit::Body { body: b, face, .. }) = ws.pick_hit(vp, p) else { return false };
    ws.view.hover_face = Some((b, face));
    if !clicked {
        return false;
    }
    let Some(print) = ws.pick_view(b).and_then(|v| v.topology.faces.get(face)).map(|f| f.fingerprint) else { return false };
    if b != *body {
        *body = b;
        faces.clear();
    }
    let tol = same_tolerance(ws, b);
    match faces.iter().position(|f| face_distance(f, &print, &tol).is_some()) {
        Some(k) => {
            faces.remove(k);
        }
        None => faces.push(print),
    }
    true
}

/// A click on a body picks it — among the bodies as they stood before the feature, so a
/// Combine's tool its preview has taken in is still there to click.
#[allow(dead_code)]
pub(crate) fn pick_body(ws: &mut ReverseWorkspace, vp: Viewport, p: Pos2, clicked: bool, body: &mut BodyId) -> bool {
    if !clicked {
        return false;
    }
    match ws.pick_hit(vp, p) {
        Some(super::view::Hit::Body { body: b, .. }) if b != *body => {
            *body = b;
            true
        }
        _ => false,
    }
}

/// Picked edges, drawn where they are found on the body as it stood before the feature — a dot
/// at a pick's middle where it is not found there.
pub(crate) fn paint_edges(ws: &ReverseWorkspace, painter: &egui::Painter, vp: Viewport, body: BodyId, edges: &[EdgeRef]) {
    let shown = ws.pick_view(body);
    for e in edges {
        let found = shown.and_then(|b| {
            let tol = match_tolerance(b.mesh.bounds().diagonal());
            let prints = b.topology.edge_prints();
            match_edge(e, &prints, &tol).and_then(|i| b.topology.edges.get(i)).filter(|x| edge_distance(e, &x.fingerprint, &tol).is_some())
        });
        match found {
            Some(x) => {
                let pts: Vec<Pos2> = x.polyline.iter().filter_map(|q| ws.project(vp, *q)).collect();
                painter.add(egui::Shape::line(pts, Stroke::new(3.0, PICKED)));
            }
            None => dot(ws, painter, vp, e.mid),
        }
    }
}

/// Picked faces, a dot on each where it is found on the body as it stood before the feature,
/// at its centroid where not.
pub(crate) fn paint_faces(ws: &ReverseWorkspace, painter: &egui::Painter, vp: Viewport, body: BodyId, faces: &[FaceRef]) {
    let shown = ws.pick_view(body);
    for f in faces {
        let at = shown
            .and_then(|b| {
                let tol = match_tolerance(b.mesh.bounds().diagonal());
                match_face(f, &b.topology.face_prints(), &tol).and_then(|i| b.topology.faces.get(i)).map(|x| x.point)
            })
            .unwrap_or(f.centroid);
        dot(ws, painter, vp, at);
    }
}

fn dot(ws: &ReverseWorkspace, painter: &egui::Painter, vp: Viewport, at: DVec3) {
    if let Some(q) = ws.project(vp, at) {
        painter.circle_filled(q, 4.0, PICKED);
        painter.circle_stroke(q, 4.0, Stroke::new(1.0, Color32::BLACK));
    }
}

// ---- DRAFT ----

/// Whether the part is shown without the open box's feature in it: a Move face that edits the
/// history, or is followed by a draft, is shown as a ghost of the faces where they will lie;
/// a new Draft or Move face not ready to create has no preview yet.
pub(crate) fn previews_by_ghost(ws: &ReverseWorkspace, edit: &FeatureEdit) -> bool {
    (edit.editing.is_none() && draft_not_ready(edit).is_some()) || move_face::previews_by_ghost(ws, &edit.kind, edit.editing.is_some())
}

/// Why the open Draft or Move face box's Create does nothing yet, if it does not.
fn draft_not_ready(edit: &FeatureEdit) -> Option<&'static str> {
    match &edit.kind {
        FeatureKind::Draft(x) => draft::not_ready(x),
        FeatureKind::MoveFace(x) => move_face::not_ready(x),
        _ => None,
    }
}

/// Whether the bodies are drawn faint while the scan is clicked (a face being snapped).
pub(crate) fn bodies_faint(ws: &ReverseWorkspace) -> bool {
    move_face::bodies_faint(ws)
}

/// What the open box's Create reads where it is not the feature's own word: a Move face that
/// edits the history (`Move the face`), and a new feature whose kind words it its own way —
/// `Create loft`, `Create sweep`, `Create pattern on {Body}`, and a Move moves (or copies) the
/// body or the bodies (`loft_kit::create_label`).
pub(crate) fn create_label(ws: &ReverseWorkspace, edit: &FeatureEdit) -> Option<String> {
    match &edit.kind {
        FeatureKind::MoveFace(x) => move_face::create_label(ws, x, edit.editing.is_some()).map(str::to_string),
        _ if edit.editing.is_some() => None,
        kind => loft_kit::create_label(ws, kind),
    }
}

/// Create, where the open box makes something other than its feature (a Move face that edits
/// the history, or with a Draft after it) — or nothing yet, a Draft or a Move face not ready to
/// create (the status line says what it waits for). Returns whether it was handled.
pub(crate) fn commit_instead(ws: &mut ReverseWorkspace) -> bool {
    if let Some(why) = match &ws.mode {
        super::Mode::Feature(edit) => draft_not_ready(edit),
        _ => None,
    } {
        ws.status = why.into();
        return true;
    }
    move_face::commit_instead(ws)
}

/// A Move face box on faces of a body, Snap to scan pressed when asked for one face — what the
/// Move face, Move faces and Snap to scan chips on body faces start.
pub(crate) fn start_move_face(ws: &mut ReverseWorkspace, body: BodyId, faces: Vec<FaceRef>, snap: bool) {
    move_face::start(ws, body, faces, snap);
}

/// A Draft box about a flat face of a body, the faces to lean clicked next — what the Draft chip
/// on a body face starts.
pub(crate) fn start_draft_on(ws: &mut ReverseWorkspace, body: BodyId, neutral: FaceRef) {
    draft::start_on_face(ws, body, neutral);
}

/// The script steps of the Draft and Move face boxes (tests).
pub(crate) fn draft_script(ws: &mut ReverseWorkspace, cmd: &str, args: &[&str]) {
    match cmd {
        "draft" => draft::script(ws, args),
        "draftfit" => draft::fit_script(ws),
        "moveface" => move_face::script(ws, args),
        // draftchip:<neutral face> / movechip:<face>[,<face>…][:snap] — as the chips start them.
        "draftchip" | "movechip" => {
            let Some(body) = draft::default_body(ws) else { return };
            let faces: Vec<FaceRef> = args.first().map(|list| list.split(',').filter_map(|w| draft::faces_facing(ws, body, w).into_iter().next()).collect()).unwrap_or_default();
            match (cmd, faces.first()) {
                ("draftchip", Some(f)) => start_draft_on(ws, body, *f),
                ("movechip", Some(_)) => start_move_face(ws, body, faces, args.contains(&"snap")),
                _ => ws.status = format!("{cmd} step: no such face."),
            }
        }
        "snapface" | "snapat" => move_face::snap_script(ws, cmd, args),
        _ => {}
    }
}

// ---- LOFT ----

/// The pointer on the part while a box with grips on the part is open (Move body's gizmo, the
/// Extrude and Revolve boxes' grips): a grip under it takes the press and the drag after, and a
/// Shift+drag window its bodies — none of which the picks see. `None` where the pointer is the
/// picks', else whether the feature changed.
#[allow(clippy::too_many_arguments)]
pub(crate) fn grip_input(ws: &mut ReverseWorkspace, vp: Viewport, pos: Option<Pos2>, pressed: bool, down: bool, released: bool, shift: bool, edit: &mut FeatureEdit) -> Option<bool> {
    match &mut edit.kind {
        FeatureKind::MoveBody(x) => move_body::grip_input(ws, vp, pos, pressed, down, released, shift, x),
        // EXT: the Extrude and Revolve boxes' grips.
        FeatureKind::Extrude(e) => extrude::grip_input(ws, vp, pos, pressed, down, released, e),
        FeatureKind::Revolve(r) => revolve::grip_input(ws, vp, pos, pressed, down, released, r),
        _ => None,
    }
}

/// Whether the open Loft, Sweep, Pattern or Move box can make its feature: Create stands
/// disabled until it can, and Enter makes nothing. `None` for the other kinds.
pub(crate) fn loft_ready(kind: &FeatureKind) -> Option<bool> {
    loft_kit::ready(kind)
}

/// The bodies the open box's ghost carries at once — a Move's, drawn where the gizmo puts
/// them in place of their built meshes, nothing built.
pub(crate) fn carried_bodies(ws: &ReverseWorkspace) -> Vec<move_body::Carried> {
    match &ws.mode {
        super::Mode::Feature(edit) => match &edit.kind {
            FeatureKind::MoveBody(x) => move_body::carried(ws, x),
            _ => Vec::new(),
        },
        _ => Vec::new(),
    }
}

/// A Move opened on the bodies given — those selected in the body list or on the part, where a
/// selection starts it — about their middle.
#[allow(dead_code)]
pub(crate) fn move_bodies_feature(ws: &ReverseWorkspace, bodies: &[BodyId]) -> FeatureKind {
    move_body::new_feature_on(ws, bodies)
}

/// A script step of the Loft, Sweep, Pattern and Move body boxes (see `loft_kit::script_step`).
pub(crate) fn loft_script_step(ws: &mut ReverseWorkspace, cmd: &str, args: &[&str]) {
    loft_kit::script_step(ws, cmd, args);
}

// ---- RND ----
// What the Fillet, Chamfer, Shell and Delete face boxes share lives in `editors/rnd.rs`; these
// are the workspace's ways into it.
mod rnd;

/// The pointer while a Fillet, Chamfer, Shell or Delete face box is open, before a click reaches
/// its editor: the arrow on the part dragging the box's number, or a window dragged with Shift
/// over the bodies. `Some` with whether the box changed where these took the pointer; `None`
/// leaves it to the click.
#[allow(clippy::too_many_arguments)]
pub(crate) fn rnd_pointer(ws: &mut ReverseWorkspace, vp: Viewport, pos: Option<Pos2>, pressed: bool, down: bool, released: bool, shift: bool, edit: &mut FeatureEdit) -> Option<bool> {
    rnd::pointer(ws, vp, pos, pressed, down, released, shift, edit)
}

/// The workspace's `rnd:` script steps (see `rnd::script_step`).
pub(crate) fn rnd_script(ws: &mut ReverseWorkspace, args: &[&str]) {
    rnd::script_step(ws, args)
}

/// Whether one of these boxes is still reading the scan.
pub(crate) fn rnd_busy() -> bool {
    rnd::busy()
}

/// A box opened, new or from the list: what the last box remembered is let go, so a feature
/// reopened takes its mode, its numbers and its flip from the feature, not from a box of it
/// that was put away.
pub(crate) fn rnd_forget() {
    rnd::forget()
}

// ---- NET ----

/// The open box's keys before the workspace's own (ScanRuler's `keyboard.ts`: the network
/// layer's keys come before a feature box's Escape and Enter). True where they were taken.
pub(crate) fn keys(ws: &mut ReverseWorkspace, ctx: &egui::Context) -> bool {
    match &ws.mode {
        super::Mode::Feature(e) if matches!(e.kind, FeatureKind::Network(_)) => super::network_layer::key(ws, ctx),
        _ => false,
    }
}

/// Enter or a middle click with a box open, before it creates the feature: true where the box
/// took it (a network's patch being placed is closed first; Create waits until it is ready).
pub(crate) fn confirm(ws: &mut ReverseWorkspace) -> bool {
    match &ws.mode {
        super::Mode::Feature(e) if matches!(e.kind, FeatureKind::Network(_)) => super::network_layer::confirm(ws),
        _ => false,
    }
}

/// The wheel over the view with a box open: true where the box took it, and the view does not
/// zoom (the wheel over a network's curve sets its strip's spans).
pub(crate) fn wheel(ws: &mut ReverseWorkspace, delta: f32) -> bool {
    match &ws.mode {
        super::Mode::Feature(e) if matches!(e.kind, FeatureKind::Network(_)) => super::network_layer::wheel(ws, delta),
        _ => false,
    }
}

/// Whether the open feature can be created now: a patch network waits for a curve and a
/// filled cell; every other kind, always.
pub(crate) fn can_create(kind: &FeatureKind) -> bool {
    match kind {
        FeatureKind::Network(x) => mm_reverse::network_draft::network_ready(x),
        _ => true,
    }
}

// ---- EXT ----
// The Extrude and Revolve boxes' own words, readiness, grips, opening and script steps.

mod ext_fields;
mod ext_script;

/// What the open box's title, Create key and its tooltip say, whether it is ready to make,
/// whether it says the build's word itself (its preview line), and whether a field asked for the
/// feature to be made now (Enter in an extrude's To).
pub(crate) struct BoxWords {
    pub title: String,
    pub create: String,
    pub tip: &'static str,
    pub ready: bool,
    pub own_preview: bool,
    pub create_now: bool,
}

/// The open box's words, for the kinds whose box says its own; `None` leaves them to the shared
/// box.
pub(crate) fn box_words(ws: &ReverseWorkspace, edit: &FeatureEdit) -> Option<BoxWords> {
    match &edit.kind {
        FeatureKind::Extrude(e) => Some(extrude::box_words(ws, edit.editing, e)),
        FeatureKind::Revolve(r) => Some(revolve::box_words(ws, edit.editing, r)),
        _ => None,
    }
}

/// Why a box's feature cannot be made yet — nothing picked, a plane or an axis not chosen —
/// or `None` when it can (ScanRuler's `extrudeDraftReady`, `revolveDraftReady`). A box that is
/// not ready builds nothing and makes nothing.
pub(crate) fn not_ready(kind: &FeatureKind) -> Option<String> {
    match kind {
        FeatureKind::Extrude(e) => mm_reverse::extrude_draft::extrude_ready(e).err(),
        FeatureKind::Revolve(r) => mm_reverse::extrude_draft::revolve_ready(r).err(),
        _ => None,
    }
}

/// A new feature added to the part, named after its kind: a revolve with a pitch is a Coil,
/// counted apart from the plain ones.
pub(crate) fn add_new(part: &mut mm_reverse::model::Part, kind: FeatureKind) -> FeatureId {
    let coil = matches!(&kind, FeatureKind::Revolve(r) if r.pitch.is_some_and(|p| p != 0.0 && p.is_finite()));
    if !coil {
        return part.add(kind);
    }
    let n = part.next_of_kind.get("coil").copied().unwrap_or(0) + 1;
    let id = part.add_named(kind, format!("Coil {n}"));
    // Counted as a coil, not as a revolve.
    if let Some(r) = part.next_of_kind.get_mut("revolve") {
        *r = r.saturating_sub(1);
    }
    part.next_of_kind.insert("coil".into(), n);
    id
}

/// Open an Extrude box (see `extrude::open`) — for a chip on a region.
#[allow(dead_code)]
pub(crate) fn extrude_open(ws: &mut ReverseWorkspace, sketch: Option<FeatureId>, regions: Option<Vec<String>>) {
    extrude::open(ws, sketch, regions);
}

/// Open a Revolve box (see `revolve::open`), about an axis where one is given — a chip's "Revolve
/// about this".
#[allow(dead_code)]
pub(crate) fn revolve_open(ws: &mut ReverseWorkspace, sketch: Option<FeatureId>, regions: Option<Vec<String>>, axis: Option<mm_reverse::model::AxisRef>) {
    revolve::open(ws, sketch, regions, axis);
}

/// Open an Extrude box pulling a flat face of a body (a face's chip "Extrude"); why not where the
/// face is no flat one.
#[allow(dead_code)]
pub(crate) fn extrude_face_open(ws: &mut ReverseWorkspace, body: BodyId, face: usize) -> Result<(), String> {
    extrude::open_face(ws, body, face)
}

/// A box opens: what the last one kept between frames is let go.
pub(crate) fn box_opens() {
    ext_fields::reset_box();
}

/// The regions an Extrude or a Revolve opens with when nothing chose them: the sketch's one
/// region where it has exactly one, else none (ScanRuler's `autoRegions`).
pub(crate) fn auto_regions_of(ws: &ReverseWorkspace, sketch: FeatureId) -> Vec<String> {
    mm_reverse::extrude_draft::auto_regions(&extrude::regions_of(ws, sketch))
}

/// Every closed region of a sketch — what the sketch box's own Extrude and Revolve keys open
/// with.
pub(crate) fn every_region_of(ws: &ReverseWorkspace, sketch: FeatureId) -> Vec<String> {
    mm_reverse::extrude_draft::every_region(&extrude::regions_of(ws, sketch))
}

/// A script step for the Extrude or Revolve box; `false` for one that is none of theirs.
pub(crate) fn script_step(ws: &mut ReverseWorkspace, cmd: &str, args: &[&str]) -> bool {
    ext_script::step(ws, cmd, args)
}

// ---- BOOL ----

mod bool_fields;
mod bool_script;

pub(crate) use bool_script::bool_script_step;

/// Whether the open feature can be made, or why it waits — its Create key held down until it
/// can (ScanRuler's `featureDraftReady`), for the kinds whose box can stand without what they
/// need: a cut or a mirror with no plane chosen yet, a combine of a body with itself.
pub(crate) fn ready_to_create(kind: &FeatureKind) -> Result<(), String> {
    match kind {
        FeatureKind::CutPlane(x) => cut_plane::ready(x),
        FeatureKind::Mirror(x) => mirror::ready(x),
        FeatureKind::Boolean(x) => boolean::ready(x),
        FeatureKind::OffsetBody(x) => offset_body::ready(x),
        _ => Ok(()),
    }
}

// ---- HOLE ----

/// The script's `hole` step (see `hole::script_step`).
pub(crate) fn hole_script_step(ws: &mut ReverseWorkspace, args: &[&str]) -> String {
    hole::script_step(ws, args)
}

/// The script's `thread` step (see `thread::script_step`).
pub(crate) fn thread_script_step(ws: &mut ReverseWorkspace, args: &[&str]) -> String {
    thread::script_step(ws, args)
}

/// The script's `threadread` step (see `thread::script_read`).
pub(crate) fn thread_read_step(ws: &mut ReverseWorkspace, args: &[&str]) -> String {
    thread::script_read(ws, args)
}

// ---- SURF ----

/// A script step of the surface features (Sheet, Split, Thicken, Fit surface, Trim, Extract,
/// Import); `false` for a step that is none of theirs.
pub(crate) fn surf_script(ws: &mut ReverseWorkspace, cmd: &str, args: &[&str]) -> bool {
    surf_script::run(ws, cmd, args)
}

// ---- PANEL ----

pub(crate) use super::params::Num;
// For the feature boxes' `Num::new(…).nominal(NominalKind::…)` fields (see expr_field).
#[allow(unused_imports)]
pub(crate) use mm_reverse::nominal::NominalKind;

/// One number of a feature box, as every box shows it (ScanRuler's `Num` over `ExprInput`,
/// `ui/featureFields.tsx`): the label at the left — with an ⓘ that opens `hint` — then, at the
/// right, the nominal chip `→ 6 mm` where the number is not round (not while an expression
/// drives it), the text field, and the unit's word after it. Returns the number committed this
/// frame, for the box to write into its own field as it writes a number — its sign dropped, a
/// clamp, a radius from a diameter: whatever the box does with a typed number. Nothing is
/// written by the field itself but the expression.
///
/// The field takes an expression as well as a number: arithmetic (`+ - * / ^`, brackets,
/// `sin cos tan asin acos atan atan2 sqrt abs min max floor ceil round pow`, `pi`, `e`), units
/// (`1 in`, `2 cm`, `30 deg`; a bare number is the field's unit), and the parameters by name,
/// offered as a word is typed. A length is held in millimetres and shown and typed in the
/// display units (`ws.units()`). What was typed stays in the field, accent-coloured, with the
/// value it comes to written where the unit was; an expression that does not read — a name
/// unknown, a length added to an angle, an area where a length is wanted, a fraction where a
/// whole number is — stays in the field with why under it, and the number keeps what it was,
/// until it is put right or Escape puts the old text back. Enter or leaving the field commits;
/// neither goes on to the box (Enter does not create it, Escape does not close it), unless the
/// field was given `pass_keys()` and nothing was typed into it.
///
/// `exprs` is the feature's expression map (`x.exprs`) and `slot` the path its expression is
/// kept under — the slot path of ScanRuler's table, as `mm_reverse::param_slots` lists it for
/// every kind ("extent.to", "taper", "angleDeg", "thickness", "radius", "distance",
/// "distance2", "plane.offset", "diameter", "depth", "pitch", "count", "step", "translate.x" …).
/// An expression typed is kept there, a plain number typed lets the one there go (a parameter
/// no longer drives it), and the part's numbers are written from the expressions again
/// whenever a parameter changes and before every build — so the box must show the number in
/// the slot's own sense (a taper, not a draft; a diameter, not a radius). Give `None` for
/// `exprs`, or no slot, for a number the feature keeps no expression for: the field still
/// reads an expression into a number.
///
/// ```ignore
/// let num = Num::new("fillet-radius", "Radius", x.radius).slot("radius").nominal(NominalKind::Radius).hint("Read off the scan …");
/// if let Some(v) = expr_field(ui, ws, Some(&mut x.exprs), num) {
///     x.radius = v.abs();
///     changed = true;
/// }
/// ```
///
/// `Num::new(id, label, value)` makes a length field; `.unit("°")` (or `.nominal(Angle)`) an
/// angle, any other `.unit(word)` a plain number (`.kind(..)` says it outright, e.g. an
/// integer count). The `id` is the field's identity between frames — unique in the box, as
/// ScanRuler's test ids ("extrude-to", "chamfer-distance2") — and the name a script types into
/// (`param-type:<id>:<text>`).
#[allow(dead_code)] // the merged boxes still use their own number fields (EXT, RND, BOOL, …)
pub(crate) fn expr_field(ui: &mut egui::Ui, ws: &ReverseWorkspace, exprs: Option<&mut mm_reverse::model::Exprs>, num: Num) -> Option<f64> {
    super::params::num_field(ui, ws, exprs, num)
}

/// The expressions behind some numbers let go: a grip dragged, a number set by a key or a click
/// rather than typed (ScanRuler's `unlinkSlots`). Whether any was let go.
#[allow(dead_code)] // for the grips and the boxes' keys, once they take expr_field
pub(crate) fn unlink(exprs: &mut mm_reverse::model::Exprs, paths: &[&str]) -> bool {
    mm_reverse::part_parameters::unlink_slots(exprs, paths)
}
