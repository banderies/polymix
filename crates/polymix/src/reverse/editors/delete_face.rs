// SPDX-License-Identifier: AGPL-3.0-only
//! The Delete face feature's box: the body and the faces taken off (each kept as its
//! fingerprint), the faces round them run on over the gap. One click takes the whole of the
//! round, the hole or the boss the face is part of — the faces it is split from, the faces it
//! closes in — and with Tangent faces the rounds running on from a round; a second click lets
//! go of what that click took. Ported from ScanRuler (`plugins/reverse/ui/FeatureEditor.tsx`,
//! `plugins/reverse/state/reFeatureSlice.ts` toggleDraftFace, `plugins/reverse/core/faceGroup.ts`),
//! © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.

use egui::Pos2;
use mm_reverse::model::{DeleteFaceFeature, FeatureKind};

use super::rnd::{self, with_draft};
use crate::reverse::sketcher::plural;
use crate::reverse::{PickWhat, ReverseWorkspace};
use crate::tools::Viewport;

/// Whether the editor is built: its button stands with the others.
pub(crate) const READY: bool = true;
/// The panel's button: its label and what it does.
pub(crate) const BUTTON: (&str, &str) = ("Delete face", "Delete faces of the body and heal the gap — a fillet, a chamfer, a hole or a boss taken out, the faces round it run on to close over where it was: click a face of it, and the rest of the round or hole comes with it");
/// What a click picks first when the box opens.
pub(crate) const FIRST_PICK: PickWhat = PickWhat::Faces;

/// The feature the panel's button starts, or why it cannot.
pub(crate) fn new_feature(ws: &ReverseWorkspace) -> Result<FeatureKind, String> {
    rnd::forget();
    let body = rnd::last_solid(ws).ok_or("A face delete takes faces off a body: make a body first.")?;
    Ok(FeatureKind::DeleteFace(DeleteFaceFeature { body, faces: Vec::new(), faces_where: Vec::new() }))
}

fn session(ws: &ReverseWorkspace, x: &DeleteFaceFeature) {
    rnd::session(ws, x as *const DeleteFaceFeature as usize, |d, _| d.tangent = true, |_| ());
}

pub(crate) fn ui(ws: &mut ReverseWorkspace, ui: &mut egui::Ui, x: &mut DeleteFaceFeature, pick: &mut PickWhat) -> bool {
    session(ws, x);
    *pick = PickWhat::Faces;
    let mut changed = false;
    if rnd::body_select(ui, ws, &mut x.body, "delete_face") {
        x.faces.clear();
        x.faces_where.clear();
        changed = true;
    }
    let n = x.faces.len();
    ui.horizontal_wrapped(|ui| {
        ui.label("Faces");
        if n == 0 {
            ui.weak("Click a face of the round, the hole or the boss to take off");
        } else {
            ui.strong(format!("{n} face{} picked", plural(n)));
            if rnd::clear_picks(ui) {
                x.faces.clear();
                x.faces_where.clear();
                changed = true;
            }
        }
    });
    ui.horizontal(|ui| {
        let mut tangent = with_draft(|d| d.tangent);
        if ui.checkbox(&mut tangent, "Tangent faces").changed() {
            with_draft(|d| d.tangent = tangent);
        }
        rnd::info_dot(ui, "What a click takes", "A click takes the face, the faces it is split from on its own surface — both halves of a bore — and every face the picked ones close in on all sides: a blind hole's floor, a boss's top. With Tangent faces on, a click on a round also takes the rounds that run on from it smoothly — a fillet's whole chain, up to the flats either side — and a bore's or a boss's wall the round at its mouth or foot. Click a picked face again to let go of what that click took.");
    });
    rnd::hint(ui, "The faces are taken off and the faces round them run on to close the gap, each on its own surface. Pick every face of what is to go — a chamfer's faces one by one, or Shift+drag a window over them.", false);
    changed
}

pub(crate) fn pick(ws: &mut ReverseWorkspace, vp: Viewport, p: Pos2, clicked: bool, x: &mut DeleteFaceFeature, _pick: &mut PickWhat) -> bool {
    let was = x.body;
    let tangent = with_draft(|d| d.tangent);
    let changed = rnd::toggle_face(ws, vp, p, clicked, &mut x.body, &mut x.faces, Some(tangent));
    if x.body != was {
        x.faces_where.clear();
    }
    changed
}

pub(crate) fn paint(ws: &ReverseWorkspace, painter: &egui::Painter, vp: Viewport, x: &DeleteFaceFeature) {
    rnd::paint_face_picks(ws, painter, vp, x.body, &x.faces);
    rnd::paint_window(painter);
}

#[cfg(test)]
mod tests;
