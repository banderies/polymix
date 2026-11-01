// SPDX-License-Identifier: AGPL-3.0-only
//! The Shell feature's box: the body, the faces taken away to open it (each kept as its
//! fingerprint) — none leaves it closed and hollow inside — and the wall. A click on a face of
//! the body picks it, a second lets it go; an arrow on the first opened face drags the wall.
//! Ported from ScanRuler (`plugins/reverse/ui/FeatureEditor.tsx`, `featureFields.tsx`,
//! `plugins/reverse/ui/ReversePanel.tsx` startShell), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only.
//!
//! ScanRuler starts the wall at the mean of the part's Thickness map where one was measured;
//! Polymix measures no Thickness map, so the wall starts at 2 mm and the box says so.

use egui::Pos2;
use mm_reverse::model::{Exprs, FeatureKind, ShellFeature};
use mm_reverse::nominal::NominalKind;

use super::rnd::{self, with_draft, ThicknessFrom};
use crate::reverse::sketcher::plural;
use crate::reverse::{PickWhat, ReverseWorkspace};
use crate::tools::Viewport;

pub(crate) const READY: bool = true;
pub(crate) const BUTTON: (&str, &str) = ("Shell", "Hollow the body to a wall — click the faces to open, or none for a closed hollow; the wall comes from the Thickness map when one was measured");
pub(crate) const FIRST_PICK: PickWhat = PickWhat::Faces;

pub(crate) fn new_feature(ws: &ReverseWorkspace) -> Result<FeatureKind, String> {
    rnd::forget();
    let body = rnd::last_solid(ws).ok_or("A shell hollows a body: make a body first.")?;
    Ok(FeatureKind::Shell(ShellFeature { body, faces: Vec::new(), faces_where: Vec::new(), thickness: rnd::DEFAULT_WALL_MM, exprs: Exprs::new() }))
}

/// The box's draft for this shell: a shell reopened from the list keeps its wall as typed; a
/// new one takes the Thickness map's mean where one was measured, 2 mm otherwise.
fn session(ws: &ReverseWorkspace, x: &mut ShellFeature) {
    let at = x as *const ShellFeature as usize;
    let map = rnd::thickness_map_mean(ws).map(|m| (m * 20.0).round() / 20.0);
    let wall = rnd::session(
        ws,
        at,
        |d, editing| {
            d.thickness_from = match (editing, map) {
                (true, _) => ThicknessFrom::Typed,
                (false, Some(_)) => ThicknessFrom::Map,
                (false, None) => ThicknessFrom::Default,
            };
        },
        |d| d.thickness_from,
    );
    if let (ThicknessFrom::Map, Some(m)) = (wall, map) {
        if x.thickness == rnd::DEFAULT_WALL_MM {
            x.thickness = m;
        }
    }
}

/// A length as a row reads it: to the hundredth, no zeros carried.
fn mm(v: f64) -> String {
    format!("{} mm", mm_reverse::fillet_num::js_num((v * 100.0).round() / 100.0))
}

pub(crate) fn ui(ws: &mut ReverseWorkspace, ui: &mut egui::Ui, x: &mut ShellFeature, pick: &mut PickWhat) -> bool {
    session(ws, x);
    *pick = PickWhat::Faces;
    let mut changed = false;
    if rnd::body_select(ui, ws, &mut x.body, "shell") {
        x.faces.clear();
        x.faces_where.clear();
        changed = true;
    }
    let n = x.faces.len();
    ui.horizontal_wrapped(|ui| {
        ui.label("Open");
        if n == 0 {
            ui.weak("None — a closed hollow body. Click faces of the body to take them away");
        } else {
            ui.strong(format!("{n} face{} picked", plural(n)));
            if rnd::clear_picks(ui) {
                x.faces.clear();
                x.faces_where.clear();
                changed = true;
            }
        }
    });
    let mut t = x.thickness;
    if rnd::num(ui, "Wall", "The wall left after the inside is taken away, inward from every face that stays. The arrow on the opened face drags it.", &mut t, NominalKind::Length, 0.0..=1e4, Some((&mut x.exprs, "thickness")), "shell_thickness") {
        x.thickness = t.abs();
        with_draft(|d| d.thickness_from = ThicknessFrom::Typed);
        changed = true;
    }
    let said = match with_draft(|d| d.thickness_from) {
        ThicknessFrom::Map => "The wall is the Thickness map’s mean over the part.".to_string(),
        ThicknessFrom::Default => format!("No Thickness map measured — {} to start; measure the part in Wall thickness for its own.", mm(rnd::DEFAULT_WALL_MM)),
        ThicknessFrom::Typed => "Wall as typed.".to_string(),
    };
    rnd::hint(ui, &said, false);
    changed
}

pub(crate) fn pick(ws: &mut ReverseWorkspace, vp: Viewport, p: Pos2, clicked: bool, x: &mut ShellFeature, _pick: &mut PickWhat) -> bool {
    let was = x.body;
    let changed = super::pick_face(ws, vp, p, clicked, &mut x.body, &mut x.faces);
    if x.body != was {
        x.faces_where.clear();
    }
    changed
}

pub(crate) fn paint(ws: &ReverseWorkspace, painter: &egui::Painter, vp: Viewport, x: &ShellFeature) {
    rnd::paint_face_picks(ws, painter, vp, x.body, &x.faces);
    rnd::paint_window(painter);
    if let Some(g) = x.faces.first().and_then(|f| rnd::face_grip(ws, x.body, f)) {
        rnd::paint_grip(ws, painter, vp, g);
    }
}
