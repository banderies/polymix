// SPDX-License-Identifier: AGPL-3.0-only
//! The Split feature's box: the body, the sheets and bodies it is cut with, which fragments are
//! kept — on the scan, the largest, the smallest, or what lies between the sheets — and for
//! the scan, how near a fragment's face must lie to it and over what share. A sheet as the body
//! is trimmed: its pieces kept by the scan or by area and sewn into one. Ported from ScanRuler
//! (`plugins/reverse/ui/FeatureEditor.tsx` the split's fields, `state/reFeatureSlice.ts`
//! its defaults, `state/reDrafts.ts` DEFAULT_SPLIT_TOLERANCE_MM and DEFAULT_SPLIT_OVERLAP),
//! © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.

use egui::Pos2;
use mm_reverse::model::{BodyId, FeatureKind, SplitFeature, SplitKeep};

use super::surf_common::{last_solid, num, paint_lines, report, RED};
use super::{body_name, pick_body};
use crate::reverse::view::PICKED;
use crate::reverse::{PickWhat, ReverseWorkspace};
use crate::tools::Viewport;

/// Whether the editor is built: its button stands with the others.
pub(crate) const READY: bool = true;
/// The panel's button: its label and what it does.
pub(crate) const BUTTON: (&str, &str) = ("Split", "A body split by sheets or bodies");
/// What a click picks first when the box opens.
pub(crate) const FIRST_PICK: PickWhat = PickWhat::Tool;

/// How far a sampled point of a fragment's face may lie from the scan and still count as on
/// it, mm, to start with.
pub(crate) const DEFAULT_SPLIT_TOLERANCE_MM: f64 = 0.3;
/// The share of a fragment's samples that must lie on the scan, to start with.
pub(crate) const DEFAULT_SPLIT_OVERLAP: f64 = 0.5;

/// A new split: of the last solid body, keeping what lies on the scan.
pub(crate) fn new_feature(ws: &ReverseWorkspace) -> Result<FeatureKind, String> {
    let body = last_solid(ws).ok_or("A split cuts a body: make one first.")?;
    Ok(FeatureKind::Split(SplitFeature { body, tools: Vec::new(), keep: SplitKeep::Scan, tolerance: DEFAULT_SPLIT_TOLERANCE_MM, overlap: DEFAULT_SPLIT_OVERLAP, within: None, exprs: Default::default() }))
}

/// The body select every feature on a body has: `Body`, its bodies by name.
pub(crate) fn body_select(ui: &mut egui::Ui, ws: &ReverseWorkspace, body: &mut BodyId, salt: &str) -> bool {
    let mut changed = false;
    let bodies: Vec<(BodyId, String)> = ws.pick_set().iter().map(|b| (b.id, b.name.clone())).collect();
    ui.horizontal(|ui| {
        ui.label("Body");
        egui::ComboBox::from_id_salt(salt).selected_text(body_name(ws, *body)).show_ui(ui, |ui| {
            for (id, n) in &bodies {
                changed |= ui.selectable_value(body, *id, n).changed();
            }
        });
    });
    changed
}

pub(crate) fn ui(ws: &mut ReverseWorkspace, ui: &mut egui::Ui, x: &mut SplitFeature, _pick: &mut PickWhat) -> bool {
    let mut changed = false;
    if body_select(ui, ws, &mut x.body, "re_split_body") {
        changed = true;
        let body = x.body;
        x.tools.retain(|t| *t != body);
        if is_sheet(ws, body) && x.keep == SplitKeep::Between {
            x.keep = SplitKeep::Scan;
        }
    }
    let of_sheet = is_sheet(ws, x.body);
    let piece = if of_sheet { "piece" } else { "fragment" };
    let others: Vec<(BodyId, String)> = ws.pick_set().iter().filter(|b| b.id != x.body).map(|b| (b.id, b.name.clone())).collect();
    ui.label("Cut with");
    if others.is_empty() {
        ui.weak("Make a sheet first");
    }
    for (id, name) in &others {
        let mut on = x.tools.contains(id);
        if ui.checkbox(&mut on, name).changed() {
            changed = true;
            if on {
                x.tools.push(*id);
            } else {
                x.tools.retain(|t| t != id);
            }
        }
    }
    ui.horizontal(|ui| {
        ui.label("Keep");
        let word = |k: SplitKeep| match k {
            SplitKeep::Scan => format!("The {piece}s on the scan"),
            SplitKeep::Largest => format!("The largest {piece}{}", if of_sheet { ", by area" } else { "" }),
            SplitKeep::Smallest => format!("The smallest {piece}{}", if of_sheet { ", by area" } else { "" }),
            SplitKeep::Between => "What lies between the sheets".to_string(),
        };
        egui::ComboBox::from_id_salt("re_split_keep").selected_text(word(x.keep)).width(220.0).show_ui(ui, |ui| {
            for k in [SplitKeep::Scan, SplitKeep::Largest, SplitKeep::Smallest, SplitKeep::Between] {
                if k == SplitKeep::Between && of_sheet {
                    continue;
                }
                changed |= ui.selectable_value(&mut x.keep, k, word(k)).changed();
            }
        });
    });
    if of_sheet {
        ui.weak("The sheet is trimmed: the pieces kept are sewn into one sheet, for Thicken. A fitted skin laid past the part's edge loses its overshoot on the scan.");
    }
    if x.keep == SplitKeep::Scan {
        changed |= num(ui, "Tolerance", &mut x.tolerance, "mm", 0.001..=100.0, 0.01, "How far a sampled point of a fragment's face may lie from the scan and still count as on it — a few times the scan's scatter");
        let mut pct = (x.overlap * 100.0).round();
        if num(ui, "Overlap", &mut pct, "%", 5.0..=100.0, 1.0, "The share of a fragment's samples that must lie on the scan for it to be kept: a pocket's fragment has the pocket's roof where the scan has its floor, and falls out") {
            x.overlap = (pct / 100.0).clamp(0.05, 1.0);
            changed = true;
        }
    }
    if let Some(margin) = x.within {
        ui.horizontal_wrapped(|ui| {
            ui.weak(format!("Within the fitted sheets' regions, grown by {}: the body outside them is left as it was.", super::surf_common::mm(margin, 2)));
            if ui.small_button("Cut with the whole sheets").on_hover_text("Leave within out: the sheets cut where they reach, carried on to the body's sides").clicked() {
                x.within = None;
                changed = true;
            }
        });
    }
    if let Some(r) = report(ws) {
        for o in &r.off_scan {
            ui.weak(format!("The cut on {} stands up to {} off the scan past its region.", o.tool, super::surf_common::mm(o.max, 2)));
        }
    }
    ui.weak("Click a sheet or a body on the part to cut with it, or tick it above.");
    changed
}

/// Whether a body is a sheet, as the bodies stood before the feature.
fn is_sheet(ws: &ReverseWorkspace, id: BodyId) -> bool {
    ws.built.body(id).is_some_and(|b| b.surface) || ws.pick_view(id).is_some_and(|v| ws.built.body(v.id).is_some_and(|b| b.surface))
}

/// A click on another body takes it as a tool, or lets it go.
pub(crate) fn pick(ws: &mut ReverseWorkspace, vp: Viewport, p: Pos2, clicked: bool, x: &mut SplitFeature, _pick: &mut PickWhat) -> bool {
    let mut at = x.body;
    if !pick_body(ws, vp, p, clicked, &mut at) || at == x.body {
        return false;
    }
    match x.tools.iter().position(|t| *t == at) {
        Some(i) => {
            x.tools.remove(i);
        }
        None => x.tools.push(at),
    }
    true
}

/// The tools' edges lit on the part, and where a cut face stands off the scan.
pub(crate) fn paint(ws: &ReverseWorkspace, painter: &egui::Painter, vp: Viewport, x: &SplitFeature) {
    for t in &x.tools {
        if let Some(b) = ws.pick_view(*t) {
            let lines: Vec<Vec<glam::DVec3>> = b.topology.edges.iter().map(|e| e.polyline.clone()).collect();
            paint_lines(ws, painter, vp, &lines, PICKED);
        }
    }
    if let Some(r) = report(ws) {
        for o in r.off_scan.iter().filter(|o| o.max > x.tolerance) {
            if let Some(q) = ws.project(vp, o.at) {
                painter.circle_stroke(q, 6.0, egui::Stroke::new(2.0, RED));
            }
        }
    }
}
