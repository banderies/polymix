// SPDX-License-Identifier: AGPL-3.0-only
//! The Thicken feature's box: the sheet, the wall's thickness, which way it grows, and the
//! result. A face that bends tighter than the thickness towards the side it grows to is named
//! on the row and outlined red on the part; the wall the solid came to along the sheet's edge
//! is said. Ported from ScanRuler (`plugins/reverse/ui/FeatureEditor.tsx` the thicken's
//! fields, `state/reFeatureSlice.ts` its defaults), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only.

use egui::Pos2;
use mm_reverse::model::{BodyOp, FeatureKind, ThickenFeature, ThickenSide};

use super::body_name;
use super::surf_common::{can_cut, num, op_field, paint_lines, report, sheets, RED};
use crate::reverse::{PickWhat, ReverseWorkspace};
use crate::tools::Viewport;

/// Whether the editor is built: its button stands with the others.
pub(crate) const READY: bool = true;
/// The panel's button: its label and what it does.
pub(crate) const BUTTON: (&str, &str) = ("Thicken", "A sheet made a solid of a thickness");
/// What a click picks first when the box opens.
pub(crate) const FIRST_PICK: PickWhat = PickWhat::Body;

/// The default wall, mm.
const DEFAULT_WALL_MM: f64 = 2.0;

/// A new thicken: of the last sheet there is, 2 mm against its normal, a body of its own — a
/// wall from a skin is a part of its own more often than a boss on another.
pub(crate) fn new_feature(ws: &ReverseWorkspace) -> Result<FeatureKind, String> {
    let sheet = sheets(ws).last().copied().ok_or("Make a sheet or fit a surface first.")?;
    Ok(FeatureKind::Thicken(ThickenFeature { sheet, thickness: DEFAULT_WALL_MM, side: ThickenSide::In, op: BodyOp::New, target: None, exprs: Default::default() }))
}

pub(crate) fn ui(ws: &mut ReverseWorkspace, ui: &mut egui::Ui, x: &mut ThickenFeature, _pick: &mut PickWhat) -> bool {
    let mut changed = false;
    let list: Vec<(mm_reverse::model::BodyId, String)> = ws.pick_set().iter().filter(|b| ws.built.body(b.id).is_some_and(|x| x.surface)).map(|b| (b.id, b.name.clone())).collect();
    ui.horizontal(|ui| {
        ui.label("Sheet");
        let shown = if list.iter().any(|s| s.0 == x.sheet) { body_name(ws, x.sheet) } else if list.is_empty() { "Make a sheet or fit a surface first".into() } else { "Select…".into() };
        egui::ComboBox::from_id_salt("re_thicken_sheet").selected_text(shown).show_ui(ui, |ui| {
            for (id, n) in &list {
                changed |= ui.selectable_value(&mut x.sheet, *id, n).changed();
            }
        });
    });
    changed |= num(ui, "Thickness", &mut x.thickness, "mm", 0.001..=1000.0, 0.05, "The wall, through the solid — every point of the sheet that far from the far skin");
    ui.horizontal(|ui| {
        ui.label("Grows");
        let word = |s: ThickenSide| match s {
            ThickenSide::In => "Against the sheet's normal",
            ThickenSide::Out => "Along the sheet's normal",
            ThickenSide::Both => "Half each way",
        };
        egui::ComboBox::from_id_salt("re_thicken_side").selected_text(word(x.side)).width(200.0).show_ui(ui, |ui| {
            for s in [ThickenSide::In, ThickenSide::Out, ThickenSide::Both] {
                changed |= ui.selectable_value(&mut x.side, s, word(s)).changed();
            }
        });
    });
    changed |= op_field(ui, &mut x.op, can_cut(ws));
    ui.weak("A freeform thin shell's wall from its one skin — the sheet stays a body of its own, put out of view. A face that bends tighter than the thickness towards the side it grows to is named, and the thickness it takes.");
    if let Some(note) = report(ws).and_then(|r| r.note.as_ref()) {
        ui.weak(note);
    }
    changed
}

/// A click on a sheet takes it.
pub(crate) fn pick(ws: &mut ReverseWorkspace, vp: Viewport, p: Pos2, clicked: bool, x: &mut ThickenFeature, _pick: &mut PickWhat) -> bool {
    let mut at = x.sheet;
    if !super::pick_body(ws, vp, p, clicked, &mut at) || !ws.built.body(at).is_some_and(|b| b.surface) {
        return false;
    }
    x.sheet = at;
    true
}

/// The faces that do not take the thickness, outlined red, with a cross where each bends
/// tightest.
pub(crate) fn paint(ws: &ReverseWorkspace, painter: &egui::Painter, vp: Viewport, _x: &ThickenFeature) {
    if let Some(r) = report(ws) {
        paint_lines(ws, painter, vp, &r.where_lines, RED);
    }
}
