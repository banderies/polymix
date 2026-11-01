// SPDX-License-Identifier: AGPL-3.0-only
//! The Sheet feature's box: the sketch, the offsets along its plane's normal the surface
//! stands between, and what the sheet is. Ported from ScanRuler (`plugins/reverse/ui/`
//! `FeatureEditor.tsx` the sheet's fields, `featureFields.tsx` its preview line;
//! `state/reFeatureSlice.ts` its defaults), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.

use egui::Pos2;
use mm_reverse::model::{FeatureKind, SheetFeature};

use super::surf_common::{num, report};
use crate::reverse::{PickWhat, ReverseWorkspace};
use crate::tools::Viewport;

/// Whether the editor is built: its button stands with the others.
pub(crate) const READY: bool = true;
/// The panel's button: its label and what it does.
pub(crate) const BUTTON: (&str, &str) = ("Sheet", "A sketch's pieces extruded as a surface — a cutter to split a body with");
/// What a click picks first when the box opens.
pub(crate) const FIRST_PICK: PickWhat = PickWhat::Nothing;

/// A new sheet: of the last sketch, from −10 to 10 mm.
pub(crate) fn new_feature(ws: &ReverseWorkspace) -> Result<FeatureKind, String> {
    let sketch = ws.part.features.iter().rev().find(|f| matches!(f.kind, FeatureKind::Sketch(_))).map(|f| f.id).ok_or("A sheet is drawn from a sketch: make one first.")?;
    Ok(FeatureKind::Sheet(SheetFeature { sketch_id: sketch, from: -10.0, to: 10.0, exprs: Default::default() }))
}

pub(crate) fn ui(ws: &mut ReverseWorkspace, ui: &mut egui::Ui, x: &mut SheetFeature, _pick: &mut PickWhat) -> bool {
    let mut changed = false;
    let sketches: Vec<(u32, String)> = ws.part.features.iter().filter(|f| matches!(f.kind, FeatureKind::Sketch(_))).map(|f| (f.id, f.name.clone())).collect();
    ui.horizontal(|ui| {
        ui.label("Sketch");
        let name = sketches.iter().find(|s| s.0 == x.sketch_id).map_or("Select…".to_string(), |s| s.1.clone());
        egui::ComboBox::from_id_salt("re_sheet_sketch").selected_text(name).show_ui(ui, |ui| {
            for (id, n) in &sketches {
                changed |= ui.selectable_value(&mut x.sketch_id, *id, n).changed();
            }
        });
    });
    changed |= num(ui, "From", &mut x.from, "mm", -1e5..=1e5, 0.1, "How far along the sketch plane's normal the surface starts");
    changed |= num(ui, "To", &mut x.to, "mm", -1e5..=1e5, 0.1, "How far along the normal it ends — past the body on both sides, so the split goes right through");
    ui.weak("Every drawn piece of the sketch, open or closed, becomes the surface. It is a body in the list, marked as a surface, and never exported on its own.");
    if let Some(n) = report(ws).and_then(|r| r.sheet_faces) {
        ui.weak(format!("A sheet of {n} face{} — a cutter to split a body with, or a skin to thicken.", if n == 1 { "" } else { "s" }));
    }
    changed
}

pub(crate) fn pick(_ws: &mut ReverseWorkspace, _vp: Viewport, _p: Pos2, _clicked: bool, _x: &mut SheetFeature, _pick: &mut PickWhat) -> bool {
    false
}

pub(crate) fn paint(_ws: &ReverseWorkspace, _painter: &egui::Painter, _vp: Viewport, _x: &SheetFeature) {}
