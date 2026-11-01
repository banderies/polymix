// SPDX-License-Identifier: AGPL-3.0-only
//! The Loft feature's box: the sketches it runs through, ticked in the order the loft takes
//! them, whether it is straight between the profiles, and what it does with the bodies it
//! meets. Ported from ScanRuler (`plugins/reverse/ui/FeatureEditor.tsx` the loft fields,
//! `state/reFeatureSlice.ts` startFeature), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.

use egui::Pos2;
use mm_reverse::model::{BodyOp, FeatureKind, LoftFeature};

use super::loft_kit::sketches;
use super::op_line;
use crate::reverse::{PickWhat, ReverseWorkspace};
use crate::tools::Viewport;

/// Whether the editor is built: its button stands with the others.
pub(crate) const READY: bool = true;
/// The panel's button: its label and what it does.
pub(crate) const BUTTON: (&str, &str) = ("Loft", "A solid through the profiles of two or more sketches on their own planes");
/// What a click picks first when the box opens.
pub(crate) const FIRST_PICK: PickWhat = PickWhat::Nothing;

/// The feature the panel's button starts, or why it cannot: no sketch ticked yet, straight
/// between the profiles, joined.
pub(crate) fn new_feature(ws: &ReverseWorkspace) -> Result<FeatureKind, String> {
    if sketches(ws).len() < 2 {
        return Err("A loft runs through two sketches or more: draw them first.".into());
    }
    Ok(FeatureKind::Loft(LoftFeature { sketches: Vec::new(), ruled: true, op: BodyOp::Join, target: None }))
}

pub(crate) fn ui(ws: &mut ReverseWorkspace, ui: &mut egui::Ui, x: &mut LoftFeature, pick: &mut PickWhat) -> bool {
    let mut changed = false;
    ui.label("Sketches");
    ui.indent("re_loft_sketches", |ui| {
        for (id, name) in sketches(ws) {
            let mut on = x.sketches.contains(&id);
            if ui.checkbox(&mut on, name).changed() {
                // Ticked last, taken last: the order ticked is the loft's.
                if on {
                    x.sketches.push(id);
                } else {
                    x.sketches.retain(|s| *s != id);
                }
                changed = true;
            }
        }
    });
    changed |= ui.checkbox(&mut x.ruled, "Straight between the profiles").changed();
    changed |= op_line(ui, &mut x.op);
    ui.weak("In the order ticked, each sketch's one region — or its largest.");
    *pick = PickWhat::Nothing;
    changed
}

pub(crate) fn pick(_ws: &mut ReverseWorkspace, _vp: Viewport, _p: Pos2, _clicked: bool, _x: &mut LoftFeature, _pick: &mut PickWhat) -> bool {
    false
}

pub(crate) fn paint(_ws: &ReverseWorkspace, _painter: &egui::Painter, _vp: Viewport, _x: &LoftFeature) {}
