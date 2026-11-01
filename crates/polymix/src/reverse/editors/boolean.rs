// SPDX-License-Identifier: AGPL-3.0-only
//! The Combine (boolean) feature's box: the body, the operation — join, cut by or intersect
//! with — the tool, and whether the tool is kept as a body of its own; what becomes of the
//! tool said under them, and what becomes of a sheet when the body is one. Both bodies are
//! chosen from their lists. Ported from ScanRuler (`plugins/reverse/ui/FeatureEditor.tsx` the
//! boolean case, `state/reFeatureSlice.ts` its defaults), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only.

use egui::{Pos2, RichText};
use mm_reverse::model::{BodyId, BooleanFeature, BooleanOp, FeatureKind};

use super::bool_fields::{body_select, field, hint};
use crate::reverse::{PickWhat, ReverseWorkspace};
use crate::tools::Viewport;

pub(crate) const READY: bool = true;
pub(crate) const BUTTON: (&str, &str) = ("Combine", "Join, cut or intersect two bodies made separately — or trim a sheet by a solid, or sew two sheets into one, before Thicken");
pub(crate) const FIRST_PICK: PickWhat = PickWhat::Nothing;

/// Whether a body is a sheet (a surface body) rather than a solid.
pub(crate) fn is_sheet(ws: &ReverseWorkspace, id: BodyId) -> bool {
    ws.built.body(id).is_some_and(|b| b.surface)
}

/// The body a feature most likely works on (ScanRuler's `lastSolid`): the last solid body, a
/// sheet only when there is nothing else.
pub(crate) fn last_solid(ws: &ReverseWorkspace) -> Option<BodyId> {
    let bodies = ws.pick_set();
    bodies.iter().rev().find(|b| !is_sheet(ws, b.id)).or(bodies.last()).map(|b| b.id)
}

/// A new Combine: joining, on the last solid body, the first other solid as the tool, the tool
/// not kept.
pub(crate) fn new_feature(ws: &ReverseWorkspace) -> Result<FeatureKind, String> {
    let body = last_solid(ws).ok_or("A combine takes two bodies: make two first.")?;
    let tool = ws.pick_set().iter().find(|b| b.id != body && !is_sheet(ws, b.id)).map(|b| b.id).ok_or("A combine takes two bodies: make a second one first.")?;
    Ok(FeatureKind::Boolean(BooleanFeature { op: BooleanOp::Join, body, tool, keep: false }))
}

/// Whether the combine can be made: two bodies, not one twice.
pub(crate) fn ready(c: &BooleanFeature) -> Result<(), String> {
    if c.body == c.tool {
        return Err("Choose a tool other than the body.".into());
    }
    Ok(())
}

pub(crate) fn ui(ws: &mut ReverseWorkspace, ui: &mut egui::Ui, c: &mut BooleanFeature, _pick: &mut PickWhat) -> bool {
    let mut changed = body_select(ui, ws, "re_combine_body", "Body", &mut c.body);
    field(ui, "Operation", None, |ui| {
        let name = |op: BooleanOp| match op {
            BooleanOp::Join => "Join",
            BooleanOp::Cut => "Cut by",
            BooleanOp::Intersect => "Intersect with",
        };
        egui::ComboBox::from_id_salt("re_combine_op").selected_text(name(c.op)).show_ui(ui, |ui| {
            for op in BooleanOp::ALL {
                changed |= ui.selectable_value(&mut c.op, op, name(op)).changed();
            }
        });
    });
    changed |= body_select(ui, ws, "re_combine_tool", "Tool", &mut c.tool);
    changed |= ui
        .checkbox(&mut c.keep, RichText::new("Keep the tool body").weak())
        .on_hover_text("Keep the tool as a body of its own after the combine, as it was — the body is joined, cut or intersected with a copy of it")
        .changed();
    let sheet = is_sheet(ws, c.body);
    hint(ui, if c.keep || (sheet && c.op != BooleanOp::Join) { "The tool body stays as it is, beside the result." } else { "The tool body goes into the result." });
    if sheet {
        hint(
            ui,
            &if c.op == BooleanOp::Join {
                "A sheet joins a sheet: the two are sewn into one where they meet edge to edge — split each by the other first.".to_string()
            } else {
                format!("The sheet is trimmed to the part of it {} the solid tool, and stays a sheet for Thicken; the tool stays a body, to trim another sheet with.", if c.op == BooleanOp::Intersect { "inside" } else { "outside" })
            },
        );
    }
    changed
}

/// A Combine's bodies are chosen from their lists: a click on the part picks nothing.
pub(crate) fn pick(_ws: &mut ReverseWorkspace, _vp: Viewport, _p: Pos2, _clicked: bool, _c: &mut BooleanFeature, _pick: &mut PickWhat) -> bool {
    false
}

pub(crate) fn paint(_ws: &ReverseWorkspace, _painter: &egui::Painter, _vp: Viewport, _c: &BooleanFeature) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_combine_waits_for_a_tool_other_than_its_body() {
        let c = BooleanFeature { op: BooleanOp::Join, body: BodyId::new(2), tool: BodyId::new(2), keep: false };
        assert!(ready(&c).is_err());
        assert!(ready(&BooleanFeature { tool: BodyId::new(4), ..c }).is_ok());
    }
}
