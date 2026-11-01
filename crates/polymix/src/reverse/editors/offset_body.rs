// SPDX-License-Identifier: AGPL-3.0-only
//! The Offset body feature's box: the body, how far it grows all round (in, negative) — a
//! number or an expression — and whether the body stays as it is beside the offset or becomes
//! it. Ported from ScanRuler (`plugins/reverse/ui/FeatureEditor.tsx` the offsetBody case,
//! `featureFields.tsx` Num, `state/reFeatureSlice.ts` its defaults), © Stefan Hermann (CNC
//! Kitchen), AGPL-3.0-only.

use egui::{Pos2, RichText};
use mm_reverse::model::{Exprs, FeatureKind, OffsetBodyFeature};

use super::bool_fields::{body_select, hint, num_field};
use crate::reverse::{PickWhat, ReverseWorkspace};
use crate::tools::Viewport;

/// Whether the editor is built: its button stands with the others.
pub(crate) const READY: bool = true;
/// The panel's button: its label and what it does.
pub(crate) const BUTTON: (&str, &str) = ("Offset body", "Grow a body all round by a distance, or shrink it — kept beside it as a body of its own, a keep-out round a part or the pocket that holds it with clearance");
/// What a click picks first when the box opens.
pub(crate) const FIRST_PICK: PickWhat = PickWhat::Nothing;

/// A new offset: a keep-out a millimetre round the last solid body, the body left as it is.
pub(crate) fn new_feature(ws: &ReverseWorkspace) -> Result<FeatureKind, String> {
    let body = super::boolean::last_solid(ws).ok_or("An offset works on a body: make a body first.")?;
    Ok(FeatureKind::OffsetBody(OffsetBodyFeature { body, distance: 1.0, keep: true, exprs: Exprs::new() }))
}

/// The offset can be made as soon as it has a body; a distance of nothing says so in the build.
pub(crate) fn ready(_x: &OffsetBodyFeature) -> Result<(), String> {
    Ok(())
}

pub(crate) fn ui(ws: &mut ReverseWorkspace, ui: &mut egui::Ui, x: &mut OffsetBodyFeature, _pick: &mut PickWhat) -> bool {
    let mut changed = body_select(ui, ws, "re_offset_body", "Body", &mut x.body);
    changed |= num_field(ui, "re_offset_distance", "Distance", "How far all round — out, or in with a negative one. The convex edges come out rounded, as an offset rounds them", &mut x.distance, &mut x.exprs, "distance", ws);
    changed |= ui
        .checkbox(&mut x.keep, RichText::new("Keep the body — the offset a new one").weak())
        .on_hover_text("Leave the body as it is and make the offset a body of its own — a keep-out round a part, or what is cut from a case to hold it with clearance (Combine, Cut by, the offset as the tool)")
        .changed();
    hint(ui, if x.keep { "Cut it from a case with Combine to leave a pocket with the clearance." } else { "The body becomes the offset." });
    changed
}

/// An offset's body is chosen from its list: a click on the part picks nothing.
pub(crate) fn pick(_ws: &mut ReverseWorkspace, _vp: Viewport, _p: Pos2, _clicked: bool, _x: &mut OffsetBodyFeature, _pick: &mut PickWhat) -> bool {
    false
}

pub(crate) fn paint(_ws: &ReverseWorkspace, _painter: &egui::Painter, _vp: Viewport, _x: &OffsetBodyFeature) {}
