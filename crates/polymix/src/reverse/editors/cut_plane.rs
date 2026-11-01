// SPDX-License-Identifier: AGPL-3.0-only
//! The Cut by plane feature's box: the body, the plane it is cut at — chosen from the list of
//! sections, measured planes and coordinate planes, the one under the pointer drawn on the part
//! — and the key that keeps the other side. Ported from ScanRuler
//! (`plugins/reverse/ui/FeatureEditor.tsx` the cutPlane case, `PlaneSelect.tsx`,
//! `state/reFeatureSlice.ts` its defaults), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.

use egui::Pos2;
use mm_reverse::build::{no_plane, plane_chosen};
use mm_reverse::model::{CutPlaneFeature, Exprs, FeatureKind};

use super::bool_fields::{body_select, field, paint_plane_hint, plane_select};
use crate::reverse::{PickWhat, ReverseWorkspace};
use crate::tools::Viewport;

/// Whether the editor is built: its button stands with the others.
pub(crate) const READY: bool = true;
/// The panel's button: its label and what it does.
pub(crate) const BUTTON: (&str, &str) = ("Cut by plane", "Cut the body at a plane and keep one side");
/// What a click picks first when the box opens.
pub(crate) const FIRST_PICK: PickWhat = PickWhat::Nothing;

/// A new cut: on the last solid body, no plane chosen yet, the side the plane's normal points
/// to kept.
pub(crate) fn new_feature(ws: &ReverseWorkspace) -> Result<FeatureKind, String> {
    let body = super::boolean::last_solid(ws).ok_or("A cut works on a body: make a body first.")?;
    Ok(FeatureKind::CutPlane(CutPlaneFeature { body, plane: no_plane(), keep: 1, exprs: Exprs::new() }))
}

/// Whether the cut can be made: a plane chosen.
pub(crate) fn ready(x: &CutPlaneFeature) -> Result<(), String> {
    if plane_chosen(&x.plane) {
        Ok(())
    } else {
        Err("Choose the plane to cut at.".into())
    }
}

pub(crate) fn ui(ws: &mut ReverseWorkspace, ui: &mut egui::Ui, x: &mut CutPlaneFeature, _pick: &mut PickWhat) -> bool {
    let mut changed = body_select(ui, ws, "re_cut_body", "Body", &mut x.body);
    field(ui, "At", None, |ui| {
        let value = plane_chosen(&x.plane).then(|| x.plane.clone());
        if let Some(p) = plane_select(ui, ws, "re_cut_plane", value.as_ref(), "Choose the plane…") {
            x.plane = p;
            changed = true;
        }
    });
    let flip = egui::Button::new("⇅  Keep the other side").min_size(egui::vec2(ui.available_width(), 0.0));
    if ui.add(flip).clicked() {
        x.keep = if x.keep > 0 { -1 } else { 1 };
        changed = true;
    }
    changed
}

/// A cut's plane is chosen from its list: a click on the part picks nothing.
pub(crate) fn pick(_ws: &mut ReverseWorkspace, _vp: Viewport, _p: Pos2, _clicked: bool, _x: &mut CutPlaneFeature, _pick: &mut PickWhat) -> bool {
    false
}

/// The plane chosen, or the one under the pointer in the list, drawn on the part.
pub(crate) fn paint(ws: &ReverseWorkspace, painter: &egui::Painter, vp: Viewport, _x: &CutPlaneFeature) {
    paint_plane_hint(ws, painter, vp);
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_reverse::frame::WorldAxis;
    use mm_reverse::model::{BodyId, PlaneRef};

    #[test]
    fn a_cut_waits_for_its_plane() {
        let mut x = CutPlaneFeature { body: BodyId::new(2), plane: no_plane(), keep: 1, exprs: Exprs::new() };
        assert_eq!(ready(&x), Err("Choose the plane to cut at.".to_string()));
        x.plane = PlaneRef::world(WorldAxis::Z, 4.0);
        assert_eq!(ready(&x), Ok(()));
    }
}
