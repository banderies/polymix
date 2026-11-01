// SPDX-License-Identifier: AGPL-3.0-only
//! The Extract feature's box: the measured element or the shadow section a body is taken
//! from, a plane's slab thickness, and the result. Ported from ScanRuler
//! (`plugins/reverse/ui/FeatureEditor.tsx` the extract's fields, `state/reFeatureSlice.ts` its
//! defaults), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.

use egui::Pos2;
use mm_reverse::model::{BodyOp, ElementGeometry, ExtractFeature, FeatureKind};

use super::surf_common::{can_cut, mm, num, op_field};
use crate::reverse::{PickWhat, ReverseWorkspace};
use crate::tools::Viewport;

/// Whether the editor is built: its button stands with the others.
pub(crate) const READY: bool = true;
/// The panel's button: its label and what it does.
pub(crate) const BUTTON: (&str, &str) = ("Extract", "A measured element as a body: a plane as a slab, a cylinder, cone or sphere as a solid");
/// What a click picks first when the box opens.
pub(crate) const FIRST_PICK: PickWhat = PickWhat::Nothing;

/// A new extract: nothing chosen yet, a 2 mm slab, a body of its own — an element is
/// extracted as a tool more often than as a piece of the part.
pub(crate) fn new_feature(_ws: &ReverseWorkspace) -> Result<FeatureKind, String> {
    Ok(FeatureKind::Extract(ExtractFeature { element: None, section: None, thickness: 2.0, op: BodyOp::New, target: None, exprs: Default::default() }))
}

pub(crate) fn ui(ws: &mut ReverseWorkspace, ui: &mut egui::Ui, x: &mut ExtractFeature, _pick: &mut PickWhat) -> bool {
    let mut changed = false;
    let options: Vec<(u32, String)> = ws
        .part
        .elements
        .iter()
        .filter(|e| matches!(e.geometry, ElementGeometry::Plane { .. } | ElementGeometry::Cylinder { .. } | ElementGeometry::Cone { .. } | ElementGeometry::Sphere { .. } | ElementGeometry::Torus { .. }))
        .map(|e| (e.id, e.name.clone()))
        .collect();
    let shadows: Vec<(u32, String)> = ws.part.elements.iter().filter(|e| matches!(e.geometry, ElementGeometry::Section { .. })).map(|e| (e.id, e.name.clone())).collect();
    let shown = match (x.section, x.element) {
        (Some(s), _) => shadows.iter().find(|o| o.0 == s).map_or("(gone)".to_string(), |o| o.1.clone()),
        (None, Some(e)) => options.iter().find(|o| o.0 == e).map_or("(gone)".to_string(), |o| o.1.clone()),
        _ => "Click an element on the part, or select…".to_string(),
    };
    ui.horizontal(|ui| {
        ui.label("From");
        egui::ComboBox::from_id_salt("re_extract_from").selected_text(shown).width(220.0).show_ui(ui, |ui| {
            if !options.is_empty() {
                ui.weak("Elements");
                for (id, name) in &options {
                    if ui.selectable_label(x.element == Some(*id) && x.section.is_none(), name).clicked() {
                        x.element = Some(*id);
                        x.section = None;
                        changed = true;
                    }
                }
            }
            if !shadows.is_empty() {
                ui.weak("Shadow sections");
                for (id, name) in &shadows {
                    if ui.selectable_label(x.section == Some(*id), name).clicked() {
                        x.section = Some(*id);
                        x.element = None;
                        changed = true;
                    }
                }
            }
            if options.is_empty() && shadows.is_empty() {
                ui.weak("No element measured yet.");
            }
        });
    });
    if let Some(sec) = x.section {
        // The shadow's thickness, as the build last took it off the scan: the block's extent
        // along the section's normal.
        let frame = ws.part.element(sec).and_then(|e| if let ElementGeometry::Section { frame } = &e.geometry { Some(*frame) } else { None });
        let made = ws.preview_feature.and_then(|f| ws.bodies.iter().find(|b| b.id.maker() == f));
        let thick = frame.zip(made).map(|(f, b)| {
            let (lo, hi) = b.part_mesh.positions.iter().map(|p| (*p - f.origin).dot(f.normal)).fold((f64::INFINITY, f64::NEG_INFINITY), |a, d| (a.0.min(d), a.1.max(d)));
            hi - lo
        });
        match thick {
            Some(t) if t > 0.0 => ui.weak(format!("The outline stood up through the part, {} thick — it follows the shadow as 3D Measure takes it again.", mm(t, 2))),
            _ => ui.weak("The shadow is still being taken."),
        };
    }
    let plane = x.element.and_then(|id| ws.part.element(id)).is_some_and(|e| matches!(e.geometry, ElementGeometry::Plane { .. }));
    if plane && x.section.is_none() {
        changed |= num(ui, "Thickness", &mut x.thickness, "mm", 0.001..=1000.0, 0.05, "A plane becomes a slab this thick, centred on it");
    }
    changed |= op_field(ui, &mut x.op, can_cut(ws));
    changed
}

pub(crate) fn pick(_ws: &mut ReverseWorkspace, _vp: Viewport, _p: Pos2, _clicked: bool, _x: &mut ExtractFeature, _pick: &mut PickWhat) -> bool {
    false
}

pub(crate) fn paint(_ws: &ReverseWorkspace, _painter: &egui::Painter, _vp: Viewport, _x: &ExtractFeature) {}
