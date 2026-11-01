// SPDX-License-Identifier: AGPL-3.0-only
//! The Pattern feature's box: the body, circular about an axis or linear along a direction, how
//! many copies over what angle or at what spacing, and for a linear one a second direction —
//! a grid. The axis under the pointer in a picker, or the one chosen, is drawn on the part.
//! Ported from ScanRuler (`plugins/reverse/ui/FeatureEditor.tsx` the pattern fields,
//! `state/reFeatureSlice.ts` startFeature, `state/reDrafts.ts` featureOfDraft), © Stefan
//! Hermann (CNC Kitchen), AGPL-3.0-only.

use egui::Pos2;
use mm_reverse::frame::WorldAxis;
use mm_reverse::model::{AxisRef, FeatureKind, PatternFeature, PatternMode};

use super::loft_kit::{axis_picker, body_select, hint_axis, label_with_info, last_solid, paint_axis_hint};
use crate::reverse::{PickWhat, ReverseWorkspace};
use crate::tools::Viewport;

pub(crate) const READY: bool = true;
pub(crate) const BUTTON: (&str, &str) = ("Pattern", "Copies of the body about an axis, along a direction, or in a grid");
pub(crate) const FIRST_PICK: PickWhat = PickWhat::Nothing;

/// The rows a grid had when its second direction was let go, to take up again when one is
/// chosen: two rows ten apart at first.
fn rows_kept(ctx: &egui::Context) -> (u32, f64) {
    ctx.data(|d| d.get_temp::<(u32, f64)>(egui::Id::NULL.with("re_pattern_rows"))).unwrap_or((2, 10.0))
}

/// A linear pattern's second direction, its rows and their expressions, put aside while the
/// pattern is circular — which has none, and keeps none in the part (ScanRuler's draft holds
/// them, and its feature leaves them out) — and taken up again when it is linear once more.
type Grid = (AxisRef, Option<u32>, Option<f64>, Option<String>, Option<String>);

fn switch_grid(ctx: &egui::Context, owner: Option<u32>, x: &mut PatternFeature, to: PatternMode) {
    let id = egui::Id::NULL.with(("re_pattern_grid", owner));
    if to == PatternMode::Circular {
        if let Some(axis2) = x.axis2.take() {
            let kept: Grid = (axis2, x.count2.take(), x.step2.take(), x.exprs.remove("count2"), x.exprs.remove("step2"));
            ctx.data_mut(|d| d.insert_temp(id, kept));
        }
    } else if let Some((axis2, count2, step2, e2, s2)) = ctx.data_mut(|d| {
        let kept = d.get_temp::<Grid>(id);
        d.remove::<Grid>(id);
        kept
    }) {
        x.axis2 = Some(axis2);
        x.count2 = count2;
        x.step2 = step2;
        if let Some(e) = e2 {
            x.exprs.insert("count2", e);
        }
        if let Some(e) = s2 {
            x.exprs.insert("step2", e);
        }
    }
}

/// The last solid body, circular, four copies over a full turn — about the part's Z axis,
/// where ScanRuler's box waits for an axis to be chosen.
pub(crate) fn new_feature(ws: &ReverseWorkspace) -> Result<FeatureKind, String> {
    let body = last_solid(ws).ok_or("A pattern copies a body: make a body first.")?;
    Ok(FeatureKind::Pattern(PatternFeature { body, mode: PatternMode::Circular, axis: AxisRef::World { axis: WorldAxis::Z }, count: 4, step: 360.0, axis2: None, count2: None, step2: None, exprs: Default::default() }))
}

pub(crate) fn ui(ws: &mut ReverseWorkspace, ui: &mut egui::Ui, x: &mut PatternFeature, pick: &mut PickWhat) -> bool {
    *pick = PickWhat::Nothing;
    let owner = ws.preview_feature;
    let mut changed = body_select(ui, ws, "re_pattern_body", &mut x.body);
    ui.horizontal(|ui| {
        ui.label("Kind");
        let name = |m: PatternMode| if m == PatternMode::Circular { "Circular, about an axis" } else { "Linear, along a direction" };
        egui::ComboBox::from_id_salt("re_pattern_mode").selected_text(name(x.mode)).show_ui(ui, |ui| {
            for m in [PatternMode::Circular, PatternMode::Linear] {
                if ui.selectable_label(x.mode == m, name(m)).clicked() && x.mode != m {
                    switch_grid(ui.ctx(), owner, x, m);
                    x.mode = m;
                    x.step = if m == PatternMode::Circular { 360.0 } else { 10.0 };
                    x.exprs.remove("step");
                    changed = true;
                }
            }
        });
    });
    let mut hovered = None;
    ui.horizontal(|ui| {
        ui.label(if x.mode == PatternMode::Circular { "About" } else { "Along" });
        let (taken, over) = axis_picker(ui, ws, "re_pattern_axis", Some(&x.axis), "Choose…");
        if let Some(a) = taken {
            changed |= x.axis != a;
            x.axis = a;
        }
        hovered = hovered.or(over);
    });
    ui.horizontal(|ui| {
        label_with_info(ui, "Count", "Copies in all, the body itself counted");
        let mut n = x.count as f64;
        if ui.add(egui::DragValue::new(&mut n).speed(0.1).max_decimals(0)).changed() {
            x.count = n.abs().round().max(2.0) as u32;
            x.exprs.remove("count");
            changed = true;
        }
    });
    ui.horizontal(|ui| {
        let (label, hint, suffix) = if x.mode == PatternMode::Circular {
            ("Over", "The angle from the first copy to the last: 360° spreads them evenly round", "°")
        } else {
            ("Spacing", "The distance from one copy to the next", " mm")
        };
        label_with_info(ui, label, hint);
        let mut v = x.step;
        if ui.add(egui::DragValue::new(&mut v).speed(0.1).suffix(suffix)).changed() {
            x.step = v.abs();
            x.exprs.remove("step");
            changed = true;
        }
    });
    if x.mode == PatternMode::Linear {
        ui.horizontal(|ui| {
            ui.label("Then along");
            let (taken, over) = axis_picker(ui, ws, "re_pattern_axis2", x.axis2.as_ref(), "Nothing — one row");
            if let Some(a) = taken {
                if x.axis2.is_none() {
                    let (n, s) = rows_kept(ui.ctx());
                    x.count2 = Some(n);
                    x.step2 = Some(s);
                }
                changed |= x.axis2 != Some(a);
                x.axis2 = Some(a);
            }
            hovered = hovered.or(over);
        });
        if x.axis2.is_some() {
            if ui.small_button("× one row").on_hover_text("One row only — no second direction").clicked() {
                let kept = (x.count2.unwrap_or(2), x.step2.unwrap_or(10.0));
                ui.ctx().data_mut(|d| d.insert_temp(egui::Id::NULL.with("re_pattern_rows"), kept));
                x.axis2 = None;
                x.count2 = None;
                x.step2 = None;
                changed = true;
            } else {
                ui.horizontal(|ui| {
                    label_with_info(ui, "Rows", "Rows along the second direction, the first counted — a grid");
                    let mut n = x.count2.unwrap_or(2) as f64;
                    if ui.add(egui::DragValue::new(&mut n).speed(0.1).max_decimals(0)).changed() {
                        x.count2 = Some(n.abs().round().max(1.0) as u32);
                        x.exprs.remove("count2");
                        changed = true;
                    }
                });
                ui.horizontal(|ui| {
                    label_with_info(ui, "Row spacing", "The distance from one row to the next");
                    let mut v = x.step2.unwrap_or(10.0);
                    if ui.add(egui::DragValue::new(&mut v).speed(0.1).suffix(" mm")).changed() {
                        x.step2 = Some(v.abs());
                        x.exprs.remove("step2");
                        changed = true;
                    }
                });
            }
        }
    }
    hint_axis(ui.ctx(), hovered);
    changed
}

pub(crate) fn pick(_ws: &mut ReverseWorkspace, _vp: Viewport, _p: Pos2, _clicked: bool, _x: &mut PatternFeature, _pick: &mut PickWhat) -> bool {
    false
}

/// The axis the pointer is on in a picker, else the one chosen, drawn on the part.
pub(crate) fn paint(ws: &ReverseWorkspace, painter: &egui::Painter, vp: Viewport, x: &PatternFeature) {
    paint_axis_hint(ws, painter, vp, Some(&x.axis));
}

#[cfg(test)]
mod tests {
    //! The Pattern box's second direction across a change of kind: ScanRuler's draft keeps it,
    //! and its feature leaves it out of a circular pattern (`state/reDrafts.ts` featureOfDraft).
    use super::*;

    #[test]
    fn a_circular_pattern_keeps_no_grid_and_a_linear_one_takes_its_rows_up_again() {
        let ctx = egui::Context::default();
        let mut x = PatternFeature { body: mm_reverse::model::BodyId::new(2), mode: PatternMode::Linear, axis: AxisRef::World { axis: WorldAxis::X }, count: 3, step: 5.0, axis2: Some(AxisRef::World { axis: WorldAxis::Y }), count2: Some(4), step2: Some(7.5), exprs: Default::default() };
        x.exprs.insert("step2", "pitch * 2");
        switch_grid(&ctx, Some(9), &mut x, PatternMode::Circular);
        assert_eq!((x.axis2, x.count2, x.step2, x.exprs.get("step2")), (None, None, None, None));
        // Another box's grid is its own.
        let mut other = x.clone();
        switch_grid(&ctx, Some(10), &mut other, PatternMode::Linear);
        assert_eq!(other.axis2, None);
        switch_grid(&ctx, Some(9), &mut x, PatternMode::Linear);
        assert_eq!((x.axis2, x.count2, x.step2, x.exprs.get("step2")), (Some(AxisRef::World { axis: WorldAxis::Y }), Some(4), Some(7.5), Some("pitch * 2")));
    }
}
