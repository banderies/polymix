// SPDX-License-Identifier: AGPL-3.0-only
//! The fields the Mirror, Combine, Cut by plane and Offset body boxes are built from: a field
//! row, the ⓘ card, a hint line, the body select, the plane picker whose option under the
//! pointer is drawn on the part, and a number field that takes an expression. Ported from
//! ScanRuler (`plugins/reverse/ui/FeatureEditor.tsx` bodySelect, `PlaneSelect.tsx`,
//! `featureFields.tsx` Num, `src/ui/RefPicker.tsx`, `src/ui/InfoDot.tsx`), © Stefan Hermann
//! (CNC Kitchen), AGPL-3.0-only.

use egui::{Color32, Pos2, RichText, Stroke};
use glam::DVec3;
use mm_reverse::expression::{evaluate_for, ParameterKind};
use mm_reverse::frame::WorldAxis;
use mm_reverse::model::{self, BodyId, ElementGeometry, Exprs, PlaneRef};
use mm_reverse::nominal::{nominal_of, NominalKind};
use mm_reverse::parameters::{evaluate_parameters, Parameter};

use crate::reverse::sketcher::fmt_value;
use crate::reverse::ReverseWorkspace;
use crate::tools::Viewport;

/// The amber the plane under the pointer in a list is drawn in on the part (§6.9).
const HINT: Color32 = Color32::from_rgb(0xff, 0xb0, 0x20);

/// A field row: the label at the left, dim, an ⓘ after it with its card when there is one,
/// the control after.
pub(crate) fn field<R>(ui: &mut egui::Ui, label: &str, info: Option<&str>, add: impl FnOnce(&mut egui::Ui) -> R) -> R {
    ui.horizontal(|ui| {
        ui.label(RichText::new(label).weak());
        if let Some(text) = info {
            info_dot(ui, label, text);
        }
        add(ui)
    })
    .inner
}

/// The ⓘ that opens a card: its title and its text.
pub(crate) fn info_dot(ui: &mut egui::Ui, title: &str, text: &str) {
    ui.label(RichText::new("ⓘ").weak()).on_hover_ui(|ui| {
        ui.set_max_width(280.0);
        ui.strong(title);
        ui.label(text);
    });
}

/// A small dim paragraph under the fields.
pub(crate) fn hint(ui: &mut egui::Ui, text: &str) {
    ui.add(egui::Label::new(RichText::new(text).weak()).wrap());
}

/// The body select (ScanRuler's `bodySelect`): `label`, then a select of every body by name —
/// the bodies as they stood before the feature, so a Combine's tool its preview has taken in
/// is still there to choose. Returns whether the choice changed.
pub(crate) fn body_select(ui: &mut egui::Ui, ws: &ReverseWorkspace, salt: &str, label: &str, value: &mut BodyId) -> bool {
    // Named as they stood before the feature, as the list shows them.
    let bodies: Vec<(BodyId, String)> = ws.pick_set().iter().map(|b| (b.id, b.name.clone())).collect();
    let shown = bodies.iter().find(|(id, _)| id == value).map_or_else(|| "Select…".to_string(), |(_, n)| n.clone());
    let mut changed = false;
    field(ui, label, None, |ui| {
        egui::ComboBox::from_id_salt(salt).selected_text(shown).show_ui(ui, |ui| {
            for (id, name) in &bodies {
                if ui.selectable_label(id == value, name).clicked() && id != value {
                    *value = *id;
                    changed = true;
                }
            }
        });
    });
    changed
}

// ---- the plane picker ----

/// A plane a list offers: its key, its label, the plane, and the group it is listed under.
pub(crate) struct PlaneOffer {
    pub key: String,
    pub label: String,
    pub plane: PlaneRef,
    pub group: &'static str,
}

/// The key a coordinate plane on offer is told apart by (ScanRuler's `worldPlaneKey`): the
/// plane at zero and the one through the centre are two offers on one axis.
fn world_plane_key(axis: WorldAxis, offset: f64) -> String {
    let a = match axis {
        WorldAxis::X => "x",
        WorldAxis::Y => "y",
        WorldAxis::Z => "z",
    };
    format!("world:{a}:{}", mm_reverse::js_number::js_string(mm_reverse::js_number::js_round(offset * 1000.0) / 1000.0))
}

/// The key a plane reference is found among the offers by (ScanRuler's `planeKey`).
pub(crate) fn plane_key(r: &PlaneRef) -> String {
    match r {
        PlaneRef::World { axis, offset, .. } => world_plane_key(*axis, *offset),
        PlaneRef::Fitted { .. } => "fitted".into(),
        PlaneRef::BodyFace { .. } => "bodyFace".into(),
        PlaneRef::Radial { .. } => "radial".into(),
        PlaneRef::Section { id, .. } => format!("section:{id}"),
        PlaneRef::Plane { id, .. } => format!("plane:{id}"),
    }
}

/// The planes a feature can stand on (ScanRuler's `planeOptions`): the sections, the measured
/// planes, and the coordinate planes — at zero, and through the scan's centre when there is a
/// scan (`coordinatePlaneOffers`).
pub(crate) fn plane_offers(ws: &ReverseWorkspace) -> Vec<PlaneOffer> {
    let mut out = Vec::new();
    for e in &ws.part.elements {
        if let ElementGeometry::Section { .. } = e.geometry {
            out.push(PlaneOffer { key: format!("section:{}", e.id), label: e.name.clone(), plane: PlaneRef::Section { id: e.id, offset: None, up: None }, group: "Sections" });
        }
    }
    for e in &ws.part.elements {
        if let ElementGeometry::Plane { .. } = e.geometry {
            out.push(PlaneOffer { key: format!("plane:{}", e.id), label: e.name.clone(), plane: PlaneRef::Plane { id: e.id, offset: None, up: None }, group: "Planes" });
        }
    }
    // The planes at zero, and with a scan the same planes through its centre: a part not yet
    // aligned sits where the scanner left it, and a plane at the origin would miss it.
    for axis in [WorldAxis::X, WorldAxis::Y, WorldAxis::Z] {
        out.push(PlaneOffer { key: world_plane_key(axis, 0.0), label: axis.plane_name().to_string(), plane: PlaneRef::world(axis, 0.0), group: "Coordinate planes" });
    }
    if let Some(c) = ws.scan_centre() {
        for axis in [WorldAxis::X, WorldAxis::Y, WorldAxis::Z] {
            let offset = c.dot(axis.dir());
            out.push(PlaneOffer { key: world_plane_key(axis, offset), label: format!("{} through the centre", axis.plane_name()), plane: PlaneRef::world(axis, offset), group: "Coordinate planes" });
        }
    }
    out
}

/// The plane drawn on the part as the hint: the option under the pointer in an open list, or
/// the chosen plane while its box is open.
#[derive(Clone)]
struct PlaneHint {
    plane: PlaneRef,
    name: String,
}

fn hint_id() -> egui::Id {
    egui::Id::NULL.with("re_bool_plane_hint")
}

/// The plane picker (ScanRuler's `PlaneSelect` on a `RefPicker`): the chosen plane's label, or
/// the placeholder dimmed; a list of the offers by group. The option under the pointer is drawn
/// on the part, and the chosen plane while the box is open. Returns the plane chosen this
/// frame.
pub(crate) fn plane_select(ui: &mut egui::Ui, ws: &ReverseWorkspace, salt: &str, value: Option<&PlaneRef>, placeholder: &str) -> Option<PlaneRef> {
    let offers = plane_offers(ws);
    let key = value.map(plane_key);
    let chosen = key.as_ref().and_then(|k| offers.iter().find(|o| &o.key == k));
    let text = match chosen {
        Some(o) => RichText::new(&o.label),
        None => RichText::new(placeholder).weak(),
    };
    let mut picked = None;
    let mut hovered: Option<PlaneHint> = None;
    let r = egui::ComboBox::from_id_salt(salt).selected_text(text).width(190.0).show_ui(ui, |ui| {
        if offers.is_empty() {
            ui.weak("Nothing to choose from");
        }
        let mut group = "";
        for o in &offers {
            if o.group != group {
                group = o.group;
                ui.label(RichText::new(group.to_uppercase()).small().weak());
            }
            let r = ui.selectable_label(Some(&o.key) == key.as_ref(), &o.label);
            if r.hovered() {
                hovered = Some(PlaneHint { plane: o.plane.clone(), name: o.label.clone() });
            }
            if r.clicked() {
                picked = Some(o.plane.clone());
            }
        }
    });
    // The list open: the option under the pointer; closed, the chosen plane stays shown.
    let open = r.inner.is_some();
    let shown = if open { hovered } else { None }.or_else(|| match (&picked, value) {
        (Some(p), _) => Some(PlaneHint { plane: p.clone(), name: offers.iter().find(|o| o.key == plane_key(p)).map_or_else(|| mm_reverse::build::describe_plane_ref(p, &ws.part), |o| o.label.clone()) }),
        (None, Some(v)) if !open => Some(PlaneHint { plane: v.clone(), name: chosen.map_or_else(|| mm_reverse::build::describe_plane_ref(v, &ws.part), |o| o.label.clone()) }),
        _ => None,
    });
    ui.ctx().data_mut(|d| d.insert_temp(hint_id(), shown));
    picked
}

/// Take the plane hint away (a box whose plane is a face shows none).
pub(crate) fn clear_plane_hint(ctx: &egui::Context) {
    ctx.data_mut(|d| d.insert_temp::<Option<PlaneHint>>(hint_id(), None));
}

/// The middle of the model and its radius: the scan's box, else the bodies'.
fn model_ball(ws: &ReverseWorkspace) -> Option<(DVec3, f64)> {
    let mut lo = DVec3::splat(f64::INFINITY);
    let mut hi = DVec3::splat(f64::NEG_INFINITY);
    for o in ws.scan.iter().filter(|o| o.visible) {
        let b = o.mesh.bounds();
        if b.is_empty() {
            continue;
        }
        for c in [b.min, b.max] {
            let p = ws.to_part(c);
            lo = lo.min(p);
            hi = hi.max(p);
        }
    }
    if !lo.x.is_finite() {
        for b in ws.pick_set() {
            for p in &b.part_mesh.positions {
                lo = lo.min(*p);
                hi = hi.max(*p);
            }
        }
    }
    lo.x.is_finite().then(|| ((lo + hi) / 2.0, ((hi - lo).length() / 2.0).max(1.0)))
}

/// The plane hint on the part (§6.9): a square sheet 2.3 × the model's radius wide round the
/// model's middle laid on the plane, amber at 0.18 with a fat amber border, the plane's name
/// at its centre.
pub(crate) fn paint_plane_hint(ws: &ReverseWorkspace, painter: &egui::Painter, vp: Viewport) {
    let Some(h) = painter.ctx().data(|d| d.get_temp::<Option<PlaneHint>>(hint_id())).flatten() else { return };
    let Some(frame) = ws.plane_frame(&h.plane) else { return };
    let Some((centre, radius)) = model_ball(ws) else { return };
    let mid = centre - frame.normal * (centre - frame.origin).dot(frame.normal);
    let half = 1.15 * radius;
    let corners = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)].map(|(a, b)| mid + frame.u * (a * half) + frame.v * (b * half));
    let pts: Vec<Pos2> = corners.iter().filter_map(|p| ws.project(vp, *p)).collect();
    if pts.len() == 4 {
        painter.add(egui::Shape::convex_polygon(pts.clone(), HINT.gamma_multiply(0.18), Stroke::NONE));
        painter.add(egui::Shape::closed_line(pts, Stroke::new(3.0, HINT.gamma_multiply(0.9))));
    }
    if let Some(c) = ws.project(vp, mid) {
        let font = egui::FontId::proportional(12.0);
        let galley = painter.layout_no_wrap(h.name.clone(), font, Color32::BLACK);
        let rect = egui::Rect::from_center_size(c, galley.size() + egui::vec2(10.0, 4.0));
        painter.rect_filled(rect, 3.0, HINT);
        painter.galley(rect.center() - galley.size() / 2.0, galley, Color32::BLACK);
    }
}

// ---- a number that takes an expression ----

/// A labelled number (ScanRuler's `Num`): its label and ⓘ, the round number it was probably
/// meant as on a chip when no expression drives it, and a field that takes a number or an
/// expression over the part's parameters — kept under `slot` in `exprs`. What was typed stays
/// in the field; a bad expression stays with its reason under the field until put right.
/// Returns whether the value changed.
pub(crate) fn num_field(ui: &mut egui::Ui, salt: &str, label: &str, info: &str, value: &mut f64, exprs: &mut Exprs, slot: &str, ws: &ReverseWorkspace) -> bool {
    let id = egui::Id::NULL.with("re_bool_num").with(salt);
    let expr = exprs.get(slot).map(str::to_string);
    let mut changed = false;
    // What the field shows while it is not being typed in: the expression, or the number.
    let shown = expr.clone().unwrap_or_else(|| fmt_value(*value));
    let mut text: String = ui.ctx().data(|d| d.get_temp::<String>(id)).unwrap_or_else(|| shown.clone());
    let mut error: Option<String> = ui.ctx().data(|d| d.get_temp::<Option<String>>(id.with("error"))).flatten();
    field(ui, label, Some(info), |ui| {
        if expr.is_none() {
            if let Some(round) = nominal_of(*value, NominalKind::Length) {
                if ui.add(egui::Button::new(RichText::new(format!("→ {} mm", fmt_value(round))).small().color(ui.visuals().hyperlink_color)).small()).on_hover_text("The round number this was probably meant as — click to take it").clicked() {
                    *value = round;
                    changed = true;
                }
            }
        }
        let r = ui.add(egui::TextEdit::singleline(&mut text).desired_width(90.0).id(id.with("edit")));
        if expr.is_some() {
            ui.label(RichText::new(format!("= {}", fmt_value(*value))).italics().weak()).on_hover_text(format!("{} = {}", expr.as_deref().unwrap_or(""), fmt_value(*value)));
        } else {
            ui.label(RichText::new("mm").italics().weak());
        }
        let escape = r.has_focus() && ui.input(|i| i.key_pressed(egui::Key::Escape));
        if escape {
            text = shown.clone();
            error = None;
        } else if r.lost_focus() || (r.has_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter))) {
            let typed = text.trim().to_string();
            if typed != shown {
                let table = evaluate_parameters(&part_parameters(ws));
                match evaluate_for(&typed, ParameterKind::Length, &table, false) {
                    Ok(v) => {
                        *value = v.value;
                        if mm_reverse::expression::is_plain_number(&typed) {
                            exprs.remove(slot);
                        } else {
                            exprs.insert(slot, typed.clone());
                        }
                        error = None;
                        changed = true;
                    }
                    Err(e) => error = Some(e.message),
                }
            }
        }
        if !r.has_focus() && error.is_none() {
            text = if changed { exprs.get(slot).map(str::to_string).unwrap_or_else(|| fmt_value(*value)) } else { shown.clone() };
        }
    });
    if let Some(e) = &error {
        ui.colored_label(Color32::from_rgb(226, 107, 79), e);
    }
    ui.ctx().data_mut(|d| {
        d.insert_temp(id, text);
        d.insert_temp(id.with("error"), error);
    });
    changed
}

/// The part's parameters as the expressions read them.
fn part_parameters(ws: &ReverseWorkspace) -> Vec<Parameter> {
    ws.part
        .parameters
        .iter()
        .map(|p| Parameter {
            id: p.id,
            name: p.name.clone(),
            kind: match p.kind {
                model::ParameterKind::Length => ParameterKind::Length,
                model::ParameterKind::Angle => ParameterKind::Angle,
                model::ParameterKind::Integer => ParameterKind::Integer,
                model::ParameterKind::Number => ParameterKind::Number,
            },
            expression: p.expression.clone(),
            comment: p.comment.clone(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_a_plane_as_the_lists_find_it() {
        // ScanRuler's worldPlaneKey: the offset to the micrometre, so the plane at zero and the
        // one through the centre are two offers on one axis.
        assert_eq!(world_plane_key(WorldAxis::X, 0.0), "world:x:0");
        assert_eq!(world_plane_key(WorldAxis::Z, 12.34567), "world:z:12.346");
        assert_eq!(plane_key(&PlaneRef::world(WorldAxis::Y, -3.5)), "world:y:-3.5");
        assert_eq!(plane_key(&PlaneRef::Plane { id: 4, offset: None, up: None }), "plane:4");
        assert_eq!(plane_key(&PlaneRef::Section { id: 2, offset: Some(1.0), up: None }), "section:2");
        assert_eq!(plane_key(&mm_reverse::build::no_plane()), "fitted");
    }
}
