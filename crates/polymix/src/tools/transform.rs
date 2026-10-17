//! Transform (translate / rotate / scale) and Units / Dimensions tools.

use super::*;
use crate::ui_util;
use glam::{DMat4, DQuat, EulerRot};

pub struct TransformTool {
    object: Option<ObjectId>,
    source: std::sync::Arc<Mesh>,
    preview: Option<Mesh>,
    translate: DVec3,
    rotation: DQuat,
    scale: DVec3,
    uniform: bool,
    pivot: DVec3,
    dirty: bool,
    drag: Option<GizmoDrag>,
    hover: Option<Handle>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Handle {
    Move(usize),
    Rotate(usize),
    Scale,
}

#[derive(Clone, Copy, Debug)]
struct GizmoDrag {
    handle: Handle,
    start: egui::Pos2,
    translate: DVec3,
    rotation: DQuat,
    scale: DVec3,
}

const AXES: [DVec3; 3] = [DVec3::X, DVec3::Y, DVec3::Z];
const AXIS_COLORS: [egui::Color32; 3] = [
    egui::Color32::from_rgb(230, 80, 80),
    egui::Color32::from_rgb(90, 200, 90),
    egui::Color32::from_rgb(90, 140, 240),
];

fn dist_to_segment(p: egui::Pos2, a: egui::Pos2, b: egui::Pos2) -> f32 {
    let ab = b - a;
    let t = ((p - a).dot(ab) / ab.length_sq().max(1e-6)).clamp(0.0, 1.0);
    p.distance(a + ab * t)
}

impl TransformTool {
    pub fn new(cx: &mut ToolCx) -> Self {
        let (object, source) = match cx.scene.active() {
            Some(o) => (Some(o.id), o.mesh.clone()),
            None => (None, std::sync::Arc::new(Mesh::new())),
        };
        let pivot = source.bounds().center();
        Self {
            object,
            source,
            preview: None,
            translate: DVec3::ZERO,
            rotation: DQuat::IDENTITY,
            scale: DVec3::ONE,
            uniform: true,
            pivot,
            dirty: false,
            drag: None,
            hover: None,
        }
    }

    fn matrix(&self) -> DMat4 {
        DMat4::from_translation(self.pivot + self.translate)
            * DMat4::from_quat(self.rotation)
            * DMat4::from_scale(self.scale)
            * DMat4::from_translation(-self.pivot)
    }

    fn rebuild(&mut self) {
        let m = self.matrix();
        let mut mesh = (*self.source).clone();
        mesh.transform(|p| m.transform_point3(p));
        if m.determinant() < 0.0 {
            mesh.reverse_orientation();
        }
        self.preview = Some(mesh);
        self.dirty = false;
    }

    fn is_identity(&self) -> bool {
        self.translate == DVec3::ZERO && self.rotation.angle_between(DQuat::IDENTITY) < 1e-12 && self.scale == DVec3::ONE
    }

    fn commit(&mut self, cx: &mut ToolCx) -> bool {
        let Some(id) = self.object else { return true };
        if self.is_identity() {
            return true;
        }
        if !cx.scene.get(id).map(|o| std::sync::Arc::ptr_eq(&o.mesh, &self.source)).unwrap_or(false) {
            *cx.status = "Transform cancelled: the object changed while the tool was open.".into();
            return true;
        }
        if self.dirty || self.preview.is_none() {
            self.rebuild();
        }
        cx.history.push("Transform", cx.scene);
        if let (Some(o), Some(m)) = (cx.scene.get_mut(id), self.preview.take()) {
            let sel = o.selection.clone();
            o.mesh = std::sync::Arc::new(m);
            o.selection = sel;
            o.touch_selection();
        }
        true
    }

    /// Gizmo size in world units (constant on screen).
    fn gizmo_len(&self, cx: &ToolCx) -> f64 {
        cx.world_per_point(self.pivot + self.translate) * 90.0
    }

    fn hit_test(&self, cx: &ToolCx, p: egui::Pos2) -> Option<Handle> {
        let c3 = self.pivot + self.translate;
        let c = cx.project(c3)?;
        if (p - c).abs().max_elem() < 8.0 {
            return Some(Handle::Scale);
        }
        let len = self.gizmo_len(cx);
        let mut best: Option<(f32, Handle)> = None;
        for (i, a) in AXES.iter().enumerate() {
            if let Some(tip) = cx.project(c3 + *a * len) {
                let d = dist_to_segment(p, c, tip);
                if d < 8.0 && best.map(|b| d < b.0).unwrap_or(true) {
                    best = Some((d, Handle::Move(i)));
                }
            }
            // Rotation ring of radius 0.7 * len around axis i.
            let (u, v) = mm_core::geom::orthonormal_basis(*a);
            let pts: Vec<egui::Pos2> = (0..=48)
                .filter_map(|k| {
                    let t = k as f64 / 48.0 * std::f64::consts::TAU;
                    cx.project(c3 + (u * t.cos() + v * t.sin()) * len * 0.7)
                })
                .collect();
            for w in pts.windows(2) {
                let d = dist_to_segment(p, w[0], w[1]);
                if d < 6.0 && best.map(|b| d < b.0).unwrap_or(true) {
                    best = Some((d, Handle::Rotate(i)));
                }
            }
        }
        best.map(|b| b.1)
    }

    fn drag_to(&mut self, cx: &ToolCx, p: egui::Pos2) {
        let Some(d) = self.drag else { return };
        let c3 = self.pivot + d.translate;
        let Some(c) = cx.project(c3) else { return };
        match d.handle {
            Handle::Move(i) => {
                let len = self.gizmo_len(cx);
                let Some(tip) = cx.project(c3 + AXES[i] * len) else { return };
                let axis_px = tip - c;
                let px_len = axis_px.length().max(1e-3);
                let along = (p - d.start).dot(axis_px / px_len);
                self.translate = d.translate + AXES[i] * (along as f64 / px_len as f64 * len);
            }
            Handle::Rotate(i) => {
                let a0 = (d.start - c).angle();
                let a1 = (p - c).angle();
                let mut ang = (a1 - a0) as f64;
                // Screen angles run clockwise; flip when the axis points away from the viewer.
                let toward_viewer = AXES[i].dot(-cx.camera.forward()) > 0.0;
                if toward_viewer {
                    ang = -ang;
                }
                self.rotation = DQuat::from_axis_angle(AXES[i], ang) * d.rotation;
            }
            Handle::Scale => {
                let f = (1.0 + (d.start.y - p.y) as f64 * 0.01).clamp(0.05, 20.0);
                self.scale = d.scale * f;
            }
        }
        self.dirty = true;
    }
}

impl Tool for TransformTool {
    fn name(&self) -> &'static str {
        "Transform"
    }

    fn panel(&mut self, ui: &mut egui::Ui, cx: &mut ToolCx) -> ToolOutcome {
        ui_util::section(ui, "Transform");
        if self.object.is_none() {
            ui.label("No active object.");
            return if ui.button("Close").clicked() { ToolOutcome::Close } else { ToolOutcome::Continue };
        }
        ui.small("Drag the arrows to move, the rings to rotate, the centre square to scale — or type values.");
        let mut changed = false;
        let step = (self.source.bounds().diagonal() * 0.002).max(1e-4);
        ui.label("Translate");
        ui.horizontal(|ui| {
            for (i, n) in ["X", "Y", "Z"].iter().enumerate() {
                ui.label(*n);
                changed |= ui.add(egui::DragValue::new(&mut self.translate[i]).speed(step).max_decimals(3)).changed();
            }
        });
        ui.label("Rotate (degrees, XYZ)");
        let (rx, ry, rz) = self.rotation.to_euler(EulerRot::XYZ);
        let mut deg = [rx.to_degrees() + 0.0, ry.to_degrees() + 0.0, rz.to_degrees() + 0.0];
        let mut rot_changed = false;
        ui.horizontal(|ui| {
            for (i, n) in ["X", "Y", "Z"].iter().enumerate() {
                ui.label(*n);
                rot_changed |= ui.add(egui::DragValue::new(&mut deg[i]).speed(0.5).range(-360.0..=360.0).max_decimals(2)).changed();
            }
        });
        if rot_changed {
            self.rotation = DQuat::from_euler(EulerRot::XYZ, deg[0].to_radians(), deg[1].to_radians(), deg[2].to_radians());
            changed = true;
        }
        ui.horizontal(|ui| {
            ui.label("Scale");
            changed |= ui.checkbox(&mut self.uniform, "uniform").changed();
        });
        ui.horizontal(|ui| {
            if self.uniform {
                let mut s = self.scale.x;
                if ui.add(egui::DragValue::new(&mut s).speed(0.005).range(1e-6..=1e6).max_decimals(4)).changed() {
                    self.scale = DVec3::splat(s);
                    changed = true;
                }
            } else {
                for (i, n) in ["X", "Y", "Z"].iter().enumerate() {
                    ui.label(*n);
                    changed |= ui.add(egui::DragValue::new(&mut self.scale[i]).speed(0.005).range(1e-6..=1e6).max_decimals(4)).changed();
                }
            }
        });
        ui.add_space(4.0);
        ui.horizontal_wrapped(|ui| {
            let b = self.source.bounds();
            if ui.button("Center at origin").clicked() {
                self.translate = -b.center();
                changed = true;
            }
            if ui.button("Drop to ground").on_hover_text("Move the lowest point onto the ground plane").clicked() {
                let up = cx.camera.world_up();
                let low = if up == DVec3::Z { b.min.z } else { b.min.y };
                self.translate = -up * low;
                changed = true;
            }
            if ui.button("Align principal axes").on_hover_text("Rotate so the object's longest extent lies along X").clicked() {
                let mut tmp = (*self.source).clone();
                let m = mm_edit::align_to_principal_axes(&mut tmp);
                let (_, r, _) = m.to_scale_rotation_translation();
                self.rotation = r;
                changed = true;
            }
            if ui.button("Reset").clicked() {
                self.translate = DVec3::ZERO;
                self.rotation = DQuat::IDENTITY;
                self.scale = DVec3::ONE;
                changed = true;
            }
        });
        if changed {
            self.dirty = true;
        }
        if self.dirty {
            self.rebuild();
        }
        let (accept, cancel) = ui_util::accept_cancel(ui, true);
        if accept && self.commit(cx) {
            return ToolOutcome::Close;
        }
        if cancel {
            return ToolOutcome::Close;
        }
        ToolOutcome::Continue
    }

    fn pointer(&mut self, input: &PointerInput, cx: &mut ToolCx) {
        let Some(pos) = input.pos else { return };
        if self.drag.is_none() {
            self.hover = self.hit_test(cx, pos);
        }
        if input.pressed {
            if let Some(h) = self.hit_test(cx, pos) {
                self.drag = Some(GizmoDrag { handle: h, start: pos, translate: self.translate, rotation: self.rotation, scale: self.scale });
            }
        }
        if input.down && self.drag.is_some() {
            self.drag_to(cx, pos);
            if self.dirty {
                self.rebuild();
            }
        }
        if input.released {
            self.drag = None;
        }
    }

    fn paint(&mut self, painter: &egui::Painter, cx: &mut ToolCx) {
        if self.object.is_none() {
            return;
        }
        let c3 = self.pivot + self.translate;
        let Some(c) = cx.project(c3) else { return };
        let len = self.gizmo_len(cx);
        let active = self.drag.map(|d| d.handle).or(self.hover);
        for (i, a) in AXES.iter().enumerate() {
            let hot = active == Some(Handle::Rotate(i));
            let (u, v) = mm_core::geom::orthonormal_basis(*a);
            let pts: Vec<egui::Pos2> = (0..=64)
                .filter_map(|k| {
                    let t = k as f64 / 64.0 * std::f64::consts::TAU;
                    cx.project(c3 + (u * t.cos() + v * t.sin()) * len * 0.7)
                })
                .collect();
            let col = if hot { egui::Color32::YELLOW } else { AXIS_COLORS[i].gamma_multiply(0.7) };
            painter.add(egui::Shape::line(pts, egui::Stroke::new(if hot { 3.0 } else { 1.5 }, col)));
        }
        for (i, a) in AXES.iter().enumerate() {
            let hot = active == Some(Handle::Move(i));
            if let Some(tip) = cx.project(c3 + *a * len) {
                let col = if hot { egui::Color32::YELLOW } else { AXIS_COLORS[i] };
                painter.arrow(c, tip - c, egui::Stroke::new(if hot { 4.0 } else { 2.5 }, col));
            }
        }
        let sq = egui::Rect::from_center_size(c, egui::vec2(10.0, 10.0));
        painter.rect_filled(sq, 1.0, if active == Some(Handle::Scale) { egui::Color32::YELLOW } else { egui::Color32::from_gray(230) });
    }

    fn preview(&self) -> Option<(ObjectId, &Mesh)> {
        Some((self.object?, self.preview.as_ref()?))
    }

    fn accept(&mut self, cx: &mut ToolCx) -> bool {
        self.commit(cx)
    }
}

/// Units / Dimensions: inspect size and rescale (e.g. metres → millimetres).
pub struct UnitsTool {
    object: Option<ObjectId>,
    target_axis: usize,
    target_value: f64,
}

impl UnitsTool {
    pub fn new(cx: &mut ToolCx) -> Self {
        let object = cx.scene.active;
        let e = cx.scene.active().map(|o| o.bounds().extents()).unwrap_or(DVec3::ONE);
        Self { object, target_axis: 0, target_value: e.x }
    }

    fn scale_by(&self, cx: &mut ToolCx, s: f64, label: &str) {
        let Some(id) = self.object.filter(|id| cx.scene.get(*id).is_some()) else { return };
        cx.history.push(label, cx.scene);
        let Some(o) = cx.scene.get_mut(id) else { return };
        let c = o.bounds().center();
        o.mesh_mut().transform(|p| c + (p - c) * s);
        *cx.status = format!("{label}: scaled by {s}");
    }
}

impl Tool for UnitsTool {
    fn survives_scene_change(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "Units / Dimensions"
    }

    fn panel(&mut self, ui: &mut egui::Ui, cx: &mut ToolCx) -> ToolOutcome {
        ui_util::section(ui, "Units / Dimensions");
        let Some(o) = self.object.and_then(|id| cx.scene.get(id)) else {
            ui.label("No active object.");
            return if ui.button("Close").clicked() { ToolOutcome::Close } else { ToolOutcome::Continue };
        };
        let b = o.bounds();
        let e = b.extents();
        egui::Grid::new("dims").num_columns(2).show(ui, |ui| {
            for (i, n) in ["Width (X)", "Height (Y)", "Depth (Z)"].iter().enumerate() {
                ui.label(*n);
                ui.label(format!("{:.4}", e[i]));
                ui.end_row();
            }
            ui.label("Min");
            ui.label(format!("{:.3}, {:.3}, {:.3}", b.min.x, b.min.y, b.min.z));
            ui.end_row();
            ui.label("Max");
            ui.label(format!("{:.3}, {:.3}, {:.3}", b.max.x, b.max.y, b.max.z));
            ui.end_row();
        });
        ui.small("Mesh files carry no units; medical exports are usually millimetres.");
        ui_util::section(ui, "Scale to size");
        ui.horizontal(|ui| {
            egui::ComboBox::from_id_salt("axis").selected_text(["X", "Y", "Z"][self.target_axis]).show_ui(ui, |ui| {
                for (i, n) in ["X", "Y", "Z"].iter().enumerate() {
                    ui.selectable_value(&mut self.target_axis, i, *n);
                }
            });
            ui.add(egui::DragValue::new(&mut self.target_value).speed(0.1).max_decimals(4));
        });
        let cur = e[self.target_axis];
        if ui.add_enabled(cur > 0.0 && self.target_value > 0.0, egui::Button::new("Apply uniform scale")).clicked() {
            let s = self.target_value / cur;
            self.scale_by(cx, s, "Scale to size");
        }
        ui_util::section(ui, "Convert units");
        ui.horizontal_wrapped(|ui| {
            for (label, s) in [("m → mm", 1000.0), ("cm → mm", 10.0), ("in → mm", 25.4), ("mm → m", 0.001), ("mm → in", 1.0 / 25.4)] {
                if ui.button(label).clicked() {
                    self.scale_by(cx, s, label);
                }
            }
        });
        ui.add_space(8.0);
        if ui.button("Done").clicked() {
            return ToolOutcome::Close;
        }
        ToolOutcome::Continue
    }
}
