//! Face selection tool (Meshmixer "Select", hotkey S): sphere brush and lasso, with the
//! selection-based Edit / Modify / Deform commands in its panel.

use super::*;
use crate::commands::{self, CmdCx, SelectCmd};
use crate::ui_util;
use rayon::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectMode {
    Brush,
    Lasso,
}

pub struct SelectTool {
    pub mode: SelectMode,
    /// Brush radius in screen points.
    pub brush_px: f32,
    /// Lasso selects through the model (back-facing / occluded triangles too).
    pub through: bool,
    stroke: Option<Stroke>,
    armed: bool,
    lasso: Vec<egui::Pos2>,
    hover_brush: Option<(DVec3, f64)>,
    /// Sub-tool requested from the panel (opened by the app).
    pub request: Option<Box<dyn Tool>>,
}

struct Stroke {
    object: ObjectId,
    deselect: bool,
    last: Option<DVec3>,
}

impl Default for SelectTool {
    fn default() -> Self {
        Self { mode: SelectMode::Brush, brush_px: 28.0, through: false, stroke: None, armed: false, lasso: Vec::new(), hover_brush: None, request: None }
    }
}

fn point_in_polygon(p: egui::Pos2, poly: &[egui::Pos2]) -> bool {
    let mut inside = false;
    let n = poly.len();
    let mut j = n - 1;
    for i in 0..n {
        let (a, b) = (poly[i], poly[j]);
        if (a.y > p.y) != (b.y > p.y) && p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x {
            inside = !inside;
        }
        j = i;
    }
    inside
}

impl SelectTool {
    fn paint_at(&mut self, cx: &mut ToolCx, pos: egui::Pos2) {
        let Some(stroke) = &mut self.stroke else { return };
        let ray = cx.ray_at(pos);
        let Some((_, hit)) = pick_ray(cx.scene, &ray, Some(stroke.object)) else {
            stroke.last = None;
            return;
        };
        let r = cx.world_per_point(hit.point) * self.brush_px as f64;
        let deselect = stroke.deselect;
        let last = stroke.last.replace(hit.point);
        let Some(o) = cx.scene.get_mut(stroke.object) else {
            self.stroke = None;
            return;
        };
        let bvh = o.bvh();
        let mesh = o.mesh.clone();
        o.selection.sync(&mesh);
        // Stamp along the segment from the previous sample so fast strokes stay continuous.
        let mut stamps = vec![(hit.tid, hit.point)];
        if let Some(prev) = last {
            let d = prev.distance(hit.point);
            let steps = (d / (r * 0.5)).floor() as usize;
            for k in 1..steps.min(64) {
                let p = prev.lerp(hit.point, k as f64 / steps as f64);
                if let Some(n) = bvh.nearest(&mesh, p, r) {
                    stamps.push((n.tid, n.point));
                }
            }
        }
        let mut changed = 0;
        for (tid, p) in stamps {
            changed += o.selection.paint_sphere(&mesh, &bvh, tid, p, r, deselect);
        }
        if changed > 0 {
            o.touch_selection();
        }
    }

    fn finish_lasso(&mut self, cx: &mut ToolCx, deselect: bool) {
        let poly = std::mem::take(&mut self.lasso);
        if poly.len() < 3 {
            return;
        }
        let Some(id) = cx.scene.active else { return };
        let vp = cx.vp;
        let aspect = vp.aspect();
        let view_proj = cx.camera.view_proj(aspect);
        let eye = cx.camera.eye();
        let ortho = cx.camera.orthographic;
        let fwd = cx.camera.forward();
        let through = self.through;
        cx.snapshot("Lasso select");
        let o = cx.scene.get_mut(id).unwrap();
        let bvh = o.bvh();
        let mesh = o.mesh.clone();
        let min = poly.iter().fold(egui::pos2(f32::MAX, f32::MAX), |a, p| a.min(*p));
        let max = poly.iter().fold(egui::pos2(f32::MIN, f32::MIN), |a, p| a.max(*p));
        let tris: Vec<u32> = mesh.triangle_ids().collect();
        let hits: Vec<u32> = tris
            .par_iter()
            .copied()
            .filter(|&t| {
                let c = mesh.tri_centroid(t);
                let clip = view_proj * c.extend(1.0);
                if clip.w <= 0.0 {
                    return false;
                }
                let ndc = clip.truncate() / clip.w;
                let sp = vp.to_screen(ndc);
                if sp.x < min.x || sp.y < min.y || sp.x > max.x || sp.y > max.y || !point_in_polygon(sp, &poly) {
                    return false;
                }
                if through {
                    return true;
                }
                let to_eye = if ortho { -fwd } else { (eye - c).normalize() };
                if mesh.tri_normal(t).dot(to_eye) <= 0.0 {
                    return false;
                }
                // Occlusion: cast from the centroid toward the eye.
                let origin = c + to_eye * 1e-6 * (1.0 + c.length());
                let ray = Ray::new(origin, to_eye);
                let max_t = if ortho { f64::INFINITY } else { eye.distance(c) };
                bvh.raycast_filtered(&mesh, &ray, max_t, |x| x != t).is_none()
            })
            .collect();
        o.selection.sync(&mesh);
        for t in hits {
            if deselect {
                o.selection.remove(t);
            } else {
                o.selection.insert(t);
            }
        }
        o.touch_selection();
        *cx.status = format!("{} triangles selected", o.selection.len());
    }

    fn cmd(&mut self, cx: &mut ToolCx, f: impl FnOnce(&mut CmdCx)) {
        let mut c = CmdCx { scene: cx.scene, history: cx.history, status: cx.status };
        f(&mut c);
    }
}

impl Tool for SelectTool {
    fn survives_scene_change(&self) -> bool {
        true
    }

    fn script(&mut self, cmd: &str, _cx: &mut ToolCx) {
        match cmd {
            "lasso" => self.mode = SelectMode::Lasso,
            "brush" => self.mode = SelectMode::Brush,
            _ => {}
        }
    }

    fn name(&self) -> &'static str {
        "Select"
    }

    fn panel(&mut self, ui: &mut egui::Ui, cx: &mut ToolCx) -> ToolOutcome {
        ui_util::section(ui, "Select");
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.mode, SelectMode::Brush, "Brush");
            ui.selectable_value(&mut self.mode, SelectMode::Lasso, "Lasso");
        });
        match self.mode {
            SelectMode::Brush => {
                ui.horizontal(|ui| {
                    ui.label("Size");
                    ui.add(egui::Slider::new(&mut self.brush_px, 3.0..=250.0).logarithmic(true).suffix(" px"));
                });
            }
            SelectMode::Lasso => {
                ui.checkbox(&mut self.through, "Select through (include hidden faces)");
            }
        }
        let (n_sel, n_tri) = cx.scene.active().map(|o| (o.selection.len(), o.mesh.triangle_count())).unwrap_or((0, 0));
        ui.label(format!("{} / {} triangles selected", ui_util::format_count(n_sel), ui_util::format_count(n_tri)));
        ui.small("Drag to select · Shift+drag to deselect · Double-click: connected");

        ui_util::section(ui, "Selection");
        ui.horizontal_wrapped(|ui| {
            let mut sc = |ui: &mut egui::Ui, label: &str, key: &str, c: SelectCmd, this: &mut Self| {
                if ui.button(label).on_hover_text(key).clicked() {
                    this.cmd(cx, |x| commands::selection_command(x, c));
                }
            };
            sc(ui, "All", "Ctrl/Cmd+A", SelectCmd::All, self);
            sc(ui, "Clear", "Esc", SelectCmd::Clear, self);
            sc(ui, "Invert", "I", SelectCmd::Invert, self);
            sc(ui, "Grow", ">", SelectCmd::Expand, self);
            sc(ui, "Shrink", "<", SelectCmd::Contract, self);
            sc(ui, "Connected", "E", SelectCmd::Connected, self);
            sc(ui, "Invert connected", "Shift+I", SelectCmd::InvertConnected, self);
            sc(ui, "To groups", "G", SelectCmd::ExpandToGroups, self);
            sc(ui, "Optimize boundary", "O", SelectCmd::OptimizeBoundary, self);
        });

        let has_sel = n_sel > 0;
        ui_util::section(ui, "Edit");
        ui.add_enabled_ui(has_sel, |ui| {
            ui.horizontal_wrapped(|ui| {
                if ui.button("Discard").on_hover_text("X").clicked() {
                    self.cmd(cx, commands::discard_selection);
                }
                if ui.button("Erase & Fill").on_hover_text("F").clicked() {
                    self.request = Some(Box::new(ops::EraseFillTool::new(cx)));
                }
                if ui.button("Extrude").on_hover_text("D").clicked() {
                    self.request = Some(Box::new(ops::ExtrudeTool::new(cx, false)));
                }
                if ui.button("Offset").on_hover_text("Ctrl/Cmd+D").clicked() {
                    self.request = Some(Box::new(ops::ExtrudeTool::new(cx, true)));
                }
                if ui.button("Separate").clicked() {
                    self.cmd(cx, |x| commands::separate_selection(x, false));
                }
                if ui.button("Duplicate").clicked() {
                    self.cmd(cx, |x| commands::separate_selection(x, true));
                }
                if ui.button("Flip normals").clicked() {
                    self.cmd(cx, commands::flip_selection_normals);
                }
                if ui.button("Create face group").on_hover_text("Ctrl/Cmd+G").clicked() {
                    self.cmd(cx, |x| commands::create_face_group(x, false));
                }
                if ui.button("Clear face group").clicked() {
                    self.cmd(cx, |x| commands::create_face_group(x, true));
                }
            });
        });
        ui_util::section(ui, "Modify");
        ui.add_enabled_ui(has_sel, |ui| {
            ui.horizontal_wrapped(|ui| {
                if ui.button("Smooth").on_hover_text("Deform > Smooth").clicked() {
                    self.request = Some(Box::new(ops::SmoothTool::new(cx)));
                }
                if ui.button("Remesh").on_hover_text("R").clicked() {
                    self.request = Some(Box::new(ops::RemeshTool::new(cx)));
                }
                if ui.button("Reduce").on_hover_text("Shift+R").clicked() {
                    self.request = Some(Box::new(ops::ReduceTool::new(cx)));
                }
            });
        });
        ui.add_space(10.0);
        if ui.button("Done").clicked() {
            return ToolOutcome::Close;
        }
        ToolOutcome::Continue
    }

    fn pointer(&mut self, input: &PointerInput, cx: &mut ToolCx) {
        let Some(pos) = input.pos else { return };
        if input.double_clicked {
            if let Some((id, hit)) = cx.pick(pos, None) {
                cx.snapshot("Select connected");
                cx.scene.active = Some(id);
                let o = cx.scene.get_mut(id).unwrap();
                let mesh = o.mesh.clone();
                o.selection.sync(&mesh);
                for t in mm_core::topology::flood_fill_triangles(&mesh, &[hit.tid], |_| true) {
                    if input.shift {
                        o.selection.remove(t);
                    } else {
                        o.selection.insert(t);
                    }
                }
                o.touch_selection();
            }
            self.stroke = None;
            return;
        }
        match self.mode {
            SelectMode::Brush => {
                if input.pressed {
                    self.armed = true;
                }
                // A stroke begins as soon as the dragged cursor reaches a surface.
                if input.down && self.armed && self.stroke.is_none() {
                    let target = cx.pick(pos, cx.scene.active).or_else(|| cx.pick(pos, None));
                    if let Some((id, _)) = target {
                        cx.scene.active = Some(id);
                        cx.snapshot("Brush select");
                        self.stroke = Some(Stroke { object: id, deselect: input.shift || input.command, last: None });
                    }
                }
                if input.down && self.stroke.is_some() {
                    self.paint_at(cx, pos);
                }
                if input.released {
                    self.armed = false;
                    self.stroke = None;
                    if let Some(o) = cx.scene.active() {
                        *cx.status = format!("{} triangles selected", o.selection.len());
                    }
                }
                // Hover cursor.
                self.hover_brush = cx
                    .pick(pos, cx.scene.active)
                    .map(|(_, h)| (h.point, cx.world_per_point(h.point) * self.brush_px as f64));
            }
            SelectMode::Lasso => {
                self.hover_brush = None;
                if input.pressed {
                    self.lasso.clear();
                }
                if input.down && self.lasso.last().map(|l| l.distance(pos) > 2.0).unwrap_or(true) {
                    self.lasso.push(pos);
                }
                if input.released {
                    let deselect = input.shift || input.command;
                    self.finish_lasso(cx, deselect);
                }
            }
        }
    }

    fn overlay(&mut self, ov: &mut Overlay, _cx: &mut ToolCx) {
        if let Some((c, r)) = self.hover_brush {
            ov.brush = Some((c.as_vec3(), r as f32));
        }
    }

    fn paint(&mut self, painter: &egui::Painter, _cx: &mut ToolCx) {
        if self.lasso.len() > 1 {
            let mut pts = self.lasso.clone();
            pts.push(pts[0]);
            painter.add(egui::Shape::line(pts, egui::Stroke::new(1.5, ui_util::ACCENT)));
        }
    }

    fn keys(&mut self, i: &egui::InputState, cx: &mut ToolCx) {
        use egui::Key;
        let cmd = i.modifiers.command;
        let shift = i.modifiers.shift;
        let mut run = |f: &mut dyn FnMut(&mut CmdCx)| {
            let mut c = CmdCx { scene: cx.scene, history: cx.history, status: cx.status };
            f(&mut c);
        };
        if i.key_pressed(Key::A) && cmd {
            run(&mut |x| commands::selection_command(x, SelectCmd::All));
        } else if i.key_pressed(Key::I) && !cmd {
            let c = if shift { SelectCmd::InvertConnected } else { SelectCmd::Invert };
            run(&mut |x| commands::selection_command(x, c));
        } else if i.key_pressed(Key::E) && !cmd {
            run(&mut |x| commands::selection_command(x, SelectCmd::Connected));
        } else if i.key_pressed(Key::G) && cmd {
            run(&mut |x| commands::create_face_group(x, shift));
        } else if i.key_pressed(Key::G) {
            run(&mut |x| commands::selection_command(x, SelectCmd::ExpandToGroups));
        } else if i.key_pressed(Key::O) && !cmd {
            run(&mut |x| commands::selection_command(x, SelectCmd::OptimizeBoundary));
        } else if i.key_pressed(Key::Period) || (i.key_pressed(Key::Equals) && !cmd) {
            run(&mut |x| commands::selection_command(x, SelectCmd::Expand));
        } else if i.key_pressed(Key::Comma) || (i.key_pressed(Key::Minus) && !cmd) {
            run(&mut |x| commands::selection_command(x, SelectCmd::Contract));
        } else if i.key_pressed(Key::X) || i.key_pressed(Key::Delete) || i.key_pressed(Key::Backspace) {
            run(&mut commands::discard_selection);
        } else if i.key_pressed(Key::OpenBracket) {
            self.brush_px = (self.brush_px / 1.15).max(3.0);
        } else if i.key_pressed(Key::CloseBracket) {
            self.brush_px = (self.brush_px * 1.15).min(250.0);
        } else if i.key_pressed(Key::F) && !cmd {
            self.request = Some(Box::new(ops::EraseFillTool::new(cx)));
        } else if i.key_pressed(Key::D) && !cmd && !shift {
            self.request = Some(Box::new(ops::ExtrudeTool::new(cx, false)));
        } else if i.key_pressed(Key::D) && cmd {
            self.request = Some(Box::new(ops::ExtrudeTool::new(cx, true)));
        } else if i.key_pressed(Key::R) && !cmd && shift {
            self.request = Some(Box::new(ops::ReduceTool::new(cx)));
        } else if i.key_pressed(Key::R) && !cmd {
            self.request = Some(Box::new(ops::RemeshTool::new(cx)));
        } else if i.key_pressed(Key::B) && !cmd {
            self.request = Some(Box::new(ops::SmoothTool::new(cx)));
        }
    }

    fn cancel(&mut self, cx: &mut ToolCx) -> bool {
        // Esc in the select tool clears the selection first, then exits.
        if cx.scene.active().map(|o| !o.selection.is_empty()).unwrap_or(false) {
            let mut c = CmdCx { scene: cx.scene, history: cx.history, status: cx.status };
            commands::selection_command(&mut c, SelectCmd::Clear);
            return false;
        }
        true
    }

    fn take_request(&mut self) -> Option<Box<dyn Tool>> {
        self.request.take()
    }
}
