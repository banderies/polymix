//! Face selection tool (Meshmixer "Select", hotkey S): sphere brush, lasso and laser lasso,
//! with the selection-based Edit / Modify / Deform commands in its panel.
//!
//! * Brush: drag on the surface. The brush has a fixed size on the model (zooming does not
//!   change what it covers).
//! * Lasso / Laser Lasso: a drag that starts on the background (or any drag in Lasso mode)
//!   draws a screen curve. A closed curve selects what is inside it; an open curve cuts
//!   across the model and selects the smaller side (see [`super::lasso`]).
//! * Double-click: on a triangle at an open boundary, selects the ring of triangles around
//!   that hole (ready for Erase & Fill); elsewhere, selects the connected part.

use super::lasso::ScreenCurve;
use super::*;
use crate::commands::{self, CmdCx, SelectCmd};
use crate::ui_util;
use mm_core::TriSelection;
use rayon::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectMode {
    Brush,
    Lasso,
}

pub struct SelectTool {
    pub mode: SelectMode,
    /// Brush radius in percent of the model size.
    pub size_pct: f32,
    /// Lasso / laser select through the model (occluded and back-facing faces too).
    pub through: bool,
    stroke: Option<Stroke>,
    /// Screen curve being drawn (lasso or laser).
    curve: Option<Vec<egui::Pos2>>,
    hover_brush: Option<(DVec3, f64)>,
    /// Selection before the latest click, so a double-click can undo its two brush dabs.
    before_click: Option<(ObjectId, TriSelection, f64)>,
    /// Right-drag expand/contract in progress.
    expand: Option<ExpandSession>,
    /// Sub-tool requested from the panel (opened by the app).
    pub request: Option<Box<dyn Tool>>,
}

/// Right-drag grow/shrink (Meshmixer's "Expand Mode: Geodesic Distance"): drag right to grow
/// the selection by a surface distance, left to shrink it.
struct ExpandSession {
    object: ObjectId,
    original: TriSelection,
    /// Per triangle: surface distance from the selection border (max over its vertices).
    dist: Vec<f32>,
    /// World length per screen point at the press position.
    scale: f64,
}

impl ExpandSession {
    fn new(mesh: &Mesh, object: ObjectId, original: TriSelection, scale: f64) -> Self {
        // Border vertices: on edges between selected and unselected triangles.
        let mut sources = Vec::new();
        let mut mark = vec![false; mesh.max_vid()];
        for t in original.iter() {
            for e in mesh.tri_edges(t) {
                let o = mesh.edge_other_triangle(e, t);
                if o != mm_core::INVALID && !original.contains(o) {
                    for v in mesh.edge_vertices(e) {
                        if !std::mem::replace(&mut mark[v as usize], true) {
                            sources.push(v);
                        }
                    }
                }
            }
        }
        let vd = crate::centerline::geodesic_distance(mesh, &sources);
        let mut dist = vec![f32::INFINITY; mesh.max_tid()];
        for t in mesh.triangle_ids() {
            dist[t as usize] = mesh.triangle(t).iter().map(|&v| vd[v as usize]).fold(0.0, f64::max) as f32;
        }
        Self { object, original, dist, scale }
    }

    /// Selection for a signed surface distance `d` (positive grows).
    fn apply(&self, mesh: &Mesh, d: f64, out: &mut TriSelection) {
        *out = self.original.clone();
        let d = d as f32;
        for t in mesh.triangle_ids() {
            let within = self.dist[t as usize] <= d.abs();
            if d > 0.0 && within {
                out.insert(t);
            } else if d < 0.0 && within {
                out.remove(t);
            }
        }
    }
}

struct Stroke {
    object: ObjectId,
    deselect: bool,
    last: Option<DVec3>,
}

impl Default for SelectTool {
    fn default() -> Self {
        Self {
            mode: SelectMode::Brush,
            size_pct: 2.5,
            through: true,
            stroke: None,
            curve: None,
            hover_brush: None,
            before_click: None,
            expand: None,
            request: None,
        }
    }
}

/// Clicks closer together than this (seconds) may form a double-click.
const DOUBLE_CLICK_WINDOW: f64 = 0.6;

fn to_dvec(p: egui::Pos2) -> DVec2 {
    DVec2::new(p.x as f64, p.y as f64)
}

impl SelectTool {
    fn paint_at(&mut self, cx: &mut ToolCx, pos: egui::Pos2) {
        let Some(stroke) = &mut self.stroke else { return };
        let ray = cx.ray_at(pos);
        let Some((_, hit)) = pick_ray(cx.scene, &ray, Some(stroke.object)) else {
            stroke.last = None;
            return;
        };
        let object = stroke.object;
        let r = brush_radius(cx.scene, object, self.size_pct);
        let Some(stroke) = &mut self.stroke else { return };
        let deselect = stroke.deselect;
        let last = stroke.last.replace(hit.point);
        let Some(o) = cx.scene.get_mut(object) else {
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

    /// Apply a finished lasso / laser curve to the active object.
    fn finish_curve(&mut self, cx: &mut ToolCx, deselect: bool) {
        let Some(pts) = self.curve.take() else { return };
        let pts: Vec<DVec2> = pts.into_iter().map(to_dvec).collect();
        let Some(curve) = ScreenCurve::new(&pts) else { return };
        let Some(id) = cx.scene.active.filter(|id| cx.scene.get(*id).is_some()) else { return };
        let vp = cx.vp;
        let view_proj = cx.camera.view_proj(vp.aspect());
        let eye = cx.camera.eye();
        let ortho = cx.camera.orthographic;
        let fwd = cx.camera.forward();
        let through = self.through;
        cx.snapshot(if curve.closed { "Lasso select" } else { "Laser select" });
        let o = cx.scene.get_mut(id).unwrap();
        let bvh = o.bvh();
        let mesh = o.mesh.clone();
        // Side of the curve for every triangle in front of the camera (by centroid).
        let side: Vec<(u32, bool)> = mesh
            .triangle_ids()
            .collect::<Vec<_>>()
            .par_iter()
            .filter_map(|&t| {
                let clip = view_proj * mesh.tri_centroid(t).extend(1.0);
                if clip.w <= 0.0 {
                    return None;
                }
                let sp = vp.to_screen(clip.truncate() / clip.w);
                Some((t, curve.contains(to_dvec(sp))))
            })
            .collect();
        // Closed: the inside. Open (laser): the side with less surface area.
        let pick_inside = curve.closed || {
            let (a_in, a_out) = side.iter().fold((0.0, 0.0), |(i, o), &(t, inside)| {
                let a = mesh.tri_area(t);
                if inside { (i + a, o) } else { (i, o + a) }
            });
            a_in <= a_out
        };
        let hits: Vec<u32> = side
            .par_iter()
            .filter(|&&(_, inside)| inside == pick_inside)
            .map(|&(t, _)| t)
            .filter(|&t| {
                if through {
                    return true;
                }
                let c = mesh.tri_centroid(t);
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

    /// Double-click: the ring of triangles around the hole(s) at this triangle, or else the
    /// connected part.
    fn double_click(&mut self, cx: &mut ToolCx, pos: egui::Pos2, deselect: bool) {
        let Some((id, hit)) = cx.pick(pos, None) else { return };
        // Drop the brush dabs the two clicks just painted.
        let before = self.before_click.take().filter(|(bid, _, _)| *bid == id).map(|(_, sel, _)| sel);
        cx.snapshot("Select connected");
        cx.scene.active = Some(id);
        let o = cx.scene.get_mut(id).unwrap();
        let mesh = o.mesh.clone();
        if let Some(sel) = before {
            o.selection = sel;
        }
        o.selection.sync(&mesh);
        let ring = hole_ring(&mesh, hit.tid);
        let what = if ring.is_empty() { "connected part" } else { "hole border" };
        let tris = if ring.is_empty() { mm_core::topology::flood_fill_triangles(&mesh, &[hit.tid], |_| true) } else { ring };
        for t in tris {
            if deselect {
                o.selection.remove(t);
            } else {
                o.selection.insert(t);
            }
        }
        o.touch_selection();
        *cx.status = format!("Selected {what}: {} triangles selected", o.selection.len());
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
                let r = cx.scene.active.map(|id| brush_radius(cx.scene, id, self.size_pct));
                brush_size_ui(ui, &mut self.size_pct, r);
            }
            SelectMode::Lasso => {}
        }
        ui.checkbox(&mut self.through, "Lasso selects through the model")
            .on_hover_text("Include faces hidden behind the visible surface (Meshmixer's laser always cuts through). Turn off to select only what you can see.");
        let (n_sel, n_tri) = cx.scene.active().map(|o| (o.selection.len(), o.mesh.triangle_count())).unwrap_or((0, 0));
        ui.label(format!("{} / {} triangles selected", ui_util::format_count(n_sel), ui_util::format_count(n_tri)));
        ui.small(match self.mode {
            SelectMode::Brush => "Drag on the model to paint · Drag from the background to lasso (open curve = laser cut, selects the smaller side) · Shift: deselect · Size: [ ] or the wheel (Shift+wheel in Fusion 360 navigation)",
            SelectMode::Lasso => "Draw a closed loop to select inside, or an open curve across the model to select the smaller side · Shift: deselect",
        });
        ui.small("Double-click: hole border (on an open edge) or connected part · Right-drag: grow / shrink · S: clear");

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
                if ui.button("Smooth").on_hover_text("Deform > Smooth (Cmd+F)").clicked() {
                    self.request = Some(Box::new(ops::SmoothTool::new(cx)));
                }
                if ui.button("Smooth Boundary").on_hover_text("Smooth the border of the selection (B)").clicked() {
                    self.request = Some(Box::new(ops::SmoothBoundaryTool::new(cx)));
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
        let deselect = input.shift || input.command;
        if input.double_clicked {
            self.stroke = None;
            self.curve = None;
            self.double_click(cx, pos, deselect);
            return;
        }
        if input.pressed {
            let now = cx.egui.input(|i| i.time);
            let hit = cx.pick(pos, cx.scene.active).or_else(|| cx.pick(pos, None));
            // Remember the selection before this click unless it may be the second click of
            // a double-click (then keep the state from before the first one).
            let recent = self.before_click.as_ref().is_some_and(|(_, _, t)| now - t < DOUBLE_CLICK_WINDOW);
            if !recent {
                self.before_click = hit.as_ref().and_then(|(id, _)| cx.scene.get(*id).map(|o| (*id, o.selection.clone(), now)));
            }
            match (self.mode, hit) {
                (SelectMode::Brush, Some((id, _))) => {
                    cx.scene.active = Some(id);
                    cx.snapshot("Brush select");
                    self.stroke = Some(Stroke { object: id, deselect, last: None });
                }
                // Lasso mode, or a brush drag starting on the background: draw a curve.
                _ => self.curve = Some(vec![pos]),
            }
        }
        if input.down {
            if self.stroke.is_some() {
                self.paint_at(cx, pos);
            }
            if let Some(c) = &mut self.curve {
                if c.last().map(|l| l.distance(pos) > 2.0).unwrap_or(true) {
                    c.push(pos);
                }
            }
        }
        if input.released {
            if self.stroke.take().is_some() {
                if let Some(o) = cx.scene.active() {
                    *cx.status = format!("{} triangles selected", o.selection.len());
                }
            }
            if self.curve.is_some() {
                self.finish_curve(cx, deselect);
            }
        }
        // Hover cursor (brush mode, not while drawing a curve).
        self.hover_brush = match (self.mode, &self.curve) {
            (SelectMode::Brush, None) => cx.pick(pos, cx.scene.active).map(|(id, h)| (h.point, brush_radius(cx.scene, id, self.size_pct))),
            _ => None,
        };
    }

    fn wheel(&mut self, delta: f32, _cx: &mut ToolCx) -> bool {
        if self.mode != SelectMode::Brush {
            return false;
        }
        self.size_pct = (self.size_pct * (delta * 0.004).exp()).clamp(BRUSH_SIZE_RANGE.0, BRUSH_SIZE_RANGE.1);
        true
    }

    fn claims_secondary_drag(&self, scene: &Scene) -> bool {
        scene.active().is_some_and(|o| !o.selection.is_empty())
    }

    fn secondary_drag(&mut self, phase: DragPhase, pos: egui::Pos2, total: egui::Vec2, cx: &mut ToolCx) {
        match phase {
            DragPhase::Start => {
                let Some(id) = cx.scene.active else { return };
                let at = cx.pick(pos, Some(id)).map(|(_, h)| h.point).unwrap_or(cx.camera.target);
                let scale = cx.world_per_point(at);
                cx.snapshot("Expand selection");
                let o = cx.scene.get_mut(id).unwrap();
                let mesh = o.mesh.clone();
                o.selection.sync(&mesh);
                self.expand = Some(ExpandSession::new(&mesh, id, o.selection.clone(), scale));
            }
            DragPhase::Move | DragPhase::End => {
                let Some(session) = &self.expand else { return };
                let Some(o) = cx.scene.get_mut(session.object) else {
                    self.expand = None;
                    return;
                };
                let mesh = o.mesh.clone();
                let d = total.x as f64 * session.scale;
                session.apply(&mesh, d, &mut o.selection);
                o.touch_selection();
                *cx.status = format!("{} by {:.3} · {} triangles selected", if d >= 0.0 { "Grow" } else { "Shrink" }, d.abs(), o.selection.len());
                if phase == DragPhase::End {
                    self.expand = None;
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
        let Some(curve) = &self.curve else { return };
        if curve.len() < 2 {
            return;
        }
        let stroke = egui::Stroke::new(1.5, ui_util::ACCENT);
        painter.add(egui::Shape::line(curve.clone(), stroke));
        let pts: Vec<DVec2> = curve.iter().copied().map(to_dvec).collect();
        if super::lasso::is_closed(&pts) {
            painter.line_segment([*curve.last().unwrap(), curve[0]], stroke);
        } else if let Some(ext) = super::lasso::extensions(&pts, 4000.0) {
            // The laser continues straight past both ends.
            let faint = egui::Stroke::new(1.0, ui_util::ACCENT.gamma_multiply(0.45));
            for (a, b) in ext {
                painter.line_segment([egui::pos2(a.x as f32, a.y as f32), egui::pos2(b.x as f32, b.y as f32)], faint);
            }
        }
    }

    fn keys(&mut self, i: &egui::InputState, cx: &mut ToolCx) -> bool {
        use egui::Key;
        let cmd = i.modifiers.command;
        let shift = i.modifiers.shift;
        let mut run = |f: &mut dyn FnMut(&mut CmdCx)| {
            let mut c = CmdCx { scene: cx.scene, history: cx.history, status: cx.status };
            f(&mut c);
        };
        if i.key_pressed(Key::A) && cmd {
            run(&mut |x| commands::selection_command(x, SelectCmd::All));
        } else if i.key_pressed(Key::S) && !cmd && !shift {
            // S again while selecting starts over.
            run(&mut |x| commands::selection_command(x, SelectCmd::Clear));
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
            step_brush_size(&mut self.size_pct, false);
        } else if i.key_pressed(Key::CloseBracket) {
            step_brush_size(&mut self.size_pct, true);
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
            self.request = Some(Box::new(ops::SmoothBoundaryTool::new(cx)));
        } else if i.key_pressed(Key::F) && cmd {
            self.request = Some(Box::new(ops::SmoothTool::new(cx)));
        } else if i.key_pressed(Key::B) && cmd {
            self.request = Some(Box::new(ops::BridgeTool::new(cx)));
        } else if i.key_pressed(Key::Y) && !cmd && !shift {
            run(&mut |x| commands::separate_selection(x, false));
        } else {
            return false;
        }
        true
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

/// Triangles touching the open boundary loop(s) that pass through triangle `t` — the ring
/// to erase and refill for a clean hole fill. Empty when `t` is not at an open boundary.
pub fn hole_ring(mesh: &Mesh, t: u32) -> Vec<u32> {
    let at_boundary: Vec<u32> = mesh.triangle(t).into_iter().filter(|&v| mesh.is_boundary_vertex(v)).collect();
    if at_boundary.is_empty() {
        return Vec::new();
    }
    let mut ring = Vec::new();
    for lp in mm_core::topology::boundary_loops(mesh) {
        if lp.vertices.iter().any(|v| at_boundary.contains(v)) {
            for &v in &lp.vertices {
                ring.extend(mesh.vertex_triangles(v));
            }
        }
    }
    ring.sort_unstable();
    ring.dedup();
    ring
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_core::primitives;

    #[test]
    fn hole_ring_surrounds_the_hole() {
        // Open cylinder: double-clicking a triangle on the bottom rim selects that rim's ring.
        let m = primitives::cylinder(1.0, 2.0, 24, 6, false);
        let rim_tri = m.triangle_ids().find(|&t| m.triangle(t).iter().any(|&v| m.vertex(v).z == 0.0)).unwrap();
        let ring = hole_ring(&m, rim_tri);
        // One row of quads (2 triangles each) touches the bottom rim.
        assert_eq!(ring.len(), 48);
        assert!(ring.iter().all(|&t| m.triangle(t).iter().any(|&v| m.vertex(v).z == 0.0)));
        // A triangle away from both rims has no ring.
        let mid = m.triangle_ids().find(|&t| m.triangle(t).iter().all(|&v| m.vertex(v).z > 0.5 && m.vertex(v).z < 1.5)).unwrap();
        assert!(hole_ring(&m, mid).is_empty());
    }

    #[test]
    fn right_drag_grows_and_shrinks_by_surface_distance() {
        let m = primitives::plane_grid(10.0, 10.0, 20, 20);
        // Select the triangles within 2 units of the centre.
        let mut sel = TriSelection::new(&m);
        for t in m.triangle_ids() {
            if m.tri_centroid(t).length() < 2.0 {
                sel.insert(t);
            }
        }
        let n0 = sel.len();
        let session = ExpandSession::new(&m, 0, sel.clone(), 1.0);
        let mut out = TriSelection::new(&m);
        session.apply(&m, 0.0, &mut out);
        assert_eq!(out.len(), n0);
        session.apply(&m, 1.0, &mut out);
        let grown = out.len();
        assert!(grown > n0);
        session.apply(&m, 2.0, &mut out);
        assert!(out.len() > grown, "a longer drag grows further");
        // Growth follows distance: nothing beyond ~2 + 2 units from the centre.
        assert!(out.iter().all(|t| m.tri_centroid(t).length() < 4.6));
        session.apply(&m, -0.8, &mut out);
        assert!(out.len() < n0 && !out.is_empty());
        assert!(out.iter().all(|t| sel.contains(t)), "shrinking only removes");
    }

    #[test]
    fn erase_and_fill_of_the_ring_closes_the_hole() {
        use crate::tools::ops::{EraseFillSpec, OpInfo, OpSpec};
        let m = primitives::cylinder(1.0, 2.0, 24, 6, false);
        let rim_tri = m.triangle_ids().find(|&t| m.triangle(t).iter().any(|&v| m.vertex(v).z == 0.0)).unwrap();
        let ring = hole_ring(&m, rim_tri);
        let info = OpInfo { mesh: &m, selection: Some(&ring), avg_edge: m.average_edge_length(), bbox_diag: m.bounds().diagonal() };
        let out = EraseFillSpec::run(&m, Some(&ring), &EraseFillSpec::defaults(&info), &|_| true).unwrap();
        assert_eq!(mm_core::topology::boundary_loops(&out.mesh).len(), 1, "only the top rim stays open");
        out.mesh.check_validity().unwrap();
    }
}
