//! Stamp (Meshmixer's Stamp tool, hotkey 4): pick a stencil, then drag on the model from
//! where the stamp should be centred; the drag sets its size. On release the outline is cut
//! into the surface and the inside becomes a new face group (ready for Extrude, Offset…) or
//! is removed, leaving a hole of that shape. Stencils made with Select > Convert To > Stamp
//! appear after the built-in ones.

use super::*;
use crate::ui_util;
use mm_edit::{StampFrame, Stencil};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StampMode {
    FaceGroup,
    Hole,
}

struct StampDrag {
    object: ObjectId,
    seed: u32,
    frame: StampFrame,
    radius: f64,
}

pub struct StampTool {
    /// Built-in and user stencils (name, outline in the unit circle).
    stencils: Vec<(String, Vec<DVec2>)>,
    choice: usize,
    mode: StampMode,
    /// Stencil rotation in the view, degrees.
    rotation: f64,
    /// Whole-unit stamp sizes (Meshmixer "Snap Dimension").
    snap: bool,
    drag: Option<StampDrag>,
}

impl StampTool {
    pub fn new(_cx: &mut ToolCx) -> Self {
        let mut stencils: Vec<(String, Vec<DVec2>)> = Stencil::ALL.iter().map(|s| (s.label().to_string(), s.outline())).collect();
        stencils.extend(crate::parts::user_stencils());
        Self { stencils, choice: 0, mode: StampMode::FaceGroup, rotation: 0.0, snap: false, drag: None }
    }

    fn name_of(&self) -> &str {
        &self.stencils[self.choice].0
    }

    /// The stencil outline in the drag's plane coordinates.
    fn outline(&self, radius: f64) -> Vec<DVec2> {
        let (s, c) = self.rotation.to_radians().sin_cos();
        self.stencils[self.choice].1.iter().map(|p| DVec2::new(p.x * c - p.y * s, p.x * s + p.y * c) * radius).collect()
    }

    fn snapped(&self, radius: f64) -> f64 {
        if self.snap {
            // Whole-unit diameters.
            ((2.0 * radius).round().max(1.0)) * 0.5
        } else {
            radius
        }
    }

    fn apply(&mut self, d: StampDrag, cx: &mut ToolCx) {
        let Some(o) = cx.scene.get(d.object) else { return };
        let mut mesh = (*o.mesh).clone();
        let poly = self.outline(d.radius);
        match mm_edit::stamp_polygon(&mut mesh, d.seed, &d.frame, &poly) {
            Ok(r) => {
                let what = match self.mode {
                    StampMode::FaceGroup => {
                        let g = mesh.max_group_id() + 1;
                        for &t in &r.inside {
                            mesh.set_tri_group(t, g);
                        }
                        format!("new face group {g}")
                    }
                    StampMode::Hole => {
                        mesh.remove_triangles(r.inside.iter().copied());
                        "hole".to_string()
                    }
                };
                cx.history.push(format!("Stamp {}", self.name_of()), cx.scene);
                if let Some(o) = cx.scene.get_mut(d.object) {
                    o.set_mesh(mesh);
                }
                *cx.status = format!("Stamp {}: Ø {:.3}, {what} ({} triangles)", self.name_of(), 2.0 * d.radius, r.inside.len());
            }
            Err(e) => *cx.status = format!("Stamp: {e}"),
        }
    }
}

/// Small drawing of a stencil, for its button.
fn stencil_button(ui: &mut egui::Ui, name: &str, outline: &[DVec2], selected: bool) -> egui::Response {
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(44.0, 44.0), egui::Sense::click());
    let v = ui.visuals();
    let (bg, fg) = if selected { (v.selection.bg_fill, v.selection.stroke.color) } else if resp.hovered() { (v.widgets.hovered.bg_fill, v.text_color()) } else { (v.widgets.inactive.bg_fill, v.text_color()) };
    let p = ui.painter();
    p.rect_filled(rect, 4.0, bg);
    let r = rect.width() * 0.36;
    let pts: Vec<egui::Pos2> = outline.iter().map(|q| rect.center() + egui::vec2(q.x as f32, -q.y as f32) * r).collect();
    p.add(egui::Shape::closed_line(pts, egui::Stroke::new(1.6, fg)));
    resp.on_hover_text(name)
}

impl Tool for StampTool {
    fn name(&self) -> &'static str {
        "Stamp"
    }

    fn settings(&self) -> Option<serde_json::Value> {
        Some(serde_json::json!({"stencil": self.name_of(), "mode": if self.mode == StampMode::Hole { "hole" } else { "face_group" }, "rotation_deg": self.rotation, "whole_unit_sizes": self.snap}))
    }

    fn settings_schema(&self) -> Option<serde_json::Value> {
        let names: Vec<&str> = self.stencils.iter().map(|s| s.0.as_str()).collect();
        Some(serde_json::json!({"type": "object", "properties": {
            "stencil": {"type": "string", "enum": names, "description": "The outline stamped (built-in shapes, then the user's own)"},
            "mode": {"type": "string", "enum": ["face_group", "hole"], "description": "The inside becomes a new face group, or is removed"},
            "rotation_deg": {"type": "number", "description": "Turn the stencil in its plane, degrees"},
            "whole_unit_sizes": {"type": "boolean", "description": "Round diameters to whole millimetres"},
        }}))
    }

    fn set_settings(&mut self, changes: &serde_json::Value, _cx: &mut ToolCx) -> Result<(), String> {
        let mut v = self.settings().unwrap_or_default();
        merge_settings(&mut v, changes)?;
        let name = v["stencil"].as_str().unwrap_or_default().to_string();
        self.choice = self.stencils.iter().position(|s| s.0 == name).ok_or_else(|| format!("no stencil {name}"))?;
        self.mode = if v["mode"] == "hole" { StampMode::Hole } else { StampMode::FaceGroup };
        self.rotation = v["rotation_deg"].as_f64().ok_or("rotation_deg must be a number")?;
        self.snap = v["whole_unit_sizes"].as_bool().unwrap_or(false);
        Ok(())
    }

    fn actions(&self) -> Vec<Action> {
        use serde_json::json;
        vec![Action::new(
            "stamp_at",
            "Cut the stencil into the active object, centred at the surface point nearest `at` and facing out of the surface there (as pressing there and dragging to the size does)",
            json!({"at": arg::point("Centre, mm"), "diameter_mm": arg::number("The stamp's size"), "up": arg::point("Which way the stencil's top points (default the view's up)")}),
        )
        .required(&["at", "diameter_mm"])]
    }

    fn act(&mut self, action: &str, args: &serde_json::Value, cx: &mut ToolCx) -> Result<String, String> {
        let a = Args(args);
        match action {
            "stamp_at" => {
                let id = cx.scene.active.ok_or("no active object")?;
                let o = cx.scene.get_mut(id).ok_or("no object")?;
                let h = o.bvh().nearest(&o.mesh, DVec3::from_array(a.point("at")?), f64::INFINITY).ok_or("no surface")?;
                let n = o.mesh.triangle(h.tid).iter().map(|&v| o.mesh.vertex_normal(v)).sum::<DVec3>().normalize_or(o.mesh.tri_normal(h.tid));
                let up = a.opt_point("up").map(DVec3::from_array).unwrap_or_else(|| cx.camera.up());
                let d = a.f64("diameter_mm")?;
                if d <= 0.0 {
                    return Err("diameter_mm must be positive".into());
                }
                let drag = StampDrag { object: id, seed: h.tid, frame: StampFrame::new(h.point, n, up), radius: self.snapped(0.5 * d) };
                let before = cx.history.generation();
                self.apply(drag, cx);
                if cx.history.generation() == before { Err(cx.status.clone()) } else { Ok(cx.status.clone()) }
            }
            _ => Err(format!("Stamp has no action {action}")),
        }
    }

    fn shows_groups(&self) -> bool {
        true
    }

    fn survives_scene_change(&self) -> bool {
        true
    }

    fn panel(&mut self, ui: &mut egui::Ui, _cx: &mut ToolCx) -> ToolOutcome {
        ui_util::section(ui, "Stamp");
        ui.horizontal_wrapped(|ui| {
            for i in 0..self.stencils.len() {
                let (name, outline) = &self.stencils[i];
                if stencil_button(ui, name, outline, self.choice == i).clicked() {
                    self.choice = i;
                }
            }
        });
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.mode, StampMode::FaceGroup, "Face group").on_hover_text("The inside becomes a new face group");
            ui.selectable_value(&mut self.mode, StampMode::Hole, "Hole").on_hover_text("The inside is removed");
        });
        ui.add(egui::Slider::new(&mut self.rotation, -180.0..=180.0).text("rotation").suffix("°"));
        ui.checkbox(&mut self.snap, "Snap Dimension").on_hover_text("Whole-unit stamp diameters");
        ui.small("Drag on the model from the centre of the stamp outward to size it. The outline is cut into the surface facing you; a new face group can then be extruded or offset (double-click it in Select).");
        ui.add_space(8.0);
        if ui.button("Done").clicked() {
            return ToolOutcome::Close;
        }
        ToolOutcome::Continue
    }

    fn pointer(&mut self, input: &PointerInput, cx: &mut ToolCx) {
        let Some(pos) = input.pos else { return };
        if input.pressed {
            self.drag = cx.pick(pos, cx.scene.active).map(|(id, h)| {
                let o = cx.scene.get(id).unwrap();
                // Smoothed normal under the click, so faceting does not tilt the stamp.
                let n = o.mesh.triangle(h.tid).iter().map(|&v| o.mesh.vertex_normal(v)).sum::<DVec3>().normalize_or(o.mesh.tri_normal(h.tid));
                StampDrag { object: id, seed: h.tid, frame: StampFrame::new(h.point, n, cx.camera.up()), radius: 0.0 }
            });
        }
        if let Some(d) = &mut self.drag {
            if input.down {
                // Size from where the pointer ray meets the stencil plane.
                let ray = cx.ray_at(pos);
                let denom = ray.dir.dot(d.frame.n);
                if denom.abs() > 1e-9 {
                    let t = (d.frame.origin - ray.origin).dot(d.frame.n) / denom;
                    d.radius = d.frame.project(ray.origin + ray.dir * t).length();
                }
            }
        }
        if input.released {
            if let Some(mut d) = self.drag.take() {
                // A click without a drag does nothing.
                if d.radius > cx.world_per_point(d.frame.origin) * 4.0 {
                    d.radius = self.snapped(d.radius);
                    self.apply(d, cx);
                }
            }
        }
    }

    fn overlay(&mut self, ov: &mut Overlay, cx: &mut ToolCx) {
        let Some(d) = &self.drag else { return };
        if d.radius <= 0.0 {
            return;
        }
        let Some(o) = cx.scene.get_mut(d.object) else { return };
        let bvh = o.bvh();
        let poly = self.outline(self.snapped(d.radius));
        // Outline dropped onto the surface (edges subdivided so it follows the curvature).
        let mut pts = Vec::new();
        for i in 0..poly.len() {
            let (a, b) = (poly[i], poly[(i + 1) % poly.len()]);
            let n = ((b - a).length() / (d.radius * 0.08)).ceil().max(1.0) as usize;
            for k in 0..n {
                let q = d.frame.lift(a.lerp(b, k as f64 / n as f64));
                pts.push(bvh.nearest(&o.mesh, q, d.radius * 2.0).map_or(q, |h| h.point + o.mesh.tri_normal(h.tid) * (d.radius * 0.01)));
            }
        }
        let col = Vec4::new(1.0, 0.75, 0.2, 1.0);
        for i in 0..pts.len() {
            ov.line(pts[i], pts[(i + 1) % pts.len()], col);
        }
    }

    fn script(&mut self, cmd: &str, _cx: &mut ToolCx) {
        // `stencil:Star`, `mode:hole`, `rotate:30`.
        match cmd.split_once(':') {
            Some(("stencil", name)) => self.choice = self.stencils.iter().position(|s| s.0.eq_ignore_ascii_case(name)).unwrap_or(self.choice),
            Some(("mode", m)) => self.mode = if m == "hole" { StampMode::Hole } else { StampMode::FaceGroup },
            Some(("rotate", a)) => self.rotation = a.parse().unwrap_or(self.rotation),
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotation_and_snapping() {
        let stencils = vec![("Square".to_string(), Stencil::Square.outline())];
        let mut t = StampTool { stencils, choice: 0, mode: StampMode::FaceGroup, rotation: 45.0, snap: false, drag: None };
        // A square rotated by 45° has its corners on the axes.
        let p = t.outline(2.0);
        assert!(p.iter().any(|q| (q.x - 2.0).abs() < 1e-9 && q.y.abs() < 1e-9));
        assert_eq!(t.snapped(1.3), 1.3);
        t.snap = true;
        assert_eq!(t.snapped(1.3), 1.5);
        assert_eq!(t.snapped(0.1), 0.5);
    }
}
