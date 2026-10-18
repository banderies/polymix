//! Stamp (Meshmixer's Stamp tool, hotkey 4): pick a stencil, then drag on the model from
//! where the stamp should be centred; the drag sets its size. On release the outline is cut
//! into the surface and the inside becomes a new face group (ready for Extrude, Offset…) or
//! is removed, leaving a hole of that shape.

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
    stencil: Stencil,
    mode: StampMode,
    /// Stencil rotation in the view, degrees.
    rotation: f64,
    /// Whole-unit stamp sizes (Meshmixer "Snap Dimension").
    snap: bool,
    drag: Option<StampDrag>,
}

impl StampTool {
    pub fn new(_cx: &mut ToolCx) -> Self {
        Self { stencil: Stencil::Circle, mode: StampMode::FaceGroup, rotation: 0.0, snap: false, drag: None }
    }

    /// The stencil outline in the drag's plane coordinates.
    fn outline(&self, radius: f64) -> Vec<DVec2> {
        let (s, c) = self.rotation.to_radians().sin_cos();
        self.stencil.outline().iter().map(|p| DVec2::new(p.x * c - p.y * s, p.x * s + p.y * c) * radius).collect()
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
                cx.history.push(format!("Stamp {}", self.stencil.label()), cx.scene);
                if let Some(o) = cx.scene.get_mut(d.object) {
                    o.set_mesh(mesh);
                }
                *cx.status = format!("Stamp {}: Ø {:.3}, {what} ({} triangles)", self.stencil.label(), 2.0 * d.radius, r.inside.len());
            }
            Err(e) => *cx.status = format!("Stamp: {e}"),
        }
    }
}

/// Small drawing of a stencil, for its button.
fn stencil_button(ui: &mut egui::Ui, stencil: Stencil, selected: bool) -> egui::Response {
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(44.0, 44.0), egui::Sense::click());
    let v = ui.visuals();
    let (bg, fg) = if selected { (v.selection.bg_fill, v.selection.stroke.color) } else if resp.hovered() { (v.widgets.hovered.bg_fill, v.text_color()) } else { (v.widgets.inactive.bg_fill, v.text_color()) };
    let p = ui.painter();
    p.rect_filled(rect, 4.0, bg);
    let r = rect.width() * 0.36;
    let pts: Vec<egui::Pos2> = stencil.outline().iter().map(|q| rect.center() + egui::vec2(q.x as f32, -q.y as f32) * r).collect();
    p.add(egui::Shape::closed_line(pts, egui::Stroke::new(1.6, fg)));
    resp.on_hover_text(stencil.label())
}

impl Tool for StampTool {
    fn name(&self) -> &'static str {
        "Stamp"
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
            for s in Stencil::ALL {
                if stencil_button(ui, s, self.stencil == s).clicked() {
                    self.stencil = s;
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
            Some(("stencil", name)) => self.stencil = Stencil::ALL.into_iter().find(|s| s.label().eq_ignore_ascii_case(name)).unwrap_or(self.stencil),
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
        let mut t = StampTool { stencil: Stencil::Square, mode: StampMode::FaceGroup, rotation: 45.0, snap: false, drag: None };
        // A square rotated by 45° has its corners on the axes.
        let p = t.outline(2.0);
        assert!(p.iter().any(|q| (q.x - 2.0).abs() < 1e-9 && q.y.abs() < 1e-9));
        assert_eq!(t.snapped(1.3), 1.3);
        t.snap = true;
        assert_eq!(t.snapped(1.3), 1.5);
        assert_eq!(t.snapped(0.1), 0.5);
    }
}
