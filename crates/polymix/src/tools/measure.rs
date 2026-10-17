//! Measure tool: point-to-point distance and local vessel diameter.

use super::*;
use crate::ui_util;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Distance,
    Diameter,
}

struct Measurement {
    a: DVec3,
    b: DVec3,
    label: String,
    ring: Vec<[DVec3; 2]>,
}

pub struct MeasureTool {
    mode: Mode,
    first: Option<DVec3>,
    items: Vec<Measurement>,
}

impl MeasureTool {
    pub fn new(_cx: &mut ToolCx) -> Self {
        Self { mode: Mode::Diameter, first: None, items: Vec::new() }
    }
}

/// Distance across the surface from `p` along the inward normal (the "diameter" of a
/// tube at that point). Returns the opposite wall point.
pub fn opposite_wall(scene: &mut crate::scene::Scene, id: ObjectId, p: DVec3, tid: u32) -> Option<DVec3> {
    let o = scene.get_mut(id)?;
    let bvh = o.bvh();
    let n = o.mesh.tri_normal(tid);
    let eps = 1e-6 * (1.0 + p.length());
    let ray = Ray::new(p - n * eps, -n);
    bvh.raycast_filtered(&o.mesh, &ray, f64::INFINITY, |t| t != tid).map(|h| h.point)
}

impl Tool for MeasureTool {
    fn survives_scene_change(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "Measure"
    }

    fn panel(&mut self, ui: &mut egui::Ui, _cx: &mut ToolCx) -> ToolOutcome {
        ui_util::section(ui, "Measure");
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.mode, Mode::Diameter, "Diameter");
            ui.selectable_value(&mut self.mode, Mode::Distance, "Distance");
        });
        ui.small(match self.mode {
            Mode::Diameter => "Click a vessel wall: shows the equivalent diameter of the cross-section perpendicular to the vessel (green ring) and the wall-to-wall distance.",
            Mode::Distance => "Click two surface points.",
        });
        ui.separator();
        let mut remove = None;
        for (i, m) in self.items.iter().enumerate() {
            ui.horizontal(|ui| {
                ui.label(&m.label);
                if ui.small_button("x").clicked() {
                    remove = Some(i);
                }
            });
        }
        if let Some(i) = remove {
            self.items.remove(i);
        }
        if !self.items.is_empty() && ui.button("Clear all").clicked() {
            self.items.clear();
        }
        ui.add_space(8.0);
        if ui.button("Done").clicked() {
            return ToolOutcome::Close;
        }
        ToolOutcome::Continue
    }

    fn pointer(&mut self, input: &PointerInput, cx: &mut ToolCx) {
        if !input.clicked {
            return;
        }
        let Some(pos) = input.pos else { return };
        let Some((id, hit)) = cx.pick(pos, None) else { return };
        match self.mode {
            Mode::Diameter => {
                let wall = opposite_wall(cx.scene, id, hit.point, hit.tid);
                let o = cx.scene.get_mut(id).unwrap();
                let bvh = o.bvh();
                let mesh = o.mesh.clone();
                let section = mm_edit::estimate_vessel_cross_section(&mesh, Some(&bvh), hit.point, None).ok();
                let ring = section
                    .as_ref()
                    .and_then(|x| mm_edit::nearest_intersection_curve(&mesh, x.plane, hit.point))
                    .filter(|c| c.closed)
                    .map(|c| c.segments)
                    .unwrap_or_default();
                let eq = section.as_ref().and_then(|x| x.area).map(|a| 2.0 * (a / std::f64::consts::PI).sqrt());
                let wall_d = wall.map(|q| hit.point.distance(q));
                let label = match (eq, wall_d) {
                    (Some(e), Some(w)) => format!("⌀ {e:.3} (wall-to-wall {w:.3})"),
                    (Some(e), None) => format!("⌀ {e:.3}"),
                    (None, Some(w)) => format!("⌀ {w:.3} (wall-to-wall)"),
                    (None, None) => return,
                };
                *cx.status = format!("Diameter: {label}");
                let b = wall.unwrap_or(hit.point);
                self.items.push(Measurement { a: hit.point, b, label, ring });
            }
            Mode::Distance => match self.first.take() {
                None => self.first = Some(hit.point),
                Some(a) => {
                    let d = a.distance(hit.point);
                    self.items.push(Measurement { a, b: hit.point, label: format!("↔ {d:.3}"), ring: Vec::new() });
                    *cx.status = format!("Distance: {d:.4}");
                }
            },
        }
    }

    fn overlay(&mut self, ov: &mut Overlay, _cx: &mut ToolCx) {
        let c = Vec4::new(1.0, 0.85, 0.2, 1.0);
        for m in &self.items {
            ov.line(m.a, m.b, c);
            for s in &m.ring {
                ov.line(s[0], s[1], Vec4::new(0.3, 1.0, 0.6, 1.0));
            }
        }
    }

    fn paint(&mut self, painter: &egui::Painter, cx: &mut ToolCx) {
        for m in &self.items {
            if let Some(p) = cx.project((m.a + m.b) * 0.5) {
                painter.text(p, egui::Align2::LEFT_BOTTOM, &m.label, egui::FontId::proportional(14.0), egui::Color32::from_rgb(255, 220, 80));
            }
            for q in [m.a, m.b] {
                if let Some(p) = cx.project(q) {
                    painter.circle_filled(p, 3.0, egui::Color32::from_rgb(255, 220, 80));
                }
            }
        }
        if let Some(a) = self.first {
            if let Some(p) = cx.project(a) {
                painter.circle_filled(p, 4.0, ui_util::ACCENT);
            }
        }
    }
}
