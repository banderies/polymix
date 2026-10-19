//! Clearance (Analysis > Clearance): how close the active object comes to another one.
//! Both surfaces are coloured by distance (red below the minimum clearance), and the
//! closest points are joined by a line.

use super::*;
use crate::ui_util;
use std::sync::Arc;

pub struct ClearanceTool {
    object: Option<ObjectId>,
    other: Option<ObjectId>,
    min_clearance: f64,
    both_ways: bool,
    result: Option<mm_core::measure::Clearance>,
    /// Objects coloured by the tool (cleared on close).
    coloured: Vec<ObjectId>,
    dirty: bool,
}

/// Red below `limit`, then yellow → green → blue as the distance grows to 4 × `limit`.
fn shade(d: f64, limit: f64) -> u32 {
    let pack = |c: [f64; 3]| {
        let b = |x: f64| (x.clamp(0.0, 1.0) * 255.0).round() as u32;
        (b(c[0]) << 16) | (b(c[1]) << 8) | b(c[2])
    };
    if d < limit {
        return pack([0.9, 0.15, 0.12]);
    }
    let t = ((d - limit) / (3.0 * limit.max(1e-300))).clamp(0.0, 1.0);
    let stops = [[0.95, 0.85, 0.2], [0.3, 0.8, 0.35], [0.25, 0.45, 0.9]];
    let x = t * 2.0;
    let i = (x.floor() as usize).min(1);
    let f = x - i as f64;
    pack([0, 1, 2].map(|k| stops[i][k] + (stops[i + 1][k] - stops[i][k]) * f))
}

impl ClearanceTool {
    pub fn new(cx: &mut ToolCx) -> Self {
        let object = cx.scene.active;
        let other = cx.scene.objects.iter().find(|o| Some(o.id) != object && o.visible).map(|o| o.id);
        let edge = cx.scene.active().map(|o| o.mesh.average_edge_length()).unwrap_or(1.0);
        Self { object, other, min_clearance: edge * 5.0, both_ways: true, result: None, coloured: Vec::new(), dirty: true }
    }

    fn compute(&mut self, cx: &mut ToolCx) {
        self.dirty = false;
        self.clear(cx);
        let (Some(a), Some(b)) = (self.object, self.other) else { return };
        let (Some(oa), Some(ob)) = (cx.scene.get_mut(a).map(|o| (o.mesh.clone(), o.bvh())), cx.scene.get_mut(b).map(|o| (o.mesh.clone(), o.bvh()))) else {
            return;
        };
        let r = mm_core::measure::clearance(&oa.0, &oa.1, &ob.0, &ob.1, self.both_ways);
        let colour = |m: &Mesh, d: &[f64]| -> Vec<u32> {
            let mut out = vec![0u32; m.max_tid()];
            for t in m.triangle_ids() {
                let dt = m.triangle(t).iter().map(|&v| d[v as usize]).fold(f64::INFINITY, f64::min);
                out[t as usize] = shade(dt, self.min_clearance);
            }
            out
        };
        let ca = colour(&oa.0, &r.dist_a);
        let cb = r.dist_b.as_ref().map(|d| colour(&ob.0, d));
        if let Some(o) = cx.scene.get_mut(a) {
            o.colors = Some((crate::scene::next_stamp(), Arc::new(ca)));
            self.coloured.push(a);
        }
        if let (Some(cb), Some(o)) = (cb, cx.scene.get_mut(b)) {
            o.colors = Some((crate::scene::next_stamp(), Arc::new(cb)));
            self.coloured.push(b);
        }
        *cx.status = if r.intersecting { "The objects intersect".into() } else { format!("Minimum distance {:.4}", r.min) };
        self.result = Some(r);
    }

    fn clear(&mut self, cx: &mut ToolCx) {
        for id in self.coloured.drain(..) {
            if let Some(o) = cx.scene.get_mut(id) {
                o.colors = None;
            }
        }
    }
}

impl Tool for ClearanceTool {
    fn pending(&self) -> bool {
        self.dirty && self.other.is_some()
    }

    fn settings(&self) -> Option<serde_json::Value> {
        Some(serde_json::json!({"other": self.other, "min_clearance_mm": self.min_clearance, "both_ways": self.both_ways}))
    }

    fn settings_schema(&self) -> Option<serde_json::Value> {
        Some(serde_json::json!({"type": "object", "properties": {
            "other": {"type": "integer", "description": "Object id of the object to measure against"},
            "min_clearance_mm": {"type": "number", "description": "Closer than this shows red"},
            "both_ways": {"type": "boolean", "description": "Also colour (and measure from) the other object"},
        }}))
    }

    fn set_settings(&mut self, changes: &serde_json::Value, cx: &mut ToolCx) -> Result<(), String> {
        let mut v = self.settings().unwrap_or_default();
        merge_settings(&mut v, changes)?;
        self.other = Some(v["other"].as_u64().filter(|id| cx.scene.get(*id).is_some() && Some(*id) != self.object).ok_or("other must be the id of another object")?);
        self.min_clearance = v["min_clearance_mm"].as_f64().filter(|d| *d >= 0.0).ok_or("min_clearance_mm must be a number ≥ 0")?;
        self.both_ways = v["both_ways"].as_bool().ok_or("both_ways must be true or false")?;
        self.compute(cx);
        Ok(())
    }

    fn state(&self, _scene: &Scene) -> serde_json::Value {
        use polymix_workspace::agent::{round, vec3};
        serde_json::json!({
            "object": self.object,
            "other": self.other,
            "result": self.result.as_ref().map(|r| serde_json::json!({
                "intersecting": r.intersecting,
                "min_distance_mm": round(r.min, 5),
                "closest_points_mm": [vec3(r.closest.0.to_array(), 4), vec3(r.closest.1.to_array(), 4)],
                "vertices_closer_than_min_clearance": r.dist_a.iter().filter(|&&d| d < self.min_clearance).count(),
            })),
        })
    }

    fn actions(&self) -> Vec<Action> {
        vec![Action::plain("compute", "Measure again (after the objects changed)")]
    }

    fn act(&mut self, action: &str, _args: &serde_json::Value, cx: &mut ToolCx) -> Result<String, String> {
        match action {
            "compute" => {
                self.compute(cx);
                Ok(cx.status.clone())
            }
            _ => Err(format!("Clearance has no action {action}")),
        }
    }

    fn name(&self) -> &'static str {
        "Clearance"
    }

    fn panel(&mut self, ui: &mut egui::Ui, cx: &mut ToolCx) -> ToolOutcome {
        ui_util::section(ui, "Clearance");
        let Some(a) = self.object.filter(|id| cx.scene.get(*id).is_some()) else {
            ui.label("No active object.");
            return if ui.button("Close").clicked() { ToolOutcome::Close } else { ToolOutcome::Continue };
        };
        let names: Vec<(ObjectId, String)> = cx.scene.objects.iter().filter(|o| o.id != a).map(|o| (o.id, o.name.clone())).collect();
        if names.is_empty() {
            ui.label("Needs a second object.");
            return if ui.button("Close").clicked() { ToolOutcome::Close } else { ToolOutcome::Continue };
        }
        let before = (self.other, self.min_clearance, self.both_ways);
        egui::ComboBox::from_label("to")
            .selected_text(self.other.and_then(|id| names.iter().find(|n| n.0 == id)).map(|n| n.1.clone()).unwrap_or_default())
            .show_ui(ui, |ui| {
                for (id, n) in &names {
                    ui.selectable_value(&mut self.other, Some(*id), n);
                }
            });
        let max = cx.scene.get(a).map(|o| o.bounds().diagonal() * 0.2).unwrap_or(1.0).max(self.min_clearance);
        ui.add(egui::Slider::new(&mut self.min_clearance, 0.0..=max).text("min clearance")).on_hover_text("Closer than this shows red");
        ui.checkbox(&mut self.both_ways, "Bi-directional").on_hover_text("Also colour the other object");
        if (self.other, self.min_clearance, self.both_ways) != before {
            self.dirty = true;
        }
        if self.dirty {
            self.compute(cx);
        }
        if let Some(r) = &self.result {
            if r.intersecting {
                ui.colored_label(egui::Color32::from_rgb(240, 110, 90), "The objects intersect.");
            } else {
                ui.label(format!("Min distance {:.4}", r.min));
            }
            let below = r.dist_a.iter().filter(|&&d| d < self.min_clearance).count();
            ui.label(format!("{below} vertices closer than the minimum clearance"));
        }
        ui.add_space(8.0);
        if ui.button("Done").clicked() {
            self.clear(cx);
            return ToolOutcome::Close;
        }
        ToolOutcome::Continue
    }

    fn overlay(&mut self, ov: &mut Overlay, _cx: &mut ToolCx) {
        if let Some(r) = self.result.as_ref().filter(|r| !r.intersecting) {
            ov.line(r.closest.0, r.closest.1, Vec4::new(1.0, 0.85, 0.2, 1.0));
        }
    }

    fn paint(&mut self, painter: &egui::Painter, cx: &mut ToolCx) {
        if let Some(r) = self.result.as_ref().filter(|r| !r.intersecting) {
            if let Some(p) = cx.project((r.closest.0 + r.closest.1) * 0.5) {
                painter.text(p, egui::Align2::LEFT_BOTTOM, format!("{:.3}", r.min), egui::FontId::proportional(14.0), egui::Color32::from_rgb(255, 220, 80));
            }
        }
    }

    fn cancel(&mut self, cx: &mut ToolCx) -> bool {
        self.clear(cx);
        true
    }

    fn accept(&mut self, cx: &mut ToolCx) -> bool {
        self.clear(cx);
        true
    }

    fn on_close(&mut self, cx: &mut ToolCx) {
        self.clear(cx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shading_bands() {
        assert_eq!(shade(0.1, 1.0), shade(0.9, 1.0), "everything below the limit is red");
        assert_ne!(shade(1.0, 1.0), shade(4.0, 1.0));
        assert_eq!(shade(10.0, 1.0), shade(4.0, 1.0), "far is blue");
    }
}
