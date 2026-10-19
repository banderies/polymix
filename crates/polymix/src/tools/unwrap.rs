//! Unwrap (Edit > Unwrap): compute UVs — charts by face group or component, flattened
//! conformally or as rigidly as possible, packed on a sheet (see [`mm_edit::uv`]). The
//! object shows a UV checker while the tool is open; the layout is drawn in the panel.
//! UVs are exported with OBJ and kept in projects.

use super::*;
use crate::jobs::Job;
use crate::scene::UvSet;
use crate::ui_util;
use mm_edit::{Packing, UnwrapElements, UnwrapParams, UnwrapType};
use std::sync::Arc;

/// Computed UVs and a summary.
type Unwrapped = (Arc<UvSet>, String);

pub struct UnwrapTool {
    object: Option<ObjectId>,
    params: UnwrapParams,
    job: Option<(UnwrapParams, Job<Unwrapped>)>,
    /// Computed UVs (shown on the object) and the parameters they were made with.
    result: Option<(UnwrapParams, Arc<UvSet>, String)>,
    /// UVs the object had before, restored on cancel.
    before: Option<Option<Arc<UvSet>>>,
    error: Option<String>,
}

impl UnwrapTool {
    pub fn new(cx: &mut ToolCx) -> Self {
        let mut t = Self { object: cx.scene.active, params: UnwrapParams::default(), job: None, result: None, before: None, error: None };
        t.start(cx);
        t
    }

    fn start(&mut self, cx: &mut ToolCx) {
        let Some(o) = self.object.and_then(|id| cx.scene.get(id)) else { return };
        let mesh = o.mesh.clone();
        let p = self.params;
        self.error = None;
        self.job = Some((
            p,
            Job::spawn("Unwrapping", cx.egui.clone(), move |_| {
                let t0 = std::time::Instant::now();
                let l = mm_edit::unwrap(&mesh, &p)?;
                let corners = l.uv.iter().map(|c| c.map(|q| q.as_vec2().to_array())).collect();
                let msg = format!(
                    "{} charts, {:.0}% of true size, area distortion {:.1}% ({:.1}s)",
                    l.charts,
                    l.scale * 100.0,
                    l.area_distortion * 100.0,
                    t0.elapsed().as_secs_f32()
                );
                Ok((Arc::new(UvSet::new(&mesh, corners)), msg))
            }),
        ));
    }

    /// Show `uvs` on the object (remembering what it had).
    fn show(&mut self, cx: &mut ToolCx, uvs: Arc<UvSet>) {
        let Some(o) = self.object.and_then(|id| cx.scene.get_mut(id)) else { return };
        if self.before.is_none() {
            self.before = Some(o.uvs.clone());
        }
        o.uvs = Some(uvs);
    }

    fn restore(&mut self, cx: &mut ToolCx) {
        if let (Some(before), Some(o)) = (self.before.take(), self.object.and_then(|id| cx.scene.get_mut(id))) {
            o.uvs = before;
        }
    }

    fn poll(&mut self) -> Option<(UnwrapParams, Arc<UvSet>, String)> {
        let (p, job) = self.job.as_ref()?;
        let r = job.poll()?;
        let p = *p;
        self.job = None;
        match r {
            Ok((uvs, msg)) => Some((p, uvs, msg)),
            Err(e) => {
                self.error = Some(format!("{e:#}"));
                None
            }
        }
    }
}

/// Draw the UV layout (triangle edges) into a sheet-shaped box.
fn draw_layout(ui: &mut egui::Ui, uvs: &UvSet, mesh: &Mesh, aspect: f32) {
    let w = ui.available_width().min(260.0);
    let (rect, _) = ui.allocate_exact_size(egui::vec2(w, w * aspect), egui::Sense::hover());
    let p = ui.painter_at(rect);
    p.rect_filled(rect, 2.0, ui.visuals().extreme_bg_color);
    let n = mesh.triangle_count();
    let step = n.div_ceil(12_000).max(1);
    let stroke = egui::Stroke::new(0.5, ui.visuals().text_color().gamma_multiply(0.6));
    let to = |q: [f32; 2]| egui::pos2(rect.min.x + q[0] * rect.width(), rect.max.y - q[1] * rect.height());
    for t in mesh.triangle_ids().step_by(step) {
        if let Some(c) = uvs.corners.get(t as usize) {
            p.add(egui::Shape::closed_line(c.iter().map(|q| to(*q)).collect(), stroke));
        }
    }
    p.rect_stroke(rect, 2.0, egui::Stroke::new(1.0, ui.visuals().widgets.inactive.fg_stroke.color), egui::StrokeKind::Inside);
}

impl Tool for UnwrapTool {
    fn name(&self) -> &'static str {
        "Unwrap"
    }

    fn settings(&self) -> Option<serde_json::Value> {
        serde_json::to_value(self.params).ok()
    }

    fn settings_schema(&self) -> Option<serde_json::Value> {
        Some(schema_of::<UnwrapParams>())
    }

    fn set_settings(&mut self, changes: &serde_json::Value, cx: &mut ToolCx) -> Result<(), String> {
        self.params = merged(&self.params, changes).map_err(|e| format!("Unwrap: {e}"))?;
        self.start(cx);
        Ok(())
    }

    fn state(&self, _scene: &Scene) -> serde_json::Value {
        serde_json::json!({
            "object": self.object,
            "computing": self.job.is_some(),
            "result": self.result.as_ref().map(|(p, _, msg)| serde_json::json!({"message": msg, "for_current_settings": *p == self.params})),
            "error": self.error,
            "next": "accept (mesh_tool accept) to keep the UVs (written with OBJ export)",
        })
    }

    fn shows_uvs(&self) -> bool {
        true
    }

    fn busy(&self) -> Option<(String, f32)> {
        self.job.as_ref().map(|(_, j)| ("Unwrapping".to_string(), j.progress()))
    }

    fn panel(&mut self, ui: &mut egui::Ui, cx: &mut ToolCx) -> ToolOutcome {
        if let Some((p, uvs, msg)) = self.poll() {
            self.show(cx, uvs.clone());
            self.result = Some((p, uvs, msg));
        }
        ui_util::section(ui, "Unwrap");
        let Some(o) = self.object.and_then(|id| cx.scene.get(id)) else {
            ui.label("No active object.");
            return if ui.button("Close").clicked() { ToolOutcome::Close } else { ToolOutcome::Continue };
        };
        let mesh = o.mesh.clone();
        let p = &mut self.params;
        egui::ComboBox::from_label("Unwrap Elements")
            .selected_text(if p.elements == UnwrapElements::Groups { "By Groups" } else { "By Components" })
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut p.elements, UnwrapElements::Groups, "By Groups").on_hover_text("One chart per face group (cut further where needed)");
                ui.selectable_value(&mut p.elements, UnwrapElements::Components, "By Components");
            });
        egui::ComboBox::from_label("Unwrap Type")
            .selected_text(if p.kind == UnwrapType::Rigid { "Rigid" } else { "Conformal" })
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut p.kind, UnwrapType::Rigid, "Rigid").on_hover_text("Keeps lengths and areas (as rigid as possible)");
                ui.selectable_value(&mut p.kind, UnwrapType::Conformal, "Conformal").on_hover_text("Keeps angles");
            });
        egui::ComboBox::from_label("Packing Type")
            .selected_text(match p.packing {
                Packing::Stacked => "Stacked",
                Packing::Line => "Line",
                Packing::Optimized => "Optimized",
            })
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut p.packing, Packing::Optimized, "Optimized");
                ui.selectable_value(&mut p.packing, Packing::Line, "Line");
                ui.selectable_value(&mut p.packing, Packing::Stacked, "Stacked");
            });
        ui.label("Pack Settings");
        egui::Grid::new("pack").num_columns(2).show(ui, |ui| {
            for (label, v) in [("Width", &mut p.width), ("Height", &mut p.height), ("Border", &mut p.border)] {
                ui.label(label);
                ui.add(egui::DragValue::new(v).speed(1.0).range(0.0..=10_000.0).suffix(" mm"));
                ui.end_row();
            }
        });
        let stale = self.result.as_ref().is_none_or(|r| r.0 != self.params) && self.job.as_ref().is_none_or(|j| j.0 != self.params);
        if ui.add_enabled(stale, egui::Button::new("Update")).clicked() {
            self.start(cx);
        }
        if let Some((_, j)) = &self.job {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.add(egui::ProgressBar::new(j.progress()));
            });
            cx.egui.request_repaint_after(std::time::Duration::from_millis(50));
        }
        if let Some((_, uvs, msg)) = &self.result {
            ui.label(msg);
            draw_layout(ui, uvs, &mesh, (self.params.height / self.params.width.max(1e-9)) as f32);
        }
        if let Some(e) = &self.error {
            ui.colored_label(ui.visuals().warn_fg_color, e);
        }
        ui.small("UVs are kept while the shape changes (sculpting, transforms) and dropped by edits that change triangles. Export them with OBJ.");
        let (accept, cancel) = ui_util::accept_cancel(ui, self.result.is_some());
        if accept && self.accept(cx) {
            return ToolOutcome::Close;
        }
        if cancel {
            self.cancel(cx);
            return ToolOutcome::Close;
        }
        ToolOutcome::Continue
    }

    fn accept(&mut self, cx: &mut ToolCx) -> bool {
        if self.job.is_some() {
            return false;
        }
        let Some((_, uvs, msg)) = self.result.clone() else { return true };
        // Snapshot with the UVs from before, then apply.
        self.restore(cx);
        cx.history.push("Unwrap", cx.scene);
        if let Some(o) = self.object.and_then(|id| cx.scene.get_mut(id)) {
            o.uvs = Some(uvs);
        }
        *cx.status = format!("Unwrap: {msg}");
        true
    }

    fn cancel(&mut self, cx: &mut ToolCx) -> bool {
        if let Some((_, j)) = &self.job {
            j.cancel();
        }
        self.restore(cx);
        true
    }
}
