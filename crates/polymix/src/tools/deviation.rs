//! Deviation analysis: per-vertex distance from the active object to a reference object
//! (e.g. the original segmentation), shown as a diverging colour map with summary
//! statistics — useful for documenting how much a cleanup changed the geometry.

use super::*;
use crate::jobs::Job;
use crate::ui_util;
use rayon::prelude::*;
use std::sync::Arc;

#[derive(Clone, Debug, Default)]
pub struct DeviationStats {
    pub mean_abs: f64,
    pub rms: f64,
    pub p95: f64,
    pub max_out: f64,
    pub max_in: f64,
    /// Max distance from the reference back to the object (for the symmetric Hausdorff).
    pub reverse_max: f64,
}

pub struct DeviationResult {
    /// Signed distance per vertex id (positive = outside the reference).
    pub per_vertex: Vec<f64>,
    pub stats: DeviationStats,
}

/// Signed vertex distances from `mesh` to `reference`. The sign comes from the reference's
/// triangle normal at the nearest point (outward-oriented references give + outside).
pub fn compute_deviation(mesh: &Mesh, reference: &Mesh) -> DeviationResult {
    let bvh = mm_core::Bvh::build(reference);
    let ids: Vec<u32> = mesh.vertex_ids().collect();
    let vals: Vec<(u32, f64)> = ids
        .par_iter()
        .map(|&v| {
            let p = mesh.vertex(v);
            let d = bvh
                .nearest(reference, p, f64::INFINITY)
                .map(|h| {
                    let d = h.dist_sq.sqrt();
                    if (p - h.point).dot(reference.tri_normal(h.tid)) >= 0.0 {
                        d
                    } else {
                        -d
                    }
                })
                .unwrap_or(0.0);
            (v, d)
        })
        .collect();
    let mut per_vertex = vec![0.0; mesh.max_vid()];
    for &(v, d) in &vals {
        per_vertex[v as usize] = d;
    }
    let n = vals.len().max(1) as f64;
    let mut abs: Vec<f64> = vals.iter().map(|(_, d)| d.abs()).collect();
    abs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p95 = abs.get(((abs.len() as f64 * 0.95) as usize).min(abs.len().saturating_sub(1))).copied().unwrap_or(0.0);
    // Reverse direction: reference vertices to the object.
    let obvh = mm_core::Bvh::build(mesh);
    let rids: Vec<u32> = reference.vertex_ids().collect();
    let reverse_max = rids
        .par_iter()
        .map(|&v| obvh.nearest(mesh, reference.vertex(v), f64::INFINITY).map(|h| h.dist_sq.sqrt()).unwrap_or(0.0))
        .reduce(|| 0.0, f64::max);
    let stats = DeviationStats {
        mean_abs: abs.iter().sum::<f64>() / n,
        rms: (vals.iter().map(|(_, d)| d * d).sum::<f64>() / n).sqrt(),
        p95,
        max_out: vals.iter().map(|(_, d)| *d).fold(0.0, f64::max),
        max_in: vals.iter().map(|(_, d)| -*d).fold(0.0, f64::max),
        reverse_max,
    };
    DeviationResult { per_vertex, stats }
}

pub struct DeviationTool {
    object: Option<ObjectId>,
    reference: Option<ObjectId>,
    range: f64,
    auto_range: bool,
    job: Option<(MeshKey, Job<DeviationResult>)>,
    result: Option<(MeshKey, Arc<DeviationResult>)>,
}

/// Identity of the (object, reference) mesh pair a result was computed for.
type MeshKey = (usize, u64, usize, u64);

fn mesh_key(cx: &ToolCx, a: ObjectId, b: ObjectId) -> Option<MeshKey> {
    let (ma, mb) = (&cx.scene.get(a)?.mesh, &cx.scene.get(b)?.mesh);
    Some((Arc::as_ptr(ma) as usize, ma.shape_version(), Arc::as_ptr(mb) as usize, mb.shape_version()))
}

impl DeviationTool {
    pub fn new(cx: &mut ToolCx) -> Self {
        let object = cx.scene.active;
        let reference = cx.scene.objects.iter().find(|o| Some(o.id) != object).map(|o| o.id);
        Self { object, reference, range: 1.0, auto_range: true, job: None, result: None }
    }

    fn start(&mut self, cx: &mut ToolCx) {
        let (Some(a), Some(b)) = (self.object, self.reference) else { return };
        let Some(key) = mesh_key(cx, a, b) else { return };
        let (Some(ma), Some(mb)) = (cx.scene.get(a).map(|o| o.mesh.clone()), cx.scene.get(b).map(|o| o.mesh.clone())) else { return };
        self.job = Some((key, Job::spawn("Deviation", cx.egui.clone(), move |_| Ok(compute_deviation(&ma, &mb)))));
    }

    /// Push per-triangle colours (mean of vertex deviations mapped to [0, 1]).
    fn apply_colors(&mut self, cx: &mut ToolCx) {
        let (Some(id), Some(b)) = (self.object, self.reference) else { return };
        let Some((key, r)) = &self.result else { return };
        if mesh_key(cx, id, b) != Some(*key) {
            return;
        }
        let Some(o) = cx.scene.get_mut(id) else { return };
        let m = &o.mesh;
        let range = self.range.max(1e-12);
        let mut vals = vec![0.5f32; m.max_tid()];
        for t in m.triangle_ids() {
            let [a, b, c] = m.triangle(t);
            let at = |v: u32| r.per_vertex.get(v as usize).copied().unwrap_or(0.0);
            let d = (at(a) + at(b) + at(c)) / 3.0;
            vals[t as usize] = (0.5 + 0.5 * (d / range)).clamp(0.0, 1.0) as f32;
        }
        o.scalars = Some((crate::scene::next_stamp(), Arc::new(vals)));
    }

    fn clear(&self, cx: &mut ToolCx) {
        if let Some(o) = self.object.and_then(|id| cx.scene.get_mut(id)) {
            o.scalars = None;
        }
    }
}

impl Tool for DeviationTool {
    fn name(&self) -> &'static str {
        "Deviation"
    }

    fn panel(&mut self, ui: &mut egui::Ui, cx: &mut ToolCx) -> ToolOutcome {
        ui_util::section(ui, "Deviation analysis");
        let Some(a) = self.object.filter(|id| cx.scene.get(*id).is_some()) else {
            ui.label("No active object.");
            return if ui.button("Close").clicked() { ToolOutcome::Close } else { ToolOutcome::Continue };
        };
        ui.small("Distance from the active object to a reference (e.g. a duplicate of the original segmentation made before cleanup). Red = outside the reference, blue = inside.");
        let names: Vec<(ObjectId, String)> = cx.scene.objects.iter().filter(|o| o.id != a).map(|o| (o.id, o.name.clone())).collect();
        if names.is_empty() {
            ui.label("Need a second object as reference.");
            if ui.button("Duplicate active as reference").clicked() {
                let mut c = crate::commands::CmdCx { scene: cx.scene, history: cx.history, status: cx.status };
                crate::commands::duplicate_object(&mut c);
                cx.scene.active = Some(a);
            }
            return if ui.button("Close").clicked() {
                self.clear(cx);
                ToolOutcome::Close
            } else {
                ToolOutcome::Continue
            };
        }
        let before = self.reference;
        egui::ComboBox::from_label("reference")
            .selected_text(self.reference.and_then(|id| names.iter().find(|n| n.0 == id)).map(|n| n.1.clone()).unwrap_or_default())
            .show_ui(ui, |ui| {
                for (id, n) in &names {
                    ui.selectable_value(&mut self.reference, Some(*id), n);
                }
            });
        let current = self.reference.and_then(|b| mesh_key(cx, a, b));
        let have = self.job.as_ref().map(|(k, _)| *k).or(self.result.as_ref().map(|(k, _)| *k));
        if self.reference != before || (current.is_some() && have != current) {
            self.clear(cx);
            self.result = None;
            self.start(cx);
        }
        if let Some((key, j)) = &self.job {
            let key = *key;
            if let Some(r) = j.poll() {
                self.job = None;
                if let Ok(r) = r {
                    if self.auto_range {
                        self.range = r.stats.p95.max(r.stats.mean_abs * 2.0).max(1e-9);
                    }
                    self.result = Some((key, Arc::new(r)));
                    self.apply_colors(cx);
                }
            } else {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label("Computing distances…");
                });
            }
        }
        if let Some((_, r)) = self.result.clone() {
            let s = &r.stats;
            egui::Grid::new("dev").num_columns(2).show(ui, |ui| {
                for (k, v) in [
                    ("Mean |d|", s.mean_abs),
                    ("RMS", s.rms),
                    ("95th percentile", s.p95),
                    ("Max outward", s.max_out),
                    ("Max inward", s.max_in),
                    ("Hausdorff (symmetric)", s.max_out.max(s.max_in).max(s.reverse_max)),
                ] {
                    ui.label(k);
                    ui.label(format!("{v:.4}"));
                    ui.end_row();
                }
            });
            let mut changed = ui.add(egui::Slider::new(&mut self.range, (s.p95 * 0.05).max(1e-9)..=(s.max_out.max(s.max_in) * 1.5).max(1e-6)).logarithmic(true).text("colour range ±")).changed();
            changed |= ui.checkbox(&mut self.auto_range, "Auto range").changed() && self.auto_range;
            if changed {
                if self.auto_range {
                    self.range = s.p95.max(s.mean_abs * 2.0).max(1e-9);
                }
                self.apply_colors(cx);
            }
            if ui.button("Copy statistics").clicked() {
                let txt = format!(
                    "mean_abs={:.6}\nrms={:.6}\np95={:.6}\nmax_outward={:.6}\nmax_inward={:.6}\nhausdorff={:.6}",
                    s.mean_abs,
                    s.rms,
                    s.p95,
                    s.max_out,
                    s.max_in,
                    s.max_out.max(s.max_in).max(s.reverse_max)
                );
                cx.egui.copy_text(txt);
            }
        }
        ui.add_space(8.0);
        if ui.button("Done").clicked() {
            self.clear(cx);
            return ToolOutcome::Close;
        }
        ToolOutcome::Continue
    }

    fn paint(&mut self, painter: &egui::Painter, cx: &mut ToolCx) {
        if self.result.is_none() {
            return;
        }
        // Colour bar legend in the lower right of the viewport.
        let r = cx.vp.rect;
        let bar = egui::Rect::from_min_size(r.right_bottom() + egui::vec2(-60.0, -230.0), egui::vec2(18.0, 180.0));
        let n = 32;
        for i in 0..n {
            let t = 1.0 - i as f32 / n as f32;
            let c = diverging(t);
            let y0 = bar.top() + bar.height() * i as f32 / n as f32;
            painter.rect_filled(egui::Rect::from_min_max(egui::pos2(bar.left(), y0), egui::pos2(bar.right(), y0 + bar.height() / n as f32 + 0.5)), 0.0, c);
        }
        let font = egui::FontId::proportional(12.0);
        let col = egui::Color32::from_gray(220);
        painter.text(bar.right_top() + egui::vec2(4.0, 0.0), egui::Align2::LEFT_CENTER, format!("+{:.3}", self.range), font.clone(), col);
        painter.text(bar.right_center() + egui::vec2(4.0, 0.0), egui::Align2::LEFT_CENTER, "0", font.clone(), col);
        painter.text(bar.right_bottom() + egui::vec2(4.0, 0.0), egui::Align2::LEFT_CENTER, format!("−{:.3}", self.range), font, col);
    }

    fn cancel(&mut self, cx: &mut ToolCx) -> bool {
        self.clear(cx);
        true
    }

    fn on_close(&mut self, cx: &mut ToolCx) {
        self.clear(cx);
    }
}

fn diverging(t: f32) -> egui::Color32 {
    let lerp = |a: [f32; 3], b: [f32; 3], s: f32| [a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s, a[2] + (b[2] - a[2]) * s];
    let (blue, white, red) = ([0.18, 0.36, 0.85], [0.93, 0.93, 0.93], [0.85, 0.2, 0.15]);
    let c = if t < 0.5 { lerp(blue, white, t * 2.0) } else { lerp(white, red, (t - 0.5) * 2.0) };
    egui::Color32::from_rgb((c[0] * 255.0) as u8, (c[1] * 255.0) as u8, (c[2] * 255.0) as u8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_core::primitives;

    #[test]
    fn deviation_of_scaled_sphere() {
        let reference = primitives::icosphere(1.0, 3);
        let mut grown = reference.clone();
        grown.transform(|p| p * 1.05);
        let r = compute_deviation(&grown, &reference);
        assert!((r.stats.mean_abs - 0.05).abs() < 0.005, "{:?}", r.stats);
        assert!(r.stats.max_in < 1e-9, "all outward");
        let mut shrunk = reference.clone();
        shrunk.transform(|p| p * 0.95);
        let r = compute_deviation(&shrunk, &reference);
        assert!(r.stats.max_out < 1e-9 && r.stats.max_in > 0.04);
    }
}
