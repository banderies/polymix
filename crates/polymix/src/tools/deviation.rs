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
    pub stats: DeviationStats,
    /// Mean of each triangle's signed vertex distances (positive = outside the reference), per
    /// triangle id: what the colours show, mapped again at each change of their range.
    pub per_triangle: Vec<f64>,
}

/// Signed vertex distances from `mesh` to `reference`. The sign comes from the reference's
/// triangle normal at the nearest point (outward-oriented references give + outside).
///
/// `mesh_bvh` is the object's BVH if one is at hand (`None`, or one for another state of the
/// mesh: built here). Only the distance back from the reference uses it, which is the same
/// from any tree over the mesh: the nearest distance does not depend on which of two equally
/// near triangles is met first. The distances to the reference take their sign from the
/// nearest triangle, and at exact ties (a vertex nearest to a corner of the reference, common
/// for an unchanged region) another tree, such as the scene's refitted one, could pick another
/// triangle with another sign; so the reference's tree is always built here, as it always was.
pub fn compute_deviation(mesh: &Mesh, reference: &Mesh, mesh_bvh: Option<Arc<mm_core::Bvh>>) -> DeviationResult {
    // The two trees build at once (each build is serial).
    let (bvh, obvh) = rayon::join(|| mm_core::Bvh::build(reference), || mesh_bvh.filter(|b| b.is_current(mesh)).unwrap_or_else(|| Arc::new(mm_core::Bvh::build(mesh))));
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
    // Equal magnitudes are equal bits (no −0 after abs), so an unstable sort gives the same
    // order, and the mean below the same sum.
    abs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let p95 = abs.get(((abs.len() as f64 * 0.95) as usize).min(abs.len().saturating_sub(1))).copied().unwrap_or(0.0);
    // Reverse direction: reference vertices to the object.
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
    let per_triangle = (0..mesh.max_tid() as u32)
        .into_par_iter()
        .map(|t| {
            if !mesh.is_triangle(t) {
                return 0.0;
            }
            let [a, b, c] = mesh.triangle(t);
            let at = |v: u32| per_vertex.get(v as usize).copied().unwrap_or(0.0);
            (at(a) + at(b) + at(c)) / 3.0
        })
        .collect();
    DeviationResult { stats, per_triangle }
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
    scene_key(cx.scene, a, b)
}

fn scene_key(scene: &Scene, a: ObjectId, b: ObjectId) -> Option<MeshKey> {
    let (ma, mb) = (&scene.get(a)?.mesh, &scene.get(b)?.mesh);
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
        let bvh = cx.scene.get(a).and_then(|o| o.cached_bvh());
        self.job = Some((key, Job::spawn("Deviation", cx.egui.clone(), move |_| Ok(compute_deviation(&ma, &mb, bvh)))));
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
        // The result is for this mesh (the key matched), so its triangle ids are this mesh's.
        // A plain loop: about 1.5 ms at 1.3M triangles, where a parallel one on the UI thread
        // was at best twice as fast and often slower while other work held the cores.
        let vals: Vec<f32> = (0..m.max_tid() as u32).map(|t| if m.is_triangle(t) { (0.5 + 0.5 * (r.per_triangle[t as usize] / range)).clamp(0.0, 1.0) as f32 } else { 0.5 }).collect();
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

    fn busy(&self) -> Option<(String, f32)> {
        self.job.as_ref().map(|_| ("Deviation".to_string(), 0.5))
    }

    fn pending(&self) -> bool {
        self.job.is_none() && self.result.is_none() && self.reference.is_some()
    }

    fn settings(&self) -> Option<serde_json::Value> {
        Some(serde_json::json!({"reference": self.reference, "colour_range_mm": (!self.auto_range).then_some(self.range)}))
    }

    fn settings_schema(&self) -> Option<serde_json::Value> {
        Some(serde_json::json!({"type": "object", "properties": {
            "reference": {"type": "integer", "description": "Object id of the reference (e.g. a duplicate of the original segmentation)"},
            "colour_range_mm": {"type": ["number", "null"], "description": "Distance shown at full colour (±), mm; null for automatic (the 95th percentile)"},
        }}))
    }

    fn set_settings(&mut self, changes: &serde_json::Value, cx: &mut ToolCx) -> Result<(), String> {
        let mut v = self.settings().unwrap_or_default();
        merge_settings(&mut v, changes)?;
        let reference = v["reference"].as_u64().filter(|id| cx.scene.get(*id).is_some() && Some(*id) != self.object).ok_or("reference must be the id of another object")?;
        match &v["colour_range_mm"] {
            serde_json::Value::Null => self.auto_range = true,
            x => {
                self.range = x.as_f64().filter(|r| *r > 0.0).ok_or("colour_range_mm must be a positive number or null")?;
                self.auto_range = false;
            }
        }
        if Some(reference) != self.reference {
            self.reference = Some(reference);
            self.clear(cx);
            self.result = None;
            self.start(cx);
        } else {
            if let (true, Some((_, r))) = (self.auto_range, &self.result) {
                self.range = r.stats.p95.max(r.stats.mean_abs * 2.0).max(1e-9);
            }
            self.apply_colors(cx);
        }
        Ok(())
    }

    fn state(&self, scene: &Scene) -> serde_json::Value {
        use polymix_workspace::agent::round;
        use serde_json::json;
        let current = self.object.zip(self.reference).and_then(|(a, b)| scene_key(scene, a, b));
        json!({
            "object": self.object,
            "reference": self.reference,
            "computing": self.job.is_some(),
            "statistics_mm": self.result.as_ref().map(|(key, r)| {
                let s = &r.stats;
                json!({
                    "for_current_meshes": Some(*key) == current,
                    "mean_abs": round(s.mean_abs, 5),
                    "rms": round(s.rms, 5),
                    "p95": round(s.p95, 5),
                    "max_outward": round(s.max_out, 5),
                    "max_inward": round(s.max_in, 5),
                    "hausdorff_symmetric": round(s.max_out.max(s.max_in).max(s.reverse_max), 5),
                })
            }),
            "colour_range_mm": round(self.range, 5),
            "sign": "positive = outside the reference, negative = inside",
        })
    }

    fn actions(&self) -> Vec<Action> {
        vec![
            Action::plain("compute", "Compute the distances to the reference again"),
            Action::plain("duplicate_as_reference", "Duplicate the active object to serve as the reference (before changing the original)"),
        ]
    }

    fn act(&mut self, action: &str, _args: &serde_json::Value, cx: &mut ToolCx) -> Result<String, String> {
        match action {
            "compute" => {
                if self.reference.is_none() {
                    return Err("set a reference first (settings reference)".into());
                }
                self.clear(cx);
                self.result = None;
                self.start(cx);
                Ok("Computing; app_wait, then read the statistics".into())
            }
            "duplicate_as_reference" => {
                let a = self.object.filter(|id| cx.scene.get(*id).is_some()).ok_or("no active object")?;
                let mut c = crate::commands::CmdCx { scene: cx.scene, history: cx.history, status: cx.status };
                crate::commands::duplicate_object(&mut c);
                let copy = cx.scene.active.filter(|id| *id != a);
                cx.scene.active = Some(a);
                self.reference = copy;
                self.result = None;
                self.start(cx);
                Ok(format!("Duplicated as reference (object {})", copy.map_or("?".into(), |c| c.to_string())))
            }
            _ => Err(format!("Deviation has no action {action}")),
        }
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
                    polymix_workspace::spinner(ui);
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
        let r = compute_deviation(&grown, &reference, None);
        assert!((r.stats.mean_abs - 0.05).abs() < 0.005, "{:?}", r.stats);
        assert!(r.stats.max_in < 1e-9, "all outward");
        let mut shrunk = reference.clone();
        shrunk.transform(|p| p * 0.95);
        let r = compute_deviation(&shrunk, &reference, None);
        assert!(r.stats.max_out < 1e-9 && r.stats.max_in > 0.04);
    }

    /// The deviation as it was computed before the object's BVH could be passed in and the
    /// colours precomputed per triangle (kept here to check the new path against).
    fn old_deviation(mesh: &Mesh, reference: &Mesh) -> (Vec<f64>, DeviationStats) {
        let bvh = mm_core::Bvh::build(reference);
        let ids: Vec<u32> = mesh.vertex_ids().collect();
        let vals: Vec<(u32, f64)> = ids
            .par_iter()
            .map(|&v| {
                let p = mesh.vertex(v);
                let d = bvh.nearest(reference, p, f64::INFINITY).map(|h| if (p - h.point).dot(reference.tri_normal(h.tid)) >= 0.0 { h.dist_sq.sqrt() } else { -h.dist_sq.sqrt() }).unwrap_or(0.0);
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
        let obvh = mm_core::Bvh::build(mesh);
        let rids: Vec<u32> = reference.vertex_ids().collect();
        let reverse_max = rids.par_iter().map(|&v| obvh.nearest(mesh, reference.vertex(v), f64::INFINITY).map(|h| h.dist_sq.sqrt()).unwrap_or(0.0)).reduce(|| 0.0, f64::max);
        let stats = DeviationStats {
            mean_abs: abs.iter().sum::<f64>() / n,
            rms: (vals.iter().map(|(_, d)| d * d).sum::<f64>() / n).sqrt(),
            p95,
            max_out: vals.iter().map(|(_, d)| *d).fold(0.0, f64::max),
            max_in: vals.iter().map(|(_, d)| -*d).fold(0.0, f64::max),
            reverse_max,
        };
        (per_vertex, stats)
    }

    fn bits(s: &DeviationStats) -> [u64; 6] {
        [s.mean_abs, s.rms, s.p95, s.max_out, s.max_in, s.reverse_max].map(f64::to_bits)
    }

    #[test]
    fn deviation_matches_the_old_computation_bit_for_bit() {
        // A bumpy, partly moved copy of a sphere against the sphere, with a hole so that the
        // triangle ids have gaps: many vertices are unchanged (exact ties at the reference's
        // corners), the others in and out.
        let reference = primitives::icosphere(2.0, 4);
        let mut mesh = primitives::icosphere(2.0, 4);
        let gone: Vec<u32> = mesh.triangle_ids().filter(|&t| mesh.tri_centroid(t).z > 1.8).collect();
        mesh.remove_triangles(gone);
        // The scene's tree as a sculpt leaves it: built before the vertices moved, then refitted.
        let mut scene_bvh = mm_core::Bvh::build(&mesh);
        let moved: Vec<u32> = mesh.vertex_ids().filter(|&v| mesh.vertex(v).x > 0.3).collect();
        for v in moved {
            let p = mesh.vertex(v);
            mesh.set_vertex(v, p * (1.0 + 0.05 * (p.y * 3.0).sin()));
        }
        scene_bvh.refit(&mesh);
        let scene_bvh = Arc::new(scene_bvh);
        let (per_vertex, stats) = old_deviation(&mesh, &reference);
        for r in [compute_deviation(&mesh, &reference, None), compute_deviation(&mesh, &reference, Some(scene_bvh))] {
            assert_eq!(bits(&r.stats), bits(&stats), "{:?} vs {stats:?}", r.stats);
            // Each triangle's mean of its vertex values (a sign flip at one vertex shows here).
            for t in mesh.triangle_ids() {
                let [a, b, c] = mesh.triangle(t);
                let old = (per_vertex[a as usize] + per_vertex[b as usize] + per_vertex[c as usize]) / 3.0;
                assert_eq!(r.per_triangle[t as usize].to_bits(), old.to_bits(), "triangle {t}");
            }
            // The colours as they were made from the vertex values, per range.
            for range in [0.01, 0.07, 1.0] {
                let old: Vec<f32> = {
                    let mut vals = vec![0.5f32; mesh.max_tid()];
                    for t in mesh.triangle_ids() {
                        let [a, b, c] = mesh.triangle(t);
                        let at = |v: u32| per_vertex.get(v as usize).copied().unwrap_or(0.0);
                        let d = (at(a) + at(b) + at(c)) / 3.0;
                        vals[t as usize] = (0.5 + 0.5 * (d / range)).clamp(0.0, 1.0) as f32;
                    }
                    vals
                };
                let new: Vec<f32> = (0..mesh.max_tid() as u32)
                    .map(|t| if mesh.is_triangle(t) { (0.5 + 0.5 * (r.per_triangle[t as usize] / range)).clamp(0.0, 1.0) as f32 } else { 0.5 })
                    .collect();
                assert!(old.iter().map(|x| x.to_bits()).eq(new.iter().map(|x| x.to_bits())), "range {range}");
            }
        }
        assert!(stats.max_in > 0.0 && stats.max_out > 0.0);
    }
}
