//! Boolean (union / difference / intersection) of the active object with another object.
//! Meshmixer's solution modes: Fast Approximate is volumetric (robust to messy input, the
//! whole result is re-meshed); Precise is an exact mesh boolean that keeps every triangle
//! away from the intersection curve; Max Quality also remeshes a band around the curve.

use super::*;
use crate::jobs::{Job, ProgressFn};
use crate::ui_util;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Op {
    Union,
    Difference,
    Intersection,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Solution {
    FastApproximate,
    Precise,
    MaxQuality,
}

#[derive(Clone, Copy, PartialEq, Debug)]
struct Key {
    other: ObjectId,
    op: Op,
    solution: Solution,
    accuracy: f64,
    auto_reduce: bool,
    target_edge_scale: f64,
    merge_rings: u32,
}

pub struct BooleanTool {
    object: Option<ObjectId>,
    other: Option<ObjectId>,
    op: Op,
    solution: Solution,
    accuracy: f64,
    auto_reduce: bool,
    target_edge_scale: f64,
    merge_rings: u32,
    job: Option<(Key, Job<(Mesh, String)>)>,
    result: Option<(Key, Mesh)>,
    note: String,
    error: Option<String>,
}

impl BooleanTool {
    pub fn new(cx: &mut ToolCx) -> Self {
        let object = cx.scene.active;
        let other = cx.scene.objects.iter().find(|o| Some(o.id) != object && o.visible).map(|o| o.id);
        Self {
            object,
            other,
            op: Op::Union,
            solution: Solution::Precise,
            accuracy: 0.5,
            auto_reduce: true,
            target_edge_scale: 1.0,
            merge_rings: 2,
            job: None,
            result: None,
            note: String::new(),
            error: None,
        }
    }

    fn key(&self) -> Option<Key> {
        Some(Key {
            other: self.other?,
            op: self.op,
            solution: self.solution,
            accuracy: self.accuracy,
            auto_reduce: self.auto_reduce,
            target_edge_scale: self.target_edge_scale,
            merge_rings: self.merge_rings,
        })
    }

    fn start(&mut self, cx: &mut ToolCx) {
        let (Some(a), Some(key)) = (self.object, self.key()) else { return };
        let (Some(ma), Some(mb)) = (cx.scene.get(a).map(|o| o.mesh.clone()), cx.scene.get(key.other).map(|o| o.mesh.clone())) else {
            return;
        };
        if let Some((_, j)) = &self.job {
            j.cancel();
        }
        self.error = None;
        let job = Job::spawn("Boolean", cx.egui.clone(), move |progress| match key.solution {
            Solution::FastApproximate => {
                let bounds = ma.bounds().union(&mb.bounds());
                let cells = (64.0 * 16f64.powf(key.accuracy)).round() as u32;
                let h = mm_solid::cell_size_for_resolution(&bounds, cells);
                let op = match key.op {
                    Op::Union => mm_solid::BoolOp::Union,
                    Op::Difference => mm_solid::BoolOp::Difference,
                    Op::Intersection => mm_solid::BoolOp::Intersection,
                };
                let m = mm_solid::sdf_boolean(&ma, &mb, op, h, Some(&|f| progress(f))).map_err(|e| anyhow::anyhow!("{e}"))?;
                Ok((m, String::new()))
            }
            Solution::Precise | Solution::MaxQuality => precise(&ma, &mb, &key, progress),
        });
        self.job = Some((key, job));
    }
}

/// Triangles within `rings` rings of the seam vertices.
fn seam_band(m: &Mesh, seam: &[u32], rings: u32) -> Vec<u32> {
    let mut band = mm_core::TriSelection::new(m);
    for &v in seam {
        for t in m.vertex_triangles(v) {
            band.insert(t);
        }
    }
    for _ in 1..rings {
        band.expand(m);
    }
    band.iter().collect()
}

/// Exact boolean, then Auto-Reduce (collapse the slivers along the seam) or, for Max
/// Quality, a remesh of the band around the seam. The seam is a group border (the second
/// object's groups are renumbered), so both keep it in place.
fn precise(a: &Mesh, b: &Mesh, key: &Key, progress: ProgressFn) -> anyhow::Result<(Mesh, String)> {
    let op = match key.op {
        Op::Union => mm_edit::BooleanOp::Union,
        Op::Difference => mm_edit::BooleanOp::Difference,
        Op::Intersection => mm_edit::BooleanOp::Intersection,
    };
    let r = mm_edit::mesh_boolean(a, b, op)?;
    progress(0.5);
    let mut m = r.mesh;
    let note = if r.perturbed { "Coplanar faces were nudged apart by a tiny amount.".to_string() } else { String::new() };
    if r.seam_vertices.is_empty() || m.triangle_count() == 0 {
        return Ok((m, note));
    }
    let edge = 0.5 * (a.average_edge_length() + b.average_edge_length());
    let band = seam_band(&m, &r.seam_vertices, key.merge_rings.max(1));
    if key.solution == Solution::MaxQuality {
        let params = mm_remesh::RemeshParams {
            target_edge_length: edge * key.target_edge_scale,
            iterations: 6,
            preserve_group_boundaries: true,
            region: Some(band),
            ..Default::default()
        };
        mm_remesh::remesh(&mut m, &params, Some(&|f| progress(0.5 + 0.5 * f)))?;
    } else if key.auto_reduce {
        mm_remesh::collapse_short_edges(&mut m, 0.2 * edge, Some(&band), None)?;
    }
    Ok((m, note))
}

impl Tool for BooleanTool {
    fn name(&self) -> &'static str {
        "Boolean"
    }

    fn panel(&mut self, ui: &mut egui::Ui, cx: &mut ToolCx) -> ToolOutcome {
        ui_util::section(ui, "Boolean");
        let Some(a) = self.object.filter(|id| cx.scene.get(*id).is_some()) else {
            ui.label("No active object.");
            return if ui.button("Close").clicked() { ToolOutcome::Close } else { ToolOutcome::Continue };
        };
        let names: Vec<(ObjectId, String)> = cx.scene.objects.iter().filter(|o| o.id != a).map(|o| (o.id, o.name.clone())).collect();
        if names.is_empty() {
            ui.label("Import or create a second object to combine with.");
            return if ui.button("Close").clicked() { ToolOutcome::Close } else { ToolOutcome::Continue };
        }
        let before = self.key();
        ui.label(format!("Active: {}", cx.scene.get(a).map(|o| o.name.clone()).unwrap_or_default()));
        egui::ComboBox::from_label("with")
            .selected_text(self.other.and_then(|id| names.iter().find(|n| n.0 == id)).map(|n| n.1.clone()).unwrap_or_default())
            .show_ui(ui, |ui| {
                for (id, n) in &names {
                    ui.selectable_value(&mut self.other, Some(*id), n);
                }
            });
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.op, Op::Union, "Union");
            ui.selectable_value(&mut self.op, Op::Difference, "Difference");
            ui.selectable_value(&mut self.op, Op::Intersection, "Intersection");
        });
        egui::ComboBox::from_label("Solution mode")
            .selected_text(match self.solution {
                Solution::FastApproximate => "Fast Approximate",
                Solution::Precise => "Precise",
                Solution::MaxQuality => "Max Quality",
            })
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut self.solution, Solution::FastApproximate, "Fast Approximate")
                    .on_hover_text("Volumetric: robust to messy or open input; the whole result is re-meshed");
                ui.selectable_value(&mut self.solution, Solution::Precise, "Precise")
                    .on_hover_text("Exact: keeps every triangle away from the intersection curve (needs closed meshes)");
                ui.selectable_value(&mut self.solution, Solution::MaxQuality, "Max Quality")
                    .on_hover_text("Exact, then remeshes a band around the intersection curve");
            });
        match self.solution {
            Solution::FastApproximate => {
                ui.add(egui::Slider::new(&mut self.accuracy, 0.0..=1.0).text("accuracy"));
            }
            Solution::Precise => {
                ui.checkbox(&mut self.auto_reduce, "Auto-Reduce Result").on_hover_text("Collapse the slivers the cut leaves along the seam");
            }
            Solution::MaxQuality => {
                ui.add(egui::Slider::new(&mut self.target_edge_scale, 0.25..=4.0).logarithmic(true).text("target edge scale"));
                ui.add(egui::Slider::new(&mut self.merge_rings, 1..=6).text("merge border rings"))
                    .on_hover_text("Width of the remeshed band around the seam, in triangle rings");
            }
        }
        if self.key() != before || (self.result.is_none() && self.job.is_none() && self.error.is_none()) {
            self.start(cx);
        }
        if let Some((_, j)) = &self.job {
            if let Some(r) = j.poll() {
                let (k, _) = self.job.take().unwrap();
                match r {
                    Ok((m, note)) => {
                        self.result = Some((k, m));
                        self.note = note;
                    }
                    Err(e) => self.error = Some(format!("{e:#}")),
                }
            } else {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.add(egui::ProgressBar::new(j.progress()));
                });
            }
        }
        if let Some((_, m)) = &self.result {
            ui.label(format!("Result: {} triangles", ui_util::format_count(m.triangle_count())));
            if !self.note.is_empty() {
                ui.weak(&self.note);
            }
        }
        if let Some(e) = &self.error {
            ui.colored_label(egui::Color32::from_rgb(240, 110, 90), e);
            if self.solution != Solution::FastApproximate && ui.button("Use Fast Approximate").on_hover_text("Volumetric boolean: works on open or messy meshes").clicked() {
                self.solution = Solution::FastApproximate;
            }
        }
        let ready = self.result.as_ref().map(|(k, _)| Some(*k) == self.key()).unwrap_or(false);
        let (accept, cancel) = ui_util::accept_cancel(ui, ready);
        if accept && self.accept(cx) {
            return ToolOutcome::Close;
        }
        if cancel {
            return ToolOutcome::Close;
        }
        ToolOutcome::Continue
    }

    fn preview(&self) -> Option<(ObjectId, &Mesh)> {
        let (k, m) = self.result.as_ref()?;
        (Some(*k) == self.key()).then_some((self.object?, m))
    }

    fn accept(&mut self, cx: &mut ToolCx) -> bool {
        // Only commit a finished result for the current operation and operands.
        if self.job.is_some() || self.result.as_ref().map(|(k, _)| Some(*k) != self.key()).unwrap_or(true) {
            return false;
        }
        let Some((k, m)) = self.result.take() else { return false };
        let Some(a) = self.object else { return true };
        if cx.scene.get(a).is_none() || cx.scene.get(k.other).is_none() {
            return true;
        }
        cx.history.push("Boolean", cx.scene);
        let Some(o) = cx.scene.get_mut(a) else { return true };
        o.set_mesh(m);
        cx.scene.remove(k.other);
        cx.scene.active = Some(a);
        *cx.status = "Boolean applied".into();
        true
    }

    fn busy(&self) -> Option<(String, f32)> {
        self.job.as_ref().map(|(_, j)| ("Boolean".to_string(), j.progress()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_core::{measure, primitives, DVec3};

    fn min_angle(m: &Mesh) -> f64 {
        m.triangle_ids()
            .map(|t| {
                let p = m.triangle(t).map(|v| m.vertex(v));
                (0..3).map(|i| (p[(i + 1) % 3] - p[i]).angle_between(p[(i + 2) % 3] - p[i])).fold(f64::INFINITY, f64::min)
            })
            .fold(f64::INFINITY, f64::min)
            .to_degrees()
    }

    #[test]
    fn precise_and_max_quality() {
        let a = primitives::icosphere(1.0, 3);
        let mut b = primitives::icosphere(0.7, 3);
        b.transform(|p| p + DVec3::new(1.1, 0.2, 0.1));
        let key = |solution, auto_reduce| Key {
            other: 0,
            op: Op::Union,
            solution,
            accuracy: 0.5,
            auto_reduce,
            target_edge_scale: 1.0,
            merge_rings: 2,
        };
        let run = |k: Key| precise(&a, &b, &k, &|_| true).unwrap().0;
        let raw = run(key(Solution::Precise, false));
        let reduced = run(key(Solution::Precise, true));
        let quality = run(key(Solution::MaxQuality, true));
        for m in [&raw, &reduced, &quality] {
            m.check_validity().unwrap();
            assert!(m.is_closed());
            assert!((measure::volume(m) - measure::volume(&raw)).abs() < 0.01 * measure::volume(&raw));
        }
        assert!(reduced.triangle_count() < raw.triangle_count(), "auto-reduce removes slivers");
        assert!(min_angle(&quality) > min_angle(&raw), "{} > {}", min_angle(&quality), min_angle(&raw));
    }
}
