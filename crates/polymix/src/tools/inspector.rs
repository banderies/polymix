//! Inspector (hotkey I): find holes, non-manifold vertices and small floating parts, show
//! them as markers (blue = hole, red = non-manifold, magenta = small part), and repair them
//! one at a time by clicking a marker or all at once (Auto Repair All / Q).

use super::*;
use crate::jobs::Job;
use crate::ui_util;
use mm_repair::{Defect, FillMode, FillParams, InspectParams, InspectionReport, RepairParams};

pub struct InspectorTool {
    object: Option<ObjectId>,
    report: Option<InspectionReport>,
    analyzed: (u64, u64),
    fill_mode: FillMode,
    small_fraction: f64,
    keep_flat_ends: bool,
    job: Option<Job<(Mesh, String)>>,
    last_message: Option<String>,
}

/// Most common face group (fills inherit it so they don't become separate patches).
pub fn dominant_group(mesh: &Mesh) -> i32 {
    let mut counts = std::collections::HashMap::new();
    for t in mesh.triangle_ids() {
        *counts.entry(mesh.tri_group(t)).or_insert(0usize) += 1;
    }
    counts.into_iter().max_by_key(|(_, n)| *n).map(|(g, _)| g).unwrap_or(0)
}

/// Group of the triangle next to the first edge of a boundary loop.
fn loop_group(mesh: &Mesh, lp: &mm_core::topology::EdgeLoop) -> i32 {
    lp.edges.first().map(|&e| mesh.tri_group(mesh.edge_triangles(e)[0])).unwrap_or(0)
}

/// True if a boundary loop is nearly planar (a cut vessel end rather than a wall hole).
pub fn is_planar_loop(mesh: &Mesh, lp: &mm_core::topology::EdgeLoop) -> bool {
    let pts = lp.positions(mesh);
    if pts.len() < 3 {
        return false;
    }
    let c = lp.centroid(mesh);
    let mut n = DVec3::ZERO;
    for i in 0..pts.len() {
        n += pts[i].cross(pts[(i + 1) % pts.len()]);
    }
    let area = 0.5 * n.length();
    let n = n.normalize_or_zero();
    let diameter = 2.0 * (area / std::f64::consts::PI).sqrt();
    let off = pts.iter().map(|p| (*p - c).dot(n).abs()).fold(0.0, f64::max);
    // Cut ends are planar and enclose a sizeable area relative to their perimeter.
    let perimeter = lp.perimeter(mesh);
    off < 0.15 * diameter && perimeter > 0.0 && area / (perimeter * perimeter) > 0.03
}

impl InspectorTool {
    pub fn new(cx: &mut ToolCx) -> Self {
        Self {
            object: cx.scene.active,
            report: None,
            analyzed: (u64::MAX, u64::MAX),
            fill_mode: FillMode::Smooth,
            small_fraction: 0.01,
            keep_flat_ends: false,
            job: None,
            last_message: None,
        }
    }

    fn params(&self) -> InspectParams {
        InspectParams { small_component_area_fraction: self.small_fraction, ..Default::default() }
    }

    fn analyze(&mut self, cx: &mut ToolCx) {
        let Some(o) = self.object.and_then(|id| cx.scene.get(id)) else { return };
        let key = (std::sync::Arc::as_ptr(&o.mesh) as usize as u64, o.mesh.shape_version());
        if self.analyzed == key && self.report.is_some() {
            return;
        }
        let mut rep = mm_repair::inspect(&o.mesh, &self.params());
        if self.keep_flat_ends {
            rep.holes.retain(|h| !is_planar_loop(&o.mesh, &h.edge_loop));
        }
        self.report = Some(rep);
        self.analyzed = key;
    }

    fn fill_params(&self, mesh: &Mesh, lp: Option<&mm_core::topology::EdgeLoop>) -> FillParams {
        let group = lp.map(|l| loop_group(mesh, l)).unwrap_or_else(|| dominant_group(mesh));
        FillParams { mode: self.fill_mode, group: Some(group), ..Default::default() }
    }

    fn repair_one(&mut self, cx: &mut ToolCx, defect: Defect) {
        let Some(o) = self.object.and_then(|id| cx.scene.get(id)) else { return };
        let id = o.id;
        let mut m = (*o.mesh).clone();
        let lp = if let Defect::Hole(l) = &defect { Some(l.clone()) } else { None };
        let fp = self.fill_params(&m, lp.as_ref());
        match mm_repair::repair_single(&mut m, &defect, &fp) {
            Ok(r) => {
                cx.history.push("Inspector repair", cx.scene);
                if let Some(o) = cx.scene.get_mut(id) {
                    o.set_mesh(m);
                }
                self.last_message = Some(match defect {
                    Defect::Hole(_) => format!("Filled hole with {} triangles", r.fill.new_triangles.len()),
                    Defect::BowtieVertex(_) => "Split non-manifold vertex".into(),
                    Defect::SmallComponent(_) => format!("Removed {} triangles", r.removed_triangles),
                    Defect::DegenerateTriangle(_) => "Fixed degenerate triangle".into(),
                });
                self.report = None;
            }
            Err(e) => self.last_message = Some(format!("Repair failed: {e}")),
        }
    }

    fn auto_repair(&mut self, cx: &mut ToolCx) {
        let Some(o) = self.object.and_then(|id| cx.scene.get(id)) else { return };
        let src = o.mesh.clone();
        let mode = self.fill_mode;
        let frac = self.small_fraction;
        let keep_ends = self.keep_flat_ends;
        self.job = Some(Job::spawn("Auto repair", cx.egui.clone(), move |progress| {
            let mut m = (*src).clone();
            let group = dominant_group(&m);
            let params = RepairParams {
                fill_mode: mode,
                fill_group: Some(group),
                small_component_area_fraction: frac,
                fill_holes: !keep_ends,
                ..Default::default()
            };
            let rep = mm_repair::auto_repair(&mut m, &params, Some(&|f| progress(if keep_ends { f * 0.8 } else { f })));
            let mut filled = rep.holes_filled;
            if keep_ends {
                // Fill only the non-planar holes, leaving flat vessel ends open.
                for _ in 0..10_000 {
                    let holes: Vec<_> = mm_core::topology::boundary_loops(&m).into_iter().filter(|l| !is_planar_loop(&m, l)).collect();
                    let Some(lp) = holes.into_iter().next() else { break };
                    let fp = FillParams { mode, group: Some(loop_group(&m, &lp)), ..Default::default() };
                    if mm_repair::fill_hole(&mut m, &lp, &fp).is_err() {
                        break;
                    }
                    filled += 1;
                }
            }
            let shells = mm_core::topology::connected_components(&m).len();
            let msg = format!(
                "Auto repair: {} hole(s) filled, {} part(s) removed, {} non-manifold vertex(es) split, {} crack vertex(es) welded → {}",
                filled,
                rep.components_removed,
                rep.bowties_split,
                rep.welded_vertices,
                if m.is_closed() && shells == 1 { "closed, single shell".to_string() } else { format!("{} open edges, {} shell(s)", m.boundary_edge_ids().count(), shells) }
            );
            Ok((m, msg))
        }));
    }
}

impl Tool for InspectorTool {
    fn survives_scene_change(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "Inspector"
    }

    fn panel(&mut self, ui: &mut egui::Ui, cx: &mut ToolCx) -> ToolOutcome {
        ui_util::section(ui, "Inspector");
        // Poll first so a job whose object vanished still finishes (and stops being busy).
        if let Some(j) = &self.job {
            if let Some(r) = j.poll() {
                self.job = None;
                let id = self.object.filter(|id| cx.scene.get(*id).is_some());
                match (r, id) {
                    (Ok((m, msg)), Some(id)) => {
                        cx.history.push("Auto repair", cx.scene);
                        if let Some(o) = cx.scene.get_mut(id) {
                            o.set_mesh(m);
                        }
                        *cx.status = msg.clone();
                        self.last_message = Some(msg);
                        self.report = None;
                    }
                    (Err(e), _) => self.last_message = Some(format!("Auto repair failed: {e:#}")),
                    (Ok(_), None) => self.last_message = None,
                }
            } else {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.add(egui::ProgressBar::new(j.progress()).show_percentage());
                });
                return ToolOutcome::Continue;
            }
        }
        if self.object.and_then(|id| cx.scene.get(id)).is_none() {
            ui.label("No active object.");
            return if ui.button("Close").clicked() { ToolOutcome::Close } else { ToolOutcome::Continue };
        }
        self.analyze(cx);
        let Some(r) = &self.report else { return ToolOutcome::Continue };
        let row = |ui: &mut egui::Ui, label: &str, n: usize, color: egui::Color32| {
            ui.horizontal(|ui| {
                ui.colored_label(color, "●");
                ui.label(format!("{label}: {n}"));
            });
        };
        row(ui, "Holes", r.holes.len(), egui::Color32::from_rgb(60, 140, 255));
        row(ui, "Non-manifold vertices", r.bowtie_vertices.len(), egui::Color32::from_rgb(230, 60, 60));
        row(ui, "Small parts", r.small_components.len(), egui::Color32::from_rgb(220, 60, 220));
        ui.weak(format!(
            "{} shell(s) · {} degenerate triangle(s) · {} inconsistently oriented edge(s)",
            r.num_components,
            r.degenerate_triangles.len(),
            r.inconsistent_orientation_edges
        ));
        if r.is_clean() {
            ui.colored_label(egui::Color32::from_rgb(120, 200, 120), "No defects found.");
        }
        ui_util::section(ui, "Options");
        ui.horizontal(|ui| {
            ui.label("Hole fill mode");
            ui.selectable_value(&mut self.fill_mode, FillMode::Minimal, "Minimal");
            ui.selectable_value(&mut self.fill_mode, FillMode::Flat, "Flat");
            ui.selectable_value(&mut self.fill_mode, FillMode::Smooth, "Smooth");
        });
        let mut changed = false;
        ui.horizontal(|ui| {
            ui.label("Small thresh");
            changed |= ui
                .add(egui::Slider::new(&mut self.small_fraction, 0.0001..=0.2).logarithmic(true).custom_formatter(|v, _| format!("{:.2}%", v * 100.0)))
                .changed();
        });
        changed |= ui
            .checkbox(&mut self.keep_flat_ends, "Keep flat open ends (vessel inlets/outlets)")
            .on_hover_text("Planar boundary loops are treated as intentional cut ends and are not filled")
            .changed();
        if changed {
            self.report = None;
        }
        ui.add_space(6.0);
        if ui.add(egui::Button::new(egui::RichText::new("Auto Repair All").strong())).on_hover_text("Q").clicked() {
            self.auto_repair(cx);
        }
        ui.small("Click a marker to repair just that defect.");
        if let Some(m) = &self.last_message {
            ui.label(m);
        }
        ui.add_space(8.0);
        if ui.button("Done").clicked() {
            return ToolOutcome::Close;
        }
        ToolOutcome::Continue
    }

    fn pointer(&mut self, input: &PointerInput, cx: &mut ToolCx) {
        if !input.clicked || self.job.is_some() {
            return;
        }
        let (Some(pos), Some(r), Some(id)) = (input.pos, &self.report, self.object) else { return };
        let Some(o) = cx.scene.get(id) else { return };
        let mesh = o.mesh.clone();
        let mut best: Option<(f32, Defect)> = None;
        let mut consider = |p: DVec3, d: Defect, cx: &ToolCx| {
            if let Some(sp) = cx.project(p) {
                let dist = sp.distance(pos);
                if dist < 12.0 && best.as_ref().map(|(b, _)| dist < *b).unwrap_or(true) {
                    best = Some((dist, d));
                }
            }
        };
        for h in &r.holes {
            consider(h.centroid, Defect::Hole(h.edge_loop.clone()), cx);
        }
        for &v in &r.bowtie_vertices {
            consider(mesh.vertex(v), Defect::BowtieVertex(v), cx);
        }
        for c in &r.small_components {
            consider(c.centroid, Defect::SmallComponent(c.triangles.clone()), cx);
        }
        if let Some((_, d)) = best {
            self.repair_one(cx, d);
        }
    }

    fn paint(&mut self, painter: &egui::Painter, cx: &mut ToolCx) {
        let Some(r) = &self.report else { return };
        let Some(o) = self.object.and_then(|id| cx.scene.get(id)) else { return };
        let mesh = o.mesh.clone();
        let stroke = egui::Stroke::new(1.5, egui::Color32::WHITE);
        for h in &r.holes {
            if let Some(p) = cx.project(h.centroid) {
                painter.circle(p, 7.0, egui::Color32::from_rgb(60, 140, 255), stroke);
            }
        }
        for &v in &r.bowtie_vertices {
            if let Some(p) = cx.project(mesh.vertex(v)) {
                painter.circle(p, 6.0, egui::Color32::from_rgb(230, 60, 60), stroke);
            }
        }
        for c in &r.small_components {
            if let Some(p) = cx.project(c.centroid) {
                painter.circle(p, 6.0, egui::Color32::from_rgb(220, 60, 220), stroke);
            }
        }
    }

    fn keys(&mut self, i: &egui::InputState, cx: &mut ToolCx) -> bool {
        if i.key_pressed(egui::Key::Q) && self.job.is_none() {
            self.auto_repair(cx);
            return true;
        }
        false
    }

    fn script(&mut self, cmd: &str, cx: &mut ToolCx) {
        match cmd {
            "auto" => self.auto_repair(cx),
            "keep_ends" => {
                self.keep_flat_ends = true;
                self.report = None;
            }
            _ => {}
        }
    }

    fn busy(&self) -> Option<(String, f32)> {
        self.job.as_ref().map(|j| ("Repairing".to_string(), j.progress()))
    }
}
