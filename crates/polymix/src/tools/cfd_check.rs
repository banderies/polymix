//! "Check for CFD": a readiness checklist for exporting a surface to a CFD mesher —
//! watertight, manifold, single shell, outward orientation, no self-intersections,
//! reasonable triangle quality, and named boundary patches.

use super::*;
use crate::jobs::Job;
use crate::ui_util;

#[derive(Clone, Debug, Default)]
pub struct CfdReport {
    pub open_edges: usize,
    pub bowties: usize,
    pub shells: usize,
    pub inconsistent: bool,
    pub volume: f64,
    pub self_intersecting: Vec<u32>,
    pub min_angle_deg: f64,
    pub sliver_fraction: f64,
    pub edge_min: f64,
    pub edge_mean: f64,
    pub edge_max: f64,
    pub groups: usize,
}

pub fn analyze(mesh: &Mesh) -> CfdReport {
    let bvh = mm_core::Bvh::build(mesh);
    let pairs = mm_core::intersect::self_intersections(mesh, &bvh);
    let mut tris: Vec<u32> = pairs.iter().flat_map(|(a, b)| [*a, *b]).collect();
    tris.sort_unstable();
    tris.dedup();
    let mut min_angle = f64::INFINITY;
    let mut slivers = 0usize;
    for t in mesh.triangle_ids() {
        let [a, b, c] = mesh.tri_positions(t);
        let ang = |p: DVec3, q: DVec3, r: DVec3| (q - p).angle_between(r - p).to_degrees();
        let m = ang(a, b, c).min(ang(b, c, a)).min(ang(c, a, b));
        if m.is_finite() {
            min_angle = min_angle.min(m);
        }
        if m.is_nan() || m < 10.0 {
            slivers += 1;
        }
    }
    let lens: Vec<f64> = mesh.edge_ids().map(|e| mesh.edge_length(e)).collect();
    let mut groups: Vec<i32> = mesh.triangle_ids().map(|t| mesh.tri_group(t)).collect();
    groups.sort_unstable();
    groups.dedup();
    CfdReport {
        open_edges: mesh.boundary_edge_ids().count(),
        bowties: mesh.vertex_ids().filter(|&v| mesh.is_bowtie_vertex(v)).count(),
        shells: mm_core::topology::connected_components(mesh).len(),
        inconsistent: !mesh.is_consistently_oriented(),
        volume: mm_core::measure::volume(mesh),
        self_intersecting: tris,
        min_angle_deg: if min_angle.is_finite() { min_angle } else { 0.0 },
        sliver_fraction: slivers as f64 / mesh.triangle_count().max(1) as f64,
        edge_min: lens.iter().copied().fold(f64::INFINITY, f64::min),
        edge_mean: lens.iter().sum::<f64>() / lens.len().max(1) as f64,
        edge_max: lens.iter().copied().fold(0.0, f64::max),
        groups: groups.len(),
    }
}

pub struct CfdCheckTool {
    object: Option<ObjectId>,
    job: Option<Job<CfdReport>>,
    report: Option<CfdReport>,
    analyzed: u64,
    request: Option<Box<dyn Tool>>,
}

impl CfdCheckTool {
    pub fn new(cx: &mut ToolCx) -> Self {
        Self { object: cx.scene.active, job: None, report: None, analyzed: 0, request: None }
    }
}

fn status_row(ui: &mut egui::Ui, ok: Option<bool>, label: &str, detail: String) {
    let (sym, col) = match ok {
        Some(true) => ("✔", egui::Color32::from_rgb(120, 200, 120)),
        Some(false) => ("X", egui::Color32::from_rgb(240, 100, 90)),
        None => ("!", egui::Color32::from_rgb(240, 180, 80)),
    };
    ui.horizontal(|ui| {
        ui.add_sized([14.0, 14.0], egui::Label::new(egui::RichText::new(sym).strong().color(col)));
        ui.label(label);
    });
    ui.indent(label, |ui| {
        ui.add(egui::Label::new(egui::RichText::new(detail).weak()).wrap());
    });
}

impl Tool for CfdCheckTool {
    fn survives_scene_change(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "Check for CFD"
    }

    fn panel(&mut self, ui: &mut egui::Ui, cx: &mut ToolCx) -> ToolOutcome {
        ui_util::section(ui, "Check for CFD");
        let Some(o) = self.object.and_then(|id| cx.scene.get(id)) else {
            ui.label("No active object.");
            return if ui.button("Close").clicked() { ToolOutcome::Close } else { ToolOutcome::Continue };
        };
        let version = o.mesh.shape_version();
        if self.analyzed != version {
            // Never show (or act on) a report for a different mesh.
            self.report = None;
        }
        if self.job.is_none() && (self.report.is_none() || self.analyzed != version) {
            let m = o.mesh.clone();
            self.analyzed = version;
            self.job = Some(Job::spawn("CFD check", cx.egui.clone(), move |_| Ok(analyze(&m))));
        }
        if let Some(j) = &self.job {
            if let Some(r) = j.poll() {
                self.job = None;
                if let Ok(r) = r {
                    self.report = Some(r);
                }
            } else {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label("Checking…");
                });
            }
        }
        let mut open: Option<Box<dyn Tool>> = None;
        if let Some(r) = self.report.clone() {
            status_row(ui, Some(r.open_edges == 0), "Watertight", if r.open_edges == 0 { "no open edges".into() } else { format!("{} open edges", r.open_edges) });
            status_row(ui, Some(r.bowties == 0), "Manifold", format!("{} non-manifold vertices", r.bowties));
            status_row(ui, Some(r.shells == 1), "Single shell", format!("{} shell(s)", r.shells));
            status_row(ui, Some(!r.inconsistent && r.volume > 0.0), "Outward orientation", format!("volume {:.3}", r.volume));
            status_row(
                ui,
                Some(r.self_intersecting.is_empty()),
                "No self-intersections",
                format!("{} intersecting triangles", r.self_intersecting.len()),
            );
            status_row(
                ui,
                if r.min_angle_deg >= 10.0 { Some(true) } else { None },
                "Triangle quality",
                format!("min angle {:.1}°, {:.2}% below 10°", r.min_angle_deg, r.sliver_fraction * 100.0),
            );
            status_row(
                ui,
                if r.groups > 1 { Some(true) } else { None },
                "Boundary patches",
                if r.groups > 1 { format!("{} face groups (wall + caps)", r.groups) } else { "only one patch: name inlets/outlets for boundary conditions".into() },
            );
            ui.weak(format!("Edge length min {:.4} · mean {:.4} · max {:.4}", r.edge_min, r.edge_mean, r.edge_max));

            ui_util::section(ui, "Fix");
            ui.horizontal_wrapped(|ui| {
                if (r.open_edges > 0 || r.bowties > 0 || r.shells > 1) && ui.button("Inspector").clicked() {
                    open = Some(Box::new(crate::tools::inspector::InspectorTool::new(cx)));
                }
                if !r.self_intersecting.is_empty() {
                    if ui.button("Select intersecting").clicked() {
                        cx.history.push("Select intersecting", cx.scene);
                        if let Some(o) = self.object.and_then(|id| cx.scene.get_mut(id)) {
                            let m = o.mesh.clone();
                            o.selection.sync(&m);
                            for &t in r.self_intersecting.iter().filter(|&&t| m.is_triangle(t)) {
                                o.selection.insert(t);
                            }
                            o.touch_selection();
                        }
                    }
                    if ui.button("Make Solid").on_hover_text("Rebuilds a clean, non-intersecting surface").clicked() {
                        open = Some(Box::new(crate::tools::ops::MakeSolidTool::new(cx)));
                    }
                }
                if r.min_angle_deg < 10.0 && ui.button("Remesh").clicked() {
                    open = Some(Box::new(crate::tools::ops::RemeshTool::new(cx)));
                }
                if ui.button("Boundary patches").clicked() {
                    open = Some(Box::new(crate::tools::patches::PatchesTool::new(cx)));
                }
            });
        }
        self.request = open;
        ui.add_space(8.0);
        if ui.button("Done").clicked() {
            return ToolOutcome::Close;
        }
        ToolOutcome::Continue
    }

    fn take_request(&mut self) -> Option<Box<dyn Tool>> {
        self.request.take()
    }
}
