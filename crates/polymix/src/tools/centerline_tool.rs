//! Centerline tool: choose the inlet, compute the vessel centerline tree, inspect branch
//! lengths and diameters, and export it (CSV / VTK for ParaView or VMTK).

use super::*;
use crate::centerline::{self, Centerline, CenterlineParams};
use crate::jobs::Job;
use crate::tools::vessel::{find_vessel_ends, EndKind, VesselEnd};
use crate::ui_util;

pub struct CenterlineTool {
    object: Option<ObjectId>,
    ends: Vec<VesselEnd>,
    inlet: Option<usize>,
    /// Inlet chosen by clicking the surface (closed meshes without ends).
    inlet_point: Option<DVec3>,
    step: f64,
    auto_step: bool,
    job: Option<Job<Centerline>>,
    result: Option<Centerline>,
    hovered_branch: Option<usize>,
    message: Option<String>,
}

const BRANCH_COLORS: [[f32; 3]; 8] = [
    [1.0, 0.85, 0.2],
    [0.3, 0.9, 0.5],
    [0.35, 0.7, 1.0],
    [1.0, 0.45, 0.35],
    [0.85, 0.5, 1.0],
    [0.3, 0.95, 0.95],
    [1.0, 0.6, 0.1],
    [0.7, 1.0, 0.3],
];

impl CenterlineTool {
    pub fn new(cx: &mut ToolCx) -> Self {
        let ends = cx.scene.active().map(|o| find_vessel_ends(&o.mesh)).unwrap_or_default();
        // The inlet is usually the widest end.
        let inlet = ends
            .iter()
            .enumerate()
            .filter(|(_, e)| !matches!(e.kind, EndKind::Open) || e.planar)
            .max_by(|a, b| a.1.diameter.partial_cmp(&b.1.diameter).unwrap())
            .map(|(i, _)| i);
        Self {
            object: cx.scene.active,
            ends,
            inlet,
            inlet_point: None,
            step: 0.5,
            auto_step: true,
            job: None,
            result: None,
            hovered_branch: None,
            message: None,
        }
    }

    fn start(&mut self, cx: &mut ToolCx) {
        let Some(o) = self.object.and_then(|id| cx.scene.get(id)) else { return };
        let mesh = o.mesh.clone();
        let (sources, radius): (Vec<u32>, f64) = if let Some(i) = self.inlet {
            let e = &self.ends[i];
            let verts: Vec<u32> = match e.kind {
                EndKind::Cap(g) => {
                    let mut v: Vec<u32> = mesh.triangle_ids().filter(|&t| mesh.tri_group(t) == g).flat_map(|t| mesh.triangle(t)).collect();
                    v.sort_unstable();
                    v.dedup();
                    v
                }
                EndKind::Open => mm_core::topology::boundary_loops(&mesh)
                    .into_iter()
                    .min_by(|a, b| a.centroid(&mesh).distance(e.center).partial_cmp(&b.centroid(&mesh).distance(e.center)).unwrap())
                    .map(|l| l.vertices)
                    .unwrap_or_default(),
            };
            (verts, e.diameter * 0.5)
        } else if let Some(p) = self.inlet_point {
            let bvh = mm_core::Bvh::build(&mesh);
            let Some(h) = bvh.nearest(&mesh, p, f64::INFINITY) else { return };
            let r = mm_edit::estimate_vessel_cross_section(&mesh, Some(&bvh), h.point, None).map(|x| x.radius).unwrap_or(mesh.average_edge_length() * 5.0);
            (mesh.triangle(h.tid).to_vec(), r)
        } else {
            self.message = Some("Pick an inlet first.".into());
            return;
        };
        if sources.is_empty() {
            self.message = Some("Could not find the inlet boundary.".into());
            return;
        }
        let params = CenterlineParams { step: if self.auto_step { 0.0 } else { self.step }, ..Default::default() };
        self.message = None;
        self.job = Some(Job::spawn("Centerline", cx.egui.clone(), move |_| Ok(centerline::compute(&mesh, &sources, radius, &params))));
    }

    fn save(&mut self, csv: bool) {
        let Some(cl) = &self.result else { return };
        let (ext, text) = if csv { ("csv", cl.to_csv()) } else { ("vtk", cl.to_vtk()) };
        if let Some(path) = rfd::FileDialog::new().add_filter(ext, &[ext]).set_file_name(format!("centerline.{ext}")).save_file() {
            self.message = Some(match std::fs::write(&path, text) {
                Ok(()) => format!("Saved {}", path.display()),
                Err(e) => format!("Save failed: {e}"),
            });
        }
    }
}

impl Tool for CenterlineTool {
    fn survives_scene_change(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "Centerline"
    }

    fn panel(&mut self, ui: &mut egui::Ui, cx: &mut ToolCx) -> ToolOutcome {
        ui_util::section(ui, "Centerline");
        if self.object.and_then(|id| cx.scene.get(id)).is_none() {
            ui.label("No active object.");
            return if ui.button("Close").clicked() { ToolOutcome::Close } else { ToolOutcome::Continue };
        }
        ui.small("Click an end marker to set the inlet (or Shift+click the surface near the inlet on a closed model), then Compute.");
        match (self.inlet, self.inlet_point) {
            (Some(i), _) => ui.label(format!("Inlet: end {} (⌀ {:.2})", i + 1, self.ends[i].diameter)),
            (None, Some(_)) => ui.label("Inlet: picked surface point"),
            _ => ui.colored_label(egui::Color32::from_rgb(240, 170, 90), "Inlet: not set"),
        };
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.auto_step, "Auto spacing");
            if !self.auto_step {
                ui.add(egui::DragValue::new(&mut self.step).speed(0.01).range(1e-4..=1e4).prefix("step "));
            }
        });
        if ui.add_enabled(self.job.is_none(), egui::Button::new("Compute centerline")).clicked() {
            self.start(cx);
        }
        if let Some(j) = &self.job {
            if let Some(r) = j.poll() {
                self.job = None;
                match r {
                    Ok(cl) => {
                        *cx.status = format!("Centerline: {} branch(es)", cl.branches.len());
                        self.result = Some(cl);
                    }
                    Err(e) => self.message = Some(format!("{e:#}")),
                }
            } else {
                ui.spinner();
            }
        }
        self.hovered_branch = None;
        if let Some(cl) = &self.result {
            ui_util::section(ui, "Branches");
            egui::Grid::new("branches").striped(true).num_columns(5).show(ui, |ui| {
                ui.strong("#");
                ui.strong("Length");
                ui.strong("⌀ mean");
                ui.strong("⌀ min");
                ui.strong("⌀ max");
                ui.end_row();
                for (i, b) in cl.branches.iter().enumerate() {
                    let c = BRANCH_COLORS[i % BRANCH_COLORS.len()];
                    let r = ui.colored_label(egui::Color32::from_rgb((c[0] * 255.0) as u8, (c[1] * 255.0) as u8, (c[2] * 255.0) as u8), format!("{}", i + 1));
                    let mut hovered = r.hovered();
                    hovered |= ui.label(format!("{:.2}", b.length)).hovered();
                    hovered |= ui.label(format!("{:.2}", 2.0 * b.mean_radius)).hovered();
                    hovered |= ui.label(format!("{:.2}", 2.0 * b.min_radius)).hovered();
                    hovered |= ui.label(format!("{:.2}", 2.0 * b.max_radius)).hovered();
                    if hovered {
                        self.hovered_branch = Some(i);
                    }
                    ui.end_row();
                }
            });
            ui.weak("Diameters are 2 × mean ring radius; hover a row to highlight the branch.");
            ui.horizontal(|ui| {
                if ui.button("Export CSV…").clicked() {
                    self.save(true);
                }
                if ui.button("Export VTK…").on_hover_text("Polyline with Radius point data (ParaView, VMTK)").clicked() {
                    self.save(false);
                }
            });
        }
        if let Some(m) = &self.message {
            ui.label(m);
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
        if input.shift {
            if let Some((_, h)) = cx.pick(pos, self.object) {
                self.inlet_point = Some(h.point);
                self.inlet = None;
            }
            return;
        }
        let hit = self
            .ends
            .iter()
            .enumerate()
            .filter_map(|(i, e)| cx.project(e.center).map(|p| (i, p.distance(pos))))
            .filter(|(_, d)| *d < 14.0)
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        if let Some((i, _)) = hit {
            self.inlet = Some(i);
            self.inlet_point = None;
        }
    }

    fn paint(&mut self, painter: &egui::Painter, cx: &mut ToolCx) {
        for (i, e) in self.ends.iter().enumerate() {
            if let Some(p) = cx.project(e.center) {
                let is_inlet = self.inlet == Some(i);
                painter.circle(p, if is_inlet { 9.0 } else { 6.0 }, if is_inlet { ui_util::ACCENT } else { egui::Color32::from_rgba_unmultiplied(60, 140, 255, 150) }, egui::Stroke::new(1.5, egui::Color32::WHITE));
                if is_inlet {
                    painter.text(p + egui::vec2(12.0, -12.0), egui::Align2::LEFT_BOTTOM, "inlet", egui::FontId::proportional(13.0), egui::Color32::WHITE);
                }
            }
        }
        if let Some(p) = self.inlet_point.and_then(|p| cx.project(p)) {
            painter.circle(p, 8.0, ui_util::ACCENT, egui::Stroke::new(1.5, egui::Color32::WHITE));
        }
        // Centerline drawn on top of the surface (it lies inside the vessel).
        if let Some(cl) = &self.result {
            for (i, b) in cl.branches.iter().enumerate() {
                let c = BRANCH_COLORS[i % BRANCH_COLORS.len()];
                let hot = self.hovered_branch == Some(i);
                let col = if hot { egui::Color32::WHITE } else { egui::Color32::from_rgb((c[0] * 255.0) as u8, (c[1] * 255.0) as u8, (c[2] * 255.0) as u8) };
                let pts: Vec<egui::Pos2> = b.nodes.iter().filter_map(|&n| cx.project(cl.nodes[n].position)).collect();
                painter.add(egui::Shape::line(pts, egui::Stroke::new(if hot { 4.0 } else { 2.5 }, col)));
            }
        }
        // Branch labels at their midpoints.
        if let Some(cl) = &self.result {
            for (i, b) in cl.branches.iter().enumerate() {
                let mid = cl.nodes[b.nodes[b.nodes.len() / 2]].position;
                if let Some(p) = cx.project(mid) {
                    painter.text(p, egui::Align2::LEFT_CENTER, format!(" {}", i + 1), egui::FontId::proportional(13.0), egui::Color32::WHITE);
                }
            }
        }
    }

    fn script(&mut self, cmd: &str, cx: &mut ToolCx) {
        if cmd == "compute" {
            self.start(cx);
        }
    }

    fn busy(&self) -> Option<(String, f32)> {
        self.job.as_ref().map(|_| ("Centerline".to_string(), 0.5))
    }
}
