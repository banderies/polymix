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

/// Check `mesh`, with its BVH if one is at hand (`None`, or one for another state of the mesh:
/// built here). Any tree over the mesh finds the same intersecting pairs.
pub fn analyze(mesh: &Mesh, bvh: Option<std::sync::Arc<mm_core::Bvh>>) -> CfdReport {
    let bvh = bvh.filter(|b| b.is_current(mesh)).unwrap_or_else(|| std::sync::Arc::new(mm_core::Bvh::build(mesh)));
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

    /// Check the object's current mesh unless that is done or under way (`again`: check anew).
    fn check(&mut self, cx: &mut ToolCx, again: bool) {
        let Some(o) = self.object.and_then(|id| cx.scene.get(id)) else { return };
        let version = o.mesh.shape_version();
        if self.analyzed != version {
            // Never show (or act on) a report for a different mesh.
            self.report = None;
        }
        if self.job.is_none() && (again || self.report.is_none() || self.analyzed != version) {
            let (m, bvh) = (o.mesh.clone(), o.cached_bvh());
            self.analyzed = version;
            self.job = Some(Job::spawn("CFD check", cx.egui.clone(), move |_| Ok(analyze(&m, bvh))));
        }
    }

    fn poll(&mut self) {
        if let Some(r) = self.job.as_ref().and_then(|j| j.poll()) {
            self.job = None;
            if let Ok(r) = r {
                self.report = Some(r);
            }
        }
    }

    fn select_intersecting(&self, cx: &mut ToolCx, tris: &[u32]) {
        cx.history.push("Select intersecting", cx.scene);
        if let Some(o) = self.object.and_then(|id| cx.scene.get_mut(id)) {
            let m = o.mesh.clone();
            o.selection.sync(&m);
            for &t in tris.iter().filter(|&&t| m.is_triangle(t)) {
                o.selection.insert(t);
            }
            o.touch_selection();
        }
    }
}

impl CfdReport {
    /// Each check: whether it passed (`None`: a warning), and the detail.
    fn checks(&self) -> [(&'static str, Option<bool>, String); 7] {
        [
            ("Watertight", Some(self.open_edges == 0), if self.open_edges == 0 { "no open edges".into() } else { format!("{} open edges", self.open_edges) }),
            ("Manifold", Some(self.bowties == 0), format!("{} non-manifold vertices", self.bowties)),
            ("Single shell", Some(self.shells == 1), format!("{} shell(s)", self.shells)),
            ("Outward orientation", Some(!self.inconsistent && self.volume > 0.0), format!("volume {:.3}", self.volume)),
            ("No self-intersections", Some(self.self_intersecting.is_empty()), format!("{} intersecting triangles", self.self_intersecting.len())),
            ("Triangle quality", if self.min_angle_deg >= 10.0 { Some(true) } else { None }, format!("min angle {:.1}°, {:.2}% below 10°", self.min_angle_deg, self.sliver_fraction * 100.0)),
            ("Boundary patches", if self.groups > 1 { Some(true) } else { None }, if self.groups > 1 { format!("{} face groups (wall + caps)", self.groups) } else { "only one patch: name inlets/outlets for boundary conditions".into() }),
        ]
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
        let _ = o;
        self.check(cx, false);
        self.poll();
        if self.job.is_some() {
            ui.horizontal(|ui| {
                polymix_workspace::spinner(ui);
                ui.label("Checking…");
            });
        }
        let mut open: Option<Box<dyn Tool>> = None;
        if let Some(r) = self.report.clone() {
            for (label, ok, detail) in r.checks() {
                status_row(ui, ok, label, detail);
            }
            ui.weak(format!("Edge length min {:.4} · mean {:.4} · max {:.4}", r.edge_min, r.edge_mean, r.edge_max));

            ui_util::section(ui, "Fix");
            ui.horizontal_wrapped(|ui| {
                if (r.open_edges > 0 || r.bowties > 0 || r.shells > 1) && ui.button("Inspector").clicked() {
                    open = Some(Box::new(crate::tools::inspector::InspectorTool::new(cx)));
                }
                if !r.self_intersecting.is_empty() {
                    if ui.button("Select intersecting").clicked() {
                        self.select_intersecting(cx, &r.self_intersecting);
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

    fn busy(&self) -> Option<(String, f32)> {
        self.job.as_ref().map(|j| ("Checking for CFD".to_string(), j.progress()))
    }

    fn pending(&self) -> bool {
        self.job.is_none() && self.report.is_none() && self.object.is_some()
    }

    fn state(&self, scene: &Scene) -> serde_json::Value {
        use polymix_workspace::agent::round;
        use serde_json::json;
        let current = self.object.and_then(|id| scene.get(id)).is_some_and(|o| o.mesh.shape_version() == self.analyzed);
        json!({
            "object": self.object,
            "checking": self.job.is_some(),
            "report": self.report.as_ref().map(|r| {
                let checks = r.checks();
                json!({
                    "for_current_mesh": current,
                    "ready_for_cfd": checks.iter().all(|(_, ok, _)| *ok != Some(false)),
                    "checks": checks.iter().map(|(label, ok, detail)| json!({"check": label, "result": match ok { Some(true) => "pass", Some(false) => "fail", None => "warning" }, "detail": detail})).collect::<Vec<_>>(),
                    "edge_length_mm": {"min": round(r.edge_min, 5), "mean": round(r.edge_mean, 5), "max": round(r.edge_max, 5)},
                })
            }),
            "next": if self.report.is_none() { "wait for the check (app_wait)" } else { "fix what fails (open_fix), then check again" },
        })
    }

    fn actions(&self) -> Vec<Action> {
        use serde_json::json;
        vec![
            Action::plain("check", "Check the object again now (the check also reruns by itself when the mesh changes)"),
            Action::plain("select_intersecting", "Select the self-intersecting triangles the last check found"),
            Action::new("open_fix", "Open the tool that fixes what failed (the check stays open underneath)", json!({"tool": arg::choice(&["inspector", "makesolid", "remesh", "patches"], "inspector for holes, non-manifold vertices and extra shells; makesolid for self-intersections; remesh for poor triangles; patches to name inlets and outlets")})).required(&["tool"]),
        ]
    }

    fn act(&mut self, action: &str, args: &serde_json::Value, cx: &mut ToolCx) -> Result<String, String> {
        self.poll();
        match action {
            "check" => {
                self.check(cx, true);
                Ok("Checking; app_wait, then read the report".into())
            }
            "select_intersecting" => {
                let r = self.report.clone().ok_or("no check yet: app_wait for it")?;
                if r.self_intersecting.is_empty() {
                    return Err("the last check found no self-intersections".into());
                }
                self.select_intersecting(cx, &r.self_intersecting);
                Ok(format!("Selected {} intersecting triangles", r.self_intersecting.len()))
            }
            "open_fix" => {
                let tool: Box<dyn Tool> = match Args(args).str("tool")? {
                    "inspector" => Box::new(crate::tools::inspector::InspectorTool::new(cx)),
                    "makesolid" => Box::new(crate::tools::ops::MakeSolidTool::new(cx)),
                    "remesh" => Box::new(crate::tools::ops::RemeshTool::new(cx)),
                    _ => Box::new(crate::tools::patches::PatchesTool::new(cx)),
                };
                let name = tool.name();
                self.request = Some(tool);
                Ok(format!("{name} opened over the check"))
            }
            _ => Err(format!("Check for CFD has no action {action}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_core::primitives;

    #[test]
    fn a_refitted_scene_bvh_finds_the_same_defects() {
        // Two overlapping spheres in one object (self-intersecting, two shells), one of them
        // moved after the tree was built, as a sculpt or a transform leaves the scene's tree.
        let mut m = primitives::icosphere(1.0, 3);
        let mut b = primitives::icosphere(0.8, 3);
        b.transform(|p| p + DVec3::new(1.2, 0.1, 0.0));
        m.append_mesh(&b, 1);
        let mut bvh = mm_core::Bvh::build(&m);
        let moved: Vec<u32> = m.vertex_ids().filter(|&v| m.vertex(v).x > 1.0).collect();
        for v in moved {
            let p = m.vertex(v);
            m.set_vertex(v, p - DVec3::new(0.15, 0.0, 0.0));
        }
        bvh.refit(&m);
        let fresh = analyze(&m, None);
        let reused = analyze(&m, Some(std::sync::Arc::new(bvh)));
        assert!(!fresh.self_intersecting.is_empty());
        assert_eq!(format!("{fresh:?}"), format!("{reused:?}"));
    }
}
