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
    /// The inlet point was clicked, so it lies on the surface (an agent's may lie off it).
    inlet_clicked: bool,
    step: f64,
    auto_step: bool,
    job: Option<Job<Option<Centerline>>>,
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

/// The inlet near a picked point `p` on a closed model: the vertices of the triangle nearest
/// to it, and the vessel radius there. `bvh` is the mesh's tree if one is at hand (else, or if
/// it is for another state of the mesh, one is built).
fn inlet_near(mesh: &Mesh, bvh: Option<std::sync::Arc<mm_core::Bvh>>, p: DVec3) -> Option<(Vec<u32>, f64)> {
    let bvh = bvh.filter(|b| b.is_current(mesh)).unwrap_or_else(|| std::sync::Arc::new(mm_core::Bvh::build(mesh)));
    let h = bvh.nearest(mesh, p, f64::INFINITY)?;
    let r = mm_edit::estimate_vessel_cross_section(mesh, Some(&bvh), h.point, None).map(|x| x.radius).unwrap_or(mesh.average_edge_length() * 5.0);
    Some((mesh.triangle(h.tid).to_vec(), r))
}

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
            inlet_clicked: false,
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
        // The inlet's vertices and radius, or a picked point to find them from in the job.
        let inlet: Result<(Vec<u32>, f64), DVec3> = if let Some(i) = self.inlet {
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
            Ok((verts, e.diameter * 0.5))
        } else if let Some(p) = self.inlet_point {
            Err(p)
        } else {
            self.message = Some("Pick an inlet first.".into());
            return;
        };
        if inlet.as_ref().is_ok_and(|(sources, _)| sources.is_empty()) {
            self.message = Some("Could not find the inlet boundary.".into());
            return;
        }
        let params = CenterlineParams { step: if self.auto_step { 0.0 } else { self.step }, ..Default::default() };
        self.message = None;
        // Building the BVH takes about half a second on a large mesh, so that is done in the
        // job, or the scene's is used if it is up to date (the Shift+click that picked the point
        // made it). A clicked point nearly always lies inside a triangle, which any tree finds
        // alike. Where several triangles are equally near, which one is taken depends on the
        // order a tree meets them in: a click exactly on a shared edge or vertex can then start
        // from the other triangle there if the scene's tree was refitted rather than built, a
        // difference of one source vertex. A point off the surface (an agent's) is often
        // equally near several triangles around a vertex, so for those the job builds a fresh
        // tree, as before.
        let bvh = if self.inlet_clicked { o.cached_bvh() } else { None };
        self.job = Some(Job::spawn("Centerline", cx.egui.clone(), move |_| {
            let (sources, radius) = match inlet {
                Ok(s) => s,
                Err(p) => match inlet_near(&mesh, bvh, p) {
                    Some(s) => s,
                    None => return Ok(None),
                },
            };
            Ok(Some(centerline::compute(&mesh, &sources, radius, &params)))
        }));
    }

    fn save(&mut self, csv: bool) {
        let ext = if csv { "csv" } else { "vtk" };
        if self.result.is_none() {
            return;
        }
        if let Some(path) = rfd::FileDialog::new().add_filter(ext, &[ext]).set_file_name(format!("centerline.{ext}")).save_file() {
            self.message = Some(self.save_to(&path).unwrap_or_else(|e| e));
        }
    }

    /// Write the centerline as CSV or VTK (by the extension).
    fn save_to(&self, path: &std::path::Path) -> Result<String, String> {
        let cl = self.result.as_ref().ok_or("compute the centerline first")?;
        let text = match path.extension().and_then(|e| e.to_str()).map(str::to_lowercase).as_deref() {
            Some("csv") => cl.to_csv(),
            Some("vtk") => cl.to_vtk(),
            _ => return Err("the file must end in .csv or .vtk".into()),
        };
        std::fs::write(path, text).map(|()| format!("Saved {}", path.display())).map_err(|e| format!("Save failed: {e}"))
    }

    fn poll(&mut self, cx: &mut ToolCx) {
        if let Some(r) = self.job.as_ref().and_then(|j| j.poll()) {
            self.job = None;
            match r {
                Ok(Some(cl)) => {
                    *cx.status = format!("Centerline: {} branch(es)", cl.branches.len());
                    self.result = Some(cl);
                }
                // No surface near the picked point (an empty mesh): nothing to compute.
                Ok(None) => {}
                Err(e) => self.message = Some(format!("{e:#}")),
            }
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
        self.poll(cx);
        if self.job.is_some() {
            polymix_workspace::spinner(ui);
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
                self.inlet_clicked = true;
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

    fn settings(&self) -> Option<serde_json::Value> {
        Some(serde_json::json!({"spacing_mm": (!self.auto_step).then_some(self.step)}))
    }

    fn settings_schema(&self) -> Option<serde_json::Value> {
        Some(serde_json::json!({"type": "object", "properties": {"spacing_mm": {"type": ["number", "null"], "description": "Distance between centerline nodes along the vessel, mm; null for automatic (half the inlet radius)"}}}))
    }

    fn set_settings(&mut self, changes: &serde_json::Value, _cx: &mut ToolCx) -> Result<(), String> {
        let mut v = self.settings().unwrap_or_default();
        merge_settings(&mut v, changes)?;
        match &v["spacing_mm"] {
            serde_json::Value::Null => self.auto_step = true,
            x => {
                self.step = x.as_f64().filter(|s| *s > 0.0).ok_or("spacing_mm must be a positive number or null")?;
                self.auto_step = false;
            }
        }
        Ok(())
    }

    fn state(&self, _scene: &Scene) -> serde_json::Value {
        use polymix_workspace::agent::{round, vec3};
        use serde_json::json;
        json!({
            "object": self.object,
            "ends": self.ends.iter().enumerate().map(|(i, e)| json!({"end": i + 1, "centre_mm": vec3(e.center.to_array(), 3), "diameter_mm": round(e.diameter, 3), "inlet": self.inlet == Some(i)})).collect::<Vec<_>>(),
            "inlet": match (self.inlet, self.inlet_point) {
                (Some(i), _) => json!({"end": i + 1}),
                (None, Some(p)) => json!({"near_mm": vec3(p.to_array(), 3)}),
                _ => serde_json::Value::Null,
            },
            "computing": self.job.is_some(),
            "branches": self.result.as_ref().map(|cl| cl.branches.iter().enumerate().map(|(i, b)| {
                // Up to 12 samples along the branch, ends included.
                let n = b.nodes.len();
                let samples: Vec<serde_json::Value> = (0..n.min(12)).map(|k| if n <= 12 { k } else { k * (n - 1) / 11 }).map(|k| {
                    let node = &cl.nodes[b.nodes[k]];
                    json!({"at_mm": vec3(node.position.to_array(), 3), "diameter_mm": round(2.0 * node.radius, 3)})
                }).collect();
                json!({
                    "branch": i + 1,
                    "parent": b.parent.map(|p| p + 1),
                    "length_mm": round(b.length, 3),
                    "diameter_mm": {"mean": round(2.0 * b.mean_radius, 3), "min": round(2.0 * b.min_radius, 3), "max": round(2.0 * b.max_radius, 3)},
                    "samples": samples,
                })
            }).collect::<Vec<_>>()),
            "message": self.message,
            "next": if self.result.is_some() { "read the branches, or export" } else if self.inlet.is_some() || self.inlet_point.is_some() { "compute" } else { "set_inlet" },
        })
    }

    fn actions(&self) -> Vec<Action> {
        use serde_json::json;
        vec![
            Action::new("set_inlet", "Choose the inlet: an end by number (from the state's ends), or on a closed model a point near it", json!({"end": arg::integer("End number (from 1)"), "at": arg::point("A point in mm on or near the inlet")})),
            Action::plain("compute", "Compute the centerline tree from the inlet (in the background)"),
            Action::new("export", "Save the centerline: CSV (nodes, branches, radii) or VTK (polyline with Radius point data, for ParaView or VMTK), by the extension", json!({"path": arg::text("Absolute path ending in .csv or .vtk"), "overwrite": arg::boolean("Replace an existing file")})).required(&["path"]),
        ]
    }

    fn act(&mut self, action: &str, args: &serde_json::Value, cx: &mut ToolCx) -> Result<String, String> {
        let a = Args(args);
        self.poll(cx);
        match action {
            "set_inlet" => match (a.opt_usize("end"), a.opt_point("at")) {
                (Some(k), _) => {
                    let i = k.checked_sub(1).filter(|i| *i < self.ends.len()).ok_or_else(|| format!("there is no end {k}: the object has {}", self.ends.len()))?;
                    (self.inlet, self.inlet_point) = (Some(i), None);
                    Ok(format!("Inlet: end {k} (⌀ {:.2} mm)", self.ends[i].diameter))
                }
                (None, Some(p)) => {
                    (self.inlet, self.inlet_point, self.inlet_clicked) = (None, Some(DVec3::from_array(p)), false);
                    Ok("Inlet: near the given point".into())
                }
                _ => Err("give `end` or `at`".into()),
            },
            "compute" => {
                if self.job.is_some() {
                    return Err("already computing: app_wait".into());
                }
                self.start(cx);
                match &self.message {
                    Some(m) if self.job.is_none() => Err(m.clone()),
                    _ => Ok("Computing the centerline; app_wait, then read the branches".into()),
                }
            }
            "export" => {
                let path = std::path::PathBuf::from(a.str("path")?);
                if !path.is_absolute() {
                    return Err(format!("{} is not an absolute path", path.display()));
                }
                if path.exists() && !a.opt_bool("overwrite").unwrap_or(false) {
                    return Err(format!("{} exists: pass overwrite: true to replace it", path.display()));
                }
                let msg = self.save_to(&path)?;
                self.message = Some(msg.clone());
                Ok(msg)
            }
            _ => Err(format!("Centerline has no action {action}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_core::primitives;

    #[test]
    fn the_inlet_near_a_clicked_point_is_the_same_from_the_scene_tree() {
        // A closed, bent vessel; its tree built, then the vessel bent further (a sculpt), so the
        // scene's tree is a refitted one. Clicks land where rays from around hit it.
        let path: Vec<DVec3> = (0..30).map(|i| DVec3::new((i as f64 * 0.2).sin() * 3.0, 0.0, i as f64 * 0.6)).collect();
        let mut m = primitives::tube_along(&path, &[1.5; 30], 32);
        let caps: Vec<Vec<u32>> = mm_core::topology::boundary_loops(&m).into_iter().map(|l| l.vertices).collect();
        mm_edit::cap_planar_loops(&mut m, &caps, None, mm_edit::CutFill::Flat).unwrap();
        let mut refit = mm_core::Bvh::build(&m);
        let ids: Vec<u32> = m.vertex_ids().collect();
        for v in ids {
            let p = m.vertex(v);
            m.set_vertex(v, p + DVec3::new(0.0, 0.02 * p.z * p.z, 0.0));
        }
        refit.refit(&m);
        let refit = std::sync::Arc::new(refit);
        let fresh = mm_core::Bvh::build(&m);
        let mut clicks = 0;
        for k in 0..24 {
            let a = k as f64 * 0.7;
            let origin = DVec3::new(20.0 * a.cos(), 20.0 * a.sin(), 1.0 + k as f64 * 0.6);
            let Some(hit) = fresh.raycast(&m, &mm_core::Ray::new(origin, DVec3::new(0.0, 0.0, origin.z) - origin)) else { continue };
            let p = origin + (DVec3::new(0.0, 0.0, origin.z) - origin).normalize() * hit.t;
            let (built, reused) = (inlet_near(&m, None, p).unwrap(), inlet_near(&m, Some(refit.clone()), p).unwrap());
            assert_eq!(built.0, reused.0, "click {k}");
            assert_eq!(built.1.to_bits(), reused.1.to_bits(), "click {k}");
            clicks += 1;
        }
        assert!(clicks > 10, "{clicks} clicks hit");
    }
}
