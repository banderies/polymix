//! Inspector (hotkey I): find holes, non-manifold vertices and small floating parts, show
//! them as markers (blue = hole, red = non-manifold, magenta = small part), and repair them
//! one at a time by clicking a marker or all at once (Auto Repair All / Q).

use super::*;
use crate::jobs::Job;
use crate::scene::SceneObject;
use crate::ui_util;
use mm_repair::{ComponentInfo, Defect, FillMode, FillParams, HoleInfo, InspectParams, InspectionReport, RepairParams};
use std::sync::Arc;

pub struct InspectorTool {
    object: Option<ObjectId>,
    /// The defects of the object's mesh, whatever the options (see [`Analysis`]).
    analysis: Option<Analysis>,
    /// What the options list of it.
    shown: Shown,
    /// An analysis under way in the background, and the mesh it is for.
    inspecting: Option<(MeshKey, Job<Analysis>)>,
    /// An analysis that failed, and why (not tried again for the same mesh).
    failed: Option<(MeshKey, String)>,
    fill_mode: FillMode,
    small_fraction: f64,
    keep_flat_ends: bool,
    job: Option<Repair>,
    last_message: Option<String>,
}

/// A mesh as the Inspector tells meshes apart: its allocation and shape version.
type MeshKey = (u64, u64);

fn mesh_key(o: &SceneObject) -> MeshKey {
    (Arc::as_ptr(&o.mesh) as usize as u64, o.mesh.shape_version())
}

/// A mesh's defects with the options left open: every small part a threshold up to 100 % of
/// the area could list, and every hole with whether it is planar. Moving the threshold or
/// toggling flat ends then only filters it ([`Analysis::filter`]) instead of inspecting the
/// whole mesh again, which takes a good part of a second on large meshes.
struct Analysis {
    key: MeshKey,
    report: InspectionReport,
    /// Per hole of the report: a nearly planar loop (a cut vessel end), see [`is_planar_loop`].
    planar: Vec<bool>,
}

/// The holes and small parts the options list, as indices into [`Analysis::report`].
#[derive(Clone, Debug, Default, PartialEq)]
struct Shown {
    holes: Vec<usize>,
    parts: Vec<usize>,
}

impl Analysis {
    fn of(mesh: &Mesh, key: MeshKey) -> Self {
        let report = mm_repair::inspect(mesh, &InspectParams { small_component_area_fraction: 1.0, ..Default::default() });
        let planar = report.holes.iter().map(|h| is_planar_loop(mesh, &h.edge_loop)).collect();
        Self { key, report, planar }
    }

    /// What `inspect` lists with these options (and the flat-ends filter): its test of a small
    /// part, `area < fraction · total area || triangles < 10`, keeps a subset of the parts
    /// listed at 100 %, in the same order, and the total area is the same sum.
    fn filter(&self, small_fraction: f64, keep_flat_ends: bool) -> Shown {
        let r = &self.report;
        let min_triangles = InspectParams::default().small_component_tri_count;
        Shown {
            holes: (0..r.holes.len()).filter(|&i| !(keep_flat_ends && self.planar[i])).collect(),
            parts: (0..r.small_components.len())
                .filter(|&i| r.small_components[i].area < small_fraction * r.total_area || r.small_components[i].triangles.len() < min_triangles)
                .collect(),
        }
    }
}

/// A repair under way in the background.
enum Repair {
    /// Auto Repair All: the repaired mesh and its summary.
    All(Job<(Mesh, String)>),
    /// One clicked defect, repaired on a copy of `source`: the mesh and its message, or why
    /// the repair failed. It is kept only if the object still holds `source`.
    One { source: Arc<Mesh>, job: Job<Result<(Mesh, String), String>> },
}

impl Repair {
    fn progress(&self) -> f32 {
        match self {
            Repair::All(j) => j.progress(),
            Repair::One { job, .. } => job.progress(),
        }
    }
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

/// Repair `defect` on a copy of `src`: the repaired mesh and what was done, or why it failed.
fn repaired(src: &Mesh, defect: &Defect, mode: FillMode) -> Result<(Mesh, String), String> {
    let mut m = src.clone();
    let lp = if let Defect::Hole(l) = defect { Some(l) } else { None };
    let group = lp.map(|l| loop_group(&m, l)).unwrap_or_else(|| dominant_group(&m));
    let fp = FillParams { mode, group: Some(group), ..Default::default() };
    match mm_repair::repair_single(&mut m, defect, &fp) {
        Ok(r) => {
            let msg = match defect {
                Defect::Hole(_) => format!("Filled hole with {} triangles", r.fill.new_triangles.len()),
                Defect::BowtieVertex(_) => "Split non-manifold vertex".into(),
                Defect::SmallComponent(_) => format!("Removed {} triangles", r.removed_triangles),
                Defect::DegenerateTriangle(_) => "Fixed degenerate triangle".into(),
            };
            Ok((m, msg))
        }
        Err(e) => Err(format!("Repair failed: {e}")),
    }
}

impl InspectorTool {
    pub fn new(cx: &mut ToolCx) -> Self {
        Self {
            object: cx.scene.active,
            analysis: None,
            shown: Shown::default(),
            inspecting: None,
            failed: None,
            fill_mode: FillMode::Smooth,
            small_fraction: 0.01,
            keep_flat_ends: false,
            job: None,
            last_message: None,
        }
    }

    /// The analysis, if it is of the object's mesh as it is now.
    fn current<'a>(&'a self, scene: &Scene) -> Option<&'a Analysis> {
        let o = self.object.and_then(|id| scene.get(id))?;
        self.analysis.as_ref().filter(|a| a.key == mesh_key(o))
    }

    /// List what the options select of the analysis.
    fn refilter(&mut self) {
        self.shown = self.analysis.as_ref().map(|a| a.filter(self.small_fraction, self.keep_flat_ends)).unwrap_or_default();
    }

    fn set_analysis(&mut self, a: Analysis) {
        self.analysis = Some(a);
        self.refilter();
    }

    /// The mesh was changed here: its analysis is gone, and so is a failure of one (so that
    /// [`Tool::pending`] holds until the new mesh is analysed).
    fn forget_analysis(&mut self) {
        self.analysis = None;
        self.failed = None;
    }

    /// Analyse the object's mesh now unless that is done (for agents and scripts, which read
    /// the result at once).
    fn analyze(&mut self, cx: &mut ToolCx) {
        let Some(o) = self.object.and_then(|id| cx.scene.get(id)) else { return };
        let key = mesh_key(o);
        if self.analysis.as_ref().is_some_and(|a| a.key == key) {
            return;
        }
        let a = Analysis::of(&o.mesh, key);
        self.inspecting = None;
        self.set_analysis(a);
    }

    /// Keep the analysis up to date in the background: collect a finished one, and start one
    /// when the mesh is not the one analysed. Without the object, one under way is dropped, so
    /// the tool does not stay busy.
    fn update_analysis(&mut self, cx: &mut ToolCx) {
        let Some(o) = self.object.and_then(|id| cx.scene.get(id)) else {
            self.inspecting = None;
            return;
        };
        let key = mesh_key(o);
        if let Some(r) = self.inspecting.as_ref().and_then(|(_, j)| j.poll()) {
            let (k, _) = self.inspecting.take().unwrap();
            match r {
                Ok(a) if a.key == key => self.set_analysis(a),
                Ok(_) => {}
                Err(e) => self.failed = Some((k, format!("Inspection failed: {e:#}"))),
            }
        }
        let done = self.analysis.as_ref().is_some_and(|a| a.key == key) || self.inspecting.as_ref().is_some_and(|(k, _)| *k == key) || self.failed.as_ref().is_some_and(|(k, _)| *k == key);
        if !done {
            let mesh = o.mesh.clone();
            self.inspecting = Some((key, Job::spawn("Inspector", cx.egui.clone(), move |_| Ok(Analysis::of(&mesh, key)))));
        }
    }

    /// The listed holes.
    fn holes<'a>(&'a self, a: &'a Analysis) -> impl Iterator<Item = &'a HoleInfo> + 'a {
        self.shown.holes.iter().map(|&i| &a.report.holes[i])
    }

    /// The listed small parts.
    fn parts<'a>(&'a self, a: &'a Analysis) -> impl Iterator<Item = &'a ComponentInfo> + 'a {
        self.shown.parts.iter().map(|&i| &a.report.small_components[i])
    }

    /// As [`InspectionReport::is_clean`] for what is listed.
    fn is_clean(&self, a: &Analysis) -> bool {
        let r = &a.report;
        self.shown.holes.is_empty() && r.bowtie_vertices.is_empty() && self.shown.parts.is_empty() && r.degenerate_triangles.is_empty() && r.inconsistent_orientation_edges == 0
    }

    /// Repair one defect now (agents).
    fn repair_one(&mut self, cx: &mut ToolCx, defect: Defect) {
        let Some(o) = self.object.and_then(|id| cx.scene.get(id)) else { return };
        let id = o.id;
        match repaired(&o.mesh, &defect, self.fill_mode) {
            Ok((m, msg)) => {
                cx.history.push("Inspector repair", cx.scene);
                if let Some(o) = cx.scene.get_mut(id) {
                    o.set_mesh(m);
                }
                self.last_message = Some(msg);
                self.forget_analysis();
            }
            Err(e) => self.last_message = Some(e),
        }
    }

    /// Repair one clicked defect in the background (a fill can take seconds on a large mesh).
    fn start_repair_one(&mut self, cx: &mut ToolCx, defect: Defect) {
        let Some(o) = self.object.and_then(|id| cx.scene.get(id)) else { return };
        let (source, src) = (o.mesh.clone(), o.mesh.clone());
        let mode = self.fill_mode;
        self.job = Some(Repair::One { source, job: Job::spawn("Inspector repair", cx.egui.clone(), move |_| Ok(repaired(&src, &defect, mode))) });
    }

    fn auto_repair(&mut self, cx: &mut ToolCx) {
        let Some(o) = self.object.and_then(|id| cx.scene.get(id)) else { return };
        let src = o.mesh.clone();
        let mode = self.fill_mode;
        let frac = self.small_fraction;
        let keep_ends = self.keep_flat_ends;
        self.job = Some(Repair::All(Job::spawn("Auto repair", cx.egui.clone(), move |progress| {
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
        })));
    }

    /// Collect a finished repair. Returns false while one is still running.
    fn poll_repair(&mut self, cx: &mut ToolCx) -> bool {
        let id = self.object.filter(|id| cx.scene.get(*id).is_some());
        match &self.job {
            None => {}
            Some(Repair::All(j)) => {
                let Some(r) = j.poll() else { return false };
                self.job = None;
                match (r, id) {
                    (Ok((m, msg)), Some(id)) => {
                        cx.history.push("Auto repair", cx.scene);
                        if let Some(o) = cx.scene.get_mut(id) {
                            o.set_mesh(m);
                        }
                        *cx.status = msg.clone();
                        self.last_message = Some(msg);
                        self.forget_analysis();
                    }
                    (Err(e), _) => self.last_message = Some(format!("Auto repair failed: {e:#}")),
                    (Ok(_), None) => self.last_message = None,
                }
            }
            Some(Repair::One { source, job }) => {
                let Some(r) = job.poll() else { return false };
                let unchanged = id.and_then(|id| cx.scene.get(id)).is_some_and(|o| Arc::ptr_eq(&o.mesh, source));
                self.job = None;
                match (r, id) {
                    (Ok(Ok((m, msg))), Some(id)) if unchanged => {
                        cx.history.push("Inspector repair", cx.scene);
                        if let Some(o) = cx.scene.get_mut(id) {
                            o.set_mesh(m);
                        }
                        self.last_message = Some(msg);
                        self.forget_analysis();
                    }
                    (Ok(Ok(_)), Some(_)) => self.last_message = Some("Repair skipped: the object changed meanwhile".into()),
                    (Ok(Err(e)), _) => self.last_message = Some(e),
                    (Err(e), _) => self.last_message = Some(format!("Repair failed: {e:#}")),
                    (Ok(Ok(_)), None) => self.last_message = None,
                }
            }
        }
        true
    }
}

const FILL_MODES: [(&str, FillMode); 3] = [("minimal", FillMode::Minimal), ("flat", FillMode::Flat), ("smooth", FillMode::Smooth)];

impl Tool for InspectorTool {
    fn survives_scene_change(&self) -> bool {
        true
    }

    fn settings(&self) -> Option<serde_json::Value> {
        let mode = FILL_MODES.iter().find(|(_, m)| *m == self.fill_mode).map_or("smooth", |(k, _)| *k);
        Some(serde_json::json!({"fill_mode": mode, "small_part_fraction": self.small_fraction, "keep_flat_ends": self.keep_flat_ends}))
    }

    fn settings_schema(&self) -> Option<serde_json::Value> {
        Some(serde_json::json!({"type": "object", "properties": {
            "fill_mode": {"type": "string", "enum": FILL_MODES.map(|(k, _)| k), "description": "How holes are filled: minimal (no new vertices), flat (refined membrane) or smooth (refined, continuing the surrounding curvature)"},
            "small_part_fraction": {"type": "number", "description": "Parts with less than this fraction of the total area are small parts (0.0001..0.2; 0.01 = 1 %)"},
            "keep_flat_ends": {"type": "boolean", "description": "Treat flat open boundaries as intentional vessel inlets / outlets: not listed as holes, not filled"},
        }}))
    }

    fn set_settings(&mut self, changes: &serde_json::Value, _cx: &mut ToolCx) -> Result<(), String> {
        let mut v = self.settings().unwrap_or_default();
        merge_settings(&mut v, changes)?;
        let mode = v["fill_mode"].as_str().unwrap_or_default();
        self.fill_mode = FILL_MODES.iter().find(|(k, _)| *k == mode).map(|(_, m)| *m).ok_or_else(|| format!("fill_mode must be one of {:?}", FILL_MODES.map(|(k, _)| k)))?;
        self.small_fraction = v["small_part_fraction"].as_f64().ok_or("small_part_fraction must be a number")?.clamp(0.0001, 0.2);
        self.keep_flat_ends = v["keep_flat_ends"].as_bool().ok_or("keep_flat_ends must be true or false")?;
        self.refilter();
        Ok(())
    }

    fn pending(&self) -> bool {
        self.analysis.is_none() && self.inspecting.is_none() && self.failed.is_none() && self.job.is_none() && self.object.is_some()
    }

    fn state(&self, scene: &Scene) -> serde_json::Value {
        use polymix_workspace::agent::{round, vec3};
        use serde_json::json;
        let o = self.object.and_then(|id| scene.get(id));
        let current = self.current(scene).is_some();
        json!({
            "object": self.object,
            "repairing": self.job.as_ref().map(|j| round(j.progress() as f64, 2)),
            "report": self.analysis.as_ref().map(|a| {
                let r = &a.report;
                json!({
                    "for_current_mesh": current,
                    "clean": self.is_clean(a),
                    "holes": self.holes(a).enumerate().map(|(i, h)| json!({"hole": i + 1, "centre_mm": vec3(h.centroid.to_array(), 3), "edges": h.num_edges, "perimeter_mm": round(h.perimeter, 3), "touches_non_manifold": h.touches_bowtie})).collect::<Vec<_>>(),
                    "non_manifold_vertices": r.bowtie_vertices.iter().enumerate().map(|(i, &v)| json!({"vertex": i + 1, "at_mm": o.map(|o| vec3(o.mesh.vertex(v).to_array(), 3))})).collect::<Vec<_>>(),
                    "small_parts": self.parts(a).enumerate().map(|(i, c)| json!({"part": i + 1, "centre_mm": vec3(c.centroid.to_array(), 3), "triangles": c.triangles.len(), "area_mm2": round(c.area, 3)})).collect::<Vec<_>>(),
                    "shells": r.num_components,
                    "degenerate_triangles": r.degenerate_triangles.len(),
                    "inconsistently_oriented_edges": r.inconsistent_orientation_edges,
                    "crack_vertices": r.crack_vertices,
                })
            }),
            "message": self.last_message,
        })
    }

    fn actions(&self) -> Vec<Action> {
        use serde_json::json;
        vec![
            Action::plain("inspect", "Look for defects now (also done by itself while the panel is shown)"),
            Action::plain("repair_all", "Auto Repair All: fill holes (keeping flat ends if set), remove small parts, split non-manifold vertices, weld cracks, fix orientation"),
            Action::new("repair", "Repair one defect from the report", json!({"defect": arg::choice(&["hole", "non_manifold_vertex", "small_part"], "Which list"), "number": arg::integer("Its number in that list (from 1)")})).required(&["defect", "number"]),
        ]
    }

    fn act(&mut self, action: &str, args: &serde_json::Value, cx: &mut ToolCx) -> Result<String, String> {
        if self.job.is_some() {
            return Err("a repair is running: app_wait".into());
        }
        let a = Args(args);
        match action {
            "inspect" => {
                self.forget_analysis();
                self.analyze(cx);
                let an = self.analysis.as_ref().ok_or("no active object")?;
                Ok(format!("{} hole(s), {} non-manifold vertex(es), {} small part(s), {} shell(s)", self.shown.holes.len(), an.report.bowtie_vertices.len(), self.shown.parts.len(), an.report.num_components))
            }
            "repair_all" => {
                self.object.and_then(|id| cx.scene.get(id)).ok_or("no active object")?;
                self.auto_repair(cx);
                Ok("Repairing; app_wait, then read the message and report".into())
            }
            "repair" => {
                self.analyze(cx);
                let an = self.analysis.as_ref().ok_or("no active object")?;
                let k = a.usize("number")?.checked_sub(1).ok_or("numbers start at 1")?;
                let defect = match a.str("defect")? {
                    "hole" => self.shown.holes.get(k).map(|&i| Defect::Hole(an.report.holes[i].edge_loop.clone())),
                    "non_manifold_vertex" => an.report.bowtie_vertices.get(k).map(|&v| Defect::BowtieVertex(v)),
                    _ => self.shown.parts.get(k).map(|&i| Defect::SmallComponent(an.report.small_components[i].triangles.clone())),
                }
                .ok_or_else(|| format!("there is no {} {}", a.str("defect").unwrap_or_default(), k + 1))?;
                self.repair_one(cx, defect);
                self.analyze(cx);
                Ok(self.last_message.clone().unwrap_or_default())
            }
            _ => Err(format!("Inspector has no action {action}")),
        }
    }

    fn name(&self) -> &'static str {
        "Inspector"
    }

    fn panel(&mut self, ui: &mut egui::Ui, cx: &mut ToolCx) -> ToolOutcome {
        ui_util::section(ui, "Inspector");
        // Poll first so a job whose object vanished still finishes (and stops being busy).
        if !self.poll_repair(cx) {
            let p = self.job.as_ref().map_or(0.0, Repair::progress);
            ui.horizontal(|ui| {
                polymix_workspace::spinner(ui);
                ui.add(egui::ProgressBar::new(p).show_percentage());
            });
            return ToolOutcome::Continue;
        }
        self.update_analysis(cx);
        if self.object.and_then(|id| cx.scene.get(id)).is_none() {
            ui.label("No active object.");
            return if ui.button("Close").clicked() { ToolOutcome::Close } else { ToolOutcome::Continue };
        }
        let Some(a) = self.current(cx.scene) else {
            match self.failed.as_ref().filter(|_| self.inspecting.is_none()) {
                Some((_, e)) => {
                    ui.colored_label(egui::Color32::from_rgb(240, 110, 90), e);
                }
                None => {
                    ui.horizontal(|ui| {
                        polymix_workspace::spinner(ui);
                        ui.label("Inspecting…");
                    });
                }
            }
            return ToolOutcome::Continue;
        };
        let r = &a.report;
        let row = |ui: &mut egui::Ui, label: &str, n: usize, color: egui::Color32| {
            ui.horizontal(|ui| {
                ui.colored_label(color, "●");
                ui.label(format!("{label}: {n}"));
            });
        };
        row(ui, "Holes", self.shown.holes.len(), egui::Color32::from_rgb(60, 140, 255));
        row(ui, "Non-manifold vertices", r.bowtie_vertices.len(), egui::Color32::from_rgb(230, 60, 60));
        row(ui, "Small parts", self.shown.parts.len(), egui::Color32::from_rgb(220, 60, 220));
        ui.weak(format!(
            "{} shell(s) · {} degenerate triangle(s) · {} inconsistently oriented edge(s)",
            r.num_components,
            r.degenerate_triangles.len(),
            r.inconsistent_orientation_edges
        ));
        if self.is_clean(a) {
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
            self.refilter();
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
        // Markers of an analysis of an earlier mesh are hidden and not clickable.
        let (Some(pos), Some(a), Some(id)) = (input.pos, self.current(cx.scene), self.object) else { return };
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
        for h in self.holes(a) {
            consider(h.centroid, Defect::Hole(h.edge_loop.clone()), cx);
        }
        for &v in &a.report.bowtie_vertices {
            consider(mesh.vertex(v), Defect::BowtieVertex(v), cx);
        }
        for c in self.parts(a) {
            consider(c.centroid, Defect::SmallComponent(c.triangles.clone()), cx);
        }
        if let Some((_, d)) = best {
            self.start_repair_one(cx, d);
        }
    }

    fn paint(&mut self, painter: &egui::Painter, cx: &mut ToolCx) {
        let Some(a) = self.current(cx.scene) else { return };
        let Some(o) = self.object.and_then(|id| cx.scene.get(id)) else { return };
        let mesh = o.mesh.clone();
        let stroke = egui::Stroke::new(1.5, egui::Color32::WHITE);
        for h in self.holes(a) {
            if let Some(p) = cx.project(h.centroid) {
                painter.circle(p, 7.0, egui::Color32::from_rgb(60, 140, 255), stroke);
            }
        }
        for &v in &a.report.bowtie_vertices {
            if let Some(p) = cx.project(mesh.vertex(v)) {
                painter.circle(p, 6.0, egui::Color32::from_rgb(230, 60, 60), stroke);
            }
        }
        for c in self.parts(a) {
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
                self.refilter();
            }
            _ => {}
        }
    }

    fn busy(&self) -> Option<(String, f32)> {
        match (&self.job, &self.inspecting) {
            (Some(j), _) => Some(("Repairing".to_string(), j.progress())),
            (None, Some(_)) => Some(("Inspecting".to_string(), 0.5)),
            (None, None) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_core::primitives;

    /// An open tube (two flat ends) with a curved hole in its wall, and floating parts from
    /// tiny to large, some with fewer than ten triangles.
    fn defective() -> Mesh {
        let path: Vec<DVec3> = (0..40).map(|i| DVec3::new(0.0, 0.0, i as f64 * 0.5)).collect();
        let mut m = primitives::tube_along(&path, &[2.0; 40], 48);
        let centre = DVec3::new(2.0, 0.0, 10.0);
        let wall: Vec<u32> = m.triangle_ids().filter(|&t| m.tri_centroid(t).distance(centre) < 2.5).collect();
        m.remove_triangles(wall);
        for (k, r) in [0.05, 0.2, 0.5, 0.9, 1.5, 2.5].into_iter().enumerate() {
            let mut s = primitives::icosphere(r, 2);
            s.transform(|p| p + DVec3::new(10.0 + 6.0 * k as f64, 0.0, 0.0));
            m.append_mesh(&s, 0);
        }
        // A lone triangle, and a tetrahedron.
        let a = m.append_vertex(DVec3::new(-10.0, 0.0, 0.0));
        let b = m.append_vertex(DVec3::new(-9.0, 0.0, 0.0));
        let c = m.append_vertex(DVec3::new(-10.0, 1.0, 0.0));
        m.append_triangle([a, b, c], 0).unwrap();
        let mut tet = primitives::icosphere(3.0, 0);
        tet.transform(|p| p + DVec3::new(-20.0, 0.0, 0.0));
        m.append_mesh(&tet, 0);
        m
    }

    #[test]
    fn filtering_one_analysis_lists_what_inspecting_with_the_options_lists() {
        let m = defective();
        let a = Analysis::of(&m, (0, 0));
        let mut kept_ends = false;
        for fraction in [0.0001, 0.001, 0.004, 0.01, 0.03, 0.08, 0.2] {
            for keep_flat_ends in [false, true] {
                // The Inspector's old path: inspect with the threshold, then drop flat ends.
                let mut direct = mm_repair::inspect(&m, &InspectParams { small_component_area_fraction: fraction, ..Default::default() });
                if keep_flat_ends {
                    direct.holes.retain(|h| !is_planar_loop(&m, &h.edge_loop));
                }
                let shown = a.filter(fraction, keep_flat_ends);
                let holes: Vec<&HoleInfo> = shown.holes.iter().map(|&i| &a.report.holes[i]).collect();
                let parts: Vec<&ComponentInfo> = shown.parts.iter().map(|&i| &a.report.small_components[i]).collect();
                // Debug prints floats exactly (shortest round trip), so equal text is equal bits.
                assert_eq!(format!("{holes:?}"), format!("{:?}", direct.holes.iter().collect::<Vec<_>>()), "{fraction} {keep_flat_ends}");
                assert_eq!(parts, direct.small_components.iter().collect::<Vec<_>>(), "{fraction} {keep_flat_ends}");
                assert_eq!(format!("{parts:?}"), format!("{:?}", direct.small_components.iter().collect::<Vec<_>>()));
                assert_eq!(a.report.total_area.to_bits(), direct.total_area.to_bits());
                assert_eq!(format!("{:?}", a.report.bowtie_vertices), format!("{:?}", direct.bowtie_vertices));
                assert_eq!(a.report.num_components, direct.num_components);
                kept_ends |= keep_flat_ends && direct.holes.len() < a.report.holes.len();
            }
        }
        // The case exercises both filters.
        assert!(kept_ends, "flat ends were dropped somewhere");
        assert!(a.filter(0.0001, false).parts.len() < a.filter(0.2, false).parts.len());
        assert!(a.report.holes.len() >= 3);
    }

    #[test]
    fn the_panel_inspects_and_repairs_in_the_background_as_agents_do_at_once() {
        use serde_json::json;
        use std::time::{Duration, Instant};
        let mut scene = Scene::default();
        let id = scene.add("broken", defective());
        let (mut history, mut camera, mut status) = (crate::scene::History::default(), crate::camera::Camera::default(), String::new());
        let ctx = egui::Context::default();
        let vp = Viewport { rect: egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0)) };
        let mut cx = ToolCx { scene: &mut scene, history: &mut history, camera: &mut camera, status: &mut status, egui: &ctx, vp, brush_scroll: crate::navigation::BrushScroll::Shift };
        let mut tool = InspectorTool::new(&mut cx);
        assert!(tool.pending());
        let wait = |tool: &mut InspectorTool, cx: &mut ToolCx, done: &dyn Fn(&InspectorTool, &Scene) -> bool| {
            let t0 = Instant::now();
            while !done(tool, cx.scene) && t0.elapsed() < Duration::from_secs(30) {
                tool.update_analysis(cx);
                tool.poll_repair(cx);
                std::thread::sleep(Duration::from_millis(5));
            }
        };
        // The panel analyses in a job (busy meanwhile), then lists the defects.
        tool.update_analysis(&mut cx);
        assert!(tool.busy().is_some() && tool.current(cx.scene).is_none());
        wait(&mut tool, &mut cx, &|t, s| t.current(s).is_some());
        assert!(tool.busy().is_none() && !tool.pending());
        // An agent's inspect does it at once, and reports the same.
        let mut agent = InspectorTool::new(&mut cx);
        agent.act("inspect", &json!({}), &mut cx).unwrap();
        assert_eq!(agent.state(cx.scene), tool.state(cx.scene));
        // Settings filter the analysis as inspecting with them does.
        let s = json!({"small_part_fraction": 0.2, "keep_flat_ends": true});
        tool.set_settings(&s, &mut cx).unwrap();
        agent.set_settings(&s, &mut cx).unwrap();
        agent.act("inspect", &json!({}), &mut cx).unwrap();
        assert_eq!(agent.state(cx.scene), tool.state(cx.scene));
        // A clicked repair runs in a job and gives the agent's result.
        let hole = Defect::Hole(tool.holes(tool.analysis.as_ref().unwrap()).next().unwrap().edge_loop.clone());
        let holes = tool.shown.holes.len();
        let before = cx.scene.get(id).unwrap().mesh.clone();
        tool.start_repair_one(&mut cx, hole.clone());
        assert!(tool.busy().is_some());
        wait(&mut tool, &mut cx, &|t, _| t.job.is_none());
        let clicked = cx.scene.get(id).unwrap().mesh.clone();
        assert!(!Arc::ptr_eq(&before, &clicked), "{:?}", tool.last_message);
        cx.scene.get_mut(id).unwrap().mesh = before.clone();
        agent.repair_one(&mut cx, hole.clone());
        assert_eq!(agent.last_message, tool.last_message);
        let by_agent = cx.scene.get(id).unwrap().mesh.clone();
        assert_eq!(by_agent.to_arrays().1, clicked.to_arrays().1);
        // The panel analyses the repaired mesh again.
        wait(&mut tool, &mut cx, &|t, s| t.current(s).is_some());
        assert_eq!(tool.shown.holes.len() + 1, holes);
        // A repair of a mesh changed meanwhile is dropped.
        cx.scene.get_mut(id).unwrap().mesh = before;
        tool.start_repair_one(&mut cx, hole);
        cx.scene.get_mut(id).unwrap().mesh = by_agent;
        wait(&mut tool, &mut cx, &|t, _| t.job.is_none());
        assert_eq!(tool.last_message.as_deref(), Some("Repair skipped: the object changed meanwhile"));
        // After a repair, a failed analysis of the mesh before it does not end the wait for the
        // next one.
        wait(&mut tool, &mut cx, &|t, s| t.current(s).is_some());
        let part = Defect::SmallComponent(tool.parts(tool.analysis.as_ref().unwrap()).next().unwrap().triangles.clone());
        tool.failed = Some((mesh_key(cx.scene.get(id).unwrap()), "Inspection failed".into()));
        tool.repair_one(&mut cx, part);
        assert!(tool.pending() && tool.failed.is_none());
        // Without its object, an analysis under way is dropped, and the tool is not busy.
        tool.update_analysis(&mut cx);
        assert!(tool.busy().is_some());
        cx.scene.remove(id);
        tool.update_analysis(&mut cx);
        assert!(tool.busy().is_none() && tool.inspecting.is_none());
    }
}
