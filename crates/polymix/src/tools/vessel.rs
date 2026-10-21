//! Vessel / CFD preparation tools: square branch trimming, flow extensions, capping, and
//! cross-section estimation.

use super::*;
use crate::ui_util;
use polymix_workspace::agent::{round, vec3};
use serde_json::{json, Value};

// ---------------------------------------------------------------------------
// Vessel ends (open boundary loops and planar caps)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub enum EndKind {
    /// An open boundary loop (vertex ids at detection time).
    Open,
    /// A planar cap made of one face group.
    Cap(i32),
}

#[derive(Clone, Debug)]
pub struct VesselEnd {
    pub kind: EndKind,
    pub center: DVec3,
    /// Outward normal.
    #[allow(dead_code)]
    pub normal: DVec3,
    pub diameter: f64,
    /// Open loops only: whether the loop is nearly planar (a cut end rather than a wall hole).
    pub planar: bool,
}

/// Newell normal and area of a closed polygon.
fn polygon_normal_area(pts: &[DVec3]) -> (DVec3, f64) {
    let mut n = DVec3::ZERO;
    for i in 0..pts.len() {
        let (a, b) = (pts[i], pts[(i + 1) % pts.len()]);
        n += a.cross(b);
    }
    let area = 0.5 * n.length();
    (n.normalize_or_zero(), area)
}

/// Find open boundary loops and flat single-group caps (inlets / outlets).
pub fn find_vessel_ends(mesh: &Mesh) -> Vec<VesselEnd> {
    let mut ends = Vec::new();
    let mesh_c = mesh.bounds().center();
    for l in mm_core::topology::boundary_loops(mesh) {
        if l.len() < 3 {
            continue;
        }
        let pts = l.positions(mesh);
        let c = l.centroid(mesh);
        let (mut n, area) = polygon_normal_area(&pts);
        // Loop edges follow their triangles, so the Newell normal points into the surface
        // side; orient it away from the mesh instead.
        if (c - mesh_c).dot(n) < 0.0 {
            n = -n;
        }
        let diameter = 2.0 * (area / std::f64::consts::PI).sqrt();
        let max_off = pts.iter().map(|p| (*p - c).dot(n).abs()).fold(0.0, f64::max);
        let planar = max_off < 0.15 * diameter.max(1e-12);
        ends.push(VesselEnd { kind: EndKind::Open, center: c, normal: n, diameter, planar });
    }
    let total = mesh.triangle_count().max(1);
    for (g, tris) in group_triangles(mesh) {
        if g == 0 || tris.len() * 5 > total {
            continue;
        }
        let mut nsum = DVec3::ZERO;
        let mut csum = DVec3::ZERO;
        let mut area = 0.0;
        for &t in &tris {
            let a = mesh.tri_area(t);
            nsum += mesh.tri_cross(t);
            csum += mesh.tri_centroid(t) * a;
            area += a;
        }
        if area <= 0.0 {
            continue;
        }
        let n = nsum.normalize_or_zero();
        let flat = tris.iter().all(|&t| mesh.tri_normal(t).dot(n) > 0.97);
        if flat {
            ends.push(VesselEnd { kind: EndKind::Cap(g), center: csum / area, normal: n, diameter: 2.0 * (area / std::f64::consts::PI).sqrt(), planar: true });
        }
    }
    ends
}

fn group_triangles(mesh: &Mesh) -> std::collections::BTreeMap<i32, Vec<u32>> {
    let mut m: std::collections::BTreeMap<i32, Vec<u32>> = Default::default();
    for t in mesh.triangle_ids() {
        m.entry(mesh.tri_group(t)).or_default().push(t);
    }
    m
}

/// Boundary loop whose centroid is nearest `p`.
fn nearest_loop(mesh: &Mesh, p: DVec3) -> Option<Vec<u32>> {
    mm_core::topology::boundary_loops(mesh)
        .into_iter()
        .min_by(|a, b| a.centroid(mesh).distance(p).partial_cmp(&b.centroid(mesh).distance(p)).unwrap())
        .map(|l| l.vertices)
}

/// Next free "outlet_N" style name.
fn next_name(names: &std::collections::HashMap<i32, String>, prefix: &str) -> String {
    let mut k = 1;
    loop {
        let n = format!("{prefix}_{k}");
        if !names.values().any(|v| *v == n) {
            return n;
        }
        k += 1;
    }
}

fn draw_end_markers(painter: &egui::Painter, cx: &mut ToolCx, ends: &[VesselEnd], selected: &[bool]) {
    for (i, e) in ends.iter().enumerate() {
        if let Some(p) = cx.project(e.center) {
            let on = selected.get(i).copied().unwrap_or(false);
            let fill = if on { ui_util::ACCENT } else { egui::Color32::from_rgba_unmultiplied(60, 140, 255, 160) };
            painter.circle(p, 8.0, fill, egui::Stroke::new(1.5, egui::Color32::WHITE));
            painter.text(p + egui::vec2(11.0, -11.0), egui::Align2::LEFT_BOTTOM, format!("{}", i + 1), egui::FontId::proportional(13.0), egui::Color32::WHITE);
        }
    }
}

/// Vessel ends as agents read them (numbered from 1, as the markers are).
fn ends_state(ends: &[VesselEnd], picked: impl Fn(usize) -> bool) -> Vec<Value> {
    ends.iter()
        .enumerate()
        .map(|(i, e)| {
            let kind = match e.kind {
                EndKind::Open if e.planar => json!("open end"),
                EndKind::Open => json!("hole (not flat: fill it with the Inspector)"),
                EndKind::Cap(g) => json!(format!("flat cap, face group {g}")),
            };
            json!({"end": i + 1, "kind": kind, "centre_mm": vec3(e.center.to_array(), 3), "outward": vec3(e.normal.to_array(), 3), "diameter_mm": round(e.diameter, 3), "picked": picked(i)})
        })
        .collect()
}

/// End numbers (from 1) in an action's `ends` argument.
fn end_numbers(args: &Value, count: usize) -> Result<Vec<usize>, String> {
    let list = args.get("ends").and_then(Value::as_array).ok_or("`ends` must be a list of end numbers")?;
    list.iter()
        .map(|v| match v.as_u64() {
            Some(k) if k >= 1 && (k as usize) <= count => Ok(k as usize - 1),
            _ => Err(format!("{v} is not an end number (1 to {count})")),
        })
        .collect()
}

fn ends_arg(about: &str) -> Value {
    json!({"type": "array", "items": {"type": "integer"}, "description": about})
}

fn end_under_cursor(cx: &mut ToolCx, ends: &[VesselEnd], pos: egui::Pos2) -> Option<usize> {
    ends.iter()
        .enumerate()
        .filter_map(|(i, e)| cx.project(e.center).map(|p| (i, p.distance(pos))))
        .filter(|(_, d)| *d < 14.0)
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|(i, _)| i)
}

// ---------------------------------------------------------------------------
// Trim branch
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TrimFill {
    Remeshed,
    Flat,
    Open,
}

impl TrimFill {
    const KEYS: [(&str, TrimFill); 3] = [("flat_cap_remeshed", TrimFill::Remeshed), ("flat_cap", TrimFill::Flat), ("open", TrimFill::Open)];

    fn key(self) -> &'static str {
        Self::KEYS.iter().find(|(_, f)| *f == self).map(|(k, _)| *k).unwrap_or("flat_cap_remeshed")
    }
}

struct TrimCandidate {
    seed: DVec3,
    plane: mm_core::Plane,
    radius: f64,
    curve: Vec<[DVec3; 2]>,
}

struct TrimOut {
    key: (DVec3, f64, TrimFill),
    source: std::sync::Arc<Mesh>,
    mesh: Mesh,
    caps: Vec<i32>,
    msg: String,
}

/// Click a vessel branch to cut it perpendicular to its axis (only that branch) and cap it.
pub struct TrimTool {
    object: Option<ObjectId>,
    fill: TrimFill,
    cand: Option<TrimCandidate>,
    job: Option<crate::jobs::Job<TrimOut>>,
    preview: Option<TrimOut>,
    error: Option<String>,
    trims: usize,
}

impl TrimTool {
    pub fn new(cx: &mut ToolCx) -> Self {
        Self { object: cx.scene.active, fill: TrimFill::Remeshed, cand: None, job: None, preview: None, error: None, trims: 0 }
    }

    fn key(&self) -> Option<(DVec3, f64, TrimFill)> {
        self.cand.as_ref().map(|c| (c.plane.normal, c.plane.d, self.fill))
    }

    fn refresh_curve(&mut self, mesh: &Mesh) {
        if let Some(c) = &mut self.cand {
            c.curve = mm_edit::nearest_intersection_curve(mesh, c.plane, c.seed).map(|k| k.segments).unwrap_or_default();
        }
    }

    fn start(&mut self, cx: &mut ToolCx) {
        let (Some(id), Some(c)) = (self.object, &self.cand) else { return };
        let Some(o) = cx.scene.get(id) else { return };
        let src = o.mesh.clone();
        let source = src.clone();
        let (plane, seed, fill) = (c.plane, c.seed, self.fill);
        let key = self.key().unwrap();
        let target = src.average_edge_length();
        if let Some(j) = &self.job {
            j.cancel();
        }
        self.error = None;
        self.job = Some(crate::jobs::Job::spawn("Trim", cx.egui.clone(), move |_| {
            let mut m = (*src).clone();
            let cut_fill = match fill {
                TrimFill::Remeshed => mm_edit::CutFill::FlatRemeshed { target_edge_length: target },
                TrimFill::Flat => mm_edit::CutFill::Flat,
                TrimFill::Open => mm_edit::CutFill::None,
            };
            let r = mm_edit::local_plane_cut(&mut m, plane, seed, mm_edit::KeepSide::Below, cut_fill)?;
            let msg = format!("removed {} triangles", r.cut.removed_triangles);
            Ok(TrimOut { key, source, mesh: m, caps: r.cut.cap_groups.clone(), msg })
        }));
    }

    /// A cut across the vessel at the wall point `p` of object `id`.
    fn cut_at(&mut self, cx: &mut ToolCx, id: ObjectId, p: DVec3) -> Result<f64, String> {
        self.object = Some(id);
        cx.scene.active = Some(id);
        let o = cx.scene.get_mut(id).ok_or("no object")?;
        let bvh = o.bvh();
        let mesh = o.mesh.clone();
        let x = mm_edit::estimate_vessel_cross_section(&mesh, Some(&bvh), p, None).map_err(|e| {
            self.error = Some(format!("{e}"));
            format!("{e}")
        })?;
        self.cand = Some(TrimCandidate { seed: p, plane: x.plane, radius: x.radius, curve: Vec::new() });
        self.preview = None;
        self.refresh_curve(&mesh);
        self.start(cx);
        Ok(2.0 * x.radius)
    }

    /// Move the cut along the vessel (positive: toward the tip, removing less) and / or flip
    /// which side is removed.
    fn shift(&mut self, cx: &mut ToolCx, moved: f64, flip: bool) {
        let Some(c) = &mut self.cand else { return };
        if moved != 0.0 {
            c.plane.d += moved;
            c.seed += c.plane.normal * moved;
        }
        if flip {
            c.plane = mm_core::Plane { normal: -c.plane.normal, d: -c.plane.d };
        }
        if let Some(mesh) = self.object.and_then(|id| cx.scene.get(id)).map(|o| o.mesh.clone()) {
            self.refresh_curve(&mesh);
            self.start(cx);
        }
    }

    fn poll(&mut self) {
        if let Some(r) = self.job.as_ref().and_then(|j| j.poll()) {
            self.job = None;
            match r {
                Ok(out) => self.preview = Some(out),
                Err(e) => {
                    self.preview = None;
                    self.error = Some(format!("{e:#}"));
                }
            }
        }
    }

    fn apply(&mut self, cx: &mut ToolCx) -> bool {
        let Some(out) = self.preview.take() else { return false };
        if Some(out.key) != self.key() {
            self.preview = Some(out);
            return false;
        }
        let Some(id) = self.object else { return false };
        if !cx.scene.get(id).map(|o| std::sync::Arc::ptr_eq(&o.mesh, &out.source)).unwrap_or(false) {
            self.cand = None;
            self.error = Some("The object changed; click the branch again.".into());
            return false;
        }
        cx.history.push("Trim branch", cx.scene);
        let Some(o) = cx.scene.get_mut(id) else { return false };
        o.set_mesh(out.mesh);
        for g in out.caps {
            let n = next_name(&o.group_names, "outlet");
            o.group_names.insert(g, n);
        }
        self.trims += 1;
        *cx.status = format!("Trimmed branch: {}", out.msg);
        self.cand = None;
        true
    }
}

impl Tool for TrimTool {
    fn name(&self) -> &'static str {
        "Trim branch"
    }

    fn panel(&mut self, ui: &mut egui::Ui, cx: &mut ToolCx) -> ToolOutcome {
        ui_util::section(ui, "Trim branch (square cut)");
        if self.object.is_none() {
            ui.label("No active object.");
            return if ui.button("Close").clicked() { ToolOutcome::Close } else { ToolOutcome::Continue };
        }
        ui.small("Click on a vessel where the inlet/outlet should be. The cut is perpendicular to the local axis and only removes the part beyond it (red preview). Press Enter to apply, then click the next branch.");
        ui.horizontal(|ui| {
            ui.label("End");
            let before = self.fill;
            ui.selectable_value(&mut self.fill, TrimFill::Remeshed, "Flat cap (remeshed)");
            ui.selectable_value(&mut self.fill, TrimFill::Flat, "Flat cap");
            ui.selectable_value(&mut self.fill, TrimFill::Open, "Leave open");
            if before != self.fill && self.cand.is_some() {
                self.start(cx);
            }
        });
        self.poll();
        if self.job.is_some() {
            ui.horizontal(|ui| {
                polymix_workspace::spinner(ui);
                ui.label("Computing cut…");
            });
        }
        if self.cand.is_some() {
            let r = self.cand.as_ref().unwrap().radius;
            let mut moved = 0.0;
            ui.horizontal(|ui| {
                ui.label("Move cut");
                if ui.button("◀ toward trunk").clicked() {
                    moved = -0.25 * r;
                }
                if ui.button("toward tip ▶").clicked() {
                    moved = 0.25 * r;
                }
            });
            let mut flip = false;
            ui.horizontal(|ui| {
                flip = ui.button("Flip (remove the other side)").clicked();
                if ui.button("Clear").clicked() {
                    self.cand = None;
                    self.preview = None;
                }
            });
            if moved != 0.0 || flip {
                self.shift(cx, moved, flip);
            }
            if let Some(c) = &self.cand {
                ui.label(format!("Local diameter ≈ {:.3}", 2.0 * c.radius));
            }
            if let Some(p) = &self.preview {
                ui.label(&p.msg);
            }
            let can = self.preview.as_ref().map(|p| Some(p.key) == self.key()).unwrap_or(false);
            if ui.add_enabled(can, egui::Button::new("Apply cut  (Enter)")).clicked() {
                self.apply(cx);
            }
        }
        if let Some(e) = &self.error {
            ui.colored_label(egui::Color32::from_rgb(240, 110, 90), e);
        }
        ui.add_space(8.0);
        if self.trims > 0 {
            ui.label(format!("{} branch(es) trimmed in this session", self.trims));
        }
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
        let Some((id, hit)) = cx.pick(pos, None) else { return };
        let _ = self.cut_at(cx, id, hit.point);
    }

    fn overlay(&mut self, ov: &mut Overlay, _cx: &mut ToolCx) {
        if let Some(c) = &self.cand {
            for s in &c.curve {
                ov.line(s[0], s[1], Vec4::new(1.0, 0.85, 0.2, 1.0));
            }
            if self.preview.is_none() {
                ov.cut_plane = Some((c.plane, Some((c.seed, c.radius * 6.0))));
                ov.cut_target = self.object;
            }
            let centre = c.plane.project(c.seed);
            ov.line(centre, centre + c.plane.normal * c.radius * 1.5, Vec4::new(1.0, 0.3, 0.2, 1.0));
        }
    }

    fn preview(&self) -> Option<(ObjectId, &Mesh)> {
        let p = self.preview.as_ref()?;
        if Some(p.key) == self.key() {
            Some((self.object?, &p.mesh))
        } else {
            None
        }
    }

    fn accept(&mut self, cx: &mut ToolCx) -> bool {
        self.apply(cx);
        false
    }

    fn cancel(&mut self, _cx: &mut ToolCx) -> bool {
        if self.cand.is_some() {
            self.cand = None;
            self.preview = None;
            return false;
        }
        true
    }

    fn busy(&self) -> Option<(String, f32)> {
        self.job.as_ref().map(|j| ("Trim".to_string(), j.progress()))
    }

    fn settings(&self) -> Option<Value> {
        Some(json!({"end": self.fill.key()}))
    }

    fn settings_schema(&self) -> Option<Value> {
        Some(json!({"type": "object", "properties": {"end": {"type": "string", "enum": TrimFill::KEYS.map(|(k, _)| k), "description": "How the cut end is closed: a flat cap remeshed to the surrounding edge length, a plain flat cap, or left open"}}}))
    }

    fn set_settings(&mut self, changes: &Value, cx: &mut ToolCx) -> Result<(), String> {
        let mut v = self.settings().unwrap_or_default();
        merge_settings(&mut v, changes)?;
        let key = v["end"].as_str().unwrap_or_default();
        let fill = TrimFill::KEYS.iter().find(|(k, _)| *k == key).map(|(_, f)| *f).ok_or_else(|| format!("end must be one of {:?}", TrimFill::KEYS.map(|(k, _)| k)))?;
        if fill != self.fill {
            self.fill = fill;
            if self.cand.is_some() {
                self.start(cx);
            }
        }
        Ok(())
    }

    fn state(&self, _scene: &Scene) -> Value {
        let current = self.preview.as_ref().filter(|p| Some(p.key) == self.key());
        json!({
            "object": self.object,
            "cut": self.cand.as_ref().map(|c| json!({
                "centre_mm": vec3(c.plane.project(c.seed).to_array(), 3),
                "toward_removed_side": vec3(c.plane.normal.to_array(), 4),
                "local_diameter_mm": round(2.0 * c.radius, 3),
                "outline_segments": c.curve.len(),
            })),
            "computing": self.job.is_some(),
            "preview": current.map(|p| p.msg.clone()),
            "error": self.error,
            "trimmed_in_this_session": self.trims,
            "next": if self.cand.is_none() { "cut_at a point on the branch where the inlet / outlet should be" } else if current.is_some() { "apply (or move / flip the cut first)" } else { "wait for the preview (app_wait)" },
        })
    }

    fn actions(&self) -> Vec<Action> {
        vec![
            Action::new("cut_at", "Place a cut square to the vessel's axis through the wall point nearest `at` (as clicking the branch does); only the part beyond it is removed, previewed", json!({"at": arg::point("A point in mm on or near the branch")})).required(&["at"]),
            Action::new("move", "Move the cut along the vessel", json!({"mm": arg::number("Distance in mm: positive toward the tip (removing less), negative toward the trunk")})).required(&["mm"]),
            Action::plain("flip", "Remove the other side of the cut instead"),
            Action::plain("clear", "Drop the cut"),
            Action::plain("apply", "Make the previewed cut (the cap becomes a patch named outlet_N); the tool stays open for the next branch"),
        ]
    }

    fn act(&mut self, action: &str, args: &Value, cx: &mut ToolCx) -> Result<String, String> {
        let a = Args(args);
        match action {
            "cut_at" => {
                let id = self.object.or(cx.scene.active).ok_or("no active object")?;
                let o = cx.scene.get_mut(id).ok_or("no object")?;
                let p = o.bvh().nearest(&o.mesh, DVec3::from_array(a.point("at")?), f64::INFINITY).ok_or("the object has no surface")?.point;
                let d = self.cut_at(cx, id, p)?;
                Ok(format!("Cut placed at a ⌀ {d:.2} mm cross-section; previewing"))
            }
            "move" | "flip" => {
                if self.cand.is_none() {
                    return Err("place a cut first (cut_at)".into());
                }
                self.shift(cx, if action == "move" { a.f64("mm")? } else { 0.0 }, action == "flip");
                Ok(if action == "move" { "Cut moved; previewing".into() } else { "Cut flipped; previewing".into() })
            }
            "clear" => {
                self.cand = None;
                self.preview = None;
                Ok("Cut cleared".into())
            }
            "apply" => {
                self.poll();
                if self.job.is_some() {
                    return Err("the preview is still being computed: app_wait, then apply".into());
                }
                if self.apply(cx) { Ok(cx.status.clone()) } else { Err(self.error.clone().unwrap_or_else(|| "nothing to apply: place a cut first (cut_at)".into())) }
            }
            _ => Err(format!("Trim branch has no action {action}")),
        }
    }
}

// ---------------------------------------------------------------------------
// Flow extensions
// ---------------------------------------------------------------------------

#[derive(Clone, PartialEq)]
struct FlowParams {
    diameters: f64,
    absolute: Option<f64>,
    circular: bool,
    cap: bool,
    selected: Vec<bool>,
}

struct FlowOut {
    source: std::sync::Arc<Mesh>,
    mesh: Mesh,
    /// Cap groups created for previously open ends (to be named outlet_N).
    new_caps: Vec<i32>,
    msg: String,
}

/// Extrude straight extensions from vessel ends (open loops or flat caps).
pub struct FlowExtensionTool {
    object: Option<ObjectId>,
    ends: Vec<VesselEnd>,
    params: FlowParams,
    job: Option<(FlowParams, crate::jobs::Job<FlowOut>)>,
    preview: Option<(FlowParams, FlowOut)>,
    error: Option<String>,
    /// The settings the last error came from (not built again until they change).
    failed: Option<FlowParams>,
    /// Settings changed in the panel, waiting to settle before they are built, and since when.
    changed: Option<(FlowParams, std::time::Instant)>,
}

impl FlowExtensionTool {
    pub fn new(cx: &mut ToolCx) -> Self {
        let object = cx.scene.active;
        let ends = cx.scene.active().map(|o| find_vessel_ends(&o.mesh)).unwrap_or_default();
        let n = ends.len();
        Self {
            object,
            ends,
            params: FlowParams { diameters: 5.0, absolute: None, circular: true, cap: true, selected: vec![false; n] },
            job: None,
            preview: None,
            error: None,
            failed: None,
            changed: None,
        }
    }

    /// Whether the current settings have a preview, a build under way or an error.
    fn up_to_date(&self) -> bool {
        let p = Some(&self.params);
        self.preview.as_ref().map(|x| &x.0) == p || self.job.as_ref().map(|x| &x.0) == p || self.failed.as_ref() == p
    }

    /// Collect a finished build, and build the panel's settings once they have stayed put for
    /// [`SETTLE`] (a slider drag would otherwise start a whole-mesh build every frame).
    fn update(&mut self, cx: &mut ToolCx) {
        self.update_at(std::time::Instant::now(), cx);
    }

    /// [`Self::update`] at `now`.
    fn update_at(&mut self, now: std::time::Instant, cx: &mut ToolCx) {
        if let Some((_, j)) = &self.job {
            if let Some(r) = j.poll() {
                let (p, _) = self.job.take().unwrap();
                match r {
                    Ok(out) => self.preview = Some((p, out)),
                    Err(e) => {
                        self.error = Some(format!("{e:#}"));
                        self.failed = Some(p);
                    }
                }
            }
        }
        if !self.params.selected.contains(&true) {
            // Nothing to build: drop the preview at once.
            self.changed = None;
            if self.preview.is_some() || self.job.is_some() {
                self.start(cx);
            }
            return;
        }
        if self.up_to_date() {
            self.changed = None;
            return;
        }
        match &self.changed {
            Some((p, since)) if *p == self.params && now.saturating_duration_since(*since) >= SETTLE => self.start(cx),
            Some((p, since)) if *p == self.params => cx.egui.request_repaint_after(SETTLE - now.saturating_duration_since(*since)),
            _ => {
                self.changed = Some((self.params.clone(), now));
                cx.egui.request_repaint_after(SETTLE);
            }
        }
    }

    fn start(&mut self, cx: &mut ToolCx) {
        self.changed = None;
        let Some(o) = self.object.and_then(|id| cx.scene.get(id)) else { return };
        let src = o.mesh.clone();
        let source = src.clone();
        let ends: Vec<VesselEnd> = self.ends.iter().zip(&self.params.selected).filter(|(_, s)| **s).map(|(e, _)| e.clone()).collect();
        if ends.is_empty() {
            self.preview = None;
            // A build for ends no longer chosen would only keep the tool busy.
            if let Some((_, j)) = self.job.take() {
                j.cancel();
            }
            return;
        }
        let p = self.params.clone();
        let p2 = p.clone();
        if let Some((_, j)) = &self.job {
            j.cancel();
        }
        self.error = None;
        self.failed = None;
        self.job = Some((
            p,
            crate::jobs::Job::spawn("Flow extensions", cx.egui.clone(), move |progress| {
                let mut m = (*src).clone();
                let n = ends.len();
                let mut new_caps = Vec::new();
                for (k, e) in ends.iter().enumerate() {
                    if !progress(k as f32 / n as f32) {
                        anyhow::bail!("cancelled");
                    }
                    let keep_group = match e.kind {
                        EndKind::Cap(g) => {
                            let tris: Vec<u32> = m.triangle_ids().filter(|&t| m.tri_group(t) == g).collect();
                            m.remove_triangles(tris);
                            Some(g)
                        }
                        EndKind::Open => None,
                    };
                    let lp = nearest_loop(&m, e.center).ok_or_else(|| anyhow::anyhow!("end {} has no open boundary", k + 1))?;
                    let edge = {
                        let mut s = 0.0;
                        for i in 0..lp.len() {
                            s += m.vertex(lp[i]).distance(m.vertex(lp[(i + 1) % lp.len()]));
                        }
                        s / lp.len() as f64
                    };
                    let length = p2.absolute.unwrap_or(p2.diameters * e.diameter);
                    let segments = ((length / (2.0 * edge)).ceil() as usize).clamp(2, 150);
                    let params = mm_edit::FlowExtensionParams {
                        length: match p2.absolute {
                            Some(a) => mm_edit::ExtensionLength::Absolute(a),
                            None => mm_edit::ExtensionLength::Diameters(p2.diameters),
                        },
                        segments,
                        make_circular: p2.circular,
                        cap: p2.cap,
                        ..Default::default()
                    };
                    let r = mm_edit::extrude_boundary_loop(&mut m, &lp, &params)?;
                    for &t in &r.wall_triangles {
                        m.set_tri_group(t, 0);
                    }
                    match (keep_group, r.cap_group) {
                        (Some(g), Some(_)) => {
                            for &t in &r.cap_triangles {
                                m.set_tri_group(t, g);
                            }
                        }
                        (None, Some(g)) => new_caps.push(g),
                        _ => {}
                    }
                }
                m.bump_shape_version();
                Ok(FlowOut { source, mesh: m, new_caps, msg: format!("extended {n} end(s)") })
            }),
        ));
    }
}

impl Tool for FlowExtensionTool {
    fn name(&self) -> &'static str {
        "Flow extensions"
    }

    fn panel(&mut self, ui: &mut egui::Ui, cx: &mut ToolCx) -> ToolOutcome {
        ui_util::section(ui, "Flow extensions");
        if self.object.is_none() {
            ui.label("No active object.");
            return if ui.button("Close").clicked() { ToolOutcome::Close } else { ToolOutcome::Continue };
        }
        ui.small("Click end markers (or tick them below) to choose inlets/outlets. Ends can be open cuts or flat caps; caps keep their patch names.");
        let before = self.params.clone();
        if self.ends.is_empty() {
            ui.label("No open ends or flat caps found. Use Trim branch first.");
        }
        for (i, e) in self.ends.iter().enumerate() {
            let kind = match e.kind {
                EndKind::Open if e.planar => "open end".to_string(),
                EndKind::Open => "hole (fill it in the Inspector)".to_string(),
                EndKind::Cap(g) => cx.scene.get(self.object.unwrap()).map(|o| crate::tools::patches::patch_name(o, g)).unwrap_or_default(),
            };
            ui.checkbox(&mut self.params.selected[i], format!("{}: {kind}, ⌀ {:.2}", i + 1, e.diameter));
        }
        ui.horizontal(|ui| {
            if ui.small_button("All").clicked() {
                self.params.selected.iter_mut().for_each(|s| *s = true);
            }
            if ui.small_button("None").clicked() {
                self.params.selected.iter_mut().for_each(|s| *s = false);
            }
        });
        ui_util::section(ui, "Extension");
        let mut abs = self.params.absolute.is_some();
        ui.horizontal(|ui| {
            ui.selectable_value(&mut abs, false, "Diameters");
            ui.selectable_value(&mut abs, true, "Absolute");
        });
        if abs {
            let mut a = self.params.absolute.unwrap_or(10.0);
            ui.add(egui::Slider::new(&mut a, 0.1..=200.0).logarithmic(true).text("length"));
            self.params.absolute = Some(a);
        } else {
            self.params.absolute = None;
            ui.add(egui::Slider::new(&mut self.params.diameters, 0.5..=20.0).text("× diameter"));
        }
        ui.checkbox(&mut self.params.circular, "Blend to circular cross-section");
        ui.checkbox(&mut self.params.cap, "Cap the ends");
        if self.params != before {
            // Built once the settings settle (see `update`).
            self.changed = Some((self.params.clone(), std::time::Instant::now()));
        }
        self.update(cx);
        if self.job.is_some() || self.changed.is_some() {
            ui.horizontal(|ui| {
                polymix_workspace::spinner(ui);
                ui.label("Building…");
            });
        }
        if let Some(e) = &self.error {
            ui.colored_label(egui::Color32::from_rgb(240, 110, 90), e);
        }
        let ready = self.preview.as_ref().map(|(p, _)| *p == self.params).unwrap_or(false);
        let (accept, cancel) = ui_util::accept_cancel(ui, ready);
        if accept && self.accept(cx) {
            return ToolOutcome::Close;
        }
        if cancel {
            return ToolOutcome::Close;
        }
        ToolOutcome::Continue
    }

    fn pointer(&mut self, input: &PointerInput, cx: &mut ToolCx) {
        if !input.clicked {
            return;
        }
        let Some(pos) = input.pos else { return };
        if let Some(i) = end_under_cursor(cx, &self.ends, pos) {
            self.params.selected[i] = !self.params.selected[i];
            self.start(cx);
        }
    }

    fn paint(&mut self, painter: &egui::Painter, cx: &mut ToolCx) {
        if self.preview.is_none() || self.job.is_some() || self.changed.is_some() {
            draw_end_markers(painter, cx, &self.ends, &self.params.selected);
        }
    }

    fn preview(&self) -> Option<(ObjectId, &Mesh)> {
        let (p, out) = self.preview.as_ref()?;
        if *p == self.params {
            Some((self.object?, &out.mesh))
        } else {
            None
        }
    }

    fn accept(&mut self, cx: &mut ToolCx) -> bool {
        if self.job.is_some() {
            return false;
        }
        if self.changed.is_some() && !self.up_to_date() {
            // Settings still settling: build them now, and accept once built.
            self.start(cx);
            return false;
        }
        let Some((p, out)) = self.preview.take() else { return false };
        if p != self.params {
            self.preview = Some((p, out));
            return false;
        }
        let Some(id) = self.object else { return true };
        if !cx.scene.get(id).map(|o| std::sync::Arc::ptr_eq(&o.mesh, &out.source)).unwrap_or(false) {
            *cx.status = "Flow extensions cancelled: the object changed.".into();
            return true;
        }
        cx.history.push("Flow extensions", cx.scene);
        let Some(o) = cx.scene.get_mut(id) else { return true };
        o.set_mesh(out.mesh);
        for g in out.new_caps {
            let n = next_name(&o.group_names, "outlet");
            o.group_names.insert(g, n);
        }
        *cx.status = format!("Flow extensions: {}", out.msg);
        true
    }

    fn busy(&self) -> Option<(String, f32)> {
        self.job.as_ref().map(|(_, j)| ("Flow extensions".to_string(), j.progress()))
    }

    fn pending(&self) -> bool {
        // Settling, or chosen ends whose extensions are not built yet.
        self.job.is_none() && self.params.selected.contains(&true) && !self.up_to_date()
    }

    fn settings(&self) -> Option<Value> {
        let p = &self.params;
        Some(json!({"length_in_diameters": p.diameters, "length_mm": p.absolute, "circular": p.circular, "cap": p.cap}))
    }

    fn settings_schema(&self) -> Option<Value> {
        Some(json!({"type": "object", "properties": {
            "length_in_diameters": {"type": "number", "description": "Extension length as a multiple of each end's diameter (0.5..20), used unless length_mm is set"},
            "length_mm": {"type": ["number", "null"], "description": "Extension length in mm for every end (0.1..200); null to use length_in_diameters"},
            "circular": {"type": "boolean", "description": "Blend each extension to a circular cross-section"},
            "cap": {"type": "boolean", "description": "Cap the extensions' ends"},
        }}))
    }

    fn set_settings(&mut self, changes: &Value, cx: &mut ToolCx) -> Result<(), String> {
        let mut v = self.settings().unwrap_or_default();
        merge_settings(&mut v, changes)?;
        let p = &mut self.params;
        p.diameters = v["length_in_diameters"].as_f64().ok_or("length_in_diameters must be a number")?.clamp(0.5, 20.0);
        p.absolute = match &v["length_mm"] {
            Value::Null => None,
            x => Some(x.as_f64().ok_or("length_mm must be a number or null")?.clamp(0.1, 200.0)),
        };
        p.circular = v["circular"].as_bool().ok_or("circular must be true or false")?;
        p.cap = v["cap"].as_bool().ok_or("cap must be true or false")?;
        self.start(cx);
        Ok(())
    }

    fn state(&self, _scene: &Scene) -> Value {
        let current = self.preview.as_ref().filter(|(p, _)| *p == self.params);
        json!({
            "object": self.object,
            "ends": ends_state(&self.ends, |i| self.params.selected[i]),
            "computing": self.job.is_some(),
            "preview": current.map(|(_, out)| out.msg.clone()),
            "error": self.error,
            "next": if !self.params.selected.contains(&true) { "select the ends to extend" } else if current.is_some() { "accept (mesh_tool accept) to keep the extensions" } else { "wait for the preview (app_wait)" },
        })
    }

    fn actions(&self) -> Vec<Action> {
        vec![Action::new(
            "select",
            "Choose the ends to extend (numbers from the state's ends, as the markers show): replaces the choice unless `add`",
            json!({"ends": ends_arg("End numbers (from 1)"), "add": arg::boolean("Add to the ends already chosen")}),
        )
        .required(&["ends"])]
    }

    fn act(&mut self, action: &str, args: &Value, cx: &mut ToolCx) -> Result<String, String> {
        match action {
            "select" => {
                let picked = end_numbers(args, self.ends.len())?;
                if !Args(args).opt_bool("add").unwrap_or(false) {
                    self.params.selected.iter_mut().for_each(|s| *s = false);
                }
                for i in &picked {
                    self.params.selected[*i] = true;
                }
                self.start(cx);
                let n = self.params.selected.iter().filter(|s| **s).count();
                Ok(format!("{n} end(s) chosen; previewing"))
            }
            _ => Err(format!("Flow extensions has no action {action}")),
        }
    }
}

// ---------------------------------------------------------------------------
// Cap open ends
// ---------------------------------------------------------------------------

/// Cap every open boundary loop with a flat (remeshed) cap; each cap becomes a patch.
pub struct CapTool {
    object: Option<ObjectId>,
    ends: Vec<VesselEnd>,
    remeshed: bool,
    message: Option<String>,
    /// The panel's Cap All, running in the background (seconds on a large mesh), and the mesh
    /// it caps a copy of.
    job: Option<(std::sync::Arc<Mesh>, crate::jobs::Job<Capped>)>,
}

/// Cap All's result.
struct Capped {
    mesh: Mesh,
    /// The caps' new face groups, in the order they were made (named outlet_N in that order).
    groups: Vec<i32>,
    message: String,
    /// The open ends left.
    ends: Vec<VesselEnd>,
}

/// Cap every open boundary of a copy of `src`, one loop at a time so each gets its own plane
/// fit and group. `progress` gets the fraction of loops done and may return false to stop
/// (then `None`).
fn cap_loops(src: &Mesh, remeshed: bool, progress: &dyn Fn(f32) -> bool) -> Option<Capped> {
    let target = src.average_edge_length();
    let fill = if remeshed { mm_edit::CutFill::FlatRemeshed { target_edge_length: target } } else { mm_edit::CutFill::Flat };
    let loops: Vec<Vec<u32>> = mm_core::topology::boundary_loops(src).into_iter().map(|l| l.vertices).collect();
    let mut m = src.clone();
    let mut capped = 0;
    let mut failed = 0;
    let mut groups = Vec::new();
    for (k, lp) in loops.iter().enumerate() {
        if !progress(k as f32 / loops.len() as f32) {
            return None;
        }
        match mm_edit::cap_planar_loops(&mut m, std::slice::from_ref(lp), None, fill) {
            Ok(r) if r.failed_loops.is_empty() => {
                capped += 1;
                groups.extend(r.groups);
            }
            _ => failed += 1,
        }
    }
    let message = format!("Capped {capped} loop(s){}", if failed > 0 { format!(", {failed} could not be capped flat") } else { String::new() });
    let ends = find_vessel_ends(&m).into_iter().filter(|e| matches!(e.kind, EndKind::Open)).collect();
    Some(Capped { mesh: m, groups, message, ends })
}

impl CapTool {
    pub fn new(cx: &mut ToolCx) -> Self {
        let ends = cx.scene.active().map(|o| find_vessel_ends(&o.mesh)).unwrap_or_default().into_iter().filter(|e| matches!(e.kind, EndKind::Open)).collect();
        Self { object: cx.scene.active, ends, remeshed: true, message: None, job: None }
    }
}

impl Tool for CapTool {
    fn name(&self) -> &'static str {
        "Cap open ends"
    }

    fn panel(&mut self, ui: &mut egui::Ui, cx: &mut ToolCx) -> ToolOutcome {
        ui_util::section(ui, "Cap open ends");
        self.poll(cx);
        let Some(id) = self.object.filter(|id| cx.scene.get(*id).is_some()) else {
            ui.label("No active object.");
            return if ui.button("Close").clicked() { ToolOutcome::Close } else { ToolOutcome::Continue };
        };
        ui.label(format!("{} open boundary loop(s)", self.ends.len()));
        ui.checkbox(&mut self.remeshed, "Remesh caps to match the surrounding edge length");
        ui.small("Each cap becomes its own named patch (outlet_N). Non-planar holes are better filled with the Inspector's smooth fill.");
        if ui.add_enabled(!self.ends.is_empty() && self.job.is_none(), egui::Button::new("Cap all")).clicked() {
            self.start(cx, id);
        }
        if let Some((_, j)) = &self.job {
            ui.horizontal(|ui| {
                polymix_workspace::spinner(ui);
                ui.add(egui::ProgressBar::new(j.progress()).show_percentage());
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

    fn paint(&mut self, painter: &egui::Painter, cx: &mut ToolCx) {
        draw_end_markers(painter, cx, &self.ends, &[]);
    }

    fn busy(&self) -> Option<(String, f32)> {
        self.job.as_ref().map(|(_, j)| ("Capping".to_string(), j.progress()))
    }

    fn settings(&self) -> Option<Value> {
        Some(json!({"remesh_caps": self.remeshed}))
    }

    fn settings_schema(&self) -> Option<Value> {
        Some(json!({"type": "object", "properties": {"remesh_caps": {"type": "boolean", "description": "Remesh the caps to match the surrounding edge length"}}}))
    }

    fn set_settings(&mut self, changes: &Value, _cx: &mut ToolCx) -> Result<(), String> {
        let mut v = self.settings().unwrap_or_default();
        merge_settings(&mut v, changes)?;
        self.remeshed = v["remesh_caps"].as_bool().ok_or("remesh_caps must be true or false")?;
        Ok(())
    }

    fn state(&self, _scene: &Scene) -> Value {
        json!({"object": self.object, "open_ends": ends_state(&self.ends, |_| false), "message": self.message})
    }

    fn actions(&self) -> Vec<Action> {
        vec![Action::plain("cap_all", "Cap every open boundary flat; each cap becomes its own patch named outlet_N")]
    }

    fn act(&mut self, action: &str, _args: &Value, cx: &mut ToolCx) -> Result<String, String> {
        match action {
            "cap_all" => {
                let id = self.object.filter(|id| cx.scene.get(*id).is_some()).ok_or("no active object")?;
                if self.job.is_some() {
                    return Err("Cap all is running: app_wait".into());
                }
                if self.ends.is_empty() {
                    return Err("there are no open ends to cap".into());
                }
                Ok(self.cap_all(cx, id))
            }
            _ => Err(format!("Cap open ends has no action {action}")),
        }
    }
}

impl CapTool {
    /// Cap All now (agents).
    fn cap_all(&mut self, cx: &mut ToolCx, id: ObjectId) -> String {
        cx.history.push("Cap open ends", cx.scene);
        let o = cx.scene.get_mut(id).unwrap();
        let capped = cap_loops(&o.mesh, self.remeshed, &|_| true).expect("not cancelled");
        self.apply(o, capped, cx.status);
        self.message.clone().unwrap_or_default()
    }

    /// The panel's Cap All, in the background.
    fn start(&mut self, cx: &mut ToolCx, id: ObjectId) {
        let Some(o) = cx.scene.get(id) else { return };
        let (source, src) = (o.mesh.clone(), o.mesh.clone());
        let remeshed = self.remeshed;
        self.job = Some((source, crate::jobs::Job::spawn("Cap open ends", cx.egui.clone(), move |progress| cap_loops(&src, remeshed, progress).ok_or_else(|| anyhow::anyhow!("cancelled")))));
    }

    /// Keep a finished Cap All, if the object still holds the mesh it capped.
    fn poll(&mut self, cx: &mut ToolCx) {
        let Some(r) = self.job.as_ref().and_then(|(_, j)| j.poll()) else { return };
        let (source, _) = self.job.take().unwrap();
        let id = self.object.filter(|id| cx.scene.get(*id).is_some_and(|o| std::sync::Arc::ptr_eq(&o.mesh, &source)));
        match (r, id) {
            (Ok(capped), Some(id)) => {
                cx.history.push("Cap open ends", cx.scene);
                let o = cx.scene.get_mut(id).unwrap();
                self.apply(o, capped, cx.status);
            }
            (Ok(_), None) => self.message = Some("Nothing was capped: the object changed meanwhile.".into()),
            (Err(e), _) => self.message = Some(format!("Capping failed: {e:#}")),
        }
    }

    fn apply(&mut self, o: &mut crate::scene::SceneObject, c: Capped, status: &mut String) {
        o.set_mesh(c.mesh);
        for g in c.groups {
            let n = next_name(&o.group_names, "outlet");
            o.group_names.insert(g, n);
        }
        self.message = Some(c.message);
        *status = self.message.clone().unwrap();
        self.ends = c.ends;
    }
}

// ---------------------------------------------------------------------------
// Bridge / join two open boundaries
// ---------------------------------------------------------------------------

/// Join two open boundary loops with a tube (Meshmixer Bridge / Join), lofted like Erase &
/// Fill's Loft: the first marker picked is End 1. The tube is previewed live once two ends
/// are picked (built in the background a moment after the settings stop changing); Join
/// keeps it.
pub struct BridgeTool {
    object: Option<ObjectId>,
    ends: Vec<VesselEnd>,
    picked: Vec<usize>,
    loft: ops::LoftUi,
    refine: f64,
    bulge: f64,
    job: Option<(BridgeKey, crate::jobs::Job<BridgeOut>)>,
    preview: Option<(BridgeKey, BridgeOut)>,
    /// Settings waiting to settle before they are previewed, and since when.
    changed: Option<(BridgeKey, std::time::Instant)>,
    error: Option<(BridgeKey, String)>,
}

/// Bridge's settings as agents set them.
#[derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
struct BridgeSettings {
    /// The tube's shape and how it meets each end (as Erase & Fill's Loft).
    loft: ops::LoftUi,
    /// 0..100: tube density; 50 matches the ends' edge length.
    refine: f64,
    /// -100..100: pushes the middle of the tube out (+) or in (−).
    bulge: f64,
}

/// What a preview was made from.
#[derive(Clone, Copy, PartialEq, Debug)]
struct BridgeKey {
    picked: [usize; 2],
    loft: ops::LoftUi,
    refine: f64,
    bulge: f64,
}

struct BridgeOut {
    source: std::sync::Arc<Mesh>,
    mesh: Mesh,
    message: String,
}

/// How long the settings must stay put before they are previewed (slider drags).
const SETTLE: std::time::Duration = std::time::Duration::from_millis(120);

impl BridgeTool {
    pub fn new(cx: &mut ToolCx) -> Self {
        let ends = cx.scene.active().map(|o| find_vessel_ends(&o.mesh)).unwrap_or_default().into_iter().filter(|e| matches!(e.kind, EndKind::Open)).collect();
        Self { object: cx.scene.active, ends, picked: Vec::new(), loft: ops::LoftUi::default(), refine: 50.0, bulge: 0.0, job: None, preview: None, changed: None, error: None }
    }

    fn key(&self) -> Option<BridgeKey> {
        (self.picked.len() == 2).then(|| BridgeKey { picked: [self.picked[0], self.picked[1]], loft: self.loft, refine: self.refine, bulge: self.bulge })
    }

    /// The tube for `key` on `source` (in the background, or at once for scripts).
    fn build(source: &Mesh, ends: [DVec3; 2], key: &BridgeKey) -> anyhow::Result<(Mesh, String)> {
        let mut m = source.clone();
        let (Some(a), Some(b)) = (nearest_loop(&m, ends[0]), nearest_loop(&m, ends[1])) else {
            anyhow::bail!("boundary not found");
        };
        if a == b {
            anyhow::bail!("pick two different boundaries");
        }
        let mean_edge = |l: &[u32]| (0..l.len()).map(|i| m.vertex(l[i]).distance(m.vertex(l[(i + 1) % l.len()]))).sum::<f64>() / l.len() as f64;
        let h = ops::refined_edge_length(0.5 * (mean_edge(&a) + mean_edge(&b)), key.refine);
        let r = ops::run_loft(&mut m, &a, &b, &key.loft, h, key.bulge, None)?;
        let message = format!("Bridged with {} triangles{}", r.triangles.len(), ops::crossing_warning(&m, &r.triangles));
        m.bump_shape_version();
        Ok((m, message))
    }

    /// Keep the preview up to date: collect a finished build, and start one for settings
    /// that have settled.
    fn update(&mut self, cx: &mut ToolCx) {
        if let Some((k, j)) = &self.job {
            if let Some(r) = j.poll() {
                let k = *k;
                self.job = None;
                match r {
                    Ok(out) => {
                        self.preview = Some((k, out));
                        self.error = None;
                    }
                    Err(e) => self.error = Some((k, format!("{e:#}"))),
                }
            }
        }
        let Some(key) = self.key() else {
            self.preview = None;
            self.changed = None;
            if let Some((_, j)) = self.job.take() {
                j.cancel();
            }
            return;
        };
        let current = |k: Option<&BridgeKey>| k == Some(&key);
        if current(self.preview.as_ref().map(|p| &p.0)) || current(self.job.as_ref().map(|j| &j.0)) || current(self.error.as_ref().map(|e| &e.0)) {
            self.changed = None;
            return;
        }
        let now = std::time::Instant::now();
        match self.changed {
            Some((k, since)) if k == key && now - since >= SETTLE => self.start(key, cx),
            Some((k, since)) if k == key => cx.egui.request_repaint_after(SETTLE - (now - since)),
            _ => {
                self.changed = Some((key, now));
                cx.egui.request_repaint_after(SETTLE);
            }
        }
    }

    fn start(&mut self, key: BridgeKey, cx: &mut ToolCx) {
        self.changed = None;
        let Some(o) = self.object.and_then(|id| cx.scene.get(id)) else { return };
        if let Some((_, j)) = self.job.take() {
            j.cancel();
        }
        let source = o.mesh.clone();
        let ends = [self.ends[key.picked[0]].center, self.ends[key.picked[1]].center];
        self.job = Some((
            key,
            crate::jobs::Job::spawn("Bridge", cx.egui.clone(), move |_| {
                let (mesh, message) = Self::build(&source, ends, &key)?;
                Ok(BridgeOut { source, mesh, message })
            }),
        ));
    }

    /// Keep the tube: the preview if it is up to date, else built now. Returns the status.
    fn join(&mut self, id: ObjectId, cx: &mut ToolCx) -> Result<String, String> {
        let key = self.key().ok_or("pick two ends")?;
        let o = cx.scene.get(id).ok_or("no active object")?;
        let (mesh, message) = match self.preview.take() {
            Some((k, out)) if k == key && std::sync::Arc::ptr_eq(&out.source, &o.mesh) => (out.mesh, out.message),
            _ => {
                let ends = [self.ends[key.picked[0]].center, self.ends[key.picked[1]].center];
                Self::build(&o.mesh, ends, &key).map_err(|e| format!("{e:#}"))?
            }
        };
        if let Some((_, j)) = self.job.take() {
            j.cancel();
        }
        cx.history.push("Bridge", cx.scene);
        cx.scene.get_mut(id).unwrap().set_mesh(mesh);
        self.ends = find_vessel_ends(&cx.scene.get(id).unwrap().mesh).into_iter().filter(|e| matches!(e.kind, EndKind::Open)).collect();
        self.picked.clear();
        self.error = None;
        Ok(message)
    }
}

impl Tool for BridgeTool {
    fn name(&self) -> &'static str {
        "Bridge"
    }

    fn script(&mut self, cmd: &str, cx: &mut ToolCx) {
        // `pick:<marker number>`, `join`, and the loft's settings (see `LoftUi::script`).
        let (key, val) = cmd.split_once(':').unwrap_or((cmd, ""));
        match key {
            "pick" => {
                if let Some(i) = val.parse::<usize>().ok().and_then(|k| k.checked_sub(1)).filter(|&i| i < self.ends.len() && !self.picked.contains(&i)) {
                    if self.picked.len() == 2 {
                        self.picked.remove(0);
                    }
                    self.picked.push(i);
                }
            }
            "join" if self.picked.len() == 2 => {
                if let Some(id) = self.object {
                    match self.join(id, cx) {
                        Ok(msg) => *cx.status = msg,
                        Err(e) => *cx.status = format!("Bridge: {e}"),
                    }
                }
            }
            "refine" => self.refine = val.parse().unwrap_or(self.refine),
            "bulge" => self.bulge = val.parse().unwrap_or(self.bulge),
            _ => {
                self.loft.script(key, val);
            }
        }
        self.update(cx);
    }

    fn preview(&self) -> Option<(ObjectId, &Mesh)> {
        let (k, out) = self.preview.as_ref()?;
        (Some(*k) == self.key()).then_some((self.object?, &out.mesh))
    }

    fn accept(&mut self, cx: &mut ToolCx) -> bool {
        if self.job.is_some() {
            return false;
        }
        if let (Some(id), true) = (self.object, self.picked.len() == 2) {
            match self.join(id, cx) {
                Ok(msg) => *cx.status = msg,
                Err(e) => *cx.status = format!("Bridge: {e}"),
            }
        }
        true
    }

    fn busy(&self) -> Option<(String, f32)> {
        self.job.as_ref().map(|(_, j)| ("Bridge".to_string(), j.progress()))
    }

    fn pending(&self) -> bool {
        let Some(key) = self.key() else { return false };
        let done = |k: Option<&BridgeKey>| k == Some(&key);
        self.job.is_none() && !done(self.preview.as_ref().map(|p| &p.0)) && !done(self.error.as_ref().map(|e| &e.0))
    }

    fn settings(&self) -> Option<Value> {
        serde_json::to_value(BridgeSettings { loft: self.loft, refine: self.refine, bulge: self.bulge }).ok()
    }

    fn settings_schema(&self) -> Option<Value> {
        Some(schema_of::<BridgeSettings>())
    }

    fn set_settings(&mut self, changes: &Value, cx: &mut ToolCx) -> Result<(), String> {
        let mut v = self.settings().unwrap_or_default();
        merge_settings(&mut v, changes)?;
        let b: BridgeSettings = serde_json::from_value(v).map_err(|e| format!("Bridge: {e}"))?;
        (self.loft, self.refine, self.bulge) = (b.loft, b.refine.clamp(0.0, 100.0), b.bulge.clamp(-100.0, 100.0));
        self.update(cx);
        Ok(())
    }

    fn state(&self, _scene: &Scene) -> Value {
        let key = self.key();
        json!({
            "object": self.object,
            "open_ends": ends_state(&self.ends, |i| self.picked.contains(&i)),
            "picked": self.picked.iter().map(|i| i + 1).collect::<Vec<_>>(),
            "computing": self.job.is_some(),
            "preview": self.preview.as_ref().filter(|p| Some(p.0) == key).map(|p| p.1.message.clone()),
            "error": self.error.as_ref().filter(|e| Some(e.0) == key).map(|e| e.1.clone()),
            "next": if self.picked.len() < 2 { "pick two open ends (End 1 first)" } else { "join, once previewed (app_wait)" },
        })
    }

    fn actions(&self) -> Vec<Action> {
        vec![
            Action::new("pick", "Choose the two open ends to connect (numbers from the state's open_ends); the first is End 1", json!({"ends": ends_arg("Two end numbers (from 1)")})).required(&["ends"]),
            Action::plain("join", "Keep the tube (the preview, or built now)"),
        ]
    }

    fn act(&mut self, action: &str, args: &Value, cx: &mut ToolCx) -> Result<String, String> {
        match action {
            "pick" => {
                let picked = end_numbers(args, self.ends.len())?;
                if picked.len() != 2 || picked[0] == picked[1] {
                    return Err("pick two different ends".into());
                }
                self.picked = picked;
                self.update(cx);
                Ok(format!("Ends {} and {} picked; previewing", self.picked[0] + 1, self.picked[1] + 1))
            }
            "join" => {
                let id = self.object.filter(|id| cx.scene.get(*id).is_some()).ok_or("no active object")?;
                let msg = self.join(id, cx)?;
                *cx.status = msg.clone();
                Ok(msg)
            }
            _ => Err(format!("Bridge has no action {action}")),
        }
    }

    fn panel(&mut self, ui: &mut egui::Ui, cx: &mut ToolCx) -> ToolOutcome {
        ui_util::section(ui, "Bridge boundaries");
        let Some(id) = self.object.filter(|id| cx.scene.get(*id).is_some()) else {
            ui.label("No active object.");
            return if ui.button("Close").clicked() { ToolOutcome::Close } else { ToolOutcome::Continue };
        };
        ui.small("Click two open-boundary markers to connect them with a tube: the first is End 1. The tube is previewed as you change the settings.");
        self.loft.ui(ui, &mut self.refine, &mut self.bulge);
        self.update(cx);
        ui.label(format!("Picked: {:?}", self.picked.iter().map(|i| i + 1).collect::<Vec<_>>()));
        let key = self.key();
        let ready = key.is_some() && self.preview.as_ref().map(|p| p.0) == key;
        ui.horizontal(|ui| {
            if ui.add_enabled(ready, egui::Button::new("Join")).on_hover_text("Keep the previewed tube").clicked() {
                match self.join(id, cx) {
                    Ok(msg) => *cx.status = msg,
                    Err(e) => self.error = key.map(|k| (k, e)),
                }
            }
            if key.is_some() && !ready && self.error.as_ref().map(|e| e.0) != key {
                polymix_workspace::spinner(ui);
                ui.weak("Previewing…");
            }
        });
        if let Some(message) = self.preview.as_ref().filter(|p| Some(p.0) == key).map(|p| p.1.message.clone()) {
            ui.weak(message);
        }
        if let Some((_, e)) = self.error.as_ref().filter(|e| Some(e.0) == key) {
            ui.colored_label(egui::Color32::from_rgb(240, 110, 90), e);
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
        if let Some(i) = end_under_cursor(cx, &self.ends, pos) {
            if let Some(k) = self.picked.iter().position(|&x| x == i) {
                self.picked.remove(k);
            } else {
                if self.picked.len() == 2 {
                    self.picked.remove(0);
                }
                self.picked.push(i);
            }
        }
    }

    fn paint(&mut self, painter: &egui::Painter, cx: &mut ToolCx) {
        let sel: Vec<bool> = (0..self.ends.len()).map(|i| self.picked.contains(&i)).collect();
        draw_end_markers(painter, cx, &self.ends, &sel);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_core::primitives;
    use std::time::{Duration, Instant};

    fn vessel_scene() -> Scene {
        let mut scene = Scene::default();
        let path: Vec<DVec3> = (0..30).map(|i| DVec3::new(0.0, (i as f64 * 0.1).sin(), i as f64 * 0.5)).collect();
        scene.add("vessel", primitives::tube_along(&path, &[1.0; 30], 32));
        scene
    }

    #[test]
    fn flow_extensions_build_a_slider_drag_once_it_settles() {
        let mut scene = vessel_scene();
        let (mut history, mut camera, mut status) = (crate::scene::History::default(), crate::camera::Camera::default(), String::new());
        let ctx = egui::Context::default();
        let vp = Viewport { rect: egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0)) };
        let mut cx = ToolCx { scene: &mut scene, history: &mut history, camera: &mut camera, status: &mut status, egui: &ctx, vp, brush_scroll: crate::navigation::BrushScroll::Shift };
        let mut tool = FlowExtensionTool::new(&mut cx);
        assert_eq!(tool.ends.len(), 2);
        // Tick an end and drag the length, a frame every 10 ms: no build starts meanwhile. (The
        // test sets the clock, so a slow machine cannot make the drag look settled.)
        let start = Instant::now();
        tool.params.selected[0] = true;
        for k in 0..20 {
            tool.params.diameters = 2.0 + 0.1 * k as f64;
            tool.update_at(start + Duration::from_millis(10 * k), &mut cx);
            assert!(tool.job.is_none(), "frame {k}");
            assert!(tool.pending());
        }
        let last = start + Duration::from_millis(10 * 19);
        tool.update_at(last + SETTLE - Duration::from_millis(1), &mut cx);
        assert!(tool.job.is_none() && tool.pending());
        // It settles: one build, of the last value, and the preview.
        let t0 = Instant::now();
        let mut builds = 0;
        while tool.preview.as_ref().is_none_or(|(p, _)| *p != tool.params) && t0.elapsed() < Duration::from_secs(30) {
            let had = tool.job.is_some();
            tool.update_at(last + SETTLE + t0.elapsed(), &mut cx);
            builds += (!had && tool.job.is_some()) as usize;
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(builds, 1);
        assert!(!tool.pending() && tool.busy().is_none());
        assert!(tool.preview().is_some(), "{:?}", tool.error);
        // Marker clicks and agents still build at once; no end chosen drops the build.
        tool.params.selected[1] = true;
        tool.start(&mut cx);
        assert!(tool.job.is_some());
        tool.params.selected = vec![false; 2];
        tool.update(&mut cx);
        assert!(tool.job.is_none() && tool.preview.is_none() && !tool.pending());
    }

    #[test]
    fn cap_all_in_the_background_caps_as_the_agents_do() {
        let ctx = egui::Context::default();
        let vp = Viewport { rect: egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0)) };
        let (mut camera, mut status) = (crate::camera::Camera::default(), String::new());
        // At once, as the agents' cap_all does it.
        let (mut direct, mut history) = (vessel_scene(), crate::scene::History::default());
        let mut cx = ToolCx { scene: &mut direct, history: &mut history, camera: &mut camera, status: &mut status, egui: &ctx, vp, brush_scroll: crate::navigation::BrushScroll::Shift };
        let mut tool = CapTool::new(&mut cx);
        let id = cx.scene.active.unwrap();
        let reply = tool.act("cap_all", &json!({}), &mut cx).unwrap();
        let direct_ends = tool.ends.len();
        // In the background, as the panel's button does it.
        let (mut scene, mut history2, mut status2) = (vessel_scene(), crate::scene::History::default(), String::new());
        let mut cx = ToolCx { scene: &mut scene, history: &mut history2, camera: &mut camera, status: &mut status2, egui: &ctx, vp, brush_scroll: crate::navigation::BrushScroll::Shift };
        let mut tool = CapTool::new(&mut cx);
        let id2 = cx.scene.active.unwrap();
        tool.start(&mut cx, id2);
        assert!(tool.busy().is_some());
        let t0 = Instant::now();
        while tool.job.is_some() && t0.elapsed() < Duration::from_secs(30) {
            std::thread::sleep(Duration::from_millis(5));
            tool.poll(&mut cx);
        }
        assert_eq!(tool.message.as_deref(), Some(reply.as_str()));
        assert_eq!(*cx.status, reply);
        assert_eq!(tool.ends.len(), direct_ends);
        let (a, b) = (direct.get(id).unwrap(), cx.scene.get(id2).unwrap());
        assert_eq!(a.mesh.to_arrays().1, b.mesh.to_arrays().1);
        assert_eq!(a.mesh.to_arrays().2, b.mesh.to_arrays().2);
        assert_eq!(a.group_names, b.group_names);
        assert!(cx.history.can_undo());
        // A mesh changed meanwhile is not overwritten.
        tool.start(&mut cx, id2);
        cx.scene.get_mut(id2).unwrap().set_mesh(primitives::icosphere(1.0, 1));
        while tool.job.is_some() {
            std::thread::sleep(Duration::from_millis(5));
            tool.poll(&mut cx);
        }
        assert_eq!(cx.scene.get(id2).unwrap().mesh.triangle_count(), primitives::icosphere(1.0, 1).triangle_count());
    }
}
