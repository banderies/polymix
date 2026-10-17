//! Vessel / CFD preparation tools: square branch trimming, flow extensions, capping, and
//! cross-section estimation.

use super::*;
use crate::ui_util;

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
        // Poll the preview job.
        if let Some(j) = &self.job {
            if let Some(r) = j.poll() {
                self.job = None;
                match r {
                    Ok(out) => self.preview = Some(out),
                    Err(e) => {
                        self.preview = None;
                        self.error = Some(format!("{e:#}"));
                    }
                }
            } else {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label("Computing cut…");
                });
            }
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
            if let Some(c) = &mut self.cand {
                if moved != 0.0 {
                    c.plane.d += moved;
                    c.seed += c.plane.normal * moved;
                }
                if flip {
                    c.plane = mm_core::Plane { normal: -c.plane.normal, d: -c.plane.d };
                }
                if moved != 0.0 || flip {
                    if let Some(mesh) = self.object.and_then(|id| cx.scene.get(id)).map(|o| o.mesh.clone()) {
                        self.refresh_curve(&mesh);
                        self.start(cx);
                    }
                }
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
        self.object = Some(id);
        cx.scene.active = Some(id);
        let o = cx.scene.get_mut(id).unwrap();
        let bvh = o.bvh();
        let mesh = o.mesh.clone();
        match mm_edit::estimate_vessel_cross_section(&mesh, Some(&bvh), hit.point, None) {
            Ok(x) => {
                self.cand = Some(TrimCandidate { seed: hit.point, plane: x.plane, radius: x.radius, curve: Vec::new() });
                self.preview = None;
                self.refresh_curve(&mesh);
                self.start(cx);
            }
            Err(e) => self.error = Some(format!("{e}")),
        }
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
        }
    }

    fn start(&mut self, cx: &mut ToolCx) {
        let Some(o) = self.object.and_then(|id| cx.scene.get(id)) else { return };
        let src = o.mesh.clone();
        let source = src.clone();
        let ends: Vec<VesselEnd> = self.ends.iter().zip(&self.params.selected).filter(|(_, s)| **s).map(|(e, _)| e.clone()).collect();
        if ends.is_empty() {
            self.preview = None;
            return;
        }
        let p = self.params.clone();
        let p2 = p.clone();
        if let Some((_, j)) = &self.job {
            j.cancel();
        }
        self.error = None;
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
            self.start(cx);
        }
        if let Some((_, j)) = &self.job {
            if let Some(r) = j.poll() {
                let (p, _) = self.job.take().unwrap();
                match r {
                    Ok(out) => self.preview = Some((p, out)),
                    Err(e) => self.error = Some(format!("{e:#}")),
                }
            } else {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label("Building…");
                });
            }
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
        if self.preview.is_none() || self.job.is_some() {
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
}

impl CapTool {
    pub fn new(cx: &mut ToolCx) -> Self {
        let ends = cx.scene.active().map(|o| find_vessel_ends(&o.mesh)).unwrap_or_default().into_iter().filter(|e| matches!(e.kind, EndKind::Open)).collect();
        Self { object: cx.scene.active, ends, remeshed: true, message: None }
    }
}

impl Tool for CapTool {
    fn name(&self) -> &'static str {
        "Cap open ends"
    }

    fn panel(&mut self, ui: &mut egui::Ui, cx: &mut ToolCx) -> ToolOutcome {
        ui_util::section(ui, "Cap open ends");
        let Some(id) = self.object.filter(|id| cx.scene.get(*id).is_some()) else {
            ui.label("No active object.");
            return if ui.button("Close").clicked() { ToolOutcome::Close } else { ToolOutcome::Continue };
        };
        ui.label(format!("{} open boundary loop(s)", self.ends.len()));
        ui.checkbox(&mut self.remeshed, "Remesh caps to match the surrounding edge length");
        ui.small("Each cap becomes its own named patch (outlet_N). Non-planar holes are better filled with the Inspector's smooth fill.");
        if ui.add_enabled(!self.ends.is_empty(), egui::Button::new("Cap all")).clicked() {
            cx.history.push("Cap open ends", cx.scene);
            let o = cx.scene.get_mut(id).unwrap();
            let target = o.mesh.average_edge_length();
            let fill = if self.remeshed { mm_edit::CutFill::FlatRemeshed { target_edge_length: target } } else { mm_edit::CutFill::Flat };
            let loops: Vec<Vec<u32>> = mm_core::topology::boundary_loops(&o.mesh).into_iter().map(|l| l.vertices).collect();
            let mut m = (*o.mesh).clone();
            let mut capped = 0;
            let mut failed = 0;
            let mut groups = Vec::new();
            // Cap loops one at a time so each gets its own plane fit and group.
            for lp in &loops {
                match mm_edit::cap_planar_loops(&mut m, std::slice::from_ref(lp), None, fill) {
                    Ok(r) if r.failed_loops.is_empty() => {
                        capped += 1;
                        groups.extend(r.groups);
                    }
                    _ => failed += 1,
                }
            }
            o.set_mesh(m);
            for g in groups {
                let n = next_name(&o.group_names, "outlet");
                o.group_names.insert(g, n);
            }
            self.message = Some(format!("Capped {capped} loop(s){}", if failed > 0 { format!(", {failed} could not be capped flat") } else { String::new() }));
            *cx.status = self.message.clone().unwrap();
            self.ends = find_vessel_ends(&o.mesh).into_iter().filter(|e| matches!(e.kind, EndKind::Open)).collect();
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
}

// ---------------------------------------------------------------------------
// Bridge / join two open boundaries
// ---------------------------------------------------------------------------

/// Join two open boundary loops with a tube of triangles (Meshmixer Bridge / Join).
pub struct BridgeTool {
    object: Option<ObjectId>,
    ends: Vec<VesselEnd>,
    picked: Vec<usize>,
    smooth: bool,
    error: Option<String>,
}

impl BridgeTool {
    pub fn new(cx: &mut ToolCx) -> Self {
        let ends = cx.scene.active().map(|o| find_vessel_ends(&o.mesh)).unwrap_or_default().into_iter().filter(|e| matches!(e.kind, EndKind::Open)).collect();
        Self { object: cx.scene.active, ends, picked: Vec::new(), smooth: true, error: None }
    }
}

impl Tool for BridgeTool {
    fn name(&self) -> &'static str {
        "Bridge"
    }

    fn panel(&mut self, ui: &mut egui::Ui, cx: &mut ToolCx) -> ToolOutcome {
        ui_util::section(ui, "Bridge boundaries");
        let Some(id) = self.object.filter(|id| cx.scene.get(*id).is_some()) else {
            ui.label("No active object.");
            return if ui.button("Close").clicked() { ToolOutcome::Close } else { ToolOutcome::Continue };
        };
        ui.small("Click two open-boundary markers to connect them with a strip of triangles.");
        ui.checkbox(&mut self.smooth, "Smooth (curved) bridge");
        ui.label(format!("Picked: {:?}", self.picked.iter().map(|i| i + 1).collect::<Vec<_>>()));
        if ui.add_enabled(self.picked.len() == 2, egui::Button::new("Join")).clicked() {
            let o = cx.scene.get(id).unwrap();
            let mut m = (*o.mesh).clone();
            let a = nearest_loop(&m, self.ends[self.picked[0]].center);
            let b = nearest_loop(&m, self.ends[self.picked[1]].center);
            let r = match (a, b) {
                (Some(a), Some(b)) => mm_edit::join_loops(&mut m, &a, &b, &mm_edit::BridgeParams { segments: 0, smooth: self.smooth }).map_err(|e| e.to_string()),
                _ => Err("boundary not found".into()),
            };
            match r {
                Ok(res) => {
                    cx.history.push("Bridge", cx.scene);
                    cx.scene.get_mut(id).unwrap().set_mesh(m);
                    *cx.status = format!("Bridged with {} triangles", res.triangles.len());
                    self.ends = find_vessel_ends(&cx.scene.get(id).unwrap().mesh).into_iter().filter(|e| matches!(e.kind, EndKind::Open)).collect();
                    self.picked.clear();
                    self.error = None;
                }
                Err(e) => self.error = Some(e),
            }
        }
        if let Some(e) = &self.error {
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
