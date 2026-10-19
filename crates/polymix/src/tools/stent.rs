//! Virtual stent (Vessel → Virtual stent, or from two Measure → Vessel diameter sections):
//! click the vessel at the two ends of the stretch to treat, choose the stent's diameter and
//! length, how straight it holds and how hard it pushes, and whether it is covered; the vessel
//! is reshaped around it (see [`mm_edit::stent`]), previewed live, then kept in place or as a
//! copy.
//!
//! The vessel's centerline is traced from the first section (geodesic rings, as the
//! Centerline tool does) through the second and on past both, and the vertices whose rings lie
//! on it are the stented vessel's own wall; side branches follow them with distance.

use super::*;
use crate::centerline::{self, Centerline, CenterlineParams};
use crate::jobs::Job;
use crate::ui_util;
use mm_edit::EndContinuity;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// One end of the stretch: the vessel's cross-section through a point of its wall.
#[derive(Clone, Debug)]
pub struct Section {
    pub center: DVec3,
    pub normal: DVec3,
    /// Equivalent diameter (of a circle with the cross-section's area).
    pub diameter: f64,
    /// The cross-section's outline.
    pub ring: Vec<[DVec3; 2]>,
}

impl Section {
    /// The cross-section through `point` on the wall (as Measure → Vessel diameter).
    pub fn at(mesh: &Mesh, bvh: &mm_core::Bvh, point: DVec3) -> Option<Self> {
        let x = mm_edit::estimate_vessel_cross_section(mesh, Some(bvh), point, None).ok()?;
        let curve = mm_edit::nearest_intersection_curve(mesh, x.plane, point).filter(|c| c.closed)?;
        let diameter = x.area.map_or(2.0 * x.radius, |a| 2.0 * (a / std::f64::consts::PI).sqrt());
        Some(Self { center: x.center, normal: x.axis, diameter, ring: curve.segments })
    }

    fn stent(&self) -> mm_edit::StentSection {
        mm_edit::StentSection { center: self.center, normal: self.normal }
    }
}

/// The stent's settings as shown (percentages for the 0..1 ones).
#[derive(Clone, Copy, PartialEq, Debug)]
struct StentUi {
    diameter: f64,
    length: f64,
    straightness: f64,
    continuity: EndContinuity,
    stiffness: f64,
    end_blend: f64,
    covered: bool,
    remesh: bool,
}

impl Default for StentUi {
    fn default() -> Self {
        Self { diameter: 8.0, length: 40.0, straightness: 50.0, continuity: EndContinuity::G1, stiffness: 100.0, end_blend: 2.0, covered: false, remesh: true }
    }
}

impl StentUi {
    fn params(&self) -> mm_edit::StentParams {
        mm_edit::StentParams {
            diameter: self.diameter,
            length: self.length,
            straightness: self.straightness / 100.0,
            continuity: self.continuity,
            stiffness: self.stiffness / 100.0,
            end_blend: self.end_blend,
            covered: self.covered,
            remesh: self.remesh,
            edge_length: None,
        }
    }
}

/// The vessel's centerline through the two sections and on past both, and which vertices
/// belong to it.
struct VesselTrace {
    path: mm_edit::VesselPath,
    owner: Vec<bool>,
    /// Centerline length between the sections.
    span: f64,
}

/// What a preview was made from: the sections (by revision) and the settings.
#[derive(Clone, Copy, PartialEq, Debug)]
struct StentKey {
    sections: u64,
    ui: StentUi,
}

struct StentOut {
    source: Arc<Mesh>,
    mesh: Mesh,
    placement: mm_edit::StentPlacement,
    report: mm_edit::StentReport,
    crossings: usize,
}

const SETTLE: Duration = Duration::from_millis(150);

pub struct StentTool {
    object: Option<ObjectId>,
    sections: Vec<Section>,
    /// Bumped whenever the sections change.
    revision: u64,
    ui: StentUi,
    /// Keep the result in the object itself rather than a copy.
    in_place: bool,
    /// The diameter / length were typed in (otherwise they follow the sections).
    sized: [bool; 2],
    trace: Option<(u64, Arc<Mesh>, Arc<VesselTrace>)>,
    trace_job: Option<(u64, Job<VesselTrace>)>,
    job: Option<(StentKey, Job<StentOut>)>,
    preview: Option<(StentKey, StentOut)>,
    changed: Option<(StentKey, Instant)>,
    error: Option<(StentKey, String)>,
    pressed: Option<egui::Pos2>,
}

impl StentTool {
    pub fn new(cx: &mut ToolCx) -> Self {
        Self {
            object: cx.scene.active,
            sections: Vec::new(),
            revision: 0,
            ui: StentUi::default(),
            in_place: false,
            sized: [false; 2],
            trace: None,
            trace_job: None,
            job: None,
            preview: None,
            changed: None,
            error: None,
            pressed: None,
        }
    }

    /// Start from two cross-sections (e.g. Measure's vessel diameters) of `object`.
    pub fn with_sections(cx: &mut ToolCx, object: ObjectId, sections: [Section; 2]) -> Self {
        let mut t = Self::new(cx);
        t.object = Some(object);
        t.sections = sections.to_vec();
        t.sections_changed();
        t
    }

    fn sections_changed(&mut self) {
        self.revision += 1;
        self.trace = None;
        if let Some((_, j)) = self.trace_job.take() {
            j.cancel();
        }
        if self.sections.len() == 2 && !self.sized[0] {
            // The wider end's diameter, to the half millimetre.
            let d = self.sections.iter().map(|s| s.diameter).fold(0.0, f64::max);
            self.ui.diameter = ((d * 2.0).round() / 2.0).max(0.5);
        }
    }

    fn key(&self) -> Option<StentKey> {
        (self.sections.len() == 2).then_some(StentKey { sections: self.revision, ui: self.ui })
    }

    /// Set an end at the wall point `p`: the first two clicks set ends 1 and 2, later ones
    /// move the nearer end.
    fn set_end(&mut self, cx: &mut ToolCx, id: ObjectId, p: DVec3) {
        if self.object != Some(id) {
            self.object = Some(id);
            self.sections.clear();
        }
        let Some(o) = cx.scene.get_mut(id) else { return };
        let bvh = o.bvh();
        let Some(s) = Section::at(&o.mesh, &bvh, p) else {
            *cx.status = "No vessel cross-section there: click the wall of a tube-like vessel.".into();
            return;
        };
        if self.sections.len() < 2 {
            self.sections.push(s);
        } else {
            let i = if self.sections[0].center.distance(p) <= self.sections[1].center.distance(p) { 0 } else { 1 };
            self.sections[i] = s;
        }
        self.sections_changed();
    }

    /// The reshaped vessel for `key` (in the background, or at once for scripts).
    fn build(source: &Mesh, trace: &VesselTrace, sections: [mm_edit::StentSection; 2], key: &StentKey) -> anyhow::Result<(Mesh, mm_edit::StentPlacement, mm_edit::StentReport, usize)> {
        let params = key.ui.params();
        let placement = mm_edit::place_stent(&trace.path, sections, &params)?;
        let mut mesh = source.clone();
        let report = mm_edit::apply_stent(&mut mesh, &placement, &params, Some(&trace.owner))?;
        // Only crossings the stent made (a mesh may cross itself already, e.g. where
        // vessels were merged without a union).
        let crossings = ops::crossings(&mesh, &report.triangles).saturating_sub(ops::crossings(source, &report.source_triangles));
        Ok((mesh, placement, report, crossings))
    }

    /// Keep the trace and preview up to date: collect finished jobs, and start one for
    /// settings that have settled.
    fn update(&mut self, cx: &mut ToolCx) {
        let Some(o) = self.object.and_then(|id| cx.scene.get(id)) else { return };
        let source = o.mesh.clone();
        if let Some((rev, j)) = &self.trace_job {
            if let Some(r) = j.poll() {
                let rev = *rev;
                self.trace_job = None;
                match r {
                    Ok(t) if rev == self.revision => {
                        if !self.sized[1] {
                            // The stretch's length, up to the next 5 mm.
                            self.ui.length = ((t.span / 5.0).ceil() * 5.0).max(5.0);
                        }
                        self.trace = Some((rev, source.clone(), Arc::new(t)));
                    }
                    Ok(_) => {}
                    Err(e) => {
                        if let Some(k) = self.key() {
                            self.error = Some((k, format!("{e:#}")));
                        }
                    }
                }
            }
        }
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
        let current = |k: Option<&StentKey>| k == Some(&key);
        if current(self.preview.as_ref().map(|p| &p.0)) || current(self.job.as_ref().map(|j| &j.0)) || current(self.error.as_ref().map(|e| &e.0)) {
            self.changed = None;
            return;
        }
        // The trace first (once per pair of sections and mesh).
        let traced = self.trace.as_ref().is_some_and(|(rev, m, _)| *rev == self.revision && Arc::ptr_eq(m, &source));
        if !traced {
            if self.trace_job.as_ref().is_none_or(|(rev, _)| *rev != self.revision) {
                let sections = [self.sections[0].clone(), self.sections[1].clone()];
                let mesh = source.clone();
                self.trace_job = Some((self.revision, Job::spawn("Centerline", cx.egui.clone(), move |_| trace_vessel(&mesh, &sections))));
            }
            return;
        }
        let now = Instant::now();
        match self.changed {
            Some((k, since)) if k == key && now - since >= SETTLE => self.start(key, cx),
            Some((k, since)) if k == key => cx.egui.request_repaint_after(SETTLE - (now - since)),
            _ => {
                self.changed = Some((key, now));
                cx.egui.request_repaint_after(SETTLE);
            }
        }
    }

    fn start(&mut self, key: StentKey, cx: &mut ToolCx) {
        self.changed = None;
        let Some((_, source, trace)) = self.trace.clone() else { return };
        if let Some((_, j)) = self.job.take() {
            j.cancel();
        }
        let sections = [self.sections[0].stent(), self.sections[1].stent()];
        self.job = Some((
            key,
            Job::spawn("Stent", cx.egui.clone(), move |_| {
                let (mesh, placement, report, crossings) = Self::build(&source, &trace, sections, &key)?;
                Ok(StentOut { source, mesh, placement, report, crossings })
            }),
        ));
    }

    /// Keep the reshaped vessel (the preview if it is up to date, else built now), in the
    /// object or as a copy. Returns the status.
    fn apply(&mut self, cx: &mut ToolCx) -> Result<String, String> {
        let key = self.key().ok_or("click the vessel at both ends of the stretch first")?;
        let id = self.object.ok_or("no object")?;
        let o = cx.scene.get(id).ok_or("no object")?;
        let (source, name, frame) = (o.mesh.clone(), o.name.clone(), o.frame);
        let (mesh, report) = match self.preview.take() {
            Some((k, out)) if k == key && Arc::ptr_eq(&out.source, &source) => (out.mesh, out.report),
            _ => {
                let trace = match &self.trace {
                    Some((rev, m, t)) if *rev == self.revision && Arc::ptr_eq(m, &source) => t.clone(),
                    _ => Arc::new(trace_vessel(&source, &[self.sections[0].clone(), self.sections[1].clone()]).map_err(|e| format!("{e:#}"))?),
                };
                let (mesh, _, report, _) = Self::build(&source, &trace, [self.sections[0].stent(), self.sections[1].stent()], &key).map_err(|e| format!("{e:#}"))?;
                (mesh, report)
            }
        };
        if let Some((_, j)) = self.job.take() {
            j.cancel();
        }
        let label = format!("⌀ {} × {} mm{}", trim(self.ui.diameter), trim(self.ui.length), if self.ui.covered { " covered" } else { "" });
        cx.history.push("Stent", cx.scene);
        if self.in_place {
            cx.scene.get_mut(id).unwrap().set_mesh(mesh);
        } else {
            let nid = cx.scene.add(format!("{name} stent {label}"), mesh);
            if let Some(n) = cx.scene.get_mut(nid) {
                n.frame = frame;
            }
            if let Some(o) = cx.scene.get_mut(id) {
                o.visible = false;
            }
            self.object = Some(nid);
        }
        self.sections.clear();
        self.sections_changed();
        self.error = None;
        Ok(format!("Stent {label}: narrowest lumen ⌀ {:.2} → {:.2} mm{}", 2.0 * report.narrowest[0], 2.0 * report.narrowest[1], sealed_text(&report)))
    }
}

fn trim(x: f64) -> String {
    let s = format!("{x:.2}");
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

fn sealed_text(r: &mm_edit::StentReport) -> String {
    match r.sealed_openings {
        0 if r.removed_triangles > 0 => "; the stretch is lined with the stent".into(),
        0 => String::new(),
        n => format!("; {n} open end{} sealed off and removed", if n == 1 { "" } else { "s" }),
    }
}

fn point_segment_distance(p: DVec3, a: DVec3, b: DVec3) -> f64 {
    let ab = b - a;
    let t = ((p - a).dot(ab) / ab.length_squared().max(1e-300)).clamp(0.0, 1.0);
    p.distance(a + ab * t)
}

/// Trace the vessel's centerline from the first section through the second and on past
/// both (the straightest way at each junction).
fn trace_vessel(mesh: &Mesh, sections: &[Section; 2]) -> anyhow::Result<VesselTrace> {
    let [a, b] = sections;
    let h = mesh.average_edge_length();
    let n = a.normal.normalize_or_zero();
    let sources: Vec<u32> = mesh
        .vertex_ids()
        .filter(|&v| {
            let p = mesh.vertex(v);
            (p - a.center).dot(n).abs() < h && a.ring.iter().any(|s| point_segment_distance(p, s[0], s[1]) < h)
        })
        .collect();
    if sources.is_empty() {
        anyhow::bail!("could not find the first end's cross-section on the surface");
    }
    let cl = centerline::compute(mesh, &sources, 0.5 * a.diameter, &CenterlineParams::default());
    let mut on_branch = vec![false; cl.nodes.len()];
    let mut adjacent: Vec<Vec<usize>> = vec![Vec::new(); cl.nodes.len()];
    for br in &cl.branches {
        for w in br.nodes.windows(2) {
            adjacent[w[0]].push(w[1]);
            adjacent[w[1]].push(w[0]);
        }
        for &i in &br.nodes {
            on_branch[i] = true;
        }
    }
    let near = |p: DVec3| (0..cl.nodes.len()).filter(|&i| on_branch[i]).min_by(|&i, &j| cl.nodes[i].position.distance(p).total_cmp(&cl.nodes[j].position.distance(p)));
    let (Some(na), Some(nb)) = (near(a.center), near(b.center)) else {
        anyhow::bail!("could not trace the vessel's centerline");
    };
    let chain = chain_between(&cl, &adjacent, na, nb).ok_or_else(|| anyhow::anyhow!("the two ends are not on one connected vessel"))?;
    if chain.len() < 2 {
        anyhow::bail!("the two ends are too close together along the vessel");
    }
    let pos = |i: usize| cl.nodes[i].position;
    let length = |c: &[usize]| c.windows(2).map(|w| pos(w[0]).distance(pos(w[1]))).sum::<f64>();
    let reach = (3.0 * length(&chain)).max(20.0 * a.diameter.max(b.diameter));
    let before = walk_on(&cl, &adjacent, chain[0], chain[1], reach);
    let after = walk_on(&cl, &adjacent, chain[chain.len() - 1], chain[chain.len() - 2], reach);
    let nodes: Vec<usize> = before.iter().rev().chain(&chain).chain(&after).copied().collect();
    let on_path: std::collections::HashSet<usize> = nodes.iter().copied().collect();
    let owner = cl.vertex_node.iter().map(|n| on_path.contains(n)).collect();
    let path = mm_edit::VesselPath { points: nodes.iter().map(|&i| pos(i)).collect(), radii: nodes.iter().map(|&i| cl.nodes[i].radius).collect() };
    let span = path.span([a.stent(), b.stent()]).unwrap_or(0.0);
    Ok(VesselTrace { path, owner, span })
}

/// Nodes from `from` to `to` through the branch tree (breadth first over branch links).
fn chain_between(cl: &Centerline, adjacent: &[Vec<usize>], from: usize, to: usize) -> Option<Vec<usize>> {
    let mut prev = vec![usize::MAX; cl.nodes.len()];
    let mut queue = std::collections::VecDeque::from([from]);
    prev[from] = from;
    while let Some(x) = queue.pop_front() {
        if x == to {
            break;
        }
        for &y in &adjacent[x] {
            if prev[y] == usize::MAX {
                prev[y] = x;
                queue.push_back(y);
            }
        }
    }
    if prev[to] == usize::MAX {
        return None;
    }
    let mut out = vec![to];
    while *out.last().unwrap() != from {
        out.push(prev[*out.last().unwrap()]);
    }
    out.reverse();
    Some(out)
}

/// Nodes on past `end` away from `back`, taking the straightest way at junctions, for about
/// `reach`.
fn walk_on(cl: &Centerline, adjacent: &[Vec<usize>], end: usize, back: usize, reach: f64) -> Vec<usize> {
    let pos = |i: usize| cl.nodes[i].position;
    let (mut prev, mut cur, mut len) = (back, end, 0.0);
    let mut seen: std::collections::HashSet<usize> = [back, end].into_iter().collect();
    let mut out = Vec::new();
    while len < reach {
        let dir = (pos(cur) - pos(prev)).normalize_or_zero();
        let Some(next) = adjacent[cur].iter().copied().filter(|n| !seen.contains(n)).max_by(|&x, &y| (pos(x) - pos(cur)).normalize_or_zero().dot(dir).total_cmp(&(pos(y) - pos(cur)).normalize_or_zero().dot(dir))) else {
            break;
        };
        len += pos(cur).distance(pos(next));
        seen.insert(next);
        out.push(next);
        (prev, cur) = (cur, next);
    }
    out
}

impl Tool for StentTool {
    fn name(&self) -> &'static str {
        "Virtual stent"
    }

    fn script(&mut self, cmd: &str, cx: &mut ToolCx) {
        // `end:x,y,z` (the nearest wall point), `clear`, `apply`, and the settings:
        // `diameter`, `length`, `straightness` / `stiffness` (%), `ends:g0|g1|g2`, `blend`,
        // `covered:on|off`, `remesh:on|off`, `inplace:on|off`.
        let (key, val) = cmd.split_once(':').unwrap_or((cmd, ""));
        let num = val.parse::<f64>().ok();
        let on = val != "off" && val != "0";
        match key {
            "end" => {
                let v: Vec<f64> = val.split(',').filter_map(|x| x.trim().parse().ok()).collect();
                if let (Some(id), &[x, y, z]) = (self.object.or(cx.scene.active), &v[..]) {
                    let point = cx.scene.get_mut(id).and_then(|o| o.bvh().nearest(&o.mesh, DVec3::new(x, y, z), f64::INFINITY)).map(|h| h.point);
                    if let Some(p) = point {
                        self.set_end(cx, id, p);
                    }
                }
            }
            "clear" => {
                self.sections.clear();
                self.sections_changed();
            }
            "apply" => {
                // Scripts wait for the trace, then apply (built at once if no preview).
                match self.apply(cx) {
                    Ok(msg) => *cx.status = msg,
                    Err(e) => *cx.status = format!("Stent: {e}"),
                }
            }
            "diameter" => {
                self.ui.diameter = num.unwrap_or(self.ui.diameter);
                self.sized[0] = true;
            }
            "length" => {
                self.ui.length = num.unwrap_or(self.ui.length);
                self.sized[1] = true;
            }
            "straightness" => self.ui.straightness = num.unwrap_or(self.ui.straightness).clamp(0.0, 100.0),
            "stiffness" => self.ui.stiffness = num.unwrap_or(self.ui.stiffness).clamp(0.0, 100.0),
            "blend" => self.ui.end_blend = num.unwrap_or(self.ui.end_blend).max(0.0),
            "ends" => {
                self.ui.continuity = match val {
                    "g0" => EndContinuity::G0,
                    "g2" => EndContinuity::G2,
                    _ => EndContinuity::G1,
                }
            }
            "covered" => self.ui.covered = on,
            "remesh" => self.ui.remesh = on,
            "inplace" => self.in_place = on,
            _ => {}
        }
        self.update(cx);
    }

    fn preview(&self) -> Option<(ObjectId, &Mesh)> {
        let (k, out) = self.preview.as_ref()?;
        (Some(*k) == self.key()).then_some((self.object?, &out.mesh))
    }

    fn accept(&mut self, cx: &mut ToolCx) -> bool {
        if self.job.is_some() || self.trace_job.is_some() {
            return false;
        }
        if self.sections.len() == 2 {
            match self.apply(cx) {
                Ok(msg) => *cx.status = msg,
                Err(e) => *cx.status = format!("Stent: {e}"),
            }
        }
        true
    }

    fn busy(&self) -> Option<(String, f32)> {
        if let Some((_, j)) = &self.trace_job {
            return Some(("Tracing the centerline".into(), j.progress()));
        }
        self.job.as_ref().map(|(_, j)| ("Stent".to_string(), j.progress()))
    }

    fn panel(&mut self, ui: &mut egui::Ui, cx: &mut ToolCx) -> ToolOutcome {
        ui_util::section(ui, "Virtual stent");
        if self.object.and_then(|id| cx.scene.get(id)).is_none() {
            ui.label("No active object.");
            return if ui.button("Close").clicked() { ToolOutcome::Close } else { ToolOutcome::Continue };
        }
        ui.small("Click the vessel wall at both ends of the stretch to treat (green cross-sections); click again to move the nearer end. The stent is centred on the two ends and the vessel is reshaped around it, previewed as you change the settings.");
        ui.horizontal(|ui| {
            for (i, s) in self.sections.iter().enumerate() {
                ui.label(format!("End {}: ⌀ {:.2}", i + 1, s.diameter));
            }
            if self.sections.is_empty() {
                ui.weak("No ends yet.");
            }
            if !self.sections.is_empty() && ui.small_button("Clear").clicked() {
                self.sections.clear();
                self.sections_changed();
            }
        });
        if let Some((_, _, t)) = self.trace.as_ref().filter(|t| t.0 == self.revision) {
            ui.label(format!("Centerline between the ends: {:.1} mm", t.span));
        }
        ui.separator();
        egui::Grid::new("stent_size").num_columns(2).show(ui, |ui| {
            ui.label("Diameter");
            if ui.add(egui::DragValue::new(&mut self.ui.diameter).speed(0.05).range(0.5..=60.0).max_decimals(2).suffix(" mm")).on_hover_text("The stent's nominal (fully expanded) diameter").changed() {
                self.sized[0] = true;
            }
            ui.end_row();
            ui.label("Length");
            if ui.add(egui::DragValue::new(&mut self.ui.length).speed(0.25).range(1.0..=400.0).max_decimals(1).suffix(" mm")).on_hover_text("Along the stent's own axis; it is centred on the two ends, so any extra length reaches equally far past each").changed() {
                self.sized[1] = true;
            }
            ui.end_row();
        });
        ui.add(egui::Slider::new(&mut self.ui.straightness, 0.0..=100.0).suffix("%").text("Straightness"))
            .on_hover_text("0 % follows the vessel's centerline as it is; 100 % takes the smoothest path between the two ends (where a free stent held at both would settle), straightening the vessel where the stent grips it");
        ui.horizontal(|ui| {
            ui.label("Smoothest path");
            ui.selectable_value(&mut self.ui.continuity, EndContinuity::G0, "G0").on_hover_text("A straight line between the ends' centres");
            ui.selectable_value(&mut self.ui.continuity, EndContinuity::G1, "G1").on_hover_text("Leaves each end square to its cross-section");
            ui.selectable_value(&mut self.ui.continuity, EndContinuity::G2, "G2").on_hover_text("Also continues the vessel's bend at each end");
        });
        ui.add(egui::Slider::new(&mut self.ui.stiffness, 0.0..=100.0).suffix("%").text("Stiffness"))
            .on_hover_text("Radial force: how far narrower wall is opened towards the stent's diameter. 100 % makes a perfect tube of the stent's diameter; 0 % leaves the wall as it is. Wall wider than the stent is never pulled in (unless covered)");
        ui.add(egui::Slider::new(&mut self.ui.end_blend, 0.0..=10.0).suffix(" mm").text("End blend"))
            .on_hover_text("How far past each end of the stent the wall eases back to its own shape");
        ui.checkbox(&mut self.ui.covered, "Covered (stent graft)")
            .on_hover_text("The lumen along the stent becomes the stent's tube: wider wall (an aneurysm, a dilated stretch) is narrowed to the stent's diameter, and side branches opening along it are sealed off and removed");
        ui.checkbox(&mut self.ui.remesh, "Remesh the reshaped wall").on_hover_text("Even triangles over the reshaped region (recommended for CFD)");
        ui.checkbox(&mut self.in_place, "Edit in place").on_hover_text("Reshape this object itself; otherwise the result is a new object (this one is hidden)");
        self.update(cx);
        let key = self.key();
        let current = self.preview.as_ref().filter(|p| Some(p.0) == key).map(|p| &p.1);
        if let Some(out) = current {
            let pl = &out.placement;
            let [a, b] = pl.overhang();
            let past = |x: f64| if x >= 0.0 { format!("{x:.1} mm past") } else { format!("{:.1} mm short of", -x) };
            ui.separator();
            ui.label(format!("Stent axis between the ends: {:.1} mm (centerline {:.1} mm)", pl.path_length, pl.centerline_length));
            ui.label(format!("Reaches {} end 1 and {} end 2", past(a), past(b)));
            ui.label(format!("Narrowest lumen ⌀ {:.2} → {:.2} mm", 2.0 * out.report.narrowest[0], 2.0 * out.report.narrowest[1]));
            let sealed = sealed_text(&out.report);
            if !sealed.is_empty() {
                ui.label(sealed.trim_start_matches("; "));
            }
            let warn = egui::Color32::from_rgb(240, 170, 80);
            if let Some(e) = (0..2).filter(|&e| pl.return_bend[e] > 0.7).max_by(|&a, &b| pl.return_bend[a].total_cmp(&pl.return_bend[b])) {
                ui.colored_label(warn, format!("Past end {} the vessel has to bend back sharply onto its course, so its wall may crease: lower the straightness.", e + 1));
            }
            let moved = out.report.moved_openings.saturating_sub(out.report.sealed_openings);
            if moved > 0 {
                ui.colored_label(warn, format!("The reshaping reaches {moved} open end{} of the mesh and moves {}: check the inlets and outlets, or lower the straightness.", if moved == 1 { "" } else { "s" }, if moved == 1 { "it" } else { "them" }));
            }
            if out.report.outside {
                ui.colored_label(warn, "The stent's axis runs outside the vessel (a tight bend): lower the straightness.");
            }
            if out.crossings > 0 {
                ui.colored_label(warn, format!("The reshaped wall runs into itself ({} new crossings): lower the straightness or the stent's length.", out.crossings));
            }
        }
        let ready = current.is_some();
        ui.horizontal(|ui| {
            if ui.add_enabled(ready, egui::Button::new("Apply")).on_hover_text("Keep the previewed result (Enter)").clicked() {
                match self.apply(cx) {
                    Ok(msg) => *cx.status = msg,
                    Err(e) => self.error = key.map(|k| (k, e)),
                }
            }
            if key.is_some() && !ready && self.error.as_ref().map(|e| e.0) != key {
                ui.spinner();
                ui.weak(if self.trace_job.is_some() { "Tracing the centerline…" } else { "Previewing…" });
            }
        });
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
        let Some(pos) = input.pos else { return };
        if input.pressed {
            self.pressed = Some(pos);
        }
        if !input.released {
            return;
        }
        let Some(p0) = self.pressed.take() else { return };
        if p0.distance(pos) > 5.0 {
            return;
        }
        if let Some((id, hit)) = cx.pick(pos, self.object.filter(|_| !self.sections.is_empty())) {
            self.set_end(cx, id, hit.point);
        }
    }

    fn overlay(&mut self, ov: &mut Overlay, _cx: &mut ToolCx) {
        let green = Vec4::new(0.3, 1.0, 0.6, 1.0);
        for s in &self.sections {
            for seg in &s.ring {
                ov.line(seg[0], seg[1], green);
            }
        }
        let Some((_, out)) = self.preview.as_ref().filter(|p| Some(p.0) == self.key()) else { return };
        let pl = &out.placement;
        let colour = if self.ui.covered { Vec4::new(0.55, 0.75, 1.0, 1.0) } else { Vec4::new(0.85, 0.88, 0.95, 1.0) };
        let count = ((pl.length / (0.75 * pl.radius).max(0.25)).round() as usize).clamp(4, 200);
        let rings = pl.rings(24, count);
        for (j, ring) in rings.iter().enumerate() {
            let c = if j == 0 || j == count { Vec4::new(1.0, 0.75, 0.3, 1.0) } else { colour };
            for i in 0..ring.len() {
                ov.line(ring[i], ring[(i + 1) % ring.len()], c);
            }
        }
        for i in (0..24).step_by(3) {
            for w in rings.windows(2) {
                ov.line(w[0][i], w[1][i], colour);
            }
        }
        let axis: Vec<DVec3> = (0..pl.s.len()).filter(|&i| (0.0..=pl.length).contains(&pl.s[i])).map(|i| pl.axis[i]).collect();
        for w in axis.windows(2) {
            ov.line(w[0], w[1], Vec4::new(1.0, 0.85, 0.2, 1.0));
        }
    }

    fn paint(&mut self, painter: &egui::Painter, cx: &mut ToolCx) {
        for (i, s) in self.sections.iter().enumerate() {
            if let Some(p) = cx.project(s.center) {
                painter.text(p, egui::Align2::LEFT_BOTTOM, format!("{} ⌀ {:.2}", i + 1, s.diameter), egui::FontId::proportional(14.0), egui::Color32::from_rgb(120, 255, 170));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A vessel of radius 4 bending through 120° on a radius of 25 (with 15 mm straight
    /// leads), narrowed eccentrically to about 1.8 halfway round.
    fn bent_stenosis() -> (Mesh, impl Fn(f64) -> DVec3) {
        let arc = 25.0 * 120f64.to_radians();
        let centre = move |s: f64| -> DVec3 {
            if s < 15.0 {
                DVec3::new(25.0, s - 15.0, 0.0)
            } else if s < 15.0 + arc {
                let a = (s - 15.0) / 25.0;
                DVec3::new(25.0 * a.cos(), 25.0 * a.sin(), 0.0)
            } else {
                let a = 120f64.to_radians();
                DVec3::new(25.0 * a.cos(), 25.0 * a.sin(), 0.0) + DVec3::new(-a.sin(), a.cos(), 0.0) * (s - 15.0 - arc)
            }
        };
        let total = 30.0 + arc;
        let mid = 15.0 + 0.5 * arc;
        let n = (total / 0.35) as usize;
        let path: Vec<DVec3> = (0..=n).map(|i| centre(total * i as f64 / n as f64)).collect();
        let radii: Vec<f64> = (0..=n).map(|i| 4.0 - 2.2 * (-((total * i as f64 / n as f64 - mid) / 4.0).powi(2)).exp()).collect();
        (mm_core::primitives::tube_along(&path, &radii, 72), centre)
    }

    /// Sharpest bend between neighbouring triangles of `tris` (degrees).
    fn sharpest(m: &Mesh, tris: &[u32]) -> f64 {
        let set: std::collections::HashSet<u32> = tris.iter().copied().collect();
        m.edge_ids()
            .filter_map(|e| {
                let [s, t] = m.edge_triangles(e);
                (t != mm_core::INVALID && set.contains(&s) && set.contains(&t)).then(|| m.tri_normal(s).angle_between(m.tri_normal(t)).to_degrees())
            })
            .fold(0.0, f64::max)
    }

    #[test]
    fn stent_in_a_bent_stenosed_vessel() {
        // Traced from the first end, the stent opens the narrowing; straightened (G0) and
        // covered, the wall stays free of crossings and folds.
        let (m, centre) = bent_stenosis();
        let bvh = mm_core::Bvh::build(&m);
        let wall = |s: f64| {
            let c = centre(s);
            bvh.nearest(&m, c + DVec3::Z * 4.0, f64::INFINITY).unwrap().point
        };
        let a0 = 15.0 + 25.0 * 30f64.to_radians();
        let b0 = 15.0 + 25.0 * 90f64.to_radians();
        let sections = [Section::at(&m, &bvh, wall(a0)).unwrap(), Section::at(&m, &bvh, wall(b0)).unwrap()];
        let trace = trace_vessel(&m, &sections).unwrap();
        assert!((trace.span - (b0 - a0)).abs() < 0.6, "span {} vs {}", trace.span, b0 - a0);
        for (ui, label) in [
            (StentUi { length: 30.0, ..Default::default() }, "bare, half straight"),
            (StentUi { length: 30.0, straightness: 100.0, continuity: EndContinuity::G0, ..Default::default() }, "bare, straight"),
            (StentUi { length: 30.0, straightness: 100.0, continuity: EndContinuity::G0, covered: true, stiffness: 50.0, ..Default::default() }, "covered, straight"),
        ] {
            let key = StentKey { sections: 1, ui };
            let (out, pl, rep, crossings) = StentTool::build(&m, &trace, [sections[0].stent(), sections[1].stent()], &key).unwrap();
            out.check_validity().unwrap();
            assert!(out.is_consistently_oriented());
            let bend = sharpest(&out, &rep.triangles);
            let [a, b] = pl.overhang();
            assert!((a - b).abs() < 1e-6 && a > 0.0, "{label}: centred ({a:.3} / {b:.3})");
            assert_eq!(crossings, 0, "{label}");
            assert!(bend < 30.0, "{label}: {bend:.1}°");
            let want = 4.0 + (ui.stiffness / 100.0) * 4.0;
            assert!(2.0 * rep.narrowest[1] > want - 0.3, "{label}: {:?}", rep.narrowest);
            assert_eq!(mm_core::topology::boundary_loops(&out).len(), 2, "{label}");
        }
    }

    /// A real mesh (`POLYMIX_STENT_MESH`, local only): stents along the longest branch of its
    /// centerline, traced from the widest opening. Prints counts and timings only.
    #[test]
    #[ignore]
    fn stent_on_a_real_mesh() {
        let Some(path) = std::env::var_os("POLYMIX_STENT_MESH") else { return };
        let m = mm_io::read_mesh(std::path::Path::new(&path)).unwrap().mesh;
        let loops = mm_core::topology::boundary_loops(&m);
        let (sources, r) = match loops.iter().max_by(|a, b| a.perimeter(&m).total_cmp(&b.perimeter(&m))) {
            Some(inlet) => (inlet.vertices.clone(), inlet.perimeter(&m) / std::f64::consts::TAU),
            None => {
                // Closed: from the widest flat cap.
                let ends = super::super::vessel::find_vessel_ends(&m);
                let Some(e) = ends.iter().max_by(|a, b| a.diameter.total_cmp(&b.diameter)) else {
                    eprintln!("{} triangles, closed with no flat caps: skipped", m.triangle_count());
                    return;
                };
                let super::super::vessel::EndKind::Cap(g) = e.kind else { panic!("no cap") };
                let mut v: Vec<u32> = m.triangle_ids().filter(|&t| m.tri_group(t) == g).flat_map(|t| m.triangle(t)).collect();
                v.sort_unstable();
                v.dedup();
                (v, 0.5 * e.diameter)
            }
        };
        let cl = centerline::compute(&m, &sources, r, &CenterlineParams::default());
        let branch = cl.branches.iter().max_by(|a, b| a.length.total_cmp(&b.length)).unwrap();
        let all: Vec<u32> = m.triangle_ids().collect();
        eprintln!("input crossings {}", ops::crossings(&m, &all));
        eprintln!("{} triangles, {} open ends, longest branch {:.1} mm over {} nodes", m.triangle_count(), loops.len(), branch.length, branch.nodes.len());
        let bvh = mm_core::Bvh::build(&m);
        // Ends where the cross-section matches the centerline's ring (a clean stretch of tube).
        let clean: Vec<(f64, Section)> = {
            let mut along = 0.0;
            let mut out = Vec::new();
            for (k, &i) in branch.nodes.iter().enumerate() {
                if k > 0 {
                    along += cl.nodes[i].position.distance(cl.nodes[branch.nodes[k - 1]].position);
                }
                let node = cl.nodes[i];
                let wall = bvh.nearest(&m, node.position, f64::INFINITY).unwrap().point;
                if let Some(sec) = Section::at(&m, &bvh, wall) {
                    if (sec.diameter / (2.0 * node.radius) - 1.0).abs() < 0.25 && sec.center.distance(node.position) < 0.5 * node.radius {
                        out.push((along, sec));
                    }
                }
            }
            out
        };
        eprintln!("{} clean sections along it", clean.len());
        let mut pairs = Vec::new();
        let inner = |i: usize| clean[i].0 > 0.15 * branch.length && clean[i].0 < 0.85 * branch.length;
        for want in [3.0, 5.0] {
            let d = clean.first().map_or(1.0, |c| c.1.diameter);
            if let Some((i, j)) = (0..clean.len()).flat_map(|i| (i + 1..clean.len()).map(move |j| (i, j))).filter(|&(i, j)| inner(i) && inner(j)).min_by(|&(a, b), &(c, e)| ((clean[b].0 - clean[a].0) - want * d).abs().total_cmp(&((clean[e].0 - clean[c].0) - want * d).abs())) {
                pairs.push((clean[i].1.clone(), clean[j].1.clone()));
            }
        }
        for (a, b) in pairs {
            let t0 = std::time::Instant::now();
            let sections = [a, b];
            eprintln!("sections {:.1} mm apart", sections[0].center.distance(sections[1].center));
            let trace = match trace_vessel(&m, &sections) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("trace failed: {e:#}");
                    continue;
                }
            };
            eprintln!("ends: ⌀ {:.2} / {:.2}, span {:.1} mm, traced in {:.0} ms", sections[0].diameter, sections[1].diameter, trace.span, t0.elapsed().as_secs_f64() * 1e3);
            let d = ((sections[0].diameter.max(sections[1].diameter) * 2.0).round() / 2.0).max(0.5);
            let length = (trace.span / 5.0).ceil() * 5.0 + 5.0;
            for (ui, label) in [
                (StentUi { diameter: d, length, straightness: 0.0, ..Default::default() }, "bare, follows"),
                (StentUi { diameter: d, length, straightness: 100.0, ..Default::default() }, "bare, G1 smoothest"),
                (StentUi { diameter: d, length, straightness: 100.0, continuity: EndContinuity::G0, ..Default::default() }, "bare, straight"),
                (StentUi { diameter: d, length, straightness: 50.0, covered: true, ..Default::default() }, "covered"),
            ] {
                let t0 = std::time::Instant::now();
                let key = StentKey { sections: 1, ui };
                match StentTool::build(&m, &trace, [sections[0].stent(), sections[1].stent()], &key) {
                    Ok((out, pl, rep, crossings)) => eprintln!(
                        "  {label}: {:.0} ms, valid {}, oriented {}, crossings {crossings}, sharpest {:.1}°, narrowest ⌀ {:.2} -> {:.2}, moved {}, overhang {:.1}/{:.1}, outside {}, open ends {} -> {}, sealed {}",
                        t0.elapsed().as_secs_f64() * 1e3,
                        out.check_validity().is_ok(),
                        out.is_consistently_oriented(),
                        sharpest(&out, &rep.triangles),
                        2.0 * rep.narrowest[0],
                        2.0 * rep.narrowest[1],
                        rep.moved,
                        pl.overhang()[0],
                        pl.overhang()[1],
                        rep.outside,
                        loops.len(),
                        mm_core::topology::boundary_loops(&out).len(),
                        rep.sealed_openings
                    ),
                    Err(e) => eprintln!("  {label}: failed: {e:#}"),
                }
            }
        }
    }
}
