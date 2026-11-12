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
    /// Outward normal of the end's plane (away from the vessel behind it).
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
        // Away from the vessel: the triangles along the loop lie on its side. Where they do
        // not tell (a hole in a flat sheet), away from the middle of the mesh.
        let (mut side, mut scale) = (0.0, 0.0);
        for &v in &l.vertices {
            for t in mesh.vertex_triangles(v) {
                let d = mesh.tri_centroid(t) - c;
                side += d.dot(n);
                scale += d.length();
            }
        }
        let behind = if side.abs() > 0.05 * scale { side } else { -(c - mesh_c).dot(n) };
        if behind > 0.0 {
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
pub(crate) fn next_name(names: &std::collections::HashMap<i32, String>, prefix: &str) -> String {
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

/// Which way the extensions run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Heading {
    /// Square to the end: the normal of its cut (or cap).
    CutNormal,
    /// On along the vessel: a line fitted to its centerline just inside the end.
    Centerline,
}

#[derive(Clone, PartialEq)]
struct FlowParams {
    diameters: f64,
    absolute: Option<f64>,
    circular: bool,
    cap: bool,
    heading: Heading,
    /// The stretch of centerline the heading is fitted to, in diameters of the end
    /// ([`Heading::Centerline`]).
    fit: f64,
    selected: Vec<bool>,
    /// Ends aimed by hand (their arrows dragged), by end: the unit direction, in place of the
    /// heading.
    aims: Vec<Option<DVec3>>,
    /// Each extension the diameter and length of its end's `sizes` (a hose to fit), tapering
    /// to it from the opening, rather than sized from the cut.
    by_hand: bool,
    /// By end: its extension's size by hand.
    sizes: Vec<EndSize>,
    /// Sized by hand: how much of each extension tapers from its opening to its diameter, mm
    /// (all of a shorter one); the rest is straight.
    taper: f64,
    /// Sized by hand: a barb near each extension's end for its hose to grip.
    barb: bool,
    /// How much wider than the extension its barb's crest is, on the diameter, mm.
    barb_width: f64,
    /// From the extension's end back to where its barb starts to rise, mm.
    barb_setback: f64,
}

/// An extension's size set by hand, mm.
#[derive(Clone, Copy, Debug, PartialEq)]
struct EndSize {
    diameter: f64,
    /// The taper included.
    length: f64,
}

/// What an extension's diameter, length and taper can be set to by hand, mm.
const SIZE_RANGE: std::ops::RangeInclusive<f64> = 0.1..=200.0;
/// A taper steeper than this off the extension's axis (degrees) is pointed out.
const STEEP: f64 = 45.0;
/// Fewest rings a taper is built with (where the extension's length allows).
const TAPER_RINGS: f64 = 8.0;
/// What a barb's width (on the diameter) can be set to, mm.
const BARB_WIDTH_RANGE: std::ops::RangeInclusive<f64> = 0.01..=50.0;
/// How far back from the end a barb can be set, mm.
const BARB_SETBACK_RANGE: std::ops::RangeInclusive<f64> = 0.0..=200.0;
/// A barb's ramp, off the extension's axis, degrees: gentle enough to push a hose up.
const BARB_RAMP: f64 = 15.0;

/// Length of the ramp of a barb `width` wider than its extension (on the diameter), mm.
fn barb_ramp(width: f64) -> f64 {
    0.5 * width / BARB_RAMP.to_radians().tan()
}

/// `x` to the nearest `step`.
fn to_step(x: f64, step: f64) -> f64 {
    (x / step).round() * step
}

/// Longest stretch of centerline a heading is fitted to, in diameters of the end (the
/// vessel is traced within [`TRACE_REACH`] radii of it).
const MAX_FIT: f64 = 5.0;
/// How far from an end its vessel is traced, in radii of the end.
const TRACE_REACH: f64 = 12.0;
/// Shortest stretch of centerline a heading is fitted to, in diameters of the end: where the
/// vessel forks sooner, the centerline there runs from the opening's centre to the fork's,
/// which says little about the vessel's own course.
const MIN_FIT: f64 = 0.5;
/// The vessel behind an end is followed while its radius stays under this multiple of the
/// end's: wider, its centerline has entered a bigger vessel (a short branch off a trunk,
/// whose fork the trace of a region only a few trunk radii across can miss) or a sac.
const WIDEN: f64 = 1.5;
/// Extensions further than this off their cut's normal (degrees) are pointed out.
const OBLIQUE: f64 = 45.0;

/// What comes of an extension [`OBLIQUE`]ly off its cut's normal, sized by hand or not.
fn oblique_why(by_hand: bool) -> &'static str {
    if by_hand {
        "seen along the extension the opening is a sliver, so the taper from it is steep"
    } else {
        "the extension is narrower than the opening"
    }
}
/// Furthest an end can be aimed by hand off its cut's normal, degrees: beyond, the opening
/// seen along the extension is a sliver.
const MAX_AIM: f64 = 75.0;
/// Length of an end's arrow, in diameters of the end (its tip is the handle).
const ARROW: f64 = 1.5;
/// How near the pointer must be to an arrow's tip to take it, points.
const GRAB: f32 = 10.0;

/// `dir` (a unit vector) turned toward `normal` (unit) until it is at most `max_deg` off it.
fn within_cone(dir: DVec3, normal: DVec3, max_deg: f64) -> DVec3 {
    let max = max_deg.to_radians();
    if dir.dot(normal).clamp(-1.0, 1.0).acos() <= max {
        return dir;
    }
    // Along the great circle from the normal through `dir` (any way, if `dir` is opposite).
    let side = (dir - normal * dir.dot(normal)).try_normalize().unwrap_or_else(|| normal.any_orthonormal_vector());
    (normal * max.cos() + side * max.sin()).normalize()
}

/// Where the ray `ray` meets the sphere about `centre` of radius `r` nearest `near` (a point
/// on it), or, where the ray passes it by, the sphere's point nearest the ray: an arrow's tip
/// dragged round its end.
fn on_sphere(ray: &mm_core::Ray, centre: DVec3, r: f64, near: DVec3) -> Option<DVec3> {
    let oc = ray.origin - centre;
    let b = oc.dot(ray.dir);
    let disc = b * b - (oc.length_squared() - r * r);
    let p = if disc >= 0.0 {
        let s = disc.sqrt();
        let (a, c) = (ray.origin + ray.dir * (-b - s), ray.origin + ray.dir * (-b + s));
        if a.distance(near) <= c.distance(near) { a } else { c }
    } else {
        let closest = ray.origin + ray.dir * (-b);
        centre + (closest - centre).try_normalize()? * r
    };
    p.is_finite().then_some(p)
}

/// Which way one extension ran, and why.
#[derive(Clone, Debug)]
struct EndHeading {
    /// Index of the end.
    end: usize,
    /// Unit direction (out of the vessel).
    dir: DVec3,
    /// Angle to the cut's normal, degrees.
    off_normal: f64,
    /// Length of centerline fitted, mm (none when square to the cut).
    fitted: Option<f64>,
    /// Why the heading is not what was asked, or a caution about it.
    note: Option<String>,
    /// Aimed by hand.
    aimed: bool,
}

struct FlowOut {
    source: std::sync::Arc<Mesh>,
    mesh: Mesh,
    /// Cap groups created for previously open ends (to be named outlet_N).
    new_caps: Vec<i32>,
    /// Wall groups of the extensions (to be named extension_N).
    new_walls: Vec<i32>,
    /// Groups of caps removed and not replaced ("Cap the ends" off): their names go.
    removed: Vec<i32>,
    /// Which way each extension ran, in the order of the ends.
    heads: Vec<EndHeading>,
    msg: String,
}

/// The centerline of the vessel behind each end, by end: its points and radii from the
/// opening into the vessel, up to its first fork or [`TRACE_REACH`] radii in, traced once
/// (the object does not change while the tool is open) and shared with the builds. `Err`
/// where it could not be traced.
type Traces = std::sync::Arc<std::sync::Mutex<std::collections::HashMap<usize, Result<std::sync::Arc<Vec<(DVec3, f64)>>, String>>>>;

/// The centerline through the lumen behind end `i` of `ends` (see [`Traces`]), as the loft
/// traces an opening's, with the mesh's open ends capped by `capped` (or a capping kept there
/// used); `Ok(None)` once `going` returns false.
fn trace_behind(mesh: &Mesh, capped: &crate::centerline::CappedCache, ends: &[VesselEnd], i: usize, going: &dyn Fn(f32) -> bool) -> anyhow::Result<Option<Vec<(DVec3, f64)>>> {
    use crate::centerline::{Anchor, LumenEnd, LumenInput, LumenParams};
    let end = &ends[i];
    let radius = 0.5 * end.diameter;
    // Flat caps are not vessel wall: radii are measured to the wall.
    let cap_groups: Vec<i32> = ends.iter().filter_map(|e| if let EndKind::Cap(g) = e.kind { Some(g) } else { None }).collect();
    let input = LumenInput { inlet: LumenEnd { at: end.center, radius, anchor: Anchor::Opening }, outlets: Vec::new(), cap_groups };
    let params = LumenParams { within: Some((end.center, TRACE_REACH * radius)), ..Default::default() };
    let Some(lumen) = crate::centerline::compute_lumen(mesh, capped, &input, &params, Some(going))? else { return Ok(None) };
    let cl = lumen.centerline;
    let root = cl.branches.iter().find(|b| b.parent.is_none()).ok_or_else(|| anyhow::anyhow!("found no vessel behind the end"))?;
    let points: Vec<(DVec3, f64)> = root.nodes.iter().map(|&k| (cl.nodes[k].position, cl.nodes[k].radius)).collect();
    if points.len() < 2 {
        anyhow::bail!("the vessel behind the end is too short to follow");
    }
    Ok(Some(points))
}

/// End `e`'s heading from its centerline `points` (see [`Traces`]) fitted over `fit` of its
/// diameters: the direction out of the vessel, the stretch fitted (from the end in) and its
/// length; or why there is none (then the extension is square to the cut).
fn centerline_heading(points: &[(DVec3, f64)], e: &VesselEnd, fit: f64) -> Result<(DVec3, Vec<DVec3>, f64), String> {
    let widest = WIDEN * 0.5 * e.diameter;
    let own = points.iter().skip(1).position(|p| p.1 > widest).map_or(points.len(), |k| k + 1);
    let own: Vec<DVec3> = points[..own].iter().map(|p| p.0).collect();
    let length: f64 = own.windows(2).map(|w| w[0].distance(w[1])).sum();
    if length < MIN_FIT.min(fit) * e.diameter {
        return Err(format!("the vessel forks or widens {length:.1} mm in, too soon to follow"));
    }
    let (dir, stretch) = fit_heading(&own, fit * e.diameter).ok_or("its centerline has no length")?;
    let length = stretch.windows(2).map(|w| w[0].distance(w[1])).sum();
    Ok((dir, stretch, length))
}

/// The line through the first `reach` mm of `points` (a centerline from an end into its
/// vessel): its unit direction out of the vessel (toward the end), and the stretch fitted,
/// from the end in. The stretch is shorter where the centerline is (it stops at the vessel's
/// first fork). `None` for a centerline with no length.
fn fit_heading(points: &[DVec3], reach: f64) -> Option<(DVec3, Vec<DVec3>)> {
    if reach.is_nan() || reach <= 0.0 {
        return None;
    }
    // The stretch, cut off at `reach`.
    let mut stretch = vec![*points.first()?];
    let mut s = 0.0;
    for w in points.windows(2) {
        let step = w[0].distance(w[1]);
        if s + step >= reach {
            if step > 0.0 {
                stretch.push(w[0].lerp(w[1], (reach - s) / step));
            }
            s = reach;
            break;
        }
        s += step;
        stretch.push(w[1]);
    }
    if !s.is_finite() || s <= 0.0 || stretch.len() < 2 {
        return None;
    }
    // Least squares for x(t) = X + t D with t the distance along the stretch, sampled
    // evenly (a bend's extra points would weigh it otherwise).
    const SAMPLES: usize = 33;
    let mut samples = Vec::with_capacity(SAMPLES);
    let (mut seg, mut walked) = (0, 0.0);
    for i in 0..SAMPLES {
        let t = s * i as f64 / (SAMPLES - 1) as f64;
        while seg + 2 < stretch.len() && walked + stretch[seg].distance(stretch[seg + 1]) < t {
            walked += stretch[seg].distance(stretch[seg + 1]);
            seg += 1;
        }
        let len = stretch[seg].distance(stretch[seg + 1]);
        let f = if len > 0.0 { ((t - walked) / len).clamp(0.0, 1.0) } else { 0.0 };
        samples.push((t, stretch[seg].lerp(stretch[seg + 1], f)));
    }
    let n = samples.len() as f64;
    let tm = samples.iter().map(|x| x.0).sum::<f64>() / n;
    let xm = samples.iter().map(|x| x.1).sum::<DVec3>() / n;
    let (mut num, mut den) = (DVec3::ZERO, 0.0);
    for &(t, x) in &samples {
        num += (x - xm) * (t - tm);
        den += (t - tm) * (t - tm);
    }
    // D runs into the vessel; the extension runs the other way.
    let dir = -(num / den).normalize_or_zero();
    (den > 0.0 && dir != DVec3::ZERO && dir.is_finite()).then_some((dir, stretch))
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
    /// The vessel's centerline behind each end, as the builds trace them.
    traces: Traces,
    /// The object capped, for the traces (built by the first, kept for the rest).
    capped: crate::centerline::CappedCache,
    /// The end whose arrow's tip the button went down on, where, and whether it has moved
    /// since (only then is the end aimed).
    drag: Option<(usize, egui::Pos2, bool)>,
    /// The end whose arrow's tip is under the pointer.
    hover: Option<usize>,
    /// The panel's row of sizes for every chosen end at once.
    all: EndSize,
    /// Whether the sizes by hand have been started from the sizes from the cut (the first time
    /// they are asked for, so that switching over changes nothing until a size is).
    seeded: bool,
}

impl FlowExtensionTool {
    pub fn new(cx: &mut ToolCx) -> Self {
        let object = cx.scene.active;
        let ends = cx.scene.active().map(|o| find_vessel_ends(&o.mesh)).unwrap_or_default();
        let n = ends.len();
        let mut tool = Self {
            object,
            ends,
            params: FlowParams {
                diameters: 5.0,
                absolute: None,
                circular: true,
                cap: true,
                heading: Heading::CutNormal,
                fit: 2.0,
                selected: vec![false; n],
                aims: vec![None; n],
                by_hand: false,
                sizes: Vec::new(),
                taper: 5.0,
                barb: false,
                barb_width: 1.0,
                barb_setback: 2.0,
            },
            job: None,
            preview: None,
            error: None,
            failed: None,
            changed: None,
            traces: Default::default(),
            capped: Default::default(),
            drag: None,
            hover: None,
            all: EndSize { diameter: 5.0, length: 20.0 },
            seeded: false,
        };
        // A size for every end; started again from the cut when first asked for (the length
        // may have been set meanwhile).
        tool.seed_sizes();
        tool.seeded = false;
        tool
    }

    /// Start the sizes by hand from the sizes from the cut: each end's diameter, its length as
    /// set, and a taper of the widest end's diameter.
    fn seed_sizes(&mut self) {
        let p = &mut self.params;
        p.sizes = self.ends.iter().map(|e| EndSize { diameter: to_step(e.diameter, 0.01).max(0.1), length: to_step(p.absolute.unwrap_or(p.diameters * e.diameter), 0.1).max(0.1) }).collect();
        let widest = self.ends.iter().map(|e| e.diameter).fold(0.0, f64::max);
        p.taper = if widest > 0.0 { to_step(widest, 0.5).clamp(0.5, *SIZE_RANGE.end()) } else { 5.0 };
        let first = p.selected.iter().position(|&s| s).unwrap_or(0);
        self.all = p.sizes.get(first).copied().unwrap_or(self.all);
        self.seeded = true;
    }

    /// Size the extensions by hand (`true`), started from the sizes from the cut the first
    /// time, or from the cut.
    fn set_by_hand(&mut self, on: bool) {
        if on && !self.seeded {
            self.seed_sizes();
        }
        self.params.by_hand = on;
    }

    /// The taper of end `i`'s extension sized by hand: its length (no longer than the
    /// extension), and its angle off the extension's axis from the cut's diameter, degrees.
    fn taper_of(&self, i: usize) -> (f64, f64) {
        let s = self.params.sizes[i];
        let length = self.params.taper.min(s.length);
        (length, (0.5 * (s.diameter - self.ends[i].diameter).abs()).atan2(length).to_degrees())
    }

    /// The barb on the extensions (sized by hand, with a barb), mm.
    fn barb(&self) -> Option<mm_edit::Barb> {
        let p = &self.params;
        (p.by_hand && p.barb).then(|| mm_edit::Barb { width: p.barb_width, setback: p.barb_setback, length: barb_ramp(p.barb_width) })
    }

    /// Sized by hand: what to point out about the chosen ends' tapers and barbs, by end.
    fn size_notes(&self) -> Vec<(usize, String)> {
        if !self.params.by_hand {
            return Vec::new();
        }
        (0..self.ends.len())
            .filter(|&i| self.params.selected[i])
            .filter_map(|i| {
                let (length, angle) = self.taper_of(i);
                let mut notes = Vec::new();
                if length < self.params.taper {
                    notes.push("shorter than the taper, so it tapers all its length".to_string());
                }
                if angle > STEEP {
                    notes.push(format!("tapers steeply ({angle:.0}° off its axis): a longer taper is gentler"));
                }
                if let Some(b) = self.barb() {
                    // From the opening to the barb's crest.
                    let crest = self.params.sizes[i].length - b.setback - b.length;
                    if crest <= 0.0 {
                        notes.push(format!(
                            "too short for the barb ({:.2} mm back from the end and {:.2} mm of ramp): lengthen it or bring the barb nearer the end",
                            b.setback, b.length
                        ));
                    } else if crest < length {
                        notes.push(format!("the barb's back is on the taper ({crest:.2} mm from the opening, the taper {length:.2} mm): it stands on a slope"));
                    }
                }
                (!notes.is_empty()).then(|| (i, notes.join("; ")))
            })
            .collect()
    }

    /// The panel's sizes by hand: a diameter and length for each chosen end (and a row
    /// setting them all), and the taper.
    fn sizes_ui(&mut self, ui: &mut egui::Ui) {
        let chosen: Vec<usize> = (0..self.ends.len()).filter(|&i| self.params.selected[i]).collect();
        fn drag(v: &mut f64, speed: f64) -> egui::DragValue<'_> {
            egui::DragValue::new(v).speed(speed).range(SIZE_RANGE).min_decimals(2).max_decimals(3)
        }
        if chosen.is_empty() {
            ui.weak("Choose ends to size their extensions.");
        } else {
            egui::Grid::new("flow_extension_sizes").num_columns(3).spacing([10.0, 4.0]).show(ui, |ui| {
                ui.label("");
                ui.label("⌀ mm");
                ui.label("length mm");
                ui.end_row();
                if chosen.len() > 1 {
                    ui.label("All").on_hover_text("Set every chosen end at once");
                    if ui.add(drag(&mut self.all.diameter, 0.05)).changed() {
                        chosen.iter().for_each(|&i| self.params.sizes[i].diameter = self.all.diameter);
                    }
                    if ui.add(drag(&mut self.all.length, 0.1)).changed() {
                        chosen.iter().for_each(|&i| self.params.sizes[i].length = self.all.length);
                    }
                    ui.end_row();
                }
                for &i in &chosen {
                    ui.label(format!("End {}", i + 1)).on_hover_text(format!("Its cut is ⌀ {:.2} mm", self.ends[i].diameter));
                    ui.add(drag(&mut self.params.sizes[i].diameter, 0.05));
                    ui.add(drag(&mut self.params.sizes[i].length, 0.1));
                    ui.end_row();
                }
            });
        }
        ui.horizontal(|ui| {
            ui.add(drag(&mut self.params.taper, 0.1).suffix(" mm"));
            ui.label("taper").on_hover_text("How much of each extension tapers from its opening to its diameter; the rest is a straight cylinder of that diameter");
        });
        ui.checkbox(&mut self.params.barb, "Barb")
            .on_hover_text("A barb near each extension's end for its hose to grip: it ramps out from the extension, then steps straight back down");
        if self.params.barb {
            let crests = chosen.iter().map(|&i| format!("End {}: ⌀ {:.2} mm", i + 1, self.params.sizes[i].diameter + self.params.barb_width)).collect::<Vec<_>>().join("\n");
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut self.params.barb_width).speed(0.01).range(BARB_WIDTH_RANGE).min_decimals(2).max_decimals(3).suffix(" mm"));
                ui.label("barb width").on_hover_text(format!("How much wider the barb's crest is than the extension, on the diameter. Across the crest:\n{crests}"));
            });
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut self.params.barb_setback).speed(0.05).range(BARB_SETBACK_RANGE).min_decimals(2).max_decimals(3).suffix(" mm"));
                ui.label("back from the end").on_hover_text("How far back from the extension's end the barb starts to rise (0: at the end)");
            });
            ui.weak(format!("It ramps out over {:.2} mm ({BARB_RAMP:.0}° off the axis), so its crest is {:.2} mm back from the end.", barb_ramp(self.params.barb_width), self.params.barb_setback + barb_ramp(self.params.barb_width)));
        }
        for (i, note) in self.size_notes() {
            ui.colored_label(egui::Color32::from_rgb(230, 170, 70), format!("End {}: {note}", i + 1));
        }
    }

    /// Which way end `i`'s extension runs with the current settings: its aim, or else its
    /// heading.
    fn direction(&self, i: usize) -> DVec3 {
        self.params.aims[i].unwrap_or_else(|| self.heading_of(i))
    }

    /// Which way end `i`'s extension runs unless aimed by hand: along its centerline once
    /// traced (with [`Heading::Centerline`]), else square to its cut.
    fn heading_of(&self, i: usize) -> DVec3 {
        match self.params.heading {
            Heading::Centerline => self.fitted(i).map_or(self.ends[i].normal, |(d, _)| d),
            Heading::CutNormal => self.ends[i].normal,
        }
    }

    /// The tip of end `i`'s arrow.
    fn tip(&self, i: usize) -> DVec3 {
        self.ends[i].center + self.direction(i) * ARROW * self.ends[i].diameter
    }

    /// The chosen end whose arrow's tip is nearest `pos`, within [`GRAB`] points.
    fn tip_under(&self, cx: &ToolCx, pos: egui::Pos2) -> Option<usize> {
        (0..self.ends.len())
            .filter(|&i| self.params.selected[i])
            .filter_map(|i| cx.project(self.tip(i)).map(|p| (i, p.distance(pos))))
            .filter(|&(_, d)| d < GRAB)
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(i, _)| i)
    }

    /// Aim end `i` along `dir` (turned back to within [`MAX_AIM`] of its cut's normal, which
    /// is returned with the angle to it), or let it run its heading again (`None`).
    fn aim(&mut self, i: usize, dir: Option<DVec3>) -> Option<(DVec3, f64)> {
        let n = self.ends[i].normal;
        let aim = dir.and_then(|d| d.try_normalize()).map(|d| within_cone(d, n, MAX_AIM));
        self.params.aims[i] = aim;
        aim.map(|d| (d, d.dot(n).clamp(-1.0, 1.0).acos().to_degrees()))
    }
    /// The heading fitted to end `i`'s centerline with the current settings, and the stretch
    /// fitted, if that end has been traced.
    fn fitted(&self, i: usize) -> Option<(DVec3, Vec<DVec3>)> {
        let traces = self.traces.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let points = traces.get(&i)?.as_ref().ok()?.clone();
        drop(traces);
        centerline_heading(&points, &self.ends[i], self.params.fit).ok().map(|(dir, stretch, _)| (dir, stretch))
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
        if self.drag.is_some_and(|d| d.2) {
            // An arrow held: built once let go (and settled), not whenever it pauses.
            self.changed = Some((self.params.clone(), now));
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
        let ends: Vec<(usize, VesselEnd)> = self.ends.iter().cloned().enumerate().filter(|(i, _)| self.params.selected[*i]).collect();
        let (all_ends, traces, capped) = (self.ends.clone(), self.traces.clone(), self.capped.clone());
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
        let barb = self.barb();
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
                let (mut new_caps, mut new_walls, mut removed, mut heads) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
                // Each end's heading first, from the object as it was (tracing a vessel
                // takes most of the time: each end is traced once, then kept).
                let along = p2.heading == Heading::Centerline;
                let mut dirs: Vec<Option<DVec3>> = vec![None; n];
                for (k, (i, e)) in ends.iter().enumerate() {
                    let at = |x: f32| progress(if along { 0.8 * (k as f32 + x) / n as f32 } else { 0.0 });
                    if !at(0.0) {
                        anyhow::bail!("cancelled");
                    }
                    let square = e.normal.normalize_or_zero();
                    let mut head = EndHeading { end: *i, dir: square, off_normal: 0.0, fitted: None, note: None, aimed: false };
                    if let Some(aim) = p2.aims[*i] {
                        head.dir = aim;
                        head.aimed = true;
                        head.off_normal = aim.dot(square).clamp(-1.0, 1.0).acos().to_degrees();
                        if head.off_normal > OBLIQUE {
                            head.note = Some(format!("aimed far off the cut's normal: {}", oblique_why(p2.by_hand)));
                        }
                        dirs[k] = Some(aim);
                    } else if along {
                        let kept = traces.lock().unwrap_or_else(std::sync::PoisonError::into_inner).get(i).cloned();
                        let trace = match kept {
                            Some(t) => t,
                            None => {
                                let t = match trace_behind(&src, &capped, &all_ends, *i, &at) {
                                    Ok(Some(points)) => Ok(std::sync::Arc::new(points)),
                                    Ok(None) => anyhow::bail!("cancelled"),
                                    Err(err) => Err(format!("{err:#}")),
                                };
                                traces.lock().unwrap_or_else(std::sync::PoisonError::into_inner).insert(*i, t.clone());
                                t
                            }
                        };
                        match trace.map_err(|err| format!("no centerline ({err})")).and_then(|points| centerline_heading(&points, e, p2.fit)) {
                            Ok((dir, _, fitted)) => {
                                head.off_normal = dir.dot(square).abs().clamp(0.0, 1.0).acos().to_degrees();
                                head.dir = dir;
                                head.fitted = Some(fitted);
                                let mut notes = Vec::new();
                                if fitted < 0.98 * p2.fit * e.diameter {
                                    notes.push(format!("fitted to the {fitted:.1} mm of centerline before the vessel forks or widens"));
                                }
                                if head.off_normal > OBLIQUE {
                                    notes.push("far off the cut's normal: check that the centerline follows this vessel".to_string());
                                }
                                head.note = (!notes.is_empty()).then(|| notes.join("; "));
                                dirs[k] = Some(dir);
                            }
                            Err(why) => head.note = Some(format!("{why}: square to the cut instead")),
                        }
                    }
                    heads.push(head);
                }
                for (k, (i, e)) in ends.iter().enumerate() {
                    if !progress(if along { 0.8 + 0.2 * k as f32 / n as f32 } else { k as f32 / n as f32 }) {
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
                        for j in 0..lp.len() {
                            s += m.vertex(lp[j]).distance(m.vertex(lp[(j + 1) % lp.len()]));
                        }
                        s / lp.len() as f64
                    };
                    let mut params = mm_edit::FlowExtensionParams { direction: dirs[k], make_circular: p2.circular, cap: p2.cap, ..Default::default() };
                    let length = if p2.by_hand {
                        // Tapered from the opening to the diameter, then straight.
                        let s = p2.sizes[*i];
                        params.length = mm_edit::ExtensionLength::Absolute(s.length);
                        params.diameter = Some(s.diameter);
                        params.transition = (p2.taper / s.length).min(1.0);
                        params.barb = barb;
                        s.length
                    } else {
                        params.length = match p2.absolute {
                            Some(a) => mm_edit::ExtensionLength::Absolute(a),
                            None => mm_edit::ExtensionLength::Diameters(p2.diameters),
                        };
                        p2.absolute.unwrap_or(p2.diameters * e.diameter)
                    };
                    let mut segments = ((length / (2.0 * edge)).ceil() as usize).clamp(2, 150);
                    if p2.by_hand {
                        // Rings enough for the taper's curve, short as it may be.
                        segments = segments.max((TAPER_RINGS / params.transition).ceil() as usize).min(150);
                    }
                    params.segments = segments;
                    let r = mm_edit::extrude_boundary_loop(&mut m, &lp, &params).map_err(|err| anyhow::anyhow!("end {}: {err}", ends[k].0 + 1))?;
                    if dirs[k].is_none() {
                        // Square to the cut: the side the extrusion took (the loop's own).
                        heads[k].dir = r.direction;
                    }
                    match (keep_group, r.cap_group) {
                        (Some(g), Some(_)) => {
                            for &t in &r.cap_triangles {
                                m.set_tri_group(t, g);
                            }
                        }
                        (None, Some(g)) => new_caps.push(g),
                        (Some(g), None) => removed.push(g),
                        (None, None) => {}
                    }
                    // The wall keeps a group of its own (its own patch, apart from the vessel
                    // wall). Its id was free when built, so it can be the removed cap's.
                    let mut wall = r.wall_group;
                    if keep_group == Some(wall) {
                        wall = m.max_group_id() + 1;
                        for &t in &r.wall_triangles {
                            m.set_tri_group(t, wall);
                        }
                    }
                    new_walls.push(wall);
                }
                m.bump_shape_version();
                Ok(FlowOut { source, mesh: m, new_caps, new_walls, removed, heads, msg: format!("extended {n} end(s)") })
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
        let mut by_hand = self.params.by_hand;
        ui.horizontal(|ui| {
            ui.selectable_value(&mut by_hand, false, "From the cut").on_hover_text("Each extension as wide as its cut, as long as set below");
            ui.selectable_value(&mut by_hand, true, "Set by hand").on_hover_text("Each extension a diameter and length of your own, in mm (a hose to fit), tapering to it from the opening");
        });
        if by_hand != self.params.by_hand {
            self.set_by_hand(by_hand);
        }
        if self.params.by_hand {
            self.sizes_ui(ui);
        } else {
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
        }
        ui.checkbox(&mut self.params.cap, "Cap the ends");
        ui_util::section(ui, "Direction");
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.params.heading, Heading::CutNormal, "Cut normal").on_hover_text("Square to each end: along the normal of its cut or cap");
            ui.selectable_value(&mut self.params.heading, Heading::Centerline, "Centerline")
                .on_hover_text("On along the vessel: a line fitted to its centerline just inside each end, so an end cut at a slant still runs straight on");
        });
        if self.params.heading == Heading::Centerline {
            ui.add(egui::Slider::new(&mut self.params.fit, 0.5..=MAX_FIT).text("× diameter fitted"))
                .on_hover_text("How much of the vessel the direction follows: a short stretch is the vessel's heading at the end, a long one its overall course (up to where it forks)");
        }
        ui.small("Drag the tip of an end's arrow to aim it by hand; double-click the tip to let it go.");
        let aimed: Vec<usize> = (0..self.ends.len()).filter(|&i| self.params.selected[i] && self.params.aims[i].is_some()).collect();
        for &i in &aimed {
            let off = self.params.aims[i].map_or(0.0, |d| d.dot(self.ends[i].normal).clamp(-1.0, 1.0).acos().to_degrees());
            ui.horizontal(|ui| {
                let text = format!("End {}: aimed by hand, {off:.0}° off the cut normal", i + 1);
                if off > OBLIQUE {
                    let why = oblique_why(self.params.by_hand);
                    ui.colored_label(egui::Color32::from_rgb(230, 170, 70), text).on_hover_text(format!("{}{}", why[..1].to_uppercase(), &why[1..]));
                } else {
                    ui.label(text);
                }
                if ui.small_button("Reset").on_hover_text("Run this end's heading again").clicked() {
                    self.aim(i, None);
                }
            });
        }
        if aimed.len() > 1 && ui.small_button("Reset all").clicked() {
            self.params.aims.iter_mut().for_each(|a| *a = None);
        }
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
        if let Some((_, out)) = self.preview.as_ref().filter(|(p, _)| *p == self.params && p.heading == Heading::Centerline) {
            for h in out.heads.iter().filter(|h| !h.aimed) {
                let angle = h.fitted.map(|_| format!("{:.0}° off the cut normal", h.off_normal));
                match (angle, &h.note) {
                    (Some(a), None) => {
                        ui.weak(format!("End {}: {a}", h.end + 1));
                    }
                    (a, Some(note)) => {
                        let text = a.map_or(format!("End {}: {note}", h.end + 1), |a| format!("End {}: {a}; {note}", h.end + 1));
                        ui.colored_label(egui::Color32::from_rgb(230, 170, 70), text);
                    }
                    (None, None) => {}
                }
            }
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
        let Some(pos) = input.pos else { return };
        if self.drag.is_none() {
            self.hover = self.tip_under(cx, pos);
        }
        if input.pressed {
            self.drag = self.tip_under(cx, pos).map(|i| (i, pos, false));
        }
        if input.down {
            if let Some((i, from, moved)) = self.drag {
                // A click on the tip leaves the end as it was: it is aimed once the pointer moves.
                if moved || from.distance(pos) > 3.0 {
                    self.drag = Some((i, from, true));
                    let e = &self.ends[i];
                    if let Some(p) = on_sphere(&cx.ray_at(pos), e.center, ARROW * e.diameter, self.tip(i)) {
                        self.aim(i, Some(p - e.center));
                    }
                }
            }
        }
        if input.released {
            let held = self.drag.take().map(|d| d.0);
            if input.double_clicked {
                if let Some(i) = held.or_else(|| self.tip_under(cx, pos)) {
                    self.aim(i, None);
                }
            } else if input.clicked && held.is_none() {
                // (A click on an arrow's tip is not one on the marker under it.)
                if let Some(i) = end_under_cursor(cx, &self.ends, pos) {
                    self.params.selected[i] = !self.params.selected[i];
                    self.start(cx);
                }
            }
        }
    }

    fn paint(&mut self, painter: &egui::Painter, cx: &mut ToolCx) {
        // Each chosen end's arrow, its tip a handle to aim it by; along a centerline, the
        // stretch its heading is fitted to too (drawn on top: they lie inside the vessel).
        for i in (0..self.ends.len()).filter(|&i| self.params.selected[i]) {
            let e = &self.ends[i];
            if self.params.heading == Heading::Centerline {
                if let Some((_, stretch)) = self.fitted(i) {
                    let pts: Vec<egui::Pos2> = stretch.iter().filter_map(|&p| cx.project(p)).collect();
                    painter.add(egui::Shape::line(pts, egui::Stroke::new(2.5, egui::Color32::from_rgb(255, 200, 60))));
                }
            }
            let Some(a) = cx.project(e.center) else { continue };
            let aimed = self.params.aims[i].is_some();
            if aimed {
                // Where it would run unaimed, for reference.
                if let Some(b) = cx.project(e.center + self.heading_of(i) * ARROW * e.diameter) {
                    painter.line_segment([a, b], egui::Stroke::new(1.0, egui::Color32::from_white_alpha(110)));
                }
            }
            let Some(b) = cx.project(self.tip(i)) else { continue };
            let hot = self.drag.map(|d| d.0) == Some(i) || (self.drag.is_none() && self.hover == Some(i));
            let colour = if aimed { ui_util::ACCENT } else { egui::Color32::WHITE };
            painter.arrow(a, b - a, egui::Stroke::new(if hot { 3.0 } else { 2.0 }, colour));
            painter.circle(b, if hot { 7.0 } else { 5.0 }, colour, egui::Stroke::new(1.5, egui::Color32::from_gray(30)));
        }
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
        for g in out.removed {
            o.group_names.remove(&g);
        }
        for g in out.new_caps {
            let n = next_name(&o.group_names, "outlet");
            o.group_names.insert(g, n);
        }
        for g in out.new_walls {
            let n = next_name(&o.group_names, "extension");
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
        let direction = match p.heading {
            Heading::CutNormal => "cut_normal",
            Heading::Centerline => "centerline",
        };
        Some(json!({"length_in_diameters": p.diameters, "length_mm": p.absolute, "circular": p.circular, "cap": p.cap, "direction": direction, "fit_diameters": p.fit, "sized_by_hand": p.by_hand, "taper_mm": p.taper, "barb": p.barb, "barb_width_mm": p.barb_width, "barb_setback_mm": p.barb_setback}))
    }

    fn settings_schema(&self) -> Option<Value> {
        Some(json!({"type": "object", "properties": {
            "length_in_diameters": {"type": "number", "description": "Extension length as a multiple of each end's diameter (0.5..20), used unless length_mm is set (not sized by hand)"},
            "length_mm": {"type": ["number", "null"], "description": "Extension length in mm for every end (0.1..200); null to use length_in_diameters (not sized by hand)"},
            "circular": {"type": "boolean", "description": "Blend each extension to a circular cross-section (not sized by hand: those always end circular)"},
            "sized_by_hand": {"type": "boolean", "description": "Each extension a diameter and length of its own in mm (the action `size`; e.g. a hose to fit), tapering to that diameter from the opening, instead of as wide as its cut. Started from the cut's diameters and the length set, the first time"},
            "taper_mm": {"type": "number", "description": "Sized by hand: how much of each extension tapers from its opening to its diameter, mm (0.1..200; all of a shorter one); the rest is a straight cylinder"},
            "barb": {"type": "boolean", "description": "Sized by hand: a barb near each extension's end for its hose to grip, ramping out (15° off the axis) and stepping straight back down"},
            "barb_width_mm": {"type": "number", "description": "How much wider the barb's crest is than its extension, on the diameter, mm (0.01..50)"},
            "barb_setback_mm": {"type": "number", "description": "From each extension's end back to where its barb starts to rise, mm (0..200; 0 = at the end)"},
            "cap": {"type": "boolean", "description": "Cap the extensions' ends"},
            "direction": {"type": "string", "enum": ["cut_normal", "centerline"], "description": "Which way the extensions run: square to each end's cut, or on along the vessel (a line fitted to its centerline just inside the end)"},
            "fit_diameters": {"type": "number", "description": "With direction centerline: the stretch of centerline inside each end the direction is fitted to, in diameters of the end (0.5..5; shorter where the vessel forks sooner)"},
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
        p.heading = match v["direction"].as_str() {
            Some("cut_normal") => Heading::CutNormal,
            Some("centerline") => Heading::Centerline,
            _ => return Err("direction must be \"cut_normal\" or \"centerline\"".into()),
        };
        p.fit = v["fit_diameters"].as_f64().ok_or("fit_diameters must be a number")?.clamp(0.5, MAX_FIT);
        p.taper = v["taper_mm"].as_f64().ok_or("taper_mm must be a number")?.clamp(*SIZE_RANGE.start(), *SIZE_RANGE.end());
        p.barb = v["barb"].as_bool().ok_or("barb must be true or false")?;
        p.barb_width = v["barb_width_mm"].as_f64().filter(|x| x.is_finite()).ok_or("barb_width_mm must be a number")?.clamp(*BARB_WIDTH_RANGE.start(), *BARB_WIDTH_RANGE.end());
        p.barb_setback = v["barb_setback_mm"].as_f64().filter(|x| x.is_finite()).ok_or("barb_setback_mm must be a number")?.clamp(*BARB_SETBACK_RANGE.start(), *BARB_SETBACK_RANGE.end());
        let by_hand = v["sized_by_hand"].as_bool().ok_or("sized_by_hand must be true or false")?;
        let taper = p.taper;
        self.set_by_hand(by_hand);
        if changes.get("taper_mm").is_some() {
            // (Not the seeded one.)
            self.params.taper = taper;
        }
        self.start(cx);
        Ok(())
    }

    fn state(&self, _scene: &Scene) -> Value {
        let current = self.preview.as_ref().filter(|(p, _)| *p == self.params);
        let heads: Option<Vec<Value>> = current.map(|(_, out)| {
            out.heads
                .iter()
                .map(|h| json!({"end": h.end + 1, "direction": vec3(h.dir.to_array(), 4), "off_cut_normal_deg": round(h.off_normal, 1), "aimed_by_hand": h.aimed, "centerline_fitted_mm": h.fitted.map(|x| round(x, 2)), "note": h.note}))
                .collect()
        });
        json!({
            "object": self.object,
            "ends": ends_state(&self.ends, |i| self.params.selected[i]),
            "computing": self.job.is_some(),
            "preview": current.map(|(_, out)| out.msg.clone()),
            "directions": heads,
            "sized_by_hand": self.params.by_hand,
            "sizes": self.params.by_hand.then(|| (0..self.ends.len()).filter(|&i| self.params.selected[i]).map(|i| {
                let (s, (taper, angle)) = (self.params.sizes[i], self.taper_of(i));
                let barb = self.barb().map(|b| json!({"crest_diameter_mm": round(s.diameter + b.width, 3), "starts_from_end_mm": round(b.setback, 3), "crest_from_end_mm": round(b.setback + b.length, 3), "ramp_mm": round(b.length, 3)}));
                json!({"end": i + 1, "diameter_mm": round(s.diameter, 3), "length_mm": round(s.length, 3), "taper_mm": round(taper, 3), "taper_off_axis_deg": round(angle, 1), "cut_diameter_mm": round(self.ends[i].diameter, 3), "barb": barb})
            }).collect::<Vec<_>>()),
            "size_notes": self.size_notes().into_iter().map(|(i, note)| json!({"end": i + 1, "note": note})).collect::<Vec<_>>(),
            "aims": (0..self.ends.len()).filter_map(|i| self.params.aims[i].map(|d| json!({"end": i + 1, "direction": vec3(d.to_array(), 4), "off_cut_normal_deg": round(d.dot(self.ends[i].normal).clamp(-1.0, 1.0).acos().to_degrees(), 1), "picked": self.params.selected[i]}))).collect::<Vec<_>>(),
            "error": self.error,
            "next": if !self.params.selected.contains(&true) { "select the ends to extend" } else if current.is_some() { "accept (mesh_tool accept) to keep the extensions" } else { "wait for the preview (app_wait)" },
        })
    }

    fn actions(&self) -> Vec<Action> {
        vec![
            Action::new(
                "select",
                "Choose the ends to extend (numbers from the state's ends, as the markers show): replaces the choice unless `add`",
                json!({"ends": ends_arg("End numbers (from 1)"), "add": arg::boolean("Add to the ends already chosen")}),
            )
            .required(&["ends"]),
            Action::new(
                "aim",
                "Aim one end's extension by hand, as dragging its arrow does (within 75° of its outward cut normal, the state's ends' `outward`), or without `direction` let it run its heading again",
                json!({"end": arg::integer("End number (from 1)"), "direction": arg::point("Direction out of the vessel (any length); leave out to undo the aim")}),
            )
            .required(&["end"]),
            Action::new(
                "size",
                "Size ends' extensions by hand (e.g. a hose to fit): a diameter and/or length in mm for each end listed, tapering from its opening over taper_mm. Turns sized_by_hand on",
                json!({"ends": ends_arg("End numbers (from 1)"), "diameter_mm": arg::number("Diameter of their extensions, mm (0.1..200)"), "length_mm": arg::number("Length of their extensions, the taper included, mm (0.1..200)")}),
            )
            .required(&["ends"]),
        ]
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
            "aim" => {
                let a = Args(args);
                let k = a.usize("end")?;
                if k == 0 || k > self.ends.len() {
                    return Err(format!("{k} is not an end number (1 to {})", self.ends.len()));
                }
                let dir = match a.opt_point("direction") {
                    Some(d) => Some(DVec3::from_array(d).try_normalize().ok_or("direction cannot be zero")?),
                    None => None,
                };
                let aimed = self.aim(k - 1, dir);
                self.start(cx);
                let picked = if self.params.selected[k - 1] { "" } else { " (it is not chosen: select it to extend it)" };
                Ok(match (dir, aimed) {
                    (Some(d), Some((got, off))) if got.dot(d) < 1.0 - 1e-9 => format!("End {k} aimed {off:.1}° off its cut normal, as near the direction asked as it can be{picked}"),
                    (_, Some((_, off))) => format!("End {k} aimed {off:.1}° off its cut normal{picked}"),
                    _ => format!("End {k} runs its heading again{picked}"),
                })
            }
            "size" => {
                let a = Args(args);
                let picked = end_numbers(args, self.ends.len())?;
                let value = |k: &str| -> Result<Option<f64>, String> {
                    if !a.has(k) {
                        return Ok(None);
                    }
                    match a.opt_f64(k) {
                        Some(x) if x.is_finite() && x > 0.0 => Ok(Some(x.clamp(*SIZE_RANGE.start(), *SIZE_RANGE.end()))),
                        _ => Err(format!("`{k}` must be a positive number of mm")),
                    }
                };
                let (diameter, length) = (value("diameter_mm")?, value("length_mm")?);
                if diameter.is_none() && length.is_none() {
                    return Err("give diameter_mm, length_mm or both".into());
                }
                let was = self.params.by_hand;
                self.set_by_hand(true);
                for &i in &picked {
                    let s = &mut self.params.sizes[i];
                    s.diameter = diameter.unwrap_or(s.diameter);
                    s.length = length.unwrap_or(s.length);
                }
                self.start(cx);
                let list = |v: &[usize]| v.iter().map(|i| (i + 1).to_string()).collect::<Vec<_>>().join(", ");
                let sized = picked.iter().map(|&i| format!("end {}: ⌀ {:.3} mm, {:.3} mm long", i + 1, self.params.sizes[i].diameter, self.params.sizes[i].length)).collect::<Vec<_>>().join("; ");
                let unpicked: Vec<usize> = picked.iter().copied().filter(|&i| !self.params.selected[i]).collect();
                let mut reply = format!("Sized {sized}");
                if !was {
                    reply += " (sized by hand now: the other ends from their cuts' diameters)";
                }
                if !unpicked.is_empty() {
                    reply += &format!("; end(s) {} not chosen: select them to extend them", list(&unpicked));
                }
                Ok(reply)
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
    fn flow_extension_walls_get_named_groups_of_their_own() {
        let ctx = egui::Context::default();
        let vp = Viewport { rect: egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0)) };
        let (mut camera, mut status) = (crate::camera::Camera::default(), String::new());
        // Extend both ends of a capped vessel, capping the extensions (`cap`) or not.
        let extend = |cap: bool, camera: &mut crate::camera::Camera, status: &mut String| {
            let (mut scene, mut history) = (vessel_scene(), crate::scene::History::default());
            let mut cx = ToolCx { scene: &mut scene, history: &mut history, camera, status, egui: &ctx, vp, brush_scroll: crate::navigation::BrushScroll::Shift };
            CapTool::new(&mut cx).act("cap_all", &json!({}), &mut cx).unwrap();
            let caps = cx.scene.active().unwrap().group_names.clone();
            let mut tool = FlowExtensionTool::new(&mut cx);
            assert!(tool.ends.iter().all(|e| matches!(e.kind, EndKind::Cap(_))));
            tool.params.selected = vec![true; tool.ends.len()];
            tool.params.cap = cap;
            tool.start(&mut cx);
            let t0 = Instant::now();
            while tool.preview.is_none() && t0.elapsed() < Duration::from_secs(30) {
                std::thread::sleep(Duration::from_millis(5));
                tool.update(&mut cx);
            }
            assert!(tool.accept(&mut cx), "{:?}", tool.error);
            let o = cx.scene.active().unwrap();
            (caps, o.group_names.clone(), crate::tools::patches::group_stats(&o.mesh).keys().copied().collect::<Vec<i32>>(), find_vessel_ends(&o.mesh))
        };
        let (caps, names, groups, ends) = extend(true, &mut camera, &mut status);
        assert_eq!(caps.len(), 2);
        // The vessel wall, two extension walls and the two caps, which keep their groups and names.
        assert_eq!(groups.len(), 5, "{groups:?}");
        for (g, n) in &caps {
            assert_eq!(names.get(g), Some(n));
        }
        let walls: Vec<i32> = groups.iter().copied().filter(|g| *g != 0 && !caps.contains_key(g)).collect();
        let mut wall_names: Vec<&str> = walls.iter().map(|g| names[g].as_str()).collect();
        wall_names.sort();
        assert_eq!(wall_names, ["extension_1", "extension_2"]);
        // Only the caps are taken for ends: the extension walls are tubes, not flat.
        let mut cap_ends: Vec<i32> = ends.iter().filter_map(|e| if let EndKind::Cap(g) = e.kind { Some(g) } else { None }).collect();
        cap_ends.sort();
        let mut cap_groups: Vec<i32> = caps.keys().copied().collect();
        cap_groups.sort();
        assert_eq!(cap_ends, cap_groups);
        // Not capped: the old caps' names go with their triangles, and no wall takes them over.
        let (caps, names, groups, _) = extend(false, &mut camera, &mut status);
        assert!(caps.keys().all(|g| !names.contains_key(g) && !groups.contains(g)), "{names:?} {groups:?}");
        assert_eq!(groups.len(), 3, "{groups:?}");
        let mut named: Vec<&str> = names.iter().filter(|(g, _)| **g != 0).map(|(_, n)| n.as_str()).collect();
        named.sort();
        assert_eq!(named, ["extension_1", "extension_2"]);
    }

    #[test]
    fn heading_is_the_line_through_the_stretch_asked_for() {
        // From an end at the origin into a vessel along -z, bending toward +x 6 mm in.
        let mut pts: Vec<DVec3> = (0..=12).map(|i| DVec3::new(0.0, 0.0, -0.5 * i as f64)).collect();
        pts.extend((1..=12).map(|i| DVec3::new(0.5 * i as f64, 0.0, -6.0)));
        let (dir, stretch) = fit_heading(&pts, 4.0).unwrap();
        assert!(dir.dot(DVec3::Z) > 1.0 - 1e-12, "{dir}");
        let len: f64 = stretch.windows(2).map(|w| w[0].distance(w[1])).sum();
        assert!((len - 4.0).abs() < 1e-12 && stretch[0] == DVec3::ZERO && (stretch.last().unwrap().z + 4.0).abs() < 1e-12);
        // Past the bend the line leans toward where the vessel goes (outward: away from it).
        let (dir, _) = fit_heading(&pts, 10.0).unwrap();
        assert!(dir.z > 0.5 && dir.x < -0.1, "{dir}");
        // Longer than the centerline: all of it.
        let (_, stretch) = fit_heading(&pts, 100.0).unwrap();
        assert_eq!(stretch.len(), pts.len());
        assert!(fit_heading(&pts, 0.0).is_none() && fit_heading(&pts[..1], 1.0).is_none());
    }

    #[test]
    fn heading_follows_the_vessel_only_until_it_widens() {
        // An end of diameter 2 on a branch running in along -z for 1.5 mm, then into a trunk
        // (radius 3) along +x: only the branch counts, however much is asked for.
        let e = VesselEnd { kind: EndKind::Open, center: DVec3::ZERO, normal: DVec3::Z, diameter: 2.0, planar: true };
        let mut pts: Vec<(DVec3, f64)> = (0..=6).map(|i| (DVec3::new(0.0, 0.0, -0.25 * i as f64), 1.0 + 0.05 * i as f64)).collect();
        pts.extend((1..=20).map(|i| (DVec3::new(0.5 * i as f64, 0.0, -1.5 - 0.1 * i as f64), 3.0)));
        let (dir, _, length) = centerline_heading(&pts, &e, 3.0).unwrap();
        assert!(dir.dot(DVec3::Z) > 1.0 - 1e-12 && (length - 1.5).abs() < 1e-12, "{dir} {length}");
        // Narrower ends, the same: the vessel widens too soon (under half a diameter in) for
        // a heading, and the extension stays square to the cut.
        let narrow = VesselEnd { diameter: 1.0, ..e };
        assert!(centerline_heading(&pts, &narrow, 3.0).unwrap_err().contains("too soon"));
    }

    #[test]
    fn flow_extensions_run_on_along_the_vessel_from_a_slanted_cut() {
        // A straight vessel along z, radius 1, its top end cut 25° off square.
        let slant = 25f64.to_radians();
        let mut scene = Scene::default();
        let path: Vec<DVec3> = (0..=40).map(|i| DVec3::new(0.0, 0.0, 0.5 * i as f64)).collect();
        let mut m = primitives::tube_along(&path, &[1.0; 41], 48);
        let plane = mm_core::Plane::from_point_normal(DVec3::new(0.0, 0.0, 15.0), DVec3::new(slant.sin(), 0.0, slant.cos()));
        mm_edit::plane_cut(&mut m, plane, &mm_edit::PlaneCutParams { keep: mm_edit::KeepSide::Below, fill: mm_edit::CutFill::None, ..Default::default() }).unwrap();
        scene.add("vessel", m);
        let (mut history, mut camera, mut status) = (crate::scene::History::default(), crate::camera::Camera::default(), String::new());
        let ctx = egui::Context::default();
        let vp = Viewport { rect: egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0)) };
        let mut cx = ToolCx { scene: &mut scene, history: &mut history, camera: &mut camera, status: &mut status, egui: &ctx, vp, brush_scroll: crate::navigation::BrushScroll::Shift };
        let mut tool = FlowExtensionTool::new(&mut cx);
        let top = tool.ends.iter().position(|e| e.center.z > 10.0).expect("the slanted end");
        let extend = |tool: &mut FlowExtensionTool, cx: &mut ToolCx, heading: &str| {
            tool.set_settings(&json!({"direction": heading, "fit_diameters": 2.0, "length_in_diameters": 4.0}), cx).unwrap();
            tool.act("select", &json!({"ends": [top + 1]}), cx).unwrap();
            let t0 = Instant::now();
            while tool.preview.as_ref().is_none_or(|(p, _)| *p != tool.params) && tool.error.is_none() && t0.elapsed() < Duration::from_secs(60) {
                std::thread::sleep(Duration::from_millis(5));
                tool.update(cx);
            }
            let (_, out) = tool.preview.as_ref().unwrap_or_else(|| panic!("{:?}", tool.error));
            // The centre of the extension's far end.
            let far: Vec<DVec3> = out.mesh.vertex_ids().map(|v| out.mesh.vertex(v)).filter(|p| p.z > 18.0).collect();
            let tip = far.iter().copied().sum::<DVec3>() / far.len().max(1) as f64;
            (out.heads.clone(), tip)
        };
        // Square to the cut: 25° off the vessel, the far end well off its axis.
        let (heads, tip) = extend(&mut tool, &mut cx, "cut_normal");
        assert_eq!(heads.len(), 1);
        assert!((heads[0].dir.angle_between(DVec3::Z).to_degrees() - 25.0).abs() < 1.0, "{:?}", heads[0]);
        assert!(tip.x > 2.0, "{tip}");
        // Along the centerline: on along the vessel, 25° off the cut's normal, its far end on
        // the vessel's axis.
        let (heads, tip) = extend(&mut tool, &mut cx, "centerline");
        let h = &heads[0];
        assert!(h.dir.angle_between(DVec3::Z).to_degrees() < 2.0, "{h:?}");
        assert!((h.off_normal - 25.0).abs() < 2.0 && h.note.is_none(), "{h:?}");
        assert!((h.fitted.unwrap() - 2.0 * tool.ends[top].diameter).abs() < 1e-6, "{h:?}");
        assert!(tip.truncate().length() < 0.15, "{tip}");
        // Painted from the trace kept for the end, which the next build uses.
        assert!(tool.fitted(top).is_some() && tool.traces.lock().unwrap().len() == 1);
        let state = tool.state(cx.scene);
        assert_eq!(state["directions"][0]["end"], json!(top + 1));
    }

    #[test]
    fn arrow_tips_turn_on_their_sphere_and_stay_in_the_cone() {
        let n = DVec3::Z;
        // Turned back to 75° in its own plane; within it, as it is; straight back, still 75°.
        let c = within_cone(DVec3::new(1.0, 0.0, 0.1).normalize(), n, 75.0);
        assert!((c.angle_between(n).to_degrees() - 75.0).abs() < 1e-9 && c.y.abs() < 1e-12 && c.x > 0.0, "{c}");
        let d = DVec3::new(0.0, 0.5, 1.0).normalize();
        assert_eq!(within_cone(d, n, 75.0), d);
        assert!((within_cone(-n, n, 75.0).angle_between(n).to_degrees() - 75.0).abs() < 1e-9);
        // A ray through the sphere: the crossing nearer the tip as it was.
        let ray = mm_core::Ray { origin: DVec3::new(0.5, -10.0, 0.0), dir: DVec3::Y };
        let front = on_sphere(&ray, DVec3::ZERO, 1.0, -DVec3::Y).unwrap();
        assert!((front.length() - 1.0).abs() < 1e-12 && front.y < 0.0 && (front.x - 0.5).abs() < 1e-12, "{front}");
        assert!(on_sphere(&ray, DVec3::ZERO, 1.0, DVec3::Y).unwrap().y > 0.0);
        // One passing it by: the rim nearest the ray.
        let ray = mm_core::Ray { origin: DVec3::new(3.0, -10.0, 0.0), dir: DVec3::Y };
        assert!(on_sphere(&ray, DVec3::ZERO, 1.0, DVec3::Y).unwrap().distance(DVec3::X) < 1e-12);
    }

    #[test]
    fn vessel_ends_face_out_of_the_vessel_even_curled_back() {
        // A hook: along +x, round a bend and back along -x for 2 mm. Its tip faces -x, though
        // it lies on the +x side of the middle of the mesh.
        let mut path: Vec<DVec3> = (0..=20).map(|i| DVec3::new(0.5 * i as f64, 0.0, 0.0)).collect();
        path.extend((1..=16).map(|k| {
            let a = std::f64::consts::PI * k as f64 / 16.0;
            DVec3::new(10.0 + 2.0 * a.sin(), 2.0 - 2.0 * a.cos(), 0.0)
        }));
        path.extend((1..=4).map(|i| DVec3::new(10.0 - 0.5 * i as f64, 4.0, 0.0)));
        let m = primitives::tube_along(&path, &vec![0.6; path.len()], 24);
        let ends = find_vessel_ends(&m);
        assert_eq!(ends.len(), 2);
        let start = ends.iter().find(|e| e.center.x < 1.0).unwrap();
        let tip = ends.iter().find(|e| e.center.y > 3.0).unwrap();
        assert!(start.normal.dot(-DVec3::X) > 0.99, "{:?}", start.normal);
        assert!(tip.normal.dot(-DVec3::X) > 0.99, "{:?}", tip.normal);
        assert!((tip.center - m.bounds().center()).x > 1.0);
    }

    #[test]
    fn dragging_an_arrow_aims_its_end() {
        let mut scene = Scene::default();
        let path: Vec<DVec3> = (0..=40).map(|i| DVec3::new(0.0, 0.0, 0.5 * i as f64)).collect();
        scene.add("vessel", primitives::tube_along(&path, &[1.0; 41], 32));
        let (mut history, mut camera, mut status) = (crate::scene::History::default(), crate::camera::Camera::default(), String::new());
        let ctx = egui::Context::default();
        let vp = Viewport { rect: egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0)) };
        let mut cx = ToolCx { scene: &mut scene, history: &mut history, camera: &mut camera, status: &mut status, egui: &ctx, vp, brush_scroll: crate::navigation::BrushScroll::Shift };
        // From the side.
        cx.camera.look_along(DVec3::X);
        let b = cx.scene.bounds();
        cx.camera.fit(&b, vp.aspect());
        let mut tool = FlowExtensionTool::new(&mut cx);
        let top = tool.ends.iter().position(|e| e.center.z > 10.0).unwrap();
        assert!(tool.ends[top].normal.dot(DVec3::Z) > 0.999);
        tool.act("select", &json!({"ends": [top + 1]}), &mut cx).unwrap();
        let settle = |tool: &mut FlowExtensionTool, cx: &mut ToolCx| {
            let t0 = Instant::now();
            while tool.preview.as_ref().is_none_or(|(p, _)| *p != tool.params) && tool.error.is_none() && t0.elapsed() < Duration::from_secs(30) {
                std::thread::sleep(Duration::from_millis(5));
                tool.update(cx);
            }
            assert!(tool.preview.is_some(), "{:?}", tool.error);
        };
        settle(&mut tool, &mut cx);
        let at = |p: egui::Pos2| PointerInput { pos: Some(p), ..Default::default() };
        let tip = cx.project(tool.tip(top)).unwrap();
        // A click on the tip: the end neither aimed nor let go (its marker is not clicked).
        tool.pointer(&PointerInput { pressed: true, down: true, ..at(tip) }, &mut cx);
        tool.pointer(&PointerInput { released: true, clicked: true, ..at(tip) }, &mut cx);
        assert!(tool.params.aims[top].is_none() && tool.params.selected[top]);
        // Dragged across the arrow on screen: aimed that way, and not built until let go.
        let along = (tip - cx.project(tool.ends[top].center).unwrap()).normalized();
        let across = egui::vec2(-along.y, along.x);
        // (Within the sphere's outline on screen, where the tip can follow the pointer.)
        let to = tip + across * 40.0;
        tool.pointer(&PointerInput { pressed: true, down: true, ..at(tip) }, &mut cx);
        for k in 1..=8 {
            tool.pointer(&PointerInput { down: true, ..at(tip + across * (5.0 * k as f32)) }, &mut cx);
            tool.update_at(Instant::now() + SETTLE * 4, &mut cx);
            assert!(tool.job.is_none(), "frame {k}");
        }
        let aim = tool.params.aims[top].expect("aimed");
        // (Screen y runs down.)
        let side = (cx.camera.right() * across.x as f64 - cx.camera.up() * across.y as f64).normalize();
        assert!(aim.dot(side) > 0.2 && aim.dot(DVec3::Z) > 0.5, "{aim}");
        // The tip followed the pointer.
        assert!(cx.project(tool.tip(top)).unwrap().distance(to) < 1.0);
        tool.pointer(&PointerInput { released: true, ..at(to) }, &mut cx);
        settle(&mut tool, &mut cx);
        let (_, out) = tool.preview.as_ref().unwrap();
        let h = &out.heads[0];
        assert!(h.aimed && h.dir == aim && (h.off_normal - aim.angle_between(DVec3::Z).to_degrees()).abs() < 1e-6, "{h:?}");
        // The extension leans that way.
        let far: Vec<DVec3> = out.mesh.vertex_ids().map(|v| out.mesh.vertex(v)).filter(|p| p.z > 22.0).collect();
        assert!(!far.is_empty() && (far.iter().copied().sum::<DVec3>() / far.len() as f64).dot(side) > 1.0);
        // A double-click on the tip lets it go.
        let tip = cx.project(tool.tip(top)).unwrap();
        tool.pointer(&PointerInput { pressed: true, down: true, ..at(tip) }, &mut cx);
        tool.pointer(&PointerInput { released: true, clicked: true, double_clicked: true, ..at(tip) }, &mut cx);
        assert!(tool.params.aims[top].is_none() && tool.params.selected[top]);
        // Agents aim as the arrow does: no further than 75° off the cut's normal.
        let reply = tool.act("aim", &json!({"end": top + 1, "direction": [1.0, 0.0, 0.05]}), &mut cx).unwrap();
        assert!(reply.contains("75.0°") && reply.contains("as near"), "{reply}");
        assert_eq!(tool.state(cx.scene)["aims"][0]["off_cut_normal_deg"], json!(75.0));
        tool.act("aim", &json!({"end": top + 1}), &mut cx).unwrap();
        assert!(tool.params.aims[top].is_none());
        assert!(tool.act("aim", &json!({"end": 9, "direction": [0, 0, 1]}), &mut cx).is_err());
    }

    #[test]
    fn flow_extensions_sized_by_hand_fit_their_hoses() {
        // A straight vessel along z, radius 1: the top end out to a 6.35 mm hose, 15 mm long
        // with a 4 mm taper; the bottom down to a 1.5 mm one, 8 mm long.
        let mut scene = Scene::default();
        let path: Vec<DVec3> = (0..=40).map(|i| DVec3::new(0.0, 0.0, 0.5 * i as f64)).collect();
        scene.add("vessel", primitives::tube_along(&path, &[1.0; 41], 32));
        let (mut history, mut camera, mut status) = (crate::scene::History::default(), crate::camera::Camera::default(), String::new());
        let ctx = egui::Context::default();
        let vp = Viewport { rect: egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0)) };
        let mut cx = ToolCx { scene: &mut scene, history: &mut history, camera: &mut camera, status: &mut status, egui: &ctx, vp, brush_scroll: crate::navigation::BrushScroll::Shift };
        let mut tool = FlowExtensionTool::new(&mut cx);
        let top = tool.ends.iter().position(|e| e.center.z > 10.0).unwrap();
        let bottom = 1 - top;
        let settle = |tool: &mut FlowExtensionTool, cx: &mut ToolCx| {
            let t0 = Instant::now();
            while tool.preview.as_ref().is_none_or(|(p, _)| *p != tool.params) && tool.error.is_none() && t0.elapsed() < Duration::from_secs(30) {
                std::thread::sleep(Duration::from_millis(5));
                tool.update(cx);
            }
            assert!(tool.preview.is_some(), "{:?}", tool.error);
        };
        // Switched over, nothing changes until a size does: each end as wide as its cut, as
        // long as the length set.
        tool.set_settings(&json!({"length_mm": 12.0}), &mut cx).unwrap();
        tool.set_settings(&json!({"sized_by_hand": true}), &mut cx).unwrap();
        let d = tool.ends[top].diameter;
        assert_eq!(tool.params.sizes[top], EndSize { diameter: to_step(d, 0.01), length: 12.0 });
        assert!((tool.params.taper - to_step(d, 0.5)).abs() < 1e-12);
        tool.act("select", &json!({"ends": [1, 2]}), &mut cx).unwrap();
        tool.set_settings(&json!({"taper_mm": 4.0}), &mut cx).unwrap();
        let reply = tool.act("size", &json!({"ends": [top + 1], "diameter_mm": 6.35, "length_mm": 15.0}), &mut cx).unwrap();
        assert!(reply.contains("⌀ 6.350 mm, 15.000 mm long"), "{reply}");
        tool.act("size", &json!({"ends": [bottom + 1], "diameter_mm": 1.5, "length_mm": 8.0}), &mut cx).unwrap();
        assert!(tool.act("size", &json!({"ends": [1]}), &mut cx).is_err());
        assert!(tool.act("size", &json!({"ends": [1], "diameter_mm": -2.0}), &mut cx).is_err());
        settle(&mut tool, &mut cx);
        let (_, out) = tool.preview.as_ref().unwrap();
        let m = &out.mesh;
        let (z0, z1) = (tool.ends[bottom].center.z, tool.ends[top].center.z);
        // Each extension's far end where its length puts it, and from the end of the taper on
        // a cylinder of its diameter.
        let zs: Vec<f64> = m.vertex_ids().map(|v| m.vertex(v).z).collect();
        let (lo, hi) = zs.iter().fold((f64::MAX, f64::MIN), |(a, b), &z| (a.min(z), b.max(z)));
        assert!((hi - (z1 + 15.0)).abs() < 1e-9 && (lo - (z0 - 8.0)).abs() < 1e-9, "{lo} {hi}");
        let radial = |lo: f64, hi: f64| -> Vec<f64> { m.vertex_ids().map(|v| m.vertex(v)).filter(|p| p.z > lo && p.z < hi).map(|p| p.truncate().length()).collect() };
        let wide = radial(z1 + 4.0 + 1e-6, z1 + 15.0 - 1e-6);
        let narrow = radial(z0 - 8.0 + 1e-6, z0 - 4.0 - 1e-6);
        assert!(!wide.is_empty() && wide.iter().all(|r| (r - 3.175).abs() < 1e-9), "{wide:?}");
        assert!(!narrow.is_empty() && narrow.iter().all(|r| (r - 0.75).abs() < 1e-9), "{narrow:?}");
        // The taper, in between: no wider than the hose, and a valid mesh throughout.
        assert!(radial(z1, z1 + 4.0).iter().all(|&r| r < 3.175 + 1e-9));
        m.check_validity().unwrap();
        assert!(m.is_closed(), "the capped extensions close the vessel");
        // Agents read the sizes back; a short taper of a big step is pointed out.
        let state = tool.state(cx.scene);
        assert_eq!(state["sizes"].as_array().unwrap().len(), 2);
        let s = state["sizes"].as_array().unwrap().iter().find(|s| s["end"] == json!(top + 1)).unwrap();
        assert_eq!((s["diameter_mm"].as_f64(), s["length_mm"].as_f64(), s["taper_mm"].as_f64()), (Some(6.35), Some(15.0), Some(4.0)));
        assert!(state["size_notes"].as_array().unwrap().is_empty());
        tool.set_settings(&json!({"taper_mm": 1.0}), &mut cx).unwrap();
        let notes = tool.size_notes();
        assert!(notes.len() == 1 && notes[0].0 == top && notes[0].1.contains("steeply"), "{notes:?}");
        tool.set_settings(&json!({"taper_mm": 10.0}), &mut cx).unwrap();
        assert!(tool.size_notes().iter().any(|(i, n)| *i == bottom && n.contains("all its length")));
        // From the cut again: the sizes are kept for later, and not reseeded.
        tool.set_settings(&json!({"sized_by_hand": false}), &mut cx).unwrap();
        tool.set_settings(&json!({"sized_by_hand": true}), &mut cx).unwrap();
        assert_eq!(tool.params.sizes[top].diameter, 6.35);
        assert_eq!(tool.params.taper, 10.0);
        // The panel draws the sizes (a row for all, one per end) and leaves them be.
        let before = tool.params.clone();
        for _ in 0..3 {
            let raw = egui::RawInput { screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(400.0, 900.0))), ..Default::default() };
            let _ = ctx.run_ui(raw, |ui| assert!(matches!(tool.panel(ui, &mut cx), ToolOutcome::Continue)));
        }
        assert!(tool.params.by_hand && tool.params.sizes == before.sizes && tool.params.taper == before.taper);
    }

    #[test]
    fn flow_extensions_sized_by_hand_take_a_barb() {
        // A straight vessel along z, radius 1: the top end out to a 6.35 mm hose, 15 mm long
        // with a 4 mm taper, and a barb 1 mm wider starting 2 mm back from the end.
        let mut scene = Scene::default();
        let path: Vec<DVec3> = (0..=40).map(|i| DVec3::new(0.0, 0.0, 0.5 * i as f64)).collect();
        scene.add("vessel", primitives::tube_along(&path, &[1.0; 41], 32));
        let (mut history, mut camera, mut status) = (crate::scene::History::default(), crate::camera::Camera::default(), String::new());
        let ctx = egui::Context::default();
        let vp = Viewport { rect: egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0)) };
        let mut cx = ToolCx { scene: &mut scene, history: &mut history, camera: &mut camera, status: &mut status, egui: &ctx, vp, brush_scroll: crate::navigation::BrushScroll::Shift };
        let mut tool = FlowExtensionTool::new(&mut cx);
        let top = tool.ends.iter().position(|e| e.center.z > 10.0).unwrap();
        let settle = |tool: &mut FlowExtensionTool, cx: &mut ToolCx| {
            let t0 = Instant::now();
            while tool.preview.as_ref().is_none_or(|(p, _)| *p != tool.params) && tool.error.is_none() && t0.elapsed() < Duration::from_secs(30) {
                std::thread::sleep(Duration::from_millis(5));
                tool.update(cx);
            }
        };
        tool.act("select", &json!({"ends": [top + 1]}), &mut cx).unwrap();
        tool.act("size", &json!({"ends": [top + 1], "diameter_mm": 6.35, "length_mm": 15.0}), &mut cx).unwrap();
        tool.set_settings(&json!({"taper_mm": 4.0, "barb": true, "barb_width_mm": 1.0, "barb_setback_mm": 2.0}), &mut cx).unwrap();
        settle(&mut tool, &mut cx);
        let (_, out) = tool.preview.as_ref().unwrap_or_else(|| panic!("{:?}", tool.error));
        let m = &out.mesh;
        m.check_validity().unwrap();
        assert_eq!(mm_core::topology::boundary_loops(m).len(), 1, "only the bottom end, not extended, stays open");
        let z1 = tool.ends[top].center.z;
        let (ramp, end) = (barb_ramp(1.0), z1 + 15.0);
        let crest = end - 2.0 - ramp;
        // The widest of it is the barb's crest, 0.5 mm proud of the hose's 3.175 radius, a
        // ring square to the axis; straight on from where it starts to rise to the end.
        let pts: Vec<DVec3> = m.vertex_ids().map(|v| m.vertex(v)).filter(|p| p.z > z1 + 4.0 + 1e-6).collect();
        let widest = pts.iter().map(|p| p.truncate().length()).fold(0.0, f64::max);
        assert!((widest - 3.675).abs() < 1e-9, "{widest}");
        assert!(pts.iter().filter(|p| (p.truncate().length() - 3.675).abs() < 1e-9).all(|p| (p.z - crest).abs() < 1e-9));
        assert!(pts.iter().filter(|p| p.z > end - 2.0 + 1e-6 && p.z < end - 1e-6).all(|p| (p.truncate().length() - 3.175).abs() < 1e-9));
        assert!(pts.iter().filter(|p| p.z > z1 + 4.0 + 1e-6 && p.z < crest - 1e-6).all(|p| (p.truncate().length() - 3.175).abs() < 1e-9));
        // Agents read it back.
        let state = tool.state(cx.scene);
        let b = &state["sizes"][0]["barb"];
        assert_eq!(b["crest_diameter_mm"].as_f64(), Some(7.35));
        assert_eq!(b["starts_from_end_mm"].as_f64(), Some(2.0));
        assert!((b["crest_from_end_mm"].as_f64().unwrap() - (2.0 + ramp)).abs() < 1e-3);
        assert_eq!(tool.settings().unwrap()["barb_width_mm"].as_f64(), Some(1.0));
        assert!(state["size_notes"].as_array().unwrap().is_empty());
        // Set back onto the taper it is pointed out; past the opening it does not fit, and the
        // build says so.
        tool.set_settings(&json!({"barb_setback_mm": 10.0}), &mut cx).unwrap();
        assert!(tool.size_notes().iter().any(|(i, n)| *i == top && n.contains("on the taper")), "{:?}", tool.size_notes());
        tool.set_settings(&json!({"barb_setback_mm": 14.0}), &mut cx).unwrap();
        assert!(tool.size_notes().iter().any(|(i, n)| *i == top && n.contains("too short for the barb")));
        settle(&mut tool, &mut cx);
        assert!(tool.error.as_deref().is_some_and(|e| e.contains("does not fit")), "{:?}", tool.error);
        // Only sized by hand: from the cut, no barb.
        tool.set_settings(&json!({"barb_setback_mm": 2.0, "sized_by_hand": false}), &mut cx).unwrap();
        assert!(tool.barb().is_none());
        // The panel draws the barb's settings and leaves them be.
        tool.set_settings(&json!({"sized_by_hand": true}), &mut cx).unwrap();
        let before = tool.params.clone();
        for _ in 0..3 {
            let raw = egui::RawInput { screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(400.0, 900.0))), ..Default::default() };
            let _ = ctx.run_ui(raw, |ui| assert!(matches!(tool.panel(ui, &mut cx), ToolOutcome::Continue)));
        }
        assert!(tool.params.barb && tool.params.barb_width == before.barb_width && tool.params.barb_setback == before.barb_setback);
    }

    /// A real mesh (`POLYMIX_FLOW_MESH`, local only): every flat end extended along its
    /// centerline (`POLYMIX_FLOW_FIT` diameters, default 2). Prints each end's direction, its
    /// angle to the cut's normal and the build's time.
    #[test]
    #[ignore]
    fn flow_extensions_along_the_centerline_on_a_real_mesh() {
        let Some(path) = std::env::var_os("POLYMIX_FLOW_MESH") else { return };
        let fit: f64 = std::env::var("POLYMIX_FLOW_FIT").ok().and_then(|x| x.parse().ok()).unwrap_or(2.0);
        let mut scene = Scene::default();
        scene.add("vessel", mm_io::read_mesh(std::path::Path::new(&path)).unwrap().mesh);
        let (mut history, mut camera, mut status) = (crate::scene::History::default(), crate::camera::Camera::default(), String::new());
        let ctx = egui::Context::default();
        let vp = Viewport { rect: egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0)) };
        let mut cx = ToolCx { scene: &mut scene, history: &mut history, camera: &mut camera, status: &mut status, egui: &ctx, vp, brush_scroll: crate::navigation::BrushScroll::Shift };
        let mut tool = FlowExtensionTool::new(&mut cx);
        let flat: Vec<usize> = (0..tool.ends.len()).filter(|&i| tool.ends[i].planar).map(|i| i + 1).collect();
        eprintln!("{} triangles, {} ends, {} flat", cx.scene.active().unwrap().mesh.triangle_count(), tool.ends.len(), flat.len());
        tool.set_settings(&json!({"direction": "centerline", "fit_diameters": fit}), &mut cx).unwrap();
        let t0 = Instant::now();
        tool.act("select", &json!({"ends": flat}), &mut cx).unwrap();
        while tool.preview.as_ref().is_none_or(|(p, _)| *p != tool.params) && tool.error.is_none() && t0.elapsed() < Duration::from_secs(600) {
            std::thread::sleep(Duration::from_millis(5));
            tool.update(&mut cx);
        }
        eprintln!("built in {:.2?}; {:?}", t0.elapsed(), tool.error);
        for h in tool.preview.iter().flat_map(|(_, out)| &out.heads) {
            eprintln!("end {}: ⌀ {:.2}, {:.1}° off the cut normal, fitted {:?} mm, dir {:.3}; {}", h.end + 1, tool.ends[h.end].diameter, h.off_normal, h.fitted.map(|x| (x * 100.0).round() / 100.0), h.dir, h.note.as_deref().unwrap_or(""));
        }
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
