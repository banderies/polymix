//! Boolean (union / difference / intersection) of the active object with another object.
//! Meshmixer's solution modes: Fast Approximate is volumetric (robust to messy input, the
//! whole result is re-meshed); Precise is an exact mesh boolean that keeps every triangle
//! away from the intersection curve; Max Quality also remeshes a band around the curve.

use super::*;
use crate::jobs::{Job, ProgressFn};
use crate::ui_util;
use std::sync::{Arc, Mutex, OnceLock, PoisonError, Weak};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Op {
    Union,
    Difference,
    Intersection,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Solution {
    FastApproximate,
    Precise,
    MaxQuality,
}

#[derive(Clone, Copy, PartialEq, Debug)]
struct Key {
    other: ObjectId,
    op: Op,
    solution: Solution,
    accuracy: f64,
    auto_reduce: bool,
    target_edge_scale: f64,
    merge_rings: u32,
}

pub struct BooleanTool {
    object: Option<ObjectId>,
    other: Option<ObjectId>,
    op: Op,
    solution: Solution,
    accuracy: f64,
    auto_reduce: bool,
    target_edge_scale: f64,
    merge_rings: u32,
    job: Option<(Key, Job<(Mesh, String)>)>,
    result: Option<(Key, Mesh)>,
    note: String,
    error: Option<String>,
    /// When a slider last moved, while its value waits to settle before it is computed.
    changed_at: Option<Instant>,
    /// The last exact boolean, shared with the jobs (see [`RawCache`]).
    raw: RawCache,
}

/// How long slider values must stay put before they are computed (a drag would otherwise
/// start a boolean every frame).
const SETTLE: Duration = Duration::from_millis(180);

/// The exact boolean's output, before Auto-Reduce or the Max Quality remesh.
struct Raw {
    mesh: Mesh,
    seam: Vec<u32>,
    perturbed: bool,
}

/// One operand as the exact boolean saw it. Held weakly: while held, its allocation cannot
/// be reused for another mesh (an edited mesh is always a new allocation then), yet a mesh the
/// scene lets go of is still freed.
struct Operand(Weak<Mesh>, u64, u64);

impl Operand {
    fn of(m: &Arc<Mesh>) -> Self {
        Self(Arc::downgrade(m), m.shape_version(), m.group_version())
    }

    fn is(&self, m: &Arc<Mesh>) -> bool {
        std::ptr::eq(self.0.as_ptr(), Arc::as_ptr(m)) && self.1 == m.shape_version() && self.2 == m.group_version()
    }
}

/// The last exact boolean (its operation and operands, and its output once computed, or why
/// it failed), shared by the tool's jobs: a change of Auto-Reduce or of the Max Quality
/// settings reuses it rather than computing the boolean again, which takes seconds on large
/// meshes. A job that needs a boolean another job is still computing waits for it.
type RawCache = Arc<Mutex<Option<(Op, Operand, Operand, Arc<OnceLock<Result<Arc<Raw>, String>>>)>>>;

impl BooleanTool {
    pub fn new(cx: &mut ToolCx) -> Self {
        let object = cx.scene.active;
        let other = cx.scene.objects.iter().find(|o| Some(o.id) != object && o.visible).map(|o| o.id);
        Self {
            object,
            other,
            op: Op::Union,
            solution: Solution::Precise,
            accuracy: 0.5,
            auto_reduce: true,
            target_edge_scale: 1.0,
            merge_rings: 2,
            job: None,
            result: None,
            note: String::new(),
            error: None,
            changed_at: None,
            raw: RawCache::default(),
        }
    }

    /// Whether the current settings are computed or being computed.
    fn up_to_date(&self) -> bool {
        let key = self.key();
        key.is_some() && (self.job.as_ref().map(|j| j.0) == key || self.result.as_ref().map(|r| r.0) == key)
    }

    fn key(&self) -> Option<Key> {
        Some(Key {
            other: self.other?,
            op: self.op,
            solution: self.solution,
            accuracy: self.accuracy,
            auto_reduce: self.auto_reduce,
            target_edge_scale: self.target_edge_scale,
            merge_rings: self.merge_rings,
        })
    }

    /// After the panel's settings (they were `before`), at `now`: picks and toggles compute at
    /// once, slider values once they have stayed put for [`SETTLE`].
    fn settings_changed(&mut self, before: Option<Key>, now: Instant, cx: &mut ToolCx) {
        let key = self.key();
        if key != before {
            let sliders_only = matches!((before, key), (Some(b), Some(k)) if b.other == k.other && b.op == k.op && b.solution == k.solution && b.auto_reduce == k.auto_reduce);
            if sliders_only {
                self.changed_at = Some(now);
            } else {
                self.start(cx);
            }
        }
        if let Some(t) = self.changed_at {
            let waited = now.saturating_duration_since(t);
            if waited < SETTLE {
                cx.egui.request_repaint_after(SETTLE - waited);
            } else if self.up_to_date() {
                self.changed_at = None;
            } else {
                self.start(cx);
            }
        } else if self.result.is_none() && self.job.is_none() && self.error.is_none() {
            self.start(cx);
        }
    }

    fn start(&mut self, cx: &mut ToolCx) {
        self.changed_at = None;
        let (Some(a), Some(key)) = (self.object, self.key()) else { return };
        let (Some(ma), Some(mb)) = (cx.scene.get(a).map(|o| o.mesh.clone()), cx.scene.get(key.other).map(|o| o.mesh.clone())) else {
            return;
        };
        if let Some((_, j)) = &self.job {
            j.cancel();
        }
        self.error = None;
        let raw = self.raw.clone();
        let job = Job::spawn("Boolean", cx.egui.clone(), move |progress| match key.solution {
            Solution::FastApproximate => {
                let bounds = ma.bounds().union(&mb.bounds());
                let cells = (64.0 * 16f64.powf(key.accuracy)).round() as u32;
                let h = mm_solid::cell_size_for_resolution(&bounds, cells);
                let op = match key.op {
                    Op::Union => mm_solid::BoolOp::Union,
                    Op::Difference => mm_solid::BoolOp::Difference,
                    Op::Intersection => mm_solid::BoolOp::Intersection,
                };
                let m = mm_solid::sdf_boolean(&ma, &mb, op, h, Some(&|f| progress(f))).map_err(|e| anyhow::anyhow!("{e}"))?;
                Ok((m, String::new()))
            }
            Solution::Precise | Solution::MaxQuality => precise_cached(&ma, &mb, &key, progress, &raw),
        });
        self.job = Some((key, job));
    }
}

/// Triangles within `rings` rings of the seam vertices.
fn seam_band(m: &Mesh, seam: &[u32], rings: u32) -> Vec<u32> {
    let mut band = mm_core::TriSelection::new(m);
    for &v in seam {
        for t in m.vertex_triangles(v) {
            band.insert(t);
        }
    }
    for _ in 1..rings {
        band.expand(m);
    }
    band.iter().collect()
}

/// The exact boolean of `a` and `b`.
fn exact(a: &Mesh, b: &Mesh, op: Op) -> anyhow::Result<Raw> {
    let op = match op {
        Op::Union => mm_edit::BooleanOp::Union,
        Op::Difference => mm_edit::BooleanOp::Difference,
        Op::Intersection => mm_edit::BooleanOp::Intersection,
    };
    let r = mm_edit::mesh_boolean(a, b, op)?;
    Ok(Raw { mesh: r.mesh, seam: r.seam_vertices, perturbed: r.perturbed })
}

/// Exact boolean, then Auto-Reduce (collapse the slivers along the seam) or, for Max
/// Quality, a remesh of the band around the seam. The seam is a group border (the second
/// object's groups are renumbered), so both keep it in place. (The tool goes through
/// [`precise_cached`]; this is its plain form, for tests.)
#[cfg(test)]
fn precise(a: &Mesh, b: &Mesh, key: &Key, progress: ProgressFn) -> anyhow::Result<(Mesh, String)> {
    let r = exact(a, b, key.op)?;
    if !progress(0.5) {
        anyhow::bail!("cancelled");
    }
    finish(r.mesh, &r.seam, r.perturbed, a, b, key, progress)
}

/// Exact boolean, then Auto-Reduce or the Max Quality remesh (as `precise`), taking the exact
/// boolean from `cache` when it holds this one (else computing it and keeping it there). The
/// output is the same: the steps after the boolean get an identical copy of its mesh.
fn precise_cached(a: &Arc<Mesh>, b: &Arc<Mesh>, key: &Key, progress: ProgressFn, cache: &RawCache) -> anyhow::Result<(Mesh, String)> {
    let cell = {
        let mut c = cache.lock().unwrap_or_else(PoisonError::into_inner);
        match &*c {
            Some((op, ka, kb, cell)) if *op == key.op && ka.is(a) && kb.is(b) => cell.clone(),
            _ => {
                let cell: Arc<OnceLock<_>> = Arc::default();
                *c = Some((key.op, Operand::of(a), Operand::of(b), cell.clone()));
                cell
            }
        }
    };
    let r = cell.get_or_init(|| exact(a, b, key.op).map(Arc::new).map_err(|e| format!("{e:#}"))).clone().map_err(|e| anyhow::anyhow!(e))?;
    if !progress(0.5) {
        anyhow::bail!("cancelled");
    }
    finish(r.mesh.clone(), &r.seam, r.perturbed, a, b, key, progress)
}

/// Auto-Reduce or the Max Quality remesh of an exact boolean (`m`, with its seam vertices).
fn finish(mut m: Mesh, seam: &[u32], perturbed: bool, a: &Mesh, b: &Mesh, key: &Key, progress: ProgressFn) -> anyhow::Result<(Mesh, String)> {
    let note = if perturbed { "Coplanar faces were nudged apart by a tiny amount.".to_string() } else { String::new() };
    if seam.is_empty() || m.triangle_count() == 0 {
        return Ok((m, note));
    }
    let edge = 0.5 * (a.average_edge_length() + b.average_edge_length());
    let band = seam_band(&m, seam, key.merge_rings.max(1));
    if key.solution == Solution::MaxQuality {
        let params = mm_remesh::RemeshParams {
            target_edge_length: edge * key.target_edge_scale,
            iterations: 6,
            preserve_group_boundaries: true,
            region: Some(band),
            ..Default::default()
        };
        mm_remesh::remesh(&mut m, &params, Some(&|f| progress(0.5 + 0.5 * f)))?;
    } else if key.auto_reduce {
        mm_remesh::collapse_short_edges(&mut m, 0.2 * edge, Some(&band), None)?;
    }
    Ok((m, note))
}

impl Tool for BooleanTool {
    fn name(&self) -> &'static str {
        "Boolean"
    }

    fn panel(&mut self, ui: &mut egui::Ui, cx: &mut ToolCx) -> ToolOutcome {
        ui_util::section(ui, "Boolean");
        let Some(a) = self.object.filter(|id| cx.scene.get(*id).is_some()) else {
            ui.label("No active object.");
            return if ui.button("Close").clicked() { ToolOutcome::Close } else { ToolOutcome::Continue };
        };
        let names: Vec<(ObjectId, String)> = cx.scene.objects.iter().filter(|o| o.id != a).map(|o| (o.id, o.name.clone())).collect();
        if names.is_empty() {
            ui.label("Import or create a second object to combine with.");
            return if ui.button("Close").clicked() { ToolOutcome::Close } else { ToolOutcome::Continue };
        }
        let before = self.key();
        ui.label(format!("Active: {}", cx.scene.get(a).map(|o| o.name.clone()).unwrap_or_default()));
        egui::ComboBox::from_label("with")
            .selected_text(self.other.and_then(|id| names.iter().find(|n| n.0 == id)).map(|n| n.1.clone()).unwrap_or_default())
            .show_ui(ui, |ui| {
                for (id, n) in &names {
                    ui.selectable_value(&mut self.other, Some(*id), n);
                }
            });
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.op, Op::Union, "Union");
            ui.selectable_value(&mut self.op, Op::Difference, "Difference");
            ui.selectable_value(&mut self.op, Op::Intersection, "Intersection");
        });
        egui::ComboBox::from_label("Solution mode")
            .selected_text(match self.solution {
                Solution::FastApproximate => "Fast Approximate",
                Solution::Precise => "Precise",
                Solution::MaxQuality => "Max Quality",
            })
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut self.solution, Solution::FastApproximate, "Fast Approximate")
                    .on_hover_text("Volumetric: robust to messy or open input; the whole result is re-meshed");
                ui.selectable_value(&mut self.solution, Solution::Precise, "Precise")
                    .on_hover_text("Exact: keeps every triangle away from the intersection curve (needs closed meshes)");
                ui.selectable_value(&mut self.solution, Solution::MaxQuality, "Max Quality")
                    .on_hover_text("Exact, then remeshes a band around the intersection curve");
            });
        match self.solution {
            Solution::FastApproximate => {
                ui.add(egui::Slider::new(&mut self.accuracy, 0.0..=1.0).text("accuracy"));
            }
            Solution::Precise => {
                ui.checkbox(&mut self.auto_reduce, "Auto-Reduce Result").on_hover_text("Collapse the slivers the cut leaves along the seam");
            }
            Solution::MaxQuality => {
                ui.add(egui::Slider::new(&mut self.target_edge_scale, 0.25..=4.0).logarithmic(true).text("target edge scale"));
                ui.add(egui::Slider::new(&mut self.merge_rings, 1..=6).text("merge border rings"))
                    .on_hover_text("Width of the remeshed band around the seam, in triangle rings");
            }
        }
        self.settings_changed(before, Instant::now(), cx);
        if let Some((_, j)) = &self.job {
            if let Some(r) = j.poll() {
                let (k, _) = self.job.take().unwrap();
                match r {
                    Ok((m, note)) => {
                        self.result = Some((k, m));
                        self.note = note;
                    }
                    Err(e) => self.error = Some(format!("{e:#}")),
                }
            } else {
                ui.horizontal(|ui| {
                    polymix_workspace::spinner(ui);
                    ui.add(egui::ProgressBar::new(j.progress()));
                });
            }
        }
        if let Some((_, m)) = &self.result {
            ui.label(format!("Result: {} triangles", ui_util::format_count(m.triangle_count())));
            if !self.note.is_empty() {
                ui.weak(&self.note);
            }
        }
        if let Some(e) = &self.error {
            ui.colored_label(egui::Color32::from_rgb(240, 110, 90), e);
            if self.solution != Solution::FastApproximate && ui.button("Use Fast Approximate").on_hover_text("Volumetric boolean: works on open or messy meshes").clicked() {
                self.solution = Solution::FastApproximate;
            }
        }
        let ready = self.result.as_ref().map(|(k, _)| Some(*k) == self.key()).unwrap_or(false);
        let (accept, cancel) = ui_util::accept_cancel(ui, ready);
        if accept && self.accept(cx) {
            return ToolOutcome::Close;
        }
        if cancel {
            return ToolOutcome::Close;
        }
        ToolOutcome::Continue
    }

    fn preview(&self) -> Option<(ObjectId, &Mesh)> {
        let (k, m) = self.result.as_ref()?;
        (Some(*k) == self.key()).then_some((self.object?, m))
    }

    fn accept(&mut self, cx: &mut ToolCx) -> bool {
        if self.changed_at.is_some() && !self.up_to_date() {
            // A slider value still settling: compute it now, and accept once computed.
            self.start(cx);
            return false;
        }
        // Only commit a finished result for the current operation and operands.
        if self.job.is_some() || self.result.as_ref().map(|(k, _)| Some(*k) != self.key()).unwrap_or(true) {
            return false;
        }
        let Some((k, m)) = self.result.take() else { return false };
        let Some(a) = self.object else { return true };
        if cx.scene.get(a).is_none() || cx.scene.get(k.other).is_none() {
            return true;
        }
        cx.history.push("Boolean", cx.scene);
        let Some(o) = cx.scene.get_mut(a) else { return true };
        o.set_mesh(m);
        cx.scene.remove(k.other);
        cx.scene.active = Some(a);
        *cx.status = "Boolean applied".into();
        true
    }

    fn busy(&self) -> Option<(String, f32)> {
        self.job.as_ref().map(|(_, j)| ("Boolean".to_string(), j.progress()))
    }

    fn pending(&self) -> bool {
        self.changed_at.is_some() || (self.job.is_none() && self.error.is_none() && self.key().is_some() && self.result.as_ref().is_none_or(|(k, _)| Some(*k) != self.key()))
    }

    fn settings(&self) -> Option<serde_json::Value> {
        let op = match self.op {
            Op::Union => "union",
            Op::Difference => "difference",
            Op::Intersection => "intersection",
        };
        let solution = match self.solution {
            Solution::FastApproximate => "fast_approximate",
            Solution::Precise => "precise",
            Solution::MaxQuality => "max_quality",
        };
        Some(serde_json::json!({"with": self.other, "operation": op, "solution": solution, "accuracy": self.accuracy, "auto_reduce": self.auto_reduce, "target_edge_scale": self.target_edge_scale, "merge_border_rings": self.merge_rings}))
    }

    fn settings_schema(&self) -> Option<serde_json::Value> {
        Some(serde_json::json!({"type": "object", "properties": {
            "with": {"type": "integer", "description": "Object id of the other operand (it is removed when the result is accepted)"},
            "operation": {"type": "string", "enum": ["union", "difference", "intersection"], "description": "Difference removes the other object from the active one"},
            "solution": {"type": "string", "enum": ["fast_approximate", "precise", "max_quality"], "description": "fast_approximate: volumetric, robust to open or messy input, the whole result re-meshed; precise: exact, needs closed meshes; max_quality: exact, then remeshes a band around the seam"},
            "accuracy": {"type": "number", "description": "fast_approximate: 0..1 (64 to 1024 cells across)"},
            "auto_reduce": {"type": "boolean", "description": "precise: collapse the slivers the cut leaves along the seam"},
            "target_edge_scale": {"type": "number", "description": "max_quality: edge length around the seam relative to the inputs' (0.25..4)"},
            "merge_border_rings": {"type": "integer", "description": "max_quality: width of the remeshed band, in triangle rings (1..6)"},
        }}))
    }

    fn set_settings(&mut self, changes: &serde_json::Value, cx: &mut ToolCx) -> Result<(), String> {
        let mut v = self.settings().unwrap_or_default();
        merge_settings(&mut v, changes)?;
        self.other = Some(v["with"].as_u64().filter(|id| cx.scene.get(*id).is_some() && Some(*id) != self.object).ok_or("with must be the id of another object")?);
        self.op = match v["operation"].as_str() {
            Some("union") => Op::Union,
            Some("difference") => Op::Difference,
            Some("intersection") => Op::Intersection,
            _ => return Err("operation must be union, difference or intersection".into()),
        };
        self.solution = match v["solution"].as_str() {
            Some("fast_approximate") => Solution::FastApproximate,
            Some("precise") => Solution::Precise,
            Some("max_quality") => Solution::MaxQuality,
            _ => return Err("solution must be fast_approximate, precise or max_quality".into()),
        };
        self.accuracy = v["accuracy"].as_f64().ok_or("accuracy must be a number")?.clamp(0.0, 1.0);
        self.auto_reduce = v["auto_reduce"].as_bool().ok_or("auto_reduce must be true or false")?;
        self.target_edge_scale = v["target_edge_scale"].as_f64().ok_or("target_edge_scale must be a number")?.clamp(0.25, 4.0);
        self.merge_rings = (v["merge_border_rings"].as_u64().ok_or("merge_border_rings must be a whole number")? as u32).clamp(1, 6);
        self.error = None;
        self.start(cx);
        Ok(())
    }

    fn state(&self, _scene: &Scene) -> serde_json::Value {
        serde_json::json!({
            "object": self.object,
            "with": self.other,
            "computing": self.job.as_ref().map(|(_, j)| polymix_workspace::agent::round(j.progress() as f64, 2)),
            "result": self.result.as_ref().map(|(k, m)| serde_json::json!({"triangles": m.triangle_count(), "for_current_settings": Some(*k) == self.key(), "note": (!self.note.is_empty()).then(|| self.note.clone())})),
            "error": self.error,
            "next": if self.error.is_some() { "try solution fast_approximate (works on open or messy meshes)" } else { "accept (mesh_tool accept) once the result is ready" },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_core::{measure, primitives, DVec3};

    fn min_angle(m: &Mesh) -> f64 {
        m.triangle_ids()
            .map(|t| {
                let p = m.triangle(t).map(|v| m.vertex(v));
                (0..3).map(|i| (p[(i + 1) % 3] - p[i]).angle_between(p[(i + 2) % 3] - p[i])).fold(f64::INFINITY, f64::min)
            })
            .fold(f64::INFINITY, f64::min)
            .to_degrees()
    }

    #[test]
    fn precise_and_max_quality() {
        let a = primitives::icosphere(1.0, 3);
        let mut b = primitives::icosphere(0.7, 3);
        b.transform(|p| p + DVec3::new(1.1, 0.2, 0.1));
        let key = |solution, auto_reduce| Key {
            other: 0,
            op: Op::Union,
            solution,
            accuracy: 0.5,
            auto_reduce,
            target_edge_scale: 1.0,
            merge_rings: 2,
        };
        let run = |k: Key| precise(&a, &b, &k, &|_| true).unwrap().0;
        let raw = run(key(Solution::Precise, false));
        let reduced = run(key(Solution::Precise, true));
        let quality = run(key(Solution::MaxQuality, true));
        for m in [&raw, &reduced, &quality] {
            m.check_validity().unwrap();
            assert!(m.is_closed());
            assert!((measure::volume(m) - measure::volume(&raw)).abs() < 0.01 * measure::volume(&raw));
        }
        assert!(reduced.triangle_count() < raw.triangle_count(), "auto-reduce removes slivers");
        assert!(min_angle(&quality) > min_angle(&raw), "{} > {}", min_angle(&quality), min_angle(&raw));
    }

    /// Bit for bit the same mesh, ids included.
    fn assert_same(a: &Mesh, b: &Mesh, what: &str) {
        assert_eq!((a.max_vid(), a.max_tid(), a.vertex_count(), a.triangle_count(), a.edge_count()), (b.max_vid(), b.max_tid(), b.vertex_count(), b.triangle_count(), b.edge_count()), "{what}");
        assert!(a.vertex_ids().eq(b.vertex_ids()), "{what}");
        assert!(a.vertex_ids().all(|v| a.vertex(v).to_array().map(f64::to_bits) == b.vertex(v).to_array().map(f64::to_bits)), "{what}");
        assert!(a.triangle_ids().eq(b.triangle_ids()), "{what}");
        assert!(a.triangle_ids().all(|t| a.triangle(t) == b.triangle(t) && a.tri_group(t) == b.tri_group(t)), "{what}");
    }

    #[test]
    fn reusing_the_exact_boolean_changes_nothing() {
        let a = Arc::new(primitives::icosphere(1.0, 3));
        let mut b = primitives::icosphere(0.7, 3);
        b.transform(|p| p + DVec3::new(1.1, 0.2, 0.1));
        let b = Arc::new(b);
        let key = |op, solution, auto_reduce, target_edge_scale, merge_rings| Key { other: 0, op, solution, accuracy: 0.5, auto_reduce, target_edge_scale, merge_rings };
        let cache = RawCache::default();
        let mut keys = Vec::new();
        for op in [Op::Union, Op::Difference] {
            keys.extend([key(op, Solution::Precise, false, 1.0, 2), key(op, Solution::Precise, true, 1.0, 2), key(op, Solution::MaxQuality, true, 1.0, 2), key(op, Solution::MaxQuality, true, 0.6, 3)]);
        }
        for k in &keys {
            let direct = precise(&a, &b, k, &|_| true).unwrap();
            // Computed (on the first key of an operation) or taken from the cache.
            let cached = precise_cached(&a, &b, k, &|_| true, &cache).unwrap();
            assert_same(&direct.0, &cached.0, &format!("{k:?}"));
            assert_eq!(direct.1, cached.1);
            let again = precise_cached(&a, &b, k, &|_| true, &cache).unwrap();
            assert_same(&direct.0, &again.0, &format!("{k:?} again"));
        }
        // The cache holds the last operation's boolean; another operand is not mistaken for it.
        let mut c = (*b).clone();
        c.transform(|p| p + DVec3::new(0.05, 0.0, 0.0));
        let c = Arc::new(c);
        let k = key(Op::Difference, Solution::Precise, false, 1.0, 2);
        assert_same(&precise(&a, &c, &k, &|_| true).unwrap().0, &precise_cached(&a, &c, &k, &|_| true, &cache).unwrap().0, "moved operand");
        // Errors come back the same way.
        let open = Arc::new(primitives::plane_grid(1.0, 1.0, 2, 2));
        let e1 = format!("{:#}", precise(&a, &open, &k, &|_| true).unwrap_err());
        let e2 = format!("{:#}", precise_cached(&a, &open, &k, &|_| true, &cache).unwrap_err());
        assert_eq!(e1, e2);
    }

    #[test]
    fn a_slider_drag_is_computed_once_it_settles() {
        let mut scene = Scene::default();
        let a = scene.add("a", primitives::icosphere(1.0, 2));
        let mut b = primitives::icosphere(0.7, 2);
        b.transform(|p| p + DVec3::new(1.1, 0.2, 0.1));
        scene.add("b", b);
        scene.active = Some(a);
        let (mut history, mut camera, mut status) = (crate::scene::History::default(), crate::camera::Camera::default(), String::new());
        let ctx = egui::Context::default();
        let vp = Viewport { rect: egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0)) };
        let mut cx = ToolCx { scene: &mut scene, history: &mut history, camera: &mut camera, status: &mut status, egui: &ctx, vp, brush_scroll: crate::navigation::BrushScroll::Shift };
        let mut tool = BooleanTool::new(&mut cx);
        tool.solution = Solution::MaxQuality;
        // The test sets the clock, so a slow machine cannot make the drag look settled.
        let t0 = Instant::now();
        let at = |ms: u64| t0 + Duration::from_millis(ms);
        // The panel's first frame computes the settings at once.
        tool.settings_changed(tool.key(), at(0), &mut cx);
        let first = tool.job.as_ref().map(|j| j.0);
        assert!(first.is_some());
        // A drag of a slider, a frame every 10 ms: nothing new starts while it moves.
        for k in 0..20 {
            let before = tool.key();
            tool.target_edge_scale = 1.0 + 0.05 * (k + 1) as f64;
            tool.settings_changed(before, at(10 * k), &mut cx);
            assert_eq!(tool.job.as_ref().map(|j| j.0), first, "frame {k}");
            assert!(tool.pending());
        }
        // Still settling just before SETTLE has passed since the last change.
        let last = 10 * 19;
        tool.settings_changed(tool.key(), at(last) + SETTLE - Duration::from_millis(1), &mut cx);
        assert_eq!(tool.job.as_ref().map(|j| j.0), first);
        // Once it has settled, its last value.
        tool.settings_changed(tool.key(), at(last) + SETTLE, &mut cx);
        assert_eq!(tool.job.as_ref().map(|j| j.0), tool.key());
        assert!(tool.changed_at.is_none());
        // Picks compute at once.
        let before = tool.key();
        tool.op = Op::Difference;
        tool.settings_changed(before, at(last) + SETTLE, &mut cx);
        assert_eq!(tool.job.as_ref().map(|j| j.0), tool.key());
    }
}
