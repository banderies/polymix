//! Parameterised mesh operations with live background preview (Remesh, Reduce, Smooth,
//! Make Solid, …). Each operation is an [`OpSpec`]; [`OpTool`] supplies the shared
//! preview / accept / cancel behaviour.

use super::*;
use crate::jobs::{Job, ProgressFn};
use crate::ui_util;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Result of running an operation on a copy of the mesh.
#[derive(Default)]
pub struct OpOutput {
    pub mesh: Mesh,
    /// Triangles to select afterwards (e.g. the remeshed region).
    pub selection: Option<Vec<u32>>,
    pub message: String,
    /// Objects to add to the scene on accept (e.g. Extract), as (name suffix, mesh).
    pub new_objects: Vec<(String, Mesh)>,
    /// What to show while previewing, when it differs from `mesh` (e.g. the object plus the
    /// objects it will add).
    pub preview: Option<Mesh>,
    /// Markers drawn in the view while previewing (e.g. a loft's End 1 / End 2).
    pub labels: Vec<(DVec3, String)>,
}

/// Context shown to operation panels.
pub struct OpInfo<'a> {
    pub mesh: &'a Mesh,
    pub selection: Option<&'a [u32]>,
    pub avg_edge: f64,
    pub bbox_diag: f64,
    /// The scene's target surface, if one is set (and is not this object).
    pub target: Option<&'a TargetSurface>,
}

pub trait OpSpec: Send + 'static {
    /// Also what agents read and set (serde), described by its JSON Schema (doc comments
    /// included).
    type Params: Clone + PartialEq + Send + serde::Serialize + serde::de::DeserializeOwned + schemars::JsonSchema + 'static;
    const NAME: &'static str;
    /// Whether the op restricts itself to the selection when one exists.
    const USES_SELECTION: bool = true;
    /// Recompute automatically while parameters change.
    const LIVE: bool = true;
    /// Show face-group colours while the tool is open.
    const SHOWS_GROUPS: bool = false;
    fn defaults(info: &OpInfo) -> Self::Params;
    fn ui(ui: &mut egui::Ui, p: &mut Self::Params, info: &OpInfo);
    fn run(mesh: &Mesh, selection: Option<&[u32]>, p: &Self::Params, progress: ProgressFn) -> anyhow::Result<OpOutput>;
    /// [`OpSpec::run`] with the scene's target surface, for ops that use one.
    fn run_with(mesh: &Mesh, selection: Option<&[u32]>, p: &Self::Params, _target: Option<&TargetSurface>, progress: ProgressFn) -> anyhow::Result<OpOutput> {
        Self::run(mesh, selection, p, progress)
    }
    /// In-tool hotkeys (Meshmixer: D/F density ±10, T adaptive, G groups). Return true if
    /// the parameters changed.
    fn keys(_p: &mut Self::Params, _i: &egui::InputState) -> bool {
        false
    }
    /// Test-script settings (`toolcmd:<cmd>`).
    fn script(_p: &mut Self::Params, _cmd: &str) {}
}

/// Plain (unmodified) key press.
fn plain(i: &egui::InputState, k: egui::Key) -> bool {
    i.key_pressed(k) && !i.modifiers.command && !i.modifiers.alt && !i.modifiers.shift
}

pub struct OpTool<S: OpSpec> {
    object: ObjectId,
    source: Arc<Mesh>,
    selection: Option<Arc<Vec<u32>>>,
    target: Option<TargetSurface>,
    pub params: S::Params,
    avg_edge: f64,
    bbox_diag: f64,
    /// Params of the running job / of the current result.
    job: Option<(S::Params, Job<OpOutput>)>,
    result: Option<(S::Params, OpOutput)>,
    error: Option<String>,
    /// Params whose run failed (with `error`): running them again would fail again.
    failed: Option<S::Params>,
    changed_at: Option<Instant>,
    pending_accept: bool,
    invalid: bool,
}

impl<S: OpSpec> OpTool<S> {
    pub fn new(cx: &mut ToolCx) -> Self {
        let target = cx.scene.target_surface(cx.scene.active);
        let Some(o) = cx.scene.active() else {
            return Self::invalid();
        };
        let selection = if S::USES_SELECTION && !o.selection.is_empty() { Some(Arc::new(o.selection.to_vec())) } else { None };
        let source = o.mesh.clone();
        let avg_edge = source.average_edge_length();
        let bbox_diag = source.bounds().diagonal();
        let info = OpInfo { mesh: &source, selection: selection.as_deref().map(|v| v.as_slice()), avg_edge, bbox_diag, target: target.as_ref() };
        let params = S::defaults(&info);
        Self {
            object: o.id,
            source,
            selection,
            target,
            params,
            avg_edge,
            bbox_diag,
            job: None,
            result: None,
            error: None,
            failed: None,
            changed_at: if S::LIVE { Some(Instant::now() - Duration::from_secs(1)) } else { None },
            pending_accept: false,
            invalid: false,
        }
    }

    fn invalid() -> Self {
        let info = OpInfo { mesh: &Mesh::new(), selection: None, avg_edge: 1.0, bbox_diag: 1.0, target: None };
        let params = S::defaults(&info);
        Self {
            object: 0,
            source: Arc::new(Mesh::new()),
            selection: None,
            target: None,
            params,
            avg_edge: 1.0,
            bbox_diag: 1.0,
            job: None,
            result: None,
            error: Some("No active object.".into()),
            failed: None,
            changed_at: None,
            pending_accept: false,
            invalid: true,
        }
    }

    fn start(&mut self, ctx: &egui::Context) {
        if let Some((_, j)) = &self.job {
            j.cancel();
        }
        let p = self.params.clone();
        let src = self.source.clone();
        let sel = self.selection.clone();
        let target = self.target.clone();
        let p2 = p.clone();
        let job = Job::spawn(S::NAME, ctx.clone(), move |progress| S::run_with(&src, sel.as_deref().map(|v| v.as_slice()), &p2, target.as_ref(), progress));
        self.job = Some((p, job));
        self.error = None;
    }

    fn poll(&mut self) {
        let Some((_, job)) = &self.job else { return };
        if let Some(r) = job.poll() {
            let (p, _) = self.job.take().unwrap();
            match r {
                Ok(out) => {
                    self.result = Some((p, out));
                }
                Err(e) => {
                    if e.to_string() != "cancelled" {
                        self.error = Some(format!("{e:#}"));
                        self.failed = Some(p);
                    }
                }
            }
        }
    }

    fn up_to_date(&self) -> bool {
        self.result.as_ref().map(|(p, _)| *p == self.params).unwrap_or(false)
    }

    fn commit(&mut self, cx: &mut ToolCx) -> bool {
        // Never commit a result computed from a mesh the object no longer has.
        match cx.scene.get(self.object) {
            None => return true,
            Some(o) if !Arc::ptr_eq(&o.mesh, &self.source) => {
                self.source = o.mesh.clone();
                self.result = None;
                self.error = Some("The object changed; recomputing the preview.".into());
                self.start(cx.egui);
                return false;
            }
            _ => {}
        }
        let Some((_, out)) = self.result.take() else { return false };
        cx.history.push(S::NAME, cx.scene);
        let Some(o) = cx.scene.get_mut(self.object) else { return true };
        let base_name = o.name.clone();
        o.set_mesh(out.mesh);
        if let Some(sel) = out.selection {
            let m = o.mesh.clone();
            for t in sel {
                if m.is_triangle(t) {
                    o.selection.insert(t);
                }
            }
            o.touch_selection();
        }
        for (suffix, mesh) in out.new_objects {
            let id = cx.scene.add(format!("{base_name} {suffix}"), mesh);
            cx.scene.active = Some(id);
        }
        *cx.status = format!("{}: {}", S::NAME, out.message);
        true
    }
}

impl<S: OpSpec> Tool for OpTool<S> {
    fn name(&self) -> &'static str {
        S::NAME
    }

    fn shows_groups(&self) -> bool {
        S::SHOWS_GROUPS
    }

    fn script(&mut self, cmd: &str, _cx: &mut ToolCx) {
        S::script(&mut self.params, cmd);
        self.changed_at = Some(Instant::now() - Duration::from_secs(1));
    }

    fn keys(&mut self, i: &egui::InputState, _cx: &mut ToolCx) -> bool {
        if S::keys(&mut self.params, i) {
            self.changed_at = Some(Instant::now());
            return true;
        }
        false
    }

    fn panel(&mut self, ui: &mut egui::Ui, cx: &mut ToolCx) -> ToolOutcome {
        self.poll();
        ui_util::section(ui, S::NAME);
        if self.invalid {
            ui.label("Select an object first.");
            return if ui.button("Close").clicked() { ToolOutcome::Close } else { ToolOutcome::Continue };
        }
        if let Some(sel) = &self.selection {
            ui.label(RichTextExt::info(format!("Acting on {} selected triangles", ui_util::format_count(sel.len()))));
        }
        let before = self.params.clone();
        {
            let info = OpInfo {
                mesh: &self.source,
                selection: self.selection.as_deref().map(|v| v.as_slice()),
                avg_edge: self.avg_edge,
                bbox_diag: self.bbox_diag,
                target: self.target.as_ref(),
            };
            S::ui(ui, &mut self.params, &info);
        }
        if self.params != before {
            self.changed_at = Some(Instant::now());
        }
        // Debounced live update.
        if S::LIVE {
            if let Some(t) = self.changed_at {
                if t.elapsed() > Duration::from_millis(180) && !self.up_to_date() {
                    let running_same = self.job.as_ref().map(|(p, _)| *p == self.params).unwrap_or(false);
                    if !running_same {
                        self.start(cx.egui);
                    }
                    self.changed_at = None;
                } else if !self.up_to_date() {
                    cx.egui.request_repaint_after(Duration::from_millis(60));
                }
            }
        } else if ui.button("Update preview").clicked() {
            self.start(cx.egui);
        }

        ui.add_space(6.0);
        if let Some((p, j)) = &self.job {
            let _ = p;
            ui.horizontal(|ui| {
                ui.spinner();
                ui.add(egui::ProgressBar::new(j.progress()).show_percentage());
            });
            cx.egui.request_repaint_after(Duration::from_millis(50));
        } else if let Some((_, out)) = &self.result {
            ui.label(&out.message);
            ui.label(format!("Result: {} triangles", ui_util::format_count(out.mesh.triangle_count())));
        }
        if let Some(e) = &self.error {
            ui.colored_label(egui::Color32::from_rgb(240, 110, 90), e);
        }

        let can_accept = self.result.is_some() || self.job.is_some() || !S::LIVE;
        let (accept, cancel) = ui_util::accept_cancel(ui, can_accept);
        if accept {
            self.pending_accept = true;
            if self.result.is_none() && self.job.is_none() {
                self.start(cx.egui);
            }
        }
        if self.pending_accept && self.job.is_none() {
            if !self.up_to_date() && self.error.is_none() {
                self.start(cx.egui);
            } else if self.commit(cx) {
                return ToolOutcome::Close;
            } else {
                self.pending_accept = false;
            }
        }
        if cancel {
            if let Some((_, j)) = &self.job {
                j.cancel();
            }
            return ToolOutcome::Close;
        }
        ToolOutcome::Continue
    }

    fn paint(&mut self, painter: &egui::Painter, cx: &mut ToolCx) {
        let Some((_, out)) = &self.result else { return };
        for (p, text) in &out.labels {
            if let Some(c) = cx.project(*p) {
                painter.circle(c, 9.0, ui_util::ACCENT, egui::Stroke::new(1.5, egui::Color32::from_gray(20)));
                painter.text(c, egui::Align2::CENTER_CENTER, text, egui::FontId::proportional(12.0), egui::Color32::WHITE);
            }
        }
    }

    fn preview(&self) -> Option<(ObjectId, &Mesh)> {
        self.result.as_ref().map(|(_, out)| (self.object, out.preview.as_ref().unwrap_or(&out.mesh)))
    }

    fn accept(&mut self, cx: &mut ToolCx) -> bool {
        self.poll();
        // The current settings already failed: there is nothing to apply, and running them
        // again would only fail again (switching tools retries an accept until it succeeds).
        if self.job.is_none() && self.failed.as_ref() == Some(&self.params) {
            *cx.status = format!("{}: nothing applied ({})", S::NAME, self.error.as_deref().unwrap_or("failed"));
            return true;
        }
        if self.job.is_some() || !self.up_to_date() {
            self.pending_accept = true;
            if self.job.is_none() {
                self.start(cx.egui);
            }
            return false;
        }
        self.commit(cx)
    }

    fn cancel(&mut self, _cx: &mut ToolCx) -> bool {
        if let Some((_, j)) = &self.job {
            j.cancel();
        }
        true
    }

    fn busy(&self) -> Option<(String, f32)> {
        self.job.as_ref().map(|(_, j)| (S::NAME.to_string(), j.progress()))
    }

    fn pending(&self) -> bool {
        !self.invalid && self.job.is_none() && (self.pending_accept || (S::LIVE && self.changed_at.is_some() && !self.up_to_date()))
    }

    fn settings(&self) -> Option<serde_json::Value> {
        serde_json::to_value(&self.params).ok().filter(|v| v.is_object())
    }

    fn settings_schema(&self) -> Option<serde_json::Value> {
        self.settings().map(|_| schema_of::<S::Params>())
    }

    fn set_settings(&mut self, changes: &serde_json::Value, cx: &mut ToolCx) -> Result<(), String> {
        if self.invalid {
            return Err(format!("{}: no active object", S::NAME));
        }
        let mut v = self.settings().ok_or_else(|| format!("{} has no settings", S::NAME))?;
        merge_settings(&mut v, changes)?;
        self.params = serde_json::from_value(v).map_err(|e| format!("{}: {e}", S::NAME))?;
        // Previews start now, not after the usual typing pause (nor only once the panel is
        // drawn); accept collects the result.
        if S::LIVE && !self.up_to_date() {
            self.start(cx.egui);
            self.changed_at = None;
        }
        Ok(())
    }

    fn actions(&self) -> Vec<Action> {
        if S::LIVE {
            return Vec::new();
        }
        vec![Action::plain("update", "Compute the preview for the current settings (this operation does not preview by itself; accept computes it too)")]
    }

    fn act(&mut self, action: &str, _args: &serde_json::Value, cx: &mut ToolCx) -> Result<String, String> {
        match action {
            "update" if !S::LIVE => {
                if self.invalid {
                    return Err(format!("{}: no active object", S::NAME));
                }
                self.start(cx.egui);
                Ok(format!("{}: computing the preview; app_wait, then mesh_tool_state", S::NAME))
            }
            _ => Err(format!("{} has no action {action}", S::NAME)),
        }
    }

    fn state(&self, _scene: &Scene) -> serde_json::Value {
        serde_json::json!({
            "computing": self.job.as_ref().map(|(_, j)| j.progress()),
            "acting_on_selected_triangles": self.selection.as_ref().map(|s| s.len()),
            "result": self.result.as_ref().map(|(p, out)| serde_json::json!({"message": out.message, "triangles": out.mesh.triangle_count(), "for_current_settings": *p == self.params})),
            "error": self.error,
            "updates": if S::LIVE { "live (recomputed after every change)" } else { "on accept" },
        })
    }
}

/// Helper for informational text styling.
struct RichTextExt;
impl RichTextExt {
    fn info(s: String) -> egui::RichText {
        egui::RichText::new(s).color(egui::Color32::from_rgb(255, 170, 90))
    }
}

// ----------------------------------------------------------------------------
// Operation specs. Those whose algorithms live in sibling crates are filled in as the
// crates land; until then they report that they are unavailable.
// ----------------------------------------------------------------------------

pub use crate::tools::boolean::BooleanTool;
pub use crate::tools::vessel::BridgeTool;

// ---------------------------------------------------------------------------
// Extrude / Offset (selection)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub enum ExtrudeMode {
    /// Move the region and stitch side walls (Meshmixer Extrude, D).
    Extrude,
    /// Move the region with a soft transition, no new walls (Meshmixer Offset, Ctrl+D).
    Offset,
    /// Copy the region into a new, thickened solid object (Meshmixer Extract, Shift+D).
    Extract,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub enum ExtrudeDir {
    Normal,
    /// The region's average normal.
    Constant,
    Axis(usize),
}

#[derive(Clone, Copy, PartialEq, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct ExtrudeParamsUi {
    pub mode: ExtrudeMode,
    pub distance: f64,
    pub direction: ExtrudeDir,
    /// End type: flatten the moved region (Flat) or keep its shape (Offset).
    pub flat_end: bool,
    /// 0..100: rows of side-wall triangles.
    pub density: f64,
    /// 0..100: > 0 gives the side walls their own group (a crease).
    pub harden: f64,
    /// Keep the region's face groups (otherwise it becomes a new group).
    pub preserve_groups: bool,
    /// Offset: width of the soft transition.
    pub falloff: f64,
    /// Offset: open-boundary vertices stay put.
    pub preserve_boundary: bool,
    /// Offset: 0..100, relax the moved triangles afterwards.
    pub regularity: f64,
}

pub struct ExtrudeSpec;
impl OpSpec for ExtrudeSpec {
    type Params = ExtrudeParamsUi;
    const NAME: &'static str = "Extrude / Offset";
    fn defaults(info: &OpInfo) -> Self::Params {
        ExtrudeParamsUi {
            mode: ExtrudeMode::Extrude,
            distance: (info.avg_edge * 3.0).max(1e-4),
            direction: ExtrudeDir::Normal,
            flat_end: false,
            density: 20.0,
            harden: 0.0,
            preserve_groups: false,
            falloff: 0.0,
            preserve_boundary: false,
            regularity: 25.0,
        }
    }
    fn ui(ui: &mut egui::Ui, p: &mut Self::Params, info: &OpInfo) {
        ui.horizontal(|ui| {
            ui.selectable_value(&mut p.mode, ExtrudeMode::Extrude, "Extrude").on_hover_text("D");
            ui.selectable_value(&mut p.mode, ExtrudeMode::Offset, "Offset").on_hover_text("Cmd+D");
            ui.selectable_value(&mut p.mode, ExtrudeMode::Extract, "Extract").on_hover_text("Shift+D: a new solid object from the selection");
        });
        let span = (info.bbox_diag * 0.25).max(info.avg_edge * 10.0);
        ui.add(egui::Slider::new(&mut p.distance, -span..=span).text(if p.mode == ExtrudeMode::Offset { "Distance" } else { "Offset" }));
        if p.mode != ExtrudeMode::Offset {
            egui::ComboBox::from_label("Direction")
                .selected_text(match p.direction {
                    ExtrudeDir::Normal => "Normal",
                    ExtrudeDir::Constant => "Constant",
                    ExtrudeDir::Axis(0) => "X Axis",
                    ExtrudeDir::Axis(1) => "Y Axis",
                    ExtrudeDir::Axis(_) => "Z Axis",
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut p.direction, ExtrudeDir::Normal, "Normal").on_hover_text("Each vertex along its normal");
                    ui.selectable_value(&mut p.direction, ExtrudeDir::Constant, "Constant").on_hover_text("The region's average normal");
                    for (i, n) in ["X Axis", "Y Axis", "Z Axis"].iter().enumerate() {
                        ui.selectable_value(&mut p.direction, ExtrudeDir::Axis(i), *n);
                    }
                });
            ui.horizontal(|ui| {
                ui.label("End type");
                ui.selectable_value(&mut p.flat_end, false, "Offset");
                ui.selectable_value(&mut p.flat_end, true, "Flat");
            });
        }
        match p.mode {
            ExtrudeMode::Extrude => {
                ui.add(egui::Slider::new(&mut p.harden, 0.0..=100.0).text("Harden")).on_hover_text("> 0: side walls get their own face group (a crease)");
                ui.add(egui::Slider::new(&mut p.density, 0.0..=100.0).text("Density")).on_hover_text("Rows of side-wall triangles");
                ui.checkbox(&mut p.preserve_groups, "Preserve groups");
            }
            ExtrudeMode::Offset => {
                ui.add(egui::Slider::new(&mut p.falloff, 0.0..=span).text("Soft transition"));
                ui.add(egui::Slider::new(&mut p.regularity, 0.0..=100.0).text("Regularity")).on_hover_text("Relax the moved triangles");
                ui.checkbox(&mut p.preserve_boundary, "Preserve boundary");
            }
            ExtrudeMode::Extract => {
                ui.small("Creates a new object: the selection thickened into a closed solid. The original is unchanged.");
            }
        }
        if info.selection.is_none() {
            ui.colored_label(egui::Color32::from_rgb(240, 170, 90), "Select a region first.");
        }
    }
    fn run(mesh: &Mesh, sel: Option<&[u32]>, p: &Self::Params, _progress: ProgressFn) -> anyhow::Result<OpOutput> {
        let tris = sel.ok_or_else(|| anyhow::anyhow!("select a region first"))?;
        let direction = match p.direction {
            ExtrudeDir::Normal => mm_edit::ExtrudeDirection::Normal,
            ExtrudeDir::Constant => mm_edit::ExtrudeDirection::Average,
            ExtrudeDir::Axis(i) => {
                let mut d = DVec3::ZERO;
                d[i.min(2)] = 1.0;
                mm_edit::ExtrudeDirection::Fixed(d)
            }
        };
        match p.mode {
            ExtrudeMode::Extrude => {
                let mut m = mesh.clone();
                let params = mm_edit::ExtrudeParams {
                    distance: p.distance,
                    direction,
                    side_segments: (1.0 + p.density / 20.0).round().max(1.0) as usize,
                    harden_sides: p.harden > 0.0,
                    flat_end: p.flat_end,
                    new_region_group: !p.preserve_groups,
                };
                let r = mm_edit::extrude_selection(&mut m, tris, &params)?;
                let n = r.side_tris.len();
                Ok(OpOutput { mesh: m, selection: Some(r.new_region_tris), message: format!("extruded with {n} side triangles"), ..Default::default() })
            }
            ExtrudeMode::Offset => {
                let mut m = mesh.clone();
                let n = mm_edit::offset_selection_with(&mut m, tris, p.distance, p.falloff, p.preserve_boundary)?;
                let mut selection = tris.to_vec();
                let iterations = (p.regularity / 10.0).round() as usize;
                if iterations > 0 {
                    let params = mm_remesh::RemeshParams {
                        iterations,
                        preserve_group_boundaries: true,
                        region: Some(tris.to_vec()),
                        ..Default::default()
                    };
                    let r = mm_remesh::remesh(&mut m, &params, None)?;
                    selection = r.region_triangles.unwrap_or(selection);
                }
                Ok(OpOutput { mesh: m, selection: Some(selection), message: format!("moved {n} vertices"), ..Default::default() })
            }
            ExtrudeMode::Extract => {
                let solid = mm_edit::extract_solid(mesh, tris, p.distance, direction, p.flat_end)?;
                let mut preview = mesh.clone();
                preview.append_mesh(&solid, preview.max_group_id() + 1);
                let n = solid.triangle_count();
                Ok(OpOutput {
                    mesh: mesh.clone(),
                    selection: Some(tris.to_vec()),
                    message: format!("extracted a solid of {n} triangles"),
                    new_objects: vec![("extract".into(), solid)],
                    preview: Some(preview),
                    ..Default::default()
                })
            }
        }
    }
}

pub struct ExtrudeTool;
impl ExtrudeTool {
    #[allow(clippy::new_ret_no_self)]
    pub fn new(cx: &mut ToolCx, mode: ExtrudeMode) -> OpTool<ExtrudeSpec> {
        let mut t: OpTool<ExtrudeSpec> = OpTool::new(cx);
        t.params.mode = mode;
        t
    }
}

// ---------------------------------------------------------------------------
// Mirror
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct MirrorParams {
    pub axis: usize,
    /// Plane position as a fraction of the bounding box along the axis (0 = min, 1 = max).
    pub position: f64,
    pub weld: bool,
}

pub struct MirrorSpec;
impl OpSpec for MirrorSpec {
    type Params = MirrorParams;
    const NAME: &'static str = "Mirror";
    const USES_SELECTION: bool = false;
    fn defaults(_: &OpInfo) -> Self::Params {
        MirrorParams { axis: 0, position: 0.0, weld: true }
    }
    fn ui(ui: &mut egui::Ui, p: &mut Self::Params, _: &OpInfo) {
        ui.horizontal(|ui| {
            ui.label("Mirror across");
            for (i, n) in ["X", "Y", "Z"].iter().enumerate() {
                ui.selectable_value(&mut p.axis, i, *n);
            }
        });
        ui.add(egui::Slider::new(&mut p.position, 0.0..=1.0).text("plane position (bbox)"));
        ui.checkbox(&mut p.weld, "Weld at the plane");
    }
    fn run(mesh: &Mesh, _: Option<&[u32]>, p: &Self::Params, _: ProgressFn) -> anyhow::Result<OpOutput> {
        let b = mesh.bounds();
        let mut n = DVec3::ZERO;
        n[p.axis] = 1.0;
        let at = b.min[p.axis] + (b.max[p.axis] - b.min[p.axis]) * p.position;
        let mut m = mesh.clone();
        let r = mm_edit::mirror(&mut m, mm_core::Plane { normal: n, d: at }, p.weld)?;
        Ok(OpOutput { mesh: m, selection: None, message: format!("{} triangles added, {} welded", r.new_triangles.len(), r.welded_vertices), ..Default::default() })
    }
}
pub type MirrorTool = OpTool<MirrorSpec>;

// ---------------------------------------------------------------------------
// Remesh (Meshmixer Edit > Remesh, hotkey R)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub enum RemeshMode {
    RelativeDensity,
    AdaptiveDensity,
    TargetEdgeLength,
    /// Split edges to the target length without moving any vertex.
    LinearSubdivision,
}

impl RemeshMode {
    fn label(self) -> &'static str {
        match self {
            RemeshMode::RelativeDensity => "Relative Density",
            RemeshMode::AdaptiveDensity => "Adaptive Density",
            RemeshMode::TargetEdgeLength => "Target Edge Length",
            RemeshMode::LinearSubdivision => "Linear Subdivision",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub enum RemeshBoundary {
    Fixed,
    Refined,
    Free,
}

#[derive(Clone, Copy, PartialEq, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct RemeshUi {
    pub mode: RemeshMode,
    /// -100..100 %: 0 keeps the current mean edge length, +50 halves it, −50 doubles it.
    pub density: f64,
    pub edge_length: f64,
    /// 0..100: tangential relaxation (triangle evenness).
    pub regularity: f64,
    /// 0..100: how strongly Adaptive Density follows curvature.
    pub threshold: f64,
    pub iterations: usize,
    /// Width of the band around a selection where the edge length blends back to the
    /// surroundings (world units; 0 = none).
    pub transition: f64,
    /// Keep face-group borders exactly where they are.
    pub preserve_group_borders: bool,
    /// Keep face-group borders, but smooth them.
    pub smooth_group_boundaries: bool,
    pub preserve_sharp: bool,
    pub sharp_threshold: f64,
    pub boundary: RemeshBoundary,
    pub base_edge: f64,
}

impl RemeshUi {
    fn target(&self) -> f64 {
        match self.mode {
            RemeshMode::TargetEdgeLength => self.edge_length,
            _ => self.base_edge * 2f64.powf(-self.density / 50.0),
        }
    }
}

/// The selection plus a band of `width` around it, and per-vertex target sizes that blend
/// from `target` inside the selection to the local edge length at the band's outer edge.
fn transition_band(m: &Mesh, sel: &[u32], width: f64, target: f64) -> (Vec<u32>, Vec<f64>) {
    use std::collections::BinaryHeap;
    #[derive(PartialEq)]
    struct Item(f64, u32);
    impl Eq for Item {}
    impl PartialOrd for Item {
        fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(o))
        }
    }
    impl Ord for Item {
        fn cmp(&self, o: &Self) -> std::cmp::Ordering {
            o.0.total_cmp(&self.0)
        }
    }
    let s = mm_core::TriSelection::from_tris(m, sel.iter().copied());
    let mut inside = vec![false; m.max_vid()];
    for &t in sel {
        for v in m.triangle(t) {
            inside[v as usize] = true;
        }
    }
    let mut dist = vec![f64::INFINITY; m.max_vid()];
    let mut heap = BinaryHeap::new();
    for v in s.boundary_vertices(m) {
        dist[v as usize] = 0.0;
        heap.push(Item(0.0, v));
    }
    while let Some(Item(d, v)) = heap.pop() {
        if d > dist[v as usize] || d > width {
            continue;
        }
        for u in m.vertex_neighbors(v) {
            if inside[u as usize] {
                continue;
            }
            let nd = d + m.vertex(v).distance(m.vertex(u));
            if nd < dist[u as usize] {
                dist[u as usize] = nd;
                heap.push(Item(nd, u));
            }
        }
    }
    let mut region = sel.to_vec();
    region.extend(m.triangle_ids().filter(|&t| !s.contains(t) && m.triangle(t).iter().all(|&v| inside[v as usize] || dist[v as usize] <= width)));
    let mut sizes = vec![target; m.max_vid()];
    for v in m.vertex_ids() {
        if inside[v as usize] || !dist[v as usize].is_finite() {
            continue;
        }
        let local = m.vertex_edges(v).iter().map(|&e| m.edge_length(e)).sum::<f64>() / m.vertex_valence(v).max(1) as f64;
        let x = (dist[v as usize] / width).clamp(0.0, 1.0);
        let w = x * x * (3.0 - 2.0 * x);
        sizes[v as usize] = target + (local - target) * w;
    }
    (region, sizes)
}

pub struct RemeshSpec;
impl OpSpec for RemeshSpec {
    type Params = RemeshUi;
    const NAME: &'static str = "Remesh";
    fn keys(p: &mut Self::Params, i: &egui::InputState) -> bool {
        let before = *p;
        if plain(i, egui::Key::D) {
            p.density = (p.density + 10.0).min(100.0);
        }
        if plain(i, egui::Key::F) {
            p.density = (p.density - 10.0).max(-100.0);
        }
        if (plain(i, egui::Key::D) || plain(i, egui::Key::F)) && p.mode == RemeshMode::TargetEdgeLength {
            p.mode = RemeshMode::RelativeDensity;
        }
        if plain(i, egui::Key::T) {
            p.mode = if p.mode == RemeshMode::AdaptiveDensity { RemeshMode::RelativeDensity } else { RemeshMode::AdaptiveDensity };
        }
        if plain(i, egui::Key::G) {
            p.preserve_group_borders = !p.preserve_group_borders;
        }
        *p != before
    }
    fn defaults(info: &OpInfo) -> Self::Params {
        let base = match info.selection {
            Some(sel) => mm_remesh::edge_length_stats_in(info.mesh, sel).mean,
            None => info.avg_edge,
        }
        .max(1e-9);
        // Meshmixer's defaults.
        RemeshUi {
            mode: RemeshMode::RelativeDensity,
            density: 0.0,
            edge_length: base,
            regularity: 25.0,
            threshold: 50.0,
            iterations: 10,
            transition: 0.0,
            preserve_group_borders: false,
            smooth_group_boundaries: true,
            preserve_sharp: false,
            sharp_threshold: 45.0,
            boundary: RemeshBoundary::Fixed,
            base_edge: base,
        }
    }
    fn ui(ui: &mut egui::Ui, p: &mut Self::Params, info: &OpInfo) {
        egui::ComboBox::from_label("Remesh mode").selected_text(p.mode.label()).show_ui(ui, |ui| {
            for m in [RemeshMode::RelativeDensity, RemeshMode::AdaptiveDensity, RemeshMode::TargetEdgeLength, RemeshMode::LinearSubdivision] {
                ui.selectable_value(&mut p.mode, m, m.label());
            }
        });
        if p.mode == RemeshMode::TargetEdgeLength {
            ui.add(egui::Slider::new(&mut p.edge_length, (p.base_edge * 0.1)..=(p.base_edge * 10.0)).logarithmic(true).text("Edge length"));
        } else {
            ui.add(egui::Slider::new(&mut p.density, -100.0..=100.0).suffix(" %").text("Density")).on_hover_text("D / F: ±10");
            ui.weak(format!("edge length ≈ {:.4}", p.target()));
        }
        let smoothing = p.mode != RemeshMode::LinearSubdivision;
        ui.add_enabled(smoothing, egui::Slider::new(&mut p.regularity, 0.0..=100.0).text("Regularity"));
        ui.add_enabled(p.mode == RemeshMode::AdaptiveDensity, egui::Slider::new(&mut p.threshold, 0.0..=100.0).text("Threshold"))
            .on_hover_text("How strongly Adaptive Density refines curved areas");
        ui.add_enabled(smoothing, egui::Slider::new(&mut p.iterations, 1..=40).text("Iterations"));
        if info.selection.is_some() {
            let span = (p.base_edge * 20.0).max(1e-6);
            ui.add_enabled(smoothing, egui::Slider::new(&mut p.transition, 0.0..=span).text("Transition"))
                .on_hover_text("Blend the edge length back to the surroundings over this distance outside the selection");
        }
        ui.checkbox(&mut p.preserve_group_borders, "Preserve group borders").on_hover_text("G");
        ui.add_enabled(!p.preserve_group_borders, egui::Checkbox::new(&mut p.smooth_group_boundaries, "Smooth group boundaries"));
        ui.checkbox(&mut p.preserve_sharp, "Preserve sharp edges");
        ui.add_enabled(p.preserve_sharp, egui::Slider::new(&mut p.sharp_threshold, 5.0..=90.0).suffix("°").text("Sharp threshold"));
        egui::ComboBox::from_label("Boundary mode")
            .selected_text(match p.boundary {
                RemeshBoundary::Fixed => "Fixed Boundary",
                RemeshBoundary::Refined => "Refined Boundary",
                RemeshBoundary::Free => "Free Boundary",
            })
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut p.boundary, RemeshBoundary::Fixed, "Fixed Boundary");
                ui.selectable_value(&mut p.boundary, RemeshBoundary::Refined, "Refined Boundary");
                ui.selectable_value(&mut p.boundary, RemeshBoundary::Free, "Free Boundary");
            });
        let est = (info.mesh.triangle_count() as f64 * (p.base_edge / p.target()).powi(2)) as usize;
        if info.selection.is_none() {
            ui.weak(format!("≈ {} triangles", crate::ui_util::format_count(est)));
        }
    }
    fn run(mesh: &Mesh, sel: Option<&[u32]>, p: &Self::Params, progress: ProgressFn) -> anyhow::Result<OpOutput> {
        let mut m = mesh.clone();
        let target = p.target();
        if p.mode == RemeshMode::LinearSubdivision {
            let r = mm_remesh::refine_uniform(&mut m, target, sel, Some(progress))?;
            if r.cancelled {
                anyhow::bail!("cancelled");
            }
            return Ok(OpOutput {
                message: format!("{} → {} triangles ({} splits)", mesh.triangle_count(), m.triangle_count(), r.splits),
                selection: r.region_triangles,
                mesh: m,
                ..Default::default()
            });
        }
        let (region, sizes) = match sel {
            Some(sel) if p.transition > 0.0 => {
                let (r, s) = transition_band(mesh, sel, p.transition, target);
                (Some(r), Some(s))
            }
            _ => (sel.map(|s| s.to_vec()), None),
        };
        let params = mm_remesh::RemeshParams {
            target_edge_length: target,
            iterations: p.iterations,
            smoothing: (p.regularity / 100.0).clamp(0.0, 1.0).sqrt(),
            boundary: match p.boundary {
                RemeshBoundary::Fixed => mm_remesh::BoundaryMode::Fixed,
                RemeshBoundary::Refined => mm_remesh::BoundaryMode::ConstrainToBoundary,
                RemeshBoundary::Free => mm_remesh::BoundaryMode::Free,
            },
            preserve_sharp_edges: p.preserve_sharp.then_some(p.sharp_threshold),
            preserve_group_boundaries: p.preserve_group_borders || p.smooth_group_boundaries,
            smooth_group_boundaries: (!p.preserve_group_borders && p.smooth_group_boundaries)
                .then_some(mm_remesh::GroupBorderSmoothing { strength: 0.5, anchor: 0.1, split: None }),
            adaptive: p.mode == RemeshMode::AdaptiveDensity,
            adaptivity: p.threshold / 100.0,
            vertex_sizes: sizes,
            project_to_original: true,
            region,
        };
        let r = mm_remesh::remesh(&mut m, &params, Some(progress))?;
        if r.cancelled {
            anyhow::bail!("cancelled");
        }
        Ok(OpOutput {
            mesh: m,
            selection: r.region_triangles,
            message: format!(
                "{} → {} triangles, mean edge {:.4} ({:.2}s)",
                r.initial_triangles,
                r.final_triangles,
                r.edge_lengths.mean,
                r.elapsed.as_secs_f32()
            ),
            ..Default::default()
        })
    }
}
pub type RemeshTool = OpTool<RemeshSpec>;

// ---------------------------------------------------------------------------
// Smooth Boundary (B with a selection; Meshmixer "Smooth Loop")
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct SmoothBoundaryUi {
    /// 0..100: how strongly the border curve is smoothed each pass.
    pub smoothness: f64,
    /// 0..100: how much of the original border shape is kept.
    pub preserve_shape: f64,
    pub iterations: usize,
    /// Rings of triangles on each side of the border that are re-triangulated.
    pub border_rings: usize,
    pub preserve_mesh_boundary: bool,
    /// Other face-group borders near the selection stay exact (otherwise they are smoothed
    /// too).
    pub preserve_group_borders: bool,
    /// The selected region becomes one new face group.
    pub create_new_groups: bool,
    /// Snap the smoothed band onto the target surface.
    pub project_to_target: bool,
}

pub struct SmoothBoundarySpec;
impl OpSpec for SmoothBoundarySpec {
    type Params = SmoothBoundaryUi;
    const NAME: &'static str = "Smooth Boundary";
    fn defaults(_info: &OpInfo) -> Self::Params {
        // Meshmixer's defaults.
        SmoothBoundaryUi {
            smoothness: 50.0,
            preserve_shape: 50.0,
            iterations: 30,
            border_rings: 2,
            preserve_mesh_boundary: false,
            preserve_group_borders: true,
            create_new_groups: true,
            project_to_target: false,
        }
    }
    fn ui(ui: &mut egui::Ui, p: &mut Self::Params, info: &OpInfo) {
        ui.add(egui::Slider::new(&mut p.smoothness, 0.0..=100.0).text("Smoothness"));
        ui.add(egui::Slider::new(&mut p.preserve_shape, 0.0..=100.0).text("Preserve shape"));
        ui.add(egui::Slider::new(&mut p.iterations, 1..=100).text("Iterations"));
        ui.add(egui::Slider::new(&mut p.border_rings, 1..=8).text("Border rings"));
        ui.checkbox(&mut p.preserve_mesh_boundary, "Preserve mesh boundary");
        ui.checkbox(&mut p.preserve_group_borders, "Preserve group borders");
        ui.checkbox(&mut p.create_new_groups, "Create new groups");
        project_to_target_ui(ui, &mut p.project_to_target, info);
    }
    fn run(mesh: &Mesh, sel: Option<&[u32]>, p: &Self::Params, progress: ProgressFn) -> anyhow::Result<OpOutput> {
        Self::run_with(mesh, sel, p, None, progress)
    }
    fn run_with(mesh: &Mesh, sel: Option<&[u32]>, p: &Self::Params, surface: Option<&TargetSurface>, progress: ProgressFn) -> anyhow::Result<OpOutput> {
        let tris = sel.ok_or_else(|| anyhow::anyhow!("select a region first"))?;
        let mut m = mesh.clone();
        let selection = mm_core::TriSelection::from_tris(&m, tris.iter().copied());
        // Give the selected triangles their own (offset) groups, so the selection border is
        // a group border the remesher keeps as a curve and smooths.
        let offset = m.max_group_id().max(0) + 1;
        for &t in tris {
            m.set_tri_group(t, m.tri_group(t) + offset);
        }
        // Band of triangles around the border (interior border only: open mesh boundaries
        // are not part of the selection border).
        let mut ring: Vec<u32> = Vec::new();
        for t in selection.iter() {
            for e in m.tri_edges(t) {
                let o = m.edge_other_triangle(e, t);
                if o != mm_core::INVALID && !selection.contains(o) {
                    ring.extend(m.edge_vertices(e));
                }
            }
        }
        if ring.is_empty() {
            anyhow::bail!("the selection has no border inside the mesh");
        }
        let mut band = mm_core::TriSelection::new(&m);
        for &v in &ring {
            for t in m.vertex_triangles(v) {
                band.insert(t);
            }
        }
        for _ in 1..p.border_rings {
            band.expand(&m);
        }
        let band: Vec<u32> = band.iter().collect();
        let target = mm_remesh::edge_length_stats_in(&m, &band).mean;
        let params = mm_remesh::RemeshParams {
            target_edge_length: target,
            iterations: p.iterations,
            smoothing: 0.5,
            boundary: if p.preserve_mesh_boundary { mm_remesh::BoundaryMode::Fixed } else { mm_remesh::BoundaryMode::ConstrainToBoundary },
            preserve_sharp_edges: None,
            preserve_group_boundaries: true,
            adaptive: false,
            project_to_original: true,
            region: Some(band),
            smooth_group_boundaries: Some(mm_remesh::GroupBorderSmoothing {
                strength: (p.smoothness / 100.0).clamp(0.0, 1.0),
                anchor: 0.2 * (p.preserve_shape / 100.0).clamp(0.0, 1.0).powi(2),
                split: p.preserve_group_borders.then_some(offset),
            }),
            ..Default::default()
        };
        let r = mm_remesh::remesh(&mut m, &params, Some(progress))?;
        if r.cancelled {
            anyhow::bail!("cancelled");
        }
        if let (true, Some(t), Some(band)) = (p.project_to_target, surface, r.region_triangles.as_deref()) {
            let verts = interior_vertices(&m, band);
            mm_edit::project_to_target(&mut m, &verts, &t.mesh, &t.bvh, 0.0, None);
        }
        // The selection is whatever carries an offset group; restore or merge groups.
        let selected: Vec<u32> = m.triangle_ids().filter(|&t| m.tri_group(t) >= offset).collect();
        for &t in &selected {
            let g = if p.create_new_groups { offset } else { m.tri_group(t) - offset };
            m.set_tri_group(t, g);
        }
        Ok(OpOutput { message: format!("smoothed the border of {} triangles", selected.len()), mesh: m, selection: Some(selected), ..Default::default() })
    }
}
pub type SmoothBoundaryTool = OpTool<SmoothBoundarySpec>;

// ---------------------------------------------------------------------------
// Close Cracks (Edit)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct CloseCracksUi {
    /// Weld tolerance as log10 of a fraction of the model size.
    pub tolerance_exp: f64,
}

pub struct CloseCracksSpec;
impl OpSpec for CloseCracksSpec {
    type Params = CloseCracksUi;
    const NAME: &'static str = "Close Cracks";
    const USES_SELECTION: bool = false;
    fn defaults(_: &OpInfo) -> Self::Params {
        CloseCracksUi { tolerance_exp: -5.0 }
    }
    fn ui(ui: &mut egui::Ui, p: &mut Self::Params, info: &OpInfo) {
        ui.add(egui::Slider::new(&mut p.tolerance_exp, -7.0..=-2.0).text("Tolerance").custom_formatter(|v, _| format!("{:.2e}", 10f64.powf(v) * info.bbox_diag)))
            .on_hover_text("Open edges closer than this are zipped together");
        ui.small(format!("Open edges now: {}", info.mesh.boundary_edge_ids().count()));
        ui.small("Zips cracks: open boundaries lying on top of each other (e.g. CAD surfaces exported without shared vertices), including T-junctions.");
    }
    fn run(mesh: &Mesh, _sel: Option<&[u32]>, p: &Self::Params, _progress: ProgressFn) -> anyhow::Result<OpOutput> {
        let mut m = mesh.clone();
        let before = m.boundary_edge_ids().count();
        let r = mm_repair::close_cracks(&mut m, 10f64.powf(p.tolerance_exp) * mesh.bounds().diagonal());
        let after = m.boundary_edge_ids().count();
        Ok(OpOutput {
            mesh: m,
            message: format!("open edges {before} → {after} ({} welded, {} edges split)", r.weld.welded, r.splits),
            ..Default::default()
        })
    }
}
pub type CloseCracksTool = OpTool<CloseCracksSpec>;

// ---------------------------------------------------------------------------
// Weld Boundaries and Join (selection)
// ---------------------------------------------------------------------------

/// Open (mesh-boundary) edges touching the selection's vertices.
fn selection_open_edges(m: &Mesh, tris: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let mut verts: Vec<u32> = tris.iter().flat_map(|&t| m.triangle(t)).collect();
    verts.sort_unstable();
    verts.dedup();
    let edges: Vec<u32> = m.boundary_edge_ids().filter(|&e| m.edge_vertices(e).iter().any(|v| verts.binary_search(v).is_ok())).collect();
    (verts, edges)
}

#[derive(Clone, Copy, PartialEq, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct WeldBoundariesUi {
    /// Tolerance in multiples of the mean open-edge length near the selection.
    pub tolerance: f64,
}

pub struct WeldBoundariesSpec;
impl OpSpec for WeldBoundariesSpec {
    type Params = WeldBoundariesUi;
    const NAME: &'static str = "Weld Boundaries";
    fn defaults(_: &OpInfo) -> Self::Params {
        WeldBoundariesUi { tolerance: 0.5 }
    }
    fn ui(ui: &mut egui::Ui, p: &mut Self::Params, info: &OpInfo) {
        ui.add(egui::Slider::new(&mut p.tolerance, 0.01..=5.0).logarithmic(true).text("Tolerance (× edge length)"));
        ui.small("Welds the open edges of the selected faces that lie close together.");
        if info.selection.is_none() {
            ui.colored_label(egui::Color32::from_rgb(240, 170, 90), "Select faces along the open edges first.");
        }
    }
    fn run(mesh: &Mesh, sel: Option<&[u32]>, p: &Self::Params, _progress: ProgressFn) -> anyhow::Result<OpOutput> {
        let tris = sel.ok_or_else(|| anyhow::anyhow!("select faces along the open edges first"))?;
        let (verts, edges) = selection_open_edges(mesh, tris);
        if edges.is_empty() {
            anyhow::bail!("the selection has no open edges");
        }
        let mean = edges.iter().map(|&e| mesh.edge_length(e)).sum::<f64>() / edges.len() as f64;
        let mut m = mesh.clone();
        let before = m.boundary_edge_ids().count();
        let r = mm_repair::close_cracks_in(&mut m, p.tolerance * mean, Some(&verts));
        let after = m.boundary_edge_ids().count();
        Ok(OpOutput { mesh: m, message: format!("open edges {before} → {after} ({} welded, {} split)", r.weld.welded, r.splits), ..Default::default() })
    }
}
pub type WeldBoundariesTool = OpTool<WeldBoundariesSpec>;

#[derive(Clone, Copy, PartialEq, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct JoinUi;

pub struct JoinSpec;
impl OpSpec for JoinSpec {
    type Params = JoinUi;
    const NAME: &'static str = "Join";
    fn defaults(_: &OpInfo) -> Self::Params {
        JoinUi
    }
    fn ui(ui: &mut egui::Ui, _p: &mut Self::Params, info: &OpInfo) {
        ui.small("Stitches the two open boundaries next to the selected faces together with one strip of triangles.");
        if info.selection.is_none() {
            ui.colored_label(egui::Color32::from_rgb(240, 170, 90), "Select faces next to the two open boundaries first.");
        }
    }
    fn run(mesh: &Mesh, sel: Option<&[u32]>, _p: &Self::Params, _progress: ProgressFn) -> anyhow::Result<OpOutput> {
        let tris = sel.ok_or_else(|| anyhow::anyhow!("select faces next to two open boundaries"))?;
        let (verts, _) = selection_open_edges(mesh, tris);
        // The two boundary loops the selection touches most.
        let mut loops: Vec<(usize, Vec<u32>)> = mm_core::topology::boundary_loops(mesh)
            .into_iter()
            .map(|l| (l.vertices.iter().filter(|v| verts.binary_search(v).is_ok()).count(), l.vertices))
            .filter(|(n, _)| *n > 0)
            .collect();
        loops.sort_by_key(|(n, _)| std::cmp::Reverse(*n));
        if loops.len() < 2 {
            anyhow::bail!("the selection touches {} open boundary loop(s); Join needs two", loops.len());
        }
        let mut m = mesh.clone();
        let r = mm_edit::join_loops(&mut m, &loops[0].1, &loops[1].1, &mm_edit::BridgeParams { segments: 1, smooth: false })?;
        Ok(OpOutput { message: format!("joined with {} triangles", r.triangles.len()), selection: Some(r.triangles), mesh: m, ..Default::default() })
    }
}
pub type JoinTool = OpTool<JoinSpec>;

// ---------------------------------------------------------------------------
// Bridge / Tube Handle between two selected regions (Cmd+B)
// ---------------------------------------------------------------------------

/// The tube is a loft ([`LoftUi`]): Bridge is straight, Tube Handle a smooth tube leaving
/// each opening along its normal.
#[derive(Clone, Copy, PartialEq, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct RegionBridgeUi {
    pub loft: LoftUi,
    /// 0..100: ring density; 50 matches the regions' edge length, +25 halves it.
    pub refine: f64,
    /// Percent wider (+) or narrower (−) in the middle.
    pub bulge: f64,
}

impl RegionBridgeUi {
    pub fn tube_handle() -> Self {
        Self { loft: LoftUi::handle(), refine: 50.0, bulge: 0.0 }
    }
}

pub struct RegionBridgeSpec;
impl OpSpec for RegionBridgeSpec {
    type Params = RegionBridgeUi;
    const NAME: &'static str = "Bridge";
    fn defaults(_: &OpInfo) -> Self::Params {
        let p = RegionBridgeUi::tube_handle();
        RegionBridgeUi { loft: LoftUi { shape: mm_edit::LoftShape::Straight, ..p.loft }, ..p }
    }
    fn ui(ui: &mut egui::Ui, p: &mut Self::Params, info: &OpInfo) {
        p.loft.ui(ui, &mut p.refine, &mut p.bulge);
        ui.small("Select two separate regions: they are removed and the openings joined by a tube (Straight: Bridge; Smooth: Tube Handle).");
        if info.selection.is_none() {
            ui.colored_label(egui::Color32::from_rgb(240, 170, 90), "Select two regions first.");
        }
    }
    fn script(p: &mut Self::Params, cmd: &str) {
        // The loft's settings (see `LoftUi::script`), `refine:60`, `bulge:20`.
        let Some((key, val)) = cmd.split_once(':') else { return };
        if !p.loft.script(key, val) {
            script_refine_bulge(&mut p.refine, &mut p.bulge, key, val);
        }
    }
    fn run(mesh: &Mesh, sel: Option<&[u32]>, p: &Self::Params, _progress: ProgressFn) -> anyhow::Result<OpOutput> {
        let tris = sel.ok_or_else(|| anyhow::anyhow!("select two regions"))?;
        // The two largest edge-connected parts of the selection.
        let s = mm_core::TriSelection::from_tris(mesh, tris.iter().copied());
        let mut parts: Vec<Vec<u32>> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for &t in tris {
            if seen.contains(&t) {
                continue;
            }
            let part = mm_core::topology::flood_fill_triangles(mesh, &[t], |x| s.contains(x));
            seen.extend(part.iter().copied());
            parts.push(part);
        }
        parts.sort_by_key(|p| std::cmp::Reverse(p.len()));
        if parts.len() < 2 {
            anyhow::bail!("select two separate regions (found {})", parts.len());
        }
        let picked: Vec<u32> = parts[..2].iter().flatten().copied().collect();
        let target = refined_edge_length(mm_remesh::edge_length_stats_in(mesh, &picked).mean, p.refine);
        let mut m = mesh.clone();
        let mut ends = Vec::new();
        for part in &parts[..2] {
            let border: std::collections::HashSet<u32> = mm_core::TriSelection::from_tris(&m, part.iter().copied()).boundary_vertices(&m).into_iter().collect();
            m.remove_triangles(part.iter().copied());
            let lp = mm_core::topology::boundary_loops(&m)
                .into_iter()
                .filter(|l| l.vertices.iter().any(|v| border.contains(v)))
                .max_by_key(|l| l.vertices.len())
                .ok_or_else(|| anyhow::anyhow!("a region did not leave an opening (does it cover a whole part?)"))?;
            ends.push(lp);
        }
        // End 1 is the opening with the lowest vertex id, as in Erase & Fill's Loft.
        ends.sort_by_key(|l| l.vertices.iter().min().copied());
        let labels = ends.iter().enumerate().map(|(k, l)| (l.centroid(&m), format!("{}", k + 1))).collect();
        let r = run_loft(&mut m, &ends[0].vertices, &ends[1].vertices, &p.loft, target, p.bulge, None)?;
        let message = format!("bridged with {} triangles{}", r.triangles.len(), crossing_warning(&m, &r.triangles));
        Ok(OpOutput { message, selection: Some(r.triangles), mesh: m, labels, ..Default::default() })
    }
}
pub type RegionBridgeTool = OpTool<RegionBridgeSpec>;

// ---------------------------------------------------------------------------
// Generate Face Groups (Edit)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct FaceGroupsUi {
    pub mode: mm_edit::FaceGroupMode,
    pub angle: f64,
    /// Groups with fewer triangles are merged into a neighbour.
    pub min_size: usize,
}

pub struct FaceGroupsSpec;
impl OpSpec for FaceGroupsSpec {
    type Params = FaceGroupsUi;
    const NAME: &'static str = "Generate Face Groups";
    const USES_SELECTION: bool = false;
    const SHOWS_GROUPS: bool = true;
    fn defaults(_: &OpInfo) -> Self::Params {
        FaceGroupsUi { mode: mm_edit::FaceGroupMode::EdgeAngle, angle: 10.0, min_size: 0 }
    }
    fn ui(ui: &mut egui::Ui, p: &mut Self::Params, _info: &OpInfo) {
        egui::ComboBox::from_label("Mode")
            .selected_text(match p.mode {
                mm_edit::FaceGroupMode::EdgeAngle => "Edge Angle",
                mm_edit::FaceGroupMode::NormalAngle => "Normal Angle",
            })
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut p.mode, mm_edit::FaceGroupMode::EdgeAngle, "Edge Angle").on_hover_text("Split along edges sharper than the threshold");
                ui.selectable_value(&mut p.mode, mm_edit::FaceGroupMode::NormalAngle, "Normal Angle").on_hover_text("Near-planar patches: faces within the threshold of each patch's first face");
            });
        ui.add(egui::Slider::new(&mut p.angle, 1.0..=90.0).suffix("°").text("Angle threshold"));
        ui.add(egui::Slider::new(&mut p.min_size, 0..=2000).logarithmic(true).text("Size threshold (triangles)"));
        ui.small("Replaces all face groups. Group colours are shown while this tool is open.");
    }
    fn run(mesh: &Mesh, _sel: Option<&[u32]>, p: &Self::Params, _progress: ProgressFn) -> anyhow::Result<OpOutput> {
        let mut m = mesh.clone();
        let n = mm_edit::generate_face_groups(&mut m, p.mode, p.angle, p.min_size);
        Ok(OpOutput { mesh: m, message: format!("{n} face groups"), ..Default::default() })
    }
}
pub type FaceGroupsTool = OpTool<FaceGroupsSpec>;

// ---------------------------------------------------------------------------
// Reduce (Shift+R)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub enum ReduceGoal {
    Percentage,
    TriangleBudget,
    MaxDeviation,
}

#[derive(Clone, Copy, PartialEq, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct ReduceUi {
    pub goal: ReduceGoal,
    /// Uniform (even triangles) instead of Shape Preserving (quadric error).
    pub uniform: bool,
    pub percentage: f64,
    pub budget: usize,
    pub deviation: f64,
    pub preserve_boundary: bool,
    pub preserve_groups: bool,
}

pub struct ReduceSpec;
impl OpSpec for ReduceSpec {
    type Params = ReduceUi;
    const NAME: &'static str = "Reduce";
    fn keys(p: &mut Self::Params, i: &egui::InputState) -> bool {
        let before = *p;
        // D / F: keep 10 % more / less.
        if plain(i, egui::Key::D) || plain(i, egui::Key::F) {
            let step = if plain(i, egui::Key::D) { 10.0 } else { -10.0 };
            p.goal = ReduceGoal::Percentage;
            p.percentage = (p.percentage + step).clamp(1.0, 99.0);
        }
        if plain(i, egui::Key::G) {
            p.preserve_groups = !p.preserve_groups;
        }
        if plain(i, egui::Key::T) {
            p.uniform = !p.uniform;
        }
        *p != before
    }
    fn defaults(info: &OpInfo) -> Self::Params {
        let n = info.selection.map(|s| s.len()).unwrap_or(info.mesh.triangle_count());
        ReduceUi {
            goal: ReduceGoal::Percentage,
            uniform: false,
            percentage: 50.0,
            budget: (n / 2).max(4),
            deviation: info.avg_edge * 0.1,
            preserve_boundary: true,
            preserve_groups: true,
        }
    }
    fn ui(ui: &mut egui::Ui, p: &mut Self::Params, info: &OpInfo) {
        egui::ComboBox::from_label("Reduce target")
            .selected_text(match p.goal {
                ReduceGoal::Percentage => "Percentage",
                ReduceGoal::TriangleBudget => "Triangle Budget",
                ReduceGoal::MaxDeviation => "Max Deviation",
            })
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut p.goal, ReduceGoal::Percentage, "Percentage");
                ui.selectable_value(&mut p.goal, ReduceGoal::TriangleBudget, "Triangle Budget");
                ui.selectable_value(&mut p.goal, ReduceGoal::MaxDeviation, "Max Deviation");
            });
        egui::ComboBox::from_label("Reduce type")
            .selected_text(if p.uniform { "Uniform" } else { "Shape Preserving" })
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut p.uniform, true, "Uniform").on_hover_text("Even triangles (T)");
                ui.selectable_value(&mut p.uniform, false, "Shape Preserving").on_hover_text("Least shape change (T)");
            });
        let n = info.selection.map(|s| s.len()).unwrap_or(info.mesh.triangle_count());
        match p.goal {
            ReduceGoal::Percentage => {
                ui.add(egui::Slider::new(&mut p.percentage, 1.0..=99.0).suffix(" %").text("keep"));
            }
            ReduceGoal::TriangleBudget => {
                ui.add(egui::Slider::new(&mut p.budget, 4..=n.max(5)).logarithmic(true).text("triangles"));
            }
            ReduceGoal::MaxDeviation => {
                ui.add(egui::Slider::new(&mut p.deviation, (info.avg_edge * 0.001)..=(info.avg_edge * 5.0)).logarithmic(true).text("max deviation"));
            }
        }
        ui.checkbox(&mut p.preserve_boundary, "Preserve boundaries");
        ui.checkbox(&mut p.preserve_groups, "Preserve group boundaries");
    }
    fn run(mesh: &Mesh, sel: Option<&[u32]>, p: &Self::Params, progress: ProgressFn) -> anyhow::Result<OpOutput> {
        let mut m = mesh.clone();
        let params = mm_remesh::ReduceParams {
            target: match p.goal {
                ReduceGoal::Percentage => mm_remesh::ReduceTarget::Percentage(p.percentage),
                ReduceGoal::TriangleBudget => mm_remesh::ReduceTarget::TriangleCount(p.budget),
                ReduceGoal::MaxDeviation => mm_remesh::ReduceTarget::MaxDeviation(p.deviation),
            },
            preserve_boundary: p.preserve_boundary,
            preserve_group_boundaries: p.preserve_groups,
            uniform: p.uniform,
            region: sel.map(|s| s.to_vec()),
            ..Default::default()
        };
        let r = mm_remesh::reduce(&mut m, &params, Some(progress))?;
        if r.cancelled {
            anyhow::bail!("cancelled");
        }
        let dev = r.max_deviation.map(|d| format!(", max deviation {d:.4}")).unwrap_or_default();
        Ok(OpOutput {
            mesh: m,
            selection: r.region_triangles,
            message: format!("{} → {} triangles{dev} ({:.2}s)", r.initial_triangles, r.final_triangles, r.elapsed.as_secs_f32()),
            ..Default::default()
        })
    }
}
pub type ReduceTool = OpTool<ReduceSpec>;

// ---------------------------------------------------------------------------
// Make Solid
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub enum SolidType {
    Blocky,
    Fast,
    Accurate,
    SharpEdges,
}

#[derive(Clone, Copy, PartialEq, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct MakeSolidUi {
    pub solid_type: SolidType,
    /// 0..1 → 64..1024 cells along the largest dimension.
    pub accuracy: f64,
    /// 0..1 → coarse..fine output.
    pub density: f64,
    pub offset: f64,
    pub use_min_thickness: bool,
    pub min_thickness: f64,
    pub close_holes: bool,
    pub remove_small: bool,
    pub remesh_result: bool,
    pub transfer_groups: bool,
    /// "Cull Edges Threshold", 0..100: scales the short-edge collapse from none to full.
    pub cull_edges: f64,
    pub color_transfer: ColorTransfer,
}

/// Make Solid's "Color Transfer Mode".
#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub enum ColorTransfer {
    /// Vertex-to-vertex when the input has vertex colours.
    Automatic,
    NoColors,
    VertexToVertex,
}

impl MakeSolidUi {
    fn cells(&self) -> u32 {
        (64.0 * 16f64.powf(self.accuracy)).round() as u32
    }
}

pub struct MakeSolidSpec;
impl OpSpec for MakeSolidSpec {
    type Params = MakeSolidUi;
    const NAME: &'static str = "Make Solid";
    const USES_SELECTION: bool = false;
    fn defaults(info: &OpInfo) -> Self::Params {
        MakeSolidUi {
            solid_type: SolidType::Accurate,
            accuracy: 0.5,
            density: 0.5,
            offset: 0.0,
            use_min_thickness: false,
            min_thickness: info.avg_edge * 2.0,
            close_holes: true,
            remove_small: true,
            remesh_result: true,
            transfer_groups: false,
            cull_edges: 100.0,
            color_transfer: ColorTransfer::Automatic,
        }
    }
    fn ui(ui: &mut egui::Ui, p: &mut Self::Params, info: &OpInfo) {
        ui.horizontal(|ui| {
            ui.label("Solid type");
            ui.selectable_value(&mut p.solid_type, SolidType::Blocky, "Blocky");
            ui.selectable_value(&mut p.solid_type, SolidType::Fast, "Fast");
            ui.selectable_value(&mut p.solid_type, SolidType::Accurate, "Accurate");
            ui.selectable_value(&mut p.solid_type, SolidType::SharpEdges, "Sharp Edge Preserve")
                .on_hover_text("Keep the crisp edges and corners of CAD parts");
        });
        let ext = info.mesh.bounds().extents();
        let m = ext.x.max(ext.y).max(ext.z);
        ui.add(egui::Slider::new(&mut p.accuracy, 0.0..=1.0).text("solid accuracy").show_value(false));
        ui.weak(format!("{} cells, voxel size {:.4}", p.cells(), m / p.cells() as f64));
        ui.add(egui::Slider::new(&mut p.density, 0.0..=1.0).text("mesh density"));
        ui.add(egui::Slider::new(&mut p.cull_edges, 0.0..=100.0).text("cull edges threshold").integer())
            .on_hover_text("How short an edge must be (relative to the cell size) to be collapsed; 0 keeps every edge");
        let span = (info.bbox_diag * 0.05).max(info.avg_edge * 5.0);
        ui.add(egui::Slider::new(&mut p.offset, -span..=span).text("offset distance"));
        ui.horizontal(|ui| {
            ui.checkbox(&mut p.use_min_thickness, "Min thickness");
            if p.use_min_thickness {
                ui.add(egui::DragValue::new(&mut p.min_thickness).speed(info.avg_edge * 0.05).range(0.0..=span));
            }
        });
        ui.checkbox(&mut p.close_holes, "Close open boundaries");
        ui.checkbox(&mut p.remove_small, "Remove small floating parts");
        ui.checkbox(&mut p.remesh_result, "Remesh result (better triangles for CFD)");
        ui.checkbox(&mut p.transfer_groups, "Transfer groups").on_hover_text("Copy the face groups of the input onto the solid");
        egui::ComboBox::from_label("Color transfer")
            .selected_text(match p.color_transfer {
                ColorTransfer::Automatic => "Automatic",
                ColorTransfer::NoColors => "No Colors",
                ColorTransfer::VertexToVertex => "Vertex-to-Vertex",
            })
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut p.color_transfer, ColorTransfer::Automatic, "Automatic").on_hover_text("Keep the vertex colours if the input has any");
                ui.selectable_value(&mut p.color_transfer, ColorTransfer::NoColors, "No Colors");
                ui.selectable_value(&mut p.color_transfer, ColorTransfer::VertexToVertex, "Vertex-to-Vertex");
            });
        if p.cells() >= 512 {
            ui.colored_label(egui::Color32::from_rgb(240, 170, 90), "High accuracy: this can take a while and use a lot of memory.");
        }
    }
    fn run(mesh: &Mesh, _: Option<&[u32]>, p: &Self::Params, progress: ProgressFn) -> anyhow::Result<OpOutput> {
        let mut params = mm_solid::MakeSolidParams::from_sliders(mesh, p.accuracy, p.density);
        params.mode = match p.solid_type {
            SolidType::Blocky => mm_solid::SolidMode::Blocky,
            SolidType::Fast => mm_solid::SolidMode::Fast,
            SolidType::Accurate => mm_solid::SolidMode::Accurate,
            SolidType::SharpEdges => mm_solid::SolidMode::SharpEdges,
        };
        // Groups and colours go on after the remesh so its group constraints don't keep the
        // raw borders.
        params.transfer_groups = p.transfer_groups && !p.remesh_result;
        let colors = p.color_transfer != ColorTransfer::NoColors && mesh.has_vertex_colors();
        params.transfer_colors = colors && !p.remesh_result;
        params.offset = p.offset;
        params.min_edge_length = params.min_edge_length.map(|l| l * (p.cull_edges / 100.0).clamp(0.0, 1.0));
        params.min_thickness = p.use_min_thickness.then_some(p.min_thickness);
        params.close_holes = p.close_holes;
        if !p.remove_small {
            params.min_shell_volume = Some(0.0);
        }
        let t0 = std::time::Instant::now();
        let solid_progress = |f: f32| progress(if p.remesh_result { f * 0.7 } else { f });
        let mut out = mm_solid::make_solid(mesh, &params, Some(&solid_progress)).map_err(|e| anyhow::anyhow!("{e}"))?;
        if p.remesh_result {
            let target = params.cell_size * (1.5 + (0.3 - 1.5) * p.density.clamp(0.0, 1.0)).max(0.5);
            let sharp = (p.solid_type == SolidType::SharpEdges).then_some(params.sharp_angle_deg);
            let rp = mm_remesh::RemeshParams { target_edge_length: target, iterations: 6, preserve_sharp_edges: sharp, ..Default::default() };
            let remesh_progress = |f: f32| progress(0.7 + 0.3 * f);
            mm_remesh::remesh(&mut out, &rp, Some(&remesh_progress))?;
            if p.transfer_groups {
                mm_solid::transfer_face_groups(&mut out, mesh);
            }
            if colors {
                mm_solid::transfer_vertex_colors(&mut out, mesh);
            }
        }
        let shells = mm_core::topology::connected_components(&out).len();
        Ok(OpOutput {
            message: format!(
                "{} triangles, {} shell(s), closed: {} ({:.1}s)",
                out.triangle_count(),
                shells,
                if out.is_closed() { "yes" } else { "no" },
                t0.elapsed().as_secs_f32()
            ),
            mesh: out,
            selection: None,
            ..Default::default()
        })
    }
}
pub type MakeSolidTool = OpTool<MakeSolidSpec>;

// ---------------------------------------------------------------------------
// Offset / Hollow (robust, SDF based)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct OffsetUi {
    pub hollow: bool,
    pub distance: f64,
    pub accuracy: f64,
}

pub struct OffsetSolidSpec;
impl OpSpec for OffsetSolidSpec {
    type Params = OffsetUi;
    const NAME: &'static str = "Offset / Hollow";
    const USES_SELECTION: bool = false;
    fn defaults(info: &OpInfo) -> Self::Params {
        OffsetUi { hollow: false, distance: info.avg_edge * 2.0, accuracy: 0.5 }
    }
    fn ui(ui: &mut egui::Ui, p: &mut Self::Params, info: &OpInfo) {
        ui.horizontal(|ui| {
            ui.selectable_value(&mut p.hollow, false, "Offset surface");
            ui.selectable_value(&mut p.hollow, true, "Hollow (wall)");
        });
        let span = (info.bbox_diag * 0.1).max(info.avg_edge * 10.0);
        let range = if p.hollow { (info.avg_edge * 0.1)..=span } else { -span..=span };
        ui.add(egui::Slider::new(&mut p.distance, range).text(if p.hollow { "wall thickness" } else { "distance" }));
        ui.add(egui::Slider::new(&mut p.accuracy, 0.0..=1.0).text("accuracy"));
        ui.small("Robust volumetric offset: the result is always a closed, manifold solid. Negative distances shrink.");
    }
    fn run(mesh: &Mesh, _: Option<&[u32]>, p: &Self::Params, progress: ProgressFn) -> anyhow::Result<OpOutput> {
        let cells = (64.0 * 16f64.powf(p.accuracy)).round() as u32;
        let h = mm_solid::cell_size_for_resolution(&mesh.bounds().expanded(p.distance.abs()), cells);
        let out = if p.hollow {
            mm_solid::hollow(mesh, p.distance, h, Some(&|f| progress(f)))
        } else {
            mm_solid::offset_solid(mesh, p.distance, h, Some(&|f| progress(f)))
        }
        .map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(OpOutput { message: format!("{} triangles", out.triangle_count()), mesh: out, selection: None, ..Default::default() })
    }
}
pub type OffsetSolidTool = OpTool<OffsetSolidSpec>;

// ---------------------------------------------------------------------------
// Erase & Fill (F)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub enum FillKind {
    /// Fewest triangles, no new vertices.
    FlatMinimal,
    /// Flat patch with triangles matching the surroundings.
    FlatRemeshed,
    /// Curved patch continuing the surrounding surface (Meshmixer "Smooth MVC").
    SmoothMvc,
    /// Join two openings with a tube (Polymix).
    Loft,
}

impl FillKind {
    const ALL: [FillKind; 4] = [FillKind::FlatMinimal, FillKind::FlatRemeshed, FillKind::SmoothMvc, FillKind::Loft];

    fn label(self) -> &'static str {
        match self {
            FillKind::FlatMinimal => "Flat Minimal",
            FillKind::FlatRemeshed => "Flat Remeshed",
            FillKind::SmoothMvc => "Smooth MVC",
            FillKind::Loft => "Loft (join two holes)",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct EraseFillUi {
    pub kind: FillKind,
    /// 0..100: patch density; 50 matches the surrounding edge length, +25 halves it.
    pub refine: f64,
    /// 0..100: how much of the smooth patch's curvature is kept (0 = flat).
    pub smooth: f64,
    /// Multiplies the patch's height above the hole's plane.
    pub scale: f64,
    /// -100..100: pushes the middle of the patch out (+) or in (−), relative to the hole size.
    pub bulge: f64,
    /// The patch gets its own face group (otherwise the surrounding group).
    pub create_new_groups: bool,
    /// Snap the patch onto the target surface.
    pub project_to_target: bool,
    pub loft: LoftUi,
}

/// Whether `sel` is the edge rings of two holes: two separate parts, each touching an open
/// edge of the mesh (what double-clicking an open edge selects).
fn is_two_hole_rings(mesh: &Mesh, sel: &[u32]) -> bool {
    let set = mm_core::TriSelection::from_tris(mesh, sel.iter().copied());
    let mut seen = std::collections::HashSet::new();
    let mut parts = 0;
    for &t in sel {
        if seen.contains(&t) {
            continue;
        }
        let part = mm_core::topology::flood_fill_triangles(mesh, &[t], |x| set.contains(x));
        let open = part.iter().any(|&x| mesh.tri_edges(x).iter().any(|&e| mesh.is_boundary_edge(e)));
        if !open {
            return false;
        }
        seen.extend(part);
        parts += 1;
        if parts > 2 {
            return false;
        }
    }
    parts == 2
}

/// Pairs of triangles where `part` crosses the rest of the mesh or itself (neighbours
/// sharing a vertex are not counted).
pub(crate) fn crossings(m: &Mesh, part: &[u32]) -> usize {
    if part.is_empty() {
        return 0;
    }
    let bvh = mm_core::Bvh::build_subset(m, part);
    let inside: std::collections::HashSet<u32> = part.iter().copied().collect();
    let bounds = mm_core::Aabb::from_points(part.iter().flat_map(|&t| m.tri_positions(t)));
    m.triangle_ids()
        .filter(|&t| mm_core::Aabb::from_points(m.tri_positions(t)).intersects(&bounds))
        .map(|t| {
            let (tv, tp) = (m.triangle(t), m.tri_positions(t));
            let mut n = 0;
            bvh.query_aabb(&mm_core::Aabb::from_points(tp), |o| {
                if (inside.contains(&t) && o <= t) || m.triangle(o).iter().any(|v| tv.contains(v)) {
                    return;
                }
                if mm_core::intersect::triangles_intersect(tp, m.tri_positions(o)) {
                    n += 1;
                }
            });
            n
        })
        .sum()
}

pub struct EraseFillSpec;
impl OpSpec for EraseFillSpec {
    type Params = EraseFillUi;
    const NAME: &'static str = "Erase & Fill";
    fn defaults(info: &OpInfo) -> Self::Params {
        // Meshmixer's defaults; the edge rings of two holes are lofted together.
        let loft = info.selection.is_some_and(|s| is_two_hole_rings(info.mesh, s));
        EraseFillUi {
            kind: if loft { FillKind::Loft } else { FillKind::SmoothMvc },
            refine: 50.0,
            smooth: 100.0,
            scale: 1.0,
            bulge: 0.0,
            create_new_groups: false,
            project_to_target: false,
            loft: LoftUi::default(),
        }
    }
    fn ui(ui: &mut egui::Ui, p: &mut Self::Params, info: &OpInfo) {
        egui::ComboBox::from_label("Fill type").selected_text(p.kind.label()).show_ui(ui, |ui| {
            for k in FillKind::ALL {
                ui.selectable_value(&mut p.kind, k, k.label());
            }
        });
        if p.kind == FillKind::Loft {
            p.loft.ui(ui, &mut p.refine, &mut p.bulge);
            ui.checkbox(&mut p.create_new_groups, "Create new groups");
            ui.small("Deletes the selection and joins the two openings it leaves with a tube.");
            if info.selection.is_none() {
                ui.colored_label(egui::Color32::from_rgb(240, 170, 90), "Select the edge rings of two holes first (double-click an open edge).");
            }
            return;
        }
        let patch = p.kind != FillKind::FlatMinimal;
        ui.add_enabled(patch, egui::Slider::new(&mut p.refine, 0.0..=100.0).text("Refine"));
        ui.add_enabled(p.kind == FillKind::SmoothMvc, egui::Slider::new(&mut p.smooth, 0.0..=100.0).text("Smooth"));
        ui.add_enabled(patch, egui::Slider::new(&mut p.scale, 0.0..=4.0).text("Scale"));
        ui.add_enabled(patch, egui::Slider::new(&mut p.bulge, -100.0..=100.0).text("Bulge"));
        ui.checkbox(&mut p.create_new_groups, "Create new groups");
        project_to_target_ui(ui, &mut p.project_to_target, info);
        ui.small("Deletes the selection and fills the opening so it blends into the surrounding surface.");
        if info.selection.is_none() {
            ui.colored_label(egui::Color32::from_rgb(240, 170, 90), "Select a region first.");
        }
    }
    fn script(p: &mut Self::Params, cmd: &str) {
        // The loft's settings (see `LoftUi::script`), `refine:60`, `bulge:20`.
        let Some((key, val)) = cmd.split_once(':') else { return };
        if key == "loft" {
            p.kind = FillKind::Loft;
        }
        if !p.loft.script(key, val) {
            script_refine_bulge(&mut p.refine, &mut p.bulge, key, val);
        }
    }
    fn run(mesh: &Mesh, sel: Option<&[u32]>, p: &Self::Params, progress: ProgressFn) -> anyhow::Result<OpOutput> {
        Self::run_with(mesh, sel, p, None, progress)
    }
    fn run_with(mesh: &Mesh, sel: Option<&[u32]>, p: &Self::Params, surface: Option<&TargetSurface>, progress: ProgressFn) -> anyhow::Result<OpOutput> {
        let tris = sel.ok_or_else(|| anyhow::anyhow!("select a region first"))?;
        let mut m = mesh.clone();
        let selection = mm_core::TriSelection::from_tris(&m, tris.iter().copied());
        let border: std::collections::HashSet<u32> = selection.boundary_vertices(&m).into_iter().collect();
        let target = refined_edge_length(mm_remesh::edge_length_stats_in(&m, tris).mean, p.refine);
        let new_group = m.max_group_id().max(0) + 1;
        let erased = mm_core::Aabb::from_points(tris.iter().flat_map(|&t| m.tri_positions(t)));
        m.remove_triangles(tris.iter().copied());
        let fragments = remove_cut_off_fragments(&mut m, &border, &erased);
        let loops: Vec<_> = mm_core::topology::boundary_loops(&m).into_iter().filter(|l| l.vertices.iter().any(|v| border.contains(v))).collect();
        if loops.is_empty() {
            anyhow::bail!("the selection did not leave a hole to fill (did it cover a whole part?)");
        }
        let mut loops = loops;
        let mut new_tris = Vec::new();
        let mut labels = Vec::new();
        let mut warning = String::new();
        let surrounding_group = |m: &Mesh, lp: &mm_core::topology::EdgeLoop| lp.edges.first().map(|&e| m.tri_group(m.edge_triangles(e)[0])).unwrap_or(0);
        if p.kind == FillKind::Loft {
            if loops.len() < 2 {
                anyhow::bail!("Loft joins two openings: select the edge rings of two holes (this selection leaves {})", loops.len());
            }
            // The two longest openings; End 1 is the one with the lowest vertex id, so the
            // ends keep their numbers while the settings change.
            loops.sort_by_key(|l| std::cmp::Reverse(l.vertices.len()));
            let mut ends: Vec<mm_core::topology::EdgeLoop> = loops.drain(..2).collect();
            ends.sort_by_key(|l| l.vertices.iter().min().copied());
            for (k, lp) in ends.iter().enumerate() {
                labels.push((lp.centroid(&m), format!("{}", k + 1)));
            }
            let group = if p.create_new_groups { new_group } else { surrounding_group(&m, &ends[0]) };
            let r = run_loft(&mut m, &ends[0].vertices, &ends[1].vertices, &p.loft, target, p.bulge, Some(group))?;
            warning = crossing_warning(&m, &r.triangles);
            new_tris.extend(r.triangles);
        }
        let mode = match p.kind {
            FillKind::FlatMinimal => mm_repair::FillMode::Minimal,
            FillKind::FlatRemeshed => mm_repair::FillMode::Flat,
            FillKind::SmoothMvc | FillKind::Loft => mm_repair::FillMode::Smooth,
        };
        let n = loops.len();
        for (k, lp) in loops.iter().enumerate() {
            if !progress(k as f32 / n as f32) {
                anyhow::bail!("cancelled");
            }
            let group = if p.create_new_groups { new_group } else { surrounding_group(&m, lp) };
            let fp = mm_repair::FillParams { mode, target_edge_length: Some(target), group: Some(group), ..Default::default() };
            let r = mm_repair::fill_hole(&mut m, lp, &fp)?;
            if p.kind != FillKind::Loft {
                shape_patch(&mut m, lp, &r, p);
            }
            new_tris.extend(r.new_triangles);
        }
        if let (true, Some(t), false) = (p.project_to_target, surface, p.kind == FillKind::Loft) {
            let verts = interior_vertices(&m, &new_tris);
            mm_edit::project_to_target(&mut m, &verts, &t.mesh, &t.bvh, 0.0, None);
        }
        let mut message = format!("replaced {} triangles with {}", tris.len(), new_tris.len());
        if fragments > 0 {
            message += &format!(", removed {fragments} loose fragment{}", if fragments == 1 { "" } else { "s" });
        }
        message += &warning;
        Ok(OpOutput { message, mesh: m, selection: Some(new_tris), labels, ..Default::default() })
    }
}

/// After erasing a selection, delete the pieces it cut loose: parts of the surface left
/// within the erased region (`erased`, its bounding box) and attached to the rest only
/// through it, such as the far side of a spike the brush did not reach (faces turned away
/// from the eye). Filled, they would float in space as small closed blobs. The largest
/// piece next to the hole is always kept, as is anything reaching outside the region (both
/// halves of a tube cut through). Returns how many pieces were removed.
fn remove_cut_off_fragments(m: &mut Mesh, border: &std::collections::HashSet<u32>, erased: &mm_core::Aabb) -> usize {
    let mut pieces: Vec<Vec<u32>> = mm_core::topology::connected_components(m)
        .into_iter()
        .filter(|c| c.iter().any(|&t| m.triangle(t).iter().any(|v| border.contains(v))))
        .collect();
    if pieces.len() < 2 {
        return 0;
    }
    let area = |c: &Vec<u32>| c.iter().map(|&t| m.tri_area(t)).sum::<f64>();
    pieces.sort_by(|a, b| area(b).total_cmp(&area(a)));
    let inside = erased.expanded(erased.diagonal() * 0.05);
    let cut_off: Vec<Vec<u32>> = pieces.into_iter().skip(1).filter(|c| c.iter().flat_map(|&t| m.tri_positions(t)).all(|p| inside.contains(p))).collect();
    for c in &cut_off {
        m.remove_triangles(c.iter().copied());
    }
    cut_off.len()
}

/// Loft settings, shared by Erase & Fill's Loft, Bridge / Tube Handle (Cmd+B) and Vessel →
/// Bridge boundaries. Refine and bulge belong to each tool (Erase & Fill shares them with
/// its patch fills), so they are passed to [`LoftUi::ui`] and [`LoftUi::params`].
#[derive(Clone, Copy, PartialEq, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct LoftUi {
    pub shape: mm_edit::LoftShape,
    pub ends: [mm_edit::LoftEnd; 2],
    /// Extra twist (degrees); roundness of the middle, fairing, how much of the vessels
    /// behind the openings steers the tube and how far it is blended into them (percent).
    pub twist: f64,
    pub round: f64,
    pub fair: f64,
    /// Per end.
    pub context: [f64; 2],
    pub blend: f64,
    /// Even triangles over the tube and the blended bands.
    pub remesh: bool,
    /// Fit each end's axis to its vessel's centerline (traced from the opening) instead of
    /// slices of the wall.
    pub centerline: bool,
}

impl Default for LoftUi {
    fn default() -> Self {
        Self { shape: mm_edit::LoftShape::Smooth, ends: [mm_edit::LoftEnd::default(); 2], twist: 0.0, round: 0.0, fair: 0.0, context: [50.0; 2], blend: 50.0, remesh: true, centerline: false }
    }
}

impl LoftUi {
    /// A tube standing up from two openings in a surface (Tube Handle). Its ends turn half
    /// round, where tension 1 (an arc's length) gives a flattened arch whose cross-section
    /// can fold; about 4/π gives a round one.
    pub fn handle() -> Self {
        let end = mm_edit::LoftEnd { direction: mm_edit::EndDirection::Normal, tension: 1.3, ..Default::default() };
        Self { ends: [end; 2], ..Default::default() }
    }

    /// The library's parameters; `edge_length` is the target edge length (Refine applied),
    /// `bulge` in percent.
    pub fn params(&self, edge_length: f64, bulge: f64, group: Option<i32>) -> mm_edit::LoftParams {
        let mut ends = self.ends;
        for (e, c) in ends.iter_mut().zip(self.context) {
            e.context = (c / 100.0).clamp(0.0, 1.0);
        }
        mm_edit::LoftParams {
            shape: self.shape,
            ends,
            edge_length: Some(edge_length),
            twist_deg: self.twist,
            bulge: (bulge / 100.0).max(-0.9),
            round: (self.round / 100.0).clamp(0.0, 1.0),
            fair: (self.fair / 100.0).clamp(0.0, 1.0),
            blend: (self.blend / 100.0).clamp(0.0, 1.0),
            remesh: self.remesh && self.shape != mm_edit::LoftShape::Minimal,
            group,
        }
    }

    pub fn ui(&mut self, ui: &mut egui::Ui, refine: &mut f64, bulge: &mut f64) {
        use mm_edit::{EndContinuity, EndDirection, LoftShape};
        ui.horizontal(|ui| {
            ui.label("Shape");
            ui.selectable_value(&mut self.shape, LoftShape::Minimal, "Minimal").on_hover_text("One band of triangles straight across, no new vertices");
            ui.selectable_value(&mut self.shape, LoftShape::Straight, "Straight").on_hover_text("Straight tube, like a cylinder, blending from one opening's shape to the other's");
            ui.selectable_value(&mut self.shape, LoftShape::Smooth, "Smooth").on_hover_text("Curved tube that meets each opening as set below");
            ui.selectable_value(&mut self.shape, LoftShape::Sweep, "Along the axis").on_hover_text("The tube follows a curve joining the two vessels' axes, its cross-section morphing from one opening's shape to the other's (no pinching between differently shaped or turned openings). Needs a vessel behind each opening");
        });
        let tube = self.shape != LoftShape::Minimal;
        let smooth = matches!(self.shape, LoftShape::Smooth | LoftShape::Sweep);
        ui.add_enabled(tube, egui::Slider::new(refine, 0.0..=100.0).text("Refine"));
        ui.add_enabled_ui(smooth, |ui| {
            ui.horizontal(|ui| {
                ui.label("Leave");
                let mut d = self.ends[0].direction;
                ui.selectable_value(&mut d, EndDirection::Wall, "Along the wall").on_hover_text("The tube continues the surrounding wall, e.g. a vessel cut across");
                ui.selectable_value(&mut d, EndDirection::Normal, "Out of the opening").on_hover_text("The tube stands up from the surface around each opening, like a handle");
                if d != self.ends[0].direction {
                    self.ends.iter_mut().for_each(|e| e.direction = d);
                }
            });
        });
        let along = if self.ends[0].direction == EndDirection::Wall { "the surrounding wall" } else { "the opening's normal" };
        egui::Grid::new("loft_ends").num_columns(3).spacing([8.0, 4.0]).show(ui, |ui| {
            for (k, end) in self.ends.iter_mut().enumerate() {
                ui.label(format!("End {}", k + 1)).on_hover_text("Marked with its number in the view");
                ui.add_enabled_ui(smooth, |ui| {
                    ui.horizontal(|ui| {
                        ui.selectable_value(&mut end.continuity, EndContinuity::G0, "G0").on_hover_text("Position: the tube meets the opening (a crease may form)");
                        ui.selectable_value(&mut end.continuity, EndContinuity::G1, "G1").on_hover_text(format!("Tangent: the tube leaves along {along}"));
                        ui.selectable_value(&mut end.continuity, EndContinuity::G2, "G2").on_hover_text("Curvature: the tube also continues the vessel's bend (starts straight when leaving out of the opening)");
                    });
                });
                ui.add_enabled(smooth && end.continuity != EndContinuity::G0, egui::Slider::new(&mut end.tension, 0.2..=3.0).text("tension"))
                    .on_hover_text("How far the tube keeps to its starting direction before turning (1 follows a circular arc)");
                ui.end_row();
            }
        });
        ui.add_enabled(tube, egui::Slider::new(&mut self.twist, -180.0..=180.0).suffix("°").text("Twist")).on_hover_text("Turn one end against the other (they are lined up automatically)");
        ui.add_enabled(tube, egui::Slider::new(bulge, -90.0..=100.0).suffix("%").text("Bulge")).on_hover_text("Widen (+) or narrow (−) the middle of the tube");
        ui.add_enabled(tube, egui::Slider::new(&mut self.round, 0.0..=100.0).suffix("%").text("Round")).on_hover_text("Make the middle of the tube circular, however ragged the openings are");
        ui.add_enabled(smooth, egui::Slider::new(&mut self.fair, 0.0..=100.0).suffix("%").text("Fair"))
            .on_hover_text("Smooth the tube's middle with its ends kept: thin-plate for G0 / G1 ends, curvature-continuous (G2) when an end is G2");
        let along_wall = smooth && self.ends.iter().any(|e| e.direction == EndDirection::Wall);
        let context_tip = "How far back along each vessel its end of the tube takes its direction and bend from: 0 = the edge of the opening only; more = the vessel's axis fitted further back (up to five diameters at 100 %), so a ragged or lumpy end does not steer the tube";
        ui.add_enabled_ui(along_wall, |ui| {
            egui::Grid::new("loft_context").num_columns(2).spacing([8.0, 2.0]).show(ui, |ui| {
                for k in 0..2 {
                    if k == 0 {
                        ui.label("Context").on_hover_text(context_tip);
                    } else {
                        ui.label("");
                    }
                    ui.add(egui::Slider::new(&mut self.context[k], 0.0..=100.0).suffix("%").text(format!("end {}", k + 1))).on_hover_text(context_tip);
                    ui.end_row();
                }
            });
            ui.checkbox(&mut self.centerline, "Follow the vessel centerlines")
                .on_hover_text("Trace each vessel's centerline from its opening and fit the end to it (more faithful round bends and near branches than the wall; slower on big meshes)");
        })
        .response
        .on_disabled_hover_text("For tubes leaving along the wall (Smooth, Along the axis)");
        ui.add_enabled(tube, egui::Slider::new(&mut self.blend, 0.0..=100.0).suffix("%").text("Blend"))
            .on_hover_text("Blend the tube into the vessels: ragged openings are tidied first, and the join is smoothed over a band of each vessel behind it (up to a diameter at 100 %). 0 leaves the vessels untouched");
        ui.add_enabled(tube, egui::Checkbox::new(&mut self.remesh, "Remesh the join")).on_hover_text("Even triangles over the tube and the blended bands, so no thin slivers show as lines along the seams");
    }

    /// Test-script settings: `loft:smooth` (or `minimal`, `straight`, `axis`), `end1:g2`,
    /// `tension2:1.5`, `dir:normal`, `round:50`, `fair:100`, `twist:30`, `context:50` (both
    /// ends) or `context1:30`, `blend:50`, `remesh:off`, `centerline:on`. Returns false for
    /// other keys.
    pub fn script(&mut self, key: &str, val: &str) -> bool {
        use mm_edit::{EndContinuity, EndDirection, LoftShape};
        let num = val.parse::<f64>().ok();
        let end = |k: &str| k.chars().last().and_then(|c| c.to_digit(10)).map(|d| (d as usize).clamp(1, 2) - 1);
        match key {
            "loft" => {
                self.shape = match val {
                    "minimal" => LoftShape::Minimal,
                    "straight" => LoftShape::Straight,
                    "axis" => LoftShape::Sweep,
                    _ => LoftShape::Smooth,
                };
            }
            k if k.starts_with("end") => {
                if let Some(i) = end(k) {
                    self.ends[i].continuity = match val {
                        "g0" => EndContinuity::G0,
                        "g2" => EndContinuity::G2,
                        _ => EndContinuity::G1,
                    };
                }
            }
            k if k.starts_with("tension") => {
                if let (Some(i), Some(x)) = (end(k), num) {
                    self.ends[i].tension = x;
                }
            }
            "dir" => {
                let d = if val == "normal" { EndDirection::Normal } else { EndDirection::Wall };
                self.ends.iter_mut().for_each(|e| e.direction = d);
            }
            "round" => self.round = num.unwrap_or(self.round),
            "fair" => self.fair = num.unwrap_or(self.fair),
            "twist" => self.twist = num.unwrap_or(self.twist),
            "context" => self.context = [num.unwrap_or(self.context[0]); 2],
            k if k.starts_with("context") => {
                if let (Some(i), Some(x)) = (end(k), num) {
                    self.context[i] = x;
                }
            }
            "centerline" => self.centerline = val == "on",
            "blend" => self.blend = num.unwrap_or(self.blend),
            "remesh" => self.remesh = val != "off",
            _ => return false,
        }
        true
    }
}

/// Test-script `refine:<0..100>` and `bulge:<percent>`.
fn script_refine_bulge(refine: &mut f64, bulge: &mut f64, key: &str, val: &str) {
    let Ok(x) = val.parse::<f64>() else { return };
    match key {
        "refine" => *refine = x,
        "bulge" => *bulge = x,
        _ => {}
    }
}

/// Loft openings `a` and `b` with the panel's settings, fitting the ends to their vessels'
/// centerlines (traced from each opening) when that is on.
pub(crate) fn run_loft(m: &mut Mesh, a: &[u32], b: &[u32], loft: &LoftUi, edge_length: f64, bulge: f64, group: Option<i32>) -> Result<mm_edit::BridgeResult, mm_edit::EditError> {
    let params = loft.params(edge_length, bulge, group);
    let along_wall = matches!(params.shape, mm_edit::LoftShape::Smooth | mm_edit::LoftShape::Sweep);
    let guides = if loft.centerline && along_wall { [a, b].map(|l| opening_centerline(m, l)) } else { [None, None] };
    mm_edit::loft_guided(m, a, b, &params, [guides[0].as_ref(), guides[1].as_ref()])
}

/// The centerline of the vessel behind opening `lp`: the first branch of a centerline grown
/// from the opening (up to the vessel's first fork), as a loft guide.
fn opening_centerline(m: &Mesh, lp: &[u32]) -> Option<mm_edit::AxisGuide> {
    let pts: Vec<DVec3> = lp.iter().map(|&v| m.vertex(v)).collect();
    let c = pts.iter().sum::<DVec3>() / pts.len().max(1) as f64;
    let radius = pts.iter().map(|p| p.distance(c)).sum::<f64>() / pts.len().max(1) as f64;
    let cl = crate::centerline::compute(m, lp, radius, &crate::centerline::CenterlineParams::default());
    let root = cl.branches.iter().find(|b| b.parent.is_none())?;
    let points: Vec<DVec3> = root.nodes.iter().map(|&k| cl.nodes[k].position).collect();
    let radii: Vec<f64> = root.nodes.iter().map(|&k| cl.nodes[k].radius).collect();
    (points.len() >= 3).then_some(mm_edit::AxisGuide { points, radii })
}

/// Edge length for a tube with `refine` (0..100; 50 keeps `base`, +25 halves it).
pub(crate) fn refined_edge_length(base: f64, refine: f64) -> f64 {
    base * 2f64.powf((50.0 - refine) / 25.0)
}

/// " — warning: …" when a new tube crosses the rest of the mesh, else empty.
pub(crate) fn crossing_warning(m: &Mesh, tube: &[u32]) -> String {
    match crossings(m, tube) {
        0 => String::new(),
        n => format!(" — warning: the tube runs into the surface ({n} crossing{}); try more tension or a twist", if n == 1 { "" } else { "s" }),
    }
}

/// Vertices whose triangles all belong to `tris` (the region's interior).
fn interior_vertices(m: &Mesh, tris: &[u32]) -> Vec<u32> {
    let set: std::collections::HashSet<u32> = tris.iter().copied().collect();
    let mut v: Vec<u32> = tris.iter().flat_map(|&t| m.triangle(t)).filter(|&v| m.vertex_triangles(v).iter().all(|t| set.contains(t))).collect();
    v.sort_unstable();
    v.dedup();
    v
}

/// "Project to Target" checkbox, available once a target surface is set.
fn project_to_target_ui(ui: &mut egui::Ui, on: &mut bool, info: &OpInfo) {
    let has = info.target.is_some();
    if !has {
        *on = false;
    }
    ui.add_enabled(has, egui::Checkbox::new(on, "Project to target"))
        .on_hover_text("Snap the new surface onto the target surface")
        .on_disabled_hover_text("Set a target surface first (Edit → Set as target)");
}

/// Apply Smooth / Scale / Bulge to a fill patch's interior vertices, relative to the plane
/// of its boundary loop.
fn shape_patch(m: &mut Mesh, lp: &mm_core::topology::EdgeLoop, r: &mm_repair::FillResult, p: &EraseFillUi) {
    if r.new_vertices.is_empty() {
        return;
    }
    let height = match p.kind {
        FillKind::SmoothMvc => p.smooth / 100.0 * p.scale,
        _ => p.scale,
    };
    if height == 1.0 && p.bulge == 0.0 {
        return;
    }
    let pts = lp.positions(m);
    let c = pts.iter().copied().sum::<DVec3>() / pts.len() as f64;
    // Newell normal of the loop, turned to point out of the surface (along the patch).
    let mut nrm = DVec3::ZERO;
    for i in 0..pts.len() {
        let (a, b) = (pts[i], pts[(i + 1) % pts.len()]);
        nrm += DVec3::new((a.y - b.y) * (a.z + b.z), (a.z - b.z) * (a.x + b.x), (a.x - b.x) * (a.y + b.y));
    }
    let patch_n: DVec3 = r.new_triangles.iter().map(|&t| m.tri_cross(t)).sum();
    let mut n = nrm.normalize_or_zero();
    if n.dot(patch_n) < 0.0 {
        n = -n;
    }
    let radius = pts.iter().map(|q| q.distance(c)).sum::<f64>() / pts.len() as f64;
    // Distance from the loop across the patch, for the bulge profile.
    let dist = patch_distance(m, lp, r);
    let dmax = dist.values().copied().fold(0.0, f64::max).max(1e-12);
    for &v in &r.new_vertices {
        let q = m.vertex(v);
        let d = (q - c).dot(n);
        let flat = q - n * d;
        let h = dist.get(&v).copied().unwrap_or(dmax) / dmax;
        let bump = (h * std::f64::consts::FRAC_PI_2).sin();
        m.set_vertex(v, flat + n * (d * height + p.bulge / 100.0 * radius * bump));
    }
}

/// Edge-path distance from the loop to each patch vertex (Dijkstra over the patch only).
fn patch_distance(m: &Mesh, lp: &mm_core::topology::EdgeLoop, r: &mm_repair::FillResult) -> std::collections::HashMap<u32, f64> {
    use std::collections::{BinaryHeap, HashMap, HashSet};
    #[derive(PartialEq)]
    struct Item(f64, u32);
    impl Eq for Item {}
    impl PartialOrd for Item {
        fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(o))
        }
    }
    impl Ord for Item {
        fn cmp(&self, o: &Self) -> std::cmp::Ordering {
            o.0.total_cmp(&self.0)
        }
    }
    let inside: HashSet<u32> = r.new_vertices.iter().copied().chain(lp.vertices.iter().copied()).collect();
    let mut dist: HashMap<u32, f64> = HashMap::new();
    let mut heap = BinaryHeap::new();
    for &v in &lp.vertices {
        dist.insert(v, 0.0);
        heap.push(Item(0.0, v));
    }
    while let Some(Item(d, v)) = heap.pop() {
        if d > dist.get(&v).copied().unwrap_or(f64::INFINITY) {
            continue;
        }
        for u in m.vertex_neighbors(v) {
            if !inside.contains(&u) {
                continue;
            }
            let nd = d + m.vertex(v).distance(m.vertex(u));
            if nd < dist.get(&u).copied().unwrap_or(f64::INFINITY) {
                dist.insert(u, nd);
                heap.push(Item(nd, u));
            }
        }
    }
    dist
}
pub type EraseFillTool = OpTool<EraseFillSpec>;

// ---------------------------------------------------------------------------
// Smooth (Deform > Smooth, B with a selection)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub enum SmoothType {
    /// Volume-preserving Taubin iterations: removes noise / staircase artefacts.
    ShapePreserving,
    /// Uniform-weight relaxation: evens out the triangles as it smooths.
    UniformTriangles,
    /// Thin-plate (bilaplacian) fairing: smoothest surface continuing the surroundings.
    MaxSmoothness,
    /// Shape Preserving with face-group border vertices held in place.
    PreserveGroupBorders,
    /// Membrane fairing of each face group separately (group borders held).
    PerGroupMembranes,
}

impl SmoothType {
    const ALL: [SmoothType; 5] = [Self::ShapePreserving, Self::UniformTriangles, Self::MaxSmoothness, Self::PreserveGroupBorders, Self::PerGroupMembranes];
    fn label(self) -> &'static str {
        match self {
            Self::ShapePreserving => "Shape Preserving",
            Self::UniformTriangles => "Uniform Triangles",
            Self::MaxSmoothness => "Max Smoothness",
            Self::PreserveGroupBorders => "Preserve Group Borders",
            Self::PerGroupMembranes => "Per-Group Membranes",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct SmoothUi {
    pub kind: SmoothType,
    /// 0..1: strength (blend toward the fully smoothed result).
    pub amount: f64,
    /// 1..10: how much smoothing (iterations for the iterative types).
    pub scale: f64,
    /// Rings at the selection border over which the result blends into the surroundings.
    pub constraint_rings: usize,
    pub cotan: bool,
    pub preserve_boundary: bool,
}

pub struct SmoothSpec;
impl OpSpec for SmoothSpec {
    type Params = SmoothUi;
    const NAME: &'static str = "Smooth";
    fn defaults(_info: &OpInfo) -> Self::Params {
        // Meshmixer's defaults.
        SmoothUi { kind: SmoothType::ShapePreserving, amount: 1.0, scale: 4.0, constraint_rings: 3, cotan: true, preserve_boundary: true }
    }
    fn ui(ui: &mut egui::Ui, p: &mut Self::Params, info: &OpInfo) {
        egui::ComboBox::from_label("Smoothing type").selected_text(p.kind.label()).show_ui(ui, |ui| {
            for k in SmoothType::ALL {
                ui.selectable_value(&mut p.kind, k, k.label());
            }
        });
        ui.add(egui::Slider::new(&mut p.amount, 0.0..=1.0).text("Smoothing"));
        ui.add(egui::Slider::new(&mut p.scale, 1.0..=10.0).text("Smoothing scale"));
        if info.selection.is_some() {
            ui.add(egui::Slider::new(&mut p.constraint_rings, 0..=10).text("Constraint rings")).on_hover_text("Blend into the surroundings over this many rings at the selection's edge");
        }
        ui.checkbox(&mut p.cotan, "Cotangent weights (follow the surface geometry)");
        ui.checkbox(&mut p.preserve_boundary, "Preserve open boundaries");
        if info.selection.is_none() && p.kind == SmoothType::MaxSmoothness {
            ui.colored_label(egui::Color32::from_rgb(240, 170, 90), "Max Smoothness fairs a selected region; without one, Shape Preserving is used.");
        }
    }
    fn run(mesh: &Mesh, sel: Option<&[u32]>, p: &Self::Params, progress: ProgressFn) -> anyhow::Result<OpOutput> {
        let mut m = mesh.clone();
        let weights = if p.cotan { mm_repair::LaplacianWeights::Cotangent } else { mm_repair::LaplacianWeights::Uniform };
        // Vertices free to move, with their distance (in rings) from the selection border.
        let (free, rings): (Vec<u32>, std::collections::HashMap<u32, usize>) = match sel {
            Some(tris) => {
                let s = mm_core::TriSelection::from_tris(&m, tris.iter().copied());
                let border: Vec<u32> = s.boundary_vertices(&m);
                let inside: std::collections::HashSet<u32> = s.vertices(&m).into_iter().collect();
                // Breadth-first ring distance from the border, inside the selection.
                let mut ring: std::collections::HashMap<u32, usize> = border.iter().map(|&v| (v, 0)).collect();
                let mut front = border;
                let mut k = 0;
                while !front.is_empty() {
                    k += 1;
                    let mut next = Vec::new();
                    for v in front {
                        for u in m.vertex_neighbors(v) {
                            if inside.contains(&u) && !ring.contains_key(&u) {
                                ring.insert(u, k);
                                next.push(u);
                            }
                        }
                    }
                    front = next;
                }
                let free = inside.iter().copied().filter(|v| ring.get(v).copied().unwrap_or(1) > 0).collect();
                (free, ring)
            }
            None => (m.vertex_ids().collect(), Default::default()),
        };
        let free: Vec<u32> = match p.kind {
            SmoothType::PreserveGroupBorders | SmoothType::PerGroupMembranes => free.into_iter().filter(|&v| {
                let ts = m.vertex_triangles(v);
                ts.iter().all(|&t| m.tri_group(t) == m.tri_group(ts[0]))
            }).collect(),
            _ => free,
        };
        let free: Vec<u32> = if p.preserve_boundary { free.into_iter().filter(|&v| !m.is_boundary_vertex(v)).collect() } else { free };
        if free.is_empty() {
            anyhow::bail!("nothing to smooth (no free vertices)");
        }
        let before: Vec<DVec3> = free.iter().map(|&v| m.vertex(v)).collect();
        let kind = if sel.is_none() && p.kind == SmoothType::MaxSmoothness { SmoothType::ShapePreserving } else { p.kind };
        let iterations = (p.scale * 10.0).round().max(1.0) as usize;
        match kind {
            SmoothType::ShapePreserving | SmoothType::PreserveGroupBorders | SmoothType::UniformTriangles => {
                let uniform = kind == SmoothType::UniformTriangles;
                let sp = mm_repair::SmoothParams {
                    iterations,
                    strength: 0.5,
                    weights: if uniform { mm_repair::LaplacianWeights::Uniform } else { weights },
                    preserve_boundary: p.preserve_boundary,
                    preserve_volume: !uniform,
                    ..Default::default()
                };
                mm_repair::smooth_vertices_with_progress(&mut m, Some(&free), &sp, Some(&|f| progress(f)))?;
            }
            SmoothType::MaxSmoothness => {
                mm_repair::fair_region(&mut m, &free, 2, weights)?;
            }
            SmoothType::PerGroupMembranes => {
                // Each group's interior is faired on its own (its borders are not free).
                let mut by_group: std::collections::HashMap<i32, Vec<u32>> = std::collections::HashMap::new();
                for &v in &free {
                    by_group.entry(m.tri_group(m.vertex_triangles(v)[0])).or_default().push(v);
                }
                for verts in by_group.values() {
                    mm_repair::fair_region(&mut m, verts, 1, weights)?;
                }
            }
        }
        // Strength, and a soft blend over the constraint rings at the selection's edge.
        let amount = p.amount.clamp(0.0, 1.0);
        for (k, &v) in free.iter().enumerate() {
            let edge_blend = match rings.get(&v) {
                Some(&r) if p.constraint_rings > 0 && r < p.constraint_rings => r as f64 / p.constraint_rings as f64,
                _ => 1.0,
            };
            let w = amount * edge_blend;
            let done = m.vertex(v);
            m.set_vertex(v, before[k].lerp(done, w));
        }
        let vol0 = mm_core::measure::volume(mesh);
        let vol1 = mm_core::measure::volume(&m);
        let msg = if mesh.is_closed() && vol0.abs() > 0.0 { format!("volume change {:+.2}%", (vol1 / vol0 - 1.0) * 100.0) } else { "smoothed".into() };
        Ok(OpOutput { mesh: m, selection: sel.map(|s| s.to_vec()), message: msg, ..Default::default() })
    }
}
pub type SmoothTool = OpTool<SmoothSpec>;

// ---------------------------------------------------------------------------
// Attract / Align To Target
// ---------------------------------------------------------------------------

const NO_TARGET: &str = "No target surface set: set one with Edit → Set as target or the object list.";

#[derive(Clone, Copy, PartialEq, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct AttractToTargetUi {
    pub offset: f64,
    /// 0..100: relaxation between projections.
    pub smooth: f64,
    /// Only attract vertices whose normal is within this angle of the target's (degrees).
    pub normal_threshold: f64,
    pub preserve_mesh_boundaries: bool,
    pub preserve_group_boundaries: bool,
    /// Remesh the selection first (Density as in Remesh).
    pub refine: bool,
    pub density: f64,
}

pub struct AttractToTargetSpec;
impl OpSpec for AttractToTargetSpec {
    type Params = AttractToTargetUi;
    const NAME: &'static str = "Attract To Target";
    fn defaults(_: &OpInfo) -> Self::Params {
        AttractToTargetUi {
            offset: 0.0,
            smooth: 50.0,
            normal_threshold: 90.0,
            preserve_mesh_boundaries: true,
            preserve_group_boundaries: false,
            refine: false,
            density: 0.0,
        }
    }
    fn ui(ui: &mut egui::Ui, p: &mut Self::Params, info: &OpInfo) {
        if info.target.is_none() {
            ui.colored_label(egui::Color32::from_rgb(240, 170, 90), NO_TARGET);
        }
        let span = (info.bbox_diag * 0.05).max(info.avg_edge * 5.0);
        ui.add(egui::Slider::new(&mut p.offset, -span..=span).text("offset"));
        ui.add(egui::Slider::new(&mut p.smooth, 0.0..=100.0).text("smooth"));
        ui.add(egui::Slider::new(&mut p.normal_threshold, 0.0..=180.0).suffix("°").text("normal threshold"))
            .on_hover_text("Only attract where the surface faces the same way as the target");
        ui.checkbox(&mut p.preserve_mesh_boundaries, "Preserve mesh boundaries");
        ui.checkbox(&mut p.preserve_group_boundaries, "Preserve group boundaries");
        ui.checkbox(&mut p.refine, "Enable refinement").on_hover_text("Remesh the selection first so it can follow the target's detail");
        ui.add_enabled(p.refine, egui::Slider::new(&mut p.density, -100.0..=100.0).text("density"));
    }
    fn run(mesh: &Mesh, sel: Option<&[u32]>, p: &Self::Params, progress: ProgressFn) -> anyhow::Result<OpOutput> {
        Self::run_with(mesh, sel, p, None, progress)
    }
    fn run_with(mesh: &Mesh, sel: Option<&[u32]>, p: &Self::Params, surface: Option<&TargetSurface>, _progress: ProgressFn) -> anyhow::Result<OpOutput> {
        let t = surface.ok_or_else(|| anyhow::anyhow!(NO_TARGET))?;
        let tris = sel.ok_or_else(|| anyhow::anyhow!("select the faces to attract first"))?;
        let mut m = mesh.clone();
        let mut region = tris.to_vec();
        if p.refine {
            let edge = mm_remesh::edge_length_stats_in(&m, tris).mean * 2f64.powf(-p.density / 50.0);
            let rp = mm_remesh::RemeshParams { target_edge_length: edge, iterations: 6, region: Some(region.clone()), ..Default::default() };
            let r = mm_remesh::remesh(&mut m, &rp, None)?;
            region = r.region_triangles.unwrap_or(region);
        }
        let ap = mm_edit::AttractParams {
            offset: p.offset,
            smooth: p.smooth / 100.0,
            normal_threshold_deg: p.normal_threshold,
            preserve_mesh_boundary: p.preserve_mesh_boundaries,
            preserve_group_boundaries: p.preserve_group_boundaries,
            ..Default::default()
        };
        let moved = mm_edit::attract_to_target(&mut m, &region, &t.mesh, &t.bvh, &ap);
        Ok(OpOutput { message: format!("attracted {} vertices", moved.len()), mesh: m, selection: Some(region), ..Default::default() })
    }
}
pub type AttractToTargetTool = OpTool<AttractToTargetSpec>;

#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub enum AlignSource {
    SelectedFaces,
    AllFaces,
}

#[derive(Clone, Copy, PartialEq, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct AlignToTargetUi {
    pub source: AlignSource,
    pub rotate: bool,
    pub max_iterations: usize,
    pub tolerance: f64,
}

pub struct AlignToTargetSpec;
impl OpSpec for AlignToTargetSpec {
    type Params = AlignToTargetUi;
    const NAME: &'static str = "Align To Target";
    fn defaults(info: &OpInfo) -> Self::Params {
        AlignToTargetUi {
            source: if info.selection.is_some() { AlignSource::SelectedFaces } else { AlignSource::AllFaces },
            rotate: true,
            max_iterations: 100,
            tolerance: info.avg_edge * 1e-4,
        }
    }
    fn ui(ui: &mut egui::Ui, p: &mut Self::Params, info: &OpInfo) {
        if info.target.is_none() {
            ui.colored_label(egui::Color32::from_rgb(240, 170, 90), NO_TARGET);
        }
        ui.horizontal(|ui| {
            ui.label("Source");
            ui.add_enabled_ui(info.selection.is_some(), |ui| {
                ui.selectable_value(&mut p.source, AlignSource::SelectedFaces, "Selected faces");
            });
            ui.selectable_value(&mut p.source, AlignSource::AllFaces, "All faces");
        });
        ui.horizontal(|ui| {
            ui.selectable_value(&mut p.rotate, true, "Translate and Rotate");
            ui.selectable_value(&mut p.rotate, false, "Translate Only");
        });
        ui.add(egui::Slider::new(&mut p.max_iterations, 1..=500).text("max iterations"));
        ui.small("Moves the whole object so the source faces fit the target surface (iterative closest point).");
    }
    fn run(mesh: &Mesh, sel: Option<&[u32]>, p: &Self::Params, progress: ProgressFn) -> anyhow::Result<OpOutput> {
        Self::run_with(mesh, sel, p, None, progress)
    }
    fn run_with(mesh: &Mesh, sel: Option<&[u32]>, p: &Self::Params, surface: Option<&TargetSurface>, _progress: ProgressFn) -> anyhow::Result<OpOutput> {
        let t = surface.ok_or_else(|| anyhow::anyhow!(NO_TARGET))?;
        let mut verts: Vec<u32> = match (p.source, sel) {
            (AlignSource::SelectedFaces, Some(tris)) => tris.iter().flat_map(|&t| mesh.triangle(t)).collect(),
            _ => mesh.vertex_ids().collect(),
        };
        verts.sort_unstable();
        verts.dedup();
        // Enough samples for a stable fit without slowing down on dense meshes.
        let stride = (verts.len() / 20_000).max(1);
        let pts: Vec<DVec3> = verts.iter().step_by(stride).map(|&v| mesh.vertex(v)).collect();
        let ip = mm_edit::IcpParams { rotate: p.rotate, max_iterations: p.max_iterations, tolerance: p.tolerance };
        let r = mm_edit::icp_align(&pts, &t.mesh, &t.bvh, &ip);
        let mut m = mesh.clone();
        m.transform(|x| r.apply(x));
        Ok(OpOutput {
            message: format!("aligned in {} iterations, RMS distance {:.4}", r.iterations, r.rms),
            mesh: m,
            selection: sel.map(|s| s.to_vec()),
            ..Default::default()
        })
    }
}
pub type AlignToTargetTool = OpTool<AlignToTargetSpec>;

// ---------------------------------------------------------------------------
// Fit Primitive
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct FitPrimitiveUi {
    pub kind: mm_edit::PrimitiveType,
    pub density: usize,
    /// One primitive for the whole selection instead of one per connected piece.
    pub single: bool,
    pub create_new_objects: bool,
}

/// Connected pieces of a triangle set.
fn selection_pieces(m: &Mesh, tris: &[u32]) -> Vec<Vec<u32>> {
    let set: std::collections::HashSet<u32> = tris.iter().copied().collect();
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for &t in tris {
        if !seen.insert(t) {
            continue;
        }
        let mut piece = vec![t];
        let mut stack = vec![t];
        while let Some(x) = stack.pop() {
            for n in m.tri_neighbors(x) {
                if n != mm_core::INVALID && set.contains(&n) && seen.insert(n) {
                    piece.push(n);
                    stack.push(n);
                }
            }
        }
        out.push(piece);
    }
    out
}

pub struct FitPrimitiveSpec;
impl OpSpec for FitPrimitiveSpec {
    type Params = FitPrimitiveUi;
    const NAME: &'static str = "Fit Primitive";
    fn defaults(_: &OpInfo) -> Self::Params {
        // Meshmixer's defaults.
        FitPrimitiveUi { kind: mm_edit::PrimitiveType::Square, density: 32, single: false, create_new_objects: false }
    }
    fn ui(ui: &mut egui::Ui, p: &mut Self::Params, info: &OpInfo) {
        egui::ComboBox::from_label("Primitive type").selected_text(p.kind.label()).show_ui(ui, |ui| {
            for k in mm_edit::PrimitiveType::ALL {
                ui.selectable_value(&mut p.kind, k, k.label());
            }
        });
        ui.add(egui::Slider::new(&mut p.density, 4..=128).text("mesh density"));
        ui.checkbox(&mut p.single, "Single primitive").on_hover_text("One primitive for the whole selection instead of one per piece");
        ui.checkbox(&mut p.create_new_objects, "Create new objects").on_hover_text("Each primitive becomes its own object (otherwise it is added to this one)");
        if info.selection.is_none() {
            ui.colored_label(egui::Color32::from_rgb(240, 170, 90), "Select the faces to fit first.");
        }
    }
    fn run(mesh: &Mesh, sel: Option<&[u32]>, p: &Self::Params, _progress: ProgressFn) -> anyhow::Result<OpOutput> {
        let tris = sel.ok_or_else(|| anyhow::anyhow!("select the faces to fit first"))?;
        let pieces = if p.single { vec![tris.to_vec()] } else { selection_pieces(mesh, tris) };
        let mut fits = Vec::new();
        for piece in &pieces {
            fits.push(mm_edit::fit_primitive(mesh, piece, p.kind, p.density)?);
        }
        let message = fits.iter().map(|f| f.description.clone()).collect::<Vec<_>>().join("; ");
        let mut out = mesh.clone();
        if p.create_new_objects {
            let mut preview = mesh.clone();
            for f in &fits {
                preview.append_mesh(&f.mesh, preview.max_group_id() + 1);
            }
            let name = p.kind.label().to_lowercase();
            return Ok(OpOutput {
                mesh: out,
                selection: Some(tris.to_vec()),
                message,
                new_objects: fits.into_iter().map(|f| (name.clone(), f.mesh)).collect(),
                preview: Some(preview),
                ..Default::default()
            });
        }
        for f in &fits {
            let g = out.max_group_id() + 1;
            out.append_mesh(&f.mesh, g);
        }
        Ok(OpOutput { mesh: out, selection: Some(tris.to_vec()), message, ..Default::default() })
    }
}
pub type FitPrimitiveTool = OpTool<FitPrimitiveSpec>;

#[cfg(test)]
mod tests {
    use super::*;
    use mm_core::primitives;

    fn messy() -> Mesh {
        let mut m = Mesh::new();
        let path: Vec<DVec3> = (0..=30).map(|i| DVec3::new((i as f64 * 0.2).sin(), 0.0, i as f64 * 0.5)).collect();
        m.append_mesh(&primitives::tube_along(&path, &vec![1.2; path.len()], 24), 0);
        let mut s = primitives::icosphere(1.5, 2);
        s.transform(|p| p + DVec3::new(0.0, 0.0, 8.0));
        m.append_mesh(&s, 0);
        let hole: Vec<u32> = m.triangle_ids().filter(|&t| (m.tri_centroid(t) - DVec3::new(1.2, 0.0, 4.0)).length() < 0.5).collect();
        m.remove_triangles(hole);
        m
    }

    /// Input state with one key press.
    fn key_input(key: egui::Key) -> egui::InputState {
        let raw = egui::RawInput {
            events: vec![egui::Event::Key { key, physical_key: None, pressed: true, repeat: false, modifiers: Default::default() }],
            ..Default::default()
        };
        let mut state = egui::InputState::default();
        state = state.begin_pass(raw, false, 1.0, egui::InputOptions::default());
        state
    }

    /// Total turning angle along the selection border (lower = smoother).
    fn border_turning(m: &Mesh, sel: &[u32]) -> f64 {
        let s = mm_core::TriSelection::from_tris(m, sel.iter().copied());
        let edges: Vec<u32> = s.boundary_edges(m).into_iter().filter(|&e| !m.is_boundary_edge(e)).collect();
        let mut nb: std::collections::HashMap<u32, Vec<u32>> = std::collections::HashMap::new();
        for &e in &edges {
            let [a, b] = m.edge_vertices(e);
            nb.entry(a).or_default().push(b);
            nb.entry(b).or_default().push(a);
        }
        nb.iter()
            .filter(|(_, n)| n.len() == 2)
            .map(|(&v, n)| {
                let p = m.vertex(v);
                let d1 = (p - m.vertex(n[0])).normalize_or_zero();
                let d2 = (m.vertex(n[1]) - p).normalize_or_zero();
                d1.dot(d2).clamp(-1.0, 1.0).acos()
            })
            .sum()
    }

    #[test]
    fn erase_fill_lofts_two_hole_rings() {
        // Two tubes facing each other across a gap; the rings of triangles around the facing
        // holes are selected, as double-clicking each open edge does.
        let mut m = primitives::cylinder(0.5, 2.0, 24, 8, false);
        let mut t2 = primitives::cylinder(0.6, 2.0, 30, 8, false);
        t2.transform(|p| p + DVec3::new(0.7, 0.0, 3.5));
        m.append_mesh(&t2, 1);
        let ring = |z: f64| -> Vec<u32> { m.triangle_ids().filter(|&t| m.triangle(t).iter().any(|&v| (m.vertex(v).z - z).abs() < 1e-9)).collect() };
        let rings: Vec<u32> = ring(2.0).into_iter().chain(ring(3.5)).collect();
        let info = |sel| OpInfo { mesh: &m, selection: Some(sel), avg_edge: m.average_edge_length(), bbox_diag: m.bounds().diagonal(), target: None };
        let p = EraseFillSpec::defaults(&info(&rings));
        assert_eq!(p.kind, FillKind::Loft, "two hole rings are lofted");
        assert_eq!(EraseFillSpec::defaults(&info(&ring(2.0))).kind, FillKind::SmoothMvc, "one hole is filled");
        for shape in [mm_edit::LoftShape::Minimal, mm_edit::LoftShape::Straight, mm_edit::LoftShape::Smooth] {
            let out = EraseFillSpec::run(&m, Some(&rings), &EraseFillUi { loft: LoftUi { shape, ..p.loft }, ..p }, &|_| true).unwrap();
            assert!(out.mesh.check_validity().is_ok());
            assert_eq!(mm_core::topology::connected_components(&out.mesh).len(), 1, "{shape:?}: one tube");
            assert_eq!(mm_core::topology::boundary_loops(&out.mesh).len(), 2, "{shape:?}: only the far ends stay open");
            assert_eq!(out.labels.len(), 2);
            assert!(!out.message.contains("warning"), "{shape:?}: {}", out.message);
        }
        let ends = [mm_edit::LoftEnd { continuity: mm_edit::EndContinuity::G2, ..Default::default() }; 2];
        let out = EraseFillSpec::run(&m, Some(&rings), &EraseFillUi { loft: LoftUi { ends, round: 100.0, fair: 100.0, ..p.loft }, ..p }, &|_| true).unwrap();
        assert!(out.mesh.check_validity().is_ok());
        assert_eq!(mm_core::topology::connected_components(&out.mesh).len(), 1, "G2, round and fair");
        let err = EraseFillSpec::run(&m, Some(&ring(2.0)), &p, &|_| true).err().unwrap().to_string();
        assert!(err.contains("two openings"), "{err}");
    }

    #[test]
    fn accepting_a_failed_op_does_not_retry_it() {
        // Erasing a whole closed sphere leaves no hole to fill, so Erase & Fill fails. Switching
        // tools then accepts it: that must close it unchanged, not start the run again.
        let m = primitives::icosphere(1.0, 2);
        let mut scene = crate::scene::Scene::default();
        let id = scene.add("ball", m.clone());
        scene.get_mut(id).unwrap().selection = mm_core::TriSelection::all(&m);
        let (mut history, mut camera, mut status) = (crate::scene::History::default(), crate::camera::Camera::default(), String::new());
        let ctx = egui::Context::default();
        let vp = Viewport { rect: egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0)) };
        let mut cx = ToolCx { scene: &mut scene, history: &mut history, camera: &mut camera, status: &mut status, egui: &ctx, vp, brush_scroll: crate::navigation::BrushScroll::Shift };
        let mut tool = EraseFillTool::new(&mut cx);
        tool.start(&ctx);
        let t0 = Instant::now();
        while tool.job.is_some() && t0.elapsed() < Duration::from_secs(10) {
            std::thread::sleep(Duration::from_millis(5));
            tool.poll();
        }
        assert!(tool.error.is_some() && tool.failed.is_some(), "the run fails");
        assert!(tool.accept(&mut cx), "accepting closes the tool");
        assert!(tool.busy().is_none(), "without starting another run");
        assert!(cx.status.contains("nothing applied"));
        assert_eq!(cx.scene.get(id).unwrap().mesh.triangle_count(), m.triangle_count());
    }

    #[test]
    fn erase_fill_leaves_no_fragments() {
        // A spike pulled out of a sphere at 45° to the view (looking down −z). The brush only
        // takes faces turned toward the eye, so the spike's far side stays unselected; erased
        // around, it would float in space as a fragment.
        let mut m = mm_core::primitives::icosphere(1.0, 4);
        let dir = DVec3::new(1.0, 0.0, 1.0).normalize();
        let tip = m.vertex_ids().max_by(|&a, &b| m.vertex(a).dot(dir).total_cmp(&m.vertex(b).dot(dir))).unwrap();
        let base = m.vertex(tip);
        m.set_vertex(tip, base * 2.5);
        let sel: Vec<u32> = m.triangle_ids().filter(|&t| m.tri_normal(t).z > 0.0 && m.triangle(t).iter().any(|&v| m.vertex(v).distance(base) < 0.5 || v == tip)).collect();
        let far_side = m.vertex_triangles(tip).iter().filter(|t| !sel.contains(t)).count();
        assert!(far_side > 0, "part of the spike is left unselected");
        let p = EraseFillSpec::defaults(&OpInfo { mesh: &m, selection: Some(&sel), avg_edge: m.average_edge_length(), bbox_diag: m.bounds().diagonal(), target: None });
        let out = EraseFillSpec::run(&m, Some(&sel), &p, &|_| true).unwrap();
        assert_eq!(mm_core::topology::connected_components(&out.mesh).len(), 1, "no floating fragments");
        assert!(out.mesh.is_closed());
        assert!(out.mesh.vertex_ids().all(|v| out.mesh.vertex(v).length() < 1.2), "the spike is gone");

        // A lone triangle left unselected inside the selection used to fail the fill ("the loop
        // bounds an isolated triangle"); it is a cut-off piece too.
        let ball = mm_core::primitives::icosphere(1.0, 4);
        let mut cap: Vec<u32> = ball.triangle_ids().filter(|&t| ball.tri_centroid(t).z > 0.5).collect();
        let lone = *cap.iter().max_by(|&&a, &&b| ball.tri_centroid(a).z.total_cmp(&ball.tri_centroid(b).z)).unwrap();
        cap.retain(|&t| t != lone);
        let out = EraseFillSpec::run(&ball, Some(&cap), &p, &|_| true).unwrap();
        assert_eq!(mm_core::topology::connected_components(&out.mesh).len(), 1);
        assert!(out.mesh.is_closed());

        // A band erased around a tube cuts it in two; both halves reach outside the band and stay.
        let tube = mm_core::primitives::cylinder(1.0, 6.0, 32, 24, true);
        let band: Vec<u32> = tube.triangle_ids().filter(|&t| (tube.tri_centroid(t).z - 2.0).abs() < 0.4).collect();
        let out = EraseFillSpec::run(&tube, Some(&band), &p, &|_| true).unwrap();
        assert_eq!(mm_core::topology::connected_components(&out.mesh).len(), 2, "both halves kept");
        assert!(out.mesh.is_closed());
    }

    #[test]
    fn erase_fill_parameters() {
        let m = mm_core::primitives::icosphere(1.0, 4);
        let cap: Vec<u32> = m.triangle_ids().filter(|&t| m.tri_centroid(t).z > 0.6).collect();
        let info = OpInfo { mesh: &m, selection: Some(&cap), avg_edge: m.average_edge_length(), bbox_diag: m.bounds().diagonal(), target: None };
        let base = EraseFillSpec::defaults(&info);
        let run = |p: EraseFillUi| EraseFillSpec::run(&m, Some(&cap), &p, &|_| true).unwrap();
        // Height of the patch's highest point above the hole plane (the cap border sits at z ≈ 0.6).
        let top = |out: &OpOutput| out.selection.as_ref().unwrap().iter().flat_map(|&t| out.mesh.tri_positions(t)).map(|p| p.z).fold(f64::MIN, f64::max);
        let smooth = run(base);
        assert!(smooth.mesh.is_closed());
        let h_smooth = top(&smooth);
        assert!(h_smooth > 0.75, "the smooth fill bulges like the sphere ({h_smooth})");
        let flat = run(EraseFillUi { smooth: 0.0, ..base });
        assert!(top(&flat) < 0.66, "smooth 0 gives a flat patch ({})", top(&flat));
        let bulged = run(EraseFillUi { bulge: 60.0, ..base });
        assert!(top(&bulged) > h_smooth + 0.05);
        let minimal = run(EraseFillUi { kind: FillKind::FlatMinimal, ..base });
        assert!(minimal.mesh.vertex_count() < m.vertex_count(), "minimal fill adds no vertices");
        let dense = run(EraseFillUi { refine: 90.0, ..base });
        assert!(dense.selection.unwrap().len() > smooth.selection.as_ref().unwrap().len());
        let grouped = run(EraseFillUi { create_new_groups: true, ..base });
        let patch = grouped.selection.unwrap();
        let g = grouped.mesh.tri_group(patch[0]);
        assert!(g != 0 && patch.iter().all(|&t| grouped.mesh.tri_group(t) == g));
    }

    #[test]
    fn join_and_weld_boundaries() {
        use mm_core::primitives;
        // Two open tube segments with a gap between them.
        let mut m = primitives::cylinder(1.0, 1.0, 16, 2, false);
        let mut b = primitives::cylinder(1.0, 1.0, 16, 2, false);
        b.transform(|p| p + DVec3::new(0.0, 0.0, 1.3));
        m.append_mesh(&b, 0);
        assert_eq!(mm_core::topology::boundary_loops(&m).len(), 4);
        let near: Vec<u32> = m.triangle_ids().filter(|&t| (m.tri_centroid(t).z - 1.15).abs() < 0.4).collect();
        let out = run_spec::<JoinSpec>(&m, Some(&near)).unwrap();
        assert_eq!(mm_core::topology::boundary_loops(&out.mesh).len(), 2, "the gap is closed");
        assert_eq!(mm_core::topology::connected_components(&out.mesh).len(), 1);
        // Weld: two grids whose seam edges are 0.01 apart (edge length 0.5).
        let mut g = primitives::plane_grid(2.0, 4.0, 4, 8);
        g.transform(|p| p - DVec3::new(1.0, 0.0, 0.0));
        let mut h = primitives::plane_grid(2.0, 4.0, 4, 8);
        h.transform(|p| p + DVec3::new(1.01, 0.0, 0.0));
        g.append_mesh(&h, 0);
        let seam: Vec<u32> = g.triangle_ids().filter(|&t| g.tri_centroid(t).x.abs() < 0.5).collect();
        let out = run_spec::<WeldBoundariesSpec>(&g, Some(&seam)).unwrap();
        let open_at_seam = out.mesh.boundary_edge_ids().filter(|&e| out.mesh.edge_midpoint(e).x.abs() < 0.1).count();
        assert_eq!(open_at_seam, 0, "{}", out.message);
    }

    #[test]
    fn bridge_two_selected_regions() {
        use mm_core::primitives;
        // Two spheres side by side; select the caps facing each other.
        let mut m = primitives::icosphere(1.0, 3);
        let mut b = primitives::icosphere(1.0, 3);
        b.transform(|p| p + DVec3::new(3.0, 0.0, 0.0));
        m.append_mesh(&b, 0);
        let caps: Vec<u32> = m.triangle_ids().filter(|&t| (m.tri_centroid(t) - DVec3::new(1.5, 0.0, 0.0)).length() < 0.9).collect();
        let info = OpInfo { mesh: &m, selection: Some(&caps), avg_edge: m.average_edge_length(), bbox_diag: m.bounds().diagonal(), target: None };
        let bridge = RegionBridgeSpec::defaults(&info);
        assert_eq!(bridge.loft.shape, mm_edit::LoftShape::Straight, "Cmd+B bridges straight");
        let mut g2 = RegionBridgeUi::tube_handle();
        RegionBridgeSpec::script(&mut g2, "end1:g2");
        RegionBridgeSpec::script(&mut g2, "fair:100");
        for p in [bridge, RegionBridgeUi::tube_handle(), g2] {
            let out = RegionBridgeSpec::run(&m, Some(&caps), &p, &|_| true).unwrap();
            out.mesh.check_validity().unwrap();
            assert!(out.mesh.is_closed(), "the two spheres are joined into one closed surface");
            assert_eq!(mm_core::topology::connected_components(&out.mesh).len(), 1);
            assert_eq!(out.labels.len(), 2, "End 1 and End 2 are marked");
            assert!(!out.message.contains("warning"), "{}", out.message);
        }
        // One region is not enough.
        let one: Vec<u32> = caps.iter().copied().filter(|&t| m.tri_centroid(t).x < 1.5).collect();
        assert!(run_spec::<RegionBridgeSpec>(&m, Some(&one)).is_err());
    }

    #[test]
    fn tube_handle_stands_up_from_a_surface() {
        use mm_core::primitives;
        // Two patches on top of a sphere: Tube Handle removes them and arches a tube over the
        // surface between the openings, leaving one closed surface with a handle.
        let m = primitives::icosphere(2.0, 4);
        let centres = [DVec3::new(-0.9, 0.0, 1.8).normalize() * 2.0, DVec3::new(0.9, 0.0, 1.8).normalize() * 2.0];
        let patches: Vec<u32> = m.triangle_ids().filter(|&t| centres.iter().any(|c| m.tri_centroid(t).distance(*c) < 0.4)).collect();
        let out = RegionBridgeSpec::run(&m, Some(&patches), &RegionBridgeUi::tube_handle(), &|_| true).unwrap();
        out.mesh.check_validity().unwrap();
        assert!(out.mesh.is_closed() && !out.message.contains("warning"), "{}", out.message);
        let tube = out.selection.unwrap();
        let tube_verts: Vec<DVec3> = tube.iter().flat_map(|&t| out.mesh.tri_positions(t)).collect();
        let top = tube_verts.iter().map(|p| p.z).fold(f64::MIN, f64::max);
        let inner = tube_verts.iter().map(|p| p.length()).fold(f64::MAX, f64::min);
        assert!(top > 2.15, "the handle arches above the sphere ({top:.2})");
        // The selection also holds the sphere's blended bands around the feet, smoothed in
        // by about half a percent.
        assert!(inner > 1.98, "and stays outside it ({inner:.3})");
        let mut along_wall = RegionBridgeUi::tube_handle();
        RegionBridgeSpec::script(&mut along_wall, "dir:wall");
        assert_eq!(along_wall.loft.ends.map(|e| e.direction), [mm_edit::EndDirection::Wall; 2]);
    }

    #[test]
    fn tube_handle_between_branches_is_not_twisted() {
        use mm_core::primitives;
        // The two branches of a Y (z up), the right one also turned a little about z, so the
        // patches on their fronts (−y) face nearly, not exactly, the same way. The handle
        // must join top to top and outer side to outer side, not turn half over.
        let branch = |tilt: f64, turn: f64, x: f64| {
            let (dir, rot) = (DVec3::new(tilt.to_radians().sin(), 0.0, tilt.to_radians().cos()), mm_core::DQuat::from_rotation_z(turn.to_radians()));
            let path: Vec<DVec3> = (0..=40).map(|k| DVec3::new(x, 0.0, 0.0) + rot * (dir * (k as f64 * 0.5 - 10.0))).collect();
            let mut b = primitives::tube_along(&path, &[2.5; 41], 48);
            let c = DVec3::new(x, 0.0, 0.0);
            b.transform(|p| c + rot * (p - c));
            b
        };
        let mut m = branch(-20.0, 0.0, -6.0);
        m.append_mesh(&branch(20.0, 12.0, 6.0), 0);
        let front = |x: f64| m.vertex_ids().map(|v| m.vertex(v)).filter(|p| (p.x - x).abs() < 2.0 && p.z.abs() < 0.3).min_by(|a, b| a.y.total_cmp(&b.y)).unwrap();
        let centres = [front(-6.0), front(6.0)];
        let patches: Vec<u32> = m.triangle_ids().filter(|&t| centres.iter().any(|c| m.tri_centroid(t).distance(*c) < 2.0)).collect();
        let out = RegionBridgeSpec::run(&m, Some(&patches), &RegionBridgeUi::tube_handle(), &|_| true).unwrap();
        out.mesh.check_validity().unwrap();
        assert!(!out.message.contains("warning"), "{}", out.message);
        assert_eq!(mm_core::topology::connected_components(&out.mesh).len(), 1);
    }

    #[test]
    fn vessel_bridge_lofts_two_picked_boundaries() {
        use mm_core::primitives;
        // Two open tubes across a gap: pick their facing ends and join them.
        let mut m = primitives::cylinder(0.5, 2.0, 24, 8, false);
        let mut t2 = primitives::cylinder(0.6, 2.0, 30, 8, false);
        t2.transform(|p| p + DVec3::new(0.7, 0.0, 3.5));
        m.append_mesh(&t2, 1);
        let mut scene = crate::scene::Scene::default();
        let id = scene.add("tubes", m.clone());
        let (mut history, mut camera, mut status) = (crate::scene::History::default(), crate::camera::Camera::default(), String::new());
        let ctx = egui::Context::default();
        let vp = Viewport { rect: egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0)) };
        let mut cx = ToolCx { scene: &mut scene, history: &mut history, camera: &mut camera, status: &mut status, egui: &ctx, vp, brush_scroll: crate::navigation::BrushScroll::Shift };
        let mut tool = BridgeTool::new(&mut cx);
        let ends = crate::tools::vessel::find_vessel_ends(&m);
        let facing = |z: f64| 1 + ends.iter().position(|e| (e.center.z - z).abs() < 1e-6).unwrap();
        for cmd in [format!("pick:{}", facing(2.0)), format!("pick:{}", facing(3.5)), "end2:g2".into(), "join".into()] {
            tool.script(&cmd, &mut cx);
        }
        assert!(cx.status.starts_with("Bridged"), "{}", cx.status);
        let out = &cx.scene.get(id).unwrap().mesh;
        out.check_validity().unwrap();
        assert_eq!(mm_core::topology::connected_components(out).len(), 1, "one tube");
        assert_eq!(mm_core::topology::boundary_loops(out).len(), 2, "only the far ends stay open");
        assert!(out.vertex_count() > m.vertex_count(), "a smooth tube with rings");
        assert!(cx.history.can_undo());
    }

    #[test]
    fn remesh_parity_options() {
        use mm_core::primitives;
        let m = primitives::icosphere(1.0, 4);
        let info = |sel: Option<&'static [u32]>| -> RemeshUi {
            let i = OpInfo { mesh: &m, selection: sel, avg_edge: m.average_edge_length(), bbox_diag: m.bounds().diagonal(), target: None };
            RemeshSpec::defaults(&i)
        };
        let base = info(None);
        // Density 0 keeps roughly the current resolution.
        let out = RemeshSpec::run(&m, None, &base, &|_| true).unwrap();
        let ratio = out.mesh.triangle_count() as f64 / m.triangle_count() as f64;
        assert!((0.6..1.6).contains(&ratio), "density 0 keeps the size ({ratio})");
        // Linear subdivision only splits: every vertex stays on the input surface (radius 1 or
        // on a flat input triangle, so within the sphere).
        let lin = RemeshSpec::run(&m, None, &RemeshUi { mode: RemeshMode::LinearSubdivision, density: 60.0, ..base }, &|_| true).unwrap();
        assert!(lin.mesh.triangle_count() > 2 * m.triangle_count());
        assert!(lin.mesh.vertex_ids().all(|v| lin.mesh.vertex(v).length() <= 1.0 + 1e-9));
        // Adaptive threshold: on an elongated ellipsoid, the curved tips get finer than the
        // flatter middle, more so with a higher threshold.
        let mut e = primitives::icosphere(1.0, 4);
        e.transform(|p| DVec3::new(p.x * 3.0, p.y, p.z));
        let ei = OpInfo { mesh: &e, selection: None, avg_edge: e.average_edge_length(), bbox_diag: e.bounds().diagonal(), target: None };
        let eb = RemeshSpec::defaults(&ei);
        let tip_over_middle = |t: f64| {
            let o = RemeshSpec::run(&e, None, &RemeshUi { mode: RemeshMode::AdaptiveDensity, threshold: t, ..eb }, &|_| true).unwrap();
            let tips: Vec<u32> = o.mesh.triangle_ids().filter(|&t| o.mesh.tri_centroid(t).x.abs() > 2.6).collect();
            let mid: Vec<u32> = o.mesh.triangle_ids().filter(|&t| o.mesh.tri_centroid(t).x.abs() < 1.0).collect();
            mm_remesh::edge_length_stats_in(&o.mesh, &tips).mean / mm_remesh::edge_length_stats_in(&o.mesh, &mid).mean
        };
        let (flat, strong) = (tip_over_middle(0.0), tip_over_middle(100.0));
        assert!(strong < flat * 0.8, "tips refine more with a high threshold ({flat:.2} -> {strong:.2})");
        // Transition: edges just outside a refined selection are in between.
        let cap: Vec<u32> = m.triangle_ids().filter(|&t| m.tri_centroid(t).z > 0.7).collect();
        let cap: &'static [u32] = Box::leak(cap.into_boxed_slice());
        let sel_base = info(Some(cap));
        let p = RemeshUi { density: 100.0, transition: sel_base.base_edge * 5.0, ..sel_base };
        let out = RemeshSpec::run(&m, Some(cap), &p, &|_| true).unwrap();
        out.mesh.check_validity().unwrap();
        let band: Vec<u32> = out.mesh.triangle_ids().filter(|&t| (0.55..0.65).contains(&out.mesh.tri_centroid(t).z)).collect();
        let mean = mm_remesh::edge_length_stats_in(&out.mesh, &band).mean;
        assert!(mean < sel_base.base_edge * 0.95 && mean > p.target() * 1.05, "band edges {mean} between {} and {}", p.target(), sel_base.base_edge);
    }

    #[test]
    fn reduce_types_and_smooth_types() {
        use mm_core::primitives;
        // Uniform reduction evens out edge lengths more than shape-preserving reduction.
        let mut e = primitives::icosphere(1.0, 4);
        e.transform(|p| DVec3::new(p.x * 3.0, p.y, p.z));
        let ei = OpInfo { mesh: &e, selection: None, avg_edge: e.average_edge_length(), bbox_diag: e.bounds().diagonal(), target: None };
        let rb = ReduceSpec::defaults(&ei);
        let cv = |uniform: bool| {
            let o = ReduceSpec::run(&e, None, &ReduceUi { uniform, ..rb }, &|_| true).unwrap();
            let st = mm_remesh::edge_length_stats(&o.mesh);
            st.std_dev / st.mean
        };
        assert!(cv(true) < cv(false), "uniform: {:.3} vs shape preserving {:.3}", cv(true), cv(false));

        // Smooth types on a noisy sphere with two face groups.
        let mut m = primitives::icosphere(1.0, 4);
        let ids: Vec<u32> = m.vertex_ids().collect();
        for v in ids {
            let p = m.vertex(v);
            let n = ((v * 7919) % 13) as f64 / 13.0 - 0.5;
            m.set_vertex(v, p * (1.0 + 0.04 * n));
        }
        let tris: Vec<u32> = m.triangle_ids().collect();
        for &t in &tris {
            if m.tri_centroid(t).x > 0.0 {
                m.set_tri_group(t, 1);
            }
        }
        let rough = |mm: &Mesh| mm.vertex_ids().map(|v| (mm.vertex(v).length() - 1.0).abs()).sum::<f64>();
        let info = OpInfo { mesh: &m, selection: None, avg_edge: m.average_edge_length(), bbox_diag: m.bounds().diagonal(), target: None };
        let base = SmoothSpec::defaults(&info);
        let group_border: Vec<u32> = m.vertex_ids().filter(|&v| {
            let ts = m.vertex_triangles(v);
            ts.iter().any(|&t| m.tri_group(t) != m.tri_group(ts[0]))
        }).collect();
        for kind in [SmoothType::ShapePreserving, SmoothType::UniformTriangles, SmoothType::PreserveGroupBorders] {
            let out = SmoothSpec::run(&m, None, &SmoothUi { kind, ..base }, &|_| true).unwrap();
            out.mesh.check_validity().unwrap();
            assert!(rough(&out.mesh) < rough(&m), "{kind:?} smooths");
            if kind == SmoothType::PreserveGroupBorders {
                assert!(group_border.iter().all(|&v| out.mesh.vertex(v) == m.vertex(v)), "group borders held");
            }
        }
        // A selection: fairing types and the constraint-ring blend at its edge.
        let cap: Vec<u32> = m.triangle_ids().filter(|&t| m.tri_centroid(t).z > 0.3).collect();
        for kind in [SmoothType::MaxSmoothness, SmoothType::PerGroupMembranes] {
            let out = SmoothSpec::run(&m, Some(&cap), &SmoothUi { kind, ..base }, &|_| true).unwrap();
            out.mesh.check_validity().unwrap();
        }
        let out = SmoothSpec::run(&m, Some(&cap), &SmoothUi { constraint_rings: 4, ..base }, &|_| true).unwrap();
        let s = mm_core::TriSelection::from_tris(&m, cap.iter().copied());
        let border = s.boundary_vertices(&m);
        let near: Vec<u32> = border.iter().flat_map(|&v| m.vertex_neighbors(v)).collect();
        let moved_near: f64 = near.iter().map(|&v| out.mesh.vertex(v).distance(m.vertex(v))).sum::<f64>() / near.len() as f64;
        let deep: Vec<u32> = s.vertices(&m).into_iter().filter(|&v| m.vertex(v).z > 0.8).collect();
        let moved_deep: f64 = deep.iter().map(|&v| out.mesh.vertex(v).distance(m.vertex(v))).sum::<f64>() / deep.len() as f64;
        assert!(moved_near < moved_deep, "the edge of the selection moves less ({moved_near:.4} vs {moved_deep:.4})");
    }

    #[test]
    fn smooth_boundary_straightens_a_jagged_selection() {
        let m = mm_core::primitives::icosphere(1.0, 4);
        // Upper half: its border follows triangle edges, so it zigzags around the equator.
        let sel: Vec<u32> = m.triangle_ids().filter(|&t| m.tri_centroid(t).z > 0.0).collect();
        let before = border_turning(&m, &sel);
        let out = run_spec::<SmoothBoundarySpec>(&m, Some(&sel)).unwrap();
        assert!(out.mesh.is_closed());
        let new_sel = out.selection.unwrap();
        assert!(!new_sel.is_empty());
        let after = border_turning(&out.mesh, &new_sel);
        assert!(after < 0.6 * before, "border turning {before:.2} -> {after:.2}");
        // Still on the sphere, and the selection became one new group.
        assert!(out.mesh.vertex_ids().all(|v| (out.mesh.vertex(v).length() - 1.0).abs() < 0.02));
        let g = out.mesh.tri_group(new_sel[0]);
        assert!(new_sel.iter().all(|&t| out.mesh.tri_group(t) == g));
        assert!(out.mesh.triangle_ids().filter(|t| !new_sel.contains(t)).all(|t| out.mesh.tri_group(t) != g));
    }

    #[test]
    fn remesh_and_reduce_hotkeys() {
        let m = mm_core::primitives::icosphere(1.0, 2);
        let info = OpInfo { mesh: &m, selection: None, avg_edge: m.average_edge_length(), bbox_diag: m.bounds().diagonal(), target: None };
        let mut p = RemeshSpec::defaults(&info);
        let d0 = p.density;
        assert!(RemeshSpec::keys(&mut p, &key_input(egui::Key::D)));
        assert_eq!(p.density, d0 + 10.0);
        assert!(RemeshSpec::keys(&mut p, &key_input(egui::Key::F)));
        assert_eq!(p.density, d0);
        assert!(RemeshSpec::keys(&mut p, &key_input(egui::Key::T)));
        assert_eq!(p.mode, RemeshMode::AdaptiveDensity);
        let g = p.preserve_group_borders;
        assert!(RemeshSpec::keys(&mut p, &key_input(egui::Key::G)));
        assert_eq!(p.preserve_group_borders, !g);
        assert!(!RemeshSpec::keys(&mut p, &key_input(egui::Key::K)));
        let mut r = ReduceSpec::defaults(&info);
        let k0 = r.percentage;
        assert!(ReduceSpec::keys(&mut r, &key_input(egui::Key::F)));
        assert_eq!(r.percentage, k0 - 10.0);
    }

    fn run_spec<S: OpSpec>(mesh: &Mesh, sel: Option<&[u32]>) -> anyhow::Result<OpOutput> {
        let info = OpInfo { mesh, selection: sel, avg_edge: mesh.average_edge_length(), bbox_diag: mesh.bounds().diagonal(), target: None };
        let p = S::defaults(&info);
        let r = S::run(mesh, sel, &p, &|_| true);
        if let Ok(out) = &r {
            out.mesh.check_validity().unwrap_or_else(|e| panic!("{}: invalid result: {e}", S::NAME));
        }
        r
    }

    #[test]
    fn every_op_runs_with_defaults() {
        let m = messy();
        let sel: Vec<u32> = m.triangle_ids().filter(|&t| m.tri_centroid(t).z < 3.0 && m.tri_centroid(t).x > 0.0).collect();
        assert!(!sel.is_empty());
        for s in [None, Some(sel.as_slice())] {
            run_spec::<RemeshSpec>(&m, s).unwrap();
            run_spec::<ReduceSpec>(&m, s).unwrap();
            run_spec::<SmoothSpec>(&m, s).unwrap();
        }
        run_spec::<ExtrudeSpec>(&m, Some(&sel)).unwrap();
        assert!(run_spec::<ExtrudeSpec>(&m, None).is_err(), "extrude needs a selection");
        run_spec::<EraseFillSpec>(&m, Some(&sel)).unwrap();
        run_spec::<MirrorSpec>(&m, None).unwrap();
        run_spec::<CloseCracksSpec>(&m, None).unwrap();
        run_spec::<FaceGroupsSpec>(&m, None).unwrap();
        run_spec::<SmoothBoundarySpec>(&m, Some(&sel)).unwrap();
        let solid = run_spec::<MakeSolidSpec>(&m, None).unwrap();
        assert!(solid.mesh.is_closed());
        let off = run_spec::<OffsetSolidSpec>(&solid.mesh, None).unwrap();
        assert!(off.mesh.is_closed());
        // Degenerate inputs: empty mesh and a single triangle must not panic.
        let empty = Mesh::new();
        let _ = run_spec::<RemeshSpec>(&empty, None);
        let _ = run_spec::<ReduceSpec>(&empty, None);
        let _ = run_spec::<SmoothSpec>(&empty, None);
        let _ = run_spec::<MakeSolidSpec>(&empty, None);
        let (tri, _) = Mesh::from_triangles(&[DVec3::ZERO, DVec3::X, DVec3::Y], &[[0, 1, 2]], None);
        let _ = run_spec::<RemeshSpec>(&tri, None);
        let _ = run_spec::<ReduceSpec>(&tri, None);
        let _ = run_spec::<SmoothSpec>(&tri, None);
        let _ = run_spec::<MakeSolidSpec>(&tri, None);
        let _ = run_spec::<MirrorSpec>(&tri, None);
    }

    #[test]
    fn make_solid_sharp_edges_groups_and_cull() {
        let q = glam::DQuat::from_euler(glam::EulerRot::XYZ, 0.31, 0.47, 0.13);
        let mut m = mm_core::primitives::cube(1.0, 4);
        for t in m.triangle_ids().collect::<Vec<_>>() {
            let n = m.tri_normal(t);
            let axis = (0..3).max_by(|&a, &b| n[a].abs().total_cmp(&n[b].abs())).unwrap();
            m.set_tri_group(t, (2 * axis + (n[axis] > 0.0) as usize) as i32);
        }
        m.transform(|p| q * p);
        let info = OpInfo { mesh: &m, selection: None, avg_edge: m.average_edge_length(), bbox_diag: m.bounds().diagonal(), target: None };
        let base = MakeSolidUi { accuracy: 0.0, ..MakeSolidSpec::defaults(&info) };
        let run = |p: MakeSolidUi| MakeSolidSpec::run(&m, None, &p, &|_| true).unwrap().mesh;
        // Sharp Edge Preserve keeps the cube's edges through the remesh, with its groups.
        let sharp = run(MakeSolidUi { solid_type: SolidType::SharpEdges, transfer_groups: true, ..base });
        let soft = run(MakeSolidUi { solid_type: SolidType::Accurate, ..base });
        assert!(sharp.is_closed());
        let edge_gap = |out: &Mesh| {
            let bvh = mm_core::Bvh::build(out);
            (0..=20)
                .map(|k| q * DVec3::new(-0.5 + k as f64 / 20.0 * 0.8 + 0.1, 0.5, 0.5))
                .map(|p| bvh.nearest(out, p, f64::INFINITY).unwrap().dist_sq.sqrt())
                .fold(0.0, f64::max)
        };
        let h = mm_solid::cell_size_for_resolution(&m.bounds(), 64);
        assert!(edge_gap(&sharp) < 0.1 * h && edge_gap(&sharp) < 0.3 * edge_gap(&soft), "{} vs {}", edge_gap(&sharp), edge_gap(&soft));
        let mut groups: Vec<i32> = sharp.triangle_ids().map(|t| sharp.tri_group(t)).collect();
        groups.sort();
        groups.dedup();
        assert_eq!(groups.len(), 6);
        // A lower cull threshold collapses fewer short edges.
        let raw = MakeSolidUi { remesh_result: false, ..base };
        let full = run(raw);
        let none = run(MakeSolidUi { cull_edges: 0.0, ..raw });
        assert!(none.triangle_count() > full.triangle_count(), "{} > {}", none.triangle_count(), full.triangle_count());
    }
}

#[cfg(test)]
mod target_tests {
    use super::*;
    use mm_core::primitives;

    fn surface(m: Mesh) -> TargetSurface {
        let bvh = Arc::new(mm_core::Bvh::build(&m));
        TargetSurface { id: 99, mesh: Arc::new(m), bvh }
    }

    fn info<'a>(m: &'a Mesh, sel: Option<&'a [u32]>, t: Option<&'a TargetSurface>) -> OpInfo<'a> {
        OpInfo { mesh: m, selection: sel, avg_edge: m.average_edge_length(), bbox_diag: m.bounds().diagonal(), target: t }
    }

    #[test]
    fn attract_and_project_need_and_use_the_target() {
        let t = surface(primitives::icosphere(1.0, 4));
        let m = primitives::icosphere(1.15, 3);
        let sel: Vec<u32> = m.triangle_ids().filter(|&tr| m.tri_centroid(tr).z > 0.7).collect();
        let p = AttractToTargetSpec::defaults(&info(&m, Some(&sel), Some(&t)));
        assert!(AttractToTargetSpec::run_with(&m, Some(&sel), &p, None, &|_| true).is_err(), "no target, no attract");
        for refine in [false, true] {
            let out = AttractToTargetSpec::run_with(&m, Some(&sel), &AttractToTargetUi { refine, density: 30.0, ..p }, Some(&t), &|_| true).unwrap();
            out.mesh.check_validity().unwrap();
            let region = out.selection.unwrap();
            let verts = interior_vertices(&out.mesh, &region);
            assert!(!verts.is_empty() && verts.iter().all(|&v| (out.mesh.vertex(v).length() - 1.0).abs() < 0.01));
        }
        // Erase & Fill with Project to Target lands the patch on the target.
        let ep = EraseFillUi { project_to_target: true, ..EraseFillSpec::defaults(&info(&m, Some(&sel), Some(&t))) };
        let out = EraseFillSpec::run_with(&m, Some(&sel), &ep, Some(&t), &|_| true).unwrap();
        let verts = interior_vertices(&out.mesh, out.selection.as_ref().unwrap());
        let on_target = |p: DVec3| t.bvh.nearest(&t.mesh, p, 1.0).is_some_and(|h| h.dist_sq < 1e-18);
        assert!(!verts.is_empty() && verts.iter().all(|&v| on_target(out.mesh.vertex(v))));
    }

    #[test]
    fn align_to_target_registers_a_moved_copy() {
        let mut base = primitives::cube(1.0, 4);
        base.transform(|p| DVec3::new(2.0 * p.x, p.y, 0.6 * p.z));
        let t = surface(base.clone());
        let mut m = base.clone();
        let q = mm_core::DQuat::from_rotation_z(0.2);
        m.transform(|p| q * p + DVec3::new(0.1, 0.05, -0.04));
        let p = AlignToTargetSpec::defaults(&info(&m, None, Some(&t)));
        let out = AlignToTargetSpec::run_with(&m, None, &p, Some(&t), &|_| true).unwrap();
        let err = base.vertex_ids().map(|v| out.mesh.vertex(v).distance(base.vertex(v))).fold(0.0, f64::max);
        assert!(err < 1e-3, "{err}: {}", out.message);
    }
}

#[cfg(test)]
mod fit_primitive_tests {
    use super::*;
    use mm_core::primitives;

    #[test]
    fn one_primitive_per_piece_or_new_objects() {
        let m = primitives::plane_grid(6.0, 2.0, 24, 8);
        let sel: Vec<u32> = m.triangle_ids().filter(|&t| m.tri_centroid(t).x.abs() > 1.5).collect();
        let p = FitPrimitiveUi { kind: mm_edit::PrimitiveType::Rectangle, density: 4, single: false, create_new_objects: false };
        let out = FitPrimitiveSpec::run(&m, Some(&sel), &p, &|_| true).unwrap();
        assert_eq!(mm_core::topology::connected_components(&out.mesh).len(), 3, "two rectangles added");
        let single = FitPrimitiveSpec::run(&m, Some(&sel), &FitPrimitiveUi { single: true, ..p }, &|_| true).unwrap();
        assert_eq!(mm_core::topology::connected_components(&single.mesh).len(), 2);
        let objects = FitPrimitiveSpec::run(&m, Some(&sel), &FitPrimitiveUi { create_new_objects: true, ..p }, &|_| true).unwrap();
        assert_eq!(objects.new_objects.len(), 2);
        assert_eq!(objects.mesh.triangle_count(), m.triangle_count());
    }
}

#[cfg(test)]
mod color_transfer_tests {
    use super::*;

    #[test]
    fn make_solid_keeps_vertex_colors() {
        let mut m = mm_core::primitives::icosphere(1.0, 3);
        for v in m.vertex_ids().collect::<Vec<_>>() {
            let z = m.vertex(v).z as f32;
            m.set_vertex_color(v, [(z + 1.0) / 2.0, 0.2, 0.2]);
        }
        let info = OpInfo { mesh: &m, selection: None, avg_edge: m.average_edge_length(), bbox_diag: m.bounds().diagonal(), target: None };
        let base = MakeSolidUi { accuracy: 0.0, ..MakeSolidSpec::defaults(&info) };
        for remesh in [false, true] {
            let out = MakeSolidSpec::run(&m, None, &MakeSolidUi { remesh_result: remesh, ..base }, &|_| true).unwrap().mesh;
            assert!(out.has_vertex_colors());
            for v in out.vertex_ids() {
                let want = (out.vertex(v).z as f32 + 1.0) / 2.0;
                assert!((out.vertex_color(v).unwrap()[0] - want).abs() < 0.05, "remesh {remesh}");
            }
        }
        let none = MakeSolidSpec::run(&m, None, &MakeSolidUi { color_transfer: ColorTransfer::NoColors, ..base }, &|_| true).unwrap().mesh;
        assert!(!none.has_vertex_colors());
    }
}

