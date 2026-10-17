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
}

/// Context shown to operation panels.
pub struct OpInfo<'a> {
    pub mesh: &'a Mesh,
    pub selection: Option<&'a [u32]>,
    pub avg_edge: f64,
    pub bbox_diag: f64,
}

pub trait OpSpec: Send + 'static {
    type Params: Clone + PartialEq + Send + 'static;
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
    /// In-tool hotkeys (Meshmixer: D/F density ±10, T adaptive, G groups). Return true if
    /// the parameters changed.
    fn keys(_p: &mut Self::Params, _i: &egui::InputState) -> bool {
        false
    }
}

/// Plain (unmodified) key press.
fn plain(i: &egui::InputState, k: egui::Key) -> bool {
    i.key_pressed(k) && !i.modifiers.command && !i.modifiers.alt && !i.modifiers.shift
}

pub struct OpTool<S: OpSpec> {
    object: ObjectId,
    source: Arc<Mesh>,
    selection: Option<Arc<Vec<u32>>>,
    pub params: S::Params,
    avg_edge: f64,
    bbox_diag: f64,
    /// Params of the running job / of the current result.
    job: Option<(S::Params, Job<OpOutput>)>,
    result: Option<(S::Params, OpOutput)>,
    error: Option<String>,
    changed_at: Option<Instant>,
    pending_accept: bool,
    invalid: bool,
}

impl<S: OpSpec> OpTool<S> {
    pub fn new(cx: &mut ToolCx) -> Self {
        let Some(o) = cx.scene.active() else {
            return Self::invalid();
        };
        let selection = if S::USES_SELECTION && !o.selection.is_empty() { Some(Arc::new(o.selection.to_vec())) } else { None };
        let source = o.mesh.clone();
        let avg_edge = source.average_edge_length();
        let bbox_diag = source.bounds().diagonal();
        let info = OpInfo { mesh: &source, selection: selection.as_deref().map(|v| v.as_slice()), avg_edge, bbox_diag };
        let params = S::defaults(&info);
        Self {
            object: o.id,
            source,
            selection,
            params,
            avg_edge,
            bbox_diag,
            job: None,
            result: None,
            error: None,
            changed_at: if S::LIVE { Some(Instant::now() - Duration::from_secs(1)) } else { None },
            pending_accept: false,
            invalid: false,
        }
    }

    fn invalid() -> Self {
        let info = OpInfo { mesh: &Mesh::new(), selection: None, avg_edge: 1.0, bbox_diag: 1.0 };
        let params = S::defaults(&info);
        Self {
            object: 0,
            source: Arc::new(Mesh::new()),
            selection: None,
            params,
            avg_edge: 1.0,
            bbox_diag: 1.0,
            job: None,
            result: None,
            error: Some("No active object.".into()),
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
        let p2 = p.clone();
        let job = Job::spawn(S::NAME, ctx.clone(), move |progress| S::run(&src, sel.as_deref().map(|v| v.as_slice()), &p2, progress));
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
            let info = OpInfo { mesh: &self.source, selection: self.selection.as_deref().map(|v| v.as_slice()), avg_edge: self.avg_edge, bbox_diag: self.bbox_diag };
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

    fn preview(&self) -> Option<(ObjectId, &Mesh)> {
        self.result.as_ref().map(|(_, out)| (self.object, out.preview.as_ref().unwrap_or(&out.mesh)))
    }

    fn accept(&mut self, cx: &mut ToolCx) -> bool {
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

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ExtrudeMode {
    /// Move the region and stitch side walls (Meshmixer Extrude, D).
    Extrude,
    /// Move the region with a soft transition, no new walls (Meshmixer Offset, Ctrl+D).
    Offset,
    /// Copy the region into a new, thickened solid object (Meshmixer Extract, Shift+D).
    Extract,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ExtrudeDir {
    Normal,
    /// The region's average normal.
    Constant,
    Axis(usize),
}

#[derive(Clone, Copy, PartialEq, Debug)]
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

#[derive(Clone, Copy, PartialEq, Debug)]
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

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RemeshMode {
    RelativeDensity,
    TargetEdgeLength,
    AdaptiveDensity,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RemeshBoundary {
    Fixed,
    Refined,
    Free,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct RemeshUi {
    pub mode: RemeshMode,
    /// 0..100; 50 = current mean edge length, +25 halves the edge length.
    pub density: f64,
    pub edge_length: f64,
    /// Tangential relaxation 0..1 (Meshmixer "Regularity").
    pub regularity: f64,
    pub iterations: usize,
    pub preserve_sharp: bool,
    pub sharp_threshold: f64,
    pub preserve_groups: bool,
    pub boundary: RemeshBoundary,
    pub base_edge: f64,
}

impl RemeshUi {
    fn target(&self) -> f64 {
        match self.mode {
            RemeshMode::TargetEdgeLength => self.edge_length,
            _ => self.base_edge * 2f64.powf((50.0 - self.density) / 25.0),
        }
    }
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
            p.density = (p.density - 10.0).max(0.0);
        }
        if (plain(i, egui::Key::D) || plain(i, egui::Key::F)) && p.mode == RemeshMode::TargetEdgeLength {
            p.mode = RemeshMode::RelativeDensity;
        }
        if plain(i, egui::Key::T) {
            p.mode = if p.mode == RemeshMode::AdaptiveDensity { RemeshMode::RelativeDensity } else { RemeshMode::AdaptiveDensity };
        }
        if plain(i, egui::Key::G) {
            p.preserve_groups = !p.preserve_groups;
        }
        *p != before
    }
    fn defaults(info: &OpInfo) -> Self::Params {
        let base = match info.selection {
            Some(sel) => mm_remesh::edge_length_stats_in(info.mesh, sel).mean,
            None => info.avg_edge,
        }
        .max(1e-9);
        RemeshUi {
            mode: RemeshMode::RelativeDensity,
            density: 50.0,
            edge_length: base,
            regularity: 0.5,
            iterations: 10,
            preserve_sharp: false,
            sharp_threshold: 30.0,
            preserve_groups: true,
            boundary: RemeshBoundary::Fixed,
            base_edge: base,
        }
    }
    fn ui(ui: &mut egui::Ui, p: &mut Self::Params, info: &OpInfo) {
        egui::ComboBox::from_label("Remesh mode")
            .selected_text(match p.mode {
                RemeshMode::RelativeDensity => "Relative Density",
                RemeshMode::TargetEdgeLength => "Target Edge Length",
                RemeshMode::AdaptiveDensity => "Adaptive Density",
            })
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut p.mode, RemeshMode::RelativeDensity, "Relative Density");
                ui.selectable_value(&mut p.mode, RemeshMode::TargetEdgeLength, "Target Edge Length");
                ui.selectable_value(&mut p.mode, RemeshMode::AdaptiveDensity, "Adaptive Density");
            });
        match p.mode {
            RemeshMode::TargetEdgeLength => {
                ui.add(egui::Slider::new(&mut p.edge_length, (p.base_edge * 0.1)..=(p.base_edge * 10.0)).logarithmic(true).text("edge length"));
            }
            _ => {
                ui.add(egui::Slider::new(&mut p.density, 0.0..=100.0).text("density")).on_hover_text("D / F: ±10");
                ui.weak(format!("target edge length ≈ {:.4}", p.target()));
            }
        }
        ui.add(egui::Slider::new(&mut p.regularity, 0.0..=1.0).text("regularity"));
        ui.add(egui::Slider::new(&mut p.iterations, 1..=40).text("iterations"));
        ui.checkbox(&mut p.preserve_sharp, "Preserve sharp edges");
        if p.preserve_sharp {
            ui.add(egui::Slider::new(&mut p.sharp_threshold, 5.0..=90.0).text("sharp threshold °"));
        }
        ui.checkbox(&mut p.preserve_groups, "Preserve group boundaries");
        ui.horizontal(|ui| {
            ui.label("Boundary");
            ui.selectable_value(&mut p.boundary, RemeshBoundary::Fixed, "Fixed");
            ui.selectable_value(&mut p.boundary, RemeshBoundary::Refined, "Refined");
            ui.selectable_value(&mut p.boundary, RemeshBoundary::Free, "Free");
        });
        let est = (info.mesh.triangle_count() as f64 * (p.base_edge / p.target()).powi(2)) as usize;
        if info.selection.is_none() {
            ui.weak(format!("≈ {} triangles", crate::ui_util::format_count(est)));
        }
    }
    fn run(mesh: &Mesh, sel: Option<&[u32]>, p: &Self::Params, progress: ProgressFn) -> anyhow::Result<OpOutput> {
        let mut m = mesh.clone();
        let params = mm_remesh::RemeshParams {
            target_edge_length: p.target(),
            iterations: p.iterations,
            smoothing: p.regularity,
            boundary: match p.boundary {
                RemeshBoundary::Fixed => mm_remesh::BoundaryMode::Fixed,
                RemeshBoundary::Refined => mm_remesh::BoundaryMode::ConstrainToBoundary,
                RemeshBoundary::Free => mm_remesh::BoundaryMode::Free,
            },
            preserve_sharp_edges: p.preserve_sharp.then_some(p.sharp_threshold),
            preserve_group_boundaries: p.preserve_groups,
            adaptive: p.mode == RemeshMode::AdaptiveDensity,
            project_to_original: true,
            region: sel.map(|s| s.to_vec()),
            smooth_group_boundaries: None,
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

#[derive(Clone, Copy, PartialEq, Debug)]
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
        }
    }
    fn ui(ui: &mut egui::Ui, p: &mut Self::Params, _info: &OpInfo) {
        ui.add(egui::Slider::new(&mut p.smoothness, 0.0..=100.0).text("Smoothness"));
        ui.add(egui::Slider::new(&mut p.preserve_shape, 0.0..=100.0).text("Preserve shape"));
        ui.add(egui::Slider::new(&mut p.iterations, 1..=100).text("Iterations"));
        ui.add(egui::Slider::new(&mut p.border_rings, 1..=8).text("Border rings"));
        ui.checkbox(&mut p.preserve_mesh_boundary, "Preserve mesh boundary");
        ui.checkbox(&mut p.preserve_group_borders, "Preserve group borders");
        ui.checkbox(&mut p.create_new_groups, "Create new groups");
    }
    fn run(mesh: &Mesh, sel: Option<&[u32]>, p: &Self::Params, progress: ProgressFn) -> anyhow::Result<OpOutput> {
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
        };
        let r = mm_remesh::remesh(&mut m, &params, Some(progress))?;
        if r.cancelled {
            anyhow::bail!("cancelled");
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

#[derive(Clone, Copy, PartialEq, Debug)]
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

#[derive(Clone, Copy, PartialEq, Debug)]
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

#[derive(Clone, Copy, PartialEq, Debug)]
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

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct RegionBridgeUi {
    /// Smooth curved tube leaving each region along its normal (Tube Handle); otherwise a
    /// straight bridge.
    pub tube: bool,
    /// Rows of triangles along the bridge; 0 = automatic.
    pub segments: usize,
}

pub struct RegionBridgeSpec;
impl OpSpec for RegionBridgeSpec {
    type Params = RegionBridgeUi;
    const NAME: &'static str = "Bridge";
    fn defaults(_: &OpInfo) -> Self::Params {
        RegionBridgeUi { tube: false, segments: 0 }
    }
    fn ui(ui: &mut egui::Ui, p: &mut Self::Params, info: &OpInfo) {
        ui.horizontal(|ui| {
            ui.selectable_value(&mut p.tube, false, "Bridge").on_hover_text("Straight connection");
            ui.selectable_value(&mut p.tube, true, "Tube handle").on_hover_text("Curved tube leaving each region along its surface normal");
        });
        ui.add(egui::Slider::new(&mut p.segments, 0..=60).text("Segments (0 = auto)"));
        ui.small("Select two separate regions: they are removed and the openings joined by a tube.");
        if info.selection.is_none() {
            ui.colored_label(egui::Color32::from_rgb(240, 170, 90), "Select two regions first.");
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
        let mut m = mesh.clone();
        let mut loops = Vec::new();
        for part in &parts[..2] {
            let border: std::collections::HashSet<u32> = mm_core::TriSelection::from_tris(&m, part.iter().copied()).boundary_vertices(&m).into_iter().collect();
            m.remove_triangles(part.iter().copied());
            let lp = mm_core::topology::boundary_loops(&m)
                .into_iter()
                .filter(|l| l.vertices.iter().any(|v| border.contains(v)))
                .max_by_key(|l| l.vertices.len())
                .ok_or_else(|| anyhow::anyhow!("a region did not leave an opening (does it cover a whole part?)"))?;
            loops.push(lp.vertices);
        }
        let r = mm_edit::join_loops(&mut m, &loops[0], &loops[1], &mm_edit::BridgeParams { segments: p.segments, smooth: p.tube })?;
        Ok(OpOutput { message: format!("bridged with {} triangles", r.triangles.len()), selection: Some(r.triangles), mesh: m, ..Default::default() })
    }
}
pub type RegionBridgeTool = OpTool<RegionBridgeSpec>;

// ---------------------------------------------------------------------------
// Generate Face Groups (Edit)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Debug)]
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

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ReduceGoal {
    Percentage,
    TriangleBudget,
    MaxDeviation,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ReduceUi {
    pub goal: ReduceGoal,
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
        *p != before
    }
    fn defaults(info: &OpInfo) -> Self::Params {
        let n = info.selection.map(|s| s.len()).unwrap_or(info.mesh.triangle_count());
        ReduceUi {
            goal: ReduceGoal::Percentage,
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

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SolidType {
    Blocky,
    Fast,
    Accurate,
}

#[derive(Clone, Copy, PartialEq, Debug)]
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
        }
    }
    fn ui(ui: &mut egui::Ui, p: &mut Self::Params, info: &OpInfo) {
        ui.horizontal(|ui| {
            ui.label("Solid type");
            ui.selectable_value(&mut p.solid_type, SolidType::Blocky, "Blocky");
            ui.selectable_value(&mut p.solid_type, SolidType::Fast, "Fast");
            ui.selectable_value(&mut p.solid_type, SolidType::Accurate, "Accurate");
        });
        let ext = info.mesh.bounds().extents();
        let m = ext.x.max(ext.y).max(ext.z);
        ui.add(egui::Slider::new(&mut p.accuracy, 0.0..=1.0).text("solid accuracy").show_value(false));
        ui.weak(format!("{} cells, voxel size {:.4}", p.cells(), m / p.cells() as f64));
        ui.add(egui::Slider::new(&mut p.density, 0.0..=1.0).text("mesh density"));
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
        };
        params.offset = p.offset;
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
            let rp = mm_remesh::RemeshParams { target_edge_length: target, iterations: 6, ..Default::default() };
            let remesh_progress = |f: f32| progress(0.7 + 0.3 * f);
            mm_remesh::remesh(&mut out, &rp, Some(&remesh_progress))?;
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

#[derive(Clone, Copy, PartialEq, Debug)]
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

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FillKind {
    /// Fewest triangles, no new vertices.
    FlatMinimal,
    /// Flat patch with triangles matching the surroundings.
    FlatRemeshed,
    /// Curved patch continuing the surrounding surface (Meshmixer "Smooth MVC").
    SmoothMvc,
}

impl FillKind {
    fn label(self) -> &'static str {
        match self {
            FillKind::FlatMinimal => "Flat Minimal",
            FillKind::FlatRemeshed => "Flat Remeshed",
            FillKind::SmoothMvc => "Smooth MVC",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
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
}

pub struct EraseFillSpec;
impl OpSpec for EraseFillSpec {
    type Params = EraseFillUi;
    const NAME: &'static str = "Erase & Fill";
    fn defaults(_: &OpInfo) -> Self::Params {
        // Meshmixer's defaults.
        EraseFillUi { kind: FillKind::SmoothMvc, refine: 50.0, smooth: 100.0, scale: 1.0, bulge: 0.0, create_new_groups: false }
    }
    fn ui(ui: &mut egui::Ui, p: &mut Self::Params, info: &OpInfo) {
        egui::ComboBox::from_label("Fill type").selected_text(p.kind.label()).show_ui(ui, |ui| {
            for k in [FillKind::FlatMinimal, FillKind::FlatRemeshed, FillKind::SmoothMvc] {
                ui.selectable_value(&mut p.kind, k, k.label());
            }
        });
        let patch = p.kind != FillKind::FlatMinimal;
        ui.add_enabled(patch, egui::Slider::new(&mut p.refine, 0.0..=100.0).text("Refine"));
        ui.add_enabled(p.kind == FillKind::SmoothMvc, egui::Slider::new(&mut p.smooth, 0.0..=100.0).text("Smooth"));
        ui.add_enabled(patch, egui::Slider::new(&mut p.scale, 0.0..=4.0).text("Scale"));
        ui.add_enabled(patch, egui::Slider::new(&mut p.bulge, -100.0..=100.0).text("Bulge"));
        ui.checkbox(&mut p.create_new_groups, "Create new groups");
        ui.small("Deletes the selection and fills the opening so it blends into the surrounding surface.");
        if info.selection.is_none() {
            ui.colored_label(egui::Color32::from_rgb(240, 170, 90), "Select a region first.");
        }
    }
    fn run(mesh: &Mesh, sel: Option<&[u32]>, p: &Self::Params, progress: ProgressFn) -> anyhow::Result<OpOutput> {
        let tris = sel.ok_or_else(|| anyhow::anyhow!("select a region first"))?;
        let mut m = mesh.clone();
        let selection = mm_core::TriSelection::from_tris(&m, tris.iter().copied());
        let border: std::collections::HashSet<u32> = selection.boundary_vertices(&m).into_iter().collect();
        let target = mm_remesh::edge_length_stats_in(&m, tris).mean * 2f64.powf((50.0 - p.refine) / 25.0);
        let new_group = m.max_group_id().max(0) + 1;
        m.remove_triangles(tris.iter().copied());
        let loops: Vec<_> = mm_core::topology::boundary_loops(&m).into_iter().filter(|l| l.vertices.iter().any(|v| border.contains(v))).collect();
        if loops.is_empty() {
            anyhow::bail!("the selection did not leave a hole to fill (did it cover a whole part?)");
        }
        let mode = match p.kind {
            FillKind::FlatMinimal => mm_repair::FillMode::Minimal,
            FillKind::FlatRemeshed => mm_repair::FillMode::Flat,
            FillKind::SmoothMvc => mm_repair::FillMode::Smooth,
        };
        let n = loops.len();
        let mut new_tris = Vec::new();
        for (k, lp) in loops.iter().enumerate() {
            if !progress(k as f32 / n as f32) {
                anyhow::bail!("cancelled");
            }
            let group = if p.create_new_groups { new_group } else { lp.edges.first().map(|&e| m.tri_group(m.edge_triangles(e)[0])).unwrap_or(0) };
            let fp = mm_repair::FillParams { mode, target_edge_length: Some(target), group: Some(group), ..Default::default() };
            let r = mm_repair::fill_hole(&mut m, lp, &fp)?;
            shape_patch(&mut m, lp, &r, p);
            new_tris.extend(r.new_triangles);
        }
        Ok(OpOutput { message: format!("replaced {} triangles with {}", tris.len(), new_tris.len()), mesh: m, selection: Some(new_tris), ..Default::default() })
    }
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

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SmoothType {
    /// Volume-preserving Taubin iterations: removes noise / staircase artefacts.
    ShapePreserving,
    /// Membrane (Laplacian) fairing of the region with fixed surroundings.
    Membrane,
    /// Thin-plate (bilaplacian) fairing: smoothest surface continuing the surroundings.
    MaxSmoothness,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct SmoothUi {
    pub kind: SmoothType,
    pub amount: f64,
    pub cotan: bool,
    pub preserve_boundary: bool,
}

pub struct SmoothSpec;
impl OpSpec for SmoothSpec {
    type Params = SmoothUi;
    const NAME: &'static str = "Smooth";
    fn defaults(info: &OpInfo) -> Self::Params {
        SmoothUi {
            kind: if info.selection.is_some() { SmoothType::MaxSmoothness } else { SmoothType::ShapePreserving },
            amount: 0.5,
            cotan: true,
            preserve_boundary: true,
        }
    }
    fn ui(ui: &mut egui::Ui, p: &mut Self::Params, info: &OpInfo) {
        egui::ComboBox::from_label("Smoothing type")
            .selected_text(match p.kind {
                SmoothType::ShapePreserving => "Shape Preserving",
                SmoothType::Membrane => "Membrane",
                SmoothType::MaxSmoothness => "Max Smoothness",
            })
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut p.kind, SmoothType::ShapePreserving, "Shape Preserving");
                ui.selectable_value(&mut p.kind, SmoothType::Membrane, "Membrane");
                ui.selectable_value(&mut p.kind, SmoothType::MaxSmoothness, "Max Smoothness");
            });
        ui.add(egui::Slider::new(&mut p.amount, 0.0..=1.0).text("smoothing"));
        ui.checkbox(&mut p.cotan, "Cotangent weights (follow the surface geometry)");
        ui.checkbox(&mut p.preserve_boundary, "Preserve open boundaries");
        if info.selection.is_none() && p.kind != SmoothType::ShapePreserving {
            ui.colored_label(egui::Color32::from_rgb(240, 170, 90), "Fairing needs a selected region; whole-mesh smoothing uses Shape Preserving.");
        }
    }
    fn run(mesh: &Mesh, sel: Option<&[u32]>, p: &Self::Params, progress: ProgressFn) -> anyhow::Result<OpOutput> {
        let mut m = mesh.clone();
        let weights = if p.cotan { mm_repair::LaplacianWeights::Cotangent } else { mm_repair::LaplacianWeights::Uniform };
        // Vertices free to move: interior of the selection (its border stays fixed).
        let free: Option<Vec<u32>> = sel.map(|tris| {
            let s = mm_core::TriSelection::from_tris(&m, tris.iter().copied());
            let border: std::collections::HashSet<u32> = s.boundary_vertices(&m).into_iter().collect();
            s.vertices(&m).into_iter().filter(|v| !border.contains(v)).collect()
        });
        let kind = if free.is_none() { SmoothType::ShapePreserving } else { p.kind };
        match kind {
            SmoothType::ShapePreserving => {
                let sp = mm_repair::SmoothParams {
                    iterations: (2.0 + p.amount * 40.0) as usize,
                    strength: 0.5,
                    weights,
                    preserve_boundary: p.preserve_boundary,
                    preserve_volume: true,
                    ..Default::default()
                };
                mm_repair::smooth_vertices_with_progress(&mut m, free.as_deref(), &sp, Some(&|f| progress(f)))?;
            }
            SmoothType::Membrane | SmoothType::MaxSmoothness => {
                let free = free.unwrap();
                if free.is_empty() {
                    anyhow::bail!("the selection has no interior vertices to move");
                }
                let before: Vec<DVec3> = free.iter().map(|&v| m.vertex(v)).collect();
                let order = if kind == SmoothType::Membrane { 1 } else { 2 };
                mm_repair::fair_region(&mut m, &free, order, weights)?;
                // Blend between original and fully faired positions.
                for (k, &v) in free.iter().enumerate() {
                    let faired = m.vertex(v);
                    m.set_vertex(v, before[k].lerp(faired, p.amount.clamp(0.0, 1.0).sqrt()));
                }
            }
        }
        let vol0 = mm_core::measure::volume(mesh);
        let vol1 = mm_core::measure::volume(&m);
        let msg = if mesh.is_closed() && vol0.abs() > 0.0 { format!("volume change {:+.2}%", (vol1 / vol0 - 1.0) * 100.0) } else { "smoothed".into() };
        Ok(OpOutput { mesh: m, selection: sel.map(|s| s.to_vec()), message: msg, ..Default::default() })
    }
}
pub type SmoothTool = OpTool<SmoothSpec>;

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
    fn erase_fill_parameters() {
        let m = mm_core::primitives::icosphere(1.0, 4);
        let cap: Vec<u32> = m.triangle_ids().filter(|&t| m.tri_centroid(t).z > 0.6).collect();
        let info = OpInfo { mesh: &m, selection: Some(&cap), avg_edge: m.average_edge_length(), bbox_diag: m.bounds().diagonal() };
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
        for tube in [false, true] {
            let info = OpInfo { mesh: &m, selection: Some(&caps), avg_edge: m.average_edge_length(), bbox_diag: m.bounds().diagonal() };
            let p = RegionBridgeUi { tube, ..RegionBridgeSpec::defaults(&info) };
            let out = RegionBridgeSpec::run(&m, Some(&caps), &p, &|_| true).unwrap();
            out.mesh.check_validity().unwrap();
            assert!(out.mesh.is_closed(), "the two spheres are joined into one closed surface");
            assert_eq!(mm_core::topology::connected_components(&out.mesh).len(), 1);
        }
        // One region is not enough.
        let one: Vec<u32> = caps.iter().copied().filter(|&t| m.tri_centroid(t).x < 1.5).collect();
        assert!(run_spec::<RegionBridgeSpec>(&m, Some(&one)).is_err());
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
        let info = OpInfo { mesh: &m, selection: None, avg_edge: m.average_edge_length(), bbox_diag: m.bounds().diagonal() };
        let mut p = RemeshSpec::defaults(&info);
        let d0 = p.density;
        assert!(RemeshSpec::keys(&mut p, &key_input(egui::Key::D)));
        assert_eq!(p.density, d0 + 10.0);
        assert!(RemeshSpec::keys(&mut p, &key_input(egui::Key::F)));
        assert_eq!(p.density, d0);
        assert!(RemeshSpec::keys(&mut p, &key_input(egui::Key::T)));
        assert_eq!(p.mode, RemeshMode::AdaptiveDensity);
        let g = p.preserve_groups;
        assert!(RemeshSpec::keys(&mut p, &key_input(egui::Key::G)));
        assert_eq!(p.preserve_groups, !g);
        assert!(!RemeshSpec::keys(&mut p, &key_input(egui::Key::K)));
        let mut r = ReduceSpec::defaults(&info);
        let k0 = r.percentage;
        assert!(ReduceSpec::keys(&mut r, &key_input(egui::Key::F)));
        assert_eq!(r.percentage, k0 - 10.0);
    }

    fn run_spec<S: OpSpec>(mesh: &Mesh, sel: Option<&[u32]>) -> anyhow::Result<OpOutput> {
        let info = OpInfo { mesh, selection: sel, avg_edge: mesh.average_edge_length(), bbox_diag: mesh.bounds().diagonal() };
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
}
