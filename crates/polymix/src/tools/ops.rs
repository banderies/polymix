//! Parameterised mesh operations with live background preview (Remesh, Reduce, Smooth,
//! Make Solid, …). Each operation is an [`OpSpec`]; [`OpTool`] supplies the shared
//! preview / accept / cancel behaviour.

use super::*;
use crate::jobs::{Job, ProgressFn};
use crate::ui_util;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Result of running an operation on a copy of the mesh.
pub struct OpOutput {
    pub mesh: Mesh,
    /// Triangles to select afterwards (e.g. the remeshed region).
    pub selection: Option<Vec<u32>>,
    pub message: String,
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
    fn defaults(info: &OpInfo) -> Self::Params;
    fn ui(ui: &mut egui::Ui, p: &mut Self::Params, info: &OpInfo);
    fn run(mesh: &Mesh, selection: Option<&[u32]>, p: &Self::Params, progress: ProgressFn) -> anyhow::Result<OpOutput>;
}

pub struct OpTool<S: OpSpec> {
    object: ObjectId,
    source: Arc<Mesh>,
    selection: Option<Arc<Vec<u32>>>,
    params: S::Params,
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
        *cx.status = format!("{}: {}", S::NAME, out.message);
        true
    }
}

impl<S: OpSpec> Tool for OpTool<S> {
    fn name(&self) -> &'static str {
        S::NAME
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
        self.result.as_ref().map(|(_, out)| (self.object, &out.mesh))
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
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ExtrudeParamsUi {
    pub mode: ExtrudeMode,
    pub distance: f64,
    pub along_normals: bool,
    pub axis: usize,
    pub falloff: f64,
    pub harden: bool,
}

pub struct ExtrudeSpec;
impl OpSpec for ExtrudeSpec {
    type Params = ExtrudeParamsUi;
    const NAME: &'static str = "Extrude / Offset";
    fn defaults(info: &OpInfo) -> Self::Params {
        ExtrudeParamsUi { mode: ExtrudeMode::Extrude, distance: (info.avg_edge * 3.0).max(1e-4), along_normals: true, axis: 1, falloff: 0.0, harden: true }
    }
    fn ui(ui: &mut egui::Ui, p: &mut Self::Params, info: &OpInfo) {
        ui.horizontal(|ui| {
            ui.selectable_value(&mut p.mode, ExtrudeMode::Extrude, "Extrude");
            ui.selectable_value(&mut p.mode, ExtrudeMode::Offset, "Offset");
        });
        let span = (info.bbox_diag * 0.25).max(info.avg_edge * 10.0);
        ui.add(egui::Slider::new(&mut p.distance, -span..=span).text("distance"));
        if p.mode == ExtrudeMode::Extrude {
            ui.horizontal(|ui| {
                ui.label("Direction");
                ui.selectable_value(&mut p.along_normals, true, "Normals");
                ui.selectable_value(&mut p.along_normals, false, "Axis");
                if !p.along_normals {
                    for (i, n) in ["X", "Y", "Z"].iter().enumerate() {
                        ui.selectable_value(&mut p.axis, i, *n);
                    }
                }
            });
            ui.checkbox(&mut p.harden, "Sharp side edges");
        } else {
            ui.add(egui::Slider::new(&mut p.falloff, 0.0..=span).text("soft transition"));
        }
        if info.selection.is_none() {
            ui.colored_label(egui::Color32::from_rgb(240, 170, 90), "Select a region first.");
        }
    }
    fn run(mesh: &Mesh, sel: Option<&[u32]>, p: &Self::Params, _progress: ProgressFn) -> anyhow::Result<OpOutput> {
        let tris = sel.ok_or_else(|| anyhow::anyhow!("select a region first"))?;
        let mut m = mesh.clone();
        match p.mode {
            ExtrudeMode::Extrude => {
                let mut dir = DVec3::ZERO;
                dir[p.axis] = 1.0;
                let params = mm_edit::ExtrudeParams {
                    distance: p.distance,
                    direction: if p.along_normals { mm_edit::ExtrudeDirection::Normal } else { mm_edit::ExtrudeDirection::Fixed(dir) },
                    harden_sides: p.harden,
                    ..Default::default()
                };
                let r = mm_edit::extrude_selection(&mut m, tris, &params)?;
                let n = r.side_tris.len();
                Ok(OpOutput { mesh: m, selection: Some(r.new_region_tris), message: format!("extruded with {n} side triangles") })
            }
            ExtrudeMode::Offset => {
                let n = mm_edit::offset_selection(&mut m, tris, p.distance, p.falloff)?;
                Ok(OpOutput { mesh: m, selection: Some(tris.to_vec()), message: format!("moved {n} vertices") })
            }
        }
    }
}

pub struct ExtrudeTool;
impl ExtrudeTool {
    #[allow(clippy::new_ret_no_self)]
    pub fn new(cx: &mut ToolCx, offset: bool) -> OpTool<ExtrudeSpec> {
        let mut t: OpTool<ExtrudeSpec> = OpTool::new(cx);
        if offset {
            t.params.mode = ExtrudeMode::Offset;
        }
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
        Ok(OpOutput { mesh: m, selection: None, message: format!("{} triangles added, {} welded", r.new_triangles.len(), r.welded_vertices) })
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
        })
    }
}
pub type RemeshTool = OpTool<RemeshSpec>;

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
        Ok(OpOutput { message: format!("{} triangles", out.triangle_count()), mesh: out, selection: None })
    }
}
pub type OffsetSolidTool = OpTool<OffsetSolidSpec>;

// ---------------------------------------------------------------------------
// Erase & Fill (F)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct EraseFillUi {
    pub mode: mm_repair::FillMode,
}

pub struct EraseFillSpec;
impl OpSpec for EraseFillSpec {
    type Params = EraseFillUi;
    const NAME: &'static str = "Erase & Fill";
    fn defaults(_: &OpInfo) -> Self::Params {
        EraseFillUi { mode: mm_repair::FillMode::Smooth }
    }
    fn ui(ui: &mut egui::Ui, p: &mut Self::Params, info: &OpInfo) {
        ui.horizontal(|ui| {
            ui.label("Fill type");
            ui.selectable_value(&mut p.mode, mm_repair::FillMode::Minimal, "Minimal");
            ui.selectable_value(&mut p.mode, mm_repair::FillMode::Flat, "Flat");
            ui.selectable_value(&mut p.mode, mm_repair::FillMode::Smooth, "Smooth");
        });
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
        let target = mm_remesh::edge_length_stats_in(&m, tris).mean;
        m.remove_triangles(tris.iter().copied());
        let loops: Vec<_> = mm_core::topology::boundary_loops(&m).into_iter().filter(|l| l.vertices.iter().any(|v| border.contains(v))).collect();
        if loops.is_empty() {
            anyhow::bail!("the selection did not leave a hole to fill (did it cover a whole part?)");
        }
        let n = loops.len();
        let mut new_tris = Vec::new();
        for (k, lp) in loops.iter().enumerate() {
            if !progress(k as f32 / n as f32) {
                anyhow::bail!("cancelled");
            }
            let group = lp.edges.first().map(|&e| m.tri_group(m.edge_triangles(e)[0])).unwrap_or(0);
            let fp = mm_repair::FillParams { mode: p.mode, target_edge_length: Some(target), group: Some(group), ..Default::default() };
            let r = mm_repair::fill_hole(&mut m, lp, &fp)?;
            new_tris.extend(r.new_triangles);
        }
        Ok(OpOutput { message: format!("replaced {} triangles with {}", tris.len(), new_tris.len()), mesh: m, selection: Some(new_tris) })
    }
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
        Ok(OpOutput { mesh: m, selection: sel.map(|s| s.to_vec()), message: msg })
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
