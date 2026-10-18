//! The Segmentation workspace: the volume selected in the Volume workspace, in three planes —
//! axial (top left), sagittal (bottom left) and coronal (bottom right) — beside a 3D view
//! (top right), which will show the segmentation. The segmentation tools come next.
//!
//! It depends only on `polymix-workspace` and the libraries (`mm-volume`, `polymix-imaging`),
//! never on another workspace: the shell hands it the Volume workspace's selection through
//! [`SegmentationWorkspace::show_volume`].

use eframe::egui_wgpu;
use egui::{Color32, Rect, Vec2};
use glam::DVec3;
use mm_volume::load::{load_volume, LoadedVolume};
use mm_volume::model::Volume;
use mm_volume::slice::Plane;
use polymix_imaging::{SliceOptions, SliceView, ViewState, VolumeView3d};
use polymix_workspace::jobs::Job;
use polymix_workspace::Workspace;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;

/// Largest volume loaded (voxels; 4 bytes each in memory).
const MAX_VOXELS: usize = 400_000_000;
const WARN: Color32 = Color32::from_rgb(240, 170, 90);
/// Behind and between the views.
const BACKDROP: Color32 = Color32::from_rgb(12, 12, 14);
/// The planes, in the order of [`SegmentationWorkspace::views`].
const PLANES: [Plane; 3] = [Plane::Axial, Plane::Sagittal, Plane::Coronal];

/// The volume shown: enough of the Volume workspace's entry to recognise it again.
struct Source {
    name: String,
    first_file: Option<PathBuf>,
    files: usize,
    /// Its name as the Volume workspace lists it (identifiers masked when hidden there).
    label: String,
    /// Volumes in a 4D set (only the first is shown).
    volumes: usize,
}

impl Source {
    fn new(v: &Volume, label: &str) -> Self {
        Self { name: v.name.clone(), first_file: v.files.first().cloned(), files: v.files.len(), label: label.to_string(), volumes: v.geometry.as_ref().map_or(1, |g| g.dims[3]) }
    }

    fn is(&self, v: &Volume) -> bool {
        self.name == v.name && self.files == v.files.len() && self.first_file.as_ref() == v.files.first()
    }
}

/// A volume being read in the background.
struct Loading {
    job: Job<LoadedVolume>,
    progress: Arc<AtomicU32>,
    cancel: Arc<AtomicBool>,
}

impl Loading {
    fn start(v: &Volume, ctx: &egui::Context) -> Self {
        let v = v.clone();
        let progress = Arc::new(AtomicU32::new(0));
        let cancel = Arc::new(AtomicBool::new(false));
        let (p, c) = (progress.clone(), cancel.clone());
        // The loader reports from several threads, so it gets its own progress and cancel
        // flags rather than the job's.
        let job = Job::spawn("Loading volume", ctx.clone(), move |_| {
            let report = move |f: f32| {
                p.store(f.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
                !c.load(Ordering::Relaxed)
            };
            load_volume(&v, MAX_VOXELS, &report)
        });
        Self { job, progress, cancel }
    }

    fn progress(&self) -> f32 {
        f32::from_bits(self.progress.load(Ordering::Relaxed))
    }
}

impl Drop for Loading {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

pub struct SegmentationWorkspace {
    source: Option<Source>,
    volume: Option<Arc<LoadedVolume>>,
    loading: Option<Loading>,
    error: Option<String>,
    /// Cursor and window, shared by the views.
    state: Option<ViewState>,
    /// Axial, sagittal, coronal.
    views: [SliceView; 3],
    view3d: VolumeView3d,
    crosshair: bool,
    smooth: bool,
    /// The voxel under the pointer, and in which view.
    hover: Option<(Plane, DVec3)>,
    /// Where the views were drawn last frame: axial, 3D, sagittal, coronal (for scripts).
    rects: [Rect; 4],
    script: VecDeque<String>,
}

impl SegmentationWorkspace {
    pub fn new(_ctx: &egui::Context, _storage: Option<&dyn eframe::Storage>) -> Self {
        Self {
            source: None,
            volume: None,
            loading: None,
            error: None,
            state: None,
            views: PLANES.map(SliceView::new),
            view3d: VolumeView3d::new(),
            crosshair: true,
            smooth: true,
            hover: None,
            rects: [Rect::NOTHING; 4],
            script: VecDeque::new(),
        }
    }

    /// Show `v`, the volume selected in the Volume workspace, listed there as `label`. A
    /// different volume replaces the one shown; the same one is not read again.
    pub fn show_volume(&mut self, v: &Volume, label: &str, ctx: &egui::Context) {
        if let Some(s) = &mut self.source {
            if s.is(v) {
                if s.label != label {
                    s.label = label.to_string();
                }
                return;
            }
        }
        // Free the old volume first: two large volumes may not fit in memory together.
        self.volume = None;
        self.state = None;
        self.hover = None;
        self.error = None;
        self.source = Some(Source::new(v, label));
        self.loading = Some(Loading::start(v, ctx));
    }

    fn finish_loading(&mut self, result: anyhow::Result<LoadedVolume>) {
        match result {
            Ok(vol) => {
                self.state = Some(ViewState::new(&vol));
                self.views = PLANES.map(SliceView::new);
                self.view3d.reset_camera();
                self.volume = Some(Arc::new(vol));
            }
            Err(e) => {
                let msg = format!("{e:#}");
                if msg != "cancelled" {
                    self.error = Some(msg);
                }
            }
        }
    }

    fn run_script_step(&mut self, step: String) {
        let (cmd, arg) = step.split_once(':').unwrap_or((step.as_str(), ""));
        match cmd {
            // Counts and positions only: never names.
            "seg.log" => {
                let rect = |r: &Rect| format!("{:.0},{:.0},{:.0},{:.0}", r.min.x, r.min.y, r.width(), r.height());
                let rects = ["axial", "3d", "sagittal", "coronal"].iter().zip(&self.rects).map(|(n, r)| format!("{n}={}", rect(r))).collect::<Vec<_>>().join(" ");
                match (&self.volume, &self.state) {
                    (Some(v), Some(s)) => {
                        let zoom = self.views.iter().map(|v| format!("{:.2}", v.nav.zoom)).collect::<Vec<_>>().join(",");
                        let pan = self.views.iter().map(|v| format!("{:.0}/{:.0}", v.nav.pan.x, v.nav.pan.y)).collect::<Vec<_>>().join(",");
                        println!(
                            "[segmentation] {arg} dims={}x{}x{} cursor={:.1},{:.1},{:.1} window={:.1},{:.1} zoom={zoom} pan={pan} {rects}",
                            v.dims[0], v.dims[1], v.dims[2], s.cursor.x, s.cursor.y, s.cursor.z, s.window.0, s.window.1
                        );
                    }
                    _ => println!("[segmentation] {arg} no volume ({}) {rects}", self.error.as_deref().unwrap_or(if self.loading.is_some() { "loading" } else { "none selected" })),
                }
            }
            _ => log::warn!("unknown Segmentation script step {step}"),
        }
    }

    // ------------------------------------------------------------------ UI parts

    fn toolbar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let Some(src) = &self.source else {
                ui.weak("No volume");
                return;
            };
            ui.label(egui::RichText::new(&src.label).strong());
            if let Some(v) = &self.volume {
                let g = &v.geometry;
                let mut info = format!("{} × {} × {} · {:.2} × {:.2} × {:.2} mm · {}", v.dims[0], v.dims[1], v.dims[2], g.spacing[0], g.spacing[1], g.spacing[2], g.orientation());
                if src.volumes > 1 {
                    info += &format!(" · first of {} volumes", src.volumes);
                }
                ui.weak(info);
                let oblique = g.obliquity_deg();
                if oblique > 1.0 {
                    ui.colored_label(WARN, format!("⚠ oblique {oblique:.0}°")).on_hover_text("The views follow the voxel grid: each shows the patient plane nearest to it");
                }
            }
            if let Some(l) = &self.loading {
                ui.separator();
                ui.add(egui::ProgressBar::new(l.progress()).desired_width(160.0).text("Loading…"));
                if ui.button("Cancel").clicked() {
                    l.cancel.store(true, Ordering::Relaxed);
                }
            }
            let (Some(v), Some(state)) = (&self.volume, &mut self.state) else { return };
            ui.separator();
            let (c, w) = state.window;
            ui.weak(format!("Window {w:.0} / level {c:.0}"));
            if ui.small_button("Auto").on_hover_text("1st–99th percentile of the volume").clicked() {
                state.window = v.auto_window;
            }
            if let Some(fw) = v.file_window {
                if ui.small_button("File").on_hover_text("The window stored in the file").clicked() {
                    state.window = fw;
                }
            }
            ui.separator();
            if ui.button("Fit").on_hover_text("Fit every view to the image (zoom and pan back to the start)").clicked() {
                for view in &mut self.views {
                    view.nav.reset();
                }
                self.view3d.reset_camera();
            }
            ui.toggle_value(&mut self.crosshair, "Crosshair").on_hover_text("Show where the other two planes cut each view");
            ui.toggle_value(&mut self.smooth, "Smooth").on_hover_text("Interpolate between voxels (off: each voxel is a sharp square)");
        });
    }

    fn status(&self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            match (&self.volume, self.hover) {
                (Some(v), Some((plane, p))) => {
                    let ijk = [p.x.round(), p.y.round(), p.z.round()];
                    let inside = (0..3).all(|a| ijk[a] >= 0.0 && (ijk[a] as usize) < v.dims[a]);
                    if inside {
                        let (i, j, k) = (ijk[0] as usize, ijk[1] as usize, ijk[2] as usize);
                        let ras = v.geometry.world(DVec3::new(i as f64, j as f64, k as f64));
                        let value = v.get(i, j, k);
                        let value = if value.fract() == 0.0 { format!("{value:.0}") } else { format!("{value:.3}") };
                        ui.label(format!("{} · voxel {i}, {j}, {k} · R {:.1}  A {:.1}  S {:.1} mm · value {value}", plane.label(), ras.x, ras.y, ras.z));
                    }
                }
                _ => {
                    ui.weak(" ");
                }
            }
            let over_3d = ui.ctx().pointer_hover_pos().is_some_and(|p| self.rects[1].contains(p));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.weak(if over_3d {
                    "Left drag: rotate · Right drag: pan · Middle drag or wheel: zoom · Double-click: reset"
                } else {
                    "Left drag: slices · Right drag: pan · Middle drag: zoom · Left + right drag: window / level · Wheel: slices · Click: move the crosshair · Double-click: reset"
                });
            });
        });
    }

    fn views_ui(&mut self, ui: &mut egui::Ui, render_state: Option<&egui_wgpu::RenderState>) {
        let full = ui.available_rect_before_wrap();
        let gap = 2.0;
        let size = Vec2::new(((full.width() - gap) * 0.5).floor(), ((full.height() - gap) * 0.5).floor());
        let cell = |col: f32, row: f32| Rect::from_min_size(full.min + Vec2::new(col * (size.x + gap), row * (size.y + gap)), size);
        // Axial top left, 3D top right, sagittal bottom left, coronal bottom right.
        self.rects = [cell(0.0, 0.0), cell(1.0, 0.0), cell(0.0, 1.0), cell(1.0, 1.0)];
        let (Some(vol), Some(state)) = (self.volume.clone(), self.state.as_mut()) else {
            ui.allocate_rect(full, egui::Sense::hover());
            self.empty_message(ui, full);
            return;
        };
        let opts = SliceOptions { left_drag_scrolls: true, crosshair: self.crosshair, smooth: self.smooth };
        let mut hover = None;
        for (view, rect) in self.views.iter_mut().zip([self.rects[0], self.rects[2], self.rects[3]]) {
            let r = ui.scope_builder(egui::UiBuilder::new().max_rect(rect), |ui| view.ui(ui, &vol, state, None, &opts)).inner;
            if let Some(p) = r.hover_voxel {
                hover = Some((view.plane, p));
            }
        }
        self.hover = hover;
        let view3d = &mut self.view3d;
        ui.scope_builder(egui::UiBuilder::new().max_rect(self.rects[1]), |ui| view3d.ui(ui, render_state, &vol, state, None));
    }

    fn empty_message(&self, ui: &mut egui::Ui, rect: Rect) {
        let painter = ui.painter_at(rect);
        let centre = rect.center();
        let weak = Color32::from_gray(150);
        let font = egui::FontId::proportional(15.0);
        if let Some(l) = &self.loading {
            let name = self.source.as_ref().map(|s| s.label.as_str()).unwrap_or("volume");
            painter.text(centre, egui::Align2::CENTER_CENTER, format!("Loading {name}… {:.0}%", l.progress() * 100.0), font, weak);
        } else if let Some(e) = &self.error {
            painter.text(centre, egui::Align2::CENTER_CENTER, format!("Cannot show this volume: {e}"), font, WARN);
        } else {
            painter.text(centre, egui::Align2::CENTER_CENTER, "Select a series or volume in the Volume tab (⌘1) to see it here", font, weak);
        }
    }
}

impl Workspace for SegmentationWorkspace {
    fn name(&self) -> &'static str {
        "Segmentation"
    }

    fn tick(&mut self, ctx: &egui::Context, active: bool) {
        if let Some(r) = self.loading.as_ref().and_then(|l| l.job.poll()) {
            self.loading = None;
            self.finish_loading(r);
        }
        if self.loading.is_some() && active {
            // Progress comes from worker threads that do not ask for repaints.
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
        if active && self.loading.is_none() {
            if let Some(step) = self.script.pop_front() {
                self.run_script_step(step);
                ctx.request_repaint();
            }
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        egui::Panel::top("segmentation.toolbar").show(ui, |ui| self.toolbar(ui));
        egui::Panel::bottom("segmentation.status").show(ui, |ui| self.status(ui));
        let render_state = frame.wgpu_render_state();
        egui::CentralPanel::default().frame(egui::Frame::NONE.fill(BACKDROP)).show(ui, |ui| self.views_ui(ui, render_state));
    }

    fn title(&self) -> String {
        match &self.source {
            Some(s) => format!("Polymix — Segmentation · {}", s.label),
            None => "Polymix — Segmentation".into(),
        }
    }

    fn script_push(&mut self, step: String) {
        self.script.push_back(step);
    }

    fn script_idle(&self) -> bool {
        self.script.is_empty() && self.loading.is_none()
    }
}
