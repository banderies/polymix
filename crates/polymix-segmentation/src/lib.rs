//! The Segmentation workspace: the volume selected in the Volume workspace, in three planes —
//! axial (top left), sagittal (bottom left) and coronal (bottom right) — beside a 3D view
//! (top right), with an editor on the left like 3D Slicer's Segment Editor: segments, effects
//! (threshold, paint, erase, flood fill, islands, margin, smoothing, logical operators),
//! masking, undo, saving, and meshes for the Mesh workspace.
//!
//! It depends only on `polymix-workspace` and the libraries (`mm-volume`, `mm-segment`,
//! `polymix-imaging`), never on another workspace: the shell hands it the Volume workspace's
//! selection through [`SegmentationWorkspace::show_volume`] and takes its meshes through
//! [`SegmentationWorkspace::take_meshes`].

mod effects;
mod panel;
mod segmentation;

use eframe::egui_wgpu;
use effects::{Effect, Histogram, IslandOp, LogicalOp, Op, Outcome, Settings, SmoothMethod};
use egui::{Color32, Rect, Vec2};
use glam::DVec3;
use mm_core::Mesh;
use mm_segment::SegmentInfo;
use mm_volume::geometry::Geometry;
use mm_volume::load::{load_volume, LoadedVolume};
use mm_volume::model::Volume;
use mm_volume::slice::{plane_axes, Plane};
use polymix_imaging::{LabelOverlay, SliceOptions, SliceResponse, SliceView, ViewState, VolumeView3d};
use polymix_workspace::jobs::Job;
use polymix_workspace::Workspace;
use segmentation::{EditableArea, Masking, Mode, Overwrite, Segmentation};
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
#[derive(Clone)]
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

/// Whether a volume's grid is `g`'s, so a segmentation of one fits the other (e.g. an
/// N4-corrected copy of the same series).
fn same_grid(g: &Geometry, v: Option<&Geometry>) -> bool {
    v.is_some_and(|h| h.dims[..3] == g.dims[..3] && (h.matrix() - g.matrix()).to_cols_array().iter().all(|d| d.abs() < 1e-3))
}

/// A volume being read in the background.
struct Loading {
    job: Job<LoadedVolume>,
    progress: Arc<AtomicU32>,
    cancel: Arc<AtomicBool>,
    /// Keep the segmentation (the new volume has the same grid).
    keep: bool,
}

impl Loading {
    fn start(v: &Volume, keep: bool, ctx: &egui::Context) -> Self {
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
        Self { job, progress, cancel, keep }
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

/// Whole-volume work in the background.
struct Running {
    what: String,
    job: Job<Outcome>,
}

/// What the threshold preview was made from.
#[derive(Clone, Copy, PartialEq)]
struct PreviewKey {
    version: u64,
    range: (f32, f32),
    masking: Masking,
    label: u8,
}

/// What the slice views' labels show: the segmentation (at a version) or a preview.
#[derive(Clone, Copy, PartialEq)]
enum OverlayKey {
    Labels(u64),
    Preview(u64),
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
    /// Where the views were drawn last frame: axial, 3D, sagittal, coronal (for scripts);
    /// `Rect::NOTHING` for a view hidden by another filling the view area.
    rects: [Rect; 4],
    /// The view (index into `rects`) filling the view area, if one was double-clicked.
    maximized: Option<usize>,
    script: VecDeque<String>,

    seg: Option<Segmentation>,
    hist: Option<Histogram>,
    settings: Option<Settings>,
    effect: Option<Effect>,
    masking: Masking,
    /// Fill opacity of segments in the slice views.
    opacity: f32,
    running: Option<Running>,
    /// Where the brush stroke in progress last was.
    stroke: Option<DVec3>,
    preview: Option<(PreviewKey, u64, Vec<u8>)>,
    previews_made: u64,
    /// What the slice views' labels show, and the version handed to them.
    overlay: (Option<OverlayKey>, u64),
    /// The outcome of the last edit (an error when true).
    message: Option<(String, bool)>,
    /// Meshes for the Mesh workspace.
    outbox: Vec<(String, Mesh)>,
    /// A volume selected in the Volume workspace whose grid does not fit the segmentation:
    /// waiting for the user to discard the segmentation or keep working.
    pending: Option<(Source, Volume)>,
    /// A selection the user chose not to switch to.
    declined: Option<Source>,
    /// The segment being renamed, the name typed so far and whether its field has had the
    /// keyboard focus.
    renaming: Option<(usize, String, bool)>,
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
            maximized: None,
            script: VecDeque::new(),
            seg: None,
            hist: None,
            settings: None,
            effect: None,
            masking: Masking::default(),
            opacity: 0.5,
            running: None,
            stroke: None,
            preview: None,
            previews_made: 0,
            overlay: (None, 0),
            message: None,
            outbox: Vec::new(),
            pending: None,
            declined: None,
            renaming: None,
        }
    }

    /// Show `v`, the volume selected in the Volume workspace, listed there as `label`. The
    /// same volume is not read again; one on the same grid keeps the segmentation; one on
    /// another grid replaces it only after the user agrees, if there are segments.
    pub fn show_volume(&mut self, v: &Volume, label: &str, ctx: &egui::Context) {
        if let Some(s) = &mut self.source {
            if s.is(v) {
                if s.label != label {
                    s.label = label.to_string();
                }
                self.pending = None;
                return;
            }
        }
        if self.declined.as_ref().is_some_and(|d| d.is(v)) {
            return;
        }
        let has_work = self.seg.as_ref().is_some_and(|s| !s.segments.is_empty());
        let fits = self.volume.as_ref().is_some_and(|cur| same_grid(&cur.geometry, v.geometry.as_ref()));
        if has_work && !fits {
            if self.pending.as_ref().is_none_or(|(p, _)| !p.is(v)) {
                self.pending = Some((Source::new(v, label), v.clone()));
            }
            return;
        }
        self.open(v, label, has_work, ctx);
    }

    fn open(&mut self, v: &Volume, label: &str, keep: bool, ctx: &egui::Context) {
        // Free the old volume first: two large volumes may not fit in memory together.
        self.volume = None;
        self.hover = None;
        self.error = None;
        self.pending = None;
        self.declined = None;
        self.running = None;
        self.stroke = None;
        self.preview = None;
        if !keep {
            self.seg = None;
            self.state = None;
            self.message = None;
        }
        self.source = Some(Source::new(v, label));
        self.loading = Some(Loading::start(v, keep, ctx));
    }

    /// Meshes made for the Mesh workspace (Make mesh), taken by the shell.
    pub fn take_meshes(&mut self) -> Vec<(String, Mesh)> {
        std::mem::take(&mut self.outbox)
    }

    fn finish_loading(&mut self, result: anyhow::Result<LoadedVolume>, keep: bool) {
        match result {
            Ok(vol) => {
                let hist = Histogram::of(&vol, 256);
                let keep = keep && self.seg.as_ref().is_some_and(|s| s.dims == vol.dims);
                if keep {
                    // Same grid: the segments, cursor and views stay; the window is the new
                    // volume's.
                    if let Some(s) = &mut self.state {
                        s.window = vol.auto_window;
                    }
                } else {
                    self.seg = Some(Segmentation::new(vol.dims));
                    self.settings = Some(Settings::for_volume(&vol, &hist));
                    self.effect = None;
                    self.masking = Masking::default();
                    self.views = PLANES.map(SliceView::new);
                    self.view3d.reset_camera();
                }
                if self.state.is_none() {
                    self.state = Some(ViewState::new(&vol));
                }
                if self.settings.is_none() {
                    self.settings = Some(Settings::for_volume(&vol, &hist));
                }
                self.hist = Some(hist);
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

    // ------------------------------------------------------------------ editing

    fn say(&mut self, text: impl Into<String>, error: bool) {
        self.message = Some((text.into(), error));
    }

    /// Start `effect` on the whole volume (with the clicked voxel `seed` for click effects).
    fn start_op(&mut self, effect: Effect, seed: Option<[usize; 3]>, ctx: &egui::Context) {
        let (None, Some(vol), Some(seg), Some(st)) = (&self.running, &self.volume, &mut self.seg, &self.settings) else { return };
        // Threshold and flood fill make a segment when there is none; the rest change one.
        if seg.selected.is_none() && !(matches!(effect, Effect::Threshold | Effect::FloodFill) && seg.add_segment(None).is_some()) {
            self.message = Some(("Select a segment first".into(), true));
            return;
        }
        let label = seg.selected_segment().map_or(1, |s| s.label);
        let uses_segment = !matches!(effect, Effect::Threshold | Effect::FloodFill);
        let segment = if uses_segment { seg.mask_of(label) } else { mm_segment::Mask { dims: vol.dims, data: Vec::new() } };
        let other = (effect == Effect::Logical && st.logical.needs_other()).then(|| st.logical_other.map(|l| seg.mask_of(l))).flatten();
        let (s, vol) = (st.clone(), vol.clone());
        let job = Job::spawn(effect.label(), ctx.clone(), move |_| effects::run(effect, &s, &vol, Op { label, segment, other, seed }));
        self.running = Some(Running { what: effect.label().to_string(), job });
    }

    fn finish_op(&mut self, result: anyhow::Result<Outcome>) {
        let (Some(vol), Some(seg)) = (self.volume.clone(), &mut self.seg) else { return };
        let message = match result {
            Ok(Outcome::Edit { label, mask, mode, what }) => {
                seg.checkpoint();
                let n = seg.apply(&vol, &mask, label, mode, &self.masking);
                seg.changed(true);
                (format!("{what}: {} voxels changed", polymix_workspace::style::format_count(n)), false)
            }
            Ok(Outcome::Split { label, keep, islands }) => {
                seg.checkpoint();
                let base = seg.index_of(label).map(|k| seg.segments[k].name.clone()).unwrap_or_default();
                let all = Masking::default();
                seg.apply(&vol, &keep, label, Mode::Set, &all);
                let mut made = 0;
                for (k, island) in islands.iter().enumerate() {
                    let Some(idx) = seg.push_segment(Some(format!("{base}_{}", k + 2))) else { break };
                    let l = seg.segments[idx].label;
                    seg.apply(&vol, island, l, Mode::Add, &all);
                    made += 1;
                }
                seg.selected = seg.index_of(label);
                seg.changed(true);
                (format!("Split into {} segments", made + 1), false)
            }
            Ok(Outcome::Meshes(meshes)) => {
                let n = meshes.len();
                self.outbox.extend(meshes);
                (format!("{n} mesh(es) sent to the Mesh tab"), false)
            }
            Ok(Outcome::Saved { path, version }) => {
                if seg.version == version {
                    seg.modified = false;
                }
                (format!("Saved {path}"), false)
            }
            Err(e) => (format!("{e:#}"), true),
        };
        self.message = Some(message);
    }

    /// A brush dab at `at` (voxel coordinates) in the view of `plane`; the first one of a
    /// stroke makes the undo step.
    fn brush(&mut self, vol: &LoadedVolume, plane: Plane, at: DVec3, erase: bool) {
        let (Some(seg), Some(st)) = (&mut self.seg, &self.settings) else { return };
        if self.stroke.is_none() {
            if seg.selected.is_none() && (erase || seg.add_segment(None).is_none()) {
                return;
            }
            seg.checkpoint();
        }
        let Some(label) = seg.selected_segment().map(|s| s.label) else { return };
        let disc = (!st.sphere).then(|| plane_axes(&vol.geometry, plane).slice);
        let from = self.stroke.unwrap_or(at);
        let voxels = effects::stroke(vol.dims, vol.geometry.spacing, from, at, st.brush_mm as f64, disc);
        if seg.paint(vol, &voxels, label, erase, &self.masking) > 0 {
            seg.changed(false);
        }
        self.stroke = Some(at);
    }

    fn end_stroke(&mut self) {
        if self.stroke.take().is_some() {
            if let Some(seg) = &mut self.seg {
                // The 3D view catches up once per stroke.
                seg.changed(true);
            }
        }
    }

    /// The slice views' mouse input for the effect in use.
    fn view_input(&mut self, vol: &LoadedVolume, responses: &[(Plane, SliceResponse)], ctx: &egui::Context) {
        let effect = self.effect.filter(|_| self.running.is_none());
        if let Some(st) = &mut self.settings {
            for (_, r) in responses {
                if r.brush_wheel != 0.0 {
                    st.brush_mm = (st.brush_mm * 1.1f32.powf(r.brush_wheel)).clamp(0.2, 50.0);
                }
            }
        }
        match effect {
            Some(e @ (Effect::Paint | Effect::Erase)) => match responses.iter().find_map(|(p, r)| r.left_drag_voxel.map(|v| (*p, v))) {
                Some((plane, v)) => self.brush(vol, plane, v, e == Effect::Erase),
                None => self.end_stroke(),
            },
            Some(e @ (Effect::FloodFill | Effect::Islands)) => {
                self.end_stroke();
                if let Some(v) = responses.iter().find_map(|(_, r)| r.clicked_voxel) {
                    let seed = [0, 1, 2].map(|a| v[a].round().clamp(0.0, vol.dims[a] as f64 - 1.0) as usize);
                    if e == Effect::FloodFill || self.settings.as_ref().is_some_and(|s| s.islands.by_click()) {
                        self.start_op(e, Some(seed), ctx);
                    }
                }
            }
            _ => self.end_stroke(),
        }
    }

    /// Make the Threshold preview (the segment as Apply would leave it) while Threshold is in
    /// use; `None` otherwise.
    fn update_preview(&mut self, vol: &LoadedVolume) {
        let (Some(seg), Some(st)) = (&self.seg, &self.settings) else { return };
        let label = seg.selected_segment().map(|s| s.label).filter(|_| self.effect == Some(Effect::Threshold) && self.running.is_none());
        let Some(label) = label else {
            self.preview = None;
            return;
        };
        let key = PreviewKey { version: seg.version, range: st.threshold, masking: self.masking, label };
        if self.preview.as_ref().is_none_or(|(k, _, _)| *k != key) {
            let mut preview = seg.scratch_copy();
            let mask = mm_segment::ops::threshold(vol, st.threshold.0, st.threshold.1);
            preview.apply(vol, &mask, label, Mode::Set, &self.masking);
            self.previews_made += 1;
            self.preview = Some((key, self.previews_made, preview.labels));
        }
    }

    fn make_mesh(&mut self, ctx: &egui::Context) {
        let (None, Some(vol), Some(seg), Some(st)) = (&self.running, &self.volume, &self.seg, &self.settings) else { return };
        let chosen: Vec<(String, mm_segment::Mask)> = seg.segments.iter().enumerate().filter(|(k, s)| if st.mesh_all_visible { s.visible } else { seg.selected == Some(*k) }).map(|(_, s)| (s.name.clone(), seg.mask_of(s.label))).collect();
        if chosen.is_empty() {
            self.say("Select a segment first", true);
            return;
        }
        let (geometry, smoothing) = (vol.geometry.clone(), st.mesh_smoothing_mm as f64);
        let job = Job::spawn("Making meshes", ctx.clone(), move |progress| Ok(Outcome::Meshes(effects::meshes(chosen, &geometry, smoothing, progress)?)));
        self.running = Some(Running { what: "Making meshes".into(), job });
    }

    fn save_dialog(&mut self, ctx: &egui::Context) {
        let name: String = self.source.as_ref().map_or("Segmentation".into(), |s| s.label.chars().map(|c| if c.is_alphanumeric() || " -_.".contains(c) { c } else { '_' }).collect());
        let path = rfd::FileDialog::new()
            .set_title("Save segmentation")
            .set_file_name(format!("{name}.seg.nrrd"))
            .add_filter("3D Slicer segmentation (.seg.nrrd)", &["nrrd"])
            .add_filter("NIfTI label map (.nii.gz)", &["gz", "nii"])
            .save_file();
        if let Some(p) = path {
            self.save_to(p, ctx);
        }
    }

    fn save_to(&mut self, path: PathBuf, ctx: &egui::Context) {
        let (None, Some(vol), Some(seg)) = (&self.running, &self.volume, &self.seg) else { return };
        let (labels, dims, geometry, version) = (seg.labels.clone(), seg.dims, vol.geometry.clone(), seg.version);
        let infos: Vec<SegmentInfo> = seg.segments.iter().map(|s| SegmentInfo { name: s.name.clone(), colour: [s.colour.r(), s.colour.g(), s.colour.b()], label: s.label }).collect();
        let job = Job::spawn("Saving", ctx.clone(), move |_| {
            let name = path.file_name().map(|n| n.to_string_lossy().to_lowercase()).unwrap_or_default();
            if name.ends_with(".nii") || name.ends_with(".nii.gz") {
                let description = infos.iter().map(|s| format!("{}={}", s.label, s.name)).collect::<Vec<_>>().join(" ");
                mm_segment::write_nifti(&path, &labels, dims, &geometry, &description)?;
            } else {
                mm_segment::write_seg_nrrd(&path, &labels, dims, &geometry, &infos)?;
            }
            Ok(Outcome::Saved { path: path.display().to_string(), version })
        });
        self.running = Some(Running { what: "Saving".into(), job });
    }

    fn keys(&mut self, ctx: &egui::Context) {
        if ctx.egui_wants_keyboard_input() || self.running.is_some() {
            return;
        }
        let (redo, undo, escape) = ctx.input_mut(|i| {
            let redo = i.consume_key(egui::Modifiers::COMMAND | egui::Modifiers::SHIFT, egui::Key::Z);
            (redo, i.consume_key(egui::Modifiers::COMMAND, egui::Key::Z), i.consume_key(egui::Modifiers::NONE, egui::Key::Escape))
        });
        if let Some(seg) = &mut self.seg {
            if redo {
                seg.redo();
            } else if undo {
                seg.undo();
            }
        }
        if escape {
            self.effect = None;
        }
    }

    // ------------------------------------------------------------------ scripts

    fn run_script_step(&mut self, step: String, ctx: &egui::Context) {
        let (cmd, arg) = step.split_once(':').unwrap_or((step.as_str(), ""));
        let nums = |s: &str| s.split(',').filter_map(|x| x.trim().parse::<f64>().ok()).collect::<Vec<_>>();
        let seg_label = |seg: &Option<Segmentation>, k: &str| k.parse::<usize>().ok().and_then(|k| seg.as_ref()?.segments.get(k).map(|s| s.label));
        match cmd {
            // Counts and positions only: never patients' names.
            "seg.log" => self.log(arg),
            "seg.add" => {
                if let Some(seg) = &mut self.seg {
                    seg.add_segment((!arg.is_empty()).then(|| arg.to_string()));
                }
            }
            "seg.select" => {
                if let (Some(seg), Ok(k)) = (&mut self.seg, arg.parse::<usize>()) {
                    seg.selected = (k < seg.segments.len()).then_some(k);
                }
            }
            "seg.hide" => {
                if let (Some(seg), Ok(k)) = (&mut self.seg, arg.parse::<usize>()) {
                    if let Some(s) = seg.segments.get_mut(k) {
                        s.visible = !s.visible;
                        seg.changed(true);
                    }
                }
            }
            "seg.effect" => self.effect = Effect::ALL.into_iter().find(|e| e.key() == arg),
            "seg.apply" => {
                if let Some(e) = self.effect {
                    self.start_op(e, None, ctx);
                }
            }
            // seg.click:i,j,k — a click effect at a voxel.
            "seg.click" => {
                if let (Some(e), [i, j, k]) = (self.effect, nums(arg).as_slice()) {
                    self.start_op(e, Some([*i as usize, *j as usize, *k as usize]), ctx);
                }
            }
            "seg.area" => {
                let (a, k) = arg.split_once(',').unwrap_or((arg, ""));
                let l = seg_label(&self.seg, k).unwrap_or(0);
                self.masking.area = match a {
                    "inside_all" => EditableArea::InsideAll,
                    "outside_all" => EditableArea::OutsideAll,
                    "inside_visible" => EditableArea::InsideVisible,
                    "outside_visible" => EditableArea::OutsideVisible,
                    "inside" => EditableArea::Inside(l),
                    "outside" => EditableArea::Outside(l),
                    _ => EditableArea::Everywhere,
                };
            }
            "seg.range" => {
                self.masking.intensity = match nums(arg)[..] {
                    [lo, hi] => Some((lo as f32, hi as f32)),
                    _ => None,
                }
            }
            "seg.overwrite" => {
                self.masking.overwrite = match arg {
                    "visible" => Overwrite::Visible,
                    "none" => Overwrite::None,
                    _ => Overwrite::All,
                }
            }
            "seg.undo" => {
                if let Some(seg) = &mut self.seg {
                    seg.undo();
                }
            }
            "seg.redo" => {
                if let Some(seg) = &mut self.seg {
                    seg.redo();
                }
            }
            "seg.save" => self.save_to(PathBuf::from(arg), ctx),
            // The banner's buttons for a volume on another grid.
            "seg.discard" => {
                if let Some((p, v)) = self.pending.take() {
                    self.open(&v, &p.label, false, ctx);
                }
            }
            "seg.keep" => self.declined = self.pending.take().map(|(p, _)| p),
            // seg.maximize:axial|3d|sagittal|coronal — one view fills the view area (anything
            // else: back to four).
            "seg.maximize" => self.maximized = ["axial", "3d", "sagittal", "coronal"].iter().position(|n| *n == arg),
            _ => {
                // Effect settings.
                let Some(st) = &mut self.settings else { return };
                match cmd {
                    "seg.threshold" => {
                        if let [lo, hi] = nums(arg)[..] {
                            st.threshold = (lo as f32, hi as f32);
                        }
                        self.effect = Some(Effect::Threshold);
                    }
                    "seg.brush" => st.brush_mm = arg.parse().unwrap_or(st.brush_mm),
                    "seg.sphere" => st.sphere = arg == "on",
                    "seg.island" => {
                        if let Some(op) = IslandOp::ALL.into_iter().find(|o| o.key() == arg) {
                            st.islands = op;
                        }
                        self.effect = Some(Effect::Islands);
                    }
                    "seg.min_island" => st.min_island = arg.parse().unwrap_or(st.min_island),
                    "seg.flood" => {
                        st.flood_use_threshold = arg == "threshold";
                        st.flood_tolerance = arg.parse().unwrap_or(st.flood_tolerance);
                        self.effect = Some(Effect::FloodFill);
                    }
                    "seg.diameter" => st.flood_min_diameter_mm = arg.parse().unwrap_or(0.0),
                    "seg.margin" => {
                        st.margin_mm = arg.parse().unwrap_or(st.margin_mm);
                        self.effect = Some(Effect::Margin);
                    }
                    "seg.smooth" => {
                        let (m, mm) = arg.split_once(',').unwrap_or((arg, ""));
                        if let Some(m) = SmoothMethod::ALL.into_iter().find(|x| x.key() == m) {
                            st.smoothing = m;
                        }
                        st.smoothing_mm = mm.parse().unwrap_or(st.smoothing_mm);
                        self.effect = Some(Effect::Smoothing);
                    }
                    "seg.logical" => {
                        let (op, other) = arg.split_once(',').unwrap_or((arg, ""));
                        if let Some(o) = LogicalOp::ALL.into_iter().find(|x| x.key() == op) {
                            st.logical = o;
                        }
                        if let Some(l) = seg_label(&self.seg, other) {
                            st.logical_other = Some(l);
                        }
                        self.effect = Some(Effect::Logical);
                    }
                    "seg.mesh" => {
                        st.mesh_all_visible = arg == "all";
                        self.make_mesh(ctx);
                    }
                    _ => log::warn!("unknown Segmentation script step {step}"),
                }
            }
        }
    }

    fn log(&mut self, tag: &str) {
        let rect = |r: &Rect| format!("{:.0},{:.0},{:.0},{:.0}", r.min.x, r.min.y, r.width(), r.height());
        let rects = ["axial", "3d", "sagittal", "coronal"].iter().zip(&self.rects).map(|(n, r)| format!("{n}={}", rect(r))).collect::<Vec<_>>().join(" ");
        let (Some(v), Some(s)) = (&self.volume, &self.state) else {
            println!("[segmentation] {tag} no volume ({}) {rects}", self.error.as_deref().unwrap_or(if self.loading.is_some() { "loading" } else { "none selected" }));
            return;
        };
        let zoom = self.views.iter().map(|v| format!("{:.2}", v.nav.zoom)).collect::<Vec<_>>().join(",");
        let pan = self.views.iter().map(|v| format!("{:.0}/{:.0}", v.nav.pan.x, v.nav.pan.y)).collect::<Vec<_>>().join(",");
        let segments = match &mut self.seg {
            Some(seg) => {
                let counts = seg.counts().to_vec();
                let list = seg.segments.iter().map(|x| format!("{}:{}{}", x.label, counts[x.label as usize], if x.visible { "" } else { "(hidden)" })).collect::<Vec<_>>().join(",");
                format!(" segments=[{list}] selected={:?} undo={} redo={}", seg.selected, seg.can_undo(), seg.can_redo())
            }
            None => String::new(),
        };
        let mut message = self.message.as_ref().map(|(m, e)| format!(" message={}{m:?}", if *e { "ERROR " } else { "" })).unwrap_or_default();
        if self.pending.is_some() {
            message += " pending=another-grid";
        }
        println!(
            "[segmentation] {tag} dims={}x{}x{} cursor={:.1},{:.1},{:.1} window={:.1},{:.1} zoom={zoom} pan={pan}{segments}{message} {rects}",
            v.dims[0], v.dims[1], v.dims[2], s.cursor.x, s.cursor.y, s.cursor.z, s.window.0, s.window.1
        );
    }

    // ------------------------------------------------------------------ UI parts

    /// Asks before a volume on another grid replaces the segmentation. It floats over the
    /// views, so they do not move.
    fn switch_prompt(&mut self, ctx: &egui::Context) {
        let Some((p, v)) = self.pending.clone() else { return };
        egui::Window::new("Open another volume?").anchor(egui::Align2::CENTER_TOP, egui::vec2(0.0, 80.0)).collapsible(false).resizable(false).fade_in(false).show(ctx, |ui| {
            ui.label(format!("{} is selected in the Volume tab.", p.label));
            ui.label("Its voxel grid differs from this volume's, so opening it discards the segmentation.");
            ui.horizontal(|ui| {
                if ui.button("Discard and open").clicked() {
                    self.open(&v, &p.label, false, ctx);
                }
                if ui.button("Keep segmenting").clicked() {
                    self.declined = Some(p);
                    self.pending = None;
                }
            });
        });
    }

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
                        let segment = self.seg.as_ref().and_then(|s| s.index_of(s.labels[v.index(i, j, k)]).map(|x| format!(" · {}", s.segments[x].name))).unwrap_or_default();
                        ui.label(format!("{} · voxel {i}, {j}, {k} · R {:.1}  A {:.1}  S {:.1} mm · value {value}{segment}", plane.label(), ras.x, ras.y, ras.z));
                    }
                }
                _ => {
                    ui.weak(" ");
                }
            }
            let over_3d = ui.ctx().pointer_hover_pos().is_some_and(|p| self.rects[1].contains(p));
            let tool = self.effect.is_some_and(|e| self.settings.as_ref().is_some_and(|s| s.takes_left_button(e)));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.weak(if over_3d {
                    "Left drag: rotate · Right drag: pan · Middle drag or wheel: zoom · Double-click: fill the view area / back to four views"
                } else if tool {
                    "Left: the effect · Right drag: pan · Middle drag: zoom · Left + right drag: window / level · Wheel: slices · Esc: put the effect away"
                } else {
                    "Left drag: slices · Right drag: pan · Middle drag: zoom · Left + right drag: window / level · Wheel: slices · Click: move the crosshair · Double-click: fill the view area / back to four views"
                });
            });
        });
    }

    fn views_ui(&mut self, ui: &mut egui::Ui, render_state: Option<&egui_wgpu::RenderState>) {
        let full = ui.available_rect_before_wrap();
        let gap = 2.0;
        let size = Vec2::new(((full.width() - gap) * 0.5).floor(), ((full.height() - gap) * 0.5).floor());
        let cell = |col: f32, row: f32| Rect::from_min_size(full.min + Vec2::new(col * (size.x + gap), row * (size.y + gap)), size);
        // Axial top left, 3D top right, sagittal bottom left, coronal bottom right; or one of
        // them filling the area.
        self.rects = match self.maximized {
            Some(m) => std::array::from_fn(|i| if i == m { full } else { Rect::NOTHING }),
            None => [cell(0.0, 0.0), cell(1.0, 0.0), cell(0.0, 1.0), cell(1.0, 1.0)],
        };
        let Some(vol) = self.volume.clone().filter(|_| self.state.is_some()) else {
            ui.allocate_rect(full, egui::Sense::hover());
            self.empty_message(ui, full);
            return;
        };
        let ctx = ui.ctx().clone();
        let colours = self.seg.as_ref().map(|s| s.colours()).unwrap_or_default();
        let has_segments = self.seg.as_ref().is_some_and(|s| !s.segments.is_empty());
        let effect = self.effect.filter(|_| self.running.is_none());
        let opts = SliceOptions {
            left_drag_scrolls: !effect.is_some_and(|e| self.settings.as_ref().is_some_and(|s| s.takes_left_button(e))),
            crosshair: self.crosshair,
            smooth: self.smooth,
            brush_mm: matches!(effect, Some(Effect::Paint | Effect::Erase)).then(|| self.settings.as_ref().map_or(5.0, |s| s.brush_mm)),
        };
        // Slice views: the labels (or the threshold preview) under a version of their own.
        self.update_preview(&vol);
        let shown: Option<(OverlayKey, &[u8])> = match (&self.preview, &self.seg) {
            (Some((_, id, labels)), _) => Some((OverlayKey::Preview(*id), labels)),
            (None, Some(seg)) => Some((OverlayKey::Labels(seg.version), &seg.labels)),
            _ => None,
        };
        let previous = self.overlay;
        let overlay = shown.filter(|_| has_segments).map(|(key, labels)| {
            let version = if previous.0 == Some(key) { previous.1 } else { previous.1 + 1 };
            (key, LabelOverlay { labels, colours: &colours, opacity: self.opacity, version })
        });
        let state = self.state.as_mut().expect("checked above");
        let mut responses = Vec::new();
        let mut toggle = None;
        for ((view, rect), at) in self.views.iter_mut().zip([self.rects[0], self.rects[2], self.rects[3]]).zip([0, 2, 3]) {
            if !rect.is_positive() {
                continue;
            }
            let r = ui.scope_builder(egui::UiBuilder::new().max_rect(rect), |ui| view.ui(ui, &vol, state, overlay.as_ref().map(|o| &o.1), &opts)).inner;
            // Not while the left button is a tool's (a double click is two clicks of the tool).
            if r.double_clicked && opts.left_drag_scrolls {
                toggle = Some(at);
            }
            responses.push((view.plane, r));
        }
        if let Some((key, o)) = &overlay {
            self.overlay = (Some(*key), o.version);
        }
        self.hover = responses.iter().find_map(|(p, r)| r.hover_voxel.map(|v| (*p, v)));
        self.view_input(&vol, &responses, &ctx);
        // 3D: the finished segments (not every brush dab).
        let state = self.state.as_ref().expect("checked above");
        let labels3d = self.seg.as_ref().filter(|_| has_segments).map(|s| LabelOverlay { labels: &s.labels, colours: &colours, opacity: 1.0, version: s.version_3d });
        let view3d = &mut self.view3d;
        if self.rects[1].is_positive() {
            let r3 = ui.scope_builder(egui::UiBuilder::new().max_rect(self.rects[1]), |ui| view3d.ui(ui, render_state, &vol, state, labels3d.as_ref())).inner;
            if r3.double_clicked() {
                toggle = Some(1);
            }
        }
        if let Some(at) = toggle {
            self.maximized = if self.maximized.is_some() { None } else { Some(at) };
        }
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
            let keep = self.loading.as_ref().is_some_and(|l| l.keep);
            self.loading = None;
            self.finish_loading(r, keep);
        }
        if let Some(r) = self.running.as_ref().and_then(|r| r.job.poll()) {
            self.running = None;
            self.finish_op(r);
        }
        if (self.loading.is_some() || self.running.is_some()) && active {
            // Progress comes from worker threads that do not ask for repaints.
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
        if active && self.loading.is_none() && self.running.is_none() {
            if let Some(step) = self.script.pop_front() {
                self.run_script_step(step, ctx);
                ctx.request_repaint();
            }
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.keys(&ctx);
        egui::Panel::top("segmentation.toolbar").show(ui, |ui| self.toolbar(ui));
        egui::Panel::bottom("segmentation.status").show(ui, |ui| self.status(ui));
        egui::Panel::left("segmentation.editor").resizable(true).default_size(300.0).show(ui, |ui| self.editor(ui));
        let render_state = frame.wgpu_render_state();
        egui::CentralPanel::default().frame(egui::Frame::NONE.fill(BACKDROP)).show(ui, |ui| self.views_ui(ui, render_state));
        self.switch_prompt(&ctx);
    }

    fn title(&self) -> String {
        let edited = if self.seg.as_ref().is_some_and(|s| s.modified) { " (edited)" } else { "" };
        match &self.source {
            Some(s) => format!("Polymix — Segmentation · {}{edited}", s.label),
            None => "Polymix — Segmentation".into(),
        }
    }

    fn script_push(&mut self, step: String) {
        self.script.push_back(step);
    }

    fn script_idle(&self) -> bool {
        self.script.is_empty() && self.loading.is_none() && self.running.is_none()
    }
}
