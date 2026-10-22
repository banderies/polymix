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

mod agent;
mod drag;
mod effects;
mod import;
mod panel;
mod segmentation;
mod sketch;
mod vessels;

use eframe::egui_wgpu;
use effects::{Effect, Histogram, IslandOp, LogicalOp, Op, Outcome, Settings, SmoothMethod};
use egui::{Color32, Rect};
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
use rayon::prelude::*;
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
    /// Its edits ignore the masking settings (Logical's Bypass masking).
    bypass: bool,
}

/// What the threshold preview was made from.
#[derive(Clone, Copy, PartialEq)]
struct PreviewKey {
    version: u64,
    range: (f32, f32),
    masking: Masking,
    label: u8,
}

/// What the slice views' labels show: the segmentation (at a version) or a preview, in the
/// segments' colours (at a style version).
#[derive(Clone, Copy, PartialEq)]
enum OverlayKey {
    Labels(u64, u64),
    Preview(u64, u64),
    /// Select part's part over the labels: (part id, its version, style).
    Part(u64, u64, u64),
}

/// The last brush dab of the stroke going on: where, how, and the label map's version after
/// it. While all of it holds, painting that dab again would change nothing.
#[derive(Clone, Copy, PartialEq)]
struct DabKey {
    at: DVec3,
    label: u8,
    erase: bool,
    radius: f64,
    disc: Option<usize>,
    masking: Masking,
    version: u64,
}

/// The Overlay menu's image being read in the background. Dropping it (another is chosen, or
/// the overlay is removed) stops the read.
struct OverlayLoad {
    job: Job<polymix_imaging::Fusion>,
    cancel: Arc<AtomicBool>,
}

impl Drop for OverlayLoad {
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
    /// Its last dab.
    last_dab: Option<DabKey>,
    preview: Option<(PreviewKey, u64, Vec<u8>)>,
    previews_made: u64,
    /// What the slice views' labels show, and the version handed to them.
    overlay: (Option<OverlayKey>, u64),
    /// The outcome of the last edit (an error when true).
    message: Option<(String, bool)>,
    /// Meshes for the Mesh workspace.
    outbox: Vec<(String, Mesh)>,
    /// They were made for an agent (Mesh adds them without asking).
    outbox_by_agent: bool,
    /// Images saved for the Volume workspace (Mask volume).
    volume_outbox: Vec<PathBuf>,
    /// A volume selected in the Volume workspace whose grid does not fit the segmentation:
    /// waiting for the user to discard the segmentation or keep working.
    pending: Option<(Source, Volume)>,
    /// A selection the user chose not to switch to.
    declined: Option<Source>,
    /// The segment being renamed, the name typed so far and whether its field has had the
    /// keyboard focus.
    renaming: Option<(usize, String, bool)>,
    /// The operation running was an agent's: its outcome's message says so.
    agent_op: bool,
    /// Draw's or Scissors' shape in progress.
    sketch: Option<sketch::Sketch>,
    /// Level tracing's region under the pointer.
    traced: Option<sketch::Traced>,
    /// The effect in use before the current one, and the current one as last seen (Space
    /// switches between the two).
    effects_seen: (Option<Effect>, Option<Effect>),
    /// The segment list shows segments whose names contain this, with this status.
    filter: (String, Option<segmentation::Status>),
    /// Points clicked for Draw tube and Surface cut (voxel coordinates).
    markers: Vec<DVec3>,
    /// Grow from seeds' or Watershed's result, shown until applied or cancelled.
    seeded: Option<Seeded>,
    /// The brush stroke going on is in the 3D view.
    stroke_3d: bool,
    /// Scissors drawn in the 3D view: the shape so far (screen points) and where it started.
    scissors_3d: Option<(Vec<egui::Pos2>, egui::Pos2)>,
    /// How the views share the view area.
    layout: polymix_imaging::Layout,
    /// Rulers, angles and regions measured in the slice views (their tool takes the left
    /// button before the effect).
    measure: polymix_imaging::Measurements,
    /// Centerline and cross-sections of a segment.
    vessels: vessels::Vessels,
    /// Images that can be shown over this one (from the Volume workspace, asked for when the
    /// Overlay menu opens), and the one being loaded.
    overlay_choices: Vec<(Volume, String)>,
    wants_choices: bool,
    /// The Overlay menu was open last frame.
    overlay_menu_open: bool,
    fusion_job: Option<OverlayLoad>,
    /// Threshold's local histogram: the region drawn and its values.
    local: Option<sketch::LocalRegion>,
    /// Segment statistics last measured (label, measures), for the label map at a version.
    stats: Option<(u64, Vec<(u8, mm_segment::SegmentStats)>)>,
    /// Threshold grow's ⌘ / Ctrl + right drag in progress.
    grow: Option<drag::Grow>,
    /// Select part's part: being dragged out, or waiting to be used.
    part: Option<drag::Part>,
}

/// The image the Segmentation tab shows: its name, the crosshair (voxel coordinates) and the
/// window.
pub struct ShownImage {
    pub volume: Arc<LoadedVolume>,
    pub name: String,
    pub cursor: [f64; 3],
    pub window: (f64, f64),
}

/// A seeded effect's result: the labels it gives each voxel of its region (0 elsewhere), for
/// the label map at `version`, from the seed labels `seeds`.
struct Seeded {
    effect: Effect,
    version: u64,
    labels: Vec<u8>,
    seeds: Vec<u8>,
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
            last_dab: None,
            preview: None,
            previews_made: 0,
            overlay: (None, 0),
            message: None,
            outbox: Vec::new(),
            outbox_by_agent: false,
            volume_outbox: Vec::new(),
            pending: None,
            declined: None,
            renaming: None,
            agent_op: false,
            sketch: None,
            traced: None,
            effects_seen: (None, None),
            filter: (String::new(), None),
            markers: Vec::new(),
            seeded: None,
            stroke_3d: false,
            scissors_3d: None,
            layout: polymix_imaging::Layout::FourUp,
            measure: polymix_imaging::Measurements::default(),
            vessels: vessels::Vessels::default(),
            overlay_choices: Vec::new(),
            wants_choices: false,
            overlay_menu_open: false,
            fusion_job: None,
            local: None,
            stats: None,
            grow: None,
            part: None,
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
        self.agent_op = false;
        self.stroke = None;
        self.last_dab = None;
        self.preview = None;
        self.sketch = None;
        self.traced = None;
        self.markers.clear();
        self.seeded = None;
        self.local = None;
        self.stats = None;
        self.fusion_job = None;
        // The list is made again for the new volume.
        self.overlay_choices.clear();
        self.wants_choices = true;
        if !keep {
            self.measure.items.clear();
            self.vessels = vessels::Vessels::default();
        }
        if !keep {
            self.seg = None;
            self.state = None;
            self.message = None;
        }
        self.source = Some(Source::new(v, label));
        self.loading = Some(Loading::start(v, keep, ctx));
    }

    /// Meshes made for the Mesh workspace (Make mesh), taken by the shell, and whether they
    /// were made for an agent.
    pub fn take_meshes(&mut self) -> (Vec<(String, Mesh)>, bool) {
        (std::mem::take(&mut self.outbox), std::mem::take(&mut self.outbox_by_agent))
    }

    /// Measure every segment (in the background; the table shows when done).
    fn measure(&mut self, ctx: &egui::Context) {
        let (None, Some(vol), Some(seg)) = (&self.running, self.volume.clone(), &self.seg) else { return };
        let (wanted, version) = (seg.segments.iter().map(|s| s.label).collect::<Vec<_>>(), seg.version);
        // One copy of the label map; each segment's mask is made when it is measured.
        let (labels, dims) = (seg.labels.clone(), seg.dims);
        let job = Job::spawn("Measuring segments", ctx.clone(), move |progress| {
            let n = wanted.len().max(1);
            let mut rows = Vec::new();
            for (k, &label) in wanted.iter().enumerate() {
                if !progress(k as f32 / n as f32) {
                    anyhow::bail!("cancelled");
                }
                rows.extend(mm_segment::segment_stats(&mm_segment::Mask::from_labels(dims, &labels, label), &vol).map(|s| (label, s)));
            }
            Ok(Outcome::Stats { version, rows })
        });
        self.running = Some(Running { what: "Measuring segments".into(), job, bypass: false });
    }

    /// Make segments of the voxels inside a mesh from the Mesh workspace (in the background;
    /// the result arrives as a new segment).
    pub fn import_mesh(&mut self, name: String, mesh: Mesh, ctx: &egui::Context) {
        let Some(vol) = self.volume.clone() else {
            self.say(format!("{name}: select the volume to segment in the Volume tab first, then send the mesh again"), true);
            return;
        };
        if self.running.is_some() {
            self.say("Busy: send the mesh again when the work running has finished", true);
            return;
        }
        let what = format!("Mesh {name}");
        let job = Job::spawn("Voxelising the mesh", ctx.clone(), move |_| Ok(Outcome::Imported { imported: import::voxelise(&mesh, &name, &vol)?, what }));
        self.running = Some(Running { what: "Voxelising the mesh".into(), job, bypass: false });
    }

    /// Add the segments of a label map file (3D Slicer `.seg.nrrd`, NIfTI) on this volume.
    fn load_segmentation(&mut self, path: PathBuf, ctx: &egui::Context) {
        let (None, Some(vol)) = (&self.running, self.volume.clone()) else { return };
        let what = path.file_name().map_or("the file".into(), |n| n.to_string_lossy().to_string());
        let job = Job::spawn("Loading segmentation", ctx.clone(), move |_| Ok(Outcome::Imported { imported: import::read_label_map(&path, &vol)?, what }));
        self.running = Some(Running { what: "Loading segmentation".into(), job, bypass: false });
    }

    fn load_dialog(&mut self, ctx: &egui::Context) {
        let path = rfd::FileDialog::new()
            .set_title("Load segmentation")
            .add_filter("Segmentation or label map (.seg.nrrd, .nrrd, .nii, .nii.gz)", &["nrrd", "nhdr", "nii", "gz"])
            .pick_file();
        if let Some(p) = path {
            self.load_segmentation(p, ctx);
        }
    }

    /// Images saved for the Volume workspace (it lists them), taken by the shell.
    pub fn take_volumes(&mut self) -> Vec<PathBuf> {
        std::mem::take(&mut self.volume_outbox)
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
        let bypass = effect == Effect::Logical && st.bypass_masking;
        // Split volume: an image per visible segment.
        if effect == Effect::SplitVolume {
            let visible: Vec<(String, u8)> = seg.segments.iter().filter(|s| s.visible).map(|s| (s.name.clone(), s.label)).collect();
            if visible.is_empty() {
                self.message = Some(("Split volume works on the visible segments: show some".into(), true));
                return;
            }
            let (pad, blank, vol, labels) = (st.split_pad_mm as f64, st.split_blank.then_some(st.mask_outside), vol.clone(), seg.labels.clone());
            let job = Job::spawn("Splitting the volume", ctx.clone(), move |progress| Ok(Outcome::Volumes { paths: effects::split_volume(&vol, &labels, &visible, pad, blank, progress)? }));
            self.running = Some(Running { what: "Splitting the volume".into(), job, bypass: false });
            return;
        }
        // A level set following the overlay (e.g. vesselness): its window maps to the speed.
        if effect == Effect::LevelSet && st.ls_overlay {
            let Some(f) = self.state.as_ref().and_then(|s| s.fusion.clone()) else {
                self.message = Some(("Level set: show an overlay (e.g. vesselness) to follow, or use the threshold range".into(), true));
                return;
            };
            let (segment, params, vol) = (seg.mask_of(label), st.level_set_params(), vol.clone());
            if segment.count() == 0 {
                self.message = Some(("Level set: paint or threshold a rough start inside the vessel first".into(), true));
                return;
            }
            let job = Job::spawn("Level set", ctx.clone(), move |progress| {
                let (c, w) = f.window;
                let speed: Vec<f32> = f.values.par_iter().map(|&v| (0.5 + (v as f64 - c) / w.max(1e-12)).clamp(0.0, 1.0) as f32).collect();
                let grown = mm_segment::level_set(&vol, &segment, Some(&speed), &params, progress)?;
                Ok(Outcome::Edit { label, mask: grown, mode: Mode::Set, what: "Level set (overlay)".into() })
            });
            self.running = Some(Running { what: "Level set".into(), job, bypass });
            return;
        }
        // Joint smoothing: every visible segment together, as one edit.
        if effect == Effect::Smoothing && st.smoothing == SmoothMethod::Joint {
            let visible: Vec<u8> = seg.segments.iter().filter(|s| s.visible).map(|s| s.label).collect();
            if visible.is_empty() {
                self.message = Some(("Joint smoothing works on the visible segments: show some".into(), true));
                return;
            }
            let (mut labels, sigma, vol) = (seg.labels.clone(), st.smoothing_mm as f64, vol.clone());
            let job = Job::spawn("Joint smoothing", ctx.clone(), move |_| {
                mm_segment::joint_smooth(&mut labels, vol.dims, vol.geometry.spacing, &visible, sigma);
                let edits = visible.iter().map(|&l| (l, mm_segment::Mask::from_labels(vol.dims, &labels, l), Mode::Set)).collect();
                Ok(Outcome::Edits { edits, what: "Joint smoothing".into() })
            });
            self.running = Some(Running { what: "Joint smoothing".into(), job, bypass });
            return;
        }
        // Margin, Smoothing and Hollow on every visible segment, one after another, as one edit.
        if st.apply_visible && matches!(effect, Effect::Margin | Effect::Smoothing | Effect::Hollow) {
            let visible: Vec<u8> = seg.segments.iter().filter(|s| s.visible).map(|s| s.label).collect();
            // One copy of the label map; each segment's mask is made when its turn comes.
            let (s, vol, labels, dims) = (st.clone(), vol.clone(), seg.labels.clone(), seg.dims);
            let job = Job::spawn(effect.label(), ctx.clone(), move |progress| {
                let n = visible.len().max(1);
                let mut edits = Vec::new();
                for (k, &label) in visible.iter().enumerate() {
                    if !progress(k as f32 / n as f32) {
                        anyhow::bail!("cancelled");
                    }
                    let segment = mm_segment::Mask::from_labels(dims, &labels, label);
                    if let Outcome::Edit { label, mask, mode, .. } = effects::run(effect, &s, &vol, Op { label, segment, other: None, seed: None })? {
                        edits.push((label, mask, mode));
                    }
                }
                Ok(Outcome::Edits { edits, what: format!("{} (visible segments)", effect.label()) })
            });
            self.running = Some(Running { what: effect.label().to_string(), job, bypass });
            return;
        }
        let uses_segment = !matches!(effect, Effect::Threshold | Effect::FloodFill);
        let segment = if uses_segment { seg.mask_of(label) } else { mm_segment::Mask { dims: vol.dims, data: Vec::new() } };
        let other = (effect == Effect::Logical && st.logical.needs_other()).then(|| st.logical_other.map(|l| seg.mask_of(l))).flatten();
        let (s, vol) = (st.clone(), vol.clone());
        let job = Job::spawn(effect.label(), ctx.clone(), move |_| effects::run(effect, &s, &vol, Op { label, segment, other, seed }));
        self.running = Some(Running { what: effect.label().to_string(), job, bypass });
    }

    fn finish_op(&mut self, result: anyhow::Result<Outcome>, bypass: bool) {
        let agent = std::mem::take(&mut self.agent_op);
        let (Some(vol), Some(seg)) = (self.volume.clone(), &mut self.seg) else { return };
        let masking = if bypass { Masking::default() } else { self.masking };
        let message = match result {
            Ok(Outcome::Edit { label, mask, mode, what }) => {
                seg.checkpoint();
                let n = seg.apply(&vol, &mask, label, mode, &masking);
                seg.changed(true);
                (format!("{what}: {} voxels changed", polymix_workspace::style::format_count(n)), false)
            }
            Ok(Outcome::Seeded { effect, version, labels, seeds }) => {
                let n = labels.iter().filter(|&&l| l != 0).count();
                self.seeded = Some(Seeded { effect, version, labels, seeds });
                self.preview = None;
                (format!("{}: {} voxels in its region; Apply keeps the result", effect.label(), polymix_workspace::style::format_count(n)), false)
            }
            Ok(Outcome::Edits { edits, what }) => {
                seg.checkpoint();
                let n: usize = edits.iter().map(|(label, mask, mode)| seg.apply(&vol, mask, *label, *mode, &masking)).sum();
                seg.changed(true);
                (format!("{what}: {} voxels changed", polymix_workspace::style::format_count(n)), false)
            }
            Ok(Outcome::Split { label, slots, islands }) => {
                seg.checkpoint();
                let made = seg.split(&vol, label, &slots, islands);
                (format!("Split into {} segments", made + 1), false)
            }
            Ok(Outcome::Meshes(meshes)) => {
                let n = meshes.len();
                self.outbox.extend(meshes);
                self.outbox_by_agent = agent;
                (format!("{n} mesh(es) sent to the Mesh tab"), false)
            }
            Ok(Outcome::Imported { imported, what }) => {
                seg.checkpoint();
                let mut made = Vec::new();
                for f in &imported.segments {
                    let Some(k) = seg.push_segment(Some(f.name.clone())) else { break };
                    let s = &mut seg.segments[k];
                    if let Some(c) = f.colour {
                        s.colour = c;
                    }
                    if let Some(st) = segmentation::Status::ALL.into_iter().find(|x| x.key() == f.status) {
                        s.status = st;
                    }
                    made.push((f.label, s.label));
                }
                // The file's labels onto the new segments' (over whatever was there).
                let mut map = [0u8; 256];
                for &(from, to) in &made {
                    map[from as usize] = to;
                }
                seg.labels.par_iter_mut().zip(&imported.labels).for_each(|(l, &m)| {
                    if map[m as usize] != 0 {
                        *l = map[m as usize];
                    }
                });
                seg.changed(true);
                let notes = if imported.notes.is_empty() { String::new() } else { format!(" ({})", imported.notes.join("; ")) };
                (format!("{what}: {} segment(s) added{notes}", made.len()), false)
            }
            Ok(Outcome::Volumes { paths }) => {
                let n = paths.len();
                self.volume_outbox.extend(paths);
                (format!("Saved {n} cropped image(s) among the derived images (Volume tab)"), false)
            }
            Ok(Outcome::Centerline(r)) => {
                let n = r.tree.branches.len();
                let narrow = r.narrowest.iter().enumerate().filter_map(|(k, x)| x.map(|(_, p)| (k, p))).max_by(|a, b| a.1.total_cmp(&b.1));
                self.vessels.branch = narrow.map_or(0, |(k, _)| k);
                self.vessels.placing = false;
                self.vessels.result = Some(*r);
                let note = narrow.map_or(String::new(), |(k, p)| format!("; narrowest {p:.0}% (branch {})", k + 1));
                (format!("Centerline: {n} branch(es){note}"), false)
            }
            Ok(Outcome::Stats { version, rows }) => {
                let n = rows.len();
                self.stats = Some((version, rows));
                (format!("Measured {n} segment(s)"), false)
            }
            Ok(Outcome::Volume { path }) => {
                let name = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
                self.volume_outbox.push(path);
                (format!("Saved “{name}” among the derived images (Volume tab)"), false)
            }
            Ok(Outcome::Saved { path, version, style_version }) => {
                if seg.version == version && seg.style_version == style_version {
                    seg.modified = false;
                }
                (format!("Saved {path}"), false)
            }
            Err(e) => (format!("{e:#}"), true),
        };
        self.message = Some(if agent { (format!("Agent: {}", message.0), message.1) } else { message });
    }

    /// A brush dab at `at` (voxel coordinates) in the view of `plane`; the first one of a
    /// stroke makes the undo step.
    /// A brush dab or stroke at `at` (voxel coordinates): a disc in `plane`'s slice, or a ball
    /// with the sphere brush or from the 3D view (`plane` `None`).
    fn brush(&mut self, vol: &LoadedVolume, plane: Option<Plane>, at: DVec3, erase: bool) {
        let (Some(seg), Some(st)) = (&mut self.seg, &self.settings) else { return };
        if self.stroke.is_none() {
            if seg.selected.is_none() && (erase || seg.add_segment(None).is_none()) {
                return;
            }
            seg.checkpoint();
        }
        let Some(label) = seg.selected_segment().map(|s| s.label) else { return };
        let disc = plane.filter(|_| !st.sphere).map(|p| plane_axes(&vol.geometry, p).slice);
        let from = self.stroke.unwrap_or(at);
        // Pixel mode: the voxels under the stroke only.
        let radius = if st.pixel_mode { 0.0 } else { st.brush_mm as f64 };
        // The dab at `from` ended the last step of the stroke; unless something changed since,
        // painting it again would change nothing.
        let key = |at: DVec3, version: u64| DabKey { at, label, erase, radius, disc, masking: self.masking, version };
        let painted = self.stroke.is_some() && self.last_dab == Some(key(from, seg.version));
        let voxels = effects::stroke(vol.dims, vol.geometry.spacing, from, at, radius, disc, painted);
        seg.paint(vol, &voxels, label, erase, &self.masking);
        self.last_dab = Some(key(at, seg.version));
        self.stroke = Some(at);
    }

    /// Scissors drawn in the 3D view: fill or erase the voxels whose centres appear inside (or
    /// outside) `polygon` on the screen, through the whole depth.
    fn cut_3d(&mut self, vol: &LoadedVolume, polygon: &[egui::Pos2], project: impl Fn(DVec3) -> Option<egui::Pos2> + Sync) {
        let Some(op) = self.settings.as_ref().map(|s| s.scissors_op) else { return };
        if polygon.len() < 3 {
            return;
        }
        let outside = matches!(op, effects::ScissorsOp::EraseOutside | effects::ScissorsOp::FillOutside);
        let mask = mm_segment::Mask { dims: vol.dims, data: screen_cut(vol.dims, polygon, outside, project) };
        let (mode, fill) = match op {
            effects::ScissorsOp::FillInside | effects::ScissorsOp::FillOutside => (Mode::Add, true),
            _ => (Mode::Remove, false),
        };
        self.edit_now(vol, mask, mode, &format!("{} (3D)", op.label()), fill);
    }

    /// The 3D view's input for the effect in use: the voxel picked under the pointer while a
    /// stroke is drawn (⌘ / Ctrl + left drag) or on a click.
    fn input_3d(&mut self, vol: &LoadedVolume, effect: Option<Effect>, picked: Option<DVec3>, stroking: bool, clicked: bool, ctx: &egui::Context) {
        if !stroking && self.stroke_3d {
            self.stroke_3d = false;
            self.end_stroke();
        }
        let Some(v) = picked else { return };
        let seed = [0, 1, 2].map(|a| v[a].round().clamp(0.0, vol.dims[a] as f64 - 1.0) as usize);
        match effect {
            Some(e @ (Effect::Paint | Effect::Erase)) => {
                self.stroke_3d = stroking;
                self.brush(vol, None, v, e == Effect::Erase);
                if !stroking {
                    self.end_stroke();
                }
            }
            Some(e @ (Effect::FloodFill | Effect::Islands)) if clicked => {
                if e == Effect::FloodFill || self.settings.as_ref().is_some_and(|s| s.islands.by_click()) {
                    self.start_op(e, Some(seed), ctx);
                }
            }
            Some(e) if e.takes_points() && clicked => self.markers.push(v),
            _ => {}
        }
    }

    fn end_stroke(&mut self) {
        self.last_dab = None;
        if self.stroke.take().is_some() {
            if let Some(seg) = &mut self.seg {
                // The 3D view catches up once per stroke (the slice views and the voxel
                // counts followed every dab).
                seg.stroke_finished();
            }
        }
    }

    /// The slice views' mouse input for the effect in use.
    fn view_input(&mut self, vol: &LoadedVolume, responses: &[(Plane, SliceResponse)], ctx: &egui::Context) {
        // A measuring tool, or placing the centerline's ends, has the left button: no effect
        // works meanwhile.
        self.measure.input(vol, responses);
        if self.vessels.placing {
            if let Some(v) = responses.iter().find_map(|(_, r)| r.clicked_voxel) {
                self.vessels.ends.push(v);
            }
        }
        let effect = self.effect.filter(|_| self.running.is_none() && self.measure.tool.is_none() && !self.vessels.placing);
        if let Some(st) = &mut self.settings {
            for (_, r) in responses {
                if r.brush_wheel != 0.0 {
                    if st.brush_relative {
                        st.brush_percent = (st.brush_percent * 1.1f32.powf(r.brush_wheel)).clamp(0.5, 50.0);
                    } else {
                        st.brush_mm = (st.brush_mm * 1.1f32.powf(r.brush_wheel)).clamp(0.2, 50.0);
                    }
                }
            }
            // A brush relative to the view takes its size from the view under the pointer.
            if st.brush_relative {
                let spacing = vol.geometry.spacing;
                if let Some(map) = responses.iter().find(|(_, r)| r.left_drag_voxel.is_some() || r.hover_voxel.is_some()).map(|(_, r)| r.map) {
                    let side_mm = (map.rect.width().min(map.rect.height()) / map.points_per_mm(spacing).max(1e-6)) as f64;
                    st.brush_mm = (side_mm * st.brush_percent as f64 / 200.0).clamp(0.1, 500.0) as f32;
                }
            }
        }
        match effect {
            // A disc in a slice of the grid; a ball on an oblique plane.
            Some(e @ (Effect::Paint | Effect::Erase)) => match responses.iter().find_map(|(p, r)| r.left_drag_voxel.map(|v| (r.map.oblique.is_none().then_some(*p), v))) {
                Some((plane, v)) => self.brush(vol, plane, v, e == Effect::Erase),
                None if !self.stroke_3d => self.end_stroke(),
                None => {}
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
            Some(e @ (Effect::Draw | Effect::LevelTracing | Effect::Scissors)) => {
                self.end_stroke();
                self.sketch_input(vol, e, responses);
            }
            Some(Effect::Threshold) if self.settings.as_ref().is_some_and(|s| s.local_histogram) => {
                self.end_stroke();
                self.sketch_input(vol, Effect::Threshold, responses);
            }
            Some(e) if e.takes_points() => {
                self.end_stroke();
                if let Some(v) = responses.iter().find_map(|(_, r)| r.clicked_voxel) {
                    self.markers.push(v);
                }
            }
            Some(Effect::Engrave) => {
                self.end_stroke();
                if let Some((map, v)) = responses.iter().filter(|(_, r)| r.map.oblique.is_none()).find_map(|(_, r)| r.clicked_voxel.map(|v| (r.map, v))) {
                    self.engrave(vol, map.axes, v);
                }
            }
            _ => self.end_stroke(),
        }
        match effect {
            Some(e @ (Effect::ThresholdGrow | Effect::SelectPart)) => {
                if let Some((map, (d, pressed))) = responses.iter().find_map(|(_, r)| r.tool_drag.map(|t| (r.map, t))) {
                    self.tool_drag(vol, e, d, pressed, map);
                }
            }
            _ => {
                self.grow = None;
                self.part = None;
            }
        }
        if !matches!(effect, Some(Effect::Draw | Effect::Scissors | Effect::Threshold)) {
            self.sketch = None;
        }
        if self.effect != Some(Effect::Threshold) {
            self.local = None;
        }
        if !effect.is_some_and(|e| e.takes_points()) && self.running.is_none() {
            self.markers.clear();
        }
        if effect != Some(Effect::LevelTracing) {
            self.traced = None;
        }
    }

    /// Threshold grow's or Select part's ⌘ / Ctrl + right drag in a slice view: `pressed` is the
    /// voxel pressed, `map` the view's (its zoom sets how far a point of drag reaches).
    fn tool_drag(&mut self, vol: &LoadedVolume, effect: Effect, d: polymix_imaging::ToolDrag, pressed: Option<DVec3>, map: polymix_imaging::ViewMap) {
        use polymix_imaging::DragPhase;
        let Some(seg) = &mut self.seg else { return };
        if d.phase == DragPhase::Start {
            self.grow = None;
            self.part = None;
            let Some(v) = pressed else {
                self.message = Some(("Press on the image".into(), true));
                return;
            };
            let seed = drag::voxel_index(vol.dims, v);
            let mm_per_point = 1.0 / map.points_per_mm(vol.geometry.spacing).max(1e-6) as f64;
            let started = match effect {
                Effect::ThresholdGrow => {
                    if seg.selected.is_none() && seg.add_segment(None).is_none() {
                        return;
                    }
                    let (Some(label), Some(range)) = (seg.selected_segment().map(|s| s.label), self.settings.as_ref().map(|s| s.threshold)) else { return };
                    drag::Grow::start(vol, seg, seed, label, range, mm_per_point).map(|g| self.grow = Some(g))
                }
                _ => {
                    self.previews_made += 1;
                    drag::Part::start(vol, seg, seed, mm_per_point, self.previews_made).map(|p| self.part = Some(p))
                }
            };
            if let Err(e) = started {
                self.message = Some((e, true));
                return;
            }
        }
        if let Some(g) = &mut self.grow {
            let n = g.drag(vol, seg, d.total.x, &self.masking);
            self.message = Some((format!("Grow {:.1} mm · {n} voxels", g.reach), false));
            if d.phase == DragPhase::End {
                self.grow = None;
                seg.stroke_finished();
            }
        }
        if let Some(p) = &mut self.part {
            if !p.done {
                p.drag(seg, d.total.x);
                p.done = d.phase == DragPhase::End;
            }
        }
    }

    /// Do `action` to Select part's part, and drop it.
    fn use_part(&mut self, vol: &LoadedVolume, action: drag::PartAction) {
        let (Some(seg), Some(part)) = (&mut self.seg, self.part.take()) else { return };
        let message = drag::use_part(&part, vol, seg, action);
        self.message = Some((message, false));
    }

    /// Make the preview (the segmentation as Apply would leave it): Threshold's, or a seeded
    /// effect's result; `None` otherwise.
    fn update_preview(&mut self, vol: &LoadedVolume) {
        let (Some(seg), Some(st)) = (&self.seg, &self.settings) else { return };
        let effect = self.effect.filter(|_| self.running.is_none());
        if !effect.is_some_and(|e| e.seeded()) {
            self.seeded = None;
        }
        if let (Some(e), Some(sd)) = (effect, &self.seeded) {
            if e == sd.effect {
                let key = PreviewKey { version: seg.version, range: (f32::NAN, f32::NAN), masking: self.masking, label: 0 };
                if self.preview.as_ref().is_none_or(|(k, _, _)| k.version != key.version || k.masking != key.masking || !k.range.0.is_nan()) {
                    // The last preview's buffer is filled again.
                    let mut labels = self.preview.take().map(|p| p.2).unwrap_or_default();
                    seg.seeded_preview(vol, &sd.labels, &sd.seeds, &self.masking, &mut labels);
                    self.previews_made += 1;
                    self.preview = Some((key, self.previews_made, labels));
                }
                return;
            }
        }
        let label = seg.selected_segment().map(|s| s.label).filter(|_| effect == Some(Effect::Threshold));
        let Some(label) = label else {
            self.preview = None;
            return;
        };
        let key = PreviewKey { version: seg.version, range: st.threshold, masking: self.masking, label };
        if self.preview.as_ref().is_none_or(|(k, _, _)| *k != key) {
            // Every frame of a drag over the range: one pass into the last preview's buffer.
            let mut labels = self.preview.take().map(|p| p.2).unwrap_or_default();
            seg.threshold_preview(vol, st.threshold, label, &self.masking, &mut labels);
            self.previews_made += 1;
            self.preview = Some((key, self.previews_made, labels));
        }
    }

    /// The label map with only the visible segments in it (the seeds of Grow from seeds and
    /// Watershed), and its version.
    fn seed_labels(&self) -> Option<(Vec<u8>, u64)> {
        let seg = self.seg.as_ref()?;
        let mut visible = [false; 256];
        for g in seg.segments.iter().filter(|g| g.visible) {
            visible[g.label as usize] = true;
        }
        Some((seg.labels.par_iter().map(|&l| if visible[l as usize] { l } else { 0 }).collect(), seg.version))
    }

    /// Start Grow from seeds or Watershed from the visible segments.
    fn start_seeded(&mut self, effect: Effect, ctx: &egui::Context) {
        let (None, Some(vol), Some(st), Some((labels, version))) = (&self.running, &self.volume, &self.settings, self.seed_labels()) else { return };
        let (s, vol) = (st.clone(), vol.clone());
        let job = Job::spawn(effect.label(), ctx.clone(), move |_| effects::run_seeded(effect, &s, &vol, labels, version));
        self.running = Some(Running { what: effect.label().to_string(), job, bypass: false });
    }

    /// Keep the seeded effect's result: each seed segment takes the voxels it was given.
    fn apply_seeded(&mut self) {
        let (Some(vol), Some(seg), Some(sd)) = (self.volume.clone(), &mut self.seg, self.seeded.take()) else { return };
        seg.checkpoint();
        let n = seg.apply_seeded(&vol, &sd.labels, &sd.seeds, &self.masking);
        seg.changed(true);
        self.preview = None;
        self.say(format!("{}: {} voxels changed", sd.effect.label(), polymix_workspace::style::format_count(n)), false);
    }

    /// Re-run a seeded effect after its seeds were edited (Auto-update).
    fn auto_update(&mut self, ctx: &egui::Context) {
        let (Some(e), Some(seg), Some(st)) = (self.effect, &self.seg, &self.settings) else { return };
        if self.running.is_none() && st.auto_update && self.seeded.as_ref().is_some_and(|sd| sd.effect == e && sd.version != seg.version) && self.stroke.is_none() {
            self.start_seeded(e, ctx);
        }
    }

    /// Draw tube or Surface cut from the points clicked.
    /// Guided vessel: the vessel through the points clicked, from the first to the last (in the
    /// background).
    fn start_guided(&mut self, ctx: &egui::Context) {
        let (None, Some(vol), Some(seg), Some(st)) = (&self.running, self.volume.clone(), &mut self.seg, &self.settings) else { return };
        if self.markers.is_empty() {
            self.say("Guided vessel: click points along the vessel first", true);
            return;
        }
        if seg.selected.is_none() && seg.add_segment(None).is_none() {
            return;
        }
        let label = seg.selected_segment().map_or(1, |s| s.label);
        let points: Vec<[f64; 3]> = self.markers.iter().map(|p| p.to_array()).collect();
        let (range, radius, params) = (st.threshold, st.tube_mm as f64, st.level_set_params());
        let job = Job::spawn("Guided vessel", ctx.clone(), move |progress| {
            let mask = mm_segment::guided_vessel(&vol, &points, range, radius, &params, progress)?;
            if mask.count() == 0 {
                anyhow::bail!("nothing in the threshold range along the points: set the range to the vessel's values");
            }
            Ok(Outcome::Edit { label, mask, mode: Mode::Add, what: "Guided vessel".into() })
        });
        self.running = Some(Running { what: "Guided vessel".into(), job, bypass: false });
        self.markers.clear();
    }

    fn apply_markers(&mut self, effect: Effect) {
        let (Some(vol), Some(st)) = (self.volume.clone(), &self.settings) else { return };
        let (dims, spacing) = (vol.dims, vol.geometry.spacing);
        let mask = match effect {
            Effect::DrawTube if !self.markers.is_empty() => mm_segment::shapes::tube(dims, spacing, &self.markers, st.tube_mm as f64),
            Effect::SurfaceCut if self.markers.len() >= 4 => mm_segment::shapes::hull(dims, spacing, &self.markers),
            _ => {
                self.say(if effect == Effect::DrawTube { "Draw tube: click points along the structure first" } else { "Surface cut: click at least four points around the structure first" }, true);
                return;
            }
        };
        if mask.count() == 0 {
            self.say("The points span no volume: place them around the structure, not in one plane", true);
            return;
        }
        self.edit_now(&vol, mask, Mode::Add, effect.label(), true);
        self.markers.clear();
    }

    fn make_mesh(&mut self, ctx: &egui::Context) {
        let (None, Some(vol), Some(seg), Some(st)) = (&self.running, &self.volume, &self.seg, &self.settings) else { return };
        let chosen: Vec<(String, u8)> = seg.segments.iter().enumerate().filter(|(k, s)| if st.mesh_all_visible { s.visible } else { seg.selected == Some(*k) }).map(|(_, s)| (s.name.clone(), s.label)).collect();
        if chosen.is_empty() {
            self.say("Select a segment first", true);
            return;
        }
        let (geometry, smoothing, labels, dims) = (vol.geometry.clone(), st.mesh_smoothing_mm as f64, seg.labels.clone(), seg.dims);
        let job = Job::spawn("Making meshes", ctx.clone(), move |progress| Ok(Outcome::Meshes(effects::meshes(&labels, dims, &chosen, &geometry, smoothing, progress)?)));
        self.running = Some(Running { what: "Making meshes".into(), job, bypass: false });
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
        let (labels, dims, geometry, version, style_version) = (seg.labels.clone(), seg.dims, vol.geometry.clone(), seg.version, seg.style_version);
        let infos: Vec<SegmentInfo> = seg.segments.iter().map(|s| SegmentInfo { name: s.name.clone(), colour: [s.colour.r(), s.colour.g(), s.colour.b()], label: s.label, status: s.status.key().to_string() }).collect();
        let job = Job::spawn("Saving", ctx.clone(), move |_| {
            let name = path.file_name().map(|n| n.to_string_lossy().to_lowercase()).unwrap_or_default();
            if name.ends_with(".nii") || name.ends_with(".nii.gz") {
                let description = infos.iter().map(|s| format!("{}={}", s.label, s.name)).collect::<Vec<_>>().join(" ");
                mm_segment::write_nifti(&path, &labels, dims, &geometry, &description)?;
            } else {
                mm_segment::write_seg_nrrd(&path, &labels, dims, &geometry, &infos)?;
            }
            Ok(Outcome::Saved { path: path.display().to_string(), version, style_version })
        });
        self.running = Some(Running { what: "Saving".into(), job, bypass: false });
    }

    /// The keys, as in 3D Slicer's Segment Editor: ⌘Z / ⇧⌘Z undo and redo; Esc drops a shape
    /// in progress, then the effect; ← → ↑ ↓ the slice under the pointer; i intensity
    /// masking on / off (the threshold range); 1–0 the first ten effects, ⇧1–⇧0 the next;
    /// Space the effect used before; − / + the brush; q / w the previous / next segment; in
    /// Draw, Enter fills and x takes back a point.
    fn keys(&mut self, ctx: &egui::Context) {
        if self.effect != self.effects_seen.1 {
            self.effects_seen = (self.effects_seen.1, self.effect);
        }
        if ctx.egui_wants_keyboard_input() || self.running.is_some() {
            return;
        }
        use egui::{Key, Modifiers};
        let drawing = self.effect == Some(Effect::Draw);
        let numbers = [Key::Num1, Key::Num2, Key::Num3, Key::Num4, Key::Num5, Key::Num6, Key::Num7, Key::Num8, Key::Num9, Key::Num0];
        struct Pressed {
            redo: bool,
            undo: bool,
            escape: bool,
            enter: bool,
            back: bool,
            slices: i64,
            intensity: bool,
            toggle: bool,
            space: bool,
            brush: i32,
            segment: i32,
            effect: Option<usize>,
        }
        let k = ctx.input_mut(|i| {
            let mut p = Pressed {
                redo: i.consume_key(Modifiers::COMMAND | Modifiers::SHIFT, Key::Z),
                undo: i.consume_key(Modifiers::COMMAND, Key::Z),
                escape: i.consume_key(Modifiers::NONE, Key::Escape),
                enter: drawing && i.consume_key(Modifiers::NONE, Key::Enter),
                back: drawing && i.consume_key(Modifiers::NONE, Key::X),
                slices: 0,
                intensity: i.consume_key(Modifiers::NONE, Key::I),
                toggle: i.consume_key(Modifiers::NONE, Key::T),
                space: i.consume_key(Modifiers::NONE, Key::Space),
                brush: 0,
                segment: 0,
                effect: None,
            };
            for (key, d) in [(Key::ArrowUp, 1), (Key::ArrowRight, 1), (Key::ArrowDown, -1), (Key::ArrowLeft, -1)] {
                if i.consume_key(Modifiers::NONE, key) {
                    p.slices += d;
                }
            }
            for (key, d) in [(Key::Minus, -1), (Key::Plus, 1), (Key::Equals, 1)] {
                if i.consume_key(Modifiers::NONE, key) {
                    p.brush += d;
                }
            }
            for (key, d) in [(Key::Q, -1), (Key::W, 1)] {
                if i.consume_key(Modifiers::NONE, key) {
                    p.segment += d;
                }
            }
            // (Shifted first: a plain key also matches with Shift held.)
            for (n, &key) in numbers.iter().enumerate() {
                if i.consume_key(Modifiers::SHIFT, key) {
                    p.effect = Some(10 + n);
                } else if i.consume_key(Modifiers::NONE, key) {
                    p.effect = Some(n);
                }
            }
            p
        });
        if let Some(seg) = &mut self.seg {
            if k.redo {
                seg.redo();
            } else if k.undo {
                seg.undo();
            }
            if k.segment != 0 && !seg.segments.is_empty() {
                let n = seg.segments.len() as i32;
                let at = seg.selected.map_or(if k.segment > 0 { -1 } else { n }, |s| s as i32);
                seg.selected = Some((at + k.segment).clamp(0, n - 1) as usize);
            }
        }
        if k.enter {
            if let Some(vol) = self.volume.clone() {
                self.finish_draw(&vol);
            }
        }
        if k.back {
            if let Some(s) = &mut self.sketch {
                s.points.pop();
            }
        }
        // The slice of the view under the pointer, towards the patient's S / A / R for ↑ →.
        if let (true, Some((plane, _)), Some(vol), Some(state)) = (k.slices != 0, self.hover, &self.volume, &mut self.state) {
            let (axes, step) = sketch::slicing(vol, plane);
            let n = vol.dims[axes.slice] as f64;
            state.cursor[axes.slice] = (state.cursor[axes.slice].round() + (k.slices * step) as f64).clamp(0.0, n - 1.0);
        }
        // T swaps the images compared (the second over the first, or back).
        if let Some(f) = self.state.as_mut().and_then(|s| s.fusion.as_mut()).filter(|_| k.toggle) {
            match f.compare {
                polymix_imaging::Compare::Toggle => f.showing = !f.showing,
                _ => f.opacity = if f.opacity > 0.0 { 0.0 } else { 1.0 },
            }
        }
        if k.intensity {
            self.masking.intensity = match self.masking.intensity {
                Some(_) => None,
                None => self.settings.as_ref().map(|s| s.threshold),
            };
            self.say(if self.masking.intensity.is_some() { "Intensity masking on (the threshold range)" } else { "Intensity masking off" }, false);
        }
        if let Some(st) = &mut self.settings {
            if k.brush != 0 {
                st.brush_mm = (st.brush_mm * 1.1f32.powi(k.brush)).clamp(0.2, 50.0);
            }
        }
        if let Some(e) = k.effect.and_then(|n| Effect::ALL.get(n)) {
            self.effect = Some(*e);
        }
        if k.space {
            self.effect = self.effects_seen.0;
        }
        // Esc drops a shape or points in progress first, then puts the measuring tool or the
        // effect away.
        if k.escape && self.vessels.placing {
            self.vessels.placing = false;
        } else if k.escape && self.measure.tool.is_some() {
            if !self.measure.cancel_pending() {
                self.measure.tool = None;
            }
        } else if k.escape && self.part.take().is_some() {
            // The part selected first, then the effect.
        } else if k.escape && self.sketch.take().is_none() {
            if self.markers.is_empty() {
                self.effect = None;
            }
            self.markers.clear();
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
            "seg.load" => self.load_segmentation(PathBuf::from(arg), ctx),
            "seg.measure" => self.measure(ctx),
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
                    // seg.hollow:<mm>[,outside|medial|inside]
                    "seg.hollow" => {
                        let (mm, side) = arg.split_once(',').unwrap_or((arg, ""));
                        st.hollow_mm = mm.parse().unwrap_or(st.hollow_mm);
                        if let Some(h) = effects::HOLLOW_SIDES.iter().find(|h| h.2 == side) {
                            st.hollow_side = h.0;
                        }
                        self.effect = Some(Effect::Hollow);
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
        // The Overlay menu asks for its list as it opens (not every frame it is open).
        let was_open = std::mem::take(&mut self.overlay_menu_open);
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
            polymix_imaging::layers::display_controls(ui, "segmentation", state, v);
            let mut pick = None;
            let menu = ui.menu_button("Overlay ▾", |ui| {
                if self.overlay_choices.is_empty() {
                    ui.weak("No other image overlaps this one (images are listed in the Volume tab)");
                }
                egui::ScrollArea::vertical().max_height(320.0).show(ui, |ui| {
                    for (k, (_, label)) in self.overlay_choices.iter().enumerate() {
                        if ui.button(label).clicked() {
                            pick = Some(k);
                            ui.close();
                        }
                    }
                });
            });
            if menu.inner.is_some() {
                self.overlay_menu_open = true;
                if !was_open {
                    self.wants_choices = true;
                }
            }
            menu.response.on_hover_text("Show another image over this one (fusion), or compare them (wipe, toggle): e.g. vesselness over the angiogram, or before / after N4");
            if self.fusion_job.is_some() {
                polymix_workspace::spinner(ui);
            }
            if let Some(k) = pick {
                let vol = v.clone();
                self.start_overlay(vol, k, ui.ctx());
            }
            ui.separator();
            if ui.button("Fit").on_hover_text("Fit every view to the image (zoom and pan back to the start)").clicked() {
                for view in &mut self.views {
                    view.nav.reset();
                }
                self.view3d.reset_camera();
            }
            egui::ComboBox::from_id_salt("segmentation.layout").selected_text(self.layout.label()).width(110.0).show_ui(ui, |ui| {
                for l in polymix_imaging::Layout::ALL {
                    if ui.selectable_value(&mut self.layout, l, l.label()).clicked() {
                        self.maximized = None;
                    }
                }
            });
            ui.toggle_value(&mut self.crosshair, "Crosshair").on_hover_text("Show where the other two planes cut each view");
            ui.separator();
            self.measure.tool_buttons(ui);
            ui.toggle_value(&mut self.smooth, "Smooth").on_hover_text("Interpolate between voxels (off: each voxel is a sharp square)");
        });
        if let Some(f) = self.state.as_mut().and_then(|s| s.fusion.as_mut()) {
            if !polymix_imaging::layers::fusion_controls(ui, "segmentation", f) {
                self.state.as_mut().expect("checked").fusion = None;
            }
        }
    }

    /// Load overlay choice `k` (in the background) to show over `vol`.
    fn start_overlay(&mut self, vol: Arc<LoadedVolume>, k: usize, ctx: &egui::Context) {
        let Some((other, label)) = self.overlay_choices.get(k).cloned() else { return };
        let cancel = Arc::new(AtomicBool::new(false));
        let c = cancel.clone();
        // Replacing the load in progress (another choice) stops it.
        let job = Job::spawn("Loading the overlay", ctx.clone(), move |progress| {
            // (The loader reports from several threads: no progress for it, but it stops when
            // the load is no longer wanted.)
            let o = load_volume(&other, MAX_VOXELS, &|_| !c.load(Ordering::Relaxed))?;
            progress(0.8);
            let values = polymix_imaging::Fusion::resampled(&vol, &o);
            Ok(polymix_imaging::Fusion::new(label, values, polymix_imaging::ColourMap::Hot))
        });
        self.fusion_job = Some(OverlayLoad { job, cancel });
    }

    /// The image shown, with the crosshair and window: for the image slices in the Mesh tab.
    pub fn view_for_mesh(&self) -> Option<ShownImage> {
        let (v, st, src) = (self.volume.as_ref()?, self.state.as_ref()?, self.source.as_ref()?);
        Some(ShownImage { volume: v.clone(), name: src.label.clone(), cursor: st.cursor.to_array(), window: st.window })
    }

    /// Whether the Overlay menu asked for the images that can be shown over this one.
    pub fn wants_overlay_choices(&mut self) -> bool {
        std::mem::take(&mut self.wants_choices)
    }

    pub fn set_overlay_choices(&mut self, choices: Vec<(Volume, String)>) {
        self.overlay_choices = choices;
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
        // Axial, 3D, sagittal and coronal as the layout places them; or one of them filling
        // the area.
        self.rects = match self.maximized {
            Some(m) => std::array::from_fn(|i| if i == m { full } else { Rect::NOTHING }),
            None => self.layout.rects(full, 2.0),
        };
        let Some(vol) = self.volume.clone().filter(|_| self.state.is_some()) else {
            ui.allocate_rect(full, egui::Sense::hover());
            self.empty_message(ui, full);
            return;
        };
        let ctx = ui.ctx().clone();
        // A part an edit has made stale (undo, another effect) is dropped.
        if self.part.as_ref().is_some_and(|p| self.seg.as_ref().is_none_or(|s| s.version != p.seg_version)) {
            self.part = None;
        }
        let mut colours = self.seg.as_ref().map(|s| s.colours()).unwrap_or_default();
        if let (Some(p), true) = (&self.part, colours.len() == 256) {
            colours[p.mark as usize] = drag::PART_COLOUR;
        }
        let has_segments = self.seg.as_ref().is_some_and(|s| !s.segments.is_empty());
        let effect = self.effect.filter(|_| self.running.is_none() && self.measure.tool.is_none() && !self.vessels.placing);
        let opts = SliceOptions {
            left_drag_scrolls: self.measure.tool.is_none() && !self.vessels.placing && !effect.is_some_and(|e| self.settings.as_ref().is_some_and(|s| s.takes_left_button(e))),
            crosshair: self.crosshair,
            smooth: self.smooth,
            brush_mm: matches!(effect, Some(Effect::Paint | Effect::Erase)).then(|| self.settings.as_ref().map_or(5.0, |s| if s.pixel_mode { 0.3 } else { s.brush_mm })),
            tool_secondary_drag: matches!(effect, Some(Effect::ThresholdGrow | Effect::SelectPart)),
        };
        // Slice views: the labels (or the threshold preview) under a version of their own.
        self.update_preview(&vol);
        let style = self.seg.as_ref().map_or(0, |s| s.style_version);
        let shown: Option<(OverlayKey, &[u8])> = match (&self.part, &self.preview, &self.seg) {
            (Some(p), _, _) => Some((OverlayKey::Part(p.id, p.version, style), &p.overlay)),
            (None, Some((_, id, labels)), _) => Some((OverlayKey::Preview(*id, style), labels)),
            (None, None, Some(seg)) => Some((OverlayKey::Labels(seg.version, style), &seg.labels)),
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
        self.auto_update(&ctx);
        let maps: Vec<polymix_imaging::ViewMap> = responses.iter().map(|(_, r)| r.map).collect();
        self.paint_sketches(ui, ctx.pointer_hover_pos(), &maps);
        self.measure.paint(ui, &vol, &maps, ctx.pointer_hover_pos());
        // 3D: the finished segments (not every brush dab).
        let state = self.state.as_mut().expect("checked above");
        let labels3d = self.seg.as_ref().filter(|_| has_segments).map(|s| LabelOverlay { labels: &s.labels, colours: &colours, opacity: 1.0, version: s.version_3d });
        let view3d = &mut self.view3d;
        // Effects in the 3D view: a click picks the voxel under the pointer; ⌘ / Ctrl + left
        // drag paints (plain drags turn the camera).
        let effect = self.effect.filter(|_| self.running.is_none() && self.measure.tool.is_none() && !self.vessels.placing);
        let paints = matches!(effect, Some(Effect::Paint | Effect::Erase));
        let scissors = effect == Some(Effect::Scissors);
        view3d.tool_drags = paints || scissors;
        let (mut picked, mut stroking, mut clicked) = (None, false, false);
        let mut cut_3d = None;
        if self.rects[1].is_positive() {
            let r3 = ui.scope_builder(egui::UiBuilder::new().max_rect(self.rects[1]), |ui| view3d.ui(ui, render_state, &vol, state, labels3d.as_ref())).inner;
            if r3.double_clicked() {
                toggle = Some(1);
            }
            let command = ui.input(|i| i.modifiers.command);
            stroking = paints && command && r3.dragged_by(egui::PointerButton::Primary);
            clicked = r3.clicked() && !r3.double_clicked();
            if stroking || clicked {
                picked = r3.interact_pointer_pos().and_then(|p| view3d.pick(&vol, state, labels3d.as_ref(), r3.rect, p));
            }
            // Scissors in 3D: ⌘ / Ctrl + drag draws the shape; letting go cuts through the
            // whole depth along the view.
            if scissors {
                let shape = self.settings.as_ref().map_or(effects::ScissorsShape::FreeForm, |s| s.scissors_shape);
                match (command && r3.dragged_by(egui::PointerButton::Primary), r3.interact_pointer_pos()) {
                    (true, Some(p)) => {
                        let (pts, start) = self.scissors_3d.get_or_insert_with(|| (Vec::new(), p));
                        let a = *start;
                        match shape {
                            effects::ScissorsShape::FreeForm => {
                                if pts.last().is_none_or(|q| q.distance(p) >= 2.0) {
                                    pts.push(p);
                                }
                            }
                            effects::ScissorsShape::Rectangle => *pts = vec![a, egui::pos2(p.x, a.y), p, egui::pos2(a.x, p.y)],
                            effects::ScissorsShape::Circle => {
                                let r = a.distance(p);
                                *pts = (0..64).map(|k| a + egui::Vec2::angled(k as f32 / 64.0 * std::f32::consts::TAU) * r).collect();
                            }
                        }
                    }
                    _ => {
                        if let Some((pts, _)) = self.scissors_3d.take() {
                            cut_3d = Some((pts, view3d.projector(&vol, r3.rect)));
                        }
                    }
                }
                if let Some((pts, _)) = &self.scissors_3d {
                    ui.painter_at(r3.rect).add(egui::Shape::closed_line(pts.clone(), egui::Stroke::new(1.5, Color32::from_rgb(255, 230, 120))));
                }
            }
        }
        if let Some((polygon, project)) = cut_3d {
            self.cut_3d(&vol, &polygon, project);
        }
        self.input_3d(&vol, effect, picked, stroking, clicked, &ctx);
        // The centerline over the slice views and the 3D view (drawn after it).
        self.paint_vessels(ui, &maps, self.rects[1].is_positive().then_some(self.rects[1]));
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
        if let Some(r) = self.fusion_job.as_ref().and_then(|j| j.job.poll()) {
            self.fusion_job = None;
            match r {
                Ok(f) => {
                    let name = f.name.clone();
                    if let Some(s) = &mut self.state {
                        s.fusion = Some(f);
                    }
                    self.say(format!("Showing {name} over the image"), false);
                }
                Err(e) => self.say(format!("Overlay: {e:#}"), true),
            }
        }
        if let Some(r) = self.running.as_ref().and_then(|r| r.job.poll()) {
            let bypass = self.running.take().is_some_and(|r| r.bypass);
            self.finish_op(r, bypass);
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
        self.script.is_empty() && self.loading.is_none() && self.running.is_none() && self.fusion_job.is_none()
    }

    fn agent_tools() -> Vec<polymix_workspace::agent::AgentTool> {
        agent::tools()
    }

    fn agent_call(&mut self, tool: &str, args: &serde_json::Value, ctx: &egui::Context) -> polymix_workspace::agent::AgentReply {
        agent::call(self, tool, args, ctx)
    }

    fn agent_actions() -> Vec<polymix_workspace::agent::Action> {
        agent::actions()
    }

    fn agent_actions_about() -> &'static str {
        agent::ABOUT
    }

    fn agent_act(&mut self, action: &str, args: &serde_json::Value, ctx: &egui::Context) -> polymix_workspace::agent::AgentReply {
        agent::act(self, action, args, ctx)
    }
}

/// The voxels of a `dims` grid whose centres `project` puts inside `polygon` on the screen (or
/// outside it, or behind the camera, when `outside`): 1, the others 0.
fn screen_cut(dims: [usize; 3], polygon: &[egui::Pos2], outside: bool, project: impl Fn(DVec3) -> Option<egui::Pos2> + Sync) -> Vec<u8> {
    let [nx, ny, nz] = dims;
    let shape = ScreenPolygon::new(polygon);
    let mut data = vec![0u8; nx * ny * nz];
    data.par_chunks_mut(nx.max(1)).enumerate().for_each(|(r, row)| {
        let (j, k) = ((r % ny) as f64, (r / ny) as f64);
        // A row of voxels appears on the screen along the straight line between its ends (when
        // both are in front of the camera). With both ends a point or more past one side of
        // the shape's box, none of it can be inside.
        let ends = (project(DVec3::new(0.0, j, k)), project(DVec3::new((nx - 1) as f64, j, k)));
        let missed = ends.0.zip(ends.1).is_some_and(|(a, b)| shape.both_outside(a, b, 1.0));
        for (i, out) in row.iter_mut().enumerate() {
            let hit = !missed && project(DVec3::new(i as f64, j, k)).is_some_and(|p| shape.contains(p));
            *out = (hit != outside) as u8;
        }
    });
    data
}

/// A shape drawn on the screen, for telling whether points are inside it by the even-odd
/// rule. Its edges are sorted into bands across it, so that a point is tested against the few
/// edges crossing its band rather than against every edge; the others could not cross the
/// ray from it, so the answer is the same.
struct ScreenPolygon {
    lo: egui::Pos2,
    hi: egui::Pos2,
    /// Bands per point down the screen (1 unless the shape is very tall), and the last band.
    scale: f32,
    last: usize,
    /// The edges (a, b) crossing band `k` are `edges[starts[k]..starts[k + 1]]`.
    starts: Vec<usize>,
    edges: Vec<(egui::Pos2, egui::Pos2)>,
}

impl ScreenPolygon {
    /// Bands at most: a shape taller than this many points has bands of more than a point.
    const MAX_BANDS: usize = 1 << 14;

    fn new(polygon: &[egui::Pos2]) -> Self {
        let (lo, hi) = polygon.iter().fold((egui::pos2(f32::INFINITY, f32::INFINITY), egui::pos2(f32::NEG_INFINITY, f32::NEG_INFINITY)), |(a, b), p| (a.min(*p), b.max(*p)));
        let scale = if hi.y - lo.y > (Self::MAX_BANDS - 1) as f32 { (Self::MAX_BANDS - 1) as f32 / (hi.y - lo.y) } else { 1.0 };
        let last = ((hi.y - lo.y) * scale) as usize;
        let mut shape = Self { lo, hi, scale, last, starts: Vec::new(), edges: Vec::new() };
        let bands = last + 1;
        // Edge (polygon[i], polygon[j]) as the even-odd test pairs them; a level edge never
        // crosses the ray.
        let pairs = || (0..polygon.len()).map(|i| (polygon[i], polygon[(i + polygon.len() - 1) % polygon.len()])).filter(|(a, b)| a.y != b.y);
        let span = |(a, b): (egui::Pos2, egui::Pos2)| shape.band(a.y.min(b.y))..=shape.band(a.y.max(b.y));
        let mut starts = vec![0usize; bands + 1];
        for e in pairs() {
            for k in span(e) {
                starts[k + 1] += 1;
            }
        }
        for k in 0..bands {
            starts[k + 1] += starts[k];
        }
        let mut next = starts.clone();
        let mut edges = vec![(egui::Pos2::ZERO, egui::Pos2::ZERO); starts[bands]];
        for e in pairs() {
            for k in span(e) {
                edges[next[k]] = e;
                next[k] += 1;
            }
        }
        shape.starts = starts;
        shape.edges = edges;
        shape
    }

    /// The band of height `y` (within the shape's box). It never decreases as `y` grows, so
    /// an edge from `y0` to `y1` is in the band of every height between.
    fn band(&self, y: f32) -> usize {
        (((y - self.lo.y) * self.scale) as usize).min(self.last)
    }

    /// Both points are at least `margin` past the same side of the shape's box: so is the
    /// line between them.
    fn both_outside(&self, a: egui::Pos2, b: egui::Pos2, margin: f32) -> bool {
        (a.x < self.lo.x - margin && b.x < self.lo.x - margin) || (a.x > self.hi.x + margin && b.x > self.hi.x + margin) || (a.y < self.lo.y - margin && b.y < self.lo.y - margin) || (a.y > self.hi.y + margin && b.y > self.hi.y + margin)
    }

    fn contains(&self, p: egui::Pos2) -> bool {
        if p.x < self.lo.x || p.y < self.lo.y || p.x > self.hi.x || p.y > self.hi.y {
            return false;
        }
        let k = self.band(p.y);
        let mut c = false;
        for &(a, b) in &self.edges[self.starts[k]..self.starts[k + 1]] {
            if (a.y > p.y) != (b.y > p.y) && p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x {
                c = !c;
            }
        }
        c
    }
}

/// Segment statistics as CSV (a header row, then a row per segment still there).
fn stats_csv(rows: &[(u8, mm_segment::SegmentStats)], seg: &Segmentation) -> String {
    let mut out = String::from("segment,label,voxels,volume_ml,surface_cm2,mean,sd,median,min,max,max_diameter_mm,roundness,elongation,principal_1_mm,principal_2_mm,principal_3_mm,centre_r_mm,centre_a_mm,centre_s_mm\n");
    for (label, s) in rows {
        let Some(g) = seg.segments.iter().find(|g| g.label == *label) else { continue };
        let name = if g.name.contains([',', '"']) { format!("\"{}\"", g.name.replace('"', "\"\"")) } else { g.name.clone() };
        let [p1, p2, p3] = s.principal_mm;
        let [r, a, z] = s.centroid_ras;
        out += &format!(
            "{name},{label},{},{:.4},{:.4},{},{},{},{},{},{:.3},{:.4},{:.4},{p1:.3},{p2:.3},{p3:.3},{r:.3},{a:.3},{z:.3}\n",
            s.voxels,
            s.volume_mm3 / 1000.0,
            s.surface_mm2 / 100.0,
            s.mean,
            s.sd,
            s.median,
            s.min,
            s.max,
            s.feret_mm,
            s.roundness,
            s.elongation,
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xorshift(seed: u64) -> impl FnMut() -> f64 {
        let mut state = seed | 1;
        move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// The even-odd test against every edge, as the 3D scissors made it.
    fn inside_every_edge(polygon: &[egui::Pos2], p: egui::Pos2) -> bool {
        let (lo, hi) = polygon.iter().fold((egui::pos2(f32::INFINITY, f32::INFINITY), egui::pos2(f32::NEG_INFINITY, f32::NEG_INFINITY)), |(a, b), p| (a.min(*p), b.max(*p)));
        if p.x < lo.x || p.y < lo.y || p.x > hi.x || p.y > hi.y {
            return false;
        }
        let mut c = false;
        let mut j = polygon.len() - 1;
        for i in 0..polygon.len() {
            let (a, b) = (polygon[i], polygon[j]);
            if (a.y > p.y) != (b.y > p.y) && p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x {
                c = !c;
            }
            j = i;
        }
        c
    }

    #[test]
    fn banded_edges_tell_inside_as_every_edge_does() {
        let mut next = xorshift(17);
        let mut shapes: Vec<Vec<egui::Pos2>> = Vec::new();
        // Free-form loops as dragged, a circle, a rectangle, a self-crossing star, repeated
        // points and level edges, a shape taller than the bands allow at a point each, and a
        // tiny one.
        for k in 0..12 {
            let (c, r) = (egui::pos2(300.0 + next() as f32 * 200.0, 250.0 + next() as f32 * 200.0), 10.0 + next() as f32 * 250.0);
            let n = 20 + (next() * 600.0) as usize;
            shapes.push(
                (0..n)
                    .map(|i| {
                        let t = i as f32 / n as f32 * std::f32::consts::TAU;
                        let wobble = 1.0 + 0.3 * ((t * (k + 2) as f32).sin() + next() as f32 * 0.2);
                        c + egui::Vec2::angled(t) * r * wobble
                    })
                    .collect(),
            );
        }
        shapes.push((0..64).map(|k| egui::pos2(400.0, 300.0) + egui::Vec2::angled(k as f32 / 64.0 * std::f32::consts::TAU) * 123.4).collect());
        shapes.push(vec![egui::pos2(100.5, 80.25), egui::pos2(700.0, 80.25), egui::pos2(700.0, 500.75), egui::pos2(100.5, 500.75)]);
        shapes.push((0..10).map(|k| egui::pos2(400.0, 300.0) + egui::Vec2::angled(k as f32 * 0.8 * std::f32::consts::TAU) * 200.0).collect());
        shapes.push(vec![egui::pos2(10.0, 10.0), egui::pos2(10.0, 10.0), egui::pos2(50.0, 10.0), egui::pos2(50.0, 40.0), egui::pos2(30.0, 40.0), egui::pos2(30.0, 20.0), egui::pos2(10.0, 40.0)]);
        shapes.push(vec![egui::pos2(0.0, -30_000.0), egui::pos2(40.0, 25_000.0), egui::pos2(-35.0, 2.0)]);
        shapes.push(vec![egui::pos2(5.0, 5.0), egui::pos2(6.0, 5.5), egui::pos2(5.5, 5.9)]);
        for (s, polygon) in shapes.iter().enumerate() {
            let shape = ScreenPolygon::new(polygon);
            let (lo, hi) = (shape.lo, shape.hi);
            let mut points: Vec<egui::Pos2> = (0..20_000).map(|_| egui::pos2(lo.x - 5.0 + next() as f32 * (hi.x - lo.x + 10.0), lo.y - 5.0 + next() as f32 * (hi.y - lo.y + 10.0))).collect();
            // On the vertices, at the vertices' heights, on whole and half points.
            points.extend(polygon.iter().copied());
            points.extend(polygon.iter().map(|p| egui::pos2(p.x - 0.5, p.y)));
            points.extend(polygon.iter().map(|p| egui::pos2(p.x.round() + 0.5, p.y.round())));
            points.extend([egui::pos2(f32::NAN, lo.y + 1.0), egui::pos2(lo.x + 1.0, f32::NAN)]);
            for p in points {
                assert_eq!(shape.contains(p), inside_every_edge(polygon, p), "shape {s} at {p:?}");
            }
        }
    }

    #[test]
    fn scissors_cut_as_testing_every_voxel() {
        let mut next = xorshift(29);
        let dims = [37, 23, 19];
        let rect = Rect::from_min_size(egui::pos2(40.0, 30.0), egui::vec2(500.0, 380.0));
        for case in 0..40 {
            // A camera somewhere around the grid (sometimes inside it, so that some voxels are
            // behind it), looking near its middle.
            let centre = DVec3::new(18.0, 11.0, 9.0);
            let dir = DVec3::new(next() - 0.5, next() - 0.5, next() - 0.5).normalize();
            let eye = centre + dir * [6.0, 30.0, 60.0, 200.0][case % 4];
            let target = centre + DVec3::new(next() - 0.5, next() - 0.5, next() - 0.5) * 8.0;
            let look = glam::camera::rh::view::look_at_mat4(eye.as_vec3(), target.as_vec3(), glam::Vec3::Z);
            let m = (glam::camera::rh::proj::directx::perspective(0.4 + next() as f32 * 0.8, 1.3, 0.1, 1000.0) * look).as_dmat4();
            let project = move |v: DVec3| {
                let c = m * v.extend(1.0);
                if c.w <= 1e-9 {
                    return None;
                }
                Some(egui::pos2(rect.min.x + ((c.x / c.w + 1.0) * 0.5) as f32 * rect.width(), rect.min.y + ((1.0 - c.y / c.w) * 0.5) as f32 * rect.height()))
            };
            let c = rect.min + egui::vec2(next() as f32, next() as f32) * rect.size();
            let r = 5.0 + next() as f32 * 200.0;
            let n = [4, 12, 64, 300][case % 4];
            let polygon: Vec<egui::Pos2> = (0..n).map(|k| c + egui::Vec2::angled(k as f32 / n as f32 * std::f32::consts::TAU) * r * (1.0 + 0.4 * next() as f32)).collect();
            for outside in [false, true] {
                let every: Vec<u8> = (0..dims.iter().product::<usize>())
                    .map(|n| {
                        let v = DVec3::new((n % dims[0]) as f64, ((n / dims[0]) % dims[1]) as f64, (n / (dims[0] * dims[1])) as f64);
                        (project(v).is_some_and(|p| inside_every_edge(&polygon, p)) != outside) as u8
                    })
                    .collect();
                assert!(screen_cut(dims, &polygon, outside, project) == every, "case {case}");
            }
        }
    }

    /// A workspace showing a 24 × 20 × 16 volume of 0.8 × 0.6 × 1.1 mm voxels, values rising
    /// along i, with three segments.
    fn painting_ws() -> SegmentationWorkspace {
        let mut ws = SegmentationWorkspace::new(&egui::Context::default(), None);
        let dims = [24, 20, 16];
        let g = Geometry::from_axes([24, 20, 16, 1], [0.8, 0.6, 1.1], [DVec3::X, DVec3::Y, DVec3::Z], DVec3::ZERO);
        let values = (0..dims.iter().product::<usize>()).map(|n| (n % 24) as f32 * 4.0).collect();
        ws.finish_loading(Ok(LoadedVolume::from_values("t", dims, values, g)), false);
        let seg = ws.seg.as_mut().unwrap();
        for _ in 0..3 {
            seg.add_segment(None);
        }
        ws
    }

    #[test]
    fn command_right_drag_grows_a_segment_and_selects_a_part_in_a_slice_view() {
        let mut ws = painting_ws();
        let vol = ws.volume.clone().unwrap();
        // Values rise along i by 4 a voxel: the range takes i ≥ 10.
        ws.settings.as_mut().unwrap().threshold = (40.0, 100.0);
        ws.effect = Some(Effect::ThresholdGrow);
        let ctx = egui::Context::default();
        let screen = Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1000.0, 800.0));
        let command = egui::Modifiers::COMMAND;
        let mut frame = |ws: &mut SegmentationWorkspace, events: Vec<egui::Event>| {
            let input = egui::RawInput { screen_rect: Some(screen), events, ..Default::default() };
            let _ = ctx.run_ui(input, |ui| ws.views_ui(ui, None));
        };
        // ⌘ / Ctrl held from here on.
        frame(&mut ws, vec![egui::Event::ModifiersChanged(command)]);
        // A point of the axial view over a voxel well inside the range.
        let axial = ws.rects[0];
        let at = (1..20)
            .map(|k| axial.left_center() + egui::vec2(axial.width() * k as f32 / 20.0, 0.0))
            .find(|&p| {
                frame(&mut ws, vec![egui::Event::PointerMoved(p)]);
                ws.hover.is_some_and(|(_, v)| v.x >= 15.0)
            })
            .expect("a voxel in range under the axial view");
        let drag = |ws: &mut SegmentationWorkspace, frame: &mut dyn FnMut(&mut SegmentationWorkspace, Vec<egui::Event>), dx: f32| {
            let button = |p, pressed| egui::Event::PointerButton { pos: p, button: egui::PointerButton::Secondary, pressed, modifiers: command };
            frame(ws, vec![egui::Event::PointerMoved(at)]);
            frame(ws, vec![button(at, true)]);
            for k in 1..=4 {
                frame(ws, vec![egui::Event::PointerMoved(at + egui::vec2(dx * k as f32 / 4.0, 0.0))]);
            }
            frame(ws, vec![button(at + egui::vec2(dx, 0.0), false)]);
            frame(ws, Vec::new());
        };
        let label = ws.seg.as_ref().unwrap().selected_segment().unwrap().label;
        let count = |ws: &SegmentationWorkspace| ws.seg.as_ref().unwrap().labels.iter().filter(|&&l| l == label).count();
        assert_eq!(count(&ws), 0);
        drag(&mut ws, &mut frame, 120.0);
        let grown = count(&ws);
        assert!(grown > 50, "the segment grew: {grown} voxels, {:?}", ws.message);
        let seg = ws.seg.as_ref().unwrap();
        assert!(seg.labels.iter().zip(&vol.values).all(|(&l, &v)| l != label || (40.0..=100.0).contains(&v)), "only voxels in the range");
        // The view did not pan.
        assert!(ws.views.iter().all(|v| v.nav.pan == egui::Vec2::ZERO));
        // Select part: a short drag from the same voxel takes part of the segment; Erase.
        ws.effect = Some(Effect::SelectPart);
        drag(&mut ws, &mut frame, 30.0);
        let part = ws.part.as_ref().expect("a part is selected");
        assert!(part.done && part.voxels > 0 && part.voxels < grown, "{} of {grown}", part.voxels);
        let n = part.voxels;
        ws.use_part(&vol, drag::PartAction::Erase);
        assert_eq!(count(&ws), grown - n);
        // Undo: the part comes back; again: the growth goes, as one step.
        assert!(ws.seg.as_mut().unwrap().undo());
        assert_eq!(count(&ws), grown);
        assert!(ws.seg.as_mut().unwrap().undo());
        assert_eq!(count(&ws), 0);
        // With another effect, ⌘ / Ctrl + right drag pans as a right drag does.
        ws.effect = Some(Effect::Paint);
        drag(&mut ws, &mut frame, 40.0);
        assert!(ws.views[0].nav.pan.x > 30.0, "{:?}", ws.views[0].nav.pan);
        assert_eq!(count(&ws), 0);
    }

    /// The labels after painting every dab of a step from `from` to `at`, as strokes did.
    fn paint_every_dab(reference: &[u8], ws: &SegmentationWorkspace, vol: &LoadedVolume, from: DVec3, at: DVec3, plane: Option<Plane>, erase: bool) -> Vec<u8> {
        let st = ws.settings.as_ref().unwrap();
        let seg = ws.seg.as_ref().unwrap();
        let disc = plane.filter(|_| !st.sphere).map(|p| plane_axes(&vol.geometry, p).slice);
        let radius = if st.pixel_mode { 0.0 } else { st.brush_mm as f64 };
        let spacing = vol.geometry.spacing;
        let mm = (0..3).map(|k| ((at[k] - from[k]) * spacing[k]).powi(2)).sum::<f64>().sqrt();
        let step = (radius * 0.5).max(spacing.iter().cloned().fold(f64::INFINITY, f64::min) * 0.5);
        let n = (mm / step).ceil().max(1.0) as usize;
        let mut voxels = Vec::new();
        for s in 0..=n {
            effects::dab(vol.dims, spacing, from.lerp(at, s as f64 / n as f64), radius, disc, &mut voxels);
        }
        voxels.sort_unstable();
        voxels.dedup();
        let mut copy = Segmentation::new(vol.dims);
        copy.labels = reference.to_vec();
        copy.segments = seg.segments.clone();
        copy.paint(vol, &voxels, seg.selected_segment().unwrap().label, erase, &ws.masking);
        copy.labels
    }

    #[test]
    fn brush_strokes_paint_what_all_their_dabs_do() {
        let mut next = xorshift(5);
        for case in 0..24 {
            let mut ws = painting_ws();
            let vol = ws.volume.clone().unwrap();
            let mut reference = ws.seg.as_ref().unwrap().labels.clone();
            let st = ws.settings.as_mut().unwrap();
            st.brush_mm = [0.3, 1.0, 2.2, 3.5][case % 4];
            st.sphere = case % 3 == 0;
            st.pixel_mode = case % 7 == 6;
            for stroke in 0..4 {
                let erase = (case + stroke) % 3 == 2;
                let plane = [Some(Plane::Axial), Some(Plane::Sagittal), None][(case + stroke) % 3];
                ws.seg.as_mut().unwrap().selected = Some((case + stroke) % 3);
                let mut from: Option<DVec3> = None;
                for step in 0..12 {
                    // Some steps do not move; between some, the masking, the segment or the
                    // labels change (the step's start is then painted again).
                    let at = match (from, step % 4) {
                        (Some(f), 3) => f,
                        _ => DVec3::new(next() * 23.0, next() * 19.0, next() * 15.0),
                    };
                    match (step + case) % 9 {
                        4 => ws.masking.area = [EditableArea::Everywhere, EditableArea::OutsideAll, EditableArea::Inside(2)][step % 3],
                        5 => ws.masking.intensity = if ws.masking.intensity.is_some() { None } else { Some((10.0, 60.0)) },
                        6 => ws.seg.as_mut().unwrap().selected = Some(step % 3),
                        7 if from.is_some() => {
                            let seg = ws.seg.as_mut().unwrap();
                            let i = (next() * seg.labels.len() as f64) as usize;
                            seg.labels[i] = 3;
                            reference[i] = 3;
                            seg.changed(true);
                        }
                        _ => {}
                    }
                    reference = paint_every_dab(&reference, &ws, &vol, from.unwrap_or(at), at, plane, erase);
                    ws.brush(&vol, plane, at, erase);
                    from = Some(at);
                    assert!(ws.seg.as_ref().unwrap().labels == reference, "case {case}, stroke {stroke}, step {step}");
                }
                ws.end_stroke();
            }
        }
    }

    #[test]
    fn saving_clears_the_edited_mark_only_for_what_was_saved() {
        let mut ws = painting_ws();
        let saved = |ws: &mut SegmentationWorkspace, version: u64, style_version: u64| ws.finish_op(Ok(Outcome::Saved { path: "s.seg.nrrd".into(), version, style_version }), false);
        let now = |ws: &SegmentationWorkspace| (ws.seg.as_ref().unwrap().version, ws.seg.as_ref().unwrap().style_version);
        let (v, s) = now(&ws);
        saved(&mut ws, v, s);
        assert!(!ws.seg.as_ref().unwrap().modified);
        // A colour changed while saving: still edited; saved again, not.
        ws.seg.as_mut().unwrap().restyled();
        assert!(ws.seg.as_ref().unwrap().modified);
        saved(&mut ws, v, s);
        assert!(ws.seg.as_ref().unwrap().modified);
        let (v, s) = now(&ws);
        assert_eq!(v, ws.seg.as_ref().unwrap().version, "a colour leaves the labels' version");
        saved(&mut ws, v, s);
        assert!(!ws.seg.as_ref().unwrap().modified);
        // A stroke: edited, the labels at a new version.
        let vol = ws.volume.clone().unwrap();
        ws.brush(&vol, Some(Plane::Axial), DVec3::new(5.0, 5.0, 5.0), false);
        ws.end_stroke();
        assert!(ws.seg.as_ref().unwrap().modified && now(&ws).0 > v);
        saved(&mut ws, v, s);
        assert!(ws.seg.as_ref().unwrap().modified);
    }
}
