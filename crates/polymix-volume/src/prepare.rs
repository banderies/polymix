//! Image preparation for the selected image (one operation at a time, in the background):
//! N4 bias correction, denoising, resampling, crop / pad, reorientation and header fixes,
//! registration to another image, subtraction, intensity normalisation, vessel enhancement,
//! thick-slab projections, 4D time points, phase-contrast velocities, and de-identified export.
//! Results are saved as new NIfTI images in the data folder (`volume/derived`), which the
//! workspace adds to its sources, so they can be previewed and segmented like any other
//! image; de-identified copies go to a folder the user picks.

use crate::privacy::Privacy;
use mm_image::{intensity, slab::SlabMode, time::Combine, vessel::VesselParams, Interpolation, Metric, Motion, N4Params, RegisterParams};
use mm_volume::deid::DeidOptions;
use mm_volume::load::{volume_count, LoadedVolume};
use mm_volume::model::{Catalog, Kind, Volume, Voxels};
use mm_volume::slice::Plane;
use polymix_workspace::jobs::{Job, ProgressFn};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Largest image processed (voxels; several float copies live in memory while running).
const MAX_VOXELS: usize = 400_000_000;
const WARN: egui::Color32 = egui::Color32::from_rgb(240, 170, 90);

/// Where derived images are saved.
pub fn derived_dir() -> Option<PathBuf> {
    polymix_workspace::data_dir().map(|d| d.join("volume").join("derived"))
}

/// Whether image operations apply to this volume (readable intensity images).
pub fn applies_to(v: &Volume) -> bool {
    matches!(v.kind, Kind::Image | Kind::Derived | Kind::TimeSeries | Kind::Localizer) && !matches!(v.voxels, Voxels::Unsupported(_))
}

/// Run `f` (which reports progress from any thread) on its own thread, forwarding its
/// progress to `progress` as `lo + span · fraction` and cancelling when `progress` says so.
fn with_progress<T: Send>(progress: ProgressFn, lo: f32, span: f32, f: impl FnOnce(&(dyn Fn(f32) -> bool + Sync)) -> T + Send) -> T {
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
    let frac = AtomicU32::new(0f32.to_bits());
    let stop = AtomicBool::new(false);
    let done = AtomicBool::new(false);
    std::thread::scope(|s| {
        let worker = s.spawn(|| {
            let r = f(&|x: f32| {
                frac.store(x.to_bits(), Ordering::Relaxed);
                !stop.load(Ordering::Relaxed)
            });
            done.store(true, Ordering::Relaxed);
            r
        });
        while !done.load(Ordering::Relaxed) && !worker.is_finished() {
            std::thread::sleep(std::time::Duration::from_millis(50));
            if !progress(lo + span * f32::from_bits(frac.load(Ordering::Relaxed))) {
                stop.store(true, Ordering::Relaxed);
            }
        }
        worker.join().unwrap_or_else(|p| std::panic::resume_unwind(p))
    })
}

/// A file name for a derived image: the source's name made safe, with a suffix, not taken yet.
fn output_path(dir: &Path, source: &str, suffix: &str, ext: &str) -> PathBuf {
    let base = source.trim_end_matches(".gz").trim_end_matches(".nii");
    let mut safe: String = base.chars().map(|c| if c.is_alphanumeric() || " -_.".contains(c) { c } else { '_' }).collect();
    safe = safe.trim().chars().take(60).collect();
    if safe.is_empty() {
        safe = "image".into();
    }
    let mut p = dir.join(format!("{safe}{suffix}{ext}"));
    let mut n = 2;
    while p.exists() {
        p = dir.join(format!("{safe}{suffix} {n}{ext}"));
        n += 1;
    }
    p
}

/// Save an image as NIfTI next to the other derived images: 16-bit integers when every value
/// is one (crops and reorientations of integer scans stay small), else 32-bit floats.
fn save(dir: &Path, source: &str, suffix: &str, vol: &LoadedVolume, description: &str) -> anyhow::Result<PathBuf> {
    use mm_volume::write::{nifti, Data, Intent};
    std::fs::create_dir_all(dir)?;
    let path = output_path(dir, source, suffix, ".nii.gz");
    let integral = vol.values.iter().all(|&x| x.fract() == 0.0 && (i16::MIN as f32..=i16::MAX as f32).contains(&x));
    if integral {
        let v: Vec<i16> = vol.values.iter().map(|&x| x as i16).collect();
        nifti(&path, Data::I16(&v), vol.dims, &vol.geometry, description, Intent::None)?;
    } else {
        nifti(&path, Data::F32(&vol.values), vol.dims, &vol.geometry, description, Intent::None)?;
    }
    Ok(path)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    N4,
    Denoise,
    Resample,
    Crop,
    Reorient,
    Register,
    Subtract,
    Normalise,
    Vessels,
    Slab,
    Time,
    PhaseContrast,
    Export,
}

impl Op {
    pub const ALL: [Op; 13] = [Op::N4, Op::Denoise, Op::Resample, Op::Crop, Op::Reorient, Op::Register, Op::Subtract, Op::Normalise, Op::Vessels, Op::Slab, Op::Time, Op::PhaseContrast, Op::Export];

    pub fn label(self) -> &'static str {
        match self {
            Op::N4 => "N4 bias correction",
            Op::Denoise => "Denoise",
            Op::Resample => "Resample",
            Op::Crop => "Crop / pad",
            Op::Reorient => "Reorient / fix header",
            Op::Register => "Register to another image",
            Op::Subtract => "Subtract another image",
            Op::Normalise => "Normalise intensities",
            Op::Vessels => "Vessel enhancement",
            Op::Slab => "MIP / MinIP slab",
            Op::Time => "Time points (4D)",
            Op::PhaseContrast => "Phase contrast → velocity",
            Op::Export => "Export de-identified copy",
        }
    }

    pub fn key(self) -> &'static str {
        match self {
            Op::N4 => "n4",
            Op::Denoise => "denoise",
            Op::Resample => "resample",
            Op::Crop => "crop",
            Op::Reorient => "reorient",
            Op::Register => "register",
            Op::Subtract => "subtract",
            Op::Normalise => "normalise",
            Op::Vessels => "vessels",
            Op::Slab => "slab",
            Op::Time => "time",
            Op::PhaseContrast => "pc",
            Op::Export => "export",
        }
    }

    fn hint(self) -> &'static str {
        match self {
            Op::N4 => "Evens out the shading MR coils leave across an image (brighter near the coil, darker far from it), so one threshold fits the whole brain.",
            Op::Denoise => "Reduces noise. Gaussian and median blur everything; the diffusion filters and non-local means keep edges (vessel walls) sharp.",
            Op::Resample => "Changes the voxel size, e.g. to isotropic voxels, or finer voxels so thin vessels segment smoothly.",
            Op::Crop => "Keeps a box of the image (smaller images make every tool faster), or adds a border.",
            Op::Reorient => "Reorders the voxels to standard axes (the anatomy stays put), or fixes a header with a side or the axes the wrong way round.",
            Op::Register => "Lines this image up with another scan of the patient (e.g. TOF-MRA with a T1) and resamples it onto that scan's grid. Correct MR shading (N4) first for the best fit.",
            Op::Subtract => "This image minus another (post − pre contrast) for vessel images like DSA; the other image is put on this one's grid first.",
            Op::Normalise => "Puts intensities on a common scale so the same thresholds work across scans.",
            Op::Vessels => "Highlights tubes of the chosen diameters (Hessian vesselness), giving an image to threshold for vessels.",
            Op::Slab => "Each voxel becomes the maximum (MIP), minimum (MinIP) or mean across a slab of slices: vessels in TOF / CTA, veins in SWI.",
            Op::Time => "Picks one volume of a 4D image, combines them all (mean, maximum, minimum), or saves each separately.",
            Op::PhaseContrast => "Turns three flow-encoded phase images (2D / 4D flow) into velocities, speed and a phase-contrast angiogram.",
            Op::Export => "Writes a copy without names, IDs, birth dates or institution (for sharing). Faces are not removed, and text burned into pixels stays.",
        }
    }

    /// Whether the result can be saved to the derived images (Export writes elsewhere).
    fn derives(self) -> bool {
        self != Op::Export
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DenoiseMethod {
    Gaussian,
    Median,
    Gradient,
    Curvature,
    NonLocalMeans,
}

impl DenoiseMethod {
    const ALL: [DenoiseMethod; 5] = [DenoiseMethod::Gaussian, DenoiseMethod::Median, DenoiseMethod::Gradient, DenoiseMethod::Curvature, DenoiseMethod::NonLocalMeans];

    fn label(self) -> &'static str {
        match self {
            DenoiseMethod::Gaussian => "Gaussian",
            DenoiseMethod::Median => "Median",
            DenoiseMethod::Gradient => "Gradient diffusion",
            DenoiseMethod::Curvature => "Curvature diffusion",
            DenoiseMethod::NonLocalMeans => "Non-local means",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NormMode {
    Rescale,
    Clip,
    ZScore,
    Match,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReorientMode {
    Axes,
    Mirror,
    Swap,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TimeMode {
    Pick,
    Combine,
    Split,
}

/// The keys [`Prepare::set_options`] takes besides `op` (agents' `prepare` options).
pub const OPTION_KEYS: [&str; 24] = ["levels", "iterations", "distance", "shrink", "bias", "method", "sigma", "spacing", "interp", "axes", "mirror", "swap", "pad", "thickness", "slab", "norm", "min_d", "max_d", "dark", "motion", "metric", "time", "label", "out"];

/// What each of [`OPTION_KEYS`] sets, by operation (for agents).
pub fn options_about() -> String {
    let lower = |v: Vec<&str>| v.iter().map(|s| s.to_lowercase()).collect::<Vec<_>>().join(", ");
    let denoise = lower(DenoiseMethod::ALL.iter().map(|m| m.label()).collect());
    let interp = lower(Interpolation::ALL.iter().map(|m| m.label()).collect());
    let vessel = lower(mm_image::vessel::VesselMethod::ALL.iter().map(|m| m.label()).collect());
    format!(
        "Settings by key; others keep what the panel shows. n4: levels (1–6), iterations, distance (B-spline distance, mm), shrink, bias (true: also save the bias field). denoise: method ({denoise}), sigma (mm). resample: spacing (mm, isotropic), interp ({interp}). crop: crop_from and crop_to ([i, j, k], inclusive voxels; default the whole image), pad (voxels added on every side). reorient: axes (e.g. \"RAS\"), mirror (axis 0–2), swap (two header axes, e.g. \"ij\"). register / subtract / normalise with norm match: the other image is `reference`; register: motion (rigid, affine), metric (mi, correlation). normalise: norm (rescale, clip, zscore, match). vessels: method ({vessel}), min_d, max_d (vessel diameters, mm), dark (true: dark vessels on a bright background). slab: slab (max, min, mean), thickness (mm). time: time (a time point from 1, or mean, max, min, split). pc: the images are `phases` (and `magnitude`). export: label (the subject's new name), out (absolute folder)."
    )
}

/// Another image of the catalog, remembered by its first file and name (indices change on a
/// rescan).
#[derive(Clone, Debug, PartialEq)]
struct Pick {
    file: PathBuf,
    name: String,
}

impl Pick {
    fn of(v: &Volume) -> Self {
        Self { file: v.files.first().cloned().unwrap_or_default(), name: v.name.clone() }
    }

    fn find<'a>(&self, c: &'a Catalog) -> Option<&'a Volume> {
        c.volumes.iter().find(|v| v.files.first() == Some(&self.file) && v.name == self.name)
    }
}

/// Inclusive voxel ranges (first, last) per axis.
type VoxelBox = ([usize; 3], [usize; 3]);

/// What a finished run produced.
pub enum Outcome {
    /// New images in the derived folder (the first is selected), and a note for the user.
    Derived(Vec<PathBuf>, String),
    /// A de-identified copy written elsewhere.
    Exported { dir: PathBuf, files: usize, skipped: usize },
}

pub struct Prepare {
    op: Op,
    // N4
    levels: usize,
    iterations: usize,
    spline_distance_mm: f64,
    shrink: usize,
    save_bias: bool,
    // Denoise
    denoise: DenoiseMethod,
    sigma_mm: f64,
    median_radius: usize,
    diffusion_iterations: usize,
    conductance: f64,
    nlm_search: usize,
    nlm_patch: usize,
    nlm_strength: f64,
    // Resample
    isotropic: bool,
    spacing: [f64; 3],
    interpolation: Interpolation,
    // Crop
    crop_lo: [usize; 3],
    crop_hi: [usize; 3],
    pad: usize,
    /// The volume the crop box and voxel size were set for.
    set_for: Option<PathBuf>,
    // Reorient
    reorient: ReorientMode,
    axes: String,
    mirror_axis: usize,
    swap: (usize, usize),
    // Register / subtract / match histogram
    reference: Option<Pick>,
    register: RegisterParams,
    save_transform: bool,
    register_first: bool,
    // Normalise
    norm: NormMode,
    p_lo: f64,
    p_hi: f64,
    out_lo: f32,
    out_hi: f32,
    clip: bool,
    head_mask: bool,
    // Vessels
    vessel: VesselParams,
    // Slab
    plane: Plane,
    thickness_mm: f64,
    slab_mode: SlabMode,
    // Time
    time_mode: TimeMode,
    time_point: usize,
    combine: Combine,
    // Phase contrast
    phases: [Option<Pick>; 3],
    magnitude: Option<Pick>,
    pc: mm_image::pc::PcParams,
    // Export
    deid: DeidOptions,
    export_dir: Option<PathBuf>,

    job: Option<Job<Outcome>>,
    /// Open (or close) the panel on the next frame (test scripts).
    open_request: Option<bool>,
    /// Finding the head box for cropping.
    box_job: Option<Job<Option<VoxelBox>>>,
    /// The result of the last run: Ok(note) or Err(message).
    last: Option<Result<String, String>>,
}

impl Default for Prepare {
    fn default() -> Self {
        let n4 = N4Params::default();
        Self {
            op: Op::N4,
            levels: n4.iterations.len(),
            iterations: n4.iterations[0],
            spline_distance_mm: n4.spline_distance_mm,
            shrink: n4.shrink,
            save_bias: false,
            denoise: DenoiseMethod::Curvature,
            sigma_mm: 1.0,
            median_radius: 1,
            diffusion_iterations: 10,
            conductance: 1.0,
            nlm_search: 2,
            nlm_patch: 1,
            nlm_strength: 1.0,
            isotropic: true,
            spacing: [0.5; 3],
            interpolation: Interpolation::Sinc,
            crop_lo: [0; 3],
            crop_hi: [0; 3],
            pad: 0,
            set_for: None,
            reorient: ReorientMode::Axes,
            axes: "RAS".into(),
            mirror_axis: 0,
            swap: (0, 1),
            reference: None,
            register: RegisterParams::default(),
            save_transform: true,
            register_first: true,
            norm: NormMode::Rescale,
            p_lo: 1.0,
            p_hi: 99.0,
            out_lo: 0.0,
            out_hi: 1000.0,
            clip: true,
            head_mask: true,
            vessel: VesselParams::default(),
            plane: Plane::Axial,
            thickness_mm: 20.0,
            slab_mode: SlabMode::Max,
            time_mode: TimeMode::Combine,
            time_point: 0,
            combine: Combine::Mean,
            phases: [None, None, None],
            magnitude: None,
            pc: mm_image::pc::PcParams::default(),
            deid: DeidOptions { label: "Subject 01".into(), ..Default::default() },
            export_dir: None,
            job: None,
            open_request: None,
            box_job: None,
            last: None,
        }
    }
}

impl Prepare {
    /// The last run's message (test scripts).
    pub fn last_message(&self) -> String {
        match &self.last {
            Some(Ok(m)) => m.clone(),
            Some(Err(e)) => format!("error: {e}"),
            None => if self.busy() { "running".into() } else { "nothing run".into() },
        }
    }

    pub fn busy(&self) -> bool {
        self.job.is_some() || self.box_job.is_some()
    }

    /// The crop box to draw on the thumbnail, while cropping `v`.
    pub fn crop_box(&self, v: &Volume) -> Option<([usize; 3], [usize; 3])> {
        (self.op == Op::Crop && self.set_for.as_ref() == v.files.first()).then_some((self.crop_lo, self.crop_hi))
    }

    /// Defaults that depend on the image (the crop box, the voxel size), set when it changes
    /// (before options for it are set: [`Prepare::start`] fits them too).
    pub fn fit_to(&mut self, v: &Volume) {
        if self.set_for.as_ref() == v.files.first() {
            return;
        }
        self.set_for = v.files.first().cloned();
        if let Some(g) = &v.geometry {
            self.crop_lo = [0; 3];
            self.crop_hi = [0, 1, 2].map(|a| g.dims[a].saturating_sub(1));
            let fine = g.spacing.iter().copied().fold(f64::INFINITY, f64::min);
            self.spacing = [fine; 3];
        }
        self.time_point = 0;
    }

    /// Set options from a test-script argument such as `op=resample,spacing=1,interp=linear`.
    pub fn set_options(&mut self, arg: &str) {
        for (k, v) in arg.split(',').filter_map(|kv| kv.split_once('=')) {
            let v = v.trim();
            let num = |d: f64| v.parse::<f64>().unwrap_or(d);
            match k.trim() {
                "op" => {
                    if let Some(op) = Op::ALL.into_iter().find(|o| o.key() == v) {
                        self.op = op;
                        self.open_request = Some(true);
                    }
                }
                "levels" => self.levels = num(self.levels as f64).clamp(1.0, 6.0) as usize,
                "iterations" => self.iterations = num(self.iterations as f64).clamp(1.0, 500.0) as usize,
                "distance" => self.spline_distance_mm = num(self.spline_distance_mm),
                "shrink" => self.shrink = num(self.shrink as f64).clamp(1.0, 16.0) as usize,
                "bias" => self.save_bias = v == "on",
                "method" => {
                    if let Some(m) = DenoiseMethod::ALL.into_iter().find(|m| m.label().to_lowercase().starts_with(&v.to_lowercase())) {
                        self.denoise = m;
                    }
                    if let Some(m) = mm_image::vessel::VesselMethod::ALL.into_iter().find(|m| m.label().eq_ignore_ascii_case(v)) {
                        self.vessel.method = m;
                    }
                }
                "sigma" => self.sigma_mm = num(self.sigma_mm),
                "spacing" => {
                    self.spacing = [num(self.spacing[0]); 3];
                    self.isotropic = true;
                }
                "interp" => {
                    if let Some(i) = Interpolation::ALL.into_iter().find(|i| i.label().to_lowercase().starts_with(&v.to_lowercase())) {
                        self.interpolation = i;
                    }
                }
                "axes" => {
                    self.reorient = ReorientMode::Axes;
                    self.axes = v.to_uppercase();
                }
                "swap" => {
                    if let [a, b] = v.chars().filter_map(|c| "ijk".find(c)).collect::<Vec<_>>()[..] {
                        if a != b {
                            self.reorient = ReorientMode::Swap;
                            self.swap = (a.min(b), a.max(b));
                        }
                    }
                }
                "mirror" => {
                    self.reorient = ReorientMode::Mirror;
                    self.mirror_axis = num(0.0) as usize;
                }
                "thickness" => self.thickness_mm = num(self.thickness_mm),
                "pad" => self.pad = num(self.pad as f64).max(0.0) as usize,
                "slab" => {
                    if let Some(m) = SlabMode::ALL.into_iter().find(|m| m.label().to_lowercase().starts_with(&v.to_lowercase())) {
                        self.slab_mode = m;
                    }
                }
                "norm" => {
                    self.norm = match v {
                        "clip" => NormMode::Clip,
                        "zscore" => NormMode::ZScore,
                        "match" => NormMode::Match,
                        _ => NormMode::Rescale,
                    }
                }
                "min_d" => self.vessel.min_diameter_mm = num(self.vessel.min_diameter_mm),
                "max_d" => self.vessel.max_diameter_mm = num(self.vessel.max_diameter_mm),
                "dark" => self.vessel.bright = v != "on",
                "motion" => self.register.motion = if v == "affine" { Motion::Affine } else { Motion::Rigid },
                "metric" => self.register.metric = if v == "correlation" { Metric::Correlation } else { Metric::MutualInformation },
                "time" => {
                    self.time_mode = match v {
                        "split" => TimeMode::Split,
                        "mean" | "max" | "min" => {
                            self.combine = match v {
                                "max" => Combine::Max,
                                "min" => Combine::Min,
                                _ => Combine::Mean,
                            };
                            TimeMode::Combine
                        }
                        _ => {
                            self.time_point = v.parse::<usize>().unwrap_or(1).saturating_sub(1);
                            TimeMode::Pick
                        }
                    }
                }
                "label" => self.deid.label = v.to_string(),
                "out" => self.export_dir = Some(PathBuf::from(v)),
                _ => {}
            }
        }
    }

    /// Choose the phase images (x, y, z) and optionally the magnitude image by name.
    pub fn set_phases(&mut self, c: &Catalog, names: &[&str]) -> bool {
        let find = |n: &str| c.volumes.iter().find(|v| v.name.contains(n)).map(Pick::of);
        for (a, n) in names.iter().take(3).enumerate() {
            self.phases[a] = find(n);
        }
        self.magnitude = names.get(3).and_then(|n| find(n));
        self.phases.iter().all(|p| p.is_some())
    }

    /// For agents: the operation chosen, the other images chosen, whether it is running, and
    /// how the last run went.
    pub fn agent_state(&self) -> serde_json::Value {
        let last = match &self.last {
            Some(Ok(m)) => serde_json::json!({"ok": m}),
            Some(Err(e)) => serde_json::json!({"error": e}),
            None => serde_json::Value::Null,
        };
        serde_json::json!({
            "operation": self.op.key(),
            "operations": Op::ALL.map(Op::key),
            "running": self.busy(),
            "progress": self.job.as_ref().map(|j| (j.progress() * 100.0).round() / 100.0),
            "last_run": last,
            "reference": self.reference.as_ref().map(|p| p.name.clone()),
            "phases": self.phases.iter().map(|p| p.as_ref().map(|p| p.name.clone())).collect::<Vec<_>>(),
            "magnitude": self.magnitude.as_ref().map(|p| p.name.clone()),
        })
    }

    /// The other image (registration, subtraction, histogram matching).
    pub fn choose_reference(&mut self, v: &Volume) {
        self.reference = Some(Pick::of(v));
    }

    /// The phase images (x, y, z) and the magnitude image of phase contrast.
    pub fn choose_phases(&mut self, phases: [&Volume; 3], magnitude: Option<&Volume>) {
        self.phases = phases.map(|v| Some(Pick::of(v)));
        self.magnitude = magnitude.map(Pick::of);
    }

    /// The box Crop keeps of `v`: voxels `lo` to `hi`, inclusive.
    pub fn set_crop(&mut self, v: &Volume, lo: [usize; 3], hi: [usize; 3]) -> Result<(), String> {
        self.fit_to(v);
        let dims = v.geometry.as_ref().map(|g| g.dims).ok_or("This image has no voxel grid")?;
        if (0..3).any(|a| lo[a] > hi[a] || hi[a] >= dims[a]) {
            return Err(format!("The box must lie in the image ({} × {} × {} voxels) with crop_from ≤ crop_to", dims[0], dims[1], dims[2]));
        }
        (self.crop_lo, self.crop_hi) = (lo, hi);
        Ok(())
    }

    /// Stop the operation running.
    pub fn cancel(&self) -> bool {
        self.job.as_ref().inspect(|j| j.cancel()).is_some()
    }

    /// Choose the reference image (registration, subtraction, histogram matching) by name.
    pub fn set_reference(&mut self, c: &Catalog, name: &str) -> bool {
        match c.volumes.iter().find(|v| v.name.contains(name)) {
            Some(v) => {
                self.reference = Some(Pick::of(v));
                true
            }
            None => false,
        }
    }

    /// Start the chosen operation on volume `k` of `c` (`label`: its name as shown).
    pub fn start(&mut self, c: &Arc<Catalog>, k: usize, label: &str, ctx: &egui::Context) {
        let v = c.volumes[k].clone();
        self.fit_to(&v);
        let dir = derived_dir();
        if self.op.derives() && dir.is_none() {
            self.last = Some(Err("no data folder to save into".into()));
            return;
        }
        let dir = dir.unwrap_or_default();
        let reference = self.reference.as_ref().and_then(|p| p.find(c)).cloned();
        let needs_reference = matches!(self.op, Op::Register | Op::Subtract) || (self.op == Op::Normalise && self.norm == NormMode::Match);
        if needs_reference && reference.is_none() {
            self.last = Some(Err("choose the other image first".into()));
            return;
        }
        let name = v.name.clone();
        self.last = None;
        let op = self.op;
        let title = format!("{} of {label}", op.label());
        let ctx = ctx.clone();
        self.job = Some(match op {
            Op::N4 => {
                let params = N4Params { iterations: vec![self.iterations; self.levels], spline_distance_mm: self.spline_distance_mm, shrink: self.shrink, ..Default::default() };
                let save_bias = self.save_bias;
                Job::spawn(title, ctx, move |p| {
                    let vol = load(&v, p, 0.0, 0.2)?;
                    let out = with_progress(p, 0.2, 0.75, |q| mm_image::n4(&vol, None, &params, q))?;
                    let mut files = vec![save(&dir, &name, " N4", &mm_image::image::with_values(&vol, &name, out.corrected), &format!("N4 bias correction of {name}"))?];
                    if save_bias {
                        files.push(save(&dir, &name, " N4 bias field", &mm_image::image::with_values(&vol, &name, out.bias), &format!("N4 bias field of {name}"))?);
                    }
                    Ok(Outcome::Derived(files, format!("Iterations per level: {:?}", out.iterations)))
                })
            }
            Op::Denoise => {
                let (method, sigma, radius, iters, cond, search, patch, strength) = (self.denoise, self.sigma_mm, self.median_radius, self.diffusion_iterations, self.conductance, self.nlm_search, self.nlm_patch, self.nlm_strength);
                Job::spawn(title, ctx, move |p| {
                    let vol = load(&v, p, 0.0, 0.15)?;
                    let values = with_progress(p, 0.15, 0.8, |q| match method {
                        DenoiseMethod::Gaussian => mm_image::filter::gaussian_mm(&vol, sigma, q),
                        DenoiseMethod::Median => mm_image::denoise::median(&vol, radius, q),
                        DenoiseMethod::Gradient => mm_image::denoise::gradient_diffusion(&vol, iters, cond, q),
                        DenoiseMethod::Curvature => mm_image::denoise::curvature_diffusion(&vol, iters, cond, q),
                        DenoiseMethod::NonLocalMeans => mm_image::denoise::non_local_means(&vol, search, patch, strength, q),
                    })?;
                    let note = format!("Noise σ estimated at {:.1} before, {:.1} after", mm_image::denoise::noise_sigma(&vol), mm_image::denoise::noise_sigma(&mm_image::image::with_values(&vol, "", values.clone())));
                    let out = mm_image::image::with_values(&vol, &name, values);
                    Ok(Outcome::Derived(vec![save(&dir, &name, &format!(" {}", method.label().to_lowercase()), &out, &format!("{} of {name}", method.label()))?], note))
                })
            }
            Op::Resample => {
                let (spacing, interp) = (if self.isotropic { [self.spacing[0]; 3] } else { self.spacing }, self.interpolation);
                Job::spawn(title, ctx, move |p| {
                    let vol = load(&v, p, 0.0, 0.2)?;
                    let out = with_progress(p, 0.2, 0.75, |q| mm_image::resample_to_spacing(&vol, spacing, interp, q))?;
                    let suffix = if spacing[0] == spacing[1] && spacing[1] == spacing[2] { format!(" {}mm", trim_num(spacing[0])) } else { format!(" {}x{}x{}mm", trim_num(spacing[0]), trim_num(spacing[1]), trim_num(spacing[2])) };
                    let note = format!("{} → {} voxels", fmt_dims(vol.dims), fmt_dims(out.dims));
                    Ok(Outcome::Derived(vec![save(&dir, &name, &suffix, &out, &format!("{name} resampled ({})", interp.label()))?], note))
                })
            }
            Op::Crop => {
                let (lo, hi, pad) = (self.crop_lo, self.crop_hi, self.pad);
                Job::spawn(title, ctx, move |p| {
                    let vol = load(&v, p, 0.0, 0.6)?;
                    let mut out = mm_image::grid::crop(&vol, lo, hi)?;
                    if pad > 0 {
                        out = mm_image::grid::pad(&out, [pad; 3], [pad; 3], vol.range.0)?;
                    }
                    let note = format!("{} → {} voxels", fmt_dims(vol.dims), fmt_dims(out.dims));
                    Ok(Outcome::Derived(vec![save(&dir, &name, " crop", &out, &format!("{name} cropped"))?], note))
                })
            }
            Op::Reorient => {
                let (mode, axes, mirror, swap) = (self.reorient, self.axes.clone(), self.mirror_axis, self.swap);
                Job::spawn(title, ctx, move |p| {
                    let vol = load(&v, p, 0.0, 0.6)?;
                    let (out, suffix) = match mode {
                        ReorientMode::Axes => (mm_image::grid::reorient(&vol, &axes)?, format!(" {axes}")),
                        ReorientMode::Mirror => (mm_image::grid::mirror(&vol, mirror)?, format!(" mirrored {}", ["LR", "AP", "SI"][mirror.min(2)])),
                        ReorientMode::Swap => (mm_image::grid::swap_header_axes(&vol, swap.0, swap.1)?, format!(" axes {}{} swapped", ["i", "j", "k"][swap.0], ["i", "j", "k"][swap.1])),
                    };
                    let note = format!("Orientation {} → {}", mm_image::grid::orientation(&vol), mm_image::grid::orientation(&out));
                    Ok(Outcome::Derived(vec![save(&dir, &name, &suffix, &out, &format!("{name}{suffix}"))?], note))
                })
            }
            Op::Register => {
                let (fixed_v, params, save_tfm) = (reference.unwrap(), self.register.clone(), self.save_transform);
                Job::spawn(title, ctx, move |p| {
                    let moving = load(&v, p, 0.0, 0.1)?;
                    let fixed = load(&fixed_v, p, 0.1, 0.1)?;
                    let reg = with_progress(p, 0.2, 0.6, |q| mm_image::register(&fixed, &moving, &params, q))?;
                    let out = with_progress(p, 0.8, 0.15, |q| mm_image::resample_onto(&moving, &mm_image::register::grid_of(&fixed), &reg.transform, Interpolation::Cubic, 0.0, q))?;
                    let mut files = vec![save(&dir, &name, " registered", &out, &format!("{name} registered to {}", fixed_v.name))?];
                    if save_tfm {
                        let t = output_path(&dir, &name, " registration", ".tfm");
                        mm_image::write_itk_transform(&t, &reg.transform)?;
                        files.push(t);
                    }
                    let note = format!("{}: {:.4} → {:.4}", params.metric.label(), reg.metric_before, reg.metric_after);
                    Ok(Outcome::Derived(files, note))
                })
            }
            Op::Subtract => {
                let (other, register_first, params) = (reference.unwrap(), self.register_first, self.register.clone());
                Job::spawn(title, ctx, move |p| {
                    let a = load(&v, p, 0.0, 0.1)?;
                    let b = load(&other, p, 0.1, 0.1)?;
                    let transform = if register_first {
                        with_progress(p, 0.2, 0.5, |q| mm_image::register(&a, &b, &params, q))?.transform
                    } else {
                        mm_image::Transform::identity()
                    };
                    let b_on_a = with_progress(p, 0.7, 0.2, |q| mm_image::resample_onto(&b, &mm_image::register::grid_of(&a), &transform, Interpolation::Cubic, 0.0, q))?;
                    let out = mm_image::image::with_values(&a, &name, intensity::subtract(&a, &b_on_a)?);
                    Ok(Outcome::Derived(vec![save(&dir, &name, " minus other", &out, &format!("{name} minus {}", other.name))?], String::new()))
                })
            }
            Op::Normalise => {
                let (mode, p_lo, p_hi, lo, hi, clip, head) = (self.norm, self.p_lo, self.p_hi, self.out_lo, self.out_hi, self.clip, self.head_mask);
                Job::spawn(title, ctx, move |p| {
                    let vol = load(&v, p, 0.0, 0.4)?;
                    let mask = head.then(|| mm_image::auto_mask(&vol));
                    let values = match mode {
                        NormMode::Rescale => intensity::rescale(&vol, mask.as_deref(), p_lo, p_hi, lo, hi, clip)?,
                        NormMode::Clip => intensity::clip_percentiles(&vol, mask.as_deref(), p_lo, p_hi)?,
                        NormMode::ZScore => intensity::zscore(&vol, mask.as_deref())?,
                        NormMode::Match => {
                            let r = load(reference.as_ref().unwrap(), p, 0.4, 0.4)?;
                            let rmask = head.then(|| mm_image::auto_mask(&r));
                            intensity::match_histogram(&vol, mask.as_deref(), &r, rmask.as_deref())?
                        }
                    };
                    let suffix = match mode {
                        NormMode::Rescale => " rescaled",
                        NormMode::Clip => " clipped",
                        NormMode::ZScore => " z-score",
                        NormMode::Match => " matched",
                    };
                    let out = mm_image::image::with_values(&vol, &name, values);
                    Ok(Outcome::Derived(vec![save(&dir, &name, suffix, &out, &format!("{name}{suffix}"))?], String::new()))
                })
            }
            Op::Vessels => {
                let params = self.vessel.clone();
                Job::spawn(title, ctx, move |p| {
                    let vol = load(&v, p, 0.0, 0.15)?;
                    let values = with_progress(p, 0.15, 0.8, |q| mm_image::vessel::vesselness(&vol, &params, q))?;
                    let out = mm_image::image::with_values(&vol, &name, values);
                    let suffix = format!(" {} vesselness", params.method.label());
                    Ok(Outcome::Derived(vec![save(&dir, &name, &suffix, &out, &format!("{name}{suffix}"))?], "Values from 0 (no vessel) to 1: threshold around 0.05–0.2 to start".into()))
                })
            }
            Op::Slab => {
                let (plane, thickness, mode) = (self.plane, self.thickness_mm, self.slab_mode);
                Job::spawn(title, ctx, move |p| {
                    let vol = load(&v, p, 0.0, 0.3)?;
                    let values = with_progress(p, 0.3, 0.6, |q| mm_image::slab::slab(&vol, plane, thickness, mode, q))?;
                    let short = match mode {
                        SlabMode::Max => "MIP",
                        SlabMode::Min => "MinIP",
                        SlabMode::Mean => "mean",
                    };
                    let suffix = format!(" {short} {} {}mm", plane.label().to_lowercase(), trim_num(thickness));
                    let out = mm_image::image::with_values(&vol, &name, values);
                    Ok(Outcome::Derived(vec![save(&dir, &name, &suffix, &out, &format!("{name}{suffix}"))?], String::new()))
                })
            }
            Op::Time => {
                let (mode, t, how) = (self.time_mode, self.time_point, self.combine);
                Job::spawn(title, ctx, move |p| {
                    let n = volume_count(&v);
                    let files = match mode {
                        TimeMode::Pick => {
                            let vol = with_progress(p, 0.0, 0.9, |q| mm_volume::load::load_volume_at(&v, t.min(n - 1), MAX_VOXELS, q))?;
                            vec![save(&dir, &name, &format!(" volume {}", t.min(n - 1) + 1), &vol, &format!("{name}, volume {} of {n}", t + 1))?]
                        }
                        TimeMode::Combine => {
                            let vol = with_progress(p, 0.0, 0.9, |q| mm_image::time::combine(&v, how, MAX_VOXELS, q))?;
                            vec![save(&dir, &name, &format!(" {} of {n}", how.label().to_lowercase()), &vol, &format!("{} of the {n} volumes of {name}", how.label()))?]
                        }
                        TimeMode::Split => {
                            let mut files = Vec::new();
                            for t in 0..n {
                                let vol = with_progress(p, t as f32 / n as f32, 1.0 / n as f32, |q| mm_volume::load::load_volume_at(&v, t, MAX_VOXELS, q))?;
                                files.push(save(&dir, &name, &format!(" volume {}", t + 1), &vol, &format!("{name}, volume {} of {n}", t + 1))?);
                            }
                            files
                        }
                    };
                    Ok(Outcome::Derived(files, String::new()))
                })
            }
            Op::PhaseContrast => {
                let phases: Vec<Option<Volume>> = self.phases.iter().map(|p| p.as_ref().and_then(|p| p.find(c)).cloned()).collect();
                let magnitude = self.magnitude.as_ref().and_then(|p| p.find(c)).cloned();
                let params = self.pc.clone();
                if phases.iter().any(|p| p.is_none()) {
                    self.last = Some(Err("choose the three phase images first".into()));
                    return;
                }
                let phases: Vec<Volume> = phases.into_iter().flatten().collect();
                Job::spawn(title, ctx, move |p| {
                    let out = with_progress(p, 0.0, 0.85, |q| mm_image::pc::phase_contrast([&phases[0], &phases[1], &phases[2]], magnitude.as_ref(), &params, MAX_VOXELS, q))?;
                    let g = &out.grid;
                    let mut files = Vec::new();
                    let mut put = |suffix: &str, values: &[f32]| -> anyhow::Result<()> {
                        let vol = mm_image::image::with_values(g, &name, values.to_vec());
                        files.push(save(&dir, &name, suffix, &vol, &format!("{name}{suffix} (cm/s)"))?);
                        Ok(())
                    };
                    if let Some(a) = &out.angiogram {
                        put(" PC angiogram", a)?;
                    }
                    put(" PC mean speed", &out.mean_speed)?;
                    put(" PC peak speed", &out.peak_speed)?;
                    for (k, axis) in ["x", "y", "z"].iter().enumerate() {
                        put(&format!(" PC velocity {axis}"), &out.velocity[k])?;
                    }
                    Ok(Outcome::Derived(files, format!("{} cardiac phase(s) averaged", out.time_points)))
                })
            }
            Op::Export => {
                let Some(out_dir) = self.export_dir.clone() else {
                    self.last = Some(Err("choose a folder first".into()));
                    return;
                };
                let opts = self.deid.clone();
                // Named after the label and, for DICOM, the series (file names can hold names).
                let what = if v.format == mm_volume::model::Format::Dicom { v.name.clone() } else { "image".into() };
                let export_name = format!("{} {what}", opts.label);
                Job::spawn(title, ctx, move |p| {
                    let (files, skipped) = mm_volume::deid::export(&v, &out_dir, &export_name, &opts, p)?;
                    Ok(Outcome::Exported { dir: out_dir, files: files.len(), skipped })
                })
            }
        });
    }

    /// A finished run's derived images (to list and select).
    pub fn poll(&mut self) -> Option<Vec<PathBuf>> {
        let r = self.job.as_ref()?.poll()?;
        self.job = None;
        match r {
            Ok(Outcome::Derived(files, note)) => {
                let names: Vec<String> = files.iter().map(|p| p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default()).collect();
                let mut msg = format!("Saved {} among the derived images.", names.iter().map(|n| format!("“{n}”")).collect::<Vec<_>>().join(", "));
                if !note.is_empty() {
                    msg = format!("{msg} {note}.");
                }
                self.last = Some(Ok(msg));
                Some(files.into_iter().filter(|f| f.to_string_lossy().ends_with(".nii.gz")).collect())
            }
            Ok(Outcome::Exported { dir, files, skipped }) => {
                let mut msg = format!("Wrote {files} de-identified file(s) to {}.", dir.display());
                if skipped > 0 {
                    msg += &format!(" Skipped {skipped} screenshot(s) (text may be burned into them).");
                }
                self.last = Some(Ok(msg));
                None
            }
            Err(e) => {
                let msg = format!("{e:#}");
                self.last = Some(Err(if msg.contains("cancelled") { "Cancelled".into() } else { msg }));
                None
            }
        }
    }

    /// The panel for volume `k` of `c`: a collapsible "Prepare image" section.
    pub fn ui(&mut self, ui: &mut egui::Ui, c: &Arc<Catalog>, k: usize, privacy: &Privacy) {
        if !applies_to(&c.volumes[k]) {
            return;
        }
        let title = if self.busy() { "Prepare image (running…)" } else { "Prepare image" };
        let open = self.open_request.take();
        egui::CollapsingHeader::new(egui::RichText::new(title).strong()).id_salt("volume.prepare").default_open(false).open(open).show(ui, |ui| self.body(ui, c, k, privacy));
    }

    fn body(&mut self, ui: &mut egui::Ui, c: &Arc<Catalog>, k: usize, privacy: &Privacy) {
        let v = &c.volumes[k];
        self.fit_to(v);
        self.poll_box();
        egui::ComboBox::from_id_salt("volume.prepare.op").width(260.0).selected_text(self.op.label()).show_ui(ui, |ui| {
            for op in Op::ALL {
                ui.selectable_value(&mut self.op, op, op.label());
            }
        });
        ui.weak(self.op.hint());
        let g = v.geometry.clone();
        let dims = g.as_ref().map(|g| [g.dims[0], g.dims[1], g.dims[2]]).unwrap_or([1; 3]);
        let spacing = g.as_ref().map(|g| g.spacing).unwrap_or([1.0; 3]);
        let others: Vec<&Volume> = c.volumes.iter().filter(|o| applies_to(o) && o.files.first() != v.files.first()).collect();
        egui::Grid::new("volume.prepare.options").num_columns(2).spacing([10.0, 4.0]).show(ui, |ui| match self.op {
            Op::N4 => {
                ui.label("Levels").on_hover_text("Each level doubles the detail of the shading field");
                ui.add(egui::Slider::new(&mut self.levels, 1..=5));
                ui.end_row();
                ui.label("Iterations per level");
                ui.add(egui::DragValue::new(&mut self.iterations).range(5..=200));
                ui.end_row();
                ui.label("Field detail").on_hover_text("Distance between the field's control points at the first level; smaller follows finer shading");
                ui.add(egui::DragValue::new(&mut self.spline_distance_mm).range(25.0..=400.0).speed(5.0).suffix(" mm"));
                ui.end_row();
                ui.label("Shrink").on_hover_text("Voxels averaged per axis while fitting (faster; the field is smooth)");
                ui.add(egui::Slider::new(&mut self.shrink, 1..=8));
                ui.end_row();
                ui.label("");
                ui.checkbox(&mut self.save_bias, "Also save the bias field");
                ui.end_row();
            }
            Op::Denoise => {
                ui.label("Method");
                egui::ComboBox::from_id_salt("volume.prepare.denoise").selected_text(self.denoise.label()).show_ui(ui, |ui| {
                    for m in DenoiseMethod::ALL {
                        ui.selectable_value(&mut self.denoise, m, m.label());
                    }
                });
                ui.end_row();
                match self.denoise {
                    DenoiseMethod::Gaussian => {
                        ui.label("Standard deviation");
                        ui.add(egui::DragValue::new(&mut self.sigma_mm).range(0.1..=10.0).speed(0.05).suffix(" mm"));
                        ui.end_row();
                    }
                    DenoiseMethod::Median => {
                        ui.label("Radius");
                        ui.add(egui::Slider::new(&mut self.median_radius, 1..=3).suffix(" voxel(s)"));
                        ui.end_row();
                    }
                    DenoiseMethod::Gradient | DenoiseMethod::Curvature => {
                        ui.label("Iterations");
                        ui.add(egui::Slider::new(&mut self.diffusion_iterations, 1..=50));
                        ui.end_row();
                        ui.label("Conductance").on_hover_text("Higher smooths across weaker edges too");
                        ui.add(egui::Slider::new(&mut self.conductance, 0.2..=5.0));
                        ui.end_row();
                    }
                    DenoiseMethod::NonLocalMeans => {
                        ui.label("Search radius").on_hover_text("Voxels compared around each voxel; larger is slower (2 ≈ seconds, 3 ≈ a minute for big scans)");
                        ui.add(egui::Slider::new(&mut self.nlm_search, 1..=4));
                        ui.end_row();
                        ui.label("Patch radius");
                        ui.add(egui::Slider::new(&mut self.nlm_patch, 1..=2));
                        ui.end_row();
                        ui.label("Strength").on_hover_text("Times the estimated noise level");
                        ui.add(egui::Slider::new(&mut self.nlm_strength, 0.3..=3.0));
                        ui.end_row();
                    }
                }
            }
            Op::Resample => {
                ui.label("Voxel size");
                ui.horizontal(|ui| {
                    ui.checkbox(&mut self.isotropic, "Isotropic");
                    let n = if self.isotropic { 1 } else { 3 };
                    for a in 0..n {
                        ui.add(egui::DragValue::new(&mut self.spacing[a]).range(0.05..=10.0).speed(0.01).suffix(" mm"));
                    }
                });
                ui.end_row();
                ui.label("Now");
                ui.weak(format!("{:.3} × {:.3} × {:.3} mm, {}", spacing[0], spacing[1], spacing[2], fmt_dims(dims)));
                ui.end_row();
                ui.label("Interpolation");
                egui::ComboBox::from_id_salt("volume.prepare.interp").selected_text(self.interpolation.label()).show_ui(ui, |ui| {
                    for i in Interpolation::ALL {
                        ui.selectable_value(&mut self.interpolation, i, i.label());
                    }
                });
                ui.end_row();
            }
            Op::Crop => {
                for (a, axis) in ["i (columns)", "j (rows)", "k (slices)"].iter().enumerate() {
                    ui.label(*axis);
                    ui.horizontal(|ui| {
                        let max = dims[a].saturating_sub(1);
                        ui.add(egui::DragValue::new(&mut self.crop_lo[a]).range(0..=max).prefix("from "));
                        ui.add(egui::DragValue::new(&mut self.crop_hi[a]).range(0..=max).prefix("to "));
                        self.crop_hi[a] = self.crop_hi[a].max(self.crop_lo[a]);
                    });
                    ui.end_row();
                }
                ui.label("");
                ui.horizontal(|ui| {
                    if self.box_job.is_some() {
                        ui.spinner();
                    } else if ui.button("Fit to the head").on_hover_text("The box around the head (or object), with a 5 mm margin").clicked() {
                        let v = v.clone();
                        self.box_job = Some(Job::spawn("Finding the head", ui.ctx().clone(), move |p| {
                            let vol = load(&v, p, 0.0, 0.8)?;
                            Ok(mm_image::grid::auto_box(&vol, 5.0))
                        }));
                    }
                    if ui.button("Whole image").clicked() {
                        self.crop_lo = [0; 3];
                        self.crop_hi = dims.map(|d| d.saturating_sub(1));
                    }
                });
                ui.end_row();
                ui.label("Pad").on_hover_text("Voxels added on every side after cropping (filled with the image's minimum)");
                ui.add(egui::DragValue::new(&mut self.pad).range(0..=200).suffix(" voxels"));
                ui.end_row();
            }
            Op::Reorient => {
                ui.label("Do");
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.reorient, ReorientMode::Axes, "Standard axes");
                    ui.selectable_value(&mut self.reorient, ReorientMode::Mirror, "Mirror");
                    ui.selectable_value(&mut self.reorient, ReorientMode::Swap, "Swap header axes");
                });
                ui.end_row();
                match self.reorient {
                    ReorientMode::Axes => {
                        ui.label("Axes").on_hover_text("Where each voxel axis should point: RAS (NIfTI, FSL, FreeSurfer) or LPS (DICOM, ITK)");
                        ui.horizontal(|ui| {
                            for code in ["RAS", "LPS", "LAS", "RPI"] {
                                ui.selectable_value(&mut self.axes, code.to_string(), code);
                            }
                        });
                        ui.end_row();
                    }
                    ReorientMode::Mirror => {
                        ui.label("Across").on_hover_text("For a scan whose header has a side the wrong way round");
                        ui.horizontal(|ui| {
                            for (k, l) in ["Left–right", "Front–back", "Up–down"].iter().enumerate() {
                                ui.selectable_value(&mut self.mirror_axis, k, *l);
                            }
                        });
                        ui.end_row();
                    }
                    ReorientMode::Swap => {
                        ui.label("Axes").on_hover_text("For a header that calls, say, coronal slices axial: the data stay, their labels swap");
                        ui.horizontal(|ui| {
                            for (pair, l) in [((0, 1), "i ↔ j"), ((0, 2), "i ↔ k"), ((1, 2), "j ↔ k")] {
                                ui.selectable_value(&mut self.swap, pair, l);
                            }
                        });
                        ui.end_row();
                    }
                }
                ui.label("Now");
                ui.weak(g.as_ref().map(|g| g.orientation()).unwrap_or_default());
                ui.end_row();
            }
            Op::Register | Op::Subtract => {
                ui.label(if self.op == Op::Register { "Line up with" } else { "Minus" });
                pick_ui(ui, "volume.prepare.reference", &mut self.reference, &others, privacy);
                ui.end_row();
                if self.op == Op::Subtract {
                    ui.label("");
                    ui.checkbox(&mut self.register_first, "Register it to this image first");
                    ui.end_row();
                }
                if self.op == Op::Register || self.register_first {
                    ui.label("Motion");
                    ui.horizontal(|ui| {
                        ui.selectable_value(&mut self.register.motion, Motion::Rigid, "Rigid").on_hover_text("Rotation and shift: the same patient");
                        ui.selectable_value(&mut self.register.motion, Motion::Affine, "Affine").on_hover_text("Also scaling and shear (distortion, different scanners)");
                    });
                    ui.end_row();
                    ui.label("Similarity");
                    ui.horizontal(|ui| {
                        ui.selectable_value(&mut self.register.metric, Metric::MutualInformation, "Mutual information").on_hover_text("Different contrasts (TOF with T1, CT with MR)");
                        ui.selectable_value(&mut self.register.metric, Metric::Correlation, "Correlation").on_hover_text("The same contrast (pre / post, repeat scans)");
                    });
                    ui.end_row();
                }
                if self.op == Op::Register {
                    ui.label("");
                    ui.checkbox(&mut self.save_transform, "Also save the transform (.tfm for 3D Slicer)");
                    ui.end_row();
                }
            }
            Op::Normalise => {
                ui.label("Method");
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.norm, NormMode::Rescale, "Rescale");
                    ui.selectable_value(&mut self.norm, NormMode::Clip, "Clip");
                    ui.selectable_value(&mut self.norm, NormMode::ZScore, "Z-score");
                    ui.selectable_value(&mut self.norm, NormMode::Match, "Match another");
                });
                ui.end_row();
                match self.norm {
                    NormMode::Rescale | NormMode::Clip => {
                        ui.label("Percentiles");
                        ui.horizontal(|ui| {
                            ui.add(egui::DragValue::new(&mut self.p_lo).range(0.0..=50.0).speed(0.1).suffix(" %"));
                            ui.add(egui::DragValue::new(&mut self.p_hi).range(50.0..=100.0).speed(0.1).suffix(" %"));
                        });
                        ui.end_row();
                        if self.norm == NormMode::Rescale {
                            ui.label("Onto");
                            ui.horizontal(|ui| {
                                ui.add(egui::DragValue::new(&mut self.out_lo).speed(1.0));
                                ui.add(egui::DragValue::new(&mut self.out_hi).speed(1.0));
                                ui.checkbox(&mut self.clip, "Clip beyond");
                            });
                            ui.end_row();
                        }
                    }
                    NormMode::ZScore => {}
                    NormMode::Match => {
                        ui.label("Match");
                        pick_ui(ui, "volume.prepare.match", &mut self.reference, &others, privacy);
                        ui.end_row();
                    }
                }
                ui.label("");
                ui.checkbox(&mut self.head_mask, "Statistics inside the head only");
                ui.end_row();
            }
            Op::Vessels => {
                ui.label("Method");
                egui::ComboBox::from_id_salt("volume.prepare.vessel").selected_text(self.vessel.method.label()).show_ui(ui, |ui| {
                    for m in mm_image::vessel::VesselMethod::ALL {
                        ui.selectable_value(&mut self.vessel.method, m, m.label());
                    }
                });
                ui.end_row();
                ui.label("Diameters");
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut self.vessel.min_diameter_mm).range(0.1..=20.0).speed(0.05).suffix(" mm"));
                    ui.label("to");
                    ui.add(egui::DragValue::new(&mut self.vessel.max_diameter_mm).range(0.2..=40.0).speed(0.05).suffix(" mm"));
                    self.vessel.max_diameter_mm = self.vessel.max_diameter_mm.max(self.vessel.min_diameter_mm);
                });
                ui.end_row();
                ui.label("Scales");
                ui.add(egui::Slider::new(&mut self.vessel.scales, 1..=8));
                ui.end_row();
                ui.label("Vessels are");
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.vessel.bright, true, "Bright").on_hover_text("TOF, CTA, contrast-enhanced");
                    ui.selectable_value(&mut self.vessel.bright, false, "Dark").on_hover_text("SWI, black-blood");
                });
                ui.end_row();
                if self.vessel.method == mm_image::vessel::VesselMethod::Jerman {
                    ui.label("τ");
                    ui.add(egui::Slider::new(&mut self.vessel.tau, 0.5..=1.0));
                    ui.end_row();
                }
            }
            Op::Slab => {
                ui.label("Across");
                ui.horizontal(|ui| {
                    for p in Plane::ALL {
                        ui.selectable_value(&mut self.plane, p, p.label());
                    }
                });
                ui.end_row();
                ui.label("Thickness");
                ui.add(egui::DragValue::new(&mut self.thickness_mm).range(1.0..=200.0).speed(0.5).suffix(" mm"));
                ui.end_row();
                ui.label("Projection");
                ui.horizontal(|ui| {
                    for m in SlabMode::ALL {
                        ui.selectable_value(&mut self.slab_mode, m, m.label());
                    }
                });
                ui.end_row();
            }
            Op::Time => {
                let n = volume_count(v);
                ui.label("Volumes");
                ui.weak(n.to_string());
                ui.end_row();
                ui.label("Do");
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.time_mode, TimeMode::Pick, "Pick one");
                    ui.selectable_value(&mut self.time_mode, TimeMode::Combine, "Combine");
                    ui.selectable_value(&mut self.time_mode, TimeMode::Split, "Save each");
                });
                ui.end_row();
                match self.time_mode {
                    TimeMode::Pick => {
                        ui.label("Volume");
                        let mut one = self.time_point + 1;
                        ui.add(egui::Slider::new(&mut one, 1..=n.max(1)));
                        self.time_point = one - 1;
                        ui.end_row();
                    }
                    TimeMode::Combine => {
                        ui.label("By");
                        ui.horizontal(|ui| {
                            for m in Combine::ALL {
                                ui.selectable_value(&mut self.combine, m, m.label());
                            }
                        });
                        ui.end_row();
                    }
                    TimeMode::Split => {}
                }
            }
            Op::PhaseContrast => {
                for (a, axis) in ["Phase x (R–L)", "Phase y (A–P)", "Phase z (S–I)"].iter().enumerate() {
                    ui.label(*axis);
                    ui.horizontal(|ui| {
                        let all: Vec<&Volume> = c.volumes.iter().filter(|o| applies_to(o)).collect();
                        pick_ui(ui, ("volume.prepare.phase", a), &mut self.phases[a], &all, privacy);
                        ui.add(egui::DragValue::new(&mut self.pc.venc[a]).range(1.0..=1000.0).prefix("VENC ").suffix(" cm/s"));
                        ui.checkbox(&mut self.pc.flip[a], "Flip");
                    });
                    ui.end_row();
                }
                ui.label("Magnitude");
                let all: Vec<&Volume> = c.volumes.iter().filter(|o| applies_to(o)).collect();
                pick_ui(ui, "volume.prepare.magnitude", &mut self.magnitude, &all, privacy);
                ui.end_row();
            }
            Op::Export => {
                ui.label("Name as");
                ui.text_edit_singleline(&mut self.deid.label);
                ui.end_row();
                ui.label("Shift dates by");
                ui.add(egui::DragValue::new(&mut self.deid.shift_days).range(-36500..=36500).suffix(" days")).on_hover_text("0 keeps the dates (they tell visits apart)");
                ui.end_row();
                ui.label("");
                ui.checkbox(&mut self.deid.keep_private, "Keep private (vendor) attributes").on_hover_text("They can hold names; some tools (diffusion, mosaics) need them");
                ui.end_row();
                ui.label("");
                ui.checkbox(&mut self.deid.include_screenshots, "Include screenshots (text may be burned in)");
                ui.end_row();
                ui.label("Into");
                ui.horizontal(|ui| {
                    if ui.button("Choose folder…").clicked() {
                        if let Some(d) = rfd::FileDialog::new().set_title("Folder for the de-identified copy").pick_folder() {
                            self.export_dir = Some(d);
                        }
                    }
                    if let Some(d) = &self.export_dir {
                        ui.weak(d.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default());
                    }
                });
                ui.end_row();
            }
        });
        if self.op == Op::Time && volume_count(v) < 2 {
            ui.colored_label(WARN, "This image has a single volume.");
        }
        match &self.job {
            Some(job) => {
                ui.horizontal(|ui| {
                    ui.add(egui::ProgressBar::new(job.progress()).desired_width(200.0).show_percentage());
                    if ui.button("Cancel").clicked() {
                        job.cancel();
                    }
                });
            }
            None => {
                let label = if self.op == Op::Export { "Export" } else { "Run" };
                if ui.button(label).clicked() {
                    self.start(c, k, &privacy.path(&v.name), ui.ctx());
                }
            }
        }
        match &self.last {
            Some(Ok(msg)) => {
                ui.label(privacy.path(msg));
            }
            Some(Err(e)) => {
                ui.colored_label(WARN, privacy.path(e));
            }
            None => {}
        }
    }

    /// Collect a finished head-box search.
    fn poll_box(&mut self) {
        let Some(r) = self.box_job.as_ref().and_then(|j| j.poll()) else { return };
        self.box_job = None;
        match r {
            Ok(Some((lo, hi))) => (self.crop_lo, self.crop_hi) = (lo, hi),
            Ok(None) => self.last = Some(Err("no head or object found".into())),
            Err(e) => self.last = Some(Err(format!("{e:#}"))),
        }
    }
}

/// A picker for another image of the catalog.
fn pick_ui(ui: &mut egui::Ui, id: impl std::hash::Hash + std::fmt::Debug, pick: &mut Option<Pick>, choices: &[&Volume], privacy: &Privacy) {
    let shown = pick.as_ref().map(|p| privacy.path(&p.name)).unwrap_or_else(|| "Choose…".into());
    egui::ComboBox::from_id_salt(id).width(220.0).selected_text(shown).show_ui(ui, |ui| {
        for o in choices {
            let this = Pick::of(o);
            let text = format!("{} · {}", privacy.path(&o.name), o.geometry.as_ref().map(|g| fmt_dims([g.dims[0], g.dims[1], g.dims[2]])).unwrap_or_default());
            if ui.selectable_label(pick.as_ref() == Some(&this), text).clicked() {
                *pick = Some(this);
            }
        }
    });
}

/// Load the first volume of `v`, reporting into `lo..lo + span` of `p`.
fn load(v: &Volume, p: ProgressFn, lo: f32, span: f32) -> anyhow::Result<LoadedVolume> {
    with_progress(p, lo, span, |q| mm_volume::load::load_volume(v, MAX_VOXELS, q))
}

fn fmt_dims(d: [usize; 3]) -> String {
    format!("{} × {} × {}", d[0], d[1], d[2])
}

/// `0.5` → "0.5", `1.0` → "1".
fn trim_num(x: f64) -> String {
    let s = format!("{x:.2}");
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_names() {
        let dir = std::env::temp_dir().join(format!("pv-prepare-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = output_path(&dir, "t1/pre:scan.nii.gz", " N4", ".nii.gz");
        assert_eq!(p.file_name().unwrap(), "t1_pre_scan N4.nii.gz");
        std::fs::write(&p, b"").unwrap();
        assert_eq!(output_path(&dir, "t1/pre:scan.nii.gz", " N4", ".nii.gz").file_name().unwrap(), "t1_pre_scan N4 2.nii.gz");
        assert_eq!(output_path(&dir, "Series 8 · T2 AX", " N4", ".tfm").file_name().unwrap(), "Series 8 _ T2 AX N4.tfm");
        assert_eq!((trim_num(0.5), trim_num(1.0), trim_num(0.75)), ("0.5".into(), "1".into(), "0.75".into()));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn script_options() {
        let mut p = Prepare::default();
        p.set_options("op=resample,spacing=0.7,interp=linear");
        assert_eq!((p.op, p.spacing, p.interpolation), (Op::Resample, [0.7; 3], Interpolation::Linear));
        p.set_options("op=time,time=max");
        assert_eq!((p.time_mode, p.combine), (TimeMode::Combine, Combine::Max));
        p.set_options("op=vessels,method=jerman,dark=on");
        assert_eq!((p.vessel.method, p.vessel.bright), (mm_image::vessel::VesselMethod::Jerman, false));
    }
}
