//! The thumbnail of the selected volume: a slice with a slice slider and window / level, with
//! the mouse controls of every image view (`polymix_imaging::nav`). Images with several time
//! points (4D series, cine clips) get a time slider and play at their frame rate; colour
//! images (ultrasound Doppler) show in colour.

use mm_volume::load::volume_count;
use mm_volume::model::{Kind, Volume, Voxels};
use mm_volume::slice::{dicom_slice_from, file_slice, load_raw_at, open_dicom, DicomFile, raw_slice, slice_count, to_grey8, to_rgb8, SliceImage, VolumeData};
use polymix_imaging::{drag_window, Nav};
use polymix_workspace::jobs::Job;
use std::path::PathBuf;
use std::sync::Arc;

/// Largest raw volume loaded for a preview (voxels; 4 bytes each in memory).
const MAX_PREVIEW_VOXELS: usize = 300_000_000;

/// Playback speeds offered.
const SPEEDS: [f64; 5] = [0.25, 0.5, 1.0, 2.0, 4.0];

/// A DICOM file kept open between slices, so the frames of a multi-frame file (a cine clip)
/// are not read from disk again for each one.
type OpenFile = (PathBuf, Arc<DicomFile>);

enum Loaded {
    /// Time point `t` of a raw-format volume in memory, and the slice shown.
    Raw(usize, Arc<VolumeData>, SliceImage),
    /// One slice (time point, slice) of a volume stored slice by slice, and its file if open.
    Slice((usize, usize), SliceImage, Option<OpenFile>),
}

pub struct Preview {
    /// Index of the volume in the catalog.
    pub volume: usize,
    pub slice: usize,
    pub slices: usize,
    /// Time point shown (the frame of a cine clip), and how many there are.
    pub frame: usize,
    pub frames: usize,
    /// Playing through the time points.
    pub playing: bool,
    /// Milliseconds per time point at normal speed (the file's frame time, else a default).
    pub frame_ms: f64,
    /// Whether the file gives the frame time.
    pub frame_ms_known: bool,
    /// Playback speed (1 = the file's frame rate).
    pub speed: f64,
    /// While playing: when (UI seconds) playback started, and from which time point.
    play_start: Option<(f64, usize)>,
    cine: bool,
    /// The time point of a raw-format volume in memory.
    raw: Option<(usize, Arc<VolumeData>)>,
    open: Option<OpenFile>,
    image: Option<SliceImage>,
    texture: Option<egui::TextureHandle>,
    /// Window (centre, width) chosen by the user; `None` = automatic.
    window: Option<(f64, f64)>,
    file_window: Option<(f64, f64)>,
    job: Option<Job<Loaded>>,
    /// Time point and slice wanted while one is still decoding.
    wanted: Option<(usize, usize)>,
    error: Option<String>,
    dirty: bool,
    nav: Nav,
}

impl Preview {
    pub fn new(index: usize, v: &Volume, ctx: &egui::Context) -> Self {
        let slices = slice_count(v);
        let frames = volume_count(v);
        let cine = v.kind == Kind::Cine;
        let mut p = Self {
            volume: index,
            slice: slices / 2,
            slices,
            frame: 0,
            frames,
            playing: false,
            frame_ms: v.frame_time_ms.unwrap_or(if cine { 40.0 } else { 100.0 }),
            frame_ms_known: v.frame_time_ms.is_some(),
            speed: 1.0,
            play_start: None,
            cine,
            raw: None,
            open: None,
            image: None,
            texture: None,
            window: None,
            file_window: v.window,
            job: None,
            wanted: None,
            error: None,
            dirty: false,
            nav: Nav::default(),
        };
        match &v.voxels {
            Voxels::Unsupported(why) => p.error = Some(format!("Pixels not readable yet: {why}")),
            _ if slices == 0 => p.error = Some("No slices".into()),
            _ => p.start(v, ctx),
        }
        p
    }

    /// Whether a slice is loading (not counting playback, which loads frame after frame).
    pub fn busy(&self) -> bool {
        self.job.is_some() && !self.playing
    }

    fn start(&mut self, v: &Volume, ctx: &egui::Context) {
        let v = v.clone();
        let (t, k) = (self.frame, self.slice);
        self.job = Some(match &v.voxels {
            Voxels::Raw { .. } => Job::spawn("Loading preview", ctx.clone(), move |_| {
                let data = Arc::new(load_raw_at(&v, t, MAX_PREVIEW_VOXELS)?);
                let s = raw_slice(&v, &data, k)?;
                Ok(Loaded::Raw(t, data, s))
            }),
            Voxels::Dicom { slices } => {
                // Volume t's slices follow those of the volumes before it.
                let index = t * self.slices + k;
                let path = slices.get(index).map(|s| s.0.clone());
                let open = self.open.clone().filter(|(p, _)| Some(p) == path.as_ref());
                Job::spawn("Decoding slice", ctx.clone(), move |_| {
                    let path = path.ok_or_else(|| anyhow::anyhow!("slice out of range"))?;
                    let obj = match open {
                        Some((_, obj)) => obj,
                        None => Arc::new(open_dicom(&path)?),
                    };
                    let s = dicom_slice_from(&obj, &v, index)?;
                    Ok(Loaded::Slice((t, k), s, Some((path, obj))))
                })
            }
            _ => Job::spawn("Decoding slice", ctx.clone(), move |_| Ok(Loaded::Slice((t, k), file_slice(&v, k)?, None))),
        });
    }

    /// Collect a finished load; returns true when something changed.
    pub fn poll(&mut self, v: &Volume, ctx: &egui::Context) -> bool {
        let Some(job) = &self.job else { return false };
        let Some(r) = job.poll() else { return false };
        self.job = None;
        match r {
            Ok(Loaded::Raw(t, data, s)) => {
                if t == self.frame {
                    self.image = Some(s);
                }
                self.raw = Some((t, data));
                self.error = None;
            }
            Ok(Loaded::Slice(at, s, open)) => {
                if at == (self.frame, self.slice) {
                    self.image = Some(s);
                }
                if open.is_some() {
                    self.open = open;
                }
                self.error = None;
            }
            Err(e) => {
                self.error = Some(format!("{e:#}"));
                self.playing = false;
            }
        }
        self.dirty = true;
        if let Some((t, k)) = self.wanted.take() {
            self.show(t, k, v, ctx);
        }
        true
    }

    /// Show slice `k` of time point `t`.
    fn show(&mut self, t: usize, k: usize, v: &Volume, ctx: &egui::Context) {
        let (t, k) = (t.min(self.frames.saturating_sub(1)), k.min(self.slices.saturating_sub(1)));
        self.frame = t;
        self.slice = k;
        if let Some((_, data)) = self.raw.as_ref().filter(|(rt, _)| *rt == t) {
            match raw_slice(v, data, k) {
                Ok(s) => self.image = Some(s),
                Err(e) => self.error = Some(format!("{e:#}")),
            }
            self.dirty = true;
        } else if self.job.is_some() {
            self.wanted = Some((t, k));
        } else if !matches!(v.voxels, Voxels::Unsupported(_)) {
            self.start(v, ctx);
        }
    }

    /// Show slice `k` (of the time point shown).
    pub fn go_to(&mut self, k: usize, v: &Volume, ctx: &egui::Context) {
        self.show(self.frame, k, v, ctx);
    }

    /// Show time point `t` (the frame of a cine clip), at the slice shown.
    pub fn go_to_frame(&mut self, t: usize, v: &Volume, ctx: &egui::Context) {
        self.show(t, self.slice, v, ctx);
    }

    /// Play through the time points (from the one shown), or pause.
    pub fn set_playing(&mut self, on: bool) {
        self.playing = on && self.frames > 1;
        self.play_start = None;
    }

    /// The window / level shown (centre, width).
    pub fn shown_window(&self) -> (f64, f64) {
        self.current_window()
    }

    /// Show `w` (centre, width), or the automatic window.
    pub fn set_window(&mut self, w: Option<(f64, f64)>) {
        self.window = w;
        self.dirty = true;
    }

    /// The window stored in the file.
    pub fn file_window(&self) -> Option<(f64, f64)> {
        self.file_window
    }

    fn current_window(&self) -> (f64, f64) {
        self.window.or_else(|| self.image.as_ref().map(|s| s.auto_window)).unwrap_or((0.5, 1.0))
    }

    fn upload(&mut self, ctx: &egui::Context) {
        let Some(s) = &self.image else { return };
        let img = match to_rgb8(s, self.current_window()) {
            Some(rgb) => egui::ColorImage::from_rgb([s.width, s.height], &rgb),
            None => egui::ColorImage::from_gray([s.width, s.height], &to_grey8(s, self.current_window())),
        };
        match &mut self.texture {
            Some(t) => t.set(img, egui::TextureOptions::LINEAR),
            None => self.texture = Some(ctx.load_texture("volume.preview", img, egui::TextureOptions::LINEAR)),
        }
        self.dirty = false;
    }

    /// Move playback on to the time point due now, and ask to be drawn again when the next
    /// one is. Frames still decoding when the next is due are skipped, so the clip keeps time.
    fn play(&mut self, v: &Volume, ctx: &egui::Context) {
        if !self.playing || self.frames < 2 {
            return;
        }
        let now = ctx.input(|i| i.time);
        let (t0, f0) = *self.play_start.get_or_insert((now, self.frame));
        let step = (self.frame_ms / self.speed).max(1.0) / 1000.0;
        let done = ((now - t0) / step).floor();
        let f = (f0 + done as usize) % self.frames;
        if f != self.frame {
            self.go_to_frame(f, v, ctx);
        }
        let next = t0 + step * (done + 1.0);
        ctx.request_repaint_after(std::time::Duration::from_secs_f64((next - now).clamp(0.001, 1.0)));
    }

    /// `crop`: a box of voxels (inclusive) to outline on the slice.
    pub fn ui(&mut self, ui: &mut egui::Ui, v: &Volume, hidden: bool, crop: Option<([usize; 3], [usize; 3])>) {
        let ctx = ui.ctx().clone();
        if hidden {
            ui.weak("Thumbnail hidden with the identifiers (images can show names).");
            return;
        }
        self.play(v, &ctx);
        if self.dirty {
            self.upload(&ctx);
        }
        if let Some(e) = &self.error {
            ui.colored_label(egui::Color32::from_rgb(240, 170, 90), e);
        }
        let width = ui.available_width().min(420.0);
        if let (Some(t), Some(s)) = (&self.texture, &self.image) {
            // Physical aspect ratio of the slice.
            let (pw, ph) = (s.width as f64 * s.mm_per_pixel[0], s.height as f64 * s.mm_per_pixel[1]);
            let scale = (width as f64 / pw).min(width as f64 * 1.1 / ph);
            let size = egui::vec2((pw * scale) as f32, (ph * scale) as f32);
            let tex = t.id();
            let (rect, r) = ui.allocate_exact_size(size, egui::Sense::click_and_drag());
            let ev = self.nav.handle(ui, &r);
            let shown = self.nav.image_rect(rect, size);
            ui.painter_at(rect).image(tex, shown, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), egui::Color32::WHITE);
            if let (Some((lo, hi)), Some(g)) = (crop, v.geometry.as_ref()) {
                // The box in the slice's screen layout (see slice::display_axes).
                let ax = mm_volume::slice::display_axes(Some(g));
                let (a, d) = (ax.across, 1 - ax.across);
                let span = |axis: usize, flip: bool| {
                    let n = g.dims[axis].max(1) as f32;
                    let (u0, u1) = (lo[axis] as f32 / n, (hi[axis] + 1) as f32 / n);
                    if flip { (1.0 - u1, 1.0 - u0) } else { (u0, u1) }
                };
                let ((x0, x1), (y0, y1)) = (span(a, ax.flip_across), span(d, ax.flip_down));
                let r = egui::Rect::from_min_max(shown.lerp_inside(egui::vec2(x0, y0)), shown.lerp_inside(egui::vec2(x1, y1)));
                let inside = (lo[2]..=hi[2]).contains(&self.slice);
                let colour = if inside { egui::Color32::from_rgb(255, 196, 0) } else { egui::Color32::from_rgba_unmultiplied(255, 196, 0, 90) };
                ui.painter_at(rect).rect_stroke(r, 0.0, egui::Stroke::new(1.5, colour), egui::StrokeKind::Middle);
            }
            let key = ("volume.preview", self.volume, self.frame, self.slice, s.values.as_ptr() as usize, s.values.len());
            polymix_imaging::histogram::show(ui, r.id, rect, rect.bottom(), key, &s.values, self.current_window(), s.invert, self.nav.windowing());
            if let Some(d) = ev.window {
                self.window = Some(drag_window(self.current_window(), d));
                self.dirty = true;
            }
            if ev.double_clicked {
                self.window = None;
                self.nav.reset();
                self.dirty = true;
            }
            if ev.slices != 0 {
                if self.slices > 1 {
                    let k = (self.slice as i64 + ev.slices as i64).clamp(0, self.slices.saturating_sub(1) as i64) as usize;
                    self.go_to(k, v, &ctx);
                } else if self.frames > 1 {
                    // One slice: the wheel steps through the frames.
                    self.set_playing(false);
                    let t = (self.frame as i64 + ev.slices as i64).clamp(0, self.frames as i64 - 1) as usize;
                    self.go_to_frame(t, v, &ctx);
                }
            }
            let through = if self.slices > 1 { "slices" } else if self.frames > 1 { "frames" } else { "slices" };
            r.on_hover_text(format!("Left drag or wheel: {through} · Right drag: pan · Middle drag: zoom · Left + right drag: window / level · Double-click: reset"));
        } else if self.job.is_some() {
            ui.horizontal(|ui| {
                polymix_workspace::spinner(ui);
                ui.weak("Loading…");
            });
        }
        if self.slices > 1 {
            let mut k = self.slice;
            ui.add(egui::Slider::new(&mut k, 0..=self.slices - 1).text(format!("slice of {}", self.slices)));
            if k != self.slice {
                self.go_to(k, v, &ctx);
            }
        }
        if self.frames > 1 {
            self.time_ui(ui, v, &ctx);
        }
        if self.image.is_some() {
            ui.horizontal(|ui| {
                let (c, w) = self.current_window();
                ui.weak(format!("Window {w:.0} / level {c:.0}"));
                if ui.small_button("Auto").on_hover_text("1st–99th percentile of this slice (colour images: as stored)").clicked() {
                    self.window = None;
                    self.dirty = true;
                }
                if let Some(fw) = self.file_window {
                    if ui.small_button("File").on_hover_text("The window stored in the file").clicked() {
                        self.window = Some(fw);
                        self.dirty = true;
                    }
                }
                if self.job.is_some() && !self.playing {
                    polymix_workspace::spinner(ui);
                }
            });
        }
    }

    /// The time slider, play / pause and the playback speed.
    fn time_ui(&mut self, ui: &mut egui::Ui, v: &Volume, ctx: &egui::Context) {
        let what = if self.cine { "frame" } else { "time point" };
        ui.horizontal(|ui| {
            let label = if self.playing { "Pause" } else { "Play" };
            if ui.button(label).on_hover_text(format!("Play the {what}s{}", if self.frame_ms_known { " at the file's frame rate" } else { "" })).clicked() {
                let on = !self.playing;
                self.set_playing(on);
            }
            let mut t = self.frame;
            let r = ui.add(egui::Slider::new(&mut t, 0..=self.frames - 1).text(format!("{what} of {}", self.frames)));
            if t != self.frame {
                if r.dragged() || r.changed() {
                    self.set_playing(false);
                }
                self.go_to_frame(t, v, ctx);
            }
        });
        ui.horizontal(|ui| {
            let fps = 1000.0 / self.frame_ms;
            let at = self.frame as f64 * self.frame_ms / 1000.0;
            if self.frame_ms_known {
                ui.weak(format!("{at:.2} s · {fps:.1} frames/s"));
            } else {
                ui.weak(format!("{fps:.0} frames/s (no frame time in the file)"));
            }
            egui::ComboBox::from_id_salt("volume.preview.speed").width(56.0).selected_text(speed_label(self.speed)).show_ui(ui, |ui| {
                for s in SPEEDS {
                    if ui.selectable_label(self.speed == s, speed_label(s)).clicked() {
                        self.speed = s;
                        self.play_start = None;
                    }
                }
            });
        });
    }
}

fn speed_label(s: f64) -> String {
    match s {
        0.25 => "¼×".into(),
        0.5 => "½×".into(),
        _ => format!("{s}×"),
    }
}
