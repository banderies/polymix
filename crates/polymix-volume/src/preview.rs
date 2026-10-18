//! The thumbnail of the selected volume: a slice with a slice slider and window / level
//! (drag on the image: across for the width, up and down for the level; scroll for slices).

use mm_volume::model::{Volume, Voxels};
use mm_volume::slice::{dicom_slice, load_raw, raw_slice, slice_count, to_grey8, SliceImage, VolumeData};
use polymix_workspace::jobs::Job;
use std::sync::Arc;

/// Largest raw volume loaded for a preview (voxels; 4 bytes each in memory).
const MAX_PREVIEW_VOXELS: usize = 300_000_000;

enum Loaded {
    /// A raw-format volume in memory, and the slice shown.
    Raw(Arc<VolumeData>, SliceImage),
    /// One DICOM slice.
    Slice(usize, SliceImage),
}

pub struct Preview {
    /// Index of the volume in the catalog.
    pub volume: usize,
    pub slice: usize,
    pub slices: usize,
    raw: Option<Arc<VolumeData>>,
    image: Option<SliceImage>,
    texture: Option<egui::TextureHandle>,
    /// Window (centre, width) chosen by the user; `None` = automatic.
    window: Option<(f64, f64)>,
    file_window: Option<(f64, f64)>,
    job: Option<Job<Loaded>>,
    /// Slice wanted while a DICOM slice is still decoding.
    wanted: Option<usize>,
    error: Option<String>,
    dirty: bool,
}

impl Preview {
    pub fn new(index: usize, v: &Volume, ctx: &egui::Context) -> Self {
        let slices = slice_count(v);
        let mut p = Self { volume: index, slice: slices / 2, slices, raw: None, image: None, texture: None, window: None, file_window: v.window, job: None, wanted: None, error: None, dirty: false };
        match &v.voxels {
            Voxels::Unsupported(why) => p.error = Some(format!("Pixels not readable yet: {why}")),
            _ if slices == 0 => p.error = Some("No slices".into()),
            _ => p.start(v, ctx),
        }
        p
    }

    pub fn busy(&self) -> bool {
        self.job.is_some()
    }

    fn start(&mut self, v: &Volume, ctx: &egui::Context) {
        let v = v.clone();
        let k = self.slice;
        self.job = Some(match &v.voxels {
            Voxels::Raw { .. } => Job::spawn("Loading preview", ctx.clone(), move |_| {
                let data = Arc::new(load_raw(&v, MAX_PREVIEW_VOXELS)?);
                let s = raw_slice(&v, &data, k)?;
                Ok(Loaded::Raw(data, s))
            }),
            _ => Job::spawn("Decoding slice", ctx.clone(), move |_| Ok(Loaded::Slice(k, dicom_slice(&v, k)?))),
        });
    }

    /// Collect a finished load; returns true when something changed.
    pub fn poll(&mut self, v: &Volume, ctx: &egui::Context) -> bool {
        let Some(job) = &self.job else { return false };
        let Some(r) = job.poll() else { return false };
        self.job = None;
        match r {
            Ok(Loaded::Raw(data, s)) => {
                self.raw = Some(data);
                self.image = Some(s);
                self.error = None;
            }
            Ok(Loaded::Slice(k, s)) => {
                if k == self.slice {
                    self.image = Some(s);
                }
                self.error = None;
            }
            Err(e) => self.error = Some(format!("{e:#}")),
        }
        self.dirty = true;
        if let Some(k) = self.wanted.take() {
            self.go_to(k, v, ctx);
        }
        true
    }

    /// Show slice `k`.
    pub fn go_to(&mut self, k: usize, v: &Volume, ctx: &egui::Context) {
        let k = k.min(self.slices.saturating_sub(1));
        self.slice = k;
        if let Some(data) = &self.raw {
            match raw_slice(v, data, k) {
                Ok(s) => self.image = Some(s),
                Err(e) => self.error = Some(format!("{e:#}")),
            }
            self.dirty = true;
        } else if self.job.is_some() {
            self.wanted = Some(k);
        } else if !matches!(v.voxels, Voxels::Unsupported(_)) {
            self.start(v, ctx);
        }
    }

    fn current_window(&self) -> (f64, f64) {
        self.window.or_else(|| self.image.as_ref().map(|s| s.auto_window)).unwrap_or((0.5, 1.0))
    }

    fn upload(&mut self, ctx: &egui::Context) {
        let Some(s) = &self.image else { return };
        let grey = to_grey8(s, self.current_window());
        let img = egui::ColorImage::from_gray([s.width, s.height], &grey);
        match &mut self.texture {
            Some(t) => t.set(img, egui::TextureOptions::LINEAR),
            None => self.texture = Some(ctx.load_texture("volume.preview", img, egui::TextureOptions::LINEAR)),
        }
        self.dirty = false;
    }

    pub fn ui(&mut self, ui: &mut egui::Ui, v: &Volume, hidden: bool) {
        let ctx = ui.ctx().clone();
        if hidden {
            ui.weak("Thumbnail hidden with the identifiers (images can show names).");
            return;
        }
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
            let r = ui.add(egui::Image::new((t.id(), size)).sense(egui::Sense::click_and_drag()));
            if r.dragged() {
                let d = r.drag_delta();
                let (c, w) = self.current_window();
                let w2 = (w * (d.x as f64 * 0.01).exp()).max(1e-6);
                let c2 = c - d.y as f64 * w * 0.004;
                self.window = Some((c2, w2));
                self.dirty = true;
            }
            if r.double_clicked() {
                self.window = None;
                self.dirty = true;
            }
            if r.hovered() {
                // One slice per wheel notch, from the raw events.
                let dy: f32 = ctx.input(|i| i.raw.events.iter().map(|e| if let egui::Event::MouseWheel { delta, .. } = e { delta.y } else { 0.0 }).sum());
                if dy != 0.0 {
                    let k = if dy > 0.0 { self.slice.saturating_add(1) } else { self.slice.saturating_sub(1) };
                    self.go_to(k, v, &ctx);
                }
            }
            r.on_hover_text("Drag: window (across) and level (up / down) · Double-click: automatic window · Scroll: slices");
        } else if self.job.is_some() {
            ui.horizontal(|ui| {
                ui.spinner();
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
        if self.image.is_some() {
            ui.horizontal(|ui| {
                let (c, w) = self.current_window();
                ui.weak(format!("Window {w:.0} / level {c:.0}"));
                if ui.small_button("Auto").on_hover_text("1st–99th percentile of this slice").clicked() {
                    self.window = None;
                    self.dirty = true;
                }
                if let Some(fw) = self.file_window {
                    if ui.small_button("File").on_hover_text("The window stored in the file").clicked() {
                        self.window = Some(fw);
                        self.dirty = true;
                    }
                }
                if self.job.is_some() {
                    ui.spinner();
                }
            });
        }
    }
}
