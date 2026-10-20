//! The Volume tab's viewer: the image previewed in the browser in three planes side by side,
//! for preparing images — window presets, colour maps and slabs, another image over it or
//! compared with it (e.g. before and after N4 or denoising, a registration's result over its
//! target), and rulers, angles and regions.

use mm_volume::load::{load_volume, LoadedVolume};
use mm_volume::model::Volume;
use polymix_imaging::{layers, ColourMap, Fusion, Layout, Measurements, SliceOptions, SliceView, ViewState, PLANES};
use polymix_workspace::jobs::Job;
use std::sync::Arc;

/// Largest volume the viewer loads (voxels).
const MAX_VOXELS: usize = 400_000_000;

pub struct Viewer {
    pub open: bool,
    /// The catalog index shown, and the image.
    pub volume: Option<(usize, Arc<LoadedVolume>)>,
    loading: Option<(usize, Job<LoadedVolume>)>,
    pub state: Option<ViewState>,
    views: [SliceView; 3],
    pub measure: Measurements,
    fusion_job: Option<Job<Fusion>>,
    pub error: Option<String>,
    smooth: bool,
    crosshair: bool,
}

impl Default for Viewer {
    fn default() -> Self {
        Self { open: false, volume: None, loading: None, state: None, views: PLANES.map(SliceView::new), measure: Measurements::default(), fusion_job: None, error: None, smooth: true, crosshair: true }
    }
}

impl Viewer {
    pub fn busy(&self) -> bool {
        self.loading.is_some() || self.fusion_job.is_some()
    }

    /// Show catalog volume `k` (loaded in the background unless it is shown already).
    pub fn show(&mut self, k: usize, v: &Volume, ctx: &egui::Context) {
        if self.volume.as_ref().is_some_and(|(i, _)| *i == k) || self.loading.as_ref().is_some_and(|(i, _)| *i == k) {
            return;
        }
        let v = v.clone();
        self.error = None;
        // (The loader reports from several threads: no progress for it.)
        self.loading = Some((k, Job::spawn("Loading the viewer's image", ctx.clone(), move |_| load_volume(&v, MAX_VOXELS, &|_| true))));
    }

    /// Show `other` over the image (resampled onto its grid in the background).
    pub fn compare_with(&mut self, other: Volume, label: String, ctx: &egui::Context) {
        let Some((_, vol)) = self.volume.clone() else { return };
        self.fusion_job = Some(Job::spawn("Loading the image to compare", ctx.clone(), move |_| {
            let o = load_volume(&other, MAX_VOXELS, &|_| true)?;
            Ok(Fusion::new(label, Fusion::resampled(&vol, &o), ColourMap::Hot))
        }));
    }

    pub fn tick(&mut self) {
        if let Some(r) = self.loading.as_ref().and_then(|(_, j)| j.poll()) {
            let (k, _) = self.loading.take().expect("polled");
            match r {
                Ok(v) => {
                    let v = Arc::new(v);
                    // The same grid keeps the crosshair and window (e.g. after N4).
                    let keep = self.volume.as_ref().is_some_and(|(_, o)| o.dims == v.dims);
                    let mut state = ViewState::new(&v);
                    if let (true, Some(old)) = (keep, &self.state) {
                        state.cursor = old.cursor;
                        state.colour_map = old.colour_map;
                        state.slab = old.slab;
                    }
                    self.state = Some(state);
                    self.volume = Some((k, v));
                }
                Err(e) => self.error = Some(format!("{e:#}")),
            }
        }
        if let Some(r) = self.fusion_job.as_ref().and_then(|j| j.poll()) {
            self.fusion_job = None;
            match r {
                Ok(f) => {
                    if let Some(s) = &mut self.state {
                        s.fusion = Some(f);
                    }
                }
                Err(e) => self.error = Some(format!("Compare: {e:#}")),
            }
        }
    }

    /// The viewer: its controls above three views. `choices` are the images that can be
    /// compared with this one (with their labels); the one picked is returned.
    pub fn ui(&mut self, ui: &mut egui::Ui, name: &str, choices: &[(Volume, String)]) -> Option<usize> {
        let mut pick = None;
        ui.horizontal_wrapped(|ui| {
            ui.label(egui::RichText::new(name).strong());
            if self.loading.is_some() {
                ui.spinner();
                ui.weak("Loading…");
            }
            let (Some((_, vol)), Some(state)) = (&self.volume, &mut self.state) else { return };
            ui.separator();
            if ui.small_button("Auto").on_hover_text("1st–99th percentile of the volume").clicked() {
                state.window = vol.auto_window;
            }
            layers::display_controls(ui, "volume.viewer", state, vol);
            ui.menu_button("Compare with ▾", |ui| {
                if choices.is_empty() {
                    ui.weak("No other image overlaps this one");
                }
                egui::ScrollArea::vertical().max_height(320.0).show(ui, |ui| {
                    for (k, (_, label)) in choices.iter().enumerate() {
                        if ui.button(label).clicked() {
                            pick = Some(k);
                            ui.close();
                        }
                    }
                });
            })
            .response
            .on_hover_text("Another image over this one or beside it (blend, wipe, toggle with T): e.g. before and after N4, or a registered image over its target. Derived images come first.");
            if self.fusion_job.is_some() {
                ui.spinner();
            }
            ui.separator();
            ui.toggle_value(&mut self.crosshair, "Crosshair");
            ui.toggle_value(&mut self.smooth, "Smooth");
            ui.separator();
            self.measure.tool_buttons(ui);
        });
        if let Some(f) = self.state.as_mut().and_then(|s| s.fusion.as_mut()) {
            if !layers::fusion_controls(ui, "volume.viewer", f) {
                self.state.as_mut().expect("checked").fusion = None;
            }
        }
        if let Some(e) = &self.error {
            ui.colored_label(ui.visuals().warn_fg_color, e);
        }
        let (Some((_, vol)), Some(state)) = (self.volume.clone(), self.state.as_mut()) else {
            ui.weak("Select an image in the browser to view it here.");
            // Keep the panel's height while there is nothing to show.
            ui.allocate_space(ui.available_size());
            return pick;
        };
        // T swaps the images compared.
        if !ui.ctx().egui_wants_keyboard_input() && ui.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::T)) {
            if let Some(f) = &mut state.fusion {
                match f.compare {
                    polymix_imaging::Compare::Toggle => f.showing = !f.showing,
                    _ => f.opacity = if f.opacity > 0.0 { 0.0 } else { 1.0 },
                }
            }
        }
        let full = ui.available_rect_before_wrap();
        let rects = Layout::Row.rects(full, 2.0);
        let opts = SliceOptions { left_drag_scrolls: self.measure.tool.is_none(), crosshair: self.crosshair, smooth: self.smooth, brush_mm: None };
        let mut responses = Vec::new();
        for (view, rect) in self.views.iter_mut().zip([rects[0], rects[2], rects[3]]) {
            let r = ui.scope_builder(egui::UiBuilder::new().max_rect(rect), |ui| view.ui(ui, &vol, state, None, &opts)).inner;
            responses.push((view.plane, r));
        }
        ui.allocate_rect(full, egui::Sense::hover());
        self.measure.input(&vol, &responses);
        let maps: Vec<polymix_imaging::ViewMap> = responses.iter().map(|(_, r)| r.map).collect();
        self.measure.paint(ui, &vol, &maps, ui.ctx().pointer_hover_pos());
        if ui.input(|i| i.key_pressed(egui::Key::Escape)) && !self.measure.cancel_pending() {
            self.measure.tool = None;
        }
        pick
    }
}
