//! The Volume tab's viewer: the image previewed in the browser in three planes side by side,
//! for preparing images — window presets, colour maps and slabs, another image over it or
//! compared with it (e.g. before and after N4 or denoising, a registration's result over its
//! target), and rulers, angles and regions.

use mm_volume::load::{load_volume, LoadedVolume};
use mm_volume::model::Volume;
use polymix_imaging::{layers, ColourMap, Fusion, Layout, Measurements, SliceOptions, SliceView, ViewState, PLANES};
use polymix_workspace::jobs::Job;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Largest volume the viewer loads (voxels).
const MAX_VOXELS: usize = 400_000_000;

/// A load in the background that stops when dropped: browsing on to another image before
/// this one arrived ends its decoding at the loader's next check (between files), instead
/// of finishing an image nobody will see while the one wanted waits for the cores.
struct Load<T> {
    job: Job<T>,
    cancel: Arc<AtomicBool>,
}

impl<T: Send + 'static> Load<T> {
    /// `f` gets the loader's progress callback: it only says whether to go on. (The loader
    /// calls it from several threads, so it reads a flag of its own, not the job's.)
    fn spawn(label: &str, ctx: &egui::Context, f: impl FnOnce(&(dyn Fn(f32) -> bool + Sync)) -> anyhow::Result<T> + Send + 'static) -> Self {
        let cancel = Arc::new(AtomicBool::new(false));
        let stop = cancel.clone();
        let job = Job::spawn(label, ctx.clone(), move |_| f(&|_| !stop.load(Ordering::Relaxed)));
        Self { job, cancel }
    }

    fn poll(&self) -> Option<anyhow::Result<T>> {
        self.job.poll()
    }
}

impl<T> Drop for Load<T> {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

pub struct Viewer {
    pub open: bool,
    /// The catalog index shown, and the image.
    pub volume: Option<(usize, Arc<LoadedVolume>)>,
    /// The image shown is no longer the catalog's image at its index (a rescan removed or
    /// changed it): it stays up until another replaces it, but is never taken for the image
    /// now at that index.
    stale: bool,
    loading: Option<(usize, Load<LoadedVolume>)>,
    /// The catalog index whose load failed (its error is shown): not tried again until
    /// another image is asked for or the catalog changes.
    failed: Option<usize>,
    pub state: Option<ViewState>,
    views: [SliceView; 3],
    pub measure: Measurements,
    fusion_job: Option<Load<Fusion>>,
    pub error: Option<String>,
    smooth: bool,
    crosshair: bool,
}

impl Default for Viewer {
    fn default() -> Self {
        Self { open: false, volume: None, stale: false, loading: None, failed: None, state: None, views: PLANES.map(SliceView::new), measure: Measurements::default(), fusion_job: None, error: None, smooth: true, crosshair: true }
    }
}

impl Viewer {
    pub fn busy(&self) -> bool {
        self.loading.is_some() || self.fusion_job.is_some()
    }

    /// The catalog index of the image shown, if it is still that catalog image (`None` when
    /// a rescan removed or changed it, though it stays up until replaced).
    pub fn shown_index(&self) -> Option<usize> {
        self.volume.as_ref().filter(|_| !self.stale).map(|(k, _)| *k)
    }

    /// Show catalog volume `k` (loaded in the background unless it is shown already, or its
    /// load failed: then its error stays up instead of a new try every frame). A load of
    /// another image still running is stopped.
    pub fn show(&mut self, k: usize, v: &Volume, ctx: &egui::Context) {
        let shown = self.shown_index() == Some(k);
        let loading = self.loading.as_ref().is_some_and(|(i, _)| *i == k);
        if shown || loading || self.failed == Some(k) {
            // Back to the image shown after another one failed: that one's error goes.
            if (shown || loading) && self.failed.is_some_and(|f| f != k) {
                self.failed = None;
                self.error = None;
            }
            // Back to the image shown while another loads: that load would replace it, so it
            // stops.
            if !loading {
                self.loading = None;
            }
            return;
        }
        let v = v.clone();
        self.failed = None;
        self.error = None;
        // (No progress is shown for it: only the stop.)
        self.loading = Some((k, Load::spawn("Loading the viewer's image", ctx, move |go_on| load_volume(&v, MAX_VOXELS, go_on))));
    }

    /// Show `other` over the image (resampled onto its grid in the background). A comparison
    /// still loading is stopped.
    pub fn compare_with(&mut self, other: Volume, label: String, ctx: &egui::Context) {
        let Some((_, vol)) = self.volume.clone() else { return };
        self.fusion_job = Some(Load::spawn("Loading the image to compare", ctx, move |go_on| {
            let o = load_volume(&other, MAX_VOXELS, go_on)?;
            // Replaced while it loaded: no resampling either.
            if !go_on(1.0) {
                anyhow::bail!("cancelled");
            }
            Ok(Fusion::new(label, Fusion::resampled(&vol, &o), ColourMap::Hot))
        }));
    }

    /// The catalog was replaced (a rescan): its indices may name other images now, so none
    /// is trusted any more and a failed one gets another try.
    pub fn catalog_changed(&mut self) {
        self.catalog_changed_with(&|_| None);
    }

    /// [`Viewer::catalog_changed`], with `index` giving the new index of the old catalog's
    /// image at an index if the new catalog holds the same image: the image shown and the
    /// one loading keep their places under it. Any other image shown is marked stale (shown
    /// until replaced, never taken for the image now at its index) and a load is stopped.
    pub fn catalog_changed_with(&mut self, index: &dyn Fn(usize) -> Option<usize>) {
        if self.failed.take().is_some() {
            self.error = None;
        }
        if let (false, Some((k, _))) = (self.stale, &mut self.volume) {
            match index(*k) {
                Some(n) => *k = n,
                None => self.stale = true,
            }
        }
        self.loading = self.loading.take().and_then(|(k, load)| Some((index(k)?, load)));
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
                    self.stale = false;
                    // A comparison still loading was resampled onto the image just replaced:
                    // it stops instead of landing over this one.
                    self.fusion_job = None;
                }
                Err(e) => {
                    self.failed = Some(k);
                    self.error = Some(format!("{e:#}"));
                }
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
    #[allow(dead_code)] // The Volume tab masks the error (`ui_with`); kept for other callers.
    pub fn ui(&mut self, ui: &mut egui::Ui, name: &str, choices: &[(Volume, String)]) -> Option<usize> {
        self.ui_with(ui, name, choices, &|e| e.to_string())
    }

    /// [`Viewer::ui`], with an error shown as `mask` gives it: a loader's error names the
    /// file, and its path may hold a patient's name (Hide identifiers).
    pub fn ui_with(&mut self, ui: &mut egui::Ui, name: &str, choices: &[(Volume, String)], mask: &dyn Fn(&str) -> String) -> Option<usize> {
        let mut pick = None;
        ui.horizontal_wrapped(|ui| {
            ui.label(egui::RichText::new(name).strong());
            if self.loading.is_some() {
                polymix_workspace::spinner(ui);
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
                polymix_workspace::spinner(ui);
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
            ui.colored_label(ui.visuals().warn_fg_color, mask(e));
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
        let opts = SliceOptions { left_drag_scrolls: self.measure.tool.is_none(), crosshair: self.crosshair, smooth: self.smooth, brush_mm: None, tool_secondary_drag: false };
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

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec3;
    use mm_volume::geometry::Geometry;
    use mm_volume::model::{Format, Kind, Voxels};
    use std::time::{Duration, Instant};

    fn grid() -> Geometry {
        Geometry::from_axes([2, 2, 2, 1], [1.0; 3], [DVec3::X, DVec3::Y, DVec3::Z], DVec3::ZERO)
    }

    /// A catalog entry whose pixels the loader cannot read.
    fn unreadable() -> Volume {
        Volume {
            format: Format::Nifti1,
            kind: Kind::Image,
            name: "bad".into(),
            files: Vec::new(),
            bytes: 0,
            geometry: Some(grid()),
            data_type: None,
            fields: Vec::new(),
            warnings: Vec::new(),
            voxels: Voxels::Unsupported("test".into()),
            window: None,
            photometric: None,
            patient: None,
        }
    }

    /// Tick until the load running now has finished.
    fn settle(v: &mut Viewer) {
        let start = Instant::now();
        while v.loading.is_some() {
            assert!(start.elapsed() < Duration::from_secs(20), "the load finishes");
            std::thread::sleep(Duration::from_millis(1));
            v.tick();
        }
    }

    #[test]
    fn a_failed_load_is_not_tried_again_until_asked() {
        let (ctx, bad) = (egui::Context::default(), unreadable());
        // Image 1 is shown.
        let mut v = Viewer { volume: Some((1, Arc::new(LoadedVolume::from_values("good", [2, 2, 2], vec![0.0; 8], grid())))), ..Viewer::default() };
        v.show(3, &bad, &ctx);
        assert!(v.loading.is_some());
        settle(&mut v);
        assert_eq!(v.failed, Some(3));
        let error = v.error.clone().expect("the error is shown");
        assert!(error.contains("pixels not readable yet"), "{error}");
        // Asked for again on the next frames: no new load, and the error stays.
        v.show(3, &bad, &ctx);
        assert!(v.loading.is_none());
        assert_eq!(v.error.as_deref(), Some(error.as_str()));
        // A rescan (that found every image where it was): tried again.
        v.catalog_changed_with(&Some);
        assert!(v.failed.is_none() && v.error.is_none());
        v.show(3, &bad, &ctx);
        assert!(v.loading.is_some());
        settle(&mut v);
        assert_eq!(v.failed, Some(3));
        // Another unreadable image: tried, and remembered in its place.
        v.show(4, &bad, &ctx);
        assert!(v.loading.is_some() && v.error.is_none());
        settle(&mut v);
        assert_eq!(v.failed, Some(4));
        // Back to the image shown: the other one's error goes.
        v.show(1, &bad, &ctx);
        assert!(v.loading.is_none() && v.failed.is_none() && v.error.is_none());
        assert!(v.volume.as_ref().is_some_and(|(k, _)| *k == 1));
    }

    fn shown_image() -> Arc<LoadedVolume> {
        Arc::new(LoadedVolume::from_values("good", [2, 2, 2], vec![0.0; 8], grid()))
    }

    /// The stop flag of the load running now.
    fn stop_flag(v: &Viewer) -> Arc<AtomicBool> {
        v.loading.as_ref().expect("loading").1.cancel.clone()
    }

    #[test]
    fn browsing_on_stops_the_load_left_behind() {
        let (ctx, bad) = (egui::Context::default(), unreadable());
        let mut v = Viewer { volume: Some((1, shown_image())), ..Viewer::default() };
        v.show(2, &bad, &ctx);
        let first = stop_flag(&v);
        // Asked for on every frame: the same load goes on.
        v.show(2, &bad, &ctx);
        assert!(Arc::ptr_eq(&first, &stop_flag(&v)) && !first.load(Ordering::Relaxed));
        // Another image: the first load is stopped.
        v.show(3, &bad, &ctx);
        assert!(first.load(Ordering::Relaxed));
        let second = stop_flag(&v);
        assert!(!second.load(Ordering::Relaxed));
        // Back to the image shown: that load would replace it, so it stops too.
        v.show(1, &bad, &ctx);
        assert!(second.load(Ordering::Relaxed) && v.loading.is_none());
        assert!(v.volume.as_ref().is_some_and(|(k, _)| *k == 1));
        // A comparison replaced by another is stopped as well.
        v.compare_with(bad.clone(), "first".into(), &ctx);
        let compare = v.fusion_job.as_ref().expect("comparing").cancel.clone();
        v.compare_with(bad.clone(), "second".into(), &ctx);
        assert!(compare.load(Ordering::Relaxed));
        assert!(!v.fusion_job.as_ref().expect("comparing").cancel.load(Ordering::Relaxed));
    }

    #[test]
    fn a_dropped_load_stops_its_work() {
        let ctx = egui::Context::default();
        let (tx, rx) = std::sync::mpsc::channel();
        let (started_tx, started) = std::sync::mpsc::channel();
        let load: Load<()> = Load::spawn("test", &ctx, move |go_on| {
            started_tx.send(()).expect("test waiting");
            let start = Instant::now();
            // Like the loader: work on while it may.
            while go_on(0.5) && start.elapsed() < Duration::from_secs(20) {
                std::thread::sleep(Duration::from_millis(1));
            }
            tx.send(go_on(0.5)).expect("test waiting");
            anyhow::bail!("cancelled")
        });
        started.recv_timeout(Duration::from_secs(20)).expect("the work started");
        assert!(load.poll().is_none(), "still working");
        drop(load);
        assert_eq!(rx.recv_timeout(Duration::from_secs(20)), Ok(false), "the work saw the stop");
    }

    #[test]
    fn a_rescan_moves_or_retires_the_images_held() {
        let (ctx, bad) = (egui::Context::default(), unreadable());
        let mut v = Viewer { volume: Some((5, shown_image())), ..Viewer::default() };
        // Image 5 is now 7: shown under 7, and not loaded again.
        v.catalog_changed_with(&|k| (k == 5).then_some(7));
        assert!(v.volume.as_ref().is_some_and(|(k, _)| *k == 7) && !v.stale);
        v.show(7, &bad, &ctx);
        assert!(v.loading.is_none());
        // Gone: it stays up until replaced, but index 7 now names another image, loaded.
        v.catalog_changed_with(&|_| None);
        assert!(v.stale && v.volume.as_ref().is_some_and(|(k, _)| *k == 7));
        v.show(7, &bad, &ctx);
        assert!(v.loading.is_some());
        // A load follows its image, or stops when the image is gone; a stale one stays stale.
        v.catalog_changed_with(&|k| (k == 7).then_some(2));
        assert!(v.loading.as_ref().is_some_and(|(k, _)| *k == 2) && v.stale);
        let flag = stop_flag(&v);
        v.catalog_changed_with(&|_| None);
        assert!(v.loading.is_none() && flag.load(Ordering::Relaxed));
        // A load that arrives replaces the stale image (here it fails: the error shows, and
        // the stale image stays up).
        v.show(2, &bad, &ctx);
        settle(&mut v);
        assert!(v.stale && v.failed == Some(2) && v.error.is_some());
        v.catalog_changed();
        assert!(v.failed.is_none() && v.error.is_none());
    }

    /// A 2×2×2 ramp written to a temporary folder named after `test` (removed by the caller),
    /// its values, and its catalog entry.
    fn ramp(test: &str) -> (std::path::PathBuf, Vec<f32>, Volume) {
        use mm_volume::model::DataType;
        use mm_volume::write::{nifti, Data, Intent};
        let dir = std::env::temp_dir().join(format!("pv-viewer-{test}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp folder");
        let file = dir.join("ramp.nii");
        let values: Vec<f32> = (0..8).map(|x| x as f32).collect();
        nifti(&file, Data::F32(&values), [2, 2, 2], &grid(), "test", Intent::None).expect("written");
        let ramp = Volume {
            format: Format::Nifti1,
            kind: Kind::Image,
            name: "ramp".into(),
            files: vec![file.clone()],
            bytes: 0,
            geometry: Some(grid()),
            data_type: Some(DataType::F32),
            fields: Vec::new(),
            warnings: Vec::new(),
            voxels: Voxels::Raw { file, file_offset: 352, gzip: false, skip: 0, interleave: 1, dtype: DataType::F32, big_endian: false, slope: 1.0, intercept: 0.0 },
            window: None,
            photometric: None,
            patient: None,
        };
        (dir, values, ramp)
    }

    #[test]
    fn a_load_replaces_a_stale_image() {
        let (dir, values, ramp) = ramp("stale");
        let ctx = egui::Context::default();
        let mut v = Viewer { volume: Some((0, shown_image())), ..Viewer::default() };
        v.catalog_changed();
        v.show(0, &ramp, &ctx);
        settle(&mut v);
        let _ = std::fs::remove_dir_all(&dir);
        assert!(v.error.is_none(), "{:?}", v.error);
        assert!(!v.stale);
        let (k, shown) = v.volume.as_ref().expect("shown");
        assert_eq!((*k, shown.values.as_slice()), (0, values.as_slice()));
        v.show(0, &ramp, &ctx);
        assert!(v.loading.is_none());
    }

    #[test]
    fn a_new_image_stops_the_comparison_over_the_old_one() {
        let (dir, _, ramp) = ramp("compare");
        let ctx = egui::Context::default();
        let mut v = Viewer { volume: Some((1, shown_image())), ..Viewer::default() };
        // A comparison over image 1, still loading (until it is stopped).
        let (tx, rx) = std::sync::mpsc::channel();
        v.fusion_job = Some(Load::spawn("test", &ctx, move |go_on| {
            let start = Instant::now();
            while go_on(0.5) && start.elapsed() < Duration::from_secs(20) {
                std::thread::sleep(Duration::from_millis(1));
            }
            tx.send(go_on(0.5)).expect("test waiting");
            anyhow::bail!("cancelled")
        }));
        // Image 0 arrives: the comparison was resampled onto image 1, so it stops.
        v.show(0, &ramp, &ctx);
        settle(&mut v);
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(v.shown_index(), Some(0), "{:?}", v.error);
        assert!(v.fusion_job.is_none() && !v.busy());
        assert_eq!(rx.recv_timeout(Duration::from_secs(20)), Ok(false), "the comparison saw the stop");
        v.tick();
        assert!(v.state.as_ref().is_some_and(|s| s.fusion.is_none()) && v.error.is_none());
    }
}
