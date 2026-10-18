//! Image preparation for the selected image: N4 bias correction, run in the background and
//! saved as a new NIfTI image in the data folder (`volume/derived`), which the workspace adds
//! to its sources so the result can be previewed and segmented like any other image.

use mm_image::N4Params;
use mm_volume::model::{Kind, Volume, Voxels};
use polymix_workspace::jobs::{Job, ProgressFn};
use polymix_workspace::style::section;
use std::path::{Path, PathBuf};

/// Largest image corrected (voxels; about 12 bytes each in memory while running).
const MAX_VOXELS: usize = 400_000_000;

/// Where derived images are saved.
pub fn derived_dir() -> Option<PathBuf> {
    polymix_workspace::data_dir().map(|d| d.join("volume").join("derived"))
}

/// Whether N4 applies to this volume (readable intensity images, not label maps).
pub fn applies_to(v: &Volume) -> bool {
    matches!(v.kind, Kind::Image | Kind::Derived | Kind::TimeSeries) && !matches!(v.voxels, Voxels::Unsupported(_))
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
fn output_path(dir: &Path, source: &str, suffix: &str) -> PathBuf {
    let base = source.trim_end_matches(".gz").trim_end_matches(".nii");
    let mut safe: String = base.chars().map(|c| if c.is_alphanumeric() || " -_.".contains(c) { c } else { '_' }).collect();
    safe = safe.trim().chars().take(60).collect();
    if safe.is_empty() {
        safe = "image".into();
    }
    let mut p = dir.join(format!("{safe}{suffix}.nii.gz"));
    let mut n = 2;
    while p.exists() {
        p = dir.join(format!("{safe}{suffix} {n}.nii.gz"));
        n += 1;
    }
    p
}

pub struct Prepare {
    levels: usize,
    iterations: usize,
    spline_distance_mm: f64,
    shrink: usize,
    save_bias: bool,
    job: Option<Job<PathBuf>>,
    /// The result of the last run: Ok(file name) or Err(message).
    last: Option<Result<String, String>>,
}

impl Default for Prepare {
    fn default() -> Self {
        let d = N4Params::default();
        Self { levels: d.iterations.len(), iterations: d.iterations[0], spline_distance_mm: d.spline_distance_mm, shrink: d.shrink, save_bias: false, job: None, last: None }
    }
}

impl Prepare {
    pub fn busy(&self) -> bool {
        self.job.is_some()
    }

    fn params(&self) -> N4Params {
        N4Params { iterations: vec![self.iterations; self.levels], spline_distance_mm: self.spline_distance_mm, shrink: self.shrink, ..Default::default() }
    }

    /// Set options from a test-script argument such as `levels=2,iterations=10,shrink=2`.
    pub fn set_options(&mut self, arg: &str) {
        for (k, v) in arg.split(',').filter_map(|kv| kv.split_once('=')) {
            match k.trim() {
                "levels" => self.levels = v.trim().parse().unwrap_or(self.levels).clamp(1, 6),
                "iterations" => self.iterations = v.trim().parse().unwrap_or(self.iterations).clamp(1, 500),
                "distance" => self.spline_distance_mm = v.trim().parse().unwrap_or(self.spline_distance_mm),
                "shrink" => self.shrink = v.trim().parse().unwrap_or(self.shrink).clamp(1, 16),
                "bias" => self.save_bias = v.trim() == "on",
                _ => {}
            }
        }
    }

    /// Start correcting `v` (named `label` in the browser).
    pub fn start(&mut self, v: &Volume, label: &str, ctx: &egui::Context) {
        let Some(dir) = derived_dir() else {
            self.last = Some(Err("no data folder to save into".into()));
            return;
        };
        let (v, params, save_bias) = (v.clone(), self.params(), self.save_bias);
        let name = v.name.clone();
        self.last = None;
        self.job = Some(Job::spawn(format!("N4 bias correction of {label}"), ctx.clone(), move |progress| {
            let vol = with_progress(progress, 0.0, 0.2, |p| mm_volume::load::load_volume(&v, MAX_VOXELS, p))?;
            let out = with_progress(progress, 0.2, 0.75, |p| mm_image::n4(&vol, None, &params, p))?;
            std::fs::create_dir_all(&dir)?;
            let path = output_path(&dir, &name, " N4");
            let desc = format!("N4 bias correction of {name}");
            mm_volume::write::nifti(&path, mm_volume::write::Data::F32(&out.corrected), vol.dims, &vol.geometry, &desc, mm_volume::write::Intent::None)?;
            if save_bias {
                let bias_path = output_path(&dir, &name, " N4 bias field");
                mm_volume::write::nifti(&bias_path, mm_volume::write::Data::F32(&out.bias), vol.dims, &vol.geometry, &format!("N4 bias field of {name}"), mm_volume::write::Intent::None)?;
            }
            progress(1.0);
            Ok(path)
        }));
    }

    /// A finished run's saved image.
    pub fn poll(&mut self) -> Option<PathBuf> {
        let r = self.job.as_ref()?.poll()?;
        self.job = None;
        match r {
            Ok(p) => {
                self.last = Some(Ok(p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default()));
                Some(p)
            }
            Err(e) => {
                let msg = format!("{e:#}");
                self.last = Some(Err(if msg.contains("cancelled") { "Cancelled".into() } else { msg }));
                None
            }
        }
    }

    /// The panel for volume `v` (`label`: its name as shown, identifiers masked when hidden;
    /// `masked`: hides names in messages).
    pub fn ui(&mut self, ui: &mut egui::Ui, v: &Volume, label: &str, masked: &dyn Fn(&str) -> String) {
        if !applies_to(v) {
            return;
        }
        ui.add_space(8.0);
        section(ui, "Prepare");
        egui::CollapsingHeader::new("N4 bias correction").id_salt("volume.n4").show(ui, |ui| {
            ui.weak("Evens out the shading MR coils leave across an image (brighter near the coil, darker far from it), so one threshold fits the whole brain. The result is saved as a new image and selected.");
            egui::Grid::new("volume.n4_options").num_columns(2).spacing([10.0, 4.0]).show(ui, |ui| {
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
            });
            ui.checkbox(&mut self.save_bias, "Also save the bias field");
            match &self.job {
                Some(job) => {
                    ui.horizontal(|ui| {
                        ui.add(egui::ProgressBar::new(job.progress()).desired_width(180.0).show_percentage());
                        if ui.button("Cancel").clicked() {
                            job.cancel();
                        }
                    });
                }
                None => {
                    if ui.button("Correct shading").on_hover_text("Run N4 on this image (the head is found automatically)").clicked() {
                        self.start(v, label, ui.ctx());
                    }
                }
            }
            match &self.last {
                Some(Ok(name)) => {
                    ui.label(format!("Saved “{}” among the derived images.", masked(name)));
                }
                Some(Err(e)) => {
                    ui.colored_label(egui::Color32::from_rgb(240, 170, 90), masked(e));
                }
                None => {}
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_names() {
        let dir = std::env::temp_dir().join(format!("pv-prepare-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = output_path(&dir, "t1/pre:scan.nii.gz", " N4");
        assert_eq!(p.file_name().unwrap(), "t1_pre_scan N4.nii.gz");
        std::fs::write(&p, b"").unwrap();
        assert_eq!(output_path(&dir, "t1/pre:scan.nii.gz", " N4").file_name().unwrap(), "t1_pre_scan N4 2.nii.gz");
        assert_eq!(output_path(&dir, "Series 8 · T2 AX", " N4").file_name().unwrap(), "Series 8 _ T2 AX N4.nii.gz");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
