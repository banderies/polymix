//! The Volume workspace: find, organise and preview medical images.
//!
//! Add a folder and Polymix finds the imaging data in it (DICOM, NIfTI, Analyze, NRRD),
//! however messy the folder: DICOM is organised Patient → Study → Series like 3D Slicer's
//! DICOM browser, other files are listed by folder, and the selected volume shows its header
//! and a thumbnail. It depends only on `polymix-workspace` and `mm-volume`, never on the Mesh
//! workspace, so the two cannot break each other.

mod agent;
mod prepare;
mod preview;
mod viewer;
mod privacy;
mod scan_task;

use mm_volume::model::{Catalog, Kind, Volume};
use mm_volume::scan::Progress;
use polymix_workspace::style::{format_count, section};
use polymix_workspace::Workspace;
use preview::Preview;
use privacy::Privacy;
use scan_task::ScanTask;
use std::collections::{BTreeMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// File extensions the Volume workspace opens when dropped on the window. (DICOM files
/// often have none; folders are always accepted.)
const VOLUME_EXTENSIONS: [&str; 14] = ["nii", "gz", "dcm", "ima", "nrrd", "nhdr", "hdr", "img", "mha", "mhd", "mgz", "mgh", "tif", "tiff"];
const WARN: egui::Color32 = egui::Color32::from_rgb(240, 170, 90);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Selection {
    None,
    Patient(usize),
    Study(usize, usize),
    Series(usize, usize, usize),
    Volume(usize),
}

/// Which kinds of volume the browser lists.
#[derive(Clone, Copy, Debug)]
struct Show {
    images: bool,
    labels: bool,
    other: bool,
}

/// An image of the catalog and its label (names masked as the browser shows them).
pub type Choice = (Volume, String);

/// What a list of [`Choice`]s was made for: a catalog (held weakly, so its address cannot be
/// reused by a later one while the key lives, without keeping the old catalog alive), an
/// image of it, and whether identifiers were hidden (the labels are masked then).
struct ChoicesKey {
    catalog: std::sync::Weak<Catalog>,
    volume: usize,
    hide_ids: bool,
}

impl ChoicesKey {
    fn is(&self, catalog: &Arc<Catalog>, volume: usize, hide_ids: bool) -> bool {
        std::ptr::eq(self.catalog.as_ptr(), Arc::as_ptr(catalog)) && self.volume == volume && self.hide_ids == hide_ids
    }
}

pub struct VolumeWorkspace {
    /// Folders (or files) to look in.
    roots: Vec<PathBuf>,
    catalog: Option<Arc<Catalog>>,
    scan: Option<ScanTask>,
    scan_error: Option<String>,
    /// Start a scan on the next frame (folders remembered from last time).
    scan_soon: bool,
    cache_path: Option<PathBuf>,
    selection: Selection,
    show: Show,
    search: String,
    hide_ids: bool,
    privacy: Privacy,
    /// Always hiding: what agents get, whatever the window shows.
    agent_privacy: Arc<Privacy>,
    preview: Option<Preview>,
    script: VecDeque<String>,
    /// A file opened from outside (Finder, command line): select its volume after the scan.
    open_select: Option<PathBuf>,
    prepare: prepare::Prepare,
    /// Three planes of the image previewed, under the browser.
    viewer: viewer::Viewer,
    /// The images the viewer's can be compared with, and what they were listed for.
    viewer_choices: (Option<ChoicesKey>, Vec<Choice>),
}

impl VolumeWorkspace {
    pub fn new(_ctx: &egui::Context, storage: Option<&dyn eframe::Storage>) -> Self {
        let scripted = polymix_workspace::scripted();
        let st = storage.filter(|_| !scripted);
        let roots: Vec<PathBuf> = st.and_then(|s| s.get_string("volume.roots")).map(|r| r.lines().filter(|l| !l.is_empty()).map(PathBuf::from).collect()).unwrap_or_default();
        let hide_ids = st.and_then(|s| s.get_string("volume.hide_ids")).is_some_and(|v| v == "1");
        Self {
            scan_soon: !roots.is_empty(),
            roots,
            catalog: None,
            scan: None,
            scan_error: None,
            cache_path: polymix_workspace::data_dir().map(|d| d.join("volume").join("scan-cache.json")),
            selection: Selection::None,
            show: Show { images: true, labels: true, other: false },
            search: String::new(),
            hide_ids,
            privacy: Privacy::new(hide_ids, None),
            agent_privacy: Arc::new(Privacy::new(true, None)),
            preview: None,
            script: VecDeque::new(),
            open_select: None,
            prepare: prepare::Prepare::default(),
            viewer: viewer::Viewer::default(),
            viewer_choices: (None, Vec::new()),
        }
    }

    fn start_scan(&mut self, ctx: &egui::Context) {
        self.scan_soon = false;
        self.scan_error = None;
        if self.roots.is_empty() {
            self.catalog = None;
            self.viewer.catalog_changed();
            self.selection = Selection::None;
            self.preview = None;
            return;
        }
        self.scan = Some(ScanTask::start(self.roots.clone(), self.cache_path.clone(), ctx.clone()));
    }

    fn add_roots(&mut self, paths: Vec<PathBuf>, ctx: &egui::Context) {
        if let [one] = paths.as_slice() {
            self.open_select = one.is_file().then(|| one.clone());
        }
        for p in paths {
            // Files and folders already under a source are found by its scan.
            if !self.roots.iter().any(|r| p.starts_with(r)) {
                self.roots.push(p);
            }
        }
        self.start_scan(ctx);
    }

    fn finish_scan(&mut self, result: anyhow::Result<Catalog>) {
        match result {
            Ok(c) => {
                // Keep the selected volume if it is still there.
                let keep = match self.selection {
                    Selection::Volume(k) => self.catalog.as_ref().and_then(|old| old.volumes.get(k)).and_then(|v| c.volumes.iter().position(|w| w.files.first() == v.files.first() && w.name == v.name)),
                    _ => None,
                };
                // Where the old catalog's images are in the new one, if unchanged: the preview
                // and the viewer hold indices, which may name other images now.
                let old = self.catalog.take();
                let index = |k: usize| -> Option<usize> {
                    let v = old.as_ref()?.volumes.get(k)?;
                    if c.volumes.get(k).is_some_and(|w| same_image(v, w)) {
                        return Some(k);
                    }
                    c.volumes.iter().position(|w| same_image(v, w))
                };
                // The selected image's thumbnail follows it; if it changed, a new one is made
                // (`tick`).
                self.preview = self.preview.take().zip(keep).and_then(|(mut p, k)| {
                    (index(p.volume) == Some(k)).then(|| {
                        p.volume = k;
                        p
                    })
                });
                self.viewer.catalog_changed_with(&index);
                self.selection = keep.map_or(Selection::None, Selection::Volume);
                self.privacy = Privacy::new(self.hide_ids, Some(&c));
                self.agent_privacy = Arc::new(Privacy::new(true, Some(&c)));
                self.catalog = Some(Arc::new(c));
            }
            Err(e) => {
                let msg = format!("{e:#}");
                if msg != "cancelled" {
                    self.scan_error = Some(msg);
                }
            }
        }
    }

    fn select_volume(&mut self, k: usize, ctx: &egui::Context) {
        let Some(c) = self.catalog.clone() else { return };
        let Some(v) = c.volumes.get(k) else { return };
        if self.preview.as_ref().is_none_or(|p| p.volume != k) {
            self.preview = Some(Preview::new(k, v, ctx));
        }
    }

    /// The volume other workspaces show (Segmentation): the selected volume; for a series or
    /// a study, the one previewed if it is in it, else its first image. With its name as
    /// listed here (identifiers masked when hidden).
    /// List derived images saved elsewhere (e.g. a volume masked in the Segmentation
    /// workspace): the derived folder joins the folders scanned, and is scanned again;
    /// `select` selects the first file when found.
    pub fn add_derived(&mut self, paths: Vec<PathBuf>, select: bool, ctx: &egui::Context) {
        if let Some(dir) = prepare::derived_dir() {
            if !self.roots.contains(&dir) {
                self.roots.push(dir);
            }
        }
        if select {
            self.open_select = paths.into_iter().next();
        }
        self.start_scan(ctx);
    }

    pub fn current_volume(&self) -> Option<(Arc<Catalog>, usize, String)> {
        let c = self.catalog.clone()?;
        let previewed = self.preview.as_ref().map(|p| p.volume);
        let pick = |ks: &[usize]| previewed.filter(|k| ks.contains(k)).or_else(|| ks.iter().copied().find(|&k| c.volumes[k].kind == Kind::Image)).or(ks.first().copied());
        let k = match self.selection {
            Selection::Volume(k) => Some(k),
            Selection::Series(pi, si, sei) => pick(&c.patients[pi].studies[si].series[sei].volumes),
            Selection::Study(pi, si) => pick(&c.patients[pi].studies[si].series.iter().flat_map(|s| s.volumes.iter().copied()).collect::<Vec<_>>()),
            Selection::Patient(_) | Selection::None => None,
        }?;
        let label = self.privacy.path(&c.volumes.get(k)?.name);
        Some((c, k, label))
    }

    /// Images that can be shown over `current` (fusion, before / after): those of the catalog
    /// that overlap it in space by at least half of the smaller, with their labels (names
    /// masked as the browser shows them), derived images first.
    pub fn overlay_choices(&self, current: &Volume) -> Vec<(Volume, String)> {
        let (Some(c), Some(g)) = (&self.catalog, &current.geometry) else { return Vec::new() };
        // The world box of the voxels' corners.
        let bounds = |g: &mm_volume::geometry::Geometry| {
            let d = g.dims.map(|n| n as f64 - 0.5);
            (0..8).fold((glam::DVec3::INFINITY, glam::DVec3::NEG_INFINITY), |(lo, hi), c| {
                let p = g.world(glam::DVec3::new(if c & 1 > 0 { d[0] } else { -0.5 }, if c & 2 > 0 { d[1] } else { -0.5 }, if c & 4 > 0 { d[2] } else { -0.5 }));
                (lo.min(p), hi.max(p))
            })
        };
        let vol = |b: &(glam::DVec3, glam::DVec3)| (b.1 - b.0).max(glam::DVec3::ZERO).element_product();
        let here = bounds(g);
        // Looked up once, not at every comparison of the sort.
        let derived_dir = polymix_workspace::derived_dir();
        let derived = |v: &Volume| v.files.first().is_some_and(|f| derived_dir.as_ref().is_some_and(|d| f.starts_with(d)));
        let mut out: Vec<(bool, Volume, String)> = c
            .volumes
            .iter()
            .filter(|v| !(v.name == current.name && v.files == current.files) && matches!(v.kind, Kind::Image | Kind::Derived | Kind::LabelMap))
            .filter_map(|v| {
                let b = bounds(v.geometry.as_ref()?);
                let common = (here.0.max(b.0), here.1.min(b.1));
                (vol(&common) >= 0.5 * vol(&here).min(vol(&b))).then(|| (derived(v), v.clone(), self.privacy.path(&v.name)))
            })
            .take(200)
            .collect();
        // Derived images first, each part in catalog order (the sort is stable).
        out.sort_by_key(|(derived, _, _)| !derived);
        out.into_iter().map(|(_, v, label)| (v, label)).collect()
    }

    /// Whether *Hide identifiers* is on (the shell lets agents take screenshots of the image
    /// tabs only then).
    pub fn hides_identifiers(&self) -> bool {
        self.hide_ids
    }

    /// Masks the parts of the catalog's patients' names and IDs in any text, for everything
    /// agents are sent (object names in Mesh come from file names too).
    pub fn agent_mask(&self) -> Arc<dyn Fn(&str) -> String + Send + Sync> {
        let p = self.agent_privacy.clone();
        Arc::new(move |s: &str| p.path(s))
    }

    fn set_hide(&mut self, hide: bool) {
        self.hide_ids = hide;
        self.privacy = Privacy::new(hide, self.catalog.as_deref());
    }

    fn kind_shown(&self, k: Kind) -> bool {
        match k {
            Kind::Image | Kind::TimeSeries => self.show.images,
            Kind::LabelMap | Kind::Segmentation => self.show.labels,
            Kind::VectorField | Kind::Localizer | Kind::Derived => self.show.other,
        }
    }

    fn search_matches(&self, texts: &[&str]) -> bool {
        let q = self.search.trim().to_lowercase();
        q.is_empty() || texts.iter().any(|t| t.to_lowercase().contains(&q))
    }

    fn run_script_step(&mut self, step: String, ctx: &egui::Context) {
        let (cmd, arg) = step.split_once(':').unwrap_or((step.as_str(), ""));
        match cmd {
            "vol.add" => self.add_roots(vec![PathBuf::from(arg)], ctx),
            "vol.rescan" => self.start_scan(ctx),
            "vol.hide" => self.set_hide(arg == "on"),
            "vol.select" => {
                let k = self.catalog.as_ref().and_then(|c| c.volumes.iter().position(|v| v.name.contains(arg)));
                match k {
                    Some(k) => {
                        self.selection = Selection::Volume(k);
                        self.select_volume(k, ctx);
                    }
                    None => println!("[volume] no volume named like {arg}"),
                }
            }
            "vol.slice" => {
                if let (Some(p), Some(c), Ok(k)) = (&mut self.preview, &self.catalog, arg.parse::<usize>()) {
                    p.go_to(k, &c.volumes[p.volume], ctx);
                }
            }
            "vol.show" => {
                for word in arg.split(',') {
                    match word {
                        "images" => self.show.images = true,
                        "labels" => self.show.labels = true,
                        "other" => self.show.other = true,
                        _ => {}
                    }
                }
            }
            "vol.search" => self.search = arg.to_string(),
            // vol.prepare:op=resample,spacing=1 (vol.n4:… is op=n4) — on the previewed volume.
            "vol.prepare" | "vol.n4" => {
                let target = self.preview.as_ref().map(|p| p.volume).zip(self.catalog.clone());
                if let Some((k, c)) = &target {
                    self.prepare.fit_to(&c.volumes[*k]);
                }
                if cmd == "vol.n4" {
                    self.prepare.set_options("op=n4");
                }
                self.prepare.set_options(arg);
                match target {
                    Some((k, c)) => self.prepare.start(&c, k, &c.volumes[k].name, ctx),
                    None => println!("[volume] {cmd}: nothing selected"),
                }
            }
            // vol.reference:<name> — the other image for registration, subtraction, matching.
            "vol.reference" => {
                if !self.catalog.as_ref().is_some_and(|c| self.prepare.set_reference(c, arg)) {
                    println!("[volume] no volume named like {arg}");
                }
            }
            // vol.phases:x,y,z[,magnitude] — phase-contrast inputs by name.
            "vol.phases" => {
                let names: Vec<&str> = arg.split(',').collect();
                if !self.catalog.as_ref().is_some_and(|c| self.prepare.set_phases(c, &names)) {
                    println!("[volume] phase images not found");
                }
            }
            "vol.prepared" => println!("[volume] prepared: {}", self.prepare.last_message()),
            "vol.log" => {
                // Counts only: never identifiers.
                match &self.catalog {
                    Some(c) => {
                        let series: usize = c.patients.iter().flat_map(|p| &p.studies).map(|s| s.series.len()).sum();
                        let sel = self.preview.as_ref().map(|p| format!("{} slice {}/{}", c.volumes[p.volume].name, p.slice, p.slices)).unwrap_or_default();
                        println!(
                            "[volume] {arg} roots={} files={} patients={} series={} volumes={} other_files={} recognised={} problems={} skipped={} selected={sel}",
                            self.roots.len(),
                            c.files_seen,
                            c.patients.len(),
                            series,
                            c.volumes.len(),
                            c.files.len(),
                            c.recognised.len(),
                            c.problems.len(),
                            c.skipped.file_count()
                        );
                    }
                    None => println!("[volume] {arg} no catalog ({})", self.scan_error.as_deref().unwrap_or("not scanned")),
                }
            }
            _ => log::warn!("unknown Volume script step {step}"),
        }
    }

    // ------------------------------------------------------------------ UI parts

    fn toolbar(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        ui.horizontal(|ui| {
            if ui.button("Add folder…").on_hover_text("Find the images in a folder and everything under it").clicked() {
                if let Some(dir) = rfd::FileDialog::new().set_title("Folder with images").pick_folder() {
                    self.add_roots(vec![dir], &ctx);
                }
            }
            if ui.add_enabled(!self.roots.is_empty() && self.scan.is_none(), egui::Button::new("Rescan")).on_hover_text("Look again (unchanged files are not read again)").clicked() {
                self.start_scan(&ctx);
            }
            if let Some(task) = &self.scan {
                ui.separator();
                match task.progress() {
                    Some(Progress::Reading { done, total }) => {
                        ui.add(egui::ProgressBar::new(done as f32 / total.max(1) as f32).desired_width(160.0).text(format!("Reading {} / {}", format_count(done), format_count(total))));
                    }
                    Some(Progress::Listing { files }) => {
                        polymix_workspace::spinner(ui);
                        ui.label(format!("Listing files… {}", format_count(files)));
                    }
                    Some(Progress::Organising) => {
                        polymix_workspace::spinner(ui);
                        ui.label("Organising…");
                    }
                    None => {
                        polymix_workspace::spinner(ui);
                        ui.label("Scanning…");
                    }
                }
                if ui.button("Cancel").clicked() {
                    task.cancel();
                }
            }
            ui.separator();
            ui.label("Show");
            ui.toggle_value(&mut self.show.images, "Images").on_hover_text("Intensity images and time series");
            ui.toggle_value(&mut self.show.labels, "Segmentations").on_hover_text("Segmentations, masks and label maps");
            ui.toggle_value(&mut self.show.other, "Derived & other").on_hover_text("Reformats, projections and maps computed from other images, localizers, vector fields, and DICOM series without images");
            ui.separator();
            ui.add(egui::TextEdit::singleline(&mut self.search).hint_text("Search").desired_width(160.0));
            ui.separator();
            ui.toggle_value(&mut self.viewer.open, "Viewer").on_hover_text("The image selected in three planes under the browser: window presets, colour maps, slabs, compare with another image, measure");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let mut hide = self.hide_ids;
                if ui.checkbox(&mut hide, "Hide identifiers").on_hover_text("Mask patient names, IDs, birth dates and accession numbers, also inside file names; hide thumbnails of screenshots").changed() {
                    self.set_hide(hide);
                }
            });
        });
    }

    fn sources(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        section(ui, "Folders");
        if self.roots.is_empty() {
            ui.weak("Add a folder to find the images in it: DICOM, NIfTI, Analyze and NRRD, at any depth.");
        }
        let mut remove = None;
        for (k, r) in self.roots.iter().enumerate() {
            ui.horizontal(|ui| {
                if ui.small_button("×").on_hover_text("Stop looking here").clicked() {
                    remove = Some(k);
                }
                let name = if prepare::derived_dir().as_ref() == Some(r) {
                    "Derived images (made in Polymix)".to_string()
                } else {
                    r.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| r.display().to_string())
                };
                ui.label(self.privacy.path(&name)).on_hover_text(self.privacy.path(&r.display().to_string()));
            });
        }
        if let Some(k) = remove {
            self.roots.remove(k);
            self.start_scan(&ctx);
        }
        if let Some(e) = &self.scan_error {
            ui.colored_label(WARN, format!("Scan failed: {e}"));
        }
        let Some(c) = self.catalog.clone() else { return };
        section(ui, "Found");
        let series: usize = c.patients.iter().flat_map(|p| &p.studies).map(|s| s.series.len()).sum();
        let studies: usize = c.patients.iter().map(|p| p.studies.len()).sum();
        egui::Grid::new("volume.found").num_columns(2).spacing([10.0, 3.0]).show(ui, |ui| {
            let mut row = |a: &str, b: String| {
                ui.label(a);
                ui.label(b);
                ui.end_row();
            };
            row("Files looked at", format_count(c.files_seen));
            row("DICOM patients", format_count(c.patients.len()));
            row("Studies / series", format!("{} / {}", format_count(studies), format_count(series)));
            row("Other volumes", format_count(c.files.len()));
            if !c.recognised.is_empty() {
                row("Not readable yet", format_count(c.recognised.len()));
            }
            if !c.problems.is_empty() {
                row("Could not read", format_count(c.problems.len()));
            }
            row("Not imaging", format_count(c.skipped.file_count()));
        });
        if !c.skipped.folders.is_empty() {
            ui.weak(format!("Skipped {} tool folder(s) (virtual environments, version control, …)", c.skipped.folders.len()))
                .on_hover_text(c.skipped.folders.iter().map(|f| self.privacy.path(&f.display().to_string())).collect::<Vec<_>>().join("\n"));
        }
        if !c.skipped.not_downloaded.is_empty() {
            ui.colored_label(WARN, format!("{} cloud file(s) not downloaded: not read", c.skipped.not_downloaded.len()));
        }
        if !c.skipped.no_permission.is_empty() {
            ui.colored_label(WARN, format!("{} place(s) not readable (permissions)", c.skipped.no_permission.len()));
        }
        if !c.dicomdirs.is_empty() {
            ui.weak(format!("{} DICOMDIR file(s): the folder headers were read directly", c.dicomdirs.len()));
        }
    }

    fn volume_row(&mut self, ui: &mut egui::Ui, c: &Catalog, k: usize, label: Option<String>) {
        let v = &c.volumes[k];
        let selected = self.selection == Selection::Volume(k);
        let dims = v.geometry.as_ref().map(|g| if g.dims[3] > 1 { format!("{}×{}×{}×{}", g.dims[0], g.dims[1], g.dims[2], g.dims[3]) } else { format!("{}×{}×{}", g.dims[0], g.dims[1], g.dims[2]) }).unwrap_or_default();
        let name = label.unwrap_or_else(|| self.privacy.path(&v.name));
        let mut text = egui::text::LayoutJob::default();
        let fmt = |color: egui::Color32| egui::TextFormat { color, ..Default::default() };
        let strong = ui.visuals().strong_text_color();
        let weak = ui.visuals().weak_text_color();
        text.append(&name, 0.0, fmt(strong));
        text.append(&format!("   {} · {} · {dims}", kind_label(v.kind), v.format.label()), 0.0, fmt(weak));
        if !v.warnings.is_empty() {
            text.append("  ⚠", 0.0, fmt(WARN));
        }
        let r = ui.selectable_label(selected, text);
        let r = if v.warnings.is_empty() { r } else { r.on_hover_text(v.warnings.join("\n")) };
        file_menu(&r, v.files.first().map(PathBuf::as_path));
        if r.clicked() {
            self.selection = Selection::Volume(k);
            self.select_volume(k, ui.ctx());
        }
    }

    fn browser(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        let Some(c) = self.catalog.clone() else {
            ui.vertical_centered(|ui| {
                ui.add_space(ui.available_height() * 0.3);
                if self.scan.is_some() {
                    polymix_workspace::spinner(ui);
                    ui.label("Looking for images…");
                } else {
                    ui.heading("Volume");
                    ui.label("Add a folder: Polymix finds the DICOM, NIfTI, Analyze and NRRD data in it, however it is organised.");
                    if ui.button("Add folder…").clicked() {
                        if let Some(dir) = rfd::FileDialog::new().set_title("Folder with images").pick_folder() {
                            self.add_roots(vec![dir], &ctx);
                        }
                    }
                }
            });
            return;
        };
        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            // DICOM: patients → studies → series.
            let open_all = c.patients.len() <= 4;
            egui::CollapsingHeader::new(egui::RichText::new(format!("DICOM — {} patient(s)", c.patients.len())).strong()).id_salt("volume.dicom").default_open(true).show(ui, |ui| {
                if c.patients.is_empty() {
                    ui.weak("No DICOM images found.");
                }
                for (pi, p) in c.patients.iter().enumerate() {
                    let patient_text = [p.name.as_str(), p.id.as_str()];
                    let any_shown = p.studies.iter().flat_map(|s| &s.series).any(|se| self.series_shown(&c, &patient_text, se));
                    if !any_shown && p.other_volumes.is_empty() {
                        continue;
                    }
                    let label = if self.hide_ids {
                        format!("Patient {}", pi + 1)
                    } else {
                        let mut s = p.name.clone();
                        if !p.id.is_empty() {
                            s += &format!("  ·  ID {}", p.id);
                        }
                        if !p.sex.is_empty() {
                            s += &format!("  ·  {}", p.sex);
                        }
                        if !p.birth_date.is_empty() {
                            s += &format!("  ·  born {}", date(&p.birth_date));
                        }
                        s
                    };
                    let r = egui::CollapsingHeader::new(egui::RichText::new(format!("{label}   ({} studies)", p.studies.len())).strong()).id_salt(("volume.patient", pi)).default_open(open_all).show(ui, |ui| {
                        for (si, s) in p.studies.iter().enumerate() {
                            let title = format!(
                                "{}{}{}",
                                date_time(&s.date, &s.time),
                                if s.description.is_empty() { String::new() } else { format!("  ·  {}", s.description) },
                                if s.accession.is_empty() { String::new() } else { format!("  ·  Acc. {}", self.privacy.id(&s.accession)) }
                            );
                            let r = egui::CollapsingHeader::new(format!("{title}   ({} series)", s.series.len())).id_salt(("volume.study", pi, si)).default_open(true).show(ui, |ui| {
                                for (sei, se) in s.series.iter().enumerate() {
                                    if !self.series_shown(&c, &patient_text, se) {
                                        continue;
                                    }
                                    self.series_rows(ui, &c, (pi, si, sei));
                                }
                            });
                            file_menu(&r.header_response, s.series.iter().find_map(|se| series_file(&c, se)));
                            if r.header_response.clicked() {
                                self.selection = Selection::Study(pi, si);
                            }
                        }
                        if !p.other_volumes.is_empty() {
                            ui.weak("Other files with this patient's identifiers");
                            for &k in &p.other_volumes {
                                if self.kind_shown(c.volumes[k].kind) {
                                    self.volume_row(ui, &c, k, None);
                                }
                            }
                        }
                    });
                    file_menu(&r.header_response, p.studies.iter().flat_map(|s| &s.series).find_map(|se| series_file(&c, se)));
                    if r.header_response.clicked() {
                        self.selection = Selection::Patient(pi);
                    }
                }
            });
            // Other files, by folder.
            let mut folders: BTreeMap<String, Vec<usize>> = BTreeMap::new();
            for &k in &c.files {
                let v = &c.volumes[k];
                if !self.kind_shown(v.kind) || !self.search_matches(&[&v.name, &v.files[0].to_string_lossy()]) {
                    continue;
                }
                let dir = v.files[0].parent().map(|d| relative(d, &c.roots)).unwrap_or_default();
                folders.entry(dir).or_default().push(k);
            }
            let shown: usize = folders.values().map(|v| v.len()).sum();
            egui::CollapsingHeader::new(egui::RichText::new(format!("Other files — {shown} volume(s)")).strong()).id_salt("volume.files").default_open(true).show(ui, |ui| {
                if folders.is_empty() {
                    ui.weak("No NIfTI, Analyze or NRRD volumes (with these filters).");
                }
                let open = folders.len() <= 12;
                for (dir, ks) in &folders {
                    let r = egui::CollapsingHeader::new(format!("{}   ({})", self.privacy.path(dir), ks.len())).id_salt(("volume.folder", dir)).default_open(open).show(ui, |ui| {
                        for &k in ks {
                            self.volume_row(ui, &c, k, None);
                        }
                    });
                    file_menu(&r.header_response, c.volumes[ks[0]].files.first().and_then(|f| f.parent()));
                }
            });
            if !c.recognised.is_empty() {
                egui::CollapsingHeader::new(format!("Not readable yet — {}", c.recognised.len())).id_salt("volume.recognised").show(ui, |ui| {
                    for r in &c.recognised {
                        ui.label(format!("{}   {}", self.privacy.path(&relative(&r.path, &c.roots)), r.format));
                    }
                });
            }
            if !c.problems.is_empty() {
                egui::CollapsingHeader::new(egui::RichText::new(format!("Could not read — {}", c.problems.len())).color(WARN)).id_salt("volume.problems").show(ui, |ui| {
                    for p in &c.problems {
                        ui.label(format!("{}   {}", self.privacy.path(&relative(&p.path, &c.roots)), self.privacy.path(&p.message)));
                    }
                });
            }
            egui::CollapsingHeader::new(format!("Not imaging — {} file(s)", format_count(c.skipped.file_count()))).id_salt("volume.skipped").show(ui, |ui| {
                let mut by: Vec<(&String, &usize)> = c.skipped.by_extension.iter().collect();
                by.sort_by(|a, b| b.1.cmp(a.1));
                ui.label(by.iter().map(|(e, n)| format!("{} ×{n}", if e.is_empty() { "(no extension)" } else { e.as_str() })).collect::<Vec<_>>().join(",  "));
            });
        });
    }

    fn series_shown(&self, c: &Catalog, patient: &[&str], se: &mm_volume::model::Series) -> bool {
        let kinds = se.volumes.iter().any(|&k| self.kind_shown(c.volumes[k].kind)) || (se.volumes.is_empty() && self.show.other);
        let mut texts: Vec<&str> = vec![&se.description, &se.modality];
        texts.extend_from_slice(patient);
        kinds && self.search_matches(&texts)
    }

    fn series_rows(&mut self, ui: &mut egui::Ui, c: &Catalog, (pi, si, sei): (usize, usize, usize)) {
        let se = &c.patients[pi].studies[si].series[sei];
        let number = se.number.map(|n| n.to_string()).unwrap_or_default();
        let desc = if se.description.is_empty() { "(no description)".to_string() } else { se.description.clone() };
        if se.volumes.len() == 1 && se.objects.is_empty() {
            self.volume_row(ui, c, se.volumes[0], Some(format!("{number:>4}  {desc}")));
            return;
        }
        let selected = self.selection == Selection::Series(pi, si, sei);
        let extra = if se.volumes.is_empty() { format!("{} object(s)", se.objects.len()) } else { format!("{} volumes", se.volumes.len()) };
        let r = ui.selectable_label(selected, format!("{number:>4}  {desc}   · {} · {} files · {extra}", se.modality, se.instances));
        file_menu(&r, series_file(c, se));
        if r.clicked() {
            self.selection = Selection::Series(pi, si, sei);
            if let Some(&k) = se.volumes.iter().find(|&&k| self.kind_shown(c.volumes[k].kind)).or(se.volumes.first()) {
                self.select_volume(k, ui.ctx());
            }
        }
        ui.indent(("volume.series", pi, si, sei), |ui| {
            for &k in &se.volumes {
                if self.kind_shown(c.volumes[k].kind) {
                    self.volume_row(ui, c, k, None);
                }
            }
            if !se.objects.is_empty() {
                let mut kinds: BTreeMap<&str, usize> = BTreeMap::new();
                for o in &se.objects {
                    *kinds.entry(o.what.as_str()).or_default() += 1;
                }
                ui.weak(kinds.iter().map(|(w, n)| format!("{n} × {w}")).collect::<Vec<_>>().join(", "));
            }
        });
    }

    fn viewer_ui(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        let Some((c, k, label)) = self.current_volume() else {
            ui.weak("Select an image in the browser to view it here.");
            ui.allocate_space(ui.available_size());
            return;
        };
        self.viewer.show(k, &c.volumes[k], &ctx);
        if !self.viewer_choices.0.as_ref().is_some_and(|key| key.is(&c, k, self.hide_ids)) {
            let key = ChoicesKey { catalog: Arc::downgrade(&c), volume: k, hide_ids: self.hide_ids };
            self.viewer_choices = (Some(key), self.overlay_choices(&c.volumes[k]));
        }
        // Its errors name files, whose paths may hold patients' names.
        if let Some(pick) = self.viewer.ui_with(ui, &label, &self.viewer_choices.1, &|e| self.privacy.path(e)) {
            let (v, l) = self.viewer_choices.1[pick].clone();
            self.viewer.compare_with(v, l, &ctx);
        }
    }

    fn details(&mut self, ui: &mut egui::Ui) {
        let Some(c) = self.catalog.clone() else { return };
        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| match self.selection {
            Selection::None => {
                ui.weak("Select a series or file to see its details and a slice.");
            }
            Selection::Patient(pi) => {
                let p = &c.patients[pi];
                section(ui, if self.hide_ids { "Patient".to_string() } else { p.name.clone() }.as_str());
                field_grid(ui, "volume.patient_fields", &[("Name", self.privacy.id(&p.name)), ("ID", self.privacy.id(&p.id)), ("Sex", self.privacy.id(&p.sex)), ("Birth date", self.privacy.id(&date(&p.birth_date))), ("Studies", p.studies.len().to_string())]);
            }
            Selection::Study(pi, si) => {
                let s = &c.patients[pi].studies[si];
                section(ui, "Study");
                field_grid(ui, "volume.study_fields", &[("Date", date_time(&s.date, &s.time)), ("Description", s.description.clone()), ("Accession", self.privacy.id(&s.accession)), ("Series", s.series.len().to_string())]);
            }
            Selection::Series(pi, si, sei) => {
                let se = &c.patients[pi].studies[si].series[sei];
                section(ui, &format!("Series {} · {}", se.number.map(|n| n.to_string()).unwrap_or_default(), se.description));
                ui.label(format!("{} · {} files · {} volume(s)", se.modality, se.instances, se.volumes.len()));
                for o in se.objects.iter().take(1) {
                    ui.weak(format!("Also: {} (and others like it)", o.what));
                }
                if let Some(p) = &self.preview {
                    let k = p.volume;
                    if se.volumes.contains(&k) {
                        self.volume_details(ui, &c, k);
                    }
                }
            }
            Selection::Volume(k) => self.volume_details(ui, &c, k),
        });
    }

    fn volume_details(&mut self, ui: &mut egui::Ui, c: &Arc<Catalog>, k: usize) {
        let v = &c.volumes[k];
        section(ui, &self.privacy.path(&v.name));
        ui.label(format!("{} · {}", kind_label(v.kind), v.format.label()));
        for w in &v.warnings {
            ui.colored_label(WARN, format!("⚠ {w}"));
        }
        ui.add_space(4.0);
        let crop = self.prepare.crop_box(v);
        if let Some(p) = &mut self.preview {
            if p.volume == k {
                p.ui(ui, v, false, crop);
            }
        }
        ui.add_space(6.0);
        let privacy = self.privacy.clone();
        self.prepare.ui(ui, c, k, &privacy);
        ui.add_space(4.0);
        let groups = ["Patient", "Study", "Series", "Equipment", "Acquisition", "Geometry", "Pixel", "Segmentation", "File"];
        for g in groups {
            let rows: Vec<(String, String)> = v.fields.iter().filter(|f| f.group == g).map(|f| (f.name.clone(), self.privacy.field(f))).collect();
            if rows.is_empty() {
                continue;
            }
            egui::CollapsingHeader::new(g).id_salt(("volume.fields", g)).default_open(g != "File").show(ui, |ui| {
                let rows: Vec<(&str, String)> = rows.iter().map(|(a, b)| (a.as_str(), b.clone())).collect();
                field_grid(ui, ("volume.grid", g), &rows);
            });
        }
        egui::CollapsingHeader::new(format!("Files ({}, {})", v.files.len(), bytes(v.bytes))).id_salt("volume.file_list").show(ui, |ui| {
            for f in v.files.iter().take(200) {
                ui.label(egui::RichText::new(self.privacy.path(&relative(f, &c.roots))).small());
            }
            if v.files.len() > 200 {
                ui.weak(format!("… and {} more", v.files.len() - 200));
            }
        });
        ui.horizontal(|ui| {
            if ui.button("Show in Finder").clicked() {
                polymix_workspace::guard::reveal(&v.files[0]);
            }
            if ui.button("Copy path").clicked() {
                ui.ctx().copy_text(v.files[0].display().to_string());
            }
            if ui.button("Copy header").on_hover_text("All the fields above as text (identifiers masked when hidden)").clicked() {
                let text: String = v.fields.iter().map(|f| format!("{} / {}: {}\n", f.group, f.name, self.privacy.field(f))).collect();
                ui.ctx().copy_text(format!("{}\n{}", self.privacy.path(&v.name), text));
            }
        });
    }
}

fn kind_label(k: Kind) -> &'static str {
    k.label()
}

/// Right-click menu of a catalog row: show its file (or folder) in Finder, copy its path.
fn file_menu(r: &egui::Response, path: Option<&Path>) {
    let Some(path) = path else { return };
    r.context_menu(|ui| {
        if ui.button("Show in Finder").clicked() {
            polymix_workspace::guard::reveal(path);
            ui.close();
        }
        if ui.button("Copy path").clicked() {
            ui.ctx().copy_text(path.display().to_string());
            ui.close();
        }
    });
}

/// A file of a DICOM series: its first volume's first file, or its first other object.
fn series_file<'a>(c: &'a Catalog, se: &'a mm_volume::model::Series) -> Option<&'a Path> {
    se.volumes.iter().find_map(|&k| c.volumes[k].files.first()).or_else(|| se.objects.first().map(|o| &o.path)).map(PathBuf::as_path)
}

/// Whether two catalog entries (of two scans) are the same image: the same name, files and
/// size, and the same pixels on the same grid. What was loaded or decoded from one stands for
/// the other. Real numbers (grid, value scaling) only need to agree to a billionth: the first
/// scan that sees a file reads them from its header, later ones from the scan cache, and its
/// JSON does not give every f64 back to the last bit.
fn same_image(a: &Volume, b: &Volume) -> bool {
    use mm_volume::model::Voxels;
    let close = |x: f64, y: f64| x == y || (x - y).abs() <= 1e-9 * x.abs().max(y.abs()).max(1.0);
    let all_close = |x: &[f64], y: &[f64]| x.len() == y.len() && x.iter().zip(y).all(|(x, y)| close(*x, *y));
    let grid = match (&a.geometry, &b.geometry) {
        (Some(g), Some(h)) => g.dims == h.dims && all_close(&g.spacing, &h.spacing) && all_close(g.affine.as_flattened(), h.affine.as_flattened()),
        (g, h) => g.is_none() && h.is_none(),
    };
    let pixels = match (&a.voxels, &b.voxels) {
        (
            Voxels::Raw { file, file_offset, gzip, skip, interleave, dtype, big_endian, slope, intercept },
            Voxels::Raw { file: file_b, file_offset: file_offset_b, gzip: gzip_b, skip: skip_b, interleave: interleave_b, dtype: dtype_b, big_endian: big_endian_b, slope: slope_b, intercept: intercept_b },
        ) => (file, file_offset, gzip, skip, interleave, dtype, big_endian) == (file_b, file_offset_b, gzip_b, skip_b, interleave_b, dtype_b, big_endian_b) && close(*slope, *slope_b) && close(*intercept, *intercept_b),
        (p, q) => p == q,
    };
    a.name == b.name && a.bytes == b.bytes && a.files == b.files && grid && pixels
}

fn field_grid(ui: &mut egui::Ui, id: impl std::hash::Hash + std::fmt::Debug, rows: &[(&str, String)]) {
    egui::Grid::new(id).num_columns(2).spacing([12.0, 3.0]).striped(true).show(ui, |ui| {
        for (a, b) in rows {
            if b.is_empty() {
                continue;
            }
            ui.weak(*a);
            ui.add(egui::Label::new(b).wrap());
            ui.end_row();
        }
    });
}

/// `20240102` → `2024-01-02`.
fn date(d: &str) -> String {
    if d.len() == 8 && d.bytes().all(|b| b.is_ascii_digit()) { format!("{}-{}-{}", &d[..4], &d[4..6], &d[6..]) } else { d.to_string() }
}

/// `20240102`, `102024.5` → `2024-01-02 10:20`.
fn date_time(d: &str, t: &str) -> String {
    let t = if t.len() >= 4 && t.bytes().take(4).all(|b| b.is_ascii_digit()) { format!(" {}:{}", &t[..2], &t[2..4]) } else { String::new() };
    let s = format!("{}{t}", date(d));
    if s.trim().is_empty() { "(no date)".into() } else { s.trim().to_string() }
}

fn bytes(n: u64) -> String {
    let n = n as f64;
    if n >= 1e9 {
        format!("{:.1} GB", n / 1e9)
    } else if n >= 1e6 {
        format!("{:.1} MB", n / 1e6)
    } else {
        format!("{:.0} KB", n / 1e3)
    }
}

/// `path` relative to the root it is under (with the root's own name first).
fn relative(path: &Path, roots: &[PathBuf]) -> String {
    for r in roots {
        if let Ok(rest) = path.strip_prefix(r) {
            let name = r.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
            let rest = rest.to_string_lossy();
            return if rest.is_empty() { name } else { format!("{name}/{rest}") };
        }
    }
    path.display().to_string()
}

/// DICOM files often have no extension (`IM_0001`, `DICOM`): look for "DICM" at byte 128.
fn has_dicom_magic(path: &Path) -> bool {
    use std::io::Read;
    let mut head = [0u8; 132];
    std::fs::File::open(path).and_then(|mut f| f.read_exact(&mut head)).is_ok() && &head[128..132] == b"DICM"
}

impl Workspace for VolumeWorkspace {
    fn name(&self) -> &'static str {
        "Volume"
    }

    fn tick(&mut self, ctx: &egui::Context, active: bool) {
        self.viewer.tick();
        if self.scan_soon && self.scan.is_none() {
            self.start_scan(ctx);
        }
        if let Some(r) = self.scan.as_ref().and_then(|s| s.poll()) {
            self.scan = None;
            self.finish_scan(r);
            if let Some(p) = self.open_select.take() {
                let found = self.catalog.as_ref().and_then(|c| c.volumes.iter().position(|v| v.files.contains(&p)));
                if let Some(k) = found {
                    self.selection = Selection::Volume(k);
                }
            }
            // The thumbnail of the volume selected (a new one if the scan changed it).
            if let Selection::Volume(k) = self.selection {
                self.select_volume(k, ctx);
            }
        }
        if let (Some(p), Some(c)) = (&mut self.preview, &self.catalog) {
            if let Some(v) = c.volumes.get(p.volume) {
                p.poll(v, ctx);
            }
        }
        // Derived images were saved: list them (their folder becomes a source), select the first.
        if let Some(paths) = self.prepare.poll().filter(|p| !p.is_empty()) {
            self.add_derived(paths, true, ctx);
        }
        if active && self.scan.is_none() && !self.prepare.busy() && !self.preview.as_ref().is_some_and(|p| p.busy()) {
            if let Some(step) = self.script.pop_front() {
                self.run_script_step(step, ctx);
                ctx.request_repaint();
            }
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::Panel::top("volume.toolbar").show(ui, |ui| self.toolbar(ui));
        egui::Panel::left("volume.sources").resizable(true).default_size(230.0).show(ui, |ui| {
            egui::ScrollArea::vertical().id_salt("volume.sources_scroll").show(ui, |ui| self.sources(ui));
        });
        if self.catalog.is_some() {
            egui::Panel::right("volume.details").resizable(true).default_size(400.0).show(ui, |ui| self.details(ui));
        }
        if self.viewer.open {
            egui::Panel::bottom("volume.viewer").resizable(true).default_size(420.0).min_size(220.0).show(ui, |ui| self.viewer_ui(ui));
        }
        egui::CentralPanel::default().show(ui, |ui| self.browser(ui));
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        storage.set_string("volume.roots", self.roots.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join("\n"));
        storage.set_string("volume.hide_ids", if self.hide_ids { "1" } else { "0" }.into());
    }

    fn title(&self) -> String {
        match &self.catalog {
            Some(c) => format!("Polymix — Volume ({} patients, {} other volumes)", c.patients.len(), c.files.len()),
            None => "Polymix — Volume".into(),
        }
    }

    fn accepts(&self, path: &Path) -> bool {
        path.is_dir()
            || path.file_name().is_some_and(|n| n.eq_ignore_ascii_case("DICOMDIR"))
            || path.extension().and_then(|e| e.to_str()).is_some_and(|e| VOLUME_EXTENSIONS.iter().any(|x| x.eq_ignore_ascii_case(e)))
            || has_dicom_magic(path)
    }

    fn open(&mut self, paths: Vec<PathBuf>, ctx: &egui::Context) {
        self.add_roots(paths, ctx);
    }

    fn script_push(&mut self, step: String) {
        self.script.push_back(step);
    }

    fn script_idle(&self) -> bool {
        self.script.is_empty() && self.scan.is_none() && !self.scan_soon && !self.preview.as_ref().is_some_and(|p| p.busy()) && !self.prepare.busy() && !self.viewer.busy()
    }

    fn agent_tools() -> Vec<polymix_workspace::agent::AgentTool> {
        agent::tools()
    }

    fn agent_call(&mut self, tool: &str, args: &serde_json::Value, _ctx: &egui::Context) -> polymix_workspace::agent::AgentReply {
        agent::call(self, tool, args)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dates_and_paths() {
        assert_eq!(date("20240102"), "2024-01-02");
        assert_eq!(date_time("20240102", "102024.75"), "2024-01-02 10:20");
        assert_eq!(date_time("", ""), "(no date)");
        assert_eq!(relative(Path::new("/a/My MRI/7T/x.nii"), &[PathBuf::from("/a/My MRI")]), "My MRI/7T/x.nii");
        assert_eq!(bytes(23_894_278), "23.9 MB");
    }

    /// Every text drawn in `out`.
    fn texts(out: &egui::FullOutput) -> Vec<String> {
        fn walk(s: &egui::Shape, into: &mut Vec<String>) {
            match s {
                egui::Shape::Text(t) => into.push(t.galley.text().to_string()),
                egui::Shape::Vec(v) => v.iter().for_each(|s| walk(s, into)),
                _ => {}
            }
        }
        let mut into = Vec::new();
        out.shapes.iter().for_each(|c| walk(&c.shape, &mut into));
        into
    }

    #[test]
    fn right_clicking_a_row_offers_show_in_finder() {
        let mut c = Catalog::default();
        c.volumes.push(entry("brain", PathBuf::from("/nonexistent/polymix-menu-test/brain.nii"), Kind::Image, None));
        c.files.push(0);
        let ctx = egui::Context::default();
        let mut ws = VolumeWorkspace::new(&ctx, None);
        ws.catalog = Some(Arc::new(c));
        let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(900.0, 700.0));
        let mut frame = |events: Vec<egui::Event>| {
            let input = egui::RawInput { screen_rect: Some(screen), events, ..Default::default() };
            ctx.run_ui(input, |ui| ws.browser(ui))
        };
        // Where the row's name is drawn.
        fn find(s: &egui::Shape, pos: egui::Pos2, text: &str) -> Option<egui::Pos2> {
            match s {
                egui::Shape::Text(t) if t.galley.text().starts_with(text) => Some(pos + t.pos.to_vec2() + egui::vec2(4.0, t.galley.rect.height() / 2.0)),
                egui::Shape::Vec(v) => v.iter().find_map(|s| find(s, pos, text)),
                _ => None,
            }
        }
        let out = frame(Vec::new());
        let at = out.shapes.iter().find_map(|c| find(&c.shape, egui::Pos2::ZERO, "brain")).expect("the row is drawn");
        assert!(!texts(&out).iter().any(|t| t == "Show in Finder"));
        let button = |pressed| egui::Event::PointerButton { pos: at, button: egui::PointerButton::Secondary, pressed, modifiers: Default::default() };
        for events in [vec![egui::Event::PointerMoved(at)], vec![button(true)], vec![button(false)]] {
            frame(events);
        }
        let shown = texts(&frame(Vec::new()));
        assert!(shown.iter().any(|t| t == "Show in Finder") && shown.iter().any(|t| t == "Copy path"), "{shown:?}");
    }

    #[test]
    fn the_viewers_errors_hide_identifiers() {
        use mm_volume::geometry::Geometry;
        use mm_volume::model::{Format, Patient, PatientKey, Voxels};
        use std::time::{Duration, Instant};
        // A DICOM series whose file is gone: the loader's error names its path.
        let file = PathBuf::from("/nonexistent/polymix-viewer-test/Doe_Jane/IM0001");
        let mut c = Catalog::default();
        c.patients.push(Patient { key: PatientKey { id: "MRN12345".into(), name: "DOE^JANE^Q".into() }, name: "DOE JANE Q".into(), id: "MRN12345".into(), birth_date: "19800101".into(), sex: "F".into(), studies: vec![], other_volumes: vec![] });
        c.volumes.push(Volume {
            format: Format::Dicom,
            kind: Kind::Image,
            name: "T1".into(),
            files: vec![file.clone()],
            bytes: 0,
            geometry: Some(Geometry::from_axes([2, 2, 1, 1], [1.0; 3], [glam::DVec3::X, glam::DVec3::Y, glam::DVec3::Z], glam::DVec3::ZERO)),
            data_type: None,
            fields: Vec::new(),
            warnings: Vec::new(),
            voxels: Voxels::Dicom { slices: vec![(file, 0)] },
            window: None,
            photometric: None,
            patient: None,
        });
        let ctx = egui::Context::default();
        let mut ws = VolumeWorkspace::new(&ctx, None);
        ws.catalog = Some(Arc::new(c));
        ws.set_hide(true);
        ws.selection = Selection::Volume(0);
        let frame = |ws: &mut VolumeWorkspace| texts(&ctx.run_ui(egui::RawInput::default(), |ui| ws.viewer_ui(ui)));
        frame(&mut ws);
        let start = Instant::now();
        while ws.viewer.busy() {
            assert!(start.elapsed() < Duration::from_secs(20), "the load finishes");
            std::thread::sleep(Duration::from_millis(1));
            ws.viewer.tick();
        }
        let error = ws.viewer.error.clone().expect("the load failed");
        assert!(error.contains("/Doe_Jane/IM0001"), "{error}");
        // Drawn masked while identifiers are hidden, as it is otherwise.
        let shown = frame(&mut ws);
        assert!(shown.iter().any(|t| t.contains("/•••_•••/IM0001")), "{shown:?}");
        assert!(!shown.iter().any(|t| t.to_lowercase().contains("doe") || t.to_lowercase().contains("jane")), "{shown:?}");
        ws.set_hide(false);
        assert!(frame(&mut ws).iter().any(|t| t.contains("/Doe_Jane/IM0001")));
    }

    use glam::DVec3;
    use mm_volume::geometry::Geometry;
    use mm_volume::load::LoadedVolume;
    use mm_volume::model::{Format, Patient, PatientKey, Voxels};

    /// A catalog entry (its pixels unreadable: nothing here loads files) on `geometry`.
    fn entry(name: &str, file: PathBuf, kind: Kind, geometry: Option<Geometry>) -> Volume {
        Volume {
            format: Format::Nifti1,
            kind,
            name: name.into(),
            files: vec![file],
            bytes: 1000,
            geometry,
            data_type: None,
            fields: Vec::new(),
            warnings: Vec::new(),
            voxels: Voxels::Unsupported("test".into()),
            window: None,
            photometric: None,
            patient: None,
        }
    }

    /// A grid of `dims` voxels of `mm`, turned by `turn` radians about z, at `origin`.
    fn grid(dims: [usize; 3], mm: f64, turn: f64, origin: [f64; 3]) -> Geometry {
        let (s, c) = turn.sin_cos();
        Geometry::from_axes([dims[0], dims[1], dims[2], 1], [mm; 3], [DVec3::new(c, s, 0.0), DVec3::new(-s, c, 0.0), DVec3::Z], DVec3::from(origin))
    }

    /// `overlay_choices` as it was before the derived folder was looked up once and the
    /// corners folded without a list: the reference the new one must match.
    fn overlay_choices_before(ws: &VolumeWorkspace, current: &Volume) -> Vec<(Volume, String)> {
        let (Some(c), Some(g)) = (&ws.catalog, &current.geometry) else { return Vec::new() };
        let bounds = |g: &mm_volume::geometry::Geometry| {
            let d = g.dims.map(|n| n as f64 - 0.5);
            let pts: Vec<glam::DVec3> = (0..8).map(|c| g.world(glam::DVec3::new(if c & 1 > 0 { d[0] } else { -0.5 }, if c & 2 > 0 { d[1] } else { -0.5 }, if c & 4 > 0 { d[2] } else { -0.5 }))).collect();
            (pts.iter().fold(glam::DVec3::INFINITY, |a, p| a.min(*p)), pts.iter().fold(glam::DVec3::NEG_INFINITY, |a, p| a.max(*p)))
        };
        let vol = |b: &(glam::DVec3, glam::DVec3)| (b.1 - b.0).max(glam::DVec3::ZERO).element_product();
        let here = bounds(g);
        let mut out: Vec<(Volume, String)> = c
            .volumes
            .iter()
            .filter(|v| !(v.name == current.name && v.files == current.files) && matches!(v.kind, Kind::Image | Kind::Derived | Kind::LabelMap))
            .filter_map(|v| {
                let b = bounds(v.geometry.as_ref()?);
                let common = (here.0.max(b.0), here.1.min(b.1));
                (vol(&common) >= 0.5 * vol(&here).min(vol(&b))).then(|| (v.clone(), ws.privacy.path(&v.name)))
            })
            .take(200)
            .collect();
        let derived = |v: &Volume| v.files.first().is_some_and(|f| polymix_workspace::derived_dir().is_some_and(|d| f.starts_with(d)));
        out.sort_by_key(|(v, _)| !derived(v));
        out
    }

    /// A catalog of `n` images of patient Doe, some derived, some elsewhere, turned or
    /// without a grid, of every kind (a small generator: the same each run).
    fn mixed_catalog(n: usize) -> Catalog {
        let derived = polymix_workspace::derived_dir().expect("a data folder");
        let mut c = Catalog::default();
        c.patients.push(Patient { key: PatientKey { id: "MRN12345".into(), name: "DOE^JANE".into() }, name: "DOE JANE".into(), id: "MRN12345".into(), birth_date: String::new(), sex: String::new(), studies: vec![], other_volumes: vec![] });
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move |m: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % m
        };
        let kinds = [Kind::Image, Kind::Image, Kind::Derived, Kind::LabelMap, Kind::Segmentation, Kind::TimeSeries, Kind::Localizer];
        for i in 0..n {
            let file = if next(4) == 0 { derived.join(format!("doe_jane_{i} N4.nii.gz")) } else { PathBuf::from(format!("/synthetic/Doe_Jane/scan_{i}.nii.gz")) };
            let geometry = (next(10) != 0).then(|| {
                let shift = [next(200) as f64 - 100.0, next(200) as f64 - 100.0, next(100) as f64 - 50.0];
                grid([64 + next(64) as usize, 64 + next(64) as usize, 20 + next(40) as usize], 0.5 + next(3) as f64 * 0.5, next(4) as f64 * 0.3, shift)
            });
            c.volumes.push(entry(&format!("Doe_Jane {i}"), file, kinds[next(kinds.len() as u64) as usize], geometry));
        }
        c
    }

    #[test]
    fn overlay_choices_are_as_before() {
        let ctx = egui::Context::default();
        let mut ws = VolumeWorkspace::new(&ctx, None);
        let (mut longest, mut reordered) = (0, 0);
        for n in [0, 1, 7, 60, 1000] {
            ws.catalog = Some(Arc::new(mixed_catalog(n)));
            for hide in [false, true] {
                ws.set_hide(hide);
                let c = ws.catalog.clone().expect("set");
                let everywhere = entry("everywhere", "/x.nii".into(), Kind::Image, Some(grid([400, 400, 200], 1.0, 0.0, [-200.0, -200.0, -100.0])));
                for current in c.volumes.iter().step_by(1 + n / 100).chain([&everywhere]) {
                    let now = ws.overlay_choices(current);
                    assert_eq!(now, overlay_choices_before(&ws, current), "{n} volumes, hide {hide}, current {}", current.name);
                    longest = longest.max(now.len());
                    let at = |v: &Volume| c.volumes.iter().position(|w| w.name == v.name);
                    reordered += usize::from(now.windows(2).any(|w| at(&w[0].0) > at(&w[1].0)));
                }
            }
        }
        // The cases compared: lists cut at 200, and lists the sort reordered.
        assert_eq!(longest, 200);
        assert!(reordered > 20, "{reordered}");
    }

    #[test]
    fn the_viewers_choices_follow_hide_identifiers_and_the_catalog() {
        let ctx = egui::Context::default();
        let mut ws = VolumeWorkspace::new(&ctx, None);
        let mut c = mixed_catalog(12);
        for v in &mut c.volumes {
            v.kind = Kind::Image;
            v.geometry = Some(grid([10, 10, 10], 1.0, 0.0, [0.0; 3]));
        }
        ws.catalog = Some(Arc::new(c.clone()));
        ws.selection = Selection::Volume(0);
        let frame = |ws: &mut VolumeWorkspace| {
            let _ = ctx.run_ui(egui::RawInput::default(), |ui| ws.viewer_ui(ui));
        };
        let labels = |ws: &VolumeWorkspace| ws.viewer_choices.1.iter().map(|(_, l)| l.clone()).collect::<Vec<_>>();
        frame(&mut ws);
        assert_eq!(labels(&ws).len(), 11);
        assert!(labels(&ws).iter().all(|l| l.contains("Doe_Jane")), "{:?}", labels(&ws));
        // Hidden: listed again, masked.
        ws.set_hide(true);
        frame(&mut ws);
        assert!(labels(&ws).iter().all(|l| l.contains("•••_••• ") && !l.contains("Doe")), "{:?}", labels(&ws));
        ws.set_hide(false);
        frame(&mut ws);
        assert!(labels(&ws).iter().all(|l| l.contains("Doe_Jane")));
        // A rescan with fewer images: a new list, even if the new catalog sat where the old one did.
        c.volumes.truncate(4);
        ws.catalog = Some(Arc::new(c));
        frame(&mut ws);
        assert_eq!(labels(&ws).len(), 3);
    }

    #[test]
    fn a_rescan_keeps_thumbnail_and_viewer_on_their_images() {
        let ctx = egui::Context::default();
        let mut ws = VolumeWorkspace::new(&ctx, None);
        let image = |name: &str| entry(name, PathBuf::from(format!("/synthetic/{name}.nii")), Kind::Image, Some(grid([2, 2, 2], 1.0, 0.0, [0.0; 3])));
        let (a, b, c, d) = (image("a"), image("b"), image("c"), image("d"));
        ws.finish_scan(Ok(Catalog { volumes: vec![a.clone(), b.clone()], ..Catalog::default() }));
        ws.selection = Selection::Volume(1);
        ws.select_volume(1, &ctx);
        let shown = Arc::new(LoadedVolume::from_values("b", [2, 2, 2], vec![0.0; 8], grid([2, 2, 2], 1.0, 0.0, [0.0; 3])));
        ws.viewer.volume = Some((1, shown.clone()));
        // Whether the viewer starts loading catalog image `k` when it is selected; then the
        // viewer is emptied.
        let viewer_loads = |ws: &mut VolumeWorkspace, k: usize| {
            let c = ws.catalog.clone().expect("a catalog");
            ws.viewer.show(k, &c.volumes[k], &ctx);
            let loads = ws.viewer.busy();
            ws.viewer = viewer::Viewer::default();
            loads
        };
        // Another image listed before it: both follow b to its new place.
        ws.finish_scan(Ok(Catalog { volumes: vec![c.clone(), a.clone(), b.clone()], ..Catalog::default() }));
        assert_eq!(ws.selection, Selection::Volume(2));
        assert_eq!(ws.preview.as_ref().map(|p| p.volume), Some(2));
        assert_eq!(ws.viewer.volume.as_ref().map(|(k, _)| *k), Some(2));
        assert!(!viewer_loads(&mut ws, 2), "b is shown: not loaded again");
        // b is gone and d took its place: the image shown is not taken for d.
        ws.viewer.volume = Some((2, shown.clone()));
        ws.finish_scan(Ok(Catalog { volumes: vec![c.clone(), a.clone(), d], ..Catalog::default() }));
        assert_eq!(ws.selection, Selection::None);
        assert!(ws.preview.is_none());
        assert!(viewer_loads(&mut ws, 2), "d is loaded, not b shown under its name");
        // b back, then changed (a bigger file): kept selected, with a new thumbnail and image.
        ws.finish_scan(Ok(Catalog { volumes: vec![a.clone(), b.clone()], ..Catalog::default() }));
        ws.selection = Selection::Volume(1);
        ws.select_volume(1, &ctx);
        ws.viewer.volume = Some((1, shown.clone()));
        let mut bigger = b;
        bigger.bytes += 1;
        ws.finish_scan(Ok(Catalog { volumes: vec![a.clone(), bigger], ..Catalog::default() }));
        assert_eq!(ws.selection, Selection::Volume(1));
        assert!(ws.preview.is_none(), "made again by the next tick");
        assert!(viewer_loads(&mut ws, 1));
        // No folders left: no index is trusted.
        ws.viewer.volume = Some((0, shown));
        ws.roots.clear();
        ws.start_scan(&ctx);
        assert!(ws.catalog.is_none() && ws.preview.is_none());
        ws.catalog = Some(Arc::new(Catalog { volumes: vec![a], ..Catalog::default() }));
        assert!(viewer_loads(&mut ws, 0));
    }

    #[test]
    fn a_rescan_from_the_scan_cache_keeps_thumbnail_and_viewer() {
        use mm_volume::scan::{scan, FileCache, ScanOptions};
        use mm_volume::write::{nifti, Data, Intent};
        use std::time::{Duration, Instant};
        let dir = std::env::temp_dir().join(format!("pv-rescan-cache-{}", std::process::id()));
        let (data, cache_path) = (dir.join("data"), dir.join("scan-cache.json"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&data).expect("temp folder");
        // Grids from f32 headers that the cache's JSON does not give back to the last bit when
        // widened to f64: an oblique one, origins with decimals, sizes like 0.4688 mm.
        let tilt = |a: f64, b: f64| {
            let (x, y) = (DVec3::new(a.cos(), a.sin(), 0.0), DVec3::new(-a.sin() * b.cos(), a.cos() * b.cos(), b.sin()));
            [x, y, x.cross(y)]
        };
        let grids = [
            ("axial", Geometry::from_axes([8, 8, 6, 1], [0.9375, 0.9375, 1.2], [DVec3::X, DVec3::Y, DVec3::Z], DVec3::new(-120.0, -95.2654, 33.7))),
            ("oblique", Geometry::from_axes([8, 8, 6, 1], [0.4688, 0.4688, 0.6], tilt(0.17, 0.11), DVec3::new(-101.3, -87.9, 12.25))),
            ("turned", Geometry::from_axes([8, 8, 6, 1], [1.5, 1.5, 1.5], tilt(0.03, 0.0), DVec3::new(-110.25, -90.57, -40.13))),
        ];
        let values: Vec<f32> = (0..8 * 8 * 6).map(|i| (i % 50) as f32).collect();
        for (name, g) in &grids {
            nifti(&data.join(format!("{name}.nii.gz")), Data::F32(&values), [8, 8, 6], g, "test", Intent::None).expect("written");
        }
        let rescan = || {
            let mut cache = FileCache::load(&cache_path);
            let c = scan(std::slice::from_ref(&data), &ScanOptions::default(), Some(&mut cache), &|_| true).expect("scanned");
            cache.save(&cache_path).expect("cache saved");
            c
        };
        // The first scan reads the headers, the next ones the cache.
        let (fresh, cached, cached_again) = (rescan(), rescan(), rescan());
        assert_eq!(fresh.volumes.len(), 3);
        let ctx = egui::Context::default();
        for k in 0..3 {
            let mut ws = VolumeWorkspace::new(&ctx, None);
            ws.finish_scan(Ok(fresh.clone()));
            ws.selection = Selection::Volume(k);
            ws.select_volume(k, &ctx);
            // The user's slice, to tell this thumbnail from a new one.
            ws.preview.as_mut().expect("a thumbnail").slice = 1;
            let c = ws.catalog.clone().expect("scanned");
            ws.viewer.show(k, &c.volumes[k], &ctx);
            let start = Instant::now();
            while ws.viewer.busy() {
                assert!(start.elapsed() < Duration::from_secs(20), "the load finishes");
                std::thread::sleep(Duration::from_millis(1));
                ws.viewer.tick();
            }
            assert_eq!(ws.viewer.shown_index(), Some(k), "{:?}", ws.viewer.error);
            for next in [&cached, &cached_again] {
                ws.finish_scan(Ok(next.clone()));
                // What the next tick does after a scan.
                ws.select_volume(k, &ctx);
                let name = &c.volumes[k].name;
                assert_eq!(ws.selection, Selection::Volume(k), "{name}");
                assert_eq!(ws.preview.as_ref().map(|p| (p.volume, p.slice)), Some((k, 1)), "{name}: the same thumbnail");
                let c = ws.catalog.clone().expect("scanned");
                ws.viewer.show(k, &c.volumes[k], &ctx);
                assert!(!ws.viewer.busy(), "{name}: shown, not loaded again");
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn same_image_allows_for_the_cache_but_not_for_changes() {
        let raw = |slope: f64, intercept: f64| Voxels::Raw { file: "/x.nii".into(), file_offset: 352, gzip: false, skip: 0, interleave: 1, dtype: mm_volume::model::DataType::I16, big_endian: false, slope, intercept };
        let a = Volume { voxels: raw(0.1, -1024.0), ..entry("a", "/x.nii".into(), Kind::Image, Some(grid([8, 8, 6], 0.4688, 0.17, [-101.3, -87.9, 12.25]))) };
        let with = |f: &dyn Fn(&mut Volume)| {
            let mut b = a.clone();
            f(&mut b);
            same_image(&a, &b)
        };
        let nudge = |x: &mut f64, ulps: i64| *x = f64::from_bits((x.to_bits() as i64 + ulps) as u64);
        assert!(with(&|_| {}));
        // A few units in the last place, as the cache's JSON may give back.
        assert!(with(&|v| v.geometry.as_mut().into_iter().for_each(|g| g.affine.iter_mut().flatten().for_each(|x| nudge(x, 3)))));
        assert!(with(&|v| v.geometry.as_mut().into_iter().for_each(|g| g.spacing.iter_mut().for_each(|x| nudge(x, -2)))));
        assert!(with(&|v| v.voxels = raw(f64::from_bits(0.1f64.to_bits() + 1), -1024.0 + 1e-12)));
        // Changes that matter are still seen.
        assert!(!with(&|v| v.geometry.as_mut().into_iter().for_each(|g| g.affine[0][3] += 1e-6)));
        assert!(!with(&|v| v.geometry.as_mut().into_iter().for_each(|g| g.spacing[2] *= 1.0 + 1e-7)));
        assert!(!with(&|v| v.geometry.as_mut().into_iter().for_each(|g| g.dims[2] += 1)));
        assert!(!with(&|v| v.geometry = None));
        assert!(!with(&|v| v.voxels = raw(0.1000001, -1024.0)));
        assert!(!with(&|v| v.voxels = raw(0.1, -1023.999)));
        assert!(!with(&|v| v.voxels = Voxels::Unsupported("test".into())));
        assert!(!with(&|v| v.bytes += 1));
        assert!(!with(&|v| v.name.push('2')));
    }
}
