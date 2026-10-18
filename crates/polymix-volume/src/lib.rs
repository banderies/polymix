//! The Volume workspace: find, organise and preview medical images.
//!
//! Add a folder and Polymix finds the imaging data in it (DICOM, NIfTI, Analyze, NRRD),
//! however messy the folder: DICOM is organised Patient → Study → Series like 3D Slicer's
//! DICOM browser, other files are listed by folder, and the selected volume shows its header
//! and a thumbnail. It depends only on `polymix-workspace` and `mm-volume`, never on the Mesh
//! workspace, so the two cannot break each other.

mod preview;
mod privacy;
mod scan_task;

use mm_volume::model::{Catalog, Kind};
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
    preview: Option<Preview>,
    script: VecDeque<String>,
    /// A file opened from outside (Finder, command line): select its volume after the scan.
    open_select: Option<PathBuf>,
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
            preview: None,
            script: VecDeque::new(),
            open_select: None,
        }
    }

    fn start_scan(&mut self, ctx: &egui::Context) {
        self.scan_soon = false;
        self.scan_error = None;
        if self.roots.is_empty() {
            self.catalog = None;
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
                    Selection::Volume(k) => self.catalog.as_ref().and_then(|old| old.volumes.get(k)).and_then(|v| c.volumes.iter().position(|w| w.files.first() == v.files.first() && w.name == v.name)).map(Selection::Volume),
                    _ => None,
                };
                if keep.is_none() {
                    self.preview = None;
                }
                self.selection = keep.unwrap_or(Selection::None);
                self.privacy = Privacy::new(self.hide_ids, Some(&c));
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
                        ui.spinner();
                        ui.label(format!("Listing files… {}", format_count(files)));
                    }
                    Some(Progress::Organising) => {
                        ui.spinner();
                        ui.label("Organising…");
                    }
                    None => {
                        ui.spinner();
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
                let name = r.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| r.display().to_string());
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
                    ui.spinner();
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
                    egui::CollapsingHeader::new(format!("{}   ({})", self.privacy.path(dir), ks.len())).id_salt(("volume.folder", dir)).default_open(open).show(ui, |ui| {
                        for &k in ks {
                            self.volume_row(ui, &c, k, None);
                        }
                    });
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
        if ui.selectable_label(selected, format!("{number:>4}  {desc}   · {} · {} files · {extra}", se.modality, se.instances)).clicked() {
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

    fn volume_details(&mut self, ui: &mut egui::Ui, c: &Catalog, k: usize) {
        let v = &c.volumes[k];
        section(ui, &self.privacy.path(&v.name));
        ui.label(format!("{} · {}", kind_label(v.kind), v.format.label()));
        for w in &v.warnings {
            ui.colored_label(WARN, format!("⚠ {w}"));
        }
        ui.add_space(4.0);
        if let Some(p) = &mut self.preview {
            if p.volume == k {
                p.ui(ui, v, false);
            }
        }
        ui.add_space(6.0);
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
                    self.select_volume(k, ctx);
                }
            }
        }
        if let (Some(p), Some(c)) = (&mut self.preview, &self.catalog) {
            if let Some(v) = c.volumes.get(p.volume) {
                p.poll(v, ctx);
            }
        }
        if active && self.scan.is_none() && !self.preview.as_ref().is_some_and(|p| p.busy()) {
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
        self.script.is_empty() && self.scan.is_none() && !self.scan_soon && !self.preview.as_ref().is_some_and(|p| p.busy())
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
}
