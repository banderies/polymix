//! The Volume workspace: find, organise and preview medical images.
//!
//! It depends only on `polymix-workspace` (and, for the data, `mm-volume`), never on the
//! Mesh workspace, so the two cannot break each other.

use polymix_workspace::Workspace;
use std::path::{Path, PathBuf};

/// File extensions the Volume workspace opens when dropped on the window. (DICOM files
/// often have none; folders are always accepted.)
const VOLUME_EXTENSIONS: [&str; 12] = ["nii", "gz", "dcm", "ima", "nrrd", "nhdr", "hdr", "img", "mha", "mhd", "mgz", "mgh"];

pub struct VolumeWorkspace {
    /// Folders and files to look at.
    roots: Vec<PathBuf>,
}

impl VolumeWorkspace {
    pub fn new(_ctx: &egui::Context, _storage: Option<&dyn eframe::Storage>) -> Self {
        Self { roots: Vec::new() }
    }
}

impl Workspace for VolumeWorkspace {
    fn name(&self) -> &'static str {
        "Volume"
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ui, |ui| {
            ui.heading("Volume");
            ui.label("Medical image browser: coming next.");
            for r in &self.roots {
                ui.monospace(r.display().to_string());
            }
        });
    }

    fn accepts(&self, path: &Path) -> bool {
        path.is_dir()
            || path.file_name().is_some_and(|n| n.eq_ignore_ascii_case("DICOMDIR"))
            || path.extension().and_then(|e| e.to_str()).is_some_and(|e| VOLUME_EXTENSIONS.iter().any(|x| x.eq_ignore_ascii_case(e)))
    }

    fn open(&mut self, paths: Vec<PathBuf>, _ctx: &egui::Context) {
        self.roots.extend(paths);
    }
}
