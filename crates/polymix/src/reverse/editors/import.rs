// SPDX-License-Identifier: AGPL-3.0-only
//! The Import STEP box: the file — its name and size, or none chosen — the key that reads one
//! (or another in its place), and what an import makes. The file's text is kept in the part
//! (`Part::step_files`, keyed by its digest: one file imported twice is kept once). Ported from
//! ScanRuler (`plugins/reverse/ui/FeatureEditor.tsx` the import's fields, `featureFields.tsx`
//! its preview line, `state/reFeatureSlice.ts` its defaults), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only.

use std::path::Path;

use egui::Pos2;
use mm_reverse::model::{step_file_key, FeatureKind, ImportFeature, StepFile, StepFileRef};

use crate::reverse::{PickWhat, ReverseWorkspace};
use crate::tools::Viewport;

/// Whether the editor is built: its button stands with the others.
pub(crate) const READY: bool = true;
/// The panel's button: its label and what it does.
pub(crate) const BUTTON: (&str, &str) = ("Import STEP", "A STEP file brought in as bodies of their own");
/// What a click picks first when the box opens.
pub(crate) const FIRST_PICK: PickWhat = PickWhat::Nothing;

/// A new import: no file chosen yet.
pub(crate) fn new_feature(_ws: &ReverseWorkspace) -> Result<FeatureKind, String> {
    Ok(FeatureKind::Import(ImportFeature { file: StepFileRef { key: String::new(), name: String::new(), size: 0 } }))
}

/// A file's size in words.
fn size_words(n: u64) -> String {
    if n >= 1 << 20 {
        format!("{:.1} MB", n as f64 / (1u64 << 20) as f64)
    } else if n >= 1 << 10 {
        format!("{:.0} kB", n as f64 / 1024.0)
    } else {
        format!("{n} B")
    }
}

/// A STEP file read into the part, kept by its digest, and the reference an import holds.
pub(crate) fn read_step(ws: &mut ReverseWorkspace, path: &Path) -> Result<StepFileRef, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("Could not read {}: {e}", path.display()))?;
    let name = path.file_name().map_or_else(|| "part.step".to_string(), |n| n.to_string_lossy().into_owned());
    let size = text.len() as u64;
    let key = step_file_key(&text);
    ws.part.step_files.entry(key.clone()).or_insert(StepFile { name: name.clone(), size, text });
    Ok(StepFileRef { key, name, size })
}

pub(crate) fn ui(ws: &mut ReverseWorkspace, ui: &mut egui::Ui, x: &mut ImportFeature, _pick: &mut PickWhat) -> bool {
    let mut changed = false;
    let chosen = !x.file.key.is_empty();
    ui.horizontal(|ui| {
        ui.label("File");
        if chosen {
            ui.strong(format!("{} · {}", x.file.name, size_words(x.file.size)));
        } else {
            ui.weak("No file chosen");
        }
    });
    let key = if chosen { "Another file…" } else { "Choose a STEP file…" };
    let mut button = ui.button(key);
    if chosen {
        button = button.on_hover_text("Read another STEP file in this one's place — the features after it keep their picks where they still find them");
    }
    if button.clicked() {
        if let Some(path) = rfd::FileDialog::new().add_filter("STEP", &["step", "stp", "STEP", "STP"]).pick_file() {
            match read_step(ws, &path) {
                Ok(file) => {
                    x.file = file;
                    changed = true;
                }
                Err(e) => ws.status = e,
            }
        }
    }
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing.x = 0.0;
        ui.weak("A body for every solid in the file, in the frame the file was drawn in, named as the file names its parts. Put them in place with ");
        ui.label(egui::RichText::new("Move body").strong());
        ui.weak(".");
    });
    // The preview line.
    let id = ws.preview_feature;
    if !chosen {
        ui.weak("Choose the STEP file to bring in.");
    } else if let Some(fid) = id.filter(|f| !ws.built.errors.contains_key(f)) {
        let made: Vec<&super::super::BodyView> = ws.bodies.iter().filter(|b| b.id.maker() == fid).collect();
        if !made.is_empty() {
            let volume: f64 = made.iter().map(|b| b.volume).sum();
            let surface = made.iter().any(|b| ws.built.body(b.id).is_some_and(|x| x.surface));
            ui.weak(format!("The file holds {:.1} mm³ of solid{}.", volume, if surface { ", and faces without an inside — a surface body" } else { "" }));
        }
    }
    changed
}

pub(crate) fn pick(_ws: &mut ReverseWorkspace, _vp: Viewport, _p: Pos2, _clicked: bool, _x: &mut ImportFeature, _pick: &mut PickWhat) -> bool {
    false
}

pub(crate) fn paint(_ws: &ReverseWorkspace, _painter: &egui::Painter, _vp: Viewport, _x: &ImportFeature) {}
