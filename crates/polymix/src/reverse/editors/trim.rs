// SPDX-License-Identifier: AGPL-3.0-only
//! The Trim feature's box: the surfaces that close into the solid — the workspace's fitted
//! surfaces and sheets, its solid bodies, and the measured planes, cylinders, cones, spheres
//! and tori — how near a face's samples must lie to the scan and over what share. Its
//! tolerance starts at three times the median σ of the fitted surfaces in the set. A set that
//! does not close has its open edges drawn red on the part. Ported from ScanRuler
//! (`plugins/reverse/ui/FeatureEditor.tsx` the trim's fields, `featureFields.tsx` its preview
//! line, `state/reFeatureSlice.ts` its defaults, `state/reDrafts.ts` trimToleranceFor),
//! © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.

use std::cell::Cell;

use egui::{Pos2, RichText};
use mm_reverse::model::{ElementGeometry, FeatureKind, TrimFeature, TrimTool};

use super::split::{DEFAULT_SPLIT_OVERLAP, DEFAULT_SPLIT_TOLERANCE_MM};
use super::surf_common::{mm, num, paint_lines, report, RED};
use crate::reverse::{PickWhat, ReverseWorkspace};
use crate::tools::Viewport;

/// Whether the editor is built: its button stands with the others.
pub(crate) const READY: bool = true;
/// The panel's button: its label and what it does.
pub(crate) const BUTTON: (&str, &str) = ("Trim", "Surfaces closed into a solid");
/// What a click picks first when the box opens.
pub(crate) const FIRST_PICK: PickWhat = PickWhat::Nothing;

thread_local! {
    /// Whether the open trim's tolerance was typed, rather than read off the surfaces' fits.
    static TYPED: Cell<bool> = const { Cell::new(false) };
}

/// The fitted surfaces' σ in a set, by the last build's reports.
fn sigmas(ws: &ReverseWorkspace, tools: &[TrimTool]) -> Vec<f64> {
    let mut s: Vec<f64> = tools
        .iter()
        .filter_map(|t| match t {
            TrimTool::Body { id } => ws.built.surf.get(&id.maker()).and_then(|r| r.surface.as_ref()).map(|s| s.rms),
            TrimTool::Element { .. } => None,
        })
        .filter(|s| s.is_finite() && *s > 0.0)
        .collect();
    s.sort_by(|a, b| a.total_cmp(b));
    s
}

/// A trim's tolerance off its set (ScanRuler's `trimToleranceFor`): three times the median σ of
/// the fitted surfaces in it, 0.05 to 0.5 mm, to a micron — 0.3 mm with none.
pub(crate) fn tolerance_for(ws: &ReverseWorkspace, tools: &[TrimTool]) -> f64 {
    let s = sigmas(ws, tools);
    if s.is_empty() {
        return DEFAULT_SPLIT_TOLERANCE_MM;
    }
    ((3.0 * s[s.len() >> 1]).clamp(0.05, 0.5) * 1000.0).round() / 1000.0
}

/// A new trim: every surface body there is — the set a part's surfaces make.
pub(crate) fn new_feature(ws: &ReverseWorkspace) -> Result<FeatureKind, String> {
    TYPED.with(|t| t.set(false));
    let tools: Vec<TrimTool> = ws.built.bodies.iter().filter(|b| b.surface).map(|b| TrimTool::Body { id: b.id }).collect();
    let tolerance = tolerance_for(ws, &tools);
    Ok(FeatureKind::Trim(TrimFeature { tools, tolerance, overlap: DEFAULT_SPLIT_OVERLAP, exprs: Default::default() }))
}

/// The elements a trim cuts space with.
fn solids(ws: &ReverseWorkspace) -> Vec<(u32, String, &'static str)> {
    ws.part
        .elements
        .iter()
        .filter(|e| matches!(e.geometry, ElementGeometry::Plane { .. } | ElementGeometry::Cylinder { .. } | ElementGeometry::Cone { .. } | ElementGeometry::Sphere { .. } | ElementGeometry::Torus { .. }))
        .map(|e| (e.id, e.name.clone(), e.geometry.kind()))
        .collect()
}

pub(crate) fn ui(ws: &mut ReverseWorkspace, ui: &mut egui::Ui, x: &mut TrimFeature, _pick: &mut PickWhat) -> bool {
    let mut changed = false;
    let own = ws.preview_feature;
    let surfaces: Vec<(mm_reverse::model::BodyId, String, bool)> = ws.pick_set().iter().filter(|b| Some(b.id.maker()) != own).map(|b| (b.id, b.name.clone(), ws.built.body(b.id).is_some_and(|x| x.surface))).collect();
    let elements = solids(ws);
    ui.horizontal(|ui| {
        ui.label("Surfaces").on_hover_text(
            "The set that closes into the solid: fitted surfaces and sheets from this workspace, and measured planes, cylinders, cones and spheres, each reaching past the part. A box round the scan is cut by all of them, and of the faces that makes, the ones lying on the scan — within the tolerance, over the overlap share of their samples — are kept and sewn. A solid body's faces count too.\n\nA set that does not close leaves a surface body of the kept faces with its open edges drawn red on the part: fit the surface missing there and edit the trim to take it in.",
        );
        ui.label(RichText::new(format!("{} of {}", x.tools.len(), surfaces.len() + elements.len())).strong().size(12.0));
    });
    let mut toggled = false;
    egui::ScrollArea::vertical().id_salt("re_trim_tools").max_height(180.0).show(ui, |ui| {
        for (id, name, surface) in &surfaces {
            let t = TrimTool::Body { id: *id };
            let mut on = x.tools.contains(&t);
            let sigma = ws.built.surf.get(&id.maker()).and_then(|r| r.surface.as_ref()).map(|s| s.rms);
            let label = match (sigma, surface) {
                (Some(s), _) => format!("{name} · σ {}", mm(s, 3)),
                (None, true) => name.clone(),
                (None, false) => format!("{name} · solid"),
            };
            if ui.checkbox(&mut on, label).changed() {
                toggled = true;
                if on {
                    x.tools.push(t);
                } else {
                    x.tools.retain(|o| *o != t);
                }
            }
        }
        for (id, name, kind) in &elements {
            let t = TrimTool::Element { id: *id };
            let mut on = x.tools.contains(&t);
            if ui.checkbox(&mut on, format!("{name} · {kind}")).changed() {
                toggled = true;
                if on {
                    x.tools.push(t);
                } else {
                    x.tools.retain(|o| *o != t);
                }
            }
        }
        if surfaces.len() + elements.len() == 0 {
            ui.weak("No surface to close yet — fit surfaces here, or planes and cylinders in 3D Measure.");
        }
    });
    let typed = TYPED.with(Cell::get);
    if toggled {
        changed = true;
        if !typed {
            x.tolerance = tolerance_for(ws, &x.tools);
        }
    }
    if num(ui, "Tolerance", &mut x.tolerance, "mm", 0.001..=100.0, 0.005, "How far a face's sampled point may lie from the scan and still count as on it — three times the scan's scatter, read off the surfaces' own fits") {
        TYPED.with(|t| t.set(true));
        changed = true;
    }
    ui.weak(if TYPED.with(Cell::get) {
        "Tolerance as typed.".to_string()
    } else if !sigmas(ws, &x.tools).is_empty() {
        "Three times the median σ of the fitted surfaces in the set.".to_string()
    } else {
        format!("No fitted surface in the set — {} to start.", mm(DEFAULT_SPLIT_TOLERANCE_MM, 3))
    });
    let mut pct = (x.overlap * 100.0).round();
    if num(ui, "Overlap", &mut pct, "%", 5.0..=100.0, 1.0, "The share of a face's samples that must lie on the scan for the face to be kept") {
        x.overlap = (pct / 100.0).clamp(0.05, 1.0);
        changed = true;
    }
    if let Some(t) = report(ws).and_then(|r| r.trim.as_ref()) {
        let volume = ws.preview_feature.and_then(|f| ws.bodies.iter().find(|b| b.id.maker() == f)).map_or(0.0, |b| b.volume);
        if t.closed {
            ui.weak(format!("Closed: {} of the {} faces the surfaces cut lie on the scan and sew into a solid of {:.1} mm³.", t.kept, t.faces, volume));
        } else {
            ui.colored_label(
                egui::Color32::from_rgb(240, 200, 110),
                format!(
                    "⚠ Not closed: {} of {} faces lie on the scan, and they leave {} open edge{} — drawn red on the part. Fit the surface that is missing there, or widen the tolerance; Create keeps the faces as a surface body meanwhile.",
                    t.kept,
                    t.faces,
                    t.open_edges.len(),
                    if t.open_edges.len() == 1 { "" } else { "s" }
                ),
            );
        }
    }
    changed
}

pub(crate) fn pick(_ws: &mut ReverseWorkspace, _vp: Viewport, _p: Pos2, _clicked: bool, _x: &mut TrimFeature, _pick: &mut PickWhat) -> bool {
    false
}

/// The open edges of a trim that does not close, red on the part.
pub(crate) fn paint(ws: &ReverseWorkspace, painter: &egui::Painter, vp: Viewport, _x: &TrimFeature) {
    if let Some(t) = report(ws).and_then(|r| r.trim.as_ref()) {
        paint_lines(ws, painter, vp, &t.open_edges, RED);
    }
}
