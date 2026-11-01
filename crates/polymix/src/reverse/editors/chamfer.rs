// SPDX-License-Identifier: AGPL-3.0-only
//! The Chamfer feature's box: the body, one distance on both faces, two distances, or a
//! distance and an angle — measured from each edge's reference face, the face most of the edges
//! share, or flipped the other — the edges picked on the body (each kept as its fingerprint),
//! and whether the chamfer runs on along tangent edges. A click on an edge picks it, a second
//! lets it go; the edges the chain reaches are lit lighter; an arrow on the first edge drags the
//! distance. Ported from ScanRuler (`plugins/reverse/ui/FeatureEditor.tsx`,
//! `featureFields.tsx`, `plugins/reverse/state/reFeatureSlice.ts`, `reDrafts.ts`
//! featureOfDraft), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.

use egui::Pos2;
use mm_reverse::model::{ChamferFeature, ChamferMode, Exprs, FeatureKind};
use mm_reverse::nominal::NominalKind;

use super::rnd::{self, with_draft};
use crate::reverse::sketcher::plural;
use crate::reverse::{PickWhat, ReverseWorkspace};
use crate::tools::Viewport;

pub(crate) const READY: bool = true;
pub(crate) const BUTTON: (&str, &str) = ("Chamfer", "Chamfer edges of the body — click them");
pub(crate) const FIRST_PICK: PickWhat = PickWhat::Edges;

pub(crate) fn new_feature(ws: &ReverseWorkspace) -> Result<FeatureKind, String> {
    rnd::forget();
    let body = rnd::last_solid(ws).ok_or("A chamfer bevels a body's edges: make a body first.")?;
    Ok(FeatureKind::Chamfer(ChamferFeature { body, edges: Vec::new(), edges_where: Vec::new(), distance: rnd::DEFAULT_FILLET_MM, chain: true, skip_failing: false, mode: None, distance2: None, angle_deg: None, flip: false, exprs: Exprs::new() }))
}

/// The box's draft for this chamfer: a chamfer reopened from the list brings its second number
/// and its flip back; a new one starts at a millimetre and 45°.
fn session(ws: &ReverseWorkspace, x: &ChamferFeature) {
    let at = x as *const ChamferFeature as usize;
    let (d2, angle, flip) = reopened(x);
    rnd::session(
        ws,
        at,
        |d, editing| {
            if editing {
                d.distance2 = d2;
                d.angle_deg = angle;
                d.flip = flip;
            }
        },
        |_| (),
    );
}

/// What a chamfer reopened from the list brings back to its box: its second distance (its one
/// distance where it has none), its angle (45° where none) and its flip.
pub(crate) fn reopened(x: &ChamferFeature) -> (f64, f64, bool) {
    (x.distance2.unwrap_or(x.distance), x.angle_deg.unwrap_or(45.0), x.flip)
}

/// The chamfer's mode, written as the feature keeps it: an equal chamfer with no second number
/// and no flip, the others with their own number alone and the flip.
pub(crate) fn write_mode(x: &mut ChamferFeature) {
    let (d2, angle, flip) = with_draft(|d| (d.distance2, d.angle_deg, d.flip));
    match x.mode.unwrap_or(ChamferMode::Equal) {
        ChamferMode::Equal => {
            x.mode = None;
            x.distance2 = None;
            x.angle_deg = None;
            x.flip = false;
        }
        ChamferMode::TwoDistances => {
            x.distance2 = Some(d2);
            x.angle_deg = None;
            x.flip = flip;
        }
        ChamferMode::DistanceAngle => {
            x.angle_deg = Some(angle);
            x.distance2 = None;
            x.flip = flip;
        }
    }
}

/// The chamfer turned to another mode.
pub(crate) fn set_mode(x: &mut ChamferFeature, mode: ChamferMode) {
    x.mode = (mode != ChamferMode::Equal).then_some(mode);
    write_mode(x);
}

pub(crate) fn ui(ws: &mut ReverseWorkspace, ui: &mut egui::Ui, x: &mut ChamferFeature, pick: &mut PickWhat) -> bool {
    session(ws, x);
    *pick = PickWhat::Edges;
    let mut changed = false;
    if rnd::body_select(ui, ws, &mut x.body, "chamfer") {
        x.edges.clear();
        x.edges_where.clear();
        changed = true;
    }
    let mut mode = x.mode.unwrap_or(ChamferMode::Equal);
    ui.horizontal(|ui| {
        ui.label("Type");
        rnd::info_dot(ui, "Chamfer type", "Equal distance cuts both faces back alike. Two distances cuts the reference face back by the first and the face across by the second; Distance and angle cuts the reference face back by the distance, the chamfer at the angle to it. The reference face of each edge is the face most of the edges share — a rim's top — and Flip takes the other one.");
        let was = mode;
        let name = |m: ChamferMode| match m {
            ChamferMode::Equal => "Equal distance",
            ChamferMode::TwoDistances => "Two distances",
            ChamferMode::DistanceAngle => "Distance and angle",
        };
        egui::ComboBox::from_id_salt("re_chamfer_mode").selected_text(name(mode)).show_ui(ui, |ui| {
            for m in [ChamferMode::Equal, ChamferMode::TwoDistances, ChamferMode::DistanceAngle] {
                ui.selectable_value(&mut mode, m, name(m));
            }
        });
        if mode != was {
            set_mode(x, mode);
            changed = true;
        }
    });
    let picked = x.edges.len();
    ui.horizontal_wrapped(|ui| {
        ui.label("Edges");
        if picked == 0 {
            ui.weak("Click the edges of the body — a round runs on along the edges tangent to it. Click a picked edge again to let it go");
        } else {
            ui.strong(format!("{picked} edge{} picked", plural(picked)));
            if rnd::clear_picks(ui) {
                x.edges.clear();
                x.edges_where.clear();
                changed = true;
            }
        }
    });
    let equal = mode == ChamferMode::Equal;
    let mut distance = x.distance;
    let (label, said) = if equal {
        ("Distance", "How far back from the edge each face is cut. The arrow on the edge drags it.")
    } else {
        ("Distance 1", "How far back from the edge the reference face is cut. The arrow on the edge drags it.")
    };
    if rnd::num(ui, label, said, &mut distance, NominalKind::Length, 0.0..=1e4, Some((&mut x.exprs, "distance")), "chamfer_distance") {
        x.distance = distance.abs();
        changed = true;
    }
    if mode == ChamferMode::TwoDistances {
        let mut d2 = with_draft(|d| d.distance2);
        if rnd::num(ui, "Distance 2", "How far back from the edge the face across from the reference is cut.", &mut d2, NominalKind::Length, 0.0..=1e4, Some((&mut x.exprs, "distance2")), "chamfer_distance2") {
            with_draft(|d| d.distance2 = d2.abs());
            write_mode(x);
            changed = true;
        }
    }
    if mode == ChamferMode::DistanceAngle {
        let mut a = with_draft(|d| d.angle_deg);
        if rnd::num(ui, "Angle", "Degrees between the chamfer and the reference face, under 90 — 45° on a square edge is the equal chamfer.", &mut a, NominalKind::Angle, 0.1..=89.9, Some((&mut x.exprs, "angleDeg")), "chamfer_angle") {
            with_draft(|d| d.angle_deg = a.abs().clamp(0.1, 89.9));
            write_mode(x);
            changed = true;
        }
    }
    if !equal {
        let mut flip = with_draft(|d| d.flip);
        if ui.checkbox(&mut flip, "Flip — the other face is the reference").on_hover_text("Measure from the other face of each edge: the distances swap sides, the angle is read off the other face").changed() {
            with_draft(|d| d.flip = flip);
            write_mode(x);
            changed = true;
        }
    }
    ui.horizontal(|ui| {
        changed |= ui.checkbox(&mut x.chain, "Tangent propagation").changed();
        rnd::info_dot(ui, "Tangent propagation", "The chamfer runs on along every edge that leaves a picked one smoothly — all the way round a rounded rectangle from one click on its rim. The edges it reaches are lit in a lighter green on the part. Off, only the edges clicked are taken.");
    });
    changed
}

pub(crate) fn pick(ws: &mut ReverseWorkspace, vp: Viewport, p: Pos2, clicked: bool, x: &mut ChamferFeature, _pick: &mut PickWhat) -> bool {
    let was = x.body;
    let changed = super::pick_edge(ws, vp, p, clicked, &mut x.body, &mut x.edges);
    if x.body != was {
        x.edges_where.clear();
    }
    changed
}

pub(crate) fn paint(ws: &ReverseWorkspace, painter: &egui::Painter, vp: Viewport, x: &ChamferFeature) {
    rnd::paint_edge_picks(ws, painter, vp, x.body, &x.edges, x.chain);
    rnd::paint_window(painter);
    if let Some(g) = x.edges.first().and_then(|e| rnd::edge_grip(ws, x.body, e)) {
        rnd::paint_grip(ws, painter, vp, g);
    }
}

#[cfg(test)]
mod tests;
