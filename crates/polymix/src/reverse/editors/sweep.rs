// SPDX-License-Identifier: AGPL-3.0-only
//! The Sweep feature's box: the sketch of the profile and the sketch of the path, a key that
//! swaps them, and what the sweep does with the bodies it meets — or, for a sweep along a route
//! read off the scan, the route's points and bends and its bend radius. Ported from ScanRuler
//! (`plugins/reverse/ui/SweepFields.tsx`, `FeatureEditor.tsx` the sweep fields,
//! `state/reFeatureSlice.ts` startFeature), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! Also here: a bent tube's centreline read off the scan (ScanRuler's `reverse.scan.tube_trace`
//! command, `app/reSectionCommands.ts`), for a route to sweep along.

use egui::Pos2;
use glam::DVec3;
use mm_reverse::js_fmt::{num, round_to};
use mm_reverse::model::{BodyOp, FeatureId, FeatureKind, SweepFeature};
use mm_reverse::regions::find_regions;
use mm_reverse::slice::{slice_mesh, SectionCut};
use mm_reverse::tube_trace::{trace_tube, TraceInput, TubeRun};

use super::loft_kit::{info_dot, label_with_info, sketches};
use super::op_line;
use crate::reverse::{PickWhat, ReverseWorkspace};
use crate::tools::Viewport;

pub(crate) const READY: bool = true;
pub(crate) const BUTTON: (&str, &str) = ("Sweep", "A sketch's region swept along a chain of lines and arcs in another sketch — a bent bar, a handle, a rim that follows a curve");
pub(crate) const FIRST_PICK: PickWhat = PickWhat::Nothing;

/// The last two sketches: the one that closes a region the profile, the other the path — the
/// box swaps them if that guessed wrong. Joined.
pub(crate) fn new_feature(ws: &ReverseWorkspace) -> Result<FeatureKind, String> {
    let all = sketches(ws);
    if all.len() < 2 {
        return Err("A sweep takes a profile sketch and a path sketch: draw them first.".into());
    }
    let (a, b) = (all[all.len() - 2].0, all[all.len() - 1].0);
    let closes = |id: FeatureId| match ws.built.sketches.get(&id) {
        Some(s) => !s.regions.is_empty(),
        None => ws.part.sketch(id).is_some_and(|s| !find_regions(&s.sketch).is_empty()),
    };
    let (profile, path) = if closes(b) && !closes(a) { (b, a) } else { (a, b) };
    Ok(FeatureKind::Sweep(SweepFeature { profile, path: Some(path), route: None, op: BodyOp::Join, target: None, exprs: Default::default() }))
}

/// A sketch select: `label`, `Select…` and every sketch. Returns the sketch taken.
fn sketch_select(ui: &mut egui::Ui, ws: &ReverseWorkspace, label: &str, id_salt: &str, value: Option<FeatureId>) -> Option<FeatureId> {
    let all = sketches(ws);
    let mut taken = None;
    ui.horizontal(|ui| {
        ui.label(label);
        let shown = value.and_then(|v| all.iter().find(|s| s.0 == v)).map_or_else(|| "Select…".to_string(), |s| s.1.clone());
        egui::ComboBox::from_id_salt(id_salt).selected_text(shown).show_ui(ui, |ui| {
            for (id, name) in &all {
                if ui.selectable_label(value == Some(*id), name).clicked() {
                    taken = Some(*id);
                }
            }
        });
    });
    taken
}

pub(crate) fn ui(ws: &mut ReverseWorkspace, ui: &mut egui::Ui, x: &mut SweepFeature, pick: &mut PickWhat) -> bool {
    let mut changed = false;
    *pick = PickWhat::Nothing;
    if let Some(route) = &mut x.route {
        if let Some(id) = sketch_select(ui, ws, "Profile", "re_sweep_profile", Some(x.profile)) {
            changed |= x.profile != id;
            x.profile = id;
        }
        let bends = route.points.len().saturating_sub(2);
        ui.horizontal_wrapped(|ui| {
            ui.weak(format!("Along a route in 3D: {} points, {bends} bend{}.", route.points.len(), if bends == 1 { "" } else { "s" }));
            info_dot(ui, "A route", &["The centreline's points came by a command, as reverse.scan.tube_trace reads them, and reverse.feature.update changes them; there is no drawing them here. It moves with the part and needs no sketch. The profile stands square to it at its start, with the start on it."]);
        });
        ui.horizontal(|ui| {
            label_with_info(ui, "Bend radius", "On the centreline, for every bend without a radius of its own. A bend that does not fit its straights is refused, saying the most it takes.");
            let mut r = route.bend_radius;
            if ui.add(egui::DragValue::new(&mut r).speed(0.1).suffix(" mm")).changed() {
                route.bend_radius = r.abs();
                x.exprs.remove("route.bendRadius");
                changed = true;
            }
        });
        changed |= op_line(ui, &mut x.op);
        return changed;
    }
    if let Some(id) = sketch_select(ui, ws, "Profile", "re_sweep_profile", Some(x.profile)) {
        changed |= x.profile != id;
        x.profile = id;
    }
    if let Some(id) = sketch_select(ui, ws, "Path", "re_sweep_path", x.path) {
        changed |= x.path != Some(id);
        x.path = Some(id);
    }
    ui.horizontal(|ui| {
        // The profile cannot be left unchosen here: with no path there is nothing to swap in.
        let r = ui.add_enabled(x.path.is_some(), egui::Button::new("Swap profile and path")).on_hover_text("The profile's sketch for the path's, and the path's for the profile's");
        if r.clicked() {
            if let Some(path) = x.path {
                x.path = Some(x.profile);
                x.profile = path;
                changed = true;
            }
        }
    });
    ui.horizontal_wrapped(|ui| {
        ui.weak("The profile sketch's largest region is carried along the path sketch's one chain of lines and arcs, from the end nearest the profile.");
        info_dot(
            ui,
            "A sweep",
            &[
                "The section is held square to the path and turns as little as it can along it. Sketch the profile on a plane square to the path where it starts — one more than a degree off square is refused, saying by how much.",
                "The path is lines and arcs end to end — one chain, no branch; a closed path is drawn as arcs, a circle having no end to start from. Holes in the profile are swept with it.",
            ],
        );
    });
    changed |= op_line(ui, &mut x.op);
    changed
}

pub(crate) fn pick(_ws: &mut ReverseWorkspace, _vp: Viewport, _p: Pos2, _clicked: bool, _x: &mut SweepFeature, _pick: &mut PickWhat) -> bool {
    false
}

pub(crate) fn paint(_ws: &ReverseWorkspace, _painter: &egui::Painter, _vp: Viewport, _x: &SweepFeature) {}

/// How short a chain a cut of the scan keeps, mm (ScanRuler's `EDGE_MIN_FEATURE_MM`).
const EDGE_MIN_FEATURE_MM: f64 = 1.0;

/// A bent tube's centreline read off the scan from a point inside it along a way (ScanRuler's
/// `reverse.scan.tube_trace`), in words: the stations, where and why it stopped each way, the
/// route of straights and bends fitted to the centres, and how far they lie off it.
pub(crate) fn trace_tube_words(ws: &mut ReverseWorkspace, start: DVec3, dir: DVec3, both: bool) -> String {
    let parts = ws.scan_parts();
    if parts.is_empty() {
        return "There is no scan to follow a tube through.".into();
    }
    let mut cutter = |origin: DVec3, normal: DVec3| {
        let mut cut = SectionCut::default();
        for p in parts.iter() {
            let c = slice_mesh(&p.positions, &p.triangles, origin, normal, EDGE_MIN_FEATURE_MM);
            cut.chains.extend(c.chains);
            cut.normals.extend(c.normals);
        }
        cut
    };
    let mut input = TraceInput::new(start, dir);
    input.both = both;
    let t = trace_tube(&mut cutter, input);
    let r3 = |x: f64| num(round_to(x, 3));
    let at = |p: DVec3| format!("[{}, {}, {}]", r3(p.x), r3(p.y), r3(p.z));
    let stops: Vec<String> = t.stops.iter().map(|s| format!("{} at {} (s {}): {}", s.kind.name(), at(s.at), num(round_to(s.s, 2)), s.says)).collect();
    let route = match &t.route {
        Some(r) => format!("route {} points, bend radius {} mm: {}", r.points.len(), r3(r.bend_radius), r.points.iter().map(|p| at(*p)).collect::<Vec<_>>().join(" ")),
        None => "no route".into(),
    };
    let runs: Vec<String> = t
        .runs
        .iter()
        .map(|r| match r {
            TubeRun::Straight { length, major, minor, .. } => format!("straight {} mm ({} × {})", r3(*length), r3(*major), r3(*minor)),
            TubeRun::Bend { radius, turn_deg, major, minor, .. } => format!("bend R{} through {}° ({} × {})", r3(*radius), num(round_to(*turn_deg, 1)), r3(*major), r3(*minor)),
        })
        .collect();
    let residual = t.residual.map_or(String::new(), |r| format!("; the centres {} rms, {} at most off it", r3(r.rms), r3(r.max)));
    let warnings = if t.warnings.is_empty() { String::new() } else { format!("; {}", t.warnings.join(" ")) };
    format!("{} stations in {} cuts; stops: {}; {}; runs: {}{residual}{warnings}", t.stations.len(), t.cuts, stops.join(" / "), route, runs.join(", "))
}
