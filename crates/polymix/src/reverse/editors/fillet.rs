// SPDX-License-Identifier: AGPL-3.0-only
//! The Fillet feature's box: the body, a constant radius or each edge from a start radius to an
//! end radius, the edges picked on the body (each kept as its fingerprint), the radius read off
//! the scan across the first edge, and whether the round runs on along tangent edges. A click
//! on an edge picks it, a second lets it go; the edges the chain reaches are lit lighter; an
//! arrow on the first edge drags the radius; a varying edge's Start and End are named on the
//! part. Ported from ScanRuler (`plugins/reverse/ui/FeatureEditor.tsx`, `featureFields.tsx`,
//! `plugins/reverse/state/reFeatureSlice.ts`, `reDrafts.ts` filletModeSwitch,
//! `plugins/reverse/viewer/varyingEnds.ts`, `app/useReverseWorkspace.ts`), © Stefan Hermann (CNC
//! Kitchen), AGPL-3.0-only.

use egui::{Color32, Pos2, Stroke};
use mm_reverse::fillet_num::js_num;
use mm_reverse::model::{BodyId, EdgeRef, Exprs, FeatureKind, FilletFeature, VariableEdge};
use mm_reverse::nominal::NominalKind;
use mm_reverse::topo::{describe_edge, edge_distance, MatchTolerance};

use super::rnd::{self, with_draft, Estimate};
use crate::reverse::sketcher::plural;
use crate::reverse::{PickWhat, ReverseWorkspace};
use crate::tools::Viewport;

pub(crate) const READY: bool = true;
pub(crate) const BUTTON: (&str, &str) = ("Fillet", "Round edges of the body — click them; the radius is read off the scan across the first");
pub(crate) const FIRST_PICK: PickWhat = PickWhat::Edges;

pub(crate) fn new_feature(ws: &ReverseWorkspace) -> Result<FeatureKind, String> {
    // No box is open while the panel offers a new one: what the last remembered goes.
    rnd::forget();
    let body = rnd::last_solid(ws).ok_or("A fillet rounds a body's edges: make a body first.")?;
    Ok(FeatureKind::Fillet(FilletFeature { body, edges: Vec::new(), edges_where: Vec::new(), radius: rnd::DEFAULT_FILLET_MM, chain: true, skip_failing: false, variable: Vec::new(), exprs: Exprs::new() }))
}

/// The box's draft for this fillet, started again for another box: a fillet reopened from the
/// list keeps its radius as typed, and opens on its varying edges when it has nothing else.
fn session(ws: &ReverseWorkspace, x: &FilletFeature) {
    let at = x as *const FilletFeature as usize;
    let variable = opens_variable(x);
    rnd::session(
        ws,
        at,
        |d, editing| {
            if editing {
                d.radius_typed = true;
                d.variable = variable;
            }
        },
        |_| (),
    );
}

/// Whether a fillet reopened from the list opens in Variable: on its varying edges, when it has
/// nothing else.
pub(crate) fn opens_variable(x: &FilletFeature) -> bool {
    x.edges.is_empty() && !x.variable.is_empty()
}

/// The radii an edge newly picked at a varying radius starts at: the last such edge's, or the
/// fillet's one radius at both ends, to type over.
fn varying_radii(x: &FilletFeature) -> [f64; 2] {
    x.variable.last().map_or([x.radius, x.radius], |v| v.radius)
}

/// The box turned to a constant or a variable radius: the edges picked the one way carried over
/// to the other while the other has none — each edge going from the one radius at both ends, or
/// the edges taking the first one's start radius — else only what new clicks go into.
pub(crate) fn switch_mode(x: &mut FilletFeature, variable: bool) {
    if with_draft(|d| d.variable) == variable {
        return;
    }
    with_draft(|d| d.variable = variable);
    if variable && x.variable.is_empty() && !x.edges.is_empty() {
        let r = x.radius;
        x.variable = x.edges.drain(..).map(|edge| VariableEdge { edge, radius: [r, r] }).collect();
        x.edges_where.clear();
    } else if !variable && x.edges.is_empty() && !x.variable.is_empty() {
        x.radius = x.variable[0].radius[0];
        x.edges = x.variable.drain(..).map(|v| v.edge).collect();
        with_draft(|d| d.radius_typed = true);
    }
}

/// A click on an edge: an edge picked either way is let go; a new one goes in at the one radius
/// or at a varying one, as the box is set. An edge of another body starts the picks again on
/// that body.
pub(crate) fn toggle_edge(ws: &ReverseWorkspace, x: &mut FilletFeature, on: BodyId, edge: usize) -> bool {
    let Some(print) = ws.pick_view(on).and_then(|v| v.topology.edges.get(edge)).map(|e| e.fingerprint.clone()) else { return false };
    let tol = rnd::same_tolerance(ws, on);
    toggle_print(x, on, print, &tol)
}

/// [`toggle_edge`] for the edge's fingerprint, told apart from the picks within `tol`.
pub(crate) fn toggle_print(x: &mut FilletFeature, on: BodyId, print: EdgeRef, tol: &MatchTolerance) -> bool {
    if x.body != on {
        x.body = on;
        x.edges.clear();
        x.edges_where.clear();
        x.variable.clear();
    }
    let same = |a: &EdgeRef| edge_distance(a, &print, tol).is_some();
    let has = x.edges.iter().any(same);
    let varies = x.variable.iter().any(|v| same(&v.edge));
    if has || varies {
        x.edges.retain(|e| !same(e));
        x.variable.retain(|v| !same(&v.edge));
    } else if with_draft(|d| d.variable) {
        let radius = varying_radii(x);
        x.variable.push(VariableEdge { edge: print, radius });
    } else {
        x.edges.push(print);
    }
    true
}

/// Edges a window took, added as the box is set: at a varying radius in Variable.
pub(crate) fn add_edges(ws: &ReverseWorkspace, x: &mut FilletFeature, on: BodyId, added: Vec<EdgeRef>) -> bool {
    if x.body != on {
        x.body = on;
        x.edges.clear();
        x.edges_where.clear();
        x.variable.clear();
    }
    let tol = rnd::same_tolerance(ws, on);
    let fresh: Vec<EdgeRef> = added.into_iter().filter(|e| !x.edges.iter().any(|a| edge_distance(a, e, &tol).is_some()) && !x.variable.iter().any(|v| edge_distance(&v.edge, e, &tol).is_some())).collect();
    if fresh.is_empty() {
        return false;
    }
    if with_draft(|d| d.variable) {
        let radius = varying_radii(x);
        x.variable.extend(fresh.into_iter().map(|edge| VariableEdge { edge, radius }));
    } else {
        x.edges.extend(fresh);
    }
    true
}

/// A length as a row reads it: to the hundredth, no zeros carried.
fn mm(v: f64) -> String {
    format!("{} mm", js_num((v * 100.0).round() / 100.0))
}

pub(crate) fn ui(ws: &mut ReverseWorkspace, ui: &mut egui::Ui, x: &mut FilletFeature, pick: &mut PickWhat) -> bool {
    session(ws, x);
    *pick = PickWhat::Edges;
    let mut changed = rnd::estimate(ws, ui.ctx(), x.body, &x.edges, &mut x.radius);
    if rnd::body_select(ui, ws, &mut x.body, "fillet") {
        x.edges.clear();
        x.edges_where.clear();
        x.variable.clear();
        changed = true;
    }
    let mut variable = with_draft(|d| d.variable);
    ui.horizontal(|ui| {
        ui.label("Radius");
        rnd::info_dot(ui, "Constant or variable radius", "Constant rounds every edge picked at the one radius, read off the scan across the first. Variable rounds each edge from a start radius at one end to an end radius at the other, evenly along it — a round that swells along a casting's edge. The part names each edge's Start and End by its ends; ⇄ swaps its two radii.");
        let was = variable;
        egui::ComboBox::from_id_salt("re_fillet_mode").selected_text(if variable { "Variable — start to end" } else { "Constant" }).show_ui(ui, |ui| {
            ui.selectable_value(&mut variable, false, "Constant");
            ui.selectable_value(&mut variable, true, "Variable — start to end");
        });
        if variable != was {
            switch_mode(x, variable);
            changed = true;
        }
    });
    let picked = if variable { x.variable.len() } else { x.edges.len() };
    ui.horizontal_wrapped(|ui| {
        ui.label("Edges");
        if picked == 0 {
            ui.weak(if variable {
                "Click the edges of the body to round, each from a start radius to an end radius. Click a picked edge again to let it go"
            } else {
                "Click the edges of the body — a round runs on along the edges tangent to it. Click a picked edge again to let it go"
            });
        } else {
            ui.strong(format!("{picked} edge{} picked", plural(picked)));
            if rnd::clear_picks(ui) {
                if variable {
                    x.variable.clear();
                } else {
                    x.edges.clear();
                    x.edges_where.clear();
                }
                changed = true;
            }
        }
    });
    if variable {
        let rows = x.variable.len();
        for (i, v) in x.variable.iter_mut().enumerate() {
            egui::Frame::group(ui.style()).show(ui, |ui| {
                ui.horizontal(|ui| {
                    let head = format!("{}{}", if rows > 1 { format!("{} · ", i + 1) } else { String::new() }, describe_edge(&v.edge));
                    ui.label(&head).on_hover_text(describe_edge(&v.edge));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.small_button("⇄").on_hover_text("Swap the two radii — the start radius at the other end").clicked() {
                            v.radius = [v.radius[1], v.radius[0]];
                            changed = true;
                        }
                    });
                });
                let mut start = v.radius[0];
                if rnd::num(ui, "Start", "The radius at the edge's start — the end named Start on the part", &mut start, NominalKind::Radius, 0.0..=1e4, None, &format!("fillet_start_{i}")) {
                    v.radius[0] = start.abs();
                    changed = true;
                }
                let mut end = v.radius[1];
                if rnd::num(ui, "End", "The radius at the edge's other end; the round runs evenly from the one to the other", &mut end, NominalKind::Radius, 0.0..=1e4, None, &format!("fillet_end_{i}")) {
                    v.radius[1] = end.abs();
                    changed = true;
                }
            });
        }
        let mut said = if rows == 0 {
            "Each edge clicked starts at the radius the last one has; type its two radii over.".to_string()
        } else {
            format!("The part names each edge's Start and End, with the radius there{}. Each edge is rounded on its own, after the others.", if rows > 1 { " — numbered as listed here" } else { "" })
        };
        let n = x.edges.len();
        if n > 0 {
            said += &format!(" And {} at R {}, kept — Constant shows {}.", if n == 1 { "an edge".to_string() } else { format!("{n} edges") }, mm(x.radius), if n == 1 { "it" } else { "them" });
        }
        rnd::hint(ui, &said, false);
    } else {
        let mut radius = x.radius;
        if rnd::num(ui, "Radius", "Read off the scan across the first picked edge; type your own to keep it. The arrow on the edge drags it.", &mut radius, NominalKind::Radius, 0.0..=1e4, Some((&mut x.exprs, "radius")), "fillet_radius") {
            x.radius = radius.abs();
            with_draft(|d| d.radius_typed = true);
            changed = true;
        }
        let (typed, estimate) = with_draft(|d| (d.radius_typed, d.estimate));
        let (said, warn) = if typed {
            ("Radius as typed.".to_string(), false)
        } else {
            match estimate {
                Estimate::Idle => ("Pick an edge and the radius is read off the scan there.".to_string(), false),
                Estimate::Working => ("Reading the round off the scan…".to_string(), false),
                Estimate::Found { radius, points } => (format!("Radius {radius:.2} mm read off the scan across the first edge ({points} section points)."), false),
                Estimate::None => (format!("No round found on the scan at the first edge — {} to start.", mm(x.radius)), true),
            }
        };
        rnd::hint(ui, &said, warn);
        let rows = x.variable.len();
        if rows > 0 {
            let radii: Vec<String> = x.variable.iter().map(|v| format!("{} → {}", mm(v.radius[0]), mm(v.radius[1]))).collect();
            rnd::hint(ui, &format!("And {} at a varying radius — {} — kept; Variable shows {}.", if rows == 1 { "an edge".to_string() } else { format!("{rows} edges") }, radii.join(", "), if rows == 1 { "it" } else { "them" }), false);
        }
        ui.horizontal(|ui| {
            changed |= ui.checkbox(&mut x.chain, "Tangent propagation").changed();
            rnd::info_dot(ui, "Tangent propagation", "The fillet runs on along every edge that leaves a picked one smoothly — all the way round a rounded rectangle from one click on its rim. The edges it reaches are lit in a lighter green on the part. Off, only the edges clicked are taken.");
        });
    }
    changed
}

pub(crate) fn pick(ws: &mut ReverseWorkspace, vp: Viewport, p: Pos2, clicked: bool, x: &mut FilletFeature, _pick: &mut PickWhat) -> bool {
    let edge = ws.edge_at(vp, p);
    ws.view.hover_edge = edge;
    let (true, Some((b, i))) = (clicked, edge) else { return false };
    toggle_edge(ws, x, b, i)
}

pub(crate) fn paint(ws: &ReverseWorkspace, painter: &egui::Painter, vp: Viewport, x: &FilletFeature) {
    let variable = with_draft(|d| d.variable);
    // The edges at the one radius and, with Tangent propagation, those their chains reach —
    // in either mode: Variable keeps them.
    rnd::paint_edge_picks(ws, painter, vp, x.body, &x.edges, x.chain);
    // An edge at a varying radius, lit as picked; while the box shows them, a dot on each end
    // and its Start and End named by it with the radius there.
    let several = x.variable.len() > 1;
    for (index, v) in x.variable.iter().enumerate() {
        let Some(i) = rnd::edge_index(ws, x.body, &v.edge) else {
            rnd::dot(ws, painter, vp, v.edge.mid, rnd::PICK_GREEN);
            continue;
        };
        rnd::paint_edge(ws, painter, vp, x.body, i, rnd::PICK_GREEN);
        if !variable {
            continue;
        }
        let Some(e) = ws.pick_view(x.body).and_then(|b| b.topology.edges.get(i)) else { continue };
        for k in 0..2 {
            let end = e.ends[k];
            rnd::dot(ws, painter, vp, end, rnd::PICK_GREEN);
            // The tangent at the start runs into the edge, at the end out of it.
            let out = if k == 0 { -e.end_tangents[0] } else { e.end_tangents[1] };
            let word = if k == 0 { "Start" } else { "End" };
            let title = if several { format!("{word} {}", index + 1) } else { word.to_string() };
            end_label(ws, painter, vp, end, out, &title, &format!("R {} mm", js_num((v.radius[k] * 100.0).round() / 100.0)));
        }
    }
    rnd::paint_window(painter);
    // The arrow on the first edge at the one radius drags that radius, in either mode.
    if let Some(g) = x.edges.first().and_then(|e| rnd::edge_grip(ws, x.body, e)) {
        rnd::paint_grip(ws, painter, vp, g);
    }
}

/// A chip pinned by the end of an edge — its title and the radius there — standing off the end
/// the way the edge leaves it (`out`) on the screen, re-placed as the view turns; above the end
/// when seen end-on.
fn end_label(ws: &ReverseWorkspace, painter: &egui::Painter, vp: Viewport, end: glam::DVec3, out: glam::DVec3, title: &str, value: &str) {
    let Some(at) = ws.project(vp, end) else { return };
    let on_screen = ws.view.camera.view().transform_vector3(ws.to_scene(out.normalize_or_zero()));
    // Scaled so the larger part is 1: a label off a slanting edge puts its corner there.
    let m = on_screen.x.abs().max(on_screen.y.abs());
    let (x, y) = if m < 0.2 { (0.0, 1.0) } else { ((on_screen.x / m) as f32, (on_screen.y / m) as f32) };
    let font = egui::FontId::proportional(11.0);
    let accent = Color32::from_rgb(0x2f, 0x8f, 0x5a);
    let title_g = painter.layout_no_wrap(title.to_string(), font.clone(), accent);
    let value_g = painter.layout_no_wrap(value.to_string(), font, Color32::from_gray(30));
    let w = title_g.size().x.max(value_g.size().x) + 10.0;
    let h = title_g.size().y + value_g.size().y + 6.0;
    let pad = 6.0;
    let (fw, fh) = (w + 2.0 * pad, h + 2.0 * pad);
    // The frame's corner on the end: the chip stands a few pixels off it.
    let ax = 0.5 - x / 2.0;
    let ay = 0.5 + y / 2.0;
    let min = at - egui::vec2(ax * fw, ay * fh) + egui::vec2(pad, pad);
    let r = egui::Rect::from_min_size(min, egui::vec2(w, h));
    painter.rect_filled(r, 4.0, Color32::from_rgba_unmultiplied(250, 250, 250, 235));
    painter.rect_stroke(r, 4.0, Stroke::new(1.0, Color32::from_gray(150)), egui::StrokeKind::Middle);
    let title_h = title_g.size().y;
    painter.galley(r.min + egui::vec2(5.0, 3.0), title_g, accent);
    painter.galley(r.min + egui::vec2(5.0, 3.0 + title_h), value_g, Color32::from_gray(30));
}

#[cfg(test)]
mod tests;
