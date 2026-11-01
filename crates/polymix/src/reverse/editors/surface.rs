// SPDX-License-Identifier: AGPL-3.0-only
//! The Fit surface feature's box: what is fitted (a freeform B-spline, a plane or a cylinder),
//! what a plane or a cylinder is held to, the region of the scan it is fitted through —
//! gathered by clicks that flood the scan over edges turning less than the flood angle
//! (Ctrl-click takes a flood away), or marked with the tools (window, brush, lasso) — and the
//! freeform fit's net, how far it reaches past the region and how fair it is held. Ported from
//! ScanRuler (`plugins/reverse/ui/FeatureEditor.tsx` the surface's fields, `featureFields.tsx`
//! its preview line, `state/reFeatureSlice.ts` its defaults, `state/reTypes.ts`
//! DEFAULT_FLOOD_DEG), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only. The marking tools are
//! the box's own, as 3D Measure's are ScanRuler's: window, brush and lasso over the scan's
//! vertices that face the eye (or all of them), adding or, erasing, taking away.

use std::cell::RefCell;
use std::collections::HashSet;

use egui::{Color32, Pos2, RichText, Stroke};
use mm_reverse::auto_constrain::{relation_words, relations_for, FitShape, RefKind};
use mm_reverse::build::{constrain_refs, SURFACE_MIN_REGION};
use mm_reverse::model::{ConstrainMode, FeatureKind, NetSize, Relation, SurfaceFeature, SurfaceFitParams, SurfaceShape};

use super::surf_common::{count, flood_from, mm, num, report, scan_offsets, vertex_under};
use crate::reverse::view::PICKED;
use crate::reverse::{PickWhat, ReverseWorkspace};
use crate::tools::Viewport;

/// Whether the editor is built: its button stands with the others.
pub(crate) const READY: bool = true;
/// The panel's button: its label and what it does.
pub(crate) const BUTTON: (&str, &str) = ("Fit surface", "A surface fitted through a region of the scan");
/// What a click picks first when the box opens.
pub(crate) const FIRST_PICK: PickWhat = PickWhat::Nothing;

/// How far the normals may turn over one edge for a click's flood to cross it, degrees, to
/// start with.
const DEFAULT_FLOOD_DEG: f64 = 1.0;

/// The marking tools' gestures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mark {
    Navigate,
    Window,
    Brush,
    Lasso,
}

/// What the open box keeps beside its feature: the flood angle, the marking tools and the
/// gesture under way, and the keys held this frame.
struct Draft {
    flood_deg: f64,
    paint: bool,
    tool: Mark,
    brush_mm: f64,
    away: bool,
    erase: bool,
    drag: Vec<Pos2>,
    ctrl: bool,
}

impl Default for Draft {
    fn default() -> Self {
        Draft { flood_deg: DEFAULT_FLOOD_DEG, paint: false, tool: Mark::Brush, brush_mm: 4.0, away: false, erase: false, drag: Vec::new(), ctrl: false }
    }
}

thread_local! {
    static DRAFT: RefCell<Draft> = RefCell::new(Draft::default());
}

/// A new surface: no region yet, the interview's fit (Auto net, 35 % past the region,
/// smoothness 0.01), freeform, held to what it lands near.
pub(crate) fn new_feature(ws: &ReverseWorkspace) -> Result<FeatureKind, String> {
    if ws.scan.iter().all(|o| !o.visible) {
        return Err("A surface is fitted through the scan, and none is open.".into());
    }
    DRAFT.with(|d| *d.borrow_mut() = Draft::default());
    Ok(FeatureKind::Surface(SurfaceFeature {
        region: Vec::new(),
        fit: SurfaceFitParams { net: None, degree: 3, expand_pct: 35.0, smoothness: 0.01, robust: None, normal: None },
        shape: Some(SurfaceShape::Freeform),
        constrain: Some(ConstrainMode::Auto),
        region_box: None,
        region_stale: None,
        exprs: Default::default(),
    }))
}

/// What a plane or a cylinder can be held to: the part's planes, the measured elements, and the
/// planes and cylinders fitted before this one in the list — every one of them, for a surface
/// that is new.
fn holds(ws: &ReverseWorkspace, editing: Option<u32>) -> Vec<(String, String, RefKind)> {
    let mut out: Vec<(String, String, RefKind)> = constrain_refs(&ws.part).into_iter().map(|r| (r.key, r.name, r.kind)).collect();
    let at = editing.and_then(|id| ws.part.index_of(id)).unwrap_or(ws.part.features.len());
    for f in &ws.part.features[..at.min(ws.part.features.len())] {
        if let FeatureKind::Surface(s) = &f.kind {
            match s.shape {
                Some(SurfaceShape::Plane) => out.push((format!("surface:{}", f.id), f.name.clone(), RefKind::Plane)),
                Some(SurfaceShape::Cylinder) => out.push((format!("surface:{}", f.id), f.name.clone(), RefKind::Axis)),
                _ => {}
            }
        }
    }
    out
}

pub(crate) fn ui(ws: &mut ReverseWorkspace, ui: &mut egui::Ui, x: &mut SurfaceFeature, _pick: &mut PickWhat) -> bool {
    let mut changed = false;
    let mods = ui.ctx().input(|i| i.modifiers);
    DRAFT.with(|d| d.borrow_mut().ctrl = mods.ctrl || mods.command);
    let shape = x.shape.unwrap_or(SurfaceShape::Freeform);
    ui.horizontal(|ui| {
        ui.label("Shape").on_hover_text(
            "What is fitted. Freeform is a smooth B-spline surface through the region — a blend, a crowned lid. Plane and Cylinder are the exact faces a designed part is made of, and they are held to what the part already has: a fit that lands within 0.5° of parallel or square to a coordinate plane, a measured element or a surface fitted before it is fitted again with that held, and within 1 mm of a parallel plane or axis it is put on it. A trim cuts with the whole plane or cylinder, not only the patch drawn.",
        );
        let word = |s: SurfaceShape| match s {
            SurfaceShape::Freeform => "Freeform",
            SurfaceShape::Plane => "Plane",
            SurfaceShape::Cylinder => "Cylinder",
        };
        egui::ComboBox::from_id_salt("re_surface_shape").selected_text(word(shape)).show_ui(ui, |ui| {
            for s in [SurfaceShape::Freeform, SurfaceShape::Plane, SurfaceShape::Cylinder] {
                if ui.selectable_label(shape == s, word(s)).clicked() && shape != s {
                    x.shape = Some(s);
                    x.constrain = Some(ConstrainMode::Auto);
                    changed = true;
                }
            }
        });
    });
    let shape = x.shape.unwrap_or(SurfaceShape::Freeform);
    if shape != SurfaceShape::Freeform {
        let fit_shape = if shape == SurfaceShape::Plane { FitShape::Plane } else { FitShape::Cylinder };
        let options = holds(ws, ws.preview_feature.filter(|id| ws.part.feature(*id).is_some()));
        let mode = x.constrain.clone().unwrap_or(ConstrainMode::Auto);
        let shown = match &mode {
            ConstrainMode::Auto => "Auto — what it lands near".to_string(),
            ConstrainMode::Free => "Nothing — the free fit".to_string(),
            ConstrainMode::Force { relation, reference } => {
                let found = options.iter().find(|o| &o.0 == reference);
                format!("{} {}", found.map_or("Held to", |o| relation_words(fit_shape, *relation, o.2)), found.map_or(reference.clone(), |o| o.1.clone()))
            }
        };
        ui.horizontal(|ui| {
            ui.label("Hold to");
            egui::ComboBox::from_id_salt("re_surface_hold").selected_text(shown).width(230.0).show_ui(ui, |ui| {
                let mut set = |m: ConstrainMode, ui: &mut egui::Ui, label: String| {
                    if ui.selectable_label(mode == m, label).clicked() {
                        x.constrain = Some(m);
                        changed = true;
                    }
                };
                set(ConstrainMode::Auto, ui, "Auto — what it lands near".into());
                set(ConstrainMode::Free, ui, "Nothing — the free fit".into());
                for (key, name, kind) in &options {
                    for relation in relations_for(fit_shape, *kind) {
                        set(ConstrainMode::Force { relation: *relation, reference: key.clone() }, ui, format!("{} {}", relation_words(fit_shape, *relation, *kind), name));
                    }
                }
            });
        });
    }
    ui.horizontal(|ui| {
        ui.label("Region").on_hover_text(
            "The patch of the scan the surface is fitted through. Click the scan to add the face under the click — a flood from there over every edge whose normals turn by less than the angle, one flat face of a CAD-like part at 1°, a patch of a scan up to its first crease; Ctrl-click takes a flood away. Or mark it with the tools — brush, window, lasso — like a surface in 3D Measure. The region is the surface's own: it takes nothing from a 3D Measure element and may overlap a neighbour's at a seam.",
        );
        let text = if x.region.is_empty() { "None — click the scan".to_string() } else { format!("{} points", count(x.region.len())) };
        ui.label(RichText::new(text).strong().size(12.0));
        if ui.add_enabled(!x.region.is_empty(), egui::Button::new("Clear")).on_hover_text("Start the region over").clicked() {
            x.region.clear();
            x.region_box = None;
            x.region_stale = None;
            changed = true;
        }
    });
    DRAFT.with(|d| {
        let mut d = d.borrow_mut();
        let mut flood = d.flood_deg;
        if num(ui, "Flood angle", &mut flood, "°", 0.1..=89.0, 0.05, "How far the normals may turn over one edge for a click's flood to cross it") {
            d.flood_deg = flood.abs().clamp(0.1, 89.0);
        }
        ui.checkbox(&mut d.paint, "Mark with the tools");
    });
    if DRAFT.with(|d| d.borrow().paint) {
        changed |= mark_tools(ws, ui, x);
    }
    if shape == SurfaceShape::Freeform {
        let mut auto = x.fit.net.is_none();
        if ui.checkbox(&mut auto, "Net sized from the region").changed() {
            x.fit.net = if auto {
                None
            } else {
                let used = report(ws).and_then(|r| r.surface.as_ref()).map(|s| s.net).filter(|n| n.0 > 0);
                Some(used.map_or(NetSize { u: 5, v: 5 }, |n| NetSize { u: n.0 as u32, v: n.1 as u32 }))
            };
            changed = true;
        }
        if let Some(net) = &mut x.fit.net {
            let (mut u, mut v) = (net.u as f64, net.v as f64);
            if num(ui, "Net U", &mut u, "", 3.0..=20.0, 0.1, "Control points across the region's long side, 3 to 20") {
                net.u = u.round().clamp(3.0, 20.0) as u32;
                changed = true;
            }
            if num(ui, "Net V", &mut v, "", 3.0..=20.0, 0.1, "Control points across the short side, 3 to 20") {
                net.v = v.round().clamp(3.0, 20.0) as u32;
                changed = true;
            }
        }
    }
    changed |= num(ui, "Expand by", &mut x.fit.expand_pct, "%", 0.0..=200.0, 0.5, "How far past the region the surface reaches, as a share of its extent — it has to meet its neighbours for the trim");
    if shape == SurfaceShape::Freeform {
        changed |= num(ui, "Smoothness", &mut x.fit.smoothness, "", 0.0..=100.0, 0.001, "How hard the surface is held fair against the points: 0 follows them exactly, 0.01 fairs a scan's noise, more flattens detail");
    }
    // The preview line: what the fit came to.
    let line = match report(ws).and_then(|r| r.surface.as_ref()) {
        _ if x.region.len() < SURFACE_MIN_REGION => Some("Click the scan to gather the region; the surface follows.".to_string()),
        Some(s) if s.shape.is_none() => Some(format!("Fitted: σ {}, the worst point {} off, a {} × {} net over {} points.", mm(s.rms, 3), mm(s.max, 3), s.net.0, s.net.1, count(s.points))),
        Some(s) => {
            let fit_shape = if s.shape == Some(SurfaceShape::Plane) { FitShape::Plane } else { FitShape::Cylinder };
            let head = if s.shape == Some(SurfaceShape::Plane) { "A plane".to_string() } else { format!("A cylinder of R {}", mm(s.radius.unwrap_or(f64::NAN), 3)) };
            let held = if s.relations.is_empty() {
                ". Held to nothing.".to_string()
            } else {
                let said: Vec<String> = s
                    .relations
                    .iter()
                    .map(|r| format!("{} {} — the fit was {}{:.2}° off", relation_words(fit_shape, r.relation, r.ref_kind), r.name, if r.relation == Relation::On { r.shift_mm.map_or(String::new(), |m| format!("{} and ", mm(m, 2))) } else { String::new() }, r.tilt_deg))
                    .collect();
                format!(" ({} before it was held). {}.", mm(s.rms_free.unwrap_or(0.0), 3), said.join("; "))
            };
            Some(format!("{head}: σ {} over {} points{held}", mm(s.rms, 3), count(s.points)))
        }
        None => None,
    };
    if let Some(l) = line {
        ui.weak(l);
    }
    changed
}

/// The marking tools: the gestures, the brush's size, whether faces pointing away are marked
/// too, erasing, inverting and clearing the marking — and the gesture under way, read off the
/// pointer over the viewport. Whether the region changed.
fn mark_tools(ws: &mut ReverseWorkspace, ui: &mut egui::Ui, x: &mut SurfaceFeature) -> bool {
    let mut changed = false;
    let mut invert = false;
    let mut clear = false;
    DRAFT.with(|d| {
        let mut d = d.borrow_mut();
        ui.horizontal(|ui| {
            for (m, label, hint) in [(Mark::Navigate, "Navigate", "Clicks and drags move the view"), (Mark::Window, "Window", "Drag a box: the scan inside it is marked"), (Mark::Brush, "Brush", "Paint the scan under the brush"), (Mark::Lasso, "Lasso", "Draw round the scan to mark")] {
                if ui.selectable_label(d.tool == m, label).on_hover_text(hint).clicked() {
                    d.tool = m;
                    d.drag.clear();
                }
            }
        });
        let mut brush = d.brush_mm;
        if num(ui, "Brush Ø", &mut brush, "mm", 0.1..=1000.0, 0.1, "The brush's width on the scan") {
            d.brush_mm = brush;
        }
        ui.checkbox(&mut d.away, "Mark faces pointing away too");
        ui.horizontal(|ui| {
            let label = if d.erase { "Erasing" } else { "Erase" };
            if ui.selectable_label(d.erase, label).on_hover_text("Take the marked scan away instead of adding it").clicked() {
                d.erase = !d.erase;
            }
            if ui.button("Invert").on_hover_text("Mark what is not marked, and let go of what is").clicked() {
                invert = true;
            }
            if ui.button("Clear marking").clicked() {
                clear = true;
            }
        });
        ui.weak("Esc puts the tools away");
        if ui.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Escape)) {
            d.paint = false;
            d.drag.clear();
        }
    });
    if clear && !x.region.is_empty() {
        x.region.clear();
        x.region_box = None;
        changed = true;
    }
    if invert {
        let have: HashSet<u32> = x.region.iter().copied().collect();
        let mut all = Vec::new();
        let offsets = scan_offsets(ws);
        let parts = ws.scan_parts();
        for (p, (_, start, _)) in parts.iter().zip(&offsets) {
            for v in &p.live_vertices {
                let n = start + v;
                if !have.contains(&n) {
                    all.push(n);
                }
            }
        }
        x.region = all;
        x.region_box = None;
        changed = true;
    }
    changed |= gesture(ws, ui.ctx(), x);
    changed
}

/// The gesture under way over the viewport: a window or a lasso closed on release, the brush
/// as it is dragged. Whether the region changed.
fn gesture(ws: &mut ReverseWorkspace, ctx: &egui::Context, x: &mut SurfaceFeature) -> bool {
    let (tool, active) = DRAFT.with(|d| {
        let d = d.borrow();
        (d.tool, d.paint)
    });
    if !active || tool == Mark::Navigate {
        return false;
    }
    let rect = ctx.data(|d| d.get_temp::<egui::Rect>(egui::Id::NULL.with("re_vp_rect")));
    let Some(rect) = rect else { return false };
    let vp = Viewport { rect };
    let (pos, pressed, down, released, mods) = ctx.input(|i| (i.pointer.hover_pos(), i.pointer.button_pressed(egui::PointerButton::Primary), i.pointer.button_down(egui::PointerButton::Primary), i.pointer.button_released(egui::PointerButton::Primary), i.modifiers));
    if ws.nav.bindings.starts_navigation(egui::PointerButton::Primary, mods) {
        return false;
    }
    let over_panel = |p: Pos2| ws.view.panel_rect.is_some_and(|r| r.contains(p));
    let mut changed = false;
    match tool {
        Mark::Brush => {
            if let Some(p) = pos.filter(|p| rect.contains(*p) && !over_panel(*p)) {
                DRAFT.with(|d| d.borrow_mut().drag = vec![p]);
                if down && (pressed || ctx.input(|i| i.pointer.delta().length() > 0.0)) {
                    changed |= brush_at(ws, vp, p, x);
                }
            }
        }
        Mark::Window | Mark::Lasso => {
            if pressed {
                if let Some(p) = pos.filter(|p| rect.contains(*p) && !over_panel(*p)) {
                    DRAFT.with(|d| d.borrow_mut().drag = vec![p]);
                }
            } else if down {
                if let Some(p) = pos {
                    DRAFT.with(|d| {
                        let mut d = d.borrow_mut();
                        if d.drag.is_empty() {
                            return;
                        }
                        if tool == Mark::Window {
                            d.drag.truncate(1);
                            d.drag.push(p);
                        } else if d.drag.last().is_none_or(|l| (*l - p).length() > 3.0) {
                            d.drag.push(p);
                        }
                    });
                }
            } else if released {
                let shape = DRAFT.with(|d| std::mem::take(&mut d.borrow_mut().drag));
                if shape.len() >= 2 {
                    changed |= mark_inside(ws, vp, &shape, tool == Mark::Window, x);
                }
            }
        }
        Mark::Navigate => {}
    }
    if changed {
        x.region_box = None;
        ctx.request_repaint();
    }
    changed
}

/// The scan's vertices a test on the screen takes, facing the eye unless faces pointing away
/// are marked too — the build's numbers.
fn vertices_where(ws: &mut ReverseWorkspace, vp: Viewport, take: impl Fn(Pos2) -> bool) -> Vec<u32> {
    let away = DRAFT.with(|d| d.borrow().away);
    let eye = ws.to_part(ws.view.camera.eye());
    let offsets = scan_offsets(ws);
    let parts = ws.scan_parts();
    let mut out = Vec::new();
    for (p, (id, start, _)) in parts.iter().zip(&offsets) {
        let mesh = ws.scan.iter().find(|o| o.id == *id).map(|o| o.mesh.clone());
        for v in &p.live_vertices {
            let at = p.positions[*v as usize];
            let Some(q) = ws.project(vp, at) else { continue };
            if !take(q) {
                continue;
            }
            if !away {
                if let Some(m) = &mesh {
                    let n = if ws.flip { crate::app::flip_zy_out(m.vertex_normal(*v)) } else { m.vertex_normal(*v) };
                    if n.dot(eye - at) <= 0.0 {
                        continue;
                    }
                }
            }
            out.push(start + v);
        }
    }
    out
}

/// The region with these vertices added, or taken away when erasing. Whether it changed.
fn apply(x: &mut SurfaceFeature, found: Vec<u32>, remove: bool) -> bool {
    if found.is_empty() {
        return false;
    }
    if remove {
        let gone: HashSet<u32> = found.into_iter().collect();
        let before = x.region.len();
        x.region.retain(|v| !gone.contains(v));
        return x.region.len() != before;
    }
    let mut have: HashSet<u32> = x.region.iter().copied().collect();
    let before = x.region.len();
    for v in found {
        if have.insert(v) {
            x.region.push(v);
        }
    }
    x.region.len() != before
}

/// The brush on the scan at a screen point: every vertex within its radius of where the ray
/// meets the scan, facing the eye unless faces pointing away are marked too.
fn brush_at(ws: &mut ReverseWorkspace, vp: Viewport, p: Pos2, x: &mut SurfaceFeature) -> bool {
    let (radius, erase, away) = DRAFT.with(|d| {
        let d = d.borrow();
        (d.brush_mm / 2.0, d.erase, d.away)
    });
    let Some(hit) = ws.pick_hit(vp, p).filter(|h| matches!(h, crate::reverse::view::Hit::Scan { .. })) else { return false };
    let centre = ws.to_scene(hit.point());
    let eye = ws.view.camera.eye();
    let offsets = scan_offsets(ws);
    let objects: Vec<crate::reverse::ScanObject> = ws.scan.iter().filter(|o| o.visible).cloned().collect();
    let mut near = Vec::new();
    for o in &objects {
        let Some(start) = offsets.iter().find(|(id, _, _)| *id == o.id).map(|x| x.1) else { continue };
        let Some(bvh) = ws.scan_bvh(o) else { continue };
        let mut seen = HashSet::new();
        bvh.query_sphere(&o.mesh, centre, radius, |t| {
            for v in o.mesh.triangle(t) {
                if !seen.insert(v) {
                    continue;
                }
                let at = o.mesh.positions()[v as usize];
                if (at - centre).length() > radius {
                    continue;
                }
                if !away && o.mesh.vertex_normal(v).dot(eye - at) <= 0.0 {
                    continue;
                }
                near.push(start + v);
            }
        });
    }
    apply(x, near, erase)
}

/// A window (two corners) or a lasso (its points) closed over the scan.
fn mark_inside(ws: &mut ReverseWorkspace, vp: Viewport, shape: &[Pos2], window: bool, x: &mut SurfaceFeature) -> bool {
    let erase = DRAFT.with(|d| d.borrow().erase);
    let found = if window {
        let r = egui::Rect::from_two_pos(shape[0], shape[shape.len() - 1]);
        vertices_where(ws, vp, |q| r.contains(q))
    } else {
        let poly: Vec<Pos2> = shape.to_vec();
        vertices_where(ws, vp, |q| inside(&poly, q))
    };
    apply(x, found, erase)
}

/// Whether a point lies inside a polygon on the screen (even-odd).
fn inside(poly: &[Pos2], p: Pos2) -> bool {
    let mut c = false;
    let n = poly.len();
    for i in 0..n {
        let (a, b) = (poly[i], poly[(i + n - 1) % n]);
        if (a.y > p.y) != (b.y > p.y) && p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x {
            c = !c;
        }
    }
    c
}

/// A click on the scan adds the face under it — a flood over every edge turning less than the
/// flood angle; Ctrl-click takes the flood away.
pub(crate) fn pick(ws: &mut ReverseWorkspace, vp: Viewport, p: Pos2, clicked: bool, x: &mut SurfaceFeature, _pick: &mut PickWhat) -> bool {
    let (paint, tool, flood_deg, ctrl) = DRAFT.with(|d| {
        let d = d.borrow();
        (d.paint, d.tool, d.flood_deg, d.ctrl)
    });
    if !clicked || (paint && tool != Mark::Navigate) {
        return false;
    }
    let Some((object, vertex, _)) = vertex_under(ws, vp, p) else { return false };
    let flood = flood_from(ws, object, vertex, flood_deg);
    let n = flood.len();
    let before = x.region.len();
    if !apply(x, flood, ctrl) {
        return false;
    }
    x.region_box = None;
    x.region_stale = None;
    let moved = x.region.len().abs_diff(before);
    ws.status = format!("{} points {} the region — {} in all. Click more of the scan, or Create.", count(moved.min(n)), if ctrl { "taken out of" } else { "added to" }, count(x.region.len()));
    true
}

/// The region's points on the scan, and the gesture under way.
pub(crate) fn paint(ws: &ReverseWorkspace, painter: &egui::Painter, vp: Viewport, x: &SurfaceFeature) {
    if let Some(scan) = super::surf_common::scan_now(ws) {
        let step = (x.region.len() / 8000).max(1);
        for v in x.region.iter().step_by(step) {
            if let Some(q) = scan.positions().get(*v as usize).and_then(|p| ws.project(vp, *p)) {
                painter.circle_filled(q, 1.6, PICKED);
            }
        }
    }
    DRAFT.with(|d| {
        let d = d.borrow();
        if !d.paint {
            return;
        }
        let color = if d.erase { Color32::from_rgb(240, 120, 90) } else { PICKED };
        match d.tool {
            Mark::Window if d.drag.len() >= 2 => {
                painter.rect_stroke(egui::Rect::from_two_pos(d.drag[0], d.drag[d.drag.len() - 1]), 0.0, Stroke::new(1.5, color), egui::StrokeKind::Middle);
            }
            Mark::Lasso if d.drag.len() >= 2 => {
                painter.add(egui::Shape::closed_line(d.drag.clone(), Stroke::new(1.5, color)));
            }
            Mark::Brush => {
                if let Some(p) = d.drag.first() {
                    // The brush's size on the screen, at the scan under it.
                    let r = brush_pixels(ws, vp, *p, d.brush_mm / 2.0).unwrap_or(10.0);
                    painter.circle_stroke(*p, r, Stroke::new(1.5, color));
                }
            }
            _ => {}
        }
    });
}

/// How many pixels a length at a screen point's depth on the scan covers.
fn brush_pixels(ws: &ReverseWorkspace, vp: Viewport, p: Pos2, length: f64) -> Option<f32> {
    let ray = ws.view.camera.ray_from_ndc(vp.ndc(p), vp.aspect());
    let target = ws.view.camera.target;
    let depth = (target - ray.origin).dot(ray.dir).max(1e-6);
    let at = ws.to_part(ray.origin + ray.dir * depth);
    let side = ws.to_part(ws.view.camera.right());
    let a = ws.project(vp, at)?;
    let b = ws.project(vp, at + side.normalize_or_zero() * length)?;
    Some((b - a).length())
}
