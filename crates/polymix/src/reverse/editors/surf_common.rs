// SPDX-License-Identifier: AGPL-3.0-only
//! What the surface features' boxes share in the workspace: the scan as the build reads it
//! (every visible object of the Mesh workspace in turn, its vertices numbered on from the
//! last's — the numbers a fitted surface's region names), the scan's vertex under the cursor,
//! a flood over the scan from it, the vertices in a box, the region read again where a
//! parameter moved its box, what a box does once its feature is made, and the fields and
//! numbers as the boxes write them (the script steps are `surf_script.rs`'s). A port of what ScanRuler's
//! Reverse engineer workspace keeps for its sheets and surfaces (`plugins/reverse/app/`
//! `reSurfaceRegion.ts`, `ui/featureFields.tsx`), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use egui::Pos2;
use glam::DVec3;
use mm_reverse::build::BuildScan;
use mm_reverse::fit3d::flood_by_normal;
use mm_reverse::mesh_graph::MeshGraph;
use mm_reverse::model::{BodyId, BodyOp, BoxBounds, FeatureId, FeatureKind, RegionStale};

use crate::reverse::{Mode, ReverseWorkspace};
use crate::tools::Viewport;

thread_local! {
    /// The scan as the build last read it, by the workspace's scan key.
    static SCAN: RefCell<Option<(u64, Arc<BuildScan>)>> = const { RefCell::new(None) };
    /// Each scan object's graph for floods, by its id and mesh version.
    static GRAPHS: RefCell<HashMap<u64, (u64, Arc<MeshGraph>)>> = RefCell::new(HashMap::new());
}

/// Where each visible scan object's vertices start in the build's numbering: (object id, the
/// first number, how many slots).
pub(crate) fn scan_offsets(ws: &mut ReverseWorkspace) -> Vec<(u64, u32, u32)> {
    let parts = ws.scan_parts();
    let mut at = 0u32;
    parts
        .iter()
        .map(|p| {
            let start = at;
            at += p.positions.len() as u32;
            (p.id, start, p.positions.len() as u32)
        })
        .collect()
}

/// The scan as the build reads it: every visible object's vertices in part coordinates, one
/// after another, and their triangles — made once a scan and shared by every build of it.
/// `None` with no scan.
pub(crate) fn build_scan(ws: &mut ReverseWorkspace) -> Option<Arc<BuildScan>> {
    if ws.scan.iter().all(|o| !o.visible) {
        return None;
    }
    let key = ws.scan_key;
    if let Some(s) = SCAN.with(|c| c.borrow().as_ref().filter(|(k, _)| *k == key).map(|(_, s)| s.clone())) {
        return Some(s);
    }
    let parts = ws.scan_parts();
    let mut positions = Vec::new();
    let mut triangles = Vec::new();
    for p in parts.iter() {
        let base = positions.len() as u32;
        positions.extend_from_slice(&p.positions);
        triangles.extend(p.triangles.iter().map(|t| t.map(|v| v + base)));
    }
    let scan = Arc::new(BuildScan::new(positions, triangles));
    SCAN.with(|c| *c.borrow_mut() = Some((key, scan.clone())));
    Some(scan)
}

/// The scan as the build last read it, while it is the scan shown — for drawing, which may not
/// read it afresh.
pub(crate) fn scan_now(ws: &ReverseWorkspace) -> Option<Arc<BuildScan>> {
    let key = ws.scan_key;
    SCAN.with(|c| c.borrow().as_ref().filter(|(k, _)| *k == key).map(|(_, s)| s.clone()))
}

/// A scan object's graph for floods, made the first time it is asked for this mesh.
fn graph_of(o: &crate::reverse::ScanObject) -> Option<Arc<MeshGraph>> {
    let version = o.mesh.shape_version() ^ o.mesh.topology_version().rotate_left(32);
    if let Some(g) = GRAPHS.with(|c| c.borrow().get(&o.id).filter(|(v, _)| *v == version).map(|(_, g)| g.clone())) {
        return Some(g);
    }
    let g = Arc::new(MeshGraph::from_mesh(&o.mesh, false).ok()?);
    GRAPHS.with(|c| c.borrow_mut().insert(o.id, (version, g.clone())));
    Some(g)
}

/// The scan's vertex under a screen point — the nearest corner of the triangle the ray meets
/// first — as (object id, its vertex id, the build's number for it).
pub(crate) fn vertex_under(ws: &mut ReverseWorkspace, vp: Viewport, p: Pos2) -> Option<(u64, u32, u32)> {
    let ray = ws.view.camera.ray_from_ndc(vp.ndc(p), vp.aspect());
    let objects: Vec<crate::reverse::ScanObject> = ws.scan.iter().filter(|o| o.visible).cloned().collect();
    let mut best: Option<(f64, u64, u32)> = None;
    for o in &objects {
        let Some(bvh) = ws.scan_bvh(o) else { continue };
        let Some(h) = bvh.raycast(&o.mesh, &ray) else { continue };
        if best.is_some_and(|b| b.0 <= h.t) {
            continue;
        }
        let tri = o.mesh.triangle(h.tid);
        let k = if h.bary.x >= h.bary.y && h.bary.x >= h.bary.z {
            0
        } else if h.bary.y >= h.bary.z {
            1
        } else {
            2
        };
        best = Some((h.t, o.id, tri[k]));
    }
    let (_, id, v) = best?;
    let start = scan_offsets(ws).iter().find(|(o, _, _)| *o == id).map(|x| x.1)?;
    Some((id, v, start + v))
}

/// The vertices a flood from a scan vertex takes, over every edge whose normals turn by less
/// than `angle_deg` (the build's numbers).
pub(crate) fn flood_from(ws: &mut ReverseWorkspace, object: u64, vertex: u32, angle_deg: f64) -> Vec<u32> {
    let Some(o) = ws.scan.iter().find(|o| o.id == object).cloned() else { return Vec::new() };
    let Some(g) = graph_of(&o) else { return Vec::new() };
    let Some(start) = scan_offsets(ws).iter().find(|(id, _, _)| *id == object).map(|x| x.1) else { return Vec::new() };
    flood_by_normal(&g, vertex, angle_deg, None).into_iter().map(|v| v + start).collect()
}

/// How many points a region read by a box keeps unless told otherwise, and at most.
pub(crate) const SURFACE_REGION_LIMIT: usize = 20_000;
/// How far round a direction a normal may lean and still face it, degrees, unless told.
const WITHIN_DEG: f64 = 30.0;

/// The scan's vertices inside a box, facing a way where asked, less those in the boxes left
/// out, every k-th past the limit (ScanRuler's `regionOf` for a box) — the build's numbers.
pub(crate) fn region_in_box(ws: &mut ReverseWorkspace, bounds: &BoxBounds, normal: Option<(DVec3, Option<f64>)>, exclude: &[BoxBounds], limit: Option<u32>) -> Result<Vec<u32>, String> {
    for b in std::iter::once(bounds).chain(exclude) {
        if !(b.min.x <= b.max.x && b.min.y <= b.max.y && b.min.z <= b.max.z) {
            return Err("region: a box’s min lies past its max.".into());
        }
    }
    let inside = |b: &BoxBounds, p: DVec3| p.x >= b.min.x && p.x <= b.max.x && p.y >= b.min.y && p.y <= b.max.y && p.z >= b.min.z && p.z <= b.max.z;
    let facing = normal.and_then(|(d, w)| d.try_normalize().map(|d| (d, (w.unwrap_or(WITHIN_DEG)).to_radians().cos())));
    let offsets = scan_offsets(ws);
    let parts = ws.scan_parts();
    let mut kept = Vec::new();
    for (p, (id, start, _)) in parts.iter().zip(&offsets) {
        let mesh = ws.scan.iter().find(|o| o.id == *id).map(|o| o.mesh.clone());
        for v in &p.live_vertices {
            let at = p.positions[*v as usize];
            if !inside(bounds, at) || exclude.iter().any(|b| inside(b, at)) {
                continue;
            }
            if let (Some((d, cos)), Some(m)) = (facing, &mesh) {
                let n = if ws.flip { crate::app::flip_zy_out(m.vertex_normal(*v)) } else { m.vertex_normal(*v) };
                if n.dot(d) < cos {
                    continue;
                }
            }
            kept.push(start + v);
        }
    }
    let limit = limit.map_or(SURFACE_REGION_LIMIT, |l| l as usize).max(1);
    let step = kept.len().div_ceil(limit).max(1);
    Ok(if step > 1 { kept.into_iter().step_by(step).collect() } else { kept })
}

/// Before a build: each surface whose box moved with the parameters has its points read again
/// in the box (ScanRuler's `regionStale`), or says why not.
pub(crate) fn reread_stale_regions(ws: &mut ReverseWorkspace) {
    let stale: Vec<FeatureId> = ws
        .part
        .features
        .iter()
        .filter(|f| matches!(&f.kind, FeatureKind::Surface(s) if s.region_box.is_some() && matches!(s.region_stale, Some(RegionStale::Yes(true)))))
        .map(|f| f.id)
        .collect();
    if stale.is_empty() || ws.scan.is_empty() {
        return;
    }
    for id in stale {
        let Some(FeatureKind::Surface(s)) = ws.part.feature(id).map(|f| f.kind.clone()) else { continue };
        let Some(rb) = s.region_box.clone() else { continue };
        let read = region_in_box(ws, &rb.bounds, rb.normal.map(|n| (n.dir, n.within_deg)), rb.exclude.as_deref().unwrap_or(&[]), rb.limit);
        if let Some(FeatureKind::Surface(s)) = ws.part.feature_mut(id).map(|f| &mut f.kind) {
            match read {
                Ok(region) if region.len() >= mm_reverse::build::SURFACE_MIN_REGION => {
                    s.region = region;
                    s.region_stale = None;
                }
                Ok(region) => s.region_stale = Some(RegionStale::Why(format!("{} scan point{} in the box — a surface needs {} at least", region.len(), if region.len() == 1 { "" } else { "s" }, mm_reverse::build::SURFACE_MIN_REGION))),
                Err(e) => s.region_stale = Some(RegionStale::Why(e)),
            }
        }
    }
}

/// What a box does once its feature is made: a thickened sheet is put out of view.
pub(crate) fn after_commit(ws: &mut ReverseWorkspace, id: FeatureId) {
    if let Some(FeatureKind::Thicken(t)) = ws.part.feature(id).map(|f| &f.kind) {
        ws.hidden_bodies.insert(t.sheet);
    }
}

/// A length as the boxes write it: up to `digits` decimals, trailing zeros off, and the unit.
pub(crate) fn mm(v: f64, digits: usize) -> String {
    let s = format!("{v:.digits$}");
    let s = if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.').to_string() } else { s };
    format!("{s} mm")
}

/// A count with thousands set off, as the boxes write it.
pub(crate) fn count(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// The bodies standing that are sheets, in the order shown.
pub(crate) fn sheets(ws: &ReverseWorkspace) -> Vec<BodyId> {
    ws.built.bodies.iter().filter(|b| b.surface).map(|b| b.id).collect()
}

/// The default body of a feature on a body: the last solid body, else the last body.
pub(crate) fn last_solid(ws: &ReverseWorkspace) -> Option<BodyId> {
    ws.built.bodies.iter().rev().find(|b| !b.surface).or(ws.built.bodies.last()).map(|b| b.id)
}

/// Whether there is a solid body for a Cut to take from.
pub(crate) fn can_cut(ws: &ReverseWorkspace) -> bool {
    ws.built.bodies.iter().any(|b| !b.surface)
}

/// The Result select of a feature that makes a solid: Join, Cut (when there is a solid body to
/// cut), New body — each with what it does.
pub(crate) fn op_field(ui: &mut egui::Ui, op: &mut BodyOp, can_cut: bool) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label("Result");
        let name = op.name();
        egui::ComboBox::from_id_salt("re_surf_op").selected_text(name).show_ui(ui, |ui| {
            changed |= ui.selectable_value(op, BodyOp::Join, "Join").on_hover_text("Made one with every body it touches; a body of its own where it touches none, one body of all where it bridges several").changed();
            ui.add_enabled_ui(can_cut, |ui| {
                changed |= ui.selectable_value(op, BodyOp::Cut, "Cut").on_hover_text("Taken out of every body it goes into; a body it parts is a body a piece").changed();
            });
            changed |= ui.selectable_value(op, BodyOp::New, "New body").on_hover_text("A body of its own, whatever it overlaps — a tool to cut the scan with in Mesh Tools").changed();
        });
    });
    changed
}

/// A number field with its label, unit and what it means; whether it changed.
pub(crate) fn num(ui: &mut egui::Ui, label: &str, value: &mut f64, unit: &str, range: std::ops::RangeInclusive<f64>, speed: f64, hint: &str) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(label).on_hover_text(hint);
        let mut d = egui::DragValue::new(value).range(range).speed(speed).max_decimals(4);
        if !unit.is_empty() {
            d = d.suffix(format!(" {unit}"));
        }
        changed = ui.add(d).on_hover_text(hint).changed();
    });
    changed
}

/// The open box's feature's report from the last build, by the feature it previews.
pub(crate) fn report(ws: &ReverseWorkspace) -> Option<&mm_reverse::build::SurfReport> {
    ws.preview_feature.and_then(|id| ws.built.surf.get(&id))
}

/// Lines in the part drawn over the view in a colour.
pub(crate) fn paint_lines(ws: &ReverseWorkspace, painter: &egui::Painter, vp: Viewport, lines: &[Vec<DVec3>], color: egui::Color32) {
    for l in lines {
        let pts: Vec<Pos2> = l.iter().filter_map(|q| ws.project(vp, *q)).collect();
        if pts.len() >= 2 {
            painter.add(egui::Shape::line(pts, egui::Stroke::new(2.5, color)));
        }
    }
}

/// The red a failure or an open edge is drawn in on the part.
pub(crate) const RED: egui::Color32 = egui::Color32::from_rgb(235, 70, 60);

/// Whether a feature box is open on a kind.
pub(crate) fn open_kind(ws: &ReverseWorkspace) -> Option<&'static str> {
    match &ws.mode {
        Mode::Feature(e) => Some(e.kind.kind()),
        _ => None,
    }
}

/// The open edges of every trim shown that does not close, red on the part, and where a failed
/// thicken's faces fold — while no box of its own draws them.
pub(crate) fn paint_open_edges(ws: &ReverseWorkspace, painter: &egui::Painter, vp: Viewport) {
    if matches!(open_kind(ws), Some("trim") | Some("thicken")) {
        return;
    }
    for (id, r) in &ws.built.surf {
        let shown = ws.bodies.iter().any(|b| b.id.maker() == *id && !ws.hidden_bodies.contains(&b.id));
        if let Some(t) = r.trim.as_ref().filter(|t| !t.closed && shown) {
            paint_lines(ws, painter, vp, &t.open_edges, RED);
        } else if ws.built.errors.contains_key(id) && !r.where_lines.is_empty() {
            paint_lines(ws, painter, vp, &r.where_lines, RED);
        }
    }
}
