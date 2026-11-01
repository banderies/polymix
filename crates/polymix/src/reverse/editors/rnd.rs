// SPDX-License-Identifier: AGPL-3.0-only
//! What the Fillet, Chamfer, Shell and Delete face boxes share: what a box remembers that the
//! feature does not (a fillet's mode and whether its radius was typed, the reading off the scan,
//! a chamfer's numbers for the modes not chosen, where a shell's wall came from, what a click
//! on a face takes for a delete), the body select, a number with the round figure it was
//! probably meant as beside it, the picks lit on the part, the arrow on the part that drags the
//! box's number, a window dragged over the bodies, and the boxes' script steps. Ported from
//! ScanRuler (`plugins/reverse/ui/FeatureEditor.tsx`, `featureFields.tsx`,
//! `plugins/reverse/state/reFeatureSlice.ts`, `reDrafts.ts`, `reGripSlice.ts`,
//! `plugins/reverse/viewer/bodyPicks.ts`, `varyingEnds.ts`, `core/preview.ts`,
//! `app/useReverseWorkspace.ts`, `app/useReversePlugin.tsx`), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only.
//!
//! What a box remembers is kept here, one box at a time, for the box open: it is let go
//! whenever a box opens, new or from the list (`ReverseWorkspace::open_feature`), and starts
//! again when the box shown is another (a box is told by where its feature is held and the
//! feature it previews as).

use std::cell::RefCell;
use std::sync::mpsc::{channel, Receiver};
use std::sync::Arc;

use egui::{Color32, Pos2, RichText, Stroke};
use glam::DVec3;
use mm_reverse::face_group::{delete_face_group, edge_joins, FacedTriangles};
use mm_reverse::fillet_estimate::{corner_unreadable, read_corner_cut, ReadShape};
use mm_reverse::fillet_read::ScanSection;
use mm_reverse::model::{BodyId, EdgeRef, Exprs, FaceRef, FeatureId, FeatureKind};
use mm_reverse::nominal::{nominal_of, NominalKind};
use mm_reverse::topo::{edge_distance, face_distance, match_edge, match_face, match_tolerance, split_edges, tangent_chain, MatchTolerance, Topology, CHAIN_ANGLE, CHAIN_JOIN_MM};

use crate::reverse::{FeatureEdit, Mode, ReverseWorkspace};
use crate::tools::Viewport;

// ---- colours (the layout's §6.3) ----

/// What a box has picked, on the part.
pub(crate) const PICK_GREEN: Color32 = Color32::from_rgb(0x3c, 0xb3, 0x71);
/// An edge a tangent chain carries the feature along to.
pub(crate) const CHAIN_GREEN: Color32 = Color32::from_rgb(0x8f, 0xd6, 0xae);
/// A crossing window's.
const CROSSING_BLUE: Color32 = Color32::from_rgb(0x4d, 0xb8, 0xff);
/// The grip's, as the ghost of a feature on a body is drawn.
const GRIP: Color32 = Color32::from_rgb(115, 204, 140);

/// The fillet radius when the scan shows no round at the edge, and a chamfer's distances to
/// start.
pub(crate) const DEFAULT_FILLET_MM: f64 = 1.0;
/// A shell's wall to start: 2 mm — a print's, a casting's.
pub(crate) const DEFAULT_WALL_MM: f64 = 2.0;

// ---- what a box remembers ----

/// The reading of a fillet's radius off the scan across its first edge.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub(crate) enum Estimate {
    #[default]
    Idle,
    Working,
    Found {
        radius: f64,
        points: usize,
    },
    None,
}

/// Where a shell's wall came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum ThicknessFrom {
    Map,
    #[default]
    Default,
    Typed,
}

/// The mean of the part's Thickness map, where one was measured — what a new shell's wall
/// starts at, to the twentieth of a millimetre. Polymix measures no Thickness map: none.
pub(crate) fn thickness_map_mean(_ws: &ReverseWorkspace) -> Option<f64> {
    None
}

/// A grip being dragged: its number when the drag began, and where the pointer took hold.
#[derive(Debug, Clone, Copy)]
pub(crate) struct GripDrag {
    pub start: f64,
    pub at: Pos2,
}

/// What a box remembers that its feature does not.
pub(crate) struct Draft {
    session: Option<(usize, Option<FeatureId>)>,
    /// A fillet's radius: variable from start to end, not one.
    pub variable: bool,
    pub radius_typed: bool,
    pub estimate: Estimate,
    estimate_key: Option<String>,
    estimate_job: Option<Receiver<Option<(f64, usize)>>>,
    /// What the reading came to, taken in while the box was not drawn — by `busy` — for the box
    /// to take when it is.
    estimate_came: Option<Option<(f64, usize)>>,
    /// A chamfer's second distance and angle, kept while another mode is chosen, and its flip.
    pub distance2: f64,
    pub angle_deg: f64,
    pub flip: bool,
    pub thickness_from: ThicknessFrom,
    /// A delete's click takes the rounds running on from a round too.
    pub tangent: bool,
    pub grip: Option<GripDrag>,
    pub grip_hover: bool,
    /// A window being dragged over the bodies: where it began and where the pointer is.
    pub window: Option<(Pos2, Pos2)>,
    /// The viewport as last drawn, for a script step's window.
    pub vp: Option<Viewport>,
}

impl Default for Draft {
    fn default() -> Self {
        Draft {
            session: None,
            variable: false,
            radius_typed: false,
            estimate: Estimate::Idle,
            estimate_key: None,
            estimate_job: None,
            estimate_came: None,
            distance2: DEFAULT_FILLET_MM,
            angle_deg: 45.0,
            flip: false,
            thickness_from: ThicknessFrom::Default,
            tangent: true,
            grip: None,
            grip_hover: false,
            window: None,
            vp: None,
        }
    }
}

thread_local! {
    static DRAFT: RefCell<Draft> = RefCell::new(Draft::default());
    /// The scan as a section reads it, for the scan it was made of.
    static SECTION: RefCell<Option<(usize, Arc<ScanSection>)>> = const { RefCell::new(None) };
}

/// The open box's draft, to read or change.
pub(crate) fn with_draft<T>(f: impl FnOnce(&mut Draft) -> T) -> T {
    DRAFT.with(|d| f(&mut d.borrow_mut()))
}

/// A box opens: what the last one remembered is let go (the viewport kept).
pub(crate) fn forget() {
    with_draft(|d| {
        let vp = d.vp;
        *d = Draft { vp, ..Draft::default() };
    });
}

/// The draft of the box whose feature is held at `feature` (its address), started again —
/// `init` told whether the box edits a feature of the list — where it is another box than the
/// one remembered.
pub(crate) fn session<T>(ws: &ReverseWorkspace, feature: usize, init: impl FnOnce(&mut Draft, bool), f: impl FnOnce(&mut Draft) -> T) -> T {
    let key = (feature, ws.preview_feature);
    let editing = ws.preview_feature.is_some_and(|id| ws.part.feature(id).is_some());
    with_draft(|d| {
        if d.session != Some(key) {
            let vp = d.vp;
            *d = Draft { vp, session: Some(key), ..Draft::default() };
            init(d, editing);
        }
        f(d)
    })
}

/// Whether a box's work is still going: the reading of a round off the scan, until it comes in
/// — taken in here too, so a box put away before it came does not wait on it for ever.
pub(crate) fn busy() -> bool {
    with_draft(|d| take_estimate(d) && d.estimate_job.is_some())
}

/// The reading of a round taken in from its worker where it has come (or the worker is gone,
/// with nothing): kept for the box to take. Always true, for `busy` to go on with.
fn take_estimate(d: &mut Draft) -> bool {
    let Some(rx) = &d.estimate_job else { return true };
    match rx.try_recv() {
        Ok(r) => {
            d.estimate_job = None;
            d.estimate_came = Some(r);
        }
        Err(std::sync::mpsc::TryRecvError::Empty) => {}
        Err(std::sync::mpsc::TryRecvError::Disconnected) => {
            d.estimate_job = None;
            d.estimate_came = Some(None);
        }
    }
    true
}

// ---- the bodies ----

/// The body a new feature on a body starts on: the last solid body, else the last body.
pub(crate) fn last_solid(ws: &ReverseWorkspace) -> Option<BodyId> {
    let solid = ws.bodies.iter().rev().find(|b| ws.built.body(b.id).is_some_and(|x| !x.surface)).map(|b| b.id);
    solid.or_else(|| ws.bodies.last().map(|b| b.id))
}

/// The field `Body` and a select of every body by name. Returns whether it changed.
pub(crate) fn body_select(ui: &mut egui::Ui, ws: &ReverseWorkspace, body: &mut BodyId, salt: &str) -> bool {
    let bodies: Vec<(BodyId, String)> = ws.pick_set().iter().map(|b| (b.id, b.name.clone())).collect();
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label("Body");
        let shown = bodies.iter().find(|b| b.0 == *body).map_or_else(|| "Select…".to_string(), |b| b.1.clone());
        egui::ComboBox::from_id_salt(("re_rnd_body", salt)).selected_text(shown).show_ui(ui, |ui| {
            for (id, name) in &bodies {
                if ui.selectable_label(*id == *body, name).clicked() && *id != *body {
                    *body = *id;
                    changed = true;
                }
            }
        });
    });
    changed
}

/// A small ⓘ beside a label: its title and what it says on hover.
pub(crate) fn info_dot(ui: &mut egui::Ui, title: &str, text: &str) {
    ui.label(RichText::new("ⓘ").weak()).on_hover_ui(|ui| {
        ui.set_max_width(320.0);
        ui.strong(title);
        ui.label(text);
    });
}

/// The `Clear` key beside a pick count. Returns whether it was pressed.
pub(crate) fn clear_picks(ui: &mut egui::Ui) -> bool {
    ui.small_button("Clear").on_hover_text("Let every picked one go and pick again").clicked()
}

/// A hint line, in the warning colour when `warn`.
pub(crate) fn hint(ui: &mut egui::Ui, text: &str, warn: bool) {
    if warn {
        ui.colored_label(Color32::from_rgb(240, 200, 110), text);
    } else {
        ui.weak(text);
    }
}

/// The text a nominal chip reads: enough decimals to show a half.
pub(crate) fn nominal_text(value: f64, kind: NominalKind) -> String {
    match kind {
        NominalKind::Angle => format!("{value:.0}°"),
        NominalKind::Draft => {
            if value.fract() == 0.0 {
                format!("{value:.0}°")
            } else {
                format!("{value:.1}°")
            }
        }
        _ => {
            if value.fract() == 0.0 {
                format!("{value:.0} mm")
            } else {
                format!("{value:.1} mm")
            }
        }
    }
}

/// A number in the box: its label with what it does on hover, the round figure the value was
/// probably meant as — one click takes it, not while an expression drives it — and the number,
/// in millimetres or degrees. A number typed over an expression lets the expression go (the
/// slot's in `exprs`). Returns whether the value changed.
#[allow(clippy::too_many_arguments)]
pub(crate) fn num(ui: &mut egui::Ui, label: &str, hint: &str, value: &mut f64, nominal: NominalKind, range: std::ops::RangeInclusive<f64>, exprs: Option<(&mut Exprs, &str)>, salt: &str) -> bool {
    let degrees = matches!(nominal, NominalKind::Angle | NominalKind::Draft);
    let (expr, mut exprs) = match exprs {
        Some((e, slot)) => (e.get(slot).map(str::to_string), Some((e, slot))),
        None => (None, None),
    };
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(label);
        if !hint.is_empty() {
            info_dot(ui, label, hint);
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let r = ui.push_id(("re_rnd_num", salt), |ui| ui.add(egui::DragValue::new(value).range(range).speed(if degrees { 0.2 } else { 0.05 }).max_decimals(3).suffix(if degrees { "°" } else { " mm" }))).inner;
            let r = match &expr {
                Some(e) => r.on_hover_text(format!("= {e}")),
                None => r,
            };
            if r.changed() {
                changed = true;
            }
            if expr.is_none() {
                if let Some(round) = nominal_of(*value, nominal) {
                    let chip = ui.add(egui::Button::new(RichText::new(format!("→ {}", nominal_text(round, nominal))).small()).small()).on_hover_text("The round number this was probably meant as — click to take it");
                    if chip.clicked() {
                        *value = round;
                        changed = true;
                    }
                }
            }
        });
    });
    if changed {
        if let Some((e, slot)) = exprs.as_mut() {
            e.remove(slot);
        }
    }
    changed
}

// ---- picks on the bodies ----

/// How closely two fingerprints of one face or edge, read off two builds of the same body,
/// agree: a hair of the part.
pub(crate) fn same_tolerance(ws: &ReverseWorkspace, body: BodyId) -> MatchTolerance {
    let size = ws.pick_view(body).map_or(100.0, |b| {
        let bb = b.mesh.bounds();
        if bb.is_empty() {
            100.0
        } else {
            bb.diagonal()
        }
    });
    MatchTolerance { position: 1e-4 * size.max(1.0), angle: 1e-3, size: 1e-3 }
}

/// The index of a picked edge on the body as it stood before the feature, by its fingerprint.
pub(crate) fn edge_index(ws: &ReverseWorkspace, body: BodyId, e: &EdgeRef) -> Option<usize> {
    let b = ws.pick_view(body)?;
    let tol = match_tolerance(b.mesh.bounds().diagonal());
    let prints = b.topology.edge_prints();
    match_edge(e, &prints, &tol).filter(|i| b.topology.edges.get(*i).is_some_and(|x| edge_distance(e, &x.fingerprint, &tol).is_some()))
}

/// The index of a picked face on the body as it stood before the feature.
pub(crate) fn face_index(ws: &ReverseWorkspace, body: BodyId, f: &FaceRef) -> Option<usize> {
    let b = ws.pick_view(body)?;
    let tol = match_tolerance(b.mesh.bounds().diagonal());
    match_face(f, &b.topology.face_prints(), &tol)
}

/// The edges a round on these picks reaches along their tangent chains, the picks left out —
/// lit fainter, so what one click takes is seen before the kernel is asked.
pub(crate) fn chain_of(t: &Topology, picked: &[usize]) -> Vec<usize> {
    let seeds: Vec<u32> = picked.iter().map(|i| *i as u32).collect();
    tangent_chain(&t.edges, &seeds, &split_edges(t), CHAIN_ANGLE, CHAIN_JOIN_MM).into_iter().map(|h| h as usize).filter(|h| !picked.contains(h)).collect()
}

/// An edge of the body drawn in a colour, 3 px.
pub(crate) fn paint_edge(ws: &ReverseWorkspace, painter: &egui::Painter, vp: Viewport, body: BodyId, edge: usize, color: Color32) {
    let Some(e) = ws.pick_view(body).and_then(|b| b.topology.edges.get(edge)) else { return };
    let pts: Vec<Pos2> = e.polyline.iter().filter_map(|q| ws.project(vp, *q)).collect();
    if pts.len() >= 2 {
        painter.add(egui::Shape::line(pts, Stroke::new(3.0, color)));
    }
}

/// Picked edges lit green on the body, and with `chain` the edges their tangent chains reach
/// in a lighter green; a pick not found there is a dot at its middle.
pub(crate) fn paint_edge_picks(ws: &ReverseWorkspace, painter: &egui::Painter, vp: Viewport, body: BodyId, edges: &[EdgeRef], chain: bool) {
    let mut found = Vec::new();
    let mut missing = Vec::new();
    for e in edges {
        match edge_index(ws, body, e) {
            Some(i) => found.push(i),
            None => missing.push(e.clone()),
        }
    }
    super::paint_edges(ws, painter, vp, body, &missing);
    if chain {
        if let Some(b) = ws.pick_view(body) {
            for i in chain_of(&b.topology, &found) {
                paint_edge(ws, painter, vp, body, i, CHAIN_GREEN);
            }
        }
    }
    for i in found {
        paint_edge(ws, painter, vp, body, i, PICK_GREEN);
    }
}

/// A face of the body laid over in a colour: its triangles that face the camera.
pub(crate) fn paint_face(ws: &ReverseWorkspace, painter: &egui::Painter, vp: Viewport, body: BodyId, face: usize, color: Color32) {
    let Some(b) = ws.pick_view(body) else { return };
    let m = &b.part_mesh;
    let mut mesh = egui::Mesh::default();
    for (t, f) in m.triangles.iter().zip(&m.tri_face) {
        if *f as usize != face {
            continue;
        }
        let Some(p) = t.iter().map(|v| m.positions.get(*v as usize).and_then(|q| ws.project(vp, *q))).collect::<Option<Vec<Pos2>>>() else { continue };
        // Facing the camera: wound counter-clockwise on the screen (y down).
        if (p[1] - p[0]).x * (p[2] - p[0]).y - (p[1] - p[0]).y * (p[2] - p[0]).x >= 0.0 {
            continue;
        }
        let base = mesh.vertices.len() as u32;
        for q in p {
            mesh.colored_vertex(q, color);
        }
        mesh.add_triangle(base, base + 1, base + 2);
    }
    if !mesh.is_empty() {
        painter.add(egui::Shape::mesh(mesh));
    }
}

/// Picked faces laid over green on the body; a pick not found there is a dot at its centroid.
pub(crate) fn paint_face_picks(ws: &ReverseWorkspace, painter: &egui::Painter, vp: Viewport, body: BodyId, faces: &[FaceRef]) {
    let mut missing = Vec::new();
    for f in faces {
        match face_index(ws, body, f) {
            Some(i) => paint_face(ws, painter, vp, body, i, PICK_GREEN.gamma_multiply(0.5)),
            None => missing.push(*f),
        }
    }
    super::paint_faces(ws, painter, vp, body, &missing);
    // The face under the cursor, a lighter white over it, drawn through.
    if let Some((b, f)) = ws.view.hover_face {
        if b == body {
            paint_face(ws, painter, vp, b, f, Color32::from_white_alpha(90));
        }
    }
}

/// A click on a face that takes `all` (the face and what goes with it): on a face picked, all
/// that it takes let go; else what is not picked yet added, in order.
pub(crate) fn toggle_group(faces: &mut Vec<FaceRef>, clicked: &FaceRef, all: Vec<FaceRef>, tol: &MatchTolerance) {
    let same = |a: &FaceRef, c: &FaceRef| face_distance(a, c, tol).is_some();
    if faces.iter().any(|f| same(f, clicked)) {
        faces.retain(|f| !all.iter().any(|g| same(f, g)));
    } else {
        for g in all {
            if !faces.iter().any(|f| same(f, &g)) {
                faces.push(g);
            }
        }
    }
}

/// A point on the part, lit.
pub(crate) fn dot(ws: &ReverseWorkspace, painter: &egui::Painter, vp: Viewport, at: DVec3, color: Color32) {
    if let Some(q) = ws.project(vp, at) {
        painter.circle_filled(q, 5.5, color);
        painter.circle_stroke(q, 5.5, Stroke::new(1.0, Color32::BLACK));
    }
}

/// A face click for a box that takes faces: the face picked as its fingerprint, a second click
/// lets it go; a face of another body starts the picks again on that body. With `group`, what a
/// click takes for a delete: the whole of the round, the hole or the boss the face is part of,
/// and on a picked face, all that one took let go. Returns whether the picks changed.
pub(crate) fn toggle_face(ws: &mut ReverseWorkspace, vp: Viewport, p: Pos2, clicked: bool, body: &mut BodyId, faces: &mut Vec<FaceRef>, group: Option<bool>) -> bool {
    let Some(crate::reverse::view::Hit::Body { body: b, face, .. }) = ws.pick_hit(vp, p) else { return false };
    ws.view.hover_face = Some((b, face));
    if !clicked {
        return false;
    }
    let Some(view) = ws.pick_view(b) else { return false };
    let Some(print) = view.topology.faces.get(face).map(|f| f.fingerprint) else { return false };
    let all: Vec<FaceRef> = match group {
        Some(tangent) => {
            let m = &view.part_mesh;
            let joins = edge_joins(FacedTriangles { positions: &m.positions, normals: &m.normals, triangles: &m.triangles, tri_face: &m.tri_face }, &view.topology);
            delete_face_group(&view.topology, face as u32, |e| joins.get(&e).copied(), tangent).into_iter().filter_map(|h| view.topology.face(h)).map(|f| f.fingerprint).collect()
        }
        None => vec![print],
    };
    if b != *body {
        *body = b;
        faces.clear();
    }
    let tol = same_tolerance(ws, b);
    toggle_group(faces, &print, all, &tol);
    true
}

// ---- the grip ----

/// How long a feature's arrow is on the screen, in pixels, at any zoom.
pub(crate) const ARROW_PX: f32 = 44.0;

/// Where a grip stands: the point on the part and the way it points (part coordinates).
#[derive(Debug, Clone, Copy)]
pub(crate) struct Grip {
    pub at: DVec3,
    pub dir: DVec3,
}

/// The grip of a fillet or a chamfer: an arrow at the middle of its first edge, pointing into
/// the body along the bisector of the two faces' outward normals — pull it in and the round
/// grows. `None` with no edge or no faces to bisect.
pub(crate) fn edge_grip(ws: &ReverseWorkspace, body: BodyId, edge: &EdgeRef) -> Option<Grip> {
    let i = edge_index(ws, body, edge)?;
    let b = ws.pick_view(body)?;
    let e = b.topology.edges.get(i)?;
    let mut faces: Vec<u32> = Vec::new();
    for f in &e.faces {
        if !faces.contains(f) {
            faces.push(*f);
        }
    }
    let [fa, fb] = faces[..] else { return None };
    let n = |f: u32| face_normal_near(b, f as usize, e.fingerprint.mid);
    let bisector = (n(fa)? + n(fb)?).try_normalize()?;
    Some(Grip { at: e.fingerprint.mid, dir: -bisector })
}

/// The grip of a shell: an arrow on the first opened face, into the body along its normal —
/// pull it in and the wall thickens.
pub(crate) fn face_grip(ws: &ReverseWorkspace, body: BodyId, face: &FaceRef) -> Option<Grip> {
    let i = face_index(ws, body, face)?;
    let b = ws.pick_view(body)?;
    let f = b.topology.faces.get(i)?;
    let n = face_normal_near(b, i, f.point)?;
    Some(Grip { at: f.point, dir: -n })
}

/// The outward normal of a face near a point, off the body's mesh.
fn face_normal_near(b: &crate::reverse::BodyView, face: usize, at: DVec3) -> Option<DVec3> {
    let m = &b.part_mesh;
    let mut best: Option<(f64, DVec3)> = None;
    for (t, f) in m.triangles.iter().zip(&m.tri_face) {
        if *f as usize != face {
            continue;
        }
        let [a, bb, c] = t.map(|v| m.positions.get(v as usize).copied().unwrap_or_default());
        let d = ((a + bb + c) / 3.0 - at).length_squared();
        if best.is_none_or(|(x, _)| d < x) {
            best = Some((d, (bb - a).cross(c - a)));
        }
    }
    best.and_then(|(_, n)| n.try_normalize())
}

/// The grip's arrow on the screen: its foot, its tip, and how many pixels a millimetre along it
/// is there.
fn grip_screen(ws: &ReverseWorkspace, vp: Viewport, g: Grip) -> Option<(Pos2, Pos2, f32)> {
    let foot = ws.project(vp, g.at)?;
    let one = ws.project(vp, g.at + g.dir)?;
    let per_mm = (one - foot).length();
    if !(per_mm > 1e-6) {
        return None;
    }
    let tip = foot + (one - foot) / per_mm * ARROW_PX;
    Some((foot, tip, per_mm))
}

/// The grip drawn: an arrow, white while hovered or dragged.
pub(crate) fn paint_grip(ws: &ReverseWorkspace, painter: &egui::Painter, vp: Viewport, g: Grip) {
    let Some((foot, tip, _)) = grip_screen(ws, vp, g) else { return };
    let lit = with_draft(|d| d.grip_hover || d.grip.is_some());
    let color = if lit { Color32::WHITE } else { GRIP };
    let along = (tip - foot).normalized();
    let side = egui::vec2(-along.y, along.x);
    painter.line_segment([foot, tip - along * 8.0], Stroke::new(3.0, color));
    painter.add(egui::Shape::convex_polygon(vec![tip, tip - along * 12.0 + side * 6.0, tip - along * 12.0 - side * 6.0], color, Stroke::new(1.0, Color32::BLACK)));
    painter.circle_filled(foot, 3.0, color);
}

/// Distance from a point to a segment on the screen.
fn seg_dist(p: Pos2, a: Pos2, b: Pos2) -> f32 {
    let ab = b - a;
    let t = if ab.length_sq() > 0.0 { ((p - a).dot(ab) / ab.length_sq()).clamp(0.0, 1.0) } else { 0.0 };
    (a + ab * t - p).length()
}

/// A length dragged: millimetres pulled along the arrow since the drag began, applied to what
/// it began at, kept positive, to the hundredth.
pub(crate) fn drag_length(start: f64, delta: f64) -> f64 {
    ((start + delta).max(0.01) * 100.0).round() / 100.0
}

// ---- the window over the bodies ----

/// What a window dragged on screen takes of the bodies: dragged to the right, the faces and
/// edges lying wholly inside it; dragged to the left, those it touches as well — the way
/// AutoCAD and Fusion tell a window from a crossing. What lies behind a body is taken with the
/// rest: the far edges of a box are what a fillet all round it wants. Each with its body. An
/// edge no CAD system draws is not taken: a seam, where a round face meets itself, nor a split
/// between two faces on one surface — a round there has no corner to take off.
pub(crate) fn items_in_window(ws: &ReverseWorkspace, vp: Viewport, a: Pos2, b: Pos2, edges: bool, faces: bool) -> (Vec<(BodyId, usize)>, Vec<(BodyId, usize)>) {
    let crossing = b.x < a.x;
    let r = egui::Rect::from_two_pos(a, b);
    let inside = |p: Option<Pos2>| p.is_some_and(|p| r.contains(p));
    let mut out_edges = Vec::new();
    let mut out_faces = Vec::new();
    for body in ws.pick_set().iter().filter(|x| !ws.hidden_bodies.contains(&x.id)) {
        if edges {
            let splits = split_edges(&body.topology);
            for (i, e) in body.topology.edges.iter().enumerate() {
                // A seam or a split is no edge to pick.
                if e.polyline.len() < 2 || (e.faces.len() == 2 && e.faces[0] == e.faces[1]) || splits.contains(&e.hash) {
                    continue;
                }
                let pts: Vec<Option<Pos2>> = e.polyline.iter().map(|q| ws.project(vp, *q)).collect();
                let mut take = pts.iter().all(|p| inside(*p));
                if !take && crossing {
                    take = pts.iter().any(|p| inside(*p)) || pts.windows(2).any(|w| matches!((w[0], w[1]), (Some(p), Some(q)) if segment_meets_box(p, q, r)));
                }
                if take {
                    out_edges.push((body.id, i));
                }
            }
        }
        if faces {
            let m = &body.part_mesh;
            let n = body.topology.faces.len();
            let mut all = vec![true; n];
            let mut any = vec![false; n];
            let mut has = vec![false; n];
            for (t, f) in m.triangles.iter().zip(&m.tri_face) {
                let f = *f as usize;
                if f >= n {
                    continue;
                }
                has[f] = true;
                for v in t {
                    let hit = inside(m.positions.get(*v as usize).and_then(|q| ws.project(vp, *q)));
                    any[f] |= hit;
                    all[f] &= hit;
                }
            }
            for f in 0..n {
                if has[f] && if crossing { any[f] } else { all[f] } {
                    out_faces.push((body.id, f));
                }
            }
        }
    }
    (out_edges, out_faces)
}

/// Whether a segment on screen meets a box: an end inside it, or a crossing of one of its
/// sides.
fn segment_meets_box(a: Pos2, b: Pos2, r: egui::Rect) -> bool {
    if r.contains(a) || r.contains(b) {
        return true;
    }
    let corners = [r.left_top(), r.right_top(), r.right_bottom(), r.left_bottom()];
    let side = |p: Pos2, q: Pos2, s: Pos2| (q.x - p.x) * (s.y - p.y) - (q.y - p.y) * (s.x - p.x);
    (0..4).any(|i| {
        let (c, d) = (corners[i], corners[(i + 1) % 4]);
        side(a, b, c) * side(a, b, d) < 0.0 && side(c, d, a) * side(c, d, b) < 0.0
    })
}

/// The body most of the items lie on — of two with as many, the one met first.
pub(crate) fn most_on(items: &[(BodyId, usize)]) -> Option<BodyId> {
    let mut counts: Vec<(BodyId, usize)> = Vec::new();
    for (b, _) in items {
        match counts.iter_mut().find(|c| c.0 == *b) {
            Some(c) => c.1 += 1,
            None => counts.push((*b, 1)),
        }
    }
    let mut best: Option<(BodyId, usize)> = None;
    for c in counts {
        if best.is_none_or(|b| c.1 > b.1) {
            best = Some(c);
        }
    }
    best.map(|c| c.0)
}

/// Whether the pointer moved far enough from where a window began to be a window: more than
/// 6 px across and down together — a Shift+click that moves less is a click that adds.
pub(crate) fn window_moved(a: Pos2, b: Pos2) -> bool {
    (b.x - a.x).abs() + (b.y - a.y).abs() > 6.0
}

/// What a window takes, into the open box: a fillet's or a chamfer's edges, a shell's, a
/// draft's or a delete's faces — of the body the box is on, or of the body most of them lie on.
/// Returns whether the box changed.
pub(crate) fn window_into(ws: &ReverseWorkspace, vp: Viewport, a: Pos2, b: Pos2, edit: &mut FeatureEdit) -> bool {
    match &mut edit.kind {
        FeatureKind::Fillet(x) => {
            let (edges, _) = items_in_window(ws, vp, a, b, true, false);
            let body = if ws.pick_view(x.body).is_some() { Some(x.body) } else { most_on(&edges) };
            let Some(body) = body else { return false };
            let picks: Vec<EdgeRef> = edges.iter().filter(|e| e.0 == body).filter_map(|e| ws.pick_view(body).and_then(|v| v.topology.edges.get(e.1)).map(|x| x.fingerprint.clone())).collect();
            super::fillet::add_edges(ws, x, body, picks)
        }
        FeatureKind::Chamfer(x) => {
            let (edges, _) = items_in_window(ws, vp, a, b, true, false);
            let body = if ws.pick_view(x.body).is_some() { Some(x.body) } else { most_on(&edges) };
            let Some(body) = body else { return false };
            let picks: Vec<EdgeRef> = edges.iter().filter(|e| e.0 == body).filter_map(|e| ws.pick_view(body).and_then(|v| v.topology.edges.get(e.1)).map(|x| x.fingerprint.clone())).collect();
            add_edges(ws, &mut x.body, &mut x.edges, body, picks)
        }
        FeatureKind::Shell(x) => {
            let (_, faces) = items_in_window(ws, vp, a, b, false, true);
            let body = if ws.pick_view(x.body).is_some() { Some(x.body) } else { most_on(&faces) };
            let Some(body) = body else { return false };
            add_faces(ws, &mut x.body, &mut x.faces, body, &faces)
        }
        FeatureKind::DeleteFace(x) => {
            let (_, faces) = items_in_window(ws, vp, a, b, false, true);
            let body = if ws.pick_view(x.body).is_some() { Some(x.body) } else { most_on(&faces) };
            let Some(body) = body else { return false };
            add_faces(ws, &mut x.body, &mut x.faces, body, &faces)
        }
        // A draft's faces to lean, as a click adds them past its neutral face.
        FeatureKind::Draft(x) => {
            let (_, faces) = items_in_window(ws, vp, a, b, false, true);
            let body = if ws.pick_view(x.body).is_some() { Some(x.body) } else { most_on(&faces) };
            let Some(body) = body else { return false };
            add_faces(ws, &mut x.body, &mut x.faces, body, &faces)
        }
        _ => false,
    }
}

/// Edges a window took, added to a box's (none twice); of another body, they start the picks
/// again there.
pub(crate) fn add_edges(ws: &ReverseWorkspace, body: &mut BodyId, edges: &mut Vec<EdgeRef>, on: BodyId, added: Vec<EdgeRef>) -> bool {
    if *body != on {
        *body = on;
        edges.clear();
    }
    let tol = same_tolerance(ws, on);
    let mut changed = false;
    for e in added {
        if !edges.iter().any(|x| edge_distance(x, &e, &tol).is_some()) {
            edges.push(e);
            changed = true;
        }
    }
    changed
}

/// Faces a window took, added to a box's.
fn add_faces(ws: &ReverseWorkspace, body: &mut BodyId, faces: &mut Vec<FaceRef>, on: BodyId, added: &[(BodyId, usize)]) -> bool {
    if *body != on {
        *body = on;
        faces.clear();
    }
    let tol = same_tolerance(ws, on);
    let mut changed = false;
    for (b, i) in added {
        if *b != on {
            continue;
        }
        let Some(f) = ws.pick_view(on).and_then(|v| v.topology.faces.get(*i)).map(|f| f.fingerprint) else { continue };
        if !faces.iter().any(|x| face_distance(x, &f, &tol).is_some()) {
            faces.push(f);
            changed = true;
        }
    }
    changed
}

/// The window being dragged: solid green to the right, a window; dashed blue to the left, a
/// crossing.
pub(crate) fn paint_window(painter: &egui::Painter) {
    let Some((a, b)) = with_draft(|d| d.window) else { return };
    if !window_moved(a, b) {
        return;
    }
    let r = egui::Rect::from_two_pos(a, b);
    if b.x < a.x {
        painter.rect_filled(r, 0.0, CROSSING_BLUE.gamma_multiply(0.12));
        let pts = vec![r.left_top(), r.right_top(), r.right_bottom(), r.left_bottom(), r.left_top()];
        painter.add(egui::Shape::dashed_line(&pts, Stroke::new(1.0, CROSSING_BLUE), 5.0, 4.0));
    } else {
        painter.rect_filled(r, 0.0, PICK_GREEN.gamma_multiply(0.12));
        painter.rect_stroke(r, 0.0, Stroke::new(1.0, PICK_GREEN), egui::StrokeKind::Middle);
    }
}

// ---- the pointer in the viewport, while a box of these is open ----

/// The open box's grip and number to drag, by kind.
fn grip_of(ws: &ReverseWorkspace, kind: &FeatureKind) -> Option<Grip> {
    match kind {
        FeatureKind::Fillet(x) => x.edges.first().and_then(|e| edge_grip(ws, x.body, e)),
        FeatureKind::Chamfer(x) => x.edges.first().and_then(|e| edge_grip(ws, x.body, e)),
        FeatureKind::Shell(x) => x.faces.first().and_then(|f| face_grip(ws, x.body, f)),
        _ => None,
    }
}

fn grip_value(kind: &FeatureKind) -> Option<f64> {
    match kind {
        FeatureKind::Fillet(x) => Some(x.radius),
        FeatureKind::Chamfer(x) => Some(x.distance),
        FeatureKind::Shell(x) => Some(x.thickness),
        _ => None,
    }
}

/// A number dragged on the part: as typed, the expression behind it let go.
fn set_grip_value(kind: &mut FeatureKind, v: f64) {
    match kind {
        FeatureKind::Fillet(x) => {
            x.radius = v;
            x.exprs.remove("radius");
            with_draft(|d| d.radius_typed = true);
        }
        FeatureKind::Chamfer(x) => {
            x.distance = v;
            x.exprs.remove("distance");
        }
        FeatureKind::Shell(x) => {
            x.thickness = v;
            x.exprs.remove("thickness");
            with_draft(|d| d.thickness_from = ThicknessFrom::Typed);
        }
        _ => {}
    }
}

/// Whether a kind's box takes the pointer here: these, and a Draft's window of faces.
fn ours(kind: &FeatureKind) -> bool {
    matches!(kind, FeatureKind::Fillet(_) | FeatureKind::Chamfer(_) | FeatureKind::Shell(_) | FeatureKind::DeleteFace(_) | FeatureKind::Draft(_))
}

/// The pointer in the viewport while a box is open, before a click reaches the box's editor:
/// the grip taken hold of and dragged, or a window dragged with Shift over the bodies. `Some`
/// with whether the box changed where these took the pointer; `None` leaves it to the click.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pointer(ws: &mut ReverseWorkspace, vp: Viewport, pos: Option<Pos2>, pressed: bool, down: bool, released: bool, shift: bool, edit: &mut FeatureEdit) -> Option<bool> {
    with_draft(|d| d.vp = Some(vp));
    if !ours(&edit.kind) {
        return None;
    }
    // A grip being dragged.
    if let Some(drag) = with_draft(|d| d.grip) {
        let (Some(p), Some(g)) = (pos, grip_of(ws, &edit.kind)) else {
            with_draft(|d| d.grip = None);
            return Some(true);
        };
        if let Some((foot, tip, per_mm)) = grip_screen(ws, vp, g) {
            let along = (tip - foot).normalized();
            let delta = ((p - drag.at).dot(along) / per_mm) as f64;
            let v = drag_length(drag.start, delta);
            if grip_value(&edit.kind) != Some(v) {
                set_grip_value(&mut edit.kind, v);
            }
        }
        if released || !down {
            with_draft(|d| d.grip = None);
            return Some(true);
        }
        // One build at a time while the arrow is pulled; the last one lands on release.
        return Some(!ws.building());
    }
    // A window being dragged.
    if let Some((a, _)) = with_draft(|d| d.window) {
        let Some(p) = pos else {
            with_draft(|d| d.window = None);
            return Some(false);
        };
        with_draft(|d| d.window = Some((a, p)));
        if released || !down {
            with_draft(|d| d.window = None);
            if window_moved(a, p) {
                return Some(window_into(ws, vp, a, p, edit));
            }
            // A Shift+click that did not move is a click that adds: the box's own.
            return None;
        }
        return Some(false);
    }
    let p = pos?;
    // The grip under the cursor, taken hold of on a press.
    let hover = grip_of(ws, &edit.kind).and_then(|g| grip_screen(ws, vp, g)).is_some_and(|(foot, tip, _)| seg_dist(p, foot, tip) <= 8.0);
    with_draft(|d| d.grip_hover = hover);
    if hover && pressed {
        if let Some(start) = grip_value(&edit.kind) {
            with_draft(|d| d.grip = Some(GripDrag { start, at: p }));
            return Some(false);
        }
    }
    if shift && pressed {
        with_draft(|d| d.window = Some((p, p)));
        return Some(false);
    }
    None
}

// ---- the round read off the scan ----

/// The scan as one mesh a section cuts, made once for the scan as it stands.
fn scan_section(ws: &mut ReverseWorkspace) -> Option<Arc<ScanSection>> {
    if ws.scan.is_empty() {
        return None;
    }
    let parts = ws.scan_parts();
    let key = Arc::as_ptr(&parts) as usize;
    if let Some(s) = SECTION.with(|s| s.borrow().as_ref().filter(|(k, _)| *k == key).map(|(_, s)| s.clone())) {
        return Some(s);
    }
    let mut positions = Vec::new();
    let mut triangles = Vec::new();
    for p in parts.iter() {
        let base = positions.len() as u32;
        positions.extend_from_slice(&p.positions);
        triangles.extend(p.triangles.iter().map(|t| [t[0] + base, t[1] + base, t[2] + base]));
    }
    let s = Arc::new(ScanSection::new(positions, triangles));
    SECTION.with(|c| *c.borrow_mut() = Some((key, s.clone())));
    Some(s)
}

/// The fillet's radius, read off the scan across the first picked edge: a section by the plane
/// square to the edge at its middle, the two faces as lines through the edge, the round between
/// them fitted. Only a straight edge between two flats is read; anything else keeps the default
/// until typed. A typed number is left alone. Read on a worker thread; what it found is taken
/// when it comes in. Returns whether the fillet changed.
pub(crate) fn estimate(ws: &mut ReverseWorkspace, ctx: &egui::Context, body: BodyId, edges: &[EdgeRef], radius: &mut f64) -> bool {
    let (typed, key_was) = with_draft(|d| (d.radius_typed, d.estimate_key.clone()));
    let key = format!("{body}:{}:{typed}", edges.iter().map(|e| format!("{:.4},{:.4},{:.4}", e.mid.x, e.mid.y, e.mid.z)).collect::<Vec<_>>().join(";"));
    let mut changed = false;
    // What came in.
    let came = with_draft(|d| {
        take_estimate(d);
        d.estimate_came.take()
    });
    if let Some(r) = came {
        if !with_draft(|d| d.radius_typed) {
            match r {
                Some((found, points)) => {
                    *radius = (found * 100.0).round() / 100.0;
                    with_draft(|d| d.estimate = Estimate::Found { radius: found, points });
                    changed = true;
                }
                None => with_draft(|d| d.estimate = Estimate::None),
            }
        }
    }
    if with_draft(|d| d.estimate_job.is_some()) {
        ctx.request_repaint_after(std::time::Duration::from_millis(50));
    }
    if key_was.as_deref() == Some(key.as_str()) {
        return changed;
    }
    with_draft(|d| {
        d.estimate_key = Some(key);
        d.estimate_job = None;
        d.estimate_came = None;
    });
    if typed {
        return changed;
    }
    let Some(edge) = edges.first().cloned() else {
        with_draft(|d| d.estimate = Estimate::Idle);
        return changed;
    };
    if corner_unreadable(&edge).is_some() {
        with_draft(|d| d.estimate = Estimate::None);
        return changed;
    }
    let Some(n) = edge.dir.try_normalize() else { return changed };
    // No scan to read: the box waits, as it does before an edge is picked.
    let Some(section) = scan_section(ws) else {
        with_draft(|d| d.estimate = Estimate::Idle);
        return changed;
    };
    with_draft(|d| d.estimate = Estimate::Working);
    let (tx, rx) = channel();
    with_draft(|d| d.estimate_job = Some(rx));
    let reach = (section.size() * 0.05).max(3.0);
    std::thread::Builder::new()
        .name("fillet estimate".into())
        .spawn(move || {
            let cut = section.cut(edge.mid, n, 0.0);
            let reading = read_corner_cut(&edge, edge.mid, &cut, reach, ReadShape::Round, None);
            let _ = tx.send(reading.map(|r| (r.radius, r.points)));
        })
        .ok();
    ctx.request_repaint_after(std::time::Duration::from_millis(50));
    changed
}

// ---- script steps (tests) ----

/// A point given as `x:y:z` among a step's words.
fn point(args: &[&str], from: usize) -> Option<DVec3> {
    let n = |i: usize| args.get(from + i).and_then(|v| v.parse::<f64>().ok());
    Some(DVec3::new(n(0)?, n(1)?, n(2)?))
}

/// The edge of the bodies nearest a point on the part.
fn edge_near(ws: &ReverseWorkspace, at: DVec3) -> Option<(BodyId, usize)> {
    let mut best: Option<((BodyId, usize), f64)> = None;
    for b in ws.pick_set() {
        for (i, e) in b.topology.edges.iter().enumerate() {
            for w in e.polyline.windows(2) {
                let (a, d) = (w[0], w[1] - w[0]);
                let t = if d.length_squared() > 0.0 { ((at - a).dot(d) / d.length_squared()).clamp(0.0, 1.0) } else { 0.0 };
                let dist = (a + d * t - at).length();
                if best.is_none_or(|(_, x)| dist < x) {
                    best = Some(((b.id, i), dist));
                }
            }
        }
    }
    best.map(|b| b.0)
}

/// The face of the bodies nearest a point on the part.
fn face_near(ws: &ReverseWorkspace, at: DVec3) -> Option<(BodyId, usize)> {
    let mut best: Option<((BodyId, usize), f64)> = None;
    for b in ws.pick_set() {
        let m = &b.part_mesh;
        for (t, f) in m.triangles.iter().zip(&m.tri_face) {
            let [p, q, r] = t.map(|v| m.positions.get(v as usize).copied().unwrap_or_default());
            let d = ((p + q + r) / 3.0 - at).length();
            if best.is_none_or(|(_, x)| d < x) {
                best = Some(((b.id, *f as usize), d));
            }
        }
    }
    best.map(|b| b.0)
}

/// The Reverse Engineer workspace's `rnd:` script steps, for the boxes of these kinds:
///
/// - `rnd:open:<fillet|chamfer|shell|deleteFace>` — the kind's box, as its button opens it;
/// - `rnd:edge:x:y:z` / `rnd:face:x:y:z` — a click on the edge or the face nearest that point;
/// - `rnd:set:<field>:<value>` — radius, mode (constant|variable), chain, type
///   (equal|twoDistances|distanceAngle), distance, distance2, angle, flip, thickness, tangent,
///   skip, vary:<i>:<start>:<end>, swap:<i>;
/// - `rnd:box:x0:y0:x1:y1` — a window dragged over the viewport (from its corner, in points);
/// - `rnd:fit` — the open fillet's radius (or chamfer's distance) fitted to the scan;
/// - `rnd:solids:<feature>:<radius>:<z0>:<z1>` — a fillet on each solid a feature made, on the
///   edges between two heights;
/// - `rnd:create`, `rnd:report`.
pub(crate) fn script_step(ws: &mut ReverseWorkspace, args: &[&str]) {
    let word = args.first().copied().unwrap_or("");
    match word {
        "open" => {
            let kind = args.get(1).copied().unwrap_or("fillet");
            match super::new_feature(kind, ws) {
                Ok(k) => ws.open_feature(k, None),
                Err(e) => ws.status = e,
            }
        }
        "edge" | "face" => {
            let Some(at) = point(args, 1) else { return };
            let Mode::Feature(mut edit) = std::mem::replace(&mut ws.mode, Mode::Idle) else { return };
            let hit = if word == "edge" { edge_near(ws, at) } else { face_near(ws, at) };
            let changed = match hit {
                Some(h) => match &mut edit.kind {
                    FeatureKind::Fillet(x) => super::fillet::toggle_edge(ws, x, h.0, h.1),
                    FeatureKind::Chamfer(x) => toggle_edge(ws, &mut x.body, &mut x.edges, h.0, h.1),
                    FeatureKind::Shell(x) => toggle_face_at(ws, &mut x.body, &mut x.faces, h, None),
                    FeatureKind::DeleteFace(x) => {
                        let tangent = with_draft(|d| d.tangent);
                        toggle_face_at(ws, &mut x.body, &mut x.faces, h, Some(tangent))
                    }
                    _ => false,
                },
                None => false,
            };
            ws.mode = Mode::Feature(edit);
            if changed {
                ws.request_build();
            }
        }
        "set" => {
            let Mode::Feature(mut edit) = std::mem::replace(&mut ws.mode, Mode::Idle) else { return };
            let changed = set_field(&mut edit.kind, &args[1..]);
            ws.mode = Mode::Feature(edit);
            if changed {
                ws.request_build();
            }
        }
        "box" => {
            let n: Vec<f32> = args[1..].iter().filter_map(|v| v.parse().ok()).collect();
            let (Some(vp), &[x0, y0, x1, y1]) = (with_draft(|d| d.vp), &n[..]) else { return };
            let o = vp.rect.min;
            let Mode::Feature(mut edit) = std::mem::replace(&mut ws.mode, Mode::Idle) else { return };
            let changed = window_into(ws, vp, o + egui::vec2(x0, y0), o + egui::vec2(x1, y1), &mut edit);
            ws.mode = Mode::Feature(edit);
            if changed {
                ws.request_build();
            }
        }
        "fit" => {
            let Mode::Feature(mut edit) = std::mem::replace(&mut ws.mode, Mode::Idle) else { return };
            let said = fit_open(ws, &mut edit.kind);
            ws.mode = Mode::Feature(edit);
            log::info!("reverse rnd fit: {said}");
            ws.status = said;
            ws.request_build();
        }
        "solids" => {
            let n = |i: usize| args.get(i).and_then(|v| v.parse::<f64>().ok());
            let (Some(feature), Some(radius), Some(z0), Some(z1)) = (n(1), n(2), n(3), n(4)) else { return };
            let maker = feature as FeatureId;
            let query = mm_reverse::pick_query::Where { bounds: Some(mm_reverse::pick_query::Bounds { min: DVec3::new(-1e6, -1e6, z0), max: DVec3::new(1e6, 1e6, z1) }), ..Default::default() };
            let round = FeatureKind::Fillet(mm_reverse::model::FilletFeature { body: BodyId::new(maker), edges: Vec::new(), edges_where: Vec::new(), radius, chain: true, skip_failing: false, variable: Vec::new(), exprs: Exprs::new() });
            match mm_reverse::rounds::rounds_on_solids(&ws.part, maker, &round, &[query], None) {
                Ok(on) => {
                    let part = on.part;
                    ws.edit_part(|p| *p = part);
                    ws.status = on.note;
                }
                Err(e) => ws.status = e,
            }
            log::info!("reverse rnd solids: {}", ws.status);
        }
        "create" => ws.commit_feature(),
        "report" => {
            let said = match &ws.mode {
                Mode::Feature(edit) => report(&edit.kind),
                _ => "no box open".into(),
            };
            log::info!("reverse rnd report: {said}");
        }
        other => log::warn!("unknown rnd script step {other}"),
    }
}

/// A click on an edge for a chamfer (and a fillet at one radius): picked, or let go.
pub(crate) fn toggle_edge(ws: &ReverseWorkspace, body: &mut BodyId, edges: &mut Vec<EdgeRef>, on: BodyId, edge: usize) -> bool {
    let Some(print) = ws.pick_view(on).and_then(|v| v.topology.edges.get(edge)).map(|e| e.fingerprint.clone()) else { return false };
    if *body != on {
        *body = on;
        edges.clear();
    }
    let tol = same_tolerance(ws, on);
    match edges.iter().position(|e| edge_distance(e, &print, &tol).is_some()) {
        Some(k) => {
            edges.remove(k);
        }
        None => edges.push(print),
    }
    true
}

/// A click on a face given by index, as `toggle_face` takes one under the cursor.
fn toggle_face_at(ws: &ReverseWorkspace, body: &mut BodyId, faces: &mut Vec<FaceRef>, hit: (BodyId, usize), group: Option<bool>) -> bool {
    let (b, face) = hit;
    let Some(view) = ws.pick_view(b) else { return false };
    let Some(print) = view.topology.faces.get(face).map(|f| f.fingerprint) else { return false };
    let all: Vec<FaceRef> = match group {
        Some(tangent) => {
            let m = &view.part_mesh;
            let joins = edge_joins(FacedTriangles { positions: &m.positions, normals: &m.normals, triangles: &m.triangles, tri_face: &m.tri_face }, &view.topology);
            delete_face_group(&view.topology, face as u32, |e| joins.get(&e).copied(), tangent).into_iter().filter_map(|h| view.topology.face(h)).map(|f| f.fingerprint).collect()
        }
        None => vec![print],
    };
    if b != *body {
        *body = b;
        faces.clear();
    }
    let tol = same_tolerance(ws, b);
    toggle_group(faces, &print, all, &tol);
    true
}

/// A box's field set by a script step.
fn set_field(kind: &mut FeatureKind, args: &[&str]) -> bool {
    let field = args.first().copied().unwrap_or("");
    let v = |i: usize| args.get(i).and_then(|x| x.parse::<f64>().ok());
    let on = args.get(1).is_some_and(|x| *x == "1" || *x == "true" || *x == "on");
    match (kind, field) {
        (FeatureKind::Fillet(x), "radius") => {
            let Some(r) = v(1) else { return false };
            x.radius = r.abs();
            x.exprs.remove("radius");
            with_draft(|d| d.radius_typed = true);
        }
        (FeatureKind::Fillet(x), "mode") => super::fillet::switch_mode(x, args.get(1) == Some(&"variable")),
        (FeatureKind::Fillet(x), "chain") => x.chain = on,
        (FeatureKind::Fillet(x), "skip") => x.skip_failing = on,
        (FeatureKind::Fillet(x), "vary") => {
            let (Some(i), Some(a), Some(b)) = (v(1), v(2), v(3)) else { return false };
            let Some(e) = x.variable.get_mut(i as usize) else { return false };
            e.radius = [a.abs(), b.abs()];
        }
        (FeatureKind::Fillet(x), "swap") => {
            let Some(e) = v(1).and_then(|i| x.variable.get_mut(i as usize)) else { return false };
            e.radius = [e.radius[1], e.radius[0]];
        }
        (FeatureKind::Chamfer(x), "type") => {
            let mode = match args.get(1).copied() {
                Some("twoDistances") => mm_reverse::model::ChamferMode::TwoDistances,
                Some("distanceAngle") => mm_reverse::model::ChamferMode::DistanceAngle,
                _ => mm_reverse::model::ChamferMode::Equal,
            };
            super::chamfer::set_mode(x, mode);
        }
        (FeatureKind::Chamfer(x), "distance") => {
            let Some(d) = v(1) else { return false };
            x.distance = d.abs();
        }
        (FeatureKind::Chamfer(x), "distance2") => {
            let Some(d) = v(1) else { return false };
            with_draft(|dr| dr.distance2 = d.abs());
            super::chamfer::write_mode(x);
        }
        (FeatureKind::Chamfer(x), "angle") => {
            let Some(a) = v(1) else { return false };
            with_draft(|dr| dr.angle_deg = a.abs().clamp(0.1, 89.9));
            super::chamfer::write_mode(x);
        }
        (FeatureKind::Chamfer(x), "flip") => {
            with_draft(|dr| dr.flip = on);
            super::chamfer::write_mode(x);
        }
        (FeatureKind::Chamfer(x), "chain") => x.chain = on,
        (FeatureKind::Chamfer(x), "skip") => x.skip_failing = on,
        (FeatureKind::Shell(x), "thickness") => {
            let Some(t) = v(1) else { return false };
            x.thickness = t.abs();
            with_draft(|d| d.thickness_from = ThicknessFrom::Typed);
        }
        (FeatureKind::DeleteFace(_), "tangent") => with_draft(|d| d.tangent = on),
        _ => return false,
    }
    true
}

/// The open fillet's radius or chamfer's distance fitted to the scan across its edges — the
/// median of their readings, a spread refused — or, a fillet's one varying edge, its law along
/// it. What the fit says.
fn fit_open(ws: &mut ReverseWorkspace, kind: &mut FeatureKind) -> String {
    let Some(section) = scan_section(ws) else { return "There is no scan to read a round off.".into() };
    let size = section.size();
    let mut cut = |o: DVec3, n: DVec3, m: f64| section.cut(o, n, m);
    let (body, edges, shape) = match kind {
        FeatureKind::Fillet(x) if !x.variable.is_empty() && x.edges.is_empty() => (x.body, x.variable.iter().map(|v| v.edge.clone()).collect::<Vec<_>>(), None),
        FeatureKind::Fillet(x) => (x.body, x.edges.clone(), Some(ReadShape::Round)),
        FeatureKind::Chamfer(x) => {
            if x.mode.is_some_and(|m| m != mm_reverse::model::ChamferMode::Equal) {
                return "Only an even chamfer’s distance is fitted — give the numbers of one by two distances or an angle.".into();
            }
            (x.body, x.edges.clone(), Some(ReadShape::Chamfer))
        }
        _ => return "Only a fillet's radius or a chamfer's distance is fitted.".into(),
    };
    let Some(view) = ws.pick_view(body) else { return "The body is gone.".into() };
    let t = view.topology.clone();
    // Each edge found again on the body, with the body's mesh for an edge of any shape — or,
    // lost there, read off its fingerprint alone: one reading per pick, in the order picked,
    // as the fit lists them and a variable edge takes its law.
    let mesh = faced_mesh(&view.part_mesh);
    let reads = edge_reads(&t, &mesh, &edges, |e| edge_index(ws, body, e));
    if reads.is_empty() {
        return "A fitted size needs the edges to read it across.".into();
    }
    match shape {
        None => {
            let mut said = Vec::new();
            for (i, r) in reads.iter().enumerate() {
                match mm_reverse::rounds::fit_edge_law(&mut cut, r, i, None, size) {
                    Ok(fit) => {
                        if let (FeatureKind::Fillet(x), Some(law)) = (&mut *kind, fit.law) {
                            if let Some(v) = x.variable.get_mut(i) {
                                v.radius = law;
                            }
                        }
                        said.push(fit.note.unwrap_or_default());
                    }
                    Err(e) => return e,
                }
            }
            said.join(" ")
        }
        Some(shape) => match mm_reverse::rounds::fit_edges(&mut cut, &reads, shape, None, size) {
            Ok(fit) => {
                match kind {
                    FeatureKind::Fillet(x) => {
                        x.radius = fit.value;
                        with_draft(|d| d.radius_typed = true);
                    }
                    FeatureKind::Chamfer(x) => x.distance = fit.value,
                    _ => {}
                }
                let sigma = fit.sigma.map_or(String::new(), |s| format!(" ± {s}"));
                format!("{}{sigma} mm fitted over {} points, nominal {}{}", fit.value, fit.points, fit.nominal, fit.warning.map_or(String::new(), |w| format!(" — {w}")))
            }
            Err(e) => e,
        },
    }
}

/// The picked edges as a fit reads them, one a pick in the order picked: found on the body by
/// `index`, with its faces' surfaces and the body's mesh — an edge of any shape reads off it —
/// or, lost there, by its fingerprint alone (ScanRuler's `edgesOf`).
pub(crate) fn edge_reads<'m>(t: &Topology, mesh: &'m mm_reverse::curved_edge::FacedMesh, edges: &[EdgeRef], index: impl Fn(&EdgeRef) -> Option<usize>) -> Vec<mm_reverse::fillet_read::ReadEdge<'m>> {
    edges
        .iter()
        .map(|e| match index(e).and_then(|i| t.edges.get(i)) {
            Some(found) => mm_reverse::fillet_read::read_edge_of(found, t, Some(mesh)),
            None => mm_reverse::fillet_read::ReadEdge::of_fingerprint(e.clone()),
        })
        .collect()
}

/// A body's display mesh as a round is read off it: each triangle with the face it lies on.
pub(crate) fn faced_mesh(m: &mm_reverse::build::BodyMesh) -> mm_reverse::curved_edge::FacedMesh {
    mm_reverse::curved_edge::FacedMesh { positions: m.positions.clone(), normals: m.normals.clone(), triangles: m.triangles.clone(), face_hash: Some(m.tri_face.clone()) }
}

/// The open box as a line, for a test to read.
fn report(kind: &FeatureKind) -> String {
    with_draft(|d| match kind {
        FeatureKind::Fillet(x) => format!(
            "fillet body {} edges {} variable [{}] radius {} mode {} chain {} typed {} estimate {:?} skip {}",
            x.body,
            x.edges.len(),
            x.variable.iter().map(|v| format!("{}→{}", v.radius[0], v.radius[1])).collect::<Vec<_>>().join(", "),
            x.radius,
            if d.variable { "variable" } else { "constant" },
            x.chain,
            d.radius_typed,
            d.estimate,
            x.skip_failing
        ),
        FeatureKind::Chamfer(x) => format!("chamfer body {} edges {} distance {} mode {:?} distance2 {:?} angle {:?} flip {} chain {}", x.body, x.edges.len(), x.distance, x.mode, x.distance2, x.angle_deg, x.flip, x.chain),
        FeatureKind::Shell(x) => format!("shell body {} faces {} thickness {} from {:?}", x.body, x.faces.len(), x.thickness, d.thickness_from),
        FeatureKind::DeleteFace(x) => format!("deleteFace body {} faces {} tangent {}", x.body, x.faces.len(), d.tangent),
        _ => "another box".into(),
    })
}

#[cfg(test)]
mod tests;
