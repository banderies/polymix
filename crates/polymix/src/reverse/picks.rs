// SPDX-License-Identifier: AGPL-3.0-only
//! Object-first picking in the Reverse Engineer workspace's view, with no box open: a click on a
//! body takes a face, an edge or a corner (the corner beats the edges ending in it, an edge the
//! faces it bounds), Ctrl+click adds one, a Shift+drag window takes what lies in it, the arrow
//! keys step through what lies under the cursor; a click on a sketch's region or line, or on a
//! coordinate plane, selects that. What is selected is lit, and the chips by the click
//! (`chips.rs`) offer what it can become. The coordinate planes at the origin are drawn and
//! picked here too, as is the shared volume of bodies that run into each other.
//!
//! Ported from ScanRuler (`plugins/reverse/viewer/bodyPicks.ts`, `viewer/reScene.ts`,
//! `viewer/coordinatePlanes.ts`, `app/useReversePlugin.tsx`, `state/reStore.ts` selection),
//! © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.

use std::collections::{HashMap, HashSet};
use std::sync::mpsc::{channel, Receiver, TryRecvError};
use std::sync::Arc;

use egui::{Color32, Pos2, Stroke};
use glam::{DVec2, DVec3, Vec3, Vec4};
use mm_core::bvh::Bvh;
use mm_reverse::body_clearance::{clearance, clearance_text, overlaps, Overlap};
use mm_reverse::body_folders::{bodies_of_pick, ListPick};
use mm_reverse::body_style::{body_color, rgb_of};
use mm_reverse::display_units::DisplayUnits;
use mm_reverse::frame::{SectionFrame, WorldAxis};
use mm_reverse::model::{BodyId, ElementId, FeatureId, FeatureKind, PlaneRef};
use mm_reverse::regions::region_at;
use mm_reverse::selection_info::{item_key, picked_with, with_items, PickedItem};
use mm_reverse::sketch::{EntityId, Shape};
use mm_reverse::topo::split_edges;

use super::{BodyView, Mode, ReverseWorkspace};
use crate::renderer::ColorVertex;
use crate::tools::Viewport;

/// How far, in points, a click may miss an edge and still take it.
const EDGE_REACH_PX: f32 = 8.0;
/// And a corner: a smaller thing than an edge, it wins over the edges that end in it, so it
/// reaches a little further.
const VERTEX_REACH_PX: f32 = 10.0;
/// A lit corner's size on screen.
const VERTEX_LIT_PX: f32 = 11.0;
/// How far a sketch line may be missed.
const LINE_REACH_PX: f32 = 8.0;
/// How far the cursor may stray from where the arrow keys were pressed and still hold what they
/// stepped to — a hand on the mouse is not still.
const STEP_HOLD_PX: f32 = 4.0;
/// A Shift+press that moves no further than this is an additive click, not a window.
const WINDOW_MOVE_PX: f32 = 6.0;

pub(crate) const HOVER_COLOR: Color32 = Color32::WHITE;
pub(crate) const PICKED_COLOR: Color32 = Color32::from_rgb(0x3c, 0xb3, 0x71);
/// A whole body lit from its row in the body list.
pub(crate) const LIT_BODY_COLOR: Color32 = Color32::from_rgb(0x3a, 0x9b, 0xe8);
const LIT_BODY_OPACITY: f32 = 0.26;
/// The window and the crossing window over the bodies.
const WINDOW_COLOR: Color32 = Color32::from_rgb(0x3c, 0xb3, 0x71);
const CROSSING_COLOR: Color32 = Color32::from_rgb(0x4d, 0xb8, 0xff);
/// What two bodies share where they run into each other: something on the part to see to.
const OVERLAP_COLOR: Color32 = Color32::from_rgb(0xe0, 0x56, 0x4f);
const OVERLAP_HATCH_COLOR: Color32 = Color32::from_rgb(0x7a, 0x1c, 0x18);
/// The sketch colour regions are filled in.
const SKETCH_COLOR: Color32 = Color32::from_rgb(0x3d, 0x8f, 0xd6);

/// Half a coordinate plane's side, as a share of the part's size.
const PLANE_HALF: f64 = 0.5;
const SHEET_OPACITY: f32 = 0.06;
const SHEET_LIT_OPACITY: f32 = 0.2;
const BORDER_OPACITY: f32 = 0.45;
/// Half the diagonal of the blank stage a model is begun on with no scan: a part the size of a
/// hand.
const SCANLESS_RADIUS: f64 = 50.0;

/// The status line as a sketch opens on a plane from a chip.
pub(crate) const SKETCH_STARTED: &str = "Fit the slice chains, or draw the profile with the tools, then create the sketch.";

/// What a click on the part picked while no box is open: what the chips offer verbs for.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Selection {
    /// A region of a sketch shown on the part.
    Region { sketch: FeatureId, key: String },
    /// A line of a sketch shown on the part.
    SketchLine { sketch: FeatureId, entity: EntityId },
    /// A measured element (drawn and picked by the elements' layer once there is one).
    Element { id: ElementId, name: String },
    /// A coordinate plane at the origin, by the axis it stands on.
    CoordinatePlane { axis: WorldAxis },
    /// Faces, edges and corners of the bodies: one picked by a click, more with Ctrl or a window.
    Body(Vec<PickedItem>),
}

/// The selection and where the click was, for the chips to stand by.
#[derive(Debug, Clone)]
pub(crate) struct Selected {
    pub what: Selection,
    pub at: Pos2,
}

/// One of the things under the cursor a click could take.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Candidate {
    Vertex { body: BodyId, point: DVec3 },
    /// An edge by its index in the body's topology.
    Edge { body: BodyId, edge: usize },
    /// A face by its hash, and how far along the pick ray it was met (scene units).
    Face { body: BodyId, face: u32, distance: f64 },
}

impl Candidate {
    pub(crate) fn body(&self) -> BodyId {
        match self {
            Candidate::Vertex { body, .. } | Candidate::Edge { body, .. } | Candidate::Face { body, .. } => *body,
        }
    }
}

/// A step the arrow keys took through what lies under the cursor.
pub(crate) struct Stepped {
    at: Pos2,
    items: Vec<Candidate>,
    index: usize,
}

/// Where the bodies the pointer is on were lit from: their rows in the list, or the view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HoverFrom {
    List,
    View,
}

/// What a body's mesh offers picking, worked out once per mesh: the edges that are no edge to
/// pick (seams, and the cuts a boolean left across one surface), the corners, and each face's
/// triangles.
struct BodyPickInfo {
    hidden_edges: HashSet<u32>,
    corners: Vec<DVec3>,
    face_tris: HashMap<u32, Vec<u32>>,
}

/// A body as picking reads it: its mesh in scene coordinates and its topology, shared.
#[derive(Clone)]
struct Shown {
    id: BodyId,
    mesh: Arc<mm_core::Mesh>,
    topology: Arc<mm_reverse::topo::Topology>,
}

fn shown(b: &BodyView) -> Shown {
    Shown { id: b.id, mesh: b.mesh.clone(), topology: b.topology.clone() }
}

/// The bodies' shared volumes being read on a worker thread.
struct OverlapJob {
    key: u64,
    rx: Receiver<Vec<Overlap>>,
    stop: Arc<std::sync::atomic::AtomicBool>,
}

/// Everything the picking, the chips and the body list keep between frames.
pub(crate) struct PickState {
    pub selection: Option<Selected>,
    /// Rows picked in the body list.
    pub list_pick: ListPick,
    pub list_anchor: Option<super::bodies::RowKey>,
    /// The rows a right click opened the menu for.
    pub menu_pick: ListPick,
    /// Where the body list's rows stood last frame: a click on it off the rows lets them go.
    pub list_rect: Option<egui::Rect>,
    pub body_hover: Option<(Vec<BodyId>, HoverFrom)>,
    pub closed_folders: Vec<u32>,
    /// A component being named, and the text typed so far.
    pub naming_folder: Option<(u32, String)>,
    pub naming_body: Option<(BodyId, String)>,
    /// The body whose swatches are open under its row.
    pub painting: Option<BodyId>,
    pub hover: Option<Candidate>,
    pub hover_region: Option<(FeatureId, String)>,
    pub hover_line: Option<(FeatureId, EntityId)>,
    pub hover_plane: Option<WorldAxis>,
    stepped: Option<Stepped>,
    /// A Shift+drag window: where it started, where the pointer is.
    pub window: Option<(Pos2, Pos2)>,
    pointer: Option<Pos2>,
    /// The view as last drawn.
    pub vp: Option<Viewport>,
    /// The eyes of the coordinate planes (XY, XZ, YZ by the axis they stand on); `None` as set by
    /// default — shown on a blank stage, hidden over a scan.
    pub planes_shown: [Option<bool>; 3],
    /// The clearance being measured on a worker thread.
    clearance_job: Option<Receiver<Result<String, String>>>,
    /// What the bodies shown share, where they run into each other, and the hatch drawn over
    /// it (segments in part coordinates).
    pub overlaps: Vec<Overlap>,
    overlap_hatch: Vec<(DVec3, DVec3)>,
    overlaps_key: Option<u64>,
    overlap_job: Option<OverlapJob>,
    /// A Snap to scan asked for from a chip: the Move face box it opened is to lay its face on
    /// the scan where it is clicked next. Its editor takes this (`take_snap_request`).
    pub snap_request: bool,
    /// Bodies a script step lights as the pointer on their rows would.
    pub script_lit: Vec<BodyId>,
    /// When the build under way began, for the build chip to wait its 300 ms.
    build_since: Option<std::time::Instant>,
    info: HashMap<usize, Arc<BodyPickInfo>>,
    bvh: HashMap<usize, Arc<Bvh>>,
    closed: HashMap<usize, bool>,
}

impl Default for PickState {
    fn default() -> Self {
        Self {
            selection: None,
            list_pick: ListPick::default(),
            list_anchor: None,
            menu_pick: ListPick::default(),
            list_rect: None,
            body_hover: None,
            closed_folders: Vec::new(),
            naming_folder: None,
            naming_body: None,
            painting: None,
            hover: None,
            hover_region: None,
            hover_line: None,
            hover_plane: None,
            stepped: None,
            window: None,
            pointer: None,
            vp: None,
            planes_shown: [None; 3],
            clearance_job: None,
            overlaps: Vec::new(),
            overlap_hatch: Vec::new(),
            overlaps_key: None,
            overlap_job: None,
            snap_request: false,
            script_lit: Vec::new(),
            build_since: None,
            info: HashMap::new(),
            bvh: HashMap::new(),
            closed: HashMap::new(),
        }
    }
}

impl PickState {
    /// The bodies the selection is on, each once, in the order picked.
    pub(crate) fn picked_bodies(&self) -> Vec<BodyId> {
        let mut out = Vec::new();
        if let Some(Selected { what: Selection::Body(items), .. }) = &self.selection {
            for i in items {
                if !out.contains(&i.body()) {
                    out.push(i.body());
                }
            }
        }
        out
    }

    /// The faces, edges and corners selected.
    pub(crate) fn items(&self) -> &[PickedItem] {
        match &self.selection {
            Some(Selected { what: Selection::Body(items), .. }) => items,
            _ => &[],
        }
    }

    /// Whether a Move face box opened from Snap to scan is to snap; asked once.
    #[allow(dead_code)]
    pub(crate) fn take_snap_request(&mut self) -> bool {
        std::mem::take(&mut self.snap_request)
    }
}

fn axis_index(axis: WorldAxis) -> usize {
    match axis {
        WorldAxis::Z => 0,
        WorldAxis::Y => 1,
        WorldAxis::X => 2,
    }
}

/// The coordinate planes in the order the body list has them: XY, XZ, YZ.
pub(crate) const PLANE_AXES: [WorldAxis; 3] = [WorldAxis::Z, WorldAxis::Y, WorldAxis::X];

/// A plane's colour: the colour of the axis it stands on (XY the Z blue, XZ the Y green, YZ the
/// X red).
pub(crate) fn plane_color(axis: WorldAxis) -> Color32 {
    match axis {
        WorldAxis::Z => Color32::from_rgb(0x18, 0x77, 0xc0),
        WorldAxis::Y => Color32::from_rgb(0x2e, 0x7d, 0x46),
        WorldAxis::X => Color32::from_rgb(0xe5, 0x53, 0x4b),
    }
}

/// A colour for the renderer: straight RGBA in 0..1.
pub(crate) fn vec4(c: Color32, alpha: f32) -> Vec4 {
    Vec4::new(c.r() as f32 / 255.0, c.g() as f32 / 255.0, c.b() as f32 / 255.0, alpha)
}

fn seg_dist(p: Pos2, a: Pos2, b: Pos2) -> (f32, f32) {
    let ab = b - a;
    let t = if ab.length_sq() > 0.0 { ((p - a).dot(ab) / ab.length_sq()).clamp(0.0, 1.0) } else { 0.0 };
    ((a + ab * t - p).length(), t)
}

/// Whether a segment on screen meets a box.
fn segment_meets_box(a: Pos2, b: Pos2, r: egui::Rect) -> bool {
    if r.contains(a) || r.contains(b) {
        return true;
    }
    let corners = [r.left_top(), r.right_top(), r.right_bottom(), r.left_bottom()];
    let cross = |o: Pos2, p: Pos2, q: Pos2| (p.x - o.x) * (q.y - o.y) - (p.y - o.y) * (q.x - o.x);
    (0..4).any(|i| {
        let (c, d) = (corners[i], corners[(i + 1) % 4]);
        let d1 = cross(c, d, a);
        let d2 = cross(c, d, b);
        let d3 = cross(a, b, c);
        let d4 = cross(a, b, d);
        ((d1 > 0.0) != (d2 > 0.0)) && ((d3 > 0.0) != (d4 > 0.0))
    })
}

impl ReverseWorkspace {
    // ---- what the bodies offer ----

    /// A body's colour: its own, as given in the list, or the palette's by the feature that
    /// made it.
    pub(crate) fn body_color32(&self, id: BodyId) -> Color32 {
        let given = self.part.body_styles.get(&id).and_then(|s| s.color.clone());
        let hex = given.unwrap_or_else(|| body_color(id).to_string());
        rgb_of(&hex).map_or(Color32::GRAY, |[r, g, b]| Color32::from_rgb(r, g, b))
    }

    /// Whether a body is a sheet rather than a solid.
    pub(crate) fn body_is_surface(&self, id: BodyId) -> bool {
        self.built.body(id).is_some_and(|b| b.surface)
    }

    /// Whether a body's mesh closes on itself — every edge between two triangles once welded.
    pub(crate) fn body_is_closed(&mut self, id: BodyId) -> bool {
        let Some(b) = self.bodies.iter().find(|b| b.id == id) else { return false };
        let key = Arc::as_ptr(&b.part_mesh) as usize;
        if let Some(c) = self.picks.closed.get(&key) {
            return *c;
        }
        let (_, tris) = mm_reverse::build::welded(&b.part_mesh.positions, &b.part_mesh.triangles, 1e-6);
        let mut count: HashMap<(u32, u32), i32> = HashMap::new();
        for t in &tris {
            for k in 0..3 {
                let (a, c) = (t[k], t[(k + 1) % 3]);
                *count.entry((a.min(c), a.max(c))).or_insert(0) += 1;
            }
        }
        let closed = !tris.is_empty() && count.values().all(|n| *n == 2);
        self.picks.closed.insert(key, closed);
        closed
    }

    fn pick_info(&mut self, b: &Shown) -> Arc<BodyPickInfo> {
        let key = Arc::as_ptr(&b.mesh) as usize;
        if let Some(i) = self.picks.info.get(&key) {
            return i.clone();
        }
        let mut hidden_edges = split_edges(&b.topology);
        for e in &b.topology.edges {
            if e.faces.len() == 2 && e.faces[0] == e.faces[1] {
                hidden_edges.insert(e.hash);
            }
        }
        // The ends of the edges, each place once.
        let mut seen: HashSet<(i64, i64, i64)> = HashSet::new();
        let mut corners = Vec::new();
        for e in b.topology.edges.iter().filter(|e| !hidden_edges.contains(&e.hash)) {
            for p in e.ends {
                if seen.insert(((p.x * 1000.0).round() as i64, (p.y * 1000.0).round() as i64, (p.z * 1000.0).round() as i64)) {
                    corners.push(p);
                }
            }
        }
        let mut face_tris: HashMap<u32, Vec<u32>> = HashMap::new();
        for t in b.mesh.triangle_ids() {
            face_tris.entry(b.mesh.tri_group(t).max(0) as u32).or_default().push(t);
        }
        let info = Arc::new(BodyPickInfo { hidden_edges, corners, face_tris });
        // Forget what no body holds any more.
        let live: HashSet<usize> = self.bodies.iter().chain(self.pick_bodies.iter().flatten()).map(|b| Arc::as_ptr(&b.mesh) as usize).collect();
        self.picks.info.retain(|k, _| live.contains(k));
        self.picks.bvh.retain(|k, _| live.contains(k));
        self.picks.info.insert(key, info.clone());
        info
    }

    fn body_bvh_of(&mut self, b: &Shown) -> Arc<Bvh> {
        let key = Arc::as_ptr(&b.mesh) as usize;
        if let Some(bvh) = self.picks.bvh.get(&key) {
            return bvh.clone();
        }
        let bvh = Arc::new(Bvh::build(&b.mesh));
        self.picks.bvh.insert(key, bvh.clone());
        bvh
    }

    /// The bodies a click picks on, shown.
    fn playing(&self) -> Vec<Shown> {
        if !self.show_bodies {
            return Vec::new();
        }
        self.pick_set().iter().filter(|b| !self.hidden_bodies.contains(&b.id)).map(shown).collect()
    }

    /// A scene point on screen, if in front.
    pub(crate) fn project_scene(&self, vp: Viewport, p: DVec3) -> Option<Pos2> {
        let n = self.view.camera.project(p, vp.aspect())?;
        (0.0..=1.0).contains(&n.z).then(|| vp.to_screen(n))
    }

    fn ray_at(&self, vp: Viewport, p: Pos2) -> mm_core::Ray {
        self.view.camera.ray_from_ndc(vp.ndc(p), vp.aspect())
    }

    /// Half the size of what is shown: the scan and the bodies, or the blank stage.
    pub(crate) fn model_radius(&self) -> f64 {
        let mut b = mm_core::Aabb::EMPTY;
        for o in self.scan.iter().filter(|o| o.visible) {
            let ob = o.mesh.bounds();
            if !ob.is_empty() {
                b.include(ob.min);
                b.include(ob.max);
            }
        }
        for body in &self.bodies {
            let mb = body.mesh.bounds();
            if !mb.is_empty() {
                b.include(mb.min);
                b.include(mb.max);
            }
        }
        if b.is_empty() {
            SCANLESS_RADIUS
        } else {
            (b.diagonal() * 0.5).max(1e-3)
        }
    }

    /// How far along a ray a body's face may lie and still be picked past the scan: the scan's
    /// own depth, give or take its noise, where it is drawn solid. Drawn see-through it hides
    /// nothing.
    fn scan_wall(&mut self, ray: &mm_core::Ray) -> f64 {
        if !self.show_scan || self.scan_see_through {
            return f64::INFINITY;
        }
        let objects: Vec<super::ScanObject> = self.scan.iter().filter(|o| o.visible).cloned().collect();
        let mut best = f64::INFINITY;
        for o in &objects {
            if let Some(bvh) = self.scan_bvh(o) {
                if let Some(h) = bvh.raycast(&o.mesh, ray) {
                    best = best.min(h.t);
                }
            }
        }
        best + (self.model_radius() * 0.01).max(0.5)
    }

    /// The faces a ray meets on the bodies in play, the nearest first, each once.
    fn faces_along(&mut self, ray: &mm_core::Ray, all: bool) -> Vec<Candidate> {
        let mut hits: Vec<(f64, BodyId, u32)> = Vec::new();
        for b in self.playing() {
            let bvh = self.body_bvh_of(&b);
            if all {
                let mut out = Vec::new();
                bvh.raycast_all(&b.mesh, ray, &mut out);
                hits.extend(out.iter().map(|h| (h.t, b.id, b.mesh.tri_group(h.tid).max(0) as u32)));
            } else if let Some(h) = bvh.raycast(&b.mesh, ray) {
                hits.push((h.t, b.id, b.mesh.tri_group(h.tid).max(0) as u32));
            }
        }
        hits.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for (t, body, face) in hits {
            if seen.insert((body, face)) {
                out.push(Candidate::Face { body, face, distance: t });
                if !all {
                    break;
                }
            }
        }
        out
    }

    /// Whether the bodies are drawn solid — then what lies behind one is hidden and not picked.
    fn bodies_solid(&self) -> bool {
        !self.deviation.on
    }

    /// Whether a part point on a body lies behind a body as the camera sees it.
    fn hidden_point(&mut self, vp: Viewport, at: DVec3) -> bool {
        let scene = self.to_scene(at);
        let Some(s) = self.project_scene(vp, scene) else { return true };
        let ray = self.ray_at(vp, s);
        let reach = (scene - ray.origin).length();
        let mut first = f64::INFINITY;
        for b in self.bodies.iter().filter(|b| !self.hidden_bodies.contains(&b.id)).map(shown).collect::<Vec<_>>() {
            let bvh = self.body_bvh_of(&b);
            if let Some(h) = bvh.raycast(&b.mesh, &ray) {
                first = first.min(h.t);
            }
        }
        first < reach - (reach * 2e-3).max(0.3)
    }

    /// Every edge within reach of a screen point, with the point on it nearest the cursor — the
    /// nearest first.
    fn edges_near(&mut self, vp: Viewport, p: Pos2) -> Vec<(Candidate, f32, DVec3)> {
        let mut near = Vec::new();
        for b in self.playing() {
            let info = self.pick_info(&b);
            for (i, e) in b.topology.edges.iter().enumerate() {
                if info.hidden_edges.contains(&e.hash) || e.polyline.len() < 2 {
                    continue;
                }
                let mut best = EDGE_REACH_PX;
                let mut at = None;
                let pts: Vec<Option<Pos2>> = e.polyline.iter().map(|q| self.project(vp, *q)).collect();
                for k in 1..pts.len() {
                    let (Some(a), Some(c)) = (pts[k - 1], pts[k]) else { continue };
                    let (d, t) = seg_dist(p, a, c);
                    if d < best {
                        best = d;
                        at = Some(e.polyline[k - 1].lerp(e.polyline[k], t as f64));
                    }
                }
                if let Some(at) = at {
                    near.push((Candidate::Edge { body: b.id, edge: i }, best, at));
                }
            }
        }
        near.sort_by(|a, b| a.1.total_cmp(&b.1));
        near
    }

    /// Every corner within reach of a screen point — the nearest first.
    fn corners_near(&mut self, vp: Viewport, p: Pos2) -> Vec<(Candidate, f32, DVec3)> {
        let mut near = Vec::new();
        for b in self.playing() {
            let info = self.pick_info(&b);
            for c in &info.corners {
                if let Some(s) = self.project(vp, *c) {
                    let d = (s - p).length();
                    if d <= VERTEX_REACH_PX {
                        near.push((Candidate::Vertex { body: b.id, point: *c }, d, *c));
                    }
                }
            }
        }
        near.sort_by(|a, b| a.1.total_cmp(&b.1));
        near
    }

    /// The nearest few within reach, those in sight first — or, the bodies solid, those in
    /// sight alone.
    fn seen_first(&mut self, vp: Viewport, near: Vec<(Candidate, f32, DVec3)>, keep_hidden: bool) -> Vec<Candidate> {
        let few: Vec<(Candidate, f32, DVec3)> = near.into_iter().take(8).collect();
        if !self.bodies_solid() {
            return few.into_iter().map(|n| n.0).collect();
        }
        let hidden: Vec<bool> = few.iter().map(|n| self.hidden_point(vp, n.2)).collect();
        let mut out: Vec<Candidate> = few.iter().zip(&hidden).filter(|(_, h)| !**h).map(|(n, _)| n.0.clone()).collect();
        if keep_hidden {
            out.extend(few.iter().zip(&hidden).filter(|(_, h)| **h).map(|(n, _)| n.0.clone()));
        }
        out
    }

    /// What of the bodies a click at a screen point takes: a corner — the smallest thing there
    /// is — an edge, or a face at least as near as the scan; `None` for none.
    pub(crate) fn body_item_at(&mut self, vp: Viewport, p: Pos2) -> Option<Candidate> {
        let corners = self.corners_near(vp, p);
        if let Some(c) = self.seen_first(vp, corners, false).into_iter().next() {
            return Some(c);
        }
        let edges = self.edges_near(vp, p);
        if let Some(e) = self.seen_first(vp, edges, false).into_iter().next() {
            return Some(e);
        }
        let ray = self.ray_at(vp, p);
        let face = self.faces_along(&ray, false).into_iter().next()?;
        match face {
            Candidate::Face { distance, .. } if distance <= self.scan_wall(&ray) => Some(face),
            _ => None,
        }
    }

    /// Everything under a screen point, in the order the arrow keys step through it: the
    /// corners within reach, then the edges, then the faces the ray meets — those in sight
    /// before those behind a body, the faces from the front back.
    fn candidates_at(&mut self, vp: Viewport, p: Pos2) -> Vec<Candidate> {
        let corners = self.corners_near(vp, p);
        let mut out = self.seen_first(vp, corners, true);
        let edges = self.edges_near(vp, p);
        out.extend(self.seen_first(vp, edges, true));
        let ray = self.ray_at(vp, p);
        out.extend(self.faces_along(&ray, true));
        out
    }

    /// A candidate as a pick: a face with the edges bounding it, an edge, a corner.
    pub(crate) fn picked_item(&self, c: &Candidate) -> Option<PickedItem> {
        let b = self.pick_view(c.body())?;
        match c {
            Candidate::Vertex { body, point } => Some(PickedItem::Vertex { body: *body, point: *point }),
            Candidate::Edge { body, edge } => b.topology.edges.get(*edge).map(|e| PickedItem::Edge { body: *body, edge: e.clone() }),
            Candidate::Face { body, face, .. } => {
                let f = b.topology.face(*face)?;
                let edges = b.topology.edges.iter().filter(|e| e.faces.contains(&f.hash)).cloned().collect();
                Some(PickedItem::Face { body: *body, face: f.clone(), edges })
            }
        }
    }

    // ---- sketches and planes under the cursor ----

    /// The region of a visible sketch under a screen point — one the scan does not stand in
    /// front of, so a click on the part's face is not one on a sketch lying inside it.
    fn region_at_point(&mut self, vp: Viewport, p: Pos2) -> Option<(FeatureId, String)> {
        let ray = self.ray_at(vp, p);
        let wall = self.scan_wall(&ray);
        for f in self.part.features.iter().rev().filter(|f| f.visible() && !f.suppressed()) {
            let FeatureKind::Sketch(_) = &f.kind else { continue };
            let Some(frame) = self.sketch_frame(f.id) else { continue };
            let Some(sheet) = self.sheet_point(vp, p, &frame) else { continue };
            let regions = self.built.sketches.get(&f.id).map(|b| b.regions.as_slice()).unwrap_or(&[]);
            let Some(r) = region_at(regions, sheet) else { continue };
            let at = self.to_scene(frame.to_world(sheet));
            if (at - ray.origin).length() <= wall {
                return Some((f.id, r.key.clone()));
            }
        }
        None
    }

    /// The line of a visible sketch under a screen point, within a few points on screen.
    fn sketch_line_at(&self, vp: Viewport, p: Pos2) -> Option<(FeatureId, EntityId)> {
        let mut best: Option<((FeatureId, EntityId), f32)> = None;
        for f in self.part.features.iter().filter(|f| f.visible() && !f.suppressed()) {
            let FeatureKind::Sketch(s) = &f.kind else { continue };
            let frame = self.sketch_frame(f.id).unwrap_or(s.frame);
            for e in &s.sketch.entities {
                let Shape::Line { a, b } = e.shape else { continue };
                let (Some(sa), Some(sb)) = (self.project(vp, frame.to_world(a)), self.project(vp, frame.to_world(b))) else { continue };
                let (d, _) = seg_dist(p, sa, sb);
                if d < LINE_REACH_PX && best.is_none_or(|(_, bd)| d < bd) {
                    best = Some(((f.id, e.id), d));
                }
            }
        }
        best.map(|b| b.0)
    }

    /// Whether a coordinate plane is drawn: as its eye was set, else shown on a blank stage and
    /// hidden over a scan.
    pub(crate) fn plane_shown(&self, axis: WorldAxis) -> bool {
        self.picks.planes_shown[axis_index(axis)].unwrap_or(self.scan.is_empty())
    }

    pub(crate) fn set_plane_shown(&mut self, axis: WorldAxis, shown: bool) {
        self.picks.planes_shown[axis_index(axis)] = Some(shown);
        if !shown && matches!(&self.picks.selection, Some(Selected { what: Selection::CoordinatePlane { axis: a }, .. }) if *a == axis) {
            self.picks.selection = None;
        }
    }

    /// A coordinate plane's frame and the half of its side.
    fn plane_square(&self, axis: WorldAxis) -> Option<(SectionFrame, f64)> {
        let frame = PlaneRef::world(axis, 0.0).own_frame()?;
        Some((frame, self.model_radius() * PLANE_HALF))
    }

    /// The coordinate plane under a screen point — only where nothing else takes the click and
    /// the scan, drawn solid, is not in front.
    fn plane_at(&mut self, vp: Viewport, p: Pos2) -> Option<WorldAxis> {
        let ray = self.ray_at(vp, p);
        let wall = self.scan_wall(&ray);
        let mut best: Option<(WorldAxis, f64)> = None;
        for axis in PLANE_AXES {
            if !self.plane_shown(axis) {
                continue;
            }
            let Some((frame, half)) = self.plane_square(axis) else { continue };
            let Some(sheet) = self.sheet_point(vp, p, &frame) else { continue };
            if sheet.x.abs() > half || sheet.y.abs() > half {
                continue;
            }
            let at = self.to_scene(frame.to_world(sheet));
            let t = (at - ray.origin).length();
            if (at - ray.origin).dot(ray.dir) > 0.0 && t <= wall && best.is_none_or(|(_, bt)| t < bt) {
                best = Some((axis, t));
            }
        }
        best.map(|b| b.0)
    }

    // ---- the pointer with no box open ----

    /// The pointer over the view with no box open: what is under it lit, a click taking it, a
    /// Shift+drag a window of it.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn idle_pointer(&mut self, vp: Viewport, pos: Option<Pos2>, pressed: bool, down: bool, released: bool, clicked: bool, mods: egui::Modifiers) {
        self.picks.vp = Some(vp);
        self.picks.pointer = pos;
        self.picks.hover = None;
        self.picks.hover_region = None;
        self.picks.hover_line = None;
        self.picks.hover_plane = None;
        // The arrow keys' step is let go once the cursor moves off.
        if let (Some(s), Some(p)) = (&self.picks.stepped, pos) {
            if (s.at - p).length() > STEP_HOLD_PX {
                self.picks.stepped = None;
            }
        }
        // Shift and a left drag over the bodies is a selection window.
        if pressed && mods.shift && !mods.command && !mods.alt {
            if let Some(p) = pos {
                self.picks.window = Some((p, p));
            }
        }
        if let Some((a, _)) = self.picks.window {
            if let Some(p) = pos {
                self.picks.window = Some((a, p));
            }
            if released || !down {
                let (a, b) = self.picks.window.take().unwrap_or((a, a));
                if (b.x - a.x).abs() + (b.y - a.y).abs() > WINDOW_MOVE_PX {
                    self.window_pick(vp, a, b);
                    return;
                }
                // A press that did not move is the click it was — an additive one.
            } else {
                self.set_view_hover(None);
                return;
            }
        }
        let Some(p) = pos else {
            self.set_view_hover(None);
            return;
        };
        // The sketches stand in front: a region, else a line.
        let region = self.region_at_point(vp, p);
        let line = if region.is_none() { self.sketch_line_at(vp, p) } else { None };
        // Then the bodies: what the arrow keys stepped to, else what a click would take.
        let stepped = self.picks.stepped.as_ref().map(|s| s.items[s.index].clone());
        let item = if region.is_some() || line.is_some() { None } else { stepped.clone().or_else(|| self.body_item_at(vp, p)) };
        // Last, a coordinate plane where nothing else would take the click.
        let plane = if region.is_none() && line.is_none() && item.is_none() { self.plane_at(vp, p) } else { None };
        self.set_view_hover(item.as_ref().map(|c| c.body()));
        self.picks.hover = item.clone();
        self.picks.hover_region = region.clone();
        self.picks.hover_line = line;
        self.picks.hover_plane = plane;
        if !clicked {
            return;
        }
        let additive = mods.command || mods.shift;
        self.picks.stepped = None;
        if let Some((sketch, key)) = region {
            self.picks.selection = Some(Selected { what: Selection::Region { sketch, key }, at: p });
        } else if let Some((sketch, entity)) = line {
            self.picks.selection = Some(Selected { what: Selection::SketchLine { sketch, entity }, at: p });
        } else if let Some(c) = item {
            if let Some(i) = self.picked_item(&c) {
                self.pick_body_item(i, p, additive);
            }
        } else if let Some(axis) = plane {
            self.picks.selection = Some(Selected { what: Selection::CoordinatePlane { axis }, at: p });
        } else if !mods.command {
            // A click on the scan or on nothing puts the selection away; a turn, a pan or a
            // zoom of the camera keeps it, as CAD does.
            self.picks.selection = None;
            self.picks.list_pick = ListPick::default();
        }
    }

    /// The body the pointer has come onto in the view lights its row; leaving it lets go only of
    /// what the view lit.
    fn set_view_hover(&mut self, body: Option<BodyId>) {
        match body {
            Some(id) => self.picks.body_hover = Some((vec![id], HoverFrom::View)),
            None => {
                if matches!(self.picks.body_hover, Some((_, HoverFrom::View))) {
                    self.picks.body_hover = None;
                }
            }
        }
    }

    /// A face, an edge or a corner picked: a plain click is a new selection — the rows picked in
    /// the list let go with it — Ctrl adds one, or takes it off again.
    pub(crate) fn pick_body_item(&mut self, item: PickedItem, at: Pos2, additive: bool) {
        if !additive {
            self.picks.list_pick = ListPick::default();
        }
        let next = picked_with(self.picks.items(), item, additive);
        self.picks.selection = (!next.is_empty()).then_some(Selected { what: Selection::Body(next), at });
    }

    /// A window over the bodies: dragged to the right, the faces and edges wholly inside it;
    /// dragged to the left, those it touches too — what lies behind the body with the rest.
    fn window_pick(&mut self, vp: Viewport, a: Pos2, b: Pos2) {
        let crossing = b.x < a.x;
        let r = egui::Rect::from_two_pos(a, b);
        let mut items = Vec::new();
        for body in self.playing() {
            let info = self.pick_info(&body);
            for e in body.topology.edges.iter().filter(|e| !info.hidden_edges.contains(&e.hash) && e.polyline.len() >= 2) {
                let pts: Vec<Pos2> = e.polyline.iter().filter_map(|q| self.project(vp, *q)).collect();
                if pts.len() < 2 {
                    continue;
                }
                let take = pts.iter().all(|p| r.contains(*p)) || (crossing && (pts.iter().any(|p| r.contains(*p)) || pts.windows(2).any(|w| segment_meets_box(w[0], w[1], r))));
                if take {
                    items.push(PickedItem::Edge { body: body.id, edge: e.clone() });
                }
            }
            for f in &body.topology.faces {
                let Some(tris) = info.face_tris.get(&f.hash) else { continue };
                let (mut all, mut any) = (true, false);
                'tris: for t in tris {
                    for v in body.mesh.triangle(*t) {
                        let inside = self.project_scene(vp, body.mesh.positions()[v as usize]).is_some_and(|s| r.contains(s));
                        any |= inside;
                        all &= inside;
                        if (crossing && any) || (!crossing && !all) {
                            break 'tris;
                        }
                    }
                }
                if (crossing && any) || (!crossing && all) {
                    let edges = body.topology.edges.iter().filter(|e| e.faces.contains(&f.hash)).cloned().collect();
                    items.push(PickedItem::Face { body: body.id, face: f.clone(), edges });
                }
            }
        }
        let next = with_items(self.picks.items(), items);
        self.picks.selection = (!next.is_empty()).then_some(Selected { what: Selection::Body(next), at: b });
    }

    /// A measured element selected — by the elements' layer once they are drawn and picked, or a
    /// script step.
    pub(crate) fn select_element(&mut self, id: ElementId, at: Pos2) {
        if let Some(e) = self.part.element(id) {
            self.picks.selection = Some(Selected { what: Selection::Element { id, name: e.name.clone() }, at });
        }
    }

    // ---- keys with no box open ----

    /// Escape lets the selection and the rows picked go; the arrow keys step through what lies
    /// under the cursor — corners, then edges, then faces front to back (down or right the next,
    /// up or left the one before).
    pub(crate) fn idle_keys(&mut self, ctx: &egui::Context) {
        let (esc, next, prev) = ctx.input(|i| {
            let plain = !i.modifiers.any();
            (i.key_pressed(egui::Key::Escape), plain && (i.key_pressed(egui::Key::ArrowDown) || i.key_pressed(egui::Key::ArrowRight)), plain && (i.key_pressed(egui::Key::ArrowUp) || i.key_pressed(egui::Key::ArrowLeft)))
        });
        if esc {
            self.picks.selection = None;
            self.picks.list_pick = ListPick::default();
            self.picks.window = None;
        }
        if next || prev {
            self.step_candidates(if next { 1 } else { -1 });
        }
    }

    fn step_candidates(&mut self, step: i32) {
        let (Some(vp), Some(at)) = (self.picks.vp, self.picks.pointer) else { return };
        if let Some(s) = &mut self.picks.stepped {
            if (s.at - at).length() <= STEP_HOLD_PX {
                let n = s.items.len() as i32;
                s.index = ((s.index as i32 + step + n) % n) as usize;
                return;
            }
        }
        let items = self.candidates_at(vp, at);
        let lit = self.picks.hover.as_ref().and_then(|h| items.iter().position(|c| same_candidate(c, h)));
        if items.is_empty() || (items.len() == 1 && lit == Some(0)) {
            return;
        }
        let n = items.len() as i32;
        let index = match lit {
            None => {
                if step > 0 {
                    0
                } else {
                    items.len() - 1
                }
            }
            Some(l) => ((l as i32 + step + n) % n) as usize,
        };
        self.picks.stepped = Some(Stepped { at, items, index });
    }

    // ---- the bodies not in play ----

    /// The bodies dimmed while a box asks for faces or edges of others: a Shell, Hole, Mirror,
    /// Draft, Delete face, Fillet or Chamfer works on its body, a Move face on the body of its
    /// first face, a Move on the bodies it does not move (they stand where the ghost goes) — the
    /// rest are faint, and a click cannot land on one.
    pub(crate) fn dimmed_bodies(&self) -> Vec<BodyId> {
        let Mode::Feature(edit) = &self.mode else { return Vec::new() };
        let in_play: Option<Vec<BodyId>> = match &edit.kind {
            FeatureKind::Shell(x) => Some(vec![x.body]),
            FeatureKind::Hole(x) => Some(vec![x.body]),
            FeatureKind::Mirror(x) => Some(vec![x.body]),
            FeatureKind::Draft(x) => Some(vec![x.body]),
            FeatureKind::DeleteFace(x) => Some(vec![x.body]),
            FeatureKind::Fillet(x) => Some(vec![x.body]),
            FeatureKind::Chamfer(x) => Some(vec![x.body]),
            FeatureKind::MoveFace(x) => (!x.faces.is_empty()).then(|| vec![x.body]),
            FeatureKind::MoveBody(x) => (!x.copy).then(|| self.pick_set().iter().map(|b| b.id).filter(|id| !x.bodies.contains(id)).collect()),
            _ => None,
        };
        match in_play {
            Some(ids) => self.pick_set().iter().map(|b| b.id).chain(self.bodies.iter().map(|b| b.id)).filter(|id| !ids.contains(id)).fold(Vec::new(), |mut v, id| {
                if !v.contains(&id) {
                    v.push(id);
                }
                v
            }),
            None => Vec::new(),
        }
    }

    // ---- the eyes ----

    /// Show or hide bodies; the map is read again for what is shown.
    pub(crate) fn set_bodies_visible(&mut self, ids: &[BodyId], visible: bool) {
        for id in ids {
            if visible {
                self.hidden_bodies.remove(id);
            } else {
                self.hidden_bodies.insert(*id);
            }
        }
        self.deviation.key = None;
    }

    /// Whether every body but these is hidden — and there is another.
    pub(crate) fn others_hidden(&self, ids: &[BodyId]) -> bool {
        let others: Vec<&BodyView> = self.bodies.iter().filter(|b| !ids.contains(&b.id)).collect();
        !others.is_empty() && others.iter().all(|b| self.hidden_bodies.contains(&b.id))
    }

    /// These bodies shown and every other hidden.
    pub(crate) fn isolate_bodies(&mut self, ids: &[BodyId]) {
        let others: Vec<BodyId> = self.bodies.iter().map(|b| b.id).filter(|id| !ids.contains(id)).collect();
        self.set_bodies_visible(&others, false);
        self.set_bodies_visible(ids, true);
    }

    /// Every body shown, or every body hidden.
    pub(crate) fn set_all_bodies_visible(&mut self, visible: bool) {
        if visible {
            self.hidden_bodies.clear();
        } else {
            let all: Vec<BodyId> = self.bodies.iter().map(|b| b.id).collect();
            self.set_bodies_visible(&all, false);
        }
        self.deviation.key = None;
    }

    /// The bodies lit in the view from the list: those under the pointer in it, and the rows
    /// picked.
    fn lit_bodies(&self) -> Vec<BodyId> {
        let mut lit: Vec<BodyId> = match &self.picks.body_hover {
            Some((ids, HoverFrom::List)) => ids.clone(),
            _ => Vec::new(),
        };
        for b in bodies_of_pick(&self.part.body_folders, &self.picks.list_pick).into_iter().chain(self.picks.script_lit.iter().copied()) {
            if !lit.contains(&b) {
                lit.push(b);
            }
        }
        lit
    }

    // ---- the clearance and the overlaps ----

    /// How far apart two bodies stand at the closest — or how much of them runs into each other —
    /// measured on a worker thread, the answer to the status line.
    pub(crate) fn measure_clearance(&mut self, a: BodyId, b: BodyId) {
        let (Some(ba), Some(bb)) = (self.built.body(a).cloned(), self.built.body(b).cloned()) else { return };
        let (na, nb) = (self.body_name_of(a), self.body_name_of(b));
        let (tx, rx) = channel();
        self.picks.clearance_job = Some(rx);
        self.status = format!("Measuring how {na} and {nb} stand…");
        std::thread::Builder::new()
            .name("reverse clearance".into())
            .spawn(move || {
                let r = clearance(&ba.body, &bb.body).map(|c| clearance_text(&na, &nb, &c, DisplayUnits::Mm)).map_err(|e| format!("The clearance could not be measured: {e}"));
                let _ = tx.send(r);
            })
            .ok();
    }

    pub(crate) fn body_name_of(&self, id: BodyId) -> String {
        self.bodies.iter().find(|b| b.id == id).map_or_else(|| id.to_string(), |b| b.name.clone())
    }

    /// Once a frame: the clearance's answer, and the overlaps read again for the bodies shown.
    pub(crate) fn tick_picks(&mut self, ctx: &egui::Context) {
        self.picks.build_since = match (self.building(), self.picks.build_since) {
            (true, None) => Some(std::time::Instant::now()),
            (true, since) => since,
            (false, _) => None,
        };
        // A box or a sketch open takes the selection and the rows picked away, as it opens.
        if !matches!(self.mode, Mode::Idle) {
            self.picks.selection = None;
            self.picks.list_pick = ListPick::default();
            self.picks.hover = None;
            self.picks.hover_region = None;
            self.picks.hover_line = None;
            self.picks.hover_plane = None;
            self.picks.stepped = None;
            self.picks.window = None;
            if matches!(self.picks.body_hover, Some((_, HoverFrom::View))) {
                self.picks.body_hover = None;
            }
        }
        if let Some(rx) = &self.picks.clearance_job {
            match rx.try_recv() {
                Ok(r) => {
                    self.picks.clearance_job = None;
                    self.status = r.unwrap_or_else(|e| e);
                    ctx.request_repaint();
                }
                Err(TryRecvError::Empty) => ctx.request_repaint_after(std::time::Duration::from_millis(50)),
                Err(TryRecvError::Disconnected) => self.picks.clearance_job = None,
            }
        }
        // The solids shown, closed, of the last build.
        let solids: Vec<BodyId> = if self.part.overlaps_shown && self.show_bodies {
            let ids: Vec<BodyId> = self.bodies.iter().map(|b| b.id).filter(|id| !self.hidden_bodies.contains(id) && !self.body_is_surface(*id)).collect();
            ids.into_iter().filter(|id| self.body_is_closed(*id)).collect()
        } else {
            Vec::new()
        };
        let key = solids.iter().fold(self.built_generation.wrapping_mul(1_000_003) ^ 0x9e37, |h, id| h.wrapping_mul(31).wrapping_add(id.as_number().to_bits()));
        if solids.len() < 2 {
            if let Some(job) = self.picks.overlap_job.take() {
                job.stop.store(true, std::sync::atomic::Ordering::Relaxed);
            }
            self.picks.overlaps.clear();
            self.picks.overlap_hatch.clear();
            self.picks.overlaps_key = None;
            return;
        }
        if let Some(job) = &self.picks.overlap_job {
            if job.key != key {
                job.stop.store(true, std::sync::atomic::Ordering::Relaxed);
                self.picks.overlap_job = None;
            } else {
                match job.rx.try_recv() {
                    Ok(found) => {
                        self.picks.overlap_hatch = hatch(&found, self.model_radius());
                        self.picks.overlaps = found;
                        self.picks.overlaps_key = Some(key);
                        self.picks.overlap_job = None;
                        ctx.request_repaint();
                    }
                    Err(TryRecvError::Empty) => ctx.request_repaint_after(std::time::Duration::from_millis(100)),
                    Err(TryRecvError::Disconnected) => self.picks.overlap_job = None,
                }
                return;
            }
        }
        if self.picks.overlaps_key == Some(key) || self.building() {
            return;
        }
        let bodies: Vec<(BodyId, mm_reverse::build::KernelBody)> = solids.iter().filter_map(|id| self.built.body(*id).map(|b| (*id, b.body.clone()))).collect();
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let (tx, rx) = channel();
        let flag = stop.clone();
        std::thread::Builder::new()
            .name("reverse overlaps".into())
            .spawn(move || {
                let refs: Vec<(BodyId, &mm_reverse::build::KernelBody)> = bodies.iter().map(|(id, b)| (*id, b)).collect();
                let found = overlaps(&refs, &|| flag.load(std::sync::atomic::Ordering::Relaxed));
                let _ = tx.send(found);
            })
            .ok();
        self.picks.overlap_job = Some(OverlapJob { key, rx, stop });
    }

    /// Whether the overlaps are still being read.
    pub(crate) fn picks_busy(&self) -> bool {
        self.picks.clearance_job.is_some() || self.picks.overlap_job.is_some()
    }

    // ---- what is drawn ----

    /// The triangles the renderer draws over the part, depth-tested: the faces picked, the
    /// coordinate planes' sheets, and what bodies share where they run into each other.
    pub(crate) fn pick_overlay_tris(&mut self) -> Vec<ColorVertex> {
        let mut out = Vec::new();
        let eye = self.view.camera.eye();
        let nudge = |p: DVec3| -> Vec3 { (p + (eye - p) * 2e-4).as_vec3() };
        if matches!(self.mode, Mode::Idle) && self.show_bodies {
            let picked: Vec<(BodyId, u32)> = self.picks.items().iter().filter_map(|i| if let PickedItem::Face { body, face, .. } = i { Some((*body, face.hash)) } else { None }).collect();
            for (body, face) in picked {
                let Some(b) = self.pick_view(body).filter(|b| !self.hidden_bodies.contains(&b.id)).map(shown) else { continue };
                let info = self.pick_info(&b);
                for t in info.face_tris.get(&face).into_iter().flatten() {
                    for v in b.mesh.triangle(*t) {
                        out.push(ColorVertex::new(nudge(b.mesh.positions()[v as usize]), vec4(PICKED_COLOR, 0.5)));
                    }
                }
            }
        }
        if !matches!(self.mode, Mode::Sketch(_)) {
            for axis in PLANE_AXES {
                if !self.plane_shown(axis) {
                    continue;
                }
                let Some((f, half)) = self.plane_square(axis) else { continue };
                let lit = self.picks.hover_plane == Some(axis) || matches!(&self.picks.selection, Some(Selected { what: Selection::CoordinatePlane { axis: a }, .. }) if *a == axis);
                let c = vec4(plane_color(axis), if lit { SHEET_LIT_OPACITY } else { SHEET_OPACITY });
                let q = |x: f64, y: f64| ColorVertex::new(self.to_scene(f.to_world(DVec2::new(x * half, y * half))).as_vec3(), c);
                out.extend([q(-1.0, -1.0), q(1.0, -1.0), q(1.0, 1.0), q(-1.0, -1.0), q(1.0, 1.0), q(-1.0, 1.0)]);
            }
        }
        if self.part.overlaps_shown && self.show_bodies {
            for o in &self.picks.overlaps {
                for t in &o.mesh.triangles {
                    for v in t {
                        out.push(ColorVertex::new(nudge(self.to_scene(o.mesh.positions[*v as usize])), vec4(OVERLAP_COLOR, 0.6)));
                    }
                }
            }
        }
        out
    }

    /// The lines the renderer draws, depth-tested: the coordinate planes' borders, the lit
    /// bodies' edges in the list's blue, and the outlines of what bodies share.
    pub(crate) fn pick_lines(&self) -> Vec<ColorVertex> {
        let mut out = Vec::new();
        let eye = self.view.camera.eye();
        if !matches!(self.mode, Mode::Sketch(_)) {
            for axis in PLANE_AXES {
                if !self.plane_shown(axis) {
                    continue;
                }
                let Some((f, half)) = self.plane_square(axis) else { continue };
                let lit = self.picks.hover_plane == Some(axis) || matches!(&self.picks.selection, Some(Selected { what: Selection::CoordinatePlane { axis: a }, .. }) if *a == axis);
                let c = vec4(plane_color(axis), if lit { 1.0 } else { BORDER_OPACITY });
                let corner = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];
                for i in 0..4 {
                    for (x, y) in [corner[i], corner[(i + 1) % 4]] {
                        out.push(ColorVertex::new(self.to_scene(f.to_world(DVec2::new(x * half, y * half))).as_vec3(), c));
                    }
                }
            }
        }
        for id in self.lit_bodies() {
            let Some(b) = self.bodies.iter().find(|b| b.id == id) else { continue };
            for e in &b.edges {
                for w in e.windows(2) {
                    for p in [w[0], w[1]] {
                        out.push(ColorVertex::new((p + (eye - p) * 3e-4).as_vec3(), vec4(LIT_BODY_COLOR, 1.0)));
                    }
                }
            }
        }
        if self.part.overlaps_shown && self.show_bodies {
            for o in &self.picks.overlaps {
                for e in &o.mesh.edges {
                    for w in e.windows(2) {
                        for p in [w[0], w[1]] {
                            let s = self.to_scene(p);
                            out.push(ColorVertex::new((s + (eye - s) * 3e-4).as_vec3(), vec4(OVERLAP_COLOR, 1.0)));
                        }
                    }
                }
            }
            // The hatch: darker strokes across the red.
            for (a, b) in &self.picks.overlap_hatch {
                for p in [*a, *b] {
                    let s = self.to_scene(p);
                    out.push(ColorVertex::new((s + (eye - s) * 3e-4).as_vec3(), vec4(OVERLAP_HATCH_COLOR, 1.0)));
                }
            }
        }
        out
    }

    /// Triangles of a scene mesh filled on the painter, over everything.
    fn fill_scene_tris(&self, painter: &egui::Painter, vp: Viewport, mesh: &mm_core::Mesh, tris: &[u32], color: Color32) {
        let mut m = egui::epaint::Mesh::default();
        for t in tris {
            let [a, b, c] = mesh.triangle(*t);
            let p = [a, b, c].map(|v| self.project_scene(vp, mesh.positions()[v as usize]));
            if let [Some(a), Some(b), Some(c)] = p {
                let i = m.vertices.len() as u32;
                for q in [a, b, c] {
                    m.colored_vertex(q, color);
                }
                m.add_triangle(i, i + 1, i + 2);
            }
        }
        if !m.is_empty() {
            painter.add(egui::Shape::mesh(m));
        }
    }

    /// A sketch region filled on its plane, over everything.
    fn fill_region(&self, painter: &egui::Painter, vp: Viewport, sketch: FeatureId, key: &str, color: Color32) {
        let Some(frame) = self.sketch_frame(sketch) else { return };
        let Some(r) = self.built.sketches.get(&sketch).and_then(|b| b.regions.iter().find(|r| r.key == key)) else { return };
        let mut m = egui::epaint::Mesh::default();
        for t in r.region.triangulate(0.05) {
            let p = t.map(|q| self.project(vp, frame.to_world(DVec2::new(q.x, q.y))));
            if let [Some(a), Some(b), Some(c)] = p {
                let i = m.vertices.len() as u32;
                for q in [a, b, c] {
                    m.colored_vertex(q, color);
                }
                m.add_triangle(i, i + 1, i + 2);
            }
        }
        if !m.is_empty() {
            painter.add(egui::Shape::mesh(m));
        }
    }

    /// What is lit over the part with no box open — what a click would take, what is selected,
    /// the regions on offer, the window being dragged — and the bodies lit from the list.
    pub(crate) fn paint_picks(&mut self, painter: &egui::Painter, vp: Viewport) {
        self.paint_stage_chips(painter, vp);
        // The bodies the pointer is on in the list: drawn over everything in the list's blue.
        for id in self.lit_bodies() {
            let Some(b) = self.bodies.iter().find(|b| b.id == id).map(shown) else { continue };
            let info = self.pick_info(&b);
            let all: Vec<u32> = info.face_tris.values().flatten().copied().collect();
            self.fill_scene_tris(painter, vp, &b.mesh, &all, LIT_BODY_COLOR.gamma_multiply(LIT_BODY_OPACITY));
        }
        if !matches!(self.mode, Mode::Idle) {
            return;
        }
        // The sketches' regions as click targets: faint at rest, brighter under the cursor,
        // green when selected.
        let selected_region = match &self.picks.selection {
            Some(Selected { what: Selection::Region { sketch, key }, .. }) => Some((*sketch, key.clone())),
            _ => None,
        };
        for f in self.part.features.iter().filter(|f| f.visible() && !f.suppressed() && matches!(f.kind, FeatureKind::Sketch(_))) {
            let Some(b) = self.built.sketches.get(&f.id) else { continue };
            for r in &b.regions {
                let hovered = self.picks.hover_region.as_ref() == Some(&(f.id, r.key.clone()));
                let picked = selected_region.as_ref() == Some(&(f.id, r.key.clone()));
                let color = match (picked, hovered) {
                    (true, true) => PICKED_COLOR.gamma_multiply(0.62),
                    (true, false) => PICKED_COLOR.gamma_multiply(0.5),
                    (false, true) => SKETCH_COLOR.gamma_multiply(0.32),
                    (false, false) => SKETCH_COLOR.gamma_multiply(0.16),
                };
                self.fill_region(painter, vp, f.id, &r.key, color);
            }
        }
        // A sketch line under the cursor or selected.
        let line_sel = match &self.picks.selection {
            Some(Selected { what: Selection::SketchLine { sketch, entity }, .. }) => Some((*sketch, *entity)),
            _ => None,
        };
        for (l, color) in [(self.picks.hover_line, HOVER_COLOR), (line_sel, PICKED_COLOR)] {
            let Some((sketch, entity)) = l else { continue };
            let Some(s) = self.part.sketch(sketch) else { continue };
            let frame = self.sketch_frame(sketch).unwrap_or(s.frame);
            if let Some(Shape::Line { a, b }) = s.sketch.entity(entity).map(|e| e.shape.clone()) {
                if let (Some(a), Some(b)) = (self.project(vp, frame.to_world(a)), self.project(vp, frame.to_world(b))) {
                    painter.line_segment([a, b], Stroke::new(3.2, color));
                }
            }
        }
        // The faces, edges and corners selected: the faces in the renderer (depth-tested), the
        // edges and corners here.
        for i in self.picks.items().to_vec() {
            match i {
                PickedItem::Edge { edge, .. } => {
                    let pts: Vec<Pos2> = edge.polyline.iter().filter_map(|q| self.project(vp, *q)).collect();
                    painter.add(egui::Shape::line(pts, Stroke::new(3.0, PICKED_COLOR)));
                }
                PickedItem::Vertex { point, .. } => {
                    if let Some(q) = self.project(vp, point) {
                        painter.circle_filled(q, VERTEX_LIT_PX / 2.0, PICKED_COLOR);
                    }
                }
                PickedItem::Face { .. } => {}
            }
        }
        // What a click would take: white, over everything.
        if let Some(h) = self.picks.hover.clone() {
            match h {
                Candidate::Face { body, face, .. } => {
                    if let Some(b) = self.pick_view(body).map(shown) {
                        let info = self.pick_info(&b);
                        if let Some(tris) = info.face_tris.get(&face) {
                            self.fill_scene_tris(painter, vp, &b.mesh, tris, HOVER_COLOR.gamma_multiply(0.35));
                        }
                    }
                }
                Candidate::Edge { body, edge } => {
                    if let Some(e) = self.pick_view(body).and_then(|b| b.topology.edges.get(edge)) {
                        let pts: Vec<Pos2> = e.polyline.iter().filter_map(|q| self.project(vp, *q)).collect();
                        painter.add(egui::Shape::line(pts, Stroke::new(3.0, HOVER_COLOR)));
                    }
                }
                Candidate::Vertex { point, .. } => {
                    if let Some(q) = self.project(vp, point) {
                        painter.circle_filled(q, VERTEX_LIT_PX / 2.0, HOVER_COLOR);
                    }
                }
            }
        }
        // The name of the coordinate plane under the cursor, or selected.
        let named = self.picks.hover_plane.or(match &self.picks.selection {
            Some(Selected { what: Selection::CoordinatePlane { axis }, .. }) => Some(*axis),
            _ => None,
        });
        if let Some(axis) = named.filter(|a| self.plane_shown(*a)) {
            if let Some((f, half)) = self.plane_square(axis) {
                if let Some(q) = self.project(vp, f.to_world(DVec2::new(half, half))) {
                    let font = egui::FontId::proportional(12.0);
                    let galley = painter.layout_no_wrap(axis.plane_name().to_string(), font, Color32::WHITE);
                    let rect = egui::Rect::from_min_size(q + egui::vec2(4.0, -galley.size().y - 6.0), galley.size() + egui::vec2(8.0, 4.0));
                    painter.rect_filled(rect, 3.0, plane_color(axis).gamma_multiply(0.9));
                    painter.galley(rect.min + egui::vec2(4.0, 2.0), galley, Color32::WHITE);
                }
            }
        }
        // The window being dragged: to the right a window (solid), to the left a crossing
        // (dashed).
        if let Some((a, b)) = self.picks.window {
            if (b.x - a.x).abs() + (b.y - a.y).abs() > WINDOW_MOVE_PX {
                let r = egui::Rect::from_two_pos(a, b);
                if b.x < a.x {
                    painter.rect_filled(r, 0.0, CROSSING_COLOR.gamma_multiply(0.12));
                    let c = [r.left_top(), r.right_top(), r.right_bottom(), r.left_bottom(), r.left_top()];
                    painter.extend(egui::Shape::dashed_line(&c, Stroke::new(1.0, CROSSING_COLOR), 5.0, 3.0));
                } else {
                    painter.rect_filled(r, 0.0, WINDOW_COLOR.gamma_multiply(0.12));
                    painter.rect_stroke(r, 0.0, Stroke::new(1.0, WINDOW_COLOR), egui::StrokeKind::Inside);
                }
            }
        }
        if self.picks.hover.is_some() || self.picks.hover_region.is_some() || self.picks.hover_line.is_some() || self.picks.hover_plane.is_some() {
            painter.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
        }
    }
}

impl ReverseWorkspace {
    /// The chip at the top of the stage: the build under way — only once it has taken 300 ms, so
    /// a quick build never flashes — else, over a scan with nothing built and no sketch open,
    /// where to begin.
    fn paint_stage_chips(&self, painter: &egui::Painter, vp: Viewport) {
        let building = self.picks.build_since.is_some_and(|t| t.elapsed() >= std::time::Duration::from_millis(300));
        let ready = !self.scan.is_empty() && self.part.features.is_empty() && !matches!(self.mode, Mode::Sketch(_));
        let text = if building {
            "Building the part…"
        } else if ready {
            "Part loaded — choose a plane in the panel to sketch on"
        } else {
            if self.picks.build_since.is_some() {
                painter.ctx().request_repaint_after(std::time::Duration::from_millis(100));
            }
            return;
        };
        let font = egui::FontId::proportional(12.5);
        let v = painter.ctx().global_style().visuals.clone();
        let galley = painter.layout_no_wrap(text.to_string(), font, v.text_color());
        let spin = if building { 18.0 } else { 0.0 };
        let size = galley.size() + egui::vec2(20.0 + spin, 10.0);
        let rect = egui::Rect::from_min_size(egui::pos2(vp.rect.center().x - size.x / 2.0, vp.rect.top() + 10.0), size);
        painter.rect(rect, 3.0, v.window_fill.gamma_multiply(0.96), Stroke::new(1.0, v.widgets.noninteractive.bg_stroke.color), egui::StrokeKind::Inside);
        if building {
            // A spinner: an arc turning with the clock.
            let c = rect.left_center() + egui::vec2(16.0, 0.0);
            let t = painter.ctx().input(|i| i.time) as f32 * 6.0;
            let pts: Vec<Pos2> = (0..=16).map(|k| t + k as f32 * 0.3).map(|a| c + egui::vec2(a.cos(), a.sin()) * 5.5).collect();
            painter.add(egui::Shape::line(pts, Stroke::new(1.8, v.selection.bg_fill)));
            painter.ctx().request_repaint();
        }
        painter.galley(rect.min + egui::vec2(10.0 + spin, 5.0), galley, v.text_color());
    }
}

/// The hatch's strokes on what bodies share: where a family of parallel planes, a spacing
/// apart that follows the part's size, cuts its triangles.
fn hatch(overlaps: &[Overlap], model_radius: f64) -> Vec<(DVec3, DVec3)> {
    let spacing = (model_radius / 60.0).max(0.25);
    let n = DVec3::new(1.0, 1.0, 1.0).normalize();
    let mut out = Vec::new();
    for o in overlaps {
        for t in &o.mesh.triangles {
            let p = [o.mesh.positions[t[0] as usize], o.mesh.positions[t[1] as usize], o.mesh.positions[t[2] as usize]];
            let s = p.map(|q| q.dot(n));
            let lo = s.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut k = (lo / spacing).ceil();
            while k * spacing <= hi {
                let level = k * spacing;
                let mut pts = Vec::with_capacity(2);
                for (i, j) in [(0, 1), (1, 2), (2, 0)] {
                    let (a, b) = (s[i] - level, s[j] - level);
                    if (a < 0.0) != (b < 0.0) && (a - b).abs() > 1e-12 {
                        pts.push(p[i] + (p[j] - p[i]) * (a / (a - b)));
                    }
                }
                if pts.len() == 2 {
                    out.push((pts[0], pts[1]));
                }
                k += 1.0;
            }
        }
    }
    out
}

fn same_candidate(a: &Candidate, b: &Candidate) -> bool {
    match (a, b) {
        (Candidate::Face { body: x, face: f, .. }, Candidate::Face { body: y, face: g, .. }) => x == y && f == g,
        (Candidate::Vertex { body: x, point: p }, Candidate::Vertex { body: y, point: q }) => x == y && (*p - *q).length() < 1e-3,
        _ => a == b,
    }
}

/// Every pick's key, for a test or a report.
#[allow(dead_code)]
pub(crate) fn keys_of(items: &[PickedItem]) -> Vec<String> {
    items.iter().map(item_key).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_hatch_strokes_cross_what_bodies_share() {
        let mesh = mm_reverse::build::BodyMesh { positions: vec![DVec3::ZERO, DVec3::new(10.0, 0.0, 0.0), DVec3::new(0.0, 10.0, 0.0)], triangles: vec![[0, 1, 2]], ..Default::default() };
        let o = Overlap { a: BodyId::new(1), b: BodyId::new(2), mesh, volume: 1.0 };
        let strokes = hatch(&[o], 15.0);
        // A stroke every quarter millimetre along (1, 1, 1) across the triangle, each on it.
        assert!(strokes.len() > 10, "{}", strokes.len());
        for (a, b) in &strokes {
            for p in [a, b] {
                assert!(p.z.abs() < 1e-9 && p.x >= -1e-9 && p.y >= -1e-9 && p.x + p.y <= 10.0 + 1e-9);
            }
            assert!(((*a - *b).dot(DVec3::ONE)).abs() < 1e-9);
        }
    }

    #[test]
    fn a_segment_meets_a_box_it_crosses_or_ends_in() {
        let r = egui::Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(10.0, 10.0));
        assert!(segment_meets_box(Pos2::new(-5.0, 5.0), Pos2::new(15.0, 5.0), r));
        assert!(segment_meets_box(Pos2::new(5.0, 5.0), Pos2::new(50.0, 50.0), r));
        assert!(!segment_meets_box(Pos2::new(-5.0, -5.0), Pos2::new(-1.0, 20.0), r));
    }
}
