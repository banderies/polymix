// SPDX-License-Identifier: AGPL-3.0-only
//! The elements a part is modelled against, and the planes a sketch is put on. Ported from
//! ScanRuler (<https://github.com/CNCKitchen/scanruler>): the element box of its 3D Measure
//! workspace (`src/ui/Panel.tsx`, `src/ui/DraftEditor.tsx`, `src/ui/SectionEditor.tsx`,
//! `src/app/measureWorkspace.ts`, `src/commands/core/element.ts`), and from its Reverse engineer
//! workspace the plane picker (`plugins/reverse/ui/PlaneSelect.tsx`), the plane and axis hints
//! (`plugins/reverse/viewer/sketchOverlay.ts`, `app/useReverseWorkspace.ts`) and the chips an
//! element offers (`plugins/reverse/ui/ContextChips.tsx`), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only. The element logic is `mm_reverse::measure_elements`.
//!
//! ScanRuler measures its elements in the 3D Measure workspace and the Reverse engineer
//! workspace builds on them; Polymix has no Measure workspace, so the elements are made here, in
//! the panel's Elements group: a click on the scan grows the surface under it and fits a plane,
//! sphere, cylinder, cone or torus to it with its σ and form error, or a surface marked by hand
//! is fitted as it is; a point is picked, a circle goes through three or more picks, and points,
//! lines, planes and circles are constructed from other elements; a section is a plane taken
//! across a coordinate axis or an element. They are drawn on the part as datums, and a click on
//! one offers what it can become — a sketch on a plane, a revolve about an axis, a hole along a
//! cylinder, a body at its extents.
//!
//! Where this differs from ScanRuler: the part keeps an element's geometry and fit, not how it
//! was made, so an element is not opened again to be refitted (it is renamed, hidden or deleted);
//! a constructed element does not follow its sources; the marking tools are a window and a
//! brush; sections are cuts (no shadows) and are not tilted by a gizmo. Polymix cannot tell
//! whether the part has been aligned, so the coordinate planes are offered at zero and through
//! the scan's centre both.

use std::collections::{HashMap, HashSet};
use std::sync::mpsc::{channel, Receiver};
use std::sync::Arc;

use egui::{Color32, Pos2, RichText, Stroke};
use glam::{DVec3, Vec4};
use mm_core::Mesh;
use mm_reverse::build::{resolve_axis, resolve_plane_frame, up_u};
use mm_reverse::display_units::DisplayUnits;
use mm_reverse::fit3d::{ElementKind, FitData, FitOutput, FitSettings};
use mm_reverse::frame::{SectionFrame, WorldAxis};
use mm_reverse::measure_elements as me;
use mm_reverse::measure_elements::{MethodMode, PlaneOffer, RefRole, SectionAlong};
use mm_reverse::mesh_graph::MeshGraph;
use mm_reverse::model::{AxisRef, BodyOp, ElementGeometry, ElementId, ExtractFeature, FeatureKind, HoleAxis, HoleFeature, PlaneRef, RadialAxis};

use super::{Mode, ReverseWorkspace};
use crate::renderer::{ColorVertex, GpuMesh, ObjectStyle, SceneRenderer};
use crate::tools::Viewport;

/// What the status line says as a sketch opens on a plane, from the Sketch on list or a chip.
pub(crate) const SKETCH_STARTED: &str = "Fit the slice chains, or draw the profile with the tools, then create the sketch.";
/// A plane or an axis under the pointer in a list, shown on the part.
const HINT: Color32 = Color32::from_rgb(0xff, 0xb0, 0x20);
/// The sketch's colour: its plane on the part while it is open.
const SKETCH_BLUE: Color32 = Color32::from_rgb(0x3d, 0x8f, 0xd6);
/// What is marked by hand.
const MARKED: Color32 = Color32::from_rgb(0x3c, 0xb3, 0x71);
/// How near a click must come to an element's line, in pixels.
const ELEMENT_REACH_PX: f32 = 6.0;
/// The brush's radius on the screen.
const BRUSH_PX: f32 = 22.0;
/// The keys of the surfaces drawn besides the elements'.
const KEY_DRAFT: u64 = u64::MAX - 1;
const KEY_HINT: u64 = u64::MAX - 2;
const KEY_SKETCH: u64 = u64::MAX - 3;
const KEY_SECTION: u64 = u64::MAX - 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Status {
    Empty,
    Fitting,
    Ready,
    Failed,
}

/// A place clicked on the scan: the object, the corners of the triangle hit (the seeds a
/// surface grows from) and the point (part coordinates).
#[derive(Debug, Clone, Copy)]
struct Pick {
    object: u64,
    tri: [u32; 3],
    point: DVec3,
}

/// How the surface a fit is measured on is chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SelectMode {
    Auto,
    Paint,
}

/// The marking gesture in hand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MarkTool {
    Navigate,
    Window,
    Brush,
}

/// An element in the making (ScanRuler's `Draft`).
struct ElementDraft {
    kind: ElementKind,
    method: &'static str,
    picks: Vec<Pick>,
    settings: FitSettings,
    refs: Vec<Option<ElementId>>,
    params: Vec<f64>,
    status: Status,
    /// The fit, part coordinates.
    fit: Option<FitData>,
    /// The vertices it was measured on, of a scan object.
    region: Option<(u64, Vec<u32>)>,
    message: Option<String>,
    /// How numbers found on the scan were got (a centroid).
    note: Option<String>,
    /// What the fit said besides its numbers.
    told: Option<String>,
    seq: u64,
}

/// A section in the making: what it is taken across and how far along.
struct SectionDraft {
    along: Option<SectionAlong>,
    offset: f64,
    /// The chains the plane cuts through the scan, for the frame they were cut on.
    chains: Option<(SectionFrame, usize)>,
}

/// What a fit on a worker thread is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JobFor {
    Draft(u64),
    /// A plane fitted where the scan was clicked, to sketch on.
    Face,
}

/// A plane or an axis shown on the part from a list.
#[derive(Debug, Clone)]
pub(crate) enum Hint {
    Plane { plane: PlaneRef, name: String },
    Axis { axis: RadialAxis, name: String },
}

/// A plane or an axis a feature box shows on the part (part coordinates): the option of its
/// plane or axis list under the pointer, or the one chosen while the box is open (ScanRuler's
/// `planeHint` / `axisHint`, set by the boxes' pickers).
#[derive(Debug, Clone)]
pub(crate) enum BoxHint {
    /// `element`: the plane element or section it stands for, lit too.
    Plane { frame: SectionFrame, name: String, element: Option<ElementId> },
    /// `element`: the axis element it stands for, lit too.
    Axis { origin: DVec3, dir: DVec3, name: String, element: Option<ElementId> },
}

fn box_hint_id() -> egui::Id {
    egui::Id::NULL.with("re_box_hint")
}

/// Show a plane on the part this frame, amber with its name (spec §6.9) — for a feature box's
/// plane list: the option under the pointer, else the plane chosen while the box is open. A
/// box needs only `&ReverseWorkspace`; the hint is drawn by the elements' layer.
#[allow(dead_code)] // for the boxes' plane and axis lists; Extrude and Revolve paint their own (EXT)
pub(crate) fn show_plane_hint(ctx: &egui::Context, ws: &ReverseWorkspace, plane: &PlaneRef, name: &str) {
    if let Some(frame) = resolve_plane_frame(plane, &ws.part) {
        let element = match plane {
            PlaneRef::Plane { id, .. } | PlaneRef::Section { id, .. } => Some(*id),
            PlaneRef::Radial { axis: RadialAxis::Element { id }, .. } => Some(*id),
            _ => None,
        };
        put_box_hint(ctx, BoxHint::Plane { frame, name: name.to_string(), element });
    }
}

/// Show an axis on the part this frame: a fat amber line across the part with its name at the
/// foot (spec §6.9) — for a feature box's axis list. `origin` and `dir` in part coordinates
/// (what `mm_reverse::build::resolve_axis` gives).
#[allow(dead_code)] // for the boxes' plane and axis lists; Extrude and Revolve paint their own (EXT)
pub(crate) fn show_axis_hint(ctx: &egui::Context, origin: DVec3, dir: DVec3, name: &str) {
    if dir.length_squared() > 0.0 && origin.is_finite() && dir.is_finite() {
        put_box_hint(ctx, BoxHint::Axis { origin, dir, name: name.to_string(), element: None });
    }
}

/// `show_axis_hint` for an axis as a feature names it — an element's, a coordinate axis, a line
/// of `sketch` — an axis element lit with it.
#[allow(dead_code)] // for the boxes' plane and axis lists; Extrude and Revolve paint their own (EXT)
pub(crate) fn show_axis_ref_hint(ctx: &egui::Context, ws: &ReverseWorkspace, axis: &AxisRef, sketch: Option<&mm_reverse::model::SketchFeature>, name: &str) {
    let Some((origin, dir)) = resolve_axis(axis, sketch, &ws.part) else { return };
    if dir.length_squared() > 0.0 && origin.is_finite() && dir.is_finite() {
        let element = match axis {
            AxisRef::Element { id } => Some(*id),
            _ => None,
        };
        put_box_hint(ctx, BoxHint::Axis { origin, dir, name: name.to_string(), element });
    }
}

fn put_box_hint(ctx: &egui::Context, hint: BoxHint) {
    let pass = ctx.cumulative_pass_nr();
    ctx.data_mut(|d| d.insert_temp(box_hint_id(), (pass, hint)));
}

/// The hint a box set this frame (the panel is laid out before the viewport is drawn).
fn take_box_hint(ctx: &egui::Context) -> Option<BoxHint> {
    let pass = ctx.cumulative_pass_nr();
    ctx.data(|d| d.get_temp::<(u64, BoxHint)>(box_hint_id())).filter(|(p, _)| *p == pass).map(|(_, h)| h)
}

/// What a chip by an element does.
#[derive(Debug, Clone)]
enum ChipAct {
    RevolveAbout(ElementId),
    Hole(ElementId, f64),
    SketchOn(PlaneRef),
    Extract(ElementId),
}

/// The elements' layer: the box an element or a section is made in, the fits running, the
/// marking, what is hidden, picked and hinted, and the surfaces drawn.
pub(crate) struct ElementsUi {
    draft: Option<ElementDraft>,
    section: Option<SectionDraft>,
    seq: u64,
    sigma: u8,
    select_mode: SelectMode,
    mark_tool: MarkTool,
    marked: HashMap<u64, HashSet<u32>>,
    window_from: Option<Pos2>,
    graphs: HashMap<u64, (u64, Arc<MeshGraph>)>,
    graph_jobs: HashMap<u64, (u64, Receiver<Result<MeshGraph, String>>)>,
    fit_job: Option<(JobFor, u64, Receiver<Result<FitOutput, String>>)>,
    /// A place clicked to sketch on, waiting for its object's graph.
    face_seed: Option<Pick>,
    /// The next click on the scan fits a plane there to sketch on.
    face_pick: bool,
    hidden: HashSet<ElementId>,
    /// The element picked on the part, and where — its chips stand there.
    selected: Option<(ElementId, Pos2)>,
    /// The element row under the pointer.
    row_hover: Option<ElementId>,
    renaming: Option<(ElementId, String)>,
    /// The plane or axis a list's option under the pointer shows on the part.
    pub(crate) hint: Option<Hint>,
    /// The hint was set by the Sketch on list (which takes it away again when it closes).
    hint_from_list: bool,
    /// The plane or axis a feature box shows this frame (`show_plane_hint`, `show_axis_hint`).
    box_hint: Option<BoxHint>,
    meshes: HashMap<u64, (u64, Arc<Mesh>)>,
    pub(crate) gpu: HashMap<u64, GpuMesh>,
    section_counts: HashMap<ElementId, (u64, usize)>,
    /// Each section's cut through the scan (part coordinates), for the scan it was cut on.
    section_cuts: HashMap<ElementId, (u64, SectionFrame, Arc<Vec<Vec<DVec3>>>)>,
}

impl Default for ElementsUi {
    fn default() -> Self {
        Self {
            draft: None,
            section: None,
            seq: 0,
            sigma: 3,
            select_mode: SelectMode::Auto,
            mark_tool: MarkTool::Navigate,
            marked: HashMap::new(),
            window_from: None,
            graphs: HashMap::new(),
            graph_jobs: HashMap::new(),
            fit_job: None,
            face_seed: None,
            face_pick: false,
            hidden: HashSet::new(),
            selected: None,
            row_hover: None,
            renaming: None,
            hint: None,
            hint_from_list: false,
            box_hint: None,
            meshes: HashMap::new(),
            gpu: HashMap::new(),
            section_counts: HashMap::new(),
            section_cuts: HashMap::new(),
        }
    }
}

fn c32(rgb: u32) -> Color32 {
    Color32::from_rgb((rgb >> 16) as u8, (rgb >> 8) as u8, rgb as u8)
}

fn v4(c: Color32, a: f32) -> Vec4 {
    Vec4::new(c.r() as f32 / 255.0, c.g() as f32 / 255.0, c.b() as f32 / 255.0, a)
}

fn plural(n: usize) -> &'static str {
    if n == 1 {
        ""
    } else {
        "s"
    }
}

/// A hash of a geometry, to know when its drawing must be made again.
fn signature(parts: &[f64]) -> u64 {
    parts.iter().fold(1469598103934665603u64, |h, v| (h ^ v.to_bits()).wrapping_mul(1099511628211))
}

fn geometry_signature(g: &ElementGeometry, extra: &[f64]) -> u64 {
    let text = serde_json::to_string(g).unwrap_or_default();
    let mut h = signature(extra);
    for b in text.bytes() {
        h = (h ^ b as u64).wrapping_mul(1099511628211);
    }
    h
}

/// The axis a hint's radial axis stands for.
fn hint_axis(axis: RadialAxis, ws: &ReverseWorkspace) -> Option<(DVec3, DVec3)> {
    resolve_axis(&AxisRef::from(axis), None, &ws.part)
}

/// Where a ray meets a triangle, along it.
fn ray_triangle(o: DVec3, d: DVec3, a: DVec3, b: DVec3, c: DVec3) -> Option<f64> {
    let (e1, e2) = (b - a, c - a);
    let p = d.cross(e2);
    let det = e1.dot(p);
    if det.abs() < 1e-12 {
        return None;
    }
    let inv = 1.0 / det;
    let s = o - a;
    let u = s.dot(p) * inv;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = s.cross(e1);
    let v = d.dot(q) * inv;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = e2.dot(q) * inv;
    (t > 0.0).then_some(t)
}

fn seg_dist(p: Pos2, a: Pos2, b: Pos2) -> (f32, f32) {
    let ab = b - a;
    let t = if ab.length_sq() > 0.0 { ((p - a).dot(ab) / ab.length_sq()).clamp(0.0, 1.0) } else { 0.0 };
    ((a + ab * t - p).length(), t)
}

/// A plane's sheet and border as the hint draws it: a square `half` each way of the model's
/// centre laid on the plane.
fn sheet_corners(frame: &SectionFrame, centre: DVec3, half: f64) -> [DVec3; 4] {
    let c = centre - frame.normal * (centre - frame.origin).dot(frame.normal);
    let (u, v) = (frame.u * half, frame.v * half);
    [c - u - v, c + u - v, c + u + v, c - u + v]
}

impl ReverseWorkspace {
    // ---- the model's size ----

    /// The middle and the half-diagonal of what is on the stage (part coordinates): the scan, else
    /// the bodies, else a hand-sized blank stage.
    pub(crate) fn elements_model_bounds(&self) -> (DVec3, f64) {
        let mut lo = DVec3::splat(f64::INFINITY);
        let mut hi = DVec3::splat(f64::NEG_INFINITY);
        for o in self.scan.iter().filter(|o| o.visible) {
            let b = o.mesh.bounds();
            if b.is_empty() {
                continue;
            }
            for c in [b.min, b.max] {
                let p = self.to_part(c);
                lo = lo.min(p);
                hi = hi.max(p);
            }
        }
        if !lo.x.is_finite() {
            for b in &self.bodies {
                let bb = b.mesh.bounds();
                if bb.is_empty() {
                    continue;
                }
                for c in [bb.min, bb.max] {
                    let p = self.to_part(c);
                    lo = lo.min(p);
                    hi = hi.max(p);
                }
            }
        }
        if lo.x.is_finite() {
            ((lo + hi) / 2.0, ((hi - lo).length() / 2.0).max(1.0))
        } else {
            (DVec3::ZERO, 50.0)
        }
    }

    /// Whether the elements and sections are drawn on the part: with the scan they were measured
    /// on (each one's eye in the list besides). The view bar's Elements key (the VIEW package's
    /// `vb.elements_shown`) belongs here too.
    pub(crate) fn elements_drawn(&self) -> bool {
        self.show_scan
    }

    // ---- the scan under the pointer ----

    /// The scan under a screen point: its object, the triangle hit (its corners) and the point
    /// (part coordinates), and how far along the ray.
    fn elements_scan_hit(&mut self, vp: Viewport, p: Pos2) -> Option<(Pick, f64)> {
        let ray = self.view.camera.ray_from_ndc(vp.ndc(p), vp.aspect());
        let objects: Vec<super::ScanObject> = self.scan.iter().filter(|o| o.visible).cloned().collect();
        let mut best: Option<(Pick, f64)> = None;
        for o in &objects {
            let Some(bvh) = self.scan_bvh(o) else { continue };
            if let Some(h) = bvh.raycast(&o.mesh, &ray) {
                if best.as_ref().is_none_or(|b| h.t < b.1) {
                    best = Some((Pick { object: o.id, tri: o.mesh.triangle(h.tid), point: self.to_part(h.point) }, h.t));
                }
            }
        }
        best
    }

    // ---- the graphs fits grow on ----

    /// The graph of a scan object, built on a worker thread the first time it is asked for.
    fn elements_graph(&mut self, object: u64) -> Option<Arc<MeshGraph>> {
        let o = self.scan.iter().find(|o| o.id == object)?.clone();
        let version = o.mesh.shape_version() ^ o.mesh.topology_version().rotate_left(32);
        if let Some((v, g)) = self.elements.graphs.get(&object) {
            if *v == version {
                return Some(g.clone());
            }
        }
        if let Some((v, rx)) = self.elements.graph_jobs.get(&object) {
            if *v == version {
                match rx.try_recv() {
                    Ok(Ok(g)) => {
                        let g = Arc::new(g);
                        self.elements.graphs.insert(object, (version, g.clone()));
                        self.elements.graph_jobs.remove(&object);
                        return Some(g);
                    }
                    Ok(Err(e)) => {
                        self.elements.graph_jobs.remove(&object);
                        self.status = e;
                        return None;
                    }
                    Err(std::sync::mpsc::TryRecvError::Empty) => return None,
                    Err(_) => {
                        self.elements.graph_jobs.remove(&object);
                        return None;
                    }
                }
            }
        }
        let (tx, rx) = channel();
        let mesh = o.mesh.clone();
        std::thread::Builder::new()
            .name("element graph".into())
            .spawn(move || {
                let _ = tx.send(MeshGraph::from_mesh(&mesh, false).map_err(|e| e.0));
            })
            .ok();
        self.elements.graph_jobs.insert(object, (version, rx));
        None
    }

    /// Start building the graph of every object shown, so the first click fits at once.
    fn elements_warm_graphs(&mut self) {
        let ids: Vec<u64> = self.scan.iter().filter(|o| o.visible).map(|o| o.id).collect();
        for id in ids {
            let _ = self.elements_graph(id);
        }
        self.elements_warm_bvhs();
    }

    /// Start building what a click on the scan is cast against, so the first click lands.
    fn elements_warm_bvhs(&mut self) {
        let objects: Vec<super::ScanObject> = self.scan.iter().filter(|o| o.visible).cloned().collect();
        for o in &objects {
            let _ = self.scan_bvh(o);
        }
    }

    /// The graph of a scan object, built here and now (a script's step).
    fn elements_graph_now(&mut self, object: u64) -> Result<Arc<MeshGraph>, String> {
        if let Some(g) = self.elements_graph(object) {
            return Ok(g);
        }
        let o = self.scan.iter().find(|o| o.id == object).ok_or("The scan object is gone.")?.clone();
        let version = o.mesh.shape_version() ^ o.mesh.topology_version().rotate_left(32);
        let g = Arc::new(MeshGraph::from_mesh(&o.mesh, false).map_err(|e| e.0)?);
        self.elements.graphs.insert(object, (version, g.clone()));
        self.elements.graph_jobs.remove(&object);
        Ok(g)
    }

    // ---- the box an element is made in ----

    /// Open the box on a kind (pressed again, the kind in hand starts its box over).
    pub(crate) fn elements_start(&mut self, kind: ElementKind) {
        self.elements.section = None;
        self.elements.face_pick = false;
        self.elements.marked.clear();
        self.elements.seq += 1;
        let method = me::default_method(kind);
        self.elements.draft = Some(self.elements_fresh_draft(kind, method));
        self.elements_method_opened();
    }

    fn elements_fresh_draft(&self, kind: ElementKind, method: &'static str) -> ElementDraft {
        let m = me::creation_method(kind, method);
        ElementDraft {
            kind,
            method,
            picks: Vec::new(),
            settings: FitSettings { sigma: self.elements.sigma, ..FitSettings::default() },
            refs: m.map_or(Vec::new(), |m| vec![None; m.slots.len()]),
            params: m.map_or(Vec::new(), |m| vec![f64::NAN; m.params.len()]),
            status: Status::Empty,
            fit: None,
            region: None,
            message: None,
            note: None,
            told: None,
            seq: self.elements.seq,
        }
    }

    /// The status a method opens with, the graphs warmed for a fit, a centroid measured.
    fn elements_method_opened(&mut self) {
        let Some(d) = &self.elements.draft else { return };
        let Some(m) = me::creation_method(d.kind, d.method) else { return };
        self.status = if m.mode == MethodMode::Construct { "Select the source elements in the panel.".into() } else { m.hint.to_string() };
        if m.mode == MethodMode::Fit {
            self.elements_warm_graphs();
        } else if m.mode == MethodMode::Pick {
            self.elements_warm_bvhs();
        }
        if m.id == "point-centroid" {
            self.elements_measure_centroid();
        }
    }

    /// A centroid draft measures itself as it opens: the enclosed volume's, or an open scan's
    /// surface's (ScanRuler's `measureCentroid`).
    fn elements_measure_centroid(&mut self) {
        let Some(o) = self.scan.iter().find(|o| o.visible).cloned() else {
            if let Some(d) = &mut self.elements.draft {
                d.status = Status::Failed;
                d.message = Some("The centroid could not be measured.".into());
            }
            return;
        };
        let closed = o.mesh.is_closed();
        let at = self.to_part(if closed { mm_core::measure::center_of_mass(&o.mesh) } else { mm_core::measure::surface_centroid(&o.mesh) });
        let note = if closed { format!("The centroid of the enclosed volume, {} cm³.", mm_reverse::js_fmt::to_fixed(mm_core::measure::volume(&o.mesh) / 1000.0, 2)) } else { "The scan is open — this is the centroid of its surface, not of a volume.".to_string() };
        if let Some(d) = &mut self.elements.draft {
            d.params = vec![at.x, at.y, at.z];
            d.note = Some(note);
        }
        self.elements_evaluate();
    }

    /// Recompute a construction's preview from the elements as they stand.
    fn elements_evaluate(&mut self) {
        let (_, size) = self.elements_model_bounds();
        let Some(d) = &mut self.elements.draft else { return };
        let Some(m) = me::creation_method(d.kind, d.method) else { return };
        if m.mode != MethodMode::Construct {
            return;
        }
        if d.refs.iter().any(|r| r.is_none()) || d.params.iter().any(|p| !p.is_finite()) {
            d.status = Status::Empty;
            d.fit = None;
            d.message = None;
            return;
        }
        let mut fits = Vec::new();
        for id in d.refs.iter().flatten() {
            match self.part.element(*id) {
                Some(e) => fits.push(me::fit_of(e, size)),
                None => {
                    d.status = Status::Failed;
                    d.fit = None;
                    d.message = Some("A reference is unavailable.".into());
                    return;
                }
            }
        }
        match me::evaluate_construction(d.method, &fits, &d.params, size) {
            Ok(f) => {
                d.status = Status::Ready;
                d.fit = Some(f);
                d.message = None;
            }
            Err(e) => {
                d.status = Status::Failed;
                d.fit = None;
                d.message = Some(e);
            }
        }
    }

    /// A click on the scan for the open box: a point moved, a circle's point added, a surface's
    /// seed added and the fit run again on all of them.
    fn elements_pick(&mut self, pick: Pick) {
        let Some(d) = &mut self.elements.draft else { return };
        let Some(m) = me::creation_method(d.kind, d.method) else { return };
        match m.mode {
            MethodMode::Construct => {}
            MethodMode::Pick if d.kind == ElementKind::Point => {
                d.picks = vec![pick];
                d.fit = Some(me::picked_point(pick.point));
                d.status = Status::Ready;
                self.status = "Click again to move the point, or create it.".into();
            }
            MethodMode::Pick => {
                d.picks.push(pick);
                self.elements_pick_fit();
            }
            MethodMode::Fit => {
                // A fit grows on one scan object: a pick on another starts the picks again.
                if d.picks.first().is_some_and(|p| p.object != pick.object) {
                    d.picks.clear();
                }
                d.picks.push(pick);
                self.elements_launch_fit();
            }
        }
    }

    /// A circle through the picks, once there are enough.
    fn elements_pick_fit(&mut self) {
        let Some(d) = &mut self.elements.draft else { return };
        let min = me::creation_method(d.kind, d.method).map_or(1, |m| m.min_picks);
        if d.picks.len() < min {
            d.status = Status::Empty;
            d.fit = None;
            let left = min - d.picks.len();
            self.status = format!("{left} more point{} to go — spread them around the {}.", plural(left), me::kind_info(d.kind).noun);
            return;
        }
        let points: Vec<DVec3> = d.picks.iter().map(|p| p.point).collect();
        match me::circle_through(&points) {
            Ok(f) => {
                d.fit = Some(f);
                d.status = Status::Ready;
                d.message = None;
                self.status = "More points refine the fit, or create it.".into();
            }
            Err(e) => {
                d.status = Status::Failed;
                d.fit = None;
                d.message = Some(e);
            }
        }
    }

    /// Run the open box's fit on a worker thread: from its picks, or on the marked surface.
    fn elements_launch_fit(&mut self) {
        let (seq, kind, settings, how, object) = {
            let Some(d) = &mut self.elements.draft else { return };
            self.elements.seq += 1;
            d.seq = self.elements.seq;
            d.status = Status::Fitting;
            d.message = None;
            let (how, object) = if self.elements.select_mode == SelectMode::Paint {
                let Some((object, set)) = self.elements.marked.iter().max_by_key(|(_, s)| s.len()) else {
                    d.status = Status::Empty;
                    return;
                };
                let mut v: Vec<u32> = set.iter().copied().collect();
                v.sort_unstable();
                (Err(v), *object)
            } else {
                let Some(first) = d.picks.first() else {
                    d.status = Status::Empty;
                    return;
                };
                (Ok(d.picks.iter().flat_map(|p| p.tri).collect::<Vec<u32>>()), first.object)
            };
            (d.seq, d.kind, d.settings, how, object)
        };
        let Some(graph) = self.elements_graph(object) else {
            // The graph is still being built: the fit starts when it lands (see the poll).
            return;
        };
        let (tx, rx) = channel();
        std::thread::Builder::new()
            .name("element fit".into())
            .spawn(move || {
                let r = match how {
                    Ok(seeds) => me::fit_from_seeds(&graph, kind, &seeds, &settings),
                    Err(sel) => me::fit_on_marked(&graph, kind, &sel, &settings),
                };
                let _ = tx.send(r);
            })
            .ok();
        self.elements.fit_job = Some((JobFor::Draft(seq), object, rx));
    }

    /// The fits and graphs running: what has come in is taken.
    fn elements_poll(&mut self, ctx: &egui::Context) {
        let busy = !self.elements.graph_jobs.is_empty() || self.elements.fit_job.is_some();
        // Graphs landing: a fit waiting on one starts.
        let waiting: Vec<u64> = self.elements.graph_jobs.keys().copied().collect();
        for id in waiting {
            if self.elements_graph(id).is_some() {
                let draft_waits = self.elements.draft.as_ref().is_some_and(|d| d.status == Status::Fitting) && self.elements.fit_job.is_none();
                if draft_waits {
                    self.elements_launch_fit();
                }
                if let Some(seed) = self.elements.face_seed.filter(|s| s.object == id) {
                    self.elements.face_seed = None;
                    self.elements_face_fit(seed);
                }
            }
        }
        if let Some((what, object, rx)) = &self.elements.fit_job {
            match rx.try_recv() {
                Ok(r) => {
                    let (what, object) = (*what, *object);
                    self.elements.fit_job = None;
                    self.elements_fit_landed(what, object, r);
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
                Err(_) => self.elements.fit_job = None,
            }
        }
        if busy {
            ctx.request_repaint_after(std::time::Duration::from_millis(30));
        }
    }

    fn elements_fit_landed(&mut self, what: JobFor, object: u64, r: Result<FitOutput, String>) {
        let flip = self.flip;
        let to_part = move |p: DVec3| if flip { crate::app::flip_zy_out(p) } else { p };
        match what {
            JobFor::Draft(seq) => {
                let Some(d) = self.elements.draft.as_mut().filter(|d| d.seq == seq) else { return };
                match r {
                    Ok(out) => {
                        d.fit = Some(me::map_fit(&out.fit, to_part));
                        d.region = Some((object, out.region));
                        d.told = out.told.map(|t| t.note);
                        d.status = Status::Ready;
                        d.message = None;
                        let noun = me::kind_info(d.kind).noun;
                        self.status = if self.elements.select_mode == SelectMode::Paint { "Keep marking to add more; Shift rubs out.".into() } else { format!("Add more points if the {noun} is split across unconnected patches.") };
                    }
                    Err(e) => {
                        d.status = Status::Failed;
                        d.fit = None;
                        d.region = None;
                        d.message = Some(e);
                    }
                }
            }
            JobFor::Face => match r {
                Ok(out) => match me::map_fit(&out.fit, to_part) {
                    FitData::Plane(p) => match me::fitted_plane(&p) {
                        Some(plane) => self.elements_sketch_on(plane),
                        None => self.status = "That plane is not available any more.".into(),
                    },
                    _ => self.status = "That plane is not available any more.".into(),
                },
                Err(e) => self.status = e,
            },
        }
    }

    /// Fit a plane where the scan was clicked and open a sketch on it.
    fn elements_face_fit(&mut self, pick: Pick) {
        let Some(graph) = self.elements_graph(pick.object) else {
            self.elements.face_seed = Some(pick);
            return;
        };
        let (tx, rx) = channel();
        let seeds = pick.tri.to_vec();
        std::thread::spawn(move || {
            let _ = tx.send(me::fit_from_seeds(&graph, ElementKind::Plane, &seeds, &FitSettings::default()));
        });
        self.elements.fit_job = Some((JobFor::Face, pick.object, rx));
        self.status = "Fitting the face…".into();
    }

    /// Create the element the box holds; the kind stays in hand for the next one.
    fn elements_create(&mut self) {
        let Some(d) = self.elements.draft.take() else { return };
        let Some(fit) = d.fit.filter(|_| d.status == Status::Ready) else {
            self.elements.draft = Some(d);
            return;
        };
        let info = me::kind_info(d.kind);
        let (geometry, stats) = me::geometry_of(&fit);
        let name = me::next_element_name(&self.part, info.label);
        self.edit_part(|p| {
            me::add_element(p, name.clone(), geometry, stats);
        });
        self.elements.marked.clear();
        self.elements.seq += 1;
        let next = self.elements_fresh_draft(d.kind, d.method);
        self.elements.draft = Some(ElementDraft { settings: d.settings, ..next });
        self.status = format!("{name} created — the {} stays in hand for the next one; Esc or Cancel puts it down.", info.noun);
        if d.method == "point-centroid" {
            self.elements_measure_centroid();
        }
    }

    /// Put the box down.
    pub(crate) fn elements_cancel(&mut self) {
        let had = self.elements.draft.take().is_some() || self.elements.section.take().is_some();
        self.elements.marked.clear();
        self.elements.window_from = None;
        if had {
            self.status = String::new();
        }
    }

    // ---- sections ----

    pub(crate) fn elements_start_section(&mut self) {
        self.elements.draft = None;
        self.elements.face_pick = false;
        self.elements.section = Some(SectionDraft { along: None, offset: 0.0, chains: None });
        self.elements_warm_bvhs();
        self.status = "Choose what to cut across — click an element or one of the XY, YZ, XZ planes in the viewport, or pick it above. The plane runs across an element’s direction: a plane’s normal, the axis of a cylinder, cone or line, a circle’s normal.".into();
    }

    /// The open section's plane, its chains cut when it moved.
    fn elements_section_frame(&mut self) -> Option<SectionFrame> {
        let s = self.elements.section.as_ref()?;
        let frame = me::section_frame(s.along?, s.offset, &self.part)?;
        if self.elements.section.as_ref().and_then(|s| s.chains).is_none_or(|(f, _)| f != frame) {
            let n = if self.scan.is_empty() { 0 } else { self.slice(&frame).len() };
            if let Some(s) = &mut self.elements.section {
                s.chains = Some((frame, n));
            }
        }
        Some(frame)
    }

    fn elements_create_section(&mut self) {
        let Some(frame) = self.elements_section_frame() else { return };
        let name = me::next_element_name(&self.part, "Section");
        self.edit_part(|p| {
            me::add_element(p, name.clone(), ElementGeometry::Section { frame }, None);
        });
        self.elements.section = None;
        self.status = format!("{name} created.");
    }

    /// How many chains a section cuts through the scan, read once per scan.
    fn elements_section_chains(&mut self, id: ElementId, frame: &SectionFrame) -> usize {
        let key = self.scan_parts().len() as u64 ^ self.scan_parts.as_ref().map_or(0, |s| s.0);
        if let Some((k, n)) = self.elements.section_counts.get(&id) {
            if *k == key {
                return *n;
            }
        }
        let n = self.slice(frame).len();
        self.elements.section_counts.insert(id, (key, n));
        n
    }

    // ---- sketches on planes ----

    /// Open a sketch on a plane, as the Sketch on list and the chips do.
    pub(crate) fn elements_sketch_on(&mut self, plane: PlaneRef) {
        self.elements.hint = None;
        self.elements.selected = None;
        self.elements.face_pick = false;
        if resolve_plane_frame(&plane, &self.part).is_none() {
            self.status = "That plane is not available any more.".into();
            return;
        }
        self.new_sketch(plane);
        if matches!(self.mode, Mode::Sketch(_)) {
            self.status = SKETCH_STARTED.into();
        }
    }

    /// Every plane a sketch is offered: the sections, the plane elements and the coordinate
    /// planes (ScanRuler's), then — for what ScanRuler reaches through its agent's commands —
    /// a plane through each axis, and a plane fitted on a face of the scan where it is clicked.
    fn elements_sketch_offers(&self) -> Vec<PlaneOffer> {
        let centre = if self.scan.is_empty() { None } else { self.scan_centre() };
        let mut offers = me::plane_offers(&self.part, centre, true);
        for (axis, name) in me::radial_axis_offers(&self.part) {
            offers.push(PlaneOffer { key: format!("radial:{name}"), label: format!("Through {name}"), plane: PlaneRef::Radial { axis, angle_deg: 0.0, offset: None }, group: "Through an axis" });
        }
        offers
    }

    /// The Sketch on field of the Create feature group: a list of the planes, the one under the
    /// pointer shown on the part; choosing one opens a sketch on it.
    pub(crate) fn sketch_on_field(&mut self, ui: &mut egui::Ui) {
        let offers = self.elements_sketch_offers();
        let mut hovered: Option<Hint> = None;
        let mut chosen: Option<PlaneRef> = None;
        let mut face = false;
        let has_scan = !self.scan.is_empty();
        ui.horizontal(|ui| {
            ui.label("Sketch on");
            let r = egui::ComboBox::from_id_salt("re_sketch_on").selected_text(RichText::new(if self.elements.face_pick { "Click a face of the scan…" } else { "Choose a plane…" }).weak()).width(ui.available_width() - 8.0).show_ui(ui, |ui| {
                if offers.is_empty() {
                    ui.weak("Nothing to choose from");
                }
                let mut group = "";
                for o in &offers {
                    if o.group != group {
                        group = o.group;
                        ui.label(RichText::new(group).small().weak());
                    }
                    let r = ui.selectable_label(false, &o.label);
                    if r.hovered() {
                        hovered = Some(match &o.plane {
                            PlaneRef::Radial { axis, .. } => Hint::Axis { axis: *axis, name: o.label.trim_start_matches("Through ").to_string() },
                            p => Hint::Plane { plane: p.clone(), name: o.label.clone() },
                        });
                    }
                    if r.clicked() {
                        chosen = Some(o.plane.clone());
                    }
                }
                if has_scan {
                    ui.label(RichText::new("On the scan").small().weak());
                    if ui.selectable_label(false, "A face of the scan — click it…").on_hover_text("Click a flat face of the scan: a plane is fitted there and the sketch put on it").clicked() {
                        face = true;
                    }
                }
            });
            r.response.on_hover_text("The plane to sketch on — the one under the pointer is shown on the part");
        });
        if hovered.is_some() || self.elements.hint_from_list {
            self.elements.hint_from_list = hovered.is_some();
            self.elements.hint = hovered;
        }
        if face {
            self.elements.face_pick = true;
            self.elements.draft = None;
            self.elements.section = None;
            self.elements_warm_graphs();
            self.status = "Click a flat face of the scan: a plane is fitted there and the sketch put on it.".into();
        }
        if let Some(p) = chosen {
            self.elements_sketch_on(p);
        }
    }

    /// The hint under the Create feature keys while the part has no sketch yet (ScanRuler's
    /// `ReversePanel`): where a model starts. ScanRuler cuts its sections in the 3D Measure
    /// workspace; here they are cut under Create element.
    pub(crate) fn sketch_start_hint(&self, ui: &mut egui::Ui) {
        if self.last_sketch().is_some() {
            return;
        }
        ui.add_space(2.0);
        ui.weak(if self.scan.is_empty() { "Start with a sketch: choose a coordinate plane above. No scan is needed to model." } else { "Start with a sketch: choose a plane above, or cut a section under Create element below." });
    }

    /// Under the open sketch's box, for a plane through an axis its angle, and for a plane that
    /// takes one the coordinate axis its sheet's v runs up — what ScanRuler sets through its
    /// agent's commands (`angleDeg`, `up`).
    fn elements_plane_strip(&mut self, ui: &mut egui::Ui) {
        let Mode::Sketch(edit) = &self.mode else { return };
        let plane = edit.plane.clone();
        let mut next: Option<PlaneRef> = None;
        match &plane {
            PlaneRef::Radial { axis, angle_deg, offset } => {
                ui.horizontal(|ui| {
                    ui.label("Angle");
                    let mut a = *angle_deg;
                    let r = ui.add(egui::DragValue::new(&mut a).speed(0.5).suffix("°")).on_hover_text("Degrees about the axis: 0 is the coordinate axis next round (X about Z, Y about X, Z about Y), turning by the right hand — about Z, from +X towards +Y");
                    if r.changed() && a.is_finite() {
                        next = Some(PlaneRef::Radial { axis: *axis, angle_deg: a, offset: *offset });
                    }
                });
            }
            PlaneRef::World { up, .. } | PlaneRef::Section { up, .. } | PlaneRef::Plane { up, .. } => {
                let now = *up;
                ui.horizontal(|ui| {
                    ui.label("Up");
                    let name = |a: Option<WorldAxis>| match a {
                        None => "The plane’s own",
                        Some(WorldAxis::X) => "X",
                        Some(WorldAxis::Y) => "Y",
                        Some(WorldAxis::Z) => "Z",
                    };
                    egui::ComboBox::from_id_salt("re_sketch_up").selected_text(name(now)).show_ui(ui, |ui| {
                        for a in [None, Some(WorldAxis::X), Some(WorldAxis::Y), Some(WorldAxis::Z)] {
                            if ui.selectable_label(now == a, name(a)).clicked() && a != now {
                                let mut p = plane.clone();
                                if let PlaneRef::World { up, .. } | PlaneRef::Section { up, .. } | PlaneRef::Plane { up, .. } = &mut p {
                                    *up = a;
                                }
                                next = Some(p);
                            }
                        }
                    })
                    .response
                    .on_hover_text("The coordinate axis the sketch’s v runs up, as near as the plane lets it: u = up × normal, so a standing plane has that axis up whichever way its normal points. The plane’s own u by default.");
                });
            }
            _ => {}
        }
        if let Some(p) = next {
            self.elements_set_sketch_plane(p);
        }
    }

    /// The open sketch moved onto another plane (an angle, an up): its pieces keep their place on
    /// the sheet, the slice is cut again.
    pub(crate) fn elements_set_sketch_plane(&mut self, plane: PlaneRef) {
        if let PlaneRef::World { up: Some(a), .. } | PlaneRef::Section { up: Some(a), .. } | PlaneRef::Plane { up: Some(a), .. } = &plane {
            let normal = resolve_plane_frame(&plane, &self.part).map(|f| f.normal);
            if normal.is_some_and(|n| up_u(Some(*a), n).is_none()) {
                self.status = format!("up: {:?} runs along the plane’s normal and has no way up it — give one of the other two.", a);
                return;
            }
        }
        let Some(frame) = resolve_plane_frame(&plane, &self.part) else {
            self.status = "That plane is not available any more.".into();
            return;
        };
        let chains = self.slice(&me::slice_frame(&plane, &frame));
        if let Mode::Sketch(edit) = &mut self.mode {
            edit.set_plane(plane, frame, chains);
            self.view.face_plane(&edit.frame, self.flip);
        }
    }

    // ---- the panel group ----

    /// The Elements group: the kinds an element is made as, the box it is made in, the elements
    /// and the sections — with, while a sketch is open, its plane's angle or up.
    pub(crate) fn elements_group(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        self.elements_poll(&ctx);
        if !matches!(self.mode, Mode::Idle) {
            self.elements.draft = None;
            self.elements.section = None;
            self.elements.face_pick = false;
            self.elements.selected = None;
        }
        self.elements_plane_strip(ui);
        let idle = matches!(self.mode, Mode::Idle);
        let has_scan = !self.scan.is_empty();
        ui.separator();
        ui.horizontal(|ui| {
            ui.label(RichText::new("Create element").strong());
            ui.label(RichText::new("ⓘ").weak()).on_hover_text(
                "Every element starts as a plane, a cylinder, a sphere, a circle, a point or a line — the datums a part is modelled against.\n\n\
                 An element is either fitted to the scan, by clicking the feature and letting the tool find the surface; picked straight off the mesh, for a single point; or constructed from elements you already have — an axis through two circles, a midpoint between two points, a plane offset from another.\n\n\
                 Which of those a kind offers appears as Created once you choose it.\n\n\
                 The kind stays in hand after Create, ready for the next one, as a sketch tool does — Cancel or Esc puts it down.\n\n\
                 Section is not an element but a cut: the scan sliced with a plane taken across an element's direction, to sketch on.",
            );
        });
        let in_hand = self.elements.draft.as_ref().map(|d| d.kind);
        let mut start: Option<ElementKind> = None;
        let mut section = false;
        egui::Grid::new("re_element_kinds").num_columns(3).spacing([4.0, 4.0]).show(ui, |ui| {
            for (i, k) in me::ELEMENT_KINDS.iter().enumerate() {
                let on = in_hand == Some(k.kind);
                let r = ui.add_enabled(idle && has_scan, egui::Button::new(k.label).selected(on).min_size(egui::vec2(84.0, 0.0)));
                let r = if has_scan { r.on_disabled_hover_text("Finish what is open first") } else { r.on_disabled_hover_text("Open a scan in the Mesh workspace first") };
                if r.clicked() {
                    start = Some(k.kind);
                }
                if i % 3 == 2 {
                    ui.end_row();
                }
            }
            let r = ui.add_enabled(idle && has_scan, egui::Button::new("Section").selected(self.elements.section.is_some()).min_size(egui::vec2(84.0, 0.0))).on_hover_text("Cut the scan with a plane, to sketch on the cut");
            if r.clicked() {
                section = true;
            }
            ui.end_row();
        });
        if let Some(k) = start {
            self.elements_start(k);
        }
        if section {
            self.elements_start_section();
        }
        if self.elements.face_pick {
            ui.horizontal(|ui| {
                ui.colored_label(HINT, "Click a flat face of the scan to sketch on.");
                if ui.small_button("Cancel").clicked() {
                    self.elements.face_pick = false;
                    self.status.clear();
                }
            });
        }
        if self.elements.draft.is_some() {
            self.elements_draft_box(ui);
        }
        if self.elements.section.is_some() {
            self.elements_section_box(ui);
        }
        // Esc puts the box down; the open sketch and feature boxes have their own.
        if idle && (self.elements.draft.is_some() || self.elements.section.is_some() || self.elements.face_pick || self.elements.selected.is_some()) && !ctx.egui_wants_keyboard_input() && ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Escape)) {
            if self.elements.selected.take().is_none() {
                self.elements.face_pick = false;
                self.elements_cancel();
            }
        }
        self.elements_list(ui);
    }

    fn elements_draft_box(&mut self, ui: &mut egui::Ui) {
        let Some(d) = &self.elements.draft else { return };
        let info = me::kind_info(d.kind);
        let Some(method) = me::creation_method(d.kind, d.method) else { return };
        let color = c32(me::element_color(me::next_element_id(&self.part)));
        let methods = me::methods_for_kind(d.kind);
        let mut set_method: Option<&'static str> = None;
        let mut set_mode: Option<SelectMode> = None;
        let mut set_sigma: Option<u8> = None;
        let mut set_ref: Option<(usize, Option<ElementId>)> = None;
        let mut set_param: Option<(usize, f64)> = None;
        let mut create = false;
        let mut cancel = false;
        let mut undo_pick = false;
        let mut clear_marks = false;
        let mut tool: Option<MarkTool> = None;
        let paint = method.mode == MethodMode::Fit && self.elements.select_mode == SelectMode::Paint;
        let marked: usize = self.elements.marked.values().map(|s| s.len()).sum();
        let (_, size) = self.elements_model_bounds();
        egui::Frame::group(ui.style()).stroke(Stroke::new(1.0, color)).show(ui, |ui| {
            ui.horizontal(|ui| {
                let (r, _) = ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
                ui.painter().circle_filled(r.center(), 4.5, color);
                ui.label(RichText::new(format!("New {}", info.noun)).strong());
            });
            if methods.len() > 1 {
                ui.horizontal(|ui| {
                    ui.label("Created");
                    egui::ComboBox::from_id_salt("re_element_method").selected_text(method.label).show_ui(ui, |ui| {
                        for m in &methods {
                            if ui.selectable_label(m.id == method.id, m.label).clicked() && m.id != method.id {
                                set_method = Some(m.id);
                            }
                        }
                    });
                });
            }
            if method.mode == MethodMode::Fit {
                ui.horizontal(|ui| {
                    ui.label("Surface").on_hover_text(
                        "Found from a click: click once on the feature and the patch grows outward across the scan on its own, stopping where the surface stops being flat or round enough to belong. Fast, and right most of the time.\n\n\
                         Marked by hand: take exactly the triangles you want — with a window or a round brush. Use it when a click runs off across a fillet, when noise breaks the feature into unconnected patches, or when only part of a face is worth trusting.",
                    );
                    let mode = self.elements.select_mode;
                    egui::ComboBox::from_id_salt("re_element_select").selected_text(if mode == SelectMode::Auto { "Found from a click" } else { "Marked by hand" }).show_ui(ui, |ui| {
                        if ui.selectable_label(mode == SelectMode::Auto, "Found from a click").clicked() {
                            set_mode = Some(SelectMode::Auto);
                        }
                        if ui.selectable_label(mode == SelectMode::Paint, "Marked by hand").clicked() {
                            set_mode = Some(SelectMode::Paint);
                        }
                    });
                });
            }
            if paint {
                ui.horizontal(|ui| {
                    for (t, label, hint) in [(MarkTool::Navigate, "Navigate", "Clicks and drags move the view"), (MarkTool::Window, "Window", "Drag a box: the scan inside it facing you is marked"), (MarkTool::Brush, "Brush", "Drag over the scan to mark it; Shift rubs out")] {
                        if ui.selectable_label(self.elements.mark_tool == t, label).on_hover_text(hint).clicked() {
                            tool = Some(t);
                        }
                    }
                    if ui.add_enabled(marked > 0, egui::Button::new("Clear")).on_hover_text("Rub out the whole marking").clicked() {
                        clear_marks = true;
                    }
                });
            }
            if method.mode == MethodMode::Construct {
                ui.weak(method.hint);
                if let Some(n) = &d.note {
                    ui.weak(n);
                }
                for (i, s) in method.slots.iter().enumerate() {
                    let chosen = d.refs.get(i).copied().flatten();
                    let options: Vec<(ElementId, String)> = self
                        .part
                        .elements
                        .iter()
                        .filter(|e| match me::element_kind(&e.geometry) {
                            Some(k) => me::slot_takes(s, k),
                            None => s.role == RefRole::Plane && s.kinds.is_none(),
                        })
                        .map(|e| (e.id, e.name.clone()))
                        .collect();
                    ui.horizontal(|ui| {
                        ui.label(s.label);
                        let text = chosen.and_then(|c| options.iter().find(|o| o.0 == c)).map_or("Choose…".to_string(), |o| o.1.clone());
                        egui::ComboBox::from_id_salt(("re_element_slot", i)).selected_text(text).show_ui(ui, |ui| {
                            if options.is_empty() {
                                ui.weak("Nothing to choose from");
                            }
                            for (id, name) in &options {
                                if ui.selectable_label(chosen == Some(*id), name).clicked() {
                                    set_ref = Some((i, Some(*id)));
                                }
                            }
                        });
                    });
                }
                for (i, p) in method.params.iter().enumerate() {
                    if p.hidden {
                        continue;
                    }
                    ui.horizontal(|ui| {
                        ui.label(if p.length { format!("{} (mm)", p.label) } else { p.label.to_string() });
                        let held = d.params.get(i).copied().unwrap_or(f64::NAN);
                        let mut v = if held.is_finite() { held } else { 0.0 };
                        let r = ui.add(egui::DragValue::new(&mut v).speed(0.1).max_decimals(6));
                        if r.changed() || (!held.is_finite() && r.lost_focus()) {
                            set_param = Some((i, v));
                        }
                        if !held.is_finite() {
                            ui.weak("—");
                        }
                    });
                }
            } else {
                ui.weak(self.elements_pick_hint(d, method, paint, marked));
            }
            // The preview: what the box would create.
            ui.horizontal(|ui| {
                ui.label(RichText::new("Preview").small().weak());
                if paint {
                    ui.label(RichText::new(format!("{} point{} marked", mm_reverse::js_fmt::grouped(marked as f64), plural(marked))).small().weak());
                } else if method.mode != MethodMode::Construct {
                    ui.label(RichText::new(format!("{} pick{}", d.picks.len(), plural(d.picks.len()))).small().weak());
                }
            });
            egui::Frame::NONE.fill(ui.visuals().extreme_bg_color).inner_margin(egui::Margin::symmetric(8, 4)).show(ui, |ui| {
                ui.set_min_width(ui.available_width());
                match d.status {
                    Status::Empty => {
                        let text = if method.mode == MethodMode::Construct {
                            "Incomplete".to_string()
                        } else if paint {
                            "Nothing marked".to_string()
                        } else if !d.picks.is_empty() {
                            format!("{} of {} points", d.picks.len(), method.min_picks)
                        } else {
                            "No points picked".to_string()
                        };
                        ui.weak(text);
                    }
                    Status::Fitting => {
                        ui.horizontal(|ui| {
                            polymix_workspace::spinner(ui);
                            ui.label("Fitting…");
                        });
                    }
                    Status::Failed => {
                        ui.colored_label(Color32::from_rgb(0xe2, 0x6b, 0x4f), d.message.clone().unwrap_or_else(|| "Failed".into()));
                    }
                    Status::Ready => {
                        if let Some(f) = &d.fit {
                            let primary = me::format_primary(f, DisplayUnits::Mm);
                            if primary.is_empty() || matches!(f, FitData::Point(_) | FitData::Line(_)) {
                                ui.colored_label(color, format!("✓ {}", info.label));
                            } else {
                                ui.label(RichText::new(primary).color(color).strong().size(15.0).monospace());
                            }
                        }
                    }
                }
            });
            if let (Status::Ready, Some(f)) = (d.status, &d.fit) {
                ui.label(RichText::new(me::format_detail(f, DisplayUnits::Mm)).small());
            }
            if let Some(t) = &d.told {
                ui.colored_label(Color32::from_rgb(0xe2, 0xa4, 0x3b), t);
            }
            ui.add_space(2.0);
            if ui.add_enabled(d.status == Status::Ready, egui::Button::new(RichText::new(format!("Create {}", info.noun)).strong()).min_size(egui::vec2(ui.available_width(), 24.0))).clicked() {
                create = true;
            }
            ui.horizontal(|ui| {
                if method.mode != MethodMode::Construct && !paint && ui.add_enabled(!d.picks.is_empty(), egui::Button::new("Undo point")).clicked() {
                    undo_pick = true;
                }
                if ui.button("Cancel").on_hover_text("Esc").clicked() {
                    cancel = true;
                }
            });
            // How this one element is fitted: its own outlier cut-off.
            if method.mode == MethodMode::Fit {
                ui.label(RichText::new(format!("Fitting this {}", info.noun)).small().weak());
                ui.horizontal(|ui| {
                    ui.label("Method");
                    ui.add_enabled_ui(false, |ui| {
                        egui::ComboBox::from_id_salt("re_element_fit_method").selected_text("Gaussian best-fit").show_ui(ui, |_| {});
                    });
                });
                ui.horizontal(|ui| {
                    ui.label("Used points").on_hover_text(
                        "Every element is a Gaussian best fit — the plane, cylinder or sphere that minimises the squared distance to the points taken from the scan.\n\n\
                         This is the outlier cut-off. Points further from that first fit than the chosen multiple of the standard deviation are dropped and the fit is repeated, so scan noise, a stray edge triangle or a speck of spray cannot drag the result. Tighter cut-offs give a cleaner element from fewer points; all points keeps everything.\n\n\
                         The setting belongs to this element: changing it re-fits this one from the points it was built on. The last choice is what the next new element starts with.",
                    );
                    let now = d.settings.sigma;
                    let label = |k: u8| me::SIGMA_PRESETS.iter().find(|p| p.0 == k).map_or("3 sigma", |p| p.1);
                    egui::ComboBox::from_id_salt("re_element_sigma").selected_text(label(now)).show_ui(ui, |ui| {
                        for (k, l) in me::SIGMA_PRESETS {
                            if ui.selectable_label(now == k, l).clicked() && k != now {
                                set_sigma = Some(k);
                            }
                        }
                    });
                });
            }
        });
        let _ = size;
        if let Some(m) = set_method {
            let kind = self.elements.draft.as_ref().map(|d| d.kind);
            if let Some(kind) = kind {
                let settings = self.elements.draft.as_ref().map(|d| d.settings).unwrap_or_default();
                self.elements.seq += 1;
                let fresh = self.elements_fresh_draft(kind, m);
                self.elements.draft = Some(ElementDraft { settings, ..fresh });
                self.elements.marked.clear();
                self.elements_method_opened();
            }
        }
        if let Some(m) = set_mode {
            self.elements.select_mode = m;
            self.elements.marked.clear();
            if let Some(d) = &mut self.elements.draft {
                d.picks.clear();
                d.fit = None;
                d.region = None;
                d.status = Status::Empty;
            }
            self.elements.mark_tool = if m == SelectMode::Paint { MarkTool::Window } else { MarkTool::Navigate };
            self.status = if m == SelectMode::Paint { "Drag over the surface to mark it.".into() } else { String::new() };
        }
        if let Some(t) = tool {
            self.elements.mark_tool = t;
        }
        if clear_marks {
            self.elements.marked.clear();
            if let Some(d) = &mut self.elements.draft {
                d.fit = None;
                d.region = None;
                d.status = Status::Empty;
            }
        }
        if let Some(k) = set_sigma {
            self.elements.sigma = k;
            let refit = self.elements.draft.as_mut().map(|d| {
                d.settings.sigma = k;
                !d.picks.is_empty() || !self.elements.marked.is_empty()
            });
            if refit == Some(true) {
                self.elements_launch_fit();
            }
        }
        if let Some((i, r)) = set_ref {
            if let Some(d) = &mut self.elements.draft {
                if let Some(slot) = d.refs.get_mut(i) {
                    *slot = r;
                }
            }
            self.elements_evaluate();
        }
        if let Some((i, v)) = set_param {
            if let Some(d) = &mut self.elements.draft {
                if let Some(p) = d.params.get_mut(i) {
                    *p = v;
                }
            }
            self.elements_evaluate();
        }
        if undo_pick {
            let again = match &mut self.elements.draft {
                Some(d) => {
                    d.picks.pop();
                    d.fit = None;
                    d.region = None;
                    d.status = Status::Empty;
                    !d.picks.is_empty()
                }
                None => false,
            };
            let mode = self.elements.draft.as_ref().and_then(|d| me::creation_method(d.kind, d.method)).map(|m| m.mode);
            if again {
                match mode {
                    Some(MethodMode::Fit) => self.elements_launch_fit(),
                    Some(MethodMode::Pick) => self.elements_pick_fit(),
                    _ => {}
                }
            }
        }
        if create {
            self.elements_create();
        }
        if cancel {
            self.elements_cancel();
        }
    }

    /// What the box asks for next (ScanRuler's `pickHint`).
    fn elements_pick_hint(&self, d: &ElementDraft, method: &me::CreationMethod, paint: bool, marked: usize) -> String {
        let noun = me::kind_info(d.kind).noun;
        if paint {
            return match self.elements.mark_tool {
                MarkTool::Navigate => format!("Pick a marking tool above, then drag over the {noun}."),
                _ if marked == 0 => format!("Drag over the {noun} to mark it."),
                _ => "Keep marking to add more; Shift rubs out.".into(),
            };
        }
        if d.picks.is_empty() && d.status != Status::Ready {
            return method.hint.to_string();
        }
        if method.mode == MethodMode::Pick {
            if d.kind == ElementKind::Point {
                return "Click again to move the point, or create it.".into();
            }
            if d.picks.len() < method.min_picks {
                let left = method.min_picks - d.picks.len();
                return format!("{left} more point{} to go — spread them around the {noun}.", plural(left));
            }
            return "More points refine the fit, or create it.".into();
        }
        format!("Add more points if the {noun} is split across unconnected patches.")
    }

    fn elements_section_box(&mut self, ui: &mut egui::Ui) {
        let frame = self.elements_section_frame();
        let Some(s) = &self.elements.section else { return };
        let color = c32(me::element_color(me::next_element_id(&self.part)));
        let options: Vec<(ElementId, String)> = self.part.elements.iter().filter(|e| me::element_kind(&e.geometry).is_some_and(|k| me::SECTION_REF_KINDS.contains(&k))).map(|e| (e.id, e.name.clone())).collect();
        let along_name = |a: Option<SectionAlong>| match a {
            None => "Choose…".to_string(),
            Some(SectionAlong::World(ax)) => ax.plane_name().to_string(),
            Some(SectionAlong::Element(id)) => options.iter().find(|o| o.0 == id).map_or("a deleted element".into(), |o| o.1.clone()),
        };
        let mut set_along: Option<SectionAlong> = None;
        let mut set_offset: Option<f64> = None;
        let mut create = false;
        let mut cancel = false;
        egui::Frame::group(ui.style()).stroke(Stroke::new(1.0, color)).show(ui, |ui| {
            ui.horizontal(|ui| {
                let (r, _) = ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
                ui.painter().circle_filled(r.center(), 4.5, color);
                ui.label(RichText::new("New section").strong());
                ui.label(RichText::new("ⓘ").weak()).on_hover_text(
                    "A section is the scan cut with a plane: a plane to sketch on, its cut the slice the sketch fits.\n\n\
                     The plane is taken across an element’s direction — a plane’s normal, the axis of a cylinder, cone or line, a circle’s normal — or across one of the scan’s coordinate axes. The plane is moved along its direction by the offset. Zero cuts through the element’s own centre — for a face that is the face itself, so offset into the part to cut through its walls — and for a coordinate plane the offset is the plane’s coordinate on that axis.\n\n\
                     A click on the scan slides the plane, direction kept, to pass through the point clicked.\n\n\
                     The plane is fixed when the section is created. Re-fitting or deleting the element it was cut along leaves the section where it is.",
                );
            });
            ui.horizontal(|ui| {
                ui.label("Cut along");
                egui::ComboBox::from_id_salt("re_section_along").selected_text(along_name(s.along)).show_ui(ui, |ui| {
                    ui.label(RichText::new("Coordinate planes").small().weak());
                    for ax in WorldAxis::ALL {
                        if ui.selectable_label(s.along == Some(SectionAlong::World(ax)), ax.plane_name()).clicked() {
                            set_along = Some(SectionAlong::World(ax));
                        }
                    }
                    if !options.is_empty() {
                        ui.label(RichText::new("Elements").small().weak());
                    }
                    for (id, name) in &options {
                        if ui.selectable_label(s.along == Some(SectionAlong::Element(*id)), name).clicked() {
                            set_along = Some(SectionAlong::Element(*id));
                        }
                    }
                });
            });
            if s.along.is_some() {
                ui.horizontal(|ui| {
                    let label = match s.along {
                        Some(SectionAlong::World(a)) => format!("{a:?} (mm)"),
                        _ => "Offset (mm)".to_string(),
                    };
                    ui.label(label);
                    let mut v = s.offset;
                    if ui.add(egui::DragValue::new(&mut v).speed(0.1).max_decimals(3)).on_hover_text("Where the plane lies along its normal, which is where it cuts the part").changed() {
                        set_offset = Some(v);
                    }
                });
            }
            let hint = if s.along.is_none() {
                "Choose what to cut across — click an element or one of the XY, YZ, XZ planes in the viewport, or pick it above. The plane runs across an element’s direction: a plane’s normal, the axis of a cylinder, cone or line, a circle’s normal."
            } else {
                "Slide the plane along its direction — type the offset, or click the scan where it should cut — then create."
            };
            ui.weak(hint);
            ui.label(RichText::new("Cut").small().weak());
            egui::Frame::NONE.fill(ui.visuals().extreme_bg_color).inner_margin(egui::Margin::symmetric(8, 4)).show(ui, |ui| {
                ui.set_min_width(ui.available_width());
                match (s.along, frame, s.chains) {
                    (None, _, _) => {
                        ui.weak("No element chosen");
                    }
                    (Some(_), None, _) => {
                        ui.colored_label(Color32::from_rgb(0xe2, 0x6b, 0x4f), "The element this was cut along has gone.");
                    }
                    (Some(_), Some(_), Some((_, n))) if n > 0 => {
                        ui.colored_label(color, format!("{n} edge chain{}", plural(n)));
                    }
                    _ => {
                        ui.weak("The plane misses the scan");
                    }
                }
            });
            if ui.add_enabled(frame.is_some(), egui::Button::new(RichText::new("Create section").strong()).min_size(egui::vec2(ui.available_width(), 24.0))).clicked() {
                create = true;
            }
            if ui.button("Cancel").clicked() {
                cancel = true;
            }
        });
        if let Some(a) = set_along {
            // A coordinate plane is taken through the scan's centre, an element through its own.
            let offset = match a {
                SectionAlong::World(ax) => self.scan_centre().map_or(0.0, |c| c.dot(ax.dir())),
                SectionAlong::Element(_) => 0.0,
            };
            if let Some(s) = &mut self.elements.section {
                s.along = Some(a);
                s.offset = offset;
            }
        }
        if let Some(v) = set_offset {
            if let Some(s) = &mut self.elements.section {
                s.offset = v;
            }
        }
        if create {
            self.elements_create_section();
        }
        if cancel {
            self.elements_cancel();
        }
    }

    /// The elements and the sections, each a row: its colour, its name (a double click renames
    /// it), its reading, and its eye and ×.
    fn elements_list(&mut self, ui: &mut egui::Ui) {
        self.elements.row_hover = None;
        let (_, size) = self.elements_model_bounds();
        let elements: Vec<_> = self.part.elements.iter().filter(|e| me::element_kind(&e.geometry).is_some()).cloned().collect();
        let sections: Vec<_> = self.part.elements.iter().filter(|e| me::element_kind(&e.geometry).is_none()).cloned().collect();
        let mut toggle: Option<ElementId> = None;
        let mut delete: Option<ElementId> = None;
        let mut set_all: Option<(bool, bool)> = None;
        let mut rename: Option<(ElementId, String)> = None;
        let mut hover: Option<ElementId> = None;
        for (title, list, is_section) in [("Elements", &elements, false), ("Sections", &sections, true)] {
            if list.is_empty() {
                continue;
            }
            ui.horizontal(|ui| {
                ui.label(RichText::new(title.to_uppercase()).small().strong().weak());
                let any_shown = list.iter().any(|e| !self.elements.hidden.contains(&e.id));
                let what = title.to_lowercase();
                if ui.small_button(if any_shown { "Hide all" } else { "Show all" }).on_hover_text(if any_shown { format!("Hide all {what} in the viewport") } else { format!("Show all {what}") }).clicked() {
                    set_all = Some((is_section, !any_shown));
                }
                ui.label(RichText::new(list.len().to_string()).strong());
            });
            for e in list {
                let shown = !self.elements.hidden.contains(&e.id);
                let r = ui.horizontal(|ui| {
                    let (dot, _) = ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
                    let col = c32(me::element_color(e.id));
                    ui.painter().circle_filled(dot.center(), 4.5, if shown { col } else { col.gamma_multiply(0.35) });
                    let editing = self.elements.renaming.as_ref().filter(|(id, _)| *id == e.id).map(|(_, t)| t.clone());
                    match editing {
                        Some(mut text) => {
                            let r = ui.add(egui::TextEdit::singleline(&mut text).desired_width(120.0));
                            r.request_focus();
                            if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                                self.elements.renaming = None;
                            } else if r.lost_focus() || ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                                rename = Some((e.id, text));
                            } else {
                                self.elements.renaming = Some((e.id, text));
                            }
                        }
                        None => {
                            let mut name = RichText::new(&e.name);
                            if !shown {
                                name = name.weak();
                            }
                            let r = ui.add(egui::Label::new(name).sense(egui::Sense::click())).on_hover_text("Double-click to rename");
                            if r.double_clicked() {
                                self.elements.renaming = Some((e.id, e.name.clone()));
                            }
                        }
                    }
                    let reading = if is_section {
                        let ElementGeometry::Section { frame } = e.geometry else { unreachable!("a section") };
                        let n = self.elements_section_chains(e.id, &frame);
                        format!("{n} edge{}", plural(n))
                    } else {
                        me::format_primary(&me::fit_of(e, size), DisplayUnits::Mm)
                    };
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.small_button("×").on_hover_text(format!("Delete {}", e.name)).clicked() {
                            delete = Some(e.id);
                        }
                        if ui.small_button(if shown { "👁" } else { "–" }).on_hover_text(if shown { "Hide it in the viewport" } else { "Show it in the viewport" }).clicked() {
                            toggle = Some(e.id);
                        }
                        ui.label(RichText::new(reading).strong().monospace());
                    });
                });
                let tip = if is_section {
                    let ElementGeometry::Section { frame } = e.geometry else { unreachable!("a section") };
                    let n = frame.normal;
                    format!("Through ({}, {}, {}), normal ({}, {}, {})", mm_reverse::js_fmt::to_fixed(frame.origin.x, 3), mm_reverse::js_fmt::to_fixed(frame.origin.y, 3), mm_reverse::js_fmt::to_fixed(frame.origin.z, 3), mm_reverse::js_fmt::to_fixed(n.x, 4), mm_reverse::js_fmt::to_fixed(n.y, 4), mm_reverse::js_fmt::to_fixed(n.z, 4))
                } else {
                    me::format_detail(&me::fit_of(e, size), DisplayUnits::Mm)
                };
                let r = r.response.on_hover_text(tip);
                if r.hovered() || r.contains_pointer() {
                    hover = Some(e.id);
                }
            }
        }
        self.elements.row_hover = hover;
        if let Some(id) = toggle {
            if !self.elements.hidden.remove(&id) {
                self.elements.hidden.insert(id);
            }
        }
        if let Some((sections, show)) = set_all {
            let ids: Vec<ElementId> = self.part.elements.iter().filter(|e| me::element_kind(&e.geometry).is_none() == sections).map(|e| e.id).collect();
            for id in ids {
                if show {
                    self.elements.hidden.remove(&id);
                } else {
                    self.elements.hidden.insert(id);
                }
            }
        }
        if let Some((id, text)) = rename {
            self.elements.renaming = None;
            let name = text.trim().to_string();
            if !name.is_empty() && self.part.element(id).is_some_and(|e| e.name != name) {
                self.edit_part(|p| {
                    if let Some(e) = p.elements.iter_mut().find(|e| e.id == id) {
                        e.name = name;
                    }
                });
            }
        }
        if let Some(id) = delete {
            let name = self.part.element(id).map(|e| e.name.clone()).unwrap_or_default();
            self.edit_part(|p| p.elements.retain(|e| e.id != id));
            self.elements.hidden.remove(&id);
            if self.elements.selected.is_some_and(|s| s.0 == id) {
                self.elements.selected = None;
            }
            self.status = format!("{name} deleted — a feature built on it says so; Undo brings it back.");
        }
    }

    // ---- the pointer over the viewport ----

    /// The pointer over the viewport, before the rest of the view takes it: a click on the scan
    /// for the open element or section box, the marking, a face clicked to sketch on, and with no
    /// box open an element picked. Whether the click was taken.
    pub(crate) fn elements_pointer(&mut self, vp: Viewport, pos: Option<Pos2>, clicked: bool, ctx: &egui::Context, mods: egui::Modifiers) -> bool {
        self.elements.box_hint = take_box_hint(ctx);
        self.elements_poll(ctx);
        self.elements_chips(ctx);
        if !matches!(self.mode, Mode::Idle) {
            return false;
        }
        let over_panel = |p: Pos2| self.view.panel_rect.is_some_and(|r| r.contains(p));
        let pos = pos.filter(|p| vp.rect.contains(*p) && !over_panel(*p));
        // The marking.
        let paint = self.elements.select_mode == SelectMode::Paint && self.elements.mark_tool != MarkTool::Navigate && self.elements.draft.as_ref().and_then(|d| me::creation_method(d.kind, d.method)).is_some_and(|m| m.mode == MethodMode::Fit);
        if paint && !self.nav.bindings.starts_navigation(egui::PointerButton::Primary, mods) {
            return self.elements_mark(vp, pos, ctx, mods);
        }
        if !clicked {
            return false;
        }
        let Some(p) = pos else { return false };
        if self.elements.face_pick {
            if let Some((pick, _)) = self.elements_scan_hit(vp, p) {
                self.elements.face_pick = false;
                self.elements_face_fit(pick);
            }
            return true;
        }
        if self.elements.section.as_ref().is_some_and(|s| s.along.is_some()) {
            // A click on the scan slides the plane through the point clicked.
            if let Some((pick, _)) = self.elements_scan_hit(vp, p) {
                let axis = self.elements.section.as_ref().and_then(|s| s.along).and_then(|a| me::section_axis(a, &self.part));
                if let (Some(axis), Some(s)) = (axis, &mut self.elements.section) {
                    s.offset = (pick.point - axis.origin).dot(axis.dir);
                }
            }
            return true;
        }
        if let Some(d) = &self.elements.draft {
            let mode = me::creation_method(d.kind, d.method).map(|m| m.mode);
            if matches!(mode, Some(MethodMode::Fit) | Some(MethodMode::Pick)) {
                if let Some((pick, _)) = self.elements_scan_hit(vp, p) {
                    self.elements_pick(pick);
                }
                return true;
            }
        }
        // An element (or a section) picked: its chips by the click. A sketch's region in front
        // takes the click first, as the sketches stand in front of the elements.
        if self.elements_region_in_front(vp, p) {
            self.elements.selected = None;
            return false;
        }
        match self.element_at(vp, p) {
            Some(id) => {
                if let Some(e) = self.elements.section.as_mut().filter(|s| s.along.is_none()) {
                    e.along = Some(SectionAlong::Element(id));
                }
                self.elements_select(id, p);
                true
            }
            None => {
                self.elements.selected = None;
                false
            }
        }
    }

    /// An element (or a section) picked on the part at a screen point: it is lit and its chips
    /// stand by the click. The one place an element becomes the selection — where the
    /// workspace's own selection (a pick list with chips for every kind of thing) takes it, if
    /// there is one.
    pub(crate) fn elements_select(&mut self, id: ElementId, at: Pos2) {
        self.elements.selected = Some((id, at));
    }

    /// Whether a region of a sketch shown on the part lies under the point.
    fn elements_region_in_front(&self, vp: Viewport, p: Pos2) -> bool {
        self.part.features.iter().filter(|f| f.visible() && !f.suppressed()).any(|f| {
            let FeatureKind::Sketch(_) = &f.kind else { return false };
            let Some(frame) = self.sketch_frame(f.id) else { return false };
            let Some(sheet) = self.sheet_point(vp, p, &frame) else { return false };
            self.built.sketches.get(&f.id).is_some_and(|b| mm_reverse::regions::region_at(&b.regions, sheet).is_some())
        })
    }

    /// The element or section under a screen point: within reach of its lines, or on its surface
    /// — in front of the scan.
    pub(crate) fn element_at(&mut self, vp: Viewport, p: Pos2) -> Option<ElementId> {
        let ray = self.view.camera.ray_from_ndc(vp.ndc(p), vp.aspect());
        let (o, d) = (self.to_part(ray.origin), self.to_part(ray.dir));
        let (_, size) = self.elements_model_bounds();
        let wall = self.elements_scan_hit(vp, p).map_or(f64::INFINITY, |(_, t)| t + (size * 0.01).max(0.5));
        let mut best: Option<(ElementId, f64)> = None;
        if !self.elements_drawn() {
            return None;
        }
        for e in self.part.elements.iter().filter(|e| !self.elements.hidden.contains(&e.id)) {
            let mut t_hit: Option<f64> = None;
            for w in self.elements_wires_of(e, size).iter() {
                for s in w.windows(2) {
                    let (Some(a), Some(b)) = (self.project(vp, s[0]), self.project(vp, s[1])) else { continue };
                    let (dist, t) = seg_dist(p, a, b);
                    if dist <= ELEMENT_REACH_PX {
                        let q = s[0] + (s[1] - s[0]) * t as f64;
                        let along = (q - o).dot(d);
                        t_hit = Some(t_hit.map_or(along, |x: f64| x.min(along)));
                    }
                }
            }
            if let ElementGeometry::Point { center } = e.geometry {
                if self.project(vp, center).is_some_and(|q| (q - p).length() <= ELEMENT_REACH_PX + 3.0) {
                    t_hit = Some((center - o).dot(d));
                }
            }
            // A section is picked by its cut alone: its plane is not drawn.
            let (pts, tris) = if me::element_kind(&e.geometry).is_some() { me::element_surface(&e.geometry, size) } else { (Vec::new(), Vec::new()) };
            for tri in &tris {
                if let Some(t) = ray_triangle(o, d, pts[tri[0] as usize], pts[tri[1] as usize], pts[tri[2] as usize]) {
                    t_hit = Some(t_hit.map_or(t, |x| x.min(t)));
                }
            }
            if let Some(t) = t_hit.filter(|t| *t <= wall) {
                if best.is_none_or(|(_, bt)| t < bt) {
                    best = Some((e.id, t));
                }
            }
        }
        best.map(|b| b.0)
    }

    /// The marking under way: a window closed on release, the brush as it is dragged (Shift rubs
    /// out). Whether the pointer was taken.
    fn elements_mark(&mut self, vp: Viewport, pos: Option<Pos2>, ctx: &egui::Context, mods: egui::Modifiers) -> bool {
        let (pressed, down, released) = ctx.input(|i| (i.pointer.button_pressed(egui::PointerButton::Primary), i.pointer.button_down(egui::PointerButton::Primary), i.pointer.button_released(egui::PointerButton::Primary)));
        let erase = mods.shift;
        let mut changed = false;
        match self.elements.mark_tool {
            MarkTool::Window => {
                if pressed {
                    self.elements.window_from = pos;
                }
                if released {
                    if let (Some(a), Some(b)) = (self.elements.window_from.take(), pos.or(self.view.panel_rect.map(|_| ctx.input(|i| i.pointer.interact_pos().unwrap_or_default())))) {
                        let r = egui::Rect::from_two_pos(a, b);
                        if r.width() + r.height() > 4.0 {
                            changed = self.elements_mark_where(vp, erase, true, |q, _| r.contains(q));
                        }
                    }
                }
            }
            MarkTool::Brush => {
                if let (true, Some(p)) = (down, pos) {
                    if let Some((pick, _)) = self.elements_scan_hit(vp, p) {
                        let centre = pick.point;
                        // The brush's radius on the scan, from its radius on the screen there.
                        let side = self.to_part(self.view.camera.ray_from_ndc(vp.ndc(p + egui::vec2(BRUSH_PX, 0.0)), vp.aspect()).origin);
                        let o = self.to_part(self.view.camera.ray_from_ndc(vp.ndc(p), vp.aspect()).origin);
                        let eye = self.to_part(self.view.camera.eye());
                        let scale = (centre - eye).length() / (o - eye).length().max(1e-9);
                        let radius = ((side - o).length() * scale).max(1e-3);
                        changed = self.elements_mark_where(vp, erase, false, |_, q3| (q3 - centre).length() <= radius);
                    }
                }
            }
            MarkTool::Navigate => return false,
        }
        if changed {
            self.elements_launch_fit();
        }
        true
    }

    /// Mark (or rub out) the scan's vertices facing the eye that pass a test on their screen and
    /// part positions — with `seen`, only those the eye sees (none hidden behind the scan).
    fn elements_mark_where(&mut self, vp: Viewport, erase: bool, seen: bool, take: impl Fn(Pos2, DVec3) -> bool) -> bool {
        let eye = self.view.camera.eye();
        let (_, size) = self.elements_model_bounds();
        let slack = (size * 0.002).max(0.05);
        let mut changed = false;
        let objects: Vec<super::ScanObject> = self.scan.iter().filter(|o| o.visible).cloned().collect();
        for o in &objects {
            let graph = self.elements.graphs.get(&o.id).map(|g| g.1.clone());
            let mut found = Vec::new();
            for v in o.mesh.vertex_ids() {
                let at = o.mesh.positions()[v as usize];
                let n = match &graph {
                    Some(g) => g.normals.get(v as usize).copied().unwrap_or(DVec3::ZERO),
                    None => o.mesh.vertex_normal(v),
                };
                if n.dot(eye - at) <= 0.0 {
                    continue;
                }
                let part = self.to_part(at);
                let Some(q) = self.project(vp, part) else { continue };
                if take(q, part) {
                    found.push((v, at));
                }
            }
            if seen && !found.is_empty() {
                // Seen from the eye: nothing of the scan between.
                let bvhs: Vec<(Arc<mm_core::bvh::Bvh>, super::ScanObject)> = objects.iter().filter_map(|x| self.scan_bvh(x).map(|b| (b, x.clone()))).collect();
                found.retain(|(_, at)| {
                    let d = *at - eye;
                    let far = d.length();
                    let ray = mm_core::geom::Ray::new(eye, d);
                    !bvhs.iter().any(|(b, x)| b.raycast(&x.mesh, &ray).is_some_and(|h| h.t < far - slack))
                });
            }
            let found: Vec<u32> = found.into_iter().map(|f| f.0).collect();
            if found.is_empty() {
                continue;
            }
            let set = self.elements.marked.entry(o.id).or_default();
            for v in found {
                changed |= if erase { set.remove(&v) } else { set.insert(v) };
            }
        }
        self.elements.marked.retain(|_, s| !s.is_empty());
        changed
    }

    /// The chips by an element picked on the part (ScanRuler's `chipsFor`, the element's case):
    /// a revolve about its axis, a hole along a cylinder or a circle, a sketch on a plane, a body
    /// at its extents.
    fn elements_chips(&mut self, ctx: &egui::Context) {
        let Some((id, at)) = self.elements.selected else { return };
        if !matches!(self.mode, Mode::Idle) {
            self.elements.selected = None;
            return;
        }
        let Some(e) = self.part.element(id).cloned() else {
            self.elements.selected = None;
            return;
        };
        let name = e.name.clone();
        let has_body = !self.bodies.is_empty();
        let mut chips: Vec<(&str, String, ChipAct)> = Vec::new();
        match e.geometry {
            ElementGeometry::Line { .. } | ElementGeometry::Cylinder { .. } | ElementGeometry::Cone { .. } | ElementGeometry::Circle { .. } => {
                chips.push(("Revolve about this", format!("Revolve the last sketch's profile about {name}"), ChipAct::RevolveAbout(id)));
            }
            _ => {}
        }
        if has_body {
            if let ElementGeometry::Cylinder { radius, .. } | ElementGeometry::Circle { radius, .. } = e.geometry {
                chips.push(("Hole", format!("Drill a hole through the body along {name}"), ChipAct::Hole(id, radius)));
            }
        }
        match e.geometry {
            ElementGeometry::Plane { .. } => chips.push(("Sketch", format!("Start a sketch on {name}"), ChipAct::SketchOn(PlaneRef::Plane { id, offset: None, up: None }))),
            ElementGeometry::Section { .. } => chips.push(("Sketch", format!("Start a sketch on {name}"), ChipAct::SketchOn(PlaneRef::Section { id, offset: None, up: None }))),
            _ => {}
        }
        if matches!(e.geometry, ElementGeometry::Plane { .. } | ElementGeometry::Cylinder { .. } | ElementGeometry::Cone { .. } | ElementGeometry::Sphere { .. } | ElementGeometry::Torus { .. }) {
            chips.push(("Extract as body", format!("{name} as a solid at its extents, extended as in 3D Measure"), ChipAct::Extract(id)));
        }
        let mut run: Option<ChipAct> = None;
        egui::Area::new(egui::Id::NULL.with("re_element_chips")).fixed_pos(at + egui::vec2(12.0, 12.0)).order(egui::Order::Foreground).constrain(true).fade_in(false).show(ctx, |ui| {
            egui::Frame::popup(ui.style()).show(ui, |ui| {
                ui.horizontal(|ui| {
                    let (dot, _) = ui.allocate_exact_size(egui::vec2(9.0, 9.0), egui::Sense::hover());
                    ui.painter().circle_filled(dot.center(), 4.5, c32(me::element_color(id)));
                    ui.label(RichText::new(&name).strong());
                });
                if chips.is_empty() {
                    ui.weak("Nothing is built on this element here.");
                }
                ui.horizontal_wrapped(|ui| {
                    for (label, title, act) in &chips {
                        if ui.button(*label).on_hover_text(title).clicked() {
                            run = Some(act.clone());
                        }
                    }
                });
            });
        });
        if let Some(act) = run {
            self.elements.selected = None;
            self.elements_run_chip(act);
        }
    }

    fn elements_run_chip(&mut self, act: ChipAct) {
        match act {
            ChipAct::SketchOn(plane) => self.elements_sketch_on(plane),
            ChipAct::RevolveAbout(id) => {
                self.new_revolve(None, Vec::new());
                if let Mode::Feature(edit) = &mut self.mode {
                    if let FeatureKind::Revolve(r) = &mut edit.kind {
                        r.axis = AxisRef::Element { id };
                    }
                }
                self.request_build();
            }
            ChipAct::Hole(id, radius) => {
                let Some(body) = self.bodies.last().map(|b| b.id) else { return };
                self.open_feature(FeatureKind::Hole(HoleFeature { body, axis: HoleAxis::Element { id }, radius, depth: None, insert: None, exprs: Default::default() }), None);
            }
            ChipAct::Extract(id) => {
                self.open_feature(FeatureKind::Extract(ExtractFeature { element: Some(id), section: None, thickness: 2.0, op: BodyOp::New, target: None, exprs: Default::default() }), None);
            }
        }
    }

    // ---- drawing ----

    /// The plane the hint shows, its colour, its name and how see-through its sheet is: the list's
    /// option under the pointer, else the open sketch's plane.
    fn elements_hint_plane(&self) -> Option<(SectionFrame, Color32, String, f32)> {
        if let Some(Hint::Plane { plane, name }) = &self.elements.hint {
            return resolve_plane_frame(plane, &self.part).map(|f| (f, HINT, name.clone(), 0.18));
        }
        if let Some(BoxHint::Plane { frame, name, .. }) = &self.elements.box_hint {
            return Some((*frame, HINT, name.clone(), 0.18));
        }
        if let Mode::Sketch(edit) = &self.mode {
            let name = edit.feature.and_then(|id| self.part.feature(id)).map_or_else(|| format!("Sketch {}", self.part.next_of_kind.get("sketch").copied().unwrap_or(0) + 1), |f| f.name.clone());
            return Some((edit.frame, SKETCH_BLUE, name, 0.06));
        }
        None
    }

    /// The see-through surfaces of the elements, of the box's preview, of the open section and of
    /// the hinted plane, as meshes in the scene (made again only when they change).
    pub(crate) fn elements_surfaces(&mut self) -> Vec<(u64, Arc<Mesh>, Vec4)> {
        let (centre, size) = self.elements_model_bounds();
        let flip = self.flip as u8 as f64;
        let mut want: Vec<(u64, ElementGeometry, Vec4)> = Vec::new();
        let hinted = match (&self.elements.hint, &self.elements.box_hint) {
            (Some(Hint::Plane { plane: PlaneRef::Plane { id, .. } | PlaneRef::Section { id, .. }, .. }), _) => Some(*id),
            (Some(Hint::Axis { axis: RadialAxis::Element { id }, .. }), _) => Some(*id),
            (None, Some(BoxHint::Plane { element, .. } | BoxHint::Axis { element, .. })) => *element,
            _ => None,
        };
        self.elements_refresh_cuts();
        if self.elements_drawn() {
            for e in self.part.elements.iter().filter(|e| !self.elements.hidden.contains(&e.id) && me::element_kind(&e.geometry).is_some()) {
                let lit = hinted == Some(e.id) || self.elements.row_hover == Some(e.id) || self.elements.selected.is_some_and(|s| s.0 == e.id);
                let alpha = if matches!(e.geometry, ElementGeometry::Section { .. }) { 0.06 } else { 0.22 };
                let color = if hinted == Some(e.id) { v4(HINT, 0.35) } else { v4(c32(me::element_color(e.id)), if lit { alpha + 0.2 } else { alpha }) };
                want.push((e.id as u64, e.geometry.clone(), color));
            }
        }
        let next_color = c32(me::element_color(me::next_element_id(&self.part)));
        if let Some(f) = self.elements.draft.as_ref().and_then(|d| d.fit.as_ref()) {
            want.push((KEY_DRAFT, me::geometry_of(f).0, v4(next_color, 0.3)));
        }
        if let Some(frame) = self.elements.section.as_ref().and_then(|s| s.along.map(|a| (a, s.offset))).and_then(|(a, o)| me::section_frame(a, o, &self.part)) {
            want.push((KEY_SECTION, ElementGeometry::Section { frame }, v4(next_color, 0.12)));
        }
        if let Some((frame, color, _, alpha)) = self.elements_hint_plane() {
            let corners = sheet_corners(&frame, centre, size * 1.15);
            let key = if color == HINT { KEY_HINT } else { KEY_SKETCH };
            let sig = signature(&[corners[0].x, corners[0].y, corners[0].z, corners[2].x, corners[2].y, corners[2].z, flip]);
            let mesh = match self.elements.meshes.get(&key) {
                Some((s, m)) if *s == sig => m.clone(),
                _ => {
                    let pts: Vec<DVec3> = corners.iter().map(|p| self.to_scene(*p)).collect();
                    let (m, _) = Mesh::from_triangles(&pts, &[[0, 1, 2], [0, 2, 3]], None);
                    let m = Arc::new(m);
                    self.elements.meshes.insert(key, (sig, m.clone()));
                    m
                }
            };
            let mut out = self.elements_meshes_for(&want, size, flip);
            out.push((key, mesh, v4(color, alpha)));
            return out;
        }
        self.elements_meshes_for(&want, size, flip)
    }

    fn elements_meshes_for(&mut self, want: &[(u64, ElementGeometry, Vec4)], size: f64, flip: f64) -> Vec<(u64, Arc<Mesh>, Vec4)> {
        let mut out = Vec::new();
        let keep: HashSet<u64> = want.iter().map(|w| w.0).chain([KEY_HINT, KEY_SKETCH]).collect();
        self.elements.meshes.retain(|k, _| keep.contains(k));
        for (key, g, color) in want {
            let sig = geometry_signature(g, &[size, flip]);
            let mesh = match self.elements.meshes.get(key) {
                Some((s, m)) if *s == sig => Some(m.clone()),
                _ => {
                    let (pts, tris) = me::element_surface(g, size);
                    if tris.is_empty() {
                        None
                    } else {
                        let scene: Vec<DVec3> = pts.iter().map(|p| self.to_scene(*p)).collect();
                        let (m, _) = Mesh::from_triangles(&scene, &tris, None);
                        let m = Arc::new(m);
                        self.elements.meshes.insert(*key, (sig, m.clone()));
                        Some(m)
                    }
                }
            };
            if let Some(m) = mesh {
                out.push((*key, m, *color));
            }
        }
        out
    }

    /// The lines an element is drawn and picked by: a section's cut through the scan (its plane's
    /// square where the scan is not cut), any other element's own.
    fn elements_wires_of(&self, e: &mm_reverse::model::Element, size: f64) -> Arc<Vec<Vec<DVec3>>> {
        if let ElementGeometry::Section { frame } = e.geometry {
            if let Some((_, f, cut)) = self.elements.section_cuts.get(&e.id) {
                if *f == frame && !cut.is_empty() {
                    return cut.clone();
                }
            }
            return Arc::new(me::element_wires(&e.geometry, size * 0.6));
        }
        Arc::new(me::element_wires(&e.geometry, size))
    }

    /// Cut every section shown through the scan again where the scan or the section changed.
    fn elements_refresh_cuts(&mut self) {
        if self.scan.is_empty() {
            self.elements.section_cuts.clear();
            return;
        }
        let key = self.scan_parts().len() as u64 ^ self.scan_parts.as_ref().map_or(0, |s| s.0);
        let sections: Vec<(ElementId, SectionFrame)> = self.part.elements.iter().filter_map(|e| match e.geometry {
            ElementGeometry::Section { frame } => Some((e.id, frame)),
            _ => None,
        }).collect();
        self.elements.section_cuts.retain(|id, _| sections.iter().any(|s| s.0 == *id));
        for (id, frame) in sections {
            if self.elements.section_cuts.get(&id).is_some_and(|(k, f, _)| *k == key && *f == frame) {
                continue;
            }
            let cut: Vec<Vec<DVec3>> = self.slice(&frame).into_iter().map(|c| c.into_iter().map(|p| frame.to_world(p)).collect()).collect();
            self.elements.section_counts.insert(id, (key, cut.len()));
            self.elements.section_cuts.insert(id, (key, frame, Arc::new(cut)));
        }
    }

    /// The elements' lines and the box's preview, into the view's lines (scene coordinates,
    /// depth-tested).
    pub(crate) fn element_lines(&self, lines: &mut Vec<ColorVertex>) {
        let (_, size) = self.elements_model_bounds();
        let mut add = |w: &[DVec3], c: Vec4| {
            for s in w.windows(2) {
                lines.push(ColorVertex::new(self.to_scene(s[0]).as_vec3(), c));
                lines.push(ColorVertex::new(self.to_scene(s[1]).as_vec3(), c));
            }
        };
        if self.elements_drawn() {
            for e in self.part.elements.iter().filter(|e| !self.elements.hidden.contains(&e.id)) {
                let lit = self.elements.row_hover == Some(e.id) || self.elements.selected.is_some_and(|s| s.0 == e.id);
                let c = if lit { Vec4::new(1.0, 1.0, 1.0, 1.0) } else { v4(c32(me::element_color(e.id)), 0.95) };
                for w in self.elements_wires_of(e, size).iter() {
                    add(w, c);
                }
            }
        }
        let next = v4(c32(me::element_color(me::next_element_id(&self.part))), 1.0);
        if let Some(f) = self.elements.draft.as_ref().and_then(|d| d.fit.as_ref()) {
            for w in me::element_wires(&me::geometry_of(f).0, size) {
                add(&w, next);
            }
        }
        if let Some(frame) = self.elements.section.as_ref().and_then(|s| s.along.map(|a| (a, s.offset))).and_then(|(a, o)| me::section_frame(a, o, &self.part)) {
            for w in me::element_wires(&ElementGeometry::Section { frame }, size) {
                add(&w, next);
            }
        }
    }

    /// Painted over the view: the elements' names, the picks, the region measured and the
    /// marking, the gesture under way, and the hinted plane's border and name or axis (on top).
    pub(crate) fn paint_elements(&self, painter: &egui::Painter, vp: Viewport) {
        let (centre, size) = self.elements_model_bounds();
        let font = egui::FontId::proportional(11.0);
        if self.elements_drawn() {
            for e in self.part.elements.iter().filter(|e| !self.elements.hidden.contains(&e.id)) {
                let col = c32(me::element_color(e.id));
                if let ElementGeometry::Point { center } = e.geometry {
                    if let Some(q) = self.project(vp, center) {
                        painter.circle_filled(q, 4.0, col);
                    }
                }
                let anchor = match self.elements.section_cuts.get(&e.id).and_then(|(_, _, c)| c.first().and_then(|c| c.first().copied())) {
                    Some(p) => p,
                    None => me::element_anchor(&e.geometry),
                };
                if let Some(q) = self.project(vp, anchor) {
                    painter.text(q + egui::vec2(6.0, -6.0), egui::Align2::LEFT_BOTTOM, &e.name, font.clone(), col);
                }
            }
        }
        let next = c32(me::element_color(me::next_element_id(&self.part)));
        if let Some(d) = &self.elements.draft {
            if let Some((object, region)) = &d.region {
                if let Some(o) = self.scan.iter().find(|o| o.id == *object) {
                    let step = (region.len() / 6000).max(1);
                    for v in region.iter().step_by(step) {
                        if let Some(q) = o.mesh.positions().get(*v as usize).and_then(|p| self.project(vp, self.to_part(*p))) {
                            painter.circle_filled(q, 1.4, next.gamma_multiply(0.8));
                        }
                    }
                }
            }
            for p in &d.picks {
                if let Some(q) = self.project(vp, p.point) {
                    painter.circle_filled(q, 4.0, next);
                    painter.circle_stroke(q, 4.0, Stroke::new(1.0, Color32::WHITE));
                }
            }
        }
        for (object, set) in &self.elements.marked {
            let Some(o) = self.scan.iter().find(|o| o.id == *object) else { continue };
            let step = (set.len() / 8000).max(1);
            for v in set.iter().step_by(step) {
                if let Some(q) = o.mesh.positions().get(*v as usize).and_then(|p| self.project(vp, self.to_part(*p))) {
                    painter.circle_filled(q, 1.6, MARKED);
                }
            }
        }
        if let (Some(a), Some(b)) = (self.elements.window_from, painter.ctx().input(|i| i.pointer.hover_pos())) {
            painter.rect_stroke(egui::Rect::from_two_pos(a, b), 0.0, Stroke::new(1.0, MARKED), egui::StrokeKind::Middle);
        }
        let painting = self.elements.select_mode == SelectMode::Paint && self.elements.mark_tool == MarkTool::Brush && self.elements.draft.is_some();
        if let (true, Some(p)) = (painting, painter.ctx().input(|i| i.pointer.hover_pos()).filter(|p| vp.rect.contains(*p))) {
            painter.circle_stroke(p, BRUSH_PX, Stroke::new(1.0, MARKED));
        }
        // The plane or axis hinted, on top.
        if let Some((frame, color, name, _)) = self.elements_hint_plane() {
            let corners = sheet_corners(&frame, centre, size * 1.15);
            let pts: Vec<Pos2> = corners.iter().chain(std::iter::once(&corners[0])).filter_map(|p| self.project(vp, *p)).collect();
            if pts.len() == 5 {
                painter.add(egui::Shape::line(pts, Stroke::new(2.5, color.gamma_multiply(0.9))));
            }
            let mid = centre - frame.normal * (centre - frame.origin).dot(frame.normal);
            if let Some(q) = self.project(vp, mid) {
                pin_label(painter, q, &name, color);
            }
        }
        let axis_hint = match (&self.elements.hint, &self.elements.box_hint) {
            (Some(Hint::Axis { axis, name }), _) => hint_axis(*axis, self).map(|(o, d)| (o, d, name)),
            (Some(Hint::Plane { .. }), _) => None,
            (None, Some(BoxHint::Axis { origin, dir, name, .. })) => Some((*origin, *dir, name)),
            (None, _) => None,
        };
        if let Some((o, d, name)) = axis_hint {
            if let Some(d) = d.try_normalize() {
                let foot = o + d * (centre - o).dot(d);
                let reach = size * 1.3;
                let (a, b) = (foot - d * reach, foot + d * reach);
                if let (Some(qa), Some(qb)) = (self.project(vp, a), self.project(vp, b)) {
                    painter.line_segment([qa, qb], Stroke::new(3.0, HINT.gamma_multiply(0.95)));
                    pin_label(painter, qa, name, HINT);
                }
            }
        }
    }

    // ---- script steps (headless tests) ----

    /// The scan point nearest a part point: its object, the triangle at its vertex, the point.
    fn elements_pick_near(&self, at: DVec3) -> Option<Pick> {
        let target = self.to_scene(at);
        let mut best: Option<(f64, Pick)> = None;
        for o in self.scan.iter().filter(|o| o.visible) {
            for v in o.mesh.vertex_ids() {
                let p = o.mesh.positions()[v as usize];
                let dd = (p - target).length_squared();
                if best.as_ref().is_none_or(|b| dd < b.0) {
                    let Some(t) = o.mesh.vertex_triangles(v).first().copied() else { continue };
                    best = Some((dd, Pick { object: o.id, tri: o.mesh.triangle(t), point: self.to_part(p) }));
                }
            }
        }
        best.map(|b| b.1)
    }

    /// The Reverse Engineer script's steps for the elements and the planes. Whether the step was
    /// one of these:
    /// * `element:<kind>:<x>:<y>:<z>[:<x>:<y>:<z>…]` — a surface fitted where the scan is nearest
    ///   the point (a point picked there, a circle through three or more), created at once;
    /// * `construct:<kind>:<method>:<element ids, comma>:<numbers, comma>` — a construction;
    /// * `elementbox:<kind>` — the box opened on a kind; `section:<x|y|z|element id>:<offset>`;
    /// * `sketchon:<offer key or label>` — a sketch on a plane of the Sketch on list;
    /// * `sketchradial:<x|y|z|element id>:<angle>`, `sketchface:<x>:<y>:<z>`, `sketchup:<x|y|z|own>`,
    ///   `sketchangle:<degrees>`;
    /// * `hint:<offer key or label>` (empty clears it), `selectelement:<id>`, `elementhide:<id>`;
    /// * `elementsreport` — the elements to the log.
    pub(crate) fn plane_script_step(&mut self, cmd: &str, args: &[&str]) -> bool {
        let num = |i: usize| args.get(i).and_then(|v| v.parse::<f64>().ok());
        let kind_of = |s: Option<&&str>| me::ELEMENT_KINDS.iter().find(|k| s.is_some_and(|s| k.noun == *s || k.label.eq_ignore_ascii_case(s))).map(|k| k.kind);
        let along_of = |s: Option<&&str>| match s.copied() {
            Some("x") => Some(SectionAlong::World(WorldAxis::X)),
            Some("y") => Some(SectionAlong::World(WorldAxis::Y)),
            Some("z") => Some(SectionAlong::World(WorldAxis::Z)),
            Some(id) => id.parse::<ElementId>().ok().map(SectionAlong::Element),
            None => None,
        };
        let find_offer = |ws: &Self, key: &str| ws.elements_sketch_offers().into_iter().find(|o| o.key == key || o.label.eq_ignore_ascii_case(key));
        match cmd {
            "element" => {
                let Some(kind) = kind_of(args.first()) else {
                    log::warn!("element step: no such kind {:?}", args.first());
                    return true;
                };
                let points: Vec<DVec3> = args[1..].chunks(3).filter_map(|c| (c.len() == 3).then(|| DVec3::new(c[0].parse().unwrap_or(0.0), c[1].parse().unwrap_or(0.0), c[2].parse().unwrap_or(0.0)))).collect();
                let picks: Vec<Pick> = points.iter().filter_map(|p| self.elements_pick_near(*p)).collect();
                if picks.is_empty() {
                    self.status = "element: no scan near the point".into();
                    return true;
                }
                self.elements_start(kind);
                let fit = match kind {
                    ElementKind::Point => Ok(me::picked_point(picks[0].point)),
                    ElementKind::Circle => me::circle_through(&picks.iter().map(|p| p.point).collect::<Vec<_>>()),
                    ElementKind::Line => Err("A line is constructed: use construct.".into()),
                    _ => self.elements_graph_now(picks[0].object).and_then(|g| {
                        let seeds: Vec<u32> = picks.iter().filter(|p| p.object == picks[0].object).flat_map(|p| p.tri).collect();
                        me::fit_from_seeds(&g, kind, &seeds, &FitSettings { sigma: self.elements.sigma, ..FitSettings::default() }).map(|out| {
                            let flip = self.flip;
                            me::map_fit(&out.fit, |p| if flip { crate::app::flip_zy_out(p) } else { p })
                        })
                    }),
                };
                match fit {
                    Ok(f) => {
                        if let Some(d) = &mut self.elements.draft {
                            d.fit = Some(f);
                            d.picks = picks;
                            d.status = Status::Ready;
                        }
                        self.elements_create();
                        let said = std::mem::take(&mut self.status);
                        self.elements_cancel();
                        self.status = said;
                    }
                    Err(e) => {
                        self.elements_cancel();
                        self.status = e;
                    }
                }
                log::info!("reverse element: {}", self.status);
            }
            "construct" => {
                let (Some(kind), Some(method)) = (kind_of(args.first()), args.get(1).copied()) else { return true };
                let Some(m) = me::creation_method(kind, method) else {
                    self.status = format!("construct: no method {method}");
                    return true;
                };
                self.elements.draft = Some(self.elements_fresh_draft(kind, m.id));
                let refs: Vec<Option<ElementId>> = args.get(2).map_or(Vec::new(), |s| s.split(',').filter(|x| !x.is_empty()).map(|x| x.parse().ok()).collect());
                let params: Vec<f64> = args.get(3).map_or(Vec::new(), |s| s.split(',').filter(|x| !x.is_empty()).map(|x| x.parse().unwrap_or(f64::NAN)).collect());
                if let Some(d) = &mut self.elements.draft {
                    for (i, r) in refs.into_iter().enumerate() {
                        if let Some(slot) = d.refs.get_mut(i) {
                            *slot = r;
                        }
                    }
                    for (i, v) in params.into_iter().enumerate() {
                        if let Some(p) = d.params.get_mut(i) {
                            *p = v;
                        }
                    }
                }
                if m.id == "point-centroid" {
                    self.elements_measure_centroid();
                }
                self.elements_evaluate();
                let failed = self.elements.draft.as_ref().filter(|d| d.status != Status::Ready).map(|d| d.message.clone().unwrap_or_else(|| "Incomplete".into()));
                match failed {
                    Some(why) => {
                        self.elements_cancel();
                        self.status = why;
                    }
                    None => {
                        self.elements_create();
                        let said = std::mem::take(&mut self.status);
                        self.elements_cancel();
                        self.status = said;
                    }
                }
                log::info!("reverse construct: {}", self.status);
            }
            "elementbox" => {
                if let Some(kind) = kind_of(args.first()) {
                    self.elements_start(kind);
                }
            }
            // elementmark:<window|brush|click> — how the open box's surface is chosen.
            "elementmark" => {
                let (mode, tool) = match args.first().copied() {
                    Some("window") => (SelectMode::Paint, MarkTool::Window),
                    Some("brush") => (SelectMode::Paint, MarkTool::Brush),
                    _ => (SelectMode::Auto, MarkTool::Navigate),
                };
                self.elements.select_mode = mode;
                self.elements.mark_tool = tool;
                self.elements.marked.clear();
            }
            "elementcreate" => self.elements_create(),
            "section" => {
                let Some(along) = along_of(args.first()) else { return true };
                self.elements_start_section();
                if let Some(s) = &mut self.elements.section {
                    s.along = Some(along);
                    s.offset = num(1).unwrap_or(0.0);
                }
                if args.get(2) != Some(&"open") {
                    self.elements_create_section();
                }
                log::info!("reverse section: {}", self.status);
            }
            "sketchon" => {
                let key = args.join(":");
                match find_offer(self, &key) {
                    Some(o) => self.elements_sketch_on(o.plane),
                    None => self.status = format!("sketchon: no plane {key} on offer"),
                }
                log::info!("reverse sketchon: {}", self.status);
            }
            "sketchradial" => {
                let axis = match args.first().copied() {
                    Some("x") => RadialAxis::World { axis: WorldAxis::X },
                    Some("y") => RadialAxis::World { axis: WorldAxis::Y },
                    Some("z") | None => RadialAxis::World { axis: WorldAxis::Z },
                    Some(id) => RadialAxis::Element { id: id.parse().unwrap_or(0) },
                };
                self.elements_sketch_on(PlaneRef::Radial { axis, angle_deg: num(1).unwrap_or(0.0), offset: None });
                log::info!("reverse sketchradial: {}", self.status);
            }
            "sketchangle" => {
                if let Mode::Sketch(edit) = &self.mode {
                    if let PlaneRef::Radial { axis, offset, .. } = edit.plane.clone() {
                        self.elements_set_sketch_plane(PlaneRef::Radial { axis, angle_deg: num(0).unwrap_or(0.0), offset });
                    }
                }
            }
            "sketchup" => {
                if let Mode::Sketch(edit) = &self.mode {
                    let a = match args.first().copied() {
                        Some("x") => Some(WorldAxis::X),
                        Some("y") => Some(WorldAxis::Y),
                        Some("z") => Some(WorldAxis::Z),
                        _ => None,
                    };
                    let mut p = edit.plane.clone();
                    if let PlaneRef::World { up, .. } | PlaneRef::Section { up, .. } | PlaneRef::Plane { up, .. } = &mut p {
                        *up = a;
                    }
                    self.elements_set_sketch_plane(p);
                }
            }
            "sketchface" => {
                let at = DVec3::new(num(0).unwrap_or(0.0), num(1).unwrap_or(0.0), num(2).unwrap_or(0.0));
                let Some(pick) = self.elements_pick_near(at) else { return true };
                // The fit comes in the scene's axes, as a worker's does.
                let fit = self.elements_graph_now(pick.object).and_then(|g| me::fit_from_seeds(&g, ElementKind::Plane, &pick.tri, &FitSettings::default()));
                self.elements_fit_landed(JobFor::Face, pick.object, fit);
                log::info!("reverse sketchface: {}", self.status);
            }
            "hint" => {
                let key = args.join(":");
                self.elements.hint = find_offer(self, &key).map(|o| match o.plane {
                    PlaneRef::Radial { axis, .. } => Hint::Axis { axis, name: o.label.trim_start_matches("Through ").to_string() },
                    p => Hint::Plane { plane: p, name: o.label },
                });
            }
            "selectelement" => {
                let id = num(0).map(|v| v as ElementId);
                let at = self.view.panel_rect.map_or(Pos2::new(700.0, 300.0), |r| r.right_top() + egui::vec2(80.0, 120.0));
                self.elements.selected = id.filter(|id| self.part.element(*id).is_some()).map(|id| (id, at));
            }
            "elementhide" => {
                if let Some(id) = num(0).map(|v| v as ElementId) {
                    self.elements.hidden.insert(id);
                }
            }
            "elementsreport" => {
                let (_, size) = self.elements_model_bounds();
                let rows: Vec<String> = self.part.elements.iter().map(|e| format!("{} {} [{}] {}", e.id, e.name, e.geometry.kind(), me::format_detail(&me::fit_of(e, size), DisplayUnits::Mm))).collect();
                let plane = match &self.mode {
                    Mode::Sketch(edit) => format!("sketch on {} normal ({:.3}, {:.3}, {:.3}) u ({:.3}, {:.3}, {:.3}), {} chains", mm_reverse::build::describe_plane_ref(&edit.plane, &self.part), edit.frame.normal.x, edit.frame.normal.y, edit.frame.normal.z, edit.frame.u.x, edit.frame.u.y, edit.frame.u.z, edit.chains.len()),
                    _ => "no sketch open".into(),
                };
                let (c, r) = self.elements_model_bounds();
                log::info!("reverse elements: model centre ({:.2}, {:.2}, {:.2}) radius {:.2}; [{}]; {}; status: {}", c.x, c.y, c.z, r, rows.join("; "), plane, self.status);
            }
            _ => return false,
        }
        true
    }
}

/// A name pinned on the part: a small dark chip with the name in its colour.
fn pin_label(painter: &egui::Painter, at: Pos2, name: &str, color: Color32) {
    let font = egui::FontId::proportional(12.0);
    let galley = painter.layout_no_wrap(name.to_string(), font, color);
    let rect = egui::Rect::from_center_size(at, galley.size() + egui::vec2(10.0, 4.0));
    painter.rect_filled(rect, 3.0, Color32::from_black_alpha(170));
    painter.rect_stroke(rect, 3.0, Stroke::new(1.0, color), egui::StrokeKind::Inside);
    painter.galley(rect.min + egui::vec2(5.0, 2.0), galley, color);
}

/// Put the surfaces on the GPU, see-through, never cut away by the sketch's plane.
pub(crate) fn sync_surfaces(gpu: &mut HashMap<u64, GpuMesh>, renderer: &SceneRenderer, rs: &eframe::egui_wgpu::RenderState, surfaces: &[(u64, Arc<Mesh>, Vec4)], base: ObjectStyle) {
    gpu.retain(|k, _| surfaces.iter().any(|s| s.0 == *k));
    let mut moves = Vec::new();
    for (key, mesh, color) in surfaces {
        let style = ObjectStyle { color: *color, crease_deg: None, cut_away: false, ..base };
        let g = gpu.entry(*key).or_insert_with(|| renderer.create_mesh(&rs.device));
        renderer.sync_mesh(&rs.device, &rs.queue, g, mesh, None, 0, &style, (&mut moves, 0), None, None, None);
    }
}

/// The surfaces as the frame draws them: see-through.
pub(crate) fn surface_objects<'a>(gpu: &'a HashMap<u64, GpuMesh>, surfaces: &[(u64, Arc<Mesh>, Vec4)]) -> Vec<(&'a GpuMesh, bool)> {
    surfaces.iter().filter_map(|s| gpu.get(&s.0)).map(|g| (g, true)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_ray_meets_a_triangle_in_front_of_it_only() {
        let (a, b, c) = (DVec3::new(-1.0, -1.0, 0.0), DVec3::new(1.0, -1.0, 0.0), DVec3::new(0.0, 1.0, 0.0));
        assert!((ray_triangle(DVec3::new(0.0, 0.0, 5.0), -DVec3::Z, a, b, c).unwrap() - 5.0).abs() < 1e-12);
        assert!(ray_triangle(DVec3::new(0.0, 0.0, 5.0), DVec3::Z, a, b, c).is_none());
        assert!(ray_triangle(DVec3::new(3.0, 0.0, 5.0), -DVec3::Z, a, b, c).is_none());
    }

    #[test]
    fn a_hint_sheet_is_centred_on_the_models_centre_laid_on_the_plane() {
        let frame = PlaneRef::world(WorldAxis::Z, 4.0).own_frame().unwrap();
        let c = sheet_corners(&frame, DVec3::new(1.0, 2.0, 30.0), 10.0);
        let mid = (c[0] + c[2]) / 2.0;
        assert!((mid - DVec3::new(1.0, 2.0, 4.0)).length() < 1e-12);
        assert!(((c[2] - c[0]).length() - 20.0 * 2f64.sqrt()).abs() < 1e-9);
    }

    #[test]
    fn a_geometry_reads_the_same_until_it_changes() {
        let g = ElementGeometry::Sphere { center: DVec3::ONE, radius: 2.0 };
        let h = ElementGeometry::Sphere { center: DVec3::ONE, radius: 2.5 };
        assert_eq!(geometry_signature(&g, &[1.0]), geometry_signature(&g.clone(), &[1.0]));
        assert_ne!(geometry_signature(&g, &[1.0]), geometry_signature(&h, &[1.0]));
        assert_ne!(geometry_signature(&g, &[1.0]), geometry_signature(&g, &[2.0]));
    }
}
