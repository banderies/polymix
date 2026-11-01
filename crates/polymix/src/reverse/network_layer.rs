// SPDX-License-Identifier: AGPL-3.0-only
//! The Patch network's layer in the 3D view and what its box keeps while it is open: the
//! network's curves drawn on the scan — each the smooth curve fitted to its trace, a crease
//! dashed — its nodes as dots and the patches' corners as squares, the grips between them, why
//! a region was refused; and the patch being placed, its sides along the scan and its area
//! shaded, from its first corner on. While the box is open a click on the scan is the
//! network's. Ported from ScanRuler (`plugins/reverse/viewer/networkLayer.ts`,
//! `plugins/reverse/app/useNetworkLayer.ts`, `plugins/reverse/app/keyboard.ts`), © Stefan
//! Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! Draw, patch first: each click is a corner — on the scan, on a node, on a curve (split
//! there) or on a grip (made a node); the sides follow the scan between them, and a click back
//! on the first corner, Enter, a double click or a click of the middle button closes the patch,
//! which is filled at once — shaded where it will be until its fit lands. Enter with two
//! corners makes the line between them alone; Backspace takes the last corner back, Esc lets
//! the patch go. An edge of one patch only, dragged, is copied out to where it is let go, and
//! the patch between the two filled: the next patch of a strip, its new corners put on nodes
//! within NODE_PX.
//!
//! A patch made is edited where it lies: a node or a grip is dragged along the scan; an edge
//! two patches share is bent, a grip put in where it was pressed and dragged; the patches with
//! a moved corner keep their seeds in their middles. A click on a patch picks it, Delete takes
//! the picked one out with the edges no other patch has; Shift-click on a curve puts a grip in,
//! Delete takes out what is under the cursor, C makes the curve under it a crease, the wheel
//! over a curve sets its strip's span count. Fill: a click in a closed region fills it, on a
//! patch picks it, with Ctrl empties it. The creases the scan suggests are drawn dashed in
//! their own colour until taken or dismissed; a click on one takes it as a crease curve.
//!
//! The build reads the network (`mm_reverse::build::network`): the open box's network is built
//! as its preview — the patches meshed, no kernel faces — and its report comes back here
//! through the build's sink; the last report is kept while the next is built. The scan is read
//! for paths once per scan, here and for the build alike. While the box is open the scan is
//! coloured by its curvature, where the deviation map is not on.

use std::collections::{HashMap, VecDeque};
use std::sync::mpsc::{channel, Receiver};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use egui::{Color32, Pos2, Stroke};
use glam::{DMat4, DVec3};
use mm_reverse::build::{NetworkContext, NetworkOutcome};
use mm_reverse::model::{BodyId, CurveSide, FeatureId, FeatureKind, NetworkFeature, Part, SideMode};
use mm_reverse::network::CellOutcome;
use mm_reverse::network_body::NetworkReport;
use mm_reverse::network_creases::{suggest_creases, CreaseOptions};
use mm_reverse::network_curve::{fit_provisional, sample_curve, trace_curve, CurveTrace, DEFAULT_TOLERANCE, SPANS_MAX, SPANS_MIN};
use mm_reverse::network_draft::{self as nd, network_ready, patch_under, ring_of, CellPoint, NetworkDraft, NetworkStart, NetworkTool, PlacedCorner, SuggestedCrease};
use mm_reverse::network_seed::seed_inside;
use mm_reverse::scan_path::ScanPaths;
use mm_reverse::topo::{describe_face, match_edge, match_tolerance, FaceFingerprint};

use super::{FeatureEdit, Mode, PickWhat, ReverseWorkspace, ScanPart};
use crate::tools::Viewport;

/// What a box's first pick says while the network's box has just opened: a fresh draft is
/// taken of its feature, and the pick turns to [`NET_OPEN`].
pub(crate) const NET_FRESH: u32 = 0x4e45_5401;
pub(crate) const NET_OPEN: u32 = 0x4e45_5402;

/// How near the cursor a node, a grip or a curve has to be to be under it, px.
const NODE_PX: f32 = 10.0;
const GRIP_PX: f32 = 8.0;
const CURVE_PX: f32 = 8.0;
/// How far a press on a node or a grip moves before it is a drag, px.
const DRAG_SLACK_PX: f32 = 6.0;
/// Two clicks this close in time and place are a double click.
const DOUBLE_MS: u128 = 400;
const DOUBLE_PX: f32 = 6.0;
/// The samples a curve is drawn with, per span of its fit.
const DRAW_PER_SPAN: usize = 8;
const CURVE_COLOR: Color32 = Color32::from_rgb(0xf2, 0xa5, 0x16);
const LIT_COLOR: Color32 = Color32::WHITE;
const REFUSED_COLOR: Color32 = Color32::from_rgb(0xe5, 0x48, 0x4d);
const PICKED_COLOR: Color32 = Color32::from_rgb(0x4f, 0xb3, 0xff);
const SUGGESTED_COLOR: Color32 = Color32::from_rgb(0xb7, 0x94, 0xf6);
const MARK_RIM: Color32 = Color32::from_rgb(0x1d, 0x20, 0x26);
/// The colour the network's body is drawn in while its box is open (the view's preview).
const BODY_COLOR: Color32 = Color32::from_rgb(115, 204, 140);
/// The colour of a body's edge a curve runs along (the view's body colour).
const ALONG_COLOR: Color32 = Color32::from_rgb(143, 168, 204);
/// A suggested crease's samples between two read for the cursor.
const SUGGESTED_STRIDE: usize = 4;
/// How strongly the patch being placed is shaded.
const PREVIEW_OPACITY: f32 = 0.22;
/// The sides between placed corners kept traced, at most.
const SIDE_CACHE: usize = 64;
/// The width of a curve as drawn, px (the section line's, and one more).
const LINE: f32 = 2.0;
/// The stamps of the curvature colouring, apart from the deviation map's.
const CURVATURE_STAMP: u64 = 1 << 62;

/// What lies under the cursor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum NetHit {
    Node(u32),
    /// A corner of the patch being placed that is no node yet.
    Placed(usize),
    /// A split cell's centre, by the feature's cell.
    Centre(usize),
    /// A body's edge (PN11): a curve along it.
    BodyEdge { body: BodyId, edge: usize },
    /// A crease suggested, by its index among those offered.
    Suggested(usize),
    Grip { curve: u32, index: usize },
    Curve { curve: u32, run: usize, at: DVec3 },
    /// A patch of the last build, under the cursor away from its edges.
    Patch { patch: usize, at: DVec3 },
    Scan(DVec3),
}

impl NetHit {
    fn drags(&self) -> bool {
        matches!(self, NetHit::Node(_) | NetHit::Grip { .. } | NetHit::Centre(_) | NetHit::Curve { .. })
    }
}

/// A curve as drawn: its trace on the scan and its fit, sampled.
struct DrawnCurve {
    key: Vec<u64>,
    trace: CurveTrace,
    points: Vec<DVec3>,
}

/// A patch about to be made of an edge dragged out: its corners — the edge's two nodes, then
/// the copies — the new edge's grips, and its four sides along the scan, round it in order.
struct Extension {
    corners: Vec<PlacedCorner>,
    vias: Vec<Option<Vec<DVec3>>>,
    sides: Vec<Vec<DVec3>>,
}

/// A press on what a drag moves, until it is let go.
struct Drag {
    hit: NetHit,
    before: NetworkFeature,
    moved: bool,
    from: Pos2,
    /// An edge two patches share: bent, a grip put in where it was pressed.
    bend: bool,
}

/// An edge being dragged out: the curve, where it was pressed, and where the cursor is on the
/// scan now.
struct Extending {
    curve: u32,
    from: DVec3,
    to: DVec3,
}

/// The patch being placed, or the edge being dragged out, as the layer draws it — its sides
/// along the scan in order (the one dashed marked), and its corners — and what it was drawn
/// for: traced again only when that changes, the sides between corners placed kept traced.
struct Preview {
    key: PreviewKey,
    lines: Vec<(Vec<DVec3>, bool)>,
    corners: Vec<DVec3>,
}

/// What a preview is drawn for: what is under the cursor, the corners placed (where they are,
/// and the first as placed), the edge dragged out, the scan.
#[derive(Debug, Clone, PartialEq)]
struct PreviewKey {
    hovered: Option<NetHit>,
    placed: Vec<DVec3>,
    first: Option<PlacedCorner>,
    extending: Option<(u32, DVec3, DVec3)>,
    scan: u64,
}

/// A patch closed and not yet in a report: shaded where it will be, by how many cells the
/// draft had with it, until its fit lands.
struct Waiting {
    cells: usize,
    since: u64,
    outline: Vec<DVec3>,
}

/// The scan read for paths, per scan: made once, by whichever asks first — the build's thread
/// or the box's own.
type PathsCell = Arc<OnceLock<Option<Arc<ScanPaths>>>>;

/// The curvature colouring of the scan, per object: (stamp, colour per triangle id), and the
/// radius the full colour stands for.
#[derive(Default)]
struct Curvature {
    key: u64,
    job: Option<Receiver<Vec<(u64, Vec<u32>, f64)>>>,
    colors: HashMap<u64, (u64, Arc<Vec<u32>>)>,
    range: f64,
}

/// What the network's box and layer keep while the box is open.
#[derive(Default)]
pub(crate) struct NetState {
    /// The draft: the feature the box edits, and its undo, the patch being placed, the patch
    /// picked, the tool and the creases offered.
    draft: Option<NetworkDraft>,
    /// The scan read for paths, by the scan it was read off.
    paths: Option<(u64, PathsCell)>,
    paths_job: bool,
    /// Each network's outcome as the build said it, with its build's generation.
    outcomes: Arc<Mutex<VecDeque<(u64, NetworkOutcome)>>>,
    /// The open box's last outcome, and its build's generation.
    outcome: Option<(u64, NetworkOutcome)>,
    /// The last report, kept while the next is built.
    report: Option<NetworkReport>,
    curves: HashMap<u32, DrawnCurve>,
    sides: HashMap<Vec<u64>, Vec<DVec3>>,
    hovered: Option<NetHit>,
    last_click: Option<(Instant, Pos2)>,
    drag: Option<Drag>,
    extending: Option<Extending>,
    /// The patch the edge being dragged out makes, as it stands where the cursor is now.
    extension: Option<Extension>,
    /// The patch being placed or dragged out, as drawn.
    preview: Option<Preview>,
    waiting: Vec<Waiting>,
    suggest: Option<Receiver<Vec<SuggestedCrease>>>,
    pending_build: bool,
    curvature: Curvature,
    /// The view's context, to read the pointer and the keys by.
    ctx: Option<egui::Context>,
    /// The build the box's session began at: outcomes of builds before it are another box's.
    since: u64,
    /// The network as it stood when a drag of a fit's number began.
    fit_drag: Option<NetworkFeature>,
}

impl NetState {
    /// The scan's curvature colouring, while the network's box is open (`mode`).
    pub(crate) fn curvature_field(&self, id: u64, mode: &Mode) -> crate::renderer::ScalarField<'_> {
        let open = matches!(mode, Mode::Feature(e) if matches!(e.kind, FeatureKind::Network(_)));
        if !open {
            return None;
        }
        self.curvature.colors.get(&id).map(|(stamp, c)| (*stamp, crate::renderer::TriField::Rgb(c.as_slice())))
    }
}

// ---- the scan ----

/// The scan's visible objects as one mesh in the part's frame, prepared for paths along it.
fn merged_paths(parts: &[ScanPart]) -> Option<Arc<ScanPaths>> {
    let mut positions: Vec<f32> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    for p in parts {
        let mut map: HashMap<u32, u32> = HashMap::new();
        for t in &p.triangles {
            for v in t {
                let at = *map.entry(*v).or_insert_with(|| {
                    let q = p.positions.get(*v as usize).copied().unwrap_or_default();
                    positions.extend_from_slice(&[q.x as f32, q.y as f32, q.z as f32]);
                    (positions.len() / 3 - 1) as u32
                });
                indices.push(at);
            }
        }
    }
    (!indices.is_empty()).then(|| Arc::new(ScanPaths::new(positions, indices)))
}

/// The scan read for paths for the scan shown now (the cell, empty until read).
fn paths_cell(ws: &mut ReverseWorkspace) -> PathsCell {
    let key = ws.scan_key;
    match &ws.net.paths {
        Some((k, cell)) if *k == key => cell.clone(),
        _ => {
            // Another scan: the curves and sides traced on the last are let go.
            let cell: PathsCell = Arc::new(OnceLock::new());
            ws.net.paths = Some((key, cell.clone()));
            ws.net.paths_job = false;
            ws.net.curves.clear();
            ws.net.sides.clear();
            cell
        }
    }
}

/// The scan read for paths, or `None` while it is being read (on a thread of its own).
fn paths(ws: &mut ReverseWorkspace) -> Option<Arc<ScanPaths>> {
    let cell = paths_cell(ws);
    if let Some(p) = cell.get() {
        return p.clone();
    }
    if !ws.net.paths_job {
        ws.net.paths_job = true;
        let parts = ws.scan_parts();
        let ctx = ws.net.ctx.clone();
        std::thread::spawn(move || {
            // The search along the edges, where a side cannot be cut as a plane, made ready
            // with the rest.
            if let Some(p) = cell.get_or_init(|| merged_paths(&parts)) {
                p.prepare_edge_routes();
            }
            if let Some(c) = ctx {
                c.request_repaint();
            }
        });
    }
    None
}

/// The scan read for paths, now — read on this thread if it is not yet (script steps).
fn paths_now(ws: &mut ReverseWorkspace) -> Option<Arc<ScanPaths>> {
    let cell = paths_cell(ws);
    let parts = ws.scan_parts();
    cell.get_or_init(|| merged_paths(&parts)).clone()
}

/// What a build of the part reads for its networks (see `mm_reverse::build::network`): the
/// scan, read for paths the first time a network asks, the open box's feature, and where each
/// network's outcome goes — `None` for a part with no network.
pub(crate) fn build_context(ws: &mut ReverseWorkspace, part: &Part, preview: Option<FeatureId>, generation: u64) -> Option<NetworkContext> {
    if !part.features.iter().any(|f| matches!(f.kind, FeatureKind::Network(_)) && !f.suppressed()) {
        return None;
    }
    let cell = paths_cell(ws);
    let parts = ws.scan_parts();
    let outcomes = ws.net.outcomes.clone();
    let ctx = ws.net.ctx.clone();
    Some(NetworkContext {
        scan: Box::new(move || cell.get_or_init(|| merged_paths(&parts)).clone()),
        preview,
        sink: Some(Box::new(move |o| {
            if let Ok(mut q) = outcomes.lock() {
                q.push_back((generation, o));
                while q.len() > 64 {
                    q.pop_front();
                }
            }
            if let Some(c) = &ctx {
                c.request_repaint();
            }
        })),
    })
}

/// The open box's outcomes as they came in: the newest of its feature's kept, its report the
/// one the box and the layer read until the next.
fn take_outcomes(ws: &mut ReverseWorkspace) {
    let Some(feature) = ws.preview_feature else { return };
    let came: Vec<(u64, NetworkOutcome)> = match ws.net.outcomes.lock() {
        Ok(mut q) => q.drain(..).collect(),
        Err(_) => return,
    };
    for (generation, o) in came {
        if o.feature != feature || !o.preview || generation < ws.net.since {
            continue;
        }
        if ws.net.outcome.as_ref().is_some_and(|(g, _)| *g > generation) {
            continue;
        }
        ws.net.report = o.report.clone();
        ws.net.outcome = Some((generation, o));
    }
    // The patches closed that a report has now, or an undo took away.
    let cells = ws.net.draft.as_ref().map_or(0, |d| d.data.cells.len());
    let latest = ws.net.outcome.as_ref().map(|(g, _)| *g);
    let report_cells = ws.net.report.as_ref().map(|r| r.cells.len());
    ws.net.waiting.retain(|w| cells >= w.cells && !(latest.is_some_and(|g| g > w.since) && report_cells.is_some_and(|n| n >= w.cells)));
}

// ---- the session ----

/// The box opened, or it stands open: a fresh draft of its feature when it has just opened,
/// the feature taken in where it changed from outside the box.
fn begin(ws: &mut ReverseWorkspace, x: &NetworkFeature, pick: &mut PickWhat) {
    let fresh = *pick != PickWhat::Custom(NET_OPEN) || ws.net.draft.is_none();
    if fresh {
        *pick = PickWhat::Custom(NET_OPEN);
        let keep = (ws.net.paths.take(), std::mem::take(&mut ws.net.curvature), ws.net.ctx.take(), ws.net.outcomes.clone());
        ws.net = NetState::default();
        ws.net.paths = keep.0;
        ws.net.curvature = keep.1;
        ws.net.ctx = keep.2;
        ws.net.outcomes = keep.3;
        ws.net.since = ws.build_generation;
        ws.net.draft = Some(NetworkDraft::of(x));
        return;
    }
    if let Some(d) = &mut ws.net.draft {
        if d.data != *x {
            let next = NetworkDraft::of(x).next_id.max(d.next_id);
            d.data = x.clone();
            d.next_id = next;
            d.placing = None;
        }
    }
}

fn draft(ws: &ReverseWorkspace) -> Option<&NetworkDraft> {
    ws.net.draft.as_ref()
}

/// An edit of the draft, its data the box's feature.
fn edit(ws: &mut ReverseWorkspace, x: &mut NetworkFeature, change: impl FnOnce(&NetworkDraft) -> NetworkDraft) {
    let Some(d) = &ws.net.draft else { return };
    let next = change(d);
    *x = next.data.clone();
    ws.net.draft = Some(next);
}

/// The preview to be built again for an edit made while a drag goes on: when no build is under
/// way — a drag's moves come faster than the preview is built. The box's caller asks for the
/// build, with the box back in its place (see [`build_due`]).
fn rebuild(ws: &mut ReverseWorkspace) {
    ws.net.pending_build = true;
}

/// Whether an edit held back is to be built now: no build under way.
fn build_due(ws: &mut ReverseWorkspace) -> bool {
    if ws.net.pending_build && !ws.building() {
        ws.net.pending_build = false;
        return true;
    }
    false
}

// ---- the curves as drawn ----

fn bits(points: &[DVec3], crease: bool) -> Vec<u64> {
    points.iter().flat_map(|p| [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()]).chain(std::iter::once(crease as u64)).collect()
}

fn as_points(xyz: &[f32]) -> Vec<DVec3> {
    xyz.chunks_exact(3).map(|p| DVec3::new(p[0] as f64, p[1] as f64, p[2] as f64)).collect()
}

/// The curves traced and fitted again where their points changed.
fn refresh_curves(ws: &mut ReverseWorkspace) {
    let Some(paths) = paths(ws) else { return };
    // Each curve's points — its ends and grips — and whether it is a crease.
    let Some(wanted) = draft(ws).map(|d| {
        let at = |id: u32| d.data.nodes.iter().find(|n| n.id == id).map(|n| n.at);
        d.data
            .curves
            .iter()
            .filter_map(|c| {
                let (a, b) = (at(c.from)?, at(c.to)?);
                let points: Vec<DVec3> = std::iter::once(a).chain(c.via.iter().copied()).chain(std::iter::once(b)).collect();
                Some((c.id, points, c.crease == Some(true)))
            })
            .collect::<Vec<_>>()
    }) else {
        return;
    };
    let mut keep: Vec<u32> = Vec::new();
    for (id, points, crease) in wanted {
        keep.push(id);
        let key = bits(&points, crease);
        if ws.net.curves.get(&id).is_some_and(|x| x.key == key) {
            continue;
        }
        ws.net.curves.remove(&id);
        let trace = trace_curve(&paths, &points);
        if trace.points.len() < 6 {
            continue;
        }
        let fit = fit_provisional(trace.as_points(), paths.edge());
        let samples = sample_curve(&fit, fit.spans * DRAW_PER_SPAN + 1);
        let drawn = samples.chunks_exact(3).map(|p| DVec3::new(p[0], p[1], p[2])).collect();
        ws.net.curves.insert(id, DrawnCurve { key, trace, points: drawn });
    }
    ws.net.curves.retain(|id, _| keep.contains(id));
}

/// The scan's path from one point to another, through grips: the ones between corners placed
/// kept, the ones to the cursor made afresh.
fn side_between(ws: &mut ReverseWorkspace, paths: &ScanPaths, points: &[DVec3], keep: bool) -> Option<Vec<DVec3>> {
    let key = bits(points, false);
    if let Some(had) = ws.net.sides.get(&key) {
        return Some(had.clone());
    }
    let trace = trace_curve(paths, points);
    if trace.points.len() < 6 {
        return None;
    }
    let out = as_points(&trace.points);
    if keep {
        if ws.net.sides.len() >= SIDE_CACHE {
            ws.net.sides.clear();
        }
        ws.net.sides.insert(key, out.clone());
    }
    Some(out)
}

/// A closed loop through points, along the scan side by side.
fn loop_through(ws: &mut ReverseWorkspace, paths: &ScanPaths, points: &[DVec3]) -> Option<Vec<DVec3>> {
    let mut out: Vec<DVec3> = Vec::new();
    for i in 0..points.len() {
        if let Some(side) = side_between(ws, paths, &[points[i], points[(i + 1) % points.len()]], true) {
            out.extend(side);
        }
    }
    (!out.is_empty()).then_some(out)
}

fn flat(points: &[DVec3]) -> Vec<f64> {
    points.iter().flat_map(|p| [p.x, p.y, p.z]).collect()
}

// ---- where things are on the screen ----

/// Points of the part where the view draws them, the camera read once.
struct Projector {
    vp: Viewport,
    m: DMat4,
    flip: bool,
}

impl Projector {
    fn of(ws: &ReverseWorkspace, vp: Viewport) -> Projector {
        Projector { vp, m: ws.view.camera.view_proj(vp.aspect()), flip: ws.flip }
    }

    fn at(&self, p: DVec3) -> Option<Pos2> {
        let q = if self.flip { crate::app::flip_zy_in(p) } else { p };
        let c = self.m * q.extend(1.0);
        if c.w <= 1e-12 {
            return None;
        }
        let n = c.truncate() / c.w;
        (0.0..=1.0).contains(&n.z).then(|| self.vp.to_screen(n))
    }
}

/// The scan under a screen point (part coordinates): the view's own cast against the scan,
/// whatever body stands in front of it.
fn scan_at(ws: &mut ReverseWorkspace, vp: Viewport, p: Pos2) -> Option<DVec3> {
    let ray = ws.view.camera.ray_from_ndc(vp.ndc(p), vp.aspect());
    let objects: Vec<super::ScanObject> = ws.scan.iter().filter(|o| o.visible).cloned().collect();
    let mut best: Option<(f64, DVec3)> = None;
    for o in &objects {
        let Some(bvh) = ws.scan_bvh(o) else { continue };
        if let Some(h) = bvh.raycast(&o.mesh, &ray) {
            if best.is_none_or(|(t, _)| h.t < t) {
                best = Some((h.t, h.point));
            }
        }
    }
    best.map(|(_, q)| ws.to_part(q))
}

fn seg_dist(p: Pos2, a: Pos2, b: Pos2) -> f32 {
    let ab = b - a;
    let t = if ab.length_sq() > 0.0 { ((p - a).dot(ab) / ab.length_sq()).clamp(0.0, 1.0) } else { 0.0 };
    (a + ab * t - p).length()
}

/// What is under a screen point: a node or a corner placed, a grip, a centre, a curve, a
/// crease offered, a body's edge, a patch, the scan.
fn hit_at(ws: &mut ReverseWorkspace, vp: Viewport, p: Pos2) -> Option<NetHit> {
    let (near, free) = {
        let d = ws.net.draft.as_ref()?;
        (hit_near(ws, d, vp, p), d.tool == NetworkTool::Draw && d.placing.is_none())
    };
    if near.is_some() {
        return near;
    }
    // A body's edge, while no patch is being placed: a curve along it.
    if free {
        if let Some((body, edge)) = ws.edge_at(vp, p) {
            return Some(NetHit::BodyEdge { body, edge });
        }
    }
    let scan = scan_at(ws, vp, p)?;
    // A patch, away from its edges: picked by a click while none is being placed.
    if free {
        if let Some(r) = &ws.net.report {
            if let Some(patch) = patch_under(r.patches.iter().map(|p| p.grid.as_slice()), scan) {
                return Some(NetHit::Patch { patch, at: scan });
            }
        }
    }
    Some(NetHit::Scan(scan))
}

/// What of the network's own is under a screen point, by where it is drawn: a node or a corner
/// placed, a grip, a centre, a curve, a crease offered — read off the draft as it stands, not a
/// copy of it (a pick runs every frame the pointer is over the view).
fn hit_near(ws: &ReverseWorkspace, d: &NetworkDraft, vp: Viewport, p: Pos2) -> Option<NetHit> {
    let screen = Projector::of(ws, vp);
    let mut best: Option<NetHit> = None;
    let mut best_d = NODE_PX;
    for n in &d.data.nodes {
        if let Some(s) = screen.at(n.at) {
            let dist = (s - p).length();
            if dist <= best_d {
                best_d = dist;
                best = Some(NetHit::Node(n.id));
            }
        }
    }
    for (index, c) in d.placing.iter().flatten().enumerate() {
        if matches!(c, PlacedCorner::Node(_)) {
            continue;
        }
        let Some(at) = nd::corner_point(&d.data, c) else { continue };
        if let Some(s) = screen.at(at) {
            let dist = (s - p).length();
            if dist <= best_d {
                best_d = dist;
                best = Some(NetHit::Placed(index));
            }
        }
    }
    if best.is_some() {
        return best;
    }
    best_d = GRIP_PX;
    for c in &d.data.curves {
        for (index, q) in c.via.iter().enumerate() {
            if let Some(s) = screen.at(*q) {
                let dist = (s - p).length();
                if dist <= best_d {
                    best_d = dist;
                    best = Some(NetHit::Grip { curve: c.id, index });
                }
            }
        }
    }
    if best.is_some() {
        return best;
    }
    for c in ws.net.report.iter().flat_map(|r| r.centres.iter()) {
        if let Some(s) = screen.at(c.at) {
            let dist = (s - p).length();
            if dist <= best_d {
                best_d = dist;
                best = Some(NetHit::Centre(c.cell));
            }
        }
    }
    if best.is_some() {
        return best;
    }
    // A curve: the nearest of its trace's points, on the scan, and the run between grips it
    // is on.
    best_d = CURVE_PX;
    for (id, c) in &ws.net.curves {
        let t = &c.trace;
        for i in 0..t.points.len() / 3 {
            let q = t.point(i);
            let Some(s) = screen.at(q) else { continue };
            let dist = (s - p).length();
            if dist > best_d {
                continue;
            }
            best_d = dist;
            let mut run = 0;
            while run + 1 < t.grips.len() && t.grips[run + 1] <= i {
                run += 1;
            }
            best = Some(NetHit::Curve { curve: *id, run: run.min(t.grips.len().saturating_sub(2)), at: q });
        }
    }
    if best.is_some() {
        return best;
    }
    let free = d.tool == NetworkTool::Draw && d.placing.is_none();
    // A crease offered, while no patch is being placed: read by the chords between every
    // fourth of its samples.
    if free {
        for (index, s) in d.suggested.iter().flatten().enumerate() {
            let point = |i: usize| DVec3::new(s.points[i * 3] as f64, s.points[i * 3 + 1] as f64, s.points[i * 3 + 2] as f64);
            let count = s.points.len() / 3;
            if count == 0 {
                continue;
            }
            let last = count - 1;
            let mut a = screen.at(point(0));
            let mut i = 0;
            while i < last {
                i = (i + SUGGESTED_STRIDE).min(last);
                let b = screen.at(point(i));
                if let (Some(a), Some(b)) = (a, b) {
                    let dist = seg_dist(p, a, b);
                    if dist <= best_d {
                        best_d = dist;
                        best = Some(NetHit::Suggested(index));
                    }
                }
                a = b;
            }
        }
    }
    best
}

fn hit_point(ws: &ReverseWorkspace, hit: &NetHit) -> Option<DVec3> {
    let d = draft(ws)?;
    match *hit {
        NetHit::BodyEdge { .. } | NetHit::Suggested(_) => None,
        NetHit::Node(id) => d.data.nodes.iter().find(|n| n.id == id).map(|n| n.at),
        NetHit::Placed(index) => d.placing.as_ref().and_then(|p| p.get(index)).and_then(|c| nd::corner_point(&d.data, c)),
        NetHit::Centre(cell) => ws.net.report.as_ref().and_then(|r| r.centres.iter().find(|c| c.cell == cell)).map(|c| c.at),
        NetHit::Grip { curve, index } => d.data.curves.iter().find(|c| c.id == curve).and_then(|c| c.via.get(index)).copied(),
        NetHit::Curve { at, .. } | NetHit::Patch { at, .. } | NetHit::Scan(at) => Some(at),
    }
}

/// A corner of the patch for what was clicked.
fn corner_of(ws: &ReverseWorkspace, hit: &NetHit) -> Option<PlacedCorner> {
    match *hit {
        NetHit::Node(id) => Some(PlacedCorner::Node(id)),
        NetHit::Grip { curve, index } => {
            let at = draft(ws)?.data.curves.iter().find(|c| c.id == curve)?.via.get(index).copied()?;
            Some(PlacedCorner::Grip { curve, grip: index, at })
        }
        NetHit::Curve { curve, at, .. } => Some(PlacedCorner::Curve { curve, at }),
        NetHit::Scan(at) => Some(PlacedCorner::At(at)),
        _ => None,
    }
}

// ---- placing a patch ----

/// The patch being placed closed and made, filled in its middle.
fn close_placing(ws: &mut ReverseWorkspace, x: &mut NetworkFeature) {
    let Some((corners, before)) = draft(ws).and_then(|d| d.placing.as_ref().map(|p| (p.iter().filter_map(|c| nd::corner_point(&d.data, c)).collect::<Vec<DVec3>>(), d.data.cells.len()))) else { return };
    let Some(paths) = paths(ws) else { return };
    if corners.len() < 3 {
        return;
    }
    let outline = loop_through(ws, &paths, &corners);
    let seed = outline.as_ref().and_then(|o| seed_inside(&paths, &flat(o)));
    edit(ws, x, |d| nd::commit_placing(d, true, seed));
    wait_for(ws, outline, before);
}

/// An edit that may have made a patch: if it did, its outline shaded until its fit lands.
fn wait_for(ws: &mut ReverseWorkspace, outline: Option<Vec<DVec3>>, cells_before: usize) {
    let cells = ws.net.draft.as_ref().map_or(0, |d| d.data.cells.len());
    if cells > cells_before {
        if let Some(outline) = outline.filter(|o| o.len() >= 3) {
            let since = ws.net.outcome.as_ref().map_or(0, |(g, _)| *g);
            ws.net.waiting.push(Waiting { cells, since, outline });
        }
    }
}

/// The patch being placed finished — Enter, a double click, the middle button: closed with
/// three corners or more, a line with two, let go with one.
fn finish_placing(ws: &mut ReverseWorkspace, x: &mut NetworkFeature) {
    let count = ws.net.draft.as_ref().and_then(|d| d.placing.as_ref()).map_or(0, Vec::len);
    if count >= 3 {
        close_placing(ws, x);
    } else if count == 2 {
        edit(ws, x, |d| nd::commit_placing(d, false, None));
    } else if count == 1 {
        edit(ws, x, nd::drop_placing);
    }
}

/// A patch of the last report deleted, with the edges no other patch has; false where there is
/// none.
fn delete_patch(ws: &mut ReverseWorkspace, x: &mut NetworkFeature, patch: Option<usize>) -> bool {
    let Some(report) = ws.net.report.clone() else { return false };
    let Some(patch) = patch.filter(|p| *p < report.patches.len()) else { return false };
    edit(ws, x, |d| nd::remove_reported_patch(d, &report, patch));
    ws.net.hovered = None;
    true
}

/// The patch the edge being dragged out makes: the edge copied by how far the cursor moved on
/// the scan, each copied end on a node within NODE_PX of it, its grips copied too.
fn extension(ws: &mut ReverseWorkspace, vp: Viewport) -> Option<Extension> {
    let e = ws.net.extending.as_ref()?;
    let (curve_id, from, to) = (e.curve, e.from, e.to);
    let (c, nodes) = {
        let d = draft(ws)?;
        (d.data.curves.iter().find(|x| x.id == curve_id)?.clone(), d.data.nodes.iter().map(|n| (n.id, n.at)).collect::<Vec<(u32, DVec3)>>())
    };
    let paths = paths(ws)?;
    let at = |id: u32| nodes.iter().find(|n| n.0 == id).map(|n| n.1);
    let (a, b) = (at(c.from)?, at(c.to)?);
    let off = to - from;
    let moved = |p: DVec3| paths.closest_inside(p + off).map_or(p + off, |q| q.point);
    let screen = Projector::of(ws, vp);
    let copy = |p: DVec3| -> PlacedCorner {
        let q = moved(p);
        let Some(s) = screen.at(q) else { return PlacedCorner::At(q) };
        let mut best: Option<(u32, f32)> = None;
        for &(id, n_at) in &nodes {
            if id == c.from || id == c.to {
                continue;
            }
            let Some(t) = screen.at(n_at) else { continue };
            let dist = (t - s).length();
            if dist <= NODE_PX && best.is_none_or(|(_, bd)| dist < bd) {
                best = Some((id, dist));
            }
        }
        match best {
            Some((id, _)) => PlacedCorner::Node(id),
            None => PlacedCorner::At(q),
        }
    };
    let b_new = copy(b);
    let a_new = copy(a);
    let point = |corner: &PlacedCorner| match corner {
        PlacedCorner::Node(id) => at(*id),
        PlacedCorner::At(q) | PlacedCorner::Curve { at: q, .. } | PlacedCorner::Grip { at: q, .. } => Some(*q),
    };
    let (pb, pa) = (point(&b_new)?, point(&a_new)?);
    let mut via: Vec<DVec3> = c.via.iter().map(|p| moved(*p)).collect();
    via.reverse();
    let own = match ws.net.curves.get(&c.id) {
        Some(x) => Some(as_points(&x.trace.points)),
        None => side_between(ws, &paths, &[vec![a], c.via.clone(), vec![b]].concat(), false),
    };
    let sides = vec![own, side_between(ws, &paths, &[b, pb], false), side_between(ws, &paths, &[vec![pb], via.clone(), vec![pa]].concat(), false), side_between(ws, &paths, &[pa, a], false)];
    let sides: Option<Vec<Vec<DVec3>>> = sides.into_iter().collect();
    Some(Extension { corners: vec![PlacedCorner::Node(c.from), PlacedCorner::Node(c.to), b_new, a_new], vias: vec![None, None, Some(via), None], sides: sides? })
}

/// New seeds for the patches placed by their corners with one of these among them: each in its
/// middle again.
fn seeds_after_move(ws: &mut ReverseWorkspace, nodes: &[u32]) -> Vec<(usize, DVec3)> {
    if nodes.is_empty() {
        return Vec::new();
    }
    // The cells placed by their corners with one of these among them: each with its corners'
    // points.
    let Some(cells) = draft(ws).map(|d| {
        d.data
            .cells
            .iter()
            .enumerate()
            .filter_map(|(index, cell)| {
                let corners = cell.corners.as_ref()?;
                corners.iter().any(|id| nodes.contains(id)).then(|| (index, corners.iter().filter_map(|id| d.data.nodes.iter().find(|n| n.id == *id).map(|n| n.at)).collect::<Vec<DVec3>>()))
            })
            .collect::<Vec<_>>()
    }) else {
        return Vec::new();
    };
    let Some(paths) = paths(ws) else { return Vec::new() };
    let mut out = Vec::new();
    for (index, points) in cells {
        if points.len() < 3 {
            continue;
        }
        let outline = loop_through(ws, &paths, &points);
        if let Some(seed) = outline.and_then(|o| seed_inside(&paths, &flat(&o))) {
            out.push((index, seed));
        }
    }
    out
}

// ---- clicks, drags and keys ----

/// A click in the view: true where it changed the network.
fn click(ws: &mut ReverseWorkspace, vp: Viewport, p: Pos2, x: &mut NetworkFeature) -> bool {
    let Some((tool, placing)) = draft(ws).map(|d| (d.tool, d.placing.clone())) else { return false };
    let Some(hit) = hit_at(ws, vp, p) else { return false };
    let (shift, additive) = ws.net.ctx.as_ref().map_or((false, false), |c| c.input(|i| (i.modifiers.shift, i.modifiers.command || i.modifiers.ctrl)));
    let now = Instant::now();
    let double = ws.net.last_click.is_some_and(|(t, at)| now.duration_since(t).as_millis() < DOUBLE_MS && (p - at).length() < DOUBLE_PX);
    ws.net.last_click = Some((now, p));
    match hit {
        NetHit::Suggested(index) => {
            edit(ws, x, |d| nd::accept_creases(d, &[index]));
            ws.net.hovered = None;
            return true;
        }
        NetHit::Centre(_) => return false,
        NetHit::BodyEdge { body, edge } => {
            // The face beside the edge the curve's patch meets: the first of its two, until the
            // box says otherwise.
            let Some(view) = ws.pick_view(body) else { return false };
            let Some(e) = view.topology.edges.get(edge) else { return false };
            let face = e.faces.iter().find_map(|h| view.topology.face(*h)).map(|f| f.fingerprint);
            let (print, ends) = (e.fingerprint.clone(), e.ends);
            edit(ws, x, |d| nd::add_edge_curve(d, body, &print, ends, face));
            return true;
        }
        _ => {}
    }
    if tool == NetworkTool::Fill {
        let at = match hit {
            NetHit::Scan(at) | NetHit::Curve { at, .. } | NetHit::Patch { at, .. } => at,
            _ => return false,
        };
        return fill_click(ws, x, at, additive);
    }
    let Some(placing) = placing else {
        if let NetHit::Patch { patch, .. } = hit {
            edit(ws, x, |d| NetworkDraft { picked: if d.picked == Some(patch) { None } else { Some(patch) }, ..d.clone() });
            return true;
        }
        if let (NetHit::Curve { curve, run, at }, true) = (hit, shift) {
            edit(ws, x, |d| nd::insert_grip(d, curve, run, at, true));
            return true;
        }
        if let Some(corner) = corner_of(ws, &hit) {
            edit(ws, x, |d| nd::place_corner(d, corner));
        }
        return true;
    };
    let count = placing.len();
    // A double click closes the patch at the corner its first click put down; with two
    // corners, it makes the line between them.
    if double {
        if count >= 2 {
            finish_placing(ws, x);
        }
        return true;
    }
    let first = placing[0];
    let on_first = matches!(hit, NetHit::Placed(0)) || matches!((hit, first), (NetHit::Node(h), PlacedCorner::Node(f)) if h == f);
    if count >= 3 && on_first {
        close_placing(ws, x);
        return true;
    }
    if matches!(hit, NetHit::Placed(_)) {
        return true;
    }
    if let NetHit::Node(id) = hit {
        if placing.iter().any(|c| *c == PlacedCorner::Node(id)) {
            return true;
        }
    }
    if let Some(corner) = corner_of(ws, &hit) {
        edit(ws, x, |d| nd::place_corner(d, corner));
    }
    true
}

/// A click in Fill: on a patch, pick it (Ctrl: empty it); elsewhere, fill the region.
fn fill_click(ws: &mut ReverseWorkspace, x: &mut NetworkFeature, at: DVec3, empty: bool) -> bool {
    let report = ws.net.report.clone();
    let patch = report.as_ref().and_then(|r| patch_under(r.patches.iter().map(|p| p.grid.as_slice()), at));
    if let (Some(patch), Some(r)) = (patch, &report) {
        let seed = r.cells.iter().position(|c| matches!(c, CellOutcome::Filled { patches, .. } if patches.contains(&patch)));
        match (empty, seed) {
            (true, Some(seed)) => edit(ws, x, |d| nd::remove_seed(d, seed)),
            _ => edit(ws, x, |d| NetworkDraft { picked: if d.picked == Some(patch) { None } else { Some(patch) }, ..d.clone() }),
        }
        return true;
    }
    if !empty {
        edit(ws, x, |d| nd::add_seed(d, at));
        return true;
    }
    false
}

/// The pointer over the view while the box is open: the hover, a press on what a drag moves,
/// the drag and its release, and a click. True where the network changed and is to be built
/// again now.
pub(crate) fn pointer(ws: &mut ReverseWorkspace, vp: Viewport, p: Pos2, clicked: bool, x: &mut NetworkFeature, pick: &mut PickWhat) -> bool {
    begin(ws, x, pick);
    take_outcomes(ws);
    let Some(ctx) = ws.net.ctx.clone() else { return false };
    let (pressed, down, released, mods) = ctx.input(|i| (i.pointer.button_pressed(egui::PointerButton::Primary), i.pointer.button_down(egui::PointerButton::Primary), i.pointer.button_released(egui::PointerButton::Primary), i.modifiers));
    let shift = mods.shift;
    // A left press the view's navigation takes (Alt held) turns the camera, and drags nothing.
    let navigating = ws.nav.bindings.starts_navigation(egui::PointerButton::Primary, mods);
    let mut changed = false;
    let tool = draft(ws).map_or(NetworkTool::Draw, |d| d.tool);
    let placing = draft(ws).is_some_and(|d| d.placing.is_some());
    // A press on what a drag moves: the layer's, not a click's.
    if pressed && !navigating && ws.net.drag.is_none() && tool == NetworkTool::Draw && !placing {
        if let Some(hit) = hit_at(ws, vp, p).filter(NetHit::drags) {
            // Shift on an edge is the click's: a grip put in.
            if !(matches!(hit, NetHit::Curve { .. }) && shift) {
                let bend = matches!(hit, NetHit::Curve { curve, .. } if ws.net.report.as_ref().is_some_and(|r| nd::shared_curves(r).contains(&curve)));
                let before = draft(ws).map(|d| d.data.clone()).unwrap_or_else(nd::empty_network);
                ws.net.drag = Some(Drag { hit, before, moved: false, from: p, bend });
            }
        }
    }
    let held = ws.net.drag.as_mut().map(|drag| {
        if down && !drag.moved && (p.x - drag.from.x).abs() + (p.y - drag.from.y).abs() > DRAG_SLACK_PX {
            drag.moved = true;
        }
        (drag.hit, drag.moved, drag.bend)
    });
    if let Some((hit, moved, bend)) = held {
        if down && moved {
            if let Some(at) = scan_at(ws, vp, p) {
                match hit {
                    NetHit::Curve { curve, run, at: pressed_at } if bend => {
                        // The grip the bend drags, put in where the edge was pressed — no step of
                        // its own: the drag's one undo takes it back too.
                        let n = draft(ws).and_then(|d| d.data.curves.iter().find(|c| c.id == curve)).map_or(0, |c| c.via.len());
                        let index = run.min(n);
                        edit(ws, x, |d| nd::move_grip(&nd::insert_grip(d, curve, index, pressed_at, false), curve, index, at, false));
                        if let Some(drag) = &mut ws.net.drag {
                            drag.hit = NetHit::Grip { curve, index };
                        }
                        rebuild(ws);
                    }
                    NetHit::Node(id) => {
                        edit(ws, x, |d| nd::move_node(d, id, at, false));
                        rebuild(ws);
                    }
                    NetHit::Centre(cell) => {
                        edit(ws, x, |d| nd::move_cell(d, cell, CellPoint::Centre, at, false));
                        rebuild(ws);
                    }
                    NetHit::Grip { curve, index } => {
                        edit(ws, x, |d| nd::move_grip(d, curve, index, at, false));
                        rebuild(ws);
                    }
                    NetHit::Curve { curve, at: from, .. } => {
                        ws.net.extending = Some(Extending { curve, from, to: at });
                        ws.net.extension = extension(ws, vp);
                    }
                    _ => {}
                }
                if moved {
                    refresh_curves(ws);
                }
            }
        }
        if released || !down {
            if let Some(drag) = ws.net.drag.take() {
                if let NetHit::Curve { .. } = drag.hit {
                    let q = if drag.moved { extension(ws, vp) } else { None };
                    ws.net.extending = None;
                    ws.net.extension = None;
                    if let Some(q) = q {
                        let outline: Vec<DVec3> = q.sides.concat();
                        let seed = paths(ws).and_then(|paths| seed_inside(&paths, &flat(&outline)));
                        let before = draft(ws).map_or(0, |d| d.data.cells.len());
                        edit(ws, x, |d| nd::add_patch(d, &q.corners, true, seed, Some(&q.vias)));
                        wait_for(ws, Some(outline), before);
                        changed = true;
                    } else if !drag.moved {
                        changed |= click(ws, vp, p, x);
                    }
                } else if drag.moved {
                    // The patches with the node — or the grip's curve's ends — among their
                    // corners keep their seeds in their middles.
                    let touched: Vec<u32> = match drag.hit {
                        NetHit::Node(id) => vec![id],
                        NetHit::Grip { curve, .. } => draft(ws).and_then(|d| d.data.curves.iter().find(|c| c.id == curve)).map_or(Vec::new(), |c| vec![c.from, c.to]),
                        _ => Vec::new(),
                    };
                    let before = drag.before.clone();
                    edit(ws, x, |d| nd::record_drag(d, &before));
                    let seeds = seeds_after_move(ws, &touched);
                    edit(ws, x, |d| nd::set_cell_seeds(d, &seeds));
                    changed = true;
                } else {
                    // The press was taken: one that never moved is a click on what it pressed.
                    changed |= click(ws, vp, p, x);
                }
            }
        }
    } else if clicked {
        changed |= click(ws, vp, p, x);
    }
    // The hover, where no drag holds the pointer.
    if ws.net.drag.is_none() {
        let hit = hit_at(ws, vp, p);
        ws.net.hovered = hit;
        let placing = draft(ws).is_some_and(|d| d.placing.is_some());
        let lit = hit.is_some_and(|h| !matches!(h, NetHit::Scan(_)));
        let drags = hit.is_some_and(|h| h.drags());
        let icon = match tool {
            NetworkTool::Fill => egui::CursorIcon::Cell,
            NetworkTool::Draw if drags && !placing => egui::CursorIcon::Move,
            NetworkTool::Draw if lit => egui::CursorIcon::PointingHand,
            NetworkTool::Draw => egui::CursorIcon::Crosshair,
        };
        ctx.set_cursor_icon(icon);
    } else {
        ctx.set_cursor_icon(egui::CursorIcon::Move);
    }
    if changed {
        refresh_curves(ws);
        ws.net.pending_build = false;
    }
    update_preview(ws);
    if ws.net.drag.is_some() || ws.net.suggest.is_some() || ws.net.pending_build {
        ctx.request_repaint();
    }
    changed || build_due(ws)
}

/// The edit the box holds, taken out of the mode for a key or a confirm, and put back; `None`
/// where no network's box is open.
fn with_edit<R>(ws: &mut ReverseWorkspace, f: impl FnOnce(&mut ReverseWorkspace, &mut NetworkFeature, &mut PickWhat) -> R) -> Option<R> {
    if !matches!(&ws.mode, Mode::Feature(e) if matches!(e.kind, FeatureKind::Network(_))) {
        return None;
    }
    let Mode::Feature(mut e) = std::mem::replace(&mut ws.mode, Mode::Idle) else { return None };
    let FeatureEdit { kind, pick, .. } = &mut *e;
    let out = match kind {
        FeatureKind::Network(x) => Some(f(ws, x, pick)),
        _ => None,
    };
    ws.mode = Mode::Feature(e);
    out
}

/// A key with the network's box open: true when it was the network's (ScanRuler's
/// `NetworkLayer.key`, and its own undo while the box is open).
pub(crate) fn key(ws: &mut ReverseWorkspace, ctx: &egui::Context) -> bool {
    let (esc, enter, backspace, delete, c, undo, redo, command) = ctx.input(|i| {
        (
            i.key_pressed(egui::Key::Escape),
            i.key_pressed(egui::Key::Enter),
            i.key_pressed(egui::Key::Backspace),
            i.key_pressed(egui::Key::Delete),
            i.key_pressed(egui::Key::C),
            i.modifiers.command && !i.modifiers.shift && i.key_pressed(egui::Key::Z),
            i.modifiers.command && (i.modifiers.shift && i.key_pressed(egui::Key::Z) || i.key_pressed(egui::Key::Y)),
            i.modifiers.command || i.modifiers.ctrl,
        )
    });
    let taken = with_edit(ws, |ws, x, pick| {
        begin(ws, x, pick);
        if undo || redo {
            edit(ws, x, if undo { nd::undo_network } else { nd::redo_network });
            return true;
        }
        let Some(d) = ws.net.draft.clone() else { return false };
        if d.placing.is_some() {
            if esc {
                edit(ws, x, nd::drop_placing);
            } else if enter {
                finish_placing(ws, x);
            } else if backspace {
                edit(ws, x, nd::unplace_corner);
            } else {
                return delete;
            }
            return true;
        }
        let h = ws.net.hovered;
        if delete || backspace {
            // What is under the cursor — or, away from the network's points and edges, the
            // patch picked, else the patch under the cursor.
            match h {
                Some(NetHit::Grip { curve, index }) => edit(ws, x, |d| nd::remove_grip(d, curve, index)),
                Some(NetHit::Node(id)) => edit(ws, x, |d| nd::remove_node(d, id)),
                Some(NetHit::Curve { curve, .. }) => edit(ws, x, |d| nd::remove_curve(d, curve)),
                Some(NetHit::Centre(cell)) => edit(ws, x, |d| nd::reset_centre(d, cell)),
                _ if d.picked.is_some() => return delete_patch(ws, x, d.picked),
                Some(NetHit::Patch { patch, .. }) => return delete_patch(ws, x, Some(patch)),
                _ => return false,
            }
            ws.net.hovered = None;
            return true;
        }
        if esc && d.picked.is_some() {
            edit(ws, x, |d| NetworkDraft { picked: None, ..d.clone() });
            return true;
        }
        if c && !command {
            if let Some(NetHit::Curve { curve, .. } | NetHit::Grip { curve, .. }) = h {
                edit(ws, x, |d| nd::toggle_crease(d, curve));
                return true;
            }
        }
        false
    });
    let taken = taken.unwrap_or(false);
    if taken {
        refresh_curves(ws);
        update_preview(ws);
        ws.request_build();
    }
    taken
}

/// Enter or a middle click with the network's box open: the patch being placed closed first,
/// never the network made under it; Create waits for a filled cell. True where it was taken.
pub(crate) fn confirm(ws: &mut ReverseWorkspace) -> bool {
    let taken = with_edit(ws, |ws, x, pick| {
        begin(ws, x, pick);
        if ws.net.draft.as_ref().is_some_and(|d| d.placing.is_some()) {
            finish_placing(ws, x);
            return true;
        }
        !network_ready(x)
    });
    let taken = taken.unwrap_or(false);
    if taken {
        refresh_curves(ws);
        update_preview(ws);
        ws.request_build();
    }
    taken
}

/// The wheel over a curve: its strip's span count, one up or down. True where it was taken
/// (the view does not zoom).
pub(crate) fn wheel(ws: &mut ReverseWorkspace, delta: f32) -> bool {
    let h = ws.net.hovered;
    let Some(NetHit::Curve { curve, .. } | NetHit::Grip { curve, .. }) = h else { return false };
    let Some(strip) = ws.net.report.as_ref().and_then(|r| r.strips.iter().find(|s| s.curves.contains(&curve))).cloned() else { return false };
    let next = (strip.spans as i64 + if delta > 0.0 { 1 } else { -1 }).clamp(SPANS_MIN as i64, SPANS_MAX as i64) as usize;
    if next != strip.spans {
        let taken = with_edit(ws, |ws, x, pick| {
            begin(ws, x, pick);
            edit(ws, x, |d| nd::set_strip_spans(d, &strip.curves, Some(next as f64)));
        });
        if taken.is_some() {
            ws.request_build();
        }
    }
    true
}

// ---- the curvature colouring ----

/// The scan coloured by its curvature while the box is open — the creases and the bends a
/// network's curves are drawn along: flats green, convex towards red, hollows towards blue,
/// sharper than the scale in dark caps, the rim bare. Read once per scan, on a thread.
fn update_curvature(ws: &mut ReverseWorkspace) {
    if let Some(rx) = &ws.net.curvature.job {
        match rx.try_recv() {
            Ok(read) => {
                ws.net.curvature.job = None;
                ws.net.curvature.colors.clear();
                let mut range = 0.0f64;
                for (id, colors, r) in read {
                    let stamp = CURVATURE_STAMP | (ws.net.curvature.key & 0xffff_ffff) << 8 | (id & 0xff);
                    ws.net.curvature.colors.insert(id, (stamp, Arc::new(colors)));
                    range = range.max(r);
                }
                ws.net.curvature.range = range;
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => return,
            Err(_) => ws.net.curvature.job = None,
        }
    }
    if ws.scan.is_empty() || ws.net.curvature.key == ws.scan_key || ws.net.curvature.job.is_some() {
        return;
    }
    ws.net.curvature.key = ws.scan_key;
    let objects: Vec<(u64, Arc<mm_core::Mesh>, [f32; 3])> = ws.scan.iter().filter(|o| o.visible).map(|o| (o.id, o.mesh.clone(), [o.color.x, o.color.y, o.color.z])).collect();
    let (tx, rx) = channel();
    ws.net.curvature.job = Some(rx);
    let ctx = ws.net.ctx.clone();
    std::thread::spawn(move || {
        let read: Vec<(u64, Vec<u32>, f64)> = objects
            .iter()
            .filter_map(|(id, mesh, base)| {
                let graph = mm_reverse::mesh_graph::MeshGraph::from_mesh(mesh.as_ref(), false).ok()?;
                let values = mm_reverse::mesh_graph::mean_curvature(&graph, mm_reverse::mesh_graph::CURVATURE_SMOOTHING);
                let range = mm_reverse::mesh_graph::curvature_range(&values, graph.bbox_diag);
                let bare = super::rgb(*base);
                let mut colors = vec![bare; mesh.max_tid()];
                for t in mesh.triangle_ids() {
                    let tri = mesh.triangle(t);
                    let vals: Vec<f64> = tri.iter().map(|v| values.get(*v as usize).copied().map_or(f64::NAN, f64::from)).collect();
                    if vals.iter().any(|v| v.is_nan()) {
                        continue;
                    }
                    if let Some(slot) = colors.get_mut(t as usize) {
                        *slot = super::ramp(vals.iter().sum::<f64>() / 3.0 / range);
                    }
                }
                Some((*id, colors, range))
            })
            .collect();
        let _ = tx.send(read);
        if let Some(c) = ctx {
            c.request_repaint();
        }
    });
}

// ---- the box ----

/// A small ⓘ beside a field: its card on hover.
fn info(ui: &mut egui::Ui, title: &str, paragraphs: &[&str]) {
    ui.label(egui::RichText::new("ⓘ").weak()).on_hover_ui(|ui| {
        ui.set_max_width(360.0);
        ui.strong(title);
        for p in paragraphs {
            ui.label(*p);
        }
    });
}

fn len(mm: f64, decimals: usize, bare: bool) -> String {
    mm_reverse::display_units::fmt_len(mm, mm_reverse::display_units::DisplayUnits::Mm, decimals, mm_reverse::display_units::FmtOptions { bare, signed: false })
}

fn signed_len(mm: f64, decimals: usize) -> String {
    mm_reverse::display_units::fmt_len(mm, mm_reverse::display_units::DisplayUnits::Mm, decimals, mm_reverse::display_units::FmtOptions { bare: false, signed: true })
}

fn warn_color() -> Color32 {
    Color32::from_rgb(240, 200, 110)
}

/// The preview line: what the ghost says of the network so far.
fn preview_line(ws: &ReverseWorkspace, ui: &mut egui::Ui) {
    let newest = ws.net.outcome.as_ref().map_or(0, |(g, _)| *g);
    if ws.building() && newest < ws.build_generation {
        ui.horizontal(|ui| {
            polymix_workspace::spinner(ui);
            ui.weak("Building the preview…");
        });
        return;
    }
    match ws.net.outcome.as_ref().map(|(_, o)| o) {
        Some(o) if o.failure.is_some() => {
            ui.colored_label(warn_color(), format!("⚠ {}", o.failure.as_deref().unwrap_or_default()));
        }
        Some(NetworkOutcome { report: Some(_), closed: true, volume, .. }) => {
            ui.weak(format!("A solid of {}: the network closes.", mm_reverse::display_units::fmt_vol(*volume, mm_reverse::display_units::DisplayUnits::Mm, 1, Default::default())));
        }
        Some(NetworkOutcome { report: Some(_), .. }) => {
            ui.weak("A sheet: the network is open — Trim, Thicken or Split take it as they take a fitted surface.");
        }
        _ => {
            ui.weak("Click the corners of a patch on the scan; the first again closes it.");
        }
    }
}

/// The open network's box (ScanRuler's `NetworkFields`): how to place patches and change them,
/// the tool a click on the scan is, Fill all, Suggest creases, the box's own undo, the patch
/// being placed, the status line, a row per patch with the picked one's strips and its Delete,
/// the regions refused and why, the curves along bodies, and the fit's settings. Returns
/// whether the network changed.
pub(crate) fn box_ui(ws: &mut ReverseWorkspace, ui: &mut egui::Ui, x: &mut NetworkFeature, pick: &mut PickWhat) -> bool {
    ws.net.ctx = Some(ui.ctx().clone());
    begin(ws, x, pick);
    // The pointer off the view: nothing of the network is under it.
    let over_view = ui.ctx().pointer_hover_pos().is_some_and(|p| !ws.view.panel_rect.is_some_and(|r| r.contains(p)));
    if !over_view && ws.net.drag.is_none() {
        ws.net.hovered = None;
    }
    take_outcomes(ws);
    poll_suggest(ws, x);
    update_curvature(ws);
    let _ = paths(ws);
    // The scan's ray casts made ready while nothing is dragged: a drag reads the scan under the
    // cursor every frame.
    let objects: Vec<super::ScanObject> = ws.scan.iter().filter(|o| o.visible).cloned().collect();
    for o in &objects {
        let _ = ws.scan_bvh(o);
    }
    let before = x.clone();
    let Some(d) = ws.net.draft.clone() else { return false };
    let report = ws.net.report.clone();
    // Click to: the tool.
    ui.horizontal(|ui| {
        ui.label("Click to");
        info(
            ui,
            "Placing patches",
            &[
                "Draw: click the corners of a patch one after the other — on the scan, or on a corner, an edge or a point already there, to join the patches. The sides follow the scan between them, and the patch shows shaded as you go. Click the first corner again, press Enter, double-click or click the middle button to close it: it is fitted at once. Four corners make a patch as it is; three, five or six are split from a centre into four-sided ones (drag its square; Delete puts it back in the middle). Backspace takes the last corner back, Esc lets the patch go; Enter with two corners makes a line. Drag an outer edge of a patch across the scan to make the next patch beside it; its new corners snap to corners near them.",
                "Changing a patch: drag a corner or a point to move it; drag an edge two patches share to bend it, a point put in where you pressed. Shift-click an edge puts a point in — on an outer edge, whose drag makes the next patch, drag that point. Click a patch to pick it: its spans come up below, and Delete (or Delete patch) takes it out with the edges no other patch has. Delete takes the point, corner or edge under the cursor out; C makes the edge under it a crease — a fold or a parting line the patches meet at an angle on; the wheel over an edge sets its strip's spans. Click a body's edge to run an edge along it.",
                "Fill: click inside a region the edges close to fill it, or on a patch to pick it; Ctrl-click a patch to empty it. Fill all fills every closed region. The curvature colouring on the scan shows where the creases and the bends are.",
                "Suggest creases finds the scan's sharp edges and folds — not its rounds — and draws them dashed: click one to take it as a crease curve, or Accept all. Two taken that meet share a node there.",
            ],
        );
        for (t, label) in [(NetworkTool::Draw, "Draw"), (NetworkTool::Fill, "Fill")] {
            if ui.selectable_label(d.tool == t, label).clicked() && d.tool != t {
                edit(ws, x, |d| NetworkDraft { tool: t, placing: if t == NetworkTool::Fill { None } else { d.placing.clone() }, ..d.clone() });
            }
        }
    });
    // The tool row.
    ui.horizontal_wrapped(|ui| {
        let unfilled = report.as_ref().map_or(0, |r| r.unfilled.len());
        let label = if unfilled > 0 { format!("Fill all ({unfilled})") } else { "Fill all".to_string() };
        if ui.add_enabled(unfilled > 0, egui::Button::new(label)).on_hover_text("Fill every region the curves close with a patch").clicked() {
            if let Some(r) = &report {
                let seeds: Vec<DVec3> = r.unfilled.iter().map(|u| u.seed).collect();
                edit(ws, x, |d| nd::add_seeds(d, &seeds));
            }
        }
        let finding = ws.net.suggest.is_some();
        if ui.add_enabled(!finding, egui::Button::new(if finding { "Finding creases…" } else { "Suggest creases" })).on_hover_text("Find the scan's sharp edges and folds, to take as crease curves").clicked() {
            start_suggest(ws);
        }
        if ui.add_enabled(!d.history.is_empty(), egui::Button::new("Undo")).on_hover_text("Undo the last edit of the network (Ctrl+Z)").clicked() {
            edit(ws, x, nd::undo_network);
        }
        if ui.add_enabled(!d.future.is_empty(), egui::Button::new("Redo")).on_hover_text("Redo (Ctrl+Y)").clicked() {
            edit(ws, x, nd::redo_network);
        }
    });
    // The creases suggested.
    if let Some(s) = &d.suggested {
        ui.horizontal_wrapped(|ui| {
            ui.weak(if s.is_empty() { "No creases found on the scan.".to_string() } else { format!("{} crease{} suggested, dashed — click one to take it.", s.len(), if s.len() == 1 { "" } else { "s" }) });
            if !s.is_empty() && ui.small_button("Accept all").on_hover_text("Take every crease suggested as a crease curve").clicked() {
                let all: Vec<usize> = (0..s.len()).collect();
                edit(ws, x, |d| nd::accept_creases(d, &all));
            }
            if ui.small_button("Dismiss").on_hover_text("Put the suggestions away").clicked() {
                edit(ws, x, |d| nd::set_suggested(d, None));
            }
        });
    }
    // The patch being placed.
    if let Some(p) = &d.placing {
        let n = p.len();
        let next = if n >= 3 {
            "click the first corner, press Enter, double-click or middle-click to close the patch;"
        } else if n == 2 {
            "one more for a patch, or Enter for a line;"
        } else {
            "click the next corner;"
        };
        ui.weak(format!("{n} corner{} — {next} Backspace takes the last back, Esc lets it go.", if n == 1 { "" } else { "s" }));
    }
    // The status line.
    let status = nd::network_status(report.as_ref(), d.data.curves.len(), d.placing.is_some(), mm_reverse::display_units::DisplayUnits::Mm);
    if !status.is_empty() {
        ui.weak(status);
    }
    if d.placing.is_none() && d.tool == NetworkTool::Draw && report.as_ref().is_some_and(|r| !r.patches.is_empty()) {
        ui.weak("Drag a corner or an edge to change a patch; click a patch to pick it.");
    }
    // A row per patch.
    if let Some(r) = report.as_ref().filter(|r| !r.patches.is_empty()) {
        let tolerance = d.data.fit.tolerance.unwrap_or(DEFAULT_TOLERANCE);
        for (i, p) in r.patches.iter().enumerate() {
            let worst: Vec<String> = p.worst.iter().map(|w| format!("{} at ({}, {}, {})", signed_len(w.off, 3), len(w.at.x, 1, true), len(w.at.y, 1, true), len(w.at.z, 1, true))).collect();
            let tip = format!("Off the scan beyond its noise: {} (the tolerance {}). The patch's worst points: {}", len(p.misfit, 3, false), len(tolerance, 3, true), worst.join("; "));
            let text = egui::RichText::new(format!("Patch {} · σ {} · worst {} · {} × {}", i + 1, len(p.rms, 3, true), len(p.max, 3, false), p.net.u, p.net.v)).size(11.0);
            if ui.selectable_label(d.picked == Some(i), text).on_hover_text(tip).clicked() {
                edit(ws, x, |d| NetworkDraft { picked: if d.picked == Some(i) { None } else { Some(i) }, ..d.clone() });
            }
        }
    }
    // The picked patch: its strips, and its Delete.
    if let (Some(r), Some(picked)) = (&report, d.picked) {
        if let Some(p) = r.patches.get(picked) {
            for (label, side) in [("Spans along", 0usize), ("Spans across", 1)] {
                let Some(strip) = nd::strip_of(&r.strips, p.sides.get(side).and_then(|s| s.first()).copied()).cloned() else { continue };
                ui.horizontal(|ui| {
                    ui.label(label);
                    let mut auto = strip.auto;
                    if ui.checkbox(&mut auto, "Auto").on_hover_text("The strip's span count chosen by the fit: 4, and more only where a patch on it misses the scan by more than the tolerance beyond its noise, while that helps — up to 20").changed() {
                        let spans = if auto { None } else { Some(strip.spans as f64) };
                        edit(ws, x, |d| nd::set_strip_spans(d, &strip.curves, spans));
                    }
                    let mut n = strip.spans as f64;
                    if ui.add_enabled(!strip.auto, egui::DragValue::new(&mut n).range(SPANS_MIN as f64..=SPANS_MAX as f64).speed(0.1).max_decimals(0)).changed() {
                        let n = mm_reverse::js_number::js_round(n).clamp(SPANS_MIN as f64, SPANS_MAX as f64);
                        edit(ws, x, |d| nd::set_strip_spans(d, &strip.curves, Some(n)));
                    }
                    ui.label("spans");
                });
            }
            if ui.button(format!("Delete patch {}", picked + 1)).on_hover_text("Take the patch out, with the edges no other patch has (Delete)").clicked() {
                let r = r.clone();
                edit(ws, x, |d| nd::remove_reported_patch(d, &r, picked));
            }
        }
    }
    // The regions refused, and why.
    if let Some(r) = &report {
        for (i, c) in r.cells.iter().enumerate() {
            let CellOutcome::Refused { seed, refused, .. } = c else { continue };
            ui.horizontal_wrapped(|ui| {
                ui.colored_label(warn_color(), format!("⚠ Region at ({}, {}, {}): {refused}", len(seed.x, 1, true), len(seed.y, 1, true), len(seed.z, 1, true)));
                if ui.small_button("×").on_hover_text("Empty the region: take its seed away").clicked() {
                    edit(ws, x, |d| nd::remove_seed(d, i));
                }
            });
        }
    }
    body_sides(ws, ui, x, report.as_ref());
    // The fit's settings.
    ui.horizontal(|ui| {
        ui.label("Smoothness");
        info(ui, "Smoothness", &["How hard the patches and their edges are held fair against the scan: 0 follows it exactly, 0.001 fairs its noise, 0.1 smooths well over it, from 1 detail goes"]);
        let mut s = d.data.fit.smoothness;
        let r = ui.add(egui::DragValue::new(&mut s).speed(0.0001).range(0.0..=f64::MAX).max_decimals(6));
        fit_field(ws, x, &r, |f| f.smoothness = s.abs());
    });
    ui.horizontal(|ui| {
        ui.label("Tolerance");
        info(ui, "Tolerance", &["How far a patch may stray from the scan beyond its noise before Auto gives its strips more spans: they start at 4 and get more only where a patch misses by more than this, and only while that helps. Fewer spans are smoother"]);
        let mut t = d.data.fit.tolerance.unwrap_or(DEFAULT_TOLERANCE);
        let r = ui.add(egui::DragValue::new(&mut t).speed(0.001).range(0.0..=f64::MAX).max_decimals(4).suffix(" mm"));
        fit_field(ws, x, &r, |f| f.tolerance = Some(t.abs()));
    });
    let mut robust = d.data.fit.robust == Some(true);
    if ui.checkbox(&mut robust, "Leave out stray points").on_hover_text("Fit again without the points past three sigmas of the first fit — a rib or a stray edge caught in a cell").changed() {
        edit(ws, x, |d| nd::set_network_fit(d, |f| f.robust = Some(robust)));
    }
    preview_line(ws, ui);
    // The curvature colouring's scale, while it is shown.
    if !ws.deviation.on && ws.net.curvature.range > 0.0 {
        ui.weak(format!("Curvature on the scan: full colour at R {}", len(1.0 / (2.0 * ws.net.curvature.range), 1, false)));
    }
    if ws.scan.is_empty() {
        ui.colored_label(warn_color(), "⚠ A patch network is drawn on a scan: open one in the Mesh workspace.");
    } else if ws.net.paths.as_ref().is_some_and(|(_, c)| c.get().is_none()) {
        ui.horizontal(|ui| {
            polymix_workspace::spinner(ui);
            ui.weak("Reading the scan for paths…");
        });
    }
    let changed = *x != before;
    if changed {
        refresh_curves(ws);
        ws.net.pending_build = false;
    } else if ws.net.curves.is_empty() && !x.curves.is_empty() {
        refresh_curves(ws);
    }
    update_preview(ws);
    changed || build_due(ws)
}

/// A number of the fit changed in its field: typed, one step of the undo; dragged, the whole
/// drag one step, the preview built as it goes.
fn fit_field(ws: &mut ReverseWorkspace, x: &mut NetworkFeature, r: &egui::Response, change: impl FnOnce(&mut mm_reverse::model::NetworkFitParams)) {
    if r.drag_started() {
        ws.net.fit_drag = ws.net.draft.as_ref().map(|d| d.data.clone());
    }
    if r.changed() {
        if r.dragged() && ws.net.fit_drag.is_some() {
            edit(ws, x, |d| {
                let mut data = d.data.clone();
                change(&mut data.fit);
                NetworkDraft { data, ..d.clone() }
            });
        } else {
            edit(ws, x, |d| nd::set_network_fit(d, change));
        }
    }
    if r.drag_stopped() {
        if let Some(before) = ws.net.fit_drag.take() {
            if ws.net.draft.as_ref().is_some_and(|d| d.data != before) {
                edit(ws, x, |d| nd::record_drag(d, &before));
            }
        }
    }
}

/// The faces beside a body's edge, as the body was last built: the edge found by its
/// fingerprint, its two faces with theirs.
fn faces_beside(ws: &ReverseWorkspace, body: BodyId, edge: &mm_reverse::model::EdgeRef) -> Vec<FaceFingerprint> {
    let Some(view) = ws.pick_view(body).or_else(|| ws.body_view(body)) else { return Vec::new() };
    let t = &view.topology;
    let diagonal = t.faces.iter().map(|f| f.fingerprint.centroid.length()).fold(1.0, f64::max);
    let Some(e) = match_edge(edge, &t.edge_prints(), &match_tolerance(diagonal)).and_then(|i| t.edges.get(i)) else { return Vec::new() };
    e.faces.iter().filter_map(|h| t.face(*h)).map(|f| f.fingerprint).collect()
}

/// The curves along a body's edge (PN11), a row each: the patch beside it meeting the body's
/// face on its edge (Contact), or tangent to the face too — and which of the edge's two faces.
fn body_sides(ws: &mut ReverseWorkspace, ui: &mut egui::Ui, x: &mut NetworkFeature, report: Option<&NetworkReport>) {
    let along: Vec<(u32, BodyId, mm_reverse::model::EdgeRef, Option<CurveSide>)> = x.curves.iter().filter_map(|c| c.along.as_ref().map(|a| (c.id, a.body, a.edge.clone(), c.side.clone()))).collect();
    for (id, body, edge, side) in along {
        let faces = faces_beside(ws, body, &edge);
        let reading = report.and_then(|r| r.bodies.iter().find(|b| b.curve == id));
        let name = ws.body_view(body).or_else(|| ws.pick_view(body)).map_or(format!("body {}", body.as_number()), |b| b.name.clone());
        let which = side.as_ref().and_then(|s| faces.iter().position(|f| f.centroid == s.face.centroid).map(|i| (s.mode, i)));
        let label_of = |w: Option<(SideMode, usize)>| match w {
            None => "On the edge, no face".to_string(),
            Some((SideMode::Contact, i)) => format!("Contact — {}", faces.get(i).map(describe_face).unwrap_or_default()),
            Some((SideMode::Tangent, i)) => format!("Tangent to {}", faces.get(i).map(describe_face).unwrap_or_default()),
        };
        let mut chosen = which;
        ui.horizontal(|ui| {
            ui.label(format!("Along {name}{}", reading.map_or(String::new(), |r| format!(" · {}°", mm_reverse::js_number::to_fixed(r.angle, 2)))));
            egui::ComboBox::from_id_salt(("re_network_side", id))
                .width(ui.available_width().clamp(80.0, 200.0))
                .truncate()
                .selected_text(label_of(which))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut chosen, None, label_of(None));
                    for i in 0..faces.len() {
                        ui.selectable_value(&mut chosen, Some((SideMode::Contact, i)), label_of(Some((SideMode::Contact, i))));
                    }
                    for i in 0..faces.len() {
                        ui.selectable_value(&mut chosen, Some((SideMode::Tangent, i)), label_of(Some((SideMode::Tangent, i))));
                    }
                })
                .response
                .on_hover_text("How the patch beside this curve meets the body: on the body's edge (Contact), or tangent to one of the edge's two faces as well");
        });
        if chosen != which {
            let side = chosen.and_then(|(mode, i)| faces.get(i).or(faces.first()).map(|f| CurveSide { body, face: *f, mode }));
            edit(ws, x, |d| nd::set_side(d, id, side));
        }
    }
}

// ---- suggested creases ----

/// The scan's creases found on a thread of its own, and offered on the draft when they come.
fn start_suggest(ws: &mut ReverseWorkspace) {
    if ws.net.suggest.is_some() {
        return;
    }
    // The scan read for paths there, if it is not yet: a click while it is being read waits for
    // it rather than doing nothing.
    let cell = paths_cell(ws);
    let parts = ws.scan_parts();
    let (tx, rx) = channel();
    ws.net.suggest = Some(rx);
    let ctx = ws.net.ctx.clone();
    std::thread::spawn(move || {
        let found = cell.get_or_init(|| merged_paths(&parts)).as_deref().map(suggested).unwrap_or_default();
        let _ = tx.send(found);
        if let Some(c) = ctx {
            c.request_repaint();
        }
    });
}

/// The scan's creases as a network's suggestions: each its ends' points, its grips, its
/// samples, its length and its angle.
fn suggested(paths: &ScanPaths) -> Vec<SuggestedCrease> {
    let found = suggest_creases(paths, &CreaseOptions::default());
    let at = |id: u32| found.nodes.iter().find(|n| n.id == id).map(|n| n.at);
    found
        .curves
        .iter()
        .filter_map(|c| Some(SuggestedCrease { from: at(c.from)?, to: at(c.to)?, via: c.via.clone(), points: c.points.clone(), length: c.length, angle: c.angle }))
        .collect()
}

fn poll_suggest(ws: &mut ReverseWorkspace, x: &mut NetworkFeature) {
    let Some(rx) = &ws.net.suggest else { return };
    match rx.try_recv() {
        Ok(found) => {
            ws.net.suggest = None;
            edit(ws, x, |d| nd::set_suggested(d, Some(found)));
        }
        Err(std::sync::mpsc::TryRecvError::Empty) => {}
        Err(_) => ws.net.suggest = None,
    }
}

// ---- painting ----

fn mark(painter: &egui::Painter, at: Pos2, size: f32, square: bool) {
    let r = size / 2.0;
    if square {
        painter.rect_filled(egui::Rect::from_center_size(at, egui::vec2(size, size)), 0.0, MARK_RIM);
        painter.rect_filled(egui::Rect::from_center_size(at, egui::vec2(size * 18.0 / 32.0, size * 18.0 / 32.0)), 0.0, Color32::WHITE);
    } else {
        painter.circle_filled(at, r, MARK_RIM);
        painter.circle_filled(at, r * 9.0 / 16.0, Color32::WHITE);
    }
}

fn polyline(painter: &egui::Painter, screen: &Projector, points: &[DVec3], width: f32, color: Color32, dashed: bool) {
    let pts: Vec<Pos2> = points.iter().filter_map(|p| screen.at(*p)).collect();
    if pts.len() < 2 {
        return;
    }
    let stroke = Stroke::new(width, color);
    if dashed {
        painter.extend(egui::Shape::dashed_line(&pts, stroke, 4.0, 3.0));
    } else {
        painter.add(egui::Shape::line(pts, stroke));
    }
}

/// A loop's area shaded: a fan from its middle, see-through, over the scan — what the patch
/// will cover.
fn shade(painter: &egui::Painter, screen: &Projector, outline: &[DVec3]) {
    if outline.len() < 3 {
        return;
    }
    let c = outline.iter().copied().sum::<DVec3>() / outline.len() as f64;
    let Some(centre) = screen.at(c) else { return };
    let color = BODY_COLOR.gamma_multiply(PREVIEW_OPACITY);
    let mut mesh = egui::Mesh::default();
    mesh.colored_vertex(centre, color);
    let pts: Vec<Option<Pos2>> = outline.iter().map(|p| screen.at(*p)).collect();
    for p in &pts {
        mesh.colored_vertex(p.unwrap_or(centre), color);
    }
    let m = outline.len() as u32;
    for i in 0..m {
        mesh.add_triangle(0, 1 + i, 1 + (i + 1) % m);
    }
    painter.add(egui::Shape::mesh(mesh));
}

/// What the network's box paints over the view: the layer.
pub(crate) fn paint(ws: &ReverseWorkspace, painter: &egui::Painter, vp: Viewport, x: &NetworkFeature) {
    let Some(d) = draft(ws) else { return };
    let screen = Projector::of(ws, vp);
    let report = ws.net.report.as_ref();
    let h = ws.net.hovered;
    let lit_curve = match (&ws.net.extending, h) {
        (Some(e), _) => Some(e.curve),
        (None, Some(NetHit::Curve { curve, .. } | NetHit::Grip { curve, .. })) => Some(curve),
        _ => None,
    };
    // The patches waiting for their fit, and the one being placed or dragged out, shaded: under
    // the curves and the points, as ScanRuler's layer orders them.
    for w in &ws.net.waiting {
        shade(painter, &screen, &w.outline);
    }
    let preview = ws.net.preview.as_ref().filter(|_| d.tool == NetworkTool::Draw);
    if let Some(pv) = preview.filter(|pv| pv.lines.len() >= 3) {
        let outline: Vec<DVec3> = pv.lines.iter().flat_map(|l| l.0.iter().copied()).collect();
        shade(painter, &screen, &outline);
    }
    // A split cell's spokes, thin and dashed.
    for spoke in report.iter().flat_map(|r| r.spokes.iter()) {
        polyline(painter, &screen, &as_points(&spoke.points), LINE, CURVE_COLOR, true);
    }
    // The curves: each its fit, sampled.
    for c in &x.curves {
        let Some(drawn) = ws.net.curves.get(&c.id) else { continue };
        let lit = lit_curve == Some(c.id);
        let color = if lit {
            LIT_COLOR
        } else if c.along.is_some() {
            ALONG_COLOR
        } else {
            CURVE_COLOR
        };
        polyline(painter, &screen, &drawn.points, LINE + if lit { 2.0 } else { 1.0 }, color, c.crease == Some(true));
    }
    // The creases offered.
    for (i, s) in d.suggested.iter().flatten().enumerate() {
        let lit = h == Some(NetHit::Suggested(i));
        polyline(painter, &screen, &as_points(&s.points), LINE + if lit { 2.0 } else { 1.0 }, if lit { LIT_COLOR } else { SUGGESTED_COLOR }, true);
    }
    // The picked patch's outline, and the one under the cursor.
    if let (Some(r), Some(picked)) = (report, d.picked) {
        if let Some(p) = r.patches.get(picked) {
            polyline(painter, &screen, &ring_of(&p.grid), LINE + 2.0, PICKED_COLOR, false);
        }
    }
    if let (Some(r), Some(NetHit::Patch { patch, .. })) = (report, h) {
        if d.picked != Some(patch) {
            if let Some(p) = r.patches.get(patch) {
                polyline(painter, &screen, &ring_of(&p.grid), LINE + 1.0, LIT_COLOR, false);
            }
        }
    }
    // The patch being placed: its sides along the scan, white, the side back to the first
    // dashed.
    for (l, dashed) in preview.iter().flat_map(|pv| pv.lines.iter()) {
        polyline(painter, &screen, l, LINE + 1.0, LIT_COLOR, *dashed);
    }
    // The nodes, the corners of the patches, and the grips.
    let corners: std::collections::HashSet<u32> = report.iter().flat_map(|r| r.patches.iter().flat_map(|p| p.corners.iter().copied())).chain(x.cells.iter().flat_map(|c| c.corners.iter().flatten().copied())).collect();
    for n in &x.nodes {
        if let Some(s) = screen.at(n.at) {
            let lit = h == Some(NetHit::Node(n.id));
            if corners.contains(&n.id) {
                mark(painter, s, if lit { 14.0 } else { 11.0 }, true);
            } else {
                mark(painter, s, if lit { 13.0 } else { 10.0 }, false);
            }
        }
    }
    for c in &x.curves {
        for (i, q) in c.via.iter().enumerate() {
            if let Some(s) = screen.at(*q) {
                let lit = h == Some(NetHit::Grip { curve: c.id, index: i });
                mark(painter, s, if lit { 10.0 } else { 7.0 }, false);
            }
        }
    }
    for c in report.iter().flat_map(|r| r.centres.iter()) {
        if let Some(s) = screen.at(c.at) {
            mark(painter, s, 11.0, true);
        }
    }
    // The corners of the patch being placed, over all of it.
    for c in preview.iter().flat_map(|pv| pv.corners.iter()) {
        if let Some(s) = screen.at(*c) {
            mark(painter, s, 11.0, true);
        }
    }
    // The refusals, by their seeds.
    for (i, cell) in x.cells.iter().enumerate() {
        let Some(CellOutcome::Refused { refused, .. }) = report.and_then(|r| r.cells.get(i)) else { continue };
        let Some(s) = screen.at(cell.seed) else { continue };
        let font = egui::FontId::proportional(11.0);
        let head = painter.layout_no_wrap("Not filled".into(), egui::FontId::proportional(11.0), Color32::WHITE);
        let body = painter.layout(refused.clone(), font, Color32::from_gray(235), 220.0);
        let w = head.size().x.max(body.size().x) + 10.0;
        let hgt = head.size().y + body.size().y + 8.0;
        let rect = egui::Rect::from_min_size(s + egui::vec2(8.0, -hgt / 2.0), egui::vec2(w, hgt));
        painter.rect_filled(rect, 3.0, Color32::from_black_alpha(200));
        painter.rect_stroke(rect, 3.0, Stroke::new(1.0, REFUSED_COLOR), egui::StrokeKind::Inside);
        painter.circle_filled(s, 4.0, REFUSED_COLOR);
        let below = head.size().y + 5.0;
        painter.galley(rect.min + egui::vec2(5.0, 3.0), head, REFUSED_COLOR);
        painter.galley(rect.min + egui::vec2(5.0, below), body, Color32::from_gray(235));
    }
}

/// The patch being placed — its corners, its sides along the scan from corner to corner and on
/// to the cursor, the side back to the first dashed — or the edge being dragged out, as the layer
/// draws it (ScanRuler's `drawPreview`): traced again only when what it is drawn for changed,
/// the sides between corners placed kept traced and the ones to the cursor made afresh.
fn update_preview(ws: &mut ReverseWorkspace) {
    let key = match draft(ws) {
        Some(d) if d.tool == NetworkTool::Draw && (ws.net.extending.is_some() || d.placing.is_some()) => PreviewKey {
            hovered: ws.net.hovered,
            placed: d.placing.iter().flatten().filter_map(|c| nd::corner_point(&d.data, c)).collect(),
            first: d.placing.as_ref().and_then(|p| p.first().copied()),
            extending: ws.net.extending.as_ref().map(|e| (e.curve, e.from, e.to)),
            scan: ws.scan_key,
        },
        _ => {
            ws.net.preview = None;
            return;
        }
    };
    if ws.net.preview.as_ref().is_some_and(|p| p.key == key) {
        return;
    }
    let Some(paths) = paths(ws) else {
        ws.net.preview = None;
        return;
    };
    let mut lines: Vec<(Vec<DVec3>, bool)> = Vec::new();
    let mut corners: Vec<DVec3> = Vec::new();
    if key.extending.is_some() {
        // The edge dragged out: the patch it makes, its own edge dashed.
        if let (Some(q), Some(d)) = (&ws.net.extension, draft(ws)) {
            lines = q.sides.iter().enumerate().map(|(i, s)| (s.clone(), i == 0)).collect();
            corners = q.corners.iter().filter_map(|c| nd::corner_point(&d.data, c)).collect();
        }
    } else if !key.placed.is_empty() {
        let placed = key.placed.clone();
        let h = key.hovered;
        let closing = matches!(h, Some(NetHit::Placed(0))) || matches!((h, key.first), (Some(NetHit::Node(n)), Some(PlacedCorner::Node(f))) if n == f);
        let cursor = if closing { None } else { h.and_then(|h| hit_point(ws, &h)) };
        let mut points = placed.clone();
        if let Some(c) = cursor.filter(|c| Some(c) != placed.last()) {
            points.push(c);
        }
        for i in 0..points.len().saturating_sub(1) {
            if let Some(side) = side_between(ws, &paths, &[points[i], points[i + 1]], i + 1 < placed.len()) {
                lines.push((side, false));
            }
        }
        if points.len() >= 3 {
            if let Some(back) = side_between(ws, &paths, &[points[points.len() - 1], points[0]], false) {
                lines.push((back, !closing));
            }
        }
        corners = placed;
    }
    ws.net.preview = Some(Preview { key, lines, corners });
}

// ---- script steps (tests) ----

/// A network script step (`net:<what>[:…]`), for tests run without a person at the mouse:
/// `new`, `corner:x:y:z` (a corner on the scan nearest the point, part coordinates; a node
/// within half a millimetre is that node), `edge:x:y:z` (a curve along the body edge nearest the
/// point), `line:x0:y0:z0:x1:y1:z1`, `close`
/// (Enter), `fillall`, `suggest`, `accept`, `pick:i`, `spans:<curve>:<n|auto>`,
/// `crease:<curve>`, `delete`, `undo`, `redo`, `smooth:v`, `tolerance:v`, `robust:0|1`,
/// `tool:draw|fill`, `create`, `report`.
pub(crate) fn script(ws: &mut ReverseWorkspace, args: &[&str]) {
    let num = |i: usize| args.get(i).and_then(|v| v.parse::<f64>().ok());
    match args.first().copied() {
        Some("new") => {
            if let Ok(k) = super::editors::new_feature("network", ws) {
                ws.open_feature(k, None);
            }
        }
        Some("report") => {
            let d = ws.net.draft.clone();
            let status = nd::network_status(ws.net.report.as_ref(), d.as_ref().map_or(0, |d| d.data.curves.len()), d.as_ref().is_some_and(|d| d.placing.is_some()), mm_reverse::display_units::DisplayUnits::Mm);
            let failure = ws.net.outcome.as_ref().and_then(|(_, o)| o.failure.clone()).unwrap_or_default();
            let rows: Vec<String> = ws.net.report.iter().flat_map(|r| r.patches.iter().map(|p| format!("σ {:.4} worst {:.4} net {}×{}", p.rms, p.max, p.net.u, p.net.v))).collect();
            log::info!("reverse network: {status} | {failure} | {}", rows.join("; "));
            // The draft itself, for tests: what the edits made of it.
            if let Some(d) = &d {
                let creases = d.data.curves.iter().filter(|c| c.crease == Some(true)).count();
                let vias: usize = d.data.curves.iter().map(|c| c.via.len()).sum();
                let spans: Vec<String> = d.data.spans.iter().map(|(k, v)| format!("{k}={v}")).collect();
                log::info!(
                    "reverse network draft: nodes {} curves {} (drawn {}, lost {:?}) creases {} grips {} cells {} spans [{}] placing {} picked {:?} tool {:?} history {} future {} suggested {:?} hovered {:?}",
                    d.data.nodes.len(),
                    d.data.curves.len(),
                    ws.net.curves.len(),
                    ws.net.report.as_ref().map(|r| r.lost.clone()).unwrap_or_default(),
                    creases,
                    vias,
                    d.data.cells.len(),
                    spans.join(","),
                    d.placing.as_ref().map_or(0, Vec::len),
                    d.picked,
                    d.tool,
                    d.history.len(),
                    d.future.len(),
                    d.suggested.as_ref().map(Vec::len),
                    ws.net.hovered
                );
            }
        }
        Some("create") => {
            if !confirm(ws) {
                ws.commit_feature();
            }
        }
        Some(step) => {
            let step = step.to_string();
            let paths = paths_now(ws);
            let _ = with_edit(ws, |ws, x, pick| {
                begin(ws, x, pick);
                match step.as_str() {
                    "corner" => {
                        let (Some(px), Some(py), Some(pz), Some(paths)) = (num(1), num(2), num(3), &paths) else { return };
                        let Some(q) = paths.closest(DVec3::new(px, py, pz)) else { return };
                        // A node of the network within half a millimetre is that node, as a click
                        // within NODE_PX of it is.
                        let near = ws.net.draft.as_ref().and_then(|d| d.data.nodes.iter().find(|n| (n.at - q.point).length() <= 0.5).map(|n| n.id));
                        let corner = near.map_or(PlacedCorner::At(q.point), PlacedCorner::Node);
                        edit(ws, x, |d| nd::place_corner(d, corner));
                    }
                    "close" => finish_placing(ws, x),
                    "edge" => {
                        // A body's edge nearest the point, as a click on it takes it.
                        let (Some(px), Some(py), Some(pz)) = (num(1), num(2), num(3)) else { return };
                        let q = DVec3::new(px, py, pz);
                        let mut best: Option<(f64, BodyId, usize)> = None;
                        for b in ws.pick_set() {
                            for (i, e) in b.topology.edges.iter().enumerate() {
                                let d = e.polyline.windows(2).map(|w| {
                                    let (a, c) = (w[0], w[1]);
                                    let t = if (c - a).length_squared() > 0.0 { ((q - a).dot(c - a) / (c - a).length_squared()).clamp(0.0, 1.0) } else { 0.0 };
                                    (a + (c - a) * t - q).length()
                                }).fold(f64::INFINITY, f64::min);
                                if best.is_none_or(|(bd, _, _)| d < bd) {
                                    best = Some((d, b.id, i));
                                }
                            }
                        }
                        let Some((_, body, edge)) = best else { return };
                        let Some(view) = ws.pick_view(body) else { return };
                        let Some(e) = view.topology.edges.get(edge) else { return };
                        let face = e.faces.iter().find_map(|h| view.topology.face(*h)).map(|f| f.fingerprint);
                        let (print, ends) = (e.fingerprint.clone(), e.ends);
                        edit(ws, x, |d| nd::add_edge_curve(d, body, &print, ends, face));
                    }
                    "fillall" => {
                        let seeds: Vec<DVec3> = ws.net.report.iter().flat_map(|r| r.unfilled.iter().map(|u| u.seed)).collect();
                        edit(ws, x, |d| nd::add_seeds(d, &seeds));
                    }
                    "suggest" => {
                        if let Some(paths) = &paths {
                            let found = suggested(paths);
                            edit(ws, x, |d| nd::set_suggested(d, Some(found)));
                        }
                    }
                    "accept" => {
                        let n = ws.net.draft.as_ref().and_then(|d| d.suggested.as_ref()).map_or(0, Vec::len);
                        let all: Vec<usize> = (0..n).collect();
                        edit(ws, x, |d| nd::accept_creases(d, &all));
                    }
                    "pick" => {
                        let i = num(1).map(|v| v as usize);
                        edit(ws, x, |d| NetworkDraft { picked: i, ..d.clone() });
                    }
                    "spans" => {
                        let Some(curve) = num(1).map(|v| v as u32) else { return };
                        let strip = ws.net.report.as_ref().and_then(|r| r.strips.iter().find(|s| s.curves.contains(&curve))).map(|s| s.curves.clone()).unwrap_or_else(|| vec![curve]);
                        let n = num(2);
                        edit(ws, x, |d| nd::set_strip_spans(d, &strip, n));
                    }
                    "crease" => {
                        if let Some(curve) = num(1).map(|v| v as u32) {
                            edit(ws, x, |d| nd::toggle_crease(d, curve));
                        }
                    }
                    "delete" => {
                        let picked = ws.net.draft.as_ref().and_then(|d| d.picked);
                        delete_patch(ws, x, picked);
                    }
                    "undo" => edit(ws, x, nd::undo_network),
                    "redo" => edit(ws, x, nd::redo_network),
                    "smooth" => {
                        if let Some(v) = num(1) {
                            edit(ws, x, |d| nd::set_network_fit(d, |f| f.smoothness = v.abs()));
                        }
                    }
                    "tolerance" => {
                        if let Some(v) = num(1) {
                            edit(ws, x, |d| nd::set_network_fit(d, |f| f.tolerance = Some(v.abs())));
                        }
                    }
                    "robust" => {
                        let on = args.get(1) == Some(&"1");
                        edit(ws, x, |d| nd::set_network_fit(d, |f| f.robust = Some(on)));
                    }
                    "tool" => {
                        let t = if args.get(1) == Some(&"fill") { NetworkTool::Fill } else { NetworkTool::Draw };
                        edit(ws, x, |d| NetworkDraft { tool: t, placing: None, ..d.clone() });
                    }
                    "line" => {
                        let c: Vec<f64> = (1..=6).filter_map(num).collect();
                        let (6, Some(paths)) = (c.len(), &paths) else { return };
                        let (Some(p), Some(q)) = (paths.closest(DVec3::new(c[0], c[1], c[2])), paths.closest(DVec3::new(c[3], c[4], c[5]))) else { return };
                        edit(ws, x, |d| nd::add_curve(d, NetworkStart::At(p.point), &[], NetworkStart::At(q.point)));
                    }
                    other => log::warn!("unknown patch network step {other}"),
                }
            });
            refresh_curves(ws);
            ws.request_build();
        }
        None => log::warn!("a patch network step needs a word"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The scan's objects read as one mesh for paths: each object's live vertices once, its
    /// triangles on them, whatever slots its ids leave dead.
    #[test]
    fn merges_the_scan_objects_into_one_mesh_for_paths() {
        let square = |x0: f64, dead: usize| {
            let mut positions = vec![DVec3::splat(f64::NAN); dead];
            positions.extend([DVec3::new(x0, 0.0, 0.0), DVec3::new(x0 + 10.0, 0.0, 0.0), DVec3::new(x0 + 10.0, 10.0, 0.0), DVec3::new(x0, 10.0, 0.0)]);
            let v = |k: usize| (dead + k) as u32;
            ScanPart { id: x0 as u64, positions, triangles: vec![[v(0), v(1), v(2)], [v(0), v(2), v(3)]], tri_ids: vec![0, 1], max_tid: 2, live_vertices: (0..4).map(v).collect() }
        };
        let paths = merged_paths(&[square(0.0, 0), square(20.0, 3)]).expect("a mesh");
        assert_eq!(paths.vertex_count(), 8);
        assert_eq!(paths.triangle_count(), 4);
        assert!(paths.positions().iter().all(|x| x.is_finite()));
        let near = paths.closest(DVec3::new(25.0, 5.0, 1.0)).expect("a point");
        assert!((near.point - DVec3::new(25.0, 5.0, 0.0)).length() < 1e-6);
        assert!(merged_paths(&[]).is_none());
    }
}
