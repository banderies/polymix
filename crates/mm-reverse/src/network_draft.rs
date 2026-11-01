// SPDX-License-Identifier: AGPL-3.0-only
//! A patch network's draft edited: a patch placed by its corners — on the scan, on the
//! network's nodes, curves and grips — and filled at once, an edge dragged out into the next
//! patch, curves made between two points, grips moved, put in and taken out, a curve made a
//! crease, a strip's span count set, a region filled, a patch deleted — each a pure step from
//! one draft to the next, the one before kept for the editor's own undo. The viewport's layer
//! and the box edit through these.
//!
//! Ported from ScanRuler (`plugins/reverse/state/reNetworkDraft.ts`, the draft's types of
//! `plugins/reverse/state/reTypes.ts`, `patchUnder` and `ringOf` of
//! `plugins/reverse/viewer/networkLayer.ts`, and `networkStatus` of
//! `plugins/reverse/ui/NetworkEditor.tsx`), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! The network's data is the feature as the part holds it ([`NetworkFeature`]): the draft
//! is that and what the box keeps beside it while it is open — the next id, the patch being
//! placed, the history, the patch picked, the tool and the creases suggested.

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use glam::DVec3;

use crate::display_units::{fmt_len, DisplayUnits, FmtOptions};
use crate::js_number::to_fixed;
use crate::model::{BodyId, CurveAlong, CurveSide, EdgeRef, FaceRef, NetworkCell, NetworkCurve, NetworkFeature, NetworkFitParams, NetworkNode, NodeOn, SideMode};
use crate::network::CellOutcome;
use crate::network_body::{NetworkReport, PATCH_GRID};
use crate::network_fit::Strip;

/// The smoothness a new network opens with. A tenth of Fit Surface's: what it fairs is the
/// departure from the Coons surface of a patch's sides, and on a large, strongly bent patch
/// that surface is not the part's — six patches round a sphere of R 40 at 0.02 mm noise, 49 mm
/// a side with 120° corners, came out 0.040 mm RMS off the true sphere at 0.01 (0.17 at
/// worst) and 0.012 at 0.001 (0.065). At 0 the corners of three open to 13°.
pub const DEFAULT_SMOOTHNESS: f64 = 0.001;

/// How many edits the editor's undo keeps.
const HISTORY: usize = 200;

/// A chain of curves already joining two corners is a patch's side when it is no longer than
/// this times their distance apart — the two halves of a side a neighbour's corner split, not
/// the way round another patch — and of at most CHAIN_CURVES curves.
const CHAIN_SLACK: f64 = 1.25;
const CHAIN_CURVES: usize = 4;

/// How near a node a body edge's end, or a suggested crease's, may be and be that node, mm.
pub const EDGE_NODE_MM: f64 = 0.5;

/// The fit a new network opens with.
pub fn default_network_fit() -> NetworkFitParams {
    NetworkFitParams { smoothness: DEFAULT_SMOOTHNESS, robust: None, tolerance: None }
}

/// A network with nothing in it yet, as the box opens a new one.
pub fn empty_network() -> NetworkFeature {
    NetworkFeature { nodes: Vec::new(), curves: Vec::new(), cells: Vec::new(), spans: BTreeMap::new(), fit: default_network_fit() }
}

/// Where a curve starts or ends: a node of the network, or a point of the scan that becomes
/// one when the curve is made.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NetworkStart {
    Node(u32),
    At(DVec3),
}

/// A patch's corner as it is placed: a node of the network, a point of the scan, a point on a
/// curve (split there with a node when the patch is made) or a curve's grip (made a node
/// there).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PlacedCorner {
    Node(u32),
    At(DVec3),
    Curve { curve: u32, at: DVec3 },
    Grip { curve: u32, grip: usize, at: DVec3 },
}

/// What a click on the scan does: place a patch's corner, or fill the region it lands in (or
/// pick the patch filling it).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NetworkTool {
    #[default]
    Draw,
    Fill,
}

/// A crease the scan suggests: a crease curve's two ends and its grips — two suggestions that
/// meet share an end's point — with the fitted curve sampled to draw it (x, y, z each), its
/// length (mm) and how far the scan folds along it (degrees).
#[derive(Debug, Clone, PartialEq)]
pub struct SuggestedCrease {
    pub from: DVec3,
    pub to: DVec3,
    pub via: Vec<DVec3>,
    pub points: Vec<f32>,
    pub length: f64,
    pub angle: f64,
}

/// Which point of a cell a drag moves: the point inside it, or a split cell's centre.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellPoint {
    Seed,
    Centre,
}

/// A patch network being drawn: the network as the feature holds it, the patch being placed,
/// the editor's own undo, the patch picked in the readout, the tool and the creases offered.
#[derive(Debug, Clone, PartialEq)]
pub struct NetworkDraft {
    pub data: NetworkFeature,
    /// The next id a node or a curve takes.
    pub next_id: u32,
    /// The patch being placed: its corners clicked so far, in order.
    pub placing: Option<Vec<PlacedCorner>>,
    /// The network as it was before each edit, and the edits undone — each kept once and
    /// shared by the drafts that follow, as ScanRuler's history holds the data it replaced
    /// rather than copies of it.
    pub history: Vec<Arc<NetworkFeature>>,
    pub future: Vec<Arc<NetworkFeature>>,
    /// The patch picked in the readout or the view, by its index in the report.
    pub picked: Option<usize>,
    pub tool: NetworkTool,
    /// The scan's creases suggested and not yet taken or dismissed; `None` when none were
    /// asked for. Not part of the network, nor of its undo.
    pub suggested: Option<Vec<SuggestedCrease>>,
}

impl NetworkDraft {
    /// A draft of a network with nothing in it.
    pub fn empty() -> NetworkDraft {
        NetworkDraft::of(&empty_network())
    }

    /// A draft of a network feature, to edit it.
    pub fn of(f: &NetworkFeature) -> NetworkDraft {
        let top = f.nodes.iter().map(|n| n.id).chain(f.curves.iter().map(|c| c.id)).max().unwrap_or(0);
        NetworkDraft { data: f.clone(), next_id: top + 1, placing: None, history: Vec::new(), future: Vec::new(), picked: None, tool: NetworkTool::Draw, suggested: None }
    }
}

/// The draft with `next` for its data, the data before kept for undo.
fn edited(d: &NetworkDraft, next: NetworkFeature) -> NetworkDraft {
    let mut out = d.clone();
    push_history(&mut out.history, d.data.clone());
    out.data = next;
    out.future.clear();
    out
}

fn push_history(history: &mut Vec<Arc<NetworkFeature>>, data: NetworkFeature) {
    history.push(Arc::new(data));
    if history.len() > HISTORY {
        let extra = history.len() - HISTORY;
        history.drain(..extra);
    }
}

/// Several steps made one: `after`, with `before`'s data the one undo takes it back to.
fn one_step(before: &NetworkDraft, after: NetworkDraft) -> NetworkDraft {
    let mut out = after;
    out.history = before.history.clone();
    push_history(&mut out.history, before.data.clone());
    out.future.clear();
    out
}

pub fn undo_network(d: &NetworkDraft) -> NetworkDraft {
    let Some(last) = d.history.last() else { return d.clone() };
    let mut out = d.clone();
    out.data = NetworkFeature::clone(last);
    out.placing = None;
    out.history.pop();
    out.future.push(Arc::new(d.data.clone()));
    out
}

pub fn redo_network(d: &NetworkDraft) -> NetworkDraft {
    let Some(next) = d.future.last() else { return d.clone() };
    let mut out = d.clone();
    out.data = NetworkFeature::clone(next);
    out.placing = None;
    out.history.push(Arc::new(d.data.clone()));
    out.future.pop();
    out
}

/// A curve made from one point to another through grips: an end at a point of the scan
/// becomes a node. A curve back to its own start with fewer than two grips is no curve.
pub fn add_curve(d: &NetworkDraft, from: NetworkStart, via: &[DVec3], to: NetworkStart) -> NetworkDraft {
    let mut next_id = d.next_id;
    let mut nodes = d.data.nodes.clone();
    let mut node_of = |s: NetworkStart| -> u32 {
        match s {
            NetworkStart::Node(id) => id,
            NetworkStart::At(at) => {
                let id = next_id;
                next_id += 1;
                nodes.push(NetworkNode { id, at, on: None, corner: None });
                id
            }
        }
    };
    let a = node_of(from);
    let b = node_of(to);
    if a == b && via.len() < 2 {
        return d.clone();
    }
    let curve = NetworkCurve { id: next_id, from: a, to: b, via: via.to_vec(), crease: None, along: None, side: None };
    next_id += 1;
    let mut data = d.data.clone();
    data.nodes = nodes;
    data.curves.push(curve);
    let mut out = edited(d, data);
    out.next_id = next_id;
    out
}

// ---- a patch placed by its corners ----

/// A corner put down on the patch being placed — the first starts it.
pub fn place_corner(d: &NetworkDraft, corner: PlacedCorner) -> NetworkDraft {
    let mut out = d.clone();
    let mut corners = d.placing.clone().unwrap_or_default();
    corners.push(corner);
    out.placing = Some(corners);
    out
}

/// The last corner put down taken back; with none left, the patch let go.
pub fn unplace_corner(d: &NetworkDraft) -> NetworkDraft {
    let mut out = d.clone();
    let corners = d.placing.clone().unwrap_or_default();
    out.placing = if corners.len() > 1 { Some(corners[..corners.len() - 1].to_vec()) } else { None };
    out
}

/// The patch being placed let go.
pub fn drop_placing(d: &NetworkDraft) -> NetworkDraft {
    let mut out = d.clone();
    out.placing = None;
    out
}

/// Where a placed corner is.
pub fn corner_point(d: &NetworkFeature, c: &PlacedCorner) -> Option<DVec3> {
    match c {
        PlacedCorner::Node(id) => d.nodes.iter().find(|n| n.id == *id).map(|n| n.at),
        PlacedCorner::At(at) | PlacedCorner::Curve { at, .. } | PlacedCorner::Grip { at, .. } => Some(*at),
    }
}

/// The patch being placed made: closed, a patch of its corners filled at `seed`; open, the
/// curves between them alone — a line from one point to another. One step of the undo.
pub fn commit_placing(d: &NetworkDraft, closed: bool, seed: Option<DVec3>) -> NetworkDraft {
    let corners = d.placing.clone().unwrap_or_default();
    let mut cleared = d.clone();
    cleared.placing = None;
    if corners.len() < 2 {
        return cleared;
    }
    add_patch(&cleared, &corners, closed, seed, None)
}

/// A patch made of its corners in order: each a node — an existing one, a new one on the
/// scan, or one splitting a curve where it was put — and each corner joined to the next by
/// the curve already joining them, or a chain of curves the next corner's way (a side a
/// neighbour's corner split), or else a new curve along the scan (through `vias[i]`, for the
/// side from corner i, where given). Closed, the last corner is joined to the first, and the
/// region they close is filled at `seed` with them for its corners. One step of the undo.
pub fn add_patch(d: &NetworkDraft, corners: &[PlacedCorner], closed: bool, seed: Option<DVec3>, vias: Option<&[Option<Vec<DVec3>>]>) -> NetworkDraft {
    let mut x = d.clone();
    // The pieces each curve split here became, to find a later corner on.
    let mut pieces: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    let mut ids: Vec<u32> = Vec::new();
    // The vias by side, kept to the corners that became nodes.
    let mut side_vias: Vec<Option<Vec<DVec3>>> = Vec::new();
    for (i, c) in corners.iter().enumerate() {
        let node = match c {
            PlacedCorner::Node(id) => x.data.nodes.iter().any(|n| n.id == *id).then_some(*id),
            PlacedCorner::At(at) => {
                let id = x.next_id;
                x.data.nodes.push(NetworkNode { id, at: *at, on: None, corner: None });
                x.next_id = id + 1;
                Some(id)
            }
            PlacedCorner::Curve { curve, at } | PlacedCorner::Grip { curve, at, .. } => {
                // On a curve, or one of its grips: split there — on the piece it lies on, if a
                // corner before this one split the curve already.
                let of = pieces.get(curve).cloned().unwrap_or_else(|| vec![*curve]);
                let grip = matches!(c, PlacedCorner::Grip { .. });
                let found = nearest_piece(&x.data, &of, *at, grip);
                let split = found.and_then(|f| match f.grip {
                    Some(g) => split_at_grip(&x, f.curve, g),
                    None => split_curve(&x, f.curve, f.run, *at),
                });
                match split {
                    Some(s) => {
                        x = s.draft;
                        let mut list = of;
                        list.push(s.second);
                        pieces.insert(*curve, list);
                        Some(s.node)
                    }
                    None => None,
                }
            }
        };
        let Some(node) = node else { continue };
        if ids.last() == Some(&node) {
            continue;
        }
        ids.push(node);
        side_vias.push(vias.and_then(|v| v.get(i).cloned().flatten()));
    }
    if closed && ids.len() > 1 && ids.first() == ids.last() {
        ids.pop();
    }
    if ids.len() < 2 {
        return one_step(d, x);
    }
    let sides = if closed && ids.len() > 2 { ids.len() } else { ids.len() - 1 };
    let mut next_id = x.next_id;
    let mut curves = x.data.curves.clone();
    for i in 0..sides {
        let a = ids[i];
        let b = ids[(i + 1) % ids.len()];
        if chain_between(&curves, &x.data.nodes, a, b) {
            continue;
        }
        curves.push(NetworkCurve { id: next_id, from: a, to: b, via: side_vias.get(i).cloned().flatten().unwrap_or_default(), crease: None, along: None, side: None });
        next_id += 1;
    }
    let mut cells = x.data.cells.clone();
    if let (true, true, Some(seed)) = (closed, ids.len() > 2, seed) {
        // A patch placed again on the same corners is the one there.
        let key = sorted(&ids);
        let had = cells.iter().any(|c| c.corners.as_ref().is_some_and(|k| sorted(k) == key));
        if !had {
            cells.push(NetworkCell { seed, centre: None, corners: Some(ids.clone()) });
        }
    }
    x.data.curves = curves;
    x.data.cells = cells;
    x.next_id = next_id;
    one_step(d, x)
}

fn sorted(ids: &[u32]) -> Vec<u32> {
    let mut v = ids.to_vec();
    v.sort_unstable();
    v
}

/// The polyline a curve is drawn through: its ends and grips.
fn polyline_of(c: &NetworkCurve, nodes: &[NetworkNode]) -> Vec<DVec3> {
    let at = |id: u32| nodes.iter().find(|n| n.id == id).map(|n| n.at);
    match (at(c.from), at(c.to)) {
        (Some(a), Some(b)) => std::iter::once(a).chain(c.via.iter().copied()).chain(std::iter::once(b)).collect(),
        _ => Vec::new(),
    }
}

fn distance(p: DVec3, q: DVec3) -> f64 {
    ((p.x - q.x).powi(2) + (p.y - q.y).powi(2) + (p.z - q.z).powi(2)).sqrt()
}

fn segment_distance(p: DVec3, a: DVec3, b: DVec3) -> f64 {
    let d = b - a;
    let l2 = d.x * d.x + d.y * d.y + d.z * d.z;
    let t = if l2 > 0.0 { (((p.x - a.x) * d.x + (p.y - a.y) * d.y + (p.z - a.z) * d.z) / l2).clamp(0.0, 1.0) } else { 0.0 };
    distance(p, a + d * t)
}

struct Piece {
    curve: u32,
    run: usize,
    grip: Option<usize>,
}

/// Which of a curve's pieces a point lies on — by the polyline of its ends and grips — the
/// run it is on, and, for a grip, its index.
fn nearest_piece(d: &NetworkFeature, ids: &[u32], p: DVec3, grip: bool) -> Option<Piece> {
    let mut best: Option<Piece> = None;
    let mut best_d = f64::INFINITY;
    for &id in ids {
        let Some(c) = d.curves.iter().find(|x| x.id == id) else { continue };
        if grip {
            if let Some(k) = c.via.iter().position(|v| distance(*v, p) < 1e-6) {
                return Some(Piece { curve: id, run: k, grip: Some(k) });
            }
        }
        let line = polyline_of(c, &d.nodes);
        for k in 0..line.len().saturating_sub(1) {
            let dist = segment_distance(p, line[k], line[k + 1]);
            if dist < best_d {
                best_d = dist;
                best = Some(Piece { curve: id, run: k, grip: None });
            }
        }
    }
    best
}

/// Whether a curve, or a short chain of them the straight way, already joins two nodes.
fn chain_between(curves: &[NetworkCurve], nodes: &[NetworkNode], a: u32, b: u32) -> bool {
    if curves.iter().any(|c| (c.from == a && c.to == b) || (c.from == b && c.to == a)) {
        return true;
    }
    let at = |id: u32| nodes.iter().find(|n| n.id == id).map(|n| n.at);
    let (Some(pa), Some(pb)) = (at(a), at(b)) else { return false };
    let limit = CHAIN_SLACK * distance(pa, pb);
    let length_of = |c: &NetworkCurve| {
        let line = polyline_of(c, nodes);
        line.windows(2).map(|w| distance(w[0], w[1])).sum::<f64>()
    };
    fn walk(curves: &[NetworkCurve], length_of: &dyn Fn(&NetworkCurve) -> f64, b: u32, limit: f64, node: u32, seen: &mut Vec<u32>, length: f64, depth: usize) -> bool {
        if node == b {
            return true;
        }
        if depth >= CHAIN_CURVES {
            return false;
        }
        for c in curves {
            let next = if c.from == node {
                c.to
            } else if c.to == node {
                c.from
            } else {
                continue;
            };
            if seen.contains(&next) {
                continue;
            }
            let l = length + length_of(c);
            if l > limit {
                continue;
            }
            seen.push(next);
            if walk(curves, length_of, b, limit, next, seen, l, depth + 1) {
                return true;
            }
            seen.pop();
        }
        false
    }
    let mut seen = vec![a];
    walk(curves, &length_of, b, limit, a, &mut seen, 0.0, 0)
}

/// What a split made: the draft, the node put in, and the curve that goes on from it.
pub struct Split {
    pub draft: NetworkDraft,
    pub node: u32,
    pub second: u32,
}

/// A curve split at a point along it — on its run after grip `run` (0 is the run from its
/// first end) — by a new node there: the curve keeps its id up to the node, a new one
/// (`second`) goes on from it.
pub fn split_curve(d: &NetworkDraft, curve_id: u32, run: usize, at: DVec3) -> Option<Split> {
    let c = d.data.curves.iter().find(|x| x.id == curve_id)?.clone();
    let r = run.min(c.via.len());
    Some(split_with(d, &c, at, c.via[..r].to_vec(), c.via[r..].to_vec()))
}

/// A curve split at one of its grips, the grip made the node there.
pub fn split_at_grip(d: &NetworkDraft, curve_id: u32, index: usize) -> Option<Split> {
    let c = d.data.curves.iter().find(|x| x.id == curve_id)?.clone();
    if index >= c.via.len() {
        return None;
    }
    Some(split_with(d, &c, c.via[index], c.via[..index].to_vec(), c.via[index + 1..].to_vec()))
}

fn split_with(d: &NetworkDraft, c: &NetworkCurve, at: DVec3, first: Vec<DVec3>, rest: Vec<DVec3>) -> Split {
    let node = d.next_id;
    let second = d.next_id + 1;
    let a = NetworkCurve { to: node, via: first, ..c.clone() };
    let b = NetworkCurve { id: second, from: node, via: rest, ..c.clone() };
    let mut data = d.data.clone();
    data.nodes.push(NetworkNode { id: node, at, on: None, corner: None });
    data.curves = d.data.curves.iter().flat_map(|x| if x.id == c.id { vec![a.clone(), b.clone()] } else { vec![x.clone()] }).collect();
    // The second half runs in the same strip as the first did.
    if let Some(s) = d.data.spans.get(&c.id.to_string()).copied() {
        data.spans.insert(second.to_string(), s);
    }
    let mut draft = edited(d, data);
    draft.next_id = d.next_id + 2;
    Split { draft, node, second }
}

/// Cells' seeds put elsewhere — their patches' middles after a corner moved — as part of the
/// step that moved it, not one of their own.
pub fn set_cell_seeds(d: &NetworkDraft, seeds: &[(usize, DVec3)]) -> NetworkDraft {
    if seeds.is_empty() {
        return d.clone();
    }
    let mut out = d.clone();
    for (i, c) in out.data.cells.iter_mut().enumerate() {
        if let Some((_, s)) = seeds.iter().find(|(k, _)| *k == i) {
            c.seed = *s;
        }
    }
    out
}

/// The cells with their corners kept to the nodes there still are: a cell left with fewer
/// than three is read by the rule again.
fn cells_with_nodes(cells: &[NetworkCell], nodes: &[NetworkNode]) -> Vec<NetworkCell> {
    let alive: HashSet<u32> = nodes.iter().map(|n| n.id).collect();
    cells
        .iter()
        .map(|c| {
            let Some(corners) = &c.corners else { return c.clone() };
            if corners.iter().all(|id| alive.contains(id)) {
                return c.clone();
            }
            let kept: Vec<u32> = corners.iter().copied().filter(|id| alive.contains(id)).collect();
            NetworkCell { corners: (kept.len() >= 3).then_some(kept), ..c.clone() }
        })
        .collect()
}

/// A node moved — onto the scan where it was dragged to.
pub fn move_node(d: &NetworkDraft, id: u32, at: DVec3, record: bool) -> NetworkDraft {
    let mut data = d.data.clone();
    for n in data.nodes.iter_mut().filter(|n| n.id == id) {
        n.at = at;
    }
    if record {
        edited(d, data)
    } else {
        NetworkDraft { data, ..d.clone() }
    }
}

/// A curve's grip moved.
pub fn move_grip(d: &NetworkDraft, curve_id: u32, index: usize, at: DVec3, record: bool) -> NetworkDraft {
    let mut data = d.data.clone();
    for c in data.curves.iter_mut().filter(|c| c.id == curve_id) {
        if let Some(p) = c.via.get_mut(index) {
            *p = at;
        }
    }
    if record {
        edited(d, data)
    } else {
        NetworkDraft { data, ..d.clone() }
    }
}

/// A cell's seed or centre moved.
pub fn move_cell(d: &NetworkDraft, index: usize, what: CellPoint, at: DVec3, record: bool) -> NetworkDraft {
    let mut data = d.data.clone();
    if let Some(c) = data.cells.get_mut(index) {
        match what {
            CellPoint::Seed => c.seed = at,
            CellPoint::Centre => c.centre = Some(at),
        }
    }
    if record {
        edited(d, data)
    } else {
        NetworkDraft { data, ..d.clone() }
    }
}

/// A split cell's centre back at its polygon's middle.
pub fn reset_centre(d: &NetworkDraft, index: usize) -> NetworkDraft {
    let mut data = d.data.clone();
    if let Some(c) = data.cells.get_mut(index) {
        c.centre = None;
    }
    edited(d, data)
}

/// The draft as it stood before a drag began, put in the history once the drag is done — the
/// drag's moves themselves are not steps.
pub fn record_drag(d: &NetworkDraft, before: &NetworkFeature) -> NetworkDraft {
    let mut out = d.clone();
    push_history(&mut out.history, before.clone());
    out.future.clear();
    out
}

/// A curve along the whole of a body's edge (PN11), its ends nodes on the edge's corners — the
/// network's own where one is within a hair — and the patch beside it meeting the face given
/// on its edge, to be made tangent in the box.
pub fn add_edge_curve(d: &NetworkDraft, body: BodyId, edge: &EdgeRef, ends: [DVec3; 2], face: Option<FaceRef>) -> NetworkDraft {
    let mut next_id = d.next_id;
    let mut nodes = d.data.nodes.clone();
    let mut node_at = |at: DVec3, end: u8| -> u32 {
        if let Some(near) = nodes.iter().find(|n| distance(n.at, at) <= EDGE_NODE_MM) {
            return near.id;
        }
        let id = next_id;
        next_id += 1;
        nodes.push(NetworkNode { id, at, on: Some(NodeOn::Vertex { body, vertex: edge.clone(), end }), corner: None });
        id
    };
    let from = node_at(ends[0], 0);
    let to = node_at(ends[1], 1);
    if from == to {
        return d.clone();
    }
    let curve = NetworkCurve { id: next_id, from, to, via: Vec::new(), crease: None, along: Some(CurveAlong { body, edge: edge.clone() }), side: face.map(|face| CurveSide { body, face, mode: SideMode::Contact }) };
    next_id += 1;
    let mut data = d.data.clone();
    data.nodes = nodes;
    data.curves.push(curve);
    let mut out = edited(d, data);
    out.next_id = next_id;
    out
}

/// The patch beside a curve on a body: meeting the face on its edge (contact) or tangent to
/// it, or — `None` — beside no face.
pub fn set_side(d: &NetworkDraft, curve_id: u32, side: Option<CurveSide>) -> NetworkDraft {
    let mut data = d.data.clone();
    for c in data.curves.iter_mut().filter(|c| c.id == curve_id) {
        c.side = side.clone();
    }
    edited(d, data)
}

/// A grip put into a curve on its run after grip `run`.
pub fn insert_grip(d: &NetworkDraft, curve_id: u32, run: usize, at: DVec3, record: bool) -> NetworkDraft {
    let mut data = d.data.clone();
    for c in data.curves.iter_mut().filter(|c| c.id == curve_id) {
        let k = run.min(c.via.len());
        c.via.insert(k, at);
    }
    if record {
        edited(d, data)
    } else {
        NetworkDraft { data, ..d.clone() }
    }
}

/// A grip taken out of a curve.
pub fn remove_grip(d: &NetworkDraft, curve_id: u32, index: usize) -> NetworkDraft {
    let mut data = d.data.clone();
    for c in data.curves.iter_mut().filter(|c| c.id == curve_id) {
        if index < c.via.len() {
            c.via.remove(index);
        }
    }
    edited(d, data)
}

/// A curve deleted, and the nodes no other curve ends at with it.
pub fn remove_curve(d: &NetworkDraft, curve_id: u32) -> NetworkDraft {
    let mut data = d.data.clone();
    data.curves.retain(|c| c.id != curve_id);
    let used: HashSet<u32> = data.curves.iter().flat_map(|c| [c.from, c.to]).collect();
    data.nodes.retain(|n| used.contains(&n.id));
    data.spans.remove(&curve_id.to_string());
    data.cells = cells_with_nodes(&data.cells, &data.nodes);
    edited(d, data)
}

/// A node deleted, with every curve that ends at it.
pub fn remove_node(d: &NetworkDraft, node_id: u32) -> NetworkDraft {
    let gone: HashSet<u32> = d.data.curves.iter().filter(|x| x.from == node_id || x.to == node_id).map(|c| c.id).collect();
    let mut data = d.data.clone();
    data.curves.retain(|c| !gone.contains(&c.id));
    let used: HashSet<u32> = data.curves.iter().flat_map(|c| [c.from, c.to]).collect();
    data.nodes.retain(|n| n.id != node_id && used.contains(&n.id));
    data.spans.retain(|k, _| k.parse::<u32>().map_or(true, |id| !gone.contains(&id)));
    data.cells = cells_with_nodes(&data.cells, &data.nodes);
    edited(d, data)
}

/// A curve made a crease, or tangent again.
pub fn toggle_crease(d: &NetworkDraft, curve_id: u32) -> NetworkDraft {
    let mut data = d.data.clone();
    for c in data.curves.iter_mut().filter(|c| c.id == curve_id) {
        c.crease = if c.crease == Some(true) { None } else { Some(true) };
    }
    edited(d, data)
}

/// The creases the scan suggests, to offer — or, `None`, put away.
pub fn set_suggested(d: &NetworkDraft, suggested: Option<Vec<SuggestedCrease>>) -> NetworkDraft {
    NetworkDraft { suggested, ..d.clone() }
}

/// Suggested creases taken as crease curves, by their index among those offered, and no
/// longer offered. An end becomes the network's node there — one a suggestion taken already
/// put there, or one within EDGE_NODE_MM of it — or a new one. One step of the undo.
pub fn accept_creases(d: &NetworkDraft, indices: &[usize]) -> NetworkDraft {
    let offered = d.suggested.clone().unwrap_or_default();
    let mut take: Vec<usize> = indices.iter().copied().filter(|i| *i < offered.len()).collect();
    take.sort_unstable();
    take.dedup();
    if take.is_empty() {
        return d.clone();
    }
    let mut next_id = d.next_id;
    let mut nodes = d.data.nodes.clone();
    let mut curves = d.data.curves.clone();
    for &i in &take {
        let s = &offered[i];
        let mut node_at = |at: DVec3| -> u32 {
            let mut best: Option<u32> = None;
            let mut best_d = EDGE_NODE_MM;
            for n in &nodes {
                let dist = distance(n.at, at);
                if dist <= best_d {
                    best_d = dist;
                    best = Some(n.id);
                }
            }
            if let Some(id) = best {
                return id;
            }
            let id = next_id;
            next_id += 1;
            nodes.push(NetworkNode { id, at, on: None, corner: None });
            id
        };
        let from = node_at(s.from);
        let to = node_at(s.to);
        if from == to && s.via.len() < 2 {
            continue;
        }
        curves.push(NetworkCurve { id: next_id, from, to, via: s.via.clone(), crease: Some(true), along: None, side: None });
        next_id += 1;
    }
    let left: Vec<SuggestedCrease> = offered.into_iter().enumerate().filter(|(i, _)| !take.contains(i)).map(|(_, s)| s).collect();
    let mut data = d.data.clone();
    data.nodes = nodes;
    data.curves = curves;
    let mut out = edited(d, data);
    out.next_id = next_id;
    out.suggested = (!left.is_empty()).then_some(left);
    out
}

/// A strip's span count set — keyed by the lowest id among its curves, every other key of the
/// strip taken away — or, `None`, back to Auto.
pub fn set_strip_spans(d: &NetworkDraft, strip_curves: &[u32], spans: Option<f64>) -> NetworkDraft {
    let mut data = d.data.clone();
    for c in strip_curves {
        data.spans.remove(&c.to_string());
    }
    if let (Some(n), Some(low)) = (spans, strip_curves.iter().min()) {
        data.spans.insert(low.to_string(), crate::js_number::js_round(n).max(0.0) as u32);
    }
    edited(d, data)
}

/// A region filled, by a point inside it.
pub fn add_seed(d: &NetworkDraft, at: DVec3) -> NetworkDraft {
    let mut data = d.data.clone();
    data.cells.push(NetworkCell { seed: at, centre: None, corners: None });
    edited(d, data)
}

/// Regions filled by their seeds, all at once — Fill all.
pub fn add_seeds(d: &NetworkDraft, seeds: &[DVec3]) -> NetworkDraft {
    if seeds.is_empty() {
        return d.clone();
    }
    let mut data = d.data.clone();
    data.cells.extend(seeds.iter().map(|s| NetworkCell { seed: *s, centre: None, corners: None }));
    edited(d, data)
}

/// A filled region emptied.
pub fn remove_seed(d: &NetworkDraft, index: usize) -> NetworkDraft {
    let mut data = d.data.clone();
    if index < data.cells.len() {
        data.cells.remove(index);
    }
    edited(d, data)
}

/// A patch deleted: its cell, the curves of its own round it — `own`, the ones no other patch
/// runs along — and the nodes no curve ends at once they are gone. One step.
pub fn remove_patch(d: &NetworkDraft, index: usize, own: &[u32]) -> NetworkDraft {
    let gone: HashSet<u32> = own.iter().copied().collect();
    let mut data = d.data.clone();
    data.curves.retain(|c| !gone.contains(&c.id));
    let used: HashSet<u32> = data.curves.iter().flat_map(|c| [c.from, c.to]).collect();
    data.nodes.retain(|n| used.contains(&n.id));
    data.spans.retain(|k, _| k.parse::<u32>().map_or(true, |id| !gone.contains(&id)));
    let kept: Vec<NetworkCell> = data.cells.iter().enumerate().filter(|(i, _)| *i != index).map(|(_, c)| c.clone()).collect();
    data.cells = cells_with_nodes(&kept, &data.nodes);
    let mut out = edited(d, data);
    out.picked = None;
    out
}

/// The patch a report names deleted ([`remove_patch`]): the cell it is of — a split cell's
/// patches go together — with those of the curves round it that are the draft's and no other
/// patch of the report runs along. `sides` is each patch's sides' curves, `cells` the report's
/// cells.
pub fn remove_patch_of(d: &NetworkDraft, sides: &[Vec<Vec<u32>>], cells: &[CellOutcome], patch: usize) -> NetworkDraft {
    let Some(cell) = cells.iter().position(|c| matches!(c, CellOutcome::Filled { patches, .. } if patches.contains(&patch))) else { return d.clone() };
    let CellOutcome::Filled { patches: mine, .. } = &cells[cell] else { return d.clone() };
    let others: HashSet<u32> = sides.iter().enumerate().filter(|(i, _)| !mine.contains(i)).flat_map(|(_, s)| s.iter().flatten().copied()).collect();
    let known: HashSet<u32> = d.data.curves.iter().map(|c| c.id).collect();
    let mut own: Vec<u32> = Vec::new();
    for i in mine {
        for id in sides.get(*i).into_iter().flatten().flatten() {
            if known.contains(id) && !others.contains(id) && !own.contains(id) {
                own.push(*id);
            }
        }
    }
    remove_patch(d, cell, &own)
}

/// [`remove_patch_of`] by a report.
pub fn remove_reported_patch(d: &NetworkDraft, report: &NetworkReport, patch: usize) -> NetworkDraft {
    let sides: Vec<Vec<Vec<u32>>> = report.patches.iter().map(|p| p.sides.clone()).collect();
    remove_patch_of(d, &sides, &report.cells, patch)
}

/// The network's fit changed.
pub fn set_network_fit(d: &NetworkDraft, change: impl FnOnce(&mut NetworkFitParams)) -> NetworkDraft {
    let mut data = d.data.clone();
    change(&mut data.fit);
    edited(d, data)
}

/// Whether a network has anything to build: Create waits for a curve and a filled cell.
pub fn network_ready(d: &NetworkFeature) -> bool {
    !d.curves.is_empty() && !d.cells.is_empty()
}

// ---- reading the report ----

/// The strip a curve is in, by the report.
pub fn strip_of(strips: &[Strip], curve: Option<u32>) -> Option<&Strip> {
    let c = curve?;
    strips.iter().find(|s| s.curves.contains(&c))
}

/// The curves two patches share, by the last build.
pub fn shared_curves(report: &NetworkReport) -> HashSet<u32> {
    let mut count: BTreeMap<u32, usize> = BTreeMap::new();
    for p in &report.patches {
        for side in &p.sides {
            for id in side {
                *count.entry(*id).or_insert(0) += 1;
            }
        }
    }
    count.into_iter().filter(|(_, n)| *n >= 2).map(|(id, _)| id).collect()
}

/// The patch a point of the scan lies on, by each patch's samples (PATCH_GRID × PATCH_GRID,
/// row-major along u, x, y, z each): of the patches whose sampled surface has the point over
/// one of its triangles — inside its rim, not beside it — the nearest, if no further off it
/// than a tenth of its size; or `None`.
pub fn patch_under<'a>(grids: impl IntoIterator<Item = &'a [f32]>, at: DVec3) -> Option<usize> {
    let n = PATCH_GRID;
    let mut best: Option<usize> = None;
    let mut best_d = f64::INFINITY;
    for (i, g) in grids.into_iter().enumerate() {
        if g.len() < n * n * 3 {
            continue;
        }
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for (k, v) in g.iter().enumerate() {
            lo[k % 3] = lo[k % 3].min(*v as f64);
            hi[k % 3] = hi[k % 3].max(*v as f64);
        }
        let reach = 0.1 * ((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2) + (hi[2] - lo[2]).powi(2)).sqrt();
        let p = [at.x, at.y, at.z];
        if (0..3).any(|k| p[k] < lo[k] - reach || p[k] > hi[k] + reach) {
            continue;
        }
        let corner = |r: usize, c: usize| DVec3::new(g[(r * n + c) * 3] as f64, g[(r * n + c) * 3 + 1] as f64, g[(r * n + c) * 3 + 2] as f64);
        for r in 0..n - 1 {
            for c in 0..n - 1 {
                for [a, b, e] in [[corner(r, c), corner(r + 1, c), corner(r + 1, c + 1)], [corner(r, c), corner(r + 1, c + 1), corner(r, c + 1)]] {
                    if let Some(off) = over_triangle(at, a, b, e) {
                        if off <= reach && off < best_d {
                            best_d = off;
                            best = Some(i);
                        }
                    }
                }
            }
        }
    }
    best
}

/// How far p is off the triangle abc's plane, where it lies over the triangle — its foot
/// inside it — or `None`.
fn over_triangle(p: DVec3, a: DVec3, b: DVec3, c: DVec3) -> Option<f64> {
    let u = b - a;
    let v = c - a;
    let w = p - a;
    let uu = u.dot(u);
    let uv = u.dot(v);
    let vv = v.dot(v);
    let wu = w.dot(u);
    let wv = w.dot(v);
    let det = uu * vv - uv * uv;
    if !(det > 0.0) {
        return None;
    }
    let s = (vv * wu - uv * wv) / det;
    let t = (uu * wv - uv * wu) / det;
    if s < 0.0 || t < 0.0 || s + t > 1.0 {
        return None;
    }
    let n = u.cross(v);
    Some(w.dot(n).abs() / det.sqrt())
}

/// A patch's sampled grid round its rim, as a closed polyline.
pub fn ring_of(g: &[f32]) -> Vec<DVec3> {
    let n = PATCH_GRID;
    if g.len() < n * n * 3 {
        return Vec::new();
    }
    let at = |i: usize, j: usize| DVec3::new(g[(i * n + j) * 3] as f64, g[(i * n + j) * 3 + 1] as f64, g[(i * n + j) * 3 + 2] as f64);
    let mut ring = Vec::with_capacity(4 * n);
    for i in 0..n {
        ring.push(at(i, 0));
    }
    for j in 1..n {
        ring.push(at(n - 1, j));
    }
    for i in (0..n - 1).rev() {
        ring.push(at(i, n - 1));
    }
    for j in (0..n - 1).rev() {
        ring.push(at(0, j));
    }
    ring
}

/// The box's status line: patches, regions closed but not filled, the worst angle across a
/// tangent curve and the worst gap — or what there is yet. The gap is in micrometres, or in
/// inches when they are shown.
pub fn network_status(report: Option<&NetworkReport>, curves: usize, placing: bool, units: DisplayUnits) -> String {
    let Some(report) = report else {
        // Nothing yet: the hint under the box says how to start, and the patch being placed
        // says the rest.
        return if curves == 0 {
            if placing {
                String::new()
            } else {
                "No patches yet.".to_string()
            }
        } else {
            format!("{curves} curve{} · building…", if curves == 1 { "" } else { "s" })
        };
    };
    let n = report.patches.len();
    let open = report.unfilled.len();
    let refused = report.cells.iter().filter(|c| matches!(c, CellOutcome::Refused { .. })).count();
    let mut parts = vec![format!("{n} patch{}", if n == 1 { "" } else { "es" })];
    if open > 0 {
        parts.push(format!("{open} open cell{}", if open == 1 { "" } else { "s" }));
    }
    if refused > 0 {
        parts.push(format!("{refused} refused"));
    }
    if n > 0 {
        parts.push(format!("worst edge {}°", to_fixed(report.worst_angle, if report.worst_angle < 1.0 { 2 } else { 1 })));
        let gap = match units {
            DisplayUnits::In => fmt_len(report.worst_gap, units, 4, FmtOptions::default()),
            DisplayUnits::Mm => format!("{} µm", to_fixed(report.worst_gap * 1000.0, if report.worst_gap < 0.01 { 1 } else { 0 })),
        };
        parts.push(format!("worst gap {gap}"));
        // A corner of three, five or six patches is held near tangent, not tangent: its
        // worst angle stands beside the edges'.
        if !report.corners.is_empty() {
            let worst = report.corners.iter().map(|c| c.angle).fold(f64::NEG_INFINITY, f64::max);
            parts.push(format!("worst corner {}°", to_fixed(worst, 1)));
        }
        if report.closed {
            parts.push("closed — a solid".to_string());
        }
    }
    parts.join(" · ")
}

#[cfg(test)]
mod tests {
    //! ScanRuler's `plugins/reverse/tests/reNetworkDraft.test.ts` (the draft's edits),
    //! `reNetworkPick.test.ts` (the patch under the cursor) and `reNetworkProject.test.ts` (a
    //! network in a project file).

    use super::*;
    use crate::network_body::{ReportNet, ReportPatch, ReportTimes};

    fn v(x: f64, y: f64, z: f64) -> DVec3 {
        DVec3::new(x, y, z)
    }

    fn crease(from: DVec3, to: DVec3) -> SuggestedCrease {
        SuggestedCrease { from, to, via: vec![(from + to) / 2.0], points: vec![from.x as f32, from.y as f32, from.z as f32, to.x as f32, to.y as f32, to.z as f32], length: (to - from).length(), angle: 90.0 }
    }

    /// The node at a point, by its id.
    fn node_at(d: &NetworkDraft, at: DVec3) -> Option<u32> {
        d.data.nodes.iter().find(|n| (n.at - at).length() < 1e-9).map(|n| n.id)
    }

    /// Whether two nodes are joined by a curve.
    fn joins(d: &NetworkDraft, a: u32, b: u32) -> bool {
        d.data.curves.iter().any(|c| (c.from == a && c.to == b) || (c.from == b && c.to == a))
    }

    /// A square patch placed at x0 by its four corners on the scan.
    fn square(d: &NetworkDraft, x0: f64) -> NetworkDraft {
        add_patch(d, &[PlacedCorner::At(v(x0, 0.0, 0.0)), PlacedCorner::At(v(x0 + 10.0, 0.0, 0.0)), PlacedCorner::At(v(x0 + 10.0, 10.0, 0.0)), PlacedCorner::At(v(x0, 10.0, 0.0))], true, Some(v(x0 + 5.0, 5.0, 0.0)), None)
    }

    #[test]
    fn makes_four_nodes_four_curves_and_a_cell_keeping_its_corners_in_one_undo_step() {
        let d = square(&NetworkDraft::empty(), 0.0);
        assert_eq!(d.data.nodes.len(), 4);
        assert_eq!(d.data.curves.len(), 4);
        assert_eq!(d.data.cells, vec![NetworkCell { seed: v(5.0, 5.0, 0.0), centre: None, corners: Some(d.data.nodes.iter().map(|n| n.id).collect()) }]);
        assert_eq!(d.history.len(), 1);
        assert_eq!(undo_network(&d).data.curves.len(), 0);
    }

    #[test]
    fn places_corners_one_by_one_takes_one_back_and_closes() {
        let mut d = NetworkDraft::empty();
        for at in [v(0.0, 0.0, 0.0), v(10.0, 0.0, 0.0), v(99.0, 99.0, 0.0)] {
            d = place_corner(&d, PlacedCorner::At(at));
        }
        d = unplace_corner(&d);
        d = place_corner(&place_corner(&d, PlacedCorner::At(v(10.0, 10.0, 0.0))), PlacedCorner::At(v(0.0, 10.0, 0.0)));
        assert_eq!(d.placing.as_ref().map(Vec::len), Some(4));
        assert_eq!(d.history.len(), 0);
        let made = commit_placing(&d, true, Some(v(5.0, 5.0, 0.0)));
        assert!(made.placing.is_none());
        assert_eq!(made.data.cells.len(), 1);
        assert!(node_at(&made, v(99.0, 99.0, 0.0)).is_none());
    }

    #[test]
    fn shares_the_edge_with_the_patch_beside_it() {
        let a = square(&NetworkDraft::empty(), 0.0);
        let right = node_at(&a, v(10.0, 0.0, 0.0)).expect("node");
        let top = node_at(&a, v(10.0, 10.0, 0.0)).expect("node");
        let b = add_patch(&a, &[PlacedCorner::Node(right), PlacedCorner::At(v(20.0, 0.0, 0.0)), PlacedCorner::At(v(20.0, 10.0, 0.0)), PlacedCorner::Node(top)], true, Some(v(15.0, 5.0, 0.0)), None);
        assert_eq!(b.data.nodes.len(), 6);
        assert_eq!(b.data.curves.len(), 7);
        assert_eq!(b.data.cells[1].corners, Some(vec![right, node_at(&b, v(20.0, 0.0, 0.0)).expect("n"), node_at(&b, v(20.0, 10.0, 0.0)).expect("n"), top]));
        // The same corners again fill nothing more.
        let again = add_patch(&b, &[PlacedCorner::Node(right), PlacedCorner::Node(node_at(&b, v(20.0, 0.0, 0.0)).expect("n")), PlacedCorner::Node(node_at(&b, v(20.0, 10.0, 0.0)).expect("n")), PlacedCorner::Node(top)], true, Some(v(15.0, 5.0, 0.0)), None);
        assert_eq!(again.data.cells.len(), 2);
    }

    #[test]
    fn takes_a_side_split_by_a_neighbours_corner_as_it_is() {
        let mut d = add_patch(&NetworkDraft::empty(), &[PlacedCorner::At(v(10.0, 0.0, 0.0)), PlacedCorner::At(v(20.0, 0.0, 0.0)), PlacedCorner::At(v(20.0, 10.0, 0.0)), PlacedCorner::At(v(10.0, 10.0, 0.0))], true, Some(v(15.0, 5.0, 0.0)), None);
        let mid = node_at(&d, v(10.0, 10.0, 0.0)).expect("n");
        d = add_patch(&d, &[PlacedCorner::Node(mid), PlacedCorner::Node(node_at(&d, v(20.0, 10.0, 0.0)).expect("n")), PlacedCorner::At(v(20.0, 20.0, 0.0)), PlacedCorner::At(v(10.0, 20.0, 0.0))], true, Some(v(15.0, 15.0, 0.0)), None);
        let curves = d.data.curves.len();
        let low = node_at(&d, v(10.0, 0.0, 0.0)).expect("n");
        let high = node_at(&d, v(10.0, 20.0, 0.0)).expect("n");
        d = add_patch(&d, &[PlacedCorner::At(v(0.0, 0.0, 0.0)), PlacedCorner::Node(low), PlacedCorner::Node(high), PlacedCorner::At(v(0.0, 20.0, 0.0))], true, Some(v(5.0, 10.0, 0.0)), None);
        // Three new curves: the chain low–mid–high is the wide patch's side.
        assert_eq!(d.data.curves.len(), curves + 3);
        assert!(!joins(&d, low, high));
        assert_eq!(d.data.cells[2].corners, Some(vec![node_at(&d, v(0.0, 0.0, 0.0)).expect("n"), low, high, node_at(&d, v(0.0, 20.0, 0.0)).expect("n")]));
    }

    #[test]
    fn splits_a_curve_where_a_corner_is_put_on_it_and_on_two_points_of_one_curve() {
        let mut d = add_curve(&NetworkDraft::empty(), NetworkStart::At(v(0.0, 0.0, 0.0)), &[], NetworkStart::At(v(30.0, 0.0, 0.0)));
        let id = d.data.curves[0].id;
        d = add_patch(&d, &[PlacedCorner::Curve { curve: id, at: v(10.0, 0.0, 0.0) }, PlacedCorner::Curve { curve: id, at: v(20.0, 0.0, 0.0) }, PlacedCorner::At(v(20.0, 10.0, 0.0)), PlacedCorner::At(v(10.0, 10.0, 0.0))], true, Some(v(15.0, 5.0, 0.0)), None);
        let a = node_at(&d, v(10.0, 0.0, 0.0)).expect("n");
        let b = node_at(&d, v(20.0, 0.0, 0.0)).expect("n");
        assert!(joins(&d, node_at(&d, v(0.0, 0.0, 0.0)).expect("n"), a));
        assert!(joins(&d, a, b));
        assert!(joins(&d, b, node_at(&d, v(30.0, 0.0, 0.0)).expect("n")));
        // The three pieces of the line and the patch's three new sides.
        assert_eq!(d.data.curves.len(), 6);
        assert_eq!(d.data.cells[0].corners, Some(vec![a, b, node_at(&d, v(20.0, 10.0, 0.0)).expect("n"), node_at(&d, v(10.0, 10.0, 0.0)).expect("n")]));
    }

    #[test]
    fn makes_a_grip_clicked_a_corner_the_curve_split_there_the_grip_gone() {
        let mut d = add_curve(&NetworkDraft::empty(), NetworkStart::At(v(0.0, 0.0, 0.0)), &[v(5.0, 1.0, 0.0), v(10.0, 0.0, 0.0), v(15.0, 1.0, 0.0)], NetworkStart::At(v(20.0, 0.0, 0.0)));
        let id = d.data.curves[0].id;
        let first = node_at(&d, v(0.0, 0.0, 0.0)).expect("n");
        d = add_patch(&d, &[PlacedCorner::Grip { curve: id, grip: 1, at: v(10.0, 0.0, 0.0) }, PlacedCorner::At(v(10.0, 10.0, 0.0)), PlacedCorner::At(v(0.0, 10.0, 0.0)), PlacedCorner::Node(first)], true, Some(v(5.0, 5.0, 0.0)), None);
        let g = node_at(&d, v(10.0, 0.0, 0.0)).expect("n");
        let mut halves: Vec<usize> = d.data.curves.iter().filter(|c| c.from == g || c.to == g).map(|c| c.via.len()).collect();
        halves.sort_unstable();
        assert_eq!(halves, vec![0, 1, 1]);
        assert!(!d.data.curves.iter().flat_map(|c| c.via.iter()).any(|p| p.x == 10.0 && p.y == 0.0));
    }

    #[test]
    fn makes_a_line_of_two_corners_open_with_no_cell() {
        let mut d = place_corner(&place_corner(&NetworkDraft::empty(), PlacedCorner::At(v(0.0, 0.0, 0.0))), PlacedCorner::At(v(10.0, 0.0, 0.0)));
        d = commit_placing(&d, false, None);
        assert_eq!(d.data.curves.len(), 1);
        assert_eq!(d.data.cells.len(), 0);
    }

    #[test]
    fn keeps_a_patchs_corners_to_the_nodes_there_are_and_leaves_it_to_the_rule_with_fewer_than_three() {
        let d = square(&NetworkDraft::empty(), 0.0);
        let one = remove_node(&d, node_at(&d, v(10.0, 10.0, 0.0)).expect("n"));
        assert_eq!(one.data.cells[0].corners, Some(vec![node_at(&d, v(0.0, 0.0, 0.0)).expect("n"), node_at(&d, v(10.0, 0.0, 0.0)).expect("n"), node_at(&d, v(0.0, 10.0, 0.0)).expect("n")]));
        assert_eq!(remove_node(&one, node_at(&d, v(0.0, 10.0, 0.0)).expect("n")).data.cells[0].corners, None);
    }

    /// Two squares side by side, sharing the curve at x = 10, and a report of them as a build
    /// would give it: a patch a cell, its sides' curves.
    fn pair() -> (NetworkDraft, Vec<Vec<Vec<u32>>>, Vec<CellOutcome>) {
        let a = square(&NetworkDraft::empty(), 0.0);
        let d = add_patch(&a, &[PlacedCorner::Node(node_at(&a, v(10.0, 0.0, 0.0)).expect("n")), PlacedCorner::At(v(20.0, 0.0, 0.0)), PlacedCorner::At(v(20.0, 10.0, 0.0)), PlacedCorner::Node(node_at(&a, v(10.0, 10.0, 0.0)).expect("n"))], true, Some(v(15.0, 5.0, 0.0)), None);
        let sides_of = |cell: usize| -> Vec<Vec<u32>> {
            let corners = d.data.cells[cell].corners.clone().unwrap_or_default();
            (0..corners.len())
                .map(|i| {
                    let (n, m) = (corners[i], corners[(i + 1) % 4]);
                    vec![d.data.curves.iter().find(|c| (c.from == n && c.to == m) || (c.to == n && c.from == m)).map(|c| c.id).unwrap_or(0)]
                })
                .collect()
        };
        let sides = vec![sides_of(0), sides_of(1)];
        let cells = (0..2).map(|i| CellOutcome::Filled { seed: d.data.cells[i].seed, sides: 4, patch: i, patches: vec![i] }).collect();
        (d, sides, cells)
    }

    #[test]
    fn deletes_a_patch_with_the_edges_only_it_has_keeping_the_one_it_shares_in_one_undo_step() {
        let (d, sides, cells) = pair();
        let ends = [node_at(&d, v(10.0, 0.0, 0.0)), node_at(&d, v(10.0, 10.0, 0.0))];
        let shared = d.data.curves.iter().find(|c| [c.from, c.to].iter().all(|n| ends.contains(&Some(*n)))).map(|c| c.id).expect("shared");
        let gone = remove_patch_of(&d, &sides, &cells, 1);
        assert_eq!(gone.data.cells.len(), 1);
        assert_eq!(gone.data.curves.len(), 4);
        assert!(gone.data.curves.iter().any(|c| c.id == shared));
        assert_eq!(gone.data.nodes.len(), 4);
        assert!(node_at(&gone, v(20.0, 0.0, 0.0)).is_none());
        assert_eq!(gone.picked, None);
        assert_eq!(undo_network(&gone).data.cells.len(), 2);
        // The other one leaves the shared curve to the patch still there.
        let first = remove_patch_of(&d, &sides, &cells, 0);
        assert_eq!(first.data.curves.len(), 4);
        assert_eq!(first.data.cells[0].seed, v(15.0, 5.0, 0.0));
    }

    #[test]
    fn bends_an_edge_in_one_undo_step_the_grip_put_in_and_dragged_are_one() {
        let (d, _, _) = pair();
        let before = d.data.clone();
        let id = d.data.curves[0].id;
        let mut x = insert_grip(&d, id, 0, v(5.0, 0.0, 0.0), false);
        assert_eq!(x.history.len(), d.history.len());
        x = move_grip(&x, id, 0, v(5.0, -2.0, 0.0), false);
        x = record_drag(&x, &before);
        assert_eq!(x.data.curves[0].via, vec![v(5.0, -2.0, 0.0)]);
        assert_eq!(x.history.len(), d.history.len() + 1);
        assert_eq!(undo_network(&x).data.curves[0].via, Vec::<DVec3>::new());
    }

    #[test]
    fn takes_all_of_an_l_as_two_crease_curves_meeting_at_one_node_in_one_undo_step() {
        let d = set_suggested(&NetworkDraft::empty(), Some(vec![crease(v(0.0, 0.0, 0.0), v(10.0, 0.0, 0.0)), crease(v(10.0, 0.0, 0.0), v(10.0, 10.0, 0.0))]));
        assert_eq!(d.history.len(), 0);
        let taken = accept_creases(&d, &[0, 1]);
        assert_eq!(taken.data.curves.len(), 2);
        assert!(taken.data.curves.iter().all(|c| c.crease == Some(true)));
        assert_eq!(taken.data.nodes.len(), 3);
        assert_eq!(taken.data.curves[0].to, taken.data.curves[1].from);
        assert!(taken.suggested.is_none());
        assert_eq!(taken.history.len(), 1);
        assert_eq!(undo_network(&taken).data.curves.len(), 0);
    }

    #[test]
    fn takes_one_leaves_the_rest_offered_and_ends_it_on_a_node_already_there() {
        let mut d = NetworkDraft::empty();
        d = add_curve(&d, NetworkStart::At(v(0.0, 0.0, 0.0)), &[], NetworkStart::At(v(0.0, 20.0, 0.0)));
        assert_eq!(d.data.nodes.len(), 2);
        d = set_suggested(&d, Some(vec![crease(v(0.3, 0.0, 0.0), v(20.0, 0.0, 0.0)), crease(v(20.0, 0.0, 0.0), v(20.0, 20.0, 0.0))]));
        let taken = accept_creases(&d, &[0]);
        assert_eq!(taken.data.nodes.len(), 3);
        assert_eq!(taken.data.curves[1].from, d.data.nodes[0].id);
        assert_eq!(taken.suggested.as_ref().map(Vec::len), Some(1));
        assert_eq!(taken.suggested.as_ref().map(|s| s[0].from), Some(v(20.0, 0.0, 0.0)));
        // An index past those offered takes nothing.
        assert_eq!(accept_creases(&taken, &[5]), taken);
    }

    // ---- the patch under the cursor (reNetworkPick.test.ts) ----

    /// Grids of patches sampled from (u, v) maps, as the build gives them.
    fn grids_of(maps: &[&dyn Fn(f64, f64) -> DVec3]) -> Vec<Vec<f32>> {
        let n = PATCH_GRID;
        maps.iter()
            .map(|at| {
                let mut g = vec![0.0f32; n * n * 3];
                for i in 0..n {
                    for j in 0..n {
                        let p = at(i as f64 / (n - 1) as f64, j as f64 / (n - 1) as f64);
                        g[(i * n + j) * 3] = p.x as f32;
                        g[(i * n + j) * 3 + 1] = p.y as f32;
                        g[(i * n + j) * 3 + 2] = p.z as f32;
                    }
                }
                g
            })
            .collect()
    }

    #[test]
    fn finds_the_patch_a_point_lies_on_and_none_beside_it() {
        let flat0 = |u: f64, w: f64| v(20.0 * u, 20.0 * w, 0.0);
        let flat20 = |u: f64, w: f64| v(20.0 + 20.0 * u, 20.0 * w, 0.0);
        let grids = grids_of(&[&flat0, &flat20]);
        let under = |p: DVec3| patch_under(grids.iter().map(Vec::as_slice), p);
        assert_eq!(under(v(5.0, 5.0, 0.05)), Some(0));
        assert_eq!(under(v(25.0, 15.0, -0.1)), Some(1));
        // Past the rim, though nearer the patch than a point above it may be.
        assert_eq!(under(v(-1.0, 5.0, 0.0)), None);
        assert_eq!(under(v(5.0, 21.0, 0.0)), None);
        // Far above it is no point of it.
        assert_eq!(under(v(5.0, 5.0, 8.0)), None);
    }

    #[test]
    fn reads_a_bent_patch_by_its_own_surface_not_its_corners_plane() {
        let dome = |u: f64, w: f64| v(20.0 * u, 20.0 * w, 6.0 * (std::f64::consts::PI * u).sin() * (std::f64::consts::PI * w).sin());
        let grids = grids_of(&[&dome]);
        assert_eq!(patch_under(grids.iter().map(Vec::as_slice), v(10.0, 10.0, 6.0)), Some(0));
        assert_eq!(patch_under(grids.iter().map(Vec::as_slice), v(10.0, 10.0, 0.0)), None);
    }

    #[test]
    fn rings_a_patch_round_its_rim() {
        let flat = |u: f64, w: f64| v(20.0 * u, 20.0 * w, 0.0);
        let grids = grids_of(&[&flat]);
        let ring = ring_of(&grids[0]);
        assert_eq!(ring.len(), 4 * (PATCH_GRID - 1) + 1);
        assert_eq!(ring.first(), ring.last());
        assert!(ring.iter().all(|p| p.x == 0.0 || p.x == 20.0 || p.y == 0.0 || p.y == 20.0));
    }

    // ---- the status line ----

    fn report(patches: usize, corners: &[f64], closed: bool) -> NetworkReport {
        let patch = ReportPatch { seed: DVec3::ZERO, rms: 0.01, max: 0.03, misfit: 0.0, net: ReportNet { u: 4, v: 4 }, points: 100, worst: Vec::new(), dropped: None, sides: Vec::new(), corners: Vec::new(), grid: Vec::new() };
        NetworkReport {
            patches: vec![patch; patches],
            cells: vec![CellOutcome::Refused { seed: DVec3::ZERO, refused: "2 sides".into(), at: None }],
            unfilled: vec![crate::network_cells::Unfilled { seed: DVec3::ZERO, sides: 4 }; 2],
            curves: Vec::new(),
            corners: corners.iter().map(|a| crate::network_fit::IrregularCorner { node: 1, at: DVec3::ZERO, patches: 3, angle: *a }).collect(),
            strips: Vec::new(),
            lost: Vec::new(),
            spokes: Vec::new(),
            centres: Vec::new(),
            bodies: Vec::new(),
            closed,
            worst_angle: 0.125,
            worst_gap: 0.0042,
            ms: ReportTimes::default(),
        }
    }

    #[test]
    fn reads_the_status_line_as_the_box_shows_it() {
        assert_eq!(network_status(None, 0, false, DisplayUnits::Mm), "No patches yet.");
        assert_eq!(network_status(None, 0, true, DisplayUnits::Mm), "");
        assert_eq!(network_status(None, 3, false, DisplayUnits::Mm), "3 curves · building…");
        assert_eq!(network_status(Some(&report(6, &[2.04, 3.06], true)), 12, false, DisplayUnits::Mm), "6 patches · 2 open cells · 1 refused · worst edge 0.13° · worst gap 4.2 µm · worst corner 3.1° · closed — a solid");
        assert_eq!(network_status(Some(&report(1, &[], false)), 4, false, DisplayUnits::In), "1 patch · 2 open cells · 1 refused · worst edge 0.13° · worst gap 0.00017 in");
    }

    // ---- a network in a project file (reNetworkProject.test.ts) ----

    #[test]
    fn a_network_in_a_project_file_opens_with_its_cells_their_corners_and_a_centre() {
        let mut d = square(&NetworkDraft::empty(), 0.0);
        d = add_patch(&d, &[PlacedCorner::Node(node_at(&d, v(10.0, 0.0, 0.0)).expect("n")), PlacedCorner::At(v(20.0, 0.0, 0.0)), PlacedCorner::Node(node_at(&d, v(10.0, 10.0, 0.0)).expect("n"))], true, Some(v(13.0, 3.0, 0.0)), None);
        d.data.cells[1].centre = Some(v(13.0, 3.0, 0.0));
        let mut part = crate::model::Part::default();
        part.add(crate::model::FeatureKind::Network(d.data.clone()));
        let back = crate::model::Part::from_json(&part.to_json()).expect("opens");
        match &back.features[0].kind {
            crate::model::FeatureKind::Network(f) => assert_eq!(f, &d.data),
            other => panic!("not a network: {}", other.kind_name()),
        }
        // A cell without its seed, or with corners that are no ids, is refused.
        let json: serde_json::Value = serde_json::from_str(&part.to_json()).expect("json");
        let mut no_seed = json.clone();
        no_seed["features"][0]["cells"] = serde_json::json!([{ "corners": [1, 2, 3] }]);
        assert!(crate::model::Part::from_json(&no_seed.to_string()).is_err());
        let mut bad = json;
        bad["features"][0]["cells"] = serde_json::json!([{ "seed": [0, 0, 0], "corners": [1.5] }]);
        assert!(crate::model::Part::from_json(&bad.to_string()).is_err());
    }
}
