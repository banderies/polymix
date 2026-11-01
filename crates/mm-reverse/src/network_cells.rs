// SPDX-License-Identifier: AGPL-3.0-only
//! The cells of a patch network: the regions of the scan its curves close.
//!
//! Ported from ScanRuler (`plugins/reverse/core/networkCells.ts`), © Stefan Hermann
//! (CNC Kitchen), AGPL-3.0-only.
//!
//! Two readings of the same network meet here. The curves as a graph (the nodes, and the
//! curves leaving each in the order they leave it round the scan's normal there) have faces:
//! walk along a curve, and at each node take the next curve clockwise from the one arrived
//! by, and the walk goes round the face on its left. That gives a face's sides in order and
//! its corners, but says nothing about which triangles it covers. The curves as cuts in the
//! scan do: each trace marks the triangle edges it crosses (a section's point lies on one; a
//! run along the edges marks the edges it runs along), and the triangles are flooded without
//! crossing a marked edge. Each piece of the flood is a region; the curve on either side of
//! the edges round it says which face of the graph it is.
//!
//! A face is a cell when its piece is closed (the flood stays inside the box round the
//! curves and meets the scan's rim only round holes wholly inside it, which the map fills
//! with a fan) and it has three to six sides. A node is a corner of the face where the
//! boundary turns more than 30° there, unless the node says otherwise: a T-node on a
//! straight run, where a neighbour was split, is not one.

use std::collections::{HashMap, HashSet};

use glam::DVec3;
use serde::{Deserialize, Serialize};

use crate::network_curve::{end_tangent, CurveFit, CurvePoints, CurveTrace};
use crate::scan_path::{PathWay, ScanPaths};

/// How far the boundary turns at a node for the node to be a corner, deg.
pub const CORNER_DEG: f64 = 30.0;
/// The fewest and most sides a cell is filled with.
pub const SIDES_MIN: usize = 3;
pub const SIDES_MAX: usize = 6;
/// How far past the box round the curves a piece may reach before it is taken for the
/// outside of the network: this share of the box's diagonal (a cell bulges past the curves
/// round it, a dome above its rim).
const BOX_MARGIN: f64 = 0.5;
/// The same for the regions no seed fills, which Fill all offers: a dome that bulges further
/// past its curves is filled by a click in it.
const UNFILLED_MARGIN: f64 = 0.25;
/// The share of the edges round a piece that have to agree on its face.
const FACE_AGREEMENT: f64 = 0.9;
/// A piece of fewer triangles than this is a crumb between curves at a node, not a cell.
const CRUMB_TRIANGLES: usize = 12;

/// A node as the cells read it: where it is on the scan.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: u32,
    pub at: DVec3,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub corner: Option<bool>,
}

/// A curve as the cells read it: its ends, its trace, and a fit for the directions it leaves
/// its ends by. A curve along a body's edge has the edge's own points (`exact`), which it is
/// fitted to; its trace on the scan is what cuts the cells.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphCurve {
    pub id: u32,
    pub from: u32,
    pub to: u32,
    pub trace: CurveTrace,
    pub fit: CurveFit,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exact: Option<CurvePoints>,
}

/// A curve run one way: forward from its `from` to its `to`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HalfEdge {
    pub curve: u32,
    pub forward: bool,
}

/// A face of the curve graph: the half-edges round it with it on their left, in order, and
/// the node each arrives at.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphFace {
    pub edges: Vec<HalfEdge>,
    pub nodes: Vec<u32>,
}

/// A cell's side: the curves along it from one corner to the next.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CellSide {
    pub edges: Vec<HalfEdge>,
}

/// A filled cell.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CellFill {
    pub seed: DVec3,
    pub face: usize,
    /// The corners' nodes, in order round the cell with it on the left.
    pub corners: Vec<u32>,
    pub sides: Vec<CellSide>,
    /// The scan's triangles in the cell.
    pub triangles: Vec<u32>,
    /// The rim loops of scan holes wholly inside it, by scan vertex.
    pub holes: Vec<Vec<u32>>,
}

/// A region that is not a cell, and why.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CellRefusal {
    pub seed: DVec3,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub at: Option<DVec3>,
}

/// A filled cell, or why the region is not one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "lowercase")]
pub enum CellReading {
    Ok(CellFill),
    Refused(CellRefusal),
}

impl CellReading {
    pub fn seed(&self) -> DVec3 {
        match self {
            CellReading::Ok(c) => c.seed,
            CellReading::Refused(r) => r.seed,
        }
    }

    pub fn ok(&self) -> Option<&CellFill> {
        match self {
            CellReading::Ok(c) => Some(c),
            CellReading::Refused(_) => None,
        }
    }

    fn refused(seed: DVec3, message: impl Into<String>, at: Option<DVec3>) -> Self {
        CellReading::Refused(CellRefusal { seed, message: message.into(), at })
    }
}

/// A closed region of three to six sides no seed fills, with a point inside it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Unfilled {
    pub seed: DVec3,
    pub sides: usize,
}

/// The graph's faces, and the scan read into regions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NetworkCells {
    pub faces: Vec<GraphFace>,
    pub cells: Vec<CellReading>,
    /// Closed regions of three to six sides no seed fills, with a point inside each: what
    /// Fill all fills.
    pub unfilled: Vec<Unfilled>,
}

/// The marks the curves leave on the scan's edges.
struct Marks {
    /// By edge key: the marks on that edge; and the keys in the order first marked.
    by_edge: HashMap<u64, Vec<usize>>,
    order: Vec<u64>,
    curve: Vec<u32>,
    /// Where the curve crossed or ran along the edge, and which way; NaN where a run's end
    /// triangle is marked all round.
    at: Vec<f64>,
    dir: Vec<f64>,
}

/// The triangles round each vertex (CSR), made once a scan.
pub(crate) struct Incidence {
    pub offsets: Vec<u32>,
    pub list: Vec<u32>,
}

pub(crate) fn incidence(scan: &ScanPaths) -> std::sync::Arc<Incidence> {
    scan.memo(|scan| {
        let vertex_count = scan.vertex_count();
        let idx = scan.indices();
        let mut offsets = vec![0u32; vertex_count + 1];
        for &v in idx {
            offsets[v as usize + 1] += 1;
        }
        for v in 0..vertex_count {
            offsets[v + 1] += offsets[v];
        }
        let mut list = vec![0u32; idx.len()];
        let mut cursor: Vec<u32> = offsets[..vertex_count].to_vec();
        for (i, &v) in idx.iter().enumerate() {
            list[cursor[v as usize] as usize] = (i / 3) as u32;
            cursor[v as usize] += 1;
        }
        Incidence { offsets, list }
    })
}

/// The triangles across edge (a, b) from t: one on a manifold edge, none on the rim.
fn across(scan: &ScanPaths, inc: &Incidence, t: u32, a: u32, b: u32, out: &mut Vec<u32>) {
    out.clear();
    let idx = scan.indices();
    for k in inc.offsets[a as usize]..inc.offsets[a as usize + 1] {
        let u = inc.list[k as usize];
        if u == t {
            continue;
        }
        let u3 = u as usize * 3;
        if idx[u3] == b || idx[u3 + 1] == b || idx[u3 + 2] == b {
            out.push(u);
        }
    }
}

/// Every edge the curves cross or run along, with the curve and its direction there; and the
/// triangles a run along the edges starts or ends in, all of whose edges are marked: the run
/// joins its end through the triangle's inside, which would otherwise be a way round it.
fn mark_edges(scan: &ScanPaths, curves: &[GraphCurve]) -> Marks {
    let vertex_count = scan.vertex_count() as u64;
    let mut marks = Marks { by_edge: HashMap::new(), order: Vec::new(), curve: Vec::new(), at: Vec::new(), dir: Vec::new() };
    let key = |a: u32, b: u32| if a < b { a as u64 * vertex_count + b as u64 } else { b as u64 * vertex_count + a as u64 };
    let mark = |marks: &mut Marks, a: u32, b: u32, curve: u32, p: Option<DVec3>, d: Option<DVec3>| {
        let k = key(a, b);
        let m = marks.curve.len();
        marks.curve.push(curve);
        let p = p.unwrap_or(DVec3::NAN);
        let d = d.unwrap_or(DVec3::NAN);
        marks.at.extend_from_slice(&[p.x, p.y, p.z]);
        marks.dir.extend_from_slice(&[d.x, d.y, d.z]);
        match marks.by_edge.get_mut(&k) {
            Some(list) => list.push(m),
            None => {
                marks.by_edge.insert(k, vec![m]);
                marks.order.push(k);
            }
        }
    };
    let vc = scan.vertex_count() as i32;
    for c in curves {
        let t = &c.trace;
        let n = (t.points.len() / 3) as i64;
        let on = |i: usize| t.on.get(i).copied().unwrap_or(-1);
        for run in &t.runs {
            if run.way == PathWay::Section {
                let from = (run.from as i64).max(1);
                let to = (run.to as i64).min(n - 2);
                for i in from..=to {
                    let i = i as usize;
                    let a = on(i * 2);
                    let b = on(i * 2 + 1);
                    if a < 0 || a == b || a >= vc || b < 0 || b >= vc {
                        continue;
                    }
                    let q0 = t.point(i - 1);
                    let q1 = t.point(i + 1);
                    mark(&mut marks, a as u32, b as u32, c.id, Some(t.point(i)), Some(q1 - q0));
                }
            } else {
                for i in run.from..run.to {
                    let a = on(i * 2);
                    let b = on((i + 1) * 2);
                    if a < 0 || b < 0 || a != on(i * 2 + 1) || b != on((i + 1) * 2 + 1) || a == b || a >= vc || b >= vc {
                        continue;
                    }
                    let pa = scan.vertex(a as u32);
                    let pb = scan.vertex(b as u32);
                    mark(&mut marks, a as u32, b as u32, c.id, Some(pa), Some(pb - pa));
                }
                for &tri in &run.ends {
                    if tri < 0 || tri as usize >= scan.triangle_count() {
                        continue;
                    }
                    let v = scan.triangle(tri as u32);
                    for e in 0..3 {
                        mark(&mut marks, v[e], v[(e + 1) % 3], c.id, None, None);
                    }
                }
            }
        }
    }
    marks
}

/// The graph's faces from the order the curves leave each node in.
pub fn graph_faces(scan: &ScanPaths, nodes: &[GraphNode], curves: &[GraphCurve]) -> Vec<GraphFace> {
    // At each node, the curves leaving it (a curve from a node to itself twice) by their
    // angle round the scan's normal there.
    let mut around: HashMap<u32, Vec<(HalfEdge, f64)>> = HashMap::new();
    let node_at: HashMap<u32, &GraphNode> = nodes.iter().map(|n| (n.id, n)).collect();
    let mut add = |node: u32, edge: HalfEdge, dir: DVec3| {
        let Some(n) = node_at.get(&node) else { return };
        let normal = match scan.closest(n.at) {
            Some(c) => scan.normal_at(c.point, c.triangle),
            None => DVec3::Z,
        };
        let e1 = perpendicular(normal);
        let e2 = DVec3::new(normal.y * e1.z - normal.z * e1.y, normal.z * e1.x - normal.x * e1.z, normal.x * e1.y - normal.y * e1.x);
        let angle = dir.dot(e2).atan2(dir.dot(e1));
        around.entry(node).or_default().push((edge, angle));
    };
    for c in curves {
        let out = end_tangent(&c.fit, 0);
        let into = end_tangent(&c.fit, 1);
        add(c.from, HalfEdge { curve: c.id, forward: true }, out);
        add(c.to, HalfEdge { curve: c.id, forward: false }, -into);
    }
    for list in around.values_mut() {
        list.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    }
    let curve_of: HashMap<u32, &GraphCurve> = curves.iter().map(|c| (c.id, c)).collect();
    let arrives = |h: HalfEdge| curve_of.get(&h.curve).map(|c| if h.forward { c.to } else { c.from });
    // The half-edge after h round its face: at the node h arrives at, the curve next
    // clockwise from the one h came by.
    let next = |h: HalfEdge| -> Option<HalfEdge> {
        let list = around.get(&arrives(h)?)?;
        let back = list.iter().position(|l| l.0.curve == h.curve && l.0.forward == !h.forward)?;
        Some(list[(back + list.len() - 1) % list.len()].0)
    };
    let mut seen: HashSet<HalfEdge> = HashSet::new();
    let mut faces = Vec::new();
    for c in curves {
        for forward in [true, false] {
            let start = HalfEdge { curve: c.id, forward };
            if seen.contains(&start) {
                continue;
            }
            let mut edges = Vec::new();
            let mut nodes_of = Vec::new();
            let mut h = Some(start);
            let mut guard = 0;
            while let Some(e) = h {
                if seen.contains(&e) || guard >= 4 * curves.len() + 4 {
                    break;
                }
                seen.insert(e);
                edges.push(e);
                if let Some(n) = arrives(e) {
                    nodes_of.push(n);
                }
                h = next(e);
                guard += 1;
            }
            if !edges.is_empty() {
                faces.push(GraphFace { edges, nodes: nodes_of });
            }
        }
    }
    faces
}

fn perpendicular(n: DVec3) -> DVec3 {
    let a = if n.x.abs() < 0.9 { DVec3::X } else { DVec3::Y };
    let d = a.dot(n);
    let u = a - d * n;
    let l = u.length();
    let l = if l != 0.0 && !l.is_nan() { l } else { 1.0 };
    u / l
}

/// Why a face has no sides to fill, and where.
#[derive(Debug, Clone, PartialEq)]
pub struct FaceRefusal {
    pub message: String,
    pub at: Option<DVec3>,
}

/// A face's corners and sides, or why it has none to fill: its corners are the nodes where
/// it turns more than CORNER_DEG, or as a node says; or, given, exactly the nodes of
/// `corners` (a quad of a split cell).
pub fn face_sides(face: &GraphFace, nodes: &[GraphNode], curves: &[GraphCurve], corners_given: Option<&HashSet<u32>>) -> Result<(Vec<u32>, Vec<CellSide>), FaceRefusal> {
    let curve_of: HashMap<u32, &GraphCurve> = curves.iter().map(|c| (c.id, c)).collect();
    let node_of: HashMap<u32, &GraphNode> = nodes.iter().map(|n| (n.id, n)).collect();
    let m = face.edges.len();
    // A curve run both ways round one face ends inside it, at a node only that curve reaches.
    if face.edges.iter().any(|h| face.edges.iter().any(|g| g.curve == h.curve && g.forward != h.forward)) {
        let mut valence: HashMap<u32, usize> = HashMap::new();
        for c in curves {
            *valence.entry(c.from).or_default() += 1;
            *valence.entry(c.to).or_default() += 1;
        }
        let dangling = face.nodes.iter().filter_map(|id| node_of.get(id)).find(|n| valence.get(&n.id) == Some(&1));
        let message = match dangling {
            Some(n) => format!("a curve ends inside it at {} — take it on to another curve, or delete it", where_text(n.at)),
            None => "a curve ends inside it — take it on to another curve, or delete it".to_string(),
        };
        return Err(FaceRefusal { message, at: dangling.map(|n| n.at) });
    }
    let leave = |h: HalfEdge| -> DVec3 {
        let Some(c) = curve_of.get(&h.curve) else { return DVec3::X };
        if h.forward {
            end_tangent(&c.fit, 0)
        } else {
            -end_tangent(&c.fit, 1)
        }
    };
    let mut corner_after: Vec<usize> = Vec::new();
    for i in 0..m {
        let h = face.edges[i];
        let g = face.edges[(i + 1) % m];
        let Some(&node) = face.nodes.get(i) else { continue };
        // Arriving along h is leaving by its reverse, backwards.
        let back = leave(HalfEdge { curve: h.curve, forward: !h.forward });
        let out = leave(g);
        let cos = -back.dot(out);
        let turn = cos.clamp(-1.0, 1.0).acos().to_degrees();
        let given = match corners_given {
            Some(set) => Some(set.contains(&node)),
            None => node_of.get(&node).and_then(|n| n.corner),
        };
        if given.unwrap_or(turn > CORNER_DEG) {
            corner_after.push(i);
        }
    }
    if corner_after.is_empty() {
        return Err(FaceRefusal { message: "one side — a closed curve with no corner; add a node where it turns".to_string(), at: None });
    }
    // The sides run from each corner to the next.
    let mut sides = Vec::with_capacity(corner_after.len());
    for k in 0..corner_after.len() {
        let from = corner_after[k];
        let to = corner_after[(k + 1) % corner_after.len()];
        let mut edges = Vec::new();
        let mut i = (from + 1) % m;
        loop {
            edges.push(face.edges[i]);
            if i == to {
                break;
            }
            i = (i + 1) % m;
        }
        sides.push(CellSide { edges });
    }
    // The first corner is the node the last side arrives at.
    let ordered = corner_after.iter().map(|&i| face.nodes[i]).collect();
    Ok((ordered, sides))
}

/// A number as JavaScript's `toFixed(1)` writes it. Both round the number's exact decimal
/// value to the nearest tenth, and part only where that value lies exactly halfway between
/// two tenths, which for a double is an odd number of quarters (0.25, 0.75, 1.25 …):
/// `toFixed` takes the larger magnitude there, Rust's formatting the even digit. A number that
/// comes to a half only once multiplied by ten is no tie: 0.35 is 0.3499…, and both write
/// "0.3". A negative zero loses its sign; NaN, the infinities and magnitudes from 1e21 on are
/// written as JavaScript writes them.
pub(crate) fn fixed1(x: f64) -> String {
    if x.is_nan() {
        return "NaN".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 { "Infinity" } else { "-Infinity" }.to_string();
    }
    if x.abs() >= 1e21 {
        // JavaScript's own number to string: the shortest digits, the exponent signed.
        let s = format!("{x:e}");
        return match s.split_once('e') {
            Some((m, e)) if !e.starts_with('-') => format!("{m}e+{e}"),
            _ => s,
        };
    }
    let quarters = x * 4.0;
    let tie = quarters.fract() == 0.0 && quarters % 2.0 != 0.0;
    let x = if tie { (x * 10.0).round() / 10.0 } else { x };
    let x = if x == 0.0 { 0.0 } else { x };
    format!("{x:.1}")
}

/// JavaScript's `Math.round`: the nearest whole number, a half rounded up (toward +∞).
pub(crate) fn js_round(x: f64) -> f64 {
    let f = x.floor();
    if x - f >= 0.5 {
        f + 1.0
    } else {
        f
    }
}

/// A point as a refusal names it: each coordinate `Math.round(x * 10) / 10`, then
/// `toFixed(1)`, as ScanRuler's `where`.
pub(crate) fn where_text(p: DVec3) -> String {
    let r = |x: f64| fixed1(js_round(x * 10.0) / 10.0);
    format!("({}, {}, {})", r(p.x), r(p.y), r(p.z))
}

/// A piece of the flood: its triangles, whether it reached past the box and where, the rim
/// edges it met, and the votes of the marked edges round it for the half-edges it lies left
/// of (in the order first voted).
struct Piece {
    triangles: Vec<u32>,
    leaked: Option<DVec3>,
    rim: Vec<(u32, u32)>,
    votes: Vec<(HalfEdge, usize)>,
}

/// Whether the scan has no rim: every edge has a triangle on both sides; made once a scan.
struct ScanClosed(bool);

fn scan_is_closed(scan: &ScanPaths) -> bool {
    scan.memo(|scan| {
        let inc = incidence(scan);
        let idx = scan.indices();
        let mut nb = Vec::new();
        for t in 0..scan.triangle_count() {
            for e in 0..3 {
                across(scan, &inc, t as u32, idx[t * 3 + e], idx[t * 3 + (e + 1) % 3], &mut nb);
                if nb.is_empty() {
                    return ScanClosed(false);
                }
            }
        }
        ScanClosed(true)
    })
    .0
}

/// The network's cells: the faces, each seed's region read (see the module documentation),
/// and the closed regions no seed fills. `corners_of`, by seed, gives a cell its corners
/// instead of the rule.
pub fn read_cells(scan: &ScanPaths, nodes: &[GraphNode], curves: &[GraphCurve], seeds: &[DVec3], corners_of: &[Option<HashSet<u32>>]) -> NetworkCells {
    let faces = graph_faces(scan, nodes, curves);
    if curves.is_empty() {
        let cells = seeds.iter().map(|&seed| CellReading::refused(seed, "no curves close it", None)).collect();
        return NetworkCells { faces, cells, unfilled: Vec::new() };
    }
    let inc = incidence(scan);
    let marks = mark_edges(scan, curves);
    let idx = scan.indices();
    let vertex_count = scan.vertex_count() as u64;
    let edge_key = |a: u32, b: u32| if a < b { a as u64 * vertex_count + b as u64 } else { b as u64 * vertex_count + a as u64 };
    // The box round the curves, grown by the margin.
    let mut lo = DVec3::INFINITY;
    let mut hi = DVec3::NEG_INFINITY;
    for c in curves {
        for i in 0..c.trace.points.len() / 3 {
            let p = c.trace.point(i);
            lo = lo.min(p);
            hi = hi.max(p);
        }
    }
    let diagonal = (hi - lo).length();
    let margin = BOX_MARGIN * diagonal + 4.0 * scan.edge();
    let near = UNFILLED_MARGIN * diagonal + 4.0 * scan.edge();
    let near_lo = lo - DVec3::splat(near);
    let near_hi = hi + DVec3::splat(near);
    let lo = lo - DVec3::splat(margin);
    let hi = hi + DVec3::splat(margin);
    let centroid = |t: u32| {
        let [a, b, c] = scan.triangle(t);
        (scan.vertex(a) + scan.vertex(b) + scan.vertex(c)) / 3.0
    };
    let box_test = |a: DVec3, b: DVec3, t: u32| {
        let c = centroid(t);
        (0..3).all(|k| !(c[k] < a[k] || c[k] > b[k]))
    };
    let in_box = |t: u32| box_test(lo, hi, t);
    let in_near = |t: u32| box_test(near_lo, near_hi, t);
    let mut face_of: HashMap<HalfEdge, usize> = HashMap::new();
    for (i, f) in faces.iter().enumerate() {
        for h in &f.edges {
            face_of.insert(*h, i);
        }
    }
    let mut label = vec![-1i32; scan.triangle_count()];
    let mut pieces: Vec<Piece> = Vec::new();
    let mut nb: Vec<u32> = Vec::new();

    // A piece flooded from a triangle, inside `within`; `stop`: given up the moment it leaks,
    // a piece left open being no cell whatever else it holds, and one that runs into a piece
    // given up so is open too.
    let mut flood = |label: &mut Vec<i32>, pieces: &mut Vec<Piece>, start: u32, within: &dyn Fn(u32) -> bool, stop: bool| -> usize {
        let id = pieces.len();
        let mut piece = Piece { triangles: Vec::new(), leaked: None, rim: Vec::new(), votes: Vec::new() };
        let mut queue = vec![start];
        label[start as usize] = id as i32;
        while !(stop && piece.leaked.is_some()) {
            let Some(t) = queue.pop() else { break };
            piece.triangles.push(t);
            let tv = scan.triangle(t);
            for e in 0..3 {
                let a = tv[e];
                let b = tv[(e + 1) % 3];
                let w = tv[(e + 2) % 3];
                if let Some(marked) = marks.by_edge.get(&edge_key(a, b)) {
                    for &m in marked {
                        let d = DVec3::new(marks.dir[m * 3], marks.dir[m * 3 + 1], marks.dir[m * 3 + 2]);
                        if d.x.is_nan() {
                            continue;
                        }
                        // Left of the curve, seen from outside: on the side n × d.
                        let n = scan.triangle_normal(t);
                        let left = DVec3::new(n.y * d.z - n.z * d.y, n.z * d.x - n.x * d.z, n.x * d.y - n.y * d.x);
                        let at = DVec3::new(marks.at[m * 3], marks.at[m * 3 + 1], marks.at[m * 3 + 2]);
                        let side = (scan.vertex(w) - at).dot(left);
                        let k = HalfEdge { curve: marks.curve[m], forward: side > 0.0 };
                        match piece.votes.iter_mut().find(|v| v.0 == k) {
                            Some(v) => v.1 += 1,
                            None => piece.votes.push((k, 1)),
                        }
                    }
                    continue;
                }
                across(scan, &inc, t, a, b, &mut nb);
                if nb.is_empty() {
                    piece.rim.push((a, b));
                    continue;
                }
                for &u in nb.iter() {
                    let lu = label[u as usize];
                    if lu >= 0 {
                        if lu as usize != id && piece.leaked.is_none() {
                            if let Some(leak) = pieces.get(lu as usize).and_then(|p| p.leaked) {
                                piece.leaked = Some(leak);
                            }
                        }
                        continue;
                    }
                    if !within(u) {
                        if piece.leaked.is_none() {
                            piece.leaked = Some(scan.vertex(a));
                        }
                        continue;
                    }
                    label[u as usize] = id as i32;
                    queue.push(u);
                }
            }
        }
        pieces.push(piece);
        id
    };

    // Where a region is open: the node nearest it that ends one curve alone, or where the
    // flood left the box.
    let mut curve_count: HashMap<u32, usize> = HashMap::new();
    for c in curves {
        *curve_count.entry(c.from).or_default() += 1;
        *curve_count.entry(c.to).or_default() += 1;
    }
    let ends_one = |id: u32| curve_count.get(&id).copied().unwrap_or(0) == 1;
    let open_at = |near: DVec3, fallback: DVec3| -> DVec3 {
        let mut best: Option<DVec3> = None;
        let mut best_d = f64::INFINITY;
        for n in nodes {
            if !ends_one(n.id) {
                continue;
            }
            let d = (n.at - near).length();
            if d < best_d {
                best_d = d;
                best = Some(n.at);
            }
        }
        best.unwrap_or(fallback)
    };
    // A loop's signed turning round the scan's normal, radians.
    let turning = |lp: &[u32]| -> f64 {
        let mut sum = 0.0;
        let m = lp.len();
        for i in 0..m {
            let p = scan.vertex(lp[(i + m - 1) % m]);
            let q = scan.vertex(lp[i]);
            let r = scan.vertex(lp[(i + 1) % m]);
            let a = q - p;
            let b = r - q;
            let n = scan.normal_at(q, 0);
            let c = DVec3::new(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
            sum += c.dot(n).atan2(a.dot(b));
        }
        sum
    };
    // The rim loops a piece meets, and whether each is wholly inside it.
    let holes_of = |piece: &Piece| -> Option<Vec<Vec<u32>>> {
        if piece.rim.is_empty() {
            return Some(Vec::new());
        }
        let mut next_of: HashMap<u32, Vec<u32>> = HashMap::new();
        for &(a, b) in &piece.rim {
            next_of.entry(a).or_default().push(b);
        }
        let mut used: HashSet<(u32, u32)> = HashSet::new();
        let mut loops = Vec::new();
        for &(a0, b0) in &piece.rim {
            if used.contains(&(a0, b0)) {
                continue;
            }
            let mut lp = vec![a0];
            let mut a = a0;
            let mut b = b0;
            loop {
                used.insert((a, b));
                if b == a0 {
                    break;
                }
                lp.push(b);
                let option = next_of.get(&b).and_then(|l| l.iter().copied().find(|&x| !used.contains(&(b, x))))?;
                a = b;
                b = option;
            }
            // A hole runs round with the region on its left and itself on the right:
            // clockwise seen from outside, a turn of −360°. A loop that turns the other way
            // round holds the region in: the scan's own edge is the region's boundary, not a
            // hole in it.
            if turning(&lp) > 0.0 {
                return None;
            }
            loops.push(lp);
        }
        Some(loops)
    };

    // The triangles a curve ends in alone: a region that holds one runs out through the gap
    // there.
    let mut dangling_in: HashMap<u32, DVec3> = HashMap::new();
    for n in nodes {
        if !ends_one(n.id) {
            continue;
        }
        if let Some(c) = scan.closest_inside(n.at) {
            dangling_in.insert(c.triangle, n.at);
        }
    }
    let gap_in = |piece: &Piece| -> Option<DVec3> { piece.triangles.iter().find_map(|t| dangling_in.get(t).copied()) };
    let read = |piece: &Piece, seed: DVec3, corners: Option<&HashSet<u32>>| -> CellReading {
        if let Some(leaked) = piece.leaked {
            let at = gap_in(piece).unwrap_or_else(|| open_at(seed, leaked));
            return CellReading::refused(seed, format!("not closed — open at {}", where_text(at)), Some(at));
        }
        let Some(holes) = holes_of(piece) else {
            if let Some(gap) = gap_in(piece) {
                return CellReading::refused(seed, format!("not closed — open at {}", where_text(gap)), Some(gap));
            }
            let at = piece.rim.first().map(|r| scan.vertex(r.0));
            return CellReading::refused(seed, "the scan’s rim runs through it", at);
        };
        let mut total = 0usize;
        let mut by_face: Vec<(usize, usize)> = Vec::new();
        for (k, v) in &piece.votes {
            total += v;
            if let Some(&f) = face_of.get(k) {
                match by_face.iter_mut().find(|e| e.0 == f) {
                    Some(e) => e.1 += v,
                    None => by_face.push((f, *v)),
                }
            }
        }
        let mut face: Option<usize> = None;
        let mut best = 0usize;
        for &(f, v) in &by_face {
            if v > best {
                best = v;
                face = Some(f);
            }
        }
        let face = match face {
            Some(f) if (best as f64) >= FACE_AGREEMENT * total as f64 => f,
            // The votes split between faces: two curves cross with no node.
            _ => return CellReading::refused(seed, "two curves cross where no node joins them — snap one to the other", None),
        };
        let (corners, sides) = match face_sides(&faces[face], nodes, curves, corners) {
            Ok(s) => s,
            Err(r) => return CellReading::refused(seed, r.message, r.at),
        };
        let count = corners.len();
        if count < SIDES_MIN {
            return CellReading::refused(seed, if count == 2 { "two sides" } else { "one side" }, None);
        }
        if count > SIDES_MAX {
            return CellReading::refused(seed, format!("{count} sides — split it with a curve"), None);
        }
        CellReading::Ok(CellFill { seed, face, corners, sides, triangles: piece.triangles.clone(), holes })
    };

    // Each seed's region.
    let mut cells = Vec::with_capacity(seeds.len());
    let mut seeded: HashSet<usize> = HashSet::new();
    for (k, &seed) in seeds.iter().enumerate() {
        let Some(c) = scan.closest(seed) else {
            cells.push(CellReading::refused(seed, "not on the scan", None));
            continue;
        };
        let l = label[c.triangle as usize];
        let id = if l >= 0 { l as usize } else { flood(&mut label, &mut pieces, c.triangle, &in_box, false) };
        let reading = read(&pieces[id], seed, corners_of.get(k).and_then(|c| c.as_ref()));
        // A second seed in a region filled already fills nothing more.
        if seeded.contains(&id) && matches!(reading, CellReading::Ok(_)) {
            cells.push(CellReading::refused(seed, "filled already by another seed", None));
            continue;
        }
        seeded.insert(id);
        cells.push(reading);
    }
    // The rest of the regions round the curves, for Fill all: every closed one of three to
    // six sides, but the largest on a closed scan, which is the network's outside. Each
    // touches a curve, so each is flooded from a triangle beside one, in the nearer box and
    // given up as soon as it leaks: flooded to the seeds' box, the network's outside alone
    // was most of the scan round it (0.2 s of every read on a 2-million-triangle scan, for a
    // count that is nearly always nought).
    for &k in &marks.order {
        let a = (k / vertex_count) as u32;
        let b = (k - a as u64 * vertex_count) as u32;
        for i in inc.offsets[a as usize]..inc.offsets[a as usize + 1] {
            let t = inc.list[i as usize];
            if label[t as usize] >= 0 || !in_near(t) {
                continue;
            }
            let t3 = t as usize * 3;
            if idx[t3] == b || idx[t3 + 1] == b || idx[t3 + 2] == b {
                flood(&mut label, &mut pieces, t, &in_near, true);
            }
        }
    }
    struct Candidate {
        id: usize,
        size: usize,
        sides: usize,
        seeded: bool,
    }
    let mut candidates: Vec<Candidate> = Vec::new();
    for (id, piece) in pieces.iter().enumerate() {
        if piece.triangles.len() < CRUMB_TRIANGLES && !seeded.contains(&id) {
            continue;
        }
        if let CellReading::Ok(r) = read(piece, DVec3::ZERO, None) {
            candidates.push(Candidate { id, size: piece.triangles.len(), sides: r.corners.len(), seeded: seeded.contains(&id) });
        }
    }
    // On a closed scan the outside of the network is closed too; it is the largest region,
    // unless a seed was put in it.
    if !candidates.is_empty() && scan_is_closed(scan) {
        let mut largest = 0;
        for (i, c) in candidates.iter().enumerate() {
            if c.size > candidates[largest].size {
                largest = i;
            }
        }
        candidates.remove(largest);
    }
    let unfilled = candidates.iter().filter(|c| !c.seeded).map(|c| Unfilled { seed: inside_point(scan, &pieces[c.id].triangles), sides: c.sides }).collect();
    NetworkCells { faces, cells, unfilled }
}

/// A point well inside a region: the middle of its triangle nearest the region's centroid.
fn inside_point(scan: &ScanPaths, triangles: &[u32]) -> DVec3 {
    let mid = |t: u32| {
        let [a, b, c] = scan.triangle(t);
        (scan.vertex(a) + scan.vertex(b) + scan.vertex(c)) / 3.0
    };
    if triangles.is_empty() {
        return DVec3::NAN;
    }
    let mut c = DVec3::ZERO;
    for &t in triangles {
        c += mid(t);
    }
    c /= triangles.len() as f64;
    let mut best = triangles[0];
    let mut best_d = f64::INFINITY;
    for &t in triangles {
        let d = (mid(t) - c).length();
        if d < best_d {
            best_d = d;
            best = t;
        }
    }
    mid(best)
}

/// Ported from ScanRuler's `plugins/reverse/tests/reNetworkCells.test.ts`: a patch network's
/// cells and their map onto the square (PN5, PN7). A 2 × 2 network on a sheet reads into four
/// four-sided cells and nothing else closed; a curve taken out opens a cell, named where; a
/// strip round 270° of a cylinder (one cell that folds back over itself) maps one-to-one, as
/// a saddle does; a scan hole inside a cell is closed by a fan; a cell of two sides and one
/// of seven are refused by name.
#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use super::*;
    use crate::network::{trace_network, TracedNetwork};
    use crate::network_feature::NetworkFeature;
    use crate::network_map::{cell_mesh, map_cell, CellMap};
    use crate::network_meshes::{curve, curve_via, grid_network, grid_scan, network, GridOptions, TestMesh};

    fn cells_of(scan: &ScanPaths, f: &NetworkFeature) -> (TracedNetwork, NetworkCells) {
        let t = trace_network(scan, &f.nodes, &f.curves, None, None);
        let seeds: Vec<DVec3> = f.cells.iter().map(|c| c.seed).collect();
        let cells = read_cells(scan, &t.nodes, &t.curves, &seeds, &[]);
        (t, cells)
    }

    fn mapped(scan: &ScanPaths, traced: &TracedNetwork, cell: &CellReading) -> Result<CellMap, String> {
        let cell = match cell {
            CellReading::Ok(c) => c,
            CellReading::Refused(r) => return Err(r.message.clone()),
        };
        let at: HashMap<u32, DVec3> = traced.nodes.iter().map(|n| (n.id, n.at)).collect();
        let corners: Vec<DVec3> = cell.corners.iter().map(|c| at[c]).collect();
        let mesh = cell_mesh(scan, &cell.triangles, &cell.holes, &corners)?;
        map_cell(mesh, None)
    }

    fn sheet(u: f64, v: f64) -> DVec3 {
        DVec3::new(100.0 * u, 80.0 * v, 3.0 * (2.0 * u).sin() * v.cos())
    }

    fn scan_of(m: &TestMesh) -> ScanPaths {
        ScanPaths::new(m.positions.clone(), m.indices.clone())
    }

    fn status(c: &CellReading) -> &'static str {
        match c {
            CellReading::Ok(_) => "ok",
            CellReading::Refused(_) => "refused",
        }
    }

    /// Not one of ScanRuler's tests: the numbers a refusal names, written as JavaScript's
    /// `toFixed(1)` and `Math.round` write them (the table read off Node).
    #[test]
    fn writes_numbers_as_javascript_does() {
        let table: [(f64, &str); 28] = [
            (0.35, "0.3"),
            (0.15, "0.1"),
            (12.35, "12.3"),
            (-12.35, "-12.3"),
            (0.25, "0.3"),
            (-0.25, "-0.3"),
            (0.75, "0.8"),
            (1.25, "1.3"),
            (-1.75, "-1.8"),
            (2.45, "2.5"),
            (0.05, "0.1"),
            (-0.04, "-0.0"),
            (-0.0, "0.0"),
            (1e21, "1e+21"),
            (1.5e22, "1.5e+22"),
            (-2.5e30, "-2.5e+30"),
            (f64::NAN, "NaN"),
            (f64::INFINITY, "Infinity"),
            (f64::NEG_INFINITY, "-Infinity"),
            (123.456, "123.5"),
            (-7.05, "-7.0"),
            (0.95, "0.9"),
            (9.95, "9.9"),
            (99.95, "100.0"),
            (1e20 + 0.25, "100000000000000000000.0"),
            (3.0000000000000004, "3.0"),
            (1023.75, "1023.8"),
            (-0.05, "-0.1"),
        ];
        for (x, want) in table {
            assert_eq!(fixed1(x), want, "{x}");
        }
        for (x, want) in [(0.49999999999999994, 0.0), (0.45, 0.0), (-0.45, 0.0), (2.5, 3.0), (-2.5, -2.0)] {
            assert_eq!(js_round(x), want, "{x}");
        }
        assert_eq!(where_text(DVec3::new(0.049999999999999996, -0.06, 12.35)), "(0.0, -0.1, 12.4)");
    }

    #[test]
    fn reads_a_2x2_network_on_a_sheet_into_four_four_sided_cells() {
        let scan = scan_of(&grid_scan(120, 96, &sheet, &GridOptions::default()));
        let f = grid_network(2, 2, &sheet, 1, [0.15, 0.85, 0.15, 0.85]);
        let (_, cells) = cells_of(&scan, &f);
        assert_eq!(cells.cells.iter().map(status).collect::<Vec<_>>(), vec!["ok"; 4]);
        for c in &cells.cells {
            if let CellReading::Ok(c) = c {
                assert_eq!(c.sides.len(), 4);
            }
        }
        // Each cell its own share of the sheet, about a quarter of the network's 70 × 56 mm.
        let counts: Vec<usize> = cells.cells.iter().map(|c| c.ok().map(|c| c.triangles.len()).unwrap_or(0)).collect();
        for &n in &counts {
            assert!((n as f64 / counts[0] as f64 - 1.0).abs() < 0.05, "{counts:?}");
        }
        // Nothing else is closed: the outside runs off to the sheet's rim.
        assert_eq!(cells.unfilled, vec![]);
    }

    #[test]
    fn offers_the_cells_no_seed_fills_and_names_where_a_cell_is_open() {
        let scan = scan_of(&grid_scan(120, 96, &sheet, &GridOptions::default()));
        let f = grid_network(2, 2, &sheet, 0, [0.15, 0.85, 0.15, 0.85]);
        let mut two = f.clone();
        two.cells.truncate(2);
        assert_eq!(cells_of(&scan, &two).1.unfilled.len(), 2);
        // The first row's left curve taken away: the first cell opens.
        let mut open = f.clone();
        open.curves.retain(|c| !(c.from == 1 && c.to == 4));
        let reading = cells_of(&scan, &open).1.cells[0].clone();
        match reading {
            CellReading::Refused(r) => assert!(r.message.starts_with("not closed — open at ("), "{}", r.message),
            CellReading::Ok(_) => panic!("the cell should be open"),
        }
    }

    #[test]
    fn maps_a_strip_round_270_degrees_of_a_cylinder_one_to_one() {
        let r = 20.0;
        // The scan: the whole cylinder, 40 mm tall; the cell: 270° of it, 30 mm tall, its arcs
        // drawn through grips a quarter turn apart.
        let scan = scan_of(&grid_scan(180, 40, &|u, v| DVec3::new(r * (2.0 * PI * u).cos(), r * (2.0 * PI * u).sin(), 40.0 * v), &GridOptions { wrap: true, noise: 0.01, ..Default::default() }));
        let on = |deg: f64, z: f64| DVec3::new(r * ((deg * PI) / 180.0).cos(), r * ((deg * PI) / 180.0).sin(), z);
        let f = network(
            &[(1, on(0.0, 5.0)), (2, on(270.0, 5.0)), (3, on(270.0, 35.0)), (4, on(0.0, 35.0))],
            &[curve_via(1, 2, vec![on(90.0, 5.0), on(180.0, 5.0)]), curve(2, 3), curve_via(3, 4, vec![on(180.0, 35.0), on(90.0, 35.0)]), curve(4, 1)],
            &[on(135.0, 20.0)],
        );
        let (traced, cells) = cells_of(&scan, &f);
        let cell = &cells.cells[0];
        assert_eq!(status(cell), "ok");
        let map = mapped(&scan, &traced, cell).expect("maps");
        // The map runs round the cylinder with the angle one way and up it the other; which
        // way round the square depends on the corner it starts at, so each of the square's
        // eight symmetries is tried.
        let mesh = &map.mesh;
        let uv = &map.uv;
        let symmetries: [fn(f64, f64) -> (f64, f64); 8] = [
            |a, b| (a, b),
            |a, b| (1.0 - a, b),
            |a, b| (a, 1.0 - b),
            |a, b| (1.0 - a, 1.0 - b),
            |a, b| (b, a),
            |a, b| (1.0 - b, a),
            |a, b| (b, 1.0 - a),
            |a, b| (1.0 - b, 1.0 - a),
        ];
        let worst = symmetries.iter().map(|s| {
            let mut w = 0.0f64;
            for i in 0..mesh.scan_vertex.len() {
                let p = mesh.position(i);
                let mut deg = (p.y.atan2(p.x) * 180.0) / PI;
                if deg < -45.0 {
                    deg += 360.0;
                }
                let (a, b) = s(uv[i * 2], uv[i * 2 + 1]);
                w = w.max((a - deg / 270.0).abs()).max((b - (p.z - 5.0) / 30.0).abs());
            }
            w
        });
        // The boundary's vertices lie up to a grid row (1 mm of the 30) outside the curves;
        // the rest within a hundredth.
        let best = worst.fold(f64::INFINITY, f64::min);
        assert!(best < 0.04, "{best}");
    }

    #[test]
    fn maps_a_saddle_and_closes_a_scan_hole_inside_a_cell_with_a_fan() {
        let saddle = |u: f64, v: f64| DVec3::new(60.0 * u - 30.0, 60.0 * v - 30.0, ((60.0 * u - 30.0).powi(2) - (60.0 * v - 30.0).powi(2)) / 60.0);
        let scan = scan_of(&grid_scan(90, 90, &saddle, &GridOptions { hole: Some(&|u, v| (u - 0.5).hypot(v - 0.5) < 0.08), ..Default::default() }));
        let mut f = grid_network(1, 1, &saddle, 1, [0.2, 0.8, 0.2, 0.8]);
        f.cells = vec![crate::network_feature::NetworkCell { seed: saddle(0.3, 0.3), centre: None, corners: None }];
        let (traced, cells) = cells_of(&scan, &f);
        let cell = &cells.cells[0];
        let CellReading::Ok(fill) = cell else { panic!("{cell:?}") };
        assert_eq!(fill.holes.len(), 1);
        let map = mapped(&scan, &traced, cell).expect("maps");
        // The fan's middle is mapped inside, near the square's middle.
        let middle = map.mesh.scan_vertex.iter().position(|&v| v == -1).expect("a fan");
        assert!(middle > 0);
        assert!((map.uv[middle * 2] - 0.5).abs() < 0.05);
        assert!((map.uv[middle * 2 + 1] - 0.5).abs() < 0.05);
    }

    #[test]
    fn refuses_a_cell_of_two_sides_and_one_of_seven_by_name() {
        let scan = scan_of(&grid_scan(100, 100, &|u, v| DVec3::new(100.0 * u, 100.0 * v, 0.0), &GridOptions::default()));
        // Two curves from one node to another, bowed apart: a lens.
        let lens = network(
            &[(1, DVec3::new(30.0, 50.0, 0.0)), (2, DVec3::new(70.0, 50.0, 0.0))],
            &[curve_via(1, 2, vec![DVec3::new(50.0, 65.0, 0.0)]), curve_via(2, 1, vec![DVec3::new(50.0, 35.0, 0.0)])],
            &[DVec3::new(50.0, 50.0, 0.0)],
        );
        match &cells_of(&scan, &lens).1.cells[0] {
            CellReading::Refused(r) => assert_eq!(r.message, "two sides"),
            other => panic!("{other:?}"),
        }
        // A heptagon.
        let corners: Vec<(u32, DVec3)> = (0..7).map(|k| (k + 1, DVec3::new(50.0 + 30.0 * ((2.0 * PI * k as f64) / 7.0).cos(), 50.0 + 30.0 * ((2.0 * PI * k as f64) / 7.0).sin(), 0.0))).collect();
        let seven = network(&corners, &(0..7).map(|k| curve(k + 1, ((k + 1) % 7) + 1)).collect::<Vec<_>>(), &[DVec3::new(50.0, 50.0, 0.0)]);
        match &cells_of(&scan, &seven).1.cells[0] {
            CellReading::Refused(r) => assert_eq!(r.message, "7 sides — split it with a curve"),
            other => panic!("{other:?}"),
        }
    }
}
