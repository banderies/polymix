// SPDX-License-Identifier: AGPL-3.0-only
//! A patch network read off the scan, stage by stage.
//!
//! Ported from ScanRuler (`plugins/reverse/core/network.ts`, with the input types of
//! `plugins/reverse/core/networkFit.ts` that it hands the fit), © Stefan Hermann
//! (CNC Kitchen), AGPL-3.0-only.
//!
//! Its nodes are taken onto the scan and its curves traced along it
//! ([`crate::network_curve`]), the regions the curves close read into cells
//! ([`crate::network_cells`]), each four-sided cell mapped onto the square
//! ([`crate::network_map`]), and the patches fitted together (the network fit, which
//! [`read_network`] is handed as a function: [`crate::network_fit::fit_network`], as
//! [`crate::network_fit::read_network_fitted`] reads it). Pure geometry over the scan's
//! triangles: the build runs it, a test runs it on a synthetic scan.
//!
//! The viewport's preview reads the network again at every edit, and an edit changes a curve
//! or two and a cell or two: each curve's trace is kept by the points it was traced through,
//! and each cell's points and their (u, v) by the triangles and corners they were mapped
//! from, for the next read on the same scan; what the last few reads did not use is let go.
//! The same input gives the same answer, kept or not; the map was half of a 30-patch read,
//! and nearly all of it was a cell read before.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Mutex;
use std::time::Instant;

use glam::DVec3;
use serde::{Deserialize, Serialize};

use crate::network_cells::{read_cells, CellReading, CellSide, GraphCurve, GraphNode, NetworkCells};
use crate::network_curve::{fit_provisional, trace_curve, CurveFit, CurvePoints, CurveTrace};
use crate::network_feature::{EdgeRef, FaceRef, NetworkCurve, NetworkFeature, NetworkNode, NodeOn, SideMode};
use crate::network_map::{cell_mesh, map_cell};
use crate::network_split::{split_polygons, Centre, Spoke};
use crate::scan_path::ScanPaths;

// ---------------------------------------------------------------------------------------
// What the fit is handed (the input types of ScanRuler's networkFit.ts).

/// A cell to fit: its four sides in order with it on their left, the corners, its scan
/// points and their (u, v) from the map.
#[derive(Debug, Clone, PartialEq)]
pub struct PatchInput {
    pub sides: Vec<CellSide>,
    pub corners: Vec<u32>,
    /// x, y, z per point.
    pub xyz: Vec<f64>,
    /// u, v per point.
    pub uv: Vec<f64>,
    pub seed: DVec3,
}

/// A curve to fit: traced, fitted provisionally (the directions it leaves its nodes by), and
/// whether the patches either side may meet at an angle on it.
#[derive(Debug, Clone, PartialEq)]
pub struct CurveInput {
    pub id: u32,
    pub from: u32,
    pub to: u32,
    pub trace: CurveTrace,
    pub fit: CurveFit,
    pub crease: bool,
    /// A curve along a body's edge: the edge's points, fitted in place of the trace.
    pub exact: Option<CurvePoints>,
}

/// A curve beside a body's face (PN11): the patch on the curve meets the face on its edge
/// (contact) or tangent to it too, by the face's unit normals at the points of it nearest
/// those given.
pub struct BodySide<'a> {
    pub curve: u32,
    pub mode: SideMode,
    pub normals: Box<dyn Fn(&[DVec3]) -> Option<Vec<DVec3>> + 'a>,
}

impl std::fmt::Debug for BodySide<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BodySide").field("curve", &self.curve).field("mode", &self.mode).finish_non_exhaustive()
    }
}

/// What the network fit is handed.
#[derive(Debug)]
pub struct NetworkFitInput<'a> {
    pub patches: Vec<PatchInput>,
    pub curves: Vec<CurveInput>,
    pub nodes: Vec<GraphNode>,
    /// Strips' span counts, by a curve id in the strip.
    pub spans: BTreeMap<String, f64>,
    pub smoothness: f64,
    pub robust: Option<bool>,
    /// How far Auto lets a patch's shape stray from its points beyond their scatter before
    /// it gives its strips more spans, mm; `network_curve::DEFAULT_TOLERANCE` when not given.
    pub tolerance: Option<f64>,
    /// The scan's mean edge, mm: what Auto holds a strip's spans to.
    pub spacing: Option<f64>,
    pub body_sides: Vec<BodySide<'a>>,
}

// ---------------------------------------------------------------------------------------
// The reading.

/// What a seed's cell came to: a patch (or, split from its centre, a patch a side:
/// `patches`; `patch` is the first) or why not.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CellOutcome {
    Filled {
        seed: DVec3,
        sides: usize,
        patch: usize,
        patches: Vec<usize>,
    },
    Refused {
        seed: DVec3,
        refused: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        at: Option<DVec3>,
    },
}

impl CellOutcome {
    pub fn seed(&self) -> DVec3 {
        match self {
            CellOutcome::Filled { seed, .. } | CellOutcome::Refused { seed, .. } => *seed,
        }
    }
}

/// The cells split from their centres: the spokes and the centres.
#[derive(Debug, Clone, PartialEq)]
pub struct SplitDrawing {
    pub spokes: Vec<Spoke>,
    pub centres: Vec<Centre>,
}

/// How long each stage of a read took, ms.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ReadTimes {
    pub trace: f64,
    pub cells: f64,
    pub map: f64,
    pub fit: f64,
}

/// A network read off the scan: its traced curves (with the spokes and the sides' middles of
/// a split cell), its cells, and the patches fitted to them (`F`, what the fit handed to
/// [`read_network`] gives), or `None` where no cell could be filled.
#[derive(Debug, Clone)]
pub struct NetworkReading<F> {
    pub traced: TracedNetwork,
    pub cells: NetworkCells,
    pub outcomes: Vec<CellOutcome>,
    pub fit: Option<F>,
    /// The cells split from their centres: the spokes and the centres.
    pub split: Option<SplitDrawing>,
    pub ms: ReadTimes,
}

/// What a network reads off the bodies it rests on (PN11): the build's bodies as they stand
/// before it; a body's edge as its points (x, y, z each), from its first end; a face's
/// normals at the points of it nearest those given. `None` where the body, the edge or the
/// face is not found.
pub trait NetworkBodies {
    fn edge(&self, body: u32, edge: &EdgeRef) -> Option<Vec<f32>>;
    fn normals(&self, body: u32, face: &FaceRef, points: &[DVec3]) -> Option<Vec<DVec3>>;
}

/// The nodes on the scan and the curves traced between them; a curve whose node is gone is
/// left out, by id.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TracedNetwork {
    pub nodes: Vec<GraphNode>,
    pub curves: Vec<GraphCurve>,
    /// Curves left out: their node is gone, or their trace is empty.
    pub lost: Vec<u32>,
    /// Why each curve on a body was left out, by id.
    pub lost_why: Option<BTreeMap<u32, String>>,
}

// ---------------------------------------------------------------------------------------
// What the reads keep.

/// How many reads back what was kept is still kept: a build reads every network of the
/// history, the preview the one being edited.
const KEPT_READS: i64 = 4;

/// The points a cell is fitted to and their (u, v) from its map, or why it has none.
pub type CellData = Result<(Vec<f64>, Vec<f64>), String>;

/// What the reads of networks on one scan keep for the next (see the module documentation).
#[derive(Debug, Default)]
pub struct ReadCache {
    read: i64,
    traces: HashMap<Vec<u64>, (CurveTrace, CurveFit, i64)>,
    cells: HashMap<String, (CellData, i64)>,
}

impl ReadCache {
    /// A new read begins: what the last KEPT_READS did not use goes.
    pub fn begin(&mut self) {
        self.read += 1;
        let read = self.read;
        self.traces.retain(|_, v| v.2 > read - KEPT_READS);
        self.cells.retain(|_, v| v.1 > read - KEPT_READS);
    }

    /// The trace through `points`, and its provisional fit; the points of a kept one copied,
    /// the caller setting its ends.
    pub fn trace(&mut self, scan: &ScanPaths, points: &[DVec3]) -> Option<(CurveTrace, CurveFit)> {
        let key: Vec<u64> = points.iter().flat_map(|p| [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()]).collect();
        let read = self.read;
        if let Some(had) = self.traces.get_mut(&key) {
            had.2 = read;
            return Some((had.0.clone(), had.1.clone()));
        }
        let (trace, fit) = trace_ends(scan, points)?;
        self.traces.insert(key, (trace.clone(), fit.clone(), read));
        Some((trace, fit))
    }

    /// A cell's points to fit and their (u, v) from its map, or why it has none; made by
    /// `make` where no read made them for the same triangles, holes and corners.
    pub fn cell(&mut self, triangles: &[u32], holes: &[Vec<u32>], corners: &[DVec3], make: impl FnOnce() -> CellData) -> CellData {
        let key = cell_key(triangles, holes, corners);
        let read = self.read;
        let had = self.cells.entry(key).or_insert_with(|| (make(), 0));
        had.1 = read;
        had.0.clone()
    }
}

/// A number in base 36, as JavaScript's `toString(36)` writes a whole number.
fn base36(mut x: u32) -> String {
    const DIGITS: &[u8; 36] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    if x == 0 {
        return "0".to_string();
    }
    let mut out = Vec::new();
    while x > 0 {
        out.push(DIGITS[(x % 36) as usize]);
        x /= 36;
    }
    out.reverse();
    String::from_utf8(out).unwrap_or_default()
}

/// A cell's input as a key: two 32-bit hashes of its triangles, its holes and its corners'
/// coordinates, and their counts.
fn cell_key(triangles: &[u32], holes: &[Vec<u32>], corners: &[DVec3]) -> String {
    let mut a: u32 = 0x811c_9dc5;
    let mut b: u32 = 0x9e37_79b9;
    let mut take = |x: u32| {
        a = (a ^ x).wrapping_mul(0x0100_0193);
        b = (b ^ x).wrapping_mul(0x5bd1_e995) ^ (b >> 15);
    };
    for &t in triangles {
        take(t);
    }
    for h in holes {
        take(u32::MAX);
        for &v in h {
            take(v);
        }
    }
    for c in corners {
        for x in [c.x, c.y, c.z] {
            let bits = x.to_bits();
            take(bits as u32);
            take((bits >> 32) as u32);
        }
    }
    format!("{}:{}:{}:{}:{}", triangles.len(), holes.len(), corners.len(), base36(a), base36(b))
}

/// The cache of reads on this scan.
struct ScanReads(Mutex<ReadCache>);

// ---------------------------------------------------------------------------------------
// Reading.

/// The network read off the scan, every stage (see the module documentation). `fit` is the
/// network fit: it is handed the patches, the curves, the nodes and the fit's settings, and
/// is called only where at least one cell is filled.
pub fn read_network<F>(scan: &ScanPaths, f: &NetworkFeature, bodies: Option<&dyn NetworkBodies>, fit: impl FnOnce(NetworkFitInput<'_>) -> F) -> NetworkReading<F> {
    let t0 = Instant::now();
    let reads = scan.memo(|_| ScanReads(Mutex::new(ReadCache::default())));
    let mut cache = reads.0.lock().unwrap_or_else(|e| e.into_inner());
    cache.begin();
    let traced = trace_network(scan, &f.nodes, &f.curves, bodies, Some(&mut cache));
    let t1 = Instant::now();
    // A patch placed by its corners has them as its corners, not the rule's.
    let seeds: Vec<DVec3> = f.cells.iter().map(|c| c.seed).collect();
    let given: Vec<Option<HashSet<u32>>> = f.cells.iter().map(|c| c.corners.as_ref().filter(|k| k.len() >= 3).map(|k| k.iter().copied().collect())).collect();
    let first = read_cells(scan, &traced.nodes, &traced.curves, &seeds, &given);
    // A cell of three, five or six sides split from its centre, and the cells read again
    // with the spokes and the sides' middles in (PN6).
    let centres: Vec<Option<DVec3>> = f.cells.iter().map(|c| c.centre).collect();
    let split = split_polygons(scan, &traced, &first.cells, &centres);
    let (graph, cells, owner) = match &split {
        Some(split) => {
            let again = read_cells(scan, &split.traced.nodes, &split.traced.curves, &split.seeds, &split.corners);
            let cells = NetworkCells { faces: first.faces.clone(), cells: again.cells, unfilled: again.unfilled };
            (split.traced.clone(), cells, split.owner.clone())
        }
        None => (traced.clone(), first.clone(), (0..first.cells.len()).collect()),
    };
    let t2 = Instant::now();
    let at: HashMap<u32, DVec3> = graph.nodes.iter().map(|n| (n.id, n.at)).collect();
    let mut patches: Vec<PatchInput> = Vec::new();
    let read: Vec<Result<usize, (String, Option<DVec3>)>> = cells
        .cells
        .iter()
        .map(|c| {
            let c = match c {
                CellReading::Ok(c) => c,
                CellReading::Refused(r) => return Err((r.message.clone(), r.at)),
            };
            if c.sides.len() != 4 {
                return Err((format!("{} sides — split from its centre, it did not come out four-sided", c.sides.len()), None));
            }
            let corners: Vec<DVec3> = c.corners.iter().map(|id| at.get(id).copied().unwrap_or(DVec3::NAN)).collect();
            let data = cache.cell(&c.triangles, &c.holes, &corners, || {
                let mesh = cell_mesh(scan, &c.triangles, &c.holes, &corners)?;
                let map = map_cell(mesh, None)?;
                let mesh = &map.mesh;
                // The points to fit: the cell's scan vertices off its boundary; those on it
                // lie on the curves' side of the cut, on the fixed rows.
                let mut on_loop = vec![false; mesh.scan_vertex.len()];
                for &v in &mesh.boundary {
                    on_loop[v] = true;
                }
                let mut xyz = Vec::new();
                let mut uv = Vec::new();
                for v in 0..mesh.scan_vertex.len() {
                    if mesh.scan_vertex[v] < 0 || on_loop[v] {
                        continue;
                    }
                    xyz.extend_from_slice(&mesh.positions[v * 3..v * 3 + 3]);
                    uv.extend_from_slice(&map.uv[v * 2..v * 2 + 2]);
                }
                Ok((xyz, uv))
            });
            let (xyz, uv) = data.map_err(|why| (why, None))?;
            patches.push(PatchInput { sides: c.sides.clone(), corners: c.corners.clone(), xyz, uv, seed: c.seed });
            Ok(patches.len() - 1)
        })
        .collect();
    drop(cache);
    // Each of the feature's cells from its own readings: one, or one a quad of a split
    // cell; refused if any is.
    let outcomes: Vec<CellOutcome> = f
        .cells
        .iter()
        .enumerate()
        .map(|(i, cell)| {
            let sides = first.cells.get(i).and_then(|c| c.ok()).map(|c| c.sides.len()).unwrap_or(0);
            if let Some(why) = split.as_ref().and_then(|s| s.refused.get(&i)) {
                return CellOutcome::Refused { seed: cell.seed, refused: format!("{sides} sides, and it could not be split from its centre: {why}"), at: None };
            }
            let mine: Vec<&Result<usize, (String, Option<DVec3>)>> = read.iter().enumerate().filter(|(k, _)| owner.get(*k) == Some(&i)).map(|(_, r)| r).collect();
            if let Some(Err((why, at))) = mine.iter().find(|r| r.is_err()) {
                let refused = if sides > 0 && sides != 4 { format!("split from its centre: {why}") } else { why.clone() };
                return CellOutcome::Refused { seed: cell.seed, refused, at: *at };
            }
            let list: Vec<usize> = mine.iter().filter_map(|r| r.as_ref().ok().copied()).collect();
            match list.first() {
                None => CellOutcome::Refused { seed: cell.seed, refused: "nothing came of it".to_string(), at: None },
                Some(&patch) => CellOutcome::Filled { seed: cell.seed, sides, patch, patches: list },
            }
        })
        .collect();
    let t3 = Instant::now();
    // A piece of a curve split at a side's middle is a crease where the curve is; a spoke is
    // not.
    let crease: HashMap<u32, bool> = f.curves.iter().map(|c| (c.id, c.crease == Some(true))).collect();
    let crease_of = |id: u32| -> bool {
        if let Some(&c) = crease.get(&id) {
            return c;
        }
        split.as_ref().and_then(|s| s.origin_of.get(&id)).and_then(|o| crease.get(o)).copied().unwrap_or(false)
    };
    let fitted = if patches.is_empty() {
        None
    } else {
        // The patches beside a body's face: held tangent to it, by the face's normals where
        // it is asked; on its edge, either way.
        let body_sides: Vec<BodySide<'_>> = match bodies {
            Some(bodies) => f
                .curves
                .iter()
                .filter_map(|c| {
                    let side = c.side.as_ref()?;
                    Some(BodySide { curve: c.id, mode: side.mode, normals: Box::new(move |points: &[DVec3]| bodies.normals(side.body, &side.face, points)) })
                })
                .collect(),
            None => Vec::new(),
        };
        let input = NetworkFitInput {
            patches,
            curves: graph
                .curves
                .iter()
                .map(|c| CurveInput { id: c.id, from: c.from, to: c.to, trace: c.trace.clone(), fit: c.fit.clone(), crease: crease_of(c.id), exact: c.exact.clone() })
                .collect(),
            nodes: graph.nodes.clone(),
            spans: f.spans.clone(),
            smoothness: f.fit.smoothness,
            robust: f.fit.robust,
            tolerance: f.fit.tolerance,
            spacing: Some(scan.edge()),
            body_sides,
        };
        Some(fit(input))
    };
    let t4 = Instant::now();
    let ms = |a: Instant, b: Instant| (b - a).as_secs_f64() * 1e3;
    NetworkReading {
        traced: graph,
        cells,
        outcomes,
        fit: fitted,
        split: split.map(|s| SplitDrawing { spokes: s.spokes, centres: s.centres }),
        ms: ReadTimes { trace: ms(t0, t1), cells: ms(t1, t2), map: ms(t2, t3), fit: ms(t3, t4) },
    }
}

/// How far apart the grips a curve along a body's edge is traced on the scan by lie, mm:
/// about as far as a grip a person clicks.
const EDGE_GRIP_MM: f64 = 3.0;

/// A body edge's points no further apart than this, mm: a straight edge comes as its two
/// ends, which a curve of several spans cannot be fitted to alone.
const EDGE_STEP_MM: f64 = 0.5;

/// A polyline with points put in along each piece so none is longer than `step`.
fn densified(points: &[f32], step: f64) -> Vec<f32> {
    let mut out: Vec<f64> = Vec::new();
    let n = points.len() / 3;
    let p = |i: usize, k: usize| points[i * 3 + k] as f64;
    for i in 0..n {
        if i > 0 {
            let d = ((p(i, 0) - p(i - 1, 0)).powi(2) + (p(i, 1) - p(i - 1, 1)).powi(2) + (p(i, 2) - p(i - 1, 2)).powi(2)).sqrt();
            let k = (d / step).ceil();
            let k = if k.is_finite() { k as usize } else { 1 };
            for j in 1..k {
                for c in 0..3 {
                    out.push(p(i - 1, c) + ((p(i, c) - p(i - 1, c)) * j as f64) / k as f64);
                }
            }
        }
        out.extend_from_slice(&[p(i, 0), p(i, 1), p(i, 2)]);
    }
    out.into_iter().map(|x| x as f32).collect()
}

/// The point a share `t` of the way along a polyline, by arc length.
fn along_polyline(p: &CurvePoints, t: f64) -> DVec3 {
    let n = p.at.len();
    if n == 0 {
        return DVec3::NAN;
    }
    if n == 1 {
        return p.as_points().point(0);
    }
    let want = t.clamp(0.0, 1.0) * p.length;
    let mut i = 1;
    while i < n - 1 && p.at[i] < want {
        i += 1;
    }
    let span = p.at[i] - p.at[i - 1];
    let span = if span != 0.0 && !span.is_nan() { span } else { 1.0 };
    let f = ((want - p.at[i - 1]) / span).clamp(0.0, 1.0);
    let a = p.as_points().point(i - 1);
    let b = p.as_points().point(i);
    DVec3::new(a.x + f * (b.x - a.x), a.y + f * (b.y - a.y), a.z + f * (b.z - a.z))
}

/// The index of a polyline's point nearest q.
fn nearest_index(p: &[f32], q: DVec3) -> usize {
    let mut best = 0;
    let mut best_d = f64::INFINITY;
    for i in 0..p.len() / 3 {
        let d = DVec3::new(p[i * 3] as f64 - q.x, p[i * 3 + 1] as f64 - q.y, p[i * 3 + 2] as f64 - q.z).length();
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    best
}

/// The trace through `points` with its ends set on them, and its provisional fit; `None`
/// where it is too short to be a curve.
fn trace_ends(scan: &ScanPaths, points: &[DVec3]) -> Option<(CurveTrace, CurveFit)> {
    let mut trace = trace_curve(scan, points);
    if trace.points.len() < 6 {
        return None;
    }
    let last = trace.points.len() / 3 - 1;
    trace.set_point(0, *points.first()?);
    trace.set_point(last, *points.last()?);
    let fit = fit_provisional(trace.as_points(), scan.edge());
    Some((trace, fit))
}

/// Each node taken onto the scan (or onto the body edge or corner it is on), each curve
/// traced from its node through its grips to its other node, and fitted at its own Auto for
/// the directions it leaves its nodes by. A curve along a body's edge takes the edge's
/// points between its two nodes, and is traced on the scan through grips along them.
pub fn trace_network(scan: &ScanPaths, nodes: &[NetworkNode], curves: &[NetworkCurve], bodies: Option<&dyn NetworkBodies>, mut cache: Option<&mut ReadCache>) -> TracedNetwork {
    // The nodes placed, by id, in the order first given (a later node of the same id taking
    // its place, as a JavaScript Map keeps it).
    let mut placed: Vec<GraphNode> = Vec::new();
    let mut index: HashMap<u32, usize> = HashMap::new();
    for n in nodes {
        let mut at: Option<DVec3> = None;
        if let (Some(on), Some(bodies)) = (&n.on, bodies) {
            if let Some(edge) = bodies.edge(on.body(), on.edge()) {
                if edge.len() >= 6 {
                    let first = DVec3::new(edge[0] as f64, edge[1] as f64, edge[2] as f64);
                    let l = edge.len();
                    let last = DVec3::new(edge[l - 3] as f64, edge[l - 2] as f64, edge[l - 1] as f64);
                    at = Some(match on {
                        NodeOn::Edge { t, .. } => along_polyline(&CurvePoints::from_polyline(edge), *t),
                        NodeOn::Vertex { end, .. } => {
                            if *end == 0 {
                                first
                            } else {
                                last
                            }
                        }
                    });
                }
            }
        }
        let at = at.unwrap_or_else(|| scan.closest_inside(n.at).map(|c| c.point).unwrap_or(n.at));
        let node = GraphNode { id: n.id, at, corner: n.corner };
        match index.get(&n.id) {
            Some(&i) => placed[i] = node,
            None => {
                index.insert(n.id, placed.len());
                placed.push(node);
            }
        }
    }
    // A node a curve along a body's edge ends at is on that edge: taken from the scan, where
    // it was clicked, to the edge's nearest point; the scan's noise off the edge would bend
    // the curve off it at its end.
    let mut edges: HashMap<u32, Option<Vec<f32>>> = HashMap::new();
    let mut edge_of = |c: &NetworkCurve| -> Option<Vec<f32>> {
        let along = c.along.as_ref()?;
        edges
            .entry(c.id)
            .or_insert_with(|| bodies.and_then(|b| b.edge(along.body, &along.edge)).map(|found| densified(&found, EDGE_STEP_MM)))
            .clone()
    };
    for c in curves {
        let Some(edge) = edge_of(c) else { continue };
        for id in [c.from, c.to] {
            let Some(&i) = index.get(&id) else { continue };
            let own = nodes.iter().find(|x| x.id == id);
            if own.is_some_and(|o| o.on.is_some()) {
                continue;
            }
            if edge.len() < 3 {
                continue;
            }
            let k = nearest_index(&edge, placed[i].at);
            placed[i].at = DVec3::new(edge[k * 3] as f64, edge[k * 3 + 1] as f64, edge[k * 3 + 2] as f64);
        }
    }
    let mut traced: Vec<GraphCurve> = Vec::new();
    let mut lost: Vec<u32> = Vec::new();
    let mut lost_why: BTreeMap<u32, String> = BTreeMap::new();
    for c in curves {
        let (Some(&ia), Some(&ib)) = (index.get(&c.from), index.get(&c.to)) else {
            lost.push(c.id);
            continue;
        };
        let a = placed[ia].at;
        let b = placed[ib].at;
        let mut exact: Option<CurvePoints> = None;
        let mut grips: Vec<DVec3> = c.via.clone();
        if c.along.is_some() {
            let edge = match edge_of(c) {
                Some(edge) if edge.len() >= 6 => edge,
                _ => {
                    lost.push(c.id);
                    lost_why.insert(c.id, "the body edge it runs along is not found on the body as built".to_string());
                    continue;
                }
            };
            // The edge between the two nodes, run from the first.
            let i = nearest_index(&edge, a);
            let j = nearest_index(&edge, b);
            let mut pts: Vec<f64> = Vec::new();
            let mut push = |k: usize| pts.extend_from_slice(&[edge[k * 3] as f64, edge[k * 3 + 1] as f64, edge[k * 3 + 2] as f64]);
            if i <= j {
                (i..=j).for_each(&mut push);
            } else {
                (j..=i).rev().for_each(&mut push);
            }
            if i == j {
                pts = vec![a.x, a.y, a.z, b.x, b.y, b.z];
            } else {
                pts[..3].copy_from_slice(&[a.x, a.y, a.z]);
                let l = pts.len();
                pts[l - 3..].copy_from_slice(&[b.x, b.y, b.z]);
            }
            let e = CurvePoints::from_polyline(pts.into_iter().map(|x| x as f32).collect());
            let count = ((e.length / EDGE_GRIP_MM).floor() - 1.0).max(0.0);
            let count = if count.is_finite() { count as usize } else { 0 };
            grips = (0..count).map(|k| along_polyline(&e, (k + 1) as f64 / (count + 1) as f64)).collect();
            exact = Some(e);
        }
        // The trace ends where the scan is nearest each node, which is where the node was
        // put: the curves meeting there meet exactly.
        let mut through = Vec::with_capacity(grips.len() + 2);
        through.push(a);
        through.extend_from_slice(&grips);
        through.push(b);
        let got = match cache.as_deref_mut() {
            Some(cache) => cache.trace(scan, &through),
            None => trace_ends(scan, &through),
        };
        let Some((trace, fit)) = got else {
            lost.push(c.id);
            continue;
        };
        let fit = match &exact {
            Some(e) => fit_provisional(e.as_points(), scan.edge()),
            None => fit,
        };
        traced.push(GraphCurve { id: c.id, from: c.from, to: c.to, trace, fit, exact });
    }
    TracedNetwork { nodes: placed, curves: traced, lost, lost_why: if lost_why.is_empty() { None } else { Some(lost_why) } }
}
