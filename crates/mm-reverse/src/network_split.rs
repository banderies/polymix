// SPDX-License-Identifier: AGPL-3.0-only
//! A cell of three, five or six sides split into four-sided ones from its centre (PN6).
//!
//! Ported from ScanRuler (`plugins/reverse/core/networkSplit.ts`), © Stefan Hermann
//! (CNC Kitchen), AGPL-3.0-only.
//!
//! The cell is mapped onto a regular polygon of as many corners ([`crate::network_map`]);
//! its centre is the scan point at the polygon's middle, or where the operator dragged it.
//! Each side is split at its middle by arc length with a node (a T-node for the cell on its
//! other side, whose side then runs through it straight on) and a spoke runs from the centre
//! to each: n four-sided cells, meeting at a corner of n patches at the centre, where the fit
//! reports the angle instead of holding it.
//!
//! The spokes and the nodes are made for the build and are not the feature's: they are made
//! again from the cell's sides every build, so they follow the sides when a curve moves, and
//! only the centre a hand dragged is kept, on the cell. Their ids are counted from
//! DERIVED_IDS on, in the order the cells and their sides come, so the same network gives the
//! same ids build after build: what a strip's span count is keyed by.

use std::collections::{BTreeMap, HashMap, HashSet};

use glam::DVec3;

use crate::network::TracedNetwork;
use crate::network_cells::{CellReading, CellSide, GraphCurve, GraphNode};
use crate::network_curve::{fit_provisional, trace_curve, CurveTrace, TraceRun};
use crate::network_map::{cell_mesh, map_cell, polygon_corners, CellMap};
use crate::scan_path::ScanPaths;

/// Where the ids of the nodes and curves made for a split start.
pub const DERIVED_IDS: u32 = 1_000_000;
/// How near the middle of a side a node already on it may be, as a share of the side's
/// length, for it to be the middle: a side two split cells share is split once.
const REUSE_SHARE: f64 = 0.02;

/// A spoke made for a split cell, for drawing.
#[derive(Debug, Clone, PartialEq)]
pub struct Spoke {
    /// The feature's cell it splits.
    pub cell: usize,
    pub curve: u32,
    pub points: Vec<f32>,
}

/// A split cell's centre, for drawing.
#[derive(Debug, Clone, PartialEq)]
pub struct Centre {
    pub cell: usize,
    pub at: DVec3,
}

/// The network with its cells of three, five or six sides split.
#[derive(Debug, Clone, PartialEq)]
pub struct PolygonSplit {
    pub traced: TracedNetwork,
    /// Every seed the cells are read by again: a cell's own where it was not split, one in
    /// each of its quads where it was.
    pub seeds: Vec<DVec3>,
    /// By seed: which of the feature's cells it is for.
    pub owner: Vec<usize>,
    /// By seed: a quad's four corners, which the corner rule is not asked for (off its
    /// middle, a centre opens past the rule's 150° on a quad).
    pub corners: Vec<Option<HashSet<u32>>>,
    /// The spokes and centres made, by the feature's cell, for drawing.
    pub spokes: Vec<Spoke>,
    pub centres: Vec<Centre>,
    /// Cells that could not be split, and why.
    pub refused: BTreeMap<usize, String>,
    /// A piece of a curve split at a side's middle: the curve it is of.
    pub origin_of: HashMap<u32, u32>,
}

/// The point of a mapped cell at (u, v) of its polygon, or `None` off it.
pub fn point_at_uv(map: &CellMap, u: f64, v: f64) -> Option<DVec3> {
    let mesh = &map.mesh;
    let t = &mesh.triangles;
    let uv = &map.uv;
    for f in 0..t.len() / 3 {
        let (a, b, c) = (t[f * 3] as usize, t[f * 3 + 1] as usize, t[f * 3 + 2] as usize);
        let (x0, y0) = (uv[a * 2], uv[a * 2 + 1]);
        let (x1, y1) = (uv[b * 2] - x0, uv[b * 2 + 1] - y0);
        let (x2, y2) = (uv[c * 2] - x0, uv[c * 2 + 1] - y0);
        let det = x1 * y2 - x2 * y1;
        if !(det.abs() > 1e-18) {
            continue;
        }
        let s = ((u - x0) * y2 - x2 * (v - y0)) / det;
        let r = (x1 * (v - y0) - (u - x0) * y1) / det;
        if s < -1e-9 || r < -1e-9 || s + r > 1.0 + 1e-9 {
            continue;
        }
        let w = 1.0 - s - r;
        return Some(w * mesh.position(a) + s * mesh.position(b) + r * mesh.position(c));
    }
    None
}

/// A trace's points from `from` to `to`, its arc length counted again from the first.
fn slice_trace(t: &CurveTrace, from: usize, to: usize) -> CurveTrace {
    let mut at: Vec<f64> = t.at[from..=to].to_vec();
    let base = at[0];
    for x in at.iter_mut() {
        *x -= base;
    }
    let length = at.last().copied().unwrap_or(0.0);
    let mut grips = vec![0];
    grips.extend(t.grips.iter().filter(|&&g| g > from && g < to).map(|g| g - from));
    grips.push(to - from);
    let runs = t
        .runs
        .iter()
        .filter(|r| r.to > from && r.from < to)
        .map(|r| TraceRun { from: r.from.max(from) - from, to: r.to.min(to) - from, ..r.clone() })
        .collect();
    CurveTrace { points: t.points[from * 3..(to + 1) * 3].to_vec(), on: t.on[from * 2..(to + 1) * 2].to_vec(), at, length, grips, runs }
}

/// The curves as they are now cut, in the order they were first had (as a JavaScript `Map`
/// keeps them).
struct Curves {
    list: Vec<GraphCurve>,
    index: HashMap<u32, usize>,
}

impl Curves {
    fn get(&self, id: u32) -> Option<&GraphCurve> {
        self.index.get(&id).map(|&i| &self.list[i])
    }

    fn set(&mut self, c: GraphCurve) {
        match self.index.get(&c.id) {
            Some(&i) => self.list[i] = c,
            None => {
                self.index.insert(c.id, self.list.len());
                self.list.push(c);
            }
        }
    }
}

/// The network's cells of three, five or six sides split (see the module documentation), or
/// `None` when there is none.
pub fn split_polygons(scan: &ScanPaths, traced: &TracedNetwork, cells: &[CellReading], centres_given: &[Option<DVec3>]) -> Option<PolygonSplit> {
    if !cells.iter().any(|c| c.ok().is_some_and(|c| c.sides.len() != 4)) {
        return None;
    }
    let mut nodes: Vec<GraphNode> = traced.nodes.clone();
    let mut curves = Curves { list: Vec::new(), index: HashMap::new() };
    for c in &traced.curves {
        curves.set(c.clone());
    }
    // Each curve of the network as it is now cut: its pieces in order.
    let mut pieces: Vec<(u32, Vec<u32>)> = traced.curves.iter().map(|c| (c.id, vec![c.id])).collect();
    let mut piece_of: HashMap<u32, usize> = HashMap::new();
    for (i, (id, _)) in pieces.iter().enumerate() {
        piece_of.entry(*id).or_insert(i);
    }
    let node_at = |nodes: &[GraphNode], id: u32| nodes.iter().find(|n| n.id == id).map(|n| n.at);
    let mut next = DERIVED_IDS;
    let mut seeds: Vec<DVec3> = Vec::new();
    let mut owner: Vec<usize> = Vec::new();
    let mut quad_corners: Vec<Option<HashSet<u32>>> = Vec::new();
    let mut spokes: Vec<Spoke> = Vec::new();
    let mut centres: Vec<Centre> = Vec::new();
    let mut refused: BTreeMap<usize, String> = BTreeMap::new();

    // A side's pieces in its order, each forwards or backwards.
    let side_pieces = |pieces: &[(u32, Vec<u32>)], piece_of: &HashMap<u32, usize>, side: &CellSide| -> Vec<(u32, bool)> {
        side.edges
            .iter()
            .flat_map(|h| {
                let list = piece_of.get(&h.curve).map(|&i| pieces[i].1.clone()).unwrap_or_else(|| vec![h.curve]);
                let ordered: Vec<u32> = if h.forward { list } else { list.into_iter().rev().collect() };
                ordered.into_iter().map(move |curve| (curve, h.forward))
            })
            .collect()
    };

    for (index, cell) in cells.iter().enumerate() {
        let fill = match cell {
            CellReading::Ok(fill) if fill.sides.len() != 4 => fill,
            _ => {
                seeds.push(cell.seed());
                owner.push(index);
                // A four-sided cell keeps the corners it was read with: a side's middle put in
                // by a neighbour's split is a T-node on it, not a corner, and a corner given
                // turns as little as it likes.
                quad_corners.push(cell.ok().map(|c| c.corners.iter().copied().collect()));
                continue;
            }
        };
        let n = fill.sides.len();
        let corner_points: Vec<DVec3> = fill.corners.iter().map(|&id| node_at(&nodes, id).unwrap_or(DVec3::NAN)).collect();
        let map = cell_mesh(scan, &fill.triangles, &fill.holes, &corner_points).and_then(|mesh| map_cell(mesh, None));
        let map = match map {
            Ok(map) => map,
            Err(why) => {
                refused.insert(index, why);
                seeds.push(fill.seed);
                owner.push(index);
                quad_corners.push(None);
                continue;
            }
        };
        // The centre: dragged, or the polygon's middle.
        let middle = centres_given.get(index).copied().flatten().or_else(|| point_at_uv(&map, 0.5, 0.5));
        let Some(on_scan) = middle.and_then(|m| scan.closest_inside(m)) else {
            refused.insert(index, "its centre is off the scan".to_string());
            seeds.push(fill.seed);
            owner.push(index);
            quad_corners.push(None);
            continue;
        };
        let centre = next;
        next += 1;
        nodes.push(GraphNode { id: centre, at: on_scan.point, corner: None });
        centres.push(Centre { cell: index, at: on_scan.point });
        // The node at each side's middle by arc length: one already there within REUSE_SHARE
        // of it, or a new one splitting the piece it falls on.
        let mut mids: Vec<Option<u32>> = Vec::with_capacity(n);
        for side in &fill.sides {
            let parts = side_pieces(&pieces, &piece_of, side);
            let lengths: Vec<f64> = parts.iter().map(|p| curves.get(p.0).map(|c| c.trace.length).unwrap_or(0.0)).collect();
            let total: f64 = lengths.iter().sum();
            let mut found: Option<u32> = None;
            // A node between two pieces near enough the middle is it.
            let mut acc = 0.0;
            for k in 0..parts.len().saturating_sub(1) {
                acc += lengths[k];
                if (acc - total / 2.0).abs() <= REUSE_SHARE * total {
                    if let Some(c) = curves.get(parts[k].0) {
                        found = Some(if parts[k].1 { c.to } else { c.from });
                    }
                    break;
                }
            }
            if found.is_none() {
                acc = 0.0;
                for k in 0..parts.len() {
                    if acc + lengths[k] < total / 2.0 && k + 1 < parts.len() {
                        acc += lengths[k];
                        continue;
                    }
                    let (piece, forward) = parts[k];
                    let Some(c) = curves.get(piece).cloned() else { break };
                    let t = &c.trace;
                    if t.at.len() < 2 {
                        found = Some(c.to);
                        break;
                    }
                    let along = if forward { total / 2.0 - acc } else { t.length - (total / 2.0 - acc) };
                    let mut i = 1;
                    while i + 2 < t.at.len() && t.at[i] < along {
                        i += 1;
                    }
                    // The trace's own point there: it lies on the scan, on a triangle's edge,
                    // and both halves keep it; the edges the curve crossed stay crossed.
                    let node = next;
                    let second = next + 1;
                    next += 2;
                    nodes.push(GraphNode { id: node, at: t.point(i), corner: None });
                    let a = slice_trace(t, 0, i);
                    let b = slice_trace(t, i, t.at.len() - 1);
                    let fa = fit_provisional(a.as_points(), scan.edge());
                    let fb = fit_provisional(b.as_points(), scan.edge());
                    curves.set(GraphCurve { id: c.id, from: c.from, to: node, trace: a, fit: fa, exact: c.exact.clone() });
                    curves.set(GraphCurve { id: second, from: node, to: c.to, trace: b, fit: fb, exact: None });
                    for (_, list) in pieces.iter_mut() {
                        if let Some(at2) = list.iter().position(|&x| x == c.id) {
                            list.insert(at2 + 1, second);
                            break;
                        }
                    }
                    found = Some(node);
                    break;
                }
                if found.is_none() {
                    found = parts.last().and_then(|p| curves.get(p.0)).map(|c| c.to);
                }
            }
            mids.push(found);
        }
        for &mid in mids.iter().take(n) {
            let id = next;
            next += 1;
            // A side with no middle (no curves along it) has no spoke.
            let Some(end) = mid.and_then(|m| node_at(&nodes, m)) else { continue };
            let mut trace = trace_curve(scan, &[on_scan.point, end]);
            if trace.points.len() < 6 {
                continue;
            }
            let last = trace.points.len() / 3 - 1;
            trace.set_point(0, on_scan.point);
            trace.set_point(last, end);
            let fit = fit_provisional(trace.as_points(), scan.edge());
            spokes.push(Spoke { cell: index, curve: id, points: trace.points.clone() });
            curves.set(GraphCurve { id, from: centre, to: mid.unwrap_or(centre), trace, fit, exact: None });
        }
        // A seed in each quad: the middle of its four corners on the polygon: the centre, the
        // side's middle before the corner, the corner and the side's middle after it.
        let poly = polygon_corners(n);
        let mid_of = |k: isize| -> [f64; 2] {
            let n = n as isize;
            let a = poly[(((k % n) + n) % n) as usize];
            let b = poly[((((k + 1) % n) + n) % n) as usize];
            [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0]
        };
        for k in 0..n {
            let quad = [[0.5, 0.5], mid_of(k as isize - 1), poly[k], mid_of(k as isize)];
            let u = quad.iter().map(|q| q[0]).sum::<f64>() / 4.0;
            let v = quad.iter().map(|q| q[1]).sum::<f64>() / 4.0;
            let Some(seed) = point_at_uv(&map, u, v) else { continue };
            seeds.push(scan.closest_inside(seed).map(|c| c.point).unwrap_or(seed));
            owner.push(index);
            let mut set: HashSet<u32> = HashSet::new();
            set.insert(centre);
            if let Some(m) = mids[(k + n - 1) % n] {
                set.insert(m);
            }
            set.insert(fill.corners[k]);
            if let Some(m) = mids[k] {
                set.insert(m);
            }
            quad_corners.push(Some(set));
        }
    }
    let mut origin_of: HashMap<u32, u32> = HashMap::new();
    for (original, list) in &pieces {
        for &id in list {
            if id != *original {
                origin_of.insert(id, *original);
            }
        }
    }
    Some(PolygonSplit {
        traced: TracedNetwork { nodes, curves: curves.list, lost: traced.lost.clone(), lost_why: None },
        seeds,
        owner,
        corners: quad_corners,
        spokes,
        centres,
        refused,
        origin_of,
    })
}

/// Ported from ScanRuler's `plugins/reverse/tests/reNetworkSplit.test.ts`: cells of three,
/// five and six sides split from their centres (PN6). A triangle and a pentagon on a noisy
/// sphere come out as three and five four-sided patches at the noise, tangent across the
/// spokes within half a degree, the centre reported as a corner of three or five patches; a
/// centre dragged is taken; and a triangle beside a four-sided cell splits the side they
/// share with a T-node, the T-junction held tangent and its gap and angle read. Each read
/// runs the real network fit ([`crate::network_fit::read_network_fitted`]).
#[cfg(test)]
mod tests {
    use std::f64::consts::PI;
    use std::sync::OnceLock;

    use super::*;
    use crate::network::{CellOutcome, NetworkReading};
    use crate::network_feature::NetworkFeature;
    use crate::network_fit::{read_network_fitted, NetworkFitResult};
    use crate::network_meshes::{curve, network, sphere_scan};

    const R: f64 = 40.0;
    const NOISE: f64 = 0.02;

    /// A point on the sphere `deg` from its +x pole, turned `turn` round it.
    fn cap(deg: f64, turn: f64) -> DVec3 {
        let a = (deg * PI) / 180.0;
        let t = (turn * PI) / 180.0;
        DVec3::new(R * a.cos(), R * a.sin() * t.cos(), R * a.sin() * t.sin())
    }

    /// A point inside a cell on the sphere: its corners' middle, on it.
    fn inside(corners: &[DVec3]) -> DVec3 {
        let c = corners.iter().fold(DVec3::ZERO, |s, p| DVec3::new(s.x + p.x, s.y + p.y, s.z + p.z));
        let l = c.length();
        DVec3::new((R * c.x) / l, (R * c.y) / l, (R * c.z) / l)
    }

    /// The noisy sphere the tests share, as ScanRuler's share theirs.
    fn scan() -> &'static ScanPaths {
        static SCAN: OnceLock<ScanPaths> = OnceLock::new();
        SCAN.get_or_init(|| {
            let m = sphere_scan(R, NOISE, 6);
            ScanPaths::new(m.positions, m.indices)
        })
    }

    /// A polygon of n corners round the pole, `deg` out, as a network. Straight sides, great
    /// circles: bowed out much, a pentagon's sides meet at under the corner rule's 30° and it
    /// reads as one round side.
    fn polygon(n: u32, deg: f64, centre: Option<DVec3>) -> NetworkFeature {
        let nodes: Vec<(u32, DVec3)> = (0..n).map(|k| (k + 1, cap(deg, (360.0 * k as f64) / n as f64))).collect();
        let curves: Vec<_> = (0..n).map(|k| curve(k + 1, ((k + 1) % n) + 1)).collect();
        let mut f = network(&nodes, &curves, &[cap(2.0, 0.0)]);
        f.cells[0].centre = centre;
        f
    }

    fn read(scan: &ScanPaths, f: &NetworkFeature) -> NetworkReading<NetworkFitResult> {
        match read_network_fitted(scan, f, None) {
            Ok(r) => r,
            Err(e) => panic!("{}", e.0),
        }
    }

    fn patches(o: &CellOutcome) -> Option<&Vec<usize>> {
        match o {
            CellOutcome::Filled { patches, .. } => Some(patches),
            CellOutcome::Refused { .. } => None,
        }
    }

    fn splits_a_polygon_on_a_sphere_cap(n: usize) {
        let r = read(scan(), &polygon(n as u32, 35.0, None));
        let out = &r.outcomes[0];
        let CellOutcome::Filled { sides, patches, .. } = out else { panic!("{out:?}") };
        assert_eq!(patches.len(), n);
        assert_eq!(*sides, n);
        let fit = r.fit.as_ref().expect("a fit");
        for p in &fit.patches {
            assert!(p.rms > 0.5 * NOISE, "rms {}", p.rms);
            assert!(p.rms < 1.5 * NOISE, "rms {}", p.rms);
        }
        // The spokes are shared between the quads, and tangent across.
        let split = r.split.as_ref().expect("split");
        let spokes: Vec<u32> = split.spokes.iter().map(|s| s.curve).collect();
        assert_eq!(spokes.len(), n);
        let across: Vec<_> = fit.shared.iter().filter(|s| spokes.contains(&s.curve)).collect();
        assert_eq!(across.len(), n);
        for s in across {
            assert!(s.gap < 1e-9, "gap {}", s.gap);
            assert!(s.angle < 0.5, "angle {}", s.angle);
        }
        // The centre: a corner of n patches, its angle reported.
        assert_eq!(fit.corners.len(), 1);
        assert_eq!(fit.corners[0].patches, n);
        assert!(fit.corners[0].angle.is_finite());
        // The centre is the cap's middle, near the pole.
        let c = split.centres[0].at;
        assert!(c.y.hypot(c.z) < 3.0, "{c}");
    }

    #[test]
    fn splits_a_triangle_on_a_sphere_cap_into_3_patches() {
        splits_a_polygon_on_a_sphere_cap(3);
    }

    #[test]
    fn splits_a_pentagon_on_a_sphere_cap_into_5_patches() {
        splits_a_polygon_on_a_sphere_cap(5);
    }

    #[test]
    fn takes_a_centre_dragged_elsewhere() {
        let want = cap(8.0, 90.0);
        let r = read(scan(), &polygon(3, 35.0, Some(want)));
        assert_eq!(patches(&r.outcomes[0]).map(|p| p.len()), Some(3));
        let c = r.split.as_ref().expect("split").centres[0].at;
        assert!((c - want).length() < 0.2, "{c}");
    }

    /// A quad and, beside it, a triangle on its right side.
    fn beside_triangle() -> NetworkFeature {
        let nodes = [(1, cap(30.0, 150.0)), (2, cap(30.0, 210.0)), (3, cap(8.0, 240.0)), (4, cap(8.0, 120.0)), (5, cap(40.0, 290.0))];
        network(&nodes, &[curve(1, 2), curve(2, 3), curve(3, 4), curve(4, 1), curve(3, 5), curve(5, 2)], &[cap(18.0, 180.0), inside(&[nodes[2].1, nodes[4].1, nodes[1].1])])
    }

    #[test]
    fn splits_the_side_a_triangle_shares_with_a_four_sided_cell_by_a_t_node() {
        let r = read(scan(), &beside_triangle());
        let counts: Vec<String> = r
            .outcomes
            .iter()
            .map(|o| match o {
                CellOutcome::Filled { patches, .. } => patches.len().to_string(),
                CellOutcome::Refused { refused, .. } => refused.clone(),
            })
            .collect();
        assert_eq!(counts, vec!["1", "3"]);
        let tee: Vec<_> = r.fit.as_ref().expect("a fit").shared.iter().filter(|s| s.tee).copied().collect();
        // The quad's side 2→3 is now two curves through the triangle's T-node: each half is a
        // whole side of one of the triangle's quads.
        assert_eq!(tee.len(), 2);
        // A T-junction's two edges are fitted apart: microns, not nothing — the long side
        // fitted to its two curves as fitted.
        for s in &tee {
            assert!(s.gap < 0.005, "gap {}", s.gap);
        }
        let split = r.split.as_ref().expect("split");
        let node = r.traced.nodes.iter().find(|n| n.id >= DERIVED_IDS && !split.centres.iter().any(|c| c.at == n.at));
        assert!(node.is_some());
    }

    #[test]
    fn holds_a_t_junction_tangent() {
        // Noisier, on a net of sixteen spans: left free, the quad and the triangle's quads met
        // at 2.1° across the T-junction.
        let noisy = sphere_scan(R, 0.05, 6);
        let mut f = beside_triangle();
        f.spans = f.curves.iter().map(|c| (c.id.to_string(), 16.0)).collect();
        let r = read(&ScanPaths::new(noisy.positions, noisy.indices), &f);
        let tee: Vec<_> = r.fit.as_ref().expect("a fit").shared.iter().filter(|s| s.tee).copied().collect();
        assert_eq!(tee.len(), 2);
        for s in &tee {
            assert!(s.angle < 1.0, "angle {}", s.angle);
        }
    }

    #[test]
    fn tmp_right_triangle_on_a_flat_grid() {
        use crate::network_meshes::{grid_scan, GridOptions};
        let m = grid_scan(240, 160, &|u, v| DVec3::new(-120.0 + 120.0 * u, -40.0 + 80.0 * v, 0.0), &GridOptions { noise: 0.0, ..Default::default() });
        let scan = ScanPaths::new(m.positions, m.indices);
        for (name, c) in [("right", DVec3::new(-100.0, 10.0, 0.0)), ("iso", DVec3::new(-90.0, 8.0, 0.0))] {
            let nodes = [(1, DVec3::new(-100.0, -10.0, 0.0)), (2, DVec3::new(-80.0, -10.0, 0.0)), (3, c)];
            let f = network(&nodes, &[curve(1, 2), curve(2, 3), curve(3, 1)], &[(nodes[0].1 + nodes[1].1 + nodes[2].1) / 3.0]);
            let r = read(&scan, &f);
            eprintln!("{name}: {:?}", r.outcomes);
        }
    }
}
