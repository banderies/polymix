// SPDX-License-Identifier: AGPL-3.0-only
//! Suggest creases (PN12): the scan's sharp edges and folds, traced and offered as crease
//! curves for a patch network; candidates drawn dashed, accepted one by one or all at once.
//!
//! Ported from ScanRuler (`plugins/reverse/core/networkCreases.ts`), © Stefan Hermann
//! (CNC Kitchen), AGPL-3.0-only.
//!
//! An edge of the scan is on a crease where the surface turns sharply across it. Raw, the
//! angle between an edge's two triangles is mostly noise: 0.02 mm of it on edges of 0.5 mm
//! tips neighbouring triangles up to 27° apart. So the triangles' normals are filtered first,
//! three rounds of a bilateral mean: each becomes the mean of the normals of the triangles
//! sharing a corner with it, by area, each weighed by a Gaussian in how far it is from its
//! own. Noise is averaged away (on the same sheet no edge then turns more than 0.8°) while
//! across a crease the other side's normals are tens of degrees off and weigh next to
//! nothing, so a 90° edge keeps its 90°. A plain mean over the same ring does not: one round
//! takes a 35° fold down to 11° and a 90° edge to 27°, three rounds to 7° and 17°.
//!
//! Two angles are read at an edge. Its own, between its two triangles' filtered normals. And
//! the angle across it a few edges out (`across_angle`): between the mean normals of the
//! triangles 1.5 to 4 edges off it on either side. A crease that crosses the triangles
//! slantwise shares its turn out between two edges (a 30° fold puts no more than 24° on any
//! one) but across it reads its whole 30°; a speck of noise the filter left turns its own
//! edges and nothing a few edges out.
//!
//! The candidates are the edges whose own angle is over the weak threshold, thinned to one
//! line by non-maximum suppression across the crease: an edge goes where a near-parallel
//! edge beside it turns more. Their vertices, joined by every edge of the scan between two of
//! them, make pieces, and a piece is kept that holds an edge whose angle across is over the
//! strong threshold: hysteresis, the seed read across, not at the edge alone, which is what
//! finds a slantwise fold and leaves noise out.
//!
//! Each piece is traced into paths, TEASAR's way: from the vertex farthest from the root
//! along the tree of shortest ways back to the root or to a path already taken, the band
//! round each path covered, until none is left. The ways keep to the kept edges (a step off
//! them costs three times its length) so they follow the crease, not the inside of its
//! bends. A tip just past the band of a path already taken (the other half of a loop, a
//! crease run into a corner) is joined to it.
//!
//! A round is no crease, and its edges can pass the weak threshold: a radius r over edges h
//! apart turns about h / r an edge, 14° at 2 mm on 0.5 mm edges. What tells it from a crease
//! is where the turn is: a crease puts it on one edge, or two slantwise; a round a little on
//! every edge. So along each path the share of the angle across that the edge itself turns
//! is read, as a running median (`crease_runs`), and the path kept only where it is 0.4 or
//! more: 0.51 to 1 on every crease measured, 0.16 to 0.31 along the rounds. The ridges are
//! traced again without the rounds found, since a branch down a round, traced first, takes
//! the band of the crease it leaves with it.
//!
//! The paths' ends within MERGE_MM are one node, nearest first and never wider than MERGE_MM
//! across: a crease cut slantwise dips under the weak angle for an edge or two at a time, and
//! its pieces a millimetre apart would otherwise all run together into one node. Loose ends
//! pointing at each other a little farther apart are bridged; two chains through a node no
//! other reaches are one; spurs and crumbs go; and a chain shorter than four times the
//! scatter-scaled span is dropped: the span Auto would give it, eight times the scan's
//! scatter, never under three mean edges. On the scans measured the floor rules (the scatter
//! is 0.01 to 0.02 mm, the span it gives 0.16 mm): without it, a boss 1 mm across left three
//! to five curves of 2 to 6 mm, and noise of 0.12 mm on 0.5 mm edges a chain of 3.5 mm; with
//! it (and a group of curves shorter than it all together dropped) neither, while the rims
//! of a boss 2 mm across, 8 mm round its top, stay.
//!
//! The chains are cut at their corners, where they turn 45° or more over 4 mm either side,
//! and a closed one with fewer than three into four, so that every curve runs between two
//! nodes. A node where chains end is put where they meet carried on straight, and taken onto
//! the scan. Each chain is fitted as a network curve is (PN2), and its grips are the fitted
//! curve's points, each as far on from the last as the chord between them stays within a
//! tenth of an edge of it (so that the scan path between two grips, a plane section, keeps
//! to the crease round its bends) and never more than 10 mm apart. Its angle is the mean
//! angle across it, read along it on the scan.
//!
//! Measured (ScanRuler's `reNetworkCreases.test.ts`, ported below): a box with sharp rims and
//! rounds of 4 mm on 0.5 mm edges, or of 2 mm, comes out as its two rims and nothing on the
//! rounds, the curves within 0.03 mm of the rims, the scan paths through their grips within
//! 0.09 mm, the angles 90° ± 0.15°; one with every edge sharp as its twelve edges, their
//! nodes within 0.015 mm of its corners; a 35° or 40° fold across a sheet with 0.02 mm of
//! noise as one curve within 0.13 mm of it, end to end, and a 30° fold within 0.2 mm, short
//! of the edges by 1.1 and 1.8 mm, where its angle dips under the weak one. A box of 320,000
//! triangles in a 2-million-triangle scan takes 0.25 to 0.3 s in ScanRuler, half of it the
//! filter; the whole scan, 1.4 s.
//!
//! Two of ScanRuler's numbers are kept as it keeps them, in `Float32Array`s: each edge's
//! own angle, and the angle across an edge once read (the first read of it is used at full
//! precision, as the assignment ScanRuler caches it by gives it back).

use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;

use glam::DVec3;
use serde::{Deserialize, Serialize};

use crate::network_curve::{auto_spans, fit_curve, sample_curve, CurveEnds, CurvePoints};
use crate::scan_path::ScanPaths;

/// The hysteresis thresholds, deg: an edge whose own angle is over WEAK_DEG is a candidate,
/// and a piece of candidates is kept that holds one whose angle across is over STRONG_DEG.
pub const STRONG_DEG: f64 = 25.0;
pub const WEAK_DEG: f64 = 12.0;
/// Chain ends this close are one node, mm.
pub const MERGE_MM: f64 = 2.0;
/// The normal filter's range in |n − n'| (0.2 is 11.5°) and its rounds. At 0.3 a slantwise
/// 35° fold came out at 16° on its best edge; at 0.15, noise of a tenth of an edge stuck at
/// up to 45°.
const RANGE: f64 = 0.2;
const ROUNDS: usize = 3;
/// How near parallel an edge beside another has to be to suppress it, as the cosine of
/// 35°.
fn parallel_cos() -> f64 {
    ((35.0 * PI) / 180.0).cos()
}
/// The angle across an edge: the triangles from ACROSS_SKIP to ACROSS_REACH mean edges off
/// it, either side; past the two edges a slantwise crease shares its turn between, and short
/// of a step's other rim on most parts.
const ACROSS_SKIP: f64 = 1.5;
const ACROSS_REACH: f64 = 4.0;
/// How far round a path its band of ridge vertices is taken as covered, in mean edges, and
/// how much farther a new path's tip may lie from one already taken to be joined to it.
const COVER_EDGES: f64 = 2.0;
const JOIN_EDGES: f64 = 2.0;
/// What a step off the kept edges costs a path, per its length.
const OFF_RIDGE: f64 = 3.0;
/// A path shorter than this many mean edges is a spur off the band.
const SPUR_EDGES: f64 = 2.0;
/// The least share of the angle across a crease's edges turn themselves, as a running median
/// over this many steps either side (`crease_runs`).
const CONCENTRATION: f64 = 0.4;
const CONCENTRATION_WINDOW: usize = 3;
/// How many times the ridges are traced, each without the rounds the one before found.
const TRACE_PASSES: usize = 3;
/// The shortest chain kept: MIN_SPANS of the scatter-scaled span (Auto's, SCATTER_SPAN times
/// the scan's scatter) never under SPAN_FLOOR_EDGES mean edges (see the module
/// documentation).
const MIN_SPANS: f64 = 4.0;
const SCATTER_SPAN: f64 = 8.0;
const SPAN_FLOOR_EDGES: f64 = 3.0;
/// The vertices sampled for the scan's scatter.
const SCATTER_SAMPLES: usize = 4000;
/// Loose ends up to BRIDGE_MM apart, pointing at each other within BRIDGE_DEG, are bridged.
const BRIDGE_MM: f64 = 2.0 * MERGE_MM;
const BRIDGE_DEG: f64 = 20.0;
/// A chain has a corner where its chords CORNER_REACH mm back and on turn CORNER_TURN_DEG or
/// more: the middle of a 4 mm round in a rim turns 55° so, a 20 mm circle 11°.
const CORNER_TURN_DEG: f64 = 45.0;
const CORNER_REACH: f64 = 2.0 * MERGE_MM;
/// A closed chain with fewer than three corners is cut into this many.
const LOOP_PIECES: usize = 4;
/// A node is put where the lines its chains come in by meet: each from the chain's point
/// NODE_FAR mm from the node to its point NODE_NEAR mm from it, held to the ends' mean by
/// NODE_HOLD a line.
const NODE_FAR: f64 = 2.0 * MERGE_MM;
const NODE_NEAR: f64 = MERGE_MM / 2.0;
const NODE_HOLD: f64 = 1e-3;
/// The grips: the chord between two may leave the fitted curve by this many mean edges, and
/// they are never more than GRIP_SPACING mm apart.
const GRIP_SAG_EDGES: f64 = 0.1;
const GRIP_SPACING: f64 = 10.0;
/// A fitted curve is sampled every SAMPLE_EDGES mean edges, at most MAX_SAMPLES times.
const SAMPLE_EDGES: f64 = 0.5;
const MAX_SAMPLES: usize = 4000;
/// The angle along a curve: the scan's normal ANGLE_OFFSET_EDGES off it on either side,
/// averaged over ANGLE_RADIUS_EDGES, every ANGLE_STEP mm.
const ANGLE_OFFSET_EDGES: f64 = 3.0;
const ANGLE_RADIUS_EDGES: f64 = 1.5;
const ANGLE_STEP: f64 = 2.0;

/// A node where suggested curves end, ends within MERGE_MM of each other merged.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CreaseNode {
    pub id: u32,
    pub at: DVec3,
}

/// A suggested crease: its two nodes, a few grips along it (as `NetworkCurve::via` takes
/// them: points on the scan, enough that the scan path between consecutive ones follows the
/// crease), its length (mm), its mean dihedral angle (deg), and the fitted curve's samples.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SuggestedCrease {
    pub from: u32,
    pub to: u32,
    pub via: Vec<DVec3>,
    pub length: f64,
    pub angle: f64,
    /// x, y, z per sample of the fitted curve.
    pub points: Vec<f32>,
}

/// The scan's creases as a network's nodes and curves.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct CreaseSuggestion {
    pub nodes: Vec<CreaseNode>,
    pub curves: Vec<SuggestedCrease>,
}

/// A box the suggestion reads the scan's triangles in: those whose middles lie in it.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CreaseBox {
    pub min: DVec3,
    pub max: DVec3,
}

/// What a suggestion is asked for: the box (the whole scan when absent) and the thresholds
/// (STRONG_DEG and WEAK_DEG when absent; the weak one never over the strong).
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct CreaseOptions {
    #[serde(default, rename = "box", skip_serializing_if = "Option::is_none")]
    pub bounds: Option<CreaseBox>,
    #[serde(default, rename = "strongDeg", skip_serializing_if = "Option::is_none")]
    pub strong_deg: Option<f64>,
    #[serde(default, rename = "weakDeg", skip_serializing_if = "Option::is_none")]
    pub weak_deg: Option<f64>,
}

/// The triangles a suggestion reads, as a mesh of their own.
struct Region {
    /// Three local vertices per triangle.
    tri: Vec<u32>,
    triangles: usize,
    /// x, y, z per local vertex.
    xyz: Vec<f64>,
    vertices: usize,
    /// The triangles round each vertex (CSR).
    vt_offsets: Vec<u32>,
    vt_list: Vec<u32>,
    /// Each edge's two vertices, lower first, and the triangles either side: the second -1 on
    /// a rim, or where more than two meet.
    ea: Vec<u32>,
    eb: Vec<u32>,
    t1: Vec<i32>,
    t2: Vec<i32>,
    edges: usize,
    /// Each triangle's three edges, side k from corner k to k + 1.
    tri_edge: Vec<u32>,
    /// The edges round each vertex (CSR).
    ve_offsets: Vec<u32>,
    ve_list: Vec<u32>,
}

impl Region {
    fn at(&self, v: usize) -> DVec3 {
        DVec3::new(self.xyz[v * 3], self.xyz[v * 3 + 1], self.xyz[v * 3 + 2])
    }

    /// The edges round vertex v.
    fn edges_of(&self, v: usize) -> &[u32] {
        &self.ve_list[self.ve_offsets[v] as usize..self.ve_offsets[v + 1] as usize]
    }

    /// The other end of edge e from v.
    fn other(&self, e: usize, v: usize) -> usize {
        if self.ea[e] as usize == v {
            self.eb[e] as usize
        } else {
            self.ea[e] as usize
        }
    }

    /// How far apart local vertices v and w are.
    fn step(&self, v: usize, w: usize) -> f64 {
        hypot3(self.xyz[w * 3] - self.xyz[v * 3], self.xyz[w * 3 + 1] - self.xyz[v * 3 + 1], self.xyz[w * 3 + 2] - self.xyz[v * 3 + 2])
    }
}

/// √(x² + y² + z²).
fn hypot3(x: f64, y: f64, z: f64) -> f64 {
    (x * x + y * y + z * z).sqrt()
}

/// The angle in degrees whose cosine is `c`, held to [-1, 1].
fn acos_deg(c: f64) -> f64 {
    (c.clamp(-1.0, 1.0).acos() * 180.0) / PI
}

/// The angle across each edge, read once and kept as ScanRuler keeps it (see the module
/// documentation).
struct Across<'a> {
    r: &'a Region,
    normals: &'a [f64],
    edge: f64,
    kept: Vec<f32>,
    stamp: Vec<i32>,
    mark: i32,
    queue: Vec<usize>,
}

impl<'a> Across<'a> {
    fn new(r: &'a Region, normals: &'a [f64], edge: f64) -> Self {
        Across { r, normals, edge, kept: vec![-1.0; r.edges], stamp: vec![0; r.triangles], mark: 0, queue: Vec::new() }
    }

    fn get(&mut self, e: usize) -> f64 {
        if self.kept[e] >= 0.0 {
            return self.kept[e] as f64;
        }
        let a = across_angle(self.r, self.normals, e, self.edge, &mut self.stamp, &mut self.mark, &mut self.queue);
        self.kept[e] = a as f32;
        a
    }
}

/// The scan's creases, in `opts.bounds` when given: see the module documentation.
pub fn suggest_creases(scan: &ScanPaths, opts: &CreaseOptions) -> CreaseSuggestion {
    let strong = opts.strong_deg.unwrap_or(STRONG_DEG);
    let weak = opts.weak_deg.unwrap_or(WEAK_DEG).min(strong);
    let r = region_of(scan, opts.bounds.as_ref());
    if r.triangles == 0 {
        return CreaseSuggestion::default();
    }
    let normals = filtered_normals(&r);
    let angle = edge_angles(&r, &normals);
    let kept = thinned(&r, &angle, weak);
    let min_length = MIN_SPANS * (SCATTER_SPAN * scatter_of(&r, &normals)).max(SPAN_FLOOR_EDGES * scan.edge());
    let mut across = Across::new(&r, &normals, scan.edge());
    // Traced again without the vertices of the runs along rounds: a branch down a round,
    // traced first, takes the band round the crease it leaves with it (a rim's corner where a
    // round meets it).
    let mut off = vec![false; r.vertices];
    let mut runs: Vec<Vec<usize>> = Vec::new();
    for _ in 0..TRACE_PASSES {
        let paths = ridge_paths(&r, &kept, &mut |e| across.get(e) >= strong, scan.edge(), &off);
        runs = crease_runs(&r, &paths, &kept, &mut |e| angle[e] as f64 / (angle[e] as f64).max(across.get(e)), SPUR_EDGES * scan.edge());
        let mut in_run = vec![false; r.vertices];
        for p in &runs {
            for &v in p {
                in_run[v] = true;
            }
        }
        let mut dropped = 0;
        for p in &paths {
            for &v in p {
                if in_run[v] || off[v] {
                    continue;
                }
                off[v] = true;
                dropped += 1;
            }
        }
        if dropped == 0 {
            break;
        }
    }
    let linked = settle_nodes(scan, split_chains(link_chains(&r, &runs, min_length)), min_length);
    curves_of(scan, linked, weak, min_length)
}

// ---------------------------------------------------------------------------------------
// The region and its edges.

/// The triangles in `bounds` (their middles in it) or all of them, as a mesh of their own
/// with its edges. An edge on the box's side has one triangle in it and is a rim.
fn region_of(scan: &ScanPaths, bounds: Option<&CreaseBox>) -> Region {
    let pos = scan.positions();
    let idx = scan.indices();
    let chosen: Vec<u32> = match bounds {
        Some(b) => scan
            .triangles_in(b.min, b.max)
            .into_iter()
            .filter(|&t| {
                let t = t as usize;
                (0..3).all(|k| {
                    let c = (pos[idx[t * 3] as usize * 3 + k] as f64 + pos[idx[t * 3 + 1] as usize * 3 + k] as f64 + pos[idx[t * 3 + 2] as usize * 3 + k] as f64) / 3.0;
                    !(c < b.min[k] || c > b.max[k])
                })
            })
            .collect(),
        None => (0..scan.triangle_count() as u32).collect(),
    };
    let n = chosen.len();
    let mut local_of = vec![-1i64; pos.len() / 3];
    let mut tri = vec![0u32; n * 3];
    let mut m = 0usize;
    let mut xyz: Vec<f64> = Vec::with_capacity((n * 3).min(pos.len() / 3) * 3);
    for (i, &t) in chosen.iter().enumerate() {
        for k in 0..3 {
            let v = idx[t as usize * 3 + k] as usize;
            let mut l = local_of[v];
            if l < 0 {
                l = m as i64;
                m += 1;
                local_of[v] = l;
                xyz.extend_from_slice(&[pos[v * 3] as f64, pos[v * 3 + 1] as f64, pos[v * 3 + 2] as f64]);
            }
            tri[i * 3 + k] = l as u32;
        }
    }
    // The triangles round each vertex.
    let mut vt_offsets = vec![0u32; m + 1];
    for &v in &tri {
        vt_offsets[v as usize + 1] += 1;
    }
    for v in 0..m {
        vt_offsets[v + 1] += vt_offsets[v];
    }
    let mut vt_list = vec![0u32; n * 3];
    {
        let mut cursor: Vec<u32> = vt_offsets[..m].to_vec();
        for (i, &v) in tri.iter().enumerate() {
            vt_list[cursor[v as usize] as usize] = (i / 3) as u32;
            cursor[v as usize] += 1;
        }
    }
    // The edges: each triangle side put under its lower vertex, and the sides under one
    // vertex with the same upper vertex paired; a few a vertex, so a scan of them is cheaper
    // than a map of all.
    let next_corner = |i: usize| tri[i - (i % 3) + ((i % 3) + 1) % 3];
    let mut below = vec![0u32; m + 1];
    for i in 0..n * 3 {
        let (a, b) = (tri[i], next_corner(i));
        below[a.min(b) as usize + 1] += 1;
    }
    for v in 0..m {
        below[v + 1] += below[v];
    }
    let mut upper = vec![0u32; n * 3];
    let mut side = vec![0u32; n * 3];
    {
        let mut cursor: Vec<u32> = below[..m].to_vec();
        for i in 0..n * 3 {
            let (a, b) = (tri[i], next_corner(i));
            let lo = a.min(b) as usize;
            let at = cursor[lo] as usize;
            cursor[lo] += 1;
            upper[at] = a.max(b);
            side[at] = i as u32;
        }
    }
    let mut edge_of = vec![-1i64; n * 3];
    let mut mate_of = vec![-1i64; n * 3];
    let mut edges = 0usize;
    for v in 0..m {
        let end = below[v + 1] as usize;
        for i in below[v] as usize..end {
            if edge_of[i] >= 0 {
                continue;
            }
            let mut mate: i64 = -1;
            let mut many = false;
            for j in i + 1..end {
                if edge_of[j] >= 0 || upper[j] != upper[i] {
                    continue;
                }
                if mate < 0 {
                    mate = j as i64;
                } else {
                    many = true;
                }
            }
            if many {
                // More than two triangles on one edge: each its own, a rim.
                for j in i..end {
                    if edge_of[j] < 0 && upper[j] == upper[i] {
                        edge_of[j] = edges as i64;
                        edges += 1;
                    }
                }
                continue;
            }
            edge_of[i] = edges as i64;
            if mate >= 0 {
                edge_of[mate as usize] = edges as i64;
                mate_of[i] = mate;
            }
            edges += 1;
        }
    }
    let mut ea = vec![0u32; edges];
    let mut eb = vec![0u32; edges];
    let mut t1 = vec![-1i32; edges];
    let mut t2 = vec![-1i32; edges];
    let mut tri_edge = vec![0u32; n * 3];
    for v in 0..m {
        for i in below[v] as usize..below[v + 1] as usize {
            let e = edge_of[i] as usize;
            tri_edge[side[i] as usize] = e as u32;
            if t1[e] >= 0 {
                continue;
            }
            ea[e] = v as u32;
            eb[e] = upper[i];
            t1[e] = (side[i] / 3) as i32;
            if mate_of[i] >= 0 {
                t2[e] = (side[mate_of[i] as usize] / 3) as i32;
            }
        }
    }
    // The edges round each vertex.
    let mut ve_offsets = vec![0u32; m + 1];
    for e in 0..edges {
        ve_offsets[ea[e] as usize + 1] += 1;
        ve_offsets[eb[e] as usize + 1] += 1;
    }
    for v in 0..m {
        ve_offsets[v + 1] += ve_offsets[v];
    }
    let mut ve_list = vec![0u32; edges * 2];
    {
        let mut cursor: Vec<u32> = ve_offsets[..m].to_vec();
        for e in 0..edges {
            for v in [ea[e] as usize, eb[e] as usize] {
                ve_list[cursor[v] as usize] = e as u32;
                cursor[v] += 1;
            }
        }
    }
    Region { tri, triangles: n, xyz, vertices: m, vt_offsets, vt_list, ea, eb, t1, t2, edges, tri_edge, ve_offsets, ve_list }
}

// ---------------------------------------------------------------------------------------
// The normals and the angle at each edge.

/// The triangles' normals, filtered (see the module documentation).
fn filtered_normals(r: &Region) -> Vec<f64> {
    let n = r.triangles;
    let tri = &r.tri;
    let xyz = &r.xyz;
    let mut cur = vec![0.0f64; n * 3];
    let mut area = vec![0.0f64; n];
    for t in 0..n {
        let (a, b, c) = (tri[t * 3] as usize * 3, tri[t * 3 + 1] as usize * 3, tri[t * 3 + 2] as usize * 3);
        let (ux, uy, uz) = (xyz[b] - xyz[a], xyz[b + 1] - xyz[a + 1], xyz[b + 2] - xyz[a + 2]);
        let (vx, vy, vz) = (xyz[c] - xyz[a], xyz[c + 1] - xyz[a + 1], xyz[c + 2] - xyz[a + 2]);
        let (nx, ny, nz) = (uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx);
        let l = hypot3(nx, ny, nz);
        area[t] = l / 2.0;
        if l > 0.0 {
            cur[t * 3] = nx / l;
            cur[t * 3 + 1] = ny / l;
            cur[t * 3 + 2] = nz / l;
        }
    }
    let mut next = vec![0.0f64; n * 3];
    let mut stamp = vec![-1i64; n];
    let inv = 1.0 / (2.0 * RANGE * RANGE);
    for round in 0..ROUNDS {
        for t in 0..n {
            let (nx, ny, nz) = (cur[t * 3], cur[t * 3 + 1], cur[t * 3 + 2]);
            let (mut sx, mut sy, mut sz) = (0.0, 0.0, 0.0);
            let mark = (round * n + t) as i64;
            for k in 0..3 {
                let v = tri[t * 3 + k] as usize;
                for i in r.vt_offsets[v] as usize..r.vt_offsets[v + 1] as usize {
                    let u = r.vt_list[i] as usize;
                    if stamp[u] == mark {
                        continue;
                    }
                    stamp[u] = mark;
                    let (mx, my, mz) = (cur[u * 3], cur[u * 3 + 1], cur[u * 3 + 2]);
                    let d = 2.0 - 2.0 * (nx * mx + ny * my + nz * mz);
                    let w = area[u] * (-d * inv).exp();
                    sx += w * mx;
                    sy += w * my;
                    sz += w * mz;
                }
            }
            let l = hypot3(sx, sy, sz);
            if l > 0.0 {
                next[t * 3] = sx / l;
                next[t * 3 + 1] = sy / l;
                next[t * 3 + 2] = sz / l;
            } else {
                next[t * 3] = nx;
                next[t * 3 + 1] = ny;
                next[t * 3 + 2] = nz;
            }
        }
        std::mem::swap(&mut cur, &mut next);
    }
    cur
}

/// Each edge's angle between its two triangles' filtered normals, deg; 0 on a rim.
fn edge_angles(r: &Region, normals: &[f64]) -> Vec<f32> {
    let mut out = vec![0.0f32; r.edges];
    for (e, o) in out.iter_mut().enumerate() {
        let (a, b) = (r.t1[e], r.t2[e]);
        if a < 0 || b < 0 {
            continue;
        }
        let (a, b) = (a as usize * 3, b as usize * 3);
        let c = normals[a] * normals[b] + normals[a + 1] * normals[b + 1] + normals[a + 2] * normals[b + 2];
        *o = acos_deg(c) as f32;
    }
    out
}

/// The edges over `weak` that no near-parallel edge beside them, across the crease, outdoes:
/// non-maximum suppression across it.
fn thinned(r: &Region, angle: &[f32], weak: f64) -> Vec<bool> {
    let mut kept = vec![false; r.edges];
    let xyz = &r.xyz;
    let parallel = parallel_cos();
    let mut near: Vec<usize> = Vec::new();
    for e in 0..r.edges {
        let s = angle[e];
        if (s as f64) < weak || r.t2[e] < 0 {
            continue;
        }
        // The edges of the two triangles, and of the triangles across those.
        near.clear();
        for t in [r.t1[e], r.t2[e]] {
            let t = t as usize;
            for k in 0..3 {
                let f = r.tri_edge[t * 3 + k] as usize;
                if f == e {
                    continue;
                }
                near.push(f);
                let u = if r.t1[f] as usize == t { r.t2[f] } else { r.t1[f] };
                if u < 0 {
                    continue;
                }
                for j in 0..3 {
                    let g = r.tri_edge[u as usize * 3 + j] as usize;
                    if g != f {
                        near.push(g);
                    }
                }
            }
        }
        let (a, b) = (r.ea[e] as usize * 3, r.eb[e] as usize * 3);
        let (mut dx, mut dy, mut dz) = (xyz[b] - xyz[a], xyz[b + 1] - xyz[a + 1], xyz[b + 2] - xyz[a + 2]);
        let dl = hypot3(dx, dy, dz);
        let dl = if dl != 0.0 && !dl.is_nan() { dl } else { 1.0 };
        dx /= dl;
        dy /= dl;
        dz /= dl;
        let (mx, my, mz) = ((xyz[a] + xyz[b]) / 2.0, (xyz[a + 1] + xyz[b + 1]) / 2.0, (xyz[a + 2] + xyz[b + 2]) / 2.0);
        let mut top = true;
        for &f in &near {
            if f == e || angle[f] <= s {
                continue;
            }
            let (p, q) = (r.ea[f] as usize * 3, r.eb[f] as usize * 3);
            let (mut fx, mut fy, mut fz) = (xyz[q] - xyz[p], xyz[q + 1] - xyz[p + 1], xyz[q + 2] - xyz[p + 2]);
            let fl = hypot3(fx, fy, fz);
            let fl = if fl != 0.0 && !fl.is_nan() { fl } else { 1.0 };
            fx /= fl;
            fy /= fl;
            fz /= fl;
            if (fx * dx + fy * dy + fz * dz).abs() < parallel {
                continue;
            }
            // Beside it, not on along the crease: its middle farther off e's line than along
            // it.
            let ox = (xyz[p] + xyz[q]) / 2.0 - mx;
            let oy = (xyz[p + 1] + xyz[q + 1]) / 2.0 - my;
            let oz = (xyz[p + 2] + xyz[q + 2]) / 2.0 - mz;
            let along = ox * dx + oy * dy + oz * dz;
            let off = ox * ox + oy * oy + oz * oz - along * along;
            if off > along * along {
                top = false;
                break;
            }
        }
        if top {
            kept[e] = true;
        }
    }
    kept
}

/// The scan's scatter about its own surface: each sampled vertex's offset along its normal
/// from the middle of its neighbours, their median scaled to what noise σ on a plane gives a
/// vertex of six; 0.0200 mm read off a sheet with 0.02 of it, 0.0117 off a box with 0.01.
fn scatter_of(r: &Region, normals: &[f64]) -> f64 {
    let xyz = &r.xyz;
    let step = (r.vertices / SCATTER_SAMPLES).max(1);
    let mut offsets: Vec<f64> = Vec::new();
    let mut v = 0;
    while v < r.vertices {
        let k = (r.ve_offsets[v + 1] - r.ve_offsets[v]) as f64;
        if k >= 3.0 {
            let (mut cx, mut cy, mut cz) = (0.0, 0.0, 0.0);
            for &e in r.edges_of(v) {
                let w = r.other(e as usize, v) * 3;
                cx += xyz[w];
                cy += xyz[w + 1];
                cz += xyz[w + 2];
            }
            let (mut nx, mut ny, mut nz) = (0.0, 0.0, 0.0);
            for i in r.vt_offsets[v] as usize..r.vt_offsets[v + 1] as usize {
                let t = r.vt_list[i] as usize;
                nx += normals[t * 3];
                ny += normals[t * 3 + 1];
                nz += normals[t * 3 + 2];
            }
            let nl = hypot3(nx, ny, nz);
            if nl > 0.0 {
                offsets.push(((xyz[v * 3] - cx / k) * nx + (xyz[v * 3 + 1] - cy / k) * ny + (xyz[v * 3 + 2] - cz / k) * nz).abs() / nl);
            }
        }
        v += step;
    }
    if offsets.is_empty() {
        return 0.0;
    }
    offsets.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // |N(0, σ√(7/6))| has its median at 0.674 σ √(7/6) = 0.73 σ.
    offsets[offsets.len() >> 1] / 0.73
}

// ---------------------------------------------------------------------------------------
// The ridges: the angle across, and the paths.

/// The angle across edge e a few edges out, deg: between the mean filtered normals of the
/// triangles beside it on either side, ACROSS_SKIP to ACROSS_REACH mean edges off it; past
/// the triangles a crease crosses slantwise, which share its angle out between two edges.
fn across_angle(r: &Region, normals: &[f64], e: usize, edge: f64, stamp: &mut [i32], mark: &mut i32, queue: &mut Vec<usize>) -> f64 {
    let xyz = &r.xyz;
    let (a, b) = (r.ea[e] as usize * 3, r.eb[e] as usize * 3);
    let (mut dx, mut dy, mut dz) = (xyz[b] - xyz[a], xyz[b + 1] - xyz[a + 1], xyz[b + 2] - xyz[a + 2]);
    let dl = hypot3(dx, dy, dz);
    let dl = if dl != 0.0 && !dl.is_nan() { dl } else { 1.0 };
    dx /= dl;
    dy /= dl;
    dz /= dl;
    let (f, g) = (r.t1[e], r.t2[e]);
    if f < 0 || g < 0 {
        return 0.0;
    }
    let (f, g) = (f as usize, g as usize);
    let nx = normals[f * 3] + normals[g * 3];
    let ny = normals[f * 3 + 1] + normals[g * 3 + 1];
    let nz = normals[f * 3 + 2] + normals[g * 3 + 2];
    // Across: square to the edge, in the plane between its two triangles.
    let (mut sx, mut sy, mut sz) = (dy * nz - dz * ny, dz * nx - dx * nz, dx * ny - dy * nx);
    let sl = hypot3(sx, sy, sz);
    if !(sl > 0.0) {
        return 0.0;
    }
    sx /= sl;
    sy /= sl;
    sz /= sl;
    let (mx, my, mz) = ((xyz[a] + xyz[b]) / 2.0, (xyz[a + 1] + xyz[b + 1]) / 2.0, (xyz[a + 2] + xyz[b + 2]) / 2.0);
    let skip = ACROSS_SKIP * edge;
    let reach = ACROSS_REACH * edge;
    let mut side = [0.0f64; 6];
    *mark += 1;
    let m = *mark;
    queue.clear();
    queue.push(f);
    queue.push(g);
    stamp[f] = m;
    stamp[g] = m;
    let mut q = 0;
    while q < queue.len() {
        let t = queue[q];
        q += 1;
        let (i, j, k) = (r.tri[t * 3] as usize * 3, r.tri[t * 3 + 1] as usize * 3, r.tri[t * 3 + 2] as usize * 3);
        let cx = (xyz[i] + xyz[j] + xyz[k]) / 3.0 - mx;
        let cy = (xyz[i + 1] + xyz[j + 1] + xyz[k + 1]) / 3.0 - my;
        let cz = (xyz[i + 2] + xyz[j + 2] + xyz[k + 2]) / 3.0 - mz;
        if cx * cx + cy * cy + cz * cz > reach * reach {
            continue;
        }
        let off = cx * sx + cy * sy + cz * sz;
        if off.abs() >= skip {
            let o = if off > 0.0 { 0 } else { 3 };
            side[o] += normals[t * 3];
            side[o + 1] += normals[t * 3 + 1];
            side[o + 2] += normals[t * 3 + 2];
        }
        for n in 0..3 {
            let h = r.tri_edge[t * 3 + n] as usize;
            let u = if r.t1[h] as usize == t { r.t2[h] } else { r.t1[h] };
            if u < 0 || stamp[u as usize] == m {
                continue;
            }
            stamp[u as usize] = m;
            queue.push(u as usize);
        }
    }
    let l1 = hypot3(side[0], side[1], side[2]);
    let l2 = hypot3(side[3], side[4], side[5]);
    if !(l1 > 0.0 && l2 > 0.0) {
        return 0.0;
    }
    acos_deg((side[0] * side[3] + side[1] * side[4] + side[2] * side[5]) / (l1 * l2))
}

/// A binary min-heap of (vertex, distance).
#[derive(Default)]
struct Heap {
    ids: Vec<usize>,
    keys: Vec<f64>,
}

impl Heap {
    fn len(&self) -> usize {
        self.ids.len()
    }

    fn push(&mut self, id: usize, key: f64) {
        let mut i = self.ids.len();
        self.ids.push(id);
        self.keys.push(key);
        while i > 0 {
            let parent = (i - 1) >> 1;
            if self.keys[parent] <= key {
                break;
            }
            self.ids[i] = self.ids[parent];
            self.keys[i] = self.keys[parent];
            i = parent;
        }
        self.ids[i] = id;
        self.keys[i] = key;
    }

    fn pop(&mut self) -> Option<(usize, f64)> {
        let top = (*self.ids.first()?, self.keys[0]);
        let last_id = self.ids.pop()?;
        let last_key = self.keys.pop()?;
        let n = self.ids.len();
        if n > 0 {
            let mut i = 0;
            loop {
                let l = 2 * i + 1;
                if l >= n {
                    break;
                }
                let r = l + 1;
                let c = if r < n && self.keys[r] < self.keys[l] { r } else { l };
                if self.keys[c] >= last_key {
                    break;
                }
                self.ids[i] = self.ids[c];
                self.keys[i] = self.keys[c];
                i = c;
            }
            self.ids[i] = last_id;
            self.keys[i] = last_key;
        }
        Some(top)
    }
}

/// What the ridge tracing's Dijkstra keeps between its searches.
struct Search {
    piece: Vec<i64>,
    dist: Vec<f64>,
    parent: Vec<i64>,
    seen: Vec<i64>,
    settled: Vec<i64>,
    blocked: Vec<i64>,
    stamp: i64,
    heap: Heap,
}

impl Search {
    /// Dijkstra over piece `id` from `sources`, out to `limit`, round the vertices blocked
    /// under this stamp: the vertices reached, nearest first; their distances and parents in
    /// `dist` and `parent`. `along` weighs an edge of the scan that is not a kept one
    /// OFF_RIDGE times its length, so that a way keeps to the crease's own edges and does not
    /// cut the inside of its bends a row of vertices in.
    fn reach(&mut self, r: &Region, kept: &[bool], id: i64, sources: &[usize], limit: f64, along: bool) -> Vec<usize> {
        let mut out = Vec::new();
        let stamp = self.stamp;
        for &s in sources {
            self.dist[s] = 0.0;
            self.parent[s] = -1;
            self.seen[s] = stamp;
            self.heap.push(s, 0.0);
        }
        while self.heap.len() > 0 {
            let Some((v, d)) = self.heap.pop() else { break };
            if self.settled[v] == stamp || d > self.dist[v] {
                continue;
            }
            self.settled[v] = stamp;
            out.push(v);
            for &e in r.edges_of(v) {
                let e = e as usize;
                let w = r.other(e, v);
                if self.piece[w] != id || self.settled[w] == stamp || self.blocked[w] == stamp {
                    continue;
                }
                let nd = d + if along && !kept[e] { OFF_RIDGE } else { 1.0 } * r.step(v, w);
                if nd > limit {
                    continue;
                }
                if self.seen[w] != stamp || nd < self.dist[w] {
                    self.seen[w] = stamp;
                    self.dist[w] = nd;
                    self.parent[w] = v as i64;
                    self.heap.push(w, nd);
                }
            }
        }
        out
    }
}

/// The ridge vertices' paths: the kept edges' vertices, joined by every edge of the scan
/// between two of them; the pieces that hold a seed edge (hysteresis), each traced into paths
/// by repeated longest ways (see the module documentation). Each path is a list of vertices.
fn ridge_paths(r: &Region, kept: &[bool], seed: &mut dyn FnMut(usize) -> bool, edge: f64, off: &[bool]) -> Vec<Vec<usize>> {
    let m = r.vertices;
    let cover = COVER_EDGES * edge;
    let join_reach = cover + JOIN_EDGES * edge;
    let spur = SPUR_EDGES * edge;
    let mut ridge = vec![false; m];
    for e in 0..r.edges {
        if !kept[e] {
            continue;
        }
        ridge[r.ea[e] as usize] = true;
        ridge[r.eb[e] as usize] = true;
    }
    for v in 0..m {
        if off[v] {
            ridge[v] = false;
        }
    }
    let mut s = Search {
        piece: vec![-1; m],
        dist: vec![0.0; m],
        parent: vec![0; m],
        seen: vec![-1; m],
        settled: vec![-1; m],
        blocked: vec![-1; m],
        stamp: 0,
        heap: Heap::default(),
    };
    let mut on_path = vec![-1i64; m];
    let mut covered = vec![false; m];
    let mut from_root = vec![0.0f64; m];
    let mut up = vec![0i64; m];
    let mut paths: Vec<Vec<usize>> = Vec::new();
    let mut pieces: i64 = 0;
    for v0 in 0..m {
        if !ridge[v0] || s.piece[v0] >= 0 {
            continue;
        }
        // The piece: the ridge vertices joined to v0, and its kept edges.
        let id = pieces;
        pieces += 1;
        let mut members = vec![v0];
        let mut edges: Vec<usize> = Vec::new();
        s.piece[v0] = id;
        let mut q = 0;
        while q < members.len() {
            let v = members[q];
            q += 1;
            for &e in r.edges_of(v) {
                let e = e as usize;
                let w = r.other(e, v);
                if kept[e] && r.ea[e] as usize == v {
                    edges.push(e);
                }
                if ridge[w] && s.piece[w] < 0 {
                    s.piece[w] = id;
                    members.push(w);
                }
            }
        }
        if members.len() < 2 || !edges.iter().any(|&e| seed(e)) {
            continue;
        }
        // The root: the vertex farthest from any one; the tree of shortest ways from it.
        s.stamp += 1;
        let order = s.reach(r, kept, id, &[v0], f64::INFINITY, true);
        let mut root = v0;
        for &v in &order {
            if s.dist[v] > s.dist[root] {
                root = v;
            }
        }
        s.stamp += 1;
        let order = s.reach(r, kept, id, &[root], f64::INFINITY, true);
        for &v in &order {
            from_root[v] = s.dist[v];
            up[v] = s.parent[v];
        }
        // Paths from the farthest vertex not yet covered back along the tree to the root or
        // to a path already taken; the band round each covered.
        loop {
            let mut tip: Option<usize> = None;
            let mut far = -1.0;
            for &v in &order {
                if covered[v] {
                    continue;
                }
                if from_root[v] > far {
                    far = from_root[v];
                    tip = Some(v);
                }
            }
            let Some(tip) = tip else { break };
            let mut path = vec![tip];
            let mut v = tip;
            let mut len = 0.0;
            while on_path[v] < 0 && v != root {
                let w = up[v];
                if w < 0 {
                    break;
                }
                let w = w as usize;
                len += r.step(v, w);
                v = w;
                path.push(v);
            }
            // A tip just past the band of a path already taken (the other half of a loop, a
            // crease run into a corner) is joined to that path by the shortest way that keeps
            // off its own.
            s.stamp += 1;
            for &u in &path[1..] {
                s.blocked[u] = s.stamp;
            }
            let end = path[path.len() - 1];
            let near = s.reach(r, kept, id, &[tip], join_reach, false).into_iter().find(|&u| on_path[u] >= 0 && u != end);
            if let Some(near) = near {
                let mut lead: Vec<usize> = Vec::new();
                let mut u = near;
                while u != tip {
                    lead.push(u);
                    let p = s.parent[u];
                    if p < 0 {
                        break;
                    }
                    u = p as usize;
                }
                for i in 0..lead.len() {
                    len += r.step(lead[i], if i + 1 < lead.len() { lead[i + 1] } else { tip });
                }
                lead.extend_from_slice(&path);
                path = lead;
            }
            s.stamp += 1;
            for u in s.reach(r, kept, id, &path, cover, false) {
                covered[u] = true;
            }
            for &u in &path {
                covered[u] = true;
            }
            if len < spur {
                continue;
            }
            for &u in &path {
                if on_path[u] < 0 {
                    on_path[u] = paths.len() as i64;
                }
            }
            paths.push(path);
        }
    }
    split_at_joins(&paths, &on_path)
}

/// The paths cut where another path's end joins one partway along it: every piece runs from
/// end or join to end or join.
fn split_at_joins(paths: &[Vec<usize>], on_path: &[i64]) -> Vec<Vec<usize>> {
    let mut cuts: Vec<Vec<usize>> = vec![Vec::new(); paths.len()];
    for (p, path) in paths.iter().enumerate() {
        for end in [path[0], path[path.len() - 1]] {
            let q = on_path[end];
            if q < 0 || q as usize == p {
                continue;
            }
            let q = q as usize;
            if let Some(at) = paths[q].iter().position(|&x| x == end) {
                if at > 0 && at < paths[q].len() - 1 && !cuts[q].contains(&at) {
                    cuts[q].push(at);
                }
            }
        }
    }
    let mut out = Vec::new();
    for (p, path) in paths.iter().enumerate() {
        let mut at = cuts[p].clone();
        at.sort_unstable();
        let mut from = 0;
        for c in at {
            out.push(path[from..=c].to_vec());
            from = c;
        }
        out.push(path[from..].to_vec());
    }
    out
}

/// The paths cut to their runs along creases, not rounds. A crease puts its turn on one
/// edge, or two where it crosses the triangles slantwise; a round turns a little at every
/// edge. So along a path each kept edge's angle is set against the angle across it a few
/// edges out: the median of that share over CONCENTRATION_WINDOW steps either side is from
/// 0.51 to 1 on every path along a crease measured (sharp edges, rims, folds of 25° to 35°
/// cut slantwise with noise to 0.08 mm) and from 0.16 to 0.31 on every path down a round of
/// 1 to 2 mm on edges of 0.3 and 0.5 mm (0.38 once, on a round of 1 mm on 0.3 mm edges,
/// three edges in radius; no edge of a 4 mm round passed the weak angle at all). A run is
/// kept where it is CONCENTRATION or more.
fn crease_runs(r: &Region, paths: &[Vec<usize>], kept: &[bool], share: &mut dyn FnMut(usize) -> f64, spur: f64) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    let mut near: Vec<f64> = Vec::new();
    for p in paths {
        let steps = p.len().saturating_sub(1);
        let mut ratio = vec![f64::NAN; steps];
        for i in 0..steps {
            let (u, v) = (p[i], p[i + 1]);
            for &e in r.edges_of(u) {
                let e = e as usize;
                if (r.ea[e] as usize == v || r.eb[e] as usize == v) && kept[e] {
                    ratio[i] = share(e);
                }
            }
        }
        let mut good = vec![false; steps];
        for i in 0..steps {
            near.clear();
            for &x in &ratio[i.saturating_sub(CONCENTRATION_WINDOW)..=(i + CONCENTRATION_WINDOW).min(steps - 1)] {
                if !x.is_nan() {
                    near.push(x);
                }
            }
            if near.is_empty() {
                continue;
            }
            near.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            good[i] = near[near.len() >> 1] >= CONCENTRATION;
        }
        let mut i = 0;
        while i < steps {
            if !good[i] {
                i += 1;
                continue;
            }
            let mut j = i;
            let mut len = 0.0;
            while j < steps && good[j] {
                len += r.step(p[j], p[j + 1]);
                j += 1;
            }
            if len >= spur {
                out.push(p[i..=j].to_vec());
            }
            i = j;
        }
    }
    out
}

// ---------------------------------------------------------------------------------------
// Chains between nodes.

#[derive(Debug, Clone)]
struct Chain {
    /// x, y, z per point, from its `from` node to its `to` node.
    pts: Vec<f64>,
    from: usize,
    to: usize,
}

struct Linked {
    nodes: Vec<DVec3>,
    chains: Vec<Chain>,
    /// The nodes a chain was cut at, which lie on it as they are.
    cut: HashSet<usize>,
}

/// A chain's ends as they are when a loose end is read: where it is, and the way it points
/// out of its chain.
struct Loose {
    node: usize,
    chain: usize,
    at: DVec3,
    out: DVec3,
}

/// Two loose ends that point at each other across a gap of up to BRIDGE_MM, within
/// BRIDGE_DEG (a crease whose angle dips under the weak one for longer than MERGE_MM, as it
/// does in noise of a tenth of an edge) made one node, between them; nearest first. A
/// chain's own two ends are bridged only when it is long enough to be a loop.
fn bridge(nodes: &mut [DVec3], chains: &mut [Chain], min_length: f64) {
    let mut valence = vec![0usize; nodes.len()];
    for c in chains.iter() {
        valence[c.from] += 1;
        valence[c.to] += 1;
    }
    let mut loose: Vec<Loose> = Vec::new();
    for (i, c) in chains.iter().enumerate() {
        let length = polyline_length(&c.pts);
        for end in [0, 1] {
            let node = if end == 0 { c.from } else { c.to };
            if valence[node] != 1 {
                continue;
            }
            let pts = if end == 0 { c.pts.clone() } else { reversed(&c.pts) };
            let at = DVec3::new(pts[0], pts[1], pts[2]);
            let back = point_along(&pts, NODE_FAR.min(0.5 * length));
            let u = at - back;
            let l = u.length();
            if l > 0.0 {
                loose.push(Loose { node, chain: i, at, out: u / l });
            }
        }
    }
    let cos = ((BRIDGE_DEG * PI) / 180.0).cos();
    let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
    for i in 0..loose.len() {
        for j in i + 1..loose.len() {
            let (a, b) = (&loose[i], &loose[j]);
            if a.chain == b.chain && polyline_length(&chains[a.chain].pts) < min_length {
                continue;
            }
            let g = b.at - a.at;
            let d = g.length();
            if d > BRIDGE_MM {
                continue;
            }
            let along = if d > 0.0 { a.out.dot(g) / d } else { 1.0 };
            let back = if d > 0.0 { -b.out.dot(g) / d } else { 1.0 };
            let facing = -a.out.dot(b.out);
            if along >= cos && back >= cos && facing >= cos {
                pairs.push((d, i, j));
            }
        }
    }
    pairs.sort_by(|p, q| p.0.partial_cmp(&q.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut done = vec![false; loose.len()];
    for (_, i, j) in pairs {
        if done[i] || done[j] {
            continue;
        }
        done[i] = true;
        done[j] = true;
        let (a, b) = (&loose[i], &loose[j]);
        nodes[a.node] = DVec3::new((a.at.x + b.at.x) / 2.0, (a.at.y + b.at.y) / 2.0, (a.at.z + b.at.z) / 2.0);
        for c in chains.iter_mut() {
            if c.from == b.node {
                c.from = a.node;
            }
            if c.to == b.node {
                c.to = a.node;
            }
        }
    }
}

/// The points gathered into clusters, nearest pairs first, each no wider than MERGE_MM across
/// and never holding two points `apart` says must stay apart (transitively, ends a little
/// over a millimetre apart along a crease broken every millimetre would all be one): each
/// point's cluster, by its first point.
fn clusters(at: &[DVec3], apart: &dyn Fn(usize, usize) -> bool) -> Vec<usize> {
    let n = at.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| (at[a].x - at[b].x).partial_cmp(&0.0).unwrap_or(std::cmp::Ordering::Equal));
    let d = |i: usize, j: usize| hypot3(at[i].x - at[j].x, at[i].y - at[j].y, at[i].z - at[j].z);
    let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
    for a in 0..n {
        let mut b = a + 1;
        while b < n && at[order[b]].x - at[order[a]].x < MERGE_MM {
            let dist = d(order[a], order[b]);
            if dist < MERGE_MM {
                pairs.push((dist, order[a], order[b]));
            }
            b += 1;
        }
    }
    pairs.sort_by(|p, q| p.0.partial_cmp(&q.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut of: Vec<usize> = (0..n).collect();
    let mut members: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    for (_, i, j) in pairs {
        let (a, b) = (of[i], of[j]);
        if a == b {
            continue;
        }
        if members[a].iter().any(|&p| members[b].iter().any(|&q| apart(p, q) || d(p, q) >= MERGE_MM)) {
            continue;
        }
        let moved = std::mem::take(&mut members[b]);
        for &q in &moved {
            of[q] = a;
        }
        members[a].extend(moved);
    }
    of
}

fn polyline_length(pts: &[f64]) -> f64 {
    let mut l = 0.0;
    let mut i = 3;
    while i + 2 < pts.len() {
        l += hypot3(pts[i] - pts[i - 3], pts[i + 1] - pts[i - 2], pts[i + 2] - pts[i - 1]);
        i += 3;
    }
    l
}

fn reversed(pts: &[f64]) -> Vec<f64> {
    pts.chunks_exact(3).rev().flatten().copied().collect()
}

/// Two chains through a node no other reaches made one: the pieces of a crease cut
/// slantwise, whose angle dips under the weak one an edge or two at a time, and a chain cut
/// where a spur left it. One pass: joining leaves every other node's count as it was.
fn joined_through(nodes: &[DVec3], chains: Vec<Chain>) -> Vec<Chain> {
    let mut live: Vec<Option<Chain>> = chains.into_iter().map(Some).collect();
    let mut at: Vec<Vec<usize>> = vec![Vec::new(); nodes.len()];
    for (i, c) in live.iter().enumerate() {
        if let Some(c) = c {
            at[c.from].push(i);
            at[c.to].push(i);
        }
    }
    for n in 0..nodes.len() {
        if at[n].len() != 2 || at[n][0] == at[n][1] {
            continue;
        }
        let (ia, ib) = (at[n][0], at[n][1]);
        if live[ia].is_none() || live[ib].is_none() {
            continue;
        }
        let (Some(a), Some(b)) = (live[ia].take(), live[ib].take()) else { continue };
        let into = if a.to == n { a.pts.clone() } else { reversed(&a.pts) };
        let out = if b.from == n { b.pts.clone() } else { reversed(&b.pts) };
        let mut pts: Vec<f64> = into[..into.len().saturating_sub(3)].to_vec();
        pts.extend_from_slice(&[nodes[n].x, nodes[n].y, nodes[n].z]);
        pts.extend_from_slice(&out[3.min(out.len())..]);
        let c = Chain { pts, from: if a.to == n { a.from } else { a.to }, to: if b.from == n { b.to } else { b.from } };
        let ic = live.len();
        let ends = if c.from == c.to { vec![c.from] } else { vec![c.from, c.to] };
        live.push(Some(c));
        at[n].clear();
        for end in ends {
            for i in at[end].iter_mut() {
                if *i == ia || *i == ib {
                    *i = ic;
                }
            }
        }
    }
    live.into_iter().flatten().collect()
}

/// The paths' ends merged into nodes within MERGE_MM; chains through a node only two reach
/// joined into one; short chains hanging off a node dropped.
fn link_chains(r: &Region, paths: &[Vec<usize>], min_length: f64) -> Linked {
    let mut ends: Vec<DVec3> = Vec::new();
    for p in paths {
        for v in [p[0], p[p.len() - 1]] {
            ends.push(r.at(v));
        }
    }
    // Ends 2i and 2i + 1 are path i's.
    let group_of = clusters(&ends, &|i, j| i >> 1 == j >> 1);
    let mut node_of: HashMap<usize, usize> = HashMap::new();
    let mut sums: Vec<[f64; 4]> = Vec::new();
    let end_node: Vec<usize> = ends
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let root = group_of[i];
            let n = *node_of.entry(root).or_insert_with(|| {
                sums.push([0.0; 4]);
                sums.len() - 1
            });
            sums[n][0] += e.x;
            sums[n][1] += e.y;
            sums[n][2] += e.z;
            sums[n][3] += 1.0;
            n
        })
        .collect();
    let mut nodes: Vec<DVec3> = sums.iter().map(|s| DVec3::new(s[0] / s[3], s[1] / s[3], s[2] / s[3])).collect();
    let mut chains: Vec<Chain> = paths
        .iter()
        .enumerate()
        .map(|(i, p)| Chain { pts: p.iter().flat_map(|&v| [r.xyz[v * 3], r.xyz[v * 3 + 1], r.xyz[v * 3 + 2]]).collect(), from: end_node[i * 2], to: end_node[i * 2 + 1] })
        .collect();
    bridge(&mut nodes, &mut chains, min_length);
    loop {
        chains = joined_through(&nodes, chains);
        // A short chain off a node no other reaches, or round from a node to itself, is a
        // spur or a crumb.
        let mut valence = vec![0usize; nodes.len()];
        for c in &chains {
            valence[c.from] += 1;
            valence[c.to] += 1;
        }
        let before = chains.len();
        chains.retain(|c| !(polyline_length(&c.pts) < min_length && (valence[c.from] == 1 || valence[c.to] == 1 || c.from == c.to)));
        if chains.len() == before {
            break;
        }
    }
    // Each chain runs from its node to its node.
    for c in chains.iter_mut() {
        set_ends(&mut c.pts, nodes[c.from], nodes[c.to]);
    }
    Linked { nodes, chains, cut: HashSet::new() }
}

/// A chain's first and last points set to its nodes.
fn set_ends(pts: &mut [f64], from: DVec3, to: DVec3) {
    if pts.len() < 3 {
        return;
    }
    pts[..3].copy_from_slice(&[from.x, from.y, from.z]);
    let l = pts.len();
    pts[l - 3..].copy_from_slice(&[to.x, to.y, to.z]);
}

/// The chains cut at their corners (where they turn CORNER_TURN_DEG or more over
/// CORNER_REACH either side, the sharpest first and no two closer than twice that), a closed
/// chain also where other chains meet it, and one of fewer than three corners into
/// LOOP_PIECES, the longest piece halved each time.
fn split_chains(linked: Linked) -> Linked {
    let mut nodes = linked.nodes.clone();
    let mut cut = linked.cut.clone();
    let mut valence = vec![0usize; nodes.len()];
    for c in &linked.chains {
        valence[c.from] += 1;
        valence[c.to] += 1;
    }
    let mut chains: Vec<Chain> = Vec::new();
    for c in linked.chains {
        let closed = c.from == c.to;
        let n = c.pts.len() / 3;
        let p = |i: usize| DVec3::new(c.pts[i * 3], c.pts[i * 3 + 1], c.pts[i * 3 + 2]);
        let mut s = vec![0.0f64; n];
        for i in 1..n {
            s[i] = s[i - 1] + hypot3(c.pts[i * 3] - c.pts[i * 3 - 3], c.pts[i * 3 + 1] - c.pts[i * 3 - 2], c.pts[i * 3 + 2] - c.pts[i * 3 - 1]);
        }
        let total = if n > 0 { s[n - 1] } else { 0.0 };
        if n < 3 || total <= 0.0 {
            chains.push(c);
            continue;
        }
        // The point at arc length t, round the loop on a closed chain.
        let point_at = |t: f64| -> Option<DVec3> {
            let t = if closed {
                ((t % total) + total) % total
            } else if t < 0.0 || t > total {
                return None;
            } else {
                t
            };
            let mut lo = 0;
            let mut hi = n - 1;
            while hi - lo > 1 {
                let mid = (lo + hi) >> 1;
                if s[mid] <= t {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            let f = if s[hi] > s[lo] { (t - s[lo]) / (s[hi] - s[lo]) } else { 0.0 };
            let (a, b) = (p(lo), p(hi));
            Some(DVec3::new(a.x + f * (b.x - a.x), a.y + f * (b.y - a.y), a.z + f * (b.z - a.z)))
        };
        let mut turn = vec![0.0f64; n];
        let last = if closed { n - 1 } else { n };
        for i in 0..last {
            let (Some(b), Some(f)) = (point_at(s[i] - CORNER_REACH), point_at(s[i] + CORNER_REACH)) else { continue };
            let here = p(i);
            let u = here - b;
            let w = f - here;
            let l = u.length() * w.length();
            if l > 0.0 {
                turn[i] = acos_deg(u.dot(w) / l);
            }
        }
        let apart = |i: usize, j: usize| {
            let d = (s[i] - s[j]).abs();
            if closed {
                d.min(total - d)
            } else {
                d
            }
        };
        let mut cuts: Vec<usize> = Vec::new();
        // A closed chain through a node others reach is cut there.
        if closed && valence[c.from] > 2 {
            cuts.push(0);
        }
        let mut order: Vec<usize> = (0..last).filter(|&i| turn[i] >= CORNER_TURN_DEG).collect();
        order.sort_by(|&a, &b| turn[b].partial_cmp(&turn[a]).unwrap_or(std::cmp::Ordering::Equal));
        for i in order {
            if cuts.iter().all(|&j| apart(i, j) >= 2.0 * CORNER_REACH) {
                cuts.push(i);
            }
        }
        if closed && cuts.len() < 3 {
            if cuts.is_empty() {
                let mut best = 0;
                for i in 1..last {
                    if turn[i] > turn[best] {
                        best = i;
                    }
                }
                cuts.push(best);
            }
            while cuts.len() < LOOP_PIECES {
                cuts.sort_by(|&a, &b| s[a].partial_cmp(&s[b]).unwrap_or(std::cmp::Ordering::Equal));
                let mut gap = -1.0;
                let mut at = 0;
                for k in 0..cuts.len() {
                    let a = s[cuts[k]];
                    let b = if k + 1 < cuts.len() { s[cuts[k + 1]] } else { s[cuts[0]] + total };
                    if b - a > gap {
                        gap = b - a;
                        at = k;
                    }
                }
                let mid = (s[cuts[at]] + gap / 2.0) % total;
                let mut best = 0;
                for i in 0..last {
                    if (s[i] - mid).abs() < (s[best] - mid).abs() {
                        best = i;
                    }
                }
                if cuts.contains(&best) {
                    break;
                }
                cuts.push(best);
            }
        }
        cuts.sort_by(|&a, &b| s[a].partial_cmp(&s[b]).unwrap_or(std::cmp::Ordering::Equal));
        if cuts.is_empty() {
            chains.push(c);
            continue;
        }
        let node_at: Vec<usize> = cuts
            .iter()
            .map(|&i| {
                if closed && i == 0 && valence[c.from] > 2 {
                    return c.from;
                }
                nodes.push(p(i));
                cut.insert(nodes.len() - 1);
                nodes.len() - 1
            })
            .collect();
        // From point i to point j, on round the loop past its end if need be.
        let slice = |i: usize, j: usize| -> Vec<f64> {
            if j > i {
                return c.pts[i * 3..j * 3 + 3].to_vec();
            }
            let mut out = c.pts[(i * 3).min((n - 1) * 3)..(n - 1) * 3].to_vec();
            out.extend_from_slice(&c.pts[..j * 3 + 3]);
            out
        };
        if closed {
            for k in 0..cuts.len() {
                let (i, j) = (cuts[k], cuts[(k + 1) % cuts.len()]);
                chains.push(Chain { pts: slice(i, j), from: node_at[k], to: node_at[(k + 1) % cuts.len()] });
            }
        } else {
            let mut at = vec![0];
            at.extend_from_slice(&cuts);
            at.push(n - 1);
            let mut ids = vec![c.from];
            ids.extend_from_slice(&node_at);
            ids.push(c.to);
            for k in 0..at.len() - 1 {
                let (a, b) = (at[k] * 3, at[k + 1] * 3 + 3);
                chains.push(Chain { pts: if b > a { c.pts[a..b].to_vec() } else { Vec::new() }, from: ids[k], to: ids[k + 1] });
            }
        }
    }
    Linked { nodes, chains, cut }
}

/// The nodes the cuts made merged with any within MERGE_MM; crumbs (a short chain round from
/// a node to itself) dropped; each node where chains ended put where their ends, carried on
/// straight, meet (a path stops short of a corner, or cuts it, by an edge or two); not one a
/// chain was cut at, which lies on it, unless three chains or more meet there; each taken
/// onto the scan; and the chains run to it.
fn settle_nodes(scan: &ScanPaths, linked: Linked, min_length: f64) -> Linked {
    let n = linked.nodes.len();
    let mut used = vec![false; n];
    for c in &linked.chains {
        used[c.from] = true;
        used[c.to] = true;
    }
    // Two nodes a chain longer than the shortest joins stay two: merged, the chain would
    // close on itself.
    let mut long: HashSet<usize> = HashSet::new();
    for c in &linked.chains {
        if polyline_length(&c.pts) >= min_length {
            long.insert(c.from.min(c.to) * n + c.from.max(c.to));
        }
    }
    let ids: Vec<usize> = (0..n).filter(|&i| used[i]).collect();
    let group_of = clusters(&ids.iter().map(|&i| linked.nodes[i]).collect::<Vec<_>>(), &|a, b| long.contains(&(ids[a].min(ids[b]) * n + ids[a].max(ids[b]))));
    let mut position = vec![0usize; n];
    for (k, &id) in ids.iter().enumerate() {
        position[id] = k;
    }
    let find = |i: usize| ids[group_of[position[i]]];
    let mut sums: Vec<[f64; 4]> = vec![[0.0; 4]; n];
    for i in 0..n {
        if !used[i] {
            continue;
        }
        let r = find(i);
        sums[r][0] += linked.nodes[i].x;
        sums[r][1] += linked.nodes[i].y;
        sums[r][2] += linked.nodes[i].z;
        sums[r][3] += 1.0;
    }
    let mut nodes: Vec<DVec3> = sums.iter().enumerate().map(|(i, v)| if v[3] > 0.0 { DVec3::new(v[0] / v[3], v[1] / v[3], v[2] / v[3]) } else { linked.nodes[i] }).collect();
    // A node is a cut's when all it was merged from were.
    let mut ended: HashSet<usize> = HashSet::new();
    for &i in &ids {
        if !linked.cut.contains(&i) {
            ended.insert(find(i));
        }
    }
    let mut chains: Vec<Chain> = linked
        .chains
        .into_iter()
        .map(|c| Chain { from: find(c.from), to: find(c.to), pts: c.pts })
        .filter(|c| !(c.from == c.to && polyline_length(&c.pts) < min_length))
        .collect();
    // The lines the chains come in by: from a point NODE_FAR along each to one NODE_NEAR from
    // its end.
    let mut lines: Vec<(usize, Vec<(DVec3, DVec3)>)> = Vec::new();
    let mut line_of: HashMap<usize, usize> = HashMap::new();
    for c in &chains {
        let length = polyline_length(&c.pts);
        for end in [0, 1] {
            let pts = if end == 0 { c.pts.clone() } else { reversed(&c.pts) };
            let far = point_along(&pts, NODE_FAR.min(0.5 * length));
            let near = point_along(&pts, NODE_NEAR.min(0.15 * length));
            let u = near - far;
            let l = u.length();
            if !(l > 0.0) {
                continue;
            }
            let node = if end == 0 { c.from } else { c.to };
            let k = *line_of.entry(node).or_insert_with(|| {
                lines.push((node, Vec::new()));
                lines.len() - 1
            });
            lines[k].1.push((far, u / l));
        }
    }
    // Least squares: the point nearest the lines, held lightly to the mean of the ends, which
    // is where it stays along lines that run on straight.
    for (node, list) in &lines {
        if list.len() < 2 || (list.len() < 3 && !ended.contains(node)) {
            continue;
        }
        let x0 = nodes[*node];
        let mut m = [0.0f64; 9];
        let mut b = [0.0f64; 3];
        let hold = NODE_HOLD * list.len() as f64;
        for i in 0..3 {
            m[i * 4] += hold;
            b[i] += hold * x0[i];
        }
        for (p, u) in list {
            for i in 0..3 {
                for j in 0..3 {
                    let a = if i == j { 1.0 } else { 0.0 } - u[i] * u[j];
                    m[i * 3 + j] += a;
                    b[i] += a * p[j];
                }
            }
        }
        if let Some(x) = solve3(&m, &b) {
            if (x - x0).length() < MERGE_MM {
                nodes[*node] = x;
            }
        }
    }
    for (node, _) in &lines {
        if let Some(c) = scan.closest(nodes[*node]) {
            nodes[*node] = c.point;
        }
    }
    for c in chains.iter_mut() {
        set_ends(&mut c.pts, nodes[c.from], nodes[c.to]);
    }
    Linked { nodes, chains, cut: HashSet::new() }
}

/// The point at arc length t along a polyline, held to its ends.
fn point_along(pts: &[f64], t: f64) -> DVec3 {
    let mut left = t;
    let mut i = 3;
    while i + 2 < pts.len() {
        let l = hypot3(pts[i] - pts[i - 3], pts[i + 1] - pts[i - 2], pts[i + 2] - pts[i - 1]);
        if l >= left && l > 0.0 {
            let f = left / l;
            return DVec3::new(pts[i - 3] + f * (pts[i] - pts[i - 3]), pts[i - 2] + f * (pts[i + 1] - pts[i - 2]), pts[i - 1] + f * (pts[i + 2] - pts[i - 1]));
        }
        left -= l;
        i += 3;
    }
    let l = pts.len();
    if l < 3 {
        return DVec3::NAN;
    }
    DVec3::new(pts[l - 3], pts[l - 2], pts[l - 1])
}

/// The 3 × 3 system m x = b, row-major, or `None` when singular.
fn solve3(m: &[f64; 9], b: &[f64; 3]) -> Option<DVec3> {
    let det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
    if !(det.abs() > 1e-12) {
        return None;
    }
    let x = (b[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (b[1] * m[8] - m[5] * b[2]) + m[2] * (b[1] * m[7] - m[4] * b[2])) / det;
    let y = (m[0] * (b[1] * m[8] - m[5] * b[2]) - b[0] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * b[2] - b[1] * m[6])) / det;
    let z = (m[0] * (m[4] * b[2] - b[1] * m[7]) - m[1] * (m[3] * b[2] - b[1] * m[6]) + b[0] * (m[3] * m[7] - m[4] * m[6])) / det;
    Some(DVec3::new(x, y, z))
}

// ---------------------------------------------------------------------------------------
// The curves.

/// A chain fitted: its nodes, the fitted curve's samples and their arc length, and its
/// angle.
struct Fitted {
    from: usize,
    to: usize,
    line: Vec<f64>,
    s: Vec<f64>,
    angle: f64,
}

/// Each chain fitted (PN2), its grips, length and angle (a chain whose angle across comes to
/// less than `weak` is no crease, and is left out, and so is a group of curves joined at
/// their nodes that all together is shorter than the shortest chain: a speck, a dent or a
/// chip or a burst of noise, however its corners cut it up) and the nodes the curves reach,
/// numbered from 1.
fn curves_of(scan: &ScanPaths, linked: Linked, weak: f64, min_length: f64) -> CreaseSuggestion {
    let mut fitted: Vec<Fitted> = Vec::new();
    for c in &linked.chains {
        let count = c.pts.len() / 3;
        if count < 2 {
            continue;
        }
        let trace = CurvePoints::from_polyline(c.pts.iter().map(|&x| x as f32).collect());
        if !(trace.length > 0.0) {
            continue;
        }
        let fit = fit_curve(trace.as_points(), auto_spans(trace.as_points()), &CurveEnds::default(), 0.0);
        let samples = ((trace.length / (SAMPLE_EDGES * scan.edge())).ceil() + 1.0).min(MAX_SAMPLES as f64).max(8.0);
        let samples = if samples.is_finite() { samples as usize } else { 8 };
        let line = sample_curve(&fit, samples);
        let mut s = vec![0.0f64; samples];
        for i in 1..samples {
            s[i] = s[i - 1] + hypot3(line[i * 3] - line[i * 3 - 3], line[i * 3 + 1] - line[i * 3 - 2], line[i * 3 + 2] - line[i * 3 - 1]);
        }
        let angle = angle_along(scan, &line, &s);
        if angle >= weak {
            fitted.push(Fitted { from: c.from, to: c.to, line, s, angle });
        }
    }
    let mut group: Vec<usize> = (0..linked.nodes.len()).collect();
    fn root(group: &mut [usize], mut i: usize) -> usize {
        while group[i] != i {
            group[i] = group[group[i]];
            i = group[i];
        }
        i
    }
    for f in &fitted {
        let (a, b) = (root(&mut group, f.from), root(&mut group, f.to));
        group[a] = b;
    }
    let mut total = vec![0.0f64; linked.nodes.len()];
    for f in &fitted {
        let r = root(&mut group, f.from);
        total[r] += f.s[f.s.len() - 1];
    }
    let mut ids: HashMap<usize, u32> = HashMap::new();
    let mut nodes: Vec<CreaseNode> = Vec::new();
    let mut id_of = |n: usize, nodes: &mut Vec<CreaseNode>| -> u32 {
        *ids.entry(n).or_insert_with(|| {
            let id = nodes.len() as u32 + 1;
            nodes.push(CreaseNode { id, at: linked.nodes[n] });
            id
        })
    };
    let mut curves: Vec<SuggestedCrease> = Vec::new();
    for f in &fitted {
        let r = root(&mut group, f.from);
        if total[r] < min_length {
            continue;
        }
        let via = grips_of(&f.line, &f.s, GRIP_SAG_EDGES * scan.edge())
            .into_iter()
            .map(|i| {
                let p = DVec3::new(f.line[i * 3], f.line[i * 3 + 1], f.line[i * 3 + 2]);
                scan.closest(p).map(|c| c.point).unwrap_or(p)
            })
            .collect();
        let from = id_of(f.from, &mut nodes);
        let to = id_of(f.to, &mut nodes);
        curves.push(SuggestedCrease { from, to, via, length: f.s[f.s.len() - 1], angle: f.angle, points: f.line.iter().map(|&x| x as f32).collect() });
    }
    CreaseSuggestion { nodes, curves }
}

/// The samples to take as grips, the ends left out: each as far on from the last as the
/// chord between them stays within `sag` of the curve, and GRIP_SPACING at most.
fn grips_of(line: &[f64], s: &[f64], sag: f64) -> Vec<usize> {
    let n = s.len();
    let mut out = Vec::new();
    let mut i = 0;
    while i + 1 < n {
        let mut j = i + 1;
        while j + 1 < n && s[j + 1] - s[i] <= GRIP_SPACING && chord_holds(line, i, j + 1, sag) {
            j += 1;
        }
        if j >= n - 1 {
            break;
        }
        out.push(j);
        i = j;
    }
    out
}

/// Whether every sample between i and j lies within `sag` of the chord.
fn chord_holds(line: &[f64], i: usize, j: usize, sag: f64) -> bool {
    let (ax, ay, az) = (line[i * 3], line[i * 3 + 1], line[i * 3 + 2]);
    let (dx, dy, dz) = (line[j * 3] - ax, line[j * 3 + 1] - ay, line[j * 3 + 2] - az);
    let dd = dx * dx + dy * dy + dz * dz;
    for k in i + 1..j {
        let (px, py, pz) = (line[k * 3] - ax, line[k * 3 + 1] - ay, line[k * 3 + 2] - az);
        let t = if dd > 0.0 { ((px * dx + py * dy + pz * dz) / dd).clamp(0.0, 1.0) } else { 0.0 };
        if hypot3(px - t * dx, py - t * dy, pz - t * dz) > sag {
            return false;
        }
    }
    true
}

/// The mean angle across the crease, deg: at every ANGLE_STEP along it, the scan's normal a
/// few edges off it either side.
fn angle_along(scan: &ScanPaths, line: &[f64], s: &[f64]) -> f64 {
    let n = s.len();
    let total = s[n - 1];
    let off = ANGLE_OFFSET_EDGES * scan.edge();
    let radius = ANGLE_RADIUS_EDGES * scan.edge();
    let mut sum = 0.0;
    let mut count = 0usize;
    // Not at the very ends, where other creases meet it; a short curve is read in its middle
    // half.
    let margin = off.min(total / 4.0);
    let steps = (total / ANGLE_STEP).round().max(2.0);
    let steps = if steps.is_finite() { steps as usize } else { 2 };
    let mut k = 0;
    for step in 0..=steps {
        let want = (total * step as f64) / steps as f64;
        if want < margin || want > total - margin {
            continue;
        }
        while k + 1 < n && s[k + 1] <= want {
            k += 1;
        }
        let a = k.saturating_sub(1);
        let b = (k + 1).min(n - 1);
        let p = DVec3::new(line[k * 3], line[k * 3 + 1], line[k * 3 + 2]);
        let t = DVec3::new(line[b * 3] - line[a * 3], line[b * 3 + 1] - line[a * 3 + 1], line[b * 3 + 2] - line[a * 3 + 2]);
        let Some(c) = scan.closest(p) else { continue };
        let normal = scan.normal_at(c.point, c.triangle);
        let across = cross(t, normal);
        let al = across.length();
        if !(al > 0.0) {
            continue;
        }
        let one = normal_near(scan, DVec3::new(p.x + (across.x / al) * off, p.y + (across.y / al) * off, p.z + (across.z / al) * off), radius);
        let two = normal_near(scan, DVec3::new(p.x - (across.x / al) * off, p.y - (across.y / al) * off, p.z - (across.z / al) * off), radius);
        let (Some(one), Some(two)) = (one, two) else { continue };
        sum += acos_deg(one.x * two.x + one.y * two.y + one.z * two.z);
        count += 1;
    }
    if count > 0 {
        sum / count as f64
    } else {
        0.0
    }
}

/// The scan's normal at its nearest point to q: its triangles' within `radius` of that
/// point, by area.
fn normal_near(scan: &ScanPaths, q: DVec3, radius: f64) -> Option<DVec3> {
    let c = scan.closest(q)?;
    let p = c.point;
    let mut sum = DVec3::ZERO;
    let mut near = DVec3::ZERO;
    let rr = DVec3::splat(radius);
    for t in scan.triangles_in(p - rr, p + rr) {
        if scan.closest_on_triangle(t, p, &mut near) > radius {
            continue;
        }
        sum += scan.triangle_cross(t);
    }
    let l = sum.length();
    Some(if l > 0.0 { sum / l } else { c.normal })
}

fn cross(a: DVec3, b: DVec3) -> DVec3 {
    DVec3::new(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)
}

/// Ported from ScanRuler's `plugins/reverse/tests/reNetworkCreases.test.ts`: a box with
/// sharp rims and rounded upright edges gives its two rims and nothing on the rounds (rounds
/// of 4 mm on edges of 0.5 mm, and of 2 mm, only four edges across) and the scan path through
/// each curve's grips keeps to its rim; a box with every edge sharp gives twelve curves
/// meeting in its eight corners; a fold across a noisy sheet, which its triangles cross
/// slantwise, comes out as one curve from edge to edge and nothing else, and the sheet
/// without the fold as nothing; and a box of 320,000 triangles in a 2-million-triangle scan
/// is read in well under a second.
#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;
    use crate::network_curve::trace_curve;
    use crate::network_meshes::{grid_scan, GridOptions};

    /// A piece of a rounded rectangle's outline: its length and its point a share t along it.
    struct Piece {
        length: f64,
        at: Box<dyn Fn(f64) -> [f64; 2]>,
    }

    /// A rectangle of half-sides a, b with corners rounded to r, in the plane: its point at a
    /// share u of the way round, counter-clockwise from the middle of its +x side, and the
    /// signed distance of (x, y) from it.
    struct RoundedRect {
        a: f64,
        b: f64,
        r: f64,
        pieces: Vec<Piece>,
        perimeter: f64,
    }

    impl RoundedRect {
        fn new(a: f64, b: f64, r: f64) -> Self {
            let line = |p: [f64; 2], q: [f64; 2]| Piece { length: (q[0] - p[0]).hypot(q[1] - p[1]), at: Box::new(move |t| [p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]) };
            let arc = |c: [f64; 2], from: f64| Piece { length: (PI * r) / 2.0, at: Box::new(move |t| [c[0] + r * (from + (t * PI) / 2.0).cos(), c[1] + r * (from + (t * PI) / 2.0).sin()]) };
            let pieces = vec![
                line([a, 0.0], [a, b - r]),
                arc([a - r, b - r], 0.0),
                line([a - r, b], [-a + r, b]),
                arc([-a + r, b - r], PI / 2.0),
                line([-a, b - r], [-a, -b + r]),
                arc([-a + r, -b + r], PI),
                line([-a + r, -b], [a - r, -b]),
                arc([a - r, -b + r], (3.0 * PI) / 2.0),
                line([a, -b + r], [a, 0.0]),
            ];
            let perimeter = pieces.iter().map(|p| p.length).sum();
            RoundedRect { a, b, r, pieces, perimeter }
        }

        fn at(&self, u: f64) -> [f64; 2] {
            let mut d = (((u % 1.0) + 1.0) % 1.0) * self.perimeter;
            for p in &self.pieces {
                if d <= p.length {
                    return (p.at)(if p.length > 0.0 { d / p.length } else { 0.0 });
                }
                d -= p.length;
            }
            (self.pieces[self.pieces.len() - 1].at)(1.0)
        }

        fn signed(&self, x: f64, y: f64) -> f64 {
            let qx = x.abs() - (self.a - self.r);
            let qy = y.abs() - (self.b - self.r);
            qx.max(0.0).hypot(qy.max(0.0)) + qx.max(qy).min(0.0) - self.r
        }
    }

    /// A 40 × 30 × 20 mm box scanned as one grid round it.
    struct RoundedBox {
        scan: ScanPaths,
        rect: RoundedRect,
        height: f64,
    }

    impl RoundedBox {
        /// How far a point is from the nearer rim, the edge where the top or the bottom meets
        /// the sides.
        fn off_rim(&self, p: &[f32], i: usize) -> f64 {
            let (x, y, z) = (p[i] as f64, p[i + 1] as f64, p[i + 2] as f64);
            self.rect.signed(x, y).hypot(z.abs().min((z - self.height).abs()))
        }
    }

    /// A ring of its bottom, its four sides, a ring of its top: the rims between them sharp,
    /// the upright edges rounded to r, with edges about h long and noise.
    fn rounded_box(r: f64, h: f64, noise: f64) -> RoundedBox {
        let height = 20.0;
        let rect = RoundedRect::new(20.0, 15.0, r);
        let ring = (6.0 / h).round();
        let wall = (height / h).round();
        let rows = 2.0 * ring + wall;
        // The rings run from 0.6 of the way out to the rim.
        let inner = 0.6;
        let at = |u: f64, v: f64| -> DVec3 {
            let j = (v * rows).max(0.0).min(rows);
            let [x, y] = rect.at(u);
            if j <= ring {
                let s = inner + ((1.0 - inner) * j) / ring;
                return DVec3::new(s * x, s * y, 0.0);
            }
            if j <= ring + wall {
                return DVec3::new(x, y, (height * (j - ring)) / wall);
            }
            let s = 1.0 - ((1.0 - inner) * (j - ring - wall)) / ring;
            DVec3::new(s * x, s * y, height)
        };
        let mesh = grid_scan((rect.perimeter / h).round() as usize, rows as usize, &at, &GridOptions { wrap: true, noise, ..Default::default() });
        RoundedBox { scan: ScanPaths::new(mesh.positions, mesh.indices), rect, height }
    }

    /// A 60 × 40 mm sheet on edges of 0.5 mm folded up by `deg` along the line from (12, 0)
    /// to (42, 40), slantwise to its triangles, with noise; and how far a point is from the
    /// fold, in plan.
    struct FoldedSheet {
        scan: ScanPaths,
        a: [f64; 2],
        b: [f64; 2],
        across: [f64; 2],
    }

    impl FoldedSheet {
        fn off_fold(&self, p: &[f32], i: usize) -> f64 {
            ((p[i] as f64 - self.a[0]) * self.across[0] + (p[i + 1] as f64 - self.a[1]) * self.across[1]).abs()
        }
    }

    fn folded_sheet(deg: f64, noise: f64) -> FoldedSheet {
        let a = [12.0f64, 0.0];
        let b = [42.0f64, 40.0];
        let l = (b[0] - a[0]).hypot(b[1] - a[1]);
        let across = [(b[1] - a[1]) / l, -(b[0] - a[0]) / l];
        let off = |x: f64, y: f64| (x - a[0]) * across[0] + (y - a[1]) * across[1];
        let slope = ((deg * PI) / 180.0).tan();
        let mesh = grid_scan(120, 80, &|u, v| DVec3::new(60.0 * u, 40.0 * v, slope * off(60.0 * u, 40.0 * v).max(0.0)), &GridOptions { noise, ..Default::default() });
        FoldedSheet { scan: ScanPaths::new(mesh.positions, mesh.indices), a, b, across }
    }

    /// The scan path through a suggested curve's nodes and grips, as the network traces a
    /// curve.
    fn traced_points(scan: &ScanPaths, s: &CreaseSuggestion, c: &SuggestedCrease) -> Vec<f32> {
        let at: HashMap<u32, DVec3> = s.nodes.iter().map(|n| (n.id, n.at)).collect();
        let mut through = vec![at[&c.from]];
        through.extend_from_slice(&c.via);
        through.push(at[&c.to]);
        trace_curve(scan, &through).points
    }

    fn worst(points: &[f32], off: impl Fn(&[f32], usize) -> f64) -> f64 {
        let mut w = 0.0f64;
        let mut i = 0;
        while i + 2 < points.len() {
            w = w.max(off(points, i));
            i += 3;
        }
        w
    }

    fn valence(s: &CreaseSuggestion, id: u32) -> usize {
        s.curves.iter().filter(|c| c.from == id || c.to == id).count()
    }

    #[test]
    fn gives_a_rounded_box_its_two_sharp_rims_and_nothing_on_the_rounds() {
        for (r, h) in [(4.0, 0.5), (2.0, 0.5)] {
            let bx = rounded_box(r, h, 0.01);
            let s = suggest_creases(&bx.scan, &CreaseOptions::default());
            // Each rim in four, cut at the middles of its rounded corners: eight curves round
            // eight nodes, two curves at each.
            assert_eq!(s.curves.len(), 8, "r {r}");
            assert_eq!(s.nodes.len(), 8, "r {r}");
            for n in &s.nodes {
                assert_eq!(valence(&s, n.id), 2);
            }
            for c in &s.curves {
                // On a rim all along (none runs down a round) and square.
                assert!(worst(&c.points, |p, i| bx.off_rim(p, i)) < 0.1);
                assert!(c.angle > 89.0 && c.angle < 91.0, "angle {}", c.angle);
                // The grips on the scan, and the scan path through them on the rim round its
                // rounded corners too.
                for &g in &c.via {
                    assert!(bx.scan.closest(g).expect("on the scan").distance < 1e-3);
                }
                assert!(worst(&traced_points(&bx.scan, &s, c), |p, i| bx.off_rim(p, i)) < 0.1);
            }
            // Each rim whole, round its perimeter.
            for z in [0.0, bx.height] {
                let round: f64 = s.curves.iter().filter(|c| (c.points[2] as f64 - z).abs() < 1.0).map(|c| c.length).sum();
                assert!((round - bx.rect.perimeter).abs() / bx.rect.perimeter < 0.01, "rim at {z}: {round}");
            }
        }
    }

    #[test]
    fn gives_a_sharp_box_its_twelve_edges_meeting_in_its_eight_corners() {
        let bx = rounded_box(0.0, 0.5, 0.01);
        let s = suggest_creases(&bx.scan, &CreaseOptions::default());
        assert_eq!(s.curves.len(), 12);
        assert_eq!(s.nodes.len(), 8);
        for n in &s.nodes {
            let corner = DVec3::new(n.at.x.abs() - 20.0, n.at.y.abs() - 15.0, n.at.z.abs().min((n.at.z - bx.height).abs())).length();
            assert!(corner < 0.05, "{corner}");
            assert_eq!(valence(&s, n.id), 3);
        }
        // Four 40 mm edges, four of 30 and four of 20, each square.
        let mut lengths: Vec<i64> = s.curves.iter().map(|c| c.length.round() as i64).collect();
        lengths.sort_unstable();
        assert_eq!(lengths, vec![20, 20, 20, 20, 30, 30, 30, 30, 40, 40, 40, 40]);
        for c in &s.curves {
            assert!((c.angle - 90.0).abs() < 1.0);
        }
    }

    #[test]
    fn finds_a_fold_across_a_noisy_sheet_as_one_curve_from_edge_to_edge_and_nothing_else() {
        for deg in [30.0, 35.0, 40.0] {
            let sheet = folded_sheet(deg, 0.02);
            let s = suggest_creases(&sheet.scan, &CreaseOptions::default());
            assert_eq!(s.curves.len(), 1, "{deg}°");
            assert_eq!(s.nodes.len(), 2, "{deg}°");
            let c = &s.curves[0];
            assert!(worst(&c.points, |p, i| sheet.off_fold(p, i)) < 0.25);
            assert!(worst(&traced_points(&sheet.scan, &s, c), |p, i| sheet.off_fold(p, i)) < 0.25);
            assert!((c.angle - deg).abs() < 1.0, "angle {}", c.angle);
            // Its ends at the sheet's edges, where the fold leaves it.
            for n in &s.nodes {
                let d = (n.at.x - sheet.a[0]).hypot(n.at.y - sheet.a[1]).min((n.at.x - sheet.b[0]).hypot(n.at.y - sheet.b[1]));
                assert!(d < 2.0, "{deg}°: end {d} from the edge");
            }
            assert!(c.length > 46.0, "{}", c.length);
        }
        // The same sheet, noisier, and not folded: nothing.
        assert_eq!(suggest_creases(&folded_sheet(0.0, 0.05).scan, &CreaseOptions::default()).curves, vec![]);
    }

    #[test]
    fn reads_320000_triangles_in_a_box_of_a_2_million_triangle_scan_in_well_under_a_second() {
        // A corrugated sheet 300 mm square on edges of 0.3 to 0.4 mm: folds of 35° every
        // 12.5 mm, slantwise to the triangles.
        let period = 25.0;
        let slope = ((17.5 * PI) / 180.0).tan();
        let dir = [((20.0 * PI) / 180.0).cos(), ((20.0 * PI) / 180.0).sin()];
        let along = move |x: f64, y: f64| x * dir[0] + y * dir[1];
        let mesh = grid_scan(
            1000,
            1000,
            &|u, v| {
                let m = (((along(300.0 * u, 300.0 * v) % period) + period) % period) - period / 2.0;
                DVec3::new(300.0 * u, 300.0 * v, slope * (period / 2.0 - m.abs()))
            },
            &GridOptions { noise: 0.02, ..Default::default() },
        );
        let scan = ScanPaths::new(mesh.positions, mesh.indices);
        assert_eq!(scan.triangle_count(), 2_000_000);
        let opts = CreaseOptions { bounds: Some(CreaseBox { min: DVec3::new(90.0, 90.0, -10.0), max: DVec3::new(210.0, 210.0, 10.0) }), ..Default::default() };
        suggest_creases(&scan, &opts);
        let t0 = Instant::now();
        let s = suggest_creases(&scan, &opts);
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        // The twelve folds that cross the box, each whole, each one curve.
        assert_eq!(s.curves.len(), 12);
        let off_fold = |p: &[f32], i: usize| {
            let d = along(p[i] as f64, p[i + 1] as f64);
            (d - (period / 2.0) * (d / (period / 2.0)).round()).abs()
        };
        for c in &s.curves {
            assert!((c.angle - 35.0).abs() < 1.0, "angle {}", c.angle);
            assert!(worst(&c.points, off_fold) < 0.3);
        }
        assert!(ms < 1500.0, "{ms} ms");
    }
}
