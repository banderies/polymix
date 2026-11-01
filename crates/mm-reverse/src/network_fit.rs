// SPDX-License-Identifier: AGPL-3.0-only
//! A patch network's patches fitted to the scan together.
//!
//! Ported from ScanRuler (`plugins/reverse/core/networkFit.ts`), © Stefan Hermann
//! (CNC Kitchen), AGPL-3.0-only. Its input types (`PatchInput`, `CurveInput`, `BodySide`,
//! `NetworkFitInput`) live in [`crate::network`], which hands them to [`fit_network`].
//!
//! Each four-sided cell is a cubic B-spline patch over the unit square its map put the cell's
//! points on ([`crate::network_map`]). Its four boundary rows of poles are its sides' curves,
//! fitted first ([`crate::network_curve`]) and faired by the same smoothness as the patches:
//! a side that is one curve takes that curve's poles, so two neighbours share the poles of
//! the curve between them and meet on it exactly. For that their knots along it must be the
//! same, and so must a patch's on its opposite sides: the sides tied by either rule —
//! opposite in a cell, the same curve in two cells — are put together by union-find into
//! strips that run across the network, each with one span count, Auto's or the operator's.
//! Auto starts every strip at four and grows it only where a patch on it misses its points by
//! more than a tolerance beyond their noise ([`fit_network`]).
//!
//! The inner poles of all the patches are the unknowns of one system, three coordinates
//! each, solved by PN0's conjugate gradient ([`crate::sparse`]). Its terms:
//!
//! - the data: each scan point of a cell at its (u, v) on the patch;
//! - the fairing: Fit Surface's term — third divided differences of the net along u, along v
//!   and mixed (see Fit Surface for why third and not the thin plate's second), weighted the
//!   same way for the patch's size, its slider `smoothness`;
//! - the pull: net cells with no point within a cell of them (a scan gap) held near the Coons
//!   surface of the patch's four sides, where Fit Surface holds them near its plane;
//! - tangency across every shared curve not marked a crease: with B a pole of the curve and A
//!   and C the first inner poles either side of it, (B − A) − λ (C − B) = 0, λ the ratio of
//!   the two cross-spacings, read off a first fit without it. Along the curve it is held
//!   exactly — A and C are written as B − d and B + λ d, one unknown d for the two poles —
//!   and at the two poles next to each corner, which two curves' rows share, as a heavy
//!   weight. At a corner where three, five or six patches meet it is dropped there: G1 cannot
//!   be held at such a corner by its poles (the vertex enclosure problem); the angle there is
//!   reported instead. Nor can it at a corner of four whose ties do not agree round it —
//!   cells drawn askew, a fine scan's noise in the ratios: held by weight, they folded the
//!   patches over themselves next to the corner, on a Joy-Con ten patches of seventeen, a
//!   point 2.7 mm off. So the ties next to each corner are tried, and let go where a patch
//!   there turned more than CORNER_TURN from the first fit's, which no tie bends; the fit is
//!   solved again without them, and such a corner is reported as one of three, five or six
//!   is. A T-junction's curve — one patch's whole side, a piece of another's — shares no poles
//!   with the longer side, and is held tangent by weight, pole by pole, as a body's face holds
//!   a patch.
//!
//! The data's (u, v) come from the map; after each solve every point is projected onto its
//! patch (Gauss–Newton from where it was) and the fit solved again from the last — three
//! rounds, or fewer once the RMS moves less than 1 %. A robust fit leaves out the points
//! beyond three robust sigmas of the first pass.
//!
//! ScanRuler's `Math.hypot` is a square root of a sum of squares here, and its `**` is
//! `powf` (a square, `x * x`); the eigenvector of the tangent plane comes from
//! [`crate::linalg::symmetric_eigen_n`] at n = 3, the same Jacobi sweeps as ScanRuler's
//! `symmetricEigen3`.

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet};

use glam::DVec3;
use serde::{Deserialize, Serialize};

use crate::fit2d::FitError;
use crate::linalg::symmetric_eigen_n;
use crate::network::{BodySide, CurveInput, NetworkBodies, NetworkFitInput, NetworkReading, PatchInput};
use crate::network_cells::{fixed1, CellSide, HalfEdge};
use crate::network_curve::{auto_spans, curve_samples, end_tangent, fit_curve, join_traces, leaving_direction, reversed_poles, CurveEnds, CurveFit, CurvePoints, PointsRef, DEFAULT_TOLERANCE, SPANS_MAX, SPANS_MIN};
use crate::network_feature::{NetworkFeature, SideMode};
use crate::network_spline::{basis_at, curve_derivative, curve_point, divided_weights, greville_of, patch_normal, patch_point, PatchNet, DEGREE};
use crate::scan_path::ScanPaths;
use crate::sparse::{conjugate_gradient, CgOptions, LinearOperator};

/// What a slider of 1 means against the data: Fit Surface's own scale.
const SMOOTHNESS_SCALE: f64 = 1e-2;
/// The pull toward the Coons surface in a net cell with no points within PULL_REACH cells:
/// its weight as a share of an average held cell's points, and the samples a side standing
/// in for points there.
const PULL: f64 = 0.01;
const PULL_REACH: usize = 1;
const PULL_SAMPLES: usize = 3;
/// The weight of the tangency held at the poles next to a corner, as a multiple of the
/// data's mean diagonal: heavy enough that the angle it leaves is a hundredth of a degree,
/// light enough to leave the conjugate gradient its few extra steps.
const TIE_WEIGHT: f64 = 1e4;
/// How far a patch next to a corner may turn from the first fit's — free of the ties, and
/// never folded — before the ties there are taken to fight each other round the corner and
/// are let go, deg. Ties that agree turn it a few degrees at most.
const CORNER_TURN: f64 = 30.0;
/// The cross-spacings' ratio λ is held within this factor either way.
const RATIO_LIMIT: f64 = 5.0;
/// The most points a patch is fitted through, per net cell: past it the points are taken
/// evenly — fifty a cell resolve the patch's noise within a hundredth of its RMS and the fit
/// takes a tenth of the time.
const POINTS_PER_CELL: usize = 50;
/// Parameter correction: the most rounds, and the RMS's change that ends them sooner; the
/// Gauss–Newton steps each point takes a round.
const ROUNDS: usize = 3;
const ROUNDS_SETTLE: f64 = 0.01;
const NEWTON_STEPS: usize = 3;
/// Robust: points past this many robust sigmas of the first pass are left out, never closer
/// than ROBUST_FLOOR mm.
const ROBUST_SIGMAS: f64 = 3.0;
const ROBUST_FLOOR: f64 = 0.01;
/// Auto's span count is held to a span every this many of the scan's edges along a side —
/// some ten points a net cell. The noise rule alone sizes the net by what the points
/// resolve, not by how many there are: on a scan with a 1.1 mm edge it gave 20 spans over a
/// 20 mm side, a thousand points for four hundred net cells, and the fairing then shaped the
/// patch more than they did — 0.04–0.06 mm RMS at 0.02 mm noise.
const EDGES_PER_SPAN: f64 = 3.0;
/// Auto: the spans a strip starts at, how much a round grows one (4, 6, 9, 14, 20), and the
/// most rounds, a round that tried and kept nothing among them. A patch over the tolerance
/// grows its strips whose spans are at least AUTO_ALIKE of its longer one's; a growth is kept
/// where it took a patch over the tolerance to AUTO_HELPS of where it was, or below.
const AUTO_START: usize = 4;
const AUTO_GROWTH: f64 = 1.5;
const AUTO_ROUNDS: usize = 8;
const AUTO_ALIKE: f64 = 0.75;
const AUTO_HELPS: f64 = 0.8;
/// The points a cell of a patch holds on the average where its misfit is read
/// ([`misfit_of`]).
const MISFIT_POINTS: f64 = 8.0;
/// The conjugate gradient's tolerance.
const SOLVE_TOLERANCE: f64 = 1e-8;
/// How many points along a shared curve its angle and gap are read at.
const EDGE_SAMPLES: usize = 24;
/// How many worst points a patch's answer names.
const WORST_POINTS: usize = 3;
/// A pole's band: its 7 × 7 neighbours in the net, the most a cubic's terms reach.
const BAND: usize = 49;
/// The conjugate gradient's most iterations a solve.
const SOLVE_MAX_ITERATIONS: usize = 20000;
/// The samples a side a curve's samples are taken at to fit a T-junction's side to
/// (ScanRuler's `curveSamples` default).
const SIDE_SAMPLES_PER_SPAN: usize = 8;

// ---------------------------------------------------------------------------------------
// The answer.

/// A patch's side as fitted: the half-edges along it and its row of poles as it runs.
#[derive(Debug, Clone, PartialEq)]
pub struct FittedSide {
    pub edges: Vec<HalfEdge>,
    /// The side's row of poles, from its start, x, y, z each.
    pub poles: Vec<f64>,
    /// The side is one curve, whose poles it shares.
    pub single: bool,
}

/// One of the points a patch misses most.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WorstPoint {
    pub at: DVec3,
    /// Its distance from the patch, signed along the patch's normal, mm.
    pub off: f64,
}

/// A fitted patch, the curves of its sides, and how it sits on its points.
#[derive(Debug, Clone, PartialEq)]
pub struct FittedPatch {
    pub net: PatchNet,
    /// Per side, the half-edges along it and its row of poles as it runs.
    pub sides: Vec<FittedSide>,
    pub corners: Vec<u32>,
    pub seed: DVec3,
    pub points: usize,
    pub rms: f64,
    pub max: f64,
    /// How far the patch's shape strays from its points beyond their scatter, RMS, mm: its
    /// residuals' RMS with their scatter taken out — 0 where that is within what the scatter
    /// alone reads (see `misfit_of`).
    pub misfit: f64,
    pub worst: Vec<WorstPoint>,
    /// A robust fit's points left out.
    pub dropped: Option<usize>,
}

/// A curve two patches share.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SharedCurve {
    pub curve: u32,
    pub patches: [usize; 2],
    /// How far apart the two patches' edges lie on it, mm: zero where they share its poles;
    /// a T-junction's side, made of more than one curve, does not.
    pub gap: f64,
    /// The largest angle between the two patches' normals along it, deg.
    pub angle: f64,
    pub crease: bool,
    /// The curve is the whole side of the first patch and a piece of the second's: a
    /// T-junction's, whose two edges do not share poles — held tangent by weight, not
    /// exactly.
    pub tee: bool,
}

/// A corner where three, five or six patches meet, or one of four whose ties were let go:
/// the largest angle between two of them across a curve near it, deg — within the two spans
/// of it where the tie is let go (see [`fit_network`]).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct IrregularCorner {
    pub node: u32,
    pub at: DVec3,
    pub patches: usize,
    pub angle: f64,
}

/// A strip: its span count, whether Auto chose it, and its curves.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Strip {
    pub spans: usize,
    pub auto: bool,
    pub curves: Vec<u32>,
}

/// A patch side beside a body's face: the largest angle between the patch and the face along
/// it (deg) and how far the curve stands off the body's edge points it was fitted to (mm).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BodyReading {
    pub curve: u32,
    pub patch: usize,
    pub mode: SideMode,
    pub angle: f64,
    pub gap: f64,
}

/// The network's patches fitted together.
#[derive(Debug, Clone, PartialEq)]
pub struct NetworkFitResult {
    pub patches: Vec<FittedPatch>,
    /// Every curve as fitted at its strip's span count, by id.
    pub curves: BTreeMap<u32, CurveFit>,
    pub shared: Vec<SharedCurve>,
    pub corners: Vec<IrregularCorner>,
    /// Each strip: its span count, whether Auto chose it, and its curves.
    pub strips: Vec<Strip>,
    /// Each patch side beside a body's face (see [`BodyReading`]).
    pub bodies: Vec<BodyReading>,
    pub rounds: usize,
    pub iterations: usize,
}

// ---------------------------------------------------------------------------------------
// Small helpers.

/// A number JavaScript takes as true: present, not zero and not NaN (`input.spacing ? …`).
fn truthy(x: Option<f64>) -> Option<f64> {
    x.filter(|s| *s != 0.0 && !s.is_nan())
}

/// JavaScript's `Math.round`: the nearest whole number, a half rounded up.
fn js_round(x: f64) -> f64 {
    let f = x.floor();
    if x - f >= 0.5 {
        f + 1.0
    } else {
        f
    }
}

fn hypot3(x: f64, y: f64, z: f64) -> f64 {
    (x * x + y * y + z * z).sqrt()
}

fn dot(a: DVec3, b: DVec3) -> f64 {
    a.x * b.x + a.y * b.y + a.z * b.z
}

/// A curve's own points: a body edge's where it runs along one, else its trace.
fn own_points(c: &CurveInput) -> PointsRef<'_> {
    match &c.exact {
        Some(e) => e.as_points(),
        None => c.trace.as_points(),
    }
}

/// The span count Auto holds a curve or a strip to by the scan's spacing: a span every
/// EDGES_PER_SPAN edges, at least SPANS_MIN.
fn dense_spans(length: f64, spacing: f64) -> usize {
    let d = (length / (EDGES_PER_SPAN * spacing)).floor();
    if d.is_nan() {
        return SPANS_MIN;
    }
    d.max(SPANS_MIN as f64).min(1e9) as usize
}

/// The input checked for what the fit reads off it: four sides a patch, each of curves the
/// network has, corners it has, and a point for every (u, v).
fn check(input: &NetworkFitInput<'_>) -> Result<(), FitError> {
    let curves: HashSet<u32> = input.curves.iter().map(|c| c.id).collect();
    let nodes: HashSet<u32> = input.nodes.iter().map(|n| n.id).collect();
    for p in &input.patches {
        if p.sides.len() != 4 {
            return Err(FitError::new(format!("A patch takes four sides; the cell at {} has {}.", where_(p.seed), p.sides.len())));
        }
        for s in &p.sides {
            if s.edges.is_empty() {
                return Err(FitError::new(format!("A side of the cell at {} runs along no curve.", where_(p.seed))));
            }
            if let Some(h) = s.edges.iter().find(|h| !curves.contains(&h.curve)) {
                return Err(FitError::new(format!("The cell at {} runs along curve {}, which the network does not have.", where_(p.seed), h.curve)));
            }
        }
        if let Some(c) = p.corners.iter().find(|c| !nodes.contains(c)) {
            return Err(FitError::new(format!("The cell at {} has node {} for a corner, which the network does not have.", where_(p.seed), c)));
        }
        if p.xyz.len() < (p.uv.len() / 2) * 3 {
            return Err(FitError::new(format!("The cell at {} has fewer points than (u, v).", where_(p.seed))));
        }
    }
    Ok(())
}

fn where_(p: DVec3) -> String {
    format!("({}, {}, {})", fixed1(p.x), fixed1(p.y), fixed1(p.z))
}

// ---------------------------------------------------------------------------------------
// The strips.

/// A network's strips: each patch's sides' strips, whether Auto sizes each strip and its
/// curves, and each patch's sides' lengths, mm.
struct Strips {
    of: Vec<[usize; 4]>,
    auto: Vec<bool>,
    curves: Vec<Vec<u32>>,
    side_length: Vec<[f64; 4]>,
}

/// The sides' strips, by union-find: the strip of each patch's side, and how many strips.
fn strips_of(patches: &[PatchInput]) -> (Vec<[usize; 4]>, usize) {
    let n = patches.len() * 4;
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    fn join(parent: &mut [usize], a: usize, b: usize) {
        let x = find(parent, a);
        let y = find(parent, b);
        if x != y {
            parent[x.max(y)] = x.min(y);
        }
    }
    let mut by_curve: HashMap<u32, usize> = HashMap::new();
    for (i, p) in patches.iter().enumerate() {
        join(&mut parent, i * 4, i * 4 + 2);
        join(&mut parent, i * 4 + 1, i * 4 + 3);
        for (k, s) in p.sides.iter().enumerate().take(4) {
            if s.edges.len() != 1 {
                continue;
            }
            let c = s.edges[0].curve;
            match by_curve.get(&c) {
                Some(&had) => join(&mut parent, had, i * 4 + k),
                None => {
                    by_curve.insert(c, i * 4 + k);
                }
            }
        }
    }
    let mut ids: HashMap<usize, usize> = HashMap::new();
    let mut of = Vec::with_capacity(patches.len());
    for i in 0..patches.len() {
        let mut row = [0usize; 4];
        for (k, slot) in row.iter_mut().enumerate() {
            let r = find(&mut parent, i * 4 + k);
            let next = ids.len();
            *slot = *ids.entry(r).or_insert(next);
        }
        of.push(row);
    }
    (of, ids.len())
}

/// Where pole `s` (from the side's start) of a patch's side k is in its net, and the first
/// inner pole in from it: ((i, j) boundary, (i, j) inner).
fn side_pole(nu: usize, nv: usize, k: usize, s: usize) -> ((usize, usize), (usize, usize)) {
    match k {
        0 => ((s, 0), (s, 1)),
        1 => ((nu + 2, s), (nu + 1, s)),
        2 => (((nu + 2).saturating_sub(s), nv + 2), ((nu + 2).saturating_sub(s), nv + 1)),
        _ => ((0, (nv + 2).saturating_sub(s)), (1, (nv + 2).saturating_sub(s))),
    }
}

// ---------------------------------------------------------------------------------------
// The fit.

/// The patches fitted together — see the module documentation. Auto's strips start at
/// AUTO_START spans and grow only where a patch strays from its points by more than the
/// tolerance beyond their scatter ([`FittedPatch::misfit`]), and only while that helps: a net
/// no finer than the shape asks for follows less of the scan's noise — the Joy-Con's 5 mm
/// patches at Auto's 13–19 spans by the old rule (a span every few of the scan's scatters)
/// rippled three times what they did at four, 0.004 mm RMS off the scan against 0.0055, in
/// six times the poles.
pub fn fit_network(input: &NetworkFitInput<'_>) -> Result<NetworkFitResult, FitError> {
    check(input)?;
    let patches = &input.patches;
    let curve_of: HashMap<u32, &CurveInput> = input.curves.iter().map(|c| (c.id, c)).collect();
    // ---- the strips and their span counts --------------------------------
    let (of, count) = strips_of(patches);
    let side_trace = |side: &CellSide| -> CurvePoints {
        let parts: Vec<(PointsRef<'_>, bool)> = side.edges.iter().filter_map(|h| curve_of.get(&h.curve).map(|c| (own_points(c), !h.forward))).collect();
        join_traces(&parts)
    };
    let mut strips = Strips { of: of.clone(), auto: vec![true; count], curves: vec![Vec::new(); count], side_length: vec![[0.0; 4]; patches.len()] };
    let spacing = truthy(input.spacing);
    // The most spans a strip takes: what its longest side's points resolve.
    let mut cap: Vec<usize> = vec![SPANS_MIN; count];
    for (i, p) in patches.iter().enumerate() {
        for (k, s) in p.sides.iter().enumerate().take(4) {
            let st = of[i][k];
            for h in &s.edges {
                if !strips.curves[st].contains(&h.curve) {
                    strips.curves[st].push(h.curve);
                }
            }
            let length = side_trace(s).length;
            strips.side_length[i][k] = length;
            let dense = match spacing {
                Some(sp) => dense_spans(length, sp),
                None => SPANS_MAX,
            };
            cap[st] = cap[st].max(SPANS_MAX.min(dense));
        }
    }
    let mut spans: Vec<usize> = cap.iter().map(|&c| AUTO_START.min(c)).collect();
    for (st, curves) in strips.curves.iter().enumerate() {
        let mut sorted = curves.clone();
        sorted.sort_unstable();
        let given = sorted.iter().find_map(|c| input.spans.get(&c.to_string()).copied());
        if let Some(given) = given {
            spans[st] = js_round(given).min(SPANS_MAX as f64).max(SPANS_MIN as f64) as usize;
            strips.auto[st] = false;
        }
    }
    let mut fit = fit_at(input, &strips, &spans);
    // ---- Auto's rounds ---------------------------------------------------------
    let tolerance = input.tolerance.unwrap_or(DEFAULT_TOLERANCE);
    let mut patches_of: Vec<Vec<usize>> = vec![Vec::new(); count];
    for (i, sides) in of.iter().enumerate() {
        let mut seen: Vec<usize> = Vec::with_capacity(4);
        for &st in sides {
            if !seen.contains(&st) {
                seen.push(st);
                patches_of[st].push(i);
            }
        }
    }
    // A strip whose step did not help leaps the next time — a fillet beside the scooter
    // cover's crest came no nearer at six spans than at four, 0.029 mm against 0.030, and at
    // nine 0.021 — and stays where it was once a leap does not help either.
    let mut leaps: HashSet<usize> = HashSet::new();
    let mut stays: HashSet<usize> = HashSet::new();
    for _round in 0..AUTO_ROUNDS {
        let grows = |st: usize, spans: &[usize], stays: &HashSet<usize>| strips.auto[st] && !stays.contains(&st) && spans[st] < cap[st];
        // Each patch over the tolerance grows its coarser way — its strip whose spans are the
        // longer — or both, where they are alike; the other way once the coarser stays.
        let mut grow: Vec<usize> = Vec::new();
        for (i, p) in fit.patches.iter().enumerate() {
            if !(p.misfit > tolerance) {
                continue;
            }
            let side = strips.side_length[i];
            let ways: Vec<(usize, f64)> = [(of[i][0], (side[0] + side[2]) / 2.0 / p.net.nu as f64), (of[i][1], (side[1] + side[3]) / 2.0 / p.net.nv as f64)]
                .into_iter()
                .filter(|w| grows(w.0, &spans, &stays))
                .collect();
            let longest = ways.iter().fold(0.0f64, |m, w| m.max(w.1));
            for w in ways {
                if w.1 >= AUTO_ALIKE * longest && !grow.contains(&w.0) {
                    grow.push(w.0);
                }
            }
        }
        if grow.is_empty() {
            break;
        }
        let mut next = spans.clone();
        for &st in &grow {
            let factor = if leaps.contains(&st) { AUTO_GROWTH * AUTO_GROWTH } else { AUTO_GROWTH };
            next[st] = cap[st].min((spans[st] as f64 * factor).ceil() as usize);
        }
        let tried = fit_at(input, &strips, &next);
        // A strip's growth is kept where it brought a patch of it that was over the tolerance
        // some way down; else the strip goes back, to leap or to stay (above) — a patch the
        // smoothness holds off its points, more spans do not bring nearer.
        let mut kept = next.clone();
        for &st in &grow {
            let helped = patches_of[st].iter().any(|&i| fit.patches[i].misfit > tolerance && tried.patches[i].misfit <= AUTO_HELPS * fit.patches[i].misfit);
            if helped {
                leaps.remove(&st);
                continue;
            }
            kept[st] = spans[st];
            if leaps.contains(&st) {
                stays.insert(st);
            } else {
                leaps.insert(st);
            }
        }
        if kept == next {
            spans = next;
            fit = tried;
            continue;
        }
        // None helped: the fit stands, and the next round tries the strips that have not
        // been.
        if kept == spans {
            continue;
        }
        spans = kept;
        fit = fit_at(input, &strips, &spans);
    }
    Ok(fit)
}

/// The network read off the scan and its patches fitted together: [`crate::network::read_network`]
/// with [`fit_network`] for its fit.
pub fn read_network_fitted(scan: &ScanPaths, f: &NetworkFeature, bodies: Option<&dyn NetworkBodies>) -> Result<NetworkReading<NetworkFitResult>, FitError> {
    let r = crate::network::read_network(scan, f, bodies, |input| fit_network(&input));
    let fit = r.fit.transpose()?;
    Ok(NetworkReading { traced: r.traced, cells: r.cells, outcomes: r.outcomes, fit, split: r.split, ms: r.ms })
}

/// The way a curve leaves one of its nodes.
#[derive(Debug, Clone, Copy)]
struct Leave {
    curve: u32,
    end: u8,
    dir: DVec3,
}

/// A patch's side the curve is the whole of.
#[derive(Debug, Clone, Copy)]
struct User {
    patch: usize,
    side: usize,
    forward: bool,
}

/// A curve two patches share whole.
#[derive(Debug, Clone, Copy)]
struct SharedLink {
    curve: u32,
    patches: [usize; 2],
    crease: bool,
    sides: [usize; 2],
}

/// A tie across a shared curve.
#[derive(Debug, Clone, Copy)]
struct Tie {
    curve: u32,
    /// The two inner poles, global, and the boundary pole between.
    a: usize,
    c: usize,
    b: usize,
    exact: bool,
    ratio: f64,
    /// Next to a corner: its node.
    node: Option<u32>,
}

/// An inner pole held to leave its side along a plane: n · (A − B) = 0 by weight, with
/// `rhs` = n · B.
#[derive(Debug, Clone, Copy)]
struct FaceRow {
    pole: usize,
    n: DVec3,
    rhs: f64,
}

/// The poles as unknowns: each pole's coordinates a constant and up to one unknown times a
/// factor (a free inner pole; a pole tied to its partner across a curve through a shared d).
struct Layout {
    count: usize,
    base: Vec<f64>,
    z: Vec<i64>,
    coef: Vec<f64>,
    /// The ties held by weight.
    soft: Vec<Tie>,
}

impl Layout {
    fn new(tied: bool, ties: &[Tie], all_poles: &[f64], is_inner: &[bool]) -> Layout {
        let pole_count = is_inner.len();
        let mut base = all_poles.to_vec();
        let mut z = vec![-1i64; pole_count * 3];
        let mut coef = vec![0.0f64; pole_count * 3];
        let mut k: i64 = 0;
        let mut soft = Vec::new();
        if tied {
            for t in ties {
                if !t.exact {
                    soft.push(*t);
                    continue;
                }
                for c in 0..3 {
                    let zi = k;
                    k += 1;
                    base[t.a * 3 + c] = all_poles[t.b * 3 + c];
                    base[t.c * 3 + c] = all_poles[t.b * 3 + c];
                    z[t.a * 3 + c] = zi;
                    coef[t.a * 3 + c] = -1.0;
                    z[t.c * 3 + c] = zi;
                    coef[t.c * 3 + c] = t.ratio;
                }
            }
        }
        for g in 0..pole_count {
            if !is_inner[g] {
                continue;
            }
            for c in 0..3 {
                if z[g * 3 + c] >= 0 {
                    continue;
                }
                base[g * 3 + c] = 0.0;
                z[g * 3 + c] = k;
                k += 1;
                coef[g * 3 + c] = 1.0;
            }
        }
        Layout { count: k as usize, base, z, coef, soft }
    }

    /// The poles of the unknowns `zv`, with the constants or without.
    fn to_poles(&self, zv: &[f64], with_base: bool, y: &mut [f64]) {
        for (i, yi) in y.iter_mut().enumerate() {
            let zi = self.z[i];
            *yi = (if with_base { self.base[i] } else { 0.0 }) + if zi >= 0 { self.coef[i] * zv[zi as usize] } else { 0.0 };
        }
    }

    /// A gradient over the poles taken back to the unknowns.
    fn gather(&self, g: &[f64], out: &mut [f64]) {
        out.fill(0.0);
        for (i, gi) in g.iter().enumerate() {
            let zi = self.z[i];
            if zi >= 0 {
                out[zi as usize] += self.coef[i] * gi;
            }
        }
    }

    /// The unknowns read off the poles as they stand.
    fn z_of(&self, all_poles: &[f64]) -> Vec<f64> {
        let mut zv = vec![0.0; self.count];
        let mut seen = vec![false; self.count];
        for (i, &p) in all_poles.iter().enumerate() {
            let zi = self.z[i];
            if zi < 0 {
                continue;
            }
            let zi = zi as usize;
            let v = (p - self.base[i]) / self.coef[i];
            zv[zi] = if seen[zi] { 0.5 * (zv[zi] + v) } else { v };
            seen[zi] = true;
        }
        zv
    }
}

/// The system's parts a solve reads: each patch's band and right-hand side, where its poles
/// start among all of them, the poles as they stand, and the rows held by weight.
struct System<'a> {
    dims: Vec<(usize, usize)>,
    hessians: &'a [Vec<f64>],
    rhs: &'a [Vec<f64>],
    pole_base: &'a [usize],
    patch_of_pole: &'a [usize],
    all_poles: &'a [f64],
    face_rows: &'a [FaceRow],
}

impl System<'_> {
    fn pole_count(&self) -> usize {
        self.patch_of_pole.len()
    }

    /// G = H y over every patch, plus the soft ties and the face rows; their constants
    /// subtracted where asked.
    fn apply_poles(&self, soft: &[Tie], ties_weight: f64, y: &[f64], g: &mut [f64], with_constants: bool) {
        g.fill(0.0);
        for (p, &(nu, nv)) in self.dims.iter().enumerate() {
            let from = self.pole_base[p] * 3;
            let to = (self.pole_base[p] + (nu + 3) * (nv + 3)) * 3;
            band_multiply(&self.hessians[p], nu, nv, &y[from..to], &mut g[from..]);
        }
        for t in soft {
            let w = ties_weight / (1.0 + t.ratio * t.ratio);
            for c in 0..3 {
                let r = y[t.a * 3 + c] + t.ratio * y[t.c * 3 + c] - if with_constants { (1.0 + t.ratio) * self.all_poles[t.b * 3 + c] } else { 0.0 };
                g[t.a * 3 + c] += w * r;
                g[t.c * 3 + c] += w * t.ratio * r;
            }
        }
        for row in self.face_rows {
            let p = row.pole * 3;
            let r = row.n.x * y[p] + row.n.y * y[p + 1] + row.n.z * y[p + 2] - if with_constants { row.rhs } else { 0.0 };
            g[p] += ties_weight * row.n.x * r;
            g[p + 1] += ties_weight * row.n.y * r;
            g[p + 2] += ties_weight * row.n.z * r;
        }
    }

    /// The layout solved from `start`: its unknowns and the iterations it took.
    fn solve(&self, lay: &Layout, start: Option<Vec<f64>>) -> (Vec<f64>, usize) {
        // The tie weight against the data's mean diagonal.
        let mut diag_sum = 0.0;
        let mut diag_count = 0usize;
        for h in self.hessians {
            for p in 0..h.len() / BAND {
                diag_sum += h[p * BAND + 24];
                diag_count += 1;
            }
        }
        let ties_weight = TIE_WEIGHT * (diag_sum / diag_count.max(1) as f64);
        let n3 = self.pole_count() * 3;
        let mut y = vec![0.0; n3];
        let mut g = vec![0.0; n3];
        // The right-hand side: the data's, less what the constants do.
        let zero = vec![0.0; lay.count];
        lay.to_poles(&zero, true, &mut y);
        self.apply_poles(&lay.soft, ties_weight, &y, &mut g, true);
        for v in g.iter_mut() {
            *v = -*v;
        }
        for (p, r) in self.rhs.iter().enumerate() {
            let from = self.pole_base[p] * 3;
            for (i, ri) in r.iter().enumerate() {
                g[from + i] += ri;
            }
        }
        let mut b = vec![0.0; lay.count];
        lay.gather(&g, &mut b);
        // The diagonal, for the preconditioner.
        let mut diagonal = vec![0.0; lay.count];
        for i in 0..n3 {
            let zi = lay.z[i];
            if zi < 0 {
                continue;
            }
            let gp = i / 3;
            let p = self.patch_of_pole[gp];
            diagonal[zi as usize] += lay.coef[i] * lay.coef[i] * self.hessians[p][(gp - self.pole_base[p]) * BAND + 24];
        }
        for t in &lay.soft {
            let w = ties_weight / (1.0 + t.ratio * t.ratio);
            for c in 0..3 {
                let za = lay.z[t.a * 3 + c];
                let zc = lay.z[t.c * 3 + c];
                if za >= 0 {
                    let k = lay.coef[t.a * 3 + c];
                    diagonal[za as usize] += w * (k * k);
                }
                if zc >= 0 {
                    let k = t.ratio * lay.coef[t.c * 3 + c];
                    diagonal[zc as usize] += w * (k * k);
                }
            }
        }
        for row in self.face_rows {
            for c in 0..3 {
                let z = lay.z[row.pole * 3 + c];
                if z >= 0 {
                    let k = row.n[c] * lay.coef[row.pole * 3 + c];
                    diagonal[z as usize] += ties_weight * (k * k);
                }
            }
        }
        let op = FitOperator { system: self, lay, ties_weight, diagonal, y: RefCell::new(y), g: RefCell::new(g) };
        let r = conjugate_gradient(&op, &b, &CgOptions { x0: start, tolerance: SOLVE_TOLERANCE, max_iterations: Some(SOLVE_MAX_ITERATIONS), precondition: true });
        (r.x, r.iterations)
    }
}

/// The system as the conjugate gradient sees it: the unknowns to poles, the bands and the
/// rows held by weight, and back.
struct FitOperator<'a, 'b> {
    system: &'a System<'b>,
    lay: &'a Layout,
    ties_weight: f64,
    diagonal: Vec<f64>,
    y: RefCell<Vec<f64>>,
    g: RefCell<Vec<f64>>,
}

impl LinearOperator for FitOperator<'_, '_> {
    fn size(&self) -> usize {
        self.lay.count
    }

    fn apply(&self, x: &[f64], out: &mut [f64]) {
        let mut y = self.y.borrow_mut();
        let mut g = self.g.borrow_mut();
        self.lay.to_poles(x, false, &mut y);
        self.system.apply_poles(&self.lay.soft, self.ties_weight, &y, &mut g, false);
        self.lay.gather(&g, out);
    }

    fn diagonal(&self) -> Option<Vec<f64>> {
        Some(self.diagonal.clone())
    }
}

/// The poles of a layout's unknowns written to all the poles and each patch's net.
fn poles_of(lay: &Layout, zv: &[f64], all_poles: &mut [f64], nets: &mut [PatchNet], pole_base: &[usize]) {
    for (i, p) in all_poles.iter_mut().enumerate() {
        let zi = lay.z[i];
        *p = lay.base[i] + if zi >= 0 { lay.coef[i] * zv[zi as usize] } else { 0.0 };
    }
    for (p, net) in nets.iter_mut().enumerate() {
        let from = pole_base[p] * 3;
        let len = net.poles.len();
        net.poles.copy_from_slice(&all_poles[from..from + len]);
    }
}

/// Each patch's normal equations over its own poles: a band of 7 × 7 a pole, the data, the
/// fairing and the pull, and the data's right-hand side, three coordinates.
struct Bands {
    hessians: Vec<Vec<f64>>,
    rhs: Vec<Vec<f64>>,
    fairings: Vec<Vec<f64>>,
    fairing_rhs: Vec<Vec<f64>>,
}

impl Bands {
    /// Every patch's band and right-hand side made again from its fairing, its points as they
    /// stand and the pull.
    fn assemble(&mut self, nets: &[PatchNet], data: &[PatchData]) {
        for (i, net) in nets.iter().enumerate() {
            let h = &mut self.hessians[i];
            let g = &mut self.rhs[i];
            h.copy_from_slice(&self.fairings[i]);
            g.copy_from_slice(&self.fairing_rhs[i]);
            let d = &data[i];
            let cols = net.nv + 3;
            let mut bu = [0.0f64; 4];
            let mut bv = [0.0f64; 4];
            let mut held = vec![false; net.nu * net.nv];
            for k in 0..d.count {
                if !d.keep[k] {
                    continue;
                }
                let su = basis_at(net.nu, d.uv[k * 2], &mut bu);
                let sv = basis_at(net.nv, d.uv[k * 2 + 1], &mut bv);
                held[(su - DEGREE) * net.nv + (sv - DEGREE)] = true;
                add_point(h, g, cols, su, sv, &bu, &bv, d.xyz[k * 3], d.xyz[k * 3 + 1], d.xyz[k * 3 + 2], 1.0);
            }
            add_pull(h, g, net, &held, d.kept);
        }
    }
}

/// The patches fitted together at the strips' span counts.
fn fit_at(input: &NetworkFitInput<'_>, strips: &Strips, strip_spans: &[usize]) -> NetworkFitResult {
    let patches = &input.patches;
    let side_length = &strips.side_length;
    let curve_of: HashMap<u32, &CurveInput> = input.curves.iter().map(|c| (c.id, c)).collect();
    let node_of: HashMap<u32, DVec3> = input.nodes.iter().map(|n| (n.id, n.at)).collect();
    let spacing = truthy(input.spacing);
    let crease_of = |id: u32| curve_of.get(&id).is_some_and(|c| c.crease);
    // A curve's span count is its strip's where it is a side of one cell; one that is no
    // cell's whole side keeps its own Auto.
    let mut curve_spans: HashMap<u32, usize> = HashMap::new();
    for (i, p) in patches.iter().enumerate() {
        for (k, s) in p.sides.iter().enumerate().take(4) {
            if s.edges.len() == 1 {
                curve_spans.insert(s.edges[0].curve, strip_spans[strips.of[i][k]]);
            }
        }
    }

    // ---- the curves, held tangent where they go on through a node ---------
    // At each node the curves leaving it nearly straight on from each other are paired, and
    // each pair is given one tangent line there. The way each curve leaves each of its
    // nodes, read off its points near the node (leaving_direction); every end is held to it,
    // unless it goes on straight through the node with another, or the node's tangent plane
    // or a body's face holds it otherwise (below).
    let mut leave_keys: Vec<u32> = Vec::new();
    let mut leaving: HashMap<u32, Vec<Leave>> = HashMap::new();
    let mut held: HashMap<(u32, u8), DVec3> = HashMap::new();
    for c in &input.curves {
        let own = own_points(c);
        let out = match spacing {
            Some(sp) => leaving_direction(own, 0, sp),
            None => end_tangent(&c.fit, 0),
        };
        let back = match spacing {
            Some(sp) => leaving_direction(own, 1, sp),
            None => -end_tangent(&c.fit, 1),
        };
        for (node, x) in [(c.from, Leave { curve: c.id, end: 0, dir: out }), (c.to, Leave { curve: c.id, end: 1, dir: back })] {
            leaving
                .entry(node)
                .or_insert_with(|| {
                    leave_keys.push(node);
                    Vec::new()
                })
                .push(x);
        }
        held.insert((c.id, 0), out);
        held.insert((c.id, 1), back);
    }
    let straight_on = (30.0f64.to_radians()).cos();
    // At a node on a crease only the crease's own pieces go on through it: two curves either
    // side of the fold meet at its angle, and held to one tangent line they flattened the
    // fold — a 30° fold's corner tangents made level carried a level cross slope along the
    // whole crease into the Coons surface, and one patch read 0.8° across it instead of 30°.
    let mut on_crease: HashSet<u32> = HashSet::new();
    for c in &input.curves {
        if !c.crease {
            continue;
        }
        on_crease.insert(c.from);
        on_crease.insert(c.to);
    }
    for node in &leave_keys {
        let Some(list) = leaving.get(node) else { continue };
        let mut pairs: Vec<(usize, usize, f64)> = Vec::new();
        for a in 0..list.len() {
            for b in a + 1..list.len() {
                if list[a].curve == list[b].curve {
                    continue;
                }
                if on_crease.contains(node) && !(crease_of(list[a].curve) && crease_of(list[b].curve)) {
                    continue;
                }
                let cos = -(list[a].dir.x * list[b].dir.x + list[a].dir.y * list[b].dir.y + list[a].dir.z * list[b].dir.z);
                if cos > straight_on {
                    pairs.push((a, b, cos));
                }
            }
        }
        pairs.sort_by(|x, y| y.2.total_cmp(&x.2));
        let mut taken: HashSet<usize> = HashSet::new();
        for (a, b, _) in pairs {
            if taken.contains(&a) || taken.contains(&b) {
                continue;
            }
            taken.insert(a);
            taken.insert(b);
            let d = list[a].dir;
            let e = list[b].dir;
            let l = DVec3::new(d.x - e.x, d.y - e.y, d.z - e.z);
            let n = hypot3(l.x, l.y, l.z);
            let n = if n != 0.0 && !n.is_nan() { n } else { 1.0 };
            let line = DVec3::new(l.x / n, l.y / n, l.z / n);
            held.insert((list[a].curve, list[a].end), line);
            held.insert((list[b].curve, list[b].end), -line);
        }
    }
    // Where three curves or more meet and none of them is a crease, every one leaves in one
    // tangent plane — the plane through their directions nearest them all — so the patches
    // meeting there have one normal there. Left to each curve's own end, the noise in each
    // tilts them apart: the centre of a pentagon split from it read 4.5° between its patches.
    for node in &leave_keys {
        let Some(list) = leaving.get(node) else { continue };
        if list.len() < 3 || on_crease.contains(node) {
            continue;
        }
        let dirs: Vec<DVec3> = list.iter().map(|x| held.get(&(x.curve, x.end)).copied().unwrap_or(x.dir)).collect();
        let mut m = [0.0f64; 9];
        for d in &dirs {
            for a in 0..3 {
                for b in 0..3 {
                    m[a * 3 + b] += d[a] * d[b];
                }
            }
        }
        let (_, vectors) = symmetric_eigen_n(3, &m);
        let Some(normal) = vectors.first().filter(|v| v.len() == 3).map(|v| DVec3::new(v[0], v[1], v[2])) else { continue };
        for (i, x) in list.iter().enumerate() {
            let d = dirs[i];
            let s = dot(d, normal);
            let p = DVec3::new(d.x - s * normal.x, d.y - s * normal.y, d.z - s * normal.z);
            let l = hypot3(p.x, p.y, p.z);
            if l > 1e-9 {
                held.insert((x.curve, x.end), DVec3::new(p.x / l, p.y / l, p.z / l));
            }
        }
    }
    // Where a curve runs beside a body's face held tangent to it, every curve leaving its two
    // nodes leaves in the face's tangent plane there: the patches' corners on the body's edge
    // then meet the face tangent.
    let mut tangent_to: Vec<(u32, &BodySide<'_>)> = Vec::new();
    for b in input.body_sides.iter().filter(|b| b.mode == SideMode::Tangent) {
        match tangent_to.iter_mut().find(|t| t.0 == b.curve) {
            Some(t) => t.1 = b,
            None => tangent_to.push((b.curve, b)),
        }
    }
    for &(curve, side) in &tangent_to {
        let Some(c) = curve_of.get(&curve) else { continue };
        for node in [c.from, c.to] {
            let normal = node_of.get(&node).and_then(|&at| (side.normals)(&[at])).and_then(|n| n.first().copied());
            let Some(normal) = normal else { continue };
            let Some(list) = leaving.get(&node) else { continue };
            for x in list {
                let d = held.get(&(x.curve, x.end)).copied().unwrap_or(x.dir);
                let s = dot(d, normal);
                let p = DVec3::new(d.x - s * normal.x, d.y - s * normal.y, d.z - s * normal.z);
                let l = hypot3(p.x, p.y, p.z);
                if l > 1e-9 {
                    held.insert((x.curve, x.end), DVec3::new(p.x / l, p.y / l, p.z / l));
                }
            }
        }
    }
    // The smoothness fairs the curves as well, as hard as the patches beside them: each patch
    // is held to its sides, so their noise was carried across it — on a Joy-Con's 5 mm
    // patches from 0.001 to 10 the patches' ripple did not move, the curves' fairing being a
    // fixed whisper. A patch's fairing against its points stands at s A² ∫ |∂³P|² dA over N
    // points on its area A; a curve's against its trace's n points over its length L
    // flattens the same waves at s A³ n / (N L) (fairing_of). A curve on two patches takes
    // their mean; a body's edge, no scan noise in it, none.
    let smoothness = input.smoothness.max(0.0) * SMOOTHNESS_SCALE;
    let patch_fairing: Vec<f64> = patches
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let area = ((side_length[i][0] + side_length[i][2]) / 2.0) * ((side_length[i][1] + side_length[i][3]) / 2.0);
            let points = ((p.uv.len() / 2) as f64).min((POINTS_PER_CELL * strip_spans[strips.of[i][0]] * strip_spans[strips.of[i][1]]) as f64);
            (smoothness * area.powf(3.0)) / points.max(1.0)
        })
        .collect();
    let mut on_patches: HashMap<u32, Vec<usize>> = HashMap::new();
    for (i, p) in patches.iter().enumerate() {
        for s in &p.sides {
            for h in &s.edges {
                on_patches.entry(h.curve).or_default().push(i);
            }
        }
    }
    let fairing_of = |trace: PointsRef<'_>, on: &[usize]| -> f64 {
        if !(smoothness > 0.0) || on.is_empty() || !(trace.length > 0.0) {
            return 0.0;
        }
        let sum = on.iter().fold(0.0, |sum, &i| sum + patch_fairing[i]);
        ((trace.points.len() / 3) as f64 / trace.length) * (sum / on.len() as f64)
    };
    let mut fits: BTreeMap<u32, CurveFit> = BTreeMap::new();
    for c in &input.curves {
        let own = own_points(c);
        let spans = match curve_spans.get(&c.id) {
            Some(&s) => s,
            None => auto_spans(own).min(match spacing {
                Some(sp) => dense_spans(own.length, sp),
                None => SPANS_MAX,
            }),
        };
        let out = held.get(&(c.id, 0)).copied();
        let back = held.get(&(c.id, 1)).copied();
        let fairing = if c.exact.is_some() { 0.0 } else { fairing_of(own, on_patches.get(&c.id).map(|v| v.as_slice()).unwrap_or(&[])) };
        fits.insert(c.id, fit_curve(own, spans, &CurveEnds { start_tangent: out, end_tangent: back.map(|b| -b) }, fairing));
    }

    // ---- the nets: boundary rows from the sides -----------------------------
    let mut nets: Vec<PatchNet> = Vec::with_capacity(patches.len());
    let mut side_rows: Vec<Vec<FittedSide>> = Vec::with_capacity(patches.len());
    let mut pole_base: Vec<usize> = Vec::with_capacity(patches.len());
    let mut pole_count = 0usize;
    for (i, p) in patches.iter().enumerate() {
        let nu = strip_spans[strips.of[i][0]];
        let nv = strip_spans[strips.of[i][1]];
        let mut net = PatchNet { nu, nv, poles: vec![0.0; (nu + 3) * (nv + 3) * 3] };
        let mut rows: Vec<FittedSide> = Vec::with_capacity(4);
        for (k, s) in p.sides.iter().enumerate().take(4) {
            let spans = if k % 2 == 0 { nu } else { nv };
            let poles: Vec<f64> = if s.edges.len() == 1 {
                match fits.get(&s.edges[0].curve) {
                    Some(f) if s.edges[0].forward => f.poles.clone(),
                    Some(f) => reversed_poles(&f.poles),
                    None => Vec::new(),
                }
            } else {
                // A side of more than one curve — a T-junction's — fitted to its curves as
                // fitted, not to their traces: faired by the smoothness, the curves left the
                // traces, and a side fitted to the traces stood 50 µm off them.
                let samples: Vec<(CurvePoints, bool)> = s.edges.iter().filter_map(|h| fits.get(&h.curve).map(|f| (curve_samples(f, SIDE_SAMPLES_PER_SPAN), !h.forward))).collect();
                let parts: Vec<(PointsRef<'_>, bool)> = samples.iter().map(|(c, r)| (c.as_points(), *r)).collect();
                fit_curve(join_traces(&parts).as_points(), spans, &CurveEnds::default(), 0.0).poles
            };
            for m in 0..spans + 3 {
                let ((bi, bj), _) = side_pole(nu, nv, k, m);
                let at = (bi * (nv + 3) + bj) * 3;
                if m * 3 + 3 <= poles.len() && at + 3 <= net.poles.len() {
                    net.poles[at..at + 3].copy_from_slice(&poles[m * 3..m * 3 + 3]);
                }
            }
            rows.push(FittedSide { edges: s.edges.clone(), poles, single: s.edges.len() == 1 });
        }
        // The inner poles start on the Coons surface of the four rows.
        net.poles = coons_poles(&net);
        nets.push(net);
        side_rows.push(rows);
        pole_base.push(pole_count);
        pole_count += (nu + 3) * (nv + 3);
    }

    // ---- the data ------------------------------------------------------------
    // The map puts the cell's boundary vertices on the square's sides, though they lie up to
    // a triangle off the curves — so near a side its (u, v) are squeezed by a row of the
    // scan's triangles, and a cylinder's strip fitted at them was 0.12 mm off at eight spans.
    // Each point is taken from where the map put it onto the Coons surface of the patch's
    // four sides, which the sides parameterise: the map keeps it in the right place on a cell
    // that folds back, the Coons surface makes its (u, v) the sides'.
    let mut data: Vec<PatchData> = patches.iter().zip(&nets).map(|(p, net)| take_evenly(p, POINTS_PER_CELL * net.nu * net.nv)).collect();
    project(&nets, &mut data);

    // ---- the shared curves, the ties across them, and the corners -------------
    let mut shared_keys: Vec<u32> = Vec::new();
    let mut shared_by: HashMap<u32, Vec<User>> = HashMap::new();
    for (i, p) in patches.iter().enumerate() {
        for (k, s) in p.sides.iter().enumerate().take(4) {
            if s.edges.len() != 1 {
                continue;
            }
            let h = s.edges[0];
            shared_by
                .entry(h.curve)
                .or_insert_with(|| {
                    shared_keys.push(h.curve);
                    Vec::new()
                })
                .push(User { patch: i, side: k, forward: h.forward });
        }
    }
    let mut corner_keys: Vec<u32> = Vec::new();
    let mut corner_patches: HashMap<u32, usize> = HashMap::new();
    for p in patches {
        for &c in &p.corners {
            *corner_patches.entry(c).or_insert_with(|| {
                corner_keys.push(c);
                0
            }) += 1;
        }
    }
    // A node all round which patches stand, with three, five or six of them.
    let surrounded = |node: u32| leaving.get(&node).map(|list| list.iter().all(|x| shared_by.get(&x.curve).map(|u| u.len()).unwrap_or(0) == 2)).unwrap_or(true);
    let mut irregular: Vec<u32> = Vec::new();
    for &node in &corner_keys {
        let count = corner_patches.get(&node).copied().unwrap_or(0);
        if count != 4 && count >= 3 && surrounded(node) {
            irregular.push(node);
        }
    }

    let mut ties: Vec<Tie> = Vec::new();
    let mut shared: Vec<SharedLink> = Vec::new();
    for &curve in &shared_keys {
        let Some(users) = shared_by.get(&curve) else { continue };
        if users.len() != 2 {
            continue;
        }
        let (Some(c), Some(f)) = (curve_of.get(&curve), fits.get(&curve)) else { continue };
        shared.push(SharedLink { curve, patches: [users[0].patch, users[1].patch], crease: c.crease, sides: [users[0].side, users[1].side] });
        if c.crease {
            continue;
        }
        let n = f.spans + 3;
        let (pu, qu) = (users[0], users[1]);
        for m in 1..=n - 2 {
            // Next to a corner the tie is the twist pole's, which two rows share.
            let near_from = m == 1;
            let near_to = m == n - 2;
            if (near_from && irregular.contains(&c.from)) || (near_to && irregular.contains(&c.to)) {
                continue;
            }
            let at = |u: User| -> (usize, usize) {
                let net = &nets[u.patch];
                let s = if u.forward { m } else { n - 1 - m };
                let ((bi, bj), (ii, ij)) = side_pole(net.nu, net.nv, u.side, s);
                (pole_base[u.patch] + bi * (net.nv + 3) + bj, pole_base[u.patch] + ii * (net.nv + 3) + ij)
            };
            let (pb, pi) = at(pu);
            let (_, qi) = at(qu);
            ties.push(Tie {
                curve,
                a: pi,
                c: qi,
                b: pb,
                exact: !(near_from || near_to),
                ratio: 1.0,
                node: if near_from {
                    Some(c.from)
                } else if near_to {
                    Some(c.to)
                } else {
                    None
                },
            });
        }
    }

    // ---- the patches beside a body's face, tangent to it --------------------------
    // With A a pole of the first inner row and B the pole of the side's row beside it,
    // n · (A − B) = 0, n the face's normal by the curve there: the patch leaves its side
    // along the face. One row a pole, by weight; it ties x, y and z together, which is why
    // the fit is one system of all three rather than three of one.
    let mut face_rows: Vec<FaceRow> = Vec::new();
    for &(curve, side) in &tangent_to {
        let users: &[User] = shared_by.get(&curve).map(|u| u.as_slice()).unwrap_or(&[]);
        let Some(f) = fits.get(&curve) else { continue };
        let n = f.spans + 3;
        let gr = greville_of(f.spans);
        let at: Vec<DVec3> = (0..n).map(|m| curve_point(&f.poles, f.spans, gr[m])).collect();
        let Some(normals) = (side.normals)(&at) else { continue };
        for u in users {
            let net = &nets[u.patch];
            for m in 1..=n - 2 {
                let s = if u.forward { m } else { n - 1 - m };
                let ((bi, bj), (ii, ij)) = side_pole(net.nu, net.nv, u.side, s);
                let b = (bi * (net.nv + 3) + bj) * 3;
                let a = pole_base[u.patch] + ii * (net.nv + 3) + ij;
                let Some(&nm) = normals.get(m) else { continue };
                face_rows.push(FaceRow { pole: a, n: nm, rhs: nm.x * net.poles[b] + nm.y * net.poles[b + 1] + nm.z * net.poles[b + 2] });
            }
        }
    }

    // ---- the system -------------------------------------------------------------
    let mut all_poles = vec![0.0f64; pole_count * 3];
    let mut is_inner = vec![false; pole_count];
    let mut patch_of_pole = vec![0usize; pole_count];
    for (i, net) in nets.iter().enumerate() {
        all_poles[pole_base[i] * 3..pole_base[i] * 3 + net.poles.len()].copy_from_slice(&net.poles);
        for a in 1..=net.nu + 1 {
            for b in 1..=net.nv + 1 {
                is_inner[pole_base[i] + a * (net.nv + 3) + b] = true;
            }
        }
        patch_of_pole[pole_base[i]..pole_base[i] + (net.nu + 3) * (net.nv + 3)].fill(i);
    }
    let dims: Vec<(usize, usize)> = nets.iter().map(|n| (n.nu, n.nv)).collect();
    let fairings: Vec<Vec<f64>> = nets.iter().zip(&side_rows).map(|(net, rows)| fairing_band(net, input.smoothness, rows)).collect();
    // The fairing is of the patch's departure from its sides' Coons surface, not of the
    // patch: a circle's poles have third differences of their own, and faired toward none a
    // strip round a cylinder was pulled 0.05 mm in. The Coons surface carries the sides' bend
    // across; what is faired is the shape the points add to it. Its right-hand side: H_fair
    // times the Coons poles.
    let fairing_rhs: Vec<Vec<f64>> = nets
        .iter()
        .zip(&fairings)
        .map(|(net, h)| {
            let mut out = vec![0.0; (net.nu + 3) * (net.nv + 3) * 3];
            band_multiply(h, net.nu, net.nv, &coons_poles(net), &mut out);
            out
        })
        .collect();
    let mut bands = Bands {
        hessians: nets.iter().map(|net| vec![0.0; (net.nu + 3) * (net.nv + 3) * BAND]).collect(),
        rhs: nets.iter().map(|net| vec![0.0; (net.nu + 3) * (net.nv + 3) * 3]).collect(),
        fairings,
        fairing_rhs,
    };

    // ---- the first fit, free of the ties: the cross-spacings' ratios --------
    bands.assemble(&nets, &data);
    let free = Layout::new(false, &ties, &all_poles, &is_inner);
    let mut iterations = 0usize;
    let (first, its) = System { dims: dims.clone(), hessians: &bands.hessians, rhs: &bands.rhs, pole_base: &pole_base, patch_of_pole: &patch_of_pole, all_poles: &all_poles, face_rows: &face_rows }.solve(&free, None);
    iterations += its;
    poles_of(&free, &first, &mut all_poles, &mut nets, &pole_base);
    for t in ties.iter_mut() {
        let mut num = 0.0;
        let mut den = 0.0;
        for c in 0..3 {
            let x = all_poles[t.c * 3 + c] - all_poles[t.b * 3 + c];
            let y = all_poles[t.b * 3 + c] - all_poles[t.a * 3 + c];
            num += x * x;
            den += y * y;
        }
        t.ratio = if den > 0.0 { (num / den).sqrt() } else { 1.0 };
    }
    smooth_ratios(&mut ties);
    for t in ties.iter_mut() {
        t.ratio = t.ratio.clamp(1.0 / RATIO_LIMIT, RATIO_LIMIT);
    }
    let free_poles = all_poles.clone();

    // ---- tangency across a T-junction ---------------------------------------------
    // A curve that is one patch's whole side and a piece of another's — a split cell's middle
    // on its neighbour's side — shares no poles with that side, so no tie holds the two
    // tangent across it: on a Joy-Con they met at up to 14°. Each is held as a body's face
    // holds a patch (above): its first inner row leaves its side square to one normal there —
    // the two patches' normals in the first fit, met halfway and made square to the curve.
    // The poles beside a corner are left to the ties round it.
    for (qi, rows) in side_rows.iter().enumerate() {
        for (k, side) in rows.iter().enumerate() {
            if side.single {
                continue;
            }
            let q_net = &nets[qi];
            let q_spans = if k % 2 == 0 { q_net.nu } else { q_net.nv };
            let gq = greville_of(q_spans);
            for h in &side.edges {
                let Some(p) = shared_by.get(&h.curve).and_then(|u| u.iter().find(|u| u.patch != qi).copied()) else { continue };
                if crease_of(h.curve) {
                    continue;
                }
                let Some(f) = fits.get(&h.curve) else { continue };
                let n = f.spans + 3;
                // The normal both are held to at t on the curve, s on Q's side.
                let normal_at = |t: f64, s: f64| -> DVec3 {
                    let (pu, pv) = on_side(p.side, if p.forward { t } else { 1.0 - t });
                    let a = patch_normal(&nets[p.patch], pu, pv);
                    let (qu, qv) = on_side(k, s);
                    let b = patch_normal(q_net, qu, qv);
                    let sign = if dot(a, b) < 0.0 { -1.0 } else { 1.0 };
                    let mut m = DVec3::new(a.x + sign * b.x, a.y + sign * b.y, a.z + sign * b.z);
                    let d = curve_derivative(&f.poles, f.spans, t);
                    let along = dot(m, d) / (d.x * d.x + d.y * d.y + d.z * d.z).max(1e-30);
                    for c in 0..3 {
                        m[c] -= along * d[c];
                    }
                    let l = hypot3(m.x, m.y, m.z);
                    let l = if l != 0.0 && !l.is_nan() { l } else { 1.0 };
                    DVec3::new(m.x / l, m.y / l, m.z / l)
                };
                let hold = |patch: usize, side_of: usize, at: usize, normal: DVec3, rows: &mut Vec<FaceRow>| {
                    let net = &nets[patch];
                    let ((bi, bj), (ii, ij)) = side_pole(net.nu, net.nv, side_of, at);
                    let b = (bi * (net.nv + 3) + bj) * 3;
                    rows.push(FaceRow { pole: pole_base[patch] + ii * (net.nv + 3) + ij, n: normal, rhs: normal.x * net.poles[b] + normal.y * net.poles[b + 1] + normal.z * net.poles[b + 2] });
                };
                // The whole side's poles: the curve's own.
                let gp = greville_of(f.spans);
                for m in 2..=n - 3 {
                    let x = curve_point(&f.poles, f.spans, gp[m]);
                    hold(p.patch, p.side, if p.forward { m } else { n - 1 - m }, normal_at(gp[m], nearest_on(&side.poles, q_spans, x)), &mut face_rows);
                }
                // The other's along the piece of its side the curve is.
                let ends = [0.0, 1.0].map(|t| nearest_on(&side.poles, q_spans, curve_point(&f.poles, f.spans, t)));
                let lo = ends[0].min(ends[1]);
                let hi = ends[0].max(ends[1]);
                for m in 2..=q_spans {
                    if !(gq[m] > lo && gq[m] < hi) {
                        continue;
                    }
                    let y = curve_point(&side.poles, q_spans, gq[m]);
                    hold(qi, k, m, normal_at(nearest_on(&f.poles, f.spans, y), gq[m]), &mut face_rows);
                }
            }
        }
    }

    // ---- tied, and the parameters corrected -------------------------------------
    let mut lay = Layout::new(!ties.is_empty(), &ties, &all_poles, &is_inner);
    let (mut rms, mut res) = residual_rms(&nets, &data);
    let mut rounds = 0usize;
    while rounds < ROUNDS {
        let start = lay.z_of(&all_poles);
        let (z, its) = System { dims: dims.clone(), hessians: &bands.hessians, rhs: &bands.rhs, pole_base: &pole_base, patch_of_pole: &patch_of_pole, all_poles: &all_poles, face_rows: &face_rows }.solve(&lay, Some(start));
        iterations += its;
        poles_of(&lay, &z, &mut all_poles, &mut nets, &pole_base);
        if rounds == 0 {
            // The corners whose ties could not be held (see above): let go, and the fit
            // solved again from the first.
            let tied: HashSet<u32> = ties.iter().filter_map(|t| t.node).collect();
            let torn = torn_corners(&nets, &free_poles, &pole_base, patches, &tied);
            if !torn.is_empty() {
                ties.retain(|t| t.node.is_none_or(|n| !torn.contains(&n)));
                for &node in &torn {
                    if !irregular.contains(&node) {
                        irregular.push(node);
                    }
                }
                all_poles.copy_from_slice(&free_poles);
                lay = Layout::new(!ties.is_empty(), &ties, &all_poles, &is_inner);
                let start = lay.z_of(&all_poles);
                let (again, its) = System { dims: dims.clone(), hessians: &bands.hessians, rhs: &bands.rhs, pole_base: &pole_base, patch_of_pole: &patch_of_pole, all_poles: &all_poles, face_rows: &face_rows }.solve(&lay, Some(start));
                iterations += its;
                poles_of(&lay, &again, &mut all_poles, &mut nets, &pole_base);
            }
        }
        if rounds == 0 && input.robust == Some(true) {
            drop_outliers(&nets, &mut data);
        }
        project(&nets, &mut data);
        // The last round's residuals are the answers': nothing moves after.
        let (next_rms, next_res) = residual_rms(&nets, &data);
        res = next_res;
        let settled = (next_rms - rms).abs() <= ROUNDS_SETTLE * rms.max(1e-9);
        rms = next_rms;
        if settled && rounds > 0 {
            rounds += 1;
            break;
        }
        if rounds < ROUNDS - 1 {
            bands.assemble(&nets, &data);
        }
        rounds += 1;
    }

    // ---- the answers ------------------------------------------------------------
    let fitted: Vec<FittedPatch> = nets
        .iter()
        .enumerate()
        .map(|(i, net)| {
            let d = &data[i];
            let r = &res[i];
            let mut sq = 0.0;
            let mut max = 0.0f64;
            let mut n = 0usize;
            // The worst few kept in a pass, not found by sorting them all: in the sort's
            // order, the earlier of two alike first.
            let mut order: Vec<usize> = Vec::with_capacity(WORST_POINTS + 1);
            for k in 0..d.count {
                if !d.keep[k] {
                    continue;
                }
                let off = r[k].abs();
                sq += r[k] * r[k];
                max = max.max(off);
                n += 1;
                let mut at = order.len();
                while at > 0 && off > r[order[at - 1]].abs() {
                    at -= 1;
                }
                if at < WORST_POINTS {
                    order.insert(at, k);
                    if order.len() > WORST_POINTS {
                        order.pop();
                    }
                }
            }
            let worst = order.iter().map(|&k| WorstPoint { at: DVec3::new(d.xyz[k * 3], d.xyz[k * 3 + 1], d.xyz[k * 3 + 2]), off: r[k] }).collect();
            let aspect = (side_length[i][0] + side_length[i][2]) / (side_length[i][1] + side_length[i][3]).max(1e-9);
            FittedPatch {
                net: net.clone(),
                sides: side_rows[i].clone(),
                corners: patches[i].corners.clone(),
                seed: patches[i].seed,
                points: n,
                rms: if n > 0 { (sq / n as f64).sqrt() } else { 0.0 },
                max,
                misfit: misfit_of(d, r, aspect),
                worst,
                dropped: if input.robust == Some(true) { Some(d.count - n) } else { None },
            }
        })
        .collect();
    // Within two spans of an irregular corner — or one whose ties were let go — the tie was
    // let go: the angle there is the corner's, read at the corner, not the curve's.
    let mut near_corner: HashMap<u32, f64> = HashMap::new();
    let mut shared_out: Vec<SharedCurve> = Vec::with_capacity(shared.len());
    for s in &shared {
        let (Some(c), Some(f)) = (curve_of.get(&s.curve), fits.get(&s.curve)) else { continue };
        let reach = 2.0 / f.spans as f64;
        let from_irregular = irregular.contains(&c.from);
        let to_irregular = irregular.contains(&c.to);
        let (reading, near_from, near_to) = edge_reading(s, &fitted, if from_irregular { reach } else { 0.0 }, if to_irregular { reach } else { 0.0 });
        if from_irregular && !s.crease {
            let e = near_corner.entry(c.from).or_insert(0.0);
            *e = e.max(near_from);
        }
        if to_irregular && !s.crease {
            let e = near_corner.entry(c.to).or_insert(0.0);
            *e = e.max(near_to);
        }
        shared_out.push(reading);
    }
    // A curve that is one patch's whole side and a piece of another's — the side through a
    // T-node a split cell's middle made: read by the nearest point of the other's side to
    // each of the curve's.
    for (qi, q) in fitted.iter().enumerate() {
        for (k, side) in q.sides.iter().enumerate() {
            if side.single {
                continue;
            }
            for h in &side.edges {
                let Some(p) = shared_by.get(&h.curve).and_then(|u| u.iter().find(|u| u.patch != qi).copied()) else { continue };
                let Some(f) = fits.get(&h.curve) else { continue };
                shared_out.push(tee_reading(h.curve, f, &fitted, [p.patch, qi], [p.side, k], p.forward, crease_of(h.curve)));
            }
        }
    }
    // An irregular corner's angle: the most between two patches across one of its curves
    // within the two spans next to it, where the tie was let go — at the corner itself the
    // curves' common tangent plane holds them alike.
    let corners: Vec<IrregularCorner> = irregular
        .iter()
        .map(|&node| {
            let at = corner_reading(node, node_of.get(&node).copied().unwrap_or(DVec3::NAN), &fitted);
            IrregularCorner { angle: at.angle.max(near_corner.get(&node).copied().unwrap_or(0.0)), ..at }
        })
        .collect();
    // Each patch side beside a body's face: the angle to the face along it, and the curve's
    // stand-off from the edge it was fitted to.
    let mut bodies_out: Vec<BodyReading> = Vec::new();
    for side in &input.body_sides {
        let Some(f) = fits.get(&side.curve) else { continue };
        for u in shared_by.get(&side.curve).map(|u| u.as_slice()).unwrap_or(&[]) {
            let ts: Vec<f64> = (0..EDGE_SAMPLES - 1).map(|k| (k + 1) as f64 / EDGE_SAMPLES as f64).collect();
            let points: Vec<DVec3> = ts.iter().map(|&t| curve_point(&f.poles, f.spans, t)).collect();
            let normals = (side.normals)(&points).unwrap_or_default();
            let mut angle = 0.0f64;
            for (k, &t) in ts.iter().enumerate() {
                let Some(&nf) = normals.get(k) else { continue };
                let (uu, vv) = on_side(u.side, if u.forward { t } else { 1.0 - t });
                let np = patch_normal(&fitted[u.patch].net, uu, vv);
                let cos = dot(np, nf).abs().min(1.0);
                angle = angle.max(cos.acos().to_degrees_js());
            }
            bodies_out.push(BodyReading { curve: side.curve, patch: u.patch, mode: side.mode, angle, gap: f.max });
        }
    }
    NetworkFitResult {
        patches: fitted,
        curves: fits,
        shared: shared_out,
        bodies: bodies_out,
        corners,
        strips: strips.curves.iter().enumerate().map(|(st, curves)| Strip { spans: strip_spans[st], auto: strips.auto[st], curves: curves.clone() }).collect(),
        rounds,
        iterations,
    }
}

/// Radians to degrees as ScanRuler writes it, `(x * 180) / Math.PI`.
trait JsDegrees {
    fn to_degrees_js(self) -> f64;
}

impl JsDegrees for f64 {
    fn to_degrees_js(self) -> f64 {
        (self * 180.0) / std::f64::consts::PI
    }
}

// ---------------------------------------------------------------------------------------
// The patches' points.

/// A patch's points to fit: up to a cap of them, taken evenly.
#[derive(Debug, Clone)]
struct PatchData {
    count: usize,
    xyz: Vec<f64>,
    uv: Vec<f64>,
    keep: Vec<bool>,
    kept: usize,
}

/// How far a patch strays from its points beyond their scatter (above). The scatter is read
/// inside cells of the patch's (u, v) so small — a few points each, MISFIT_POINTS on the
/// average, as square as the sides' lengths make them — that a shape the net cannot follow
/// hardly changes across one: the residuals' variance about their cell's mean, σ². What is
/// left of their mean square past it is the misfit's: M² = RMS² − σ². Read off the means of
/// cells half a span wide instead, it missed most of it: what a net cannot follow swings to
/// and fro within each span, and a 270° strip of a cylinder at four spans read 0.004 mm where
/// it stood 0.045 off.
fn misfit_of(d: &PatchData, r: &[f64], aspect: f64) -> f64 {
    let mut n = 0usize;
    let mut sq = 0.0;
    for k in 0..d.count {
        if !d.keep[k] {
            continue;
        }
        sq += r[k] * r[k];
        n += 1;
    }
    if n == 0 {
        return 0.0;
    }
    let cells = n as f64 / MISFIT_POINTS;
    let bu = js_round((cells * aspect).sqrt()).max(1.0);
    let bv = js_round((cells / aspect).sqrt()).max(1.0);
    let (bu, bv) = (if bu.is_finite() { bu as usize } else { 1 }, if bv.is_finite() { bv as usize } else { 1 });
    let mut sum = vec![0.0f64; bu * bv];
    let mut sum_sq = vec![0.0f64; bu * bv];
    let mut count = vec![0usize; bu * bv];
    let cell_of = |x: f64, b: usize| -> usize { (x * b as f64).floor().max(0.0).min((b - 1) as f64) as usize };
    for k in 0..d.count {
        if !d.keep[k] {
            continue;
        }
        let c = cell_of(d.uv[k * 2], bu) * bv + cell_of(d.uv[k * 2 + 1], bv);
        sum[c] += r[k];
        sum_sq[c] += r[k] * r[k];
        count[c] += 1;
    }
    let mut within = 0.0;
    let mut free = 0usize;
    for c in 0..sum.len() {
        if count[c] < 2 {
            continue;
        }
        within += sum_sq[c] - (sum[c] * sum[c]) / count[c] as f64;
        free += count[c] - 1;
    }
    if free == 0 {
        return (sq / n as f64).sqrt();
    }
    // The estimate scatters too: on noise alone it reads some (8 / n)^¼ of σ at twice its
    // spread — 0.002 mm on a sheet with 0.02 of noise — and a tolerance under that had Auto
    // chase it to twenty spans. Within it, the misfit is read as none.
    let scatter = within / free as f64;
    let misfit = (sq / n as f64 - scatter).max(0.0).sqrt();
    if misfit > scatter.sqrt() * (8.0 / n as f64).powf(0.25) {
        misfit
    } else {
        0.0
    }
}

/// A patch's points: up to `cap` of them, taken evenly. Copied, never shared even when all
/// are taken: the caller keeps the input's arrays, and the fit moves its points' (u, v).
fn take_evenly(p: &PatchInput, cap: usize) -> PatchData {
    let all = p.uv.len() / 2;
    let step = (all as f64 / cap.max(1) as f64).max(1.0);
    let count = all.min((all as f64 / step).ceil() as usize);
    let mut xyz = vec![0.0; count * 3];
    let mut uv = vec![0.0; count * 2];
    for k in 0..count {
        let i = (all - 1).min((k as f64 * step).floor() as usize);
        xyz[k * 3..k * 3 + 3].copy_from_slice(&p.xyz[i * 3..i * 3 + 3]);
        uv[k * 2..k * 2 + 2].copy_from_slice(&p.uv[i * 2..i * 2 + 2]);
    }
    PatchData { count, xyz, uv, keep: vec![true; count], kept: count }
}

// ---------------------------------------------------------------------------------------
// The bands.

/// A point's terms added to a patch's band and right-hand side, for the cubic: four by four
/// poles, each pair's term (w bu[a1] bv[b1] bu[a2]) bv[b2], its first factors taken once for
/// the four of a row — the products the general loop formed, in its order, so the band is
/// the same to the bit. Every point comes through here every assembly.
#[allow(clippy::too_many_arguments)]
fn add_point(h: &mut [f64], g: &mut [f64], cols: usize, su: usize, sv: usize, bu: &[f64; 4], bv: &[f64; 4], x: f64, y: f64, z: f64, weight: f64) {
    let (bv0, bv1, bv2, bv3) = (bv[0], bv[1], bv[2], bv[3]);
    for a1 in 0..4 {
        let wa = weight * bu[a1];
        for b1 in 0..4 {
            let w1 = wa * bv[b1];
            if w1 == 0.0 {
                continue;
            }
            let p1 = (su - 3 + a1) * cols + sv - 3 + b1;
            g[p1 * 3] += w1 * x;
            g[p1 * 3 + 1] += w1 * y;
            g[p1 * 3 + 2] += w1 * z;
            // Row a2 − a1 + 3 of the pole's 7 × 7, from column 3 − b1.
            let row = p1 * BAND + 3 - b1;
            for (a2, &bua2) in bu.iter().enumerate() {
                let w = w1 * bua2;
                let at = row + (a2 + 3 - a1) * 7;
                h[at] += w * bv0;
                h[at + 1] += w * bv1;
                h[at + 2] += w * bv2;
                h[at + 3] += w * bv3;
            }
        }
    }
}

/// out += H y over a patch's poles, three coordinates, H the 7 × 7 band. The conjugate
/// gradient spends most of its time here. Each pole's band rows and columns inside the net
/// are found once, as offsets 0 … 6, and a zero entry is added, not tested for: a sum begun
/// at +0 is the same to the bit for it.
fn band_multiply(h: &[f64], nu: usize, nv: usize, y: &[f64], out: &mut [f64]) {
    let rows = nu + 3;
    let cols = nv + 3;
    for i in 0..rows {
        let a0 = 3usize.saturating_sub(i);
        let a1 = 6.min(rows + 2 - i);
        for j in 0..cols {
            let b0 = 3usize.saturating_sub(j);
            let b1 = 6.min(cols + 2 - j);
            let p = i * cols + j;
            let row = p * BAND;
            let mut sx = 0.0;
            let mut sy = 0.0;
            let mut sz = 0.0;
            for a in a0..=a1 {
                let h0 = row + a * 7;
                // Row i + a − 3 of the net; a ≥ a0 keeps it inside.
                let q0 = (i + a - 3) * cols * 3;
                for b in b0..=b1 {
                    let hv = h[h0 + b];
                    let q = q0 + (j + b - 3) * 3;
                    sx += hv * y[q];
                    sy += hv * y[q + 1];
                    sz += hv * y[q + 2];
                }
            }
            out[p * 3] += sx;
            out[p * 3 + 1] += sy;
            out[p * 3 + 2] += sz;
        }
    }
}

/// The fairing's band for a patch: Fit Surface's third divided differences over the Greville
/// abscissae, along u, along v and mixed, each weighted by its net cell's area and by the
/// patch's sides' lengths for its size — in the patch's [0, 1]² a third derivative along u
/// stands for one along the side scaled by its length cubed, so a slider of 1 means here what
/// it means to Fit Surface over the same region.
fn fairing_band(net: &PatchNet, smoothness: f64, rows: &[FittedSide]) -> Vec<f64> {
    let rows_n = net.nu + 3;
    let cols = net.nv + 3;
    let mut h = vec![0.0f64; rows_n * cols * BAND];
    if !(smoothness > 0.0) || rows.len() < 4 {
        return h;
    }
    let length_of = |poles: &[f64]| -> f64 {
        let mut l = 0.0;
        for m in 1..poles.len() / 3 {
            l += hypot3(poles[m * 3] - poles[m * 3 - 3], poles[m * 3 + 1] - poles[m * 3 - 2], poles[m * 3 + 2] - poles[m * 3 - 1]);
        }
        l
    };
    let lu = ((length_of(&rows[0].poles) + length_of(&rows[2].poles)) / 2.0).max(1e-6);
    let lv = ((length_of(&rows[1].poles) + length_of(&rows[3].poles)) / 2.0).max(1e-6);
    let s = smoothness * SMOOTHNESS_SCALE;
    let area = (1.0 / (rows_n as f64 - 1.0).max(1.0)) * (1.0 / (cols as f64 - 1.0).max(1.0));
    let xu = greville_of(net.nu);
    let xv = greville_of(net.nv);
    let mut add = |idx: &[usize], w: &[f64], weight: f64| {
        for a in 0..idx.len() {
            let ia = idx[a] / cols;
            let ja = idx[a] % cols;
            for b in 0..idx.len() {
                let ib = idx[b] / cols;
                let jb = idx[b] % cols;
                h[idx[a] * BAND + (ib + 3 - ia) * 7 + (jb + 3 - ja)] += weight * w[a] * w[b];
            }
        }
    };
    let mut cross = |i: usize, j: usize, wu: &[f64], wv: &[f64], weight: f64| {
        let mut idx: Vec<usize> = Vec::with_capacity(wu.len() * wv.len());
        let mut w: Vec<f64> = Vec::with_capacity(wu.len() * wv.len());
        for (a, &wa) in wu.iter().enumerate() {
            for (b, &wb) in wv.iter().enumerate() {
                idx.push((i + a) * cols + j + b);
                w.push(wa * wb);
            }
        }
        add(&idx, &w, weight);
    };
    let one = [1.0];
    for i in 0..rows_n {
        for j in 0..cols {
            if i + 3 < rows_n {
                cross(i, j, &divided_weights(&xu, i, 3), &one, s * area * (lv / lu).powf(3.0));
            }
            if j + 3 < cols {
                cross(i, j, &one, &divided_weights(&xv, j, 3), s * area * (lu / lv).powf(3.0));
            }
            if i + 2 < rows_n && j + 1 < cols {
                cross(i, j, &divided_weights(&xu, i, 2), &divided_weights(&xv, j, 1), 3.0 * s * area * (lv / lu));
            }
            if i + 1 < rows_n && j + 2 < cols {
                cross(i, j, &divided_weights(&xu, i, 1), &divided_weights(&xv, j, 2), 3.0 * s * area * (lu / lv));
            }
        }
    }
    h
}

/// The pull: net cells further than PULL_REACH cells from any held one get PULL_SAMPLES²
/// samples on the Coons surface of the patch's four boundary rows, weighing a share of an
/// average held cell's points.
fn add_pull(h: &mut [f64], g: &mut [f64], net: &PatchNet, held: &[bool], points: usize) {
    let held_count = held.iter().filter(|&&x| x).count();
    let near = |a: usize, b: usize| -> bool {
        for i in a.saturating_sub(PULL_REACH)..=(net.nu - 1).min(a + PULL_REACH) {
            for j in b.saturating_sub(PULL_REACH)..=(net.nv - 1).min(b + PULL_REACH) {
                if held[i * net.nv + j] {
                    return true;
                }
            }
        }
        false
    };
    let weight = (PULL * (points.max(1) as f64)) / (held_count.max(1) as f64) / ((PULL_SAMPLES * PULL_SAMPLES) as f64);
    let cols = net.nv + 3;
    let mut bu = [0.0f64; 4];
    let mut bv = [0.0f64; 4];
    for a in 0..net.nu {
        for b in 0..net.nv {
            if near(a, b) {
                continue;
            }
            for s in 0..PULL_SAMPLES {
                for t in 0..PULL_SAMPLES {
                    let u = (a as f64 + (s as f64 + 0.5) / PULL_SAMPLES as f64) / net.nu as f64;
                    let v = (b as f64 + (t as f64 + 0.5) / PULL_SAMPLES as f64) / net.nv as f64;
                    let target = coons_point(net, u, v);
                    let su = basis_at(net.nu, u, &mut bu);
                    let sv = basis_at(net.nv, v, &mut bv);
                    add_point(h, g, cols, su, sv, &bu, &bv, target.x, target.y, target.z, weight);
                }
            }
        }
    }
}

/// The Coons surface of a net's four boundary rows at (u, v).
fn coons_point(net: &PatchNet, u: f64, v: f64) -> DVec3 {
    let rows = net.nu + 3;
    let cols = net.nv + 3;
    let poles = &net.poles;
    let row = |j: usize| -> Vec<f64> {
        let mut out = vec![0.0; rows * 3];
        for i in 0..rows {
            out[i * 3..i * 3 + 3].copy_from_slice(&poles[(i * cols + j) * 3..(i * cols + j) * 3 + 3]);
        }
        out
    };
    let col = |i: usize| &poles[i * cols * 3..(i + 1) * cols * 3];
    let a = curve_point(&row(0), net.nu, u);
    let c = curve_point(&row(cols - 1), net.nu, u);
    let d = curve_point(col(0), net.nv, v);
    let e = curve_point(col(rows - 1), net.nv, v);
    let corner = |i: usize, j: usize| DVec3::new(poles[(i * cols + j) * 3], poles[(i * cols + j) * 3 + 1], poles[(i * cols + j) * 3 + 2]);
    let (p00, p10, p01, p11) = (corner(0, 0), corner(rows - 1, 0), corner(0, cols - 1), corner(rows - 1, cols - 1));
    let mut out = DVec3::ZERO;
    for k in 0..3 {
        out[k] = (1.0 - v) * a[k] + v * c[k] + (1.0 - u) * d[k] + u * e[k] - ((1.0 - u) * (1.0 - v) * p00[k] + u * (1.0 - v) * p10[k] + (1.0 - u) * v * p01[k] + u * v * p11[k]);
    }
    out
}

/// The poles of a net's Coons surface: the bilinearly blended surface of its four boundary
/// rows, which a B-spline over the same knots holds exactly — the blend's linear factors are
/// reproduced by poles at the Greville abscissae — so its boundary poles are the rows
/// themselves.
fn coons_poles(net: &PatchNet) -> Vec<f64> {
    let rows = net.nu + 3;
    let cols = net.nv + 3;
    let gu = greville_of(net.nu);
    let gv = greville_of(net.nv);
    let p = &net.poles;
    let mut out = p.clone();
    let at = |i: usize, j: usize, k: usize| p[(i * cols + j) * 3 + k];
    for i in 1..rows - 1 {
        let u = gu[i];
        for j in 1..cols - 1 {
            let v = gv[j];
            for k in 0..3 {
                out[(i * cols + j) * 3 + k] = (1.0 - v) * at(i, 0, k) + v * at(i, cols - 1, k) + (1.0 - u) * at(0, j, k) + u * at(rows - 1, j, k)
                    - ((1.0 - u) * (1.0 - v) * at(0, 0, k) + u * (1.0 - v) * at(rows - 1, 0, k) + (1.0 - u) * v * at(0, cols - 1, k) + u * v * at(rows - 1, cols - 1, k));
            }
        }
    }
    out
}

/// The ratios of the ties along each curve smoothed over their neighbours: a ratio read off
/// one pole pair is that pair's noise as well.
fn smooth_ratios(ties: &mut [Tie]) {
    let r: Vec<f64> = ties.iter().map(|t| t.ratio).collect();
    for i in 0..ties.len() {
        let mut s = r[i];
        let mut n = 1.0;
        for j in [i.wrapping_sub(1), i + 1] {
            if j >= ties.len() || ties[j].curve != ties[i].curve {
                continue;
            }
            s += r[j];
            n += 1.0;
        }
        ties[i].ratio = s / n;
    }
}

/// The corners, of those with ties next to them, where a patch turned more than CORNER_TURN
/// from the first fit's within its two spans of the corner — read on a few samples there,
/// the free fit's poles `free`. In the order found.
fn torn_corners(nets: &[PatchNet], free: &[f64], pole_base: &[usize], patches: &[PatchInput], tied: &HashSet<u32>) -> Vec<u32> {
    let mut torn: Vec<u32> = Vec::new();
    let cos = (CORNER_TURN.to_radians()).cos();
    const SAMPLES: usize = 4;
    for (i, p) in patches.iter().enumerate() {
        let net = &nets[i];
        let from = pole_base[i] * 3;
        let was = PatchNet { nu: net.nu, nv: net.nv, poles: free[from..from + net.poles.len()].to_vec() };
        'corner: for (k, &node) in p.corners.iter().enumerate() {
            if !tied.contains(&node) || torn.contains(&node) {
                continue;
            }
            // Corner k is at (0, 0), (1, 0), (1, 1), (0, 1) in turn.
            let cu = k == 1 || k == 2;
            let cv = k >= 2;
            for s in 0..=SAMPLES {
                for t in 0..=SAMPLES {
                    let du = ((2.0 / net.nu as f64) * s as f64) / SAMPLES as f64;
                    let dv = ((2.0 / net.nv as f64) * t as f64) / SAMPLES as f64;
                    let u = if cu { 1.0 - du } else { du };
                    let v = if cv { 1.0 - dv } else { dv };
                    let a = patch_normal(net, u, v);
                    let b = patch_normal(&was, u, v);
                    if dot(a, b) < cos {
                        torn.push(node);
                        continue 'corner;
                    }
                }
            }
        }
    }
    torn
}

/// Each point projected onto its patch: Gauss–Newton from its (u, v), held inside the square.
fn project(nets: &[PatchNet], data: &mut [PatchData]) {
    for (net, d) in nets.iter().zip(data.iter_mut()) {
        for k in 0..d.count {
            let mut u = d.uv[k * 2];
            let mut v = d.uv[k * 2 + 1];
            for _ in 0..NEWTON_STEPS {
                let (p, du, dv) = patch_point(net, u, v);
                let (rx, ry, rz) = (d.xyz[k * 3] - p.x, d.xyz[k * 3 + 1] - p.y, d.xyz[k * 3 + 2] - p.z);
                let a = du.x * du.x + du.y * du.y + du.z * du.z;
                let b = du.x * dv.x + du.y * dv.y + du.z * dv.z;
                let c = dv.x * dv.x + dv.y * dv.y + dv.z * dv.z;
                let gu = du.x * rx + du.y * ry + du.z * rz;
                let gv = dv.x * rx + dv.y * ry + dv.z * rz;
                let det = a * c - b * b;
                if !(det > 1e-300) {
                    break;
                }
                u = (u + (c * gu - b * gv) / det).clamp(0.0, 1.0);
                v = (v + (a * gv - b * gu) / det).clamp(0.0, 1.0);
            }
            d.uv[k * 2] = u;
            d.uv[k * 2 + 1] = v;
        }
    }
}

/// Each point's distance from its patch at its (u, v), signed along the patch's normal.
fn residuals(net: &PatchNet, d: &PatchData) -> Vec<f64> {
    let mut out = vec![0.0; d.count];
    for (k, o) in out.iter_mut().enumerate() {
        let (p, du, dv) = patch_point(net, d.uv[k * 2], d.uv[k * 2 + 1]);
        let nx = du.y * dv.z - du.z * dv.y;
        let ny = du.z * dv.x - du.x * dv.z;
        let nz = du.x * dv.y - du.y * dv.x;
        let rx = d.xyz[k * 3] - p.x;
        let ry = d.xyz[k * 3 + 1] - p.y;
        let rz = d.xyz[k * 3 + 2] - p.z;
        let dist = hypot3(rx, ry, rz);
        // Where the normal is zero so is the product: the sign is then +.
        *o = if rx * nx + ry * ny + rz * nz < 0.0 { -dist } else { dist };
    }
    out
}

/// Each patch's residuals (above), and their RMS over the points kept.
fn residual_rms(nets: &[PatchNet], data: &[PatchData]) -> (f64, Vec<Vec<f64>>) {
    let mut sq = 0.0;
    let mut n = 0usize;
    let res: Vec<Vec<f64>> = nets
        .iter()
        .zip(data)
        .map(|(net, d)| {
            let r = residuals(net, d);
            for (k, rk) in r.iter().enumerate() {
                if !d.keep[k] {
                    continue;
                }
                sq += rk * rk;
                n += 1;
            }
            r
        })
        .collect();
    (if n > 0 { (sq / n as f64).sqrt() } else { 0.0 }, res)
}

/// The points past ROBUST_SIGMAS robust sigmas of the fit left out.
fn drop_outliers(nets: &[PatchNet], data: &mut [PatchData]) {
    for (net, d) in nets.iter().zip(data.iter_mut()) {
        let r = residuals(net, d);
        let mut sorted = r.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let med = sorted.get(sorted.len() >> 1).copied().unwrap_or(0.0);
        let mut dev: Vec<f64> = r.iter().map(|x| (x - med).abs()).collect();
        dev.sort_by(|a, b| a.total_cmp(b));
        let limit = (ROBUST_SIGMAS * 1.4826 * dev.get(dev.len() >> 1).copied().unwrap_or(0.0)).max(ROBUST_FLOOR);
        let mut kept = 0;
        for k in 0..d.count {
            d.keep[k] = (r[k] - med).abs() <= limit;
            kept += d.keep[k] as usize;
        }
        d.kept = kept;
    }
}

// ---------------------------------------------------------------------------------------
// The readings.

/// The (u, v) on a patch of the point a fraction f along its side k.
pub fn on_side(k: usize, f: f64) -> (f64, f64) {
    match k {
        0 => (f, 0.0),
        1 => (1.0, f),
        2 => (1.0 - f, 1.0),
        _ => (0.0, 1.0 - f),
    }
}

fn angle_between(np: DVec3, nq: DVec3) -> f64 {
    let cos = dot(np, nq).clamp(-1.0, 1.0);
    cos.acos().to_degrees_js()
}

/// A shared curve's angle and gap, read at EDGE_SAMPLES points along it on both patches;
/// and, near an irregular corner, the angle there apart (from its start, to its end).
fn edge_reading(s: &SharedLink, fitted: &[FittedPatch], skip_from: f64, skip_to: f64) -> (SharedCurve, f64, f64) {
    let p = &fitted[s.patches[0]];
    let q = &fitted[s.patches[1]];
    let forward_p = p.sides.get(s.sides[0]).and_then(|x| x.edges.first()).is_none_or(|e| e.forward);
    let forward_q = q.sides.get(s.sides[1]).and_then(|x| x.edges.first()).is_none_or(|e| e.forward);
    let mut angle = 0.0f64;
    let mut gap = 0.0f64;
    let mut sum = 0.0;
    let mut count = 0usize;
    // Near an irregular corner the angle is read apart, for the corner's: finer there, where
    // it changes fastest.
    let mut near_from = 0.0f64;
    let mut near_to = 0.0f64;
    let angle_at = |t: f64| -> f64 {
        let (up, vp) = on_side(s.sides[0], if forward_p { t } else { 1.0 - t });
        let (uq, vq) = on_side(s.sides[1], if forward_q { t } else { 1.0 - t });
        angle_between(patch_normal(&p.net, up, vp), patch_normal(&q.net, uq, vq))
    };
    for k in 0..=EDGE_SAMPLES {
        let t = k as f64 / EDGE_SAMPLES as f64;
        let (up, vp) = on_side(s.sides[0], if forward_p { t } else { 1.0 - t });
        let (uq, vq) = on_side(s.sides[1], if forward_q { t } else { 1.0 - t });
        let (pp, _, _) = patch_point(&p.net, up, vp);
        let (pq, _, _) = patch_point(&q.net, uq, vq);
        gap = gap.max(hypot3(pp.x - pq.x, pp.y - pq.y, pp.z - pq.z));
        // The normals a hair inside the corners: at a corner a side's normal is its cross
        // curve's as much as its own. Near an irregular corner, not at all: that is the
        // corner's reading.
        if k == 0 || k == EDGE_SAMPLES || t < skip_from || t > 1.0 - skip_to {
            continue;
        }
        let a = angle_at(t);
        angle = angle.max(a);
        sum += a;
        count += 1;
    }
    // A crease's fold is its mean: each patch's slope at a side nothing holds is read off the
    // points within a span of it, and wanders by their noise — at 20 spans over 0.02 mm of
    // noise, a 30° fold read 26–35° along its length, its largest 34.6°. A tangent curve's is
    // its largest: that is what has to stay under half a degree.
    if s.crease && count > 0 {
        angle = sum / count as f64;
    }
    for k in 1..=EDGE_SAMPLES {
        if skip_from > 0.0 {
            near_from = near_from.max(angle_at((skip_from * k as f64) / EDGE_SAMPLES as f64));
        }
        if skip_to > 0.0 {
            near_to = near_to.max(angle_at(1.0 - (skip_to * k as f64) / EDGE_SAMPLES as f64));
        }
    }
    (SharedCurve { curve: s.curve, patches: s.patches, gap, angle, crease: s.crease, tee: false }, near_from, near_to)
}

/// The parameter of a curve's point nearest x: the nearest of a few samples, then
/// Gauss–Newton along it.
fn nearest_on(poles: &[f64], spans: usize, x: DVec3) -> f64 {
    let mut s = 0.0;
    let mut best = f64::INFINITY;
    for j in 0..=64 {
        let y = curve_point(poles, spans, j as f64 / 64.0);
        let dist = hypot3(y.x - x.x, y.y - x.y, y.z - x.z);
        if dist < best {
            best = dist;
            s = j as f64 / 64.0;
        }
    }
    for _ in 0..4 {
        let y = curve_point(poles, spans, s);
        let d = curve_derivative(poles, spans, s);
        let dd = d.x * d.x + d.y * d.y + d.z * d.z;
        if !(dd > 0.0) {
            break;
        }
        s = (s + ((x.x - y.x) * d.x + (x.y - y.y) * d.y + (x.z - y.z) * d.z) / dd).clamp(0.0, 1.0);
    }
    s
}

/// A T-junction's curve read along its length: at each sample of it, the first patch's edge
/// there (the curve itself) against the nearest point of the second patch's side — the gap
/// between them, and the angle between the two patches' normals.
fn tee_reading(curve: u32, fit: &CurveFit, fitted: &[FittedPatch], patches: [usize; 2], sides: [usize; 2], forward_p: bool, crease: bool) -> SharedCurve {
    let p = &fitted[patches[0]];
    let q = &fitted[patches[1]];
    let empty: Vec<f64> = Vec::new();
    let row = q.sides.get(sides[1]).map(|s| &s.poles).unwrap_or(&empty);
    let spans = if sides[1] % 2 == 0 { q.net.nu } else { q.net.nv };
    let mut gap = 0.0f64;
    let mut angle = 0.0f64;
    for k in 1..EDGE_SAMPLES {
        let t = k as f64 / EDGE_SAMPLES as f64;
        let x = curve_point(&fit.poles, fit.spans, t);
        let s = nearest_on(row, spans, x);
        let y = curve_point(row, spans, s);
        gap = gap.max(hypot3(y.x - x.x, y.y - x.y, y.z - x.z));
        let (up, vp) = on_side(sides[0], if forward_p { t } else { 1.0 - t });
        let (uq, vq) = on_side(sides[1], s);
        angle = angle.max(angle_between(patch_normal(&p.net, up, vp), patch_normal(&q.net, uq, vq)));
    }
    SharedCurve { curve, patches, gap, angle, crease, tee: true }
}

/// The largest angle between the normals of two patches meeting at an irregular corner, read
/// at the corner.
fn corner_reading(node: u32, at: DVec3, fitted: &[FittedPatch]) -> IrregularCorner {
    let mut normals: Vec<DVec3> = Vec::new();
    for p in fitted {
        let Some(k) = p.corners.iter().position(|&c| c == node) else { continue };
        let (u, v) = on_side(k, 0.0);
        normals.push(patch_normal(&p.net, u, v));
    }
    let mut angle = 0.0f64;
    for a in 0..normals.len() {
        for b in a + 1..normals.len() {
            angle = angle.max(angle_between(normals[a], normals[b]));
        }
    }
    IrregularCorner { node, at, patches: normals.len(), angle }
}

#[cfg(test)]
mod tests {
    //! ScanRuler's `plugins/reverse/tests/reNetworkFit.test.ts`: a patch network's joint fit
    //! (PN8). A 2 × 2 network on a noisy sphere comes out tangent across every curve, within
    //! half a degree, and at the scan's noise; the patches share the curves between them
    //! exactly; a crease keeps its fold; a network read again keeps what did not change, with
    //! the same answer as a read afresh; a strip round 270° of a cylinder fits as one patch at
    //! the noise — the regression test for the fold-back the network is for; a 3 × 3 network
    //! drawn askew lets go of the ties at its inner corners, which fought each other round
    //! them and folded the patches over themselves there, 2.8 mm off the scan — where one
    //! drawn square keeps them all; a higher smoothness smooths the patches, curves and all;
    //! and Auto gives a strip four spans, and more only where a patch on it misses its points
    //! by more than the tolerance beyond their noise, while that helps — six round a 270°
    //! strip, four up it.

    use std::f64::consts::PI;
    use std::sync::OnceLock;

    use super::*;
    use crate::network::CellOutcome;
    use crate::network_meshes::{curve, curve_via, grid_network, grid_scan, network, sphere_scan, GridOptions, TestMesh};

    const R: f64 = 40.0;
    const NOISE: f64 = 0.02;

    /// A spot on the sphere by (u, v) over a square of it a cube's face wide.
    fn on_sphere(u: f64, v: f64) -> DVec3 {
        let a = (u - 0.5) * 1.4;
        let b = (v - 0.5) * 1.4;
        let l = hypot3(1.0, a, b);
        DVec3::new(R / l, (R * a) / l, (R * b) / l)
    }

    fn read(scan: &ScanPaths, f: &NetworkFeature) -> NetworkReading<NetworkFitResult> {
        match read_network_fitted(scan, f, None) {
            Ok(r) => r,
            Err(e) => panic!("{}", e.0),
        }
    }

    fn fit_of(scan: &ScanPaths, f: &NetworkFeature) -> NetworkFitResult {
        read(scan, f).fit.expect("a fit")
    }

    fn poles(fit: &NetworkFitResult) -> Vec<Vec<f64>> {
        fit.patches.iter().map(|p| p.net.poles.clone()).collect()
    }

    #[test]
    fn fits_a_2x2_network_on_a_noisy_sphere_tangent_and_at_the_noise() {
        let m = sphere_scan(R, NOISE, 6);
        let scan = ScanPaths::new(m.positions.clone(), m.indices.clone());
        let f = grid_network(2, 2, &on_sphere, 1, [0.0, 1.0, 0.0, 1.0]);
        let r = read(&scan, &f);
        assert!(r.outcomes.iter().all(|o| matches!(o, CellOutcome::Filled { .. })), "{:?}", r.outcomes);
        let fit = r.fit.as_ref().expect("a fit");
        assert_eq!(fit.patches.len(), 4);
        for p in &fit.patches {
            assert!(p.rms > 0.6 * NOISE, "rms {}", p.rms);
            assert!(p.rms < 1.4 * NOISE, "rms {}", p.rms);
        }
        // The four inner curves are shared, and the patches meet on them exactly, tangent.
        assert_eq!(fit.shared.len(), 4);
        for s in &fit.shared {
            assert!(s.gap < 1e-9, "gap {}", s.gap);
            assert!(s.angle < 0.5, "angle {}", s.angle);
        }
        // Read again, as the preview does at an edit: the traces and the maps kept, the
        // answer the same to the bit. A node moved changes the patches round it, and the read
        // still matches one made afresh.
        let again = read(&scan, &f);
        assert!(again.ms.map < r.ms.map / 4.0 + 1.0, "{} against {}", again.ms.map, r.ms.map);
        assert_eq!(poles(again.fit.as_ref().expect("a fit")), poles(fit));
        let mut moved = f.clone();
        moved.nodes[4].at = on_sphere(0.55, 0.45);
        let kept = fit_of(&scan, &moved);
        let fresh = fit_of(&ScanPaths::new(m.positions, m.indices), &moved);
        assert_eq!(poles(&kept), poles(&fresh));
        assert_ne!(poles(&kept), poles(fit));
    }

    #[test]
    fn keeps_a_crease_folded() {
        // Two flat faces meeting at 30° along x = 0, a 1 × 2 network across the fold with the
        // middle curve on it marked a crease.
        let fold = |u: f64, v: f64| -> DVec3 {
            let x = 80.0 * u - 40.0;
            DVec3::new(x, 60.0 * v, x.abs() * (15.0 * PI / 180.0).tan())
        };
        let m = grid_scan(160, 120, &fold, &GridOptions { noise: NOISE, ..Default::default() });
        let scan = ScanPaths::new(m.positions, m.indices);
        let mut f = grid_network(1, 2, &fold, 1, [0.1, 0.9, 0.1, 0.9]);
        let at = |f: &NetworkFeature, id: u32| f.nodes.iter().find(|n| n.id == id).map(|n| n.at).unwrap_or(DVec3::NAN);
        let middle = f.curves.iter().position(|c| at(&f, c.from).x.abs() < 1e-6 && at(&f, c.to).x.abs() < 1e-6).expect("the middle curve");
        let id = f.curves[middle].id;
        f.curves[middle].crease = Some(true);
        let fit = fit_of(&scan, &f);
        let shared = fit.shared.iter().find(|s| s.curve == id).expect("shared");
        assert!(shared.crease);
        assert!(shared.angle > 28.0 && shared.angle < 32.0, "angle {}", shared.angle);
        for p in &fit.patches {
            assert!(p.rms < 1.5 * NOISE, "rms {}", p.rms);
        }
        // Unmarked, the same curve is held tangent and the fit cannot follow the fold.
        f.curves[middle].crease = Some(false);
        let smooth = fit_of(&scan, &f);
        assert!(smooth.shared.iter().find(|s| s.curve == id).expect("shared").angle < 0.5);
        let worst = smooth.patches.iter().fold(f64::NEG_INFINITY, |m, p| m.max(p.max));
        assert!(worst > 5.0 * NOISE, "max {worst}");
    }

    /// The cylinder of radius 20 the strip tests scan, 40 tall, 180 × 40 cells round it.
    fn cylinder() -> &'static (TestMesh, ScanPaths) {
        static CYLINDER: OnceLock<(TestMesh, ScanPaths)> = OnceLock::new();
        CYLINDER.get_or_init(|| {
            let r = 20.0;
            let m = grid_scan(180, 40, &|u, v| DVec3::new(r * (2.0 * PI * u).cos(), r * (2.0 * PI * u).sin(), 40.0 * v), &GridOptions { wrap: true, noise: NOISE, ..Default::default() });
            let scan = ScanPaths::new(m.positions.clone(), m.indices.clone());
            (m, scan)
        })
    }

    fn on_cylinder(deg: f64, z: f64) -> DVec3 {
        let r = 20.0;
        DVec3::new(r * (deg * PI / 180.0).cos(), r * (deg * PI / 180.0).sin(), z)
    }

    /// One patch round 270° of the cylinder, 30 mm tall: its strip round the bend (curves 1
    /// and 3), its strip up it (2 and 4).
    fn strip() -> NetworkFeature {
        let on = on_cylinder;
        network(
            &[(1, on(0.0, 5.0)), (2, on(270.0, 5.0)), (3, on(270.0, 35.0)), (4, on(0.0, 35.0))],
            &[curve_via(1, 2, vec![on(90.0, 5.0), on(180.0, 5.0)]), curve(2, 3), curve_via(3, 4, vec![on(180.0, 35.0), on(90.0, 35.0)]), curve(4, 1)],
            &[on(135.0, 20.0)],
        )
    }

    #[test]
    fn fits_a_strip_round_270_degrees_of_a_cylinder_as_one_patch() {
        let (_, scan) = cylinder();
        let fit = fit_of(scan, &strip());
        assert_eq!(fit.patches.len(), 1);
        let p = &fit.patches[0];
        assert!(p.rms < 1.25 * NOISE, "rms {}", p.rms);
        // On the cylinder all the way round.
        let mut off = 0.0f64;
        for i in 0..=20 {
            for j in 0..=20 {
                let (q, _, _) = patch_point(&p.net, i as f64 / 20.0, j as f64 / 20.0);
                off = off.max((q.x.hypot(q.y) - 20.0).abs());
            }
        }
        assert!(off < 0.05, "off {off}");
    }

    // ---- a 3 × 3 network on a finely scanned sheet ----------------------------------

    const SIZE: f64 = 18.0;

    fn sheet(u: f64, v: f64) -> DVec3 {
        DVec3::new(SIZE * u, SIZE * v, 1.2 * (2.2 * u).sin() * (1.7 * v).cos() + 0.6 * u * v)
    }

    fn sheet_scan() -> &'static ScanPaths {
        static SHEET: OnceLock<ScanPaths> = OnceLock::new();
        SHEET.get_or_init(|| {
            let m = grid_scan(90, 90, &sheet, &GridOptions { noise: 0.004, seed: Some(7), ..Default::default() });
            ScanPaths::new(m.positions, m.indices)
        })
    }

    /// A 3 × 3 network on the sheet, its inner nodes pushed `skew` of the sheet's side off the
    /// square grid, alternately.
    fn grid(skew: f64) -> NetworkFeature {
        let id = |i: u32, j: u32| j * 4 + i + 1;
        let mut nodes: Vec<(u32, DVec3)> = Vec::new();
        for j in 0..4u32 {
            for i in 0..4u32 {
                let u = 0.1 + (0.8 * i as f64) / 3.0 + if i > 0 && i < 3 { skew * if j % 2 == 1 { 1.0 } else { -1.0 } } else { 0.0 };
                let v = 0.1 + (0.8 * j as f64) / 3.0 + if j > 0 && j < 3 { skew * if i % 2 == 1 { -1.0 } else { 1.0 } } else { 0.0 };
                nodes.push((id(i, j), sheet(u, v)));
            }
        }
        let at = |k: u32| nodes.iter().find(|n| n.0 == k).map(|n| n.1).unwrap_or(DVec3::NAN);
        let mut curves = Vec::new();
        for j in 0..4 {
            for i in 0..3 {
                curves.push(curve(id(i, j), id(i + 1, j)));
            }
        }
        for j in 0..3 {
            for i in 0..4 {
                curves.push(curve(id(i, j), id(i, j + 1)));
            }
        }
        let mut seeds = Vec::new();
        for j in 0..3 {
            for i in 0..3 {
                let c = [id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1)].map(at);
                seeds.push(DVec3::new((0..4).map(|k| c[k].x).sum::<f64>() / 4.0, (0..4).map(|k| c[k].y).sum::<f64>() / 4.0, (0..4).map(|k| c[k].z).sum::<f64>() / 4.0));
            }
        }
        network(&nodes, &curves, &seeds)
    }

    /// How many samples of the patches face against the patch's own way.
    fn folds(fit: &NetworkFitResult) -> usize {
        let mut n = 0;
        for p in &fit.patches {
            let mut mean = DVec3::ZERO;
            for a in 0..=10 {
                for b in 0..=10 {
                    mean += patch_normal(&p.net, a as f64 / 10.0, b as f64 / 10.0);
                }
            }
            for a in 0..=60 {
                for b in 0..=60 {
                    if dot(patch_normal(&p.net, a as f64 / 60.0, b as f64 / 60.0), mean) < 0.0 {
                        n += 1;
                    }
                }
            }
        }
        n
    }

    #[test]
    fn lets_go_of_the_ties_at_the_inner_corners_of_a_network_drawn_askew_which_they_folded() {
        let fit = fit_of(sheet_scan(), &grid(0.06));
        assert_eq!(folds(&fit), 0);
        for p in &fit.patches {
            assert!(p.rms < 0.006, "rms {}", p.rms);
            assert!(p.max < 0.05, "max {}", p.max);
        }
        // The four inner corners let go, reported with their angles; the curves tangent away
        // from them.
        let mut nodes: Vec<u32> = fit.corners.iter().map(|c| c.node).collect();
        nodes.sort_unstable();
        assert_eq!(nodes, vec![6, 7, 10, 11]);
        for c in &fit.corners {
            assert!(c.angle < 2.0, "corner {} at {}", c.node, c.angle);
        }
        for s in &fit.shared {
            assert!(s.angle < 0.5, "curve {} at {}", s.curve, s.angle);
        }
    }

    #[test]
    fn keeps_the_ties_at_the_corners_of_one_drawn_square() {
        let fit = fit_of(sheet_scan(), &grid(0.0));
        assert_eq!(folds(&fit), 0);
        assert_eq!(fit.corners, vec![]);
        for s in &fit.shared {
            assert!(s.angle < 0.1, "curve {} at {}", s.curve, s.angle);
        }
    }

    #[test]
    fn smooths_the_curves_with_the_patches_so_a_higher_smoothness_smooths_over_the_noise() {
        /// How much the normals wander from their neighbours' mean over the patches, RMS, deg
        /// — the scan's noise left in them.
        fn ripple(fit: &NetworkFitResult) -> f64 {
            const S: usize = 48;
            let mut sum = 0.0;
            let mut count = 0usize;
            for p in &fit.patches {
                let n: Vec<Vec<DVec3>> = (0..=S).map(|i| (0..=S).map(|j| patch_normal(&p.net, i as f64 / S as f64, j as f64 / S as f64)).collect()).collect();
                for i in 1..S {
                    for j in 1..S {
                        let m = n[i - 1][j] + n[i + 1][j] + n[i][j - 1] + n[i][j + 1];
                        let cos = dot(m, n[i][j]) / hypot3(m.x, m.y, m.z);
                        let a = cos.min(1.0).acos();
                        sum += a * a;
                        count += 1;
                    }
                }
            }
            ((sum / count as f64).sqrt() * 180.0) / PI
        }
        // Ten spans a strip, as many as the scan resolves: on a net that fine the noise has
        // the most room.
        let f = grid(0.0);
        let at = |smoothness: f64| {
            let mut g = f.clone();
            g.spans = g.curves.iter().map(|c| (c.id.to_string(), 10.0)).collect();
            g.fit = crate::network_feature::NetworkFitParams { smoothness, robust: None, tolerance: None };
            fit_of(sheet_scan(), &g)
        };
        let usual = at(0.001);
        let smooth = at(0.1);
        // Held to curves that followed their traces' noise whatever the smoothness, the
        // patches kept it: 0.054° at 0.001 and 0.057° at 0.1 here — now 0.007° and 0.002°.
        let (ru, rs) = (ripple(&usual), ripple(&smooth));
        assert!(rs < ru / 2.0, "{rs} against {ru}");
        assert!(ru < 0.02, "{ru}");
        for p in &smooth.patches {
            assert!(p.rms < 0.01, "rms {}", p.rms);
        }
        for s in &smooth.shared {
            assert!(s.angle < 0.1, "curve {} at {}", s.curve, s.angle);
        }
    }

    // ---- Auto's span counts ------------------------------------------------------

    fn spans_of(fit: &NetworkFitResult) -> [usize; 2] {
        let round = fit.strips.iter().find(|s| s.curves.contains(&1)).map(|s| s.spans).unwrap_or(0);
        let up = fit.strips.iter().find(|s| s.curves.contains(&2)).map(|s| s.spans).unwrap_or(0);
        [round, up]
    }

    #[test]
    fn starts_at_four_spans_and_adds_them_only_round_the_bend_where_they_bring_the_patch_to_the_scan() {
        let (_, scan) = cylinder();
        // Held at four, the patch cannot follow the bend: its misfit — what it misses the
        // points by beyond their noise — is most of its RMS.
        let mut held = strip();
        held.spans = [("1".to_string(), 4.0), ("2".to_string(), 4.0)].into_iter().collect();
        let four = fit_of(scan, &held);
        assert!(four.patches[0].misfit > 0.03, "misfit {}", four.patches[0].misfit);
        assert!(four.patches[0].rms > 2.0 * NOISE, "rms {}", four.patches[0].rms);
        // Auto: six round, four up it, within the tolerance and at the noise.
        let auto = fit_of(scan, &strip());
        assert_eq!(spans_of(&auto), [6, 4]);
        assert!(auto.strips.iter().all(|s| s.auto));
        assert!(auto.patches[0].misfit < DEFAULT_TOLERANCE, "misfit {}", auto.patches[0].misfit);
        assert!(auto.patches[0].rms < 1.1 * NOISE, "rms {}", auto.patches[0].rms);
    }

    #[test]
    fn goes_finer_for_a_finer_tolerance_no_finer_than_the_noise() {
        let (_, scan) = cylinder();
        // A tolerance of 2 µm: nine round, where what is left is the noise — read as no
        // misfit — not the twenty a misfit read off the noise itself chased it to.
        let mut f = strip();
        f.fit = crate::network_feature::NetworkFitParams { smoothness: 0.001, robust: None, tolerance: Some(0.002) };
        let fine = fit_of(scan, &f);
        assert_eq!(spans_of(&fine), [9, 4]);
        assert_eq!(fine.patches[0].misfit, 0.0);
    }

    #[test]
    fn reads_a_sheets_noise_as_no_misfit_and_keeps_it_at_four_spans() {
        let sheet = grid_scan(100, 100, &|u, v| DVec3::new(30.0 * u, 30.0 * v, 0.0), &GridOptions { noise: NOISE, ..Default::default() });
        let flat = ScanPaths::new(sheet.positions, sheet.indices);
        let square = network(
            &[(1, DVec3::new(3.0, 3.0, 0.0)), (2, DVec3::new(27.0, 3.0, 0.0)), (3, DVec3::new(27.0, 27.0, 0.0)), (4, DVec3::new(3.0, 27.0, 0.0))],
            &[curve(1, 2), curve(2, 3), curve(3, 4), curve(4, 1)],
            &[DVec3::new(15.0, 15.0, 0.0)],
        );
        for n in [4.0, 12.0] {
            let mut f = square.clone();
            f.spans = [("1".to_string(), n), ("2".to_string(), n)].into_iter().collect();
            assert_eq!(fit_of(&flat, &f).patches[0].misfit, 0.0, "at {n} spans");
        }
        let auto = fit_of(&flat, &square);
        assert_eq!((auto.patches[0].net.nu, auto.patches[0].net.nv), (4, 4));
    }

    // ---- the fit's own pieces ---------------------------------------------------------

    #[test]
    fn the_coons_poles_hold_a_bilinear_patch_and_the_band_is_its_product() {
        // Boundary rows on a hyperbolic paraboloid z = x y: the Coons surface of its four
        // boundary lines is the surface itself, at every (u, v).
        let (nu, nv) = (3, 5);
        let gu = greville_of(nu);
        let gv = greville_of(nv);
        let mut net = PatchNet { nu, nv, poles: vec![0.0; (nu + 3) * (nv + 3) * 3] };
        for i in 0..nu + 3 {
            for j in 0..nv + 3 {
                if i == 0 || j == 0 || i == nu + 2 || j == nv + 2 {
                    let (x, y) = (gu[i] * 10.0, gv[j] * 6.0);
                    net.poles[(i * (nv + 3) + j) * 3..(i * (nv + 3) + j) * 3 + 3].copy_from_slice(&[x, y, x * y / 10.0]);
                }
            }
        }
        net.poles = coons_poles(&net);
        for &(u, v) in &[(0.2, 0.3), (0.5, 0.5), (0.9, 0.1)] {
            let (p, _, _) = patch_point(&net, u, v);
            assert!((p - DVec3::new(10.0 * u, 6.0 * v, 6.0 * u * v)).length() < 1e-9, "{p}");
            assert!((coons_point(&net, u, v) - p).length() < 1e-9);
        }
        // A band built from one point's terms multiplies as the dense outer product does.
        let cols = nv + 3;
        let mut h = vec![0.0; (nu + 3) * cols * BAND];
        let mut g = vec![0.0; (nu + 3) * cols * 3];
        let (mut bu, mut bv) = ([0.0; 4], [0.0; 4]);
        let su = basis_at(nu, 0.4, &mut bu);
        let sv = basis_at(nv, 0.7, &mut bv);
        add_point(&mut h, &mut g, cols, su, sv, &bu, &bv, 1.0, 2.0, 3.0, 2.0);
        let y: Vec<f64> = (0..(nu + 3) * cols * 3).map(|i| (i as f64 * 0.37).sin()).collect();
        let mut out = vec![0.0; y.len()];
        band_multiply(&h, nu, nv, &y, &mut out);
        // Dense: w (Σ N y) N.
        let mut wy = [0.0; 3];
        for a in 0..4 {
            for b in 0..4 {
                let p = (su - 3 + a) * cols + sv - 3 + b;
                for c in 0..3 {
                    wy[c] += bu[a] * bv[b] * y[p * 3 + c];
                }
            }
        }
        for a in 0..4 {
            for b in 0..4 {
                let p = (su - 3 + a) * cols + sv - 3 + b;
                for c in 0..3 {
                    assert!((out[p * 3 + c] - 2.0 * bu[a] * bv[b] * wy[c]).abs() < 1e-12);
                }
            }
        }
    }

    #[test]
    fn refuses_a_patch_without_four_sides() {
        let input = NetworkFitInput { patches: vec![PatchInput { sides: vec![], corners: vec![], xyz: vec![], uv: vec![], seed: DVec3::ZERO }], curves: vec![], nodes: vec![], spans: BTreeMap::new(), smoothness: 0.001, robust: None, tolerance: None, spacing: None, body_sides: vec![] };
        match fit_network(&input) {
            Ok(_) => panic!("a patch without four sides fitted"),
            Err(e) => assert!(e.0.contains("four sides"), "{}", e.0),
        }
    }

    // ---- against a body (ScanRuler's reNetworkBodies.test.ts, PN11) ---------------------
    //
    // A half-cylinder modelled, the scan of it running on past its straight edge as a wall
    // that leans 6° off the cylinder's tangent there, and one patch on the wall whose top
    // curve runs along the body's edge. Held tangent to the cylinder's face, the patch leaves
    // the edge within half a degree of the face and bends to the wall below; held only on the
    // edge (contact), it takes the wall's 6°. The curve along the edge is the edge: its fit
    // stands off it by microns. ScanRuler models the body in the kernel (a half-disc
    // extruded) and finds its edge and face by their fingerprints; here the body is the same
    // half-cylinder read off analytically, behind `NetworkBodies`.

    const BODY_R: f64 = 15.0;

    fn lean() -> f64 {
        (6.0 * PI / 180.0).tan()
    }

    /// The half-cylinder x² + y² = R², y ≥ 0, 0 ≤ z ≤ 40, as the build's bodies read it:
    /// its straight edge at (R, 0) and its curved face.
    struct HalfCylinder;

    impl NetworkBodies for HalfCylinder {
        fn edge(&self, body: u32, _edge: &crate::network_feature::EdgeRef) -> Option<Vec<f32>> {
            (body == 2).then(|| vec![BODY_R as f32, 0.0, 0.0, BODY_R as f32, 0.0, 40.0])
        }

        fn normals(&self, body: u32, _face: &crate::network_feature::FaceRef, points: &[DVec3]) -> Option<Vec<DVec3>> {
            (body == 2).then(|| {
                points
                    .iter()
                    .map(|p| {
                        let a = p.y.atan2(p.x).clamp(0.0, PI);
                        DVec3::new(a.cos(), a.sin(), 0.0)
                    })
                    .collect()
            })
        }
    }

    /// The scan: the cylinder from 70° down to its edge at (R, 0), then the wall leaning out
    /// below it to y = −25, z from 0 to 40; u along that profile, v up z.
    fn body_scan() -> &'static ScanPaths {
        static SCAN: OnceLock<ScanPaths> = OnceLock::new();
        SCAN.get_or_init(|| {
            let arc = (BODY_R * 70.0 * PI) / 180.0;
            let wall = 25.0 * (1.0 + lean() * lean()).sqrt();
            let at = |u: f64, v: f64| -> DVec3 {
                let s = u * (arc + wall);
                let z = 40.0 * v;
                if s <= wall {
                    let d = (wall - s) / (1.0 + lean() * lean()).sqrt();
                    return DVec3::new(BODY_R + lean() * d, -d, z);
                }
                let a = (s - wall) / BODY_R;
                DVec3::new(BODY_R * a.cos(), BODY_R * a.sin(), z)
            };
            let m = grid_scan(120, 80, &at, &GridOptions { noise: 0.01, ..Default::default() });
            ScanPaths::new(m.positions, m.indices)
        })
    }

    fn holds_a_patch_against_a_cylinder_body(mode: SideMode) {
        use crate::network_feature::{CurveAlong, CurveSide, NetworkCell, NetworkCurve, NetworkFitParams, NetworkNode};
        use crate::topo::{EdgeFingerprint, EdgeKind, FaceFingerprint, FaceKind};
        let edge = EdgeFingerprint { kind: EdgeKind::Line, mid: DVec3::new(BODY_R, 0.0, 20.0), dir: DVec3::Z, length: 40.0, radius: None, faces: vec![], origin: 2 };
        let face = FaceFingerprint { kind: FaceKind::Cylinder, centroid: DVec3::new(0.0, 2.0 * BODY_R / PI, 20.0), dir: DVec3::Z, area: PI * BODY_R * 40.0, radius: Some(BODY_R), center: Some(DVec3::ZERO), origin: 2 };
        let wall_at = |z: f64| DVec3::new(BODY_R + lean() * 25.0, -25.0, z);
        let node = |id: u32, at: DVec3| NetworkNode { id, at, on: None, corner: None };
        let plain = |id: u32, from: u32, to: u32| NetworkCurve { id, from, to, via: vec![], crease: None, along: None, side: None };
        let f = NetworkFeature {
            nodes: vec![node(1, DVec3::new(BODY_R, 0.0, 5.0)), node(2, DVec3::new(BODY_R, 0.0, 35.0)), node(3, wall_at(35.0)), node(4, wall_at(5.0))],
            curves: vec![
                NetworkCurve { along: Some(CurveAlong { body: 2, edge }), side: Some(CurveSide { body: 2, face, mode }), ..plain(5, 1, 2) },
                plain(6, 2, 3),
                plain(7, 3, 4),
                plain(8, 4, 1),
            ],
            cells: vec![NetworkCell { seed: DVec3::new(BODY_R + lean() * 12.0, -12.0, 20.0), centre: None, corners: None }],
            spans: BTreeMap::new(),
            fit: NetworkFitParams { smoothness: 0.001, robust: None, tolerance: None },
        };
        let r = match read_network_fitted(body_scan(), &f, Some(&HalfCylinder)) {
            Ok(r) => r,
            Err(e) => panic!("{}", e.0),
        };
        let fit = r.fit.as_ref().unwrap_or_else(|| panic!("no fit: {:?}", r.outcomes));
        assert_eq!(fit.patches.len(), 1);
        let side = fit.bodies.iter().find(|x| x.curve == 5).expect("the side on the body");
        assert_eq!(side.mode, mode);
        // The curve along the edge is the edge.
        assert!(side.gap < 0.005, "gap {}", side.gap);
        match mode {
            SideMode::Tangent => assert!(side.angle < 0.5, "angle {}", side.angle),
            SideMode::Contact => assert!(side.angle > 4.0, "angle {}", side.angle),
        }
        // Below the edge the patch still follows the wall.
        let rms = fit.patches[0].rms;
        assert!(rms < if mode == SideMode::Tangent { 0.15 } else { 0.03 }, "rms {rms}");
    }

    #[test]
    fn holds_a_patch_tangent_to_a_cylinder_body() {
        holds_a_patch_against_a_cylinder_body(SideMode::Tangent);
    }

    #[test]
    fn holds_a_patch_on_the_edge_of_a_cylinder_body() {
        holds_a_patch_against_a_cylinder_body(SideMode::Contact);
    }
}
