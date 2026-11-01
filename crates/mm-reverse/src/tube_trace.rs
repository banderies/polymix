// SPDX-License-Identifier: AGPL-3.0-only
//! A bent tube's centreline read from the scan. Ported from ScanRuler
//! (`plugins/reverse/core/tubeTrace.ts`), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! From a point and a way, the scan is cut square to the tube station by station, each cut's
//! loop round the centre giving a centre and a size, until the loop is lost or its size jumps
//! (a bellows, a flange, a junction): that is said, never crossed. The centres are then fitted
//! as one chain of straights and tangent arcs, corners added where they stray most, and cut
//! again square to the chain, which is what a bend's radius is read from: a plane through the
//! bend's axis has the true section, the marching plane lags behind the bend and leans.

use std::collections::BTreeMap;
use std::f64::consts::PI;

use glam::DVec3;
use serde::{Deserialize, Serialize};

use crate::fit2d::solve_linear;
use crate::frame::SectionFrame;
use crate::js_fmt::{js_round, num, round_to};
use crate::loop_piece::normalize3;
use crate::slice::SectionCut;
use crate::sweep_route::{route_pieces, SweepRoute};

/// A station of the trace: a cut square to the tube and what its loop says.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TubeStation {
    /// How far along the centres from the first, mm.
    pub s: f64,
    pub at: DVec3,
    /// The way the plane cut this station stood.
    pub dir: DVec3,
    /// Widest and narrowest across the loop's principal axes, mm.
    pub major: f64,
    pub minor: f64,
    /// The widest way, in the plane.
    pub major_dir: DVec3,
    /// The diameter of a round of the loop's area, mm.
    pub size: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TubeStopKind {
    Length,
    Lost,
    Open,
    Size,
    Centre,
}

impl TubeStopKind {
    /// The kind as ScanRuler names it.
    pub fn name(self) -> &'static str {
        match self {
            TubeStopKind::Length => "length",
            TubeStopKind::Lost => "lost",
            TubeStopKind::Open => "open",
            TubeStopKind::Size => "size",
            TubeStopKind::Centre => "centre",
        }
    }
}

/// Where a trace stopped, and why in words.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TubeStop {
    pub kind: TubeStopKind,
    pub s: f64,
    pub at: DVec3,
    pub says: String,
}

/// A straight or a bend of the fitted chain, with the size of the tube there.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TubeRun {
    Straight { from: DVec3, to: DVec3, length: f64, major: f64, minor: f64, rms: f64 },
    Bend { corner: DVec3, radius: f64, turn_deg: f64, normal: DVec3, major: f64, minor: f64, rms: f64 },
}

impl TubeRun {
    pub fn kind_name(&self) -> &'static str {
        match self {
            TubeRun::Straight { .. } => "straight",
            TubeRun::Bend { .. } => "bend",
        }
    }
}

/// How far the station centres lie off the route, mm, and where the worst is.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TubeResidual {
    pub rms: f64,
    pub max: f64,
    pub at: DVec3,
}

/// What a trace read.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TubeTrace {
    pub stations: Vec<TubeStation>,
    /// Where it stopped and why: the one way, or with `both` the end the stations start from,
    /// then the end they run to.
    pub stops: Vec<TubeStop>,
    /// `None` with fewer than TRACE_MIN_STATIONS stations.
    pub route: Option<SweepRoute>,
    pub runs: Vec<TubeRun>,
    /// Station centres against the route, mm.
    pub residual: Option<TubeResidual>,
    pub warnings: Vec<String>,
    pub cuts: usize,
}

/// Where a trace starts and how far it goes.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TraceInput {
    pub start: DVec3,
    pub dir: DVec3,
    /// Follow the tube the other way from `start` too.
    pub both: bool,
    /// The step between stations, mm; an eighth of the tube's size by default.
    pub step_mm: Option<f64>,
    /// How far to follow it at most, mm; 3000 by default.
    pub max_length_mm: Option<f64>,
}

impl TraceInput {
    pub fn new(start: DVec3, dir: DVec3) -> Self {
        Self { start, dir, both: false, step_mm: None, max_length_mm: None }
    }
}

/// A loop's size off the running size by more than this share is a stop: no bent tube's
/// section changes so much; a fold, a flange or a branch does.
pub const SIZE_JUMP: f64 = 0.15;
/// Off by more than this share is said, and the trace goes on: a cuff.
const SIZE_STEP: f64 = 0.04;
/// A centre off where it was expected by more than this share of the size is a stop: the loop
/// is another's, or the tube ended in a joint.
const CENTRE_JUMP: f64 = 0.3;
/// How many stations the running size is the median of.
const RUNNING: usize = 8;
/// How many stations the next plane is reckoned from.
const AHEAD: usize = 6;
/// Fewer stations than this are no centreline.
const TRACE_MIN_STATIONS: usize = 4;
/// A bend is looked for between chords this many sizes long either side of a station, turned
/// by more than BEND_SPAN_TURN_DEG: at half a size and 5°, a bend tighter than about six sizes
/// in radius, clear of a surface's lumps.
const BEND_SPAN: f64 = 0.5;
const BEND_SPAN_TURN_DEG: f64 = 5.0;
/// Corners at most: past this it is no chain of straights and bends.
const MAX_CORNERS: usize = 12;
/// A fitted corner turning less than this is none, degrees.
const MIN_TURN_DEG: f64 = 3.0;
/// A corner has to bring the worst station down by this share to stay.
const CORNER_GAIN: f64 = 0.2;
/// How often the stations are cut again square to the fit.
const RECUTS: usize = 2;
/// Worst station off the route over this, as a share of the size, is said.
const RESIDUAL_WARN: f64 = 0.02;

#[derive(Debug, Clone, Copy)]
struct Round {
    at: DVec3,
    major: f64,
    minor: f64,
    major_dir: DVec3,
    size: f64,
}

/// What a cut gave round the centre.
enum Found {
    Round(Round),
    Open,
    None,
}

/// The tube's loop on a cut through `at`: the smallest closed chain round it with its
/// material on the inside — a scanned bore has it outside — or why there is none: an open
/// chain near it, or nothing.
fn round_at(cut: &SectionCut, at: DVec3, normal: DVec3, near: f64) -> Found {
    let Some(frame) = SectionFrame::sketch_on(at, normal, None) else { return Found::None };
    let (u, v) = (frame.u, frame.v);
    let mut best: Option<(f64, Vec<[f64; 2]>)> = None;
    let mut open = false;
    for (c, chain) in cut.chains.iter().enumerate() {
        let n = chain.len();
        if n < 3 {
            continue;
        }
        let xy: Vec<[f64; 2]> = chain.iter().map(|p| [(*p - at).dot(u), (*p - at).dot(v)]).collect();
        let closed = xy[0][0] == xy[n - 1][0] && xy[0][1] == xy[n - 1][1];
        if !closed {
            if !open && xy.iter().any(|q| q[0].hypot(q[1]) < near) {
                open = true;
            }
            continue;
        }
        let mut inside = false;
        let mut twice = 0.0;
        for i in 0..n - 1 {
            let ([x0, y0], [x1, y1]) = (xy[i], xy[i + 1]);
            twice += x0 * y1 - x1 * y0;
            if (y0 > 0.0) != (y1 > 0.0) && 0.0 < x0 - (y0 * (x1 - x0)) / (y1 - y0) {
                inside = !inside;
            }
        }
        if !inside {
            continue;
        }
        // Which side the material is on, where the cut carries it.
        if let Some(normals) = cut.normals.get(c) {
            if normals.len() == n {
                let outward: f64 = (0..n).map(|i| xy[i][0] * normals[i].dot(u) + xy[i][1] * normals[i].dot(v)).sum();
                if outward < 0.0 {
                    continue;
                }
            }
        }
        let area = twice.abs() / 2.0;
        if best.as_ref().is_none_or(|b| area < b.0) {
            best = Some((area, xy));
        }
    }
    let Some((_, xy)) = best else { return if open { Found::Open } else { Found::None } };
    // The loop's centre and principal axes, by its area's moments.
    let n = xy.len();
    let (mut a, mut cx, mut cy, mut xx, mut yy, mut xy2) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for i in 0..n - 1 {
        let ([x0, y0], [x1, y1]) = (xy[i], xy[i + 1]);
        let w = x0 * y1 - x1 * y0;
        a += w;
        cx += (x0 + x1) * w;
        cy += (y0 + y1) * w;
        xx += (x0 * x0 + x0 * x1 + x1 * x1) * w;
        yy += (y0 * y0 + y0 * y1 + y1 * y1) * w;
        xy2 += (x0 * y1 + 2.0 * x0 * y0 + 2.0 * x1 * y1 + x1 * y0) * w;
    }
    a /= 2.0;
    cx /= 6.0 * a;
    cy /= 6.0 * a;
    let mxx = xx / (12.0 * a) - cx * cx;
    let myy = yy / (12.0 * a) - cy * cy;
    let mxy = xy2 / (24.0 * a) - cx * cy;
    let angle = 0.5 * (2.0 * mxy).atan2(mxx - myy);
    let (ex, ey) = (angle.cos(), angle.sin());
    let (mut lo1, mut hi1, mut lo2, mut hi2) = (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
    for q in &xy {
        let p = q[0] * ex + q[1] * ey;
        let r = -q[0] * ey + q[1] * ex;
        lo1 = lo1.min(p);
        hi1 = hi1.max(p);
        lo2 = lo2.min(r);
        hi2 = hi2.max(r);
    }
    let (w1, w2) = (hi1 - lo1, hi2 - lo2);
    let along = u * ex + v * ey;
    Found::Round(Round {
        at: at + u * cx + v * cy,
        major: w1.max(w2),
        minor: w1.min(w2),
        major_dir: if w1 >= w2 { along } else { normal.cross(along) },
        size: ((4.0 * a.abs()) / PI).sqrt(),
    })
}

/// The upper median, as ScanRuler takes it: the middle of the sorted values, the later of two.
fn median(xs: &[f64]) -> f64 {
    let mut s = xs.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    s.get(s.len() >> 1).copied().unwrap_or(f64::NAN)
}

/// Where the next plane stands, and which way: the last stations' centres as a parabola along
/// s, read a step on. Before there are enough, straight on.
fn reckon(stations: &[TubeStation], way: DVec3, step: f64) -> (DVec3, DVec3) {
    let last = stations[stations.len() - 1];
    let n = AHEAD.min(stations.len());
    if n < 4 {
        return (last.at + way * step, way);
    }
    let recent = &stations[stations.len() - n..];
    let mut a = [0.0f64; 9];
    let mut rhs = [[0.0f64; 3]; 3];
    for x in recent {
        let t = x.s - last.s;
        let row = [1.0, t, t * t];
        for i in 0..3 {
            for j in 0..3 {
                a[i * 3 + j] += row[i] * row[j];
            }
            for k in 0..3 {
                rhs[k][i] += row[i] * x.at[k];
            }
        }
    }
    let mut fit: Vec<Vec<f64>> = Vec::with_capacity(3);
    for b in &rhs {
        let mut m = a;
        match solve_linear(3, &mut m, b) {
            Some(f) => fit.push(f),
            None => return (last.at + way * step, way),
        }
    }
    let dir = normalize3(DVec3::new(fit[0][1] + 2.0 * fit[0][2] * step, fit[1][1] + 2.0 * fit[1][2] * step, fit[2][1] + 2.0 * fit[2][2] * step)).unwrap_or(way);
    (last.at + dir * step, dir)
}

// ---- the chain fitted ------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct Corner {
    at: DVec3,
    radius: f64,
    turn: f64,
    /// The arc's ends and centre; on a corner with no arc, all `at`.
    enter: DVec3,
    exit: DVec3,
    centre: DVec3,
}

#[derive(Debug, Clone)]
struct Chain {
    ends: [DVec3; 2],
    corners: Vec<Corner>,
}

/// A chain from its points and radii. A radius too big for its straights is brought down to
/// what they hold, with its neighbour's, so a fit passing through such a place still has a
/// chain to measure.
fn chain_of(points: &[DVec3], radii: &[f64]) -> Chain {
    let n = points.len();
    let mut dirs: Vec<DVec3> = Vec::with_capacity(n);
    let mut runs: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n.saturating_sub(1) {
        let d = points[i + 1] - points[i];
        runs.push(d.length());
        dirs.push(normalize3(d).unwrap_or(DVec3::X));
    }
    let turns: Vec<f64> = (0..radii.len()).map(|j| dirs[j].dot(dirs[j + 1]).clamp(-1.0, 1.0).acos()).collect();
    let mut reach: Vec<f64> = radii.iter().enumerate().map(|(j, r)| r.max(0.0) * (turns[j].min(3.0) / 2.0).tan()).collect();
    for j in 0..radii.len() {
        // The straight before this corner, shared with the corner before.
        let prev = if j > 0 { reach[j - 1] } else { 0.0 };
        let room = runs[j];
        if prev + reach[j] > room {
            let k = room / (prev + reach[j]);
            if j > 0 {
                reach[j - 1] *= k;
            }
            reach[j] *= k;
        }
        if j == radii.len() - 1 && reach[j] > runs[j + 1] {
            reach[j] = runs[j + 1];
        }
    }
    let corners = (0..radii.len())
        .map(|j| {
            let at = points[j + 1];
            let turn = turns[j];
            let slope = (turn.min(3.0) / 2.0).tan();
            let radius = if slope > 1e-9 { reach[j] / slope } else { 0.0 };
            let inward = normalize3(dirs[j + 1] - dirs[j]);
            let enter = at + dirs[j] * -reach[j];
            let exit = at + dirs[j + 1] * reach[j];
            Corner { at, radius, turn, enter, exit, centre: inward.map_or(at, |w| at + w * (radius / (turn / 2.0).cos().max(1e-6))) }
        })
        .collect();
    Chain { ends: [points[0], points[n - 1]], corners }
}

/// The nearest point of a segment to q; the chain's ends run on as lines.
fn on_segment(q: DVec3, a: DVec3, b: DVec3, open_start: bool, open_end: bool) -> DVec3 {
    let d = b - a;
    let l2 = d.dot(d);
    if l2 < 1e-18 {
        return a;
    }
    let mut t = (q - a).dot(d) / l2;
    if !open_start && t < 0.0 {
        t = 0.0;
    }
    if !open_end && t > 1.0 {
        t = 1.0;
    }
    a + d * t
}

/// The nearest point of a corner's arc to q.
fn on_arc(q: DVec3, c: &Corner) -> DVec3 {
    let e1 = c.enter - c.centre;
    let e2 = c.exit - c.centre;
    let Some(axis) = normalize3(e1.cross(e2)).filter(|_| c.radius >= 1e-9) else { return c.enter };
    let flat = q - c.centre;
    let in_plane = flat - axis * flat.dot(axis);
    let u = normalize3(e1).unwrap_or(DVec3::X);
    let v = axis.cross(u);
    let mut phi = in_plane.dot(v).atan2(in_plane.dot(u));
    let sweep = c.turn;
    if phi < 0.0 {
        phi = if phi < (sweep - 2.0 * PI) / 2.0 { phi + 2.0 * PI } else { 0.0 };
    }
    if phi > sweep {
        phi = if phi - sweep < 2.0 * PI - phi { sweep } else { 0.0 };
    }
    c.centre + u * (c.radius * phi.cos()) + v * (c.radius * phi.sin())
}

/// The chain's nearest point to q, the way it runs there, and on which piece: pieces count
/// straight 0, corner 0, straight 1, corner 1, …
fn nearest_on(q: DVec3, chain: &Chain) -> (DVec3, DVec3, usize) {
    let Chain { ends, corners } = chain;
    let mut best = (ends[0], DVec3::X, 0usize);
    let mut far = f64::INFINITY;
    let mut take = |at: DVec3, dir: DVec3, piece: usize| {
        let d = (q - at).length();
        if d < far {
            far = d;
            best = (at, dir, piece);
        }
    };
    let m = corners.len();
    for i in 0..=m {
        let a = if i == 0 { ends[0] } else { corners[i - 1].exit };
        let b = if i == m { ends[1] } else { corners[i].enter };
        let way = normalize3(b - a)
            .or_else(|| normalize3((if i == m { ends[1] } else { corners[i].at }) - (if i == 0 { ends[0] } else { corners[i - 1].at })))
            .unwrap_or(DVec3::X);
        take(on_segment(q, a, b, i == 0, i == m), way, i * 2);
        if i < m {
            let c = &corners[i];
            let at = on_arc(q, c);
            let axis = normalize3((c.enter - c.centre).cross(c.exit - c.centre));
            let out = axis.and_then(|ax| normalize3(ax.cross(at - c.centre)));
            take(at, out.unwrap_or(way), i * 2 + 1);
        }
    }
    best
}

fn sum_sq(v: &[f64]) -> f64 {
    v.iter().fold(0.0, |a, b| a + b * b)
}

/// The chain's points and radii brought nearer the centres by damped least squares over the
/// three components of every centre's miss.
fn fit_chain(centres: &[DVec3], points0: &[DVec3], radii0: &[f64], floor: f64, iterations: usize) -> (Vec<DVec3>, Vec<f64>, f64) {
    let np = points0.len();
    let unpack = |x: &[f64]| -> (Vec<DVec3>, Vec<f64>) {
        let points = (0..np).map(|i| DVec3::new(x[i * 3], x[i * 3 + 1], x[i * 3 + 2])).collect();
        let radii = (0..radii0.len()).map(|j| floor.max(x[np * 3 + j])).collect();
        (points, radii)
    };
    let miss = |x: &[f64]| -> Vec<f64> {
        let (points, radii) = unpack(x);
        let chain = chain_of(&points, &radii);
        let mut out = vec![0.0; centres.len() * 3];
        for (i, c) in centres.iter().enumerate() {
            let n = nearest_on(*c, &chain).0;
            for k in 0..3 {
                out[i * 3 + k] = c[k] - n[k];
            }
        }
        out
    };
    let size = np * 3 + radii0.len();
    let mut x = vec![0.0; size];
    for (i, p) in points0.iter().enumerate() {
        x[i * 3] = p.x;
        x[i * 3 + 1] = p.y;
        x[i * 3 + 2] = p.z;
    }
    x[np * 3..].copy_from_slice(radii0);
    let mut r = miss(&x);
    let mut cost = sum_sq(&r);
    let mut lambda = 1e-3;
    let delta = 1e-3;
    let mut it = 0;
    while it < iterations {
        let mut jac: Vec<Vec<f64>> = Vec::with_capacity(size);
        for p in 0..size {
            let mut moved = x.clone();
            moved[p] += delta;
            let rp = miss(&moved);
            jac.push((0..r.len()).map(|i| (rp[i] - r[i]) / delta).collect());
        }
        let mut jtj = vec![0.0; size * size];
        let mut jtr = vec![0.0; size];
        for a in 0..size {
            for b in a..size {
                let mut v = 0.0;
                for i in 0..r.len() {
                    v += jac[a][i] * jac[b][i];
                }
                jtj[a * size + b] = v;
                jtj[b * size + a] = v;
            }
            for i in 0..r.len() {
                jtr[a] -= jac[a][i] * r[i];
            }
        }
        let mut better = false;
        let mut tries = 0;
        while tries < 8 && !better {
            let mut damped = jtj.clone();
            for a in 0..size {
                damped[a * size + a] += lambda * (jtj[a * size + a] + 1e-6);
            }
            if let Some(step) = solve_linear(size, &mut damped, &jtr) {
                let next: Vec<f64> = (0..size).map(|a| x[a] + step[a]).collect();
                let rn = miss(&next);
                let c = sum_sq(&rn);
                if c < cost {
                    let done = cost - c < 1e-9 * cost;
                    x = next;
                    r = rn;
                    cost = c;
                    lambda = (lambda / 3.0).max(1e-9);
                    better = true;
                    if done {
                        it = iterations;
                    }
                }
            }
            if !better {
                lambda *= 4.0;
            }
            tries += 1;
        }
        if !better {
            break;
        }
        it += 1;
    }
    let (points, radii) = unpack(&x);
    (points, radii, cost)
}

/// Where a chain's ends lie: the first and last centres, on the straights.
fn ends_on(centres: &[DVec3], points: &[DVec3], radii: &[f64]) -> Vec<DVec3> {
    let chain = chain_of(points, radii);
    let mut out = points.to_vec();
    out[0] = nearest_on(centres[0], &chain).0;
    let last = out.len() - 1;
    out[last] = nearest_on(centres[centres.len() - 1], &chain).0;
    out
}

/// How far points lie off a route, by the chain's own measure, as (rms, worst); for checking
/// one route against centres read for another. A route of one point measures to that point;
/// with no centres, or no route at all, the rms is NaN and the worst −1, as ScanRuler's
/// `missOf` leaves them.
pub fn route_miss(centres: &[DVec3], route: &SweepRoute) -> (f64, f64) {
    if route.points.is_empty() {
        return (f64::NAN, -1.0);
    }
    let inner = route.points.len().saturating_sub(2);
    let radii: Vec<f64> = (0..inner).map(|j| route.radii.as_ref().and_then(|r| r.get(j).copied().flatten()).unwrap_or(route.bend_radius)).collect();
    let (rms, max, _) = miss_of(centres, &chain_of(&route.points, &radii));
    (rms, max)
}

/// How far the centres lie off a chain: rms, worst, where.
fn miss_of(centres: &[DVec3], chain: &Chain) -> (f64, f64, usize) {
    let mut sum = 0.0;
    let mut max = -1.0;
    let mut at = 0;
    for (i, c) in centres.iter().enumerate() {
        let d = (*c - nearest_on(*c, chain).0).length();
        sum += d * d;
        if d > max {
            max = d;
            at = i;
        }
    }
    ((sum / centres.len() as f64).sqrt(), max, at)
}

/// Where the centres turn: each run of stations turning faster than BEND_SPAN_TURN allows, as
/// a corner where the straights either side of it meet, with the radius that brings the arc to
/// the centre nearest the corner.
fn seed_corners(centres: &[DVec3], size: f64) -> (Vec<DVec3>, Vec<f64>) {
    let n = centres.len();
    let spacing = (centres[n - 1] - centres[0]).length() / (n - 1) as f64;
    let span = 2f64.max(js_round((BEND_SPAN * size) / spacing.max(1e-6))) as isize;
    let chord = |a: isize, b: isize| normalize3(centres[(b.min(n as isize - 1)).max(0) as usize] - centres[a.max(0) as usize]);
    let bending: Vec<bool> = (0..n)
        .map(|i| {
            if i == 0 || i == n - 1 {
                return false;
            }
            let i = i as isize;
            match (chord(i - span, i), chord(i, i + span)) {
                (Some(back), Some(on)) => back.dot(on).clamp(-1.0, 1.0).acos().to_degrees() > BEND_SPAN_TURN_DEG,
                _ => false,
            }
        })
        .collect();
    let mut points = vec![centres[0]];
    let mut radii = Vec::new();
    let mut i = 0;
    while i < n {
        if !bending[i] {
            i += 1;
            continue;
        }
        let mut j = i;
        while j + 1 < n && bending[j + 1] {
            j += 1;
        }
        let (ii, jj) = (i as isize, j as isize);
        if let (Some(a), Some(b)) = (chord(ii - span, ii), chord(jj, jj + span)) {
            // Where the two lines come nearest.
            let w = centres[i] - centres[j];
            let (ab, aw, bw) = (a.dot(b), a.dot(w), b.dot(w));
            let under = 1.0 - ab * ab;
            let turn = ab.clamp(-1.0, 1.0).acos();
            if under > 1e-6 && turn.to_degrees() >= MIN_TURN_DEG {
                let along = (ab * bw - aw) / under;
                let on = (bw - ab * aw) / under;
                let pa = centres[i] + a * along;
                let corner = pa + ((centres[j] + b * on) - pa) * 0.5;
                let off = centres[i..=j].iter().map(|c| (*c - corner).length()).fold(f64::INFINITY, f64::min);
                let stand = 1.0 / (turn.min(3.0) / 2.0).cos() - 1.0;
                points.push(corner);
                radii.push(if stand > 1e-6 { off / stand } else { size });
            }
        }
        i = j + 1;
    }
    points.push(centres[n - 1]);
    (points, radii)
}

/// The fewest corners that hold the centres: those they turn at, then one more where they
/// stray most, for as long as that brings the worst down.
fn grow_chain(centres: &[DVec3], tolerance: f64, size: f64) -> (Vec<DVec3>, Vec<f64>) {
    let floor = size / 2.0;
    let settle = |p: &[DVec3], r: &[f64]| {
        let (fp, fr, _) = fit_chain(centres, p, r, floor, 25);
        (ends_on(centres, &fp, &fr), fr)
    };
    let (seed_p, seed_r) = seed_corners(centres, size);
    let (mut points, mut radii) = settle(&seed_p, &seed_r);
    let first_radius = size;
    let mut held = miss_of(centres, &chain_of(&points, &radii));
    let mut n = 0;
    while n < MAX_CORNERS && held.1 > tolerance {
        let chain = chain_of(&points, &radii);
        let worst = centres[held.2];
        let piece = nearest_on(worst, &chain).2;
        // On a straight, a corner within it; on an arc, the arc is two bends: the new corner
        // goes on the side of its middle the worst centre lies.
        let after = if piece % 2 == 0 {
            piece / 2
        } else {
            let c = &chain.corners[(piece - 1) / 2];
            (piece - 1) / 2 + usize::from((worst - c.at).dot(c.exit - c.enter) > 0.0)
        };
        let mut tp: Vec<DVec3> = points[..after + 1].to_vec();
        tp.push(worst);
        tp.extend_from_slice(&points[after + 1..]);
        let mut tr: Vec<f64> = radii[..after].to_vec();
        tr.push(first_radius);
        tr.extend_from_slice(&radii[after..]);
        let (trial_p, trial_r) = settle(&tp, &tr);
        let chain_now = chain_of(&trial_p, &trial_r);
        let kept = chain_now.corners.iter().all(|c| c.turn.to_degrees() >= MIN_TURN_DEG);
        let now = miss_of(centres, &chain_now);
        if !kept || now.1 > held.1 * (1.0 - CORNER_GAIN) {
            break;
        }
        points = trial_p;
        radii = trial_r;
        held = now;
        n += 1;
    }
    (points, radii)
}

// ---- the trace --------------------------------------------------------------

/// `Math.round(x * 10) / 10 || 0`.
fn r1(x: f64) -> f64 {
    round_to(x, 1)
}

/// `Math.round(x * 1000) / 1000 || 0`.
fn r3(x: f64) -> f64 {
    round_to(x, 3)
}

fn at_words(p: DVec3) -> String {
    format!("({}, {}, {})", num(r3(p.x)), num(r3(p.y)), num(r3(p.z)))
}

/// The cutter the trace reads the scan with: the section through `origin` square to `normal`.
pub trait TubeCutter {
    fn cut(&mut self, origin: DVec3, normal: DVec3) -> SectionCut;
}

impl<F: FnMut(DVec3, DVec3) -> SectionCut> TubeCutter for F {
    fn cut(&mut self, origin: DVec3, normal: DVec3) -> SectionCut {
        self(origin, normal)
    }
}

/// The trace's working state: the cutter, how many cuts it took, what it has to say.
struct Tracer<'a> {
    cutter: &'a mut dyn TubeCutter,
    cuts: usize,
    warnings: Vec<String>,
}

impl Tracer<'_> {
    fn cut_at(&mut self, origin: DVec3, normal: DVec3, near: f64) -> Found {
        self.cuts += 1;
        let cut = self.cutter.cut(origin, normal);
        round_at(&cut, origin, normal, near)
    }

    /// One way along the tube from the first station: the stations, it first, and where and
    /// why it stopped.
    fn march(&mut self, station0: TubeStation, going: DVec3, step: f64, max_length: f64) -> (Vec<TubeStation>, TubeStop) {
        let mut stations = vec![TubeStation { dir: going, ..station0 }];
        let mut last_step = f64::NEG_INFINITY;
        let mut dir = going;
        loop {
            let last = stations[stations.len() - 1];
            if last.s + step > max_length {
                let says = format!("maxLengthMm: {} mm followed and the tube goes on.", num(r1(max_length)));
                return (stations, TubeStop { kind: TubeStopKind::Length, s: last.s, at: last.at, says });
            }
            let (ahead_at, ahead_dir) = reckon(&stations, dir, step);
            let sizes: Vec<f64> = stations[stations.len().saturating_sub(RUNNING)..].iter().map(|x| x.size).collect();
            let running = median(&sizes);
            let found = self.cut_at(ahead_at, ahead_dir, running);
            let s = last.s + step;
            let place = format!("{} mm from the start, at {}", num(r1(s)), at_words(ahead_at));
            let found = match found {
                Found::Open => return (stations, TubeStop { kind: TubeStopKind::Open, s, at: ahead_at, says: format!("The wall is open {place}: the scan has a hole or an edge there.") }),
                Found::None => {
                    let says = format!("The loop is lost {place}: nothing closed round the centreline, so the tube ends here, or turned tighter than the step follows.");
                    return (stations, TubeStop { kind: TubeStopKind::Lost, s, at: ahead_at, says });
                }
                Found::Round(r) => r,
            };
            let jump = found.size / running - 1.0;
            if jump.abs() > SIZE_JUMP {
                let says = format!(
                    "The size jumps {} → {} mm ({} %) {place}: a flange, a bellows or a junction. The trace stops here, and does not cross it; start again past it for what follows.",
                    num(r1(running)),
                    num(r1(found.size)),
                    num(r1(jump * 100.0))
                );
                return (stations, TubeStop { kind: TubeStopKind::Size, s, at: ahead_at, says });
            }
            let off = (found.at - ahead_at).length();
            if off > CENTRE_JUMP * running {
                let says = format!("The centre moves {} mm off its line {place}: this loop is not the tube's, or a junction joins it. The trace stops here.", num(r1(off)));
                return (stations, TubeStop { kind: TubeStopKind::Centre, s, at: ahead_at, says });
            }
            if jump.abs() > SIZE_STEP && s - last_step > 4.0 * step {
                self.warnings.push(format!(
                    "The size goes {} → {} mm {place}: a step in the tube (a cuff, a sleeve, a weld) the trace went through. One profile will not fit both sides.",
                    num(r1(running)),
                    num(r1(found.size))
                ));
                last_step = s;
            }
            dir = normalize3(found.at - last.at).unwrap_or(ahead_dir);
            stations.push(TubeStation { s: last.s + (found.at - last.at).length(), at: found.at, dir: ahead_dir, major: found.major, minor: found.minor, major_dir: found.major_dir, size: found.size });
        }
    }
}

/// Follow a tube from `input.start` the way `input.dir` goes, and with `both` the other way
/// too, until it ends or stops being one: see the top of the file. The step is by default an
/// eighth of its size.
pub fn trace_tube(cutter: &mut dyn TubeCutter, input: TraceInput) -> TubeTrace {
    let mut tr = Tracer { cutter, cuts: 0, warnings: Vec::new() };
    let empty = |tr: Tracer, kind: TubeStopKind, says: String, at: DVec3| TubeTrace { stations: vec![], stops: vec![TubeStop { kind, s: 0.0, at, says }], route: None, runs: vec![], residual: None, warnings: tr.warnings, cuts: tr.cuts };
    let Some(way) = normalize3(input.dir) else { return empty(tr, TubeStopKind::Lost, "dir has no length.".into(), input.start) };
    let first = match tr.cut_at(input.start, way, f64::INFINITY) {
        Found::Round(r) => r,
        Found::Open => {
            let says = format!("The scan cut square to dir at start {} has an open chain and no closed loop round start: the tube's wall has a hole there. Start where it is whole.", at_words(input.start));
            return empty(tr, TubeStopKind::Open, says, input.start);
        }
        Found::None => {
            let says = format!("No closed loop round start {} on a plane square to dir: start has to lie inside the tube, dir along it.", at_words(input.start));
            return empty(tr, TubeStopKind::Lost, says, input.start);
        }
    };
    let step = input.step_mm.unwrap_or(0.25f64.max(first.size / 8.0));
    let max_length = input.max_length_mm.unwrap_or(3000.0);
    let station0 = TubeStation { s: 0.0, at: first.at, dir: way, major: first.major, minor: first.minor, major_dir: first.major_dir, size: first.size };

    let (on_stations, on_stop) = tr.march(station0, way, step, max_length);
    let back = if input.both { Some(tr.march(station0, -way, step, max_length)) } else { None };
    // Run the whole of it one way: from the far end of the way back.
    let (mut stations, stops) = match back {
        Some((back_stations, back_stop)) => {
            let mut all: Vec<TubeStation> = back_stations[1..].iter().rev().map(|x| TubeStation { dir: -x.dir, ..*x }).collect();
            all.extend(on_stations);
            (all, vec![back_stop, on_stop])
        }
        None => (on_stations, vec![on_stop]),
    };
    if stations.len() < TRACE_MIN_STATIONS {
        tr.warnings.push(format!("Only {} station{} before the stop: too few for a centreline.", stations.len(), if stations.len() == 1 { "" } else { "s" }));
        return TubeTrace { stations, stops, route: None, runs: vec![], residual: None, warnings: tr.warnings, cuts: tr.cuts };
    }
    // The chain from the march's centres; then the stations cut again square to it, and it
    // fitted again to those.
    let size = median(&stations.iter().map(|x| x.size).collect::<Vec<_>>());
    let tolerance = 0.1f64.max(0.008 * size);
    let centres: Vec<DVec3> = stations.iter().map(|x| x.at).collect();
    let (mut points, mut radii) = grow_chain(&centres, tolerance, size);
    let mut again = 0;
    while again < RECUTS && !radii.is_empty() {
        let chain = chain_of(&points, &radii);
        for station in stations.iter_mut() {
            let (there_at, there_dir, _) = nearest_on(station.at, &chain);
            let Found::Round(found) = tr.cut_at(there_at, there_dir, size) else { continue };
            if (found.size / size - 1.0).abs() > SIZE_JUMP {
                continue;
            }
            *station = TubeStation { at: found.at, dir: there_dir, major: found.major, minor: found.minor, major_dir: found.major_dir, size: found.size, ..*station };
        }
        let centres: Vec<DVec3> = stations.iter().map(|x| x.at).collect();
        let (fp, fr, _) = fit_chain(&centres, &points, &radii, size / 2.0, 25);
        points = ends_on(&centres, &fp, &fr);
        radii = fr;
        again += 1;
    }
    let mut s = 0.0;
    for i in 0..stations.len() {
        if i > 0 {
            s += (stations[i].at - stations[i - 1].at).length();
        }
        stations[i].s = s;
    }
    let chain = chain_of(&points, &radii);
    let centres: Vec<DVec3> = stations.iter().map(|x| x.at).collect();
    let held = miss_of(&centres, &chain);
    if held.1 > RESIDUAL_WARN * size {
        tr.warnings.push(format!(
            "A centre lies {} mm off the route {} mm along, at {}: the tube is dented or lumped there, or it is no chain of straights and round bends. Check it by a section.",
            num(r3(held.1)),
            num(r1(stations[held.2].s)),
            at_words(stations[held.2].at)
        ));
    }
    // What goes out takes the radii the chain really has, each rounded a hair down so that
    // what it was fitted to still fits once rounded.
    let route = SweepRoute {
        points: points.iter().map(|p| DVec3::new(r3(p.x), r3(p.y), r3(p.z))).collect(),
        bend_radius: if radii.is_empty() { r3(size) } else { r3(median(&chain.corners.iter().map(|c| c.radius).collect::<Vec<_>>())) },
        radii: if radii.is_empty() { None } else { Some(chain.corners.iter().map(|c| Some((c.radius * 1000.0).floor() / 1000.0)).collect()) },
    };
    if let Err(e) = route_pieces(&route, route.points[0]) {
        tr.warnings.push(format!("The route as rounded does not build: {e}"));
    }
    let runs = runs_of(&stations, &chain);
    TubeTrace {
        residual: Some(TubeResidual { rms: r3(held.0), max: r3(held.1), at: stations[held.2].at }),
        stations,
        stops,
        route: Some(route),
        runs,
        warnings: tr.warnings,
        cuts: tr.cuts,
    }
}

/// The chain's straights and bends with the tube's size on each, the medians of the stations
/// nearest it.
fn runs_of(stations: &[TubeStation], chain: &Chain) -> Vec<TubeRun> {
    let Chain { corners, ends } = chain;
    let mut on: BTreeMap<usize, Vec<TubeStation>> = BTreeMap::new();
    for x in stations {
        let piece = nearest_on(x.at, chain).2;
        on.entry(piece).or_default().push(*x);
    }
    let read = |piece: usize| -> (f64, f64, f64) {
        let here = on.get(&piece).map_or(&[][..], |v| v.as_slice());
        if here.is_empty() {
            return (0.0, 0.0, 0.0);
        }
        let rms = (here.iter().map(|x| (x.at - nearest_on(x.at, chain).0).length().powi(2)).sum::<f64>() / here.len() as f64).sqrt();
        (r3(median(&here.iter().map(|x| x.major).collect::<Vec<_>>())), r3(median(&here.iter().map(|x| x.minor).collect::<Vec<_>>())), r3(rms))
    };
    let round3 = |p: DVec3| DVec3::new(r3(p.x), r3(p.y), r3(p.z));
    let r4 = |x: f64| round_to(x, 4);
    let mut out = Vec::new();
    let m = corners.len();
    for i in 0..=m {
        let a = if i == 0 { ends[0] } else { corners[i - 1].exit };
        let b = if i == m { ends[1] } else { corners[i].enter };
        if (b - a).length() > 1e-3 {
            let (major, minor, rms) = read(i * 2);
            out.push(TubeRun::Straight { from: round3(a), to: round3(b), length: r3((b - a).length()), major, minor, rms });
        }
        if i == m {
            continue;
        }
        let c = &corners[i];
        let normal = normalize3((c.enter - c.centre).cross(c.exit - c.centre)).unwrap_or(DVec3::Z);
        let (major, minor, rms) = read(i * 2 + 1);
        out.push(TubeRun::Bend { corner: round3(c.at), radius: r3(c.radius), turn_deg: r3(c.turn.to_degrees()), normal: DVec3::new(r4(normal.x), r4(normal.y), r4(normal.z)), major, minor, rms });
    }
    out
}

#[cfg(test)]
mod tests {
    //! Ported from ScanRuler's `plugins/reverse/tests/reTubeTrace.test.ts` (the synthetic
    //! tubes; the real scan the test reads when told where one is, is not).
    use super::*;
    use crate::loop_piece::LoopPiece;
    use crate::slice::slice_mesh;

    fn v(x: f64, y: f64, z: f64) -> DVec3 {
        DVec3::new(x, y, z)
    }

    struct TubeMesh {
        positions: Vec<DVec3>,
        triangles: Vec<[u32; 3]>,
    }

    /// A closed tube mesh about a centreline given as frames, its rim at each frame a loop
    /// `radius_at(i, angle)` round — a welded skin, its ends capped, wound with its normals
    /// out. Positions rounded to single precision, as the test's Float32Array holds them.
    fn tube_mesh(line: &[(DVec3, DVec3)], radius_at: impl Fn(usize, f64) -> f64, around: usize) -> TubeMesh {
        let f32r = |p: DVec3| DVec3::new(p.x as f32 as f64, p.y as f32 as f64, p.z as f32 as f64);
        let mut positions = Vec::new();
        let mut triangles = Vec::new();
        let d0 = line[0].1;
        let mut u = normalize3(d0.cross(if d0.x.abs() < 0.9 { DVec3::X } else { DVec3::Y })).unwrap_or(DVec3::X);
        for (i, (at, dir)) in line.iter().enumerate() {
            // Carried along with the least turn, as a sweep's section is.
            u = normalize3(u - *dir * u.dot(*dir)).unwrap_or(u);
            let w = dir.cross(u);
            for k in 0..around {
                let a = 2.0 * PI * k as f64 / around as f64;
                let r = radius_at(i, a);
                positions.push(f32r(*at + u * (r * a.cos()) + w * (r * a.sin())));
            }
        }
        let around32 = around as u32;
        for i in 0..line.len() as u32 - 1 {
            for k in 0..around32 {
                let (a, b) = (i * around32 + k, i * around32 + (k + 1) % around32);
                let (c, d) = (a + around32, b + around32);
                triangles.push([a, b, d]);
                triangles.push([a, d, c]);
            }
        }
        // The ends: a fan to a middle point each.
        let start_mid = positions.len() as u32;
        positions.push(f32r(line[0].0));
        let end_mid = positions.len() as u32;
        positions.push(f32r(line[line.len() - 1].0));
        let last_ring = (line.len() as u32 - 1) * around32;
        for k in 0..around32 {
            triangles.push([start_mid, (k + 1) % around32, k]);
            triangles.push([end_mid, last_ring + k, last_ring + (k + 1) % around32]);
        }
        TubeMesh { positions, triangles }
    }

    /// Frames along a route's pieces, about every `each` mm.
    fn frames_along(route: &SweepRoute, each: f64) -> Vec<(DVec3, DVec3)> {
        let made = route_pieces(route, route.points[0]).expect("the route builds");
        let mut out = Vec::new();
        for p in &made.pieces {
            match *p {
                LoopPiece::Line { a, b } => {
                    let n = ((a - b).length() / each).ceil().max(1.0) as usize;
                    let dir = (b - a).normalize();
                    for i in 0..n {
                        out.push((a + (b - a) * (i as f64 / n as f64), dir));
                    }
                }
                LoopPiece::Arc { a, m, b } => {
                    // The centre from the three points: a and b equally far from m.
                    let chord = b - a;
                    let mid = (a + b) / 2.0;
                    let to_m = m - mid;
                    let sag = to_m.length();
                    let half = chord.length() / 2.0;
                    let r = (sag * sag + half * half) / (2.0 * sag);
                    let c = m + to_m.normalize() * -r;
                    let e1 = (a - c).normalize();
                    let axis = (a - c).cross(b - c).normalize();
                    let e2 = axis.cross(e1);
                    let sweep = 2.0 * (half / r).min(1.0).asin();
                    let n = ((sweep * r / each).ceil() as usize).max(2);
                    for i in 0..n {
                        let t = sweep * i as f64 / n as f64;
                        out.push((c + e1 * (r * t.cos()) + e2 * (r * t.sin()), e1 * -t.sin() + e2 * t.cos()));
                    }
                }
                LoopPiece::Circle { .. } => {}
            }
        }
        if let Some(LoopPiece::Line { a, b } | LoopPiece::Arc { a, b, .. }) = made.pieces.last() {
            out.push((*b, (*b - *a).normalize()));
        }
        out
    }

    fn cutter(m: &TubeMesh) -> impl FnMut(DVec3, DVec3) -> SectionCut + '_ {
        move |origin, normal| slice_mesh(&m.positions, &m.triangles, origin, normal, 0.5)
    }

    /// A tube: 60 straight, bend in XZ, 50 straight, bend in another plane, 60 on.
    fn route() -> SweepRoute {
        SweepRoute { points: vec![v(0.0, 0.0, 0.0), v(0.0, 0.0, 60.0), v(50.0, 0.0, 60.0), v(50.0, 40.0, 90.0)], bend_radius: 20.0, radii: None }
    }

    fn radius(t: &TubeTrace, j: usize) -> f64 {
        t.route.as_ref().and_then(|r| r.radii.as_ref()).and_then(|r| r.get(j).copied().flatten()).unwrap_or(f64::NAN)
    }

    fn point(t: &TubeTrace, j: usize) -> DVec3 {
        t.route.as_ref().map_or(DVec3::splat(f64::NAN), |r| r.points[j])
    }

    #[test]
    fn reads_a_synthetic_tube_back_corners_and_radius_in_two_bend_planes() {
        let route = route();
        let mesh = tube_mesh(&frames_along(&route, 0.5), |_, _| 8.0, 72);
        let t = trace_tube(&mut cutter(&mesh), TraceInput::new(v(0.0, 0.0, 1.0), v(0.0, 0.0, 1.0)));
        assert!(matches!(t.stops[0].kind, TubeStopKind::Lost | TubeStopKind::Size), "{}", t.stops[0].says);
        let got = t.route.as_ref().expect("a route");
        assert_eq!(got.points.len(), 4);
        for j in [1, 2] {
            assert!((got.points[j] - route.points[j]).length() < 0.1, "{:?}", got.points);
        }
        assert!((radius(&t, 0) - 20.0).abs() < 0.15, "{}", radius(&t, 0));
        assert!((radius(&t, 1) - 20.0).abs() < 0.15, "{}", radius(&t, 1));
        assert!(t.residual.expect("a residual").max < 0.05);
        assert!((t.stations[0].major - 16.0).abs() < 0.5);
        assert_eq!(t.runs.iter().map(|r| r.kind_name()).collect::<Vec<_>>(), ["straight", "bend", "straight", "bend", "straight"]);
        let TubeRun::Bend { turn_deg, normal, .. } = t.runs[1] else { panic!("not a bend") };
        assert!((turn_deg - 90.0).abs() < 0.5);
        assert!((normal.y.abs() - 1.0).abs() < 0.005);
        assert!(t.warnings.is_empty(), "{:?}", t.warnings);
        // What it answers builds: the same chain, ready for a sweep.
        assert!(route_pieces(got, got.points[0]).is_ok());
    }

    #[test]
    fn reads_a_short_straight_between_two_bends_and_bends_of_different_radii() {
        let route = SweepRoute { points: vec![v(0.0, 0.0, 0.0), v(0.0, 0.0, 50.0), v(40.0, 0.0, 50.0), v(40.0, 45.0, 50.0)], bend_radius: 15.0, radii: Some(vec![Some(15.0), Some(22.0)]) };
        let mesh = tube_mesh(&frames_along(&route, 0.5), |_, _| 6.0, 72);
        let t = trace_tube(&mut cutter(&mesh), TraceInput::new(v(0.0, 0.0, 0.5), v(0.0, 0.0, 1.0)));
        assert_eq!(t.route.as_ref().map(|r| r.points.len()), Some(4));
        assert!((radius(&t, 0) - 15.0).abs() < 0.3, "{}", radius(&t, 0));
        assert!((radius(&t, 1) - 22.0).abs() < 0.3, "{}", radius(&t, 1));
        assert!((point(&t, 1) - route.points[1]).length() < 0.15);
        assert!((point(&t, 2) - route.points[2]).length() < 0.15);
    }

    #[test]
    fn reads_an_oval_the_same_with_its_two_diameters_and_which_way() {
        let route = route();
        let mesh = tube_mesh(
            &frames_along(&route, 0.5),
            |_, a| {
                // An ellipse of semi-axes 9 and 7 by its polar radius.
                let (c, s) = (a.cos() / 9.0, a.sin() / 7.0);
                1.0 / c.hypot(s)
            },
            96,
        );
        let t = trace_tube(&mut cutter(&mesh), TraceInput::new(v(0.0, 0.0, 1.0), v(0.0, 0.0, 1.0)));
        assert!((t.stations[1].major - 18.0).abs() < 0.5, "{}", t.stations[1].major);
        assert!((t.stations[1].minor - 14.0).abs() < 0.5, "{}", t.stations[1].minor);
        assert!((point(&t, 1) - route.points[1]).length() < 0.1);
        assert!((radius(&t, 0) - 20.0).abs() < 0.2);
    }

    #[test]
    fn follows_it_both_ways_from_a_straight_the_stations_in_one_run_a_stop_at_each_end() {
        let route = route();
        let mesh = tube_mesh(&frames_along(&route, 0.5), |_, _| 8.0, 72);
        let t = trace_tube(&mut cutter(&mesh), TraceInput { both: true, ..TraceInput::new(v(20.0, 0.0, 60.0), v(1.0, 0.0, 0.0)) });
        assert_eq!(t.stops.len(), 2);
        for s in &t.stops {
            assert!(matches!(s.kind, TubeStopKind::Lost | TubeStopKind::Size), "{}", s.says);
        }
        assert!(t.stations[0].at.z < 3.0);
        assert!(t.stations[t.stations.len() - 1].at.y > 38.0);
        assert_eq!(t.route.as_ref().map(|r| r.points.len()), Some(4));
        assert!((point(&t, 1) - route.points[1]).length() < 0.1);
        assert!((point(&t, 2) - route.points[2]).length() < 0.1);
        // Started off centre and off the tube's way, it finds both.
        let rough = trace_tube(&mut cutter(&mesh), TraceInput { both: true, ..TraceInput::new(v(20.0, 2.0, 62.0), v(1.0, 0.2, 0.2)) });
        assert!((point(&rough, 1) - route.points[1]).length() < 0.1);
        assert!((point(&rough, 2) - route.points[2]).length() < 0.1);
        // A route another way round measures the same centres the same.
        assert!(route_miss(&t.stations.iter().map(|x| x.at).collect::<Vec<_>>(), &route).1 < 0.05);
    }

    #[test]
    fn measures_against_a_route_of_one_point_and_says_nothing_of_no_centres() {
        // As ScanRuler's chain of one point: the centres' distances to it.
        let one = SweepRoute { points: vec![v(0.0, 0.0, 0.0)], bend_radius: 5.0, radii: None };
        let (rms, max) = route_miss(&[v(3.0, 4.0, 0.0), v(0.0, 0.0, 5.0)], &one);
        assert!((rms - 5.0).abs() < 1e-12 && (max - 5.0).abs() < 1e-12, "{rms} {max}");
        let (rms, max) = route_miss(&[], &route());
        assert!(rms.is_nan() && max == -1.0);
    }

    #[test]
    fn stops_at_a_bellows_and_a_flange_saying_where_and_why_and_does_not_cross_them() {
        // Straight 60, then folds: the radius 8 to 12 and back, pitch 4.
        let straight = frames_along(&SweepRoute { points: vec![v(0.0, 0.0, 0.0), v(0.0, 0.0, 120.0)], bend_radius: 1.0, radii: None }, 0.25);
        let folded = |i: usize, _a: f64| {
            let z = straight[i].0.z;
            if !(60.0..=90.0).contains(&z) {
                8.0
            } else {
                8.0 + 4.0 * (PI * (z - 60.0) / 4.0).sin().abs()
            }
        };
        let bellows = tube_mesh(&straight, folded, 72);
        let t = trace_tube(&mut cutter(&bellows), TraceInput::new(v(0.0, 0.0, 2.0), v(0.0, 0.0, 1.0)));
        assert_eq!(t.stops[0].kind, TubeStopKind::Size, "{}", t.stops[0].says);
        assert!(t.stops[0].says.contains("size jumps"));
        assert!(t.stops[0].at.z > 58.0 && t.stops[0].at.z < 66.0, "{:?}", t.stops[0].at);
        assert!(t.stations[t.stations.len() - 1].at.z < 62.0);
        // A flange: a step out, at once.
        let flange = tube_mesh(&straight, |i, _| if straight[i].0.z < 80.0 { 8.0 } else { 20.0 }, 72);
        let f = trace_tube(&mut cutter(&flange), TraceInput::new(v(0.0, 0.0, 2.0), v(0.0, 0.0, 1.0)));
        assert_eq!(f.stops[0].kind, TubeStopKind::Size);
        assert!(f.stops[0].at.z > 76.0);
        // A small step is gone through, and said.
        let cuff = tube_mesh(&straight, |i, _| if straight[i].0.z < 60.0 { 8.0 } else { 8.5 }, 72);
        let c = trace_tube(&mut cutter(&cuff), TraceInput { max_length_mm: Some(100.0), ..TraceInput::new(v(0.0, 0.0, 2.0), v(0.0, 0.0, 1.0)) });
        assert_eq!(c.stops[0].kind, TubeStopKind::Length);
        assert!(c.stops[0].says.contains("maxLengthMm"));
        assert!(c.warnings.join(" ").contains("size goes 16 → 16.8"), "{:?}", c.warnings);
    }

    #[test]
    fn says_it_when_start_is_not_in_a_tube_or_the_wall_is_open() {
        let straight = frames_along(&SweepRoute { points: vec![v(0.0, 0.0, 0.0), v(0.0, 0.0, 80.0)], bend_radius: 1.0, radii: None }, 0.5);
        let mesh = tube_mesh(&straight, |_, _| 8.0, 72);
        let out = trace_tube(&mut cutter(&mesh), TraceInput::new(v(30.0, 0.0, 40.0), v(0.0, 0.0, 1.0)));
        assert!(out.route.is_none());
        assert_eq!(out.stops[0].kind, TubeStopKind::Lost);
        assert!(out.stops[0].says.contains("start has to lie inside the tube"));
        // A hole in the wall from 40 on: the triangles of one side left out.
        let holed = TubeMesh {
            positions: mesh.positions.clone(),
            triangles: mesh
                .triangles
                .iter()
                .filter(|t| {
                    let c = (mesh.positions[t[0] as usize] + mesh.positions[t[1] as usize] + mesh.positions[t[2] as usize]) / 3.0;
                    !(c.z > 40.0 && c.z < 60.0 && c.x > 4.0)
                })
                .copied()
                .collect(),
        };
        let open = trace_tube(&mut cutter(&holed), TraceInput::new(v(0.0, 0.0, 2.0), v(0.0, 0.0, 1.0)));
        assert_eq!(open.stops[0].kind, TubeStopKind::Open, "{}", open.stops[0].says);
        assert!(open.stops[0].at.z > 38.0 && open.stops[0].at.z < 44.0);
    }
}
