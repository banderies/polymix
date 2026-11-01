// SPDX-License-Identifier: AGPL-3.0-only
//! What the fitted pieces of a profile nearly are — with the evidence, and without the
//! verdict. Ported from ScanRuler (`plugins/reverse/core/intent.ts`), © Stefan Hermann
//! (CNC Kitchen), AGPL-3.0-only.
//!
//! A wall fitted to a scan comes out 89.7° to the floor beside it; whether that is a right
//! angle the scanner blurred or a draft the designer drew is not a question the numbers settle
//! alone, and an agent (or an operator) with the part in mind settles it better than a
//! threshold would. What the numbers do settle is how far the pieces are from each relation
//! and how well the scan pins them down, and that is what this module hands over: for every
//! piece how well it sits on its points and the uncertainty its fit carries, the round figure
//! a designer would have drawn beside each measured one, and every relation the pieces come
//! within a wide net of — parallel, square, in line, equal, concentric, tangent, on a
//! coordinate plane, mirror images — each with how far off it is and how many of the fit's own
//! sigmas that is. Under about two the relation is within the scan's noise; well above it the
//! pieces were measured apart. The caller decides, and the answer carries each relation ready
//! to add to the sketch as a constraint.
//!
//! Tangent, equal radius and a round angle are offered between neighbours only: pieces joined
//! at an end, or whose ends lie within a few of their scatters. Between pieces apart along the
//! profile they hold by chance — a wall's carrier that happens to graze a curve elsewhere, a
//! rim's lip as round as the bore's corner — and taken they pull one piece across the profile
//! to the other. Equal radius reaches one piece further, to two arcs at either end of one
//! side: a slot's two ends, a plate's corners. Circles are whole pieces with no ends: one is a
//! neighbour of what ends on it or touches it, and two holes may be the same size wherever
//! they are.
//!
//! The uncertainties are the fits' own, from the normal equations of the least squares behind
//! each piece (the Gauss–Newton covariance): a line's direction is known to σ / √Σt² —
//! tightly on a long side, loosely on a stub — and a circle's radius and centre to what the
//! arc it was fitted on allows, which on a short arc is little. The per-point σ is the piece's
//! own residual, floored at the chain's scatter so a piece that happens to sit on its points
//! does not claim certainty the scan does not have.
//!
//! The sketch model here has no splines, so a spline's residual (ScanRuler reads it off the
//! curve through its fit points) is not part of this port.

use std::collections::HashSet;

use glam::DVec2;
use serde::{Deserialize, Serialize};

use crate::datums::{DatumPlane, DatumTrace};
use crate::nominal::{nominal_of, nominal_step, NominalKind};
use crate::sketch::{constraint_entities, Anchor, AnchorAt, Entity, EntityId, MirrorAxis, OnTarget, Rule, Shape, ShapeKind};

/// How wide the net is cast: a relation farther off than this is not offered. Reporting
/// thresholds, not decisions — wide on purpose.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct IntentNet {
    /// Degrees off parallel, square, level or upright.
    pub angle_deg: f64,
    /// Millimetres off in line, concentric, tangent, equal, on a plane, or mirrored.
    pub mm: f64,
}

pub const DEFAULT_NET: IntentNet = IntentNet { angle_deg: 5.0, mm: 1.0 };

impl Default for IntentNet {
    fn default() -> Self {
        DEFAULT_NET
    }
}

/// How many candidates an answer carries unless told otherwise.
pub const DEFAULT_LIMIT: usize = 60;

/// A measured figure beside the round one a designer would have drawn.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Rounding {
    pub measured: f64,
    pub nominal: f64,
    /// measured − nominal, in the figure's unit.
    pub off: f64,
    /// |off| over the fit's sigma for that figure; `None` where the fit does not know one.
    pub sigmas: Option<f64>,
}

/// How well a piece sits on its slice points, mm, and how many there are.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Residual {
    pub rms: f64,
    pub max: f64,
    pub points: usize,
}

/// The fit's own uncertainty: a line's direction (`angle_deg`, degrees) and its place across
/// itself (`offset`, mm); a round piece's radius and its centre (mm).
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct PieceSigma {
    pub angle_deg: Option<f64>,
    pub offset: Option<f64>,
    pub radius: Option<f64>,
    pub center: Option<f64>,
}

/// A line's length and the angle it runs at, a round piece's radius, each beside its nearest
/// round value.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct PieceNominal {
    pub length: Option<Rounding>,
    pub angle_deg: Option<Rounding>,
    pub radius: Option<Rounding>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PieceReading {
    pub id: EntityId,
    /// A line, an arc or a circle.
    pub kind: ShapeKind,
    /// `None` for a piece drawn rather than fitted.
    pub residual: Option<Residual>,
    /// `None` where there are too few points to say.
    pub sigma: Option<PieceSigma>,
    pub nominal: PieceNominal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CandidateKind {
    Horizontal,
    Vertical,
    Parallel,
    Perpendicular,
    Collinear,
    Angle,
    Equal,
    Concentric,
    Tangent,
    OnDatum,
    Symmetric,
}

/// The unit a candidate's offset and sigma are in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OffUnit {
    Deg,
    Mm,
}

/// For an equal candidate: what is equal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EqualWhat {
    Length,
    Radius,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Candidate {
    pub kind: CandidateKind,
    /// The piece, or the two, by id.
    pub pieces: Vec<EntityId>,
    /// For on-datum and symmetric: which of the part's coordinate planes.
    pub datum: Option<DatumPlane>,
    /// How far the pieces are from the relation.
    pub off: f64,
    pub unit: OffUnit,
    /// The fit's sigma for that figure, same unit; `None` where not known.
    pub sigma: Option<f64>,
    /// off over sigma: under about 2, within the scan's noise.
    pub sigmas: Option<f64>,
    /// For angle: the round angle between the lines' directions (30, 45, 60 …) the pair is
    /// near.
    pub nominal: Option<f64>,
    /// For equal: what is equal.
    pub what: Option<EqualWhat>,
    /// For on-datum on a line, the direction's share too, degrees.
    pub also_deg: Option<f64>,
    /// The relation as the sketch takes it, ready to add.
    pub constraints: Vec<Rule>,
}

const EPS_MM: f64 = 1e-6;
/// A per-point sigma is never taken under this, mm.
const SIGMA_FLOOR: f64 = 1e-6;
/// Round angles between two lines a pair is held near, degrees — besides 0 and 90, which are
/// parallel and perpendicular.
const ROUND_ANGLES: [f64; 10] = [15.0, 30.0, 45.0, 60.0, 75.0, 105.0, 120.0, 135.0, 150.0, 165.0];
/// How near two pieces' ends have to be to make them neighbours when no relation joins them,
/// in the larger of their scatters — and never under `NEAR_FLOOR_MM`, for pieces drawn rather
/// than fitted.
const NEAR_SCATTERS: f64 = 4.0;
const NEAR_FLOOR_MM: f64 = 1e-3;

/// `Math.round` as JavaScript has it: halves round up, towards +∞.
fn js_round(x: f64) -> f64 {
    let f = x.floor();
    if x - f >= 0.5 {
        f + 1.0
    } else {
        f
    }
}

/// Four decimals, for an answer to be read rather than computed on.
pub fn round4(v: f64) -> f64 {
    js_round(v * 10_000.0) / 10_000.0
}

fn to_deg(r: f64) -> f64 {
    (r * 180.0) / std::f64::consts::PI
}

/// The root of the sum of squares, or `None` when any term is unknown or not finite.
fn hyp(v: &[Option<f64>]) -> Option<f64> {
    let mut sum = 0.0;
    for x in v {
        let x = (*x)?;
        if !x.is_finite() {
            return None;
        }
        sum += x * x;
    }
    Some(sum.sqrt())
}

fn over(off: f64, sigma: Option<f64>) -> Option<f64> {
    match sigma {
        Some(s) if s > 0.0 => Some(round4(off / s)),
        _ => None,
    }
}

// ---- the fits' own statistics ----

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LineStats {
    pub rms: f64,
    pub max: f64,
    pub n: usize,
    /// Radians.
    pub sigma_angle: Option<f64>,
    pub sigma_offset: Option<f64>,
    /// The per-point sigma the uncertainties rest on.
    pub sigma_point: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RoundStats {
    pub rms: f64,
    pub max: f64,
    pub n: usize,
    pub sigma_radius: Option<f64>,
    pub sigma_center: Option<f64>,
    pub sigma_point: f64,
}

/// The residuals of a line's points and the covariance of its fit: the direction is known to
/// σ/√Σ(t − t̄)², the offset to σ/√n. `None` for anything but a fitted line of some length.
pub fn line_stats(e: &Entity) -> Option<LineStats> {
    let Shape::Line { a, b } = e.shape else { return None };
    let pts = &e.points;
    if pts.is_empty() {
        return None;
    }
    let (dx, dy) = (b.x - a.x, b.y - a.y);
    let l = dx.hypot(dy);
    if l < EPS_MM {
        return None;
    }
    let (ux, uy) = (dx / l, dy / l);
    let n = pts.len();
    let (mut sum_sq, mut max, mut st) = (0.0, 0.0f64, 0.0);
    let mut ts = vec![0.0; n];
    for i in 0..n {
        let (px, py) = (pts[i].x - a.x, pts[i].y - a.y);
        let r = -uy * px + ux * py;
        sum_sq += r * r;
        if r.abs() > max {
            max = r.abs();
        }
        ts[i] = ux * px + uy * py;
        st += ts[i];
    }
    let rms = (sum_sq / n as f64).sqrt();
    if n < 3 {
        return Some(LineStats { rms, max, n, sigma_angle: None, sigma_offset: None, sigma_point: rms.max(e.scatter).max(SIGMA_FLOOR) });
    }
    let mean = st / n as f64;
    let mut stt = 0.0;
    for t in &ts {
        stt += (t - mean).powi(2);
    }
    let sigma_point = (sum_sq / (n - 2) as f64).sqrt().max(e.scatter).max(SIGMA_FLOOR);
    Some(LineStats {
        rms,
        max,
        n,
        sigma_angle: (stt > 1e-12).then(|| sigma_point / stt.sqrt()),
        sigma_offset: Some(sigma_point / (n as f64).sqrt()),
        sigma_point,
    })
}

/// The residuals of a round piece's points and the covariance of its fit, from the normal
/// matrix of the radial residuals: on a full circle the radius is known to σ/√n, on a short
/// arc to far less. `None` for anything but a fitted arc or circle.
pub fn round_stats(e: &Entity) -> Option<RoundStats> {
    let (c, radius) = match e.shape {
        Shape::Arc { c, r, .. } | Shape::Circle { c, r } => (c, r),
        _ => return None,
    };
    let pts = &e.points;
    if pts.is_empty() {
        return None;
    }
    let n = pts.len();
    let (mut sum_sq, mut max) = (0.0, 0.0f64);
    let (mut scc, mut sss, mut scs, mut sc, mut ss) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for p in pts {
        let (dx, dy) = (p.x - c.x, p.y - c.y);
        let rho = dx.hypot(dy);
        let r = rho - radius;
        sum_sq += r * r;
        if r.abs() > max {
            max = r.abs();
        }
        if rho < EPS_MM {
            continue;
        }
        let (cc, s) = (dx / rho, dy / rho);
        scc += cc * cc;
        sss += s * s;
        scs += cc * s;
        sc += cc;
        ss += s;
    }
    let rms = (sum_sq / n as f64).sqrt();
    let sigma_point = (if n > 3 { (sum_sq / (n - 3) as f64).sqrt() } else { rms }).max(e.scatter).max(SIGMA_FLOOR);
    if n < 4 {
        return Some(RoundStats { rms, max, n, sigma_radius: None, sigma_center: None, sigma_point });
    }
    // The normal matrix [[Σc², Σcs, Σc], [Σcs, Σs², Σs], [Σc, Σs, n]] inverted.
    let (a, b, cm, d, f, g) = (scc, scs, sc, sss, ss, n as f64);
    let det = a * (d * g - f * f) - b * (b * g - f * cm) + cm * (b * f - d * cm);
    let nf = n as f64;
    if !(det.abs() > 1e-14 * nf * nf * nf) {
        return Some(RoundStats { rms, max, n, sigma_radius: None, sigma_center: None, sigma_point });
    }
    let i00 = (d * g - f * f) / det;
    let i11 = (a * g - cm * cm) / det;
    let i22 = (a * d - b * b) / det;
    let v = sigma_point * sigma_point;
    Some(RoundStats {
        rms,
        max,
        n,
        sigma_radius: (i22 > 0.0).then(|| (v * i22).sqrt()),
        sigma_center: (i00 + i11 > 0.0).then(|| (v * (i00 + i11)).sqrt()),
        sigma_point,
    })
}

/// A fitted spline's residual against its points: each point's distance to the curve.
fn spline_residual(sp: &crate::sketch_spline::SplineShape, points: &[DVec2]) -> Option<Residual> {
    if points.is_empty() {
        return None;
    }
    let fit = crate::sketch_spline::spline_curve(sp);
    let miss: Vec<f64> = points.iter().map(|p| crate::sketch_spline::nearest_on_spline(&fit, *p, 8).dist).collect();
    let rms = (miss.iter().map(|m| m * m).sum::<f64>() / miss.len() as f64).sqrt();
    Some(Residual { rms, max: miss.iter().fold(f64::NEG_INFINITY, |a, b| a.max(*b)), points: miss.len() })
}

/// A fitted piece's residual against its points — rms and worst miss, mm — or `None` for a
/// piece without points.
fn piece_residual(e: &Entity) -> Option<Residual> {
    match e.shape {
        Shape::Line { .. } => line_stats(e).map(|s| Residual { rms: s.rms, max: s.max, points: s.n }),
        Shape::Arc { .. } | Shape::Circle { .. } => round_stats(e).map(|s| Residual { rms: s.rms, max: s.max, points: s.n }),
        Shape::Spline(ref sp) => spline_residual(sp, &e.points),
        Shape::Point { .. } => None,
    }
}

/// The per-point sigma a fitted piece's uncertainties rest on, mm — its own residual floored
/// at its chain's scatter, as the candidates take it — or `None` for a piece drawn rather than
/// fitted.
pub fn point_sigma(e: &Entity) -> Option<f64> {
    match e.shape {
        Shape::Line { .. } => line_stats(e).map(|s| s.sigma_point),
        Shape::Arc { .. } | Shape::Circle { .. } => round_stats(e).map(|s| s.sigma_point),
        Shape::Point { .. } | Shape::Spline(_) => piece_residual(e).map(|r| r.rms.max(e.scatter).max(SIGMA_FLOOR)),
    }
}

/// The piece that misses worst, and by how much.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WorstPiece {
    /// `None` when every piece sits exactly on its points.
    pub piece: Option<EntityId>,
    pub mm: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SketchResidual {
    pub rms: f64,
    pub max: f64,
    pub worst: WorstPiece,
    pub pieces: usize,
}

/// The residual over every fitted piece of a sketch: rms and worst miss, and which piece
/// misses worst. `None` when nothing was fitted.
pub fn sketch_residual(entities: &[Entity]) -> Option<SketchResidual> {
    let (mut sum_sq, mut n, mut max, mut worst, mut pieces) = (0.0, 0usize, 0.0, None, 0usize);
    for e in entities {
        let Some(r) = piece_residual(e) else { continue };
        pieces += 1;
        sum_sq += r.rms * r.rms * r.points as f64;
        n += r.points;
        if r.max > max {
            max = r.max;
            worst = Some(e.id);
        }
    }
    if n == 0 {
        return None;
    }
    Some(SketchResidual { rms: round4((sum_sq / n as f64).sqrt()), max: round4(max), worst: WorstPiece { piece: worst, mm: round4(max) }, pieces })
}

// ---- the pieces, read ----

pub fn rounding(measured: f64, kind: NominalKind, sigma: Option<f64>) -> Rounding {
    let step = nominal_step(measured, kind);
    let nominal = nominal_of(measured, kind).unwrap_or_else(|| js_round(measured / step) * step);
    let off = measured - nominal;
    Rounding { measured: round4(measured), nominal: round4(nominal), off: round4(off), sigmas: over(off.abs(), sigma) }
}

/// The direction a line runs at, degrees from +u, 0 to 180.
pub fn line_angle_deg(a: DVec2, b: DVec2) -> f64 {
    ((to_deg((b.y - a.y).atan2(b.x - a.x)) % 180.0) + 180.0) % 180.0
}

/// Every line, arc and circle of the sketch read: residual, uncertainty and nominal figures.
/// Construction pieces are left out.
pub fn read_pieces(entities: &[Entity]) -> Vec<PieceReading> {
    let mut out = Vec::new();
    for e in entities {
        if e.construction {
            continue;
        }
        match e.shape {
            Shape::Line { a, b } => {
                let s = line_stats(e);
                let angle_sigma = s.and_then(|s| s.sigma_angle).map(to_deg);
                let l = (b.x - a.x).hypot(b.y - a.y);
                out.push(PieceReading {
                    id: e.id,
                    kind: ShapeKind::Line,
                    residual: s.map(|s| Residual { rms: round4(s.rms), max: round4(s.max), points: s.n }),
                    sigma: s.and_then(|s| {
                        let angle = s.sigma_angle?;
                        Some(PieceSigma { angle_deg: Some(round4(to_deg(angle))), offset: s.sigma_offset.map(round4), ..Default::default() })
                    }),
                    nominal: PieceNominal {
                        length: Some(rounding(l, NominalKind::Length, s.map(|s| std::f64::consts::SQRT_2 * s.sigma_point))),
                        angle_deg: Some(rounding(line_angle_deg(a, b), NominalKind::Angle, angle_sigma)),
                        radius: None,
                    },
                });
            }
            Shape::Arc { r, .. } | Shape::Circle { r, .. } => {
                let s = round_stats(e);
                out.push(PieceReading {
                    id: e.id,
                    kind: e.shape.kind(),
                    residual: s.map(|s| Residual { rms: round4(s.rms), max: round4(s.max), points: s.n }),
                    sigma: s.and_then(|s| {
                        let radius = s.sigma_radius?;
                        Some(PieceSigma { radius: Some(round4(radius)), center: s.sigma_center.map(round4), ..Default::default() })
                    }),
                    nominal: PieceNominal { radius: Some(rounding(r, NominalKind::Radius, s.and_then(|s| s.sigma_radius))), ..Default::default() },
                });
            }
            Shape::Point { .. } | Shape::Spline(_) => {}
        }
    }
    out
}

// ---- the candidates ----

#[derive(Debug, Clone, Copy)]
struct LineRead {
    id: EntityId,
    a: DVec2,
    b: DVec2,
    u: DVec2,
    theta: f64,
    length: f64,
    mid: DVec2,
    sigma_angle_deg: Option<f64>,
    sigma_offset: Option<f64>,
    sigma_point: Option<f64>,
}

#[derive(Debug, Clone, Copy)]
struct RoundRead {
    id: EntityId,
    kind: ShapeKind,
    c: DVec2,
    r: f64,
    ends: Option<[DVec2; 2]>,
    sigma_radius: Option<f64>,
    sigma_center: Option<f64>,
    sigma_point: Option<f64>,
}

/// A line or a round piece, as the neighbour test takes either.
#[derive(Debug, Clone, Copy)]
enum Read<'a> {
    Line(&'a LineRead),
    Round(&'a RoundRead),
}

impl Read<'_> {
    fn id(&self) -> EntityId {
        match self {
            Read::Line(l) => l.id,
            Read::Round(r) => r.id,
        }
    }

    fn sigma_point(&self) -> Option<f64> {
        match self {
            Read::Line(l) => l.sigma_point,
            Read::Round(r) => r.sigma_point,
        }
    }
}

fn d2(p: DVec2, q: DVec2) -> f64 {
    (p.x - q.x).hypot(p.y - q.y)
}

/// A point's signed distance from a line's carrier, and where along it the foot lies, from
/// the line's start.
fn across_line(l: &LineRead, p: DVec2) -> (f64, f64) {
    let (px, py) = (p.x - l.a.x, p.y - l.a.y);
    (-l.u.y * px + l.u.x * py, l.u.x * px + l.u.y * py)
}

/// The undirected angle between two lines' directions, degrees 0 to 90.
fn between(a: &LineRead, b: &LineRead) -> f64 {
    let d = (a.theta - b.theta).abs();
    if d > 90.0 {
        180.0 - d
    } else {
        d
    }
}

/// A point's mirror image in a datum trace.
fn mirrored(t: &DatumTrace, p: DVec2) -> DVec2 {
    let off = t.offset(p);
    DVec2::new(p.x - 2.0 * off * t.n.x, p.y - 2.0 * off * t.n.y)
}

/// The kind a relation goes by, as the sketch stores it.
fn rule_kind(rule: &Rule) -> &'static str {
    match rule {
        Rule::Coincident { .. } => "coincident",
        Rule::Horizontal { .. } => "horizontal",
        Rule::Vertical { .. } => "vertical",
        Rule::Parallel { .. } => "parallel",
        Rule::Perpendicular { .. } => "perpendicular",
        Rule::Tangent { .. } => "tangent",
        Rule::Concentric { .. } => "concentric",
        Rule::Equal { .. } => "equal",
        Rule::PointOn { .. } => "pointOn",
        Rule::Collinear { .. } => "collinear",
        Rule::Symmetric { .. } => "symmetric",
        Rule::Length { .. } => "length",
        Rule::Distance { .. } => "distance",
        Rule::Radius { .. } => "radius",
        Rule::Diameter { .. } => "diameter",
        Rule::Angle { .. } => "angle",
    }
}

/// The coordinate plane a relation is held to, where it is one: a point or a line on it, or a
/// mirror image across it.
fn rule_datum(rule: &Rule) -> Option<DatumPlane> {
    match *rule {
        Rule::PointOn { on: OnTarget::Datum(d), .. } | Rule::Collinear { on: OnTarget::Datum(d), .. } => Some(d),
        Rule::Symmetric { about: MirrorAxis::Datum(d), .. } => Some(d),
        _ => None,
    }
}

fn datum_name(d: DatumPlane) -> &'static str {
    match d {
        DatumPlane::Xy => "xy",
        DatumPlane::Yz => "yz",
        DatumPlane::Zx => "zx",
    }
}

/// The key a relation goes by, to tell one already in the sketch from a candidate: kind, the
/// pieces, and the datum where there is one.
fn relation_key(kind: &str, pieces: &[EntityId], datum: Option<DatumPlane>) -> String {
    let mut sorted = pieces.to_vec();
    sorted.sort_unstable();
    let ids: Vec<String> = sorted.iter().map(|p| p.to_string()).collect();
    match datum {
        Some(d) => format!("{kind}:{}:{}", ids.join(","), datum_name(d)),
        None => format!("{kind}:{}", ids.join(",")),
    }
}

fn rule_key(rule: &Rule) -> String {
    relation_key(rule_kind(rule), &constraint_entities(rule), rule_datum(rule))
}

fn held_keys(constraints: &[Rule]) -> HashSet<String> {
    constraints.iter().map(rule_key).collect()
}

/// The neighbour test: whether two pieces are neighbours along the profile — an end of one
/// joined to the other by a relation the sketch holds (coincident, or an end held on it), or
/// their ends within a few scatters; for a circle, an end of the other on its rim, or the two
/// rims touching.
struct Neighbours {
    joined: HashSet<(EntityId, EntityId)>,
}

fn pair(p: EntityId, q: EntityId) -> (EntityId, EntityId) {
    if p < q {
        (p, q)
    } else {
        (q, p)
    }
}

fn ends_of_read(x: &Read) -> Option<[DVec2; 2]> {
    match x {
        Read::Line(l) => Some([l.a, l.b]),
        Read::Round(r) => r.ends,
    }
}

/// How far a circle's rim is from touching another piece.
fn apart(r: &RoundRead, o: &Read) -> f64 {
    match o {
        Read::Round(o) => {
            let dc = d2(r.c, o.c);
            (dc - (r.r + o.r)).abs().min((dc - (r.r - o.r).abs()).abs())
        }
        Read::Line(o) => {
            let t = 0f64.max(o.length.min(across_line(o, r.c).1));
            (d2(r.c, DVec2::new(o.a.x + t * o.u.x, o.a.y + t * o.u.y)) - r.r).abs()
        }
    }
}

impl Neighbours {
    fn new(held: &[Rule]) -> Self {
        let mut joined = HashSet::new();
        let is_end = |at: AnchorAt| at == AnchorAt::A || at == AnchorAt::B;
        for c in held {
            match *c {
                Rule::Coincident { p, q } if p.entity != q.entity && is_end(p.at) && is_end(q.at) => {
                    joined.insert(pair(p.entity, q.entity));
                }
                Rule::PointOn { p, on: OnTarget::Line(l), .. } if is_end(p.at) => {
                    joined.insert(pair(p.entity, l));
                }
                Rule::PointOn { p, on: OnTarget::Curve(l), .. } if is_end(p.at) => {
                    joined.insert(pair(p.entity, l));
                }
                _ => {}
            }
        }
        Self { joined }
    }

    fn adjacent(&self, x: Read, y: Read) -> bool {
        if self.joined.contains(&pair(x.id(), y.id())) {
            return true;
        }
        let near = (NEAR_SCATTERS * x.sigma_point().unwrap_or(0.0).max(y.sigma_point().unwrap_or(0.0))).max(NEAR_FLOOR_MM);
        let ex = ends_of_read(&x);
        let ey = ends_of_read(&y);
        if let (Some(ex), Some(ey)) = (ex, ey) {
            return ex.iter().any(|p| ey.iter().any(|q| d2(*p, *q) <= near));
        }
        let on_rim = |r: &RoundRead, p: DVec2| (d2(r.c, p) - r.r).abs();
        // A circle: the other's end on its rim, or the two touching.
        if let (Read::Round(yr), None) = (&y, ey) {
            return ex.is_some_and(|ex| ex.iter().any(|p| on_rim(yr, *p) <= near)) || apart(yr, &x) <= near;
        }
        if let (Read::Round(xr), None) = (&x, ex) {
            return ey.is_some_and(|ey| ey.iter().any(|p| on_rim(xr, *p) <= near)) || apart(xr, &y) <= near;
        }
        false
    }
}

/// A candidate as it is found, before its figures are rounded.
struct Found {
    kind: CandidateKind,
    pieces: Vec<EntityId>,
    datum: Option<DatumPlane>,
    off: f64,
    unit: OffUnit,
    sigma: Option<f64>,
    nominal: Option<f64>,
    what: Option<EqualWhat>,
    also_deg: Option<f64>,
    constraints: Vec<Rule>,
}

impl Found {
    fn new(kind: CandidateKind, pieces: Vec<EntityId>, off: f64, unit: OffUnit, sigma: Option<f64>, constraints: Vec<Rule>) -> Self {
        Self { kind, pieces, datum: None, off, unit, sigma, nominal: None, what: None, also_deg: None, constraints }
    }
}

/// The relations the pieces come within the net of, each with the evidence, nearest the mark
/// first. Relations the sketch already holds (`held`) are left out, and so are the pairs an
/// axis relation on both pieces already says: two level lines are parallel without being told.
pub fn candidates_of(entities: &[Entity], datums: &[DatumTrace], net: IntentNet, held: &[Rule], limit: usize) -> Vec<Candidate> {
    let mut lines: Vec<LineRead> = Vec::new();
    let mut rounds: Vec<RoundRead> = Vec::new();
    for e in entities {
        if e.construction {
            continue;
        }
        match e.shape {
            Shape::Line { a, b } => {
                let l = (b.x - a.x).hypot(b.y - a.y);
                if l < EPS_MM {
                    continue;
                }
                let s = line_stats(e);
                lines.push(LineRead {
                    id: e.id,
                    a,
                    b,
                    u: DVec2::new((b.x - a.x) / l, (b.y - a.y) / l),
                    theta: line_angle_deg(a, b),
                    length: l,
                    mid: DVec2::new((a.x + b.x) / 2.0, (a.y + b.y) / 2.0),
                    sigma_angle_deg: s.and_then(|s| s.sigma_angle).map(to_deg),
                    sigma_offset: s.and_then(|s| s.sigma_offset),
                    sigma_point: s.map(|s| s.sigma_point),
                });
            }
            Shape::Arc { c, r, .. } | Shape::Circle { c, r } => {
                let s = round_stats(e);
                let ends = match e.shape {
                    Shape::Arc { start, sweep, .. } => Some([
                        DVec2::new(c.x + r * start.cos(), c.y + r * start.sin()),
                        DVec2::new(c.x + r * (start + sweep).cos(), c.y + r * (start + sweep).sin()),
                    ]),
                    _ => None,
                };
                rounds.push(RoundRead {
                    id: e.id,
                    kind: e.shape.kind(),
                    c,
                    r,
                    ends,
                    sigma_radius: s.and_then(|s| s.sigma_radius),
                    sigma_center: s.and_then(|s| s.sigma_center),
                    sigma_point: s.map(|s| s.sigma_point),
                });
            }
            Shape::Point { .. } | Shape::Spline(_) => {}
        }
    }
    let mut out: Vec<Candidate> = Vec::new();
    let mut add = |c: Found| {
        out.push(Candidate {
            kind: c.kind,
            pieces: c.pieces,
            datum: c.datum,
            off: round4(c.off),
            unit: c.unit,
            sigma: c.sigma.map(round4),
            sigmas: over(c.off, c.sigma),
            nominal: c.nominal,
            what: c.what,
            also_deg: c.also_deg,
            constraints: c.constraints,
        })
    };

    let adjacent = Neighbours::new(held);

    // Lines along the sheet's axes.
    let mut on_axis: HashSet<EntityId> = HashSet::new();
    for l in &lines {
        let level = l.theta.min(180.0 - l.theta);
        let upright = (l.theta - 90.0).abs();
        if level <= net.angle_deg {
            on_axis.insert(l.id);
            add(Found::new(CandidateKind::Horizontal, vec![l.id], level, OffUnit::Deg, l.sigma_angle_deg, vec![Rule::Horizontal { entity: l.id }]));
        } else if upright <= net.angle_deg {
            on_axis.insert(l.id);
            add(Found::new(CandidateKind::Vertical, vec![l.id], upright, OffUnit::Deg, l.sigma_angle_deg, vec![Rule::Vertical { entity: l.id }]));
        }
    }

    // Pairs of lines.
    for i in 0..lines.len() {
        for j in i + 1..lines.len() {
            let (a, b) = (&lines[i], &lines[j]);
            let d = between(a, b);
            let sigma_angle = hyp(&[a.sigma_angle_deg, b.sigma_angle_deg]);
            let both = on_axis.contains(&a.id) && on_axis.contains(&b.id);
            if d <= net.angle_deg {
                // In line as well as parallel?
                let off_b = across_line(a, b.mid).0.abs();
                let off_a = across_line(b, a.mid).0.abs();
                let off = off_a.max(off_b);
                if off <= net.mm {
                    let gap = d2(a.mid, b.mid);
                    let rad = |deg: Option<f64>| deg.map(|d| (d * std::f64::consts::PI) / 180.0);
                    let sigma = hyp(&[a.sigma_offset, b.sigma_offset, rad(a.sigma_angle_deg).map(|r| gap * r), rad(b.sigma_angle_deg).map(|r| gap * r)]);
                    add(Found::new(CandidateKind::Collinear, vec![a.id, b.id], off, OffUnit::Mm, sigma, vec![Rule::Collinear { line: b.id, on: OnTarget::Line(a.id) }]));
                } else if !both {
                    add(Found::new(CandidateKind::Parallel, vec![a.id, b.id], d, OffUnit::Deg, sigma_angle, vec![Rule::Parallel { a: a.id, b: b.id }]));
                }
            } else if (d - 90.0).abs() <= net.angle_deg {
                if !both {
                    add(Found::new(CandidateKind::Perpendicular, vec![a.id, b.id], (d - 90.0).abs(), OffUnit::Deg, sigma_angle, vec![Rule::Perpendicular { a: a.id, b: b.id }]));
                }
            } else if adjacent.adjacent(Read::Line(a), Read::Line(b)) {
                // A round angle between neighbours — the directed angle, as the sketch's angle
                // dimension reads it.
                let phi = to_deg((a.u.x * b.u.x + a.u.y * b.u.y).clamp(-1.0, 1.0).acos());
                let mut best: Option<f64> = None;
                for n in ROUND_ANGLES {
                    if best.is_none_or(|b| (phi - n).abs() < (phi - b).abs()) {
                        best = Some(n);
                    }
                }
                if let Some(best) = best {
                    if (phi - best).abs() <= net.angle_deg {
                        let mut f = Found::new(CandidateKind::Angle, vec![a.id, b.id], (phi - best).abs(), OffUnit::Deg, sigma_angle, vec![Rule::Angle { a: a.id, b: b.id, value: best }]);
                        f.nominal = Some(best);
                        add(f);
                    }
                }
            }
            // Equal lengths: the ends are corners the fit put there, so the sigma is rough —
            // the noise of a point at each end.
            let dl = (a.length - b.length).abs();
            if dl <= net.mm {
                let sl = |l: &LineRead| l.sigma_point.map(|s| std::f64::consts::SQRT_2 * s);
                let mut f = Found::new(CandidateKind::Equal, vec![a.id, b.id], dl, OffUnit::Mm, hyp(&[sl(a), sl(b)]), vec![Rule::Equal { a: a.id, b: b.id }]);
                f.what = Some(EqualWhat::Length);
                add(f);
            }
        }
    }

    // Pairs of round pieces.
    for i in 0..rounds.len() {
        for j in i + 1..rounds.len() {
            let (a, b) = (&rounds[i], &rounds[j]);
            let dc = d2(a.c, b.c);
            let dr = (a.r - b.r).abs();
            // Two holes alike anywhere; two arcs of an outline side by side, or at either end
            // of one piece.
            let besides = (a.kind == ShapeKind::Circle && b.kind == ShapeKind::Circle)
                || adjacent.adjacent(Read::Round(a), Read::Round(b))
                || lines.iter().map(Read::Line).chain(rounds.iter().map(Read::Round)).any(|m| m.id() != a.id && m.id() != b.id && adjacent.adjacent(Read::Round(a), m) && adjacent.adjacent(m, Read::Round(b)));
            if dr <= net.mm && besides {
                let mut f = Found::new(CandidateKind::Equal, vec![a.id, b.id], dr, OffUnit::Mm, hyp(&[a.sigma_radius, b.sigma_radius]), vec![Rule::Equal { a: a.id, b: b.id }]);
                f.what = Some(EqualWhat::Radius);
                add(f);
            }
            if dc <= net.mm {
                add(Found::new(CandidateKind::Concentric, vec![a.id, b.id], dc, OffUnit::Mm, hyp(&[a.sigma_center, b.sigma_center]), vec![Rule::Concentric { a: a.id, b: b.id }]));
                continue;
            }
            let touch = (dc - (a.r + b.r)).abs().min((dc - (a.r - b.r).abs()).abs());
            if touch <= net.mm && adjacent.adjacent(Read::Round(a), Read::Round(b)) {
                add(Found::new(CandidateKind::Tangent, vec![a.id, b.id], touch, OffUnit::Mm, hyp(&[a.sigma_center, b.sigma_center, a.sigma_radius, b.sigma_radius]), vec![Rule::Tangent { a: a.id, b: b.id }]));
            }
        }
    }

    // A line tangent to a round piece — where the round piece lies beside the line, not
    // somewhere far along its carrier, and meets it.
    for l in &lines {
        for r in &rounds {
            let (off, t) = across_line(l, r.c);
            if t < -r.r || t > l.length + r.r || !adjacent.adjacent(Read::Line(l), Read::Round(r)) {
                continue;
            }
            let touch = (off.abs() - r.r).abs();
            if touch <= net.mm {
                add(Found::new(CandidateKind::Tangent, vec![l.id, r.id], touch, OffUnit::Mm, hyp(&[l.sigma_offset, r.sigma_center, r.sigma_radius]), vec![Rule::Tangent { a: l.id, b: r.id }]));
            }
        }
    }

    // On a coordinate plane, and mirror images across one.
    for t in datums {
        let dir = t.line().1;
        for l in &lines {
            let lean = to_deg((l.u.x * dir.x + l.u.y * dir.y).abs().min(1.0).acos());
            let off = t.offset(l.mid).abs();
            if lean <= net.angle_deg && off <= net.mm {
                let mut f = Found::new(CandidateKind::OnDatum, vec![l.id], off, OffUnit::Mm, l.sigma_offset, vec![Rule::Collinear { line: l.id, on: OnTarget::Datum(t.plane) }]);
                f.datum = Some(t.plane);
                f.also_deg = Some(round4(lean));
                add(f);
            }
        }
        for r in &rounds {
            let off = t.offset(r.c).abs();
            if off <= net.mm {
                let mut f = Found::new(CandidateKind::OnDatum, vec![r.id], off, OffUnit::Mm, r.sigma_center, vec![Rule::PointOn { p: Anchor::new(r.id, AnchorAt::Center), on: OnTarget::Datum(t.plane), bounded: false }]);
                f.datum = Some(t.plane);
                add(f);
            }
        }
        for i in 0..lines.len() {
            for j in i + 1..lines.len() {
                let (a, b) = (&lines[i], &lines[j]);
                let (ma, mb) = (mirrored(t, a.a), mirrored(t, a.b));
                let straight = d2(ma, b.a).max(d2(mb, b.b));
                let crossed = d2(ma, b.b).max(d2(mb, b.a));
                let off = straight.min(crossed);
                if off > net.mm {
                    continue;
                }
                let q = if straight <= crossed { [AnchorAt::A, AnchorAt::B] } else { [AnchorAt::B, AnchorAt::A] };
                let mut f = Found::new(
                    CandidateKind::Symmetric,
                    vec![a.id, b.id],
                    off,
                    OffUnit::Mm,
                    hyp(&[a.sigma_point, b.sigma_point]),
                    vec![
                        Rule::Symmetric { p: Anchor::new(a.id, AnchorAt::A), q: Anchor::new(b.id, q[0]), about: MirrorAxis::Datum(t.plane) },
                        Rule::Symmetric { p: Anchor::new(a.id, AnchorAt::B), q: Anchor::new(b.id, q[1]), about: MirrorAxis::Datum(t.plane) },
                    ],
                );
                f.datum = Some(t.plane);
                add(f);
            }
        }
        for i in 0..rounds.len() {
            for j in i + 1..rounds.len() {
                let (a, b) = (&rounds[i], &rounds[j]);
                if a.kind != b.kind {
                    continue;
                }
                let mut off = d2(mirrored(t, a.c), b.c).max((a.r - b.r).abs());
                if let (Some(ae), Some(be)) = (a.ends, b.ends) {
                    let (ma, mb) = (mirrored(t, ae[0]), mirrored(t, ae[1]));
                    off = off.max(d2(ma, be[0]).max(d2(mb, be[1])).min(d2(ma, be[1]).max(d2(mb, be[0]))));
                }
                if off > net.mm {
                    continue;
                }
                let mut f = Found::new(
                    CandidateKind::Symmetric,
                    vec![a.id, b.id],
                    off,
                    OffUnit::Mm,
                    hyp(&[a.sigma_center, b.sigma_center, a.sigma_radius, b.sigma_radius]),
                    vec![Rule::Symmetric { p: Anchor::new(a.id, AnchorAt::Center), q: Anchor::new(b.id, AnchorAt::Center), about: MirrorAxis::Datum(t.plane) }, Rule::Equal { a: a.id, b: b.id }],
                );
                f.datum = Some(t.plane);
                add(f);
            }
        }
    }

    let have = held_keys(held);
    let mut kept: Vec<Candidate> = out.into_iter().filter(|c| !c.constraints.iter().all(|k| have.contains(&rule_key(k)))).collect();
    // Nearest the mark first, by sigmas where known, by the plain offset where not — then
    // capped.
    // ScanRuler's `(x.sigmas ?? ∞) − (y.sigmas ?? ∞) || x.off − y.off` as a total order (the
    // sort may not be handed one that is not): the sigmas first, the offset where they tie —
    // two unknown sigmas tie, as ∞ − ∞ falls through to the offset there. A negative zero is
    // read as zero, as the subtraction reads it.
    let total = |a: f64, b: f64| (a + 0.0).total_cmp(&(b + 0.0));
    kept.sort_by(|x, y| total(x.sigmas.unwrap_or(f64::INFINITY), y.sigmas.unwrap_or(f64::INFINITY)).then_with(|| total(x.off, y.off)));
    let mut seen = HashSet::new();
    let mut unique = Vec::new();
    for c in kept {
        let key = match c.constraints.first() {
            Some(first) => relation_key(rule_kind(first), &c.pieces, c.datum),
            None => relation_key(&format!("{:?}", c.kind), &c.pieces, c.datum),
        };
        if !seen.insert(key) {
            continue;
        }
        unique.push(c);
    }
    unique.truncate(limit);
    unique
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datums::datum_traces;
    use crate::frame::SectionFrame;
    use crate::sketch::Sketch;
    use glam::DVec3;
    use std::f64::consts::PI;

    /// Deterministic noise in ±half (the tests' linear congruential generator).
    struct Noise(u32);

    impl Noise {
        fn next(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
            self.0 as f64 / 4294967296.0 - 0.5
        }
    }

    /// `n` points along a → b scattered ±`spread` across it.
    fn line_points(a: DVec2, b: DVec2, n: usize, spread: f64, seed: u32) -> Vec<DVec2> {
        let mut rand = Noise(seed);
        let (dx, dy) = (b.x - a.x, b.y - a.y);
        let l = dx.hypot(dy);
        let (nx, ny) = (-dy / l, dx / l);
        (0..n)
            .map(|i| {
                let t = i as f64 / (n - 1) as f64;
                let e = rand.next() * 2.0 * spread;
                DVec2::new(a.x + dx * t + nx * e, a.y + dy * t + ny * e)
            })
            .collect()
    }

    /// `n` points round an arc about c of radius r from `start` through `sweep`, scattered
    /// ±`spread` along the radius.
    fn arc_points(c: DVec2, r: f64, start: f64, sweep: f64, n: usize, spread: f64, seed: u32) -> Vec<DVec2> {
        let mut rand = Noise(seed);
        (0..n)
            .map(|i| {
                let t = start + (sweep * i as f64) / (n - 1) as f64;
                let rr = r + rand.next() * 2.0 * spread;
                DVec2::new(c.x + rr * t.cos(), c.y + rr * t.sin())
            })
            .collect()
    }

    fn line(sk: &mut Sketch, a: [f64; 2], b: [f64; 2], n: usize, spread: f64, seed: u32) -> EntityId {
        let (a, b) = (DVec2::from(a), DVec2::from(b));
        sk.add(Entity::new(Shape::Line { a, b }).fitted(line_points(a, b, n, spread, seed), spread))
    }

    fn line_d(sk: &mut Sketch, a: [f64; 2], b: [f64; 2]) -> EntityId {
        line(sk, a, b, 40, 0.02, 1)
    }

    #[allow(clippy::too_many_arguments)]
    fn arc(sk: &mut Sketch, c: [f64; 2], r: f64, start: f64, sweep: f64, n: usize, spread: f64, seed: u32) -> EntityId {
        let c = DVec2::from(c);
        sk.add(Entity::new(Shape::Arc { c, r, start, sweep }).fitted(arc_points(c, r, start, sweep, n, spread, seed), spread))
    }

    fn circle(sk: &mut Sketch, c: [f64; 2], r: f64, n: usize, spread: f64, seed: u32) -> EntityId {
        let c = DVec2::from(c);
        sk.add(Entity::new(Shape::Circle { c, r }).fitted(arc_points(c, r, 0.0, 2.0 * PI, n, spread, seed), spread))
    }

    fn find<'a>(cs: &'a [Candidate], kind: CandidateKind, pieces: &[EntityId]) -> Option<&'a Candidate> {
        cs.iter().find(|c| c.kind == kind && pieces.iter().all(|p| c.pieces.contains(p)))
    }

    fn close(a: f64, b: f64, digits: i32) -> bool {
        (a - b).abs() < 0.5 * 10f64.powi(-digits)
    }

    fn xy() -> SectionFrame {
        SectionFrame { origin: DVec3::ZERO, normal: DVec3::Z, u: DVec3::X, v: DVec3::Y }
    }

    #[test]
    fn knows_a_long_sides_direction_far_better_than_a_stubs_and_says_how_well_each_sits() {
        let mut sk = Sketch::default();
        line(&mut sk, [0.0, 0.0], [40.0, 0.0], 40, 0.02, 1);
        line(&mut sk, [0.0, 5.0], [4.0, 5.0], 8, 0.02, 4);
        let s_long = line_stats(&sk.entities[0]).unwrap();
        let s_stub = line_stats(&sk.entities[1]).unwrap();
        assert!(s_long.rms < 0.03);
        assert!(s_long.max < 0.05);
        assert_eq!(s_long.n, 40);
        assert!(s_long.sigma_angle.is_some());
        assert!(s_stub.sigma_angle.unwrap() > 5.0 * s_long.sigma_angle.unwrap());
        // A piece without points, or with two, has a residual but no sigma.
        sk.add_shape(Shape::Line { a: DVec2::new(0.0, 9.0), b: DVec2::new(9.0, 9.0) });
        assert!(line_stats(&sk.entities[2]).is_none());
        sk.add(Entity::new(Shape::Line { a: DVec2::new(0.0, 12.0), b: DVec2::new(9.0, 12.0) }).fitted(vec![DVec2::new(0.0, 12.01), DVec2::new(9.0, 11.99)], 0.0));
        let two = line_stats(&sk.entities[3]).unwrap();
        assert_eq!(two.n, 2);
        assert!(two.sigma_angle.is_none());
    }

    #[test]
    fn knows_a_full_circles_radius_well_and_a_short_arcs_poorly() {
        let mut sk = Sketch::default();
        circle(&mut sk, [0.0, 0.0], 5.0, 60, 0.02, 3);
        arc(&mut sk, [20.0, 0.0], 5.0, 0.0, (15.0 * PI) / 180.0, 12, 0.02, 2);
        let full = round_stats(&sk.entities[0]).unwrap();
        let short = round_stats(&sk.entities[1]).unwrap();
        assert!(full.sigma_radius.unwrap() < 0.01);
        assert!(short.sigma_radius.unwrap() > 10.0 * full.sigma_radius.unwrap());
        assert!(short.sigma_center.unwrap() > full.sigma_center.unwrap());
    }

    #[test]
    fn reads_each_piece_residual_sigma_and_the_round_figure_beside_the_measured_one() {
        let mut sk = Sketch::default();
        line_d(&mut sk, [0.0, 0.0], [39.98, 0.0]);
        arc(&mut sk, [50.0, 0.0], 4.98, 0.0, PI / 2.0, 40, 0.01, 2);
        let t = (44.6 * PI) / 180.0;
        line(&mut sk, [0.0, 10.0], [10.0 * t.cos(), 10.0 + 10.0 * t.sin()], 40, 0.01, 7);
        let read = read_pieces(&sk.entities);
        let (l, a, d) = (&read[0], &read[1], &read[2]);
        assert_eq!((l.id, l.kind), (1, ShapeKind::Line));
        assert_eq!(l.residual.unwrap().points, 40);
        let length = l.nominal.length.unwrap();
        assert_eq!((length.measured, length.nominal), (39.98, 40.0));
        assert!(close(length.off, -0.02, 6));
        let angle = l.nominal.angle_deg.unwrap();
        assert_eq!((angle.measured, angle.nominal, angle.off), (0.0, 0.0, 0.0));
        let radius = a.nominal.radius.unwrap();
        assert_eq!((radius.measured, radius.nominal), (4.98, 5.0));
        assert!(radius.sigmas.unwrap() > 0.0);
        let diag = d.nominal.angle_deg.unwrap();
        assert_eq!((diag.measured, diag.nominal), (44.6, 45.0));
        assert!(close(diag.off, -0.4, 6));
        // The sketch's residual as a whole names the worst piece.
        let whole = sketch_residual(&sk.entities).unwrap();
        assert_eq!(whole.pieces, 3);
        assert!(whole.rms > 0.0);
        assert!([1, 2, 3].contains(&whole.worst.piece.unwrap()));
    }

    #[test]
    fn offers_level_upright_equal_and_keeps_what_is_held_out() {
        let mut sk = Sketch::default();
        let bottom = line_d(&mut sk, [0.0, 0.0], [40.0, 0.0]);
        let right = line(&mut sk, [40.0, 0.0], [40.0, 20.0], 30, 0.02, 2);
        let top = line(&mut sk, [40.0, 20.0], [0.0, 20.2], 40, 0.02, 3);
        let left = line(&mut sk, [0.0, 20.2], [0.0, 0.0], 30, 0.02, 4);
        let net = IntentNet { angle_deg: 5.0, mm: 1.0 };
        let cs = candidates_of(&sk.entities, &[], net, &[], DEFAULT_LIMIT);
        let top_level = find(&cs, CandidateKind::Horizontal, &[top]).unwrap();
        assert!(close(top_level.off, 0.2865, 3));
        assert_eq!(top_level.unit, OffUnit::Deg);
        assert!(top_level.sigma.unwrap() > 0.0);
        // A third of a degree on a 40 mm side with 0.02 mm of noise is many sigmas: the top
        // was measured leaning.
        assert!(top_level.sigmas.unwrap() > 3.0);
        assert_eq!(top_level.constraints, vec![Rule::Horizontal { entity: top }]);
        assert!(find(&cs, CandidateKind::Horizontal, &[bottom]).unwrap().sigmas.unwrap() < 2.0);
        assert!(find(&cs, CandidateKind::Vertical, &[right]).is_some());
        assert!(find(&cs, CandidateKind::Vertical, &[left]).is_some());
        // Two lines held level are parallel without being told, so the pair is not offered;
        // nor is the right angle between a level and an upright.
        assert!(find(&cs, CandidateKind::Parallel, &[bottom, top]).is_none());
        assert!(find(&cs, CandidateKind::Perpendicular, &[bottom, right]).is_none());
        // Equal lengths, the sides as well as the ends.
        assert_eq!(find(&cs, CandidateKind::Equal, &[bottom, top]).unwrap().what, Some(EqualWhat::Length));
        assert!(close(find(&cs, CandidateKind::Equal, &[right, left]).unwrap().off, 0.2, 3));
        // Nearest the mark first.
        for i in 1..cs.len() {
            assert!(cs[i].sigmas.unwrap_or(f64::INFINITY) >= cs[i - 1].sigmas.unwrap_or(f64::INFINITY));
        }
        // What the sketch already holds is not offered again.
        let held = candidates_of(&sk.entities, &[], net, &[Rule::Horizontal { entity: top }], DEFAULT_LIMIT);
        assert!(find(&held, CandidateKind::Horizontal, &[top]).is_none());
        assert!(find(&held, CandidateKind::Horizontal, &[bottom]).is_some());
    }

    #[test]
    fn offers_parallel_perpendicular_and_in_line_off_the_axes_and_a_45_degree_chamfer() {
        let mut sk = Sketch::default();
        let t = |deg: f64| (deg * PI) / 180.0;
        let a = line_d(&mut sk, [0.0, 0.0], [20.0 * t(30.0).cos(), 20.0 * t(30.0).sin()]);
        let b = line(&mut sk, [0.0, 5.0], [20.0 * t(30.3).cos(), 5.0 + 20.0 * t(30.3).sin()], 40, 0.02, 5);
        let c = line(&mut sk, [10.0, 0.0], [10.0 + 10.0 * t(120.4).cos(), 10.0 * t(120.4).sin()], 40, 0.02, 6);
        let d = line(&mut sk, [30.0, 30.0], [30.0 + 10.0 * t(30.1).cos(), 30.0 + 10.0 * t(30.1).sin()], 20, 0.02, 7);
        let net = IntentNet { angle_deg: 5.0, mm: 1.0 };
        let cs = candidates_of(&sk.entities, &[], net, &[], DEFAULT_LIMIT);
        assert!(close(find(&cs, CandidateKind::Parallel, &[a, b]).unwrap().off, 0.3, 3));
        assert!(close(find(&cs, CandidateKind::Perpendicular, &[a, c]).unwrap().off, 0.4, 3));
        assert!(close(find(&cs, CandidateKind::Parallel, &[a, d]).unwrap().off, 0.1, 3));
        // A chamfer: two lines meeting near 45°, the directed angle offered as the sketch's
        // angle dimension reads it.
        let mut sk2 = Sketch::default();
        let e = line_d(&mut sk2, [0.0, 0.0], [10.0, 0.0]);
        let f = line(&mut sk2, [10.0, 0.0], [10.0 + 5.0 * t(45.4).cos(), 5.0 * t(45.4).sin()], 20, 0.01, 8);
        let cs2 = candidates_of(&sk2.entities, &[], DEFAULT_NET, &[], DEFAULT_LIMIT);
        let chamfer = find(&cs2, CandidateKind::Angle, &[e, f]).unwrap();
        assert_eq!(chamfer.nominal, Some(45.0));
        assert!(close(chamfer.off, 0.4, 3));
        assert_eq!(chamfer.constraints, vec![Rule::Angle { a: e, b: f, value: 45.0 }]);
        // Two lines in line, 0.05 apart.
        let mut sk3 = Sketch::default();
        let g = line_d(&mut sk3, [0.0, 0.0], [10.0, 0.0]);
        let h = line(&mut sk3, [15.0, 0.05], [25.0, 0.05], 40, 0.02, 9);
        let cs3 = candidates_of(&sk3.entities, &[], DEFAULT_NET, &[], DEFAULT_LIMIT);
        let inline = find(&cs3, CandidateKind::Collinear, &[g, h]).unwrap();
        assert!(close(inline.off, 0.05, 3));
        assert_eq!(inline.constraints, vec![Rule::Collinear { line: h, on: OnTarget::Line(g) }]);
        assert!(find(&cs3, CandidateKind::Parallel, &[g, h]).is_none());
    }

    #[test]
    fn offers_tangent_concentric_and_equal_radii_between_round_pieces_and_lines() {
        let mut sk = Sketch::default();
        let l = line_d(&mut sk, [0.0, 0.0], [12.0, 0.0]);
        let fillet = arc(&mut sk, [12.0, 2.03], 2.0, PI, PI / 2.0, 30, 0.005, 2);
        let bore = circle(&mut sk, [30.0, 10.0], 3.0, 60, 0.02, 3);
        let boss = circle(&mut sk, [30.02, 10.0], 6.0, 60, 0.02, 4);
        // An arc ending where the fillet starts, and one of its size far off.
        let next = arc(&mut sk, [8.0, 2.03], 2.01, 0.0, PI / 2.0, 30, 0.005, 5);
        let other = arc(&mut sk, [50.0, 10.0], 2.01, 0.0, PI / 2.0, 30, 0.005, 7);
        let far = arc(&mut sk, [80.0, 40.0], 2.0, 0.0, PI / 2.0, 30, 0.005, 6);
        let hole = circle(&mut sk, [60.0, -20.0], 3.02, 60, 0.02, 8);
        let cs = candidates_of(&sk.entities, &[], DEFAULT_NET, &[], DEFAULT_LIMIT);
        let tangent = find(&cs, CandidateKind::Tangent, &[l, fillet]).unwrap();
        assert!(close(tangent.off, 0.03, 3));
        assert_eq!(tangent.constraints, vec![Rule::Tangent { a: l, b: fillet }]);
        assert!(close(find(&cs, CandidateKind::Concentric, &[bore, boss]).unwrap().off, 0.02, 3));
        assert!(find(&cs, CandidateKind::Equal, &[bore, boss]).is_none());
        assert_eq!(find(&cs, CandidateKind::Equal, &[fillet, next]).unwrap().what, Some(EqualWhat::Radius));
        assert!(close(find(&cs, CandidateKind::Equal, &[fillet, next]).unwrap().off, 0.01, 3));
        // Arcs of an outline apart from each other are not offered as equal; two holes are,
        // wherever they are.
        assert!(find(&cs, CandidateKind::Equal, &[fillet, other]).is_none());
        assert_eq!(find(&cs, CandidateKind::Equal, &[bore, hole]).unwrap().what, Some(EqualWhat::Radius));
        // A round piece far along the line's carrier is not tangent to it.
        assert!(find(&cs, CandidateKind::Tangent, &[l, far]).is_none());
    }

    #[test]
    fn reads_the_parts_coordinate_planes_on_one_and_mirror_images_across_one() {
        let mut sk = Sketch::default();
        let traces = datum_traces(&xy());
        let mut planes: Vec<DatumPlane> = traces.iter().map(|t| t.plane).collect();
        planes.sort_by_key(|p| datum_name(*p));
        assert_eq!(planes, vec![DatumPlane::Yz, DatumPlane::Zx]);
        let hole = circle(&mut sk, [0.03, 7.0], 2.0, 60, 0.02, 3);
        let wall_l = line_d(&mut sk, [-10.0, 0.0], [-10.0, 8.0]);
        let wall_r = line(&mut sk, [10.05, 0.0], [10.05, 8.0], 40, 0.02, 2);
        let floor = line(&mut sk, [-10.0, 0.02], [10.0, 0.02], 40, 0.02, 3);
        let cs = candidates_of(&sk.entities, &traces, DEFAULT_NET, &[], DEFAULT_LIMIT);
        let centred = find(&cs, CandidateKind::OnDatum, &[hole]).unwrap();
        assert_eq!((centred.datum, centred.unit), (Some(DatumPlane::Yz), OffUnit::Mm));
        assert!(close(centred.off, 0.03, 3));
        assert_eq!(centred.constraints, vec![Rule::PointOn { p: Anchor::new(hole, AnchorAt::Center), on: OnTarget::Datum(DatumPlane::Yz), bounded: false }]);
        let on_floor = cs.iter().find(|c| c.kind == CandidateKind::OnDatum && c.pieces[0] == floor).unwrap();
        assert_eq!(on_floor.datum, Some(DatumPlane::Zx));
        assert!(close(on_floor.off, 0.02, 3));
        assert_eq!(on_floor.constraints, vec![Rule::Collinear { line: floor, on: OnTarget::Datum(DatumPlane::Zx) }]);
        let mirror = find(&cs, CandidateKind::Symmetric, &[wall_l, wall_r]).unwrap();
        assert_eq!(mirror.datum, Some(DatumPlane::Yz));
        assert!(close(mirror.off, 0.05, 3));
        assert_eq!(mirror.constraints.len(), 2);
        assert!(matches!(mirror.constraints[0], Rule::Symmetric { about: MirrorAxis::Datum(DatumPlane::Yz), .. }));
        // The net decides what is offered at all.
        let tight = candidates_of(&sk.entities, &traces, IntentNet { angle_deg: 5.0, mm: 0.01 }, &[], DEFAULT_LIMIT);
        assert!(find(&tight, CandidateKind::Symmetric, &[wall_l, wall_r]).is_none());
        assert!(find(&tight, CandidateKind::OnDatum, &[hole]).is_none());
        // And the cap.
        assert_eq!(candidates_of(&sk.entities, &traces, DEFAULT_NET, &[], 2).len(), 2);
    }
}
