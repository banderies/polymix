// SPDX-License-Identifier: AGPL-3.0-only
//! Rounds and chamfers read across curved edges. Ported from ScanRuler
//! (`plugins/reverse/core/curvedEdge.ts`), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! A circle or an arc between a flat face and a cylinder, a cone or a sphere (a boss's foot, a
//! hole's mouth, a plate outline's corner, a bowl's rim), and a straight edge between a flat
//! face and a cylinder are read as `fillet_estimate` reads a straight edge between two flats:
//! the scan cut by the plane square to the edge at a few places along it, the round fitted in
//! each cut between the faces' traces. Cut square to a circle, the plane holds the axis: a
//! cylinder's trace is a line along it, a cone's a line at its angle, a sphere's a circle
//! about a centre on it. Cut square to a straight edge, a cylinder's trace is a circle.
//!
//! The model gives each trace. Which way along it the face runs is read off the scan — the
//! face's points lie on one side of the corner only — and the flat face's outward normal,
//! which the model gives with its sign, then says whether the corner is convex (a round takes
//! material off) or concave (a fillet adds it, at a boss's foot). A concave corner is read as
//! the convex one its normals turned round make: the same two faces, the same round.
//!
//! An edge of any other shape — a blade's foot on a dish — is read off the body's mesh instead
//! of its faces' surfaces; see `read_free_cut` below.

use std::f64::consts::{PI, TAU};

use glam::{DVec2, DVec3};
use serde::{Deserialize, Serialize};

use crate::fillet_estimate::{
    corner_window, cut_chains, estimate_chamfer, estimate_chamfer_setbacks, estimate_fillet_radius, face_end_why, face_window, few_why, golden, log_search, unit3, ChamferSetbacks, FaceWindow,
    FilletReading, Miss, ReadShape,
};
use crate::fillet_num::{js_num, js_round};
use crate::slice::SectionCut;
use crate::topo::{EdgeFingerprint, EdgeKind, EdgeSide, FaceFingerprint, FaceKind};

/// A face of the edge as the reading takes it: kind and direction (the edge's fingerprint has
/// them), radius and centre where the body's topology gives them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EdgeSurface {
    pub kind: FaceKind,
    pub dir: DVec3,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub radius: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub center: Option<DVec3>,
}

impl From<&EdgeSide> for EdgeSurface {
    fn from(f: &EdgeSide) -> Self {
        EdgeSurface { kind: f.kind, dir: f.dir, radius: None, center: None }
    }
}

impl From<&FaceFingerprint> for EdgeSurface {
    fn from(f: &FaceFingerprint) -> Self {
        EdgeSurface { kind: f.kind, dir: f.dir, radius: f.radius, center: f.center }
    }
}

/// A curved edge as the reading takes it: its fingerprint, its two ends where the body gives
/// them (a circle edge needs them), and its faces' surfaces in the fingerprint's order, from
/// the topology.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CurvedEdge {
    pub fingerprint: EdgeFingerprint,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ends: Option<[DVec3; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub surfaces: Option<Vec<Option<EdgeSurface>>>,
}

/// Where a cut is taken: the corner, the plane's normal, and the plane's axes — for a circle,
/// x out from the axis and y along it.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Station {
    pub at: DVec3,
    pub normal: DVec3,
    pub x: DVec3,
    pub y: DVec3,
    /// Where along the edge, 0 at its first end and 1 at its other — on a closed edge, from
    /// its seam one way round.
    pub t: Option<f64>,
}

/// A face's trace in a cut: a line through the corner, or a circle about `q` of radius `r`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TraceCurve {
    Line,
    Circle { q: DVec2, r: f64 },
}

/// A face's trace in a cut: a line through the corner with normal `m`, or a circle; `d` the
/// way the face runs from the corner. `m` is the normal at the corner with the sign the
/// reading gives it: the round's centre lies where every trace's signed distance is −r.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Trace2 {
    pub curve: TraceCurve,
    pub m: DVec2,
    pub d: DVec2,
}

impl Trace2 {
    pub fn is_line(&self) -> bool {
        matches!(self.curve, TraceCurve::Line)
    }

    /// The same trace with its normal turned round.
    fn turned(self) -> Trace2 {
        Trace2 { m: -self.m, ..self }
    }
}

/// A countersink as a drawing calls it out: the included angle, the diameter where it meets
/// the flat face, and its depth along the axis, with their sigmas.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Countersink {
    pub angle_deg: f64,
    pub angle_sigma: f64,
    pub diameter: f64,
    pub diameter_sigma: f64,
    pub depth: f64,
    pub depth_sigma: f64,
}

/// A station's reading: the round (or even chamfer), and for a chamfer the setbacks apart
/// and, on a hole's mouth, the countersink's cone.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CurvedRead {
    pub reading: FilletReading,
    pub setbacks: Option<ChamferSetbacks>,
    pub countersink: Option<Countersink>,
}

impl CurvedRead {
    fn of(reading: FilletReading) -> CurvedRead {
        CurvedRead { reading, setbacks: None, countersink: None }
    }
}

/// How far a round face's axis may lean off the edge's and still be it: cos 3°.
fn cos_axis() -> f64 {
    (3.0f64).to_radians().cos()
}

/// Faces meeting within this many degrees of tangent have no corner: sin 10°.
fn sin_tangent() -> f64 {
    (10.0f64).to_radians().sin()
}

/// A radius as JavaScript tests one: set, and neither 0 nor NaN.
fn truthy(r: f64) -> bool {
    r != 0.0 && !r.is_nan()
}

fn surface_of(edge: &CurvedEdge, i: usize) -> Option<EdgeSurface> {
    if let Some(Some(s)) = edge.surfaces.as_ref().and_then(|s| s.get(i)) {
        return Some(s.clone());
    }
    edge.fingerprint.faces.get(i).map(EdgeSurface::from)
}

/// Whether the curved reader is the one for an edge: a circle, or a straight edge with a
/// round face on a side.
pub fn is_curved_edge(fp: &EdgeFingerprint) -> bool {
    fp.kind == EdgeKind::Circle || (fp.kind == EdgeKind::Line && fp.faces.len() >= 2 && fp.faces.iter().any(|f| f.kind != FaceKind::Plane))
}

/// Why a curved edge cannot be read, or `None` when it can.
pub fn curved_unreadable(edge: &CurvedEdge) -> Option<String> {
    let fp = &edge.fingerprint;
    if fp.faces.len() < 2 {
        return Some("an edge with one face — nothing to round between".to_string());
    }
    let (Some(s0), Some(s1)) = (surface_of(edge, 0), surface_of(edge, 1)) else {
        return Some("an edge with one face — nothing to round between".to_string());
    };
    let s = [s0, s1];
    let Some(flat) = s.iter().position(|f| f.kind == FaceKind::Plane) else {
        return Some(format!("an edge between a {} and a {} — a curved edge is read with a flat face on one side", s[0].kind.name(), s[1].kind.name()));
    };
    let other = &s[1 - flat];
    let (Some(axis), Some(od)) = (unit3(fp.dir), unit3(other.dir)) else {
        return Some("an edge without a direction".to_string());
    };
    if fp.kind == EdgeKind::Circle {
        if !(fp.radius.unwrap_or(f64::NAN) > 0.0) {
            return Some("a circle edge without its radius".to_string());
        }
        if edge.ends.is_none() {
            return Some("a circle edge known by its fingerprint only — pick it on the body".to_string());
        }
        if other.kind == FaceKind::Cylinder || other.kind == FaceKind::Cone {
            return if od.dot(axis).abs() >= cos_axis() { None } else { Some(format!("a circle edge on a {} of another axis", other.kind.name())) };
        }
        if other.kind == FaceKind::Sphere {
            return if other.radius.is_some_and(truthy) { None } else { Some("a circle edge on a sphere whose radius is not known here — pick it on the body".to_string()) };
        }
        return Some(format!("a circle edge on a {} face — a cylinder, a cone or a sphere is read", other.kind.name()));
    }
    if fp.kind == EdgeKind::Line {
        if other.kind != FaceKind::Cylinder {
            return Some(format!("an edge between a plane and a {} — a straight edge is read between two flat faces or a flat face and a cylinder", other.kind.name()));
        }
        let (Some(_), Some(center)) = (other.radius.filter(|r| truthy(*r)), other.center) else {
            return Some("an edge on a cylinder whose radius is not known here — pick it on the body".to_string());
        };
        if od.dot(axis).abs() < cos_axis() {
            return Some("a straight edge across a cylinder".to_string());
        }
        let off = fp.mid - center;
        let radial = unit3(off - od * off.dot(od));
        let n = unit3(s[flat].dir);
        if let (Some(radial), Some(n)) = (radial, n) {
            if radial.dot(n).abs() > (10.0f64).to_radians().cos() {
                return Some("a smooth edge — its faces meet tangent, there is no corner to read".to_string());
            }
        }
        return None;
    }
    Some(format!("a {} edge — a straight edge or a circle is read", fp.kind.name()))
}

/// The places along a curved edge a cut is taken: for a circle, evenly round it — over the
/// middle `share` of an arc, all round a full one; for a straight edge, over its middle.
/// `None` where the edge's geometry falls short.
pub fn curved_stations(edge: &CurvedEdge, n: usize, share: f64) -> Option<Vec<Station>> {
    let fp = &edge.fingerprint;
    let axis = unit3(fp.dir)?;
    if fp.kind == EdgeKind::Circle {
        let r = fp.radius?;
        let [e0, e1] = edge.ends?;
        // The centre lies a radius from the middle towards the chord's middle — for an arc of
        // any span, and for a full circle, whose ends meet opposite its middle.
        let chord = (e0 + e1) / 2.0;
        let mut v = chord - fp.mid;
        v -= axis * v.dot(axis);
        let lv = v.length();
        if !(lv > 1e-9) {
            return None;
        }
        let c = fp.mid + v * (r / lv);
        let rho0 = v * (-1.0 / lv);
        let w = axis.cross(rho0);
        let span = TAU.min(fp.length / r);
        let full = span > TAU - 1e-3;
        let thetas: Vec<f64> = if full {
            (0..n).map(|i| TAU * i as f64 / n as f64).collect()
        } else if n == 1 {
            vec![0.0]
        } else {
            (0..n).map(|i| span * share * (i as f64 / (n - 1) as f64 - 0.5)).collect()
        };
        let rho_at = |t: f64| {
            let raw = rho0 * t.cos() + w * t.sin();
            unit3(raw).unwrap_or(raw)
        };
        // Which way round the first end lies: the arc's end at −span/2 nearer it, or the
        // other; a full circle's ends meet opposite its middle.
        let back = c + rho_at(-span / 2.0) * r;
        let sign = if full || (back - e0).length() <= (back - e1).length() { 1.0 } else { -1.0 };
        return Some(
            thetas
                .into_iter()
                .map(|t| {
                    let rho = rho_at(t);
                    let along = if full { ((t + PI) / TAU) % 1.0 } else { 0.5 + sign * t / span };
                    Station { at: c + rho * r, normal: axis.cross(rho), x: rho, y: axis, t: Some(along) }
                })
                .collect(),
        );
    }
    let flat = [surface_of(edge, 0), surface_of(edge, 1)].into_iter().flatten().find(|f| f.kind == FaceKind::Plane)?;
    let y = unit3(flat.dir - axis * flat.dir.dot(axis))?;
    let x = y.cross(axis);
    let half = fp.length / 2.0;
    let [a, b] = edge.ends.unwrap_or([fp.mid - axis * half, fp.mid + axis * half]);
    let ts: Vec<f64> = if n == 1 { vec![0.5] } else { (0..n).map(|i| 0.5 - share / 2.0 + share * i as f64 / (n - 1) as f64).collect() };
    Some(ts.into_iter().map(|t| Station { at: a + (b - a) * t, normal: axis, x, y, t: Some(t) }).collect())
}

fn unit2(v: DVec2) -> Option<DVec2> {
    let l = v.length();
    if l > 1e-9 {
        Some(v / l)
    } else {
        None
    }
}

fn perp(v: DVec2) -> DVec2 {
    DVec2::new(-v.y, v.x)
}

/// The sign that makes a circle trace's distance, |x − q| − r, grow along its normal m at the
/// corner.
fn circle_sign(q: DVec2, m: DVec2) -> f64 {
    let s = -q.dot(m);
    if s < 0.0 {
        -1.0
    } else {
        1.0
    }
}

/// A trace's signed distance at x.
fn phi(t: &Trace2, x: DVec2) -> f64 {
    match t.curve {
        TraceCurve::Line => x.dot(t.m),
        TraceCurve::Circle { q, r } => circle_sign(q, t.m) * ((x - q).length() - r),
    }
}

/// A trace's gradient at x.
fn grad(t: &Trace2, x: DVec2) -> DVec2 {
    match t.curve {
        TraceCurve::Line => t.m,
        TraceCurve::Circle { q, .. } => unit2(x - q).unwrap_or(t.m) * circle_sign(q, t.m),
    }
}

/// Which way along a curve from the corner the scan's points lie: the points within `tol` of
/// it, clear of the corner, counted either side.
fn sides(window: &[DVec2], dist: impl Fn(DVec2) -> f64, tangent: DVec2, tol: f64) -> (usize, usize) {
    let mut plus = 0;
    let mut minus = 0;
    for p in window {
        if p.length() < 3.0 * tol || dist(*p) > tol {
            continue;
        }
        if p.dot(tangent) > 0.0 {
            plus += 1;
        } else {
            minus += 1;
        }
    }
    (plus, minus)
}

/// A trace a face may have, before the scan says which way along it the face runs.
#[derive(Debug, Clone, Copy)]
struct Candidate {
    curve: TraceCurve,
    m: DVec2,
    tangent: DVec2,
}

impl Candidate {
    fn dist(&self, p: DVec2) -> f64 {
        match self.curve {
            TraceCurve::Line => p.dot(self.m).abs(),
            TraceCurve::Circle { q, r } => ((p - q).length() - r).abs(),
        }
    }
}

/// The candidate and the way along it the scan bears out: the most points, at least three,
/// and twice any other's.
fn best_of(window: &[DVec2], cands: &[Candidate], tol: f64) -> Option<Trace2> {
    let mut scored: Vec<(Trace2, usize)> = Vec::new();
    for c in cands {
        let (plus, minus) = sides(window, |p| c.dist(p), c.tangent, tol);
        scored.push((Trace2 { curve: c.curve, m: c.m, d: c.tangent }, plus));
        scored.push((Trace2 { curve: c.curve, m: c.m, d: -c.tangent }, minus));
    }
    scored.sort_by_key(|s| std::cmp::Reverse(s.1));
    let first = scored.first()?;
    if first.1 < 3 || (scored.len() > 1 && first.1 < 2 * scored[1].1) {
        return None;
    }
    Some(first.0)
}

fn line_cand(m: DVec2) -> Candidate {
    Candidate { curve: TraceCurve::Line, m, tangent: perp(m) }
}

fn circle_cand(q: DVec2, r: f64) -> Candidate {
    let u = -q / r;
    Candidate { curve: TraceCurve::Circle { q, r }, m: u, tangent: perp(u) }
}

/// A cone's trace through the corner, read off the scan: the median direction of the points
/// out along it, off the flat face's line.
fn cone_trace(window: &[DVec2], flat: &Trace2, reach: f64, tol: f64) -> Option<Trace2> {
    let far: Vec<DVec2> = window.iter().copied().filter(|p| p.dot(flat.m).abs() > tol && p.length() >= (3.0 * tol).max(0.35 * reach)).collect();
    if far.len() < 4 {
        return None;
    }
    let mean = unit2(far.iter().fold(DVec2::ZERO, |a, p| a + *p / p.length()))?;
    let mut angles: Vec<f64> = far.iter().map(|p| p.dot(perp(mean)).atan2(p.dot(mean))).collect();
    angles.sort_by(f64::total_cmp);
    let a = angles[angles.len() >> 1];
    let d = DVec2::new(mean.x * a.cos() - mean.y * a.sin(), mean.x * a.sin() + mean.y * a.cos());
    Some(Trace2 { curve: TraceCurve::Line, m: perp(d), d })
}

/// A round's profile between two traces at one radius: its centre, the tangent points on the
/// two traces, and the ways the faces run on from them.
#[derive(Debug, Clone, Copy)]
struct Profile {
    c: DVec2,
    t: [DVec2; 2],
    e: [DVec2; 2],
}

/// The round fitted between two traces, line or circle, as `estimate_fillet_radius` fits it
/// between two lines: one unknown, the radius; for each the centre where both traces' signed
/// distances are −r (a Newton step or two from the two lines' answer), the profile the faces
/// from their tangent points with the arc between, every point near the corner read against
/// it, capped; the radius the points lie nearest, and one only where it does markedly better
/// than a sharp corner.
pub fn estimate_round_on_traces(points: &[DVec2], t1: &Trace2, t2: &Trace2, reach: f64) -> Option<FilletReading> {
    let pts: Vec<DVec2> = points.iter().copied().filter(|p| p.length() <= reach).collect();
    if pts.len() < 8 {
        return None;
    }
    let mut off: Vec<f64> = Vec::new();
    for p in &pts {
        let d = phi(t1, *p).abs().min(phi(t2, *p).abs());
        if d <= 0.3 {
            off.push(d);
        }
    }
    off.sort_by(f64::total_cmp);
    let tau = if off.len() < 4 { 0.05 } else { (3.0 * off[off.len() >> 1]).max(0.02) };
    let k = 1.0 + t1.m.dot(t2.m);
    if !(k > 1e-6) {
        return None;
    }
    let b = -(t1.m + t2.m) / k;
    let traces = [t1, t2];
    let profile = |r: f64| -> Option<Profile> {
        let mut c = b * r;
        for _ in 0..20 {
            let f1 = phi(t1, c) + r;
            let f2 = phi(t2, c) + r;
            if f1.abs() + f2.abs() < 1e-10 * (1.0 + r) {
                break;
            }
            let g1 = grad(t1, c);
            let g2 = grad(t2, c);
            let det = g1.x * g2.y - g1.y * g2.x;
            if det.abs() < 1e-9 {
                return None;
            }
            c = DVec2::new(c.x - (f1 * g2.y - f2 * g1.y) / det, c.y - (g1.x * f2 - g2.x * f1) / det);
        }
        if (phi(t1, c) + r).abs() + (phi(t2, c) + r).abs() > 1e-6 * (1.0 + r) {
            return None;
        }
        let t = traces.map(|tr| c + grad(tr, c) * r);
        let e = traces.map(|tr| {
            let g = perp(grad(tr, c));
            if g.dot(tr.d) >= 0.0 {
                g
            } else {
                -g
            }
        });
        Some(Profile { c, t, e })
    };
    let face_dist = |p: DVec2, pr: &Profile, i: usize| -> f64 {
        let tr = traces[i];
        let tp = pr.t[i];
        let e = pr.e[i];
        let s = (p - tp).dot(e);
        if s < 0.0 {
            return (p - tp).length();
        }
        match tr.curve {
            TraceCurve::Line => (p - (tp + e * s)).length(),
            TraceCurve::Circle { .. } => phi(tr, p).abs(),
        }
    };
    let distance = |p: DVec2, r: f64, pr: &Profile| -> (f64, bool) {
        let w = p - pr.c;
        if w.dot(pr.e[0]) <= 0.0 && w.dot(pr.e[1]) <= 0.0 {
            return ((w.length() - r).abs(), true);
        }
        (face_dist(p, pr, 0).min(face_dist(p, pr, 1)), false)
    };
    let cap = 5.0 * tau;
    let cost = |r: f64| -> f64 {
        let Some(pr) = profile(r) else { return f64::INFINITY };
        let mut sum = 0.0;
        for p in &pts {
            let d = distance(*p, r, &pr).0.min(cap);
            sum += d * d;
        }
        sum
    };
    let r_min = tau.max(0.02);
    if !(reach > r_min) {
        return None;
    }
    let (best, best_cost) = log_search(r_min, reach, &cost);
    if !best_cost.is_finite() {
        return None;
    }
    let r = golden(best, r_min, reach, &cost);
    let pr = profile(r)?;
    if !(r > 2.0 * tau) || cost(r) > 0.5 * cost(r_min) {
        return None;
    }
    let near = 2.0 * r + 3.0 * tau;
    let mut on_arc = 0usize;
    let mut arc_total = 0usize;
    let mut sum = 0.0;
    let h = (r * 1e-3).max(1e-4);
    let pr_up = profile(r + h);
    let pr_down = profile(r - h);
    let mut jj = 0.0;
    for p in &pts {
        if p.length() > near {
            continue;
        }
        let (d, arc) = distance(*p, r, &pr);
        if !arc {
            continue;
        }
        arc_total += 1;
        if d > 3.0 * tau {
            continue;
        }
        on_arc += 1;
        sum += d * d;
        if let (Some(up), Some(down)) = (&pr_up, &pr_down) {
            let j = ((*p - up.c).length() - (r + h) - ((*p - down.c).length() - (r - h))) / (2.0 * h);
            jj += j * j;
        }
    }
    if on_arc < 3 || (on_arc as f64) < 0.6 * arc_total as f64 {
        return None;
    }
    let rms = (sum / on_arc as f64).sqrt();
    let scatter = (rms * (on_arc as f64 / (on_arc as f64 - 1.0).max(1.0)).sqrt()).max(tau / 3.0);
    Some(FilletReading { radius: r, points: on_arc, rms, sigma: if jj > 1e-12 { scatter / jj.sqrt() } else { r } })
}

/// How near a face's trace a scan point counts as on it, for telling which way the face runs:
/// wide, since the model's face may stand a little off the scan's — a face fitted leaning, a
/// part not quite flat.
fn trace_tol(reach: f64) -> f64 {
    (0.08 * reach).max(0.3)
}

/// How far off the model's line a face's scan points are taken for where the scan has the
/// face, mm.
const SCENE_BAND: f64 = 0.4;
/// How far the scan's corner may stand off the model's and still be taken, mm.
const SCENE_SHIFT: f64 = 0.5;

/// Two traces moved onto the scan near the corner: each face's points clear of the round
/// (beyond `from` along it, within a band of the model's line), the median of their offsets
/// from the line — the face where the scan has it — and the corner where the two moved lines
/// meet. The model's faces were fitted over the whole part; a face that leans a little, a
/// part a little bent, puts the scan's corner a tenth or two off the model's — which a round
/// of a millimetre reads as another round. The lines keep the model's directions: a degree's
/// lean is hundredths over the few millimetres read, and a median is not pulled by the points
/// of a chamfer or a countersink the band takes in. `None` where neither face says, or the
/// scan's corner stands too far off.
fn scene_corner(window: &[DVec2], t: &[Trace2; 2], from: f64, reach: f64) -> Option<DVec2> {
    // A circle's trace is moved as a line is, by its offset along the normal at the corner:
    // the shift is a tenth or two, the circle many times that.
    let c = t.map(|tr| {
        let mut off: Vec<f64> = window.iter().filter(|p| p.dot(tr.d) >= from && p.length() <= reach).map(|p| phi(&tr, *p)).filter(|d| d.abs() <= SCENE_BAND).collect();
        if off.len() < 6 {
            return None;
        }
        off.sort_by(f64::total_cmp);
        Some(off[off.len() >> 1])
    });
    if c[0].is_none() && c[1].is_none() {
        return None;
    }
    let (a, b) = (c[0].unwrap_or(0.0), c[1].unwrap_or(0.0));
    let (n, m) = (t[0].m, t[1].m);
    let det = n.x * m.y - n.y * m.x;
    if det.abs() < 1e-6 {
        return None;
    }
    let corner = DVec2::new((a * m.y - b * n.y) / det, (n.x * b - m.x * a) / det);
    if corner.length() <= SCENE_SHIFT {
        Some(corner)
    } else {
        None
    }
}

/// The reaches a station is read at, as shares of the one asked, where nothing reads: a
/// neighbouring feature the reach takes in — a countersink a millimetre past a round, a face
/// ending short of the reach that the face window did not find — read past. None under
/// `RETRY_FLOOR` mm.
const RETRIES: [f64; 4] = [1.0, 0.6, 0.4, 0.25];
const RETRY_FLOOR: f64 = 1.0;

fn retried(reach: f64) -> Vec<f64> {
    RETRIES.iter().enumerate().map(|(i, k)| (i, k * reach)).filter(|(i, r)| *i == 0 || *r >= RETRY_FLOOR).map(|(_, r)| r).collect()
}

/// Why a station read nothing: its first read's why (at the reach asked), or a later one's
/// where that said nothing.
fn first_why(miss: &mut Option<&mut Miss>, own: &Miss) {
    if let Some(m) = miss.as_deref_mut() {
        let said = m.why.as_deref().is_some_and(|w| !w.is_empty());
        if !said {
            if let Some(w) = own.why.as_deref().filter(|w| !w.is_empty()) {
                m.why = Some(w.to_string());
            }
        }
    }
}

/// The round (or chamfer) across a curved edge at one station, read off the scan's cut by the
/// station's plane. `None` where the cut shows none, or does not bear out which way the faces
/// run; `miss` then says why. Where the reach takes in a neighbouring feature — a countersink
/// a millimetre past a round — and nothing reads, the corner is read again over shorter
/// reaches (`RETRIES`).
pub fn read_curved_cut(edge: &CurvedEdge, st: &Station, cut: &SectionCut, reach: f64, shape: ReadShape, mut miss: Option<&mut Miss>) -> Option<CurvedRead> {
    let to2 = |p: DVec3| {
        let d = p - st.at;
        DVec2::new(d.dot(st.x), d.dot(st.y))
    };
    let chains = cut_chains(cut, to2);
    for r in retried(reach) {
        let mut own = Miss::default();
        if let Some(read) = read_curved_at(edge, st, &chains, &to2, r, shape, &mut own, false) {
            return Some(read);
        }
        first_why(&mut miss, &own);
    }
    None
}

fn none<T>(miss: &mut Miss, why: impl Into<String>) -> Option<T> {
    miss.why = Some(why.into());
    None
}

#[allow(clippy::too_many_arguments)]
fn read_curved_at(edge: &CurvedEdge, st: &Station, chains: &[Vec<DVec2>], to2: &dyn Fn(DVec3) -> DVec2, reach: f64, shape: ReadShape, miss: &mut Miss, capped: bool) -> Option<CurvedRead> {
    let dir2 = |v: DVec3| unit2(DVec2::new(v.dot(st.x), v.dot(st.y)));
    let window = corner_window(chains, reach);
    if window.len() < 8 {
        return none(miss, few_why(&window, reach));
    }
    let tol = trace_tol(reach);
    let fp = &edge.fingerprint;
    let (Some(s0), Some(s1)) = (surface_of(edge, 0), surface_of(edge, 1)) else {
        return none(miss, "an edge with one face — nothing to round between");
    };
    let surf = [s0, s1];
    let Some(fi) = surf.iter().position(|f| f.kind == FaceKind::Plane) else { return none(miss, "no flat face") };
    let other = &surf[1 - fi];
    let Some(mp) = dir2(surf[fi].dir) else { return none(miss, "the flat face lies in the cut") };
    let Some(flat) = best_of(&window, &[line_cand(mp)], tol) else { return none(miss, "the faces not found in the cut") };
    let mut trace: Option<Trace2> = None;
    if fp.kind == EdgeKind::Circle && other.kind == FaceKind::Cylinder {
        if let Some(along) = dir2(fp.dir) {
            trace = best_of(&window, &[line_cand(perp(along))], tol);
        }
    } else if fp.kind == EdgeKind::Circle && other.kind == FaceKind::Cone {
        trace = cone_trace(&window, &flat, reach, tol);
    } else if fp.kind == EdgeKind::Circle && other.kind == FaceKind::Sphere {
        if let (Some(r), Some(rs)) = (fp.radius, other.radius) {
            let h = (rs * rs - r * r).max(0.0).sqrt();
            let qs: Vec<DVec2> = if h > 1e-6 { vec![DVec2::new(-r, h), DVec2::new(-r, -h)] } else { vec![DVec2::new(-r, 0.0)] };
            let cands: Vec<Candidate> = qs.into_iter().map(|q| circle_cand(q, rs)).collect();
            trace = best_of(&window, &cands, tol);
        }
    } else if fp.kind == EdgeKind::Line && other.kind == FaceKind::Cylinder {
        if let (Some(center), Some(r)) = (other.center, other.radius) {
            let q = to2(center);
            trace = if q.length() > 1e-6 { best_of(&window, &[circle_cand(q, r)], tol) } else { None };
        }
    }
    let Some(trace) = trace else { return none(miss, "the faces not found in the cut") };
    // Convex where the other face runs off against the flat's outward normal; its own normal
    // then points away from the flat, and towards it on a concave corner.
    let s = flat.m.dot(trace.d);
    if s.abs() < sin_tangent() {
        return none(miss, "the faces meet near tangent");
    }
    let across = trace.m.dot(flat.d);
    if across.abs() < 1e-3 {
        return none(miss, "the faces meet near tangent");
    }
    let oriented = if (across > 0.0) == (s > 0.0) { trace } else { trace.turned() };
    let concave = s > 0.0;
    let turn = |t: Trace2| if concave { t.turned() } else { t };
    let mut t: [Trace2; 2] = if fi == 0 { [turn(flat), turn(oriented)] } else { [turn(oriented), turn(flat)] };
    if !t[0].is_line() || !t[1].is_line() {
        if shape == ReadShape::Chamfer {
            // A chamfer is read against the curved face's tangent at the corner, over a reach
            // short enough for the face to stay by it.
            let near = tangent_reach(&t);
            if !capped && near < reach {
                if let Some(read) = read_curved_at(edge, st, chains, to2, near, shape, miss, true) {
                    return Some(read);
                }
                let why = tangent_why(miss.why.as_deref(), near, &t);
                return none(miss, why);
            }
            t = [as_tangent(t[0]), as_tangent(t[1])];
        } else {
            return match estimate_round_on_traces(&window, &t[0], &t[1], reach) {
                Some(reading) => Some(CurvedRead::of(reading)),
                None => none(miss, "sharp"),
            };
        }
    }
    // Read against the model's corner, then against the scan's own corner found from the face
    // points clear of what was read — both on the face window, which ends where a face does
    // within the reach.
    let m = [t[0].m, t[1].m];
    let fw = face_window(chains, reach, m, [t[0].d, t[1].d]);
    let held = fw.reach;
    let mut own = Miss::default();
    let estimate = |w: &[DVec2], mm: Option<&mut Miss>| match shape {
        ReadShape::Round => estimate_fillet_radius(w, m[0], m[1], held, mm),
        ReadShape::Chamfer => estimate_chamfer(w, m[0], m[1], held, mm),
    };
    let mut reading = estimate(&fw.points, Some(&mut own));
    // How far along a face the profile leaves it, for a radius or setback of one.
    let k = 1.0 + m[0].dot(m[1]);
    let tangent = if shape == ReadShape::Round && k > 1e-6 { (-(m[0] + m[1]) / k + m[0]).length() } else { 1.0 };
    // Where a face ends within the reach, neither runs far enough clear of the round to say
    // where the scan has it: the model's corner, for either shape alike.
    let from = (0.4 * held).max(reading.map_or(0.0, |r| 1.25 * r.radius * tangent));
    let scene = if from < 0.9 * held && fw.end.is_none() { scene_corner(&fw.points, &t, from, held) } else { None };
    let mut w: Vec<DVec2> = fw.points.clone();
    let mut corner = DVec2::ZERO;
    if let Some(scene) = scene.filter(|s| s.length() > 1e-3) {
        let moved: Vec<DVec2> = fw.points.iter().map(|p| *p - scene).collect();
        if let Some(again) = estimate(&moved, None) {
            reading = Some(again);
            w = moved;
            corner = scene;
        }
    }
    let Some(reading) = reading else {
        let why = match &fw.end {
            Some(end) => face_end_why(fw.reach, end),
            None => own.why.unwrap_or_else(|| "sharp".to_string()),
        };
        return none(miss, why);
    };
    if shape == ReadShape::Round {
        return Some(CurvedRead::of(reading));
    }
    let setbacks = estimate_chamfer_setbacks(&w, m[0], m[1], held);
    // A hole's mouth — a cylinder whose outward normal points at its axis — cut by a cone: the
    // countersink, as a drawing calls it out.
    let Some(sb) = setbacks.filter(|_| fp.kind == EdgeKind::Circle && other.kind == FaceKind::Cylinder && oriented.m.x < 0.0) else {
        return Some(CurvedRead { reading, setbacks, countersink: None });
    };
    let s_flat = sb.s[fi];
    let s_wall = sb.s[1 - fi];
    let half = sb.dir.y.abs().min(1.0).acos();
    Some(CurvedRead {
        reading,
        setbacks,
        countersink: Some(Countersink {
            angle_deg: 2.0 * half * 180.0 / PI,
            angle_sigma: 2.0 * sb.dir_sigma * 180.0 / PI,
            diameter: 2.0 * (fp.radius.unwrap_or(f64::NAN) + corner.x + s_flat * flat.d.x),
            diameter_sigma: 2.0 * sb.sigma[fi],
            depth: s_wall * oriented.d.y.abs(),
            depth_sigma: sb.sigma[1 - fi],
        }),
    })
}

// An edge of any shape — the curve where a blade's side meets a dish, a spline's edge, a
// straight edge between a flat and a cone — read off the body it was picked on rather than
// off its faces' surfaces: the cuts are placed by length along the edge's polyline, each
// square to the edge there, so the faces' normals at the station lie in the cut; each face's
// normal there comes from the body's mesh, whose triangles carry the face they lie on; and
// each face's trace is the body's mesh cut by the same plane — a line through the corner
// square to the normal, or a circle tangent to it where the face curves within the reach (a
// dish's cone cut across a blade's foot, a torus). The round is then fitted between the
// traces as on any other edge.

/// A face of an edge as the body's mesh gives it: its triangles, three corners each, and the
/// surface's outward normal at every corner.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FaceTriangles {
    pub positions: Vec<DVec3>,
    pub normals: Vec<DVec3>,
}

/// An edge of any shape, off the body: the polyline it is drawn by and its two faces'
/// triangles.
#[derive(Debug, Clone, PartialEq)]
pub struct FreeEdge {
    pub polyline: Vec<DVec3>,
    pub faces: [FaceTriangles; 2],
}

/// A body's mesh as the reading takes it: the kernel's, which says which face each triangle
/// lies on (`face_hash`, one a triangle).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FacedMesh {
    pub positions: Vec<DVec3>,
    pub normals: Vec<DVec3>,
    pub triangles: Vec<[u32; 3]>,
    pub face_hash: Option<Vec<u32>>,
}

/// The triangles of the face `hash` off a body's mesh, `None` where the mesh does not say
/// which face a triangle lies on or has none of it.
pub fn face_triangles(mesh: &FacedMesh, hash: u32) -> Option<FaceTriangles> {
    let fh = mesh.face_hash.as_ref()?;
    let count = fh.iter().filter(|h| **h == hash).count();
    if count == 0 {
        return None;
    }
    let mut out = FaceTriangles { positions: Vec::with_capacity(count * 3), normals: Vec::with_capacity(count * 3) };
    for (t, h) in fh.iter().enumerate() {
        if *h != hash {
            continue;
        }
        let tri = mesh.triangles.get(t).copied().unwrap_or([u32::MAX; 3]);
        for v in tri {
            out.positions.push(mesh.positions.get(v as usize).copied().unwrap_or(DVec3::NAN));
            out.normals.push(mesh.normals.get(v as usize).copied().unwrap_or(DVec3::NAN));
        }
    }
    Some(out)
}

/// An edge of any shape off a body's mesh: its polyline and its two faces' triangles; `None`
/// for an edge with one face (a seam) or a face the mesh does not hold.
pub fn free_edge_of(polyline: &[DVec3], faces: &[u32], mesh: &FacedMesh) -> Option<FreeEdge> {
    if polyline.len() < 2 || faces.len() < 2 || faces[0] == faces[1] {
        return None;
    }
    let a = face_triangles(mesh, faces[0])?;
    let b = face_triangles(mesh, faces[1])?;
    Some(FreeEdge { polyline: polyline.to_vec(), faces: [a, b] })
}

/// The point of a triangle nearest p, as weights on its corners (Ericson, Real-Time Collision
/// Detection, 5.1.5).
fn nearest_on_triangle(p: DVec3, a: DVec3, b: DVec3, c: DVec3) -> [f64; 3] {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return [1.0, 0.0, 0.0];
    }
    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return [0.0, 1.0, 0.0];
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return [1.0 - v, v, 0.0];
    }
    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return [0.0, 0.0, 1.0];
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return [1.0 - w, 0.0, w];
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
        let w = (d4 - d3) / (d4 - d3 + (d5 - d6));
        return [0.0, 1.0 - w, w];
    }
    let den = 1.0 / (va + vb + vc);
    let v = vb * den;
    let w = vc * den;
    [1.0 - v - w, v, w]
}

/// The surface's outward normal on a face at the point of it nearest `at`: the normals at the
/// corners of the triangle it lies on, weighted as the point lies between them — on the edge,
/// the normals the mesher put on the edge itself, however long the face's slivers run off it.
pub fn face_normal_near(face: &FaceTriangles, at: DVec3) -> Option<DVec3> {
    let p = &face.positions;
    let mut best: Option<(usize, [f64; 3])> = None;
    let mut best_d = f64::INFINITY;
    let mut t = 0;
    while t + 2 < p.len() {
        let c = [p[t], p[t + 1], p[t + 2]];
        let w = nearest_on_triangle(at, c[0], c[1], c[2]);
        let q = c[0] * w[0] + c[1] * w[1] + c[2] * w[2];
        let d = (q - at).length();
        if d < best_d {
            best_d = d;
            best = Some((t, w));
        }
        t += 3;
    }
    let (t, w) = best?;
    let n = &face.normals;
    let (n0, n1, n2) = (n.get(t)?, n.get(t + 1)?, n.get(t + 2)?);
    unit3(*n0 * w[0] + *n1 * w[1] + *n2 * w[2])
}

/// A point of a face's cut, in the cut: where it is, and the surface's outward normal there
/// flattened into the cut, unit length.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CutPoint {
    pub p: DVec2,
    pub n: DVec2,
}

/// Where a face's triangles cross the plane through `at` square to `normal`, within `reach` of
/// `at`, mapped into the plane by the axes x and y: points along each triangle's cut a
/// fifteenth of the reach apart, with the normals at its corners carried along — a flat or a
/// cone is meshed in long slivers, whose cut runs across the whole reach with no crossing
/// inside it.
pub fn face_cut(face: &FaceTriangles, at: DVec3, normal: DVec3, x: DVec3, y: DVec3, reach: f64) -> Vec<CutPoint> {
    let p = &face.positions;
    let nn = &face.normals;
    let mut out: Vec<CutPoint> = Vec::new();
    let step = reach / 15.0;
    // A reach of none would step along each cut for ever.
    if !(step > 0.0) {
        return out;
    }
    let lerp = |a: DVec3, b: DVec3, f: f64| a + (b - a) * f;
    let mut t = 0;
    while t + 2 < p.len() && t + 2 < nn.len() {
        let c = [p[t], p[t + 1], p[t + 2]];
        let s = c.map(|q| (q - at).dot(normal));
        let mut ends: Vec<(DVec2, DVec3)> = Vec::new();
        for k in 0..3 {
            let (sa, sb) = (s[k], s[(k + 1) % 3]);
            if (sa < 0.0) == (sb < 0.0) {
                continue;
            }
            let f = sa / (sa - sb);
            let q = lerp(c[k], c[(k + 1) % 3], f) - at;
            ends.push((DVec2::new(q.dot(x), q.dot(y)), lerp(nn[t + k], nn[t + (k + 1) % 3], f)));
        }
        t += 3;
        if ends.len() < 2 {
            continue;
        }
        let (a, b) = (ends[0].0, ends[1].0);
        let l = (b - a).length();
        // Clear of the reach: the segment's nearest point to the corner.
        let f = if l > 0.0 { (-(a.dot(b - a)) / (l * l)).clamp(0.0, 1.0) } else { 0.0 };
        if (a + (b - a) * f).length() > reach {
            continue;
        }
        let n = (l / step).ceil().max(1.0) as usize;
        for i in 0..=n {
            let fi = i as f64 / n as f64;
            let q = DVec2::new(a.x + (b.x - a.x) * i as f64 / n as f64, a.y + (b.y - a.y) * i as f64 / n as f64);
            if q.length() > reach {
                continue;
            }
            let m = lerp(ends[0].1, ends[1].1, fi);
            if let Some(m2) = unit2(DVec2::new(m.dot(x), m.dot(y))) {
                out.push(CutPoint { p: q, n: m2 });
            }
        }
    }
    out
}

/// The point at length s along a polyline, and the unit direction there taken over `h` either
/// side — a chord across a few of the polyline's segments, not one segment's own.
fn along_polyline(poly: &[DVec3], cum: &[f64], s: f64, h: f64) -> (DVec3, Option<DVec3>) {
    let total = cum.last().copied().unwrap_or(0.0);
    let point = |u: f64| -> DVec3 {
        let x = u.min(total).max(0.0);
        let mut i = 1;
        while i < cum.len() - 1 && cum[i] < x {
            i += 1;
        }
        let seg = cum[i] - cum[i - 1];
        let f = if seg > 0.0 { (x - cum[i - 1]) / seg } else { 0.0 };
        poly[i - 1] + (poly[i] - poly[i - 1]) * f
    };
    (point(s), unit3(point(s + h) - point(s - h)))
}

/// The places along an edge of any shape a cut is taken: by length along its polyline over the
/// middle `share`, each plane square to the edge there, X along the first face's outward
/// normal flattened into it. `None` where the polyline is too short or a face's normal is not
/// found.
pub fn free_stations(edge: &FreeEdge, n: usize, share: f64) -> Option<Vec<Station>> {
    let poly = &edge.polyline;
    let mut cum = vec![0.0];
    for i in 1..poly.len() {
        cum.push(cum[i - 1] + (poly[i] - poly[i - 1]).length());
    }
    let total = cum.last().copied().unwrap_or(0.0);
    if !(total > 1e-6) || poly.len() < 2 {
        return None;
    }
    let h = (0.05 * total).min(0.5);
    let ts: Vec<f64> = if n == 1 { vec![0.5] } else { (0..n).map(|i| 0.5 - share / 2.0 + share * i as f64 / (n - 1) as f64).collect() };
    let mut out = Vec::with_capacity(ts.len());
    for t in ts {
        let (at, dir) = along_polyline(poly, &cum, t * total, h);
        let dir = dir?;
        let n0 = face_normal_near(&edge.faces[0], at);
        let x = n0.and_then(|n0| unit3(n0 - dir * n0.dot(dir)))?;
        out.push(Station { at, normal: dir, x, y: dir.cross(x), t: Some(t) });
    }
    Some(out)
}

/// Why an edge of any shape cannot be read, or `None` when it can: its faces meet tangent at
/// its middle — a round's seam, no corner.
pub fn free_unreadable(edge: &FreeEdge) -> Option<String> {
    let Some(st) = free_stations(edge, 1, 0.0).and_then(|s| s.into_iter().next()) else {
        return Some("the edge’s geometry places no cut across it".to_string());
    };
    let (Some(m0), Some(m1)) = (face_normal_near(&edge.faces[0], st.at), face_normal_near(&edge.faces[1], st.at)) else {
        return Some("the edge’s faces are not found on the body’s mesh".to_string());
    };
    if m0.cross(m1).length() < sin_tangent() && m0.dot(m1) > 0.0 {
        return Some("a smooth edge — its faces meet tangent, there is no corner to read".to_string());
    }
    None
}

/// How far a face's trace may bow off a line over the reach, mm, and still be read as one.
const TRACE_SAG: f64 = 0.005;

/// How far a curved face may bow off its tangent at the corner, mm, over the reach a chamfer
/// is read at against that tangent: a twentieth — the scan's own scatter, near enough, where a
/// chamfer of a millimetre reads a few hundredths off.
const CHAMFER_SAG: f64 = 0.05;

/// The smallest radius among the curved traces, ∞ with none.
fn least_radius(t: &[Trace2]) -> f64 {
    t.iter().fold(f64::INFINITY, |r, tr| match tr.curve {
        TraceCurve::Circle { r: tr_r, .. } => r.min(tr_r),
        TraceCurve::Line => r,
    })
}

/// The reach over which every curved trace stays within `CHAMFER_SAG` of its tangent at the
/// corner: a circle of radius R bows reach² / 2R off it.
fn tangent_reach(t: &[Trace2]) -> f64 {
    (2.0 * least_radius(t) * CHAMFER_SAG).sqrt()
}

/// A trace as its tangent at the corner: a circle's line through the corner square to its
/// normal there.
fn as_tangent(t: Trace2) -> Trace2 {
    Trace2 { curve: TraceCurve::Line, m: t.m, d: t.d }
}

/// Why a chamfer read against a curved face's tangent read nothing: what the short read said,
/// with the reach it was held to and why.
fn tangent_why(why: Option<&str>, near: f64, t: &[Trace2]) -> String {
    let r = least_radius(t);
    let r2 = |x: f64| js_num(js_round(x * 100.0) / 100.0);
    format!(
        "{} — read against the curved face's tangent within {} mm, where the R{} face stays within {} mm of it",
        why.unwrap_or("sharp"),
        r2(near),
        r2(r),
        js_num(CHAMFER_SAG)
    )
}

/// A face's trace through the corner, off the body's own cut: m the outward normal at the
/// corner in the cut, d the way the face runs from the corner (where its cut points lie), and
/// a circle tangent to the line where the face bows off it by more than `TRACE_SAG` over the
/// reach. The curvature is read off the normals, not the points: a mesh's points lie on
/// chords, off the surface by as much as its chord tolerance, which is the bow being read,
/// while its normals are the surface's own. Along a circle the normal turns from m towards d
/// by κ for every millimetre; κ is the least-squares one.
pub fn trace_from_cut(points: &[CutPoint], m: DVec2, reach: f64) -> Option<Trace2> {
    let clear: Vec<&CutPoint> = points.iter().filter(|c| c.p.length() >= 0.1 * reach).collect();
    if clear.len() < 2 {
        return None;
    }
    // The face runs from the corner the way its nearest points lie: a cut may meet the same
    // face again past a thin wall — the dish on the far side of a blade — but only further
    // off.
    let mut nearest = clear[0];
    for c in &clear[1..] {
        if c.p.length() < nearest.p.length() {
            nearest = c;
        }
    }
    let t = perp(m);
    let d = if nearest.p.dot(t) >= 0.0 { t } else { -t };
    let own: Vec<&&CutPoint> = clear.iter().filter(|c| c.p.dot(d) > 0.0).collect();
    if own.len() < 2 {
        return None;
    }
    let mut num = 0.0;
    let mut den = 0.0;
    for c in own {
        let s = c.p.dot(d);
        num += c.n.dot(d).atan2(c.n.dot(m)) * s;
        den += s * s;
    }
    let kappa = if den > 0.0 { num / den } else { 0.0 };
    if kappa.abs() * reach * reach < 2.0 * TRACE_SAG {
        return Some(Trace2 { curve: TraceCurve::Line, m, d });
    }
    let k = 1.0 / kappa;
    Some(Trace2 { curve: TraceCurve::Circle { q: -m * k, r: k.abs() }, m, d })
}

/// How well a round on an edge of any shape is known at one place, at best, as a share of its
/// radius.
const FREE_SIGMA_SHARE: f64 = 0.02;

/// Each face's normal and cut at a station, kept with the station: a face is read through all
/// its triangles, and a station is read at several reaches and for both shapes. The cut is
/// kept at the largest reach asked and filtered for a shorter one. One of these goes with each
/// station `read_free_cut` is called for (ScanRuler keeps them in a map keyed by the station).
#[derive(Debug, Clone, Default)]
pub struct StationFaces {
    kept: [Option<FaceAt>; 2],
}

#[derive(Debug, Clone)]
struct FaceAt {
    normal: Option<DVec3>,
    cut: Vec<CutPoint>,
    reach: f64,
}

impl StationFaces {
    fn face_at(&mut self, st: &Station, i: usize, face: &FaceTriangles, reach: f64) -> &FaceAt {
        let fresh = match &self.kept[i] {
            Some(known) if known.reach >= reach => None,
            known => {
                let normal = known.as_ref().and_then(|k| k.normal).or_else(|| face_normal_near(face, st.at));
                Some(FaceAt { normal, cut: face_cut(face, st.at, st.normal, st.x, st.y, reach), reach })
            }
        };
        if let Some(f) = fresh {
            self.kept[i] = Some(f);
        }
        // Set just above when it was not.
        self.kept[i].get_or_insert_with(|| FaceAt { normal: None, cut: Vec::new(), reach })
    }
}

/// The round (or, between straight traces, the chamfer) across an edge of any shape at one
/// station, read off the scan's cut by the station's plane. The body gives both traces and
/// which way each face runs; the corner is convex where the second face runs off against the
/// first's outward normal, and a concave one is read as the convex one its normals turned
/// round make, as `read_curved_cut` reads it. `faces` keeps the faces' cuts at this station
/// between reads.
pub fn read_free_cut(edge: &FreeEdge, st: &Station, faces: &mut StationFaces, cut: &SectionCut, reach: f64, shape: ReadShape, mut miss: Option<&mut Miss>) -> Option<CurvedRead> {
    let to2 = |p: DVec3| {
        let d = p - st.at;
        DVec2::new(d.dot(st.x), d.dot(st.y))
    };
    let chains = cut_chains(cut, to2);
    for r in retried(reach) {
        let mut own = Miss::default();
        let read = read_free_at(edge, st, faces, &chains, r, shape, &mut own, false);
        match read {
            None => first_why(&mut miss, &own),
            // The traces are the model's faces, fitted over the whole part: a few hundredths
            // off the scan's, the round reads a few per cent off, which the fit's own sigma
            // does not know.
            Some(mut read) => {
                read.reading.sigma = read.reading.sigma.max(FREE_SIGMA_SHARE * read.reading.radius);
                return Some(read);
            }
        }
    }
    None
}

#[allow(clippy::too_many_arguments)]
fn read_free_at(edge: &FreeEdge, st: &Station, faces: &mut StationFaces, chains: &[Vec<DVec2>], reach: f64, shape: ReadShape, miss: &mut Miss, capped: bool) -> Option<CurvedRead> {
    let window = corner_window(chains, reach);
    if window.len() < 8 {
        return none(miss, few_why(&window, reach));
    }
    let mut traces: Vec<Trace2> = Vec::with_capacity(2);
    for (i, face) in edge.faces.iter().enumerate() {
        let at = faces.face_at(st, i, face, reach);
        let m = at.normal.and_then(|normal| unit2(DVec2::new(normal.dot(st.x), normal.dot(st.y))));
        let tr = m.and_then(|m| {
            let within: Vec<CutPoint> = at.cut.iter().copied().filter(|c| c.p.length() <= reach).collect();
            trace_from_cut(&within, m, reach)
        });
        let Some(tr) = tr else { return none(miss, "the faces not found in the cut") };
        traces.push(tr);
    }
    let s = traces[0].m.dot(traces[1].d);
    if s.abs() < sin_tangent() {
        return none(miss, "the faces meet near tangent");
    }
    let concave = s > 0.0;
    let mut t: [Trace2; 2] = [traces[0], traces[1]].map(|tr| if concave { tr.turned() } else { tr });
    // A chamfer is read against a curved face's tangent at the corner, over a reach short
    // enough for the face to stay by it (`tangent_reach`).
    if shape == ReadShape::Chamfer && (!t[0].is_line() || !t[1].is_line()) {
        let near = tangent_reach(&t);
        if !capped && near < reach {
            if let Some(read) = read_free_at(edge, st, faces, chains, near, shape, miss, true) {
                return Some(read);
            }
            let why = tangent_why(miss.why.as_deref(), near, &t);
            return none(miss, why);
        }
        t = [as_tangent(t[0]), as_tangent(t[1])];
    }
    let lines = t[0].is_line() && t[1].is_line();
    let m = [t[0].m, t[1].m];
    // Between straight traces, on the face window, as `read_curved_cut` reads.
    let fw = if lines { face_window(chains, reach, m, [t[0].d, t[1].d]) } else { FaceWindow { points: window, reach, end: None } };
    let held = fw.reach;
    let mut own = Miss::default();
    let estimate = |w: &[DVec2], mm: Option<&mut Miss>| {
        if !lines {
            estimate_round_on_traces(w, &t[0], &t[1], held)
        } else if shape == ReadShape::Round {
            estimate_fillet_radius(w, m[0], m[1], held, mm)
        } else {
            estimate_chamfer(w, m[0], m[1], held, mm)
        }
    };
    // Read against the model's corner, then against the scan's own corner found from the face
    // points clear of what was read.
    let mut reading = estimate(&fw.points, Some(&mut own));
    let k = 1.0 + m[0].dot(m[1]);
    let tangent = if shape == ReadShape::Round && k > 1e-6 { (-(m[0] + m[1]) / k + m[0]).length() } else { 1.0 };
    // Where a face ends within the reach, neither runs far enough clear of the round to say
    // where the scan has it: the model's corner, for either shape alike.
    let from = (0.4 * held).max(reading.map_or(0.0, |r| 1.25 * r.radius * tangent));
    let scene = if from < 0.9 * held && fw.end.is_none() { scene_corner(&fw.points, &t, from, held) } else { None };
    let mut w: Vec<DVec2> = fw.points.clone();
    if let Some(scene) = scene.filter(|s| s.length() > 1e-3) {
        let moved: Vec<DVec2> = fw.points.iter().map(|p| *p - scene).collect();
        if let Some(again) = estimate(&moved, None) {
            reading = Some(again);
            w = moved;
        }
    }
    let Some(reading) = reading else {
        let why = match &fw.end {
            Some(end) => face_end_why(fw.reach, end),
            None => own.why.unwrap_or_else(|| "sharp".to_string()),
        };
        return none(miss, why);
    };
    if shape == ReadShape::Round {
        return Some(CurvedRead::of(reading));
    }
    let setbacks = estimate_chamfer_setbacks(&w, m[0], m[1], held);
    Some(CurvedRead { reading, setbacks, countersink: None })
}

#[cfg(test)]
mod tests {
    //! Ported from ScanRuler `plugins/reverse/tests/reFilletFree.test.ts`, "a face's trace off
    //! the body's cut" (the rest of that file builds the wheel on the kernel).
    use super::*;

    /// Points along a circle through the corner with outward normal m there and curvature κ,
    /// running off along d, with the circle's normals.
    fn along(kappa: f64, m: DVec2, d: DVec2) -> Vec<CutPoint> {
        (0..30)
            .map(|i| {
                let s = 0.1 * (i as f64 + 1.0);
                let (sn, cs) = ((kappa * s).sin(), (kappa * s).cos());
                let k = if kappa == 0.0 { 0.0 } else { 1.0 / kappa };
                let p = if kappa == 0.0 { d * s } else { DVec2::new(d.x * k * sn - m.x * k * (1.0 - cs), d.y * k * sn - m.y * k * (1.0 - cs)) };
                CutPoint { p, n: DVec2::new(d.x * sn + m.x * cs, d.y * sn + m.y * cs) }
            })
            .collect()
    }

    fn close(a: f64, b: f64, digits: i32) -> bool {
        (a - b).abs() < 10f64.powi(-digits) / 2.0
    }

    #[test]
    fn is_a_line_where_the_face_runs_straight_a_circle_of_the_face_s_curvature_where_it_bows() {
        let line = trace_from_cut(&along(0.0, DVec2::X, DVec2::Y), DVec2::X, 3.0).expect("a trace");
        assert!(line.is_line());
        assert!(close(line.d.x, 0.0, 12));
        assert_eq!(line.d.y, 1.0);
        // Convex (bowing away from the normal) and concave, either way along.
        for (kappa, d) in [(1.0 / 20.0, DVec2::new(0.0, -1.0)), (-1.0 / 15.0, DVec2::Y)] {
            let t = trace_from_cut(&along(kappa, DVec2::X, d), DVec2::X, 3.0).expect("a trace");
            let TraceCurve::Circle { q, r } = t.curve else { panic!("a circle") };
            assert!((r * kappa.abs() - 1.0).abs() < 0.01);
            // The circle passes through the corner, square to m there.
            assert!(close(q.length(), r, 6));
            assert!(close(q.y, 0.0, 9));
            assert_eq!((-q.x).signum(), kappa.signum());
        }
    }

    #[test]
    fn takes_the_way_the_face_runs_from_its_points_nearest_the_corner_not_the_same_face_met_again_past_a_thin_wall() {
        let near = along(0.0, DVec2::Y, DVec2::X);
        let far: Vec<CutPoint> = [2.6, 2.8, 3.0].iter().map(|x| CutPoint { p: DVec2::new(-x, 0.0), n: DVec2::Y }).collect();
        let pts: Vec<CutPoint> = near[..8].iter().copied().chain(far).collect();
        assert_eq!(trace_from_cut(&pts, DVec2::Y, 3.0).expect("a trace").d.x, 1.0);
    }
}
