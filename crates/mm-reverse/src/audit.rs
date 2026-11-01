// SPDX-License-Identifier: AGPL-3.0-only
//! A built body looked over for what a scan's noise leaves in a model. Ported from ScanRuler
//! (`plugins/reverse/core/audit.ts`, `core/stepsBetween.ts`, `core/stepWords.ts`), © Stefan
//! Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! A wall a third of a degree off square, two faces meant to be one lying a twentieth of a
//! millimetre apart, a round that meets its neighbour at a hair's angle instead of tangent, an
//! edge too short to be a feature. None of it fails the kernel's check — the body is a valid
//! closed solid — and all of it shows up downstream. The kernel's own check (`body_check.rs`)
//! says whether the body is sound; this says whether it is clean.
//!
//! Everything here is read off the topology a build already carries (the faces' and edges'
//! fingerprints, the faces either side of each edge, the edges' ends) — no kernel call. Each
//! finding of a face traces back to the sketch piece that made it where it can
//! ([`trace_face`]), which is where the repair goes.

use std::collections::{HashMap, HashSet};

use glam::DVec3;

use crate::model::{ExtrudeExtent, Feature, FeatureKind};
use crate::sketch::Shape;
use crate::topo::{EdgeInfo, EdgeKind, FaceFingerprint, FaceInfo, FaceKind};

// ---- steps between faces (stepsBetween.ts) ----

/// Two faces closer than this are one surface.
const SAME_MM: f64 = 1e-4;
/// How far apart two faces' extents may lie, mm, and still be taken to meet.
const MEET_MM: f64 = 0.1;
/// Steps under this, mm, are hairlines.
pub const HAIR_MM: f64 = 1e-3;

/// What a step is between.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StepForm {
    Flat,
    Round,
    Foot,
    Tangent,
}

impl StepForm {
    pub fn name(self) -> &'static str {
        match self {
            StepForm::Flat => "flat",
            StepForm::Round => "round",
            StepForm::Foot => "foot",
            StepForm::Tangent => "tangent",
        }
    }
}

/// A step between two faces of a body, by index into its face list.
#[derive(Debug, Clone)]
pub struct StepPair {
    pub form: StepForm,
    pub faces: [usize; 2],
    pub off: f64,
    pub at: DVec3,
    pub riser: Option<usize>,
    pub radii: Option<[f64; 2]>,
    pub axis_off: Option<f64>,
    pub crease_deg: Option<f64>,
}

pub struct StepOptions<'a> {
    pub step_mm: f64,
    pub parallel_deg: f64,
    pub pair: Option<&'a dyn Fn(&FaceInfo, &FaceInfo) -> bool>,
}

pub const STEP_DEFAULTS: (f64, f64) = (0.3, 1.0);

impl Default for StepOptions<'_> {
    fn default() -> Self {
        StepOptions { step_mm: STEP_DEFAULTS.0, parallel_deg: STEP_DEFAULTS.1, pair: None }
    }
}

fn angle_deg(a: DVec3, b: DVec3) -> f64 {
    let (la, lb) = (a.length(), b.length());
    if !(la > 0.0 && lb > 0.0) {
        return 0.0;
    }
    (a.dot(b) / (la * lb)).clamp(-1.0, 1.0).acos().to_degrees()
}

fn lean_deg(a: DVec3, b: DVec3) -> f64 {
    let d = angle_deg(a, b);
    if d > 90.0 {
        180.0 - d
    } else {
        d
    }
}

fn facing_deg(a: DVec3, b: DVec3) -> f64 {
    let (la, lb) = (a.length(), b.length());
    if !(la > 0.0 && lb > 0.0) {
        return 180.0;
    }
    (a.dot(b) / (la * lb)).clamp(-1.0, 1.0).acos().to_degrees()
}

fn basis_of(n: DVec3) -> (DVec3, DVec3) {
    let helper = if n.x.abs() < 0.9 { DVec3::X } else { DVec3::Y };
    let u = n.cross(helper).try_normalize().unwrap_or(DVec3::X);
    (u, n.cross(u))
}

fn mid3(a: DVec3, b: DVec3) -> DVec3 {
    (a + b) / 2.0
}

fn face_index(faces: &[FaceInfo]) -> HashMap<u32, usize> {
    faces.iter().enumerate().map(|(i, f)| (f.hash, i)).collect()
}

fn unique(v: &[u32]) -> Vec<u32> {
    let mut out = Vec::new();
    for x in v {
        if !out.contains(x) {
            out.push(*x);
        }
    }
    out
}

/// The points of the edges round each face, by face index.
fn extent_points(faces: &[FaceInfo], edges: &[EdgeInfo]) -> Vec<Vec<DVec3>> {
    let index = face_index(faces);
    let mut out = vec![Vec::new(); faces.len()];
    for e in edges {
        let pts: Vec<DVec3> = if e.polyline.len() >= 2 { e.polyline.clone() } else { vec![e.ends[0], e.ends[1], e.fingerprint.mid] };
        for h in unique(&e.faces) {
            if let Some(i) = index.get(&h) {
                out[*i].extend_from_slice(&pts);
            }
        }
    }
    out
}

fn neighbours(faces: &[FaceInfo], edges: &[EdgeInfo]) -> Vec<HashSet<usize>> {
    let index = face_index(faces);
    let mut out = vec![HashSet::new(); faces.len()];
    for e in edges {
        let fs: Vec<usize> = unique(&e.faces).iter().filter_map(|h| index.get(h).copied()).collect();
        for a in &fs {
            for b in &fs {
                if a != b {
                    out[*a].insert(*b);
                }
            }
        }
    }
    out
}

#[derive(Clone, Copy)]
struct Box2 {
    lo: [f64; 2],
    hi: [f64; 2],
}

fn box2(points: &[DVec3], u: DVec3, v: DVec3) -> Option<Box2> {
    if points.is_empty() {
        return None;
    }
    let mut b = Box2 { lo: [f64::INFINITY; 2], hi: [f64::NEG_INFINITY; 2] };
    for p in points {
        let (a, c) = (p.dot(u), p.dot(v));
        b.lo[0] = b.lo[0].min(a);
        b.hi[0] = b.hi[0].max(a);
        b.lo[1] = b.lo[1].min(c);
        b.hi[1] = b.hi[1].max(c);
    }
    Some(b)
}

fn boxes_meet(a: Box2, b: Box2, m: f64) -> bool {
    a.lo[0] <= b.hi[0] + m && b.lo[0] <= a.hi[0] + m && a.lo[1] <= b.hi[1] + m && b.lo[1] <= a.hi[1] + m
}

fn span(points: &[DVec3], origin: DVec3, dir: DVec3) -> Option<[f64; 2]> {
    if points.is_empty() {
        return None;
    }
    let mut s = [f64::INFINITY, f64::NEG_INFINITY];
    for p in points {
        let t = (*p - origin).dot(dir);
        s[0] = s[0].min(t);
        s[1] = s[1].max(t);
    }
    Some(s)
}

/// A round face's radius along its axis and how fast it grows.
fn radius_along(fp: &FaceFingerprint, points: &[DVec3]) -> Option<(f64, f64)> {
    if fp.kind == FaceKind::Cylinder {
        return fp.radius.map(|r| (r, 0.0));
    }
    let c = fp.center?;
    if fp.kind != FaceKind::Cone || points.len() < 2 {
        return None;
    }
    let (mut n, mut st, mut sr, mut stt, mut str_) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for p in points {
        let d = *p - c;
        let t = d.dot(fp.dir);
        let r = (d - fp.dir * t).length();
        n += 1.0;
        st += t;
        sr += r;
        stt += t * t;
        str_ += t * r;
    }
    let den = n * stt - st * st;
    if den.abs() < 1e-12 {
        return None;
    }
    let k = (n * str_ - st * sr) / den;
    Some(((sr - k * st) / n, k))
}

fn surface_key(fp: &FaceFingerprint) -> String {
    let r = |x: f64, d: i32| format!("{:.*}", d as usize, (x * 10f64.powi(d)).round() / 10f64.powi(d));
    let v = |p: DVec3| format!("{},{},{}", r(p.x, 3), r(p.y, 3), r(p.z, 3));
    match (fp.kind, fp.center) {
        (FaceKind::Plane, _) => format!("p|{}|{}", v(fp.dir), r(fp.centroid.dot(fp.dir), 4)),
        (FaceKind::Cylinder | FaceKind::Cone, Some(c)) => {
            let foot = c - fp.dir * c.dot(fp.dir);
            format!("{}|{}|{}|{}", fp.kind.name(), v(fp.dir), v(foot), fp.radius.map_or(String::new(), |x| r(x, 4)))
        }
        _ => format!("{}|{}", fp.kind.name(), v(fp.centroid)),
    }
}

/// The steps among a body's faces (ScanRuler's `stepPairs`).
pub fn step_pairs(faces: &[FaceInfo], edges: &[EdgeInfo], options: &StepOptions) -> Vec<StepPair> {
    let step_mm = options.step_mm;
    let parallel_deg = options.parallel_deg;
    let points = extent_points(faces, edges);
    let next = neighbours(faces, edges);
    let mut out = Vec::new();
    let mut told: HashSet<String> = HashSet::new();
    let mut keep = |p: StepPair, out: &mut Vec<StepPair>| {
        let ka = format!("{}|{}", faces[p.faces[0]].fingerprint.origin, surface_key(&faces[p.faces[0]].fingerprint));
        let kb = format!("{}|{}", faces[p.faces[1]].fingerprint.origin, surface_key(&faces[p.faces[1]].fingerprint));
        let key = if ka < kb { format!("{}|{ka}|{kb}", p.form.name()) } else { format!("{}|{kb}|{ka}", p.form.name()) };
        if told.insert(key) {
            out.push(p);
        }
    };
    let riser_of = |a: usize, b: usize| -> Option<usize> {
        let mut best: Option<usize> = None;
        let mut list: Vec<&usize> = next[a].iter().collect();
        list.sort();
        for r in list {
            if next[b].contains(r) && best.is_none_or(|x| faces[*r].fingerprint.area < faces[x].fingerprint.area) {
                best = Some(*r);
            }
        }
        best
    };
    let mut edges_of: Vec<Vec<&EdgeInfo>> = vec![Vec::new(); faces.len()];
    {
        let index = face_index(faces);
        for e in edges {
            for h in unique(&e.faces) {
                if let Some(i) = index.get(&h) {
                    edges_of[*i].push(e);
                }
            }
        }
    }
    let near_face = |ri: usize, pi: usize| -> Option<StepPair> {
        let (r, p) = (&faces[ri].fingerprint, &faces[pi].fingerprint);
        if r.kind != FaceKind::Cylinder || p.kind != FaceKind::Plane || r.origin == p.origin {
            return None;
        }
        if r.dir.dot(p.dir).abs() > parallel_deg.to_radians().sin() {
            return None;
        }
        let (u, v) = basis_of(p.dir);
        let bp = box2(&points[pi], u, v);
        let off = |q: DVec3| (q - p.centroid).dot(p.dir);
        if faces[ri].full_turn != Some(true) {
            let mut best: Option<(f64, DVec3)> = None;
            for e in &edges_of[ri] {
                if e.fingerprint.kind != EdgeKind::Line || lean_deg(e.fingerprint.dir, r.dir) > parallel_deg {
                    continue;
                }
                let d = off(e.fingerprint.mid).abs();
                if !(d > SAME_MM && d <= step_mm) {
                    continue;
                }
                let be = box2(&[e.ends[0], e.ends[1]], u, v);
                if let (Some(bp), Some(be)) = (bp, be) {
                    if !boxes_meet(bp, be, MEET_MM) {
                        continue;
                    }
                }
                if best.is_none_or(|b| d < b.0) {
                    best = Some((d, e.fingerprint.mid));
                }
            }
            if let Some((d, at)) = best {
                return Some(StepPair { form: StepForm::Foot, faces: [ri, pi], off: d, at, riser: None, radii: None, axis_off: None, crease_deg: None });
            }
        }
        let (radius, center) = (r.radius?, r.center?);
        let h = off(center).abs();
        let dip = (h - radius).abs();
        if !(dip > SAME_MM && dip <= step_mm) {
            return None;
        }
        let near: Vec<DVec3> = points[ri].iter().copied().filter(|q| off(*q).abs() <= step_mm + dip).collect();
        if !points[ri].is_empty() && near.is_empty() {
            return None;
        }
        let bn = box2(&near, u, v);
        if let (Some(bp), Some(bn)) = (bp, bn) {
            if !boxes_meet(bp, bn, MEET_MM) {
                return None;
            }
        }
        let at = if near.is_empty() {
            r.centroid
        } else {
            let lo = near.iter().fold(near[0], |a, q| a.min(*q));
            let hi = near.iter().fold(near[0], |a, q| a.max(*q));
            mid3(lo, hi)
        };
        Some(StepPair { form: StepForm::Tangent, faces: [ri, pi], off: dip, at, riser: None, radii: None, axis_off: None, crease_deg: (h < radius).then(|| (h / radius).acos().to_degrees()) })
    };
    let where_ = |a: usize, b: usize, riser: Option<usize>, meet: Option<DVec3>| -> DVec3 {
        match riser {
            Some(r) => faces[r].fingerprint.centroid,
            None => meet.unwrap_or(mid3(faces[a].fingerprint.centroid, faces[b].fingerprint.centroid)),
        }
    };
    for x in 0..faces.len() {
        for y in (x + 1)..faces.len() {
            let (fa, fb) = (&faces[x], &faces[y]);
            let (a, b) = (&fa.fingerprint, &fb.fingerprint);
            if let Some(pair) = options.pair {
                if !pair(fa, fb) {
                    continue;
                }
            }
            if a.kind == FaceKind::Plane && b.kind == FaceKind::Plane {
                if facing_deg(a.dir, b.dir) > parallel_deg {
                    continue;
                }
                let n = (a.dir + b.dir).try_normalize().unwrap_or(a.dir);
                let gap = (b.centroid - a.centroid).dot(n).abs();
                if !(gap > SAME_MM && gap <= step_mm) {
                    continue;
                }
                let (u, v) = basis_of(n);
                let (ba, bb) = (box2(&points[x], u, v), box2(&points[y], u, v));
                if let (Some(ba), Some(bb)) = (ba, bb) {
                    if !boxes_meet(ba, bb, MEET_MM) {
                        continue;
                    }
                }
                let meet = match (ba, bb) {
                    (Some(ba), Some(bb)) => {
                        let c = mid3(a.centroid, b.centroid);
                        let cu = (ba.lo[0].max(bb.lo[0]) + ba.hi[0].min(bb.hi[0])) / 2.0 - c.dot(u);
                        let cv = (ba.lo[1].max(bb.lo[1]) + ba.hi[1].min(bb.hi[1])) / 2.0 - c.dot(v);
                        Some(c + u * cu + v * cv)
                    }
                    _ => None,
                };
                let riser = riser_of(x, y);
                keep(StepPair { form: StepForm::Flat, faces: [x, y], off: gap, at: where_(x, y, riser, meet), riser, radii: None, axis_off: None, crease_deg: None }, &mut out);
                continue;
            }
            let round = |k: FaceKind| matches!(k, FaceKind::Cylinder | FaceKind::Cone);
            if round(a.kind) && round(b.kind) && a.kind == b.kind {
                let (Some(ca), Some(cb)) = (a.center, b.center) else { continue };
                if lean_deg(a.dir, b.dir) > parallel_deg {
                    continue;
                }
                let axis_off = (cb - ca).cross(a.dir).length();
                if axis_off > step_mm {
                    continue;
                }
                let (sa, sb) = (span(&points[x], ca, a.dir), span(&points[y], ca, a.dir));
                let t = match (sa, sb) {
                    (Some(sa), Some(sb)) => (sa[0].max(sb[0]) + sa[1].min(sb[1])) / 2.0,
                    _ => 0.0,
                };
                if let (Some(sa), Some(sb)) = (sa, sb) {
                    if (sa[0].max(sb[0]) - sa[1].min(sb[1])).abs() > MEET_MM {
                        continue;
                    }
                }
                let (Some(ra), Some(rb)) = (radius_along(a, &points[x]), radius_along(b, &points[y])) else { continue };
                let s = {
                    let d = a.dir.dot(b.dir).signum();
                    if d == 0.0 {
                        1.0
                    } else {
                        d
                    }
                };
                let tb = (ca - cb).dot(b.dir) + t * s;
                let (r1, r2) = (ra.0 + ra.1 * t, rb.0 + rb.1 * tb);
                let dr = (r1 - r2).abs();
                if dr > step_mm || (dr <= SAME_MM && axis_off <= SAME_MM) {
                    continue;
                }
                if (ra.1.atan() - (rb.1 * s).atan()).abs().to_degrees() > parallel_deg {
                    continue;
                }
                let riser = riser_of(x, y);
                keep(StepPair { form: StepForm::Round, faces: [x, y], off: dr + axis_off, at: where_(x, y, riser, None), riser, radii: Some([r1, r2]), axis_off: Some(axis_off), crease_deg: None }, &mut out);
                continue;
            }
            if let Some(hit) = near_face(x, y).or_else(|| near_face(y, x)) {
                keep(hit, &mut out);
            }
        }
    }
    out
}

/// A step as a build's row keeps it: the two faces as fingerprints — the feature's own first.
#[derive(Debug, Clone)]
pub struct StepBetween {
    pub form: StepForm,
    pub off: f64,
    pub at: DVec3,
    pub faces: [FaceFingerprint; 2],
    pub radii: Option<[f64; 2]>,
    pub axis_off: Option<f64>,
    pub crease_deg: Option<f64>,
}

/// The steps a join left between the faces it brought (`mine`) and the faces it found.
pub fn steps_of_join(faces: &[FaceInfo], edges: &[EdgeInfo], mine: &dyn Fn(i64) -> bool) -> Vec<StepBetween> {
    let pair = |a: &FaceInfo, b: &FaceInfo| mine(a.fingerprint.origin) != mine(b.fingerprint.origin);
    let pairs = step_pairs(faces, edges, &StepOptions { pair: Some(&pair), ..Default::default() });
    let mut out: Vec<StepBetween> = pairs
        .iter()
        .map(|p| {
            let swapped = !mine(faces[p.faces[0]].fingerprint.origin);
            let (x, y) = if swapped { (p.faces[1], p.faces[0]) } else { (p.faces[0], p.faces[1]) };
            StepBetween { form: p.form, off: p.off, at: p.at, faces: [faces[x].fingerprint, faces[y].fingerprint], radii: p.radii.map(|r| if swapped { [r[1], r[0]] } else { r }), axis_off: p.axis_off, crease_deg: p.crease_deg }
        })
        .collect();
    out.sort_by(|a, b| ((a.off < HAIR_MM) as u8).cmp(&((b.off < HAIR_MM) as u8)).then(a.off.total_cmp(&b.off)));
    out
}

// ---- the audit (audit.ts) ----

#[derive(Debug, Clone, Copy)]
pub struct AuditOptions {
    pub kink_deg: f64,
    pub step_mm: f64,
    pub parallel_deg: f64,
    pub short_edge: f64,
    pub limit: usize,
    pub alike: bool,
    pub tolerance: Option<f64>,
}

pub const AUDIT_DEFAULTS: AuditOptions = AuditOptions { kink_deg: 5.0, step_mm: 0.3, parallel_deg: 1.0, short_edge: 0.1, limit: 12, alike: false, tolerance: None };

/// A step under this, mm — or under half the tolerance given, if more — is asked about.
pub const ASK_MM: f64 = 0.1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum FindingKind {
    Kink,
    Step,
    NearParallel,
    NearAxis,
    ShortEdge,
    FreeEdge,
    NonManifold,
    Contact,
}

impl FindingKind {
    pub fn name(self) -> &'static str {
        match self {
            FindingKind::Kink => "kink",
            FindingKind::Step => "step",
            FindingKind::NearParallel => "nearParallel",
            FindingKind::NearAxis => "nearAxis",
            FindingKind::ShortEdge => "shortEdge",
            FindingKind::FreeEdge => "freeEdge",
            FindingKind::NonManifold => "nonManifold",
            FindingKind::Contact => "contact",
        }
    }

    /// What the findings are listed under in the body check.
    pub fn list(self) -> &'static str {
        match self {
            FindingKind::Kink => "kinks",
            FindingKind::Step => "steps",
            FindingKind::NearParallel => "nearParallel",
            FindingKind::NearAxis => "nearAxis",
            FindingKind::ShortEdge => "shortEdges",
            FindingKind::FreeEdge => "freeEdges",
            FindingKind::NonManifold => "nonManifold",
            FindingKind::Contact => "contacts",
        }
    }
}

/// An edge the kernel found odd: one face only, three or more, its one face on both sides of
/// it, or two faces facing opposite ways.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OddKind {
    Free,
    Inside,
    NonManifold,
    Knife,
}

impl OddKind {
    pub fn name(self) -> &'static str {
        match self {
            OddKind::Free => "free",
            OddKind::Inside => "inside",
            OddKind::NonManifold => "nonManifold",
            OddKind::Knife => "knife",
        }
    }
}

#[derive(Debug, Clone)]
pub struct OddEdge {
    pub hash: u32,
    pub kind: OddKind,
    pub at: DVec3,
    pub length: f64,
    pub faces: Vec<u32>,
}

#[derive(Debug, Clone)]
pub struct Finding {
    pub kind: FindingKind,
    pub faces: Vec<usize>,
    pub edges: Vec<usize>,
    pub at: DVec3,
    pub off: f64,
    pub deg: bool,
    pub axis: Option<char>,
    pub form: Option<StepForm>,
    pub between: Option<[i64; 2]>,
    pub count: Option<usize>,
    pub places: Option<Vec<DVec3>>,
    pub hair: bool,
    pub riser: Option<usize>,
    pub note: String,
}

impl Finding {
    fn new(kind: FindingKind, faces: Vec<usize>, edges: Vec<usize>, at: DVec3, off: f64, deg: bool, note: String) -> Finding {
        Finding { kind, faces, edges, at, off, deg, axis: None, form: None, between: None, count: None, places: None, hair: false, riser: None, note }
    }
}

/// Under this, two directions are one.
const SAME_DEG: f64 = 0.01;
/// How near a face must lie to the sketch piece it is traced to, mm.
const ON_MM: f64 = 2e-3;
const PLACES_SHOWN: usize = 3;

pub fn round4(x: f64) -> f64 {
    (x * 10000.0).round() / 10000.0
}

/// A number written as JavaScript writes it (no trailing zeros).
pub fn js(x: f64) -> String {
    crate::js_fmt::num(x)
}

/// A face's outward normal at a point of its edge, where the fingerprint says it.
fn normal_at(face: &FaceInfo, p: DVec3, is_boss: Option<&dyn Fn(&FaceInfo) -> Option<bool>>) -> Option<DVec3> {
    let fp = &face.fingerprint;
    match (fp.kind, fp.center) {
        (FaceKind::Plane, _) => Some(fp.dir),
        (FaceKind::Cylinder, Some(c)) => {
            let boss = is_boss.and_then(|f| f(face))?;
            let d = p - c;
            let n = (d - fp.dir * d.dot(fp.dir)).try_normalize()?;
            Some(if boss { n } else { -n })
        }
        (FaceKind::Sphere, Some(c)) => {
            let boss = is_boss.and_then(|f| f(face))?;
            let n = (p - c).try_normalize()?;
            Some(if boss { n } else { -n })
        }
        _ => None,
    }
}

fn one_of_alike(list: Vec<Finding>, faces: &[FaceInfo]) -> Vec<Finding> {
    let mut groups: Vec<(String, Vec<Finding>)> = Vec::new();
    for f in list {
        let mut made: Vec<String> = f.faces.iter().map(|i| format!("{}:{}", faces[*i].fingerprint.origin, faces[*i].fingerprint.kind.name())).collect();
        made.sort();
        let key = format!("{}|{}|{}", f.kind.name(), f.off, made.join(","));
        match groups.iter_mut().find(|g| g.0 == key) {
            Some(g) => g.1.push(f),
            None => groups.push((key, vec![f])),
        }
    }
    groups
        .into_iter()
        .map(|(_, mut g)| {
            let rest: Vec<Finding> = g.split_off(1);
            let mut first = g.remove(0);
            if !rest.is_empty() {
                first.count = Some(rest.len() + 1);
                first.places = Some(rest.iter().take(PLACES_SHOWN).map(|f| f.at).collect());
            }
            first
        })
        .collect()
}

fn face_word(fp: &FaceFingerprint) -> &'static str {
    match fp.kind {
        FaceKind::Plane => "flat face",
        FaceKind::Other => "face",
        k => k.name(),
    }
}

/// What the audit came to.
#[derive(Debug, Clone, Default)]
pub struct Audit {
    pub findings: Vec<Finding>,
    pub steps_between_features: usize,
    pub found: Vec<(FindingKind, usize)>,
    pub under_half_t: Option<usize>,
}

/// How far a lean of `deg` moves a face across `span` mm.
pub fn lean_mm(deg: f64, span: f64) -> f64 {
    span * deg.to_radians().tan()
}

/// Each face's span, mm: the diagonal of the box round its edges.
pub fn face_spans(faces: &[FaceInfo], edges: &[EdgeInfo]) -> Vec<f64> {
    let index = face_index(faces);
    let mut lo = vec![DVec3::splat(f64::INFINITY); faces.len()];
    let mut hi = vec![DVec3::splat(f64::NEG_INFINITY); faces.len()];
    for e in edges {
        for h in &e.faces {
            let Some(i) = index.get(h) else { continue };
            let pts: &[DVec3] = if e.polyline.is_empty() { &e.ends } else { &e.polyline };
            for q in pts {
                lo[*i] = lo[*i].min(*q);
                hi[*i] = hi[*i].max(*q);
            }
        }
    }
    (0..faces.len()).map(|i| if lo[i].x <= hi[i].x { (hi[i] - lo[i]).length() } else { 0.0 }).collect()
}

fn step_mm_word(x: f64) -> f64 {
    if x >= HAIR_MM {
        round4(x)
    } else {
        crate::js_fmt::to_precision_num(x, 2)
    }
}

fn step_note(p: &StepPair, between: bool) -> String {
    let off = js(step_mm_word(p.off));
    match p.form {
        StepForm::Round => {
            let [ra, rb] = p.radii.unwrap_or([0.0, 0.0]);
            let axes = match p.axis_off {
                Some(a) if a > SAME_MM => format!(", axes {} mm apart", js(round4(a))),
                _ => String::new(),
            };
            format!("two rounds on nearly one axis meet end to end, R{} and R{}{} — one round drawn twice{}", js(round4(ra)), js(round4(rb)), axes, if between { ", in two features" } else { "" })
        }
        StepForm::Foot => format!("a round starts {off} mm off a flat face of another feature — a profile that nearly starts on that face"),
        StepForm::Tangent => match p.crease_deg {
            Some(c) => format!("a round dips {off} mm through a flat face of another feature, meeting it at {}° — a round meant to touch that face", js(round4(c))),
            None => format!("a round stands {off} mm short of a flat face of another feature — a round meant to touch that face"),
        },
        StepForm::Flat => {
            if between {
                format!("two flat faces of two features facing one way lie {off} mm apart — a wall or a height drawn twice, in two sketches")
            } else {
                format!("two flat faces facing one way lie {off} mm apart — one face, or a step of {off} mm")
            }
        }
    }
}

fn step_lines(pairs: &[StepPair], faces: &[FaceInfo], limit: usize, ask: f64) -> Vec<Finding> {
    let mut groups: Vec<(String, Vec<&StepPair>)> = Vec::new();
    for p in pairs {
        let (oa, ob) = (faces[p.faces[0]].fingerprint.origin, faces[p.faces[1]].fingerprint.origin);
        let key = format!("{},{},{},{}", oa.min(ob), oa.max(ob), p.form.name(), p.off < HAIR_MM);
        match groups.iter_mut().find(|g| g.0 == key) {
            Some(g) => g.1.push(p),
            None => groups.push((key, vec![p])),
        }
    }
    let mut lines: Vec<Finding> = groups
        .iter()
        .map(|(_, g)| {
            let w = g.iter().copied().reduce(|a, b| if b.off > a.off { b } else { a }).expect("a group has a pair");
            let (oa, ob) = (faces[w.faces[0]].fingerprint.origin, faces[w.faces[1]].fingerprint.origin);
            let between = oa != ob;
            let hair = w.off < HAIR_MM;
            let off = step_mm_word(w.off);
            let at = (w.at * 1000.0).round() / 1000.0;
            let mut note = if g.len() == 1 {
                step_note(w, between)
            } else if hair {
                String::new()
            } else {
                format!("{} steps between the {} of {}, the widest {} mm: {}", g.len(), if w.form == StepForm::Round { "rounds" } else { "faces" }, if between { "two features" } else { "one feature" }, js(off), step_note(w, between))
            };
            if hair {
                note = format!(
                    "{} under {} mm between the faces of {}{} — one face cut in two by a piece fitted twice",
                    if g.len() == 1 { "a hairline step".to_string() } else { format!("{} hairline steps", g.len()) },
                    js(HAIR_MM),
                    if between { "two features" } else { "one feature" },
                    if g.len() > 1 { format!(", the widest {} mm", js(off)) } else { format!(" ({} mm)", js(off)) }
                );
            } else if w.off < ask {
                note += &format!(" — a step of {} mm at ({}, {}, {}): meant? A step under the tolerance is almost never the part’s", js(off), js(at.x), js(at.y), js(at.z));
            }
            let mut f = Finding::new(FindingKind::Step, w.faces.to_vec(), Vec::new(), w.at, off, false, note);
            f.form = Some(w.form);
            f.between = Some([oa.min(ob), oa.max(ob)]);
            f.count = Some(g.len());
            f.hair = hair;
            f.riser = w.riser;
            f
        })
        .collect();
    let rank = |f: &Finding| if f.between.is_some_and(|b| b[0] != b[1]) { 0 } else { 2 } + f.hair as i32;
    lines.sort_by(|a, b| rank(a).cmp(&rank(b)).then(a.off.total_cmp(&b.off)));
    lines.truncate(limit);
    lines
}

/// The body's topology looked over (ScanRuler's `auditBody`).
pub fn audit_body(faces: &[FaceInfo], edges: &[EdgeInfo], o: &AuditOptions, is_boss: Option<&dyn Fn(&FaceInfo) -> Option<bool>>, odd: &[OddEdge]) -> Audit {
    let index = face_index(faces);
    let mut kinks = Vec::new();
    let pairs = step_pairs(faces, edges, &StepOptions { step_mm: o.step_mm, parallel_deg: o.parallel_deg, pair: None });
    let steps_between_features = pairs.iter().filter(|p| faces[p.faces[0]].fingerprint.origin != faces[p.faces[1]].fingerprint.origin).count();
    let steps = step_lines(&pairs, faces, o.limit, ASK_MM.max(o.tolerance.unwrap_or(0.0) / 2.0));
    let mut near_parallel = Vec::new();
    let mut near_axis = Vec::new();
    let mut short_edges = Vec::new();
    for (ie, e) in edges.iter().enumerate() {
        let fp = &e.fingerprint;
        if (fp.length > 0.0 || !e.polyline.is_empty()) && fp.length < o.short_edge {
            let hair = fp.length < 1e-4;
            let fs: Vec<usize> = e.faces.iter().filter_map(|h| index.get(h).copied()).collect();
            let note = if hair { "an edge under 0.0001 mm long — a hair a round left, which a STEP reader may drop from one face".to_string() } else { format!("an edge {} mm long — a sliver's, or a corner that nearly meets", js(round4(fp.length))) };
            short_edges.push(Finding::new(FindingKind::ShortEdge, fs, vec![ie], fp.mid, round4(fp.length), false, note));
        }
        if e.faces.len() != 2 {
            continue;
        }
        let (Some(ia), Some(ib)) = (index.get(&e.faces[0]).copied(), index.get(&e.faces[1]).copied()) else { continue };
        let (fa, fb) = (&faces[ia], &faces[ib]);
        let (Some(na), Some(nb)) = (normal_at(fa, fp.mid, is_boss), normal_at(fb, fp.mid, is_boss)) else { continue };
        let d = angle_deg(na, nb);
        if d <= SAME_DEG || d > o.kink_deg {
            continue;
        }
        let flat = fa.fingerprint.kind == FaceKind::Plane && fb.fingerprint.kind == FaceKind::Plane;
        let note = if flat { format!("two flat faces meet at {}° — a crease where one face was meant, or a step's riser seen edge on", js(round4(d))) } else { format!("a {} and a {} meet at {}° where a round would run on tangent", face_word(&fa.fingerprint), face_word(&fb.fingerprint), js(round4(d))) };
        kinks.push(Finding::new(FindingKind::Kink, vec![ia, ib], vec![ie], fp.mid, round4(d), true, note));
    }
    let planes: Vec<usize> = (0..faces.len()).filter(|i| faces[*i].fingerprint.kind == FaceKind::Plane).collect();
    for x in 0..planes.len() {
        for y in (x + 1)..planes.len() {
            let (a, b) = (&faces[planes[x]].fingerprint, &faces[planes[y]].fingerprint);
            let lean = lean_deg(a.dir, b.dir);
            if lean > o.parallel_deg || lean <= SAME_DEG {
                continue;
            }
            near_parallel.push(Finding::new(FindingKind::NearParallel, vec![planes[x], planes[y]], Vec::new(), mid3(a.centroid, b.centroid), round4(lean), true, format!("two flat faces lean {}° to each other — parallel, if meant so", js(round4(lean)))));
        }
    }
    for (i, f) in faces.iter().enumerate() {
        let fp = &f.fingerprint;
        if !matches!(fp.kind, FaceKind::Plane | FaceKind::Cylinder | FaceKind::Cone) {
            continue;
        }
        let mut best: Option<(char, f64)> = None;
        for (axis, dir) in [('x', DVec3::X), ('y', DVec3::Y), ('z', DVec3::Z)] {
            let lean = lean_deg(fp.dir, dir);
            if best.is_none_or(|b| lean < b.1) {
                best = Some((axis, lean));
            }
        }
        let Some((axis, lean)) = best else { continue };
        if lean <= SAME_DEG || lean > o.parallel_deg {
            continue;
        }
        let note = if fp.kind == FaceKind::Plane { format!("a flat face leans {}° off square to the {axis} axis", js(round4(lean))) } else { format!("a {}'s axis leans {}° off the {axis} axis", fp.kind.name(), js(round4(lean))) };
        let mut finding = Finding::new(FindingKind::NearAxis, vec![i], Vec::new(), fp.centroid, round4(lean), true, note);
        finding.axis = Some(axis);
        near_axis.push(finding);
    }
    let edge_index: HashMap<u32, usize> = edges.iter().enumerate().map(|(i, e)| (e.hash, i)).collect();
    let (mut free, mut non_manifold, mut contacts) = (Vec::new(), Vec::new(), Vec::new());
    for e in odd {
        let fs: Vec<usize> = e.faces.iter().filter_map(|h| index.get(h).copied()).collect();
        let es: Vec<usize> = edge_index.get(&e.hash).map(|i| vec![*i]).unwrap_or_default();
        let len = js(round4(e.length));
        let make = |kind, note: String| Finding::new(kind, fs.clone(), es.clone(), e.at, round4(e.length), false, note);
        match e.kind {
            OddKind::Free => free.push(make(FindingKind::FreeEdge, format!("an edge {len} mm long with one face — a hole in the skin: the body is not closed here"))),
            OddKind::NonManifold => non_manifold.push(make(FindingKind::NonManifold, format!("{} faces meet at one edge {len} mm long — two parts of the body touch along it instead of joining; a STEP reader takes it for no closed solid", e.faces.len()))),
            OddKind::Inside => contacts.push(make(FindingKind::Contact, format!("an edge {len} mm long inside its one face — another face only touched this one at a tangent there, and the boolean kept the line"))),
            OddKind::Knife => contacts.push(make(FindingKind::Contact, format!("two faces meet facing opposite ways along an edge {len} mm long — the body is no thick there, a tangent touch"))),
        }
    }
    let mut under_half_t = None;
    if let Some(t) = o.tolerance {
        let spans = face_spans(faces, edges);
        let counts = |f: &Finding| {
            let across = f.faces.iter().map(|i| spans.get(*i).copied().unwrap_or(0.0)).fold(f64::NEG_INFINITY, f64::max);
            !(across > 0.0) || lean_mm(f.off, across) >= t / 2.0
        };
        let before = near_parallel.len() + near_axis.len();
        near_parallel.retain(|f| counts(f));
        near_axis.retain(|f| counts(f));
        under_half_t = Some(before - near_parallel.len() - near_axis.len());
    }
    let by_off = |a: &Finding, b: &Finding| a.off.total_cmp(&b.off);
    let longest = |a: &Finding, b: &Finding| b.off.total_cmp(&a.off);
    let mut found = Vec::new();
    for (k, n) in [(FindingKind::FreeEdge, free.len()), (FindingKind::NonManifold, non_manifold.len()), (FindingKind::Contact, contacts.len()), (FindingKind::Kink, kinks.len()), (FindingKind::Step, steps.len()), (FindingKind::NearParallel, near_parallel.len()), (FindingKind::NearAxis, near_axis.len()), (FindingKind::ShortEdge, short_edges.len())] {
        if n > 0 {
            found.push((k, n));
        }
    }
    let once = |list: Vec<Finding>| if o.alike { one_of_alike(list, faces) } else { list };
    let mut findings = Vec::new();
    let mut take = |mut list: Vec<Finding>, cmp: &dyn Fn(&Finding, &Finding) -> std::cmp::Ordering, alike: bool| {
        list.sort_by(cmp);
        let list = if alike { once(list) } else { list };
        findings.extend(list.into_iter().take(o.limit));
    };
    take(free, &longest, false);
    take(non_manifold, &longest, false);
    take(contacts, &longest, false);
    take(kinks, &by_off, true);
    take(steps, &|_, _| std::cmp::Ordering::Equal, false);
    take(near_parallel, &by_off, true);
    take(near_axis, &by_off, true);
    take(short_edges, &by_off, true);
    Audit { findings, steps_between_features, found, under_half_t }
}

// ---- back to the sketch ----

/// Where a face came from, as far as the feature list says.
#[derive(Debug, Clone)]
pub struct FaceTrace {
    pub feature: (u32, String, &'static str),
    pub sketch: Option<(u32, String)>,
    pub piece: Option<(u32, &'static str)>,
    pub role: &'static str,
    pub end: Option<&'static str>,
}

fn shape_word(s: &Shape) -> &'static str {
    match s {
        Shape::Line { .. } => "line",
        Shape::Arc { .. } => "arc",
        Shape::Circle { .. } => "circle",
        Shape::Point { .. } => "point",
        Shape::Spline(_) => "spline",
    }
}

/// Where a face came from (ScanRuler's `traceFace`).
pub fn trace_face(features: &[Feature], face: &FaceFingerprint) -> Option<FaceTrace> {
    let maker = features.iter().find(|f| f.id as i64 == face.origin)?;
    let feature = (maker.id, maker.name.clone(), maker.kind.kind_name());
    let sketch_id = match &maker.kind {
        FeatureKind::Extrude(e) => Some(e.sketch_id),
        FeatureKind::Revolve(r) => Some(r.sketch_id),
        _ => None,
    };
    let Some(sketch_id) = sketch_id else { return Some(FaceTrace { feature, sketch: None, piece: None, role: "other", end: None }) };
    let Some((sid, sname, s)) = features.iter().find_map(|f| match &f.kind {
        FeatureKind::Sketch(s) if f.id == sketch_id => Some((f.id, f.name.clone(), s)),
        _ => None,
    }) else {
        return Some(FaceTrace { feature, sketch: None, piece: None, role: "other", end: None });
    };
    let frame = s.frame;
    let named = Some((sid, sname));
    let pieces: Vec<&crate::sketch::Entity> = s.sketch.entities.iter().filter(|e| !e.construction).collect();
    let lift = |p: glam::DVec2| frame.to_world(p);
    if face.kind == FaceKind::Plane {
        let along = face.dir.dot(frame.normal).abs();
        if let (FeatureKind::Extrude(e), true) = (&maker.kind, along > 1.0 - 1e-6) {
            let at = (face.centroid - frame.origin).dot(frame.normal);
            let ends = match &e.extent {
                ExtrudeExtent::Distance { from, to } => Some((*from, *to)),
                ExtrudeExtent::Symmetric { total } => Some((-total / 2.0, total / 2.0)),
                _ => None,
            };
            let end = ends.and_then(|(from, to)| if (at - to).abs() <= ON_MM { Some("to") } else if (at - from).abs() <= ON_MM { Some("from") } else { None });
            return Some(FaceTrace { feature, sketch: named, piece: None, role: "cap", end });
        }
        if along < 1e-4 {
            let off = |p: DVec3| (p - face.centroid).dot(face.dir).abs();
            let mut best: Option<(u32, &'static str)> = None;
            let mut nearest = f64::INFINITY;
            for e in &pieces {
                let Shape::Line { a, b } = e.shape else { continue };
                let (a, b) = (lift(a), lift(b));
                if off(a) > ON_MM || off(b) > ON_MM {
                    continue;
                }
                let d = (mid3(a, b) - face.centroid).length();
                if d < nearest {
                    nearest = d;
                    best = Some((e.id, "line"));
                }
            }
            return Some(FaceTrace { feature, sketch: named, piece: best, role: "wall", end: None });
        }
        return Some(FaceTrace { feature, sketch: named, piece: None, role: "other", end: None });
    }
    if let (FaceKind::Cylinder, Some(r), Some(c)) = (face.kind, face.radius, face.center) {
        let off_axis = |p: DVec3| (p - c).cross(face.dir).length();
        for e in &pieces {
            let (ec, er) = match e.shape {
                Shape::Circle { c, r } | Shape::Arc { c, r, .. } => (c, r),
                _ => continue,
            };
            if (er - r).abs() <= ON_MM && off_axis(lift(ec)) <= ON_MM {
                return Some(FaceTrace { feature, sketch: named, piece: Some((e.id, shape_word(&e.shape))), role: "round", end: None });
            }
        }
        if matches!(maker.kind, FeatureKind::Revolve(_)) {
            for e in &pieces {
                let Shape::Line { a, b } = e.shape else { continue };
                if (off_axis(lift(a)) - r).abs() <= ON_MM && (off_axis(lift(b)) - r).abs() <= ON_MM {
                    return Some(FaceTrace { feature, sketch: named, piece: Some((e.id, "line")), role: "round", end: None });
                }
            }
        }
        return Some(FaceTrace { feature, sketch: named, piece: None, role: "round", end: None });
    }
    Some(FaceTrace { feature, sketch: named, piece: None, role: "other", end: None })
}

// ---- the steps a join left, said (stepWords.ts) ----

/// How many steps a change's answer lists.
pub const STEPS_LISTED: usize = 5;
const STEPS_WORDED: usize = 3;

fn r2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

fn mm_words(x: f64) -> String {
    if x >= 0.01 {
        js((x * 1000.0).round() / 1000.0)
    } else if x >= 1e-4 {
        js((x * 1e5).round() / 1e5)
    } else {
        js_exponential(x)
    }
}

/// A number as JavaScript's `toExponential(1)` writes it: `4.0e-5`.
fn js_exponential(x: f64) -> String {
    let s = format!("{:.1e}", x);
    match s.split_once('e') {
        Some((m, e)) if !e.starts_with('-') => format!("{m}e+{e}"),
        _ => s,
    }
}

/// Where a flat face lies: its coordinate where it is square to an axis, else its middle.
pub fn place_of(fp: &FaceFingerprint) -> String {
    let square = 0.01f64.to_radians().cos();
    if fp.kind == FaceKind::Plane {
        for (k, a) in ["x", "y", "z"].iter().enumerate() {
            if fp.dir[k].abs() > square {
                return format!("{a} {}", js(r2(fp.centroid[k])));
            }
        }
    }
    format!("at ({}, {}, {})", js(r2(fp.centroid.x)), js(r2(fp.centroid.y)), js(r2(fp.centroid.z)))
}

fn piece_of(t: &Option<FaceTrace>) -> Option<String> {
    let t = t.as_ref()?;
    match (&t.piece, &t.sketch) {
        (Some((id, kind)), Some((_, name))) => Some(format!("{kind} {id} of {name}")),
        _ => None,
    }
}

fn maker_of(t: &Option<FaceTrace>, fp: &FaceFingerprint) -> String {
    t.as_ref().map_or(format!("feature {}", fp.origin), |t| t.feature.1.clone())
}

fn face_words(t: &Option<FaceTrace>, fp: &FaceFingerprint, radius: Option<f64>) -> String {
    let what = if matches!(fp.kind, FaceKind::Cylinder | FaceKind::Cone) {
        "round"
    } else {
        match t.as_ref().map(|t| t.role) {
            Some("wall") => "wall",
            Some("cap") => "end face",
            _ => "face",
        }
    };
    let place = radius.map_or_else(|| place_of(fp), |r| format!("R{}", mm_words(r)));
    match piece_of(t) {
        Some(p) => format!("{what} ({place}, {p})"),
        None => format!("{what} ({place})"),
    }
}

fn fix_words(a: &Option<FaceTrace>, b: &Option<FaceTrace>) -> String {
    match (a.as_ref().and_then(|t| t.sketch.clone().map(|s| (t, s))), b.as_ref().and_then(|t| t.sketch.clone().map(|s| (t, s)))) {
        (Some((ta, sa)), Some((tb, sb))) if sa.0 == sb.0 => match (ta.piece, tb.piece) {
            (Some(pa), Some(pb)) => format!("make {} {} and {} {} of {} one (a collinear or concentric relation, or one piece for both)", pa.1, pa.0, pb.1, pb.0, sa.1),
            _ => format!("draw them as one piece in {}", sa.1),
        },
        (Some((_, sa)), Some((tb, sb))) => {
            let project = match tb.piece {
                Some(pb) => format!("project {} {} (project: [{{ sketch: {:?}, pieces: [{}] }}]) into {} in its place", pb.1, pb.0, sb.1, pb.0, sa.1),
                None => format!("project {}'s piece into {}", sb.1, sa.1),
            };
            format!("fit {}'s level into {} (fit.level), or {project}", sa.1, sb.1)
        }
        _ => "one surface for both: project the one into the other’s sketch, or dimension both from one plane".into(),
    }
}

/// One step in a sentence, the join's own face first.
pub fn step_sentence(s: &StepBetween, features: &[Feature]) -> String {
    let [fa, fb] = &s.faces;
    let (ta, tb) = (trace_face(features, fa), trace_face(features, fb));
    let (a, b) = (maker_of(&ta, fa), maker_of(&tb, fb));
    let off = mm_words(s.off);
    match s.form {
        StepForm::Round => {
            let [ra, rb] = s.radii.unwrap_or([fa.radius.unwrap_or(0.0), fb.radius.unwrap_or(0.0)]);
            let axes = match s.axis_off {
                Some(x) if x > 1e-4 => format!(", axes {} mm apart", mm_words(x)),
                _ => String::new(),
            };
            format!("{a}'s {} runs {off} mm off {b}'s {}{axes}: one round drawn twice — {}", face_words(&ta, fa, Some(ra)), face_words(&tb, fb, Some(rb)).trim_start_matches("round ").to_string(), fix_words(&ta, &tb))
        }
        StepForm::Foot | StepForm::Tangent => {
            let round_first = fa.kind == FaceKind::Cylinder;
            let (tr, fr, tp, fpl) = if round_first { (&ta, fa, &tb, fb) } else { (&tb, fb, &ta, fa) };
            let sketch = tr.as_ref().and_then(|t| t.sketch.as_ref().map(|s| s.1.clone()));
            let round = format!("{}'s {}", maker_of(tr, fr), face_words(tr, fr, fr.radius));
            let fw = face_words(tp, fpl, None);
            let fw = ["wall ", "end face ", "face "].iter().find_map(|p| fw.strip_prefix(p)).map_or(fw.clone(), |rest| format!("face {rest}"));
            let face = format!("{}'s {fw}", maker_of(tp, fpl));
            if s.form == StepForm::Foot {
                return format!("{round} starts {off} mm off {face}: a profile that nearly starts on that face — hold its end on the face{}, or let it start clearly off it", sketch.as_ref().map_or(String::new(), |s| format!(" (project the face’s edge into {s} and put the end on it)")));
            }
            let how = match s.crease_deg {
                Some(c) => format!("dips {off} mm through {face}, meeting it at {}°", js((c * 10.0).round() / 10.0)),
                None => format!("stands {off} mm short of {face}"),
            };
            format!("{round} {how}: a round meant to touch that face — make it tangent to the face{}, or let it cross clearly", sketch.as_ref().map_or(String::new(), |s| format!(" (project the face’s edge into {s} and hold the arc tangent to it)")))
        }
        StepForm::Flat => {
            let walls = ta.as_ref().is_some_and(|t| t.role == "wall") && tb.as_ref().is_some_and(|t| t.role == "wall");
            let caps = ta.as_ref().is_some_and(|t| t.role == "cap") && tb.as_ref().is_some_and(|t| t.role == "cap");
            let fwb = face_words(&tb, fb, None);
            let other = ["wall ", "end face ", "face "].iter().find_map(|p| fwb.strip_prefix(p)).map_or(fwb.clone(), |r| r.to_string());
            if walls {
                format!("{a}'s {} stands {off} mm off {b}'s {other}: one wall drawn twice — {}", face_words(&ta, fa, None), fix_words(&ta, &tb))
            } else if caps {
                format!("{a}'s {} lies {off} mm off {b}'s {other}: two heights meant as one — give both extents one parameter, or extrude {a} up to {b}'s face", face_words(&ta, fa, None))
            } else {
                format!("{a}'s {} lies {off} mm off {b}'s {}, facing one way: two faces meant as one — {}", face_words(&ta, fa, None), fwb, fix_words(&ta, &tb))
            }
        }
    }
}

/// A join's steps as its row's warning says them.
pub fn steps_warning(steps: &[StepBetween], features: &[Feature]) -> String {
    let said: Vec<String> = steps.iter().take(STEPS_WORDED).map(|s| format!("{}.", step_sentence(s, features))).collect();
    let more = steps.len() - said.len();
    let hairs = steps.iter().skip(STEPS_WORDED).filter(|s| s.off < HAIR_MM).count();
    let lead = if steps.len() == 1 { "A step between features:".to_string() } else { format!("{} steps between features:", steps.len()) };
    let rest = if more > 0 {
        let h = if hairs > 0 { format!(" ({} hairline{} under {} mm)", if hairs == more { "all".to_string() } else { hairs.to_string() }, if hairs == 1 { "" } else { "s" }, js(HAIR_MM)) } else { String::new() };
        format!(" And {more} more{h} — reverse.body.check lists them by feature.")
    } else {
        String::new()
    };
    format!("{lead} {}{rest}", said.join(" "))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topo::{EdgeFingerprint, EdgeSide};

    fn plane(hash: u32, origin: i64, centroid: DVec3, dir: DVec3, area: f64) -> FaceInfo {
        FaceInfo { hash, fingerprint: FaceFingerprint { kind: FaceKind::Plane, centroid, dir, area, radius: None, center: None, origin }, full_turn: None, point: centroid }
    }

    fn edge(hash: u32, faces: Vec<u32>, a: DVec3, b: DVec3) -> EdgeInfo {
        let mid = (a + b) / 2.0;
        EdgeInfo { hash, fingerprint: EdgeFingerprint { kind: EdgeKind::Line, mid, dir: (b - a).normalize_or_zero(), length: (b - a).length(), radius: None, faces: vec![EdgeSide { kind: FaceKind::Plane, dir: DVec3::Z }], origin: 1 }, faces, ends: [a, b], end_tangents: [DVec3::X; 2], polyline: vec![a, b] }
    }

    // Ported from ScanRuler's tests/reStepsBetween.test.ts: two tops a hair apart that meet
    // across a riser.
    #[test]
    fn two_flat_faces_a_hair_apart_meeting_across_a_riser_are_a_step() {
        let faces = vec![
            plane(0, 1, DVec3::new(-5.0, 0.0, 10.0), DVec3::Z, 100.0),
            plane(1, 2, DVec3::new(5.0, 0.0, 10.05), DVec3::Z, 100.0),
            plane(2, 2, DVec3::new(0.0, 0.0, 10.025), DVec3::X, 0.5),
        ];
        let edges = vec![
            edge(0, vec![0, 2], DVec3::new(0.0, -5.0, 10.0), DVec3::new(0.0, 5.0, 10.0)),
            edge(1, vec![1, 2], DVec3::new(0.0, -5.0, 10.05), DVec3::new(0.0, 5.0, 10.05)),
            edge(2, vec![0], DVec3::new(-10.0, -5.0, 10.0), DVec3::new(0.0, -5.0, 10.0)),
            edge(3, vec![1], DVec3::new(0.0, -5.0, 10.05), DVec3::new(10.0, -5.0, 10.05)),
        ];
        let pairs = step_pairs(&faces, &edges, &StepOptions::default());
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].form, StepForm::Flat);
        assert!((pairs[0].off - 0.05).abs() < 1e-9);
        assert_eq!(pairs[0].riser, Some(2));
        // Two apart by more than the step are not.
        let far = vec![faces[0].clone(), plane(1, 2, DVec3::new(5.0, 0.0, 11.0), DVec3::Z, 100.0)];
        assert!(step_pairs(&far, &[], &StepOptions::default()).is_empty());
        // Only the pairs asked for.
        let mine = |o: i64| o == 2;
        let steps = steps_of_join(&faces, &edges, &mine);
        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0].faces[0].origin, 2, "the join's own face first");
    }

    // Ported from ScanRuler's tests/reAudit.test.ts.
    #[test]
    fn the_audit_names_leans_steps_and_short_edges() {
        let lean = DVec3::new(0.005, 0.0, 1.0).normalize();
        let faces = vec![
            plane(0, 1, DVec3::new(0.0, 0.0, 10.0), lean, 100.0),
            plane(1, 2, DVec3::new(0.0, 0.0, 0.0), -DVec3::Z, 100.0),
            plane(2, 1, DVec3::new(20.0, 0.0, 5.0), DVec3::X, 50.0),
        ];
        let edges = vec![edge(0, vec![0, 2], DVec3::new(20.0, 0.0, 10.0), DVec3::new(20.0, 0.05, 10.0)), edge(1, vec![1, 2], DVec3::new(20.0, -5.0, 0.0), DVec3::new(20.0, 5.0, 0.0))];
        let audit = audit_body(&faces, &edges, &AUDIT_DEFAULTS, None, &[]);
        let kinds: Vec<FindingKind> = audit.findings.iter().map(|f| f.kind).collect();
        assert!(kinds.contains(&FindingKind::NearAxis), "{kinds:?}");
        assert!(kinds.contains(&FindingKind::NearParallel));
        assert!(kinds.contains(&FindingKind::ShortEdge));
        let axis = audit.findings.iter().find(|f| f.kind == FindingKind::NearAxis).unwrap();
        assert_eq!(axis.axis, Some('z'));
        assert!((axis.off - 0.2865).abs() < 1e-3);
        let short = audit.findings.iter().find(|f| f.kind == FindingKind::ShortEdge).unwrap();
        assert!((short.off - 0.05).abs() < 1e-9);
        // With a tolerance the lean that moves its face under T/2 is left out.
        let o = AuditOptions { tolerance: Some(1.0), ..AUDIT_DEFAULTS };
        let audit = audit_body(&faces, &edges, &o, None, &[]);
        assert!(audit.under_half_t.unwrap() >= 1);
        // Odd edges come first, longest first.
        let odd = [OddEdge { hash: 1, kind: OddKind::Free, at: DVec3::ZERO, length: 3.0, faces: vec![1] }, OddEdge { hash: 0, kind: OddKind::Free, at: DVec3::ZERO, length: 5.0, faces: vec![0] }];
        let audit = audit_body(&faces, &edges, &AUDIT_DEFAULTS, None, &odd);
        assert_eq!(audit.findings[0].kind, FindingKind::FreeEdge);
        assert!((audit.findings[0].off - 5.0).abs() < 1e-9);
    }

    #[test]
    fn findings_alike_come_once() {
        let faces: Vec<FaceInfo> = (0..4).map(|i| plane(i, 1, DVec3::new(i as f64 * 10.0, 0.0, 0.0), DVec3::Z, 1.0)).collect();
        let edges: Vec<EdgeInfo> = (0..4).map(|i| edge(i, vec![i], DVec3::new(i as f64 * 10.0, 0.0, 0.0), DVec3::new(i as f64 * 10.0 + 0.05, 0.0, 0.0))).collect();
        let o = AuditOptions { alike: true, ..AUDIT_DEFAULTS };
        let audit = audit_body(&faces, &edges, &o, None, &[]);
        let short: Vec<&Finding> = audit.findings.iter().filter(|f| f.kind == FindingKind::ShortEdge).collect();
        assert_eq!(short.len(), 1);
        assert_eq!(short[0].count, Some(4));
        assert_eq!(short[0].places.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn a_place_is_its_coordinate_when_square_to_an_axis() {
        let f = plane(0, 1, DVec3::new(10.98, 3.0, 4.0), DVec3::X, 1.0).fingerprint;
        assert_eq!(place_of(&f), "x 10.98");
        let g = plane(0, 1, DVec3::new(1.0, 2.0, 3.0), DVec3::new(1.0, 1.0, 0.0).normalize(), 1.0).fingerprint;
        assert_eq!(place_of(&g), "at (1, 2, 3)");
    }
}
