// SPDX-License-Identifier: AGPL-3.0-only
//! Faces and edges picked by a query — every face or edge of a body that answers its terms: a
//! box, a point it runs through, a kind, a size, the feature that made it, the faces either
//! side. Ported from ScanRuler (`plugins/reverse/core/where.ts`), © Stefan Hermann (CNC
//! Kitchen), AGPL-3.0-only; the module is not called `where`, which Rust keeps for itself.
//!
//! A feature given a query keeps it (a stored pick, beside the fingerprints of what it found),
//! and the builder runs it again on the body as it stands before the feature, so a sketch
//! redrawn under a round still rounds the edges the query names — a fifth side's too. The
//! fingerprints are the fallback, for a query that finds nothing on the part as it stands, and
//! for a part saved before. What the commands read off the page — the feature a `madeBy` names,
//! how faces meet along an edge (sharp) — is asked of the caller ([`WhereAsk`]).

use std::collections::HashSet;

use glam::DVec3;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::topo::{angle_between, match_edge, match_face, EdgeFingerprint, EdgeInfo, EdgeKind, FaceFingerprint, FaceInfo, FaceKind, MatchTolerance, Topology};

/// A feature as a query names it: by id or by name.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FeatureRef {
    Id(i64),
    Name(String),
}

/// A box in the part's frame.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Bounds {
    pub min: DVec3,
    pub max: DVec3,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Near {
    pub at: DVec3,
    pub radius: f64,
}

/// Edges whose curve passes within `within` mm (0.01 by default) of the point.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Through {
    pub at: DVec3,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub within: Option<f64>,
}

/// Circle edges, and round faces, whose axis lies on this line within `within` mm (0.05 by
/// default), parallel within a degree.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Coaxial {
    pub at: DVec3,
    pub dir: DVec3,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub within: Option<f64>,
}

/// Which loop of a flat face's boundary an edge query reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LoopSide {
    Outer,
    Inner,
}

/// Edges between a face one feature made and a face another made.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Sides<R> {
    pub made_by: Vec<R>,
}

/// What a query leaves out: another query, or indices.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Exclude {
    Indices(Vec<usize>),
    Query(Box<Where>),
}

/// What faces and edges are asked for, of all a body has.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Where {
    #[serde(rename = "box", default, skip_serializing_if = "Option::is_none")]
    pub bounds: Option<Bounds>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub near: Option<Near>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub through: Option<Through>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub between: Option<Vec<DVec3>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub within_deg: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub made_by: Option<FeatureRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_length: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_length: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_area: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_area: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub radius: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diameter: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size_within: Option<f64>,
    /// The two faces either side of an edge, by their index; for a face query, those faces.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub faces: Option<Vec<usize>>,
    /// What answers all the rest less these: another query, or indices.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exclude: Option<Exclude>,
    /// With `faces: [i]` on an edge query — the edges of face i's outer boundary, or those
    /// round its holes.
    #[serde(rename = "loop", default, skip_serializing_if = "Option::is_none")]
    pub loop_side: Option<LoopSide>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coaxial: Option<Coaxial>,
    /// A pick that may name nothing: it then adds nothing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub optional: Option<bool>,
    /// Only edges whose two faces meet at an angle.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sharp: Option<bool>,
    /// Edges between a face one feature made and a face another made.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sides: Option<Sides<FeatureRef>>,
    /// The same terms wrapped once more — taken as the query itself.
    #[serde(rename = "where", default, skip_serializing_if = "Option::is_none")]
    pub inner: Option<Box<Where>>,
}

/// Faces or edges.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum What {
    Face,
    Edge,
}

impl What {
    fn name(self) -> &'static str {
        match self {
            What::Face => "face",
            What::Edge => "edge",
        }
    }
}

/// What a query asks of where it is run.
pub struct WhereAsk<'a> {
    /// The id of the feature a `madeBy` names — refused where none is.
    pub feature_id: &'a dyn Fn(&FeatureRef) -> Result<i64, String>,
    /// Whether an edge, by its index, runs between faces that meet at an angle — what `sharp`
    /// keeps; `None` where that cannot be read here.
    pub sharp: Option<&'a dyn Fn(usize) -> bool>,
}

/// A pick or a query given wrapped — `{ where: { sides: … } }` — as the query itself, with any
/// terms beside `where` kept; refused where a term stands both in it and beside it.
fn unwrapped_value(pick: Value, what: What) -> Result<Value, String> {
    let Value::Object(mut map) = pick else { return Ok(pick) };
    let Some(inner) = map.remove("where") else { return Ok(Value::Object(map)) };
    let written = if what == What::Edge { "edges: [{ sides: { madeBy: [...] } }]" } else { "faces: [{ madeBy: … }]" };
    let Value::Object(inner) = inner else {
        return Err(format!("A {} pick is the query itself — {written} — and where in it is {inner}, no query.", what.name()));
    };
    let twice: Vec<&String> = map.keys().filter(|k| inner.contains_key(*k)).collect();
    if !twice.is_empty() {
        let names: Vec<&str> = twice.iter().map(|s| s.as_str()).collect();
        return Err(format!("A {} pick is the query itself — {written} — and {} stand{} both in where and beside it: give each once, with no where.", what.name(), names.join(", "), if names.len() == 1 { "s" } else { "" }));
    }
    for (k, v) in inner {
        map.insert(k, v);
    }
    unwrapped_value(Value::Object(map), what)
}

/// [`unwrapped_value`] for a typed query.
pub fn unwrapped(given: &Where, what: What) -> Result<Where, String> {
    if given.inner.is_none() {
        return Ok(given.clone());
    }
    let v = serde_json::to_value(given).map_err(|e| e.to_string())?;
    serde_json::from_value(unwrapped_value(v, what)?).map_err(|e| e.to_string())
}

/// How far off parallel a coaxial pick's axis may lean, radians.
const COAXIAL_ANGLE: f64 = std::f64::consts::PI / 180.0;
/// How far a coaxial pick's axis may stand off the line unless told, mm.
const COAXIAL_MM: f64 = 0.05;
/// How far an edge may pass from a through point unless told, mm.
const THROUGH_MM: f64 = 0.01;

/// How far a point lies from the line through `at` along `dir`.
fn off_line(p: DVec3, at: DVec3, dir: DVec3) -> f64 {
    let l = dir.length();
    let l = if l > 0.0 { l } else { 1.0 };
    (p - at).cross(dir).length() / l
}

fn angle_or_pi(a: DVec3, b: DVec3) -> f64 {
    if !(a.length() > 0.0 && b.length() > 0.0) {
        return std::f64::consts::PI;
    }
    angle_between(a, b)
}

fn parallel(a: DVec3, b: DVec3) -> bool {
    let t = angle_or_pi(a, b);
    t.min(std::f64::consts::PI - t) <= COAXIAL_ANGLE
}

/// Whether a round face's axis lies on the query's line: a cylinder's, cone's or torus's axis
/// along it, a sphere's centre on it.
fn face_coaxial(f: &FaceInfo, c: &Coaxial) -> bool {
    let fp = &f.fingerprint;
    let within = c.within.unwrap_or(COAXIAL_MM);
    let Some(center) = fp.center else { return false };
    if fp.kind == FaceKind::Sphere {
        return off_line(center, c.at, c.dir) <= within;
    }
    if !fp.kind.is_round() {
        return false;
    }
    parallel(fp.dir, c.dir) && off_line(center, c.at, c.dir) <= within
}

/// Whether a circle edge winds round the query's line: square to it, and every point of it its
/// radius off it.
fn edge_coaxial(e: &EdgeInfo, c: &Coaxial) -> bool {
    let fp = &e.fingerprint;
    let Some(radius) = fp.radius else { return false };
    if fp.kind != EdgeKind::Circle || !parallel(fp.dir, c.dir) {
        return false;
    }
    let within = c.within.unwrap_or(COAXIAL_MM);
    let mut points = vec![fp.mid, e.ends[0], e.ends[1]];
    let n = e.polyline.len();
    if n > 2 {
        for k in [0.25, 0.5, 0.75] {
            let i = (k * (n - 1) as f64).floor() as usize;
            points.push(e.polyline[i]);
        }
    }
    points.iter().all(|p| (off_line(*p, c.at, c.dir) - radius).abs() <= within)
}

/// Whether an edge runs round a round face on the query's line — one of `rounds`, the faces
/// [`face_coaxial`] takes, by hash: a bore's rim where it meets a curved face is no circle, but
/// it bounds the bore. Not its seam, nor a straight side of a part round.
fn edge_rounds_coaxial(e: &EdgeInfo, rounds: &HashSet<u32>) -> bool {
    let faces: HashSet<u32> = e.faces.iter().copied().collect();
    e.fingerprint.kind != EdgeKind::Line && faces.len() == 2 && faces.iter().any(|h| rounds.contains(h))
}

/// The edges of a flat face's boundary, as loops: the edges round it joined end to end, the
/// loop whose extent in the face's plane holds the others' the outer one, the rest round its
/// holes. By index.
fn face_loops(t: &Topology, face_index: usize, body: &str) -> Result<(Vec<usize>, Vec<usize>), String> {
    let face = t.faces.get(face_index).ok_or_else(|| format!("{body} has no face {face_index} — it has {}; reverse.body.describe lists them.", t.faces.len()))?;
    let fp = &face.fingerprint;
    if fp.kind != FaceKind::Plane {
        return Err(format!("loop reads a flat face’s boundary — face {face_index} of {body} is a {}.", fp.kind.name()));
    }
    let edges: Vec<usize> = t.edges.iter().enumerate().filter(|(_, e)| e.faces.contains(&face.hash)).map(|(i, _)| i).collect();
    // Joined end to end: ends within a micron are one point.
    let mut parent: Vec<usize> = (0..edges.len()).collect();
    fn root(parent: &mut [usize], k: usize) -> usize {
        let mut k = k;
        while parent[k] != k {
            parent[k] = parent[parent[k]];
            k = parent[k];
        }
        k
    }
    let near = |a: DVec3, b: DVec3| (a - b).length() <= 1e-3;
    for a in 0..edges.len() {
        for b in a + 1..edges.len() {
            let (ea, eb) = (&t.edges[edges[a]].ends, &t.edges[edges[b]].ends);
            if ea.iter().any(|p| eb.iter().any(|q| near(*p, *q))) {
                let (ra, rb) = (root(&mut parent, a), root(&mut parent, b));
                parent[ra] = rb;
            }
        }
    }
    let mut loops: Vec<(usize, Vec<usize>)> = Vec::new();
    for (k, &i) in edges.iter().enumerate() {
        let r = root(&mut parent, k);
        match loops.iter_mut().find(|(x, _)| *x == r) {
            Some((_, l)) => l.push(i),
            None => loops.push((r, vec![i])),
        }
    }
    // Each loop's extent in the face's plane.
    let n = fp.dir;
    let u = if n.x.abs() < 0.9 { DVec3::new(0.0, -n.z, n.y) } else { DVec3::new(-n.z, 0.0, n.x) };
    let ul = u.length();
    let u = u / if ul > 0.0 { ul } else { 1.0 };
    let v = n.cross(u);
    let extent = |l: &[usize]| {
        let (mut u0, mut u1, mut v0, mut v1) = (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
        let mut take = |p: DVec3| {
            let (a, b) = (p.dot(u), p.dot(v));
            u0 = u0.min(a);
            u1 = u1.max(a);
            v0 = v0.min(b);
            v1 = v1.max(b);
        };
        for &i in l {
            let e = &t.edges[i];
            for p in &e.polyline {
                take(*p);
            }
            take(e.ends[0]);
            take(e.ends[1]);
            take(e.fingerprint.mid);
        }
        (u1 - u0) * (v1 - v0)
    };
    let mut ranked: Vec<(f64, Vec<usize>)> = loops.into_iter().map(|(_, l)| (extent(&l), l)).collect();
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
    let outer = ranked.first().map(|r| r.1.clone()).unwrap_or_default();
    let inner = ranked.iter().skip(1).flat_map(|r| r.1.clone()).collect();
    Ok((outer, inner))
}

/// Whether an edge, by its index, runs between a face the one feature made and a face the
/// other made, either way round.
fn sides_made_by(sides: &Sides<FeatureRef>, t: &Topology, ask: &WhereAsk) -> Result<impl Fn(usize) -> bool, String> {
    if sides.made_by.len() != 2 {
        return Err("sides.madeBy names two features: the edges between a face of the one and a face of the other ({ madeBy: [\"Blade\", \"Dish\"] }).".into());
    }
    let a = (ask.feature_id)(&sides.made_by[0])?;
    let b = (ask.feature_id)(&sides.made_by[1])?;
    let origin: Vec<(u32, i64)> = t.faces.iter().map(|f| (f.hash, f.fingerprint.origin)).collect();
    let edges: Vec<Vec<u32>> = t.edges.iter().map(|e| e.faces.clone()).collect();
    Ok(move |i: usize| {
        let mut faces: Vec<u32> = Vec::new();
        for f in edges.get(i).map(|v| v.as_slice()).unwrap_or(&[]) {
            if !faces.contains(f) {
                faces.push(*f);
            }
        }
        if faces.len() > 2 || faces.is_empty() {
            return false;
        }
        let (p, q) = if faces.len() == 1 { (faces[0], faces[0]) } else { (faces[0], faces[1]) };
        let of = |h: u32| origin.iter().find(|(x, _)| *x == h).map(|(_, o)| *o);
        let (x, y) = (of(p), of(q));
        (x == Some(a) && y == Some(b)) || (x == Some(b) && y == Some(a))
    })
}

/// The indices of a body's faces, or edges, a query names — every term held to, its loop read
/// off the face named, less what it excludes.
pub fn where_indices(given: &Where, t: &Topology, body: &str, what: What, ask: &WhereAsk) -> Result<Vec<usize>, String> {
    let w = unwrapped(given, what)?;
    let made_by = match &w.made_by {
        Some(r) => Some((ask.feature_id)(r)?),
        None => None,
    };
    let sides = sides_of(&w, &t.faces, body)?;
    let mut hits: Vec<usize> = Vec::new();
    match what {
        What::Face => {
            if w.sides.is_some() {
                return Err("sides picks edges — those between faces two features made; for faces give madeBy.".into());
            }
            for (i, f) in t.faces.iter().enumerate() {
                if face_passes(f, &w, made_by, sides.as_deref())? {
                    hits.push(i);
                }
            }
        }
        What::Edge => {
            let mut in_loop: Option<HashSet<usize>> = None;
            if let Some(side) = w.loop_side {
                let faces = w.faces.as_deref().unwrap_or(&[]);
                if faces.len() != 1 {
                    return Err("loop reads one face’s boundary — give that face alone in faces ({ faces: [i], loop: \"outer\" }).".into());
                }
                let (outer, inner) = face_loops(t, faces[0], body)?;
                in_loop = Some(if side == LoopSide::Outer { outer } else { inner }.into_iter().collect());
            }
            // Coaxial: the round faces on the line too, whose edges round it are.
            let rounds: Option<HashSet<u32>> = w.coaxial.as_ref().map(|c| t.faces.iter().filter(|f| face_coaxial(f, c)).map(|f| f.hash).collect());
            for (i, e) in t.edges.iter().enumerate() {
                if in_loop.as_ref().is_none_or(|l| l.contains(&i)) && edge_passes(e, &w, made_by, sides.as_deref(), rounds.as_ref()) {
                    hits.push(i);
                }
            }
            if let Some(s) = &w.sides {
                let keep = sides_made_by(s, t, ask)?;
                hits.retain(|i| keep(*i));
            }
            // Sharp: only the edges whose faces meet at an angle.
            if w.sharp == Some(true) {
                let Some(sharp) = ask.sharp else {
                    return Err(format!("sharp reads how the faces meet off {body}’s mesh, and there is none for it here — pick without sharp, by through or near."));
                };
                hits.retain(|i| sharp(*i));
            }
        }
    }
    if let Some(ex) = &w.exclude {
        let gone: HashSet<usize> = match ex {
            Exclude::Indices(v) => v.iter().copied().collect(),
            Exclude::Query(q) => where_indices(q, t, body, what, ask)?.into_iter().collect(),
        };
        hits.retain(|i| !gone.contains(i));
    }
    Ok(hits)
}

fn in_box(p: DVec3, w: &Where) -> bool {
    let boxed = w.bounds.is_none_or(|b| p.cmpge(b.min).all() && p.cmple(b.max).all());
    boxed && w.near.is_none_or(|n| (p - n.at).length() <= n.radius)
}

/// How far a point lies from the polyline drawn for an edge, or from its ends where it has none.
fn off_polyline(e: &EdgeInfo, p: DVec3) -> f64 {
    let q = &e.polyline;
    if q.len() < 2 {
        return e.ends.iter().map(|x| (*x - p).length()).fold(f64::INFINITY, f64::min);
    }
    q.windows(2)
        .map(|w| {
            let d = w[1] - w[0];
            let l2 = d.length_squared();
            let t = if l2 > 0.0 { ((p - w[0]).dot(d) / l2).clamp(0.0, 1.0) } else { 0.0 };
            (w[0] + d * t - p).length()
        })
        .fold(f64::INFINITY, f64::min)
}

/// How far a point lies from an edge: exactly for a line (its two ends) and a circle (its
/// middle, ends, axis and radius give the centre), from the drawn polyline otherwise.
fn off_edge(e: &EdgeInfo, p: DVec3) -> f64 {
    let fp = &e.fingerprint;
    let [a, b] = e.ends;
    if fp.kind == EdgeKind::Line {
        let d = b - a;
        let l2 = d.length_squared();
        let t = if l2 > 0.0 { ((p - a).dot(d) / l2).clamp(0.0, 1.0) } else { 0.0 };
        return (a + d * t - p).length();
    }
    let coarse = off_polyline(e, p);
    let (EdgeKind::Circle, Some(r)) = (fp.kind, fp.radius) else { return coarse };
    let m = fp.mid;
    let ax = fp.dir.try_normalize().unwrap_or(DVec3::Z);
    // The centre: half way to the seam on a whole circle, else a radius from the middle
    // towards the chord's middle.
    let c = if (a - b).length() < 1e-6 {
        (m + a) / 2.0
    } else {
        let k = (a + b) / 2.0 - m;
        let lk = k.length();
        m + k * (r / if lk > 0.0 { lk } else { 1.0 })
    };
    let v = p - c;
    let h = v.dot(ax);
    let rho = (v - ax * h).length();
    let exact = (rho - r).hypot(h);
    // The nearest point of the whole circle must lie on this stretch of it.
    if coarse < exact + 0.1 * r + 0.05 {
        exact
    } else {
        coarse
    }
}

/// Whether a radius answers the query's radius or diameter, within `sizeWithin` — 2 % of it,
/// 0.05 mm at least, unless given.
fn size_passes(radius: Option<f64>, w: &Where) -> bool {
    let Some(want) = w.radius.or(w.diameter.map(|d| d / 2.0)) else { return true };
    let Some(r) = radius else { return false };
    (r - want).abs() <= w.size_within.unwrap_or((0.02 * want).max(0.05))
}

/// The faces a query's `faces` names, by hash — `None` with no such term; refused for an index
/// the body has not got.
pub fn sides_of(w: &Where, faces: &[FaceInfo], body: &str) -> Result<Option<Vec<u32>>, String> {
    let Some(list) = &w.faces else { return Ok(None) };
    list.iter()
        .map(|i| faces.get(*i).map(|f| f.hash).ok_or_else(|| format!("{body} has no face {i} — it has {}; reverse.body.describe lists them.", faces.len())))
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

pub fn face_passes(f: &FaceInfo, w: &Where, made_by: Option<i64>, sides: Option<&[u32]>) -> Result<bool, String> {
    if w.through.is_some() {
        return Err("through picks edges — the edge a point lies on; for a face give near, box or its fingerprint.".into());
    }
    let fp = &f.fingerprint;
    let within = w.within_deg.unwrap_or(10.0).to_radians();
    if sides.is_some_and(|s| !s.contains(&f.hash)) {
        return Ok(false);
    }
    if !in_box(fp.centroid, w) {
        return Ok(false);
    }
    if w.kind.as_deref().is_some_and(|k| k != fp.kind.name()) {
        return Ok(false);
    }
    if made_by.is_some_and(|m| fp.origin != m) {
        return Ok(false);
    }
    if w.min_area.is_some_and(|a| fp.area < a) || w.max_area.is_some_and(|a| fp.area > a) {
        return Ok(false);
    }
    if !size_passes(fp.radius, w) {
        return Ok(false);
    }
    if w.between.as_ref().is_some_and(|b| !b.iter().any(|n| angle_or_pi(fp.dir, *n) <= within)) {
        return Ok(false);
    }
    if w.coaxial.as_ref().is_some_and(|c| !face_coaxial(f, c)) {
        return Ok(false);
    }
    Ok(true)
}

pub fn edge_passes(e: &EdgeInfo, w: &Where, made_by: Option<i64>, sides: Option<&[u32]>, rounds: Option<&HashSet<u32>>) -> bool {
    let fp = &e.fingerprint;
    let within = w.within_deg.unwrap_or(10.0).to_radians();
    // Between the faces named: each of them on one side of it.
    if sides.is_some_and(|s| !s.iter().all(|h| e.faces.contains(h))) {
        return false;
    }
    if !in_box(fp.mid, w) {
        return false;
    }
    if w.through.is_some_and(|t| off_edge(e, t.at) > t.within.unwrap_or(THROUGH_MM)) {
        return false;
    }
    if w.kind.as_deref().is_some_and(|k| k != fp.kind.name()) {
        return false;
    }
    if made_by.is_some_and(|m| fp.origin != m) {
        return false;
    }
    if w.min_length.is_some_and(|l| fp.length < l) || w.max_length.is_some_and(|l| fp.length > l) {
        return false;
    }
    if !size_passes(fp.radius, w) {
        return false;
    }
    if w.between.as_ref().is_some_and(|b| !b.iter().all(|n| fp.faces.iter().any(|f| angle_or_pi(f.dir, *n) <= within))) {
        return false;
    }
    if let Some(c) = &w.coaxial {
        if !edge_coaxial(e, c) && !rounds.is_some_and(|r| edge_rounds_coaxial(e, r)) {
            return false;
        }
    }
    true
}

// ---- sharp, read off the surfaces ----

/// Two faces whose normals agree this closely all along an edge meet tangent there.
fn tangent_cos() -> f64 {
    0.25f64.to_radians().cos()
}

/// Whether an edge runs between faces that meet at an angle, read off the faces' own surfaces at
/// its middle and a quarter of the way from each end. `normals` gives a face's unit normals, by
/// its hash, at the points of it nearest those asked; their sign does not count. An edge with one
/// face, or more than two, is sharp; a seam, one face meeting itself, is not.
pub fn sharp_by_surfaces(e: &EdgeInfo, normals: &dyn Fn(u32, &[DVec3]) -> Vec<Option<DVec3>>) -> bool {
    let mut faces: Vec<u32> = Vec::new();
    for f in &e.faces {
        if !faces.contains(f) {
            faces.push(*f);
        }
    }
    if e.faces.len() < 2 || faces.len() > 2 {
        return true;
    }
    if faces.len() == 1 {
        return false;
    }
    let mut points = vec![e.fingerprint.mid];
    let n = e.polyline.len();
    if n > 2 {
        for k in [0.25, 0.75] {
            let i = (k * (n - 1) as f64).round() as usize;
            points.push(e.polyline[i]);
        }
    }
    let a = normals(faces[0], &points);
    let b = normals(faces[1], &points);
    (0..points.len()).any(|i| match (a.get(i).copied().flatten(), b.get(i).copied().flatten()) {
        (Some(x), Some(y)) => x.dot(y).abs() < tangent_cos(),
        _ => true,
    })
}

// ---- queries a feature keeps ----

/// A fingerprint a kept query names by: a face's or an edge's.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Print {
    Face(FaceFingerprint),
    Edge(EdgeFingerprint),
}

/// What a kept query leaves out: another query, or what it named by index then, by fingerprint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StoredExclude {
    Prints { prints: Vec<Print> },
    Query(Box<StoredWhere>),
}

/// A query as a feature keeps it, to be run again at build: `madeBy` and `sides` by feature
/// id, and what it named by index — the faces either side, the face whose loop it reads, what it
/// excludes — by their fingerprints: an index names a face of the part as it stood when the
/// query was given, and the next build may number them otherwise.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StoredWhere {
    #[serde(rename = "box", default, skip_serializing_if = "Option::is_none")]
    pub bounds: Option<Bounds>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub near: Option<Near>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub through: Option<Through>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub between: Option<Vec<DVec3>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub within_deg: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub made_by: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_length: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_length: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_area: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_area: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub radius: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diameter: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size_within: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub faces: Option<Vec<FaceFingerprint>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exclude: Option<StoredExclude>,
    #[serde(rename = "loop", default, skip_serializing_if = "Option::is_none")]
    pub loop_side: Option<LoopSide>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coaxial: Option<Coaxial>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub optional: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sharp: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sides: Option<Sides<i64>>,
}

/// A query a feature keeps, and what it found when it was given — the fingerprints the
/// feature's own list holds for it too. While all of those are still in that list the query
/// stands; an edit by hand that took one of them out lets it go, and the list is the pick from
/// then on.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StoredPick<F> {
    #[serde(rename = "where")]
    pub query: StoredWhere,
    pub found: Vec<F>,
}

/// A query made ready to keep, on the topology it was run on: features by id, indices by
/// fingerprint.
pub fn stored_where(given: &Where, t: &Topology, what: What, feature_id: &dyn Fn(&FeatureRef) -> Result<i64, String>) -> Result<StoredWhere, String> {
    let w = unwrapped(given, what)?;
    let face = |i: usize| t.faces.get(i).map(|f| f.fingerprint).ok_or_else(|| format!("There is no face {i} — the body has {}; reverse.body.describe lists them.", t.faces.len()));
    let made_by = match &w.made_by {
        Some(r) => Some(feature_id(r)?),
        None => None,
    };
    let sides = match &w.sides {
        Some(s) => Some(Sides { made_by: s.made_by.iter().map(feature_id).collect::<Result<Vec<_>, _>>()? }),
        None => None,
    };
    let faces = match &w.faces {
        Some(list) => Some(list.iter().map(|i| face(*i)).collect::<Result<Vec<_>, _>>()?),
        None => None,
    };
    let exclude = match &w.exclude {
        Some(Exclude::Indices(list)) => Some(StoredExclude::Prints {
            prints: list
                .iter()
                .filter_map(|i| match what {
                    What::Face => t.faces.get(*i).map(|f| Print::Face(f.fingerprint)),
                    What::Edge => t.edges.get(*i).map(|e| Print::Edge(e.fingerprint.clone())),
                })
                .collect(),
        }),
        Some(Exclude::Query(q)) => Some(StoredExclude::Query(Box::new(stored_where(q, t, what, feature_id)?))),
        None => None,
    };
    Ok(StoredWhere {
        bounds: w.bounds,
        near: w.near,
        through: w.through,
        kind: w.kind.clone(),
        between: w.between.clone(),
        within_deg: w.within_deg,
        made_by,
        min_length: w.min_length,
        max_length: w.max_length,
        min_area: w.min_area,
        max_area: w.max_area,
        radius: w.radius,
        diameter: w.diameter,
        size_within: w.size_within,
        faces,
        exclude,
        loop_side: w.loop_side,
        coaxial: w.coaxial,
        optional: w.optional,
        sharp: w.sharp,
        sides,
    })
}

/// A kept query as it is run on a topology: its fingerprints found there again, by index —
/// `None` where a face it names by one is not found. An exclusion not found is gone, and
/// excludes nothing.
fn where_on(s: &StoredWhere, t: &Topology, what: What, tol: &MatchTolerance) -> Option<Where> {
    let mut out = Where {
        bounds: s.bounds,
        near: s.near,
        through: s.through,
        kind: s.kind.clone(),
        between: s.between.clone(),
        within_deg: s.within_deg,
        made_by: s.made_by.map(FeatureRef::Id),
        min_length: s.min_length,
        max_length: s.max_length,
        min_area: s.min_area,
        max_area: s.max_area,
        radius: s.radius,
        diameter: s.diameter,
        size_within: s.size_within,
        faces: None,
        exclude: None,
        loop_side: s.loop_side,
        coaxial: s.coaxial,
        optional: s.optional,
        sharp: s.sharp,
        sides: s.sides.as_ref().map(|x| Sides { made_by: x.made_by.iter().map(|i| FeatureRef::Id(*i)).collect() }),
        inner: None,
    };
    let face_prints = t.face_prints();
    if let Some(faces) = &s.faces {
        let at: Option<Vec<usize>> = faces.iter().map(|f| match_face(f, &face_prints, tol)).collect();
        out.faces = Some(at?);
    }
    match &s.exclude {
        Some(StoredExclude::Prints { prints }) => {
            let edge_prints = t.edge_prints();
            out.exclude = Some(Exclude::Indices(
                prints
                    .iter()
                    .filter_map(|p| match (what, p) {
                        (What::Face, Print::Face(f)) => match_face(f, &face_prints, tol),
                        (What::Edge, Print::Edge(e)) => match_edge(e, &edge_prints, tol),
                        _ => None,
                    })
                    .collect(),
            ));
        }
        Some(StoredExclude::Query(q)) => out.exclude = Some(Exclude::Query(Box::new(where_on(q, t, what, tol)?))),
        None => {}
    }
    Some(out)
}

/// A kept query run on a body as it stands, by index — `None` where it cannot be read there (a
/// face it names is gone, a term refused).
pub fn run_stored_where(s: &StoredWhere, t: &Topology, body: &str, what: What, sharp: Option<&dyn Fn(usize) -> bool>, tol: &MatchTolerance) -> Option<Vec<usize>> {
    let w = where_on(s, t, what, tol)?;
    let feature_id = |r: &FeatureRef| match r {
        FeatureRef::Id(i) => Ok(*i),
        FeatureRef::Name(_) => Ok(i64::MIN),
    };
    where_indices(&w, t, body, what, &WhereAsk { feature_id: &feature_id, sharp }).ok()
}

/// Two fingerprints the same, by value — as a file holds them.
fn same_print<F: Serialize>(a: &F, b: &F) -> bool {
    match (serde_json::to_string(a), serde_json::to_string(b)) {
        (Ok(x), Ok(y)) => x == y,
        _ => false,
    }
}

/// The kept queries of a feature that still stand on its list of fingerprints — each whose finds
/// are all in it still, each on a fingerprint of its own — and the fingerprints left over, picked
/// one by one. `same` says a find is still in the list: by value, or — for the build — as near
/// as the fingerprints match.
pub fn live_picks<F: Serialize + Clone>(prints: &[F], picks: Option<&[StoredPick<F>]>, same: Option<&dyn Fn(&F, &F) -> bool>) -> (Vec<StoredPick<F>>, Vec<F>) {
    let Some(picks) = picks.filter(|p| !p.is_empty()) else { return (Vec::new(), prints.to_vec()) };
    let mut claimed: HashSet<usize> = HashSet::new();
    let mut live = Vec::new();
    for p in picks {
        let mut mine: Vec<usize> = Vec::new();
        for f in &p.found {
            let open = |i: usize| !claimed.contains(&i) && !mine.contains(&i);
            let exact = prints.iter().enumerate().position(|(i, x)| open(i) && same_print(f, x));
            let i = exact.or_else(|| same.and_then(|s| prints.iter().enumerate().position(|(i, x)| open(i) && s(f, x))));
            let Some(i) = i else { break };
            mine.push(i);
        }
        if mine.len() < p.found.len() {
            continue;
        }
        live.push(p.clone());
        claimed.extend(mine);
    }
    let rest = prints.iter().enumerate().filter(|(i, _)| !claimed.contains(i)).map(|(_, f)| f.clone()).collect();
    (live, rest)
}

/// A kept query in words, for a row that says it found nothing: its terms, what it named by
/// fingerprint counted.
pub fn where_words(s: &StoredWhere) -> String {
    let Ok(Value::Object(map)) = serde_json::to_value(s) else { return "{ }".into() };
    let terms: Vec<String> = map
        .iter()
        .map(|(k, v)| match (k.as_str(), v) {
            ("faces", Value::Array(a)) => format!("faces: {}", a.len()),
            ("exclude", _) => "exclude".to_string(),
            _ => format!("{k}: {v}"),
        })
        .collect();
    format!("{{ {} }}", terms.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topo::{match_tolerance, EdgeSide};

    /// A box's faces and edges as a topology: a top, a side, and the edge between them — in an
    /// order of our choosing.
    fn topology(ab: bool) -> Topology {
        let top = FaceInfo { hash: 11, point: DVec3::new(0.0, 0.0, 4.0), fingerprint: FaceFingerprint { kind: FaceKind::Plane, centroid: DVec3::new(0.0, 0.0, 4.0), dir: DVec3::Z, area: 200.0, radius: None, center: None, origin: 2 }, full_turn: None };
        let side = FaceInfo { hash: 12, point: DVec3::new(0.0, -5.0, 2.0), fingerprint: FaceFingerprint { kind: FaceKind::Plane, centroid: DVec3::new(0.0, -5.0, 2.0), dir: DVec3::new(0.0, -1.0, 0.0), area: 80.0, radius: None, center: None, origin: 2 }, full_turn: None };
        let edge = EdgeInfo {
            hash: 21,
            fingerprint: EdgeFingerprint {
                kind: EdgeKind::Line,
                mid: DVec3::new(0.0, -5.0, 4.0),
                dir: DVec3::X,
                length: 20.0,
                radius: None,
                faces: vec![EdgeSide { kind: FaceKind::Plane, dir: DVec3::Z }, EdgeSide { kind: FaceKind::Plane, dir: DVec3::new(0.0, -1.0, 0.0) }],
                origin: 2,
            },
            faces: vec![11, 12],
            ends: [DVec3::new(-10.0, -5.0, 4.0), DVec3::new(10.0, -5.0, 4.0)],
            end_tangents: [DVec3::X, DVec3::X],
            polyline: vec![DVec3::new(-10.0, -5.0, 4.0), DVec3::new(0.0, -5.0, 4.0), DVec3::new(10.0, -5.0, 4.0)],
        };
        Topology { faces: if ab { vec![top, side] } else { vec![side, top] }, edges: vec![edge] }
    }

    #[test]
    fn name_faces_by_fingerprint_and_features_by_id_and_run_again_on_faces_numbered_otherwise() {
        let t = topology(true);
        let given = Where { faces: Some(vec![0, 1]), made_by: Some(FeatureRef::Name("Prism".into())), ..Default::default() };
        let stored = stored_where(&given, &t, What::Edge, &|_| Ok(2)).unwrap();
        assert_eq!(stored.made_by, Some(2));
        assert_eq!(stored.faces, Some(vec![t.faces[0].fingerprint, t.faces[1].fingerprint]));
        let tol = match_tolerance(25.0);
        assert_eq!(run_stored_where(&stored, &topology(false), "Prism", What::Edge, None, &tol), Some(vec![0]));
        // A face it names gone: the query cannot be read there.
        let mut gone = topology(true);
        gone.faces = vec![gone.faces[1].clone()];
        assert_eq!(run_stored_where(&stored, &gone, "Prism", What::Edge, None, &tol), None);
        // Sharp asks how the faces meet, of whoever runs it.
        let sharp = StoredWhere { sharp: Some(true), ..Default::default() };
        let never = |_: usize| false;
        assert_eq!(run_stored_where(&sharp, &t, "Prism", What::Edge, Some(&never), &tol), Some(vec![]));
        assert_eq!(run_stored_where(&sharp, &t, "Prism", What::Edge, None, &tol), None);
    }

    #[test]
    fn stand_while_what_they_found_is_picked() {
        #[derive(Debug, Clone, PartialEq, Serialize)]
        struct P {
            kind: &'static str,
            mid: [f64; 3],
        }
        let p = |x: f64| P { kind: "line", mid: [x, 0.0, 0.0] };
        let (a, b, c) = (p(0.0), p(1.0), p(2.0));
        let picks = vec![StoredPick { query: StoredWhere { bounds: Some(Bounds { min: DVec3::ZERO, max: DVec3::ONE }), ..Default::default() }, found: vec![a.clone(), b.clone()] }];
        assert_eq!(live_picks(&[a.clone(), b.clone(), c.clone()], Some(&picks), None), (picks.clone(), vec![c.clone()]));
        // b taken out by hand: the query goes, a and c are picked as they are.
        assert_eq!(live_picks(&[a.clone(), c.clone()], Some(&picks), None), (vec![], vec![a.clone(), c.clone()]));
        // The panel puts a reopened pick on the edge it finds now: near enough stands for it,
        // each find on a fingerprint of its own.
        let near = |f: &P, q: &P| (f.mid[0] - q.mid[0]).abs() < 0.3;
        let moved = p(1.2);
        assert_eq!(live_picks(&[a.clone(), moved, c.clone()], Some(&picks), Some(&near)), (picks.clone(), vec![c.clone()]));
        assert!(live_picks(&[a, c], Some(&picks), Some(&near)).0.is_empty());
    }

    #[test]
    fn read_sharp_off_the_faces_surfaces_with_no_mesh() {
        let e = topology(true).edges[0].clone();
        let flat = |n: DVec3| vec![Some(n); 3];
        assert!(sharp_by_surfaces(&e, &|face, _| if face == 11 { flat(DVec3::Z) } else { flat(DVec3::new(0.0, -1.0, 0.0)) }));
        // Tangent, a reversed face's normal unturned: not sharp.
        assert!(!sharp_by_surfaces(&e, &|face, _| if face == 11 { flat(DVec3::Z) } else { flat(DVec3::NEG_Z) }));
        // A seam, one face meeting itself: not sharp; an open edge: sharp.
        assert!(!sharp_by_surfaces(&EdgeInfo { faces: vec![11, 11], ..e.clone() }, &|_, _| flat(DVec3::Z)));
        assert!(sharp_by_surfaces(&EdgeInfo { faces: vec![11], ..e }, &|_, _| flat(DVec3::Z)));
    }

    #[test]
    fn a_query_wrapped_in_where_is_the_query_and_a_term_given_twice_is_refused() {
        let t = topology(true);
        let ask = WhereAsk { feature_id: &|_| Ok(2), sharp: None };
        let wrapped = Where { inner: Some(Box::new(Where { kind: Some("line".into()), ..Default::default() })), ..Default::default() };
        assert_eq!(where_indices(&wrapped, &t, "Prism", What::Edge, &ask), Ok(vec![0]));
        let twice = Where { kind: Some("line".into()), inner: Some(Box::new(Where { kind: Some("circle".into()), ..Default::default() })), ..Default::default() };
        assert!(where_indices(&twice, &t, "Prism", What::Edge, &ask).is_err_and(|e| e.contains("kind stands both in where and beside it")));
        // Faces by a direction, and an edge through a point on it.
        let up = Where { between: Some(vec![DVec3::Z]), ..Default::default() };
        assert_eq!(where_indices(&up, &t, "Prism", What::Face, &ask), Ok(vec![0]));
        let through = Where { through: Some(Through { at: DVec3::new(3.0, -5.0, 4.0), within: None }), ..Default::default() };
        assert_eq!(where_indices(&through, &t, "Prism", What::Edge, &ask), Ok(vec![0]));
        assert!(where_indices(&through, &t, "Prism", What::Face, &ask).is_err());
        let json = r#"{"where":{"box":{"min":[-1,-6,3],"max":[1,-4,5]}},"exclude":[0]}"#;
        let w: Where = serde_json::from_str(json).unwrap();
        assert_eq!(where_indices(&w, &t, "Prism", What::Edge, &ask), Ok(vec![]));
    }

    #[test]
    fn a_stored_pick_goes_to_json_under_where() {
        let pick = StoredPick { query: StoredWhere { made_by: Some(3), kind: Some("line".into()), ..Default::default() }, found: vec![topology(true).edges[0].fingerprint.clone()] };
        let text = serde_json::to_string(&pick).unwrap();
        assert!(text.starts_with(r#"{"where":{"kind":"line","madeBy":3}"#), "{text}");
        let back: StoredPick<EdgeFingerprint> = serde_json::from_str(&text).unwrap();
        assert_eq!(back, pick);
        assert_eq!(where_words(&pick.query), r#"{ kind: "line", madeBy: 3 }"#);
    }
}
