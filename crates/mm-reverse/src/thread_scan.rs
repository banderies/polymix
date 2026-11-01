// SPDX-License-Identifier: AGPL-3.0-only
//! A thread read off the scan along an axis — a cylinder element's, a round face's of a body
//! (along its coaxial run), or one given outright — and set against the standard sizes: what
//! the `reverse.thread.read` command answers, and the Thread box offers. Ported from
//! ScanRuler (`plugins/reverse/app/reThreadCommand.ts`, with `src/core/geometry/query.ts`
//! queryVertices), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only. The reading itself is
//! [`crate::thread_read`]'s.
//!
//! Here too the round faces a body's round is made of: the kernel keeps a whole bore as two
//! halves, so a round face's length along its axis is read off the faces of its cylinder joined
//! to it, where ScanRuler reads it off the one face. And a thread's size as it is typed — "M10x1",
//! "1/4-20 UNC", "Ø16×3.5 print" — read into a size of the tables and a pitch (ScanRuler's
//! `plugins/reverse/app/reFeatureInput.ts` readDesignation and threadSize).

use std::f64::consts::PI;

use glam::DVec3;
use serde::Serialize;
use serde_json::{json, Map, Value};

use crate::model::{BodyId, Element, ElementGeometry, FaceRef};
use crate::pick_query::{face_passes, Coaxial, Where};
use crate::round::{js_round, number_text, round3, round4};
use crate::thread::{thread_sizes, ThreadStandard};
use crate::thread_read::{basic_depth, is_thread, match_thread, read_thread, ThreadCandidate, ThreadReadInput, ThreadReading, ALL_STANDARDS};
use crate::topo::{on_one_surface, FaceKind, Topology};

/// How many scan points a thread is read from at most.
pub const THREAD_POINTS: usize = 20_000;

/// The band about the radius a point counts in: this many mm, or this share of the radius when
/// that is more.
pub const THREAD_BAND_MM: f64 = 0.6;
pub const THREAD_BAND_SHARE: f64 = 0.2;

/// How sure a left hand must read for the feature to take it: right is every thread but a few,
/// and an axis a little aslant blurs the hand.
pub const LEFT_SURE: f64 = 0.2;

/// The band a reading of a round of radius `radius` takes when none is asked.
pub fn default_band(radius: f64) -> f64 {
    THREAD_BAND_MM.max(THREAD_BAND_SHARE * radius)
}

/// The faces of one round: the face and every face of its cylinder joined to it across an edge
/// (the two halves of a bore).
pub fn round_faces(t: &Topology, face: usize) -> Vec<usize> {
    let mut group = vec![face];
    let mut i = 0;
    while i < group.len() {
        let at = group[i] as u32;
        i += 1;
        let Some(a) = t.faces.get(at as usize).map(|f| f.fingerprint) else { continue };
        for e in t.edges.iter().filter(|e| e.faces.contains(&at)) {
            for h in &e.faces {
                let h = *h as usize;
                if group.contains(&h) {
                    continue;
                }
                if t.faces.get(h).is_some_and(|b| b.fingerprint.kind == FaceKind::Cylinder && on_one_surface(&a, &b.fingerprint)) {
                    group.push(h);
                }
            }
        }
    }
    group
}

/// How long a round face runs along its axis, read off its area as a full cylinder's — its
/// own and that of the faces of its cylinder joined to it — to a micrometre: what a thread
/// that does not run the whole face starts at (ScanRuler's `roundFaceLength`). 10 where the
/// face has no radius.
pub fn round_length(t: &Topology, face: usize) -> f64 {
    let Some(r) = t.faces.get(face).and_then(|f| f.fingerprint.radius).filter(|r| *r > 0.0) else { return 10.0 };
    let area: f64 = round_faces(t, face).iter().filter_map(|i| t.faces.get(*i)).map(|f| f.fingerprint.area).sum();
    round3(area / (2.0 * PI * r))
}

/// A box of the scan and how many of its points are read at most.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VertexQuery {
    pub min: DVec3,
    pub max: DVec3,
    pub limit: usize,
}

/// The points a query names: past the limit every k-th of those that match is kept, spread
/// evenly through the box.
#[derive(Debug, Clone, PartialEq)]
pub struct VertexQueryResult {
    pub positions: Vec<DVec3>,
    pub normals: Vec<DVec3>,
    /// How many matched before the limit thinned them.
    pub matched: usize,
    /// Every k-th matching point was kept — 1 when all were.
    pub step: usize,
}

/// The scan's points inside a box (ScanRuler's `queryVertices`), from clouds of points and
/// their normals — each object of the scan its own, read in turn as one.
pub fn query_vertices(clouds: &[(&[DVec3], &[DVec3])], q: VertexQuery) -> VertexQueryResult {
    let mut hits: Vec<(usize, usize)> = Vec::new();
    for (c, (points, _)) in clouds.iter().enumerate() {
        for (v, p) in points.iter().enumerate() {
            if p.x < q.min.x || p.x > q.max.x || p.y < q.min.y || p.y > q.max.y || p.z < q.min.z || p.z > q.max.z {
                continue;
            }
            hits.push((c, v));
        }
    }
    let matched = hits.len();
    let limit = q.limit.max(1);
    let step = if matched <= limit { 1 } else { matched.div_ceil(limit) };
    let kept: Vec<(usize, usize)> = hits.into_iter().step_by(step).collect();
    let positions = kept.iter().map(|(c, v)| clouds[*c].0[*v]).collect();
    let normals = kept.iter().map(|(c, v)| clouds[*c].1.get(*v).copied().unwrap_or(DVec3::ZERO)).collect();
    VertexQueryResult { positions, normals, matched, step }
}

/// What a thread is read along.
#[derive(Debug, Clone, Copy)]
pub enum ReadAlong<'a> {
    /// A cylinder fitted in the bore or on the boss.
    Element(&'a Element),
    /// A round face of a body — the bore or the boss as modelled — read along its coaxial run.
    Face { body: BodyId, body_name: &'a str, topology: &'a Topology, face: usize },
    /// An axis given outright: a point on it — the middle of the length read — its direction,
    /// the nominal radius, and the length along it.
    Line { origin: DVec3, dir: DVec3, radius: f64, length: f64 },
}

/// What is asked of the reading besides the axis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReadAsk {
    /// The standard to match: `None` for any.
    pub standard: Option<ThreadStandard>,
    /// Millimetres either side of the radius a point still counts; by default
    /// [`default_band`].
    pub band: Option<f64>,
    /// Millimetres along the axis to read, centred; the element's length or the face's coaxial
    /// run by default.
    pub length: Option<f64>,
    pub limit: usize,
    /// Add the radius along the axis.
    pub profile: bool,
}

impl Default for ReadAsk {
    fn default() -> Self {
        ReadAsk { standard: Some(ThreadStandard::Metric), band: None, length: None, limit: THREAD_POINTS, profile: false }
    }
}

/// Why a thread could not be read: the command's error code and its sentence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadRefusal {
    /// `no_scan`, `not_found`, `invalid` or `failed`.
    pub code: &'static str,
    pub message: String,
}

fn refuse(code: &'static str, message: impl Into<String>) -> ReadRefusal {
    ReadRefusal { code, message: message.into() }
}

/// The thread a reading suggests for a round face of a body: the Thread feature ready to add.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ThreadSuggestion {
    pub standard: ThreadStandard,
    pub size: String,
    pub diameter: f64,
    pub pitch: f64,
    pub left_handed: bool,
}

/// A thread read off the scan, as the command answers it.
#[derive(Debug, Clone, PartialEq)]
pub struct ThreadReadAnswer {
    /// The whole answer, in the command's shape.
    pub json: Value,
    pub reading: ThreadReading,
    /// Whether it is a thread: a pitch found, and teeth standing clear of the scan's noise.
    pub thread: bool,
    pub candidates: Vec<ThreadCandidate>,
    pub suggested: Option<ThreadCandidate>,
    /// The feature ready to add, where the axis was a face of a body.
    pub feature: Option<ThreadSuggestion>,
    /// The note the answer ends on.
    pub note: String,
}

/// The faces a round face runs on into along its axis, each reached from it across an edge:
/// the round faces on the same axis — a hole's wall in pieces, its countersink — and those
/// whose edges lie all within `band` of the radius, as a modelled thread's flanks do. Where
/// they reach along the axis (`dir`, a unit vector), mm from `at`, read off their edges.
pub fn coaxial_run(t: &Topology, face: usize, at: DVec3, dir: DVec3, radius: f64, band: f64) -> (Vec<usize>, f64, f64) {
    let edges_of = |h: u32| t.edges.iter().filter(move |e| e.faces.contains(&h));
    let along = |q: DVec3| {
        let d = q - at;
        let z = d.dot(dir);
        (z, (d - dir * z).length())
    };
    let within = |h: u32| edges_of(h).all(|e| e.polyline.iter().all(|q| (along(*q).1 - radius).abs() <= band));
    let coaxial = Where { coaxial: Some(Coaxial { at, dir, within: None }), ..Default::default() };
    let Some(start) = t.faces.get(face).map(|f| f.hash) else { return (Vec::new(), f64::INFINITY, f64::NEG_INFINITY) };
    let mut run: Vec<u32> = vec![start];
    let mut queue = vec![start];
    while let Some(h) = queue.pop() {
        for e in edges_of(h) {
            for g in &e.faces {
                let Some(f) = t.face(*g) else { continue };
                if run.contains(g) || !(face_passes(f, &coaxial, None, None).unwrap_or(false) || within(*g)) {
                    continue;
                }
                run.push(*g);
                queue.push(*g);
            }
        }
    }
    let (mut from, mut to) = (f64::INFINITY, f64::NEG_INFINITY);
    for h in &run {
        for e in edges_of(*h) {
            for q in &e.polyline {
                let z = along(*q).0;
                from = from.min(z);
                to = to.max(z);
            }
        }
    }
    let faces = t.faces.iter().enumerate().filter(|(_, f)| run.contains(&f.hash)).map(|(i, _)| i).collect();
    (faces, from, to)
}

/// A number as JavaScript writes it.
fn js(v: f64) -> String {
    number_text(v)
}

/// Which windows read the pitch, for the note: none to say with one.
fn agree_in(r: &ThreadReading) -> String {
    match r.threaded {
        Some(t) if r.windows.len() >= 2 => {
            if r.agree > 0 {
                format!("; {} of {} windows agree, from {} to {} mm along the axis", r.agree, r.windows.len(), js(t[0]), js(t[1]))
            } else {
                "; over the whole length, too coarse for a window".to_string()
            }
        }
        _ => String::new(),
    }
}

fn v3(p: DVec3) -> Value {
    json!([round4(p.x), round4(p.y), round4(p.z)])
}

/// A thread read off the scan along an axis and set against the standard sizes (ScanRuler's
/// `reverse.thread.read`): the scan's points in a thin shell about the axis read for their
/// radius along it, and in that profile the pitch, the crests and the root, the hand and
/// whether it is a bore; the nearest sizes, the one near enough to suggest, and — along a face
/// of a body — the Thread feature ready to add. `clouds` is the scan, each object's points and
/// their normals in the part's coordinates.
pub fn thread_read(along: ReadAlong<'_>, clouds: &[(&[DVec3], &[DVec3])], ask: &ReadAsk) -> Result<ThreadReadAnswer, ReadRefusal> {
    if clouds.iter().all(|(p, _)| p.is_empty()) {
        return Err(refuse("no_scan", "There is no scan to read a thread off."));
    }
    let mut on = Map::new();
    let (origin, dir, radius, span) = match along {
        ReadAlong::Element(e) => {
            let ElementGeometry::Cylinder { center, axis, radius, length, .. } = e.geometry else {
                return Err(refuse("invalid", format!("{} is a {} — a thread is read along a cylinder fitted in the bore or on the boss.", e.name, e.geometry.kind())));
            };
            on.insert("element".into(), json!({ "id": e.id, "name": e.name }));
            (center, axis, radius, length)
        }
        ReadAlong::Face { body, body_name, topology, face } => {
            let Some(f) = topology.faces.get(face) else { return Err(refuse("not_found", format!("{body_name} has no face {face}."))) };
            let fp = f.fingerprint;
            let (FaceKind::Cylinder, Some(center), Some(r)) = (fp.kind, fp.center, fp.radius.filter(|r| *r > 0.0)) else {
                return Err(refuse("invalid", format!("That face of {body_name} is a {} — a thread is read along a round face.", fp.kind.name())));
            };
            let d = fp.dir.try_normalize().ok_or_else(|| refuse("invalid", "The axis has no direction."))?;
            // Along the face's coaxial run, not the one face: a hole cut in pieces, its
            // countersink and a modelled thread's flanks are one wall to the scan, and a piece a
            // millimetre long holds no pitch.
            let (faces, from, to) = coaxial_run(topology, face, center, d, r, ask.band.unwrap_or_else(|| default_band(r)));
            let span = to - from;
            on.insert("body".into(), json!({ "id": body, "name": body_name }));
            on.insert("face".into(), json!({ "index": face, "kind": fp.kind.name(), "diameter": round4(2.0 * r), "length": round_length(topology, face) }));
            if faces.len() > 1 {
                on.insert("run".into(), json!({ "faces": faces, "length": round3(span) }));
            }
            (center + d * ((from + to) / 2.0), d, r, span)
        }
        ReadAlong::Line { origin, dir, radius, length } => (origin, dir, radius, length),
    };
    let d = dir.try_normalize().ok_or_else(|| refuse("invalid", "The axis has no direction."))?;
    let half_length = ask.length.unwrap_or(span) / 2.0;
    let used = ask.band.unwrap_or_else(|| default_band(radius));
    let reach = radius + used;
    let extent = d.abs() * half_length + DVec3::splat(reach);
    let q = query_vertices(clouds, VertexQuery { min: origin - extent, max: origin + extent, limit: ask.limit.clamp(100, THREAD_POINTS) });
    let reading = read_thread(&ThreadReadInput { points: &q.positions, normals: &q.normals, origin, dir: d, radius, band: used, half_length }).ok_or_else(|| {
        refuse(
            "failed",
            format!("Too few scan points lie within {} mm of radius {} along the axis ({} in the box about it) — check the axis and the radius, or widen band.", js(round3(used)), js(round3(radius)), q.matched),
        )
    })?;
    let standards: Vec<ThreadStandard> = match ask.standard {
        Some(s) => vec![s],
        None => ALL_STANDARDS.to_vec(),
    };
    let thread = is_thread(&reading);
    // A repeat that stands within the scan's noise suggests no size.
    let matched = if thread { match_thread(&reading, &standards) } else { crate::thread_read::ThreadMatch { candidates: Vec::new(), suggested: None } };
    let feature = match (&matched.suggested, along) {
        (Some(s), ReadAlong::Face { .. }) => Some(ThreadSuggestion {
            standard: s.standard,
            size: s.size.clone(),
            diameter: s.diameter,
            pitch: s.pitch,
            left_handed: reading.left_handed == Some(true) && reading.hand_confidence >= LEFT_SURE,
        }),
        _ => None,
    };
    let what = match reading.internal {
        Some(true) => "bore",
        Some(false) => "boss",
        None => "round",
    };
    let note = if thread {
        let near = match &matched.suggested {
            Some(s) => format!("nearest {} (pitch off {} %, diameter off {} %)", s.designation, js(js_round(s.pitch_off * 100.0)), js(js_round(s.diameter_off * 100.0))),
            None => "no standard size near enough to suggest — see candidates".to_string(),
        };
        format!(
            "The profile repeats every {} mm (strength {}{}), teeth {} mm against the scan's noise of {} mm; {near}. A scan reads a thread shallower than the basic profile cuts it.",
            js(reading.pitch.unwrap_or(f64::NAN)),
            js(reading.strength),
            agree_in(&reading),
            js(reading.contrast.unwrap_or(f64::NAN)),
            js(reading.noise)
        )
    } else if let Some(p) = reading.pitch {
        format!(
            "The radius repeats every {} mm{}, but by {} mm — within the scan's noise of {} mm (a thread's teeth stand at least three times it): the scan's own ripple on a plain {what}, not a thread.",
            js(p),
            agree_in(&reading),
            reading.contrast.map_or("null".to_string(), js),
            js(reading.noise)
        )
    } else {
        format!("The radius does not repeat along the axis (strength {}) — a plain {what}, or a thread finer than the scan resolves.", js(reading.strength))
    };
    // The answer, in the command's shape.
    let mut out = Map::new();
    out.insert("axis".into(), json!({ "origin": v3(origin), "dir": v3(d), "radius": round4(radius), "length": round4(2.0 * half_length) }));
    out.extend(on);
    out.insert("band".into(), json!(round3(used)));
    out.insert("inBox".into(), json!(q.matched));
    out.insert("thread".into(), json!(thread));
    if let Ok(Value::Object(read)) = serde_json::to_value(&reading) {
        for (k, v) in read {
            if k != "profile" {
                out.insert(k, v);
            }
        }
    }
    // The depth of the suggested size's form; with none, of the form asked for — the 60° one
    // for any.
    if let Some(p) = reading.pitch {
        let form = matched.suggested.as_ref().map(|s| s.standard).or(ask.standard).unwrap_or(ThreadStandard::Metric);
        out.insert("basicDepth".into(), json!(basic_depth(p, form)));
    }
    out.insert("standards".into(), json!(standards.iter().map(|s| s.id()).collect::<Vec<_>>()));
    out.insert("candidates".into(), serde_json::to_value(&matched.candidates).unwrap_or(Value::Null));
    out.insert("suggested".into(), serde_json::to_value(&matched.suggested).unwrap_or(Value::Null));
    if let (Some(f), ReadAlong::Face { body_name, topology, face, .. }) = (&feature, along) {
        let mut x = Map::new();
        x.insert("kind".into(), json!("thread"));
        x.insert("body".into(), json!(body_name));
        x.insert("face".into(), serde_json::to_value(topology.faces.get(face).map(|f| f.fingerprint)).unwrap_or(Value::Null));
        x.insert("standard".into(), json!(f.standard.id()));
        x.insert("size".into(), json!(f.size));
        if thread_sizes(f.standard).iter().find(|z| z.label == f.size).and_then(|z| z.pitches.first()) != Some(&f.pitch) {
            x.insert("pitch".into(), json!(f.pitch));
        }
        if f.left_handed {
            x.insert("leftHanded".into(), json!(true));
        }
        x.insert("chamfer".into(), json!(true));
        out.insert("feature".into(), Value::Object(x));
    }
    if ask.profile {
        out.insert("profile".into(), serde_json::to_value(&reading.profile).unwrap_or(Value::Null));
    }
    out.insert("note".into(), json!(note));
    Ok(ThreadReadAnswer { json: Value::Object(out), reading, thread, candidates: matched.candidates, suggested: matched.suggested, feature, note })
}

/// The face a Thread built from a suggestion is cut on: the fingerprint of the face read along.
pub fn suggestion_face(along: &ReadAlong<'_>) -> Option<FaceRef> {
    match along {
        ReadAlong::Face { topology, face, .. } => topology.faces.get(*face).map(|f| f.fingerprint),
        _ => None,
    }
}

// ---- a size as it is typed ----

/// A designation read: the size of the table as written, the pitch in millimetres it names,
/// and the standard it names (UNC, UNF, print).
#[derive(Debug, Clone, PartialEq)]
pub struct Designation {
    pub size: String,
    pub pitch: Option<f64>,
    pub standard: Option<ThreadStandard>,
}

/// A number as `\d+(?:[.,]\d+)?` reads it, the comma a point: the whole of `t`, or `None`.
fn decimal(t: &str, comma: bool) -> Option<f64> {
    let (whole, frac) = match t.find(|c: char| c == '.' || (comma && c == ',')) {
        Some(i) => (&t[..i], Some(&t[i + 1..])),
        None => (t, None),
    };
    let digits = |s: &str| !s.is_empty() && s.chars().all(|c| c.is_ascii_digit());
    if !digits(whole) || frac.is_some_and(|f| !digits(f)) {
        return None;
    }
    t.replace(',', ".").parse().ok()
}

/// A size as it is written with its pitch — "M10x1", "M10×1", "M10 x 1", "Ø16×3.5 print",
/// "1/4-20 UNC" — split into the size of the table and the pitch in millimetres; the standard
/// a designation names (UNC, UNF, print) with it. A size alone has no pitch. `asked` is the
/// standard the size is read for: a Unified one reads a dash as threads per inch.
pub fn read_designation(text: &str, asked: Option<ThreadStandard>) -> Designation {
    let mut t = text.trim().to_string();
    let mut standard = None;
    let lower = t.to_lowercase();
    for (word, s) in [("unc", ThreadStandard::Unc), ("unf", ThreadStandard::Unf), ("print", ThreadStandard::Print45)] {
        // The word at the end, after a word boundary.
        if lower.ends_with(word) && lower[..lower.len() - word.len()].chars().last().is_none_or(|c| !c.is_alphanumeric() && c != '_') {
            standard = Some(s);
            t = t[..t.len() - word.len()].trim_end().to_string();
            break;
        }
    }
    // Metric and the printing form: the pitch after an x or a ×.
    if !t.contains('/') && !t.starts_with('#') {
        for (i, _) in t.char_indices().skip(1) {
            let rest = t[i..].trim_start();
            let Some(sep) = rest.chars().next().filter(|c| matches!(c, 'x' | 'X' | '×' | '*')) else { continue };
            let after = rest[sep.len_utf8()..].trim();
            let after = after.strip_suffix("mm").or_else(|| after.strip_suffix("MM")).or_else(|| after.strip_suffix("Mm")).or_else(|| after.strip_suffix("mM")).map_or(after, str::trim_end);
            if let Some(p) = decimal(after, true) {
                return Designation { size: t[..i].to_string(), pitch: Some(p), standard };
            }
        }
    }
    // Unified: threads per inch after a dash.
    let inch = t.contains('/') || t.contains('#') || [standard, asked].iter().any(|x| matches!(x, Some(ThreadStandard::Unc) | Some(ThreadStandard::Unf)));
    if inch {
        for (i, _) in t.char_indices().skip(1) {
            let rest = t[i..].trim_start();
            let Some(after) = rest.strip_prefix('-') else { continue };
            if let Some(n) = decimal(after.trim_start(), false) {
                return Designation { size: t[..i].to_string(), pitch: Some(25.4 / n), standard };
            }
        }
    }
    Designation { size: t, pitch: None, standard }
}

/// A size as it is matched: no spaces, no case, no diameter sign — the printing form's "Ø10"
/// is "10" too.
fn size_key(label: &str) -> String {
    let t: String = label.trim().to_lowercase().chars().filter(|c| !c.is_whitespace()).collect();
    t.strip_prefix('ø').or_else(|| t.strip_prefix('⌀')).map_or(t.clone(), str::to_string)
}

/// What the sizes of a standard are called in a refusal.
fn standard_name(s: ThreadStandard) -> &'static str {
    match s {
        ThreadStandard::Metric => "ISO metric",
        ThreadStandard::Unc => "UNC",
        ThreadStandard::Unf => "UNF",
        ThreadStandard::Print45 => "45° printing",
    }
}

/// A thread's diameter and pitch from a size of the table as typed — alone, or as a
/// designation with its pitch — and a pitch given beside it, if any; refused where the size is
/// none of the standard's or the two pitches disagree.
pub fn thread_size_named(standard: ThreadStandard, size: &str, pitch: Option<f64>) -> Result<(f64, f64), String> {
    let read = read_designation(size, Some(standard));
    let want = size_key(&read.size);
    let sizes = thread_sizes(standard);
    let Some(found) = sizes.iter().find(|z| size_key(&z.label) == want) else {
        let labels: Vec<&str> = sizes.iter().map(|z| z.label.as_str()).collect();
        return Err(format!(
            "There is no {size} among the {} sizes — they are {}{}.",
            standard_name(standard),
            labels.join(", "),
            if read.pitch.is_some() { " (a pitch after the size: \"M10x1\", \"1/4-20\")" } else { "" }
        ));
    };
    if let (Some(r), Some(g)) = (read.pitch, pitch) {
        if (g - r).abs() > 1e-4 {
            return Err(format!("size “{size}” says a pitch of {} mm and pitch says {} — give one of them.", number_text(js_round(r * 10000.0) / 10000.0), number_text(g)));
        }
    }
    Ok((found.diameter, pitch.or(read.pitch).or_else(|| found.pitches.first().copied()).unwrap_or(f64::NAN)))
}

#[cfg(test)]
mod tests {
    //! ScanRuler's `reCommandsScan.test.ts`, "reverse.thread.read finds an M6×1 in a bore,
    //! along an axis given and along a body's face", on a scan made up the same way.
    use super::*;

    /// A block with a bore of crests `r_low` and root `r_high` at `pitch`, `length` long about
    /// the Z axis, as a triangle soup's points and their normals — the vertices of the bore's
    /// grid (the scan's points the reading takes), their normals facing the axis.
    fn bore_cloud(r_low: f64, r_high: f64, pitch: f64, length: f64, half: f64) -> (Vec<DVec3>, Vec<DVec3>) {
        let (angles, rows) = (72, 160);
        let mut pts = Vec::new();
        let mut normals = Vec::new();
        for j in 0..rows {
            let z = -length / 2.0 + length * j as f64 / (rows - 1) as f64;
            for k in 0..angles {
                let th = k as f64 / angles as f64 * 2.0 * PI;
                let phase = (z - pitch * th / (2.0 * PI)) / pitch;
                let t = phase - phase.floor();
                let r = r_low + (r_high - r_low) * if t < 0.5 { t * 2.0 } else { 2.0 - t * 2.0 };
                pts.push(DVec3::new(r * th.cos(), r * th.sin(), z));
                normals.push(DVec3::new(-th.cos(), -th.sin(), 0.0));
            }
        }
        // The square block's faces round it, a few points each.
        for k in 0..40 {
            let s = -half + 2.0 * half * k as f64 / 39.0;
            for z in [-length / 2.0, length / 2.0] {
                pts.push(DVec3::new(s, half, z));
                normals.push(DVec3::Z * z.signum());
            }
        }
        (pts, normals)
    }

    #[test]
    fn finds_an_m6_1_in_a_bore_along_an_axis_given() {
        let (pts, normals) = bore_cloud(2.459, 3.0, 1.0, 8.0, 10.0);
        let clouds: Vec<(&[DVec3], &[DVec3])> = vec![(&pts, &normals)];
        let along = ReadAlong::Line { origin: DVec3::ZERO, dir: DVec3::Z, radius: 2.75, length: 8.0 };
        let r = thread_read(along, &clouds, &ReadAsk::default()).expect("a reading");
        assert!(r.thread, "{}", r.note);
        assert_eq!(r.reading.internal, Some(true));
        assert!((r.reading.pitch.unwrap_or(0.0) - 1.0).abs() < 0.04);
        assert!((r.reading.minor - 4.92).abs() < 0.5, "{}", r.reading.minor);
        assert!((r.reading.major - 6.0).abs() < 0.5, "{}", r.reading.major);
        assert_eq!(r.reading.left_handed, Some(false));
        assert_eq!(r.suggested.as_ref().map(|s| s.designation.as_str()), Some("M6×1"));
        assert!(r.feature.is_none());
        assert!(r.note.contains("repeats every"), "{}", r.note);
        // The pitch read window by window; the profile only when asked.
        assert!(r.reading.windows.len() > 1 && r.reading.agree > 1);
        assert!(r.json.get("profile").is_none());
        let looked = thread_read(along, &clouds, &ReadAsk { profile: true, ..ReadAsk::default() }).expect("a reading");
        assert!(looked.json.get("profile").and_then(Value::as_array).is_some_and(|p| p.len() > 50));
        // An axis that misses the scan is refused.
        let missed = thread_read(ReadAlong::Line { origin: DVec3::new(50.0, 50.0, 0.0), dir: DVec3::Z, radius: 2.75, length: 8.0 }, &clouds, &ReadAsk::default());
        assert_eq!(missed.err().map(|e| e.code), Some("failed"));
        // No scan, no reading.
        let none: Vec<(&[DVec3], &[DVec3])> = vec![(&[], &[])];
        assert_eq!(thread_read(along, &none, &ReadAsk::default()).err().map(|e| e.code), Some("no_scan"));
    }

    /// `reCommandsPicks.test.ts`, "thread sizes as designations".
    #[test]
    fn reads_a_pitch_from_the_size_and_refuses_a_pitch_that_says_otherwise() {
        let d = |size: &str, pitch: Option<f64>, standard: Option<ThreadStandard>| Designation { size: size.to_string(), pitch, standard };
        assert_eq!(read_designation("M10x1", None), d("M10", Some(1.0), None));
        assert_eq!(read_designation("M10 × 1.25", None), d("M10", Some(1.25), None));
        assert_eq!(read_designation("m10X1", None), d("m10", Some(1.0), None));
        assert_eq!(read_designation("M10", None), d("M10", None, None));
        assert_eq!(read_designation("Ø16×3.5 print", None), d("Ø16", Some(3.5), Some(ThreadStandard::Print45)));
        let unc = read_designation("1/4-20 UNC", None);
        assert_eq!((unc.size.as_str(), unc.standard), ("1/4", Some(ThreadStandard::Unc)));
        assert!((unc.pitch.unwrap_or(0.0) - 25.4 / 20.0).abs() < 1e-9);
        assert_eq!(read_designation("#10-32", Some(ThreadStandard::Unf)).size, "#10");
        for (size, pitch) in [("M10x1", 1.0), ("M10 × 1.25", 1.25), ("M10", 1.5)] {
            assert_eq!(thread_size_named(ThreadStandard::Metric, size, None), Ok((10.0, pitch)), "{size}");
        }
        let both = thread_size_named(ThreadStandard::Metric, "M10x1", Some(1.5)).err().unwrap_or_default();
        assert!(both.contains("says a pitch of 1 mm and pitch says 1.5"), "{both}");
        // The same pitch twice is no conflict.
        assert_eq!(thread_size_named(ThreadStandard::Metric, "M10×1", Some(1.0)), Ok((10.0, 1.0)));
        assert_eq!(thread_size_named(ThreadStandard::Print45, "Ø16", None).map(|x| x.0), Ok(16.0));
        let none = thread_size_named(ThreadStandard::Metric, "M9x1", None).err().unwrap_or_default();
        assert!(none.starts_with("There is no M9x1 among the ISO metric sizes — they are M1, ") && none.ends_with(" (a pitch after the size: \"M10x1\", \"1/4-20\")."), "{none}");
    }

    #[test]
    fn thins_the_points_in_the_box_evenly_past_the_limit() {
        let pts: Vec<DVec3> = (0..1000).map(|i| DVec3::new(i as f64, 0.0, 0.0)).collect();
        let normals = vec![DVec3::Z; 1000];
        let q = query_vertices(&[(&pts, &normals)], VertexQuery { min: DVec3::new(100.0, -1.0, -1.0), max: DVec3::new(399.0, 1.0, 1.0), limit: 100 });
        assert_eq!(q.matched, 300);
        assert_eq!(q.step, 3);
        assert_eq!(q.positions.len(), 100);
        assert_eq!(q.positions[1].x, 103.0);
    }
}
