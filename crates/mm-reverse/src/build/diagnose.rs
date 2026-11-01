// SPDX-License-Identifier: AGPL-3.0-only
//! Why a round, a chamfer or a shell the kernel refused could not be made, read off the body: a
//! round larger than a face beside its edges, edges that do not round at any size, chains that
//! round alone and not together, a wall that does not fit the body or a face that will not
//! open. Ported from ScanRuler (`plugins/reverse/core/diagnose.ts` diagnoseRound,
//! diagnoseShell and their helpers, `kernelShell.ts` largestSizeWhile, `trapEdges.ts`, and
//! `build.ts` fail, inputFault, lastChanged), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! Each try is a whole kernel operation on the body as it stood before the step, so a diagnosis
//! is bounded by a budget off the step's own time; what the budget did not reach is said.
//!
//! Not ported yet: a shell's thin spots (ScanRuler measures the body's thickness down its
//! mesh's normals) and its faces offset one at a time (`facesAlone`), a draft's and an
//! offset's diagnosis (their features are not built yet), and a round's helper bodies (the
//! build keeps no record of a cut's tool to name). Those failures keep the kernel's own words.

use std::collections::HashSet;
use std::time::{Duration, Instant};

use glam::DVec3;
use solidcraft_geom::Vec3 as SVec3;
use solidcraft_kernel::{self as kernel, Body, ChamferSide};

use super::{sv3, Builder, FaceMesh};
use crate::js_number::{js_round, js_string, to_fixed};
use crate::model::{BodyId, ChamferFeature, ChamferMode, Feature, FeatureId, FeatureKind, FilletFeature, Part, ShellFeature};
use crate::topo::{split_edges, tangent_chain, EdgeInfo, FaceKind, Topology, CHAIN_ANGLE, CHAIN_JOIN_MM};

/// How many picks a diagnosis tries one at a time, at most — each try is a kernel operation on
/// the whole body.
const DIAGNOSE_PICKS: usize = 24;

/// A face beside a round or a chamfer under this many times the size squared is too small to
/// carry it, and is named as the likely reason.
const SMALL_FACE_SIZES: f64 = 4.0;

/// A failed round's diagnosis spends at most this many times the step's own time, and never
/// less than `ROUND_DIAGNOSIS_LEAST_MS`.
pub const ROUND_DIAGNOSIS_TIMES: f64 = 4.0;
pub const ROUND_DIAGNOSIS_LEAST_MS: f64 = 10_000.0;

/// A failed shell's diagnosis spends at most this many times the step's own time, and never
/// less than `SHELL_DIAGNOSIS_LEAST_MS`.
pub const SHELL_DIAGNOSIS_TIMES: f64 = 2.0;
pub const SHELL_DIAGNOSIS_LEAST_MS: f64 = 5000.0;

/// Of its budget, what a diagnosis of several chains spends on them alone.
const ALONE_SHARE: f64 = 0.5;

/// Where a diagnosis ran out of time before it had the size.
const TRY_IT: &str = "Try a smaller size on the body, or round fewer edges at once.";

/// No round or chamfer is tried smaller than this, mm, nor than an eighth of the size asked.
const LEAST_TRY_MM: f64 = 0.1;
const LEAST_TRY_SHARE: f64 = 1.0 / 8.0;

/// Sizes above the one asked tried too, as shares of it.
const SIZES_ABOVE: [f64; 3] = [1.05, 1.1, 1.2];

/// How many sizes are tried evenly between the largest the halving found and the size asked.
const SIZES_BETWEEN: usize = 4;

/// At most this many sizes of each kind, built and failed, are named.
const SIZES_NAMED: usize = 5;

/// The least number of halvings `largest_size_while` makes, and how near it brings a size.
const LEAST_HALVINGS: usize = 6;
const SIZE_WITHIN_MM: f64 = 0.01;

/// Lengths and areas of two chains alike agree within this share of the larger, and a hair.
const ALIKE_SHARE: f64 = 0.002;
const ALIKE_HAIR: f64 = 1e-3;

/// How many points along an edge its faces are read at, at most.
const APART_POINTS: usize = 16;

/// Flat faces meeting at under this many degrees — but not at none — are a crease a round
/// cannot end on; under `SAME_PLANE_DEG` two flat faces are one.
const SHALLOW_CREASE_DEG: f64 = 5.0;
const SAME_PLANE_DEG: f64 = 0.05;
/// How near an edge's end lies to a chain's to meet it, mm.
const MEET_MM: f64 = 1e-3;

/// A face at a chain's open end faces back along the chain when its normal there leans further
/// than this, as a cosine, against the way the chain was going.
const RUNS_INTO_COS: f64 = -0.1;

/// How many failing edges or chains a message names one by one, and how many groups past that.
const PLACES_LISTED: usize = 3;
const GROUPS_LISTED: usize = 4;

/// How many middles a crash names.
const TRAP_MIDS: usize = 6;

/// Under this area a face has no centroid worth the name, mm².
pub const NO_AREA_MM2: f64 = 1e-6;

fn r2(v: f64) -> String {
    js_string(js_round(v * 100.0) / 100.0 + 0.0)
}

/// A size in millimetres as a sentence says it: two decimals at most.
fn mm(v: f64) -> String {
    format!("{} mm", r2(v))
}

fn point(p: DVec3) -> String {
    format!("({}, {}, {})", r2(p.x), r2(p.y), r2(p.z))
}

fn point1(p: DVec3) -> String {
    format!("({}, {}, {})", to_fixed(p.x, 1), to_fixed(p.y, 1), to_fixed(p.z, 1))
}

/// Degrees as a sentence gives them: one decimal under ten, whole above.
fn deg(v: f64) -> String {
    let n = if v < 10.0 { js_round(v * 10.0) / 10.0 } else { js_round(v) };
    format!("{}°", js_string(n))
}

/// What an operation the kernel refused said, without the name of the kernel call it was in
/// (ScanRuler's `withoutCallName`): "Filleting: filletWithHistory: …" is "Filleting: …".
pub fn without_call_name(message: &str) -> String {
    let Some((head, rest)) = message.split_once(": ") else { return message.to_string() };
    let is_words = head.chars().next().is_some_and(|c| c.is_ascii_uppercase()) && head.chars().skip(1).all(|c| c.is_ascii_lowercase() || c == ' ');
    if !is_words {
        return message.to_string();
    }
    if let Some((call, after)) = rest.split_once(": ") {
        let camel = call.chars().next().is_some_and(|c| c.is_ascii_lowercase()) && call.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') && call.chars().any(|c| c.is_ascii_uppercase());
        if camel {
            return format!("{head}: {after}");
        }
    }
    message.to_string()
}

/// What a failure said, without the operation's name the kernel put before it.
fn said(message: &str) -> String {
    let told = without_call_name(message);
    let trimmed = match told.split_once(": ") {
        Some((head, rest)) if head.chars().next().is_some_and(|c| c.is_ascii_uppercase()) && head.chars().skip(1).all(|c| c.is_ascii_lowercase() || c == ' ') => rest.to_string(),
        _ => told,
    };
    trimmed.strip_suffix('.').map_or(trimmed.clone(), str::to_string)
}

/// A diagnosis: what it says, and the edges or outlines to draw red.
pub struct Diagnosis {
    pub message: String,
    pub where_: Vec<Vec<DVec3>>,
}

/// What a diagnosis may still spend: its budget against what one more try is expected to take.
struct Budget {
    start: Instant,
    ms: f64,
    guess: f64,
    slowest: Option<f64>,
}

impl Budget {
    fn new(ms: f64, guess: f64) -> Budget {
        Budget { start: Instant::now(), ms, guess, slowest: None }
    }

    fn expected(&self) -> f64 {
        self.slowest.unwrap_or(self.guess)
    }

    fn spent(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1000.0
    }

    fn affords(&self) -> bool {
        self.spent() + self.expected() <= self.ms
    }

    fn timed<T>(&mut self, f: impl FnOnce() -> T) -> T {
        let began = Instant::now();
        let out = f();
        let took = began.elapsed().as_secs_f64() * 1000.0;
        self.slowest = Some(self.slowest.unwrap_or(0.0).max(took));
        out
    }

    fn ahead(&mut self, guess: f64) {
        self.guess = guess.max(self.expected());
        self.slowest = None;
    }

    fn over(&self) -> String {
        let s = |ms: f64| if ms >= 1000.0 { format!("{} s", js_round(ms / 1000.0)) } else { format!("{} ms", js_round(ms)) };
        format!("over budget (a try takes about {}, the diagnosis {} at most)", s(self.expected()), s(self.ms))
    }
}

/// The largest size below `size` at which `works` holds, by halving, asking `go_on` before each
/// try (ScanRuler's `largestSizeWhile`): the size found (zero for none) and whether the halving
/// was done. With a `floor`, nothing under it is tried, and the floor itself last.
fn largest_size_while(size: f64, works: &mut dyn FnMut(f64) -> bool, go_on: &mut dyn FnMut() -> bool, floor: f64) -> (f64, bool) {
    if floor >= size {
        return (0.0, true);
    }
    let halvings = LEAST_HALVINGS.max(((size - floor) / SIZE_WITHIN_MM).log2().ceil().max(0.0) as usize);
    let (mut lo, mut hi) = (floor, size);
    let mut held = floor == 0.0;
    for _ in 0..halvings {
        if !go_on() {
            return (if held { lo } else { 0.0 }, false);
        }
        let mid = (lo + hi) / 2.0;
        if works(mid) {
            lo = mid;
            held = true;
        } else {
            hi = mid;
        }
    }
    if held {
        return (lo, true);
    }
    if !go_on() {
        return (0.0, false);
    }
    (if works(floor) { floor } else { 0.0 }, true)
}

/// What the sizes tried of a round or a chamfer came to (ScanRuler's `SizesRead`).
struct SizesRead {
    up_to: f64,
    floor: f64,
    settled: bool,
    built: Vec<f64>,
    failed: Vec<f64>,
}

/// The sizes a round or a chamfer takes, read both ways from the size asked, which failed.
fn read_sizes(size: f64, works: &mut dyn FnMut(f64) -> bool, go_on: &mut dyn FnMut() -> bool) -> SizesRead {
    let floor = size.min(LEAST_TRY_MM.max(size * LEAST_TRY_SHARE));
    let mut built: Vec<f64> = Vec::new();
    let mut failed: Vec<f64> = vec![size];
    let (up_to, halved_settled) = {
        let mut probe = |at: f64| {
            let ok = works(at);
            if ok {
                built.push(at);
            } else {
                failed.push(at);
            }
            ok
        };
        largest_size_while(size, &mut probe, go_on, floor)
    };
    let from = if up_to > 0.0 { up_to } else { floor };
    let between: Vec<f64> = (0..SIZES_BETWEEN).map(|i| from + (size - from) * (i + 1) as f64 / (SIZES_BETWEEN + 1) as f64).collect();
    let mut settled = halved_settled;
    for at in SIZES_ABOVE.iter().map(|f| size * f).chain(between) {
        if built.iter().chain(&failed).any(|x| (x - at).abs() < size * 0.01) {
            continue;
        }
        if !go_on() {
            settled = false;
            break;
        }
        if works(at) {
            built.push(at);
        } else {
            failed.push(at);
        }
    }
    SizesRead { up_to, floor, settled, built, failed }
}

/// Whether a size built above one that failed: the round does not fail for its size.
fn not_by_size(s: &SizesRead) -> bool {
    s.built.iter().any(|b| s.failed.iter().any(|f| b > f))
}

fn nearest(built: &[f64], size: f64) -> f64 {
    built.iter().copied().fold(built.first().copied().unwrap_or(size), |a, b| if (b - size).abs() < (a - size).abs() { b } else { a })
}

/// The sizes tried that tell a round does not fail for its size, as a sentence says them.
fn both_ways(s: &SizesRead, size: f64, outside: usize, builds: &str, verb: &str, kind: &str) -> String {
    let top = s.built.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut built: Vec<f64> = s.built.iter().copied().filter(|b| *b >= s.up_to).map(|b| js_round(b * 100.0) / 100.0).collect();
    built.sort_by(|a, b| a.total_cmp(b));
    built.dedup();
    built.truncate(SIZES_NAMED);
    let mut failed: Vec<f64> = s.failed.iter().copied().filter(|f| *f < top).map(|f| js_round(f * 100.0) / 100.0).collect();
    failed.sort_by(|a, b| a.total_cmp(b));
    failed.dedup();
    let failed: Vec<f64> = failed[failed.len().saturating_sub(SIZES_NAMED)..].to_vec();
    let near = nearest(&s.built, size);
    let todo = if outside > 0 {
        format!(
            "{verb} the {} there not in this {kind} with {}, every edge at that corner in one {kind} — or take {}, the nearest size that builds",
            if outside == 1 { "edge".to_string() } else { format!("{outside} edges") },
            if builds == "build" { "them" } else { "it" },
            mm(near)
        )
    } else {
        format!("every edge there is in this {kind} already; take {}, the nearest size that builds", mm(near))
    };
    let list = |v: &[f64]| v.iter().map(|x| js_string(*x)).collect::<Vec<_>>().join(", ");
    format!("{builds} at {}; {} at {} mm — not a size limit: the corner where the {verb}s meet; {todo}", list(&built), if builds == "builds" { "fails" } else { "fail" }, list(&failed))
}

/// A round's or a chamfer's tangent chain: the edges picked on it, and all of it.
#[derive(Clone)]
struct Chain {
    picked: Vec<u32>,
    whole: Vec<u32>,
}

/// Chains alike, and the one tried for them all.
struct AlikeSet {
    one: Chain,
    all: Vec<Chain>,
}

fn alike_within(a: f64, b: f64) -> bool {
    (a - b).abs() <= a.max(b) * ALIKE_SHARE + ALIKE_HAIR
}

/// The chains grouped by what is the same in each — the same lengths between the same faces,
/// ending against the same faces — as a pattern's copies are (ScanRuler's `alikeSets`).
fn alike_sets(chains: &[Chain], t: &Topology) -> Vec<AlikeSet> {
    if chains.len() < 2 {
        return chains.iter().map(|c| AlikeSet { one: c.clone(), all: vec![c.clone()] }).collect();
    }
    let vkey = |p: DVec3| format!("{},{},{}", js_round(p.x * 100.0), js_round(p.y * 100.0), js_round(p.z * 100.0));
    let mut at_vertex: std::collections::HashMap<String, Vec<u32>> = std::collections::HashMap::new();
    for e in &t.edges {
        for p in e.ends {
            at_vertex.entry(vkey(p)).or_default().extend(e.faces.iter().copied());
        }
    }
    let faces_read = |hashes: Vec<u32>| -> (String, Vec<f64>) {
        let mut seen: Vec<u32> = Vec::new();
        for h in hashes {
            if !seen.contains(&h) {
                seen.push(h);
            }
        }
        let mut read: Vec<(String, f64)> = seen.iter().map(|h| t.face(*h).map_or((String::from("?/?"), 0.0), |f| (format!("{}/{}", f.fingerprint.kind.name(), f.fingerprint.origin), f.fingerprint.area))).collect();
        read.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
        (read.iter().map(|x| x.0.clone()).collect::<Vec<_>>().join(" "), read.iter().map(|x| x.1).collect())
    };
    struct Print {
        picked: usize,
        kinds: String,
        lengths: Vec<f64>,
        spread: f64,
        sides: String,
        side_areas: Vec<f64>,
        at_ends: String,
        end_areas: Vec<f64>,
    }
    let print_of = |c: &Chain| -> Print {
        let mut own: Vec<&EdgeInfo> = c.whole.iter().filter_map(|h| t.edge(*h)).collect();
        own.sort_by(|a, b| a.fingerprint.length.total_cmp(&b.fingerprint.length));
        let ends: Vec<DVec3> = own.iter().flat_map(|e| e.ends).collect();
        let middle = if ends.is_empty() { DVec3::ZERO } else { ends.iter().fold(DVec3::ZERO, |s, p| s + *p) / ends.len() as f64 };
        let (sides, side_areas) = faces_read(own.iter().flat_map(|e| e.faces.iter().copied()).collect());
        let mine: HashSet<u32> = own.iter().flat_map(|e| e.faces.iter().copied()).collect();
        let (at_ends, end_areas) = faces_read(ends.iter().flat_map(|p| at_vertex.get(&vkey(*p)).cloned().unwrap_or_default()).filter(|h| !mine.contains(h)).collect());
        let mut kinds: Vec<&str> = own.iter().map(|e| e.fingerprint.kind.name()).collect();
        kinds.sort_unstable();
        Print { picked: c.picked.len(), kinds: kinds.join(" "), lengths: own.iter().map(|e| e.fingerprint.length).collect(), spread: ends.iter().map(|p| (*p - middle).length()).fold(0.0, f64::max), sides, side_areas, at_ends, end_areas }
    };
    let all_within = |a: &[f64], b: &[f64]| a.len() == b.len() && a.iter().zip(b).all(|(x, y)| alike_within(*x, *y));
    let alike = |a: &Print, b: &Print| a.picked == b.picked && a.kinds == b.kinds && a.sides == b.sides && a.at_ends == b.at_ends && alike_within(a.spread, b.spread) && all_within(&a.lengths, &b.lengths) && all_within(&a.side_areas, &b.side_areas) && all_within(&a.end_areas, &b.end_areas);
    let mut sets: Vec<(AlikeSet, Print)> = Vec::new();
    for c in chains {
        let print = print_of(c);
        match sets.iter_mut().find(|(_, p)| alike(p, &print)) {
            Some((set, _)) => set.all.push(c.clone()),
            None => sets.push((AlikeSet { one: c.clone(), all: vec![c.clone()] }, print)),
        }
    }
    sets.into_iter().map(|(s, _)| s).collect()
}

/// Where a face is, as a message names it: its centroid, or where it has no area the middle of
/// its edges.
fn face_middle(t: &Topology, h: u32) -> DVec3 {
    let Some(f) = t.face(h) else { return DVec3::ZERO };
    if f.fingerprint.area >= NO_AREA_MM2 {
        return f.fingerprint.centroid;
    }
    let round: Vec<&EdgeInfo> = t.edges.iter().filter(|e| e.faces.contains(&h)).collect();
    if round.is_empty() {
        return f.fingerprint.centroid;
    }
    round.iter().fold(DVec3::ZERO, |s, e| s + e.fingerprint.mid) / round.len() as f64
}

/// A feature as a diagnosis names it: its name and id, or its id alone.
fn feature_named(b: &Builder, id: i64) -> String {
    match u32::try_from(id).ok().and_then(|i| b.track.name_of(i)) {
        Some(name) => format!("{name} (feature {id})"),
        None => format!("feature {id}"),
    }
}

/// A face's maker as a diagnosis says it.
fn maker_of(b: &Builder, t: &Topology, h: u32) -> Option<String> {
    let made = t.face(h)?.fingerprint.origin;
    (made >= 0).then(|| format!("a face of {}", feature_named(b, made)))
}

/// The faces beside the given edges that are too small to hold a round or a chamfer of `size`,
/// as a message names them, the smallest first, three at most.
fn small_faces_beside(b: &Builder, t: &Topology, hashes: &[u32], size: f64) -> Vec<String> {
    let beside: HashSet<u32> = hashes.iter().filter_map(|h| t.edge(*h)).flat_map(|e| e.faces.iter().copied()).collect();
    let mut small: Vec<_> = t.faces.iter().filter(|f| beside.contains(&f.hash) && f.fingerprint.area < SMALL_FACE_SIZES * size * size).collect();
    small.sort_by(|a, b| a.fingerprint.area.total_cmp(&b.fingerprint.area));
    small
        .into_iter()
        .take(3)
        .map(|f| {
            let area = f.fingerprint.area;
            let kind = if f.fingerprint.kind == FaceKind::Other { "face" } else { f.fingerprint.kind.name() };
            let area_words = if area < 0.01 { js_string(crate::js_number::to_precision_number(area, 2)) } else { r2(area) };
            let by = maker_of(b, t, f.hash).map_or(String::new(), |w| format!(", {w}"));
            format!("a {kind} of {area_words} mm² at {}{by}", point1(face_middle(t, f.hash)))
        })
        .collect()
}

/// The shallow creases a chain of edges runs into: edges not of the chain, between two flat
/// faces at a few degrees, that touch one of its ends — three at most.
fn shallow_creases_at(t: &Topology, hashes: &[u32]) -> Vec<String> {
    let chain: HashSet<u32> = hashes.iter().copied().collect();
    let ends: Vec<DVec3> = t.edges.iter().filter(|e| chain.contains(&e.hash)).flat_map(|e| e.ends).collect();
    if ends.is_empty() {
        return Vec::new();
    }
    let meets = |p: DVec3| ends.iter().any(|q| (p - *q).length() <= MEET_MM);
    let mut out: Vec<(f64, DVec3)> = Vec::new();
    for e in &t.edges {
        if chain.contains(&e.hash) || e.faces.len() != 2 || (!meets(e.ends[0]) && !meets(e.ends[1])) {
            continue;
        }
        let (Some(a), Some(c)) = (t.face(e.faces[0]), t.face(e.faces[1])) else { continue };
        if a.fingerprint.kind != FaceKind::Plane || c.fingerprint.kind != FaceKind::Plane {
            continue;
        }
        let d = a.fingerprint.dir.dot(c.fingerprint.dir).abs().min(1.0).acos().to_degrees();
        if d > SAME_PLANE_DEG && d < SHALLOW_CREASE_DEG {
            out.push((d, e.fingerprint.mid));
        }
    }
    out.sort_by(|x, y| x.0.total_cmp(&y.0));
    out.into_iter().take(3).map(|(d, at)| format!("a crease of {}° between two flat faces at {}", js_string(js_round(d * 100.0) / 100.0), point1(at))).collect()
}

/// How far apart the two faces along some edges stand, by their normals — the least and the
/// most, read at points along each edge — and where they come closest to flat.
fn faces_apart_along(fm: Option<&FaceMesh>, t: &Topology, hashes: &[u32]) -> Option<(f64, f64, DVec3)> {
    let fm = fm?;
    let (mut least, mut most, mut at) = (f64::INFINITY, f64::NEG_INFINITY, None);
    for h in hashes {
        let Some(e) = t.edge(*h) else { continue };
        let faces: Vec<u32> = e.faces.iter().copied().collect::<HashSet<_>>().into_iter().collect();
        if faces.len() != 2 || e.polyline.len() < 2 {
            continue;
        }
        let n = e.polyline.len();
        let take = n.min(APART_POINTS);
        let points: Vec<DVec3> = (0..take).map(|i| e.polyline[((i * (n - 1)) as f64 / (take.max(2) - 1) as f64).round() as usize]).collect();
        let a = fm.normals(faces[0] as usize, &points);
        let c = fm.normals(faces[1] as usize, &points);
        for (i, p) in points.iter().enumerate() {
            let (Some(x), Some(y)) = (a[i], c[i]) else { continue };
            let d = x.dot(y).clamp(-1.0, 1.0).acos().to_degrees();
            if d < least {
                least = d;
                at = Some(*p);
            }
            most = most.max(d);
        }
    }
    at.map(|p| (least, most, p))
}

/// The faces a chain runs into at its open ends: faces at an end that are not either side of
/// any of its edges, whose normal there faces back against the chain.
fn faces_run_into(fm: Option<&FaceMesh>, t: &Topology, whole: &[u32]) -> Vec<(u32, DVec3)> {
    let chain: HashSet<u32> = whole.iter().copied().collect();
    let mut own: HashSet<u32> = HashSet::new();
    let mut ends: Vec<(DVec3, DVec3)> = Vec::new();
    for e in t.edges.iter().filter(|e| chain.contains(&e.hash)) {
        own.extend(e.faces.iter().copied());
        ends.push((e.ends[0], -e.end_tangents[0]));
        ends.push((e.ends[1], e.end_tangents[1]));
    }
    let near = |p: DVec3, q: DVec3| (p - q).length() <= MEET_MM;
    let open: Vec<(DVec3, DVec3)> = ends.iter().enumerate().filter(|(i, x)| !ends.iter().enumerate().any(|(j, y)| j != *i && near(x.0, y.0))).map(|(_, x)| *x).collect();
    let mut out: Vec<(u32, DVec3)> = Vec::new();
    for (at_end, outward) in open {
        let mut at: Vec<u32> = Vec::new();
        for e in &t.edges {
            if chain.contains(&e.hash) || !(near(e.ends[0], at_end) || near(e.ends[1], at_end)) {
                continue;
            }
            for f in &e.faces {
                if !own.contains(f) && !at.contains(f) {
                    at.push(*f);
                }
            }
        }
        for h in at {
            if out.iter().any(|(x, _)| *x == h) {
                continue;
            }
            let n = match t.face(h) {
                Some(f) if f.fingerprint.kind == FaceKind::Plane => Some(f.fingerprint.dir),
                _ => fm.and_then(|m| m.normals(h as usize, &[at_end]).first().copied().flatten()),
            };
            if n.is_some_and(|n| n.dot(outward) < RUNS_INTO_COS) {
                out.push((h, at_end));
            }
        }
    }
    out
}

/// How many edges meet the given edges' ends that are not among `asked` — the edges a corner's
/// round would want in the same feature — seams left out.
fn edges_at_ends_beside(t: &Topology, hashes: &[u32], asked: &HashSet<u32>) -> usize {
    let of: HashSet<u32> = hashes.iter().copied().collect();
    let ends: Vec<DVec3> = t.edges.iter().filter(|e| of.contains(&e.hash)).flat_map(|e| e.ends).collect();
    let meets = |p: DVec3| ends.iter().any(|q| (p - *q).length() <= MEET_MM);
    t.edges.iter().filter(|e| !asked.contains(&e.hash) && e.faces.iter().collect::<HashSet<_>>().len() == 2 && (meets(e.ends[0]) || meets(e.ends[1]))).count()
}

/// A failing edge or chain's place and the makers of the faces either side.
fn edge_where(b: &Builder, t: &Topology, c: &Chain) -> (String, String) {
    let Some(first) = c.picked.first().and_then(|h| t.edge(*h)).or_else(|| c.whole.first().and_then(|h| t.edge(*h))) else { return (String::new(), String::new()) };
    let mut faces: Vec<u32> = Vec::new();
    for e in t.edges.iter().filter(|e| c.whole.contains(&e.hash)) {
        for f in &e.faces {
            if !faces.contains(f) {
                faces.push(*f);
            }
        }
    }
    let mut words: Vec<String> = Vec::new();
    for h in &faces {
        let w = maker_of(b, t, *h).unwrap_or_else(|| "a face".to_string());
        if !words.contains(&w) {
            words.push(w);
        }
    }
    let own = "a face of ";
    let between = if words.len() == 1 && faces.len() > 1 {
        match words[0].strip_prefix(own) {
            Some(rest) => format!("two faces of {rest}"),
            None => format!("two faces, each {}", words[0]),
        }
    } else {
        words.iter().take(3).cloned().collect::<Vec<_>>().join(" and ")
    };
    (point(first.fingerprint.mid), between)
}

fn edge_place(b: &Builder, t: &Topology, c: &Chain) -> String {
    let (at, between) = edge_where(b, t, c);
    if at.is_empty() {
        String::new()
    } else {
        format!(" at {at} between {between}")
    }
}

/// The failing edges or chains of a message: each by itself up to `PLACES_LISTED`, past that
/// grouped by what lies either side, with the first place of each.
fn list_failing(failing: &[(String, String, String)], unit: &str) -> String {
    if failing.len() <= PLACES_LISTED {
        return failing.iter().map(|f| f.0.clone()).collect::<Vec<_>>().join(", ");
    }
    let mut groups: Vec<(String, Vec<&(String, String, String)>)> = Vec::new();
    for f in failing {
        match groups.iter_mut().find(|(k, _)| *k == f.2) {
            Some((_, g)) => g.push(f),
            None => groups.push((f.2.clone(), vec![f])),
        }
    }
    groups.sort_by(|a, b| b.1.len().cmp(&a.1.len()));
    let said: Vec<String> = groups.iter().take(GROUPS_LISTED).map(|(_, g)| if g.len() == 1 { g[0].0.clone() } else { format!("{} {unit} between {}, the first at {}", g.len(), g[0].2, g[0].1) }).collect();
    let left: usize = groups.iter().skip(GROUPS_LISTED).map(|(_, g)| g.len()).sum();
    format!("{}{}", said.join("; "), if left > 0 { format!("; and {left} more") } else { String::new() })
}

/// A face of a body as a diagnosis names it: its index, surface and maker.
fn face_named(b: &Builder, t: &Topology, h: u32) -> String {
    let Some(f) = t.face(h) else { return "face ?".to_string() };
    let made = if f.fingerprint.origin >= 0 { format!(", made by {}", feature_named(b, f.fingerprint.origin)) } else { String::new() };
    format!("face {h} ({}{made})", f.fingerprint.kind.name())
}

/// What a failing round's or chamfer's chains run into at their ends, and the ways round.
fn runs_into_text(b: &Builder, t: &Topology, fm: Option<&FaceMesh>, failing: &[Chain], verb: &str, subject: &dyn Fn(&Chain) -> String) -> String {
    let mut clauses: Vec<String> = Vec::new();
    let mut later: Vec<i64> = Vec::new();
    for c in failing {
        let into = faces_run_into(fm, t, &c.whole);
        if into.is_empty() {
            continue;
        }
        clauses.push(format!("{} runs into {}", subject(c), into.iter().map(|(h, at)| format!("{} at {}", face_named(b, t, *h), point(*at))).collect::<Vec<_>>().join(" and ")));
        let makers: HashSet<i64> = t.edges.iter().filter(|e| c.whole.contains(&e.hash)).flat_map(|e| e.faces.iter()).filter_map(|f| t.face(*f)).map(|f| f.fingerprint.origin).filter(|o| *o >= 0).collect();
        for (h, _) in &into {
            let Some(m) = t.face(*h).map(|f| f.fingerprint.origin).filter(|o| *o >= 0) else { continue };
            if !makers.is_empty() && makers.iter().all(|o| *o < m) && !later.contains(&m) {
                later.push(m);
            }
        }
    }
    if clauses.is_empty() {
        return String::new();
    }
    if clauses.len() > PLACES_LISTED {
        let more = clauses.len() - PLACES_LISTED;
        clauses.truncate(PLACES_LISTED);
        clauses.push(format!("and {more} more"));
    }
    let in_sketch = if verb == "round" { "round it in the sketch instead" } else { "draw it in the sketch instead" };
    let before = if later.is_empty() { String::new() } else { format!(", or {verb} before {}", later.iter().map(|m| feature_named(b, *m)).collect::<Vec<_>>().join(" and ")) };
    format!(" {} — a {verb} that ends against a wall often fails there: end it at a sharp corner, {in_sketch}{before}.", clauses.join("; "))
}

/// A round or a chamfer as the diagnosis tries it.
enum Blend<'a> {
    Fillet(&'a FilletFeature),
    Chamfer(&'a ChamferFeature),
}

impl Blend<'_> {
    fn size(&self) -> f64 {
        match self {
            Blend::Fillet(x) => x.radius,
            Blend::Chamfer(x) => x.distance,
        }
    }

    /// The blend of these edges at this size, made as the build makes it — a two-sided chamfer
    /// keeping the proportion of its distances.
    fn make(&self, body: &Body, points: &[SVec3], at: f64) -> bool {
        let r = match self {
            Blend::Fillet(_) => kernel::fillet(body, points, at),
            Blend::Chamfer(x) => match x.mode {
                Some(ChamferMode::TwoDistances) => kernel::chamfer_sides(body, points, at, ChamferSide::Distance(x.distance2.unwrap_or(at) * at / x.distance.max(1e-12)), x.flip),
                Some(ChamferMode::DistanceAngle) => kernel::chamfer_sides(body, points, at, ChamferSide::Angle(x.angle_deg.unwrap_or(45.0).to_radians()), x.flip),
                _ => kernel::chamfer(body, points, at),
            },
        };
        kernel::guard("diagnosis", || r).is_ok()
    }
}

fn points_of(t: &Topology, hashes: &[u32]) -> Vec<SVec3> {
    hashes.iter().filter_map(|h| t.edge(*h)).map(|e| sv3(e.fingerprint.mid)).collect()
}

/// Why a round or a chamfer the kernel refused could not be made (ScanRuler's
/// `diagnoseRound`): each tangent chain of its edges tried alone at the size asked — one of each
/// set alike, its answer carried to the set — then the sizes the failing ones take, read both
/// ways, and whether all of them take that size together; or, each taking the size alone, the
/// sizes they take together.
fn diagnose_round(b: &mut Builder, blend: Blend, body: BodyId, edges: (&[crate::model::EdgeRef], &[crate::pick_query::StoredPick<crate::model::EdgeRef>]), chain: bool, limit: (f64, f64)) -> Option<Diagnosis> {
    let quiet = std::mem::take(&mut b.warnings);
    let found = b.find_edges(body, edges.0, edges.1);
    b.warnings = quiet;
    let found = found.ok()?;
    let state = b.body(body)?.clone();
    if state.surface {
        return None;
    }
    let t = state.topology();
    let fm = FaceMesh::of(&state.shape);
    let splits = split_edges(&t);
    let chained = |seeds: &[u32]| tangent_chain(&t.edges, seeds, &splits, CHAIN_ANGLE, CHAIN_JOIN_MM);
    let seeds: Vec<u32> = found.iter().map(|i| *i as u32).collect();
    let hashes: Vec<u32> = if chain { chained(&seeds) } else { seeds };
    let size = blend.size();
    let (verb, kind) = match blend {
        Blend::Fillet(_) => ("round", "fillet"),
        Blend::Chamfer(_) => ("chamfer", "chamfer"),
    };
    let mut budget = Budget::new(limit.0, limit.1);
    let asked: HashSet<u32> = hashes.iter().copied().collect();
    let shape = state.shape.clone();
    let attempt = |budget: &mut Budget, hs: &[u32], at: f64| -> bool {
        let pts = points_of(&t, hs);
        budget.timed(|| blend.make(&shape, &pts, at))
    };
    let drawn = |hs: &[u32]| -> Vec<Vec<DVec3>> { hs.iter().filter_map(|h| t.edge(*h)).filter(|e| e.polyline.len() >= 2).map(|e| e.polyline.clone()).collect() };
    // The kernel rounds a tangent chain as one, so what is probed is the chains, each once.
    let mut left: HashSet<u32> = hashes.iter().copied().collect();
    let mut chains: Vec<Chain> = Vec::new();
    for h in &hashes {
        if !left.contains(h) {
            continue;
        }
        let whole = chained(&[*h]);
        let picked: Vec<u32> = whole.iter().copied().filter(|x| left.contains(x)).collect();
        for x in &picked {
            left.remove(x);
        }
        chains.push(Chain { picked, whole });
    }
    let sets = alike_sets(&chains, &t);
    let mut probed: Vec<usize> = Vec::new();
    let mut failing_sets: Vec<usize> = Vec::new();
    for (i, set) in sets.iter().enumerate().take(DIAGNOSE_PICKS) {
        if !budget.affords() {
            break;
        }
        probed.push(i);
        if !attempt(&mut budget, &set.one.picked, size) {
            failing_sets.push(i);
        }
    }
    let failing: Vec<Chain> = failing_sets.iter().flat_map(|i| sets[*i].all.clone()).collect();
    let m = chains.len();
    let edge_count = hashes.len();
    let unit = if m == edge_count { "edges" } else { "chains" };
    let reached: usize = probed.iter().map(|i| sets[*i].all.len()).sum();
    let cut = probed.len() < sets.len().min(DIAGNOSE_PICKS);
    let per_try = limit.1;
    if !failing.is_empty() {
        let n = failing.len();
        // The sizes one chain of each failing set takes, the largest set first.
        let mut order = failing_sets.clone();
        order.sort_by(|a, b| sets[*b].all.len().cmp(&sets[*a].all.len()));
        let mut reads: Vec<(Chain, SizesRead)> = Vec::new();
        for i in order {
            let alone_ok = |bg: &Budget| if m > 1 { bg.affords() && bg.spent() <= bg.ms * ALONE_SHARE } else { bg.affords() };
            if !alone_ok(&budget) {
                break;
            }
            let one = sets[i].one.clone();
            let budget_cell = std::cell::RefCell::new(&mut budget);
            let read = {
                let mut works = |at: f64| {
                    let mut bg = budget_cell.borrow_mut();
                    attempt(&mut bg, &one.picked, at)
                };
                let mut go_on = || alone_ok(&budget_cell.borrow());
                read_sizes(size, &mut works, &mut go_on)
            };
            reads.push((one, read));
        }
        let known: Vec<&SizesRead> = reads.iter().map(|(_, r)| r).filter(|r| r.settled || r.up_to > 0.0).collect();
        let up_to = if known.is_empty() { 0.0 } else { known.iter().map(|r| r.up_to).fold(f64::INFINITY, f64::min) };
        let all_read = reads.len() == failing_sets.len();
        let halved_all = reads.iter().all(|(_, r)| r.settled);
        let settled = all_read && halved_all;
        let odd = reads.iter().position(|(_, r)| not_by_size(r));
        let tried = if cut {
            format!(" (the first {reached} tried; the rest not: {})", budget.over())
        } else if reached < m {
            format!(" (the first {reached} tried)")
        } else {
            String::new()
        };
        let chained_failing: Vec<&Chain> = failing.iter().filter(|c| c.whole.len() > 1).collect();
        let what = |c: &Chain| {
            let head = if c.whole.len() > 1 { format!("the chain of {} tangent edges{}", c.whole.len(), if c.picked.len() < c.whole.len() { format!(" ({} picked)", c.picked.len()) } else { String::new() }) } else { "the edge".to_string() };
            format!("{head}{}", edge_place(b, &t, c))
        };
        let capital = |s: String| {
            let mut c = s.chars();
            c.next().map(|f| f.to_uppercase().collect::<String>() + c.as_str()).unwrap_or_default()
        };
        let which = if m == 1 {
            format!("{} does not {verb}", if !chained_failing.is_empty() { format!("The chain of {} tangent edges", failing[0].whole.len()) } else { "The edge".to_string() })
        } else if n == m {
            format!("None of the {m} {} {verb}s", if m == edge_count { "edges".to_string() } else { format!("chains ({edge_count} edges)") })
        } else {
            let list: Vec<(String, String, String)> = failing.iter().map(|c| {
                let (at, between) = edge_where(b, &t, c);
                (what(c), at, between)
            }).collect();
            format!("{n} of the {m} {} do not {verb}: {}", if m == edge_count { "edges".to_string() } else { format!("chains ({edge_count} edges)") }, list_failing(&list, if m == edge_count { "edges" } else { "chains" }))
        };
        let as_one = if chained_failing.is_empty() { String::new() } else { format!(" — the kernel {verb}s a tangent chain as one, so no edge of {} {verb}s alone either", if chained_failing.len() == 1 && n == 1 { "it" } else { "such a chain" }) };
        let shown = if n == m { " — drawn red on the part" } else { " — drawn red on the part; the others do" };
        let read_on = if reads.len() == n {
            String::new()
        } else if all_read {
            format!(" Read on {} of the {n} {unit}: the others are alike — the same lengths and the same faces either side — and taken to match.", reads.len())
        } else {
            format!(" That holds for the {} of the {n} {unit} read and those alike; the rest were not read{}.", reads.len(), if halved_all { format!(": {}", budget.over()) } else { String::new() })
        };
        let over = budget.over();
        let unfound = |they: &str| format!("The size {they} was not found: {over}. {TRY_IT}");
        let halved = |at: f64, is_settled: bool, over: &str| format!("{}{}", mm(at), if is_settled { String::new() } else { format!(" (not halved further: {over})") });
        let together_words = |tr: &SizesRead, from: f64, over: &str| {
            if not_by_size(tr) {
                format!("Together they {}.", both_ways(tr, from, edges_at_ends_beside(&t, &hashes, &asked), "build", verb, kind))
            } else if tr.up_to > 0.0 {
                format!("Together they {verb} at up to {}.", halved(tr.up_to, tr.settled, over))
            } else {
                format!("Together no size was found{}.", if tr.settled { String::new() } else { format!(": {over}") })
            }
        };
        let together = |budget: &mut Budget, from: f64| -> SizesRead {
            budget.ahead(per_try);
            let cell = std::cell::RefCell::new(budget);
            let mut works = |at: f64| {
                let mut bg = cell.borrow_mut();
                attempt(&mut bg, &hashes, at)
            };
            let mut go_on = || cell.borrow().affords();
            read_sizes(from, &mut works, &mut go_on)
        };
        let mut with_the_rest = false;
        let can: String;
        if odd.is_some() || up_to > 0.0 {
            let of = odd.map(|i| &reads[i]);
            let said = match of {
                Some((one, r)) => format!("{} {}.", if n == 1 { "It".to_string() } else { capital(what(one)) }, both_ways(r, size, edges_at_ends_beside(&t, &one.whole, &asked), "builds", verb, kind)),
                None => format!("{} {verb}{} at up to {}.", if n == 1 { "It" } else { "They" }, if n == 1 { "s" } else { "" }, halved(up_to, halved_all, &budget.over())),
            };
            let claim = match of {
                Some((_, r)) => nearest(&r.built, size),
                None => up_to,
            };
            let mut all = String::new();
            if m > 1 {
                budget.ahead(per_try);
                if budget.affords() {
                    let them = if m == 2 { "Both".to_string() } else { format!("All {m}") };
                    if attempt(&mut budget, &hashes, claim) {
                        all = format!(" {them} together build at {}.", mm(claim));
                    } else {
                        let tr = together(&mut budget, claim);
                        all = format!(" {them} together do not build at {}. {}", mm(claim), together_words(&tr, claim, &budget.over()));
                    }
                }
            }
            can = format!("{said}{read_on}{all}{}", if settled { String::new() } else { format!(" {TRY_IT}") });
        } else if m > 1 {
            let tr = together(&mut budget, size);
            with_the_rest = !tr.built.is_empty();
            let none_alone = settled && reads.iter().all(|(_, r)| r.built.is_empty());
            let floor = reads.first().map_or(size, |(_, r)| r.floor);
            can = if with_the_rest {
                let head = if settled { format!("Alone {} not {verb} even at {}, but", if n == 1 { "it does" } else if none_alone { "they do" } else { "some do" }, mm(floor)) } else { format!("Alone no size was found ({}), but", budget.over()) };
                format!("{head} they do with the rest of the feature. {}{read_on}", together_words(&tr, size, &budget.over()))
            } else if settled && tr.settled {
                format!("{}{read_on}", at_no_size(&fm, &t, &failing, n, floor, size, ", alone or together", verb))
            } else {
                unfound(if n == 1 { "it takes" } else { "they take" })
            };
        } else {
            let floor = reads.first().map_or(size, |(_, r)| r.floor);
            can = if settled { at_no_size(&fm, &t, &failing, n, floor, size, "", verb) } else { unfound(if n == 1 { "it takes" } else { "they take" }) };
        }
        let failing_whole: Vec<u32> = failing.iter().flat_map(|c| c.whole.clone()).collect();
        let beside = small_faces_beside(b, &t, &failing_whole, size);
        let creases = shallow_creases_at(&t, &failing_whole);
        let crease = if creases.is_empty() {
            String::new()
        } else {
            format!(" {} {} — two faces a few degrees apart, which a {verb} often cannot end on or run across: if the size is not the reason, make them one face (one profile curve in place of the two lines), or a clear corner.", if n == 1 { "It meets" } else { "They meet" }, creases.join("; "))
        };
        let why = if !beside.is_empty() {
            format!(" Beside {}: {} — a face too small for the {verb}; take it off, {verb} the rest first, or {verb} smaller there.", if n == 1 { "it" } else { "them" }, beside.join("; "))
        } else if !creases.is_empty() || odd.is_some() || with_the_rest {
            String::new()
        } else {
            format!(" The faces beside {} are too narrow for the {verb}, or meet at an angle the kernel cannot {verb}.", if n == 1 { "it" } else { "them" })
        } + &crease;
        let ones: Vec<Chain> = failing_sets.iter().map(|i| sets[*i].one.clone()).collect();
        let subject = |c: &Chain| if n == 1 { if c.whole.len() > 1 { "The chain".to_string() } else { "The edge".to_string() } } else { capital(what(c)) };
        let into = runs_into_text(b, &t, fm.as_ref(), &ones, verb, &subject);
        let picked: Vec<u32> = failing.iter().flat_map(|c| c.picked.clone()).collect();
        return Some(Diagnosis { message: format!("{which} at {}{tried}{as_one}{shown}. {can}{why}{into}", mm(size)), where_: drawn(&picked) });
    }
    if m < 2 || reached < m {
        return None;
    }
    // Every chain takes the size alone: it is where two meet that fails.
    budget.ahead(per_try);
    let read = {
        let cell = std::cell::RefCell::new(&mut budget);
        let mut works = |at: f64| {
            let mut bg = cell.borrow_mut();
            attempt(&mut bg, &hashes, at)
        };
        let mut go_on = || cell.borrow().affords();
        read_sizes(size, &mut works, &mut go_on)
    };
    let can = if not_by_size(&read) {
        format!("Together they {}.", both_ways(&read, size, edges_at_ends_beside(&t, &hashes, &asked), "build", verb, kind))
    } else if read.up_to > 0.0 {
        format!("Together they {verb} at up to {}{}; or {verb} them in two features.", mm(read.up_to), if read.settled { String::new() } else { format!(" (not halved further: {})", budget.over()) })
    } else {
        format!("{} them in two features{}.", if verb == "round" { "Round" } else { "Chamfer" }, if read.settled { String::new() } else { format!(" — the size they take together was not found: {}", budget.over()) })
    };
    let read_alone = if probed.len() < m { format!(" {} of the {m} were tried alone, the others alike.", probed.len()) } else { String::new() };
    Some(Diagnosis { message: format!("Each of the {m} {unit} {verb}s alone at {}; together they do not — where two meet, one runs into the other. {can}{read_alone}", mm(size)), where_: drawn(&hashes) })
}

/// Not even the floor: it is the edge, not the size — said with how far apart its faces stand
/// along it, and where they come closest to flat.
#[allow(clippy::too_many_arguments)]
fn at_no_size(fm: &Option<FaceMesh>, t: &Topology, failing: &[Chain], n: usize, floor: f64, asked: f64, how: &str, verb: &str) -> String {
    let whole: Vec<u32> = failing.iter().flat_map(|c| c.whole.clone()).collect();
    let apart = faces_apart_along(fm.as_ref(), t, &whole);
    let least = if floor < asked / 8.0 + 1e-9 && floor > LEAST_TRY_MM { "an eighth of the size asked" } else { "the least tried" };
    let along = match apart {
        Some((lo, hi, at)) => format!(
            ": {} two faces stand {} apart along {} (0° is flat), closest to flat at {}",
            if n == 1 { "its" } else { "their" },
            if (hi - lo).abs() < 0.05 { deg(lo) } else { format!("{}–{}", deg(lo), deg(hi)) },
            if n == 1 { "it" } else { "them" },
            point(at)
        ),
        None => String::new(),
    };
    format!("{} not {verb} even at {}, {least}{how} — it is the edge, not the size{along}.", if n == 1 { "It does" } else { "They do" }, mm(floor))
}

/// Outside rounds of a body no larger than a wall: round faces bulging out whose radius the wall
/// reaches — a shell's inside has nowhere to go there (ScanRuler's kernel `collapsingRounds`,
/// read off the faces' fingerprints).
fn collapsing_rounds(t: &Topology, fm: Option<&FaceMesh>, wall: f64) -> Vec<u32> {
    t.faces
        .iter()
        .filter(|f| matches!(f.fingerprint.kind, FaceKind::Cylinder | FaceKind::Sphere) && f.fingerprint.radius.is_some_and(|r| r <= wall + 1e-9))
        .filter(|f| {
            let Some(c) = f.fingerprint.center else { return false };
            let out = match f.fingerprint.kind {
                FaceKind::Cylinder => {
                    let a = f.fingerprint.dir.normalize_or_zero();
                    let d = f.point - c;
                    d - a * d.dot(a)
                }
                _ => f.point - c,
            };
            let n = fm.and_then(|m| m.normals(f.hash as usize, &[f.point]).first().copied().flatten());
            n.is_some_and(|n| n.dot(out) > 0.0)
        })
        .map(|f| f.hash)
        .collect()
}

/// Why a shell the kernel refused could not be made, within a budget (ScanRuler's
/// `diagnoseShell`, without the thin spots and the faces offset alone): an outside round no
/// larger than the wall; whether the wall fits with no face opened; the largest wall, by
/// halving; the opened faces one at a time; and together.
fn diagnose_shell(b: &mut Builder, x: &ShellFeature, limit: (f64, f64), refusal: &str) -> Option<Diagnosis> {
    let quiet = std::mem::take(&mut b.warnings);
    let found = b.find_faces(x.body, &x.faces, &x.faces_where);
    b.warnings = quiet;
    let found = found.ok()?;
    let state = b.body(x.body)?.clone();
    if state.surface {
        return None;
    }
    let t = state.topology();
    let fm = FaceMesh::of(&state.shape);
    let wall = x.thickness;
    let mut budget = Budget::new(limit.0, limit.1);
    let shape = state.shape.clone();
    let points = |faces: &[u32]| -> Vec<SVec3> { faces.iter().filter_map(|h| t.face(*h)).map(|f| sv3(f.point)).collect() };
    let mut attempt = |budget: &mut Budget, faces: &[u32], at: f64| -> bool {
        let pts = points(faces);
        budget.timed(|| kernel::guard("diagnosis", || kernel::shell(&shape, &pts, at)).is_ok())
    };
    let outlines = |faces: &[u32]| -> Vec<Vec<DVec3>> { t.edges.iter().filter(|e| e.faces.iter().any(|h| faces.contains(h)) && e.polyline.len() >= 2).map(|e| e.polyline.clone()).collect() };
    let halved = |budget: &mut Budget, attempt: &mut dyn FnMut(&mut Budget, &[u32], f64) -> bool, faces: &[u32]| -> (f64, bool) {
        let cell = std::cell::RefCell::new(budget);
        let mut works = |at: f64| {
            let mut bg = cell.borrow_mut();
            attempt(&mut bg, faces, at)
        };
        let mut go_on = || cell.borrow().affords();
        largest_size_while(wall, &mut works, &mut go_on, 0.0)
    };
    let hollows_at = |(size, settled): (f64, bool), over: &str| -> String {
        if settled {
            if size > 0.0 { format!("It hollows at a wall of up to {}.", mm(size)) } else { "It does not hollow at any wall.".to_string() }
        } else if size > 0.0 {
            format!("It hollows at a wall of {} at least — not bisected further: {over}.", mm(size))
        } else {
            format!("Not bisected: {over}.")
        }
    };
    let opened: Vec<u32> = found.iter().map(|i| *i as u32).collect();
    let m = opened.len();
    let rounds = collapsing_rounds(&t, fm.as_ref(), wall);
    // Whether the wall fits the body with no face opened — the kernel's shell opens one at
    // least, so with none opened that is the step's own try, failed already.
    let closed = if m == 0 { Some(false) } else if budget.affords() { Some(attempt(&mut budget, &[], wall)) } else { None };
    if closed == Some(false) || (closed.is_none() && !rounds.is_empty()) {
        let fits = format!("A wall of {} does not fit the body{}", mm(wall), if closed == Some(false) && m > 0 { " even with no face opened" } else { "" });
        if !rounds.is_empty() {
            let n = rounds.len();
            let mut named: Vec<u32> = rounds.clone();
            named.sort_by(|a, c| t.face(*a).and_then(|f| f.fingerprint.radius).unwrap_or(f64::INFINITY).total_cmp(&t.face(*c).and_then(|f| f.fingerprint.radius).unwrap_or(f64::INFINITY)));
            let listed: Vec<String> = named.iter().take(3).map(|h| {
                let f = t.face(*h);
                let what = f.map_or(String::new(), |f| format!("{}{}", f.fingerprint.kind.name(), f.fingerprint.radius.map_or(String::new(), |r| format!(" of {} radius", mm(r)))));
                let made = f.filter(|f| f.fingerprint.origin >= 0).map_or(String::new(), |f| format!(", made by feature {}", f.fingerprint.origin));
                format!("face {h} ({what}{made})")
            }).collect();
            let more = if n > 3 { format!(" and {} more", n - 3) } else { String::new() };
            let why = format!("{} no larger than the wall — {}{more}, drawn red on the part — and the wall could not be cut back behind {}", if n == 1 { "an outside round of the body is".to_string() } else { format!("{n} outside rounds of the body are") }, listed.join(", "), if n == 1 { "it" } else { "them" });
            let up = halved(&mut budget, &mut attempt, &[]);
            return Some(Diagnosis { message: format!("{fits}: {why}. {}", hollows_at(up, &budget.over())), where_: outlines(&rounds) });
        }
        let up = halved(&mut budget, &mut attempt, &[]);
        let failed = if refusal.is_empty() { String::new() } else { format!(" ({refusal})") };
        return Some(Diagnosis { message: format!("{fits}: the kernel's offset failed{failed} and no outside round of the body is no larger than the wall — so it fails where faces meet: an edge or a corner the offset faces cannot be joined at, or a face narrower than the wall. {}", hollows_at(up, &budget.over())), where_: Vec::new() });
    }
    if closed.is_none() {
        return Some(Diagnosis { message: format!("The shell at a wall of {} failed: no outside round of the body is no larger than the wall; whether the wall fits with no face opened, and the faces opened one at a time, were not tried: {}.", mm(wall), budget.over()), where_: Vec::new() });
    }
    // The wall fits closed: it is the faces opened.
    let probed: Vec<u32> = opened.iter().copied().take(DIAGNOSE_PICKS).collect();
    let mut tried: Vec<u32> = Vec::new();
    let mut failing: Vec<u32> = Vec::new();
    for h in &probed {
        if !budget.affords() {
            break;
        }
        tried.push(*h);
        if !attempt(&mut budget, &[*h], wall) {
            failing.push(*h);
        }
    }
    let cut = tried.len() < probed.len();
    let short = if cut { format!(" (the first {} tried; the rest not: {})", tried.len(), budget.over()) } else if probed.len() < m { format!(" (the first {} tried)", probed.len()) } else { String::new() };
    if !failing.is_empty() {
        let n = failing.len();
        let each: Vec<(f64, bool)> = failing.iter().map(|h| halved(&mut budget, &mut attempt, &[*h])).collect();
        let up_to = each.iter().map(|x| x.0).fold(f64::INFINITY, f64::min);
        let which = if m == 1 { "The face cannot be opened".to_string() } else if n == m { format!("None of the {m} faces can be opened") } else { format!("{n} of the {m} faces cannot be opened") };
        let shown = if n < tried.len() { " — outlined red on the part; the others tried can" } else { " — outlined red on the part" };
        let can = if each.iter().any(|x| !x.1) {
            format!("{} largest wall was not bisected: {}.", if n == 1 { "Its" } else { "Their" }, budget.over())
        } else if up_to > 0.0 {
            format!("{} at a wall of up to {}.", if n == 1 { "It opens" } else { "They open" }, mm(up_to))
        } else {
            format!("{} not open at any wall — the wall cannot follow the edges round {}; open a neighbouring face as well, or leave {} closed.", if n == 1 { "It does" } else { "They do" }, if n == 1 { "the face" } else { "them" }, if n == 1 { "it" } else { "them" })
        };
        return Some(Diagnosis { message: format!("{which} at a wall of {}{short}{shown}. {can}", mm(wall)), where_: outlines(&failing) });
    }
    if cut {
        return Some(Diagnosis { message: format!("A wall of {} fits the body closed, and the first {} of the {m} faces open alone at it; the rest were not tried: {}.", mm(wall), tried.len(), budget.over()), where_: Vec::new() });
    }
    if m < 2 || probed.len() < m {
        return None;
    }
    let together = halved(&mut budget, &mut attempt, &opened);
    let can = if !together.1 { format!("Together they were not bisected: {}.", budget.over()) } else if together.0 > 0.0 { format!("Together they open at a wall of up to {}.", mm(together.0)) } else { "Open them in two shells, or leave one closed.".to_string() };
    Some(Diagnosis { message: format!("Each of the {m} faces opens alone at a wall of {}; together they do not. {can}", mm(wall)), where_: outlines(&opened) })
}

/// The edges a round or a chamfer the kernel crashed on handed it, said: how many, between the
/// faces of which features, and where the first of them lie (ScanRuler's `trapEdgesText`).
fn trap_edges_text(b: &Builder, verb: &str, t: &Topology, hashes: &[u32], picked: &[u32]) -> String {
    let mut seen: Vec<u32> = Vec::new();
    let mut mids: Vec<DVec3> = Vec::new();
    let mut pairs: Vec<(Vec<i64>, usize)> = Vec::new();
    let mut own_makers: HashSet<i64> = HashSet::new();
    let mut run_makers: Vec<i64> = Vec::new();
    let mut run_on = 0;
    for h in hashes {
        if seen.contains(h) {
            continue;
        }
        seen.push(*h);
        let Some(e) = t.edge(*h) else { continue };
        if mids.len() < TRAP_MIDS {
            mids.push(e.fingerprint.mid);
        }
        let mut made: Vec<i64> = e.faces.iter().collect::<HashSet<_>>().into_iter().map(|f| t.face(*f).map_or(-1, |x| x.fingerprint.origin)).collect();
        made.sort_unstable();
        match pairs.iter_mut().find(|(k, _)| *k == made) {
            Some((_, n)) => *n += 1,
            None => pairs.push((made.clone(), 1)),
        }
        if picked.contains(h) {
            own_makers.extend(made);
        } else {
            run_on += 1;
            for m in made {
                if !run_makers.contains(&m) {
                    run_makers.push(m);
                }
            }
        }
    }
    pairs.sort_by(|a, b| b.1.cmp(&a.1));
    let name = |id: i64| if id < 0 { "nothing known".to_string() } else { u32::try_from(id).ok().and_then(|i| b.track.name_of(i)).map_or_else(|| format!("feature {id}"), str::to_string) };
    let sides: Vec<String> = pairs.iter().map(|(m, c)| if m.len() == 1 { format!("{} alone ({c})", name(m[0])) } else { format!("{} and {} ({c})", name(m[0]), name(m[1])) }).collect();
    let at: Vec<String> = mids.iter().map(|p| format!("({}, {}, {})", to_fixed(p.x, 2), to_fixed(p.y, 2), to_fixed(p.z, 2))).collect();
    let count = seen.len();
    let ran_on = if run_on > 0 {
        let fresh: Vec<i64> = run_makers.iter().copied().filter(|m| !own_makers.contains(m)).collect();
        let mut makers = if fresh.is_empty() { run_makers.clone() } else { fresh };
        makers.sort_unstable();
        format!(" {run_on} of them, not picked, {} on tangent into faces made by {}.", if run_on == 1 { "runs" } else { "run" }, makers.iter().map(|m| name(*m)).collect::<Vec<_>>().join(", "))
    } else {
        String::new()
    };
    format!("It was to {verb} {count} edge{} between faces made by {}, at {}{}.{ran_on}", if count == 1 { "" } else { "s" }, sides.join(", "), at.join(", "), if count > mids.len() { " …" } else { "" })
}

/// Why a failed step may have failed through no fault of its own: the body it works on already
/// fails the kernel's check (ScanRuler's `inputFault`) — said with the step that last changed
/// it. `None` when it passes, or the step works on no body.
fn input_fault(b: &Builder, f: &Feature) -> Option<String> {
    let body = f.kind.body()?;
    let state = b.body(body)?;
    if state.surface {
        return None;
    }
    let problems = kernel::guard("check", || Ok(state.shape.validity())).unwrap_or_default();
    let first = problems.first()?;
    let id = b.resolve(body);
    // The last feature before this one that changed the body and built, else its maker.
    let mut last: FeatureId = id.maker();
    let touched: Vec<FeatureId> = b.out.touched.iter().filter(|(fid, ids)| **fid != f.id && ids.contains(&id) && !b.out.errors.contains_key(fid)).map(|(fid, _)| *fid).collect();
    if let Some(m) = touched.into_iter().max() {
        last = last.max(m);
    }
    let name = b.track.name_of(last).map_or_else(|| format!("feature {last}"), str::to_string);
    Some(format!("the body it works on already fails the kernel check (last changed by {name}): mend that step — the check says {first}"))
}

/// A step's failure as its row says it (ScanRuler's `fail`): the body it works on blamed when
/// it already fails the kernel's check; else a round's, a chamfer's or a shell's diagnosis,
/// the kernel's own words after it where the diagnosis does not quote them; a crash said with
/// the edges it was given.
pub(crate) fn diagnosed(b: &mut Builder, _part: &Part, f: &Feature, message: String, step_time: Duration) -> String {
    let step_ms = step_time.as_secs_f64() * 1000.0;
    let applies = matches!(f.kind, FeatureKind::Fillet(_) | FeatureKind::Chamfer(_) | FeatureKind::Shell(_) | FeatureKind::Draft(_) | FeatureKind::OffsetBody(_));
    // A failure that is not the kernel's to explain — a pick gone, a body gone, a size not
    // set, a kind not built yet — is said as it is.
    let kernels = message.starts_with("the operation failed") || message.starts_with("kernel internal error") || message.starts_with("invalid input");
    if !applies || !kernels {
        return message;
    }
    if let Some(input) = input_fault(b, f) {
        return format!("{} — {input}.", message.trim_end_matches('.'));
    }
    let crashed = message.starts_with("kernel internal error");
    let round_budget = (ROUND_DIAGNOSIS_LEAST_MS.max(ROUND_DIAGNOSIS_TIMES * step_ms), step_ms);
    let shell_budget = (SHELL_DIAGNOSIS_LEAST_MS.max(SHELL_DIAGNOSIS_TIMES * step_ms), step_ms);
    let told = without_call_name(&message);
    let d = match &f.kind {
        FeatureKind::Fillet(x) if x.variable.is_empty() && !crashed => diagnose_round(b, Blend::Fillet(x), x.body, (&x.edges, &x.edges_where), x.chain, round_budget),
        FeatureKind::Chamfer(x) if !crashed => diagnose_round(b, Blend::Chamfer(x), x.body, (&x.edges, &x.edges_where), x.chain, round_budget),
        FeatureKind::Shell(x) if !crashed => diagnose_shell(b, x, shell_budget, &said(&message)),
        _ => None,
    };
    if crashed {
        // A crash says which edges it was given: the set to change.
        let edges = match &f.kind {
            FeatureKind::Fillet(x) => Some(("round", x.body, &x.edges, &x.edges_where, x.chain)),
            FeatureKind::Chamfer(x) => Some(("bevel", x.body, &x.edges, &x.edges_where, x.chain)),
            _ => None,
        };
        let where_ = edges.and_then(|(verb, body, e, w, chain)| {
            let quiet = std::mem::take(&mut b.warnings);
            let found = b.find_edges(body, e, w);
            b.warnings = quiet;
            let found = found.ok()?;
            let t = b.topology_of(body)?;
            let picked: Vec<u32> = found.iter().map(|i| *i as u32).collect();
            let hashes = if chain { tangent_chain(&t.edges, &picked, &split_edges(&t), CHAIN_ANGLE, CHAIN_JOIN_MM) } else { picked.clone() };
            Some(format!(" {}", trap_edges_text(b, verb, &t, &hashes, &picked)))
        });
        return format!("The CAD kernel crashed on this step ({}); the rest of the part was built without it.{}", said(&message), where_.unwrap_or_default());
    }
    if let Some(d) = d.as_ref().filter(|d| !d.where_.is_empty()) {
        b.out.failed_at.insert(f.id, d.where_.clone());
    }
    match d {
        Some(d) => {
            let s = said(&message);
            if !s.is_empty() && d.message.contains(&s) {
                d.message
            } else {
                format!("{} ({told})", d.message)
            }
        }
        None => told,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halves_to_the_largest_size_that_holds() {
        let mut tries = 0;
        let (s, settled) = largest_size_while(4.0, &mut |at| {
            tries += 1;
            at <= 1.3
        }, &mut || true, 0.0);
        assert!(settled);
        assert!((s - 1.3).abs() < 0.01, "{s}");
        assert!(tries >= LEAST_HALVINGS);
        // With a floor nothing under it is tried, and the floor last when nothing above it held.
        let (s, _) = largest_size_while(4.0, &mut |at| at <= 0.5, &mut || true, 0.5);
        assert_eq!(s, 0.5);
        let (s, _) = largest_size_while(4.0, &mut |_| false, &mut || true, 0.5);
        assert_eq!(s, 0.0);
        // Stopped by the budget: not settled.
        let mut n = 0;
        let (_, settled) = largest_size_while(4.0, &mut |_| true, &mut || {
            n += 1;
            n < 3
        }, 0.0);
        assert!(!settled);
    }

    #[test]
    fn reads_sizes_both_ways_and_tells_a_corner_from_a_size() {
        // Builds above the size asked: not a size limit.
        let r = read_sizes(2.0, &mut |at| at >= 2.1 || at < 1.0, &mut || true);
        assert!(not_by_size(&r));
        let words = both_ways(&r, 2.0, 1, "builds", "round", "fillet");
        assert!(words.starts_with("builds at "), "{words}");
        assert!(words.contains("not a size limit: the corner where the rounds meet; round the edge there not in this fillet with it"), "{words}");
        // Too large plainly.
        let r = read_sizes(2.0, &mut |at| at <= 1.0, &mut || true);
        assert!(!not_by_size(&r));
        assert!((r.up_to - 1.0).abs() < 0.02, "{}", r.up_to);
    }

    #[test]
    fn strips_the_kernels_call_name() {
        assert_eq!(without_call_name("Filleting: filletWithHistory: produced an invalid solid"), "Filleting: produced an invalid solid");
        assert_eq!(without_call_name("the operation failed: not supported yet: fillet"), "the operation failed: not supported yet: fillet");
        assert_eq!(said("Shelling: the wall is too thick."), "the wall is too thick");
    }
}
