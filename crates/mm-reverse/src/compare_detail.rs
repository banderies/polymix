// SPDX-License-Identifier: AGPL-3.0-only
//! The comparison of the scan with the bodies, told by where. Ported from ScanRuler
//! (`plugins/reverse/core/compareDetail.ts`, the comparison of `app/reBodyCommands.ts` and
//! `app/reCompareFeature.ts`), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! Beside the live map's field, which face of which body each scan vertex lies nearest, so the
//! reading can be grouped by face — and by the feature that made the face — and the worst spots
//! named with a place to look; the patches over tolerance, each on one face; and the scan the
//! field does not hold (past the reach of every body: a hole not cut, a boss not modelled),
//! told by side. A single RMS over the part says that something is off; this says what.
//!
//! The answers are given as JSON in the shape ScanRuler's `reverse.body.compare` answers in, for
//! the agent commands and the script steps to hand on.

use std::collections::{HashMap, HashSet};

use glam::DVec3;
use serde_json::{json, Map, Value};

use crate::build::BodyMesh;
use crate::compare::{live_map_reach, vertices_near_box, Read, ScanSide, SideCounts, SideReader, SparseField, Surface, SurfaceBody, Unfacing, DEFAULT_FACING_DEG};
use crate::hotspots::{hotspots, Hotspot, HotspotOptions, PatchKind, FEATURE_DEPTH};
use crate::model::{BodyId, Feature, FeatureId};
use crate::topo::{FaceInfo, Topology};

/// A body as a comparison reads it.
pub struct CompareBody<'a> {
    pub id: BodyId,
    pub name: &'a str,
    pub mesh: &'a BodyMesh,
    pub sheet: bool,
    pub topology: Option<&'a Topology>,
}

/// One face of one body, as the scan reads against it.
#[derive(Debug, Clone, Default)]
pub struct FaceReading {
    /// The body, by its index among those compared.
    pub body: usize,
    pub face_hash: i64,
    pub read: usize,
    pub within: usize,
    pub sum: f64,
    pub sum_sq: f64,
    pub outside: f64,
    pub inside: f64,
}

/// A place where the scan lies farthest from the bodies.
#[derive(Debug, Clone)]
pub struct WorstSpot {
    pub vertex: u32,
    pub at: DVec3,
    pub value: f64,
    pub body: usize,
    pub face_hash: i64,
}

/// A patch of the scan over tolerance, told against the bodies.
#[derive(Debug, Clone)]
pub struct PatchReading {
    pub patch: Hotspot,
    pub body: Option<usize>,
    pub face_hash: Option<i64>,
    pub vertex: u32,
    pub vertices: Vec<u32>,
}

/// One side of the scan past the reach.
#[derive(Debug, Clone, Default)]
pub struct FarSide {
    pub count: usize,
    pub patches: Vec<PatchReading>,
    pub found: usize,
}

/// The scan the field does not hold.
#[derive(Debug, Clone, Default)]
pub struct FarReading {
    pub reach: f64,
    pub count: usize,
    pub inside: FarSide,
    pub outside: FarSide,
}

/// How a comparison's patches are asked for.
#[derive(Debug, Clone, Default)]
pub struct PatchRequest {
    pub count: usize,
    pub tolerance: f64,
    pub grain: Option<f64>,
    /// Only the scan nearest these faces — each body's (by index) by their hashes.
    pub faces: Option<Vec<(usize, Vec<i64>)>>,
}

#[derive(Debug, Clone, Default)]
pub struct CompareDetail {
    pub field: SparseField,
    pub faces: Vec<FaceReading>,
    pub spots: Vec<WorstSpot>,
    pub patches: Vec<PatchReading>,
    pub patches_found: usize,
    pub over: usize,
    pub spacing: f64,
    pub features_count: usize,
    pub features_share: f64,
    pub far: Option<FarReading>,
    pub side: Option<SideCounts>,
}

/// How many of a patch's vertices are named.
const PATCH_VERTICES: usize = 12;

fn named_vertices(members: &[usize], subset: &[u32]) -> Vec<u32> {
    let stride = (members.len() / PATCH_VERTICES).max(1);
    let mut out = Vec::new();
    let mut k = 0;
    while k < members.len() && out.len() < PATCH_VERTICES {
        out.push(subset[members[k]]);
        k += stride;
    }
    out
}

/// The scan against the bodies, in detail: the field of signed distances for the vertices
/// within `reach`, each vertex's nearest face, and the `spot_count` worst places no two nearer
/// than `apart` — on the faces `patches.faces` names, when it names any.
#[allow(clippy::too_many_arguments)]
pub fn compare_bodies_detail(bodies: &[CompareBody], points: &[DVec3], side: Option<ScanSide>, range: f64, reach: f64, spot_count: usize, apart: f64, patches: Option<&PatchRequest>, within: Option<(DVec3, DVec3)>, far: Option<f64>) -> CompareDetail {
    let list: Vec<SurfaceBody> = bodies.iter().enumerate().map(|(i, b)| SurfaceBody { tag: i as u64, mesh: b.mesh, sheet: b.sheet }).collect();
    let Some(surface) = Surface::of_bodies(&list) else { return CompareDetail::default() };
    let in_box = |p: DVec3| within.is_none_or(|(lo, hi)| !(p.x < lo.x || p.x > hi.x || p.y < lo.y || p.y > hi.y || p.z < lo.z || p.z > hi.z));
    let near = vertices_near_box(points, surface.bbox_min, surface.bbox_max, reach);
    let reader = SideReader::new(&surface, points, side, &near, range);
    let mut counts = SideCounts::default();
    let want_far = far.is_some_and(|f| f > 0.0);
    let mut read_mark = if want_far { vec![false; points.len()] } else { Vec::new() };
    let mut subset = Vec::new();
    let mut values: Vec<f64> = Vec::new();
    let mut tri_of = Vec::new();
    let mut spot_at = Vec::new();
    for v in &near {
        let p = points[*v as usize];
        if !in_box(p) {
            continue;
        }
        let took = reader.read(*v as usize, p, reach, reach, &mut counts);
        let hit = match took {
            Read::Out => {
                if want_far {
                    read_mark[*v as usize] = true;
                }
                continue;
            }
            Read::None => continue,
            Read::Read(h) => h,
        };
        if !(hit.signed.abs() <= reach) {
            continue;
        }
        if want_far {
            read_mark[*v as usize] = true;
        }
        subset.push(*v);
        values.push(hit.signed);
        tri_of.push(hit.tri);
        spot_at.push(p);
    }
    let w = subset.len();
    let body_at = |i: usize| surface.body_of.get(tri_of[i]).copied().unwrap_or(0) as usize;
    let face_at = |i: usize| surface.face_of.get(tri_of[i]).copied().unwrap_or(-1);
    // By face.
    let mut rows: Vec<FaceReading> = Vec::new();
    let mut row_of: HashMap<(usize, i64), usize> = HashMap::new();
    for i in 0..w {
        let key = (body_at(i), face_at(i));
        let r = *row_of.entry(key).or_insert_with(|| {
            rows.push(FaceReading { body: key.0, face_hash: key.1, ..Default::default() });
            rows.len() - 1
        });
        let d = values[i];
        let row = &mut rows[r];
        row.read += 1;
        if d.abs() <= range {
            row.within += 1;
        }
        row.sum += d;
        row.sum_sq += d * d;
        if d > row.outside {
            row.outside = d;
        }
        if d < row.inside {
            row.inside = d;
        }
    }
    let kept: Option<HashSet<(usize, i64)>> = patches.and_then(|p| p.faces.as_ref()).map(|f| f.iter().flat_map(|(b, hs)| hs.iter().map(move |h| (*b, *h))).collect());
    let on_kept = |i: usize| kept.as_ref().is_none_or(|k| k.contains(&(body_at(i), face_at(i))));
    // The worst spots.
    let mut order: Vec<usize> = (0..w).collect();
    order.sort_by(|a, b| values[*b].abs().total_cmp(&values[*a].abs()));
    let mut spots: Vec<WorstSpot> = Vec::new();
    for i in order {
        if spots.len() >= spot_count {
            break;
        }
        if !on_kept(i) {
            continue;
        }
        let at = spot_at[i];
        if spots.iter().any(|s| s.at.distance_squared(at) < apart * apart) {
            continue;
        }
        spots.push(WorstSpot { vertex: subset[i], at, value: values[i], body: body_at(i), face_hash: face_at(i) });
    }
    // The patches over tolerance, each on one face.
    let mut found = Vec::new();
    let (mut patches_found, mut over, mut spacing, mut features_count, mut features_share) = (0, 0, 0.0, 0, 0.0);
    if let Some(req) = patches {
        let mut keys: HashMap<(usize, i64), u64> = HashMap::new();
        let key_list: Vec<u64> = (0..w)
            .map(|i| {
                let k = (body_at(i), face_at(i));
                let n = keys.len() as u64;
                *keys.entry(k).or_insert(n)
            })
            .collect();
        let group = |i: usize| key_list[i];
        let only = |i: usize| on_kept(i);
        let opts = HotspotOptions { grain: req.grain, limit: req.count, group: Some(&group), only: if kept.is_some() { Some(&only) } else { None }, ..Default::default() };
        let report = hotspots(&spot_at, &values, req.tolerance, &opts);
        patches_found = report.found;
        over = report.over;
        spacing = report.spacing;
        features_count = report.features_count;
        features_share = report.features_share_of_excess;
        for p in report.patches {
            let vertices = named_vertices(&p.members, &subset);
            let (b, f) = (body_at(p.index), face_at(p.index));
            let vertex = subset[p.index];
            found.push(PatchReading { patch: p, body: Some(b), face_hash: Some(f), vertex, vertices });
        }
    }
    // The scan the field does not hold.
    let mut far_reading = None;
    if let (true, Some(far)) = (want_far, far) {
        let lo = surface.bbox_min - DVec3::splat(far);
        let hi = surface.bbox_max + DVec3::splat(far);
        let mut far_subset = Vec::new();
        let mut far_values = Vec::new();
        let mut far_tris: Vec<Option<usize>> = Vec::new();
        let mut far_at = Vec::new();
        for (v, p) in points.iter().enumerate() {
            if read_mark[v] || !in_box(*p) {
                continue;
            }
            let mut value = far;
            let mut tri = None;
            let near_box = !(p.x < lo.x || p.x > hi.x || p.y < lo.y || p.y > hi.y || p.z < lo.z || p.z > hi.z);
            if near_box {
                if let Some(h) = surface.closest(*p, far) {
                    if h.signed.abs() <= far {
                        if !reader.keeps_far(v, &h) {
                            continue;
                        }
                        value = h.signed;
                        tri = Some(h.tri);
                    }
                }
            }
            far_subset.push(v as u32);
            far_values.push(value);
            far_tris.push(tri);
            far_at.push(*p);
        }
        let side_of = |inside: bool| -> FarSide {
            let want = |i: usize| (far_values[i] < 0.0) == inside;
            let count = (0..far_values.len()).filter(|i| want(*i)).count();
            let Some(req) = patches else { return FarSide { count, ..Default::default() } };
            if count == 0 {
                return FarSide { count, ..Default::default() };
            }
            let opts = HotspotOptions { grain: req.grain, limit: req.count, feature_depth: Some(FEATURE_DEPTH * req.tolerance), only: Some(&want), ..Default::default() };
            let report = hotspots(&far_at, &far_values, 0.0, &opts);
            let list = report
                .patches
                .into_iter()
                .map(|h| {
                    let t = far_tris[h.index];
                    let vertices = named_vertices(&h.members, &far_subset);
                    let vertex = far_subset[h.index];
                    PatchReading { body: t.map(|t| surface.body_of[t] as usize), face_hash: t.map(|t| surface.face_of[t]), vertex, vertices, patch: h }
                })
                .collect();
            FarSide { count, patches: list, found: report.found }
        };
        far_reading = Some(FarReading { reach: far, count: far_values.len(), inside: side_of(true), outside: side_of(false) });
    }
    CompareDetail {
        field: SparseField { subset, values: values.iter().map(|v| *v as f32).collect(), far_inside: None },
        faces: rows,
        spots,
        patches: found,
        patches_found,
        over,
        spacing,
        features_count,
        features_share,
        far: far_reading,
        side: side.map(|_| counts),
    }
}

// ---- the answer (app/reBodyCommands.ts compare, compareBodies, scoreOf) ----

/// How many worst spots, faces, patches and features a comparison names unless told otherwise.
pub const COMPARE_SPOTS: usize = 8;
pub const COMPARE_FACES: usize = 12;
pub const COMPARE_PATCHES: usize = 8;
pub const COMPARE_FEATURES: usize = 8;
/// How far past the reach the scan is still looked for, mm.
pub const COMPARE_FAR_MM: f64 = 25.0;
/// What detail "score" names unless told otherwise.
pub const SCORE_PATCHES: usize = 5;
pub const SCORE_FAR: usize = 2;
pub const SCORE_FEATURES: usize = 3;

const FEATURES_NOTE: &str = "each feature patch is material missing or misplaced: model it or name it";

/// What a comparison is asked.
#[derive(Debug, Clone)]
pub struct CompareInput {
    pub range: f64,
    pub spots: usize,
    pub faces: usize,
    pub patches: usize,
    pub features: usize,
    pub patch_vertices: bool,
    pub tolerance: Option<f64>,
    pub grain: Option<f64>,
    pub within: Option<(DVec3, DVec3)>,
    pub far: f64,
    /// The facing limit: `None` the default, `Some(None)` the nearest whatever it faces.
    pub facing_deg: Option<Option<f64>>,
    pub unfacing: Option<Unfacing>,
    pub skin_outer: bool,
}

impl CompareInput {
    pub fn new(range: f64) -> CompareInput {
        CompareInput { range, spots: COMPARE_SPOTS, faces: COMPARE_FACES, patches: COMPARE_PATCHES, features: COMPARE_FEATURES, patch_vertices: false, tolerance: None, grain: None, within: None, far: COMPARE_FAR_MM, facing_deg: None, unfacing: None, skin_outer: false }
    }

    /// As detail "score" asks: no spots or face rows, the score's patches and features.
    pub fn score(range: f64) -> CompareInput {
        CompareInput { spots: 0, faces: 0, patches: SCORE_PATCHES, features: SCORE_FEATURES, ..CompareInput::new(range) }
    }
}

pub fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

pub fn round4(x: f64) -> f64 {
    (x * 10000.0).round() / 10000.0
}

fn v3(p: DVec3) -> Value {
    json!([round3(p.x), round3(p.y), round3(p.z)])
}

/// A face of a compared body by hash, as an answer names it: its index, its kind and the
/// feature that made it.
pub fn face_out(topology: Option<&Topology>, hash: i64, features: &[Feature]) -> Value {
    let Some(t) = topology else { return Value::Null };
    let Some(i) = t.faces.iter().position(|f| f.hash as i64 == hash) else { return Value::Null };
    let fp = &t.faces[i].fingerprint;
    json!({"index": i, "kind": fp.kind.name(), "madeBy": {"id": fp.origin, "name": feature_name(features, fp.origin)}})
}

fn feature_name(features: &[Feature], id: i64) -> Value {
    features.iter().find(|f| f.id as i64 == id).map_or(Value::Null, |f| Value::String(f.name.clone()))
}

/// A patch's figures as every answer with patches gives them.
pub fn patch_figures(p: &Hotspot) -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("centroid".into(), v3(p.centroid));
    m.insert("box".into(), json!({"min": v3(p.min), "max": v3(p.max)}));
    m.insert("area".into(), json!(round3(p.area)));
    m.insert("count".into(), json!(p.count));
    m.insert("mean".into(), json!(round3(p.mean)));
    m.insert("extreme".into(), json!(round3(p.extreme)));
    m.insert("depth".into(), json!(round3(p.depth)));
    m.insert("volume".into(), json!(round3(p.volume)));
    m.insert("kind".into(), json!(p.kind.name()));
    m
}

fn features_out(count: usize, share: f64) -> Value {
    let mut m = json!({"count": count, "shareOfExcess": round3(share)});
    if count > 0 {
        m["note"] = json!(FEATURES_NOTE);
    }
    m
}

/// The bodies against the scan — ScanRuler's `compareBodies`. `points` and `normals` are the
/// scan's vertices (normals may be empty: the scan is then read against the nearest surface).
pub fn compare_answer(bodies: &[CompareBody], features: &[Feature], points: &[DVec3], normals: &[DVec3], input: &CompareInput, on: Option<Vec<(usize, Vec<i64>)>>) -> Result<Value, String> {
    if bodies.is_empty() {
        return Err("There is no solid body to compare — build one first.".into());
    }
    let within = input.range;
    let facing = input.facing_deg.unwrap_or(Some(DEFAULT_FACING_DEG));
    let side = (!normals.is_empty() && normals.len() == points.len()).then_some(ScanSide { normals, facing_deg: facing, unfacing: input.unfacing.unwrap_or_default(), skin_outer: input.skin_outer });
    let tolerance = input.tolerance.unwrap_or(within);
    let req = PatchRequest { count: input.patches, tolerance, grain: input.grain, faces: on };
    let reach = live_map_reach(within);
    let detail = compare_bodies_detail(bodies, points, side, within, reach, input.spots, reach.max(1.0), Some(&req), input.within, (input.far > 0.0).then_some(input.far));
    let v = &detail.field.values;
    let n = v.len();
    let (mut sum, mut sum_sq, mut outside, mut inside, mut within_count) = (0.0f64, 0.0f64, f64::NEG_INFINITY, f64::INFINITY, 0);
    for d in v.iter().map(|x| *x as f64) {
        sum += d;
        sum_sq += d * d;
        outside = outside.max(d);
        inside = inside.min(d);
        if d.abs() <= within {
            within_count += 1;
        }
    }
    let mut abs: Vec<f64> = v.iter().map(|x| (*x as f64).abs()).collect();
    abs.sort_by(f64::total_cmp);
    let pct = |p: f64| if n > 0 { json!(abs[(n - 1).min((p * n as f64).floor() as usize)]) } else { Value::Null };
    let name = |i: usize| bodies.get(i).map_or(String::new(), |b| b.name.to_string());
    let id = |i: usize| bodies.get(i).map_or(Value::Null, |b| json!(b.id.as_number()));
    let face = |i: usize, h: i64| face_out(bodies.get(i).and_then(|b| b.topology), h, features);
    // Rows by face, the farthest first.
    struct Row {
        body: usize,
        face: Value,
        read: usize,
        within: usize,
        sum: f64,
        sum_sq: f64,
        outside: f64,
        inside: f64,
    }
    let mut rows: Vec<Row> = detail.faces.iter().map(|r| Row { body: r.body, face: face(r.body, r.face_hash), read: r.read, within: r.within, sum: r.sum, sum_sq: r.sum_sq, outside: round3(r.outside), inside: round3(r.inside) }).collect();
    rows.sort_by(|a, b| b.outside.max(-b.inside).total_cmp(&a.outside.max(-a.inside)));
    // By feature.
    struct Group {
        id: i64,
        name: Value,
        faces: usize,
        read: usize,
        within: usize,
        sum: f64,
        sum_sq: f64,
        outside: f64,
        inside: f64,
    }
    let mut groups: Vec<Group> = Vec::new();
    for r in &rows {
        let gid = r.face.get("madeBy").and_then(|m| m.get("id")).and_then(|x| x.as_i64()).unwrap_or(-1);
        let gname = r.face.get("madeBy").and_then(|m| m.get("name")).cloned().unwrap_or(Value::Null);
        let g = match groups.iter_mut().position(|g| g.id == gid) {
            Some(i) => &mut groups[i],
            None => {
                groups.push(Group { id: gid, name: gname, faces: 0, read: 0, within: 0, sum: 0.0, sum_sq: 0.0, outside: 0.0, inside: 0.0 });
                groups.last_mut().expect("just pushed")
            }
        };
        g.faces += 1;
        g.read += r.read;
        g.within += r.within;
        g.sum += r.sum;
        g.sum_sq += r.sum_sq;
        g.outside = g.outside.max(r.outside);
        g.inside = g.inside.min(r.inside);
    }
    let mut out = Map::new();
    out.insert("bodies".into(), Value::Array(bodies.iter().map(|b| json!({"id": b.id.as_number(), "name": b.name})).collect()));
    if let Some((lo, hi)) = input.within {
        out.insert("box".into(), json!({"min": [lo.x, lo.y, lo.z], "max": [hi.x, hi.y, hi.z]}));
    }
    out.insert("range".into(), json!(within));
    out.insert("reach".into(), json!(reach));
    out.insert("scanVertices".into(), json!(points.len()));
    out.insert("read".into(), json!(n));
    out.insert("share".into(), json!(n as f64 / points.len().max(1) as f64));
    out.insert("within".into(), json!(within_count));
    let some = |x: f64| if n > 0 { json!(x) } else { Value::Null };
    out.insert("mean".into(), some(sum / n.max(1) as f64));
    out.insert("rms".into(), some((sum_sq / n.max(1) as f64).sqrt()));
    out.insert("outside".into(), some(outside));
    out.insert("inside".into(), some(inside));
    out.insert("p50".into(), pct(0.5));
    out.insert("p95".into(), pct(0.95));
    let patch_out = |p: &PatchReading, side_word: &str| -> Value {
        let mut m = patch_figures(&p.patch);
        m.insert("side".into(), json!(side_word));
        m.insert("at".into(), v3(p.patch.at));
        m.insert("vertex".into(), json!(p.vertex));
        if input.patch_vertices {
            m.insert("vertices".into(), json!(p.vertices));
        }
        m.insert("body".into(), p.body.map_or(Value::Null, |b| json!({"id": id(b), "name": name(b)})));
        m.insert("face".into(), match (p.body, p.face_hash) {
            (Some(b), Some(h)) => face(b, h),
            _ => Value::Null,
        });
        Value::Object(m)
    };
    out.insert("patches".into(), Value::Array(detail.patches.iter().map(|p| patch_out(p, if p.patch.extreme > 0.0 { "outside" } else { "inside" })).collect()));
    out.insert("patchesFound".into(), json!(detail.patches_found));
    out.insert("features".into(), features_out(detail.features_count, detail.features_share));
    out.insert("over".into(), json!(detail.over));
    out.insert("tolerance".into(), json!(tolerance));
    if let Some(far) = &detail.far {
        let side_out = |s: &FarSide, word: &str| json!({"count": s.count, "found": s.found, "patches": s.patches.iter().map(|p| patch_out(p, word)).collect::<Vec<_>>()});
        out.insert("far".into(), json!({"reach": far.reach, "count": far.count, "share": far.count as f64 / points.len().max(1) as f64, "inside": side_out(&far.inside, "inside"), "outside": side_out(&far.outside, "outside")}));
    }
    if detail.spacing > 0.0 {
        out.insert("spacing".into(), json!(round3(detail.spacing)));
    }
    out.insert(
        "side".into(),
        match (&side, detail.side) {
            (Some(s), Some(c)) => {
                let mut m = json!({"facingDeg": s.facing_deg, "refaced": c.refaced, "back": c.back, "inward": c.inward});
                if s.skin_outer {
                    m["skin"] = json!("outer");
                }
                m[if s.unfacing == Unfacing::Drop { "unfacingDropped" } else { "unfacingKept" }] = json!(c.unfacing);
                m
            }
            _ => json!({"off": "the scan came without its normals: every vertex read against the nearest surface, whatever it faces, a sheet on both sides"}),
        },
    );
    out.insert("worst".into(), Value::Array(detail.spots.iter().map(|s| json!({"at": v3(s.at), "deviation": round3(s.value), "body": {"id": id(s.body), "name": name(s.body)}, "face": face(s.body, s.face_hash)})).collect()));
    out.insert(
        "faces".into(),
        Value::Array(rows.iter().take(input.faces).map(|r| json!({"body": {"id": id(r.body), "name": name(r.body)}, "face": r.face, "read": r.read, "within": r.within, "mean": round3(r.sum / r.read.max(1) as f64), "rms": round3((r.sum_sq / r.read.max(1) as f64).sqrt()), "outside": r.outside, "inside": r.inside})).collect()),
    );
    if input.features > 0 {
        let mut by: Vec<Value> = groups.iter().map(|g| json!({"id": g.id, "name": g.name, "faces": g.faces, "read": g.read, "within": g.within, "mean": round3(g.sum / g.read.max(1) as f64), "rms": round3((g.sum_sq / g.read.max(1) as f64).sqrt()), "outside": round3(g.outside), "inside": round3(g.inside)})).collect();
        by.sort_by(|a, b| {
            let k = |x: &Value| x["outside"].as_f64().unwrap_or(0.0).max(-x["inside"].as_f64().unwrap_or(0.0));
            k(b).total_cmp(&k(a))
        });
        by.truncate(input.features);
        out.insert("byFeature".into(), Value::Array(by));
    }
    if groups.len() > input.features {
        out.insert("featuresFound".into(), json!(groups.len()));
    }
    Ok(Value::Object(out))
}

/// A comparison as detail "score" gives it — ScanRuler's `scoreOf`.
pub fn score_of(out: &Value) -> Value {
    let several = out["bodies"].as_array().is_some_and(|b| b.len() > 1);
    let r2 = |v: &Value| -> Value { Value::Array(v.as_array().map(|a| a.iter().map(|x| json!((x.as_f64().unwrap_or(0.0) * 100.0).round() / 100.0)).collect()).unwrap_or_default()) };
    let r1 = |v: &Value| json!((v.as_f64().unwrap_or(0.0) * 10.0).round() / 10.0);
    let slim = |p: &Value| -> Value {
        let mut m = json!({"centroid": r2(&p["centroid"]), "box": {"min": r2(&p["box"]["min"]), "max": r2(&p["box"]["max"])}, "area": r1(&p["area"]), "mean": p["mean"], "extreme": p["extreme"], "depth": p["depth"], "volume": r1(&p["volume"]), "kind": p["kind"], "at": r2(&p["at"]), "face": p["face"]});
        if let Some(v) = p.get("vertices") {
            m["vertices"] = v.clone();
        }
        if several {
            m["body"] = p["body"].clone();
        }
        m
    };
    let far_slim = |s: &Value| -> Value {
        let largest: Vec<Value> = s["patches"]
            .as_array()
            .map(|a| {
                a.iter()
                    .take(SCORE_FAR)
                    .map(|p| {
                        let mut m = json!({"centroid": r2(&p["centroid"]), "box": {"min": r2(&p["box"]["min"]), "max": r2(&p["box"]["max"])}, "area": r1(&p["area"]), "depth": p["depth"], "volume": r1(&p["volume"]), "kind": p["kind"], "face": p["face"]});
                        if several {
                            m["body"] = p["body"].clone();
                        }
                        m
                    })
                    .collect()
            })
            .unwrap_or_default();
        json!({"count": s["count"], "found": s["found"], "largest": largest})
    };
    let read = out["read"].as_f64().unwrap_or(0.0);
    let r3v = |v: &Value| if v.is_null() { Value::Null } else { json!(round3(v.as_f64().unwrap_or(0.0))) };
    let mut m = Map::new();
    if several {
        m.insert("bodies".into(), out["bodies"].clone());
    }
    if let Some(b) = out.get("box") {
        m.insert("box".into(), b.clone());
    }
    for k in ["range", "tolerance", "read"] {
        m.insert(k.into(), out[k].clone());
    }
    m.insert("share".into(), json!(round3(out["share"].as_f64().unwrap_or(0.0))));
    m.insert("within".into(), out["within"].clone());
    m.insert("withinShare".into(), if read > 0.0 { json!(round3(out["within"].as_f64().unwrap_or(0.0) / read)) } else { Value::Null });
    for k in ["mean", "rms", "p50", "p95"] {
        m.insert(k.into(), r3v(&out[k]));
    }
    m.insert("patchesFound".into(), out["patchesFound"].clone());
    m.insert("over".into(), out["over"].clone());
    m.insert("features".into(), json!({"count": out["features"]["count"], "shareOfExcess": out["features"]["shareOfExcess"]}));
    m.insert("patches".into(), Value::Array(out["patches"].as_array().map(|a| a.iter().map(slim).collect()).unwrap_or_default()));
    if let Some(far) = out.get("far") {
        m.insert("far".into(), json!({"count": far["count"], "share": round3(far["share"].as_f64().unwrap_or(0.0)), "inside": far_slim(&far["inside"]), "outside": far_slim(&far["outside"])}));
    }
    if let Some(b) = out.get("byFeature") {
        m.insert("byFeature".into(), b.clone());
    }
    Value::Object(m)
}

// ---- one feature against the scan (app/reCompareFeature.ts) ----

/// The faces a feature made, on every body that holds any, by hash (body index → hashes).
pub fn faces_made_by(bodies: &[CompareBody], id: FeatureId) -> Vec<(usize, Vec<u32>)> {
    bodies
        .iter()
        .enumerate()
        .filter_map(|(i, b)| {
            let t = b.topology?;
            let hashes: Vec<u32> = t.faces.iter().filter(|f| f.fingerprint.origin == id as i64).map(|f| f.hash).collect();
            (!hashes.is_empty()).then_some((i, hashes))
        })
        .collect()
}

/// The box round those faces, from the bodies' meshes.
pub fn box_of_faces(bodies: &[CompareBody], made: &[(usize, Vec<u32>)]) -> Option<(DVec3, DVec3)> {
    let mut lo = DVec3::splat(f64::INFINITY);
    let mut hi = DVec3::splat(f64::NEG_INFINITY);
    for (b, hashes) in made {
        let Some(body) = bodies.get(*b) else { continue };
        let m = body.mesh;
        for (t, f) in m.tri_face.iter().enumerate() {
            if !hashes.contains(f) {
                continue;
            }
            for v in m.triangles[t] {
                let p = m.positions[v as usize];
                lo = lo.min(p);
                hi = hi.max(p);
            }
        }
    }
    lo.x.is_finite().then_some((lo, hi))
}

/// A place of a feature: faces of it on one body that join each other edge to edge.
#[derive(Debug, Clone)]
pub struct Place {
    pub body: usize,
    pub hashes: Vec<u32>,
    pub faces: Vec<FaceInfo>,
    pub area: f64,
    pub centroid: DVec3,
}

/// The feature's faces gathered into its places, the largest first.
pub fn places_of(bodies: &[CompareBody], made: &[(usize, Vec<u32>)]) -> Vec<Place> {
    let mut out = Vec::new();
    for (b, hashes) in made {
        let Some(t) = bodies.get(*b).and_then(|x| x.topology) else { continue };
        let faces: Vec<&FaceInfo> = t.faces.iter().filter(|f| hashes.contains(&f.hash)).collect();
        let mut parent: HashMap<u32, u32> = faces.iter().map(|f| (f.hash, f.hash)).collect();
        fn root(parent: &mut HashMap<u32, u32>, h: u32) -> u32 {
            let p = parent.get(&h).copied().unwrap_or(h);
            if p == h {
                return h;
            }
            let r = root(parent, p);
            parent.insert(h, r);
            r
        }
        for e in &t.edges {
            let mine: Vec<u32> = e.faces.iter().copied().filter(|h| hashes.contains(h)).collect();
            for i in 1..mine.len() {
                let (a, b) = (root(&mut parent, mine[i]), root(&mut parent, mine[0]));
                parent.insert(a, b);
            }
        }
        let mut groups: Vec<(u32, Vec<FaceInfo>)> = Vec::new();
        for f in &faces {
            let r = root(&mut parent, f.hash);
            match groups.iter_mut().find(|g| g.0 == r) {
                Some(g) => g.1.push((*f).clone()),
                None => groups.push((r, vec![(*f).clone()])),
            }
        }
        for (_, g) in groups {
            let area: f64 = g.iter().map(|f| f.fingerprint.area).sum();
            let centroid = g.iter().map(|f| f.fingerprint.centroid * f.fingerprint.area).fold(DVec3::ZERO, |a, b| a + b) / if area != 0.0 { area } else { 1.0 };
            out.push(Place { body: *b, hashes: g.iter().map(|f| f.hash).collect(), faces: g, area, centroid });
        }
    }
    out.sort_by(|a, b| b.area.total_cmp(&a.area));
    out
}

/// One feature against the scan — ScanRuler's `reverse.body.compare { feature }`: the scan read
/// only about the faces the feature made. The feature's numbers fitted to the scan and the
/// picture across it are not read here (their fits and pictures are other commands'); the
/// answer says so.
pub fn compare_feature(bodies: &[CompareBody], features: &[Feature], feature: FeatureId, points: &[DVec3], normals: &[DVec3], input: &CompareInput) -> Result<Value, String> {
    let f = features.iter().find(|f| f.id == feature).ok_or_else(|| format!("There is no feature {feature}."))?;
    let made = faces_made_by(bodies, feature);
    if made.is_empty() {
        return Err(format!("{} made no face on a body as the part stands — a sketch, a feature rolled back or suppressed, or one whose faces later features took away.", f.name));
    }
    let Some((lo, hi)) = box_of_faces(bodies, &made) else { return Err(format!("The faces {} made have no mesh to read the scan about.", f.name)) };
    let pad = live_map_reach(input.range) + 1.0;
    let boxed = (lo - DVec3::splat(pad), hi + DVec3::splat(pad));
    let ids: Vec<usize> = made.iter().map(|m| m.0).collect();
    let sub: Vec<CompareBody> = ids.iter().filter_map(|i| bodies.get(*i)).map(|b| CompareBody { id: b.id, name: b.name, mesh: b.mesh, sheet: b.sheet, topology: b.topology }).collect();
    let on: Vec<(usize, Vec<i64>)> = made.iter().enumerate().map(|(k, (_, hs))| (k, hs.iter().map(|h| *h as i64).collect())).collect();
    let ask = CompareInput { within: Some(boxed), faces: 200, features: 0, ..input.clone() };
    let all = compare_answer(&sub, features, points, normals, &ask, Some(on))?;
    let mine = |face: &Value| face.get("madeBy").and_then(|m| m.get("id")).and_then(|x| x.as_i64()) == Some(feature as i64);
    let own = |row: &Value| -> Value {
        let mut r = row.clone();
        if let Some(face) = r.get_mut("face") {
            if mine(face) {
                if let Some(o) = face.as_object_mut() {
                    o.remove("madeBy");
                }
            }
        }
        r
    };
    let rows: Vec<&Value> = all["faces"].as_array().map(|a| a.iter().filter(|r| mine(&r["face"])).collect()).unwrap_or_default();
    let read: f64 = rows.iter().map(|r| r["read"].as_f64().unwrap_or(0.0)).sum();
    let within: f64 = rows.iter().map(|r| r["within"].as_f64().unwrap_or(0.0)).sum();
    let patches: Vec<Value> = all["patches"].as_array().map(|a| a.iter().filter(|p| mine(&p["face"])).map(own).collect()).unwrap_or_default();
    let worst: Vec<Value> = all["worst"].as_array().map(|a| a.iter().filter(|p| mine(&p["face"])).map(own).collect()).unwrap_or_default();
    let mut answer = json!({
        "feature": {"id": f.id, "name": f.name, "kind": f.kind.kind_name()},
        "bodies": all["bodies"],
        "box": {"min": v3(lo), "max": v3(hi)},
        "range": all["range"],
        "faces": rows.len(),
        "read": read,
        "within": within,
        "mean": if read > 0.0 { json!(round4(rows.iter().map(|r| r["mean"].as_f64().unwrap_or(0.0) * r["read"].as_f64().unwrap_or(0.0)).sum::<f64>() / read)) } else { Value::Null },
        "rms": if read > 0.0 { json!(round4((rows.iter().map(|r| r["rms"].as_f64().unwrap_or(0.0).powi(2) * r["read"].as_f64().unwrap_or(0.0)).sum::<f64>() / read).sqrt())) } else { Value::Null },
        "outside": rows.iter().map(|r| r["outside"].as_f64().unwrap_or(0.0)).reduce(f64::max).map_or(Value::Null, |x| json!(x)),
        "inside": rows.iter().map(|r| r["inside"].as_f64().unwrap_or(0.0)).reduce(f64::min).map_or(Value::Null, |x| json!(x)),
        "patches": patches,
        "patchesFound": all["patchesFound"],
        "features": all["features"],
        "over": all["over"],
        "tolerance": all["tolerance"],
        "side": all["side"],
        "faceRows": rows.iter().take(input.faces).map(|r| own(r)).collect::<Vec<_>>(),
        "numbers": {"note": "The feature's numbers fitted to the scan are not read here: the fits for a round's radius, a chamfer's distance, a draft's angle and an extrude's end live in their own feature boxes."},
    });
    if input.spots > 0 {
        answer["worst"] = Value::Array(worst);
    }
    if let Some(far) = all.get("far") {
        answer["far"] = far.clone();
    }
    let places = places_of(bodies, &made);
    if places.len() > 1 {
        answer["places"] = Value::Array(places.iter().map(|p| json!({"at": v3(p.centroid), "faces": p.faces.len(), "area": round3(p.area)})).collect());
    }
    let _ = PatchKind::Feature;
    Ok(answer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compare::vertex_normals;

    fn box_mesh(size: f64, at: DVec3, face: u32) -> BodyMesh {
        let m = mm_core::primitives::cube(size, 4);
        let positions: Vec<DVec3> = m.positions().iter().map(|p| *p + at).collect();
        let ids: Vec<u32> = m.triangle_ids().collect();
        let triangles: Vec<[u32; 3]> = ids.iter().map(|t| m.triangle(*t)).collect();
        // Each triangle's face by which side of the box it lies on.
        let tri_face: Vec<u32> = triangles
            .iter()
            .map(|t| {
                let n = (positions[t[1] as usize] - positions[t[0] as usize]).cross(positions[t[2] as usize] - positions[t[0] as usize]).normalize();
                let k = if n.x.abs() > 0.9 { 0 } else if n.y.abs() > 0.9 { 2 } else { 4 };
                face + k + if n.x + n.y + n.z > 0.0 { 0 } else { 1 }
            })
            .collect();
        BodyMesh { normals: vertex_normals(&positions, &triangles), positions, triangles, tri_face, ..Default::default() }
    }

    /// A grid of scan points on the box's top face, lifted by `lift` over a patch.
    fn top_scan(lift: f64) -> Vec<DVec3> {
        let mut out = Vec::new();
        for i in 0..40 {
            for j in 0..40 {
                let (x, y) = (-9.75 + i as f64 * 0.5, -9.75 + j as f64 * 0.5);
                let z = 10.0 + if x > 0.0 && y > 0.0 { lift } else { 0.0 };
                out.push(DVec3::new(x, y, z));
            }
        }
        out
    }

    // Ported from ScanRuler's tests/reCompareFaces.test.ts: the reading told by face.
    #[test]
    fn reads_by_face_with_patches_on_one_face_and_worst_spots_apart() {
        let m = box_mesh(20.0, DVec3::ZERO, 0);
        let bodies = [CompareBody { id: BodyId::new(1), name: "Solid 1", mesh: &m, sheet: false, topology: None }];
        let pts = top_scan(0.4);
        let req = PatchRequest { count: 8, tolerance: 0.1, grain: None, faces: None };
        let d = compare_bodies_detail(&bodies, &pts, None, 0.1, 1.0, 3, 1.0, Some(&req), None, None);
        assert_eq!(d.field.values.len(), pts.len());
        // Every vertex lies nearest the top face (+Z: face 4).
        assert_eq!(d.faces.len(), 1);
        assert_eq!(d.faces[0].face_hash, 4);
        assert_eq!(d.faces[0].read, 1600);
        assert_eq!(d.faces[0].within, 1200);
        assert!((d.faces[0].outside - 0.4).abs() < 1e-6);
        assert_eq!(d.patches.len(), 1);
        assert_eq!(d.patches[0].patch.count, 400);
        assert_eq!(d.patches[0].patch.kind, PatchKind::Feature);
        assert_eq!(d.spots.len(), 3);
        for a in &d.spots {
            for b in &d.spots {
                assert!(a.vertex == b.vertex || a.at.distance(b.at) >= 1.0);
            }
        }
    }

    #[test]
    fn the_scan_past_the_reach_is_told_by_side() {
        let m = box_mesh(20.0, DVec3::ZERO, 0);
        let bodies = [CompareBody { id: BodyId::new(1), name: "Solid 1", mesh: &m, sheet: false, topology: None }];
        // A boss standing 3 mm proud of the top over 5 × 5 mm, and a pocket 4 mm deep.
        let mut pts = top_scan(0.0);
        for i in 0..10 {
            for j in 0..10 {
                pts.push(DVec3::new(i as f64 * 0.5, j as f64 * 0.5, 13.0));
                pts.push(DVec3::new(-5.0 + i as f64 * 0.5, -5.0 + j as f64 * 0.5, 6.0));
            }
        }
        let req = PatchRequest { count: 8, tolerance: 0.1, grain: None, faces: None };
        let d = compare_bodies_detail(&bodies, &pts, None, 0.1, 1.0, 0, 1.0, Some(&req), None, Some(25.0));
        let far = d.far.unwrap();
        assert_eq!(far.outside.count, 100);
        assert_eq!(far.inside.count, 100);
        assert_eq!(far.outside.patches.len(), 1);
        assert!((far.outside.patches[0].patch.mean - 3.0).abs() < 1e-6);
        assert!((far.inside.patches[0].patch.mean + 4.0).abs() < 1e-6);
    }

    #[test]
    fn the_answer_and_its_score() {
        let m = box_mesh(20.0, DVec3::ZERO, 0);
        let bodies = [CompareBody { id: BodyId::new(1), name: "Solid 1", mesh: &m, sheet: false, topology: None }];
        let pts = top_scan(0.4);
        let out = compare_answer(&bodies, &[], &pts, &[], &CompareInput::new(0.1), None).unwrap();
        assert_eq!(out["read"], json!(1600));
        assert_eq!(out["within"], json!(1200));
        assert!((out["p95"].as_f64().unwrap() - 0.4).abs() < 1e-6);
        assert_eq!(out["patches"].as_array().unwrap().len(), 1);
        assert_eq!(out["patches"][0]["side"], json!("outside"));
        assert!(out["side"]["off"].is_string());
        let score = score_of(&out);
        assert!(score.get("worst").is_none() && score.get("faces").is_none());
        assert_eq!(score["withinShare"], json!(0.75));
        assert!(compare_answer(&[], &[], &pts, &[], &CompareInput::new(0.1), None).is_err());
    }

    #[test]
    fn a_box_keeps_the_reading_to_one_region() {
        let m = box_mesh(20.0, DVec3::ZERO, 0);
        let bodies = [CompareBody { id: BodyId::new(1), name: "Solid 1", mesh: &m, sheet: false, topology: None }];
        let pts = top_scan(0.4);
        let d = compare_bodies_detail(&bodies, &pts, None, 0.1, 1.0, 0, 1.0, None, Some((DVec3::new(-20.0, -20.0, 0.0), DVec3::new(0.0, 0.0, 20.0))), None);
        assert_eq!(d.field.values.len(), 400);
        assert!(d.field.values.iter().all(|v| v.abs() < 1e-6));
    }
}
