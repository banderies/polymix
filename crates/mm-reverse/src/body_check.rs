// SPDX-License-Identifier: AGPL-3.0-only
//! A body looked over for what will trouble the next feature or the export, and two bodies
//! measured against each other. Ported from ScanRuler (`plugins/reverse/app/reBodyCommands.ts`
//! check and clearance, `core/kernelEdges.ts`, the body check of `core/kernel.ts`, the
//! clearance text of `app/reSession.ts`), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only — with
//! Open CASCADE's calls replaced by SolidCraft's.
//!
//! The check says whether a body is sound: valid, closed, one solid, its volume and area, the
//! faces too small or too narrow to be meant, the faces the mesher drew nothing for, the edges
//! that do not join two faces as a closed solid's do (one face — a hole in the skin; three or
//! more; a line inside its one face; two faces facing opposite ways), the edges a hair long a
//! STEP reader may drop, the voids sealed inside, and the body written as STEP and read back.
//! The audit (`audit.rs`) says whether it is clean.

use glam::DVec3;
use serde_json::{json, Map, Value};
use solidcraft_kernel::{self as kernel, Body, BoolOp};

use crate::audit::{audit_body, js, trace_face, AuditOptions, Audit, Finding, FindingKind, OddEdge, OddKind};
use crate::build::{dv3, BodyMesh};
use crate::model::Feature;
use crate::thread::{round_face_is_boss, FaceMesh};
use crate::topo::{FaceInfo, FaceKind, Topology};

/// A face under this area, mm², is a sliver.
pub const TINY_FACE_MM2: f64 = 0.01;
/// How narrow a face may be, mm — twice its area over its perimeter.
pub const NARROW_MM: f64 = 0.01;
/// Under this an edge between two faces is short enough for a STEP reader to drop.
pub const SHORT_EDGE_MM: f64 = 1e-4;
/// Under this an edge has no length — a pole, an apex.
const NO_LENGTH_MM: f64 = 1e-7;
/// Two outward normals within this many degrees of opposite meet at a knife edge.
const KNIFE_DEG: f64 = 5.0;
/// How many findings of each kind audit "brief" gives a line.
pub const BRIEF_PER_KIND: usize = 3;

/// The odd edges of a body and how many of each kind.
#[derive(Debug, Clone, Default)]
pub struct OddEdges {
    pub edges: Vec<OddEdge>,
    pub free: usize,
    pub inside: usize,
    pub non_manifold: usize,
    pub knife: usize,
}

/// A void sealed inside a body.
#[derive(Debug, Clone)]
pub struct Void {
    pub solid: usize,
    pub volume: f64,
    pub min: DVec3,
    pub max: DVec3,
}

/// The body written as STEP and read back.
#[derive(Debug, Clone, Default)]
pub struct ReadBack {
    /// The edges that then do not join two faces: where, how long.
    pub edges: Vec<(DVec3, f64)>,
    pub closed: bool,
    pub valid: bool,
    /// Why it could not be done, if it could not.
    pub failed: Option<String>,
}

/// What the check found.
#[derive(Debug, Clone, Default)]
pub struct BodyCheck {
    pub valid: bool,
    pub closed: bool,
    pub solids: usize,
    pub volume: f64,
    pub area: f64,
    pub faces: usize,
    /// (hash, area, centroid)
    pub tiny: Vec<(u32, f64, DVec3)>,
    /// (hash, width, area, centroid)
    pub narrow: Vec<(u32, f64, f64, DVec3)>,
    pub untessellated: Vec<u32>,
    pub odd: OddEdges,
    /// Edges between two faces under [`SHORT_EDGE_MM`]: (where, length, faces).
    pub short: Vec<(DVec3, f64, Vec<u32>)>,
    pub shells: usize,
    pub voids: Vec<Void>,
    pub read_back: Option<ReadBack>,
    pub problems_of_kernel: Vec<String>,
}

fn normal_of_face_near(mesh: &BodyMesh, face: u32, p: DVec3) -> Option<DVec3> {
    let mut best: Option<(f64, DVec3)> = None;
    for (t, f) in mesh.triangles.iter().zip(&mesh.tri_face) {
        if *f != face {
            continue;
        }
        let [a, b, c] = [mesh.positions[t[0] as usize], mesh.positions[t[1] as usize], mesh.positions[t[2] as usize]];
        let (q, _) = mm_core::geom::closest_point_on_triangle(p, a, b, c);
        let d = q.distance_squared(p);
        let n = (b - a).cross(c - a);
        if n.length_squared() > 0.0 && best.is_none_or(|x| d < x.0) {
            best = Some((d, n.normalize()));
        }
    }
    best.map(|b| b.1)
}

fn face_near(mesh: &BodyMesh, face: u32, p: DVec3) -> f64 {
    let mut best = f64::INFINITY;
    for (t, f) in mesh.triangles.iter().zip(&mesh.tri_face) {
        if *f != face {
            continue;
        }
        let [a, b, c] = [mesh.positions[t[0] as usize], mesh.positions[t[1] as usize], mesh.positions[t[2] as usize]];
        let (q, _) = mm_core::geom::closest_point_on_triangle(p, a, b, c);
        best = best.min(q.distance(p));
    }
    best
}

/// The odd edges of a body, read off its topology and mesh: an edge with three or more faces;
/// with one (a seam has its one face twice and is none of these) — inside its face when the
/// face lies on both sides of it at its middle, else free; with two whose outward normals there
/// face opposite ways, a knife.
pub fn odd_edges(topology: &Topology, mesh: &BodyMesh) -> OddEdges {
    let mut out = OddEdges::default();
    let knife_cos = KNIFE_DEG.to_radians().cos();
    for e in &topology.edges {
        let faces: Vec<u32> = {
            let mut v = Vec::new();
            for f in &e.faces {
                if !v.contains(f) {
                    v.push(*f);
                }
            }
            v
        };
        let seam = e.faces.len() == 2 && e.faces[0] == e.faces[1];
        let length = e.fingerprint.length;
        if !(length > NO_LENGTH_MM) {
            continue;
        }
        let mid = e.fingerprint.mid;
        let kind = if faces.len() >= 3 {
            Some(OddKind::NonManifold)
        } else if faces.is_empty() {
            Some(OddKind::Free)
        } else if faces.len() == 1 && !seam {
            // Points a hair off the edge either way across it, on the face: both in the face
            // means the edge lies inside it.
            let n = normal_of_face_near(mesh, faces[0], mid).unwrap_or(DVec3::Z);
            let across = n.cross(e.fingerprint.dir).try_normalize();
            let step = (length * 1e-3).clamp(1e-4, 1e-2);
            let inside = across.is_some_and(|a| [1.0, -1.0].iter().all(|s| face_near(mesh, faces[0], mid + a * (*s * step * 10.0)) <= step * 2.0));
            Some(if inside { OddKind::Inside } else { OddKind::Free })
        } else if faces.len() == 2 {
            match (normal_of_face_near(mesh, faces[0], mid), normal_of_face_near(mesh, faces[1], mid)) {
                (Some(a), Some(b)) if a.dot(b) < -knife_cos => Some(OddKind::Knife),
                _ => None,
            }
        } else {
            None
        };
        let Some(kind) = kind else { continue };
        match kind {
            OddKind::Free => out.free += 1,
            OddKind::Inside => out.inside += 1,
            OddKind::NonManifold => out.non_manifold += 1,
            OddKind::Knife => out.knife += 1,
        }
        out.edges.push(OddEdge { hash: e.hash, kind, at: mid, length, faces });
    }
    out
}

/// The connected pieces of a mesh, welded by where their points lie, each with its signed
/// volume and box.
fn mesh_shells(mesh: &BodyMesh) -> Vec<(f64, DVec3, DVec3)> {
    let mut key: std::collections::HashMap<[i64; 3], usize> = std::collections::HashMap::new();
    let mut id = vec![0usize; mesh.positions.len()];
    for (i, p) in mesh.positions.iter().enumerate() {
        let k = [(p.x * 1e5).round() as i64, (p.y * 1e5).round() as i64, (p.z * 1e5).round() as i64];
        let n = key.len();
        id[i] = *key.entry(k).or_insert(n);
    }
    let mut parent: Vec<usize> = (0..key.len()).collect();
    fn find(p: &mut [usize], mut a: usize) -> usize {
        while p[a] != a {
            p[a] = p[p[a]];
            a = p[a];
        }
        a
    }
    for t in &mesh.triangles {
        let (a, b, c) = (id[t[0] as usize], id[t[1] as usize], id[t[2] as usize]);
        let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
        parent[rb] = ra;
        let (ra, rc) = (find(&mut parent, a), find(&mut parent, c));
        parent[rc] = ra;
    }
    let mut shells: Vec<(usize, f64, DVec3, DVec3)> = Vec::new();
    for t in &mesh.triangles {
        let r = find(&mut parent, id[t[0] as usize]);
        let [a, b, c] = [mesh.positions[t[0] as usize], mesh.positions[t[1] as usize], mesh.positions[t[2] as usize]];
        let v = a.dot(b.cross(c)) / 6.0;
        let lo = a.min(b).min(c);
        let hi = a.max(b).max(c);
        match shells.iter_mut().find(|s| s.0 == r) {
            Some(s) => {
                s.1 += v;
                s.2 = s.2.min(lo);
                s.3 = s.3.max(hi);
            }
            None => shells.push((r, v, lo, hi)),
        }
    }
    shells.into_iter().map(|s| (s.1, s.2, s.3)).collect()
}

/// Look a body over (ScanRuler's `Builder.check`): faces under `tiny_area` listed as tiny.
pub fn check_body(body: &Body, topology: &Topology, mesh: &BodyMesh, tiny_area: f64) -> BodyCheck {
    let validity = body.validity();
    let closed = !validity.iter().any(|v| v.contains("not closed"));
    let m = kernel::measure(body).ok();
    let solids = body.lumps().map(|l| l.len()).unwrap_or(1);
    let mut c = BodyCheck {
        valid: validity.is_empty(),
        closed: closed && solids == 1,
        solids,
        volume: m.as_ref().map_or(0.0, |m| m.volume),
        area: m.as_ref().map_or(0.0, |m| m.area),
        faces: topology.faces.len(),
        shells: body.shell_count(),
        problems_of_kernel: validity,
        ..Default::default()
    };
    // Each face's perimeter, from its edges.
    let mut perimeter = vec![0.0; topology.faces.len()];
    for e in &topology.edges {
        let mut seen = Vec::new();
        for f in &e.faces {
            if seen.contains(f) {
                continue;
            }
            seen.push(*f);
            if let Some(p) = perimeter.get_mut(*f as usize) {
                *p += e.fingerprint.length;
            }
        }
    }
    let meshed: std::collections::HashSet<u32> = mesh.tri_face.iter().copied().collect();
    for (i, f) in topology.faces.iter().enumerate() {
        let fp = &f.fingerprint;
        if fp.area < tiny_area {
            c.tiny.push((f.hash, fp.area, fp.centroid));
        }
        let per = perimeter.get(i).copied().unwrap_or(0.0);
        if per > 0.0 {
            let width = 2.0 * fp.area / per;
            if width < NARROW_MM && fp.area >= tiny_area {
                c.narrow.push((f.hash, width, fp.area, fp.centroid));
            }
        }
        if !meshed.contains(&f.hash) {
            c.untessellated.push(f.hash);
        }
    }
    c.odd = odd_edges(topology, mesh);
    for e in &topology.edges {
        let two = e.faces.len() == 2 && e.faces[0] != e.faces[1];
        if two && e.fingerprint.length > NO_LENGTH_MM && e.fingerprint.length < SHORT_EDGE_MM {
            c.short.push((e.fingerprint.mid, e.fingerprint.length, e.faces.clone()));
        }
    }
    // The voids: a shell of the mesh wound inward — its volume negative — inside an outer one.
    let shells = mesh_shells(mesh);
    let outers: Vec<&(f64, DVec3, DVec3)> = shells.iter().filter(|s| s.0 > 0.0).collect();
    for s in shells.iter().filter(|s| s.0 < 0.0) {
        let solid = outers.iter().position(|o| o.1.cmple(s.1).all() && s.2.cmple(o.2).all()).unwrap_or(0);
        c.voids.push(Void { solid, volume: -s.0, min: s.1, max: s.2 });
    }
    c.read_back = Some(read_back(body));
    c
}

/// The body written as STEP and read back in memory: whether it is still a closed, valid
/// solid, and the edges that then do not join two faces.
pub fn read_back(body: &Body) -> ReadBack {
    let export = kernel::ExportBody { name: "check".into(), body, color: None };
    let text = match kernel::step_export_bodies(&[export], &kernel::StepHeader::default()) {
        Ok(t) => t,
        Err(e) => return ReadBack { failed: Some(format!("it could not be written as STEP: {e}")), ..Default::default() },
    };
    let read = match kernel::step_import(&text) {
        Ok(r) => r,
        Err(e) => return ReadBack { failed: Some(format!("the STEP written could not be read back: {e}")), ..Default::default() },
    };
    let mut out = ReadBack { closed: !read.bodies.is_empty(), valid: !read.bodies.is_empty(), ..Default::default() };
    for b in &read.bodies {
        let v = b.body.validity();
        if v.iter().any(|x| x.contains("not closed")) {
            out.closed = false;
        }
        if !v.is_empty() {
            out.valid = false;
        }
        let t = crate::build::read_topology(&b.body, &[]);
        for e in &t.edges {
            let seam = e.faces.len() == 2 && e.faces[0] == e.faces[1];
            if e.faces.len() < 2 && !seam && e.fingerprint.length > NO_LENGTH_MM {
                out.edges.push((e.fingerprint.mid, e.fingerprint.length));
            }
        }
    }
    out
}

fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

fn r3s(p: DVec3) -> Value {
    json!([round3(p.x), round3(p.y), round3(p.z)])
}

fn to_precision(x: f64, p: usize) -> f64 {
    crate::js_fmt::to_precision_num(x, p)
}

/// A face by hash as an answer names it.
fn face_named(t: &Topology, features: &[Feature], hash: u32) -> (Option<usize>, Option<FaceKind>, Option<(i64, Option<String>)>) {
    match t.faces.iter().position(|f| f.hash == hash) {
        Some(i) => {
            let fp = &t.faces[i].fingerprint;
            (Some(i), Some(fp.kind), Some((fp.origin, features.iter().find(|f| f.id as i64 == fp.origin).map(|f| f.name.clone()))))
        }
        None => (None, None, None),
    }
}

fn face_json(t: &Topology, features: &[Feature], hash: u32) -> Value {
    let (i, kind, made) = face_named(t, features, hash);
    let mut m = Map::new();
    m.insert("index".into(), i.map_or(Value::Null, |i| json!(i)));
    if let Some(k) = kind {
        m.insert("kind".into(), json!(k.name()));
    }
    if let Some((id, name)) = made {
        m.insert("madeBy".into(), json!({"id": id, "name": name}));
    }
    Value::Object(m)
}

/// The makers of faces, each once: their indices and the features' names.
fn makers_of(t: &Topology, features: &[Feature], hashes: &[u32]) -> (Vec<usize>, Vec<String>) {
    let mut indices = Vec::new();
    let mut makers: Vec<String> = Vec::new();
    for h in hashes {
        let (i, _, made) = face_named(t, features, *h);
        if let Some(i) = i {
            indices.push(i);
        }
        if let Some((_, Some(n))) = made {
            if !makers.contains(&n) {
                makers.push(n);
            }
        }
    }
    (indices, makers)
}

fn plural(n: usize) -> &'static str {
    if n == 1 {
        ""
    } else {
        "s"
    }
}

/// How the audit is asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuditMode {
    Off,
    Brief,
    Lists,
    Full,
}

/// A body's check as ScanRuler's `reverse.body.check` answers it.
#[allow(clippy::too_many_arguments)]
pub fn check_answer(id: f64, name: &str, body: &Body, topology: &Topology, mesh: &BodyMesh, features: &[Feature], tiny_area: f64, audit: AuditMode, options: AuditOptions, limit: Option<usize>) -> Value {
    let c = check_body(body, topology, mesh, tiny_area);
    let t = topology;
    let tiny: Vec<Value> = c.tiny.iter().map(|(h, a, at)| {
        let mut f = face_json(t, features, *h);
        f["area"] = json!(a);
        f["centroid"] = json!([at.x, at.y, at.z]);
        f
    }).collect();
    let narrow: Vec<Value> = c.narrow.iter().map(|(h, w, a, at)| {
        let mut f = face_json(t, features, *h);
        f["width"] = json!(to_precision(*w, 3));
        f["area"] = json!(to_precision(*a, 3));
        f["centroid"] = r3s(*at);
        f
    }).collect();
    let contacts = c.odd.inside + c.odd.knife;
    let mut edge_counts = json!({"free": c.odd.free, "nonManifold": c.odd.non_manifold, "contacts": contacts});
    if !c.short.is_empty() {
        edge_counts["hairs"] = json!(c.short.len());
    }
    let odd_where = |kinds: &[OddKind]| -> String {
        let Some(e) = c.odd.edges.iter().find(|e| kinds.contains(&e.kind)) else { return String::new() };
        let (indices, makers) = makers_of(t, features, &e.faces);
        let idx: Vec<String> = indices.iter().map(|i| i.to_string()).collect();
        format!(" — the first at ({}, {}, {}), face{} {}{}", js(round3(e.at.x)), js(round3(e.at.y)), js(round3(e.at.z)), plural(indices.len()), if idx.is_empty() { "?".into() } else { idx.join(", ") }, if makers.is_empty() { String::new() } else { format!(" made by {}", makers.join(", ")) })
    };
    let mut problems: Vec<String> = Vec::new();
    if !c.valid {
        problems.push("The kernel’s check fails on it.".into());
    }
    if !c.closed {
        problems.push(if c.solids == 0 { "It is no solid.".into() } else if c.solids > 1 { format!("It is {} solids, not one.", c.solids) } else { "It is not closed.".into() });
    }
    let index_of = |h: u32| t.faces.iter().position(|f| f.hash == h).map_or("?".to_string(), |i| i.to_string());
    if !c.tiny.is_empty() {
        let list: Vec<String> = c.tiny.iter().map(|(h, a, at)| format!("{} ({} mm² at {}, {}, {})", index_of(*h), js(to_precision(*a, 2)), js(round3(at.x)), js(round3(at.y)), js(round3(at.z)))).collect();
        problems.push(format!("{} face{} under {} mm²: {}.", c.tiny.len(), plural(c.tiny.len()), js(tiny_area), list.join(", ")));
    }
    if !c.narrow.is_empty() {
        let list: Vec<String> = c.narrow.iter().take(6).map(|(h, w, _, at)| {
            let (_, _, made) = face_named(t, features, *h);
            let by = made.and_then(|m| m.1).map_or(String::new(), |n| format!(", made by {n}"));
            format!("{} ({} mm wide at {}, {}, {}{by})", index_of(*h), js(to_precision(*w, 3)), js(round3(at.x)), js(round3(at.y)), js(round3(at.z)))
        }).collect();
        let more = if c.narrow.len() > 6 { format!(" and {} more", c.narrow.len() - 6) } else { String::new() };
        problems.push(format!("{} face{} under {} mm wide — a sliver two faces a hair apart leave, which a round beside it stumbles on: {}{more}.", c.narrow.len(), plural(c.narrow.len()), js(NARROW_MM), list.join(", ")));
    }
    if c.odd.free > 0 {
        problems.push(format!("{} edge{} with one face — a hole in the skin{}.", c.odd.free, plural(c.odd.free), odd_where(&[OddKind::Free])));
    }
    if c.odd.non_manifold > 0 {
        problems.push(format!("{} edge{} where three or more faces meet — the skin runs into itself, which a STEP reader takes for no closed solid{}.", c.odd.non_manifold, plural(c.odd.non_manifold), odd_where(&[OddKind::NonManifold])));
    }
    let read_back = c.read_back.clone().unwrap_or_default();
    let whole = read_back.closed && read_back.valid;
    if let Some(first) = read_back.edges.first() {
        let n = read_back.edges.len();
        let tail = if first.1 < 0.01 { " — a round’s curve ending a hair off its corner, which reverse.export.step mends where it can; roundTrip: true says whether" } else { " — roundTrip: true for the faces there" };
        problems.push(format!("Written as STEP and read back, {} not join two faces — the first at ({}, {}, {}), {} mm long: a hole a reader makes that the body in memory does not show{tail}.", if n == 1 { "an edge does".to_string() } else { format!("{n} edges do") }, js(round3(first.0.x)), js(round3(first.0.y)), js(round3(first.0.z)), js(to_precision(first.1, 2))));
    } else if read_back.failed.is_none() && !whole && c.valid {
        problems.push(format!("Written as STEP and read back, it is {}, with no edge open — a face the reader lost whole, as an off-centre dome turned from its base to its pole is: roundTrip: true names the face.", if !read_back.closed { "no closed solid" } else { "a solid the kernel’s check fails" }));
    }
    if !c.untessellated.is_empty() {
        let list: Vec<String> = c.untessellated.iter().map(|h| index_of(*h)).collect();
        problems.push(format!("{} face{} the mesher drew nothing for: {}.", c.untessellated.len(), plural(c.untessellated.len()), list.join(", ")));
    }
    // The audit.
    let is_boss = |f: &FaceInfo| -> Option<bool> {
        let center = f.fingerprint.center?;
        round_face_is_boss(FaceMesh { positions: &mesh.positions, triangles: &mesh.triangles, tri_face: &mesh.tri_face }, f.hash, center, f.fingerprint.dir)
    };
    let looked: Audit = audit_body(&t.faces, &t.edges, &AuditOptions { alike: audit != AuditMode::Full, ..options }, Some(&is_boss), &c.odd.edges);
    let findings: Vec<Finding> = if audit == AuditMode::Off { Vec::new() } else { looked.findings.clone() };
    let audited = audit_out(&findings, t, features, audit == AuditMode::Full, if audit == AuditMode::Brief { limit.unwrap_or(BRIEF_PER_KIND) } else { usize::MAX });
    let contact_warning = (contacts > 0).then(|| format!("{contacts} tangent contact{} — a face that only touched another and left its line{}: let the one cross the other clearly (0.05 mm or more) or meet it with a round; reverse.export.step may read back as no closed solid.", plural(contacts), odd_where(&[OddKind::Inside, OddKind::Knife])));
    let hair_warning = c.short.first().map(|(at, _, faces)| {
        let (indices, makers) = makers_of(t, features, faces);
        let idx: Vec<String> = indices.iter().map(|i| i.to_string()).collect();
        format!("{} between two faces under 0.0001 mm — a hair a round left, which a STEP reader may drop from one face and read back as a hole: the first at ({}, {}, {}), faces {}{}; reverse.export.step {{ verify: true }} says whether it holds.", if c.short.len() == 1 { "An edge".to_string() } else { format!("{} edges", c.short.len()) }, js(round3(at.x)), js(round3(at.y)), js(round3(at.z)), if idx.is_empty() { "?".into() } else { idx.join(", ") }, if makers.is_empty() { String::new() } else { format!(" made by {}", makers.join(", ")) })
    });
    let void_list: Vec<Value> = c.voids.iter().map(|v| {
        let mut m = json!({"volume": round3(v.volume), "box": {"min": r3s(v.min), "max": r3s(v.max)}});
        if c.shells > 1 && c.solids > 1 {
            m["solid"] = json!(v.solid);
        }
        m
    }).collect();
    let void_warning = (!c.voids.is_empty()).then(|| {
        let n = c.voids.len();
        let list: Vec<String> = c.voids.iter().map(|v| format!("{} mm³ in ({}, {}, {})–({}, {}, {})", js(round3(v.volume)), js(round3(v.min.x)), js(round3(v.min.y)), js(round3(v.min.z)), js(round3(v.max.x)), js(round3(v.max.y)), js(round3(v.max.z)))).collect();
        format!("{n} void{} sealed inside the body — a shell of {} own the outside does not show: {}. Fill {} (a join of a body over {}), or build the bands to meet the core.", plural(n), if n == 1 { "its" } else { "their" }, list.join("; "), if n == 1 { "it" } else { "them" }, if n == 1 { "it" } else { "them" })
    });
    let mut warnings: Vec<String> = Vec::new();
    warnings.extend(void_warning);
    warnings.extend(contact_warning);
    warnings.extend(hair_warning);
    if audit != AuditMode::Brief {
        warnings.extend(audited.warnings.clone());
    }
    let mut out = Map::new();
    out.insert("id".into(), json!(id));
    out.insert("name".into(), json!(name));
    out.insert("ok".into(), json!(problems.is_empty()));
    out.insert("valid".into(), json!(c.valid));
    out.insert("closed".into(), json!(c.closed));
    out.insert("stepsBetweenFeatures".into(), json!(looked.steps_between_features));
    out.insert("solids".into(), json!(c.solids));
    out.insert("volume".into(), json!(c.volume));
    out.insert("area".into(), json!(c.area));
    out.insert("faces".into(), json!(c.faces));
    out.insert("tiny".into(), Value::Array(tiny));
    out.insert("narrow".into(), Value::Array(narrow));
    out.insert("untessellated".into(), Value::Array(c.untessellated.iter().map(|h| face_json(t, features, *h)).collect()));
    out.insert("edges".into(), edge_counts);
    if !c.short.is_empty() {
        out.insert("hairEdges".into(), Value::Array(c.short.iter().map(|(at, l, faces)| json!({"at": r3s(*at), "length": to_precision(*l, 2), "faces": makers_of(t, features, faces).0})).collect()));
    }
    if !read_back.edges.is_empty() || (!whole && read_back.failed.is_none()) {
        let mut rb = json!({"edges": read_back.edges.len()});
        if let Some(first) = read_back.edges.first() {
            rb["first"] = json!({"at": r3s(first.0), "length": to_precision(first.1, 2), "kind": "free"});
        }
        if !whole {
            rb["closed"] = json!(read_back.closed);
            rb["valid"] = json!(read_back.valid);
        }
        out.insert("readBack".into(), rb);
    }
    if let Some(f) = &read_back.failed {
        out.insert("readBackFailed".into(), json!(f));
    }
    out.insert("shells".into(), json!(c.shells));
    out.insert("voids".into(), json!(c.voids.len()));
    if !c.voids.is_empty() {
        out.insert("voidList".into(), Value::Array(void_list));
    }
    if !problems.is_empty() {
        out.insert("problems".into(), json!(problems));
    }
    if audit != AuditMode::Off {
        out.insert("clean".into(), json!(findings.is_empty()));
        if audit == AuditMode::Brief {
            out.insert("findings".into(), json!(audited.brief));
            if !findings.is_empty() {
                out.insert("counts".into(), Value::Object(looked.found.iter().map(|(k, n)| (k.name().to_string(), json!(n))).collect()));
            }
        } else {
            for (k, v) in audited.lists {
                out.insert(k, Value::Array(v));
            }
        }
        if let Some(u) = looked.under_half_t {
            if u > 0 {
                out.insert("underHalfT".into(), json!(u));
            }
        }
    }
    if !warnings.is_empty() {
        out.insert("warnings".into(), json!(warnings));
    }
    Value::Object(out)
}

/// The audit's findings as the check lists them.
pub struct AuditOut {
    pub lists: Vec<(String, Vec<Value>)>,
    pub warnings: Vec<String>,
    pub brief: Vec<String>,
}

fn trace_json(t: &crate::audit::FaceTrace) -> Value {
    let mut m = json!({"feature": {"id": t.feature.0, "name": t.feature.1, "kind": t.feature.2}, "role": t.role});
    if let Some((id, name)) = &t.sketch {
        m["sketch"] = json!({"id": id, "name": name});
    }
    if let Some((id, kind)) = &t.piece {
        m["piece"] = json!({"id": id, "kind": kind});
    }
    if let Some(e) = t.end {
        m["end"] = json!(e);
    }
    m
}

/// The audit's findings as the check lists them (ScanRuler's `auditOut`).
pub fn audit_out(findings: &[Finding], t: &Topology, features: &[Feature], full: bool, brief_per_kind: usize) -> AuditOut {
    let mut lists: Vec<(String, Vec<Value>)> = Vec::new();
    let mut brief = Vec::new();
    let mut counts: Vec<(FindingKind, Vec<&Finding>)> = Vec::new();
    for f in findings {
        let traced: Vec<(usize, FaceKind, i64, Option<String>, Option<crate::audit::FaceTrace>, DVec3)> = f
            .faces
            .iter()
            .filter_map(|i| t.faces.get(*i).map(|x| (*i, x)))
            .map(|(i, x)| {
                let fp = &x.fingerprint;
                (i, fp.kind, fp.origin, features.iter().find(|y| y.id as i64 == fp.origin).map(|y| y.name.clone()), trace_face(features, fp), fp.centroid)
            })
            .collect();
        let faces_json: Vec<Value> = traced
            .iter()
            .map(|(i, k, o, n, tr, c)| {
                let mut m = json!({"index": i, "kind": k.name(), "madeBy": {"id": o, "name": n}});
                if full {
                    m["centroid"] = json!([c.x, c.y, c.z]);
                    if let Some(tr) = tr {
                        m["trace"] = trace_json(tr);
                    }
                }
                m
            })
            .collect();
        let edges_json: Vec<Value> = f.edges.iter().filter_map(|i| t.edges.get(*i).map(|e| (*i, e))).map(|(i, e)| json!({"index": i, "kind": e.fingerprint.kind.name(), "length": e.fingerprint.length, "mid": [e.fingerprint.mid.x, e.fingerprint.mid.y, e.fingerprint.mid.z]})).collect();
        let repair = repair_for(f, &traced);
        let mut row = json!({"at": [f.at.x, f.at.y, f.at.z], "off": f.off, "unit": if f.deg { "deg" } else { "mm" }, "note": f.note, "faces": faces_json, "edges": edges_json, "repair": repair});
        if let Some(a) = f.axis {
            row["axis"] = json!(a.to_string());
        }
        if f.kind == FindingKind::Step {
            if let Some(b) = f.between {
                let named = |id: i64| features.iter().find(|x| x.id as i64 == id).map_or(Value::Null, |x| json!(x.name));
                row["form"] = json!(f.form.map(|x| x.name()));
                row["between"] = if b[0] == b[1] { json!([named(b[0])]) } else { json!([named(b[0]), named(b[1])]) };
                row["count"] = json!(f.count);
                if f.hair {
                    row["hair"] = json!(true);
                }
                if let Some(r) = f.riser {
                    row["riser"] = json!(r);
                }
            }
        } else if let Some(places) = &f.places {
            row["count"] = json!(f.count);
            row["places"] = Value::Array(places.iter().map(|p| r3s(*p)).collect());
        }
        let list = f.kind.list().to_string();
        match lists.iter_mut().find(|l| l.0 == list) {
            Some(l) => l.1.push(row),
            None => lists.push((list, vec![row])),
        }
        let seen = counts.iter().find(|c| c.0 == f.kind).map_or(0, |c| c.1.len());
        if seen < brief_per_kind {
            brief.push(brief_line(f, &traced, &repair));
        }
        match counts.iter_mut().find(|c| c.0 == f.kind) {
            Some(c) => c.1.push(f),
            None => counts.push((f.kind, vec![f])),
        }
    }
    let get = |k: FindingKind| counts.iter().find(|c| c.0 == k).map(|c| c.1.clone());
    let many = |fs: &[&Finding]| fs.iter().map(|f| if f.places.is_some() { f.count.unwrap_or(1) } else { 1 }).sum::<usize>();
    let list = |fs: &[&Finding], what: &dyn Fn(&Finding) -> String| fs.iter().map(|f| format!("{}{}", what(f), if f.places.is_some() { format!(" ×{}", f.count.unwrap_or(1)) } else { String::new() })).collect::<Vec<_>>().join(", ");
    let pair = |f: &Finding| format!("{}+{} ({} {})", f.faces.first().copied().unwrap_or(0), f.faces.get(1).copied().unwrap_or(0), js(f.off), if f.deg { "°" } else { "mm" });
    let at = |f: &Finding| format!("({}, {}, {})", js(round3(f.at.x)), js(round3(f.at.y)), js(round3(f.at.z)));
    let edge0 = |f: &Finding| f.edges.first().map_or("?".to_string(), |e| e.to_string());
    let mut warnings = Vec::new();
    if let Some(k) = get(FindingKind::Kink) {
        let n = many(&k);
        warnings.push(format!("{n} place{} where two faces meet at a hair's angle instead of flush or tangent: edge{} {}.", plural(n), plural(k.len()), list(&k, &|f| format!("{} ({}°)", edge0(f), js(f.off)))));
    }
    if let Some(steps) = get(FindingKind::Step) {
        let name = |id: i64| features.iter().find(|x| x.id as i64 == id).map_or(format!("feature {id}"), |x| x.name.clone());
        let line = |f: &Finding| {
            let who = match f.between {
                Some(b) if b[0] == b[1] => name(b[0]),
                Some(b) => format!("{}+{}", name(b[0]), name(b[1])),
                None => "?".into(),
            };
            format!("{who} {}{} ({}{} mm, faces {}+{})", f.count.unwrap_or(1), if f.hair { " hairline" } else { "" }, if f.count.unwrap_or(1) > 1 { "widest " } else { "" }, js(f.off), f.faces.first().copied().unwrap_or(0), f.faces.get(1).copied().unwrap_or(0))
        };
        let across: Vec<&Finding> = steps.iter().copied().filter(|f| f.between.is_some_and(|b| b[0] != b[1])).collect();
        let within: Vec<&Finding> = steps.iter().copied().filter(|f| !f.between.is_some_and(|b| b[0] != b[1])).collect();
        let mut parts = Vec::new();
        if !across.is_empty() {
            parts.push(format!("between features: {}", across.iter().map(|f| line(f)).collect::<Vec<_>>().join(", ")));
        }
        if !within.is_empty() {
            parts.push(format!("within one feature: {}", within.iter().map(|f| line(f)).collect::<Vec<_>>().join(", ")));
        }
        warnings.push(format!("Faces a step apart — {}.", parts.join("; ")));
    }
    if let Some(p) = get(FindingKind::NearParallel) {
        let n = many(&p);
        warnings.push(format!("{n} pair{} of flat faces nearly parallel: faces {}.", plural(n), list(&p, &pair)));
    }
    if let Some(a) = get(FindingKind::NearAxis) {
        let n = many(&a);
        warnings.push(format!("{n} face{} nearly square to a coordinate axis but off it: {}.", plural(n), list(&a, &|f| format!("{} ({}° off {})", f.faces.first().copied().unwrap_or(0), js(f.off), f.axis.unwrap_or('?')))));
    }
    if let Some(e) = get(FindingKind::FreeEdge) {
        warnings.push(format!("{} edge{} with one face: edge{} {}.", e.len(), plural(e.len()), plural(e.len()), list(&e, &|f| format!("{} at {}", edge0(f), at(f)))));
    }
    if let Some(e) = get(FindingKind::NonManifold) {
        warnings.push(format!("{} edge{} where three or more faces meet: edge{} {}.", e.len(), plural(e.len()), plural(e.len()), list(&e, &|f| format!("{} at {}", edge0(f), at(f)))));
    }
    if let Some(s) = get(FindingKind::ShortEdge) {
        let n = many(&s);
        warnings.push(format!("{n} edge{} under a tenth of a millimetre or so: {}.", plural(n), list(&s, &|f| format!("{} ({} mm)", edge0(f), js(f.off)))));
    }
    AuditOut { lists, warnings, brief }
}

type Traced = (usize, FaceKind, i64, Option<String>, Option<crate::audit::FaceTrace>, DVec3);

fn brief_line(f: &Finding, faces: &[Traced], repair: &str) -> String {
    let r = |v: f64| js((v * 100.0).round() / 100.0);
    let at = format!("{}, {}, {}", r(f.at.x), r(f.at.y), r(f.at.z));
    let off = format!("{}{}{}", js(f.off), if f.deg { "°" } else { " mm" }, f.axis.map_or(String::new(), |a| format!(" off {a}")));
    let edges = if f.edges.is_empty() { String::new() } else { format!("edge{} {}, ", plural(f.edges.len()), f.edges.iter().map(|e| e.to_string()).collect::<Vec<_>>().join(", ")) };
    let named = faces.iter().map(|x| format!("{} ({})", x.0, x.3.clone().unwrap_or_else(|| "?".into()))).collect::<Vec<_>>().join(" + ");
    let also = match &f.places {
        Some(p) => format!(", ×{} alike, also at {}", f.count.unwrap_or(1), p.iter().map(|q| format!("({}, {}, {})", r(q.x), r(q.y), r(q.z))).collect::<Vec<_>>().join(", ")),
        None => String::new(),
    };
    format!("{} {off} at ({at}){also}: {edges}face{} {named} — {repair}", f.kind.name(), plural(faces.len()))
}

fn repair_for(f: &Finding, faces: &[Traced]) -> String {
    let piece = |x: &Traced| -> Option<String> {
        let t = x.4.as_ref()?;
        match (&t.piece, &t.sketch) {
            (Some((id, kind)), Some((_, name))) => Some(format!("{kind} {id} of {name}")),
            _ => None,
        }
    };
    let sketch_id = |x: &Traced| x.4.as_ref().and_then(|t| t.sketch.as_ref().map(|s| s.0));
    let a = faces.first();
    let b = faces.get(1);
    match f.kind {
        FindingKind::NearAxis => {
            if let Some(p) = a.and_then(piece) {
                return format!("{p} leans {}° — a horizontal or vertical relation on it (reverse.sketch.edit), or the fit with alignAxes, squares it; reverse.sketch.candidates shows what else it is near.", js(f.off));
            }
            if let Some(t) = a.and_then(|x| x.4.as_ref()).filter(|t| t.role == "cap") {
                return format!("the cap of {} leans with its sketch plane — the plane, not the sketch, is what to square.", t.feature.1);
            }
            format!("the face leans {}° off the {} axis — square the sketch plane or the piece behind it, if it was meant square.", js(f.off), f.axis.unwrap_or('?'))
        }
        FindingKind::NearParallel => match (a.and_then(piece), b.and_then(piece)) {
            (Some(pa), Some(pb)) if a.and_then(sketch_id) == b.and_then(sketch_id) => format!("a parallel relation between {pa} and {pb} (reverse.sketch.edit) — or horizontal or vertical on both."),
            (Some(pa), Some(pb)) => format!("{pa} and {pb} are in different sketches: hold both to a coordinate plane, or project one into the other's sketch (project)."),
            _ => format!("the two faces lean {}° to each other — parallel in the sketches behind them, if meant so.", js(f.off)),
        },
        FindingKind::Step => {
            let (pa, pb) = (a.and_then(piece), b.and_then(piece));
            match f.form {
                Some(crate::audit::StepForm::Round) => match (pa, pb) {
                    (Some(pa), Some(pb)) if a.and_then(sketch_id) == b.and_then(sketch_id) => format!("{pa} and {pb} are one round drawn twice — one piece for both, or a concentric relation and equal radii (reverse.sketch.edit)."),
                    (Some(pa), Some(pb)) => {
                        let sa = a.and_then(|x| x.4.as_ref()).and_then(|t| t.sketch.as_ref()).map_or(String::new(), |s| s.1.clone());
                        let sb = b.and_then(|x| x.4.as_ref()).and_then(|t| t.sketch.as_ref()).map_or(String::new(), |s| s.1.clone());
                        let ap = a.and_then(|x| x.4.as_ref()).and_then(|t| t.piece).map_or(String::new(), |p| format!("{} {}", p.1, p.0));
                        format!("{pa} and {pb} are one round drawn twice in two sketches — project {pb} into {sa} in place of {ap}, or fit that level into {sb} (fit.level).")
                    }
                    _ => "one round drawn twice — one sketch piece for both: project the one, or fit both levels into one sketch (fit.level).".into(),
                },
                Some(crate::audit::StepForm::Tangent) => {
                    let round = faces.iter().find(|x| x.1 == FaceKind::Cylinder).or(a);
                    format!("{} was likely meant to touch the other feature’s face — a tangent relation to the face’s edge projected into its sketch, or let it cross clearly.", round.and_then(piece).unwrap_or_else(|| "the round".into()))
                }
                Some(crate::audit::StepForm::Foot) => {
                    let round = faces.iter().find(|x| x.1 == FaceKind::Cylinder).or(a);
                    format!("{} nearly starts on the other feature’s face — hold its end on that face (project the face’s edge into the sketch and put the end on it), or let it start clearly off it.", round.and_then(piece).unwrap_or_else(|| "the round’s profile".into()))
                }
                _ => {
                    let caps = a.and_then(|x| x.4.as_ref()).is_some_and(|t| t.role == "cap") && b.and_then(|x| x.4.as_ref()).is_some_and(|t| t.role == "cap");
                    if caps {
                        let na = a.and_then(|x| x.4.as_ref()).map_or(String::new(), |t| t.feature.1.clone());
                        let nb = b.and_then(|x| x.4.as_ref()).map_or(String::new(), |t| t.feature.1.clone());
                        return format!("{na} and {nb} end {} mm apart — give both extents one parameter, or extrude the later one up to the earlier one's face.", js(f.off));
                    }
                    match (pa, pb) {
                        (Some(pa), Some(pb)) => format!("{pa} and {pb} lie {} mm apart across their faces — project one into the other's sketch (project) so they are one line, or dimension both from one coordinate plane.", js(f.off)),
                        _ => format!("two faces {} mm apart that were likely meant as one — one height or one plane for both: a shared parameter, or the earlier face's edge projected into the later sketch.", js(f.off)),
                    }
                }
            }
        }
        FindingKind::Kink => match (a.and_then(piece), b.and_then(piece)) {
            (Some(pa), Some(pb)) if a.and_then(sketch_id) == b.and_then(sketch_id) => format!("a tangent relation between {pa} and {pb} (reverse.sketch.edit) — reverse.sketch.candidates lists it with the evidence."),
            _ => "a tangent relation in the sketch where the two pieces meet, or a round (fillet) in place of the sketched arc.".into(),
        },
        FindingKind::ShortEdge => "a sliver — a boolean's leftover (tiny lists faces under an area), or two corners that nearly meet: join them in the sketch, or round the edge away.".into(),
        FindingKind::FreeEdge => "the body is open here — a face is missing beside this edge: find the step that opened it (roundTrip: \"find\"), and close it there.".into(),
        FindingKind::NonManifold | FindingKind::Contact => {
            let mut makers: Vec<String> = Vec::new();
            for x in faces {
                if let Some(n) = &x.3 {
                    if !makers.contains(n) {
                        makers.push(n.clone());
                    }
                }
            }
            format!("{} only touch here at a tangent: let the one cross the other clearly (0.05 mm or more — a circle a hair larger or moved in) or meet it with a round; a point contact is never meant.", if makers.is_empty() { "two faces".to_string() } else { format!("{} ", makers.join(" and ")).trim_end().to_string() })
        }
    }
}

// ---- clearance ----

/// How two bodies stand to each other: the least distance between them, and the volume they
/// share.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Clearance {
    pub gap: f64,
    pub overlap: f64,
}

/// The clearance between two solid bodies (ScanRuler's kernel `clearance`): the exact minimum
/// distance, 0 where they touch or run into each other, and the volume of their common part.
pub fn clearance(a: &Body, b: &Body) -> Result<Clearance, String> {
    let ma = a.display_mesh(crate::build::mesh_deflection(crate::build::diagonal_of(a))).map_err(|e| e.to_string())?;
    let mb = b.display_mesh(crate::build::mesh_deflection(crate::build::diagonal_of(b))).map_err(|e| e.to_string())?;
    let ca = crate::build::mesh_deflection(crate::build::diagonal_of(a));
    let cb = crate::build::mesh_deflection(crate::build::diagonal_of(b));
    let sa = kernel::DistanceShape::new(a, &ma, ca);
    let sb = kernel::DistanceShape::new(b, &mb, cb);
    let overlap = match kernel::boolean(a, b, BoolOp::Intersect) {
        Ok(Some(common)) => kernel::measure(&common).map(|m| m.volume.max(0.0)).unwrap_or(0.0),
        Ok(None) => 0.0,
        Err(e) => return Err(format!("The overlap could not be measured: {e}")),
    };
    if overlap > 0.0 {
        return Ok(Clearance { gap: 0.0, overlap });
    }
    let d = kernel::body_distance(&sa, &sb, f64::INFINITY).map_err(|e| e.to_string())?;
    let gap = d.map_or(f64::INFINITY, |d| d.distance);
    Ok(Clearance { gap: if gap < 1e-9 { 0.0 } else { gap }, overlap: 0.0 })
}

/// A length in millimetres as the status line writes it ("12.700 mm").
pub fn len_words(mm: f64) -> String {
    format!("{} mm", crate::js_fmt::to_fixed(mm, 3))
}

/// The clearance in a sentence, as the status line says it (ScanRuler's `session.clearance`).
pub fn clearance_text(a: &str, b: &str, c: &Clearance) -> String {
    if c.gap > 0.0 {
        format!("{a} and {b} stand {} apart at the closest.", len_words(c.gap))
    } else if c.overlap > 1e-6 {
        format!("{a} and {b} run into each other: {} mm³ in common.", crate::js_fmt::to_fixed(c.overlap, 3))
    } else {
        format!("{a} and {b} touch, with no clearance and no overlap.")
    }
}

/// What two solid bodies share, meshed for drawing, with its volume — `None` where they share
/// none (ScanRuler's `sharedSolid`, on the kernel's common rather than Manifold's).
pub fn shared_solid(a: &Body, b: &Body) -> Option<(BodyMesh, f64)> {
    let common = kernel::boolean(a, b, BoolOp::Intersect).ok()??;
    let volume = kernel::measure(&common).ok()?.volume;
    if !(volume > 1e-9) {
        return None;
    }
    let m = crate::build::body_mesh(&common).ok()?;
    Some((m, volume))
}

/// The positions of a kernel point.
pub fn point(p: solidcraft_geom::Vec3) -> DVec3 {
    dv3(p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audit::AUDIT_DEFAULTS;
    use crate::build::{body_mesh, build};
    use crate::frame::WorldAxis;
    use crate::model::{BodyOp, ExtrudeExtent, ExtrudeFeature, FeatureKind, Part, SketchFeature};
    use crate::sketch::Sketch;
    use glam::DVec2;

    fn boxes(second_at: f64) -> crate::build::Built {
        let mut part = Part::default();
        for (x0, op) in [(0.0, BodyOp::Join), (second_at, BodyOp::New)] {
            let mut s = Sketch::default();
            s.add_rectangle(DVec2::new(x0, 0.0), DVec2::new(x0 + 10.0, 10.0));
            let regions = crate::build::default_regions(&crate::regions::find_regions(&s));
            let sk = part.add(FeatureKind::Sketch(SketchFeature::on_world(WorldAxis::Z, 0.0, s)));
            part.add(FeatureKind::Extrude(ExtrudeFeature::new(sk, regions, ExtrudeExtent::Distance { from: 0.0, to: 10.0 }, op)));
        }
        build(&part)
    }

    // Ported from ScanRuler's tests/reCheckVoids.test.ts: a hollow sealed inside is a shell of
    // its own, wound inward, with its volume and box.
    #[test]
    fn a_void_sealed_inside_is_found_with_its_volume_and_box() {
        let cube = |size: f64, inward: bool| {
            let m = mm_core::primitives::cube(size, 1);
            let pos: Vec<DVec3> = m.positions().to_vec();
            let tris: Vec<[u32; 3]> = m.triangle_ids().map(|t| m.triangle(t)).map(|t| if inward { [t[0], t[2], t[1]] } else { t }).collect();
            (pos, tris)
        };
        let (mut positions, mut triangles) = cube(10.0, false);
        let (inner, inner_tris) = cube(4.0, true);
        let base = positions.len() as u32;
        positions.extend(inner);
        triangles.extend(inner_tris.iter().map(|t| [t[0] + base, t[1] + base, t[2] + base]));
        let mesh = BodyMesh { tri_face: vec![0; triangles.len()], positions, triangles, ..Default::default() };
        let shells = mesh_shells(&mesh);
        assert_eq!(shells.len(), 2);
        let void = shells.iter().find(|s| s.0 < 0.0).unwrap();
        assert!((void.0 + 64.0).abs() < 1e-9);
        assert!((void.1 - DVec3::splat(-2.0)).length() < 1e-9 && (void.2 - DVec3::splat(2.0)).length() < 1e-9);
    }

    #[test]
    fn a_box_is_sound_and_clean() {
        let built = boxes(30.0);
        let b = &built.bodies[0];
        let mesh = body_mesh(&b.body).unwrap();
        let c = check_body(&b.body, &b.topology, &mesh, TINY_FACE_MM2);
        assert!(c.valid && c.closed, "{:?}", c.problems_of_kernel);
        assert_eq!(c.solids, 1);
        assert!((c.volume - 1000.0).abs() < 1e-3);
        assert!((c.area - 600.0).abs() < 1e-3);
        assert_eq!(c.faces, 6);
        assert!(c.tiny.is_empty() && c.narrow.is_empty() && c.untessellated.is_empty());
        assert_eq!(c.odd.edges.len(), 0);
        assert!(c.voids.is_empty());
        let rb = c.read_back.unwrap();
        assert!(rb.failed.is_none() && rb.closed && rb.edges.is_empty(), "{rb:?}");
        let answer = check_answer(1.0, "Solid 1", &b.body, &b.topology, &mesh, &[], TINY_FACE_MM2, AuditMode::Brief, AUDIT_DEFAULTS, None);
        assert_eq!(answer["ok"], json!(true));
        assert_eq!(answer["clean"], json!(true));
        assert!(answer.get("problems").is_none());
    }

    #[test]
    fn two_boxes_apart_stand_their_gap_apart_and_overlapping_share_volume() {
        let built = boxes(15.0);
        let c = clearance(&built.bodies[0].body, &built.bodies[1].body).unwrap();
        assert!((c.gap - 5.0).abs() < 1e-6, "{c:?}");
        assert_eq!(c.overlap, 0.0);
        assert_eq!(clearance_text("Solid 1", "Solid 2", &c), "Solid 1 and Solid 2 stand 5.000 mm apart at the closest.");
        let built = boxes(5.0);
        let c = clearance(&built.bodies[0].body, &built.bodies[1].body).unwrap();
        assert_eq!(c.gap, 0.0);
        assert!((c.overlap - 500.0).abs() < 1e-3, "{c:?}");
        let (mesh, v) = shared_solid(&built.bodies[0].body, &built.bodies[1].body).unwrap();
        assert!((v - 500.0).abs() < 1e-3);
        assert!(!mesh.triangles.is_empty());
        assert_eq!(clearance_text("A", "B", &c), "A and B run into each other: 500.000 mm³ in common.");
    }
}
