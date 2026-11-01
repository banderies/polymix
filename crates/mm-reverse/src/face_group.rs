// SPDX-License-Identifier: AGPL-3.0-only
//! What one click on a face takes for a Delete face: the whole of the round, the hole or the
//! boss the face is part of, so the kernel has a feature to take off and heal over rather than
//! a face out of the middle of one — and how the faces either side of each of a body's edges
//! meet, read off its mesh, which that takes. Ported from ScanRuler
//! (`plugins/reverse/core/faceGroup.ts`, and the joins of `plugins/reverse/core/edgeLines.ts`),
//! © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! The kernel meshes each face with vertices of its own, so a side of a triangle that no other
//! triangle has is on its face's outline; the face across an edge has its own side in the same
//! place, and the outward surface normals at the two sides' ends say how the faces meet there:
//! of one face, a seam; agreeing, tangent — or a split where the two faces lie on one surface;
//! else sharp. Each edge takes the join most of its sides have.

use std::collections::{HashMap, HashSet, VecDeque};

use glam::DVec3;

use crate::topo::{on_one_surface, FaceInfo, FaceKind, Topology};

/// How the faces either side of an edge meet: at an angle (`Sharp`); smoothly, one surface
/// running into another (`Tangent`); smoothly on one surface cut in two faces (`Split`); or one
/// face meeting itself where it closes round (`Seam`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EdgeJoin {
    Sharp,
    Tangent,
    Split,
    Seam,
}

/// The joins in the order a tie between them goes: the plainest first.
const JOINS: [EdgeJoin; 4] = [EdgeJoin::Sharp, EdgeJoin::Tangent, EdgeJoin::Split, EdgeJoin::Seam];

/// Two faces whose normals agree this closely at both ends of a side meet tangent there. A
/// round or a tangent relation is out by far less; a draft of half a degree is an edge to see.
fn tangent_cos() -> f64 {
    (0.25f64).to_radians().cos()
}

/// A body's mesh as the kernel makes it for display: each face meshed with vertices of its own,
/// each vertex carrying its face's outward surface normal, and each triangle's face.
#[derive(Debug, Clone, Copy)]
pub struct FacedTriangles<'a> {
    pub positions: &'a [DVec3],
    pub normals: &'a [DVec3],
    pub triangles: &'a [[u32; 3]],
    pub tri_face: &'a [u32],
}

/// Every side of the triangles that one triangle alone has, as (its two vertices, the
/// triangle): the faces' outlines — every edge of the body, a periodic face's seam among them
/// (its two sides are different vertices in one place). A side with no length — the pole of a
/// sphere — is left out.
fn outline_sides(positions: &[DVec3], triangles: &[[u32; 3]]) -> Vec<(u32, u32, usize)> {
    // A side by its two vertices, the lower first: the triangle that has it, or none once a
    // second one does.
    let mut owner: HashMap<(u32, u32), Option<usize>> = HashMap::new();
    let mut order: Vec<(u32, u32)> = Vec::new();
    for (t, tri) in triangles.iter().enumerate() {
        for k in 0..3 {
            let (a, b) = (tri[k], tri[(k + 1) % 3]);
            let key = if a < b { (a, b) } else { (b, a) };
            match owner.get_mut(&key) {
                Some(slot) => *slot = None,
                None => {
                    owner.insert(key, Some(t));
                    order.push(key);
                }
            }
        }
    }
    let mut out = Vec::new();
    for key in order {
        let Some(Some(t)) = owner.get(&key).copied() else { continue };
        let (Some(pa), Some(pb)) = (positions.get(key.0 as usize), positions.get(key.1 as usize)) else { continue };
        if pa == pb {
            continue;
        }
        out.push((key.0, key.1, t));
    }
    out
}

/// A point by where it lies, to a hundred-thousandth of a millimetre.
fn place_of(p: DVec3) -> (i64, i64, i64) {
    ((p.x * 1e5).round() as i64, (p.y * 1e5).round() as i64, (p.z * 1e5).round() as i64)
}

/// A side by where it lies, either way round — the faces on either side of an edge have the
/// same nodes there, each in its own vertices.
fn place_key(positions: &[DVec3], a: u32, b: u32) -> ((i64, i64, i64), (i64, i64, i64)) {
    let at = |v: u32| positions.get(v as usize).copied().map(place_of).unwrap_or((0, 0, 0));
    let (ka, kb) = (at(a), at(b));
    if ka < kb {
        (ka, kb)
    } else {
        (kb, ka)
    }
}

/// The squared distance from a point to a polyline, or `limit` once it is plain that it is no
/// nearer.
fn polyline_distance2(p: &[DVec3], at: DVec3, limit: f64) -> f64 {
    let mut best = limit;
    for w in p.windows(2) {
        let (a, d) = (w[0], w[1] - w[0]);
        let ll = d.length_squared();
        let t = if ll > 0.0 { ((at - a).dot(d) / ll).clamp(0.0, 1.0) } else { 0.0 };
        let e = a + d * t - at;
        best = best.min(e.length_squared());
    }
    best
}

/// How the faces either side of each of a body's edges meet, by the edge's hash (ScanRuler's
/// `edgeJoins` over `edgeShares`): an outline side goes to the edge of its face whose polyline
/// passes nearest the side's middle, and each edge takes the join most of its sides have — a
/// tie going to the plainest. An edge given no side is sharp.
pub fn edge_joins(mesh: FacedTriangles, topology: &Topology) -> HashMap<u32, EdgeJoin> {
    let positions = mesh.positions;
    let by_hash: HashMap<u32, &FaceInfo> = topology.faces.iter().map(|f| (f.hash, f)).collect();
    let one_surface = |a: u32, b: u32| match (by_hash.get(&a), by_hash.get(&b)) {
        (Some(fa), Some(fb)) => on_one_surface(&fa.fingerprint, &fb.fingerprint),
        _ => false,
    };
    // The edges round each face, each with the box round its polyline.
    struct Candidate<'e> {
        hash: u32,
        p: &'e [DVec3],
        lo: DVec3,
        hi: DVec3,
    }
    let mut round: HashMap<u32, Vec<Candidate>> = HashMap::new();
    for e in &topology.edges {
        if e.polyline.len() < 2 {
            continue;
        }
        let lo = e.polyline.iter().fold(DVec3::splat(f64::INFINITY), |m, p| m.min(*p));
        let hi = e.polyline.iter().fold(DVec3::splat(f64::NEG_INFINITY), |m, p| m.max(*p));
        let mut seen: Vec<u32> = Vec::new();
        for f in &e.faces {
            if seen.contains(f) {
                continue;
            }
            seen.push(*f);
            round.entry(*f).or_default().push(Candidate { hash: e.hash, p: &e.polyline, lo, hi });
        }
    }
    // Each outline side's join, from its partner in the same place.
    let sides = outline_sides(positions, mesh.triangles);
    let keys: Vec<_> = sides.iter().map(|(a, b, _)| place_key(positions, *a, *b)).collect();
    let mut places: HashMap<_, Vec<usize>> = HashMap::new();
    for (s, k) in keys.iter().enumerate() {
        places.entry(*k).or_default().push(s);
    }
    let face_of = |t: usize| mesh.tri_face.get(t).copied();
    let normal = |v: u32| mesh.normals.get(v as usize).copied().unwrap_or(DVec3::ZERO);
    let pos = |v: u32| positions.get(v as usize).copied().unwrap_or(DVec3::ZERO);
    let mut joins = vec![EdgeJoin::Sharp; sides.len()];
    let cos = tangent_cos();
    for list in places.values() {
        let [s, t] = list[..] else { continue };
        let (fs, ft) = (face_of(sides[s].2), face_of(sides[t].2));
        let mut join = EdgeJoin::Sharp;
        if fs.is_some() && fs == ft {
            join = EdgeJoin::Seam;
        } else if !mesh.normals.is_empty() {
            let (a, b) = (sides[s].0, sides[s].1);
            // The partner's vertices at a's place and at b's.
            let (mut ta, mut tb) = (sides[t].0, sides[t].1);
            if (pos(a) - pos(ta)).length_squared() > (pos(a) - pos(tb)).length_squared() {
                std::mem::swap(&mut ta, &mut tb);
            }
            if normal(a).dot(normal(ta)) >= cos && normal(b).dot(normal(tb)) >= cos {
                join = match (fs, ft) {
                    (Some(x), Some(y)) if one_surface(x, y) => EdgeJoin::Split,
                    _ => EdgeJoin::Tangent,
                };
            }
        }
        joins[s] = join;
        joins[t] = join;
    }
    // Each edge's sides counted by join.
    let mut votes: HashMap<u32, [usize; 4]> = HashMap::new();
    for (s, (a, b, tri)) in sides.iter().enumerate() {
        let Some(candidates) = face_of(*tri).and_then(|f| round.get(&f)) else { continue };
        let m = (pos(*a) + pos(*b)) / 2.0;
        let mut best = f64::INFINITY;
        let mut which: Option<u32> = None;
        for c in candidates {
            let outside = (c.lo - m).max(DVec3::ZERO).max(m - c.hi);
            if outside.length_squared() >= best {
                continue;
            }
            let d = polyline_distance2(c.p, m, best);
            if d < best {
                best = d;
                which = Some(c.hash);
            }
        }
        let Some(which) = which else { continue };
        let count = votes.entry(which).or_insert([0; 4]);
        let k = JOINS.iter().position(|j| *j == joins[s]).unwrap_or(0);
        count[k] += 1;
    }
    topology
        .edges
        .iter()
        .map(|e| {
            let join = match votes.get(&e.hash) {
                Some(count) => {
                    let most = count.iter().copied().max().unwrap_or(0);
                    JOINS[count.iter().position(|c| *c == most).unwrap_or(0)]
                }
                None => EdgeJoin::Sharp,
            };
            (e.hash, join)
        })
        .collect()
}

/// A round a tangent flood runs on into: a face of a fillet's chain — not a flat, which is what
/// a round runs into either side, nor a whole-turn cylinder or cone, which is a boss's wall or a
/// bore's.
fn blend(f: &FaceInfo) -> bool {
    match f.fingerprint.kind {
        FaceKind::Plane => false,
        FaceKind::Cylinder | FaceKind::Cone => f.full_turn == Some(false),
        _ => true,
    }
}

/// Whether a face is a whole-turn cylinder or cone: a boss's wall or a bore's.
fn wall(f: &FaceInfo) -> bool {
    matches!(f.fingerprint.kind, FaceKind::Cylinder | FaceKind::Cone) && f.full_turn != Some(false)
}

/// The faces, by hash, a click on `seed` takes for a delete:
///
/// - the face, and the faces it was split from on its own surface — the two halves of a bore,
///   which the viewport draws as one face;
/// - with `tangent`, from a round face on along every edge it meets another round at tangent
///   ([`blend`]) — the whole of a fillet's chain, up to the flats either side, a boss's wall
///   with the round at its foot;
/// - then every face the faces taken close in on all sides — the floor of a blind hole, the top
///   of a boss, a drill point's cone — but not a flat the rounds run into smoothly all the way
///   round: a block rounded on every edge keeps its flats.
///
/// `join_of` says how an edge's faces meet ([`edge_joins`]).
pub fn delete_face_group(topology: &Topology, seed: u32, join_of: impl Fn(u32) -> Option<EdgeJoin>, tangent: bool) -> Vec<u32> {
    let face_by_hash: HashMap<u32, &FaceInfo> = topology.faces.iter().map(|f| (f.hash, f)).collect();
    if !face_by_hash.contains_key(&seed) {
        return Vec::new();
    }
    let mut edges_of: HashMap<u32, Vec<usize>> = HashMap::new();
    for (i, e) in topology.edges.iter().enumerate() {
        let mut seen: HashSet<u32> = HashSet::new();
        for f in &e.faces {
            if seen.insert(*f) {
                edges_of.entry(*f).or_default().push(i);
            }
        }
    }
    let mut taken: HashSet<u32> = HashSet::from([seed]);
    let mut order = vec![seed];
    let mut queue = VecDeque::from([seed]);
    while let Some(h) = queue.pop_front() {
        let Some(at) = face_by_hash.get(&h).copied() else { continue };
        for i in edges_of.get(&at.hash).cloned().unwrap_or_default() {
            let Some(e) = topology.edges.get(i) else { continue };
            let Some(other) = e.faces.iter().copied().find(|f| *f != at.hash) else { continue };
            if taken.contains(&other) {
                continue;
            }
            let Some(next) = face_by_hash.get(&other).copied() else { continue };
            let join = join_of(e.hash);
            let split = join == Some(EdgeJoin::Split) && on_one_surface(&at.fingerprint, &next.fingerprint);
            // A boss's or a bore's wall clicked runs on into the round at its foot or its
            // mouth; past that, rounds only.
            let from = blend(at) || (at.hash == seed && at.fingerprint.kind != FaceKind::Plane);
            if !split && !(tangent && join == Some(EdgeJoin::Tangent) && from && blend(next)) {
                continue;
            }
            taken.insert(other);
            order.push(other);
            queue.push_back(other);
        }
    }
    // The faces closed in: every edge of the face has a face taken on its other side — a floor,
    // a cap — where it meets them at an edge somewhere, or they hold a bore's or a boss's wall.
    // A flat the rounds of a block run into smoothly all round is what they round, not a part
    // of them.
    let mut grew = true;
    while grew {
        grew = false;
        let walled = taken.iter().any(|h| face_by_hash.get(h).is_some_and(|f| wall(f)));
        for f in &topology.faces {
            if taken.contains(&f.hash) {
                continue;
            }
            let round: Vec<(u32, Option<EdgeJoin>)> = edges_of
                .get(&f.hash)
                .map(|list| list.iter().filter_map(|i| topology.edges.get(*i)).filter_map(|e| e.faces.iter().copied().find(|x| *x != f.hash).map(|other| (other, join_of(e.hash)))).collect())
                .unwrap_or_default();
            if round.is_empty() || !round.iter().all(|(other, _)| taken.contains(other)) {
                continue;
            }
            if !walled && !round.iter().any(|(_, j)| matches!(j, Some(EdgeJoin::Sharp) | None)) {
                continue;
            }
            taken.insert(f.hash);
            order.push(f.hash);
            grew = true;
        }
    }
    order
}
