// SPDX-License-Identifier: AGPL-3.0-only
//! Stable names for the faces and edges of a body. Ported from ScanRuler
//! (`plugins/reverse/core/topo.ts`, with the face and edge records of `buildTypes.ts`),
//! © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! A feature that fillets an edge or opens a face for a shell has to say which one, and an
//! index into the kernel's face list is not a name: the list is remade on every rebuild and an
//! earlier dimension changed reorders it. So a pick is stored as a fingerprint — the kind of
//! surface or curve, where its middle is, which way it faces or runs, how big it is, and which
//! feature first made it — and found again on the body as it stands before the feature, the
//! best candidate within a tolerance winning. A pick that cannot be found fails its feature
//! loudly, with the fingerprint's description in the message, never a silently substituted
//! neighbour.
//!
//! The kernel side (the builder, `build/mod.rs`) reads the geometry off the bodies and carries
//! the creating feature across steps through the kernel's history; this module is the
//! geometry and the matching, testable without the kernel.

use std::collections::HashSet;

use glam::{DAffine3, DVec3};
use serde::{Deserialize, Serialize};

/// What surface a face lies on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FaceKind {
    Plane,
    Cylinder,
    Cone,
    Sphere,
    Torus,
    Other,
}

impl FaceKind {
    pub fn name(self) -> &'static str {
        match self {
            FaceKind::Plane => "plane",
            FaceKind::Cylinder => "cylinder",
            FaceKind::Cone => "cone",
            FaceKind::Sphere => "sphere",
            FaceKind::Torus => "torus",
            FaceKind::Other => "other",
        }
    }

    /// Whether the face's direction is an axis, which has no sign of its own.
    pub fn is_round(self) -> bool {
        matches!(self, FaceKind::Cylinder | FaceKind::Cone | FaceKind::Torus)
    }
}

/// What curve an edge runs along.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EdgeKind {
    Line,
    Circle,
    Other,
}

impl EdgeKind {
    pub fn name(self) -> &'static str {
        match self {
            EdgeKind::Line => "line",
            EdgeKind::Circle => "circle",
            EdgeKind::Other => "other",
        }
    }
}

/// A face as a pick stores it.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FaceFingerprint {
    pub kind: FaceKind,
    /// The area-weighted centre of the face.
    pub centroid: DVec3,
    /// The outward normal at the middle of a plane (the sign tells a top from a bottom); the
    /// axis of a cylinder, cone or torus, its sign made canonical since an axis has none; the
    /// normal at the middle of anything else.
    pub dir: DVec3,
    pub area: f64,
    /// A cylinder's or a sphere's radius.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub radius: Option<f64>,
    /// A point on a cylinder's or a cone's axis, a sphere's centre: where a hole cut along the
    /// face's axis starts from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub center: Option<DVec3>,
    /// The id of the feature whose step first made the face; -1 where it is not known.
    pub origin: i64,
}

/// One face beside an edge, as an edge's fingerprint keeps it.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EdgeSide {
    pub kind: FaceKind,
    pub dir: DVec3,
}

/// An edge as a pick stores it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EdgeFingerprint {
    pub kind: EdgeKind,
    /// The point halfway along the edge.
    pub mid: DVec3,
    /// The tangent at the middle of a line, its sign made canonical; the axis of a circle; the
    /// tangent at the middle of anything else.
    pub dir: DVec3,
    pub length: f64,
    /// A circle's radius.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub radius: Option<f64>,
    /// The faces on either side — kind and direction only, which is what tells a top edge from
    /// a bottom one when both are the same line.
    pub faces: Vec<EdgeSide>,
    /// The id of the feature whose step first made the edge's newest face; -1 where none is
    /// known.
    pub origin: i64,
}

/// How far a candidate may be from the pick and still be it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MatchTolerance {
    /// Millimetres between the middles.
    pub position: f64,
    /// Radians between the directions.
    pub angle: f64,
    /// Relative difference in size (area, length, radius).
    pub size: f64,
}

/// The tolerance for a body of the given diagonal: an earlier dimension moved by a tenth of the
/// part still lands on the same face, a face turned by more than ten degrees is another face.
pub fn match_tolerance(diagonal: f64) -> MatchTolerance {
    MatchTolerance { position: (diagonal * 0.1).max(1.0), angle: 10f64.to_radians(), size: 0.5 }
}

/// A direction with its sign made canonical — the first component that is clearly not zero is
/// positive — for the things that have no sign of their own: an axis, a line's tangent.
pub fn canonical_dir(v: DVec3) -> DVec3 {
    let n = v.try_normalize().unwrap_or(DVec3::Z);
    for k in 0..3 {
        if n[k].abs() > 1e-6 {
            return if n[k] < 0.0 { -n } else { n };
        }
    }
    n
}

/// A face's fingerprint carried through a rigid motion of the part — an alignment applied after
/// the pick — so it is found again on the bodies rebuilt in the new frame. A plane's normal keeps
/// its sign; an axis is made canonical again, as the rebuilt body's will be.
pub fn transform_fingerprint(face: &FaceFingerprint, m: &DAffine3) -> FaceFingerprint {
    let turned = m.transform_vector3(face.dir);
    FaceFingerprint {
        centroid: m.transform_point3(face.centroid),
        dir: if face.kind.is_round() { canonical_dir(turned) } else { turned },
        center: face.center.map(|c| m.transform_point3(c)),
        ..*face
    }
}

/// An edge's fingerprint carried through a rigid motion of the part, as a face's is: its middle
/// moved, its direction and its faces' turned, made canonical again where they have no sign.
pub fn transform_edge_fingerprint(edge: &EdgeFingerprint, m: &DAffine3) -> EdgeFingerprint {
    EdgeFingerprint {
        mid: m.transform_point3(edge.mid),
        dir: canonical_dir(m.transform_vector3(edge.dir)),
        faces: edge
            .faces
            .iter()
            .map(|f| {
                let d = m.transform_vector3(f.dir);
                EdgeSide { kind: f.kind, dir: if f.kind.is_round() { canonical_dir(d) } else { d } }
            })
            .collect(),
        ..edge.clone()
    }
}

/// The angle between two directions, radians.
pub fn angle_between(a: DVec3, b: DVec3) -> f64 {
    let d = (a.dot(b) / (a.length().max(1e-12) * b.length().max(1e-12))).clamp(-1.0, 1.0);
    d.acos()
}

fn relative_difference(a: f64, b: f64) -> f64 {
    let m = a.abs().max(b.abs());
    if m > 1e-12 {
        (a - b).abs() / m
    } else {
        0.0
    }
}

/// How far a candidate face is from the wanted one as a sum of its distances in position,
/// direction and size, each in units of the tolerance; `None` when it is not a candidate at all
/// — another kind of surface, or past the tolerance on any count. A different creating feature
/// costs half a tolerance: the same face made by another feature is still the same face, but the
/// one made by the same feature is the better bet when two are close.
pub fn face_distance(wanted: &FaceFingerprint, candidate: &FaceFingerprint, tol: &MatchTolerance) -> Option<f64> {
    if wanted.kind != candidate.kind {
        return None;
    }
    let pos = (wanted.centroid - candidate.centroid).length() / tol.position;
    let ang = angle_between(wanted.dir, candidate.dir) / tol.angle;
    let mut size = relative_difference(wanted.area, candidate.area) / tol.size;
    if let (Some(a), Some(b)) = (wanted.radius, candidate.radius) {
        size = size.max(relative_difference(a, b) / tol.size);
    }
    if pos > 1.0 || ang > 1.0 || size > 1.0 {
        return None;
    }
    Some(pos + ang + size + if wanted.origin == candidate.origin { 0.0 } else { 0.5 })
}

/// The faces of an edge, matched by kind and direction alone: every face of the wanted edge
/// must find one of the candidate's within the angle.
fn faces_agree(wanted: &[EdgeSide], candidate: &[EdgeSide], tol: &MatchTolerance) -> Option<f64> {
    if wanted.is_empty() || candidate.is_empty() {
        return Some(0.0);
    }
    let mut taken: HashSet<usize> = HashSet::new();
    let mut sum = 0.0;
    for w in wanted {
        let mut best: Option<usize> = None;
        let mut best_ang = f64::INFINITY;
        for (i, c) in candidate.iter().enumerate() {
            if taken.contains(&i) || c.kind != w.kind {
                continue;
            }
            let ang = angle_between(w.dir, c.dir);
            if ang < best_ang {
                best_ang = ang;
                best = Some(i);
            }
        }
        let i = best?;
        if best_ang > tol.angle {
            return None;
        }
        taken.insert(i);
        sum += best_ang / tol.angle;
    }
    Some(sum / wanted.len() as f64)
}

/// As [`face_distance`], for an edge: its faces' kinds and directions too.
pub fn edge_distance(wanted: &EdgeFingerprint, candidate: &EdgeFingerprint, tol: &MatchTolerance) -> Option<f64> {
    if wanted.kind != candidate.kind {
        return None;
    }
    let pos = (wanted.mid - candidate.mid).length() / tol.position;
    let ang = angle_between(wanted.dir, candidate.dir) / tol.angle;
    let mut size = relative_difference(wanted.length, candidate.length) / tol.size;
    if let (Some(a), Some(b)) = (wanted.radius, candidate.radius) {
        size = size.max(relative_difference(a, b) / tol.size);
    }
    if pos > 1.0 || ang > 1.0 || size > 1.0 {
        return None;
    }
    let faces = faces_agree(&wanted.faces, &candidate.faces, tol)?;
    Some(pos + ang + size + faces + if wanted.origin == candidate.origin { 0.0 } else { 0.5 })
}

/// The tolerance of the second look, for a pick the first found nothing for: an earlier
/// dimension changed by more than a tenth of the part — a boss pulled from 7 to 17 mm moves its
/// top face and every edge on it by 10, and its side edges change length by more than half — is
/// still the same face or edge, with the same creating feature, the same kind, facing the same
/// way. The second look is taken only among the candidates the same feature made, so a face
/// that is truly gone is not quietly stood in for by a neighbour: position four times as far,
/// the size hardly counted, the direction as tight as ever.
fn relaxed_tolerance(tol: &MatchTolerance) -> MatchTolerance {
    MatchTolerance { position: tol.position * 4.0, angle: tol.angle, size: 0.9 }
}

/// The index of the candidate nearest the wanted fingerprint within the tolerance — or, when
/// none is, the nearest of those the same feature made within the relaxed tolerance — or `None`
/// when none is near enough.
pub fn match_face(wanted: &FaceFingerprint, candidates: &[FaceFingerprint], tol: &MatchTolerance) -> Option<usize> {
    let strict = best(candidates, |c| face_distance(wanted, c, tol));
    if strict.is_some() || wanted.origin < 0 {
        return strict;
    }
    let loose = relaxed_tolerance(tol);
    best(candidates, |c| if c.origin == wanted.origin { face_distance(wanted, c, &loose) } else { None })
}

pub fn match_edge(wanted: &EdgeFingerprint, candidates: &[EdgeFingerprint], tol: &MatchTolerance) -> Option<usize> {
    let strict = best(candidates, |c| edge_distance(wanted, c, tol));
    if strict.is_some() || wanted.origin < 0 {
        return strict;
    }
    let loose = relaxed_tolerance(tol);
    best(candidates, |c| if c.origin == wanted.origin { edge_distance(wanted, c, &loose) } else { None })
}

fn best<T>(candidates: &[T], distance: impl Fn(&T) -> Option<f64>) -> Option<usize> {
    let mut best_index = None;
    let mut best_distance = f64::INFINITY;
    for (i, c) in candidates.iter().enumerate() {
        if let Some(d) = distance(c) {
            if d < best_distance {
                best_distance = d;
                best_index = Some(i);
            }
        }
    }
    best_index
}

// ---- a body's faces and edges as a build reads them ----

/// A face of a built body: its fingerprint — what a pick stores and what a rebuild finds it
/// again by — and its hash, what the body's mesh names it by for the length of this build (its
/// index in the kernel's face order).
#[derive(Debug, Clone, PartialEq)]
pub struct FaceInfo {
    pub hash: u32,
    pub fingerprint: FaceFingerprint,
    /// For a round face read off a build, whether it goes the whole way round its axis — not
    /// part of the name a pick stores, so a pick taken back from a feature does not say.
    pub full_turn: Option<bool>,
    /// A point on the face itself (the centroid may lie off it, in a ring's hole): what the
    /// kernel's calls that name a face by a point are given.
    pub point: DVec3,
}

/// An edge of a built body.
#[derive(Debug, Clone, PartialEq)]
pub struct EdgeInfo {
    /// Its index in the kernel's edge order (and the display mesh's).
    pub hash: u32,
    pub fingerprint: EdgeFingerprint,
    /// The faces on either side, by hash — what says which edges bound a face.
    pub faces: Vec<u32>,
    /// The two ends and the unit tangent at each, pointing along the edge away from its start —
    /// what a tangent chain is walked by.
    pub ends: [DVec3; 2],
    pub end_tangents: [DVec3; 2],
    /// The edge sampled for drawing.
    pub polyline: Vec<DVec3>,
}

/// Every face and edge of a body with its fingerprint.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Topology {
    pub faces: Vec<FaceInfo>,
    pub edges: Vec<EdgeInfo>,
}

impl Topology {
    pub fn face(&self, hash: u32) -> Option<&FaceInfo> {
        self.faces.iter().find(|f| f.hash == hash)
    }

    pub fn edge(&self, hash: u32) -> Option<&EdgeInfo> {
        self.edges.iter().find(|e| e.hash == hash)
    }

    pub fn face_prints(&self) -> Vec<FaceFingerprint> {
        self.faces.iter().map(|f| f.fingerprint).collect()
    }

    pub fn edge_prints(&self) -> Vec<EdgeFingerprint> {
        self.edges.iter().map(|e| e.fingerprint.clone()).collect()
    }
}

/// The edges tangent-continuous with the seeds: from each seed, every edge meeting it end to end
/// whose tangent there agrees within `angle`, and so on along the chain — a fillet picked on one
/// side of a rounded rectangle runs all the way round it, as CAD fillets do. The seeds are in the
/// answer; the order is the walk's. The walk never steps onto an edge in `skip` — the splits
/// ([`split_edges`]) a picked edge can run on into in a straight line, where a face lying on its
/// neighbour's surface reaches on past the edge seen.
pub fn tangent_chain(edges: &[EdgeInfo], seeds: &[u32], skip: &HashSet<u32>, angle: f64, join_tolerance: f64) -> Vec<u32> {
    let mut taken: HashSet<u32> = HashSet::new();
    let mut out = Vec::new();
    let mut queue = std::collections::VecDeque::new();
    for &s in seeds {
        if edges.iter().any(|e| e.hash == s) && taken.insert(s) {
            out.push(s);
            queue.push_back(s);
        }
    }
    let cos_limit = angle.cos();
    while let Some(h) = queue.pop_front() {
        let Some(e) = edges.iter().find(|e| e.hash == h) else { continue };
        for k in 0..2 {
            let end = e.ends[k];
            let t = e.end_tangents[k];
            for other in edges {
                if taken.contains(&other.hash) || skip.contains(&other.hash) {
                    continue;
                }
                for j in 0..2 {
                    if (other.ends[j] - end).length() > join_tolerance {
                        continue;
                    }
                    if t.dot(other.end_tangents[j]).abs() < cos_limit {
                        continue;
                    }
                    taken.insert(other.hash);
                    out.push(other.hash);
                    queue.push_back(other.hash);
                    break;
                }
            }
        }
    }
    out
}

/// The angle a tangent chain is walked across by default: three degrees.
pub const CHAIN_ANGLE: f64 = 3.0 * std::f64::consts::PI / 180.0;
/// How far apart two ends may be and still meet, by default, mm.
pub const CHAIN_JOIN_MM: f64 = 1e-3;

/// How far apart, in millimetres, two surfaces may lie and still be taken for one — far looser
/// than the kernel's own, which keeps a face cut in two where its halves' radii came out a
/// millionth apart.
const ONE_SURFACE_MM: f64 = 1e-3;
/// And how far apart their axes may turn, in radians.
const ONE_SURFACE_ANGLE: f64 = 1e-4;

/// Whether two faces that meet tangent lie on one surface: the two a boolean leaves either side
/// of where it cut through a plane or a cylinder. The tangency is the caller's to have found.
/// With it, two planes meeting along an edge are one plane, and two cylinders or cones on one
/// axis are one (a cone's slope agrees where the two meet); two spheres have to share their
/// centre and radius. A torus or a surface of no named kind never is: its fingerprint does not
/// hold enough to tell.
pub fn on_one_surface(a: &FaceFingerprint, b: &FaceFingerprint) -> bool {
    if a.kind != b.kind {
        return false;
    }
    match a.kind {
        FaceKind::Plane => true,
        FaceKind::Cylinder | FaceKind::Cone => {
            let (Some(ac), Some(bc)) = (a.center, b.center) else { return false };
            if a.dir.dot(b.dir).abs() < ONE_SURFACE_ANGLE.cos() {
                return false;
            }
            // How far the one's axis point lies off the other's axis.
            let d = bc - ac;
            if (d - a.dir * d.dot(a.dir)).length() > ONE_SURFACE_MM {
                return false;
            }
            a.kind == FaceKind::Cone || matches!((a.radius, b.radius), (Some(ra), Some(rb)) if (ra - rb).abs() <= ONE_SURFACE_MM)
        }
        FaceKind::Sphere => match (a.center, b.center, a.radius, b.radius) {
            (Some(ac), Some(bc), Some(ra), Some(rb)) => (ac - bc).length() <= ONE_SURFACE_MM && (ra - rb).abs() <= ONE_SURFACE_MM,
            _ => false,
        },
        _ => false,
    }
}

/// The edges between two faces of one surface — the lines the viewer neither draws nor picks,
/// where a boolean left a face cut in two or a face lying on its neighbour's surface beside it.
/// They are no edge of the part's shape, so no round or chamfer runs along one. Two planes are
/// one where their outward normals agree, and two faces on one cylinder or sphere meet smoothly
/// wherever they meet. Two cones on one axis can meet at an angle, which their fingerprints do
/// not tell, so an edge between cones is kept.
pub fn split_edges(topology: &Topology) -> HashSet<u32> {
    let mut out = HashSet::new();
    for e in &topology.edges {
        let mut sides: Vec<u32> = Vec::new();
        for f in &e.faces {
            if !sides.contains(f) {
                sides.push(*f);
            }
        }
        let (Some(a), Some(b)) = (sides.first().and_then(|h| topology.face(*h)), sides.get(1).and_then(|h| topology.face(*h))) else { continue };
        let (a, b) = (&a.fingerprint, &b.fingerprint);
        if a.kind != b.kind {
            continue;
        }
        let one = if a.kind == FaceKind::Plane { a.dir.dot(b.dir) >= ONE_SURFACE_ANGLE.cos() } else { matches!(a.kind, FaceKind::Cylinder | FaceKind::Sphere) && on_one_surface(a, b) };
        if one {
            out.insert(e.hash);
        }
    }
    out
}

/// A length in a face's or an edge's words: tenths of a millimetre.
fn tenths(mm: f64) -> String {
    format!("{mm:.1}")
}

fn at(p: DVec3) -> String {
    format!("({}, {}, {})", tenths(p.x), tenths(p.y), tenths(p.z))
}

/// A face as a message names it: "cylinder Ø6.0 at (10.0, 10.0, 5.0)".
pub fn describe_face(f: &FaceFingerprint) -> String {
    let size = match f.kind {
        FaceKind::Cylinder | FaceKind::Sphere => f.radius.map_or(String::new(), |r| format!(" Ø{}", tenths(2.0 * r))),
        _ => format!(" {:.0} mm²", f.area),
    };
    let kind = if f.kind == FaceKind::Other { "face" } else { f.kind.name() };
    format!("{kind}{size} at {}", at(f.centroid))
}

/// An edge as a message names it: "circle Ø6.0 at (10.0, 10.0, 10.0)", "line 40.0 mm at
/// (20.0, 0.0, 10.0)".
pub fn describe_edge(e: &EdgeFingerprint) -> String {
    if let (EdgeKind::Circle, Some(r)) = (e.kind, e.radius) {
        return format!("circle Ø{} at {}", tenths(2.0 * r), at(e.mid));
    }
    let kind = if e.kind == EdgeKind::Other { "edge" } else { e.kind.name() };
    format!("{kind} {:.1} mm at {}", e.length, at(e.mid))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plane(centroid: [f64; 3], dir: [f64; 3], area: f64, origin: i64) -> FaceFingerprint {
        FaceFingerprint { kind: FaceKind::Plane, centroid: DVec3::from(centroid), dir: DVec3::from(dir), area, radius: None, center: None, origin }
    }

    fn near(a: DVec3, b: [f64; 3]) -> bool {
        (a - DVec3::from(b)).length() < 1e-9
    }

    #[test]
    fn make_an_axis_canonical_and_leave_a_normal_its_sign() {
        assert!(near(canonical_dir(DVec3::new(0.0, 0.0, -1.0)), [0.0, 0.0, 1.0]));
        assert!(near(canonical_dir(DVec3::new(-0.6, 0.8, 0.0)), [0.6, -0.8, 0.0]));
        assert!(near(canonical_dir(DVec3::new(0.0, -3.0, 4.0)), [0.0, 0.6, -0.8]));
    }

    #[test]
    fn match_the_nearest_face_of_the_same_kind_within_the_tolerance_and_no_other() {
        let tol = match_tolerance(50.0);
        let top = plane([20.0, 10.0, 10.0], [0.0, 0.0, 1.0], 800.0, 1);
        let candidates = [plane([20.0, 10.0, 0.0], [0.0, 0.0, -1.0], 800.0, 1), plane([21.0, 10.0, 10.0], [0.0, 0.0, 1.0], 840.0, 1), plane([20.0, 10.0, 40.0], [0.0, 0.0, 1.0], 800.0, 1)];
        // The bottom faces the other way, the far top is past the position tolerance (five
        // millimetres on this part, twenty on the second look), the moved top is it.
        assert_eq!(match_face(&top, &candidates, &tol), Some(1));
        assert_eq!(match_face(&FaceFingerprint { kind: FaceKind::Cylinder, ..top }, &candidates, &tol), None);
        assert_eq!(match_face(&top, &candidates[2..], &tol), None);
        // Made by the same feature wins between two equally near.
        let twins = [plane([20.0, 10.0, 10.0], [0.0, 0.0, 1.0], 800.0, 7), plane([20.0, 10.0, 10.0], [0.0, 0.0, 1.0], 800.0, 1)];
        assert_eq!(match_face(&top, &twins, &tol), Some(1));
    }

    fn line(mid: [f64; 3], up: f64) -> EdgeFingerprint {
        EdgeFingerprint {
            kind: EdgeKind::Line,
            mid: DVec3::from(mid),
            dir: DVec3::X,
            length: 40.0,
            radius: None,
            faces: vec![EdgeSide { kind: FaceKind::Plane, dir: DVec3::new(0.0, -1.0, 0.0) }, EdgeSide { kind: FaceKind::Plane, dir: DVec3::new(0.0, 0.0, up) }],
            origin: 1,
        }
    }

    #[test]
    fn tell_a_top_edge_from_the_bottom_one_under_it_by_their_faces() {
        let tol = match_tolerance(50.0);
        let wanted = line([20.0, 0.0, 10.0], 1.0);
        assert_eq!(match_edge(&wanted, &[line([20.0, 0.0, 0.0], -1.0), line([20.0, 0.0, 10.0], -1.0), line([20.0, 0.0, 10.0], 1.0)], &tol), Some(2));
        assert_eq!(match_edge(&wanted, &[line([20.0, 0.0, 10.0], -1.0)], &tol), None);
    }

    #[test]
    fn take_a_second_look_among_the_faces_the_same_feature_made() {
        // A 20 × 20 boss pulled from 7 to 17 mm: its top face moves 10 mm along its normal, its
        // side edges grow to more than double.
        let tol = match_tolerance((20.0f64 * 20.0 + 20.0 * 20.0 + 17.0 * 17.0).sqrt());
        let top = plane([10.0, 10.0, 7.0], [0.0, 0.0, 1.0], 400.0, 2);
        let grown = [plane([10.0, 10.0, 0.0], [0.0, 0.0, -1.0], 400.0, 2), plane([10.0, 10.0, 17.0], [0.0, 0.0, 1.0], 400.0, 2), plane([10.0, 0.0, 8.5], [0.0, -1.0, 0.0], 340.0, 2)];
        assert_eq!(match_face(&top, &grown, &tol), Some(1));
        // Not among faces another feature made: a face that is gone stays gone.
        assert_eq!(match_face(&top, &[plane([10.0, 10.0, 17.0], [0.0, 0.0, 1.0], 400.0, 5)], &tol), None);
        // A pick with no maker known takes no second look either.
        assert_eq!(match_face(&FaceFingerprint { origin: -1, ..top }, &grown, &tol), None);
        let upright = |mid: [f64; 3], length: f64, origin: i64| EdgeFingerprint {
            kind: EdgeKind::Line,
            mid: DVec3::from(mid),
            dir: DVec3::Z,
            length,
            radius: None,
            faces: vec![EdgeSide { kind: FaceKind::Plane, dir: DVec3::new(0.0, -1.0, 0.0) }, EdgeSide { kind: FaceKind::Plane, dir: DVec3::new(-1.0, 0.0, 0.0) }],
            origin,
        };
        let side = upright([0.0, 0.0, 3.5], 7.0, 2);
        assert_eq!(match_edge(&side, &[upright([0.0, 0.0, 8.5], 17.0, 2), upright([20.0, 0.0, 8.5], 17.0, 2)], &tol), Some(0));
        assert_eq!(match_edge(&side, &[upright([0.0, 0.0, 8.5], 17.0, 5)], &tol), None);
    }

    #[test]
    fn describe_themselves_for_a_message() {
        let cyl = FaceFingerprint { kind: FaceKind::Cylinder, centroid: DVec3::new(10.0, 10.0, 5.0), dir: DVec3::Z, area: 94.2, radius: Some(3.0), center: None, origin: 2 };
        assert_eq!(describe_face(&cyl), "cylinder Ø6.0 at (10.0, 10.0, 5.0)");
        assert_eq!(describe_face(&plane([20.0, 10.0, 10.0], [0.0, 0.0, 1.0], 800.0, 1)), "plane 800 mm² at (20.0, 10.0, 10.0)");
        let circle = EdgeFingerprint { kind: EdgeKind::Circle, mid: DVec3::new(7.0, 10.0, 10.0), dir: DVec3::Z, length: 18.85, radius: Some(3.0), faces: vec![], origin: 2 };
        assert_eq!(describe_edge(&circle), "circle Ø6.0 at (7.0, 10.0, 10.0)");
        let l = EdgeFingerprint { kind: EdgeKind::Line, mid: DVec3::new(20.0, 0.0, 10.0), dir: DVec3::X, length: 40.0, radius: None, faces: vec![], origin: 2 };
        assert_eq!(describe_edge(&l), "line 40.0 mm at (20.0, 0.0, 10.0)");
    }

    #[test]
    fn tell_two_faces_meeting_tangent_on_one_surface_from_two_surfaces_running_into_each_other() {
        let round = |kind: FaceKind, center: [f64; 3], dir: [f64; 3], radius: Option<f64>| FaceFingerprint { kind, centroid: DVec3::from(center), dir: DVec3::from(dir), area: 50.0, radius, center: Some(DVec3::from(center)), origin: 1 };
        let z = [0.0, 0.0, 1.0];
        // A cylinder's halves, along its axis and with radii a millionth apart.
        assert!(on_one_surface(&round(FaceKind::Cylinder, [0.0, 0.0, 0.0], z, Some(33.92)), &round(FaceKind::Cylinder, [0.0, 0.0, 12.0], z, Some(33.919999432))));
        assert!(!on_one_surface(&round(FaceKind::Cylinder, [0.0; 3], z, Some(5.0)), &round(FaceKind::Cylinder, [0.0; 3], z, Some(5.1))));
        // A round of the same radius beside it, on an axis of its own.
        assert!(!on_one_surface(&round(FaceKind::Cylinder, [0.0; 3], z, Some(5.0)), &round(FaceKind::Cylinder, [10.0, 0.0, 0.0], z, Some(5.0))));
        assert!(!on_one_surface(&round(FaceKind::Cylinder, [0.0; 3], z, Some(5.0)), &round(FaceKind::Cylinder, [0.0; 3], [0.0, 1.0, 0.0], Some(5.0))));
        assert!(on_one_surface(&round(FaceKind::Cone, [0.0; 3], z, None), &round(FaceKind::Cone, [0.0, 0.0, 7.0], z, None)));
        assert!(!on_one_surface(&round(FaceKind::Cone, [0.0; 3], z, None), &round(FaceKind::Cylinder, [0.0; 3], z, Some(5.0))));
        assert!(on_one_surface(&plane([0.0; 3], z, 100.0, 1), &plane([30.0, 0.0, 0.0], z, 100.0, 1)));
        let sphere = |center: [f64; 3], r: f64| FaceFingerprint { kind: FaceKind::Sphere, centroid: DVec3::from(center), dir: DVec3::Z, area: 50.0, radius: Some(r), center: Some(DVec3::from(center)), origin: 1 };
        assert!(on_one_surface(&sphere([1.0, 2.0, 3.0], 4.0), &sphere([1.0, 2.0, 3.0], 4.0)));
        assert!(!on_one_surface(&sphere([1.0, 2.0, 3.0], 4.0), &sphere([1.0, 2.0, 5.0], 4.0)));
        // A torus's fingerprint does not say enough to tell: never one surface.
        let torus = FaceFingerprint { kind: FaceKind::Torus, centroid: DVec3::ZERO, dir: DVec3::Z, area: 50.0, radius: None, center: None, origin: 1 };
        assert!(!on_one_surface(&torus, &torus));
    }

    #[test]
    fn a_fingerprint_carried_through_an_alignment_moves_and_turns() {
        // A quarter turn about Z, then 5 mm along X.
        let m = DAffine3::from_translation(DVec3::new(5.0, 0.0, 0.0)) * DAffine3::from_rotation_z(std::f64::consts::FRAC_PI_2);
        let p = plane([10.0, 0.0, 3.0], [0.0, -1.0, 0.0], 12.0, 1);
        let moved = transform_fingerprint(&p, &m);
        assert!((moved.centroid - DVec3::new(5.0, 10.0, 3.0)).length() < 1e-9);
        // A plane's normal keeps its sign — a bottom stays a bottom.
        assert!((moved.dir - DVec3::X).length() < 1e-9);
        assert_eq!(moved.area, 12.0);
        assert_eq!(moved.origin, 1);
        let bore = FaceFingerprint { kind: FaceKind::Cylinder, centroid: DVec3::new(10.0, 0.0, 3.0), dir: DVec3::Y, area: 12.0, radius: Some(2.0), center: Some(DVec3::new(10.0, 0.0, 0.0)), origin: 1 };
        let turned = transform_fingerprint(&bore, &m);
        // The axis (0, 1, 0) turns to (−1, 0, 0), which canonical makes (1, 0, 0).
        assert!((turned.dir - DVec3::X).length() < 1e-9);
        assert!((turned.center.unwrap_or_default() - DVec3::new(5.0, 10.0, 0.0)).length() < 1e-9);

        let edge = EdgeFingerprint {
            kind: EdgeKind::Line,
            mid: DVec3::new(10.0, 0.0, 3.0),
            dir: DVec3::Y,
            length: 20.0,
            radius: None,
            faces: vec![EdgeSide { kind: FaceKind::Plane, dir: DVec3::new(0.0, -1.0, 0.0) }, EdgeSide { kind: FaceKind::Cylinder, dir: DVec3::Y }],
            origin: 2,
        };
        let moved = transform_edge_fingerprint(&edge, &m);
        assert!((moved.mid - DVec3::new(5.0, 10.0, 3.0)).length() < 1e-9);
        assert!((moved.dir - DVec3::X).length() < 1e-9);
        assert!((moved.faces[0].dir - DVec3::X).length() < 1e-9);
        assert!((moved.faces[1].dir - DVec3::X).length() < 1e-9);
        assert_eq!(moved.length, 20.0);
        assert_eq!(match_edge(&moved, &[edge.clone(), moved.clone()], &match_tolerance(100.0)), Some(1));
    }

    #[test]
    fn a_tangent_chain_runs_round_a_rounded_corner_and_stops_at_a_sharp_one() {
        let e = |hash: u32, a: [f64; 3], b: [f64; 3], ta: [f64; 3], tb: [f64; 3]| EdgeInfo {
            hash,
            fingerprint: line([0.0; 3], 1.0),
            faces: vec![],
            ends: [DVec3::from(a), DVec3::from(b)],
            end_tangents: [DVec3::from(ta), DVec3::from(tb)],
            polyline: vec![DVec3::from(a), DVec3::from(b)],
        };
        // A line, an arc tangent to it, a line tangent to the arc, then a sharp turn.
        let edges = [
            e(0, [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
            e(1, [10.0, 0.0, 0.0], [12.0, 2.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            e(2, [12.0, 2.0, 0.0], [12.0, 10.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]),
            e(3, [12.0, 10.0, 0.0], [2.0, 10.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]),
        ];
        let chain = tangent_chain(&edges, &[0], &HashSet::new(), CHAIN_ANGLE, CHAIN_JOIN_MM);
        assert_eq!(chain, vec![0, 1, 2]);
        let skipped = tangent_chain(&edges, &[0], &HashSet::from([1]), CHAIN_ANGLE, CHAIN_JOIN_MM);
        assert_eq!(skipped, vec![0]);
    }

    #[test]
    fn fingerprints_go_to_json_as_scanruler_writes_them() {
        let f = FaceFingerprint { kind: FaceKind::Cylinder, centroid: DVec3::new(1.0, 2.0, 3.0), dir: DVec3::Z, area: 4.0, radius: Some(2.0), center: None, origin: 3 };
        let text = serde_json::to_string(&f).unwrap_or_default();
        assert_eq!(text, r#"{"kind":"cylinder","centroid":[1.0,2.0,3.0],"dir":[0.0,0.0,1.0],"area":4.0,"radius":2.0,"origin":3}"#);
        let back: FaceFingerprint = serde_json::from_str(&text).unwrap_or(plane([0.0; 3], [0.0; 3], 0.0, 0));
        assert_eq!(back, f);
    }
}
