// SPDX-License-Identifier: AGPL-3.0-only
//! The Draft feature's build: faces of a body leaned about a neutral plane — a moulding's draft
//! put on after the fact. The neutral plane is where the faces keep their place (a flat face of
//! the body, the body lying behind it, or any plane), and a positive angle leans them in as they
//! go away from it into the body. More faces may lean at angles of their own (groups), after
//! the first ones, in the order given; a face in two groups leans by the first.
//!
//! The kernel ScanRuler builds on leans a face's tangent neighbours with it — a wall's round
//! and the wall beyond it go with the wall — so a face picked later that the draft reached
//! already is passed over (said on the row) where it leans by its own angle, and refused where
//! it is asked another. SolidCraft's draft leans only the faces it is handed, so the tangent
//! neighbours are found here and handed over with the face that reaches them, to the same end.
//! A draft that fails is diagnosed as ScanRuler diagnoses it: the faces tried one at a time,
//! with the largest angle each takes, or together; the faces meeting the picked ones at a
//! tangent edge and not picked named; a join a hair off tangent named.
//!
//! SolidCraft leans walls that stand along the pull (flat faces, and rounds about axes along
//! the pull) between faces square to it; a face leaning already cannot be drafted further here,
//! and fails by name.
//!
//! Ported from ScanRuler (`plugins/reverse/core/build.ts` draftNeutral, draftPicks and the
//! draft step; `core/kernelFaces.ts` draftFaces; `core/diagnose.ts` diagnoseDraft,
//! grazingEdges, tangentNeighbours; `core/kernelShell.ts` largestSize; `core/resolve.ts` the
//! draft step), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only, on the SolidCraft kernel's
//! `draft`.

use std::collections::HashMap;

use glam::DVec3;
use solidcraft_geom::Plane;
use solidcraft_kernel::{self as kernel, Body, FaceSurface};

use super::{describe_plane_ref, dv3, kerr, origins_after, resolve_plane, sv3, Builder};
use crate::model::{DraftFeature, Feature, MirrorPlane, Part};
use crate::pick_query::sharp_by_surfaces;
use crate::round::{number_text, round_to};
use crate::snap_face::same_face;
use crate::topo::{describe_face, FaceKind, Topology};

/// How many picks a diagnosis tries one at a time, at most — each try is a kernel operation on
/// the whole body.
const DIAGNOSE_PICKS: usize = 24;
/// How near a diagnosis finds the largest size that works: it halves until the size that works
/// and the one that does not are this close — in radians here, the angle being what is halved,
/// as ScanRuler's `largestSize` is handed it.
const SIZE_WITHIN: f64 = 0.1;
/// The least number of halvings a diagnosis makes.
const LEAST_HALVINGS: usize = 6;
/// Creases a draft cannot lean across, in degrees: a hair off tangent and no more — under it
/// the faces lean together, past it as a corner.
const GRAZING_DEG: [f64; 2] = [0.01, 0.5];

pub(crate) fn build(b: &mut Builder, part: &Part, _f: &Feature, x: &DraftFeature) -> Result<(), String> {
    let bad = |deg: f64| !(deg.abs() > 1e-9) || deg.abs() >= 89.0;
    if x.faces.is_empty() || x.groups.iter().any(|g| g.faces.is_empty()) {
        return Err("Pick the faces to draft.".into());
    }
    if bad(x.angle_deg) || x.groups.iter().any(|g| bad(g.angle_deg)) {
        return Err("The draft needs an angle, under 89°.".into());
    }
    let given = match &x.neutral {
        MirrorPlane::Plane(p) => Some(resolve_plane(p, part).ok_or_else(|| format!("The neutral plane of this draft is gone ({}).", describe_plane_ref(p, part)))?),
        MirrorPlane::Face { .. } => None,
    };
    let (id, state) = b.body_for(x.body)?;
    // Where the neutral plane lies and which way it pulls: a face of the body — the body lying
    // behind it, the faces leaning in as they go into it — or a plane given.
    let (origin, pull) = match (&x.neutral, given) {
        (_, Some(p)) => p,
        (MirrorPlane::Face { face }, None) => {
            let found = b.find_faces(id, std::slice::from_ref(face), &[]).map_err(|lost| format!("The neutral face of this draft is gone (was: {lost})."))?;
            let t = b.topology_of(id).unwrap_or_else(|| state.topology());
            let f = found.first().and_then(|i| t.faces.get(*i)).map(|f| f.fingerprint).filter(|f| f.kind == FaceKind::Plane).ok_or("The neutral face of this draft is not a plane.")?;
            (f.centroid, -f.dir)
        }
        (MirrorPlane::Plane(_), None) => return Err("The neutral plane of this draft is gone.".into()),
    };
    // The faces found on the body, each with the angle it leans by — the first group's, then
    // each further group's in order — and the pick each came from. A face in two groups (or a
    // half of one picked before it: the kernel keeps a whole bore as two) leans by the first.
    let t = b.topology_of(id).unwrap_or_else(|| state.topology());
    let mut groups: Vec<(&[_], &[_], f64)> = vec![(&x.faces, &x.faces_where, x.angle_deg.to_radians())];
    for g in &x.groups {
        groups.push((&g.faces, &[], g.angle_deg.to_radians()));
    }
    let mut hashes: Vec<usize> = Vec::new();
    let mut angles: Vec<f64> = Vec::new();
    for (faces, asked, angle) in groups {
        let found = b.find_faces(id, faces, asked).map_err(|lost| format!("A face this draft leans is gone (was: {lost})."))?;
        for h in found {
            if t.faces.get(h).is_none() || hashes.iter().any(|k| same_face(&t, *k, h)) {
                continue;
            }
            hashes.push(h);
            angles.push(angle);
        }
    }
    // What this kernel cannot lean at all is said as it is; what it refused is diagnosed.
    if let Some(why) = unsupported(&state.shape, &t, &hashes, pull) {
        return Err(format!("Drafting: {why}"));
    }
    let attempt = draft_faces(&state.shape, &t, &hashes, origin, pull, &angles).map_err(|e| format!("Drafting: {e}"));
    let failure = match &attempt {
        Ok((made, _)) if !holds(made, &state.shape) => Some("The draft left no closed solid.".to_string()),
        Ok(_) => None,
        Err(e) => Some(e.clone()),
    };
    if let Some(told) = failure {
        let Some((message, outline)) = diagnose(&state.shape, &t, &hashes, &angles, origin, pull) else { return Err(told) };
        if !outline.is_empty() {
            b.out.failed_where.insert(b.step, outline);
        }
        let said = told.split_once(": ").map_or(told.as_str(), |(_, rest)| rest).trim_end_matches('.');
        return Err(if !said.is_empty() && message.contains(said) { message } else { format!("{message} ({told})") });
    }
    let Ok((made, skipped)) = attempt else { return Err("The draft could not be made.".into()) };
    if !skipped.is_empty() {
        let named: Vec<String> = skipped.iter().filter_map(|i| hashes.get(*i)).filter_map(|h| t.faces.get(*h)).map(|f| describe_face(&f.fingerprint)).collect();
        let one = skipped.len() == 1;
        b.warn(format!(
            "{} picked already leaned by {} angle — drafted along with a tangent neighbour before {} — and {} passed over: {}.",
            if one { "A face".to_string() } else { format!("{} faces", skipped.len()) },
            if one { "its" } else { "their" },
            if one { "it" } else { "them" },
            if one { "was" } else { "were" },
            named.join("; ")
        ));
    }
    let origins = origins_after(&made, &[(&state.shape, &state.origins)], b.step);
    let ids = b.settle(id, made, origins, false)?;
    b.touched(ids);
    Ok(())
}

/// Whether a draft's result stands: a closed solid that keeps every face of the body — a face
/// leaned so far that it runs past its neighbours takes faces out, which a draft never means.
fn holds(made: &Body, was: &Body) -> bool {
    made.validity().is_empty() && made.face_count() >= was.face_count()
}

/// A face's unit normal at a point of it, off its surface (the sign is not kept), where the
/// surface is one of the named kinds.
fn normal_at(body: &Body, face: usize, p: DVec3) -> Option<DVec3> {
    match kernel::face_surface(body, face)? {
        FaceSurface::Plane { normal, .. } => Some(dv3(normal.into())),
        FaceSurface::Cylinder { origin, axis, .. } => {
            let (o, a) = (dv3(origin), dv3(axis.into()));
            let d = p - o;
            (d - a * d.dot(a)).try_normalize()
        }
        FaceSurface::Cone { apex, axis, half_angle } => {
            let (o, a) = (dv3(apex), dv3(axis.into()));
            let d = p - o;
            let radial = (d - a * d.dot(a)).try_normalize()?;
            (radial * half_angle.cos() - a * half_angle.sin()).try_normalize()
        }
        FaceSurface::Sphere { center, .. } => (p - dv3(center)).try_normalize(),
        FaceSurface::Torus { center, axis, major, .. } => {
            let (c, a) = (dv3(center), dv3(axis.into()));
            let d = p - c;
            let radial = (d - a * d.dot(a)).try_normalize()?;
            (p - (c + radial * major)).try_normalize()
        }
    }
}

/// The edges between two faces that meet smoothly — tangent, or two faces of one surface — by
/// the faces either side, read off the faces' own surfaces at the edge's middle and a quarter of
/// the way from each end.
fn smooth_joins(body: &Body, t: &Topology) -> Vec<(usize, usize)> {
    let normals = |face: u32, points: &[DVec3]| points.iter().map(|p| normal_at(body, face as usize, *p)).collect::<Vec<_>>();
    let mut out = Vec::new();
    for e in &t.edges {
        let mut sides: Vec<u32> = Vec::new();
        for f in &e.faces {
            if !sides.contains(f) {
                sides.push(*f);
            }
        }
        if sides.len() != 2 || sharp_by_surfaces(e, &normals) {
            continue;
        }
        out.push((sides[0] as usize, sides[1] as usize));
    }
    out
}

/// How the picked faces run on into others across smooth joins, as ScanRuler's kernel leans
/// them: each face reached, by the pick (its index) it was reached from first.
fn reach(picked: &[usize], joins: &[(usize, usize)]) -> (HashMap<usize, usize>, Vec<(usize, usize)>) {
    let mut from: HashMap<usize, usize> = HashMap::new();
    let mut clashes = Vec::new();
    for (i, p) in picked.iter().enumerate() {
        if let Some(j) = from.get(p) {
            if *j != i {
                clashes.push((*j, i));
            }
            continue;
        }
        from.insert(*p, i);
        let mut queue = std::collections::VecDeque::from([*p]);
        while let Some(h) = queue.pop_front() {
            for (a, b) in joins {
                let other = if *a == h {
                    *b
                } else if *b == h {
                    *a
                } else {
                    continue;
                };
                if from.contains_key(&other) {
                    continue;
                }
                from.insert(other, i);
                queue.push_back(other);
            }
        }
    }
    (from, clashes)
}

/// An angle in radians, as a sentence says it in degrees: two decimals at most.
fn deg_of(rad: f64) -> String {
    format!("{}°", number_text(round_to(rad.abs().to_degrees(), 2)))
}

/// Faces of a solid leaned about a neutral plane (ScanRuler's `draftFaces`): each face named
/// (by index) is turned by its angle about the line where it meets the plane through `origin`
/// square to `pull`, leaning in as it goes along `pull` for a positive angle — a mould's draft.
/// A face leans its tangent neighbours with it; a pick reached so already is passed over (its
/// index in the second list) where it leans by its own angle, and refused where it is asked
/// another.
pub(crate) fn draft_faces(body: &Body, t: &Topology, faces: &[usize], origin: DVec3, pull: DVec3, angles: &[f64]) -> Result<(Body, Vec<usize>), String> {
    let dir = pull.try_normalize().ok_or("the pull direction has no length.")?;
    if angles.len() != faces.len() || angles.iter().any(|a| !(a.abs() > 1e-9)) {
        return Err("the draft has no angle.".into());
    }
    if faces.is_empty() {
        return Err("no face given.".into());
    }
    for h in faces {
        let f = t.faces.get(*h).ok_or("the face is not on the body.")?;
        if f.fingerprint.kind == FaceKind::Plane && f.fingerprint.dir.dot(dir).abs() > 0.999 {
            return Err("a face square to the pull direction cannot be drafted — it is the kind of face the neutral plane is.".into());
        }
    }
    let joins = smooth_joins(body, t);
    let (from, clashes) = reach(faces, &joins);
    let mut skipped = Vec::new();
    for (j, i) in clashes {
        let tol = 1e-3f64.min(angles[i].abs() / 3.0);
        if (angles[j].abs() - angles[i].abs()).abs() <= tol {
            skipped.push(i);
            continue;
        }
        return Err(format!("pick #{} was leaned already, by {}, with a tangent neighbour drafted before it — faces that meet tangent lean together, at one angle; give them the same angle ({} is asked).", i + 1, deg_of(angles[j]), deg_of(angles[i])));
    }
    // Every face that leans, with the angle of the pick that reached it.
    let mut leaning: Vec<(usize, f64)> = from.iter().map(|(face, pick)| (*face, angles[*pick])).collect();
    leaning.sort_by_key(|(face, _)| *face);
    let neutral = Plane::from_normal(sv3(origin), sv3(dir)).ok_or("the pull direction has no length.")?;
    // One kernel call an angle, in the order the angles were first asked; the faces of a later
    // call are found again by a point on them, which a draft of others does not move.
    let mut order: Vec<f64> = Vec::new();
    for (_, a) in &leaning {
        if !order.iter().any(|x| x == a) {
            order.push(*a);
        }
    }
    order.sort_by_key(|a| angles.iter().position(|x| x == a).unwrap_or(usize::MAX));
    let mut shape = body.clone();
    for a in order {
        let points: Vec<_> = leaning.iter().filter(|(_, x)| *x == a).filter_map(|(face, _)| t.faces.get(*face)).map(|f| sv3(f.point)).collect();
        shape = kernel::draft(&shape, &points, &neutral, sv3(dir), a).map_err(kerr)?;
    }
    Ok((shape, skipped))
}

/// Why SolidCraft cannot lean these faces at all, where it cannot: it leans walls that stand
/// along the pull — flat faces, and rounds about axes along it — and nothing else, so a face
/// that leans already, or curves another way, among the faces picked or the faces they run on
/// into, is said by name. The kernel ScanRuler builds on leans such faces too.
fn unsupported(body: &Body, t: &Topology, faces: &[usize], pull: DVec3) -> Option<String> {
    let dir = pull.try_normalize()?;
    let joins = smooth_joins(body, t);
    let (from, _) = reach(faces, &joins);
    let mut leaning: Vec<usize> = from.keys().copied().collect();
    leaning.sort_unstable();
    for face in leaning {
        let fp = t.faces.get(face)?.fingerprint;
        // A face square to the pull is the neutral plane's kind: the draft says so itself.
        if fp.kind == FaceKind::Plane && fp.dir.dot(dir).abs() > 0.999 {
            return None;
        }
        let wall = match fp.kind {
            FaceKind::Plane => fp.dir.dot(dir).abs() < 1e-6,
            FaceKind::Cylinder => fp.dir.dot(dir).abs() > 1.0 - 1e-9,
            _ => false,
        };
        if !wall {
            return Some(format!("{} does not stand along the pull: leaning a face that leans already, or curves another way, is not built yet — only flat walls standing square to the neutral plane, and rounds about axes along the pull, are leaned here.", capitalised(&describe_face(&fp))));
        }
    }
    None
}

/// What a diagnosis names a face by: its index in the body's faces, its kind and a round's
/// radius.
fn face_name(t: &Topology, face: usize) -> String {
    match t.faces.get(face) {
        Some(f) => {
            let r = f.fingerprint.radius.map_or(String::new(), |r| format!(" of {} mm radius", number_text(round_to(r, 2))));
            format!("face {face} ({}{r})", f.fingerprint.kind.name())
        }
        None => format!("face {face}"),
    }
}

fn capitalised(s: &str) -> String {
    let mut c = s.chars();
    c.next().map(|f| f.to_uppercase().collect::<String>() + c.as_str()).unwrap_or_default()
}

/// The largest size below `size` at which `works` holds, by halving to within `SIZE_WITHIN`
/// (and at least six times) — zero when it holds at none tried (ScanRuler's `largestSize`).
fn largest_size(size: f64, works: impl Fn(f64) -> bool) -> f64 {
    let halvings = LEAST_HALVINGS.max((size / SIZE_WITHIN).log2().ceil().max(0.0) as usize);
    let (mut lo, mut hi) = (0.0, size);
    for _ in 0..halvings {
        let mid = (lo + hi) / 2.0;
        if works(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Why a draft the kernel refused could not be made (ScanRuler's `diagnoseDraft`): the faces
/// tried one at a time at the angle each was asked — those that do not lean alone, with the
/// largest angle each takes — or, each leaning alone, the faces together. A face meeting one
/// picked at a tangent edge, and not picked itself, is named: the draft leans it with the face
/// it meets. A sentence, and the outlines to draw red on the part; `None` where there is
/// nothing to add to the kernel's own words.
fn diagnose(body: &Body, t: &Topology, hashes: &[usize], angles: &[f64], origin: DVec3, pull: DVec3) -> Option<(String, Vec<Vec<DVec3>>)> {
    let m = hashes.len();
    if m == 0 {
        return None;
    }
    let angle_of: HashMap<usize, f64> = hashes.iter().copied().zip(angles.iter().copied()).collect();
    let attempt = |faces: &[usize], scale: f64| -> bool {
        let a: Vec<f64> = faces.iter().map(|h| scale * angle_of.get(h).copied().unwrap_or(0.0)).collect();
        matches!(draft_faces(body, t, faces, origin, pull, &a), Ok((made, _)) if holds(&made, body))
    };
    let outlines = |faces: &[usize]| -> Vec<Vec<DVec3>> { t.edges.iter().filter(|e| e.faces.iter().any(|f| faces.contains(&(*f as usize))) && e.polyline.len() >= 2).map(|e| e.polyline.clone()).collect() };
    let instead = "Or draw the slope into the profile instead — a drafted extrude, or a sketch that draws the sloped side — which leans the face as it is made.";
    let one = angles.iter().all(|a| *a == angles[0]);
    let at = if one { format!(" at {}", deg_of(angles[0])) } else { " at its angle".to_string() };
    let joins = smooth_joins(body, t);
    let (from, clashes) = reach(hashes, &joins);
    // Two picks leaned together, asked at two angles: that is the failure, whatever else the
    // body does.
    if let Some((j, i)) = clashes.iter().find(|(j, i)| (angles[*j] - angles[*i]).abs() > 1e-9) {
        let (a, b) = (hashes[*j], hashes[*i]);
        return Some((
            format!("{} and {} meet at a tangent edge, or through faces that do, and are asked to lean at {} and {}: the kernel leans faces that meet tangent together, at one angle — give them the same angle, or draft before the round between them.", capitalised(&face_name(t, a)), face_name(t, b), deg_of(angles[*j]), deg_of(angles[*i])),
            outlines(&[a, b]),
        ));
    }
    // The faces reached that are not picked, in the order they were reached.
    let mut beside: Vec<usize> = Vec::new();
    {
        let mut order: Vec<(usize, usize)> = from.iter().filter(|(f, _)| !hashes.contains(f)).map(|(f, p)| (*p, *f)).collect();
        order.sort();
        // Breadth first from each pick: nearer faces first.
        for (p, _) in &order {
            let mut queue = std::collections::VecDeque::from([hashes[*p]]);
            let mut seen = vec![hashes[*p]];
            while let Some(h) = queue.pop_front() {
                for (a, b) in &joins {
                    let other = if *a == h {
                        *b
                    } else if *b == h {
                        *a
                    } else {
                        continue;
                    };
                    if seen.contains(&other) || from.get(&other) != Some(p) {
                        continue;
                    }
                    seen.push(other);
                    if !hashes.contains(&other) && !beside.contains(&other) {
                        beside.push(other);
                    }
                    queue.push_back(other);
                }
            }
        }
    }
    let them = |one: &str, more: &str| if beside.len() == 1 { one.to_string() } else { more.to_string() };
    let beside_said = if beside.is_empty() {
        String::new()
    } else {
        let names: Vec<String> = beside.iter().take(4).map(|h| face_name(t, *h)).collect();
        let more = if beside.len() > 4 { format!(" and {} more", beside.len() - 4) } else { String::new() };
        format!(
            " {} those picked at a tangent edge and {} not picked — {}{}: the kernel leans {} with the face {}, so the draft runs on round {}; pick {} too, at the same angle, or draft before the round that joins {}.",
            them("A face meets", &format!("{} faces meet", beside.len())),
            them("is", "are"),
            names.join(", "),
            more,
            them("it", "them"),
            them("it meets", "they meet"),
            them("it", "them"),
            them("it", "them"),
            them("it", "them")
        )
    };
    // A picked face meeting another a hair off tangent: leaned, the two meet at a grazing angle
    // the kernel cannot join — named before any probing, which would only find a small angle
    // and blame the room.
    let creased = grazing_edges(body, t, hashes);
    if !creased.is_empty() {
        let named: Vec<String> = creased
            .iter()
            .take(3)
            .map(|(e, a, b, deg)| {
                let mid = t.edges.get(*e).map(|x| x.fingerprint.mid).unwrap_or_default();
                format!("{} and {} meet {}° off tangent at ({}, {}, {})", face_name(t, *a), face_name(t, *b), number_text(to_precision(*deg, 2)), number_text(round_to(mid.x, 2)), number_text(round_to(mid.y, 2)), number_text(round_to(mid.z, 2)))
            })
            .collect();
        let more = if creased.len() > named.len() { format!(" (and {} more)", creased.len() - named.len()) } else { String::new() };
        let single = creased.len() == 1;
        return Some((
            format!(
                "{}{more}: leaned, {} at a grazing angle the kernel cannot join — make {} of the profile tangent (reverse.sketch.edit rounds, a tangent relation) or a real corner (≥ 0.2°), or draw the slope into the profile with a drafted extrude, which makes such joins tangent itself.{beside_said}",
                capitalised(&named.join("; ")),
                if single { "they meet" } else { "the faces there meet" },
                if single { "that join" } else { "those joins" }
            ),
            creased.iter().filter_map(|(e, ..)| t.edges.get(*e)).map(|e| e.polyline.clone()).collect(),
        ));
    }
    let probed: Vec<usize> = hashes.iter().copied().take(DIAGNOSE_PICKS).collect();
    let failing: Vec<usize> = probed.iter().copied().filter(|h| !attempt(&[*h], 1.0)).collect();
    if !failing.is_empty() {
        let n = failing.len();
        let up_to = failing
            .iter()
            .map(|h| {
                let a = angle_of.get(h).copied().unwrap_or(0.0).abs();
                largest_size(a, |x| attempt(&[*h], x / a))
            })
            .fold(f64::INFINITY, f64::min);
        let which = if m == 1 {
            "The face does not lean".to_string()
        } else if n == m {
            format!("None of the {m} faces leans")
        } else {
            format!("{n} of the {m} faces do not lean")
        };
        let shown = if n < m { " — outlined red on the part; the others do" } else { " — outlined red on the part" };
        let can = if up_to > 0.0 {
            format!(
                "{} up to {} — past that {} out of room against the faces beside {}: a round or a sliver there (draft before rounding, or take the sliver out upstream), or the body's own width.",
                if n == 1 { "It leans" } else { "They lean" },
                deg_of(up_to),
                if n == 1 { "it runs" } else { "they run" },
                if n == 1 { "it" } else { "them" }
            )
        } else {
            format!(
                "{} not lean at any angle about this neutral plane — {} edges with the neighbouring faces cannot follow; draft the neighbours with {}, or leave {} out.",
                if n == 1 { "It does" } else { "They do" },
                if n == 1 { "its" } else { "their" },
                if n == 1 { "it" } else { "them" },
                if n == 1 { "it" } else { "them" }
            )
        };
        let mut lines = outlines(&failing);
        lines.extend(outlines(&beside));
        return Some((format!("{which}{at}{shown}. {can}{beside_said} {instead}"), lines));
    }
    if m < 2 || probed.len() < m {
        return (!beside.is_empty()).then(|| (format!("The draft fails{at}.{beside_said} {instead}"), outlines(&beside)));
    }
    let together = largest_size(1.0, |f| attempt(hashes, f));
    let can = if together > 0.0 {
        if one {
            format!("Together they lean up to {}.", deg_of(together * angles[0]))
        } else {
            format!("Together they lean up to {} % of the angles asked.", (together * 100.0).floor())
        }
    } else {
        "Draft them in two features, one after the other.".to_string()
    };
    let mut lines = outlines(hashes);
    lines.extend(outlines(&beside));
    Some((format!("Each of the {m} faces leans alone{at}; together they do not — they run into each other, or meet at an edge the kernel cannot lean both ways. {can}{beside_said} {instead}"), lines))
}

/// A number to `p` significant digits, as JavaScript's `Number(x.toPrecision(p))` reads.
fn to_precision(x: f64, p: i32) -> f64 {
    if x == 0.0 || !x.is_finite() {
        return x;
    }
    let mag = x.abs().log10().floor() as i32;
    round_to(x, p - 1 - mag)
}

/// The edges of picked faces where the two faces meet a hair off tangent (`GRAZING_DEG`), by
/// their normals at the edge's middle — the most nearly tangent first: (edge, face, face,
/// degrees).
fn grazing_edges(body: &Body, t: &Topology, picked: &[usize]) -> Vec<(usize, usize, usize, f64)> {
    let mut out = Vec::new();
    for (i, e) in t.edges.iter().enumerate() {
        let mut sides: Vec<usize> = Vec::new();
        for f in &e.faces {
            if !sides.contains(&(*f as usize)) {
                sides.push(*f as usize);
            }
        }
        if sides.len() != 2 || !sides.iter().any(|f| picked.contains(f)) {
            continue;
        }
        let p = e.fingerprint.mid;
        let (Some(a), Some(b)) = (normal_at(body, sides[0], p), normal_at(body, sides[1], p)) else { continue };
        let deg = a.cross(b).length().atan2(a.dot(b).abs()).to_degrees();
        if deg >= GRAZING_DEG[0] && deg <= GRAZING_DEG[1] {
            out.push((i, sides[0], sides[1], deg));
        }
    }
    out.sort_by(|x, y| x.3.total_cmp(&y.3));
    out
}

#[cfg(test)]
mod tests {
    //! ScanRuler's `reDraftGroups.test.ts`, `reDraftDiagnose.test.ts` and the kernel part of
    //! `reSnapDraft.test.ts`, on the SolidCraft kernel.
    use glam::DVec2;

    use super::*;
    use crate::build::{build, default_regions, volume};
    use crate::frame::WorldAxis;
    use crate::model::{BodyId, DraftGroup, ExtrudeExtent, ExtrudeFeature, FeatureKind, FilletFeature, SketchFeature, BodyOp, Exprs};
    use crate::regions::find_regions;
    use crate::sketch::Sketch;
    use crate::topo::FaceFingerprint;

    fn prints(t: &Topology) -> Vec<FaceFingerprint> {
        t.faces.iter().map(|f| f.fingerprint).collect()
    }

    const BOX: f64 = 40.0 * 20.0 * 10.0;

    /// What a wall 10 high and 20 long leaned in by `deg` about its foot takes off the box.
    fn wedge(deg: f64) -> f64 {
        0.5 * 10.0 * (10.0 * deg.to_radians().tan()) * 20.0
    }

    /// The 40 × 20 × 10 block: sketch 1, extrude 2.
    fn block() -> Part {
        let mut part = Part::default();
        let mut s = Sketch::default();
        s.add_rectangle(DVec2::ZERO, DVec2::new(40.0, 20.0));
        let regions = default_regions(&find_regions(&s));
        let sk = part.add(FeatureKind::Sketch(SketchFeature::on_world(WorldAxis::Z, 0.0, s)));
        part.add(FeatureKind::Extrude(ExtrudeFeature::new(sk, regions, ExtrudeExtent::Distance { from: 0.0, to: 10.0 }, BodyOp::New)));
        part
    }

    fn faces_of(part: &Part) -> Vec<FaceFingerprint> {
        let built = build(part);
        assert!(built.errors.is_empty(), "{:?}", built.errors);
        prints(&built.bodies[0].topology)
    }

    fn draft(neutral: FaceFingerprint, faces: Vec<FaceFingerprint>, angle_deg: f64, groups: Vec<DraftGroup>) -> FeatureKind {
        FeatureKind::Draft(DraftFeature { body: BodyId::new(2), neutral: MirrorPlane::Face { face: neutral }, faces, faces_where: Vec::new(), angle_deg, groups, exprs: Exprs::new() })
    }

    fn find(faces: &[FaceFingerprint], pick: impl Fn(&FaceFingerprint) -> bool) -> FaceFingerprint {
        faces.iter().copied().find(|f| pick(f)).unwrap_or_else(|| panic!("no such face"))
    }

    #[test]
    fn leans_two_walls_at_two_angles_in_one_feature() {
        let mut part = block();
        let faces = faces_of(&part);
        let bottom = find(&faces, |f| f.dir.z < -0.9);
        let right = find(&faces, |f| f.dir.x > 0.9);
        let left = find(&faces, |f| f.dir.x < -0.9);
        let id = part.add(draft(bottom, vec![right], 1.8, vec![DraftGroup { faces: vec![left], angle_deg: 2.7 }]));
        let out = build(&part);
        assert!(out.errors.is_empty(), "{:?}", out.errors);
        let v = volume(&out.bodies[0].body).unwrap_or(f64::NAN);
        assert!((v - (BOX - wedge(1.8) - wedge(2.7))).abs() < 1e-3, "{v}");
        // Each wall leans by its own angle: its normal rises by the sine.
        let after = prints(&out.bodies[0].topology);
        assert!((find(&after, |f| f.dir.x > 0.9).dir.z - 1.8f64.to_radians().sin()).abs() < 1e-6);
        assert!((find(&after, |f| f.dir.x < -0.9).dir.z - 2.7f64.to_radians().sin()).abs() < 1e-6);
        // The drafted walls keep the extrude as their maker.
        assert_eq!(find(&after, |f| f.dir.x > 0.9).origin, 2);
        // A feature saved before groups builds as it did.
        if let Some(f) = part.feature_mut(id) {
            f.kind = draft(bottom, vec![right, left], 1.8, Vec::new());
        }
        let old = build(&part);
        let v = volume(&old.bodies[0].body).unwrap_or(f64::NAN);
        assert!((v - (BOX - 2.0 * wedge(1.8))).abs() < 1e-3, "{v}");
    }

    /// The block with one upright corner rounded R5: a cylinder tangent to the two walls either
    /// side. The part, its faces, and the two walls beside the round.
    fn rounded() -> (Part, Vec<FaceFingerprint>, [FaceFingerprint; 2]) {
        let mut part = block();
        let built = build(&part);
        let edge = built.bodies[0].topology.edges.iter().find(|e| e.fingerprint.dir.z.abs() > 0.9 && (e.fingerprint.mid.x - 40.0).abs() < 1e-6 && (e.fingerprint.mid.y - 20.0).abs() < 1e-6).map(|e| e.fingerprint.clone()).unwrap_or_else(|| panic!("the corner"));
        part.add(FeatureKind::Fillet(FilletFeature { body: BodyId::new(2), edges: vec![edge], edges_where: Vec::new(), radius: 5.0, chain: false, skip_failing: false, variable: Vec::new(), exprs: Exprs::new() }));
        let faces = faces_of(&part);
        let walls: Vec<FaceFingerprint> = faces.iter().copied().filter(|f| f.kind == FaceKind::Plane && f.dir.z.abs() < 0.1).collect();
        let beside: Vec<FaceFingerprint> = walls.iter().copied().filter(|f| f.area < (if f.dir.x.abs() > 0.5 { 20.0 } else { 40.0 }) * 10.0 - 1.0).collect();
        assert_eq!(beside.len(), 2);
        (part, faces, [beside[0], beside[1]])
    }

    fn tried(part: &Part, kind: FeatureKind) -> (Result<(), String>, Vec<String>) {
        let mut p = part.clone();
        let id = p.add(kind);
        let out = build(&p);
        (out.errors.get(&id).map_or(Ok(()), |e| Err(e.clone())), out.warnings.get(&id).cloned().unwrap_or_default())
    }

    #[test]
    fn passes_over_a_face_leaned_with_its_tangent_neighbour_refuses_it_at_another_angle_and_names_a_neighbour_left_out() {
        let (part, faces, beside) = rounded();
        let bottom = find(&faces, |f| f.dir.z < -0.9);
        let round = find(&faces, |f| f.kind == FaceKind::Cylinder);
        let walls: Vec<FaceFingerprint> = faces.iter().copied().filter(|f| f.kind == FaceKind::Plane && f.dir.z.abs() < 0.1).collect();
        // Every wall and the round at 5°: the first wall leaned takes the round and the wall
        // beyond along — those are passed over.
        let mut all = vec![beside[0], round, beside[1]];
        all.extend(walls.iter().copied().filter(|w| !beside.contains(w)));
        let (r, warnings) = tried(&part, draft(bottom, all, 5.0, Vec::new()));
        assert!(r.is_ok(), "{r:?}");
        let w = warnings.join(" ");
        assert!(w.contains("2 faces picked already leaned by their angle — drafted along with a tangent neighbour before them — and were passed over: ") && w.contains("cylinder Ø10"), "{w}");
        // The round in a group of its own at another angle: refused, saying so.
        let (r, _) = tried(&part, draft(bottom, vec![beside[0]], 5.0, vec![DraftGroup { faces: vec![round], angle_deg: 3.0 }]));
        let e = r.unwrap_err();
        assert!(e.starts_with("Face ") && e.contains(" (plane) and face ") && e.contains(" (cylinder of 5 mm radius) meet at a tangent edge, or through faces that do, and are asked to lean at 5° and 3°"), "{e}");
        // A wall alone leaned past what the box takes: the diagnosis names the round it meets
        // at a tangent edge, which the draft leans with it.
        let (r, _) = tried(&part, draft(bottom, vec![beside[0]], 80.0, Vec::new()));
        let e = r.unwrap_err();
        assert!(e.contains("meet those picked at a tangent edge and are not picked — face ") && e.contains(" (cylinder of 5 mm radius)"), "{e}");
    }

    #[test]
    fn names_the_wall_that_will_not_lean_and_how_far_it_would_or_the_walls_that_will_not_lean_together() {
        let part = block();
        let faces = faces_of(&part);
        let bottom = find(&faces, |f| f.dir.z < -0.9);
        let right = find(&faces, |f| f.dir.x > 0.9);
        let left = find(&faces, |f| f.dir.x < -0.9);
        // 10 high, leaned 80°, the wall would run 57 mm in across a box 40 wide.
        let (r, _) = tried(&part, draft(bottom, vec![right], 80.0, Vec::new()));
        let e = r.unwrap_err();
        assert!(e.contains("The face does not lean at 80°"), "{e}");
        let up_to: f64 = e.split("leans up to ").nth(1).and_then(|s| s.split('°').next()).and_then(|s| s.parse().ok()).unwrap_or(0.0);
        assert!(up_to > 70.0 && up_to < 80.0, "{e}");
        assert!(e.contains("draw the slope into the profile"));
        // At 70° each wall runs 27 mm in: alone it fits, the two together not.
        let (r, _) = tried(&part, draft(bottom, vec![right, left], 70.0, Vec::new()));
        let e = r.unwrap_err();
        assert!(e.contains("Each of the 2 faces leans alone at 70°; together they do not"), "{e}");
        assert!(e.contains("Together they lean up to"), "{e}");
    }

    #[test]
    fn refuses_what_has_no_angle_no_face_or_a_lost_neutral_face() {
        let part = block();
        let faces = faces_of(&part);
        let bottom = find(&faces, |f| f.dir.z < -0.9);
        let right = find(&faces, |f| f.dir.x > 0.9);
        assert_eq!(tried(&part, draft(bottom, Vec::new(), 2.0, Vec::new())).0, Err("Pick the faces to draft.".to_string()));
        assert_eq!(tried(&part, draft(bottom, vec![right], 89.0, Vec::new())).0, Err("The draft needs an angle, under 89°.".to_string()));
        assert_eq!(tried(&part, draft(bottom, vec![right], 2.0, vec![DraftGroup { faces: vec![right], angle_deg: 0.0 }])).0, Err("The draft needs an angle, under 89°.".to_string()));
        let gone = FaceFingerprint { centroid: DVec3::new(500.0, 0.0, 0.0), ..bottom };
        assert!(tried(&part, draft(gone, vec![right], 2.0, Vec::new())).0.unwrap_err().starts_with("The neutral face of this draft is gone (was: "));
        // A face square to the pull is refused, and the refusal diagnosed as any other is.
        let e = tried(&part, draft(right, vec![right], 2.0, Vec::new())).0.unwrap_err();
        assert!(e.starts_with("The face does not lean at 2°") && e.ends_with("(Drafting: a face square to the pull direction cannot be drafted — it is the kind of face the neutral plane is.)"), "{e}");
    }

    #[test]
    fn lays_a_snapped_face_on_the_scans_leaning_plane_after_the_move_keeping_the_volume_about_the_hinge() {
        // The scan's top: 2 mm above the block's, dipping 3° as it goes along +X. The move goes
        // into the extrude's distance; the draft comes after it.
        use crate::fit3d::types::{FitBase, FitData, PlaneFit};
        use crate::snap_edit::{plan_snap, SnapPlan};
        use crate::snap_face::{snap_draft_feature, snap_reading};
        let t = 3f64.to_radians();
        let normal = DVec3::new(t.sin(), 0.0, t.cos());
        let scan_top = FitData::Plane(PlaneFit { base: FitBase { sigma: 0.02, used_points: 500, region_size: 500, ..FitBase::default() }, center: DVec3::new(20.0, 10.0, 12.0), normal, basis_u: DVec3::new(t.cos(), 0.0, -t.sin()), basis_v: DVec3::Y, extent_u: 15.0, extent_v: 8.0 });
        let part = block();
        let top = find(&faces_of(&part), |f| f.dir.z > 0.9);
        let reading = snap_reading(&top, &scan_top).unwrap_or_else(|e| panic!("{e}"));
        let offset = crate::round::round3(reading.offset);
        let SnapPlan::Extent { features, .. } = plan_snap(&part.features, &top, offset, &[], crate::display_units::DisplayUnits::Mm) else { panic!("an extent edit") };
        let lean = snap_draft_feature(BodyId::new(2), &top, offset, &reading.draft.unwrap_or_else(|| panic!("a lean"))).unwrap_or_else(|| panic!("a draft"));
        assert!((lean.angle_deg - 3.0).abs() < 0.005);
        let mut next = Part { features, next_id: 3, ..Part::default() };
        let id = next.add(FeatureKind::Draft(lean));
        assert_eq!(id, 3);
        let out = build(&next);
        assert!(out.errors.is_empty(), "{:?}", out.errors);
        let leaned = find(&prints(&out.bodies[0].topology), |f| f.kind == FaceKind::Plane && f.dir.z > 0.9);
        // Its normal is the scan's, and its plane goes through the scan's patch.
        assert!(leaned.dir.dot(normal) > 0.01f64.to_radians().cos(), "{:?}", leaned.dir);
        assert!((leaned.centroid - DVec3::new(20.0, 10.0, 12.0)).dot(normal).abs() < 1e-3);
        // Hinged through the middle, what the +X half loses the −X half gains.
        let v = volume(&out.bodies[0].body).unwrap_or(f64::NAN);
        assert!((v - 40.0 * 20.0 * 12.0).abs() < 1e-2, "{v}");
    }

    #[test]
    fn leans_two_walls_in_line_either_side_of_a_notch_each() {
        // The block with a notch 10 wide cut 5 into its +Y side: that side is two walls on one
        // plane, apart. Both picked, both lean.
        let mut part = block();
        let mut s = Sketch::default();
        s.add_rectangle(DVec2::new(15.0, 15.0), DVec2::new(25.0, 25.0));
        let regions = default_regions(&find_regions(&s));
        let sk = part.add(FeatureKind::Sketch(SketchFeature::on_world(WorldAxis::Z, 0.0, s)));
        part.add(FeatureKind::Extrude(ExtrudeFeature::new(sk, regions, ExtrudeExtent::Distance { from: -1.0, to: 11.0 }, BodyOp::Cut)));
        let faces = faces_of(&part);
        let bottom = find(&faces, |f| f.dir.z < -0.9);
        let walls: Vec<FaceFingerprint> = faces.iter().copied().filter(|f| f.kind == FaceKind::Plane && f.dir.y > 0.9 && (f.centroid.y - 20.0).abs() < 1e-6).collect();
        assert_eq!(walls.len(), 2);
        let (r, _) = tried(&part, draft(bottom, walls, 3.0, Vec::new()));
        assert!(r.is_ok(), "{r:?}");
        let mut p = part.clone();
        p.add(draft(bottom, faces.iter().copied().filter(|f| f.kind == FaceKind::Plane && f.dir.y > 0.9 && (f.centroid.y - 20.0).abs() < 1e-6).collect(), 3.0, Vec::new()));
        let out = build(&p);
        let v = volume(&out.bodies[0].body).unwrap_or(f64::NAN);
        let wedge15 = 0.5 * 10.0 * (10.0 * 3f64.to_radians().tan()) * 15.0;
        assert!((v - (BOX - 500.0 - 2.0 * wedge15)).abs() < 1e-3, "{v}");
    }

    #[test]
    fn says_by_name_that_a_face_leaning_already_is_not_drafted_here() {
        let mut part = block();
        let faces = faces_of(&part);
        let bottom = find(&faces, |f| f.dir.z < -0.9);
        let right = find(&faces, |f| f.dir.x > 0.9);
        part.add(draft(bottom, vec![right], 5.0, Vec::new()));
        let leaned = find(&faces_of(&part), |f| f.dir.x > 0.9);
        let (r, _) = tried(&part, draft(bottom, vec![leaned], 3.0, Vec::new()));
        let e = r.unwrap_err();
        assert!(e.contains("does not stand along the pull: leaning a face that leans already, or curves another way, is not built yet"), "{e}");
    }
}
