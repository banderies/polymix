// SPDX-License-Identifier: AGPL-3.0-only
//! The Shell feature's build: a body hollowed to a wall, the picked faces taken away to open it
//! — found again by fingerprint on the body as it stands — or, with none picked, left closed and
//! hollow inside. Ported from ScanRuler (`plugins/reverse/core/resolve.ts` the shell step,
//! `build.ts` applyToBody, `kernel.ts` shellTracked, `kernelShell.ts`), © Stefan Hermann (CNC
//! Kitchen), AGPL-3.0-only, on the SolidCraft kernel.
//!
//! Three ways are tried in turn, each saying why it failed, and the failure answers the reason
//! that says most: the kernel's own shell (a closed hollow body is the body less its own inward
//! offset); the wall made on the body with its rounds no larger than the wall taken off, and cut
//! back to the body's outside — rounded outside as modelled, sharp inside where the round ran
//! out; and the long way, by offset and cut — the body less its inward offset and the walls over
//! the opened faces. A shell built another way than the first says so, and why the first gave
//! up, on the feature's row.
//!
//! The body's inward offset is SolidCraft's: every face moved in by the wall, its neighbours
//! meeting it along new edges, a face moved to nothing gone — a body of planes, cylinders,
//! cones, spheres and tori. Where two faces meet at an inside corner the offset meets them in a
//! sharp corner where Open CASCADE's runs a round of the wall's radius, so the wall there is a
//! little thicker than the wall. A wall under a face that is not flat is the face copied out as
//! a sheet and thickened into the body.

use std::collections::HashSet;

use glam::DVec3;
use solidcraft_kernel::{self as kernel, BoolOp, Body, FaceSurface, WholeFace};

use super::fillet::{closed_solid, same_surface};
use super::{dv3, kerr, origins_after, read_topology, sv3, volume, Builder, FaceMesh};
use crate::fillet_num::js_num;
use crate::model::{Feature, Part, ShellFeature};
use crate::topo::{face_distance, FaceKind, MatchTolerance, Topology};

pub(crate) fn build(b: &mut Builder, _part: &Part, _f: &Feature, x: &ShellFeature) -> Result<(), String> {
    if !(x.thickness > 0.0) {
        return Err("The wall needs a thickness.".into());
    }
    let (id, state) = b.body_for(x.body)?;
    let found = if x.faces.is_empty() && x.faces_where.is_empty() { Vec::new() } else { b.find_faces(id, &x.faces, &x.faces_where).map_err(|lost| format!("The face this shell opens is gone (was: {lost})."))? };
    let t = b.topology_of(id).unwrap_or_else(|| state.topology());
    let (made, note) = shell_tracked(&state.shape, &t, &found, x.thickness).map_err(|e| format!("Shelling: {e}"))?;
    if let Some(n) = note {
        b.warn(n);
    }
    let origins = origins_after(&made, &[(&state.shape, &state.origins)], b.step);
    if let Some(w) = slivers_left(&t, &read_topology(&made, &origins)) {
        b.warn(w);
    }
    let ids = b.settle(id, made, origins, false)?;
    b.touched(ids);
    Ok(())
}

/// A face narrower than this, twice its area over its perimeter, is a sliver, mm.
const SLIVER_MM: f64 = 0.01;
/// A face smaller than this is a sliver too, mm².
const TINY_FACE_MM2: f64 = 0.01;

/// The shell's new faces that are slivers — narrower than `SLIVER_MM` or under `TINY_FACE_MM2`
/// — warned of by index, the smallest first: a round on the edges beside one fails. The ways
/// of shelling by booleans leave them where the wall meets the body's own faces (ScanRuler's
/// `shellSlivers`). A face the body had already, where it was, is not the shell's to warn of.
fn slivers_left(before: &Topology, after: &Topology) -> Option<String> {
    let tol = MatchTolerance { position: 1e-6 * after.faces.iter().fold(1.0f64, |m, f| m.max(f.fingerprint.centroid.abs().max_element())), angle: 1e-6, size: 1e-6 };
    let own = |f: &crate::topo::FaceFingerprint| before.faces.iter().any(|b| face_distance(&b.fingerprint, f, &tol).is_some());
    let mut tiny: Vec<(usize, f64, DVec3)> = Vec::new();
    for (index, f) in after.faces.iter().enumerate() {
        if own(&f.fingerprint) {
            continue;
        }
        let perimeter: f64 = after.edges.iter().filter(|e| e.faces.contains(&f.hash)).map(|e| e.fingerprint.length).sum();
        let narrow = perimeter > 0.0 && 2.0 * f.fingerprint.area / perimeter < SLIVER_MM;
        if narrow || f.fingerprint.area < TINY_FACE_MM2 {
            tiny.push((index, f.fingerprint.area, f.fingerprint.centroid));
        }
    }
    if tiny.is_empty() {
        return None;
    }
    let n = tiny.len();
    tiny.sort_by(|a, b| a.1.total_cmp(&b.1));
    let named: Vec<String> = tiny
        .iter()
        .take(3)
        .map(|(index, area, c)| {
            let a = if *area < 0.01 { crate::fillet_num::to_precision(*area, 2) } else { js_num((area * 100.0).round() / 100.0) };
            format!("face {index} ({a} mm² at ({:.1}, {:.1}, {:.1}))", c.x, c.y, c.z)
        })
        .collect();
    let more = if n > named.len() { format!(" and {} more", n - named.len()) } else { String::new() };
    Some(format!(
        "The shell left {n} sliver face{}, under {SLIVER_MM} mm wide or {TINY_FACE_MM2} mm²: {}{more} — a round on the edges beside {} will likely fail.",
        if n == 1 { "" } else { "s" },
        named.join(", "),
        if n == 1 { "it" } else { "them" }
    ))
}

/// One way of hollowing failed: why, and whether that is the kernel's own words.
struct Reason {
    why: String,
    raw: bool,
}

/// What one way of hollowing came to: the hollow body with what the row says of it, why it
/// failed, or nothing — the way does not apply.
enum Way {
    Made(Body, String),
    Why(String),
    None,
}

/// Hollow a body to a wall of `thickness`, the given faces (by index) taken away to open it;
/// none leaves a closed hollow body — the body less its own inward offset (ScanRuler's
/// `shellTracked`). The hollow body, and what the row says where another way than the first
/// made it.
fn shell_tracked(body: &Body, t: &Topology, open: &[usize], thickness: f64) -> Result<(Body, Option<String>), String> {
    let mut reasons: Vec<Reason> = Vec::new();
    match shell_plain(body, t, open, thickness) {
        Ok(made) => return Ok((made, None)),
        Err(r) => reasons.push(r),
    }
    // A round no larger than the wall has no inward offset — its inner radius would be nothing
    // or less — and the kernel's shell gives up on the whole body over it. The wall is then made
    // on the body with those rounds taken off, and cut back to the body's own outside: rounded
    // outside as modelled, sharp inside where the round ran out.
    let gave_up = format!(" The kernel's thick solid gave up on it: {}.", most_specific(&reasons[..1]));
    match shell_sharp_inside(body, t, open, thickness) {
        Way::Made(made, note) => return Ok((made, Some(format!("{note}{gave_up}").trim().to_string()))),
        Way::Why(why) => reasons.push(Reason { why: format!("with the rounds no larger than the wall left sharp inside, {why}"), raw: false }),
        Way::None => {}
    }
    // The kernel's shell also gives up on bodies it has no business giving up on. The wall is
    // then made the long way, by offset and cut.
    match shell_by_offset_cut(body, t, open, thickness) {
        Way::Made(made, _) => return Ok((made, Some(format!("Built the long way, by offset and cut, with booleans: the body less its own inward offset and the walls over the opened faces.{gave_up}")))),
        Way::Why(why) => reasons.push(Reason { why: format!("by offset and cut, {why}"), raw: false }),
        Way::None => {}
    }
    Err(most_specific(&reasons))
}

/// The kernel's own shell, the body hollowed and opened at the faces given; with none, the
/// body less its own inward offset. Held to taking something away: a hollow weighs no more
/// than the closed one — the body less its core.
fn shell_plain(body: &Body, t: &Topology, open: &[usize], thickness: f64) -> Result<Body, Reason> {
    let core = inner_core(body, t, thickness);
    if open.is_empty() {
        let core = core.map_err(|why| Reason { why, raw: false })?;
        return match kernel::boolean(body, &core, BoolOp::Cut) {
            Ok(Some(made)) => Ok(made),
            Ok(None) => Err(Reason { why: "the core took the whole body.".into(), raw: false }),
            Err(e) => Err(Reason { why: raw_refusal(&kerr(e)), raw: true }),
        };
    }
    let v = volume(body).unwrap_or(0.0);
    let most = match &core {
        Ok(c) => (v * (1.0 - 1e-6)).min(v - volume(c).unwrap_or(0.0) + v * 1e-4),
        Err(_) => v * (1.0 - 1e-6),
    };
    let points: Vec<_> = open.iter().filter_map(|i| t.faces.get(*i)).map(|f| sv3(f.point)).collect();
    let made = kernel::shell(body, &points, thickness).map_err(|e| Reason { why: raw_refusal(&kerr(e)), raw: true })?;
    if !closed_solid(&made) || !(volume(&made).unwrap_or(f64::INFINITY) < most) {
        return Err(Reason { why: "the thick solid handed the body back without hollowing it.".into(), raw: false });
    }
    Ok(made)
}

/// What is left inside a wall of `distance`: the body's inward offset, or why there is none —
/// see [`offset_fault`].
fn inner_core(body: &Body, t: &Topology, distance: f64) -> Result<Body, String> {
    let points: Vec<_> = t.faces.iter().map(|f| sv3(f.point)).collect();
    let core = kernel::offset_faces(body, &points, -distance).map_err(|e| format!("the kernel's offset of the body inward by the wall failed ({}).", raw_refusal(&kerr(e))))?;
    if !(volume(&core).unwrap_or(0.0) > 1e-9) {
        return Err("the wall is as thick as the body — nothing is left inside to take away.".into());
    }
    match offset_fault(body, &core, -distance) {
        None => Ok(core),
        Some(fault) => Err(format!("the kernel's offset of the body inward by the wall came out broken: {fault}.")),
    }
}

/// The sides of a box, as [`offset_fault`] names them.
const SIDES: [&str; 6] = ["−x", "−y", "−z", "+x", "+y", "+z"];

/// What is wrong with the offset `made` the kernel made of `solid` by `distance` (negative
/// inward), as a sentence ends — `None` when nothing is. A true offset has the body's shells,
/// grows or shrinks, and moves each side of the body's box by the distance or more (a quarter
/// of it is allowed: the boxes are the surfaces' own extremes, and a side on an edge or a slope
/// moves further). What is said is what was found, not why (ScanRuler's `offsetFault`).
pub(crate) fn offset_fault(solid: &Body, made: &Body, distance: f64) -> Option<String> {
    let vm = volume(made).unwrap_or(0.0);
    if !(vm > 1e-9) {
        return Some("it has no volume".into());
    }
    let vs = volume(solid).unwrap_or(0.0);
    let grown = distance > 0.0;
    if if grown { !(vm > vs) } else { !(vm < vs) } {
        return Some(format!("it is no {} than the body", if grown { "larger" } else { "smaller" }));
    }
    let outer = solid.outline_bounds();
    let bx = made.outline_bounds();
    let (omin, omax, bmin, bmax) = (dv3(outer.min), dv3(outer.max), dv3(bx.min), dv3(bx.max));
    // How far each side of the box moved out — in, negative.
    let moved = [omin.x - bmin.x, omin.y - bmin.y, omin.z - bmin.z, bmax.x - omax.x, bmax.y - omax.y, bmax.z - omax.z];
    if let Some(short) = moved.iter().position(|m| if grown { *m < distance * 0.75 } else { *m > distance * 0.75 }) {
        let m = if grown { moved[short] } else { -moved[short] };
        let by = format!("{} mm", js_num((m.abs() * 100.0).round() / 100.0));
        let way = if grown { "out" } else { "in" };
        if m.abs() < 0.005 {
            return Some(format!("its box does not move {way} on the {} side", SIDES[short]));
        }
        return Some(if m < 0.0 { format!("its box moves {} by {by} on the {} side, not {way}", if grown { "in" } else { "out" }, SIDES[short]) } else { format!("its box moves {way} by only {by} on the {} side", SIDES[short]) });
    }
    let (shells, want) = (made.shell_count(), solid.shell_count());
    if shells != want {
        return Some(format!("it comes in {shells} shell{}, where the body has {want}", if shells == 1 { "" } else { "s" }));
    }
    if !made.validity().is_empty() {
        return Some("the kernel's own check finds it broken".into());
    }
    None
}

/// What the kernel said when it refused, as it said it.
fn raw_refusal(e: &str) -> String {
    e.strip_suffix('.').unwrap_or(e).to_string()
}

const REFUSED_STEP: &str = "the kernel refused a step of it (";

/// A way of hollowing failed on a step the kernel refused: its own words, as a reason the shell
/// can tell from the reasons found by looking.
fn refused_step(e: &str) -> String {
    format!("{REFUSED_STEP}{}).", raw_refusal(e))
}

/// Of the reasons the ways of hollowing gave, in the order tried, the one that says most: a
/// fact about the body (a number, a place, nothing left inside) before the kernel's own words,
/// and those before a reason that only says a way gave up. The first of the best kind.
fn most_specific(reasons: &[Reason]) -> String {
    let score = |r: &Reason| {
        if r.raw || r.why.contains(REFUSED_STEP) || r.why.contains("failed (") {
            2
        } else if r.why.chars().any(|c| c.is_ascii_digit()) || r.why.contains("nothing is left") || r.why.contains("came out broken") {
            3
        } else {
            1
        }
    };
    let Some(mut best) = reasons.first() else { return "the shell could not be made".into() };
    for r in reasons {
        if score(r) > score(best) {
            best = r;
        }
    }
    if best.raw {
        format!("the kernel refused ({})", best.why)
    } else {
        best.why.strip_suffix('.').unwrap_or(&best.why).to_string()
    }
}

/// Whether a face is a convex round that an inward offset of `distance` would shrink to
/// nothing: its outward normals fan apart about a centre no further behind the face than the
/// distance. A hole's wall and an inside fillet grow under the same offset and are left alone.
fn collapses_inward(body: &Body, mesh: Option<&FaceMesh>, face: usize, at: DVec3, distance: f64) -> bool {
    let limit = distance * (1.0 + 1e-6) + 1e-6;
    let Some(n) = mesh.and_then(|m| m.normals(face, &[at]).first().copied().flatten()) else { return false };
    match kernel::face_surface(body, face) {
        Some(FaceSurface::Cylinder { origin, axis, radius }) => {
            let (o, a) = (dv3(origin), dv3(axis.into()));
            let off = at - o;
            let radial = off - a * off.dot(a);
            n.dot(radial) > 0.0 && radius <= limit
        }
        Some(FaceSurface::Sphere { center, radius }) => n.dot(at - dv3(center)) > 0.0 && radius <= limit,
        Some(FaceSurface::Torus { center, axis, major, minor }) => {
            let (c, a) = (dv3(center), dv3(axis.into()));
            let off = at - c;
            let radial = (off - a * off.dot(a)).normalize_or_zero();
            // Across the tube: about the tube's centre circle.
            let tube = c + radial * major;
            let across = n.dot(at - tube) > 0.0 && minor <= limit;
            // Round the axis: the ring's own turn, at the point's distance off the axis.
            let ring = n.dot(radial);
            let along = ring > 1e-9 && (off - a * off.dot(a)).length() / ring <= limit;
            across || along
        }
        _ => false,
    }
}

/// The shell of a body whose convex rounds are no larger than the wall: the rounds taken off,
/// the wall made on the body without them and cut back to the body's outside (ScanRuler's
/// `shellSharpInside`). Not applicable where there are none such.
fn shell_sharp_inside(body: &Body, t: &Topology, open: &[usize], thickness: f64) -> Way {
    let opened: HashSet<usize> = open.iter().copied().collect();
    let mesh = FaceMesh::of(body);
    let rounds: Vec<usize> = t.faces.iter().enumerate().filter(|(i, f)| !opened.contains(i) && collapses_inward(body, mesh.as_ref(), *i, f.point, thickness)).map(|(i, _)| i).collect();
    if rounds.is_empty() {
        return Way::None;
    }
    let sharp = match kernel::delete_faces(body, &rounds) {
        Ok(s) if volume(&s).unwrap_or(0.0) > 1e-9 => s,
        _ => return Way::Why("the rounds no larger than the wall cannot be taken off the body.".into()),
    };
    // The faces to open, found again on the body without its rounds: a flat face has grown into
    // the corner the round left, on its plane.
    let after = read_topology(&sharp, &vec![-1; sharp.face_count()]);
    let after_prints = after.face_prints();
    let mut reopened: Vec<usize> = Vec::new();
    for i in open {
        let Some(was) = t.faces.get(*i) else { return Way::Why("a face to open is lost with the rounds taken off.".into()) };
        match same_surface(&was.fingerprint, &after_prints) {
            Some(k) => reopened.push(k),
            None => return Way::Why("a face to open is lost with the rounds taken off.".into()),
        }
    }
    let wall = if reopened.is_empty() {
        let core = match inner_core(&sharp, &after, thickness) {
            Ok(c) => c,
            Err(why) => return Way::Why(why),
        };
        match kernel::boolean(&sharp, &core, BoolOp::Cut) {
            Ok(Some(w)) => w,
            Ok(None) => return Way::Why("the core took the whole body.".into()),
            Err(e) => return Way::Why(refused_step(&kerr(e))),
        }
    } else {
        let points: Vec<_> = reopened.iter().filter_map(|k| after.faces.get(*k)).map(|f| sv3(f.point)).collect();
        match kernel::shell(&sharp, &points, thickness) {
            Ok(w) => w,
            Err(e) => return Way::Why(refused_step(&kerr(e))),
        }
    };
    let result = match kernel::boolean(body, &wall, BoolOp::Intersect) {
        Ok(Some(r)) => r,
        Ok(None) => return Way::Why("the wall cut back to the body is no closed solid.".into()),
        Err(e) => return Way::Why(refused_step(&kerr(e))),
    };
    // The outside is the body's own and must all be there still: a pin thinner than the wall
    // that went with the rounds is not a corner left sharp, it is a feature lost. And something
    // must have been taken away.
    let left = read_topology(&result, &vec![-1; result.face_count()]).face_prints();
    let kept = |i: usize| t.faces.get(i).is_some_and(|f| same_surface(&f.fingerprint, &left).is_some());
    let why = if !closed_solid(&result) {
        Some("the wall cut back to the body is no closed solid.")
    } else if !(volume(&result).unwrap_or(f64::INFINITY) < volume(body).unwrap_or(0.0) * (1.0 - 1e-6)) {
        Some("the wall cut back to the body took nothing away.")
    } else if (0..t.faces.len()).any(|i| !opened.contains(&i) && !kept(i)) {
        Some("a face of the body went with the rounds — a pin or a strip thinner than the wall.")
    } else {
        None
    };
    if let Some(w) = why {
        return Way::Why(w.into());
    }
    let n = rounds.len();
    Way::Made(result, format!("Built by cutting back, with booleans: {n} round{} no larger than the wall taken off, the wall made on the body without {} and cut back to its outside — the inside is left sharp there.", if n == 1 { "" } else { "s" }, if n == 1 { "it" } else { "them" }))
}

/// The body's material a wall deep under one of its faces: the face pushed straight in when it
/// is flat, thickened into the body otherwise — the face copied out as a sheet and thickened
/// against its outward normal, which goes in on a face of a solid; the slab is checked to lie
/// in the body (more than half of it) and made the other way if not. `None` when neither way
/// does (ScanRuler's `wallUnder`).
fn wall_under(body: &Body, t: &Topology, face: usize, thickness: f64) -> Option<Body> {
    let f = t.faces.get(face)?;
    if f.fingerprint.kind == FaceKind::Plane {
        let normal = sv3(f.fingerprint.dir);
        return kernel::face_prism(body, WholeFace { face, normal, height: 0.0 }, thickness, false, body).ok();
    }
    let sheet = kernel::copy_faces(body, &[sv3(f.point)]).ok()?;
    for sign in [-1.0, 1.0] {
        let Ok(slab) = kernel::thicken(&sheet, sign * thickness, false) else { continue };
        let whole = volume(&slab).unwrap_or(0.0);
        let within = match kernel::boolean(&slab, body, BoolOp::Intersect) {
            Ok(Some(common)) => volume(&common).unwrap_or(0.0),
            _ => 0.0,
        };
        if whole > 0.0 && within > 0.5 * whole {
            return Some(slab);
        }
    }
    None
}

/// The shell made by hand where the kernel's gives up: the body less its own inward offset is
/// the closed hollow body, and the wall over each opened face is taken off it — the face pushed
/// in by the wall, less the walls of the closed faces next to it, which reach the opened face's
/// plane as a shell's walls do (ScanRuler's `shellByOffsetCut`). Not applicable with no face to
/// open.
fn shell_by_offset_cut(body: &Body, t: &Topology, open: &[usize], thickness: f64) -> Way {
    if open.is_empty() {
        return Way::None;
    }
    let opened: HashSet<usize> = open.iter().copied().collect();
    if open.iter().any(|i| *i >= t.faces.len()) {
        return Way::Why("a face to open is not on the body.".into());
    }
    // The closed faces sharing an edge with an opened one: their walls are what the wall over
    // an opened face has to leave standing.
    let mut beside: HashSet<usize> = HashSet::new();
    for e in &t.edges {
        let at: Vec<usize> = e.faces.iter().map(|h| *h as usize).collect();
        if at.iter().any(|h| opened.contains(h)) {
            beside.extend(at.iter().filter(|h| !opened.contains(h)));
        }
    }
    let core = match inner_core(body, t, thickness) {
        Ok(c) => c,
        Err(why) => return Way::Why(why),
    };
    // The walls under the faces, the opened ones less the standing ones: cut from the body one
    // at a time after its core — the kernel's join of walls that only touch along an edge is
    // refused, where cuts one after another are not.
    let mut over: Vec<Body> = Vec::new();
    let mut standing: Vec<Body> = Vec::new();
    for (i, f) in t.faces.iter().enumerate() {
        if !opened.contains(&i) && !beside.contains(&i) {
            continue;
        }
        let Some(wall) = wall_under(body, t, i, thickness) else {
            let c = f.fingerprint.centroid;
            let kind = if f.fingerprint.kind == FaceKind::Other { "freeform" } else { f.fingerprint.kind.name() };
            return Way::Why(format!("the wall under a {kind} face at ({:.1}, {:.1}, {:.1}) cannot be made.", c.x, c.y, c.z));
        };
        if opened.contains(&i) {
            over.push(wall);
        } else {
            standing.push(wall);
        }
    }
    if over.is_empty() {
        return Way::Why("no wall over the opened faces could be made.".into());
    }
    let cut = |a: &Body, b: &Body| -> Result<Option<Body>, String> { kernel::boolean(a, b, BoolOp::Cut).map_err(|e| refused_step(&kerr(e))) };
    let mut result = match cut(body, &core) {
        Ok(Some(r)) => r,
        Ok(None) => return Way::Why("the core and the openings cut out leave no closed solid.".into()),
        Err(why) => return Way::Why(why),
    };
    for wall in over {
        let mut w = Some(wall);
        for s in &standing {
            let Some(a) = w.take() else { break };
            if !a.meets(s).unwrap_or(true) {
                w = Some(a);
                continue;
            }
            w = match cut(&a, s) {
                Ok(left) => left,
                Err(why) => return Way::Why(why),
            };
        }
        let Some(w) = w else { continue };
        result = match cut(&result, &w) {
            Ok(Some(r)) => r,
            Ok(None) => return Way::Why("the core and the openings cut out leave no closed solid.".into()),
            Err(why) => return Way::Why(why),
        };
    }
    let closed = closed_solid(&result);
    let (vr, vb, vc) = (volume(&result).unwrap_or(f64::INFINITY), volume(body).unwrap_or(0.0), volume(&core).unwrap_or(0.0));
    if !closed || !(vr < vb - vc) {
        return Way::Why(if closed { "the core and the openings cut out took less than the core.".into() } else { "the core and the openings cut out leave no closed solid.".into() });
    }
    Way::Made(result, String::new())
}

#[cfg(test)]
mod tests;
