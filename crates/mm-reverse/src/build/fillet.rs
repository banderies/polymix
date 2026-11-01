// SPDX-License-Identifier: AGPL-3.0-only
//! The Fillet feature's build: edges of a body rounded at a radius — the picked ones found again
//! by fingerprint (and by the queries they were picked by) on the body as it stands, with the
//! edges tangent to them where the round chains — and edges rounded at a radius running evenly
//! from one end to the other. Ported from ScanRuler (`plugins/reverse/core/resolve.ts` the
//! fillet step, `build.ts` applyToBody, roundWhatHolds and roundVarying, `kernel.ts`
//! filletTracked and filletVariable, `kernelRounds.ts`), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only, on the SolidCraft kernel.
//!
//! What a round or a chamfer is more than one call to the kernel is here and shared with the
//! chamfer's build: the tangent chain, a round of what the kernel can (`skip_failing`), and a
//! round whose edges take the whole face between them made a hair under the size, the sliver it
//! leaves taken off ([`full_width`]).

use std::collections::HashSet;

use glam::DVec3;
use solidcraft_geom::Vec3 as SVec3;
use solidcraft_kernel::{self as kernel, BoolOp, Body, FaceSurface};

use super::{dv3, kerr, origins_after, read_topology, sv3, Builder, FaceMesh};
use crate::fillet_num::js_num;
use crate::js_number::to_fixed;
use crate::model::{Feature, FeatureId, FilletFeature, Part, VariableEdge};
use crate::topo::{describe_edge, match_edge, match_tolerance, split_edges, tangent_chain, FaceFingerprint, FaceKind, Topology, CHAIN_ANGLE, CHAIN_JOIN_MM};

pub(crate) fn build(b: &mut Builder, _part: &Part, _f: &Feature, x: &FilletFeature) -> Result<(), String> {
    if x.edges.is_empty() && x.variable.is_empty() {
        return Err("Pick the edges to round.".into());
    }
    if !x.edges.is_empty() && !(x.radius > 0.0) {
        return Err("The fillet needs a radius.".into());
    }
    if x.variable.iter().any(|v| !(v.radius[0] > 0.0 && v.radius[1] > 0.0)) {
        return Err("An edge rounded at a varying radius needs both its radii.".into());
    }
    let (id, state) = b.body_for(x.body)?;
    let found = if !x.edges.is_empty() || x.variable.is_empty() {
        b.find_edges(id, &x.edges, &x.edges_where).map_err(|lost| format!("The edge this fillet was on is gone (was: {lost})."))?
    } else {
        Vec::new()
    };
    let t = b.topology_of(id).unwrap_or_else(|| state.topology());
    let picked = chained(&t, &found, x.chain);
    // The tangent chain of an edge at a varying radius stays out of the plain edges: the kernel
    // rounds an edge's whole chain with it, so the plain round would take the varying edge
    // before its own round — and the varying round takes the chain anyway.
    let at = if x.variable.is_empty() { Vec::new() } else { varying_edges(b, id, &x.variable)? };
    let theirs: HashSet<usize> = if at.is_empty() { HashSet::new() } else { chained(&t, &at, true).into_iter().collect() };
    let hashes: Vec<usize> = picked.into_iter().filter(|h| !theirs.contains(h)).collect();
    let merged = found.iter().filter(|h| theirs.contains(h) && !at.contains(h)).count();
    if merged > 0 {
        let one = merged == 1;
        b.warn(format!("{merged} of the edges picked at the one radius {} on the tangent chain of an edge at a varying radius, and {} rounded with that edge.", if one { "lies" } else { "lie" }, if one { "is" } else { "are" }));
    }
    let step = b.step;
    let made = if x.skip_failing && x.variable.is_empty() {
        let radius = x.radius;
        round_what_holds(b, &state.shape, &state.origins, &t, &hashes, Verb::Round, radius, |body, origins, hs| {
            full_width(body, origins, step, radius, |r| kernel::fillet(body, &edge_points(&t, hs), r)).map_err(|e| format!("Filleting: {e}"))
        })?
    } else if x.variable.is_empty() {
        full_width(&state.shape, &state.origins, step, x.radius, |r| kernel::fillet(&state.shape, &edge_points(&t, &hashes), r)).map_err(|e| format!("Filleting: {e}"))?
    } else {
        round_varying(&state.shape, &state.origins, step, &t, &hashes, x.radius, &x.variable, &at)?
    };
    let ids = b.settle(id, made.shape, made.origins, false)?;
    b.touched(ids);
    Ok(())
}

/// The varying edges found again on the body as it stands before the feature, by hash.
fn varying_edges(b: &mut Builder, id: crate::model::BodyId, varying: &[VariableEdge]) -> Result<Vec<usize>, String> {
    let prints: Vec<_> = varying.iter().map(|v| v.edge.clone()).collect();
    b.find_edges(id, &prints, &[]).map_err(|lost| format!("The edge this fillet rounds at a varying radius was on is gone (was: {lost})."))
}

/// The edges picked and, with `chain`, those tangent-continuous with them — never across an
/// edge between two faces of one surface.
pub(crate) fn chained(t: &Topology, found: &[usize], chain: bool) -> Vec<usize> {
    if !chain {
        return found.to_vec();
    }
    let seeds: Vec<u32> = found.iter().map(|i| *i as u32).collect();
    tangent_chain(&t.edges, &seeds, &split_edges(t), CHAIN_ANGLE, CHAIN_JOIN_MM).into_iter().map(|h| h as usize).collect()
}

/// A point on each edge, as the kernel's blends find an edge by.
pub(crate) fn edge_points(t: &Topology, edges: &[usize]) -> Vec<SVec3> {
    edges.iter().filter_map(|i| t.edges.get(*i)).map(|e| sv3(e.fingerprint.mid)).collect()
}

/// A solid the kernel made, with the feature that first made each of its faces.
pub(crate) struct Made {
    pub(crate) shape: Body,
    pub(crate) origins: Vec<i64>,
}

/// Whether a body is a closed solid: every shell closed, every face meshed, a volume.
pub(crate) fn closed_solid(body: &Body) -> bool {
    body.validity().is_empty()
}

/// How far under the asked size a round or a chamfer that takes the whole face between its
/// edges is made — see [`full_width`] — tried in turn, the least first. Relative: at the least a
/// face 5 mm wide is left 0.05 µm of itself; a chamfer wants the most, 50 µm in a metre.
const FULL_WIDTH_SLACK: [f64; 3] = [1e-8, 1e-6, 1e-5];

/// A round or a chamfer whose edges take the whole face between them — a wall 5 mm thick
/// rounded 2.5 mm on both edges into a half round, a bar's ends into a pill's — which the
/// kernel refuses at the exact size: the face between would close up to nothing. It is made a
/// hair under the size instead (`FULL_WIDTH_SLACK`), which leaves that face as a sliver far
/// thinner than any tolerance, and the sliver is then taken off, its neighbours closing over it
/// — the two rounds one half round, as the size asked. Anything the operation does at the exact
/// size stands (ScanRuler's `fullWidth`).
pub(crate) fn full_width(body: &Body, origins: &[i64], step: FeatureId, size: f64, make: impl Fn(f64) -> kernel::Result<Body>) -> Result<Made, String> {
    let made = |shape: Body| {
        let o = origins_after(&shape, &[(body, &origins.to_vec())], step);
        Made { shape, origins: o }
    };
    let mut exact: Option<Body> = None;
    let mut failure: Option<String> = None;
    match make(size) {
        Ok(m) if closed_solid(&m) => return Ok(made(m)),
        Ok(m) => exact = Some(m),
        Err(e) => failure = Some(kerr(e)),
    }
    let mut near: Option<(Body, f64)> = None;
    for s in FULL_WIDTH_SLACK {
        if let Ok(m) = make(size * (1.0 - s)) {
            if closed_solid(&m) {
                near = Some((m, s));
                break;
            }
        }
    }
    let Some((near, slack)) = near else {
        return match exact {
            Some(m) => Ok(made(m)),
            None => Err(failure.unwrap_or_else(|| "the kernel made nothing.".into())),
        };
    };
    let width = 8.0 * size * slack;
    let near_made = made(near);
    if slivers(&near_made.shape, width).is_empty() {
        return Ok(near_made);
    }
    // The sliver taken off and its neighbours closed over it; or, where the kernel will not,
    // each pair of rounds on one axis filled out to the one round of the asked radius they
    // stand for. Failing both, the sliver stays: a face far thinner than any tolerance, the
    // body right.
    let closers: [&dyn Fn() -> Option<Body>; 2] = [&|| defeature_slivers(&near_made.shape, width), &|| fill_full_rounds(&near_made.shape, body, size, width)];
    for close in closers {
        let Some(closed) = close() else { continue };
        if closed_solid(&closed) && slivers(&closed, width).is_empty() {
            let o = origins_after(&closed, &[(&near_made.shape, &near_made.origins)], step);
            return Ok(Made { shape: closed, origins: o });
        }
    }
    Ok(near_made)
}

/// The faces of a body, each with its triangles' area and box, off one mesh of it.
fn face_boxes(body: &Body) -> Vec<(f64, DVec3, DVec3)> {
    let n = body.face_count();
    let mut out = vec![(0.0, DVec3::splat(f64::INFINITY), DVec3::splat(f64::NEG_INFINITY)); n];
    let Some(fm) = FaceMesh::of(body) else { return out };
    for (t, f) in fm.mesh.triangles.iter().zip(&fm.mesh.tri_face) {
        let Some(slot) = out.get_mut(*f as usize) else { continue };
        let [a, b, c] = fm.tri(t);
        slot.0 += (b - a).cross(c - a).length() * 0.5;
        for p in [a, b, c] {
            slot.1 = slot.1.min(p);
            slot.2 = slot.2.max(p);
        }
    }
    out
}

/// The faces of a body narrower than `width` — the area over the span — by index.
fn slivers(body: &Body, width: f64) -> Vec<usize> {
    face_boxes(body)
        .into_iter()
        .enumerate()
        .filter(|(_, (area, lo, hi))| {
            let span = if lo.x.is_finite() { (*hi - *lo).length() } else { 0.0 };
            span > 0.0 && area / span < width
        })
        .map(|(i, _)| i)
        .collect()
}

/// The slivers of a body taken off by the kernel's face delete, their neighbours extended to
/// close over them; `None` when it makes nothing.
fn defeature_slivers(body: &Body, width: f64) -> Option<Body> {
    let thin = slivers(body, width);
    if thin.is_empty() {
        return None;
    }
    kernel::delete_faces(body, &thin).ok()
}

/// The rounds of a body that come in pairs on one axis — the two halves of a full round made a
/// hair under its radius, a sliver between — each pair filled out by a cylinder of the full
/// `radius` on their common axis, kept within the body before the round (`solid`), so what is
/// added is the hair the slack left and nothing past the round's ends.
fn fill_full_rounds(shape: &Body, solid: &Body, radius: f64, width: f64) -> Option<Body> {
    struct Round {
        center: DVec3,
        dir: DVec3,
        lo: f64,
        hi: f64,
    }
    let boxes = face_boxes(shape);
    let mut rounds: Vec<Round> = Vec::new();
    for (i, (_, lo, hi)) in boxes.iter().enumerate() {
        let Some(FaceSurface::Cylinder { origin, axis, radius: r }) = kernel::face_surface(shape, i) else { continue };
        if (r - radius).abs() >= width || !lo.x.is_finite() {
            continue;
        }
        let (o, dir) = (dv3(origin), dv3(axis.into()));
        let mid = (*lo + *hi) / 2.0;
        let center = o + dir * (mid - o).dot(dir);
        let mut along = Vec::with_capacity(8);
        for x in [lo.x, hi.x] {
            for y in [lo.y, hi.y] {
                for z in [lo.z, hi.z] {
                    along.push((DVec3::new(x, y, z) - center).dot(dir));
                }
            }
        }
        let (a, b) = along.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), v| (a.min(*v), b.max(*v)));
        rounds.push(Round { center, dir, lo: a, hi: b });
    }
    let mut fills: Vec<Body> = Vec::new();
    let mut used: HashSet<usize> = HashSet::new();
    for i in 0..rounds.len() {
        if used.contains(&i) {
            continue;
        }
        let a = &rounds[i];
        let pair = (0..rounds.len()).find(|j| {
            let b = &rounds[*j];
            if *j <= i || used.contains(j) || a.dir.dot(b.dir).abs() < 1.0 - 1e-9 {
                return false;
            }
            let off = b.center - a.center;
            (off - a.dir * off.dot(a.dir)).length() < width
        });
        let Some(j) = pair else { continue };
        used.insert(i);
        used.insert(j);
        let b = &rounds[j];
        let shift = (b.center - a.center).dot(a.dir);
        let margin = radius * 1e-3;
        let lo = a.lo.min(b.lo + shift) - margin;
        let hi = a.hi.max(b.hi + shift) + margin;
        let center = (a.center + b.center) * 0.5;
        let axis_point = center - a.dir * (shift / 2.0);
        let base = axis_point + a.dir * lo;
        fills.push(kernel::cylinder(sv3(base), sv3(a.dir), radius, hi - lo).ok()?);
    }
    let mut filler = fills.first()?.clone();
    for f in fills.iter().skip(1) {
        filler = kernel::boolean(&filler, f, BoolOp::Union).ok()??;
    }
    let kept = kernel::boolean(&filler, solid, BoolOp::Intersect).ok()??;
    kernel::boolean(shape, &kept, BoolOp::Union).ok()?
}

/// A round or a bevel, as a note names what it does.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Verb {
    Round,
    Bevel,
}

impl Verb {
    fn word(self) -> &'static str {
        match self {
            Verb::Round => "round",
            Verb::Bevel => "bevel",
        }
    }
}

/// A round or a chamfer of many edges that leaves out what the kernel cannot do: the whole set
/// first, then — where that fails — each tangent chain added in turn and kept where the body
/// still comes out a closed solid. The feature's row says how many chains were left out and
/// where (ScanRuler's `roundWhatHolds`). `op` makes the round of the edges given by hash on the
/// body as it stands before the feature.
#[allow(clippy::too_many_arguments)]
pub(crate) fn round_what_holds(b: &mut Builder, body: &Body, origins: &[i64], t: &Topology, hashes: &[usize], verb: Verb, size: f64, op: impl Fn(&Body, &[i64], &[usize]) -> Result<Made, String>) -> Result<Made, String> {
    let holds = |hs: &[usize]| -> Option<Made> {
        match op(body, origins, hs) {
            Ok(m) if closed_solid(&m.shape) => Some(m),
            _ => None,
        }
    };
    if let Some(whole) = holds(hashes) {
        return Ok(whole);
    }
    let splits = split_edges(t);
    let mut left: HashSet<usize> = hashes.iter().copied().collect();
    let mut chains: Vec<Vec<usize>> = Vec::new();
    for h in hashes {
        if !left.contains(h) {
            continue;
        }
        let picked: Vec<usize> = tangent_chain(&t.edges, &[*h as u32], &splits, CHAIN_ANGLE, CHAIN_JOIN_MM).into_iter().map(|x| x as usize).filter(|x| left.contains(x)).collect();
        for x in &picked {
            left.remove(x);
        }
        chains.push(picked);
    }
    let mut kept: Vec<usize> = Vec::new();
    let mut made: Option<Made> = None;
    let mut skipped: Vec<Vec<usize>> = Vec::new();
    for c in &chains {
        let mut trial = kept.clone();
        trial.extend(c.iter().copied());
        match holds(&trial) {
            Some(r) => {
                made = Some(r);
                kept = trial;
            }
            None => skipped.push(c.clone()),
        }
    }
    let verb_word = verb.word();
    let Some(made) = made else { return Err(format!("None of the {} chains {verb_word}s at {} mm.", chains.len(), js_num(size))) };
    if !skipped.is_empty() {
        let at: Vec<String> = skipped
            .iter()
            .take(12)
            .map(|c| match c.first().and_then(|h| t.edges.get(*h)) {
                Some(e) => {
                    let m = e.fingerprint.mid;
                    format!("({}, {}, {})", to_fixed(m.x, 3), to_fixed(m.y, 3), to_fixed(m.z, 3))
                }
                None => "(?)".to_string(),
            })
            .collect();
        let more = if skipped.len() > at.len() { " …" } else { "" };
        b.warn(format!("{} of the {} chains left out — the kernel cannot {verb_word} {} with the rest: at {}{more}.", skipped.len(), chains.len(), if skipped.len() == 1 { "it" } else { "them" }, at.join(", ")));
    }
    Ok(made)
}

/// A fillet with edges at a varying radius: the edges at the one radius rounded first, as any
/// fillet's, then each varying edge on its own — the kernel takes one at a time — found again
/// on the shape the rounds before it left (ScanRuler's `roundVarying`). The kernel takes a
/// straight edge between two flat faces whose ends each meet one more flat face; the radius at
/// the edge's start (its first end, as the part names it) runs evenly to the other.
#[allow(clippy::too_many_arguments)]
fn round_varying(body: &Body, origins: &[i64], step: FeatureId, t: &Topology, hashes: &[usize], radius: f64, varying: &[VariableEdge], first: &[usize]) -> Result<Made, String> {
    let mut current = if hashes.is_empty() {
        Made { shape: body.clone(), origins: origins.to_vec() }
    } else {
        full_width(body, origins, step, radius, |r| kernel::fillet(body, &edge_points(t, hashes), r)).map_err(|e| format!("Filleting: {e}"))?
    };
    let mut rounded = !hashes.is_empty();
    for (i, v) in varying.iter().enumerate() {
        // Where the edge starts on the body as it stood: the start a later shape's edge is
        // turned to, whichever way the kernel now runs it.
        let Some(was) = first.get(i).and_then(|h| t.edges.get(*h)) else { continue };
        let start = was.ends[0];
        let now = if rounded { read_topology(&current.shape, &current.origins) } else { t.clone() };
        let j = if !rounded {
            first[i]
        } else {
            let tol = match_tolerance(super::diagonal_of(&current.shape));
            let prints = now.edge_prints();
            let Some(j) = match_edge(&v.edge, &prints, &tol) else {
                return Err(format!(
                    "An edge this fillet rounds at a varying radius is gone once its other edges are rounded (was: {}): one radius law per tangent chain; the kernel rounds the whole chain, and a plain round at its ends shortened or took the edge — round it in a fillet of its own, before the plain rounds.",
                    describe_edge(&v.edge)
                ));
            };
            j
        };
        let Some(e) = now.edges.get(j) else { continue };
        let from = if (e.ends[0] - start).length() <= (e.ends[1] - start).length() { e.ends[0] } else { e.ends[1] };
        // The round runs on along the edge's whole tangent chain. At one radius that is any
        // fillet of the chain; a radius that varies the kernel takes along one straight edge
        // between two flat faces alone.
        let chain = chained(&now, &[j], true);
        let next = if (v.radius[0] - v.radius[1]).abs() <= 1e-12 {
            kernel::fillet(&current.shape, &edge_points(&now, &chain), v.radius[0])
        } else if chain.len() > 1 {
            return Err(format!("{}: a radius varying along a tangent chain is not built — the kernel rounds one straight edge between two flat faces at a varying radius. Round the chain at one radius, or the edge where the chain breaks.", capitalised(&describe_edge(&v.edge))));
        } else {
            kernel::fillet_variable(&current.shape, sv3(e.fingerprint.mid), sv3(from), v.radius[0], v.radius[1])
        }
        .map_err(|e| format!("Filleting: {}", kerr(e)))?;
        let o = origins_after(&next, &[(&current.shape, &current.origins)], step);
        current = Made { shape: next, origins: o };
        rounded = true;
    }
    Ok(current)
}

/// A sentence's first word with its capital.
fn capitalised(s: &str) -> String {
    let mut c = s.chars();
    c.next().map(|f| f.to_uppercase().collect::<String>() + c.as_str()).unwrap_or_default()
}

/// The face among `candidates` lying on the same surface as `was` — the same plane, the same
/// cylinder — nearest to where it was: how a face is found again on a body whose neighbouring
/// rounds were taken off, which changes its outline and leaves its surface alone (ScanRuler's
/// `sameSurface`). Its index among the candidates.
pub(crate) fn same_surface(was: &FaceFingerprint, candidates: &[FaceFingerprint]) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for (i, f) in candidates.iter().enumerate() {
        if f.kind != was.kind {
            continue;
        }
        if was.kind == FaceKind::Plane {
            if f.dir.dot(was.dir) < 1.0 - 1e-6 {
                continue;
            }
            if (f.centroid - was.centroid).dot(was.dir).abs() > 1e-4 {
                continue;
            }
        } else {
            if f.dir.dot(was.dir).abs() < 1.0 - 1e-6 {
                continue;
            }
            if let Some(r) = was.radius {
                if f.radius.is_none_or(|fr| (fr - r).abs() > 1e-4) {
                    continue;
                }
            }
            if let (Some(wc), Some(fc)) = (was.center, f.center) {
                let off = fc - wc;
                let along = off.dot(was.dir);
                let across = off - was.dir * along;
                // A sphere's centre is a point; an axis may be read anywhere along it.
                if across.length() > 1e-4 || (was.kind == FaceKind::Sphere && along.abs() > 1e-4) {
                    continue;
                }
            }
        }
        let d = (f.centroid - was.centroid).length();
        if best.is_none_or(|(_, b)| d < b) {
            best = Some((i, d));
        }
    }
    best.map(|(i, _)| i)
}

#[cfg(test)]
pub(crate) mod tests;
