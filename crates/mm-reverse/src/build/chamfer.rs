// SPDX-License-Identifier: AGPL-3.0-only
//! The Chamfer feature's build: edges of a body bevelled — by one distance on both faces, by
//! two, or by a distance and an angle — the picked edges found again by fingerprint on the body
//! as it stands. Ported from ScanRuler (`plugins/reverse/core/resolve.ts` the chamfer step,
//! `build.ts` applyToBody and chamferSided, `chamferSides.ts`, `kernel.ts` chamferTracked),
//! © Stefan Hermann (CNC Kitchen), AGPL-3.0-only, on the SolidCraft kernel.
//!
//! A chamfer by two distances or by a distance and an angle measures from each edge's reference
//! face: the face most of the edges share — a rim's top — and, flipped, the other one
//! (`chamfer_sides::reference_faces`). The SolidCraft kernel measures its first distance on the
//! face of each edge that faces up most, and its chamfers meet at a rim's corners when asked
//! that way round; so, as ScanRuler asks Open CASCADE from its first face (`angledAsk`), each
//! tangent chain is asked for the setback on that face and on the other — the reference face's
//! own on whichever it is, and for an angle the other setback by the sines of the triangle the
//! chamfer cuts off — the chains asking alike in one call, so a rim's corners meet as one
//! chamfer.

use std::collections::HashSet;

use solidcraft_kernel::{self as kernel, Body, ChamferSide};

use super::fillet::{chained, edge_points, full_width, round_what_holds, same_surface, Made, Verb};
use super::{diagonal_of, kerr, origins_after, read_topology, Builder, FaceMesh};
use crate::chamfer_sides::{face_angle, reference_faces};
use crate::js_number::to_fixed;
use crate::model::{ChamferFeature, ChamferMode, Feature, FeatureId, Part};
use crate::topo::{describe_edge, match_edge, match_tolerance, split_edges, tangent_chain, EdgeFingerprint, FaceFingerprint, Topology, CHAIN_ANGLE, CHAIN_JOIN_MM};

pub(crate) fn build(b: &mut Builder, _part: &Part, _f: &Feature, x: &ChamferFeature) -> Result<(), String> {
    if x.edges.is_empty() {
        return Err("Pick the edges to chamfer.".into());
    }
    if !(x.distance > 0.0) {
        return Err("The chamfer needs a distance.".into());
    }
    let side = match x.mode {
        Some(ChamferMode::TwoDistances) => {
            let d2 = x.distance2.unwrap_or(0.0);
            if !(d2 > 0.0) {
                return Err("The chamfer needs its second distance.".into());
            }
            Some(ChamferSide::Distance(d2))
        }
        Some(ChamferMode::DistanceAngle) => {
            let a = x.angle_deg.unwrap_or(0.0);
            if !(a > 0.0 && a < 90.0) {
                return Err("The chamfer’s angle is between 0° and 90°.".into());
            }
            Some(ChamferSide::Angle(a.to_radians()))
        }
        Some(ChamferMode::Equal) | None => None,
    };
    let (id, state) = b.body_for(x.body)?;
    let found = b.find_edges(id, &x.edges, &x.edges_where).map_err(|lost| format!("The edge this chamfer was on is gone (was: {lost})."))?;
    let t = b.topology_of(id).unwrap_or_else(|| state.topology());
    let hashes = chained(&t, &found, x.chain);
    let step = b.step;
    let two = x.mode == Some(ChamferMode::TwoDistances);
    let bevel = |body: &Body, origins: &[i64], hs: &[usize]| -> Result<Made, String> {
        match side {
            Some(s) => chamfer_sided(body, origins, step, &t, hs, x.distance, s, x.flip, two),
            None => full_width(body, origins, step, x.distance, |d| kernel::chamfer(body, &edge_points(&t, hs), d)).map_err(|e| format!("Chamfering: {e}")),
        }
    };
    let made = if x.skip_failing { round_what_holds(b, &state.shape, &state.origins, &t, &hashes, Verb::Bevel, x.distance, bevel)? } else { bevel(&state.shape, &state.origins, &hashes)? };
    let ids = b.settle(id, made.shape, made.origins, false)?;
    b.touched(ids);
    Ok(())
}

/// The direction the kernel weighs which face of an edge faces up most by — the face whose
/// outward normal leans furthest along it measures the first distance (then +y, then +x).
const KERNEL_UP: glam::DVec3 = glam::DVec3::new(1e-6, 1e-3, 1.0);

/// One tangent chain of the picked edges: its edges' fingerprints, and its reference face.
struct Chain {
    prints: Vec<EdgeFingerprint>,
    reference: Option<FaceFingerprint>,
}

/// A chamfer by two distances or by a distance and an angle (ScanRuler's `chamferSided`): the
/// picked edges by tangent chain, each chain's reference face its first edge's; each chain asked
/// of the kernel the way that measures the first distance on that face — the chains asking
/// alike in one call, any others after them, each found again on what the call before left.
#[allow(clippy::too_many_arguments)]
fn chamfer_sided(body: &Body, origins: &[i64], step: FeatureId, t: &Topology, hashes: &[usize], distance: f64, side: ChamferSide, flip: bool, two: bool) -> Result<Made, String> {
    let how = if two { "by two distances" } else { "at an angle" };
    let mut sides: Vec<[u32; 2]> = Vec::with_capacity(hashes.len());
    for h in hashes {
        let e = t.edges.get(*h).ok_or_else(|| "An edge of the chamfer is not on the body.".to_string())?;
        let mut two_faces: Vec<u32> = Vec::new();
        for f in &e.faces {
            if !two_faces.contains(f) {
                two_faces.push(*f);
            }
        }
        match two_faces[..] {
            [a, b] => sides.push([a, b]),
            _ => return Err(format!("A chamfer {how} goes on an edge between two faces — {} has one.", describe_edge(&e.fingerprint))),
        }
    }
    let refs = reference_faces(&sides, flip);
    // The picked edges by tangent chain, each chain once; the reference face of its first edge
    // is the chain's.
    let splits = split_edges(t);
    let mut left: HashSet<usize> = hashes.iter().copied().collect();
    let mut pending: Vec<Chain> = Vec::new();
    for (i, h) in hashes.iter().enumerate() {
        if !left.contains(h) {
            continue;
        }
        let picked: Vec<usize> = tangent_chain(&t.edges, &[*h as u32], &splits, CHAIN_ANGLE, CHAIN_JOIN_MM).into_iter().map(|x| x as usize).filter(|x| left.contains(x)).collect();
        for x in &picked {
            left.remove(x);
        }
        let prints = picked.iter().filter_map(|x| t.edges.get(*x)).map(|e| e.fingerprint.clone()).collect();
        let reference = refs.get(i).and_then(|r| t.face(*r)).map(|f| f.fingerprint);
        pending.push(Chain { prints, reference });
    }
    let mut current = Made { shape: body.clone(), origins: origins.to_vec() };
    let mut first_call = true;
    while !pending.is_empty() {
        let now = if first_call { t.clone() } else { read_topology(&current.shape, &current.origins) };
        let tol = match_tolerance(diagonal_of(&current.shape));
        let prints_now = now.edge_prints();
        let faces_now = now.face_prints();
        let mesh = FaceMesh::of(&current.shape);
        let mut found: Vec<Vec<usize>> = Vec::with_capacity(pending.len());
        for c in &pending {
            let mut hit = Vec::new();
            for (j, fp) in c.prints.iter().enumerate() {
                match match_edge(fp, &prints_now, &tol) {
                    Some(at) => hit.push(at),
                    None if j == 0 => return Err(format!("An edge this chamfer cuts is gone once its other edges are chamfered (was: {}) — chamfer it in a feature of its own.", describe_edge(fp))),
                    None => {}
                }
            }
            found.push(hit);
        }
        // Each chain's ask: the setback on the face the kernel measures from first — the one
        // facing up most — and on the other, the reference face's own setback on whichever of
        // the two it is.
        let mut asks: Vec<(f64, f64)> = Vec::with_capacity(pending.len());
        for (c, hit) in pending.iter().zip(&found) {
            let e = hit.first().and_then(|h| now.edges.get(*h)).ok_or("An edge of the chamfer is not on the body.")?;
            let mut two_faces: Vec<u32> = Vec::new();
            for f in &e.faces {
                if !two_faces.contains(f) {
                    two_faces.push(*f);
                }
            }
            let [fa, fb] = two_faces[..] else { return Err(format!("An edge this chamfer cuts no longer lies between two faces (was: {}).", describe_edge(&c.prints[0]))) };
            let normal = |face: u32| mesh.as_ref().and_then(|m| surface_normal(m, face as usize, e.fingerprint.mid).or_else(|| m.normals(face as usize, &[e.fingerprint.mid]).first().copied().flatten())).unwrap_or(glam::DVec3::ZERO);
            let kernel_first = if normal(fb).dot(KERNEL_UP) > normal(fa).dot(KERNEL_UP) { fb } else { fa };
            // The reference face found again on the face of the two on its surface; the
            // kernel's first where neither is (ScanRuler's `sameSurface` over the edge's sides).
            let sided: Vec<(u32, FaceFingerprint)> = [fa, fb].iter().filter_map(|h| faces_now.get(*h as usize).map(|f| (*h, *f))).collect();
            let candidates: Vec<FaceFingerprint> = sided.iter().map(|(_, f)| *f).collect();
            let reference = c.reference.as_ref().and_then(|r| same_surface(r, &candidates)).and_then(|k| sided.get(k)).map(|(h, _)| *h).unwrap_or(kernel_first);
            let other = if reference == fa { fb } else { fa };
            // The angle the faces meet at, as the chamfer's cross-section sees it: none to speak
            // of, and there is no corner (`angledAsk`).
            let phi = face_angle([normal(reference), normal(other)]);
            if !(phi > 1e-3 && phi < std::f64::consts::PI - 1e-3) {
                return Err(format!("At {}: the faces at the edge run on into each other — there is no corner to chamfer.", describe_edge(&c.prints[0])));
            }
            let along_other = match side {
                ChamferSide::Distance(d2) => d2,
                ChamferSide::Angle(theta) => {
                    // The triangle the chamfer cuts off has the faces' angle at the edge and the
                    // chamfer's angle at the reference face: the other setback by the sines.
                    if !(phi + theta < std::f64::consts::PI - 1e-6) {
                        return Err(format!("At {}: at {}° the chamfer does not reach the other face, which meets this one at {}°.", describe_edge(&c.prints[0]), tenth(theta.to_degrees()), tenth(phi.to_degrees())));
                    }
                    distance * theta.sin() / (phi + theta).sin()
                }
            };
            asks.push(if reference == kernel_first { (distance, along_other) } else { (along_other, distance) });
        }
        let key = |a: (f64, f64)| (to_fixed(a.0, 6), to_fixed(a.1, 6));
        let taken: Vec<bool> = asks.iter().map(|a| key(*a) == key(asks[0])).collect();
        let edges: Vec<usize> = found.iter().zip(&taken).filter(|(_, k)| **k).flat_map(|(h, _)| h.iter().copied()).collect();
        let next = kernel::chamfer_sides(&current.shape, &edge_points(&now, &edges), asks[0].0, ChamferSide::Distance(asks[0].1), false).map_err(|e| format!("Chamfering: {}", kerr(e)))?;
        let o = origins_after(&next, &[(&current.shape, &current.origins)], step);
        current = Made { shape: next, origins: o };
        first_call = false;
        let mut k = 0;
        pending.retain(|_| {
            let keep = !taken[k];
            k += 1;
            keep
        });
    }
    Ok(current)
}

/// The outward surface normal of a face at its mesh's vertex nearest a point — the surface's
/// own there, where a triangle's of a round face is off it by half a chord's turn: how the
/// faces meet at an edge is read as the kernel reads it, at the edge (ScanRuler's
/// `edgeSides`, the faces' normals at the edge's middle).
fn surface_normal(mesh: &FaceMesh, face: usize, at: glam::DVec3) -> Option<glam::DVec3> {
    let m = &mesh.mesh;
    let mut best: Option<(f64, glam::DVec3)> = None;
    for (t, f) in m.triangles.iter().zip(&m.tri_face) {
        if *f as usize != face {
            continue;
        }
        for i in t {
            let (Some(p), Some(n)) = (m.positions.get(*i as usize), m.normals.get(*i as usize)) else { continue };
            let d = (super::dv3(*p) - at).length_squared();
            if best.is_none_or(|(b, _)| d < b) {
                best = Some((d, super::dv3(*n)));
            }
        }
    }
    best.and_then(|(_, n)| n.try_normalize())
}

/// A tenth of a degree, as the reasons write an angle.
fn tenth(x: f64) -> String {
    crate::fillet_num::js_num((x * 10.0).round() / 10.0)
}

#[cfg(test)]
mod tests;
