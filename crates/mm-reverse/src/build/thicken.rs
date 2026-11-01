// SPDX-License-Identifier: AGPL-3.0-only
//! The Thicken feature's build: a sheet — a fitted surface, a sheet drawn from a sketch — made
//! a solid of a thickness, against its normal, along it, or half each way: a freeform thin
//! shell's wall from its one skin. The sheet stays a body of its own; the solid is a new body,
//! or joined to or cut from what it meets as a loft's is, its skin keeping the makers of the
//! sheet's faces. A face that bends tighter than the thickness towards the side it grows to is
//! named before the kernel is asked — it would fold the far skin over itself — with the
//! thickness it takes; the wall the solid came to along the sheet's edge is said. Ported from
//! ScanRuler (`plugins/reverse/core/build.ts` sheetFor, thickenOf, playThicken;
//! `kernelThicken.ts` thickenSheet, wallAlongEdge; `diagnoseThicken.ts`; `kernelShell.ts`
//! faceOffsetLimits, tightestBend, largestSize), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only, on the SolidCraft kernel. The bend is read off the face's display mesh — its
//! points and the surface's normals at them — where ScanRuler reads the surface's own points
//! and normals over its parameters.

use glam::DVec3;
use solidcraft_kernel::{self as kernel, Body, FaceSurface};

use super::surf_kernel::{along_polylines, MeshIndex, ShapeMesh};
use super::surf_split::on_tolerance;
use super::{diagonal_of, dv3, kerr, mesh_deflection, sv3, BodyId, BodyState, Builder};
use crate::model::{Feature, FeatureKind, Part, ThickenFeature, ThickenSide};

/// How many points along a sheet's edge its solid's wall is measured at.
const WALL_POINTS: usize = 48;

/// How far under the thickness the wall along the sheet's edge may come and still be said to be
/// the thickness: a fiftieth.
const WALL_SHORT: f64 = 0.02;

/// At most this many faces are named.
const NAMED: usize = 3;

/// How near a diagnosis finds the largest size that works, mm.
const SIZE_WITHIN_MM: f64 = 0.1;

/// The least number of halvings `largest_size` makes.
const LEAST_HALVINGS: usize = 6;

/// At most this many curved faces are offset alone.
const FACE_PROBES: usize = 48;

pub(crate) fn build(b: &mut Builder, part: &Part, _f: &Feature, x: &ThickenFeature) -> Result<(), String> {
    if !(x.thickness > 0.0) {
        return Err("The sheet needs a thickness.".into());
    }
    let (_, state) = b.body_for(x.sheet).map_err(|_| "The sheet this thickens is gone, or failed to build.".to_string())?;
    let name = body_words(part, x.sheet);
    if !state.surface {
        return Err(format!("{name} is a solid, not a sheet — a thicken takes a sheet; a solid is grown or shrunk by an offset, or hollowed by a shell."));
    }
    let sheet = state.shape.clone();
    // A patch network's sheet both ways: one way at a time it thickens as one.
    let network = matches!(part.feature(x.sheet.maker()).map(|f| &f.kind), Some(FeatureKind::Network(_)));
    if x.side == ThickenSide::Both && network && sheet.face_count() > 1 {
        return Err("A patch network’s sheet of several patches thickens one way at a time, Out or In: both ways, the kernel’s offset of it to the middle does not come back as the sheet.".into());
    }
    if let Some(folds) = thicken_folds(&sheet, &state.origins, x.thickness, x.side) {
        b.report().where_lines = folds.where_lines;
        return Err(folds.message);
    }
    let t = x.thickness;
    let said = format!("the kernel's thickening of the sheet {} by {} mm", match x.side {
        ThickenSide::Both => "both ways",
        ThickenSide::Out => "along its normal",
        ThickenSide::In => "against its normal",
    }, crate::surface_fit::js_num(t));
    let made = kernel::guard("thicken", || kernel::thicken(&sheet, if x.side == ThickenSide::In { -t } else { t }, x.side == ThickenSide::Both)).map_err(|e| {
        format!("The sheet does not thicken {} by {} mm: {said} failed ({}); every face of it takes the thickness alone, so it fails at its edge or where its faces meet.", thicken_way(x.side), mm_num(t), kernel_said(&kerr(e)))
    })?;
    let lumps = made.lumps().map(|l| l.len()).unwrap_or(1);
    if made.is_surface() || lumps != 1 || super::volume(&made).is_none_or(|v| !(v > 0.0)) {
        return Err(format!(
            "The sheet does not thicken {} by {} mm: {said} came out broken: {}; every face of it takes the thickness alone, so it fails at its edge or where its faces meet.",
            thicken_way(x.side),
            mm_num(t),
            if lumps == 1 { "the kernel's own check finds it no closed solid".to_string() } else { format!("it is {lumps} solids, not one") }
        ));
    }
    let wall = wall_along_edge(&sheet, &made, t, x.side);
    let ids = b.place(x.op, x.target, vec![made])?;
    // The skin keeps the makers of the sheet's faces.
    if x.side != ThickenSide::Both {
        keep_skin_makers(b, &ids, &sheet, &state.origins);
    }
    if let Some(w) = wall {
        match wall_said(&w, t) {
            Said::Note(n) => b.report().note = Some(n),
            Said::Warning(w) => b.warn(w),
        }
    }
    Ok(())
}

/// A body as a sentence names it: the feature that made it, as ScanRuler's builder names its
/// bodies.
fn body_words(part: &Part, id: BodyId) -> String {
    part.feature(id.maker()).map_or_else(|| format!("Body {id}"), |f| f.name.clone())
}

/// The kernel's words, less the kind of failure it starts with and the stop it ends with.
fn kernel_said(e: &str) -> String {
    let mut s = e;
    for prefix in ["invalid input: ", "the operation failed: ", "kernel internal error: "] {
        s = s.strip_prefix(prefix).unwrap_or(s);
    }
    s.trim_end_matches('.').to_string()
}

/// A length as the build's sentences give it: to a hundredth, as a number is written.
fn mm_num(v: f64) -> String {
    crate::surface_fit::js_num((v * 100.0).round() / 100.0)
}

/// Which way a sheet is thickened, as a sentence says it.
pub fn thicken_way(side: ThickenSide) -> &'static str {
    match side {
        ThickenSide::Out => "along its normal",
        ThickenSide::In => "against its normal",
        ThickenSide::Both => "both ways",
    }
}

/// The wall a thickened sheet came to along its edge: the thinnest of the points measured, and
/// where.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThickenWall {
    pub least: f64,
    pub at: DVec3,
    pub points: usize,
}

/// What a thicken's result says of the wall along the sheet's edge.
#[derive(Debug, Clone, PartialEq)]
pub enum Said {
    Note(String),
    Warning(String),
}

fn point_words(p: DVec3) -> String {
    format!("({}, {}, {})", mm_num(p.x), mm_num(p.y), mm_num(p.z))
}

/// A note where the wall along the sheet's edge is the thickness, a warning where it comes
/// short of it (ScanRuler's `wallSaid`).
pub fn wall_said(wall: &ThickenWall, thickness: f64) -> Said {
    if wall.least >= thickness * (1.0 - WALL_SHORT) {
        return Said::Note(format!("The wall along the sheet's edge is {} mm at its thinnest, measured at {} points.", mm_num(wall.least), wall.points));
    }
    Said::Warning(format!(
        "The wall along the sheet's edge comes to {} mm at {}, under the {} mm asked (measured at {} points): the kernel cut the wall short there, or the far skin comes back near it.",
        mm_num(wall.least),
        point_words(wall.at),
        mm_num(thickness),
        wall.points
    ))
}

/// The largest size below `size` at which `works` holds, by halving to within SIZE_WITHIN_MM
/// (and at least six times) — zero when it holds at none tried (ScanRuler's `largestSize`).
pub fn largest_size(size: f64, mut works: impl FnMut(f64) -> bool) -> f64 {
    let halvings = LEAST_HALVINGS.max((size / SIZE_WITHIN_MM).log2().ceil().max(0.0) as usize);
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

/// One face of a sheet offset on its own: the largest distance it takes, where it bends
/// tightest towards the side it goes and that radius, and what the kernel said where it would
/// not offset the face alone by the distance asked.
#[derive(Debug, Clone, PartialEq)]
struct FaceLimit {
    face: usize,
    surface: &'static str,
    up_to: f64,
    bend: Option<(f64, DVec3)>,
    refused: Option<String>,
}

/// A surface as a sentence names it.
fn surface_name(body: &Body, face: usize) -> &'static str {
    match kernel::face_surface(body, face) {
        Some(FaceSurface::Plane { .. }) => "plane",
        Some(FaceSurface::Cylinder { .. }) => "cylinder",
        Some(FaceSurface::Cone { .. }) => "cone",
        Some(FaceSurface::Sphere { .. }) => "sphere",
        Some(FaceSurface::Torus { .. }) => "torus",
        None => "B-spline",
    }
}

/// Where each face of a mesh curves tightest towards the side `sign` goes (negative: against
/// the normals, where it is convex), and that radius: the shape operator of each triangle read
/// off its corners' points and normals — the normals' turn over the triangle's edges by least
/// squares, whose larger eigenvalue is the tightest convex curvature and the smaller, negated,
/// the tightest hollow (as ScanRuler's `tightestBend` reads it off the surface).
fn tightest_bends(m: &ShapeMesh, normals: &[DVec3], sign: f64) -> Vec<Option<(f64, DVec3)>> {
    let faces = m.tri_face.iter().copied().max().map_or(0, |f| f as usize + 1);
    let mut best: Vec<Option<(f64, DVec3)>> = vec![None; faces];
    for (t, f) in m.triangles.iter().zip(&m.tri_face) {
        let p = t.map(|v| m.positions.get(v as usize).copied().unwrap_or_default());
        let n = t.map(|v| normals.get(v as usize).copied().unwrap_or_default());
        let tn = (p[1] - p[0]).cross(p[2] - p[0]);
        let (Some(e1), Some(tn)) = ((p[1] - p[0]).try_normalize(), tn.try_normalize()) else { continue };
        let e2 = tn.cross(e1);
        // dn = S dp over the three edges, S symmetric [[a, b], [b, c]].
        let (mut ata, mut atb) = ([[0.0f64; 3]; 3], [0.0f64; 3]);
        for (i, j) in [(0, 1), (1, 2), (2, 0)] {
            let (dp, dn) = (p[j] - p[i], n[j] - n[i]);
            let (x, y) = (dp.dot(e1), dp.dot(e2));
            let (u, v) = (dn.dot(e1), dn.dot(e2));
            for (row, rhs) in [([x, y, 0.0], u), ([0.0, x, y], v)] {
                for r in 0..3 {
                    for c in 0..3 {
                        ata[r][c] += row[r] * row[c];
                    }
                    atb[r] += row[r] * rhs;
                }
            }
        }
        let Some([a, bb, c]) = solve3(ata, atb) else { continue };
        let half = (a + c) / 2.0;
        let spread = (half * half - (a * c - bb * bb)).max(0.0).sqrt();
        let k = if sign < 0.0 { half + spread } else { -(half - spread) };
        if !(k.is_finite() && k > 1e-9) {
            continue;
        }
        let slot = &mut best[*f as usize];
        if slot.is_none_or(|(r, _)| 1.0 / k < r) {
            *slot = Some((1.0 / k, (p[0] + p[1] + p[2]) / 3.0));
        }
    }
    best
}

fn solve3(m: [[f64; 3]; 3], b: [f64; 3]) -> Option<[f64; 3]> {
    let det = |m: [[f64; 3]; 3]| m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    let d = det(m);
    if !(d.abs() > 1e-18) {
        return None;
    }
    let mut out = [0.0; 3];
    for (k, o) in out.iter_mut().enumerate() {
        let mut mk = m;
        for r in 0..3 {
            mk[r][k] = b[r];
        }
        *o = det(mk) / d;
    }
    Some(out)
}

/// A face's own offset refused or not: a sheet of that face alone thickened by `d`.
fn refusal(face: &Body, d: f64) -> Option<String> {
    match kernel::guard("thicken", || kernel::thicken(face, d, false)) {
        Ok(b) if !b.is_surface() && b.face_count() > 0 => None,
        Ok(_) => Some("no face came back".into()),
        Err(e) => Some(kernel_said(&kerr(e))),
    }
}

/// Each curved face of a sheet offset alone by `distance` (negative: against its normals): how
/// far it goes (ScanRuler's `faceOffsetLimits`). A face fails in two ways: it curves tighter
/// than the distance towards the side the offset goes, and its offset would fold over itself
/// there; or the kernel refuses its offset — bisected for the largest distance it takes.
fn face_offset_limits(sheet: &Body, distance: f64) -> Vec<FaceLimit> {
    let t = distance.abs();
    let sign = if distance < 0.0 { -1.0 } else { 1.0 };
    let tol = mesh_deflection(diagonal_of(sheet)) / 4.0;
    let Ok(dm) = sheet.display_mesh(tol) else { return Vec::new() };
    let m = ShapeMesh { positions: dm.positions.iter().map(|p| dv3(*p)).collect(), triangles: dm.triangles.clone(), tri_face: dm.tri_face.clone(), edges: Vec::new(), edge_faces: Vec::new(), seams: Vec::new() };
    let normals: Vec<DVec3> = dm.normals.iter().map(|n| dv3(*n)).collect();
    let bends = tightest_bends(&m, &normals, sign);
    let points = m.face_points(sheet.face_count());
    let curved: Vec<usize> = (0..sheet.face_count()).filter(|i| !matches!(kernel::face_surface(sheet, *i), Some(FaceSurface::Plane { .. }))).collect();
    // Freeform faces first: a round's limit is its radius.
    let (free, analytic): (Vec<usize>, Vec<usize>) = curved.into_iter().partition(|i| kernel::face_surface(sheet, *i).is_none());
    let mut out = Vec::new();
    for face in free.into_iter().chain(analytic).take(FACE_PROBES) {
        let bend = bends.get(face).copied().flatten();
        let reach = t.min(bend.map_or(f64::INFINITY, |b| b.0));
        let alone = points.get(face).copied().flatten().and_then(|p| kernel::copy_faces(sheet, &[sv3(p)]).ok());
        let refused = alone.as_ref().and_then(|f| refusal(f, sign * t));
        let mut up_to = reach;
        if let (Some(_), Some(f)) = (&refused, &alone) {
            if reach == t || refusal(f, sign * reach).is_some() {
                up_to = largest_size(reach, |at| refusal(f, sign * at).is_none());
            }
        }
        out.push(FaceLimit { face, surface: surface_name(sheet, face), up_to, bend, refused });
    }
    out
}

/// Why a sheet does not thicken: the faces that do not take the thickness alone, named, and the
/// lines to draw red.
struct Folds {
    message: String,
    where_lines: Vec<Vec<DVec3>>,
}

/// The faces of a sheet that keep it from thickening by `thickness` the way asked: each face
/// offset alone towards the side the solid grows to — by the thickness, or by half of it each
/// way for both — and those that go less far named, the nearest to nothing first, with the
/// thickness each takes, the point where it bends tightest, and its outline and a cross to draw
/// red (ScanRuler's `thickenFolds`). `None` when every face takes it.
fn thicken_folds(sheet: &Body, origins: &[i64], thickness: f64, side: ThickenSide) -> Option<Folds> {
    let t = thickness;
    let distances: Vec<f64> = match side {
        ThickenSide::Out => vec![t],
        ThickenSide::In => vec![-t],
        ThickenSide::Both => vec![t / 2.0, -t / 2.0],
    };
    let mut worst: Vec<(FaceLimit, f64, f64)> = Vec::new();
    for d in distances {
        let scale = t / d.abs();
        for f in face_offset_limits(sheet, d) {
            if !(f.up_to < d.abs() * (1.0 - 1e-6)) {
                continue;
            }
            let up_to = f.up_to * scale;
            match worst.iter_mut().find(|w| w.0.face == f.face) {
                Some(w) if up_to < w.1 => *w = (f, up_to, d),
                Some(_) => {}
                None => worst.push((f, up_to, d)),
            }
        }
    }
    if worst.is_empty() {
        return None;
    }
    worst.sort_by(|a, b| a.1.total_cmp(&b.1));
    let name = |f: &FaceLimit| {
        let made = origins.get(f.face).copied().filter(|o| *o >= 0);
        format!("face {} ({}{})", f.face, f.surface, made.map_or(String::new(), |m| format!(", made by feature {m}")))
    };
    let folds = |f: &FaceLimit, d: f64| f.bend.is_some_and(|(r, _)| r < d.abs() * (1.0 - 1e-6));
    let towards = |d: f64| if d > 0.0 { "the side its normal points to" } else { "the side against its normal" };
    let how = |(f, _, d): &(FaceLimit, f64, f64)| match f.bend {
        Some((r, at)) if folds(f, *d) => format!("it bends at a radius of {} mm towards {} at {}, where the far skin folds over itself", mm_num(r), towards(*d), point_words(at)),
        _ => format!("the kernel refuses its offset past that{}", f.refused.as_ref().map_or(String::new(), |r| format!(" ({r})"))),
    };
    let named: Vec<String> = worst.iter().take(NAMED).map(|w| format!("{} by {} mm at most — {}", name(&w.0), mm_num(w.1), how(w))).collect();
    let more = if worst.len() > named.len() { format!("; and {} more", worst.len() - named.len()) } else { String::new() };
    let n = worst.len();
    let up_to = worst[0].1;
    let can = if up_to > 0.0 {
        format!("It thickens {} by up to {} mm{}.", thicken_way(side), mm_num(up_to), if side == ThickenSide::Both { "" } else { "; or thicken it the other way, or both ways" })
    } else {
        format!("It does not thicken {} at any thickness.", thicken_way(side))
    };
    let message = format!(
        "The sheet does not thicken {} by {} mm: {} less alone: {}{} — outlined red on the part. {}",
        thicken_way(side),
        mm_num(t),
        if n == 1 { "a face of it takes".to_string() } else { format!("{n} faces of it take") },
        named.join("; "),
        more,
        can
    );
    let faces: Vec<u32> = worst.iter().map(|w| w.0.face as u32).collect();
    let mut where_lines: Vec<Vec<DVec3>> = Vec::new();
    if let Ok(m) = ShapeMesh::of(sheet, 1.0) {
        for (e, on) in m.edges.iter().zip(&m.edge_faces) {
            if e.len() >= 2 && on.iter().any(|f| faces.contains(f)) {
                where_lines.push(e.clone());
            }
        }
    }
    let r = t.max(diagonal_of(sheet) * 0.02);
    for (f, _, d) in &worst {
        if let (true, Some((_, at))) = (folds(f, *d), f.bend) {
            for axis in [DVec3::X, DVec3::Y, DVec3::Z] {
                where_lines.push(vec![at - axis * r, at + axis * r]);
            }
        }
    }
    Some(Folds { message, where_lines })
}

/// The wall of a thickened sheet along the sheet's edge (ScanRuler's `wallAlongEdge`): from
/// points spaced evenly along its open edges, the distance through the solid — to its far skin
/// one way, or to both skins for a sheet thickened both ways. A thickening that runs true is
/// the thickness everywhere; less is where the kernel trimmed the wall short or the far skin
/// comes back near. `None` where no edge or no skin is found.
fn wall_along_edge(sheet: &Body, solid: &Body, thickness: f64, side: ThickenSide) -> Option<ThickenWall> {
    let sheet_mesh = ShapeMesh::of(sheet, 1.0).ok()?;
    let sheet_index = MeshIndex::of(&sheet_mesh)?;
    let open = sheet_mesh.open_edges();
    if open.is_empty() {
        return None;
    }
    let m = ShapeMesh::of(solid, 1.0).ok()?;
    let tol = on_tolerance(diagonal_of(sheet), diagonal_of(solid));
    let faces = m.tri_face.iter().copied().max().map_or(0, |f| f as usize + 1);
    // Each face of the solid: a skin (its normal along the sheet's) on one side or the other,
    // the sheet itself, or a wall round its edge.
    let mut skins: [Vec<u32>; 2] = [Vec::new(), Vec::new()];
    for f in 0..faces as u32 {
        let tris = m.face_triangles(f);
        let Some(t) = tris.get(tris.len() / 2) else { continue };
        let [a, b, c] = t.map(|v| m.positions.get(v as usize).copied().unwrap_or_default());
        let (p, n) = ((a + b + c) / 3.0, (b - a).cross(c - a).normalize_or_zero());
        let Some((q, d, _, sn)) = sheet_index.nearest(p, thickness * 2.0 + tol) else { continue };
        if n.dot(sn).abs() < 0.9 || d <= tol {
            continue;
        }
        let along = (p - q).dot(sn) > 0.0;
        skins[along as usize].push(f);
    }
    let index_of = |list: &[u32]| {
        let tris: Vec<[u32; 3]> = m.triangles.iter().zip(&m.tri_face).filter(|(_, f)| list.contains(f)).map(|(t, _)| *t).collect();
        let tf: Vec<u32> = m.tri_face.iter().copied().filter(|f| list.contains(f)).collect();
        MeshIndex::new(&m.positions, &tris, &tf)
    };
    let reach = thickness * 4.0 + 1.0;
    let points = along_polylines(&open, WALL_POINTS);
    let mut least = f64::INFINITY;
    let mut at = DVec3::ZERO;
    let mut counted = 0;
    match side {
        ThickenSide::Both => {
            let (Some(a), Some(b)) = (index_of(&skins[0]), index_of(&skins[1])) else { return None };
            for p in &points {
                let (Some(x), Some(y)) = (a.nearest(*p, reach), b.nearest(*p, reach)) else { continue };
                counted += 1;
                if x.1 + y.1 < least {
                    least = x.1 + y.1;
                    at = *p;
                }
            }
        }
        _ => {
            let far = index_of(&skins[(side == ThickenSide::Out) as usize])?;
            for p in &points {
                let Some(x) = far.nearest(*p, reach) else { continue };
                counted += 1;
                if x.1 < least {
                    least = x.1;
                    at = *p;
                }
            }
        }
    }
    (counted > 0).then_some(ThickenWall { least, at, points: counted })
}

/// The faces of the bodies a thicken went into that lie on the sheet keep the makers of the
/// sheet's faces they lie on: the skin is the surface's.
fn keep_skin_makers(b: &mut Builder, ids: &[BodyId], sheet: &Body, origins: &[i64]) {
    let Ok(sm) = ShapeMesh::of(sheet, 1.0) else { return };
    let Some(index) = MeshIndex::of(&sm) else { return };
    for id in ids {
        let Some(state) = b.body(*id).cloned() else { continue };
        let Ok(m) = ShapeMesh::of(&state.shape, 1.0) else { continue };
        let tol = on_tolerance(diagonal_of(sheet), diagonal_of(&state.shape));
        let mut next = state.origins.clone();
        let mut changed = false;
        for (face, p) in m.face_points(state.shape.face_count()).into_iter().enumerate() {
            let Some(p) = p else { continue };
            if next.get(face) != Some(&(b.step as i64)) {
                continue;
            }
            if let Some((_, _, on, _)) = index.nearest(p, tol) {
                if let Some(o) = origins.get(on as usize).copied().filter(|o| *o >= 0) {
                    next[face] = o;
                    changed = true;
                }
            }
        }
        if changed {
            let mut s = BodyState::new(state.shape.clone(), next);
            s.surface = state.surface;
            s.label = state.label.clone();
            b.set_body(*id, s);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build::surface::tests::{flange, patch_scan, surface};
    use crate::build::{build, build_scanned, BuildScan};
    use crate::frame::WorldAxis;
    use crate::model::{BodyOp, SheetFeature, SketchFeature};
    use crate::sketch::{Shape, Sketch};
    use glam::DVec2;
    use std::sync::Arc;

    fn thicken(sheet: u32, side: ThickenSide, t: f64) -> FeatureKind {
        FeatureKind::Thicken(ThickenFeature { sheet: BodyId::new(sheet), thickness: t, side, op: BodyOp::New, target: None, exprs: Default::default() })
    }

    // reThicken.test.ts: makes a fitted B-spline skin one valid solid of 1.8 mm, each way about
    // its area times the thickness; and keeps the skin the surface's.
    #[test]
    fn makes_a_fitted_skin_one_solid_each_way_about_its_area_times_the_thickness() {
        let (pos, tris, region) = patch_scan(15.0, flange);
        let scan = Arc::new(BuildScan::new(pos, tris));
        for side in [ThickenSide::In, ThickenSide::Out, ThickenSide::Both] {
            let mut part = Part::default();
            let skin = part.add(surface(region.clone(), 5, 5.0));
            let th = part.add(thicken(skin, side, 1.8));
            let built = build_scanned(&part, Some(scan.clone()));
            assert!(built.errors.is_empty(), "{side:?}: {:?}", built.errors);
            let sheet = built.bodies.iter().find(|b| b.id == BodyId::new(skin)).unwrap();
            assert!(sheet.surface);
            let area = super::super::surf_kernel::surface_area(&sheet.body);
            let solid = built.bodies.iter().find(|b| b.id == BodyId::new(th)).unwrap();
            assert!(!solid.surface);
            let v = super::super::volume(&solid.body).unwrap();
            assert!((v / (area * 1.8) - 1.0).abs() < 0.03, "{side:?}: {v} against {}", area * 1.8);
            let note = built.surf.get(&th).and_then(|r| r.note.clone()).unwrap_or_default();
            assert!(note.starts_with("The wall along the sheet's edge is 1.8 mm at its thinnest, measured at "), "{side:?}: {note} {:?}", built.warnings.get(&th));
            // Out grows along the normal, in against it, both half each way.
            let zs = |body: &Body| {
                let m = ShapeMesh::of(body, 1.0).unwrap();
                m.positions.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), p| (lo.min(p.z), hi.max(p.z)))
            };
            let (sb, b) = (zs(&sheet.body), zs(&solid.body));
            // Along the face's normal at its middle.
            let up = sheet.topology.faces[0].fingerprint.dir.z.signum();
            let (above, below) = if up > 0.0 { (b.1 - sb.1, sb.0 - b.0) } else { (sb.0 - b.0, b.1 - sb.1) };
            match side {
                ThickenSide::Out => assert!(above > 1.0 && below < 0.3, "{above} {below}"),
                ThickenSide::In => assert!(below > 1.0 && above < 0.3, "{above} {below}"),
                ThickenSide::Both => assert!(above.min(below) > 0.4, "{above} {below}"),
            }
            if side == ThickenSide::In {
                let faces = &solid.topology.faces;
                assert_eq!(faces.iter().filter(|f| f.fingerprint.origin == skin as i64).count(), 1);
                assert!(faces.iter().filter(|f| f.fingerprint.origin == th as i64).count() > 1);
            }
        }
    }

    // reThicken.test.ts: names the face that bends tighter than the thickness, and the thickness
    // it takes.
    #[test]
    fn names_the_face_that_bends_tighter_than_the_thickness() {
        let mut outcomes = Vec::new();
        for side in [ThickenSide::In, ThickenSide::Out] {
            let mut part = Part::default();
            let mut s = Sketch::default();
            s.add_shape(Shape::Line { a: DVec2::new(-20.0, 0.0), b: DVec2::new(0.0, 0.0) });
            s.add_shape(Shape::Arc { c: DVec2::new(0.0, 2.0), r: 2.0, start: -std::f64::consts::FRAC_PI_2, sweep: std::f64::consts::FRAC_PI_2 });
            s.add_shape(Shape::Line { a: DVec2::new(2.0, 2.0), b: DVec2::new(2.0, 20.0) });
            let sk = part.add(FeatureKind::Sketch(SketchFeature::on_world(WorldAxis::Z, 0.0, s)));
            let strip = part.add(FeatureKind::Sheet(SheetFeature { sketch_id: sk, from: 0.0, to: 20.0, exprs: Default::default() }));
            let th = part.add(thicken(strip, side, 3.0));
            let built = build(&part);
            outcomes.push((built.errors.get(&th).cloned(), built.surf.get(&th).map(|r| r.where_lines.len()).unwrap_or(0), strip));
        }
        let failed: Vec<_> = outcomes.iter().filter(|o| o.0.is_some()).collect();
        assert_eq!(failed.len(), 1, "{outcomes:?}");
        let (message, lines, strip) = failed[0].clone();
        let message = message.unwrap();
        let head = format!("face 1 (cylinder, made by feature {strip}) by ");
        assert!(message.starts_with("The sheet does not thicken ") && message.contains(" by 3 mm: a face of it takes less alone: ") && message.contains(&head), "{message}");
        assert!(message.contains(" mm at most — it bends at a radius of 2 mm towards the side "), "{message}");
        let up_to: f64 = message.split(&head).nth(1).and_then(|s| s.split(' ').next()).and_then(|s| s.parse().ok()).unwrap();
        assert!(up_to > 1.9 && up_to <= 2.0, "{up_to}");
        assert!(message.contains("; or thicken it the other way, or both ways."), "{message}");
        assert!(lines > 0);
    }

    // reThicken.test.ts: refuses a solid for a sheet, by name.
    #[test]
    fn refuses_a_solid_for_a_sheet_by_name() {
        let mut part = Part::default();
        let mut s = Sketch::default();
        s.add_rectangle(DVec2::new(0.0, 0.0), DVec2::new(10.0, 10.0));
        let regions = crate::build::default_regions(&crate::regions::find_regions(&s));
        let sk = part.add(FeatureKind::Sketch(SketchFeature::on_world(WorldAxis::Z, 0.0, s)));
        let ex = part.add(FeatureKind::Extrude(crate::model::ExtrudeFeature::new(sk, regions, crate::model::ExtrudeExtent::Distance { from: 0.0, to: 5.0 }, BodyOp::New)));
        let th = part.add(thicken(ex, ThickenSide::In, 1.0));
        let built = build(&part);
        assert!(built.errors.get(&th).is_some_and(|e| e.contains("is a solid, not a sheet")), "{:?}", built.errors);
    }

    // reThicken.test.ts: joins the wall to the body it names.
    #[test]
    fn joins_the_wall_to_the_body_it_names() {
        let (pos, tris, region) = patch_scan(15.0, |_, _| 10.0);
        let scan = Arc::new(BuildScan::new(pos, tris));
        let mut part = Part::default();
        let skin = part.add(surface(region, 4, 10.0));
        let mut s = Sketch::default();
        s.add_rectangle(DVec2::new(-20.0, -20.0), DVec2::new(20.0, 20.0));
        let regions = crate::build::default_regions(&crate::regions::find_regions(&s));
        let sk = part.add(FeatureKind::Sketch(SketchFeature::on_world(WorldAxis::Z, 0.0, s)));
        let ex = part.add(FeatureKind::Extrude(crate::model::ExtrudeFeature::new(sk, regions, crate::model::ExtrudeExtent::Distance { from: 0.0, to: 10.0 }, BodyOp::New)));
        let th = part.add(FeatureKind::Thicken(ThickenFeature { sheet: BodyId::new(skin), thickness: 2.0, side: ThickenSide::Both, op: BodyOp::Join, target: Some(BodyId::new(ex)), exprs: Default::default() }));
        let built = build_scanned(&part, Some(scan));
        assert!(built.errors.is_empty(), "{:?}", built.errors);
        assert_eq!(built.touched.get(&th), Some(&vec![BodyId::new(ex)]));
        let block = built.bodies.iter().find(|b| b.id == BodyId::new(ex)).unwrap();
        let area = super::super::surf_kernel::surface_area(&built.bodies.iter().find(|b| b.id == BodyId::new(skin)).unwrap().body);
        let v = super::super::volume(&block.body).unwrap();
        assert!((v - (40.0 * 40.0 * 10.0 + area * 1.0)).abs() < 1.0, "{v}");
        assert!(!built.bodies.iter().any(|b| b.id == BodyId::new(th)));
    }

    #[test]
    fn says_the_wall_short_of_the_thickness_as_a_warning() {
        let w = ThickenWall { least: 1.5, at: DVec3::new(1.0, 2.0, 3.0), points: 48 };
        assert_eq!(wall_said(&w, 2.0), Said::Warning("The wall along the sheet's edge comes to 1.5 mm at (1, 2, 3), under the 2 mm asked (measured at 48 points): the kernel cut the wall short there, or the far skin comes back near it.".into()));
        assert_eq!(largest_size(2.0, |x| x <= 1.0), 1.0);
        assert_eq!(largest_size(2.0, |_| false), 0.0);
    }
}
