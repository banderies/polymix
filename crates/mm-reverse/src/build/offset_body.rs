// SPDX-License-Identifier: AGPL-3.0-only
//! The Offset body feature's build: a body grown all round by a distance — its convex edges
//! rounded over, as an offset rounds them — or shrunk by a negative one, kept beside the body
//! as a body of its own (a keep-out round a part, or what is cut from a case to hold it with
//! clearance) or put in its place. Ported from ScanRuler (`plugins/reverse/core/build.ts` the
//! offsetBody step, `kernel.ts` offsetSolid, `kernelShell.ts` offsetFault), © Stefan Hermann
//! (CNC Kitchen), AGPL-3.0-only.
//!
//! ScanRuler asks Open CASCADE's offset (`BRepOffsetAPI_MakeOffsetShape`, arcs at the gaps);
//! the SolidCraft kernel has no body offset, so it is built here on its public operations, in
//! two steps that come to the same solid:
//!
//! 1. Every face is moved along its outward normal by the distance, the body's topology kept
//!    (`offset_faces`: a plane is carried, a cylinder, a sphere, a cone or a torus changes its
//!    radius) — the offset with sharp edges, where each pair of moved faces meets again.
//! 2. Where the move opens a gap — the edges that are convex for a body grown, concave for one
//!    shrunk — the gap is closed by a round of the distance (`fillet`): its axis is the edge as
//!    it stood, so the round is the cylinder (the torus, along a circular edge) every point of
//!    which lies the distance from the edge, and where three rounds meet at a corner, the ball
//!    round it — what an offset with arcs makes. Edges where the faces run on smoothly open no
//!    gap and are left alone.
//!
//! A face of a freeform surface has no offset the kernel moves, and fails the offset by name,
//! as does a part thinner than twice the distance, a face curving tighter than it, or a face
//! narrower than it: the kernel refuses, or the result fails the checks ScanRuler makes of
//! OCCT's offset (`offset_fault`).

use glam::DVec3;
use solidcraft_kernel::{self as kernel, Body, FaceSurface};

use super::{diagonal_of, dv3, kerr, mesh_deflection, sv3, volume, Builder};
use crate::model::{Feature, OffsetBodyFeature, Part};

pub(crate) fn build(b: &mut Builder, _part: &Part, _f: &Feature, x: &OffsetBodyFeature) -> Result<(), String> {
    let (id, state) = b.body_for(x.body)?;
    if state.surface {
        return Err(if x.keep { "A sheet has no inside to offset — offset a solid.".into() } else { super::cut_plane::SHEET_NOT_SOLID.into() });
    }
    let made = offset_solid(&state.shape, x.distance)?;
    // Every face of an offset is new, and the offset's own.
    let own = b.own_origins(&made);
    let ids = if x.keep {
        // An offset that keeps its body is a body of its own beside it.
        let fresh = b.fresh_id()?;
        b.settle(fresh, made, own, false)?
    } else {
        b.settle(id, made, own, false)?
    };
    b.touched(ids);
    Ok(())
}

/// The sides of a box, as `offset_fault` names them.
const SIDES: [&str; 6] = ["−x", "−y", "−z", "+x", "+y", "+z"];

/// Two faces whose normals across an edge are this close to one line run on smoothly there:
/// moved by one distance they still meet, and no gap opens (about 2°).
const SMOOTH_COS: f64 = 0.9994;

/// A solid grown all round by `distance` — its convex edges rounded over, as an offset rounds
/// them — or shrunk by a negative one (ScanRuler's `offsetSolid`). Refused where the kernel
/// refuses, in its own words, and where what comes out is broken (`offset_fault` says how).
pub fn offset_solid(solid: &Body, distance: f64) -> Result<Body, String> {
    if !(distance.abs() > 1e-9) {
        return Err("Offsetting: the offset needs a distance.".into());
    }
    let what = format!("the kernel's offset of the body {} by {} mm", if distance > 0.0 { "outward" } else { "inward" }, crate::js_number::js_string(distance.abs()));
    let made = offset_rounded(solid, distance).map_err(|e| format!("Offsetting: {what} failed ({}).", e.trim_end_matches('.')))?;
    if let Some(fault) = offset_fault(solid, &made, distance) {
        return Err(format!("Offsetting: {what} came out broken: {fault}."));
    }
    Ok(made)
}

/// The two steps: every face moved, then the gaps rounded.
fn offset_rounded(solid: &Body, distance: f64) -> Result<Body, String> {
    let read = read_edges(solid)?;
    if read.faces.is_empty() {
        return Err("the body has no faces".into());
    }
    let sharp = kernel::offset_faces(solid, &read.faces.iter().map(|p| sv3(*p)).collect::<Vec<_>>(), distance).map_err(kerr)?;
    // The gaps the move opened, each found again on the moved body where its two faces' moved
    // surfaces meet.
    let gaps: Vec<DVec3> = read.edges.iter().filter(|e| if distance > 0.0 { e.convex } else { !e.convex }).map(|e| e.moved(distance)).collect();
    if gaps.is_empty() {
        return Ok(sharp);
    }
    kernel::fillet(&sharp, &gaps.iter().map(|p| sv3(*p)).collect::<Vec<_>>(), distance.abs()).map_err(kerr)
}

/// An edge between two faces that meet at an angle: its middle, the faces' outward normals
/// there, and whether it is convex (the body's inside narrower than a half turn across it).
struct SharpEdge {
    mid: DVec3,
    normals: [DVec3; 2],
    convex: bool,
}

impl SharpEdge {
    /// Where the edge lies once both its faces are moved by `distance`: the point the two
    /// moved tangent planes share nearest the edge (exact where the faces are flat).
    fn moved(&self, distance: f64) -> DVec3 {
        let [a, b] = self.normals;
        let along = (a + b) / (1.0 + a.dot(b)).max(1e-6);
        self.mid + along * distance
    }
}

/// What the offset reads of a body: a point on each face, and its edges that are not smooth.
struct Read {
    faces: Vec<DVec3>,
    edges: Vec<SharpEdge>,
}

/// The faces and the sharp edges of a body, off one fine tessellation: a point in the middle
/// of each face, and for each edge between two faces its middle, the normal of each face beside
/// it, and which way it bends — the face's inside, seen from the edge, against the other face's
/// normal.
fn read_edges(body: &Body) -> Result<Read, String> {
    let size = diagonal_of(body).max(1e-3);
    let mesh = body.display_mesh_angular(mesh_deflection(size).min(size * 2e-3).max(1e-4), kernel::DISPLAY_ANGLE / 2.0).map_err(kerr)?;
    let nf = body.face_count();
    let tri = |t: &[u32; 3]| [t[0], t[1], t[2]].map(|i| mesh.positions.get(i as usize).copied().map(dv3).unwrap_or_default());
    // Per face: its triangles' middles, normals and areas.
    let mut on: Vec<Vec<(DVec3, DVec3, f64)>> = vec![Vec::new(); nf];
    for (t, f) in mesh.triangles.iter().zip(&mesh.tri_face) {
        let Some(list) = on.get_mut(*f as usize) else { continue };
        let [a, b, c] = tri(t);
        let n = (b - a).cross(c - a);
        let area = n.length() * 0.5;
        if area > 0.0 {
            list.push(((a + b + c) / 3.0, n.normalize(), area));
        }
    }
    let mut faces = Vec::with_capacity(nf);
    for list in &on {
        // The triangle nearest the face's area-weighted middle: a point on the face, not
        // across a hole in it.
        let total: f64 = list.iter().map(|x| x.2).sum();
        if !(total > 0.0) {
            continue;
        }
        let centre = list.iter().fold(DVec3::ZERO, |s, x| s + x.0 * x.2) / total;
        if let Some(best) = list.iter().min_by(|p, q| (p.0 - centre).length_squared().total_cmp(&(q.0 - centre).length_squared())) {
            faces.push(best.0);
        }
    }
    let mut edges = Vec::new();
    for (i, pts) in mesh.edges.iter().enumerate() {
        if mesh.seams.get(i).copied().unwrap_or(false) {
            continue;
        }
        let Some(sides) = mesh.edge_faces.get(i) else { continue };
        let [f0, f1] = match sides.as_slice() {
            [a, b] if a != b => [*a as usize, *b as usize],
            _ => continue,
        };
        let poly: Vec<DVec3> = pts.iter().map(|p| dv3(*p)).collect();
        let Some(mid) = polyline_mid(&poly) else { continue };
        let tangent = edge_tangent(&poly, mid);
        let (Some(a), Some(b)) = (on.get(f0).and_then(|l| near_triangle(l, mid)), on.get(f1).and_then(|l| near_triangle(l, mid))) else { continue };
        // Each face's own normal at the edge where its surface says it exactly — a facet's
        // beside a round leans by half the facet's turn — the way the facets face.
        let a = (a.0, surface_normal(body, f0, mid).map_or(a.1, |n| if n.dot(a.1) < 0.0 { -n } else { n }));
        let b = (b.0, surface_normal(body, f1, mid).map_or(b.1, |n| if n.dot(b.1) < 0.0 { -n } else { n }));
        if a.1.dot(b.1) > SMOOTH_COS {
            continue;
        }
        // Into each face from the edge, square to it.
        let into = |c: DVec3| {
            let w = c - mid;
            (w - tangent * w.dot(tangent)).normalize_or_zero()
        };
        let (wa, wb) = (into(a.0), into(b.0));
        // Convex: each face falls away below the other's outward normal.
        let bend = wa.dot(b.1) + wb.dot(a.1);
        edges.push(SharpEdge { mid, normals: [a.1, b.1], convex: bend < 0.0 });
    }
    Ok(Read { faces, edges })
}

/// The unit normal of a face's surface at a point on (or near) it, up to its sign, where the
/// surface is one of the analytic kinds.
fn surface_normal(body: &Body, face: usize, p: DVec3) -> Option<DVec3> {
    let n = match kernel::face_surface(body, face)? {
        FaceSurface::Plane { normal, .. } => dv3(normal.into()),
        FaceSurface::Cylinder { origin, axis, .. } => {
            let (o, a) = (dv3(origin), dv3(axis.into()));
            let w = p - o;
            w - a * w.dot(a)
        }
        FaceSurface::Sphere { center, .. } => p - dv3(center),
        FaceSurface::Cone { apex, axis, half_angle } => {
            let (v, a) = (dv3(apex), dv3(axis.into()));
            let w = p - v;
            let q = (w - a * w.dot(a)).try_normalize()?;
            q * half_angle.cos() - a * half_angle.sin()
        }
        FaceSurface::Torus { center, axis, major, .. } => {
            let (c, a) = (dv3(center), dv3(axis.into()));
            let w = p - c;
            let radial = (w - a * w.dot(a)).try_normalize()?;
            p - (c + radial * major)
        }
    };
    n.try_normalize()
}

/// The triangle of a face nearest a point: its middle and its normal.
fn near_triangle(list: &[(DVec3, DVec3, f64)], p: DVec3) -> Option<(DVec3, DVec3)> {
    list.iter().min_by(|x, y| (x.0 - p).length_squared().total_cmp(&(y.0 - p).length_squared())).map(|x| (x.0, x.1))
}

/// The point halfway along a polyline.
fn polyline_mid(pts: &[DVec3]) -> Option<DVec3> {
    let total: f64 = pts.windows(2).map(|w| (w[1] - w[0]).length()).sum();
    if !(total > 0.0) {
        return pts.first().copied();
    }
    let mut acc = 0.0;
    for w in pts.windows(2) {
        let l = (w[1] - w[0]).length();
        if acc + l >= total / 2.0 && l > 0.0 {
            return Some(w[0] + (w[1] - w[0]) * ((total / 2.0 - acc) / l));
        }
        acc += l;
    }
    pts.last().copied()
}

/// The direction of a polyline at the point of it nearest `p`.
fn edge_tangent(pts: &[DVec3], p: DVec3) -> DVec3 {
    let mut best: Option<(f64, DVec3)> = None;
    for w in pts.windows(2) {
        let d = w[1] - w[0];
        let t = if d.length_squared() > 0.0 { ((p - w[0]).dot(d) / d.length_squared()).clamp(0.0, 1.0) } else { 0.0 };
        let dist = (w[0] + d * t - p).length_squared();
        if best.is_none_or(|(x, _)| dist < x) {
            best = Some((dist, d.normalize_or_zero()));
        }
    }
    best.map_or(DVec3::X, |b| b.1)
}

/// A body's box as the kernel's bounding box reads it (OCCT's `Bnd_Box`): off a fine
/// tessellation, so a face that bulges past its edges — a sphere, a cylinder lying aslant —
/// counts where it reaches, which a box of the edges' control points does not; the box of the
/// outline where the body does not mesh.
pub(crate) fn true_box(body: &Body) -> solidcraft_geom::Aabb3 {
    let size = diagonal_of(body).max(1e-3);
    match body.tessellate((size * 2e-4).max(1e-5)) {
        Ok(m) if !m.positions.is_empty() => m.bounds(),
        _ => body.outline_bounds(),
    }
}

/// What is wrong with the offset `made` of `solid` by `distance` (negative inward), as a
/// sentence ends — `None` when nothing is (ScanRuler's `offsetFault`). A true offset has the
/// body's shells, grows or shrinks, and moves each side of the body's box by the distance or
/// more (a quarter of it is allowed: a side on an edge or a slope moves further).
pub fn offset_fault(solid: &Body, made: &Body, distance: f64) -> Option<String> {
    let after = volume(made).unwrap_or(0.0);
    if !(after > 1e-9) {
        return Some("it has no volume".into());
    }
    let before = volume(solid).unwrap_or(0.0);
    let grown = distance > 0.0;
    if if grown { !(after > before) } else { !(after < before) } {
        return Some(format!("it is no {} than the body", if grown { "larger" } else { "smaller" }));
    }
    let (outer, bx) = (true_box(solid), true_box(made));
    let moved = [outer.min.x - bx.min.x, outer.min.y - bx.min.y, outer.min.z - bx.min.z, bx.max.x - outer.max.x, bx.max.y - outer.max.y, bx.max.z - outer.max.z];
    let short = moved.iter().position(|m| if grown { *m < distance * 0.75 } else { *m > distance * 0.75 });
    if let Some(i) = short {
        let m = if grown { moved[i] } else { -moved[i] };
        let by = format!("{} mm", crate::js_number::js_string(crate::js_number::js_round(m.abs() * 100.0) / 100.0));
        let way = if grown { "out" } else { "in" };
        if m.abs() < 0.005 {
            return Some(format!("its box does not move {way} on the {} side", SIDES[i]));
        }
        return Some(if m < 0.0 { format!("its box moves {} by {by} on the {} side, not {way}", if grown { "in" } else { "out" }, SIDES[i]) } else { format!("its box moves {way} by only {by} on the {} side", SIDES[i]) });
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
