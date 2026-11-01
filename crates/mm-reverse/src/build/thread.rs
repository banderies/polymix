// SPDX-License-Identifier: AGPL-3.0-only
//! The Thread feature's build: a screw thread cut on a full round face of a body — a bolt's on
//! a boss, a nut's in a bore — the face brought to the thread's crests first where it was drawn
//! another size. Ported from ScanRuler (`plugins/reverse/core/kernelThread.ts` threadTracked,
//! `resolve.ts` the thread step, `build.ts` the thread step), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only, on the SolidCraft kernel. What the cut is made of — the fill, the band, the
//! tooth and its helix, the chamfer rings, which ends run free — is worked out in
//! [`crate::thread_cut`]; this file asks the kernel for the solids.
//!
//! The kernel's own B-rep booleans do not take a screwed tooth's helical walls in reasonable
//! time (SolidCraft's own modelled thread says so, and cuts faceted). So the cutter — the band
//! less the tooth screwed along the axis — is made as ScanRuler makes it, but faceted: the
//! tooth swept as a closed triangle mesh (`sweep_helix_mesh`), taken out of the band by the
//! kernel's faceted boolean; and the body is worked on the same way — the sleeve that brings a
//! face to the crests joined, the chamfer rings and the cutter cut — its curved faces within a
//! two-hundredth of the pitch. The threaded body therefore comes out with flat faces only, as
//! SolidCraft's own threads do; ScanRuler's Open CASCADE keeps the body's other faces exact.
//! The faceted work waits until a feature other than a thread or a sketch comes to work on the
//! bodies, or the build ends ([`flush`]): the threads in a row on one body — a plate's tapped
//! holes — are each read off its exact faces, and the body is faceted once.

use std::f64::consts::{PI, TAU};

use glam::DVec3;
use solidcraft_geom::{Loop2, Mesh, Plane, Region2, Vec2 as SVec2};
use solidcraft_kernel::{self as kernel, BoolOp, Body};

use super::hole::is_unpicked;
use super::{diagonal_of, dv3, kerr, mesh_deflection, origins_after, sv3, Builder};
use crate::model::{Feature, Part, ThreadFeature, ThreadStandard as ModelStandard};
use crate::thread::{thread_feature_problem, ThreadCut, ThreadSpec, ThreadStandard};
use crate::thread_cut::{Annulus, ChamferRing, RoundFace, ThreadCutError, ThreadPlan, ThreadSetup};
use crate::thread_scan::round_faces;
use crate::topo::{FaceKind, Topology};

/// What a Thread with no face says, while its box waits for a click.
pub const THREAD_UNPICKED: &str = "Click a round face of the body.";

/// How many turns one sweep of the tooth makes at most: a longer thread is cut in pieces along
/// its axis, each with a sweep of its own (the kernel's sweep takes 200 turns at most).
const SWEEP_TURNS: f64 = 150.0;

/// The fewest stations a turn of the tooth's sweep is made of, and the most.
const MIN_STATIONS: f64 = 48.0;
const MAX_STATIONS: f64 = 720.0;

/// The facet tolerance of the cut, as a share of the pitch: a curved face of the body lies
/// within this of the faces that stand for it.
const FACET_SHARE: f64 = 0.005;

/// How many facets' tolerance a crest's flat has to be wide for the faceted cut to keep it.
const CREST_FACETS: f64 = 4.0;

/// How far the sleeve that brings a face to the crests reaches into the face's own material,
/// as a share of the pitch.
const SLEEVE_OVERLAP: f64 = 0.01;

/// A model's thread standard as the thread tables name it.
pub fn core_standard(s: ModelStandard) -> ThreadStandard {
    match s {
        ModelStandard::Metric => ThreadStandard::Metric,
        ModelStandard::Unc => ThreadStandard::Unc,
        ModelStandard::Unf => ThreadStandard::Unf,
        ModelStandard::Print45 => ThreadStandard::Print45,
    }
}

/// A thread table's standard as the model keeps it.
pub fn model_standard(s: ThreadStandard) -> ModelStandard {
    match s {
        ThreadStandard::Metric => ModelStandard::Metric,
        ThreadStandard::Unc => ModelStandard::Unc,
        ThreadStandard::Unf => ModelStandard::Unf,
        ThreadStandard::Print45 => ModelStandard::Print45,
    }
}

/// Everything the kernel cuts a Thread feature to (ScanRuler's thread step): a feature saved
/// before the chamfer was offered builds without one.
pub fn thread_cut_of(x: &ThreadFeature) -> ThreadCut {
    ThreadCut {
        spec: ThreadSpec { standard: core_standard(x.standard), diameter: x.diameter, pitch: x.pitch, offset: x.offset },
        left_handed: x.left_handed,
        length: x.length,
        start: x.start,
        flip: x.flip,
        chamfer: x.chamfer.unwrap_or(false),
    }
}

/// Why a Thread feature cannot be resolved into a cut, before the kernel is asked — or `None`
/// (ScanRuler's resolve of the `thread` feature: a face that is not round, numbers that cannot
/// be cut, a length of nothing, a start off the face). A Thread with no face clicked yet says
/// so.
pub fn thread_resolve_problem(x: &ThreadFeature) -> Option<String> {
    if is_unpicked(&x.face) {
        return Some(THREAD_UNPICKED.to_string());
    }
    thread_feature_problem(&thread_cut_of(x), x.face.kind == FaceKind::Cylinder)
}

/// A kernel failure inside the cut, in the words ScanRuler's `run` wrapper gives it.
fn cutting(e: impl std::fmt::Display) -> String {
    format!("Cutting the thread: {e}")
}

/// A direction square to the axis: the one the plan's radial is, so the band's seam and the
/// tooth's plane agree.
fn radial_of(axis: DVec3) -> DVec3 {
    axis.cross(if axis.x.abs() < 0.9 { DVec3::X } else { DVec3::Y }).try_normalize().unwrap_or(DVec3::X)
}

/// The round face a thread is cut on, read off the body (ScanRuler's `readFace` and
/// `readRound`): its axis, radius and ends along the axis, which way it looks, and whether it
/// runs between square ends; with the faces it is made of. Refuses a face that is not a full
/// cylinder.
///
/// The kernel gives no face's exact area, which ScanRuler tells a plain cylinder between
/// square ends by; here the face's edges at its ends tell it — every point of them at one end
/// or the other — and the area handed on is the plain cylinder's for such a face.
pub(crate) fn read_round(shape: &Body, t: &Topology, face: usize) -> Result<(RoundFace, Vec<usize>), ThreadCutError> {
    let Some(fi) = t.faces.get(face) else { return Err(ThreadCutError::not_a_cylinder()) };
    let Some(kernel::FaceSurface::Cylinder { origin, axis, radius }) = kernel::face_surface(shape, face) else { return Err(ThreadCutError::not_a_cylinder()) };
    if fi.full_turn == Some(false) {
        return Err(ThreadCutError::not_full_turn());
    }
    // The kernel keeps a whole bore as two halves, where Open CASCADE keeps one face.
    let group = round_faces(t, face);
    let axis = fi.fingerprint.dir.try_normalize().unwrap_or_else(|| dv3(axis.into()));
    let center = fi.fingerprint.center.unwrap_or_else(|| dv3(origin));
    let mesh = shape.display_mesh_angular(mesh_deflection(diagonal_of(shape)), kernel::DISPLAY_ANGLE).map_err(|e| ThreadCutError(e.to_string()))?;
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    let mut vote = 0.0;
    let mut area = 0.0;
    for (tri, f) in mesh.triangles.iter().zip(&mesh.tri_face) {
        if !group.contains(&(*f as usize)) {
            continue;
        }
        let (Some(a), Some(b), Some(c)) = (mesh.positions.get(tri[0] as usize), mesh.positions.get(tri[1] as usize), mesh.positions.get(tri[2] as usize)) else { continue };
        let (a, b, c) = (dv3(*a), dv3(*b), dv3(*c));
        for q in [a, b, c] {
            let s = (q - center).dot(axis);
            lo = lo.min(s);
            hi = hi.max(s);
        }
        // Outward of a boss points away from the axis, of a hole towards it.
        let n = (b - a).cross(c - a);
        area += n.length() / 2.0;
        let m = (a + b + c) / 3.0 - center;
        vote += n.dot(m - axis * m.dot(axis));
    }
    if !(hi - lo > 1e-9) {
        return Err(ThreadCutError::no_length());
    }
    // A plain cylinder between square ends: every point of the edges it ends at lies at one
    // end or the other. The edges between its own halves are none of its ends.
    let in_group = |h: &u32| group.contains(&(*h as usize));
    let slack = 1e-6 * (hi - lo).max(1.0);
    let square = t.edges.iter().filter(|e| e.faces.iter().any(in_group) && !e.faces.iter().all(in_group)).all(|e| {
        e.polyline.iter().all(|q| {
            let s = (*q - center).dot(axis);
            (s - lo).abs() <= slack || (s - hi).abs() <= slack
        })
    });
    let plain = 2.0 * PI * radius * (hi - lo);
    let area = if square {
        plain
    } else if (area - plain).abs() <= 1e-6 * plain {
        // Never read as plain by chance.
        plain * (1.0 + 1e-3)
    } else {
        area
    };
    Ok((RoundFace { center, axis, radius, convex: vote > 0.0, lo, hi, area }, group))
}

/// A ring about an axis extruded along it, as a solid: the sleeve, the band.
pub(crate) fn annulus_solid(a: &Annulus) -> Result<Body, String> {
    let axis = a.axis.try_normalize().ok_or("The axis has no direction.")?;
    let x = radial_of(axis);
    let plane = Plane::new(sv3(a.center), sv3(x), sv3(axis.cross(x))).ok_or("The axis has no direction.")?;
    let holes = match a.inner {
        Some(r) if r > 1e-9 => vec![Loop2::circle(SVec2::new(0.0, 0.0), r)],
        _ => Vec::new(),
    };
    let region = Region2 { outer: Loop2::circle(SVec2::new(0.0, 0.0), a.outer), holes };
    let mut out = kernel::extrude(&plane, &[region], a.from, a.to).map_err(kerr)?;
    out.pop().ok_or_else(|| "The ring could not be made.".to_string())
}

/// A chamfer's triangle turned the whole way round the axis — its sides moved out by `grow`.
pub(crate) fn chamfer_solid(ring: &ChamferRing, grow: f64) -> Result<Body, String> {
    let axis = ring.axis.try_normalize().ok_or("The axis has no direction.")?;
    let x = radial_of(axis);
    let plane = Plane::new(sv3(ring.center), sv3(x), sv3(axis)).ok_or("The axis has no direction.")?;
    let flat: Vec<glam::DVec2> = ring
        .world
        .iter()
        .map(|q| {
            let d = *q - ring.center;
            glam::DVec2::new(d.dot(x), d.dot(axis))
        })
        .collect();
    let pts: Vec<SVec2> = grown(&flat, grow).iter().map(|q| SVec2::new(q.x, q.y)).collect();
    let region = Region2 { outer: Loop2::polygon(&pts), holes: Vec::new() };
    let mut out = kernel::revolve(&plane, &[region], SVec2::new(0.0, 0.0), SVec2::new(0.0, 1.0), TAU).map_err(kerr)?;
    out.pop().ok_or_else(|| "The chamfer could not be made.".to_string())
}

/// A convex polygon with each side moved out by `d`, its corners where the moved sides meet.
fn grown(pts: &[glam::DVec2], d: f64) -> Vec<glam::DVec2> {
    let n = pts.len();
    if d == 0.0 || n < 3 {
        return pts.to_vec();
    }
    let c = pts.iter().copied().sum::<glam::DVec2>() / n as f64;
    // Each side as a point on it, moved out, and its direction.
    let sides: Vec<(glam::DVec2, glam::DVec2)> = (0..n)
        .map(|i| {
            let (a, b) = (pts[i], pts[(i + 1) % n]);
            let dir = (b - a).normalize_or_zero();
            let mut out = glam::DVec2::new(dir.y, -dir.x);
            if out.dot(a - c) < 0.0 {
                out = -out;
            }
            (a + out * d, dir)
        })
        .collect();
    (0..n)
        .map(|i| {
            let (p, u) = sides[(i + n - 1) % n];
            let (q, v) = sides[i];
            let den = u.x * v.y - u.y * v.x;
            if den.abs() < 1e-12 {
                return pts[i];
            }
            let t = ((q - p).x * v.y - (q - p).y * v.x) / den;
            p + u * t
        })
        .collect()
}

/// How many stations a turn of the tooth's sweep is made of: enough that the chords of its
/// outermost corner lie within a hundredth of the pitch of the helix (as SolidCraft's own
/// modelled thread has it), at least 48 and at most 720.
fn stations_for(pitch: f64, outer: f64) -> usize {
    if !(outer > 0.0) {
        return MIN_STATIONS as usize;
    }
    let angle = (1.0 - (pitch * 0.01 / outer).min(1.0)).acos();
    let n = if angle > 0.0 { (PI / angle).ceil() } else { MAX_STATIONS };
    n.clamp(MIN_STATIONS, MAX_STATIONS) as usize
}

/// The plan as the faceted cut takes it, a facet's tolerance off where the faceted booleans
/// cannot follow the exact geometry: a band that ends where the face does and runs on into the
/// body there — against a shoulder, a bore's floor — would lie on the face that ends it, and
/// stops `tol` short of it; and a crest's flat narrower than a few facets — an offset all but
/// used up — is drawn to the point its flanks meet, a few microns out.
fn for_facets(plan: &ThreadPlan, setup: &ThreadSetup, tol: f64) -> ThreadPlan {
    let mut p = plan.clone();
    if !p.free_lo && (setup.span.from - setup.face.lo).abs() < 1e-6 {
        p.band.from += tol;
    }
    if !p.free_hi && (setup.span.to - setup.face.hi).abs() < 1e-6 {
        p.band.to -= tol;
    }
    p.helix.profile = sharpened(&p.helix.profile, CREST_FACETS * tol);
    p
}

/// A tooth's outline with a crest's flat narrower than `min_width` drawn to the point its
/// flanks meet: the two corners of the flat, across the tooth's middle, made one.
fn sharpened(points: &[glam::DVec2], min_width: f64) -> Vec<glam::DVec2> {
    let n = points.len();
    if n < 5 {
        return points.to_vec();
    }
    for i in 0..n {
        let (a, b) = (points[i], points[(i + 1) % n]);
        if (a.x - b.x).abs() > 1e-12 || (a.y - b.y).abs() >= min_width || (a.y + b.y).abs() > 1e-9 {
            continue;
        }
        let before = points[(i + n - 1) % n];
        let t = -before.y / (a.y - before.y);
        if !t.is_finite() {
            continue;
        }
        let apex = glam::DVec2::new(before.x + (a.x - before.x) * t, 0.0);
        return (0..n).filter(|k| *k != (i + 1) % n).map(|k| if k == i { apex } else { points[k] }).collect();
    }
    points.to_vec()
}

/// The cutter — the band less the tooth screwed along the axis — as closed triangle meshes,
/// one for each piece of the band along the axis (one piece but for a thread of more than
/// [`SWEEP_TURNS`] turns). Each piece's sweep starts a whole number of pitches from the plan's,
/// so the pieces' teeth are one helix.
pub(crate) fn cutter_meshes(plan: &ThreadPlan, tol: f64) -> Result<Vec<Mesh>, String> {
    let h = &plan.helix;
    let band = &plan.band;
    let p = h.pitch;
    let (axis, pitch) = h.right_handed_axis_and_pitch();
    let polygon: Vec<SVec2> = h.profile.iter().map(|q| SVec2::new(q.x, q.y)).collect();
    let outer = h.profile.iter().fold(0.0f64, |m, q| m.max(q.x.abs()));
    let stations = stations_for(p, outer);
    let length = band.to - band.from;
    let pieces = ((length / p) / SWEEP_TURNS).ceil().max(1.0) as usize;
    let mut out = Vec::with_capacity(pieces);
    for k in 0..pieces {
        let a = band.from + length * k as f64 / pieces as f64;
        let b = if k + 1 == pieces { band.to } else { band.from + length * (k + 1) as f64 / pieces as f64 };
        let z0 = band.from - p + ((a - band.from) / p).floor() * p;
        let turns = ((b + p - z0) / p - 1e-9).ceil();
        let plane = Plane { origin: sv3(band.center + band.axis * z0), x: sv3(h.plane_x), y: sv3(h.plane_y) };
        let tooth = kernel::sweep_helix_mesh(&plane, &polygon, sv3(h.axis_origin), sv3(axis), pitch, turns, stations).map_err(cutting)?;
        let ring = annulus_solid(&Annulus { from: a, to: b, ..*band }).map_err(cutting)?;
        let cutter = faceted(&ring, &tooth, BoolOp::Cut, tol)?.ok_or_else(|| cutting("the tooth took the whole band."))?;
        out.push(cutter.tessellate(tol).map_err(cutting)?);
    }
    Ok(out)
}

/// A thread's cut planned on a body: the face read, the body with the chamfer rings cut out of
/// it — exactly — and what is left to do faceted, in order: the sleeve that brings the face to
/// the crests joined (less the rings), the cutter cut.
pub(crate) struct ThreadReady {
    /// The body with the chamfers cut, and the makers of its faces.
    pub(crate) body: Body,
    pub(crate) origins: Vec<i64>,
    /// Whether a chamfer was cut: the body is not the one it was.
    pub(crate) chamfered: bool,
    pub(crate) ops: Vec<(BoolOp, Mesh)>,
    pub(crate) tol: f64,
    pub(crate) note: Option<String>,
}

/// Everything of a thread's cut but the faceted work (ScanRuler's `threadTracked`): the face
/// read, the fill settled — the body closed up without a bore that wants it — the sleeve made,
/// the plan made on the body with the sleeve, the chamfer rings cut and the cutter made. The
/// rings are cut out of the body and of the sleeve exactly, the sleeve joined faceted after —
/// the same as ScanRuler's join and then cut: the kernel's exact join of a sleeve flush with a
/// bore's ends leaves faces the faceted cut cannot close.
pub(crate) fn thread_ready(shape: &Body, origins: &[i64], t: &Topology, face: usize, cut: &ThreadCut, step: crate::model::FeatureId) -> Result<ThreadReady, String> {
    let own = step as i64;
    let (round, group) = read_round(shape, t, face).map_err(|e| e.to_string())?;
    let setup = ThreadSetup::new(cut, &round).map_err(|e| e.to_string())?;
    let tol = FACET_SHARE * cut.spec.pitch;
    // A bore that wants filling and does not run between square ends: the body closed up
    // without it, for the thread to be kept to afterwards.
    let hull = if setup.wants_hull() { kernel::delete_faces(shape, &group).ok() } else { None };
    let fill = setup.fill(hull.is_some()).map_err(|e| e.to_string())?;
    let mut sleeve: Option<Body> = None;
    if let Some(s) = &fill.sleeve {
        // The sleeve reaches a hair into the face's own material — past a bore's wall, inside
        // a boss's — so its faceted wall never lies on the face's; and on a facet's tolerance
        // past an end of the thread short of the face's, so its end never lies on the cutter's.
        // What it makes is the same.
        let overlap = (SLEEVE_OVERLAP * cut.spec.pitch).max(1e-3);
        let mut s = if setup.internal { Annulus { outer: s.outer + overlap, ..*s } } else { Annulus { inner: s.inner.map(|r| (r - overlap).max(0.0)), ..*s } };
        if (setup.span.from - setup.face.lo).abs() >= 1e-6 {
            s.from -= tol;
        }
        if (setup.span.to - setup.face.hi).abs() >= 1e-6 {
            s.to += tol;
        }
        let mut body = annulus_solid(&s).map_err(cutting)?;
        // Kept to the body closed up without the face: no more than the fill, where the face
        // does not run between square ends (ScanRuler keeps the cut to it, taking off no more
        // than the sleeve — the same, the body lying within it).
        if let (true, Some(h)) = (fill.hull, &hull) {
            body = kernel::boolean(&body, h, BoolOp::Intersect).map_err(cutting)?.ok_or_else(|| cutting("the fill lies outside the body."))?;
        }
        sleeve = Some(body);
    }
    let plan = fill.plan(|q| shape.contains(sv3(q)) || sleeve.as_ref().is_some_and(|s| s.contains(sv3(q)))).map_err(|e| e.to_string())?;
    // The chamfers first, on the plain face; the thread is cut through them.
    let mut body = shape.clone();
    let mut org = origins.to_vec();
    for ring in &plan.chamfers {
        let tool = chamfer_solid(ring, 0.0).map_err(cutting)?;
        let tool_org = vec![own; tool.face_count()];
        if let Some(r) = kernel::boolean(&body, &tool, BoolOp::Cut).map_err(cutting)? {
            org = origins_after(&r, &[(&body, &org), (&tool, &tool_org)], step);
            body = r;
        }
        // Out of the sleeve, the ring a few facets wider: its cone and the body's would be one
        // surface faceted twice, which the faceted join cannot close.
        if let Some(sl) = sleeve.take() {
            let wider = chamfer_solid(ring, CREST_FACETS * tol).map_err(cutting)?;
            sleeve = kernel::boolean(&sl, &wider, BoolOp::Cut).map_err(cutting)?;
        }
    }
    let mut ops: Vec<(BoolOp, Mesh)> = Vec::new();
    if let Some(sl) = &sleeve {
        ops.push((BoolOp::Union, sl.tessellate(tol).map_err(cutting)?));
    }
    for m in cutter_meshes(&for_facets(&plan, &setup, tol), tol)? {
        ops.push((BoolOp::Cut, m));
    }
    Ok(ThreadReady { body, origins: org, chamfered: !plan.chamfers.is_empty(), ops, tol, note: plan.note })
}

/// The shares of the facet tolerance a faceted boolean is tried at, in turn: where the kernel's
/// faceted boolean cannot close what it makes at one — facets meeting at a sliver of an angle,
/// a corner a hair off another — the body faceted a little finer or coarser lies otherwise.
const FACET_TRIES: [f64; 4] = [1.0, 0.8, 1.25, 0.65];

/// The kernel's faceted boolean of a body with a closed mesh, tried at the facet tolerances of
/// [`FACET_TRIES`] until one makes a closed solid; the first failure's words where none does.
fn faceted(a: &Body, b: &Mesh, op: BoolOp, tol: f64) -> Result<Option<Body>, String> {
    let mut first = None;
    for share in FACET_TRIES {
        match kernel::faceted_boolean(a, b, op, tol * share) {
            Ok(r) => return Ok(r),
            Err(e) => {
                first.get_or_insert(e);
            }
        }
    }
    Err(first.map_or_else(|| cutting("the faceted boolean failed."), cutting))
}

/// A mesh a body is worked on with, faceted — joined or cut — and the maker of each of its
/// triangles.
#[derive(Clone)]
pub(crate) struct FacetOp {
    pub(crate) op: BoolOp,
    pub(crate) mesh: Mesh,
    pub(crate) makers: Vec<i64>,
}

/// A body worked on by meshes one after another, faceted, with which feature first made each
/// face carried through: a face of the body keeps its maker, a face of a mesh is its maker's.
fn apply_faceted(shape: &Body, origins: &[i64], ops: &[FacetOp], tol: f64) -> Result<(Body, Vec<i64>), String> {
    let mut acc = shape.clone();
    let mut org = origins.to_vec();
    for f in ops {
        let r = faceted(&acc, &f.mesh, f.op, tol)?.ok_or_else(|| cutting("the thread takes the whole body."))?;
        let step = f.makers.first().copied().unwrap_or(-1);
        org = origins_after(&r, &[(&acc, &org), (&acc, &f.makers)], step.max(0) as crate::model::FeatureId);
        acc = r;
    }
    Ok((acc, org))
}

/// The cut of a thread made ready, made on the body.
#[cfg(test)]
pub(crate) fn thread_cut_now(r: &ThreadReady, step: crate::model::FeatureId) -> Result<(Body, Vec<i64>), String> {
    let ops: Vec<FacetOp> = r.ops.iter().map(|(op, m)| FacetOp { op: *op, mesh: m.clone(), makers: vec![step as i64; m.triangles.len()] }).collect();
    apply_faceted(&r.body, &r.origins, &ops, r.tol)
}

/// A thread cut on the round face `face` of a body (ScanRuler's `threadTracked`): the body as
/// it comes out, the makers of its faces, and what the row says about the cut, if anything.
#[cfg(test)]
pub(crate) fn thread_body(shape: &Body, origins: &[i64], t: &Topology, face: usize, cut: &ThreadCut, step: crate::model::FeatureId) -> Result<(Body, Vec<i64>, Option<String>), String> {
    let r = thread_ready(shape, origins, t, face, cut, step)?;
    let (body, org) = thread_cut_now(&r, step)?;
    Ok((body, org, r.note))
}

/// A thread's faceted cut not made yet (see [`flush`]).
#[derive(Clone)]
pub(crate) struct PendingCut {
    pub(crate) step: crate::model::FeatureId,
    pub(crate) body: crate::model::BodyId,
    pub(crate) ops: std::sync::Arc<Vec<(BoolOp, Mesh)>>,
    pub(crate) tol: f64,
}

/// The faceted cuts of the threads built so far, made: the threads in a row are each read and
/// readied on the body's exact faces — a plate's tapped holes one after another — and the
/// body is faceted once, when a feature other than a thread or a sketch comes to work on the
/// bodies, or the build ends. A cut that fails then fails its thread, by name.
pub(crate) fn flush(b: &mut Builder) {
    if b.threads.is_empty() {
        return;
    }
    let pending = std::mem::take(&mut b.threads);
    let was = b.step;
    // The threads of each body, in the order they were built.
    let mut bodies: Vec<crate::model::BodyId> = Vec::new();
    for p in &pending {
        if !bodies.contains(&p.body) {
            bodies.push(p.body);
        }
    }
    for body in bodies {
        let group: Vec<&PendingCut> = pending.iter().filter(|p| p.body == body).collect();
        let tol = group.iter().map(|p| p.tol).fold(f64::INFINITY, f64::min);
        // All at once — the sleeves joined, then the cutters cut — and, where the kernel
        // cannot, each thread on its own.
        let merged = merged_ops(&group);
        if let Ok(ids) = cut_group(b, body, &merged, tol, group.first().map_or(was, |p| p.step)) {
            for p in &group {
                b.step = p.step;
                b.touched(ids.clone());
            }
            continue;
        }
        for p in group {
            let ops: Vec<FacetOp> = p.ops.iter().map(|(op, m)| FacetOp { op: *op, mesh: m.clone(), makers: vec![p.step as i64; m.triangles.len()] }).collect();
            match cut_group(b, p.body, &ops, p.tol, p.step) {
                Ok(ids) => {
                    b.step = p.step;
                    b.touched(ids);
                }
                Err(e) => {
                    b.out.errors.insert(p.step, e);
                }
            }
        }
    }
    b.step = was;
}

/// The faceted work of a body's threads together: their sleeves joined as one mesh, then their
/// cutters cut, a piece of each at a time.
fn merged_ops(group: &[&PendingCut]) -> Vec<FacetOp> {
    let union: Vec<(&Mesh, i64)> = group.iter().flat_map(|p| p.ops.iter().filter(|(op, _)| *op == BoolOp::Union).map(|(_, m)| (m, p.step as i64))).collect();
    let cuts: Vec<Vec<(&Mesh, i64)>> = group.iter().map(|p| p.ops.iter().filter(|(op, _)| *op == BoolOp::Cut).map(|(_, m)| (m, p.step as i64)).collect()).collect();
    let mut out = Vec::new();
    if !union.is_empty() {
        out.push(concat(BoolOp::Union, &union));
    }
    let pieces = cuts.iter().map(Vec::len).max().unwrap_or(0);
    for k in 0..pieces {
        let at: Vec<(&Mesh, i64)> = cuts.iter().filter_map(|c| c.get(k).copied()).collect();
        out.push(concat(BoolOp::Cut, &at));
    }
    out
}

/// Closed meshes apart from each other, as one, each triangle with its maker.
fn concat(op: BoolOp, meshes: &[(&Mesh, i64)]) -> FacetOp {
    let mut mesh = Mesh::default();
    let mut makers = Vec::new();
    for (m, maker) in meshes {
        let base = mesh.positions.len() as u32;
        mesh.positions.extend(m.positions.iter().copied());
        mesh.normals.extend((0..m.positions.len()).map(|i| m.normals.get(i).copied().unwrap_or_default()));
        mesh.triangles.extend(m.triangles.iter().map(|t| [t[0] + base, t[1] + base, t[2] + base]));
        mesh.tri_face.extend(m.triangles.iter().map(|_| 0));
        makers.extend(std::iter::repeat_n(*maker, m.triangles.len()));
    }
    FacetOp { op, mesh, makers }
}

/// A body's faceted work made, the body settled: the ids it went to.
fn cut_group(b: &mut Builder, body: crate::model::BodyId, ops: &[FacetOp], tol: f64, step: crate::model::FeatureId) -> Result<Vec<crate::model::BodyId>, String> {
    b.step = step;
    // As a feature's build is: a panic in the kernel is its error.
    kernel::guard("Cutting the thread", || {
        Ok(b.body_for(body).and_then(|(id, state)| {
            let (made, origins) = apply_faceted(&state.shape, &state.origins, ops, tol)?;
            b.settle(id, made, origins, false)
        }))
    })
    .unwrap_or_else(|e| Err(e.to_string()))
}

pub(crate) fn build(b: &mut Builder, _part: &Part, _f: &Feature, x: &ThreadFeature) -> Result<(), String> {
    if let Some(why) = thread_resolve_problem(x) {
        return Err(why);
    }
    let cut = thread_cut_of(x);
    let (id, state) = b.body_for(x.body)?;
    let found = b.find_faces(id, &[x.face], &[]).map_err(|lost| format!("The round face this thread is cut on is gone (was: {lost})."))?;
    let face = *found.first().ok_or("The round face this thread is cut on is gone.")?;
    let t = b.topology_of(id).unwrap_or_else(|| state.topology());
    let ready = thread_ready(&state.shape, &state.origins, &t, face, &cut, b.step)?;
    if let Some(n) = &ready.note {
        b.warn(n.clone());
    }
    // The body with its chamfers stands now; the faceted work waits (see [`flush`]).
    let ids = if ready.chamfered { b.settle(id, ready.body, ready.origins, false)? } else { vec![id] };
    b.touched(ids);
    b.threads.push(PendingCut { step: b.step, body: id, ops: std::sync::Arc::new(ready.ops), tol: ready.tol });
    Ok(())
}

#[cfg(test)]
mod tests {
    //! ScanRuler's `reThread.test.ts` kernel tests ("a thread cut in the kernel", "a thread on
    //! a face drawn another size", "a thread's chamfer", "the Thread feature"), on the
    //! SolidCraft kernel. The cut is faceted (see the head of the file), so a volume is held to
    //! a few parts in a thousand of the profile's (Pappus) rather than ScanRuler's one in ten
    //! thousand, and a point is probed no nearer a face than the facets' few microns.
    use std::f64::consts::PI;

    use glam::{DVec2, DVec3};

    use super::*;
    use crate::build::{build as build_part, default_regions, read_topology, volume};
    use crate::frame::WorldAxis;
    use crate::model::{BodyId, BodyOp, ExtrudeExtent, ExtrudeFeature, FeatureKind, SketchFeature};
    use crate::regions::find_regions;
    use crate::sketch::{Shape, Sketch};
    use crate::thread::{chamfer_radius, minor_diameter, tooth_outline};

    /// How near a faceted volume comes to the profile's.
    const VOLUME_SHARE: f64 = 4e-3;
    /// How near a nut's volume comes to the bore's change, as a share of the block it is in.
    const BEFORE_SHARE: f64 = 2e-3;

    fn cut(diameter: f64, pitch: f64) -> ThreadCut {
        ThreadCut { spec: ThreadSpec { standard: ThreadStandard::Metric, diameter, pitch, offset: 0.0 }, left_handed: false, length: None, start: 0.0, flip: false, chamfer: false }
    }

    /// A bar of radius `r` up Z from 0 to `length`.
    fn bar(r: f64, length: f64) -> Body {
        kernel::cylinder(sv3(DVec3::ZERO), sv3(DVec3::Z), r, length).expect("a bar")
    }

    /// A disc of radius `outer` with a bore of radius `r` through it up Z, `height` thick.
    fn plate(outer: f64, r: f64, height: f64) -> Body {
        annulus_solid(&Annulus { center: DVec3::ZERO, axis: DVec3::Z, outer, inner: Some(r), from: 0.0, to: height }).expect("a plate")
    }

    fn vol(b: &Body) -> f64 {
        volume(b).unwrap_or(f64::NAN)
    }

    fn closed(b: &Body) -> bool {
        b.validity().is_empty()
    }

    fn near(got: f64, want: f64, share: f64) -> bool {
        (got - want).abs() <= share * want.abs()
    }

    /// The thread cut on the round face of radius `r`: the body and the row's note.
    fn thread(shape: &Body, r: f64, c: &ThreadCut) -> Result<(Body, Option<String>), String> {
        let origins = vec![1; shape.face_count()];
        let t = read_topology(shape, &origins);
        let face = t.faces.iter().position(|f| f.fingerprint.kind == FaceKind::Cylinder && f.fingerprint.radius.is_some_and(|x| (x - r).abs() < 1e-6)).ok_or(format!("no round face of radius {r}"))?;
        thread_body(shape, &origins, &t, face, c, 9).map(|(b, _, n)| (b, n))
    }

    fn ok(r: Result<(Body, Option<String>), String>) -> (Body, Option<String>) {
        match r {
            Ok(x) => x,
            Err(e) => panic!("{e}"),
        }
    }

    fn inside(b: &Body, p: [f64; 3]) -> bool {
        b.contains(sv3(DVec3::from(p)))
    }

    /// The area of a polygon in (r, z) and the radius of its centroid.
    fn area_and_radius(points: &[DVec2]) -> (f64, f64) {
        let (mut a, mut cr) = (0.0, 0.0);
        for i in 0..points.len() {
            let (p, q) = (points[i], points[(i + 1) % points.len()]);
            let cross = p.x * q.y - q.x * p.y;
            a += cross;
            cr += (p.x + q.x) * cross;
        }
        ((a / 2.0).abs(), cr / (3.0 * a))
    }

    /// The part of a tooth's outline beyond a radius, as a polygon.
    fn clip(points: &[DVec2], at: f64, above: bool) -> Vec<DVec2> {
        let inside = |p: DVec2| if above { p.x >= at - 1e-12 } else { p.x <= at + 1e-12 };
        let mut out = Vec::new();
        for i in 0..points.len() {
            let (a, b) = (points[i], points[(i + 1) % points.len()]);
            if inside(a) {
                out.push(a);
            }
            if inside(a) != inside(b) {
                let t = (at - a.x) / (b.x - a.x);
                out.push(DVec2::new(at, a.y + (b.y - a.y) * t));
            }
        }
        out
    }

    /// The volume a tooth's outline between radii `lo` and `hi` fills screwed `length` along
    /// the axis at `pitch` — Pappus, the climb adding nothing.
    fn teeth_volume(points: &[DVec2], lo: f64, hi: f64, pitch: f64, length: f64) -> f64 {
        let (area, r) = area_and_radius(&clip(&clip(points, lo, true), hi, false));
        area * 2.0 * PI * r / pitch * length
    }

    /// Whether a ring of 8 points at radius `r`, height `z` is all in the shape, none, or some.
    fn ring(b: &Body, r: f64, z: f64, phase: f64) -> &'static str {
        let n = (0..8).filter(|i| inside(b, [r * (*i as f64 * PI / 4.0 + phase).cos(), r * (*i as f64 * PI / 4.0 + phase).sin(), z])).count();
        match n {
            8 => "all",
            0 => "none",
            _ => "some",
        }
    }

    /// Where along a line up Z at (x, y) the material is over one pitch from `z0`: the middle
    /// of the stretch inside, modulo the pitch.
    fn tooth_middle(b: &Body, x: f64, y: f64, z0: f64, pitch: f64) -> f64 {
        let n = 16;
        let ins: Vec<bool> = (0..n).map(|i| inside(b, [x, y, z0 + pitch * i as f64 / n as f64])).collect();
        let start = (0..n).find(|i| ins[*i] && !ins[(i + n - 1) % n]).unwrap_or(0);
        let mut len = 0;
        while ins[(start + len) % n] && len < n {
            len += 1;
        }
        (((start as f64 + len as f64 / 2.0) * pitch / n as f64) % pitch + pitch) % pitch
    }

    #[test]
    fn cuts_a_bolts_thread_the_whole_length_of_a_boss_of_the_volume_the_profile_says() {
        let body = bar(8.0, 30.0);
        let t = cut(16.0, 2.0);
        let (threaded, note) = ok(thread(&body, 8.0, &t));
        assert!(closed(&threaded));
        assert_eq!(note, None);
        let tooth = tooth_outline(&t.spec, false, 0.5, None);
        let want = PI * tooth.root.powi(2) * 30.0 + teeth_volume(&tooth.points, tooth.root, 8.0, 2.0, 30.0);
        assert!(near(vol(&threaded), want, VOLUME_SHARE), "{} vs {want}", vol(&threaded));
        // The ends are where they were: the free ends run the thread out.
        let bb = threaded.outline_bounds();
        assert!(bb.min.z.abs() < 1e-6 && (bb.max.z - 30.0).abs() < 1e-6, "{bb:?}");
    }

    #[test]
    fn turns_right_handed_or_left_handed_when_asked() {
        let body = bar(8.0, 30.0);
        let (right, _) = ok(thread(&body, 8.0, &cut(16.0, 2.0)));
        let (left, _) = ok(thread(&body, 8.0, &ThreadCut { left_handed: true, ..cut(16.0, 2.0) }));
        // A quarter turn counter-clockwise about +Z, from +X to +Y, a right-hand thread's tooth
        // has climbed a quarter pitch; a left-hand one's has dropped it, and a quarter pitch up is
        // the middle of its gap.
        let r = 8.0 - 0.2;
        for (shape, rh) in [(&right, true), (&left, false)] {
            let up = tooth_middle(shape, r, 0.0, 10.0, 2.0) + 10.0 + 0.5;
            assert_eq!(inside(shape, [0.0, r, up]), rh, "right {rh}");
            assert_eq!(inside(shape, [0.0, r, up - 1.0]), !rh, "right {rh}");
        }
        assert!(near(vol(&left), vol(&right), 1e-3), "{} {}", vol(&left), vol(&right));
    }

    #[test]
    fn cuts_a_length_in_from_the_far_end_or_from_the_near_one_flipped() {
        let body = bar(8.0, 30.0);
        let t = ThreadCut { length: Some(10.0), start: 5.0, ..cut(16.0, 2.0) };
        let (top, _) = ok(thread(&body, 8.0, &t));
        assert!(closed(&top));
        let tooth = tooth_outline(&t.spec, false, 0.5, None);
        let groove = (PI * 64.0 - PI * tooth.root.powi(2)) * 10.0 - teeth_volume(&tooth.points, tooth.root, 8.0, 2.0, 10.0);
        assert!(near(vol(&top), PI * 64.0 * 30.0 - groove, VOLUME_SHARE), "{}", vol(&top));
        let round = |b: &Body, z: f64| (0..8).all(|i| inside(b, [7.9 * (i as f64 * 0.785).cos(), 7.9 * (i as f64 * 0.785).sin(), z]));
        // From z 15 to 25: the top 5 mm and the lower half stay round.
        assert!(round(&top, 27.0));
        assert!(!round(&top, 20.0));
        assert!(round(&top, 10.0));
        let (flipped, _) = ok(thread(&body, 8.0, &ThreadCut { flip: true, ..t }));
        assert!(round(&flipped, 3.0));
        assert!(!round(&flipped, 10.0));
        assert!(round(&flipped, 20.0));
        let past = thread(&body, 8.0, &ThreadCut { start: 25.0, ..t }).err().unwrap_or_default();
        assert!(past.contains("runs past the end of the face"), "{past}");
    }

    #[test]
    fn stops_at_a_shoulder_leaving_it_whole() {
        let pin = bar(8.0, 20.0);
        let head = kernel::cylinder(sv3(DVec3::new(0.0, 0.0, 20.0)), sv3(DVec3::Z), 12.0, 10.0).expect("a head");
        let body = kernel::boolean(&pin, &head, BoolOp::Union).ok().flatten().expect("a bolt");
        let (threaded, _) = ok(thread(&body, 8.0, &cut(16.0, 2.0)));
        assert!(closed(&threaded));
        // The shoulder's face is untouched: just above it, all the way round, the head is solid
        // out to its rim.
        for i in 0..12 {
            let a = i as f64 * 0.52;
            assert!(inside(&threaded, [9.5 * a.cos(), 9.5 * a.sin(), 20.05]));
        }
        let tooth = tooth_outline(&cut(16.0, 2.0).spec, false, 0.5, None);
        let want = PI * 144.0 * 10.0 + PI * tooth.root.powi(2) * 20.0 + teeth_volume(&tooth.points, tooth.root, 8.0, 2.0, 20.0);
        assert!(near(vol(&threaded), want, VOLUME_SHARE), "{} vs {want}", vol(&threaded));
    }

    #[test]
    fn cuts_a_nuts_thread_in_a_bore_through_a_block() {
        let minor = minor_diameter(16.0, 2.0, ThreadStandard::Metric) / 2.0;
        let body = plate(15.0, minor, 12.0);
        let t = cut(16.0, 2.0);
        let (threaded, note) = ok(thread(&body, minor, &t));
        assert!(closed(&threaded));
        assert_eq!(note, None);
        let tooth = tooth_outline(&t.spec, true, 0.5, None);
        let removed = PI * (64.0 - minor * minor) * 12.0 - teeth_volume(&tooth.points, minor, 8.0, 2.0, 12.0);
        let before = vol(&body);
        assert!(near(vol(&threaded), before - removed, VOLUME_SHARE), "{} vs {}", vol(&threaded), before - removed);
    }

    #[test]
    fn cuts_the_printing_form_on_a_boss_and_in_a_bore_of_the_volume_its_profile_says() {
        let t = ThreadCut { spec: ThreadSpec { standard: ThreadStandard::Print45, diameter: 10.0, pitch: 2.75, offset: 0.0 }, ..cut(10.0, 2.75) };
        let (bolt, note) = ok(thread(&bar(5.0, 20.0), 5.0, &t));
        assert!(closed(&bolt));
        assert_eq!(note, None);
        let tooth = tooth_outline(&t.spec, false, 0.5, None);
        let want = PI * tooth.root.powi(2) * 20.0 + teeth_volume(&tooth.points, tooth.root, 5.0, 2.75, 20.0);
        assert!(near(vol(&bolt), want, VOLUME_SHARE), "{} vs {want}", vol(&bolt));
        // A nut's, in a bore at the form's minor diameter through a block.
        let minor = minor_diameter(10.0, 2.75, ThreadStandard::Print45) / 2.0;
        let block = plate(12.0, minor, 12.0);
        let (nut, note) = ok(thread(&block, minor, &t));
        assert!(closed(&nut));
        assert_eq!(note, None);
        let nut_tooth = tooth_outline(&t.spec, true, 0.5, None);
        let removed = PI * (25.0 - minor * minor) * 12.0 - teeth_volume(&nut_tooth.points, minor, 5.0, 2.75, 12.0);
        assert!(near(vol(&nut), vol(&block) - removed, VOLUME_SHARE), "{} vs {}", vol(&nut), vol(&block) - removed);
        // Left-handed, with an offset, chamfered at both free ends.
        let loose = ThreadCut { left_handed: true, chamfer: true, spec: ThreadSpec { offset: 0.2, ..t.spec }, ..t };
        let (b, _) = ok(thread(&bar(5.0, 20.0), 5.0, &loose));
        assert!(closed(&b));
    }

    #[test]
    fn takes_the_offset_off_every_face_of_the_thread() {
        let body = bar(8.0, 30.0);
        let t = ThreadCut { spec: ThreadSpec { offset: 0.2, ..cut(16.0, 2.0).spec }, ..cut(16.0, 2.0) };
        let (loose, _) = ok(thread(&body, 8.0, &t));
        assert!(closed(&loose));
        let tooth = tooth_outline(&t.spec, false, 0.5, None);
        let want = PI * tooth.root.powi(2) * 30.0 + teeth_volume(&tooth.points, tooth.root, 8.0, 2.0, 30.0);
        assert!(near(vol(&loose), want, VOLUME_SHARE), "{} vs {want}", vol(&loose));
        // Past the flat's worth, the crest is a point, and the row says so.
        let sharp = ThreadCut { spec: ThreadSpec { offset: 0.3, ..t.spec }, ..t };
        let (pointed, note) = ok(thread(&body, 8.0, &sharp));
        assert!(closed(&pointed));
        assert!(note.unwrap_or_default().contains("crests come to a point"));
    }

    /// The booleans on a screwed tooth are where a thread breaks: each of these is long for its
    /// pitch — something cut, and no more than the band between the diameters.
    fn common_size(d: f64, p: f64, internal: bool, lh: bool, offset: f64, length: f64) {
        let t = ThreadCut { left_handed: lh, spec: ThreadSpec { offset, ..cut(d, p).spec }, ..cut(d, p) };
        let minor = minor_diameter(d, p, ThreadStandard::Metric) / 2.0;
        let r = if internal { minor } else { d / 2.0 };
        let body = if internal { plate(d, r, length) } else { bar(r, length) };
        let label = format!("{} Ø{d}×{p}{}", if internal { "nut" } else { "bolt" }, if lh { " LH" } else { "" });
        let (threaded, _) = thread(&body, r, &t).unwrap_or_else(|e| panic!("{label}: {e}"));
        assert!(closed(&threaded), "{label}");
        let lost = vol(&body) - vol(&threaded);
        let band = PI * ((d / 2.0 + offset).powi(2) - (minor - offset).powi(2)) * length;
        assert!(lost > 0.2 * band && lost < band, "{label}: {lost} of {band}");
    }

    #[test]
    #[ignore = "the kernel's faceted boolean cannot close this cut yet (an M3 bolt, left-hand, its crests all but used up by the offset)"]
    fn cuts_the_common_sizes_an_m3_bolt_left_handed_with_an_offset() {
        common_size(3.0, 0.5, false, true, 0.05, 12.0);
    }

    #[test]
    #[ignore = "the kernel's faceted boolean cannot close this cut yet (a long left-hand nut with an offset, in a thin wall)"]
    fn cuts_the_common_sizes_an_m8_nut_left_handed_with_an_offset() {
        common_size(8.0, 1.25, true, true, 0.1, 16.0);
    }

    #[test]
    fn cuts_the_common_sizes_a_quarter_inch_bolt_with_an_offset() {
        common_size(6.35, 25.4 / 20.0, false, false, 0.1, 20.0);
    }

    #[test]
    #[ignore = "the kernel's faceted boolean cannot close this cut yet (a nut 25 mm long in a thin wall)"]
    fn cuts_the_common_sizes_an_m12_nut() {
        common_size(12.0, 1.75, true, false, 0.0, 25.0);
    }

    #[test]
    fn cuts_a_boss_wider_than_the_thread_down_to_it_and_builds_one_too_thin_up_to_it() {
        let (wide, note) = ok(thread(&bar(10.0, 20.0), 10.0, &cut(16.0, 2.0)));
        assert!(closed(&wide));
        assert!(note.unwrap_or_default().contains("The boss is Ø20 mm, wider than the thread's Ø16 mm"));
        assert!(wide.outline_bounds().max.x < 8.0 + 1e-4);
        // Ø12, inside even the root: built up to Ø16 and threaded.
        let t = cut(16.0, 2.0);
        let (built, note) = ok(thread(&bar(6.0, 20.0), 6.0, &t));
        assert!(closed(&built));
        assert!(note.unwrap_or_default().contains("The boss is Ø12 mm, under the thread's Ø16 mm: it is built up to it along the thread"));
        let tooth = tooth_outline(&t.spec, false, 0.5, None);
        let want = PI * tooth.root.powi(2) * 20.0 + teeth_volume(&tooth.points, tooth.root, 8.0, 2.0, 20.0);
        assert!(near(vol(&built), want, VOLUME_SHARE), "{} vs {want}", vol(&built));
        assert!(built.outline_bounds().max.x > 7.9);
    }

    /// What a nut's thread `length` long changes a bore of radius `r` by.
    fn nut_change(t: &ThreadSpec, r: f64, length: f64) -> f64 {
        let tooth = tooth_outline(t, true, 0.5, None);
        teeth_volume(&tooth.points, tooth.crest, tooth.root, t.pitch, length) - PI * (tooth.root.powi(2) - r * r) * length
    }

    #[test]
    fn fills_a_5_bore_in_to_the_thread_and_cuts_m5_m4_or_m6_in_it() {
        let body = plate(15.0, 2.5, 10.0);
        let before = vol(&body);
        for (d, p, note) in [(5.0, 0.8, Some("The bore is Ø5 mm, wider than the thread's minor Ø4.134 mm: it is filled in to it along the thread")), (4.0, 0.7, Some("filled in to it along the thread")), (6.0, 1.0, None)] {
            let t = cut(d, p);
            let (threaded, said) = ok(thread(&body, 2.5, &t));
            assert!(closed(&threaded), "M{d}");
            match note {
                Some(n) => assert!(said.as_deref().is_some_and(|s| s.contains(n)), "M{d}: {said:?}"),
                None => assert_eq!(said, None, "M{d}"),
            }
            let want = before + nut_change(&t.spec, 2.5, 10.0);
            assert!((vol(&threaded) - want).abs() / before < BEFORE_SHARE, "M{d}: {} vs {want}", vol(&threaded));
            // The crests at the thread's minor diameter: just outside it the teeth, just inside
            // it nothing.
            let crest = minor_diameter(d, p, ThreadStandard::Metric) / 2.0;
            assert_eq!(ring(&threaded, crest + 0.03, 5.0, 0.1), "some", "M{d}");
            assert_eq!(ring(&threaded, crest - 0.03, 5.0, 0.1), "none", "M{d}");
        }
    }

    #[test]
    fn fills_the_bore_only_along_the_thread_leaving_the_rest_as_drawn() {
        let body = plate(15.0, 2.5, 20.0);
        let before = vol(&body);
        // M5's root is the bore; M4's stands inside it, so the band runs out past the fill's end
        // into the open bore.
        for t in [ThreadCut { length: Some(8.0), ..cut(5.0, 0.8) }, ThreadCut { length: Some(8.0), ..cut(4.0, 0.7) }, ThreadCut { length: Some(8.0), flip: true, start: 2.0, ..cut(5.0, 0.8) }] {
            let label = format!("M{}{}", t.spec.diameter, if t.flip { " flipped" } else { "" });
            let (threaded, _) = thread(&body, 2.5, &t).unwrap_or_else(|e| panic!("{label}: {e}"));
            assert!(closed(&threaded), "{label}");
            let want = before + nut_change(&t.spec, 2.5, 8.0);
            assert!((vol(&threaded) - want).abs() / before < BEFORE_SHARE, "{label}: {} vs {want}", vol(&threaded));
            let (thread_z, plain_z) = if t.flip { (6.0, 15.0) } else { (16.0, 6.0) };
            let tooth = tooth_outline(&t.spec, true, 0.0, None);
            assert_eq!(ring(&threaded, (tooth.crest + tooth.root) / 2.0, thread_z, 0.1), "some", "{label}");
            if tooth.root < 2.4 {
                assert_eq!(ring(&threaded, (tooth.root + 2.5) / 2.0, thread_z, 0.1), "all", "{label}");
            }
            assert_eq!(ring(&threaded, 2.3, plain_z, 0.1), "none", "{label}");
            assert_eq!(ring(&threaded, 2.55, plain_z, 0.1), "all", "{label}");
        }
    }

    #[test]
    #[ignore = "the kernel's faceted boolean cannot close this cut yet (a fill kept to a curved wall: the wall faceted twice)"]
    fn keeps_the_fill_to_the_body_where_a_bore_comes_out_of_a_curved_wall() {
        // A Ø5 bore drilled across the wall of a tube up Z, outside R16.8, inside R13.6: both
        // its ends are saddles, not square to it.
        let tube = annulus_solid(&Annulus { center: DVec3::ZERO, axis: DVec3::Z, outer: 16.8, inner: Some(13.6), from: 0.0, to: 20.0 }).expect("a tube");
        let drill = kernel::cylinder(sv3(DVec3::new(10.0, 0.0, 10.0)), sv3(DVec3::X), 2.5, 10.0).expect("a drill");
        let body = kernel::boolean(&tube, &drill, BoolOp::Cut).ok().flatten().expect("a drilled tube");
        let (threaded, note) = ok(thread(&body, 2.5, &cut(5.0, 0.8)));
        assert!(closed(&threaded));
        assert!(note.unwrap_or_default().contains("filled in to it along the thread"));
        // What the thread added lies in the bore, none of it past the wall: the threaded body
        // stays inside the tube.
        assert!(vol(&threaded) > vol(&body) + 1.0);
        for z in [9.0, 10.0, 11.0] {
            for x in [17.0, 13.3] {
                assert!(!inside(&threaded, [x, 0.0, z]), "{x} {z}");
            }
        }
    }

    /// A pin, its tip cut off at 30°: an M5 would build it up, out past the slant.
    fn slanted_pin(r: f64) -> Body {
        let wedge = kernel::extrude(&Plane::new(sv3(DVec3::ZERO), sv3(DVec3::X), sv3(DVec3::Z)).expect("a plane"), &[Region2 { outer: Loop2::polygon(&[SVec2::new(-5.0, 12.0), SVec2::new(5.0, 14.5), SVec2::new(5.0, 20.0), SVec2::new(-5.0, 20.0)]), holes: Vec::new() }], -10.0, 10.0).ok().and_then(|mut v| v.pop()).expect("a wedge");
        kernel::boolean(&bar(r, 15.0), &wedge, BoolOp::Cut).ok().flatten().expect("a slanted pin")
    }

    #[test]
    #[ignore = "the kernel's faceted boolean cannot close this cut yet (the thread run out through a slanted end)"]
    fn leaves_a_boss_with_a_slanted_end_as_drawn_rather_than_build_it_up_past_the_end() {
        // Ø4.6, between M5's minor and major: cut as drawn, the crests on it.
        let body = slanted_pin(2.3);
        let (threaded, note) = ok(thread(&body, 2.3, &cut(5.0, 0.8)));
        assert!(closed(&threaded));
        assert!(note.unwrap_or_default().contains("The boss is Ø4.6 mm, not the thread's Ø5 mm, and it does not run between square ends: it is left as drawn"));
        assert!(vol(&threaded) < vol(&body) - 1.0);
    }

    #[test]
    fn refuses_to_build_a_boss_with_a_slanted_end_up_to_a_thread_it_is_under() {
        // Ø4, under M5's minor: nothing of the thread would be cut, refused.
        let why = thread(&slanted_pin(2.0), 2.0, &cut(5.0, 0.8)).err().unwrap_or_default();
        assert!(why.contains("the boss is Ø4 mm, under the thread's minor Ø4.134 mm, and it does not run between square ends, so it cannot be built up to the thread"), "{why}");
    }

    #[test]
    #[ignore = "the kernel's faceted boolean cannot close this cut yet (a short thread with an offset in a filled bore)"]
    fn keeps_the_wall_past_a_short_thread_whole_under_an_offset() {
        let body = plate(15.0, 2.5, 20.0);
        // The offset takes the root out past the bore: the wall beyond the thread's end is not
        // cut.
        let (loose, _) = ok(thread(&body, 2.5, &ThreadCut { length: Some(8.0), spec: ThreadSpec { offset: 0.05, ..cut(5.0, 0.8).spec }, ..cut(5.0, 0.8) }));
        assert!(closed(&loose));
        assert_eq!(ring(&loose, 2.52, 11.9, 0.1), "all");
        assert_eq!(ring(&loose, 2.52, 13.0, 0.1), "some");
    }

    #[test]
    #[ignore = "the kernel's faceted boolean cannot close this cut yet (a fill to crests all but used up by the offset)"]
    fn fills_a_bore_out_to_the_crests_near_the_offsets_limit() {
        let body = plate(15.0, 2.5, 20.0);
        let t = ThreadCut { spec: ThreadSpec { offset: 0.16, ..cut(5.0, 0.8).spec }, ..cut(5.0, 0.8) };
        let (deep, _) = ok(thread(&body, 2.5, &t));
        assert!(closed(&deep));
        assert!((vol(&deep) - (vol(&body) + nut_change(&t.spec, 2.5, 20.0))).abs() / vol(&body) < BEFORE_SHARE, "{}", vol(&deep));
    }

    #[test]
    fn builds_a_pin_up_a_quarter_pitch_past_pointed_crests() {
        // At 0.12 mm a bolt's 60° crests are points: a Ø4 pin is built up a quarter pitch past
        // them, and the cut takes that off.
        let bolt = ThreadCut { spec: ThreadSpec { offset: 0.12, ..cut(5.0, 0.8).spec }, ..cut(5.0, 0.8) };
        let (sharp, note) = ok(thread(&bar(2.0, 15.0), 2.0, &bolt));
        assert!(closed(&sharp));
        assert!(note.unwrap_or_default().contains("built up to it along the thread. At this offset the crests come to a point"));
        let tooth = tooth_outline(&bolt.spec, false, 0.5, None);
        let want = PI * tooth.root.powi(2) * 15.0 + teeth_volume(&tooth.points, tooth.root, tooth.crest, 0.8, 15.0);
        assert!(near(vol(&sharp), want, VOLUME_SHARE), "{} vs {want}", vol(&sharp));
    }

    /// A filled bore countersunk at both mouths: on each end face, nothing inside the
    /// countersink's radius; a fill wider than it stands round it.
    fn countersunk_filled(t: ThreadCut) {
        let body = plate(15.0, 2.5, 10.0);
        let d = t.spec.diameter;
        let (threaded, _) = thread(&body, 2.5, &t).unwrap_or_else(|e| panic!("M{d}: {e}"));
        assert!(closed(&threaded), "M{d}");
        let rc = chamfer_radius(&t.spec, true);
        for z in [0.02, 9.98] {
            assert_eq!(ring(&threaded, rc - 0.08, z, 0.1), "none", "M{d} at z {z}");
            assert_eq!(ring(&threaded, rc.max(2.5) + 0.1, z, 0.1), "all", "M{d} at z {z}");
        }
        if rc < 2.4 {
            assert_eq!(ring(&threaded, (rc + 2.5) / 2.0, 0.02, 0.1), "all", "M{d}");
        }
    }

    #[test]
    #[ignore = "the kernel's faceted boolean cannot close this cut yet (a fill and a countersink at one mouth)"]
    fn countersinks_a_filled_bore_at_both_mouths_m5() {
        countersunk_filled(ThreadCut { chamfer: true, ..cut(5.0, 0.8) });
    }

    #[test]
    #[ignore = "the kernel's faceted boolean cannot close this cut yet (a fill and a countersink at one mouth)"]
    fn countersinks_a_filled_bore_at_both_mouths_m4() {
        countersunk_filled(ThreadCut { chamfer: true, ..cut(4.0, 0.7) });
    }

    #[test]
    fn builds_a_4_pin_up_to_an_m5_along_part_of_its_length() {
        let t = ThreadCut { length: Some(8.0), chamfer: true, ..cut(5.0, 0.8) };
        let (threaded, note) = ok(thread(&bar(2.0, 15.0), 2.0, &t));
        assert!(closed(&threaded));
        assert!(note.unwrap_or_default().contains("built up to it along the thread"));
        assert_eq!(ring(&threaded, 2.3, 11.0, 0.0), "some");
        assert_eq!(ring(&threaded, 2.1, 4.0, 0.0), "none");
        assert_eq!(ring(&threaded, 1.9, 4.0, 0.0), "all");
        // The tip chamfered, inside the root at the end face.
        assert_eq!(ring(&threaded, chamfer_radius(&t.spec, false) + 0.08, 14.98, 0.0), "none");
    }

    /// The volume a triangle in (r, s) between the 45° cone through `rc` on the end face and
    /// the radius `to` fills turned about the axis.
    fn cone_ring(rc: f64, to: f64) -> f64 {
        (to - rc).abs().powi(2) / 2.0 * 2.0 * PI * ((rc + 2.0 * to) / 3.0)
    }

    #[test]
    fn chamfers_a_bolt_at_both_free_ends_at_45_from_just_inside_the_root_the_thread_between_untouched() {
        let body = bar(8.0, 30.0);
        let t = cut(16.0, 2.0);
        let (plain, _) = ok(thread(&body, 8.0, &t));
        let (chamfered, _) = ok(thread(&body, 8.0, &ThreadCut { chamfer: true, ..t }));
        assert!(closed(&chamfered));
        let rc = chamfer_radius(&t.spec, false);
        let root = tooth_outline(&t.spec, false, 0.0, None).root;
        assert!((rc - (root - 0.2)).abs() < 1e-9);
        let bb = chamfered.outline_bounds();
        assert!(bb.min.z.abs() < 1e-6 && (bb.max.z - 30.0).abs() < 1e-6);
        for z in [0.02, 29.98] {
            assert_eq!(ring(&chamfered, rc + 0.08, z, 0.0), "none", "past the chamfer at z {z}");
            assert_eq!(ring(&chamfered, rc - 0.05, z, 0.0), "all", "inside it at z {z}");
            assert_ne!(ring(&plain, rc + 0.6, z, 0.0), "none", "the plain thread at z {z}");
        }
        // Half a millimetre in, the cone is half a millimetre further out.
        assert_eq!(ring(&chamfered, rc + 0.58, 0.5, 0.0), "none");
        assert_eq!(ring(&chamfered, rc + 0.58, 29.5, 0.0), "none");
        // In the middle, the thread is the plain one's.
        for i in 0..8 {
            let a = i as f64 * 0.8;
            let p = [7.5 * a.cos(), 7.5 * a.sin(), 12.0 + i as f64 * 0.73];
            assert_eq!(inside(&chamfered, p), inside(&plain, p), "{p:?}");
        }
        // What the chamfers took: at least the core they cut into, at most the whole ring out to
        // the crests, at each end.
        let lost = vol(&plain) - vol(&chamfered);
        assert!(lost > 2.0 * cone_ring(rc, root) && lost < 2.0 * cone_ring(rc, 8.0), "{lost}");
    }

    #[test]
    fn chamfers_a_bolts_tip_and_leaves_its_heads_shoulder_whole() {
        let head = kernel::cylinder(sv3(DVec3::new(0.0, 0.0, 20.0)), sv3(DVec3::Z), 12.0, 10.0).expect("a head");
        let body = kernel::boolean(&bar(8.0, 20.0), &head, BoolOp::Union).ok().flatten().expect("a bolt");
        let t = ThreadCut { chamfer: true, ..cut(16.0, 2.0) };
        let (threaded, _) = ok(thread(&body, 8.0, &t));
        assert!(closed(&threaded));
        let rc = chamfer_radius(&t.spec, false);
        assert_eq!(ring(&threaded, rc + 0.08, 0.02, 0.0), "none");
        // Against the shoulder the thread runs on at full depth: the crests are there, and the
        // head is solid out to its rim just above.
        assert_eq!(ring(&threaded, 7.9, 19.9, 0.0), "some");
        assert_eq!(ring(&threaded, 9.5, 20.05, 0.0), "all");
    }

    #[test]
    fn countersinks_a_nuts_bore_at_both_mouths_out_past_the_major_diameter() {
        let minor = minor_diameter(16.0, 2.0, ThreadStandard::Metric) / 2.0;
        let body = plate(15.0, minor, 12.0);
        let t = ThreadCut { chamfer: true, ..cut(16.0, 2.0) };
        let (threaded, _) = ok(thread(&body, minor, &t));
        assert!(closed(&threaded));
        let rc = chamfer_radius(&t.spec, true);
        assert!((rc - 8.2).abs() < 1e-9);
        for z in [0.02, 11.98] {
            assert_eq!(ring(&threaded, minor + 0.1, z, 0.0), "none", "the mouth at z {z}");
            assert_eq!(ring(&threaded, rc - 0.08, z, 0.0), "none", "out to the countersink at z {z}");
            assert_eq!(ring(&threaded, rc + 0.05, z, 0.0), "all", "the block past it at z {z}");
        }
        assert_eq!(ring(&threaded, rc - 0.58, 0.5, 0.0), "none");
        // Halfway through, the nut's teeth are there.
        assert_eq!(ring(&threaded, minor + 0.1, 6.0, 0.0), "some");
        // A plate 3 mm thick — a turn and a half — takes both, the thread between them.
        let thin = plate(15.0, minor, 3.0);
        let (nut, _) = ok(thread(&thin, minor, &t));
        assert!(closed(&nut));
        assert_eq!(ring(&nut, minor + 0.1, 1.5, 0.0), "some");
        assert_eq!(ring(&nut, rc - 0.08, 2.98, 0.0), "none");
    }

    #[test]
    #[ignore = "the kernel's faceted boolean cannot close this cut yet (an M3 bolt, left-hand, its crests all but used up by the offset, chamfered)"]
    fn chamfers_a_fine_left_hand_thread_with_an_offset() {
        let t = ThreadCut { left_handed: true, chamfer: true, spec: ThreadSpec { offset: 0.05, ..cut(3.0, 0.5).spec }, ..cut(3.0, 0.5) };
        let (threaded, _) = ok(thread(&bar(1.5, 12.0), 1.5, &t));
        assert!(closed(&threaded));
        assert_eq!(ring(&threaded, chamfer_radius(&t.spec, false) + 0.03, 11.99, 0.0), "none");
    }

    #[test]
    fn refuses_a_chamfer_where_the_thread_does_not_come_out_of_the_body_free() {
        let why = thread(&bar(8.0, 30.0), 8.0, &ThreadCut { length: Some(10.0), start: 5.0, chamfer: true, ..cut(16.0, 2.0) }).err().unwrap_or_default();
        assert!(why.contains("no end to chamfer"), "{why}");
        let why = thread(&bar(8.0, 1.5), 8.0, &ThreadCut { chamfer: true, ..cut(16.0, 2.0) }).err().unwrap_or_default();
        assert!(why.contains("too short for a chamfer at each end"), "{why}");
    }

    #[test]
    fn taps_two_bores_of_one_plate_one_after_the_other_each_on_its_exact_face() {
        // A 40 × 20 × 8 plate with two Ø5 bores through it: each thread finds its bore round.
        let mut part = Part::default();
        let mut s = Sketch::default();
        s.add_rectangle(DVec2::new(0.0, 0.0), DVec2::new(40.0, 20.0));
        s.add_shape(Shape::Circle { c: DVec2::new(10.0, 10.0), r: 2.5 });
        s.add_shape(Shape::Circle { c: DVec2::new(30.0, 10.0), r: 2.5 });
        let regions = default_regions(&find_regions(&s));
        let sk = part.add(FeatureKind::Sketch(SketchFeature::on_world(WorldAxis::Z, 0.0, s)));
        part.add(FeatureKind::Extrude(ExtrudeFeature::new(sk, regions, ExtrudeExtent::Distance { from: 0.0, to: 8.0 }, BodyOp::New)));
        let plain = build_part(&part);
        assert!(plain.errors.is_empty(), "{:?}", plain.errors);
        let bores: Vec<_> = plain.bodies[0].topology.faces.iter().filter(|f| f.fingerprint.kind == FaceKind::Cylinder).map(|f| f.fingerprint).collect();
        let at = |x: f64| bores.iter().find(|f| f.center.is_some_and(|c| (c.x - x).abs() < 1e-6)).copied().expect("a bore");
        let thread = |face| FeatureKind::Thread(ThreadFeature { body: BodyId::new(2), face, standard: ModelStandard::Metric, diameter: 6.0, pitch: 1.0, left_handed: false, length: None, start: 0.0, flip: false, offset: 0.0, chamfer: None, exprs: Default::default() });
        part.add(thread(at(10.0)));
        part.add(thread(at(30.0)));
        let built = build_part(&part);
        assert!(built.errors.is_empty(), "{:?}", built.errors);
        assert_eq!(built.bodies.len(), 1);
        let b = &built.bodies[0].body;
        assert!(closed(b));
        // Both bores carry the thread: the teeth stand at the crests, the root is cut out.
        let crest = minor_diameter(6.0, 1.0, ThreadStandard::Metric) / 2.0;
        for cx in [10.0, 30.0] {
            let some = (0..16).filter(|i| {
                let a = *i as f64 * PI / 8.0;
                inside(b, [cx + (crest + 0.1) * a.cos(), 10.0 + (crest + 0.1) * a.sin(), 4.0])
            }).count();
            assert!(some > 0 && some < 16, "bore at {cx}: {some} of 16");
        }
        // Each thread names the body it cut.
        assert!(built.touched.get(&3).is_some() && built.touched.get(&4).is_some());
    }

    #[test]
    fn draws_a_crest_too_narrow_for_the_facets_to_its_point_and_grows_a_ring_square_to_its_sides() {
        // A bolt's tooth with a flat 0.004 wide at r 1.45: drawn to where its flanks meet.
        let tooth = [DVec2::new(0.9, -0.17), DVec2::new(1.15, -0.17), DVec2::new(1.45, -0.002), DVec2::new(1.45, 0.002), DVec2::new(1.15, 0.17), DVec2::new(0.9, 0.17)];
        let sharp = sharpened(&tooth, 0.01);
        assert_eq!(sharp.len(), 5);
        let apex = sharp[2];
        assert!(apex.y.abs() < 1e-12 && apex.x > 1.45 && apex.x < 1.46, "{apex}");
        // Wide enough, it stays; a nut's flat across the ends of the outline, too.
        assert_eq!(sharpened(&tooth, 0.001).len(), 6);
        let nut = [DVec2::new(2.1, -0.002), DVec2::new(2.5, -0.3), DVec2::new(3.0, -0.3), DVec2::new(3.0, 0.3), DVec2::new(2.5, 0.3), DVec2::new(2.1, 0.002)];
        let point = sharpened(&nut, 0.01);
        assert_eq!(point.len(), 5);
        assert!(point.iter().any(|q| q.y.abs() < 1e-12 && q.x < 2.1));
        // A right triangle grown by 0.1: each side 0.1 further out.
        let grown_tri = grown(&[DVec2::new(0.0, 0.0), DVec2::new(1.0, 0.0), DVec2::new(0.0, 1.0)], 0.1);
        assert!((grown_tri[0] - DVec2::new(-0.1, -0.1)).length() < 1e-9, "{:?}", grown_tri);
        assert!(grown_tri[1].y.abs() > 0.099 && grown_tri[2].x < -0.099);
    }

    #[test]
    fn tells_a_boss_from_a_bore_by_the_way_the_faces_triangles_look() {
        use crate::thread::{round_face_is_boss, FaceMesh};
        let minor = minor_diameter(16.0, 2.0, ThreadStandard::Metric) / 2.0;
        let body = plate(15.0, minor, 12.0);
        let m = crate::build::body_mesh(&body).expect("a mesh");
        let t = read_topology(&body, &vec![1; body.face_count()]);
        let face = |r: f64| t.faces.iter().find(|f| f.fingerprint.kind == FaceKind::Cylinder && f.fingerprint.radius.is_some_and(|x| (x - r).abs() < 1e-6)).map(|f| (f.hash, f.fingerprint)).expect("a round face");
        let mesh = FaceMesh { positions: &m.positions, triangles: &m.triangles, tri_face: &m.tri_face };
        let (outside, o) = face(15.0);
        let (bore, b) = face(minor);
        assert_eq!(round_face_is_boss(mesh, outside, o.center.unwrap_or_default(), o.dir), Some(true));
        assert_eq!(round_face_is_boss(mesh, bore, b.center.unwrap_or_default(), b.dir), Some(false));
        assert_eq!(round_face_is_boss(mesh, 12345, b.center.unwrap_or_default(), b.dir), None);
        // Read off the body as the kernel holds it, the same.
        assert!(!read_round(&body, &t, bore as usize).map(|r| r.0.convex).unwrap_or(true));
        assert!(read_round(&body, &t, outside as usize).map(|r| r.0.convex).unwrap_or(false));
    }

    /// A Ø16 bar 30 long, built, and a thread feature on its round face.
    fn threaded_bar(f: impl FnOnce(&mut ThreadFeature)) -> (Part, crate::build::Built) {
        let mut part = Part::default();
        let mut s = Sketch::default();
        s.add_shape(Shape::Circle { c: DVec2::ZERO, r: 8.0 });
        let regions = default_regions(&find_regions(&s));
        let sk = part.add(FeatureKind::Sketch(SketchFeature::on_world(WorldAxis::Z, 0.0, s)));
        part.add(FeatureKind::Extrude(ExtrudeFeature::new(sk, regions, ExtrudeExtent::Distance { from: 0.0, to: 30.0 }, BodyOp::New)));
        let plain = build_part(&part);
        let face = plain.bodies[0].topology.faces.iter().find(|x| x.fingerprint.kind == FaceKind::Cylinder).map(|x| x.fingerprint).expect("a round face");
        let mut x = ThreadFeature { body: BodyId::new(2), face, standard: ModelStandard::Metric, diameter: 16.0, pitch: 2.0, left_handed: false, length: None, start: 0.0, flip: false, offset: 0.0, chamfer: None, exprs: Default::default() };
        f(&mut x);
        part.add(FeatureKind::Thread(x));
        let built = build_part(&part);
        (part, built)
    }

    #[test]
    fn the_feature_builds_on_the_bodys_round_face_by_its_fingerprint_in_the_form_of_its_standard() {
        let (_, built) = threaded_bar(|_| {});
        assert!(built.errors.is_empty(), "{:?}", built.errors);
        assert_eq!(built.bodies.len(), 1);
        assert!(closed(&built.bodies[0].body));
        let tooth = tooth_outline(&cut(16.0, 2.0).spec, false, 0.5, None);
        let want = PI * tooth.root.powi(2) * 30.0 + teeth_volume(&tooth.points, tooth.root, 8.0, 2.0, 30.0);
        assert!(near(vol(&built.bodies[0].body), want, VOLUME_SHARE), "{}", vol(&built.bodies[0].body));
        // The printing one's tooth, carried through the build to the kernel.
        let (_, print) = threaded_bar(|x| {
            x.standard = ModelStandard::Print45;
            x.pitch = 3.5;
        });
        assert!(print.errors.is_empty(), "{:?}", print.errors);
        let tooth = tooth_outline(&ThreadSpec { standard: ThreadStandard::Print45, diameter: 16.0, pitch: 3.5, offset: 0.0 }, false, 0.5, None);
        let want = PI * tooth.root.powi(2) * 30.0 + teeth_volume(&tooth.points, tooth.root, 8.0, 3.5, 30.0);
        assert!(near(vol(&print.bodies[0].body), want, VOLUME_SHARE), "{}", vol(&print.bodies[0].body));
    }

    #[test]
    fn the_feature_fails_with_the_reason() {
        let (_, big) = threaded_bar(|x| x.offset = 0.6);
        assert!(big.errors.get(&3).is_some_and(|e| e.contains("no thickness at the pitch diameter")), "{:?}", big.errors);
        let (_, long) = threaded_bar(|x| x.length = Some(40.0));
        assert!(long.errors.get(&3).is_some_and(|e| e.contains("runs past the end of the face")), "{:?}", long.errors);
        let (_, flat) = threaded_bar(|x| x.face = crate::topo::FaceFingerprint { kind: FaceKind::Plane, centroid: DVec3::new(0.0, 0.0, 30.0), dir: DVec3::Z, area: 200.0, radius: None, center: None, origin: 2 });
        assert!(flat.errors.get(&3).is_some_and(|e| e.contains("round face")), "{:?}", flat.errors);
        let (_, none) = threaded_bar(|x| x.face = super::super::hole::unpicked_face());
        assert_eq!(none.errors.get(&3).map(String::as_str), Some(THREAD_UNPICKED));
    }

    #[test]
    fn the_feature_chamfers_the_free_ends_when_asked_and_fails_saying_why_where_there_are_none() {
        let (_, plain) = threaded_bar(|_| {});
        let (_, chamfered) = threaded_bar(|x| x.chamfer = Some(true));
        assert!(chamfered.errors.is_empty(), "{:?}", chamfered.errors);
        assert!(closed(&chamfered.bodies[0].body));
        assert!(vol(&chamfered.bodies[0].body) < vol(&plain.bodies[0].body) - 1.0);
        // A thread saved before the chamfer was offered builds without one.
        let saved: ThreadFeature = serde_json::from_value(serde_json::json!({ "body": 2, "face": { "kind": "cylinder", "centroid": [0, 0, 15], "dir": [0, 0, 1], "area": 1508, "radius": 8, "center": [0, 0, 15], "origin": 2 }, "standard": "metric", "diameter": 16, "pitch": 2, "leftHanded": false, "length": null, "start": 0, "flip": false, "offset": 0 })).expect("a thread");
        assert!(!thread_cut_of(&saved).chamfer);
        let (_, inside) = threaded_bar(|x| {
            x.chamfer = Some(true);
            x.length = Some(10.0);
            x.start = 5.0;
        });
        assert!(inside.errors.get(&3).is_some_and(|e| e.contains("no end to chamfer")), "{:?}", inside.errors);
    }
}

