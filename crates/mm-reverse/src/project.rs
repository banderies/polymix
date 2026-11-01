// SPDX-License-Identifier: AGPL-3.0-only
//! Geometry that is already there, brought into a sketch: the edges of a body and the pieces
//! of another sketch projected onto the sketch plane as pieces of this one — the way Fusion's
//! Project and Onshape's Use bring the model into the sketch. Ported from ScanRuler
//! (`plugins/reverse/core/project.ts`), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! A projected piece is an ordinary line, arc or circle, fixed where it landed and marked
//! `projected`, so every relation and dimension the sketcher has can be hung on it and the
//! solver moves the drawn pieces to it, never it to them. A sketch opened on a flat face of a
//! body starts with that face's outline projected, which is a closed profile the moment the
//! sheet opens.
//!
//! A projection follows what it was taken from, as CAD's does: the piece keeps its source —
//! the edge by its fingerprint, the other sketch's piece by its id — and every build takes it
//! again from there before the features on the sketch are built (`build::sketch`), the drawn
//! pieces held to it following through their relations. A source the build no longer finds
//! leaves the piece where it last was.
//!
//! What projects: a straight edge (a line, or a point seen end on), a circular edge whose plane
//! is the sketch plane's (a circle or an arc) or stands square to it (the line it is seen as),
//! and anything else that happens to land on a line. A circle seen at a slant is an ellipse,
//! which the sketcher has no piece for; it is not offered.

use glam::{DVec2, DVec3};

use crate::frame::SectionFrame;
use crate::model::BodyId;
use crate::sketch::{arc_through, polyline, Entity, ProjectionSource, Shape};
use crate::topo::{EdgeFingerprint, EdgeInfo, EdgeKind};

/// Something on offer to a sketch's Project tool: the piece it would become and the polyline a
/// click is tested against, sheet millimetres.
#[derive(Debug, Clone, PartialEq)]
pub struct Projectable {
    pub entity: Entity,
    pub polyline: Vec<DVec2>,
    /// What it is, for the status line: "an edge of Extrude 1".
    pub source: String,
}

/// Shorter than this on the sheet, a projected line is a point.
const POINT_MM: f64 = 1e-6;
/// How far off a line the projected points may wander and still be one, as a share of its
/// length.
const LINE_FLATNESS: f64 = 1e-6;

fn len2(a: DVec2) -> f64 {
    a.x.hypot(a.y)
}

fn to_sheet(frame: &SectionFrame, p: DVec3) -> DVec2 {
    frame.to_sheet(p)
}

/// A projected piece: fixed where it landed, marked projected.
fn fixed(shape: Shape) -> Entity {
    let mut e = Entity::new(shape);
    e.fixed = true;
    e.projected = true;
    e
}

/// The line through the two points of a run furthest apart, when every point of the run lies
/// on it; `None` when the run bends.
fn as_line(points: &[DVec2]) -> Option<Entity> {
    if points.len() < 2 {
        return None;
    }
    let mut a = points[0];
    let mut b = points[points.len() - 1];
    // A closed run seen edge on starts and ends at the same place: take the extremes along the
    // run's longest direction instead.
    let mut far = 0.0;
    for p in points {
        let d = len2(*p - points[0]);
        if d > far {
            far = d;
            b = *p;
        }
    }
    far = 0.0;
    for p in points {
        let d = len2(*p - b);
        if d > far {
            far = d;
            a = *p;
        }
    }
    let length = len2(b - a);
    if length < POINT_MM {
        return Some(fixed(Shape::Point { p: a }));
    }
    let nx = -(b.y - a.y) / length;
    let ny = (b.x - a.x) / length;
    for p in points {
        if ((p.x - a.x) * nx + (p.y - a.y) * ny).abs() > (length * LINE_FLATNESS).max(1e-7) {
            return None;
        }
    }
    Some(fixed(Shape::Line { a, b }))
}

/// A body's edge as it is offered for projection: the kernel's drawing of it, its fingerprint
/// saying what it is, and — from a body's topology — its two ends as the kernel has them.
#[derive(Debug, Clone, Copy)]
pub struct ProjectedEdge<'a> {
    pub fingerprint: &'a EdgeFingerprint,
    pub polyline: &'a [DVec3],
    pub ends: Option<[DVec3; 2]>,
}

impl<'a> From<&'a EdgeInfo> for ProjectedEdge<'a> {
    fn from(e: &'a EdgeInfo) -> Self {
        ProjectedEdge { fingerprint: &e.fingerprint, polyline: &e.polyline, ends: Some(e.ends) }
    }
}

/// A body's edge as a piece of a sketch on `frame`, or `None` when it does not project to a
/// line, an arc or a circle. What the polyline decides is only what the edge projects to;
/// where a line or an arc lands is taken from the edge's own ends and middle, so a body pulled
/// out of a face's outline has walls that merge with the ones they continue.
pub fn project_edge(edge: ProjectedEdge, frame: &SectionFrame) -> Option<Entity> {
    let drawn = project_drawn(&edge, frame)?;
    let Some(ends) = edge.ends else { return Some(drawn) };
    let a = to_sheet(frame, ends[0]);
    let b = to_sheet(frame, ends[1]);
    match drawn.shape {
        Shape::Line { .. } if edge.fingerprint.kind == EdgeKind::Line => {
            if len2(b - a) < POINT_MM {
                Some(drawn)
            } else {
                Some(Entity { shape: Shape::Line { a, b }, ..drawn })
            }
        }
        Shape::Arc { .. } => match arc_through(a, b, to_sheet(frame, edge.fingerprint.mid)) {
            Some((c, r, start, sweep, _)) => Some(Entity { shape: Shape::Arc { c, r, start, sweep }, ..drawn }),
            None => Some(drawn),
        },
        _ => Some(drawn),
    }
}

fn project_drawn(edge: &ProjectedEdge, frame: &SectionFrame) -> Option<Entity> {
    let points: Vec<DVec2> = edge.polyline.iter().map(|p| to_sheet(frame, *p)).collect();
    if points.len() < 2 {
        return None;
    }
    let fp = edge.fingerprint;
    if fp.kind == EdgeKind::Circle {
        let along = fp.dir.dot(frame.normal).abs();
        if along > 1.0 - 1e-9 {
            let first = points[0];
            let last = points[points.len() - 1];
            let mid = points[points.len() >> 1];
            let radius = fp.radius.unwrap_or(0.0);
            if len2(first - last) < radius.max(1.0) * 1e-6 {
                // All the way round: the centre is opposite-point arithmetic away.
                let c = (first + mid) / 2.0;
                let quarter = points[points.len() >> 2];
                return Some(match arc_through(first, mid, quarter) {
                    Some((ac, ar, ..)) => fixed(Shape::Circle { c: ac, r: ar }),
                    None => fixed(Shape::Circle { c, r: len2(first - c) }),
                });
            }
            let (c, r, start, sweep, _) = arc_through(first, last, mid)?;
            return Some(fixed(Shape::Arc { c, r, start, sweep }));
        }
        // Seen edge on it is the line it spans; at a slant, an ellipse.
        return if along < 1e-6 { as_line(&points) } else { None };
    }
    as_line(&points)
}

/// A piece of another sketch as a piece of this one, or `None` — the same rules, the other
/// sketch's piece taken as the curve it is.
pub fn project_sketch_entity(e: &Entity, from: &SectionFrame, onto: &SectionFrame) -> Option<Entity> {
    if let Shape::Spline(sp) = &e.shape {
        // Onto a parallel plane the sheet is carried rigidly — turned, moved, perhaps
        // mirrored — and the curve through the points carried is the curve carried. At a
        // slant it is stretched, and the curve through the stretched points another curve:
        // not offered.
        if from.normal.dot(onto.normal).abs() < 1.0 - 1e-9 {
            return None;
        }
        let dir_of = |angle: f64| {
            let d = from.to_world(DVec2::new(angle.cos(), angle.sin())) - from.origin;
            d.dot(onto.v).atan2(d.dot(onto.u))
        };
        let shape = crate::sketch_spline::SplineShape { pts: sp.pts.iter().map(|p| to_sheet(onto, from.to_world(*p))).collect(), closed: sp.closed, start_dir: sp.start_dir.map(dir_of), end_dir: sp.end_dir.map(dir_of) };
        return Some(fixed(Shape::Spline(shape)));
    }
    match e.shape {
        Shape::Spline(_) => None,
        Shape::Point { p } => Some(fixed(Shape::Point { p: to_sheet(onto, from.to_world(p)) })),
        Shape::Line { .. } => {
            let lifted: Vec<DVec2> = polyline(&e.shape, 64).iter().map(|p| to_sheet(onto, from.to_world(*p))).collect();
            as_line(&lifted)
        }
        Shape::Arc { r, .. } | Shape::Circle { r, .. } => {
            let lifted: Vec<DVec3> = polyline(&e.shape, 64).iter().map(|p| from.to_world(*p)).collect();
            let fingerprint = EdgeFingerprint { kind: EdgeKind::Circle, mid: DVec3::ZERO, dir: from.normal, length: 0.0, radius: Some(r), faces: Vec::new(), origin: -1 };
            project_edge(ProjectedEdge { fingerprint: &fingerprint, polyline: &lifted, ends: None }, onto)
        }
    }
}

/// The outline of a body's face as pieces of a sketch on the face: every edge bounding it that
/// projects, and how many did not — a spline's edge leaves the outline open, and the caller
/// says so. Given the body, each piece keeps its edge as its source and follows it.
pub fn project_face_outline(edges: &[ProjectedEdge], frame: &SectionFrame, body: Option<BodyId>) -> (Vec<Entity>, usize) {
    let mut entities = Vec::new();
    let mut skipped = 0;
    for edge in edges {
        match project_edge(*edge, frame) {
            Some(e) if !matches!(e.shape, Shape::Point { .. }) => {
                let mut e = e;
                if let Some(body) = body {
                    e.source = Some(ProjectionSource::Edge { body, edge: edge.fingerprint.clone() });
                }
                entities.push(e);
            }
            Some(_) => {}
            None => skipped += 1,
        }
    }
    (entities, skipped)
}

/// A projected piece taken again: the piece, its id, flags and source kept, where the fresh
/// projection of its source puts it — or `None` when the source no longer projects to a piece
/// of the same kind (a line seen end on now, a circle at a slant), which leaves it where it
/// was. A line keeps the ends its relations hang on.
pub fn retake(piece: &Entity, fresh: Option<&Entity>) -> Option<Entity> {
    let fresh = fresh?;
    let shape = match (piece.shape.clone(), fresh.shape.clone()) {
        (Shape::Line { a: pa, b: pb }, Shape::Line { a: fa, b: fb }) => {
            // The ends as they were: a and b swapped would swap the relations.
            let kept = len2(fa - pa) + len2(fb - pb);
            let swapped = len2(fb - pa) + len2(fa - pb);
            if swapped < kept {
                Shape::Line { a: fb, b: fa }
            } else {
                Shape::Line { a: fa, b: fb }
            }
        }
        (Shape::Arc { .. }, Shape::Arc { c, r, start, sweep }) => Shape::Arc { c, r, start, sweep },
        (Shape::Circle { .. }, Shape::Circle { c, r }) => Shape::Circle { c, r },
        (Shape::Point { .. }, Shape::Point { p }) => Shape::Point { p },
        _ => return None,
    };
    Some(Entity { shape, ..piece.clone() })
}

/// How far a piece taken again moved, millimetres: the most any of its defining points went.
/// An arc is measured on its curve — its ends and its middle — not by its centre: a near-flat
/// arc's centre swings tens of millimetres while the curve itself hardly shifts.
pub fn retake_moved(was: &Entity, now: &Entity) -> f64 {
    let d = |p: DVec2, q: DVec2| len2(p - q);
    match (was.shape.clone(), now.shape.clone()) {
        (Shape::Line { a: wa, b: wb }, Shape::Line { a: na, b: nb }) => d(wa, na).max(d(wb, nb)),
        (Shape::Point { p }, Shape::Point { p: q }) => d(p, q),
        (Shape::Arc { c: wc, r: wr, start: ws, sweep: wsw }, Shape::Arc { c: nc, r: nr, start: ns, sweep: nsw }) => {
            let at = |c: DVec2, r: f64, start: f64, sweep: f64, t: f64| DVec2::new(c.x + r * (start + t * sweep).cos(), c.y + r * (start + t * sweep).sin());
            d(at(wc, wr, ws, wsw, 0.0), at(nc, nr, ns, nsw, 0.0)).max(d(at(wc, wr, ws, wsw, 0.5), at(nc, nr, ns, nsw, 0.5))).max(d(at(wc, wr, ws, wsw, 1.0), at(nc, nr, ns, nsw, 1.0)))
        }
        (Shape::Circle { c: wc, r: wr }, Shape::Circle { c: nc, r: nr }) => d(wc, nc).max((wr - nr).abs()),
        _ => f64::INFINITY,
    }
}

/// How far a piece's own curve moved, mm, what its ends did aside: a line cut back along itself
/// or an arc trimmed on its circle moved nothing, as a corner rounded does to them. Infinity
/// for a piece of another kind.
pub fn curve_moved(was: &Entity, now: &Entity) -> f64 {
    match (was.shape.clone(), now.shape.clone()) {
        (Shape::Line { a: wa, b: wb }, Shape::Line { a: na, b: nb }) => {
            let dx = wb.x - wa.x;
            let dy = wb.y - wa.y;
            let l = dx.hypot(dy);
            // Collapsed to a point, it has moved however it lies.
            if l < 1e-9 || (nb.x - na.x).hypot(nb.y - na.y) < 1e-6 {
                return retake_moved(was, now);
            }
            let off = |p: DVec2| ((p.x - wa.x) * dy - (p.y - wa.y) * dx).abs() / l;
            off(na).max(off(nb))
        }
        (Shape::Arc { c: wc, r: wr, .. }, Shape::Arc { c: nc, r: nr, .. }) => len2(wc - nc) + (wr - nr).abs(),
        _ => retake_moved(was, now),
    }
}

/// The distance from a point to a segment.
fn dist_to_segment(p: DVec2, a: DVec2, b: DVec2) -> f64 {
    let ab = b - a;
    let ll = ab.dot(ab);
    let t = if ll > 0.0 { ((p - a).dot(ab) / ll).clamp(0.0, 1.0) } else { 0.0 };
    len2(p - (a + ab * t))
}

/// The projectable nearest a sheet point within reach, or `None`.
pub fn projectable_at(candidates: &[Projectable], spot: DVec2, radius: f64) -> Option<&Projectable> {
    let mut best: Option<&Projectable> = None;
    let mut best_d = radius;
    for c in candidates {
        let p = &c.polyline;
        if p.len() == 1 {
            let d = len2(p[0] - spot);
            if d <= best_d {
                best_d = d;
                best = Some(c);
            }
            continue;
        }
        for w in p.windows(2) {
            let d = dist_to_segment(spot, w[0], w[1]);
            if d <= best_d {
                best_d = d;
                best = Some(c);
            }
        }
    }
    best
}

/// How far a projected piece taken again has to have moved, millimetres, for the sketch to be
/// solved again on it: a hair under what the kernel tells apart, so the same edge taken again
/// is the same piece.
pub const RETAKE_MOVED_MM: f64 = 1e-7;
/// How close a projected piece with no source has to lie to an edge of its face to be given
/// that edge as its source, millimetres.
pub const ADOPT_MM: f64 = 1e-5;

/// A sketch's projected pieces as a build places them (ScanRuler's `placeSketch`): each taken
/// again from its source, the sketch solved again where any moved.
#[derive(Debug, Clone, Default)]
pub struct PlacedProjections {
    /// The sketch as placed, when it differs from the one stored.
    pub sketch: Option<crate::sketch::Sketch>,
    /// What the sketch's row says: a source no longer found, relations no longer met.
    pub notes: Vec<String>,
}

/// Take every projected piece of `stored` again from its source, on a sketch lying in `frame`.
/// `edges_of` hands a body's edges and the tolerance its picks are matched within; `other` a
/// sketch earlier in the list, as it lies now (its frame and its pieces). `face` is the body
/// and the edges of the face the sketch lies on, found in this build: a projected piece of its
/// outline saved with no source is given the edge it lies on, from then on following it. A
/// face or a source no longer found leaves the piece where it was, and says so.
pub fn place_projections(
    stored: &crate::sketch::Sketch,
    frame: &SectionFrame,
    face: Option<(BodyId, &[EdgeInfo])>,
    mut edges_of: impl FnMut(BodyId) -> Option<(std::sync::Arc<crate::topo::Topology>, crate::topo::MatchTolerance)>,
    other: impl Fn(crate::model::FeatureId) -> Option<(SectionFrame, crate::sketch::Sketch)>,
) -> PlacedProjections {
    if !stored.entities.iter().any(|e| e.projected) {
        return PlacedProjections::default();
    }
    let mut notes = Vec::new();
    let mut sketch = stored.clone();
    // The outline of a face projected before projections followed the part has no source:
    // each piece of it is given the edge of the face it lies on exactly.
    let mut adopted = false;
    if let Some((body, outline)) = face.filter(|(_, o)| !o.is_empty()) {
        for e in sketch.entities.iter_mut() {
            if !e.projected || e.source.is_some() {
                continue;
            }
            let edge = outline.iter().find(|x| match retake(e, project_edge(ProjectedEdge::from(*x), frame).as_ref()) {
                Some(there) => retake_moved(e, &there) <= ADOPT_MM,
                None => false,
            });
            if let Some(edge) = edge {
                adopted = true;
                e.source = Some(ProjectionSource::Edge { body, edge: edge.fingerprint.clone() });
            }
        }
    }
    // Each edge is taken for one source at most, the first first, as the picks of a feature
    // are; an edge projected twice is one source, found once for both.
    let mut bodies: std::collections::HashMap<BodyId, Option<(std::sync::Arc<crate::topo::Topology>, crate::topo::MatchTolerance, std::collections::HashSet<u32>)>> = std::collections::HashMap::new();
    let mut matched: std::collections::HashMap<String, Option<EdgeInfo>> = std::collections::HashMap::new();
    let mut lost = 0usize;
    let mut moved = false;
    for i in 0..sketch.entities.len() {
        let e = &sketch.entities[i];
        let Some(source) = e.source.clone().filter(|_| e.projected) else { continue };
        let fresh: Option<Entity> = match &source {
            ProjectionSource::Sketch { sketch: id, entity } => other(*id).and_then(|(from, s)| s.entity(*entity).and_then(|piece| project_sketch_entity(piece, &from, frame))),
            ProjectionSource::Edge { body, edge } => {
                let key = format!("{body}:{}", serde_json::to_string(edge).unwrap_or_default());
                if !matched.contains_key(&key) {
                    let entry = bodies.entry(*body).or_insert_with(|| edges_of(*body).map(|(t, tol)| (t, tol, std::collections::HashSet::new())));
                    let found = entry.as_mut().and_then(|(topology, tol, taken)| {
                        let open: Vec<&EdgeInfo> = topology.edges.iter().filter(|x| !taken.contains(&x.hash)).collect();
                        let prints: Vec<EdgeFingerprint> = open.iter().map(|x| x.fingerprint.clone()).collect();
                        let j = crate::topo::match_edge(edge, &prints, tol)?;
                        taken.insert(open[j].hash);
                        Some(open[j].clone())
                    });
                    matched.insert(key.clone(), found);
                }
                matched.get(&key).cloned().flatten().and_then(|x| project_edge(ProjectedEdge::from(&x), frame))
            }
        };
        let e = &sketch.entities[i];
        match retake(e, fresh.as_ref()) {
            None => lost += 1,
            Some(now) => {
                if retake_moved(e, &now) > RETAKE_MOVED_MM {
                    moved = true;
                }
                sketch.entities[i] = now;
            }
        }
    }
    if lost > 0 {
        notes.push(if lost == 1 { "A projected piece no longer found on the part — left where it was.".to_string() } else { format!("{lost} projected pieces no longer found on the part — left where they were.") });
    }
    if moved {
        let traces = crate::datums::datum_traces(frame);
        let report = crate::solver::solve_sketch(&mut sketch, crate::solver::SolveOptions { datums: &traces, ..Default::default() });
        let n = report.unmet.len();
        if n > 0 {
            notes.push(format!("The projected pieces moved, and {} cannot be met where they are now.", if n == 1 { "a relation or a dimension".to_string() } else { format!("{n} relations and dimensions") }));
        }
    }
    PlacedProjections { sketch: (moved || adopted).then_some(sketch), notes }
}

/// The polyline a projectable is clicked against: a point is itself, anything else the piece
/// drawn at 64 steps a turn.
pub fn projectable_polyline(entity: &Entity) -> Vec<DVec2> {
    match entity.shape {
        Shape::Point { p } => vec![p],
        _ => polyline(&entity.shape, 64),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regions::find_regions;
    use crate::sketch::Sketch;
    use std::f64::consts::{PI, SQRT_2};

    fn v3(x: f64, y: f64, z: f64) -> DVec3 {
        DVec3::new(x, y, z)
    }

    fn v2(x: f64, y: f64) -> DVec2 {
        DVec2::new(x, y)
    }

    /// A sketch plane at z = 10, its X along the part's.
    fn top() -> SectionFrame {
        SectionFrame { origin: v3(0.0, 0.0, 10.0), normal: DVec3::Z, u: DVec3::X, v: DVec3::Y }
    }

    /// Points as the TS test hands them in: through 32-bit floats.
    fn flat(points: &[DVec3]) -> Vec<DVec3> {
        points.iter().map(|p| v3(p.x as f32 as f64, p.y as f32 as f64, p.z as f32 as f64)).collect()
    }

    struct Edge {
        fingerprint: EdgeFingerprint,
        polyline: Vec<DVec3>,
    }

    impl Edge {
        fn view(&self) -> ProjectedEdge<'_> {
            ProjectedEdge { fingerprint: &self.fingerprint, polyline: &self.polyline, ends: None }
        }
    }

    fn line_edge(a: DVec3, b: DVec3) -> Edge {
        Edge { fingerprint: EdgeFingerprint { kind: EdgeKind::Line, mid: (a + b) / 2.0, dir: DVec3::X, length: 1.0, radius: None, faces: Vec::new(), origin: 2 }, polyline: flat(&[a, b]) }
    }

    /// A circular edge about `axis` through `centre`, from `from` to `to` radians, sampled as
    /// the kernel would.
    fn circle_edge(centre: DVec3, axis_z: bool, r: f64, from: f64, to: f64) -> Edge {
        let n = 48;
        let points: Vec<DVec3> = (0..=n)
            .map(|i| {
                let t = from + (to - from) * i as f64 / n as f64;
                if axis_z {
                    v3(centre.x + r * t.cos(), centre.y + r * t.sin(), centre.z)
                } else {
                    v3(centre.x, centre.y + r * t.cos(), centre.z + r * t.sin())
                }
            })
            .collect();
        Edge {
            fingerprint: EdgeFingerprint { kind: EdgeKind::Circle, mid: points[n >> 1], dir: if axis_z { DVec3::Z } else { DVec3::X }, length: r * (to - from), radius: Some(r), faces: Vec::new(), origin: 2 },
            polyline: flat(&points),
        }
    }

    #[test]
    fn lays_a_straight_edge_down_as_a_fixed_line_and_one_seen_end_on_as_a_point() {
        let e = project_edge(line_edge(v3(0.0, 0.0, 0.0), v3(40.0, 0.0, 0.0)).view(), &top()).expect("a line");
        assert!(e.fixed && e.projected);
        assert_eq!(e.shape, Shape::Line { a: v2(0.0, 0.0), b: v2(40.0, 0.0) });
        // An upright edge of the box, looked down on.
        let p = project_edge(line_edge(v3(40.0, 20.0, 0.0), v3(40.0, 20.0, 10.0)).view(), &top()).expect("a point");
        assert_eq!(p.shape, Shape::Point { p: v2(40.0, 20.0) });
    }

    #[test]
    fn takes_a_circle_flat_to_the_plane_as_a_circle_an_arc_as_an_arc_and_refuses_one_at_a_slant() {
        let close = |a: f64, b: f64, d: i32| (a - b).abs() < 0.5 * 10f64.powi(-d);
        let Some(Shape::Circle { c, r }) = project_edge(circle_edge(v3(10.0, 10.0, 0.0), true, 3.0, 0.0, 2.0 * PI).view(), &top()).map(|e| e.shape) else { panic!("not a circle") };
        assert!(close(c.x, 10.0, 6) && close(c.y, 10.0, 6) && close(r, 3.0, 6));
        let Some(Shape::Arc { r, sweep, .. }) = project_edge(circle_edge(v3(0.0, 0.0, 10.0), true, 5.0, 0.0, PI / 2.0).view(), &top()).map(|e| e.shape) else { panic!("not an arc") };
        assert!(close(r, 5.0, 6) && close(sweep, PI / 2.0, 6));
        // Square to the plane it is seen as the line it spans.
        let Some(Shape::Line { a, b }) = project_edge(circle_edge(v3(0.0, 0.0, 5.0), false, 4.0, 0.0, 2.0 * PI).view(), &top()).map(|e| e.shape) else { panic!("not a line") };
        assert!(close((b.y - a.y).abs(), 8.0, 3));
        // Tilted, it is an ellipse: no piece for that.
        let h = SQRT_2 / 2.0;
        let tilted = SectionFrame { origin: DVec3::ZERO, normal: v3(0.0, h, h), u: DVec3::X, v: v3(0.0, h, -h) };
        assert!(project_edge(circle_edge(DVec3::ZERO, true, 3.0, 0.0, 2.0 * PI).view(), &tilted).is_none());
    }

    #[test]
    fn projects_the_outline_of_a_face_with_a_hole_into_a_profile_that_closes_the_hole_a_loop_of_its_own() {
        let z = 10.0;
        let edges = [
            line_edge(v3(0.0, 0.0, z), v3(40.0, 0.0, z)),
            line_edge(v3(40.0, 0.0, z), v3(40.0, 20.0, z)),
            line_edge(v3(40.0, 20.0, z), v3(0.0, 20.0, z)),
            line_edge(v3(0.0, 20.0, z), v3(0.0, 0.0, z)),
            circle_edge(v3(10.0, 10.0, z), true, 3.0, 0.0, 2.0 * PI),
        ];
        let views: Vec<ProjectedEdge> = edges.iter().map(|e| e.view()).collect();
        let (entities, skipped) = project_face_outline(&views, &top(), None);
        assert_eq!(skipped, 0);
        assert_eq!(entities.len(), 5);
        assert!(entities.iter().all(|e| e.source.is_none()));
        // Given the body, each piece keeps its edge to follow it.
        let body = BodyId::new(2);
        let (linked, _) = project_face_outline(&views, &top(), Some(body));
        let sources: Vec<Option<ProjectionSource>> = linked.iter().map(|e| e.source.clone()).collect();
        let wanted: Vec<Option<ProjectionSource>> = edges.iter().map(|e| Some(ProjectionSource::Edge { body, edge: e.fingerprint.clone() })).collect();
        assert_eq!(sources, wanted);
        let mut sketch = Sketch::default();
        for e in entities {
            sketch.add(e);
        }
        let regions = find_regions(&sketch);
        let outer = regions.iter().max_by(|a, b| a.area.total_cmp(&b.area)).expect("a region");
        assert!((outer.area - (40.0 * 20.0 - PI * 9.0)).abs() < 0.05);
        assert_eq!(outer.holes.len(), 1);
    }

    #[test]
    fn brings_a_piece_of_another_sketch_across_from_its_plane_onto_this_one() {
        // A sketch on the XZ plane: its line along X projects onto the top as the same line;
        // its circle is seen edge on.
        let side = SectionFrame { origin: DVec3::ZERO, normal: v3(0.0, -1.0, 0.0), u: DVec3::X, v: DVec3::Z };
        let line = Entity::new(Shape::Line { a: v2(5.0, 2.0), b: v2(25.0, 2.0) });
        let p = project_sketch_entity(&line, &side, &top()).expect("a line");
        let Shape::Line { a, b } = p.shape else { panic!("not a line") };
        let mut xs = [a.x, b.x];
        xs.sort_by(f64::total_cmp);
        assert_eq!(xs, [5.0, 25.0]);
        assert!(p.projected);
        let circle = Entity::new(Shape::Circle { c: v2(10.0, 5.0), r: 2.0 });
        assert!(matches!(project_sketch_entity(&circle, &side, &top()).map(|e| e.shape), Some(Shape::Line { .. })));
        let Some(Shape::Circle { r, .. }) = project_sketch_entity(&circle, &top(), &top()).map(|e| e.shape) else { panic!("not a circle") };
        assert!((r - 2.0).abs() < 1e-9);
    }

    #[test]
    fn takes_a_projected_piece_again_where_its_source_is_now_keeping_the_piece_and_the_ends_its_relations_hang_on() {
        let mut was = Entity::new(Shape::Line { a: v2(0.0, 0.0), b: v2(40.0, 0.0) });
        was.id = 7;
        was.fixed = true;
        was.projected = true;
        was.source = Some(ProjectionSource::Sketch { sketch: 1, entity: 2 });
        // The edge found again runs the other way: the ends are kept as they were, so a corner
        // joined to `a` stays joined to the same corner.
        let fresh = fixed(Shape::Line { a: v2(30.0, 2.0), b: v2(0.0, 2.0) });
        let now = retake(&was, Some(&fresh)).expect("taken again");
        assert_eq!(now, Entity { shape: Shape::Line { a: v2(0.0, 2.0), b: v2(30.0, 2.0) }, ..was.clone() });
        assert!((retake_moved(&was, &now) - 10f64.hypot(2.0)).abs() < 1e-9);
        let same = Entity::new(Shape::Line { a: v2(0.0, 0.0), b: v2(40.0, 0.0) });
        assert_eq!(retake_moved(&was, &retake(&was, Some(&same)).expect("taken")), 0.0);
        // A source that projects to another kind of piece now is not taken.
        assert!(retake(&was, Some(&Entity::new(Shape::Point { p: v2(0.0, 0.0) }))).is_none());
        assert!(retake(&was, None).is_none());
        // An arc a turn apart in its start is the same arc.
        let mut arc = Entity::new(Shape::Arc { c: v2(0.0, 0.0), r: 5.0, start: PI - 0.1, sweep: 0.5 });
        arc.id = 8;
        arc.projected = true;
        arc.fixed = true;
        let turned = Entity::new(Shape::Arc { c: v2(0.0, 0.0), r: 5.0, start: -PI - 0.1, sweep: 0.5 });
        assert!(retake_moved(&arc, &retake(&arc, Some(&turned)).expect("taken")) < 1e-9);
    }

    #[test]
    fn finds_what_the_hand_is_on_within_its_reach() {
        let candidates = vec![
            Projectable { entity: Entity::new(Shape::Line { a: v2(0.0, 0.0), b: v2(40.0, 0.0) }), polyline: vec![v2(0.0, 0.0), v2(40.0, 0.0)], source: "an edge".into() },
            Projectable { entity: Entity::new(Shape::Point { p: v2(40.0, 20.0) }), polyline: vec![v2(40.0, 20.0)], source: "an upright edge".into() },
        ];
        assert_eq!(projectable_at(&candidates, v2(20.0, 0.4), 1.0).map(|c| c.source.as_str()), Some("an edge"));
        assert_eq!(projectable_at(&candidates, v2(39.7, 19.8), 1.0).map(|c| c.source.as_str()), Some("an upright edge"));
        assert!(projectable_at(&candidates, v2(20.0, 5.0), 1.0).is_none());
    }

    /// Not a ScanRuler test (ScanRuler's is a kernel build): a piece projected from another
    /// sketch follows that sketch's piece when it moves, the drawn piece joined to it following
    /// through its coincidence; a source gone is said, the piece left where it was.
    #[test]
    fn a_projection_follows_its_source_and_the_sketch_is_solved_again() {
        use crate::sketch::{Anchor, AnchorAt, Rule, Sketch};
        let low = SectionFrame { origin: DVec3::ZERO, normal: DVec3::Z, u: DVec3::X, v: DVec3::Y };
        let high = SectionFrame { origin: v3(0.0, 0.0, 10.0), ..low };
        let mut source = Sketch::default();
        let line = source.add_shape(Shape::Line { a: v2(0.0, 0.0), b: v2(40.0, 0.0) });
        let mut sketch = Sketch::default();
        let mut taken = project_sketch_entity(source.entity(line).expect("the line"), &low, &high).expect("projected");
        taken.source = Some(ProjectionSource::Sketch { sketch: 1, entity: line });
        let p = sketch.add(taken);
        let drawn = sketch.add_shape(Shape::Line { a: v2(40.0, 0.0), b: v2(40.0, 20.0) });
        sketch.constrain(Rule::Coincident { p: Anchor::new(p, AnchorAt::B), q: Anchor::new(drawn, AnchorAt::A) });
        // Unmoved: nothing to place again.
        let same = place_projections(&sketch, &high, None, |_| None, |_| Some((low, source.clone())));
        assert!(same.sketch.is_none() && same.notes.is_empty());
        // The source moved up 5 mm: the projection with it, the drawn line's end after it.
        if let Some(e) = source.entity_mut(line) {
            e.shape = Shape::Line { a: v2(0.0, 5.0), b: v2(40.0, 5.0) };
        }
        let placed = place_projections(&sketch, &high, None, |_| None, |_| Some((low, source.clone()))).sketch.expect("moved");
        assert_eq!(placed.entity(p).map(|e| e.shape.clone()), Some(Shape::Line { a: v2(0.0, 5.0), b: v2(40.0, 5.0) }));
        let Some(Shape::Line { a, .. }) = placed.entity(drawn).map(|e| e.shape.clone()) else { panic!("the drawn line") };
        assert!((a - v2(40.0, 5.0)).length() < 1e-6, "{a}");
        // The source gone: said on the row, the piece where it was.
        let lost = place_projections(&sketch, &high, None, |_| None, |_| None);
        assert!(lost.sketch.is_none());
        assert_eq!(lost.notes, vec!["A projected piece no longer found on the part — left where it was.".to_string()]);
    }

    /// Not a ScanRuler test: a projected piece taken from a sketch keeps its source through a
    /// JSON round trip, in ScanRuler's shape.
    #[test]
    fn a_projection_source_saves_in_the_shape_scanruler_writes() {
        let mut e = Entity::new(Shape::Line { a: v2(0.0, 0.0), b: v2(1.0, 0.0) });
        e.source = Some(ProjectionSource::Sketch { sketch: 3, entity: 9 });
        let json = serde_json::to_value(&e).expect("json");
        assert_eq!(json["source"], serde_json::json!({ "sketch": 3, "entity": 9 }));
        let back: Entity = serde_json::from_value(json).expect("read back");
        assert_eq!(back.source, e.source);
    }
}
