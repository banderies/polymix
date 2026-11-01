// SPDX-License-Identifier: AGPL-3.0-only
//! What the part says about what was clicked on it: a face's area and normal, an edge's length,
//! a corner's coordinates — and, with two things picked, what lies between them: the distance
//! between two corners, from a corner to an edge or a face, between parallel faces or edges, the
//! angle between those that are not. The measuring a CAD program's Inspect does, read off the
//! bodies' own topology, in the part's coordinates. Plain data in, rows of text out: the chips by
//! the click show them.
//!
//! Ported from ScanRuler (`plugins/reverse/core/selectionInfo.ts`), © Stefan Hermann
//! (CNC Kitchen), AGPL-3.0-only.

use glam::DVec3;

use crate::display_units::{fmt_area, fmt_len, DisplayUnits, FmtOptions};
use crate::js_number::to_fixed;
use crate::model::BodyId;
use crate::topo::{EdgeInfo, EdgeKind, FaceInfo, FaceKind};

/// One thing picked on a body: a face (with the edges bounding it — its outline), an edge, or
/// a corner — an end of an edge.
#[derive(Debug, Clone, PartialEq)]
pub enum PickedItem {
    Face { body: BodyId, face: FaceInfo, edges: Vec<EdgeInfo> },
    Edge { body: BodyId, edge: EdgeInfo },
    Vertex { body: BodyId, point: DVec3 },
}

impl PickedItem {
    /// The body the pick is on.
    pub fn body(&self) -> BodyId {
        match self {
            PickedItem::Face { body, .. } | PickedItem::Edge { body, .. } | PickedItem::Vertex { body, .. } => *body,
        }
    }

    /// The pick's kind as a note or a chip names it: face, edge or corner.
    pub fn word(&self) -> &'static str {
        match self {
            PickedItem::Face { .. } => "face",
            PickedItem::Edge { .. } => "edge",
            PickedItem::Vertex { .. } => "corner",
        }
    }
}

/// One row of readings: a label and its value.
#[derive(Debug, Clone, PartialEq)]
pub struct InfoRow {
    pub label: String,
    pub value: String,
}

/// The title and the rows of a pick's reading card.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectionInfo {
    pub title: String,
    pub rows: Vec<InfoRow>,
}

fn row(label: &str, value: String) -> InfoRow {
    InfoRow { label: label.to_string(), value }
}

/// What tells two picks apart: the body and the face's or edge's hash, a corner's place to a
/// micron.
pub fn item_key(i: &PickedItem) -> String {
    match i {
        PickedItem::Face { body, face, .. } => format!("f:{}:{}", body.as_number(), face.hash),
        PickedItem::Edge { body, edge } => format!("e:{}:{}", body.as_number(), edge.hash),
        PickedItem::Vertex { body, point } => format!("v:{}:{},{},{}", body.as_number(), (point.x * 1000.0).round(), (point.y * 1000.0).round(), (point.z * 1000.0).round()),
    }
}

/// The figures a row reads, in the units shown: lengths and coordinates to a micron in
/// millimetres (inches get their own decimals), areas to a hundredth of a square millimetre.
struct Words {
    units: DisplayUnits,
}

impl Words {
    fn mm(&self, v: f64) -> String {
        fmt_len(v, self.units, 3, FmtOptions::default())
    }

    fn area(&self, v: f64) -> String {
        fmt_area(v, self.units, 2, FmtOptions::default())
    }

    fn xyz(&self, p: DVec3) -> String {
        [p.x, p.y, p.z].iter().map(|c| fmt_len(*c, self.units, 3, FmtOptions { bare: true, signed: false })).collect::<Vec<_>>().join(", ")
    }
}

fn deg(v: f64) -> String {
    format!("{}°", to_fixed(v, 2))
}

/// A direction with its noise put away: −0.000 reads as 0.000.
fn unit(d: DVec3) -> String {
    [d.x, d.y, d.z].iter().map(|c| to_fixed(if c.abs() < 5e-4 { 0.0 } else { *c }, 3)).collect::<Vec<_>>().join(", ")
}

/// Parallel within a twentieth of a degree — what a modelled part is, and a lofted or drafted
/// one is not.
fn parallel() -> f64 {
    (0.05f64).to_radians().cos()
}

/// The centre of a round edge, off three of its drawn points — the fingerprint carries the
/// radius and the axis but not the centre.
pub fn circle_centre(edge: &EdgeInfo) -> Option<DVec3> {
    let p = &edge.polyline;
    let n = p.len();
    if n < 3 {
        return None;
    }
    let a = p[0];
    let b = p[n / 3];
    let c = p[(2 * n) / 3];
    let ab = b - a;
    let ac = c - a;
    let nrm = ab.cross(ac);
    let n2 = nrm.dot(nrm);
    if n2 < 1e-18 {
        return None;
    }
    // The circumcentre of the triangle a, b, c.
    let t = nrm.cross(ab) * ac.dot(ac) + ac.cross(nrm) * ab.dot(ab);
    Some(a + t * (1.0 / (2.0 * n2)))
}

/// A straight carrier a pick stands for: a line edge, or the axis of a round face or a round
/// edge.
struct Line {
    point: DVec3,
    dir: DVec3,
    axis: bool,
}

fn line_of(i: &PickedItem) -> Option<Line> {
    match i {
        PickedItem::Edge { edge, .. } => {
            let f = &edge.fingerprint;
            match f.kind {
                EdgeKind::Line => Some(Line { point: f.mid, dir: f.dir.try_normalize().unwrap_or(DVec3::Z), axis: false }),
                EdgeKind::Circle => {
                    let c = circle_centre(edge)?;
                    let d = f.dir.try_normalize()?;
                    Some(Line { point: c, dir: d, axis: true })
                }
                EdgeKind::Other => None,
            }
        }
        PickedItem::Face { face, .. } => {
            let f = &face.fingerprint;
            let d = f.dir.try_normalize()?;
            match (f.kind, f.center) {
                (FaceKind::Cylinder | FaceKind::Cone, Some(c)) => Some(Line { point: c, dir: d, axis: true }),
                _ => None,
            }
        }
        PickedItem::Vertex { .. } => None,
    }
}

fn plane_of(i: &PickedItem) -> Option<(DVec3, DVec3)> {
    match i {
        PickedItem::Face { face, .. } if face.fingerprint.kind == FaceKind::Plane => face.fingerprint.dir.try_normalize().map(|n| (face.fingerprint.centroid, n)),
        _ => None,
    }
}

fn point_of(i: &PickedItem) -> Option<DVec3> {
    match i {
        PickedItem::Vertex { point, .. } => Some(*point),
        // A sphere stands for its centre.
        PickedItem::Face { face, .. } if face.fingerprint.kind == FaceKind::Sphere => face.fingerprint.center,
        _ => None,
    }
}

fn point_to_line(p: DVec3, l: &Line) -> f64 {
    let w = p - l.point;
    (w - l.dir * w.dot(l.dir)).length()
}

/// The title one pick's card carries.
fn name_of(i: &PickedItem) -> &'static str {
    match i {
        PickedItem::Vertex { .. } => "Point",
        PickedItem::Edge { edge, .. } => match edge.fingerprint.kind {
            EdgeKind::Line => "Straight edge",
            EdgeKind::Circle => "Round edge",
            EdgeKind::Other => "Edge",
        },
        PickedItem::Face { face, .. } => match face.fingerprint.kind {
            FaceKind::Plane => "Flat face",
            FaceKind::Cylinder => "Cylindrical face",
            FaceKind::Cone => "Conical face",
            FaceKind::Sphere => "Spherical face",
            FaceKind::Torus => "Toroidal face",
            FaceKind::Other => "Freeform face",
        },
    }
}

fn single(i: &PickedItem, w: &Words) -> Vec<InfoRow> {
    match i {
        PickedItem::Vertex { point, .. } => vec![row("X", w.mm(point.x)), row("Y", w.mm(point.y)), row("Z", w.mm(point.z))],
        PickedItem::Edge { edge, .. } => {
            let f = &edge.fingerprint;
            let mut rows = vec![row("Length", w.mm(f.length))];
            match (f.kind, f.radius) {
                (EdgeKind::Line, _) => {
                    rows.push(row("Direction", unit(f.dir.try_normalize().unwrap_or(f.dir))));
                    rows.push(row("From", w.xyz(edge.ends[0])));
                    rows.push(row("To", w.xyz(edge.ends[1])));
                }
                (EdgeKind::Circle, Some(r)) => {
                    rows.push(row("Radius", w.mm(r)));
                    rows.push(row("Diameter", w.mm(2.0 * r)));
                    if let Some(c) = circle_centre(edge) {
                        rows.push(row("Centre", w.xyz(c)));
                    }
                    rows.push(row("Axis", unit(f.dir.try_normalize().unwrap_or(f.dir))));
                }
                _ => rows.push(row("Middle", w.xyz(f.mid))),
            }
            rows
        }
        PickedItem::Face { face, .. } => {
            let f = &face.fingerprint;
            let mut rows = vec![row("Area", w.area(f.area))];
            if f.kind == FaceKind::Plane {
                rows.push(row("Normal", unit(f.dir.try_normalize().unwrap_or(f.dir))));
                rows.push(row("Centre", w.xyz(f.centroid)));
            } else {
                if let Some(r) = f.radius {
                    rows.push(row("Radius", w.mm(r)));
                    rows.push(row("Diameter", w.mm(2.0 * r)));
                }
                match (f.kind, f.center) {
                    (FaceKind::Sphere, Some(c)) => rows.push(row("Centre", w.xyz(c))),
                    (FaceKind::Cylinder | FaceKind::Cone | FaceKind::Torus, _) => rows.push(row("Axis", unit(f.dir.try_normalize().unwrap_or(f.dir)))),
                    _ => {}
                }
                if let (FaceKind::Cylinder | FaceKind::Cone, Some(c)) = (f.kind, f.center) {
                    rows.push(row("On axis", w.xyz(c)));
                }
            }
            rows
        }
    }
}

/// What lies between two picks, or nothing the part can say.
fn between(a: &PickedItem, b: &PickedItem, w: &Words) -> Vec<InfoRow> {
    let pa = point_of(a);
    let pb = point_of(b);
    if let (Some(pa), Some(pb)) = (pa, pb) {
        let d = pb - pa;
        return vec![row("Distance", w.mm(d.length())), row("ΔX", w.mm(d.x.abs())), row("ΔY", w.mm(d.y.abs())), row("ΔZ", w.mm(d.z.abs()))];
    }
    // A point against a plane or a line.
    if let Some(point) = pa.or(pb) {
        let other = if pa.is_some() { b } else { a };
        if let Some((p, n)) = plane_of(other) {
            return vec![row("Distance to the face", w.mm((point - p).dot(n).abs()))];
        }
        if let Some(line) = line_of(other) {
            return vec![row(if line.axis { "Distance to the axis" } else { "Distance to the edge" }, w.mm(point_to_line(point, &line)))];
        }
        return Vec::new();
    }
    let plane_a = plane_of(a);
    let plane_b = plane_of(b);
    if let (Some((pa, na)), Some((pb, nb))) = (plane_a, plane_b) {
        let c = na.dot(nb).abs();
        if c >= parallel() {
            return vec![row("Distance", w.mm((pb - pa).dot(na).abs())), row("Angle", "parallel".into())];
        }
        return vec![row("Angle", deg(c.min(1.0).acos().to_degrees()))];
    }
    let line_a = line_of(a);
    let line_b = line_of(b);
    let plane = plane_a.or(plane_b);
    let line = if plane_a.is_some() {
        line_of(b)
    } else if plane_b.is_some() {
        line_of(a)
    } else {
        None
    };
    if let (Some((pp, pn)), Some(line)) = (plane, line) {
        let s = line.dir.dot(pn).abs();
        // Along the face: a distance. Across it: the angle the line leaves it at.
        if s <= (0.05f64).to_radians().sin() {
            return vec![row("Distance", w.mm((line.point - pp).dot(pn).abs())), row("Angle", "parallel".into())];
        }
        return vec![row("Angle to the face", deg(s.min(1.0).asin().to_degrees()))];
    }
    if let (Some(la), Some(lb)) = (line_a, line_b) {
        let c = la.dir.dot(lb.dir).abs();
        let word = if la.axis && lb.axis { "Distance between the axes" } else { "Distance" };
        if c >= parallel() {
            return vec![row(word, w.mm(point_to_line(lb.point, &la))), row("Angle", "parallel".into())];
        }
        // Askew or crossing: the angle, and how far apart they pass.
        let gap = la.dir.cross(lb.dir).try_normalize().map_or(0.0, |n| (lb.point - la.point).dot(n).abs());
        let cross = gap < 1e-6;
        return vec![row("Angle", deg(c.min(1.0).acos().to_degrees())), row(if cross { "They cross" } else { "Closest" }, if cross { "in one point".into() } else { w.mm(gap) })];
    }
    Vec::new()
}

/// The rows for what is picked: one thing in full; two, what lies between them; more, the sums.
pub fn selection_info(items: &[PickedItem], units: DisplayUnits) -> Option<SelectionInfo> {
    let w = Words { units };
    if items.is_empty() {
        return None;
    }
    if items.len() == 1 {
        return Some(SelectionInfo { title: name_of(&items[0]).to_string(), rows: single(&items[0], &w) });
    }
    let edges: Vec<&EdgeInfo> = items.iter().filter_map(|i| if let PickedItem::Edge { edge, .. } = i { Some(edge) } else { None }).collect();
    let faces: Vec<&FaceInfo> = items.iter().filter_map(|i| if let PickedItem::Face { face, .. } = i { Some(face) } else { None }).collect();
    let points = items.iter().filter(|i| matches!(i, PickedItem::Vertex { .. })).count();
    let count = |n: usize, one: &str| match n {
        0 => String::new(),
        1 => format!("1 {one}"),
        _ => format!("{n} {one}s"),
    };
    let title = [count(faces.len(), "face"), count(edges.len(), "edge"), count(points, "point")].into_iter().filter(|s| !s.is_empty()).collect::<Vec<_>>().join(", ");
    let mut rows = Vec::new();
    if items.len() == 2 {
        rows.extend(between(&items[0], &items[1], &w));
    }
    if edges.len() > 1 || (edges.len() == 1 && items.len() > 2) {
        rows.push(row("Length, together", w.mm(edges.iter().map(|e| e.fingerprint.length).sum())));
    } else if edges.len() == 1 {
        rows.push(row("Edge length", w.mm(edges[0].fingerprint.length)));
    }
    if faces.len() > 1 || (faces.len() == 1 && items.len() > 2) {
        rows.push(row("Area, together", w.area(faces.iter().map(|f| f.fingerprint.area).sum())));
    } else if faces.len() == 1 {
        rows.push(row("Face area", w.area(faces[0].fingerprint.area)));
    }
    Some(SelectionInfo { title, rows })
}

/// A pick taken with Ctrl held, or a plain one: a plain click is a new selection; with Ctrl the
/// pick is added, or taken off again where it is held already (ScanRuler's `pickBodyItem`).
pub fn picked_with(held: &[PickedItem], item: PickedItem, additive: bool) -> Vec<PickedItem> {
    if !additive {
        return vec![item];
    }
    let k = item_key(&item);
    if held.iter().any(|i| item_key(i) == k) {
        held.iter().filter(|i| item_key(i) != k).cloned().collect()
    } else {
        let mut next = held.to_vec();
        next.push(item);
        next
    }
}

/// What a window over the bodies adds to the picks held: whatever of it is new, in order
/// (ScanRuler's `addBodyItems`).
pub fn with_items(held: &[PickedItem], items: Vec<PickedItem>) -> Vec<PickedItem> {
    let mut seen: std::collections::HashSet<String> = held.iter().map(item_key).collect();
    let mut next = held.to_vec();
    for i in items {
        if seen.insert(item_key(&i)) {
            next.push(i);
        }
    }
    next
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topo::{EdgeFingerprint, FaceFingerprint};
    use std::cell::Cell;

    thread_local! {
        static HASH: Cell<u32> = const { Cell::new(1) };
    }

    fn next_hash() -> u32 {
        HASH.with(|h| {
            let v = h.get();
            h.set(v + 1);
            v
        })
    }

    fn v(x: f64, y: f64, z: f64) -> DVec3 {
        DVec3::new(x, y, z)
    }

    fn plane(centroid: DVec3, normal: DVec3, area: f64) -> PickedItem {
        let face = FaceInfo { hash: next_hash(), fingerprint: FaceFingerprint { kind: FaceKind::Plane, centroid, dir: normal, area, radius: None, center: None, origin: 1 }, full_turn: None, point: centroid };
        PickedItem::Face { body: BodyId::new(1), face, edges: Vec::new() }
    }

    fn cylinder(center: DVec3, axis: DVec3, radius: f64) -> PickedItem {
        let face = FaceInfo { hash: next_hash(), fingerprint: FaceFingerprint { kind: FaceKind::Cylinder, centroid: center, dir: axis, area: 2.0 * std::f64::consts::PI * radius * 10.0, radius: Some(radius), center: Some(center), origin: 1 }, full_turn: None, point: center };
        PickedItem::Face { body: BodyId::new(1), face, edges: Vec::new() }
    }

    fn line_edge(a: DVec3, b: DVec3) -> PickedItem {
        let d = b - a;
        let l = d.length();
        let edge = EdgeInfo { hash: next_hash(), fingerprint: EdgeFingerprint { kind: EdgeKind::Line, mid: (a + b) / 2.0, dir: d / l, length: l, radius: None, faces: Vec::new(), origin: 1 }, faces: Vec::new(), ends: [a, b], end_tangents: [d, d], polyline: vec![a, b] };
        PickedItem::Edge { body: BodyId::new(1), edge }
    }

    fn circle_edge(c: DVec3, r: f64) -> PickedItem {
        let pts: Vec<DVec3> = (0..=32).map(|i| v(c.x + r * (i as f64 * std::f64::consts::PI / 16.0).cos(), c.y + r * (i as f64 * std::f64::consts::PI / 16.0).sin(), c.z)).collect();
        let start = v(c.x + r, c.y, c.z);
        let edge = EdgeInfo { hash: next_hash(), fingerprint: EdgeFingerprint { kind: EdgeKind::Circle, mid: v(c.x - r, c.y, c.z), dir: DVec3::Z, length: 2.0 * std::f64::consts::PI * r, radius: Some(r), faces: Vec::new(), origin: 1 }, faces: Vec::new(), ends: [start, start], end_tangents: [DVec3::Y, DVec3::Y], polyline: pts };
        PickedItem::Edge { body: BodyId::new(1), edge }
    }

    fn corner(p: DVec3) -> PickedItem {
        PickedItem::Vertex { body: BodyId::new(1), point: p }
    }

    fn value(info: &SelectionInfo, label: &str) -> Option<String> {
        info.rows.iter().find(|r| r.label == label).map(|r| r.value.clone())
    }

    fn info(items: &[PickedItem]) -> SelectionInfo {
        selection_info(items, DisplayUnits::Mm).expect("a reading")
    }

    // ---- what one pick says ----

    #[test]
    fn a_flat_face_its_area_normal_and_place() {
        let i = info(&[plane(v(20.0, 10.0, 10.0), v(0.0, 0.0, 1.0), 800.0)]);
        assert_eq!(i.title, "Flat face");
        assert_eq!(value(&i, "Area").as_deref(), Some("800.00 mm²"));
        assert_eq!(value(&i, "Normal").as_deref(), Some("0.000, 0.000, 1.000"));
        assert_eq!(value(&i, "Centre").as_deref(), Some("20.000, 10.000, 10.000"));
    }

    #[test]
    fn a_round_face_a_straight_edge_and_a_round_edge() {
        let cyl = info(&[cylinder(v(5.0, 5.0, 0.0), v(0.0, 0.0, 1.0), 3.0)]);
        assert_eq!(cyl.title, "Cylindrical face");
        assert_eq!(value(&cyl, "Radius").as_deref(), Some("3.000 mm"));
        assert_eq!(value(&cyl, "Diameter").as_deref(), Some("6.000 mm"));
        let line = info(&[line_edge(v(0.0, 0.0, 0.0), v(40.0, 0.0, 0.0))]);
        assert_eq!(value(&line, "Length").as_deref(), Some("40.000 mm"));
        assert_eq!(value(&line, "To").as_deref(), Some("40.000, 0.000, 0.000"));
        let round = circle_edge(v(12.0, 8.0, 10.0), 4.0);
        let PickedItem::Edge { edge, .. } = &round else { panic!("an edge") };
        let c = circle_centre(edge).expect("a centre");
        assert!((c - v(12.0, 8.0, 10.0)).length() < 1e-6);
        assert_eq!(value(&info(&[round]), "Centre").as_deref(), Some("12.000, 8.000, 10.000"));
    }

    #[test]
    fn a_corner_its_coordinates() {
        let i = info(&[corner(v(40.0, 20.0, 10.0))]);
        assert_eq!(i.title, "Point");
        assert_eq!(i.rows.iter().map(|r| r.value.as_str()).collect::<Vec<_>>(), vec!["40.000 mm", "20.000 mm", "10.000 mm"]);
    }

    // ---- what lies between two picks ----

    #[test]
    fn two_corners_the_distance_and_its_parts() {
        let i = info(&[corner(v(0.0, 0.0, 0.0)), corner(v(40.0, 20.0, 10.0))]);
        assert_eq!(i.title, "2 points");
        assert_eq!(value(&i, "Distance"), Some(format!("{} mm", to_fixed((40f64 * 40.0 + 400.0 + 100.0).sqrt(), 3))));
        assert_eq!(value(&i, "ΔY").as_deref(), Some("20.000 mm"));
    }

    #[test]
    fn parallel_faces_their_distance_others_their_angle() {
        assert_eq!(value(&info(&[plane(v(0.0, 0.0, 0.0), v(0.0, 0.0, -1.0), 100.0), plane(v(5.0, 5.0, 10.0), v(0.0, 0.0, 1.0), 100.0)]), "Distance").as_deref(), Some("10.000 mm"));
        let s = std::f64::consts::FRAC_1_SQRT_2;
        assert_eq!(value(&info(&[plane(v(0.0, 0.0, 0.0), v(0.0, 0.0, 1.0), 100.0), plane(v(0.0, 0.0, 0.0), v(s, 0.0, s), 100.0)]), "Angle").as_deref(), Some("45.00°"));
    }

    #[test]
    fn a_corner_to_a_face_an_edge_and_an_axis() {
        assert_eq!(value(&info(&[corner(v(3.0, 4.0, 7.0)), plane(v(0.0, 0.0, 0.0), v(0.0, 0.0, 1.0), 100.0)]), "Distance to the face").as_deref(), Some("7.000 mm"));
        assert_eq!(value(&info(&[line_edge(v(0.0, 0.0, 0.0), v(10.0, 0.0, 0.0)), corner(v(5.0, 3.0, 4.0))]), "Distance to the edge").as_deref(), Some("5.000 mm"));
        assert_eq!(value(&info(&[corner(v(8.0, 9.0, 2.0)), cylinder(v(5.0, 5.0, 0.0), v(0.0, 0.0, 1.0), 3.0)]), "Distance to the axis").as_deref(), Some("5.000 mm"));
    }

    #[test]
    fn edges_parallel_crossing_askew_and_two_holes() {
        assert_eq!(value(&info(&[line_edge(v(0.0, 0.0, 0.0), v(10.0, 0.0, 0.0)), line_edge(v(0.0, 6.0, 8.0), v(10.0, 6.0, 8.0))]), "Distance").as_deref(), Some("10.000 mm"));
        let crossing = info(&[line_edge(v(0.0, 0.0, 0.0), v(10.0, 0.0, 0.0)), line_edge(v(0.0, 0.0, 0.0), v(0.0, 10.0, 0.0))]);
        assert_eq!(value(&crossing, "Angle").as_deref(), Some("90.00°"));
        assert_eq!(value(&crossing, "They cross").as_deref(), Some("in one point"));
        assert_eq!(value(&info(&[line_edge(v(0.0, 0.0, 0.0), v(10.0, 0.0, 0.0)), line_edge(v(0.0, 0.0, 5.0), v(0.0, 10.0, 5.0))]), "Closest").as_deref(), Some("5.000 mm"));
        assert_eq!(value(&info(&[cylinder(v(0.0, 0.0, 0.0), v(0.0, 0.0, 1.0), 3.0), circle_edge(v(30.0, 40.0, 10.0), 2.0)]), "Distance between the axes").as_deref(), Some("50.000 mm"));
    }

    #[test]
    fn an_edge_along_a_face_and_one_through_it() {
        assert_eq!(value(&info(&[line_edge(v(0.0, 0.0, 4.0), v(10.0, 0.0, 4.0)), plane(v(0.0, 0.0, 0.0), v(0.0, 0.0, 1.0), 100.0)]), "Distance").as_deref(), Some("4.000 mm"));
        assert_eq!(value(&info(&[line_edge(v(0.0, 0.0, 0.0), v(0.0, 0.0, 10.0)), plane(v(0.0, 0.0, 0.0), v(0.0, 0.0, 1.0), 100.0)]), "Angle to the face").as_deref(), Some("90.00°"));
    }

    #[test]
    fn more_than_two_what_they_add_up_to() {
        let i = info(&[line_edge(v(0.0, 0.0, 0.0), v(10.0, 0.0, 0.0)), line_edge(v(0.0, 0.0, 0.0), v(0.0, 20.0, 0.0)), line_edge(v(0.0, 0.0, 0.0), v(0.0, 0.0, 5.0)), plane(v(0.0, 0.0, 0.0), v(0.0, 0.0, 1.0), 50.0)]);
        assert_eq!(i.title, "1 face, 3 edges");
        assert_eq!(value(&i, "Length, together").as_deref(), Some("35.000 mm"));
        assert_eq!(value(&i, "Area, together").as_deref(), Some("50.00 mm²"));
    }

    // ---- the selection taking one more with Ctrl ----

    #[test]
    fn a_click_replaces_ctrl_click_adds_and_takes_off_a_window_adds_what_is_new() {
        let a = corner(v(0.0, 0.0, 0.0));
        let b = corner(v(10.0, 0.0, 0.0));
        let e = line_edge(v(0.0, 0.0, 0.0), v(10.0, 0.0, 0.0));
        let mut held = picked_with(&[], a.clone(), false);
        assert_eq!(held.len(), 1);
        held = picked_with(&held, b.clone(), true);
        assert_eq!(held.iter().map(item_key).collect::<Vec<_>>(), vec![item_key(&a), item_key(&b)]);
        held = picked_with(&held, a.clone(), true);
        assert_eq!(held, vec![b.clone()]);
        held = with_items(&held, vec![b.clone(), e.clone()]);
        assert_eq!(held.iter().map(|i| i.word()).collect::<Vec<_>>(), vec!["corner", "edge"]);
        held = picked_with(&held, e.clone(), false);
        assert_eq!(held, vec![e.clone()]);
        held = picked_with(&held, e, true);
        assert!(held.is_empty());
    }

    // ---- in inches (reCoreUnits.test.ts) ----

    #[test]
    fn readings_in_inches_millimetres_unchanged() {
        let inch = |items: &[PickedItem]| selection_info(items, DisplayUnits::In).expect("a reading");
        let one = inch(&[corner(v(25.4, 12.7, 0.0))]);
        assert_eq!(value(&one, "X").as_deref(), Some("1.0000 in"));
        assert_eq!(value(&one, "Y").as_deref(), Some("0.5000 in"));
        assert_eq!(value(&inch(&[corner(v(0.0, 0.0, 0.0)), corner(v(0.0, 0.0, 50.8))]), "Distance").as_deref(), Some("2.0000 in"));
        assert_eq!(value(&inch(&[plane(v(25.4, 0.0, 0.0), v(0.0, 0.0, 1.0), 645.16)]), "Area").as_deref(), Some("1.000 in²"));
        assert_eq!(value(&inch(&[plane(v(25.4, 0.0, 0.0), v(0.0, 0.0, 1.0), 645.16)]), "Centre").as_deref(), Some("1.0000, 0.0000, 0.0000"));
        assert_eq!(value(&info(&[plane(v(25.4, 0.0, 0.0), v(0.0, 0.0, 1.0), 645.16)]), "Area").as_deref(), Some("645.16 mm²"));
        assert_eq!(value(&info(&[corner(v(25.4, 12.7, 0.0))]), "X").as_deref(), Some("25.400 mm"));
    }
}
