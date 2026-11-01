// SPDX-License-Identifier: AGPL-3.0-only
//! The shapes a sketch handed in as data may carry — a rectangle (rounded, or a slot at half
//! its shorter side), a slot by its ends or by its centre, length and angle, a regular polygon,
//! a polyline open or closed — each drawn as the panel's tool draws it, into a sketch of its
//! own, and brought into the sketch under a run of ids. Ported from ScanRuler
//! (`plugins/reverse/app/reSketchInput.ts`: `drawShape`, `slotOfRectangle`, `onSheet`,
//! `polylineClosing` and the shapes part of `sketchOf`), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only.
//!
//! A shape's points are on the sketch's sheet, `[u, v]` millimetres — or, with `world: true`,
//! world points `[x, y, z]` taken square onto the plane. A shape may be turned (`angleDeg`),
//! placed fixed or as construction. An ellipse is drawn by ScanRuler as one closed spline: the
//! sketch here has no spline piece, and an ellipse is refused by name.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::f64::consts::PI;

use glam::{DVec2, DVec3};
use serde_json::Value;

use crate::fillet_corner::fillet_lines;
use crate::frame::SectionFrame;
use crate::sketch::{anchor_point, Anchor, AnchorAt, Constraint, Entity, EntityId, Rule, Shape, Sketch};
use crate::sketch_shapes::{add_polygon, add_rectangle_centre, infer_relations, renumbered, PolygonFit, POLYGON_MAX_SIDES};

/// A polyline's last corner within this of its first is that corner, mm.
pub const SNAP_MM: f64 = 0.05;
/// A round within this of half a rectangle's shorter side is that half: the rectangle is a
/// slot.
const ROUND_FITS_MM: f64 = 1e-6;

/// A shape's pieces by what they are: `sides`, `rounds`, `ends`, `center`, `circle`, `rim`,
/// `lines` — each one id or several.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Role {
    One(EntityId),
    Many(Vec<EntityId>),
}

pub type Roles = BTreeMap<String, Role>;

/// What a shape became: its index among the shapes, its kind, its pieces by role, and whether
/// a polyline closed on its first corner.
#[derive(Debug, Clone, PartialEq)]
pub struct ShapeMade {
    pub index: usize,
    pub kind: String,
    pub roles: Roles,
    pub closed_on: bool,
}

fn round4(v: f64) -> f64 {
    (v * 1e4).round() / 1e4
}

fn num(sh: &Value, k: &str) -> Option<f64> {
    sh.get(k).and_then(Value::as_f64)
}

fn point(v: &Value) -> Option<Vec<f64>> {
    v.as_array().map(|a| a.iter().filter_map(Value::as_f64).collect())
}

fn v2(sh: &Value, k: &str) -> Option<DVec2> {
    let p = point(sh.get(k)?)?;
    (p.len() >= 2).then(|| DVec2::new(p[0], p[1]))
}

/// A shape's points on the sketch's sheet (ScanRuler's `onSheet`): with `world: true`, world
/// `[x, y, z]` taken square onto the plane into its `[u, v]`; `[u, v]` as given otherwise.
pub fn on_sheet(sh: &Value, frame: &SectionFrame, place: &str) -> Result<Value, String> {
    let world = sh.get("world").and_then(Value::as_bool) == Some(true);
    let to_sheet = |p: &Value, key: &str| -> Result<Value, String> {
        let q = point(p).unwrap_or_default();
        if world {
            if q.len() != 3 {
                return Err(format!("{place}: {key} — with world: true a point is [x, y, z] in the world, mm."));
            }
            let s = frame.to_sheet(DVec3::new(q[0], q[1], q[2]));
            return Ok(serde_json::json!([s.x, s.y]));
        }
        if q.len() != 2 {
            return Err(format!("{place}: {key} is [u, v] on the sketch’s sheet — give world: true to hand it world points [x, y, z]."));
        }
        Ok(serde_json::json!([q[0], q[1]]))
    };
    let mut out = sh.clone();
    for k in ["center", "from", "to"] {
        if let Some(v) = sh.get(k).filter(|v| v.is_array()) {
            out[k] = to_sheet(v, k)?;
        }
    }
    if let Some(points) = sh.get("points").and_then(Value::as_array) {
        let mut pts = Vec::new();
        for (i, p) in points.iter().enumerate() {
            pts.push(to_sheet(p, &format!("points[{i}]"))?);
        }
        out["points"] = Value::Array(pts);
    }
    Ok(out)
}

/// A polyline's corners as drawn (ScanRuler's `polylineClosing`): a last corner within
/// `SNAP_MM` of the first is that corner, and the polyline closes on it. Whether it did.
pub fn polyline_closing(sh: Value) -> (Value, bool) {
    if sh.get("kind").and_then(Value::as_str) != Some("polyline") {
        return (sh, false);
    }
    let Some(pts) = sh.get("points").and_then(Value::as_array).cloned() else { return (sh, false) };
    if pts.len() < 4 {
        return (sh, false);
    }
    let (first, last) = (point(&pts[0]).unwrap_or_default(), point(&pts[pts.len() - 1]).unwrap_or_default());
    if first.len() < 2 || last.len() < 2 || (last[0] - first[0]).hypot(last[1] - first[1]) > SNAP_MM {
        return (sh, false);
    }
    let mut out = sh;
    out["points"] = Value::Array(pts[..pts.len() - 1].to_vec());
    out["closed"] = Value::Bool(true);
    (out, true)
}

/// A rectangle rounded at half its shorter side: a slot along the longer one — a circle when
/// the sides are equal — its round dimensioned.
fn slot_of_rectangle(scratch: &mut Sketch, lo: DVec2, hi: DVec2, turn: &dyn Fn(DVec2) -> DVec2, place: &str) -> Result<Roles, String> {
    let (w, h) = (hi.x - lo.x, hi.y - lo.y);
    let r = w.min(h) / 2.0;
    let m = (lo + hi) / 2.0;
    let mut roles = Roles::new();
    if (w - h).abs() <= 2.0 * ROUND_FITS_MM {
        let circle = scratch.add_shape(Shape::Circle { c: turn(m), r });
        scratch.constrain(Rule::Diameter { entity: circle, value: 2.0 * r });
        roles.insert("circle".into(), Role::One(circle));
        return Ok(roles);
    }
    let along = w > h;
    let from = if along { DVec2::new(lo.x + r, m.y) } else { DVec2::new(m.x, lo.y + r) };
    let to = if along { DVec2::new(hi.x - r, m.y) } else { DVec2::new(m.x, hi.y - r) };
    let (a, b) = (turn(from), turn(to));
    let slot = serde_json::json!({ "kind": "slot", "from": [a.x, a.y], "to": [b.x, b.y], "width": 2.0 * r });
    let roles = draw_shape(scratch, &slot, place)?;
    if let Some(Role::Many(ends)) = roles.get("ends") {
        scratch.constrain(Rule::Radius { entity: ends[0], value: r });
    }
    Ok(roles)
}

/// A shape drawn as the panel's tool draws it, into a sketch of its own — its pieces numbered
/// in the order they are made. Its pieces by what they are (ScanRuler's `drawShape`).
pub fn draw_shape(scratch: &mut Sketch, sh: &Value, place: &str) -> Result<Roles, String> {
    let turn = num(sh, "angleDeg").unwrap_or(0.0) * (PI / 180.0);
    let turned = |p: DVec2, about: DVec2| -> DVec2 {
        let d = p - about;
        about + DVec2::new(d.x * turn.cos() - d.y * turn.sin(), d.x * turn.sin() + d.y * turn.cos())
    };
    let mut roles = Roles::new();
    match sh.get("kind").and_then(Value::as_str).unwrap_or("") {
        "rectangle" => {
            let (center, size, from, to) = (v2(sh, "center"), v2(sh, "size"), v2(sh, "from"), v2(sh, "to"));
            let (lo, hi) = match (center, size, from, to) {
                (Some(c), Some(s), _, _) => {
                    if !(s.x.abs() > 0.0 && s.y.abs() > 0.0) {
                        return Err(format!("{place}: a rectangle needs a width and a height."));
                    }
                    (c - s.abs() / 2.0, c + s.abs() / 2.0)
                }
                (_, _, Some(f), Some(t)) => {
                    if (t.x - f.x).abs() < 1e-9 || (t.y - f.y).abs() < 1e-9 {
                        return Err(format!("{place}: from and to lie on a level or upright line — the rectangle has no area."));
                    }
                    (f.min(t), f.max(t))
                }
                _ => return Err(format!("{place}: a rectangle takes from and to, or center and size.")),
            };
            let about = center.or(from).unwrap_or(lo);
            let radius = num(sh, "radius");
            // The most a round takes is half the shorter side, measured before any corner is
            // cut back; at that the rectangle is a slot (a circle, square), and drawn as one.
            let half = (hi.x - lo.x).min(hi.y - lo.y) / 2.0;
            if let Some(r) = radius {
                if r > half + ROUND_FITS_MM {
                    return Err(format!("{place}: the corners do not round at {r} mm — half the shorter side, {} mm, is the most a round takes (a slot).", round4(half)));
                }
                if r > half - ROUND_FITS_MM {
                    return slot_of_rectangle(scratch, lo, hi, &|p| turned(p, about), place);
                }
            }
            let corners = [lo, DVec2::new(hi.x, lo.y), hi, DVec2::new(lo.x, hi.y)];
            let sides: Vec<EntityId> = if turn == 0.0 { scratch.add_rectangle(lo, hi).to_vec() } else { scratch.add_turned_rectangle(corners.map(|c| turned(c, about))).to_vec() };
            roles.insert("sides".into(), Role::Many(sides.clone()));
            if center.is_some() {
                roles.insert("center".into(), Role::One(add_rectangle_centre(scratch, &sides)));
            }
            if let Some(r) = radius {
                // Each corner rounded as the sketch's own Round does it: the sides cut back, an
                // arc tangent to both, its radius dimensioned.
                let mut rounds = Vec::new();
                for i in 0..4 {
                    match fillet_lines(scratch, sides[i], sides[(i + 1) % 4], Some(r), Some(half + ROUND_FITS_MM)) {
                        Ok(made) => rounds.push(made.arc),
                        Err(why) => return Err(format!("{place}: the corners do not round at {r} mm — {why}")),
                    }
                }
                roles.insert("rounds".into(), Role::Many(rounds));
                // The diagonals end where the sides now start — at the tangent points, which
                // lie about the same centre as the corners did.
                let moves: Vec<(EntityId, AnchorAt, DVec2)> = scratch
                    .constraints
                    .iter()
                    .filter_map(|c| match c.rule {
                        Rule::Coincident { p, q } => {
                            let diagonal = scratch.entity(p.entity)?;
                            let side = scratch.entity(q.entity)?;
                            (matches!(diagonal.shape, Shape::Line { .. }) && diagonal.construction).then(|| (p.entity, p.at, anchor_point(&side.shape, q.at)))
                        }
                        _ => None,
                    })
                    .collect();
                for (id, at, to) in moves {
                    if let Some(Entity { shape: Shape::Line { a, b }, .. }) = scratch.entity_mut(id) {
                        if at == AnchorAt::A {
                            *a = to;
                        } else {
                            *b = to;
                        }
                    }
                }
            }
            Ok(roles)
        }
        "slot" => {
            let r = num(sh, "width").unwrap_or(f64::NAN) / 2.0;
            let (mut from, mut to) = (v2(sh, "from"), v2(sh, "to"));
            if let (Some(c), Some(length)) = (v2(sh, "center"), num(sh, "length")) {
                let half = length / 2.0;
                from = Some(c - DVec2::new(turn.cos(), turn.sin()) * half);
                to = Some(c + DVec2::new(turn.cos(), turn.sin()) * half);
            }
            let (Some(from), Some(to)) = (from, to) else { return Err(format!("{place}: a slot takes from and to, or center and length.")) };
            let d = to - from;
            let len = d.x.hypot(d.y);
            if !(len > 1e-9) {
                return Err(format!("{place}: from and to are one point — a slot that short is a circle."));
            }
            if !(r > 0.0) {
                return Err(format!("{place}: a slot needs a width."));
            }
            let u = d / len;
            let n = DVec2::new(-u.y, u.x);
            let off = |q: DVec2, k: f64| q + n * r * k;
            // Round the loop: the left side out, the round at `to` (from its right-hand end
            // counter-clockwise through the far end), the right side back, the round at `from`.
            let left = scratch.add_shape(Shape::Line { a: off(from, 1.0), b: off(to, 1.0) });
            let end_to = scratch.add_shape(Shape::Arc { c: to, r, start: (-n.y).atan2(-n.x), sweep: PI });
            let right = scratch.add_shape(Shape::Line { a: off(to, -1.0), b: off(from, -1.0) });
            let end_from = scratch.add_shape(Shape::Arc { c: from, r, start: n.y.atan2(n.x), sweep: PI });
            scratch.constrain(Rule::Coincident { p: Anchor::new(left, AnchorAt::B), q: Anchor::new(end_to, AnchorAt::B) });
            scratch.constrain(Rule::Coincident { p: Anchor::new(end_to, AnchorAt::A), q: Anchor::new(right, AnchorAt::A) });
            scratch.constrain(Rule::Coincident { p: Anchor::new(right, AnchorAt::B), q: Anchor::new(end_from, AnchorAt::B) });
            scratch.constrain(Rule::Coincident { p: Anchor::new(end_from, AnchorAt::A), q: Anchor::new(left, AnchorAt::A) });
            for (a, b) in [(left, end_to), (right, end_to), (right, end_from), (left, end_from)] {
                scratch.constrain(Rule::Tangent { a, b });
            }
            scratch.constrain(Rule::Equal { a: end_to, b: end_from });
            roles.insert("sides".into(), Role::Many(vec![left, right]));
            roles.insert("ends".into(), Role::Many(vec![end_to, end_from]));
            Ok(roles)
        }
        "polygon" => {
            let center = v2(sh, "center").unwrap_or(DVec2::ZERO);
            let r = num(sh, "radius").unwrap_or(f64::NAN);
            let sides = num(sh, "sides").unwrap_or(0.0);
            let fit = if sh.get("fit").and_then(Value::as_str) == Some("circumscribed") { PolygonFit::Circumscribed } else { PolygonFit::Inscribed };
            let at = center + DVec2::new(turn.cos(), turn.sin()) * r;
            let made = (sides.fract() == 0.0 && sides >= 0.0).then(|| add_polygon(scratch, center, at, sides as usize, fit)).flatten();
            let Some(made) = made else { return Err(format!("{place}: a polygon takes 3 to {POLYGON_MAX_SIDES} sides and a radius.")) };
            roles.insert("sides".into(), Role::Many(made.lines));
            roles.insert("circle".into(), Role::One(made.circle));
            if made.circle != made.rim {
                roles.insert("rim".into(), Role::One(made.rim));
            }
            Ok(roles)
        }
        "polyline" => {
            let pts: Vec<DVec2> = sh.get("points").and_then(Value::as_array).map(|a| a.iter().filter_map(point).filter(|p| p.len() >= 2).map(|p| DVec2::new(p[0], p[1])).collect()).unwrap_or_default();
            let closed = sh.get("closed").and_then(Value::as_bool) == Some(true);
            if closed && pts.len() < 3 {
                return Err(format!("{place}: a closed polyline needs three corners or more."));
            }
            let n = if closed { pts.len() } else { pts.len().saturating_sub(1) };
            let mut lines = Vec::new();
            for i in 0..n {
                let (a, b) = (pts[i], pts[(i + 1) % pts.len()]);
                if (b - a).x.hypot((b - a).y) < 1e-9 {
                    return Err(format!("{place}: corners {i} and {} are one point.", (i + 1) % pts.len()));
                }
                lines.push(scratch.add_shape(Shape::Line { a, b }));
            }
            for w in lines.windows(2) {
                scratch.constrain(Rule::Coincident { p: Anchor::new(w[0], AnchorAt::B), q: Anchor::new(w[1], AnchorAt::A) });
            }
            if closed && lines.len() > 1 {
                scratch.constrain(Rule::Coincident { p: Anchor::new(lines[lines.len() - 1], AnchorAt::B), q: Anchor::new(lines[0], AnchorAt::A) });
            }
            roles.insert("lines".into(), Role::Many(lines));
            Ok(roles)
        }
        "ellipse" => Err(format!("{place}: an ellipse is drawn as one closed spline, and the sketch has no spline piece yet — draw it with arcs, or as a polyline.")),
        other => Err(format!("{place}: there is no shape “{other}”.")),
    }
}

/// The shapes handed in drawn into a sketch (the shapes part of ScanRuler's `sketchOf`): each
/// on a sheet of its own, its pieces brought over under the run of ids it asked for (`id`, or
/// the next free), fixed or construction as asked, its relations with them; the relations the
/// drawing implies inferred on what is new — a rectangle's or a polygon's sides, which their
/// own relations hold square, left out of the levels.
pub fn add_shapes(sketch: &mut Sketch, frame: &SectionFrame, shapes: &[Value]) -> Result<Vec<ShapeMade>, String> {
    let mut made = Vec::new();
    let mut fresh: HashSet<EntityId> = HashSet::new();
    let mut shaped: HashSet<EntityId> = HashSet::new();
    for (i, sh) in shapes.iter().enumerate() {
        let kind = sh.get("kind").and_then(Value::as_str).unwrap_or("").to_string();
        let place = format!("shapes[{i}] ({kind})");
        let (drawn, closed_on) = polyline_closing(on_sheet(sh, frame, &place)?);
        let mut scratch = Sketch::default();
        let roles = draw_shape(&mut scratch, &drawn, &place)?;
        let start = sh.get("id").and_then(Value::as_u64).map_or(sketch.next_id, |v| v as u32);
        let map: HashMap<EntityId, EntityId> = scratch.entities.iter().enumerate().map(|(k, e)| (e.id, start + k as u32)).collect();
        let id = |old: EntityId| map.get(&old).copied().unwrap_or(old);
        for e in &scratch.entities {
            let at = id(e.id);
            if sketch.entity(at).is_some() || sketch.constraint(at).is_some() {
                return Err(format!("{place}: its pieces take the ids {start} to {}, and the id {at} is taken.", start + scratch.entities.len() as u32 - 1));
            }
        }
        for e in &scratch.entities {
            let mut copy = e.clone();
            copy.id = id(e.id);
            if sh.get("fixed").and_then(Value::as_bool) == Some(true) {
                copy.fixed = true;
            }
            if sh.get("construction").and_then(Value::as_bool) == Some(true) {
                copy.construction = true;
            }
            fresh.insert(copy.id);
            if kind != "polyline" {
                shaped.insert(copy.id);
            }
            sketch.next_id = sketch.next_id.max(copy.id + 1);
            sketch.entities.push(copy);
        }
        for c in &scratch.constraints {
            let cid = sketch.next_id;
            sketch.next_id += 1;
            sketch.constraints.push(Constraint { id: cid, rule: renumbered(&c.rule, id), inferred: c.inferred, expr: None, place: None });
        }
        let roles = roles
            .into_iter()
            .map(|(k, v)| {
                let v = match v {
                    Role::One(x) => Role::One(id(x)),
                    Role::Many(xs) => Role::Many(xs.into_iter().map(id).collect()),
                };
                (k, v)
            })
            .collect();
        made.push(ShapeMade { index: i, kind, roles, closed_on });
    }
    infer_relations(sketch, Some(&fresh), Some(&shaped));
    Ok(made)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regions::find_regions;
    use crate::solver::{solve_sketch, SolveOptions};
    use serde_json::json;

    fn xy() -> SectionFrame {
        SectionFrame { origin: DVec3::ZERO, normal: DVec3::Z, u: DVec3::X, v: DVec3::Y }
    }

    /// A sketch of shapes, solved, and its regions' areas.
    fn sketch_of(shapes: Value) -> Result<(Sketch, Vec<ShapeMade>, bool, Vec<f64>), String> {
        let mut s = Sketch::default();
        let made = add_shapes(&mut s, &xy(), shapes.as_array().expect("shapes"))?;
        let report = solve_sketch(&mut s, SolveOptions::default());
        let areas = find_regions(&s).iter().map(|r| r.area).collect();
        Ok((s, made, report.ok, areas))
    }

    fn close(a: f64, b: f64, digits: i32) -> bool {
        (a - b).abs() < 0.5 * 10f64.powi(-digits)
    }

    // Ported from ScanRuler's `reCommandsScan.test.ts` ("sketch shapes and several features at
    // once"), the sketch-level assertions: the command layer around them is not here.

    #[test]
    fn draws_a_slot_and_a_rounded_rectangle_closed_and_tangent() {
        let (_, made, ok, areas) = sketch_of(json!([{ "kind": "slot", "id": 1, "from": [-10, 0], "to": [10, 0], "width": 6, "fixed": true }])).expect("a slot");
        assert!(ok);
        assert_eq!(made[0].roles.get("sides"), Some(&Role::Many(vec![1, 3])));
        assert_eq!(made[0].roles.get("ends"), Some(&Role::Many(vec![2, 4])));
        assert_eq!(areas.len(), 1);
        assert!(close(areas[0], 20.0 * 6.0 + PI * 9.0, 1), "{}", areas[0]);
        let (_, _, _, areas) = sketch_of(json!([{ "kind": "slot", "id": 1, "center": [0, 0], "length": 20, "width": 6, "angleDeg": 90, "fixed": true }])).expect("a turned slot");
        assert!(close(areas[0], 20.0 * 6.0 + PI * 9.0, 1));
        let (_, made, ok, areas) = sketch_of(json!([{ "kind": "rectangle", "id": 1, "from": [0, 0], "to": [20, 10], "radius": 2, "fixed": true }])).expect("rounded");
        assert!(ok);
        assert_eq!(made[0].roles.get("sides"), Some(&Role::Many(vec![1, 2, 3, 4])));
        assert!(matches!(made[0].roles.get("rounds"), Some(Role::Many(r)) if r.len() == 4));
        assert!(close(areas[0], 200.0 - (4.0 - PI) * 4.0, 1), "{}", areas[0]);
        let e = sketch_of(json!([{ "kind": "rectangle", "from": [0, 0], "to": [20, 10], "radius": 8 }])).expect_err("refused");
        assert!(e.contains("do not round at 8 mm"), "{e}");
        assert!(sketch_of(json!([{ "kind": "slot", "width": 4 }])).is_err());
    }

    #[test]
    fn rounds_a_rectangle_up_to_half_its_shorter_side_and_at_half_draws_a_slot() {
        let (_, made, ok, areas) = sketch_of(json!([{ "kind": "rectangle", "id": 1, "center": [0, 0], "size": [5.9, 2.55], "radius": 1.25, "fixed": true }])).expect("a key");
        assert!(ok);
        assert!(matches!(made[0].roles.get("rounds"), Some(Role::Many(r)) if r.len() == 4));
        assert!(close(areas[0], 5.9 * 2.55 - (4.0 - PI) * 1.25f64.powi(2), 1));
        let (s, made, ok, areas) = sketch_of(json!([{ "kind": "rectangle", "id": 1, "center": [0, 0], "size": [5.9, 2.55], "radius": 1.275, "angleDeg": 90 }])).expect("slotted");
        assert!(ok);
        assert_eq!(made[0].roles.get("sides"), Some(&Role::Many(vec![1, 3])));
        assert_eq!(made[0].roles.get("ends"), Some(&Role::Many(vec![2, 4])));
        assert!(close(areas[0], (5.9 - 2.55) * 2.55 + PI * 1.275f64.powi(2), 1));
        assert!(s.constraints.iter().any(|c| matches!(c.rule, Rule::Radius { value, .. } if value == 1.275)));
        let (_, _, _, areas) = sketch_of(json!([{ "kind": "rectangle", "from": [0, 0], "to": [4, 4], "radius": 2 }])).expect("round");
        assert!(close(areas[0], PI * 4.0, 1));
        let e = sketch_of(json!([{ "kind": "rectangle", "center": [0, 0], "size": [5.9, 2.55], "radius": 1.3 }])).expect_err("refused");
        assert!(e.contains("1.275 mm, is the most"), "{e}");
    }

    #[test]
    fn takes_a_shapes_points_in_the_world_and_refuses_a_mix() {
        // A plane across X at x = 3, as the sketch rule lays it.
        let frame = SectionFrame { origin: DVec3::new(3.0, 0.0, 0.0), normal: DVec3::X, u: DVec3::Y, v: DVec3::Z };
        let mut s = Sketch::default();
        add_shapes(&mut s, &frame, &[json!({ "kind": "rectangle", "center": [5, 4, 7], "size": [2, 2], "world": true, "fixed": true })]).expect("drawn");
        let r = &find_regions(&s)[0];
        assert!(close(r.centroid.x, 4.0, 6) && close(r.centroid.y, 7.0, 6));
        let e = add_shapes(&mut Sketch::default(), &frame, &[json!({ "kind": "polyline", "points": [[0, 0], [1, 0]], "world": true })]).expect_err("refused");
        assert!(e.contains("[x, y, z]"), "{e}");
        let e = add_shapes(&mut Sketch::default(), &frame, &[json!({ "kind": "polyline", "points": [[0, 0, 0], [1, 0, 0]] })]).expect_err("refused");
        assert!(e.contains("world: true"), "{e}");
    }

    /// Not a ScanRuler test: a polyline whose last corner falls on its first closes on it, and
    /// the shape ids run on from the one asked for.
    #[test]
    fn a_polyline_closes_on_its_first_corner() {
        let (s, made, ok, areas) = sketch_of(json!([{ "kind": "polyline", "id": 60, "points": [[10, 5], [20, 5], [20, 15], [10.01, 5.01]] }])).expect("drawn");
        assert!(ok && made[0].closed_on);
        assert_eq!(made[0].roles.get("lines"), Some(&Role::Many(vec![60, 61, 62])));
        assert_eq!(s.entities.len(), 3);
        assert!(close(areas[0], 50.0, 3));
        let e = sketch_of(json!([{ "kind": "ellipse", "center": [0, 0], "a": 12, "b": 4 }])).expect_err("refused by name");
        assert!(e.contains("ellipse"), "{e}");
    }
}
