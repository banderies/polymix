// SPDX-License-Identifier: AGPL-3.0-only
//! Fitted lines laid along the sheet's axes. Ported from ScanRuler
//! (`plugins/reverse/core/axisAlign.ts`), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! A scan's side is never level: the part lay a fraction of a degree off on the table, the
//! wall bows, the fit follows both — and a profile whose sides read 0.2° and 89.6° extrudes
//! into a body no drawing would dimension. Where a fitted line runs within a few degrees of
//! the sheet's X or Y it is meant to run along it, the same reading the drawing tools make of
//! a line drawn by hand (`axis_relation_of`), and it is given the same inferred relation, a
//! chip the operator can take off. Lines near 45° are held at it against one of the lines made
//! level or upright, as an angle, since the sketch has no relation for a diagonal.
//!
//! This reads the sheet's axes, so it is only as good as the sheet's frame: on a part not yet
//! aligned the Align command lays the sheet along a side first.

use std::collections::HashSet;

use glam::DVec2;
use serde::{Deserialize, Serialize};

use crate::corners::PlacedSegment;
use crate::sketch::{axis_relation_of, measure_rule, AxisRelation, Constraint, Entity, EntityId, Rule, Shape, Sketch, INFER_AXIS_DEG};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AxisAlignOptions {
    /// How far off an axis a line may run and still be laid along it, degrees.
    pub degrees: f64,
    /// Whether lines near 45° are held there too.
    pub diagonals: bool,
}

impl Default for AxisAlignOptions {
    fn default() -> Self {
        Self { degrees: INFER_AXIS_DEG, diagonals: false }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct AxisAlignReport {
    pub horizontal: usize,
    pub vertical: usize,
    pub diagonal: usize,
}

/// Turn a line about its middle onto a direction, its length kept — so the solver starts
/// from the relation already met and only the corners move. The slice points behind it turn
/// with it: they say where the side lies and how far it runs, which the turn keeps, and at
/// what angle, which the relation now overrules — left as they were, five hundred points a
/// degree off would pull against the relation at every solve, the corners would be held a few
/// microns open by the fight, and the sketch would read as not solved. The slice itself is
/// drawn from the chain and does not move.
fn turn_to(line: &mut Entity, dx: f64, dy: f64) {
    let Shape::Line { a, b } = line.shape else { return };
    let mx = (a.x + b.x) / 2.0;
    let my = (a.y + b.y) / 2.0;
    let ux = b.x - a.x;
    let uy = b.y - a.y;
    let length = ux.hypot(uy);
    let sign = if ux * dx + uy * dy < 0.0 { -1.0 } else { 1.0 };
    let tx = dx * sign;
    let ty = dy * sign;
    // The rotation that takes the line's direction to (tx, ty).
    let cos = (ux * tx + uy * ty) / length;
    let sin = (ux * ty - uy * tx) / length;
    line.shape = Shape::Line { a: DVec2::new(mx - (tx * length) / 2.0, my - (ty * length) / 2.0), b: DVec2::new(mx + (tx * length) / 2.0, my + (ty * length) / 2.0) };
    for p in &mut line.points {
        let px = p.x - mx;
        let py = p.y - my;
        *p = DVec2::new(mx + px * cos - py * sin, my + px * sin + py * cos);
    }
}

fn push_inferred(sketch: &mut Sketch, rule: Rule) {
    let id = sketch.next_id;
    sketch.next_id += 1;
    sketch.constraints.push(Constraint { id, rule, inferred: true, expr: None, place: None });
}

/// Give every listed line that runs within `degrees` of a sheet axis the horizontal or
/// vertical relation, inferred; with `diagonals`, hold those within `degrees` of 45° at it
/// against the longest aligned line. Lines that already carry such a relation are left alone.
pub fn align_to_axes(sketch: &mut Sketch, ids: &[EntityId], opts: AxisAlignOptions) -> AxisAlignReport {
    let degrees = opts.degrees;
    let mut report = AxisAlignReport::default();
    // The held lines in the order they were found: the reference is the first of the longest.
    let mut held_order: Vec<EntityId> = Vec::new();
    let mut held: HashSet<EntityId> = HashSet::new();
    for c in &sketch.constraints {
        if let Rule::Horizontal { entity } | Rule::Vertical { entity } = c.rule {
            if held.insert(entity) {
                held_order.push(entity);
            }
        }
    }
    let lines: Vec<EntityId> = ids.iter().copied().filter(|id| matches!(sketch.entity(*id).map(|e| e.shape.clone()), Some(Shape::Line { .. }))).collect();
    for &id in &lines {
        let Some(line) = sketch.entity(id) else { continue };
        if held.contains(&id) || line.fixed {
            continue;
        }
        let Shape::Line { a, b } = line.shape else { continue };
        let Some(relation) = axis_relation_of(a, b, degrees) else { continue };
        if let Some(line) = sketch.entity_mut(id) {
            match relation {
                AxisRelation::Horizontal => turn_to(line, 1.0, 0.0),
                AxisRelation::Vertical => turn_to(line, 0.0, 1.0),
            }
        }
        push_inferred(sketch, relation.rule(id));
        if held.insert(id) {
            held_order.push(id);
        }
        match relation {
            AxisRelation::Horizontal => report.horizontal += 1,
            AxisRelation::Vertical => report.vertical += 1,
        }
    }
    if !opts.diagonals {
        return report;
    }
    // The reference: the longest line lying along an axis.
    let mut reference: Option<EntityId> = None;
    let mut longest = 0.0;
    for &id in &held_order {
        let Some(Shape::Line { a, b }) = sketch.entity(id).map(|e| e.shape.clone()) else { continue };
        let l = (b.x - a.x).hypot(b.y - a.y);
        if l > longest {
            longest = l;
            reference = Some(id);
        }
    }
    let Some(reference) = reference else { return report };
    let mut angled: HashSet<EntityId> = HashSet::new();
    for c in &sketch.constraints {
        if let Rule::Angle { a, b, .. } | Rule::Parallel { a, b } | Rule::Perpendicular { a, b } = c.rule {
            angled.insert(a);
            angled.insert(b);
        }
    }
    for &id in &lines {
        let Some(line) = sketch.entity(id) else { continue };
        if held.contains(&id) || angled.contains(&id) || line.fixed {
            continue;
        }
        let Some(measured) = measure_rule(sketch, &Rule::Angle { a: id, b: reference, value: 0.0 }, &[]) else { continue };
        let Some(value) = [45.0, 135.0].into_iter().find(|v: &f64| (measured - v).abs() <= degrees) else { continue };
        push_inferred(sketch, Rule::Angle { a: id, b: reference, value });
        report.diagonal += 1;
    }
    report
}

/// The axis relation a line carries, if any.
fn axis_of(sketch: &Sketch, id: EntityId) -> Option<AxisRelation> {
    sketch.constraints.iter().find_map(|c| match c.rule {
        Rule::Horizontal { entity } if entity == id => Some(AxisRelation::Horizontal),
        Rule::Vertical { entity } if entity == id => Some(AxisRelation::Vertical),
        _ => None,
    })
}

/// Merge `out[j]` into `out[i]`, which it follows; false when they are not two pieces of one
/// aligned wall.
fn merge(sketch: &mut Sketch, out: &mut Vec<PlacedSegment>, i: usize, j: usize, within: f64) -> bool {
    let (ida, idb) = (out[i].id, out[j].id);
    let (Some(Shape::Line { a: aa, b: ab }), Some(Shape::Line { a: ba, b: bb })) = (sketch.entity(ida).map(|e| e.shape.clone()), sketch.entity(idb).map(|e| e.shape.clone())) else { return false };
    if ida == idb {
        return false;
    }
    let Some(axis) = axis_of(sketch, ida) else { return false };
    if axis_of(sketch, idb) != Some(axis) {
        return false;
    }
    // Across the axis: how far apart the two lie.
    let k = if axis == AxisRelation::Horizontal { 1 } else { 0 };
    let off_a = (aa[k] + ab[k]) / 2.0;
    let off_b = (ba[k] + bb[k]) / 2.0;
    if (off_a - off_b).abs() > within {
        return false;
    }
    // They have to run the same way, one after the other.
    let along = 1 - k;
    if (ab[along] - aa[along]) * (bb[along] - ba[along]) <= 0.0 {
        return false;
    }
    let la = (ab[along] - aa[along]).abs();
    let lb = (bb[along] - ba[along]).abs();
    let off = (off_a * la + off_b * lb) / (la + lb);
    let mut start = aa;
    let mut end = bb;
    start[k] = off;
    end[k] = off;
    let b_points = sketch.entity(idb).map(|e| e.points.clone()).unwrap_or_default();
    if let Some(a) = sketch.entity_mut(ida) {
        a.shape = Shape::Line { a: start, b: end };
        a.points.extend(b_points);
    }
    sketch.remove_entities(&[idb]);
    out[i].join_next = out[j].join_next;
    out.remove(j);
    true
}

/// Make one line of every run of neighbouring fitted lines that were laid along the same axis
/// and lie within `within` of one line. A wall that wanders comes out of a fine fit as a row of
/// pieces, each level once aligned, each at its own height by a tenth of a millimetre — and
/// joined end to end they cannot all be level and meet, so the sketch would not solve. Level
/// and in line, they are the one wall. The merged line keeps the first piece's id and every
/// piece's slice points; the pieces merged away go, with what was on them. Returns the order
/// as it stands after.
pub fn merge_aligned_lines(sketch: &mut Sketch, order: &[PlacedSegment], closed: bool, within: f64) -> Vec<PlacedSegment> {
    let mut out = order.to_vec();
    let mut i = 0;
    while i + 1 < out.len() {
        if !merge(sketch, &mut out, i, i + 1, within) {
            i += 1;
        }
    }
    if closed && out.len() > 2 {
        let last = out.len() - 1;
        merge(sketch, &mut out, last, 0, within);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corners::{infer_corners, place_segments, CORNER_GAP_MM};
    use crate::fit2d::Mulberry32;
    use crate::segment::{segment_chain, SegmentOptions};
    use crate::sketch::AnchorAt;
    use crate::solver::{solve_sketch, SolveOptions};
    use std::f64::consts::PI;

    enum Piece {
        Line(DVec2),
        Arc(DVec2, DVec2),
    }

    /// A closed outline walked from `start` through lines and quarter-turn arcs, a point every
    /// `step` mm, with Gaussian noise across it and, on top, a `wander` of the given amplitude
    /// and wavelength along every line — what a bowed sheet-metal wall does. Closed with its
    /// first point repeated.
    fn outline(start: DVec2, pieces: &[Piece], step: f64, noise: f64, wander: f64, wavelength: f64, seed: u32) -> Vec<DVec2> {
        let mut rand = Mulberry32::new(seed);
        let mut pts = Vec::new();
        let mut at = start;
        for p in pieces {
            match *p {
                Piece::Line(to) => {
                    let (dx, dy) = (to.x - at.x, to.y - at.y);
                    let len = dx.hypot(dy);
                    let n = ((len / step).round() as usize).max(1);
                    for i in 0..n {
                        let s = (i as f64 / n as f64) * len;
                        // The wander fades to nothing at both ends, so corners stay put.
                        let e = rand.gaussian() * noise + wander * ((2.0 * PI * s) / wavelength).sin() * ((PI * s) / len).sin();
                        pts.push(DVec2::new(at.x + (dx * s) / len - (dy / len) * e, at.y + (dy * s) / len + (dx / len) * e));
                    }
                    at = to;
                }
                Piece::Arc(to, center) => {
                    let r = (at.x - center.x).hypot(at.y - center.y);
                    let a0 = (at.y - center.y).atan2(at.x - center.x);
                    let mut a1 = (to.y - center.y).atan2(to.x - center.x);
                    // The short way round.
                    while a1 - a0 > PI {
                        a1 -= 2.0 * PI;
                    }
                    while a1 - a0 < -PI {
                        a1 += 2.0 * PI;
                    }
                    let n = (((a1 - a0).abs() * r / step).round() as usize).max(2);
                    for i in 0..n {
                        let a = a0 + ((a1 - a0) * i as f64) / n as f64;
                        let rr = r + rand.gaussian() * noise;
                        pts.push(DVec2::new(center.x + rr * a.cos(), center.y + rr * a.sin()));
                    }
                    at = to;
                }
            }
        }
        pts.push(pts[0]);
        pts
    }

    /// A long thin rounded rectangle, `w` by `h`, corner radius `r`.
    fn rounded_bar(w: f64, h: f64, r: f64, wander: f64, wavelength: f64) -> Vec<DVec2> {
        let v = DVec2::new;
        outline(
            v(r, 0.0),
            &[
                Piece::Line(v(w - r, 0.0)),
                Piece::Arc(v(w, r), v(w - r, r)),
                Piece::Line(v(w, h - r)),
                Piece::Arc(v(w - r, h), v(w - r, h - r)),
                Piece::Line(v(r, h)),
                Piece::Arc(v(0.0, h - r), v(r, h - r)),
                Piece::Line(v(0.0, r)),
                Piece::Arc(v(r, 0.0), v(r, r)),
            ],
            0.15,
            0.008,
            wander,
            wavelength,
            5,
        )
    }

    fn line_of(s: &Sketch, id: EntityId) -> (DVec2, DVec2) {
        match s.entity(id).map(|e| e.shape.clone()) {
            Some(Shape::Line { a, b }) => (a, b),
            other => panic!("not a line: {other:?}"),
        }
    }

    #[test]
    fn makes_a_line_within_the_angle_level_or_upright_and_leaves_the_rest() {
        let mut s = Sketch::default();
        let level = s.add_shape(Shape::Line { a: DVec2::new(0.0, 0.0), b: DVec2::new(50.0, 0.5) });
        let upright = s.add_shape(Shape::Line { a: DVec2::new(50.0, 0.5), b: DVec2::new(49.4, 30.0) });
        let slanted = s.add_shape(Shape::Line { a: DVec2::new(0.0, 0.0), b: DVec2::new(20.0, 3.0) });
        let opts = AxisAlignOptions { degrees: 3.0, diagonals: false };
        let report = align_to_axes(&mut s, &[level, upright, slanted], opts);
        assert_eq!(report, AxisAlignReport { horizontal: 1, vertical: 1, diagonal: 0 });
        let mut kinds: Vec<&str> = s.constraints.iter().map(|c| c.rule.name()).collect();
        kinds.sort();
        assert_eq!(kinds, vec!["Horizontal", "Vertical"]);
        assert!(s.constraints.iter().all(|c| c.inferred));
        let (a, b) = line_of(&s, level);
        assert!((a.y - b.y).abs() < 1e-9);
        // The direction it was drawn in is kept.
        assert!(b.x > a.x);
        // A second pass adds nothing.
        assert_eq!(align_to_axes(&mut s, &[level, upright, slanted], opts), AxisAlignReport::default());
    }

    #[test]
    fn holds_a_line_near_45_degrees_at_it_against_an_aligned_line_when_asked() {
        let mut s = Sketch::default();
        let level = s.add_shape(Shape::Line { a: DVec2::new(0.0, 0.0), b: DVec2::new(50.0, 0.2) });
        let diagonal = s.add_shape(Shape::Line { a: DVec2::new(50.0, 0.2), b: DVec2::new(60.0, 10.6) });
        assert_eq!(align_to_axes(&mut s, &[level, diagonal], AxisAlignOptions { degrees: 3.0, diagonals: false }).diagonal, 0);
        assert_eq!(align_to_axes(&mut s, &[level, diagonal], AxisAlignOptions { degrees: 3.0, diagonals: true }).diagonal, 1);
        let angle = s.constraints.iter().find_map(|c| if let Rule::Angle { value, .. } = c.rule { Some(value) } else { None }).unwrap();
        assert!([45.0, 135.0].contains(&angle));
        assert!(solve_sketch(&mut s, SolveOptions::default()).ok);
        let (a, b) = line_of(&s, diagonal);
        assert!(((b.y - a.y).abs() - (b.x - a.x).abs()).abs() < 5e-4);
    }

    #[test]
    fn the_pieces_of_one_aligned_wall_become_one_line_and_a_wandering_bar_still_solves() {
        let chain = rounded_bar(90.0, 18.0, 2.5, 0.12, 30.0);
        let cut = segment_chain(&chain, SegmentOptions { detail: Some(1.0), ..Default::default() }).unwrap();
        let mut s = Sketch::default();
        let mut order = place_segments(&mut s, &cut);
        assert!(order.len() > 8, "{}", order.len());
        let ids: Vec<EntityId> = order.iter().map(|p| p.id).collect();
        align_to_axes(&mut s, &ids, AxisAlignOptions { degrees: 3.0, diagonals: false });
        order = merge_aligned_lines(&mut s, &order, cut.closed, 0.6);
        assert_eq!(order.len(), 8);
        assert_eq!(s.entities.len(), 8);
        // The merged side runs the whole length and carries every piece's points.
        let sides: Vec<&Entity> = s.entities.iter().filter(|e| matches!(e.shape, Shape::Line { a, b } if (b - a).length() > 80.0)).collect();
        assert_eq!(sides.len(), 2);
        for side in sides {
            assert!(side.points.len() > 500, "{}", side.points.len());
        }
        let report = infer_corners(&mut s, &order, cut.closed, CORNER_GAP_MM);
        assert_eq!(report.open, 0);
        assert!(solve_sketch(&mut s, SolveOptions::default()).ok);
    }

    #[test]
    fn leaves_two_level_lines_a_step_apart_as_two() {
        let mut s = Sketch::default();
        let a = s.add_shape(Shape::Line { a: DVec2::new(0.0, 0.0), b: DVec2::new(10.0, 0.0) });
        let b = s.add_shape(Shape::Line { a: DVec2::new(10.0, 1.2), b: DVec2::new(20.0, 1.2) });
        let order = [PlacedSegment { id: a, from: AnchorAt::A, to: AnchorAt::B, join_next: false }, PlacedSegment { id: b, from: AnchorAt::A, to: AnchorAt::B, join_next: false }];
        align_to_axes(&mut s, &[a, b], AxisAlignOptions::default());
        assert_eq!(merge_aligned_lines(&mut s, &order, false, 0.6).len(), 2);
    }
}
