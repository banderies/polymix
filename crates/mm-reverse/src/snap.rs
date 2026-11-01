// SPDX-License-Identifier: AGPL-3.0-only
//! Where a click or a dragged point on the sketch sheet lands. Ported from ScanRuler
//! (`plugins/reverse/core/snap.ts`), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! On the end or the centre of a piece already there (and is then held to it by a
//! coincidence), on the sheet's zero (held to the axes that cross there), on the middle of a
//! line (held there the same way), on one of the part's coordinate planes, on the body of a
//! drawn or a projected piece (held on it by a point-on relation), on a point of the slice
//! the scan cut, on the body of a piece fitted to the slice, or on the grid — in that order,
//! each within the hand's reach — or exactly where the cursor is when nothing is near, or when
//! snapping is switched off with Ctrl held, as Fusion has it. Everything of the sketch's own
//! holds what lands on it, as every CAD sketcher infers a coincidence from a snap; the slice
//! and the grid are places only. The grid is offered only when the operator switched grid
//! snapping on; the grid is otherwise there to read against.

use std::collections::HashSet;

use glam::DVec2;
use serde::{Deserialize, Serialize};

use crate::datums::{trace_near, DatumPlane, DatumTrace};
use crate::sketch::{anchor_point, anchors_of, nearest_on, Anchor, AnchorAt, Entity, EntityId, Shape, Sketch};

/// What there is to snap to besides the sketch itself.
#[derive(Debug, Clone, Copy, Default)]
pub struct SnapContext<'a> {
    /// The slice chains under the sketch, sheet millimetres.
    pub chains: Option<&'a [Vec<DVec2>]>,
    /// The grid's spacing at the current zoom, or none.
    pub grid: Option<f64>,
    /// The part's coordinate planes as the lines they cut the sheet in — the axes a profile is
    /// drawn on, as every CAD sketcher snaps to.
    pub datums: Option<&'a [DatumTrace]>,
    /// Snapping switched off for this pick — Ctrl held.
    pub off: bool,
}

/// The share of the hand's reach within which the middle of a line is taken.
const MIDPOINT_REACH: f64 = 0.5;
/// The share of the hand's reach within which a coordinate plane's line is taken: less than an
/// end's, so an end drawn near the axis is still the end that was meant.
const DATUM_REACH: f64 = 0.75;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SnapKind {
    Anchor,
    Origin,
    Midpoint,
    Datum,
    Edge,
    Entity,
    Grid,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SnapResult {
    pub point: DVec2,
    /// The anchor the point landed on, when it landed on one that can hold a coincidence — a
    /// line's or an arc's end, an arc's or a circle's centre.
    pub anchor: Option<Anchor>,
    /// The coordinate plane whose line the point landed on, when it did — held there by a
    /// point-on relation, as an end is held to an anchor.
    pub datum: Option<DatumPlane>,
    /// The piece whose body the point landed on, when it did — held on it by a point-on
    /// relation.
    pub piece: Option<EntityId>,
    /// The coordinate planes whose lines cross at the sheet's zero, when the point landed
    /// there — held at the crossing by a point-on relation to each. Empty when no plane's line
    /// runs through the zero.
    pub origin: Option<Vec<DatumPlane>>,
    pub kind: Option<SnapKind>,
}

impl SnapResult {
    fn at(point: DVec2, kind: Option<SnapKind>) -> Self {
        Self { point, anchor: None, datum: None, piece: None, origin: None, kind }
    }
}

/// What a drawn or dragged end is held to after a snap: the anchor it landed on, the
/// coordinate plane's line it landed on, the piece whose body it landed on, the sheet's zero
/// by the planes crossing there — or nothing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SnapHold {
    Anchor(Anchor),
    Datum(DatumPlane),
    Piece(EntityId),
    Origin(Vec<DatumPlane>),
}

pub fn snap_hold(r: &SnapResult) -> Option<SnapHold> {
    if let Some(anchor) = r.anchor {
        return Some(SnapHold::Anchor(anchor));
    }
    if let Some(datum) = r.datum {
        return Some(SnapHold::Datum(datum));
    }
    if let Some(piece) = r.piece {
        return Some(SnapHold::Piece(piece));
    }
    match &r.origin {
        Some(origin) if !origin.is_empty() => Some(SnapHold::Origin(origin.clone())),
        _ => None,
    }
}

/// `Math.round` as JavaScript has it: halves round up, towards +∞.
fn js_round(x: f64) -> f64 {
    let f = x.floor();
    if x - f >= 0.5 {
        f + 1.0
    } else {
        f
    }
}

fn len2(a: DVec2) -> f64 {
    a.x.hypot(a.y)
}

/// Snap a sheet point. `exclude` names pieces that are not snapped to — the one being
/// dragged, which would otherwise snap to itself.
pub fn snap_sketch_point(sketch: &Sketch, spot: DVec2, radius: f64, ctx: &SnapContext, exclude: Option<&HashSet<EntityId>>) -> SnapResult {
    if ctx.off {
        return SnapResult::at(spot, None);
    }
    let excluded = |id: EntityId| exclude.is_some_and(|x| x.contains(&id));

    // The ends and centres of the pieces already there — a circle's centre too: what lands on
    // it is held there, as a hole's centre is held on the line it is placed along.
    let mut best: Option<SnapResult> = None;
    let mut best_d = radius;
    for e in &sketch.entities {
        if excluded(e.id) {
            continue;
        }
        for &at in anchors_of(&e.shape) {
            let p = anchor_point(&e.shape, at);
            let d = len2(p - spot);
            if d <= best_d {
                best_d = d;
                best = Some(SnapResult { anchor: Some(Anchor::new(e.id, at)), ..SnapResult::at(p, Some(SnapKind::Anchor)) });
            }
        }
    }
    if let Some(best) = best {
        return best;
    }

    // The sheet's zero — the part's origin projected onto the plane, as the sketch rule puts
    // it: a point to start a profile on, as every CAD sketcher offers it. A point that lands
    // there is held there by the coordinate planes whose lines cross at it, as Fusion's origin
    // point holds a coincidence; a zero no plane's line runs through is a place only.
    if len2(spot) <= radius {
        let origin: Vec<DatumPlane> = ctx.datums.unwrap_or(&[]).iter().filter(|t| t.offset(DVec2::ZERO).abs() < 1e-6).map(|t| t.plane).collect();
        return SnapResult { origin: Some(origin), ..SnapResult::at(DVec2::ZERO, Some(SnapKind::Origin)) };
    }

    // The middle of a line: an end that lands there is held to it, as CAD's midpoint relation
    // holds it — the line's centre anchor is its middle. It reaches less far than an end: a
    // line fitted to the slice has slice points all along it, and those are what is wanted
    // beside its middle.
    let mut mid_d = radius * MIDPOINT_REACH;
    for e in &sketch.entities {
        if excluded(e.id) || !matches!(e.shape, Shape::Line { .. }) {
            continue;
        }
        let p = anchor_point(&e.shape, AnchorAt::Center);
        let d = len2(p - spot);
        if d <= mid_d {
            mid_d = d;
            best = Some(SnapResult { anchor: Some(Anchor::new(e.id, AnchorAt::Center)), ..SnapResult::at(p, Some(SnapKind::Midpoint)) });
        }
    }
    if let Some(best) = best {
        return best;
    }

    // A coordinate plane's line: the foot of the spot on it, held there.
    if let Some(datums) = ctx.datums {
        if !datums.is_empty() {
            if let Some(trace) = trace_near(datums, spot, radius * DATUM_REACH) {
                return SnapResult { datum: Some(trace.plane), ..SnapResult::at(trace.foot(spot), Some(SnapKind::Datum)) };
            }
        }
    }

    // The body of a piece: what lands on it is held on it — a line drawn from the side of
    // another starts a T, and stays one. A piece drawn by hand or projected from the part is
    // taken before the slice: it is what was meant there, and the scan's points run along a
    // body's edge a hair off it — the edge, not the scan, is what a new piece is to be held
    // to. A piece fitted to the slice lies among its own points, which are what a free piece is
    // drawn along beside it: it is taken after them.
    let fitted = |e: &Entity| !e.points.is_empty();
    let on_body = |want_fitted: bool, best_d: &mut f64| -> Option<SnapResult> {
        let mut found = None;
        for e in &sketch.entities {
            if excluded(e.id) || matches!(e.shape, Shape::Point { .. }) || fitted(e) != want_fitted {
                continue;
            }
            let (point, dist) = nearest_on(&e.shape, spot);
            if dist <= *best_d {
                *best_d = dist;
                found = Some(SnapResult { piece: Some(e.id), ..SnapResult::at(point, Some(SnapKind::Entity)) });
            }
        }
        found
    };
    if let Some(found) = on_body(false, &mut best_d) {
        return found;
    }

    // A point of the slice: what a free piece is usually drawn along.
    if let Some(chains) = ctx.chains {
        let mut edge = None;
        for chain in chains {
            for p in chain {
                let d = len2(*p - spot);
                if d <= best_d {
                    best_d = d;
                    edge = Some(SnapResult::at(*p, Some(SnapKind::Edge)));
                }
            }
        }
        if let Some(edge) = edge {
            return edge;
        }
    }

    if let Some(found) = on_body(true, &mut best_d) {
        return found;
    }

    // The grid, last: it is there to steady a hand, not to fight the scan.
    if let Some(g) = ctx.grid.filter(|g| *g > 0.0) {
        let p = DVec2::new(js_round(spot.x / g) * g, js_round(spot.y / g) * g);
        if len2(p - spot) <= radius {
            return SnapResult::at(p, Some(SnapKind::Grid));
        }
    }
    SnapResult::at(spot, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datums::datum_traces;
    use crate::frame::SectionFrame;
    use crate::sketch::Rule;
    use crate::solver::{solve_sketch, SolveOptions};
    use glam::DVec3;

    fn v(x: f64, y: f64) -> DVec2 {
        DVec2::new(x, y)
    }

    fn scene() -> (Sketch, EntityId, EntityId) {
        let mut sketch = Sketch::default();
        let line = sketch.add_shape(Shape::Line { a: v(0.0, 0.0), b: v(10.0, 0.0) });
        let circle = sketch.add_shape(Shape::Circle { c: v(30.0, 30.0), r: 5.0 });
        (sketch, line, circle)
    }

    fn snap(sketch: &Sketch, spot: DVec2, radius: f64, ctx: SnapContext) -> SnapResult {
        snap_sketch_point(sketch, spot, radius, &ctx, None)
    }

    fn traces_at(origin: DVec3) -> Vec<DatumTrace> {
        datum_traces(&SectionFrame { origin, normal: DVec3::Z, u: DVec3::X, v: DVec3::Y })
    }

    #[test]
    fn takes_an_end_within_reach_and_names_it_and_a_circle_centre_as_an_anchor_to_hold_to() {
        let (sketch, line, circle) = scene();
        let end = snap(&sketch, v(10.3, 0.2), 1.0, SnapContext::default());
        assert_eq!(end.kind, Some(SnapKind::Anchor));
        assert_eq!(end.point, v(10.0, 0.0));
        assert_eq!(end.anchor, Some(Anchor::new(line, AnchorAt::B)));
        let centre = snap(&sketch, v(30.4, 29.8), 1.0, SnapContext::default());
        assert_eq!(centre.kind, Some(SnapKind::Anchor));
        assert_eq!(centre.point, v(30.0, 30.0));
        assert_eq!(centre.anchor, Some(Anchor::new(circle, AnchorAt::Center)));
        assert_eq!(snap_hold(&centre), Some(SnapHold::Anchor(Anchor::new(circle, AnchorAt::Center))));
    }

    #[test]
    fn names_the_piece_whose_body_it_lands_on_a_slice_point_and_the_grid_hold_nothing() {
        let (sketch, line, circle) = scene();
        let body = snap(&sketch, v(7.1, 0.3), 1.0, SnapContext::default());
        assert_eq!(body.kind, Some(SnapKind::Entity));
        assert_eq!(body.piece, Some(line));
        assert_eq!(snap_hold(&body), Some(SnapHold::Piece(line)));
        let rim = snap(&sketch, v(35.3, 30.0), 1.0, SnapContext::default());
        assert_eq!(rim.kind, Some(SnapKind::Entity));
        assert_eq!(snap_hold(&rim), Some(SnapHold::Piece(circle)));
        let chains = vec![vec![v(17.0, 0.4)]];
        assert_eq!(snap_hold(&snap(&sketch, v(17.1, 0.3), 1.0, SnapContext { chains: Some(&chains), ..Default::default() })), None);
        assert_eq!(snap_hold(&snap(&sketch, v(20.4, 20.4), 1.0, SnapContext { grid: Some(1.0), ..Default::default() })), None);
    }

    #[test]
    fn prefers_a_drawn_piece_to_a_slice_point_a_slice_point_to_a_fitted_piece_and_a_piece_to_the_grid() {
        let (sketch, line, _) = scene();
        let chains = vec![vec![v(7.0, 0.4), v(7.5, 0.4)]];
        // A line drawn by hand is what was meant where the scan runs beside it.
        let drawn = snap(&sketch, v(7.1, 0.3), 1.0, SnapContext { chains: Some(&chains), grid: Some(1.0), ..Default::default() });
        assert_eq!(drawn.kind, Some(SnapKind::Entity));
        assert_eq!(drawn.piece, Some(line));
        assert!(drawn.point.y.abs() < 5e-10 && (drawn.point.x - 7.1).abs() < 5e-10);
        // A line fitted to the slice lies among its points: they come first.
        let mut scanned = Sketch::default();
        let fit = scanned.add(Entity::new(Shape::Line { a: v(0.0, 0.0), b: v(10.0, 0.0) }).fitted(vec![v(2.0, 0.1), v(7.0, 0.4), v(9.0, -0.1)], 0.1));
        let edge = snap(&scanned, v(7.1, 0.3), 1.0, SnapContext { chains: Some(&chains), grid: Some(1.0), ..Default::default() });
        assert_eq!(edge.kind, Some(SnapKind::Edge));
        assert_eq!(edge.point, v(7.0, 0.4));
        assert_eq!(snap(&scanned, v(7.1, 0.3), 1.0, SnapContext { grid: Some(1.0), ..Default::default() }).piece, Some(fit));
        let grid = snap(&sketch, v(20.2, 20.3), 1.0, SnapContext { grid: Some(1.0), ..Default::default() });
        assert_eq!(grid.kind, Some(SnapKind::Grid));
        assert_eq!(grid.point, v(20.0, 20.0));
    }

    #[test]
    fn holds_an_end_on_a_projected_edge_of_the_part_though_the_scan_runs_a_hair_beside_it() {
        let mut on_face = Sketch::default();
        let mut e = Entity::new(Shape::Line { a: v(0.0, 10.0), b: v(40.0, 10.0) });
        e.fixed = true;
        e.projected = true;
        let edge = on_face.add(e);
        let slice: Vec<DVec2> = (0..=40).map(|i| v(i as f64, 10.1)).collect();
        let chains = vec![slice];
        let s = snap(&on_face, v(26.2, 10.05), 1.0, SnapContext { chains: Some(&chains), ..Default::default() });
        assert_eq!(s.kind, Some(SnapKind::Entity));
        assert_eq!(snap_hold(&s), Some(SnapHold::Piece(edge)));
        assert!((s.point.y - 10.0).abs() < 5e-10);
    }

    #[test]
    fn takes_the_middle_of_a_line_and_an_end_drawn_there_is_held_to_it() {
        let (sketch, line, _) = scene();
        let mid = snap(&sketch, v(5.2, 0.3), 1.0, SnapContext::default());
        assert_eq!(mid.kind, Some(SnapKind::Midpoint));
        assert_eq!(mid.point, v(5.0, 0.0));
        assert_eq!(mid.anchor, Some(Anchor::new(line, AnchorAt::Center)));
        // Less far than an end reaches: beside the middle the line itself.
        assert_eq!(snap(&sketch, v(5.7, 0.3), 1.0, SnapContext::default()).kind, Some(SnapKind::Entity));
        // Held: a line drawn from the middle follows when the first line moves.
        let mut held = Sketch::default();
        let mut base_e = Entity::new(Shape::Line { a: v(0.0, 0.0), b: v(10.0, 0.0) });
        base_e.fixed = true;
        let base = held.add(base_e);
        let branch = held.add_shape(Shape::Line { a: v(5.4, 0.2), b: v(5.0, 6.0) });
        held.constrain(Rule::Coincident { p: Anchor::new(branch, AnchorAt::A), q: Anchor::new(base, AnchorAt::Center) });
        assert!(solve_sketch(&mut held, SolveOptions::default()).ok);
        let a = anchor_point(&held.entity(branch).unwrap().shape, AnchorAt::A);
        assert!((a.x - 5.0).abs() < 5e-7 && a.y.abs() < 5e-7);
    }

    #[test]
    fn lands_nowhere_out_of_reach_and_nowhere_at_all_with_snapping_off() {
        let (sketch, ..) = scene();
        let far = snap(&sketch, v(20.4, 20.4), 0.5, SnapContext { grid: Some(1.0), ..Default::default() });
        assert_eq!(far.kind, None);
        assert_eq!(far.point, v(20.4, 20.4));
        let raw = snap(&sketch, v(10.1, 0.1), 1.0, SnapContext { grid: Some(1.0), off: true, ..Default::default() });
        assert_eq!(raw.kind, None);
        assert_eq!(raw.point, v(10.1, 0.1));
    }

    #[test]
    fn leaves_out_the_piece_being_dragged() {
        let (sketch, line, _) = scene();
        let own = snap_sketch_point(&sketch, v(10.3, 0.2), 1.0, &SnapContext::default(), Some(&HashSet::from([line])));
        assert_eq!(own.kind, None);
    }

    #[test]
    fn the_coordinate_planes_lines_take_a_point_near_them_and_name_the_plane() {
        let traces = traces_at(DVec3::ZERO);
        let ctx = SnapContext { datums: Some(&traces), ..Default::default() };
        let mut sketch = Sketch::default();
        let on_axis = snap(&sketch, v(12.0, 0.5), 1.0, ctx);
        assert_eq!(on_axis.kind, Some(SnapKind::Datum));
        assert_eq!(on_axis.point, v(12.0, 0.0));
        assert_eq!(on_axis.datum, Some(DatumPlane::Zx));
        assert_eq!(snap_hold(&on_axis), Some(SnapHold::Datum(DatumPlane::Zx)));
        // Less far than an end: at 0.9 of the reach the axis is not taken.
        assert_eq!(snap(&sketch, v(12.0, 0.9), 1.0, ctx).kind, None);
        // An end within reach wins over the axis; a slice point loses to it.
        let line = sketch.add_shape(Shape::Line { a: v(12.0, 0.6), b: v(30.0, 5.0) });
        assert_eq!(snap(&sketch, v(12.0, 0.5), 1.0, ctx).anchor, Some(Anchor::new(line, AnchorAt::A)));
        let chains = vec![vec![v(20.0, 0.4)]];
        assert_eq!(snap(&sketch, v(20.0, 0.5), 1.0, SnapContext { chains: Some(&chains), ..ctx }).kind, Some(SnapKind::Datum));
    }

    #[test]
    fn the_origin_is_a_snap_within_reach_after_the_ends_of_the_pieces() {
        let bare = snap(&Sketch::default(), v(0.4, -0.3), 1.0, SnapContext::default());
        assert_eq!(bare.kind, Some(SnapKind::Origin));
        assert_eq!(bare.point, v(0.0, 0.0));
        assert_eq!(bare.anchor, None);
        // Without the planes' lines, nothing holds a point at the zero.
        assert_eq!(snap_hold(&bare), None);
        assert_eq!(snap(&Sketch::default(), v(3.0, 0.0), 1.0, SnapContext::default()).kind, None);
        // An end already on the origin is what the click takes, and is held.
        let mut sketch = Sketch::default();
        let line = sketch.add_shape(Shape::Line { a: v(0.0, 0.0), b: v(10.0, 0.0) });
        let end = snap(&sketch, v(0.4, -0.3), 1.0, SnapContext::default());
        assert_eq!(end.kind, Some(SnapKind::Anchor));
        assert_eq!(end.anchor, Some(Anchor::new(line, AnchorAt::A)));
    }

    #[test]
    fn the_origin_holds_a_point_that_lands_on_it_by_the_planes_whose_lines_cross_there() {
        let traces = traces_at(DVec3::ZERO);
        let zero = snap(&Sketch::default(), v(0.4, -0.3), 1.0, SnapContext { datums: Some(&traces), ..Default::default() });
        assert_eq!(zero.kind, Some(SnapKind::Origin));
        assert_eq!(snap_hold(&zero), Some(SnapHold::Origin(vec![DatumPlane::Yz, DatumPlane::Zx])));
        let lifted = traces_at(DVec3::new(0.0, 0.0, 5.0));
        assert_eq!(snap_hold(&snap(&Sketch::default(), v(0.4, -0.3), 1.0, SnapContext { datums: Some(&lifted), ..Default::default() })), Some(SnapHold::Origin(vec![DatumPlane::Yz, DatumPlane::Zx])));
        let askew = traces_at(DVec3::new(1.0, 0.0, 0.0));
        assert_eq!(snap_hold(&snap(&Sketch::default(), v(0.4, -0.3), 1.0, SnapContext { datums: Some(&askew), ..Default::default() })), Some(SnapHold::Origin(vec![DatumPlane::Zx])));
    }
}
