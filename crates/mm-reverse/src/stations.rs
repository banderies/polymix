// SPDX-License-Identifier: AGPL-3.0-only
//! The stations of a loft taken from the scan. Ported from ScanRuler
//! (`plugins/reverse/core/stations.ts`), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! A styled part — a cover's wing, a fairing — is a loft through sections of it, and a loft
//! through sections only lofts cleanly when every station has the same pieces in the same
//! order: the same lines and arcs, tangent where the others are. So a station is fitted like
//! one already made: the scan's run on its plane cut into that station's pieces, kind for kind
//! (`segment::kinds_of_chain`), its relations copied, and the pieces it was closed with — a
//! line across the base, drawn rather than fitted — copied where they lie, the run's ends run
//! on to meet them. And a thin shell's inner station is its outer one moved in by the wall:
//! each line along its normal, each arc about its centre, the corners met again — the pieces
//! drawn to close it staying where they are, so the inner loft reaches the same base and the
//! cut opens it there.

use std::collections::{HashMap, HashSet};
use std::f64::consts::{PI, TAU};

use glam::DVec2;
use serde::{Deserialize, Serialize};

use crate::chain::view_chain;
use crate::corners::place_segments;
use crate::datums::{trace_of, DatumTrace};
use crate::fit2d::{unwind_angle, FitError};
use crate::js_fmt::{js_round, num, to_precision_num};
use crate::regions::find_regions;
use crate::segment::{kinds_of_chain, PieceKind, SegmentedChain};
use crate::sketch::{anchor_point, angle_of, arc_point, constraint_entities, ends_of, measure_rule, polyline, Anchor, AnchorAt, DistanceTo, EntityId, Entity, MirrorAxis, OnTarget, Rule, Shape, ShapeKind, Sketch, TOUCH_MM};

/// One piece of a station's profile, in the order the profile runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct StationPiece {
    pub id: EntityId,
    /// A line or an arc.
    pub kind: ShapeKind,
    /// The end the profile comes into the piece at, and the one it leaves by.
    pub from: AnchorAt,
    pub to: AnchorAt,
    /// Fitted to the scan — it holds slice points; otherwise drawn.
    pub fitted: bool,
}

/// A station's profile: its pieces end to end, a closed one starting where its run of fitted
/// pieces starts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StationLoop {
    pub pieces: Vec<StationPiece>,
    pub closed: bool,
}

fn other_end(end: AnchorAt) -> AnchorAt {
    if end == AnchorAt::A {
        AnchorAt::B
    } else {
        AnchorAt::A
    }
}

fn kind_name(kind: ShapeKind) -> &'static str {
    match kind {
        ShapeKind::Line => "line",
        ShapeKind::Arc => "arc",
        ShapeKind::Circle => "circle",
        ShapeKind::Point => "point",
        ShapeKind::Spline => "spline",
    }
}

fn a_kind(kind: ShapeKind) -> String {
    let k = kind_name(kind);
    format!("{} {k}", if k.starts_with(['a', 'e', 'i', 'o', 'u']) { "an" } else { "a" })
}

/// A station's piece shorter than this is no piece: the loft through it fails in the kernel,
/// or folds where the fit cut a corner into a sliver.
pub const STATION_MIN_PIECE_MM: f64 = 0.05;

/// An arc of a station sweeping less than this, degrees, is a corner the fit cut into an arc,
/// not a round.
const STATION_MIN_ARC_DEG: f64 = 5.0;

/// How far past the run's pieces a corner may land, in the run's extents: further, two nearly
/// parallel carriers met far off — a wall run into a boss beside it, met at u = 827.
const CORNER_REACH: f64 = 2.0;

/// A length or an angle as an answer gives it: microns kept where they are all there is.
fn mm(x: f64) -> String {
    num(if x.abs() < 0.01 { to_precision_num(x, 2) } else { js_round(x * 1000.0) / 1000.0 })
}

fn deg_of(rad: f64) -> String {
    num(js_round(((rad * 180.0) / PI) * 10.0) / 10.0)
}

/// `Math.sign(x || 1)`: the sign of a sweep, a zero one taken as positive.
fn sign1(x: f64) -> f64 {
    if x == 0.0 || x.is_nan() {
        1.0
    } else {
        x.signum()
    }
}

fn normalize2(a: DVec2) -> Option<DVec2> {
    let l = a.length();
    if !(l > 1e-12) {
        None
    } else {
        Some(a / l)
    }
}

/// The box round a run of points.
fn chain_box(points: &[DVec2]) -> (DVec2, DVec2) {
    let mut min = DVec2::splat(f64::INFINITY);
    let mut max = DVec2::splat(f64::NEG_INFINITY);
    for p in points {
        min = min.min(*p);
        max = max.max(*p);
    }
    (min, max)
}

/// A line's or an arc's length.
fn piece_length(s: &Shape) -> f64 {
    match *s {
        Shape::Line { a, b } => (b - a).length(),
        Shape::Arc { r, sweep, .. } => r * sweep.abs(),
        _ => 0.0,
    }
}

/// The way a profile runs at one end of a piece it comes in by `from`: a line's one way along
/// it, an arc's tangent turned the way it is run.
fn travel_at(s: &Shape, from: AnchorAt, at: AnchorAt) -> Option<DVec2> {
    match *s {
        Shape::Line { a, b } => normalize2(if from == AnchorAt::A { b - a } else { a - b }),
        Shape::Arc { start, sweep, .. } => {
            let angle = if at == AnchorAt::A { start } else { start + sweep };
            let turn = (if from == AnchorAt::A { 1.0 } else { -1.0 }) * sign1(sweep);
            Some(DVec2::new(-angle.sin() * turn, angle.cos() * turn))
        }
        _ => None,
    }
}

fn shape_of(sketch: &Sketch, id: EntityId) -> Option<Shape> {
    sketch.entity(id).map(|e| e.shape.clone())
}

type Ends = HashMap<EntityId, (AnchorAt, AnchorAt)>;

/// Which way a profile through `ids`, in that order, runs each piece: the end it leaves a
/// piece by is the one nearest the next piece's ends (the last piece of an open one: the end
/// away from the one before). Read off the pieces as they lie, so a solve that turned one round
/// shows it.
fn ends_along(entities: &[Entity], ids: &[EntityId], closed: bool) -> Ends {
    let mut out = Ends::new();
    let n = ids.len();
    let piece = |k: usize| entities.iter().find(|e| e.id == ids[k]).map(|e| e.shape.clone());
    let nearer = |k: usize, other: usize| -> AnchorAt {
        let (Some(pk), Some(o)) = (piece(k), piece(other)) else { return AnchorAt::A };
        let gap = |end: AnchorAt| {
            let p = anchor_point(&pk, end);
            (p - anchor_point(&o, AnchorAt::A)).length().min((p - anchor_point(&o, AnchorAt::B)).length())
        };
        if gap(AnchorAt::A) <= gap(AnchorAt::B) {
            AnchorAt::A
        } else {
            AnchorAt::B
        }
    };
    for k in 0..n {
        if n == 1 {
            out.insert(ids[k], (AnchorAt::A, AnchorAt::B));
        } else if k < n - 1 || closed {
            let to = nearer(k, (k + 1) % n);
            out.insert(ids[k], (other_end(to), to));
        } else {
            let from = nearer(k, k - 1);
            out.insert(ids[k], (from, other_end(from)));
        }
    }
    out
}

/// What is wrong with a station made like `template` (whose profile `lp` is), piece by piece,
/// in words — empty when nothing is: a piece under STATION_MIN_PIECE_MM, a line run backwards
/// against the template's, an arc bending the other way or sweeping under STATION_MIN_ARC_DEG.
/// Each is a fold a loft would go through: the run did not fall into the template's pieces
/// there. `ids` names the station's pieces in the loop's order where they are not the
/// template's ids. Read off the pieces as they lie — before the solve, and after it, which can
/// fold a piece the fit left whole.
pub fn station_faults(template: &Sketch, lp: &StationLoop, made: &[Entity], ids: Option<&[EntityId]>) -> Vec<String> {
    let own_ids: Vec<EntityId> = lp.pieces.iter().map(|p| p.id).collect();
    let ids = ids.unwrap_or(&own_ids);
    let ends = ends_along(made, ids, lp.closed);
    let mut out = Vec::new();
    for (k, p) in lp.pieces.iter().enumerate() {
        let Some(&id) = ids.get(k) else { continue };
        let (Some(e), Some(t)) = (made.iter().find(|x| x.id == id).map(|x| x.shape.clone()), shape_of(template, p.id)) else { continue };
        if e.kind() != t.kind() || !matches!(e.kind(), ShapeKind::Line | ShapeKind::Arc) {
            continue;
        }
        let what = format!("piece {id} ({})", a_kind(e.kind()));
        let length = piece_length(&e);
        if length < STATION_MIN_PIECE_MM {
            out.push(format!("{what} is {} mm long — under {} mm", mm(length), num(STATION_MIN_PIECE_MM)));
            continue;
        }
        let Some(&(way_from, way_to)) = ends.get(&id) else { continue };
        let now = travel_at(&e, way_from, way_to);
        let was = travel_at(&t, p.from, p.to);
        if let Shape::Arc { sweep, .. } = e {
            let Shape::Arc { sweep: had_sweep, .. } = t else { continue };
            let sense = (if way_from == AnchorAt::A { 1.0 } else { -1.0 }) * sign1(sweep);
            let had = (if p.from == AnchorAt::A { 1.0 } else { -1.0 }) * sign1(had_sweep);
            if sense != had {
                out.push(format!("{what} bends the other way from the template’s piece {}", p.id));
            } else if sweep.abs() < STATION_MIN_ARC_DEG * PI / 180.0 {
                out.push(format!("{what} sweeps {}° — under {}°", deg_of(sweep.abs()), num(STATION_MIN_ARC_DEG)));
            }
        } else if let (Some(now), Some(was)) = (now, was) {
            if now.dot(was) < 0.0 {
                out.push(format!("{what} runs {} mm backwards against the template’s piece {}", mm(length), p.id));
            }
        }
    }
    out
}

/// What a station that folds is told to do.
pub const STATION_FOLD_HELP: &str = "the scan’s run does not fall into the template’s pieces there, and a loft through it folds. Point the fit with within or facing, give it a tolerance, or draw this station.";

/// A sketch's profile as a station: its lines and arcs end to end (what is drawn for
/// construction, and points, left aside), in the order they run — a closed profile turned to
/// start where its one run of fitted pieces starts. Why not, in words, for a sketch that holds
/// a circle, branches, or is more than one chain, or whose fitted pieces are more than one run
/// (unless `any_runs`).
pub fn station_loop(entities: &[Entity], name: &str, any_runs: bool) -> Result<StationLoop, FitError> {
    let own: Vec<&Entity> = entities.iter().filter(|e| !e.construction && e.kind() != ShapeKind::Point).collect();
    if own.is_empty() {
        return Err(FitError::new(format!("{name} has no pieces.")));
    }
    if let Some(odd) = own.iter().find(|e| !matches!(e.kind(), ShapeKind::Line | ShapeKind::Arc)) {
        return Err(FitError::new(format!("{name} has {} (piece {}) — a station is lines and arcs end to end.", a_kind(odd.kind()), odd.id)));
    }
    let pieces = own;
    // A piece shorter than touch has its two ends at one corner, and its neighbours meet across
    // it: named, not read as three pieces meeting.
    if let Some(sliver) = pieces.iter().find(|e| (anchor_point(&e.shape, AnchorAt::A) - anchor_point(&e.shape, AnchorAt::B)).length() <= TOUCH_MM && piece_length(&e.shape) <= TOUCH_MM) {
        return Err(FitError::new(format!(
            "{name}'s piece {} ({}) is {} mm long — shorter than the {} mm two ends meet within, so the pieces either side meet across it. Take it out (reverse.sketch.edit remove) and join its neighbours, or fit the station again.",
            sliver.id,
            a_kind(sliver.kind()),
            mm(piece_length(&sliver.shape)),
            num(TOUCH_MM)
        )));
    }
    // Each piece's two ends, a then b.
    let ends: Vec<(usize, AnchorAt, DVec2)> = pieces.iter().enumerate().flat_map(|(i, e)| [AnchorAt::A, AnchorAt::B].map(|end| (i, end, anchor_point(&e.shape, end)))).collect();
    let partners: Vec<Vec<usize>> = ends.iter().map(|p| (0..ends.len()).filter(|&q| ends[q].0 != p.0 && (ends[q].2 - p.2).length() <= TOUCH_MM).collect()).collect();
    if partners.iter().any(|x| x.len() > 1) {
        return Err(FitError::new(format!("{name} branches — three pieces meet at a point; a station is one chain of pieces end to end.")));
    }
    let start = (0..ends.len()).find(|&i| partners[i].is_empty()).unwrap_or(0);
    let mut order: Vec<StationPiece> = Vec::new();
    let mut seen: HashSet<usize> = HashSet::new();
    let mut at = start;
    let mut closed = false;
    loop {
        let (pi, end, _) = ends[at];
        let e = pieces[pi];
        seen.insert(pi);
        let to = other_end(end);
        order.push(StationPiece { id: e.id, kind: e.kind(), from: end, to, fitted: e.is_fitted() });
        let to_index = pi * 2 + usize::from(to == AnchorAt::B);
        let Some(&next) = partners[to_index].first() else { break };
        if seen.contains(&ends[next].0) {
            closed = ends[next].0 == ends[start].0;
            break;
        }
        at = next;
    }
    if order.len() < pieces.len() {
        return Err(FitError::new(format!(
            "{name} is more than one chain — a station is one chain of pieces end to end (its pieces {} make one; {} more lie apart).",
            order.iter().map(|p| p.id.to_string()).collect::<Vec<_>>().join(", "),
            pieces.len() - order.len()
        )));
    }
    let n = order.len();
    if closed && order.iter().any(|p| p.fitted) && order.iter().any(|p| !p.fitted) {
        if let Some(first) = (0..n).find(|&i| order[i].fitted && !order[(i + n - 1) % n].fitted) {
            order.rotate_left(first);
        }
    }
    let runs = (0..n).filter(|&i| order[i].fitted && (i == 0 || !order[i - 1].fitted)).count();
    if runs > 1 && !any_runs {
        return Err(FitError::new(format!("{name}'s fitted pieces are {runs} runs with drawn pieces between them — a station is one run fitted to the scan, closed by what is drawn.")));
    }
    Ok(StationLoop { pieces: order, closed })
}

/// Where two pieces' carriers cross nearest `near` — a line's whole line, an arc's whole
/// circle — or, for a line that runs past a circle, the point on the line nearest it: where a
/// line runs tangent into an arc. `None` for two lines that run parallel, or two circles that
/// do not meet.
fn meeting(a: &Shape, b: &Shape, near: DVec2) -> Option<DVec2> {
    let nearest = |cs: [DVec2; 2]| if (cs[1] - near).length() < (cs[0] - near).length() { cs[1] } else { cs[0] };
    match (a.clone(), b.clone()) {
        (Shape::Line { a: a0, b: a1 }, Shape::Line { a: b0, b: b1 }) => {
            let da = a1 - a0;
            let db = b1 - b0;
            let denom = da.perp_dot(db);
            if denom.abs() < 1e-9 * da.length() * db.length() {
                return None;
            }
            Some(a0 + da * ((b0 - a0).perp_dot(db) / denom))
        }
        (Shape::Arc { c: ac, r: ar, .. }, Shape::Arc { c: bc, r: br, .. }) => {
            let d = (bc - ac).length();
            if d < 1e-9 || d > ar + br || d < (ar - br).abs() {
                return None;
            }
            let along = (ar * ar - br * br + d * d) / (2.0 * d);
            let h = (ar * ar - along * along).max(0.0).sqrt();
            let u = (bc - ac) / d;
            let foot = ac + u * along;
            Some(nearest([foot + u.perp() * h, foot + u.perp() * -h]))
        }
        (Shape::Line { a: la, b: lb }, Shape::Arc { c, r, .. }) | (Shape::Arc { c, r, .. }, Shape::Line { a: la, b: lb }) => {
            let d = normalize2(lb - la)?;
            let foot = la + d * (c - la).dot(d);
            let off = (c - foot).length();
            if off >= r {
                return Some(foot);
            }
            let h = (r * r - off * off).sqrt();
            Some(nearest([foot + d * h, foot + d * -h]))
        }
        _ => None,
    }
}

/// `raw` (in [0, 2π)) or a full turn either side of it, whichever lies nearest the sweep the
/// arc had.
fn nearest_sweep(raw: f64, had: f64) -> f64 {
    [raw, raw + TAU, raw - TAU].into_iter().filter(|s| *s > 1e-9 && *s <= TAU + 1e-9).fold(raw, |x, y| if (y - had).abs() < (x - had).abs() { y } else { x })
}

/// A piece's end put at a point: a line's outright, an arc's at the angle the point lies at
/// about its centre.
fn set_end(s: &mut Shape, at: AnchorAt, p: DVec2) {
    match s {
        Shape::Line { a, b } => {
            if at == AnchorAt::A {
                *a = p;
            } else {
                *b = p;
            }
        }
        Shape::Arc { c, start, sweep, .. } => {
            let angle = angle_of(*c, p);
            if at == AnchorAt::A {
                let end = *start + *sweep;
                let raw = unwind_angle(end, angle) - angle;
                *start = angle;
                *sweep = nearest_sweep(raw, *sweep);
            } else {
                *sweep = nearest_sweep(unwind_angle(angle, *start) - *start, *sweep);
            }
        }
        _ => {}
    }
}

/// A corner met again: the two pieces, and how far it landed from the farther of the two ends
/// it was met from.
#[derive(Debug, Clone, Copy)]
struct Met {
    a: EntityId,
    b: EntityId,
    reach: f64,
}

/// The corners of a profile met again after its pieces moved: each two pieces in a row brought
/// to where their carriers cross, nearest the corner `near` gives; pieces of `still` keep to
/// their carriers and their ends slide along them, as do the others. Each corner met, with how
/// far it landed from the farther of the two ends it was met from.
fn meet_corners(sketch: &mut Sketch, lp: &StationLoop, ends: &Ends, still: &dyn Fn(EntityId) -> bool, near: &dyn Fn(usize, DVec2, DVec2) -> DVec2) -> Vec<Met> {
    let n = lp.pieces.len();
    let mut met = Vec::new();
    let count = if lp.closed { n } else { n.saturating_sub(1) };
    for i in 0..count {
        let pa_ = lp.pieces[i];
        let pb_ = lp.pieces[(i + 1) % n];
        if still(pa_.id) && still(pb_.id) {
            continue;
        }
        let (Some(ea), Some(eb)) = (shape_of(sketch, pa_.id), shape_of(sketch, pb_.id)) else { continue };
        let (Some(&(_, at_a)), Some(&(at_b, _))) = (ends.get(&pa_.id), ends.get(&pb_.id)) else { continue };
        let pa = anchor_point(&ea, at_a);
        let pb = anchor_point(&eb, at_b);
        let spot = near(i, pa, pb);
        let m = meeting(&ea, &eb, spot).unwrap_or(spot);
        if let Some(e) = sketch.entity_mut(pa_.id) {
            set_end(&mut e.shape, at_a, m);
        }
        if let Some(e) = sketch.entity_mut(pb_.id) {
            set_end(&mut e.shape, at_b, m);
        }
        met.push(Met { a: pa_.id, b: pb_.id, reach: (m - pa).length().max((m - pb).length()) });
    }
    met
}

/// A station fitted like another: its sketch, and whether the run was taken from its far end.
#[derive(Debug, Clone, PartialEq)]
pub struct LikeStation {
    pub sketch: Sketch,
    pub reversed: bool,
}

/// The anchors of a relation mapped, where it has any.
fn map_anchors(rule: &mut Rule, f: &dyn Fn(Anchor) -> Anchor) {
    match rule {
        Rule::Coincident { p, q } | Rule::Symmetric { p, q, .. } => {
            *p = f(*p);
            *q = f(*q);
        }
        Rule::PointOn { p, .. } => *p = f(*p),
        Rule::Distance { p, q, .. } => {
            *p = f(*p);
            if let DistanceTo::Anchor(a) = q {
                *a = f(*a);
            }
        }
        _ => {}
    }
}

/// The scan's run on a station's plane fitted like `template`, whose profile `lp` is: the run
/// cut into the template's fitted pieces, kind for kind in their order, each fitted to its own
/// points and given the template piece's id; the template's drawn pieces copied where they lie
/// in [u, v]; the corners met where the pieces cross — a drawn piece keeping to its line, the
/// run's end running on or back to it — and the template's relations among these pieces
/// copied, its dimensions on drawn pieces alone too (a fitted piece's size is the scan's to
/// say). The run is taken the way round that fits the pieces clearly better, or else lands the
/// pieces' ends nearer the template's. A template with no fitted piece is fitted whole, when it
/// is open. Why not, in words.
pub fn like_station(template: &Sketch, lp: &StationLoop, chain: &[DVec2], tolerance: Option<f64>) -> Result<LikeStation, FitError> {
    let all = lp.pieces.iter().all(|p| !p.fitted);
    if all && lp.closed {
        return Err(FitError::new("it is closed and has no piece fitted to the scan, so its run cannot be told from what closes it — fit it to the scan, or leave it open."));
    }
    let pieces: Vec<StationPiece> = if all { lp.pieces.iter().map(|p| StationPiece { fitted: true, ..*p }).collect() } else { lp.pieces.clone() };
    let stretch: Vec<StationPiece> = pieces.iter().filter(|p| p.fitted).copied().collect();
    let whole = lp.closed && stretch.len() == pieces.len();
    let view = view_chain(chain);
    let (points, closed) = (view.points, view.closed);
    if whole && !closed {
        return Err(FitError::new("it is a closed outline fitted whole, and this run is open — fit a closed chain (closeGaps), or close the template with a drawn piece."));
    }
    if !whole && closed && stretch.len() < pieces.len() {
        return Err(FitError::new("this chain is closed and the template’s run is open, closed by drawn pieces — keep to one run (skin, facing or within)."));
    }
    let tmpl = |id: EntityId| shape_of(template, id).ok_or_else(|| FitError::new(format!("the template has no piece {id}.")));
    let Some(head) = stretch.first() else { return Err(FitError::new("it has no pieces.")) };
    let first = anchor_point(&tmpl(head.id)?, head.from);
    let kinds: Vec<PieceKind> = stretch.iter().map(|p| if p.kind == ShapeKind::Arc { PieceKind::Arc } else { PieceKind::Line }).collect();
    // The run from its first point and from its last; a closed one from the point nearest
    // where the template's outline starts.
    let mut back = points.clone();
    back.reverse();
    let ways: Vec<Result<(SegmentedChain, f64), FitError>> = [points.clone(), back]
        .into_iter()
        .map(|pts| {
            let mut run = pts.clone();
            if whole {
                let k = (0..pts.len()).fold(0, |best, i| if (pts[i] - first).length() < (pts[best] - first).length() { i } else { best });
                run.rotate_left(k);
            }
            kinds_of_chain(&run, &kinds, tolerance, whole)
        })
        .collect();
    let fits: Vec<usize> = (0..2).filter(|&i| ways[i].is_ok()).collect();
    if fits.is_empty() {
        return Err(match &ways[0] {
            Err(e) => e.clone(),
            Ok(_) => FitError::new("the run fits no way round."),
        });
    }
    // How far the cut's piece ends land from the template's.
    let off = |cut: &SegmentedChain| -> f64 {
        cut.segments
            .iter()
            .enumerate()
            .map(|(k, s)| {
                let Ok(t) = tmpl(stretch[k].id) else { return 0.0 };
                (s.points[0] - anchor_point(&t, stretch[k].from)).length() + (s.points[s.points.len() - 1] - anchor_point(&t, stretch[k].to)).length()
            })
            .sum()
    };
    let cut_of = |i: usize| ways[i].as_ref().ok();
    let mut chosen = fits[0];
    if fits.len() == 2 {
        if let (Some((f, fc)), Some((r, rc))) = (cut_of(0), cut_of(1)) {
            let clearly = 2.0;
            chosen = if fc * clearly < *rc {
                0
            } else if rc * clearly < *fc {
                1
            } else if off(f) <= off(r) {
                0
            } else {
                1
            };
        }
    }
    let reversed = chosen == 1;
    let Some((cut, _)) = cut_of(chosen) else { return Err(FitError::new("the run fits no way round.")) };

    // The pieces: the run's under the template's ids, the drawn ones copied.
    let mut scratch = Sketch::default();
    let placed = place_segments(&mut scratch, cut);
    let mut out = Sketch::default();
    let mut ends = Ends::new();
    let mut k = 0;
    for p in &pieces {
        if p.fitted {
            let Some(made) = placed.get(k).and_then(|pl| scratch.entity(pl.id)) else { return Err(FitError::new("the run's cut has fewer pieces than the template.")) };
            out.entities.push(Entity { id: p.id, ..made.clone() });
            ends.insert(p.id, (placed[k].from, placed[k].to));
            k += 1;
        } else {
            let Some(e) = template.entity(p.id) else { return Err(FitError::new(format!("the template has no piece {}.", p.id))) };
            out.entities.push(e.clone());
            ends.insert(p.id, (p.from, p.to));
        }
    }
    // The corners: a drawn piece's end stays on its line, the run's end runs to it; two of the
    // run's pieces meet about where the cut broke them.
    let fitted: HashSet<EntityId> = pieces.iter().filter(|p| p.fitted).map(|p| p.id).collect();
    let n = pieces.len();
    let station = StationLoop { pieces: pieces.clone(), closed: lp.closed };
    let near = |i: usize, pa: DVec2, pb: DVec2| {
        if !fitted.contains(&pieces[i].id) {
            pb
        } else if !fitted.contains(&pieces[(i + 1) % n].id) {
            pa
        } else {
            (pa + pb) / 2.0
        }
    };
    let met = meet_corners(&mut out, &station, &ends, &|id| !fitted.contains(&id), &near);
    // A corner met far past the run: two carriers nearly parallel, met where nothing of the scan
    // is — the station would loft "ok" at many times its volume.
    let (lo, hi) = chain_box(&points);
    let extent = (hi - lo).length();
    let far = met.iter().fold(None::<Met>, |x, c| if c.reach > CORNER_REACH * extent && x.is_none_or(|x| c.reach > x.reach) { Some(*c) } else { x });
    if let Some(far) = far {
        return Err(FitError::new(format!(
            "the corner of pieces {} and {} lands {} mm past the run’s ends — over {}× the run’s extent ({} mm): the two run nearly parallel and meet far off, likely because the run takes in something beside the station (a boss, a rib). Point it with within or facing.",
            far.a,
            far.b,
            mm(far.reach),
            num(CORNER_REACH),
            mm(extent)
        )));
    }
    let faults = station_faults(template, &station, &out.entities, None);
    if !faults.is_empty() {
        return Err(FitError::new(format!("{} — {STATION_FOLD_HELP}", faults.join("; "))));
    }

    // The template's relations among these pieces, their anchors read by which way each piece
    // now runs.
    let by_id: HashMap<EntityId, StationPiece> = pieces.iter().map(|p| (p.id, *p)).collect();
    let map_at = |a: Anchor| -> Anchor {
        let Some(p) = by_id.get(&a.entity) else { return a };
        if !p.fitted || a.at == AnchorAt::Center {
            return a;
        }
        let Some(&(now_from, now_to)) = ends.get(&p.id) else { return a };
        Anchor { entity: a.entity, at: if a.at == p.from { now_from } else { now_to } }
    };
    let mut next = out.entities.iter().map(|e| e.id).max().unwrap_or(0) + 1;
    for c in &template.constraints {
        let named = constraint_entities(&c.rule);
        if !named.iter().all(|id| by_id.contains_key(id)) {
            continue;
        }
        if c.rule.is_dimension() && named.iter().any(|id| fitted.contains(id)) {
            continue;
        }
        let mut copy = c.clone();
        map_anchors(&mut copy.rule, &map_at);
        copy.id = next;
        next += 1;
        out.constraints.push(copy);
    }
    out.next_id = next;
    Ok(LikeStation { sketch: out, reversed })
}

/// What a station's fit was asked: the gaps it closed and which of within, skin and facing it
/// was pointed with.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RunFit {
    pub close_gaps: Option<f64>,
    pub within: bool,
    pub skin: bool,
    pub facing: bool,
}

/// What a station's fit was given and found, for a refusal to say: the run taken, the chain it
/// was taken from after closeGaps bridged what it did (and the chains that chain was made of),
/// the cut's chains as they came, how many runs within, skin or facing kept, and the fit as it
/// was asked.
#[derive(Debug, Clone, Copy)]
pub struct RunFacts<'a> {
    pub run: &'a [DVec2],
    pub joined_chain: &'a [DVec2],
    pub joined_from: &'a [usize],
    pub chains: &'a [Vec<DVec2>],
    pub runs: usize,
    pub fit: RunFit,
}

/// How many of a chain's gaps a refusal names, the widest first.
const GAPS_NAMED: usize = 5;

fn closed_chain(c: &[DVec2]) -> bool {
    c.len() > 2 && (c[0] - c[c.len() - 1]).length() < 1e-9
}

/// The gaps round an outline: those closeGaps bridged and those it left open, widest first.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GapsRound {
    pub bridged: Vec<f64>,
    pub open: Vec<f64>,
}

/// The gaps round the outline a run came from: each open chain's ends paired with the nearest
/// free end — another chain's, or its own other end — the nearest pairs first, as closeGaps
/// pairs them but with no limit, and the pairs that link the run's chains to each other and on
/// round the outline kept. Those within closeGaps are the ones it bridged; the rest stayed
/// open. A pair longer than the outline's own extent links it to something else, not across a
/// gap of it, and is left out. Widest first.
pub fn gaps_round(chains: &[Vec<DVec2>], from: &[usize], close_gaps: f64) -> GapsRound {
    let n = chains.len();
    let open: Vec<bool> = chains.iter().map(|c| c.len() >= 2 && !closed_chain(c)).collect();
    let end = |e: usize| -> DVec2 {
        let c = &chains[e >> 1];
        if e & 1 == 1 {
            c[c.len() - 1]
        } else {
            c[0]
        }
    };
    let mut pairs: Vec<(usize, usize, f64)> = Vec::new();
    for i in 0..n {
        if !open[i] {
            continue;
        }
        if chains[i].len() >= 3 {
            pairs.push((2 * i, 2 * i + 1, (end(2 * i) - end(2 * i + 1)).length()));
        }
        for j in i + 1..n {
            if !open[j] {
                continue;
            }
            for e in [2 * i, 2 * i + 1] {
                for f in [2 * j, 2 * j + 1] {
                    pairs.push((e, f, (end(e) - end(f)).length()));
                }
            }
        }
    }
    pairs.sort_by(|a, b| a.2.total_cmp(&b.2));
    let mut mate: Vec<Option<usize>> = vec![None; 2 * n];
    let mut width = vec![0.0; 2 * n];
    for &(e, f, d) in &pairs {
        if mate[e].is_some() || mate[f].is_some() {
            continue;
        }
        mate[e] = Some(f);
        mate[f] = Some(e);
        width[e] = d;
        width[f] = d;
    }
    // The outline the run's chains make, followed from them across every pair; its extent
    // bounds what is a gap of it.
    let mut seen: Vec<usize> = Vec::new();
    let mut links: Vec<usize> = Vec::new();
    let mut todo: Vec<usize> = from.iter().copied().filter(|&i| i < n).collect();
    while let Some(c) = todo.pop() {
        if seen.contains(&c) {
            continue;
        }
        seen.push(c);
        for e in [2 * c, 2 * c + 1] {
            let Some(m) = mate[e] else { continue };
            if !links.contains(&e.min(m)) {
                links.push(e.min(m));
            }
            todo.push(m >> 1);
        }
    }
    let pts: Vec<DVec2> = seen.iter().flat_map(|&c| chains[c].iter().copied()).collect();
    let (lo, hi) = chain_box(&pts);
    let extent = (hi - lo).length();
    let mut widths: Vec<f64> = links.iter().map(|&e| width[e]).filter(|&d| d > 1e-9 && d < extent).collect();
    widths.sort_by(|a, b| b.total_cmp(a));
    GapsRound { bridged: widths.iter().copied().filter(|&d| d <= close_gaps).collect(), open: widths.iter().copied().filter(|&d| d > close_gaps).collect() }
}

/// What a station's refusal adds of the numbers it decided on. For a run that is open where the
/// station is a closed outline fitted whole: how far apart its ends stand, the closeGaps given,
/// the gaps it bridged and those still open, widest first — or, where within, skin or facing
/// kept a run of a chain that had closed, that it was they that opened it. For any other
/// refusal of a run closeGaps bridged: the bridges, which a station's pieces may not fall on.
/// Empty when there is nothing to add.
pub fn run_facts_words(lp: &StationLoop, f: &RunFacts) -> String {
    let list = |ds: &[f64]| {
        let named: Vec<String> = ds.iter().take(GAPS_NAMED).map(|d| mm(*d)).collect();
        format!("{}{}", named.join(", "), if ds.len() > GAPS_NAMED { format!(" and {} more", ds.len() - GAPS_NAMED) } else { String::new() })
    };
    let GapsRound { bridged, open } = gaps_round(f.chains, f.joined_from, f.fit.close_gaps.unwrap_or(0.0));
    let given = match f.fit.close_gaps {
        Some(g) => format!("closeGaps {}", mm(g)),
        None => "no closeGaps given".to_string(),
    };
    let bridges = if bridged.is_empty() { format!("{given}, none bridged") } else { format!("{given} bridged {} gap{} ({} mm)", bridged.len(), if bridged.len() == 1 { "" } else { "s" }, list(&bridged)) };
    let whole = lp.closed && lp.pieces.iter().all(|p| p.fitted);
    if !whole || closed_chain(f.run) {
        return if bridged.is_empty() { String::new() } else { format!(" ({bridges} — a bridge is a straight run across a hole, and the station’s pieces may not fall on it)") };
    }
    let (Some(first), Some(last)) = (f.run.first(), f.run.last()) else { return String::new() };
    let ends = mm((*first - *last).length());
    let by: Vec<&str> = [("within", f.fit.within), ("skin", f.fit.skin), ("facing", f.fit.facing)].into_iter().filter(|x| x.1).map(|x| x.0).collect();
    if closed_chain(f.joined_chain) && !by.is_empty() {
        return format!(
            " The run’s ends stand {ends} mm apart: its chain closed ({bridges}), and {} split it into {} run{}, the longest taken — fit the closed chain without {}, or close the template with a drawn piece where the run stops.",
            by.join(" and "),
            f.runs,
            if f.runs == 1 { "" } else { "s" },
            if by.len() == 1 { "it" } else { "them" }
        );
    }
    let still = if open.is_empty() { String::new() } else { format!("; still open: {} mm", list(&open)) };
    let reach = if open.is_empty() {
        String::new()
    } else {
        format!(" closeGaps {} bridges them where they are holes in the scan; where they are the part’s own (a gap between keys), close the template with a drawn piece across them instead.", mm(open[0] * 1.05))
    };
    format!(" The run’s ends stand {ends} mm apart; {bridges}{still}.{reach}")
}

/// How far off a relation may be and still be held, mm (or, for one of directions, the sine of
/// the angle).
const HOLDS: f64 = 1e-5;

/// Whether a relation holds on the sketch as it lies. A dimension is not asked.
fn holds(sketch: &Sketch, rule: &Rule, datums: &[DatumTrace]) -> bool {
    let e = |id: EntityId| shape_of(sketch, id);
    let pt = |a: Anchor| e(a.entity).map(|s| anchor_point(&s, a.at));
    let dir = |id: EntityId| match e(id) {
        Some(Shape::Line { a, b }) => normalize2(b - a),
        _ => None,
    };
    let off_line = |p: DVec2, a: DVec2, d: DVec2| (p - a).dot(d.perp()).abs();
    let carrier = |on: &OnTarget| -> Option<(DVec2, DVec2)> {
        match *on {
            OnTarget::Line(l) => match e(l) {
                Some(Shape::Line { a, .. }) => dir(l).map(|d| (a, d)),
                _ => None,
            },
            OnTarget::Datum(plane) => trace_of(datums, plane).map(|t| t.line()),
            OnTarget::Curve(_) => None,
        }
    };
    let round = |s: &Shape| matches!(s, Shape::Arc { .. } | Shape::Circle { .. });
    match *rule {
        Rule::Coincident { p, q } => match (pt(p), pt(q)) {
            (Some(a), Some(b)) => (a - b).length() <= HOLDS,
            _ => true,
        },
        Rule::Horizontal { entity } => dir(entity).is_none_or(|d| d.y.abs() <= HOLDS),
        Rule::Vertical { entity } => dir(entity).is_none_or(|d| d.x.abs() <= HOLDS),
        Rule::Parallel { a, b } => match (dir(a), dir(b)) {
            (Some(a), Some(b)) => a.perp_dot(b).abs() <= HOLDS,
            _ => true,
        },
        Rule::Perpendicular { a, b } => match (dir(a), dir(b)) {
            (Some(a), Some(b)) => a.dot(b).abs() <= HOLDS,
            _ => true,
        },
        Rule::Tangent { a, b } => {
            let (Some(a), Some(b)) = (e(a), e(b)) else { return true };
            match (a, b) {
                (Shape::Line { a: la, b: lb }, r) | (r, Shape::Line { a: la, b: lb }) => {
                    let Some(d) = normalize2(lb - la) else { return true };
                    if !round(&r) {
                        return true;
                    }
                    let (Some(c), Some(rr)) = (r.centre(), r.radius()) else { return true };
                    (off_line(c, la, d) - rr).abs() <= HOLDS
                }
                (p, q) => {
                    if !round(&p) || !round(&q) {
                        return true;
                    }
                    let (Some(pc), Some(pr), Some(qc), Some(qr)) = (p.centre(), p.radius(), q.centre(), q.radius()) else { return true };
                    let d = (qc - pc).length();
                    (d - (pr + qr)).abs().min((d - (pr - qr).abs()).abs()) <= HOLDS
                }
            }
        }
        Rule::Concentric { a, b } => match (e(a), e(b)) {
            (Some(a), Some(b)) if round(&a) && round(&b) => match (a.centre(), b.centre()) {
                (Some(ac), Some(bc)) => (ac - bc).length() <= HOLDS,
                _ => true,
            },
            _ => true,
        },
        Rule::Equal { a, b } => match (e(a), e(b)) {
            (Some(Shape::Line { a: a0, b: a1 }), Some(Shape::Line { a: b0, b: b1 })) => ((a1 - a0).length() - (b1 - b0).length()).abs() <= HOLDS,
            (Some(a), Some(b)) => match (a.radius(), b.radius()) {
                (Some(ar), Some(br)) => (ar - br).abs() <= HOLDS,
                _ => true,
            },
            _ => true,
        },
        Rule::PointOn { p, on, .. } => {
            let Some(p) = pt(p) else { return true };
            match on {
                OnTarget::Curve(id) => match e(id) {
                    Some(r) => match (r.centre(), r.radius()) {
                        (Some(c), Some(rr)) => ((p - c).length() - rr).abs() <= HOLDS,
                        _ => true,
                    },
                    None => true,
                },
                OnTarget::Datum(plane) => trace_of(datums, plane).is_none_or(|t| t.offset(p).abs() <= HOLDS),
                OnTarget::Line(_) => carrier(&on).is_none_or(|(a, d)| off_line(p, a, d) <= HOLDS),
            }
        }
        Rule::Collinear { line, on } => {
            let Some(Shape::Line { a, b }) = e(line) else { return true };
            carrier(&on).is_none_or(|(o, d)| off_line(a, o, d).max(off_line(b, o, d)) <= HOLDS)
        }
        Rule::Symmetric { p, q, about } => {
            let (Some(p), Some(q)) = (pt(p), pt(q)) else { return true };
            match about {
                MirrorAxis::X => (p.y + q.y).abs() <= HOLDS && (p.x - q.x).abs() <= HOLDS,
                MirrorAxis::Y => (p.x + q.x).abs() <= HOLDS && (p.y - q.y).abs() <= HOLDS,
                MirrorAxis::Line(l) => carrier(&OnTarget::Line(l)).is_none_or(|(a, d)| ((p + q) / 2.0 - a).dot(d.perp()).abs() <= HOLDS && (q - p).dot(d).abs() <= HOLDS),
                MirrorAxis::Datum(plane) => carrier(&OnTarget::Datum(plane)).is_none_or(|(a, d)| ((p + q) / 2.0 - a).dot(d.perp()).abs() <= HOLDS && (q - p).dot(d).abs() <= HOLDS),
            }
        }
        _ => true,
    }
}

/// How much more an inset arc's sweep may change than its corners say, degrees.
const INSET_ARC_SLACK_DEG: f64 = 15.0;

/// How far a moved arc's sweep may change when its profile is moved in: nothing at a corner its
/// neighbour runs tangent into it at — both move alike and stay tangent — up to about the
/// angle between them at one they meet at an angle (twice it, for an inset deep against the
/// radius), a quarter turn at one whose neighbour is kept, its end sliding round to where the
/// kept line cuts the new circle; and INSET_ARC_SLACK_DEG over. The angle a corner is met at is
/// taken either way round, so a cusp — a piece run back along its neighbour — counts as
/// tangent, which it is to the relation that holds it.
fn corner_slack(sketch: &Sketch, lp: &StationLoop, i: usize, keep: &HashSet<EntityId>) -> f64 {
    let n = lp.pieces.len();
    let at = |k: usize, from_end: bool| -> Option<DVec2> {
        let p = lp.pieces[k];
        travel_at(&shape_of(sketch, p.id)?, p.from, if from_end { p.from } else { p.to })
    };
    let corner = |k: usize, other: usize, from_end: bool| -> f64 {
        if keep.contains(&lp.pieces[other].id) {
            return PI / 2.0;
        }
        match (at(k, from_end), at(other, !from_end)) {
            (Some(t), Some(u)) => 2.0 * t.perp_dot(u).abs().min(1.0).asin(),
            _ => PI / 2.0,
        }
    };
    corner(i, (i + n - 1) % n, true) + corner(i, (i + 1) % n, false) + INSET_ARC_SLACK_DEG * PI / 180.0
}

/// Where a station folds, in words, or `None`: a corner the profile turns back at (more than a
/// right angle — a piece run backwards along its neighbour), or a piece under
/// STATION_MIN_PIECE_MM.
fn source_folds(sketch: &Sketch, lp: &StationLoop) -> Option<String> {
    let n = lp.pieces.len();
    let piece = |k: usize| shape_of(sketch, lp.pieces[k].id);
    let short = (0..n).find(|&k| piece(k).is_some_and(|s| piece_length(&s) < STATION_MIN_PIECE_MM));
    let mut worst: Option<(EntityId, EntityId, f64)> = None;
    let count = if lp.closed { n } else { n.saturating_sub(1) };
    for k in 0..count {
        let (a, b) = (lp.pieces[k], lp.pieces[(k + 1) % n]);
        let (Some(t), Some(u)) = (piece(k).and_then(|s| travel_at(&s, a.from, a.to)), piece((k + 1) % n).and_then(|s| travel_at(&s, b.from, b.from))) else { continue };
        let turn = t.dot(u).clamp(-1.0, 1.0).acos();
        if turn > PI / 2.0 && worst.is_none_or(|w| turn > w.2) {
            worst = Some((a.id, b.id, turn));
        }
    }
    let mut words: Vec<String> = Vec::new();
    if let Some((a, b, turn)) = worst {
        words.push(format!("the source folds back at the corner of pieces {a} and {b} (it turns {}° there)", deg_of(turn)));
    }
    if let Some(k) = short {
        words.push(format!("its piece {} is {} mm long", lp.pieces[k].id, mm(piece(k).map_or(0.0, |s| piece_length(&s)))));
    }
    if words.is_empty() {
        return None;
    }
    let said = words.join("; ");
    let mut chars = said.chars();
    Some(match chars.next() {
        Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
        None => said,
    })
}

/// The pairs of a profile's pieces that cross each other — not at the corner two neighbours
/// share — read on their polylines.
fn crossing_pairs(sketch: &Sketch, lp: &StationLoop) -> Vec<(EntityId, EntityId)> {
    let n = lp.pieces.len();
    let shapes: Vec<Option<Shape>> = lp.pieces.iter().map(|p| shape_of(sketch, p.id)).collect();
    let lines: Vec<Vec<DVec2>> = shapes.iter().map(|s| s.as_ref().map_or(vec![], |s| polyline(s, 48))).collect();
    let ends: Vec<Vec<DVec2>> = shapes.iter().map(|s| s.as_ref().map_or(vec![], |s| vec![anchor_point(s, AnchorAt::A), anchor_point(s, AnchorAt::B)])).collect();
    let mut out = Vec::new();
    for i in 0..n {
        for j in i + 1..n {
            let beside = j == i + 1 || (lp.closed && i == 0 && j == n - 1);
            let at_corner = |p: DVec2| beside && ends[i].iter().chain(ends[j].iter()).any(|q| (p - *q).length() <= 10.0 * TOUCH_MM);
            let mut found = false;
            'outer: for s in 0..lines[i].len().saturating_sub(1) {
                let (p0, p1) = (lines[i][s], lines[i][s + 1]);
                let da = p1 - p0;
                for t in 0..lines[j].len().saturating_sub(1) {
                    let (q0, q1) = (lines[j][t], lines[j][t + 1]);
                    let db = q1 - q0;
                    let denom = da.perp_dot(db);
                    if denom.abs() < 1e-18 {
                        continue;
                    }
                    let w = q0 - p0;
                    let x = w.perp_dot(db) / denom;
                    let y = w.perp_dot(da) / denom;
                    if !(0.0..=1.0).contains(&x) || !(0.0..=1.0).contains(&y) {
                        continue;
                    }
                    if !at_corner(p0 + da * x) {
                        found = true;
                        break 'outer;
                    }
                }
            }
            if found {
                out.push((lp.pieces[i].id, lp.pieces[j].id));
            }
        }
    }
    out
}

/// A relation the move let go: its id and its kind, as ScanRuler names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Dropped {
    pub id: u32,
    pub kind: &'static str,
}

/// A profile moved in: the sketch, the pieces moved, and the relations the move broke, which
/// went.
#[derive(Debug, Clone, PartialEq)]
pub struct Inset {
    pub sketch: Sketch,
    pub moved: Vec<EntityId>,
    pub dropped: Vec<Dropped>,
}

/// The kind of a relation as ScanRuler's sketch names it.
pub fn rule_kind(rule: &Rule) -> &'static str {
    match rule {
        Rule::Coincident { .. } => "coincident",
        Rule::Horizontal { .. } => "horizontal",
        Rule::Vertical { .. } => "vertical",
        Rule::Parallel { .. } => "parallel",
        Rule::Perpendicular { .. } => "perpendicular",
        Rule::Tangent { .. } => "tangent",
        Rule::Concentric { .. } => "concentric",
        Rule::Equal { .. } => "equal",
        Rule::PointOn { .. } => "pointOn",
        Rule::Collinear { .. } => "collinear",
        Rule::Symmetric { .. } => "symmetric",
        Rule::Length { .. } => "length",
        Rule::Distance { .. } => "distance",
        Rule::Radius { .. } => "radius",
        Rule::Diameter { .. } => "diameter",
        Rule::Angle { .. } => "angle",
    }
}

/// The pieces a profile keeps where they lie when it is moved in and none are named: the drawn
/// ones, where it has fitted ones too — the line it was closed with across the base — and none
/// where all or none are fitted. An arc held tangent both ways (`sketch` says) is a round drawn
/// on the outline, reverse.sketch.edit's add.rounds, and moves with it.
pub fn inset_keeps(lp: &StationLoop, sketch: Option<&Sketch>) -> Vec<EntityId> {
    let fitted = lp.pieces.iter().filter(|p| p.fitted).count();
    if !(fitted > 0 && fitted < lp.pieces.len()) {
        return vec![];
    }
    let round = |id: EntityId| sketch.map_or(0, |s| s.constraints.iter().filter(|c| matches!(c.rule, Rule::Tangent { a, b } if a == id || b == id)).count()) >= 2;
    lp.pieces.iter().filter(|p| !p.fitted && !(p.kind == ShapeKind::Arc && round(p.id))).map(|p| p.id).collect()
}

/// A closed profile moved inward by `by` mm — outward for a negative `by` — as a thin shell's
/// inner station is its outer one moved in by the wall: each line along its normal into the
/// profile, each arc about its centre, `by` off its radius where the profile lies on its
/// centre's side and on where it lies off it; the corners met again where the moved pieces
/// cross. Pieces in `keep` stay on their lines, their ends sliding along them to meet the moved
/// ones. The moved pieces let their slice points go: the scan's points are the other skin's.
/// Dimensions take what they now measure (their expressions let go); a relation the move breaks
/// — a moved end held on a coordinate plane, a moved line along a kept one — goes, and
/// `dropped` lists it. Why not, in words: the profile is open, an arc would close up, a piece
/// would turn inside out — an arc's sweep changing more than its corners let it among them — or
/// the result would close more than one region; the last two name the fold in the source where
/// there is one, which is most often why.
pub fn inset_sketch(sketch: &Sketch, lp: &StationLoop, by: f64, keep: &HashSet<EntityId>, datums: &[DatumTrace]) -> Result<Inset, FitError> {
    let (mut out, moved) = moved_inward(sketch, lp, by, keep)?;
    let dropped = relations_settled(&mut out, &moved, datums);
    Ok(Inset { sketch: out, moved, dropped })
}

/// `inset_sketch`'s move: one profile's pieces moved and the corners met again, the relations
/// left as they were.
fn moved_inward(sketch: &Sketch, lp: &StationLoop, by: f64, keep: &HashSet<EntityId>) -> Result<(Sketch, Vec<EntityId>), FitError> {
    if !lp.closed {
        return Err(FitError::new("the profile is open — an inset moves a closed profile inward; close it with a line across its ends first."));
    }
    let mut out = sketch.clone();
    // Which way the profile runs: counter-clockwise, its inside on the left.
    let outline: Vec<DVec2> = lp
        .pieces
        .iter()
        .flat_map(|p| {
            let mut pts = shape_of(sketch, p.id).map_or(vec![], |s| polyline(&s, 48));
            if p.from != AnchorAt::A {
                pts.reverse();
            }
            pts
        })
        .collect();
    let left = crate::taper::signed_area(&outline) > 0.0;
    let inward = |travel: DVec2| if left { travel.perp() } else { -travel.perp() };
    let mut moved: Vec<EntityId> = Vec::new();
    for p in &lp.pieces {
        if keep.contains(&p.id) {
            continue;
        }
        let Some(e) = out.entity_mut(p.id) else { continue };
        match &mut e.shape {
            Shape::Line { a, b } => {
                let Some(d) = normalize2(if p.to == AnchorAt::B { *b - *a } else { *a - *b }) else { return Err(FitError::new(format!("piece {} has no length.", p.id))) };
                let shift = inward(d) * by;
                *a += shift;
                *b += shift;
            }
            Shape::Arc { c, r, start, sweep } => {
                let mid = *start + *sweep / 2.0;
                let m = arc_point(*c, *r, mid);
                let ccw = DVec2::new(-mid.sin(), mid.cos());
                let travel = if p.from == AnchorAt::A { ccw } else { -ccw };
                let nr = if inward(travel).dot(*c - m) > 0.0 { *r - by } else { *r + by };
                if !(nr > 1e-6) {
                    return Err(FitError::new(format!(
                        "the arc {} (R{}) is tighter than {} mm — it would close up. Make the wall thinner, or round that corner wider first.",
                        p.id,
                        num(js_round(*r * 1000.0) / 1000.0),
                        num(by.abs())
                    )));
                }
                *r = nr;
            }
            _ => {}
        }
        e.points.clear();
        e.scatter = 0.0;
        moved.push(p.id);
    }
    let ends: Ends = lp.pieces.iter().map(|p| (p.id, (p.from, p.to))).collect();
    // Each corner met where the moved pieces cross nearest where it stood.
    let stood = |i: usize, _pa: DVec2, _pb: DVec2| shape_of(sketch, lp.pieces[i].id).map_or(DVec2::ZERO, |s| anchor_point(&s, lp.pieces[i].to));
    meet_corners(&mut out, lp, &ends, &|id| keep.contains(&id), &stood);
    // A piece that came out the other way round has turned inside out. An arc keeps its sweep
    // where its neighbours run tangent into it, and changes by little more than the corners it
    // meets them at; past that, its neighbours crossed past it — most often because the source
    // folds.
    let fold = source_folds(sketch, lp);
    let why = match &fold {
        Some(f) => format!(" {f} — fit that station again (fit.like refuses a fold) before insetting it."),
        None => " Give a smaller inset.".to_string(),
    };
    for (i, p) in lp.pieces.iter().enumerate() {
        match (shape_of(sketch, p.id), shape_of(&out, p.id)) {
            (Some(Shape::Line { a: wa, b: wb }), Some(Shape::Line { a: na, b: nb })) => {
                let was = wb - wa;
                let now = nb - na;
                if !(was.dot(now) > 0.0) || now.length() < 1e-6 {
                    return Err(FitError::new(format!(
                        "piece {} would turn inside out at {} mm — the profile is narrower there than the inset takes; give it a smaller one, or keep that piece (keep).{}",
                        p.id,
                        num(by),
                        fold.as_ref().map_or(String::new(), |f| format!(" {f}."))
                    )));
                }
            }
            (Some(Shape::Arc { sweep: was, .. }), Some(Shape::Arc { sweep: now, .. })) => {
                let allowed = PI.min(corner_slack(sketch, lp, i, keep));
                if (now - was).abs() > allowed || now < 1e-6 {
                    return Err(FitError::new(format!(
                        "the arc {} would turn inside out at {} mm — its sweep goes from {}° to {}°, more than the corners it meets its neighbours at let it ({}°): they cross past it.{why}",
                        p.id,
                        num(by),
                        deg_of(was),
                        deg_of(now),
                        deg_of(allowed)
                    )));
                }
            }
            _ => {}
        }
    }
    // One closed profile in, one region out: a result that crosses itself lofts a fragment of
    // itself.
    let ids: HashSet<EntityId> = lp.pieces.iter().map(|p| p.id).collect();
    let alone = Sketch { entities: out.entities.iter().filter(|e| ids.contains(&e.id)).cloned().collect(), constraints: vec![], next_id: out.next_id };
    let regions = find_regions(&alone);
    if regions.len() > 1 {
        let pairs = crossing_pairs(&out, lp);
        return Err(FitError::new(format!(
            "the inset at {} mm crosses itself: it closes {} regions (areas {} mm²){}.{why}",
            num(by),
            regions.len(),
            regions.iter().map(|r| mm(r.area)).collect::<Vec<_>>().join(", "),
            if pairs.is_empty() { String::new() } else { format!(", pieces {} crossing", pairs.iter().map(|(x, y)| format!("{x} and {y}")).collect::<Vec<_>>().join(", ")) }
        )));
    }
    Ok((out, moved))
}

/// After a move: the dimensions on the pieces moved read again, the relations it broke let go,
/// and which went.
fn relations_settled(out: &mut Sketch, moved: &[EntityId], datums: &[DatumTrace]) -> Vec<Dropped> {
    let mut dropped = Vec::new();
    let moved: HashSet<EntityId> = moved.iter().copied().collect();
    let constraints = std::mem::take(&mut out.constraints);
    let mut kept = Vec::with_capacity(constraints.len());
    for mut c in constraints {
        if !constraint_entities(&c.rule).iter().any(|id| moved.contains(id)) {
            kept.push(c);
            continue;
        }
        if c.rule.is_dimension() {
            match measure_rule(out, &c.rule, datums) {
                Some(value) => {
                    c.rule.set_value(value);
                    c.expr = None;
                    kept.push(c);
                }
                None => dropped.push(Dropped { id: c.id, kind: rule_kind(&c.rule) }),
            }
            continue;
        }
        if holds(out, &c.rule, datums) {
            kept.push(c);
        } else {
            dropped.push(Dropped { id: c.id, kind: rule_kind(&c.rule) });
        }
    }
    out.constraints = kept;
    dropped
}

/// A sketch's drawn pieces by the chains they make: those whose ends meet (within TOUCH_MM) are
/// one, a circle one of its own.
fn chains_of(sketch: &Sketch) -> Vec<Vec<Entity>> {
    let own: Vec<&Entity> = sketch.entities.iter().filter(|e| !e.construction && e.kind() != ShapeKind::Point).collect();
    let mut parent: Vec<usize> = (0..own.len()).collect();
    fn root(parent: &mut [usize], i: usize) -> usize {
        if parent[i] == i {
            return i;
        }
        let r = root(parent, parent[i]);
        parent[i] = r;
        r
    }
    let ends: Vec<Vec<DVec2>> = own.iter().map(|e| ends_of(&e.shape).iter().map(|at| anchor_point(&e.shape, *at)).collect()).collect();
    for i in 0..own.len() {
        for j in i + 1..own.len() {
            if ends[i].iter().any(|p| ends[j].iter().any(|q| (*p - *q).length() <= TOUCH_MM)) {
                let (ri, rj) = (root(&mut parent, i), root(&mut parent, j));
                parent[ri] = rj;
            }
        }
    }
    let mut out: Vec<(usize, Vec<Entity>)> = Vec::new();
    for (i, e) in own.iter().enumerate() {
        let r = root(&mut parent, i);
        match out.iter_mut().find(|(k, _)| *k == r) {
            Some((_, list)) => list.push((*e).clone()),
            None => out.push((r, vec![(*e).clone()])),
        }
    }
    out.into_iter().map(|(_, l)| l).collect()
}

/// Whether a point lies inside a closed outline.
fn inside_polygon(poly: &[DVec2], p: DVec2) -> bool {
    let mut inn = false;
    let n = poly.len();
    let mut j = n.wrapping_sub(1);
    for i in 0..n {
        let (a, b) = (poly[i], poly[j]);
        if (a.y > p.y) != (b.y > p.y) && p.x < ((b.x - a.x) * (p.y - a.y)) / (b.y - a.y) + a.x {
            inn = !inn;
        }
        j = i;
    }
    inn
}

/// Whether two chains' outlines cross, read on their polylines.
fn outlines_cross(a: &[Vec<DVec2>], b: &[Vec<DVec2>]) -> bool {
    for la in a {
        for lb in b {
            for s in 0..la.len().saturating_sub(1) {
                let da = la[s + 1] - la[s];
                for t in 0..lb.len().saturating_sub(1) {
                    let db = lb[t + 1] - lb[t];
                    let denom = da.perp_dot(db);
                    if denom.abs() < 1e-18 {
                        continue;
                    }
                    let w = lb[t] - la[s];
                    let x = w.perp_dot(db) / denom;
                    let y = w.perp_dot(da) / denom;
                    if (0.0..=1.0).contains(&x) && (0.0..=1.0).contains(&y) {
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// The slice points a piece had, moved as the piece was: across a line's shift, along their
/// radii by a round's change of radius.
fn points_moved(was: &Entity, now: &Entity) -> Option<Vec<DVec2>> {
    if was.points.is_empty() {
        return None;
    }
    match (was.shape.clone(), now.shape.clone()) {
        (Shape::Line { a: wa, .. }, Shape::Line { a, b }) => {
            let d = normalize2(b - a)?;
            let n = d.perp();
            let by = n * n.dot(a - wa);
            Some(was.points.iter().map(|p| *p + by).collect())
        }
        (Shape::Arc { c: wc, r: wr, .. }, Shape::Arc { c, r, .. }) | (Shape::Circle { c: wc, r: wr }, Shape::Circle { c, r }) => Some(
            was.points
                .iter()
                .map(|p| {
                    let out = *p - wc;
                    let rr = out.length();
                    if rr > 1e-9 {
                        c + out * ((rr + r - wr) / rr)
                    } else {
                        *p
                    }
                })
                .collect(),
        ),
        _ => None,
    }
}

/// A chain as an inset answers with it: its pieces, how far it moved inward (outward where
/// negative), and those of them held where they lie.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChainInset {
    pub pieces: Vec<EntityId>,
    pub by: f64,
    pub kept: Vec<EntityId>,
}

/// A sketch's profiles moved inward: the sketch, what moved, the relations that went, each
/// chain's move, the ids held where they lie, and the chains left where they lie because they
/// are open.
#[derive(Debug, Clone, PartialEq)]
pub struct ChainsInset {
    pub sketch: Sketch,
    pub moved: Vec<EntityId>,
    pub dropped: Vec<Dropped>,
    pub chains: Vec<ChainInset>,
    pub kept: Vec<EntityId>,
    pub open: Vec<Vec<EntityId>>,
}

/// How far each chain moves: one distance for all, or by how many closed chains lie round it.
#[derive(Clone, Copy)]
pub enum InsetBy<'a> {
    Fixed(f64),
    ByDepth(&'a dyn Fn(usize) -> f64),
}

/// What `inset_chains` is asked: how far each chain moves, the pieces held where they lie (by
/// default each chain's `inset_keeps`), the chains to move (those with a piece in `only`; by
/// default every one), and the slice points kept, moved with their pieces, in place of let go.
#[derive(Clone, Copy)]
pub struct InsetChainsAsk<'a> {
    pub by: InsetBy<'a>,
    pub keep: Option<&'a dyn Fn(&StationLoop) -> HashSet<EntityId>>,
    pub only: Option<&'a HashSet<EntityId>>,
    pub keep_points: bool,
    pub name: &'a str,
    pub datums: &'a [DatumTrace],
}

impl<'a> InsetChainsAsk<'a> {
    pub fn new(by: f64, name: &'a str) -> Self {
        Self { by: InsetBy::Fixed(by), keep: None, only: None, keep_points: false, name, datums: &[] }
    }
}

/// A chain read for the inset: its pieces, and its loop unless it is a circle.
struct Read {
    pieces: Vec<Entity>,
    lp: Option<StationLoop>,
}

/// Every closed chain of a sketch moved inward on its own (`inset_sketch`; a circle by its
/// radius): each into its own inside, so a chain inside another moves away from it. Open
/// chains, among closed ones, stay where they lie and are named. All or nothing: a chain that
/// closes up or turns inside out, or two that come to cross, refuse the whole of it, by chain.
/// One chain alone refuses in `inset_sketch`'s own words.
pub fn inset_chains(sketch: &Sketch, ask: &InsetChainsAsk) -> Result<ChainsInset, FitError> {
    let all = chains_of(sketch);
    if all.is_empty() {
        return Err(FitError::new(format!("{} has no pieces.", ask.name)));
    }
    let chains: Vec<Vec<Entity>> = match ask.only {
        Some(only) => all.into_iter().filter(|c| c.iter().any(|e| only.contains(&e.id))).collect(),
        None => all,
    };
    let named = |c: &[Entity]| format!("pieces {}", c.iter().map(|e| e.id.to_string()).collect::<Vec<_>>().join(", "));
    let mut read: Vec<Read> = Vec::new();
    let mut refused: Vec<String> = Vec::new();
    let mut open: Vec<Vec<EntityId>> = Vec::new();
    for pieces in &chains {
        if pieces.len() == 1 && pieces[0].kind() == ShapeKind::Circle {
            read.push(Read { pieces: pieces.clone(), lp: None });
            continue;
        }
        match station_loop(pieces, ask.name, true) {
            Err(e) => refused.push(if chains.len() == 1 { e.0 } else { format!("{}: {}", named(pieces), e.0) }),
            Ok(lp) if lp.closed => read.push(Read { pieces: pieces.clone(), lp: Some(lp) }),
            Ok(lp) => {
                if chains.len() == 1 {
                    return match moved_inward(sketch, &lp, 0.0, &HashSet::new()) {
                        Err(e) => Err(e),
                        Ok(_) => Err(FitError::new("the profile is open — an inset moves a closed profile inward; close it with a line across its ends first.")),
                    };
                }
                open.push(pieces.iter().map(|e| e.id).collect());
            }
        }
    }
    let none_move = |words: &[String], of: usize| {
        if of == 1 {
            FitError::new(words[0].clone())
        } else {
            FitError::new(format!("{} of the {of} chains do not move: {} — nothing was moved.", words.len(), words.join("; ")))
        }
    };
    if !refused.is_empty() {
        return Err(none_move(&refused, chains.len()));
    }
    if read.is_empty() {
        return Err(FitError::new("every chain is open — an inset moves a closed profile inward; close them with a line across their ends first."));
    }
    // Each chain's outline before the move: how deep it lies among the others.
    let outline_of = |s: &Sketch, r: &Read| -> Vec<Vec<DVec2>> { r.pieces.iter().map(|e| shape_of(s, e.id).map_or(vec![], |sh| polyline(&sh, 48))).collect() };
    let before: Vec<Vec<Vec<DVec2>>> = read.iter().map(|r| outline_of(sketch, r)).collect();
    let depth_of = |i: usize| -> usize {
        let Some(&at) = before[i].first().and_then(|l| l.first()) else { return 0 };
        (0..read.len()).filter(|&k| k != i && inside_polygon(&joined_outline(&read[k], &before[k]), at)).count()
    };
    let mut out = sketch.clone();
    let mut moved: Vec<EntityId> = Vec::new();
    let mut chains_out: Vec<ChainInset> = Vec::new();
    let mut kept: Vec<EntityId> = Vec::new();
    for (i, r) in read.iter().enumerate() {
        let by = match ask.by {
            InsetBy::Fixed(b) => b,
            InsetBy::ByDepth(f) => f(depth_of(i)),
        };
        let mut say = |why: String| refused.push(if read.len() == 1 { why } else { format!("{}: {why}", named(&r.pieces)) });
        let Some(lp) = &r.lp else {
            let was = &r.pieces[0];
            let Shape::Circle { r: wr, .. } = was.shape else { continue };
            if !(wr - by > 1e-6) {
                say(format!("the circle {} (R{}) is tighter than {} mm — it would close up. Give a smaller inset.", was.id, mm(wr), num(by.abs())));
                continue;
            }
            if let Some(e) = out.entity_mut(was.id) {
                if let Shape::Circle { r, .. } = &mut e.shape {
                    *r = wr - by;
                }
                e.points.clear();
                e.scatter = 0.0;
            }
            moved.push(was.id);
            chains_out.push(ChainInset { pieces: vec![was.id], by, kept: vec![] });
            continue;
        };
        let keep: HashSet<EntityId> = match ask.keep {
            Some(f) => f(lp),
            None => inset_keeps(lp, Some(sketch)).into_iter().collect(),
        };
        match moved_inward(&out, lp, by, &keep) {
            Err(e) => say(e.0),
            Ok((made, made_moved)) => {
                out = made;
                moved.extend(made_moved);
                let held: Vec<EntityId> = lp.pieces.iter().filter(|p| keep.contains(&p.id)).map(|p| p.id).collect();
                kept.extend(held.iter().copied());
                chains_out.push(ChainInset { pieces: r.pieces.iter().map(|e| e.id).collect(), by, kept: held });
            }
        }
    }
    if !refused.is_empty() {
        return Err(none_move(&refused, read.len()));
    }
    let after: Vec<Vec<Vec<DVec2>>> = read.iter().map(|r| outline_of(&out, r)).collect();
    for i in 0..read.len() {
        for j in i + 1..read.len() {
            if outlines_cross(&after[i], &after[j]) && !outlines_cross(&before[i], &before[j]) {
                return Err(FitError::new(format!(
                    "moved, the chains of {} and of {} would cross — give a smaller inset, or inset them apart. Nothing was moved.",
                    named(&read[i].pieces),
                    named(&read[j].pieces)
                )));
            }
        }
    }
    if ask.keep_points {
        for id in &moved {
            let (Some(was), Some(now)) = (sketch.entity(*id), out.entity(*id)) else { continue };
            if let Some(points) = points_moved(was, now) {
                let scatter = was.scatter;
                if let Some(e) = out.entity_mut(*id) {
                    e.points = points;
                    e.scatter = scatter;
                }
            }
        }
    }
    let dropped = relations_settled(&mut out, &moved, ask.datums);
    Ok(ChainsInset { sketch: out, moved, dropped, chains: chains_out, kept, open })
}

/// A chain's polylines as one closed outline, in its order round.
fn joined_outline(r: &Read, lines: &[Vec<DVec2>]) -> Vec<DVec2> {
    let Some(lp) = &r.lp else { return lines.concat() };
    lp.pieces
        .iter()
        .flat_map(|p| {
            let Some(k) = r.pieces.iter().position(|e| e.id == p.id) else { return vec![] };
            let mut l = lines[k].clone();
            if p.from != AnchorAt::A {
                l.reverse();
            }
            l
        })
        .collect()
}

#[cfg(test)]
mod tests;
