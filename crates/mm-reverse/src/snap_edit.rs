// SPDX-License-Identifier: AGPL-3.0-only
//! Where a moved face's move goes. A face snapped onto the scan can be moved in two ways: by
//! changing what made it — the distance of the extrude whose cap it is, the sketch line whose
//! wall it is, the sketch circle whose bore or boss it is — or by working on the body as it
//! stands (the Move face feature). The first is what a CAD operator would do by hand and is
//! what is wanted wherever it can be had: the rounds, the drafts and the shells that come after
//! the extrude are simply built again on the moved wall, where the second knows only the face's
//! own outline and leaves a filleted face's rounds behind as a pad or a pocket.
//!
//! A face names the feature that made it (its fingerprint's `origin`, carried through every
//! boolean by the builder), so the trace is: that feature, if it is an extrude or a revolve; its
//! sketch; and in the sketch the piece whose lifted carrier the face lies on — for a revolve the
//! line of its section square to the axis under a shoulder, the line along the axis at a bore's
//! or a rim's radius. The move of a sketch piece goes through the solver, so the corners it
//! shares follow and a dimension that holds the piece where it is says so instead of being
//! overruled.
//!
//! Plain data in, plain data out: the feature list with the edit made, and a sentence for the
//! box. Nothing here touches the kernel.
//!
//! Ported from ScanRuler (`plugins/reverse/core/snapEdit.ts`), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only. One difference: a Polymix region key is the set of its outer pieces alone,
//! with no centroid, so a region whose pieces changed is found again by the pieces it shares
//! (the last of ScanRuler's `matchRegion` steps), never by where its middle was.

use glam::{DVec2, DVec3};

use crate::build::resolve_axis;
use crate::datums::datum_traces;
use crate::display_units::{length_decimals, to_display, unit_word, DisplayUnits};
use crate::drag::{detached, translate_entity};
use crate::model::{Element, ExtrudeExtent, ExtrudeFeature, Feature, FeatureId, FeatureKind, Part, RevolveFeature, SketchFeature};
use crate::regions::{find_regions, Region};
use crate::round::{number_text, round3, round4, round_to};
use crate::sketch::{Anchor, AnchorAt, DistanceTo, Entity, EntityId, Rule, Shape, Sketch};
use crate::solver::{solve_sketch, SolveOptions};
use crate::topo::{FaceFingerprint, FaceKind};

/// How a face moves.
#[derive(Debug, Clone, PartialEq)]
pub enum SnapPlan {
    /// An edit of an extrude's extent: the features with it made, and what it is, for the box
    /// and the status line.
    Extent { features: Vec<Feature>, text: String },
    /// An edit of a sketch's line or circle.
    Sketch { features: Vec<Feature>, text: String },
    /// No such edit: the face is moved on the body as it stands. `why` says what stood in the
    /// way.
    Direct { why: String },
}

impl SnapPlan {
    /// Whether the plan edits the history (an extent or a sketch).
    pub fn edits(&self) -> bool {
        !matches!(self, SnapPlan::Direct { .. })
    }

    /// The features with the edit made, for an edit.
    pub fn features(&self) -> Option<&[Feature]> {
        match self {
            SnapPlan::Extent { features, .. } | SnapPlan::Sketch { features, .. } => Some(features),
            SnapPlan::Direct { .. } => None,
        }
    }

    /// What the edit is, in a sentence's words, for an edit.
    pub fn text(&self) -> Option<&str> {
        match self {
            SnapPlan::Extent { text, .. } | SnapPlan::Sketch { text, .. } => Some(text),
            SnapPlan::Direct { .. } => None,
        }
    }

    /// Why the move is made on the body, for a direct one.
    pub fn why(&self) -> Option<&str> {
        match self {
            SnapPlan::Direct { why } => Some(why),
            _ => None,
        }
    }
}

fn direct(why: impl Into<String>) -> SnapPlan {
    SnapPlan::Direct { why: why.into() }
}

/// How near a face must lie to the cap or the carrier it is traced to, millimetres — past the
/// kernel's own tolerance, short of any feature.
const ON_MM: f64 = 2e-3;
/// How far beyond the target the helper line of a moved wall is laid: the solver's
/// point-to-line distance is |d| − value, smooth away from zero.
const HELPER_MM: f64 = 10.0;

/// A length of the box's sentence, in the units shown: three decimals in millimetres, as many
/// more in inches as `length_decimals` gives, and no zeros carried behind. The unit's word
/// follows it in the sentence.
fn len_text(v: f64, units: DisplayUnits) -> String {
    match units {
        DisplayUnits::Mm => number_text(round3(v)),
        _ => number_text(round_to(to_display(v, units), length_decimals(3, units) as i32)),
    }
}

/// The region a key names now: the same key; else the region sharing most of its pieces, if
/// at least half (ScanRuler's `matchRegion`; a Polymix key carries no middle to look under).
fn match_region<'a>(key: &str, regions: &'a [Region]) -> Option<&'a Region> {
    if let Some(r) = regions.iter().find(|r| r.key == key) {
        return Some(r);
    }
    let was: Vec<EntityId> = key.split(',').filter_map(|s| s.trim().parse().ok()).collect();
    if was.is_empty() {
        return None;
    }
    let mut best: Option<(&Region, f64)> = None;
    for r in regions {
        let mut ids = r.outer.clone();
        ids.sort_unstable();
        ids.dedup();
        let n = ids.iter().filter(|i| was.contains(i)).count() as f64;
        let score = n / (ids.len() as f64 + was.len() as f64 - n);
        if score >= 0.5 && best.is_none_or(|(_, s)| score > s) {
            best = Some((r, score));
        }
    }
    best.map(|b| b.0)
}

/// The regions a list of keys names now, each once, in the keys' order.
fn match_regions<'a>(keys: &[String], regions: &'a [Region]) -> Vec<&'a Region> {
    let mut out: Vec<&Region> = Vec::new();
    for k in keys {
        if let Some(r) = match_region(k, regions) {
            if !out.iter().any(|o| o.key == r.key) {
                out.push(r);
            }
        }
    }
    out
}

/// The sketch and the regions a feature builds from it, for an extrude or a revolve.
fn built_on(f: &Feature) -> Option<(FeatureId, &[String])> {
    match &f.kind {
        FeatureKind::Extrude(x) => Some((x.sketch_id, &x.regions)),
        FeatureKind::Revolve(x) => Some((x.sketch_id, &x.regions)),
        _ => None,
    }
}

/// The features with one sketch replaced, the region picks of what is built on it matched onto
/// the sketch as it now is — as a sketch edited by hand has them.
fn with_sketch(features: &[Feature], sketch_id: FeatureId, sketch: &Sketch) -> Vec<Feature> {
    let regions = find_regions(sketch);
    let rekey = |keys: &[String]| -> Vec<String> {
        let kept: Vec<String> = match_regions(keys, &regions).into_iter().map(|r| r.key.clone()).collect();
        if kept.is_empty() {
            keys.to_vec()
        } else {
            kept
        }
    };
    features
        .iter()
        .map(|f| {
            let mut f = f.clone();
            match &mut f.kind {
                FeatureKind::Sketch(s) if f.id == sketch_id => s.sketch = sketch.clone(),
                FeatureKind::Extrude(x) if x.sketch_id == sketch_id => x.regions = rekey(&x.regions),
                FeatureKind::Revolve(x) if x.sketch_id == sketch_id => x.regions = rekey(&x.regions),
                _ => {}
            }
            f
        })
        .collect()
}

/// An edit of a sketch as a plan — or none, where it changes how the sketch's profiles lie. A
/// line run into a profile beside it makes the overlap a region of its own, which no extrude
/// picked: the body would come out with a hole where the two meet, where a Move face on the
/// body leaves the two bodies overlapping as the operator dragged them.
fn sketch_plan(features: &[Feature], sketch: (&Feature, &SketchFeature), next: Sketch, text: String) -> SnapPlan {
    let (sf, sk) = sketch;
    let was = find_regions(&sk.sketch);
    let now = find_regions(&next);
    let moved = features.iter().filter_map(built_on).filter(|(s, _)| *s == sf.id).any(|(_, keys)| match_regions(keys, &now).len() != match_regions(keys, &was).len());
    if now.len() != was.len() || moved {
        return direct(format!("the move would run a profile of {} into another, or open one", sf.name));
    }
    SnapPlan::Sketch { features: with_sketch(features, sf.id, &next), text }
}

/// The sentence for constraints the solver could not meet after a move.
const HELD: &str = "a dimension or a relation of the sketch holds it where it is";

fn move_cap(features: &[Feature], maker: (&Feature, &ExtrudeFeature), sketch: &SketchFeature, face: &FaceFingerprint, offset: f64, units: DisplayUnits) -> SnapPlan {
    let (mf, x) = maker;
    let n = sketch.frame.normal;
    let along = if face.dir.dot(n) > 0.0 { 1.0 } else { -1.0 };
    let at = (face.centroid - sketch.frame.origin).dot(n);
    let (ends, symmetric) = match x.extent {
        ExtrudeExtent::Distance { from, to } => ((from, to), false),
        ExtrudeExtent::Symmetric { total } => ((-total / 2.0, total / 2.0), true),
        _ => return direct(format!("{} reaches to a plane, not by a distance", mf.name)),
    };
    let to_side = if (at - ends.1).abs() <= ON_MM {
        true
    } else if (at - ends.0).abs() <= ON_MM {
        false
    } else {
        return direct(format!("the face is not an end of {} any more", mf.name));
    };
    let before = if to_side { ends.1 } else { ends.0 };
    let after = round4(before + along * offset);
    let next = if to_side { (ends.0, after) } else { (after, ends.1) };
    if (next.1 - next.0).abs() < 1e-6 {
        return direct(format!("{} would have no length left", mf.name));
    }
    let edited = features
        .iter()
        .map(|f| {
            let mut f = f.clone();
            if f.id == mf.id {
                if let FeatureKind::Extrude(e) = &mut f.kind {
                    e.extent = ExtrudeExtent::Distance { from: next.0, to: next.1 };
                }
            }
            f
        })
        .collect();
    SnapPlan::Extent {
        features: edited,
        text: format!("{}: {} {} → {} {}{}", mf.name, if to_side { "To" } else { "From" }, len_text(before, units), len_text(after, units), unit_word(units, 1), if symmetric { " (no longer symmetric)" } else { "" }),
    }
}

/// A sheet point of a frame.
fn to2(frame: &crate::frame::SectionFrame, p: DVec3) -> DVec2 {
    DVec2::new((p - frame.origin).dot(frame.u), (p - frame.origin).dot(frame.v))
}

fn move_wall(features: &[Feature], maker: &Feature, sketch: (&Feature, &SketchFeature), face: &FaceFingerprint, offset: f64, units: DisplayUnits) -> SnapPlan {
    if let FeatureKind::Extrude(x) = &maker.kind {
        if x.draft_deg != 0.0 {
            return direct(format!("{} has a taper — its walls do not stand on the sketch's lines", maker.name));
        }
    }
    let (sf, sk) = sketch;
    let frame = &sk.frame;
    let off = |p: DVec3| (p - face.centroid).dot(face.dir).abs();
    // The line whose carrier the face lies on; of several in one carrier — two steps of a
    // profile in line — the one the face's middle is over.
    let middle = to2(frame, face.centroid);
    let mut line: Option<&Entity> = None;
    let mut nearest = f64::INFINITY;
    for e in &sk.sketch.entities {
        let Shape::Line { a, b } = e.shape else { continue };
        if e.construction || off(frame.to_world(a)) > ON_MM || off(frame.to_world(b)) > ON_MM {
            continue;
        }
        let d = ((a + b) / 2.0 - middle).length();
        if d < nearest {
            nearest = d;
            line = Some(e);
        }
    }
    let Some(line) = line else { return direct(format!("no line of {} lies under this face", sf.name)) };
    // Square to the line, the way the face's normal reads on the sheet.
    let dir2 = DVec2::new(face.dir.dot(frame.u), face.dir.dot(frame.v));
    let l = dir2.length();
    if !(l > 1e-9) {
        return direct("the face does not stand on the sketch plane");
    }
    match move_line(features, sketch, line, dir2 / l, offset, units) {
        SnapPlan::Sketch { features, .. } => SnapPlan::Sketch { features, text: format!("{}: the line under this face moves {} {} {}", sf.name, len_text(offset.abs(), units), unit_word(units, 1), if offset >= 0.0 { "out" } else { "in" }) },
        other => other,
    }
}

/// A sketch line moved `offset` along `unit`, square to itself, through the solver — the
/// corners it shares following.
fn move_line(features: &[Feature], sketch: (&Feature, &SketchFeature), line: &Entity, unit: DVec2, offset: f64, units: DisplayUnits) -> SnapPlan {
    let (sf, sk) = sketch;
    if line.fixed || line.projected {
        return direct(format!("the line of {} under this face is {}", sf.name, if line.projected { "projected from the part" } else { "fixed" }));
    }
    let delta = unit * offset;
    let mut next = sk.sketch.clone();
    let moved = translate_entity(&detached(line), delta);
    let Shape::Line { a: ma, b: mb } = moved.shape else { return direct(format!("no line of {} lies under this face", sf.name)) };
    if let Some(e) = next.entity_mut(moved.id) {
        *e = moved.clone();
    }
    // The wall is held on its new carrier while the solver carries the corners after it: its
    // two ends at a set distance from a fixed helper line laid parallel beyond it, free to
    // slide along — a neighbour that meets it at a slant draws the corner along, not the wall
    // off its place.
    let mut helper_line = Entity::new(Shape::Line { a: ma + unit * HELPER_MM, b: mb + unit * HELPER_MM });
    helper_line.fixed = true;
    helper_line.construction = true;
    let helper = next.add(helper_line);
    let holds = [
        next.constrain(Rule::Distance { p: Anchor::new(moved.id, AnchorAt::A), q: DistanceTo::Line(helper), value: HELPER_MM }),
        next.constrain(Rule::Distance { p: Anchor::new(moved.id, AnchorAt::B), q: DistanceTo::Line(helper), value: HELPER_MM }),
    ];
    let traces = datum_traces(&sk.frame);
    let report = solve_sketch(&mut next, SolveOptions { datums: &traces, ..Default::default() });
    for id in holds {
        next.remove_constraint(id);
    }
    next.remove_entities(&[helper]);
    if !report.ok {
        return direct(format!("the line of {} under this face cannot move — {HELD}", sf.name));
    }
    sketch_plan(features, sketch, next, format!("{}: a line moves {} {}", sf.name, len_text(offset.abs(), units), unit_word(units, 1)))
}

fn resize_round(features: &[Feature], maker: (&Feature, &ExtrudeFeature), sketch: (&Feature, &SketchFeature), face: &FaceFingerprint, offset: f64, units: DisplayUnits) -> SnapPlan {
    let (mf, x) = maker;
    if x.draft_deg != 0.0 {
        return direct(format!("{} has a taper — its round walls are cones", mf.name));
    }
    let (Some(face_r), Some(face_c)) = (face.radius, face.center) else { return direct("the round face has no axis") };
    let (sf, sk) = sketch;
    let frame = &sk.frame;
    let mut piece: Option<&Entity> = None;
    for e in &sk.sketch.entities {
        let (c, r) = match e.shape {
            Shape::Circle { c, r } | Shape::Arc { c, r, .. } => (c, r),
            _ => continue,
        };
        if e.construction || (r - face_r).abs() > ON_MM {
            continue;
        }
        // The centre on the face's axis: its distance from the axis line.
        if (frame.to_world(c) - face_c).cross(face.dir).length() > ON_MM {
            continue;
        }
        piece = Some(e);
        break;
    }
    let Some(piece) = piece else { return direct(format!("no circle of {} lies under this face", sf.name)) };
    if piece.fixed || piece.projected {
        return direct(format!("the circle of {} under this face is {}", sf.name, if piece.projected { "projected from the part" } else { "fixed" }));
    }
    let was_r = piece.shape.radius().unwrap_or(face_r);
    let radius = round4(was_r + offset);
    if !(radius > 1e-6) {
        return direct("the radius would not be positive");
    }
    let mut next = sk.sketch.clone();
    let id = piece.id;
    if let Some(e) = next.entity_mut(id) {
        let mut d = detached(e);
        d.shape = match d.shape {
            Shape::Circle { c, .. } => Shape::Circle { c, r: radius },
            Shape::Arc { c, start, sweep, .. } => Shape::Arc { c, r: radius, start, sweep },
            s => s,
        };
        *e = d;
    }
    // A radius or a diameter dimension on the piece is what the operator typed; the scan
    // overrules it here, and the dimension reads the new value. With none, one holds the radius
    // for the length of the solve.
    let typed = next.constraints.iter().position(|c| matches!(c.rule, Rule::Radius { entity, .. } | Rule::Diameter { entity, .. } if entity == id));
    if let Some(i) = typed {
        let c = &mut next.constraints[i];
        let v = if matches!(c.rule, Rule::Diameter { .. }) { 2.0 * radius } else { radius };
        c.rule.set_value(v);
    }
    let hold = typed.is_none().then(|| next.constrain(Rule::Radius { entity: id, value: radius }));
    let traces = datum_traces(&sk.frame);
    let report = solve_sketch(&mut next, SolveOptions { datums: &traces, ..Default::default() });
    if let Some(h) = hold {
        next.remove_constraint(h);
    }
    if !report.ok {
        return direct(format!("the circle of {} under this face cannot change — {HELD}", sf.name));
    }
    let what = if matches!(piece.shape, Shape::Arc { .. }) { "arc" } else { "circle" };
    sketch_plan(features, sketch, next, format!("{}: the {what} under this face, radius {} → {} {}", sf.name, len_text(was_r, units), len_text(radius, units), unit_word(units, 1)))
}

/// A revolve's bore, rim or hub — a cylinder about the revolve's own axis — resized by the line
/// of the section that runs along the axis at its radius: that line moves away from the axis
/// or towards it. Of two steps at one radius, the one beside the face along the axis.
#[allow(clippy::too_many_arguments)]
fn resize_turned(features: &[Feature], maker: &Feature, sketch: (&Feature, &SketchFeature), face: &FaceFingerprint, axis: (DVec3, DVec3), offset: f64, units: DisplayUnits) -> SnapPlan {
    let Some(face_r) = face.radius else { return direct("the round face has no radius") };
    let (sf, sk) = sketch;
    let frame = &sk.frame;
    let o = to2(frame, axis.0);
    let d = DVec2::new(axis.1.dot(frame.u), axis.1.dot(frame.v));
    // A sheet point's signed distance from the axis, and its place along it.
    let out = |q: DVec2| (q.x - o.x) * -d.y + (q.y - o.y) * d.x;
    let along = |q: DVec2| (q.x - o.x) * d.x + (q.y - o.y) * d.y;
    let beside = along(to2(frame, face.centroid));
    let mut line: Option<(&Entity, DVec2)> = None;
    let mut nearest = f64::INFINITY;
    for e in &sk.sketch.entities {
        let Shape::Line { a, b } = e.shape else { continue };
        if e.construction {
            continue;
        }
        let (oa, ob) = (out(a), out(b));
        if oa * ob <= 0.0 || (oa.abs() - face_r).abs() > ON_MM || (ob.abs() - face_r).abs() > ON_MM {
            continue;
        }
        let gap = ((along(a) + along(b)) / 2.0 - beside).abs();
        if gap < nearest {
            nearest = gap;
            line = Some((e, a));
        }
    }
    let Some((line, a)) = line else { return direct(format!("no line of {} runs along the axis at this face's radius", sf.name)) };
    let radius = round4(face_r + offset);
    if !(radius > 1e-6) {
        return direct("the radius would not be positive");
    }
    let side = if out(a) > 0.0 { 1.0 } else { -1.0 };
    match move_line(features, sketch, line, DVec2::new(-d.y * side, d.x * side), offset, units) {
        SnapPlan::Sketch { features, .. } => SnapPlan::Sketch { features, text: format!("{}: the line of {}'s section under this face, radius {} → {} {}", sf.name, maker.name, len_text(face_r, units), len_text(radius, units), unit_word(units, 1)) },
        other => other,
    }
}

/// A face a revolve made: a shoulder — flat, square to the axis — by the section's line under
/// it, as an extrude's wall; a cylinder about the axis by the line along it. The flat ends of a
/// part turn are the angle's, and a cone or a torus comes of a line that leans or an arc.
#[allow(clippy::too_many_arguments)]
fn snap_turned(features: &[Feature], maker: (&Feature, &RevolveFeature), sketch: (&Feature, &SketchFeature), face: &FaceFingerprint, offset: f64, elements: &[Element], units: DisplayUnits) -> SnapPlan {
    let (mf, x) = maker;
    if x.pitch.is_some_and(|p| p != 0.0) {
        return direct(format!("{} is a coil — its faces wind along the axis, and none is a shoulder or a cylinder", mf.name));
    }
    let context = Part { elements: elements.to_vec(), ..Part::default() };
    let Some((ao, ad)) = resolve_axis(&x.axis, Some(sketch.1), &context) else { return direct(format!("the axis of {} is gone", mf.name)) };
    let Some(ad) = ad.try_normalize() else { return direct(format!("the axis of {} is gone", mf.name)) };
    let frame = &sketch.1.frame;
    let n = frame.normal;
    if ad.dot(n).abs() > 1e-6 || (ao - frame.origin).dot(n).abs() > ON_MM {
        return direct(format!("the axis of {} does not lie in its sketch's plane", mf.name));
    }
    let square = face.dir.dot(ad).abs() > 1.0 - 1e-6;
    match face.kind {
        FaceKind::Plane => {
            if square {
                move_wall(features, mf, sketch, face, offset, units)
            } else {
                direct(format!("the face is a flat end of {}'s turn — that is its angle", mf.name))
            }
        }
        FaceKind::Cylinder if square && face.center.is_some_and(|c| (c - ao).cross(ad).length() <= ON_MM) => resize_turned(features, mf, sketch, face, (ao, ad), offset, units),
        FaceKind::Cone => direct(format!("the face is a cone of {} — the line of its section leans to the axis", mf.name)),
        _ => direct(format!("the face is neither a shoulder nor a cylinder about the axis of {}", mf.name)),
    }
}

/// How several faces move by one `offset`: the edits of the feature list that move them all —
/// each planned on the list as the one before left it, so both ends of one extrude change its
/// From and its To — or, when any of them traces back to nothing, the reason they are all moved
/// on the body instead, as one Move face. Half the faces in the history and half on the body
/// would be two edits for one command, and the second would be made against a body the first
/// had already changed.
pub fn plan_moves(features: &[Feature], faces: &[FaceFingerprint], offset: f64, elements: &[Element], units: DisplayUnits) -> SnapPlan {
    if faces.is_empty() {
        return direct("no face is picked");
    }
    let mut list: Vec<Feature> = features.to_vec();
    let mut sketch_route = false;
    let mut texts: Vec<String> = Vec::new();
    for face in faces {
        let plan = plan_snap(&list, face, offset, elements, units);
        let (next, text) = match plan {
            SnapPlan::Direct { why } => return if faces.len() == 1 { SnapPlan::Direct { why } } else { direct(format!("one of them cannot be traced: {why}")) },
            SnapPlan::Sketch { features, text } => {
                sketch_route = true;
                (features, text)
            }
            SnapPlan::Extent { features, text } => (features, text),
        };
        list = next;
        if !texts.contains(&text) {
            texts.push(text);
        }
    }
    let text = texts.join("; ");
    if sketch_route {
        SnapPlan::Sketch { features: list, text }
    } else {
        SnapPlan::Extent { features: list, text }
    }
}

/// How a face moves by `offset` — along its outward normal for a plane, in radius for a
/// cylinder: the edit of the feature list that does it, or the reason it has to be done on the
/// body. `elements` are the measured elements, for a revolve about one of them.
pub fn plan_snap(features: &[Feature], face: &FaceFingerprint, offset: f64, elements: &[Element], units: DisplayUnits) -> SnapPlan {
    if !(offset.abs() > 1e-9) {
        return direct("the face has nowhere to move");
    }
    let Some(maker) = features.iter().find(|f| f.id as i64 == face.origin) else { return direct("the feature that made this face is not in the list") };
    let sketch_id = match &maker.kind {
        FeatureKind::Extrude(x) => x.sketch_id,
        FeatureKind::Revolve(x) => x.sketch_id,
        _ => return direct(format!("{} made this face, and only an extrude's or a revolve's faces trace back to a sketch or a distance", maker.name)),
    };
    let Some((sf, sk)) = features.iter().find_map(|f| match &f.kind {
        FeatureKind::Sketch(s) if f.id == sketch_id => Some((f, s)),
        _ => None,
    }) else {
        return direct(format!("the sketch of {} is gone", maker.name));
    };
    let x = match &maker.kind {
        FeatureKind::Revolve(r) => return snap_turned(features, (maker, r), (sf, sk), face, offset, elements, units),
        FeatureKind::Extrude(x) => x,
        _ => return direct("the face is neither an end nor a wall of its extrude"),
    };
    let along = face.dir.dot(sk.frame.normal).abs();
    match face.kind {
        FaceKind::Plane if along > 1.0 - 1e-6 => move_cap(features, (maker, x), sk, face, offset, units),
        FaceKind::Plane if along < 1e-4 => move_wall(features, maker, (sf, sk), face, offset, units),
        FaceKind::Plane => direct("the face leans to the sketch plane"),
        FaceKind::Cylinder if along > 1.0 - 1e-6 => resize_round(features, (maker, x), (sf, sk), face, offset, units),
        _ => direct("the face is neither an end nor a wall of its extrude"),
    }
}

#[cfg(test)]
mod tests {
    //! ScanRuler's `reSnapEdit.test.ts`.
    use super::*;
    use crate::build::{build, default_regions};
    use crate::frame::{SectionFrame, WorldAxis};
    use crate::model::{AxisRef, BodyOp, FeatureFlags, PlaneRef};

    const MM: DisplayUnits = DisplayUnits::Mm;

    fn feature(id: FeatureId, name: &str, kind: FeatureKind) -> Feature {
        Feature { id, name: name.into(), flags: FeatureFlags::default(), kind }
    }

    fn sketch_feature(sketch: Sketch) -> Feature {
        feature(1, "Sketch 1", FeatureKind::Sketch(SketchFeature::on_world(WorldAxis::Z, 0.0, sketch)))
    }

    fn extrude(regions: Vec<String>, extent: ExtrudeExtent) -> Feature {
        feature(2, "Extrude 1", FeatureKind::Extrude(ExtrudeFeature::new(1, regions, extent, BodyOp::New)))
    }

    /// The 40 × 20 rectangle with a circle of R3 at (10, 10) in it, extruded 10 — the circle a
    /// hole, its region left unpicked.
    fn block() -> Vec<Feature> {
        let mut s = Sketch::default();
        s.add_rectangle(DVec2::ZERO, DVec2::new(40.0, 20.0));
        s.add_shape(Shape::Circle { c: DVec2::new(10.0, 10.0), r: 3.0 });
        let regions = find_regions(&s);
        let outer = regions.iter().max_by(|a, b| a.area.total_cmp(&b.area)).map(|r| r.key.clone()).unwrap_or_default();
        vec![sketch_feature(s), extrude(vec![outer], ExtrudeExtent::Distance { from: 0.0, to: 10.0 })]
    }

    fn plane(centroid: [f64; 3], dir: [f64; 3], origin: i64) -> FaceFingerprint {
        FaceFingerprint { kind: FaceKind::Plane, centroid: DVec3::from(centroid), dir: DVec3::from(dir), area: 100.0, radius: None, center: None, origin }
    }

    fn extent_of(f: &Feature) -> Option<&ExtrudeExtent> {
        match &f.kind {
            FeatureKind::Extrude(x) => Some(&x.extent),
            _ => None,
        }
    }

    fn sketch_of(f: &Feature) -> &Sketch {
        match &f.kind {
            FeatureKind::Sketch(s) => &s.sketch,
            _ => panic!("not a sketch"),
        }
    }

    fn edit(plan: SnapPlan) -> (Vec<Feature>, String) {
        match plan {
            SnapPlan::Extent { features, text } | SnapPlan::Sketch { features, text } => (features, text),
            SnapPlan::Direct { why } => panic!("{why}"),
        }
    }

    #[test]
    fn moves_a_cap_by_the_extrudes_distance_whichever_end_and_whichever_way_it_faces() {
        let f = block();
        let top = plan_snap(&f, &plane([20.0, 10.0, 10.0], [0.0, 0.0, 1.0], 2), 2.0, &[], MM);
        assert!(matches!(top, SnapPlan::Extent { .. }));
        let (features, text) = edit(top);
        assert_eq!(extent_of(&features[1]), Some(&ExtrudeExtent::Distance { from: 0.0, to: 12.0 }));
        assert!(text.contains("To 10 → 12 mm"), "{text}");
        // The bottom faces −Z: out is down, and it is the From that moves.
        let (features, _) = edit(plan_snap(&f, &plane([20.0, 10.0, 0.0], [0.0, 0.0, -1.0], 2), 1.5, &[], MM));
        assert_eq!(extent_of(&features[1]), Some(&ExtrudeExtent::Distance { from: -1.5, to: 10.0 }));
        // A symmetric extrude stops being one.
        let mut sym = f.clone();
        if let FeatureKind::Extrude(x) = &mut sym[1].kind {
            x.extent = ExtrudeExtent::Symmetric { total: 10.0 };
        }
        let (features, _) = edit(plan_snap(&sym, &plane([20.0, 10.0, 5.0], [0.0, 0.0, 1.0], 2), -1.0, &[], MM));
        assert_eq!(extent_of(&features[1]), Some(&ExtrudeExtent::Distance { from: -5.0, to: 4.0 }));
    }

    #[test]
    fn moves_several_faces_as_one_edit_both_ends_of_an_extrude_or_all_of_them_on_the_body_when_one_does_not_trace() {
        let f = block();
        let (features, text) = edit(plan_moves(&f, &[plane([20.0, 10.0, 10.0], [0.0, 0.0, 1.0], 2), plane([20.0, 10.0, 0.0], [0.0, 0.0, -1.0], 2)], 1.0, &[], MM));
        assert_eq!(extent_of(&features[1]), Some(&ExtrudeExtent::Distance { from: -1.0, to: 11.0 }));
        assert!(text.contains("To 10 → 11 mm") && text.contains("From 0 → -1 mm") && text.find("To 10") < text.find("From 0"), "{text}");
        // One face made by something else: no half measures, all on the body.
        let mixed = plan_moves(&f, &[plane([20.0, 10.0, 10.0], [0.0, 0.0, 1.0], 2), plane([20.0, 10.0, 0.0], [0.0, 0.0, -1.0], 99)], 1.0, &[], MM);
        assert!(mixed.why().is_some_and(|w| w.contains("one of them cannot be traced")), "{mixed:?}");
    }

    #[test]
    fn moves_a_wall_by_its_sketch_line_the_corners_following_and_the_pick_of_the_profile_kept() {
        let f = block();
        let right = plan_snap(&f, &plane([40.0, 10.0, 5.0], [1.0, 0.0, 0.0], 2), 0.8, &[], MM);
        assert!(matches!(right, SnapPlan::Sketch { .. }), "{right:?}");
        let (features, _) = edit(right);
        let s = sketch_of(&features[0]);
        let xs: Vec<f64> = s.entities.iter().flat_map(|e| match e.shape {
            Shape::Line { a, b } => vec![a.x, b.x],
            _ => vec![],
        }).collect();
        assert!((xs.iter().copied().fold(f64::MIN, f64::max) - 40.8).abs() < 1e-6);
        assert!(xs.iter().copied().fold(f64::MAX, f64::min).abs() < 1e-6);
        // Still one closed rectangle with its hole, and the extrude still picks it.
        let regions = find_regions(s);
        let outer = regions.iter().max_by(|a, b| a.area.total_cmp(&b.area)).expect("a region");
        assert!((outer.area - (40.8 * 20.0 - std::f64::consts::PI * 9.0)).abs() < 0.5, "{}", outer.area);
        match &features[1].kind {
            FeatureKind::Extrude(x) => assert_eq!(x.regions, vec![outer.key.clone()]),
            _ => panic!(),
        }
        // The helper the move was solved with is gone again.
        assert_eq!(s.entities.len(), 5);
        assert_eq!(s.constraints.len(), sketch_of(&f[0]).constraints.len());
        // Inward, the far wall.
        let (features, _) = edit(plan_snap(&f, &plane([0.0, 10.0, 5.0], [-1.0, 0.0, 0.0], 2), -0.5, &[], MM));
        let lx = sketch_of(&features[0]).entities.iter().flat_map(|e| match e.shape {
            Shape::Line { a, b } => vec![a.x, b.x],
            _ => vec![],
        }).fold(f64::MAX, f64::min);
        assert!((lx - 0.5).abs() < 1e-6, "{lx}");
    }

    #[test]
    fn resizes_a_bore_by_its_circle_and_a_typed_radius_reads_the_new_value() {
        let f = block();
        let bore = FaceFingerprint { kind: FaceKind::Cylinder, centroid: DVec3::new(10.0, 10.0, 5.0), dir: DVec3::Z, area: 188.0, radius: Some(3.0), center: Some(DVec3::new(10.0, 10.0, 5.0)), origin: 2 };
        let (features, _) = edit(plan_snap(&f, &bore, 0.4, &[], MM));
        let circle = sketch_of(&features[0]).entities.iter().find_map(|e| match e.shape {
            Shape::Circle { r, .. } => Some(r),
            _ => None,
        });
        assert!((circle.unwrap_or(0.0) - 3.4).abs() < 1e-6);
        let mut typed = block();
        if let FeatureKind::Sketch(s) = &mut typed[0].kind {
            let id = s.sketch.entities.iter().find(|e| matches!(e.shape, Shape::Circle { .. })).map(|e| e.id).unwrap_or(0);
            s.sketch.constrain(Rule::Radius { entity: id, value: 3.0 });
        }
        let (features, _) = edit(plan_snap(&typed, &bore, 0.4, &[], MM));
        let v = sketch_of(&features[0]).constraints.iter().find_map(|c| match c.rule {
            Rule::Radius { value, .. } => Some(value),
            _ => None,
        });
        assert!((v.unwrap_or(0.0) - 3.4).abs() < 1e-6);
    }

    /// A sketch on the ZX plane (u the radius, v the height) and the revolve of it about Z.
    fn turned(outline: &[[f64; 2]], angle_deg: f64) -> Vec<Feature> {
        let frame = SectionFrame { origin: DVec3::ZERO, normal: DVec3::new(0.0, -1.0, 0.0), u: DVec3::X, v: DVec3::Z };
        let mut s = Sketch::default();
        let ids: Vec<EntityId> = (0..outline.len()).map(|i| s.add_shape(Shape::Line { a: DVec2::from(outline[i]), b: DVec2::from(outline[(i + 1) % outline.len()]) })).collect();
        for i in 0..ids.len() {
            s.constrain(Rule::Coincident { p: Anchor::new(ids[i], AnchorAt::B), q: Anchor::new(ids[(i + 1) % ids.len()], AnchorAt::A) });
        }
        let regions = default_regions(&find_regions(&s));
        let sk = SketchFeature { plane: PlaneRef::world(WorldAxis::Y, 0.0), frame, sketch: s, exprs: Default::default(), image: None, inset: None };
        let rev = RevolveFeature::new(1, regions, AxisRef::World { axis: WorldAxis::Z }, angle_deg, BodyOp::New);
        vec![feature(1, "Sketch 1", FeatureKind::Sketch(sk)), feature(2, "Revolve 1", FeatureKind::Revolve(rev))]
    }

    fn line_at(features: &[Feature], pick: impl Fn(DVec2, DVec2) -> bool) -> Option<(DVec2, DVec2)> {
        sketch_of(&features[0]).entities.iter().find_map(|e| match e.shape {
            Shape::Line { a, b } if pick(a, b) => Some((a, b)),
            _ => None,
        })
    }

    fn round_face(kind: FaceKind, z: f64, radius: Option<f64>) -> FaceFingerprint {
        FaceFingerprint { kind, centroid: DVec3::new(0.0, 0.0, z), dir: DVec3::Z, area: 1000.0, radius, center: Some(DVec3::new(0.0, 0.0, z)), origin: 2 }
    }

    #[test]
    fn traces_a_revolves_faces_to_its_section_a_shoulder_to_the_line_square_to_the_axis_a_bore_or_a_rim_to_the_line_along_it() {
        // A stepped ring turned about Z: bore at r = 10, rim at r = 30 up to z = 8, a hub at
        // r = 18 up to z = 20.
        let f = turned(&[[10.0, 0.0], [30.0, 0.0], [30.0, 8.0], [18.0, 8.0], [18.0, 20.0], [10.0, 20.0]], 360.0);
        // The shoulder at z = 8 is 0.6 mm too low: its line goes up, the rim and the hub follow.
        let shoulder = plan_snap(&f, &FaceFingerprint { area: 1800.0, ..plane([0.0, 0.0, 8.0], [0.0, 0.0, 1.0], 2) }, 0.6, &[], MM);
        assert!(matches!(shoulder, SnapPlan::Sketch { .. }), "{shoulder:?}");
        let (features, text) = edit(shoulder);
        let (a, b) = line_at(&features, |a, b| (a.y - b.y).abs() < 1e-6 && a.x.min(b.x) > 17.0 && a.x.max(b.x) < 31.0).expect("the shoulder's line");
        assert!((a.y - 8.6).abs() < 1e-6, "{a:?}");
        assert!((a.x.min(b.x) - 18.0).abs() < 1e-6);
        assert!(text.contains("Sketch 1"));
        // The rim, a cylinder of R 30 about Z, reads 30.4 on the scan: the line along the axis
        // at u = 30 moves out.
        let rim = plan_snap(&f, &round_face(FaceKind::Cylinder, 4.0, Some(30.0)), 0.4, &[], MM);
        let (rim_features, text) = edit(rim);
        let (a, _) = line_at(&rim_features, |a, b| (a.x - b.x).abs() < 1e-6 && a.x > 29.0).expect("the rim's line");
        assert!((a.x - 30.4).abs() < 1e-6);
        assert!(text.contains("radius 30 → 30.4 mm"), "{text}");
        // The bore grows the same way: its line moves away from the axis.
        let (features, _) = edit(plan_snap(&f, &round_face(FaceKind::Cylinder, 10.0, Some(10.0)), 0.25, &[], MM));
        let (a, _) = line_at(&features, |a, b| (a.x - b.x).abs() < 1e-6 && a.x < 11.0).expect("the bore's line");
        assert!((a.x - 10.25).abs() < 1e-6);
        // The hub and the rim are both cylinders about Z; each finds its own line.
        let (features, _) = edit(plan_snap(&f, &round_face(FaceKind::Cylinder, 14.0, Some(18.0)), -0.5, &[], MM));
        assert!(line_at(&features, |a, b| (a.x - b.x).abs() < 1e-6 && (a.x - 17.5).abs() < 1e-3).is_some());
        // The pick of the profile is kept through the edit.
        match &rim_features[1].kind {
            FeatureKind::Revolve(r) => assert_eq!(r.regions.len(), 1),
            _ => panic!(),
        }
        // What does not trace: the flat end of a part turn, and a cone.
        let half = turned(&[[10.0, 0.0], [30.0, 0.0], [30.0, 8.0], [18.0, 8.0], [18.0, 20.0], [10.0, 20.0]], 180.0);
        assert!(!plan_snap(&half, &plane([20.0, 0.0, 4.0], [0.0, -1.0, 0.0], 2), 1.0, &[], MM).edits());
        let cone = plan_snap(&f, &round_face(FaceKind::Cone, 4.0, None), 1.0, &[], MM);
        assert!(cone.why().is_some_and(|w| w.contains("cone") || w.contains("leans")), "{cone:?}");
    }

    #[test]
    fn says_why_when_the_move_has_to_be_made_on_the_body() {
        let f = block();
        // A wall held by a dimension across the block.
        let mut held = block();
        if let FeatureKind::Sketch(s) = &mut held[0].kind {
            let line = |x: f64| s.sketch.entities.iter().find(|e| matches!(e.shape, Shape::Line { a, b } if (a.x - x).abs() < 1e-9 && (b.x - x).abs() < 1e-9)).map(|e| e.id).unwrap_or(0);
            let (left, right) = (line(0.0), line(40.0));
            if let Some(e) = s.sketch.entity_mut(left) {
                e.fixed = true;
            }
            s.sketch.constrain(Rule::Distance { p: Anchor::new(right, AnchorAt::A), q: DistanceTo::Line(left), value: 40.0 });
        }
        let stuck = plan_snap(&held, &plane([40.0, 10.0, 5.0], [1.0, 0.0, 0.0], 2), 0.8, &[], MM);
        assert!(stuck.why().is_some_and(|w| w.contains("holds it where it is")), "{stuck:?}");
        // A face a fillet made, a drafted extrude's wall, a face no line is under, an extrude
        // to a plane.
        let mut with_fillet = f.clone();
        with_fillet.push(feature(3, "Fillet 1", FeatureKind::Fillet(crate::model::FilletFeature { body: crate::model::BodyId::new(2), edges: Vec::new(), edges_where: Vec::new(), radius: 1.0, chain: true, skip_failing: false, variable: Vec::new(), exprs: Default::default() })));
        assert!(!plan_snap(&with_fillet, &plane([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 3), 1.0, &[], MM).edits());
        let mut tapered = f.clone();
        if let FeatureKind::Extrude(x) = &mut tapered[1].kind {
            x.draft_deg = 3.0;
        }
        assert!(!plan_snap(&tapered, &plane([40.0, 10.0, 5.0], [1.0, 0.0, 0.0], 2), 1.0, &[], MM).edits());
        assert!(!plan_snap(&f, &plane([25.0, 10.0, 5.0], [1.0, 0.0, 0.0], 2), 1.0, &[], MM).edits());
        let mut to_plane = f.clone();
        if let FeatureKind::Extrude(x) = &mut to_plane[1].kind {
            x.extent = ExtrudeExtent::ToPlane { from: 0.0, plane: PlaneRef::world(WorldAxis::Z, 10.0) };
        }
        assert!(!plan_snap(&to_plane, &plane([20.0, 10.0, 10.0], [0.0, 0.0, 1.0], 2), 1.0, &[], MM).edits());
    }

    #[test]
    fn leaves_the_sketch_alone_where_the_moved_line_would_run_into_another_profile() {
        // Two blocks apart in one sketch, extruded together: the smaller one's near wall pushed
        // 4 mm into the larger would make the overlap a region no extrude picked.
        let mut s = Sketch::default();
        s.add_rectangle(DVec2::ZERO, DVec2::new(20.0, 10.0));
        s.add_rectangle(DVec2::new(22.0, 0.0), DVec2::new(32.0, 10.0));
        let regions = default_regions(&find_regions(&s));
        let f = vec![sketch_feature(s), extrude(regions, ExtrudeExtent::Distance { from: 0.0, to: 10.0 })];
        let into = plan_snap(&f, &plane([22.0, 5.0, 5.0], [-1.0, 0.0, 0.0], 2), 4.0, &[], MM);
        assert!(into.why().is_some_and(|w| w.contains("run a profile of Sketch 1 into another")), "{into:?}");
        // Short of the other block, the line moves.
        assert!(matches!(plan_snap(&f, &plane([22.0, 5.0, 5.0], [-1.0, 0.0, 0.0], 2), 1.0, &[], MM), SnapPlan::Sketch { .. }));
    }

    fn part_of(features: Vec<Feature>) -> Part {
        let next_id = features.iter().map(|f| f.id + 1).max().unwrap_or(1);
        Part { features, next_id, ..Part::default() }
    }

    fn body_volume(part: &Part) -> f64 {
        let built = build(part);
        assert!(built.errors.is_empty(), "{:?}", built.errors);
        built.bodies.first().and_then(|b| crate::build::volume(&b.body)).unwrap_or(f64::NAN)
    }

    #[test]
    fn carries_a_filleted_blocks_rounds_with_the_wall() {
        let mut s = Sketch::default();
        s.add_rectangle(DVec2::ZERO, DVec2::new(40.0, 20.0));
        let regions = default_regions(&find_regions(&s));
        let mut part = part_of(vec![sketch_feature(s), extrude(regions, ExtrudeExtent::Distance { from: 0.0, to: 10.0 })]);
        let plain = build(&part);
        let edge = plain.bodies[0].topology.edges.iter().find(|e| e.fingerprint.kind == crate::topo::EdgeKind::Line && (e.fingerprint.mid.x - 40.0).abs() < 1e-6 && (e.fingerprint.mid.z - 10.0).abs() < 1e-6).map(|e| e.fingerprint.clone()).expect("the top right edge");
        part.add(FeatureKind::Fillet(crate::model::FilletFeature { body: crate::model::BodyId::new(2), edges: vec![edge], edges_where: Vec::new(), radius: 2.0, chain: true, skip_failing: false, variable: Vec::new(), exprs: Default::default() }));
        let rounded = build(&part);
        assert!(rounded.errors.is_empty(), "{:?}", rounded.errors);
        let volume = crate::build::volume(&rounded.bodies[0].body).unwrap_or(f64::NAN);
        // The right wall of the rounded block, as the viewport would hand it over.
        let wall = rounded.bodies[0].topology.faces.iter().find(|f| f.fingerprint.kind == FaceKind::Plane && f.fingerprint.dir.x > 0.9).map(|f| f.fingerprint).expect("the right wall");
        assert_eq!(wall.origin, 2);
        let (features, _) = edit(plan_snap(&part.features, &wall, 1.0, &[], MM));
        let moved = part_of(features);
        // One millimetre more of the same rounded section, all along.
        let v = body_volume(&moved);
        assert!((v - (volume + 20.0 * 10.0)).abs() / v < 1e-3, "{v} vs {}", volume + 200.0);
        assert_eq!(build(&moved).bodies[0].body.face_count(), rounded.bodies[0].body.face_count());
    }

    #[test]
    fn moves_a_revolved_rings_rim_and_its_top_through_the_kernel() {
        // A ring from r = 10 to 30, 8 high, turned about Z from its section.
        let f = turned(&[[10.0, 0.0], [30.0, 0.0], [30.0, 8.0], [10.0, 8.0]], 360.0);
        let part = part_of(f.clone());
        let ring = build(&part);
        assert!(ring.errors.is_empty(), "{:?}", ring.errors);
        let pi = std::f64::consts::PI;
        let v = crate::build::volume(&ring.bodies[0].body).unwrap_or(f64::NAN);
        assert!((v - pi * 800.0 * 8.0).abs() / v < 1e-3, "{v}");
        let faces: Vec<FaceFingerprint> = ring.bodies[0].topology.faces.iter().map(|f| f.fingerprint).collect();
        let rim = faces.iter().find(|f| f.kind == FaceKind::Cylinder && (f.radius.unwrap_or(0.0) - 30.0).abs() < 1e-6).copied().expect("the rim");
        assert_eq!(rim.origin, 2);
        let (wider, _) = edit(plan_snap(&f, &rim, 0.5, &[], MM));
        let v = body_volume(&part_of(wider));
        assert!((v - pi * (30.5f64.powi(2) - 100.0) * 8.0).abs() / v < 1e-3, "{v}");
        let top = faces.iter().find(|f| f.kind == FaceKind::Plane && f.dir.z > 0.9).copied().expect("the top");
        let (taller, _) = edit(plan_snap(&f, &top, 1.0, &[], MM));
        let v = body_volume(&part_of(taller));
        assert!((v - pi * 800.0 * 9.0).abs() / v < 1e-3, "{v}");
        // The bore grows into the ring: less of it is left.
        let bore = faces.iter().find(|f| f.kind == FaceKind::Cylinder && (f.radius.unwrap_or(0.0) - 10.0).abs() < 1e-6).copied().expect("the bore");
        let (bored, _) = edit(plan_snap(&f, &bore, 2.0, &[], MM));
        let v = body_volume(&part_of(bored));
        assert!((v - pi * (900.0 - 144.0) * 8.0).abs() / v < 1e-3, "{v}");
    }
}
