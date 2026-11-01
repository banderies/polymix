// SPDX-License-Identifier: AGPL-3.0-only
//! The parameter list's edits on a part, and the numbers that follow them. Ported from
//! ScanRuler (`plugins/reverse/state/reStore.ts` addParameter, setParameter, removeParameter
//! and setDraftExpr; `plugins/reverse/state/reFeatureHelpers.ts` followParameters,
//! unlinkSlots and keyParameterPaths; `plugins/reverse/app/projectSection.ts`
//! nextParameterId), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! ScanRuler keeps the parameters in its store beside the feature list; here the part holds
//! them (`Part::parameters`, `Part::next_parameter_id`), and is saved with them. The pure
//! pieces — evaluating the table, an edit refused or taken, a rename written through, the
//! features evaluated again — are `parameters`'; this module puts them together on a part the
//! way the store's actions do, and hands back what the workspace has to do after: build again
//! when a number moved, and put the open box's feature and the sketch being drawn in place
//! when they followed.
//!
//! The drafts open in the workspace's boxes are handed in as they stand ([`Drafts`]) and come
//! back changed in [`Followed`], so that a sketch being drawn can take the move as a step of
//! its own history, as ScanRuler's does.

use std::collections::{BTreeMap, HashSet};

use crate::datums::DatumTrace;
use crate::expression::references;
use crate::frame::SectionFrame;
use crate::model::{Feature, FeatureFlags, FeatureId, FeatureKind, Part, SketchInset};
use crate::parameters::{
    apply_parameters, apply_sketch_expressions, apply_slots, edit_parameter, evaluate_parameters, feature_reads, inset_again, new_parameter, removal_problem, rename_in_feature, rename_in_sketch, replace_sketch, ExprErrors, ExprMap, ParameterId, ParameterKind,
    ParameterPatch, ParameterTable, ParametersApplied, READ_BY_FEATURE_DRAFT, READ_BY_SKETCH_DRAFT,
};
use crate::sketch::{EntityId, Sketch};
use crate::stations::{inset_chains, InsetBy, InsetChainsAsk, StationLoop};

/// The table of a part's parameters, each read after the ones it reads.
pub fn parameter_table(part: &Part) -> ParameterTable {
    evaluate_parameters(&part.parameters)
}

/// The id the next parameter takes: the one the part keeps, or — in a part saved without one,
/// as ScanRuler reads a project — one past the largest.
pub fn next_parameter_id(part: &Part) -> ParameterId {
    let past = part.parameters.iter().map(|p| p.id.saturating_add(1)).max().unwrap_or(1);
    part.next_parameter_id.unwrap_or(past).max(past).max(1)
}

/// A new parameter at zero, named p1, p2 … — the first not taken — of a kind (ScanRuler's
/// default is a length). Its id.
pub fn add_parameter(part: &mut Part, kind: ParameterKind) -> ParameterId {
    let id = next_parameter_id(part);
    let p = new_parameter(&part.parameters, id, kind);
    part.parameters.push(p);
    part.next_parameter_id = Some(id + 1);
    id
}

/// The drafts open in the workspace's boxes as an edit of the parameters reaches them: the
/// feature in the open box (an extrude, a fillet …) as it will be, and the sketch being drawn
/// with the plane it lies in.
#[derive(Debug, Clone, Copy, Default)]
pub struct Drafts<'a> {
    pub feature: Option<&'a FeatureKind>,
    pub sketch: Option<(&'a Sketch, &'a SectionFrame)>,
}

/// What follows an edit of the parameters that was taken.
#[derive(Debug, Default)]
pub struct Followed {
    /// A number of a feature moved: the part is to be built again (ScanRuler's
    /// `buildVersion`). A change that moves no number asks for no build.
    pub rebuild: bool,
    /// The expressions that do not evaluate now, by feature.
    pub errors: BTreeMap<FeatureId, ExprErrors>,
    /// The open box's feature as it reads now, where it changed — a number moved, a rename
    /// written into its expressions.
    pub feature: Option<FeatureKind>,
    /// The sketch being drawn as it reads now, where it changed: a step of the sketch's own
    /// history when a dimension moved.
    pub sketch: Option<Sketch>,
    /// Whether the sketch being drawn changed by a dimension that moved (and not by a rename
    /// alone): what makes it a step of its history.
    pub sketch_moved: bool,
}

/// A feature of the open box as the slots read one: the kind in a feature with no name.
pub fn draft_feature(kind: &FeatureKind) -> Feature {
    Feature { id: 0, name: String::new(), flags: FeatureFlags::default(), kind: kind.clone() }
}

/// An inset profile moved in again from the sketch it came from, by the inset's wall
/// (ScanRuler's `insetChains` with the pieces kept where they lie).
fn inset_chains_of(source: &Sketch, inset: &SketchInset, datums: &[DatumTrace], name: &str) -> Result<Sketch, String> {
    let keep_ids: HashSet<EntityId> = inset.keep.iter().copied().collect();
    let keep = move |_: &StationLoop| keep_ids.clone();
    let ask = InsetChainsAsk { by: InsetBy::Fixed(inset.by), keep: Some(&keep), only: None, keep_points: false, name, datums };
    inset_chains(source, &ask).map(|moved| moved.sketch).map_err(|e| e.0)
}

/// Every feature with its expressions evaluated against a table: the numbers written, the
/// sketches with a changed dimension solved again, the features on them keeping their region
/// keys (a dimension moves pieces, it does not remake them), and a profile moved in from
/// another sketch moved in again by the wall it now comes to.
pub fn apply_table(features: &[Feature], table: &ParameterTable) -> ParametersApplied<Feature> {
    let mut keep_keys = |fs: &[Feature], id: FeatureId, sketch: Sketch| replace_sketch(fs, id, &sketch, &mut |_| None);
    let mut again = |f: &Feature, fs: &[Feature]| inset_again(f, fs, &inset_chains_of);
    apply_parameters(features, table, &mut keep_keys, &mut again)
}

/// The part's features brought up to its parameters, as they are before every build: a
/// number left behind by an older save, an undo, or a box that wrote it otherwise, is written
/// from its expression again. Whether a number moved, and the expressions that do not
/// evaluate, by feature.
pub fn apply_to_part(part: &mut Part) -> (bool, BTreeMap<FeatureId, ExprErrors>) {
    let table = parameter_table(part);
    let r = apply_table(&part.features, &table);
    if r.changed {
        part.features = r.features;
    }
    (r.changed, r.errors)
}

/// The open box's feature following the parameters: a rename written through its
/// expressions, and every expression evaluated into its numbers. None when nothing changed.
pub fn feature_draft_follows(kind: &FeatureKind, table: &ParameterTable, renamed: Option<(&str, &str)>) -> Option<FeatureKind> {
    let f = draft_feature(kind);
    let named = match renamed {
        Some((from, to)) => rename_in_feature(&f, from, to).into_owned(),
        None => f.clone(),
    };
    let next = apply_slots(&named, table).next.into_owned();
    (next != f).then_some(next.kind)
}

/// The sketch being drawn following the parameters: a rename written through its dimensions,
/// and the dimensions evaluated and the sketch solved on them. The sketch and whether a
/// dimension moved; none when nothing changed.
pub fn sketch_draft_follows(sketch: &Sketch, frame: &SectionFrame, table: &ParameterTable, renamed: Option<(&str, &str)>) -> Option<(Sketch, bool)> {
    let named = match renamed {
        Some((from, to)) => rename_in_sketch(sketch, from, to).into_owned(),
        None => sketch.clone(),
    };
    let r = apply_sketch_expressions(&named, frame, table);
    let moved = matches!(r.next, std::borrow::Cow::Owned(_));
    let next = r.next.into_owned();
    (next != *sketch).then_some((next, moved))
}

/// A parameter's name, kind, expression or note changed (ScanRuler's `setParameter`). Refused,
/// with why, when the name will not do or the table would not evaluate whole — the part
/// unchanged. Taken, every expression that reads it is evaluated again (a rename written
/// through every expression first) and the features follow; the drafts' changes come back
/// in [`Followed`]. `Ok(None)` when the edit changes nothing — no undo step, no build.
pub fn set_parameter(part: &mut Part, id: ParameterId, patch: &ParameterPatch, drafts: Drafts) -> Result<Option<Followed>, String> {
    let before = parameter_table(part);
    let edit = edit_parameter(&part.parameters, &before, id, patch)?;
    if edit.parameters == part.parameters {
        return Ok(None);
    }
    let renamed = edit.renamed.as_ref().map(|(from, to)| (from.as_str(), to.as_str()));
    let features: Vec<Feature> = match renamed {
        Some((from, to)) => part.features.iter().map(|f| rename_in_feature(f, from, to).into_owned()).collect(),
        None => part.features.clone(),
    };
    let applied = apply_table(&features, &edit.table);
    let renamed_any = features != part.features;
    let mut followed = Followed { rebuild: applied.changed, errors: applied.errors, ..Followed::default() };
    if applied.changed || renamed_any {
        part.features = applied.features;
    }
    part.parameters = edit.parameters;
    if let Some(kind) = drafts.feature {
        followed.feature = feature_draft_follows(kind, &edit.table, renamed);
    }
    if let Some((sketch, frame)) = drafts.sketch {
        if let Some((next, moved)) = sketch_draft_follows(sketch, frame, &edit.table, renamed) {
            followed.sketch = Some(next);
            followed.sketch_moved = moved;
        }
    }
    Ok(Some(followed))
}

/// Remove a parameter (ScanRuler's `removeParameter`): refused, naming its readers, while
/// anything reads it — another parameter, a feature (rolled back or not), the open box's
/// feature, the sketch being drawn. `Ok(false)` when there is no such parameter.
pub fn remove_parameter(part: &mut Part, id: ParameterId, drafts: Drafts) -> Result<bool, String> {
    let Some(p) = part.parameters.iter().find(|p| p.id == id) else { return Ok(false) };
    let mut reading: Vec<&str> = Vec::new();
    if drafts.feature.is_some_and(|k| feature_reads(&draft_feature(k), &p.name)) {
        reading.push(READ_BY_FEATURE_DRAFT);
    }
    if drafts.sketch.is_some_and(|(s, _)| sketch_reads(s, &p.name)) {
        reading.push(READ_BY_SKETCH_DRAFT);
    }
    if let Some(problem) = removal_problem(&part.parameters, &part.features, id, &reading) {
        return Err(problem);
    }
    part.parameters.retain(|q| q.id != id);
    Ok(true)
}

/// Whether a sketch's dimensions read a name.
pub fn sketch_reads(sketch: &Sketch, name: &str) -> bool {
    sketch.constraints.iter().any(|c| c.rule.is_dimension() && c.expr.as_deref().is_some_and(|e| references(e).iter().any(|r| r == name)))
}

/// One number of an open box set, with the expression it was typed as — or none, for a plain
/// number, which lets an expression it had go (ScanRuler's `setDraftExpr`). The number itself
/// the box writes, as ever.
pub fn set_draft_expr(exprs: &mut ExprMap, path: &str, expr: Option<&str>) {
    match expr.filter(|e| !e.is_empty()) {
        Some(e) => exprs.insert(path, e),
        None => {
            exprs.remove(path);
        }
    }
}

/// The expressions behind some numbers let go — a grip dragged, a number typed on its field,
/// the row's number typed over (ScanRuler's `unlinkSlots`). Whether any was let go.
pub fn unlink_slots(exprs: &mut ExprMap, paths: &[&str]) -> bool {
    let mut gone = false;
    for p in paths {
        gone |= exprs.remove(p).is_some();
    }
    gone
}

/// The slot paths a grip's drag — or the number typed into the grip's field — writes, whose
/// expressions it lets go, the number standing (ScanRuler's `gripSlotPaths`). `side` is the
/// grip as ScanRuler names it: `taper`, `extrudeFrom`, `extrudeTo`, `turn`, `radius`,
/// `thickness`, `move`, or a Move body gizmo's `moveX` … `turnZ`; `symmetric` that the extrude
/// being dragged reaches both ways.
pub fn grip_slot_paths(side: &str, symmetric: bool) -> Vec<String> {
    let paths: &[&str] = match side {
        "taper" => &["taper"],
        "extrudeFrom" | "extrudeTo" if symmetric => &["extent.total"],
        "extrudeFrom" => &["extent.from"],
        "extrudeTo" => &["extent.to"],
        "turn" => &["angleDeg", "turns", "height"],
        "radius" => &["radius", "distance"],
        "thickness" => &["thickness"],
        "move" => &["offset"],
        _ => {
            let field = if side.starts_with("move") { "translate" } else if side.starts_with("turn") { "rotate" } else { return Vec::new() };
            let axis = match side.chars().last() {
                Some('X') => "x",
                Some('Y') => "y",
                Some('Z') => "z",
                _ => return Vec::new(),
            };
            return vec![format!("{field}.{axis}")];
        }
    };
    paths.iter().map(|p| p.to_string()).collect()
}

/// The slots the one number of a feature's row writes (ScanRuler's `keyParameterPaths`, for
/// `setKeyParameter`): the expression behind each is let go when the row's number is typed
/// over.
pub fn key_parameter_paths(kind: &FeatureKind) -> &'static [&'static str] {
    match kind {
        FeatureKind::Extrude(e) if matches!(e.extent, crate::model::ExtrudeExtent::Symmetric { .. }) => &["extent.total"],
        FeatureKind::Extrude(_) => &["extent.to"],
        FeatureKind::Revolve(_) => &["angleDeg", "turns", "height"],
        FeatureKind::Shell(_) | FeatureKind::Extract(_) => &["thickness"],
        FeatureKind::Fillet(_) => &["radius"],
        FeatureKind::Chamfer(_) => &["distance"],
        FeatureKind::Hole(_) => &["diameter"],
        FeatureKind::Draft(_) => &["angleDeg"],
        FeatureKind::MoveFace(_) => &["offset"],
        _ => &[],
    }
}

#[cfg(test)]
mod tests {
    //! Ported from ScanRuler's `plugins/reverse/tests/reParameterStore.test.ts`: the parameter
    //! list ("adds a parameter at zero, named in turn …", "refuses what will not do …",
    //! "refuses a remove while anything reads the parameter"), a feature reading a parameter
    //! ("follows the parameter …", "is named when its parameter is to be removed, and follows
    //! a rename") and a sketch dimension reading one ("is solved again …", "follows the
    //! parameter while the sketch is open"), on a part rather than the store. The undo steps
    //! and the build counter are the workspace's; what they come down to here is the part
    //! left unchanged on a refusal and `rebuild`.
    use glam::{DVec2, DVec3};

    use super::*;
    use crate::frame::WorldAxis;
    use crate::model::{BodyOp, ExtrudeExtent, ExtrudeFeature, SketchFeature};
    use crate::parameters::Parameter;
    use crate::sketch::{Anchor, AnchorAt, ConstraintId, Entity, Rule, Shape};

    fn patch() -> ParameterPatch {
        ParameterPatch::default()
    }
    fn named(name: &str, expression: &str) -> ParameterPatch {
        ParameterPatch { name: Some(name.into()), expression: Some(expression.into()), ..patch() }
    }
    fn expression(e: &str) -> ParameterPatch {
        ParameterPatch { expression: Some(e.into()), ..patch() }
    }
    fn none() -> Drafts<'static> {
        Drafts::default()
    }
    fn set(part: &mut Part, id: ParameterId, p: ParameterPatch) -> Result<Option<Followed>, String> {
        set_parameter(part, id, &p, none())
    }
    fn reading(part: &Part, i: usize) -> (Option<f64>, Option<String>) {
        let t = parameter_table(part);
        (t.readings[i].value, t.readings[i].error.clone())
    }
    fn xy() -> SectionFrame {
        SectionFrame { origin: DVec3::ZERO, normal: DVec3::Z, u: DVec3::X, v: DVec3::Y }
    }

    /// A sketch with one closed square of `size`, its bottom side dimensioned (to `expr`
    /// when given). The sketch and the dimension's id.
    fn square(size: f64, expr: Option<&str>) -> (Sketch, ConstraintId) {
        let mut sketch = Sketch::default();
        let ids = sketch.add_rectangle(DVec2::ZERO, DVec2::splat(size));
        let origin = sketch.add(Entity { fixed: true, ..Entity::new(Shape::Point { p: DVec2::ZERO }) });
        sketch.constrain(Rule::Coincident { p: Anchor::new(ids[0], AnchorAt::A), q: Anchor::new(origin, AnchorAt::Center) });
        let dim = sketch.constrain(Rule::Length { entity: ids[0], value: size });
        if let Some(c) = sketch.constraint_mut(dim) {
            c.expr = expr.map(str::to_string);
        }
        sketch.constrain(Rule::Length { entity: ids[1], value: size });
        (sketch, dim)
    }
    fn bottom_length(sketch: &Sketch) -> f64 {
        match sketch.entities[0].shape {
            Shape::Line { a, b } => (b - a).length(),
            _ => f64::NAN,
        }
    }
    /// A part with a finished square sketch (feature 1).
    fn with_square(part: &mut Part, size: f64, expr: Option<&str>) -> (FeatureId, ConstraintId) {
        let (sketch, dim) = square(size, expr);
        let id = part.add(FeatureKind::Sketch(SketchFeature::on_world(WorldAxis::Z, 0.0, sketch)));
        (id, dim)
    }
    fn extrude(sketch: FeatureId, to: f64) -> ExtrudeFeature {
        ExtrudeFeature::new(sketch, vec!["r:1".to_string()], ExtrudeExtent::Distance { from: 0.0, to }, BodyOp::Join)
    }
    fn extrude_of(part: &Part, id: FeatureId) -> ExtrudeFeature {
        match part.feature(id).map(|f| &f.kind) {
            Some(FeatureKind::Extrude(e)) => e.clone(),
            other => panic!("no extrude {id}: {other:?}"),
        }
    }
    fn sketch_of(part: &Part, id: FeatureId) -> Sketch {
        part.sketch(id).map(|s| s.sketch.clone()).expect("the sketch")
    }

    #[test]
    fn the_parameter_list_adds_a_parameter_at_zero_named_in_turn_and_takes_its_name_kind_and_expression() {
        let mut part = Part::default();
        let id = add_parameter(&mut part, ParameterKind::Length);
        assert_eq!(part.parameters, vec![Parameter { id, name: "p1".into(), kind: ParameterKind::Length, expression: "0".into(), comment: None }]);
        assert_eq!(reading(&part, 0), (Some(0.0), None));
        assert_eq!(add_parameter(&mut part, ParameterKind::Integer), id + 1);
        assert_eq!((part.parameters[1].name.as_str(), part.parameters[1].kind), ("p2", ParameterKind::Integer));
        assert!(set(&mut part, id, named("height", "25.4 / 2")).is_ok_and(|f| f.is_some()));
        assert_eq!((part.parameters[0].name.as_str(), part.parameters[0].expression.as_str()), ("height", "25.4 / 2"));
        assert!((reading(&part, 0).0.unwrap_or(f64::NAN) - 12.7).abs() < 1e-9);
        let p = ParameterPatch { comment: Some("how many".into()), ..named("count", "round(height / 2 mm)") };
        assert!(set(&mut part, id + 1, p).is_ok());
        assert_eq!(reading(&part, 1), (Some(6.0), None));
        assert_eq!(part.parameters[1].comment.as_deref(), Some("how many"));
    }

    #[test]
    fn the_parameter_list_refuses_what_will_not_do_and_changes_nothing() {
        let mut part = Part::default();
        let id = add_parameter(&mut part, ParameterKind::Length);
        assert!(set(&mut part, id, named("width", "40")).is_ok());
        let before = part.clone();
        let refused = |part: &mut Part, p: ParameterPatch| set(part, id, p).err().unwrap_or_default();
        assert!(refused(&mut part, ParameterPatch { name: Some("sin".into()), ..patch() }).contains("keep"));
        assert!(refused(&mut part, ParameterPatch { name: Some("2w".into()), ..patch() }).contains("a letter or an underscore"));
        assert_eq!(refused(&mut part, expression("nothing + 1")), "there is no parameter “nothing”");
        assert!(refused(&mut part, expression("1 /")).contains("expected"));
        assert!(refused(&mut part, ParameterPatch { kind: Some(ParameterKind::Integer), ..expression("7.3") }).contains("not whole"));
        assert_eq!(part, before);
        // An edit that would leave another parameter without a value is refused too.
        let other = add_parameter(&mut part, ParameterKind::Length);
        assert!(set(&mut part, other, named("half", "width / 2")).is_ok());
        assert!(refused(&mut part, ParameterPatch { kind: Some(ParameterKind::Angle), ..patch() }).contains("would leave “half” with no value"));
        assert_eq!(part.parameters[0].kind, ParameterKind::Length);
        // The same text again changes nothing: no step to undo.
        assert!(matches!(set(&mut part, id, ParameterPatch { name: Some("width".into()), ..patch() }), Ok(None)));
    }

    #[test]
    fn the_parameter_list_refuses_a_remove_while_anything_reads_the_parameter() {
        let mut part = Part::default();
        let id = add_parameter(&mut part, ParameterKind::Length);
        assert!(set(&mut part, id, named("width", "40")).is_ok());
        let half = add_parameter(&mut part, ParameterKind::Length);
        assert!(set(&mut part, half, named("half", "width / 2")).is_ok());
        assert!(remove_parameter(&mut part, id, none()).err().unwrap_or_default().contains("half"));
        assert_eq!(remove_parameter(&mut part, half, none()), Ok(true));
        assert_eq!(remove_parameter(&mut part, id, none()), Ok(true));
        assert!(part.parameters.is_empty());
        // Ids are not given twice: the next one counts on.
        assert_eq!(add_parameter(&mut part, ParameterKind::Length), half + 1);
    }

    #[test]
    fn a_feature_reading_a_parameter_follows_it_and_asks_for_a_build() {
        let mut part = Part::default();
        let height = add_parameter(&mut part, ParameterKind::Length);
        assert!(set(&mut part, height, named("height", "20")).is_ok());
        let (sketch, _) = with_square(&mut part, 10.0, None);
        let mut draft = extrude(sketch, 25.0);
        set_draft_expr(&mut draft.exprs, "extent.to", Some("height + 5"));
        assert_eq!(draft.exprs, [("extent.to", "height + 5")].into_iter().collect());
        let id = part.add(FeatureKind::Extrude(draft));
        assert_eq!(extrude_of(&part, id).extent, ExtrudeExtent::Distance { from: 0.0, to: 25.0 });
        let f = set(&mut part, height, expression("30")).ok().flatten().expect("taken");
        assert!(f.rebuild);
        assert_eq!(extrude_of(&part, id).extent, ExtrudeExtent::Distance { from: 0.0, to: 35.0 });
        // A change that moves no number asks for no build.
        let f = set(&mut part, height, ParameterPatch { comment: Some("the boss".into()), ..patch() }).ok().flatten().expect("taken");
        assert!(!f.rebuild);
    }

    #[test]
    fn a_feature_reading_a_parameter_lets_it_go_for_a_number_typed() {
        let mut part = Part::default();
        let height = add_parameter(&mut part, ParameterKind::Length);
        assert!(set(&mut part, height, named("height", "20")).is_ok());
        let (sketch, _) = with_square(&mut part, 10.0, None);
        let mut draft = extrude(sketch, 20.0);
        set_draft_expr(&mut draft.exprs, "extent.to", Some("height"));
        // A plain number typed in the box.
        set_draft_expr(&mut draft.exprs, "extent.to", None);
        assert!(draft.exprs.is_empty());
        set_draft_expr(&mut draft.exprs, "extent.to", Some("height"));
        // The arrow dragged, the number typed on the grip, the row's number typed over.
        let paths = key_parameter_paths(&FeatureKind::Extrude(draft.clone()));
        assert!(unlink_slots(&mut draft.exprs, paths));
        assert!(draft.exprs.is_empty());
        part.add(FeatureKind::Extrude(draft));
        // A parameter nothing reads any more can go.
        assert_eq!(remove_parameter(&mut part, height, none()), Ok(true));
    }

    #[test]
    fn a_feature_reading_a_parameter_is_named_when_it_is_to_be_removed_and_follows_a_rename() {
        let mut part = Part::default();
        let height = add_parameter(&mut part, ParameterKind::Length);
        assert!(set(&mut part, height, named("height", "20")).is_ok());
        let (sketch, _) = with_square(&mut part, 10.0, None);
        let mut draft = extrude(sketch, 10.0);
        set_draft_expr(&mut draft.exprs, "extent.to", Some("height * 2"));
        let kind = FeatureKind::Extrude(draft);
        let open = Drafts { feature: Some(&kind), sketch: None };
        assert!(remove_parameter(&mut part, height, open).err().unwrap_or_default().contains("the feature being set up"));
        let f = set_parameter(&mut part, height, &ParameterPatch { name: Some("h".into()), ..patch() }, open).ok().flatten().expect("taken");
        let Some(FeatureKind::Extrude(followed)) = f.feature else { panic!("the draft did not follow") };
        assert_eq!(followed.exprs, [("extent.to", "h * 2")].into_iter().collect());
        // The draft's number follows its expression as the rename reaches it.
        assert_eq!(followed.extent, ExtrudeExtent::Distance { from: 0.0, to: 40.0 });
        let id = part.add(FeatureKind::Extrude(followed));
        assert!(remove_parameter(&mut part, height, none()).err().unwrap_or_default().contains("Extrude 1"));
        assert!(set(&mut part, height, ParameterPatch { name: Some("tall".into()), ..patch() }).is_ok());
        let e = extrude_of(&part, id);
        assert_eq!(e.exprs, [("extent.to", "tall * 2")].into_iter().collect());
        assert_eq!(e.extent, ExtrudeExtent::Distance { from: 0.0, to: 40.0 });
    }

    #[test]
    fn a_sketch_dimension_reading_a_parameter_is_solved_again_the_features_on_it_keeping_their_regions() {
        let mut part = Part::default();
        let width = add_parameter(&mut part, ParameterKind::Length);
        assert!(set(&mut part, width, named("width", "10")).is_ok());
        let (sketch, dim) = with_square(&mut part, 10.0, Some("width"));
        let extrude_id = part.add(FeatureKind::Extrude(extrude(sketch, 10.0)));
        let regions = extrude_of(&part, extrude_id).regions;
        let f = set(&mut part, width, expression("30")).ok().flatten().expect("taken");
        assert!(f.rebuild);
        assert!((bottom_length(&sketch_of(&part, sketch)) - 30.0).abs() < 1e-6);
        assert_eq!(sketch_of(&part, sketch).constraint(dim).and_then(|c| c.rule.value()), Some(30.0));
        assert_eq!(extrude_of(&part, extrude_id).regions, regions);
        assert!(remove_parameter(&mut part, width, none()).err().unwrap_or_default().contains("Sketch 1"));
    }

    #[test]
    fn a_sketch_dimension_reading_a_parameter_follows_it_while_the_sketch_is_open() {
        let mut part = Part::default();
        let width = add_parameter(&mut part, ParameterKind::Length);
        assert!(set(&mut part, width, named("width", "10")).is_ok());
        let (open, dim) = square(10.0, Some("width"));
        let frame = xy();
        let drafts = Drafts { feature: None, sketch: Some((&open, &frame)) };
        assert!(remove_parameter(&mut part, width, drafts).err().unwrap_or_default().contains("the sketch being drawn"));
        let f = set_parameter(&mut part, width, &expression("20"), drafts).ok().flatten().expect("taken");
        assert!(f.sketch_moved);
        let moved = f.sketch.expect("the open sketch followed");
        assert!((bottom_length(&moved) - 20.0).abs() < 1e-6);
        // A rename reaches into the open sketch, as no step of its own.
        let drafts = Drafts { feature: None, sketch: Some((&moved, &frame)) };
        let f = set_parameter(&mut part, width, &ParameterPatch { name: Some("w".into()), ..patch() }, drafts).ok().flatten().expect("taken");
        assert!(!f.sketch_moved);
        assert_eq!(f.sketch.and_then(|s| s.constraint(dim).and_then(|c| c.expr.clone())).as_deref(), Some("w"));
    }

    /// Not in ScanRuler's tests: what each grip lets go of.
    #[test]
    fn a_grip_lets_go_of_the_slots_it_writes() {
        assert_eq!(grip_slot_paths("extrudeTo", false), ["extent.to"]);
        assert_eq!(grip_slot_paths("extrudeFrom", true), ["extent.total"]);
        assert_eq!(grip_slot_paths("turn", false), ["angleDeg", "turns", "height"]);
        assert_eq!(grip_slot_paths("radius", false), ["radius", "distance"]);
        assert_eq!(grip_slot_paths("moveY", false), ["translate.y"]);
        assert_eq!(grip_slot_paths("turnZ", false), ["rotate.z"]);
        assert!(grip_slot_paths("nothing", false).is_empty());
    }

    /// Not in ScanRuler's tests: before a build the part's numbers are brought up to their
    /// expressions — a number an older save left behind is written again.
    #[test]
    fn the_part_is_brought_up_to_its_parameters_before_a_build() {
        let mut part = Part::default();
        let height = add_parameter(&mut part, ParameterKind::Length);
        assert!(set(&mut part, height, named("height", "12")).is_ok());
        let (sketch, _) = with_square(&mut part, 10.0, None);
        let mut e = extrude(sketch, 3.0);
        set_draft_expr(&mut e.exprs, "extent.to", Some("height"));
        let id = part.add(FeatureKind::Extrude(e));
        let (changed, errors) = apply_to_part(&mut part);
        assert!(changed);
        assert!(errors.is_empty());
        assert_eq!(extrude_of(&part, id).extent, ExtrudeExtent::Distance { from: 0.0, to: 12.0 });
        let (changed, _) = apply_to_part(&mut part);
        assert!(!changed);
        // An expression that does not evaluate leaves its number and says why.
        if let Some(FeatureKind::Extrude(x)) = part.feature_mut(id).map(|f| &mut f.kind) {
            set_draft_expr(&mut x.exprs, "extent.to", Some("gone + 1"));
        }
        let (changed, errors) = apply_to_part(&mut part);
        assert!(!changed);
        assert_eq!(errors.get(&id).and_then(|e| e.get("extent.to")), Some("there is no parameter “gone”"));
    }

    /// Not in ScanRuler's tests: the id the next parameter takes, from a part saved with one
    /// or without.
    #[test]
    fn the_next_parameter_id_counts_on_from_the_part() {
        let mut part = Part::default();
        assert_eq!(next_parameter_id(&part), 1);
        part.parameters.push(Parameter { id: 7, name: "a".into(), kind: ParameterKind::Number, expression: "1".into(), comment: None });
        assert_eq!(next_parameter_id(&part), 8);
        part.next_parameter_id = Some(12);
        assert_eq!(next_parameter_id(&part), 12);
        let json = part.to_json();
        assert!(json.contains("\"nextParameterId\": 12"), "{json}");
        assert_eq!(Part::from_json(&json).map(|p| p.next_parameter_id), Ok(Some(12)));
    }
}
