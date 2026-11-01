// SPDX-License-Identifier: AGPL-3.0-only
//! A fitted sketch's near-relations taken as the design's in one go. Ported from ScanRuler
//! (`plugins/reverse/core/accept.ts`), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! Every candidate (`crate::intent`) within so many sigmas of holding — the scan's noise says
//! it holds — is added and the sketch solved, nearest the mark first. One the solve cannot
//! meet with those before it is taken back (it fights a relation already accepted), and so is
//! one that says nothing new (it follows from them); one only part of which follows is taken
//! for the rest (its constraints say which). What is left is offered, with why. The trials are
//! solved with the part's planes, as the sketch is. A relation within the sigmas may still be
//! one the fit's pieces can only meet by leaving their points: a trial that moves any piece
//! farther than the rule's `max_move_mm` from where the fit put it is taken back too. The move
//! is read off the piece's curve — how far its ends and middle as they were stand off the line
//! or circle it lies on after — not by an arc's centre, nor by a join sliding along the pieces.
//! Unless told, the limit is three of the sigmas the candidate itself was judged by: the
//! per-point sigma of the pieces it relates, their own residual floored at the chain's
//! scatter. When every candidate of a kind is left for moving, a note says how far they went
//! and the limit that takes them.

use serde::{Deserialize, Serialize};

use crate::datums::{DatumPlane, DatumTrace};
use crate::intent::{candidates_of, point_sigma, round4, Candidate, CandidateKind, EqualWhat, IntentNet, OffUnit, DEFAULT_NET};
use crate::project::retake_moved;
use crate::sketch::{ConstraintId, Entity, EntityId, Shape, Sketch};
use crate::solver::{solve_sketch, SolveOptions};

/// What is accepted: candidates at most `sigmas` off, of the `kinds` given (every kind when
/// none), moving no piece farther than `max_move_mm` (when given).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct AcceptRule {
    pub sigmas: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kinds: Option<Vec<CandidateKind>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_move_mm: Option<f64>,
}

/// A candidate as the answer gives it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateBrief {
    pub kind: CandidateKind,
    pub pieces: Vec<EntityId>,
    pub off: f64,
    pub unit: OffUnit,
    pub sigmas: Option<f64>,
    pub datum: Option<DatumPlane>,
    pub what: Option<EqualWhat>,
    pub nominal: Option<f64>,
}

/// The piece a trial moved farthest, how far, and the limit that trial was held to, mm.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Moves {
    pub piece: EntityId,
    pub mm: f64,
    pub limit_mm: f64,
}

/// Why a candidate was not taken.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LeftWhy {
    /// Over the sigmas.
    Sigmas,
    /// Of a kind not asked for.
    Kind,
    /// In conflict with one accepted.
    Conflict,
    /// Implied by those accepted.
    Implied,
    /// Moving a piece farther than the limit (`moves` says which, and how far).
    Moved,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Accepted {
    pub brief: CandidateBrief,
    pub constraints: Vec<ConstraintId>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Left {
    pub brief: CandidateBrief,
    pub why: LeftWhy,
    pub moves: Option<Moves>,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct AcceptResult {
    pub accepted: Vec<Accepted>,
    pub left: Vec<Left>,
    /// The limit the trials were held to, mm: the rule's, or the largest of the candidates' own
    /// when it gave none.
    pub max_move_mm: f64,
    /// Where every candidate of a kind within the sigmas was left for moving: how far they
    /// went, and the `max_move_mm` that takes them.
    pub note: Option<String>,
}

/// How many candidates are looked at, at most.
const ACCEPT_LIMIT: usize = 400;
/// How many sigmas a trial may move a piece, unless the rule says otherwise.
pub const MOVE_SCATTERS: f64 = 3.0;
/// The least the default limit is, mm: a slice off a clean mesh has next to no scatter, and
/// the solver's own settling is not a move.
const MOVE_FLOOR_MM: f64 = 0.01;

/// The fit's scatter: the largest any fitted piece carries, mm. 0 for a sketch with none.
pub fn fit_scatter(sketch: &Sketch) -> f64 {
    sketch.entities.iter().filter(|e| !e.points.is_empty()).fold(0.0, |m, e| m.max(e.scatter))
}

/// The piece moved farthest between two states of a sketch, or `None` when none moved. Pieces
/// new in `now` are not counted.
fn farthest(was: &Sketch, now: &Sketch) -> Option<(EntityId, f64)> {
    let mut best: Option<(EntityId, f64)> = None;
    for e in &now.entities {
        let Some(w) = was.entity(e.id) else { continue };
        if w.kind() != e.kind() {
            continue;
        }
        let mm = off_curve(w, e);
        if mm.is_finite() && best.is_none_or(|b| mm > b.1) {
            best = Some((e.id, mm));
        }
    }
    best
}

/// How far a piece moved off where it was, mm: the most its ends and middle as they were stand
/// off the line or circle it lies on now. A join that only slides along the pieces moves
/// neither off its points, and is not counted. Other pieces as `retake_moved` has them.
pub fn off_curve(was: &Entity, now: &Entity) -> f64 {
    match (was.shape.clone(), now.shape.clone()) {
        (Shape::Line { a: wa, b: wb }, Shape::Line { a, b }) => {
            let (dx, dy) = (b.x - a.x, b.y - a.y);
            let l = dx.hypot(dy);
            if !(l > 1e-9) {
                return retake_moved(was, now);
            }
            let off = |p: glam::DVec2| ((p.x - a.x) * dy - (p.y - a.y) * dx).abs() / l;
            off(wa).max(off(wb))
        }
        (Shape::Arc { c: wc, r: wr, start, sweep }, Shape::Arc { c, r, .. }) => {
            let at = |t: f64| glam::DVec2::new(wc.x + wr * (start + t * sweep).cos(), wc.y + wr * (start + t * sweep).sin());
            let off = |p: glam::DVec2| ((p.x - c.x).hypot(p.y - c.y) - r).abs();
            off(at(0.0)).max(off(at(0.5))).max(off(at(1.0)))
        }
        _ => retake_moved(was, now),
    }
}

fn brief(c: &Candidate) -> CandidateBrief {
    CandidateBrief { kind: c.kind, pieces: c.pieces.clone(), off: c.off, unit: c.unit, sigmas: c.sigmas, datum: c.datum, what: c.what, nominal: c.nominal }
}

/// The sketch put back as it was before a trial.
fn restore(sketch: &mut Sketch, was: Sketch) {
    *sketch = was;
}

/// The sketch with the candidates the rule accepts added, solved; in place.
pub fn accept_candidates(sketch: &mut Sketch, datums: &[DatumTrace], rule: &AcceptRule) -> AcceptResult {
    accept_candidates_with(sketch, datums, rule, DEFAULT_NET)
}

pub fn accept_candidates_with(sketch: &mut Sketch, datums: &[DatumTrace], rule: &AcceptRule, net: IntentNet) -> AcceptResult {
    let held: Vec<_> = sketch.constraints.iter().map(|c| c.rule.clone()).collect();
    let candidates = candidates_of(&sketch.entities, datums, net, &held, ACCEPT_LIMIT);
    let mut accepted: Vec<Accepted> = Vec::new();
    let mut left: Vec<Left> = Vec::new();
    // The trial solves are the real one's: with the part's coordinate planes given.
    let options = || SolveOptions { datums, ..Default::default() };
    let start = solve_sketch(sketch, options());
    let mut redundant = start.redundant;
    let mut equations = start.equations;
    // Where the pieces were before anything was accepted: every trial is measured from there,
    // so small moves cannot add up past the limit.
    let fitted = sketch.clone();
    let scatter = fit_scatter(sketch);
    let sigmas: std::collections::HashMap<EntityId, f64> = fitted.entities.iter().map(|e| (e.id, point_sigma(e).unwrap_or(0.0))).collect();
    let limit_of = |c: &Candidate| -> f64 {
        rule.max_move_mm.unwrap_or_else(|| {
            let most = c.pieces.iter().map(|p| sigmas.get(p).copied().unwrap_or(0.0)).fold(scatter, f64::max);
            (MOVE_SCATTERS * most).max(MOVE_FLOOR_MM)
        })
    };
    let mut largest = rule.max_move_mm.unwrap_or((MOVE_SCATTERS * scatter).max(MOVE_FLOOR_MM));
    let too_far = |sketch: &Sketch, limit: f64| -> Option<Moves> {
        let (piece, mm) = farthest(&fitted, sketch)?;
        (mm > limit).then(|| Moves { piece, mm: round4(mm), limit_mm: round4(limit) })
    };
    for c in &candidates {
        if rule.kinds.as_ref().is_some_and(|k| !k.contains(&c.kind)) {
            left.push(Left { brief: brief(c), why: LeftWhy::Kind, moves: None });
            continue;
        }
        if c.sigmas.is_none_or(|s| s > rule.sigmas) {
            left.push(Left { brief: brief(c), why: LeftWhy::Sigmas, moves: None });
            continue;
        }
        let limit = limit_of(c);
        largest = largest.max(limit);
        let before = sketch.clone();
        let ids: Vec<ConstraintId> = c.constraints.iter().map(|k| sketch.constrain(k.clone())).collect();
        let report = solve_sketch(sketch, options());
        let moves = if report.ok { too_far(sketch, limit) } else { None };
        if let Some(moves) = moves {
            restore(sketch, before);
            left.push(Left { brief: brief(c), why: LeftWhy::Moved, moves: Some(moves) });
            continue;
        }
        if report.ok && report.redundant <= redundant {
            accepted.push(Accepted { brief: brief(c), constraints: ids });
            equations = report.equations;
            continue;
        }
        restore(sketch, before);
        // Part of it may follow from what is taken already and the rest not: each of its
        // relations in turn, kept where it pins down something the sketch did not hold.
        if report.ok {
            let mut kept = Vec::new();
            let mut refused: Option<Moves> = None;
            for k in &c.constraints {
                let was = sketch.clone();
                let id = sketch.constrain(k.clone());
                let one = solve_sketch(sketch, options());
                let gained = one.equations as i64 - equations as i64 - (one.redundant as i64 - redundant as i64);
                let moves = if one.ok && gained > 0 { too_far(sketch, limit) } else { None };
                if one.ok && gained > 0 && moves.is_none() {
                    kept.push(id);
                    redundant = one.redundant;
                    equations = one.equations;
                } else {
                    restore(sketch, was);
                    refused = moves.or(refused);
                }
            }
            if !kept.is_empty() {
                accepted.push(Accepted { brief: brief(c), constraints: kept });
                continue;
            }
            if let Some(m) = refused {
                left.push(Left { brief: brief(c), why: LeftWhy::Moved, moves: Some(m) });
                continue;
            }
        }
        left.push(Left { brief: brief(c), why: if report.ok { LeftWhy::Implied } else { LeftWhy::Conflict }, moves: None });
    }
    // The sketch as the last accepted left it, solved.
    solve_sketch(sketch, options());
    let note = moved_note(&accepted, &left);
    AcceptResult { accepted, left, max_move_mm: round4(largest), note }
}

/// What a kind's candidates are called, more than one.
fn kind_words(kind: CandidateKind) -> &'static str {
    match kind {
        CandidateKind::Horizontal => "horizontals",
        CandidateKind::Vertical => "verticals",
        CandidateKind::Parallel => "parallels",
        CandidateKind::Perpendicular => "perpendiculars",
        CandidateKind::Collinear => "in-line pairs",
        CandidateKind::Angle => "round angles",
        CandidateKind::Equal => "equal pairs",
        CandidateKind::Concentric => "concentric pairs",
        CandidateKind::Tangent => "tangents",
        CandidateKind::OnDatum => "on-plane relations",
        CandidateKind::Symmetric => "mirror pairs",
    }
}

const KINDS: [CandidateKind; 11] = [
    CandidateKind::Horizontal,
    CandidateKind::Vertical,
    CandidateKind::Parallel,
    CandidateKind::Perpendicular,
    CandidateKind::Collinear,
    CandidateKind::Angle,
    CandidateKind::Equal,
    CandidateKind::Concentric,
    CandidateKind::Tangent,
    CandidateKind::OnDatum,
    CandidateKind::Symmetric,
];

/// A move as the note gives it: hundredths, or one figure under 0.01.
fn mm_text(mm: f64) -> String {
    if mm >= 0.01 {
        format!("{mm:.2}")
    } else {
        crate::js_number::js_string(crate::js_number::to_precision_number(mm, 1))
    }
}

/// One line for each kind of which every candidate within the sigmas was left for moving: "all
/// 7 tangents moved 0.02–0.04 mm; `maxMoveMm: 0.05` takes them — compare with T". `None` when
/// no kind was.
fn moved_note(accepted: &[Accepted], left: &[Left]) -> Option<String> {
    let mut lines = Vec::new();
    for kind in KINDS {
        let tried: Vec<&Left> = left.iter().filter(|l| l.brief.kind == kind && !matches!(l.why, LeftWhy::Sigmas | LeftWhy::Kind)).collect();
        let moved: Vec<&Left> = tried.iter().copied().filter(|l| l.why == LeftWhy::Moved && l.moves.is_some()).collect();
        if moved.is_empty() || moved.len() < tried.len() || accepted.iter().any(|a| a.brief.kind == kind) {
            continue;
        }
        let mms: Vec<f64> = moved.iter().filter_map(|l| l.moves.map(|m| m.mm)).collect();
        let lo = mms.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = mms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let takes = (hi * 100.0 + 1e-9).ceil() / 100.0;
        let range = if mm_text(lo) == mm_text(hi) { format!("{} mm", mm_text(hi)) } else { format!("{}–{} mm", mm_text(lo), mm_text(hi)) };
        let words = kind_words(kind);
        let what = if moved.len() == 1 { format!("the one {} moved {range}", words.strip_suffix('s').unwrap_or(words)) } else { format!("all {} {words} moved {range}", moved.len()) };
        lines.push(format!("{what}; `maxMoveMm: {}` takes {} — compare with T", crate::js_number::js_string(takes), if moved.len() == 1 { "it" } else { "them" }));
    }
    (!lines.is_empty()).then(|| lines.join("; "))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datums::datum_traces;
    use crate::fit2d::Mulberry32;
    use crate::frame::WorldAxis;
    use crate::model::PlaneRef;
    use crate::section::{fit_all_chains, FitOptions};
    use glam::DVec2;

    /// A stadium in XY — ends R10 about (±20, 0), sides at y = ±10 — sampled every `step`,
    /// with noise along the outward normal, as ScanRuler's `reAccept.test.ts` scans it; here
    /// laid straight on the sheet as the section through it would be.
    fn stadium(step: f64, noise: f64) -> Vec<DVec2> {
        let mut rand = Mulberry32::new(9);
        let mut gauss = move || {
            let u = rand.next_f64().max(1e-12);
            let v = rand.next_f64();
            (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
        };
        let mut pts: Vec<(DVec2, DVec2)> = Vec::new();
        let mut x = -20.0;
        while x < 20.0 {
            pts.push((DVec2::new(x, -10.0), DVec2::new(0.0, -1.0)));
            x += step;
        }
        let half = std::f64::consts::FRAC_PI_2;
        let mut a = -half;
        while a < half {
            pts.push((DVec2::new(20.0 + 10.0 * a.cos(), 10.0 * a.sin()), DVec2::new(a.cos(), a.sin())));
            a += step / 10.0;
        }
        let mut x = 20.0;
        while x > -20.0 {
            pts.push((DVec2::new(x, 10.0), DVec2::new(0.0, 1.0)));
            x -= step;
        }
        let mut a = half;
        while a < 3.0 * half {
            pts.push((DVec2::new(-20.0 + 10.0 * a.cos(), 10.0 * a.sin()), DVec2::new(a.cos(), a.sin())));
            a += step / 10.0;
        }
        let mut chain: Vec<DVec2> = pts.into_iter().map(|(p, n)| p + n * (gauss() * noise)).collect();
        chain.push(chain[0]);
        chain
    }

    fn fitted_stadium() -> (Sketch, Vec<DatumTrace>) {
        let frame = PlaneRef::world(WorldAxis::Z, 0.0).own_frame().expect("a coordinate plane has a frame");
        let mut s = Sketch::default();
        let report = fit_all_chains(&mut s, &[stadium(0.3, 0.02)], &frame, FitOptions::default(), &[]);
        assert!(report.solve.ok, "{report:?}");
        (s, datum_traces(&frame))
    }

    #[test]
    fn takes_the_stadiums_relations_in_one_call_and_the_sketch_solves_on_its_points() {
        let (mut s, datums) = fitted_stadium();
        let mut kinds_fitted: Vec<&str> = s.entities.iter().map(|e| if e.kind() == crate::sketch::ShapeKind::Line { "line" } else { "arc" }).collect();
        kinds_fitted.sort();
        assert_eq!(kinds_fitted, ["arc", "arc", "line", "line"]);
        let made = accept_candidates(&mut s, &datums, &AcceptRule { sigmas: 2.0, ..Default::default() });
        let kinds: Vec<CandidateKind> = made.accepted.iter().map(|a| a.brief.kind).collect();
        let solved = solve_sketch(&mut s, SolveOptions { datums: &datums, ..Default::default() });
        assert!(solved.ok, "{solved:?}");
        // The fit joins smooth corners tangent itself; accept takes the rest.
        assert_eq!(s.constraints.iter().filter(|c| matches!(c.rule, crate::sketch::Rule::Tangent { .. })).count(), 4);
        assert!(kinds.contains(&CandidateKind::Horizontal), "{kinds:?}");
        // The two straight sides tangent to the same two rounds are as long as each other
        // whatever the rounds: equal follows, and is left.
        assert_eq!(made.left.iter().find(|l| l.brief.kind == CandidateKind::Equal && l.brief.what == Some(EqualWhat::Length)).map(|l| l.why), Some(LeftWhy::Implied));
        assert!(kinds.contains(&CandidateKind::Symmetric), "{kinds:?}");
        assert!(kinds.iter().any(|k| *k == CandidateKind::OnDatum), "{kinds:?}");
        assert!(solved.under_constrained.is_empty(), "{:?}", solved.under_constrained);
        assert_eq!(made.left.iter().find(|l| l.brief.kind == CandidateKind::Horizontal).map(|l| l.why), Some(LeftWhy::Implied));
        for a in &made.accepted {
            assert!(a.brief.sigmas.unwrap_or(f64::INFINITY) <= 2.0);
        }
        assert!(made.max_move_mm > 0.0);
        // The relations held cost the fit nothing worth the name.
        assert!(solved.soft_rms < 0.05, "rms {}", solved.soft_rms);
    }

    #[test]
    fn takes_only_the_kinds_asked_for() {
        let (mut s, datums) = fitted_stadium();
        let made = accept_candidates(&mut s, &datums, &AcceptRule { sigmas: 2.0, kinds: Some(vec![CandidateKind::Equal]), max_move_mm: None });
        let kinds: Vec<CandidateKind> = made.accepted.iter().map(|a| a.brief.kind).collect();
        assert_eq!(kinds, vec![CandidateKind::Equal]);
        assert_eq!(made.left.iter().find(|l| l.brief.kind == CandidateKind::Equal).map(|l| l.why), Some(LeftWhy::Implied));
        assert!(!made.left.iter().any(|l| l.brief.kind != CandidateKind::Equal && l.why != LeftWhy::Kind));
    }

    #[test]
    fn a_tight_move_limit_leaves_what_would_move_the_pieces_and_says_so() {
        // Not a ScanRuler test: the limit read back, and the note on a kind all moved.
        let (mut s, datums) = fitted_stadium();
        let made = accept_candidates(&mut s, &datums, &AcceptRule { sigmas: 50.0, kinds: None, max_move_mm: Some(1e-9) });
        assert_eq!(made.max_move_mm, 0.0);
        assert!(made.left.iter().any(|l| l.why == LeftWhy::Moved && l.moves.is_some_and(|m| m.mm > 0.0)));
        if let Some(note) = &made.note {
            assert!(note.contains("takes"), "{note}");
        }
    }
}
