// SPDX-License-Identifier: AGPL-3.0-only
//! A shadow section's outline taken into the workspace as the lines, arcs and circles the
//! section keeps (`shadow`), not as its sampled polylines: the profiles an Extract stands up
//! through the part, and the pieces a sketch on the shadow starts with. Ported from ScanRuler
//! (`plugins/reverse/core/shadowPieces.ts`), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.

use std::f64::consts::PI;

use glam::{DVec2, DVec3};
use serde::{Deserialize, Serialize};

use crate::frame::SectionFrame;
use crate::loop_piece::{LoopPiece, Profile};
use crate::shadow::{shadow_regions, ShadowPiece, ShadowShape};
use crate::sketch::{arc_through, Anchor, AnchorAt, Entity, Rule, Shape, ShapeKind, Sketch};

/// What a section is to an Extract or a sketch: its name, whether it is a shadow, and — once
/// the worker has taken it — the shadow's shape.
#[derive(Debug, Clone, Copy, Default)]
pub struct ShadowSource<'a> {
    pub name: Option<&'a str>,
    /// A shadow section; a cut otherwise.
    pub is_shadow: bool,
    pub shadow: Option<&'a ShadowShape>,
}

/// The block an Extract makes of a shadow: profiles in the part stood along `dir` from `from`
/// to `to` mm off their plane.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShadowPrism {
    pub profiles: Vec<Profile>,
    pub dir: DVec3,
    pub from: f64,
    pub to: f64,
}

/// Why a section stands as no block, in words for the user.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("{0}")]
pub struct ShadowError(pub String);

/// Thinner than this along the normal, mm, a shadow stands as no block.
const MIN_THICKNESS_MM: f64 = 1e-3;

/// Two pieces meeting turn by less than this, radians, and are held tangent where the sketch
/// joins them — a corner the shadow rounded.
const TANGENT_RAD: f64 = 0.5 * PI / 180.0;

/// A piece's middle point, on the sheet; a circle has none.
fn mid_of(p: &ShadowPiece) -> Option<DVec2> {
    match *p {
        ShadowPiece::Line { from, to } => Some((from + to) / 2.0),
        ShadowPiece::Arc { from, center, radius, sweep, .. } => {
            let a = (from.y - center.y).atan2(from.x - center.x) + sweep / 2.0;
            Some(center + DVec2::new(a.cos(), a.sin()) * radius)
        }
        ShadowPiece::Circle { .. } => None,
    }
}

fn loop_in(frame: &SectionFrame, pieces: &[ShadowPiece]) -> Vec<LoopPiece> {
    pieces
        .iter()
        .map(|p| match *p {
            ShadowPiece::Circle { center, radius } => LoopPiece::Circle { center: frame.to_world(center), normal: frame.normal, radius },
            ShadowPiece::Line { from, to } => LoopPiece::Line { a: frame.to_world(from), b: frame.to_world(to) },
            ShadowPiece::Arc { from, to, .. } => LoopPiece::Arc { a: frame.to_world(from), m: frame.to_world(mid_of(p).unwrap_or(from)), b: frame.to_world(to) },
        })
        .collect()
}

/// The shadow as the solid an Extract makes: each outer loop with its holes, standing along the
/// normal through the part's thickness — the block the section draws and writes to STEP — or
/// why there is none.
pub fn shadow_primitive(sec: Option<&ShadowSource>) -> Result<ShadowPrism, ShadowError> {
    let Some(sec) = sec else { return Err(ShadowError("The section this body is taken from is gone.".into())) };
    if !sec.is_shadow {
        return Err(ShadowError(format!("{} is a cut — only a shadow section stands as a body.", sec.name.unwrap_or("That section"))));
    }
    let name = sec.name.unwrap_or("The shadow");
    let Some(sh) = sec.shadow else { return Err(ShadowError(format!("{name} is still being taken — the body is built once it is."))) };
    let regions = shadow_regions(sh);
    if regions.is_empty() {
        return Err(ShadowError(format!("{name} has no outline.")));
    }
    if !(sh.extent.max - sh.extent.min > MIN_THICKNESS_MM) {
        return Err(ShadowError(format!("{name} has no thickness to stand through.")));
    }
    let profiles = regions.iter().map(|r| Profile { outer: loop_in(&sh.frame, &r.outer.pieces), holes: r.holes.iter().map(|h| loop_in(&sh.frame, &h.pieces)).collect() }).collect();
    Ok(ShadowPrism { profiles, dir: sh.frame.normal, from: sh.extent.min, to: sh.extent.max })
}

/// Which way a piece runs at its loop's start and end, on the sketch's sheet: what tells a
/// rounded corner from a sharp one. `arc` is the centre and whether it is run
/// counter-clockwise.
fn headings(a: DVec2, b: DVec2, arc: Option<(DVec2, bool)>) -> (DVec2, DVec2) {
    let Some((c, ccw)) = arc else {
        let l = (b - a).length();
        let l = if l == 0.0 { 1.0 } else { l };
        let d = (b - a) / l;
        return (d, d);
    };
    let along = |p: DVec2| {
        let r = p - c;
        let l = r.length();
        let l = if l == 0.0 { 1.0 } else { l };
        if ccw {
            DVec2::new(-r.y / l, r.x / l)
        } else {
            DVec2::new(r.y / l, -r.x / l)
        }
    };
    (along(a), along(b))
}

/// The shadow's outline as the pieces of a sketch on `frame` — every line, arc and circle it is
/// read to, where it lies, free to be edited, each end held to the next piece's and a corner the
/// shadow rounded held tangent, so the loops stay closed as they are moved. The frame is the
/// sketch's: a shadow is the same outline wherever its plane lies along the normal, so a sketch
/// stood off it starts with it too. Its ids run from `first_id`, clear of ones a sketch handed
/// in names.
pub fn shadow_sketch(sh: &ShadowShape, frame: &SectionFrame, first_id: u32) -> Sketch {
    let mut sketch = Sketch { next_id: first_id, ..Sketch::default() };
    let on_sheet = |p: DVec2| frame.to_sheet(sh.frame.to_world(p));
    for lp in &sh.loops {
        // Each piece's id, the anchor at its loop's start and at its end, and the way it runs
        // there.
        let mut placed: Vec<(u32, AnchorAt, AnchorAt, (DVec2, DVec2))> = Vec::new();
        for p in &lp.pieces {
            let (from, to) = match *p {
                ShadowPiece::Circle { center, radius } => {
                    sketch.add(Entity::new(Shape::Circle { c: on_sheet(center), r: radius }));
                    continue;
                }
                ShadowPiece::Line { from, to } | ShadowPiece::Arc { from, to, .. } => (from, to),
            };
            let a = on_sheet(from);
            let b = on_sheet(to);
            if p.is_line() {
                placed.push((sketch.add(Entity::new(Shape::Line { a, b })), AnchorAt::A, AnchorAt::B, headings(a, b, None)));
                continue;
            }
            let Some((c, r, start, sweep, starts_at_p)) = arc_through(a, b, on_sheet(mid_of(p).unwrap_or(from))) else {
                // An arc too flat to tell from its chord is that chord.
                placed.push((sketch.add(Entity::new(Shape::Line { a, b })), AnchorAt::A, AnchorAt::B, headings(a, b, None)));
                continue;
            };
            let id = sketch.add(Entity::new(Shape::Arc { c, r, start, sweep }));
            // The sketch's arc runs counter-clockwise; one the loop runs the other way starts
            // at the loop's end.
            let (s, e) = if starts_at_p { (AnchorAt::A, AnchorAt::B) } else { (AnchorAt::B, AnchorAt::A) };
            placed.push((id, s, e, headings(a, b, Some((c, starts_at_p)))));
        }
        if placed.len() > 1 {
            for i in 0..placed.len() {
                let p = placed[i];
                let q = placed[(i + 1) % placed.len()];
                sketch.constrain(Rule::Coincident { p: Anchor::new(p.0, p.2), q: Anchor::new(q.0, q.1) });
                let (pd, qd) = (p.3 .1, q.3 .0);
                let turn = (pd.x * qd.y - pd.y * qd.x).atan2(pd.x * qd.x + pd.y * qd.y);
                let curved = |id: u32| sketch.entity(id).is_some_and(|e| e.kind() == ShapeKind::Arc);
                if turn.abs() < TANGENT_RAD && (curved(p.0) || curved(q.0)) {
                    sketch.constrain(Rule::Tangent { a: p.0, b: q.0 });
                }
            }
        }
    }
    sketch
}

/// How many pieces a shadow's outline makes in a sketch.
pub fn shadow_piece_count(sh: &ShadowShape) -> usize {
    sh.loops.iter().map(|l| l.pieces.len()).sum()
}

#[cfg(test)]
mod tests {
    //! The pure parts of ScanRuler's `plugins/reverse/tests/reShadowExtract.test.ts`: why a
    //! section stands as no block, the block a rounded shadow stands as, and the sketch it
    //! starts. The block built through the kernel, and the workspace's store, are the app's.
    use super::*;
    use crate::shadow::{ShadowLoop, Span};

    fn xy() -> SectionFrame {
        SectionFrame { origin: DVec3::ZERO, normal: DVec3::Z, u: DVec3::X, v: DVec3::Y }
    }

    fn p(x: f64, y: f64) -> DVec2 {
        DVec2::new(x, y)
    }

    fn area() -> f64 {
        40.0 * 20.0 - (4.0 - PI) * 9.0 - PI * 16.0
    }

    /// A 40 × 20 outline with its corners rounded to 3 mm, run counter-clockwise, round a Ø8
    /// hole — 10 mm thick, from z = −5 to 5.
    fn rounded_shadow() -> ShadowShape {
        let r = 3.0;
        let q = PI / 2.0;
        let arc = |cx: f64, cy: f64, a0: f64| ShadowPiece::Arc { from: p(cx + r * a0.cos(), cy + r * a0.sin()), to: p(cx + r * (a0 + q).cos(), cy + r * (a0 + q).sin()), center: p(cx, cy), radius: r, sweep: q };
        ShadowShape {
            loops: vec![
                ShadowLoop {
                    hole: false,
                    pieces: vec![
                        ShadowPiece::Line { from: p(-17.0, -10.0), to: p(17.0, -10.0) },
                        arc(17.0, -7.0, -q),
                        ShadowPiece::Line { from: p(20.0, -7.0), to: p(20.0, 7.0) },
                        arc(17.0, 7.0, 0.0),
                        ShadowPiece::Line { from: p(17.0, 10.0), to: p(-17.0, 10.0) },
                        arc(-17.0, 7.0, q),
                        ShadowPiece::Line { from: p(-20.0, 7.0), to: p(-20.0, -7.0) },
                        arc(-17.0, -7.0, 2.0 * q),
                    ],
                },
                ShadowLoop { hole: true, pieces: vec![ShadowPiece::Circle { center: p(0.0, 0.0), radius: 4.0 }] },
            ],
            area: area(),
            play: 1.0,
            tolerance: 0.1,
            smooth_deg: 30.0,
            margin: Span { min: 1.0, max: 1.1 },
            cell: 0.05,
            widened: false,
            extent: Span { min: -5.0, max: 5.0 },
            frame: xy(),
        }
    }

    #[test]
    fn stands_as_its_block_through_the_part_and_says_why_not_for_a_cut_or_one_not_taken_yet() {
        assert!(shadow_primitive(None).is_err_and(|e| e.0.contains("gone")));
        let cut = ShadowSource { name: Some("Plan"), is_shadow: false, shadow: None };
        assert!(shadow_primitive(Some(&cut)).is_err_and(|e| e.0.contains("Plan is a cut")));
        let taking = ShadowSource { name: Some("Outline"), is_shadow: true, shadow: None };
        assert!(shadow_primitive(Some(&taking)).is_err_and(|e| e.0.contains("still being taken")));
        let sh = rounded_shadow();
        let block = shadow_primitive(Some(&ShadowSource { name: Some("Outline"), is_shadow: true, shadow: Some(&sh) })).expect("a block");
        // One body: the outline round the hole, its sides the outline's own pieces — four flat,
        // four rounds — and the hole's wall, standing from −5 to 5 along Z.
        assert_eq!(block.profiles.len(), 1);
        assert_eq!(block.profiles[0].outer.len(), 8);
        assert_eq!(block.profiles[0].outer.iter().filter(|q| matches!(q, LoopPiece::Arc { .. })).count(), 4);
        assert_eq!(block.profiles[0].holes.len(), 1);
        assert!(matches!(block.profiles[0].holes[0][0], LoopPiece::Circle { radius, .. } if radius == 4.0));
        assert_eq!((block.dir, block.from, block.to), (DVec3::Z, -5.0, 5.0));
        // Each round's middle on its arc.
        for q in &block.profiles[0].outer {
            if let LoopPiece::Arc { a, m, b } = *q {
                assert!(((m - a).length() - (m - b).length()).abs() < 1e-9);
            }
        }
        let thin = ShadowShape { extent: Span { min: 0.0, max: 0.0 }, ..sh.clone() };
        assert!(shadow_primitive(Some(&ShadowSource { name: Some("Outline"), is_shadow: true, shadow: Some(&thin) })).is_err_and(|e| e.0 == "Outline has no thickness to stand through."));
        let none = ShadowShape { loops: vec![], ..sh };
        assert!(shadow_primitive(Some(&ShadowSource { name: None, is_shadow: true, shadow: Some(&none) })).is_err_and(|e| e.0 == "The shadow has no outline."));
    }

    #[test]
    fn starts_a_sketch_on_it_with_its_lines_and_arcs_held_closed_and_tangent_a_region_round_the_hole() {
        let sketch = shadow_sketch(&rounded_shadow(), &xy(), 1);
        let mut kinds: Vec<&str> = sketch.entities.iter().map(|e| match e.kind() {
            ShapeKind::Line => "line",
            ShapeKind::Arc => "arc",
            ShapeKind::Circle => "circle",
            ShapeKind::Point => "point",
            ShapeKind::Spline => "spline",
        }).collect();
        kinds.sort();
        assert_eq!(kinds, ["arc", "arc", "arc", "arc", "circle", "line", "line", "line", "line"]);
        assert_eq!(sketch.constraints.iter().filter(|c| matches!(c.rule, Rule::Coincident { .. })).count(), 8);
        assert_eq!(sketch.constraints.iter().filter(|c| matches!(c.rule, Rule::Tangent { .. })).count(), 8);
        // The outline round the hole, and the hole's disc as a region of its own, as any sketch
        // has them; areas read off the drawn polygon.
        let regions = crate::regions::find_regions(&sketch);
        assert_eq!(regions.iter().map(|r| r.holes.len()).collect::<Vec<_>>(), [1, 0]);
        assert!((regions[0].area - area()).abs() < 0.5, "{}", regions[0].area);
        // Each join's ends meet.
        for c in &sketch.constraints {
            if let Rule::Coincident { p, q } = c.rule {
                let (a, b) = (sketch.anchor(p).expect("p"), sketch.anchor(q).expect("q"));
                assert!((a - b).length() < 1e-9);
            }
        }
        assert_eq!(shadow_piece_count(&rounded_shadow()), 9);
        // Its ids run from the one asked.
        let later = shadow_sketch(&rounded_shadow(), &xy(), 40);
        assert_eq!(later.entities[0].id, 40);
    }
}
