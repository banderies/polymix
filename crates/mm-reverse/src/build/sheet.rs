// SPDX-License-Identifier: AGPL-3.0-only
//! The Sheet feature's build: every drawn piece of a sketch — an open chain will do, a closed
//! one too — extruded from `from` to `to` along the sketch plane's normal as a surface, not a
//! solid: a cutter to split a body with, or a skin to thicken. A body of its own in the list,
//! marked as a surface, never joined to or cut from another and never exported alone. Ported
//! from ScanRuler (`plugins/reverse/core/build.ts` the sheet step and sheetOf, `resolve.ts` the
//! sheet's checks, `profiles.ts` sketchPieces, `kernel.ts` sheet), © Stefan Hermann (CNC
//! Kitchen), AGPL-3.0-only, on the SolidCraft kernel. ScanRuler's kernel sweeps the pieces as
//! one wire; SolidCraft's takes them as chains, so the pieces are joined end to end into chains
//! first, each swept on its own into the one sheet.

use std::sync::Arc;

use solidcraft_geom::{Loop2, Seg2, Vec2 as SVec2};
use solidcraft_kernel as kernel;

use super::surf_split::SheetChains;
use super::{absent_sketch, kernel_plane, kerr, BodyState, Builder};
use crate::model::{Feature, Part, SheetFeature};
use crate::sketch::Shape;

pub(crate) fn build(b: &mut Builder, part: &Part, _f: &Feature, x: &SheetFeature) -> Result<(), String> {
    let (Some(sketch), Some(built)) = (part.sketch(x.sketch_id), b.out.sketches.get(&x.sketch_id)) else {
        let said = absent_sketch(part, x.sketch_id, b.step, "sheet");
        return Err(if said.ends_with("is gone.") { "The sketch this sheet is drawn from is gone.".into() } else { said });
    };
    let frame = built.frame;
    let pieces = sketch_pieces(sketch.sketch.entities.iter().filter(|e| !e.construction).map(|e| e.shape.clone()));
    if pieces.is_empty() {
        return Err("The sketch has no pieces to make a sheet of.".into());
    }
    if (x.to - x.from).abs() < 1e-9 {
        return Err("The sheet has no height.".into());
    }
    let chains = chains_of(pieces);
    let plane = kernel_plane(&frame)?;
    let sheet = kernel::extrude_sheet(&plane, &chains, x.from, x.to).map_err(|e| format!("Making the sheet: {}", kerr(e)))?;
    if !sheet.is_surface() {
        return Err("The sheet did not come out as a surface.".into());
    }
    let id = b.fresh_id()?;
    let own = b.own_origins(&sheet);
    let faces = sheet.face_count();
    let mut state = BodyState::new(sheet.clone(), own);
    state.surface = true;
    b.set_body(id, state);
    b.surf.chains.insert(id, Arc::new(SheetChains { made: sheet, plane, chains, from: x.from, to: x.to }));
    b.touched(vec![id]);
    b.report().sheet_faces = Some(faces);
    Ok(())
}

/// A sketch's pieces as the kernel's segments, on its sheet (ScanRuler's `sketchPieces`): every
/// line, arc and circle — a circle as two half arcs. A point is no piece.
pub(crate) fn sketch_pieces(shapes: impl Iterator<Item = Shape>) -> Vec<Seg2> {
    let mut out = Vec::new();
    for s in shapes {
        match s {
            Shape::Line { a, b } => out.push(Seg2::Line { a: SVec2::new(a.x, a.y), b: SVec2::new(b.x, b.y) }),
            Shape::Arc { c, r, start, sweep } => out.push(Seg2::Arc { center: SVec2::new(c.x, c.y), radius: r, start, sweep }),
            Shape::Circle { c, r } => out.extend(Loop2::circle(SVec2::new(c.x, c.y), r).segs),
            Shape::Spline(sp) => {
                let fit = crate::sketch_spline::spline_curve(&sp);
                let bz = crate::sketch_spline::spline_stretch_beziers(&fit, 0.0, crate::sketch_spline::spline_end(&fit));
                let sv = |p: glam::DVec2| SVec2::new(p.x, p.y);
                out.extend(bz.pieces.iter().map(|q| Seg2::Cubic { p0: sv(q[0]), p1: sv(q[1]), p2: sv(q[2]), p3: sv(q[3]) }));
            }
            Shape::Point { .. } => {}
        }
    }
    out
}

/// Pieces joined end to end into chains — each turned to run on from the one before — open
/// ones from a free end, closed ones round. In the order the pieces came where they do not
/// join.
pub(crate) fn chains_of(pieces: Vec<Seg2>) -> Vec<Vec<Seg2>> {
    let size = pieces.iter().flat_map(|s| [s.start(), s.end()]).fold(1.0f64, |m, p| m.max(p.x.abs()).max(p.y.abs()));
    let tol = size * 1e-9 + 1e-9;
    let meets = |a: SVec2, b: SVec2| (a - b).len() <= tol;
    let mut left: Vec<Option<Seg2>> = pieces.into_iter().map(Some).collect();
    let mut chains = Vec::new();
    while let Some(i) = left.iter().position(|s| s.is_some()) {
        let Some(first) = left[i].take() else { break };
        let mut chain = vec![first];
        // On at the end, then back at the start.
        loop {
            let end = chain.last().map(|s| s.end()).unwrap_or(SVec2::new(0.0, 0.0));
            let start = chain.first().map(|s| s.start()).unwrap_or(end);
            if chain.len() > 1 && meets(end, start) {
                break;
            }
            let next = left.iter().position(|s| s.as_ref().is_some_and(|s| meets(s.start(), end) || meets(s.end(), end)));
            let Some(j) = next else { break };
            let Some(s) = left[j].take() else { break };
            chain.push(if meets(s.start(), end) { s } else { s.reversed() });
        }
        loop {
            let start = chain.first().map(|s| s.start()).unwrap_or(SVec2::new(0.0, 0.0));
            let end = chain.last().map(|s| s.end()).unwrap_or(start);
            if chain.len() > 1 && meets(end, start) {
                break;
            }
            let prev = left.iter().position(|s| s.as_ref().is_some_and(|s| meets(s.end(), start) || meets(s.start(), start)));
            let Some(j) = prev else { break };
            let Some(s) = left[j].take() else { break };
            chain.insert(0, if meets(s.end(), start) { s } else { s.reversed() });
        }
        chains.push(chain);
    }
    chains
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build::{build, Built};
    use crate::frame::WorldAxis;
    use crate::model::{FeatureKind, SketchFeature};
    use crate::sketch::Sketch;
    use glam::DVec2;

    fn sheet_of(shapes: &[Shape], from: f64, to: f64) -> (crate::model::Part, Built) {
        let mut part = crate::model::Part::default();
        let mut s = Sketch::default();
        for sh in shapes {
            s.add_shape(sh.clone());
        }
        let sk = part.add(FeatureKind::Sketch(SketchFeature::on_world(WorldAxis::Z, 0.0, s)));
        part.add(FeatureKind::Sheet(SheetFeature { sketch_id: sk, from, to, exprs: Default::default() }));
        let built = build(&part);
        (part, built)
    }

    #[test]
    fn joins_pieces_into_chains_whichever_way_they_were_drawn() {
        let l = |a: (f64, f64), b: (f64, f64)| Seg2::Line { a: SVec2::new(a.0, a.1), b: SVec2::new(b.0, b.1) };
        let chains = chains_of(vec![l((5.0, 0.0), (5.0, 5.0)), l((0.0, 0.0), (5.0, 0.0)), l((0.0, 5.0), (5.0, 5.0)), l((20.0, 0.0), (30.0, 0.0))]);
        assert_eq!(chains.len(), 2);
        assert_eq!(chains[0].len(), 3);
        for w in chains[0].windows(2) {
            assert!((w[0].end() - w[1].start()).len() < 1e-9);
        }
        let closed = chains_of(Loop2::circle(SVec2::new(0.0, 0.0), 2.0).segs);
        assert_eq!(closed.len(), 1);
        assert_eq!(closed[0].len(), 2);
    }

    #[test]
    fn makes_an_open_chain_a_surface_body_of_its_own_marked_a_surface() {
        let (_, built) = sheet_of(&[Shape::Line { a: DVec2::new(-20.0, 0.0), b: DVec2::new(0.0, 0.0) }, Shape::Arc { c: DVec2::new(0.0, 2.0), r: 2.0, start: -std::f64::consts::FRAC_PI_2, sweep: std::f64::consts::FRAC_PI_2 }, Shape::Line { a: DVec2::new(2.0, 2.0), b: DVec2::new(2.0, 20.0) }], 0.0, 20.0);
        assert!(built.errors.is_empty(), "{:?}", built.errors);
        assert_eq!(built.bodies.len(), 1);
        let b = &built.bodies[0];
        assert!(b.surface);
        assert_eq!(b.name, "Shell 1");
        assert_eq!(b.body.face_count(), 3);
        assert_eq!(built.surf.values().next().and_then(|r| r.sheet_faces), Some(3));
    }

    #[test]
    fn refuses_a_sheet_of_no_height_or_of_no_pieces() {
        let (_, built) = sheet_of(&[Shape::Line { a: DVec2::new(0.0, 0.0), b: DVec2::new(10.0, 0.0) }], 5.0, 5.0);
        assert_eq!(built.errors.values().next().map(String::as_str), Some("The sheet has no height."));
        let (_, built) = sheet_of(&[Shape::Point { p: DVec2::new(0.0, 0.0) }], 0.0, 5.0);
        assert_eq!(built.errors.values().next().map(String::as_str), Some("The sketch has no pieces to make a sheet of."));
    }
}
