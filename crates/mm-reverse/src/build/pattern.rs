// SPDX-License-Identifier: AGPL-3.0-only
//! The Pattern feature's build: copies of a body round an axis (circular: the span shared out
//! among the copies, see `pattern_pitch`) or along a direction (linear, and in a grid along a
//! second one), joined with the body — copies standing apart left bodies of their own. Ported
//! from ScanRuler (`plugins/reverse/core/resolve.ts` the pattern step, `build.ts`,
//! `kernel.ts` pattern, `patternCopies.ts`), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! Each copy is the body placed by the kernel, whose history carries every face over from the
//! face it copies: a copy's face keeps the maker of its original, so a pick by the feature that
//! made it finds it on every copy. A sheet's copies are faces with nothing to fuse: they are
//! sewn into the one sheet where they meet and stand apart where they do not — four cutters
//! round a wheel, one body to split with.

use solidcraft_kernel::{self as kernel, BoolOp, Body};

use super::{kerr, origins_after, resolve_axis, rotation_matrix, translation_matrix, BodyState, Builder};
use crate::model::{pattern_pitch, Feature, Part, PatternFeature, PatternMode};

pub(crate) fn build(b: &mut Builder, part: &Part, _f: &Feature, x: &PatternFeature) -> Result<(), String> {
    let gone = "The axis this pattern runs about is gone.";
    let (origin, dir) = resolve_axis(&x.axis, None, part).ok_or(gone)?;
    let dir = dir.try_normalize().ok_or(gone)?;
    let grid = x.mode == PatternMode::Linear && x.axis2.is_some() && x.count2.unwrap_or(1) >= 2;
    let second = match (grid, &x.axis2) {
        (true, Some(a)) => Some(resolve_axis(a, None, part).and_then(|(_, d)| d.try_normalize()).ok_or(gone)?),
        _ => None,
    };
    if x.count < 2 {
        return Err("A pattern needs a count of two or more.".into());
    }
    if !(x.step > 0.0) {
        return Err(if x.mode == PatternMode::Circular { "The pattern needs an angle.".into() } else { "The pattern needs a spacing.".into() });
    }
    if grid && !(x.step2.unwrap_or(0.0) > 0.0) {
        return Err("The second direction needs a spacing.".into());
    }
    if let Some(d2) = second {
        if dir.dot(d2).abs() > 1.0 - 1e-9 {
            return Err("The two directions of the grid run the same way.".into());
        }
    }
    let mut moves = Vec::new();
    match x.mode {
        PatternMode::Circular => {
            let pitch = pattern_pitch(x.step, x.count as f64);
            for i in 1..x.count {
                moves.push(rotation_matrix(origin, dir, (pitch * i as f64).to_radians()));
            }
        }
        PatternMode::Linear => {
            let (n2, step2) = match second {
                Some(_) => (x.count2.unwrap_or(1), x.step2.unwrap_or(0.0)),
                None => (1, 0.0),
            };
            let d2 = second.unwrap_or_default();
            for j in 0..n2 {
                for i in 0..x.count {
                    if i == 0 && j == 0 {
                        continue;
                    }
                    moves.push(translation_matrix(dir * (x.step * i as f64) + d2 * (step2 * j as f64)));
                }
            }
        }
    }
    let (id, state) = b.body_for(x.body)?;
    if state.surface {
        return pattern_sheet(b, id, &state, &moves);
    }
    let mut acc = state.shape.clone();
    let mut acc_origins = state.origins.clone();
    for m in moves {
        let copy = kernel::transform_matrix(&state.shape, m).map_err(kerr)?;
        let copy_origins = origins_after(&copy, &[(&state.shape, &state.origins)], b.step);
        let joined = match kernel::boolean(&acc, &copy, BoolOp::Union) {
            Ok(Some(j)) => j,
            _ => kernel::union_apart(&[acc.clone(), copy.clone()]).map_err(kerr)?.ok_or("The pattern's copies could not be joined.")?,
        };
        acc_origins = origins_after(&joined, &[(&acc, &acc_origins), (&copy, &copy_origins)], b.step);
        acc = joined;
    }
    let ids = b.settle(id, acc, acc_origins, false)?;
    b.touched(ids);
    Ok(())
}

/// A sheet and its copies, sewn into one sheet: a body still, however far apart they stand.
fn pattern_sheet(b: &mut Builder, id: crate::model::BodyId, state: &BodyState, moves: &[[[f64; 4]; 4]]) -> Result<(), String> {
    let mut all: Vec<(Body, Vec<i64>)> = vec![(state.shape.clone(), state.origins.clone())];
    for m in moves {
        let copy = kernel::transform_matrix(&state.shape, *m).map_err(kerr)?;
        let origins = origins_after(&copy, &[(&state.shape, &state.origins)], b.step);
        all.push((copy, origins));
    }
    let shapes: Vec<&Body> = all.iter().map(|(s, _)| s).collect();
    let sewn = kernel::stitch(&shapes, 0.0).map_err(kerr)?;
    let operands: Vec<(&Body, &Vec<i64>)> = all.iter().map(|(s, o)| (s, o)).collect();
    let origins = origins_after(&sewn, &operands, b.step);
    let mut s = BodyState::new(sewn, origins);
    s.surface = true;
    s.label = state.label.clone();
    b.set_body(id, s);
    b.touched(vec![id]);
    Ok(())
}

#[cfg(test)]
mod tests {
    //! The parts of ScanRuler's `plugins/reverse/tests/rePatternStep.test.ts` and
    //! `rePatternCopies.test.ts` the pattern's build answers, on the SolidCraft kernel.
    use glam::{DVec2, DVec3};

    use super::*;
    use crate::build::{build, default_regions, play, volume};
    use crate::frame::WorldAxis;
    use crate::model::*;
    use crate::regions::find_regions;
    use crate::sketch::Sketch;

    fn rect(part: &mut Part, x0: f64, y0: f64, w: f64, h: f64, z: f64) -> FeatureId {
        let mut s = Sketch::default();
        s.add_rectangle(DVec2::new(x0, y0), DVec2::new(x0 + w, y0 + h));
        let regions = default_regions(&find_regions(&s));
        let sk = part.add(FeatureKind::Sketch(SketchFeature::on_world(WorldAxis::Z, 0.0, s)));
        part.add(FeatureKind::Extrude(ExtrudeFeature::new(sk, regions, ExtrudeExtent::Distance { from: z, to: z + 3.0 }, BodyOp::New)))
    }

    fn circular(body: FeatureId, count: u32, step: f64) -> FeatureKind {
        FeatureKind::Pattern(PatternFeature { body: BodyId::new(body), mode: PatternMode::Circular, axis: AxisRef::World { axis: WorldAxis::Z }, count, step, axis2: None, count2: None, step2: None, exprs: Exprs::new() })
    }

    /// The angles about Z the copies' middles stand at, degrees, in order.
    fn angles(part: &Part) -> Vec<f64> {
        let built = build(part);
        let mut out: Vec<f64> = built
            .bodies
            .iter()
            .filter_map(|b| kernel::measure(&b.body).ok())
            .map(|m| {
                let a = m.centroid.y.atan2(m.centroid.x).to_degrees();
                (a * 1000.0).round() / 1000.0
            })
            .collect();
        out.sort_by(|a, b| a.total_cmp(b));
        out
    }

    #[test]
    fn spaces_an_open_span_first_to_last_and_a_full_turn_evenly_round() {
        // A block on +X, patterned in four over 90°: 0, 30, 60, 90; over 360°: 0, 90, 180, 270.
        let mut part = Part::default();
        let e = rect(&mut part, 10.0, -1.0, 4.0, 2.0, 0.0);
        part.add(circular(e, 4, 90.0));
        assert_eq!(angles(&part), vec![0.0, 30.0, 60.0, 90.0]);
        let mut part = Part::default();
        let e = rect(&mut part, 10.0, -1.0, 4.0, 2.0, 0.0);
        part.add(circular(e, 4, 360.0));
        assert_eq!(angles(&part), vec![-90.0, 0.0, 90.0, 180.0]);
    }

    #[test]
    fn copies_keep_the_maker_of_the_faces_they_copy_and_copies_apart_are_bodies_of_their_own() {
        let mut part = Part::default();
        let e = rect(&mut part, 10.0, -1.0, 4.0, 2.0, 0.0);
        let p = part.add(circular(e, 3, 360.0));
        let built = build(&part);
        assert!(built.errors.is_empty(), "{:?}", built.errors);
        assert_eq!(built.bodies.len(), 3, "three blocks apart, a body each");
        assert_eq!(built.bodies[0].id, BodyId::new(e));
        assert!(built.bodies.iter().skip(1).all(|b| b.id.feature == p || b.id.feature == e), "{:?}", built.bodies.iter().map(|b| b.id).collect::<Vec<_>>());
        for b in &built.bodies {
            assert!(b.topology.faces.iter().all(|f| f.fingerprint.origin == e as i64), "{}: every face the extrude's", b.id);
            assert!((volume(&b.body).unwrap_or(0.0) - 24.0).abs() < 1e-6);
        }
        // Copies that overlap are one body: a bar and the bar turned a quarter, a cross.
        let mut part = Part::default();
        let e = rect(&mut part, -1.0, -5.0, 2.0, 10.0, 0.0);
        part.add(circular(e, 2, 90.0));
        let built = build(&part);
        assert!(built.errors.is_empty(), "{:?}", built.errors);
        assert_eq!(built.bodies.len(), 1);
        assert!((volume(&built.bodies[0].body).unwrap_or(0.0) - (2.0 * 60.0 - 12.0)).abs() < 1e-3);
    }

    #[test]
    fn refuses_by_name_what_cannot_be_patterned() {
        let refused = |kind: FeatureKind| {
            let mut part = Part::default();
            rect(&mut part, 0.0, 0.0, 2.0, 2.0, 0.0);
            let p = part.add(kind);
            build(&part).errors.get(&p).cloned()
        };
        let with = |f: &dyn Fn(&mut PatternFeature)| {
            let mut x = PatternFeature { body: BodyId::new(2), mode: PatternMode::Linear, axis: AxisRef::World { axis: WorldAxis::X }, count: 3, step: 5.0, axis2: None, count2: None, step2: None, exprs: Exprs::new() };
            f(&mut x);
            FeatureKind::Pattern(x)
        };
        assert_eq!(refused(with(&|x| x.count = 1)).as_deref(), Some("A pattern needs a count of two or more."));
        assert_eq!(refused(with(&|x| x.step = 0.0)).as_deref(), Some("The pattern needs a spacing."));
        assert_eq!(refused(with(&|x| {
            x.mode = PatternMode::Circular;
            x.step = 0.0;
        }))
        .as_deref(), Some("The pattern needs an angle."));
        assert_eq!(refused(with(&|x| {
            x.axis2 = Some(AxisRef::World { axis: WorldAxis::Y });
            x.count2 = Some(2);
            x.step2 = Some(0.0);
        }))
        .as_deref(), Some("The second direction needs a spacing."));
        assert_eq!(refused(with(&|x| {
            x.axis2 = Some(AxisRef::World { axis: WorldAxis::X });
            x.count2 = Some(2);
            x.step2 = Some(4.0);
        }))
        .as_deref(), Some("The two directions of the grid run the same way."));
        assert_eq!(refused(with(&|x| x.axis = AxisRef::Element { id: 7 })).as_deref(), Some("The axis this pattern runs about is gone."));
        // A second direction with one row is no grid: the row alone.
        assert_eq!(refused(with(&|x| {
            x.axis2 = Some(AxisRef::World { axis: WorldAxis::X });
            x.count2 = Some(1);
        })), None);
    }

    #[test]
    fn a_sheets_copies_stay_one_sheet() {
        // A sheet stood up by the kernel, patterned round Z in four: one sheet of four faces.
        let mut part = Part::default();
        let e = rect(&mut part, 10.0, -1.0, 4.0, 2.0, 0.0);
        let p = part.add(circular(e, 4, 360.0));
        let mut b = play(&part, 2);
        let plane = solidcraft_geom::Plane::new(crate::build::sv3(DVec3::new(10.0, 0.0, 0.0)), crate::build::sv3(DVec3::Y), crate::build::sv3(DVec3::Z)).unwrap_or(solidcraft_geom::Plane::XY);
        let chain = vec![solidcraft_geom::Seg2::Line { a: solidcraft_geom::Vec2::new(-1.0, 0.0), b: solidcraft_geom::Vec2::new(1.0, 0.0) }];
        let sheet = kernel::extrude_sheet(&plane, &[chain], 0.0, 4.0).unwrap_or_else(|e| panic!("{e}"));
        let n = sheet.face_count();
        let mut s = BodyState::new(sheet, vec![e as i64; n]);
        s.surface = true;
        b.set_body(BodyId::new(e), s);
        b.step = p;
        let Some(FeatureKind::Pattern(x)) = part.feature(p).map(|f| f.kind.clone()) else { panic!("pattern") };
        let f = part.feature(p).cloned().unwrap_or_else(|| panic!("pattern"));
        build_pattern(&mut b, &part, &f, &x);
        let sheet = b.body(BodyId::new(e)).unwrap_or_else(|| panic!("the sheet"));
        assert!(sheet.surface);
        assert_eq!(sheet.shape.face_count(), 4 * n);
        assert!(sheet.origins.iter().all(|o| *o == e as i64));
        assert_eq!(b.body_ids().len(), 1);
    }

    fn build_pattern(b: &mut Builder, part: &Part, f: &Feature, x: &PatternFeature) {
        if let Err(e) = super::build(b, part, f, x) {
            panic!("{e}");
        }
    }
}
