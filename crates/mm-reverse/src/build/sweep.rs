// SPDX-License-Identifier: AGPL-3.0-only
//! The Sweep feature's build: one sketch's largest region carried along another sketch's one
//! chain of lines and arcs, or along a route of straights and bends in 3D, the section held
//! square to the path and turning as little as it can. The profile's plane stands square to the
//! path where it starts. Ported from ScanRuler (`plugins/reverse/core/` `resolve.ts` the sweep
//! step, `build.ts` sweepPlaced and sweepOf, `profiles.ts` sweepPath, sweepGeometry,
//! routeSweepGeometry, `kernel.ts` sweep, `sweepRoute.ts`), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only, on the SolidCraft kernel.
//!
//! The kernel sweeps a region along lines and arcs end to end, a line carrying the section on
//! and an arc turning it about the arc's axis — square to the path all the way, with the least
//! turn there is, as Open CASCADE's "fixed" sweep does; the holes go with it.

use glam::{DVec2, DVec3};
use solidcraft_kernel::{self as kernel, PathSeg};

use super::loft::{absent, lift_region, loft_profile, placed_sketch, profile_middle, PlacedSketch};
use super::{kernel_plane, kerr, sv3, Builder};
use crate::js_fmt::{num, round_to, to_fixed};
use crate::loop_piece::{circle_through, piece_points, LoopPiece, Profile, PIECE_STEP};
use crate::model::{Feature, Part, SweepFeature};
use crate::regions::Region;
use crate::sketch::{anchor_point, arc_point, AnchorAt, Shape, TOUCH_MM};
use crate::sweep_route::{route_pieces, SweepRoute};

/// How far off square to the path where it starts a sweep's profile may stand, degrees.
const SWEEP_SQUARE_DEG: f64 = 1.0;

/// How far off a profile's plane a route may start, mm.
const ROUTE_ON_PLANE_MM: f64 = 0.05;

pub(crate) fn build(b: &mut Builder, part: &Part, _f: &Feature, x: &SweepFeature) -> Result<(), String> {
    let profile_sketch = placed_sketch(b, part, x.profile);
    let (region, frame, path) = match &x.route {
        Some(route) => {
            let Some(p) = profile_sketch else { return Err(absent(part, x.profile, b.step, "sweep", "The sketch of the profile is gone.")) };
            let route = SweepRoute { points: route.points.clone(), bend_radius: route.bend_radius, radii: route.radii.clone() };
            let (region, path) = route_sweep_geometry(&p, &route)?;
            (region, p.frame, path)
        }
        None => {
            if Some(x.profile) == x.path {
                return Err("The profile and the path are one sketch — draw them in two.".into());
            }
            let Some(p) = profile_sketch else { return Err(absent(part, x.profile, b.step, "sweep", "The sketch of the profile is gone.")) };
            let Some(along) = x.path.and_then(|id| placed_sketch(b, part, id)) else {
                return Err(absent(part, x.path.unwrap_or(x.profile), b.step, "sweep", "The sketch of the path is gone."));
            };
            let (region, path) = sweep_geometry(&p, &along)?;
            (region, p.frame, path)
        }
    };
    let segs = path_segs(&path)?;
    let plane = kernel_plane(&frame)?;
    let solid = kernel::sweep(&plane, &region.region, &segs).map_err(kerr)?;
    b.place(x.op, x.target, vec![solid])?;
    Ok(())
}

/// A path's pieces as the kernel sweeps along them: a line as it is, an arc by its centre, its
/// axis and the angle it turns through from its start.
fn path_segs(path: &[LoopPiece]) -> Result<Vec<PathSeg>, String> {
    let mut out = Vec::new();
    for p in path {
        match *p {
            LoopPiece::Line { a, b } => out.push(PathSeg::Line { a: sv3(a), b: sv3(b) }),
            LoopPiece::Arc { a, m, b } => {
                let Some((center, _, axis)) = circle_through(a, m, b) else {
                    out.push(PathSeg::Line { a: sv3(a), b: sv3(b) });
                    continue;
                };
                // a → m → b runs counter-clockwise about the axis: b's angle about it is the turn.
                let u = (a - center).normalize_or_zero();
                let v = axis.cross(u);
                let mut angle = (b - center).dot(v).atan2((b - center).dot(u));
                if angle <= 0.0 {
                    angle += std::f64::consts::TAU;
                }
                out.push(PathSeg::Arc { a: sv3(a), center: sv3(center), axis: sv3(axis), angle });
            }
            LoopPiece::Circle { .. } => return Err("A sweep's path is lines and arcs end to end; a circle has no end to start from.".into()),
        }
    }
    if out.is_empty() {
        return Err("The sweep has no path to follow.".into());
    }
    Ok(out)
}

/// A sweep's path off a sketch: its pieces in the part, run from the end nearest the profile,
/// the point it starts at and the way it leaves.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SweepPath {
    pub(crate) pieces: Vec<LoopPiece>,
    pub(crate) start: DVec3,
    pub(crate) tangent: DVec3,
}

/// A sweep's path: the one chain of a sketch's lines and arcs, end to end, lifted into the part
/// and run from the end nearest `from` — a closed chain from the end nearest it — with the
/// point it starts at and the way it leaves; or why there is none: no piece, a circle (it has
/// no end to start from), a branch, or more than one chain (ScanRuler's `sweepPath`).
pub(crate) fn sweep_path(sketch: &PlacedSketch, from: DVec3) -> Result<SweepPath, String> {
    let name = &sketch.name;
    let own: Vec<_> = sketch.sketch.entities.iter().filter(|e| !e.construction && !matches!(e.shape, Shape::Point { .. })).collect();
    if own.iter().any(|e| matches!(e.shape, Shape::Circle { .. })) {
        return Err(format!("{name} has a circle — a sweep's path is lines, arcs and splines end to end; draw a closed path as arcs."));
    }
    let pieces: Vec<_> = own.into_iter().filter(|e| matches!(e.shape, Shape::Line { .. } | Shape::Arc { .. })).collect();
    if pieces.is_empty() {
        return Err(format!("{name} has no line, arc or spline for the profile to follow."));
    }
    let frame = sketch.frame;
    let lift = |p: DVec2| frame.to_world(p);
    #[derive(Clone, Copy)]
    struct End {
        piece: usize,
        end: AnchorAt,
        at: DVec2,
    }
    let ends: Vec<End> = pieces.iter().enumerate().flat_map(|(i, e)| [AnchorAt::A, AnchorAt::B].map(|end| End { piece: i, end, at: anchor_point(&e.shape, end) })).collect();
    let meeting = |p: &End| -> Vec<End> { ends.iter().filter(|q| q.piece != p.piece && (q.at - p.at).length() <= TOUCH_MM).copied().collect() };
    if ends.iter().any(|p| meeting(p).len() > 1) {
        return Err(format!("{name} branches — three pieces meet at a point; a sweep follows one chain."));
    }
    let loose: Vec<End> = ends.iter().filter(|p| meeting(p).is_empty()).copied().collect();
    let far = |p: &End| (lift(p.at) - from).length();
    let candidates = if loose.is_empty() { &ends } else { &loose };
    let Some(first) = candidates.iter().copied().reduce(|a, b| if far(&b) < far(&a) { b } else { a }) else {
        return Err(format!("{name} has no line, arc or spline for the profile to follow."));
    };
    let mut out = Vec::new();
    let mut used = vec![false; pieces.len()];
    let mut at = Some(first);
    while let Some(here) = at {
        if used[here.piece] {
            break;
        }
        used[here.piece] = true;
        let e = pieces[here.piece];
        let other = if here.end == AnchorAt::A { AnchorAt::B } else { AnchorAt::A };
        match e.shape {
            Shape::Line { .. } => out.push(LoopPiece::Line { a: lift(anchor_point(&e.shape, here.end)), b: lift(anchor_point(&e.shape, other)) }),
            Shape::Arc { c, r, start, sweep } => out.push(LoopPiece::Arc { a: lift(anchor_point(&e.shape, here.end)), m: lift(arc_point(c, r, start + sweep / 2.0)), b: lift(anchor_point(&e.shape, other)) }),
            _ => {}
        }
        at = ends.iter().find(|x| x.piece == here.piece && x.end == other).and_then(|x| meeting(x).first().copied());
    }
    let count = used.iter().filter(|u| **u).count();
    if count < pieces.len() {
        let chains = if loose.len() > 2 { num((loose.len() / 2) as f64) } else { "more than one".to_string() };
        return Err(format!("{name} is in {chains} chains — a sweep follows one."));
    }
    // The way the path leaves its start, in the sheet and then in the part.
    let e = pieces[first.piece];
    let d = match e.shape {
        Shape::Line { a, b } => {
            if first.end == AnchorAt::A {
                b - a
            } else {
                a - b
            }
        }
        Shape::Arc { start, sweep, .. } => {
            let angle = if first.end == AnchorAt::A { start } else { start + sweep };
            let turn = (if first.end == AnchorAt::A { 1.0 } else { -1.0 }) * sweep.signum();
            DVec2::new(-angle.sin() * turn, angle.cos() * turn)
        }
        _ => DVec2::ZERO,
    };
    let Some(tangent) = (frame.u * d.x + frame.v * d.y).try_normalize() else {
        return Err(format!("{name}'s path starts with a piece of no length."));
    };
    Ok(SweepPath { pieces: out, start: lift(first.at), tangent })
}

/// Where a sweep's path starts and which way it leaves, and the plane to sketch its profile on:
/// a coordinate plane where the path leaves along an axis, said whole; else the point and the
/// way (ScanRuler's `sweepStartWords`).
pub(crate) fn sweep_start_words(path: &str, start: DVec3, tangent: DVec3) -> String {
    let r = |x: f64| num(round_to(x, 3));
    let at = format!("({}, {}, {})", r(start.x), r(start.y), r(start.z));
    let way = format!("({}, {}, {})", r(tangent.x), r(tangent.y), r(tangent.z));
    let k = [0usize, 1, 2].into_iter().reduce(|a, b| if tangent[b].abs() > tangent[a].abs() { b } else { a }).unwrap_or(0);
    let axis = ["x", "y", "z"][k];
    let along = tangent[k].abs() >= SWEEP_SQUARE_DEG.to_radians().cos();
    let plane = if along {
        format!("the plane {{ kind: \"world\", axis: \"{axis}\", offset: {} }} is square to it there", r(start[k]))
    } else {
        "no coordinate plane is square to it there: sketch on a measured plane with that normal through that point, or start the path along a coordinate axis".to_string()
    };
    format!("{path} starts at {at} and leaves along {way}; sketch the profile on a plane through that point square to that way — {plane}.")
}

/// How far a profile's plane stands off square to a path leaving along `tangent`, degrees.
fn off_square(normal: DVec3, tangent: DVec3) -> f64 {
    tangent.dot(normal).abs().min(1.0).acos().to_degrees()
}

/// What a sweep sweeps: the profile sketch's largest region, and the path sketch's one chain run
/// from the end nearest the profile — held to the profile's plane standing square to the path
/// where it starts, within SWEEP_SQUARE_DEG; or why not, the angle said (ScanRuler's
/// `sweepGeometry`).
pub(crate) fn sweep_geometry(profile_sketch: &PlacedSketch, path_sketch: &PlacedSketch) -> Result<(Region, Vec<LoopPiece>), String> {
    let region = loft_profile(&profile_sketch.name, profile_sketch.sketch, &profile_sketch.regions, "the sweep")?.clone();
    let profile = lift_region(&profile_sketch.frame, &region);
    let from = profile_middle(&profile);
    let path = sweep_path(path_sketch, from)?;
    let off = off_square(profile_sketch.frame.normal, path.tangent);
    if off > SWEEP_SQUARE_DEG {
        return Err(format!("{}'s plane stands {}° off square to the path where it starts — {}", profile_sketch.name, to_fixed(off, 1), sweep_start_words(&path_sketch.name, path.start, path.tangent)));
    }
    Ok((region, path.pieces))
}

/// What a sweep along a route sweeps: the profile sketch's largest region, and the route's
/// straights and bends run from the end nearest the profile — held as `sweep_geometry` holds a
/// path, and the start has to lie on the profile: in its plane within ROUTE_ON_PLANE_MM and
/// inside its reach. The route comes from the scan and the profile is sketched for it, so a miss
/// is a mistake here; it would sweep a tube beside the one measured with no other word
/// (ScanRuler's `routeSweepGeometry`).
pub(crate) fn route_sweep_geometry(profile_sketch: &PlacedSketch, route: &SweepRoute) -> Result<(Region, Vec<LoopPiece>), String> {
    let region = loft_profile(&profile_sketch.name, profile_sketch.sketch, &profile_sketch.regions, "the sweep")?.clone();
    let profile: Profile = lift_region(&profile_sketch.frame, &region);
    let from = profile_middle(&profile);
    let path = route_pieces(route, from).map_err(|e| e.0)?;
    let (frame, name) = (profile_sketch.frame, &profile_sketch.name);
    let off = off_square(frame.normal, path.tangent);
    if off > SWEEP_SQUARE_DEG {
        return Err(format!("{name}'s plane stands {}° off square to the route where it starts — {}", to_fixed(off, 1), sweep_start_words("The route", path.start, path.tangent)));
    }
    let aside = (path.start - frame.origin).dot(frame.normal).abs();
    if aside > ROUTE_ON_PLANE_MM {
        return Err(format!("{name}'s plane lies {} mm off the route's start — {}", to_fixed(aside, 3), sweep_start_words("The route", path.start, path.tangent)));
    }
    let outline: Vec<DVec3> = profile.outer.iter().flat_map(|p| piece_points(p, PIECE_STEP)).collect();
    let reach = outline.iter().map(|q| (*q - from).length()).fold(f64::NEG_INFINITY, f64::max);
    let apart = (path.start - from).length();
    if apart > reach {
        return Err(format!("The route starts {} mm from the middle of {name}'s profile, which reaches {} mm — {}", to_fixed(apart, 3), to_fixed(reach, 3), sweep_start_words("The route", path.start, path.tangent)));
    }
    // A bend tighter than the section's nearest wall folds whichever way the section lies: the
    // kernel would give a closed solid of it with no word.
    let near = outline.iter().map(|q| (*q - path.start).length()).fold(f64::INFINITY, f64::min);
    if let Some(tight) = path.bends.iter().find(|b| b.radius <= near) {
        return Err(format!(
            "The bend at points[{}] of radius {} mm is tighter than {name}'s profile, every side of it at least {} mm from the route: the inside of the bend would fold through itself.",
            tight.index,
            num(tight.radius),
            to_fixed(near, 3)
        ));
    }
    Ok((region, path.pieces))
}

#[cfg(test)]
mod tests {
    //! The parts of ScanRuler's `plugins/reverse/tests/reLoftChecks.test.ts` (a sweep refused for
    //! where it starts) and `reSweepRoute.test.ts` (a sweep along a route) a sweep's own build
    //! answers, on the SolidCraft kernel.
    use super::*;
    use crate::build::{build, play, volume, Built};
    use crate::frame::WorldAxis;
    use crate::model::{BodyOp, FeatureId, FeatureKind, SketchFeature, SweepRoute as RouteOf};
    use crate::sketch::Sketch;

    fn sketch_on(part: &mut Part, name: &str, axis: WorldAxis, offset: f64, draw: impl FnOnce(&mut Sketch)) -> FeatureId {
        let mut s = Sketch::default();
        draw(&mut s);
        part.add_named(FeatureKind::Sketch(SketchFeature::on_world(axis, offset, s)), name.into())
    }

    fn placed(part: &Part, id: FeatureId) -> Option<PlacedSketch<'_>> {
        let b = play(part, part.features.len());
        placed_sketch(&b, part, id).map(|p| PlacedSketch { name: p.name, frame: p.frame, sketch: p.sketch, regions: p.regions })
    }

    fn sweep(part: &mut Part, profile: FeatureId, path: Option<FeatureId>, route: Option<RouteOf>) -> FeatureId {
        part.add(FeatureKind::Sweep(SweepFeature { profile, path, route, op: BodyOp::New, target: None, exprs: Default::default() }))
    }

    fn round(part: &mut Part) -> FeatureId {
        sketch_on(part, "Round", WorldAxis::Z, 0.0, |s| {
            s.add_shape(Shape::Circle { c: DVec2::ZERO, r: 4.0 });
        })
    }

    fn result(built: &Built, id: FeatureId) -> Result<f64, String> {
        match built.errors.get(&id) {
            Some(e) => Err(e.clone()),
            None => Ok(built.bodies.iter().filter(|b| b.id.feature == id).filter_map(|b| volume(&b.body)).sum()),
        }
    }

    #[test]
    fn a_sweep_refused_for_where_it_starts_gives_the_paths_start_and_way_and_the_plane_square_to_it() {
        let mut part = Part::default();
        let profile = sketch_on(&mut part, "Ring section", WorldAxis::Z, 0.0, |s| {
            s.add_rectangle(DVec2::splat(-2.0), DVec2::splat(2.0));
        });
        let along = sketch_on(&mut part, "Path", WorldAxis::Z, 0.0, |s| {
            s.add_shape(Shape::Line { a: DVec2::ZERO, b: DVec2::new(10.0, 0.0) });
        });
        let slant = sketch_on(&mut part, "Path", WorldAxis::Z, 0.0, |s| {
            s.add_shape(Shape::Line { a: DVec2::ZERO, b: DVec2::new(10.0, 5.0) });
        });
        let (Some(p), Some(a), Some(sl)) = (placed(&part, profile), placed(&part, along), placed(&part, slant)) else { panic!("sketches") };
        assert_eq!(
            sweep_geometry(&p, &a).err().as_deref(),
            Some("Ring section's plane stands 90.0° off square to the path where it starts — Path starts at (0, 0, 0) and leaves along (1, 0, 0); sketch the profile on a plane through that point square to that way — the plane { kind: \"world\", axis: \"x\", offset: 0 } is square to it there.")
        );
        let said = sweep_geometry(&p, &sl).err().unwrap_or_default();
        assert!(said.contains("leaves along (0.894, 0.447, 0); ") && said.contains("no coordinate plane is square to it there"), "{said}");
    }

    #[test]
    fn sweeps_a_ring_along_a_path_of_a_line_and_an_arc_from_the_end_nearest_the_profile() {
        // A Ø8 profile on XY at z 0; the path on the XZ plane: up 20 from the origin, a quarter
        // round of radius 10, drawn from its far end first.
        let mut part = Part::default();
        let profile = round(&mut part);
        let path = sketch_on(&mut part, "Path", WorldAxis::Y, 0.0, |_| {});
        let Some(frame) = placed(&part, path).map(|p| p.frame) else { panic!("path") };
        let sheet = |p: DVec3| frame.to_sheet(p);
        if let Some(FeatureKind::Sketch(s)) = part.feature_mut(path).map(|f| &mut f.kind) {
            let (o, top, c) = (sheet(DVec3::ZERO), sheet(DVec3::new(0.0, 0.0, 20.0)), sheet(DVec3::new(10.0, 0.0, 20.0)));
            let end = sheet(DVec3::new(10.0, 0.0, 30.0));
            let start = (top - c).y.atan2((top - c).x);
            let stop = (end - c).y.atan2((end - c).x);
            let mut sweep_rad = stop - start;
            if sweep_rad.abs() > std::f64::consts::PI {
                sweep_rad -= sweep_rad.signum() * std::f64::consts::TAU;
            }
            let (st, sw) = if sweep_rad < 0.0 { (stop, -sweep_rad) } else { (start, sweep_rad) };
            s.sketch.add_shape(Shape::Arc { c, r: 10.0, start: st, sweep: sw });
            s.sketch.add_shape(Shape::Line { a: top, b: o });
        }
        let id = sweep(&mut part, profile, Some(path), None);
        let built = build(&part);
        let v = result(&built, id);
        let want = std::f64::consts::PI * 16.0 * (20.0 + std::f64::consts::FRAC_PI_2 * 10.0);
        assert!(v.as_ref().is_ok_and(|v| (v - want).abs() < 0.003 * want), "{v:?} against {want}");
    }

    #[test]
    fn refuses_a_path_that_branches_a_circle_and_one_sketch_for_both() {
        let mut part = Part::default();
        let profile = round(&mut part);
        let branch = sketch_on(&mut part, "Branch", WorldAxis::Y, 0.0, |s| {
            s.add_shape(Shape::Line { a: DVec2::ZERO, b: DVec2::new(0.0, -10.0) });
            s.add_shape(Shape::Line { a: DVec2::new(0.0, -10.0), b: DVec2::new(5.0, -20.0) });
            s.add_shape(Shape::Line { a: DVec2::new(0.0, -10.0), b: DVec2::new(-5.0, -20.0) });
        });
        let ring = sketch_on(&mut part, "Ring", WorldAxis::Y, 0.0, |s| {
            s.add_shape(Shape::Circle { c: DVec2::new(10.0, 0.0), r: 3.0 });
        });
        let two = sketch_on(&mut part, "Two", WorldAxis::Y, 0.0, |s| {
            s.add_shape(Shape::Line { a: DVec2::ZERO, b: DVec2::new(0.0, -10.0) });
            s.add_shape(Shape::Line { a: DVec2::new(5.0, -12.0), b: DVec2::new(5.0, -20.0) });
        });
        let a = sweep(&mut part, profile, Some(branch), None);
        let b = sweep(&mut part, profile, Some(ring), None);
        let c = sweep(&mut part, profile, Some(two), None);
        let d = sweep(&mut part, profile, Some(profile), None);
        let built = build(&part);
        assert_eq!(built.errors.get(&a).map(String::as_str), Some("Branch branches — three pieces meet at a point; a sweep follows one chain."));
        assert_eq!(built.errors.get(&b).map(String::as_str), Some("Ring has a circle — a sweep's path is lines, arcs and splines end to end; draw a closed path as arcs."));
        assert_eq!(built.errors.get(&c).map(String::as_str), Some("Two is in 2 chains — a sweep follows one."));
        assert_eq!(built.errors.get(&d).map(String::as_str), Some("The profile and the path are one sketch — draw them in two."));
    }

    #[test]
    fn names_a_path_sketch_further_down_the_list_rather_than_calling_it_gone() {
        // ScanRuler's "a sketch the feature cannot have": a sweep above its path's sketch.
        let mut part = Part::default();
        let profile = round(&mut part);
        let s = sweep(&mut part, profile, Some(3), None);
        let path = sketch_on(&mut part, "Top", WorldAxis::Y, 0.0, |s| {
            s.add_shape(Shape::Line { a: DVec2::ZERO, b: DVec2::new(0.0, 10.0) });
        });
        assert_eq!(path, 3);
        let said = build(&part).errors.get(&s).cloned().unwrap_or_default();
        assert!(said.starts_with("Sketch 3 (Top) stands after this sweep — a feature takes only the sketches above it."), "{said}");
        let mut part = Part::default();
        let profile = round(&mut part);
        let s = sweep(&mut part, profile, Some(9), None);
        assert_eq!(build(&part).errors.get(&s).map(String::as_str), Some("The sketch of the path is gone."));
    }

    /// A centreline in three bend planes, each corner a square turn.
    const THREE_PLANES: [[f64; 3]; 5] = [[0.0, 0.0, 0.0], [0.0, 0.0, 50.0], [50.0, 0.0, 50.0], [50.0, 50.0, 50.0], [50.0, 50.0, 100.0]];

    /// Its length at a bend radius: every corner gives up 2R and takes a quarter round back.
    fn length_at(r: f64) -> f64 {
        200.0 - 3.0 * 2.0 * r + 3.0 * std::f64::consts::FRAC_PI_2 * r
    }

    fn route(points: &[[f64; 3]], bend_radius: f64) -> RouteOf {
        RouteOf { points: points.iter().map(|p| DVec3::from(*p)).collect(), bend_radius, radii: None }
    }

    #[test]
    fn builds_a_round_tube_in_three_bend_planes_to_its_length() {
        let mut part = Part::default();
        let profile = round(&mut part);
        let id = sweep(&mut part, profile, None, Some(route(&THREE_PLANES, 10.0)));
        let built = build(&part);
        let want = std::f64::consts::PI * 16.0 * length_at(10.0);
        let v = result(&built, id);
        assert!(v.as_ref().is_ok_and(|v| (v - want).abs() / want < 0.003), "{v:?} against {want}");
        let bb = built.bodies[0].body.outline_bounds();
        for (got, want) in [(bb.min.x, -4.0), (bb.min.y, -4.0), (bb.min.z, 0.0), (bb.max.x, 54.0), (bb.max.y, 54.0), (bb.max.z, 100.0)] {
            assert!((got - want).abs() < 0.01, "{bb:?}");
        }
        let step = crate::build::export_step(&[("Tube", &built.bodies[0].body)]).unwrap_or_default();
        assert!(step.starts_with("ISO-10303-21;"));
    }

    #[test]
    fn holds_the_profile_to_the_start_square_in_its_plane_and_on_it() {
        let cases: [(&[[f64; 3]], f64, &str); 6] = [
            (&[[0.0, 0.0, 0.0], [40.0, 0.0, 0.0]], 10.0, "90.0° off square to the route"),
            (&[[0.0, 0.0, 1.0], [0.0, 0.0, 40.0]], 10.0, "1.000 mm off the route"),
            (&[[9.0, 0.0, 0.0], [9.0, 0.0, 40.0]], 10.0, "The route starts 9.000 mm from the middle"),
            (&[[0.0, 0.0, 0.0], [0.0, 0.0, 5.0], [30.0, 0.0, 5.0]], 10.0, "takes a radius of 5 mm at most"),
            (&[[0.0, 0.0, 0.0], [0.0, 0.0, 20.0], [30.0, 0.0, 20.0]], 3.0, "would fold through itself"),
            (&[[0.0, 0.0, 0.0], [0.0, 0.0, 20.0], [30.0, 0.0, 20.0]], 10.0, ""),
        ];
        for (points, r, says) in cases {
            let mut part = Part::default();
            let profile = round(&mut part);
            let id = sweep(&mut part, profile, None, Some(route(points, r)));
            let built = build(&part);
            match (built.errors.get(&id), says) {
                (None, "") => {}
                (Some(e), s) if !s.is_empty() && e.contains(s) => {
                    if s.starts_with("90.0") {
                        assert!(e.contains("{ kind: \"world\", axis: \"x\", offset: 0 }"), "{e}");
                    }
                }
                (got, s) => panic!("{points:?} at {r}: {got:?}, wanted {s:?}"),
            }
        }
    }
}
