// SPDX-License-Identifier: AGPL-3.0-only
//! The Revolve feature's build: a sketch's picked regions turned about an axis — a line of the
//! sketch, a coordinate axis, or a line, cylinder, cone or circle element's axis — joined, cut or
//! kept a body of its own; with a pitch, screwed about the axis, climbing the pitch every turn:
//! a coil, a spring, a thread, right- or left-handed, of as many turns as the angle holds.
//! Ported from ScanRuler (`plugins/reverse/core/resolve.ts` the revolve step, `build.ts`
//! solidOf and coilOf, `kernel.ts` screw), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! On the SolidCraft kernel a plain revolve turns about an axis in the sketch's plane (one
//! beside the plane is refused by name), and a coil is the kernel's helical sweep of each
//! profile's outline, its holes swept the same way and cut out, as ScanRuler's screw does —
//! 200 turns at most, the kernel's limit, where ScanRuler takes a thousand.

use glam::{DVec2, DVec3};
use solidcraft_geom::Plane;
use solidcraft_kernel::{self as kernel, BoolOp, Body};

use super::{apply_matrix, kerr, kernel_plane, resolve_axis, rotation_matrix, sv2, sv3, volume, Builder};
use crate::extrude_draft::axis_chosen;
use crate::frame::SectionFrame;
use crate::js_number::{js_round, js_string, to_fixed, to_precision_number};
use crate::model::{revolve_max_deg, AxisRef, BodyOp, Feature, Part, RevolveFeature, FULL_TURN_DEG};
use crate::profile_axis::{lift_profiles, profile_across_axis, profile_span, profiles_meet_axis};
use crate::sketch::Shape;

/// The most turns the kernel winds a coil.
pub const KERNEL_MAX_TURNS: f64 = 200.0;

/// How far a hole's sweep runs on past either end of its coil, degrees: the cut goes through
/// the coil's end faces instead of lying on them, which the kernel's boolean is slow to read.
const HOLE_OVERRUN_DEG: f64 = 2.0;

pub(crate) fn build(b: &mut Builder, part: &Part, _f: &Feature, r: &RevolveFeature) -> Result<(), String> {
    let (frame, regions) = b.sketch_profiles(part, r.sketch_id, &r.regions, r.region_pick.as_ref(), "revolve")?;
    if !axis_chosen(&r.axis) {
        return Err("Choose the axis the revolve turns about.".into());
    }
    let (origin, dir) = axis_of(part, r, &frame).ok_or("The axis this revolve turns about is gone.")?;
    let pitch = r.pitch.filter(|p| p.is_finite()).unwrap_or(0.0);
    if !(r.angle_deg > 0.0) {
        return Err("The revolve needs an angle.".into());
    }
    if pitch == 0.0 && r.angle_deg > FULL_TURN_DEG + 1e-9 {
        return Err(format!("A revolve turns {FULL_TURN_DEG}° at most — give it a pitch to coil on past a full turn."));
    }
    let angle = r.angle_deg.min(revolve_max_deg(Some(pitch))).to_radians();
    let tools = if pitch != 0.0 { coil(&frame, &regions, origin, dir, angle, pitch, r.left_handed)? } else { plain(&frame, &regions, origin, dir, angle, r.op)? };
    let made: f64 = tools.iter().filter_map(volume).sum();
    b.place(r.op, r.target, tools)?;
    // ---- EXT ----: the ghost's volume, for the box's preview line.
    b.out.ghost_volumes.insert(b.step, made);
    Ok(())
}

/// The axis a revolve turns about, in the part: a line of its sketch read off the sketch as
/// placed, or the axis the reference names.
pub(crate) fn axis_of(part: &Part, r: &RevolveFeature, frame: &SectionFrame) -> Option<(DVec3, DVec3)> {
    match r.axis {
        AxisRef::SketchLine { entity } => {
            let s = part.sketch(r.sketch_id)?;
            match s.sketch.entity(entity).map(|e| e.shape.clone()) {
                Some(Shape::Line { a, b }) => {
                    let (pa, pb) = (frame.to_world(a), frame.to_world(b));
                    Some((pa, (pb - pa).try_normalize()?))
                }
                _ => None,
            }
        }
        other => {
            let (o, d) = resolve_axis(&other, None, part)?;
            Some((o, d.try_normalize()?))
        }
    }
}

/// A point as a refusal names it: each coordinate to a thousandth, as JavaScript writes it.
fn at_words(p: DVec3) -> String {
    let c = |v: f64| js_string(js_round(v * 1000.0) / 1000.0);
    format!("({}, {}, {})", c(p.x), c(p.y), c(p.z))
}

/// A plain revolve: each region turned about an axis in its plane. A profile across the axis is
/// named before the kernel's failure; one ending on the axis (a pole) or running along it is
/// turned.
fn plain(frame: &SectionFrame, regions: &[solidcraft_geom::Region2], origin: DVec3, dir: DVec3, angle: f64, op: BodyOp) -> Result<Vec<Body>, String> {
    let lifted = lift_profiles(frame, regions);
    if let Some((across, at)) = profile_across_axis(&lifted, origin, dir, 1e-6) {
        return Err(if across > 0.0 {
            format!(
                "The profile crosses the revolve’s axis — it reaches {} mm over it at {}. A revolve turns what lies on one side of its axis: end the pieces on the axis (a profile may end on it, as a sphere’s arc does, or run along it), or pick only the region on one side.",
                js_string(to_precision_number(across, 3)),
                at_words(at)
            )
        } else {
            format!("The revolve’s axis runs through the profile at {} — a revolve turns a profile beside its axis, or one with the axis on its edge.", at_words(at))
        });
    }
    if dir.dot(frame.normal).abs() > 1e-6 || (origin - frame.origin).dot(frame.normal).abs() > 1e-6 {
        return Err("The axis does not lie in the sketch's plane — a revolve here turns a profile about an axis in its own plane. Sketch the profile on a plane through the axis, or pick an axis that lies in it.".into());
    }
    let plane = kernel_plane(frame)?;
    let o = frame.to_sheet(origin);
    let d = DVec2::new(dir.dot(frame.u), dir.dot(frame.v));
    if d.length() < 1e-9 {
        return Err("The axis this revolve turns about has no direction.".into());
    }
    if op == BodyOp::New { kernel::revolve_whole(&plane, regions, sv2(o), sv2(d), angle) } else { kernel::revolve(&plane, regions, sv2(o), sv2(d), angle) }.map_err(kerr)
}

/// A revolve with a pitch: each profile screwed about the axis, climbing the pitch every turn,
/// for as many turns as the angle holds — a coil, a thread. Turned a whole turn or more, a
/// profile reaching as far along the axis as the pitch would run into itself a turn on, and a
/// profile on the axis has no helix to follow: both are refused, saying by how much. A positive
/// pitch climbs along the axis, a negative one against it; a right-handed coil turns
/// counter-clockwise seen down the way it climbs, a left-handed one clockwise.
fn coil(frame: &SectionFrame, regions: &[solidcraft_geom::Region2], origin: DVec3, dir: DVec3, angle: f64, pitch: f64, left_handed: bool) -> Result<Vec<Body>, String> {
    let dir = dir.try_normalize().ok_or("The axis this revolve turns about has no direction.")?;
    let lifted = lift_profiles(frame, regions);
    if profiles_meet_axis(&lifted, origin, dir, 1e-6) {
        return Err("The profile touches or crosses the axis — a coil winds round the axis, so its profile has to stand clear of it.".into());
    }
    let (min, max) = profile_span(&lifted, origin, dir);
    let reach = max - min;
    if angle >= std::f64::consts::TAU - 1e-9 && reach >= pitch.abs() - 1e-6 {
        return Err(format!(
            "The profile reaches {} mm along the axis — as far as the pitch of {} mm or further, so each turn would run into the next. Give a pitch over {} mm, or turn less than once.",
            to_fixed(reach, 3),
            to_fixed(pitch.abs(), 3),
            to_fixed(reach, 3)
        ));
    }
    let turns = angle / std::f64::consts::TAU;
    if turns > KERNEL_MAX_TURNS + 1e-9 {
        return Err(format!("The kernel winds a coil {KERNEL_MAX_TURNS} turns at most, and this one turns {} — give it fewer turns, or a coil in parts.", js_string(js_round(turns * 100.0) / 100.0)));
    }
    // The way it climbs, and the hand about it: a right-handed screw turns about the way it
    // climbs, a left-handed one about the other way, climbing the same.
    let climb = if pitch > 0.0 { dir } else { -dir };
    let (axis, rise) = if left_handed { (-climb, -pitch.abs()) } else { (climb, pitch.abs()) };
    let plane = kernel_plane(frame)?;
    let mut out = Vec::new();
    for r in regions {
        let outer = solidcraft_geom::Region2 { outer: r.outer.ccw(), holes: Vec::new() };
        let mut solid = kernel::sweep_helix(&plane, &outer, sv3(origin), sv3(axis), rise, turns).map_err(kerr)?;
        // A hole is swept the same way and cut out — begun a little before the coil and run a
        // little past it, so the cut goes through its ends rather than lying on them.
        let over = HOLE_OVERRUN_DEG.to_radians();
        let back = rotation_matrix(origin, axis, -over);
        let lift = -axis * (rise * over / std::f64::consts::TAU);
        let moved = |p: DVec3| apply_matrix(&back, p) + lift;
        let start = Plane::new(sv3(moved(frame.origin)), sv3(apply_matrix(&back, frame.origin + frame.u) - apply_matrix(&back, frame.origin)), sv3(apply_matrix(&back, frame.origin + frame.v) - apply_matrix(&back, frame.origin))).ok_or("The coil's plane has no direction.")?;
        for h in &r.holes {
            let hole = solidcraft_geom::Region2 { outer: h.ccw(), holes: Vec::new() };
            let tool = kernel::sweep_helix(&start, &hole, sv3(origin), sv3(axis), rise, turns + 2.0 * over / std::f64::consts::TAU).map_err(kerr)?;
            solid = kernel::boolean(&solid, &tool, BoolOp::Cut).map_err(kerr)?.ok_or("The coil's hole took the whole of it.")?;
        }
        out.push(solid);
    }
    Ok(out)
}

#[cfg(test)]
mod tests;
