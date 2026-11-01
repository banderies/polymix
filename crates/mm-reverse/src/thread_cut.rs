// SPDX-License-Identifier: AGPL-3.0-only
//! A thread cut into a cylindrical face: the geometry of ReKernel's `threadTracked`. Ported
//! from ScanRuler (`plugins/reverse/core/kernelThread.ts`), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only.
//!
//! Everything here is what the cut is made of, worked out from the face and the thread's
//! numbers; the solids themselves are the kernel's. Before the kernel is asked, resolve fails
//! a Thread feature whose face is not round or whose numbers cannot be cut
//! ([`crate::thread::thread_feature_problem`]). A thread is then cut in three steps, each
//! handing the kernel what to build:
//!
//! 1. [`ThreadSetup::new`] reads the face: a boss or a bore, where along it the thread runs,
//!    whether the face has to be brought to the crests first, and whether it is a plain
//!    cylinder between square ends. Where it is not and it is a bore, the kernel is to try
//!    deleting the face to close the body up (the *hull*, [`ThreadSetup::wants_hull`]).
//! 2. [`ThreadSetup::fill`] settles the fill — a sleeve fused onto the body along the thread,
//!    filling a bore in or building a boss up to the crests — or refuses a face that cannot be
//!    brought to them.
//! 3. [`ThreadFill::plan`], once the sleeve is fused, asks the body which ends of the thread
//!    come out of it free, and gives the rest: the band the thread is cut in, the tooth and
//!    the helix it is screwed along (for a sweep such as SolidCraft's `sweep_helix`), the
//!    chamfer rings, and the note for the feature's row.
//!
//! The kernel then makes the cutter as the band less the screwed tooth, cuts the chamfer rings
//! out of the body first and the cutter after, and where a hull was made keeps the result to
//! it — taking off no more than the sleeve's volume, a hull closed up wrong not being trusted.

use std::f64::consts::PI;

use glam::{DVec2, DVec3};
use serde::{Deserialize, Serialize};

use crate::round::{js_round, number_text};
use crate::thread::{chamfer_radius, face_against_thread, fill_radius, minor_diameter, thread_span, tooth_outline, FaceAgainst, ThreadCut, ThreadSpan, ToothOutline};

/// The most turns a thread is cut with: every turn is a face a side of the tooth, and a long
/// fine thread a long boolean.
pub const MAX_THREAD_TURNS: u32 = 400;

/// What a round face has to be for a thread, as the kernel's reading of it says when it is
/// not (`readRound`'s `needs`): "the round face does not go the whole way round — a thread
/// is cut on a full cylinder."
pub const THREAD_FACE_NEEDS: &str = "a thread is cut on a full cylinder";

/// A length in a build message, also the agent's answer: to a micron, in millimetres.
fn mm_text(v: f64) -> String {
    format!("{} mm", number_text(js_round(v * 1000.0) / 1000.0))
}

/// Why a thread could not be cut, in the words the feature's row shows.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("Cutting the thread: {0}")]
pub struct ThreadCutError(pub String);

impl ThreadCutError {
    /// The face picked is not a cylinder.
    pub fn not_a_cylinder() -> Self {
        Self("a thread is cut on a round face — this one is not a cylinder.".to_string())
    }

    /// The cylinder picked does not go the whole way round (`readRound`'s first refusal).
    pub fn not_full_turn() -> Self {
        Self(format!("the round face does not go the whole way round — {THREAD_FACE_NEEDS}."))
    }

    /// The cylinder picked has no length along its axis (`readRound`'s second refusal), which
    /// [`ThreadSetup::new`] makes itself from the face's `lo` and `hi`.
    pub fn no_length() -> Self {
        Self("the round face has no length.".to_string())
    }
}

/// TypeScript's `normalize`: the unit vector, or `None` for one of no length.
fn normalize(v: DVec3) -> Option<DVec3> {
    let l = v.length();
    if l < 1e-12 {
        None
    } else {
        Some(v / l)
    }
}

/// What a thread did that was not asked for in so many words, for the feature's row: a face
/// wider or narrower than the thread, brought to it; the crests come to a point under the
/// offset; a face that wanted bringing to the crests left as drawn (`unfilled`). `None` when
/// it did none.
pub fn thread_note(t: &ThreadCut, internal: bool, face_r: f64, pointed: bool, unfilled: bool) -> Option<String> {
    let s = &t.spec;
    let mut notes: Vec<String> = Vec::new();
    let against = face_against_thread(s, internal, 2.0 * face_r);
    let face = format!("Ø{}", mm_text(2.0 * face_r));
    let minor = || format!("minor Ø{}", mm_text(minor_diameter(s.diameter, s.pitch, s.standard)));
    if unfilled {
        let crests = if internal { minor() } else { format!("Ø{}", mm_text(s.diameter)) };
        notes.push(format!(
            "The {} is {face}, not the thread's {crests}, and it does not run between square ends: it is left as drawn and the crests lie on it.",
            if internal { "bore" } else { "boss" }
        ));
    } else if !internal {
        match against {
            Some(FaceAgainst::Wider) => notes.push(format!("The boss is {face}, wider than the thread's Ø{}: what stood past the crests is cut away.", mm_text(s.diameter))),
            Some(FaceAgainst::Narrower) => notes.push(format!("The boss is {face}, under the thread's Ø{}: it is built up to it along the thread.", mm_text(s.diameter))),
            None => {}
        }
    } else {
        match against {
            Some(FaceAgainst::Wider) => notes.push(format!("The bore is {face}, wider than the thread's {}: it is filled in to it along the thread.", minor())),
            Some(FaceAgainst::Narrower) => notes.push(format!("The bore is {face}, narrower than the thread's {}: it is bored out to it along the thread.", minor())),
            None => {}
        }
    }
    if pointed {
        notes.push("At this offset the crests come to a point.".to_string());
    }
    if notes.is_empty() {
        None
    } else {
        Some(notes.join(" "))
    }
}

/// The round face a thread is cut on, as the kernel reads it (`readFace`, `readRound`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RoundFace {
    /// A point on the axis, which `lo` and `hi` are measured from.
    pub center: DVec3,
    /// The axis's direction.
    pub axis: DVec3,
    /// The radius as drawn.
    pub radius: f64,
    /// Whether the face looks away from its axis: a boss. A bore looks towards it.
    pub convex: bool,
    /// How far along the axis from `center` the face runs, its ends.
    pub lo: f64,
    pub hi: f64,
    /// The face's area, which tells a plain cylinder between square ends from one with an end
    /// cut at a slant or into a curved wall.
    pub area: f64,
}

/// A ring between two radii about the axis, extruded along it from `from` to `to` (measured
/// from `center`): the sleeve that fills a face to the crests, or the band a thread is cut
/// in. With no inner radius it is a full disc.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Annulus {
    pub center: DVec3,
    pub axis: DVec3,
    pub outer: f64,
    pub inner: Option<f64>,
    pub from: f64,
    pub to: f64,
}

/// The face read, the thread placed on it, and the fill it wants — the first step of a cut.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThreadSetup {
    pub cut: ThreadCut,
    /// The face, its axis made a unit one.
    pub face: RoundFace,
    /// A nut's thread, in a bore.
    pub internal: bool,
    /// Where along the axis the thread runs.
    pub span: ThreadSpan,
    /// The radius the face wants bringing to before the cut ([`fill_radius`]), or `None`.
    pub wanted: Option<f64>,
    /// Whether the face is a plain cylinder between square ends.
    pub square: bool,
}

impl ThreadSetup {
    /// The face read for a thread: which way it looks, where along it the thread runs, and
    /// the fill it wants. Refuses a thread that runs past the end of the face.
    pub fn new(cut: &ThreadCut, face: &RoundFace) -> Result<ThreadSetup, ThreadCutError> {
        let axis = normalize(face.axis).unwrap_or(DVec3::Z);
        let face = RoundFace { axis, ..*face };
        let internal = !face.convex;
        let (lo, hi) = (face.lo, face.hi);
        // As `readRound` refuses a face read with no length, before anything is placed on it.
        if !(hi - lo > 1e-9) {
            return Err(ThreadCutError::no_length());
        }
        let span = thread_span(lo, hi, cut.length, cut.start, cut.flip).ok_or_else(|| {
            ThreadCutError(format!(
                "the thread runs past the end of the face — {} in and {} long, on a face {} long.",
                mm_text(cut.start),
                mm_text(cut.length.unwrap_or(0.0)),
                mm_text(hi - lo)
            ))
        })?;
        // A bore wider than the crests is filled in to them along the thread, a boss under
        // them built up; from there on the face is where the fill leaves it.
        let drawn = face.radius;
        let wanted = fill_radius(&cut.spec, internal, drawn);
        // The fill runs between the face's furthest ends. Where the face is not a plain
        // cylinder between square ends — a hole drilled into a curved wall — it would stand
        // past them into the open, and its teeth with it: a nut's thread is cut and then kept
        // to the body with the bore closed up, its neighbours run on across it (cutting a
        // fill already kept to it, along those neighbours, broke the shell). Short of that
        // the face is left as drawn, the crests on it.
        let plain = 2.0 * PI * drawn * (hi - lo);
        let square = (face.area - plain).abs() <= 1e-6 * plain;
        Ok(ThreadSetup { cut: *cut, face, internal, span, wanted, square })
    }

    /// Whether the kernel is to try closing the body up without the face — deleting it, its
    /// neighbours run on across it — for the thread to be kept to afterwards: a bore that
    /// wants filling and does not run between square ends.
    pub fn wants_hull(&self) -> bool {
        self.wanted.is_some() && !self.square && self.internal
    }

    /// The fill settled, once the kernel has said whether it could close the body up without
    /// the face (`hull`; ignored where [`ThreadSetup::wants_hull`] is false). Refuses a face
    /// that wants bringing to the crests, cannot be, and would leave nothing of the thread to
    /// cut as drawn.
    pub fn fill(&self, hull: bool) -> Result<ThreadFill, ThreadCutError> {
        let t = &self.cut.spec;
        let internal = self.internal;
        let drawn = self.face.radius;
        let hull = hull && self.wants_hull();
        let fill = self.wanted.filter(|_| self.square || hull);
        let face_r = fill.unwrap_or(drawn);
        let root = tooth_outline(t, internal, 0.0, None).root;
        if self.wanted.is_some() && fill.is_none() && (if internal { drawn >= root } else { drawn <= root }) {
            let what = if internal {
                format!("bore is Ø{}, past the thread's major Ø{}", mm_text(2.0 * drawn), mm_text(2.0 * root))
            } else {
                format!("boss is Ø{}, under the thread's minor Ø{}", mm_text(2.0 * drawn), mm_text(2.0 * root))
            };
            return Err(ThreadCutError(format!(
                "the {what}, and it does not run between square ends, so it cannot be {} to the thread — draw it at the thread's size.",
                if internal { "filled in" } else { "built up" }
            )));
        }
        let sleeve = fill.map(|f| Annulus { center: self.face.center, axis: self.face.axis, outer: drawn.max(f), inner: Some(drawn.min(f)), from: self.span.from, to: self.span.to });
        Ok(ThreadFill { setup: *self, fill, face_r, hull, sleeve })
    }
}

/// The fill settled — the second step of a cut.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThreadFill {
    pub setup: ThreadSetup,
    /// The radius the face is brought to, or `None` where it is cut as drawn.
    pub fill: Option<f64>,
    /// The face's radius once filled.
    pub face_r: f64,
    /// Whether the kernel closed the body up without the face, for the thread to be kept to.
    pub hull: bool,
    /// The sleeve to fuse onto the body before the cut, along the thread between the face as
    /// drawn and the fill.
    pub sleeve: Option<Annulus>,
}

/// The tooth screwed along the axis: a profile in a plane through the axis, swept along a
/// helix.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HelixSweep {
    /// The plane the profile is drawn in: through the axis, at `plane_origin` on it, its x
    /// along `plane_x` (out from the axis), its y along `plane_y` (the axis).
    pub plane_origin: DVec3,
    pub plane_x: DVec3,
    pub plane_y: DVec3,
    /// The tooth in that plane, (r, z), counter-clockwise.
    pub profile: Vec<DVec2>,
    /// The same in the world.
    pub profile_world: Vec<DVec3>,
    /// The helix's axis.
    pub axis_origin: DVec3,
    pub axis: DVec3,
    /// How far it climbs along `axis` in a turn.
    pub pitch: f64,
    /// Whole turns, so each of the sweep's faces is a turn of it.
    pub turns: u32,
    pub left_handed: bool,
    /// Whether each turn is to be its own edge of the helix, and so its own face a side of
    /// the tooth: a sweep in one face a side failed past a dozen turns, and for a left hand
    /// at any length.
    pub turn_by_turn: bool,
}

impl HelixSweep {
    /// The angle the helix turns through, radians.
    pub fn angle(&self) -> f64 {
        2.0 * PI * f64::from(self.turns)
    }

    /// The axis and the pitch for a sweep that only turns right-handed about its axis, as
    /// SolidCraft's `sweep_helix` does: a right-hand thread as it is; a left-hand one about
    /// the reversed axis at the negated pitch, which turns it the other way and still climbs
    /// along `axis`.
    pub fn right_handed_axis_and_pitch(&self) -> (DVec3, f64) {
        if self.left_handed {
            (-self.axis, -self.pitch)
        } else {
            (self.axis, self.pitch)
        }
    }
}

/// A 45° chamfer at an end of the face the thread comes out of free: a triangle in a plane
/// through the axis, turned the whole way round it and cut out of the body before the thread.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ChamferRing {
    /// The end of the face it is at, along the axis from the face's centre.
    pub at: f64,
    /// The way out of the body there: −1 at the low end, +1 at the high one.
    pub out: f64,
    /// The triangle in (r, s): r from the axis, s into the body from the end face.
    pub section: [DVec2; 3],
    /// The same in the world, to be revolved a whole turn about the axis.
    pub world: [DVec3; 3],
    pub center: DVec3,
    pub axis: DVec3,
}

/// Everything a thread's cut is built from — the third step.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThreadPlan {
    pub internal: bool,
    /// The tooth as screwed, in (r, z).
    pub tooth: ToothOutline,
    /// The band the thread is cut in; the cutter is this less the screwed tooth.
    pub band: Annulus,
    /// The tooth's sweep.
    pub helix: HelixSweep,
    /// Whether each end of the thread comes out of the body free, and the band runs on past
    /// it.
    pub free_lo: bool,
    pub free_hi: bool,
    /// The chamfers, cut before the thread.
    pub chamfers: Vec<ChamferRing>,
    /// Whether the cut is to be kept to the hull afterwards — taking off no more than the
    /// sleeve's volume.
    pub keep_to_hull: bool,
    /// What the row says about the cut, if anything.
    pub note: Option<String>,
}

impl ThreadFill {
    /// The cut planned on the body as it stands once the sleeve is fused: `contains` says
    /// whether a point is inside it, which tells the ends the thread comes out of free —
    /// where the band runs on past the end and a chamfer may go — from those that run up to
    /// more of the body.
    pub fn plan(&self, mut contains: impl FnMut(DVec3) -> bool) -> Result<ThreadPlan, ThreadCutError> {
        let setup = &self.setup;
        let cut = &setup.cut;
        let t = &cut.spec;
        let internal = setup.internal;
        let RoundFace { center, axis, radius: drawn, lo, hi, .. } = setup.face;
        let span = setup.span;
        let p = t.pitch;
        let fill = self.fill;
        let face_r = self.face_r;
        let root = tooth_outline(t, internal, 0.0, None).root;
        // The tooth reaches into what stands behind it by half a pitch — a bolt's not so far
        // that it nears the axis.
        let tooth = tooth_outline(t, internal, if internal { p / 2.0 } else { (p / 2.0).min(root / 2.0) }, Some(face_r));
        // The band, radially: past the face on the side away from the material by a margin,
        // to the root on the other.
        let margin = p / 2.0;
        let inner = if internal { face_r.min(tooth.crest) - margin } else { tooth.root };
        let outer = if internal { tooth.root } else { face_r.max(t.diameter / 2.0) + margin };
        // And along the axis: on past an end of the thread that nothing stands beyond, within
        // the band's reach.
        let run = p / 2.0;
        let radial = normalize(axis.cross(if axis.x.abs() < 0.9 { DVec3::X } else { DVec3::Y })).unwrap_or(DVec3::X);
        let across = axis.cross(radial);
        // Whether nothing of the body lies past the end at `at`, the way `way` points,
        // between the radii, as far as `reach`.
        let mut clear = |at: f64, way: f64, near: f64, far: f64, reach: f64| -> bool {
            let r_in = near.max(1e-3);
            for along in [reach / 2.0, reach] {
                for share in [0.1, 0.5, 0.9] {
                    let r = r_in + (far - r_in) * share;
                    for turn in [0.0, (2.0 * PI) / 3.0, (4.0 * PI) / 3.0] {
                        let q = center + axis * (at + way * along) + radial * (r * f64::cos(turn)) + across * (r * f64::sin(turn));
                        if contains(q) {
                            return false;
                        }
                    }
                }
            }
            true
        };
        // An end of the face, or of a fill short of it where the face as drawn stands clear
        // of the band — an M4's in a Ø5 bore, not an M5's, whose root is the bore: run on,
        // the band's root would lie on the bore past the fill, or with an offset cut into it.
        let end_lo = (span.from - lo).abs() < 1e-6;
        let end_hi = (span.to - hi).abs() < 1e-6;
        let open = fill.is_some() && (if internal { drawn > outer + p / 20.0 } else { drawn < inner - p / 20.0 });
        let free_lo = (end_lo || open) && clear(span.from, -1.0, inner, outer, run);
        let free_hi = (end_hi || open) && clear(span.to, 1.0, inner, outer, run);
        let from = if free_lo { span.from - run } else { span.from };
        let to = if free_hi { span.to + run } else { span.to };
        // The chamfer, at each end of the face the thread runs out of free: a cone at 45°
        // through `chamfer_radius` on the end face, from a margin past the end face — so the
        // cut does not end on it — to a margin past the crests, or into the bore past them. In
        // (r, s), s into the body.
        let edge = p / 8.0;
        let rc = chamfer_radius(t, internal);
        let bore = face_r.min(tooth.crest);
        let c_far = if internal { (bore - edge).max(bore / 2.0) } else { face_r.max(t.diameter / 2.0) + edge };
        let c_near = if internal { rc + edge } else { rc - edge };
        let mut chamfer_at: Vec<(f64, f64)> = Vec::new();
        if cut.chamfer {
            let (r_in, r_out) = (c_near.min(c_far), c_near.max(c_far));
            let lo_open = clear(lo, -1.0, r_in, r_out, edge);
            let hi_open = clear(hi, 1.0, r_in, r_out, edge);
            if end_lo && free_lo && lo_open {
                chamfer_at.push((lo, -1.0));
            }
            if end_hi && free_hi && hi_open {
                chamfer_at.push((hi, 1.0));
            }
            if chamfer_at.is_empty() {
                return Err(ThreadCutError(format!(
                    "there is no end to chamfer — a chamfer goes where the thread comes out of the body free, a bolt’s tip or a bore’s mouth, and this one {}runs up to more of the body.",
                    if cut.length.is_none() { "" } else { "stops short of the face’s ends or " }
                )));
            }
            let length = hi - lo;
            // How far in the cone leaves the thread — at a bolt's crests, a nut's — and how far
            // the cut runs on, a margin past them: past the face's other end it would cut into
            // whatever stands there.
            let reach = if internal { rc - face_r.max(tooth.crest) } else { face_r.min(tooth.crest) - rc };
            let deep = (c_far - rc).abs();
            if chamfer_at.len() == 2 && 2.0 * reach >= length {
                return Err(ThreadCutError(format!("the face is {} long — too short for a chamfer at each end, each {} deep.", mm_text(length), mm_text(reach))));
            }
            if let [(at, _)] = chamfer_at.as_slice() {
                let other_open = if *at == lo { hi_open } else { lo_open };
                if deep > length && !other_open {
                    return Err(ThreadCutError(format!("the face is {} long — too short for a chamfer {} deep.", mm_text(length), mm_text(deep))));
                }
            }
        }
        let chamfers = chamfer_at
            .iter()
            .map(|&(at, out)| {
                let section = if internal {
                    [DVec2::new(c_near, -edge), DVec2::new(c_far, rc - c_far), DVec2::new(c_far, -edge)]
                } else {
                    [DVec2::new(c_near, -edge), DVec2::new(c_far, c_far - rc), DVec2::new(c_far, -edge)]
                };
                let world = section.map(|q| center + axis * (at - out * q.y) + radial * q.x);
                ChamferRing { at, out, section, world, center, axis }
            })
            .collect();
        // Whole turns, so each of the sweep's faces is a turn of it.
        let turns = ((to - from) / p - 1e-9).ceil() + 2.0;
        if !(turns <= f64::from(MAX_THREAD_TURNS)) {
            return Err(ThreadCutError(format!(
                "the thread would take {} turns — at most {MAX_THREAD_TURNS}; give it a coarser pitch or a shorter length.",
                number_text(js_round(turns))
            )));
        }
        // A span run backwards — a length of nothing or less, which resolve refuses before
        // the kernel is asked ([`crate::thread::thread_feature_problem`]) — leaves no turn to
        // screw; the kernel's screw refuses it so, inside the cut.
        if turns < 1.0 {
            return Err(ThreadCutError("Sweeping the coil: the coil has no angle.".to_string()));
        }
        let plane_origin = center + axis * (from - p);
        let profile_world = tooth.points.iter().map(|q| center + axis * (from - p + q.y) + radial * q.x).collect();
        let helix = HelixSweep {
            plane_origin,
            plane_x: radial,
            plane_y: axis,
            profile: tooth.points.clone(),
            profile_world,
            axis_origin: center,
            axis,
            pitch: p,
            // A whole count from 1 to MAX_THREAD_TURNS, checked above.
            turns: turns as u32,
            left_handed: cut.left_handed,
            turn_by_turn: true,
        };
        let band = Annulus { center, axis, outer, inner: (inner > 1e-3).then_some(inner), from, to };
        let note = thread_note(cut, internal, drawn, tooth.pointed, setup.wanted.is_some() && fill.is_none());
        Ok(ThreadPlan { internal, tooth, band, helix, free_lo, free_hi, chamfers, keep_to_hull: self.hull && self.sleeve.is_some(), note })
    }
}

#[cfg(test)]
mod tests {
    //! The kernel's own tests of a thread (reThread.test.ts, "a thread cut in the kernel" and
    //! on) build the solids in Open CASCADE and measure them. What they check of the plan —
    //! the refusals, the notes, which ends run free and take a chamfer — is checked here on
    //! bodies described by a point-inside test.
    use super::*;
    use crate::thread::{ThreadSpec, ThreadStandard};

    fn cut(diameter: f64, pitch: f64) -> ThreadCut {
        ThreadCut { spec: ThreadSpec { standard: ThreadStandard::Metric, diameter, pitch, offset: 0.0 }, left_handed: false, length: None, start: 0.0, flip: false, chamfer: false }
    }

    /// The round face of a bar of radius `r` up Z from 0 to `length`, or of a bore of that
    /// radius through a block.
    fn round(r: f64, length: f64, convex: bool) -> RoundFace {
        RoundFace { center: DVec3::ZERO, axis: DVec3::Z, radius: r, convex, lo: 0.0, hi: length, area: 2.0 * PI * r * length }
    }

    fn bar(r: f64, length: f64) -> impl Fn(DVec3) -> bool {
        move |q: DVec3| q.truncate().length() <= r && q.z >= 0.0 && q.z <= length
    }

    /// A Ø30 block `height` thick with a bore of radius `r` through it up Z.
    fn plate(r: f64, height: f64) -> impl Fn(DVec3) -> bool {
        move |q: DVec3| {
            let d = q.truncate().length();
            d >= r && d <= 15.0 && q.z >= 0.0 && q.z <= height
        }
    }

    fn plan(c: &ThreadCut, face: &RoundFace, hull: bool, contains: impl Fn(DVec3) -> bool) -> Result<ThreadPlan, ThreadCutError> {
        ThreadSetup::new(c, face)?.fill(hull)?.plan(contains)
    }

    fn message(r: Result<ThreadPlan, ThreadCutError>) -> String {
        match r {
            Ok(_) => String::new(),
            Err(e) => e.to_string(),
        }
    }

    #[test]
    fn cuts_a_bolts_thread_the_whole_length_of_a_boss_the_ends_run_free() -> Result<(), ThreadCutError> {
        let t = cut(16.0, 2.0);
        let p = plan(&t, &round(8.0, 30.0, true), false, bar(8.0, 30.0))?;
        assert!(!p.internal && p.free_lo && p.free_hi);
        assert_eq!(p.note, None);
        let root = tooth_outline(&t.spec, false, 0.0, None).root;
        assert_eq!(p.band, Annulus { center: DVec3::ZERO, axis: DVec3::Z, outer: 9.0, inner: Some(root), from: -1.0, to: 31.0 });
        assert_eq!(p.helix.turns, 18);
        assert_eq!(p.helix.profile, tooth_outline(&t.spec, false, 1.0, Some(8.0)).points);
        // The profile stands one pitch before the band's start, in the plane through +Y (the
        // axis crossed with +X).
        assert_eq!(p.helix.plane_x, DVec3::Y);
        for (q, w) in p.helix.profile.iter().zip(&p.helix.profile_world) {
            assert!((*w - DVec3::new(0.0, q.x, -3.0 + q.y)).length() < 1e-12);
        }
        assert_eq!(p.helix.right_handed_axis_and_pitch(), (DVec3::Z, 2.0));
        let left = plan(&ThreadCut { left_handed: true, ..t }, &round(8.0, 30.0, true), false, bar(8.0, 30.0))?;
        assert_eq!(left.helix.right_handed_axis_and_pitch(), (-DVec3::Z, -2.0));
        Ok(())
    }

    #[test]
    fn refuses_a_length_past_the_end_of_the_face() {
        let t = ThreadCut { length: Some(10.0), start: 25.0, ..cut(16.0, 2.0) };
        assert_eq!(message(plan(&t, &round(8.0, 30.0, true), false, bar(8.0, 30.0))), "Cutting the thread: the thread runs past the end of the face — 25 mm in and 10 mm long, on a face 30 mm long.");
    }

    #[test]
    fn stops_at_a_shoulder() -> Result<(), ThreadCutError> {
        // A pin of radius 8 up to z 20 under a head of radius 12 to z 30.
        let body = |q: DVec3| bar(8.0, 20.0)(q) || (q.truncate().length() <= 12.0 && q.z >= 20.0 && q.z <= 30.0);
        let p = plan(&cut(16.0, 2.0), &round(8.0, 20.0, true), false, body)?;
        assert!(p.free_lo && !p.free_hi);
        assert_eq!((p.band.from, p.band.to), (-1.0, 20.0));
        // A chamfer goes on the tip alone.
        let chamfered = plan(&ThreadCut { chamfer: true, ..cut(16.0, 2.0) }, &round(8.0, 20.0, true), false, body)?;
        assert_eq!(chamfered.chamfers.len(), 1);
        assert_eq!((chamfered.chamfers[0].at, chamfered.chamfers[0].out), (0.0, -1.0));
        Ok(())
    }

    #[test]
    fn chamfers_a_bolt_at_both_free_ends_at_45_degrees_from_just_inside_the_root() -> Result<(), ThreadCutError> {
        let t = ThreadCut { chamfer: true, ..cut(16.0, 2.0) };
        let p = plan(&t, &round(8.0, 30.0, true), false, bar(8.0, 30.0))?;
        let rc = chamfer_radius(&t.spec, false);
        assert!((rc - (tooth_outline(&t.spec, false, 0.0, None).root - 0.2)).abs() < 1e-9);
        assert_eq!(p.chamfers.len(), 2);
        for ring in &p.chamfers {
            let [a, b, c] = ring.section;
            // From a margin past the end face, along the 45° cone through rc, out past the
            // crests.
            assert_eq!(a, DVec2::new(rc - 0.25, -0.25));
            assert_eq!(b, DVec2::new(8.25, 8.25 - rc));
            assert_eq!(c, DVec2::new(8.25, -0.25));
            assert!(((b.y - a.y) / (b.x - a.x) - 1.0).abs() < 1e-12);
            // The end face is at s = 0, into the body positive.
            let s0 = ring.world[1].z;
            assert!((s0 - (ring.at - ring.out * (8.25 - rc))).abs() < 1e-12);
        }
        Ok(())
    }

    #[test]
    fn countersinks_a_nuts_bore_out_past_the_major_diameter() -> Result<(), ThreadCutError> {
        let minor = minor_diameter(16.0, 2.0, ThreadStandard::Metric) / 2.0;
        let t = ThreadCut { chamfer: true, ..cut(16.0, 2.0) };
        let p = plan(&t, &round(minor, 12.0, false), false, plate(minor, 12.0))?;
        assert!(p.internal);
        assert_eq!(p.note, None);
        assert!((chamfer_radius(&t.spec, true) - 8.2).abs() < 1e-9);
        assert_eq!(p.chamfers.len(), 2);
        assert_eq!(p.band.outer, 8.0);
        // A plate 3 mm thick — a turn and a half — takes both.
        let thin = plan(&t, &round(minor, 3.0, false), false, plate(minor, 3.0))?;
        assert_eq!(thin.chamfers.len(), 2);
        Ok(())
    }

    #[test]
    fn refuses_a_chamfer_where_the_thread_does_not_come_out_of_the_body_free() {
        let t = ThreadCut { length: Some(10.0), start: 5.0, chamfer: true, ..cut(16.0, 2.0) };
        let none = message(plan(&t, &round(8.0, 30.0, true), false, bar(8.0, 30.0)));
        assert!(none.contains("no end to chamfer"), "{none}");
        assert!(none.contains("stops short of the face’s ends or runs up to more of the body."), "{none}");
        let short = message(plan(&ThreadCut { chamfer: true, ..cut(16.0, 2.0) }, &round(8.0, 1.5, true), false, bar(8.0, 1.5)));
        assert!(short.contains("too short for a chamfer at each end"), "{short}");
    }

    #[test]
    fn cuts_a_boss_wider_than_the_thread_down_to_it_and_builds_one_too_thin_up_to_it() -> Result<(), ThreadCutError> {
        let wide = plan(&cut(16.0, 2.0), &round(10.0, 20.0, true), false, bar(10.0, 20.0))?;
        assert_eq!(wide.note.as_deref(), Some("The boss is Ø20 mm, wider than the thread's Ø16 mm: what stood past the crests is cut away."));
        assert_eq!(wide.band.outer, 11.0);
        let setup = ThreadSetup::new(&cut(16.0, 2.0), &round(6.0, 20.0, true))?;
        let fill = setup.fill(false)?;
        assert_eq!(fill.fill, Some(8.0));
        assert_eq!(fill.sleeve, Some(Annulus { center: DVec3::ZERO, axis: DVec3::Z, outer: 8.0, inner: Some(6.0), from: 0.0, to: 20.0 }));
        // Once the sleeve is fused the body is a bar of radius 8.
        let built = fill.plan(bar(8.0, 20.0))?;
        assert_eq!(built.note.as_deref(), Some("The boss is Ø12 mm, under the thread's Ø16 mm: it is built up to it along the thread."));
        assert!(!built.keep_to_hull);
        Ok(())
    }

    #[test]
    fn fills_a_5_mm_bore_in_to_the_thread_for_m5_m4_or_m6() -> Result<(), ThreadCutError> {
        for (d, p, note) in [
            (5.0, 0.8, Some("The bore is Ø5 mm, wider than the thread's minor Ø4.134 mm: it is filled in to it along the thread.")),
            (4.0, 0.7, Some("The bore is Ø5 mm, wider than the thread's minor Ø3.242 mm: it is filled in to it along the thread.")),
            (6.0, 1.0, None),
        ] {
            let fill = ThreadSetup::new(&cut(d, p), &round(2.5, 10.0, false))?.fill(false)?;
            let crest = minor_diameter(d, p, ThreadStandard::Metric) / 2.0;
            assert!((fill.fill.unwrap_or(f64::NAN) - crest).abs() < 1e-12, "M{d}");
            let planned = fill.plan(plate(crest, 10.0))?;
            assert_eq!(planned.note.as_deref(), note, "M{d}");
        }
        Ok(())
    }

    #[test]
    fn keeps_a_bore_out_of_a_curved_wall_to_the_body_or_refuses_it() -> Result<(), ThreadCutError> {
        // Its ends are saddles: the face's area is not a plain cylinder's.
        let face = RoundFace { area: 0.9 * 2.0 * PI * 2.5 * 3.2, ..round(2.5, 3.2, false) };
        let setup = ThreadSetup::new(&cut(5.0, 0.8), &face)?;
        assert!(!setup.square && setup.wants_hull());
        let kept = setup.fill(true)?;
        assert!(kept.fill.is_some());
        assert!(kept.plan(plate(kept.face_r, 3.2))?.keep_to_hull);
        assert_eq!(
            setup.fill(false).err().map(|e| e.to_string()).as_deref(),
            Some("Cutting the thread: the bore is Ø5 mm, past the thread's major Ø5 mm, and it does not run between square ends, so it cannot be filled in to the thread — draw it at the thread's size.")
        );
        Ok(())
    }

    #[test]
    fn leaves_a_boss_with_a_slanted_end_as_drawn_rather_than_build_it_up_past_the_end() -> Result<(), ThreadCutError> {
        // A pin, its tip cut off at a slant: the face is not a plain cylinder's.
        let slanted = |r: f64| RoundFace { area: 0.95 * 2.0 * PI * r * 15.0, ..round(r, 15.0, true) };
        // Ø4.6, between M5's minor and major: cut as drawn, the crests on it.
        let setup = ThreadSetup::new(&cut(5.0, 0.8), &slanted(2.3))?;
        assert!(!setup.wants_hull());
        let fill = setup.fill(false)?;
        assert_eq!(fill.fill, None);
        let drawn = fill.plan(bar(2.3, 15.0))?;
        assert_eq!(
            drawn.note.as_deref(),
            Some("The boss is Ø4.6 mm, not the thread's Ø5 mm, and it does not run between square ends: it is left as drawn and the crests lie on it.")
        );
        // Ø4, under M5's minor: nothing of the thread would be cut, refused.
        let thin = ThreadSetup::new(&cut(5.0, 0.8), &slanted(2.0))?.fill(false).err().map(|e| e.to_string()).unwrap_or_default();
        assert!(thin.contains("the boss is Ø4 mm, under the thread's minor Ø4.134 mm, and it does not run between square ends, so it cannot be built up to the thread"), "{thin}");
        Ok(())
    }

    #[test]
    fn says_when_the_offset_brings_the_crests_to_a_point() -> Result<(), ThreadCutError> {
        let sharp = ThreadCut { spec: ThreadSpec { offset: 0.3, ..cut(16.0, 2.0).spec }, ..cut(16.0, 2.0) };
        let p = plan(&sharp, &round(8.0, 30.0, true), false, bar(8.0, 30.0))?;
        assert_eq!(p.note.as_deref(), Some("At this offset the crests come to a point."));
        // A Ø4 pin built up to an M5 whose crests are points at 0.12 mm.
        let bolt = ThreadCut { spec: ThreadSpec { offset: 0.12, ..cut(5.0, 0.8).spec }, ..cut(5.0, 0.8) };
        let fill = ThreadSetup::new(&bolt, &round(2.0, 15.0, true))?.fill(false)?;
        let built = fill.plan(bar(fill.face_r, 15.0))?;
        let note = built.note.unwrap_or_default();
        assert!(note.contains("built up to it along the thread. At this offset the crests come to a point"), "{note}");
        Ok(())
    }

    /// A body with a fill's sleeve fused onto it, about the Z axis.
    fn with_sleeve(body: impl Fn(DVec3) -> bool, sleeve: Option<Annulus>) -> impl Fn(DVec3) -> bool {
        move |q: DVec3| {
            body(q)
                || sleeve.is_some_and(|s| {
                    let d = q.truncate().length();
                    d >= s.inner.unwrap_or(0.0) && d <= s.outer && q.z >= s.from.min(s.to) && q.z <= s.from.max(s.to)
                })
        }
    }

    /// The setup, fill and plan in one, the sleeve fused onto `body` between the last two.
    fn plan_filled(c: &ThreadCut, face: &RoundFace, body: impl Fn(DVec3) -> bool) -> Result<(ThreadFill, ThreadPlan), ThreadCutError> {
        let fill = ThreadSetup::new(c, face)?.fill(false)?;
        let planned = fill.plan(with_sleeve(body, fill.sleeve))?;
        Ok((fill, planned))
    }

    /// "fills the bore only along the thread, leaving the rest as drawn": M5's root is the
    /// bore, M4's stands inside it, so its band runs on past the fill's end into the open bore.
    #[test]
    fn fills_the_bore_only_along_the_thread_the_band_running_on_into_an_open_bore() -> Result<(), ThreadCutError> {
        let face = round(2.5, 20.0, false);
        let body = plate(2.5, 20.0);
        let m5 = ThreadCut { length: Some(8.0), ..cut(5.0, 0.8) };
        let (fill, p) = plan_filled(&m5, &face, &body)?;
        assert_eq!(fill.sleeve.map(|s| (s.from, s.to)), Some((12.0, 20.0)));
        assert!(!p.free_lo && p.free_hi);
        assert!(p.band.from == 12.0 && (p.band.to - 20.4).abs() < 1e-12);
        let m4 = ThreadCut { length: Some(8.0), ..cut(4.0, 0.7) };
        let (_, p) = plan_filled(&m4, &face, &body)?;
        assert!(p.free_lo && p.free_hi, "M4's band runs on into the bore");
        assert!((p.band.from - 11.65).abs() < 1e-12 && (p.band.to - 20.35).abs() < 1e-12);
        let flipped = ThreadCut { length: Some(8.0), start: 2.0, flip: true, ..cut(5.0, 0.8) };
        let (fill, p) = plan_filled(&flipped, &face, &body)?;
        assert_eq!(fill.sleeve.map(|s| (s.from, s.to)), Some((2.0, 10.0)));
        assert!(!p.free_lo && !p.free_hi);
        assert_eq!((p.band.from, p.band.to), (2.0, 10.0));
        Ok(())
    }

    /// "keeps the wall past a short thread whole under an offset, and fills past pointed
    /// crests": the offset takes the root out past the bore, which is no open bore beyond.
    #[test]
    fn keeps_the_wall_past_a_short_thread_whole_under_an_offset() -> Result<(), ThreadCutError> {
        let face = round(2.5, 20.0, false);
        let loose = ThreadCut { length: Some(8.0), spec: ThreadSpec { offset: 0.05, ..cut(5.0, 0.8).spec }, ..cut(5.0, 0.8) };
        let (_, p) = plan_filled(&loose, &face, plate(2.5, 20.0))?;
        assert!(!p.free_lo, "the band stops at the thread's end");
        assert_eq!(p.band.from, 12.0);
        assert!((p.band.outer - 2.55).abs() < 1e-12);
        // Near the offset's limit the fill follows the crests out.
        let deep = ThreadCut { spec: ThreadSpec { offset: 0.16, ..cut(5.0, 0.8).spec }, ..cut(5.0, 0.8) };
        let (fill, _) = plan_filled(&deep, &face, plate(2.5, 20.0))?;
        assert!((fill.fill.unwrap_or(f64::NAN) - (minor_diameter(5.0, 0.8, ThreadStandard::Metric) / 2.0 + 0.16)).abs() < 1e-12);
        Ok(())
    }

    /// "countersinks a filled bore at both mouths".
    #[test]
    fn countersinks_a_filled_bore_at_both_mouths() -> Result<(), ThreadCutError> {
        for (d, p) in [(5.0, 0.8), (4.0, 0.7)] {
            let t = ThreadCut { chamfer: true, ..cut(d, p) };
            let (fill, planned) = plan_filled(&t, &round(2.5, 10.0, false), plate(2.5, 10.0))?;
            assert!(fill.fill.is_some(), "M{d}");
            let ends: Vec<(f64, f64)> = planned.chamfers.iter().map(|c| (c.at, c.out)).collect();
            assert_eq!(ends, [(0.0, -1.0), (10.0, 1.0)], "M{d}");
            let rc = chamfer_radius(&t.spec, true);
            for ring in &planned.chamfers {
                // Out from just past the countersink's radius on the end face.
                assert!((ring.section[0].x - (rc + p / 8.0)).abs() < 1e-12, "M{d}");
            }
        }
        Ok(())
    }

    /// "builds a Ø4 pin up to an M5 along part of its length": the band runs on past the
    /// fill's end, where the pin stands clear of it, and only the tip is chamfered.
    #[test]
    fn builds_a_pin_up_along_part_of_its_length_and_chamfers_its_tip() -> Result<(), ThreadCutError> {
        let t = ThreadCut { length: Some(8.0), chamfer: true, ..cut(5.0, 0.8) };
        let (fill, p) = plan_filled(&t, &round(2.0, 15.0, true), bar(2.0, 15.0))?;
        assert_eq!(fill.sleeve, Some(Annulus { center: DVec3::ZERO, axis: DVec3::Z, outer: 2.5, inner: Some(2.0), from: 7.0, to: 15.0 }));
        assert!(p.free_lo && p.free_hi);
        assert_eq!(p.chamfers.iter().map(|c| (c.at, c.out)).collect::<Vec<_>>(), [(15.0, 1.0)]);
        assert!(p.note.unwrap_or_default().contains("built up to it along the thread"));
        Ok(())
    }

    /// "chamfers a fine left-hand thread with an offset", and the printing form's cases of
    /// "cuts the printing form on a boss and in a bore" and "takes the offset off every face".
    #[test]
    fn plans_a_fine_left_hand_thread_and_the_printing_form_without_a_note() -> Result<(), ThreadCutError> {
        let fine = ThreadCut { left_handed: true, chamfer: true, spec: ThreadSpec { offset: 0.05, ..cut(3.0, 0.5).spec }, ..cut(3.0, 0.5) };
        let p = plan(&fine, &round(1.5, 12.0, true), false, bar(1.5, 12.0))?;
        assert_eq!(p.chamfers.len(), 2);
        assert_eq!(p.note, None);
        assert_eq!(p.helix.right_handed_axis_and_pitch(), (-DVec3::Z, -0.5));
        assert_eq!(p.helix.turns, 27);
        let print = ThreadCut { spec: ThreadSpec { standard: ThreadStandard::Print45, ..cut(10.0, 2.75).spec }, ..cut(10.0, 2.75) };
        let bolt = plan(&print, &round(5.0, 20.0, true), false, bar(5.0, 20.0))?;
        assert_eq!(bolt.note, None);
        assert_eq!(bolt.helix.profile, tooth_outline(&print.spec, false, 1.375, Some(5.0)).points);
        let minor = minor_diameter(10.0, 2.75, ThreadStandard::Print45) / 2.0;
        let nut = plan(&print, &round(minor, 12.0, false), false, plate(minor, 12.0))?;
        assert!(nut.internal);
        assert_eq!(nut.note, None);
        let loose = ThreadCut { left_handed: true, chamfer: true, spec: ThreadSpec { offset: 0.2, ..print.spec }, ..print };
        assert_eq!(plan(&loose, &round(5.0, 20.0, true), false, bar(5.0, 20.0))?.chamfers.len(), 2);
        let offset = ThreadCut { spec: ThreadSpec { offset: 0.2, ..cut(16.0, 2.0).spec }, ..cut(16.0, 2.0) };
        assert_eq!(plan(&offset, &round(8.0, 30.0, true), false, bar(8.0, 30.0))?.note, None);
        Ok(())
    }

    /// readRound's refusals, and a span run backwards, which the kernel's screw refuses.
    #[test]
    fn refuses_a_face_of_no_length_and_a_span_run_backwards() {
        let flat = RoundFace { hi: 0.0, ..round(8.0, 30.0, true) };
        assert_eq!(message(plan(&cut(16.0, 2.0), &flat, false, bar(8.0, 30.0))), "Cutting the thread: the round face has no length.");
        assert_eq!(ThreadCutError::not_full_turn().to_string(), "Cutting the thread: the round face does not go the whole way round — a thread is cut on a full cylinder.");
        let backwards = ThreadCut { length: Some(-10.0), ..cut(16.0, 2.0) };
        assert_eq!(message(plan(&backwards, &round(8.0, 30.0, true), false, bar(8.0, 30.0))), "Cutting the thread: Sweeping the coil: the coil has no angle.");
    }

    #[test]
    fn refuses_more_turns_than_it_cuts() {
        let fine = ThreadCut { spec: ThreadSpec { standard: ThreadStandard::Metric, diameter: 3.0, pitch: 0.2, offset: 0.0 }, ..cut(3.0, 0.2) };
        let long = message(plan(&fine, &round(1.5, 100.0, true), false, bar(1.5, 100.0)));
        assert_eq!(long, "Cutting the thread: the thread would take 503 turns — at most 400; give it a coarser pitch or a shorter length.");
    }
}
