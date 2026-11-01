// SPDX-License-Identifier: AGPL-3.0-only
//! Screw threads, as the Thread feature cuts them: the standard sizes and the basic profile
//! of each form, as the outline of one tooth in a plane through the axis. Ported from
//! ScanRuler (`plugins/reverse/core/thread.ts`), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only.
//!
//! The kernel brings a round face to the thread's crests, whatever size it was drawn
//! ([`fill_radius`]), screws the tooth along it and takes out of the body what lies between
//! its turns (see [`crate::thread_cut`]).
//!
//! There are two forms. ISO metric coarse and fine, Unified coarse (UNC) and fine (UNF) share
//! one: a 60° vee cut flat at the crest and at the root (ISO 68-1, ASME B1.1). The other is
//! made for 3D printing: 90° between the flanks, each at 45° to the radius, and a flat a
//! fifth of the pitch wide at the crest and at the root. Printed with its axis upright, the
//! flats stand upright and every face that looks down leans at 45°, so it prints without
//! supports; the 60° form's flanks lie 30° off level. It is a form of its own — it mates only
//! with itself — sized on the ISO metric diameters at the pitch that cuts as deep as their
//! coarse thread.
//!
//! The outline is the material that stays: a bolt's tooth standing on its core, or a nut's
//! tooth hanging from its wall. Either reaches a little past the band the thread is cut in,
//! so the tooth and the band never share a surface, and is narrower than the pitch, so the
//! turns never meet.
//!
//! The offset is a clearance for printing: every face of the thread moved into the material
//! — the flanks thinner, the crest lower, the root deeper — so a printed bolt turns into a
//! printed nut. At a quarter of the pitch times the cosine of the flank's angle — an eighth
//! of √3 times the pitch in the 60° form, of √2 times it in the 45° one — the tooth has no
//! thickness left where the threads bear, the pitch diameter, and the offset is refused
//! there. Short of that, the 60° form's crest comes to a point once the offset has used up
//! its flat; the 45° form's flats are wide enough that it never does.

use std::sync::OnceLock;

use glam::{DVec2, DVec3};
use serde::{Deserialize, Serialize};

use crate::round::{js_round, number_text};
use crate::units::{length_decimals, to_display, unit_word, DisplayUnits};

/// The standard a thread's sizes come from, and with them its form: the 60° one of the
/// metric and Unified tables, or the 45° one for printing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ThreadStandard {
    Metric,
    Unc,
    Unf,
    Print45,
}

/// The standards in the order they are offered.
pub const THREAD_STANDARDS: [ThreadStandard; 4] = [ThreadStandard::Metric, ThreadStandard::Unc, ThreadStandard::Unf, ThreadStandard::Print45];

impl ThreadStandard {
    /// Its id, as a project and a command name it.
    pub fn id(self) -> &'static str {
        match self {
            ThreadStandard::Metric => "metric",
            ThreadStandard::Unc => "unc",
            ThreadStandard::Unf => "unf",
            ThreadStandard::Print45 => "print45",
        }
    }

    /// What the Thread box calls it.
    pub fn label(self) -> &'static str {
        match self {
            ThreadStandard::Metric => "ISO metric",
            ThreadStandard::Unc => "Unified coarse (UNC)",
            ThreadStandard::Unf => "Unified fine (UNF)",
            ThreadStandard::Print45 => "3D printing — 45° flanks",
        }
    }

    /// The standard of an id, or `None` for one there is not.
    pub fn from_id(id: &str) -> Option<ThreadStandard> {
        THREAD_STANDARDS.into_iter().find(|s| s.id() == id)
    }

    fn index(self) -> usize {
        match self {
            ThreadStandard::Metric => 0,
            ThreadStandard::Unc => 1,
            ThreadStandard::Unf => 2,
            ThreadStandard::Print45 => 3,
        }
    }
}

/// A thread's basic profile, its lengths as shares of the pitch: how far the flanks lean
/// from the radius — the tangent and the secant of that angle — how deep the profile is from
/// the major diameter to the minor, and the flats of a bolt's tooth, its crest on the major
/// diameter and the gap's floor, its root, on the minor. A nut's tooth is the bolt's gap:
/// its crest as wide as the bolt's root, its root as the bolt's crest. A crest, a root and
/// the two flanks between them make the pitch: crest + root + 2 × depth × tan = 1.
#[derive(Debug, Clone, Copy)]
struct ThreadForm {
    tan: f64,
    sec: f64,
    depth: f64,
    crest: f64,
    root: f64,
}

/// The 60° form of ISO 68-1 and ASME B1.1: the flanks at 30° to the radius, five eighths of
/// the sharp vee's √3/2 deep, a bolt's crest an eighth of the pitch wide and its root a
/// quarter.
fn v60() -> ThreadForm {
    ThreadForm { tan: 1.0 / 3f64.sqrt(), sec: 2.0 / 3f64.sqrt(), depth: (5.0 / 8.0) * (3f64.sqrt() / 2.0), crest: 1.0 / 8.0, root: 1.0 / 4.0 }
}

/// The 45° form for printing: the flanks at 45° to the radius, the crest and the root a
/// fifth of the pitch wide each, and so 0.3 of it deep. The tooth and the gap are each half
/// the pitch wide halfway down.
fn print45() -> ThreadForm {
    ThreadForm { tan: 1.0, sec: std::f64::consts::SQRT_2, depth: 0.3, crest: 1.0 / 5.0, root: 1.0 / 5.0 }
}

fn form(standard: ThreadStandard) -> ThreadForm {
    match standard {
        ThreadStandard::Metric | ThreadStandard::Unc | ThreadStandard::Unf => v60(),
        ThreadStandard::Print45 => print45(),
    }
}

/// A size of a standard.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThreadSize {
    /// What the size is called: M16, 1/4, #10, Ø16.
    pub label: String,
    /// The nominal — major — diameter, millimetres.
    pub diameter: f64,
    /// The pitches it is made in, millimetres, the standard's own first: a metric size's
    /// coarse pitch, then its fine ones; a printing size's pitch as deep as the ISO coarse
    /// thread, then the ISO coarse pitch.
    pub pitches: Vec<f64>,
}

/// ISO 261, the sizes of the first and second choice up to M64: the diameter and its
/// pitches, coarse first.
const METRIC: [(f64, &[f64]); 33] = [
    (1.0, &[0.25, 0.2]),
    (1.2, &[0.25, 0.2]),
    (1.4, &[0.3, 0.2]),
    (1.6, &[0.35, 0.2]),
    (2.0, &[0.4, 0.25]),
    (2.5, &[0.45, 0.35]),
    (3.0, &[0.5, 0.35]),
    (3.5, &[0.6, 0.35]),
    (4.0, &[0.7, 0.5]),
    (5.0, &[0.8, 0.5]),
    (6.0, &[1.0, 0.75]),
    (7.0, &[1.0, 0.75]),
    (8.0, &[1.25, 1.0, 0.75]),
    (10.0, &[1.5, 1.25, 1.0, 0.75]),
    (12.0, &[1.75, 1.5, 1.25, 1.0]),
    (14.0, &[2.0, 1.5, 1.25, 1.0]),
    (16.0, &[2.0, 1.5, 1.0]),
    (18.0, &[2.5, 2.0, 1.5, 1.0]),
    (20.0, &[2.5, 2.0, 1.5, 1.0]),
    (22.0, &[2.5, 2.0, 1.5, 1.0]),
    (24.0, &[3.0, 2.0, 1.5, 1.0]),
    (27.0, &[3.0, 2.0, 1.5, 1.0]),
    (30.0, &[3.5, 3.0, 2.0, 1.5, 1.0]),
    (33.0, &[3.5, 3.0, 2.0, 1.5]),
    (36.0, &[4.0, 3.0, 2.0, 1.5]),
    (39.0, &[4.0, 3.0, 2.0, 1.5]),
    (42.0, &[4.5, 4.0, 3.0, 2.0, 1.5]),
    (45.0, &[4.5, 4.0, 3.0, 2.0, 1.5]),
    (48.0, &[5.0, 4.0, 3.0, 2.0, 1.5]),
    (52.0, &[5.0, 4.0, 3.0, 2.0, 1.5]),
    (56.0, &[5.5, 4.0, 3.0, 2.0, 1.5]),
    (60.0, &[5.5, 4.0, 3.0, 2.0, 1.5]),
    (64.0, &[6.0, 4.0, 3.0, 2.0, 1.5]),
];

/// ASME B1.1: the size, its major diameter in inches, threads per inch.
const UNC: [(&str, f64, f64); 25] = [
    ("#1", 0.073, 64.0),
    ("#2", 0.086, 56.0),
    ("#3", 0.099, 48.0),
    ("#4", 0.112, 40.0),
    ("#5", 0.125, 40.0),
    ("#6", 0.138, 32.0),
    ("#8", 0.164, 32.0),
    ("#10", 0.19, 24.0),
    ("#12", 0.216, 24.0),
    ("1/4", 0.25, 20.0),
    ("5/16", 0.3125, 18.0),
    ("3/8", 0.375, 16.0),
    ("7/16", 0.4375, 14.0),
    ("1/2", 0.5, 13.0),
    ("9/16", 0.5625, 12.0),
    ("5/8", 0.625, 11.0),
    ("3/4", 0.75, 10.0),
    ("7/8", 0.875, 9.0),
    ("1", 1.0, 8.0),
    ("1-1/8", 1.125, 7.0),
    ("1-1/4", 1.25, 7.0),
    ("1-3/8", 1.375, 6.0),
    ("1-1/2", 1.5, 6.0),
    ("1-3/4", 1.75, 5.0),
    ("2", 2.0, 4.5),
];

const UNF: [(&str, f64, f64); 24] = [
    ("#0", 0.06, 80.0),
    ("#1", 0.073, 72.0),
    ("#2", 0.086, 64.0),
    ("#3", 0.099, 56.0),
    ("#4", 0.112, 48.0),
    ("#5", 0.125, 44.0),
    ("#6", 0.138, 40.0),
    ("#8", 0.164, 36.0),
    ("#10", 0.19, 32.0),
    ("#12", 0.216, 28.0),
    ("1/4", 0.25, 28.0),
    ("5/16", 0.3125, 24.0),
    ("3/8", 0.375, 24.0),
    ("7/16", 0.4375, 20.0),
    ("1/2", 0.5, 20.0),
    ("9/16", 0.5625, 18.0),
    ("5/8", 0.625, 18.0),
    ("3/4", 0.75, 16.0),
    ("7/8", 0.875, 14.0),
    ("1", 1.0, 12.0),
    ("1-1/8", 1.125, 12.0),
    ("1-1/4", 1.25, 12.0),
    ("1-3/8", 1.375, 12.0),
    ("1-1/2", 1.5, 12.0),
];

const INCH: f64 = 25.4;

/// The smallest diameter the printing form is offered at, millimetres.
const PRINT_FROM: f64 = 3.0;

/// The printing form's pitch for an ISO coarse one: the pitch it cuts as deep at — 0.54 of
/// the pitch deep in the 60° form, 0.3 in the 45° one, so 1.8 times as long — rounded to a
/// quarter of a millimetre under 3 mm and to a half from 3 mm up. As deep as the coarse
/// thread, it keeps ISO's minor diameter: the core is as strong, and a bore drilled to the
/// tap drill's size is taken for the same size ([`size_for_face`]).
fn print_pitch(coarse: f64) -> f64 {
    let p = (coarse * v60().depth) / print45().depth;
    let step = if p < 3.0 { 0.25 } else { 0.5 };
    js_round(p / step) * step
}

fn sizes() -> &'static [Vec<ThreadSize>; 4] {
    static SIZES: OnceLock<[Vec<ThreadSize>; 4]> = OnceLock::new();
    SIZES.get_or_init(|| {
        let metric = METRIC.iter().map(|&(d, pitches)| ThreadSize { label: format!("M{}", number_text(d)), diameter: d, pitches: pitches.to_vec() }).collect();
        let unified = |table: &[(&str, f64, f64)]| table.iter().map(|&(label, d, tpi)| ThreadSize { label: label.to_string(), diameter: d * INCH, pitches: vec![INCH / tpi] }).collect::<Vec<_>>();
        // The ISO metric diameters from 3 mm, at the pitch as deep as their coarse thread
        // and then, finer and shallower, the coarse pitch itself.
        let print = METRIC
            .iter()
            .filter(|(d, _)| *d >= PRINT_FROM)
            .filter_map(|&(d, pitches)| pitches.first().map(|&coarse| ThreadSize { label: format!("Ø{}", number_text(d)), diameter: d, pitches: vec![print_pitch(coarse), coarse] }))
            .collect();
        [metric, unified(&UNC), unified(&UNF), print]
    })
}

/// The sizes of a standard, smallest first.
pub fn thread_sizes(standard: ThreadStandard) -> &'static [ThreadSize] {
    &sizes()[standard.index()]
}

/// How close a typed number has to be to a table's to be that size.
const SAME_MM: f64 = 1e-4;

/// The size of the standard a diameter is, or `None` for one it does not have.
pub fn size_of(standard: ThreadStandard, diameter: f64) -> Option<&'static ThreadSize> {
    thread_sizes(standard).iter().find(|s| (s.diameter - diameter).abs() < SAME_MM)
}

/// Threads per inch for a pitch in millimetres.
pub fn threads_per_inch(pitch: f64) -> f64 {
    INCH / pitch
}

/// A number in a thread's name: to four decimals, as JavaScript writes it.
fn trim(v: f64) -> String {
    number_text(js_round(v * 10000.0) / 10000.0)
}

/// What a thread is called — M16×2, 1/4-20 UNC, Ø16×3.5 print — or `None` for a diameter
/// and pitch the standard does not have together.
pub fn thread_designation(standard: ThreadStandard, diameter: f64, pitch: f64) -> Option<String> {
    let size = size_of(standard, diameter)?;
    if !size.pitches.iter().any(|p| (p - pitch).abs() < SAME_MM) {
        return None;
    }
    Some(match standard {
        ThreadStandard::Metric => format!("{}×{}", size.label, trim(pitch)),
        ThreadStandard::Print45 => format!("{}×{} print", size.label, trim(pitch)),
        ThreadStandard::Unc | ThreadStandard::Unf => {
            format!("{}-{} {}", size.label, trim(js_round(threads_per_inch(pitch) * 100.0) / 100.0), standard.id().to_uppercase())
        }
    })
}

/// A thread named as it reads in a row: its designation, or its diameter and pitch where it
/// is not a standard one — marked "print" in the printing form, so a row always says which
/// form it is cut in.
pub fn thread_name(standard: ThreadStandard, diameter: f64, pitch: f64) -> String {
    thread_designation(standard, diameter, pitch).unwrap_or_else(|| {
        format!(
            "Ø{} × {}{}",
            trim(js_round(diameter * 1000.0) / 1000.0),
            trim(js_round(pitch * 1000.0) / 1000.0),
            if standard == ThreadStandard::Print45 { " print" } else { "" }
        )
    })
}

/// How deep a standard's basic profile is, from the major diameter to the minor: five
/// eighths of the sharp vee's √3/2 of the pitch in the 60° form, 0.3 of the pitch in the 45°
/// one.
pub fn thread_depth(pitch: f64, standard: ThreadStandard) -> f64 {
    form(standard).depth * pitch
}

/// The basic profile's minor diameter — a bolt's root, a nut's crest.
pub fn minor_diameter(diameter: f64, pitch: f64, standard: ThreadStandard) -> f64 {
    diameter - 2.0 * thread_depth(pitch, standard)
}

/// The diameter the flanks are half tooth and half gap at, where the threads bear on each
/// other: as far in from the major diameter as the flank takes to widen a bolt's crest to
/// half the pitch.
pub fn pitch_diameter(diameter: f64, pitch: f64, standard: ThreadStandard) -> f64 {
    let f = form(standard);
    diameter - (pitch / 2.0 - f.crest * pitch) / f.tan
}

/// What a thread is cut to: the numbers the Thread feature holds, and the standard whose
/// form it is cut in.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThreadSpec {
    pub standard: ThreadStandard,
    pub diameter: f64,
    pub pitch: f64,
    /// Millimetres every face of the thread is moved into the material.
    pub offset: f64,
}

/// Everything the kernel cuts a thread to, on the face it is given.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThreadCut {
    #[serde(flatten)]
    pub spec: ThreadSpec,
    pub left_handed: bool,
    /// Millimetres along the face; `None` runs the whole face.
    pub length: Option<f64>,
    pub start: f64,
    pub flip: bool,
    /// A 45° chamfer at each end the thread runs out of the body free.
    #[serde(default)]
    pub chamfer: bool,
}

/// A round face's diameter as drawn against the one its thread is cut to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FaceAgainst {
    Wider,
    Narrower,
}

/// A round face's diameter as drawn against the one its thread is cut to — a boss's against
/// the major, a bore's against the minor: wider or narrower by more than the thread's depth,
/// half of it a side, a scan's or a print's slack; `None` within it.
pub fn face_against_thread(t: &ThreadSpec, internal: bool, face_diameter: f64) -> Option<FaceAgainst> {
    let slack = thread_depth(t.pitch, t.standard);
    let own = if internal { minor_diameter(t.diameter, t.pitch, t.standard) } else { t.diameter };
    if face_diameter > own + slack {
        Some(FaceAgainst::Wider)
    } else if face_diameter < own - slack {
        Some(FaceAgainst::Narrower)
    } else {
        None
    }
}

/// The offset at which the tooth has no thickness left at the pitch diameter — refused
/// there; the most it takes is just under. There the tooth is half the pitch wide, and the
/// offset narrows it on each side by itself over the cosine of the flank's angle: a quarter
/// of the pitch times that cosine leaves nothing.
pub fn max_thread_offset(pitch: f64, standard: ThreadStandard) -> f64 {
    pitch / (4.0 * form(standard).sec)
}

/// How near a bore's diameter has to be to a size's major diameter to be taken for drawn at
/// it: a number typed, not a scan's reading.
const DRAWN_AT_SIZE_MM: f64 = 0.01;

/// A diameter and the pitch it is cut at.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ThreadChoice {
    pub diameter: f64,
    pub pitch: f64,
}

/// The size a round face most likely carries a thread of, in a standard: a boss the size
/// whose major diameter is nearest its own; a hole drawn at a size's own diameter — Ø5 for
/// M5, as a part is designed — that size, and any other the one whose minor diameter — the
/// tap drill's — is nearest; at the standard's own pitch.
pub fn size_for_face(standard: ThreadStandard, face_diameter: f64, internal: bool) -> ThreadChoice {
    let table = thread_sizes(standard);
    let first_pitch = |s: &ThreadSize| s.pitches.first().copied().unwrap_or(f64::NAN);
    if internal {
        if let Some(drawn) = table.iter().find(|s| (s.diameter - face_diameter).abs() < DRAWN_AT_SIZE_MM) {
            return ThreadChoice { diameter: drawn.diameter, pitch: first_pitch(drawn) };
        }
    }
    let mut best = &table[0];
    let mut best_off = f64::INFINITY;
    for s in table {
        let own = if internal { minor_diameter(s.diameter, first_pitch(s), standard) } else { s.diameter };
        let off = (own - face_diameter).abs();
        if off < best_off {
            best = s;
            best_off = off;
        }
    }
    ThreadChoice { diameter: best.diameter, pitch: first_pitch(best) }
}

/// A length in a message: to a micron in millimetres, with no zeros behind it.
fn length_words(v: f64, units: DisplayUnits) -> String {
    let k = 10f64.powi(length_decimals(3, units));
    format!("{} {}", number_text(js_round(to_display(v, units) * k) / k), unit_word(units, 1))
}

/// Why a thread cannot be cut, read off its numbers alone — or `None`. The face it is on has
/// its say in the kernel. Its figures in `units`: the editor's in the units shown, a
/// resolve's or a command's in millimetres.
pub fn thread_problem(t: &ThreadSpec, units: DisplayUnits) -> Option<String> {
    let mm = |v: f64| length_words(v, units);
    if !(t.diameter > 0.0) {
        return Some("The thread needs a diameter.".to_string());
    }
    if !(t.pitch > 0.0) {
        return Some("The thread needs a pitch.".to_string());
    }
    if !(t.offset >= 0.0) {
        return Some("The offset only takes material away — it has to be zero or more.".to_string());
    }
    let max = max_thread_offset(t.pitch, t.standard);
    if t.offset >= max {
        return Some(format!(
            "An offset of {} leaves the tooth no thickness at the pitch diameter — under {} for a pitch of {}.",
            mm(t.offset),
            mm(max),
            mm(t.pitch)
        ));
    }
    if t.diameter / 2.0 - thread_depth(t.pitch, t.standard) - t.offset < t.pitch / 4.0 {
        return Some(format!("A pitch of {} is too coarse for Ø{} — the thread would reach the axis.", mm(t.pitch), mm(t.diameter)));
    }
    None
}

/// Why a Thread feature cannot be resolved into a step for the kernel — or `None`: its face
/// not a round one (`round_face`, the fingerprint a cylinder's), its numbers
/// ([`thread_problem`], in millimetres), a length of nothing, or a start off the face. What
/// ScanRuler's resolve (`resolve.ts`, the `thread` feature) fails the feature with, before
/// the kernel is asked; the kernel then has its say on the face (`thread_cut`).
pub fn thread_feature_problem(cut: &ThreadCut, round_face: bool) -> Option<String> {
    if !round_face {
        return Some("A thread is cut on a round face — the face picked is not one.".to_string());
    }
    if let Some(why) = thread_problem(&cut.spec, DisplayUnits::Mm) {
        return Some(why);
    }
    match cut.length {
        Some(length) if !(length > 0.0) => Some("The thread needs a length.".to_string()),
        Some(_) if !(cut.start >= 0.0) => Some("The thread has to start on the face — zero or more in from its end.".to_string()),
        _ => None,
    }
}

/// How small a crest's flat may get before it is taken for a point.
const POINT_FLAT: f64 = 1e-4;
/// How near the face a crest's flat may come, as a share of the pitch, before the face is
/// taken for the crest — a flat a hair inside the face would leave a sliver of the face
/// standing on it.
pub(crate) const CREST_ON_FACE: f64 = 0.01;

/// One tooth of a thread in a plane through its axis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToothOutline {
    /// The outline, counter-clockwise: each point's `x` millimetres from the axis (r), its
    /// `y` along it from the tooth's middle (z).
    pub points: Vec<DVec2>,
    /// The radius of the crest's flat, or of its point.
    pub crest: f64,
    /// The radius of the root, where the band the thread is cut in ends.
    pub root: f64,
    /// Whether the crest came to a point under the offset.
    pub pointed: bool,
}

/// The outline of one tooth of the thread in a plane through the axis — the material that
/// stays, with every face moved in by the offset — and the radius of the root, where the
/// band the thread is cut in ends: a bolt's band runs from it out past the face, a nut's
/// from inside the bore out to it.
///
/// A bolt's tooth: its flat crest at the major diameter, the gap's floor at the minor — an
/// eighth and a quarter of the pitch wide in the 60° form, a fifth each in the 45° one; a
/// nut's the other way round — its crest at the minor as wide as the bolt's gap floor, its
/// gap's floor at the major as the bolt's crest — the flanks between at the form's angle to
/// the radius, 30° or 45°. Counter-clockwise in (r, z).
///
/// The offset moves each flank square to itself: at any radius the tooth is narrower on each
/// side by the offset over the flank's cosine. The crest and the root move by the offset
/// too, towards the tooth's root, where the flanks stand wider apart by the offset times
/// their tangent each — so the tooth's half width at its crest and at its root changes by the
/// offset times (tan − sec), narrower, and the crest comes to a point once that has used up
/// its flat.
///
/// Two surfaces of the tooth must not lie on one of the band or of the body: the kernel's
/// booleans fail on a helix lying in a cylinder. So the flanks run on a little past the root
/// before the tooth turns straight into what stands behind it — `reach` below a bolt's core,
/// past a nut's wall — and the band's own cylinder makes the gap's floor. And where the
/// crest's flat would lie on the face of radius `face`, or beyond it, the tooth is drawn up
/// to the point its flanks meet, past the face: the face makes the crest.
pub fn tooth_outline(t: &ThreadSpec, internal: bool, reach: f64, face: Option<f64>) -> ToothOutline {
    let p = t.pitch;
    let c = t.offset;
    let big_r = t.diameter / 2.0;
    let ThreadForm { tan, sec, depth, crest: crest_flat, root: root_flat } = form(t.standard);
    let h = depth * p;
    // What the offset does to the tooth's half width at its crest and root.
    let shrink = c * tan - c * sec;
    // Along the flank, the half width grows by tan of the way it goes into the tooth's root.
    let past = (0.05 * p).min(reach / 2.0);
    let pt = DVec2::new;
    if !internal {
        let root = big_r - h - c;
        let root_half = ((1.0 - root_flat) * p) / 2.0 + shrink;
        let crest_half = (crest_flat * p) / 2.0 + shrink;
        let pointed = crest_half < POINT_FLAT * p;
        // Where the flanks meet, past the crest's flat or short of it.
        let apex = big_r - c + crest_half / tan;
        let crest = if pointed { apex } else { big_r - c };
        let to_apex = pointed || face.is_some_and(|f| crest >= f - CREST_ON_FACE * p);
        let bend = root - past;
        let bend_half = root_half + past * tan;
        let base = root - reach;
        let mut points = vec![pt(base, -bend_half), pt(bend, -bend_half)];
        if to_apex {
            points.push(pt(apex, 0.0));
        } else {
            points.push(pt(crest, -crest_half));
            points.push(pt(crest, crest_half));
        }
        points.push(pt(bend, bend_half));
        points.push(pt(base, bend_half));
        return ToothOutline { points, crest, root, pointed };
    }
    let minor = big_r - h;
    let root = big_r + c;
    let root_half = ((1.0 - crest_flat) * p) / 2.0 + shrink;
    let crest_half = (root_flat * p) / 2.0 + shrink;
    let pointed = crest_half < POINT_FLAT * p;
    let apex = minor + c - crest_half / tan;
    let crest = if pointed { apex } else { minor + c };
    let to_apex = pointed || face.is_some_and(|f| crest <= f + CREST_ON_FACE * p);
    let bend = root + past;
    let bend_half = root_half + past * tan;
    let outer = root + reach;
    let mut points = Vec::with_capacity(6);
    points.push(if to_apex { pt(apex, 0.0) } else { pt(crest, -crest_half) });
    points.extend([pt(bend, -bend_half), pt(outer, -bend_half), pt(outer, bend_half), pt(bend, bend_half)]);
    if !to_apex {
        points.push(pt(crest, crest_half));
    }
    ToothOutline { points, crest, root, pointed }
}

/// How far out from the axis material is added before a thread is cut, or `None` for none.
/// A thread leaves its face at its crests — a bore at the minor diameter, a boss at the
/// major, moved into the material by the offset — whatever size the face was drawn: a bore
/// narrower than that or a boss wider loses the difference to the cut itself, and a bore
/// wider or a boss narrower is first filled in or built up to it along the thread. So a hole
/// drawn Ø5 takes an M5 thread, or an M4 or an M6, and a Ø4 pin an M5.
///
/// The fill ends on the crests, so they are a piece of its cylinder, as the face's own are
/// where it was drawn at them ([`tooth_outline`]). A face within a hair of them is taken for
/// them: a fill that thin would be a sliver. Where the crests come to a point, it runs a
/// quarter of the pitch past them, for the cut to take away — their points would lie on its
/// cylinder.
pub fn fill_radius(t: &ThreadSpec, internal: bool, face_r: f64) -> Option<f64> {
    let ToothOutline { crest, pointed, .. } = tooth_outline(t, internal, 0.0, None);
    let near = CREST_ON_FACE * t.pitch;
    let at_crests = if internal { face_r <= crest + near } else { face_r >= crest - near };
    if at_crests {
        return None;
    }
    if !pointed {
        return Some(crest);
    }
    Some(if internal { (crest - t.pitch / 4.0).max(crest / 2.0) } else { crest + t.pitch / 4.0 })
}

/// How far past the root a thread's chamfer meets the end face, as a share of the pitch.
const CHAMFER_PAST_ROOT: f64 = 0.1;

/// The radius a thread's 45° chamfer meets the end of the face at: a tenth of the pitch
/// inside a bolt's root, outside a nut's — the chamfer a bolt's end is given down past the
/// minor diameter, a tapped hole's countersink out past the major. From there it runs into
/// the body at 45° until it is clear of the crests, so the thread starts on the cone instead
/// of a turn as thin as a knife's edge, and the chamfer never lies along the root.
pub fn chamfer_radius(t: &ThreadSpec, internal: bool) -> f64 {
    let root = tooth_outline(t, internal, 0.0, None).root;
    if internal {
        root + CHAMFER_PAST_ROOT * t.pitch
    } else {
        root - CHAMFER_PAST_ROOT * t.pitch
    }
}

/// A body's tessellation as the kernel hands it over: its triangles, each tagged with the
/// B-rep face it lies on (an mm-reverse `BodyMesh`'s `positions`, `triangles`, `tri_face`).
#[derive(Debug, Clone, Copy)]
pub struct FaceMesh<'a> {
    pub positions: &'a [DVec3],
    pub triangles: &'a [[u32; 3]],
    /// Each triangle's face; empty where the mesh does not say.
    pub tri_face: &'a [u32],
}

/// Whether a round face looks away from its axis — a boss, a bolt's thread — or towards it —
/// a bore, a nut's — read off its body's mesh: the triangles on the face, by its id, wound
/// outward. `None` when the mesh has none of the face.
pub fn round_face_is_boss(mesh: FaceMesh<'_>, face: u32, center: DVec3, axis: DVec3) -> Option<bool> {
    let mut vote = 0.0;
    for (t, &f) in mesh.tri_face.iter().enumerate() {
        if f != face {
            continue;
        }
        let Some(&[ia, ib, ic]) = mesh.triangles.get(t) else { continue };
        let (Some(&a), Some(&b), Some(&c)) = (mesh.positions.get(ia as usize), mesh.positions.get(ib as usize), mesh.positions.get(ic as usize)) else { continue };
        let u = b - a;
        let v = c - a;
        let n = DVec3::new(u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x);
        let m = DVec3::new((a.x + b.x + c.x) / 3.0 - center.x, (a.y + b.y + c.y) / 3.0 - center.y, (a.z + b.z + c.z) / 3.0 - center.z);
        let along = m.x * axis.x + m.y * axis.y + m.z * axis.z;
        let radial = m - along * axis;
        vote += n.x * radial.x + n.y * radial.y + n.z * radial.z;
    }
    if vote == 0.0 {
        None
    } else {
        Some(vote > 0.0)
    }
}

/// A round face a thread is to go on, as the Thread box holds it: the face's id on the body's
/// mesh (`None` while it names none), and its fingerprint's centre, radius and axis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RoundFacePick {
    pub face: Option<u32>,
    pub center: Option<DVec3>,
    pub radius: Option<f64>,
    pub dir: DVec3,
}

/// What a thread's round face says once it names a face of the body's mesh.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ThreadFaceReading {
    /// A bore.
    pub internal: bool,
    /// The standard's size for the face as drawn, while none has been chosen.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diameter: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pitch: Option<f64>,
}

/// What a thread's round face says once it names a face of the body's mesh: a boss or a bore
/// — `internal` for a bore — read off the face's triangles, and, while no size has been
/// chosen or typed (`sized`), the standard's size for it as the face is drawn: a boss's
/// major diameter, a bore's minor. `None` while the mesh does not say.
pub fn thread_face_reading(mesh: FaceMesh<'_>, face: &RoundFacePick, standard: ThreadStandard, sized: bool) -> Option<ThreadFaceReading> {
    let id = face.face?;
    let center = face.center?;
    // A radius of nothing, or none, names no round face.
    let radius = face.radius.filter(|r| *r != 0.0 && !r.is_nan())?;
    let boss = round_face_is_boss(mesh, id, center, face.dir)?;
    if sized {
        return Some(ThreadFaceReading { internal: !boss, diameter: None, pitch: None });
    }
    let size = size_for_face(standard, 2.0 * radius, !boss);
    Some(ThreadFaceReading { internal: !boss, diameter: Some(size.diameter), pitch: Some(size.pitch) })
}

/// Where along a round face a thread runs, millimetres along its axis.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ThreadSpan {
    pub from: f64,
    pub to: f64,
}

/// Where along a round face a thread runs, millimetres along its axis: the whole face
/// (`lo` to `hi`), or `length` from `start` in from one end — the far end along the axis, or
/// with `flip` the near one. `None` for one that does not fit on the face.
pub fn thread_span(lo: f64, hi: f64, length: Option<f64>, start: f64, flip: bool) -> Option<ThreadSpan> {
    let Some(length) = length else { return Some(ThreadSpan { from: lo, to: hi }) };
    let slack = 1e-6;
    if start < -slack || start + length > hi - lo + slack {
        return None;
    }
    Some(if flip { ThreadSpan { from: lo + start, to: lo + start + length } } else { ThreadSpan { from: hi - start - length, to: hi - start } })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{PI, SQRT_2};

    use ThreadStandard::{Metric, Print45, Unc, Unf};

    /// vitest's `toBeCloseTo(want, digits)`.
    #[track_caller]
    fn close(got: f64, want: f64, digits: i32) {
        let tol = 10f64.powi(-digits) / 2.0;
        assert!((got - want).abs() < tol, "{got} is not within {tol} of {want}");
    }

    fn spec(standard: ThreadStandard, diameter: f64, pitch: f64, offset: f64) -> ThreadSpec {
        ThreadSpec { standard, diameter, pitch, offset }
    }

    /// The area of a polygon in (r, z) and the radius of its centroid.
    fn area_and_radius(points: &[DVec2]) -> (f64, f64) {
        let mut a = 0.0;
        let mut cr = 0.0;
        for i in 0..points.len() {
            let (r0, z0) = (points[i].x, points[i].y);
            let q = points[(i + 1) % points.len()];
            let (r1, z1) = (q.x, q.y);
            let cross = r0 * z1 - r1 * z0;
            a += cross;
            cr += (r0 + r1) * cross;
        }
        ((a / 2.0).abs(), cr / (3.0 * a))
    }

    /// The part of a tooth's outline beyond a radius — a bolt's above its core, a nut's
    /// inside its wall — as a polygon.
    fn clip_outline(points: &[DVec2], at: f64, keep_above: bool) -> Vec<DVec2> {
        let inside = |p: DVec2| if keep_above { p.x >= at - 1e-12 } else { p.x <= at + 1e-12 };
        let mut out = Vec::new();
        for i in 0..points.len() {
            let a = points[i];
            let b = points[(i + 1) % points.len()];
            if inside(a) {
                out.push(a);
            }
            if inside(a) != inside(b) {
                let t = (at - a.x) / (b.x - a.x);
                out.push(DVec2::new(at, a.y + (b.y - a.y) * t));
            }
        }
        out
    }

    /// How wide a tooth's outline is at a radius.
    fn width_at(points: &[DVec2], r: f64) -> f64 {
        let zs: Vec<f64> = clip_outline(points, r, false).into_iter().filter(|p| (p.x - r).abs() < 1e-9).map(|p| p.y).collect();
        zs.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - zs.iter().cloned().fold(f64::INFINITY, f64::min)
    }

    fn max_abs_z(points: &[DVec2]) -> f64 {
        points.iter().map(|p| p.y.abs()).fold(f64::NEG_INFINITY, f64::max)
    }

    fn max_r(points: &[DVec2]) -> f64 {
        points.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max)
    }

    fn min_r(points: &[DVec2]) -> f64 {
        points.iter().map(|p| p.x).fold(f64::INFINITY, f64::min)
    }

    #[test]
    fn knows_the_standards_sizes_and_names_them() {
        assert_eq!(size_of(Metric, 16.0).map(|s| s.pitches[0]), Some(2.0));
        assert_eq!(thread_designation(Metric, 16.0, 2.0).as_deref(), Some("M16×2"));
        assert_eq!(thread_designation(Metric, 16.0, 1.5).as_deref(), Some("M16×1.5"));
        assert_eq!(thread_designation(Metric, 16.0, 1.75), None);
        assert_eq!(thread_designation(Unc, 6.35, 25.4 / 20.0).as_deref(), Some("1/4-20 UNC"));
        assert_eq!(thread_designation(Unf, 6.35, 25.4 / 28.0).as_deref(), Some("1/4-28 UNF"));
        assert_eq!(thread_designation(Unc, 4.826, 25.4 / 24.0).as_deref(), Some("#10-24 UNC"));
        assert_eq!(thread_name(Metric, 15.5, 1.2), "Ø15.5 × 1.2");
        // The printing form says so, in a size of its own or not.
        assert_eq!(thread_designation(Print45, 10.0, 2.75).as_deref(), Some("Ø10×2.75 print"));
        assert_eq!(thread_designation(Print45, 10.0, 1.5).as_deref(), Some("Ø10×1.5 print"));
        assert_eq!(thread_designation(Print45, 10.0, 2.0), None);
        assert_eq!(thread_name(Print45, 10.0, 2.4), "Ø10 × 2.4 print");
        assert_eq!(thread_name(Print45, 3.5, 1.0), "Ø3.5×1 print");
        assert_eq!(THREAD_STANDARDS.map(|s| s.id()), ["metric", "unc", "unf", "print45"]);
        for id in THREAD_STANDARDS {
            let sizes = thread_sizes(id);
            for i in 1..sizes.len() {
                assert!(sizes[i].diameter > sizes[i - 1].diameter);
            }
        }
    }

    #[test]
    fn sizes_the_printing_form_on_the_iso_metric_diameters_from_3_mm_as_deep_as_their_coarse_thread() {
        // The pitch as deep as the ISO coarse thread first, rounded to a quarter of a
        // millimetre under 3 mm and a half from 3 mm up; then the ISO coarse pitch.
        let table: Vec<(String, Vec<f64>)> = thread_sizes(Print45).iter().map(|s| (s.label.clone(), s.pitches.clone())).collect();
        let want: Vec<(&str, [f64; 2])> = vec![
            ("Ø3", [1.0, 0.5]),
            ("Ø3.5", [1.0, 0.6]),
            ("Ø4", [1.25, 0.7]),
            ("Ø5", [1.5, 0.8]),
            ("Ø6", [1.75, 1.0]),
            ("Ø7", [1.75, 1.0]),
            ("Ø8", [2.25, 1.25]),
            ("Ø10", [2.75, 1.5]),
            ("Ø12", [3.0, 1.75]),
            ("Ø14", [3.5, 2.0]),
            ("Ø16", [3.5, 2.0]),
            ("Ø18", [4.5, 2.5]),
            ("Ø20", [4.5, 2.5]),
            ("Ø22", [4.5, 2.5]),
            ("Ø24", [5.5, 3.0]),
            ("Ø27", [5.5, 3.0]),
            ("Ø30", [6.5, 3.5]),
            ("Ø33", [6.5, 3.5]),
            ("Ø36", [7.0, 4.0]),
            ("Ø39", [7.0, 4.0]),
            ("Ø42", [8.0, 4.5]),
            ("Ø45", [8.0, 4.5]),
            ("Ø48", [9.0, 5.0]),
            ("Ø52", [9.0, 5.0]),
            ("Ø56", [10.0, 5.5]),
            ("Ø60", [10.0, 5.5]),
            ("Ø64", [11.0, 6.0]),
        ];
        assert_eq!(table.len(), want.len());
        for ((label, pitches), (wl, wp)) in table.iter().zip(&want) {
            assert_eq!(label, wl);
            assert_eq!(pitches.as_slice(), wp.as_slice(), "{label}");
        }
        // Each keeps its ISO coarse minor diameter to within the rounding.
        for s in thread_sizes(Print45) {
            let iso = minor_diameter(s.diameter, size_of(Metric, s.diameter).map(|m| m.pitches[0]).unwrap_or(f64::NAN), Metric);
            assert!((minor_diameter(s.diameter, s.pitches[0], Print45) - iso).abs() < 0.16, "{}", s.label);
            assert_eq!(thread_problem(&spec(Print45, s.diameter, s.pitches[0], 0.0), DisplayUnits::Mm), None, "{}", s.label);
        }
        // Ø10: 8.35 against M10's 8.376.
        close(minor_diameter(10.0, 2.75, Print45), minor_diameter(10.0, 1.5, Metric), 1);
        assert!(size_of(Print45, 2.5).is_none());
    }

    #[test]
    fn reads_the_profiles_diameters_off_the_pitch_as_iso_68_1_has_them() {
        // M16×2: minor 13.835, pitch diameter 14.701.
        close(minor_diameter(16.0, 2.0, Metric), 13.835, 3);
        close(pitch_diameter(16.0, 2.0, Metric), 14.701, 3);
        close(thread_depth(1.0, Metric), 0.5413, 4);
        close(max_thread_offset(2.0, Metric), (3f64.sqrt() / 8.0) * 2.0, 12);
        // The 45° form: 0.3 of the pitch deep, the pitch diameter halfway down.
        close(thread_depth(1.0, Print45), 0.3, 12);
        close(minor_diameter(10.0, 2.75, Print45), 10.0 - 0.6 * 2.75, 12);
        close(pitch_diameter(10.0, 2.75, Print45), 10.0 - 0.3 * 2.75, 12);
        close(max_thread_offset(2.75, Print45), (SQRT_2 / 8.0) * 2.75, 12);
    }

    #[test]
    fn picks_the_size_a_boss_or_a_bore_most_likely_carries() {
        let choice = |d: f64, p: f64| ThreadChoice { diameter: d, pitch: p };
        assert_eq!(size_for_face(Metric, 15.9, false), choice(16.0, 2.0));
        // A bore drilled for M8: the tap drill's 6.8 mm, near the minor of 6.65.
        assert_eq!(size_for_face(Metric, 6.8, true), choice(8.0, 1.25));
        let quarter = size_for_face(Unc, 6.3, false);
        close(quarter.diameter, 6.35, 6);
        close(25.4 / quarter.pitch, 20.0, 6);
        // The printing form keeps ISO's minor diameter, so a bore drilled for M10 — the tap
        // drill's 8.5 mm — is taken for its Ø10.
        assert_eq!(size_for_face(Metric, 8.5, true), choice(10.0, 1.5));
        assert_eq!(size_for_face(Print45, 8.5, true), choice(10.0, 2.75));
        assert_eq!(size_for_face(Print45, 6.8, true), choice(8.0, 2.25));
        assert_eq!(size_for_face(Print45, 9.9, false), choice(10.0, 2.75));
        // A bore drawn at a size's own diameter is taken for that size: a Ø5 hole for M5,
        // not the M6 its tap drill would be; Ø14 for M14, not M16.
        assert_eq!(size_for_face(Metric, 5.0, true), choice(5.0, 0.8));
        assert_eq!(size_for_face(Metric, 14.0, true), choice(14.0, 2.0));
        assert_eq!(size_for_face(Metric, 4.98, true), choice(6.0, 1.0));
        close(size_for_face(Unc, 6.35, true).diameter, 6.35, 6);
        assert_eq!(size_for_face(Print45, 10.0, true), choice(10.0, 2.75));
    }

    #[test]
    fn brings_a_face_drawn_another_size_to_the_crests() {
        let m5 = spec(Metric, 5.0, 0.8, 0.0);
        let minor = minor_diameter(5.0, 0.8, Metric) / 2.0;
        let get = |v: Option<f64>| v.unwrap_or(f64::NAN);
        // A Ø5 bore takes M5 at its minor, M4 at its, M6 at its.
        close(get(fill_radius(&m5, true, 2.5)), minor, 12);
        close(get(fill_radius(&ThreadSpec { diameter: 4.0, pitch: 0.7, ..m5 }, true, 2.5)), minor_diameter(4.0, 0.7, Metric) / 2.0, 12);
        close(get(fill_radius(&ThreadSpec { diameter: 6.0, pitch: 1.0, ..m5 }, true, 2.5)), minor_diameter(6.0, 1.0, Metric) / 2.0, 12);
        // A bore at the minor, within a hair of it or narrower: the cut does it.
        assert_eq!(fill_radius(&m5, true, minor), None);
        assert_eq!(fill_radius(&m5, true, minor + 0.005), None);
        assert_eq!(fill_radius(&m5, true, 1.5), None);
        // The offset moves the crests, and the fill with them.
        close(get(fill_radius(&ThreadSpec { offset: 0.05, ..m5 }, true, 2.5)), minor + 0.05, 12);
        // A Ø4 pin is built up to M5's major; a Ø5 or a Ø6 one is not.
        assert_eq!(fill_radius(&m5, false, 2.0), Some(2.5));
        assert_eq!(fill_radius(&m5, false, 2.5), None);
        assert_eq!(fill_radius(&m5, false, 3.0), None);
        // A bolt's crests come to a point: the fill runs a quarter pitch past them. A nut's
        // 60° crest, the wide flat, never does short of the offset's limit.
        let sharp = ThreadSpec { offset: 0.12, ..m5 };
        let bolt = tooth_outline(&sharp, false, 0.0, None);
        assert!(bolt.pointed);
        close(get(fill_radius(&sharp, false, 2.0)), bolt.crest + 0.2, 12);
        assert!(!tooth_outline(&sharp, true, 0.0, None).pointed);
        close(get(fill_radius(&sharp, true, 2.5)), minor + 0.12, 12);
        // What the row says about the face: off by more than the depth.
        assert_eq!(face_against_thread(&m5, true, 5.0), Some(FaceAgainst::Wider));
        assert_eq!(face_against_thread(&m5, true, 2.0 * minor + 0.2), None);
        assert_eq!(face_against_thread(&m5, false, 4.0), Some(FaceAgainst::Narrower));
        assert_eq!(face_against_thread(&m5, false, 4.9), None);
    }

    #[test]
    fn draws_a_bolts_tooth_and_a_nuts_the_flats_the_standards_narrower_than_the_pitch() {
        let p = 2.0;
        let bolt = tooth_outline(&spec(Metric, 16.0, p, 0.0), false, 0.5, None);
        // From the minor diameter, three quarters of the pitch wide, to the major, an eighth
        // wide; a nut's the other way round.
        close(bolt.root, minor_diameter(16.0, p, Metric) / 2.0, 9);
        assert_eq!(bolt.crest, 8.0);
        close(width_at(&bolt.points, 8.0), p / 8.0, 9);
        close(width_at(&bolt.points, bolt.root), (3.0 * p) / 4.0, 9);
        // Narrower than the pitch all the way into the core, so the turns never meet.
        assert!(max_abs_z(&bolt.points) < p / 2.0);
        let nut = tooth_outline(&spec(Metric, 16.0, p, 0.0), true, 0.5, None);
        close(nut.crest, minor_diameter(16.0, p, Metric) / 2.0, 9);
        assert_eq!(nut.root, 8.0);
        close(width_at(&nut.points, nut.crest + 1e-12), p / 4.0, 6);
        close(width_at(&nut.points, 8.0), (7.0 * p) / 8.0, 9);
        assert!(max_abs_z(&nut.points) < p / 2.0);
        // A tooth and a gap make a pitch: the bolt's tooth is the nut's gap.
        let depth = thread_depth(p, Metric);
        let bolt_area = area_and_radius(&clip_outline(&bolt.points, bolt.root, true)).0;
        let nut_area = area_and_radius(&clip_outline(&nut.points, 8.0, false)).0;
        close(bolt_area + nut_area, depth * p, 9);
    }

    #[test]
    fn takes_the_tooth_up_to_its_point_where_the_crest_would_lie_on_the_face() {
        let t = spec(Metric, 16.0, 2.0, 0.0);
        let on_face = tooth_outline(&t, false, 0.5, Some(8.0));
        assert!(max_r(&on_face.points) > 8.1);
        close(width_at(&on_face.points, 8.0), 2.0 / 8.0, 9);
        let inside = tooth_outline(&t, false, 0.5, Some(9.0));
        assert_eq!(max_r(&inside.points), 8.0);
        let bore = tooth_outline(&t, true, 0.5, Some(minor_diameter(16.0, 2.0, Metric) / 2.0));
        assert!(min_r(&bore.points) < bore.crest - 0.1);
    }

    #[test]
    fn moves_every_face_in_by_the_offset_the_crest_to_a_point_past_what_its_flat_gives() {
        let p = 1.0;
        let plain = tooth_outline(&spec(Metric, 6.0, p, 0.0), false, 0.5, None);
        let loose = tooth_outline(&spec(Metric, 6.0, p, 0.05), false, 0.5, None);
        close(loose.crest, plain.crest - 0.05, 9);
        close(loose.root, plain.root - 0.05, 9);
        assert!(!loose.pointed);
        // The flanks moved square to themselves: at any radius the tooth is 0.05/cos 30°
        // narrower on each side.
        close(width_at(&plain.points, 2.7) - width_at(&loose.points, 2.7), (2.0 * 0.05) / (PI / 6.0).cos(), 9);
        let sharp = tooth_outline(&spec(Metric, 6.0, p, 0.15), false, 0.5, None);
        assert!(sharp.pointed);
        assert_eq!(sharp.points.iter().filter(|q| q.x == sharp.crest).count(), 1);
        // A nut's tooth: its crest comes out, its root goes deeper.
        let nut = tooth_outline(&spec(Metric, 6.0, p, 0.05), true, 0.5, None);
        close(nut.crest, minor_diameter(6.0, p, Metric) / 2.0 + 0.05, 9);
        close(nut.root, 3.05, 9);
    }

    #[test]
    fn refuses_an_offset_that_leaves_nothing_where_the_threads_bear_and_other_numbers_it_cannot_cut() {
        let mm = DisplayUnits::Mm;
        close(max_thread_offset(2.0, Metric), 0.433, 3);
        assert_eq!(thread_problem(&spec(Metric, 16.0, 2.0, 0.4), mm), None);
        let big = thread_problem(&spec(Metric, 16.0, 2.0, 0.5), mm).unwrap_or_default();
        assert!(big.contains("no thickness at the pitch diameter — under 0.433 mm"), "{big}");
        let negative = thread_problem(&spec(Metric, 16.0, 2.0, -0.1), mm).unwrap_or_default();
        assert!(negative.contains("only takes material away"), "{negative}");
        let flat = thread_problem(&spec(Metric, 16.0, 0.0, 0.0), mm).unwrap_or_default();
        assert!(flat.contains("needs a pitch"), "{flat}");
        let coarse = thread_problem(&spec(Metric, 3.0, 2.5, 0.0), mm).unwrap_or_default();
        assert!(coarse.contains("too coarse"), "{coarse}");
    }

    /// reThread.test.ts, "the Thread feature fails with the reason": the resolve-time part —
    /// an offset too large, a face that is not round — and resolve's own length and start.
    #[test]
    fn fails_a_thread_feature_with_the_reason_before_the_kernel_is_asked() {
        let cut = |offset: f64, length: Option<f64>, start: f64| ThreadCut { spec: spec(Metric, 16.0, 2.0, offset), left_handed: false, length, start, flip: false, chamfer: false };
        assert_eq!(thread_feature_problem(&cut(0.0, None, 0.0), true), None);
        let big = thread_feature_problem(&cut(0.6, None, 0.0), true).unwrap_or_default();
        assert!(big.contains("no thickness at the pitch diameter"), "{big}");
        let flat = thread_feature_problem(&cut(0.0, None, 0.0), false).unwrap_or_default();
        assert!(flat.contains("round face"), "{flat}");
        assert_eq!(thread_feature_problem(&cut(0.0, Some(0.0), 0.0), true).as_deref(), Some("The thread needs a length."));
        assert_eq!(thread_feature_problem(&cut(0.0, Some(f64::NAN), 0.0), true).as_deref(), Some("The thread needs a length."));
        assert_eq!(thread_feature_problem(&cut(0.0, Some(10.0), -1.0), true).as_deref(), Some("The thread has to start on the face — zero or more in from its end."));
        // The whole face long, the start is not read.
        assert_eq!(thread_feature_problem(&cut(0.0, None, -1.0), true), None);
        // The face is checked first, then the numbers.
        assert!(thread_feature_problem(&cut(0.6, Some(0.0), 0.0), false).unwrap_or_default().contains("round face"));
        assert!(thread_feature_problem(&cut(0.6, Some(0.0), 0.0), true).unwrap_or_default().contains("no thickness"));
    }

    /// reCoreUnits.test.ts: a thread's problem, its figures in inches when asked.
    #[test]
    fn a_threads_problem_its_figures_in_inches_when_asked() {
        let coarse = spec(Metric, 2.0, 1.27, 0.0);
        assert_eq!(thread_problem(&coarse, DisplayUnits::In).as_deref(), Some("A pitch of 0.05 in is too coarse for Ø0.0787 in — the thread would reach the axis."));
        assert_eq!(thread_problem(&coarse, DisplayUnits::Mm).as_deref(), Some("A pitch of 1.27 mm is too coarse for Ø2 mm — the thread would reach the axis."));
    }

    #[test]
    fn keeps_the_60_degree_tooth_exactly_where_the_forms_own_closed_forms_put_it() {
        // The outline as it was drawn before the forms were one: ISO 68-1's numbers written
        // out for the 60° vee.
        let s3 = 3f64.sqrt();
        let before = |t: &ThreadSpec, internal: bool, reach: f64, face: Option<f64>| -> Vec<DVec2> {
            let p = t.pitch;
            let c = t.offset;
            let big_r = t.diameter / 2.0;
            let depth = (5.0 / 8.0) * (s3 / 2.0) * p;
            let past = (0.05 * p).min(reach / 2.0);
            let pt = DVec2::new;
            if !internal {
                let root = big_r - depth - c;
                let root_half = (3.0 / 8.0) * p - c / s3;
                let crest_half = p / 16.0 - c / s3;
                let pointed = crest_half < 1e-4 * p;
                let apex = big_r - 2.0 * c + (s3 / 16.0) * p;
                let crest = if pointed { apex } else { big_r - c };
                let to_apex = pointed || face.is_some_and(|f| crest >= f - 0.01 * p);
                let bend = root - past;
                let bend_half = root_half + past / s3;
                let mut out = vec![pt(root - reach, -bend_half), pt(bend, -bend_half)];
                if to_apex {
                    out.push(pt(apex, 0.0));
                } else {
                    out.extend([pt(crest, -crest_half), pt(crest, crest_half)]);
                }
                out.extend([pt(bend, bend_half), pt(root - reach, bend_half)]);
                return out;
            }
            let minor = big_r - depth;
            let root = big_r + c;
            let root_half = (7.0 / 16.0) * p - c / s3;
            let crest_half = p / 8.0 - c / s3;
            let pointed = crest_half < 1e-4 * p;
            let apex = minor + 2.0 * c - (s3 / 8.0) * p;
            let crest = if pointed { apex } else { minor + c };
            let to_apex = pointed || face.is_some_and(|f| crest <= f + 0.01 * p);
            let bend = root + past;
            let bend_half = root_half + past / s3;
            let mut out = vec![if to_apex { pt(apex, 0.0) } else { pt(crest, -crest_half) }];
            out.extend([pt(bend, -bend_half), pt(root + reach, -bend_half), pt(root + reach, bend_half), pt(bend, bend_half)]);
            if !to_apex {
                out.push(pt(crest, crest_half));
            }
            out
        };
        let cases: [(f64, f64, f64); 7] = [(16.0, 2.0, 0.0), (16.0, 2.0, 0.2), (16.0, 2.0, 0.3), (6.0, 1.0, 0.05), (6.0, 1.0, 0.15), (3.0, 0.5, 0.05), (6.35, 25.4 / 20.0, 0.1)];
        for (d, p, c) in cases {
            for standard in [Metric, Unc, Unf] {
                let t = spec(standard, d, p, c);
                let minor = d / 2.0 - (5.0 / 8.0) * (s3 / 2.0) * p;
                for (internal, face) in [(false, None), (false, Some(d / 2.0)), (true, None), (true, Some(minor))] {
                    let got = tooth_outline(&t, internal, 0.5, face).points;
                    let want = before(&t, internal, 0.5, face);
                    let label = format!("{} Ø{d}×{p} −{c} {}{}", standard.id(), if internal { "nut" } else { "bolt" }, if face.is_none() { "" } else { " on its face" });
                    assert_eq!(got.len(), want.len(), "{label}");
                    for (g, w) in got.iter().zip(&want) {
                        assert!((g.x - w.x).abs() < 5e-13, "{label}: r {} against {}", g.x, w.x);
                        assert!((g.y - w.y).abs() < 5e-13, "{label}: z {} against {}", g.y, w.y);
                    }
                }
            }
        }
    }

    #[test]
    fn draws_the_printing_forms_tooth() {
        // 45° flanks, a fifth of the pitch flat at the crest and at the root, 0.3 of it deep.
        let p = 2.75;
        let t = spec(Print45, 10.0, p, 0.0);
        let bolt = tooth_outline(&t, false, 0.5, None);
        assert_eq!(bolt.crest, 5.0);
        close(bolt.root, 5.0 - 0.3 * p, 12);
        close(bolt.root, minor_diameter(10.0, p, Print45) / 2.0, 12);
        assert!(!bolt.pointed);
        // The crest a fifth of the pitch, the gap's floor a fifth — the tooth four fifths wide
        // there — and half the pitch at the pitch diameter.
        close(width_at(&bolt.points, 5.0), p / 5.0, 12);
        close(width_at(&bolt.points, bolt.root), (4.0 * p) / 5.0, 12);
        close(width_at(&bolt.points, pitch_diameter(10.0, p, Print45) / 2.0), p / 2.0, 12);
        // The flanks at exactly 45°: from the crest's corner to the bend past the root, as far
        // along the axis as across it.
        let slope = |a: DVec2, b: DVec2| ((b.y - a.y) / (b.x - a.x)).abs();
        assert_eq!(bolt.points.len(), 6);
        close(slope(bolt.points[1], bolt.points[2]), 1.0, 12);
        close(slope(bolt.points[3], bolt.points[4]), 1.0, 12);
        assert!(max_abs_z(&bolt.points) < p / 2.0);
        // A nut's: the crest a fifth wide at the minor, four fifths at the major.
        let nut = tooth_outline(&t, true, 0.5, None);
        assert_eq!(nut.root, 5.0);
        close(nut.crest, 5.0 - 0.3 * p, 12);
        close(width_at(&nut.points, nut.crest + 1e-12), p / 5.0, 6);
        close(width_at(&nut.points, 5.0), (4.0 * p) / 5.0, 12);
        close(slope(nut.points[0], nut.points[1]), 1.0, 12);
        close(slope(nut.points[4], nut.points[5]), 1.0, 12);
        assert!(max_abs_z(&nut.points) < p / 2.0);
        // The bolt's tooth is the nut's gap.
        let bolt_area = area_and_radius(&clip_outline(&bolt.points, bolt.root, true)).0;
        let nut_area = area_and_radius(&clip_outline(&nut.points, 5.0, false)).0;
        close(bolt_area + nut_area, 0.3 * p * p, 12);
        // The 60° form's flanks, for the difference: tan 30°.
        let vee = tooth_outline(&ThreadSpec { standard: Metric, ..t }, false, 0.5, None);
        close(slope(vee.points[1], vee.points[2]), 1.0 / 3f64.sqrt(), 12);
    }

    #[test]
    fn moves_the_printing_forms_faces_in_by_the_offset() {
        // The crest's flat narrower, the gap's floor wider, by twice (√2 − 1) the offset.
        let p = 2.75;
        let c = 0.2;
        let t = spec(Print45, 10.0, p, 0.0);
        let plain = tooth_outline(&t, false, 0.5, None);
        let loose = tooth_outline(&ThreadSpec { offset: c, ..t }, false, 0.5, None);
        close(loose.crest, 5.0 - c, 12);
        close(loose.root, plain.root - c, 12);
        let k = 2.0 * c * (SQRT_2 - 1.0);
        close(width_at(&loose.points, loose.crest), p / 5.0 - k, 12);
        close(p - width_at(&loose.points, loose.root), p / 5.0 + k, 12);
        // The flanks moved square to themselves: √2 times the offset narrower on each side at
        // any radius.
        close(width_at(&plain.points, 4.5) - width_at(&loose.points, 4.5), 2.0 * c * SQRT_2, 12);
        let nut = tooth_outline(&ThreadSpec { offset: c, ..t }, true, 0.5, None);
        close(nut.crest, minor_diameter(10.0, p, Print45) / 2.0 + c, 12);
        close(nut.root, 5.0 + c, 12);
        close(width_at(&nut.points, nut.crest + 1e-12), p / 5.0 - k, 6);
        // The flats outlast the tooth's thickness at the pitch diameter: the crest never comes
        // to a point at an offset the thread takes, and only past what is refused would it.
        let max = max_thread_offset(p, Print45);
        close(max, (SQRT_2 / 8.0) * p, 12);
        for internal in [false, true] {
            assert!(!tooth_outline(&ThreadSpec { offset: 0.999 * max, ..t }, internal, 0.5, None).pointed);
        }
        assert_eq!(thread_problem(&ThreadSpec { offset: 0.999 * max, ..t }, DisplayUnits::Mm), None);
        let at_max = thread_problem(&ThreadSpec { offset: max, ..t }, DisplayUnits::Mm).unwrap_or_default();
        assert!(at_max.contains("no thickness at the pitch diameter — under 0.486 mm"), "{at_max}");
        assert!(tooth_outline(&ThreadSpec { offset: 0.25 * p, ..t }, false, 0.5, None).pointed);
        let past = thread_problem(&ThreadSpec { offset: 0.25 * p, ..t }, DisplayUnits::Mm).unwrap_or_default();
        assert!(past.contains("no thickness at the pitch diameter"), "{past}");
    }

    /// A Thread feature's numbers as a ScanRuler project holds them.
    #[test]
    fn reads_a_thread_cut_as_a_project_saves_it() -> Result<(), serde_json::Error> {
        let json = r#"{"standard":"print45","diameter":16,"pitch":3.5,"offset":0.1,"leftHanded":true,"length":null,"start":0,"flip":false}"#;
        let cut: ThreadCut = serde_json::from_str(json)?;
        assert_eq!(cut.spec, spec(Print45, 16.0, 3.5, 0.1));
        assert!(cut.left_handed && !cut.chamfer && cut.length.is_none());
        let back = serde_json::to_value(cut)?;
        assert_eq!(back["leftHanded"], serde_json::json!(true));
        assert_eq!(back["standard"], serde_json::json!("print45"));
        assert_eq!(back["length"], serde_json::Value::Null);
        Ok(())
    }

    #[test]
    fn places_a_length_on_the_face_from_its_far_end_or_with_flip_its_near_one() {
        assert_eq!(thread_span(-15.0, 15.0, None, 3.0, false), Some(ThreadSpan { from: -15.0, to: 15.0 }));
        assert_eq!(thread_span(-15.0, 15.0, Some(10.0), 5.0, false), Some(ThreadSpan { from: 0.0, to: 10.0 }));
        assert_eq!(thread_span(-15.0, 15.0, Some(10.0), 5.0, true), Some(ThreadSpan { from: -10.0, to: 0.0 }));
        assert_eq!(thread_span(-15.0, 15.0, Some(10.0), 25.0, false), None);
    }

    /// A tube up Z tessellated as a kernel would hand it over, each triangle wound outward
    /// and tagged with its face: 0 the outside, 1 the bore, 2 and 3 the ends.
    fn tube(outer: f64, inner: f64, height: f64, segments: u32) -> (Vec<DVec3>, Vec<[u32; 3]>, Vec<u32>) {
        let mut positions = Vec::new();
        let ring = |r: f64, z: f64, positions: &mut Vec<DVec3>| -> u32 {
            let start = positions.len() as u32;
            for k in 0..segments {
                let a = 2.0 * PI * f64::from(k) / f64::from(segments);
                positions.push(DVec3::new(r * a.cos(), r * a.sin(), z));
            }
            start
        };
        let ob = ring(outer, 0.0, &mut positions);
        let ot = ring(outer, height, &mut positions);
        let ib = ring(inner, 0.0, &mut positions);
        let it = ring(inner, height, &mut positions);
        let mut triangles = Vec::new();
        let mut tri_face = Vec::new();
        for k in 0..segments {
            let n = (k + 1) % segments;
            // The outside, looking away from the axis.
            triangles.extend([[ob + k, ob + n, ot + n], [ob + k, ot + n, ot + k]]);
            tri_face.extend([0, 0]);
            // The bore, looking towards it.
            triangles.extend([[ib + k, it + n, ib + n], [ib + k, it + k, it + n]]);
            tri_face.extend([1, 1]);
            // The bottom, looking down; the top, up.
            triangles.extend([[ob + k, ib + n, ob + n], [ob + k, ib + k, ib + n]]);
            tri_face.extend([2, 2]);
            triangles.extend([[ot + k, ot + n, it + n], [ot + k, it + n, it + k]]);
            tri_face.extend([3, 3]);
        }
        (positions, triangles, tri_face)
    }

    /// Adapted from the kernel test of the same name: the block with a bore is tessellated
    /// by hand here rather than by Open CASCADE.
    #[test]
    fn tells_a_boss_from_a_bore_by_the_way_the_faces_triangles_look() {
        let minor = minor_diameter(16.0, 2.0, Metric) / 2.0;
        let (positions, triangles, tri_face) = tube(15.0, minor, 12.0, 96);
        let mesh = FaceMesh { positions: &positions, triangles: &triangles, tri_face: &tri_face };
        let (center, dir) = (DVec3::ZERO, DVec3::Z);
        assert_eq!(round_face_is_boss(mesh, 0, center, dir), Some(true));
        assert_eq!(round_face_is_boss(mesh, 1, center, dir), Some(false));
        assert_eq!(round_face_is_boss(mesh, 12345, center, dir), None);
        // And what the Thread box reads off them: a bore drawn at M16's minor is an M16.
        let bore = RoundFacePick { face: Some(1), center: Some(center), radius: Some(minor), dir };
        assert_eq!(thread_face_reading(mesh, &bore, Metric, false), Some(ThreadFaceReading { internal: true, diameter: Some(16.0), pitch: Some(2.0) }));
        assert_eq!(thread_face_reading(mesh, &bore, Metric, true), Some(ThreadFaceReading { internal: true, diameter: None, pitch: None }));
        let unpicked = RoundFacePick { face: None, ..bore };
        assert_eq!(thread_face_reading(mesh, &unpicked, Metric, false), None);
    }
}
