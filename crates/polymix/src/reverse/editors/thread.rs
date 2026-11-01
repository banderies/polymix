// SPDX-License-Identifier: AGPL-3.0-only
//! The Thread feature's box: a screw thread on a full round face of any body — a bolt's on a
//! boss, a nut's in a bore, told by which way the face looks — in ISO metric, UNC, UNF or the
//! 45° form for printing, at a listed size and pitch or one typed; either hand; the whole face
//! long or a length from a start in from either end; the free ends chamfered; an offset for
//! clearance when printing. A click on a round face takes it (on another body, the thread
//! moves there), and while no size has been chosen or typed the standard's size for the face
//! as drawn is taken with it. Ported from ScanRuler (`plugins/reverse/ui/FeatureEditor.tsx`
//! the thread case, `featureFields.tsx` Num, `state/reFeatureSlice.ts` and `state/reDrafts.ts`
//! the thread draft, `app/useReverseWorkspace.ts` the face's reading), © Stefan Hermann (CNC
//! Kitchen), AGPL-3.0-only.
//!
//! Besides ScanRuler's box, a key reads the thread off the scan along the face (ScanRuler's
//! `reverse.thread.read` command, which Polymix has no agent channel for) and offers the
//! standard size it matches.
//!
//! The fields the Hole box shares with this one — a labelled number that takes an expression,
//! the ⓘ beside a label, the hints — are here too.

use std::cell::{Cell, RefCell};

use egui::{Color32, Pos2, RichText, Stroke};
use glam::DVec3;
use mm_reverse::build::{core_standard, is_unpicked, model_standard, thread_resolve_problem, unpicked_face};
use mm_reverse::display_units::{fmt_len, DisplayUnits, FmtOptions};
use mm_reverse::expression::{evaluate_for, is_plain_number, ParameterKind};
use mm_reverse::length_fields::{len_bare, len_short};
use mm_reverse::model::{BodyId, Exprs, FaceRef, FeatureKind, Part, ThreadFeature, ThreadStandard as ModelStandard};
use mm_reverse::nominal::{nominal_of, NominalKind};
use mm_reverse::parameters::{evaluate_parameters, ParameterTable};
use mm_reverse::round::{js_round, round3};
use mm_reverse::thread::{chamfer_radius, face_against_thread, max_thread_offset, minor_diameter, size_for_face, size_of, thread_depth, thread_designation, thread_face_reading, thread_name, thread_problem, thread_sizes, threads_per_inch, FaceAgainst, FaceMesh, RoundFacePick, ThreadSpec, ThreadStandard, THREAD_STANDARDS};
use mm_reverse::thread_scan::{read_designation, round_faces, round_length, thread_read, thread_size_named, ReadAlong, ReadAsk, ThreadSuggestion};
use mm_reverse::topo::{describe_face, match_face, match_tolerance, FaceKind, Topology};
use mm_reverse::units::DisplayUnits as ThreadUnits;

use crate::reverse::view::Hit;
use crate::reverse::{PickWhat, ReverseWorkspace};
use crate::tools::Viewport;

pub(crate) const READY: bool = true;
pub(crate) const BUTTON: (&str, &str) = (
    "Thread",
    "A modeled screw thread on a round face of the body — a bolt's on a boss, a nut's in a bore: ISO metric, UNC or UNF, a 45° form that prints without supports, or a diameter and pitch of your own, with an offset for clearance when printing",
);
/// A fresh box: its own state is set up on its first frame (see [`with_box`]).
pub(crate) const FIRST_PICK: PickWhat = PickWhat::Custom(0);

/// The units the boxes show lengths in. The workspace has no inch setting yet.
pub(crate) const UNITS: DisplayUnits = DisplayUnits::Mm;

/// The size of a hint's words.
pub(crate) const HINT_SIZE: f32 = 11.5;
/// How wide a select stands.
pub(crate) const SELECT_WIDTH: f32 = 180.0;

/// The colour of a warning hint.
pub(crate) const WARN: Color32 = Color32::from_rgb(226, 164, 59);
/// The colour a hovered choice is drawn on the part in (ScanRuler's picker hint).
pub(crate) const HINT: Color32 = Color32::from_rgb(255, 176, 32);
/// The colour of what the box has picked.
pub(crate) const PICKED: Color32 = Color32::from_rgb(60, 179, 113);

/// What a Thread box holds besides the feature, for as long as it is open (ScanRuler's thread
/// draft): whether a size was chosen or typed, which way the face looks, the length and start
/// kept while the thread runs the whole face, and the last reading off the scan.
#[derive(Default)]
struct ThreadBox {
    serial: u32,
    sized: bool,
    /// The face the reading of `internal` is of.
    read_for: Option<(BodyId, FaceRef)>,
    internal: Option<bool>,
    length: f64,
    start: f64,
    scan: Option<Result<ScanRead, String>>,
}

/// The thread read off the scan along the face.
#[derive(Clone)]
struct ScanRead {
    note: String,
    feature: Option<ThreadSuggestion>,
    designation: Option<String>,
}

thread_local! {
    static BOX: RefCell<ThreadBox> = RefCell::new(ThreadBox::default());
    static SERIAL: Cell<u32> = const { Cell::new(0) };
}

/// A fresh number for a box that has just opened.
pub(crate) fn next_serial() -> u32 {
    SERIAL.with(|s| {
        let n = s.get().wrapping_add(1).max(1);
        s.set(n);
        n
    })
}

/// Whether the open box edits a feature of the part rather than making a new one.
pub(crate) fn editing(ws: &ReverseWorkspace) -> bool {
    ws.preview_feature.is_some_and(|id| ws.part.feature(id).is_some())
}

/// Set the box's own state up on its first frame: the box's pick holds its number.
fn open_box(ws: &ReverseWorkspace, x: &ThreadFeature, pick: &mut PickWhat) {
    BOX.with(|b| {
        let mut b = b.borrow_mut();
        if matches!(*pick, PickWhat::Custom(n) if n != 0 && n == b.serial) {
            return;
        }
        let serial = next_serial();
        let length = x.length.unwrap_or_else(|| face_length(ws, x.body, &x.face).unwrap_or(10.0));
        *b = ThreadBox { serial, sized: editing(ws), read_for: None, internal: None, length, start: x.start, scan: None };
        *pick = PickWhat::Custom(serial);
    });
}

/// The open box's own state.
fn with_box<R>(f: impl FnOnce(&mut ThreadBox) -> R) -> R {
    BOX.with(|b| f(&mut b.borrow_mut()))
}

/// The last solid body, else the last body: what a new feature on a body works on.
pub(crate) fn last_solid(ws: &ReverseWorkspace) -> Option<BodyId> {
    ws.built.bodies.iter().rev().find(|b| !b.surface).or(ws.built.bodies.last()).map(|b| b.id)
}

pub(crate) fn new_feature(ws: &ReverseWorkspace) -> Result<FeatureKind, String> {
    let body = last_solid(ws).ok_or("A thread is cut on a round face of a body: make a body first.")?;
    Ok(FeatureKind::Thread(ThreadFeature { body, face: unpicked_face(), standard: ModelStandard::Metric, diameter: 10.0, pitch: 1.5, left_handed: false, length: None, start: 0.0, flip: false, offset: 0.0, chamfer: Some(false), exprs: Exprs::new() }))
}

/// A Thread on a round face of a body, the standard's size for it to start: what the chip on a
/// round face opens (ScanRuler's "Thread" chip).
#[allow(dead_code)]
pub(crate) fn new_on_face(ws: &ReverseWorkspace, body: BodyId, face: FaceRef) -> Result<FeatureKind, String> {
    let FeatureKind::Thread(mut x) = new_feature(ws)? else { return Err("No thread.".into()) };
    x.body = body;
    x.face = face;
    Ok(FeatureKind::Thread(x))
}

/// Whether the box has everything it needs to be created (ScanRuler's `featureDraftReady`).
pub(crate) fn ready(x: &ThreadFeature) -> bool {
    !is_unpicked(&x.face) && thread_problem(&spec_of(x), ThreadUnits::Mm).is_none() && x.length.is_none_or(|l| l > 0.0 && x.start >= 0.0)
}

/// Why the box cannot be created yet, if it cannot.
pub(crate) fn not_ready_why(x: &ThreadFeature) -> Option<String> {
    if ready(x) {
        return None;
    }
    thread_resolve_problem(x).or_else(|| Some("The thread needs a length.".into()))
}

fn spec_of(x: &ThreadFeature) -> ThreadSpec {
    ThreadSpec { standard: core_standard(x.standard), diameter: x.diameter, pitch: x.pitch, offset: x.offset }
}

// ---- the fields the boxes share ----

/// A field's label, with the ⓘ whose card says what it is.
pub(crate) fn label(ui: &mut egui::Ui, text: &str, hint: Option<&str>) {
    ui.label(RichText::new(text).weak());
    if let Some(h) = hint.filter(|h| !h.is_empty()) {
        ui.label(RichText::new("ⓘ").weak()).on_hover_text(RichText::new(h));
    }
}

/// A small dim paragraph.
pub(crate) fn hint(ui: &mut egui::Ui, text: impl Into<String>) {
    ui.label(RichText::new(text.into()).weak().size(HINT_SIZE));
}

/// A warning paragraph.
pub(crate) fn warn(ui: &mut egui::Ui, text: impl Into<String>) {
    ui.label(RichText::new(text.into()).color(WARN).size(HINT_SIZE));
}

/// The part's parameters, read for an expression to use.
pub(crate) fn parameter_table(part: &Part) -> ParameterTable {
    let list: Vec<mm_reverse::parameters::Parameter> = part.parameters.iter().filter_map(|p| serde_json::to_value(p).ok().and_then(|v| serde_json::from_value(v).ok())).collect();
    evaluate_parameters(&list)
}

/// A labelled number (ScanRuler's `Num`): the label and its ⓘ, the round number a value was
/// probably meant as (one click takes it, not while an expression drives the field), and the
/// field — a number, or an expression over the part's parameters kept under the slot, shown
/// with the value it comes to. Committed on Enter or when the field is left; an expression
/// that cannot be read stays with why under it, the number as it was. Returns the number
/// committed.
#[allow(clippy::too_many_arguments)]
pub(crate) fn num(ui: &mut egui::Ui, scope: &ParameterTable, id: egui::Id, text: &str, hint_text: Option<&str>, value: f64, slot: &str, exprs: &mut Exprs, nominal: Option<NominalKind>) -> Option<f64> {
    let expr = exprs.get(slot).map(str::to_string);
    let shown = expr.clone().unwrap_or_else(|| len_bare(value, UNITS, 3));
    let mut out = None;
    let err_id = id.with("error");
    let buf_id = id.with("text");
    ui.horizontal(|ui| {
        label(ui, text, hint_text);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let tail = match &expr {
                Some(_) => format!("= {}", mm_reverse::expression::format_value(value, ParameterKind::Length, UNITS)),
                None => "mm".to_string(),
            };
            ui.label(RichText::new(tail).italics().weak());
            let mut buf: String = ui.data(|d| d.get_temp::<String>(buf_id)).unwrap_or_else(|| shown.clone());
            let edit = egui::TextEdit::singleline(&mut buf).desired_width(96.0).font(egui::TextStyle::Monospace);
            let r = ui.add(edit);
            let r = match &expr {
                Some(e) => r.on_hover_text(format!("{e} = {}", mm_reverse::expression::format_value(value, ParameterKind::Length, UNITS))),
                None => r,
            };
            let escape = ui.input(|i| i.key_pressed(egui::Key::Escape));
            if r.has_focus() || r.changed() {
                ui.data_mut(|d| d.insert_temp(buf_id, buf.clone()));
            }
            if escape && (r.has_focus() || r.lost_focus()) {
                // The old text comes back.
                ui.data_mut(|d| {
                    d.remove::<String>(buf_id);
                    d.remove::<String>(err_id);
                });
                r.surrender_focus();
            } else if r.lost_focus() {
                let t = buf.trim().to_string();
                ui.data_mut(|d| d.remove::<String>(buf_id));
                if t.is_empty() || t == shown {
                    ui.data_mut(|d| d.remove::<String>(err_id));
                } else {
                    match evaluate_for(&t, ParameterKind::Length, scope, false) {
                        Ok(v) => {
                            ui.data_mut(|d| d.remove::<String>(err_id));
                            if is_plain_number(&t) {
                                exprs.remove(slot);
                            } else {
                                exprs.insert(slot, t.clone());
                            }
                            out = Some(v.value);
                        }
                        Err(e) => {
                            ui.data_mut(|d| {
                                d.insert_temp(err_id, e.to_string());
                                d.insert_temp(buf_id, t.clone());
                            });
                        }
                    }
                }
            }
            if let (Some(kind), None) = (nominal, &expr) {
                if let Some(round) = nominal_of(value, kind) {
                    let text = fmt_len(round, UNITS, if round.fract() == 0.0 { 0 } else { 1 }, FmtOptions::default());
                    let chip = egui::Button::new(RichText::new(format!("→ {text}")).monospace().small()).small();
                    if ui.add(chip).on_hover_text("The round number this was probably meant as — click to take it").clicked() {
                        exprs.remove(slot);
                        out = Some(round);
                    }
                }
            }
        });
    });
    if let Some(e) = ui.data(|d| d.get_temp::<String>(err_id)) {
        warn(ui, e);
    }
    out
}

/// A select of strings: the label, the text shown, and the options with their tooltips.
/// Returns the index chosen.
pub(crate) fn select(ui: &mut egui::Ui, id: egui::Id, text: &str, hint_text: Option<&str>, shown: &str, options: &[(String, Option<String>)], hovered: Option<&mut Option<usize>>) -> Option<usize> {
    let mut chosen = None;
    let mut over = None;
    ui.horizontal(|ui| {
        label(ui, text, hint_text);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let r = egui::ComboBox::from_id_salt(id).selected_text(shown).width(SELECT_WIDTH).truncate().show_ui(ui, |ui| {
                if options.is_empty() {
                    ui.weak("Nothing to choose from");
                }
                for (i, (o, tip)) in options.iter().enumerate() {
                    let r = ui.selectable_label(o == shown, o);
                    if r.hovered() {
                        over = Some(i);
                    }
                    let r = match tip {
                        Some(t) => r.on_hover_text(t),
                        None => r,
                    };
                    if r.clicked() {
                        chosen = Some(i);
                    }
                }
            });
            r.response.on_hover_text(shown);
        });
    });
    if let Some(h) = hovered {
        *h = over;
    }
    chosen
}

/// A length to the micrometre, without its unit — a diameter after Ø.
pub(crate) fn bare3(v: f64) -> String {
    len_bare(v, UNITS, 3)
}

/// A length as a row reads it: to the hundredth of a millimetre, with its unit.
pub(crate) fn mm(v: f64) -> String {
    len_short(v, UNITS, 2)
}

/// A round face of a body of the pick set by its fingerprint: its index, the body's topology.
fn face_on(ws: &ReverseWorkspace, body: BodyId, face: &FaceRef) -> Option<(usize, std::sync::Arc<Topology>)> {
    let v = ws.pick_view(body)?;
    let tol = match_tolerance(v.mesh.bounds().diagonal());
    let i = match_face(face, &v.topology.face_prints(), &tol)?;
    Some((i, v.topology.clone()))
}

/// How long a round face runs along its axis, the faces of its cylinder joined to it with it.
fn face_length(ws: &ReverseWorkspace, body: BodyId, face: &FaceRef) -> Option<f64> {
    if is_unpicked(face) {
        return None;
    }
    let (i, t) = face_on(ws, body, face)?;
    Some(round_length(&t, i))
}

/// Which way the face looks — a bore or a boss — read off the body's mesh once it names a face
/// of it, and while no size was chosen, the standard's size for the face as drawn
/// (ScanRuler's `threadFaceReading`, run when the face is picked). Returns whether the feature
/// changed.
fn read_face(ws: &ReverseWorkspace, x: &mut ThreadFeature, b: &mut ThreadBox) -> bool {
    if is_unpicked(&x.face) || b.read_for == Some((x.body, x.face)) {
        return false;
    }
    let Some(v) = ws.pick_view(x.body) else { return false };
    let tol = match_tolerance(v.mesh.bounds().diagonal());
    let Some(i) = match_face(&x.face, &v.topology.face_prints(), &tol) else { return false };
    let m = &v.part_mesh;
    let mesh = FaceMesh { positions: &m.positions, triangles: &m.triangles, tri_face: &m.tri_face };
    let pick = RoundFacePick { face: Some(i as u32), center: x.face.center, radius: x.face.radius, dir: x.face.dir };
    let Some(r) = thread_face_reading(mesh, &pick, core_standard(x.standard), b.sized) else { return false };
    b.read_for = Some((x.body, x.face));
    b.internal = Some(r.internal);
    match (r.diameter, r.pitch) {
        (Some(d), Some(p)) => {
            x.diameter = d;
            x.pitch = p;
            true
        }
        _ => false,
    }
}

/// The size a standard offers for the face: its own for the face while none was chosen, else
/// the one nearest the diameter.
fn size_in(x: &ThreadFeature, b: &ThreadBox, standard: ThreadStandard) -> (f64, f64) {
    if let (false, Some(r), Some(internal)) = (b.sized, x.face.radius.filter(|r| *r > 0.0 && !is_unpicked(&x.face)), b.internal) {
        let c = size_for_face(standard, 2.0 * r, internal);
        return (c.diameter, c.pitch);
    }
    let sizes = thread_sizes(standard);
    let near = sizes.iter().fold(None::<&mm_reverse::thread::ThreadSize>, |a, s| match a {
        Some(a) if (s.diameter - x.diameter).abs() >= (a.diameter - x.diameter).abs() => Some(a),
        _ => Some(s),
    });
    near.map_or((x.diameter, x.pitch), |s| (s.diameter, s.pitches.first().copied().unwrap_or(x.pitch)))
}

/// The scan in the part's coordinates: each visible object's live points and their normals.
pub(crate) fn scan_clouds(ws: &mut ReverseWorkspace) -> Vec<(Vec<DVec3>, Vec<DVec3>)> {
    let parts = ws.scan_parts();
    let flip = ws.flip;
    let mut out = Vec::new();
    for p in parts.iter() {
        let Some(o) = ws.scan.iter().find(|o| o.id == p.id) else { continue };
        let normals = o.mesh.compute_vertex_normals();
        let pts: Vec<DVec3> = p.live_vertices.iter().map(|v| p.positions[*v as usize]).collect();
        let ns: Vec<DVec3> = p.live_vertices.iter().map(|v| normals.get(*v as usize).copied().unwrap_or(DVec3::ZERO)).map(|n| if flip { crate::app::flip_zy_out(n) } else { n }).collect();
        out.push((pts, ns));
    }
    out
}

/// The thread read off the scan along the box's face (ScanRuler's `reverse.thread.read`, along
/// a face of a body, the standard any).
fn read_scan(ws: &mut ReverseWorkspace, x: &ThreadFeature) -> Result<ScanRead, String> {
    if is_unpicked(&x.face) {
        return Err("Click the round face to read the thread along first.".into());
    }
    let (i, t) = face_on(ws, x.body, &x.face).ok_or("The face is not on the body as it stands.")?;
    let name = super::body_name(ws, x.body);
    let clouds = scan_clouds(ws);
    let refs: Vec<(&[DVec3], &[DVec3])> = clouds.iter().map(|(p, n)| (p.as_slice(), n.as_slice())).collect();
    let along = ReadAlong::Face { body: x.body, body_name: &name, topology: &t, face: i };
    let r = thread_read(along, &refs, &ReadAsk { standard: None, ..ReadAsk::default() }).map_err(|e| e.message)?;
    log::info!("reverse thread read: {}", r.json);
    Ok(ScanRead { note: r.note, feature: r.feature, designation: r.suggested.map(|s| s.designation) })
}

/// A thread's numbers set from what the scan read: its standard, size and pitch, its hand, and
/// the free ends chamfered, as the command's feature ready to add has it.
pub(crate) fn take_suggestion(x: &mut ThreadFeature, s: &ThreadSuggestion) {
    x.standard = model_standard(s.standard);
    x.diameter = s.diameter;
    x.pitch = s.pitch;
    x.left_handed = s.left_handed;
    x.chamfer = Some(true);
    x.exprs.remove("diameter");
    x.exprs.remove("pitch");
}

pub(crate) fn ui(ws: &mut ReverseWorkspace, ui: &mut egui::Ui, x: &mut ThreadFeature, pick: &mut PickWhat) -> bool {
    let mut changed = false;
    let id = egui::Id::NULL.with("re_thread_box");
    open_box(ws, x, pick);
    let internal = with_box(|b| {
        changed |= read_face(ws, x, b);
        b.internal
    });
    let units = ThreadUnits::Mm;
    let standard = core_standard(x.standard);
    let sizes = thread_sizes(standard);
    let size = size_of(standard, x.diameter);
    let pitch_listed = size.and_then(|s| s.pitches.iter().copied().find(|p| (p - x.pitch).abs() < 1e-4));
    let unpicked = is_unpicked(&x.face);
    let side = match internal {
        None => "",
        Some(true) => " · a bore: a nut’s thread",
        Some(false) => " · a boss: a bolt’s thread",
    };
    // 1. The face.
    ui.horizontal(|ui| {
        label(ui, "Face", None);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let text = if unpicked { "Click a round face of the body".to_string() } else { format!("{}{side}", describe_face(&x.face)) };
            ui.add(egui::Label::new(RichText::new(text).strong().size(12.0)).wrap());
        });
    });
    // The reading off the scan (Polymix's: ScanRuler offers it as a command).
    if !ws.scan.is_empty() {
        ui.horizontal(|ui| {
            let key = ui.add_enabled(!unpicked, egui::Button::new("Read off the scan").small()).on_hover_text("Read the thread on the scan along this face — its pitch, its crests and root, its hand — and the standard size it matches");
            if key.clicked() {
                let r = read_scan(ws, x);
                with_box(|b| b.scan = Some(r));
            }
        });
        let shown = with_box(|b| b.scan.clone());
        match shown {
            Some(Ok(r)) => {
                hint(ui, r.note.clone());
                if let (Some(f), Some(d)) = (r.feature.as_ref(), r.designation.as_ref()) {
                    if ui.button(format!("Use {d}{}", if f.left_handed { " left-hand" } else { "" })).on_hover_text("Take the size, the pitch and the hand read off the scan, the free ends chamfered").clicked() {
                        take_suggestion(x, f);
                        with_box(|b| b.sized = true);
                        changed = true;
                    }
                }
            }
            Some(Err(e)) => warn(ui, format!("⚠ {e}")),
            None => {}
        }
    }
    // 2. The standard.
    let std_options: Vec<(String, Option<String>)> = THREAD_STANDARDS.iter().map(|s| (s.label().to_string(), None)).collect();
    if let Some(i) = select(ui, id.with("standard"), "Type", None, standard.label(), &std_options, None) {
        let next = THREAD_STANDARDS[i];
        if next != standard {
            let (d, p) = with_box(|b| size_in(x, b, next));
            x.standard = model_standard(next);
            x.diameter = d;
            x.pitch = p;
            changed = true;
        }
    }
    // 3. The printing form's hint.
    if standard == ThreadStandard::Print45 {
        hint(ui, "45° flanks, a flat crest and root: printed with the axis upright, nothing overhangs past 45°. It mates only with itself.");
    }
    // 4. The size.
    let unified = matches!(standard, ThreadStandard::Unc | ThreadStandard::Unf);
    let size_text = |s: &mm_reverse::thread::ThreadSize| if unified { format!("{} — Ø{}", s.label, bare3(s.diameter)) } else { s.label.clone() };
    let mut size_options: Vec<(String, Option<String>)> = Vec::new();
    if size.is_none() {
        size_options.push((format!("Custom Ø{}", bare3(x.diameter)), None));
    }
    size_options.extend(sizes.iter().map(|s| (size_text(s), None)));
    let shown = size.map_or_else(|| format!("Custom Ø{}", bare3(x.diameter)), size_text);
    if let Some(i) = select(ui, id.with("size"), "Size", None, &shown, &size_options, None) {
        let k = if size.is_none() { i.checked_sub(1) } else { Some(i) };
        if let Some(s) = k.and_then(|k| sizes.get(k)) {
            x.diameter = s.diameter;
            x.pitch = s.pitches.first().copied().unwrap_or(x.pitch);
            x.exprs.remove("diameter");
            x.exprs.remove("pitch");
            with_box(|b| b.sized = true);
            changed = true;
        }
    }
    // 5. The designation.
    if let Some(s) = size {
        let suffix = |i: usize| match standard {
            ThreadStandard::Metric if s.pitches.len() > 1 => {
                if i == 0 {
                    " — coarse"
                } else {
                    " — fine"
                }
            }
            ThreadStandard::Print45 => {
                if i == 0 {
                    " — ISO coarse depth"
                } else {
                    " — ISO coarse pitch"
                }
            }
            _ => "",
        };
        let text = |i: usize, p: f64| format!("{}{}", thread_designation(standard, s.diameter, p).unwrap_or_default(), suffix(i));
        let custom = format!("Custom pitch {}", fmt_len(x.pitch, UNITS, 3, FmtOptions::default()));
        let mut options: Vec<(String, Option<String>)> = Vec::new();
        if pitch_listed.is_none() {
            options.push((custom.clone(), None));
        }
        options.extend(s.pitches.iter().enumerate().map(|(i, p)| (text(i, *p), None)));
        let shown = match pitch_listed.and_then(|p| s.pitches.iter().position(|q| *q == p)) {
            Some(i) => text(i, s.pitches[i]),
            None => custom,
        };
        if let Some(i) = select(ui, id.with("designation"), "Designation", None, &shown, &options, None) {
            let k = if pitch_listed.is_none() { i.checked_sub(1) } else { Some(i) };
            if let Some(p) = k.and_then(|k| s.pitches.get(k)).filter(|p| **p > 0.0) {
                x.pitch = *p;
                x.exprs.remove("pitch");
                with_box(|b| b.sized = true);
                changed = true;
            }
        }
    }
    // 6. The diameter, 7. the pitch.
    let part = parameter_table(&ws.part);
    if let Some(v) = num(ui, &part, id.with("diameter"), "Diameter", Some("The major diameter — a bolt's outside, a nut's at the bottom of its thread. Typed, the thread is cut to it, whatever the table has."), round3(x.diameter), "diameter", &mut x.exprs, None) {
        x.diameter = v.abs();
        with_box(|b| b.sized = true);
        changed = true;
    }
    let tpi = js_round(threads_per_inch(x.pitch) * 100.0) / 100.0;
    let pitch_hint = format!("How far the thread climbs in a turn — {} mm is {} threads per inch.", bare3(x.pitch), mm_reverse::round::number_text(tpi));
    if let Some(v) = num(ui, &part, id.with("pitch"), "Pitch", Some(&pitch_hint), round3(x.pitch), "pitch", &mut x.exprs, None) {
        x.pitch = v.abs();
        with_box(|b| b.sized = true);
        changed = true;
    }
    // 8. The hand.
    let hands = [("Right hand".to_string(), None), ("Left hand".to_string(), None)];
    if let Some(i) = select(ui, id.with("hand"), "Direction", None, if x.left_handed { "Left hand" } else { "Right hand" }, &hands, None) {
        if (i == 1) != x.left_handed {
            x.left_handed = i == 1;
            changed = true;
        }
    }
    // 9. Full length, or a length from a start.
    let mut full = x.length.is_none();
    if ui.checkbox(&mut full, RichText::new("Full length").weak()).changed() {
        with_box(|b| set_full_length(x, b, full));
        changed = true;
    }
    if let Some(length) = x.length {
        if let Some(v) = num(ui, &part, id.with("length"), "Length", Some("How far along the face the thread runs"), length, "length", &mut x.exprs, Some(NominalKind::Length)) {
            x.length = Some(v.abs());
            changed = true;
        }
        if let Some(v) = num(ui, &part, id.with("start"), "Start", Some("How far in from the end of the face the thread starts — the end further along the face's axis, or the other one with the box below ticked"), x.start, "start", &mut x.exprs, None) {
            x.start = v.abs();
            changed = true;
        }
        changed |= ui.checkbox(&mut x.flip, RichText::new("From the other end").weak()).changed();
    }
    // 10. The chamfer.
    let mut chamfer = x.chamfer.unwrap_or(false);
    if ui
        .checkbox(&mut chamfer, RichText::new("Chamfer the free ends (45°)").weak())
        .on_hover_text("A 45° chamfer across the thread where it comes out of the body free — a bolt's tip, a bore's mouth at either side of a nut — from just past the root out through the crests, so the thread starts clean instead of on a knife-edge first turn. Not where the thread ends against a shoulder or a bore's floor, or short of the face's end.")
        .changed()
    {
        x.chamfer = Some(chamfer);
        changed = true;
    }
    // 11. The offset.
    let offset_hint = format!("A clearance for 3D printing: every face of the thread — the flanks, the crest and the root — moved into the material by this much, so a printed bolt turns into a printed nut. Small values only: under {} at this pitch, where the tooth would have nothing left at the pitch diameter. A tenth or two of a millimetre is a usual start — half each on a printed bolt and nut.", mm(max_thread_offset(x.pitch, standard)));
    if let Some(v) = num(ui, &part, id.with("offset"), "Offset", Some(&offset_hint), x.offset, "offset", &mut x.exprs, None) {
        x.offset = v.abs();
        changed = true;
    }
    // 12. The reading.
    let spec = spec_of(x);
    let against = match (x.face.radius.filter(|_| !unpicked), internal) {
        (Some(r), Some(i)) if r > 0.0 => face_against_thread(&spec, i, 2.0 * r).map(|a| (a, r, i)),
        _ => None,
    };
    let brought = match against {
        None => String::new(),
        Some((a, r, i)) => {
            let what = match (i, a) {
                (true, FaceAgainst::Wider) => "filled in to the minor",
                (true, FaceAgainst::Narrower) => "bored out to the minor",
                (false, FaceAgainst::Wider) => "turned down to the major",
                (false, FaceAgainst::Narrower) => "built up to the major",
            };
            format!(" · the Ø{} {} {what}", bare3(2.0 * r), if i { "bore" } else { "boss" })
        }
    };
    let chamfer_words = match (x.chamfer.unwrap_or(false), internal) {
        (false, _) => String::new(),
        (true, None) => " · chamfered at the free ends".to_string(),
        (true, Some(i)) => format!(" · {} to Ø{} at the free ends", if i { "countersunk" } else { "chamfered" }, bare3(2.0 * chamfer_radius(&spec, i))),
    };
    hint(
        ui,
        format!(
            "{}{} · major Ø{} · minor Ø{} · {} mm deep{brought}{}{chamfer_words}",
            thread_name(standard, x.diameter, x.pitch),
            if x.left_handed { " left-hand" } else { "" },
            bare3(x.diameter),
            bare3(minor_diameter(x.diameter, x.pitch, standard)),
            bare3(thread_depth(x.pitch, standard)),
            if x.offset > 0.0 { format!(" · {} mm off every face", bare3(x.offset)) } else { String::new() },
        ),
    );
    // 13. The problem.
    if let Some(p) = thread_problem(&spec, units) {
        warn(ui, format!("⚠ {p}"));
    }
    changed
}

/// A click on a round face of any body takes it — on another body, the thread moves there; a
/// face that is not a full cylinder takes none. The face under the cursor is lit.
pub(crate) fn pick(ws: &mut ReverseWorkspace, vp: Viewport, p: Pos2, clicked: bool, x: &mut ThreadFeature, pick: &mut PickWhat) -> bool {
    let Some(Hit::Body { body: b, face, .. }) = ws.pick_hit(vp, p) else { return false };
    ws.view.hover_face = Some((b, face));
    if !clicked {
        return false;
    }
    let Some((print, full, length)) = ws.pick_view(b).and_then(|v| v.topology.faces.get(face).map(|f| (f.fingerprint, f.full_turn, round_length(&v.topology, face)))) else { return false };
    open_box(ws, x, pick);
    with_box(|st| take_face(x, st, b, print, full, length))
}

/// A face clicked, taken for the thread if it is a full round one (ScanRuler's
/// `toggleDraftFace` for a thread): on any body, the box's reading of it to come, and the whole
/// face long to start when it is the first. A fillet's part of a turn takes none.
fn take_face(x: &mut ThreadFeature, st: &mut ThreadBox, body: BodyId, print: FaceRef, full_turn: Option<bool>, length: f64) -> bool {
    if print.kind != FaceKind::Cylinder || full_turn == Some(false) {
        return false;
    }
    let fresh = is_unpicked(&x.face);
    st.internal = None;
    st.read_for = None;
    st.scan = None;
    if fresh {
        st.length = length;
        if x.length.is_some() {
            x.length = Some(length);
        }
    }
    x.body = body;
    x.face = print;
    true
}

/// Full length ticked or not: the length and the start kept while the thread runs the whole
/// face, and given back when it does not; the feature holds no length running the whole face.
fn set_full_length(x: &mut ThreadFeature, st: &mut ThreadBox, full: bool) {
    if full {
        if let Some(l) = x.length {
            st.length = l;
        }
        st.start = x.start;
        x.length = None;
        x.start = 0.0;
    } else {
        x.length = Some(st.length);
        x.start = st.start;
    }
}

/// The round face picked, its ends drawn on the part; and where a length is set, the circles
/// the thread runs between.
pub(crate) fn paint(ws: &ReverseWorkspace, painter: &egui::Painter, vp: Viewport, x: &ThreadFeature) {
    if is_unpicked(&x.face) {
        return;
    }
    let Some((i, t)) = face_on(ws, x.body, &x.face) else { return };
    let group = round_faces(&t, i);
    let mine = |h: &u32| group.contains(&(*h as usize));
    for e in t.edges.iter().filter(|e| e.faces.iter().any(mine) && !e.faces.iter().all(mine)) {
        let pts: Vec<Pos2> = e.polyline.iter().filter_map(|q| ws.project(vp, *q)).collect();
        if pts.len() >= 2 {
            painter.add(egui::Shape::line(pts, Stroke::new(3.0, PICKED)));
        }
    }
    let (Some(length), Some(center), Some(r)) = (x.length, x.face.center, x.face.radius) else { return };
    // The face's ends along its axis, off its edges.
    let axis = x.face.dir.normalize_or_zero();
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for e in t.edges.iter().filter(|e| e.faces.iter().any(mine)) {
        for q in &e.polyline {
            let s = (*q - center).dot(axis);
            lo = lo.min(s);
            hi = hi.max(s);
        }
    }
    let Some(span) = mm_reverse::thread::thread_span(lo, hi, Some(length), x.start, x.flip) else { return };
    let (u, v) = mm_reverse::linalg::ortho_basis(axis);
    for at in [span.from, span.to] {
        let pts: Vec<Pos2> = (0..=48).filter_map(|k| {
            let a = std::f64::consts::TAU * k as f64 / 48.0;
            ws.project(vp, center + axis * at + (u * a.cos() + v * a.sin()) * r)
        }).collect();
        if pts.len() >= 2 {
            painter.add(egui::Shape::dashed_line(&pts, Stroke::new(2.0, HINT), 5.0, 4.0));
        }
    }
}

// ---- the script's steps ----

/// The face of a body of the pick set nearest a point of the part: (body, face index).
pub(crate) fn face_near(ws: &ReverseWorkspace, p: DVec3) -> Option<(BodyId, usize)> {
    let mut best: Option<(f64, BodyId, usize)> = None;
    for b in ws.pick_set() {
        let m = &b.part_mesh;
        for (t, f) in m.triangles.iter().zip(&m.tri_face) {
            let [a, bb, c] = [t[0], t[1], t[2]].map(|i| m.positions.get(i as usize).copied().unwrap_or(DVec3::splat(f64::NAN)));
            let d = point_triangle(p, a, bb, c);
            if d.is_finite() && best.is_none_or(|(x, _, _)| d < x) {
                best = Some((d, b.id, *f as usize));
            }
        }
    }
    best.map(|(_, b, f)| (b, f))
}

/// How far a point lies from a triangle.
fn point_triangle(p: DVec3, a: DVec3, b: DVec3, c: DVec3) -> f64 {
    let n = (b - a).cross(c - a);
    let nn = n.length_squared();
    if nn > 0.0 {
        let q = p - n * ((p - a).dot(n) / nn);
        let inside = [(a, b), (b, c), (c, a)].iter().all(|(s, e)| (*e - *s).cross(q - *s).dot(n) >= 0.0);
        if inside {
            return (p - q).length();
        }
    }
    let seg = |s: DVec3, e: DVec3| {
        let d = e - s;
        let t = if d.length_squared() > 0.0 { ((p - s).dot(d) / d.length_squared()).clamp(0.0, 1.0) } else { 0.0 };
        (s + d * t - p).length()
    };
    seg(a, b).min(seg(b, c)).min(seg(c, a))
}

/// The script's `thread` step: a Thread on the round face nearest a point, with what the words
/// after it ask — `std=metric|unc|unf|print45`, `size=M10x1`, `d=`, `p=`, `lh`, `len=`, `start=`, `flip`,
/// `chamfer`, `offset=`, `scan` (the size read off the scan) — created, or left open with
/// `open`. Returns what it did, for the log.
pub(crate) fn script_step(ws: &mut ReverseWorkspace, args: &[&str]) -> String {
    let n = |i: usize| args.get(i).and_then(|v| v.parse::<f64>().ok());
    let (Some(px), Some(py), Some(pz)) = (n(0), n(1), n(2)) else { return "thread step: give a point on the round face, thread:x:y:z[:…]".into() };
    let Some((body, face)) = face_near(ws, DVec3::new(px, py, pz)) else { return "thread step: no body near the point".into() };
    let Some((print, length)) = ws.pick_view(body).and_then(|v| v.topology.faces.get(face).map(|f| (f.fingerprint, round_length(&v.topology, face)))) else { return "thread step: no face there".into() };
    let Ok(FeatureKind::Thread(mut x)) = new_feature(ws) else { return "thread step: no body".into() };
    x.body = body;
    x.face = print;
    // The standard's size for the face as drawn, as a click takes it.
    let mut b = ThreadBox { sized: false, length, ..ThreadBox::default() };
    if let Some(v) = ws.pick_view(body) {
        let m = &v.part_mesh;
        let mesh = FaceMesh { positions: &m.positions, triangles: &m.triangles, tri_face: &m.tri_face };
        let pk = RoundFacePick { face: Some(face as u32), center: print.center, radius: print.radius, dir: print.dir };
        if let Some(r) = thread_face_reading(mesh, &pk, core_standard(x.standard), false) {
            b.internal = Some(r.internal);
            if let (Some(d), Some(p)) = (r.diameter, r.pitch) {
                x.diameter = d;
                x.pitch = p;
            }
        }
    }
    let mut open = false;
    let mut said = Vec::new();
    for a in &args[3..] {
        let (k, v) = a.split_once('=').unwrap_or((a, ""));
        let num = v.parse::<f64>().ok();
        match (k, num) {
            ("std", _) => {
                if let Some(s) = THREAD_STANDARDS.iter().find(|s| s.id() == v) {
                    let (d, p) = size_in(&x, &b, *s);
                    x.standard = model_standard(*s);
                    x.diameter = d;
                    x.pitch = p;
                }
            }
            // A size as typed: "M10x1", "1/4-20 UNC", "Ø16×3.5 print".
            ("size", _) => {
                let named = read_designation(v, None).standard;
                let standard = named.unwrap_or(core_standard(x.standard));
                match thread_size_named(standard, v, None) {
                    Ok((d, p)) => {
                        x.standard = model_standard(standard);
                        x.diameter = d;
                        x.pitch = p;
                    }
                    Err(e) => said.push(e),
                }
            }
            ("d", Some(d)) => x.diameter = d,
            ("p", Some(p)) => x.pitch = p,
            ("lh", _) => x.left_handed = true,
            ("len", Some(l)) => x.length = Some(l),
            ("start", Some(s)) => x.start = s,
            ("flip", _) => x.flip = true,
            ("chamfer", _) => x.chamfer = Some(true),
            ("offset", Some(o)) => x.offset = o,
            ("open", _) => open = true,
            ("scan", _) => match read_scan(ws, &x) {
                Ok(r) => {
                    said.push(r.note.clone());
                    if let Some(f) = &r.feature {
                        take_suggestion(&mut x, f);
                    }
                }
                Err(e) => said.push(e),
            },
            _ => said.push(format!("{a}?")),
        }
    }
    let name = thread_name(core_standard(x.standard), x.diameter, x.pitch);
    let why = not_ready_why(&x);
    ws.open_feature(FeatureKind::Thread(x), None);
    if open {
        return format!("thread box open: {name} {}", said.join(" "));
    }
    match why {
        Some(w) => format!("thread not ready: {w}"),
        None => {
            ws.commit_feature();
            format!("thread {name} created {}", said.join(" "))
        }
    }
}

/// The script's `threadread` step: the thread read off the scan along the round face nearest a
/// point, the answer logged in the command's shape; `any` matches every standard.
pub(crate) fn script_read(ws: &mut ReverseWorkspace, args: &[&str]) -> String {
    let n = |i: usize| args.get(i).and_then(|v| v.parse::<f64>().ok());
    let (Some(px), Some(py), Some(pz)) = (n(0), n(1), n(2)) else { return "threadread step: give a point on the round face".into() };
    let Some((body, face)) = face_near(ws, DVec3::new(px, py, pz)) else { return "threadread step: no body near the point".into() };
    let Some(t) = ws.pick_view(body).map(|v| v.topology.clone()) else { return "threadread step: no body".into() };
    let name = super::body_name(ws, body);
    let clouds = scan_clouds(ws);
    let refs: Vec<(&[DVec3], &[DVec3])> = clouds.iter().map(|(p, n)| (p.as_slice(), n.as_slice())).collect();
    let any = args.contains(&"any");
    let ask = ReadAsk { standard: if any { None } else { Some(ThreadStandard::Metric) }, ..ReadAsk::default() };
    match thread_read(ReadAlong::Face { body, body_name: &name, topology: &t, face }, &refs, &ask) {
        Ok(r) => r.json.to_string(),
        Err(e) => format!("{}: {}", e.code, e.message),
    }
}

#[cfg(test)]
mod tests {
    //! ScanRuler's `reThread.test.ts`, "the Thread box": what a click takes, when the box is
    //! ready, and the length a thread runs the whole face with.
    use super::*;
    use mm_reverse::topo::FaceFingerprint;

    fn feature() -> ThreadFeature {
        ThreadFeature { body: BodyId::new(2), face: unpicked_face(), standard: ModelStandard::Metric, diameter: 10.0, pitch: 1.5, left_handed: false, length: None, start: 0.0, flip: false, offset: 0.0, chamfer: Some(false), exprs: Exprs::new() }
    }

    fn round(radius: f64) -> FaceFingerprint {
        FaceFingerprint { kind: FaceKind::Cylinder, centroid: DVec3::new(0.0, 0.0, 15.0), dir: DVec3::Z, area: 2.0 * std::f64::consts::PI * radius * 30.0, radius: Some(radius), center: Some(DVec3::new(0.0, 0.0, 15.0)), origin: 2 }
    }

    #[test]
    fn takes_a_round_face_and_no_other_the_whole_face_long_to_start_and_is_ready_once_it_has_one() {
        let mut x = feature();
        let mut st = ThreadBox::default();
        assert!(x.length.is_none() && x.standard == ModelStandard::Metric);
        assert!(!ready(&x));
        let flat = FaceFingerprint { kind: FaceKind::Plane, centroid: DVec3::new(0.0, 0.0, 30.0), dir: DVec3::Z, area: 200.0, radius: None, center: None, origin: 2 };
        assert!(!take_face(&mut x, &mut st, BodyId::new(2), flat, None, 10.0));
        assert!(is_unpicked(&x.face));
        // A fillet's quarter of a cylinder is round, but not the whole way.
        assert!(!take_face(&mut x, &mut st, BodyId::new(2), round(2.0), Some(false), 30.0));
        assert!(is_unpicked(&x.face));
        assert!(take_face(&mut x, &mut st, BodyId::new(2), round(8.0), Some(true), 30.0));
        assert_eq!(x.face, round(8.0));
        assert_eq!((st.internal, st.length), (None, 30.0));
        assert!(ready(&x));
    }

    #[test]
    fn is_not_ready_with_an_offset_past_what_the_pitch_takes_or_a_length_of_nothing() {
        let mut x = ThreadFeature { face: round(8.0), diameter: 16.0, pitch: 2.0, ..feature() };
        x.offset = 0.5;
        assert!(!ready(&x));
        x.offset = 0.2;
        x.length = Some(0.0);
        assert!(!ready(&x));
        x.length = Some(12.0);
        assert!(ready(&x));
    }

    #[test]
    fn the_whole_face_long_the_feature_holds_no_length_and_gives_the_length_back_when_let_go() {
        let mut x = ThreadFeature { face: round(8.0), length: Some(12.0), start: 3.0, flip: true, ..feature() };
        let mut st = ThreadBox { length: 30.0, ..ThreadBox::default() };
        set_full_length(&mut x, &mut st, true);
        assert_eq!((x.length, x.start), (None, 0.0));
        set_full_length(&mut x, &mut st, false);
        assert_eq!((x.length, x.start), (Some(12.0), 3.0));
    }

    #[test]
    fn a_standard_offers_the_faces_own_size_while_none_was_chosen_else_the_nearest() {
        let x = ThreadFeature { face: round(2.5), diameter: 5.0, pitch: 0.8, ..feature() };
        // A Ø5 bore drawn at M5's own diameter takes M5; a boss of Ø5 the size of its major.
        let bore = ThreadBox { internal: Some(true), ..ThreadBox::default() };
        assert_eq!(size_in(&x, &bore, ThreadStandard::Metric), (5.0, 0.8));
        let chosen = ThreadBox { internal: Some(true), sized: true, ..ThreadBox::default() };
        let (d, p) = size_in(&ThreadFeature { diameter: 6.3, ..x.clone() }, &chosen, ThreadStandard::Unc);
        assert!((d - 6.35).abs() < 1e-9 && (p - 25.4 / 20.0).abs() < 1e-9, "{d} {p}");
    }
}
