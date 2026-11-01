// SPDX-License-Identifier: AGPL-3.0-only
//! The Move face feature's box: the faces of a body to move (flat or round ones, each kept as
//! its fingerprint), how far — typed, dragged by the arrow on the face, rounded by its chip —
//! and Snap to scan: the scan clicked where the one face picked should lie, a plane or a
//! cylinder fitted there, and the move read off it (with the draft the scan's face leans by,
//! where it leans past its scatter). Where the faces trace back to the extrude or the sketch
//! that made them, the box offers to change that instead — no Move face added, everything after
//! it built again on the moved face — and Create edits the history, the draft after it, in one
//! Ctrl+Z. While the scan is being clicked the bodies are a faint tint with their edges, and a
//! move that edits the history is shown as a ghost of the faces where they will lie.
//!
//! Ported from ScanRuler (`plugins/reverse/ui/FeatureEditor.tsx` the move face fields,
//! `app/useReversePlugin.tsx` the scan click, `app/useReverseWorkspace.ts` the snapping view and
//! the moved-face ghost, `core/preview.ts` moveGrip, `state/reDrafts.ts` snapPlanOf,
//! historyEditOf and snapDraftOf, `state/reFeatureSlice.ts` toggleDraftFace and commitFeature,
//! `state/reFeatureHelpers.ts` leanAfterSnap, `state/reGripSlice.ts` the move grip), © Stefan
//! Hermann (CNC Kitchen), AGPL-3.0-only.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::mpsc::{channel, Receiver};
use std::sync::{Arc, Mutex, OnceLock};

use egui::{Color32, Pos2, RichText, Stroke};
use glam::DVec3;
use mm_core::Mesh;
use mm_reverse::display_units::{fmt_len, DisplayUnits, FmtOptions};
use mm_reverse::fit3d::fit_cylinder_from_seed::fit_cylinder_from_seed;
use mm_reverse::fit3d::fit_plane_from_seed::fit_plane_from_seed;
use mm_reverse::fit3d::types::{FitData, FitSettings};
use mm_reverse::js_number::to_fixed;
use mm_reverse::length_fields::len_short;
use mm_reverse::mesh_graph::MeshGraph;
use mm_reverse::model::{BodyId, DraftFeature, Exprs, FaceRef, FeatureId, FeatureKind, MoveFaceFeature, Part};
use mm_reverse::nominal::NominalKind;
use mm_reverse::snap_edit::{plan_moves, SnapPlan};
use mm_reverse::snap_face::{halves_of, moved_face_mesh, snap_draft_feature, snap_fit_kind, snap_reading, SnapFit, SnapReading};
use mm_reverse::topo::{describe_face, match_face, match_tolerance, FaceKind};

use super::draft::{box_key, clear_key, default_body, faces_facing, num_field, picked_as, same_box};
use super::paint_faces;
use crate::reverse::{Mode, PickWhat, ReverseWorkspace};
use crate::tools::Viewport;

pub(crate) const READY: bool = true;
pub(crate) const BUTTON: (&str, &str) = ("Move face", "Offset faces of the body by a distance — a flat face along its normal, a round face in radius; several picked together grow or shrink as a whole, a cylinder's wall and its ends. One face can be snapped onto the scan instead: pick it, take Snap to scan, and click the scan where it should lie");
pub(crate) const FIRST_PICK: PickWhat = PickWhat::Faces;

const MM: DisplayUnits = DisplayUnits::Mm;
/// The ghost's colour (a feature on a body previews in the body's colour) and the arrow's.
const GHOST: Color32 = Color32::from_rgba_premultiplied(51, 92, 63, 115);
const GRIP: Color32 = Color32::from_rgb(115, 204, 140);
/// An arrow's length on the screen, in points (§6.6).
const ARROW_PX: f32 = 44.0;
/// How near the pointer has to come to the arrow to take it, in points.
const ARROW_REACH: f32 = 8.0;

/// A fit of the scan under way: the face it is for and what it is fitting.
struct Fitting {
    rx: Receiver<Result<FitData, String>>,
    face: FaceRef,
    kind: SnapFit,
}

/// The arrow being dragged: the offset it started from, where the pointer took it, and the
/// arrow's direction on the screen with how many points a millimetre along it is.
#[derive(Clone, Copy)]
struct Drag {
    start: f64,
    from: Pos2,
    dir: egui::Vec2,
    px_per_mm: f32,
}

/// What the box keeps that the feature does not (ScanRuler's draft fields `snap`, `reading`,
/// `fitting`, `error`, `lean`, `history`).
struct MoveBox {
    frame: u64,
    key: Option<FeatureId>,
    snap: bool,
    /// What the scan said, and the face it said it of.
    reading: Option<(SnapReading, FaceRef)>,
    fitting: Option<Fitting>,
    error: Option<String>,
    lean: bool,
    history: bool,
    /// The faces as the box last saw them: a pick since lets the reading go.
    faces_seen: Vec<FaceRef>,
    drag: Option<Drag>,
    ctx: Option<egui::Context>,
    /// Set up by a script step for the box it opened: the next frame's box takes it as its own.
    adopt: bool,
}

impl MoveBox {
    fn new(frame: u64, key: Option<FeatureId>) -> MoveBox {
        MoveBox { frame, key, snap: false, reading: None, fitting: None, error: None, lean: false, history: false, faces_seen: Vec::new(), drag: None, ctx: None, adopt: false }
    }

    /// A face picked or let go: the reading off the scan was for the one face there was.
    fn picks_changed(&mut self) {
        self.snap = false;
        self.reading = None;
        self.lean = false;
        self.error = None;
        self.fitting = None;
    }
}

thread_local! {
    static BOX: RefCell<Option<MoveBox>> = const { RefCell::new(None) };
}

fn with_box<R>(ws: &ReverseWorkspace, frame: u64, f: impl FnOnce(&mut MoveBox) -> R) -> R {
    let key = box_key(ws);
    BOX.with(|b| {
        let mut b = b.borrow_mut();
        if !b.as_ref().is_some_and(|s| s.adopt || same_box(s.frame, s.key, frame, key)) {
            *b = Some(MoveBox::new(frame, key));
        }
        let s = b.get_or_insert_with(|| MoveBox::new(frame, key));
        s.frame = frame;
        s.key = key;
        s.adopt = false;
        f(s)
    })
}

/// The box's state as it stands, without opening one.
fn state<R>(f: impl FnOnce(&mut MoveBox) -> R) -> Option<R> {
    BOX.with(|b| b.borrow_mut().as_mut().map(f))
}

/// A box opened by a script step starts afresh.
fn fresh() {
    BOX.with(|b| *b.borrow_mut() = None);
}

/// The state a script step sets up for the box it opens.
fn adopted(faces: &[FaceRef], set: impl FnOnce(&mut MoveBox)) {
    BOX.with(|b| {
        let mut st = MoveBox::new(0, None);
        st.faces_seen = faces.to_vec();
        st.adopt = true;
        set(&mut st);
        *b.borrow_mut() = Some(st);
    });
}

pub(crate) fn new_feature(ws: &ReverseWorkspace) -> Result<FeatureKind, String> {
    let body = default_body(ws).ok_or("A Move face moves faces of a body: make a body first.")?;
    Ok(FeatureKind::MoveFace(MoveFaceFeature { body, faces: Vec::new(), faces_where: Vec::new(), offset: 0.0, exprs: Exprs::new() }))
}

/// The one face a box reads the scan for — the scan says where a single face should lie, not
/// several.
fn one_face(x: &MoveFaceFeature) -> Option<&FaceRef> {
    (x.faces.len() == 1).then(|| &x.faces[0])
}

/// How the faces being moved could be moved by an edit of the extrudes or the sketches that
/// made them — or why they cannot — or `None` while the box has no face or no distance yet. A
/// Move face reopened for editing is a move on the body and stays one.
fn plan_of(part: &Part, x: &MoveFaceFeature, editing: bool) -> Option<SnapPlan> {
    if x.faces.is_empty() || !(x.offset.abs() > 1e-6) {
        return None;
    }
    if editing {
        return Some(SnapPlan::Direct { why: "this Move face was made on the body".into() });
    }
    Some(plan_moves(&part.features, &x.faces, x.offset, &part.elements, MM))
}

/// The edit of the history the box makes in place of a feature: the plan, where there is one
/// and the box's tick asks for it.
fn history_edit(part: &Part, x: &MoveFaceFeature, editing: bool, history: bool) -> Option<SnapPlan> {
    if !history {
        return None;
    }
    plan_of(part, x, editing).filter(SnapPlan::edits)
}

/// The Draft feature a snapped face is followed by — the scan's flat face leaning past its
/// scatter, and the offer ticked — or `None`.
fn lean_draft(x: &MoveFaceFeature, editing: bool, st: &MoveBox) -> Option<DraftFeature> {
    let face = one_face(x)?;
    if !st.lean || editing || face.kind != FaceKind::Plane {
        return None;
    }
    let (reading, _) = st.reading.as_ref()?;
    let lean = reading.draft.as_ref()?;
    snap_draft_feature(x.body, face, x.offset, lean)
}

// ---- the box ----

pub(crate) fn ui(ws: &mut ReverseWorkspace, ui: &mut egui::Ui, x: &mut MoveFaceFeature, pick: &mut PickWhat) -> bool {
    let frame = ui.ctx().cumulative_frame_nr();
    let editing = box_key(ws).is_some();
    let ctx = ui.ctx().clone();
    let mut changed = false;
    // The box's state: a pick or a number changed since lets the reading go; a fit landed is
    // read.
    let landed = with_box(ws, frame, |st| {
        st.ctx = Some(ctx.clone());
        if st.faces_seen != x.faces {
            if !st.faces_seen.is_empty() || !x.faces.is_empty() {
                st.picks_changed();
            }
            st.faces_seen = x.faces.clone();
        }
        let done = st.fitting.as_ref().and_then(|f| f.rx.try_recv().ok().map(|r| (r, f.face, f.kind)));
        if done.is_some() {
            st.fitting = None;
        } else if st.fitting.is_some() {
            ctx.request_repaint_after(std::time::Duration::from_millis(50));
        }
        done
    });
    if let Some((result, face, kind)) = landed {
        changed |= take_fit(ws, x, result, face, kind);
    }
    let st_snap = state(|s| s.snap).unwrap_or(false);
    let one = one_face(x).copied();
    let several = x.faces.len() > 1;
    let round = one.is_some_and(|f| f.kind == FaceKind::Cylinder);
    let all_round = !x.faces.is_empty() && x.faces.iter().all(|f| f.kind == FaceKind::Cylinder);
    let snapping = st_snap && one.is_some();

    // No body field: the body is the one the faces are clicked on.
    ui.horizontal_wrapped(|ui| {
        ui.label(if several { "Faces" } else { "Face" });
        ui.label(match one {
            _ if x.faces.is_empty() => "Click the faces of the body to move — flat or round ones".to_string(),
            Some(f) => describe_face(&f),
            None => format!("{} faces picked", x.faces.len()),
        });
        if !x.faces.is_empty() && clear_key(ui) {
            x.faces.clear();
            x.faces_where.clear();
            changed = true;
        }
    });
    if one.is_some_and(|f| snap_fit_kind(&f).is_some()) {
        let (label, tip) = if snapping { ("Pick faces instead", "Click faces of the body again — to add to the move, or to let one go") } else { ("Snap to scan", "Read the move off the scan: click the scan where this face should lie, and a plane or a cylinder fitted there says how far") };
        if ui.add(egui::Button::new(label).selected(snapping)).on_hover_text(tip).clicked() {
            state(|s| {
                s.snap = !snapping;
                s.error = None;
            });
            changed = true;
        }
    }
    let label = if all_round {
        "Radius by"
    } else if several {
        "Offset"
    } else {
        "Move by"
    };
    let hint = if several {
        "How far every picked face moves: a flat face along its outward normal — out adds to the body, in cuts into it — a round face in radius, a boss and a bore alike. A cylinder’s wall and its two ends together make it longer and wider at once."
    } else if round {
        "The change of the round face’s radius, the axis kept: a hole bored out or sleeved, a boss built up or turned down. Typed, dragged by the arrow, or read off a cylinder fitted where the scan was clicked."
    } else {
        "How far along the face’s outward normal: out adds to the body, in cuts into it, the walls round the face running on as they stand. Typed, dragged by the arrow, or read off a plane fitted where the scan was clicked."
    };
    let was = x.offset;
    if num_field(ui, label, &mut x.offset, "mm", Some(NominalKind::Length), hint) {
        if x.offset != was {
            // Typed, the number is the operator's: the reading and its expression go.
            x.exprs.remove("offset");
            state(|s| {
                s.reading = None;
                s.lean = false;
            });
        }
        changed = true;
    }
    if let Some(r) = one.filter(|_| round).and_then(|f| f.radius) {
        let mm = |v: f64| len_short(v, MM, 2);
        ui.weak(format!("Ø {} → {}", mm(2.0 * r), mm(2.0 * (r + x.offset))));
    }
    let (error, fitting, reading, lean_on, history) = state(|s| (s.error.clone(), s.fitting.is_some(), s.reading.map(|r| r.0), s.lean, s.history)).unwrap_or((None, false, None, false, false));
    let fmt = |v: f64, d: usize| fmt_len(v, MM, d, FmtOptions::default());
    // What the scan said, or what to do next.
    match (&error, fitting) {
        (Some(e), _) => {
            ui.colored_label(Color32::from_rgb(240, 170, 90), format!("⚠ {e}"));
        }
        (None, true) => {
            ui.horizontal(|ui| {
                polymix_workspace::spinner(ui);
                ui.weak("Fitting the scan where you clicked…");
            });
        }
        (None, false) => {
            let text = if x.faces.is_empty() {
                "Click the faces to move — one, or several to offset them together: a cylinder’s wall and both its ends, every face of a block. Shift+drag a window takes many at once.".to_string()
            } else if snapping && reading.is_none() {
                "Now click the scan where this face should lie — or type how far it moves.".to_string()
            } else if !snapping {
                if several { "Click a face to add it or to let it go.".to_string() } else { "Type how far the face moves, or drag the arrow — or take Snap to scan and click the scan where it should lie. Click more faces to move them together.".to_string() }
            } else {
                let r = reading.unwrap_or(SnapReading { offset: 0.0, tilt_deg: 0.0, axis_shift: None, sigma: 0.0, draft: None });
                let mut s = format!("Read off the scan: a {} fitted to σ {}", if round { "cylinder" } else { "plane" }, fmt(r.sigma, 3));
                if round {
                    s += &format!(", its radius {} {}", fmt(r.offset.abs(), 3), if r.offset >= 0.0 { "larger" } else { "smaller" });
                } else {
                    s += &format!(", {} {} the face", fmt(r.offset.abs(), 3), if r.offset >= 0.0 { "outside" } else { "inside" });
                }
                s += ".";
                if let Some(l) = r.draft {
                    let noise = if l.noise_deg < 0.005 { "under 0.01".to_string() } else { to_fixed(l.noise_deg, 2) };
                    s += &format!(" The scan's face leans {}° to this one — more than its scatter accounts for ({noise}°) — and the move alone keeps the body's direction.", to_fixed(l.angle_deg, 2));
                } else if r.tilt_deg > 0.5 {
                    s += &format!(" The scan's face is turned {}° to this one — the move keeps the body's direction.", to_fixed(r.tilt_deg, 1));
                }
                if let Some(shift) = r.axis_shift.filter(|a| *a > 0.1) {
                    s += &format!(" Its axis stands {} off this one's — the move keeps the body's axis.", fmt(shift, 2));
                }
                s
            };
            ui.add(egui::Label::new(RichText::new(text).weak()).wrap());
        }
    }
    let lean = reading.and_then(|r| r.draft);
    if let Some(l) = lean.filter(|_| !editing) {
        let mut on = lean_on;
        let tip = format!("A Draft feature after the move: the moved face turns {}° about the line across its middle where the scan's plane crosses it, and lies on the scan. A moulded wall modelled square gets its draft this way. Everything built after builds on the leaned face; one Ctrl+Z takes the move and the draft back together.", to_fixed(l.angle_deg, 2));
        if ui.checkbox(&mut on, format!("Draft it as well, by {}°", to_fixed(l.angle_deg, 2))).on_hover_text(tip).changed() {
            state(|s| s.lean = on);
            changed = true;
        }
    }
    let plan = plan_of(&ws.part, x, editing);
    if plan.as_ref().is_some_and(SnapPlan::edits) {
        let mut on = history;
        if ui.checkbox(&mut on, format!("Change what made {} instead", if several { "them" } else { "it" })).on_hover_text("Change what made the face — the extrude's distance, the sketch's line or circle — as you would by hand, instead of adding a Move face: the rounds, shells and cuts after it build again on the moved wall").changed() {
            state(|s| s.history = on);
            changed = true;
        }
    }
    if let Some(plan) = &plan {
        let edit = if history { plan.text().filter(|_| plan.edits()) } else { None };
        let leaning = state(|s| lean_draft(x, editing, s).is_some()).unwrap_or(false);
        let mut s = match edit {
            Some(text) => format!("Create edits {text} — no Move face is added. Everything built after it — rounds, shells, cuts — builds again on the moved {}; Ctrl+Z takes it back.", if several { "faces" } else { "face" }),
            None => {
                let tail = match plan {
                    SnapPlan::Direct { why } => format!(" — what made {} cannot be changed instead, since {why}.", if several { "them" } else { "it" }),
                    other => format!(" — tick the box to change {} instead, and they do.", other.text().unwrap_or_default().split(':').next().unwrap_or_default()),
                };
                format!("A Move face on the body: the walls round {} must stand square to it, and rounds on {} do not follow{tail}", if several { "each face" } else { "the face" }, if several { "their rims" } else { "its rim" })
            }
        };
        if let (true, Some(l)) = (leaning, lean) {
            s += &format!(" Then a Draft leans the moved face {}° onto the scan's plane, about the line across its middle — one Ctrl+Z takes both back.", to_fixed(l.angle_deg, 2));
        }
        ui.add(egui::Label::new(RichText::new(s).weak()).wrap());
    }
    *pick = PickWhat::Faces;
    changed
}

/// A fit of the scan landed: the reading it gives the face, the move read off it — or why
/// there is none. Returns whether the feature changed.
fn take_fit(ws: &mut ReverseWorkspace, x: &mut MoveFaceFeature, result: Result<FitData, String>, face: FaceRef, kind: SnapFit) -> bool {
    if one_face(x) != Some(&face) {
        return false;
    }
    let fit = match result {
        Ok(f) => f,
        Err(e) => {
            state(|s| s.error = Some(format!("No {} fits where you clicked: {e}", kind.name())));
            return false;
        }
    };
    match snap_reading(&face, &fit) {
        Err(e) => {
            state(|s| s.error = Some(e));
            false
        }
        Ok(reading) => {
            x.offset = mm_reverse::round::round3(reading.offset);
            x.exprs.remove("offset");
            state(|s| {
                s.error = None;
                s.reading = Some((reading, face));
            });
            let lean = reading.draft.map_or(String::new(), |d| format!(", which leans {}° to it — tick “Draft it as well” in the box to lean the face too", to_fixed(d.angle_deg, 2)));
            ws.status = format!("The face moves {} by {} onto the scan{lean} — Create, or click the scan again.", if reading.offset >= 0.0 { "out" } else { "in" }, fmt_len(reading.offset.abs(), MM, 3, FmtOptions::default()));
            true
        }
    }
}

// ---- the scan fitted where it is clicked ----

/// The scan objects' mesh graphs, built once per object and shape, off the UI thread.
fn graphs() -> &'static Mutex<HashMap<u64, (u64, Arc<MeshGraph>)>> {
    static GRAPHS: OnceLock<Mutex<HashMap<u64, (u64, Arc<MeshGraph>)>>> = OnceLock::new();
    GRAPHS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn graph_of(id: u64, mesh: &Mesh) -> Result<Arc<MeshGraph>, String> {
    let version = mesh.shape_version() ^ mesh.topology_version().rotate_left(32);
    if let Ok(g) = graphs().lock() {
        if let Some((v, g)) = g.get(&id) {
            if *v == version {
                return Ok(g.clone());
            }
        }
    }
    let g = Arc::new(MeshGraph::from_mesh(mesh, false).map_err(|e| e.0)?);
    if let Ok(mut all) = graphs().lock() {
        all.insert(id, (version, g.clone()));
    }
    Ok(g)
}

/// The element a face snaps onto, fitted to the scan from the vertices clicked, in the part's
/// coordinates (the scan is held in the view's).
fn fit_scan(mesh: &Mesh, id: u64, seeds: &[u32], kind: SnapFit, flip: bool) -> Result<FitData, String> {
    let g = graph_of(id, mesh)?;
    let settings = FitSettings::default();
    let out = |p: DVec3| if flip { crate::app::flip_zy_out(p) } else { p };
    match kind {
        SnapFit::Plane => {
            let r = fit_plane_from_seed(&g, seeds, &settings, None, None).map_err(|e| e.0)?;
            let mut f = r.fit;
            f.center = out(f.center);
            f.normal = out(f.normal);
            f.basis_u = out(f.basis_u);
            f.basis_v = out(f.basis_v);
            Ok(FitData::Plane(f))
        }
        SnapFit::Cylinder => {
            let r = fit_cylinder_from_seed(&g, seeds, &settings, None, None).map_err(|e| e.0)?;
            let mut f = r.fit;
            f.center = out(f.center);
            f.axis = out(f.axis);
            Ok(FitData::Cylinder(f))
        }
    }
}

/// The scan under a screen point, whatever body lies over it: the object, its mesh and the
/// triangle hit.
fn scan_under(ws: &mut ReverseWorkspace, vp: Viewport, p: Pos2) -> Option<(u64, Arc<Mesh>, u32)> {
    let ray = ws.view.camera.ray_from_ndc(vp.ndc(p), vp.aspect());
    let objects: Vec<crate::reverse::ScanObject> = ws.scan.iter().filter(|o| o.visible).cloned().collect();
    let mut best: Option<(f64, u64, Arc<Mesh>, u32)> = None;
    for o in &objects {
        let Some(bvh) = ws.scan_bvh(o) else { continue };
        if let Some(h) = bvh.raycast(&o.mesh, &ray) {
            if best.as_ref().is_none_or(|b| h.t < b.0) {
                best = Some((h.t, o.id, o.mesh.clone(), h.tid));
            }
        }
    }
    best.map(|b| (b.1, b.2, b.3))
}

/// Fit the scan at a triangle for the one face picked, on a worker thread.
fn start_fit(ws: &mut ReverseWorkspace, face: FaceRef, kind: SnapFit, id: u64, mesh: Arc<Mesh>, tid: u32) {
    let seeds = mesh.triangle(tid).to_vec();
    let flip = ws.flip;
    let (tx, rx) = channel();
    std::thread::Builder::new()
        .name("reverse snap fit".into())
        .spawn(move || {
            let _ = tx.send(fit_scan(&mesh, id, &seeds, kind, flip));
        })
        .ok();
    state(|s| {
        s.fitting = Some(Fitting { rx, face, kind });
        s.error = None;
    });
}

// ---- the arrow on the face ----

/// The arrow a Move face is dragged by (ScanRuler's `moveGrip`): on a flat face out along its
/// normal from where the face will lie; on a round face radially, at the new radius.
fn grip_of(x: &MoveFaceFeature) -> Option<(DVec3, DVec3)> {
    let face = x.faces.first()?;
    if face.kind == FaceKind::Cylinder {
        if let Some(r) = face.radius {
            let axis = face.dir.try_normalize().unwrap_or(DVec3::Z);
            // Square to the axis, off whichever coordinate axis it is furthest from.
            let reference = if axis.x.abs() < 0.6 {
                DVec3::X
            } else if axis.y.abs() < 0.6 {
                DVec3::Y
            } else {
                DVec3::Z
            };
            let out = axis.cross(reference).try_normalize().unwrap_or(DVec3::X);
            return Some((face.center.unwrap_or(face.centroid) + out * (r + x.offset), out));
        }
    }
    Some((face.centroid + face.dir * x.offset, face.dir))
}

/// The arrow on the screen: its foot, its tip, its direction and how many points a millimetre
/// along it is.
fn arrow_on_screen(ws: &ReverseWorkspace, vp: Viewport, x: &MoveFaceFeature) -> Option<(Pos2, Pos2, egui::Vec2, f32)> {
    let (at, dir) = grip_of(x)?;
    let foot = ws.project(vp, at)?;
    let ahead = ws.project(vp, at + dir)?;
    let d = ahead - foot;
    let px_per_mm = d.length();
    if !(px_per_mm > 1e-3) {
        return None;
    }
    let unit = d / px_per_mm;
    Some((foot, foot + unit * ARROW_PX, unit, px_per_mm))
}

fn near_arrow(p: Pos2, foot: Pos2, tip: Pos2) -> bool {
    let ab = tip - foot;
    let t = ((p - foot).dot(ab) / ab.length_sq().max(1e-6)).clamp(0.0, 1.0);
    (foot + ab * t - p).length() <= ARROW_REACH
}

/// The arrow taken and dragged: the offset follows the pointer along it, to the hundredth.
/// Returns (whether the offset changed, whether the pointer is on the arrow or dragging it).
fn drag_arrow(ws: &ReverseWorkspace, vp: Viewport, p: Pos2, x: &mut MoveFaceFeature) -> (bool, bool) {
    let Some(ctx) = state(|s| s.ctx.clone()).flatten() else { return (false, false) };
    let (pressed, down, mods) = ctx.input(|i| (i.pointer.button_pressed(egui::PointerButton::Primary), i.pointer.button_down(egui::PointerButton::Primary), i.modifiers));
    if let Some(d) = state(|s| s.drag).flatten() {
        if !down {
            state(|s| s.drag = None);
            return (false, true);
        }
        let along = (p - d.from).dot(d.dir) / d.px_per_mm;
        let next = mm_reverse::round::round_to(d.start + along as f64, 2);
        if next != x.offset {
            x.offset = next;
            x.exprs.remove("offset");
            state(|s| {
                s.reading = None;
                s.lean = false;
            });
            ctx.request_repaint();
            return (true, true);
        }
        return (false, true);
    }
    let Some((foot, tip, dir, px_per_mm)) = arrow_on_screen(ws, vp, x) else { return (false, false) };
    let on = near_arrow(p, foot, tip);
    // A button that moves the view is the view's.
    if on && pressed && !ws.nav.bindings.starts_navigation(egui::PointerButton::Primary, mods) {
        state(|s| s.drag = Some(Drag { start: x.offset, from: p, dir, px_per_mm }));
    }
    (false, on)
}

// ---- picks ----

/// A click: the arrow taken; while snapping, the scan where the face should lie; otherwise a
/// face of the body picked or let go. The face under the cursor is lit.
pub(crate) fn pick(ws: &mut ReverseWorkspace, vp: Viewport, p: Pos2, clicked: bool, x: &mut MoveFaceFeature, _pick: &mut PickWhat) -> bool {
    let (dragged, on_arrow) = drag_arrow(ws, vp, p, x);
    if dragged || on_arrow {
        return dragged;
    }
    let snapping = state(|s| s.snap).unwrap_or(false) && one_face(x).is_some();
    if snapping {
        if clicked {
            let Some(face) = one_face(x).copied() else { return false };
            let Some(kind) = snap_fit_kind(&face) else { return false };
            match scan_under(ws, vp, p) {
                Some((id, mesh, tid)) => start_fit(ws, face, kind, id, mesh, tid),
                None => ws.status = "Click the scan where this face should lie.".into(),
            }
        }
        return false;
    }
    let hit = ws.pick_hit(vp, p);
    let (body, face) = match hit {
        // Faces to move are clicked on their body: any body for the first, then its body.
        Some(crate::reverse::view::Hit::Body { body, .. }) if !x.faces.is_empty() && body != x.body => return false,
        Some(crate::reverse::view::Hit::Body { body, face, .. }) => (body, face),
        Some(crate::reverse::view::Hit::Scan { .. }) => {
            if clicked {
                ws.status = if x.faces.is_empty() {
                    "Click the faces of the body to move — flat or round ones.".into()
                } else if x.faces.len() > 1 {
                    "Type how far the faces move, or drag the arrow. The scan is read for one face at a time.".into()
                } else {
                    "Type how far the face moves — or take Snap to scan in the box and click the scan where it should lie.".into()
                };
            }
            return false;
        }
        None => return false,
    };
    ws.view.hover_face = Some((body, face));
    if !clicked {
        return false;
    }
    let Some(view) = ws.pick_view(body) else { return false };
    let Some(info) = view.topology.faces.get(face).cloned() else { return false };
    let is_it = picked_as(view, face);
    let print = info.fingerprint;
    if !matches!(print.kind, FaceKind::Plane | FaceKind::Cylinder) {
        state(|s| s.error = Some("Only a flat or a cylindrical face can be moved.".into()));
        return false;
    }
    if info.full_turn == Some(false) {
        state(|s| s.error = Some("A round face is resized whole — this one stops short of a full turn, as a fillet’s does.".into()));
        return false;
    }
    if x.body != body {
        x.body = body;
        x.faces.clear();
        x.faces_where.clear();
    }
    match x.faces.iter().position(is_it) {
        Some(k) => {
            x.faces.remove(k);
        }
        None => x.faces.push(print),
    }
    state(|s| {
        s.picks_changed();
        s.faces_seen = x.faces.clone();
    });
    true
}

// ---- painting ----

/// The faces of the pick set's body a pick names — each with its other half, where the kernel
/// keeps it as two (a bore) — by their index in its mesh.
fn mesh_faces(ws: &ReverseWorkspace, x: &MoveFaceFeature) -> Vec<(u32, FaceRef)> {
    let Some(b) = ws.pick_view(x.body) else { return Vec::new() };
    let prints = b.topology.face_prints();
    let tol = match_tolerance(b.mesh.bounds().diagonal());
    let mut out: Vec<(u32, FaceRef)> = Vec::new();
    for p in &x.faces {
        let Some(i) = match_face(p, &prints, &tol) else { continue };
        for h in halves_of(&b.topology, i) {
            if let Some(f) = b.topology.faces.get(h).filter(|f| !out.iter().any(|o| o.0 == f.hash)) {
                out.push((f.hash, *p));
            }
        }
    }
    out
}

pub(crate) fn paint(ws: &ReverseWorkspace, painter: &egui::Painter, vp: Viewport, x: &MoveFaceFeature) {
    paint_faces(ws, painter, vp, x.body, &x.faces);
    // A move that edits the history, or is followed by a draft, has no one step for the kernel
    // to preview: what is shown is where the faces will lie. Any other move the kernel
    // previews, snapped or typed.
    if ghost_only_of(ws, x, box_key(ws).is_some()) {
        let lean = state(|s| s.reading.and_then(|r| r.0.draft).filter(|_| s.lean && x.faces.len() == 1)).flatten();
        if let Some(b) = ws.pick_view(x.body) {
            if let Some((positions, _)) = moved_face_mesh(&b.part_mesh, &mesh_faces(ws, x), x.offset, lean.as_ref()) {
                let mut mesh = egui::Mesh::default();
                for tri in positions.chunks(3) {
                    let pts: Vec<Pos2> = tri.iter().filter_map(|q| ws.project(vp, *q)).collect();
                    if pts.len() == 3 {
                        let i = mesh.vertices.len() as u32;
                        for q in pts {
                            mesh.colored_vertex(q, GHOST);
                        }
                        mesh.add_triangle(i, i + 1, i + 2);
                    }
                }
                painter.add(egui::Shape::mesh(mesh));
            }
        }
    }
    // The arrow, white while the pointer is on it or drags it.
    if let Some((foot, tip, dir, _)) = arrow_on_screen(ws, vp, x) {
        let hot = state(|s| s.drag.is_some()).unwrap_or(false) || painter.ctx().pointer_hover_pos().is_some_and(|p| near_arrow(p, foot, tip));
        let c = if hot { Color32::WHITE } else { GRIP };
        let side = egui::vec2(-dir.y, dir.x) * 5.0;
        let neck = tip - dir * 12.0;
        painter.line_segment([foot, neck], Stroke::new(2.5, c));
        painter.add(egui::Shape::convex_polygon(vec![tip, neck + side, neck - side], c, Stroke::NONE));
        painter.circle_filled(foot, 3.0, c);
    }
}

// ---- what Create waits for ----

/// Why the Move face cannot be created yet (ScanRuler's `featureDraftReady`): no face, or no
/// distance; `None` when it can.
pub(super) fn not_ready(x: &MoveFaceFeature) -> Option<&'static str> {
    if x.faces.is_empty() {
        return Some("Create waits for a face to move: click one on the body.");
    }
    if !(x.offset.abs() > 1e-6) {
        return Some("Create waits for how far the face moves: type it, drag the arrow, or take Snap to scan.");
    }
    None
}

// ---- what the workspace asks of an open Move face ----

fn open_move_face(ws: &ReverseWorkspace) -> Option<(&MoveFaceFeature, bool)> {
    match &ws.mode {
        Mode::Feature(edit) => match &edit.kind {
            FeatureKind::MoveFace(x) => Some((x, edit.editing.is_some())),
            _ => None,
        },
        _ => None,
    }
}

/// Whether the open Move face is shown as a ghost of the faces where they will lie, rather than
/// built: a move that edits the history, or is followed by a draft.
fn ghost_only_of(ws: &ReverseWorkspace, x: &MoveFaceFeature, editing: bool) -> bool {
    let Some((history, lean)) = state(|s| (s.history, lean_draft(x, editing, s).is_some())) else { return false };
    lean || history_edit(&ws.part, x, editing, history).is_some()
}

/// Whether the part is shown without the open box's feature in it: a Move face shown as a
/// ghost.
pub(crate) fn previews_by_ghost(ws: &ReverseWorkspace, kind: &FeatureKind, editing: bool) -> bool {
    match kind {
        FeatureKind::MoveFace(x) if !editing => ghost_only_of(ws, x, editing),
        _ => false,
    }
}

/// Whether the bodies are drawn faint: a face is being snapped, and the scan (shown) is
/// clicked next.
pub(crate) fn bodies_faint(ws: &ReverseWorkspace) -> bool {
    let Some((x, _)) = open_move_face(ws) else { return false };
    ws.show_scan && one_face(x).is_some() && state(|s| s.snap).unwrap_or(false)
}

/// What Create reads for a Move face that edits the history.
pub(crate) fn create_label(ws: &ReverseWorkspace, x: &MoveFaceFeature, editing: bool) -> Option<&'static str> {
    let history = state(|s| s.history).unwrap_or(false);
    history_edit(&ws.part, x, editing, history).map(|_| if x.faces.len() > 1 { "Move the faces" } else { "Move the face" })
}

/// Create, where a Move face is made otherwise than as a feature on its own: the history edited
/// in its place, or the Draft the scan asked for added after it — one Undo step either way
/// (ScanRuler's `commitFeature` with `leanAfterSnap`). Returns whether it was made so.
pub(crate) fn commit_instead(ws: &mut ReverseWorkspace) -> bool {
    let Some((x, editing)) = open_move_face(ws) else { return false };
    let x = x.clone();
    let Some((history, lean)) = state(|s| (s.history, lean_draft(&x, editing, s))) else { return false };
    let plan = history_edit(&ws.part, &x, editing, history);
    if plan.is_none() && lean.is_none() {
        return false;
    }
    ws.mode = Mode::Idle;
    let text = plan.as_ref().and_then(|p| p.text()).map(str::to_string);
    let features = plan.as_ref().and_then(|p| p.features()).map(<[_]>::to_vec);
    ws.edit_part(|p| made_instead(p, &x, features, lean));
    ws.status = match text {
        Some(t) => format!("Edited {t} — no Move face added; Ctrl+Z takes it back."),
        None => "Move face and Draft made — one Ctrl+Z takes both back.".into(),
    };
    fresh();
    true
}

/// The part as Create leaves it where a Move face is made otherwise than on its own: the
/// features as the history edit left them, or the Move face added — and the Draft the scan asked
/// for after either (ScanRuler's `leanAfterSnap`).
fn made_instead(p: &mut Part, x: &MoveFaceFeature, features: Option<Vec<mm_reverse::model::Feature>>, lean: Option<DraftFeature>) {
    match features {
        Some(f) => p.features = f,
        None => {
            p.add(FeatureKind::MoveFace(x.clone()));
        }
    }
    if let Some(d) = lean {
        p.add(FeatureKind::Draft(d));
    }
}

// ---- script steps (tests) ----

/// `moveface:<offset>:<face>[,<face>…][:history][:create]` — a Move face on the last solid
/// body, its faces the largest facing each way named (`+z`, `-x`, `cyl` …); `history` ticks
/// Change what made it instead; `create` makes it at once.
pub(super) fn script(ws: &mut ReverseWorkspace, args: &[&str]) {
    let Some(body) = default_body(ws) else {
        ws.status = "moveface step: there is no body.".into();
        return;
    };
    let offset = args.first().and_then(|a| a.parse::<f64>().ok()).unwrap_or(0.0);
    let faces: Vec<FaceRef> = args.get(1).map(|list| list.split(',').filter_map(|w| faces_facing(ws, body, w).into_iter().next()).collect()).unwrap_or_default();
    adopted(&faces, |st| st.history = args.contains(&"history"));
    ws.open_feature(FeatureKind::MoveFace(MoveFaceFeature { body, faces, faces_where: Vec::new(), offset, exprs: Exprs::new() }), None);
    if args.contains(&"create") {
        ws.commit_feature();
    }
}

/// A Move face box opened on faces of a body — Snap to scan pressed, for one face snapped
/// (ScanRuler's `startFeature('moveFace', { body, faces, snap })`, what the Move face, Move
/// faces and Snap to scan chips on body faces start).
pub(super) fn start(ws: &mut ReverseWorkspace, body: BodyId, faces: Vec<FaceRef>, snap: bool) {
    // The box's state first: the build the box asks for as it opens reads it.
    let snap = snap && faces.len() == 1;
    adopted(&faces, |st| st.snap = snap);
    ws.open_feature(FeatureKind::MoveFace(MoveFaceFeature { body, faces, faces_where: Vec::new(), offset: 0.0, exprs: Exprs::new() }), None);
    if snap {
        ws.status = "Now click the scan where this face should lie.".into();
    }
}

/// `snapface:<face>` — a Move face on the last solid body with that one face, Snap to scan on.
/// `snapat:<x>:<y>:<z>[:lean][:history][:create]` — the scan clicked as from in front of the
/// part point given (a ray back along the open Move face's one face's normal) and fitted there,
/// at once (not on a worker thread).
pub(super) fn snap_script(ws: &mut ReverseWorkspace, cmd: &str, args: &[&str]) {
    if cmd == "snapface" {
        let Some(body) = default_body(ws) else { return };
        let Some(face) = args.first().and_then(|w| faces_facing(ws, body, w).into_iter().next()) else {
            ws.status = "snapface step: no such face.".into();
            return;
        };
        start(ws, body, vec![face], true);
        return;
    }
    // snapat
    let p = DVec3::new(args.first().and_then(|v| v.parse().ok()).unwrap_or(0.0), args.get(1).and_then(|v| v.parse().ok()).unwrap_or(0.0), args.get(2).and_then(|v| v.parse().ok()).unwrap_or(0.0));
    let Mode::Feature(edit) = &ws.mode else { return };
    let FeatureKind::MoveFace(x) = &edit.kind else { return };
    let Some(face) = one_face(x).copied() else { return };
    let Some(kind) = snap_fit_kind(&face) else { return };
    // As a click from in front of the face would land: a ray from the point's side of the face,
    // back along its normal.
    let ray = mm_core::geom::Ray { origin: ws.to_scene(p + face.dir * 50.0), dir: ws.to_scene(-face.dir).normalize_or_zero() };
    let objects: Vec<crate::reverse::ScanObject> = ws.scan.iter().filter(|o| o.visible).cloned().collect();
    let mut best: Option<(f64, &crate::reverse::ScanObject, u32)> = None;
    for o in &objects {
        let bvh = mm_core::bvh::Bvh::build(&o.mesh);
        if let Some(h) = bvh.raycast(&o.mesh, &ray) {
            if best.is_none_or(|b| h.t < b.0) {
                best = Some((h.t, o, h.tid));
            }
        }
    }
    let Some((_, o, t)) = best else {
        ws.status = "snapat step: no scan in front of that point.".into();
        return;
    };
    let seeds = o.mesh.triangle(t).to_vec();
    let result = fit_scan(&o.mesh, o.id, &seeds, kind, ws.flip);
    let mut edit = match std::mem::replace(&mut ws.mode, Mode::Idle) {
        Mode::Feature(e) => e,
        other => {
            ws.mode = other;
            return;
        }
    };
    if let FeatureKind::MoveFace(x) = &mut edit.kind {
        take_fit(ws, x, result, face, kind);
        state(|s| {
            s.lean = args.contains(&"lean");
            s.history = args.contains(&"history");
        });
    }
    ws.mode = Mode::Feature(edit);
    let (error, reading) = state(|s| (s.error.clone(), s.reading.map(|r| r.0))).unwrap_or((None, None));
    log::info!("reverse snap: {}", match (error, reading) {
        (Some(e), _) => e,
        (None, Some(r)) => format!("offset {:.4} tilt {:.3}° σ {:.4} draft {:?}", r.offset, r.tilt_deg, r.sigma, r.draft.map(|d| d.angle_deg)),
        _ => "no reading".into(),
    });
    ws.request_build();
    if args.contains(&"create") {
        ws.commit_feature();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_reverse::model::BodyId;

    fn face(dir: DVec3) -> FaceRef {
        FaceRef { kind: FaceKind::Plane, centroid: DVec3::new(20.0, 10.0, 10.0), dir, area: 800.0, radius: None, center: None, origin: 2 }
    }

    #[test]
    fn the_arrow_stands_out_along_a_flat_faces_normal_and_radially_at_a_round_faces_new_radius() {
        let mut x = MoveFaceFeature { body: BodyId::new(2), faces: vec![face(DVec3::Z)], faces_where: Vec::new(), offset: 1.5, exprs: Exprs::new() };
        let (at, dir) = grip_of(&x).expect("a grip");
        assert_eq!(at, DVec3::new(20.0, 10.0, 11.5));
        assert_eq!(dir, DVec3::Z);
        x.faces = vec![FaceRef { kind: FaceKind::Cylinder, centroid: DVec3::new(0.0, 0.0, 5.0), dir: DVec3::Z, area: 100.0, radius: Some(3.0), center: Some(DVec3::new(0.0, 0.0, 5.0)), origin: 2 }];
        x.offset = 0.5;
        let (at, dir) = grip_of(&x).expect("a grip");
        assert!((at - DVec3::new(0.0, 0.0, 5.0)).length() - 3.5 < 1e-12);
        assert!(dir.dot(DVec3::Z).abs() < 1e-12);
    }

    #[test]
    fn a_move_reopened_for_editing_stays_on_the_body_and_none_is_planned_before_it_has_a_face_and_a_distance() {
        let part = Part::default();
        let mut x = MoveFaceFeature { body: BodyId::new(2), faces: Vec::new(), faces_where: Vec::new(), offset: 2.0, exprs: Exprs::new() };
        assert!(plan_of(&part, &x, false).is_none());
        x.faces.push(face(DVec3::Z));
        x.offset = 0.0;
        assert!(plan_of(&part, &x, false).is_none());
        x.offset = 2.0;
        assert_eq!(plan_of(&part, &x, true).and_then(|p| p.why().map(str::to_string)).as_deref(), Some("this Move face was made on the body"));
        // A face nothing traces back to is moved on the body.
        assert!(plan_of(&part, &x, false).is_some_and(|p| !p.edits()));
    }

    /// The 40 × 20 × 10 block: sketch 1, extrude 2 (ScanRuler's reSnapDraft `block`).
    fn block() -> Part {
        use mm_reverse::build::default_regions;
        use mm_reverse::frame::WorldAxis;
        use mm_reverse::model::{BodyOp, ExtrudeExtent, ExtrudeFeature, SketchFeature};
        use mm_reverse::regions::find_regions;
        let mut part = Part::default();
        let mut s = mm_reverse::sketch::Sketch::default();
        s.add_rectangle(glam::DVec2::ZERO, glam::DVec2::new(40.0, 20.0));
        let regions = default_regions(&find_regions(&s));
        let sk = part.add(FeatureKind::Sketch(SketchFeature::on_world(WorldAxis::Z, 0.0, s)));
        part.add(FeatureKind::Extrude(ExtrudeFeature::new(sk, regions, ExtrudeExtent::Distance { from: 0.0, to: 10.0 }, BodyOp::New)));
        part
    }

    /// The scan's top 2 mm above the block's, dipping 3° along +X: the reading and its draft.
    fn dipping(top: &FaceRef) -> SnapReading {
        use mm_reverse::fit3d::types::{FitBase, PlaneFit};
        let t = 3f64.to_radians();
        let fit = FitData::Plane(PlaneFit { base: FitBase { sigma: 0.02, used_points: 500, region_size: 500, ..FitBase::default() }, center: DVec3::new(20.0, 10.0, 12.0), normal: DVec3::new(t.sin(), 0.0, t.cos()), basis_u: DVec3::new(t.cos(), 0.0, -t.sin()), basis_v: DVec3::Y, extent_u: 15.0, extent_v: 8.0 });
        snap_reading(top, &fit).unwrap_or_else(|e| panic!("{e}"))
    }

    fn kinds(p: &Part) -> Vec<&'static str> {
        p.features.iter().map(|f| f.kind.kind()).collect()
    }

    #[test]
    fn edits_the_extrude_and_adds_draft_1_after_it_or_after_a_move_face_made_on_the_body() {
        // ScanRuler's reSnapDraft: 'the store committing a snap with its draft'.
        let part = block();
        let top = face(DVec3::Z);
        let x = MoveFaceFeature { body: BodyId::new(2), faces: vec![top], faces_where: Vec::new(), offset: mm_reverse::round::round3(dipping(&top).offset), exprs: Exprs::new() };
        let mut st = MoveBox::new(0, None);
        st.reading = Some((dipping(&top), top));
        st.lean = true;
        st.history = true;
        let lean = lean_draft(&x, false, &st);
        let plan = history_edit(&part, &x, false, st.history).expect("an edit of the extrude");
        let mut p = part.clone();
        made_instead(&mut p, &x, plan.features().map(<[_]>::to_vec), lean.clone());
        assert_eq!(kinds(&p), ["sketch", "extrude", "draft"]);
        assert!(matches!(&p.features[1].kind, FeatureKind::Extrude(e) if e.extent == mm_reverse::model::ExtrudeExtent::Distance { from: 0.0, to: 12.0 }));
        assert_eq!(p.features[2].name, "Draft 1");
        assert!(matches!(&p.features[2].kind, FeatureKind::Draft(d) if (d.angle_deg - 3.0).abs() < 0.005));
        assert_eq!(p.next_id, 4);
        let built = mm_reverse::build::build(&p);
        assert!(built.errors.is_empty(), "{:?}", built.errors);
        // A face nothing traces back to: the Move face is made on the body, the Draft after it.
        let alien = FaceRef { origin: 9, ..top };
        let x = MoveFaceFeature { faces: vec![alien], ..x };
        assert!(history_edit(&part, &x, false, true).is_none());
        let mut p = part.clone();
        made_instead(&mut p, &x, None, lean);
        assert_eq!(kinds(&p), ["sketch", "extrude", "moveFace", "draft"]);
        assert_eq!(p.features[3].name, "Draft 1");
        assert_eq!(p.next_id, 5);
    }

    #[test]
    fn create_waits_for_a_face_and_a_distance() {
        let mut x = MoveFaceFeature { body: BodyId::new(2), faces: Vec::new(), faces_where: Vec::new(), offset: 1.0, exprs: Exprs::new() };
        assert!(not_ready(&x).is_some_and(|w| w.contains("face")));
        x.faces.push(face(DVec3::Z));
        assert_eq!(not_ready(&x), None);
        x.offset = 0.0;
        assert!(not_ready(&x).is_some_and(|w| w.contains("how far")));
    }

    #[test]
    fn no_draft_follows_without_the_tick_a_reading_or_for_a_reopened_move() {
        let x = MoveFaceFeature { body: BodyId::new(2), faces: vec![face(DVec3::Z)], faces_where: Vec::new(), offset: 2.0, exprs: Exprs::new() };
        let lean = mm_reverse::snap_face::SnapDraft { angle_deg: 3.0, origin: DVec3::new(20.0, 10.0, 12.0), pull: DVec3::X, noise_deg: 0.01 };
        let reading = SnapReading { offset: 2.0, tilt_deg: 3.0, axis_shift: None, sigma: 0.02, draft: Some(lean) };
        let mut st = MoveBox::new(0, None);
        st.reading = Some((reading, face(DVec3::Z)));
        assert!(lean_draft(&x, false, &st).is_none());
        st.lean = true;
        let d = lean_draft(&x, false, &st).expect("a draft");
        assert_eq!(d.faces[0].centroid, DVec3::new(20.0, 10.0, 12.0));
        assert!(lean_draft(&x, true, &st).is_none());
        st.reading = None;
        assert!(lean_draft(&x, false, &st).is_none());
    }
}
