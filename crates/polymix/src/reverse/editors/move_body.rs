// SPDX-License-Identifier: AGPL-3.0-only
//! The Move body box: the bodies it moves (clicked on the part, a Shift+drag window, or chosen
//! from a list; moved, the bodies picked are out of sight and only the others take a click,
//! while a copy's stay and a click on one lets it go), whether a copy is moved instead, the six
//! numbers — Move along X, Y, Z and Turn about X, then Y, then Z through the middle of the
//! bodies as picked — and the gizmo on the part that drags them: an arrow along each axis and
//! a gimbal ring about each, held at their size on the screen, with a number field on the grip
//! once dragged. The ghost is the bodies' own meshes carried by the move at once, nothing built
//! until the hand lets go. Ported from ScanRuler (`plugins/reverse/ui/FeatureEditor.tsx` the
//! move-body fields, `featureFields.tsx` XyzField, `core/preview.ts` moveGrips,
//! `core/move.ts`, `state/reFeatureSlice.ts` and `state/reDrafts.ts` the Move's bodies and
//! pivot, `state/reGripSlice.ts` the gizmo's drag and field, `src/viewer/extendGrips.ts` the
//! grip drag), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.

use std::cell::RefCell;
use std::sync::Arc;

use egui::{Color32, Pos2, Stroke};
use glam::{DMat4, DVec3, DVec4};
use mm_core::Mesh;
use mm_reverse::body_move::{pivot_of, ring_axes};
use mm_reverse::build::{apply_matrix, move_matrix};
use mm_reverse::expression::{evaluate_for, is_plain_number, ParameterKind};
use mm_reverse::js_fmt::{num, round_to};
use mm_reverse::model::{island_id, BodyId, FeatureKind, MoveBodyFeature};

use super::body_name;
use super::loft_kit::{info_dot, parameter_table};
use crate::reverse::{Mode, PickWhat, ReverseWorkspace};
use crate::tools::Viewport;

pub(crate) const READY: bool = true;
pub(crate) const BUTTON: (&str, &str) = (
    "Move body",
    "Move and turn bodies — the ones selected on the part or picked in the body list (Ctrl+click or Shift+click rows for several), or click them on the part: drag the gizmo's arrows and rings, or type the distances and angles. Tick Copy to move a copy and leave the body where it is",
);
pub(crate) const FIRST_PICK: PickWhat = PickWhat::Body;

/// The gizmo's rings on the screen, in pixels, at any zoom — as a CAD manipulator is held.
const GIZMO_PX: f64 = 90.0;
/// Its arrows: a third of the rings long.
const ARROW_PX: f64 = GIZMO_PX / 3.0;
/// A slim cone's base, as a share of its length; a ring's tube, as a share of its radius.
const CONE_RADIUS: f64 = 0.22;
const RING_TUBE: f64 = 0.045;
/// How far the number field stands off its arrow's tip, in pixels.
const FIELD_GAP_PX: f32 = 6.0;
/// How near a grip the pointer takes it, in pixels.
const ARROW_REACH: f32 = 8.0;
const RING_REACH: f32 = 6.0;
/// A Shift+click that moves no further than this is a click, not a window, in pixels.
const WINDOW_CLICK_PX: f32 = 6.0;

/// The axes' colours, X, Y and Z: the app's own red, green and blue.
const AXIS_COLORS: [Color32; 3] = [Color32::from_rgb(0xe5, 0x53, 0x4b), Color32::from_rgb(0x2e, 0x7d, 0x46), Color32::from_rgb(0x18, 0x77, 0xc0)];
const WINDOW_GREEN: Color32 = Color32::from_rgb(0x3c, 0xb3, 0x71);
const CROSSING_BLUE: Color32 = Color32::from_rgb(0x4d, 0xb8, 0xff);

/// A grip of the gizmo: the arrow along an axis (a distance), or the ring about it (an angle).
#[derive(Debug, Clone, Copy, PartialEq)]
enum Grip {
    Move(usize),
    Turn(usize),
}

impl Grip {
    fn axis(self) -> usize {
        match self {
            Grip::Move(i) | Grip::Turn(i) => i,
        }
    }

    /// The slot path its number is kept under.
    fn slot(self) -> String {
        let xyz = ["x", "y", "z"][self.axis().min(2)];
        match self {
            Grip::Move(_) => format!("translate.{xyz}"),
            Grip::Turn(_) => format!("rotate.{xyz}"),
        }
    }

    fn value(self, x: &MoveBodyFeature) -> f64 {
        match self {
            Grip::Move(i) => x.translate[i],
            Grip::Turn(i) => x.rotate[i],
        }
    }

    fn set(self, x: &mut MoveBodyFeature, v: f64) {
        match self {
            Grip::Move(i) => x.translate[i] = v,
            Grip::Turn(i) => x.rotate[i] = v,
        }
        // Dragging a grip, or typing on it, lets a parameter go.
        x.exprs.remove(&self.slot());
    }

    fn unit(self) -> &'static str {
        match self {
            Grip::Move(_) => "mm",
            Grip::Turn(_) => "°",
        }
    }

    /// What a number typed on it is: a distance on an arrow, an angle on a ring.
    fn kind(self) -> ParameterKind {
        match self {
            Grip::Move(_) => ParameterKind::Length,
            Grip::Turn(_) => ParameterKind::Angle,
        }
    }

    /// The grip a script names it by, as ScanRuler's sides are named: `moveX` … `turnZ`.
    fn named(side: &str) -> Option<Grip> {
        let axis = |c: &str| match c {
            "X" => Some(0),
            "Y" => Some(1),
            "Z" => Some(2),
            _ => None,
        };
        match side.split_at_checked(side.len().saturating_sub(1)) {
            Some(("move", c)) => axis(c).map(Grip::Move),
            Some(("turn", c)) => axis(c).map(Grip::Turn),
            _ => None,
        }
    }
}

/// A grip in hand: where its line or ring stood when the drag began, where the pointer was on
/// it, and the number it began from.
#[derive(Debug, Clone, Copy)]
struct Drag {
    grip: Grip,
    origin: DVec3,
    dir: DVec3,
    start: f64,
    /// A ring's angle basis in its plane, the last reading and the turn so far.
    b1: DVec3,
    b2: DVec3,
    last: f64,
    turned: f64,
    from: f64,
}

/// The number field on a grip: the text shown, and whether it takes typing yet (not while the
/// grip is dragged).
#[derive(Debug, Clone)]
struct Field {
    grip: Grip,
    text: String,
    typing: bool,
}

/// What the gizmo keeps between frames.
#[derive(Debug, Clone, Default)]
struct Gizmo {
    /// The box it belongs to (the feature its preview builds).
    owner: Option<u32>,
    hover: Option<Grip>,
    drag: Option<Drag>,
    field: Option<Field>,
    /// A Shift+drag window, from where it began to where the pointer is.
    window: Option<(Pos2, Pos2)>,
    /// The pass the box was last drawn in: a box drawn again after a pass without it was shut
    /// and opened anew, and starts with no field.
    seen: u64,
    /// The viewport the gizmo was last drawn in, and the size of the field last shown.
    viewport: Option<egui::Rect>,
    field_size: Option<egui::Vec2>,
}

thread_local! {
    static GIZMO: RefCell<Gizmo> = RefCell::new(Gizmo::default());
}

fn gizmo() -> Gizmo {
    GIZMO.with(|g| g.borrow().clone())
}

fn keep(g: Gizmo) {
    GIZMO.with(|c| *c.borrow_mut() = g);
}

/// The gizmo of the box open now: let go of what it held for another box.
fn gizmo_for(ws: &ReverseWorkspace) -> Gizmo {
    let g = gizmo();
    if g.owner == ws.preview_feature {
        g
    } else {
        Gizmo { owner: ws.preview_feature, ..Gizmo::default() }
    }
}

// ---- the bodies and the pivot ----

/// The box round a body as it stood before the move, in the part's coordinates.
fn body_box(ws: &ReverseWorkspace, id: BodyId) -> Option<(DVec3, DVec3)> {
    let v = ws.pick_view(id)?;
    let mut lo = DVec3::splat(f64::INFINITY);
    let mut hi = DVec3::splat(f64::NEG_INFINITY);
    for p in &v.part_mesh.positions {
        lo = lo.min(*p);
        hi = hi.max(*p);
    }
    lo.x.is_finite().then_some((lo, hi))
}

/// The middle of the bodies a Move takes, by their boxes — the origin while none is picked.
fn move_pivot(ws: &ReverseWorkspace, ids: &[BodyId]) -> DVec3 {
    let boxes: Vec<(DVec3, DVec3)> = ids.iter().filter_map(|id| body_box(ws, *id)).collect();
    pivot_of(&boxes)
}

/// A Move's bodies changed: the pivot follows them to their middle, and stays where it was
/// with none picked.
fn with_bodies(ws: &ReverseWorkspace, x: &mut MoveBodyFeature, bodies: Vec<BodyId>) {
    let mut known: Vec<BodyId> = Vec::new();
    for b in bodies {
        if !known.contains(&b) && ws.pick_view(b).is_some() {
            known.push(b);
        }
    }
    if !known.is_empty() {
        x.pivot = move_pivot(ws, &known);
    }
    x.bodies = known;
}

/// A Move opened on the bodies given (those selected, where the list or the part has a
/// selection), about their middle: none moved, nothing turned, no copy.
pub(crate) fn new_feature_on(ws: &ReverseWorkspace, bodies: &[BodyId]) -> FeatureKind {
    let mut x = MoveBodyFeature { bodies: Vec::new(), pivot: DVec3::ZERO, translate: DVec3::ZERO, rotate: DVec3::ZERO, copy: false, exprs: Default::default() };
    with_bodies(ws, &mut x, bodies.to_vec());
    FeatureKind::MoveBody(x)
}

/// The panel's button: the one body there is, or none — a Move is opened on what is to move,
/// not on a guess.
pub(crate) fn new_feature(ws: &ReverseWorkspace) -> Result<FeatureKind, String> {
    if ws.bodies.is_empty() {
        return Err("A move takes bodies: make a body first.".into());
    }
    let one: Vec<BodyId> = if ws.bodies.len() == 1 { vec![ws.bodies[0].id] } else { Vec::new() };
    Ok(new_feature_on(ws, &one))
}

/// What the box's Create says: the bodies moved, or copied.
pub(crate) fn create_label(x: &MoveBodyFeature) -> String {
    format!("{} the {}", if x.copy { "Copy" } else { "Move" }, if x.bodies.len() > 1 { "bodies" } else { "body" })
}

// ---- the box ----

/// Three numbers in a row, each underlined in its axis's colour, to `places` decimals.
fn xyz_row(ui: &mut egui::Ui, label: &str, unit: &str, hint: &str, value: &mut DVec3, speed: f64, places: usize) -> Option<usize> {
    let mut changed = None;
    ui.horizontal(|ui| {
        ui.label(format!("{label}, {unit}"));
        info_dot(ui, label, &[hint]);
    });
    ui.horizontal(|ui| {
        for i in 0..3 {
            let r = ui.add(egui::DragValue::new(&mut value[i]).speed(speed).max_decimals(places).min_decimals(0)).on_hover_text(["X", "Y", "Z"][i]);
            ui.painter().hline(r.rect.x_range(), r.rect.bottom() + 1.0, Stroke::new(2.0, AXIS_COLORS[i]));
            if r.changed() {
                changed = Some(i);
            }
        }
    });
    changed
}

pub(crate) fn ui(ws: &mut ReverseWorkspace, ui: &mut egui::Ui, x: &mut MoveBodyFeature, pick: &mut PickWhat) -> bool {
    *pick = PickWhat::Body;
    // A box shut and opened again starts with no field on a grip (ScanRuler's box closing
    // takes its grip field away).
    let now = ui.ctx().cumulative_pass_nr();
    let mut g = gizmo_for(ws);
    if g.seen + 2 < now {
        g = Gizmo { owner: ws.preview_feature, ..Gizmo::default() };
    }
    g.seen = now;
    keep(g);
    let mut changed = false;
    let picked: Vec<String> = x.bodies.iter().map(|id| body_name(ws, *id)).collect();
    ui.horizontal_wrapped(|ui| {
        ui.label(if x.bodies.len() == 1 { "Body" } else { "Bodies" });
        if x.bodies.is_empty() {
            ui.weak("Click the bodies to move on the part, or their rows in the body list");
        } else {
            ui.strong(picked.join(", "));
            if ui.small_button("Clear").on_hover_text("Let every picked one go and pick again").clicked() {
                with_bodies(ws, x, Vec::new());
                changed = true;
            }
        }
    });
    let others: Vec<(BodyId, String)> = ws.pick_set().iter().filter(|b| !x.bodies.contains(&b.id)).map(|b| (b.id, b.name.clone())).collect();
    if !others.is_empty() {
        ui.horizontal(|ui| {
            ui.label("Add");
            let placeholder = if x.bodies.is_empty() { "Or choose a body…" } else { "Another body…" };
            let mut taken = None;
            egui::ComboBox::from_id_salt("re_move_add").selected_text(egui::RichText::new(placeholder).weak()).show_ui(ui, |ui| {
                for (id, name) in &others {
                    if ui.selectable_label(false, name).clicked() {
                        taken = Some(*id);
                    }
                }
            });
            if let Some(id) = taken {
                let mut next = x.bodies.clone();
                next.push(id);
                with_bodies(ws, x, next);
                changed = true;
            }
        });
    }
    changed |= ui.checkbox(&mut x.copy, "Copy — move a copy, the body stays where it is").on_hover_text("Leave the bodies where they are and move copies of them — bodies of their own").changed();
    if let Some(i) = xyz_row(ui, "Move", "mm", "How far along the part's X, Y and Z, after the turn — the arrows on the part drag them", &mut x.translate, 0.1, 3) {
        x.exprs.remove(&Grip::Move(i).slot());
        changed = true;
    }
    if let Some(i) = xyz_row(
        ui,
        "Turn",
        "°",
        "Degrees about X, then about Y, then about Z, each through the middle of the bodies as they were picked — the rings on the part drag them; each ring stands about the axis its angle turns about",
        &mut x.rotate,
        0.5,
        2,
    ) {
        x.exprs.remove(&Grip::Turn(i).slot());
        changed = true;
    }
    ui.horizontal(|ui| {
        let zero = x.translate == DVec3::ZERO && x.rotate == DVec3::ZERO;
        if ui.add_enabled(!zero, egui::Button::new("Reset")).on_hover_text("Put the bodies back where they stood: every distance and angle zero").clicked() {
            x.translate = DVec3::ZERO;
            x.rotate = DVec3::ZERO;
            x.exprs = Default::default();
            changed = true;
        }
    });
    let p = |v: f64| num(round_to(v, 2));
    ui.weak(format!("Turned about {}, {}, {} — the middle of the {} as picked.", p(x.pivot.x), p(x.pivot.y), p(x.pivot.z), if x.bodies.len() == 1 { "body" } else { "bodies" }));
    changed |= grip_field(ws, ui.ctx(), x);
    changed
}

/// The number field on the grip last dragged: read-only while the drag is on, following the
/// number; once let go it takes typing — Enter commits a number or an expression of the
/// parameters, Escape closes it. Shows two decimals and the unit, or the expression that drives
/// the number. Enter and Escape typed there are its own: they neither make the Move nor put
/// the box away.
fn grip_field(ws: &mut ReverseWorkspace, ctx: &egui::Context, x: &mut MoveBodyFeature) -> bool {
    let mut g = gizmo_for(ws);
    let Some(mut f) = g.field.clone() else {
        keep(g);
        return false;
    };
    let vr = ctx.data(|d| d.get_temp::<egui::Rect>(egui::Id::NULL.with("re_vp_rect")));
    let (Some(rect), false) = (vr, x.bodies.is_empty()) else {
        keep(g);
        return false;
    };
    let vp = Viewport { rect };
    let Some(shown) = shown(ws, vp, x) else {
        keep(g);
        return false;
    };
    let size = g.field_size.unwrap_or(egui::vec2(96.0, 26.0));
    let corner = match f.grip {
        Grip::Turn(_) => shown.centre - size / 2.0,
        Grip::Move(i) => {
            let (base, tip) = shown.arrow(i);
            field_corner(tip, tip - base, size)
        }
    };
    let slot = f.grip.slot();
    let value = num(round_to(f.grip.value(x), 2));
    let expr = x.exprs.get(&slot).map(str::to_string);
    let text_id = egui::Id::NULL.with("re_move_grip_text");
    // Only a settled value is shown, not the one under a hand still typing: what the number
    // is written as where an expression drives it.
    if !ctx.memory(|m| m.has_focus(text_id)) {
        f.text = expr.clone().unwrap_or_else(|| value.clone());
    }
    let tooltip = match &expr {
        Some(e) => format!("{e} = {value} {}", f.grip.unit()),
        None => "The number this grip drags — type a value and press Enter; Escape closes the field".to_string(),
    };
    let mut close = false;
    let mut commit = None;
    let r = egui::Area::new(egui::Id::NULL.with("re_move_grip_field")).order(egui::Order::Foreground).fixed_pos(corner).show(ctx, |ui| {
        egui::Frame::popup(ui.style()).inner_margin(egui::Margin::same(3)).show(ui, |ui| {
            ui.horizontal(|ui| {
                let mut text = f.text.clone();
                let r = ui.add(egui::TextEdit::singleline(&mut text).id(text_id).desired_width(64.0).interactive(f.typing)).on_hover_text(tooltip);
                ui.label(egui::RichText::new(f.grip.unit()).italics());
                if f.typing {
                    f.text = text;
                    // The keys are taken here, before the workspace reads them: Enter would
                    // make the Move and Escape put the box away.
                    if r.has_focus() || r.lost_focus() {
                        let (enter, escape) = ui.input_mut(|i| (i.consume_key(egui::Modifiers::NONE, egui::Key::Enter), i.consume_key(egui::Modifiers::NONE, egui::Key::Escape)));
                        if escape {
                            close = true;
                        } else if enter {
                            commit = Some(f.text.trim().to_string());
                        }
                    }
                }
            });
        });
    });
    g.field_size = Some(r.response.rect.size());
    g.field = if close { None } else { Some(f.clone()) };
    keep(g);
    match commit {
        Some(text) if !text.is_empty() => type_on_grip(ws, x, f.grip, &text),
        _ => false,
    }
}

/// Where the field off an arrow's tip has its top-left corner: a few pixels past the tip the
/// way the tip points on the screen, the field's edge or corner nearest the tip there — seen
/// end on, to its right (ScanRuler's `placeField`).
fn field_corner(tip: Pos2, way: egui::Vec2, size: egui::Vec2) -> Pos2 {
    let len = way.length();
    let dir = if len < 1e-6 { egui::Vec2::ZERO } else { way / len };
    let at = tip + dir * FIELD_GAP_PX;
    // Scaled so the larger part is 1: a field off a slanting tip puts its corner there.
    let m = dir.x.abs().max(dir.y.abs());
    let (x, y) = if len < 0.2 * ARROW_PX as f32 || m < 0.2 { (1.0, 0.0) } else { (dir.x / m, dir.y / m) };
    at - egui::vec2(size.x * (0.5 - x / 2.0), size.y * (0.5 - y / 2.0))
}

/// A number or an expression typed on a grip, as Enter commits it: read as the grip's kind of
/// number over the parameters, its value set, and the expression kept under the grip's slot
/// (a plain number lets the one there go); or, where it does not read, why — on the status
/// line, the number left as it was.
fn type_on_grip(ws: &mut ReverseWorkspace, x: &mut MoveBodyFeature, grip: Grip, text: &str) -> bool {
    match typed_on(x, grip, text, &parameter_table(&ws.part)) {
        Ok(()) => true,
        Err(e) => {
            ws.status = e;
            false
        }
    }
}

/// `type_on_grip` over a table of the parameters: the number set and its expression kept, or
/// why not in the words the status line shows — `{text}: {why}` (ScanRuler's `readField`).
fn typed_on(x: &mut MoveBodyFeature, grip: Grip, text: &str, table: &mm_reverse::parameters::ParameterTable) -> Result<(), String> {
    let v = evaluate_for(text, grip.kind(), table, false).map_err(|e| format!("{text}: {}", e.message))?;
    grip.set(x, v.value);
    if !is_plain_number(text) {
        x.exprs.insert(grip.slot(), text.to_string());
    }
    Ok(())
}

// ---- the gizmo on the screen ----

/// The gizmo as it stands on the screen: its middle, the rings' axes, and its arrows and rings
/// projected.
struct Shown {
    at: DVec3,
    centre: Pos2,
    rings: [DVec3; 3],
    points: [Option<Pos2>; 6],
    ring_pts: [Vec<Pos2>; 3],
}

impl Shown {
    fn arrow(&self, i: usize) -> (Pos2, Pos2) {
        (self.points[2 * i].unwrap_or(self.centre), self.points[2 * i + 1].unwrap_or(self.centre))
    }
}

/// Some unit vector square to `v`: the cross with whichever axis `v` leans least along.
fn perpendicular_to(v: DVec3) -> DVec3 {
    let a = v.abs();
    let helper = if a.x <= a.y && a.x <= a.z {
        DVec3::X
    } else if a.y <= a.z {
        DVec3::Y
    } else {
        DVec3::Z
    };
    helper.cross(v).normalize_or_zero()
}

/// The ray under a screen point, in the part's coordinates.
fn ray(ws: &ReverseWorkspace, vp: Viewport, p: Pos2) -> (DVec3, DVec3) {
    let r = ws.view.camera.ray_from_ndc(vp.ndc(p), vp.aspect());
    (ws.to_part(r.origin), ws.to_part(r.dir).normalize_or_zero())
}

/// A grip of the gizmo as the part holds it: where it stands, which way, how big on the screen
/// (an arrow's length, a ring's radius) and how far an arrow stands off its point, in pixels,
/// and its colour (ScanRuler's `moveGrips`).
#[derive(Debug, Clone, Copy, PartialEq)]
struct GripSpec {
    grip: Grip,
    position: DVec3,
    dir: DVec3,
    size_px: f64,
    standoff_px: f64,
    color: Color32,
}

/// The gizmo of a Move where the moved bodies' middle stands: an arrow along X, Y and Z,
/// standing off the middle just outside the rings so the three read as a triad, and a ring for
/// each angle — about the axis that angle turns about, the X ring riding on the Y and Z turns
/// as a gimbal's does. `radius` is the rings', in pixels; the arrows are a third of it long.
fn move_grips(pivot: DVec3, translate: DVec3, rotate: DVec3, radius: f64) -> Vec<GripSpec> {
    let at = pivot + translate;
    let rings = ring_axes(rotate);
    let axes = [DVec3::X, DVec3::Y, DVec3::Z];
    let mut out = Vec::new();
    for i in 0..3 {
        out.push(GripSpec { grip: Grip::Move(i), position: at, dir: axes[i], size_px: radius / 3.0, standoff_px: radius * 1.08, color: AXIS_COLORS[i] });
        out.push(GripSpec { grip: Grip::Turn(i), position: at, dir: rings[i], size_px: radius, standoff_px: 0.0, color: AXIS_COLORS[i] });
    }
    out
}

/// The number a grip's drag comes to: an arrow carries the bodies in whole hundredths of a
/// millimetre, a ring turns them in whole degrees.
fn dragged(grip: Grip, from: f64, delta: f64) -> f64 {
    match grip {
        Grip::Move(_) => ((from + delta) * 100.0).round() / 100.0,
        Grip::Turn(_) => (from + delta).round(),
    }
}

/// The gizmo where the bodies are going — at the pivot carried with them — held on the screen.
fn shown(ws: &ReverseWorkspace, vp: Viewport, x: &MoveBodyFeature) -> Option<Shown> {
    let at = x.pivot + x.translate;
    let centre = ws.project(vp, at)?;
    // What a pixel is worth at the gizmo: a millimetre square to the view, measured there.
    let eye = ws.to_part(ws.view.camera.eye());
    let across = perpendicular_to((at - eye).normalize_or_zero());
    let one = ws.project(vp, at + across)?;
    let px = (one - centre).length() as f64;
    if !(px > 1e-6) {
        return None;
    }
    let wpp = 1.0 / px;
    let mut rings = [DVec3::X, DVec3::Y, DVec3::Z];
    let mut points = [None; 6];
    let mut ring_pts: [Vec<Pos2>; 3] = Default::default();
    for g in move_grips(x.pivot, x.translate, x.rotate, GIZMO_PX) {
        let i = g.grip.axis();
        match g.grip {
            Grip::Move(_) => {
                let base = g.position + g.dir * (g.standoff_px * wpp);
                points[2 * i] = ws.project(vp, base);
                points[2 * i + 1] = ws.project(vp, base + g.dir * (g.size_px * wpp));
            }
            Grip::Turn(_) => {
                rings[i] = g.dir;
                let b1 = perpendicular_to(g.dir);
                let b2 = g.dir.cross(b1);
                let r = g.size_px * wpp;
                ring_pts[i] = (0..=72)
                    .filter_map(|k| {
                        let t = std::f64::consts::TAU * k as f64 / 72.0;
                        ws.project(vp, g.position + (b1 * t.cos() + b2 * t.sin()) * r)
                    })
                    .collect();
            }
        }
    }
    Some(Shown { at, centre, rings, points, ring_pts })
}

fn seg_dist(p: Pos2, a: Pos2, b: Pos2) -> f32 {
    let ab = b - a;
    let t = if ab.length_sq() > 0.0 { ((p - a).dot(ab) / ab.length_sq()).clamp(0.0, 1.0) } else { 0.0 };
    (a + ab * t - p).length()
}

/// The grip under a screen point, the nearest within reach.
fn grip_at(s: &Shown, p: Pos2) -> Option<Grip> {
    let mut best: Option<(Grip, f32)> = None;
    let mut offer = |g: Grip, d: f32, reach: f32| {
        if d <= reach && best.is_none_or(|(_, bd)| d < bd) {
            best = Some((g, d));
        }
    };
    for i in 0..3 {
        let (base, tip) = s.arrow(i);
        offer(Grip::Move(i), seg_dist(p, base, tip), ARROW_REACH.max((ARROW_PX * CONE_RADIUS) as f32));
        let d = s.ring_pts[i].windows(2).map(|w| seg_dist(p, w[0], w[1])).fold(f32::INFINITY, f32::min);
        offer(Grip::Turn(i), d, RING_REACH);
    }
    best.map(|b| b.0)
}

/// Where the pointer is along a line, in millimetres from its origin: the point on the line
/// closest to the ray — `None` looking nearly down the line, where it would run off.
fn param_along(origin: DVec3, dir: DVec3, ray: (DVec3, DVec3)) -> Option<f64> {
    let (ro, rd) = ray;
    let b = dir.dot(rd);
    let denom = 1.0 - b * b;
    if denom.abs() < 1e-3 {
        return None;
    }
    let w = origin - ro;
    Some((b * w.dot(rd) - w.dot(dir)) / denom)
}

/// Where the pointer is round an axis, radians in the (b1, b2) basis — `None` with the ring seen
/// edge on.
fn angle_around(pivot: DVec3, axis: DVec3, b1: DVec3, b2: DVec3, ray: (DVec3, DVec3)) -> Option<f64> {
    let (ro, rd) = ray;
    let denom = rd.dot(axis);
    if denom.abs() < 0.1 {
        return None;
    }
    let t = (pivot - ro).dot(axis) / denom;
    let w = ro + rd * t - pivot;
    if w.length_squared() < 1e-12 {
        return None;
    }
    Some(w.dot(b2).atan2(w.dot(b1)))
}

/// The pointer on the part while the box is open: a grip under it takes a press and the drag
/// after (an arrow carries the bodies along its axis in whole hundredths of a millimetre, a
/// ring turns them about its own in whole degrees, nothing built until the hand lets go); a
/// Shift+drag window puts every body it touches into the Move. `None` where the pointer is
/// the picks' (a plain click on a body), else whether the feature changed.
#[allow(clippy::too_many_arguments)]
pub(crate) fn grip_input(ws: &mut ReverseWorkspace, vp: Viewport, pos: Option<Pos2>, pressed: bool, down: bool, released: bool, shift: bool, x: &mut MoveBodyFeature) -> Option<bool> {
    let mut g = gizmo_for(ws);
    let s = if x.bodies.is_empty() { None } else { shown(ws, vp, x) };
    let out = if let Some(mut d) = g.drag {
        if let (Some(p), true) = (pos, down || released) {
            let r = ray(ws, vp, p);
            match d.grip {
                Grip::Move(_) => {
                    if let Some(t) = param_along(d.origin, d.dir, r) {
                        d.grip.set(x, dragged(d.grip, d.from, t - d.start));
                    }
                }
                Grip::Turn(_) => {
                    if let Some(a) = angle_around(d.origin, d.dir, d.b1, d.b2, r) {
                        // The step since the last reading, taken the short way round.
                        let mut step = a - d.last;
                        if step > std::f64::consts::PI {
                            step -= std::f64::consts::TAU;
                        } else if step < -std::f64::consts::PI {
                            step += std::f64::consts::TAU;
                        }
                        d.last = a;
                        d.turned += step;
                        d.grip.set(x, dragged(d.grip, d.from, d.turned.to_degrees()));
                    }
                }
            }
        }
        if released || !down {
            g.drag = None;
            // The hand let go: the field on the grip takes typing now, once clicked.
            g.field = Some(Field { grip: d.grip, text: num(round_to(d.grip.value(x), 2)), typing: true });
            Some(true)
        } else {
            g.drag = Some(d);
            Some(false)
        }
    } else if let Some((from, _)) = g.window {
        let to = pos.unwrap_or(from);
        if released || !down {
            g.window = None;
            let taken = if (to - from).length() <= WINDOW_CLICK_PX { armed_hit(ws, vp, to, x).map(|h| h.0).into_iter().collect() } else { window_bodies(ws, vp, from, to) };
            let mut next = x.bodies.clone();
            let before = next.clone();
            for b in taken {
                if (to - from).length() <= WINDOW_CLICK_PX && next.contains(&b) {
                    next.retain(|n| *n != b);
                } else if !next.contains(&b) {
                    next.push(b);
                }
            }
            with_bodies(ws, x, next);
            Some(x.bodies != before)
        } else {
            g.window = Some((from, to));
            Some(false)
        }
    } else {
        g.hover = match (&s, pos) {
            (Some(s), Some(p)) => grip_at(s, p),
            _ => None,
        };
        match (pressed, pos) {
            (true, Some(p)) if shift => {
                g.window = Some((p, p));
                Some(false)
            }
            (true, Some(p)) => match (g.hover, &s) {
                (Some(grip), Some(s)) => {
                    let r = ray(ws, vp, p);
                    let from = grip.value(x);
                    let started = match grip {
                        Grip::Move(i) => {
                            let dir = [DVec3::X, DVec3::Y, DVec3::Z][i];
                            param_along(s.at, dir, r).map(|t| Drag { grip, origin: s.at, dir, start: t, b1: DVec3::ZERO, b2: DVec3::ZERO, last: 0.0, turned: 0.0, from })
                        }
                        Grip::Turn(i) => {
                            let axis = s.rings[i];
                            let b1 = perpendicular_to(axis);
                            let b2 = axis.cross(b1).normalize_or_zero();
                            angle_around(s.at, axis, b1, b2, r).map(|a| Drag { grip, origin: s.at, dir: axis, start: a, b1, b2, last: a, turned: 0.0, from })
                        }
                    };
                    if let Some(d) = started {
                        g.drag = Some(d);
                        g.field = Some(Field { grip, text: num(round_to(from, 2)), typing: false });
                    }
                    Some(false)
                }
                _ => None,
            },
            _ => g.hover.map(|_| false),
        }
    };
    keep(g);
    out
}

/// The body (and its face) a click lands on, among the bodies as they stood before the move
/// that a click takes: moved, the bodies picked are out of sight — the ghost stands in for them
/// where they are going — and only the others are there to click and add; copied, they stay,
/// and a click on one lets it go (ScanRuler arms the faces of the other bodies, or of all for
/// a copy).
fn armed_hit(ws: &mut ReverseWorkspace, vp: Viewport, p: Pos2, x: &MoveBodyFeature) -> Option<(BodyId, usize)> {
    let out_of_sight: Vec<BodyId> = if x.copy { Vec::new() } else { x.bodies.iter().copied().filter(|b| ws.hidden_bodies.insert(*b)).collect() };
    let hit = ws.pick_hit(vp, p);
    for b in out_of_sight {
        ws.hidden_bodies.remove(&b);
    }
    match hit {
        Some(super::super::view::Hit::Body { body, face, .. }) => Some((body, face)),
        _ => None,
    }
}

/// The bodies a window takes anything of: dragged rightward, a window takes what lies wholly
/// inside it — a face or an edge; leftward, a crossing takes what it touches, hidden edges
/// included. The bodies a Move already has are among them or not alike: a window only adds.
fn window_bodies(ws: &ReverseWorkspace, vp: Viewport, a: Pos2, b: Pos2) -> Vec<BodyId> {
    let rect = egui::Rect::from_two_pos(a, b);
    let crossing = b.x < a.x;
    let mut out = Vec::new();
    for v in ws.pick_set().iter().filter(|v| !ws.hidden_bodies.contains(&v.id)) {
        let screen: Vec<Option<Pos2>> = v.part_mesh.positions.iter().map(|p| ws.project(vp, *p)).collect();
        let inside = |i: u32| screen.get(i as usize).copied().flatten().is_some_and(|q| rect.contains(q));
        let taken = if crossing {
            screen.iter().flatten().any(|q| rect.contains(*q))
                || v.topology.edges.iter().any(|e| {
                    let pts: Vec<Pos2> = e.polyline.iter().filter_map(|q| ws.project(vp, *q)).collect();
                    pts.windows(2).any(|w| segment_meets_rect(w[0], w[1], rect))
                })
        } else {
            let faces = v.part_mesh.tri_face.iter().copied().max().map_or(0, |m| m as usize + 1);
            let mut whole = vec![true; faces];
            for (t, f) in v.part_mesh.triangles.iter().zip(&v.part_mesh.tri_face) {
                if let Some(w) = whole.get_mut(*f as usize) {
                    *w &= t.iter().all(|i| inside(*i));
                }
            }
            whole.iter().any(|w| *w) || v.topology.edges.iter().any(|e| !e.polyline.is_empty() && e.polyline.iter().all(|q| ws.project(vp, *q).is_some_and(|s| rect.contains(s))))
        };
        if taken {
            out.push(v.id);
        }
    }
    out
}

fn segment_meets_rect(a: Pos2, b: Pos2, r: egui::Rect) -> bool {
    if r.contains(a) || r.contains(b) {
        return true;
    }
    let corners = [r.left_top(), r.right_top(), r.right_bottom(), r.left_bottom()];
    let cross = |o: Pos2, p: Pos2, q: Pos2| (p.x - o.x) * (q.y - o.y) - (p.y - o.y) * (q.x - o.x);
    (0..4).any(|k| {
        let (c, d) = (corners[k], corners[(k + 1) % 4]);
        let (d1, d2, d3, d4) = (cross(c, d, a), cross(c, d, b), cross(a, b, c), cross(a, b, d));
        ((d1 > 0.0) != (d2 > 0.0)) && ((d3 > 0.0) != (d4 > 0.0))
    })
}

/// A click on a body puts it into the Move, or — picked already, and a copy's, where the body
/// stays in sight — lets it go; the pivot follows. The face under the pointer of a body a
/// click would take is lit.
pub(crate) fn pick(ws: &mut ReverseWorkspace, vp: Viewport, p: Pos2, clicked: bool, x: &mut MoveBodyFeature, _pick: &mut PickWhat) -> bool {
    let Some((body, face)) = armed_hit(ws, vp, p, x) else { return false };
    ws.view.hover_face = Some((body, face));
    if !clicked {
        return false;
    }
    let mut next = x.bodies.clone();
    if next.contains(&body) {
        next.retain(|b| *b != body);
    } else {
        next.push(body);
    }
    with_bodies(ws, x, next);
    true
}

/// The gizmo at the bodies' (moved) middle — an arrow along X, Y and Z and a ring about each,
/// in the axes' colours, the one under the pointer white — and a Shift+drag window.
pub(crate) fn paint(ws: &ReverseWorkspace, painter: &egui::Painter, vp: Viewport, x: &MoveBodyFeature) {
    GIZMO.with(|c| c.borrow_mut().viewport = Some(vp.rect));
    let g = gizmo();
    if let Some((a, b)) = g.window {
        let rect = egui::Rect::from_two_pos(a, b);
        if b.x < a.x {
            painter.rect_filled(rect, 0.0, CROSSING_BLUE.gamma_multiply(0.12));
            let c = [rect.left_top(), rect.right_top(), rect.right_bottom(), rect.left_bottom(), rect.left_top()];
            painter.extend(egui::Shape::dashed_line(&c, Stroke::new(1.0, CROSSING_BLUE), 4.0, 3.0));
        } else {
            painter.rect_filled(rect, 0.0, WINDOW_GREEN.gamma_multiply(0.12));
            painter.rect_stroke(rect, 0.0, Stroke::new(1.0, WINDOW_GREEN), egui::StrokeKind::Inside);
        }
    }
    if x.bodies.is_empty() {
        return;
    }
    let Some(s) = shown(ws, vp, x) else { return };
    let held = g.drag.map(|d| d.grip).or(g.hover);
    if g.drag.is_some() {
        painter.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
    } else if g.hover.is_some() {
        painter.ctx().set_cursor_icon(egui::CursorIcon::Grab);
    }
    let color = |grip: Grip, alpha: f32| if held == Some(grip) { Color32::WHITE } else { AXIS_COLORS[grip.axis()].gamma_multiply(alpha) };
    let tube = (2.0 * RING_TUBE * GIZMO_PX) as f32;
    for i in 0..3 {
        if s.ring_pts[i].len() >= 2 {
            painter.add(egui::Shape::line(s.ring_pts[i].clone(), Stroke::new(tube, color(Grip::Turn(i), 0.85))));
        }
    }
    for i in 0..3 {
        let (base, tip) = s.arrow(i);
        let c = color(Grip::Move(i), 0.95);
        let d = tip - base;
        let len = d.length();
        let half = (ARROW_PX * CONE_RADIUS) as f32;
        if len < 1.0 {
            painter.circle_filled(tip, half, c);
            continue;
        }
        let side = egui::vec2(-d.y, d.x) / len * half;
        painter.add(egui::Shape::convex_polygon(vec![tip, base + side, base - side], c, Stroke::NONE));
    }
}

// ---- script steps (tests) ----

/// The open Move's feature, handed to `f` with the workspace, and the box put back after; a
/// change built.
fn with_open_move(ws: &mut ReverseWorkspace, f: impl FnOnce(&mut ReverseWorkspace, &mut MoveBodyFeature) -> Result<(String, bool), String>) -> Result<String, String> {
    let mut mode = std::mem::replace(&mut ws.mode, Mode::Idle);
    let out = match &mut mode {
        Mode::Feature(edit) => match &mut edit.kind {
            FeatureKind::MoveBody(x) => f(ws, x),
            _ => Err("No Move box is open.".to_string()),
        },
        _ => Err("No Move box is open.".to_string()),
    };
    ws.mode = mode;
    match out {
        Ok((words, changed)) => {
            if changed {
                ws.request_build();
            }
            Ok(words)
        }
        Err(e) => Err(e),
    }
}

/// The viewport the gizmo was last drawn in.
fn last_viewport() -> Result<Viewport, String> {
    GIZMO.with(|c| c.borrow().viewport).map(|rect| Viewport { rect }).ok_or_else(|| "The Move box has not been drawn yet.".to_string())
}

/// A Move's bodies and numbers, in words.
fn move_words(ws: &ReverseWorkspace, x: &MoveBodyFeature) -> String {
    let v = |d: DVec3| format!("{}, {}, {}", num(round_to(d.x, 3)), num(round_to(d.y, 3)), num(round_to(d.z, 3)));
    let names: Vec<String> = x.bodies.iter().map(|id| body_name(ws, *id)).collect();
    let exprs: Vec<String> = x.exprs.iter().map(|(p, e)| format!("{p} = {e}")).collect();
    format!("bodies [{}], move ({}), turn ({}), pivot ({}){}{}", names.join(", "), v(x.translate), v(x.rotate), v(x.pivot), if x.copy { ", a copy" } else { "" }, if exprs.is_empty() { String::new() } else { format!(", exprs [{}]", exprs.join("; ")) })
}

/// The `move-grip:<side>:<dx>,<dy>` step: a grip of the open Move's gizmo (`moveX` … `turnZ`)
/// pressed where it is drawn, dragged by the pixels given and let go, as a hand would — the
/// press, the drag and the release through the gizmo's own pointer handling.
pub(crate) fn script_drag(ws: &mut ReverseWorkspace, side: &str, by: egui::Vec2) -> Result<String, String> {
    let grip = Grip::named(side).ok_or_else(|| format!("move-grip takes moveX, moveY, moveZ, turnX, turnY or turnZ, not {side}."))?;
    let vp = last_viewport()?;
    with_open_move(ws, |ws, x| {
        let s = shown(ws, vp, x).ok_or("The Move's gizmo is not in view — pick a body first.")?;
        // An arrow is taken at its middle, a ring where it stands widest on the screen.
        let at = match grip {
            Grip::Move(i) => {
                let (a, b) = s.arrow(i);
                a + (b - a) / 2.0
            }
            Grip::Turn(i) => s.ring_pts[i].iter().copied().max_by(|a, b| (*a - s.centre).length().total_cmp(&(*b - s.centre).length())).ok_or("That ring is not in view.")?,
        };
        grip_input(ws, vp, Some(at), false, false, false, false, x);
        grip_input(ws, vp, Some(at), true, true, false, false, x);
        if gizmo().drag.map(|d| d.grip) != Some(grip) {
            grip_input(ws, vp, Some(at), false, false, true, false, x);
            return Err(format!("{side} could not be taken where it is drawn, at ({}, {}) — another grip is over it, or it is seen end on.", at.x.round(), at.y.round()));
        }
        // In small steps, as a hand drags: a ring's turn is read the short way round each step.
        let steps = 16;
        for k in 1..=steps {
            grip_input(ws, vp, Some(at + by * (k as f32 / steps as f32)), false, true, false, false, x);
        }
        grip_input(ws, vp, Some(at + by), false, false, true, false, x);
        Ok((format!("{side} dragged: {}", move_words(ws, x)), true))
    })
}

/// The `grip-field:<text>` step: a number or an expression typed into the field on the grip
/// last dragged, and Enter.
pub(crate) fn script_type(ws: &mut ReverseWorkspace, text: &str) -> Result<String, String> {
    let field = gizmo_for(ws).field.ok_or("No field is open on a grip — drag one first (move-grip).")?;
    if !field.typing {
        return Err("The grip is still in hand.".into());
    }
    with_open_move(ws, |ws, x| {
        if type_on_grip(ws, x, field.grip, text) {
            Ok((format!("{} typed {text}: {}", field.grip.slot(), move_words(ws, x)), true))
        } else {
            Err(ws.status.clone())
        }
    })
}

/// The `move-click:<x>,<y>,<z>` step: a click on the part where the point (part coordinates)
/// is drawn, as the open Move takes it — a body added, or a copy's let go.
pub(crate) fn script_click(ws: &mut ReverseWorkspace, at: DVec3) -> Result<String, String> {
    let vp = last_viewport()?;
    let p = ws.project(vp, at).ok_or("That point is not in view.")?;
    with_open_move(ws, |ws, x| {
        let changed = pick(ws, vp, p, true, x, &mut PickWhat::Body);
        Ok((format!("{}: {}", if changed { "clicked" } else { "nothing taken" }, move_words(ws, x)), changed))
    })
}

// ---- the ghost ----

/// A body the Move's ghost shows: the body as it stood (or, for a copy, the copy's place in the
/// list), its mesh and edges as they stood, and the move carrying them, in the scene's
/// coordinates.
pub(crate) struct Carried {
    pub(crate) id: BodyId,
    pub(crate) mesh: Arc<Mesh>,
    pub(crate) edges: Vec<Vec<DVec3>>,
    pub(crate) model: glam::Mat4,
}

/// The bodies the open Move carries, drawn at once where the gizmo puts them: their own meshes
/// as they stood before the move, nothing built. None until the bodies as they stood have come
/// in (a Move reopened is built again as it stood before it first).
pub(crate) fn carried(ws: &ReverseWorkspace, x: &MoveBodyFeature) -> Vec<Carried> {
    let (Some(_), Some(step)) = (ws.pick_bodies.as_ref(), ws.preview_feature) else { return Vec::new() };
    let m = move_matrix(x.pivot, x.translate, x.rotate);
    if m.iter().flatten().any(|v| !v.is_finite()) {
        return Vec::new();
    }
    // The move in the scene the view draws: the part's axes there and back.
    let o = ws.to_scene(apply_matrix(&m, ws.to_part(DVec3::ZERO)));
    let col = |e: DVec3| (ws.to_scene(apply_matrix(&m, ws.to_part(e))) - o).extend(0.0);
    let model = DMat4::from_cols(col(DVec3::X), col(DVec3::Y), col(DVec3::Z), DVec4::new(o.x, o.y, o.z, 1.0));
    let mut out = Vec::new();
    let mut seen: Vec<BodyId> = Vec::new();
    for src in &x.bodies {
        if seen.contains(src) {
            continue;
        }
        seen.push(*src);
        let Some(v) = ws.pick_view(*src) else { continue };
        let id = if x.copy { island_id(step, (seen.len() - 1) as u32) } else { *src };
        let edges = v.edges.iter().map(|e| e.iter().map(|p| model.transform_point3(*p)).collect()).collect();
        out.push(Carried { id, mesh: v.mesh.clone(), edges, model: model.as_mat4() });
    }
    out
}

#[cfg(test)]
mod tests {
    //! The gizmo's parts of ScanRuler's `plugins/reverse/tests/reImportMove.test.ts` ("the move
    //! as numbers", "the Move box").
    use super::*;

    fn near(a: DVec3, b: [f64; 3]) -> bool {
        (a - DVec3::from(b)).length() < 1e-9
    }

    #[test]
    fn puts_the_gizmo_where_the_bodies_are_going_an_arrow_and_a_ring_an_axis_in_its_colour() {
        let grips = move_grips(DVec3::new(1.0, 2.0, 3.0), DVec3::new(10.0, 0.0, 0.0), DVec3::ZERO, 60.0);
        let sides: Vec<String> = grips.iter().map(|g| format!("{:?}", g.grip)).collect();
        assert_eq!(sides, ["Move(0)", "Turn(0)", "Move(1)", "Turn(1)", "Move(2)", "Turn(2)"]);
        assert!(near(grips[1].position, [11.0, 2.0, 3.0]));
        assert_eq!(grips[1].size_px, 60.0);
        // The arrow stands on the middle too, off it past the ring.
        assert!(near(grips[0].position, [11.0, 2.0, 3.0]));
        assert!(grips[0].standoff_px > 60.0);
        assert_eq!(grips[0].size_px, 20.0);
        let colors: std::collections::HashSet<[u8; 4]> = grips.iter().map(|g| g.color.to_array()).collect();
        assert_eq!(colors.len(), 3);
        // Each ring about the axis its own angle turns about: with nothing turned, X, Y and Z.
        assert!(near(grips[1].dir, [1.0, 0.0, 0.0]) && near(grips[3].dir, [0.0, 1.0, 0.0]) && near(grips[5].dir, [0.0, 0.0, 1.0]));
        let turned = move_grips(DVec3::ZERO, DVec3::ZERO, DVec3::new(0.0, 0.0, 90.0), 90.0);
        assert!(near(turned[1].dir, [0.0, 1.0, 0.0]), "the X ring rides on the Z turn: {:?}", turned[1].dir);
        assert!(near(turned[0].dir, [1.0, 0.0, 0.0]), "the arrows stay on the part's axes");
    }

    #[test]
    fn takes_the_gizmos_drags_in_hundredths_and_whole_degrees_and_typed_numbers_as_typed() {
        assert_eq!(dragged(Grip::Move(1), 0.0, 12.345), 12.35);
        assert_eq!(dragged(Grip::Turn(2), 0.0, 44.6), 45.0);
        let mut x = MoveBodyFeature { bodies: vec![BodyId::new(2)], pivot: DVec3::splat(5.0), translate: DVec3::ZERO, rotate: DVec3::ZERO, copy: false, exprs: Default::default() };
        x.exprs.insert("translate.y", "a + 1");
        Grip::Move(1).set(&mut x, dragged(Grip::Move(1), 0.0, 12.345));
        Grip::Turn(2).set(&mut x, dragged(Grip::Turn(2), 0.0, 44.6));
        let none = mm_reverse::parameters::ParameterTable::default();
        assert_eq!(typed_on(&mut x, Grip::Move(0), "-3", &none), Ok(()));
        assert_eq!(x.translate, DVec3::new(-3.0, 12.35, 0.0));
        assert_eq!(x.rotate, DVec3::new(0.0, 0.0, 45.0));
        // A drag lets the parameter behind the number go. A signed number is no plain number to
        // ScanRuler's `isPlainNumber` (a sign and a number are two tokens): kept as typed.
        assert_eq!((x.exprs.get("translate.y"), x.exprs.get("translate.x")), (None, Some("-3")));
        assert_eq!(create_label(&x), "Move the body");
        x.copy = true;
        x.bodies.push(BodyId::new(3));
        assert_eq!(create_label(&x), "Copy the bodies");
    }

    #[test]
    fn reads_a_typed_expression_over_the_parameters_and_keeps_it_under_the_grips_slot() {
        // ScanRuler's grip field: a number or an expression of the parameters, the expression
        // kept beside the number; what does not read is said with the text, the number kept.
        let mut part = mm_reverse::model::Part::default();
        part.parameters.push(mm_reverse::model::Parameter { id: 1, name: "rise".into(), kind: mm_reverse::model::ParameterKind::Length, expression: "12.5".into(), comment: None });
        part.parameters.push(mm_reverse::model::Parameter { id: 2, name: "tilt".into(), kind: mm_reverse::model::ParameterKind::Angle, expression: "30".into(), comment: None });
        let table = parameter_table(&part);
        let mut x = MoveBodyFeature { bodies: vec![BodyId::new(2)], pivot: DVec3::ZERO, translate: DVec3::ZERO, rotate: DVec3::ZERO, copy: false, exprs: Default::default() };
        assert_eq!(typed_on(&mut x, Grip::Move(2), "rise * 2", &table), Ok(()));
        assert_eq!(x.translate.z, 25.0);
        assert_eq!(x.exprs.get("translate.z"), Some("rise * 2"));
        assert_eq!(typed_on(&mut x, Grip::Turn(0), "tilt + 15", &table), Ok(()));
        assert_eq!((x.rotate.x, x.exprs.get("rotate.x")), (45.0, Some("tilt + 15")));
        // A length is no angle, and a name not a parameter's reads as nothing.
        let said = typed_on(&mut x, Grip::Turn(0), "rise", &table).err().unwrap_or_default();
        assert!(said.starts_with("rise: comes out as"), "{said}");
        assert!(typed_on(&mut x, Grip::Move(0), "one", &table).is_err_and(|e| e.starts_with("one: ")));
        assert_eq!((x.rotate.x, x.translate.x), (45.0, 0.0));
        // A plain number lets the expression there go.
        assert_eq!(typed_on(&mut x, Grip::Move(2), "4", &table), Ok(()));
        assert_eq!((x.translate.z, x.exprs.get("translate.z")), (4.0, None));
        assert_eq!(Grip::named("turnY"), Some(Grip::Turn(1)));
        assert_eq!(Grip::named("moveZ"), Some(Grip::Move(2)));
        assert_eq!(Grip::named("spinX"), None);
    }

    #[test]
    fn stands_the_field_off_an_arrows_tip_the_way_it_points_and_to_the_right_seen_end_on() {
        let size = egui::vec2(100.0, 20.0);
        let tip = Pos2::new(200.0, 200.0);
        // Pointing right: the field's left edge 6 px past the tip, centred on it.
        assert_eq!(field_corner(tip, egui::vec2(30.0, 0.0), size), Pos2::new(206.0, 190.0));
        // Pointing up the screen: the field above, centred across.
        assert_eq!(field_corner(tip, egui::vec2(0.0, -30.0), size), Pos2::new(150.0, 174.0));
        // Down and to the left at 45°: its top-right corner there.
        let c = field_corner(tip, egui::vec2(-30.0, 30.0), size);
        let at = tip + egui::vec2(-1.0, 1.0).normalized() * FIELD_GAP_PX;
        assert!((c - (at - egui::vec2(100.0, 0.0))).length() < 1e-3, "{c:?}");
        // Seen end on, the arrow points nowhere: the field stands to its right.
        assert_eq!(field_corner(tip, egui::vec2(1.0, 2.0), size), tip + egui::vec2(1.0, 2.0).normalized() * FIELD_GAP_PX - egui::vec2(0.0, 10.0));
    }

    #[test]
    fn reads_the_pointer_along_an_arrow_and_round_a_ring() {
        // A ray straight down at x = 7 meets the X axis at 7.
        let down = (DVec3::new(7.0, 0.0, 10.0), DVec3::new(0.0, 0.0, -1.0));
        assert!((param_along(DVec3::ZERO, DVec3::X, down).unwrap_or(f64::NAN) - 7.0).abs() < 1e-12);
        // Looking down the arrow there is no reading.
        assert_eq!(param_along(DVec3::ZERO, DVec3::Z, down), None);
        // Round the Z ring, from above: a quarter turn is a quarter turn, right-handed.
        let b1 = perpendicular_to(DVec3::Z);
        let b2 = DVec3::Z.cross(b1);
        let at = |p: DVec3| angle_around(DVec3::ZERO, DVec3::Z, b1, b2, (p + DVec3::Z * 10.0, -DVec3::Z)).unwrap_or(f64::NAN);
        let a0 = at(b1 * 5.0);
        let a1 = at(b2 * 5.0);
        assert!((a1 - a0 - std::f64::consts::FRAC_PI_2).abs() < 1e-9);
        // Seen edge on, none.
        assert_eq!(angle_around(DVec3::ZERO, DVec3::Z, b1, b2, (DVec3::new(0.0, -10.0, 0.0), DVec3::Y)), None);
        assert!(segment_meets_rect(Pos2::new(-5.0, 5.0), Pos2::new(15.0, 5.0), egui::Rect::from_min_max(Pos2::ZERO, Pos2::new(10.0, 10.0))));
        assert!(!segment_meets_rect(Pos2::new(-5.0, 15.0), Pos2::new(15.0, 15.0), egui::Rect::from_min_max(Pos2::ZERO, Pos2::new(10.0, 10.0))));
    }
}
