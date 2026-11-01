// SPDX-License-Identifier: AGPL-3.0-only
//! What the Extrude and the Revolve boxes share: the ⓘ cards, a number field with its round-value
//! chip, the Profiles line, the preview line, the sketch's regions drawn on the part as faces to
//! click, the plane or axis a list's option stands for drawn on the part, and the grips that drag
//! the numbers — arrows held at a size on the screen, a ring about the revolve's axis, and the
//! number field a grip shows once it has been dragged. Ported from ScanRuler
//! (`plugins/reverse/ui/featureFields.tsx`, `src/ui/InfoDot.tsx`, `src/ui/RefPicker.tsx`,
//! `viewer/sketchOverlay.ts`, `viewer/reScene.ts` the hints and grips, `state/reGripSlice.ts`), ©
//! Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! Only one box is open at a time, and what it remembers between frames that the feature does not
//! hold — the card open, the hint shown, a grip held, the field on it — is kept here, for the box
//! open now, and let go when another opens.

use std::cell::RefCell;

use egui::{Color32, Pos2, Stroke, Vec2};
use glam::{DVec2, DVec3};
use mm_reverse::display_units::{fmt_vol, DisplayUnits, FmtOptions};
use mm_reverse::extrude_grips::{FeatureGrip, GripShape, GripSide};
use mm_reverse::model::{ExtrudeExtent, FeatureId};
use mm_reverse::nominal::{nominal_of, NominalKind};

use crate::reverse::sketcher::fmt_value;
use crate::reverse::ReverseWorkspace;
use crate::tools::Viewport;

/// The sketch's blue, on the part.
pub(crate) const SKETCH_BLUE: Color32 = Color32::from_rgb(0x3d, 0x8f, 0xd6);
/// A picked region's green.
pub(crate) const PICK_GREEN: Color32 = Color32::from_rgb(0x3c, 0xb3, 0x71);
/// A hint's amber: the plane or axis an option stands for.
pub(crate) const AMBER: Color32 = Color32::from_rgb(0xff, 0xb0, 0x20);
/// A cut's red.
pub(crate) const CUT_RED: Color32 = Color32::from_rgb(0xe0, 0x56, 0x4f);
/// The ghost's colour, as the view draws the bodies a box is changing.
pub(crate) const GHOST: Color32 = Color32::from_rgb(115, 204, 140);
/// A warning line's colour.
pub(crate) const WARN: Color32 = Color32::from_rgb(240, 160, 90);

/// The plane or the axis a list's option stands for, drawn on the part.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Hint {
    Plane { origin: DVec3, normal: DVec3, name: String },
    Axis { origin: DVec3, dir: DVec3, name: String },
}

/// A grip being dragged: which, where the drag began on it, and the numbers it began with.
#[derive(Debug, Clone)]
pub(crate) struct GripDrag {
    pub side: GripSide,
    pub grip: FeatureGrip,
    /// Millimetres along an arrow, or radians about a ring, where the pointer took hold.
    pub t0: f64,
    /// A ring's last angle, and the turn wound so far (radians), round and round.
    pub last: f64,
    pub wound: f64,
    pub extent: Option<ExtrudeExtent>,
    pub taper_deg: f64,
    pub taper_run: f64,
    pub angle_deg: f64,
}

/// The number field a grip shows: read-only while it is dragged, taking typing once let go.
#[derive(Debug, Clone)]
pub(crate) struct GripField {
    pub side: GripSide,
    pub text: String,
    pub typing: bool,
    pub error: Option<String>,
}

/// What the open box keeps between frames.
#[derive(Debug, Clone, Default)]
pub(crate) struct BoxState {
    /// Which box this is: the feature it previews and its kind.
    pub key: Option<(Option<FeatureId>, &'static str)>,
    /// A face pulled follows the way it is pulled — out joins, in cuts — until a result is
    /// chosen by hand.
    pub op_auto: bool,
    /// The ⓘ card open.
    pub card: Option<egui::Id>,
    /// The plane or axis an option under the pointer stands for, and the one chosen.
    pub hover_hint: Option<Hint>,
    pub chosen_hints: Vec<Hint>,
    /// What the To field last took the keys for: the box and the profiles picked then.
    pub focused_for: Option<(Option<FeatureId>, usize)>,
    /// A field asked for the feature to be made (Enter in To).
    pub create_now: bool,
    /// The Direction list stands at "Typed…".
    pub typing_direction: bool,
    pub hover_grip: Option<GripSide>,
    pub drag: Option<GripDrag>,
    pub field: Option<GripField>,
    /// Where each grip's field stands on the screen: the arrow's tip and the way it points.
    pub anchors: Vec<(GripSide, Pos2, Vec2)>,
    /// A drag changed the feature since the last build was asked for.
    pub owed: bool,
    /// The sketch line under the pointer, for a revolve's axis: its sketch and its id.
    pub hover_line: Option<(FeatureId, u32)>,
}

thread_local! {
    static BOX: RefCell<BoxState> = RefCell::new(BoxState::default());
}

/// The open box's state, to read or change.
pub(crate) fn with_box<R>(f: impl FnOnce(&mut BoxState) -> R) -> R {
    BOX.with(|b| f(&mut b.borrow_mut()))
}

/// Start the state afresh for the box `key` when another was open before it.
pub(crate) fn box_for(ws: &ReverseWorkspace, kind: &'static str) {
    let key = (ws.preview_feature, kind);
    with_box(|b| {
        if b.key != Some(key) {
            *b = BoxState { key: Some(key), ..BoxState::default() };
        }
    });
}

/// Let go of everything a box kept: a new one opens.
pub(crate) fn reset_box() {
    with_box(|b| *b = BoxState::default());
}

// ---- the panel ----

/// A small round "i" after a label: a click opens a card — a heading and paragraphs — outside
/// the panel, level with the dot; a second click, a click anywhere else, or Escape closes it.
pub(crate) fn info_dot(ui: &mut egui::Ui, ws: &ReverseWorkspace, salt: &str, title: &str, paragraphs: &[&str]) {
    let id = ui.scope_id().with(("re_info", salt));
    let r = ui.add(egui::Button::new(egui::RichText::new("ⓘ").small()).frame(false)).on_hover_text(title);
    let open = with_box(|b| b.card == Some(id));
    if r.clicked() {
        with_box(|b| b.card = if open { None } else { Some(id) });
        return;
    }
    if !open {
        return;
    }
    let ctx = ui.ctx().clone();
    let x = ws.view.panel_rect.map_or(r.rect.right() + 12.0, |p| p.right() + 10.0);
    let area = egui::Area::new(id.with("card")).fixed_pos(egui::pos2(x, r.rect.top() - 6.0)).order(egui::Order::Foreground).show(&ctx, |ui| {
        egui::Frame::popup(ui.style()).show(ui, |ui| {
            ui.set_max_width(320.0);
            ui.label(egui::RichText::new(title).strong());
            for p in paragraphs {
                ui.label(*p);
            }
        });
    });
    let (pressed, over) = ctx.input(|i| (i.pointer.any_pressed(), i.pointer.interact_pos()));
    let inside = over.is_some_and(|p| area.response.rect.contains(p) || r.rect.contains(p));
    let esc = ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Escape));
    if (pressed && !inside) || esc {
        with_box(|b| b.card = None);
    }
}

/// The unit a number field holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Unit {
    Mm,
    Deg,
    Turns,
}

impl Unit {
    fn suffix(self) -> &'static str {
        match self {
            Unit::Mm => " mm",
            Unit::Deg => "°",
            Unit::Turns => " turns",
        }
    }
}

/// A labelled number: the label with its ⓘ (the hint's words), the round value a designer would
/// have drawn as a chip to take with one click — not where an expression drives it — and the
/// field. Returns the field's response and whether the number changed.
#[allow(clippy::too_many_arguments)]
pub(crate) fn num(ui: &mut egui::Ui, ws: &ReverseWorkspace, label: &str, value: &mut f64, unit: Unit, nominal: Option<NominalKind>, hint: Option<&str>, expr: Option<&str>, range: Option<std::ops::RangeInclusive<f64>>) -> (egui::Response, bool) {
    let mut changed = false;
    // Under an id of its own: a field's id (and the keyboard focus on it) stays the field's when
    // the lines above it come and go — a profile picked takes a hint away, and the focus on To
    // would otherwise land on From.
    let inner = ui.push_id(("re_num", label), |ui| ui.horizontal(|ui| {
        ui.label(label);
        if let Some(h) = hint {
            info_dot(ui, ws, label, label, &[h]);
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let speed = if unit == Unit::Mm { 0.1 } else { 0.5 };
            let mut dv = egui::DragValue::new(value).speed(speed).suffix(unit.suffix()).max_decimals(4);
            if let Some(r) = range.clone() {
                dv = dv.range(r);
            }
            let resp = ui.add(dv);
            changed |= resp.changed();
            let resp = match expr {
                Some(e) => resp.on_hover_text(format!("{e} = {} {}", fmt_value(*value), unit.suffix().trim())),
                None => resp,
            };
            if expr.is_none() {
                if let Some(n) = nominal.and_then(|k| nominal_of(*value, k)) {
                    let chip = egui::Button::new(egui::RichText::new(format!("→ {}", nominal_text(n, k_of(nominal)))).monospace().small().color(SKETCH_BLUE)).corner_radius(10.0).stroke(Stroke::new(1.0, SKETCH_BLUE)).fill(Color32::TRANSPARENT);
                    if ui.add(chip).on_hover_text("The round number this was probably meant as — click to take it").clicked() {
                        *value = n;
                        changed = true;
                    }
                }
            }
            resp
        })
        .inner
    }).inner);
    (inner.inner, changed)
}

fn k_of(nominal: Option<NominalKind>) -> NominalKind {
    nominal.unwrap_or(NominalKind::Length)
}

/// A round value as its chip writes it (ScanRuler's `nominalText`): an angle in whole degrees,
/// a draft to a tenth where it has one, a length in millimetres to a tenth where it has one.
pub(crate) fn nominal_text(value: f64, kind: NominalKind) -> String {
    let to_fixed = mm_reverse::js_number::to_fixed;
    match kind {
        NominalKind::Angle => format!("{}°", to_fixed(value, 0)),
        NominalKind::Draft => format!("{}°", to_fixed(value, if value % 1.0 == 0.0 { 0 } else { 1 })),
        NominalKind::Length | NominalKind::Radius => mm_reverse::display_units::fmt_len(value, DisplayUnits::Mm, if value % 1.0 == 0.0 { 0 } else { 1 }, FmtOptions::default()),
    }
}

/// A hint paragraph: small and dim, amber where it warns.
pub(crate) fn hint(ui: &mut egui::Ui, text: &str) {
    ui.add(egui::Label::new(egui::RichText::new(text).weak()).wrap());
}

/// The Profiles line: how many of the sketch's regions are picked, a key that takes them all or
/// lets them all go, its card, and what to do while none is.
pub(crate) fn profiles_field(ui: &mut egui::Ui, ws: &ReverseWorkspace, sketch: FeatureId, regions: &mut Vec<String>, region_pick: &mut Option<mm_reverse::model::RegionPick>) -> bool {
    let found: Vec<String> = ws.built.sketches.get(&sketch).map(|s| s.regions.iter().map(|r| r.key.clone()).collect()).unwrap_or_default();
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label("Profiles");
        info_dot(
            ui,
            ws,
            "profiles",
            "Profiles",
            &["The sketch's closed regions are drawn on the part while this box is open: click one to use it, click it again to let it go. An inner loop left unpicked becomes a hole of the region around it. A sketch with one region has it picked already."],
        );
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let all = !found.is_empty() && found.iter().all(|k| regions.contains(k));
            let (word, tip) = if all { ("None", "Let every region go") } else { ("All", "Use every closed region of the sketch") };
            if ui.add_enabled(!found.is_empty(), egui::Button::new(word).small()).on_hover_text(tip).clicked() {
                *regions = if all { Vec::new() } else { found.clone() };
                *region_pick = None;
                changed = true;
            }
            ui.label(egui::RichText::new(format!("{} of {} picked", regions.len(), found.len())).strong());
        });
    });
    if regions.is_empty() {
        hint(ui, if found.is_empty() { "The sketch has no closed region — edit it and close the profile." } else { "Click the sketch's regions in the viewport." });
    }
    changed
}

/// The preview line: one sentence about the ghost — what to do while there is none, a spinner
/// while it builds, its volume once built, or why it failed.
pub(crate) fn preview_line(ui: &mut egui::Ui, ws: &ReverseWorkspace, ready: bool) {
    let id = ws.preview_feature;
    if !ready {
        hint(ui, "Pick a profile to see the body.");
        return;
    }
    if ws.building() {
        ui.horizontal(|ui| {
            polymix_workspace::spinner(ui);
            hint(ui, "Building the preview…");
        });
        return;
    }
    if let Some(e) = id.and_then(|id| ws.built.errors.get(&id)) {
        ui.add(egui::Label::new(egui::RichText::new(format!("⚠ {e}")).color(WARN)).wrap());
        return;
    }
    match id.and_then(|id| ws.built.ghost_volumes.get(&id)) {
        Some(v) => hint(ui, &format!("Preview: {} — drag the grips in the viewport, or type below.", fmt_vol(*v, DisplayUnits::Mm, 1, FmtOptions::default()))),
        None => hint(ui, "Pick a profile to see the body."),
    }
}

/// An option list whose option under the pointer is reported — for the workspace to draw the
/// plane or the axis it stands for on the part — with its options in groups. Returns the key of
/// the option chosen.
pub(crate) fn ref_picker(ui: &mut egui::Ui, salt: &str, chosen: Option<&str>, placeholder: &str, options: &[(String, String, &'static str)]) -> (Option<String>, Option<String>) {
    let mut picked = None;
    let mut hovered = None;
    let shown = chosen.and_then(|k| options.iter().find(|o| o.0 == k)).map(|o| o.1.clone());
    let text = match &shown {
        Some(s) => egui::RichText::new(s),
        None => egui::RichText::new(placeholder).weak(),
    };
    egui::ComboBox::from_id_salt(salt).selected_text(text).width(170.0).show_ui(ui, |ui| {
        if options.is_empty() {
            ui.weak("Nothing to choose from");
        }
        let mut group = "";
        for (key, label, g) in options {
            if *g != group {
                group = g;
                ui.label(egui::RichText::new(g.to_uppercase()).small().weak());
            }
            let r = ui.selectable_label(chosen == Some(key.as_str()), label);
            if r.hovered() {
                hovered = Some(key.clone());
            }
            if r.clicked() {
                picked = Some(key.clone());
            }
        }
    });
    (picked, hovered)
}

// ---- the part ----

/// The middle of what is shown and how far it reaches from there (part coordinates): what a hint
/// is sized by.
pub(crate) fn model_extent(ws: &ReverseWorkspace) -> (DVec3, f64) {
    let mut lo = DVec3::splat(f64::INFINITY);
    let mut hi = DVec3::splat(f64::NEG_INFINITY);
    let mut take = |b: mm_core::Aabb| {
        if b.is_empty() {
            return;
        }
        for c in [b.min, b.max] {
            let p = ws.to_part(c);
            lo = lo.min(p);
            hi = hi.max(p);
        }
    };
    for o in ws.scan.iter().filter(|o| o.visible) {
        take(o.mesh.bounds());
    }
    for b in &ws.bodies {
        take(b.mesh.bounds());
    }
    if !lo.x.is_finite() {
        return (DVec3::ZERO, 60.0);
    }
    ((lo + hi) / 2.0, ((hi - lo).length() / 2.0).max(1.0))
}

/// A plane or an axis an option stands for, on the part: a square sheet 2.3 times the model's
/// radius wide in amber with a fat border and its name in the middle; an axis an amber line
/// across the whole part with its name at its foot.
pub(crate) fn paint_hint(ws: &ReverseWorkspace, painter: &egui::Painter, vp: Viewport, hint: &Hint) {
    let (centre, radius) = model_extent(ws);
    let half = 1.15 * radius;
    match hint {
        Hint::Plane { origin, normal, name } => {
            let Some(n) = normal.try_normalize() else { return };
            let c = centre - n * (centre - *origin).dot(n);
            let u = mm_reverse::frame::sketch_x_axis(n);
            let u = (u - n * u.dot(n)).normalize_or_zero();
            let v = n.cross(u);
            let corners: Vec<Pos2> = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)].iter().filter_map(|(a, b)| ws.project(vp, c + u * (a * half) + v * (b * half))).collect();
            if corners.len() == 4 {
                painter.add(egui::Shape::convex_polygon(corners.clone(), AMBER.gamma_multiply(0.18), Stroke::NONE));
                painter.add(egui::Shape::closed_line(corners, Stroke::new(2.5, AMBER.gamma_multiply(0.9))));
            }
            if let Some(q) = ws.project(vp, c) {
                painter.text(q, egui::Align2::CENTER_CENTER, name, egui::FontId::proportional(13.0), AMBER);
            }
        }
        Hint::Axis { origin, dir, name } => {
            let Some(d) = dir.try_normalize() else { return };
            let foot = *origin + d * (centre - *origin).dot(d);
            let (a, b) = (foot - d * half, foot + d * half);
            if let (Some(pa), Some(pb)) = (ws.project(vp, a), ws.project(vp, b)) {
                painter.line_segment([pa, pb], Stroke::new(3.0, AMBER));
            }
            // The name at the foot of the part's middle on the axis.
            if let Some(q) = ws.project(vp, foot) {
                painter.text(q + Vec2::new(8.0, -8.0), egui::Align2::LEFT_BOTTOM, name, egui::FontId::proportional(13.0), AMBER);
            }
        }
    }
}

/// The open box's hint: the option under the pointer, or the one chosen.
pub(crate) fn paint_box_hint(ws: &ReverseWorkspace, painter: &egui::Painter, vp: Viewport) {
    let (hover, chosen) = with_box(|b| (b.hover_hint.clone(), b.chosen_hints.clone()));
    match hover {
        Some(h) => paint_hint(ws, painter, vp, &h),
        None => {
            for h in &chosen {
                paint_hint(ws, painter, vp, h);
            }
        }
    }
}

/// A sketch's closed regions as flat faces on the part, holes cut out — the sketch's blue, faint
/// at rest and stronger under the pointer; picked ones green — drawn through the part.
pub(crate) fn paint_region_faces(ws: &ReverseWorkspace, painter: &egui::Painter, vp: Viewport, sketch: FeatureId, picked: &[String]) {
    let Some(frame) = ws.sketch_frame(sketch) else { return };
    let Some(built) = ws.built.sketches.get(&sketch) else { return };
    for r in &built.regions {
        let chosen = picked.contains(&r.key);
        let hovered = ws.view.hover_region.as_deref() == Some(r.key.as_str());
        let color = match (chosen, hovered) {
            (true, true) => PICK_GREEN.gamma_multiply(0.62),
            (true, false) => PICK_GREEN.gamma_multiply(0.5),
            (false, true) => SKETCH_BLUE.gamma_multiply(0.32),
            (false, false) => SKETCH_BLUE.gamma_multiply(0.16),
        };
        let mut mesh = egui::Mesh::default();
        for t in r.region.triangulate(0.05) {
            let pts: Vec<Pos2> = t.iter().filter_map(|p| ws.project(vp, frame.to_world(DVec2::new(p.x, p.y)))).collect();
            if pts.len() != 3 {
                continue;
            }
            let base = mesh.vertices.len() as u32;
            for p in pts {
                mesh.colored_vertex(p, color);
            }
            mesh.add_triangle(base, base + 1, base + 2);
        }
        painter.add(egui::Shape::mesh(mesh));
        if chosen || hovered {
            for l in r.outline(0.05) {
                let pts: Vec<Pos2> = l.iter().filter_map(|q| ws.project(vp, frame.to_world(*q))).collect();
                if pts.len() >= 2 {
                    painter.add(egui::Shape::closed_line(pts, Stroke::new(1.5, if chosen { PICK_GREEN } else { SKETCH_BLUE })));
                }
            }
        }
    }
}

// ---- grips ----

/// A grip as the screen shows it: an arrow from its foot to its tip, or a ring's points.
#[derive(Debug, Clone)]
pub(crate) enum ScreenGrip {
    Arrow { base: Pos2, tip: Pos2 },
    Ring { points: Vec<Pos2> },
}

/// The pointer's ray, in part coordinates.
pub(crate) fn part_ray(ws: &ReverseWorkspace, vp: Viewport, p: Pos2) -> (DVec3, DVec3) {
    let r = ws.view.camera.ray_from_ndc(vp.ndc(p), vp.aspect());
    (ws.to_part(r.origin), ws.to_part(r.dir).normalize_or_zero())
}

/// Where each grip stands on the screen.
pub(crate) fn on_screen(ws: &ReverseWorkspace, vp: Viewport, grips: &[FeatureGrip]) -> Vec<Option<ScreenGrip>> {
    grips
        .iter()
        .map(|g| match g.shape {
            GripShape::Arrow => {
                let base = ws.project(vp, g.position)?;
                let ahead = ws.project(vp, g.position + g.dir * 1e-3 * g.size.max(1.0))?;
                let d = ahead - base;
                let d = if d.length() > 1e-6 { d.normalized() } else { Vec2::new(0.0, -1.0) };
                Some(ScreenGrip::Arrow { base, tip: base + d * g.size as f32 })
            }
            GripShape::Ring => {
                let (e1, e2) = ring_basis(g.dir);
                let points: Vec<Pos2> = (0..=64)
                    .filter_map(|i| {
                        let a = std::f64::consts::TAU * i as f64 / 64.0;
                        ws.project(vp, g.position + (e1 * a.cos() + e2 * a.sin()) * g.size)
                    })
                    .collect();
                (points.len() > 8).then_some(ScreenGrip::Ring { points })
            }
        })
        .collect()
}

fn ring_basis(dir: DVec3) -> (DVec3, DVec3) {
    let d = dir.normalize_or_zero();
    let e1 = mm_reverse::frame::sketch_x_axis(d);
    let e1 = (e1 - d * e1.dot(d)).normalize_or_zero();
    (e1, d.cross(e1))
}

fn seg_dist(p: Pos2, a: Pos2, b: Pos2) -> f32 {
    let ab = b - a;
    let t = if ab.length_sq() > 0.0 { ((p - a).dot(ab) / ab.length_sq()).clamp(0.0, 1.0) } else { 0.0 };
    (a + ab * t - p).length()
}

/// The grip under a screen point, by its place in the list.
pub(crate) fn grip_under(screen: &[Option<ScreenGrip>], p: Pos2) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    for (i, s) in screen.iter().enumerate() {
        let d = match s {
            Some(ScreenGrip::Arrow { base, tip }) => seg_dist(p, *base, *tip),
            Some(ScreenGrip::Ring { points }) => points.windows(2).map(|w| seg_dist(p, w[0], w[1])).fold(f32::INFINITY, f32::min),
            None => continue,
        };
        if d <= 8.0 && best.is_none_or(|(_, x)| d < x) {
            best = Some((i, d));
        }
    }
    best.map(|b| b.0)
}

/// How far along an arrow's line the pointer's ray passes nearest it, mm from its foot.
pub(crate) fn arrow_param(ws: &ReverseWorkspace, vp: Viewport, p: Pos2, g: &FeatureGrip) -> Option<f64> {
    let (o, d) = part_ray(ws, vp, p);
    let a = g.dir.try_normalize()?;
    let w0 = g.position - o;
    let b = a.dot(d);
    let denom = 1.0 - b * b;
    if denom < 1e-6 {
        return None;
    }
    Some((b * d.dot(w0) - a.dot(w0)) / denom)
}

/// The angle about a ring's axis where the pointer's ray meets the ring's plane, radians.
pub(crate) fn ring_angle(ws: &ReverseWorkspace, vp: Viewport, p: Pos2, g: &FeatureGrip) -> Option<f64> {
    let (o, d) = part_ray(ws, vp, p);
    let n = g.dir.try_normalize()?;
    let c = d.dot(n);
    if c.abs() < 1e-6 {
        return None;
    }
    let s = (g.position - o).dot(n) / c;
    let q = o + d * s - g.position;
    let (e1, e2) = ring_basis(n);
    Some(q.dot(e2).atan2(q.dot(e1)))
}

/// A drag's pull since it began: millimetres along an arrow, degrees about a ring (wound round
/// and round). `None` where the pointer reads nothing on the grip.
pub(crate) fn drag_delta(ws: &ReverseWorkspace, vp: Viewport, p: Pos2, drag: &mut GripDrag) -> Option<f64> {
    match drag.grip.shape {
        GripShape::Arrow => arrow_param(ws, vp, p, &drag.grip).map(|t| t - drag.t0),
        GripShape::Ring => {
            let a = ring_angle(ws, vp, p, &drag.grip)?;
            let mut step = a - drag.last;
            while step > std::f64::consts::PI {
                step -= std::f64::consts::TAU;
            }
            while step < -std::f64::consts::PI {
                step += std::f64::consts::TAU;
            }
            drag.wound += step;
            drag.last = a;
            Some(drag.wound.to_degrees())
        }
    }
}

/// Where a drag takes hold of a grip.
pub(crate) fn drag_start(ws: &ReverseWorkspace, vp: Viewport, p: Pos2, g: &FeatureGrip) -> Option<f64> {
    match g.shape {
        GripShape::Arrow => arrow_param(ws, vp, p, g),
        GripShape::Ring => ring_angle(ws, vp, p, g),
    }
}

/// The grips in the ghost's colour (red for a cut), the one under the pointer white; arrows held
/// at their size on the screen, rings about their axis. Each arrow's tip is kept for its field.
pub(crate) fn paint_grips(ws: &ReverseWorkspace, painter: &egui::Painter, vp: Viewport, grips: &[FeatureGrip], cut: bool) {
    let color = if cut { CUT_RED } else { GHOST };
    let hover = with_box(|b| b.drag.as_ref().map(|d| d.side).or(b.hover_grip));
    if hover.is_some() {
        painter.ctx().set_cursor_icon(egui::CursorIcon::Grab);
    }
    let screen = on_screen(ws, vp, grips);
    let mut anchors = Vec::new();
    for (g, s) in grips.iter().zip(&screen) {
        let c = if hover == Some(g.side) { Color32::WHITE } else { color.gamma_multiply(0.95) };
        match s {
            Some(ScreenGrip::Arrow { base, tip }) => {
                let d = (*tip - *base).normalized();
                let side = Vec2::new(-d.y, d.x);
                let head = *tip - d * 12.0;
                painter.line_segment([*base, head], Stroke::new(3.0, c));
                painter.add(egui::Shape::convex_polygon(vec![*tip, head + side * 6.0, head - side * 6.0], c, Stroke::new(1.0, Color32::BLACK.gamma_multiply(0.5))));
                painter.circle_filled(*base, 3.0, c);
                anchors.push((g.side, *tip, d));
            }
            Some(ScreenGrip::Ring { points }) => {
                painter.add(egui::Shape::line(points.clone(), Stroke::new(4.0, c.gamma_multiply(0.85 / 0.95))));
                if let Some(top) = points.iter().min_by(|a, b| a.y.total_cmp(&b.y)) {
                    anchors.push((g.side, *top, Vec2::new(0.0, -1.0)));
                }
            }
            None => {}
        }
    }
    with_box(|b| b.anchors = anchors);
}

/// The number a grip's field reads, and its unit word.
pub(crate) fn field_words(value: f64, side: GripSide) -> (String, &'static str) {
    match side {
        GripSide::Taper | GripSide::Turn => (format!("{value:.2}"), "°"),
        _ => (format!("{value:.2}"), "mm"),
    }
}

/// The field on a grip: beside the tip of the arrow it belongs to, read-only while the grip is
/// dragged, taking a number or an expression once let go — Enter keeps it, Escape closes it.
/// Returns the value typed when Enter keeps one.
pub(crate) fn grip_field(ctx: &egui::Context, current: impl Fn(GripSide) -> Option<f64>) -> Option<(GripSide, f64)> {
    let (field, anchor) = with_box(|b| {
        let f = b.field.clone()?;
        let a = b.anchors.iter().find(|a| a.0 == f.side).map(|a| (a.1, a.2))?;
        Some((f, a))
    })?;
    let (tip, dir) = anchor;
    let at = tip + dir * 6.0;
    let mut out = None;
    let mut close = false;
    let mut next = field.clone();
    egui::Area::new(egui::Id::NULL.with("re_grip_field")).fixed_pos(at + Vec2::new(if dir.x < 0.0 { -96.0 } else { 0.0 }, if dir.y < 0.0 { -24.0 } else { 0.0 })).order(egui::Order::Foreground).show(ctx, |ui| {
        egui::Frame::popup(ui.style()).inner_margin(4.0).show(ui, |ui| {
            let unit = field_words(0.0, field.side).1;
            if !field.typing {
                let v = current(field.side).unwrap_or(f64::NAN);
                ui.label(egui::RichText::new(format!("{} {unit}", field_words(v, field.side).0)).monospace());
                return;
            }
            ui.horizontal(|ui| {
                let r = ui.add(egui::TextEdit::singleline(&mut next.text).desired_width(70.0)).on_hover_text("The number this grip drags — type a value and press Enter; Escape closes the field");
                ui.label(egui::RichText::new(unit).italics());
                let (enter, esc) = ui.input(|i| (i.key_pressed(egui::Key::Enter), i.key_pressed(egui::Key::Escape)));
                if esc {
                    close = true;
                } else if enter && (r.has_focus() || r.lost_focus()) {
                    let kind = if unit == "°" { mm_reverse::expression::ParameterKind::Angle } else { mm_reverse::expression::ParameterKind::Length };
                    match mm_reverse::expression::evaluate_for(&next.text, kind, &mm_reverse::expression::NoNames, false) {
                        Ok(v) => {
                            out = Some((field.side, v.value));
                            close = true;
                        }
                        Err(e) => next.error = Some(e.message),
                    }
                }
            });
            if let Some(e) = &next.error {
                ui.colored_label(WARN, e);
            }
        });
    });
    with_box(|b| b.field = if close { None } else { Some(next) });
    out
}

/// The outline of a prism being dragged, drawn at once over the ghost the kernel has not built
/// yet: the profile's loops at both ends and lines between them — the ghost stretched, frame by
/// frame, until the new one lands.
pub(crate) fn paint_stretch(ws: &ReverseWorkspace, painter: &egui::Painter, vp: Viewport, loops: &[Vec<DVec3>], along: DVec3, from: f64, to: f64, cut: bool) {
    let color = if cut { CUT_RED } else { GHOST };
    let stroke = Stroke::new(1.5, color);
    for l in loops {
        for h in [from, to] {
            let pts: Vec<Pos2> = l.iter().filter_map(|p| ws.project(vp, *p + along * h)).collect();
            if pts.len() >= 2 {
                painter.add(egui::Shape::closed_line(pts, stroke));
            }
        }
        let step = (l.len() / 8).max(1);
        for p in l.iter().step_by(step) {
            if let (Some(a), Some(b)) = (ws.project(vp, *p + along * from), ws.project(vp, *p + along * to)) {
                painter.line_segment([a, b], stroke);
            }
        }
    }
}
