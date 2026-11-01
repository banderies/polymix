// SPDX-License-Identifier: AGPL-3.0-only
//! The sketch box in the panel: Finish at the top and the bottom, the plane, its offset and
//! the level the scan is cut at, the picture under the sketch, the fit commands, the tools —
//! each shape tool with the ways it draws in a strip under its key — the Dimension key and the
//! nine relations, the edits, the Pattern box, the relations inferred while drawing, the list
//! of relations and dimensions and the solver's verdict. Ported from ScanRuler
//! (`plugins/reverse/ui/SketchEditor.tsx`, `FitOptionsModal.tsx`), © Stefan Hermann (CNC
//! Kitchen), AGPL-3.0-only. The box says little in prose: what a key does is in its tooltip,
//! and only an error or a wait gets a line.

use egui::{Color32, RichText, Sense, Stroke};
use glam::DVec2;
use mm_reverse::corners::CORNER_GAP_MM;
use mm_reverse::draw::{ArcMode, CircleMode, RectangleMode};
use mm_reverse::model::{plane_offset, Part, PlaneRef};
use mm_reverse::nominal::{nominal_of, NominalKind};
use mm_reverse::sketch::{ConstraintId, Rule, ShapeKind};
use mm_reverse::sketch_image::{placed_image, IMAGE_MAX_PX};
use mm_reverse::sketch_shapes::{fixed_pieces, PolygonFit, SketchPatternSpec, POLYGON_MAX_SIDES};

use super::sketch_icons::{self, Ink};
use super::sketcher::{clamp_sides, constraint_label, fmt_value, plural, set_prefs, ImagePickMode, Pick, RelationKind, SketchEdit, Tool};
use super::view::SketchAction;

// ---- the look (ScanRuler's colour tokens) ----

#[derive(Debug, Clone, Copy)]
pub(crate) struct Tokens {
    pub well: Color32,
    pub line: Color32,
    pub ink: Color32,
    pub dim: Color32,
    pub faint: Color32,
    pub accent: Color32,
    pub accent_d: Color32,
    pub accent_soft: Color32,
    pub amber: Color32,
    pub red: Color32,
    pub alarm_bg: Color32,
    pub alarm_line: Color32,
}

const fn hex(c: u32) -> Color32 {
    Color32::from_rgb((c >> 16) as u8, (c >> 8) as u8, c as u8)
}

pub(crate) fn tokens(ui: &egui::Ui) -> Tokens {
    if ui.visuals().dark_mode {
        Tokens {
            well: hex(0x1e2124),
            line: hex(0x383c41),
            ink: hex(0xe4e6e8),
            dim: hex(0xa0a4a8),
            faint: hex(0x767a7f),
            accent: hex(0x3f93d6),
            accent_d: hex(0x62aae4),
            accent_soft: hex(0x223c55),
            amber: hex(0xe2a43b),
            red: hex(0xe26b4f),
            alarm_bg: hex(0x3b241d),
            alarm_line: hex(0x6e3b2c),
        }
    } else {
        Tokens {
            well: hex(0xefeee9),
            line: hex(0xcfcec7),
            ink: hex(0x26282a),
            dim: hex(0x75787b),
            faint: hex(0x9a9d9f),
            accent: hex(0x1877c0),
            accent_d: hex(0x12629f),
            accent_soft: hex(0xd9e8f5),
            amber: hex(0xc97b10),
            red: hex(0xb3361c),
            alarm_bg: hex(0xf8ece8),
            alarm_line: hex(0xddb4a8),
        }
    }
}

// ---- the tools, their tooltips and the ways the shape tools draw ----

/// A key of the tool row: the tool, its picture, its label and what it does.
struct ToolKey {
    tool: Tool,
    icon: &'static str,
    label: &'static str,
    title: &'static str,
}

const TOOLS: [ToolKey; 9] = [
    ToolKey { tool: Tool::Select, icon: "select", label: "Select", title: "Click a piece to select it, a second to pair it; click empty sheet to clear. Drag over empty sheet for a box: rightward it takes what lies wholly inside, leftward whatever it touches; hold Shift or Ctrl to add to the selection. Drag an end, a centre or a piece to move it — it snaps to other ends, the slice and the pieces, and a line moves square to itself; hold Ctrl to move it freely. The right button pans" },
    ToolKey { tool: Tool::Line, icon: "line", label: "Line (L)", title: "Click the start, then every next corner — each line carries on from the last. Press on the last end and drag for an arc tangent to the piece before it. Escape, Enter or a second click on the last end ends the chain; so does landing on another piece’s end. Hold Ctrl to place a point without snapping" },
    ToolKey { tool: Tool::Rectangle, icon: "rectangle", label: "Rectangle (R)", title: "Two opposite corners, the centre and a corner, or one side and the far side at any angle — rest the cursor on the key to choose; the key shows the way last chosen. The rectangle follows the cursor between the clicks" },
    ToolKey { tool: Tool::Polygon, icon: "polygon", label: "Polygon", title: "A regular polygon: its centre, then a corner — or the middle of a side, for one sized across its flats like a hex nut. The sides are equal and drawn about a construction circle to dimension; how many there are is set under the keys while the tool is in hand" },
    ToolKey { tool: Tool::Circle, icon: "circle", label: "Circle (C)", title: "The centre and a point on the rim, the two ends of a diameter, or three points on the rim — rest the cursor on the key to choose; the key shows the way last chosen. The circle follows the cursor between the clicks" },
    ToolKey { tool: Tool::Arc, icon: "arc", label: "Arc", title: "The start, the end and a point it passes through, or the centre, the start and where it ends — rest the cursor on the key to choose; the key shows the way last chosen. The arc follows the cursor" },
    ToolKey { tool: Tool::Fillet, icon: "filletCorner", label: "Fillet (F)", title: "Round a corner: click the corner point where two lines meet, or one line and then the other. With two lines selected, the key rounds those. The lines need not touch — a short piece bridging their ends is replaced by the round. The radius comes out as a dimension to type over; the next round starts with the last radius" },
    ToolKey { tool: Tool::Chamfer, icon: "chamferCorner", label: "Chamfer", title: "Cut a corner off straight: click the corner point where two lines meet, or one line and then the other. With two lines selected, the key chamfers those. The two distances come out as dimensions to type over — alike for an even chamfer, apart for an uneven one; the next chamfer starts with the last distance" },
    ToolKey { tool: Tool::Project, icon: "project", label: "Project", title: "Click an edge of a body, or a piece of another sketch, on the part under the sheet: it comes into this sketch as a piece of its own, fixed where it lies — to close a profile with, or to hang relations and dimensions on. It lights under the cursor when it can be taken" },
];

/// One way a shape tool draws: its picture, label, tooltip and what each click asks for.
pub struct DrawWay {
    pub icon: &'static str,
    pub label: &'static str,
    pub title: &'static str,
    pub clicks: &'static [&'static str],
}

const RECTANGLE_WAYS: [DrawWay; 3] = [
    DrawWay { icon: "rectangle", label: "Corner rectangle", title: "Two opposite corners", clicks: &["Click one corner.", "Click the opposite corner."] },
    DrawWay { icon: "rectangleCenter", label: "Centre rectangle", title: "The centre, then a corner — the rectangle stays about its centre, a point where its construction diagonals cross", clicks: &["Click the centre.", "Click a corner."] },
    DrawWay { icon: "rectangleThreePoint", label: "3-point rectangle", title: "The two ends of one side, then where the opposite side lies — a rectangle at any angle", clicks: &["Click one end of a side.", "Click the other end of that side.", "Click where the opposite side lies."] },
];
const CIRCLE_WAYS: [DrawWay; 3] = [
    DrawWay { icon: "circle", label: "Centre circle", title: "The centre, then a point on the rim", clicks: &["Click the centre.", "Click a point on the rim."] },
    DrawWay { icon: "circleTwoPoint", label: "2-point circle", title: "The two ends of a diameter", clicks: &["Click one end of a diameter.", "Click the other end."] },
    DrawWay { icon: "circleThreePoint", label: "3-point circle", title: "Three points on the rim", clicks: &["Click a point on the rim — 1 of 3.", "Click a point on the rim — 2 of 3.", "Click a point on the rim — 3 of 3."] },
];
const ARC_WAYS: [DrawWay; 2] = [
    DrawWay { icon: "arc", label: "3-point arc", title: "The start, the end, then a point the arc passes through", clicks: &["Click the start.", "Click the end.", "Click a point the arc passes through."] },
    DrawWay { icon: "arcCenter", label: "Centre arc", title: "The centre, the start, then where it ends — the shorter way round; a longer arc is drawn through three points", clicks: &["Click the centre.", "Click the start — how far it lies is the radius.", "Click where it ends — it runs the shorter way round."] },
];
const POLYGON_WAYS: [DrawWay; 2] = [
    DrawWay { icon: "polygon", label: "Inscribed polygon", title: "The centre, then a corner — the corners lie on the construction circle", clicks: &["Click the centre.", "Click a corner."] },
    DrawWay { icon: "polygonOutside", label: "Circumscribed polygon", title: "The centre, then the middle of a side — the sides touch the construction circle, whose diameter is the width across flats", clicks: &["Click the centre.", "Click the middle of a side."] },
];

/// The ways a shape tool draws, and which is chosen now.
pub fn draw_ways(edit: &SketchEdit, tool: Tool) -> (&'static [DrawWay], usize) {
    let m = &edit.prefs.draw_modes;
    match tool {
        Tool::Rectangle => (&RECTANGLE_WAYS, match m.rectangle {
            RectangleMode::Corners => 0,
            RectangleMode::Center => 1,
            RectangleMode::ThreePoint => 2,
        }),
        Tool::Circle => (&CIRCLE_WAYS, match m.circle {
            CircleMode::Center => 0,
            CircleMode::TwoPoint => 1,
            CircleMode::ThreePoint => 2,
        }),
        Tool::Arc => (&ARC_WAYS, match m.arc {
            ArcMode::ThreePoint => 0,
            ArcMode::Center => 1,
        }),
        Tool::Polygon => (&POLYGON_WAYS, match m.polygon {
            PolygonFit::Inscribed => 0,
            PolygonFit::Circumscribed => 1,
        }),
        _ => (&[], 0),
    }
}

/// Choose a shape tool's way: remembered, and the tool taken up in it — a way chosen half way
/// through starts the shape afresh.
pub fn choose_way(edit: &mut SketchEdit, tool: Tool, way: usize) {
    let m = &mut edit.prefs.draw_modes;
    match tool {
        Tool::Rectangle => m.rectangle = [RectangleMode::Corners, RectangleMode::Center, RectangleMode::ThreePoint][way.min(2)],
        Tool::Circle => m.circle = [CircleMode::Center, CircleMode::TwoPoint, CircleMode::ThreePoint][way.min(2)],
        Tool::Arc => m.arc = [ArcMode::ThreePoint, ArcMode::Center][way.min(1)],
        Tool::Polygon => m.polygon = [PolygonFit::Inscribed, PolygonFit::Circumscribed][way.min(1)],
        _ => return,
    }
    set_prefs(edit.prefs);
    edit.set_tool(tool);
}

const RELATION_ICONS: [&str; 9] = ["coincident", "horizontal", "vertical", "parallel", "perpendicular", "tangent", "concentric", "equal", "symmetric"];

const DIMENSION_TITLE: &str = "Dimension (D) — click a line for its length, an arc for its radius, a circle for its diameter (Ø — typed as R… it goes by the radius instead); click a second piece for what lies between the two: the distance between parallel lines, the angle between others, the distance from an end or a centre to a line or to another, or to one of the part’s coordinate planes (the amber lines named XY, YZ, ZX). The number then follows the cursor: click where it goes, and type the value; Escape takes it back. With pieces selected, the key dimensions those.";

const FINISH_TITLE: &str = "Finish the sketch — Enter or a middle click does the same. With nothing drawn, nothing is made.";

const LEVEL_TITLE: &str = "Cut the scan this far off the sketch’s plane, along its normal, and fit that level’s outline into this sketch — another height of a stacked part. Fit all chains then lands on the pieces already drawn: a wall they have is not fitted again, and the new walls end on theirs. 0 cuts at the plane (a face’s a hair inside it).";

/// The box's title: `New sketch`, or `Edit {name}`.
pub fn title(part: &Part, edit: &SketchEdit) -> String {
    match edit.feature.and_then(|id| part.feature(id)) {
        Some(f) => format!("Edit {}", f.name),
        None => "New sketch".into(),
    }
}

// ---- small widgets ----

/// A key that is a picture only, its name in its tooltip.
#[allow(clippy::too_many_arguments)]
fn icon_key(ui: &mut egui::Ui, size: egui::Vec2, icon: &str, icon_px: f32, on: bool, enabled: bool, tick: bool, tip: &str) -> egui::Response {
    let t = tokens(ui);
    let (rect, resp) = ui.allocate_exact_size(size, if enabled { Sense::click() } else { Sense::hover() });
    let lit = enabled && (on || resp.hovered());
    let painter = ui.painter();
    painter.rect_filled(rect, 3.0, if lit { t.accent_soft } else { t.well });
    painter.rect_stroke(rect, 3.0, Stroke::new(1.0, if lit { t.accent } else { t.line }), egui::StrokeKind::Inside);
    let ink = if !enabled {
        t.faint
    } else if lit {
        t.accent_d
    } else {
        t.ink
    };
    sketch_icons::paint(painter, egui::Rect::from_center_size(rect.center(), egui::vec2(icon_px, icon_px)), icon, Ink { ink, accent: t.accent, soft: t.accent_soft, primary: false });
    if tick {
        // A small corner mark: the key has more ways.
        let c = rect.right_bottom() + egui::vec2(-3.0, -3.0);
        painter.add(egui::Shape::convex_polygon(vec![c, c + egui::vec2(-5.0, 0.0), c + egui::vec2(0.0, -5.0)], ink.gamma_multiply(0.45), Stroke::NONE));
    }
    if tip.is_empty() {
        resp
    } else if enabled {
        resp.on_hover_text(tip)
    } else {
        resp.on_disabled_hover_text(tip)
    }
}

/// The widths of a row of picture keys: as many columns of at least 36 points as fit.
fn key_columns(width: f32, min: f32, gap: f32) -> (usize, f32) {
    let cols = (((width + gap) / (min + gap)).floor() as usize).max(1);
    (cols, (width - gap * (cols as f32 - 1.0)) / cols as f32)
}

/// A primary full-width key.
fn primary_block(ui: &mut egui::Ui, text: &str, tip: &str) -> egui::Response {
    let t = tokens(ui);
    let w = ui.available_width();
    ui.add(egui::Button::new(RichText::new(text).strong().color(Color32::WHITE)).fill(t.accent).min_size(egui::vec2(w, 26.0))).on_hover_text(tip)
}

/// A primary key with a picture before its word.
fn primary_icon(ui: &mut egui::Ui, icon: &str, text: &str, tip: &str) -> bool {
    let t = tokens(ui);
    let r = ui.add(egui::Button::new(RichText::new(format!("      {text}")).strong().color(Color32::WHITE)).fill(t.accent).min_size(egui::vec2(0.0, 26.0))).on_hover_text(tip);
    let at = egui::Rect::from_min_size(r.rect.left_center() + egui::vec2(6.0, -9.0), egui::vec2(18.0, 18.0));
    sketch_icons::paint(ui.painter(), at, icon, Ink { ink: Color32::WHITE, accent: t.accent, soft: t.accent_soft, primary: true });
    r.clicked()
}

/// A key with a picture before its word.
fn icon_button(ui: &mut egui::Ui, icon: &str, text: &str, enabled: bool, tip: &str) -> bool {
    let t = tokens(ui);
    let r = ui.add_enabled(enabled, egui::Button::new(format!("      {text}")).min_size(egui::vec2(0.0, 24.0)));
    let ink = if enabled { t.ink } else { t.faint };
    let at = egui::Rect::from_min_size(r.rect.left_center() + egui::vec2(5.0, -9.0), egui::vec2(18.0, 18.0));
    sketch_icons::paint(ui.painter(), at, icon, Ink { ink, accent: t.accent, soft: t.accent_soft, primary: false });
    let r = if enabled { r.on_hover_text(tip) } else { r.on_disabled_hover_text(tip) };
    r.clicked()
}

/// A pill chip that takes one click (`chip nominal`).
fn chip(ui: &mut egui::Ui, text: &str, tip: &str) -> bool {
    let t = tokens(ui);
    let galley = ui.painter().layout_no_wrap(text.to_string(), egui::FontId::monospace(10.5), t.accent);
    let size = galley.size() + egui::vec2(14.0, 4.0);
    let (rect, resp) = ui.allocate_exact_size(size, Sense::click());
    let hovered = resp.hovered();
    let painter = ui.painter();
    if hovered {
        painter.rect_filled(rect, 10.0, t.accent);
    }
    painter.rect_stroke(rect, 10.0, Stroke::new(1.0, t.accent), egui::StrokeKind::Inside);
    painter.text(rect.center(), egui::Align2::CENTER_CENTER, text, egui::FontId::monospace(10.5), if hovered { Color32::WHITE } else { t.accent });
    resp.on_hover_text(tip).clicked()
}

/// The ⓘ after a heading: a click opens a card beside the panel; a second click, a click
/// anywhere else or Escape closes it.
fn info_dot(ui: &mut egui::Ui, id: &str, heading: &str, paragraphs: &[&str]) {
    let t = tokens(ui);
    let key = egui::Id::NULL.with(("re_info_dot", id));
    let mut open = ui.ctx().data(|d| d.get_temp::<bool>(key)).unwrap_or(false);
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(15.0, 15.0), Sense::click());
    let painter = ui.painter();
    painter.circle_stroke(rect.center(), 6.5, Stroke::new(1.0, if open || resp.hovered() { t.accent } else { t.dim }));
    painter.text(rect.center(), egui::Align2::CENTER_CENTER, "i", egui::FontId::proportional(10.0), if open { t.accent } else { t.dim });
    if resp.clicked() {
        open = !open;
    }
    if open {
        let x = ui.clip_rect().right() + 16.0;
        let area = egui::Area::new(key.with("card")).fixed_pos(egui::pos2(x, rect.top())).order(egui::Order::Foreground).show(ui.ctx(), |ui| {
            egui::Frame::popup(ui.style()).show(ui, |ui| {
                ui.set_max_width(340.0);
                ui.label(RichText::new(heading).strong());
                for p in paragraphs {
                    ui.add_space(4.0);
                    ui.label(*p);
                }
            });
        });
        let elsewhere = ui.input(|i| i.pointer.any_click()) && !resp.clicked() && !ui.input(|i| i.pointer.interact_pos()).is_some_and(|p| area.response.rect.contains(p));
        if elsewhere || ui.input(|i| i.key_pressed(egui::Key::Escape)) {
            open = false;
        }
    }
    ui.ctx().data_mut(|d| d.insert_temp(key, open));
}

/// A length typed into a field, committed on Enter or when the field is left.
fn number_field(ui: &mut egui::Ui, id: egui::Id, value: f64, width: f32) -> Option<f64> {
    let mut text = ui.ctx().data(|d| d.get_temp::<String>(id)).unwrap_or_else(|| fmt_value(value));
    let r = ui.add(egui::TextEdit::singleline(&mut text).desired_width(width).font(egui::TextStyle::Monospace));
    let mut out = None;
    if r.has_focus() {
        ui.ctx().data_mut(|d| d.insert_temp(id, text.clone()));
    }
    if r.lost_focus() {
        if let Ok(v) = text.trim().trim_end_matches("mm").trim_end_matches('°').trim().parse::<f64>() {
            out = Some(v);
        }
        ui.ctx().data_mut(|d| d.remove::<String>(id));
    } else if !r.has_focus() {
        ui.ctx().data_mut(|d| d.remove::<String>(id));
    }
    out
}

/// A labelled row: the label at the left, the control at the right.
fn field_row<R>(ui: &mut egui::Ui, label: &str, tip: &str, add: impl FnOnce(&mut egui::Ui) -> R) -> R {
    let t = tokens(ui);
    ui.horizontal(|ui| {
        let r = ui.label(RichText::new(label).color(t.dim).size(12.0));
        if !tip.is_empty() {
            r.on_hover_text(tip);
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| add(ui)).inner
    })
    .inner
}

/// A hint line: a small dim paragraph.
fn hint(ui: &mut egui::Ui, text: &str) {
    let t = tokens(ui);
    ui.label(RichText::new(text).color(t.dim).size(11.5));
}

// ---- the picture: a file read, scaled down and kept as a data URL ----

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn base64(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let n = (chunk[0] as u32) << 16 | (*chunk.get(1).unwrap_or(&0) as u32) << 8 | *chunk.get(2).unwrap_or(&0) as u32;
        for i in 0..4 {
            if i <= chunk.len() {
                out.push(B64[((n >> (18 - 6 * i)) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

/// The bytes a data URL carries, or `None` for anything else.
pub fn data_url_bytes(src: &str) -> Option<Vec<u8>> {
    let (_, data) = src.split_once(";base64,")?;
    let mut out = Vec::with_capacity(data.len() / 4 * 3);
    let mut acc = 0u32;
    let mut bits = 0;
    for c in data.bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            b'=' => break,
            _ => continue,
        };
        acc = (acc << 6) | v as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    Some(out)
}

/// A picture file read and scaled down to what a sketch keeps: a PNG stays a PNG, a
/// drawing's lines kept sharp; a photo is kept as a JPEG.
pub fn read_picture(path: &std::path::Path) -> Result<(String, u32, u32), String> {
    let img = image::open(path).map_err(|e| e.to_string())?;
    let k = (IMAGE_MAX_PX / img.width().max(img.height()) as f64).min(1.0);
    let w = ((img.width() as f64 * k).round() as u32).max(1);
    let h = ((img.height() as f64 * k).round() as u32).max(1);
    let img = if k < 1.0 { img.resize_exact(w, h, image::imageops::FilterType::Triangle) } else { img };
    let png = path.extension().and_then(|e| e.to_str()).is_some_and(|e| e.eq_ignore_ascii_case("png"));
    let mut bytes = Vec::new();
    if png {
        img.write_to(&mut std::io::Cursor::new(&mut bytes), image::ImageFormat::Png).map_err(|e| e.to_string())?;
        Ok((format!("data:image/png;base64,{}", base64(&bytes)), w, h))
    } else {
        let rgb = img.to_rgb8();
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut bytes, 90).encode_image(&rgb).map_err(|e| e.to_string())?;
        Ok((format!("data:image/jpeg;base64,{}", base64(&bytes)), w, h))
    }
}

/// Lay a picture from a file under the sketch, about the middle of what is drawn.
pub fn lay_picture(edit: &mut SketchEdit, path: &std::path::Path) {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("Picture").to_string();
    match read_picture(path) {
        Ok((src, w, h)) => {
            let centre = edit.sketch.bounds().map_or(DVec2::ZERO, |(lo, hi)| (lo + hi) / 2.0);
            edit.image = Some(placed_image(&src, w, h, centre, Some(&name)));
            edit.image_pick = None;
            edit.error = None;
        }
        Err(_) => edit.message = format!("{name} could not be read as a picture."),
    }
}

/// The picture row (ScanRuler's `PictureRow`).
fn picture_row(ui: &mut egui::Ui, edit: &mut SketchEdit) {
    let t = tokens(ui);
    let Some(image) = edit.image.clone() else {
        if icon_button(ui, "sketchImage", "Picture…", true, "Lay a photo of the part or a dimensioned drawing under the sketch, to trace — then bring it to scale by two points a known length apart") {
            if let Some(path) = rfd::FileDialog::new().add_filter("Picture", &["png", "jpg", "jpeg", "bmp", "tga"]).pick_file() {
                lay_picture(edit, &path);
            }
        }
        return;
    };
    egui::Frame::new().fill(t.well).stroke(Stroke::new(1.0, t.line)).corner_radius(3.0).inner_margin(8.0).show(ui, |ui| {
        ui.horizontal(|ui| {
            let (r, _) = ui.allocate_exact_size(egui::vec2(16.0, 16.0), Sense::hover());
            sketch_icons::paint(ui.painter(), r, "sketchImage", Ink { ink: t.ink, accent: t.accent, soft: t.accent_soft, primary: false });
            let name = image.name.clone().unwrap_or_else(|| "Picture".into());
            ui.label(RichText::new(&name).size(12.0)).on_hover_text(&name);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.small_button("×").on_hover_text("Take the picture away").clicked() {
                    edit.image = None;
                    edit.image_pick = None;
                }
                ui.label(RichText::new(format!("{:.1} µm/px", image.mm_per_px * 1000.0)).strong()).on_hover_text("Micrometres per pixel of the picture");
            });
        });
        ui.horizontal(|ui| {
            let mode = edit.image_pick.as_ref().map(|p| p.mode);
            for (m, label, tip) in [
                (ImagePickMode::Scale, "Scale", "Bring the picture to scale: click two points on it, then type how far apart they are on the part"),
                (ImagePickMode::Move, "Move", "Move the picture: click a point on it, then where that point belongs — an end, the origin, an axis snaps"),
                (ImagePickMode::Level, "Level", "Turn the picture level: click two points on it that should lie on a horizontal line"),
            ] {
                if ui.selectable_label(mode == Some(m), label).on_hover_text(tip).clicked() {
                    edit.start_image_pick(if mode == Some(m) { None } else { Some(m) });
                }
            }
        });
        if edit.image_pick.as_ref().is_some_and(|p| p.mode == ImagePickMode::Scale && p.points.len() >= 2) {
            let key = egui::Id::NULL.with("re_sketch_image_length");
            let mut length: f64 = ui.ctx().data(|d| d.get_temp(key)).unwrap_or(0.0);
            let mut apply = false;
            ui.horizontal(|ui| {
                ui.label(RichText::new("Apart on the part").color(t.dim).size(12.0));
                let r = ui.add(egui::DragValue::new(&mut length).speed(0.1).range(0.0..=f64::MAX).suffix(" mm"));
                apply |= r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) && length > 0.0;
                if ui.add_enabled(length > 0.0, egui::Button::new(RichText::new("Scale").color(Color32::WHITE)).fill(t.accent)).clicked() {
                    apply = true;
                }
            });
            ui.ctx().data_mut(|d| d.insert_temp(key, length));
            if apply {
                edit.scale_image(length);
            }
        }
        ui.horizontal(|ui| {
            ui.label(RichText::new("Shown").color(t.dim).size(12.0)).on_hover_text("How much of the picture shows: faded, the scan and the pieces show through it");
            let mut opacity = image.opacity;
            if ui.add(egui::Slider::new(&mut opacity, 0.1..=1.0).step_by(0.05).show_value(false)).on_hover_text("How much of the picture shows: faded, the scan and the pieces show through it").changed() {
                if let Some(img) = &mut edit.image {
                    img.opacity = opacity;
                }
            }
        });
    });
}

// ---- the Pattern box ----

fn pattern_box(ui: &mut egui::Ui, edit: &mut SketchEdit) {
    let Some(mut spec) = edit.pattern else { return };
    let t = tokens(ui);
    let selected = edit.selected_entities().len();
    let count = |v: f64| (v.abs().round() as usize).clamp(1, 500);
    let spacing = 10.0;
    egui::Frame::new().fill(t.well).stroke(Stroke::new(1.0, t.line)).corner_radius(3.0).inner_margin(8.0).show(ui, |ui| {
        ui.horizontal(|ui| {
            let grid = matches!(spec, SketchPatternSpec::Rectangular { .. });
            if ui.selectable_label(grid, "▦ Grid").clicked() && !grid {
                spec = SketchPatternSpec::Rectangular { step: DVec2::new(spacing, 0.0), count: 3, step2: None, count2: None };
            }
            if ui.selectable_label(!grid, "◌ Round").clicked() && grid {
                spec = SketchPatternSpec::Circular { centre: DVec2::ZERO, count: 6, angle_deg: 360.0 };
            }
        });
        let num = |ui: &mut egui::Ui, label: &str, v: &mut f64, unit: &str| {
            let counted = unit == "×";
            field_row(ui, label, "", |ui| ui.add(egui::DragValue::new(v).speed(if counted { 0.05 } else { 0.1 }).max_decimals(if counted { 0 } else { 3 }).suffix(format!(" {unit}"))).changed())
        };
        match &mut spec {
            SketchPatternSpec::Rectangular { step, count: n, step2, count2 } => {
                let mut c = *n as f64;
                if num(ui, "Along X", &mut c, "×") {
                    *n = count(c);
                }
                num(ui, "X spacing", &mut step.x, "mm");
                step.y = 0.0;
                let mut c2 = count2.unwrap_or(1) as f64;
                if num(ui, "Along Y", &mut c2, "×") {
                    *count2 = Some(count(c2));
                    if step2.is_none() {
                        *step2 = Some(DVec2::new(0.0, if step.x != 0.0 { step.x } else { spacing }));
                    }
                }
                if count2.unwrap_or(1) > 1 {
                    let mut y = step2.map_or(spacing, |s| s.y);
                    if num(ui, "Y spacing", &mut y, "mm") {
                        *step2 = Some(DVec2::new(0.0, y));
                    }
                }
                hint(ui, "Counts take the selection itself in; a negative spacing runs the other way.");
            }
            SketchPatternSpec::Circular { centre, count: n, angle_deg } => {
                let mut c = *n as f64;
                if num(ui, "Count", &mut c, "×") {
                    *n = count(c);
                }
                num(ui, "Over", angle_deg, "°");
                num(ui, "Centre X", &mut centre.x, "mm");
                num(ui, "Centre Y", &mut centre.y, "mm");
                hint(ui, "360° spreads the copies evenly round; less spans that angle, first to last.");
            }
        }
        edit.pattern = Some(spec);
        ui.horizontal(|ui| {
            if ui.add_enabled(selected > 0, egui::Button::new(RichText::new(format!("Pattern {selected} piece{}", plural(selected))).color(Color32::WHITE)).fill(t.accent)).clicked() {
                edit.apply_pattern();
            }
            if ui.button("Cancel").clicked() {
                edit.pattern = None;
            }
        });
    });
}

// ---- the fit options window ----

const DETAIL_NAMES: [&str; 5] = ["Coarse", "Coarser", "Medium", "Finer", "Fine"];

fn fit_options_window(ui: &mut egui::Ui, edit: &mut SketchEdit) {
    let key = egui::Id::NULL.with("re_sketch_fit_options");
    let mut open = ui.ctx().data(|d| d.get_temp::<bool>(key)).unwrap_or(false);
    if !open {
        return;
    }
    let x = ui.clip_rect().right() + 16.0;
    let mut refit = false;
    egui::Window::new("Fit options").id(key.with("window")).open(&mut open).default_pos(egui::pos2(x, 96.0)).default_width(400.0).resizable(false).collapsible(false).show(ui.ctx(), |ui| {
        let t = tokens(ui);
        let o = &mut edit.fit_options;
        ui.horizontal(|ui| {
            ui.label("Detail");
            let r = ui.add(egui::Slider::new(&mut o.detail, 0.0..=1.0).step_by(0.05).show_value(false));
            ui.label(DETAIL_NAMES[((o.detail * 5.0).floor() as usize).min(4)]);
            refit |= r.drag_stopped() || (r.changed() && !r.dragged());
        });
        ui.label(RichText::new("Coarse gives few, long pieces: a wall that bows or wanders is one line, a blob in a corner is cut out for the corner its sides make, and a round no bigger than the wander becomes a sharp corner. Fine gives every small feature a piece of its own.").small().color(t.dim));
        ui.horizontal(|ui| {
            ui.label("Fit tolerance");
            let mut text = o.tolerance.map(fmt_value).unwrap_or_default();
            let r = ui.add(egui::TextEdit::singleline(&mut text).hint_text("auto").desired_width(70.0));
            if r.changed() {
                o.tolerance = text.trim().parse::<f64>().ok().filter(|v| *v > 0.0);
            }
            ui.label(RichText::new("mm").italics());
            refit |= r.lost_focus();
        });
        ui.label(RichText::new("How far an arc may miss its slice points. Left blank the chain's own scatter decides; a noisier scan wants it raised. Detail works on top of it.").small().color(t.dim));
        refit |= ui.checkbox(&mut o.join_corners, "Join corners — sides carried on to where they cross, fillets made tangent").changed();
        ui.separator();
        ui.horizontal(|ui| {
            let pieces: usize = edit.fitted.iter().map(|f| f.order.len()).sum();
            let chains = edit.fitted.len();
            ui.label(RichText::new(if chains > 0 { format!("{pieces} piece{} on {chains} chain{} — changes refit them", plural(pieces), plural(chains)) } else { "Takes effect on Fit all chains".into() }).small().color(t.dim));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("Reset").clicked() {
                    edit.fit_options = mm_reverse::section::FitOptions::default();
                    refit = true;
                }
            });
        });
    });
    if refit && !edit.fitted.is_empty() {
        edit.refit();
    }
    ui.ctx().data_mut(|d| d.insert_temp(key, open));
}

// ---- the box ----

/// The sketch box in the panel.
pub fn sketch_box(ui: &mut egui::Ui, edit: &mut SketchEdit, no_scan: bool, plane_label: &str, title: &str) -> SketchAction {
    let mut action = SketchAction::None;
    let t = tokens(ui);
    let frame = egui::Frame::new().fill(t.well).stroke(Stroke::new(1.0, t.line)).corner_radius(3.0).inner_margin(egui::Margin { left: 14, right: 11, top: 11, bottom: 11 });
    let shown = frame.show(ui, |ui| {
        ui.spacing_mut().item_spacing.y = 9.0;
        box_contents(ui, edit, no_scan, plane_label, title, &mut action);
    });
    // The box's accent bar at its left.
    let r = shown.response.rect;
    ui.painter().rect_filled(egui::Rect::from_min_size(r.min, egui::vec2(3.0, r.height())), 0.0, t.accent);
    fit_options_window(ui, edit);
    action
}

fn box_contents(ui: &mut egui::Ui, edit: &mut SketchEdit, no_scan: bool, plane_label: &str, title: &str, action: &mut SketchAction) {
    let t = tokens(ui);
    // 1. The title and its card.
    ui.horizontal(|ui| {
        ui.label(RichText::new(title.to_uppercase()).strong().size(11.5));
        info_dot(
            ui,
            "sketching",
            "Sketching",
            &[
                "The sketch lies on its plane — a section, a plane element, a coordinate plane or a face of a body — and is drawn on the part, which faces the plane and is cut open at it; the view bar's Model and Slice keys put the part away and show it whole.",
                "Fit all chains cuts every slice chain into lines and arcs, each keeping its slice points: the solver holds every piece to its points while the relations and the dimensions you type are met exactly. Infer corners joins ends within 2 mm and adds a tangency where a line runs into an arc; Close bridges what is left open.",
                "Every closed region is filled faintly; Extrude and Revolve appear as soon as one closes and take it straight into a feature. Finish sketch keeps the drawing as a feature for later; Escape only puts the tool down, and a sketch you did not want is undone with Ctrl+Z from the feature list.",
                "Dimension (D) and the relations work both ways round: on the pieces selected, or, with nothing selected, on what is clicked next. The Line tool (L) draws a chain, each line from the end of the last; press on that end and drag for a tangent arc. A new dimension's number follows the cursor until a click puts it down, and opens there to be typed; Escape takes it back. Later it is dragged to where it reads best, and clicked to be typed over.",
            ],
        );
    });
    // 2. Finish, at the top.
    if primary_block(ui, "Finish sketch", FINISH_TITLE).clicked() {
        *action = SketchAction::Finish;
    }
    // 4. The plane, and how many chains its slice has.
    hint(ui, &format!("On {plane_label}{}", if no_scan { String::new() } else { format!(" · {} slice chain{}", edit.chains.len(), plural(edit.chains.len())) }));
    // 5. Where the plane lies.
    if super::offsettable(&edit.plane) {
        let world = match &edit.plane {
            PlaneRef::World { axis, .. } => Some(format!("{axis:?}").to_uppercase()),
            _ => None,
        };
        let (label, tip) = match &world {
            Some(a) => (format!("Plane at {a} ="), "Where the coordinate plane lies along its axis — a number, or an expression over the parameters the sketch then follows"),
            None => ("Offset".to_string(), "How far the sketch stands off its plane, along the plane’s normal — out of a face is positive. A number, or an expression over the parameters"),
        };
        let now = plane_offset(&edit.plane);
        let typed = field_row(ui, &label, tip, |ui| {
            ui.label(RichText::new("mm").italics());
            number_field(ui, egui::Id::NULL.with("re_sketch_offset"), now, 110.0)
        });
        if let Some(v) = typed.filter(|v| *v != now) {
            *action = SketchAction::Offset(v);
        }
    }
    // 6. The level the scan is cut at.
    let typed = field_row(ui, "Level", LEVEL_TITLE, |ui| {
        ui.label(RichText::new("mm").italics());
        number_field(ui, egui::Id::NULL.with("re_sketch_level"), edit.level, 110.0)
    });
    if let Some(v) = typed.filter(|v| *v != edit.level) {
        *action = SketchAction::Level(v);
    }
    // 7. The picture under the sketch.
    picture_row(ui, edit);
    // 9. The fit commands.
    if !no_scan {
        let unfitted = edit.chains.len().saturating_sub(edit.fitted.len());
        let fitted = !edit.fitted.is_empty();
        ui.horizontal_wrapped(|ui| {
            if icon_button(ui, "fitAll", "Fit all chains", !edit.chains.is_empty() && unfitted > 0, "Cut every slice chain into lines and arcs, each with its slice points behind it") {
                edit.fit_all();
            }
            if icon_button(ui, "inferCorners", "Infer corners", fitted, &format!("Join neighbouring ends within {} mm, with a tangency where a line meets an arc", fmt_value(CORNER_GAP_MM))) {
                edit.infer_corners();
            }
            if icon_button(ui, "closeChain", "Close", fitted, "Bridge every gap left open along the fitted chains") {
                edit.close();
            }
        });
        // 10. The fit options, in a window of their own.
        if chip(ui, "Fit options…", "How fine the slice is cut, the fit tolerance, corners joined, lines laid along the axes — in a window of its own; a change there refits the chains") {
            let key = egui::Id::NULL.with("re_sketch_fit_options");
            let open = ui.ctx().data(|d| d.get_temp::<bool>(key)).unwrap_or(false);
            ui.ctx().data_mut(|d| d.insert_temp(key, !open));
        }
    }
    // 11. The tools.
    tool_row(ui, edit);
    // 12. The polygon's sides, while the tool is in hand.
    if edit.tool == Tool::Polygon {
        ui.horizontal(|ui| {
            ui.label(RichText::new("Sides").size(12.0)).on_hover_text("How many sides the polygon has — three and up");
            let mut n = edit.prefs.polygon_sides as f64;
            if ui.add(egui::DragValue::new(&mut n).range(3.0..=POLYGON_MAX_SIDES as f64).speed(0.1).max_decimals(0)).on_hover_text("How many sides the polygon has — three and up").changed() {
                edit.prefs.polygon_sides = clamp_sides(n);
                set_prefs(edit.prefs);
            }
        });
    }
    // 14. The hint line.
    if let Some(h) = hint_line(edit) {
        let color = if edit.error.is_some() { t.amber } else { t.dim };
        ui.label(RichText::new(h).color(color).size(11.5));
    }
    // 15–16. The Dimension key and the relations.
    ui.horizontal(|ui| {
        ui.label(RichText::new("DIMENSION · RELATIONS").size(10.0).color(t.dim));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(RichText::new(format!("{} selected", edit.selected_entities().len())).strong().size(10.5));
        });
    });
    relation_row(ui, edit);
    // 17. The edits.
    edit_row(ui, edit);
    // 18. The Pattern box.
    pattern_box(ui, edit);
    // 19. The relations inferred while drawing.
    inferred_chips(ui, edit);
    // 20. The relations and dimensions.
    constraint_list(ui, edit);
    // 21. The solver's verdict.
    status_dro(ui, edit);
    // 22. Extrude and Revolve, as soon as a region closes.
    if !edit.regions.is_empty() {
        ui.horizontal(|ui| {
            if primary_icon(ui, "extrude", "Extrude", "Finish the sketch and extrude every region of it — the body shows as a ghost with arrows to drag; a click on a region lets it go") {
                *action = SketchAction::Extrude(None);
            }
            if primary_icon(ui, "revolve", "Revolve", "Finish the sketch and revolve every region of it about an axis picked next") {
                *action = SketchAction::Revolve(None);
            }
        });
    }
    // 23. Finish, at the bottom.
    if primary_block(ui, "Finish sketch", FINISH_TITLE).clicked() {
        *action = SketchAction::Finish;
    }
}

/// What the hint line says, when there is something to say (ScanRuler's order: the error,
/// the picture's clicks, the shape tool's next click, the Dimension tool, the corner tools,
/// the relation in hand).
pub fn hint_line(edit: &SketchEdit) -> Option<String> {
    if let Some(e) = &edit.error {
        return Some(e.clone());
    }
    if let Some(h) = edit.image_pick_hint() {
        return Some(h.into());
    }
    if edit.tool.is_shape() {
        let (ways, at) = draw_ways(edit, edit.tool);
        let clicks = ways.get(at)?.clicks;
        return Some(clicks[edit.pending.len().min(clicks.len() - 1)].into());
    }
    match edit.tool {
        Tool::Dimension => Some(if edit.dim_first().is_some() { "Click the second piece, end or coordinate plane." } else { "Click what to dimension: a line, a round piece, an end — or a coordinate plane’s line." }.into()),
        Tool::Fillet | Tool::Chamfer => Some(if edit.selected_entities().is_empty() { "Click a corner point, or the first of its two lines." } else { "Click the second line of the corner." }.into()),
        Tool::Relation(RelationKind::Symmetric) => Some(
            match edit.selected_entities().len() {
                0 => "Symmetric: click the first of the two pieces that mirror each other.",
                1 => "Click the piece that mirrors it.",
                _ => "Click the line they mirror about — a construction line, or a coordinate plane’s line on the sheet.",
            }
            .into(),
        ),
        Tool::Relation(RelationKind::Coincident) => Some(if edit.rel_picks.is_empty() { "Coincident: click an end, a centre or a piece — then what it lies on: another end, a line, an arc, or a coordinate plane’s line (the amber lines)." } else { "Now the second: an end, a centre, a piece, or a coordinate plane’s line." }.into()),
        Tool::Relation(_) => Some(if edit.selected_entities().is_empty() { "Click the piece to put it on." } else { "Click the second piece." }.into()),
        _ => None,
    }
}

/// The tool row: the tools, Snap to grid and Show relations.
fn tool_row(ui: &mut egui::Ui, edit: &mut SketchEdit) {
    let gap = 6.0;
    let (cols, w) = key_columns(ui.available_width(), 36.0, gap);
    let size = egui::vec2(w, 34.0);
    let n = TOOLS.len() + 2;
    let mut i = 0;
    while i < n {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = gap;
            for _ in 0..cols {
                if i >= n {
                    break;
                }
                if i < TOOLS.len() {
                    let k = &TOOLS[i];
                    let on = edit.tool == k.tool;
                    if k.tool.is_shape() {
                        shape_key(ui, edit, k, size);
                    } else if icon_key(ui, size, k.icon, 22.0, on, true, false, &format!("{} — {}", k.label, k.title)).clicked() {
                        edit.press_tool(k.tool);
                    }
                } else if i == TOOLS.len() {
                    let on = edit.prefs.grid_snap;
                    if icon_key(ui, size, "gridSnap", 22.0, on, true, false, "Snap to grid — drawn and dragged points snap to the grid as well as to ends, the slice and pieces — off, the grid is only there to read against").clicked() {
                        edit.prefs.grid_snap = !on;
                        set_prefs(edit.prefs);
                    }
                } else {
                    let on = edit.prefs.show_relations;
                    if icon_key(ui, size, if on { "eye" } else { "eyeOff" }, 22.0, on, true, false, "Show relations — draw every relation beside its piece as a glyph. Off, only the one selected in the list below is drawn; the dimensions are always shown").clicked() {
                        edit.prefs.show_relations = !on;
                        set_prefs(edit.prefs);
                    }
                }
                i += 1;
            }
        });
    }
}

/// A shape tool's key, as CAD has it: the picture of the way it draws now, and — while the
/// cursor rests on it 250 ms, or when it is pressed again while in hand — a strip of the ways
/// under it, closing 180 ms after the pointer leaves.
fn shape_key(ui: &mut egui::Ui, edit: &mut SketchEdit, k: &ToolKey, size: egui::Vec2) {
    let (ways, chosen) = draw_ways(edit, k.tool);
    let Some(way) = ways.get(chosen) else { return };
    let in_hand = edit.tool == k.tool;
    let id = egui::Id::NULL.with(("re_sketch_flyout", k.label));
    #[derive(Clone, Copy, Default)]
    struct Fly {
        open: bool,
        rest: Option<f64>,
        left: Option<f64>,
        strip: Option<egui::Rect>,
    }
    let ctx = ui.ctx().clone();
    let now = ui.input(|i| i.time);
    let mut fly: Fly = ctx.data(|d| d.get_temp(id)).unwrap_or_default();
    // The key's own tooltip stands down while its ways are open under it.
    let tip = if fly.open { String::new() } else { format!("{} — {}", k.label, k.title) };
    let resp = icon_key(ui, size, way.icon, 22.0, in_hand, true, true, &tip);
    let pointer = ui.input(|i| i.pointer.hover_pos());
    let over_strip = fly.strip.is_some_and(|r| pointer.is_some_and(|p| r.expand(6.0).contains(p)));
    if resp.hovered() || over_strip {
        fly.left = None;
        if !fly.open {
            let since = *fly.rest.get_or_insert(now);
            if now - since >= 0.25 {
                fly.open = true;
            } else {
                ctx.request_repaint_after(std::time::Duration::from_millis(60));
            }
        }
    } else {
        fly.rest = None;
        if fly.open {
            let since = *fly.left.get_or_insert(now);
            if now - since >= 0.18 {
                fly.open = false;
                fly.left = None;
            } else {
                ctx.request_repaint_after(std::time::Duration::from_millis(40));
            }
        }
    }
    if resp.clicked() {
        // On a touch screen there is no resting on a key: pressed again while in hand, it
        // opens its ways.
        if in_hand {
            fly.open = true;
        }
        edit.press_tool(k.tool);
    }
    fly.strip = None;
    if fly.open {
        let t = tokens(ui);
        let width = ways.len() as f32 * 44.0 + (ways.len() as f32 - 1.0) * 4.0 + 8.0;
        let at = egui::pos2(resp.rect.center().x - width / 2.0, resp.rect.bottom() + 4.0);
        let shown = egui::Area::new(id.with("strip")).fixed_pos(at).order(egui::Order::Foreground).show(&ctx, |ui| {
            egui::Frame::popup(ui.style()).inner_margin(4.0).show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 4.0;
                    for (j, w) in ways.iter().enumerate() {
                        if icon_key(ui, egui::vec2(44.0, 34.0), w.icon, 22.0, j == chosen, true, false, &format!("{} — {}", w.label, w.title)).clicked() {
                            choose_way(edit, k.tool, j);
                            fly.open = false;
                        }
                    }
                });
            });
        });
        let _ = t;
        fly.strip = Some(shown.response.rect);
    }
    ctx.data_mut(|d| d.insert_temp(id, fly));
}

/// The Dimension key and the nine relations after it, ten to a row.
fn relation_row(ui: &mut egui::Ui, edit: &mut SketchEdit) {
    let gap = 2.0;
    let w = (ui.available_width() - gap * 9.0) / 10.0;
    let size = egui::vec2(w.max(20.0), 30.0);
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = gap;
        let has_selection = !edit.selected_entities().is_empty();
        if icon_key(ui, size, "dimDistance", 20.0, edit.tool == Tool::Dimension, true, false, DIMENSION_TITLE).clicked() {
            if has_selection {
                edit.dimension_selection();
            } else {
                edit.set_tool(if edit.tool == Tool::Dimension { Tool::Select } else { Tool::Dimension });
            }
        }
        for (kind, icon) in RelationKind::ALL.into_iter().zip(RELATION_ICONS) {
            let armed = edit.tool == Tool::Relation(kind);
            if icon_key(ui, size, icon, 20.0, armed, true, false, kind.label()).clicked() {
                edit.press_relation(kind);
            }
        }
    });
}

/// The edits: Delete, Sharp corner, Fix, Construction, Pattern and Align.
fn edit_row(ui: &mut egui::Ui, edit: &mut SketchEdit) {
    let gap = 6.0;
    let (_, w) = key_columns(ui.available_width(), 36.0, gap);
    let size = egui::vec2(w, 34.0);
    let ids = edit.selected_entities();
    let has = !ids.is_empty();
    let one_line = ids.len() == 1 && edit.sketch.entity(ids[0]).is_some_and(|e| e.kind() == ShapeKind::Line);
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing.x = gap;
        if icon_key(ui, size, "del", 22.0, false, has, false, "Delete the selected pieces").clicked() {
            edit.delete_selection();
        }
        if icon_key(ui, size, "sharpCorner", 22.0, false, has, false, "Sharp corner — Remove the selected fillet arc or chamfer line and carry its two lines to the corner").clicked() {
            edit.sharpen();
        }
        if icon_key(ui, size, "fix", 22.0, false, has, false, "Fix — Lock the selected pieces where they are").clicked() {
            edit.toggle_fix();
        }
        if icon_key(ui, size, "construction", 22.0, false, has, false, "Construction — Construction geometry takes no part in the regions — an axis to revolve about, a mirror line").clicked() {
            edit.toggle_construction();
        }
        if icon_key(ui, size, "sketchPattern", 22.0, edit.pattern.is_some(), has, false, "Pattern — Repeat the selected pieces in a row, a grid or round a centre. Each copy is a piece of its own and keeps the relations and dimensions the pieces have among themselves").clicked() {
            if edit.pattern.is_some() {
                edit.pattern = None;
            } else {
                edit.open_pattern();
            }
        }
        if icon_key(ui, size, "alignSheet", 22.0, false, one_line, false, "Align — Turn the sheet so the selected line lies along its X axis — a crooked edge of the scan becomes the horizontal the rest is drawn against. Nothing moves on the part.").clicked() {
            edit.align_to_selected();
        }
    });
}

/// The relations inferred while drawing, one dashed pill each: a click selects it, × takes it
/// off.
fn inferred_chips(ui: &mut egui::Ui, edit: &mut SketchEdit) {
    let inferred = edit.inferred();
    if inferred.is_empty() {
        return;
    }
    let t = tokens(ui);
    let chosen = edit.selected_constraint();
    let mut remove = None;
    let mut select = None;
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing = egui::vec2(4.0, 4.0);
        for id in inferred {
            let Some(c) = edit.sketch.constraint(id) else { continue };
            let label = constraint_label(&c.rule, &edit.sketch);
            let unmet = edit.report.unmet.iter().any(|u| u.0 == id);
            let sel = chosen == Some(id);
            let color = if unmet {
                t.amber
            } else if sel {
                t.ink
            } else {
                t.dim
            };
            let border = if unmet {
                t.amber
            } else if sel {
                t.accent
            } else {
                t.line
            };
            let font = egui::FontId::proportional(11.0);
            let galley = ui.painter().layout_no_wrap(label, font, color);
            let size = galley.size() + egui::vec2(8.0 + 3.0 + 14.0, 2.0);
            let (rect, resp) = ui.allocate_exact_size(size, Sense::click());
            let painter = ui.painter();
            let outline = rect.shrink(0.5);
            if sel || unmet {
                painter.rect_stroke(outline, 10.0, Stroke::new(1.0, border), egui::StrokeKind::Inside);
            } else {
                let pts = vec![outline.left_top(), outline.right_top(), outline.right_bottom(), outline.left_bottom(), outline.left_top()];
                painter.extend(egui::Shape::dashed_line(&pts, Stroke::new(1.0, border), 3.0, 2.0));
            }
            painter.galley(rect.left_center() + egui::vec2(8.0, -galley.size().y / 2.0), galley, color);
            let x = egui::Rect::from_min_max(egui::pos2(rect.right() - 14.0, rect.top()), rect.right_bottom());
            painter.text(x.center(), egui::Align2::CENTER_CENTER, "×", egui::FontId::proportional(12.0), color);
            let over_x = ui.input(|i| i.pointer.hover_pos()).is_some_and(|p| x.contains(p));
            if resp.clicked() {
                if over_x {
                    remove = Some(id);
                } else {
                    select = Some(id);
                }
            }
            resp.on_hover_text(if over_x { "Take the inferred relation off" } else { "Inferred while drawing — the piece was drawn nearly so. Click to select it, × to take it off." });
        }
    });
    if let Some(id) = select {
        edit.select_constraint(id);
    }
    if let Some(id) = remove {
        edit.remove_constraint(id);
    }
}

/// The list of dimensions, relations and Fixes.
fn constraint_list(ui: &mut egui::Ui, edit: &mut SketchEdit) {
    let dims: Vec<ConstraintId> = edit.sketch.constraints.iter().filter(|c| c.rule.is_dimension()).map(|c| c.id).collect();
    let rels: Vec<ConstraintId> = edit.sketch.constraints.iter().filter(|c| !c.rule.is_dimension() && !c.inferred).map(|c| c.id).collect();
    let fixes: Vec<u32> = fixed_pieces(&edit.sketch).iter().map(|e| e.id).collect();
    if dims.is_empty() && rels.is_empty() && fixes.is_empty() {
        return;
    }
    let t = tokens(ui);
    let chosen = edit.selected_constraint();
    enum Do {
        Select(ConstraintId),
        Remove(ConstraintId),
        Value(ConstraintId, String),
        Flip(ConstraintId),
        Unfix(u32),
    }
    let mut todo: Vec<Do> = Vec::new();
    let mut hovered = None;
    egui::ScrollArea::vertical().id_salt("re_sketch_constraints").max_height(220.0).show(ui, |ui| {
        let row_bg = |ui: &mut egui::Ui, sel: bool| {
            if sel {
                let r = ui.max_rect();
                ui.painter().rect_filled(r, 2.0, t.accent_soft);
            }
        };
        for id in &dims {
            let Some(c) = edit.sketch.constraint(*id) else { continue };
            let rule = c.rule.clone();
            let label = constraint_label(&rule, &edit.sketch);
            let value = rule.value().unwrap_or(0.0);
            let unmet = edit.report.unmet.iter().any(|u| u.0 == *id);
            let r = ui.horizontal(|ui| {
                row_bg(ui, chosen == Some(*id));
                if ui.add(egui::Label::new(RichText::new(&label).size(12.0).color(t.dim)).sense(Sense::click())).clicked() {
                    todo.push(Do::Select(*id));
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.small_button("×").on_hover_text("Remove the dimension").clicked() {
                        todo.push(Do::Remove(*id));
                    }
                    if unmet {
                        ui.label(RichText::new("⚠").strong().color(t.amber));
                    }
                    ui.label(RichText::new(if matches!(rule, Rule::Angle { .. }) { "°" } else { "mm" }).italics());
                    if let Some(v) = number_field(ui, egui::Id::NULL.with(("re_sketch_dim", *id)), value, 64.0) {
                        todo.push(Do::Value(*id, fmt_value(v)));
                    }
                    let kind = match rule {
                        Rule::Radius { .. } => NominalKind::Radius,
                        Rule::Angle { .. } => NominalKind::Angle,
                        _ => NominalKind::Length,
                    };
                    if c.expr.is_none() {
                        if let Some(n) = nominal_of(value, kind) {
                            let text = if kind == NominalKind::Angle { format!("→ {n:.0}°") } else { format!("→ {} mm", if n % 1.0 == 0.0 { format!("{n:.0}") } else { format!("{n:.1}") }) };
                            if chip(ui, &text, "The round number this was probably designed at — click to take it") {
                                todo.push(Do::Value(*id, fmt_value(n)));
                            }
                        }
                    }
                    match rule {
                        Rule::Radius { .. } => {
                            if chip(ui, "→ Ø", "Give it by its diameter instead") {
                                todo.push(Do::Flip(*id));
                            }
                        }
                        Rule::Diameter { .. } => {
                            if chip(ui, "→ R", "Give it by its radius instead") {
                                todo.push(Do::Flip(*id));
                            }
                        }
                        _ => {}
                    }
                });
            });
            if r.response.hovered() {
                hovered = Some(*id);
            }
        }
        for id in &rels {
            let Some(c) = edit.sketch.constraint(*id) else { continue };
            let label = constraint_label(&c.rule, &edit.sketch);
            let unmet = edit.report.unmet.iter().any(|u| u.0 == *id);
            let r = ui.horizontal(|ui| {
                row_bg(ui, chosen == Some(*id));
                if ui.add(egui::Label::new(RichText::new(&label).size(12.0).color(t.dim)).sense(Sense::click())).clicked() {
                    todo.push(Do::Select(*id));
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.small_button("×").on_hover_text("Remove the relation").clicked() {
                        todo.push(Do::Remove(*id));
                    }
                    if unmet {
                        ui.label(RichText::new("⚠").strong().color(t.amber));
                    }
                });
            });
            if r.response.hovered() {
                hovered = Some(*id);
            }
        }
        for id in &fixes {
            ui.horizontal(|ui| {
                ui.label(RichText::new(format!("Fix {}", super::sketcher::entity_name(&edit.sketch, *id))).size(12.0).color(t.dim));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.small_button("×").on_hover_text("Take the Fix off — the piece is free to move again").clicked() {
                        todo.push(Do::Unfix(*id));
                    }
                });
            });
        }
    });
    if hovered.is_some() || matches!(edit.hover, Some(Pick::Constraint(_))) {
        edit.hover = hovered.map(Pick::Constraint);
    }
    for d in todo {
        match d {
            Do::Select(id) => edit.select_constraint(id),
            Do::Remove(id) => edit.remove_constraint(id),
            Do::Value(id, text) => {
                edit.set_dimension(id, &text);
            }
            Do::Flip(id) => edit.flip_dimension_form(id),
            Do::Unfix(id) => edit.unfix(id),
        }
    }
}

/// The solver's verdict in a window, with the counts above it.
fn status_dro(ui: &mut egui::Ui, edit: &SketchEdit) {
    let t = tokens(ui);
    let pieces = edit.sketch.entities.len();
    let regions = edit.regions.len();
    let loose = if edit.sketch.entities.is_empty() { 0 } else { edit.report.under_constrained.len() };
    ui.horizontal(|ui| {
        ui.label(RichText::new("SKETCH").size(10.0).color(t.dim));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if loose > 0 {
                ui.label(RichText::new(format!("· {loose} loose")).size(11.0).color(t.dim)).on_hover_text("Pieces a relation or dimension does not pin down yet — drawn in the loose colour on the sheet; the rest in the constrained colour");
            }
            ui.label(RichText::new(format!("{pieces} piece{} · {regions} region{}", plural(pieces), plural(regions))).size(11.0).color(t.dim));
        });
    });
    let (text, alarm, tip) = edit.status();
    let empty = edit.sketch.entities.is_empty();
    let fill = if alarm { t.alarm_bg } else { t.well };
    let line = if alarm { t.alarm_line } else { t.line };
    let r = egui::Frame::new().fill(fill).stroke(Stroke::new(1.0, line)).corner_radius(2.0).inner_margin(egui::Margin::symmetric(10, 5)).show(ui, |ui| {
        ui.set_width(ui.available_width());
        let rt = if empty {
            RichText::new(&text).size(12.0).color(t.dim)
        } else if alarm {
            RichText::new(&text).size(12.0).strong().color(t.red)
        } else {
            RichText::new(&text).size(13.0).strong()
        };
        ui.label(rt);
    });
    if !tip.is_empty() {
        r.response.on_hover_text(tip);
    }
}

// ---- script steps (headless tests) ----

impl super::ReverseWorkspace {
    /// The sketch's own script steps (see `run_script_step`). Returns whether it took the step.
    pub(crate) fn sketch_script_step(&mut self, cmd: &str, args: &[&str]) -> bool {
        let num = |i: usize, default: f64| args.get(i).and_then(|v| v.parse::<f64>().ok()).unwrap_or(default);
        let super::Mode::Sketch(e) = &mut self.mode else {
            return false;
        };
        let p = |i: usize| DVec2::new(num(i, 0.0), num(i + 1, 0.0));
        match cmd {
            // tool:<name>[:<way>]
            "tool" => {
                let tool = match args.first().copied() {
                    Some("select") => Tool::Select,
                    Some("line") => Tool::Line,
                    Some("rect") | Some("rectangle") => Tool::Rectangle,
                    Some("polygon") => Tool::Polygon,
                    Some("circle") => Tool::Circle,
                    Some("arc") => Tool::Arc,
                    Some("fillet") => Tool::Fillet,
                    Some("chamfer") => Tool::Chamfer,
                    Some("project") => Tool::Project,
                    Some("dimension") => Tool::Dimension,
                    _ => return false,
                };
                let way = match (tool, args.get(1).copied()) {
                    (Tool::Rectangle, Some("corners")) | (Tool::Circle, Some("center")) | (Tool::Arc, Some("threePoint")) | (Tool::Polygon, Some("inscribed")) => Some(0),
                    (Tool::Rectangle, Some("center")) | (Tool::Circle, Some("twoPoint")) | (Tool::Arc, Some("center")) | (Tool::Polygon, Some("circumscribed")) => Some(1),
                    (Tool::Rectangle, Some("threePoint")) | (Tool::Circle, Some("threePoint")) => Some(2),
                    _ => None,
                };
                match way {
                    Some(w) if tool.is_shape() => choose_way(e, tool, w),
                    _ => e.press_tool(tool),
                }
            }
            // click:<x>:<y>[:shift|ctrl] — a click on the sheet at a sheet point, mm.
            "click" => {
                let extra = args.get(2).copied();
                let at = p(0);
                // The Project tool's candidates, found as the stage finds them before a click.
                if let super::Mode::Sketch(mut edit) = std::mem::replace(&mut self.mode, super::Mode::Idle) {
                    self.refresh_projectables(&mut edit);
                    edit.sheet_click(at, extra == Some("shift"), extra == Some("ctrl"));
                    self.mode = super::Mode::Sketch(edit);
                }
            }
            "sides" => {
                e.prefs.polygon_sides = clamp_sides(num(0, 6.0));
                set_prefs(e.prefs);
            }
            "grid" => {
                e.prefs.grid_snap = args.first() == Some(&"on");
                set_prefs(e.prefs);
            }
            "relations" => {
                e.prefs.show_relations = args.first() == Some(&"on");
                set_prefs(e.prefs);
            }
            // select:<id>,<id>…
            "select" => {
                e.selection = args.first().map_or(Vec::new(), |s| s.split(',').filter_map(|x| x.parse().ok()).map(Pick::Entity).collect());
            }
            // rel:<kind> — the relation key pressed.
            "rel" => {
                let kind = RelationKind::ALL.into_iter().find(|k| k.label().eq_ignore_ascii_case(args.first().copied().unwrap_or("")));
                if let Some(kind) = kind {
                    e.press_relation(kind);
                }
            }
            // pattern:grid:<count>:<step>[:<count2>:<step2>] | pattern:round:<count>:<deg>:<cx>:<cy> | pattern:apply
            "pattern" => match args.first().copied() {
                Some("grid") => {
                    e.open_pattern();
                    let c2 = num(3, 1.0) as usize;
                    e.pattern = Some(SketchPatternSpec::Rectangular { step: DVec2::new(num(2, 10.0), 0.0), count: num(1, 3.0) as usize, step2: (c2 > 1).then(|| DVec2::new(0.0, num(4, 10.0))), count2: (c2 > 1).then_some(c2) });
                }
                Some("round") => {
                    e.open_pattern();
                    e.pattern = Some(SketchPatternSpec::Circular { centre: p(3), count: num(1, 6.0) as usize, angle_deg: num(2, 360.0) });
                }
                Some("apply") => e.apply_pattern(),
                _ => e.pattern = None,
            },
            "align" => e.align_to_selected(),
            // round:<piece>:<piece>[:<radius>] — a corner rounded where the scan is round (the
            // agent's sketch.edit rounds), its radius read off the slice points when none is given.
            "round" => {
                let (a, b) = (num(0, 0.0) as u32, num(1, 0.0) as u32);
                let radius = args.get(2).and_then(|v| v.parse::<f64>().ok());
                let before = e.sketch.clone();
                match mm_reverse::round_corner::round_corner(&mut e.sketch, a, b, radius) {
                    Ok(made) => {
                        let after = std::mem::replace(&mut e.sketch, before);
                        e.checkpoint();
                        e.sketch = after;
                        e.solve();
                        e.error = None;
                        e.message = format!("Rounded {} and {}: arc {} of radius {}.", a, b, made.arc, fmt_value(made.radius));
                    }
                    Err(why) => {
                        e.sketch = before;
                        e.error = Some(why);
                    }
                }
            }
            // shape:<json> — shapes as a sketch handed in as data carries them (one, or a list).
            "shape" => {
                let text = args.join(":");
                match serde_json::from_str::<serde_json::Value>(&text) {
                    Ok(v) => {
                        let shapes = match v {
                            serde_json::Value::Array(list) => list,
                            one => vec![one],
                        };
                        let before = e.sketch.clone();
                        let frame = e.frame;
                        match mm_reverse::sketch_draw::add_shapes(&mut e.sketch, &frame, &shapes) {
                            Ok(_) => {
                                let after = std::mem::replace(&mut e.sketch, before);
                                e.checkpoint();
                                e.sketch = after;
                                e.solve();
                                e.error = None;
                            }
                            Err(why) => {
                                e.sketch = before;
                                e.error = Some(why);
                            }
                        }
                    }
                    Err(err) => e.error = Some(format!("shape: {err}")),
                }
            }
            "level" => {
                let v = num(0, 0.0);
                if e.set_level(v) {
                    let frame = e.slice_frame();
                    let chains = self.slice(&frame);
                    if let super::Mode::Sketch(e) = &mut self.mode {
                        e.chains = chains;
                    }
                }
            }
            "picture" => {
                let path = std::path::PathBuf::from(args.join(":"));
                lay_picture(e, &path);
            }
            // picture-pick:scale|move|level then clicks; picture-scale:<length>
            "picture-pick" => e.start_image_pick(match args.first().copied() {
                Some("scale") => Some(ImagePickMode::Scale),
                Some("move") => Some(ImagePickMode::Move),
                Some("level") => Some(ImagePickMode::Level),
                _ => None,
            }),
            "picture-scale" => e.scale_image(num(0, 0.0)),
            "escape" => e.escape(),
            "sketchreport" => {
                let rules: Vec<String> = e.sketch.constraints.iter().map(|c| constraint_label(&c.rule, &e.sketch)).collect();
                let (status, _, _) = e.status();
                let image = e.image.as_ref().map(|i| format!("{}×{} at {:.3} mm/px, {:.1}°", i.width, i.height, i.mm_per_px, i.angle_deg)).unwrap_or_else(|| "none".into());
                log::info!(
                    "reverse sketch: {} pieces, {} regions, tool {:?}, selected {:?}, status {status}; picture {image}; error {:?}; relations [{}]",
                    e.sketch.entities.len(),
                    e.regions.len(),
                    e.tool,
                    e.selected_entities(),
                    e.error,
                    rules.join("; ")
                );
            }
            _ => return false,
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_round_trips() {
        for bytes in [&b""[..], b"a", b"ab", b"abc", b"abcd", &[0u8, 255, 128, 7, 9]] {
            let url = format!("data:image/png;base64,{}", base64(bytes));
            assert_eq!(data_url_bytes(&url).expect("bytes"), bytes);
        }
        assert_eq!(base64(b"Man"), "TWFu");
        assert_eq!(base64(b"Ma"), "TWE=");
    }

    #[test]
    fn key_columns_fill_the_row() {
        let (cols, w) = key_columns(300.0, 36.0, 6.0);
        assert_eq!(cols, 7);
        assert!(w >= 36.0 && (cols as f32 * w + 6.0 * (cols as f32 - 1.0) - 300.0).abs() < 1e-3);
    }
}
