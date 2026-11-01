// SPDX-License-Identifier: AGPL-3.0-only
//! The open sketch drawn on its sheet over the part: the millimetre grid, the picture laid
//! under it, the slice, the closed regions filled faintly, the pieces in their colours with a
//! dot on every end, the coordinate planes' lines, the drawing tool's ghost rubber-banded to
//! the cursor with a mark where the click would snap, the Pattern box's copies, the Project
//! tool's candidate, the relation glyphs and a selection box. Ported from ScanRuler
//! (`plugins/reverse/app/useSketchSceneSync.ts`, `viewer/sketchPalette.ts`), © Stefan
//! Hermann (CNC Kitchen), AGPL-3.0-only — painted through the view's projection rather than on
//! a sheet of its own.

use std::cell::RefCell;
use std::collections::HashMap;

use egui::{Color32, Pos2, Shape as EShape, Stroke};
use glam::DVec2;
use mm_reverse::draw::{arc_about, circle_from, rectangle_corners, ArcMode, CircleMode, DrawModes, RectangleMode};
use mm_reverse::model::SketchImage;
use mm_reverse::sketch::{anchor_point, anchors_of, arc_through, constraint_entities, polygon_corners, polyline, Anchor, DistanceTo, MirrorAxis, OnTarget, Rule, Shape, Sketch};
use mm_reverse::sketch_image::image_corners;
use mm_reverse::sketch_shapes::{fixed_pieces, PolygonFit};
use mm_reverse::snap::SnapKind;

use super::sketcher::{grid_spacing, Pick, SketchEdit, Tool};

/// The colours a sketch is drawn in, over the part cut open at its plane (ScanRuler's
/// `SketchPalette`): the pieces saturated and at full strength, the slice a thin muted line.
#[derive(Debug, Clone, Copy)]
pub struct Palette {
    pub slice: Color32,
    pub fitted: Color32,
    /// A free piece still loose (under-constrained), and one fully constrained.
    pub free: Color32,
    pub constrained: Color32,
    pub projected: Color32,
    pub construction: Color32,
    pub selected: Color32,
    /// What a click would take, under the cursor.
    pub hover: Color32,
    pub datum: Color32,
    pub region: Color32,
    pub dot: Color32,
    pub dimension: Color32,
}

const fn hex(c: u32) -> Color32 {
    Color32::from_rgb((c >> 16) as u8, (c >> 8) as u8, c as u8)
}

pub const DARK: Palette = Palette {
    slice: hex(0xaeb8c4),
    fitted: hex(0x4db8ff),
    free: hex(0xc48bff),
    constrained: hex(0xf2f5f8),
    projected: hex(0xff7ac8),
    construction: hex(0x9aa5b1),
    selected: hex(0xffb020),
    hover: hex(0xffe27a),
    datum: hex(0xe8a33d),
    region: hex(0x3cb371),
    dot: hex(0xffffff),
    dimension: hex(0xe8edf2),
};

pub const LIGHT: Palette = Palette {
    slice: hex(0x4a5560),
    fitted: hex(0x0a63c9),
    free: hex(0x7a3cc9),
    constrained: hex(0x14181d),
    projected: hex(0xc2268a),
    construction: hex(0x5d6875),
    selected: hex(0xe07a00),
    hover: hex(0xff9d14),
    datum: hex(0xa8660a),
    region: hex(0x1f8a4c),
    dot: hex(0x14181d),
    dimension: hex(0x14181d),
};

pub fn palette(dark: bool) -> Palette {
    if dark {
        DARK
    } else {
        LIGHT
    }
}

/// Line weights of the sketch in points (ScanRuler's `SKETCH_LINE`).
pub mod line {
    pub const PIECE: f32 = 2.5;
    pub const SELECTED: f32 = 3.5;
    pub const HOVER: f32 = 5.0;
    pub const CONSTRUCTION: f32 = 1.5;
    pub const DATUM: f32 = 1.2;
}

/// The dots, points across (ScanRuler's `SKETCH_DOT`).
pub mod dot {
    pub const END: f32 = 4.0;
    pub const PICKED: f32 = 8.0;
    pub const HOVER: f32 = 10.0;
    pub const ORIGIN: f32 = 6.0;
}

/// A selection box's colours: a window solid green, a crossing box dashed blue.
const WINDOW: Color32 = hex(0x3cb371);
const CROSSING: Color32 = hex(0x4db8ff);

/// What a drawing tool is about to make, rubber-banded from the points it has to the cursor
/// in the way it is drawn (ScanRuler's `toolPreview`): a line from its first end, the whole
/// rectangle from its first corner or its centre (and its diagonals), a turned rectangle's first
/// side and then the whole of it, the polygon and the circle it is drawn about, the circle
/// about its centre or through its points, an arc's chord or radius and then the arc.
pub fn tool_preview(tool: Tool, pending: &[DVec2], cursor: DVec2, modes: &DrawModes, sides: usize) -> Vec<Shape> {
    let p = pending;
    let line = |a: DVec2, b: DVec2| -> Vec<Shape> {
        if (b - a).length() > 1e-12 {
            vec![Shape::Line { a, b }]
        } else {
            Vec::new()
        }
    };
    let ring = |corners: Option<Vec<DVec2>>| -> Vec<Shape> {
        corners.map_or(Vec::new(), |c| (0..c.len()).flat_map(|i| line(c[i], c[(i + 1) % c.len()])).collect())
    };
    match tool {
        Tool::Line if p.len() == 1 => line(p[0], cursor),
        Tool::Rectangle => {
            if modes.rectangle == RectangleMode::ThreePoint {
                return match p.len() {
                    1 => line(p[0], cursor),
                    2 => match rectangle_corners(RectangleMode::ThreePoint, &[p[0], p[1], cursor]) {
                        Some(c) => ring(Some(c.to_vec())),
                        None => line(p[0], p[1]),
                    },
                    _ => Vec::new(),
                };
            }
            if p.len() != 1 {
                return Vec::new();
            }
            let corners = rectangle_corners(modes.rectangle, &[p[0], cursor]);
            let mut out = ring(corners.map(|c| c.to_vec()));
            // About its centre, the diagonals it carries are drawn too.
            if let (Some(c), RectangleMode::Center) = (corners, modes.rectangle) {
                out.extend(line(c[0], c[2]));
                out.extend(line(c[1], c[3]));
            }
            out
        }
        Tool::Polygon if p.len() == 1 => {
            let mut out = ring(polygon_corners(p[0], cursor, sides, modes.polygon == PolygonFit::Circumscribed));
            out.push(Shape::Circle { c: p[0], r: (cursor - p[0]).length() });
            out
        }
        Tool::Circle => {
            if modes.circle == CircleMode::ThreePoint {
                return match p.len() {
                    1 => line(p[0], cursor),
                    2 => match circle_from(CircleMode::ThreePoint, &[p[0], p[1], cursor]) {
                        Some((c, r)) => vec![Shape::Circle { c, r }],
                        None => line(p[0], p[1]),
                    },
                    _ => Vec::new(),
                };
            }
            if p.len() != 1 {
                return Vec::new();
            }
            let mut out: Vec<Shape> = circle_from(modes.circle, &[p[0], cursor]).map(|(c, r)| Shape::Circle { c, r }).into_iter().collect();
            if modes.circle == CircleMode::TwoPoint {
                out.extend(line(p[0], cursor));
            }
            out
        }
        Tool::Arc => {
            if modes.arc == ArcMode::Center {
                // The radius to the start, then the arc from it round to the cursor.
                return match p.len() {
                    1 => line(p[0], cursor),
                    2 => {
                        let mut out = line(p[0], p[1]);
                        out.extend(arc_about(p[0], p[1], cursor).map(|a| a.shape()));
                        out
                    }
                    _ => Vec::new(),
                };
            }
            match p.len() {
                1 => line(p[0], cursor),
                2 => match arc_through(p[0], p[1], cursor) {
                    Some((c, r, start, sweep, _)) => vec![Shape::Arc { c, r, start, sweep }],
                    None => line(p[0], p[1]),
                },
                _ => Vec::new(),
            }
        }
        _ => Vec::new(),
    }
}

/// The mark drawn where the cursor will land (ScanRuler's `snapMark`): a cross on an end, an
/// upright cross on the origin, a triangle on the middle of a line, a diamond on a slice
/// point, a tick on a piece or a plane's line, a small square on the grid.
pub fn snap_mark(kind: SnapKind, at: DVec2, s: f64) -> Vec<(DVec2, DVec2)> {
    let (x, y) = (at.x, at.y);
    let v = DVec2::new;
    match kind {
        SnapKind::Anchor => vec![(v(x - s, y - s), v(x + s, y + s)), (v(x - s, y + s), v(x + s, y - s))],
        SnapKind::Origin => vec![(v(x - s * 1.4, y), v(x + s * 1.4, y)), (v(x, y - s * 1.4), v(x, y + s * 1.4))],
        SnapKind::Midpoint => vec![(v(x - s, y - s * 0.7), v(x + s, y - s * 0.7)), (v(x + s, y - s * 0.7), v(x, y + s)), (v(x, y + s), v(x - s, y - s * 0.7))],
        SnapKind::Edge => vec![(v(x - s, y), v(x, y + s)), (v(x, y + s), v(x + s, y)), (v(x + s, y), v(x, y - s)), (v(x, y - s), v(x - s, y))],
        SnapKind::Entity | SnapKind::Datum => vec![(v(x - s, y), v(x + s, y))],
        SnapKind::Grid => vec![(v(x - s, y - s), v(x + s, y - s)), (v(x + s, y - s), v(x + s, y + s)), (v(x + s, y + s), v(x - s, y + s)), (v(x - s, y + s), v(x - s, y - s))],
    }
}

/// A relation's glyph (ScanRuler's `GLYPHS`), and its name under the cursor.
pub fn glyph(rule: &Rule) -> Option<(&'static str, &'static str)> {
    Some(match rule {
        Rule::Coincident { .. } => ("•", "Coincident"),
        Rule::Horizontal { .. } => ("H", "Horizontal"),
        Rule::Vertical { .. } => ("V", "Vertical"),
        Rule::Parallel { .. } => ("∥", "Parallel"),
        Rule::Perpendicular { .. } => ("⊥", "Perpendicular"),
        Rule::Tangent { .. } => ("◠", "Tangent"),
        Rule::Concentric { .. } => ("◎", "Concentric"),
        Rule::Equal { .. } => ("=", "Equal"),
        Rule::Symmetric { .. } => ("⋈", "Symmetric"),
        Rule::PointOn { .. } => ("•", "Point on"),
        Rule::Collinear { .. } => ("≡", "Collinear"),
        _ => return None,
    })
}

/// The middle of a piece, where its glyph sits: a line's middle, an arc's, a circle's top.
fn piece_middle(shape: &Shape) -> DVec2 {
    match *shape {
        Shape::Line { a, b } => (a + b) / 2.0,
        Shape::Arc { c, r, start, sweep } => mm_reverse::sketch::arc_point(c, r, start + sweep / 2.0),
        Shape::Circle { c, r } => c + DVec2::new(0.0, r),
        Shape::Point { p } => p,
        Shape::Spline(_) => along(shape, 0.5),
    }
}

/// Where a relation's glyph sits (ScanRuler's `glyphSpot`).
pub fn glyph_spot(rule: &Rule, sketch: &Sketch) -> Option<DVec2> {
    let mid = |id| sketch.entity(id).map(|e| piece_middle(&e.shape));
    match *rule {
        Rule::Coincident { p, .. } | Rule::PointOn { p, .. } => sketch.anchor(p),
        Rule::Collinear { line, .. } => mid(line),
        Rule::Symmetric { p, q, .. } => Some((sketch.anchor(p)? + sketch.anchor(q)?) / 2.0),
        Rule::Horizontal { entity } | Rule::Vertical { entity } => mid(entity),
        Rule::Tangent { a, b } => {
            // At the point the two touch: the coincidence they share, if any.
            let shared = sketch.constraints.iter().find_map(|x| match x.rule {
                Rule::Coincident { p, q } if [p.entity, q.entity].contains(&a) && [p.entity, q.entity].contains(&b) => Some(p),
                _ => None,
            });
            shared.and_then(|p| sketch.anchor(p)).or_else(|| mid(a))
        }
        Rule::Parallel { a, .. } | Rule::Perpendicular { a, .. } | Rule::Concentric { a, .. } | Rule::Equal { a, .. } => mid(a),
        _ => None,
    }
}

/// Where a Fix's glyph sits: a third of the way along a line, at a round piece's centre, on a
/// point.
fn fix_spot(shape: &Shape) -> DVec2 {
    match *shape {
        Shape::Line { a, b } => a + (b - a) / 3.0,
        Shape::Arc { c, .. } | Shape::Circle { c, .. } => c,
        Shape::Point { p } => p,
        Shape::Spline(_) => along(shape, 1.0 / 3.0),
    }
}

/// The point a share of the way along a piece's polyline.
fn along(shape: &Shape, share: f64) -> DVec2 {
    let pts = mm_reverse::sketch::polyline(shape, 96);
    pts.get(((pts.len().saturating_sub(1)) as f64 * share).round() as usize).copied().unwrap_or(DVec2::ZERO)
}

/// The ends a relation or a dimension holds — lit as points while it is pointed at.
fn constraint_anchors(rule: &Rule) -> Vec<Anchor> {
    match *rule {
        Rule::Coincident { p, q } | Rule::Symmetric { p, q, .. } => vec![p, q],
        Rule::Distance { p, q: DistanceTo::Anchor(q), .. } => vec![p, q],
        Rule::Distance { p, .. } | Rule::PointOn { p, .. } => vec![p],
        _ => Vec::new(),
    }
}

/// The coordinate plane a relation or a dimension holds to, if one.
fn constraint_datum(rule: &Rule) -> Option<mm_reverse::datums::DatumPlane> {
    match *rule {
        Rule::Distance { q: DistanceTo::Datum(d), .. } => Some(d),
        Rule::PointOn { on: OnTarget::Datum(d), .. } | Rule::Collinear { on: OnTarget::Datum(d), .. } => Some(d),
        Rule::Symmetric { about: MirrorAxis::Datum(d), .. } => Some(d),
        _ => None,
    }
}

thread_local! {
    /// The pictures under sketches, as textures, by their source.
    static PICTURES: RefCell<HashMap<u64, Option<egui::TextureHandle>>> = RefCell::new(HashMap::new());
}

fn source_key(src: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    src.hash(&mut h);
    h.finish()
}

/// A picture's texture, read from its data URL once.
fn picture_texture(ctx: &egui::Context, image: &SketchImage) -> Option<egui::TextureHandle> {
    let key = source_key(&image.src);
    PICTURES.with(|cache| {
        let mut cache = cache.borrow_mut();
        cache
            .entry(key)
            .or_insert_with(|| {
                let bytes = super::sketch_box::data_url_bytes(&image.src)?;
                let decoded = image::load_from_memory(&bytes).ok()?.to_rgba8();
                let size = [decoded.width() as usize, decoded.height() as usize];
                let color = egui::ColorImage::from_rgba_unmultiplied(size, decoded.as_raw());
                Some(ctx.load_texture(format!("re_sketch_picture_{key}"), color, egui::TextureOptions::LINEAR))
            })
            .clone()
    })
}

/// Paint the open sketch through `to_screen` (a sheet point to the screen, if in front);
/// `unproject` lays a screen point on the sheet, for the grid's extent.
pub fn paint_sheet(edit: &SketchEdit, painter: &egui::Painter, to_screen: &dyn Fn(DVec2) -> Option<Pos2>, unproject: &dyn Fn(Pos2) -> Option<DVec2>, show_slice: bool) {
    let dark = painter.ctx().global_style().visuals.dark_mode;
    let pal = palette(dark);
    let path = |pts: &[DVec2]| -> Vec<Pos2> { pts.iter().filter_map(|p| to_screen(*p)).collect() };
    let stroke_shape = |shape: &Shape, width: f32, color: Color32| {
        let pts = path(&polyline(shape, 96));
        match shape {
            Shape::Point { .. } => {
                if let Some(p) = pts.first() {
                    painter.circle_filled(*p, width.max(2.0), color);
                }
            }
            _ => {
                if pts.len() >= 2 {
                    painter.add(EShape::line(pts, Stroke::new(width, color)));
                }
            }
        }
    };
    let clip = painter.clip_rect();
    let view: Option<[DVec2; 4]> = (|| Some([unproject(clip.left_top())?, unproject(clip.right_top())?, unproject(clip.right_bottom())?, unproject(clip.left_bottom())?]))();
    // The scale at the middle of the view: what the grid's spacing follows.
    let scale = view.and_then(|v| {
        let c = (v[0] + v[2]) / 2.0;
        let (a, b) = (to_screen(c)?, to_screen(c + DVec2::X)?);
        let d = (b - a).length() as f64;
        (d > 1e-6).then(|| 1.0 / d)
    });

    // The picture under the sketch.
    if let Some(image) = &edit.image {
        if let Some(tex) = picture_texture(painter.ctx(), image) {
            let c = image_corners(image);
            if let (Some(bl), Some(br), Some(tr), Some(tl)) = (to_screen(c[0]), to_screen(c[1]), to_screen(c[2]), to_screen(c[3])) {
                let tint = Color32::from_white_alpha((image.opacity.clamp(0.0, 1.0) * 255.0) as u8);
                let mut mesh = egui::Mesh::with_texture(tex.id());
                mesh.vertices.push(egui::epaint::Vertex { pos: bl, uv: egui::pos2(0.0, 1.0), color: tint });
                mesh.vertices.push(egui::epaint::Vertex { pos: br, uv: egui::pos2(1.0, 1.0), color: tint });
                mesh.vertices.push(egui::epaint::Vertex { pos: tr, uv: egui::pos2(1.0, 0.0), color: tint });
                mesh.vertices.push(egui::epaint::Vertex { pos: tl, uv: egui::pos2(0.0, 0.0), color: tint });
                mesh.indices.extend([0, 1, 2, 0, 2, 3]);
                painter.add(EShape::mesh(mesh));
            }
        }
    }

    // A millimetre grid on the sheet's own frame, anchored on the origin, its spacing
    // following the zoom; the two axes in the tool amber.
    if let (Some(v), Some(scale)) = (view, scale) {
        let s = grid_spacing(scale);
        let lo = v.iter().fold(DVec2::splat(f64::INFINITY), |m, p| m.min(*p));
        let hi = v.iter().fold(DVec2::splat(f64::NEG_INFINITY), |m, p| m.max(*p));
        let lines = ((hi.x - lo.x) / s) + ((hi.y - lo.y) / s);
        if lines.is_finite() && lines < 800.0 {
            let minor = Stroke::new(1.0, hex(0x8891a0).gamma_multiply(0.3));
            let axes = Stroke::new(1.0, hex(0xe8a33d).gamma_multiply(0.85));
            let mut u = (lo.x / s).ceil() * s;
            while u <= hi.x {
                if let (Some(a), Some(b)) = (to_screen(DVec2::new(u, lo.y)), to_screen(DVec2::new(u, hi.y))) {
                    painter.line_segment([a, b], if u.abs() < s / 2.0 { axes } else { minor });
                }
                u += s;
            }
            let mut w = (lo.y / s).ceil() * s;
            while w <= hi.y {
                if let (Some(a), Some(b)) = (to_screen(DVec2::new(lo.x, w)), to_screen(DVec2::new(hi.x, w))) {
                    painter.line_segment([a, b], if w.abs() < s / 2.0 { axes } else { minor });
                }
                w += s;
            }
        }
    }

    // Every closed region as a translucent face, its holes cut out.
    let fill = pal.region.gamma_multiply(0.16);
    for r in &edit.regions {
        let tol = (scale.unwrap_or(0.05) * 0.5).max(1e-3);
        let mut mesh = egui::Mesh::default();
        for t in r.region.triangulate(tol) {
            let pts: Vec<Pos2> = t.iter().filter_map(|q| to_screen(DVec2::new(q.x, q.y))).collect();
            if pts.len() == 3 {
                let base = mesh.vertices.len() as u32;
                for p in pts {
                    mesh.colored_vertex(p, fill);
                }
                mesh.add_triangle(base, base + 1, base + 2);
            }
        }
        painter.add(EShape::mesh(mesh));
    }

    // The slice chains, hairline and half see-through, with a dot on every end.
    if show_slice {
        let slice = pal.slice.gamma_multiply(0.5);
        for c in &edit.chains {
            let pts = path(c);
            if pts.len() >= 2 {
                painter.add(EShape::line(pts.clone(), Stroke::new(1.0, slice)));
            }
            for end in [c.first(), c.last()].into_iter().flatten() {
                if let Some(p) = to_screen(*end) {
                    painter.circle_filled(p, 1.5, slice);
                }
            }
        }
    }

    // What is lit: what a click would take, what a selection box would take, what the relation
    // or dimension pointed at in the list holds.
    let sketch = &edit.sketch;
    let pointed = match edit.hover {
        Some(Pick::Constraint(id)) => sketch.constraint(id).map(|c| c.rule.clone()),
        _ => None,
    };
    let mut lit: Vec<u32> = edit.lit.pieces.clone();
    if let Some(b) = edit.boxing.as_ref().filter(|b| b.moved) {
        lit.extend(&b.ids);
    }
    if let Some(r) = &pointed {
        lit.extend(constraint_entities(r));
    }
    let mut lit_points: Vec<DVec2> = edit.lit.point.into_iter().collect();
    if let Some(r) = &pointed {
        lit_points.extend(constraint_anchors(r).into_iter().filter_map(|a| sketch.anchor(a)));
    }
    let lit_datums: Vec<_> = edit.lit.datum.into_iter().chain(pointed.as_ref().and_then(constraint_datum)).collect();
    // The Dimension tool's first pick, and Coincident's, are lit like a selection.
    let mut selected: Vec<u32> = edit.selected_entities();
    let mut picked_points: Vec<DVec2> = Vec::new();
    let mut picked_datums = Vec::new();
    match edit.dim_first() {
        Some(Pick::Entity(id)) => selected.push(id),
        Some(Pick::Anchor(a)) => picked_points.extend(sketch.anchor(a)),
        _ => {}
    }
    for p in &edit.rel_picks {
        match *p {
            mm_reverse::draw::DimPick::Piece(id) => selected.push(id),
            mm_reverse::draw::DimPick::Point(a) => picked_points.extend(sketch.anchor(a)),
            mm_reverse::draw::DimPick::Datum(d) => picked_datums.push(d),
        }
    }

    // The part's coordinate planes, as the lines they cut the sheet in, drawn about the
    // sketch and named.
    let around = sketch.bounds();
    let middle = around.map_or(DVec2::ZERO, |(lo, hi)| (lo + hi) / 2.0);
    let reach = around.map_or(100.0, |(lo, hi)| (2.0 * (hi - lo).length()).max(100.0));
    for t in &edit.traces {
        let (a, b) = t.segment(middle, reach);
        let over = lit_datums.contains(&t.plane);
        let on = picked_datums.contains(&t.plane);
        let (color, width) = if over {
            (pal.hover, line::HOVER)
        } else if on {
            (pal.selected, line::SELECTED)
        } else {
            (pal.datum, line::DATUM)
        };
        if let (Some(pa), Some(pb)) = (to_screen(a), to_screen(b)) {
            painter.line_segment([pa, pb], Stroke::new(width, color));
            painter.text(pb, egui::Align2::RIGHT_BOTTOM, t.plane.name(), egui::FontId::proportional(11.0), color);
        }
    }

    // The pieces, in their colours: hover over selected over construction over projected over
    // fitted over loose over constrained.
    for e in &sketch.entities {
        let (color, width) = if lit.contains(&e.id) {
            (pal.hover, line::HOVER)
        } else if selected.contains(&e.id) {
            (pal.selected, line::SELECTED)
        } else if e.construction {
            (pal.construction, line::CONSTRUCTION)
        } else if e.projected {
            (pal.projected, line::PIECE)
        } else if e.is_fitted() {
            (pal.fitted, line::PIECE)
        } else if edit.report.under_constrained.contains(&e.id) {
            (pal.free, line::PIECE)
        } else {
            (pal.constrained, line::PIECE)
        };
        stroke_shape(&e.shape, width, color);
        if let Some(c) = e.shape.centre().filter(|_| e.shape.is_round()) {
            if let Some(p) = to_screen(c) {
                painter.line_segment([p - egui::vec2(3.0, 0.0), p + egui::vec2(3.0, 0.0)], Stroke::new(1.0, color));
                painter.line_segment([p - egui::vec2(0.0, 3.0), p + egui::vec2(0.0, 3.0)], Stroke::new(1.0, color));
            }
        }
    }
    // A dot on every end of a line or an arc: where one piece stops and the next begins.
    for e in &sketch.entities {
        if !matches!(e.shape, Shape::Line { .. } | Shape::Arc { .. }) {
            continue;
        }
        let color = if selected.contains(&e.id) { pal.selected } else { pal.dot };
        for &at in anchors_of(&e.shape) {
            if let Some(p) = to_screen(anchor_point(&e.shape, at)) {
                painter.circle_filled(p, dot::END / 2.0, color);
            }
        }
    }
    for p in picked_points.iter().filter_map(|p| to_screen(*p)) {
        painter.circle_filled(p, dot::PICKED / 2.0, pal.selected);
    }
    for p in lit_points.iter().filter_map(|p| to_screen(*p)) {
        painter.circle_filled(p, dot::HOVER / 2.0, pal.hover);
    }
    // The origin, as every CAD sketcher marks it, when it lies on the sheet laid round the
    // slice and the sketch.
    let mut sheet: Option<(DVec2, DVec2)> = around;
    for p in edit.chains.iter().flatten() {
        sheet = Some(sheet.map_or((*p, *p), |(lo, hi)| (lo.min(*p), hi.max(*p))));
    }
    let (lo, hi) = sheet.unwrap_or((DVec2::splat(-50.0), DVec2::splat(50.0)));
    let pad = (0.15 * (hi - lo).length()).max(5.0);
    if lo.x - pad <= 0.0 && 0.0 <= hi.x + pad && lo.y - pad <= 0.0 && 0.0 <= hi.y + pad {
        if let Some(o) = to_screen(DVec2::ZERO) {
            painter.circle_filled(o, dot::ORIGIN / 2.0, pal.datum);
        }
    }

    // The Pattern box's copies, where they would land.
    for shape in edit.pattern_preview() {
        stroke_shape(&shape, line::PIECE, pal.free);
    }
    // The Project tool's candidate, as the piece it would become.
    if edit.tool == Tool::Project {
        if let Some(found) = edit.cursor_at.and_then(|p| mm_reverse::project::projectable_at(&edit.projectables, p, edit.reach())) {
            stroke_shape(&found.entity.shape, line::PIECE, pal.projected);
        }
    }
    // The arc in the hand, tangent to the piece the chain ended with; or the drawing tool's
    // ghost and the mark where the click would snap.
    let mark = edit.reach() * 0.6;
    let draw_mark = |kind: SnapKind, at: DVec2| {
        for (a, b) in snap_mark(kind, at, mark) {
            if let (Some(a), Some(b)) = (to_screen(a), to_screen(b)) {
                painter.line_segment([a, b], Stroke::new(1.5, pal.free));
            }
        }
    };
    if let Some((arc, snap)) = edit.tangent_arc_preview() {
        stroke_shape(&arc, line::PIECE, pal.free);
        if let Some(kind) = snap.kind {
            draw_mark(kind, snap.point);
        }
    } else if let Some(snap) = &edit.cursor {
        let p = &edit.prefs;
        for shape in tool_preview(edit.tool, &edit.pending, snap.point, &p.draw_modes, p.polygon_sides) {
            stroke_shape(&shape, 1.5, pal.free);
        }
        if let Some(kind) = snap.kind {
            draw_mark(kind, snap.point);
        }
    }
    // The drawing tool's clicks so far, as pins — a chain's carried-on start is an end already
    // drawn, not a pin.
    if edit.chain.is_none() {
        for p in edit.pending.iter().filter_map(|p| to_screen(*p)) {
            painter.circle_filled(p, 3.5, pal.free);
        }
    }
    // The picture's clicks.
    if let Some(pick) = &edit.image_pick {
        for p in pick.points.iter().filter_map(|p| to_screen(*p)) {
            painter.circle_stroke(p, 5.0, Stroke::new(2.0, pal.selected));
        }
    }

    // Relation glyphs, lifted off their spot: every one with Show relations, else the one
    // selected in the list.
    let chosen = edit.selected_constraint();
    let font = egui::FontId::proportional(11.5);
    for c in &sketch.constraints {
        if c.rule.is_dimension() || (!edit.prefs.show_relations && chosen != Some(c.id)) {
            continue;
        }
        let (Some((text, _)), Some(at)) = (glyph(&c.rule), glyph_spot(&c.rule, sketch)) else { continue };
        if let Some(p) = to_screen(at) {
            let color = if chosen == Some(c.id) { pal.selected } else { pal.dimension };
            painter.text(p + egui::vec2(6.0, -7.0), egui::Align2::LEFT_BOTTOM, text, font.clone(), color);
        }
    }
    for e in fixed_pieces(sketch) {
        let id = mm_reverse::sketch_shapes::fix_id(e.id);
        let shown = edit.prefs.show_relations || edit.selection.iter().any(|p| matches!(p, Pick::Constraint(c) if *c as i64 == id));
        if !shown {
            continue;
        }
        if let Some(p) = to_screen(fix_spot(&e.shape)) {
            painter.text(p + egui::vec2(6.0, -7.0), egui::Align2::LEFT_BOTTOM, "⚓", font.clone(), pal.dimension);
        }
    }

    // A selection box being dragged: rightward a window, solid; leftward a crossing box,
    // dashed.
    if let Some((a, b)) = edit.boxing.as_ref().filter(|b| b.moved).and_then(|b| b.screen) {
        let r = egui::Rect::from_two_pos(a, b);
        let crossing = edit.boxing.as_ref().is_some_and(|b| b.crossing);
        let color = if crossing { CROSSING } else { WINDOW };
        painter.rect_filled(r, 0.0, color.gamma_multiply(0.12));
        if crossing {
            let corners = vec![r.left_top(), r.right_top(), r.right_bottom(), r.left_bottom(), r.left_top()];
            painter.add(EShape::dashed_line(&corners, Stroke::new(1.0, color), 4.0, 3.0));
        } else {
            painter.rect_stroke(r, 0.0, Stroke::new(1.0, color), egui::StrokeKind::Inside);
        }
    }
}
