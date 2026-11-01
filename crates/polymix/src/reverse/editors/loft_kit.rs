// SPDX-License-Identifier: AGPL-3.0-only
//! What the Loft, Sweep, Pattern and Move body boxes share: a field's ⓘ card, the body
//! select, the axis picker with its hint drawn on the part, and the script steps that drive the
//! four boxes headlessly. Ported from ScanRuler (`plugins/reverse/ui/FeatureEditor.tsx`,
//! `featureFields.tsx`, `PlaneSelect.tsx` RefPicker; `src/ui/InfoDot.tsx`), © Stefan Hermann
//! (CNC Kitchen), AGPL-3.0-only.

use egui::{Color32, Pos2, Stroke};
use glam::{DVec2, DVec3};
use mm_reverse::body_move::moves_nothing;
use mm_reverse::build::resolve_axis;
use mm_reverse::expression::ParameterKind;
use mm_reverse::frame::WorldAxis;
use mm_reverse::model::{AxisRef, BodyId, BodyOp, ElementGeometry, FeatureId, FeatureKind, Part, PatternMode};
use mm_reverse::parameters::{evaluate_parameters, Parameter, ParameterTable};

use crate::reverse::{Mode, ReverseWorkspace};
use crate::tools::Viewport;

/// The amber a plane or an axis the pointer is on in a picker is drawn in on the part.
pub(crate) const HINT: Color32 = Color32::from_rgb(0xff, 0xb0, 0x20);

/// A field's ⓘ: a small mark whose card, on hover, reads the title and the paragraphs.
pub(crate) fn info_dot(ui: &mut egui::Ui, title: &str, paragraphs: &[&str]) {
    ui.add(egui::Label::new(egui::RichText::new("ⓘ").weak().small()).sense(egui::Sense::hover())).on_hover_ui(|ui| {
        ui.set_max_width(300.0);
        ui.strong(title);
        for p in paragraphs {
            ui.label(*p);
        }
    });
}

/// A label with its ⓘ card, as a field's label reads.
pub(crate) fn label_with_info(ui: &mut egui::Ui, label: &str, hint: &str) {
    ui.label(label);
    info_dot(ui, label, &[hint]);
}

/// The last solid body there is, else the last body: what a feature on a body starts on.
pub(crate) fn last_solid(ws: &ReverseWorkspace) -> Option<BodyId> {
    let surface = |id: BodyId| ws.built.body(id).is_some_and(|b| b.surface);
    ws.bodies.iter().rev().find(|b| !surface(b.id)).or_else(|| ws.bodies.last()).map(|b| b.id)
}

/// The body select: `Body` and every body by name. Returns whether the body changed.
pub(crate) fn body_select(ui: &mut egui::Ui, ws: &ReverseWorkspace, id_salt: &str, body: &mut BodyId) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label("Body");
        let shown = ws.pick_set().iter().find(|b| b.id == *body).map_or_else(|| "Select…".to_string(), |b| b.name.clone());
        egui::ComboBox::from_id_salt(id_salt).selected_text(shown).show_ui(ui, |ui| {
            for b in ws.pick_set() {
                changed |= ui.selectable_value(body, b.id, &b.name).changed();
            }
        });
    });
    changed
}

/// An axis a pattern can take: its key, its name, the group it is listed under and the axis.
pub(crate) struct AxisOption {
    pub(crate) label: String,
    pub(crate) group: &'static str,
    pub(crate) axis: AxisRef,
}

/// The axes a pattern runs about or along: the line, cylinder, cone and circle elements, then
/// the coordinate axes.
pub(crate) fn axis_options(ws: &ReverseWorkspace) -> Vec<AxisOption> {
    let mut out: Vec<AxisOption> = ws
        .part
        .elements
        .iter()
        .filter(|e| matches!(e.geometry, ElementGeometry::Line { .. } | ElementGeometry::Cylinder { .. } | ElementGeometry::Cone { .. } | ElementGeometry::Circle { .. }))
        .map(|e| AxisOption { label: e.name.clone(), group: "Elements", axis: AxisRef::Element { id: e.id } })
        .collect();
    for a in WorldAxis::ALL {
        out.push(AxisOption { label: format!("{} axis", axis_letter(a)), group: "Coordinate axes", axis: AxisRef::World { axis: a } });
    }
    out
}

fn axis_letter(a: WorldAxis) -> &'static str {
    match a {
        WorldAxis::X => "X",
        WorldAxis::Y => "Y",
        WorldAxis::Z => "Z",
    }
}

/// The name an axis is listed by, or `None` where it is no longer there.
pub(crate) fn axis_label(ws: &ReverseWorkspace, axis: &AxisRef) -> Option<String> {
    axis_options(ws).into_iter().find(|o| o.axis == *axis).map(|o| o.label)
}

/// The axis picker (ScanRuler's RefPicker): a button with the chosen axis's name, or the
/// placeholder dimmed; its list grouped under Elements and Coordinate axes. Hovering an option
/// reports it, for the part to show it. Returns what was taken, and the axis under the pointer.
pub(crate) fn axis_picker(ui: &mut egui::Ui, ws: &ReverseWorkspace, id_salt: &str, value: Option<&AxisRef>, placeholder: &str) -> (Option<AxisRef>, Option<AxisRef>) {
    let options = axis_options(ws);
    let shown = value.and_then(|v| options.iter().find(|o| o.axis == *v)).map(|o| egui::RichText::new(&o.label)).unwrap_or_else(|| egui::RichText::new(placeholder).weak());
    let mut taken = None;
    let mut hovered = None;
    let r = egui::ComboBox::from_id_salt(id_salt).selected_text(shown).show_ui(ui, |ui| {
        if options.is_empty() {
            ui.weak("Nothing to choose from");
        }
        let mut group = "";
        for o in &options {
            if o.group != group {
                group = o.group;
                ui.label(egui::RichText::new(group.to_uppercase()).small().weak());
            }
            let r = ui.selectable_label(value == Some(&o.axis), &o.label);
            if r.hovered() {
                hovered = Some(o.axis);
            }
            if r.clicked() {
                taken = Some(o.axis);
            }
        }
    });
    if r.response.hovered() && hovered.is_none() {
        hovered = value.copied();
    }
    (taken, hovered)
}

/// Where the axis a picker's pointer is on runs, kept from the panel to the view for a frame.
pub(crate) fn hint_axis(ctx: &egui::Context, axis: Option<AxisRef>) {
    let id = egui::Id::NULL.with("re_loft_axis_hint");
    ctx.data_mut(|d| match axis {
        Some(a) => {
            d.insert_temp(id, a);
        }
        None => {
            d.remove::<AxisRef>(id);
        }
    });
}

/// The axis a picker hovers (or the one chosen, while the box is open), drawn on the part in
/// amber across the whole part, its name at its foot.
pub(crate) fn paint_axis_hint(ws: &ReverseWorkspace, painter: &egui::Painter, vp: Viewport, chosen: Option<&AxisRef>) {
    let hovered = painter.ctx().data(|d| d.get_temp::<AxisRef>(egui::Id::NULL.with("re_loft_axis_hint")));
    let Some(axis) = hovered.or(chosen.copied()) else { return };
    let Some((origin, dir)) = resolve_axis(&axis, None, &ws.part) else { return };
    let Some(dir) = dir.try_normalize() else { return };
    let reach = part_radius(ws);
    let centre = part_centre(ws);
    // The line through the axis, across the part: from the foot nearest the part's middle.
    let foot = origin + dir * (centre - origin).dot(dir);
    let (a, b) = (foot - dir * reach, foot + dir * reach);
    let n = 48;
    let pts: Vec<Pos2> = (0..=n).filter_map(|i| ws.project(vp, a.lerp(b, i as f64 / n as f64))).collect();
    if pts.len() >= 2 {
        painter.add(egui::Shape::line(pts.clone(), Stroke::new(3.0, HINT)));
        let name = axis_label(ws, &axis).unwrap_or_default();
        // At the foot of the line as it shows: its first point on the stage and clear of the panel.
        let shows = |p: &Pos2| vp.rect.shrink(8.0).contains(*p) && !ws.view.panel_rect.is_some_and(|r| r.expand(4.0).contains(*p));
        if let Some(at) = pts.iter().find(|p| shows(p)).or(pts.first()) {
            painter.text(*at + egui::vec2(6.0, -6.0), egui::Align2::LEFT_BOTTOM, name, egui::FontId::proportional(12.0), HINT);
        }
    }
}

/// The middle of what is on the stage, in part coordinates.
pub(crate) fn part_centre(ws: &ReverseWorkspace) -> DVec3 {
    let (lo, hi) = part_box(ws);
    if lo.x.is_finite() {
        (lo + hi) / 2.0
    } else {
        ws.scan_centre().unwrap_or(DVec3::ZERO)
    }
}

/// Half the diagonal of what is on the stage, or 60 mm on a blank one.
pub(crate) fn part_radius(ws: &ReverseWorkspace) -> f64 {
    let (lo, hi) = part_box(ws);
    if lo.x.is_finite() {
        ((hi - lo).length() / 2.0).max(1.0) * 1.15
    } else {
        60.0
    }
}

fn part_box(ws: &ReverseWorkspace) -> (DVec3, DVec3) {
    let mut lo = DVec3::splat(f64::INFINITY);
    let mut hi = DVec3::splat(f64::NEG_INFINITY);
    for b in &ws.bodies {
        for p in &b.part_mesh.positions {
            lo = lo.min(*p);
            hi = hi.max(*p);
        }
    }
    for o in ws.scan.iter().filter(|o| o.visible) {
        let bb = o.mesh.bounds();
        if !bb.is_empty() {
            for c in [bb.min, bb.max] {
                let p = ws.to_part(c);
                lo = lo.min(p);
                hi = hi.max(p);
            }
        }
    }
    (lo, hi)
}

/// The sketches of the part, by id and name, in the list's order.
pub(crate) fn sketches(ws: &ReverseWorkspace) -> Vec<(FeatureId, String)> {
    ws.part.features.iter().filter(|f| matches!(f.kind, FeatureKind::Sketch(_))).map(|f| (f.id, f.name.clone())).collect()
}

/// The part's parameters evaluated: what an expression typed into a field reads its names in.
pub(crate) fn parameter_table(part: &Part) -> ParameterTable {
    let parameters: Vec<Parameter> = part
        .parameters
        .iter()
        .map(|p| Parameter {
            id: p.id,
            name: p.name.clone(),
            kind: match p.kind {
                mm_reverse::model::ParameterKind::Length => ParameterKind::Length,
                mm_reverse::model::ParameterKind::Angle => ParameterKind::Angle,
                mm_reverse::model::ParameterKind::Integer => ParameterKind::Integer,
                mm_reverse::model::ParameterKind::Number => ParameterKind::Number,
            },
            expression: p.expression.clone(),
            comment: p.comment.clone(),
        })
        .collect();
    evaluate_parameters(&parameters)
}

/// Whether the open box of a Loft, a Sweep, a Pattern or a Move can make its feature — what its
/// Create waits for, as ScanRuler's `featureDraftReady` reads a draft: a loft through two sketches
/// or more; a sweep with its profile and a path of another sketch, or a route of two points or
/// more; a pattern of two copies or more at a step, and a grid's rows at a spacing; a move of a
/// body or more that moves them, or copies them where they stand. `None` for the other kinds.
pub(crate) fn ready(kind: &FeatureKind) -> Option<bool> {
    Some(match kind {
        FeatureKind::Loft(x) => x.sketches.len() >= 2,
        FeatureKind::Sweep(x) => match &x.route {
            Some(r) => r.points.len() >= 2,
            None => x.path.is_some_and(|p| p != x.profile),
        },
        FeatureKind::Pattern(x) => x.count >= 2 && x.step > 0.0 && (x.mode == PatternMode::Circular || x.axis2.is_none() || x.count2.unwrap_or(1) < 2 || x.step2.unwrap_or(0.0) > 0.0),
        FeatureKind::MoveBody(x) => !x.bodies.is_empty() && x.pivot.is_finite() && x.translate.is_finite() && x.rotate.is_finite() && (x.copy || !moves_nothing(x.translate, x.rotate)),
        _ => return None,
    })
}

/// What a new feature's Create says on these boxes: `Create loft`, `Create sweep`, `Create
/// pattern on {Body}`, and a Move's `Move the body` or `Copy the bodies`. `None` for the other
/// kinds.
pub(crate) fn create_label(ws: &ReverseWorkspace, kind: &FeatureKind) -> Option<String> {
    match kind {
        FeatureKind::Loft(_) => Some("Create loft".into()),
        FeatureKind::Sweep(_) => Some("Create sweep".into()),
        FeatureKind::Pattern(x) => {
            let named = ws.pick_set().iter().chain(ws.bodies.iter()).find(|b| b.id == x.body).map(|b| b.name.clone());
            Some(named.map_or_else(|| "Create pattern".to_string(), |n| format!("Create pattern on {n}")))
        }
        FeatureKind::MoveBody(x) => Some(super::move_body::create_label(x)),
        _ => None,
    }
}

// ---- script steps (tests) ----

/// The sketch a script names: by name, else by id.
fn sketch_named(ws: &ReverseWorkspace, name: &str) -> Option<FeatureId> {
    let found = ws.part.features.iter().find(|f| matches!(f.kind, FeatureKind::Sketch(_)) && f.name == name).map(|f| f.id);
    found.or_else(|| name.parse().ok().filter(|id| ws.part.sketch(*id).is_some()))
}

/// The body a script names: by name, else by its id as the list writes it.
fn body_named(ws: &ReverseWorkspace, name: &str) -> Option<BodyId> {
    ws.bodies.iter().find(|b| b.name == name || b.id.to_string() == name).map(|b| b.id)
}

fn op_of(word: Option<&&str>) -> Option<BodyOp> {
    match word.copied() {
        Some("cut") => Some(BodyOp::Cut),
        Some("new") => Some(BodyOp::New),
        Some("join") => Some(BodyOp::Join),
        _ => None,
    }
}

fn world_axis(word: &str) -> Option<AxisRef> {
    match word {
        "x" => Some(AxisRef::World { axis: WorldAxis::X }),
        "y" => Some(AxisRef::World { axis: WorldAxis::Y }),
        "z" => Some(AxisRef::World { axis: WorldAxis::Z }),
        _ => None,
    }
}

/// A script step of the Loft, Sweep, Pattern and Move boxes, run as a user would run the box:
/// opened, its fields set, Create.
///
/// * `loft:<sketch>,<sketch>[,…][:ruled|smooth[:join|cut|new]]`
/// * `sweep:<profile>:<path>[:join|cut|new]`, `sweep-route:<profile>:<bend radius>:<x,y,z>/<x,y,z>/…[:op]`
/// * `pattern:circular:<axis x|y|z>:<count>:<degrees>` or
///   `pattern:linear:<axis>:<count>:<spacing>[:<axis>:<rows>:<row spacing>]` — on the last solid body
/// * `move:<body>[,<body>…]:<dx>,<dy>,<dz>[:<rx>,<ry>,<rz>[:copy]]`
/// * `move-grip:<moveX … turnZ>:<dx>,<dy>` — a grip of the open Move's gizmo dragged by the pixels
///   given; `grip-field:<number or expression>` — typed into the field on it, and Enter;
///   `move-click:<x>,<y>,<z>` — a click where that point of the part is drawn
/// * `tube:<x>,<y>,<z>:<dx>,<dy>,<dz>[:both]` — a bent tube's centreline read off the scan, logged
/// * `loft-stations:<name>:<like>:<section>,<section>,…[:<inner>[:ruled|smooth[:outer|inner|-[:<u0>,<v0>,<u1>,<v1>]]]]`
///   — a loft through stations fitted like the sketch `like`; a section is a sketch's name or a
///   coordinate plane, `z=20`
///
/// Each box step with `-open` after its name (`loft-open`, `pattern-open`, …) leaves the box open.
pub(crate) fn script_step(ws: &mut ReverseWorkspace, cmd: &str, args: &[&str]) {
    // `<step>-open`: the box set up and left open, to look at.
    let (cmd, open) = match cmd.strip_suffix("-open") {
        Some(c) => (c, true),
        None => (cmd, false),
    };
    let create = |ws: &mut ReverseWorkspace| {
        if !open {
            ws.commit_feature();
            // A box not ready keeps its Create waiting, and stays open.
            if matches!(ws.mode, Mode::Feature(_)) {
                ws.status = "Create waits: the box is not ready to make its feature.".into();
                log::info!("reverse {cmd}: {}", ws.status);
            }
        }
    };
    let nums = |s: Option<&&str>| -> Vec<f64> { s.map(|s| s.split(',').filter_map(|v| v.trim().parse().ok()).collect()).unwrap_or_default() };
    let vec3 = |s: Option<&&str>| -> Option<DVec3> {
        let v = nums(s);
        (v.len() == 3).then(|| DVec3::new(v[0], v[1], v[2]))
    };
    match cmd {
        "loft" => {
            let ids: Vec<FeatureId> = args.first().map(|s| s.split(',').filter_map(|n| sketch_named(ws, n.trim())).collect()).unwrap_or_default();
            match super::loft::new_feature(ws) {
                Ok(FeatureKind::Loft(mut x)) => {
                    x.sketches = ids;
                    x.ruled = args.get(1).copied() != Some("smooth");
                    if let Some(op) = op_of(args.get(2)) {
                        x.op = op;
                    }
                    ws.open_feature(FeatureKind::Loft(x), None);
                    create(ws);
                }
                Ok(_) => {}
                Err(e) => ws.status = e,
            }
        }
        "sweep" | "sweep-route" => {
            // A route needs no path sketch: the profile's alone.
            let made = match cmd {
                "sweep" => super::sweep::new_feature(ws),
                _ => match sketches(ws).last() {
                    Some((profile, _)) => Ok(FeatureKind::Sweep(mm_reverse::model::SweepFeature { profile: *profile, path: None, route: None, op: BodyOp::Join, target: None, exprs: Default::default() })),
                    None => Err("A sweep along a route takes a profile sketch: draw one first.".into()),
                },
            };
            let Ok(FeatureKind::Sweep(mut x)) = made else {
                ws.status = made.err().unwrap_or_default();
                return;
            };
            if let Some(p) = args.first().and_then(|n| sketch_named(ws, n)) {
                x.profile = p;
            }
            let op_at = if cmd == "sweep" {
                x.path = args.get(1).and_then(|n| sketch_named(ws, n));
                2
            } else {
                let points: Vec<DVec3> = args.get(2).map(|s| s.split('/').filter_map(|p| vec3(Some(&p))).collect()).unwrap_or_default();
                x.route = Some(mm_reverse::model::SweepRoute { points, bend_radius: args.get(1).and_then(|v| v.parse().ok()).unwrap_or(10.0), radii: None });
                x.path = None;
                3
            };
            if let Some(op) = op_of(args.get(op_at)) {
                x.op = op;
            }
            ws.open_feature(FeatureKind::Sweep(x), None);
            create(ws);
        }
        "pattern" => {
            let made = super::pattern::new_feature(ws);
            let Ok(FeatureKind::Pattern(mut x)) = made else {
                ws.status = made.err().unwrap_or_default();
                return;
            };
            x.mode = if args.first().copied() == Some("linear") { PatternMode::Linear } else { PatternMode::Circular };
            if let Some(a) = args.get(1).and_then(|a| world_axis(a)) {
                x.axis = a;
            }
            x.count = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(x.count);
            x.step = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(if x.mode == PatternMode::Circular { 360.0 } else { 10.0 });
            if x.mode == PatternMode::Linear {
                if let Some(a) = args.get(4).and_then(|a| world_axis(a)) {
                    x.axis2 = Some(a);
                    x.count2 = args.get(5).and_then(|v| v.parse().ok()).or(Some(2));
                    x.step2 = args.get(6).and_then(|v| v.parse().ok()).or(Some(10.0));
                }
            }
            ws.open_feature(FeatureKind::Pattern(x), None);
            create(ws);
        }
        "move" => {
            let bodies: Vec<BodyId> = args.first().map(|s| s.split(',').filter_map(|n| body_named(ws, n.trim())).collect()).unwrap_or_default();
            let made = super::move_body::new_feature_on(ws, &bodies);
            let FeatureKind::MoveBody(mut x) = made else { return };
            if let Some(t) = vec3(args.get(1)) {
                x.translate = t;
            }
            if let Some(r) = vec3(args.get(2)) {
                x.rotate = r;
            }
            x.copy = args.get(3).copied() == Some("copy");
            ws.open_feature(FeatureKind::MoveBody(x), None);
            create(ws);
        }
        "loft-stations" => {
            let made = loft_stations_step(ws, args);
            ws.status = match made {
                Ok(words) => words,
                Err(e) => e,
            };
        }
        "move-grip" => {
            let by = nums(args.get(1));
            ws.status = match (args.first(), by.as_slice()) {
                (Some(side), [dx, dy]) => super::move_body::script_drag(ws, side, egui::vec2(*dx as f32, *dy as f32)).unwrap_or_else(|e| e),
                _ => "move-grip takes a grip and the pixels to drag it by: move-grip:<moveX … turnZ>:<dx>,<dy>".into(),
            };
            log::info!("reverse move-grip: {}", ws.status);
        }
        "grip-field" => {
            // The text whole, colons and all.
            let text = args.join(":");
            ws.status = super::move_body::script_type(ws, text.trim()).unwrap_or_else(|e| e);
            log::info!("reverse grip-field: {}", ws.status);
        }
        "move-click" => {
            ws.status = match vec3(args.first()) {
                Some(at) => super::move_body::script_click(ws, at).unwrap_or_else(|e| e),
                None => "move-click takes a point on the part: move-click:<x>,<y>,<z>".into(),
            };
            log::info!("reverse move-click: {}", ws.status);
        }
        "tube" => {
            let (Some(start), Some(dir)) = (vec3(args.first()), vec3(args.get(1))) else {
                ws.status = "tube takes a start and a direction: tube:<x>,<y>,<z>:<dx>,<dy>,<dz>[:both]".into();
                return;
            };
            ws.status = super::sweep::trace_tube_words(ws, start, dir, args.get(2).copied() == Some("both"));
            log::info!("reverse tube: {}", ws.status);
        }
        _ => {}
    }
    if !matches!(ws.mode, Mode::Feature(_)) {
        log::info!("reverse {cmd}: {}", ws.status);
    }
}

/// The `loft-stations` step: the loft and its stations made from the scan (see
/// `loft_stations::add_loft_stations`).
fn loft_stations_step(ws: &mut ReverseWorkspace, args: &[&str]) -> Result<String, String> {
    use super::loft_stations::{add_loft_stations, made_words, LoftStations, Section, StationFit};
    use mm_reverse::chain_outline::SheetBox;
    use mm_reverse::model::PlaneRef;
    use mm_reverse::section_skin::{SidedChains, Skin};
    use mm_reverse::slice::slice_mesh;
    let name = args.first().copied().filter(|n| !n.is_empty()).map(str::to_string);
    let like = args.get(1).and_then(|n| sketch_named(ws, n));
    let mut sections = Vec::new();
    for s in args.get(2).map_or(Vec::new(), |s| s.split(',').collect()) {
        let s = s.trim();
        let plane = s.split_once('=').and_then(|(a, v)| {
            let axis = match a {
                "x" => WorldAxis::X,
                "y" => WorldAxis::Y,
                "z" => WorldAxis::Z,
                _ => return None,
            };
            v.parse().ok().map(|o| PlaneRef::world(axis, o))
        });
        match (plane, sketch_named(ws, s)) {
            (Some(p), _) => sections.push(Section::Plane(p)),
            (None, Some(id)) => sections.push(Section::Sketch(id)),
            (None, None) => return Err(format!("There is no sketch named {s}.")),
        }
    }
    let inner = args.get(3).and_then(|v| v.parse::<f64>().ok());
    let ruled = args.get(4).map(|r| *r != "smooth");
    let skin = match args.get(5).copied() {
        Some("outer") => Some(Skin::Outer),
        Some("inner") => Some(Skin::Inner),
        _ => None,
    };
    let b: Vec<f64> = args.get(6).map_or(Vec::new(), |s| s.split(',').filter_map(|v| v.trim().parse().ok()).collect());
    let within = (b.len() == 4).then(|| SheetBox { min: DVec2::new(b[0], b[1]), max: DVec2::new(b[2], b[3]) });
    let fit = like.map(|like| StationFit { like, skin, within, tolerance: None });
    let ask = LoftStations { name, sections, fit, inner, ruled, ..Default::default() };
    let parts = ws.scan_parts();
    let cut = |frame: &mm_reverse::frame::SectionFrame| {
        let mut out = SidedChains::default();
        for p in parts.iter() {
            let c = slice_mesh(&p.positions, &p.triangles, frame.origin, frame.normal, 1.0);
            out.chains.extend(c.chains.iter().map(|ch| ch.iter().map(|q| frame.to_sheet(*q)).collect::<Vec<_>>()));
            out.normals.extend(c.normals.iter().map(|ns| ns.iter().map(|n| DVec2::new(n.dot(frame.u), n.dot(frame.v))).collect::<Vec<_>>()));
        }
        out
    };
    let frame_of = |part: &mm_reverse::model::Part, plane: &PlaneRef| mm_reverse::build::resolve_plane_frame(plane, part);
    let mut part = ws.part.clone();
    let made = add_loft_stations(&mut part, &ask, &cut, &frame_of)?;
    let words = made_words(&part, &made);
    ws.edit_part(move |p| *p = part);
    Ok(words)
}

#[cfg(test)]
mod tests {
    //! What the four boxes wait for before their Create, as ScanRuler's `featureDraftReady`
    //! (`state/reDrafts.ts`) reads a draft, and what their Create says.
    use glam::DVec3;
    use mm_reverse::frame::WorldAxis;
    use mm_reverse::model::{AxisRef, BodyId, BodyOp, FeatureKind, LoftFeature, MoveBodyFeature, PatternFeature, PatternMode, SweepFeature, SweepRoute};

    use super::ready;

    #[test]
    fn waits_for_two_sketches_a_path_of_another_sketch_a_step_and_a_move_that_moves() {
        let loft = |n: u32| FeatureKind::Loft(LoftFeature { sketches: (1..=n).collect(), ruled: true, op: BodyOp::Join, target: None });
        assert_eq!((ready(&loft(1)), ready(&loft(2))), (Some(false), Some(true)));
        let sweep = |path: Option<u32>, route: Option<SweepRoute>| FeatureKind::Sweep(SweepFeature { profile: 1, path, route, op: BodyOp::Join, target: None, exprs: Default::default() });
        assert_eq!(ready(&sweep(None, None)), Some(false));
        assert_eq!(ready(&sweep(Some(1), None)), Some(false), "the profile's own sketch is no path");
        assert_eq!(ready(&sweep(Some(2), None)), Some(true));
        let route = |n: usize| Some(SweepRoute { points: vec![DVec3::ZERO; n], bend_radius: 5.0, radii: None });
        assert_eq!((ready(&sweep(None, route(1))), ready(&sweep(None, route(2)))), (Some(false), Some(true)));
        let pattern = |f: &dyn Fn(&mut PatternFeature)| {
            let mut x = PatternFeature { body: BodyId::new(2), mode: PatternMode::Linear, axis: AxisRef::World { axis: WorldAxis::X }, count: 3, step: 5.0, axis2: None, count2: None, step2: None, exprs: Default::default() };
            f(&mut x);
            ready(&FeatureKind::Pattern(x))
        };
        assert_eq!(pattern(&|_| {}), Some(true));
        assert_eq!(pattern(&|x| x.step = 0.0), Some(false));
        assert_eq!(pattern(&|x| x.count = 1), Some(false));
        let grid = |x: &mut PatternFeature, rows: u32, spacing: f64| {
            x.axis2 = Some(AxisRef::World { axis: WorldAxis::Y });
            x.count2 = Some(rows);
            x.step2 = Some(spacing);
        };
        assert_eq!(pattern(&|x| grid(x, 2, 0.0)), Some(false));
        assert_eq!(pattern(&|x| grid(x, 1, 0.0)), Some(true), "one row is no grid");
        assert_eq!(pattern(&|x| grid(x, 2, 4.0)), Some(true));
        let mover = |f: &dyn Fn(&mut MoveBodyFeature)| {
            let mut x = MoveBodyFeature { bodies: vec![BodyId::new(2)], pivot: DVec3::splat(5.0), translate: DVec3::ZERO, rotate: DVec3::ZERO, copy: false, exprs: Default::default() };
            f(&mut x);
            ready(&FeatureKind::MoveBody(x))
        };
        // Nothing moved is not yet a move — but a copy where it stands is.
        assert_eq!(mover(&|_| {}), Some(false));
        assert_eq!(mover(&|x| x.copy = true), Some(true));
        assert_eq!(mover(&|x| x.rotate.z = 360.0), Some(false), "a whole turn moves nothing");
        assert_eq!(mover(&|x| x.translate.y = 0.5), Some(true));
        assert_eq!(mover(&|x| x.bodies.clear()), Some(false));
        assert_eq!(mover(&|x| x.translate.x = f64::NAN), Some(false));
        assert_eq!(ready(&FeatureKind::Sketch(mm_reverse::model::SketchFeature::on_world(WorldAxis::Z, 0.0, Default::default()))), None);
    }
}
