// SPDX-License-Identifier: AGPL-3.0-only
//! The sketch editor of the Reverse Engineer workspace: the sketch open on its plane, the
//! slice the plane cuts through the scan, and the tools that draw, fit, constrain and
//! dimension it. Its behaviour follows ScanRuler's sketch editor (`plugins/reverse/state/
//! reSketchSlice.ts`, `state/reSketchDraft.ts`, `app/keyboard.ts`, `app/useSketchSceneSync.ts`),
//! © Stefan Hermann (CNC Kitchen), AGPL-3.0-only: every click snaps (to an end already there,
//! the sheet's zero, a line's middle, a coordinate plane's line, a piece, a point of the slice,
//! the grid; Ctrl places a point exactly where the cursor is) and the new end is held to what
//! it landed on; a line drawn within 3° of level or upright is made so; the Line tool draws a
//! chain, and a press on its last end dragged swings a tangent arc out of it; the shape tools
//! draw each of their ways; a dragged end moves with the solver holding every relation, and a
//! line dragged by its body moves square to itself.
//!
//! Everything here is in sheet millimetres; the workspace hands in where the cursor lies on
//! the plane and how many millimetres a pixel is there, and paints through a projection (see
//! `sketch_paint`); the box in the panel is `sketch_box`.

use std::collections::HashSet;
use std::sync::Mutex;

use egui::Pos2;
use glam::DVec2;
use mm_reverse::box_select::pieces_in_box;
use mm_reverse::corners::{close_chain, join_ends, sharpen_corner, CORNER_GAP_MM};
use mm_reverse::datums::{datum_traces, trace_near, DatumPlane, DatumTrace};
use mm_reverse::drag::{detached, move_anchor, rigid_cluster, rotate_sketch, translate_entity, GrabAt};
use mm_reverse::draw::{arc_about, circle_from, coincident_for, dimension_for, draw_clicks, outward_tangent, rectangle_corners, rel_pick_at, tangent_arc, ArcMode, CircleMode, DimPick, DrawModes, RectangleMode, DEFAULT_DRAW_MODES};
use mm_reverse::fillet_corner::{chamfer_lines, corner_lines_at, fillet_lines};
use mm_reverse::frame::SectionFrame;
use mm_reverse::mirror_relation::{symmetric_pairs, MirrorLine};
use mm_reverse::model::{FeatureId, PlaneRef, SketchImage};
use mm_reverse::nominal::nominal_of;
use mm_reverse::project::{projectable_at, Projectable};
use mm_reverse::regions::{find_regions, Region};
use mm_reverse::section::{fit_all_chains, infer_all_corners, FitOptions, FittedChain};
use mm_reverse::sketch::{
    anchor_point, anchors_of, arc_through, axis_relation_of, measure_rule, nearest_on, Anchor, AnchorAt, ConstraintId, DistanceTo, Entity, EntityId, MirrorAxis, OnTarget, Rule, Shape, ShapeKind,
    Sketch, INFER_AXIS_DEG,
};
use mm_reverse::sketch_image::{level_image, move_image, scale_image};
use mm_reverse::sketch_shapes::{add_polygon, add_rectangle_centre, pattern_pieces, PolygonFit, SketchPatternSpec, POLYGON_MAX_SIDES};
use mm_reverse::snap::{snap_hold, snap_sketch_point, SnapContext, SnapHold, SnapResult};
use mm_reverse::solver::{solve_sketch, Pull, SolveOptions, SolveReport};

/// How near the cursor has to come to something to snap to it or pick it, in points: the
/// hand's reach on the sheet, as ScanRuler's stage hands it (`8 * unitsPerScreenPx`).
pub const SNAP_PX: f64 = 8.0;

/// The relations the panel's keys put on pieces, in the order of the keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelationKind {
    Coincident,
    Horizontal,
    Vertical,
    Parallel,
    Perpendicular,
    Tangent,
    Concentric,
    Equal,
    Symmetric,
}

impl RelationKind {
    pub const ALL: [RelationKind; 9] = [
        RelationKind::Coincident,
        RelationKind::Horizontal,
        RelationKind::Vertical,
        RelationKind::Parallel,
        RelationKind::Perpendicular,
        RelationKind::Tangent,
        RelationKind::Concentric,
        RelationKind::Equal,
        RelationKind::Symmetric,
    ];

    pub fn label(self) -> &'static str {
        match self {
            RelationKind::Coincident => "Coincident",
            RelationKind::Horizontal => "Horizontal",
            RelationKind::Vertical => "Vertical",
            RelationKind::Parallel => "Parallel",
            RelationKind::Perpendicular => "Perpendicular",
            RelationKind::Tangent => "Tangent",
            RelationKind::Concentric => "Concentric",
            RelationKind::Equal => "Equal",
            RelationKind::Symmetric => "Symmetric",
        }
    }

    /// The kind as `mm_reverse::sketch::relation_takes` names it.
    fn key(self) -> &'static str {
        match self {
            RelationKind::Coincident => "coincident",
            RelationKind::Horizontal => "horizontal",
            RelationKind::Vertical => "vertical",
            RelationKind::Parallel => "parallel",
            RelationKind::Perpendicular => "perpendicular",
            RelationKind::Tangent => "tangent",
            RelationKind::Concentric => "concentric",
            RelationKind::Equal => "equal",
            RelationKind::Symmetric => "symmetric",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tool {
    Select,
    Line,
    Rectangle,
    Polygon,
    Circle,
    Arc,
    Fillet,
    Chamfer,
    Project,
    Dimension,
    /// A relation in hand: the pieces it takes are clicked next.
    Relation(RelationKind),
}

impl Tool {
    /// The tools the keyboard takes, by the letters CAD has them on (`app/keyboard.ts`).
    pub const KEYED: [Tool; 5] = [Tool::Dimension, Tool::Line, Tool::Rectangle, Tool::Circle, Tool::Fillet];

    pub fn key(self) -> Option<egui::Key> {
        match self {
            Tool::Line => Some(egui::Key::L),
            Tool::Rectangle => Some(egui::Key::R),
            Tool::Circle => Some(egui::Key::C),
            Tool::Fillet => Some(egui::Key::F),
            Tool::Dimension => Some(egui::Key::D),
            _ => None,
        }
    }

    /// The shape tools, whose keys have more than one way to draw.
    pub fn is_shape(self) -> bool {
        matches!(self, Tool::Rectangle | Tool::Polygon | Tool::Circle | Tool::Arc)
    }

    /// The tools whose clicks take what is there rather than placing points.
    fn picks(self) -> bool {
        matches!(self, Tool::Dimension | Tool::Relation(_) | Tool::Fillet | Tool::Chamfer)
    }

    fn name(self) -> &'static str {
        match self {
            Tool::Line => "line",
            Tool::Rectangle => "rectangle",
            Tool::Polygon => "polygon",
            Tool::Circle => "circle",
            Tool::Arc => "arc",
            _ => "",
        }
    }
}

/// What can be picked on the sheet.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Pick {
    Entity(EntityId),
    Anchor(Anchor),
    Constraint(ConstraintId),
}

// ---- what the sketcher remembers between sessions ----

/// The sketch's settings that outlive a sketch: the way each shape tool draws, a polygon's
/// sides, grid snapping and the relation glyphs (ScanRuler's `rePrefs`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SketchPrefs {
    pub draw_modes: DrawModes,
    pub polygon_sides: usize,
    pub grid_snap: bool,
    pub show_relations: bool,
}

impl SketchPrefs {
    pub const DEFAULT: SketchPrefs = SketchPrefs { draw_modes: DEFAULT_DRAW_MODES, polygon_sides: 6, grid_snap: false, show_relations: false };
}

static PREFS: Mutex<SketchPrefs> = Mutex::new(SketchPrefs::DEFAULT);

/// The sketch settings as they stand.
pub fn prefs() -> SketchPrefs {
    PREFS.lock().map(|p| *p).unwrap_or(SketchPrefs::DEFAULT)
}

pub fn set_prefs(p: SketchPrefs) {
    if let Ok(mut held) = PREFS.lock() {
        *held = p;
    }
}

/// A polygon's sides as kept: three and up, at most the most a polygon is drawn with.
pub fn clamp_sides(n: f64) -> usize {
    if n.is_finite() && n >= 3.0 {
        (n.round() as usize).min(POLYGON_MAX_SIDES)
    } else {
        6
    }
}

/// The settings from the last session (none in a scripted run).
pub fn load_prefs(storage: Option<&dyn eframe::Storage>) {
    let Some(st) = storage.filter(|_| !polymix_workspace::scripted()) else { return };
    let mut p = SketchPrefs::DEFAULT;
    if let Some(m) = st.get_string("reverse.sketch.draw_modes").and_then(|s| serde_json::from_str::<DrawModes>(&s).ok()) {
        p.draw_modes = m;
    }
    if let Some(n) = st.get_string("reverse.sketch.polygon_sides").and_then(|s| s.parse::<f64>().ok()) {
        p.polygon_sides = clamp_sides(n);
    }
    p.grid_snap = st.get_string("reverse.sketch.grid_snap").is_some_and(|v| v == "1");
    p.show_relations = st.get_string("reverse.sketch.relations").is_some_and(|v| v == "1");
    set_prefs(p);
}

pub fn save_prefs(storage: &mut dyn eframe::Storage) {
    if polymix_workspace::scripted() {
        return;
    }
    let p = prefs();
    if let Ok(m) = serde_json::to_string(&p.draw_modes) {
        storage.set_string("reverse.sketch.draw_modes", m);
    }
    storage.set_string("reverse.sketch.polygon_sides", p.polygon_sides.to_string());
    storage.set_string("reverse.sketch.grid_snap", if p.grid_snap { "1" } else { "0" }.into());
    storage.set_string("reverse.sketch.relations", if p.show_relations { "1" } else { "0" }.into());
}

/// The grid's spacing at a scale (ScanRuler's `gridSpacing`): the first rung of the ladder at
/// least 28 points apart on the screen.
pub fn grid_spacing(mm_per_px: f64) -> f64 {
    const LADDER: [f64; 10] = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0];
    for s in LADDER {
        if s / mm_per_px >= 28.0 {
            return s;
        }
    }
    LADDER[LADDER.len() - 1]
}

// ---- the input and what the sketch keeps between frames ----

/// What the workspace hands the sketcher each frame.
#[derive(Clone, Copy, Default)]
pub struct SketchInput<'a> {
    /// The cursor on the sheet.
    pub pos: Option<DVec2>,
    /// Millimetres per point at the cursor.
    pub mm_per_px: f64,
    pub pressed: bool,
    pub down: bool,
    pub released: bool,
    pub clicked: bool,
    pub shift: bool,
    /// Ctrl/Cmd: place the point exactly where the cursor is, no snapping; a drag of a line
    /// moves it freely; a box adds to the selection.
    pub exact: bool,
    /// The cursor on the screen, and a screen point laid on the sheet — what a selection box's
    /// corners are found by.
    pub screen: Option<Pos2>,
    pub unproject: Option<&'a dyn Fn(Pos2) -> Option<DVec2>>,
}

/// What a click would take in the sketch, lit under the cursor before the click (ScanRuler's
/// `SketchHover`): pieces, an end or a centre, or a coordinate plane's line.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SketchHover {
    pub pieces: Vec<EntityId>,
    pub point: Option<DVec2>,
    pub datum: Option<DatumPlane>,
}

/// A piece being dragged on the sheet (ScanRuler's `sketchDrag`).
struct Drag {
    entity: EntityId,
    at: GrabAt,
    start: DVec2,
    press: Option<Pos2>,
    origin: Entity,
    snapped: Option<SnapHold>,
    pull: Vec<Pull>,
    begun: bool,
    /// For a shape defined in all but its place: every piece of it as it was when the drag
    /// began, carried whole by the hand.
    rigid: Option<Vec<Entity>>,
}

/// A selection box dragged over the bare sheet.
#[derive(Debug, Clone)]
pub struct BoxDrag {
    pub start_screen: Pos2,
    pub moved: bool,
    /// The box laid on the sheet, the press's corner first.
    pub corners: Option<[DVec2; 4]>,
    /// Dragged leftward: a crossing box, which takes whatever it touches.
    pub crossing: bool,
    /// The pieces it would take.
    pub ids: Vec<EntityId>,
    pub screen: Option<(Pos2, Pos2)>,
}

/// The Line tool's press on the chain's last end: let go there it is a click; held and
/// dragged it swings a tangent arc out of the end.
#[derive(Debug, Clone, Copy)]
struct ArcDrag {
    press: Option<Pos2>,
    moved: bool,
    at: DVec2,
    raw: bool,
}

/// What the picture row's clicks are for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImagePickMode {
    Scale,
    Move,
    Level,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImagePick {
    pub mode: ImagePickMode,
    pub points: Vec<DVec2>,
}

/// A sketch open for editing.
pub struct SketchEdit {
    /// The feature being edited, or `None` for a new sketch.
    pub feature: Option<FeatureId>,
    pub plane: PlaneRef,
    pub frame: SectionFrame,
    pub sketch: Sketch,
    pub traces: Vec<DatumTrace>,
    /// The slice the plane cuts through the scan, laid flat.
    pub chains: Vec<Vec<DVec2>>,
    pub fitted: Vec<FittedChain>,
    pub tool: Tool,
    /// A drawing tool's points so far, and what each was snapped to.
    pub pending: Vec<DVec2>,
    snaps: Vec<Option<SnapHold>>,
    /// The end a chain of lines carries on from.
    pub chain: Option<Anchor>,
    pub selection: Vec<Pick>,
    pub hover: Option<Pick>,
    /// What a click would take under the cursor, lit before the click.
    pub lit: SketchHover,
    /// The Dimension tool's first pick.
    dim_first: Option<Pick>,
    /// Coincident's first pick, while it waits for the second.
    pub rel_picks: Vec<DimPick>,
    pub report: SolveReport,
    pub regions: Vec<Region>,
    drag: Option<Drag>,
    pub boxing: Option<BoxDrag>,
    arc_drag: Option<ArcDrag>,
    undo: Vec<Sketch>,
    redo: Vec<Sketch>,
    /// A dimension open for typing: its id and the text so far.
    pub edit_dim: Option<(ConstraintId, String)>,
    /// What the status line says about the last action.
    pub message: String,
    /// What went wrong, or what is being waited for — the box's hint line.
    pub error: Option<String>,
    /// Where the next click would land, and the cursor on the sheet.
    pub cursor: Option<SnapResult>,
    pub cursor_at: Option<DVec2>,
    /// Ctrl held at the cursor.
    pub raw: bool,
    /// Millimetres per point at the cursor, last frame.
    pub mm_per_px: f64,
    pub fit_options: FitOptions,
    pub prefs: SketchPrefs,
    /// The Pattern box, while it is open.
    pub pattern: Option<SketchPatternSpec>,
    /// The picture under the sketch, and the clicks its row asked for.
    pub image: Option<SketchImage>,
    pub image_pick: Option<ImagePick>,
    /// How far off the plane the scan is cut for this sketch's slice.
    pub level: f64,
    /// The setback the next chamfer starts with.
    last_chamfer: Option<f64>,
    /// The sheet was turned (Align): the view swings round to it.
    pub frame_turned: bool,
    /// What the Project tool can bring in, and what they were found for.
    pub projectables: Vec<Projectable>,
    pub projectables_key: Option<u64>,
}

impl SketchEdit {
    pub fn new(feature: Option<FeatureId>, plane: PlaneRef, frame: SectionFrame, sketch: Sketch, chains: Vec<Vec<DVec2>>, fit_options: FitOptions) -> Self {
        let mut e = SketchEdit {
            feature,
            plane,
            frame,
            traces: datum_traces(&frame),
            sketch,
            chains,
            fitted: Vec::new(),
            tool: Tool::Select,
            pending: Vec::new(),
            snaps: Vec::new(),
            chain: None,
            selection: Vec::new(),
            hover: None,
            lit: SketchHover::default(),
            dim_first: None,
            rel_picks: Vec::new(),
            report: SolveReport { ok: true, ..Default::default() },
            regions: Vec::new(),
            drag: None,
            boxing: None,
            arc_drag: None,
            undo: Vec::new(),
            redo: Vec::new(),
            edit_dim: None,
            message: String::new(),
            error: None,
            cursor: None,
            cursor_at: None,
            raw: false,
            mm_per_px: 0.1,
            fit_options,
            prefs: prefs(),
            pattern: None,
            image: None,
            image_pick: None,
            level: 0.0,
            last_chamfer: None,
            frame_turned: false,
            projectables: Vec::new(),
            projectables_key: None,
        };
        e.solve();
        e
    }

    /// Move the plane (a new offset): the frame and the slice follow; the pieces stay where
    /// they are on the sheet.
    pub fn set_plane(&mut self, plane: PlaneRef, frame: SectionFrame, chains: Vec<Vec<DVec2>>) {
        self.plane = plane;
        self.frame = frame;
        self.traces = datum_traces(&self.frame);
        self.chains = chains;
        self.fitted.clear();
        self.projectables_key = None;
        self.solve();
    }

    /// Solve the sketch and find its regions again — what every change ends with.
    pub fn solve(&mut self) {
        self.report = solve_sketch(&mut self.sketch, SolveOptions { datums: &self.traces, ..Default::default() });
        self.regions = find_regions(&self.sketch);
    }

    /// One step of a drag: solved briefly with the hand's pull, the loose pieces and the
    /// regions left as the last full solve found them.
    fn solve_quick(&mut self, pull: &[Pull]) {
        let loose = std::mem::take(&mut self.report.under_constrained);
        self.report = solve_sketch(&mut self.sketch, SolveOptions { pull, quick: true, datums: &self.traces });
        self.report.under_constrained = loose;
    }

    /// Settled in full with the hand still on what it held.
    fn solve_pulled(&mut self, pull: &[Pull]) {
        self.report = solve_sketch(&mut self.sketch, SolveOptions { pull, quick: false, datums: &self.traces });
        self.regions = find_regions(&self.sketch);
    }

    /// Keep the sketch as it is for Undo, before a change.
    pub fn checkpoint(&mut self) {
        self.undo.push(self.sketch.clone());
        if self.undo.len() > 200 {
            self.undo.remove(0);
        }
        self.redo.clear();
    }

    pub fn undo(&mut self) -> bool {
        let Some(s) = self.undo.pop() else { return false };
        self.redo.push(std::mem::replace(&mut self.sketch, s));
        self.after_history();
        true
    }

    pub fn redo(&mut self) -> bool {
        let Some(s) = self.redo.pop() else { return false };
        self.undo.push(std::mem::replace(&mut self.sketch, s));
        self.after_history();
        true
    }

    fn after_history(&mut self) {
        self.tool_idle();
        self.selection.clear();
        self.edit_dim = None;
        let ids: HashSet<EntityId> = self.sketch.entities.iter().map(|e| e.id).collect();
        self.fitted.retain(|f| f.order.iter().all(|p| ids.contains(&p.id)));
        self.solve();
    }

    /// Nothing in hand: no half-placed points, no chain, no pick.
    fn tool_idle(&mut self) {
        self.pending.clear();
        self.snaps.clear();
        self.chain = None;
        self.dim_first = None;
        self.rel_picks.clear();
        self.arc_drag = None;
    }

    /// Take up a tool (ScanRuler's `setSketchTool`): what was in hand is put down; the
    /// Dimension tool and the corner tools start with nothing selected.
    pub fn set_tool(&mut self, tool: Tool) {
        self.tool_idle();
        self.tool = tool;
        self.error = None;
        self.message.clear();
        if matches!(tool, Tool::Dimension | Tool::Fillet | Tool::Chamfer) || matches!(tool, Tool::Relation(_)) {
            self.selection.clear();
        }
    }

    /// A tool's letter pressed (`app/keyboard.ts`): D with pieces selected under the Select tool
    /// dimensions them; the tool in hand's own letter puts it down.
    pub fn key_tool(&mut self, tool: Tool) {
        if tool == Tool::Dimension && self.tool == Tool::Select && !self.selected_entities().is_empty() {
            self.dimension_selection();
        } else if self.tool == tool {
            self.set_tool(Tool::Select);
        } else {
            self.set_tool(tool);
        }
    }

    /// A tool's key pressed in the panel: a corner tool on two lines already selected works on
    /// those; Fillet and Chamfer pressed again while in hand are put down.
    pub fn press_tool(&mut self, tool: Tool) {
        let corner = matches!(tool, Tool::Fillet | Tool::Chamfer);
        let ids = self.selected_entities();
        if corner && ids.len() == 2 && self.tool != tool {
            if tool == Tool::Fillet {
                self.fillet_draft(ids[0], ids[1]);
            } else {
                self.chamfer_draft(ids[0], ids[1]);
            }
            return;
        }
        self.set_tool(if corner && self.tool == tool { Tool::Select } else { tool });
    }

    /// Whether a tool has clicks in progress (a chain of lines, a dimension's first pick).
    pub fn busy(&self) -> bool {
        !self.pending.is_empty() || self.chain.is_some() || self.dim_first.is_some() || !self.rel_picks.is_empty()
    }

    /// Escape (`app/keyboard.ts`): backs out one step — the points in hand, the pick, the
    /// selection or the selected relation first, then the tool. Never closes the sketch.
    pub fn escape(&mut self) {
        let selected = !self.selection.is_empty();
        if self.busy() || selected || self.edit_dim.is_some() {
            self.clear_selection();
        } else if self.tool != Tool::Select {
            self.set_tool(Tool::Select);
        }
    }

    /// Everything in hand put down, the selection let go, the Pattern box and the picture's
    /// clicks closed (ScanRuler's `clearSelection`).
    pub fn clear_selection(&mut self) {
        self.tool_idle();
        self.selection.clear();
        self.edit_dim = None;
        self.pattern = None;
        self.image_pick = None;
    }

    /// Enter or a middle click (`confirmable`): a chain in progress is ended first — the
    /// points in hand put down. Returns whether that is what it did; when not, the sketch is
    /// to be finished.
    pub fn confirm_step(&mut self) -> bool {
        if self.edit_dim.is_some() {
            return true;
        }
        if !self.pending.is_empty() {
            self.clear_selection();
            return true;
        }
        false
    }

    // ---- fitting ----

    /// Fit all chains: every chain of the slice not fitted yet cut into lines and arcs, their
    /// corners joined, the sketch solved onto the slice points.
    pub fn fit_all(&mut self) {
        if self.chains.is_empty() {
            self.message = "The plane cuts nothing of the scan here.".into();
            return;
        }
        self.checkpoint();
        let done: Vec<usize> = self.fitted.iter().map(|f| f.chain).collect();
        let report = fit_all_chains(&mut self.sketch, &self.chains, &self.frame, self.fit_options, &done);
        self.regions = find_regions(&self.sketch);
        self.report = report.solve.clone();
        let mut msg = format!(
            "Fitted {} chain{}: {} line{}, {} arc{}; {} corner{} joined ({} tangent)",
            report.fitted.len(),
            plural(report.fitted.len()),
            report.lines,
            plural(report.lines),
            report.arcs,
            plural(report.arcs),
            report.joined,
            plural(report.joined),
            report.tangent
        );
        if report.open > 0 {
            msg += &format!(", {} gap{} left open (Close bridges them)", report.open, plural(report.open));
        }
        if !report.failed.is_empty() {
            msg += &format!("; {} chain{} too short to fit", report.failed.len(), plural(report.failed.len()));
        }
        self.message = msg + ".";
        self.selection.clear();
        self.fitted.extend(report.fitted);
    }

    /// Refit what Fit all chains made, with the options as they are now (one Undo step).
    pub fn refit(&mut self) {
        if self.fitted.is_empty() {
            return self.fit_all();
        }
        let made: Vec<EntityId> = self.fitted.iter().flat_map(|f| f.order.iter().map(|p| p.id)).collect();
        self.checkpoint();
        self.sketch.remove_entities(&made);
        self.fitted.clear();
        let undo = self.undo.pop();
        self.fit_all();
        if let Some(u) = undo {
            // One step for the whole refit.
            self.undo.pop();
            self.undo.push(u);
        }
    }

    pub fn infer_corners(&mut self) {
        self.checkpoint();
        let n = infer_all_corners(&mut self.sketch, &self.fitted) + self.sketch.infer_relations();
        self.solve();
        self.message = format!("{n} relation{} added.", plural(n));
    }

    pub fn close(&mut self) {
        self.checkpoint();
        let mut n: usize = self.fitted.iter().map(|f| close_chain(&mut self.sketch, &f.order, CORNER_GAP_MM)).sum();
        n += self.sketch.infer_relations();
        self.solve();
        self.error = (n == 0).then(|| "Nothing left to close.".to_string());
        self.message = if n == 0 { "Nothing was left open.".into() } else { format!("{n} gap{} closed.", plural(n)) };
    }

    /// Cut the scan this far off the plane for the slice (Level): the chains of another level
    /// are other chains, none of them fitted. Returns whether it changed.
    pub fn set_level(&mut self, level: f64) -> bool {
        if !level.is_finite() || self.level == level {
            return false;
        }
        self.level = level;
        self.fitted.clear();
        self.error = None;
        true
    }

    /// The frame the slice is cut on: the sketch's own, moved to the level.
    pub fn slice_frame(&self) -> SectionFrame {
        if self.level == 0.0 {
            self.frame
        } else {
            self.frame.offset(self.level)
        }
    }

    // ---- the selection ----

    pub fn selected_entities(&self) -> Vec<EntityId> {
        self.selection.iter().filter_map(|p| if let Pick::Entity(id) = p { Some(*id) } else { None }).collect()
    }

    pub fn selected_constraint(&self) -> Option<ConstraintId> {
        self.selection.iter().find_map(|p| if let Pick::Constraint(id) = p { Some(*id) } else { None })
    }

    /// Select a relation or a dimension (its row clicked, its glyph or its number).
    pub fn select_constraint(&mut self, id: ConstraintId) {
        self.selection.retain(|p| !matches!(p, Pick::Constraint(_)));
        self.selection.push(Pick::Constraint(id));
    }

    /// A relation put on the pieces selected (ScanRuler's `addRelation`); with fewer than it
    /// takes, the relation is taken in hand to click its pieces on the sheet.
    pub fn add_relation(&mut self, kind: RelationKind) {
        let picked: Vec<Entity> = self.selected_entities().iter().filter_map(|id| self.sketch.entity(*id).cloned()).collect();
        if kind == RelationKind::Coincident {
            // On two selected pieces: two lines lie along each other; a circle meets by its
            // centre; anything else meets at its nearest ends.
            if picked.len() != 2 {
                return self.arm_relation(kind);
            }
            self.checkpoint();
            if picked[0].kind() == ShapeKind::Line && picked[1].kind() == ShapeKind::Line {
                let id = self.sketch.constrain(Rule::Collinear { line: picked[0].id, on: OnTarget::Line(picked[1].id) });
                self.solve();
                self.select_constraint(id);
                return;
            }
            match nearest_ends(&picked[0], &picked[1]) {
                Some((p, q)) => {
                    join_ends(&mut self.sketch, p, q, f64::INFINITY);
                    self.solve();
                }
                None => {
                    let circle = picked.iter().find(|e| e.kind() == ShapeKind::Circle);
                    let other = picked.iter().find(|e| Some(e.id) != circle.map(|c| c.id));
                    let (Some(circle), Some(other)) = (circle, other) else {
                        self.undo.pop();
                        self.error = Some("Two circles are held by Concentric, not Coincident.".into());
                        return;
                    };
                    match coincident_for(&self.sketch, DimPick::Point(Anchor::new(circle.id, AnchorAt::Center)), DimPick::Piece(other.id)) {
                        Ok(rule) => {
                            let id = self.sketch.constrain(rule);
                            self.solve();
                            self.select_constraint(id);
                        }
                        Err(why) => {
                            self.undo.pop();
                            self.error = Some(why);
                        }
                    }
                }
            }
            return;
        }
        if kind == RelationKind::Symmetric {
            // Two pieces and the line they mirror about, in the order picked; with the two
            // selected the next click is the mirror line.
            if picked.len() == 2 {
                self.tool_idle();
                self.tool = Tool::Relation(kind);
                self.error = None;
                return;
            }
            if picked.len() != 3 {
                self.error = Some("Symmetric takes the two pieces that mirror each other, then the line they mirror about.".into());
                return;
            }
            return self.hold_symmetric(&picked[0], &picked[1], MirrorLine::Piece(&picked[2]));
        }
        let kinds: Vec<ShapeKind> = picked.iter().map(|e| e.kind()).collect();
        if picked.is_empty() {
            return self.arm_relation(kind);
        }
        if !mm_reverse::sketch::relation_takes(kind.key(), &kinds) {
            self.error = Some(format!("{} does not apply to what is selected.", kind.label()));
            return;
        }
        self.checkpoint();
        match kind {
            RelationKind::Horizontal | RelationKind::Vertical => {
                for e in &picked {
                    self.sketch.constrain(if kind == RelationKind::Horizontal { Rule::Horizontal { entity: e.id } } else { Rule::Vertical { entity: e.id } });
                }
            }
            RelationKind::Parallel => {
                self.sketch.constrain(Rule::Parallel { a: picked[0].id, b: picked[1].id });
            }
            RelationKind::Perpendicular => {
                self.sketch.constrain(Rule::Perpendicular { a: picked[0].id, b: picked[1].id });
            }
            RelationKind::Tangent => {
                self.sketch.constrain(Rule::Tangent { a: picked[0].id, b: picked[1].id });
            }
            RelationKind::Concentric => {
                self.sketch.constrain(Rule::Concentric { a: picked[0].id, b: picked[1].id });
            }
            RelationKind::Equal => {
                self.sketch.constrain(Rule::Equal { a: picked[0].id, b: picked[1].id });
            }
            _ => {}
        }
        self.solve();
        self.error = None;
    }

    /// A relation taken in hand: its pieces are clicked next (ScanRuler's `armRelation`).
    fn arm_relation(&mut self, kind: RelationKind) {
        self.tool_idle();
        self.tool = Tool::Relation(kind);
        self.selection.clear();
        self.error = None;
    }

    /// The relation key pressed: with a selection (and not in hand) it is put on it; in hand it
    /// is put down; else taken in hand.
    pub fn press_relation(&mut self, kind: RelationKind) {
        let armed = self.tool == Tool::Relation(kind);
        if !self.selected_entities().is_empty() && !armed {
            self.add_relation(kind);
        } else if armed {
            self.set_tool(Tool::Select);
        } else {
            self.arm_relation(kind);
        }
    }

    /// Symmetric between two pieces about a sketch line or one of the part's coordinate planes
    /// (ScanRuler's `holdSymmetric`).
    fn hold_symmetric(&mut self, a: &Entity, b: &Entity, axis: MirrorLine) {
        let about = match axis {
            MirrorLine::Piece(e) => MirrorAxis::Line(e.id),
            MirrorLine::Datum(t) => MirrorAxis::Datum(t.plane),
        };
        match symmetric_pairs(a, b, axis) {
            Err(why) => self.error = Some(why),
            Ok(pairs) => {
                self.checkpoint();
                for pair in pairs {
                    self.sketch.constrain(Rule::Symmetric { p: pair.p, q: pair.q, about });
                }
                self.solve();
                self.error = None;
            }
        }
    }

    /// Toggle Fix on the selected pieces; a projected piece stays where the part put it.
    pub fn toggle_fix(&mut self) {
        let ids: Vec<EntityId> = self.selected_entities().into_iter().filter(|id| self.sketch.entity(*id).is_some_and(|e| !e.projected)).collect();
        if self.selected_entities().is_empty() {
            return;
        }
        self.checkpoint();
        let fix = !ids.iter().all(|id| self.sketch.entity(*id).is_some_and(|e| e.fixed));
        for id in ids {
            if let Some(e) = self.sketch.entity_mut(id) {
                e.fixed = fix;
            }
        }
        self.solve();
    }

    /// Take a piece's Fix off — the piece is free to move again.
    pub fn unfix(&mut self, id: EntityId) {
        self.checkpoint();
        if let Some(e) = self.sketch.entity_mut(id) {
            e.fixed = false;
        }
        self.solve();
    }

    /// Toggle Construction on the selected pieces.
    pub fn toggle_construction(&mut self) {
        let ids = self.selected_entities();
        if ids.is_empty() {
            return;
        }
        self.checkpoint();
        let on = !ids.iter().all(|id| self.sketch.entity(*id).is_some_and(|e| e.construction));
        for id in ids {
            if let Some(e) = self.sketch.entity_mut(id) {
                e.construction = on;
            }
        }
        self.solve();
    }

    /// Sharp corner: a fillet arc (or a chamfer's line) between two lines taken out, the lines
    /// carried on to their corner.
    pub fn sharpen(&mut self) {
        let ids: Vec<EntityId> = self.selected_entities().into_iter().filter(|id| self.sketch.entity(*id).is_some_and(|e| matches!(e.kind(), ShapeKind::Arc | ShapeKind::Line))).collect();
        if ids.is_empty() {
            self.error = Some("Select the fillet arc or the chamfer line to make sharp.".into());
            return;
        }
        self.checkpoint();
        let mut done = 0;
        for id in ids {
            if sharpen_corner(&mut self.sketch, id) {
                done += 1;
            }
        }
        if done == 0 {
            self.undo.pop();
            self.error = Some("The piece is not a fillet or a chamfer between two lines.".into());
            return;
        }
        self.selection.clear();
        self.solve();
        self.error = None;
    }

    pub fn delete_selection(&mut self) {
        if self.selection.is_empty() {
            return;
        }
        self.checkpoint();
        let entities = self.selected_entities();
        for p in &self.selection {
            if let Pick::Constraint(id) = p {
                self.sketch.remove_constraint(*id);
            }
        }
        self.sketch.remove_entities(&entities);
        self.fitted.iter_mut().for_each(|f| f.order.retain(|p| !entities.contains(&p.id)));
        self.selection.clear();
        self.edit_dim = None;
        self.solve();
    }

    /// Delete / Backspace: the selected relation or dimension, else the selected pieces.
    pub fn delete_key(&mut self) {
        if let Some(id) = self.selected_constraint() {
            self.remove_constraint(id);
        } else {
            self.delete_selection();
        }
    }

    pub fn remove_constraint(&mut self, id: ConstraintId) {
        self.checkpoint();
        self.sketch.remove_constraint(id);
        self.selection.retain(|p| *p != Pick::Constraint(id));
        if self.edit_dim.as_ref().is_some_and(|d| d.0 == id) {
            self.edit_dim = None;
        }
        self.solve();
    }

    /// Set a dimension's value (typed); `R6` or `Ø12` turns a round piece's dimension over.
    pub fn set_dimension(&mut self, id: ConstraintId, text: &str) -> bool {
        let t = text.trim();
        let (diameter, number) = if let Some(rest) = t.strip_prefix('Ø').or_else(|| t.strip_prefix('ø')) {
            (Some(true), rest)
        } else if let Some(rest) = t.strip_prefix('R').or_else(|| t.strip_prefix('r')) {
            (Some(false), rest)
        } else {
            (None, t)
        };
        let Ok(v) = number.trim().trim_end_matches("mm").trim_end_matches('°').trim().parse::<f64>() else {
            self.message = format!("“{t}” is not a number.");
            return false;
        };
        if !(v.is_finite()) || (v <= 0.0 && !matches!(self.sketch.constraint(id).map(|c| &c.rule), Some(Rule::Angle { .. }))) {
            self.message = "A dimension has to be more than zero.".into();
            return false;
        }
        self.checkpoint();
        if let Some(c) = self.sketch.constraint_mut(id) {
            c.rule = match (c.rule.clone(), diameter) {
                (Rule::Radius { entity, .. }, Some(true)) => Rule::Diameter { entity, value: v },
                (Rule::Diameter { entity, .. }, Some(false)) => Rule::Radius { entity, value: v },
                (mut r, _) => {
                    r.set_value(v);
                    r
                }
            };
        }
        self.solve();
        if let Some((_, r)) = self.report.unmet.iter().find(|u| u.0 == id) {
            self.message = format!("That dimension cannot be met with the others (off by {r:.3} mm).");
        }
        true
    }

    /// A round piece's dimension given the other way — by its diameter, or its radius — the
    /// size held (ScanRuler's `setDimensionForm`).
    pub fn flip_dimension_form(&mut self, id: ConstraintId) {
        let Some(rule) = self.sketch.constraint(id).map(|c| c.rule.clone()) else { return };
        let flipped = match rule {
            Rule::Radius { entity, value } => Rule::Diameter { entity, value: 2.0 * value },
            Rule::Diameter { entity, value } => Rule::Radius { entity, value: value / 2.0 },
            _ => return,
        };
        self.checkpoint();
        if let Some(c) = self.sketch.constraint_mut(id) {
            c.rule = flipped;
        }
        self.solve();
    }

    /// Dimension what is selected (the D key, or the Dimension key with pieces selected): a
    /// line's length, a round piece's size, or what lies between two pieces; the number opens
    /// for typing.
    pub fn dimension_selection(&mut self) {
        let ids = self.selected_entities();
        let rule = match ids[..] {
            [a] => dimension_for(&self.sketch, DimPick::Piece(a), None, &self.traces),
            [a, b] => dimension_for(&self.sketch, DimPick::Piece(a), Some(DimPick::Piece(b)), &self.traces),
            _ => None,
        };
        let Some(rule) = rule else {
            self.error = Some("No dimension applies to what is selected — one piece, or two.".into());
            return;
        };
        self.tool_idle();
        self.tool = Tool::Dimension;
        self.add_dimension(rule);
    }

    /// A dimension added at what it measures, opened for typing.
    fn add_dimension(&mut self, mut rule: Rule) {
        let Some(value) = measure_rule(&self.sketch, &rule, &self.traces) else { return };
        rule.set_value(round_value(value, matches!(rule, Rule::Angle { .. })));
        self.checkpoint();
        let id = self.sketch.constrain(rule);
        self.solve();
        let shown = self.sketch.constraint(id).and_then(|c| c.rule.value()).unwrap_or(value);
        self.selection = vec![Pick::Constraint(id)];
        self.edit_dim = Some((id, fmt_value(shown)));
        self.error = None;
    }

    // ---- picking and snapping ----

    /// The reach of the hand on the sheet, millimetres.
    pub fn reach(&self) -> f64 {
        SNAP_PX * self.mm_per_px
    }

    /// The pick under a sheet point: an end or a centre first, then a piece — what the
    /// Dimension tool clicks.
    pub fn pick_at(&self, p: DVec2, mm_per_px: f64) -> Option<Pick> {
        let r = SNAP_PX * mm_per_px;
        let mut best: Option<(Pick, f64)> = None;
        for e in &self.sketch.entities {
            let anchors: &[AnchorAt] = match e.shape {
                Shape::Line { .. } | Shape::Arc { .. } => &[AnchorAt::A, AnchorAt::B, AnchorAt::Center],
                _ => &[AnchorAt::Center],
            };
            for &at in anchors {
                if matches!(e.shape, Shape::Line { .. }) && at == AnchorAt::Center {
                    continue;
                }
                let d = (anchor_point(&e.shape, at) - p).length();
                if d <= r && best.is_none_or(|b| d < b.1) {
                    let pick = if matches!(e.shape, Shape::Point { .. }) { Pick::Entity(e.id) } else { Pick::Anchor(Anchor::new(e.id, at)) };
                    best = Some((pick, d));
                }
            }
        }
        if best.is_some() {
            return best.map(|b| b.0);
        }
        for e in &self.sketch.entities {
            let d = nearest_on(&e.shape, p).1;
            if d <= r && best.is_none_or(|b| d < b.1) {
                best = Some((Pick::Entity(e.id), d));
            }
        }
        best.map(|b| b.0)
    }

    /// What a click or a drag may land on besides the sketch: the slice chains, the grid when
    /// grid snapping is on, the part's coordinate planes — or nothing, with Ctrl held.
    fn snap_at(&self, spot: DVec2, raw: bool, exclude: Option<&HashSet<EntityId>>) -> SnapResult {
        let grid = self.prefs.grid_snap.then(|| grid_spacing(self.mm_per_px));
        let ctx = SnapContext { chains: Some(&self.chains), grid, datums: Some(&self.traces), off: raw };
        snap_sketch_point(&self.sketch, spot, self.reach(), &ctx, exclude)
    }

    // ---- the pointer ----

    /// One frame of pointer input. Returns whether the sketch changed.
    pub fn pointer(&mut self, input: &SketchInput) -> bool {
        let Some(pos) = input.pos else {
            // A row of the list pointed at in the panel stays lit.
            if !matches!(self.hover, Some(Pick::Constraint(_))) {
                self.hover = None;
            }
            self.cursor = None;
            self.cursor_at = None;
            self.lit = SketchHover::default();
            return false;
        };
        self.mm_per_px = input.mm_per_px.max(1e-9);
        self.raw = input.exact;
        self.cursor_at = Some(pos);
        let mut changed = false;
        if input.pressed {
            self.press(pos, input);
        }
        let mut gestured = false;
        if input.down {
            changed |= self.drag_move(pos, input);
        }
        if input.released || !input.down {
            let (done, ch) = self.release(input);
            gestured = done;
            changed |= ch;
        }
        if input.clicked && !gestured {
            changed |= self.sheet_click(pos, input.shift, input.exact);
        }
        self.update_hover(pos);
        changed
    }

    /// A press on the sheet: the chain's last end with the Line tool (a tangent arc, if
    /// dragged), a piece to drag with the Select tool, else the corner of a selection box.
    fn press(&mut self, pos: DVec2, input: &SketchInput) {
        if self.chain_arc_start(pos, self.reach()).is_some() {
            self.arc_drag = Some(ArcDrag { press: input.screen, moved: false, at: pos, raw: input.exact });
            return;
        }
        if self.tool != Tool::Select || !self.pending.is_empty() || self.image_pick.is_some() {
            return;
        }
        if self.drag_start(pos, input.screen) {
            return;
        }
        if let Some(screen) = input.screen {
            self.boxing = Some(BoxDrag { start_screen: screen, moved: false, corners: None, crossing: false, ids: Vec::new(), screen: None });
        }
    }

    /// A press the sketch may take as a drag of a piece (ScanRuler's `sketchDragStart`): an end,
    /// a centre or a piece under the hand, not fixed. Nothing changes yet: a press let go where
    /// it landed is a click, and the click is what selects.
    fn drag_start(&mut self, pos: DVec2, press: Option<Pos2>) -> bool {
        let Some((entity, at)) = drag_handle_at(&self.sketch, pos, self.reach()) else { return false };
        let Some(e) = self.sketch.entity(entity) else { return false };
        if e.fixed {
            return false;
        }
        let rigid = rigid_cluster(&self.sketch, entity, at).map(|ids| ids.iter().filter_map(|id| self.sketch.entity(*id).cloned()).collect());
        self.drag = Some(Drag { entity, at, start: pos, press, origin: detached(e), snapped: None, pull: Vec::new(), begun: false, rigid });
        true
    }

    /// The hand moved with the button down.
    fn drag_move(&mut self, pos: DVec2, input: &SketchInput) -> bool {
        let far = |press: Option<Pos2>, limit: f32| match (press, input.screen) {
            (Some(a), Some(b)) => (b - a).length() >= limit,
            _ => true,
        };
        if let Some(arc) = &mut self.arc_drag {
            if arc.moved || far(arc.press, 3.0) {
                arc.moved = true;
                arc.at = pos;
                arc.raw = input.exact;
            }
            return false;
        }
        if let Some(d) = &self.drag {
            if d.begun || far(d.press, 3.0) {
                return self.sketch_drag(pos, input.exact);
            }
            return false;
        }
        if let (Some(b), Some(screen)) = (&mut self.boxing, input.screen) {
            let d = screen - b.start_screen;
            if !b.moved && d.x.abs() + d.y.abs() <= 6.0 {
                return false;
            }
            b.moved = true;
            b.crossing = screen.x < b.start_screen.x;
            b.screen = Some((b.start_screen, screen));
            let corners = input.unproject.and_then(|f| Some([f(b.start_screen)?, f(egui::pos2(screen.x, b.start_screen.y))?, f(screen)?, f(egui::pos2(b.start_screen.x, screen.y))?]));
            b.corners = corners;
            b.ids = corners.map(|c| pieces_in_box(&self.sketch, &c, b.crossing)).unwrap_or_default();
        }
        false
    }

    /// The button let go: a drag, a tangent arc or a box finished. Returns whether a gesture
    /// took the release (so it is no click), and whether the sketch changed.
    fn release(&mut self, input: &SketchInput) -> (bool, bool) {
        if let Some(arc) = self.arc_drag.take() {
            if arc.moved {
                self.tangent_arc_to(arc.at, arc.raw);
                return (true, true);
            }
            return (false, false);
        }
        if let Some(d) = self.drag.take() {
            if d.begun {
                self.sketch_drag_end(d);
                return (true, true);
            }
            return (false, false);
        }
        if let Some(b) = self.boxing.take() {
            if b.moved {
                if let Some(corners) = b.corners {
                    let taken = pieces_in_box(&self.sketch, &corners, b.crossing);
                    let additive = input.shift || input.exact;
                    if !additive {
                        self.selection.clear();
                    }
                    for id in taken {
                        if !self.selection.contains(&Pick::Entity(id)) {
                            self.selection.push(Pick::Entity(id));
                        }
                    }
                    self.selection.retain(|p| !matches!(p, Pick::Constraint(_)));
                    self.error = None;
                }
                return (true, false);
            }
        }
        (false, false)
    }

    /// A click on the sheet, by the tool in hand (ScanRuler's `sketchClick`).
    pub fn sheet_click(&mut self, spot: DVec2, additive: bool, raw: bool) -> bool {
        let reach = self.reach();
        // Points clicked on the picture under the sketch: two, then it is moved or levelled —
        // or, to scale it, the length between them asked for.
        if let (Some(pick), Some(image)) = (self.image_pick.clone(), self.image.clone()) {
            let at = if pick.mode == ImagePickMode::Move && pick.points.len() == 1 { self.snap_at(spot, raw, None).point } else { spot };
            let points = if pick.points.len() >= 2 { vec![at] } else { [pick.points.clone(), vec![at]].concat() };
            if points.len() < 2 || pick.mode == ImagePickMode::Scale {
                self.image_pick = Some(ImagePick { mode: pick.mode, points });
                self.error = None;
                return false;
            }
            let moved = if pick.mode == ImagePickMode::Move { Some(move_image(&image, points[0], points[1])) } else { level_image(&image, points[0], points[1]) };
            self.error = if moved.is_some() { None } else { Some("The two points are one — click two points apart.".into()) };
            self.image = Some(moved.unwrap_or(image));
            self.image_pick = None;
            return true;
        }
        match self.tool {
            Tool::Project => self.project_click(spot, reach),
            Tool::Relation(RelationKind::Coincident) => self.coincident_click(spot, reach),
            Tool::Relation(kind) => self.relation_click(kind, spot, reach),
            Tool::Dimension => {
                self.dimension_click(spot, self.mm_per_px);
                true
            }
            Tool::Fillet | Tool::Chamfer => self.corner_click(spot, reach),
            Tool::Select => {
                self.select_click(spot, reach, additive);
                false
            }
            _ => self.draw_click(spot, raw),
        }
    }

    /// The Select tool's click: a piece selected, a second paired with it; Shift adds; a click
    /// on a selected piece lets it go; on empty sheet the selection goes.
    fn select_click(&mut self, spot: DVec2, reach: f64, additive: bool) {
        let Some(hit) = piece_near(&self.sketch, spot, reach) else {
            self.selection.clear();
            self.error = None;
            return;
        };
        self.selection.retain(|p| !matches!(p, Pick::Constraint(_) | Pick::Anchor(_)));
        let pick = Pick::Entity(hit);
        if let Some(i) = self.selection.iter().position(|q| *q == pick) {
            self.selection.remove(i);
        } else if additive || self.selection.len() < 2 {
            self.selection.push(pick);
        } else {
            self.selection = vec![pick];
        }
        self.error = None;
    }

    /// A relation in hand: the pieces clicked are collected until it has what it takes — one
    /// line for level or upright, two pieces otherwise, three for Symmetric — and it stays in
    /// hand for the next.
    fn relation_click(&mut self, kind: RelationKind, spot: DVec2, reach: f64) -> bool {
        let Some(hit) = piece_near(&self.sketch, spot, reach) else {
            // Symmetric's mirror line may be one of the part's coordinate planes, clicked on
            // the line it cuts the sheet in — where no piece of the sketch is.
            let ids = self.selected_entities();
            let trace = (kind == RelationKind::Symmetric && ids.len() == 2).then(|| trace_near(&self.traces, spot, reach)).flatten();
            let a = ids.first().and_then(|id| self.sketch.entity(*id).cloned());
            let b = ids.get(1).and_then(|id| self.sketch.entity(*id).cloned());
            if let (Some(trace), Some(a), Some(b)) = (trace, a, b) {
                self.hold_symmetric(&a, &b, MirrorLine::Datum(&trace));
                self.selection.clear();
                return true;
            }
            self.selection.clear();
            self.error = None;
            return false;
        };
        if self.selection.contains(&Pick::Entity(hit)) {
            return false;
        }
        self.selection.retain(|p| matches!(p, Pick::Entity(_)));
        self.selection.push(Pick::Entity(hit));
        self.error = None;
        let need = match kind {
            RelationKind::Horizontal | RelationKind::Vertical => 1,
            RelationKind::Symmetric => 3,
            _ => 2,
        };
        if self.selection.len() < need {
            return false;
        }
        self.add_relation(kind);
        self.tool = Tool::Relation(kind);
        self.selection.clear();
        true
    }

    /// Coincident in hand takes what CAD takes: an end, a centre, a piece or a coordinate
    /// plane's line, then a second — two points meet, a point lies on a piece or a plane's
    /// line, a line lies along a line or a plane's line. It stays in hand.
    fn coincident_click(&mut self, spot: DVec2, reach: f64) -> bool {
        let Some(pick) = rel_pick_at(&self.sketch, spot, reach, &self.traces) else {
            self.rel_picks.clear();
            self.selection.clear();
            self.error = None;
            return false;
        };
        let Some(first) = self.rel_picks.first().copied() else {
            self.rel_picks = vec![pick];
            self.selection.clear();
            self.error = None;
            return false;
        };
        self.rel_picks.clear();
        match coincident_for(&self.sketch, first, pick) {
            Err(why) => {
                self.error = Some(why);
                false
            }
            Ok(rule) => {
                self.checkpoint();
                let id = self.sketch.constrain(rule);
                self.solve();
                self.select_constraint(id);
                true
            }
        }
    }

    /// A corner tool's click: a corner clicked is rounded or cut there and then; otherwise two
    /// lines are collected.
    fn corner_click(&mut self, spot: DVec2, reach: f64) -> bool {
        let corner = anchor_near(&self.sketch, spot, reach, None);
        if let Some([a, b]) = corner.and_then(|(anchor, _)| corner_lines_at(&self.sketch, anchor)) {
            return self.corner_draft(a, b);
        }
        let Some(hit) = piece_near(&self.sketch, spot, reach) else {
            self.selection.clear();
            self.error = None;
            return false;
        };
        if self.sketch.entity(hit).map(|e| e.kind()) != Some(ShapeKind::Line) {
            self.error = Some(format!("A {} goes between two lines — click a corner, or one line and then the other.", if self.tool == Tool::Fillet { "round" } else { "chamfer" }));
            return false;
        }
        if self.selection.contains(&Pick::Entity(hit)) {
            return false;
        }
        let ids = self.selected_entities();
        match ids[..] {
            [first] => self.corner_draft(first, hit),
            _ => {
                self.selection = vec![Pick::Entity(hit)];
                self.error = None;
                false
            }
        }
    }

    fn corner_draft(&mut self, a: EntityId, b: EntityId) -> bool {
        if self.tool == Tool::Chamfer {
            self.chamfer_draft(a, b)
        } else {
            self.fillet_draft(a, b)
        }
    }

    /// The corner between two lines rounded: the radius dimension comes out open for typing,
    /// and the last radius typed in this sketch is the one the round starts with.
    pub fn fillet_draft(&mut self, a: EntityId, b: EntityId) -> bool {
        let before = self.sketch.clone();
        let mut last = None;
        for c in &self.sketch.constraints {
            if let Rule::Radius { entity, value } = c.rule {
                if self.sketch.entity(entity).map(|e| e.kind()) == Some(ShapeKind::Arc) {
                    last = Some(value);
                }
            }
        }
        // A last radius that does not fit this corner gives way to the corner's own.
        let mut made = fillet_lines(&mut self.sketch, a, b, last, None);
        if made.is_err() && last.is_some() {
            made = fillet_lines(&mut self.sketch, a, b, None, None);
        }
        self.selection.clear();
        match made {
            Err(why) => {
                self.error = Some(why);
                false
            }
            Ok(r) => {
                self.push_undo(before);
                self.solve();
                self.error = None;
                self.open_dimension(r.dimension);
                true
            }
        }
    }

    /// The corner between two lines cut off straight: the first of its two distances comes out
    /// open for typing, and the next chamfer starts with the last setback that fitted.
    pub fn chamfer_draft(&mut self, a: EntityId, b: EntityId) -> bool {
        let before = self.sketch.clone();
        let mut made = chamfer_lines(&mut self.sketch, a, b, self.last_chamfer);
        if made.is_err() && self.last_chamfer.is_some() {
            made = chamfer_lines(&mut self.sketch, a, b, None);
        }
        self.selection.clear();
        match made {
            Err(why) => {
                self.error = Some(why);
                false
            }
            Ok(r) => {
                self.push_undo(before);
                self.last_chamfer = Some(r.setback);
                self.solve();
                self.error = None;
                self.open_dimension(r.dimensions[0]);
                true
            }
        }
    }

    fn push_undo(&mut self, before: Sketch) {
        self.undo.push(before);
        self.redo.clear();
    }

    /// A dimension selected with its number open for typing.
    fn open_dimension(&mut self, id: ConstraintId) {
        if let Some(v) = self.sketch.constraint(id).and_then(|c| c.rule.value()) {
            self.selection = vec![Pick::Constraint(id)];
            self.edit_dim = Some((id, fmt_value(v)));
        }
    }

    /// The Project tool's click: the edge or the piece of another sketch nearest it comes into
    /// this sketch, fixed where it lands.
    fn project_click(&mut self, spot: DVec2, reach: f64) -> bool {
        let found = projectable_at(&self.projectables, spot, reach).cloned();
        let Some(found) = found else {
            self.message = "Nothing to project there — click an edge of a body, or a piece of another sketch, that lies flat to this plane or square to it.".into();
            return false;
        };
        self.checkpoint();
        let id = self.sketch.add(found.entity);
        self.selection = vec![Pick::Entity(id)];
        self.solve();
        self.message = format!("Projected {} into the sketch — fixed where it lies; hang relations and dimensions on it.", found.source);
        true
    }

    /// A drawing tool's click: it snaps, and once the tool has its clicks the piece is made,
    /// each point held to what it snapped to.
    fn draw_click(&mut self, spot: DVec2, raw: bool) -> bool {
        let snap = self.snap_at(spot, raw, None);
        let at = snap.point;
        self.pending.push(at);
        self.snaps.push(snap_hold(&snap));
        let modes = self.prefs.draw_modes;
        let need = draw_clicks(self.tool.name(), &modes);
        if self.pending.len() < need {
            self.error = None;
            return false;
        }
        let pending = std::mem::take(&mut self.pending);
        let snaps = std::mem::take(&mut self.snaps);
        self.chain = None;
        let (p, q) = (pending[0], pending[1]);
        let r = pending.get(2).copied().unwrap_or(q);
        // A second click where the first landed makes nothing and puts the points down: it is
        // how a chain of lines is ended on the sheet.
        if (p - q).length() < 1e-6 {
            self.tool_idle();
            self.error = None;
            return false;
        }
        let hold = |i: usize| snaps.get(i).cloned().flatten();
        let before = self.sketch.clone();
        let mut carry: Option<Anchor> = None;
        match self.tool {
            Tool::Line => {
                let id = self.sketch.add_shape(Shape::Line { a: p, b: q });
                // A line that ends on another piece's end, or on the sheet's zero, has arrived;
                // one that ends on the open sheet, on an axis or on a piece's body is the start
                // of the next, as CAD chains them.
                if matches!(hold(1), None | Some(SnapHold::Datum(_)) | Some(SnapHold::Piece(_))) {
                    carry = Some(Anchor::new(id, AnchorAt::B));
                }
                if let Some(h) = hold(0) {
                    hold_at(&mut self.sketch, Anchor::new(id, AnchorAt::A), &h);
                }
                if let Some(h) = hold(1) {
                    hold_at(&mut self.sketch, Anchor::new(id, AnchorAt::B), &h);
                }
                // A line drawn nearly level or nearly upright was meant so: it gets the
                // relation, shown as a chip that can be taken off again.
                if let Some(axis) = axis_relation_of(p, q, INFER_AXIS_DEG) {
                    self.sketch.infer(axis.rule(id));
                }
            }
            Tool::Rectangle => {
                let mode = modes.rectangle;
                let Some(corners) = rectangle_corners(mode, &pending) else {
                    self.tool_idle();
                    self.error = Some("That rectangle has no area — the clicks lie on a line.".into());
                    return false;
                };
                let ids: Vec<EntityId> = if mode == RectangleMode::ThreePoint { self.sketch.add_turned_rectangle(corners).to_vec() } else { self.sketch.add_rectangle(corners[0], corners[2]).to_vec() };
                // The corners the clicks landed on are held where they snapped; each click's
                // corner is found again by where it lies.
                let corner_of = |s: &Sketch, pt: DVec2| ids.iter().copied().find(|i| matches!(s.entity(*i).map(|e| e.shape.clone()), Some(Shape::Line { a, .. }) if (a - pt).length() < 1e-9));
                let held: Vec<(DVec2, Option<SnapHold>)> = if mode == RectangleMode::Center { vec![(q, hold(1))] } else { vec![(p, hold(0)), (q, hold(1))] };
                for (pt, h) in held {
                    if let (Some(h), Some(id)) = (h, corner_of(&self.sketch, pt)) {
                        hold_at(&mut self.sketch, Anchor::new(id, AnchorAt::A), &h);
                    }
                }
                if mode == RectangleMode::Center {
                    // The centre is a point of its own where the diagonals cross, held where
                    // the first click snapped: the rectangle stays about it.
                    let centre = add_rectangle_centre(&mut self.sketch, &ids);
                    if let Some(h) = hold(0) {
                        hold_at(&mut self.sketch, Anchor::new(centre, AnchorAt::Center), &h);
                    }
                }
                // The third click of the three-point way says where the far side lies; on
                // another piece's end, the far side runs through it.
                if mode == RectangleMode::ThreePoint {
                    if let Some(SnapHold::Anchor(a)) = hold(2) {
                        self.sketch.constrain(Rule::PointOn { p: a, on: OnTarget::Line(ids[2]), bounded: false });
                    }
                }
            }
            Tool::Polygon => {
                let Some(made) = add_polygon(&mut self.sketch, p, q, self.prefs.polygon_sides, modes.polygon) else {
                    self.tool_idle();
                    self.error = Some("That polygon has no size — click the centre, then a corner or the middle of a side.".into());
                    return false;
                };
                // The centre is held where it snapped, as a circle's is; a corner put on
                // another piece's end goes through it.
                if let Some(h) = hold(0) {
                    hold_at(&mut self.sketch, Anchor::new(made.circle, AnchorAt::Center), &h);
                }
                if let (Some(SnapHold::Anchor(a)), PolygonFit::Inscribed) = (hold(1), modes.polygon) {
                    self.sketch.constrain(Rule::Coincident { p: a, q: Anchor::new(made.lines[0], AnchorAt::A) });
                }
            }
            Tool::Circle => {
                let mode = modes.circle;
                let Some((c, radius)) = circle_from(mode, &pending) else {
                    self.tool_idle();
                    self.error = Some("The three points lie on a line — no circle runs through them.".into());
                    return false;
                };
                let id = self.sketch.add_shape(Shape::Circle { c, r: radius });
                // A centre placed on an end, a centre, an axis or a piece is held there; a
                // point of the rim placed on another piece's end runs through it.
                if mode == CircleMode::Center {
                    if let Some(h) = hold(0) {
                        hold_at(&mut self.sketch, Anchor::new(id, AnchorAt::Center), &h);
                    }
                } else {
                    for h in snaps.iter().flatten() {
                        if let SnapHold::Anchor(a) = h {
                            self.sketch.constrain(Rule::PointOn { p: *a, on: OnTarget::Curve(id), bounded: false });
                        }
                    }
                }
            }
            Tool::Arc if modes.arc == ArcMode::Center => {
                // The centre, the start, and where it ends: the end click gives the direction
                // only, the start the radius.
                let Some(arc) = arc_about(p, q, r) else {
                    self.tool_idle();
                    self.error = Some("The start lies on the centre — the arc has no radius.".into());
                    return false;
                };
                let id = self.sketch.add_shape(arc.shape());
                if let Some(h) = hold(0) {
                    hold_at(&mut self.sketch, Anchor::new(id, AnchorAt::Center), &h);
                }
                if let Some(h) = hold(1) {
                    hold_at(&mut self.sketch, Anchor::new(id, if arc.starts_at_p { AnchorAt::A } else { AnchorAt::B }), &h);
                }
            }
            Tool::Arc => {
                // Three points: start, end, and one the arc passes through.
                let Some((c, radius, start, sweep, at_p)) = arc_through(p, q, r) else {
                    self.tool_idle();
                    self.error = Some("The three points lie on a line — no arc runs through them.".into());
                    return false;
                };
                let id = self.sketch.add_shape(Shape::Arc { c, r: radius, start, sweep });
                let (ap, aq) = if at_p { (AnchorAt::A, AnchorAt::B) } else { (AnchorAt::B, AnchorAt::A) };
                if let Some(h) = hold(0) {
                    hold_at(&mut self.sketch, Anchor::new(id, ap), &h);
                }
                if let Some(h) = hold(1) {
                    hold_at(&mut self.sketch, Anchor::new(id, aq), &h);
                }
            }
            _ => return false,
        }
        self.push_undo(before);
        self.tool_idle();
        self.solve();
        self.error = None;
        // From where the solver left the end, not from where it was clicked: a line made level
        // has moved under the hand.
        if let Some(carry) = carry {
            if let Some(end) = self.sketch.anchor(carry) {
                self.pending = vec![end];
                self.snaps = vec![Some(SnapHold::Anchor(carry))];
                self.chain = Some(carry);
            }
        }
        true
    }

    /// Where a tangent arc would start under a press with the Line tool: the chain's last end,
    /// with the direction the piece before it runs on in — or `None` when the press is not on
    /// it.
    pub fn chain_arc_start(&self, spot: DVec2, radius: f64) -> Option<(DVec2, DVec2)> {
        if self.tool != Tool::Line || self.pending.len() != 1 {
            return None;
        }
        let chain = self.chain?;
        let e = self.sketch.entity(chain.entity)?;
        let t = outward_tangent(e, chain.at)?;
        let p = anchor_point(&e.shape, chain.at);
        ((p - spot).length() <= radius).then_some((p, t))
    }

    /// The tangent arc the Line tool's drag from the chain's last end would make, to `spot`
    /// snapped — for the sheet to draw while the hand holds it.
    pub fn tangent_arc_preview(&self) -> Option<(Shape, SnapResult)> {
        let arc = self.arc_drag.filter(|a| a.moved)?;
        let (p, t) = self.chain_arc_start(self.pending.first().copied()?, f64::INFINITY)?;
        let snap = self.snap_at(arc.at, arc.raw, None);
        let placed = tangent_arc(p, t, snap.point)?;
        Some((placed.shape(), snap))
    }

    /// The tangent arc made: from the chain's last end, tangent to the piece before it, to where
    /// the hand let go (ScanRuler's `sketchTangentArc`).
    fn tangent_arc_to(&mut self, spot: DVec2, raw: bool) {
        let Some(chain) = self.chain else { return };
        let Some(start) = self.pending.first().copied() else { return };
        let Some((p, t)) = self.chain_arc_start(start, f64::INFINITY) else { return };
        let snap = self.snap_at(spot, raw, None);
        let Some(arc) = tangent_arc(p, t, snap.point) else {
            self.error = Some("Straight ahead no arc is tangent — drag to one side.".into());
            return;
        };
        self.checkpoint();
        let id = self.sketch.add_shape(arc.shape());
        let (near, far) = if arc.starts_at_p { (AnchorAt::A, AnchorAt::B) } else { (AnchorAt::B, AnchorAt::A) };
        self.sketch.constrain(Rule::Coincident { p: Anchor::new(id, near), q: chain });
        self.sketch.constrain(Rule::Tangent { a: chain.entity, b: id });
        if let Some(h) = snap_hold(&snap) {
            hold_at(&mut self.sketch, Anchor::new(id, far), &h);
        }
        self.tool_idle();
        self.solve();
        self.error = None;
        // The chain runs on from the arc's far end, unless it landed on a piece's end.
        if snap.anchor.is_none() {
            let carry = Anchor::new(id, far);
            if let Some(end) = self.sketch.anchor(carry) {
                self.pending = vec![end];
                self.snaps = vec![Some(SnapHold::Anchor(carry))];
                self.chain = Some(carry);
            }
        }
    }

    /// The hand moved with a piece in it (ScanRuler's `sketchDrag`).
    fn sketch_drag(&mut self, spot: DVec2, raw: bool) -> bool {
        let Some(mut drag) = self.drag.take() else { return false };
        if !drag.begun {
            // The sketch as it stands goes into the history once, with the first move: the
            // drag is one edit however many moves it takes, and the piece in the hand is the
            // selection.
            self.checkpoint();
            self.selection = vec![Pick::Entity(drag.entity)];
            self.error = None;
            drag.begun = true;
        }
        let Some(e) = self.sketch.entity(drag.entity).cloned() else {
            self.drag = Some(drag);
            return false;
        };
        let mut pull = Vec::new();
        if let Some(rigid) = &drag.rigid {
            // The whole shape goes with the hand, as it is: by the hand's move for a piece
            // carried, by where the end or centre snaps for a point.
            let delta = match drag.at {
                GrabAt::Body => {
                    drag.snapped = None;
                    spot - drag.start
                }
                GrabAt::Anchor(at) => {
                    let exclude: HashSet<EntityId> = rigid.iter().map(|x| x.id).collect();
                    let snap = self.snap_at(spot, raw, Some(&exclude));
                    drag.snapped = snap_hold(&snap);
                    snap.point - anchor_point(&drag.origin.shape, at)
                }
            };
            for x in rigid {
                let carried = translate_entity(x, delta);
                if let Some(slot) = self.sketch.entity_mut(x.id) {
                    *slot = carried;
                }
            }
            if let Some(held) = self.sketch.entity(drag.entity) {
                let ats: Vec<AnchorAt> = match drag.at {
                    GrabAt::Body => anchors_of(&held.shape).to_vec(),
                    GrabAt::Anchor(at) => vec![at],
                };
                for at in ats {
                    pull.push(Pull { anchor: Anchor::new(held.id, at), to: anchor_point(&held.shape, at) });
                }
            }
        } else {
            let moved = match drag.at {
                GrabAt::Body => {
                    // The whole piece rides with the hand, raw. A line moves square to itself —
                    // along its own length there is nothing to move it to — unless Ctrl asks
                    // for it free. From where the drag began, not from the last step.
                    let mut delta = spot - drag.start;
                    if let (Shape::Line { a, b }, false) = (e.shape, raw) {
                        if let Some(dir) = (b - a).try_normalize() {
                            let n = dir.perp();
                            delta = n * delta.dot(n);
                        }
                    }
                    drag.snapped = None;
                    let moved = translate_entity(&drag.origin, delta);
                    for &at in anchors_of(&moved.shape) {
                        pull.push(Pull { anchor: Anchor::new(moved.id, at), to: anchor_point(&moved.shape, at) });
                    }
                    moved
                }
                GrabAt::Anchor(at) => {
                    // An end or a centre lands on what is near — another end, a slice point,
                    // the grid — the hand's piece left out of the search.
                    let exclude = HashSet::from([e.id]);
                    let snap = self.snap_at(spot, raw, Some(&exclude));
                    drag.snapped = snap_hold(&snap);
                    pull.push(Pull { anchor: Anchor::new(e.id, at), to: snap.point });
                    move_anchor(&detached(&e), at, snap.point)
                }
            };
            let id = moved.id;
            if let Some(slot) = self.sketch.entity_mut(id) {
                *slot = moved;
            }
            // The piece dragged out of its slice points is free now; the fitted chain it
            // belonged to is left as it is.
            for f in &mut self.fitted {
                f.order.retain(|p| p.id != id);
            }
        }
        drag.pull = pull.clone();
        self.drag = Some(drag);
        self.solve_quick(&pull);
        true
    }

    /// The hand let go of a dragged piece: dropped on another piece's end, its body, an axis
    /// or the zero, it is held there; settled once more in full, the hand still on it.
    fn sketch_drag_end(&mut self, drag: Drag) {
        if let (GrabAt::Anchor(at), Some(h)) = (drag.at, &drag.snapped) {
            hold_at(&mut self.sketch, Anchor::new(drag.entity, at), h);
        }
        self.solve_pulled(&drag.pull);
    }

    /// What a click would take, lit under the cursor before the click (ScanRuler's
    /// `sketchHoverAt`), and where a drawing tool's click would land.
    fn update_hover(&mut self, pos: DVec2) {
        let reach = self.reach();
        self.hover = if self.drag.as_ref().is_some_and(|d| d.begun) { self.hover } else { None };
        self.cursor = (!matches!(self.tool, Tool::Select | Tool::Project) && !self.tool.picks()).then(|| self.snap_at(pos, self.raw, None));
        let piece = |id: EntityId| SketchHover { pieces: vec![id], ..Default::default() };
        let boxing = self.boxing.as_ref().is_some_and(|b| b.moved);
        self.lit = if boxing || self.arc_drag.is_some_and(|a| a.moved) {
            SketchHover::default()
        } else {
            match self.tool {
                Tool::Select => {
                    if self.pending.is_empty() {
                        piece_near(&self.sketch, pos, reach).map(piece).unwrap_or_default()
                    } else {
                        SketchHover::default()
                    }
                }
                Tool::Dimension => match self.pick_at(pos, self.mm_per_px) {
                    Some(Pick::Entity(id)) => piece(id),
                    Some(Pick::Anchor(a)) => SketchHover { point: self.sketch.anchor(a), ..Default::default() },
                    _ => trace_near(&self.traces, pos, reach).map(|t| SketchHover { datum: Some(t.plane), ..Default::default() }).unwrap_or_default(),
                },
                Tool::Relation(RelationKind::Coincident) => self.from_pick(rel_pick_at(&self.sketch, pos, reach, &self.traces)),
                Tool::Relation(kind) => match piece_near(&self.sketch, pos, reach) {
                    Some(id) => piece(id),
                    None if kind == RelationKind::Symmetric && self.selected_entities().len() == 2 => trace_near(&self.traces, pos, reach).map(|t| SketchHover { datum: Some(t.plane), ..Default::default() }).unwrap_or_default(),
                    None => SketchHover::default(),
                },
                Tool::Fillet | Tool::Chamfer => {
                    let corner = anchor_near(&self.sketch, pos, reach, None);
                    match corner.and_then(|(a, p)| corner_lines_at(&self.sketch, a).map(|l| (l, p))) {
                        Some((lines, p)) => SketchHover { pieces: lines.to_vec(), point: Some(p), datum: None },
                        None => match piece_near(&self.sketch, pos, reach) {
                            Some(id) if self.sketch.entity(id).map(|e| e.kind()) == Some(ShapeKind::Line) => piece(id),
                            _ => SketchHover::default(),
                        },
                    }
                }
                _ => SketchHover::default(),
            }
        };
        if let Some(id) = self.lit.pieces.first().filter(|_| self.tool == Tool::Select) {
            self.hover = Some(Pick::Entity(*id));
        }
    }

    /// The cursor over the sheet: a crosshair for a drawing tool, a pointer where a click takes
    /// something, a grabbing hand while a piece is dragged.
    pub fn cursor_icon(&self) -> Option<egui::CursorIcon> {
        if self.drag.as_ref().is_some_and(|d| d.begun) {
            return Some(egui::CursorIcon::Grabbing);
        }
        match self.tool {
            Tool::Select => (!self.lit.pieces.is_empty()).then_some(egui::CursorIcon::PointingHand),
            Tool::Dimension | Tool::Relation(_) | Tool::Fillet | Tool::Chamfer | Tool::Project => Some(egui::CursorIcon::PointingHand),
            _ => Some(egui::CursorIcon::Crosshair),
        }
    }

    fn from_pick(&self, pick: Option<DimPick>) -> SketchHover {
        match pick {
            Some(DimPick::Piece(id)) => SketchHover { pieces: vec![id], ..Default::default() },
            Some(DimPick::Datum(plane)) => SketchHover { datum: Some(plane), ..Default::default() },
            Some(DimPick::Point(a)) => SketchHover { point: self.sketch.anchor(a), ..Default::default() },
            None => SketchHover::default(),
        }
    }

    // ---- the Dimension tool (as Polymix had it) ----

    fn dimension_click(&mut self, pos: DVec2, px: f64) {
        let pick = self.pick_at(pos, px);
        let Some(first) = self.dim_first else {
            match pick {
                Some(p @ (Pick::Entity(_) | Pick::Anchor(_))) => {
                    self.dim_first = Some(p);
                    self.message = "Click a second piece or end for the distance or angle between, or empty space to dimension this one.".into();
                }
                _ => self.message = "Click a piece or an end to dimension.".into(),
            }
            return;
        };
        self.dim_first = None;
        self.message.clear();
        let shape = |id: EntityId| self.sketch.entity(id).map(|e| e.shape.clone());
        let rule = match (first, pick) {
            (Pick::Entity(id), None) | (Pick::Entity(id), Some(Pick::Entity(_))) if pick.is_none() || pick == Some(Pick::Entity(id)) => match shape(id) {
                Some(Shape::Line { .. }) => Some(Rule::Length { entity: id, value: 0.0 }),
                Some(Shape::Arc { .. }) => Some(Rule::Radius { entity: id, value: 0.0 }),
                Some(Shape::Circle { .. }) => Some(Rule::Diameter { entity: id, value: 0.0 }),
                _ => None,
            },
            (Pick::Entity(a), Some(Pick::Entity(b))) => match (shape(a), shape(b)) {
                (Some(Shape::Line { a: a0, b: a1 }), Some(Shape::Line { a: b0, b: b1 })) => {
                    let da = (a1 - a0).normalize_or_zero();
                    let db = (b1 - b0).normalize_or_zero();
                    if (da.x * db.y - da.y * db.x).abs() < 0.02 {
                        // Parallel lines: the distance between them.
                        Some(Rule::Distance { p: Anchor::new(b, AnchorAt::A), q: DistanceTo::Line(a), value: 0.0 })
                    } else {
                        Some(Rule::Angle { a, b, value: 0.0 })
                    }
                }
                (Some(Shape::Line { .. }), Some(s)) if s.centre().is_some() => Some(Rule::Distance { p: Anchor::new(b, AnchorAt::Center), q: DistanceTo::Line(a), value: 0.0 }),
                (Some(s), Some(Shape::Line { .. })) if s.centre().is_some() => Some(Rule::Distance { p: Anchor::new(a, AnchorAt::Center), q: DistanceTo::Line(b), value: 0.0 }),
                (Some(s), Some(t)) if s.centre().is_some() && t.centre().is_some() => Some(Rule::Distance { p: Anchor::new(a, AnchorAt::Center), q: DistanceTo::Anchor(Anchor::new(b, AnchorAt::Center)), value: 0.0 }),
                _ => None,
            },
            (Pick::Anchor(p), Some(Pick::Anchor(q))) => Some(Rule::Distance { p, q: DistanceTo::Anchor(q), value: 0.0 }),
            (Pick::Anchor(p), Some(Pick::Entity(l))) | (Pick::Entity(l), Some(Pick::Anchor(p))) => match shape(l) {
                Some(Shape::Line { .. }) => Some(Rule::Distance { p, q: DistanceTo::Line(l), value: 0.0 }),
                Some(s) if s.centre().is_some() => Some(Rule::Distance { p, q: DistanceTo::Anchor(Anchor::new(l, AnchorAt::Center)), value: 0.0 }),
                _ => None,
            },
            (Pick::Anchor(p), None) => {
                // An end and the nearest coordinate plane's line: the distance to it.
                let near = mm_reverse::datums::trace_near(&self.traces, pos, 1e9).map(|t| t.plane);
                near.map(|plane| Rule::Distance { p, q: DistanceTo::Datum(plane), value: 0.0 })
            }
            _ => None,
        };
        let Some(rule) = rule else {
            self.message = "Those two cannot be dimensioned against each other.".into();
            return;
        };
        self.add_dimension(rule);
    }

    /// The pieces the Dimension tool picked first, for highlighting.
    pub fn dim_first(&self) -> Option<Pick> {
        self.dim_first
    }

    // ---- Pattern, Align, the picture ----

    /// Open the Pattern box on the pieces selected: a row of three, clear of the pieces by half
    /// their width, on a round number.
    pub fn open_pattern(&mut self) {
        let ids = self.selected_entities();
        if ids.is_empty() {
            self.error = Some("Select the pieces to repeat first.".into());
            return;
        }
        let picked = Sketch { entities: self.sketch.entities.iter().filter(|e| ids.contains(&e.id)).cloned().collect(), ..Default::default() };
        let width = picked.bounds().map_or(0.0, |(lo, hi)| hi.x - lo.x);
        let step = if width > 1e-6 { nominal_of(width * 1.5, mm_reverse::nominal::NominalKind::Length).unwrap_or(width * 1.5) } else { 10.0 };
        self.tool_idle();
        self.tool = Tool::Select;
        self.error = None;
        self.pattern = Some(SketchPatternSpec::Rectangular { step: DVec2::new(step, 0.0), count: 3, step2: None, count2: None });
    }

    /// Make the pattern's copies: each a piece of its own, the relations among the pieces
    /// copied; the copies are selected.
    pub fn apply_pattern(&mut self) {
        let Some(spec) = self.pattern else { return };
        let ids = self.selected_entities();
        if ids.is_empty() {
            self.error = Some("Select the pieces to repeat.".into());
            return;
        }
        let before = self.sketch.clone();
        let copies = pattern_pieces(&mut self.sketch, &ids, &spec);
        self.pattern = None;
        if copies.is_empty() {
            self.error = Some("The pattern makes no copies — give it a count of two or more.".into());
            return;
        }
        self.push_undo(before);
        self.solve();
        self.error = None;
        self.selection = copies.into_iter().flatten().map(Pick::Entity).collect();
    }

    /// The copies the Pattern box would make, as pieces — at most 50 × 50 in a grid, 360 round.
    pub fn pattern_preview(&self) -> Vec<Shape> {
        let Some(spec) = self.pattern else { return Vec::new() };
        let ids = self.selected_entities();
        if ids.is_empty() {
            return Vec::new();
        }
        let mut trial = Sketch { entities: self.sketch.entities.iter().filter(|e| ids.contains(&e.id)).cloned().collect(), constraints: Vec::new(), next_id: self.sketch.next_id };
        let capped = match spec {
            SketchPatternSpec::Rectangular { step, count, step2, count2 } => SketchPatternSpec::Rectangular { step, count: count.min(50), step2, count2: count2.map(|c| c.min(50)) },
            SketchPatternSpec::Circular { centre, count, angle_deg } => SketchPatternSpec::Circular { centre, count: count.min(360), angle_deg },
        };
        let first = trial.next_id;
        pattern_pieces(&mut trial, &ids, &capped);
        trial.entities.iter().filter(|e| e.id >= first && e.kind() != ShapeKind::Point).map(|e| e.shape.clone()).collect()
    }

    /// Align: turn the sheet so the selected line lies along its X axis; the drawing turns back
    /// by the same, every point staying where it is on the part.
    pub fn align_to_selected(&mut self) {
        let picked: Vec<Entity> = self.selected_entities().iter().filter_map(|id| self.sketch.entity(*id).cloned()).collect();
        let line = match picked[..] {
            [ref one] => match one.shape {
                Shape::Line { a, b } => Some((a, b)),
                _ => None,
            },
            _ => None,
        };
        let Some((a, b)) = line else {
            self.error = Some("Select one line to lay the sheet along.".into());
            return;
        };
        let Some(dir) = (b - a).try_normalize() else {
            self.error = Some("The line has no length.".into());
            return;
        };
        let theta = dir.y.atan2(dir.x);
        self.checkpoint();
        rotate_sketch(&mut self.sketch, -theta);
        let (s, c) = theta.sin_cos();
        let u = self.frame.u * c + self.frame.v * s;
        let v = self.frame.normal.cross(u);
        self.frame = SectionFrame { u, v, ..self.frame };
        self.traces = datum_traces(&self.frame);
        // The slice is the same cut, read in the turned sheet.
        let turn = |p: DVec2| DVec2::new(p.x * c + p.y * s, -p.x * s + p.y * c);
        for chain in &mut self.chains {
            for p in chain.iter_mut() {
                *p = turn(*p);
            }
        }
        self.tool_idle();
        self.projectables_key = None;
        self.frame_turned = true;
        self.solve();
        self.error = None;
    }

    /// The picture row's clicks taken up — or put down with `None`.
    pub fn start_image_pick(&mut self, mode: Option<ImagePickMode>) {
        if mode.is_some() && self.image.is_none() {
            self.error = Some("Lay a picture under the sketch first.".into());
            return;
        }
        self.tool_idle();
        self.tool = Tool::Select;
        self.selection.clear();
        self.pattern = None;
        self.image_pick = mode.map(|mode| ImagePick { mode, points: Vec::new() });
        self.error = None;
    }

    /// The picture brought to scale: the two points picked are `length` apart on the part.
    pub fn scale_image(&mut self, length: f64) {
        let (Some(image), Some(pick)) = (&self.image, &self.image_pick) else { return };
        if pick.mode != ImagePickMode::Scale || pick.points.len() < 2 {
            return;
        }
        match scale_image(image, pick.points[0], pick.points[1], length) {
            Some(scaled) => {
                self.image = Some(scaled);
                self.image_pick = None;
                self.error = None;
            }
            None => self.error = Some("The two points are one — click two points apart.".into()),
        }
    }

    /// The hint line's words for the picture's clicks.
    pub fn image_pick_hint(&self) -> Option<&'static str> {
        let pick = self.image_pick.as_ref()?;
        let n = pick.points.len();
        Some(match pick.mode {
            ImagePickMode::Scale => match n {
                0 => "Click a point on the picture whose distance to another you know.",
                1 => "Click the other point.",
                _ => "Type how far apart the two points are on the part.",
            },
            ImagePickMode::Move => {
                if n == 0 {
                    "Click a point on the picture."
                } else {
                    "Click where that point belongs."
                }
            }
            ImagePickMode::Level => {
                if n == 0 {
                    "Click a point on a line of the picture that should lie level."
                } else {
                    "Click another point on that line."
                }
            }
        })
    }

    // ---- what the box reads ----

    /// The relations the sketcher added on its own while drawing.
    pub fn inferred(&self) -> Vec<ConstraintId> {
        self.sketch.constraints.iter().filter(|c| c.inferred && !c.rule.is_dimension()).map(|c| c.id).collect()
    }

    /// What the status window says (ScanRuler's sketch DRO): `Needs input` on an empty sketch,
    /// the relations not met, or `Solved` with the closed regions — and the solver's own words
    /// for the tooltip. The bool says whether it is an alarm.
    pub fn status(&self) -> (String, bool, String) {
        if self.sketch.entities.is_empty() {
            return ("Needs input".into(), false, String::new());
        }
        if !self.report.ok && !self.report.unmet.is_empty() {
            let n = self.report.unmet.len();
            let why: Vec<String> = self.report.unmet.iter().map(|(id, r)| match self.sketch.constraint(*id) {
                Some(c) => format!("{} — off by {r:.3}", constraint_label(&c.rule, &self.sketch)),
                None => format!("#{id}"),
            }).collect();
            return (format!("⚠ {n} relation{} not met", plural(n)), true, why.join("\n"));
        }
        let found = self.regions.len();
        let mut tip = Vec::new();
        tip.push(if self.report.soft_rms > 0.0 { format!("The fitted pieces sit on the slice to {} mm RMS", fmt_value(self.report.soft_rms)) } else { "Every relation and dimension is met".into() });
        if self.report.redundant > 0 {
            tip.push(format!("{} relation{} say nothing new (redundant)", self.report.redundant, plural(self.report.redundant)));
        }
        let loose = self.report.under_constrained.len();
        tip.push(if loose > 0 { format!("{loose} piece{} not fully constrained — drawn in the loose colour", plural(loose)) } else { "Every piece is fully constrained".into() });
        tip.push(format!("{} solver step{}", self.report.iterations, plural(self.report.iterations)));
        let text = if found > 0 { format!("Solved · {found} closed region{}", plural(found)) } else { "Solved · no closed region yet".into() };
        (text, false, tip.join("\n"))
    }
}

// ---- free helpers, as ScanRuler's `reSketchDraft` has them ----

/// The nearest piece to a spot within reach.
pub fn piece_near(sketch: &Sketch, spot: DVec2, radius: f64) -> Option<EntityId> {
    let mut best = None;
    let mut best_d = radius;
    for e in &sketch.entities {
        let d = nearest_on(&e.shape, spot).1;
        if d <= best_d {
            best_d = d;
            best = Some(e.id);
        }
    }
    best
}

/// The nearest anchor of any piece within reach — what a drawn end snaps to; a circle's centre
/// is not one.
pub fn anchor_near(sketch: &Sketch, spot: DVec2, radius: f64, except: Option<EntityId>) -> Option<(Anchor, DVec2)> {
    let mut best = None;
    let mut best_d = radius;
    for e in &sketch.entities {
        if Some(e.id) == except {
            continue;
        }
        for &at in anchors_of(&e.shape) {
            if e.kind() == ShapeKind::Circle && at == AnchorAt::Center {
                continue;
            }
            let p = anchor_point(&e.shape, at);
            let d = (p - spot).length();
            if d <= best_d {
                best_d = d;
                best = Some((Anchor::new(e.id, at), p));
            }
        }
    }
    best
}

/// What lies under a hand on the sheet with the Select tool: an end or a centre to drag, a
/// piece to carry, or nothing.
pub fn drag_handle_at(sketch: &Sketch, spot: DVec2, radius: f64) -> Option<(EntityId, GrabAt)> {
    let mut best = None;
    let mut best_d = radius;
    for e in &sketch.entities {
        for &at in anchors_of(&e.shape) {
            let d = (anchor_point(&e.shape, at) - spot).length();
            if d <= best_d {
                best_d = d;
                best = Some((e.id, GrabAt::Anchor(at)));
            }
        }
    }
    best.or_else(|| piece_near(sketch, spot, radius).map(|id| (id, GrabAt::Body)))
}

/// The two pieces' anchors nearest each other.
pub fn nearest_ends(a: &Entity, b: &Entity) -> Option<(Anchor, Anchor)> {
    if a.kind() == ShapeKind::Circle || b.kind() == ShapeKind::Circle {
        return None;
    }
    let mut best = None;
    let mut best_d = f64::INFINITY;
    for &pa in anchors_of(&a.shape) {
        for &pb in anchors_of(&b.shape) {
            let d = (anchor_point(&a.shape, pa) - anchor_point(&b.shape, pb)).length();
            if d < best_d {
                best_d = d;
                best = Some((Anchor::new(a.id, pa), Anchor::new(b.id, pb)));
            }
        }
    }
    best
}

/// Whether an anchor is already held where a snap would hold it: the relation is not added
/// twice, and an end joined to a piece's end is not held on the piece's body as well.
fn held_by(sketch: &Sketch, p: Anchor, hold: &SnapHold) -> bool {
    let cs = &sketch.constraints;
    match hold {
        SnapHold::Datum(plane) => cs.iter().any(|c| matches!(c.rule, Rule::PointOn { p: q, on: OnTarget::Datum(d), .. } if d == *plane && q == p)),
        SnapHold::Origin(planes) => planes.iter().all(|plane| held_by(sketch, p, &SnapHold::Datum(*plane))),
        SnapHold::Piece(piece) => {
            if p.entity == *piece {
                return true;
            }
            let round = sketch.entity(*piece).map(|e| e.kind()) != Some(ShapeKind::Line);
            // A round piece's centre is not on its rim.
            let on_it = |q: Anchor| q.entity == *piece && !(round && q.at == AnchorAt::Center);
            cs.iter().any(|c| match c.rule {
                Rule::PointOn { p: q, on: OnTarget::Line(l) | OnTarget::Curve(l), .. } => q == p && l == *piece,
                Rule::Coincident { p: a, q: b } => (a == p && on_it(b)) || (b == p && on_it(a)),
                _ => false,
            })
        }
        SnapHold::Anchor(to) => cs.iter().any(|c| matches!(c.rule, Rule::Coincident { p: a, q: b } if (a == p && b == *to) || (b == p && a == *to))),
    }
}

/// Hold a drawn or dragged end where it snapped: to an anchor by a coincidence, to a coordinate
/// plane's line or a piece's body by a point-on relation, to the sheet's zero by one to each
/// plane's line crossing there. Nothing is added where the end is held so already.
pub fn hold_at(sketch: &mut Sketch, p: Anchor, hold: &SnapHold) {
    if held_by(sketch, p, hold) {
        return;
    }
    match hold {
        SnapHold::Datum(plane) => {
            sketch.constrain(Rule::PointOn { p, on: OnTarget::Datum(*plane), bounded: false });
        }
        SnapHold::Origin(planes) => {
            for plane in planes {
                hold_at(sketch, p, &SnapHold::Datum(*plane));
            }
        }
        SnapHold::Piece(piece) => {
            let Some(kind) = sketch.entity(*piece).map(|e| e.kind()) else { return };
            match kind {
                ShapeKind::Line => sketch.constrain(Rule::PointOn { p, on: OnTarget::Line(*piece), bounded: false }),
                ShapeKind::Arc | ShapeKind::Circle | ShapeKind::Spline => sketch.constrain(Rule::PointOn { p, on: OnTarget::Curve(*piece), bounded: false }),
                ShapeKind::Point => sketch.constrain(Rule::Coincident { p, q: Anchor::new(*piece, AnchorAt::Center) }),
            };
        }
        SnapHold::Anchor(to) => {
            sketch.constrain(Rule::Coincident { p, q: *to });
        }
    }
}

/// A piece's name in a row: its kind and its id, `line 3`.
pub fn entity_name(sketch: &Sketch, id: EntityId) -> String {
    match sketch.entity(id) {
        Some(e) => format!(
            "{} {}",
            match e.kind() {
                ShapeKind::Line => "line",
                ShapeKind::Arc => "arc",
                ShapeKind::Circle => "circle",
                ShapeKind::Point => "point",
                ShapeKind::Spline => "spline",
            },
            e.id
        ),
        None => format!("#{id}"),
    }
}

/// A relation's or a dimension's label in the list, as ScanRuler words it (`constraintLabel`).
pub fn constraint_label(rule: &Rule, sketch: &Sketch) -> String {
    let name = |id: EntityId| entity_name(sketch, id);
    let datum = |d: DatumPlane| d.name().to_string();
    match *rule {
        Rule::Coincident { p, q } => format!("Coincident {} · {}", name(p.entity), name(q.entity)),
        Rule::Horizontal { entity } => format!("H {}", name(entity)),
        Rule::Vertical { entity } => format!("V {}", name(entity)),
        Rule::Parallel { a, b } => format!("Parallel {} · {}", name(a), name(b)),
        Rule::Perpendicular { a, b } => format!("Perpendicular {} · {}", name(a), name(b)),
        Rule::Tangent { a, b } => format!("Tangent {} · {}", name(a), name(b)),
        Rule::Concentric { a, b } => format!("Concentric {} · {}", name(a), name(b)),
        Rule::Equal { a, b } => format!("Equal {} · {}", name(a), name(b)),
        Rule::Angle { a, b, .. } => format!("Angle {} · {}", name(a), name(b)),
        Rule::Symmetric { p, q, about } => format!("Symmetric {} · {}{}", name(p.entity), name(q.entity), if let MirrorAxis::Datum(d) = about { format!(" about {}", datum(d)) } else { String::new() }),
        Rule::Length { entity, .. } => format!("L {}", name(entity)),
        Rule::Radius { entity, .. } => format!("R {}", name(entity)),
        Rule::Diameter { entity, .. } => format!("Ø {}", name(entity)),
        Rule::Distance { p, q, .. } => format!(
            "d {} · {}",
            name(p.entity),
            match q {
                DistanceTo::Datum(d) => datum(d),
                DistanceTo::Line(l) => name(l),
                DistanceTo::Anchor(a) => name(a.entity),
            }
        ),
        Rule::PointOn { p, on, .. } => format!(
            "On {} · {}",
            name(p.entity),
            match on {
                OnTarget::Datum(d) => datum(d),
                OnTarget::Line(l) | OnTarget::Curve(l) => name(l),
            }
        ),
        Rule::Collinear { line, on } => format!(
            "Collinear {} · {}",
            name(line),
            match on {
                OnTarget::Datum(d) => datum(d),
                OnTarget::Line(l) | OnTarget::Curve(l) => name(l),
            }
        ),
    }
}

pub fn plural(n: usize) -> &'static str {
    if n == 1 {
        ""
    } else {
        "s"
    }
}

/// The round value a measured number was probably designed at: halves under 10 mm, whole
/// millimetres above, angles to 5° — only when it lies within half a percent (or 0.03).
pub fn round_value(v: f64, angle: bool) -> f64 {
    let step = if angle {
        5.0
    } else if v < 10.0 {
        0.5
    } else {
        1.0
    };
    let r = (v / step).round() * step;
    if r > 0.0 && (r - v).abs() <= (0.005 * v.abs()).max(0.03) {
        r
    } else {
        (v * 1000.0).round() / 1000.0
    }
}

pub fn fmt_value(v: f64) -> String {
    let s = format!("{v:.3}");
    let s = s.trim_end_matches('0').trim_end_matches('.');
    s.to_string()
}

// ---- painting ----

/// Colours of the sketch on the part (the committed sketches, and what the feature boxes
/// pick); the open sketch's sheet is painted in ScanRuler's palette (`sketch_paint`).
pub mod colors {
    use egui::Color32;
    pub const FREE: Color32 = Color32::from_rgb(235, 238, 245);
    pub const CONSTRUCTION: Color32 = Color32::from_rgb(120, 150, 200);
    pub const SELECTED: Color32 = Color32::from_rgb(255, 225, 70);
    pub const HOVER: Color32 = Color32::from_rgb(160, 255, 180);
    pub const UNMET: Color32 = Color32::from_rgb(255, 80, 80);
    pub const DIM: Color32 = Color32::from_rgb(255, 240, 200);
}

/// Where a dimension's number is shown on the sheet.
pub fn dimension_anchor(sketch: &Sketch, rule: &Rule, traces: &[DatumTrace]) -> Option<DVec2> {
    let shape = |id: EntityId| sketch.entity(id).map(|e| e.shape.clone());
    match *rule {
        Rule::Length { entity, .. } => match shape(entity)? {
            Shape::Line { a, b } => Some((a + b) / 2.0),
            _ => None,
        },
        Rule::Radius { entity, .. } | Rule::Diameter { entity, .. } => match shape(entity)? {
            Shape::Arc { c, r, start, sweep } => Some(c + DVec2::new((start + sweep / 2.0).cos(), (start + sweep / 2.0).sin()) * r),
            Shape::Circle { c, r } => Some(c + DVec2::new(0.7071, 0.7071) * r),
            _ => None,
        },
        Rule::Angle { a, b, .. } => match (shape(a)?, shape(b)?) {
            (Shape::Line { a: a0, b: a1 }, Shape::Line { a: b0, b: b1 }) => {
                Some(mm_reverse::corners::line_intersection((a0, a1), (b0, b1)).unwrap_or((a0 + a1 + b0 + b1) / 4.0))
            }
            _ => None,
        },
        Rule::Distance { p, q, .. } => {
            let pp = sketch.anchor(p)?;
            let qq = match q {
                DistanceTo::Anchor(q) => sketch.anchor(q)?,
                DistanceTo::Line(l) => match shape(l)? {
                    Shape::Line { a, b } => {
                        let d = (b - a).normalize_or_zero();
                        a + d * (pp - a).dot(d)
                    }
                    _ => return None,
                },
                DistanceTo::Datum(plane) => mm_reverse::datums::trace_of(traces, plane)?.foot(pp),
            };
            Some((pp + qq) / 2.0)
        }
        _ => None,
    }
}

/// The text a dimension shows.
pub fn dimension_text(rule: &Rule) -> Option<String> {
    Some(match *rule {
        Rule::Radius { value, .. } => format!("R{}", fmt_value(value)),
        Rule::Diameter { value, .. } => format!("Ø{}", fmt_value(value)),
        Rule::Angle { value, .. } => format!("{}°", fmt_value(value)),
        Rule::Length { value, .. } | Rule::Distance { value, .. } => fmt_value(value),
        _ => return None,
    })
}

/// Paint every dimension's number and return where each was drawn (for clicks).
pub fn paint_dimensions(edit: &SketchEdit, painter: &egui::Painter, to_screen: &dyn Fn(DVec2) -> Option<Pos2>) -> Vec<(ConstraintId, egui::Rect)> {
    let mut out = Vec::new();
    for c in &edit.sketch.constraints {
        let (Some(text), Some(at)) = (dimension_text(&c.rule), dimension_anchor(&edit.sketch, &c.rule, &edit.traces)) else { continue };
        let Some(p) = to_screen(at) else { continue };
        let unmet = edit.report.unmet.iter().any(|u| u.0 == c.id);
        let lit = edit.selection.contains(&Pick::Constraint(c.id)) || edit.hover == Some(Pick::Constraint(c.id));
        let color = if unmet {
            colors::UNMET
        } else if lit {
            colors::SELECTED
        } else {
            colors::DIM
        };
        let galley = painter.layout_no_wrap(text, egui::FontId::proportional(12.5), color);
        let rect = egui::Rect::from_center_size(p + egui::vec2(0.0, -12.0), galley.size() + egui::vec2(8.0, 4.0));
        painter.rect_filled(rect, 3.0, egui::Color32::from_black_alpha(170));
        painter.galley(rect.min + egui::vec2(4.0, 2.0), galley, color);
        out.push((c.id, rect));
    }
    out
}


#[cfg(test)]
mod tests {
    use super::*;
    use mm_reverse::frame::WorldAxis;

    fn edit() -> SketchEdit {
        let plane = PlaneRef::world(WorldAxis::Z, 0.0);
        let frame = plane.own_frame().expect("a coordinate plane has a frame");
        let mut e = SketchEdit::new(None, plane, frame, Sketch::default(), Vec::new(), FitOptions::default());
        e.prefs = SketchPrefs::DEFAULT;
        e.mm_per_px = 0.1;
        e
    }

    fn click(e: &mut SketchEdit, x: f64, y: f64) {
        let input = SketchInput { pos: Some(DVec2::new(x, y)), mm_per_px: 0.1, clicked: true, ..Default::default() };
        e.pointer(&input);
    }

    fn v(x: f64, y: f64) -> DVec2 {
        DVec2::new(x, y)
    }

    #[test]
    fn the_line_tool_draws_a_chain_held_together_and_level() {
        let mut e = edit();
        e.set_tool(Tool::Line);
        click(&mut e, 0.0, 0.0);
        click(&mut e, 20.0, 0.3);
        click(&mut e, 20.0, 10.0);
        assert_eq!(e.sketch.entities.len(), 2);
        assert!(e.report.ok);
        let rules: Vec<&str> = e.sketch.constraints.iter().map(|c| c.rule.name()).collect();
        assert!(rules.contains(&"Horizontal") && rules.contains(&"Vertical") && rules.contains(&"Coincident"), "{rules:?}");
        let Shape::Line { a, b } = e.sketch.entities[0].shape else { panic!() };
        assert!((a.y - b.y).abs() < 1e-6, "made level");
        // A second click on the last end ends the chain.
        click(&mut e, 20.0, 10.0);
        assert!(e.chain.is_none() && e.pending.is_empty());
        assert_eq!(e.sketch.entities.len(), 2);
    }

    #[test]
    fn a_rectangle_and_a_dimension_typed_on_it() {
        let mut e = edit();
        e.set_tool(Tool::Rectangle);
        click(&mut e, 0.0, 0.0);
        click(&mut e, 30.0, 20.0);
        assert_eq!(e.regions.len(), 1);
        e.set_tool(Tool::Dimension);
        click(&mut e, 15.0, 0.0);
        click(&mut e, 15.0, 50.0);
        let (id, text) = e.edit_dim.clone().expect("a dimension is open for typing");
        assert_eq!(text, "30");
        assert!(e.set_dimension(id, "42"));
        let Shape::Line { a, b } = e.sketch.entities[0].shape else { panic!() };
        assert!(((b - a).length() - 42.0).abs() < 1e-6);
    }

    #[test]
    fn round_values_snap_only_when_near() {
        assert_eq!(round_value(29.98, false), 30.0);
        assert_eq!(round_value(4.49, false), 4.5);
        assert_eq!(round_value(27.3, false), 27.3);
        assert_eq!(round_value(89.7, true), 90.0);
    }

    // The tests below are ported from ScanRuler's `reDraw.test.ts` (the store-driven ones:
    // the Line tool, the click tools, selection and the draw modes), © Stefan Hermann (CNC
    // Kitchen), AGPL-3.0-only, driven through this editor instead of the zustand store.

    #[test]
    fn a_centre_rectangle_stays_about_its_centre_point_on_its_construction_diagonals() {
        let mut e = edit();
        e.prefs.draw_modes.rectangle = RectangleMode::Center;
        e.set_tool(Tool::Rectangle);
        click(&mut e, 10.0, 5.0);
        click(&mut e, 20.0, 12.0);
        // Four sides, two diagonals and the centre point.
        assert_eq!(e.sketch.entities.len(), 7);
        assert_eq!(e.sketch.entities.iter().filter(|x| x.construction).count(), 3);
        assert_eq!(e.regions.len(), 1);
        let centre = e.sketch.entities.iter().find(|x| matches!(x.shape, Shape::Point { .. })).expect("a centre");
        let Shape::Point { p } = centre.shape else { panic!() };
        assert!((p - v(10.0, 5.0)).length() < 1e-6, "{p}");
        let (lo, hi) = e.sketch.entities.iter().filter(|x| !x.construction).fold((v(1e9, 1e9), v(-1e9, -1e9)), |(lo, hi), x| match x.shape {
            Shape::Line { a, b } => (lo.min(a).min(b), hi.max(a).max(b)),
            _ => (lo, hi),
        });
        assert!((lo - v(0.0, -2.0)).length() < 1e-6 && (hi - v(20.0, 12.0)).length() < 1e-6, "{lo} {hi}");
    }

    #[test]
    fn a_three_point_rectangle_lies_at_the_angle_of_its_first_side() {
        let mut e = edit();
        e.prefs.draw_modes.rectangle = RectangleMode::ThreePoint;
        e.set_tool(Tool::Rectangle);
        click(&mut e, 0.0, 0.0);
        click(&mut e, 30.0, 40.0);
        assert!(e.sketch.entities.is_empty(), "not made yet");
        click(&mut e, -8.0, 6.0);
        assert_eq!(e.sketch.entities.len(), 4);
        assert!(e.report.ok);
        let Shape::Line { a, b } = e.sketch.entities[1].shape else { panic!() };
        assert!(((b - a).length() - 10.0).abs() < 1e-6);
        assert_eq!(e.regions.len(), 1);
    }

    #[test]
    fn a_circle_by_two_points_and_by_three() {
        let mut e = edit();
        e.prefs.draw_modes.circle = CircleMode::TwoPoint;
        e.set_tool(Tool::Circle);
        click(&mut e, 0.0, 0.0);
        click(&mut e, 10.0, 0.0);
        let Shape::Circle { c, r } = e.sketch.entities[0].shape else { panic!() };
        assert!((c - v(5.0, 0.0)).length() < 1e-9 && (r - 5.0).abs() < 1e-9);
        e.prefs.draw_modes.circle = CircleMode::ThreePoint;
        e.set_tool(Tool::Circle);
        click(&mut e, 30.0, 0.0);
        click(&mut e, 40.0, 10.0);
        click(&mut e, 50.0, 0.0);
        let Shape::Circle { c, r } = e.sketch.entities[1].shape else { panic!() };
        assert!((c - v(40.0, 0.0)).length() < 1e-6 && (r - 10.0).abs() < 1e-6);
        // Three points on a line: no circle, and a sentence.
        click(&mut e, 0.0, 30.0);
        click(&mut e, 10.0, 30.0);
        click(&mut e, 20.0, 30.0);
        assert_eq!(e.sketch.entities.len(), 2);
        assert!(e.error.as_deref().is_some_and(|m| m.contains("on a line")));
    }

    #[test]
    fn a_centre_arc_runs_the_shorter_way_round() {
        let mut e = edit();
        e.prefs.draw_modes.arc = ArcMode::Center;
        e.set_tool(Tool::Arc);
        click(&mut e, 0.0, 0.0);
        click(&mut e, 10.0, 0.0);
        click(&mut e, 0.0, -30.0);
        let Shape::Arc { c, r, sweep, .. } = e.sketch.entities[0].shape else { panic!() };
        assert!(c.length() < 1e-9 && (r - 10.0).abs() < 1e-9);
        assert!((sweep - std::f64::consts::FRAC_PI_2).abs() < 1e-9);
    }

    #[test]
    fn a_polygon_is_drawn_about_its_construction_circle_with_equal_sides() {
        let mut e = edit();
        e.prefs.polygon_sides = 5;
        e.set_tool(Tool::Polygon);
        click(&mut e, 0.0, 0.0);
        click(&mut e, 10.0, 0.0);
        assert_eq!(e.sketch.entities.iter().filter(|x| x.kind() == ShapeKind::Line).count(), 5);
        assert_eq!(e.sketch.entities.iter().filter(|x| x.construction).count(), 1);
        assert_eq!(e.regions.len(), 1);
        assert!(e.report.ok);
        e.prefs.draw_modes.polygon = PolygonFit::Circumscribed;
        e.prefs.polygon_sides = 6;
        e.set_tool(Tool::Polygon);
        click(&mut e, 40.0, 0.0);
        click(&mut e, 50.0, 0.0);
        // Across the flats: the inner circle's radius is the click's distance.
        let inner = e.sketch.entities.iter().filter(|x| x.construction).filter_map(|x| x.shape.radius()).fold(f64::INFINITY, f64::min);
        assert!((inner - 10.0).abs() < 1e-6, "{inner}");
    }

    #[test]
    fn a_press_on_the_chains_last_end_dragged_makes_an_arc_tangent_to_it() {
        let mut e = edit();
        e.set_tool(Tool::Line);
        click(&mut e, 0.0, 0.0);
        click(&mut e, 20.0, 0.0);
        let at = |x: f64, y: f64, s: (f32, f32)| SketchInput { pos: Some(v(x, y)), mm_per_px: 0.1, screen: Some(egui::pos2(s.0, s.1)), ..Default::default() };
        e.pointer(&SketchInput { pressed: true, down: true, ..at(20.0, 0.0, (200.0, 0.0)) });
        e.pointer(&SketchInput { down: true, ..at(30.0, 10.0, (300.0, -100.0)) });
        e.pointer(&SketchInput { released: true, ..at(30.0, 10.0, (300.0, -100.0)) });
        let arc = e.sketch.entities.iter().find(|x| x.kind() == ShapeKind::Arc).expect("an arc");
        let Shape::Arc { c, r, .. } = arc.shape else { panic!() };
        assert!((c - v(20.0, 10.0)).length() < 1e-6 && (r - 10.0).abs() < 1e-6, "{c} {r}");
        assert!(e.sketch.constraints.iter().any(|x| matches!(x.rule, Rule::Tangent { .. })));
        assert!(e.chain.is_some_and(|c| c.entity == arc.id), "the chain runs on from the arc");
    }

    #[test]
    fn a_line_dragged_by_its_body_moves_square_to_itself() {
        // A line of a chain the drag cannot carry whole (the next line's far end is free): the
        // solver's, and it moves square to itself. A shape defined in all but its place goes
        // any way (see `a_shape_defined_in_all_but_its_place_goes_any_way_by_an_edge`).
        let mut e = edit();
        e.set_tool(Tool::Line);
        click(&mut e, 5.0, 5.0);
        click(&mut e, 25.0, 5.2);
        click(&mut e, 25.0, 20.0);
        e.escape();
        e.set_tool(Tool::Select);
        let Shape::Line { a: a0, b: b0 } = e.sketch.entities[0].shape else { panic!() };
        let at = |x: f64, y: f64, s: (f32, f32)| SketchInput { pos: Some(v(x, y)), mm_per_px: 0.1, screen: Some(egui::pos2(s.0, s.1)), ..Default::default() };
        e.pointer(&SketchInput { pressed: true, down: true, ..at(15.0, a0.y, (150.0, -50.0)) });
        e.pointer(&SketchInput { down: true, ..at(19.0, a0.y + 5.0, (190.0, -100.0)) });
        e.pointer(&SketchInput { released: true, ..at(19.0, a0.y + 5.0, (190.0, -100.0)) });
        let Shape::Line { a, b } = e.sketch.entities[0].shape else { panic!() };
        assert!((a - a0 - v(0.0, 5.0)).length() < 1e-4 && (b - b0 - v(0.0, 5.0)).length() < 1e-4, "{a} {b}");
    }

    #[test]
    fn a_window_takes_what_lies_inside_a_crossing_box_what_it_touches() {
        let mut e = edit();
        e.set_tool(Tool::Rectangle);
        click(&mut e, 0.0, 0.0);
        click(&mut e, 10.0, 10.0);
        e.set_tool(Tool::Select);
        let unproject = |p: Pos2| Some(v(p.x as f64 / 10.0, -p.y as f64 / 10.0));
        let at = |x: f32, y: f32| SketchInput { pos: unproject(egui::pos2(x, y)), mm_per_px: 0.1, screen: Some(egui::pos2(x, y)), unproject: Some(&unproject), ..Default::default() };
        // Rightward over the left half: only the left side lies wholly inside.
        e.pointer(&SketchInput { pressed: true, down: true, ..at(-20.0, 20.0) });
        e.pointer(&SketchInput { down: true, ..at(50.0, -120.0) });
        assert!(e.boxing.as_ref().is_some_and(|b| !b.crossing && b.ids.len() == 1));
        e.pointer(&SketchInput { released: true, ..at(50.0, -120.0) });
        assert_eq!(e.selected_entities().len(), 1);
        // Leftward over the same: crossing, it takes the three it touches.
        e.pointer(&SketchInput { pressed: true, down: true, ..at(50.0, 20.0) });
        e.pointer(&SketchInput { down: true, ..at(-20.0, -120.0) });
        e.pointer(&SketchInput { released: true, ..at(-20.0, -120.0) });
        assert_eq!(e.selected_entities().len(), 3);
    }

    #[test]
    fn the_fillet_tool_rounds_a_corner_with_its_radius_open_for_typing() {
        let mut e = edit();
        e.set_tool(Tool::Rectangle);
        click(&mut e, 0.0, 0.0);
        click(&mut e, 30.0, 20.0);
        e.press_tool(Tool::Fillet);
        click(&mut e, 30.0, 20.0);
        let (id, _) = e.edit_dim.clone().expect("the radius is open");
        assert!(matches!(e.sketch.constraint(id).map(|c| &c.rule), Some(Rule::Radius { .. })));
        assert!(e.set_dimension(id, "4"));
        assert!(e.report.ok);
        // The next round starts with the last radius.
        click(&mut e, 0.0, 0.0);
        let (id2, text) = e.edit_dim.clone().expect("the second radius");
        assert_ne!(id, id2);
        assert_eq!(text, "4");
        assert_eq!(e.regions.len(), 1);
    }

    #[test]
    fn a_pattern_repeats_a_circle_in_a_row_and_round_a_centre() {
        let mut e = edit();
        e.set_tool(Tool::Circle);
        click(&mut e, 0.0, 0.0);
        click(&mut e, 2.0, 0.0);
        e.set_tool(Tool::Select);
        click(&mut e, 2.0, 0.0);
        e.open_pattern();
        assert!(matches!(e.pattern, Some(SketchPatternSpec::Rectangular { count: 3, .. })));
        assert_eq!(e.pattern_preview().len(), 2);
        e.apply_pattern();
        assert_eq!(e.sketch.entities.len(), 3);
        assert_eq!(e.selected_entities().len(), 2);
    }

    #[test]
    fn align_turns_the_sheet_under_a_crooked_line_and_nothing_moves_on_the_part() {
        let mut e = edit();
        e.set_tool(Tool::Line);
        click(&mut e, 0.0, 0.0);
        click(&mut e, 10.0, 7.0);
        e.escape();
        let before: Vec<glam::DVec3> = match e.sketch.entities[0].shape {
            Shape::Line { a, b } => vec![e.frame.to_world(a), e.frame.to_world(b)],
            _ => panic!(),
        };
        e.set_tool(Tool::Select);
        click(&mut e, 5.0, 3.5);
        e.align_to_selected();
        let Shape::Line { a, b } = e.sketch.entities[0].shape else { panic!() };
        assert!((a.y - b.y).abs() < 1e-9, "the line lies along X");
        assert!((e.frame.to_world(a) - before[0]).length() < 1e-9 && (e.frame.to_world(b) - before[1]).length() < 1e-9);
        assert!(e.frame_turned);
    }

    // ---- ported from ScanRuler's `reDraw.test.ts`, the store-driven cases ----

    /// A click as ScanRuler's store takes it: at a spot, with the hand's reach, raw or snapping.
    fn at(e: &mut SketchEdit, p: [f64; 2], radius: f64, raw: bool) {
        e.mm_per_px = radius / SNAP_PX;
        e.sheet_click(v(p[0], p[1]), false, raw);
    }

    fn drag_start(e: &mut SketchEdit, p: [f64; 2], radius: f64) -> bool {
        e.mm_per_px = radius / SNAP_PX;
        e.drag_start(v(p[0], p[1]), None)
    }

    fn drag(e: &mut SketchEdit, p: [f64; 2], radius: f64, raw: bool) {
        e.mm_per_px = radius / SNAP_PX;
        e.sketch_drag(v(p[0], p[1]), raw);
    }

    fn drag_end(e: &mut SketchEdit) {
        if let Some(d) = e.drag.take() {
            if d.begun {
                e.sketch_drag_end(d);
            }
        }
    }

    fn corner(e: &SketchEdit, i: usize) -> Option<[f64; 2]> {
        match e.sketch.entities[i].shape {
            Shape::Line { a, .. } => Some([(a.x * 1e6).round() / 1e6, (a.y * 1e6).round() / 1e6]),
            _ => None,
        }
    }

    #[test]
    fn each_line_starts_where_the_last_ended_joined_to_it() {
        let mut e = edit();
        e.set_tool(Tool::Line);
        at(&mut e, [0.0, 0.0], 0.5, true);
        at(&mut e, [10.0, 3.0], 0.5, true);
        assert_eq!(e.sketch.entities.len(), 1);
        assert_eq!(e.pending, vec![v(10.0, 3.0)]);
        at(&mut e, [12.0, 9.0], 0.5, true);
        assert_eq!(e.sketch.entities.len(), 2);
        assert_eq!(e.sketch.constraints.iter().filter(|c| matches!(c.rule, Rule::Coincident { .. })).count(), 1);
        assert_eq!(e.pending.len(), 1);
    }

    #[test]
    fn a_second_click_on_the_last_end_ends_the_chain_and_makes_nothing() {
        let mut e = edit();
        e.set_tool(Tool::Line);
        at(&mut e, [0.0, 0.0], 0.5, true);
        at(&mut e, [10.0, 3.0], 0.5, true);
        at(&mut e, [10.0, 3.0], 0.5, true);
        assert_eq!(e.sketch.entities.len(), 1);
        assert!(e.pending.is_empty());
        assert!(e.error.is_none());
    }

    #[test]
    fn landing_on_another_pieces_end_it_has_arrived() {
        let mut e = edit();
        e.set_tool(Tool::Line);
        at(&mut e, [0.0, 0.0], 0.5, true);
        at(&mut e, [10.0, 3.0], 0.5, true);
        at(&mut e, [12.0, 9.0], 0.5, true);
        // Back onto the first line's start, snapping this time.
        at(&mut e, [0.1, 0.1], 0.5, false);
        assert_eq!(e.sketch.entities.len(), 3);
        assert!(e.pending.is_empty());
        assert_eq!(e.regions.len(), 1);
    }

    #[test]
    fn the_chain_continues_from_where_the_solver_put_a_line_made_level() {
        let mut e = edit();
        e.set_tool(Tool::Line);
        at(&mut e, [0.0, 0.0], 0.5, true);
        at(&mut e, [10.0, 0.2], 0.5, true);
        let line = &e.sketch.entities[0];
        assert_eq!(e.pending[0], anchor_point(&line.shape, AnchorAt::B));
        assert!((e.pending[0].y - anchor_point(&line.shape, AnchorAt::A).y).abs() < 5e-7);
    }

    #[test]
    fn a_drag_out_of_the_last_end_is_an_arc_tangent_to_the_line_and_the_chain_runs_on_from_it() {
        let mut e = edit();
        e.set_tool(Tool::Line);
        at(&mut e, [0.0, 0.0], 0.5, true);
        at(&mut e, [10.0, 0.0], 0.5, true);
        e.mm_per_px = 0.5 / SNAP_PX;
        e.tangent_arc_to(v(15.0, 5.0), true);
        let kinds: Vec<ShapeKind> = e.sketch.entities.iter().map(|x| x.kind()).collect();
        assert_eq!(kinds, vec![ShapeKind::Line, ShapeKind::Arc]);
        assert!(e.sketch.constraints.iter().any(|c| matches!(c.rule, Rule::Tangent { .. })));
        let arc = &e.sketch.entities[1];
        assert!((arc.shape.radius().unwrap_or(0.0) - 5.0).abs() < 5e-4);
        assert!(e.report.ok);
        assert!((e.pending[0] - v(15.0, 5.0)).length() < 1e-3);
        assert_eq!(e.chain, Some(Anchor::new(arc.id, AnchorAt::B)));
    }

    /// The L the click tools are tried on.
    fn draw_l(e: &mut SketchEdit) {
        e.set_tool(Tool::Line);
        at(e, [0.0, 0.0], 0.5, true);
        at(e, [10.0, 0.0], 0.5, true);
        at(e, [10.0, 8.0], 0.5, true);
        e.set_tool(Tool::Select);
    }

    #[test]
    fn a_relation_with_nothing_selected_is_put_on_the_pieces_clicked_next_and_stays_in_hand() {
        let mut e = edit();
        draw_l(&mut e);
        let lines: Vec<EntityId> = e.sketch.entities.iter().map(|x| x.id).collect();
        let before = e.sketch.constraints.len();
        e.press_relation(RelationKind::Perpendicular);
        at(&mut e, [5.0, 0.0], 0.5, false);
        assert_eq!(e.selected_entities(), vec![lines[0]]);
        at(&mut e, [10.0, 4.0], 0.5, false);
        assert_eq!(e.sketch.constraints.len(), before + 1);
        assert_eq!(e.sketch.constraints.last().map(|c| c.rule.clone()), Some(Rule::Perpendicular { a: lines[0], b: lines[1] }));
        assert!(e.selected_entities().is_empty());
        assert_eq!(e.tool, Tool::Relation(RelationKind::Perpendicular));
    }

    #[test]
    fn the_selection_dimensions_too() {
        let mut e = edit();
        draw_l(&mut e);
        at(&mut e, [5.0, 0.0], 0.5, false);
        at(&mut e, [10.0, 4.0], 0.5, false);
        e.dimension_selection();
        let angle = e.sketch.constraints.iter().find(|c| matches!(c.rule, Rule::Angle { .. })).expect("an angle");
        assert_eq!(angle.rule.value(), Some(90.0));
        assert_eq!(e.tool, Tool::Dimension);
        assert!(e.selected_entities().is_empty());
    }

    #[test]
    fn the_press_that_might_have_been_a_drag_leaves_the_selection_to_the_click() {
        let mut e = edit();
        e.set_tool(Tool::Line);
        at(&mut e, [0.0, 0.0], 0.5, true);
        at(&mut e, [10.0, 5.0], 0.5, true);
        e.set_tool(Tool::Select);
        let history = e.undo.len();
        assert!(drag_start(&mut e, [5.0, 2.5], 0.5));
        assert!(e.selected_entities().is_empty());
        drag_end(&mut e);
        at(&mut e, [5.0, 2.5], 0.5, false);
        assert_eq!(e.selected_entities(), vec![1]);
        assert_eq!(e.undo.len(), history);
    }

    #[test]
    fn a_drag_selects_the_piece_it_carries_and_is_one_step_of_the_history() {
        let mut e = edit();
        e.set_tool(Tool::Line);
        at(&mut e, [0.0, 0.0], 0.5, true);
        at(&mut e, [10.0, 5.0], 0.5, true);
        e.set_tool(Tool::Select);
        let history = e.undo.len();
        drag_start(&mut e, [10.0, 5.0], 0.5);
        drag(&mut e, [12.0, 6.0], 0.5, true);
        drag(&mut e, [14.0, 7.0], 0.5, true);
        drag_end(&mut e);
        assert_eq!(e.selected_entities(), vec![1]);
        assert_eq!(e.undo.len(), history + 1);
    }

    #[test]
    fn a_rectangle_about_its_centre_held_where_it_was_clicked_grows_about_it() {
        let mut e = edit();
        e.prefs.draw_modes.rectangle = RectangleMode::Center;
        e.set_tool(Tool::Rectangle);
        // The first click on the sheet's zero: the centre is held there.
        at(&mut e, [0.01, 0.01], 0.5, false);
        at(&mut e, [10.0, 6.0], 0.01, true);
        assert!(e.error.is_none());
        let sides: Vec<usize> = (0..e.sketch.entities.len()).filter(|i| e.sketch.entities[*i].kind() == ShapeKind::Line && !e.sketch.entities[*i].construction).collect();
        let corners: Vec<[f64; 2]> = sides.iter().filter_map(|i| corner(&e, *i)).collect();
        assert_eq!(corners, vec![[-10.0, -6.0], [10.0, -6.0], [10.0, 6.0], [-10.0, 6.0]]);
        assert!(e.sketch.entities.iter().any(|x| x.kind() == ShapeKind::Point && x.construction));
        assert_eq!(e.sketch.entities.iter().filter(|x| x.kind() == ShapeKind::Line && x.construction).count(), 2);
        assert!(e.report.ok);
        // Its width typed: it grows about the centre, which stays on the zero.
        let bottom = e.sketch.entities[sides[0]].id;
        e.set_tool(Tool::Dimension);
        at(&mut e, [0.0, -6.0], 0.5, false);
        at(&mut e, [0.0, -30.0], 0.5, false);
        let (width, _) = e.edit_dim.clone().expect("the width");
        assert!(matches!(e.sketch.constraint(width).map(|c| &c.rule), Some(Rule::Length { entity, .. }) if *entity == bottom));
        assert!(e.set_dimension(width, "30"));
        let Some(Shape::Point { p }) = e.sketch.entities.iter().find(|x| x.kind() == ShapeKind::Point).map(|x| x.shape.clone()) else { panic!("the centre") };
        assert!(p.length() < 1e-6);
        let Some(Shape::Line { a, b }) = e.sketch.entity(bottom).map(|x| x.shape.clone()) else { panic!("the bottom") };
        assert!((a.x + 15.0).abs() < 1e-6 && (b.x - 15.0).abs() < 1e-6);
    }

    #[test]
    fn a_rectangle_by_three_points_at_an_angle_and_level_when_its_first_side_is_nearly_so() {
        let mut e = edit();
        e.prefs.draw_modes.rectangle = RectangleMode::ThreePoint;
        e.set_tool(Tool::Rectangle);
        for p in [[0.0, 0.0], [6.0, 8.0], [-4.0, 3.0]] {
            at(&mut e, p, 0.01, true);
        }
        assert!(e.error.is_none());
        assert_eq!(e.sketch.entities.len(), 4);
        assert_eq!(e.sketch.constraints.iter().filter(|c| matches!(c.rule, Rule::Parallel { .. })).count(), 2);
        assert_eq!(e.sketch.constraints.iter().filter(|c| matches!(c.rule, Rule::Perpendicular { .. })).count(), 1);
        assert!(e.report.ok);
        let far = corner(&e, 2).expect("a corner");
        assert!((far[0] - 2.0).abs() < 5e-3 && (far[1] - 11.0).abs() < 5e-3);
        e.clear_selection();
        for p in [[20.0, 0.0], [30.0, 0.1], [25.0, 5.0]] {
            at(&mut e, p, 0.01, true);
        }
        assert!(e.sketch.constraints.iter().filter(|c| matches!(c.rule, Rule::Horizontal { .. } | Rule::Vertical { .. })).count() >= 4);
        assert!(e.report.ok);
    }

    #[test]
    fn an_arc_about_its_centre_the_start_gives_the_radius_the_end_only_the_direction() {
        let mut e = edit();
        e.prefs.draw_modes.arc = ArcMode::Center;
        e.set_tool(Tool::Arc);
        for p in [[0.0, 0.0], [5.0, 0.0], [0.0, 9.0]] {
            at(&mut e, p, 0.01, true);
        }
        let Shape::Arc { c, r, start, sweep } = e.sketch.entities[0].shape else { panic!("an arc") };
        assert!(c.length() < 1e-12 && r == 5.0 && start == 0.0);
        assert!((sweep - std::f64::consts::FRAC_PI_2).abs() < 1e-9);
    }

    /// A 20 × 10 rectangle at (10, 10), its width and height dimensioned.
    fn sized() -> SketchEdit {
        let mut e = edit();
        e.set_tool(Tool::Rectangle);
        at(&mut e, [10.0, 10.0], 0.01, true);
        at(&mut e, [30.0, 20.0], 0.01, true);
        e.set_tool(Tool::Dimension);
        for p in [[20.0, 10.0], [30.0, 15.0]] {
            at(&mut e, p, 0.5, false);
            at(&mut e, [p[0] + 0.0, p[1] + 40.0], 0.5, false);
            e.edit_dim = None;
        }
        e.set_tool(Tool::Select);
        e
    }

    #[test]
    fn a_shape_defined_in_all_but_its_place_follows_the_hand_whole_by_a_corner() {
        let mut e = sized();
        assert_eq!(rigid_cluster(&e.sketch, 2, GrabAt::Anchor(AnchorAt::B)), Some(vec![1, 2, 3, 4]));
        drag_start(&mut e, [30.0, 20.0], 0.5);
        drag(&mut e, [30.5, 20.3], 0.5, false);
        assert_eq!(corner(&e, 2), Some([30.5, 20.3]));
        assert_eq!(corner(&e, 0), Some([10.5, 10.3]));
        drag(&mut e, [35.0, 23.0], 0.5, false);
        assert_eq!(corner(&e, 0), Some([15.0, 13.0]));
        drag_end(&mut e);
        assert_eq!(corner(&e, 0), Some([15.0, 13.0]));
        assert_eq!(corner(&e, 2), Some([35.0, 23.0]));
    }

    #[test]
    fn a_shape_defined_in_all_but_its_place_goes_any_way_by_an_edge() {
        let mut e = sized();
        drag_start(&mut e, [10.0, 15.0], 0.5);
        drag(&mut e, [6.0, 21.0], 0.5, false);
        drag_end(&mut e);
        assert_eq!(corner(&e, 0), Some([6.0, 16.0]));
        assert_eq!(corner(&e, 2), Some([26.0, 26.0]));
    }

    #[test]
    fn a_shape_dropped_by_its_corner_on_the_sheets_zero_is_held_there() {
        let mut e = sized();
        drag_start(&mut e, [10.0, 10.0], 0.5);
        drag(&mut e, [0.1, 0.1], 0.5, false);
        drag_end(&mut e);
        assert_eq!(corner(&e, 0), Some([0.0, 0.0]));
        assert!(e.report.under_constrained.is_empty());
        assert_eq!(rigid_cluster(&e.sketch, 2, GrabAt::Anchor(AnchorAt::B)), None);
    }

    #[test]
    fn a_shape_with_a_freedom_left_is_the_solvers_a_corner_of_a_bare_rectangle_resizes_it() {
        let mut e = edit();
        e.set_tool(Tool::Rectangle);
        at(&mut e, [10.0, 10.0], 0.01, true);
        at(&mut e, [30.0, 20.0], 0.01, true);
        e.set_tool(Tool::Select);
        assert_eq!(rigid_cluster(&e.sketch, 2, GrabAt::Anchor(AnchorAt::B)), None);
        drag_start(&mut e, [30.0, 20.0], 0.5);
        drag(&mut e, [35.0, 23.0], 0.5, true);
        drag_end(&mut e);
        let r = |v: f64| (v * 1e3).round() / 1e3;
        let (c0, c2) = (corner(&e, 0).expect("corner"), corner(&e, 2).expect("corner"));
        assert_eq!([r(c0[0]), r(c0[1]), r(c2[0]), r(c2[1])], [10.0, 10.0, 35.0, 23.0]);
    }

    // ---- ported from ScanRuler's `reSketchImage.test.ts`, the editor case ----

    #[test]
    fn the_picture_is_scaled_moved_and_levelled_by_clicks_on_the_sheet() {
        let mut e = edit();
        e.image = Some(mm_reverse::sketch_image::placed_image("data:image/png;base64,AAAA", 200, 100, DVec2::ZERO, Some("drawing.png")));
        e.start_image_pick(Some(ImagePickMode::Scale));
        at(&mut e, [-50.0, 0.0], 0.5, true);
        at(&mut e, [50.0, 0.0], 0.5, true);
        assert_eq!(e.image_pick.as_ref().map(|p| p.points.len()), Some(2));
        e.scale_image(25.0);
        assert!((e.image.as_ref().expect("a picture").mm_per_px - 0.125).abs() < 1e-12);
        assert!(e.image_pick.is_none());
        // A point of the picture onto the origin: the second click snaps.
        e.start_image_pick(Some(ImagePickMode::Move));
        at(&mut e, [-50.0, 0.0], 0.5, true);
        at(&mut e, [0.2, 0.1], 0.5, false);
        let p = mm_reverse::sketch_image::image_point(e.image.as_ref().expect("a picture"), v(0.0, 50.0));
        assert!(p.length() < 1e-6, "{p}");
        e.start_image_pick(Some(ImagePickMode::Level));
        at(&mut e, [0.0, 0.0], 0.5, true);
        at(&mut e, [10.0, 10.0], 0.5, true);
        assert!((e.image.as_ref().expect("a picture").angle_deg + 45.0).abs() < 1e-9);
    }

    #[test]
    fn grid_spacing_follows_the_zoom() {
        assert_eq!(grid_spacing(0.1), 5.0);
        assert_eq!(grid_spacing(0.01), 0.5);
        assert_eq!(grid_spacing(10.0), 100.0);
    }
}
