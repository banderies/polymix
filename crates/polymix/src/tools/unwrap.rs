//! Unwrap (Edit > Unwrap): compute UVs — charts by face group or component, flattened
//! conformally or as rigidly as possible, packed on a sheet (see [`mm_edit::uv`]) — or assign
//! them without unwrapping by projecting the selection or the whole object (SetUV: planar,
//! cylindrical, spherical or box; see [`mm_edit::uv_project`]).
//!
//! The layout can then be edited (see [`mm_edit::uv_edit`]): the UV Layout window shows the
//! sheet large (wheel zooms, right- or middle-drag pans, double-click frames); click an island
//! or drag a box to select (Shift adds), drag the selection to move it, its corner handles to
//! scale (Shift: freely) and its round handle to turn it (Shift: 15° steps), or type exact
//! position, size and angle; flip, fit on the sheet, re-pack the rest around the selection.
//! Clicking the model selects the island under the pointer, and the 3D view highlights the
//! selected islands. Every step can be undone inside the tool (Undo / Redo, ⌘Z / ⇧⌘Z);
//! Accept commits the UVs as one undo step, Cancel puts back what the object had.
//!
//! The object shows a UV checker while the tool is open. When it already has UVs, the tool
//! starts from them (to edit them) instead of unwrapping again. UVs are exported with OBJ and
//! kept in projects.

use super::*;
use crate::jobs::Job;
use crate::scene::{next_stamp, UvSet};
use crate::ui_util;
use glam::DAffine2;
use mm_edit::uv_edit::{self as uve, Corners, UvIslands};
use mm_edit::uv_project::{self, Projection, ProjectionAxis, SetUvParams};
use mm_edit::{Packing, UnwrapElements, UnwrapParams, UnwrapType};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum UvMethod {
    /// Flatten charts and pack them on the sheet.
    Unwrap,
    /// SetUV: project the selected triangles (or the whole object) without unwrapping.
    Project,
}

/// The tool's settings (agents see them flat: the Unwrap parameters, `method` and `projection`).
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct UnwrapSettings {
    /// unwrap: flatten and pack charts on the width × height sheet; project: SetUV.
    pub method: UvMethod,
    #[serde(flatten)]
    pub unwrap: UnwrapParams,
    /// SetUV projection (method project); it uses the sheet's proportions too.
    pub projection: SetUvParams,
}

/// What made the layout being edited.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Made {
    /// The object's own UVs, from before the tool.
    Existing,
    Unwrap(UnwrapParams),
    /// A projection, with the view it was taken from and the sheet's aspect.
    Project(SetUvParams, [f64; 6], f64),
}

/// UVs with their islands.
#[derive(Clone)]
struct Layout {
    uvs: Arc<UvSet>,
    islands: Arc<UvIslands>,
    /// Edges where the UVs break (island borders, seams): (triangle, corner) for the edge
    /// from that corner to the next.
    border: Arc<Vec<(u32, u8)>>,
}

impl Layout {
    fn new(mesh: &Mesh, uvs: Arc<UvSet>) -> Self {
        let islands = uve::uv_islands(mesh, &uvs.corners);
        let c = &uvs.corners;
        let same = |a: [f32; 2], b: [f32; 2]| (a[0] - b[0]).abs() <= 1e-6 && (a[1] - b[1]).abs() <= 1e-6;
        let mut border = Vec::new();
        for t in mesh.triangle_ids().filter(|&t| (t as usize) < c.len()) {
            let tv = mesh.triangle(t);
            for (k, e) in mesh.tri_edges(t).into_iter().enumerate() {
                let o = mesh.edge_other_triangle(e, t);
                let joined = o != mm_core::INVALID && (o as usize) < c.len() && {
                    let ov = mesh.triangle(o);
                    [tv[k], tv[(k + 1) % 3]].iter().all(|&v| {
                        let (i, j) = (tv.iter().position(|&x| x == v).unwrap_or(0), ov.iter().position(|&x| x == v).unwrap_or(0));
                        same(c[t as usize][i], c[o as usize][j])
                    })
                };
                if !joined {
                    border.push((t, k as u8));
                }
            }
        }
        Self { uvs, islands: Arc::new(islands), border: Arc::new(border) }
    }

    /// The same islands with other corners (an edit).
    fn with(&self, corners: Vec<Corners>) -> Self {
        Self { uvs: Arc::new(UvSet { topo: self.uvs.topo, corners, stamp: next_stamp() }), islands: self.islands.clone(), border: self.border.clone() }
    }
}

/// What the tool shows: the layout and what made it (with a summary). Undo / redo swap these.
#[derive(Clone, Default)]
struct Shown {
    layout: Option<Layout>,
    made: Option<(Made, String)>,
    /// Edited by hand since it was made.
    edited: bool,
}

/// A mouse drag in the layout window.
enum Drag {
    Move { base: Layout, from: DVec2 },
    Turn { base: Layout, centre: DVec2, from: f64 },
    Scale { base: Layout, pivot: DVec2, from: DVec2 },
    Select { from: egui::Pos2, to: egui::Pos2, add: bool },
    Pan,
}

struct Editor {
    open: bool,
    /// Selected islands (for the islands of `of`).
    selected: Vec<bool>,
    of: usize,
    zoom: f32,
    pan: egui::Vec2,
    drag: Option<Drag>,
    snap: bool,
    keep_ratio: bool,
    turn_by: f64,
    /// Where the layout was drawn last.
    canvas: Option<egui::Rect>,
}

impl Default for Editor {
    fn default() -> Self {
        Self { open: false, selected: Vec::new(), of: 0, zoom: 1.0, pan: egui::Vec2::ZERO, drag: None, snap: true, keep_ratio: true, turn_by: 45.0, canvas: None }
    }
}

/// Unwrapped layout and a summary.
type Unwrapped = (Layout, String);

pub struct UnwrapTool {
    object: Option<ObjectId>,
    s: UnwrapSettings,
    /// The camera's right and up, for projections along the view.
    view: (DVec3, DVec3),
    job: Option<(UnwrapParams, Job<Unwrapped>)>,
    shown: Shown,
    undo: Vec<Shown>,
    redo: Vec<Shown>,
    /// UVs the object had before, restored on cancel.
    before: Option<Option<Arc<UvSet>>>,
    /// UVs a projection of part of the object keeps on the other triangles.
    project_base: Option<(Vec<u32>, Option<Arc<UvSet>>)>,
    error: Option<String>,
    editor: Editor,
}

/// How many steps the tool keeps for undo.
const UNDO_STEPS: usize = 40;

/// Above this many triangles, drags update the model only when they end.
const LIVE_DRAG_TRIANGLES: usize = 250_000;

impl UnwrapTool {
    pub fn new(cx: &mut ToolCx) -> Self {
        Self::open(cx, UvMethod::Unwrap)
    }

    /// SetUV from the start (the menu's Set UVs).
    pub fn new_projection(cx: &mut ToolCx) -> Self {
        Self::open(cx, UvMethod::Project)
    }

    fn open(cx: &mut ToolCx, method: UvMethod) -> Self {
        let mut t = Self {
            object: cx.scene.active,
            s: UnwrapSettings { method, unwrap: UnwrapParams::default(), projection: SetUvParams { axis: ProjectionAxis::View, ..Default::default() } },
            view: (cx.camera.right(), cx.camera.up()),
            job: None,
            shown: Shown::default(),
            undo: Vec::new(),
            redo: Vec::new(),
            before: None,
            project_base: None,
            error: None,
            editor: Editor::default(),
        };
        // Edit the UVs the object has; unwrap otherwise.
        let existing = t.object.and_then(|id| cx.scene.get(id)).and_then(|o| Some((o.mesh.clone(), o.uvs()?.corners.clone())));
        match existing {
            Some((mesh, corners)) => {
                let layout = Layout::new(&mesh, Arc::new(UvSet::new(&mesh, corners)));
                let msg = format!("The object's UVs: {} islands", layout.islands.len());
                t.shown = Shown { layout: Some(layout), made: Some((Made::Existing, msg)), edited: false };
            }
            None if method == UvMethod::Unwrap => t.start(cx),
            None => {}
        }
        if method == UvMethod::Project {
            t.project(cx);
        }
        t
    }

    fn mesh(&self, cx: &ToolCx) -> Option<Arc<Mesh>> {
        self.object.and_then(|id| cx.scene.get(id)).map(|o| o.mesh.clone())
    }

    fn sheet(&self) -> DVec2 {
        DVec2::new(self.s.unwrap.width.max(1e-9), self.s.unwrap.height.max(1e-9))
    }

    fn start(&mut self, cx: &mut ToolCx) {
        let Some(mesh) = self.mesh(cx) else { return };
        if let Some((_, j)) = &self.job {
            j.cancel();
        }
        let p = self.s.unwrap;
        self.error = None;
        self.job = Some((
            p,
            Job::spawn("Unwrapping", cx.egui.clone(), move |_| {
                let t0 = std::time::Instant::now();
                let l = mm_edit::unwrap(&mesh, &p)?;
                let corners = l.uv.iter().map(|c| c.map(|q| q.as_vec2().to_array())).collect();
                let layout = Layout::new(&mesh, Arc::new(UvSet::new(&mesh, corners)));
                let msg = format!(
                    "{} charts, {:.0}% of true size, area distortion {:.1}% ({:.1}s)",
                    l.charts,
                    l.scale * 100.0,
                    l.area_distortion * 100.0,
                    t0.elapsed().as_secs_f32()
                );
                Ok((layout, msg))
            }),
        ));
    }

    /// Take a finished unwrap.
    fn poll(&mut self, cx: &mut ToolCx) {
        let Some((p, job)) = &self.job else { return };
        let Some(r) = job.poll() else { return };
        let p = *p;
        self.job = None;
        match r {
            Ok((layout, msg)) => self.set(cx, Shown { layout: Some(layout), made: Some((Made::Unwrap(p), msg)), edited: false }, true),
            Err(e) => self.error = Some(format!("{e:#}")),
        }
    }

    fn targets(&mut self, cx: &mut ToolCx) -> Option<(Arc<Mesh>, Vec<u32>, bool)> {
        targets(cx.scene, self.object)
    }

    fn view_key(&self) -> [f64; 6] {
        let (r, u) = self.view;
        [r.x, r.y, r.z, u.x, u.y, u.z]
    }

    fn aspect(&self) -> f64 {
        self.sheet().y / self.sheet().x
    }

    /// SetUV: project now (it is quick).
    fn project(&mut self, cx: &mut ToolCx) {
        // An unwrap still running would replace the projection when it ends.
        if let Some((_, j)) = self.job.take() {
            j.cancel();
        }
        let Some((mesh, tris, partial)) = self.targets(cx) else { return };
        // The other triangles keep the UVs they had when these triangles were first projected
        // (so adjusting the projection does not pile up).
        if self.project_base.as_ref().is_none_or(|b| b.0 != tris) {
            let own = self.object.and_then(|id| cx.scene.get(id)).and_then(|o| o.uvs.clone());
            self.project_base = Some((tris.clone(), self.shown.layout.as_ref().map(|l| l.uvs.clone()).or(own)));
        }
        let p = self.s.projection;
        match uv_project::project_uvs(&mesh, &tris, &p, Some(self.view), self.aspect()) {
            Ok(q) => {
                let base = self.project_base.clone().and_then(|b| b.1).filter(|b| b.topo == mesh.topology_version());
                let mut corners: Vec<Corners> = base.map(|b| b.corners.clone()).unwrap_or_default();
                corners.resize(mesh.max_tid(), [[0.0; 2]; 3]);
                for (k, &t) in tris.iter().enumerate() {
                    corners[t as usize] = q[k].map(|c| [c.x as f32, c.y as f32]);
                }
                let layout = Layout::new(&mesh, Arc::new(UvSet::new(&mesh, corners)));
                let kind = match p.projection {
                    Projection::Planar => "Planar",
                    Projection::Cylindrical => "Cylindrical",
                    Projection::Spherical => "Spherical",
                    Projection::Box => "Box",
                };
                let of = if partial { format!("{} selected triangles", tris.len()) } else { "the whole object".to_string() };
                let msg = format!("{kind} projection of {of}: {} islands", layout.islands.len());
                // Changing the projection's settings replaces the last projection (one undo
                // step for the whole adjustment) unless it was edited since.
                let record = self.shown.edited || !matches!(self.shown.made, Some((Made::Project(..), _)));
                let made = Made::Project(p, self.view_key(), self.aspect());
                self.set(cx, Shown { layout: Some(layout), made: Some((made, msg)), edited: false }, record);
                self.error = None;
            }
            Err(e) => self.error = Some(format!("SetUV: {e}")),
        }
    }

    /// Show `shown` (recording the current state for undo if `record`).
    fn set(&mut self, cx: &mut ToolCx, shown: Shown, record: bool) {
        if record && self.shown.layout.is_some() {
            self.undo.push(std::mem::take(&mut self.shown));
            if self.undo.len() > UNDO_STEPS {
                self.undo.remove(0);
            }
            self.redo.clear();
        }
        self.shown = shown;
        self.show(cx);
    }

    /// Put the shown UVs on the object (remembering what it had).
    fn show(&mut self, cx: &mut ToolCx) {
        let Some(uvs) = self.shown.layout.as_ref().map(|l| l.uvs.clone()) else { return };
        let Some(o) = self.object.and_then(|id| cx.scene.get_mut(id)) else { return };
        if self.before.is_none() {
            self.before = Some(o.uvs.clone());
        }
        o.uvs = Some(uvs);
    }

    fn restore(&mut self, cx: &mut ToolCx) {
        if let (Some(before), Some(o)) = (self.before.take(), self.object.and_then(|id| cx.scene.get_mut(id))) {
            o.uvs = before;
        }
    }

    fn undo(&mut self, cx: &mut ToolCx) -> bool {
        let Some(prev) = self.undo.pop() else { return false };
        self.editor.drag = None;
        self.redo.push(std::mem::replace(&mut self.shown, prev));
        self.show(cx);
        true
    }

    fn redo(&mut self, cx: &mut ToolCx) -> bool {
        let Some(next) = self.redo.pop() else { return false };
        self.editor.drag = None;
        self.undo.push(std::mem::replace(&mut self.shown, next));
        self.show(cx);
        true
    }

    /// Keep the island selection the size of the islands shown.
    fn sync_selection(&mut self) {
        let (n, of) = self.shown.layout.as_ref().map_or((0, 0), |l| (l.islands.len(), Arc::as_ptr(&l.islands) as usize));
        if self.editor.of != of || self.editor.selected.len() != n {
            self.editor.selected = vec![false; n];
            self.editor.of = of;
        }
    }

    fn which(&self) -> Vec<usize> {
        self.editor.selected.iter().enumerate().filter(|(_, &s)| s).map(|(i, _)| i).collect()
    }

    /// Edit the corners of the selected islands (one undo step).
    fn edit(&mut self, cx: &mut ToolCx, f: impl FnOnce(&mut [Corners], &UvIslands, &[usize], DVec2) -> Result<(), String>) -> Result<(), String> {
        self.sync_selection();
        let which = self.which();
        let Some(layout) = self.shown.layout.clone() else { return Err("no UVs yet".into()) };
        let mut corners = layout.uvs.corners.clone();
        f(&mut corners, &layout.islands, &which, self.sheet())?;
        let made = self.shown.made.clone();
        self.set(cx, Shown { layout: Some(layout.with(corners)), made, edited: true }, true);
        Ok(())
    }

    fn edit_selected(&mut self, cx: &mut ToolCx, f: impl FnOnce(&mut [Corners], &UvIslands, &[usize], DVec2)) -> Result<(), String> {
        self.sync_selection();
        if !self.editor.selected.contains(&true) {
            return Err("select islands first".into());
        }
        self.edit(cx, |c, i, w, s| {
            f(c, i, w, s);
            Ok(())
        })
    }

    fn repack(&mut self, cx: &mut ToolCx, all: bool) -> Result<String, String> {
        let border = self.s.unwrap.border;
        let mut factor = 1.0;
        self.edit(cx, |c, isl, which, sheet| {
            let mut fixed = vec![false; isl.len()];
            if !all {
                for &i in which {
                    fixed[i] = true;
                }
            }
            factor = uve::repack_islands(c, isl, &fixed, sheet, border)?;
            Ok(())
        })?;
        Ok(if factor < 1.0 { format!("Re-packed, shrunk to {:.0}% to fit", factor * 100.0) } else { "Re-packed".to_string() })
    }

    /// Select islands: replace (or add to) the selection.
    fn select(&mut self, islands: &[usize], add: bool) {
        self.sync_selection();
        if !add {
            self.editor.selected.iter_mut().for_each(|s| *s = false);
        }
        for &i in islands {
            if let Some(s) = self.editor.selected.get_mut(i) {
                *s = true;
            }
        }
    }

    fn label(&self) -> &'static str {
        match (self.shown.made.as_ref().map(|m| m.0), self.shown.edited) {
            (Some(Made::Project(..)), _) => "Set UVs",
            (Some(Made::Unwrap(_)), _) => "Unwrap",
            _ => "Edit UVs",
        }
    }
}

/// The triangles a projection covers: the selection, or the whole object (and whether it is
/// the selection).
fn targets(scene: &mut Scene, object: Option<ObjectId>) -> Option<(Arc<Mesh>, Vec<u32>, bool)> {
    let o = object.and_then(|id| scene.get_mut(id))?;
    let mesh = o.mesh.clone();
    o.selection.sync(&mesh);
    if o.selection.is_empty() {
        Some((mesh.clone(), mesh.triangle_ids().collect(), false))
    } else {
        Some((mesh, o.selection.to_vec(), true))
    }
}

/// Screen placement of the sheet in a layout view.
#[derive(Clone, Copy)]
struct View {
    sheet: DVec2,
    /// Points per millimetre.
    scale: f32,
    /// Screen position of the sheet's centre.
    centre: egui::Pos2,
}

impl View {
    fn fit(rect: egui::Rect, sheet: DVec2, zoom: f32, pan: egui::Vec2) -> Self {
        let margin = 14.0;
        let base = ((rect.width() - 2.0 * margin).max(10.0) / sheet.x as f32).min((rect.height() - 2.0 * margin).max(10.0) / sheet.y as f32);
        Self { sheet, scale: base * zoom, centre: rect.center() + pan }
    }

    fn screen(&self, s: DVec2) -> egui::Pos2 {
        egui::pos2(self.centre.x + ((s.x - self.sheet.x * 0.5) as f32) * self.scale, self.centre.y - ((s.y - self.sheet.y * 0.5) as f32) * self.scale)
    }

    fn sheet_at(&self, p: egui::Pos2) -> DVec2 {
        DVec2::new(((p.x - self.centre.x) / self.scale) as f64 + self.sheet.x * 0.5, (-(p.y - self.centre.y) / self.scale) as f64 + self.sheet.y * 0.5)
    }

    fn uv(&self, q: [f32; 2]) -> egui::Pos2 {
        self.screen(uve::to_sheet(q, self.sheet))
    }
}

/// Draw the sheet and the layout: every island's edges (thinned out on big meshes), the UV
/// breaks stronger, the selected islands filled.
fn paint_layout(painter: &egui::Painter, view: &View, layout: &Layout, selected: &[bool], visuals: &egui::Visuals, budget: usize) {
    let sheet = egui::Rect::from_two_pos(view.screen(DVec2::ZERO), view.screen(view.sheet));
    painter.rect_filled(sheet, 0.0, visuals.extreme_bg_color);
    let faint = egui::Stroke::new(0.5, visuals.text_color().gamma_multiply(0.12));
    for k in 1..10 {
        let f = k as f64 / 10.0;
        painter.line_segment([view.screen(DVec2::new(view.sheet.x * f, 0.0)), view.screen(DVec2::new(view.sheet.x * f, view.sheet.y))], faint);
        painter.line_segment([view.screen(DVec2::new(0.0, view.sheet.y * f)), view.screen(DVec2::new(view.sheet.x, view.sheet.y * f))], faint);
    }
    let c = &layout.uvs.corners;
    let isl = &layout.islands;
    let is_sel = |t: u32| isl.island_of(t).is_some_and(|i| selected.get(i).copied().unwrap_or(false));
    // Selected islands filled.
    let fill = ui_util::ACCENT.gamma_multiply(0.35);
    let sel_tris: Vec<u32> = isl.islands.iter().enumerate().filter(|(i, _)| selected.get(*i).copied().unwrap_or(false)).flat_map(|(_, ts)| ts.iter().copied()).collect();
    if !sel_tris.is_empty() {
        let step = sel_tris.len().div_ceil(budget * 2).max(1);
        let mut m = egui::Mesh::default();
        for &t in sel_tris.iter().step_by(step) {
            if let Some(q) = c.get(t as usize) {
                let i = m.vertices.len() as u32;
                for p in q {
                    m.colored_vertex(view.uv(*p), fill);
                }
                m.add_triangle(i, i + 1, i + 2);
            }
        }
        painter.add(egui::Shape::mesh(m));
    }
    // Inner edges, thinned out.
    let wire = egui::Stroke::new(0.5, visuals.text_color().gamma_multiply(0.45));
    let n = isl.of_tri.iter().filter(|&&i| i != u32::MAX).count();
    let step = n.div_ceil(budget).max(1);
    for q in c.iter().zip(&isl.of_tri).filter(|(_, &i)| i != u32::MAX).map(|(q, _)| q).step_by(step) {
        painter.add(egui::Shape::closed_line(q.iter().map(|p| view.uv(*p)).collect(), wire));
    }
    // Breaks.
    let edge = egui::Stroke::new(1.0, visuals.text_color().gamma_multiply(0.8));
    let edge_sel = egui::Stroke::new(1.5, ui_util::ACCENT);
    let step = layout.border.len().div_ceil(budget * 4).max(1);
    for &(t, k) in layout.border.iter().step_by(step) {
        if let Some(q) = c.get(t as usize) {
            let (a, b) = (q[k as usize], q[(k as usize + 1) % 3]);
            painter.line_segment([view.uv(a), view.uv(b)], if is_sel(t) { edge_sel } else { edge });
        }
    }
    painter.rect_stroke(sheet, 0.0, egui::Stroke::new(1.0, visuals.widgets.inactive.fg_stroke.color), egui::StrokeKind::Outside);
}

/// Handles of a selection box on screen: the four corners and the turn handle.
struct Handles {
    rect: egui::Rect,
    corners: [egui::Pos2; 4],
    turn: egui::Pos2,
}

impl Handles {
    fn new(view: &View, lo: DVec2, hi: DVec2) -> Self {
        let rect = egui::Rect::from_two_pos(view.screen(lo), view.screen(hi));
        Self { rect, corners: [rect.left_bottom(), rect.right_bottom(), rect.right_top(), rect.left_top()], turn: rect.center_top() - egui::vec2(0.0, 24.0) }
    }
}

const HANDLE: f32 = 6.0;

impl UnwrapTool {
    /// The UV Layout window's controls and canvas.
    fn editor_ui(&mut self, ui: &mut egui::Ui, cx: &mut ToolCx) {
        self.sync_selection();
        let Some(layout) = self.shown.layout.clone() else {
            ui.label(if self.job.is_some() { "Unwrapping…" } else { "No UVs yet." });
            return;
        };
        let sheet = self.sheet();
        let mut status: Option<Result<String, String>> = None;
        ui.horizontal_wrapped(|ui| {
            ui.label("Select");
            if ui.button("All").clicked() {
                self.editor.selected.iter_mut().for_each(|s| *s = true);
            }
            if ui.button("None").clicked() {
                self.editor.selected.iter_mut().for_each(|s| *s = false);
            }
            if ui.button("Invert").clicked() {
                self.editor.selected.iter_mut().for_each(|s| *s = !*s);
            }
            ui.separator();
            if ui.add_enabled(!self.undo.is_empty(), egui::Button::new("Undo")).on_hover_text("⌘Z").clicked() {
                self.undo(cx);
            }
            if ui.add_enabled(!self.redo.is_empty(), egui::Button::new("Redo")).on_hover_text("⇧⌘Z").clicked() {
                self.redo(cx);
            }
            ui.separator();
            if ui.button("Frame").on_hover_text("Show the whole sheet (or double-click the background)").clicked() {
                self.editor.zoom = 1.0;
                self.editor.pan = egui::Vec2::ZERO;
            }
            ui.checkbox(&mut self.editor.snap, "Snap").on_hover_text("Moved islands snap to the sheet's edges and centre");
        });
        let any = self.editor.selected.contains(&true);
        ui.horizontal_wrapped(|ui| {
            ui.add_enabled_ui(any, |ui| {
                if ui.button("⟲ 90°").clicked() {
                    status = Some(self.edit_selected(cx, |c, i, w, s| uve::rotate_islands(c, i, w, 90.0, s)).map(|_| "Turned".into()));
                }
                if ui.button("⟳ 90°").clicked() {
                    status = Some(self.edit_selected(cx, |c, i, w, s| uve::rotate_islands(c, i, w, -90.0, s)).map(|_| "Turned".into()));
                }
                if ui.button("Flip U").on_hover_text("Mirror left to right").clicked() {
                    status = Some(self.edit_selected(cx, |c, i, w, s| uve::flip_islands(c, i, w, true, s)).map(|_| "Flipped".into()));
                }
                if ui.button("Flip V").on_hover_text("Mirror top to bottom").clicked() {
                    status = Some(self.edit_selected(cx, |c, i, w, s| uve::flip_islands(c, i, w, false, s)).map(|_| "Flipped".into()));
                }
                let border = self.s.unwrap.border;
                if ui.button("Fit in sheet").on_hover_text("Move the selection onto the sheet, shrinking it if it is bigger").clicked() {
                    status = Some(self.edit_selected(cx, |c, i, w, s| uve::fit_into_sheet(c, i, w, s, border)).map(|_| "Fitted".into()));
                }
                if ui.button("Re-pack the rest").on_hover_text("Keep the selection where it is; pack the other islands around it").clicked() {
                    status = Some(self.repack(cx, false));
                }
            });
            if ui.button("Re-pack all").on_hover_text("Pack every island on the sheet, as turned now").clicked() {
                status = Some(self.repack(cx, true));
            }
        });
        // Exact values for the selection.
        let which = self.which();
        if let Some((lo, hi)) = uve::islands_bounds(&layout.uvs.corners, &layout.islands, &which, sheet) {
            let (mut pos, mut size) = (lo, hi - lo);
            let mut turn_now = false;
            let mut resp = Vec::new();
            ui.horizontal_wrapped(|ui| {
                ui.label("Position");
                resp.push(("pos", ui.add(egui::DragValue::new(&mut pos.x).speed(0.5).prefix("X ").suffix(" mm"))));
                resp.push(("pos", ui.add(egui::DragValue::new(&mut pos.y).speed(0.5).prefix("Y ").suffix(" mm"))));
                ui.label("Size");
                resp.push(("w", ui.add(egui::DragValue::new(&mut size.x).speed(0.5).range(0.001..=1e6).prefix("W ").suffix(" mm"))));
                resp.push(("h", ui.add(egui::DragValue::new(&mut size.y).speed(0.5).range(0.001..=1e6).prefix("H ").suffix(" mm"))));
                ui.checkbox(&mut self.editor.keep_ratio, "Keep proportions");
                ui.add(egui::DragValue::new(&mut self.editor.turn_by).speed(1.0).range(-360.0..=360.0).suffix("°"));
                turn_now = ui.button("Turn").on_hover_text("Turn the selection about its centre (counter-clockwise)").clicked();
            });
            for (what, r) in resp {
                if !r.changed() {
                    continue;
                }
                // One undo step per drag of a field.
                let coalesce = r.dragged() && !r.drag_started();
                let new_size = match (what, self.editor.keep_ratio) {
                    ("w", true) => DVec2::new(size.x, size.x * (hi - lo).y / (hi - lo).x.max(1e-12)),
                    ("h", true) => DVec2::new(size.y * (hi - lo).x / (hi - lo).y.max(1e-12), size.y),
                    _ => size,
                };
                let f = new_size / (hi - lo).max(DVec2::splat(1e-12));
                let xf = DAffine2::from_translation(pos) * DAffine2::from_scale(if what == "pos" { DVec2::ONE } else { f }) * DAffine2::from_translation(-lo);
                if coalesce {
                    let mut corners = layout.uvs.corners.clone();
                    uve::transform_islands(&mut corners, &layout.islands, &which, xf, sheet);
                    self.shown = Shown { layout: Some(layout.with(corners)), made: self.shown.made.clone(), edited: true };
                    self.show(cx);
                } else {
                    status = Some(self.edit_selected(cx, |c, i, w, s| uve::transform_islands(c, i, w, xf, s)).map(|_| "Moved".into()));
                }
                break;
            }
            if turn_now {
                let a = self.editor.turn_by;
                status = Some(self.edit_selected(cx, |c, i, w, s| uve::rotate_islands(c, i, w, a, s)).map(|_| "Turned".into()));
            }
            let tris: usize = which.iter().map(|&i| layout.islands.islands[i].len()).sum();
            ui.small(format!("{} of {} islands selected ({} triangles)", which.len(), layout.islands.len(), ui_util::format_count(tris)));
        } else {
            ui.small(format!("{} islands. Click an island or drag a box to select; click the model to select its island.", layout.islands.len()));
        }
        if let Some(Err(e)) = status {
            *cx.status = format!("UV layout: {e}");
        }
        // Canvas.
        let layout = self.shown.layout.clone().unwrap_or(layout);
        // A fixed id: the rows above come and go with the selection, and a drag must not lose
        // its widget when they do.
        let (rect, _) = ui.allocate_exact_size(ui.available_size().max(egui::vec2(200.0, 200.0)), egui::Sense::hover());
        let response = ui.interact(rect, egui::Id::NULL.with("uv_layout_canvas"), egui::Sense::click_and_drag());
        self.canvas(ui, cx, rect, &response, &layout);
    }

    fn canvas(&mut self, ui: &mut egui::Ui, cx: &mut ToolCx, rect: egui::Rect, response: &egui::Response, layout: &Layout) {
        let sheet = self.sheet();
        let view = View::fit(rect, sheet, self.editor.zoom, self.editor.pan);
        self.editor.canvas = Some(rect);
        let mods = ui.input(|i| i.modifiers);
        let pointer = response.interact_pointer_pos().or(response.hover_pos());
        // Wheel / pinch zoom about the pointer.
        if response.hovered() {
            let line = ui.ctx().options(|o| o.input_options.line_scroll_speed);
            let (wheel, pinch) = ui.input(|i| crate::navigation::wheel_zoom(&i.raw.events, line));
            let f = (wheel * 0.0025).exp() * pinch;
            if f != 1.0 {
                if let Some(p) = response.hover_pos() {
                    let z = (self.editor.zoom * f).clamp(0.2, 200.0);
                    let k = z / self.editor.zoom;
                    // Keep the point under the pointer still.
                    let c = rect.center() + self.editor.pan;
                    self.editor.pan += (p - c) * (1.0 - k);
                    self.editor.zoom = z;
                }
            }
        }
        if response.double_clicked() {
            self.editor.zoom = 1.0;
            self.editor.pan = egui::Vec2::ZERO;
        }
        let which = self.which();
        let bounds = uve::islands_bounds(&layout.uvs.corners, &layout.islands, &which, sheet);
        let handles = bounds.map(|(lo, hi)| Handles::new(&view, lo, hi));
        let island_at = |p: egui::Pos2| uve::island_at(&layout.uvs.corners, &layout.islands, view.sheet_at(p) / sheet);
        // Start a drag.
        if response.drag_started_by(egui::PointerButton::Secondary) || response.drag_started_by(egui::PointerButton::Middle) {
            self.editor.drag = Some(Drag::Pan);
        } else if response.drag_started_by(egui::PointerButton::Primary) {
            let p = ui.input(|i| i.pointer.press_origin()).or(pointer).unwrap_or(rect.center());
            let near = |q: egui::Pos2| q.distance(p) <= HANDLE + 3.0;
            let start = view.sheet_at(p);
            self.editor.drag = Some(match (&handles, bounds) {
                (Some(h), Some((lo, hi))) if near(h.turn) => {
                    let centre = (lo + hi) * 0.5;
                    Drag::Turn { base: layout.clone(), centre, from: (start - centre).to_angle() }
                }
                (Some(h), Some((lo, hi))) if h.corners.iter().any(|&q| near(q)) => {
                    let k = h.corners.iter().position(|&q| near(q)).unwrap_or(0);
                    let pts = [lo, DVec2::new(hi.x, lo.y), hi, DVec2::new(lo.x, hi.y)];
                    Drag::Scale { base: layout.clone(), pivot: pts[(k + 2) % 4], from: pts[k] }
                }
                _ => match island_at(p) {
                    Some(i) => {
                        if !self.editor.selected.get(i).copied().unwrap_or(false) {
                            self.select(&[i], mods.shift);
                        }
                        Drag::Move { base: layout.clone(), from: start }
                    }
                    None => Drag::Select { from: p, to: p, add: mods.shift },
                },
            });
            // Moving, turning and scaling are one undo step each.
            if matches!(self.editor.drag, Some(Drag::Move { .. } | Drag::Turn { .. } | Drag::Scale { .. })) {
                self.undo.push(self.shown.clone());
                if self.undo.len() > UNDO_STEPS {
                    self.undo.remove(0);
                }
                self.redo.clear();
            }
        }
        // Follow it.
        let live = layout.islands.of_tri.len() <= LIVE_DRAG_TRIANGLES;
        if let Some(drag) = &mut self.editor.drag {
            let cur = pointer.map(|p| view.sheet_at(p));
            let which = self.editor.selected.iter().enumerate().filter(|(_, &s)| s).map(|(i, _)| i).collect::<Vec<_>>();
            let xf = match (drag, cur) {
                (Drag::Pan, _) => {
                    self.editor.pan += response.drag_delta();
                    None
                }
                (Drag::Select { to, .. }, _) => {
                    if let Some(p) = pointer {
                        *to = p;
                    }
                    None
                }
                (Drag::Move { base, from }, Some(cur)) => {
                    let mut d = cur - *from;
                    if self.editor.snap {
                        if let Some((lo, hi)) = uve::islands_bounds(&base.uvs.corners, &base.islands, &which, sheet) {
                            d += uve::snap_to_sheet(lo + d, hi + d, sheet, 8.0 / view.scale as f64);
                        }
                    }
                    Some((base.clone(), DAffine2::from_translation(d)))
                }
                (Drag::Turn { base, centre, from }, Some(cur)) => {
                    let mut a = (cur - *centre).to_angle() - *from;
                    if mods.shift {
                        let step = 15f64.to_radians();
                        a = (a / step).round() * step;
                    }
                    Some((base.clone(), uve::about(*centre, DAffine2::from_angle(a))))
                }
                (Drag::Scale { base, pivot, from }, Some(cur)) => {
                    let (d0, d1) = (*from - *pivot, cur - *pivot);
                    let f = if mods.shift {
                        DVec2::new(if d0.x.abs() > 1e-12 { d1.x / d0.x } else { 1.0 }, if d0.y.abs() > 1e-12 { d1.y / d0.y } else { 1.0 })
                    } else {
                        DVec2::splat(d1.dot(d0) / d0.length_squared().max(1e-24))
                    };
                    let f = f.signum() * f.abs().max(DVec2::splat(1e-3));
                    Some((base.clone(), uve::about(*pivot, DAffine2::from_scale(f))))
                }
                _ => None,
            };
            if let Some((base, xf)) = xf {
                let mut corners = base.uvs.corners.clone();
                uve::transform_islands(&mut corners, &base.islands, &which, xf, sheet);
                self.shown = Shown { layout: Some(base.with(corners)), made: self.shown.made.clone(), edited: true };
                if live {
                    self.show(cx);
                }
            }
        }
        let released = !ui.input(|i| i.pointer.any_down());
        if released {
            match self.editor.drag.take() {
                Some(Drag::Select { from, to, add }) => {
                    let (a, b) = (view.sheet_at(from) / sheet, view.sheet_at(to) / sheet);
                    let hit = uve::islands_in_box(&layout.uvs.corners, &layout.islands, a, b);
                    self.select(&hit, add);
                }
                Some(Drag::Move { .. } | Drag::Turn { .. } | Drag::Scale { .. }) => self.show(cx),
                _ => {}
            }
        }
        if response.clicked() {
            if let Some(p) = response.interact_pointer_pos() {
                match island_at(p) {
                    Some(i) if mods.shift => {
                        self.sync_selection();
                        if let Some(s) = self.editor.selected.get_mut(i) {
                            *s = !*s;
                        }
                    }
                    Some(i) => self.select(&[i], false),
                    None if !mods.shift => self.select(&[], false),
                    None => {}
                }
            }
        }
        // Paint.
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, ui.visuals().faint_bg_color);
        let layout = self.shown.layout.clone().unwrap_or_else(|| layout.clone());
        let view = View::fit(rect, sheet, self.editor.zoom, self.editor.pan);
        paint_layout(&painter, &view, &layout, &self.editor.selected, ui.visuals(), 40_000);
        let which = self.which();
        if let Some((lo, hi)) = uve::islands_bounds(&layout.uvs.corners, &layout.islands, &which, sheet) {
            let h = Handles::new(&view, lo, hi);
            let stroke = egui::Stroke::new(1.0, ui_util::ACCENT);
            painter.rect_stroke(h.rect, 0.0, stroke, egui::StrokeKind::Middle);
            painter.line_segment([h.rect.center_top(), h.turn], stroke);
            painter.circle(h.turn, HANDLE, ui.visuals().extreme_bg_color, stroke);
            for q in h.corners {
                painter.rect(egui::Rect::from_center_size(q, egui::vec2(2.0 * HANDLE - 2.0, 2.0 * HANDLE - 2.0)), 0.0, ui.visuals().extreme_bg_color, stroke, egui::StrokeKind::Middle);
            }
        }
        if let Some(Drag::Select { from, to, .. }) = &self.editor.drag {
            let r = egui::Rect::from_two_pos(*from, *to);
            painter.rect(r, 0.0, ui_util::ACCENT.gamma_multiply(0.1), egui::Stroke::new(1.0, ui_util::ACCENT), egui::StrokeKind::Middle);
        }
        if let Some(p) = response.hover_pos() {
            let s = view.sheet_at(p);
            painter.text(rect.left_bottom() + egui::vec2(6.0, -4.0), egui::Align2::LEFT_BOTTOM, format!("U {:.3}  V {:.3}  ({:.1}, {:.1} mm)", s.x / sheet.x, s.y / sheet.y, s.x, s.y), egui::FontId::monospace(11.0), ui.visuals().weak_text_color());
        }
        if self.editor.drag.is_some() {
            ui.ctx().request_repaint();
        }
    }

    /// The SetUV settings. Returns whether they changed.
    fn projection_ui(&mut self, ui: &mut egui::Ui, cx: &mut ToolCx) -> bool {
        let before = (self.s.projection, self.view);
        let object = self.object;
        let Some(selected) = object.and_then(|id| cx.scene.get(id)).map(|o| o.selection.len()) else { return false };
        let aspect = self.aspect();
        let p = &mut self.s.projection;
        egui::ComboBox::from_label("Projection")
            .selected_text(match p.projection {
                Projection::Planar => "Planar",
                Projection::Cylindrical => "Cylindrical",
                Projection::Spherical => "Spherical",
                Projection::Box => "Box",
            })
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut p.projection, Projection::Planar, "Planar").on_hover_text("Straight along the axis, like a slide projector");
                ui.selectable_value(&mut p.projection, Projection::Cylindrical, "Cylindrical").on_hover_text("Around the axis: U is the angle, V the height");
                ui.selectable_value(&mut p.projection, Projection::Spherical, "Spherical").on_hover_text("U is the longitude, V the latitude");
                ui.selectable_value(&mut p.projection, Projection::Box, "Box").on_hover_text("Each face straight along the axis closest to its normal");
            });
        ui.horizontal(|ui| {
            egui::ComboBox::from_id_salt("setuv_axis")
                .selected_text(match p.axis {
                    ProjectionAxis::View => "View",
                    ProjectionAxis::X => "X",
                    ProjectionAxis::Y => "Y",
                    ProjectionAxis::Z => "Z",
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut p.axis, ProjectionAxis::View, "View").on_hover_text("Along the view, as it was when taken");
                    ui.selectable_value(&mut p.axis, ProjectionAxis::X, "X");
                    ui.selectable_value(&mut p.axis, ProjectionAxis::Y, "Y");
                    ui.selectable_value(&mut p.axis, ProjectionAxis::Z, "Z");
                });
            ui.label("Axis");
            if ui.button("From view").on_hover_text("Take the current view's direction").clicked() {
                p.axis = ProjectionAxis::View;
                self.view = (cx.camera.right(), cx.camera.up());
            }
        });
        let mut auto = p.center.is_none();
        if ui.checkbox(&mut auto, "Centre in the middle").on_hover_text("The middle of the projected triangles").changed() {
            p.center = (!auto).then(|| targets(cx.scene, object).map_or(DVec3::ZERO, |(mesh, tris, _)| uv_project::projection_center(&mesh, &tris)).to_array());
        }
        if let Some(c) = &mut p.center {
            ui.horizontal(|ui| {
                for (v, l) in c.iter_mut().zip(["X ", "Y ", "Z "]) {
                    ui.add(egui::DragValue::new(v).speed(0.5).prefix(l));
                }
            });
        }
        if p.projection != Projection::Spherical {
            let mut fit = p.size.is_none();
            let hint = if p.projection == Projection::Cylindrical { "V only: U is one turn across the sheet" } else { "Model length across the sheet's width" };
            if ui.checkbox(&mut fit, "Fit on the sheet").on_hover_text(hint).changed() {
                p.size = (!fit).then(|| {
                    let Some((mesh, tris, _)) = targets(cx.scene, object) else { return 100.0 };
                    let centre = p.center.map(DVec3::from).unwrap_or_else(|| uv_project::projection_center(&mesh, &tris));
                    let f = uv_project::projection_frame(p, centre, Some(self.view));
                    uv_project::fit_size(&mesh, &tris, p, &f, aspect)
                });
            }
            if let Some(s) = &mut p.size {
                ui.add(egui::DragValue::new(s).speed(0.5).range(1e-6..=1e9).prefix("Size ").suffix(" mm")).on_hover_text(hint);
            }
        }
        egui::Grid::new("setuv").num_columns(3).show(ui, |ui| {
            ui.label("Scale");
            ui.add(egui::DragValue::new(&mut p.scale[0]).speed(0.01).prefix("U "));
            ui.add(egui::DragValue::new(&mut p.scale[1]).speed(0.01).prefix("V "));
            ui.end_row();
            ui.label("Offset");
            ui.add(egui::DragValue::new(&mut p.offset[0]).speed(0.005).prefix("U "));
            ui.add(egui::DragValue::new(&mut p.offset[1]).speed(0.005).prefix("V "));
            ui.end_row();
        });
        ui.add(egui::Slider::new(&mut p.rotation, -180.0..=180.0).suffix("°").text("rotation")).on_hover_text("Turn the projection about its axis");
        ui.small(if selected > 0 { format!("Projects the {selected} selected triangles; the others keep their UVs.") } else { "Projects the whole object (select triangles first to project part of it).".to_string() });
        (self.s.projection, self.view) != before
    }
}

/// The small layout picture in the panel.
fn draw_layout(ui: &mut egui::Ui, layout: &Layout, selected: &[bool], sheet: DVec2) -> egui::Response {
    let w = ui.available_width().min(260.0);
    let h = w * (sheet.y / sheet.x) as f32;
    let (rect, response) = ui.allocate_exact_size(egui::vec2(w, h.min(400.0)), egui::Sense::click());
    let view = View::fit(rect, sheet, 1.0, egui::Vec2::ZERO);
    paint_layout(&ui.painter_at(rect), &view, layout, selected, ui.visuals(), 12_000);
    response.on_hover_text("Open the UV Layout editor")
}

impl Tool for UnwrapTool {
    fn name(&self) -> &'static str {
        "Unwrap"
    }

    fn settings(&self) -> Option<serde_json::Value> {
        serde_json::to_value(self.s).ok()
    }

    fn settings_schema(&self) -> Option<serde_json::Value> {
        Some(schema_of::<UnwrapSettings>())
    }

    fn set_settings(&mut self, changes: &serde_json::Value, cx: &mut ToolCx) -> Result<(), String> {
        let old = self.s;
        self.s = merged(&self.s, changes).map_err(|e| format!("Unwrap: {e}"))?;
        match self.s.method {
            UvMethod::Unwrap if old.method != UvMethod::Unwrap || old.unwrap != self.s.unwrap || self.shown.layout.is_none() => {
                self.project_base = None;
                self.start(cx);
            }
            UvMethod::Unwrap => {}
            UvMethod::Project => self.project(cx),
        }
        Ok(())
    }

    fn state(&self, _scene: &Scene) -> serde_json::Value {
        let layout = self.shown.layout.as_ref();
        let which = self.which();
        serde_json::json!({
            "object": self.object,
            "computing": self.job.is_some(),
            "result": self.shown.made.as_ref().map(|(m, msg)| serde_json::json!({
                "message": msg,
                "for_current_settings": match (m, self.s.method) {
                    (Made::Unwrap(p), UvMethod::Unwrap) => *p == self.s.unwrap,
                    (Made::Project(p, ..), UvMethod::Project) => *p == self.s.projection,
                    _ => false,
                },
                "edited": self.shown.edited,
            })),
            "islands": layout.map(|l| l.islands.len()),
            "selected_islands": which,
            "selection_box_mm": layout.and_then(|l| uve::islands_bounds(&l.uvs.corners, &l.islands, &which, self.sheet())).map(|(lo, hi)| [lo.x, lo.y, hi.x, hi.y]),
            "undo_steps": self.undo.len(),
            "error": self.error,
            "next": "accept (mesh_tool accept) to keep the UVs (written with OBJ export); edit the layout with the actions",
        })
    }

    fn actions(&self) -> Vec<Action> {
        vec![
            Action::new("select_islands", "Select UV islands", serde_json::json!({
                "islands": arg::text("all, none, invert, or island numbers separated by commas"),
                "triangle": arg::integer("Or: the island of this triangle"),
                "add": arg::boolean("Add to the selection instead of replacing it"),
            })),
            Action::new("move", "Move the selected islands (mm on the sheet)", serde_json::json!({"du": arg::number("Right, in mm"), "dv": arg::number("Up, in mm")})).required(&["du", "dv"]),
            Action::new("rotate", "Turn the selected islands about their centre", serde_json::json!({"degrees": arg::number("Counter-clockwise")})).required(&["degrees"]),
            Action::new("scale", "Scale the selected islands about their centre", serde_json::json!({"factor": arg::number("Scale (V too unless factor_v is given)"), "factor_v": arg::number("V scale")})).required(&["factor"]),
            Action::new("flip", "Mirror the selected islands", serde_json::json!({"axis": arg::choice(&["u", "v"], "u: left to right; v: top to bottom")})).required(&["axis"]),
            Action::plain("fit_in_sheet", "Move the selected islands onto the sheet (shrinking them if needed)"),
            Action::new("repack", "Pack the islands on the sheet", serde_json::json!({"which": arg::choice(&["rest", "all"], "rest: keep the selection, pack the others around it")})).required(&["which"]),
            Action::plain("undo", "Undo the last change in the tool"),
            Action::plain("redo", "Redo"),
        ]
    }

    fn act(&mut self, action: &str, args: &serde_json::Value, cx: &mut ToolCx) -> Result<String, String> {
        let a = Args(args);
        self.sync_selection();
        let r = match action {
            "select_islands" => {
                let n = self.editor.selected.len();
                let add = a.opt_bool("add").unwrap_or(false);
                if let Some(t) = a.opt_usize("triangle") {
                    let i = self.shown.layout.as_ref().and_then(|l| l.islands.island_of(t as u32)).ok_or("no island has that triangle")?;
                    self.select(&[i], add);
                } else {
                    match a.str("islands")?.trim() {
                        "all" => self.editor.selected.iter_mut().for_each(|s| *s = true),
                        "none" => self.select(&[], false),
                        "invert" => self.editor.selected.iter_mut().for_each(|s| *s = !*s),
                        list => {
                            let ids: Vec<usize> = list.split(',').map(|x| x.trim().parse::<usize>().ok().filter(|&i| i < n)).collect::<Option<_>>().ok_or(format!("islands must be all, none, invert or numbers below {n}"))?;
                            self.select(&ids, add);
                        }
                    }
                }
                Ok(format!("{} islands selected", self.which().len()))
            }
            "move" => {
                let d = DVec2::new(a.f64("du")?, a.f64("dv")?);
                self.edit_selected(cx, |c, i, w, s| uve::move_islands(c, i, w, d, s)).map(|_| "Moved".into())
            }
            "rotate" => {
                let deg = a.f64("degrees")?;
                self.edit_selected(cx, |c, i, w, s| uve::rotate_islands(c, i, w, deg, s)).map(|_| "Turned".into())
            }
            "scale" => {
                let f = a.f64("factor")?;
                let f = DVec2::new(f, a.opt_f64("factor_v").unwrap_or(f));
                if f.x == 0.0 || f.y == 0.0 || !f.is_finite() {
                    return Err("the factor must not be zero".into());
                }
                self.edit_selected(cx, |c, i, w, s| uve::scale_islands(c, i, w, f, s)).map(|_| "Scaled".into())
            }
            "flip" => {
                let h = a.str("axis")? == "u";
                self.edit_selected(cx, |c, i, w, s| uve::flip_islands(c, i, w, h, s)).map(|_| "Flipped".into())
            }
            "fit_in_sheet" => {
                let border = self.s.unwrap.border;
                self.edit_selected(cx, |c, i, w, s| uve::fit_into_sheet(c, i, w, s, border)).map(|_| "Fitted".into())
            }
            "repack" => self.repack(cx, a.str("which")? == "all"),
            "undo" => self.undo(cx).then(|| "Undone".to_string()).ok_or_else(|| "nothing to undo".to_string()),
            "redo" => self.redo(cx).then(|| "Redone".to_string()).ok_or_else(|| "nothing to redo".to_string()),
            _ => Err(format!("Unwrap has no action {action}")),
        };
        r
    }

    fn shows_uvs(&self) -> bool {
        true
    }

    fn busy(&self) -> Option<(String, f32)> {
        self.job.as_ref().map(|(_, j)| ("Unwrapping".to_string(), j.progress()))
    }

    fn keys(&mut self, input: &egui::InputState, cx: &mut ToolCx) -> bool {
        // ⌘Z / ⇧⌘Z step through the tool's own changes while it has some (the scene's undo
        // would close the tool).
        if input.modifiers.command && input.key_pressed(egui::Key::Z) && !cx.egui.egui_wants_keyboard_input() {
            return if input.modifiers.shift { self.redo(cx) } else { self.undo(cx) };
        }
        false
    }

    fn pointer(&mut self, input: &PointerInput, cx: &mut ToolCx) {
        // Clicking the model selects the island under the pointer.
        if !input.clicked {
            return;
        }
        let Some(pos) = input.pos else { return };
        let Some((_, hit)) = cx.pick(pos, self.object) else { return };
        let Some(i) = self.shown.layout.as_ref().and_then(|l| l.islands.island_of(hit.tid)) else { return };
        self.sync_selection();
        if input.shift {
            if let Some(s) = self.editor.selected.get_mut(i) {
                *s = !*s;
            }
        } else {
            self.select(&[i], false);
        }
        self.editor.open = true;
    }

    fn overlay(&mut self, ov: &mut Overlay, cx: &mut ToolCx) {
        // The selected islands, tinted (pulled a little toward the eye so they win the depth test).
        let Some(layout) = &self.shown.layout else { return };
        let Some(o) = self.object.and_then(|id| cx.scene.get(id)) else { return };
        if layout.uvs.topo != o.mesh.topology_version() {
            return;
        }
        let eye = cx.camera.eye();
        let col = Vec4::new(1.0, 0.62, 0.2, 0.38);
        let mut n = 0;
        for (i, _) in self.editor.selected.iter().enumerate().filter(|(_, &s)| s) {
            for &t in layout.islands.islands.get(i).into_iter().flatten() {
                if n > 200_000 {
                    return;
                }
                n += 1;
                let [a, b, c] = o.mesh.tri_positions(t).map(|p| p + (eye - p) * 0.0015);
                ov.tri(a, b, c, col);
            }
        }
    }

    fn panel(&mut self, ui: &mut egui::Ui, cx: &mut ToolCx) -> ToolOutcome {
        self.poll(cx);
        ui_util::section(ui, "Unwrap");
        if self.mesh(cx).is_none() {
            ui.label("No active object.");
            return if ui.button("Close").clicked() { ToolOutcome::Close } else { ToolOutcome::Continue };
        }
        let method = self.s.method;
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.s.method, UvMethod::Unwrap, "Unwrap").on_hover_text("Flatten charts and pack them on the sheet");
            ui.selectable_value(&mut self.s.method, UvMethod::Project, "Project (SetUV)").on_hover_text("Project the selection (or the whole object) without unwrapping");
        });
        if self.s.method != method {
            self.project_base = None;
            if self.s.method == UvMethod::Project {
                self.project(cx);
            }
        }
        let p = &mut self.s.unwrap;
        if self.s.method == UvMethod::Unwrap {
            egui::ComboBox::from_label("Unwrap Elements")
                .selected_text(if p.elements == UnwrapElements::Groups { "By Groups" } else { "By Components" })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut p.elements, UnwrapElements::Groups, "By Groups").on_hover_text("One chart per face group (cut further where needed)");
                    ui.selectable_value(&mut p.elements, UnwrapElements::Components, "By Components");
                });
            egui::ComboBox::from_label("Unwrap Type")
                .selected_text(if p.kind == UnwrapType::Rigid { "Rigid" } else { "Conformal" })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut p.kind, UnwrapType::Rigid, "Rigid").on_hover_text("Keeps lengths and areas (as rigid as possible)");
                    ui.selectable_value(&mut p.kind, UnwrapType::Conformal, "Conformal").on_hover_text("Keeps angles");
                });
            egui::ComboBox::from_label("Packing Type")
                .selected_text(match p.packing {
                    Packing::Stacked => "Stacked",
                    Packing::Line => "Line",
                    Packing::Optimized => "Optimized",
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut p.packing, Packing::Optimized, "Optimized");
                    ui.selectable_value(&mut p.packing, Packing::Line, "Line");
                    ui.selectable_value(&mut p.packing, Packing::Stacked, "Stacked");
                });
        }
        ui.label(if self.s.method == UvMethod::Unwrap { "Pack Settings" } else { "Sheet" });
        let sheet_before = (p.width, p.height);
        egui::Grid::new("pack").num_columns(2).show(ui, |ui| {
            let rows: Vec<(&str, &mut f64)> = if self.s.method == UvMethod::Unwrap {
                vec![("Width", &mut p.width), ("Height", &mut p.height), ("Border", &mut p.border)]
            } else {
                vec![("Width", &mut p.width), ("Height", &mut p.height)]
            };
            for (label, v) in rows {
                ui.label(label);
                ui.add(egui::DragValue::new(v).speed(1.0).range(0.0..=10_000.0).suffix(" mm"));
                ui.end_row();
            }
        });
        let sheet_changed = (p.width, p.height) != sheet_before;
        match self.s.method {
            UvMethod::Unwrap => {
                let current = self.s.unwrap;
                let stale = self.shown.made.as_ref().is_none_or(|m| m.0 != Made::Unwrap(current)) && self.job.as_ref().is_none_or(|j| j.0 != current);
                if ui.add_enabled(stale, egui::Button::new(if self.shown.layout.is_some() { "Update" } else { "Unwrap" })).clicked() {
                    self.project_base = None;
                    self.start(cx);
                }
                if let Some((_, j)) = &self.job {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.add(egui::ProgressBar::new(j.progress()));
                    });
                    cx.egui.request_repaint_after(std::time::Duration::from_millis(50));
                }
            }
            UvMethod::Project => {
                if self.projection_ui(ui, cx) || sheet_changed {
                    self.project(cx);
                }
            }
        }
        if let Some((_, msg)) = &self.shown.made {
            ui.label(msg);
        }
        if self.shown.edited {
            ui.small("Edited in the layout window.");
        }
        self.sync_selection();
        if let Some(layout) = self.shown.layout.clone() {
            if draw_layout(ui, &layout, &self.editor.selected, self.sheet()).clicked() {
                self.editor.open = true;
            }
            ui.horizontal(|ui| {
                if ui.selectable_label(self.editor.open, "Edit Layout…").on_hover_text("Select, move, turn, scale and re-pack islands in a larger window").clicked() {
                    self.editor.open = !self.editor.open;
                }
                if ui.add_enabled(!self.undo.is_empty(), egui::Button::new("Undo")).on_hover_text("⌘Z").clicked() {
                    self.undo(cx);
                }
                if ui.add_enabled(!self.redo.is_empty(), egui::Button::new("Redo")).on_hover_text("⇧⌘Z").clicked() {
                    self.redo(cx);
                }
            });
        }
        if let Some(e) = &self.error {
            ui.colored_label(ui.visuals().warn_fg_color, e);
        }
        ui.small("UVs are kept while the shape changes (sculpting, transforms) and dropped by edits that change triangles. Export them with OBJ.");
        let (accept, cancel) = ui_util::accept_cancel(ui, self.shown.layout.is_some() && self.job.is_none());
        if self.editor.open && self.shown.layout.is_some() {
            let ctx = cx.egui.clone();
            let mut open = true;
            egui::Window::new("UV Layout")
                .id(egui::Id::NULL.with("uv_layout_editor"))
                .open(&mut open)
                .default_size([560.0, 640.0])
                .min_size([320.0, 320.0])
                .resizable(true)
                .collapsible(true)
                .show(&ctx, |ui| self.editor_ui(ui, cx));
            self.editor.open &= open;
        }
        if accept && self.accept(cx) {
            return ToolOutcome::Close;
        }
        if cancel {
            self.cancel(cx);
            return ToolOutcome::Close;
        }
        ToolOutcome::Continue
    }

    fn accept(&mut self, cx: &mut ToolCx) -> bool {
        self.poll(cx);
        if self.job.is_some() {
            return false;
        }
        let Some(layout) = self.shown.layout.clone() else { return true };
        let unchanged = matches!(self.shown.made, Some((Made::Existing, _))) && !self.shown.edited && self.undo.is_empty();
        let msg = self.shown.made.as_ref().map(|m| m.1.clone()).unwrap_or_default();
        // Snapshot with the UVs from before, then apply.
        self.restore(cx);
        if unchanged {
            return true;
        }
        let label = self.label();
        cx.history.push(label, cx.scene);
        if let Some(o) = self.object.and_then(|id| cx.scene.get_mut(id)) {
            o.uvs = Some(layout.uvs);
        }
        *cx.status = if self.shown.edited { format!("{label}: {msg}; layout edited") } else { format!("{label}: {msg}") };
        true
    }

    fn cancel(&mut self, cx: &mut ToolCx) -> bool {
        if let Some((_, j)) = &self.job {
            j.cancel();
        }
        self.restore(cx);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::History;
    use mm_core::primitives;
    use std::time::{Duration, Instant};

    fn with_tool(mesh: Mesh, f: impl FnOnce(&mut ToolCx, ObjectId)) {
        let mut scene = Scene::default();
        let id = scene.add("part", mesh);
        let (mut history, mut camera, mut status) = (History::default(), crate::camera::Camera::default(), String::new());
        let ctx = egui::Context::default();
        let vp = Viewport { rect: egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0)) };
        let mut cx = ToolCx { scene: &mut scene, history: &mut history, camera: &mut camera, status: &mut status, egui: &ctx, vp, brush_scroll: crate::navigation::BrushScroll::Shift };
        f(&mut cx, id);
    }

    fn wait(tool: &mut UnwrapTool, cx: &mut ToolCx) {
        let t0 = Instant::now();
        while tool.job.is_some() && t0.elapsed() < Duration::from_secs(20) {
            std::thread::sleep(Duration::from_millis(5));
            tool.poll(cx);
        }
        assert!(tool.job.is_none() && tool.error.is_none(), "{:?}", tool.error);
    }

    /// Two separate squares (two islands).
    fn two_squares() -> Mesh {
        let mut m = Mesh::new();
        for i in 0..2 {
            let mut q = primitives::plane_grid(20.0, 10.0, 4, 2);
            q.transform(|p| p + DVec3::new(i as f64 * 50.0, 0.0, 0.0));
            m.append_mesh(&q, i);
        }
        m
    }

    #[test]
    fn unwrap_edit_islands_undo_and_accept() {
        with_tool(two_squares(), |cx, id| {
            let mut tool = UnwrapTool::new(cx);
            wait(&mut tool, cx);
            let layout = tool.shown.layout.clone().unwrap();
            assert_eq!(layout.islands.len(), 2);
            let uv0 = layout.uvs.corners.clone();
            // Move island 0 by (10, 5) mm: island 1 stays.
            tool.act("select_islands", &serde_json::json!({"islands": "0"}), cx).unwrap();
            tool.act("move", &serde_json::json!({"du": 10.0, "dv": 5.0}), cx).unwrap();
            let uv1 = tool.shown.layout.as_ref().unwrap().uvs.corners.clone();
            let sheet = tool.sheet();
            for t in cx.scene.get(id).unwrap().mesh.triangle_ids() {
                let moved = layout.islands.island_of(t) == Some(0);
                for k in 0..3 {
                    let d = uve::to_sheet(uv1[t as usize][k], sheet) - uve::to_sheet(uv0[t as usize][k], sheet);
                    let want = if moved { DVec2::new(10.0, 5.0) } else { DVec2::ZERO };
                    assert!((d - want).length() < 1e-3, "{t}: {d}");
                }
            }
            // The object shows the edit at once; undo / redo inside the tool.
            assert_eq!(cx.scene.get(id).unwrap().uvs().unwrap().corners, uv1);
            tool.act("undo", &serde_json::Value::Null, cx).unwrap();
            assert_eq!(cx.scene.get(id).unwrap().uvs().unwrap().corners, uv0);
            tool.act("redo", &serde_json::Value::Null, cx).unwrap();
            // Turn and flip island 1 too, then accept: one undo step for the scene.
            tool.act("select_islands", &serde_json::json!({"triangle": layout.islands.islands[1][0]}), cx).unwrap();
            tool.act("rotate", &serde_json::json!({"degrees": 90.0}), cx).unwrap();
            tool.act("flip", &serde_json::json!({"axis": "u"}), cx).unwrap();
            let last = tool.shown.layout.as_ref().unwrap().uvs.corners.clone();
            assert!(tool.accept(cx));
            assert_eq!(cx.scene.get(id).unwrap().uvs().unwrap().corners, last);
            assert!(cx.status.starts_with("Unwrap"), "{}", cx.status);
            // Reopening edits the UVs the object has instead of unwrapping again.
            let tool = UnwrapTool::new(cx);
            assert!(tool.job.is_none());
            assert_eq!(tool.shown.layout.as_ref().unwrap().uvs.corners, last);
        });
    }

    #[test]
    fn unwrap_layout_window_clicks_and_drags_islands() {
        with_tool(two_squares(), |cx, _| {
            let mut tool = UnwrapTool::new(cx);
            wait(&mut tool, cx);
            tool.editor.open = true;
            tool.editor.snap = false;
            let ctx = cx.egui.clone();
            let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1600.0, 1200.0));
            let frame = |tool: &mut UnwrapTool, cx: &mut ToolCx, events: Vec<egui::Event>| {
                let raw = egui::RawInput { screen_rect: Some(screen), events, ..Default::default() };
                let _ = ctx.run_ui(raw, |ui| {
                    let _ = tool.panel(ui, cx);
                });
            };
            frame(&mut tool, cx, vec![]);
            frame(&mut tool, cx, vec![]);
            assert!(tool.editor.canvas.is_some(), "the layout window draws its canvas");
            let view = |tool: &UnwrapTool| View::fit(tool.editor.canvas.unwrap(), tool.sheet(), tool.editor.zoom, tool.editor.pan);
            let layout = tool.shown.layout.clone().unwrap();
            // Where island `i` is on screen now.
            let centre = |tool: &UnwrapTool, i: usize| {
                let t = layout.islands.islands[i][0] as usize;
                let c = layout.uvs.corners[t];
                view(tool).uv([(c[0][0] + c[1][0] + c[2][0]) / 3.0, (c[0][1] + c[1][1] + c[2][1]) / 3.0])
            };
            let press = |p: egui::Pos2, pressed: bool| egui::Event::PointerButton { pos: p, button: egui::PointerButton::Primary, pressed, modifiers: egui::Modifiers::NONE };
            // Click island 1: it alone is selected.
            let p = centre(&tool, 1);
            frame(&mut tool, cx, vec![egui::Event::PointerMoved(p), press(p, true)]);
            frame(&mut tool, cx, vec![press(p, false)]);
            frame(&mut tool, cx, vec![]);
            assert_eq!(tool.which(), vec![1]);
            // Drag island 0 right by 80 points: it moves by 80 / scale mm, island 1 stays.
            let p = centre(&tool, 0);
            let scale = view(&tool).scale as f64;
            frame(&mut tool, cx, vec![egui::Event::PointerMoved(p), press(p, true)]);
            for k in 1..=4 {
                frame(&mut tool, cx, vec![egui::Event::PointerMoved(p + egui::vec2(20.0 * k as f32, 0.0))]);
            }
            frame(&mut tool, cx, vec![press(p + egui::vec2(80.0, 0.0), false)]);
            frame(&mut tool, cx, vec![]);
            assert_eq!(tool.which(), vec![0], "dragging an unselected island selects it");
            let after = tool.shown.layout.clone().unwrap();
            let sheet = tool.sheet();
            let moved = |i: usize| {
                let t = layout.islands.islands[i][0] as usize;
                uve::to_sheet(after.uvs.corners[t][0], sheet) - uve::to_sheet(layout.uvs.corners[t][0], sheet)
            };
            let want = 80.0 / scale;
            assert!((moved(0) - DVec2::new(want, 0.0)).length() < 0.05 * want, "{} vs {want}", moved(0));
            assert_eq!(moved(1), DVec2::ZERO);
            assert_eq!(cx.scene.active().unwrap().uvs().unwrap().corners, after.uvs.corners, "the model shows the drag");
            assert_eq!(tool.undo.len(), 1, "the drag is one undo step");
            // A box around everything selects both islands.
            let canvas = tool.editor.canvas.unwrap();
            let (a, b) = (canvas.left_top() + egui::vec2(3.0, 3.0), canvas.right_bottom() - egui::vec2(3.0, 3.0));
            frame(&mut tool, cx, vec![egui::Event::PointerMoved(a), press(a, true)]);
            frame(&mut tool, cx, vec![egui::Event::PointerMoved(a.lerp(b, 0.5))]);
            frame(&mut tool, cx, vec![egui::Event::PointerMoved(b)]);
            frame(&mut tool, cx, vec![press(b, false)]);
            assert_eq!(tool.which(), vec![0, 1]);
            // Only island 1 again; drag its corner handle outward: it grows about the opposite
            // corner; then its turn handle sideways: it turns about its centre.
            tool.select(&[1], false);
            frame(&mut tool, cx, vec![]);
            let area = |tool: &UnwrapTool| {
                let l = tool.shown.layout.as_ref().unwrap();
                l.islands.islands[1].iter().map(|&t| {
                    let q = l.uvs.corners[t as usize].map(|c| uve::to_sheet(c, sheet));
                    (q[1] - q[0]).perp_dot(q[2] - q[0]).abs() * 0.5
                }).sum::<f64>()
            };
            let bounds = |tool: &UnwrapTool| {
                let l = tool.shown.layout.as_ref().unwrap();
                uve::islands_bounds(&l.uvs.corners, &l.islands, &[1], sheet).unwrap()
            };
            let drag = |tool: &mut UnwrapTool, cx: &mut ToolCx, from: egui::Pos2, by: egui::Vec2| {
                frame(tool, cx, vec![egui::Event::PointerMoved(from), press(from, true)]);
                for k in 1..=4 {
                    frame(tool, cx, vec![egui::Event::PointerMoved(from + by * (k as f32 / 4.0))]);
                }
                frame(tool, cx, vec![press(from + by, false)]);
                frame(tool, cx, vec![]);
            };
            let (a0, (lo, hi)) = (area(&tool), bounds(&tool));
            let h = Handles::new(&view(&tool), lo, hi);
            drag(&mut tool, cx, h.corners[2], egui::vec2(30.0, -30.0));
            let (lo2, _) = bounds(&tool);
            assert!(area(&tool) > a0 * 1.2, "{} vs {a0}", area(&tool));
            assert!((lo2 - lo).length() < 1e-3, "the opposite corner stays: {lo} {lo2}");
            let (a1, (lo, hi)) = (area(&tool), bounds(&tool));
            let h = Handles::new(&view(&tool), lo, hi);
            drag(&mut tool, cx, h.turn, egui::vec2(60.0, 0.0));
            let (lo3, hi3) = bounds(&tool);
            assert!((area(&tool) - a1).abs() < 1e-3 * a1, "turning keeps the area");
            assert!(((lo3 + hi3) - (lo + hi)).length() * 0.5 < 1e-3, "about the centre");
            assert!((hi3 - lo3 - (hi - lo)).length() > 1e-3, "the box changes as it turns");
            assert_eq!(tool.undo.len(), 3);
            // ⌘Z inside the tool steps back.
            assert!(tool.undo(cx));
            assert!(((bounds(&tool).1 - bounds(&tool).0) - (hi - lo)).length() < 1e-6);
        });
    }

    #[test]
    fn unwrap_set_uvs_starts_projecting() {
        with_tool(primitives::cube(10.0, 2), |cx, id| {
            let mut tool = UnwrapTool::new_projection(cx);
            assert!(tool.job.is_none(), "no unwrap behind the projection");
            assert!(matches!(tool.shown.made, Some((Made::Project(..), _))));
            assert!(cx.scene.get(id).unwrap().uvs().is_some());
            tool.set_settings(&serde_json::json!({"projection": {"projection": "Box", "axis": "Z"}}), cx).unwrap();
            // A box projection of a cube: one island per side.
            assert_eq!(tool.shown.layout.as_ref().unwrap().islands.len(), 6);
            assert!(tool.accept(cx));
            assert!(cx.status.starts_with("Set UVs"));
        });
    }

    #[test]
    fn unwrap_panels_draw_in_both_methods() {
        with_tool(primitives::icosphere(10.0, 2), |cx, _| {
            let mut tool = UnwrapTool::new(cx);
            wait(&mut tool, cx);
            tool.editor.open = true;
            let ctx = cx.egui.clone();
            let run = |tool: &mut UnwrapTool, cx: &mut ToolCx| {
                let raw = egui::RawInput { screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1200.0, 900.0))), ..Default::default() };
                let _ = ctx.run_ui(raw, |ui| {
                    let _ = tool.panel(ui, cx);
                });
            };
            run(&mut tool, cx);
            for projection in ["Planar", "Cylindrical", "Spherical", "Box"] {
                tool.set_settings(&serde_json::json!({"method": "project", "projection": {"projection": projection, "center": [1.0, 2.0, 3.0], "size": 25.0}}), cx).unwrap();
                tool.select(&[0], false);
                run(&mut tool, cx);
                run(&mut tool, cx);
                assert!(tool.error.is_none(), "{projection}: {:?}", tool.error);
                assert!(!tool.shown.layout.as_ref().unwrap().islands.is_empty());
            }
        });
    }

    #[test]
    fn unwrap_cancel_restores_and_repack_keeps_the_selection() {
        with_tool(two_squares(), |cx, id| {
            let mut tool = UnwrapTool::new(cx);
            wait(&mut tool, cx);
            tool.act("select_islands", &serde_json::json!({"islands": "1"}), cx).unwrap();
            tool.act("move", &serde_json::json!({"du": 60.0, "dv": 100.0}), cx).unwrap();
            let fixed: Vec<Corners> = {
                let l = tool.shown.layout.as_ref().unwrap();
                l.islands.islands[1].iter().map(|&t| l.uvs.corners[t as usize]).collect()
            };
            tool.act("repack", &serde_json::json!({"which": "rest"}), cx).unwrap();
            let l = tool.shown.layout.clone().unwrap();
            assert_eq!(l.islands.islands[1].iter().map(|&t| l.uvs.corners[t as usize]).collect::<Vec<_>>(), fixed);
            assert!(tool.act("move", &serde_json::json!({"du": 1.0}), cx).is_err(), "dv is required");
            tool.cancel(cx);
            assert!(cx.scene.get(id).unwrap().uvs().is_none(), "cancel puts back no UVs");
        });
    }

    #[test]
    fn unwrap_tool_setuv_projects_the_selection() {
        // A 20 × 10 plate in the XY plane: a planar projection along Z on a square sheet maps
        // its corners to the sheet's (fit) and keeps proportions.
        with_tool(primitives::plane_grid(20.0, 10.0, 2, 1), |cx, id| {
            let mut tool = UnwrapTool::new(cx);
            wait(&mut tool, cx);
            tool.set_settings(&serde_json::json!({"method": "project", "width": 100.0, "height": 100.0, "projection": {"projection": "Planar", "axis": "Z"}}), cx).unwrap();
            let o = cx.scene.get(id).unwrap();
            let uv = &o.uvs().unwrap().corners;
            for t in o.mesh.triangle_ids() {
                for (k, p) in o.mesh.tri_positions(t).into_iter().enumerate() {
                    let want = [p.x / 20.0 + 0.5, p.y / 20.0 + 0.5];
                    let got = uv[t as usize][k];
                    assert!((got[0] as f64 - want[0]).abs() < 1e-6 && (got[1] as f64 - want[1]).abs() < 1e-6, "{p}: {got:?}");
                }
            }
            // Project only the selected triangle (cylindrically); the other keeps its UVs.
            let keep = uv[1];
            cx.scene.get_mut(id).unwrap().selection.insert(0);
            tool.set_settings(&serde_json::json!({"projection": {"projection": "Cylindrical"}}), cx).unwrap();
            let o = cx.scene.get(id).unwrap();
            assert_eq!(o.uvs().unwrap().corners[1], keep);
            assert!(tool.state(cx.scene)["result"]["message"].as_str().unwrap().contains("1 selected"));
            assert!(tool.accept(cx));
            assert!(cx.status.starts_with("Set UVs"), "{}", cx.status);
        });
    }
}
