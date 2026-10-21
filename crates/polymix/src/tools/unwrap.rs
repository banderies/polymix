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
use crate::jobs::{Job, ProgressFn};
use crate::renderer::ColorVertex;
use crate::scene::{next_stamp, UvSet};
use crate::ui_util;
use glam::DAffine2;
use mm_edit::uv_edit::{self as uve, Corners, UvIslands};
use mm_edit::uv_project::{self, Projection, ProjectionAxis, SetUvParams};
use mm_edit::{Packing, UnwrapElements, UnwrapParams, UnwrapType};
use std::sync::Arc;
use std::time::{Duration, Instant};

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
    /// Triangles in the islands (those with UVs).
    tris: usize,
}

impl Layout {
    fn new(mesh: &Mesh, uvs: Arc<UvSet>) -> Self {
        let (islands, border) = uve::uv_islands_and_breaks(mesh, &uvs.corners);
        let tris = islands.islands.iter().map(Vec::len).sum();
        Self { uvs, islands: Arc::new(islands), border: Arc::new(border), tris }
    }

    /// The same islands with other corners (an edit).
    fn with(&self, corners: Vec<Corners>) -> Self {
        Self { uvs: Arc::new(UvSet { topo: self.uvs.topo, corners, stamp: next_stamp() }), islands: self.islands.clone(), border: self.border.clone(), tris: self.tris }
    }
}

/// What a SetUV projection works from, taken from the tool and the scene when it starts.
struct ProjectionInput {
    mesh: Arc<Mesh>,
    tris: Vec<u32>,
    /// The selection rather than the whole object.
    partial: bool,
    p: SetUvParams,
    view: (DVec3, DVec3),
    aspect: f64,
    /// The UVs the other triangles keep.
    base: Option<Arc<UvSet>>,
}

/// A projected layout and its summary, or why the projection failed.
type Projected = Result<(Layout, String), String>;

impl ProjectionInput {
    /// The projection. `progress` hears of each step; it stops the work (`None`) when it
    /// returns false.
    fn compute(self, progress: ProgressFn) -> Option<Projected> {
        let Self { mesh, tris, partial, p, view, aspect, base } = self;
        let q = match uv_project::project_uvs(&mesh, &tris, &p, Some(view), aspect) {
            Ok(q) => q,
            Err(e) => return Some(Err(e.to_string())),
        };
        if !progress(0.5) {
            return None;
        }
        let mut corners: Vec<Corners> = base.map(|b| b.corners.clone()).unwrap_or_default();
        corners.resize(mesh.max_tid(), [[0.0; 2]; 3]);
        for (k, &t) in tris.iter().enumerate() {
            corners[t as usize] = q[k].map(|c| [c.x as f32, c.y as f32]);
        }
        if !progress(0.6) {
            return None;
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
        Some(Ok((layout, msg)))
    }
}

/// SetUV settings bit for bit, so that a value typed as "nan" is the same value from one
/// frame to the next (and is projected once, to report it, rather than waited on forever).
type SettingsKey = (Projection, ProjectionAxis, Option<[u64; 3]>, Option<u64>, [u64; 2], [u64; 2], u64);

fn settings_key(p: &SetUvParams) -> SettingsKey {
    (p.projection, p.axis, p.center.map(|c| c.map(f64::to_bits)), p.size.map(f64::to_bits), p.scale.map(f64::to_bits), p.offset.map(f64::to_bits), p.rotation.to_bits())
}

/// What a background projection was started for: its result is shown only while all of it is
/// still so.
#[derive(Clone, PartialEq)]
struct ProjectionTag {
    p: SettingsKey,
    view: [u64; 6],
    aspect: u64,
    /// The UVs shown when it started.
    shown: Option<u64>,
}

/// How long the SetUV settings must stay unchanged before they are projected (a drag of a
/// field or slider projects once it pauses, not at every step).
const PROJECT_AFTER: Duration = Duration::from_millis(180);

/// The layout drawing (see [`paint_layout`]): everything it was drawn from, and once it has
/// stayed the same for a frame, its triangles.
struct Picture {
    uvs: u64,
    islands: Arc<UvIslands>,
    border: Arc<Vec<(u32, u8)>>,
    selected: Vec<bool>,
    view: (DVec2, f32, egui::Pos2),
    budget: usize,
    clip: egui::Rect,
    pixels_per_point: f32,
    options: egui::epaint::TessellationOptions,
    colours: [egui::Color32; 3],
    mesh: Option<Arc<egui::Mesh>>,
}

/// A selection box (see [`uve::islands_bounds`]) with what it was computed from.
struct KeptBounds {
    uvs: u64,
    islands: Arc<UvIslands>,
    which: Vec<usize>,
    sheet: DVec2,
    bounds: Option<(DVec2, DVec2)>,
}

/// The box of the islands `which` of `layout` (see [`uve::islands_bounds`]), from `kept` when
/// it was computed for the same UVs: a frame asks for it several times (the fields, the
/// handles, a move's snapping), and once all islands are selected each scan reads every
/// corner.
fn island_bounds(kept: &mut Vec<KeptBounds>, layout: &Layout, which: &[usize], sheet: DVec2) -> Option<(DVec2, DVec2)> {
    if let Some(k) = kept.iter().find(|k| k.uvs == layout.uvs.stamp && Arc::ptr_eq(&k.islands, &layout.islands) && k.which == which && k.sheet == sheet) {
        return k.bounds;
    }
    let bounds = uve::islands_bounds(&layout.uvs.corners, &layout.islands, which, sheet);
    // A few: a drag asks for its starting layout's box and the moved one's.
    kept.insert(0, KeptBounds { uvs: layout.uvs.stamp, islands: layout.islands.clone(), which: which.to_vec(), sheet, bounds });
    kept.truncate(4);
    bounds
}

/// The selected islands' tint in the 3D view, with what it was made from.
struct Tint {
    islands: Arc<UvIslands>,
    selected: Vec<bool>,
    shape: u64,
    eye: DVec3,
    tris: Vec<ColorVertex>,
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
    /// What the current move, turn or scale drag did last: its transform, the islands it
    /// moved and the UVs that made (a frame with the same transform has nothing to do).
    applied: Option<(DAffine2, Vec<usize>, u64)>,
    /// Selection boxes of the layouts shown lately (see [`island_bounds`]).
    bounds: Vec<KeptBounds>,
    /// The layout window's drawing.
    picture: Option<Picture>,
}

impl Default for Editor {
    fn default() -> Self {
        Self {
            open: false,
            selected: Vec::new(),
            of: 0,
            zoom: 1.0,
            pan: egui::Vec2::ZERO,
            drag: None,
            snap: true,
            keep_ratio: true,
            turn_by: 45.0,
            canvas: None,
            applied: None,
            bounds: Vec::new(),
            picture: None,
        }
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
    /// When the SetUV settings changed, while they wait to settle before being projected.
    projecting: Option<Instant>,
    /// A projection computing in the background, and what it is for.
    project_job: Option<(ProjectionTag, Job<Option<Projected>>)>,
    /// The object's own UVs being read into a layout (Unwrap opened on an object with UVs).
    loading: Option<Job<Layout>>,
    error: Option<String>,
    editor: Editor,
    /// The panel's small drawing of the layout.
    thumbnail: Option<Picture>,
    /// The selected islands' tint in the 3D view.
    tint: Option<Tint>,
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
            projecting: None,
            project_job: None,
            loading: None,
            error: None,
            editor: Editor::default(),
            thumbnail: None,
            tint: None,
        };
        // Edit the UVs the object has; unwrap otherwise.
        let existing = t.object.and_then(|id| cx.scene.get(id)).and_then(|o| Some((o.mesh.clone(), o.uvs.clone().filter(|u| u.topo == o.mesh.topology_version())?)));
        match existing {
            // Finding the islands of a big layout takes a moment: the panel waits for it
            // (and shows it) instead of the tool opening late.
            Some((mesh, uvs)) if method == UvMethod::Unwrap => {
                t.loading = Some(Job::spawn("Reading the UVs", cx.egui.clone(), move |_| Ok(Layout::new(&mesh, Arc::new(UvSet::new(&mesh, uvs.corners.clone()))))));
            }
            // A projection follows at once, and keeps these as its undo step.
            Some((mesh, uvs)) => t.show_existing(Layout::new(&mesh, Arc::new(UvSet::new(&mesh, uvs.corners.clone())))),
            None if method == UvMethod::Unwrap => t.start(cx),
            None => {}
        }
        if method == UvMethod::Project {
            t.project(cx);
        }
        t
    }

    /// Show the object's own UVs (as read when the tool opened).
    fn show_existing(&mut self, layout: Layout) {
        let msg = format!("The object's UVs: {} islands", layout.islands.len());
        self.shown = Shown { layout: Some(layout), made: Some((Made::Existing, msg)), edited: false };
    }

    /// Take the object's own layout once it has been read.
    fn poll_loading(&mut self) {
        if let Some(r) = self.loading.as_ref().and_then(|j| j.poll()) {
            self.loaded(r);
        }
    }

    /// Wait for the object's own layout if it is still being read: what follows (an unwrap,
    /// a projection, an edit) acts on it, as when it was read on opening.
    fn settle_loading(&mut self) {
        if let Some(job) = self.loading.take() {
            self.loaded(wait_for(&job));
        }
    }

    /// The object's own layout, read: shown unless something else is by now.
    fn loaded(&mut self, r: anyhow::Result<Layout>) {
        self.loading = None;
        match r {
            Ok(layout) if self.shown.layout.is_none() => self.show_existing(layout),
            Ok(_) => {}
            Err(e) => self.error = Some(format!("{e:#}")),
        }
    }

    fn mesh(&self, cx: &ToolCx) -> Option<Arc<Mesh>> {
        self.object.and_then(|id| cx.scene.get(id)).map(|o| o.mesh.clone())
    }

    fn sheet(&self) -> DVec2 {
        DVec2::new(self.s.unwrap.width.max(1e-9), self.s.unwrap.height.max(1e-9))
    }

    fn start(&mut self, cx: &mut ToolCx) {
        // The unwrap's result is an undo step on top of the object's own layout.
        self.settle_loading();
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

    /// SetUV: project now (agents' settings, opening, switching to it), in place of any
    /// projection still waiting or computing.
    fn project(&mut self, cx: &mut ToolCx) {
        self.projecting = None;
        self.project_job = None;
        let Some(input) = self.projection_input(cx) else { return };
        if let Some(r) = input.compute(&|_| true) {
            self.projected(cx, r);
        }
    }

    /// What a projection of the current settings works from.
    fn projection_input(&mut self, cx: &mut ToolCx) -> Option<ProjectionInput> {
        // The object's own layout, if it is still being read, is what a projection replaces.
        self.settle_loading();
        // An unwrap still running would replace the projection when it ends.
        if let Some((_, j)) = self.job.take() {
            j.cancel();
        }
        let (mesh, tris, partial) = self.targets(cx)?;
        // The other triangles keep the UVs they had when these triangles were first projected
        // (so adjusting the projection does not pile up).
        if self.project_base.as_ref().is_none_or(|b| b.0 != tris) {
            let own = self.object.and_then(|id| cx.scene.get(id)).and_then(|o| o.uvs.clone());
            self.project_base = Some((tris.clone(), self.shown.layout.as_ref().map(|l| l.uvs.clone()).or(own)));
        }
        let base = self.project_base.clone().and_then(|b| b.1).filter(|b| b.topo == mesh.topology_version());
        Some(ProjectionInput { mesh, tris, partial, p: self.s.projection, view: self.view, aspect: self.aspect(), base })
    }

    /// Show a projection of the current settings.
    fn projected(&mut self, cx: &mut ToolCx, r: Projected) {
        match r {
            Ok((layout, msg)) => {
                // Changing the projection's settings replaces the last projection (one undo
                // step for the whole adjustment) unless it was edited since.
                let record = self.shown.edited || !matches!(self.shown.made, Some((Made::Project(..), _)));
                let made = Made::Project(self.s.projection, self.view_key(), self.aspect());
                self.set(cx, Shown { layout: Some(layout), made: Some((made, msg)), edited: false }, record);
                self.error = None;
            }
            Err(e) => self.error = Some(format!("SetUV: {e}")),
        }
    }

    /// The SetUV settings changed in the panel: they are projected once they have stayed so
    /// for a moment (in the background; a projection of a big object takes a while).
    fn project_soon(&mut self) {
        // One of older settings is of no use now.
        self.project_job = None;
        self.projecting = Some(Instant::now());
    }

    fn projection_tag(&self) -> ProjectionTag {
        ProjectionTag { p: settings_key(&self.s.projection), view: self.view_key().map(f64::to_bits), aspect: self.aspect().to_bits(), shown: self.shown.layout.as_ref().map(|l| l.uvs.stamp) }
    }

    /// Start the projection of settled settings.
    fn start_projection(&mut self, cx: &mut ToolCx) {
        let Some(changed) = self.projecting else { return };
        let left = PROJECT_AFTER.saturating_sub(changed.elapsed());
        if !left.is_zero() {
            cx.egui.request_repaint_after(left);
            return;
        }
        self.projecting = None;
        let Some(input) = self.projection_input(cx) else { return };
        let tag = self.projection_tag();
        self.project_job = Some((tag, Job::spawn("Projecting", cx.egui.clone(), move |progress| Ok(input.compute(progress)))));
    }

    /// Take a finished background projection.
    fn poll_projection(&mut self, cx: &mut ToolCx) {
        if let Some(r) = self.project_job.as_ref().and_then(|(_, j)| j.poll()) {
            if let Some((tag, _)) = self.project_job.take() {
                self.projection_done(cx, &tag, r);
            }
        }
    }

    /// A background projection's result: shown if it is still for the settings and the
    /// layout it started from (whatever changes the layout settles projections first, so
    /// it is).
    fn projection_done(&mut self, cx: &mut ToolCx, tag: &ProjectionTag, r: anyhow::Result<Option<Projected>>) {
        match r {
            Ok(Some(r)) if *tag == self.projection_tag() => self.projected(cx, r),
            Ok(_) => {}
            Err(e) => self.error = Some(format!("SetUV: {e:#}")),
        }
    }

    /// Finish what the layout still waits for (the object's own layout being read, SetUV
    /// settings waiting to settle or being projected) before acting on it or on its islands,
    /// so that edits, undo and accepting act on what the settings show, as when projections
    /// were taken at once.
    fn settle(&mut self, cx: &mut ToolCx) {
        self.settle_loading();
        if let Some((tag, job)) = self.project_job.take() {
            let r = wait_for(&job);
            self.projection_done(cx, &tag, r);
        }
        if self.projecting.is_some() {
            self.project(cx);
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
        self.settle(cx);
        let Some(prev) = self.undo.pop() else { return false };
        self.editor.drag = None;
        self.redo.push(std::mem::replace(&mut self.shown, prev));
        self.show(cx);
        true
    }

    fn redo(&mut self, cx: &mut ToolCx) -> bool {
        self.settle(cx);
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
        self.settle(cx);
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
        self.settle(cx);
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

/// The result of `job`, waiting for it here (as long as the work took when it was done in
/// place).
fn wait_for<T: Send + 'static>(job: &Job<T>) -> anyhow::Result<T> {
    loop {
        if let Some(r) = job.poll() {
            return r;
        }
        std::thread::sleep(Duration::from_millis(1));
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

/// The sheet and the layout: every island's edges (thinned out on big meshes), the UV breaks
/// stronger, the selected islands filled.
fn layout_shapes(view: &View, layout: &Layout, selected: &[bool], visuals: &egui::Visuals, budget: usize) -> Vec<egui::Shape> {
    let mut shapes = Vec::new();
    let sheet = egui::Rect::from_two_pos(view.screen(DVec2::ZERO), view.screen(view.sheet));
    shapes.push(egui::epaint::RectShape::filled(sheet, 0.0, visuals.extreme_bg_color).into());
    let faint = egui::Stroke::new(0.5, visuals.text_color().gamma_multiply(0.12));
    for k in 1..10 {
        let f = k as f64 / 10.0;
        shapes.push(egui::Shape::LineSegment { points: [view.screen(DVec2::new(view.sheet.x * f, 0.0)), view.screen(DVec2::new(view.sheet.x * f, view.sheet.y))], stroke: faint });
        shapes.push(egui::Shape::LineSegment { points: [view.screen(DVec2::new(0.0, view.sheet.y * f)), view.screen(DVec2::new(view.sheet.x, view.sheet.y * f))], stroke: faint });
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
        shapes.push(egui::Shape::mesh(m));
    }
    // Inner edges, thinned out.
    let wire = egui::Stroke::new(0.5, visuals.text_color().gamma_multiply(0.45));
    let step = layout.tris.div_ceil(budget).max(1);
    for q in c.iter().zip(&isl.of_tri).filter(|(_, &i)| i != u32::MAX).map(|(q, _)| q).step_by(step) {
        shapes.push(egui::Shape::closed_line(q.iter().map(|p| view.uv(*p)).collect(), wire));
    }
    // Breaks.
    let edge = egui::Stroke::new(1.0, visuals.text_color().gamma_multiply(0.8));
    let edge_sel = egui::Stroke::new(1.5, ui_util::ACCENT);
    let step = layout.border.len().div_ceil(budget * 4).max(1);
    for &(t, k) in layout.border.iter().step_by(step) {
        if let Some(q) = c.get(t as usize) {
            let (a, b) = (q[k as usize], q[(k as usize + 1) % 3]);
            shapes.push(egui::Shape::LineSegment { points: [view.uv(a), view.uv(b)], stroke: if is_sel(t) { edge_sel } else { edge } });
        }
    }
    shapes.push(egui::epaint::RectShape::stroke(sheet, 0.0, egui::Stroke::new(1.0, visuals.widgets.inactive.fg_stroke.color), egui::StrokeKind::Outside).into());
    shapes
}

/// Draw [`layout_shapes`]. Tens of thousands of thin lines took milliseconds to tessellate
/// again every frame the panel or window was drawn, so once the picture has stayed the same
/// for a frame they are tessellated here, the way egui does at the end of the frame (the same
/// pixels per point, options and culling by the painter's clip), and the triangles are kept in
/// `kept` with everything they were drawn from. A picture that changes every frame (a drag, a
/// zoom) is drawn as shapes, as before, rather than tessellated twice.
fn paint_layout(painter: &egui::Painter, view: &View, layout: &Layout, selected: &[bool], visuals: &egui::Visuals, budget: usize, kept: &mut Option<Picture>) {
    let ctx = painter.ctx();
    let pixels_per_point = ctx.pixels_per_point();
    let options = ctx.tessellation_options(|o| *o);
    let colours = [visuals.extreme_bg_color, visuals.text_color(), visuals.widgets.inactive.fg_stroke.color];
    let clip = painter.clip_rect();
    let same = |p: &Picture| {
        p.uvs == layout.uvs.stamp
            && Arc::ptr_eq(&p.islands, &layout.islands)
            && Arc::ptr_eq(&p.border, &layout.border)
            && p.selected == selected
            && p.view == (view.sheet, view.scale, view.centre)
            && p.budget == budget
            && p.clip == clip
            && p.pixels_per_point == pixels_per_point
            && p.options == options
            && p.colours == colours
    };
    match kept {
        Some(p) if same(p) => {
            let mesh = p.mesh.get_or_insert_with(|| {
                // Lines, rectangles and plain triangles use no font texture and no prepared
                // discs.
                let mut tessellator = egui::epaint::Tessellator::new(pixels_per_point, options, ctx.fonts(|f| f.font_image_size()), Vec::new());
                tessellator.set_clip_rect(clip);
                let mut mesh = egui::Mesh::default();
                for shape in layout_shapes(view, layout, selected, visuals, budget) {
                    tessellator.tessellate_shape(shape, &mut mesh);
                }
                Arc::new(mesh)
            });
            painter.add(egui::Shape::Mesh(mesh.clone()));
        }
        _ => {
            painter.extend(layout_shapes(view, layout, selected, visuals, budget));
            *kept = Some(Picture {
                uvs: layout.uvs.stamp,
                islands: layout.islands.clone(),
                border: layout.border.clone(),
                selected: selected.to_vec(),
                view: (view.sheet, view.scale, view.centre),
                budget,
                clip,
                pixels_per_point,
                options,
                colours,
                mesh: None,
            });
        }
    }
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
        // A click or drag here acts on the islands shown: a projection still to come is taken
        // first, as when projections were taken at once.
        let pressed = ui.input(|i| i.pointer.any_pressed() && i.pointer.press_origin().is_some_and(|p| ui.clip_rect().contains(p)));
        if pressed && (self.projecting.is_some() || self.project_job.is_some()) {
            self.settle(cx);
        }
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
        if let Some((lo, hi)) = island_bounds(&mut self.editor.bounds, &layout, &which, sheet) {
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
        self.canvas(ui, cx, rect, &response, layout);
    }

    /// `base` with the islands `which` moved by `xf`, for a move, turn or scale drag. The
    /// drag's own UVs from its last frame are rewritten in place when nothing else holds them
    /// (there only these islands differ from `base`), with a new stamp, instead of copying all
    /// of `base`'s corners every frame; a UV set held anywhere else is never changed.
    fn dragged(&mut self, cx: &mut ToolCx, base: &Layout, which: &[usize], xf: DAffine2, sheet: DVec2) -> Layout {
        let last = self.editor.applied.as_ref().filter(|(_, w, _)| w == which).map(|a| a.2);
        let ours = self.shown.layout.as_ref().is_some_and(|l| Some(l.uvs.stamp) == last && Arc::ptr_eq(&l.islands, &base.islands));
        if let Some(mut layout) = self.shown.layout.take_if(|_| ours) {
            // A live drag shows them on the object too; the caller puts the new ones there.
            if let Some(o) = self.object.and_then(|id| cx.scene.get_mut(id)) {
                if o.uvs.as_ref().is_some_and(|u| Arc::ptr_eq(u, &layout.uvs)) {
                    o.uvs = None;
                }
            }
            if let Some(uvs) = Arc::get_mut(&mut layout.uvs) {
                uve::transform_islands_from(&mut uvs.corners, &base.uvs.corners, &base.islands, which, xf, sheet);
                uvs.stamp = next_stamp();
                return layout;
            }
        }
        let mut corners = base.uvs.corners.clone();
        uve::transform_islands(&mut corners, &base.islands, which, xf, sheet);
        base.with(corners)
    }

    fn canvas(&mut self, ui: &mut egui::Ui, cx: &mut ToolCx, rect: egui::Rect, response: &egui::Response, layout: Layout) {
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
        let bounds = island_bounds(&mut self.editor.bounds, &layout, &which, sheet);
        let handles = bounds.map(|(lo, hi)| Handles::new(&view, lo, hi));
        let island_at = |layout: &Layout, p: egui::Pos2| uve::island_at(&layout.uvs.corners, &layout.islands, view.sheet_at(p) / sheet);
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
                _ => match island_at(&layout, p) {
                    Some(i) => {
                        if !self.editor.selected.get(i).copied().unwrap_or(false) {
                            self.select(&[i], mods.shift);
                        }
                        Drag::Move { base: layout.clone(), from: start }
                    }
                    None => Drag::Select { from: p, to: p, add: mods.shift },
                },
            });
            self.editor.applied = None;
            // Moving, turning and scaling are one undo step each.
            if matches!(self.editor.drag, Some(Drag::Move { .. } | Drag::Turn { .. } | Drag::Scale { .. })) {
                self.undo.push(self.shown.clone());
                if self.undo.len() > UNDO_STEPS {
                    self.undo.remove(0);
                }
                self.redo.clear();
            }
        }
        // Follow it. This frame's own handle on the shown layout goes first, so that a drag's
        // UVs are held only where they are shown (see `dragged`); only a move, turn or scale
        // changes them, and then nothing below needs the layout from before.
        let live = layout.islands.of_tri.len() <= LIVE_DRAG_TRIANGLES;
        drop(layout);
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
                        if let Some((lo, hi)) = island_bounds(&mut self.editor.bounds, base, &which, sheet) {
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
                // Held still: the islands are where this transform put them already.
                let shown = self.shown.layout.as_ref().map(|l| l.uvs.stamp);
                let done = self.editor.applied.as_ref().is_some_and(|a| a.0 == xf && a.1 == which && Some(a.2) == shown);
                if !done {
                    let moved = self.dragged(cx, &base, &which, xf, sheet);
                    self.editor.applied = Some((xf, which, moved.uvs.stamp));
                    self.shown = Shown { layout: Some(moved), made: self.shown.made.clone(), edited: true };
                    if live {
                        self.show(cx);
                    }
                }
            }
        }
        let Some(layout) = self.shown.layout.clone() else { return };
        let released = !ui.input(|i| i.pointer.any_down());
        if released {
            self.editor.applied = None;
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
                match island_at(&layout, p) {
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
        let layout = self.shown.layout.clone().unwrap_or(layout);
        let view = View::fit(rect, sheet, self.editor.zoom, self.editor.pan);
        paint_layout(&painter, &view, &layout, &self.editor.selected, ui.visuals(), 40_000, &mut self.editor.picture);
        let which = self.which();
        if let Some((lo, hi)) = island_bounds(&mut self.editor.bounds, &layout, &which, sheet) {
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
        let before = (settings_key(&self.s.projection), self.view);
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
                    ui.add(egui::DragValue::new(v).speed(0.5).prefix(l).update_while_editing(false));
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
                ui.add(egui::DragValue::new(s).speed(0.5).range(1e-6..=1e9).prefix("Size ").suffix(" mm").update_while_editing(false)).on_hover_text(hint);
            }
        }
        egui::Grid::new("setuv").num_columns(3).show(ui, |ui| {
            ui.label("Scale");
            ui.add(egui::DragValue::new(&mut p.scale[0]).speed(0.01).prefix("U ").update_while_editing(false));
            ui.add(egui::DragValue::new(&mut p.scale[1]).speed(0.01).prefix("V ").update_while_editing(false));
            ui.end_row();
            ui.label("Offset");
            ui.add(egui::DragValue::new(&mut p.offset[0]).speed(0.005).prefix("U ").update_while_editing(false));
            ui.add(egui::DragValue::new(&mut p.offset[1]).speed(0.005).prefix("V ").update_while_editing(false));
            ui.end_row();
        });
        ui.add(egui::Slider::new(&mut p.rotation, -180.0..=180.0).suffix("°").text("rotation").update_while_editing(false)).on_hover_text("Turn the projection about its axis");
        ui.small(if selected > 0 { format!("Projects the {selected} selected triangles; the others keep their UVs.") } else { "Projects the whole object (select triangles first to project part of it).".to_string() });
        (settings_key(&self.s.projection), self.view) != before
    }
}

/// The small layout picture in the panel.
fn draw_layout(ui: &mut egui::Ui, layout: &Layout, selected: &[bool], sheet: DVec2, kept: &mut Option<Picture>) -> egui::Response {
    let w = ui.available_width().min(260.0);
    let h = w * (sheet.y / sheet.x) as f32;
    let (rect, response) = ui.allocate_exact_size(egui::vec2(w, h.min(400.0)), egui::Sense::click());
    let view = View::fit(rect, sheet, 1.0, egui::Vec2::ZERO);
    paint_layout(&ui.painter_at(rect), &view, layout, selected, ui.visuals(), 12_000, kept);
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
        // Changes from the panel come first.
        self.settle(cx);
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
            "computing": self.job.is_some() || self.loading.is_some() || self.projecting.is_some() || self.project_job.is_some(),
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
        self.settle(cx);
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
        if let Some((_, j)) = &self.job {
            return Some(("Unwrapping".to_string(), j.progress()));
        }
        if let Some(j) = &self.loading {
            return Some(("Reading the UVs".to_string(), j.progress()));
        }
        self.project_job.as_ref().map(|(_, j)| ("Projecting".to_string(), j.progress()))
    }

    fn pending(&self) -> bool {
        self.projecting.is_some()
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
        self.settle(cx);
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
        let (selected, shape) = (&self.editor.selected, o.mesh.shape_version());
        // Made again only when the islands, the selection, the shape or the eye change (not
        // for every frame the panel redraws, say).
        let current = self.tint.as_ref().is_some_and(|k| Arc::ptr_eq(&k.islands, &layout.islands) && k.selected == *selected && k.shape == shape && k.eye == eye);
        if !current {
            let col = Vec4::new(1.0, 0.62, 0.2, 0.38);
            let mut tint = Overlay::default();
            let mut n = 0;
            'islands: for (i, _) in selected.iter().enumerate().filter(|(_, &s)| s) {
                for &t in layout.islands.islands.get(i).into_iter().flatten() {
                    if n > 200_000 {
                        break 'islands;
                    }
                    n += 1;
                    let [a, b, c] = o.mesh.tri_positions(t).map(|p| p + (eye - p) * 0.0015);
                    tint.tri(a, b, c, col);
                }
            }
            self.tint = Some(Tint { islands: layout.islands.clone(), selected: selected.clone(), shape, eye, tris: tint.tris });
        }
        if let Some(k) = &self.tint {
            ov.tris.extend_from_slice(&k.tris);
        }
    }

    fn panel(&mut self, ui: &mut egui::Ui, cx: &mut ToolCx) -> ToolOutcome {
        self.poll(cx);
        self.poll_loading();
        self.poll_projection(cx);
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
            // A projection still to come is the last of the method left.
            if method == UvMethod::Project {
                self.settle(cx);
            }
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
            // In SetUV the sheet's proportions are projected: a typed size counts once it is
            // entered, not at each digit.
            let typing = self.s.method == UvMethod::Unwrap;
            for (label, v) in rows {
                ui.label(label);
                ui.add(egui::DragValue::new(v).speed(1.0).range(0.0..=10_000.0).suffix(" mm").update_while_editing(typing));
                ui.end_row();
            }
        });
        let sheet_changed = (p.width, p.height) != sheet_before;
        match self.s.method {
            UvMethod::Unwrap => {
                let current = self.s.unwrap;
                let stale = self.shown.made.as_ref().is_none_or(|m| m.0 != Made::Unwrap(current)) && self.job.as_ref().is_none_or(|j| j.0 != current);
                if ui.add_enabled(stale && self.loading.is_none(), egui::Button::new(if self.shown.layout.is_some() { "Update" } else { "Unwrap" })).clicked() {
                    self.project_base = None;
                    self.start(cx);
                }
                if let Some((_, j)) = &self.job {
                    ui.horizontal(|ui| {
                        polymix_workspace::spinner(ui);
                        ui.add(egui::ProgressBar::new(j.progress()));
                    });
                    cx.egui.request_repaint_after(std::time::Duration::from_millis(50));
                }
                if self.loading.is_some() {
                    ui.horizontal(|ui| {
                        polymix_workspace::spinner(ui);
                        ui.label("Reading the object's UVs…");
                    });
                }
            }
            UvMethod::Project => {
                if self.projection_ui(ui, cx) || sheet_changed {
                    self.project_soon();
                }
                self.start_projection(cx);
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
            if draw_layout(ui, &layout, &self.editor.selected, self.sheet(), &mut self.thumbnail).clicked() {
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
        let waiting = self.job.is_some() || self.projecting.is_some() || self.project_job.is_some();
        let (accept, cancel) = ui_util::accept_cancel(ui, self.shown.layout.is_some() && !waiting);
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
        // What the settings show is what is accepted.
        self.settle(cx);
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
        // Dropped jobs are told to stop.
        self.loading = None;
        self.projecting = None;
        self.project_job = None;
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

    /// Until the unwrap, the reading of the object's UVs and a background projection are done.
    fn wait(tool: &mut UnwrapTool, cx: &mut ToolCx) {
        let t0 = Instant::now();
        while (tool.job.is_some() || tool.loading.is_some() || tool.project_job.is_some()) && t0.elapsed() < Duration::from_secs(20) {
            std::thread::sleep(Duration::from_millis(5));
            tool.poll(cx);
            tool.poll_loading();
            tool.poll_projection(cx);
        }
        assert!(tool.job.is_none() && tool.loading.is_none() && tool.project_job.is_none() && tool.error.is_none(), "{:?}", tool.error);
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
            // Reopening edits the UVs the object has instead of unwrapping again (read in the
            // background).
            let mut tool = UnwrapTool::new(cx);
            assert!(tool.job.is_none());
            wait(&mut tool, cx);
            assert_eq!(tool.shown.layout.as_ref().unwrap().uvs.corners, last);
            assert!(matches!(tool.shown.made, Some((Made::Existing, _))));
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

    /// The scene's undo and redo steps.
    fn history_labels(cx: &ToolCx) -> (Vec<String>, Vec<String>) {
        let (undo, redo) = cx.history.labels();
        (undo.iter().map(|s| s.to_string()).collect(), redo.iter().map(|s| s.to_string()).collect())
    }

    /// An icosphere unwrapped by components (several islands, many breaks).
    fn sphere_layout() -> (Mesh, Layout) {
        let m = primitives::icosphere(10.0, 3);
        let l = mm_edit::unwrap(&m, &UnwrapParams { elements: UnwrapElements::Components, kind: UnwrapType::Conformal, ..Default::default() }).unwrap();
        let corners = l.uv.iter().map(|c| c.map(|q| q.as_vec2().to_array())).collect();
        let layout = Layout::new(&m, Arc::new(UvSet::new(&m, corners)));
        (m, layout)
    }

    #[test]
    fn unwrap_layout_picture_is_what_egui_would_draw() {
        // The kept picture gives the very primitives egui made of the shapes drawn one by one:
        // on screen pixels and on a Retina screen, zoomed in so that most shapes fall outside
        // the clip, and with budgets that thin the lines out.
        let (_, layout) = sphere_layout();
        assert!(layout.islands.len() > 1 && !layout.border.is_empty());
        let mut selected = vec![false; layout.islands.len()];
        selected[1] = true;
        let sheet = DVec2::new(200.0, 250.0);
        let rect = egui::Rect::from_min_size(egui::pos2(30.5, 40.0), egui::vec2(410.0, 300.0));
        for (ppp, zoom, budget) in [(1.0, 1.0, 40_000), (2.0, 1.0, 12_000), (2.0, 3.0, 300), (1.5, 0.7, 7)] {
            let ctx = egui::Context::default();
            let frame = |paint: &mut dyn FnMut(&egui::Painter, &View, &egui::Visuals)| {
                let mut raw = egui::RawInput { screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(900.0, 700.0))), ..Default::default() };
                raw.viewports.insert(egui::ViewportId::ROOT, egui::ViewportInfo { native_pixels_per_point: Some(ppp), ..Default::default() });
                let out = ctx.run_ui(raw, |ui| {
                    let view = View::fit(rect, sheet, zoom, egui::vec2(12.0, -7.0));
                    paint(&ui.painter_at(rect), &view, ui.visuals());
                });
                assert_eq!(out.pixels_per_point, ppp);
                ctx.tessellate(out.shapes, out.pixels_per_point)
            };
            frame(&mut |_, _, _| {});
            let shapes = frame(&mut |painter, view, visuals| painter.extend(layout_shapes(view, &layout, &selected, visuals, budget)));
            // Drawn as shapes, then tessellated here (the picture stayed the same), then reused.
            let mut kept = None;
            let mut draw = || frame(&mut |painter, view, visuals| paint_layout(painter, view, &layout, &selected, visuals, budget, &mut kept));
            let made = draw();
            let tessellated = draw();
            let again = draw();
            let mesh = |kept: &Option<Picture>| kept.as_ref().and_then(|p| p.mesh.clone());
            let first = mesh(&kept).unwrap();
            let again2 = frame(&mut |painter, view, visuals| paint_layout(painter, view, &layout, &selected, visuals, budget, &mut kept));
            assert!(Arc::ptr_eq(&first, &mesh(&kept).unwrap()), "later frames reuse the picture");
            for got in [&made, &tessellated, &again, &again2] {
                assert_eq!(got.len(), shapes.len());
                for (a, b) in got.iter().zip(&shapes) {
                    assert_eq!(a.clip_rect, b.clip_rect);
                    let (egui::epaint::Primitive::Mesh(a), egui::epaint::Primitive::Mesh(b)) = (&a.primitive, &b.primitive) else { panic!("meshes only") };
                    assert!(!b.vertices.is_empty());
                    assert_eq!(a, b, "ppp {ppp} zoom {zoom} budget {budget}");
                }
            }
        }
        // Anything it shows changing draws it again.
        let ctx = egui::Context::default();
        let mut kept = None;
        let mut draw = |selected: &[bool], layout: &Layout, zoom: f32| {
            for _ in 0..2 {
                let _ = ctx.run_ui(egui::RawInput::default(), |ui| paint_layout(&ui.painter_at(rect), &View::fit(rect, sheet, zoom, egui::Vec2::ZERO), layout, selected, ui.visuals(), 500, &mut kept));
            }
            kept.as_ref().and_then(|p| p.mesh.clone()).unwrap()
        };
        let a = draw(&selected, &layout, 1.0);
        assert!(Arc::ptr_eq(&a, &draw(&selected, &layout, 1.0)));
        let b = draw(&vec![true; layout.islands.len()], &layout, 1.0);
        assert!(!Arc::ptr_eq(&a, &b));
        let c = draw(&vec![true; layout.islands.len()], &layout, 2.0);
        assert!(!Arc::ptr_eq(&b, &c));
        let moved = layout.with(layout.uvs.corners.clone());
        assert!(!Arc::ptr_eq(&c, &draw(&vec![true; layout.islands.len()], &moved, 2.0)));
    }

    #[test]
    fn unwrap_kept_selection_boxes_are_the_boxes() {
        let (_, layout) = sphere_layout();
        let mut kept = Vec::new();
        let n = layout.islands.len();
        for which in [vec![], vec![0], vec![1, 2], (0..n).collect::<Vec<_>>(), vec![0]] {
            for sheet in [DVec2::new(200.0, 250.0), DVec2::new(1e-9, 3.0)] {
                let want = uve::islands_bounds(&layout.uvs.corners, &layout.islands, &which, sheet);
                assert_eq!(island_bounds(&mut kept, &layout, &which, sheet), want);
                assert_eq!(island_bounds(&mut kept, &layout, &which, sheet), want, "kept");
            }
        }
        assert!(kept.len() <= 4);
        // Other corners (an edit) are other boxes.
        let mut corners = layout.uvs.corners.clone();
        uve::move_islands(&mut corners, &layout.islands, &[0], DVec2::new(5.0, 1.0), DVec2::new(200.0, 250.0));
        let moved = layout.with(corners);
        let want = uve::islands_bounds(&moved.uvs.corners, &moved.islands, &[0], DVec2::new(200.0, 250.0));
        assert_eq!(island_bounds(&mut kept, &moved, &[0], DVec2::new(200.0, 250.0)), want);
        assert_ne!(want, island_bounds(&mut kept, &layout, &[0], DVec2::new(200.0, 250.0)));
    }

    #[test]
    fn unwrap_tint_is_kept_until_it_changes() {
        // A grid of more than 200 000 triangles in one island: the tint stops after 200 001 of
        // them, as before.
        let m = primitives::plane_grid(50.0, 50.0, 330, 310);
        let corners: Vec<Corners> = (0..m.max_tid() as u32).map(|t| m.tri_positions(t).map(|p| [(p.x / 50.0 + 0.5) as f32, (p.y / 50.0 + 0.5) as f32])).collect();
        with_tool(m, |cx, id| {
            let o = cx.scene.get_mut(id).unwrap();
            o.uvs = Some(Arc::new(UvSet::new(&o.mesh, corners)));
            let mut tool = UnwrapTool::new(cx);
            wait(&mut tool, cx);
            assert_eq!(tool.shown.layout.as_ref().unwrap().islands.len(), 1);
            // As it was made every frame before.
            let reference = |tool: &UnwrapTool, cx: &ToolCx| {
                let mut ov = Overlay::default();
                let layout = tool.shown.layout.as_ref().unwrap();
                let o = cx.scene.get(tool.object.unwrap()).unwrap();
                let eye = cx.camera.eye();
                let mut n = 0;
                'all: for (i, _) in tool.editor.selected.iter().enumerate().filter(|(_, &s)| s) {
                    for &t in layout.islands.islands.get(i).into_iter().flatten() {
                        if n > 200_000 {
                            break 'all;
                        }
                        n += 1;
                        let [a, b, c] = o.mesh.tri_positions(t).map(|p| p + (eye - p) * 0.0015);
                        ov.tri(a, b, c, Vec4::new(1.0, 0.62, 0.2, 0.38));
                    }
                }
                ov.tris
            };
            let tint = |tool: &mut UnwrapTool, cx: &mut ToolCx| {
                let mut ov = Overlay::default();
                ov.tri(DVec3::ZERO, DVec3::X, DVec3::Y, Vec4::ONE);
                tool.overlay(&mut ov, cx);
                ov.tris[3..].to_vec()
            };
            let bytes = |v: &[ColorVertex]| bytemuck::cast_slice::<ColorVertex, u8>(v).to_vec();
            assert!(tint(&mut tool, cx).is_empty(), "nothing selected");
            tool.select(&[0], false);
            let first = tint(&mut tool, cx);
            assert_eq!(first.len(), 200_001 * 3);
            assert_eq!(bytes(&first), bytes(&reference(&tool, cx)));
            assert_eq!(bytes(&tint(&mut tool, cx)), bytes(&first), "kept");
            // The eye moves: made again.
            cx.camera.tumble(30.0, 10.0);
            let turned = tint(&mut tool, cx);
            assert_eq!(bytes(&turned), bytes(&reference(&tool, cx)));
            assert_ne!(bytes(&turned), bytes(&first));
            // The shape changes: made again.
            let o = cx.scene.get_mut(id).unwrap();
            let p = o.mesh.vertex(5);
            o.mesh_mut().set_vertex(5, p + DVec3::Z);
            assert_eq!(bytes(&tint(&mut tool, cx)), bytes(&reference(&tool, cx)));
            tool.select(&[], false);
            assert!(tint(&mut tool, cx).is_empty());
        });
    }

    #[test]
    fn unwrap_drag_rewrites_only_its_own_uvs() {
        with_tool(two_squares(), |cx, id| {
            let mut tool = UnwrapTool::new(cx);
            wait(&mut tool, cx);
            tool.editor.open = true;
            tool.editor.snap = false;
            let ctx = cx.egui.clone();
            let frame = |tool: &mut UnwrapTool, cx: &mut ToolCx, events: Vec<egui::Event>| {
                let raw = egui::RawInput { screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1600.0, 1200.0))), events, ..Default::default() };
                let _ = ctx.run_ui(raw, |ui| {
                    let _ = tool.panel(ui, cx);
                });
            };
            // Island 0 selected first: the window's rows for the selection are there before the
            // drag, so the canvas stays where it is.
            tool.select(&[0], false);
            frame(&mut tool, cx, vec![]);
            frame(&mut tool, cx, vec![]);
            let base = tool.shown.layout.clone().unwrap();
            let base_corners = base.uvs.corners.clone();
            let view = View::fit(tool.editor.canvas.unwrap(), tool.sheet(), tool.editor.zoom, tool.editor.pan);
            let sheet = tool.sheet();
            // The middle of the square, away from the selection's handles.
            let (lo, hi) = uve::islands_bounds(&base.uvs.corners, &base.islands, &[0], sheet).unwrap();
            let p = view.screen((lo + hi) * 0.5);
            let press = |p: egui::Pos2, pressed: bool| egui::Event::PointerButton { pos: p, button: egui::PointerButton::Primary, pressed, modifiers: egui::Modifiers::NONE };
            frame(&mut tool, cx, vec![egui::Event::PointerMoved(p), press(p, true)]);
            // What the islands should be with the pointer at `q`: the drag's start moved there.
            let expect = |q: egui::Pos2| {
                let mut c = base_corners.clone();
                uve::transform_islands(&mut c, &base.islands, &[0], DAffine2::from_translation(view.sheet_at(q) - view.sheet_at(p)), sheet);
                c
            };
            let mut held: Option<(Arc<UvSet>, Vec<Corners>, u64)> = None;
            let mut last = std::ptr::null();
            for k in 1..=6 {
                let q = p + egui::vec2(9.0 * k as f32, -4.0 * k as f32);
                frame(&mut tool, cx, vec![egui::Event::PointerMoved(q)]);
                let shown = tool.shown.layout.as_ref().unwrap().uvs.clone();
                assert_eq!(shown.corners, expect(q), "move {k}");
                assert_eq!(cx.scene.get(id).unwrap().uvs.as_ref().map(|u| u.stamp), Some(shown.stamp), "the model shows it");
                // The first move copies the drag's start; later ones rewrite their own UVs, but
                // not those something else held.
                assert_eq!(Arc::as_ptr(&shown) == last, k != 1 && k != 4, "move {k}");
                last = Arc::as_ptr(&shown);
                // Held still: nothing is made again.
                frame(&mut tool, cx, vec![]);
                frame(&mut tool, cx, vec![]);
                assert_eq!(tool.shown.layout.as_ref().unwrap().uvs.stamp, shown.stamp, "held still");
                if k == 3 {
                    // Something else holds these UVs (an autosave, say): they never change.
                    held = Some((shown.clone(), shown.corners.clone(), shown.stamp));
                }
                drop(shown);
            }
            frame(&mut tool, cx, vec![press(p + egui::vec2(54.0, -24.0), false)]);
            frame(&mut tool, cx, vec![]);
            let (h, corners, stamp) = held.unwrap();
            assert_eq!((h.corners.clone(), h.stamp), (corners, stamp));
            let after = tool.shown.layout.clone().unwrap();
            assert_eq!(after.uvs.corners, expect(p + egui::vec2(54.0, -24.0)));
            assert_eq!(cx.scene.get(id).unwrap().uvs().unwrap().corners, after.uvs.corners);
            // The drag's start was never touched, and undo brings it back.
            assert_eq!(base.uvs.corners, base_corners);
            assert_eq!(tool.undo.len(), 1);
            assert!(tool.undo(cx));
            assert_eq!(cx.scene.get(id).unwrap().uvs().unwrap().corners, base_corners);
            assert!(tool.redo(cx));
            assert_eq!(cx.scene.get(id).unwrap().uvs().unwrap().corners, after.uvs.corners);
        });
    }

    #[test]
    fn unwrap_setuv_projects_once_the_settings_settle() {
        with_tool(primitives::icosphere(10.0, 3), |cx, id| {
            let mut tool = UnwrapTool::new_projection(cx);
            assert!(matches!(tool.shown.made, Some((Made::Project(..), _))));
            let steps = tool.undo.len();
            // A field dragged: each change waits; nothing is projected meanwhile.
            for k in 1..=3 {
                tool.s.projection.rotation = 10.0 * k as f64;
                tool.project_soon();
                tool.start_projection(cx);
                assert!(tool.pending() && tool.busy().is_none() && tool.project_job.is_none());
            }
            assert_eq!(tool.state(cx.scene)["computing"], true);
            std::thread::sleep(PROJECT_AFTER + Duration::from_millis(20));
            tool.start_projection(cx);
            assert!(!tool.pending() && tool.busy().is_some_and(|b| b.0 == "Projecting"));
            wait(&mut tool, cx);
            let shown = tool.shown.layout.clone().unwrap();
            assert_eq!(cx.scene.get(id).unwrap().uvs.as_ref().map(|u| u.stamp), Some(shown.uvs.stamp));
            assert_eq!(tool.undo.len(), steps, "adjusting a projection replaces it");
            // The same as projecting at once.
            tool.project(cx);
            assert_eq!(tool.shown.layout.as_ref().unwrap().uvs.corners, shown.uvs.corners);
            assert_eq!(*tool.shown.layout.as_ref().unwrap().border, *shown.border);
            assert_eq!(*tool.shown.layout.as_ref().unwrap().islands, *shown.islands);
            // A projection of settings changed since is dropped.
            tool.s.projection.rotation = 50.0;
            tool.project_soon();
            tool.projecting = Some(Instant::now() - PROJECT_AFTER);
            tool.start_projection(cx);
            let (tag, job) = tool.project_job.take().unwrap();
            tool.s.projection.rotation = 60.0;
            let r = wait_for(&job);
            let before = tool.shown.layout.as_ref().unwrap().uvs.stamp;
            tool.projection_done(cx, &tag, r);
            assert_eq!(tool.shown.layout.as_ref().unwrap().uvs.stamp, before);
            // A value typed as "nan" is projected once, to say so, and not waited on forever.
            tool.s.projection.offset[0] = f64::NAN;
            tool.project_soon();
            tool.projecting = Some(Instant::now() - PROJECT_AFTER);
            tool.start_projection(cx);
            while tool.project_job.is_some() {
                std::thread::sleep(Duration::from_millis(2));
                tool.poll_projection(cx);
            }
            assert!(tool.error.as_deref().is_some_and(|e| e.starts_with("SetUV")), "{:?}", tool.error);
            // (The field shows it as infinity, which is projected too, once.)
            let ctx = cx.egui.clone();
            let t0 = Instant::now();
            for k in 0.. {
                let raw = egui::RawInput { screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1200.0, 900.0))), ..Default::default() };
                let _ = ctx.run_ui(raw, |ui| {
                    let _ = tool.panel(ui, cx);
                });
                if k > 3 && !tool.pending() && tool.project_job.is_none() {
                    break;
                }
                assert!(t0.elapsed() < Duration::from_secs(20), "settles");
                std::thread::sleep(Duration::from_millis(5));
            }
            assert!(tool.error.as_deref().is_some_and(|e| e.starts_with("SetUV")), "{:?}", tool.error);
        });
    }

    #[test]
    fn unwrap_setuv_waiting_is_taken_before_what_follows() {
        // Undo, an edit, an agent's settings and accepting each act as if the projection of
        // the settings shown had been taken when they changed.
        let changed = |tool: &mut UnwrapTool, cx: &mut ToolCx| {
            tool.s.projection.rotation = 33.0;
            tool.project_soon();
            tool.start_projection(cx);
            assert!(tool.pending());
        };
        let at_once = |tool: &mut UnwrapTool, cx: &mut ToolCx| {
            tool.s.projection.rotation = 33.0;
            tool.project(cx);
        };
        type Then = fn(&mut UnwrapTool, &mut ToolCx);
        let thens: [(&str, Then); 4] = [
            ("undo", |t, cx| assert!(t.undo(cx))),
            ("edit", |t, cx| {
                t.act("select_islands", &serde_json::json!({"islands": "all"}), cx).unwrap();
                t.act("rotate", &serde_json::json!({"degrees": 90.0}), cx).unwrap();
            }),
            ("settings", |t, cx| t.set_settings(&serde_json::json!({"projection": {"projection": "Box"}}), cx).unwrap()),
            ("accept", |t, cx| assert!(t.accept(cx))),
        ];
        for (what, then) in thens {
            let mut results = Vec::new();
            for wait_first in [false, true] {
                with_tool(primitives::cube(10.0, 3), |cx, id| {
                    let mut tool = UnwrapTool::new_projection(cx);
                    tool.act("select_islands", &serde_json::json!({"islands": "0"}), cx).unwrap();
                    tool.act("move", &serde_json::json!({"du": 3.0, "dv": 1.0}), cx).unwrap();
                    if wait_first {
                        changed(&mut tool, cx);
                    } else {
                        at_once(&mut tool, cx);
                    }
                    then(&mut tool, cx);
                    assert!(!tool.pending() && tool.project_job.is_none());
                    let shown = tool.shown.layout.as_ref().map(|l| l.uvs.corners.clone());
                    let object = cx.scene.get(id).unwrap().uvs().map(|u| u.corners.clone());
                    results.push((shown, object, tool.undo.len(), tool.redo.len(), history_labels(cx)));
                });
            }
            assert!(results[0] == results[1], "{what}");
        }
    }

    #[test]
    fn unwrap_reads_existing_uvs_in_the_background() {
        let (m, layout) = sphere_layout();
        let corners = layout.uvs.corners.clone();
        for method in [UvMethod::Unwrap, UvMethod::Project] {
            with_tool(m.clone(), |cx, id| {
                let o = cx.scene.get_mut(id).unwrap();
                o.uvs = Some(Arc::new(UvSet::new(&o.mesh, corners.clone())));
                let mut tool = if method == UvMethod::Unwrap { UnwrapTool::new(cx) } else { UnwrapTool::new_projection(cx) };
                if method == UvMethod::Project {
                    // Projecting stays at once, with the object's layout its undo step.
                    assert!(tool.loading.is_none() && tool.undo.len() == 1);
                    assert_eq!(tool.undo[0].layout.as_ref().unwrap().uvs.corners, corners);
                    return;
                }
                assert!(tool.shown.layout.is_none() && tool.busy().is_some_and(|b| b.0 == "Reading the UVs"));
                wait(&mut tool, cx);
                let read = tool.shown.layout.clone().unwrap();
                assert_eq!(read.uvs.corners, corners);
                assert_eq!(*read.islands, *layout.islands);
                assert_eq!(*read.border, *layout.border);
                assert_eq!(read.tris, layout.tris);
                // Accepting them unchanged adds no undo step to the scene.
                let steps = history_labels(cx);
                assert!(tool.accept(cx));
                assert_eq!(history_labels(cx), steps);
            });
        }
        // An agent's settings while they are read: the unwrap starts on top of them.
        with_tool(m, |cx, id| {
            let o = cx.scene.get_mut(id).unwrap();
            o.uvs = Some(Arc::new(UvSet::new(&o.mesh, corners.clone())));
            let mut tool = UnwrapTool::new(cx);
            tool.set_settings(&serde_json::json!({"border": 2.0}), cx).unwrap();
            assert!(tool.loading.is_none() && tool.job.is_some());
            wait(&mut tool, cx);
            assert_eq!(tool.undo.len(), 1);
            assert!(matches!(tool.undo[0].made, Some((Made::Existing, _))));
        });
    }
}
