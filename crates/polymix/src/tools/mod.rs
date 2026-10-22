//! Interactive tools. Each tool draws its own property panel, may consume viewport
//! input, may show a preview mesh, and ends with Accept / Cancel.

pub use polymix_workspace::agent::{arg, Action, Args};
use crate::camera::Camera;
use crate::renderer::ColorVertex;
use crate::scene::{History, ObjectId, Scene, TargetSurface};
use glam::{DVec2, DVec3, Vec3, Vec4};
use mm_core::bvh::RayHit;
use mm_core::{Mesh, Ray};

pub mod select;
pub mod lasso;
pub mod add_tube;
pub mod align;
pub mod clearance;
pub mod mesh_query;
pub mod surface_lasso;
pub mod units;
pub mod warp;
pub mod transform;
pub mod inspector;
pub mod ops;
pub mod plane_cut;
pub mod vessel;
pub mod measure;
pub mod sculpt;
pub mod stamp;
pub mod drop_part;
pub mod pattern;
pub mod unwrap;
pub mod hair;
pub mod pivot;
pub mod patches;
pub mod boolean;
pub mod deviation;
pub mod cfd_check;
pub mod centerline_tool;
pub mod stent;
pub mod complex;

/// Screen-space description of the 3D viewport.
#[derive(Clone, Copy, Debug)]
pub struct Viewport {
    pub rect: egui::Rect,
}

impl Viewport {
    pub fn aspect(&self) -> f64 {
        (self.rect.width() / self.rect.height().max(1.0)) as f64
    }
    pub fn ndc(&self, p: egui::Pos2) -> DVec2 {
        DVec2::new(
            ((p.x - self.rect.min.x) / self.rect.width() * 2.0 - 1.0) as f64,
            (1.0 - (p.y - self.rect.min.y) / self.rect.height() * 2.0) as f64,
        )
    }
    pub fn to_screen(self, ndc: DVec3) -> egui::Pos2 {
        egui::pos2(
            self.rect.min.x + ((ndc.x as f32 + 1.0) * 0.5) * self.rect.width(),
            self.rect.min.y + ((1.0 - ndc.y as f32) * 0.5) * self.rect.height(),
        )
    }
}

/// Overlay geometry a tool wants drawn this frame.
#[derive(Default)]
pub struct Overlay {
    pub lines: Vec<ColorVertex>,
    pub tris: Vec<ColorVertex>,
    pub brush: Option<(Vec3, f32)>,
    /// Plane whose positive side is tinted as "to be discarded", optionally limited to a sphere.
    pub cut_plane: Option<(mm_core::Plane, Option<(DVec3, f64)>)>,
    /// Object the cut tint applies to (None = every object).
    pub cut_target: Option<ObjectId>,
}

impl Overlay {
    pub fn line(&mut self, a: DVec3, b: DVec3, c: Vec4) {
        self.lines.push(ColorVertex::new(a.as_vec3(), c));
        self.lines.push(ColorVertex::new(b.as_vec3(), c));
    }
    pub fn tri(&mut self, a: DVec3, b: DVec3, c: DVec3, col: Vec4) {
        self.tris.push(ColorVertex::new(a.as_vec3(), col));
        self.tris.push(ColorVertex::new(b.as_vec3(), col));
        self.tris.push(ColorVertex::new(c.as_vec3(), col));
    }
    /// Filled square patch on a plane, with outline.
    pub fn plane_patch(&mut self, center: DVec3, normal: DVec3, half: f64, fill: Vec4, outline: Vec4) {
        let (u, v) = mm_core::geom::orthonormal_basis(normal);
        let p = [
            center + (-u - v) * half,
            center + (u - v) * half,
            center + (u + v) * half,
            center + (-u + v) * half,
        ];
        self.tri(p[0], p[1], p[2], fill);
        self.tri(p[0], p[2], p[3], fill);
        for i in 0..4 {
            self.line(p[i], p[(i + 1) % 4], outline);
        }
    }
}

/// Everything a tool may touch.
pub struct ToolCx<'a> {
    pub scene: &'a mut Scene,
    pub history: &'a mut History,
    pub camera: &'a mut Camera,
    pub status: &'a mut String,
    pub egui: &'a egui::Context,
    pub vp: Viewport,
    /// Modifier that turns the wheel into brush size (for hints).
    pub brush_scroll: crate::navigation::BrushScroll,
}

impl ToolCx<'_> {
    pub fn ray_at(&self, p: egui::Pos2) -> Ray {
        self.camera.ray_from_ndc(self.vp.ndc(p), self.vp.aspect())
    }
    pub fn project(&self, p: DVec3) -> Option<egui::Pos2> {
        let n = self.camera.project(p, self.vp.aspect())?;
        if n.z < 0.0 || n.z > 1.0 {
            return None;
        }
        Some(self.vp.to_screen(n))
    }
    /// World-space length of one screen point at the depth of `p`.
    pub fn world_per_point(&self, p: DVec3) -> f64 {
        let d = (p - self.camera.eye()).dot(self.camera.forward()).max(1e-9);
        let h = self.vp.rect.height().max(1.0) as f64;
        if self.camera.orthographic {
            self.camera.pixel_size_at_target(h)
        } else {
            2.0 * d * (self.camera.fov_y * 0.5).tan() / h
        }
    }
    /// Closest hit among visible objects (optionally restricted to one object).
    pub fn pick(&mut self, p: egui::Pos2, only: Option<ObjectId>) -> Option<(ObjectId, RayHit)> {
        let ray = self.ray_at(p);
        pick_ray(self.scene, &ray, only)
    }
    pub fn snapshot(&mut self, label: &str) {
        self.history.push(label, self.scene);
    }
}

/// World radius of a brush whose size is given in percent of the object's bounding-box
/// diagonal. As in Meshmixer, brushes keep their size on the model when the view zooms.
pub fn brush_radius(scene: &mut Scene, id: ObjectId, size_pct: f32) -> f64 {
    scene.get_mut(id).map(|o| o.bvh().bounds().diagonal() * size_pct as f64 / 100.0).unwrap_or(0.0)
}

/// A brush's size in percent of the object's size. The workspace's is shared by every brush
/// tool (Select, Sculpt): they hold it rather than a size of their own, so one size carries
/// across tools and from one use of a tool to the next.
#[derive(Clone, Debug)]
pub struct BrushSize(std::sync::Arc<std::sync::atomic::AtomicU32>);

impl BrushSize {
    pub fn new(pct: f32) -> Self {
        let b = Self(Default::default());
        b.set(pct);
        b
    }

    pub fn get(&self) -> f32 {
        f32::from_bits(self.0.load(std::sync::atomic::Ordering::Relaxed))
    }

    /// Set it, within [`BRUSH_SIZE_RANGE`].
    pub fn set(&self, pct: f32) {
        let pct = if pct.is_finite() { pct.clamp(BRUSH_SIZE_RANGE.0, BRUSH_SIZE_RANGE.1) } else { self.get() };
        self.0.store(pct.to_bits(), std::sync::atomic::Ordering::Relaxed);
    }

    /// Whether `self` and `other` are the same size (not just equal).
    pub fn same(&self, other: &Self) -> bool {
        std::sync::Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Default for BrushSize {
    fn default() -> Self {
        Self::new(2.5)
    }
}

/// Brush size slider shared by the selection and sculpt brushes (percent of model size).
pub fn brush_size_ui(ui: &mut egui::Ui, size: &BrushSize, radius: Option<f64>) {
    ui.horizontal(|ui| {
        ui.label("Size");
        let mut pct = size.get();
        if ui.add(egui::Slider::new(&mut pct, BRUSH_SIZE_RANGE.0..=BRUSH_SIZE_RANGE.1).logarithmic(true).suffix("%"))
            .on_hover_text("Brush radius as a percentage of the model size, the same in every brush tool. [ ] change it.")
            .changed()
        {
            size.set(pct);
        }
    });
    if let Some(r) = radius {
        ui.small(format!("Radius {r:.3} (model units)"));
    }
}

pub const BRUSH_SIZE_RANGE: (f32, f32) = (0.1, 25.0);

/// Step the brush size for the [ and ] keys.
pub fn step_brush_size(size: &BrushSize, grow: bool) {
    let f = if grow { 1.15 } else { 1.0 / 1.15 };
    size.set(size.get() * f);
}

pub fn pick_ray(scene: &mut Scene, ray: &Ray, only: Option<ObjectId>) -> Option<(ObjectId, RayHit)> {
    let mut best: Option<(ObjectId, RayHit)> = None;
    for o in scene.objects.iter_mut() {
        // References are guides, not picked (unless asked for by id).
        if !o.visible || only.is_some_and(|id| id != o.id) || (o.reference && only.is_none()) {
            continue;
        }
        let bvh = o.bvh();
        if let Some(h) = bvh.raycast(&o.mesh, ray) {
            if best.as_ref().map(|(_, b)| h.t < b.t).unwrap_or(true) {
                best = Some((o.id, h));
            }
        }
    }
    best
}

/// Pointer state for the viewport this frame, filtered to what tools care about
/// (navigation gestures are removed before tools see the input).
#[derive(Clone, Debug, Default)]
pub struct PointerInput {
    pub pos: Option<egui::Pos2>,
    pub pressed: bool,
    pub down: bool,
    pub released: bool,
    pub clicked: bool,
    pub double_clicked: bool,
    pub shift: bool,
    pub command: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DragPhase {
    Start,
    Move,
    End,
}

pub enum ToolOutcome {
    Continue,
    Close,
}

pub trait Tool {
    fn name(&self) -> &'static str;
    /// Draw the property panel; return `Close` when the tool finished (accepted or cancelled).
    fn panel(&mut self, ui: &mut egui::Ui, cx: &mut ToolCx) -> ToolOutcome;
    /// Handle left-button interaction inside the viewport.
    fn pointer(&mut self, _input: &PointerInput, _cx: &mut ToolCx) {}
    /// Add 3D overlay geometry.
    fn overlay(&mut self, _ov: &mut Overlay, _cx: &mut ToolCx) {}
    /// Paint 2D overlays on top of the viewport.
    fn paint(&mut self, _painter: &egui::Painter, _cx: &mut ToolCx) {}
    /// Mesh to show instead of an object's own mesh while the tool is active.
    fn preview(&self) -> Option<(ObjectId, &Mesh)> {
        None
    }
    /// The vertices whose positions the preview changed since [`Tool::clear_preview_moves`]
    /// was last called, with the preview's shape version then (`from`) and now (`to`), so the
    /// renderer can upload just those when it holds the preview as it was at `from`. None
    /// when the tool does not keep the list, or when the preview was replaced or its topology
    /// changed since: then only a full upload is right.
    #[allow(dead_code)] // For the renderer's partial preview upload, which does not use it yet.
    fn preview_moves(&self) -> Option<(u64, u64, &[u32])> {
        None
    }
    /// The renderer has the preview as it is now: start a new [`Tool::preview_moves`] list.
    #[allow(dead_code)] // For the renderer's partial preview upload, which does not use it yet.
    fn clear_preview_moves(&mut self) {}
    /// When the preview is the active object's own mesh moved by a similarity transform
    /// (rotation, uniform positive scale and translation, so its shading is the same), that
    /// transform: the renderer can then draw the object's own GPU mesh with it (in the
    /// preview's style) instead of uploading the preview.
    #[allow(dead_code)] // For the renderer's preview drawing, which does not use it yet.
    fn preview_transform(&self) -> Option<glam::DMat4> {
        None
    }
    /// Keyboard shortcuts while the tool is active. Return true if a key was used (the app
    /// then skips its own shortcuts for it).
    fn keys(&mut self, _input: &egui::InputState, _cx: &mut ToolCx) -> bool {
        false
    }
    /// Mouse wheel over the viewport, when the navigation mode routes it to tools (brush
    /// tools resize their brush). Return true if used; otherwise the view zooms.
    fn wheel(&mut self, _delta: f32, _cx: &mut ToolCx) -> bool {
        false
    }
    /// Whether a right-button drag starting now belongs to the tool instead of navigation
    /// (Meshmixer: right-drag expands the selection / sets brush strength).
    fn claims_secondary_drag(&self, _scene: &Scene) -> bool {
        false
    }
    /// A right-button drag owned by the tool; `total` is the pointer offset since the press.
    fn secondary_drag(&mut self, _phase: DragPhase, _pos: egui::Pos2, _total: egui::Vec2, _cx: &mut ToolCx) {}
    /// A right-click (no drag) in the viewport.
    fn secondary_click(&mut self, _pos: egui::Pos2, _cx: &mut ToolCx) {}
    /// Text for Edit → Copy (⌘C): the tool's readout, e.g. its measurements.
    fn copy_text(&self) -> Option<String> {
        None
    }
    /// The brush size, for tools with a brush (the hotbox and ⌘ + middle drag size it).
    fn brush_size(&self) -> Option<&BrushSize> {
        None
    }
    /// Take the workspace's brush size in place of the tool's own (see [`BrushSize`]).
    fn share_brush(&mut self, _size: &BrushSize) {}
    /// Brush strength in 0..1 (for the hotbox).
    fn brush_strength(&mut self) -> Option<&mut f64> {
        None
    }
    /// Esc pressed. Return true to close the tool.
    fn cancel(&mut self, _cx: &mut ToolCx) -> bool {
        true
    }
    /// Enter pressed. Return true to close the tool (after committing its result).
    fn accept(&mut self, _cx: &mut ToolCx) -> bool {
        false
    }
    /// Called whenever the tool is removed (closed, replaced, or invalidated).
    fn on_close(&mut self, _cx: &mut ToolCx) {}
    /// Whether the tool stays valid when the scene changes underneath it (undo/redo,
    /// object deletion…). Tools holding previews computed from a mesh return false and are
    /// closed instead of committing stale results.
    fn survives_scene_change(&self) -> bool {
        false
    }
    /// Scripted command (automated tests), e.g. `auto` for the Inspector's Auto Repair All.
    fn script(&mut self, _cmd: &str, _cx: &mut ToolCx) {}
    /// A sub-tool this tool wants opened on top of itself.
    fn take_request(&mut self) -> Option<Box<dyn Tool>> {
        None
    }
    /// Show face-group colours while this tool is open (its result is about groups).
    fn shows_groups(&self) -> bool {
        false
    }
    /// Show the UV checker while this tool is open.
    fn shows_uvs(&self) -> bool {
        false
    }
    /// Whether the tool is computing something (shows a spinner, blocks other tools).
    fn busy(&self) -> Option<(String, f32)> {
        None
    }
    /// Work waiting for the panel to be drawn (a preview to recompute after a change, an
    /// accept to finish): not [`Tool::busy`] yet, but not done either.
    fn pending(&self) -> bool {
        false
    }
    /// The settings agents can read and change, as a JSON object (none for most interactive
    /// tools: agents work their panels instead).
    fn settings(&self) -> Option<serde_json::Value> {
        None
    }
    /// JSON Schema of [`Tool::settings`] (with each setting's description).
    fn settings_schema(&self) -> Option<serde_json::Value> {
        None
    }
    /// Change settings for an agent: the keys given replace those values.
    fn set_settings(&mut self, _settings: &serde_json::Value, _cx: &mut ToolCx) -> Result<(), String> {
        Err(format!("{} has no settings agents can set directly; work its panel with app_ui / app_ui_action", self.name()))
    }
    /// What the tool holds and shows besides its settings, for agents: picked points,
    /// readouts, results, warnings, errors.
    fn state(&self, _scene: &Scene) -> serde_json::Value {
        serde_json::Value::Null
    }
    /// What agents can ask the tool to do besides changing settings: its panel's buttons and
    /// its clicks, as named operations with typed arguments.
    fn actions(&self) -> Vec<Action> {
        Vec::new()
    }
    /// Do one of [`Tool::actions`] (arguments already checked). `Ok` says what was done.
    fn act(&mut self, action: &str, _args: &serde_json::Value, _cx: &mut ToolCx) -> Result<String, String> {
        Err(format!("{} has no action {action}", self.name()))
    }
}

/// Vertices a tool's preview moved since the renderer last took it (see
/// [`Tool::preview_moves`]).
#[derive(Default)]
#[allow(dead_code)] // Read through Tool::preview_moves, which the renderer does not use yet.
pub struct PreviewMoves {
    /// The preview's shape version when the list was started, and after its latest change.
    from: u64,
    to: u64,
    verts: Vec<u32>,
    /// The list covers every change since `from`.
    valid: bool,
}

#[allow(dead_code)] // Read through Tool::preview_moves, which the renderer does not use yet.
impl PreviewMoves {
    /// The preview was replaced or its topology changed: only a full upload is right until
    /// the next [`PreviewMoves::clear`].
    pub fn reset(&mut self) {
        self.verts.clear();
        self.valid = false;
    }

    /// `verts` were rewritten in place, taking the preview from shape version `before` to
    /// `after`. The list is dropped (a full upload is then needed) once it grows past
    /// `limit` entries, or if `before` is not where the list ended.
    pub fn record(&mut self, before: u64, after: u64, verts: impl IntoIterator<Item = u32>, limit: usize) {
        if !self.valid || self.to != before {
            self.reset();
            return;
        }
        self.verts.extend(verts);
        self.to = after;
        if self.verts.len() > limit {
            self.reset();
        }
    }

    /// (from, to, vertices), if the list is complete.
    pub fn get(&self) -> Option<(u64, u64, &[u32])> {
        self.valid.then_some((self.from, self.to, &self.verts[..]))
    }

    /// The renderer has the preview at shape version `now` (None: there is no preview).
    pub fn clear(&mut self, now: Option<u64>) {
        self.verts.clear();
        match now {
            Some(v) => {
                (self.from, self.to, self.valid) = (v, v, true);
            }
            None => self.valid = false,
        }
    }
}

/// Overwrite the live vertex positions of `dst`, a copy of `src` (same vertices), with `f` of
/// the source's, as [`Mesh::transform`] does on a fresh copy, in parallel for big meshes.
pub(crate) fn transform_from(dst: &mut Mesh, src: &Mesh, f: impl Fn(DVec3) -> DVec3 + Sync) {
    use rayon::prelude::*;
    let from = src.positions();
    let pos = dst.positions_mut();
    let write = |(i, p): (usize, &mut DVec3)| {
        if src.is_vertex(i as u32) {
            *p = f(from[i]);
        }
    };
    if pos.len() >= 16 * 1024 {
        pos.par_iter_mut().enumerate().with_min_len(4096).for_each(write);
    } else {
        pos.iter_mut().enumerate().for_each(write);
    }
    dst.bump_shape_version();
}

/// `changes` merged into `settings` (both JSON objects; nested objects merge too). Every key
/// must already exist, so a misspelt setting is an error rather than ignored.
pub fn merge_settings(settings: &mut serde_json::Value, changes: &serde_json::Value) -> Result<(), String> {
    let (Some(cur), Some(new)) = (settings.as_object_mut(), changes.as_object()) else {
        return Err("settings must be a JSON object".into());
    };
    let known = cur.keys().cloned().collect::<Vec<_>>().join(", ");
    for (k, v) in new {
        match cur.get_mut(k) {
            None => return Err(format!("no setting \"{k}\" (the settings are {known})")),
            Some(old) if old.is_object() && v.is_object() => merge_settings(old, v).map_err(|e| format!("{k}: {e}"))?,
            Some(old) => *old = v.clone(),
        }
    }
    Ok(())
}

/// `params` with `changes` merged in (see [`merge_settings`]), for tools whose settings are a
/// serde type.
pub fn merged<T: serde::Serialize + serde::de::DeserializeOwned>(params: &T, changes: &serde_json::Value) -> Result<T, String> {
    let mut v = serde_json::to_value(params).map_err(|e| e.to_string())?;
    merge_settings(&mut v, changes)?;
    serde_json::from_value(v).map_err(|e| e.to_string())
}

/// A type's JSON Schema for agents.
pub fn schema_of<T: schemars::JsonSchema>() -> serde_json::Value {
    let mut v = serde_json::to_value(schemars::schema_for!(T)).unwrap_or_default();
    if let Some(o) = v.as_object_mut() {
        o.remove("$schema");
    }
    v
}
