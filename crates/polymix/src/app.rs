//! Application shell: layout, viewport navigation, tool dispatch, file IO.

mod agent;
mod command;
pub(crate) mod slices;

pub use command::{MeshCommand, Shape, STANDARD_VIEWS};

use crate::camera::{Camera, UpAxis, ViewHistory};
use crate::commands::{self, CmdCx};
use crate::jobs::Job;
use crate::navigation::{BrushScroll, NavAction, NavBindings, NavMode};
use crate::renderer::{self, ColorVertex, GpuMesh, ObjectStyle, SceneRenderer};
use crate::spacemouse::{self, SpaceMouse};
use crate::scene::{History, ObjectId, Scene};
use crate::tools::{self, DragPhase, Overlay, PointerInput, Tool, ToolCx, ToolOutcome, Viewport};
use crate::ui_util;
use eframe::egui_wgpu;
use egui::{Color32, Key, PointerButton, RichText};
use glam::{DVec3, Vec4};
use mm_core::hash::{FastMap, FastSet};
use mm_core::Mesh;
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Category {
    Meshmix,
    Edit,
    Vessel,
    Analysis,
    Sculpt,
    Shaders,
}

/// How shading normals are made (Meshmixer's Mesh Normal Mode).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormalMode {
    /// Smooth across the surface (crisp only at sharp edges, if enabled).
    Vertex,
    /// Flat triangles.
    Face,
    /// Smooth within each face group, crisp along group borders.
    Group,
}

impl NormalMode {
    pub const ALL: [NormalMode; 3] = [NormalMode::Vertex, NormalMode::Face, NormalMode::Group];
    pub fn label(self) -> &'static str {
        match self {
            NormalMode::Vertex => "Vertex normals",
            NormalMode::Face => "Face normals",
            NormalMode::Group => "Group normals",
        }
    }
    fn key(self) -> &'static str {
        match self {
            NormalMode::Vertex => "vertex",
            NormalMode::Face => "face",
            NormalMode::Group => "group",
        }
    }
    fn from_key(k: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|m| m.key() == k)
    }
}

/// What colours the surface (Meshmixer's Mesh Color Mode).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorMode {
    /// The object colour.
    Constant,
    /// A colour per face group.
    Group,
    /// The mesh's vertex colours (the object colour when it has none).
    Vertex,
    /// A checker texture through the UVs (Unwrap); the object colour without UVs.
    UvChecker,
    /// The object's texture image through its UVs, lit like the surface.
    Texture,
    /// The texture's colours as they are (no lighting).
    TextureUnlit,
}

impl ColorMode {
    pub const ALL: [ColorMode; 6] = [ColorMode::Vertex, ColorMode::Group, ColorMode::Constant, ColorMode::UvChecker, ColorMode::Texture, ColorMode::TextureUnlit];
    pub fn label(self) -> &'static str {
        match self {
            ColorMode::Constant => "Constant color",
            ColorMode::Group => "Group colors",
            ColorMode::Vertex => "Vertex colors",
            ColorMode::UvChecker => "UV checker",
            ColorMode::Texture => "Texture",
            ColorMode::TextureUnlit => "Texture (unlit)",
        }
    }
    fn key(self) -> &'static str {
        match self {
            ColorMode::Constant => "constant",
            ColorMode::Group => "group",
            ColorMode::Vertex => "vertex",
            ColorMode::UvChecker => "uv",
            ColorMode::Texture => "texture",
            ColorMode::TextureUnlit => "texture_unlit",
        }
    }
    fn from_key(k: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|m| m.key() == k)
    }
}

/// File and behaviour preferences (Preferences window).
pub struct Prefs {
    /// Frame the scene after every import (Meshmixer: Reset Camera on Scene Change).
    pub reset_camera_on_import: bool,
    /// ⌘E exports every visible object as one mesh instead of the active object.
    pub export_whole_scene: bool,
    /// OBJ export writes face groups as `g` groups.
    pub obj_export_groups: bool,
    /// OBJ import turns `g` / `o` groups into face groups.
    pub obj_import_groups: bool,
    /// Files are Z-up: turn them Y-up on import and back on export (Meshmixer's "Flip Z-Y
    /// axis on Import-Export"), so a round trip keeps the file's coordinates.
    pub flip_zy: bool,
    /// Importing into a non-empty scene.
    pub import_mode: ImportMode,
    /// Models of implausible size (under 1 or over 10 000 units across).
    pub scale_mode: ScaleMode,
    /// Minutes between autosaves of a changed scene (0 = off).
    pub autosave_minutes: u32,
}

/// What an export writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExportScope {
    Active,
    AllVisible,
    /// The active object's selected triangles.
    Selection,
    /// One file per face group of the active object (Meshmixer's "sequence, one file per
    /// region"), named after the group.
    PerGroup,
    /// One file per visible object, named after it.
    PerObject,
}

pub(crate) use polymix_workspace::data_dir;

/// Autosave state: the files, what was saved last, and the job writing it.
struct Autosave {
    file: PathBuf,
    marker: PathBuf,
    saved_generation: u64,
    last_time: f64,
    job: Option<Job<()>>,
}

/// The Statistics panel's whole-mesh values of an object, so frames that do not change its
/// shape (hovering, orbiting) do not walk the mesh again.
#[derive(Default)]
struct StatsCache {
    /// Open-edge count per (object, topology version): it lasts through brush strokes.
    open: Option<(ObjectId, u64, usize)>,
    /// The rest per (object, shape version).
    shape: Option<(ObjectId, u64, ShapeStats)>,
}

#[derive(Clone, Copy)]
struct ShapeStats {
    extents: DVec3,
    open: usize,
    /// For a closed mesh (no open edges) only.
    volume: Option<f64>,
    center_of_mass: DVec3,
    area: f64,
}

impl StatsCache {
    /// The values for `o`'s mesh as it is now: the same functions as without the cache, but
    /// the bounds and the open-edge count are computed once.
    fn get(&mut self, o: &crate::scene::SceneObject) -> ShapeStats {
        let m = &*o.mesh;
        if let Some((id, version, s)) = self.shape {
            if (id, version) == (o.id, m.shape_version()) {
                return s;
            }
        }
        let open = match self.open {
            Some((id, topo, n)) if (id, topo) == (o.id, m.topology_version()) => n,
            _ => {
                let n = m.boundary_edge_ids().count();
                self.open = Some((o.id, m.topology_version(), n));
                n
            }
        };
        let b = m.bounds();
        let s = ShapeStats {
            extents: b.extents(),
            open,
            volume: (open == 0).then(|| mm_core::measure::volume(m)),
            center_of_mass: mm_core::measure::center_of_mass_with(m, &b, open == 0),
            area: mm_core::measure::surface_area(m),
        };
        self.shape = Some((o.id, m.shape_version(), s));
        s
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImportMode {
    Ask,
    Append,
    Replace,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ScaleMode {
    Ask,
    Never,
    /// Always scale implausibly sized models by this factor.
    Always(f64),
}

/// Imported meshes waiting for an answer to the import prompts.
/// What an imported file brings besides its mesh: texture coordinates (by triangle id) and a
/// texture image.
#[derive(Default)]
struct ImportLook {
    uvs: Option<Vec<[[f32; 2]; 3]>>,
    texture: Option<std::sync::Arc<crate::scene::TextureImage>>,
    /// Imported as a reference (File → Import reference…).
    reference: bool,
}

struct PendingImport {
    /// Name, mesh, status note and bounding-box diagonal (measured where the mesh was made,
    /// off the UI thread for files, so the prompts drawn every frame need not walk it).
    meshes: Vec<(String, Mesh, String, f64)>,
    /// One per mesh.
    looks: Vec<ImportLook>,
    /// Append / Replace decided (or not needed).
    placement: Option<bool /*replace*/>,
    remember: bool,
}

/// Read one file for an import: the mesh, its status note, its bounding-box diagonal and
/// what else the file brings (UVs, texture).
fn import_file(f: &std::path::Path, obj_groups: bool, reference: bool) -> anyhow::Result<(String, Mesh, String, f64, ImportLook)> {
    let t0 = std::time::Instant::now();
    let mut imported = mm_io::read_mesh(f).map_err(|e| anyhow::anyhow!("{}: {e:#}", f.display()))?;
    if !obj_groups && imported.format == mm_io::Format::Obj {
        for t in imported.mesh.triangle_ids().collect::<Vec<_>>() {
            imported.mesh.set_tri_group(t, 0);
        }
    }
    let rep = &imported.report;
    let mut notes = format!("{}: {} triangles, {} vertices ({:.2}s)", imported.name, imported.mesh.triangle_count(), imported.mesh.vertex_count(), t0.elapsed().as_secs_f32());
    if rep.nonmanifold_split + rep.degenerate + rep.duplicate > 0 {
        notes += &format!(" — fixed on load: {} non-manifold, {} degenerate, {} duplicate triangles", rep.nonmanifold_split, rep.degenerate, rep.duplicate);
    }
    let texture = match &imported.texture {
        Some(p) => match crate::scene::TextureImage::load(p) {
            Ok(t) => {
                notes += &format!(" — texture {}", p.file_name().map_or(String::new(), |n| n.to_string_lossy().to_string()));
                Some(std::sync::Arc::new(t))
            }
            Err(e) => {
                notes += &format!(" — texture not loaded ({e})");
                None
            }
        },
        None => None,
    };
    if reference {
        notes += " — as a reference";
    }
    let diagonal = imported.mesh.bounds().diagonal();
    Ok((imported.name, imported.mesh, notes, diagonal, ImportLook { uvs: imported.uvs, texture, reference }))
}

/// Read the files of an import in an import job, up to `workers` at once: `progress` follows
/// the files done and cancels. The results come in file order; the first error in file order
/// ends the import, as when the files were read one after another.
fn import_files(files: &[PathBuf], obj_groups: bool, reference: bool, workers: usize, progress: crate::jobs::ProgressFn) -> anyhow::Result<ImportResult> {
    side_by_side(files.len(), workers, progress, |i| import_file(&files[i], obj_groups, reference))
}

/// A panic's message, as a job reports it.
fn panic_message(p: &(dyn std::any::Any + Send)) -> String {
    p.downcast_ref::<String>().cloned().or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_else(|| "operation panicked".into())
}

/// `read(i)` for each `i` in `0..n`, up to `workers` at once, in a job: `progress` follows the
/// reads done and cancels. The results come in order; the first error in that order ends it,
/// as when read one after another (a panic is an error, "internal error: …" as in a job).
fn side_by_side<T: Send>(n: usize, workers: usize, progress: crate::jobs::ProgressFn, read: impl Fn(usize) -> anyhow::Result<T> + Sync) -> anyhow::Result<Vec<T>> {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    let workers = workers.clamp(1, n.max(1));
    // One pool for all the reads, with a job's threads (one per core but one): their own
    // parallel loops share it, rather than each read starting a pool of that size.
    let pool = (workers > 1).then(|| {
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get().saturating_sub(1).max(1));
        rayon::ThreadPoolBuilder::new().num_threads(threads).thread_name(|i| format!("polymix-job-read-{i}")).build()
    });
    let pool = match pool {
        Some(Ok(pool)) => pool,
        Some(Err(e)) => {
            log::warn!("cannot start the threads to read files side by side: {e}");
            return side_by_side(n, 1, progress, read);
        }
        None => {
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                if !progress(i as f32 / n as f32) {
                    anyhow::bail!("cancelled");
                }
                out.push(read(i)?);
            }
            return Ok(out);
        }
    };
    let (next, stop) = (AtomicUsize::new(0), AtomicBool::new(false));
    let (tx, rx) = std::sync::mpsc::channel();
    let mut slots: Vec<Option<anyhow::Result<T>>> = (0..n).map(|_| None).collect();
    let cancelled = std::thread::scope(|s| {
        for _ in 0..workers {
            let tx = tx.clone();
            let (next, stop, pool, read) = (&next, &stop, &pool, &read);
            s.spawn(move || {
                // Reads are taken in order, so all those before a failed one are done.
                while !stop.load(Ordering::Relaxed) {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= n {
                        break;
                    }
                    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| pool.install(|| read(i)))).unwrap_or_else(|p| Err(anyhow::anyhow!("internal error: {}", panic_message(&*p))));
                    if r.is_err() {
                        stop.store(true, Ordering::Relaxed);
                    }
                    if tx.send((i, r)).is_err() {
                        break;
                    }
                }
            });
        }
        drop(tx);
        let mut done = 0;
        loop {
            if !progress(done as f32 / n as f32) {
                stop.store(true, Ordering::Relaxed);
                return true;
            }
            match rx.recv_timeout(std::time::Duration::from_millis(50)) {
                Ok((i, r)) => {
                    slots[i] = Some(r);
                    done += 1;
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => return false,
            }
        }
    });
    if cancelled {
        anyhow::bail!("cancelled");
    }
    // In order: the reads done, up to the first that failed.
    let mut out = Vec::with_capacity(n);
    for r in slots.into_iter().map_while(|s| s) {
        out.push(r?);
    }
    Ok(out)
}

/// What an export writes, taken from the scene without copying a mesh: the export job
/// assembles it.
enum ExportSource {
    /// An object's mesh.
    Whole(std::sync::Arc<Mesh>),
    /// The meshes of several objects, as one.
    Combined(Vec<std::sync::Arc<Mesh>>),
    /// Some triangles of a mesh.
    Part(std::sync::Arc<Mesh>, mm_core::TriSelection),
}

/// The files of an export of one file per region or per object, for its job.
enum ExportParts {
    /// The face groups of a mesh, with the group names.
    Groups(std::sync::Arc<Mesh>, std::collections::HashMap<i32, String>),
    /// Objects' names and meshes.
    Objects(Vec<(String, std::sync::Arc<Mesh>)>),
}

/// The number of face groups `m`'s triangles belong to.
fn distinct_groups(m: &Mesh) -> usize {
    let mut seen: FastSet<i32> = FastSet::default();
    let mut last = None;
    for t in m.triangle_ids() {
        // Neighbouring triangles mostly share their group.
        let g = m.tri_group(t);
        if last != Some(g) {
            seen.insert(g);
            last = Some(g);
        }
    }
    seen.len()
}

impl ExportSource {
    /// The mesh to write, turned back Z-up when `flip`; a whole mesh that needs no change is
    /// written as it is.
    fn mesh(self, flip: bool) -> std::sync::Arc<Mesh> {
        let mut m = match self {
            ExportSource::Whole(m) if !flip => return m,
            ExportSource::Whole(m) => std::sync::Arc::unwrap_or_clone(m),
            ExportSource::Combined(meshes) => {
                let mut m = Mesh::new();
                for o in &meshes {
                    m.append_mesh(o, 0);
                }
                m
            }
            ExportSource::Part(m, selection) => {
                let tris: Vec<u32> = selection.iter().collect();
                m.submesh(&tris).0
            }
        };
        if flip {
            m.transform(flip_zy_out);
        }
        std::sync::Arc::new(m)
    }
}

/// An object as a project file stores it, sharing its mesh and UVs (UVs only if they still
/// belong to the mesh's triangles), with `group_names` in the order they are written.
fn save_object(o: &crate::scene::SceneObject, group_names: Vec<(i32, String)>) -> crate::project::SaveObject {
    crate::project::SaveObject {
        name: o.name.clone(),
        mesh: o.mesh.clone(),
        visible: o.visible,
        color: o.color,
        material: o.material,
        frame: o.frame,
        group_names,
        uvs: o.uvs.clone().filter(|u| u.topo == o.mesh.topology_version()),
        pivots: o.pivots.clone(),
        texture: o.texture.as_ref().and_then(|t| t.path.clone()),
        reference: o.reference,
    }
}

/// `r.on_hover_text(text())`, with the text made only when the tooltip shows or AccessKit
/// reads it (for rows drawn every frame, hundreds of them in a long list).
pub(crate) fn hover_text_with(r: egui::Response, text: impl FnOnce() -> String) -> egui::Response {
    if r.ctx.has_accesskit_node(r.id) {
        // The text also describes the widget to agents and screen readers.
        return r.on_hover_text(text());
    }
    r.on_hover_ui(|ui| {
        // As `on_hover_text` lays it out.
        ui.set_max_width(ui.spacing().tooltip_width);
        ui.add(egui::Label::new(text()));
    })
}

/// Size outside the plausible range for millimetre models (bounding-box diagonal).
fn odd_size(m: &Mesh) -> bool {
    odd_diagonal(m.bounds().diagonal())
}

/// [`odd_size`] of a mesh whose bounding-box diagonal is `d`.
fn odd_diagonal(d: f64) -> bool {
    d.is_finite() && (d < 1.0 || d > 10_000.0)
}

/// File (Z-up) → scene (Y-up): a rotation about X, so handedness and normals are kept.
pub fn flip_zy_in(p: glam::DVec3) -> glam::DVec3 {
    glam::DVec3::new(p.x, p.z, -p.y)
}

/// Scene (Y-up) → file (Z-up), the inverse of [`flip_zy_in`].
pub fn flip_zy_out(p: glam::DVec3) -> glam::DVec3 {
    glam::DVec3::new(p.x, -p.z, p.y)
}

impl Default for Prefs {
    fn default() -> Self {
        Self { reset_camera_on_import: true, export_whole_scene: false, obj_export_groups: true, obj_import_groups: true, flip_zy: true, import_mode: ImportMode::Ask, scale_mode: ScaleMode::Ask, autosave_minutes: 5 }
    }
}

pub struct ViewOptions {
    pub wireframe: bool,
    pub normals: NormalMode,
    pub colors: ColorMode,
    pub boundaries: bool,
    pub show_stats: bool,
    /// Keep edges sharper than `crease_deg` crisp in smooth shading (CAD parts).
    pub sharp_edges: bool,
    pub crease_deg: f32,
    /// Mouse navigation scheme (Meshmixer's Preferences → Navigation Mode).
    pub nav_mode: NavMode,
    /// The user's own drag table, used in the Custom navigation mode.
    pub nav_custom: NavBindings,
    /// Modifier that turns the scroll wheel into brush size.
    pub brush_scroll: BrushScroll,
    /// Scroll-wheel zoom speed multiplier.
    pub zoom_speed: f32,
    /// Wheel zoom heads for the part of the model under the cursor (otherwise the part in
    /// the centre of the view).
    pub zoom_to_cursor: bool,
    /// Wheel zoom moves toward the point under the cursor by a fraction of the distance to
    /// it (slowing down near the surface, never passing it); otherwise toward the view
    /// target's depth.
    pub zoom_near: bool,
    pub show_grid: bool,
    /// Grid spacing in mm (0: chosen from the scene's size).
    pub grid_mm: f64,
    /// Draw the target surface see-through.
    pub transparent_target: bool,
    /// Every object see-through, edges seen edge-on most opaque.
    pub xray: bool,
    /// Background colour at the top of the view.
    pub background: [f32; 3],
    pub show_view_cube: bool,
    pub theme: ui_util::Theme,
    /// Draw objects' pivots (Meshmixer's Hide/Show Pivots).
    pub show_pivots: bool,
}

impl Default for ViewOptions {
    fn default() -> Self {
        Self { wireframe: false, normals: NormalMode::Group, colors: ColorMode::Vertex, boundaries: true, show_stats: true, sharp_edges: true, crease_deg: DEFAULT_CREASE_DEG, nav_mode: NavMode::Fusion360, nav_custom: NavBindings::FUSION360, brush_scroll: BrushScroll::Shift, zoom_speed: 1.0, zoom_to_cursor: true, zoom_near: true, show_grid: false, grid_mm: 0.0, transparent_target: true, xray: false, background: renderer::DEFAULT_BACKGROUND, show_view_cube: true, theme: ui_util::Theme::Dark, show_pivots: true }
    }
}

const DEFAULT_CREASE_DEG: f32 = 40.0;
/// How often a view showing work in progress (a tool computing, files loading) is redrawn:
/// about 15 times a second, like the throttled spinner.
const BUSY_REPAINT: std::time::Duration = std::time::Duration::from_millis(66);
/// The object browser's width range (drag its left edge).
const OBJECTS_WIDTH: (f32, f32) = (170.0, 520.0);
/// The Preferences window's 3D mouse tab (also `prefs:4` in scripts).
const PREFS_TAB_3D_MOUSE: usize = 4;
/// The circle in the middle of the view the 3D mouse's centre of rotation is looked for in:
/// its diameter as a share of the view's smaller side (half the field of view).
const CENTRE_CIRCLE: f64 = 0.5;

/// Destination of a tool switch.
#[derive(Clone, Debug)]
enum SwitchTo {
    /// Open a tool by its script name (see [`make_tool`]).
    Tool(String),
    /// Show a category palette (or nothing).
    Category(Option<Category>),
    /// The S hotkey: Select with nothing selected, whatever was selected before (e.g. the
    /// patch Erase & Fill just made).
    ClearSelect,
}

/// The 3D mouse's centre of rotation at `at`: a small translucent green ball, `a` its opacity
/// (0..1, fading out after a move).
fn paint_centre_marker(painter: &egui::Painter, at: egui::Pos2, a: f32) {
    let green = |alpha: f32| Color32::from_rgba_unmultiplied(80, 210, 110, (alpha * a * 255.0) as u8);
    painter.circle_filled(at, 4.5, green(0.32));
    painter.circle_stroke(at, 4.5, egui::Stroke::new(1.0, green(0.7)));
    // A highlight up and to the left makes it read as a ball.
    painter.circle_filled(at + egui::vec2(-1.5, -1.5), 1.5, Color32::from_rgba_unmultiplied(255, 255, 255, (0.3 * a * 255.0) as u8));
    painter.circle_filled(at, 1.0, green(0.9));
}

/// What a 3D mouse button's action is called in the settings.
fn button_action_label(a: &spacemouse::ButtonAction) -> String {
    use spacemouse::ButtonAction as A;
    match a {
        A::Hotbox => "Hotbox".into(),
        A::Fit => "Fit the view".into(),
        A::View(k) => STANDARD_VIEWS.get(*k).map_or("A view".into(), |v| format!("{} view", v.0)),
        A::Undo => "Undo".into(),
        A::Redo => "Redo".into(),
        A::Accept => "Accept the open tool".into(),
        A::Cancel => "Cancel the open tool".into(),
        A::Tool(name) => command::TOOLS.iter().find(|t| t.name == name).map_or(format!("Tool {name}"), |t| t.title.to_string()),
        A::Hold(k) => format!("Hold {}", k.label()),
        A::Key { key, modifiers } => format!("Key {}", spacemouse::key_label(*key, *modifiers)),
        A::Nothing => "Nothing".into(),
    }
}

/// The menu choosing what 3D mouse button `button` does: view and edit commands, a held
/// modifier, a keyboard key, or any tool. True when "Press a key…" was chosen: the key typed
/// next is the button's.
fn button_action_menu(ui: &mut egui::Ui, button: u32, action: &mut spacemouse::ButtonAction) -> bool {
    use spacemouse::ButtonAction as A;
    let mut capture = false;
    let is_key = matches!(action, A::Key { .. });
    egui::ComboBox::from_id_salt(("space_button", button)).selected_text(button_action_label(action)).width(190.0).height(420.0).show_ui(ui, |ui| {
        let mut choice = |ui: &mut egui::Ui, a: A| {
            let label = button_action_label(&a);
            ui.selectable_value(action, a, label);
        };
        ui.weak("View");
        choice(ui, A::Hotbox);
        choice(ui, A::Fit);
        for k in 0..STANDARD_VIEWS.len() {
            choice(ui, A::View(k));
        }
        ui.separator();
        ui.weak("Keyboard");
        if ui.selectable_label(is_key, "Press a key…").on_hover_text("Then type the key (with ⌘ ⇧ ⌥ ⌃ if you like): the button presses it, so it does what the key does where you are").clicked() {
            capture = true;
        }
        ui.weak("Hold as a key").on_hover_text("While the button is held, Polymix acts as if the key were held: e.g. ⌘ for the selection's ⌘ + right drag, or for brush size");
        for k in spacemouse::HeldKey::ALL {
            choice(ui, A::Hold(k));
        }
        ui.separator();
        ui.weak("Edit");
        for a in [A::Undo, A::Redo, A::Accept, A::Cancel] {
            choice(ui, a);
        }
        ui.separator();
        ui.weak("Tools");
        for t in command::TOOLS {
            choice(ui, A::Tool(t.name.into()));
        }
        ui.separator();
        choice(ui, A::Nothing);
    });
    capture
}

/// Create a tool by name. Shared by the palette, hotkeys, menus and test scripts.
pub(crate) fn make_tool(name: &str, cx: &mut ToolCx) -> Option<Box<dyn Tool>> {
    Some(match name {
        "select" => Box::new(tools::select::SelectTool::default()),
        "inspector" => Box::new(tools::inspector::InspectorTool::new(cx)),
        "transform" => Box::new(tools::transform::TransformTool::new(cx)),
        "facetransform" => Box::new(tools::transform::TransformTool::new_faces(cx)),
        "softtransform" => Box::new(tools::transform::TransformTool::new_soft(cx)),
        "warp" => Box::new(tools::warp::WarpTool::new(cx)),
        "align" => Box::new(tools::align::AlignTool::new(cx)),
        "attracttotarget" => Box::new(tools::ops::AttractToTargetTool::new(cx)),
        "aligntotarget" => Box::new(tools::ops::AlignToTargetTool::new(cx)),
        "fitprimitive" => Box::new(tools::ops::FitPrimitiveTool::new(cx)),
        "addtube" => Box::new(tools::add_tube::AddTubeTool::new(cx)),
        "units" => Box::new(tools::units::UnitsTool::new(cx)),
        "measure" => Box::new(tools::measure::MeasureTool::new(cx)),
        "meshquery" => Box::new(tools::mesh_query::MeshQueryTool::new(cx)),
        "clearance" => Box::new(tools::clearance::ClearanceTool::new(cx)),
        "sculpt" => Box::new(tools::sculpt::SculptTool::new(cx)),
        "stamp" => Box::new(tools::stamp::StampTool::new(cx)),
        "sculptvolume" => Box::new(tools::sculpt::SculptTool::with_mode(cx, tools::sculpt::SculptMode::Volume, None)),
        "sculptsmooth" => Box::new(tools::sculpt::SculptTool::with_mode(cx, tools::sculpt::SculptMode::Surface, Some(tools::sculpt::Brush::RobustSmooth))),
        "remesh" => Box::new(tools::ops::RemeshTool::new(cx)),
        "reduce" => Box::new(tools::ops::ReduceTool::new(cx)),
        "smooth" => Box::new(tools::ops::SmoothTool::new(cx)),
        "smoothboundary" => Box::new(tools::ops::SmoothBoundaryTool::new(cx)),
        "closecracks" => Box::new(tools::ops::CloseCracksTool::new(cx)),
        "join" => Box::new(tools::ops::JoinTool::new(cx)),
        "regionbridge" => Box::new(tools::ops::RegionBridgeTool::new(cx)),
        "tubehandle" => {
            let mut t = tools::ops::RegionBridgeTool::new(cx);
            t.params = tools::ops::RegionBridgeUi::tube_handle();
            Box::new(t)
        }
        "facegroups" => Box::new(tools::ops::FaceGroupsTool::new(cx)),
        "weldboundaries" => Box::new(tools::ops::WeldBoundariesTool::new(cx)),
        "makesolid" => Box::new(tools::ops::MakeSolidTool::new(cx)),
        "makepattern" => Box::new(tools::pattern::MakePatternTool::new(cx)),
        "unwrap" => Box::new(tools::unwrap::UnwrapTool::new(cx)),
        "convex" => Box::new(tools::pattern::ConvexTool::new(cx)),
        "hair" => Box::new(tools::hair::HairTool::new(cx)),
        "pivot" => Box::new(tools::pivot::PivotTool::new(cx)),
        "planecut" => Box::new(tools::plane_cut::PlaneCutTool::new(cx)),
        "trim" => Box::new(tools::vessel::TrimTool::new(cx)),
        "flowext" => Box::new(tools::vessel::FlowExtensionTool::new(cx)),
        "cap" => Box::new(tools::vessel::CapTool::new(cx)),
        "patches" => Box::new(tools::patches::PatchesTool::new(cx)),
        "erasefill" => Box::new(tools::ops::EraseFillTool::new(cx)),
        "extrude" => Box::new(tools::ops::ExtrudeTool::new(cx, tools::ops::ExtrudeMode::Extrude)),
        "extract" => Box::new(tools::ops::ExtrudeTool::new(cx, tools::ops::ExtrudeMode::Extract)),
        "mirror" => Box::new(tools::ops::MirrorTool::new(cx)),
        "offset" => Box::new(tools::ops::OffsetSolidTool::new(cx)),
        "thicken" => Box::new(tools::ops::ThickenTool::new(cx)),
        "boolean" => Box::new(tools::ops::BooleanTool::new(cx)),
        "bridge" => Box::new(tools::ops::BridgeTool::new(cx)),
        "deviation" => Box::new(tools::deviation::DeviationTool::new(cx)),
        "cfdcheck" => Box::new(tools::cfd_check::CfdCheckTool::new(cx)),
        "centerline" => Box::new(tools::centerline_tool::CenterlineTool::new(cx)),
        "stent" => Box::new(tools::stent::StentTool::new(cx)),
        "filtercomplex" => Box::new(tools::complex::FilterComplexTool::new(cx)),
        "setuv" => Box::new(tools::unwrap::UnwrapTool::new_projection(cx)),
        _ => return None,
    })
}

/// Imported meshes: name, mesh, status note, bounding-box diagonal (for the size prompt) and
/// what else the file brought.
type ImportResult = Vec<(String, Mesh, String, f64, ImportLook)>;
/// BVHs built in the background, by object.
type BuiltBvhs = Vec<(ObjectId, std::sync::Arc<mm_core::Bvh>)>;

/// A background build of BVHs (see [`MeshWorkspace::bvh_tick`]).
struct BvhJob {
    /// The shape version of each mesh it builds for: once one is replaced, its BVH would be
    /// turned away.
    shapes: FastMap<ObjectId, u64>,
    /// Set when one is: the builds not yet begun are dropped.
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    job: Job<BuiltBvhs>,
}

/// The BVHs of `meshes`, side by side, leaving out those not yet begun once `stop` is set.
fn build_bvhs(meshes: Vec<(ObjectId, std::sync::Arc<Mesh>)>, stop: &std::sync::atomic::AtomicBool) -> BuiltBvhs {
    use rayon::prelude::*;
    meshes.into_par_iter().filter_map(|(id, m)| (!stop.load(std::sync::atomic::Ordering::Relaxed)).then(|| (id, std::sync::Arc::new(mm_core::Bvh::build(&m))))).collect()
}

/// A project read from its file, with each object's texture image (read there too; `None`
/// where it has none or it cannot be read).
type OpenedProject = (crate::project::Project, Vec<Option<std::sync::Arc<crate::scene::TextureImage>>>);

/// The GPU copy of the open tool's preview mesh, with the object it previews and the mesh
/// state (shape and topology versions) it was last synced from.
struct PreviewGpu {
    id: ObjectId,
    shape: u64,
    topo: u64,
    gpu: GpuMesh,
}

/// An object's open-boundary lines, cached in two levels: the boundary edges of a topology
/// (they last through brush strokes), and the line vertices drawn for one state of the mesh,
/// rebuilt from those edges when only vertices moved.
#[derive(Default)]
struct BoundaryLines {
    /// Topology version, and its boundary edge ids in `boundary_edge_ids` order.
    edges: Option<(u64, Vec<u32>)>,
    /// (Shape version, mesh address), and the endpoints of those edges.
    lines: Option<((u64, usize), Vec<ColorVertex>)>,
}

pub struct MeshWorkspace {
    scene: Scene,
    history: History,
    camera: Camera,
    renderer: Option<SceneRenderer>,
    gpu: HashMap<ObjectId, GpuMesh>,
    preview_gpu: Option<PreviewGpu>,
    /// Background build of the BVHs the last commits left stale (see [`Self::bvh_tick`]).
    bvh_job: Option<BvhJob>,
    /// History generation the BVHs were last looked at for.
    bvh_generation: u64,
    /// Per object, the topology version of the BVH it was last seen with: a BVH that goes
    /// stale without a topology change is refitted where it is used, not built again.
    bvh_topo: FastMap<ObjectId, u64>,
    view: ViewOptions,
    /// Tool stack: the top tool is active; closing a sub-tool returns to the one below.
    tools: Vec<Box<dyn Tool>>,
    category: Option<Category>,
    /// Meshmix parts (loaded again in the background when the category opens; shown as they
    /// were meanwhile).
    library: Option<crate::parts::Library>,
    /// The library being loaded, with the count of library changes it started at (`u64::MAX`
    /// once Reload was clicked meanwhile: its result is then out of date at once).
    library_job: Option<(u64, Job<crate::parts::Library>)>,
    /// The count of library changes `library` was loaded at (see `parts::library_changes`).
    library_changes: u64,
    /// The count of library changes a load failed at: not tried again on its own until they
    /// change (Reload, or opening the category, tries again).
    library_failed: Option<u64>,
    /// Tool switch waiting for the current tool to finish computing (then it is accepted).
    pending_switch: Option<SwitchTo>,
    status: String,
    import_job: Option<Job<ImportResult>>,
    pending_import: Option<PendingImport>,
    /// Scripts normally skip the import prompts; `prompts:on` shows them (for screenshots).
    script_prompts: bool,
    export_job: Option<Job<String>>,
    /// Project being opened (with its objects' textures), and the file the scene was last
    /// saved to / opened from.
    project_job: Option<(PathBuf, Job<OpenedProject>)>,
    project_path: Option<PathBuf>,
    autosave: Option<Autosave>,
    /// An autosave left by a session that did not exit cleanly.
    recovery: Option<PathBuf>,
    nav_drag: Option<PointerButton>,
    /// Cmd/Ctrl+middle-drag resizing the brush.
    brush_drag: bool,
    /// Width / height of the viewport last frame (for framing the scene).
    vp_aspect: f64,
    /// Point the current navigation drag holds on to (picked when it started): orbits turn
    /// about it, pans keep it under the cursor.
    nav_anchor: Option<DVec3>,
    /// Centre of mass of an object, for the mesh version it was computed for.
    com_cache: Option<(ObjectId, u64, DVec3)>,
    /// The Statistics panel's values of the active object.
    stats_cache: StatsCache,
    /// Right-button drag owned by the active tool: press position.
    tool_rmb: Option<egui::Pos2>,
    /// Previous / Next view.
    view_history: ViewHistory,
    /// Camera target glide (from, to, start time) for Recenter at Cursor.
    view_anim: Option<(glam::DVec3, glam::DVec3, f64)>,
    last_wheel: f64,
    space_moving: bool,
    /// Last pointer position seen over the viewport (for releases outside it).
    last_pointer: Option<egui::Pos2>,
    show_help: bool,
    /// 3D mouse (3Dconnexion) input and its settings (Preferences → 3D mouse).
    space: SpaceMouse,
    space_cfg: spacemouse::Settings,
    /// What each 3D mouse button does (by button number).
    space_buttons: Vec<(u32, spacemouse::ButtonAction)>,
    /// The next button pressed is added to `space_buttons` (Preferences → 3D mouse).
    space_capture: bool,
    /// The last 3D mouse button pressed, for the settings.
    space_last_button: Option<u32>,
    /// The button whose key the settings wait for (the next key typed).
    space_key_capture: Option<u32>,
    /// The brush size every brush tool shares (see [`tools::BrushSize`]), kept with the
    /// preferences.
    brush: tools::BrushSize,
    /// What the 3D mouse's move going on turns about (chosen as it started).
    space_centre: Option<DVec3>,
    /// Where to show that centre, and when the 3D mouse last moved the view (it fades after).
    space_marker: Option<(DVec3, std::time::Instant)>,
    /// Space-bar hotbox position (Meshmixer-style quick view settings).
    hotbox: Option<egui::Pos2>,
    show_objects: bool,
    /// The object browser folded to its title (click it to open again).
    objects_folded: bool,
    /// Meshes for the Segmentation workspace (Send to Segmentation), taken by the shell.
    pub(crate) to_segmentation: Vec<(String, Mesh)>,
    /// The Segmentation tab's image as slices among the meshes.
    pub(crate) slices: slices::ImageSlices,
    /// Width of the object browser (drag its left edge).
    objects_width: f32,
    /// Height of an object browser row, as last drawn (rows out of view are not drawn).
    objects_row_h: Option<f32>,
    prefs: Prefs,
    show_prefs: bool,
    prefs_tab: usize,
    last_export_dir: Option<PathBuf>,
    export_ascii: bool,
    /// Where the floating tool panel was last drawn (keeps viewport decorations clear of it).
    tool_panel_rect: Option<egui::Rect>,
    /// Width of the floating tool panel (drag its right edge).
    tool_panel_width: f32,
    recent: Vec<PathBuf>,
    script: Option<Script>,
    /// Boundary-edge lines per object (kept while the object is in the scene, shown or not).
    boundary_cache: FastMap<ObjectId, BoundaryLines>,
    /// History generation `boundary_cache` was last pruned at.
    boundary_pruned: u64,
    /// The viewport's line vertices, kept between frames so their memory is reused.
    lines_buf: Vec<ColorVertex>,
    /// Synthetic pointer events queued by scripts (consumed one per frame).
    synthetic: std::collections::VecDeque<PointerInput>,
    /// Files an agent opened: imported without prompts, like test scripts.
    agent_import: bool,
    /// Frames without anything left to happen since an agent's last change; after three, undo
    /// steps are the user's again (see `History::by_agent`).
    agent_quiet_frames: u32,
}

/// Scripted UI actions for automated testing (`POLYMIX_SCRIPT`), e.g.
/// `tool:inspector;toolcmd:...;log:tag`. The shell runs the script and hands the Mesh
/// workspace its steps; waits, key presses and screenshots are the shell's.
#[derive(Default)]
struct Script {
    steps: std::collections::VecDeque<String>,
    wait: u32,
}

impl MeshWorkspace {
    /// A fresh Mesh workspace (also used to restart it after a crash), reading its
    /// preferences from `storage`.
    pub fn new(ctx: &egui::Context, render_state: Option<&eframe::egui_wgpu::RenderState>, storage: Option<&dyn eframe::Storage>) -> Self {
        ui_util::apply_theme(ctx, ui_util::Theme::Dark);
        let renderer = render_state.map(SceneRenderer::new);
        let mut app = Self {
            scene: Scene::default(),
            history: History::default(),
            camera: Camera::default(),
            renderer,
            gpu: HashMap::new(),
            preview_gpu: None,
            bvh_job: None,
            bvh_generation: 0,
            bvh_topo: FastMap::default(),
            view: ViewOptions::default(),
            tools: Vec::new(),
            category: None,
            library: None,
            library_job: None,
            library_changes: crate::parts::library_changes(),
            library_failed: None,
            pending_switch: None,
            status: "Welcome to Polymix. Import a mesh to begin (Ctrl/Cmd+O, or drag a file onto the window).".into(),
            script_prompts: false,
            pending_import: None,
            import_job: None,
            export_job: None,
            project_job: None,
            project_path: None,
            autosave: None,
            recovery: None,
            nav_drag: None,
            brush_drag: false,
            vp_aspect: 1.6,
            nav_anchor: None,
            com_cache: None,
            stats_cache: StatsCache::default(),
            tool_rmb: None,
            view_history: ViewHistory::default(),
            view_anim: None,
            last_wheel: 0.0,
            space_moving: false,
            last_pointer: None,
            show_help: false,
            space: SpaceMouse::start(ctx),
            space_cfg: spacemouse::Settings::default(),
            space_buttons: spacemouse::default_buttons(),
            space_capture: false,
            space_last_button: None,
            space_key_capture: None,
            brush: tools::BrushSize::default(),
            space_centre: None,
            space_marker: None,
            hotbox: None,
            show_objects: true,
            objects_folded: false,
            to_segmentation: Vec::new(),
            slices: slices::ImageSlices::new(),
            objects_width: 230.0,
            objects_row_h: None,
            prefs: Prefs::default(),
            show_prefs: false,
            prefs_tab: 0,
            last_export_dir: None,
            export_ascii: false,
            tool_panel_rect: None,
            tool_panel_width: 300.0,
            recent: Vec::new(),
            synthetic: Default::default(),
            boundary_cache: FastMap::default(),
            boundary_pruned: 0,
            lines_buf: Vec::new(),
            script: None,
            agent_import: false,
            agent_quiet_frames: 0,
        };
        if let Some(st) = storage.filter(|_| std::env::var_os("POLYMIX_SCRIPT").is_none()) {
            let flag = |k: &str, d: bool| st.get_string(k).map(|v| v == "1").unwrap_or(d);
            app.view.wireframe = flag("wireframe", false);
            // Older settings only had a flat-shading flag.
            app.view.normals = st
                .get_string("normals")
                .and_then(|k| NormalMode::from_key(&k))
                .unwrap_or(if flag("flat", false) { NormalMode::Face } else { NormalMode::Group });
            app.view.boundaries = flag("boundaries", true);
            // Older settings only had a group-colours flag.
            app.view.colors = st
                .get_string("color_mode")
                .and_then(|k| ColorMode::from_key(&k))
                .unwrap_or(if flag("group_colors", false) { ColorMode::Group } else { ColorMode::Vertex });
            app.view.show_grid = flag("grid", false);
            app.view.grid_mm = st.get_string("grid_mm").and_then(|v| v.parse().ok()).unwrap_or(0.0);
            app.view.transparent_target = flag("transparent_target", true);
            app.view.xray = flag("xray", false);
            app.view.show_view_cube = flag("view_cube", true);
            app.view.show_pivots = flag("pivots", true);
            if let Some(t) = st.get_string("theme").and_then(|k| ui_util::Theme::from_key(&k)) {
                app.view.theme = t;
                ui_util::apply_theme(ctx, t);
            }
            if let Some(c) = st.get_string("background") {
                let v: Vec<f32> = c.split(',').filter_map(|x| x.parse().ok()).collect();
                if let [r, g, b] = v[..] {
                    app.view.background = [r, g, b];
                }
            }
            app.view.sharp_edges = flag("sharp_edges", true);
            app.view.nav_mode = st.get_string("nav_mode").and_then(|k| NavMode::from_key(&k)).unwrap_or(NavMode::Fusion360);
            app.view.nav_custom = st.get_string("nav_custom").and_then(|v| NavBindings::from_pref(&v)).unwrap_or(NavBindings::FUSION360);
            app.view.brush_scroll = st.get_string("brush_scroll").and_then(|k| BrushScroll::from_key(&k)).unwrap_or(BrushScroll::Shift);
            if let Some(pct) = st.get_string("brush_size").and_then(|v| v.parse::<f32>().ok()) {
                app.brush.set(pct);
            }
            app.view.zoom_speed = st.get_string("zoom_speed").and_then(|v| v.parse().ok()).filter(|x: &f32| x.is_finite() && *x > 0.0).unwrap_or(1.0);
            app.view.zoom_near = flag("zoom_near", true);
            app.view.zoom_to_cursor = flag("zoom_to_cursor", true);
            if let Some(cfg) = st.get_string("spacemouse").and_then(|v| spacemouse::Settings::from_pref(&v)) {
                app.space_cfg = cfg;
            }
            if let Some(v) = st.get_string("spacemouse_buttons") {
                app.space_buttons = spacemouse::buttons_from_pref(&v);
            }
            app.sync_held_keys();
            app.view.crease_deg = st.get_string("crease_deg").and_then(|v| v.parse().ok()).unwrap_or(DEFAULT_CREASE_DEG);
            app.show_objects = flag("show_objects", true);
            app.objects_folded = flag("objects_folded", false);
            app.objects_width = st.get_string("objects_width").and_then(|v| v.parse().ok()).filter(|w: &f32| w.is_finite()).unwrap_or(230.0).clamp(OBJECTS_WIDTH.0, OBJECTS_WIDTH.1);
            app.camera.orthographic = flag("orthographic", false);
            app.camera.free_orbit = flag("free_orbit", true);
            if st.get_string("up_axis").as_deref() == Some("z") {
                app.camera.up_axis = UpAxis::Z;
            }
            app.last_export_dir = st.get_string("last_export_dir").map(PathBuf::from);
            app.export_ascii = flag("export_ascii", false);
            app.prefs.reset_camera_on_import = flag("reset_camera_on_import", true);
            app.prefs.export_whole_scene = flag("export_whole_scene", false);
            app.prefs.obj_export_groups = flag("obj_export_groups", true);
            app.prefs.obj_import_groups = flag("obj_import_groups", true);
            app.prefs.flip_zy = flag("flip_zy", true);
            app.prefs.autosave_minutes = st.get_string("autosave_minutes").and_then(|v| v.parse().ok()).unwrap_or(5);
            crate::tools::ops::FILL_NEW_GROUPS.store(flag("fill_new_groups", false), std::sync::atomic::Ordering::Relaxed);
            app.prefs.import_mode = match st.get_string("import_mode").as_deref() {
                Some("append") => ImportMode::Append,
                Some("replace") => ImportMode::Replace,
                _ => ImportMode::Ask,
            };
            app.prefs.scale_mode = match st.get_string("scale_mode").as_deref() {
                Some("never") => ScaleMode::Never,
                Some(s) if s.starts_with("always:") => s[7..].parse().map(ScaleMode::Always).unwrap_or(ScaleMode::Ask),
                _ => ScaleMode::Ask,
            };
            app.recent = st.get_string("recent").map(|r| r.lines().filter(|l| !l.is_empty()).map(PathBuf::from).collect()).unwrap_or_default();
        }
        // Autosave and crash recovery (never in scripted test runs).
        if std::env::var_os("POLYMIX_SCRIPT").is_none() {
            if let Some(dir) = data_dir().filter(|d| std::fs::create_dir_all(d).is_ok()) {
                let file = dir.join(format!("autosave.{}", crate::project::EXTENSION));
                let marker = dir.join("session.lock");
                // A marker left behind means the last session did not exit cleanly.
                if marker.exists() && file.exists() {
                    app.recovery = Some(file.clone());
                }
                let _ = std::fs::write(&marker, std::process::id().to_string());
                app.autosave = Some(Autosave { file, marker, saved_generation: 0, last_time: 0.0, job: None });
            }
        }
        app
    }

    /// Edit → Copy: the open tool's readout, or the active object's statistics, as text.
    fn copy(&mut self, ctx: &egui::Context) {
        let text = self.tools.last().and_then(|t| t.copy_text()).or_else(|| {
            let o = self.scene.active()?;
            let m = &o.mesh;
            let e = m.bounds().extents();
            let mut s = format!(
                "{}\ntriangles {}\nvertices {}\nsize {:.4} x {:.4} x {:.4}\narea {:.4}\nopen edges {}",
                o.name,
                m.triangle_count(),
                m.vertex_count(),
                e.x,
                e.y,
                e.z,
                mm_core::measure::surface_area(m),
                m.boundary_edge_ids().count()
            );
            if m.is_closed() {
                s += &format!("\nvolume {:.4}", mm_core::measure::volume(m));
            }
            Some(s)
        });
        if let Some(t) = text {
            ctx.copy_text(t);
            self.status = "Copied to the clipboard".into();
        }
    }

    /// Save a changed scene to the autosave file every few minutes (in the background).
    fn autosave_tick(&mut self, ctx: &egui::Context) {
        let now = ctx.input(|i| i.time);
        if let Some(job) = self.autosave.as_ref().and_then(|a| a.job.as_ref()) {
            // Still saving: the next save is scheduled once this one is done, below.
            let Some(r) = job.poll() else { return };
            if let Err(e) = r {
                log::warn!("autosave failed: {e:#}");
            }
            if let Some(auto) = &mut self.autosave {
                auto.job = None;
            }
        }
        match self.autosave_wait(now) {
            None => return,
            Some(wait) if wait > 0.0 => {
                // Frames come only with input or work, so ask for one when the save is due.
                ctx.request_repaint_after(std::time::Duration::from_secs_f64(wait));
                return;
            }
            Some(_) => {}
        }
        // Not in the middle of a gesture (a stroke, a drag): the save waits for the release,
        // which brings a frame of its own.
        if ctx.input(|i| i.pointer.any_down()) {
            return;
        }
        let generation = self.history.generation();
        let project = self.autosave_project();
        let Some(auto) = &mut self.autosave else { return };
        auto.saved_generation = generation;
        auto.last_time = now;
        let file = auto.file.clone();
        auto.job = Some(Job::spawn("Autosave", ctx.clone(), move |_| crate::project::save_shared(&file, &project, crate::project::Compression::Fast)));
    }

    /// Seconds until the unsaved changes are due for an autosave (0 once due), or `None` when
    /// there is nothing to save, a save is running, autosave is off or recovery is asked.
    fn autosave_wait(&self, now: f64) -> Option<f64> {
        let auto = self.autosave.as_ref()?;
        let minutes = self.prefs.autosave_minutes;
        if auto.job.is_some() || self.recovery.is_some() || minutes == 0 || self.history.generation() == auto.saved_generation || self.scene.is_empty() {
            return None;
        }
        let (elapsed, period) = (now - auto.last_time, minutes as f64 * 60.0);
        Some(if elapsed < period { period - elapsed } else { 0.0 })
    }

    /// After a commit (a tool's result applied, a command, undo or redo), build in the
    /// background the BVHs that the next orbit or pick would otherwise build on the UI thread:
    /// those of visible objects whose BVH is missing or was made for another topology. Never
    /// while a button is held or anything is running; a BVH that is stale by the time it is
    /// done is turned away by `install_bvh`.
    fn bvh_tick(&mut self, ctx: &egui::Context) {
        // BVHs built or refitted by picks: one of the same topology is only refitted later.
        for o in &self.scene.objects {
            if o.bvh_if_current().is_some() {
                self.bvh_topo.insert(o.id, o.mesh.topology_version());
            }
        }
        if let Some(b) = &self.bvh_job {
            let Some(r) = b.job.poll() else {
                // A commit, undo or project opened since it started may have replaced meshes
                // it builds for. What is still needed is started once it ends; a build under
                // way cannot be stopped, but none is begun for nothing.
                if !b.stop.load(std::sync::atomic::Ordering::Relaxed) {
                    let current = self.scene.objects.iter().filter(|o| b.shapes.get(&o.id) == Some(&o.mesh.shape_version())).count();
                    if current < b.shapes.len() {
                        b.stop.store(true, std::sync::atomic::Ordering::Relaxed);
                    }
                }
                return;
            };
            self.bvh_job = None;
            match r {
                Ok(built) => {
                    for (id, bvh) in built {
                        if let Some(o) = self.scene.get_mut(id) {
                            if o.install_bvh(bvh) {
                                self.bvh_topo.insert(id, o.mesh.topology_version());
                            }
                        }
                    }
                }
                Err(e) => log::warn!("background BVH build failed: {e:#}"),
            }
        }
        let generation = self.history.generation();
        if generation == self.bvh_generation {
            return;
        }
        // An open tool picks on its own (brush cursors, hover), and work under way changes
        // the meshes: wait until both are done.
        let busy = !self.tools.is_empty()
            || self.pending_switch.is_some()
            || !self.synthetic.is_empty()
            || self.import_job.is_some()
            || self.project_job.is_some()
            || ctx.input(|i| i.pointer.any_down());
        if busy {
            return;
        }
        self.bvh_generation = generation;
        let ids: FastSet<ObjectId> = self.scene.objects.iter().map(|o| o.id).collect();
        self.bvh_topo.retain(|id, _| ids.contains(id));
        let stale: Vec<(ObjectId, std::sync::Arc<Mesh>)> = self
            .scene
            .objects
            .iter()
            .filter(|o| o.visible && !o.reference && o.bvh_if_current().is_none() && self.bvh_topo.get(&o.id) != Some(&o.mesh.topology_version()))
            .map(|o| (o.id, o.mesh.clone()))
            .collect();
        if stale.is_empty() {
            return;
        }
        let shapes = stale.iter().map(|(id, m)| (*id, m.shape_version())).collect();
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let s = stop.clone();
        let job = Job::spawn("Preparing picking", ctx.clone(), move |_| Ok(build_bvhs(stale, &s)));
        self.bvh_job = Some(BvhJob { shapes, stop, job });
    }

    /// The scene as an autosave project. It shares the scene's meshes and UVs: nothing is
    /// copied here, on the UI thread.
    fn autosave_project(&self) -> crate::project::SaveProject {
        let objects = self.scene.objects.iter().map(|o| save_object(o, o.group_names.iter().map(|(g, n)| (*g, n.clone())).collect())).collect();
        crate::project::SaveProject { objects, active: None, target: None, view: Some(self.camera.view_state()) }
    }

    /// Offer to restore the autosave of a session that crashed.
    fn recovery_window(&mut self, ctx: &egui::Context) {
        let Some(path) = self.recovery.clone() else { return };
        let when = std::fs::metadata(&path)
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.elapsed().ok())
            .map(|d| format!("{} min ago", d.as_secs() / 60))
            .unwrap_or_default();
        let mut choice = None;
        egui::Window::new("Recover unsaved work")
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .collapsible(false)
            .resizable(false)
            .show(ctx, |ui| {
                ui.label(format!("Polymix did not close properly last time. Restore the scene it autosaved ({when})?"));
                ui.horizontal(|ui| {
                    if ui.button("Recover").clicked() {
                        choice = Some(true);
                    }
                    if ui.button("Discard").clicked() {
                        choice = Some(false);
                    }
                });
            });
        match choice {
            Some(true) => {
                self.recovery = None;
                self.open_project(path, ctx);
                // Recovered work is unsaved: Save asks for a file.
                self.project_path = None;
            }
            Some(false) => {
                self.recovery = None;
                let _ = std::fs::remove_file(&path);
            }
            None => {}
        }
    }

    // ------------------------------------------------------------------
    // File IO
    // ------------------------------------------------------------------

    fn start_import(&mut self, files: Vec<PathBuf>, ctx: &egui::Context) {
        self.start_import_as(files, false, ctx);
    }

    /// [`Self::start_import`]; as references (see-through guides) when `reference`.
    fn start_import_as(&mut self, files: Vec<PathBuf>, reference: bool, ctx: &egui::Context) {
        let (projects, files): (Vec<PathBuf>, Vec<PathBuf>) =
            files.into_iter().partition(|f| f.extension().is_some_and(|e| e.eq_ignore_ascii_case(crate::project::EXTENSION)));
        if let Some(p) = projects.into_iter().next() {
            self.open_project(p, ctx);
            return;
        }
        if files.is_empty() || self.import_job.is_some() {
            return;
        }
        for f in &files {
            self.recent.retain(|r| r != f);
            self.recent.insert(0, f.clone());
        }
        self.recent.truncate(10);
        let n = files.len();
        let obj_groups = self.prefs.obj_import_groups;
        // Several files are read at once, up to half the cores (which bounds the memory of
        // files being parsed).
        let workers = std::thread::available_parallelism().map_or(1, |c| c.get() / 2);
        self.import_job = Some(Job::spawn(format!("Importing {n} file(s)"), ctx.clone(), move |progress| import_files(&files, obj_groups, reference, workers, progress)));
    }

    /// The image the Segmentation tab shows, for the image slices (`None`: none is open).
    pub fn set_image_source(&mut self, source: Option<slices::SliceSource>) {
        self.slices.set_source(source);
    }

    /// Meshes sent to the Segmentation workspace (Send to Segmentation), taken by the shell.
    pub fn take_for_segmentation(&mut self) -> Vec<(String, Mesh)> {
        std::mem::take(&mut self.to_segmentation)
    }

    /// Add meshes made in another workspace (Segmentation's Make mesh), in file coordinates
    /// (LPS, Z up, like 3D Slicer's model files): they go through the import prompts like
    /// imported files, unless they were made for an agent.
    pub fn add_meshes(&mut self, meshes: Vec<(String, Mesh)>, by_agent: bool) {
        if by_agent {
            // Added without prompts, labelled as the agent's.
            self.agent_import = true;
            self.history.by_agent = true;
            self.agent_quiet_frames = 0;
        }
        let mut meshes: Vec<(String, Mesh, String, f64)> = meshes
            .into_iter()
            .map(|(name, mesh)| {
                let note = format!("{name}: {} triangles from Segmentation", mesh.triangle_count());
                let diagonal = mesh.bounds().diagonal();
                (name, mesh, note, diagonal)
            })
            .collect();
        if let Some(p) = &mut self.pending_import {
            p.looks.extend(meshes.iter().map(|_| ImportLook::default()));
            p.meshes.append(&mut meshes);
            return;
        }
        let placement = match self.prefs.import_mode {
            _ if self.scene.is_empty() => Some(false),
            ImportMode::Ask => None,
            ImportMode::Append => Some(false),
            ImportMode::Replace => Some(true),
        };
        let looks = meshes.iter().map(|_| ImportLook::default()).collect();
        self.pending_import = Some(PendingImport { meshes, looks, placement, remember: false });
    }

    fn poll_jobs(&mut self) {
        if let Some(job) = &self.import_job {
            if let Some(res) = job.poll() {
                self.import_job = None;
                match res {
                    Ok(imported) => {
                        let placement = match self.prefs.import_mode {
                            _ if self.scene.is_empty() => Some(false),
                            ImportMode::Ask => None,
                            ImportMode::Append => Some(false),
                            ImportMode::Replace => Some(true),
                        };
                        let (meshes, looks) = imported.into_iter().map(|(n, m, note, d, look)| ((n, m, note, d), look)).unzip();
                        self.pending_import = Some(PendingImport { meshes, looks, placement, remember: false });
                    }
                    Err(e) => self.status = format!("Import failed: {e:#}"),
                }
            }
        }
        if let Some((path, job)) = &self.project_job {
            if let Some(res) = job.poll() {
                let path = path.clone();
                self.project_job = None;
                match res {
                    Ok(p) => self.apply_project(p, path),
                    Err(e) => self.status = format!("Open failed: {e:#}"),
                }
            }
        }
        if let Some(job) = &self.export_job {
            if let Some(res) = job.poll() {
                self.export_job = None;
                self.status = match res {
                    Ok(msg) => msg,
                    Err(e) => format!("Export failed: {e:#}"),
                };
            }
        }
        if let Some((changes, job)) = &self.library_job {
            if let Some(res) = job.poll() {
                let changes = *changes;
                self.library_job = None;
                match res {
                    // Newer than what is shown, but if the files changed meanwhile it is still
                    // out of date (the panel then loads it again).
                    Ok(lib) => (self.library, self.library_changes, self.library_failed) = (Some(lib), changes, None),
                    Err(e) => {
                        log::warn!("loading the parts library failed: {e:#}");
                        self.library_failed = Some(changes);
                    }
                }
            }
        }
    }

    /// Import prompts (Append / Replace, odd sizes), then add the meshes to the scene.
    fn finish_import(&mut self, ctx: &egui::Context) {
        // Test scripts and agents never wait on prompts: they append and keep sizes.
        let scripted = (polymix_workspace::scripted() && !self.script_prompts) || self.agent_import;
        let Some(pending) = &mut self.pending_import else { return };
        if scripted && pending.placement.is_none() {
            pending.placement = Some(false);
        }
        let center = |w: egui::Window<'static>| w.anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0)).collapsible(false).resizable(false);
        if pending.placement.is_none() {
            let mut choice = None;
            center(egui::Window::new("Import")).show(ctx, |ui| {
                ui.label("The scene already has objects. Append the imported mesh or replace the scene?");
                ui.checkbox(&mut self.prefs.flip_zy, "Flip Z-Y axis on import/export");
                ui.checkbox(&mut pending.remember, "Remember my choice");
                ui.horizontal(|ui| {
                    if ui.button("Append").clicked() {
                        choice = Some(false);
                    }
                    if ui.button("Replace").clicked() {
                        choice = Some(true);
                    }
                    if ui.button("Cancel").clicked() {
                        choice = Some(false);
                        pending.meshes.clear();
                        pending.looks.clear();
                    }
                });
            });
            if let Some(replace) = choice {
                if pending.remember && !pending.meshes.is_empty() {
                    self.prefs.import_mode = if replace { ImportMode::Replace } else { ImportMode::Append };
                }
                pending.placement = Some(replace);
                pending.remember = false;
            }
            return;
        }
        let odd = pending.meshes.iter().any(|(_, _, _, d)| odd_diagonal(*d));
        let factor = match self.prefs.scale_mode {
            _ if !odd || scripted => Some(1.0),
            ScaleMode::Never => Some(1.0),
            ScaleMode::Always(k) => Some(k),
            ScaleMode::Ask => {
                let d = pending.meshes.iter().map(|(_, _, _, d)| *d).fold(0.0, f64::max);
                let mut pick = None;
                center(egui::Window::new("Model size")).show(ctx, |ui| {
                    ui.label(format!(
                        "The model is {} ({d:.4} units across). Polymix works in millimetres; scale it?",
                        if d < 1.0 { "very small" } else { "very large" }
                    ));
                    ui.checkbox(&mut pending.remember, "Remember my choice");
                    ui.horizontal_wrapped(|ui| {
                        let options: &[(&str, f64)] = if d < 1.0 {
                            &[("× 1000 (m → mm)", 1000.0), ("× 10 (cm → mm)", 10.0), ("× 25.4 (in → mm)", 25.4)]
                        } else {
                            &[("× 0.001 (µm → mm)", 0.001), ("× 0.1", 0.1)]
                        };
                        for &(label, k) in options {
                            if ui.button(label).clicked() {
                                pick = Some(k);
                            }
                        }
                        if ui.button("Keep size").clicked() {
                            pick = Some(1.0);
                        }
                    });
                });
                if let Some(k) = pick {
                    if pending.remember {
                        self.prefs.scale_mode = if k == 1.0 { ScaleMode::Never } else { ScaleMode::Always(k) };
                    }
                }
                pick
            }
        };
        let Some(factor) = factor else { return };
        let Some(pending) = self.pending_import.take() else { return };
        self.agent_import = false;
        if pending.meshes.is_empty() {
            self.status = "Import cancelled".into();
            return;
        }
        self.history.push("Import", &self.scene);
        if pending.placement == Some(true) {
            self.scene.objects.clear();
            self.scene.active = None;
            self.scene.target = None;
        }
        let first_import = self.scene.is_empty();
        let mut notes = Vec::new();
        let mut looks = pending.looks.into_iter();
        for (name, mut mesh, mut note, _) in pending.meshes {
            if self.prefs.flip_zy {
                mesh.transform(flip_zy_in);
            }
            if factor != 1.0 && odd_size(&mesh) {
                mesh.transform(|p| p * factor);
                note += &format!(" — scaled × {factor}");
            }
            let look = looks.next().unwrap_or_default();
            let id = self.scene.add(name, mesh);
            if let Some(o) = self.scene.get_mut(id) {
                if let Some(uvs) = look.uvs {
                    o.uvs = Some(std::sync::Arc::new(crate::scene::UvSet::new(&o.mesh, uvs)));
                }
                o.texture = look.texture;
                if look.reference {
                    o.reference = true;
                    o.color = glam::Vec4::new(0.62, 0.66, 0.72, 0.35);
                }
            }
            notes.push(note);
        }
        self.status = notes.join("  |  ");
        if first_import || self.prefs.reset_camera_on_import {
            self.camera.scene_radius = 1e-6;
            self.camera.focus(&self.scene.bounds());
        }
    }

    // ------------------------------------------------------------------
    // Project files
    // ------------------------------------------------------------------

    fn open_project(&mut self, path: PathBuf, ctx: &egui::Context) {
        if self.project_job.is_some() {
            return;
        }
        self.recent.retain(|r| r != &path);
        self.recent.insert(0, path.clone());
        self.recent.truncate(10);
        let p = path.clone();
        self.project_job = Some((
            path,
            Job::spawn("Opening project", ctx.clone(), move |_| {
                use rayon::prelude::*;
                let project = crate::project::load(&p)?;
                // Textures are read again from their files (missing: the object shows without).
                let textures = project.objects.par_iter().map(|o| o.texture.as_ref().and_then(|t| crate::scene::TextureImage::load(t).ok()).map(std::sync::Arc::new)).collect();
                Ok((project, textures))
            }),
        ));
    }

    fn apply_project(&mut self, (p, textures): OpenedProject, path: PathBuf) {
        self.history.push("Open project", &self.scene);
        self.scene.objects.clear();
        self.scene.active = None;
        self.scene.target = None;
        let mut ids = Vec::new();
        let mut textures = textures.into_iter();
        for o in p.objects {
            let texture = textures.next().flatten();
            let id = self.scene.add(o.name, o.mesh);
            if let Some(so) = self.scene.get_mut(id) {
                so.visible = o.visible;
                so.color = o.color;
                so.material = o.material;
                so.frame = o.frame;
                so.group_names = o.group_names.into_iter().collect();
                so.uvs = o.uvs.map(|c| std::sync::Arc::new(crate::scene::UvSet::new(&so.mesh, c)));
                so.pivots = o.pivots;
                so.reference = o.reference;
                so.texture = texture;
            }
            ids.push(id);
        }
        self.scene.active = p.active.and_then(|i| ids.get(i).copied()).or(ids.last().copied());
        self.scene.target = p.target.and_then(|i| ids.get(i).copied());
        match p.view {
            Some(v) => self.camera.set_view_state(v),
            None => self.focus_scene(),
        }
        self.status = format!("Opened {}", path.display());
        self.project_path = Some(path);
    }

    fn open_project_dialog(&mut self, ctx: &egui::Context) {
        if let Some(path) = rfd::FileDialog::new().add_filter("Polymix project", &[crate::project::EXTENSION]).pick_file() {
            self.open_project(path, ctx);
        }
    }

    /// Save the scene as a project (asking for a file when there is none yet or `save_as`).
    fn save_project(&mut self, ctx: &egui::Context, save_as: bool) {
        if self.scene.is_empty() {
            self.status = "Nothing to save.".into();
            return;
        }
        let path = match (&self.project_path, save_as) {
            (Some(p), false) => p.clone(),
            _ => {
                let name = self.scene.active().map(|o| o.name.clone()).unwrap_or_else(|| "scene".into());
                let mut dlg = rfd::FileDialog::new().add_filter("Polymix project", &[crate::project::EXTENSION]).set_file_name(format!("{name}.{}", crate::project::EXTENSION));
                if let Some(d) = self.project_path.as_ref().and_then(|p| p.parent()).or(self.last_export_dir.as_deref()) {
                    dlg = dlg.set_directory(d);
                }
                let Some(p) = dlg.save_file() else { return };
                p
            }
        };
        if let Err(e) = self.save_project_to(path, ctx) {
            self.status = e;
        }
    }

    /// Save the scene as a project at `path` (in the background).
    fn save_project_to(&mut self, path: PathBuf, ctx: &egui::Context) -> Result<String, String> {
        if self.scene.is_empty() {
            return Err("Nothing to save.".into());
        }
        let index = |id: Option<ObjectId>| id.and_then(|id| self.scene.objects.iter().position(|o| o.id == id));
        // The scene's meshes and UVs are shared with the job, not copied here.
        let objects: Vec<crate::project::SaveObject> = self
            .scene
            .objects
            .iter()
            .map(|o| {
                let mut groups: Vec<(i32, String)> = o.group_names.iter().map(|(g, n)| (*g, n.clone())).collect();
                groups.sort();
                save_object(o, groups)
            })
            .collect();
        let project = crate::project::SaveProject {
            objects,
            active: index(self.scene.active),
            target: index(self.scene.target),
            view: Some(self.camera.view_state()),
        };
        self.project_path = Some(path.clone());
        let msg = format!("Saving the project to {}", path.display());
        self.export_job = Some(Job::spawn("Saving", ctx.clone(), move |_| {
            crate::project::save_shared(&path, &project, crate::project::Compression::Default)?;
            Ok(format!("Saved {}", path.display()))
        }));
        Ok(msg)
    }

    fn import_dialog(&mut self, ctx: &egui::Context) {
        let files = rfd::FileDialog::new()
            .add_filter("Meshes", &["stl", "obj", "ply", "off", "3mf", "amf", crate::project::EXTENSION])
            .add_filter("All files", &["*"])
            .pick_files();
        if let Some(files) = files {
            self.start_import(files, ctx);
        }
    }

    fn export_dialog(&mut self, ctx: &egui::Context, scope: ExportScope) {
        let Some(active) = self.scene.active() else {
            self.status = "Nothing to export.".into();
            return;
        };
        if scope == ExportScope::Selection && active.selection.is_empty() {
            self.status = "Select the triangles to export first.".into();
            return;
        }
        let mut dlg = rfd::FileDialog::new()
            .add_filter("STL (binary)", &["stl"])
            .add_filter("OBJ", &["obj"])
            .add_filter("PLY", &["ply"])
            .add_filter("OFF", &["off"])
            .add_filter("3MF", &["3mf"])
            .add_filter("AMF", &["amf"])
            .add_filter("COLLADA", &["dae"])
            .add_filter("VRML", &["wrl"])
            .add_filter("TetGen SMESH (face groups as boundary markers)", &["smesh"])
            .set_file_name(format!("{}.stl", active.name));
        if let Some(d) = &self.last_export_dir {
            dlg = dlg.set_directory(d);
        }
        let Some(path) = dlg.save_file() else { return };
        self.last_export_dir = path.parent().map(|p| p.to_path_buf());
        if let Err(e) = self.export_to(path, scope, ctx) {
            self.status = e;
        }
    }

    /// Export to `path` (format from the extension) in the background.
    fn export_to(&mut self, path: PathBuf, scope: ExportScope, ctx: &egui::Context) -> Result<String, String> {
        let Some(active) = self.scene.active() else {
            return Err("Nothing to export.".into());
        };
        if scope == ExportScope::Selection && active.selection.is_empty() {
            return Err("Select the triangles to export first.".into());
        }
        if mm_io::Format::from_path(&path).is_none() {
            return Err(format!("Unknown mesh format for {} (use .stl, .obj, .ply, .off, .3mf, .amf, .dae, .wrl or .smesh)", path.display()));
        }
        if matches!(scope, ExportScope::PerGroup | ExportScope::PerObject) {
            return self.export_sequence(path, scope, ctx);
        }
        // What to write, taken from the scene without copying a mesh: the job assembles it.
        let (source, triangles) = match scope {
            ExportScope::PerGroup | ExportScope::PerObject => unreachable!("handled above"),
            ExportScope::AllVisible => {
                let meshes: Vec<std::sync::Arc<Mesh>> = self.scene.objects.iter().filter(|o| o.visible && !o.reference).map(|o| o.mesh.clone()).collect();
                let triangles = meshes.iter().map(|m| m.triangle_count()).sum();
                (ExportSource::Combined(meshes), triangles)
            }
            ExportScope::Active => (ExportSource::Whole(active.mesh.clone()), active.mesh.triangle_count()),
            ExportScope::Selection => (ExportSource::Part(active.mesh.clone(), active.selection.clone()), active.selection.len()),
        };
        let flip = self.prefs.flip_zy;
        let p2 = path.clone();
        // UVs go out with the active object (OBJ).
        let uvs = match scope {
            ExportScope::Active => active.uvs.clone().filter(|u| u.topo == active.mesh.topology_version()),
            _ => None,
        };
        let mut opts = mm_io::WriteOptions { stl_ascii: self.export_ascii, obj_groups: self.prefs.obj_export_groups, ..Default::default() };
        let msg = format!("Exporting {triangles} triangles to {}", p2.display());
        self.export_job = Some(Job::spawn("Exporting", ctx.clone(), move |_| {
            let mesh = source.mesh(flip);
            opts.uvs = uvs.map(|u| mesh.triangle_ids().map(|t| u.corners[t as usize]).collect());
            mm_io::write_mesh(&p2, &mesh, &opts)?;
            Ok(format!("Exported {} triangles to {}", mesh.triangle_count(), p2.display()))
        }));
        Ok(msg)
    }

    /// Export one file per face group of the active object, or per visible object, next to
    /// `path`: its name with the group's (or object's) name added.
    fn export_sequence(&mut self, path: PathBuf, scope: ExportScope, ctx: &egui::Context) -> Result<String, String> {
        let safe = |s: &str| -> String { s.chars().map(|c| if c.is_alphanumeric() || "-_ ".contains(c) { c } else { '_' }).collect::<String>().trim().replace(' ', "_") };
        let stem = path.file_stem().map_or("mesh".to_string(), |s| s.to_string_lossy().to_string());
        let ext = path.extension().map_or("stl".to_string(), |e| e.to_string_lossy().to_string());
        let dir = path.parent().map(|p| p.to_path_buf()).unwrap_or_default();
        // The meshes are shared with the job, which cuts the groups out (in group order) and
        // copies what it turns, one file at a time; only the file count is needed here.
        let (sources, n): (ExportParts, usize) = match scope {
            ExportScope::PerGroup => {
                let o = self.scene.active().ok_or("Nothing to export.")?;
                let n = distinct_groups(&o.mesh);
                (ExportParts::Groups(o.mesh.clone(), o.group_names.clone()), n)
            }
            _ => {
                let parts: Vec<(String, std::sync::Arc<Mesh>)> = self.scene.objects.iter().filter(|o| o.visible && !o.reference).map(|o| (o.name.clone(), o.mesh.clone())).collect();
                let n = parts.len();
                (ExportParts::Objects(parts), n)
            }
        };
        if n == 0 {
            return Err("Nothing to export.".into());
        }
        let flip = self.prefs.flip_zy;
        let opts = mm_io::WriteOptions { stl_ascii: self.export_ascii, obj_groups: self.prefs.obj_export_groups, ..Default::default() };
        let msg = format!("Exporting {n} files to {}", dir.display());
        self.export_job = Some(Job::spawn("Exporting", ctx.clone(), move |progress| {
            let mut seen = std::collections::HashSet::new();
            let mut write = |i: usize, name: &str, source: ExportSource| -> anyhow::Result<()> {
                if !progress(i as f32 / n as f32) {
                    anyhow::bail!("cancelled");
                }
                let mut base = format!("{stem}_{}", safe(name));
                let mut k = 2;
                while !seen.insert(base.clone()) {
                    base = format!("{stem}_{}_{k}", safe(name));
                    k += 1;
                }
                mm_io::write_mesh(&dir.join(format!("{base}.{ext}")), &source.mesh(flip), &opts)
            };
            match sources {
                ExportParts::Groups(mesh, names) => {
                    let mut groups: std::collections::BTreeMap<i32, Vec<u32>> = Default::default();
                    for t in mesh.triangle_ids() {
                        groups.entry(mesh.tri_group(t)).or_default().push(t);
                    }
                    for (i, (g, tris)) in groups.into_iter().enumerate() {
                        let name = names.get(&g).cloned().unwrap_or_else(|| format!("group_{g}"));
                        write(i, &name, ExportSource::Whole(std::sync::Arc::new(mesh.submesh(&tris).0)))?;
                    }
                }
                ExportParts::Objects(parts) => {
                    for (i, (name, m)) in parts.into_iter().enumerate() {
                        write(i, &name, ExportSource::Whole(m))?;
                    }
                }
            }
            Ok(format!("Exported {n} files to {}", dir.display()))
        }));
        Ok(msg)
    }

    // ------------------------------------------------------------------
    // Tools
    // ------------------------------------------------------------------

    /// Switch tools the Meshmixer way: leaving a tool for another one (hotkey, palette, menu)
    /// *accepts* it; only Cancel / Esc discards. A tool that is still computing finishes
    /// first and the switch happens when its result has been accepted.
    fn switch_to(&mut self, to: SwitchTo, ctx: &egui::Context, vp: Viewport) {
        // Switching to Select returns to an open Select tool underneath the current sub-tool.
        let to_select = matches!(&to, SwitchTo::ClearSelect) || matches!(&to, SwitchTo::Tool(n) if n == "select");
        let keep = if to_select { self.tools.iter().position(|t| t.name() == "Select").map(|i| i + 1).unwrap_or(0) } else { 0 };
        if !self.accept_tools(keep, ctx, vp) {
            if self.pending_switch.is_none() {
                self.status = "Finishing the current tool before switching…".into();
            }
            self.pending_switch = Some(to);
            return;
        }
        self.pending_switch = None;
        match to {
            SwitchTo::Tool(name) => {
                self.category = None;
                if keep == 0 {
                    // `drop:<part name>` opens Drop Part with a library part (from the library
                    // as its files are now: loaded here if need be, for agents' calls).
                    let part = name.strip_prefix("drop:").and_then(|p| self.current_library().find(p));
                    let mut cx = self.tool_cx(ctx, vp);
                    let tool: Option<Box<dyn Tool>> = match part {
                        Some(p) => Some(Box::new(tools::drop_part::DropPartTool::new(&mut cx, p))),
                        None => make_tool(&name, &mut cx),
                    };
                    if let Some(t) = tool {
                        self.tools.push(t);
                        self.share_brush();
                    }
                }
            }
            SwitchTo::ClearSelect => {
                if self.scene.active().is_some_and(|o| !o.selection.is_empty()) {
                    commands::selection_command(&mut self.cmd_cx(), commands::SelectCmd::Clear);
                }
                self.switch_to(SwitchTo::Tool("select".into()), ctx, vp);
            }
            SwitchTo::Category(c) => {
                if c == Some(Category::Meshmix) {
                    // Pick up parts saved since the last look (in the background; the parts
                    // shown last time stay until then).
                    self.load_library(ctx);
                }
                self.category = c;
            }
        }
    }

    /// Start loading the parts library in the background, unless it is being loaded already.
    fn load_library(&mut self, ctx: &egui::Context) {
        if self.library_job.is_none() {
            let changes = crate::parts::library_changes();
            self.library_job = Some((changes, Job::spawn("Loading parts", ctx.clone(), |_| Ok(crate::parts::Library::load()))));
        }
    }

    /// Reload: the library as its files are now. A load under way may have read the folders
    /// before files were copied into them, so its result is then out of date (and loaded
    /// again while the panel is shown).
    fn reload_library(&mut self, ctx: &egui::Context) {
        self.library_failed = None;
        match &mut self.library_job {
            Some((changes, _)) => *changes = u64::MAX,
            None => self.load_library(ctx),
        }
    }

    /// Whether the shown panel should load the library, its files being at `changes`: it is
    /// missing or out of date, and a load has not failed at that count.
    fn library_load_due(&self, changes: u64) -> bool {
        !(self.library.is_some() && self.library_changes == changes) && self.library_failed != Some(changes)
    }

    /// Whether `library` is there and was loaded after the last change of its files.
    fn library_is_current(&self) -> bool {
        self.library.is_some() && self.library_changes == crate::parts::library_changes()
    }

    /// The parts library as its files are now, loaded here (on this thread) if need be.
    fn current_library(&mut self) -> &crate::parts::Library {
        if !self.library_is_current() {
            self.library_changes = crate::parts::library_changes();
            self.library = Some(crate::parts::Library::load());
        }
        self.library.get_or_insert_with(crate::parts::Library::load)
    }

    /// Accept and close tools from the top of the stack until `keep` remain. Returns false
    /// while a tool is still computing (it is accepted once its result is ready).
    fn accept_tools(&mut self, keep: usize, ctx: &egui::Context, vp: Viewport) -> bool {
        while self.tools.len() > keep {
            let mut t = self.tools.pop().unwrap();
            let mut cx = self.tool_cx(ctx, vp);
            // `accept` may start computing an out-of-date result; wait for it in that case.
            if t.busy().is_some() || (!t.accept(&mut cx) && t.busy().is_some()) {
                self.tools.push(t);
                return false;
            }
            t.on_close(&mut cx);
        }
        true
    }

    /// Accept the open tools before a command that changes the scene (menu actions). False
    /// (with a status message) while a tool is still computing.
    fn accept_tools_for_command(&mut self, ctx: &egui::Context, vp: Viewport) -> bool {
        if self.accept_tools(0, ctx, vp) {
            return true;
        }
        self.status = "Wait for the current tool to finish (or cancel it with Esc).".into();
        false
    }

    /// The scene changed underneath the tools (undo/redo, object added/removed/merged):
    /// close tools whose previews would otherwise be committed against stale meshes.
    fn scene_changed(&mut self, ctx: &egui::Context, vp: Viewport) {
        let tools = std::mem::take(&mut self.tools);
        for mut t in tools {
            if t.survives_scene_change() {
                self.tools.push(t);
            } else {
                let mut cx = self.tool_cx(ctx, vp);
                t.on_close(&mut cx);
            }
        }
    }

    fn tool_cx<'a>(&'a mut self, ctx: &'a egui::Context, vp: Viewport) -> ToolCx<'a> {
        ToolCx { scene: &mut self.scene, history: &mut self.history, camera: &mut self.camera, status: &mut self.status, egui: ctx, vp, brush_scroll: self.view.brush_scroll }
    }

    fn cmd_cx(&mut self) -> CmdCx<'_> {
        CmdCx { scene: &mut self.scene, history: &mut self.history, status: &mut self.status }
    }

    fn tool_busy(&self) -> bool {
        self.tools.last().map(|t| t.busy().is_some()).unwrap_or(false)
    }

    fn undo(&mut self, ctx: &egui::Context) {
        if self.tool_busy() {
            return;
        }
        if let Some(l) = self.history.undo(&mut self.scene) {
            self.scene.after_restore();
            self.status = format!("Undo: {l}");
            self.scene_changed(ctx, self.vp_prev(ctx));
        }
    }
    fn redo(&mut self, ctx: &egui::Context) {
        if self.tool_busy() {
            return;
        }
        if let Some(l) = self.history.redo(&mut self.scene) {
            self.scene.after_restore();
            self.status = format!("Redo: {l}");
            self.scene_changed(ctx, self.vp_prev(ctx));
        }
    }

    fn vp_prev(&self, ctx: &egui::Context) -> Viewport {
        Viewport { rect: ctx.data(|d| d.get_temp::<egui::Rect>(egui::Id::NULL.with("vp_rect"))).unwrap_or(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0))) }
    }

    // ------------------------------------------------------------------
    // UI
    // ------------------------------------------------------------------

    fn menu_bar(&mut self, ui: &mut egui::Ui, vp: Viewport) {
        let ctx = ui.ctx().clone();
        egui::MenuBar::new().ui(ui, |ui| {
            ui.menu_button("File", |ui| {
                if ui.button("Open project…").clicked() {
                    ui.close();
                    self.open_project_dialog(&ctx);
                }
                if ui.button("Save project   ⌘S").clicked() {
                    ui.close();
                    self.save_project(&ctx, false);
                }
                if ui.button("Save project as…   ⇧⌘S").clicked() {
                    ui.close();
                    self.save_project(&ctx, true);
                }
                ui.separator();
                if ui.button("Import…   ⌘O").clicked() {
                    ui.close();
                    self.import_dialog(&ctx);
                }
                if ui.button("Import reference…").on_hover_text("A mesh shown see-through as a guide (e.g. the original segmentation behind an edited model): not picked in the view nor exported with the visible objects, until made editable").clicked() {
                    ui.close();
                    if let Some(files) = rfd::FileDialog::new().add_filter("Meshes", &["stl", "obj", "ply", "off", "3mf", "amf"]).pick_files() {
                        self.start_import_as(files, true, &ctx);
                    }
                }
                if ui.button("Import parts folder…").on_hover_text("Add a folder of mesh files to Meshmix as parts").clicked() {
                    ui.close();
                    // The library is loaded again when next shown (the folder list changed).
                    if let Some(msg) = crate::parts::import_parts_folder_dialog() {
                        self.status = msg;
                    }
                }
                if ui.button("Export active…").clicked() {
                    ui.close();
                    self.export_dialog(&ctx, ExportScope::Active);
                }
                if ui.button("Export all visible as one…").clicked() {
                    ui.close();
                    self.export_dialog(&ctx, ExportScope::AllVisible);
                }
                if ui.button("Export selected triangles…").clicked() {
                    ui.close();
                    self.export_dialog(&ctx, ExportScope::Selection);
                }
                if ui.button("Export one file per region…").on_hover_text("Each face group of the active object to its own file, named after the group (e.g. a vessel's wall and its inlet / outlet patches)").clicked() {
                    ui.close();
                    self.export_dialog(&ctx, ExportScope::PerGroup);
                }
                if ui.button("Export one file per object…").on_hover_text("Each visible object to its own file, named after it").clicked() {
                    ui.close();
                    self.export_dialog(&ctx, ExportScope::PerObject);
                }
                ui.menu_button("Add shape", |ui| {
                    // Meshmixer's Import Sphere / Plane (no bunny: Polymix ships no sample scans).
                    for shape in Shape::ALL {
                        if ui.button(shape.label()).clicked() {
                            ui.close();
                            self.run(MeshCommand::AddShape(shape), &ctx, vp);
                        }
                    }
                });
                ui.menu_button("Open recent", |ui| {
                    if self.recent.is_empty() {
                        ui.weak("(none)");
                    }
                    let mut open = None;
                    for r in &self.recent {
                        let name = r.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
                        if ui.button(name).on_hover_text(r.display().to_string()).clicked() {
                            open = Some(r.clone());
                            ui.close();
                        }
                    }
                    if let Some(r) = open {
                        self.start_import(vec![r], &ctx);
                    }
                });
                ui.checkbox(&mut self.export_ascii, "Write STL as ASCII");
                ui.separator();
                if ui.button("Preferences…   ⌘,").clicked() {
                    ui.close();
                    self.show_prefs = true;
                }
                ui.separator();
                if ui.button("Clear scene").clicked() {
                    ui.close();
                    self.run(MeshCommand::ClearScene, &ctx, vp);
                }
            });
            ui.menu_button("Edit", |ui| {
                let ul = self.history.undo_label().map(|s| format!("Undo {s}")).unwrap_or("Undo".into());
                if ui.add_enabled(self.history.can_undo(), egui::Button::new(format!("{ul}   ⌘Z"))).clicked() {
                    ui.close();
                    self.run(MeshCommand::Undo, &ctx, vp);
                }
                let rl = self.history.redo_label().map(|s| format!("Redo {s}")).unwrap_or("Redo".into());
                if ui.add_enabled(self.history.can_redo(), egui::Button::new(format!("{rl}   ⇧⌘Z"))).clicked() {
                    ui.close();
                    self.run(MeshCommand::Redo, &ctx, vp);
                }
                if ui.button("Copy   ⌘C").on_hover_text("Copy the open tool's readout (measurements, query) or the object's statistics as text").clicked() {
                    ui.close();
                    self.copy(&ctx);
                }
                if ui.add_enabled(self.history.can_undo() || self.history.can_redo(), egui::Button::new("Discard undo history")).on_hover_text("Free the memory held by undo steps").clicked() {
                    ui.close();
                    self.history.clear();
                    self.status = "Undo history discarded".into();
                }
                ui.separator();
                if ui.button("Duplicate object").clicked() {
                    ui.close();
                    self.run(MeshCommand::Duplicate, &ctx, vp);
                }
                if ui.button("Combine visible objects").clicked() {
                    ui.close();
                    self.run(MeshCommand::CombineVisible, &ctx, vp);
                }
                if ui.button("Separate shells").clicked() {
                    ui.close();
                    self.run(MeshCommand::SeparateShells, &ctx, vp);
                }
                ui.separator();
                let can_target = self.scene.objects.len() >= 2 && self.scene.active.is_some();
                if ui
                    .add_enabled(can_target, egui::Button::new("Set as target"))
                    .on_hover_text("Use the active object as the target surface (Attract / Align To Target, Project to Target)")
                    .clicked()
                {
                    ui.close();
                    let active = self.scene.active;
                    self.run(MeshCommand::SetTarget(active), &ctx, vp);
                }
                if ui.add_enabled(self.scene.target.is_some(), egui::Button::new("Clear target")).clicked() {
                    ui.close();
                    self.run(MeshCommand::SetTarget(None), &ctx, vp);
                }
            });
            ui.menu_button("View", |ui| {
                ui.checkbox(&mut self.view.wireframe, "Wireframe  (W)");
                ui.horizontal(|ui| {
                    for m in NormalMode::ALL {
                        ui.selectable_value(&mut self.view.normals, m, m.label());
                    }
                });
                ui.checkbox(&mut self.view.sharp_edges, "Sharp edges")
                    .on_hover_text("Keep hard corners crisp instead of averaging normals across them (CAD parts). Edges bending more than the crease angle are shaded sharp.");
                ui.add_enabled(self.view.sharp_edges, egui::Slider::new(&mut self.view.crease_deg, 5.0..=90.0).suffix("°").text("Crease angle"));
                ui.horizontal(|ui| {
                    for m in ColorMode::ALL {
                        ui.selectable_value(&mut self.view.colors, m, m.label());
                    }
                });
                ui.checkbox(&mut self.view.boundaries, "Show open boundaries");
                ui.checkbox(&mut self.camera.orthographic, "Orthographic");
                ui.checkbox(&mut self.camera.free_orbit, "Free rotation")
                    .on_hover_text("On: tumble freely in every direction (trackball). Off: turntable that keeps the up axis vertical.");
                ui.checkbox(&mut self.show_objects, "Object browser");
                ui.checkbox(&mut self.view.show_stats, "Statistics");
                let image = self.slices.source.as_ref().map(|s| s.name.clone());
                let mut show = self.slices.show;
                if ui.checkbox(&mut show, "Image slices").on_hover_text(match &image {
                    Some(n) => format!("{n} (open in the Segmentation tab) as three planes among the meshes"),
                    None => "The image open in the Segmentation tab as three planes among the meshes (open one there first)".into(),
                }).changed() {
                    ui.close();
                    self.run(MeshCommand::ImageSlices(show), &ctx, vp);
                }
                ui.separator();
                ui.label("Navigation").on_hover_text("Mouse buttons for tumble / pan / zoom (Meshmixer: Preferences → Navigation Mode). See Help → Controls.");
                ui.horizontal(|ui| {
                    for m in NavMode::ALL {
                        ui.selectable_value(&mut self.view.nav_mode, m, m.label());
                    }
                });
                if ui.button("Mouse buttons…").on_hover_text("Choose what each mouse button does (Preferences → General)").clicked() {
                    ui.close();
                    self.show_prefs = true;
                    self.prefs_tab = 0;
                }
                if ui.button("3D mouse…").on_hover_text("3Dconnexion SpaceMouse settings (Preferences → 3D mouse)").clicked() {
                    ui.close();
                    self.show_prefs = true;
                    self.prefs_tab = PREFS_TAB_3D_MOUSE;
                }
                ui.separator();
                ui.label("Up axis");
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.camera.up_axis, UpAxis::Y, "Y up");
                    ui.selectable_value(&mut self.camera.up_axis, UpAxis::Z, "Z up (medical/RAS)");
                });
                ui.separator();
                if ui.button("Focus on scene  (F)").clicked() {
                    ui.close();
                    self.run(MeshCommand::Focus, &ctx, vp);
                }
                if ui.button("Recenter at cursor  (Alt+click)").on_hover_text("Pivot the view on the surface point last under the mouse").clicked() {
                    ui.close();
                    if let Some(p) = self.last_pointer {
                        self.recenter_at(p, self.vp_prev(&ctx), ctx.input(|i| i.time));
                    }
                }
                if ui.add_enabled(self.view_history.can_back(), egui::Button::new("Previous view   ⌘[")).clicked() {
                    ui.close();
                    self.previous_view();
                }
                if ui.add_enabled(self.view_history.can_forward(), egui::Button::new("Next view   ⌘]")).clicked() {
                    ui.close();
                    self.next_view();
                }
                ui.checkbox(&mut self.view.show_grid, "Show grid");
                ui.horizontal(|ui| {
                    ui.label("Grid size");
                    let auto = grid_step(self.scene.bounds().diagonal(), 0.0);
                    ui.add(egui::DragValue::new(&mut self.view.grid_mm).range(0.0..=10_000.0).speed(0.1).suffix(" mm").custom_formatter(move |v, _| if v <= 0.0 { format!("auto ({auto:.3} mm)") } else { format!("{v} mm") }))
                        .on_hover_text("Spacing of the grid lines (every fifth is brighter); 0: chosen from the scene's size");
                    if self.view.grid_mm > 0.0 && ui.small_button("Auto").clicked() {
                        self.view.grid_mm = 0.0;
                    }
                });
                ui.checkbox(&mut self.view.show_view_cube, "Show view cube");
                ui.checkbox(&mut self.view.show_pivots, "Show pivots");
                ui.horizontal(|ui| {
                    ui.label("Theme");
                    let before = self.view.theme;
                    ui.selectable_value(&mut self.view.theme, ui_util::Theme::Dark, "Dark");
                    ui.selectable_value(&mut self.view.theme, ui_util::Theme::Light, "Light");
                    if self.view.theme != before {
                        ui_util::apply_theme(ui.ctx(), self.view.theme);
                    }
                });
                ui.checkbox(&mut self.view.transparent_target, "Transparent target").on_hover_text("Draw the target surface see-through");
                ui.checkbox(&mut self.view.xray, "X-ray").on_hover_text("Every object see-through, the parts seen edge-on (outlines, a vessel's walls) most opaque: shows what lies inside (a stent in a vessel, a lumen in a wall)");
                ui.menu_button("Standard views", |ui| {
                    for (k, (name, _, _)) in STANDARD_VIEWS.iter().enumerate() {
                        if ui.button(*name).clicked() {
                            ui.close();
                            self.run(MeshCommand::StandardView(k), &ctx, vp);
                        }
                    }
                });
            });
            ui.menu_button("Help", |ui| {
                if ui.button("Controls & hotkeys").clicked() {
                    ui.close();
                    self.show_help = true;
                }
            });
        });
    }

    fn palette(&mut self, ui: &mut egui::Ui, vp: Viewport) {
        let ctx = ui.ctx().clone();
        ui.add_space(6.0);
        use crate::icons::{self, Icon};
        let big = |ui: &mut egui::Ui, icon: Icon, label: &str, active: bool| {
            // Leading space leaves room for the icon on the left.
            let b = egui::Button::new(RichText::new(format!("       {label}")).size(13.0))
                .wrap_mode(egui::TextWrapMode::Extend)
                .min_size(egui::vec2(108.0, 30.0))
                .selected(active);
            let r = ui.add(b);
            let col = if active { ui.visuals().selection.stroke.color } else { ui.visuals().widgets.inactive.fg_stroke.color };
            icons::paint(ui.painter(), egui::Rect::from_center_size(r.rect.left_center() + egui::vec2(16.0, 0.0), egui::vec2(16.0, 16.0)), icon, col);
            r
        };
        if big(ui, Icon::Import, "Import", false).on_hover_text("⌘O").clicked() {
            self.import_dialog(&ctx);
        }
        let sel_active = self.tools.first().map(|t| t.name() == "Select").unwrap_or(false);
        if big(ui, Icon::Select, "Select", sel_active).on_hover_text("S").clicked() {
            self.switch_to(SwitchTo::Tool("select".into()), &ctx, vp);
        }
        for (cat, icon, label) in [
            (Category::Meshmix, Icon::Meshmix, "Meshmix"),
            (Category::Sculpt, Icon::Sculpt, "Sculpt"),
            (Category::Edit, Icon::Edit, "Edit"),
            (Category::Vessel, Icon::Vessel, "Vessel"),
            (Category::Analysis, Icon::Analysis, "Analysis"),
            (Category::Shaders, Icon::Shaders, "Shaders"),
        ] {
            if big(ui, icon, label, self.category == Some(cat) && self.tools.is_empty()).clicked() {
                let to = if self.category == Some(cat) && self.tools.is_empty() { None } else { Some(cat) };
                self.switch_to(SwitchTo::Category(to), &ctx, vp);
            }
        }
        if big(ui, Icon::Export, "Export", false).on_hover_text("⌘E").clicked() {
            self.export_dialog(&ctx, if self.prefs.export_whole_scene { ExportScope::AllVisible } else { ExportScope::Active });
        }
    }

    fn category_panel(&mut self, ui: &mut egui::Ui, cat: Category, vp: Viewport) {
        let ctx = ui.ctx().clone();
        let has_obj = self.scene.active().is_some();
        let mut open: Option<&str> = None;
        let mut command: Option<MeshCommand> = None;
        match cat {
            Category::Meshmix => {
                // A library from before its files last changed (a part saved with Convert To,
                // a folder added or removed) is shown until the new one is loaded.
                if self.library_load_due(crate::parts::library_changes()) {
                    self.load_library(&ctx);
                }
                if self.library.is_none() && self.library_job.is_none() {
                    // The first load failed (it is in the log).
                    ui.label("The parts could not be loaded.");
                    if ui.button("Try again").clicked() {
                        self.reload_library(&ctx);
                    }
                } else {
                    let r = crate::parts::library_panel(ui, self.library.as_mut(), has_obj, &mut self.status);
                    if r.reload {
                        self.reload_library(&ctx);
                    }
                    if let Some(name) = r.drop {
                        self.switch_to(SwitchTo::Tool(format!("drop:{name}")), &ctx, vp);
                    }
                }
            }
            Category::Edit => {
                ui_util::section(ui, "Edit");
                ui.add_enabled_ui(has_obj, |ui| {
                    if ui.button("Transform  (T)").clicked() {
                        open = Some("transform");
                    }
                    if ui.button("Align").on_hover_text("Put a picked face on the ground / world origin or against another object").clicked() {
                        open = Some("align");
                    }
                    if ui.button("Align To Target").on_hover_text("Register the object to the target surface (iterative closest point)").clicked() {
                        open = Some("aligntotarget");
                    }
                    if ui.button("Plane Cut").clicked() {
                        open = Some("planecut");
                    }
                    if ui.button("Mirror").clicked() {
                        open = Some("mirror");
                    }
                    if ui.button("Make Solid").clicked() {
                        open = Some("makesolid");
                    }
                    if ui.button("Create Pivot").on_hover_text("Drop a point with a coordinate frame on the object (Transform can turn and scale about it)").clicked() {
                        open = Some("pivot");
                    }
                    if ui.button("Unwrap").on_hover_text("Compute UVs (texture coordinates) for OBJ export").clicked() {
                        open = Some("unwrap");
                    }
                    if ui.button("Set UVs").on_hover_text("UVs by planar, cylindrical, spherical or box projection (no unwrapping)").clicked() {
                        open = Some("setuv");
                    }
                    if ui.button("Make Pattern").on_hover_text("Turn the object into a frame of tubes, or perforate / fill it with tubes, balls or a lattice").clicked() {
                        open = Some("makepattern");
                    }
                    if ui.button("Remesh").clicked() {
                        open = Some("remesh");
                    }
                    if ui.button("Reduce").clicked() {
                        open = Some("reduce");
                    }
                    if ui.button("Smooth").clicked() {
                        open = Some("smooth");
                    }
                    if ui.button("Offset / Hollow").clicked() {
                        open = Some("offset");
                    }
                    if ui.button("Thicken").on_hover_text("A wall of a given thickness on one side of the surface, open ends kept open (to print a hollow vessel)").clicked() {
                        open = Some("thicken");
                    }
                    if ui.button("Boolean").clicked() {
                        open = Some("boolean");
                    }
                    if ui.button("Bridge boundaries").clicked() {
                        open = Some("bridge");
                    }
                    if ui.button("Add Tube").on_hover_text("A tube between two points on the surface (bypass, channel, handle)").clicked() {
                        open = Some("addtube");
                    }
                    if ui.button("Close Cracks").on_hover_text("Zip open boundaries that lie on top of each other").clicked() {
                        open = Some("closecracks");
                    }
                    if ui.button("Generate Face Groups").on_hover_text("Segment the mesh into face groups by crease or normal angle").clicked() {
                        open = Some("facegroups");
                    }
                    ui.separator();
                    if ui.button("Separate Shells  (Shift+Y)").clicked() {
                        command = Some(MeshCommand::SeparateShells);
                    }
                    if ui.button("Combine visible").clicked() {
                        command = Some(MeshCommand::CombineVisible);
                    }
                    if ui.button("Generate Complex").on_hover_text("Cells of the overlapping visible closed objects (A − B, A ∩ B, B − A, …), one face group each").clicked() {
                        command = Some(MeshCommand::GenerateComplex);
                    }
                    if ui.button("Filter Complex").on_hover_text("Keep cells of a complex by which inputs they are inside (A & B, A - B, n == 1, …)").clicked() {
                        open = Some("filtercomplex");
                    }
                    if ui.button("Split Complex").on_hover_text("One object per face group").clicked() {
                        command = Some(MeshCommand::SplitComplex);
                    }
                    if ui.button("Convex Decomposition").on_hover_text("Approximate the object by convex pieces (new object, one face group per piece)").clicked() {
                        open = Some("convex");
                    }
                    if ui.button("Duplicate").clicked() {
                        command = Some(MeshCommand::Duplicate);
                    }
                    if ui.button("Flip normals").clicked() {
                        command = Some(MeshCommand::FlipNormals);
                    }
                });
            }
            Category::Vessel => {
                ui_util::section(ui, "Vessel / CFD prep");
                ui.add_enabled_ui(has_obj, |ui| {
                    if ui.button("Trim branch (square cut)").on_hover_text("Click a vessel to cut it perpendicular to its axis and cap it").clicked() {
                        open = Some("trim");
                    }
                    if ui.button("Flow extensions").on_hover_text("Extrude straight extensions from open (cut) ends").clicked() {
                        open = Some("flowext");
                    }
                    if ui.button("Cap open ends").on_hover_text("Flat-cap every open boundary").clicked() {
                        open = Some("cap");
                    }
                    if ui.button("Thicken into a shell").on_hover_text("A wall of a given thickness around the lumen (or inside it), the cut ends kept open with flat rims: a hollow model to print").clicked() {
                        open = Some("thicken");
                    }
                    if ui.button("Measure diameter").clicked() {
                        open = Some("measure");
                    }
                    if ui.button("Centerline").on_hover_text("Vessel centerline tree with branch lengths and diameters; export CSV/VTK").clicked() {
                        open = Some("centerline");
                    }
                    if ui.button("Virtual stent").on_hover_text("Place a bare or covered stent between two cross-sections and reshape the vessel around it").clicked() {
                        open = Some("stent");
                    }
                    if ui.button("Boundary patches / CFD export").on_hover_text("Name wall/inlet/outlet patches and export multi-solid STL").clicked() {
                        open = Some("patches");
                    }
                    if ui.button("Check for CFD").on_hover_text("Watertight, manifold, self-intersections, triangle quality, patches").clicked() {
                        open = Some("cfdcheck");
                    }
                });
                ui.small("Typical workflow: Inspector → Separate shells (delete fragments) → Smooth → Trim branches → Flow extensions → Remesh → Check for CFD → Boundary patches / export.");
            }
            Category::Analysis => {
                ui_util::section(ui, "Analysis");
                ui.add_enabled_ui(has_obj, |ui| {
                    if ui.button("Inspector  (I)").clicked() {
                        open = Some("inspector");
                    }
                    if ui.button("Measure").clicked() {
                        open = Some("measure");
                    }
                    if ui.button("Mesh Query").on_hover_text("Colour by normal, position, triangle quality or curvature; click for triangle info").clicked() {
                        open = Some("meshquery");
                    }
                    if ui.button("Clearance").on_hover_text("Minimum distance to another object, coloured by distance").clicked() {
                        open = Some("clearance");
                    }
                    if ui.button("Units / Dimensions").clicked() {
                        open = Some("units");
                    }
                    if ui.button("Check for CFD").clicked() {
                        open = Some("cfdcheck");
                    }
                    if ui.button("Deviation (compare to reference)").on_hover_text("Colour map of the distance to another object, e.g. the original segmentation").clicked() {
                        open = Some("deviation");
                    }
                });
            }
            Category::Shaders => {
                ui_util::section(ui, "Shaders");
                ui.small("Click a material for the active object.");
                let current = self.scene.active().map(|o| o.material);
                let mut chosen = None;
                ui.add_enabled_ui(has_obj, |ui| {
                    egui::Grid::new("materials").num_columns(2).show(ui, |ui| {
                        for (i, m) in crate::materials::MATERIALS.iter().enumerate() {
                            let swatch = m.color.map(|c| Color32::from_rgb((c[0] * 255.0) as u8, (c[1] * 255.0) as u8, (c[2] * 255.0) as u8));
                            let text = match swatch {
                                Some(c) => RichText::new(format!("● {}", m.name)).color(c),
                                None => RichText::new(format!("○ {}", m.name)),
                            };
                            if ui.selectable_label(current == Some(i), text).clicked() {
                                chosen = Some(i);
                            }
                            if i % 2 == 1 {
                                ui.end_row();
                            }
                        }
                    });
                });
                let all = ui.add_enabled(has_obj, egui::Button::new("Apply to all objects")).clicked();
                if let Some(i) = chosen.or(all.then_some(current).flatten()) {
                    let m = crate::materials::MATERIALS[i];
                    let ids: Vec<ObjectId> = if all { self.scene.objects.iter().map(|o| o.id).collect() } else { self.scene.active.into_iter().collect() };
                    for id in ids {
                        if let Some(o) = self.scene.get_mut(id) {
                            let had_alpha = crate::materials::MATERIALS.get(o.material).and_then(|x| x.alpha).is_some();
                            o.material = i;
                            if let Some(c) = m.color {
                                o.color = Vec4::new(c[0], c[1], c[2], o.color.w);
                            }
                            match m.alpha {
                                Some(a) => o.color.w = a,
                                None if had_alpha => o.color.w = 1.0,
                                None => {}
                            }
                        }
                    }
                }
            }
            Category::Sculpt => {
                ui_util::section(ui, "Sculpt");
                ui.add_enabled_ui(has_obj, |ui| {
                    if ui.button("Volume brushes  (3)").on_hover_text("Refine the mesh as you sculpt").clicked() {
                        open = Some("sculptvolume");
                    }
                    if ui.button("Surface brushes  (2)").on_hover_text("Sculpt without changing the triangles").clicked() {
                        open = Some("sculpt");
                    }
                    if ui.button("Smoothing brush  (1)").clicked() {
                        open = Some("sculptsmooth");
                    }
                    if ui.button("Stamp  (4)").on_hover_text("Cut a stencil shape into the surface as a face group or hole").clicked() {
                        open = Some("stamp");
                    }
                });
            }
        }
        if let Some(name) = open {
            command = Some(MeshCommand::OpenTool(name.into()));
        }
        if let Some(c) = command {
            self.run(c, &ctx, vp);
        }
    }

    fn tool_panel(&mut self, ui: &mut egui::Ui, vp: Viewport) {
        let ctx = ui.ctx().clone();
        let Some(mut tool) = self.tools.pop() else { return };
        let outcome = {
            let mut cx = self.tool_cx(&ctx, vp);
            egui::ScrollArea::vertical().show(ui, |ui| tool.panel(ui, &mut cx)).inner
        };
        let request = tool.take_request();
        match outcome {
            ToolOutcome::Continue => self.tools.push(tool),
            ToolOutcome::Close => {
                let mut cx = self.tool_cx(&ctx, vp);
                tool.on_close(&mut cx);
            }
        }
        if let Some(sub) = request {
            self.tools.push(sub);
        }
    }

    /// The object browser's rows and the Statistics, in a scroll area showing `viewport` of
    /// them (content coordinates; `None`: all).
    fn objects_panel(&mut self, ui: &mut egui::Ui, viewport: Option<egui::Rect>) {
        let mut delete = None;
        let mut make_active = None;
        let mut solo = None;
        let mut show_all = false;
        let mut duplicate = None;
        let mut send = None;
        let (mut texture, mut untexture, mut toggle_reference) = (None, None, None);
        let mut new_target: Option<Option<ObjectId>> = None;
        let target = self.scene.target;
        let rename_id = egui::Id::NULL.with("renaming");
        // The object being renamed, its name before (Esc puts it back) and whether its field
        // has been given the keyboard focus.
        let mut renaming: Option<(ObjectId, String, bool)> = ui.ctx().data(|d| d.get_temp(rename_id));
        // With many objects (hundreds after Separate Shells) only the rows in view are drawn;
        // the others just take their room, as one block above and one below. Not while an
        // object is renamed, as its field may make its row taller; nor while AccessKit is on
        // (agents and screen readers find and use rows out of view through it) or a popup is
        // open (a row's context menu or colour picker stays open only while the row is drawn).
        let n = self.scene.objects.len();
        let gap = ui.spacing().item_spacing.y;
        let step = self.objects_row_h.map(|h| h + gap);
        let draw_all = renaming.is_some() || ui.ctx().has_accesskit_node(egui::accesskit_root_id()) || egui::Popup::is_any_open(ui.ctx());
        let rows = match (viewport, step) {
            (Some(vp), Some(step)) if !draw_all => {
                let first = ((vp.min.y / step).floor().max(0.0) as usize).saturating_sub(1).min(n);
                let last = ((vp.max.y / step).ceil().max(0.0) as usize + 1).clamp(first, n);
                first..last
            }
            _ => 0..n,
        };
        let step = step.unwrap_or(0.0);
        // Each row takes one id of this ui (its `push_id` scope), a block one: skip ahead by the
        // rest, so that what follows keeps its ids however far the list is scrolled.
        if rows.start > 0 {
            ui.allocate_space(egui::vec2(0.0, rows.start as f32 * step - gap));
            ui.skip_ahead_auto_ids(rows.start - 1);
        }
        let mut row_h: Option<f32> = None;
        for o in self.scene.objects[rows.clone()].iter_mut() {
            let mut renamed_row = false;
            let r = ui.push_id(o.id, |ui| ui.horizontal(|ui| {
                ui.checkbox(&mut o.visible, "").on_hover_text("Visible");
                let active = Some(o.id) == self.scene.active;
                if let Some((_, before, focused)) = renaming.as_mut().filter(|r| r.0 == o.id) {
                    let r = ui.add(egui::TextEdit::singleline(&mut o.name).desired_width(ui.available_width()));
                    // Focus once only: Enter, Esc, Tab or a click elsewhere then ends the rename
                    // (egui's focus, not `has_focus`, which is also false while another app is
                    // in front).
                    if !*focused {
                        r.request_focus();
                        *focused = true;
                    } else if !ui.memory(|m| m.has_focus(r.id)) {
                        if o.name.trim().is_empty() || ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                            o.name = std::mem::take(before);
                        } else {
                            o.name = o.name.trim().to_string();
                        }
                        renaming = None;
                    }
                    renamed_row = true;
                    return;
                }
                // The controls at the right first; the name takes the room left, cut short with
                // "…" when it does not fit (the whole name on hover).
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.small_button("x").on_hover_text("Delete object").clicked() {
                        delete = Some(o.id);
                    }
                    let mut c = [o.color.x, o.color.y, o.color.z];
                    if ui.color_edit_button_rgb(&mut c).changed() {
                        o.color = Vec4::new(c[0], c[1], c[2], o.color.w);
                    }
                    if target == Some(o.id) {
                        ui.label(RichText::new("◎").color(ui_util::ACCENT)).on_hover_text("Target surface (Attract / Align To Target, Project to Target)");
                    }
                    ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                        let label = RichText::new(&o.name).color(if active { ui_util::ACCENT } else { ui.visuals().text_color() });
                        let r = ui.add(egui::Button::selectable(active, label).truncate());
                        let r = hover_text_with(r, || format!("{}\n\nClick: make active · Double-click: rename · Right-click: more", o.name));
                        if r.clicked() {
                            make_active = Some(o.id);
                        }
                        if r.double_clicked() {
                            renaming = Some((o.id, o.name.clone(), false));
                        }
                        r.context_menu(|ui| {
                            if ui.button("Rename").clicked() {
                                renaming = Some((o.id, o.name.clone(), false));
                                ui.close();
                            }
                            if ui.button("Show only this").clicked() {
                                solo = Some(o.id);
                                ui.close();
                            }
                            if ui.button("Show all").clicked() {
                                show_all = true;
                                ui.close();
                            }
                            if ui.button("Duplicate").clicked() {
                                duplicate = Some(o.id);
                                ui.close();
                            }
                            if ui.button("Texture…").on_hover_text("An image (PNG, JPEG, BMP, TGA) shown on the object through its UVs (Edit → Unwrap makes them; OBJ files bring theirs)").clicked() {
                                texture = Some(o.id);
                                ui.close();
                            }
                            if ui.button(if o.reference { "Make editable" } else { "Make reference" }).on_hover_text("A reference is shown see-through as a guide, not picked in the view nor exported with the visible objects").clicked() {
                                toggle_reference = Some(o.id);
                                ui.close();
                            }
                            if o.texture.is_some() && ui.button("Remove texture").clicked() {
                                untexture = Some(o.id);
                                ui.close();
                            }
                            if ui.button("Send to Segmentation").on_hover_text("Its inside becomes a segment of the volume shown in the Segmentation tab (a closed mesh, in the volume's space: LPS as from Make mesh or 3D Slicer, or RAS)").clicked() {
                                send = Some(o.id);
                                ui.close();
                            }
                            if target == Some(o.id) {
                                if ui.button("Clear target").clicked() {
                                    new_target = Some(None);
                                    ui.close();
                                }
                            } else if ui.button("Set as target").on_hover_text("Target surface for Attract / Align To Target and Project to Target").clicked() {
                                new_target = Some(Some(o.id));
                                ui.close();
                            }
                            if ui.button("Delete").clicked() {
                                delete = Some(o.id);
                                ui.close();
                            }
                        });
                    });
                });
            }))
            .inner
            .response
            .rect
            .height();
            if !renamed_row {
                row_h = Some(row_h.map_or(r, |h| h.max(r)));
            }
        }
        if rows.end < n {
            ui.allocate_space(egui::vec2(0.0, (n - rows.end) as f32 * step - gap));
            ui.skip_ahead_auto_ids(n - rows.end - 1);
        }
        if row_h.is_some() {
            self.objects_row_h = row_h;
        }
        if let Some(id) = make_active {
            self.scene.active = Some(id);
        }
        ui.ctx().data_mut(|d| match renaming {
            Some(r) => {
                d.insert_temp(rename_id, r);
            }
            None => d.remove::<(ObjectId, String, bool)>(rename_id),
        });
        if let Some(id) = solo {
            for o in self.scene.objects.iter_mut() {
                o.visible = o.id == id;
            }
        }
        if show_all {
            for o in self.scene.objects.iter_mut() {
                o.visible = true;
            }
        }
        let ctx = ui.ctx().clone();
        if let Some(t) = new_target {
            commands::set_target(&mut self.cmd_cx(), t);
        }
        if let Some(id) = send {
            let vp = self.vp_prev(&ctx);
            self.run(MeshCommand::SendToSegmentation(id), &ctx, vp);
        }
        if let Some(id) = texture {
            if let Some(path) = rfd::FileDialog::new().set_title("Texture image").add_filter("Image", &["png", "jpg", "jpeg", "bmp", "tga"]).pick_file() {
                let vp = self.vp_prev(&ctx);
                self.run(MeshCommand::SetTexture(id, Some(path)), &ctx, vp);
            }
        }
        if let Some(id) = toggle_reference {
            let vp = self.vp_prev(&ctx);
            let on = !self.scene.get(id).is_some_and(|o| o.reference);
            self.run(MeshCommand::SetReference(id, on), &ctx, vp);
        }
        if let Some(id) = untexture {
            let vp = self.vp_prev(&ctx);
            self.run(MeshCommand::SetTexture(id, None), &ctx, vp);
        }
        if let Some(id) = duplicate {
            if self.accept_tools_for_command(&ctx, self.vp_prev(&ctx)) {
                self.scene.active = Some(id);
                commands::duplicate_object(&mut self.cmd_cx());
            }
        }
        if let Some(id) = delete {
            if self.accept_tools_for_command(&ctx, self.vp_prev(&ctx)) {
                commands::delete_object(&mut self.cmd_cx(), id);
                self.scene_changed(&ctx, self.vp_prev(&ctx));
            }
        }
        if self.scene.objects.is_empty() {
            ui.weak("(empty scene)");
        }
        if self.view.show_stats {
            if let Some(o) = self.scene.active() {
                ui_util::section(ui, "Active object");
                let m = &o.mesh;
                let s = self.stats_cache.get(o);
                let e = s.extents;
                egui::Grid::new("stats").num_columns(2).spacing([10.0, 2.0]).show(ui, |ui| {
                    ui.label("Triangles");
                    ui.label(ui_util::format_count(m.triangle_count()));
                    ui.end_row();
                    ui.label("Vertices");
                    ui.label(ui_util::format_count(m.vertex_count()));
                    ui.end_row();
                    ui.label("Size");
                    ui.label(format!("{:.2} × {:.2} × {:.2}", e.x, e.y, e.z));
                    ui.end_row();
                    let open = s.open;
                    ui.label("Open edges");
                    ui.label(if open == 0 { RichText::new("0 (closed)").color(Color32::from_rgb(120, 200, 120)) } else { RichText::new(open.to_string()).color(Color32::from_rgb(240, 170, 90)) });
                    ui.end_row();
                    if let Some(v) = s.volume {
                        ui.label("Volume");
                        ui.label(format!("{v:.3}"));
                        ui.end_row();
                    }
                    let c = s.center_of_mass;
                    ui.label("Centre of mass").on_hover_text(if open == 0 { "Of the enclosed solid (uniform density)" } else { "Of the surface (an open mesh encloses no solid)" });
                    ui.label(format!("{:.2}, {:.2}, {:.2}", c.x, c.y, c.z));
                    ui.end_row();
                    ui.label("Area");
                    ui.label(format!("{:.3}", s.area));
                    ui.end_row();
                    ui.label("Selected");
                    ui.label(ui_util::format_count(o.selection.len()));
                    ui.end_row();
                });
            }
        }
    }

    fn hotbox_ui(&mut self, ctx: &egui::Context) {
        let Some(pos) = self.hotbox else { return };
        let mut close = false;
        egui::Area::new(egui::Id::NULL.with("hotbox")).fixed_pos(pos - egui::vec2(150.0, 90.0)).order(egui::Order::Foreground).show(ctx, |ui| {
            egui::Frame::popup(ui.style()).show(ui, |ui| {
                ui.set_width(300.0);
                ui.strong("Camera");
                ui.horizontal(|ui| {
                    // Drag these buttons to move the camera (handy on a trackpad).
                    let h = self.vp_prev(ctx).rect.height() as f64;
                    let drag = |ui: &mut egui::Ui, label: &str, hint: &str| ui.add(egui::Button::new(label).sense(egui::Sense::drag())).on_hover_text(hint);
                    let r = drag(ui, "⟲ Tumble", "Drag to tumble");
                    if r.dragged() {
                        let d = r.drag_delta();
                        let pivot = self.center_of_mass().unwrap_or(self.camera.target);
                        self.camera.tumble_about(d.x as f64, d.y as f64, pivot);
                    }
                    let r = drag(ui, "↔ Pan", "Drag to pan");
                    if r.dragged() {
                        let d = r.drag_delta();
                        self.camera.pan(d.x as f64, d.y as f64, h);
                    }
                    let r = drag(ui, "🔍 Zoom", "Drag up / down to zoom");
                    if r.dragged() {
                        self.camera.zoom((1.0 + r.drag_delta().y as f64 * 0.005).max(0.05));
                    }
                    if ui.color_edit_button_rgb(&mut self.view.background).on_hover_text("Background colour").secondary_clicked() {
                        self.view.background = renderer::DEFAULT_BACKGROUND;
                    }
                });
                ui.horizontal(|ui| {
                    ui.toggle_value(&mut self.camera.free_orbit, "Free").on_hover_text("Free rotation (trackball) instead of a turntable");
                    if ui.button("Snap").on_hover_text("Snap to nearest axis view").clicked() {
                        self.remember_view();
                        self.camera.snap_to_axis();
                    }
                    if ui.button("Focus").clicked() {
                        self.focus_scene();
                    }
                    ui.toggle_value(&mut self.camera.orthographic, "Ortho");
                    ui.selectable_value(&mut self.camera.up_axis, UpAxis::Y, "Y-up");
                    ui.selectable_value(&mut self.camera.up_axis, UpAxis::Z, "Z-up");
                });
                ui.strong("Mesh");
                ui.horizontal(|ui| {
                    for m in NormalMode::ALL {
                        ui.selectable_value(&mut self.view.normals, m, m.label().trim_end_matches(" normals"));
                    }
                    ui.toggle_value(&mut self.view.wireframe, "Wireframe");
                });
                ui.horizontal(|ui| {
                    ui.toggle_value(&mut self.view.boundaries, "Boundaries");
                    egui::ComboBox::from_id_salt("hotbox_colors").selected_text(self.view.colors.label()).show_ui(ui, |ui| {
                        for m in ColorMode::ALL {
                            ui.selectable_value(&mut self.view.colors, m, m.label());
                        }
                    });
                    if let Some(o) = self.scene.active_mut() {
                        let mut c = [o.color.x, o.color.y, o.color.z];
                        if ui.color_edit_button_rgb(&mut c).changed() {
                            o.color = Vec4::new(c[0], c[1], c[2], o.color.w);
                        }
                        let mut a = o.color.w;
                        if ui.add(egui::Slider::new(&mut a, 0.15..=1.0).text("opacity").show_value(false)).changed() {
                            o.color.w = a;
                        }
                    }
                });
                if let Some(t) = self.tools.last_mut() {
                    let has_size = t.brush_size().is_some();
                    let has_strength = t.brush_strength().is_some();
                    if has_size || has_strength {
                        ui.strong("Brushes");
                        ui.horizontal(|ui| {
                            if let Some(b) = t.brush_size() {
                                let mut pct = b.get();
                                if ui.add(egui::Slider::new(&mut pct, tools::BRUSH_SIZE_RANGE.0..=tools::BRUSH_SIZE_RANGE.1).logarithmic(true).text("size")).changed() {
                                    b.set(pct);
                                }
                            }
                            if let Some(s) = t.brush_strength() {
                                ui.add(egui::Slider::new(s, 0.0..=1.0).text("strength"));
                            }
                        });
                    }
                }
                if ui.small_button("close (Space)").clicked() {
                    close = true;
                }
            });
        });
        if close {
            self.hotbox = None;
        }
    }

    fn preferences_window(&mut self, ctx: &egui::Context) {
        if !self.show_prefs {
            return;
        }
        let mut open = true;
        egui::Window::new("Preferences").open(&mut open).resizable(false).collapsible(false).anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0)).show(ctx, |ui| {
            ui.horizontal(|ui| {
                for (i, t) in ["General", "View", "File", "Dialogs", "3D mouse"].iter().enumerate() {
                    ui.selectable_value(&mut self.prefs_tab, i, *t);
                }
            });
            ui.separator();
            match self.prefs_tab {
                PREFS_TAB_3D_MOUSE => self.space_settings_ui(ui),
                0 => {
                    egui::Grid::new("prefs_general").num_columns(2).spacing([16.0, 8.0]).show(ui, |ui| {
                        ui.label("Navigation mode");
                        ui.horizontal(|ui| {
                            for m in NavMode::ALL {
                                ui.selectable_value(&mut self.view.nav_mode, m, m.label());
                            }
                        });
                        ui.end_row();
                        ui.label("Mouse buttons").on_hover_text("What each middle / right drag does. Changing one switches to Custom. Alt+left-drag (orbit), Alt+Shift+left-drag (pan) and Alt+Cmd+left-drag (zoom) work in every mode; the wheel zooms.");
                        crate::navigation::bindings_ui(ui, &mut self.view.nav_mode, &mut self.view.nav_custom);
                        ui.end_row();
                        ui.label("Scroll zoom");
                        ui.vertical(|ui| {
                            ui.add(egui::Slider::new(&mut self.view.zoom_speed, 0.1..=4.0).logarithmic(true).suffix("×").text("speed"))
                                .on_hover_text("How far one wheel step zooms");
                            ui.horizontal(|ui| {
                                ui.label("toward");
                                ui.selectable_value(&mut self.view.zoom_to_cursor, true, "Point under the cursor").on_hover_text("The part of the model under the cursor stays under it while you zoom");
                                ui.selectable_value(&mut self.view.zoom_to_cursor, false, "Centre of the view").on_hover_text("Zoom into whatever is in the middle of the view");
                            });
                            ui.checkbox(&mut self.view.zoom_near, "Slow down near the surface")
                                .on_hover_text("Each wheel step covers a fraction of the distance to the surface you zoom toward: steps shrink as you get close, and you never zoom through the model. Off: steps are a fraction of the distance to the view's centre depth.");
                        });
                        ui.end_row();
                        ui.label("Brush size scroll").on_hover_text("Hold this key and scroll to resize Select and Sculpt brushes; the plain wheel always zooms. Control+scroll can be taken by macOS screen zoom (System Settings → Accessibility → Zoom).");
                        ui.horizontal(|ui| {
                            for m in BrushScroll::ALL {
                                ui.selectable_value(&mut self.view.brush_scroll, m, m.label());
                            }
                        });
                        ui.end_row();
                        ui.label("Mesh color mode");
                        ui.horizontal(|ui| {
                            for m in ColorMode::ALL {
                                ui.selectable_value(&mut self.view.colors, m, m.label());
                            }
                        });
                        ui.end_row();
                        ui.label("Mesh normal mode");
                        ui.horizontal(|ui| {
                            for m in NormalMode::ALL {
                                ui.selectable_value(&mut self.view.normals, m, m.label());
                            }
                        });
                        ui.end_row();
                        ui.label("Up axis");
                        ui.horizontal(|ui| {
                            ui.selectable_value(&mut self.camera.up_axis, UpAxis::Y, "Y up");
                            ui.selectable_value(&mut self.camera.up_axis, UpAxis::Z, "Z up (medical/RAS)");
                        });
                        ui.end_row();
                    });
                }
                1 => {
                    ui.checkbox(&mut self.camera.free_orbit, "Enable free rotate").on_hover_text("Trackball instead of turntable");
                    ui.checkbox(&mut self.prefs.reset_camera_on_import, "Reset camera on scene change").on_hover_text("Frame the scene after every import");
                    ui.checkbox(&mut self.view.show_view_cube, "Show view cube");
                    ui.checkbox(&mut self.show_objects, "Show object browser");
                    ui.checkbox(&mut self.view.show_grid, "Show grid");
                    ui.checkbox(&mut self.view.sharp_edges, "Sharp edges (crease-angle shading)");
                    ui.add_enabled(self.view.sharp_edges, egui::Slider::new(&mut self.view.crease_deg, 5.0..=90.0).suffix("°").text("crease angle"));
                    ui.horizontal(|ui| {
                        ui.label("Theme");
                        let before = self.view.theme;
                        ui.selectable_value(&mut self.view.theme, ui_util::Theme::Dark, "Dark");
                        ui.selectable_value(&mut self.view.theme, ui_util::Theme::Light, "Light");
                        if self.view.theme != before {
                            ui_util::apply_theme(ui.ctx(), self.view.theme);
                        }
                    });
                }
                3 => {
                    ui.label("Importing into a scene with objects");
                    ui.horizontal(|ui| {
                        ui.selectable_value(&mut self.prefs.import_mode, ImportMode::Append, "Always append");
                        ui.selectable_value(&mut self.prefs.import_mode, ImportMode::Replace, "Always replace");
                        ui.selectable_value(&mut self.prefs.import_mode, ImportMode::Ask, "Always ask");
                    });
                    ui.label("Models of unusual size (under 1 or over 10 000 units)");
                    ui.horizontal(|ui| {
                        let always = matches!(self.prefs.scale_mode, ScaleMode::Always(_));
                        if ui.selectable_label(always, "Always scale").clicked() && !always {
                            self.prefs.scale_mode = ScaleMode::Always(1000.0);
                        }
                        ui.selectable_value(&mut self.prefs.scale_mode, ScaleMode::Never, "Don't scale");
                        ui.selectable_value(&mut self.prefs.scale_mode, ScaleMode::Ask, "Always ask");
                    });
                    if let ScaleMode::Always(k) = &mut self.prefs.scale_mode {
                        ui.add(egui::DragValue::new(k).speed(1.0).range(1e-6..=1e6).prefix("factor "));
                    }
                }
                _ => {
                    ui.label("Export (⌘E)");
                    ui.horizontal(|ui| {
                        ui.selectable_value(&mut self.prefs.export_whole_scene, false, "Selection only (active object)");
                        ui.selectable_value(&mut self.prefs.export_whole_scene, true, "Whole scene");
                    });
                    ui.checkbox(&mut self.prefs.flip_zy, "Flip Z-Y axis on import/export")
                        .on_hover_text("Z-up files (most CAD and medical exports) stand upright in the Y-up view; exported files get their original axes back");
                    ui.checkbox(&mut self.export_ascii, "Write STL as ASCII");
                    ui.checkbox(&mut self.prefs.obj_export_groups, "OBJ: export face groups as groups");
                    ui.checkbox(&mut self.prefs.obj_import_groups, "Import OBJ: groups as face groups");
                    ui.add(egui::Slider::new(&mut self.prefs.autosave_minutes, 0..=30).text("autosave every (min)"))
                        .on_hover_text("Save a changed scene in the background for crash recovery (0 = off)");
                }
            }
        });
        if !open {
            self.show_prefs = false;
        }
    }

    fn help_window(&mut self, ctx: &egui::Context) {
        egui::Window::new("Controls & hotkeys").open(&mut self.show_help).resizable(false).show(ctx, |ui| {
            egui::Grid::new("help").num_columns(2).striped(true).show(ui, |ui| {
                ui.label(RichText::new(format!("{} navigation", self.view.nav_mode.label())).strong());
                ui.end_row();
                let brush_size = format!("Select / Sculpt: [ ], {} or Cmd/Ctrl+middle-drag", self.view.brush_scroll.scroll_label());
                let nav_rows = self.view.nav_mode.bindings(self.view.nav_custom).help_rows();
                for (k, v) in nav_rows.into_iter().chain([(brush_size, "Brush size")]).chain([
                    ("F", "Focus on scene"),
                    ("Space", "Hotbox (view settings)"),
                    ("W", "Toggle wireframe"),
                    ("S", "Select tool"),
                    ("I", "Inspector (when not selecting)"),
                    ("T", "Transform"),
                    ("Shift+Y", "Separate shells"),
                    ("Select: drag / Shift+drag", "Paint selection / deselect"),
                    ("Select: double-click", "Select connected part"),
                    ("Select: Cmd/Ctrl+right-drag (with a selection)", "Grow / shrink the selection"),
                    ("Sculpt: Cmd/Ctrl+right-drag", "Brush strength"),
                    ("Select: E / I / G / O / < >", "Connected / Invert / Groups / Optimize / Shrink-Grow"),
                    ("Select: X, F, D, Cmd+D", "Discard, Erase & Fill, Extrude, Offset"),
                    ("Select: R, Shift+R, Cmd+F, B, Y, J, Cmd+B", "Remesh, Reduce, Smooth, Smooth Boundary, Separate, Join, Bridge"),
                    ("Remesh / Reduce: D / F, T, G", "Density ±10, adaptive, preserve groups"),
                    ("Enter or A / Esc", "Accept / Cancel tool"),
                    ("Cmd+Z / Shift+Cmd+Z (Cmd+Y) / Alt+← Alt+→", "Undo / Redo"),
                    ("Cmd+Shift+O", "Object browser"),
                ].map(|(k, v)| (k.to_string(), v))) {
                    ui.label(RichText::new(k).monospace());
                    ui.label(v);
                    ui.end_row();
                }
            });
        });
    }

    /// Remember the current view before it changes (for Previous view).
    fn remember_view(&mut self) {
        self.view_history.record(self.camera.view_state(), self.camera.scene_radius);
    }

    /// Frame the visible objects (F, the Fit button).
    fn focus_scene(&mut self) {
        self.remember_view();
        let mut b = self.scene.bounds();
        // Image slices shown count too.
        if let Some(planes) = self.slices.frame().and_then(|f| crate::image_planes::bounds(&f.volume, &f.settings)) {
            b = if self.scene.objects.iter().any(|o| o.visible) { b.union(&planes) } else { planes };
        }
        self.camera.fit(&b, self.vp_aspect);
    }

    fn standard_view(&mut self, yaw: f64, pitch: f64) {
        self.remember_view();
        self.camera.set_view(yaw, pitch);
    }

    fn previous_view(&mut self) {
        if let Some(v) = self.view_history.back(self.camera.view_state(), self.camera.scene_radius) {
            self.view_anim = None;
            self.camera.set_view_state(v);
        }
    }

    fn next_view(&mut self) {
        if let Some(v) = self.view_history.forward() {
            self.view_anim = None;
            self.camera.set_view_state(v);
        }
    }

    /// Glide the view's pivot to the surface point under `p` (Meshmixer: Recenter At Cursor).
    /// Centre of mass of the active object (else the first visible one), cached per mesh.
    fn center_of_mass(&mut self) -> Option<DVec3> {
        let o = self.scene.active().filter(|o| o.visible).or_else(|| self.scene.objects.iter().find(|o| o.visible))?;
        let key = (o.id, o.mesh.shape_version());
        if let Some((id, v, c)) = self.com_cache {
            if (id, v) == key {
                return Some(c);
            }
        }
        let c = mm_core::measure::center_of_mass(&o.mesh);
        self.com_cache = Some((key.0, key.1, c));
        Some(c)
    }

    /// What a navigation drag started at `p` holds on to (orbits turn about it, pans move it
    /// with the cursor): the point of the model under the cursor, or, off the model, the
    /// object's centre of mass.
    fn nav_anchor_at(&mut self, p: Option<egui::Pos2>, vp: Viewport) -> DVec3 {
        if let Some(p) = p {
            let ray = self.camera.ray_from_ndc(vp.ndc(p), vp.aspect());
            if let Some((_, h)) = tools::pick_ray(&mut self.scene, &ray, None) {
                return h.point;
            }
        }
        self.center_of_mass().unwrap_or(self.camera.target)
    }

    fn recenter_at(&mut self, p: egui::Pos2, vp: Viewport, now: f64) {
        let ray = self.camera.ray_from_ndc(vp.ndc(p), vp.aspect());
        if let Some((_, h)) = tools::pick_ray(&mut self.scene, &ray, None) {
            self.remember_view();
            self.view_anim = Some((self.camera.target, h.point, now));
        }
    }

    /// Send a right-drag to the active tool.
    fn tool_secondary_click(&mut self, ctx: &egui::Context, vp: Viewport, pos: egui::Pos2) {
        if let Some(mut t) = self.tools.pop() {
            let mut cx = self.tool_cx(ctx, vp);
            t.secondary_click(pos, &mut cx);
            self.tools.push(t);
        }
    }

    fn tool_secondary(&mut self, ctx: &egui::Context, vp: Viewport, phase: DragPhase, pos: egui::Pos2, total: egui::Vec2) {
        if let Some(mut t) = self.tools.pop() {
            let mut cx = self.tool_cx(ctx, vp);
            t.secondary_drag(phase, pos, total, &mut cx);
            self.tools.push(t);
        }
    }

    /// Offer the wheel to the active tool; true if it used it.
    fn tool_wheel(&mut self, ctx: &egui::Context, vp: Viewport, delta: f32) -> bool {
        let Some(mut t) = self.tools.pop() else { return false };
        let used = {
            let mut cx = self.tool_cx(ctx, vp);
            t.wheel(delta, &mut cx)
        };
        self.tools.push(t);
        used
    }

    /// Apply 3D mouse motion and buttons.
    fn space_mouse(&mut self, ctx: &egui::Context, vp: Viewport) {
        let (motion, pressed) = self.space.poll();
        if !self.space_cfg.enabled {
            return;
        }
        let dt = ctx.input(|i| i.stable_dt).clamp(0.001, 0.05) as f64;
        let moving = !motion.is_zero();
        if moving && !self.space_moving {
            self.remember_view();
            // A move turns about one centre, chosen as it starts.
            self.space_centre = Some(self.space_move_centre(vp));
        }
        self.space_moving = moving;
        let pivot = self.space_centre.unwrap_or(self.camera.target);
        if spacemouse::apply(&mut self.camera, motion, &self.space_cfg, dt, vp.rect.height() as f64, pivot) {
            // Keep animating while the cap is deflected.
            ctx.request_repaint();
        }
        if moving && self.space_cfg.show_centre && !self.space_cfg.lock_rotation {
            self.space_marker = Some((pivot, std::time::Instant::now()));
        }
        // Capturing a button for the settings stops when they are put away.
        if !(self.show_prefs && self.prefs_tab == PREFS_TAB_3D_MOUSE) {
            self.space_capture = false;
            self.space_key_capture = None;
        }
        for b in pressed {
            self.space_button(b, ctx, vp);
        }
    }

    /// What a 3D mouse move starting now turns about (see [`spacemouse::Centre`]).
    fn space_move_centre(&mut self, vp: Viewport) -> DVec3 {
        let in_view = match self.space_cfg.centre {
            spacemouse::Centre::View => self.centre_in_view(vp),
            spacemouse::Centre::Object => None,
        };
        in_view.or_else(|| self.center_of_mass()).unwrap_or(self.camera.target)
    }

    /// What is in view to turn about: `None` while the visible objects are all in view (their
    /// centre of mass then). Zoomed in: of the vertices seen within a circle in the middle of
    /// the view, half the field of view across, the one nearest the camera, so a near part a
    /// little off the middle wins over what lies far behind the middle; with none in the
    /// circle, the vertex seen nearest the middle.
    fn centre_in_view(&mut self, vp: Viewport) -> Option<DVec3> {
        use rayon::prelude::*;
        let aspect = vp.aspect();
        let b = self.scene.bounds();
        if b.is_empty() {
            return None;
        }
        let corner = |k: usize| DVec3::new([b.min.x, b.max.x][k & 1], [b.min.y, b.max.y][(k >> 1) & 1], [b.min.z, b.max.z][k >> 2]);
        let shown = |p: DVec3| self.camera.project(p, aspect).is_some_and(|n| n.x.abs() <= 1.0 && n.y.abs() <= 1.0 && (0.0..=1.0).contains(&n.z));
        if (0..8).all(|k| shown(corner(k))) {
            return None;
        }
        // Screen offsets from the middle of the view, in points.
        let (w, h) = (vp.rect.width().max(1.0) as f64, vp.rect.height().max(1.0) as f64);
        let (eye, fwd, right, up) = (self.camera.eye(), self.camera.forward(), self.camera.right(), self.camera.up());
        let (ortho, per_point) = (self.camera.orthographic, self.camera.pixel_size_at_target(h));
        let focal = h / 2.0 / (self.camera.fov_y / 2.0).tan();
        let radius = CENTRE_CIRCLE * 0.5 * w.min(h);
        // Per vertex: (distance from the eye, in the circle) and (squared offset, in view);
        // the smallest of each over every vertex.
        type Best = (Option<(f64, DVec3)>, Option<(f64, DVec3)>);
        let least = |a: Option<(f64, DVec3)>, b: Option<(f64, DVec3)>| match (a, b) {
            (Some(x), Some(y)) => Some(if y.0 < x.0 { y } else { x }),
            (x, y) => x.or(y),
        };
        let mut best: Best = (None, None);
        for o in self.scene.objects.iter().filter(|o| o.visible && !o.reference) {
            let m = &o.mesh;
            let found: Best = m
                .positions()
                .par_iter()
                .enumerate()
                .filter(|(i, _)| m.is_vertex(*i as u32))
                .map(|(_, &p)| {
                    let v = p - eye;
                    let z = v.dot(fwd);
                    if z <= 1e-9 && !ortho {
                        return (None, None);
                    }
                    let scale = if ortho { 1.0 / per_point } else { focal / z };
                    let (sx, sy) = (v.dot(right) * scale, v.dot(up) * scale);
                    if sx.abs() > w / 2.0 || sy.abs() > h / 2.0 {
                        return (None, None);
                    }
                    let off = sx * sx + sy * sy;
                    ((off <= radius * radius).then(|| (v.length(), p)), Some((off, p)))
                })
                .reduce(|| (None, None), |a, b| (least(a.0, b.0), least(a.1, b.1)));
            best = (least(best.0, found.0), least(best.1, found.1));
        }
        best.0.or(best.1).map(|(_, p)| p)
    }

    /// Every brush tool open takes the one brush size: none keeps a size of its own.
    fn share_brush(&mut self) {
        for t in &mut self.tools {
            t.share_brush(&self.brush);
        }
    }

    /// Hand the app's input hook the buttons that stand for held modifier keys.
    fn sync_held_keys(&self) {
        let keys = self.space_buttons.iter().filter_map(|(b, a)| if let spacemouse::ButtonAction::Hold(k) = a { Some((*b, *k)) } else { None });
        spacemouse::set_held_keys(if self.space_cfg.enabled { keys.collect() } else { Vec::new() });
    }

    /// 3D mouse button `b` was pressed: its action, or, while the settings wait for a button,
    /// it joins the list there.
    fn space_button(&mut self, b: u32, ctx: &egui::Context, vp: Viewport) {
        self.space_last_button = Some(b);
        if self.space_capture {
            if !self.space_buttons.iter().any(|(x, _)| *x == b) {
                self.space_buttons.push((b, spacemouse::ButtonAction::Nothing));
                self.space_buttons.sort_by_key(|(x, _)| *x);
            }
            self.space_capture = false;
            return;
        }
        let Some(action) = self.space_buttons.iter().find(|(x, _)| *x == b).map(|(_, a)| a.clone()) else { return };
        use spacemouse::ButtonAction as A;
        // The same commands as the menus and hotkeys.
        let cmd = match action {
            A::Hotbox => {
                self.hotbox = if self.hotbox.is_some() { None } else { Some(vp.rect.center()) };
                return;
            }
            // Held keys act while held, in the app's input (see `spacemouse::with_held_keys`).
            A::Nothing | A::Hold(_) => return,
            // The key goes into the app's input, so it does what it does wherever it is pressed.
            A::Key { key, modifiers } => {
                spacemouse::press_key(key, modifiers);
                self.status = format!("3D mouse button {b}: key {}", spacemouse::key_label(key, modifiers));
                return;
            }
            // Select as the S key opens it: with nothing selected, again and again.
            A::Tool(name) if name == "select" && self.scene.active().is_some() => {
                self.switch_to(SwitchTo::ClearSelect, ctx, vp);
                let done = if self.pending_switch.is_some() { "Select opens when the open tool has finished computing" } else { "Select, nothing selected" };
                self.status = format!("3D mouse button {b}: {done}");
                return;
            }
            A::Fit => MeshCommand::Focus,
            A::View(k) => MeshCommand::StandardView(k),
            A::Undo => MeshCommand::Undo,
            A::Redo => MeshCommand::Redo,
            A::Accept => MeshCommand::AcceptTool,
            A::Cancel => MeshCommand::CancelTool,
            A::Tool(name) => MeshCommand::OpenTool(name),
        };
        let (Ok(m) | Err(m)) = self.execute(cmd, ctx, vp);
        self.status = format!("3D mouse button {b}: {m}");
    }

    /// Preferences → 3D mouse: connection, orbit / pan / zoom speeds, locks, inverted axes and
    /// a live readout of the cap.
    fn space_settings_ui(&mut self, ui: &mut egui::Ui) {
        match &self.space.status {
            spacemouse::Status::Ready { device: Some((vendor, product)) } => {
                ui.label(format!("Connected · device {vendor:04x}:{product:04x}"));
            }
            spacemouse::Status::Ready { device: None } => {
                ui.label("3Dconnexion driver ready · no device connected");
            }
            spacemouse::Status::DriverMissing => {
                ui.label("3Dconnexion driver (3DxWare) not installed.");
            }
            spacemouse::Status::Unsupported => {
                ui.label("3D mice are not supported on this platform yet.");
            }
            spacemouse::Status::Failed(e) => {
                ui.colored_label(egui::Color32::from_rgb(220, 120, 90), format!("Could not connect: {e}"));
            }
        }
        let c = &mut self.space_cfg;
        ui.checkbox(&mut c.enabled, "Enabled");
        ui.small("Object mode: move the cap the way you want the model to move.");
        ui_util::section(ui, "Speed");
        // One row: name, multiplier slider, and the rate it gives at full deflection.
        fn speed_row(ui: &mut egui::Ui, label: &str, hint: &str, v: &mut f32, rate: impl Fn(f64) -> String) {
            ui.label(label).on_hover_text(hint);
            ui.add(egui::Slider::new(&mut *v, spacemouse::SPEED_RANGE).logarithmic(true).suffix("×")).on_hover_text(hint);
            ui.weak(rate(*v as f64)).on_hover_text("At full deflection");
            ui.end_row();
        }
        egui::Grid::new("space_speeds").num_columns(3).show(ui, |ui| {
            speed_row(ui, "Orbit", "Tilting, spinning and rolling the cap", &mut c.orbit_speed, |v| format!("{:.0}°/s", (spacemouse::ORBIT_RATE * v).to_degrees()));
            speed_row(ui, "Pan", "Pushing the cap left, right, up or down", &mut c.pan_speed, |v| format!("{:.1} view heights/s", spacemouse::PAN_RATE * v));
            speed_row(ui, "Zoom", "Pushing the cap in or pulling it out", &mut c.zoom_speed, |v| {
                format!("halves the distance in {:.2} s", std::f64::consts::LN_2 / (spacemouse::ZOOM_RATE * v))
            });
        });
        if ui.small_button("Reset speeds").clicked() {
            (c.orbit_speed, c.pan_speed, c.zoom_speed) = (1.0, 1.0, 1.0);
        }
        ui.horizontal(|ui| {
            ui.checkbox(&mut c.lock_rotation, "Lock rotation");
            ui.checkbox(&mut c.lock_translation, "Lock pan/zoom");
        });
        ui_util::section(ui, "Centre of rotation");
        ui.radio_value(&mut c.centre, spacemouse::Centre::View, "What you look at")
            .on_hover_text("Chosen again as each move starts, as in Fusion 360: the object's centre while all of it is in view; zoomed in, the surface at the middle of the view, or the visible surface nearest it");
        ui.radio_value(&mut c.centre, spacemouse::Centre::Object, "The object's centre (centre of mass)");
        ui.checkbox(&mut c.show_centre, "Show it while moving").on_hover_text("A small green ball where the view turns about, while the 3D mouse moves it");
        ui_util::section(ui, "Invert");
        ui.horizontal_wrapped(|ui| {
            for (i, name) in spacemouse::Settings::AXIS_NAMES.iter().enumerate() {
                ui.checkbox(&mut c.invert[i], *name);
            }
        });
        let (axis, buttons) = self.space.raw();
        ui_util::section(ui, "Buttons");
        ui.small("Polymix takes the device's buttons over from 3DxWare, so the button settings in 3Dconnexion's own app do not apply here: give the buttons actions below.");
        let pressed = |b: u32| b < 32 && buttons & (1 << b) != 0;
        // The key for a button: the next one typed, with the modifiers held.
        if let Some(b) = self.space_key_capture {
            let typed = ui.input(|i| i.events.iter().find_map(|e| if let egui::Event::Key { key, pressed: true, modifiers, .. } = e { Some((*key, *modifiers)) } else { None }));
            if let Some((key, modifiers)) = typed {
                if let Some((_, a)) = self.space_buttons.iter_mut().find(|(x, _)| *x == b) {
                    *a = spacemouse::ButtonAction::Key { key, modifiers };
                }
                self.space_key_capture = None;
            }
        }
        let mut remove = None;
        let mut capture = None;
        egui::Grid::new("space_buttons").num_columns(3).show(ui, |ui| {
            for (k, (b, action)) in self.space_buttons.iter_mut().enumerate() {
                let name = egui::RichText::new(format!("Button {b}"));
                ui.label(if pressed(*b) { name.strong().color(ui.visuals().selection.bg_fill) } else { name });
                if self.space_key_capture == Some(*b) {
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("Type the key…").color(ui.visuals().selection.bg_fill));
                        if ui.small_button("Cancel").clicked() {
                            capture = Some(None);
                        }
                    });
                } else if button_action_menu(ui, *b, action) {
                    capture = Some(Some(*b));
                }
                if ui.small_button("×").on_hover_text("Remove this button").clicked() {
                    remove = Some(k);
                }
                ui.end_row();
            }
        });
        if let Some(c) = capture {
            self.space_key_capture = c;
        }
        if let Some(k) = remove {
            self.space_buttons.remove(k);
        }
        ui.horizontal(|ui| {
            let label = if self.space_capture { "Press a button on the 3D mouse…" } else { "+ Add a button" };
            if ui.selectable_label(self.space_capture, label).on_hover_text("Then press the button on the 3D mouse: it is added to the list").clicked() {
                self.space_capture = !self.space_capture;
            }
            if ui.small_button("Defaults").on_hover_text("Menu: Hotbox, Fit: fit the view, Top / Right / Front: views").clicked() {
                self.space_buttons = spacemouse::default_buttons();
            }
        });
        if let Some(b) = self.space_last_button {
            let action = self.space_buttons.iter().find(|(x, _)| *x == b).map_or("no action".to_string(), |(_, a)| button_action_label(a));
            ui.weak(format!("Last pressed: button {b} ({action})"));
        }
        self.sync_held_keys();
        ui_util::section(ui, "Live input");
        let m = spacemouse::to_view(axis);
        egui::Grid::new("space_live").num_columns(2).show(ui, |ui| {
            for (name, v) in spacemouse::Settings::AXIS_NAMES.iter().zip(m.t.iter().chain(m.r.iter())) {
                ui.label(*name);
                ui.add(egui::ProgressBar::new((v * 0.5 + 0.5).clamp(0.0, 1.0)).desired_width(160.0).text(format!("{v:+.2}")));
                ui.end_row();
            }
            ui.label("Buttons");
            ui.monospace(if buttons == 0 { "–".to_string() } else { (0..32).filter(|b| buttons & (1 << b) != 0).map(|b| b.to_string()).collect::<Vec<_>>().join(" ") });
            ui.end_row();
        });
        // Keep the live readout moving.
        ui.ctx().request_repaint_after(std::time::Duration::from_millis(50));
    }

    // ------------------------------------------------------------------
    // Viewport
    // ------------------------------------------------------------------

    fn viewport(&mut self, ui: &mut egui::Ui, frame: &eframe::Frame) {
        let rect = ui.available_rect_before_wrap();
        let vp = Viewport { rect };
        self.vp_aspect = vp.aspect();
        let response = ui.allocate_rect(rect, egui::Sense::click_and_drag());
        let ctx = ui.ctx().clone();

        // --- navigation -------------------------------------------------
        let line_points = ctx.options(|o| o.input_options.line_scroll_speed);
        let (mods, (wheel, pinch), pointer_delta, hover) = ctx.input(|i| {
            (i.modifiers, crate::navigation::wheel_zoom(&i.raw.events, line_points), i.pointer.delta(), i.pointer.hover_pos())
        });
        // `hovered` respects layering, so popups over the viewport don't leak input into it.
        let in_rect = hover.map(|p| rect.contains(p)).unwrap_or(false) && (response.hovered() || response.dragged());
        let nav = self.view.nav_mode.bindings(self.view.nav_custom);
        let primary_nav = nav.starts_navigation(PointerButton::Primary, mods);
        // A plain right-drag always navigates; tools get Cmd/Ctrl+right-drag (selection
        // grow / shrink, brush strength).
        let tool_claims_rmb = mods.command && !mods.alt && !mods.shift && self.tools.last().is_some_and(|t| t.claims_secondary_drag(&self.scene));
        if response.drag_started_by(PointerButton::Secondary) && tool_claims_rmb {
            if let Some(p) = ctx.input(|i| i.pointer.press_origin()) {
                self.tool_rmb = Some(p);
                self.tool_secondary(&ctx, vp, DragPhase::Start, p, egui::Vec2::ZERO);
            }
        }
        // Cmd/Ctrl+middle-drag sizes the brush (right or up grows it), for mice whose wheel the
        // OS accelerates.
        if response.drag_started_by(PointerButton::Middle) && mods.command && self.tools.last_mut().is_some_and(|t| t.brush_size().is_some()) {
            self.brush_drag = true;
        }
        if self.brush_drag {
            let d = pointer_delta.x - pointer_delta.y;
            if d != 0.0 {
                self.tool_wheel(&ctx, vp, d);
            }
            if !ctx.input(|i| i.pointer.button_down(PointerButton::Middle)) {
                self.brush_drag = false;
            }
        }
        for btn in [PointerButton::Secondary, PointerButton::Middle, PointerButton::Primary] {
            if self.tool_rmb.is_some() && btn == PointerButton::Secondary {
                continue;
            }
            if self.brush_drag && btn == PointerButton::Middle {
                continue;
            }
            if response.drag_started_by(btn) && nav.starts_navigation(btn, mods) {
                self.nav_drag = Some(btn);
                self.remember_view();
                let origin = ctx.input(|i| i.pointer.press_origin());
                self.nav_anchor = Some(self.nav_anchor_at(origin, vp));
                break;
            }
        }
        if let Some(start) = self.tool_rmb {
            let pos = hover.unwrap_or(start);
            let down = ctx.input(|i| i.pointer.button_down(PointerButton::Secondary));
            let phase = if down { DragPhase::Move } else { DragPhase::End };
            self.tool_secondary(&ctx, vp, phase, pos, pos - start);
            if !down {
                self.tool_rmb = None;
            }
        }
        if response.secondary_clicked() && !mods.alt {
            if let Some(p) = hover {
                self.tool_secondary_click(&ctx, vp, p);
            }
        }
        let nav_active = self.nav_drag.is_some();
        if let Some(btn) = self.nav_drag {
            if response.dragged_by(btn) {
                let d = pointer_delta;
                match nav.drag_action(btn, mods) {
                    Some(NavAction::Tumble) => {
                        let pivot = self.nav_anchor.unwrap_or(self.camera.target);
                        self.camera.tumble_about(d.x as f64, d.y as f64, pivot);
                    }
                    Some(NavAction::Pan) => {
                        let anchor = self.nav_anchor.unwrap_or(self.camera.target);
                        self.camera.pan_at(d.x as f64, d.y as f64, rect.height() as f64, anchor);
                    }
                    Some(NavAction::Zoom) => self.camera.zoom((1.0 + d.y as f64 * 0.005).max(0.05)),
                    None => {}
                }
            }
            if response.drag_stopped_by(btn) || !ctx.input(|i| i.pointer.button_down(btn)) {
                self.nav_drag = None;
                self.nav_anchor = None;
            }
        }
        if in_rect {
            // The brush-scroll modifier (Shift by default) + wheel sets brush tools' size, from
            // the raw events (one step per notch, no easing lag); the plain wheel always zooms.
            let now = ctx.input(|i| i.time);
            // Whole notches at once, from the raw events (see `navigation::wheel_zoom`); a
            // notch the brush took does not zoom.
            let mut wheel_y = wheel;
            if self.view.brush_scroll.held(mods) {
                let d = ctx.input(|i| crate::navigation::brush_wheel_delta(&i.raw.events));
                if d != 0.0 && self.tool_wheel(&ctx, vp, d) {
                    wheel_y = 0.0;
                }
            }
            let mut factor = 1.0f64;
            if wheel_y != 0.0 {
                factor *= (-wheel_y as f64 * 0.0025 * self.view.zoom_speed as f64).exp();
            }
            if pinch != 1.0 {
                factor /= pinch as f64;
            }
            if factor != 1.0 {
                if now - self.last_wheel > 0.6 {
                    self.remember_view();
                }
                self.last_wheel = now;
                let aim = if self.view.zoom_to_cursor { hover } else { Some(rect.center()) };
                let pivot = aim.and_then(|p| {
                    let ray = self.camera.ray_from_ndc(vp.ndc(p), vp.aspect());
                    tools::pick_ray(&mut self.scene, &ray, None).map(|(_, h)| h.point)
                });
                match pivot {
                    Some(p) if self.view.zoom_near => self.camera.zoom_about(factor, p),
                    Some(p) => self.camera.zoom_towards(factor, p),
                    None => self.camera.zoom(factor),
                }
            }
        }
        // Alt+click (no drag) on the model: pivot the view there.
        if response.clicked_by(PointerButton::Primary) && mods.alt {
            if let Some(p) = hover {
                self.recenter_at(p, vp, ctx.input(|i| i.time));
            }
        }
        if let Some((from, to, t0)) = self.view_anim {
            let t = ((ctx.input(|i| i.time) - t0) / 0.25).clamp(0.0, 1.0);
            let e = t * t * (3.0 - 2.0 * t);
            self.camera.target = from.lerp(to, e);
            if t >= 1.0 {
                self.view_anim = None;
            } else {
                ctx.request_repaint();
            }
        }
        let b = self.scene.bounds();
        if !b.is_empty() {
            self.camera.scene_radius = (b.diagonal() * 0.5).max(1e-6).max((self.camera.target - b.center()).length());
        }

        // --- tool pointer input -------------------------------------------
        let (pressed, down, released) = ctx.input(|i| {
            (i.pointer.button_pressed(PointerButton::Primary), i.pointer.button_down(PointerButton::Primary), i.pointer.button_released(PointerButton::Primary))
        });
        let live_pos = hover.filter(|_| in_rect || response.dragged());
        if live_pos.is_some() {
            self.last_pointer = live_pos;
        }
        // A release outside the viewport is still delivered, at the last known position, so
        // tools always see the end of their drags.
        let released_outside = released && live_pos.is_none() && !nav_active;
        let input = PointerInput {
            pos: if released_outside { self.last_pointer } else { live_pos },
            pressed: pressed && in_rect && !nav_active && !primary_nav,
            down: down && !nav_active && !primary_nav && (response.dragged_by(PointerButton::Primary) || (pressed && in_rect)),
            released: released && !nav_active,
            clicked: response.clicked_by(PointerButton::Primary) && !primary_nav,
            double_clicked: response.double_clicked_by(PointerButton::Primary) && !primary_nav,
            shift: mods.shift,
            command: mods.command,
        };
        let input = self.synthetic.pop_front().unwrap_or(input);
        if !self.synthetic.is_empty() {
            ctx.request_repaint();
        }
        let mut overlay = Overlay::default();
        if let Some(mut tool) = self.tools.pop() {
            let mut cx = self.tool_cx(&ctx, vp);
            if tool.busy().is_none() {
                tool.pointer(&input, &mut cx);
            }
            tool.overlay(&mut overlay, &mut cx);
            if let Some(r) = tool.take_request() {
                self.tools.push(tool);
                self.tools.push(r);
            } else {
                self.tools.push(tool);
            }
        } else if input.clicked {
            // No tool: clicking picks the active object.
            if let Some(p) = input.pos {
                let ray = self.camera.ray_from_ndc(vp.ndc(p), vp.aspect());
                if let Some((id, _)) = tools::pick_ray(&mut self.scene, &ray, None) {
                    self.scene.active = Some(id);
                }
            }
        }

        // --- boundaries overlay (cached per topology and mesh version) -------------
        let mut lines = std::mem::take(&mut self.lines_buf);
        lines.clear();
        if self.view.boundaries {
            let preview = self.tools.last().and_then(|t| t.preview());
            // Objects come and go with history steps: forget those no longer in the scene
            // (hidden ones keep their lines).
            if self.boundary_pruned != self.history.generation() {
                self.boundary_pruned = self.history.generation();
                let ids: FastSet<ObjectId> = self.scene.objects.iter().map(|o| o.id).collect();
                self.boundary_cache.retain(|id, _| ids.contains(id));
            }
            for o in self.scene.objects.iter().filter(|o| o.visible) {
                let m = match preview.filter(|(id, _)| *id == o.id) {
                    Some((_, pm)) => pm,
                    None => &*o.mesh,
                };
                lines.extend_from_slice(boundary_lines(self.boundary_cache.entry(o.id).or_default(), m));
            }
        }
        if self.view.show_grid {
            lines.extend(grid_lines(&self.camera, &self.scene.bounds(), self.view.grid_mm));
        }
        lines.extend(overlay.lines.iter().copied());

        // --- render ---------------------------------------------------------
        let ppp = ctx.pixels_per_point();
        let size = [(rect.width() * ppp).round().max(1.0) as u32, (rect.height() * ppp).round().max(1.0) as u32];
        if let (Some(rs), Some(_)) = (frame.wgpu_render_state(), self.renderer.as_ref()) {
            let tint = overlay.cut_plane.is_some().then_some(overlay.cut_target);
            self.sync_gpu(rs, tint);
            let aspect = vp.aspect();
            let mut globals = renderer::globals(
                self.camera.view_proj_f32(aspect),
                self.camera.view().as_mat4(),
                self.camera.eye().as_vec3(),
                overlay.brush,
                size[1] as f32,
            );
            let bg = self.view.background;
            globals.background = [bg[0], bg[1], bg[2], 1.0];
            if let Some((pl, region)) = overlay.cut_plane {
                globals.cut_plane = [pl.normal.x as f32, pl.normal.y as f32, pl.normal.z as f32, pl.d as f32];
                if let Some((c, r)) = region {
                    globals.cut_region = [c.x as f32, c.y as f32, c.z as f32, r as f32];
                }
            }
            let preview_id = self.tools.last().and_then(|t| t.preview()).map(|(id, _)| id);
            let mut objs: Vec<(&GpuMesh, bool)> = Vec::new();
            for o in self.scene.objects.iter().filter(|o| o.visible) {
                if Some(o.id) == preview_id {
                    if let Some(p) = &self.preview_gpu {
                        objs.push((&p.gpu, false));
                    }
                } else if let Some(g) = self.gpu.get(&o.id) {
                    let see_through = self.view.transparent_target && self.scene.target == Some(o.id);
                    objs.push((g, o.color.w < 0.999 || see_through || self.view.xray));
                }
            }
            let input = renderer::FrameInput { globals, objects: objs, lines: &lines, overlay_tris: &overlay.tris };
            let planes = self.slices.frame();
            let r = self.renderer.as_mut().unwrap();
            r.set_image_planes(planes);
            let tex = r.render(rs, size, &input);
            ui.painter().image(tex, rect, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), Color32::WHITE);
        } else {
            ui.painter().rect_filled(rect, 0.0, Color32::from_gray(30));
            ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, "wgpu renderer unavailable", egui::FontId::proportional(16.0), Color32::WHITE);
        }
        self.lines_buf = lines;

        // --- 2D overlays ------------------------------------------------------
        let painter = ui.painter_at(rect);
        if let (Some(btn), Some(p)) = (self.nav_drag, self.nav_anchor) {
            if nav.drag_action(btn, mods) == Some(NavAction::Tumble) {
                if let Some(n) = self.camera.project(p, vp.aspect()) {
                    let c = vp.to_screen(n);
                    painter.circle(c, 4.0, ui_util::ACCENT, egui::Stroke::new(1.5, Color32::from_gray(20)));
                }
            }
        }
        // The 3D mouse's centre of rotation: while it moves the view, then fading out.
        if let Some((p, last)) = self.space_marker {
            const FADE: f32 = 0.1;
            let age = if self.space_moving { 0.0 } else { last.elapsed().as_secs_f32() };
            if age < FADE {
                if let Some(n) = self.camera.project(p, vp.aspect()).filter(|n| (0.0..=1.0).contains(&n.z)) {
                    paint_centre_marker(&painter, vp.to_screen(n), 1.0 - age / FADE);
                }
                ctx.request_repaint();
            } else {
                self.space_marker = None;
            }
        }
        if self.view.show_pivots {
            let pivots: Vec<crate::scene::Pivot> = self.scene.objects.iter().filter(|o| o.visible).flat_map(|o| o.pivots.iter().copied()).collect();
            if !pivots.is_empty() {
                let cx = self.tool_cx(&ctx, vp);
                for p in &pivots {
                    tools::pivot::paint_pivot(&painter, &cx, p, false);
                }
            }
        }
        if let Some(mut tool) = self.tools.pop() {
            let mut cx = self.tool_cx(&ctx, vp);
            tool.paint(&painter, &mut cx);
            if let Some((label, p)) = tool.busy() {
                let r = egui::Rect::from_center_size(rect.center_top() + egui::vec2(0.0, 28.0), egui::vec2(260.0, 30.0));
                painter.rect_filled(r, 6.0, Color32::from_black_alpha(180));
                painter.text(r.center(), egui::Align2::CENTER_CENTER, format!("{label}… {:.0}%", p * 100.0), egui::FontId::proportional(14.0), Color32::WHITE);
                // The percentage moves on its own, 15 times a second is plenty; the job's end
                // asks for a frame itself.
                ctx.request_repaint_after(BUSY_REPAINT);
            }
            self.tools.push(tool);
        }
        if self.scene.is_empty() && self.import_job.is_none() {
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "Drop an STL / OBJ / PLY / OFF / 3MF / AMF file or a .polymix project here\nor press ⌘O to import",
                egui::FontId::proportional(18.0),
                Color32::from_gray(150),
            );
        }
        if let Some(job) = &self.import_job {
            painter.text(rect.center(), egui::Align2::CENTER_CENTER, format!("{}…", job.label), egui::FontId::proportional(18.0), Color32::WHITE);
            ctx.request_repaint_after(BUSY_REPAINT);
        }
        self.axis_gizmo(&painter, rect);
        if self.view.show_view_cube {
            self.view_cube(ui, rect);
        }
        self.fit_button(ui, rect);
        if self.nav_drag.is_some() || response.dragged() || response.is_pointer_button_down_on() {
            // Keep press-and-hold work going without pointer motion (a Flow brush held still,
            // egui deciding between click and drag). Mere hovering needs no frames: pointer
            // motion brings its own, so the brush cursor still follows.
            ctx.request_repaint_after(std::time::Duration::from_millis(16));
        }
    }

    /// The view cube in the top-right corner: click a face / edge / corner for that view,
    /// drag to tumble. It sits above the viewport, so its input never reaches tools.
    fn view_cube(&mut self, ui: &mut egui::Ui, rect: egui::Rect) {
        let scale = 26.0;
        let centre = rect.right_top() + egui::vec2(-scale * 2.2, scale * 2.2);
        let area = egui::Rect::from_center_size(centre, egui::vec2(scale * 3.6, scale * 3.6));
        let resp = ui.interact(area, egui::Id::NULL.with("view_cube"), egui::Sense::click_and_drag());
        let orient = self.camera.orient;
        let hover = resp.hover_pos().and_then(|p| crate::view_cube::region_at(orient, centre, scale, p));
        if resp.dragged() {
            let d = resp.drag_delta();
            let pivot = self.center_of_mass().unwrap_or(self.camera.target);
            self.camera.tumble_about(d.x as f64, d.y as f64, pivot);
        } else if resp.clicked() {
            if let Some(d) = hover {
                let (yaw, pitch) = crate::view_cube::yaw_pitch(d);
                self.standard_view(yaw, pitch);
            }
        }
        crate::view_cube::draw(&ui.painter_at(rect), self.camera.orient, centre, scale, hover, ui_util::ACCENT);
    }

    /// "Fit" beside the view cube (in the corner when the cube is hidden): frame the visible
    /// objects.
    fn fit_button(&mut self, ui: &mut egui::Ui, rect: egui::Rect) {
        let scale = 26.0;
        let at = if self.view.show_view_cube {
            rect.right_top() + egui::vec2(-scale * 2.2 - scale * 1.8 - 26.0, scale * 2.2)
        } else {
            rect.right_top() + egui::vec2(-30.0, 22.0)
        };
        let button = egui::Button::new(RichText::new("Fit").size(12.0)).corner_radius(4.0);
        if ui.put(egui::Rect::from_center_size(at, egui::vec2(36.0, 22.0)), button).on_hover_text("Fit the visible objects in the view (F)").clicked() {
            self.focus_scene();
        }
    }

    fn axis_gizmo(&self, painter: &egui::Painter, rect: egui::Rect) {
        let mut c = rect.left_bottom() + egui::vec2(40.0, -40.0);
        // Beside the floating tool panel when that reaches down to it.
        if let Some(r) = self.tool_panel_rect.filter(|r| r.bottom() > c.y - 40.0) {
            c.x = r.right() + 40.0;
        }
        let view = self.camera.view();
        for (axis, col, name) in [
            (DVec3::X, Color32::from_rgb(230, 80, 80), "X"),
            (DVec3::Y, Color32::from_rgb(90, 200, 90), "Y"),
            (DVec3::Z, Color32::from_rgb(90, 140, 240), "Z"),
        ] {
            let v = view.transform_vector3(axis);
            let end = c + egui::vec2(v.x as f32, -v.y as f32) * 26.0;
            painter.line_segment([c, end], egui::Stroke::new(2.0, col));
            painter.text(end, egui::Align2::CENTER_CENTER, name, egui::FontId::proportional(11.0), col);
        }
    }

    /// `cut_tint`: Some(target) when a cut preview is active (target None = all objects).
    fn sync_gpu(&mut self, rs: &egui_wgpu::RenderState, cut_tint: Option<Option<ObjectId>>) {
        let renderer = self.renderer.as_ref().unwrap();
        let ids: FastSet<ObjectId> = self.scene.objects.iter().map(|o| o.id).collect();
        self.gpu.retain(|id, _| ids.contains(id));
        let active = self.scene.active;
        let multiple = self.scene.objects.iter().filter(|o| o.visible).count() > 1;
        let preview = self.tools.last().and_then(|t| t.preview());
        if preview.is_none() {
            // The preview has ended. When the object now holds the very mesh it showed (the
            // tool's result was accepted), its GPU copy takes over the object's: accepting
            // then uploads nothing again. Style keys, selection and texture are checked by
            // `sync_mesh` below as for any object (the preview was drawn without selection).
            // Not for an object showing its texture: the object's GPU copy keeps the image,
            // which the preview's never had (it would be uploaded again).
            let textures = matches!(self.view.colors, ColorMode::Texture | ColorMode::TextureUnlit);
            if let Some(p) = self.preview_gpu.take() {
                let shows_texture = |o: &crate::scene::SceneObject| textures && o.texture.is_some() && o.uvs.as_ref().is_some_and(|u| u.topo == o.mesh.topology_version());
                if self.scene.get(p.id).is_some_and(|o| (o.mesh.shape_version(), o.mesh.topology_version()) == (p.shape, p.topo) && !shows_texture(o)) {
                    self.gpu.insert(p.id, p.gpu);
                }
            }
        }
        let mut no_moves = Vec::new();
        let see_through = self.scene.target.filter(|_| self.view.transparent_target);
        for o in self.scene.objects.iter_mut().filter(|o| o.visible) {
            let mut color = o.color;
            if see_through == Some(o.id) {
                color.w = color.w.min(0.35);
            }
            let style = ObjectStyle {
                color,
                wireframe: self.view.wireframe,
                flat: self.view.normals == NormalMode::Face,
                group_colors: self.view.colors == ColorMode::Group || self.tools.last().is_some_and(|t| t.shows_groups()),
                dim: multiple && Some(o.id) != active,
                cut_tint: matches!(cut_tint, Some(t) if t.is_none() || t == Some(o.id)),
                crease_deg: self.view.sharp_edges.then_some(self.view.crease_deg),
                group_normals: self.view.normals == NormalMode::Group,
                vertex_colors: self.view.colors == ColorMode::Vertex,
                uv_checker: self.view.colors == ColorMode::UvChecker || self.tools.last().is_some_and(|t| t.shows_uvs()),
                texture_mode: match self.view.colors {
                    ColorMode::Texture => 1,
                    ColorMode::TextureUnlit => 2,
                    _ => 0,
                },
                xray: self.view.xray,
                material: crate::materials::MATERIALS[o.material.min(crate::materials::MATERIALS.len() - 1)],
            };
            if let Some((pid, pm)) = preview {
                if pid == o.id {
                    let p = self.preview_gpu.get_or_insert_with(|| PreviewGpu { id: pid, shape: u64::MAX, topo: u64::MAX, gpu: renderer.create_mesh(&rs.device) });
                    renderer.sync_mesh(&rs.device, &rs.queue, &mut p.gpu, pm, None, 0, &style, (&mut no_moves, 0), None, None, None);
                    // The GPU copy now holds exactly this state of the preview mesh.
                    (p.id, p.shape, p.topo) = (pid, pm.shape_version(), pm.topology_version());
                    continue;
                }
            }
            let g = self.gpu.entry(o.id).or_insert_with(|| renderer.create_mesh(&rs.device));
            let scalars = match (&o.colors, &o.scalars) {
                (Some((stamp, c)), _) => Some((*stamp, renderer::TriField::Rgb(c.as_slice()))),
                (None, Some((stamp, v))) => Some((*stamp, renderer::TriField::Scalar(v.as_slice()))),
                (None, None) => None,
            };
            let uvs = o.uvs.as_deref().filter(|u| u.topo == o.mesh.topology_version()).map(|u| (u.stamp, u.corners.as_slice()));
            let texture = o.texture.as_deref().map(|t| (t.stamp, t.width, t.height, t.rgba.as_slice()));
            renderer.sync_mesh(&rs.device, &rs.queue, g, &o.mesh, Some(&o.selection), o.selection_stamp, &style, (&mut o.moved_vertices, o.moved_from), scalars, uvs, texture);
        }
    }

    // ------------------------------------------------------------------
    // Scripting (testing)
    // ------------------------------------------------------------------

    fn run_script(&mut self, ctx: &egui::Context, vp: Viewport) {
        let busy = self.import_job.is_some() || self.tool_busy() || !self.synthetic.is_empty();
        let Some(script) = &mut self.script else { return };
        if busy {
            ctx.request_repaint();
            return;
        }
        if script.wait > 0 {
            script.wait -= 1;
            ctx.request_repaint();
            return;
        }
        let Some(step) = script.steps.pop_front() else {
            self.script = None;
            return;
        };
        ctx.request_repaint();
        let (cmd, arg) = step.split_once(':').unwrap_or((step.as_str(), ""));
        let arg = arg.to_string();
        match cmd {
            "lasso" => {
                // Polygon in viewport-relative coordinates: x0,y0,x1,y1,...
                let v: Vec<f32> = arg.split(',').filter_map(|x| x.parse().ok()).collect();
                let at = |x: f32, y: f32| vp.rect.min + egui::vec2(x * vp.rect.width(), y * vp.rect.height());
                let pts: Vec<egui::Pos2> = v.chunks(2).filter(|c| c.len() == 2).map(|c| at(c[0], c[1])).collect();
                for (k, w) in pts.windows(2).enumerate() {
                    for j in 0..8 {
                        let p = w[0] + (w[1] - w[0]) * (j as f32 / 8.0);
                        self.synthetic.push_back(PointerInput { pos: Some(p), pressed: k == 0 && j == 0, down: true, ..Default::default() });
                    }
                }
                if let Some(last) = pts.last() {
                    self.synthetic.push_back(PointerInput { pos: Some(*last), released: true, ..Default::default() });
                }
                script.wait = 2;
            }
            "rclick" => {
                let v: Vec<f32> = arg.split(',').filter_map(|x| x.parse().ok()).collect();
                if v.len() == 2 {
                    let p = vp.rect.min + egui::vec2(v[0] * vp.rect.width(), v[1] * vp.rect.height());
                    self.tool_secondary_click(ctx, vp, p);
                }
            }
            "dclickhole" => {
                // dclickhole:N — double-click a visible triangle on the rim of the active
                // object's Nth open boundary (ordered by vertex id), as a user selects a hole ring.
                let k: usize = arg.parse().unwrap_or(0);
                let target = self.scene.active().and_then(|o| {
                    let m = o.mesh.clone();
                    let mut loops = mm_core::topology::boundary_loops(&m);
                    loops.sort_by_key(|l| l.vertices.iter().min().copied());
                    let lp = loops.get(k)?;
                    Some((o.id, lp.edges.iter().map(|&e| m.edge_triangles(e)[0]).collect::<Vec<u32>>(), m))
                });
                if let Some((id, rim, m)) = target {
                    let spot = rim.iter().find_map(|&t| {
                        let p = self.camera.project(m.tri_centroid(t), vp.aspect())?;
                        let screen = vp.to_screen(p);
                        let ray = self.camera.ray_from_ndc(vp.ndc(screen), vp.aspect());
                        (vp.rect.contains(screen) && tools::pick_ray(&mut self.scene, &ray, Some(id)).is_some_and(|(_, h)| h.tid == t)).then_some(screen)
                    });
                    if let Some(p) = spot {
                        let base = PointerInput::default();
                        self.synthetic.push_back(PointerInput { pos: Some(p), ..base.clone() });
                        self.synthetic.push_back(PointerInput { pos: Some(p), pressed: true, down: true, ..base.clone() });
                        self.synthetic.push_back(PointerInput { pos: Some(p), released: true, clicked: true, double_clicked: true, ..base });
                    } else {
                        println!("[polymix] dclickhole: no visible rim triangle on boundary {k}");
                    }
                }
                script.wait = 2;
            }
            "click" | "dclick" | "drag" | "sdrag" => {
                // Viewport-relative coordinates in 0..1.
                let v: Vec<f32> = arg.split(',').filter_map(|x| x.parse().ok()).collect();
                let at = |x: f32, y: f32| vp.rect.min + egui::vec2(x * vp.rect.width(), y * vp.rect.height());
                let base = PointerInput { shift: cmd == "sdrag", ..Default::default() };
                if (cmd == "click" || cmd == "dclick") && v.len() == 2 {
                    let p = Some(at(v[0], v[1]));
                    self.synthetic.push_back(PointerInput { pos: p, ..base.clone() });
                    self.synthetic.push_back(PointerInput { pos: p, pressed: true, down: true, ..base.clone() });
                    self.synthetic.push_back(PointerInput { pos: p, released: true, clicked: true, double_clicked: cmd == "dclick", ..base.clone() });
                } else if v.len() >= 4 {
                    let steps = v.get(4).copied().unwrap_or(20.0) as usize;
                    for k in 0..=steps {
                        let t = k as f32 / steps as f32;
                        let p = Some(at(v[0] + (v[2] - v[0]) * t, v[1] + (v[3] - v[1]) * t));
                        self.synthetic.push_back(PointerInput { pos: p, pressed: k == 0, down: true, ..base.clone() });
                    }
                    let p = Some(at(v[2], v[3]));
                    self.synthetic.push_back(PointerInput { pos: p, released: true, ..base.clone() });
                }
                script.wait = 2;
            }
            "open" => self.run(MeshCommand::Open(vec![PathBuf::from(arg)]), ctx, vp),
            "wire" => self.run(MeshCommand::Wireframe(!self.view.wireframe), ctx, vp),
            "flat" => self.run(MeshCommand::Normals(if self.view.normals == NormalMode::Face { NormalMode::Group } else { NormalMode::Face }), ctx, vp),
            "normals" => self.run(MeshCommand::Normals(NormalMode::from_key(arg.as_str()).unwrap_or(self.view.normals)), ctx, vp),
            "grid" => self.run(MeshCommand::Grid(!self.view.show_grid), ctx, vp),
            "crease" => match arg.parse() {
                Ok(d) => (self.view.sharp_edges, self.view.crease_deg) = (true, d),
                Err(_) => self.view.sharp_edges = false,
            },
            "groups" => self.run(MeshCommand::Colors(if self.view.colors == ColorMode::Group { ColorMode::Vertex } else { ColorMode::Group }), ctx, vp),
            "focus" => self.camera.focus(&self.scene.bounds()),
            "zoom" => {
                // zoom:x,y,factor — zoom toward the surface under a viewport-relative point.
                let v: Vec<f64> = arg.split(',').filter_map(|x| x.parse().ok()).collect();
                if v.len() == 3 {
                    let p = vp.rect.min + egui::vec2(v[0] as f32 * vp.rect.width(), v[1] as f32 * vp.rect.height());
                    let ray = self.camera.ray_from_ndc(vp.ndc(p), vp.aspect());
                    if let Some((_, h)) = tools::pick_ray(&mut self.scene, &ray, None) {
                        self.camera.zoom_towards(v[2], h.point);
                    }
                }
            }
            "orbit" => {
                let v: Vec<f64> = arg.split(',').filter_map(|x| x.parse().ok()).collect();
                if v.len() == 2 {
                    self.camera.set_view(v[0].to_radians(), v[1].to_radians());
                }
            }
            "zup" => self.run(MeshCommand::Up(UpAxis::Z), ctx, vp),
            "tumble" => {
                // tumble:dx,dy — rotate as if dragged by (dx, dy) points.
                let v: Vec<f64> = arg.split(',').filter_map(|x| x.parse().ok()).collect();
                if v.len() == 2 {
                    for _ in 0..10 {
                        self.camera.tumble(v[0] / 10.0, v[1] / 10.0);
                    }
                }
            }
            "select_all" => self.run(MeshCommand::Select(commands::SelectCmd::All), ctx, vp),
            "duplicate" => self.run(MeshCommand::Duplicate, ctx, vp),
            "material" => {
                if let (Ok(i), Some(o)) = (arg.parse::<usize>(), self.scene.active_mut()) {
                    let m = crate::materials::MATERIALS[i.min(crate::materials::MATERIALS.len() - 1)];
                    o.material = i;
                    if let Some(c) = m.color {
                        o.color = Vec4::new(c[0], c[1], c[2], m.alpha.unwrap_or(1.0));
                    }
                }
            }
            "theme" => {
                if let Some(t) = ui_util::Theme::from_key(arg.as_str()) {
                    self.view.theme = t;
                    ui_util::apply_theme(ctx, t);
                }
            }
            "save" => self.run(MeshCommand::SaveProject(PathBuf::from(arg)), ctx, vp),
            "prompts" => self.script_prompts = arg == "on",
            "prefs" => {
                self.show_prefs = true;
                self.prefs_tab = arg.parse().unwrap_or(0);
            }
            "hotbox" => self.hotbox = if self.hotbox.is_some() { None } else { Some(vp.rect.center()) },
            // target / activate / hide:<index in the object list>
            "target" => {
                let id = arg.parse::<usize>().ok().and_then(|i| self.scene.objects.get(i)).map(|o| o.id);
                self.run(MeshCommand::SetTarget(id), ctx, vp);
            }
            "activate" | "hide" => {
                if let Some(id) = arg.parse::<usize>().ok().and_then(|i| self.scene.objects.get(i)).map(|o| o.id) {
                    self.run(if cmd == "hide" { MeshCommand::SetVisible(id, false) } else { MeshCommand::Activate(id) }, ctx, vp);
                }
            }
            "undo" => self.run(MeshCommand::Undo, ctx, vp),
            "redo" => self.run(MeshCommand::Redo, ctx, vp),
            "log" => {
                // Print scene statistics to stdout (automated checks).
                println!("[polymix] {} camera distance={:.3} tool={}", arg, self.camera.distance, self.tools.last().map(|t| t.name()).unwrap_or("none"));
                for o in &self.scene.objects {
                    println!(
                        "[polymix] {} {}: tris={} verts={} open_edges={} closed={} valid={} shells={} selected={} tool={}",
                        arg,
                        o.name,
                        o.mesh.triangle_count(),
                        o.mesh.vertex_count(),
                        o.mesh.boundary_edge_ids().count(),
                        o.mesh.is_closed(),
                        o.mesh.check_validity().is_ok(),
                        mm_core::topology::connected_components(&o.mesh).len(),
                        o.selection.len(),
                        self.tools.last().map(|t| t.name()).unwrap_or("none")
                    );
                }
            }
            "category" => {
                let c = match arg.as_str() {
                    "meshmix" => Some(Category::Meshmix),
                    "edit" => Some(Category::Edit),
                    "vessel" => Some(Category::Vessel),
                    "analysis" => Some(Category::Analysis),
                    "sculpt" => Some(Category::Sculpt),
                    "shaders" => Some(Category::Shaders),
                    _ => None,
                };
                self.switch_to(SwitchTo::Category(c), ctx, vp);
            }
            "toolcmd" => {
                if let Some(mut t) = self.tools.pop() {
                    let mut cx = self.tool_cx(ctx, vp);
                    t.script(&arg, &mut cx);
                    self.tools.push(t);
                }
            }
            "accept" => self.run(MeshCommand::AcceptTool, ctx, vp),
            "tool" => self.run(MeshCommand::OpenTool(arg.clone()), ctx, vp),
            other => log::warn!("unknown script step {other}"),
        }
    }

    // ------------------------------------------------------------------
    // Keyboard
    // ------------------------------------------------------------------

    fn handle_keys(&mut self, ctx: &egui::Context, vp: Viewport) {
        // A key typed for a 3D mouse button (Preferences → 3D mouse) does nothing else.
        if ctx.egui_wants_keyboard_input() || self.space_key_capture.is_some() {
            return;
        }
        let i = ctx.input(|i| i.clone());
        let cmd = i.modifiers.command;
        // Undo / redo: Cmd+Z, Shift+Cmd+Z / Cmd+Y, Alt+Left / Alt+Right. Alt+Shift+arrows
        // ("skip tool changes" in Meshmixer) behave the same: tools add no history entries
        // of their own.
        if (i.key_pressed(Key::Z) && cmd && !i.modifiers.shift) || (i.key_pressed(Key::ArrowLeft) && i.modifiers.alt) {
            self.run(MeshCommand::Undo, ctx, vp);
            return;
        }
        if (i.key_pressed(Key::Z) && cmd && i.modifiers.shift) || (i.key_pressed(Key::Y) && cmd) || (i.key_pressed(Key::ArrowRight) && i.modifiers.alt) {
            self.run(MeshCommand::Redo, ctx, vp);
            return;
        }
        if i.key_pressed(Key::OpenBracket) && cmd {
            self.previous_view();
            return;
        }
        if i.key_pressed(Key::CloseBracket) && cmd {
            self.next_view();
            return;
        }
        if i.key_pressed(Key::O) && cmd && i.modifiers.shift {
            self.show_objects = !self.show_objects;
            return;
        }
        if i.key_pressed(Key::O) && cmd {
            self.import_dialog(ctx);
            return;
        }
        if i.key_pressed(Key::E) && cmd {
            self.export_dialog(ctx, if self.prefs.export_whole_scene { ExportScope::AllVisible } else { ExportScope::Active });
            return;
        }
        if i.key_pressed(Key::S) && cmd {
            self.save_project(ctx, i.modifiers.shift);
            return;
        }
        if i.events.iter().any(|e| matches!(e, egui::Event::Copy)) {
            self.copy(ctx);
            return;
        }
        if i.key_pressed(Key::Comma) && cmd {
            self.show_prefs = !self.show_prefs;
            return;
        }
        if i.key_pressed(Key::Escape) {
            if let Some(mut t) = self.tools.pop() {
                let mut cx = self.tool_cx(ctx, vp);
                if t.cancel(&mut cx) {
                    t.on_close(&mut cx);
                } else {
                    self.tools.push(t);
                }
            } else {
                self.category = None;
            }
            return;
        }
        if i.key_pressed(Key::Space) {
            self.hotbox = if self.hotbox.is_some() { None } else { i.pointer.hover_pos().or(Some(vp.rect.center())) };
            return;
        }
        // Accept: Enter, or A (Meshmixer); in Sculpt, A toggles attract.
        let accept_a = i.key_pressed(Key::A) && !cmd && !i.modifiers.shift && !i.modifiers.alt && self.tools.last().is_some_and(|t| t.name() != "Sculpt");
        if i.key_pressed(Key::Enter) || accept_a {
            if let Some(mut t) = self.tools.pop() {
                let mut cx = self.tool_cx(ctx, vp);
                // Enter never accepts while the tool is still computing its result.
                if t.busy().is_none() && t.accept(&mut cx) {
                    t.on_close(&mut cx);
                } else {
                    self.tools.push(t);
                }
            }
            return;
        }
        // Tool-specific keys take precedence.
        if let Some(mut t) = self.tools.pop() {
            let mut cx = self.tool_cx(ctx, vp);
            let used = t.keys(&i, &mut cx);
            let req = t.take_request();
            let is_select = t.name() == "Select";
            self.tools.push(t);
            if let Some(r) = req {
                self.tools.push(r);
                return;
            }
            if used {
                return;
            }
            let brush_key = [Key::Num1, Key::Num2, Key::Num3, Key::Num4].iter().any(|&k| i.key_pressed(k));
            if is_select && !brush_key {
                // The select tool consumes most letter keys.
                if i.key_pressed(Key::W) {
                    self.run(MeshCommand::Wireframe(!self.view.wireframe), ctx, vp);
                }
                if i.key_pressed(Key::Y) && i.modifiers.shift {
                    self.run(MeshCommand::SeparateShells, ctx, vp);
                }
                return;
            }
        }
        if i.key_pressed(Key::W) && !cmd {
            self.run(MeshCommand::Wireframe(!self.view.wireframe), ctx, vp);
        }
        if i.key_pressed(Key::F) && !cmd {
            self.run(MeshCommand::Focus, ctx, vp);
        }
        if self.scene.active().is_some() && !cmd {
            // Tool hotkeys work from inside other tools too: the open tool is accepted first.
            // S always starts Select with nothing selected.
            let tool = |name: &str| Some(SwitchTo::Tool(name.into()));
            let to = if i.key_pressed(Key::S) {
                Some(SwitchTo::ClearSelect)
            } else if i.key_pressed(Key::I) {
                tool("inspector")
            } else if i.key_pressed(Key::T) && !i.modifiers.shift {
                tool("transform")
            } else if i.key_pressed(Key::Num1) {
                tool("sculptsmooth")
            } else if i.key_pressed(Key::Num2) {
                tool("sculpt")
            } else if i.key_pressed(Key::Num3) {
                tool("sculptvolume")
            } else if i.key_pressed(Key::Num4) {
                tool("stamp")
            } else {
                None
            };
            if let Some(to) = to {
                self.switch_to(to, ctx, vp);
            } else if i.key_pressed(Key::Y) && i.modifiers.shift {
                self.run(MeshCommand::SeparateShells, ctx, vp);
            }
        }
    }
}

impl polymix_workspace::Workspace for MeshWorkspace {
    fn name(&self) -> &'static str {
        "Mesh"
    }

    fn tick(&mut self, ctx: &egui::Context, active: bool) {
        self.poll_jobs();
        if self.agent_import {
            // An agent's import asks nothing, so it needs no drawn frame (the window may be
            // covered).
            self.finish_import(ctx);
        }
        self.autosave_tick(ctx);
        self.bvh_tick(ctx);
        if self.history.by_agent {
            self.agent_quiet_frames = if self.agent_quiet() { self.agent_quiet_frames + 1 } else { 0 };
            // An action through the panels lands a frame after the call: leave room for it.
            if self.agent_quiet_frames >= 3 {
                self.history.by_agent = false;
            } else {
                ctx.request_repaint();
            }
        }
        if !active {
            // 3D-mouse buttons meant for the Mesh view must not pile up while another
            // workspace is shown.
            let _ = self.space.poll();
        }
    }

    fn title(&self) -> String {
        match (&self.project_path, self.scene.active()) {
            (Some(p), _) => format!("Polymix — {}", p.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default()),
            (None, Some(o)) => format!("Polymix — {}", o.name),
            (None, None) => "Polymix — Mesh".to_string(),
        }
    }

    fn accepts(&self, path: &std::path::Path) -> bool {
        mm_io::Format::from_path(path).is_some() || path.extension().is_some_and(|e| e.eq_ignore_ascii_case(crate::project::EXTENSION))
    }

    fn open(&mut self, paths: Vec<PathBuf>, ctx: &egui::Context) {
        self.start_import(paths, ctx);
    }

    fn script_push(&mut self, step: String) {
        self.script.get_or_insert_with(Script::default).steps.push_back(step);
    }

    fn script_idle(&self) -> bool {
        let busy = self.import_job.is_some() || self.project_job.is_some() || self.export_job.is_some() || self.pending_import.is_some() || self.tool_busy() || self.tools.last().is_some_and(|t| t.pending()) || !self.synthetic.is_empty();
        !busy && self.script.as_ref().is_none_or(|s| s.steps.is_empty() && s.wait == 0)
    }

    fn needs_attention(&self) -> bool {
        // The recovery prompt is drawn by this workspace, and autosave waits for an answer.
        self.recovery.is_some()
    }

    fn agent_tools() -> Vec<polymix_workspace::agent::AgentTool> {
        agent::tools()
    }

    fn agent_call(&mut self, tool: &str, args: &serde_json::Value, ctx: &egui::Context) -> polymix_workspace::agent::AgentReply {
        agent::call(self, tool, args, ctx)
    }

    fn agent_acting(&mut self) {
        self.history.by_agent = true;
        self.agent_quiet_frames = 0;
    }

    fn emergency_save(&mut self) {
        // After a crash: keep unsaved work where the restarted workspace offers to recover it.
        let Some(auto) = &self.autosave else { return };
        if self.scene.is_empty() || self.history.generation() == auto.saved_generation || self.recovery.is_some() {
            return;
        }
        let file = auto.file.clone();
        // Right away, on this thread: no compression.
        if let Err(e) = crate::project::save_shared(&file, &self.autosave_project(), crate::project::Compression::Stored) {
            log::error!("emergency save failed: {e:#}");
        }
    }

    fn on_exit(&mut self) {
        if let Some(auto) = &self.autosave {
            let _ = std::fs::remove_file(&auto.marker);
            if self.recovery.is_none() {
                let _ = std::fs::remove_file(&auto.file);
            }
        }
    }

    fn save(&mut self, st: &mut dyn eframe::Storage) {
        // Scripted (test) sessions never overwrite the user's preferences.
        if std::env::var_os("POLYMIX_SCRIPT").is_some() {
            return;
        }
        let b = |v: bool| if v { "1".to_string() } else { "0".to_string() };
        st.set_string("wireframe", b(self.view.wireframe));
        st.set_string("normals", self.view.normals.key().to_string());
        st.set_string("boundaries", b(self.view.boundaries));
        st.set_string("color_mode", self.view.colors.key().to_string());
        st.set_string("grid", b(self.view.show_grid));
        st.set_string("grid_mm", self.view.grid_mm.to_string());
        st.set_string("transparent_target", b(self.view.transparent_target));
        st.set_string("xray", b(self.view.xray));
        st.set_string("view_cube", b(self.view.show_view_cube));
        st.set_string("pivots", b(self.view.show_pivots));
        st.set_string("theme", self.view.theme.key().to_string());
        let bg = self.view.background;
        st.set_string("background", format!("{},{},{}", bg[0], bg[1], bg[2]));
        st.set_string("sharp_edges", b(self.view.sharp_edges));
        st.set_string("nav_mode", self.view.nav_mode.key().to_string());
        st.set_string("nav_custom", self.view.nav_custom.to_pref());
        st.set_string("brush_scroll", self.view.brush_scroll.key().to_string());
        st.set_string("brush_size", self.brush.get().to_string());
        st.set_string("zoom_speed", self.view.zoom_speed.to_string());
        st.set_string("zoom_near", b(self.view.zoom_near));
        st.set_string("zoom_to_cursor", b(self.view.zoom_to_cursor));
        st.set_string("spacemouse", self.space_cfg.to_pref());
        st.set_string("spacemouse_buttons", spacemouse::buttons_to_pref(&self.space_buttons));
        st.set_string("crease_deg", self.view.crease_deg.to_string());
        st.set_string("show_objects", b(self.show_objects));
        st.set_string("objects_folded", b(self.objects_folded));
        st.set_string("objects_width", self.objects_width.to_string());
        st.set_string("orthographic", b(self.camera.orthographic));
        st.set_string("free_orbit", b(self.camera.free_orbit));
        st.set_string("up_axis", if self.camera.up_axis == UpAxis::Z { "z".into() } else { "y".into() });
        st.set_string("export_ascii", b(self.export_ascii));
        st.set_string("reset_camera_on_import", b(self.prefs.reset_camera_on_import));
        st.set_string("export_whole_scene", b(self.prefs.export_whole_scene));
        st.set_string("obj_export_groups", b(self.prefs.obj_export_groups));
        st.set_string("obj_import_groups", b(self.prefs.obj_import_groups));
        st.set_string("flip_zy", b(self.prefs.flip_zy));
        st.set_string("autosave_minutes", self.prefs.autosave_minutes.to_string());
        st.set_string("fill_new_groups", b(crate::tools::ops::FILL_NEW_GROUPS.load(std::sync::atomic::Ordering::Relaxed)));
        st.set_string(
            "import_mode",
            match self.prefs.import_mode {
                ImportMode::Ask => "ask",
                ImportMode::Append => "append",
                ImportMode::Replace => "replace",
            }
            .into(),
        );
        st.set_string(
            "scale_mode",
            match self.prefs.scale_mode {
                ScaleMode::Ask => "ask".into(),
                ScaleMode::Never => "never".into(),
                ScaleMode::Always(k) => format!("always:{k}"),
            },
        );
        st.set_string("recent", self.recent.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join("\n"));
        if let Some(d) = &self.last_export_dir {
            st.set_string("last_export_dir", d.display().to_string());
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.share_brush();

        // The viewport rect is only known after the side panels are laid out; tools that need it
        // during panel drawing use the previous frame's value.
        let vp_prev = Viewport { rect: ctx.data(|d| d.get_temp::<egui::Rect>(egui::Id::NULL.with("vp_rect"))).unwrap_or(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0))) };

        self.space_mouse(&ctx, vp_prev);
        if let Some(to) = self.pending_switch.clone() {
            self.switch_to(to, &ctx, vp_prev);
            if self.pending_switch.is_some() {
                // Still waiting for the tool's result: its job's end asks for a frame, this
                // only keeps looking now and then.
                ctx.request_repaint_after(BUSY_REPAINT);
            } else {
                ctx.request_repaint();
            }
        }

        egui::Panel::top("menu").show(ui, |ui| self.menu_bar(ui, vp_prev));
        egui::Panel::bottom("status").show(ui, |ui| {
            ui.horizontal(|ui| {
                if let Some(j) = &self.import_job {
                    polymix_workspace::spinner(ui);
                    ui.label(&j.label);
                }
                if let Some(j) = &self.export_job {
                    polymix_workspace::spinner(ui);
                    ui.label(&j.label);
                }
                ui.label(&self.status);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let dt = ctx.input(|i| i.stable_dt);
                    ui.weak(format!("{:.1} ms", dt * 1000.0));
                });
            });
        });
        egui::Panel::left("palette").resizable(false).exact_size(118.0).show(ui, |ui| {
            ui.vertical_centered(|ui| self.palette(ui, vp_prev));
        });
        // The tool panel and the object browser float over the viewport, so opening or
        // closing them never resizes the view (the model stays where it is on screen).
        let vr = vp_prev.rect;
        let margin = 8.0;
        let show_tool_panel = !self.tools.is_empty() || self.category.is_some();
        self.tool_panel_rect = None;
        if show_tool_panel {
            let width = self.tool_panel_width;
            let r = egui::Area::new(egui::Id::NULL.with("floating_tool_panel"))
                .fixed_pos(vr.min + egui::vec2(margin, margin))
                .order(egui::Order::Middle)
                .show(&ctx, |ui| {
                    egui::Frame::window(ui.style()).show(ui, |ui| {
                        ui.set_width(width);
                        // As tall as its content, scrolling once it reaches the bottom of the view.
                        // (An area offers its content last frame's size; allow it to grow.)
                        let max_h = (vr.height() - 2.0 * margin - 20.0).max(80.0);
                        ui.set_max_height(max_h);
                        egui::ScrollArea::vertical().max_height(max_h).show(ui, |ui| {
                            if !self.tools.is_empty() {
                                self.tool_panel(ui, vp_prev);
                            } else if let Some(cat) = self.category {
                                self.category_panel(ui, cat, vp_prev);
                            }
                        });
                    });
                });
            let rect = r.response.rect;
            // Drag the right edge to change the width.
            let edge = egui::Rect::from_min_max(egui::pos2(rect.right() - 4.0, rect.top()), egui::pos2(rect.right() + 4.0, rect.bottom()));
            let handle = egui::Area::new(egui::Id::NULL.with("tool_panel_edge")).fixed_pos(edge.min).order(egui::Order::Middle).show(&ctx, |ui| {
                ui.allocate_response(edge.size(), egui::Sense::drag()).on_hover_cursor(egui::CursorIcon::ResizeHorizontal)
            });
            if handle.inner.dragged() {
                self.tool_panel_width = (self.tool_panel_width + handle.inner.drag_delta().x).clamp(240.0, 460.0);
            }
            self.tool_panel_rect = Some(rect);
        }
        if self.show_objects {
            // Under the view cube, at the right; folded, just its title.
            let top = if self.view.show_view_cube { 118.0 } else { margin };
            let r = egui::Area::new(egui::Id::NULL.with("floating_objects"))
                .pivot(egui::Align2::RIGHT_TOP)
                .fixed_pos(vr.right_top() + egui::vec2(-margin, top))
                .order(egui::Order::Middle)
                .show(&ctx, |ui| {
                    egui::Frame::window(ui.style()).show(ui, |ui| {
                        let title = if self.objects_folded { format!("Objects ({})", self.scene.objects.len()) } else { "Objects".to_string() };
                        let strong = ui.visuals().strong_text_color();
                        // As tall as its content, scrolling once it reaches the bottom of the
                        // view. (An area offers its content last frame's size; allow it to grow.)
                        let max_h = (vr.height() - top - margin - 20.0).max(80.0);
                        if !self.objects_folded {
                            ui.set_width(self.objects_width);
                            ui.set_max_height(max_h);
                        }
                        let header = egui::CollapsingHeader::new(RichText::new(title).strong().color(strong))
                            .id_salt("objects_header")
                            .open(Some(!self.objects_folded))
                            .show(ui, |ui| {
                                egui::ScrollArea::vertical().max_height(max_h - 28.0).show_viewport(ui, |ui, viewport| self.objects_panel(ui, Some(viewport)));
                            });
                        let hint = if self.objects_folded { "Show the objects" } else { "Fold the object browser away" };
                        if header.header_response.on_hover_text(format!("{hint} (View → Object browser or Cmd+Shift+O hides it altogether)")).clicked() {
                            self.objects_folded = !self.objects_folded;
                        }
                    });
                });
            // Drag the left edge to change the width.
            if !self.objects_folded {
                let rect = r.response.rect;
                let edge = egui::Rect::from_min_max(egui::pos2(rect.left() - 4.0, rect.top()), egui::pos2(rect.left() + 4.0, rect.bottom()));
                let handle = egui::Area::new(egui::Id::NULL.with("objects_panel_edge")).fixed_pos(edge.min).order(egui::Order::Middle).show(&ctx, |ui| {
                    ui.allocate_response(edge.size(), egui::Sense::drag()).on_hover_cursor(egui::CursorIcon::ResizeHorizontal)
                });
                if handle.inner.dragged() {
                    self.objects_width = (self.objects_width - handle.inner.drag_delta().x).clamp(OBJECTS_WIDTH.0, OBJECTS_WIDTH.1);
                }
            }
        }
        egui::CentralPanel::default().frame(egui::Frame::NONE).show(ui, |ui| {
            let rect = ui.available_rect_before_wrap();
            ctx.data_mut(|d| d.insert_temp(egui::Id::NULL.with("vp_rect"), rect));
            self.viewport(ui, frame);
        });
        self.handle_keys(&ctx, vp_prev);
        self.help_window(&ctx);
        self.preferences_window(&ctx);
        self.slices.window(&ctx);
        self.finish_import(&ctx);
        self.recovery_window(&ctx);
        self.hotbox_ui(&ctx);
        self.run_script(&ctx, vp_prev);
        // An edit made above (a button released) is seen by the next frame's tick, which may
        // only come when the save is due: ask for that frame now. While a button is held the
        // save waits for its release, which brings a frame of its own.
        if let Some(wait) = self.autosave_wait(ctx.input(|i| i.time)) {
            if wait > 0.0 || !ctx.input(|i| i.pointer.any_down()) {
                ctx.request_repaint_after(std::time::Duration::from_secs_f64(wait));
            }
        }
    }
}

/// The open-boundary line vertices of `m` (two per boundary edge, in `boundary_edge_ids`
/// order), from `cache` when it holds them for this state of the mesh. When only vertices
/// moved (a brush stroke), they are rebuilt from the cached edges instead of a walk over every
/// edge of the mesh.
fn boundary_lines<'a>(cache: &'a mut BoundaryLines, m: &Mesh) -> &'a [ColorVertex] {
    const BLUE: Vec4 = Vec4::new(0.25, 0.65, 1.0, 1.0);
    let key = (m.shape_version(), m as *const Mesh as usize);
    if cache.lines.as_ref().is_none_or(|(k, _)| *k != key) {
        let topo = m.topology_version();
        if cache.edges.as_ref().is_none_or(|(t, _)| *t != topo) {
            let mut edges = cache.edges.take().map(|(_, e)| e).unwrap_or_default();
            edges.clear();
            edges.extend(m.boundary_edge_ids());
            cache.edges = Some((topo, edges));
        }
        let edges = cache.edges.as_ref().map_or(&[][..], |(_, e)| e.as_slice());
        let mut v = cache.lines.take().map(|(_, v)| v).unwrap_or_default();
        v.clear();
        for &e in edges {
            let [a, b] = m.edge_vertices(e);
            v.push(ColorVertex::new(m.vertex(a).as_vec3(), BLUE));
            v.push(ColorVertex::new(m.vertex(b).as_vec3(), BLUE));
        }
        cache.lines = Some((key, v));
    }
    cache.lines.as_ref().map_or(&[], |(_, v)| v.as_slice())
}

/// The grid's spacing: `fixed` (mm) when set, else "nice" for a scene `size` across (1, 2 or 5
/// times a power of ten, about a twentieth of it).
fn grid_step(size: f64, fixed: f64) -> f64 {
    if fixed > 0.0 {
        return fixed;
    }
    let raw = size.max(1e-6) / 20.0;
    let pow = 10f64.powf(raw.log10().floor());
    [1.0, 2.0, 5.0, 10.0].into_iter().map(|m| m * pow).find(|s| *s >= raw).unwrap_or(10.0 * pow)
}

/// Floor grid under the scene, perpendicular to the up axis, spaced `fixed` mm apart (0:
/// chosen from the scene's size), every fifth line brighter.
fn grid_lines(camera: &Camera, b: &mm_core::Aabb, fixed: f64) -> Vec<ColorVertex> {
    if b.is_empty() {
        return Vec::new();
    }
    let up = camera.world_up();
    let size = b.diagonal().max(1e-6);
    // At most 400 lines each way (a fine grid under a large scene covers its middle).
    let step = grid_step(size, fixed).max(size * 0.75 / 400.0);
    // In-plane axes and the floor height (just below the scene).
    let (u, v) = if up == glam::DVec3::Z { (glam::DVec3::X, glam::DVec3::Y) } else { (glam::DVec3::X, glam::DVec3::Z) };
    let floor = b.min.dot(up) - size * 1e-3;
    let c = b.center();
    let half = (size * 0.75 / step).ceil() * step;
    let (cu, cv) = ((c.dot(u) / step).round() * step, (c.dot(v) / step).round() * step);
    let n = (half / step).round() as i64;
    let minor = Vec4::new(0.6, 0.6, 0.62, 0.22);
    let major = Vec4::new(0.7, 0.7, 0.72, 0.45);
    let mut out = Vec::with_capacity((4 * (2 * n + 1)) as usize);
    for k in -n..=n {
        let col = if (k + ((cu / step).round() as i64)).rem_euclid(5) == 0 { major } else { minor };
        let a = up * floor + u * (cu + k as f64 * step) + v * (cv - half);
        let z = up * floor + u * (cu + k as f64 * step) + v * (cv + half);
        out.push(ColorVertex::new(a.as_vec3(), col));
        out.push(ColorVertex::new(z.as_vec3(), col));
        let col = if (k + ((cv / step).round() as i64)).rem_euclid(5) == 0 { major } else { minor };
        let a = up * floor + v * (cv + k as f64 * step) + u * (cu - half);
        let z = up * floor + v * (cv + k as f64 * step) + u * (cu + half);
        out.push(ColorVertex::new(a.as_vec3(), col));
        out.push(ColorVertex::new(z.as_vec3(), col));
    }
    out
}

#[cfg(test)]
mod grid_tests {
    use super::*;

    #[test]
    fn grid_spacing_fixed_or_from_the_scene() {
        assert_eq!(grid_step(100.0, 0.0), 5.0);
        assert_eq!(grid_step(30.0, 0.0), 2.0);
        assert_eq!(grid_step(100.0, 2.5), 2.5);
        // A fine grid under a large scene stays at 400 lines each way.
        let b = mm_core::Aabb { min: glam::DVec3::splat(-500.0), max: glam::DVec3::splat(500.0) };
        let lines = grid_lines(&Camera::default(), &b, 0.01);
        assert!(lines.len() <= 4 * 801 + 8, "{}", lines.len());
    }
}

#[cfg(test)]
mod flip_tests {
    use super::*;

    #[test]
    fn flip_zy_round_trips_and_keeps_orientation() {
        let p = glam::DVec3::new(1.0, 2.0, 3.0);
        assert_eq!(flip_zy_out(flip_zy_in(p)), p);
        // File +Z (up) becomes scene +Y.
        assert_eq!(flip_zy_in(glam::DVec3::Z), glam::DVec3::Y);
        let mut m = mm_core::primitives::icosphere(1.0, 2);
        let v0 = mm_core::measure::volume(&m);
        m.transform(flip_zy_in);
        assert!((mm_core::measure::volume(&m) - v0).abs() < 1e-12, "a rotation keeps the outward orientation");
    }

    #[test]
    fn odd_sizes() {
        let mut m = mm_core::primitives::cube(0.05, 1);
        assert!(odd_size(&m), "5 cm in metres");
        m.transform(|p| p * 1000.0);
        assert!(!odd_size(&m), "50 mm");
        m.transform(|p| p * 1000.0);
        assert!(odd_size(&m), "50 m in millimetres");
    }
}

#[cfg(test)]
mod stats_tests {
    use super::*;
    use crate::scene::SceneObject;

    /// The panel's text for `o` as it was drawn before the cache: every value computed there.
    fn uncached(o: &SceneObject) -> Vec<String> {
        let m = &*o.mesh;
        let e = m.bounds().extents();
        let open = m.boundary_edge_ids().count();
        let c = mm_core::measure::center_of_mass(m);
        let mut rows = vec![format!("{:.2} × {:.2} × {:.2}", e.x, e.y, e.z), open.to_string()];
        if open == 0 {
            rows.push(format!("{:.3}", mm_core::measure::volume(m)));
        }
        rows.push(format!("{:.2}, {:.2}, {:.2}", c.x, c.y, c.z));
        rows.push(format!("{:.3}", mm_core::measure::surface_area(m)));
        rows
    }

    fn cached(cache: &mut StatsCache, o: &SceneObject) -> Vec<String> {
        let s = cache.get(o);
        let (e, c) = (s.extents, s.center_of_mass);
        let mut rows = vec![format!("{:.2} × {:.2} × {:.2}", e.x, e.y, e.z), s.open.to_string()];
        if let Some(v) = s.volume {
            rows.push(format!("{v:.3}"));
        }
        rows.push(format!("{:.2}, {:.2}, {:.2}", c.x, c.y, c.z));
        rows.push(format!("{:.3}", s.area));
        rows
    }

    #[test]
    fn statistics_follow_strokes_and_edits_and_objects() {
        let mut cache = StatsCache::default();
        let mut sphere = SceneObject::new(1, "sphere", mm_core::primitives::icosphere(3.0, 3));
        let mut open = SceneObject::new(2, "open", mm_core::primitives::cylinder(1.0, 4.0, 24, 6, false));
        // Exact values too, not only the text: centre of mass and volume as the old calls gave.
        let s = cache.get(&sphere);
        assert_eq!(s.center_of_mass.to_array().map(f64::to_bits), mm_core::measure::center_of_mass(&sphere.mesh).to_array().map(f64::to_bits));
        assert_eq!(s.volume.map(f64::to_bits), Some(mm_core::measure::volume(&sphere.mesh).to_bits()));
        for step in 0..12u32 {
            // A dab moves vertices (same topology); then an edit that changes it.
            for o in [&mut sphere, &mut open] {
                let m = o.mesh_mut();
                let v = (step * 31) % m.max_vid() as u32;
                let p = m.vertex(v);
                m.set_vertex(v, p * 1.3 + glam::DVec3::new(0.5, 0.0, -0.25));
            }
            assert_eq!(cached(&mut cache, &sphere), uncached(&sphere), "step {step}: sphere");
            assert_eq!(cached(&mut cache, &open), uncached(&open), "step {step}: open");
            if step % 4 == 3 {
                let m = sphere.mesh_mut();
                let t = m.triangle_ids().nth(step as usize).unwrap();
                m.remove_triangle(t, true).unwrap();
            }
            assert_eq!(cached(&mut cache, &sphere), uncached(&sphere), "step {step}: after the edit");
        }
        assert!(cache.get(&sphere).volume.is_none(), "holes cut: no volume");
        // A stroke on one object: the open-edge count is kept, the rest follows the shape.
        let topo = open.mesh.topology_version();
        for v in 0..5u32 {
            let p = open.mesh.vertex(v);
            open.mesh_mut().set_vertex(v, p * 0.7);
            assert_eq!(cached(&mut cache, &open), uncached(&open), "dab {v}");
            assert_eq!(cache.open.map(|(id, t, _)| (id, t)), Some((open.id, topo)));
        }
    }

    /// Every value of `s`, as bits.
    fn bits(s: &ShapeStats) -> ([u64; 3], usize, Option<u64>, [u64; 3], u64) {
        (s.extents.to_array().map(f64::to_bits), s.open, s.volume.map(f64::to_bits), s.center_of_mass.to_array().map(f64::to_bits), s.area.to_bits())
    }

    #[test]
    fn unchanged_frames_and_undo_use_the_right_values() {
        let mut cache = StatsCache::default();
        let mut sphere = SceneObject::new(1, "sphere", mm_core::primitives::icosphere(3.0, 3));
        // Frames that change nothing get the cached values, bit for bit, under the same key.
        let first = cache.get(&sphere);
        let key = Some((sphere.id, sphere.mesh.shape_version()));
        for _ in 0..3 {
            assert_eq!(bits(&cache.get(&sphere)), bits(&first));
            assert_eq!(cache.shape.map(|(id, v, _)| (id, v)), key);
        }
        // They are not computed again: positions written without a new version (which no
        // edit does) are not seen.
        let mut moved = sphere.clone();
        moved.mesh_mut().positions_mut()[0] *= 2.0;
        assert_eq!(bits(&cache.get(&moved)), bits(&first));
        // Undo puts back the mesh the step before held, with its older version: after a cached
        // edit, and after an edit the panel never showed.
        let before = sphere.mesh.clone();
        for shown in [true, false] {
            let p = sphere.mesh.vertex(5);
            sphere.mesh_mut().set_vertex(5, p * 1.5);
            if shown {
                assert_eq!(cached(&mut cache, &sphere), uncached(&sphere));
            }
            sphere.mesh = before.clone();
            assert_eq!(cached(&mut cache, &sphere), uncached(&sphere), "undone (shown: {shown})");
            assert_eq!(bits(&cache.get(&sphere)), bits(&first));
            assert_eq!(cache.shape.map(|(id, v, _)| (id, v)), key);
        }
        // Undo of a cut: the open-edge count follows the topology back.
        let t = sphere.mesh.triangle_ids().next().unwrap();
        sphere.mesh_mut().remove_triangle(t, true).unwrap();
        assert_eq!(cached(&mut cache, &sphere), uncached(&sphere));
        assert!(cache.get(&sphere).open > 0);
        sphere.mesh = before;
        assert_eq!(cached(&mut cache, &sphere), uncached(&sphere));
        assert_eq!(bits(&cache.get(&sphere)), bits(&first));
    }
}

#[cfg(test)]
mod autosave_tests {
    use super::*;
    use std::time::{Duration, Instant};

    /// One frame at input time `time` running the autosave's tick; the wait before the next
    /// frame it asks for.
    fn frame(ws: &mut MeshWorkspace, ctx: &egui::Context, time: f64) -> Duration {
        let out = ctx.run_ui(egui::RawInput { time: Some(time), ..Default::default() }, |ui| ws.autosave_tick(ui.ctx()));
        out.viewport_output[&egui::ViewportId::ROOT].repaint_delay
    }

    #[test]
    fn saves_come_when_due_without_input() {
        let (mut ws, ctx, _) = super::command::tests::workspace();
        let dir = std::env::temp_dir().join(format!("polymix-autosave-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("scratch folder");
        let file = dir.join(format!("autosave.{}", crate::project::EXTENSION));
        ws.prefs.autosave_minutes = 1;
        ws.autosave = Some(Autosave { file: file.clone(), marker: dir.join("session.lock"), saved_generation: ws.history.generation(), last_time: 0.0, job: None });
        // Nothing to save: no frame asked for (after egui's own first frames).
        for _ in 0..3 {
            frame(&mut ws, &ctx, 1.0);
        }
        assert_eq!(frame(&mut ws, &ctx, 1.0), Duration::MAX);
        // An edit: a frame is asked for when the save is due.
        ws.scene.add("cube", mm_core::primitives::cube(10.0, 2));
        ws.history.push("Add", &ws.scene);
        assert_eq!(ws.autosave_wait(20.0), Some(40.0));
        assert_eq!(frame(&mut ws, &ctx, 20.0), Duration::from_secs(40));
        // A save running (held until told to finish) while another edit comes: nothing is
        // scheduled meanwhile.
        let (finish, held) = std::sync::mpsc::channel::<()>();
        let job = Job::spawn("Autosave", ctx.clone(), move |_| {
            let _ = held.recv();
            Ok(())
        });
        let generation = ws.history.generation();
        if let Some(a) = &mut ws.autosave {
            (a.job, a.saved_generation, a.last_time) = (Some(job), generation, 20.0);
        }
        ws.history.push("Edit", &ws.scene);
        assert_eq!(ws.autosave_wait(30.0), None);
        frame(&mut ws, &ctx, 30.0);
        // It ends after the next save is due: the frame that sees it end starts the next one,
        // so the slot is never empty in between.
        finish.send(()).expect("save waiting");
        let start = Instant::now();
        while ws.autosave.as_ref().is_some_and(|a| a.saved_generation != ws.history.generation()) {
            assert!(start.elapsed() < Duration::from_secs(20), "the next save starts");
            std::thread::sleep(Duration::from_millis(1));
            frame(&mut ws, &ctx, 90.0);
            assert!(ws.autosave.as_ref().is_some_and(|a| a.job.is_some()), "the next save starts in the same frame");
        }
        assert_eq!(ws.autosave.as_ref().map(|a| a.last_time), Some(90.0));
        // That save is written, and with nothing left to save no frame is asked for.
        while ws.autosave.as_ref().is_some_and(|a| a.job.is_some()) {
            assert!(start.elapsed() < Duration::from_secs(20), "the save finishes");
            std::thread::sleep(Duration::from_millis(1));
            frame(&mut ws, &ctx, 91.0);
        }
        assert!(file.exists());
        assert_eq!(ws.autosave_wait(91.0), None);
        for _ in 0..3 {
            frame(&mut ws, &ctx, 92.0);
        }
        assert_eq!(frame(&mut ws, &ctx, 92.0), Duration::MAX);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_due_save_waits_for_a_held_button() {
        let (mut ws, ctx, _) = super::command::tests::workspace();
        let dir = std::env::temp_dir().join(format!("polymix-autosave-held-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("scratch folder");
        let file = dir.join(format!("autosave.{}", crate::project::EXTENSION));
        ws.prefs.autosave_minutes = 1;
        ws.autosave = Some(Autosave { file: file.clone(), marker: dir.join("session.lock"), saved_generation: ws.history.generation(), last_time: 0.0, job: None });
        ws.scene.add("cube", mm_core::primitives::cube(10.0, 2));
        ws.history.push("Add", &ws.scene);
        // Due, but the primary button is down (a stroke): nothing starts, nothing is recorded.
        let press = |down: bool| egui::RawInput {
            time: Some(70.0),
            events: vec![egui::Event::PointerMoved(egui::pos2(10.0, 10.0)), egui::Event::PointerButton { pos: egui::pos2(10.0, 10.0), button: PointerButton::Primary, pressed: down, modifiers: Default::default() }],
            ..Default::default()
        };
        let _ = ctx.run_ui(press(true), |ui| ws.autosave_tick(ui.ctx()));
        assert!(ws.autosave.as_ref().is_some_and(|a| a.job.is_none() && a.last_time == 0.0 && a.saved_generation != ws.history.generation()));
        // Released: the save starts.
        let _ = ctx.run_ui(press(false), |ui| ws.autosave_tick(ui.ctx()));
        assert!(ws.autosave.as_ref().is_some_and(|a| a.job.is_some() && a.last_time == 70.0 && a.saved_generation == ws.history.generation()));
        let start = Instant::now();
        while ws.autosave.as_ref().is_some_and(|a| a.job.is_some()) {
            assert!(start.elapsed() < Duration::from_secs(20), "the save finishes");
            std::thread::sleep(Duration::from_millis(1));
            frame(&mut ws, &ctx, 71.0);
        }
        // Written (fast compression) and read back.
        let p = crate::project::load(&file).expect("autosave readable");
        assert_eq!(p.objects.len(), 1);
        assert_eq!(p.objects[0].mesh.triangle_count(), ws.scene.objects[0].mesh.triangle_count());
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// A tool that only reports being busy or not (for tests of what the app does around tools).
#[cfg(test)]
struct TestTool(bool);

#[cfg(test)]
impl Tool for TestTool {
    fn name(&self) -> &'static str {
        "Test"
    }
    fn panel(&mut self, _ui: &mut egui::Ui, _cx: &mut ToolCx) -> ToolOutcome {
        ToolOutcome::Continue
    }
    fn busy(&self) -> Option<(String, f32)> {
        self.0.then(|| ("Computing".to_string(), 0.5))
    }
}

#[cfg(test)]
mod boundary_tests {
    use super::*;

    /// The lines as drawn before the cache: from a walk over every edge of the mesh.
    fn walked(m: &Mesh) -> Vec<u8> {
        let blue = Vec4::new(0.25, 0.65, 1.0, 1.0);
        let mut v = Vec::new();
        for e in m.boundary_edge_ids() {
            let [a, b] = m.edge_vertices(e);
            v.push(ColorVertex::new(m.vertex(a).as_vec3(), blue));
            v.push(ColorVertex::new(m.vertex(b).as_vec3(), blue));
        }
        bytemuck::cast_slice(&v).to_vec()
    }

    fn lines(cache: &mut BoundaryLines, m: &Mesh) -> Vec<u8> {
        bytemuck::cast_slice(boundary_lines(cache, m)).to_vec()
    }

    #[test]
    fn boundary_lines_follow_strokes_cuts_and_previews() {
        let mut m = mm_core::primitives::cylinder(1.0, 4.0, 24, 6, false);
        let mut cache = BoundaryLines::default();
        assert_eq!(lines(&mut cache, &m), walked(&m));
        let topo = m.topology_version();
        let edges = cache.edges.as_ref().map(|(_, e)| e.clone()).expect("edges kept");
        assert!(!edges.is_empty());
        // Dabs move vertices and keep the topology: the same edges, the lines follow.
        for step in 0..10u32 {
            let v = (step * 13) % m.max_vid() as u32;
            let p = m.vertex(v);
            m.set_vertex(v, p * 1.1 + DVec3::new(0.0, 0.1, 0.0));
            assert_eq!(lines(&mut cache, &m), walked(&m), "dab {step}");
            assert_eq!(cache.edges.as_ref().map(|(t, e)| (*t, e.clone())), Some((topo, edges.clone())), "dab {step}");
        }
        // Holes cut: another topology, its own edges.
        for k in 0..3 {
            let t = m.triangle_ids().nth(5 + k * 7).expect("a triangle");
            m.remove_triangle(t, true).expect("removed");
            assert_eq!(lines(&mut cache, &m), walked(&m), "cut {k}");
        }
        assert_ne!(cache.edges.as_ref().map(|(t, _)| *t), Some(topo));
        // A preview (another mesh, of the same topology) and back.
        let mut preview = m.clone();
        preview.transform(|p| p * 2.0 + DVec3::X);
        assert_eq!(lines(&mut cache, &preview), walked(&preview));
        assert_eq!(lines(&mut cache, &m), walked(&m));
        // A closed mesh has none.
        assert!(boundary_lines(&mut cache, &mm_core::primitives::icosphere(1.0, 2)).is_empty());
        // An unchanged mesh is served from the cache, not walked again.
        lines(&mut cache, &m);
        if let Some((_, v)) = cache.lines.as_mut() {
            v[0].color[0] = 9.0;
        }
        assert_eq!(boundary_lines(&mut cache, &m)[0].color[0], 9.0);
    }
}

#[cfg(test)]
mod export_tests {
    use super::*;
    use std::time::{Duration, Instant};

    /// Run the export job to its end; the status it leaves.
    fn finish(ws: &mut MeshWorkspace) -> String {
        let start = Instant::now();
        while ws.export_job.is_some() {
            assert!(start.elapsed() < Duration::from_secs(60), "the export finishes");
            std::thread::sleep(Duration::from_millis(1));
            ws.poll_jobs();
        }
        ws.status.clone()
    }

    /// Export as it was done before the job made the meshes: built here, then written. The
    /// interim message.
    fn export_before(ws: &MeshWorkspace, path: &std::path::Path, scope: ExportScope) -> String {
        let active = ws.scene.active().expect("active");
        let mut mesh = match scope {
            ExportScope::AllVisible => {
                let mut m = Mesh::new();
                for o in ws.scene.objects.iter().filter(|o| o.visible && !o.reference) {
                    m.append_mesh(&o.mesh, 0);
                }
                m
            }
            ExportScope::Active => (*active.mesh).clone(),
            ExportScope::Selection => {
                let tris: Vec<u32> = active.selection.iter().collect();
                active.mesh.submesh(&tris).0
            }
            _ => unreachable!(),
        };
        if ws.prefs.flip_zy {
            mesh.transform(flip_zy_out);
        }
        let uvs = match scope {
            ExportScope::Active => active.uvs().map(|u| active.mesh.triangle_ids().map(|t| u.corners[t as usize]).collect()),
            _ => None,
        };
        let opts = mm_io::WriteOptions { stl_ascii: ws.export_ascii, obj_groups: ws.prefs.obj_export_groups, uvs, ..Default::default() };
        mm_io::write_mesh(path, &mesh, &opts).expect("written");
        format!("Exporting {} triangles to {}", mesh.triangle_count(), path.display())
    }

    /// One file per region or object, as before.
    fn sequence_before(ws: &MeshWorkspace, path: &std::path::Path, scope: ExportScope) {
        let safe = |s: &str| -> String { s.chars().map(|c| if c.is_alphanumeric() || "-_ ".contains(c) { c } else { '_' }).collect::<String>().trim().replace(' ', "_") };
        let stem = path.file_stem().map_or("mesh".to_string(), |s| s.to_string_lossy().to_string());
        let ext = path.extension().map_or("stl".to_string(), |e| e.to_string_lossy().to_string());
        let dir = path.parent().map(|p| p.to_path_buf()).unwrap_or_default();
        let mut parts: Vec<(String, Mesh)> = Vec::new();
        match scope {
            ExportScope::PerGroup => {
                let o = ws.scene.active().expect("active");
                let mut groups: std::collections::BTreeMap<i32, Vec<u32>> = Default::default();
                for t in o.mesh.triangle_ids() {
                    groups.entry(o.mesh.tri_group(t)).or_default().push(t);
                }
                for (g, tris) in groups {
                    let name = o.group_names.get(&g).cloned().unwrap_or_else(|| format!("group_{g}"));
                    parts.push((name, o.mesh.submesh(&tris).0));
                }
            }
            _ => {
                for o in ws.scene.objects.iter().filter(|o| o.visible && !o.reference) {
                    parts.push((o.name.clone(), (*o.mesh).clone()));
                }
            }
        }
        let opts = mm_io::WriteOptions { stl_ascii: ws.export_ascii, obj_groups: ws.prefs.obj_export_groups, ..Default::default() };
        let mut seen = std::collections::HashSet::new();
        for (name, mut m) in parts {
            if ws.prefs.flip_zy {
                m.transform(flip_zy_out);
            }
            let mut base = format!("{stem}_{}", safe(&name));
            let mut k = 2;
            while !seen.insert(base.clone()) {
                base = format!("{stem}_{}_{k}", safe(&name));
                k += 1;
            }
            mm_io::write_mesh(&dir.join(format!("{base}.{ext}")), &m, &opts).expect("written");
        }
    }

    /// The files of a folder, by name.
    fn files(dir: &std::path::Path) -> Vec<(String, Vec<u8>)> {
        let mut f: Vec<(String, Vec<u8>)> = std::fs::read_dir(dir).expect("folder").flatten().map(|e| (e.file_name().to_string_lossy().into_owned(), std::fs::read(e.path()).expect("file"))).collect();
        f.sort();
        f
    }

    #[test]
    fn exports_write_the_same_files_as_before() {
        let (mut ws, ctx, _) = super::command::tests::workspace();
        // A coloured, grouped mesh with holes (sparse triangle ids), UVs and a selection; a
        // second object, a hidden one and a reference.
        let mut a = mm_core::primitives::icosphere(5.0, 3);
        for t in a.triangle_ids().collect::<Vec<_>>() {
            let g = (a.tri_centroid(t).z > 1.0) as i32 + 2 * (a.tri_centroid(t).x > 0.0) as i32;
            a.set_tri_group(t, g);
        }
        for t in a.triangle_ids().step_by(41).collect::<Vec<_>>() {
            a.remove_triangle(t, true).expect("removed");
        }
        let v = a.vertex_ids().nth(3).expect("a vertex");
        a.set_vertex_color(v, [0.2, 0.4, 0.6]);
        let corners: Vec<[[f32; 2]; 3]> = (0..a.max_tid()).map(|t| [[t as f32 * 0.001, 0.5], [0.25, -0.0], [1.0, (t % 7) as f32 / 7.0]]).collect();
        let positions = a.positions().to_vec();
        let a_id = ws.scene.add("vessel wall", a);
        ws.scene.add("box", mm_core::primitives::cube(3.0, 3));
        let hidden = ws.scene.add("hidden", mm_core::primitives::cube(1.0, 1));
        let reference = ws.scene.add("guide", mm_core::primitives::icosphere(2.0, 1));
        ws.scene.get_mut(hidden).expect("object").visible = false;
        ws.scene.get_mut(reference).expect("object").reference = true;
        let o = ws.scene.get_mut(a_id).expect("object");
        o.uvs = Some(std::sync::Arc::new(crate::scene::UvSet::new(&o.mesh, corners)));
        o.group_names.insert(1, "inlet 1".into());
        o.group_names.insert(3, "inlet/1".into());
        o.selection = mm_core::TriSelection::from_tris(&o.mesh, o.mesh.triangle_ids().filter(|t| t % 3 == 0).collect::<Vec<_>>());
        ws.scene.active = Some(a_id);
        let root = std::env::temp_dir().join(format!("polymix-export-test-{}", std::process::id()));
        for flip in [false, true] {
            ws.prefs.flip_zy = flip;
            for (k, (scope, ext)) in [(ExportScope::Active, "obj"), (ExportScope::Active, "stl"), (ExportScope::AllVisible, "stl"), (ExportScope::AllVisible, "obj"), (ExportScope::Selection, "ply"), (ExportScope::PerGroup, "stl"), (ExportScope::PerObject, "off")].into_iter().enumerate() {
                let (now, before) = (root.join(format!("{flip}-{k}-now")), root.join(format!("{flip}-{k}-before")));
                std::fs::create_dir_all(&now).expect("folder");
                std::fs::create_dir_all(&before).expect("folder");
                let name = format!("part.{ext}");
                let msg = ws.export_to(now.join(&name), scope, &ctx).expect("export starts");
                let end = finish(&mut ws);
                if matches!(scope, ExportScope::PerGroup | ExportScope::PerObject) {
                    sequence_before(&ws, &before.join(&name), scope);
                    assert!(end.starts_with("Exported "), "{end}");
                } else {
                    let msg_before = export_before(&ws, &before.join(&name), scope);
                    assert_eq!(msg.replace(&now.display().to_string(), ""), msg_before.replace(&before.display().to_string(), ""), "{scope:?} {ext}");
                    assert!(end.starts_with(&msg.replace("Exporting", "Exported")), "{end}");
                }
                let (a, b) = (files(&now), files(&before));
                assert!(!a.is_empty());
                assert_eq!(a.iter().map(|f| &f.0).collect::<Vec<_>>(), b.iter().map(|f| &f.0).collect::<Vec<_>>(), "{scope:?} {ext} flip {flip}");
                assert!(a == b, "{scope:?} {ext} flip {flip}: the files differ");
            }
        }
        // The scene's meshes are untouched (shared with the jobs, never turned in place).
        assert_eq!(ws.scene.get(a_id).expect("object").mesh.positions(), &positions[..]);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn region_counts() {
        let mut m = mm_core::primitives::cube(2.0, 4);
        assert_eq!(distinct_groups(&m), 1);
        for t in m.triangle_ids().collect::<Vec<_>>() {
            m.set_tri_group(t, (t % 5) as i32 - 2);
        }
        assert_eq!(distinct_groups(&m), 5);
        assert_eq!(distinct_groups(&Mesh::new()), 0);
    }
}

#[cfg(test)]
mod import_tests {
    use super::*;

    /// A note without its "(0.12s)" timing.
    fn untimed(note: &str) -> String {
        match (note.find(" ("), note.find("s)")) {
            (Some(a), Some(b)) if a < b => format!("{}{}", &note[..a], &note[b + 2..]),
            _ => note.to_string(),
        }
    }

    type Summary = (String, (Vec<[u64; 3]>, Vec<[u32; 3]>, Vec<i32>), String, u64, Option<usize>, bool);

    fn summary(r: &ImportResult) -> Vec<Summary> {
        r.iter()
            .map(|(name, m, note, d, look)| {
                let (p, t, g) = m.to_arrays();
                (name.clone(), (p.iter().map(|v| v.to_array().map(f64::to_bits)).collect(), t, g), untimed(note), d.to_bits(), look.uvs.as_ref().map(Vec::len), look.reference)
            })
            .collect()
    }

    #[test]
    fn files_are_read_side_by_side_in_file_order() {
        let dir = std::env::temp_dir().join(format!("polymix-import-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("folder");
        let w = mm_io::WriteOptions::default();
        let mut grouped = mm_core::primitives::cube(4.0, 3);
        for t in grouped.triangle_ids().collect::<Vec<_>>() {
            grouped.set_tri_group(t, (t % 3) as i32);
        }
        let mut files = Vec::new();
        for (k, (m, ext)) in [(mm_core::primitives::icosphere(2.0, 3), "stl"), (grouped, "obj"), (mm_core::primitives::cylinder(1.0, 3.0, 16, 4, true), "ply"), (mm_core::primitives::cube(0.01, 2), "off"), (mm_core::primitives::icosphere(9000.0, 2), "stl")].into_iter().enumerate() {
            let f = dir.join(format!("m{k}.{ext}"));
            mm_io::write_mesh(&f, &m, &mm_io::WriteOptions { obj_groups: true, ..w.clone() }).expect("written");
            files.push(f);
        }
        let bad = |k: usize| {
            let f = dir.join(format!("bad{k}.stl"));
            std::fs::write(&f, b"solid nothing here\n").expect("written");
            f
        };
        let (bad1, bad2) = (bad(1), bad(2));
        for (groups, reference) in [(true, false), (false, true)] {
            let one = import_files(&files, groups, reference, 1, &|_| true).expect("read");
            assert_eq!(one.len(), files.len());
            for (r, f) in one.iter().zip(&files) {
                assert_eq!(r.3.to_bits(), r.1.bounds().diagonal().to_bits(), "{}", f.display());
            }
            for workers in [2, 3, 8] {
                let many = import_files(&files, groups, reference, workers, &|_| true).expect("read");
                assert!(summary(&many) == summary(&one), "{workers} workers");
            }
            assert_eq!(one[1].1.triangle_ids().map(|t| one[1].1.tri_group(t)).max(), Some(if groups { 2 } else { 0 }));
        }
        // The first failure in file order, however many read at once.
        let mixed = vec![files[0].clone(), bad1.clone(), files[1].clone(), bad2, files[2].clone()];
        let first = format!("{:#}", import_files(&mixed, true, false, 1, &|_| true).err().expect("fails"));
        assert!(first.contains(&bad1.display().to_string()), "{first}");
        for workers in [2, 5] {
            assert_eq!(format!("{:#}", import_files(&mixed, true, false, workers, &|_| true).err().expect("fails")), first);
        }
        // Cancelled.
        for workers in [1, 3] {
            assert_eq!(import_files(&files, true, false, workers, &|_| false).err().expect("cancelled").to_string(), "cancelled");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn reads_side_by_side_share_one_pool_and_a_panic_is_their_error() {
        use rayon::prelude::*;
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get().saturating_sub(1).max(1));
        // Each read, and its own parallel loops, in the one pool of a job's size.
        let r = side_by_side(12, 4, &|_| true, |i| {
            let inner: Vec<(usize, Option<String>)> = (0..64).into_par_iter().map(|_| (rayon::current_num_threads(), std::thread::current().name().map(str::to_string))).collect();
            Ok((i, inner))
        })
        .expect("read");
        for (k, (i, inner)) in r.iter().enumerate() {
            assert_eq!(*i, k);
            for (n, name) in inner {
                assert_eq!(*n, threads);
                assert!(name.as_deref().is_some_and(|n| n.starts_with("polymix-job-read-")), "{name:?}");
            }
        }
        // A read that panics fails as a job that panics does, unless one before it failed.
        let e = side_by_side(9, 3, &|_| true, |i| if i == 4 { panic!("broken file {i}") } else { Ok(i) }).expect_err("fails");
        assert_eq!(format!("{e:#}"), "internal error: broken file 4");
        let e = side_by_side(9, 3, &|_| true, |i| match i {
            2 => Err(anyhow::anyhow!("bad file 2")),
            6 => panic!("broken file 6"),
            _ => Ok(i),
        })
        .expect_err("fails");
        assert_eq!(format!("{e:#}"), "bad file 2");
    }

    #[test]
    fn odd_sizes_from_the_diagonal() {
        for m in [mm_core::primitives::cube(0.05, 1), mm_core::primitives::cube(50.0, 1), mm_core::primitives::cube(50_000.0, 1), Mesh::new()] {
            assert_eq!(odd_diagonal(m.bounds().diagonal()), odd_size(&m));
        }
    }
}

#[cfg(test)]
mod bvh_tests {
    use super::*;
    use std::time::{Duration, Instant};

    fn frame(ws: &mut MeshWorkspace, ctx: &egui::Context) {
        let _ = ctx.run_ui(egui::RawInput::default(), |ui| ws.bvh_tick(ui.ctx()));
    }

    /// Frames until the background build is done and installed.
    fn built(ws: &mut MeshWorkspace, ctx: &egui::Context) {
        let start = Instant::now();
        while ws.bvh_job.is_some() {
            assert!(start.elapsed() < Duration::from_secs(60), "the build finishes");
            std::thread::sleep(Duration::from_millis(1));
            frame(ws, ctx);
        }
    }

    fn same(a: &mm_core::Bvh, b: &mm_core::Bvh) -> bool {
        a.tris == b.tris && format!("{:?}", a.nodes) == format!("{:?}", b.nodes)
    }

    #[test]
    fn commits_get_their_bvh_in_the_background() {
        let (mut ws, ctx, _) = super::command::tests::workspace();
        ws.history.push("Add", &ws.scene);
        let id = ws.scene.add("ball", mm_core::primitives::icosphere(3.0, 4));
        let hidden = ws.scene.add("hidden", mm_core::primitives::cube(1.0, 2));
        ws.scene.get_mut(hidden).expect("object").visible = false;
        frame(&mut ws, &ctx);
        assert!(ws.bvh_job.is_some(), "started after the commit");
        built(&mut ws, &ctx);
        let o = ws.scene.get(id).expect("object");
        assert!(same(o.bvh_if_current().expect("installed"), &mm_core::Bvh::build(&o.mesh)), "the BVH a pick would build");
        assert!(ws.scene.get(hidden).expect("object").bvh_if_current().is_none(), "hidden objects are not picked");
        // Nothing committed since: nothing to do.
        frame(&mut ws, &ctx);
        assert!(ws.bvh_job.is_none());
        // A stroke (vertices moved, the same topology): the BVH is refitted where it is used.
        ws.history.push("Stroke", &ws.scene);
        let m = ws.scene.get_mut(id).expect("object").mesh_mut();
        let p = m.vertex(0);
        m.set_vertex(0, p * 1.2);
        frame(&mut ws, &ctx);
        assert!(ws.bvh_job.is_none(), "a refit, not a build");
        // A cut (new topology) while a tool is open: built once the tool is closed.
        ws.history.push("Cut", &ws.scene);
        let m = ws.scene.get_mut(id).expect("object").mesh_mut();
        let t = m.triangle_ids().next().expect("a triangle");
        m.remove_triangle(t, true).expect("removed");
        ws.tools.push(Box::new(TestTool(false)));
        frame(&mut ws, &ctx);
        assert!(ws.bvh_job.is_none(), "not while a tool is open");
        ws.tools.clear();
        frame(&mut ws, &ctx);
        assert!(ws.bvh_job.is_some());
        built(&mut ws, &ctx);
        let o = ws.scene.get(id).expect("object");
        assert!(same(o.bvh_if_current().expect("installed"), &mm_core::Bvh::build(&o.mesh)));
        // A BVH that is stale by the time it is done is turned away.
        ws.history.push("Cut again", &ws.scene);
        let m = ws.scene.get_mut(id).expect("object").mesh_mut();
        let t = m.triangle_ids().nth(10).expect("a triangle");
        m.remove_triangle(t, true).expect("removed");
        frame(&mut ws, &ctx);
        assert!(ws.bvh_job.is_some());
        let m = ws.scene.get_mut(id).expect("object").mesh_mut();
        let p = m.vertex(5);
        m.set_vertex(5, p * 0.9);
        built(&mut ws, &ctx);
        assert!(ws.scene.get(id).expect("object").bvh_if_current().is_none(), "built for an earlier state");
    }

    #[test]
    fn a_build_stops_once_its_meshes_are_replaced() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let (mut ws, ctx, _) = super::command::tests::workspace();
        let ball = ws.scene.add("ball", mm_core::primitives::icosphere(3.0, 3));
        let block = ws.scene.add("block", mm_core::primitives::cube(1.0, 2));
        let meshes = || ws.scene.objects.iter().map(|o| (o.id, o.mesh.clone())).collect::<Vec<_>>();
        // Builds not begun once stopped are left out.
        let built_now = build_bvhs(meshes(), &AtomicBool::new(false));
        assert_eq!(built_now.len(), 2);
        for (id, bvh) in &built_now {
            assert!(same(bvh, &mm_core::Bvh::build(&ws.scene.get(*id).expect("object").mesh)));
        }
        assert!(build_bvhs(meshes(), &AtomicBool::new(true)).is_empty());
        // A build under way (held until told to end), for both meshes.
        let held = |ws: &mut MeshWorkspace| {
            let (go, wait) = std::sync::mpsc::channel::<()>();
            let shapes = ws.scene.objects.iter().map(|o| (o.id, o.mesh.shape_version())).collect();
            let stop = std::sync::Arc::new(AtomicBool::new(false));
            let job = Job::spawn("Preparing picking", ctx.clone(), move |_| {
                let _ = wait.recv();
                Ok(Vec::new())
            });
            ws.bvh_job = Some(BvhJob { shapes, stop: stop.clone(), job });
            ws.bvh_generation = ws.history.generation();
            (go, stop)
        };
        let (go, stop) = held(&mut ws);
        frame(&mut ws, &ctx);
        frame(&mut ws, &ctx);
        assert!(!stop.load(Ordering::Relaxed), "nothing replaced");
        // One of them edited (a vertex moved): the rest is not begun.
        let m = ws.scene.get_mut(block).expect("object").mesh_mut();
        let p = m.vertex(0);
        m.set_vertex(0, p * 2.0);
        frame(&mut ws, &ctx);
        assert!(stop.load(Ordering::Relaxed), "an edited mesh");
        go.send(()).expect("job waiting");
        built(&mut ws, &ctx);
        // Or removed.
        let (go, stop) = held(&mut ws);
        ws.scene.remove(ball);
        frame(&mut ws, &ctx);
        assert!(stop.load(Ordering::Relaxed), "a removed object");
        go.send(()).expect("job waiting");
        built(&mut ws, &ctx);
    }
}

#[cfg(test)]
mod repaint_tests {
    use super::*;
    use std::time::Duration;

    /// One whole frame of the workspace (no GPU) a second after the last; the wait before the
    /// next frame it asks for.
    fn frame(ws: &mut MeshWorkspace, ctx: &egui::Context) -> Duration {
        let mut frame = eframe::Frame::_new_kittest();
        // Time goes on, so that animations (a panel unfolding) end.
        let time = ctx.input(|i| i.time) + 1.0;
        let input = egui::RawInput { time: Some(time), screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1200.0, 800.0))), ..Default::default() };
        let out = ctx.run_ui(input, |ui| polymix_workspace::Workspace::ui(ws, ui, &mut frame));
        out.viewport_output[&egui::ViewportId::ROOT].repaint_delay
    }

    fn settled(ws: &mut MeshWorkspace, ctx: &egui::Context) -> Duration {
        for _ in 0..6 {
            frame(ws, ctx);
        }
        frame(ws, ctx)
    }

    #[test]
    fn work_in_progress_is_redrawn_15_times_a_second() {
        let (mut ws, ctx, _) = super::command::tests::workspace();
        ws.scene.add("cube", mm_core::primitives::cube(10.0, 2));
        let idle = settled(&mut ws, &ctx);
        assert!(idle > BUSY_REPAINT, "{idle:?}");
        // A tool computing: its progress overlay.
        ws.tools.push(Box::new(TestTool(true)));
        assert_eq!(settled(&mut ws, &ctx), BUSY_REPAINT);
        ws.tools.clear();
        // Files loading: the overlay and the status bar's spinner.
        let (go, wait) = std::sync::mpsc::channel::<()>();
        ws.import_job = Some(Job::spawn("Importing 1 file(s)", ctx.clone(), move |_| {
            let _ = wait.recv();
            Ok(Vec::new())
        }));
        assert_eq!(settled(&mut ws, &ctx), BUSY_REPAINT);
        go.send(()).expect("job waiting");
    }
}

#[cfg(test)]
mod library_tests {
    use super::*;
    use std::time::{Duration, Instant};

    /// Poll the jobs until the library load is over.
    fn loaded(ws: &mut MeshWorkspace) {
        let start = Instant::now();
        while ws.library_job.is_some() {
            assert!(start.elapsed() < Duration::from_secs(60), "the load ends");
            std::thread::sleep(Duration::from_millis(1));
            ws.poll_jobs();
        }
    }

    #[test]
    fn reloads_meanwhile_and_failed_loads_are_minded() {
        let (mut ws, ctx, _) = super::command::tests::workspace();
        let changes = crate::parts::library_changes();
        // Reload clicked while a load runs (held until told to end): no second load meanwhile,
        // and its result is shown but out of date (loaded again while the panel is shown).
        let (go, wait) = std::sync::mpsc::channel::<()>();
        ws.library_job = Some((changes, Job::spawn("Loading parts", ctx.clone(), move |_| {
            let _ = wait.recv();
            Ok(crate::parts::Library::default())
        })));
        ws.reload_library(&ctx);
        assert!(ws.library_job.as_ref().is_some_and(|(c, _)| *c == u64::MAX));
        go.send(()).expect("job waiting");
        loaded(&mut ws);
        assert!(ws.library.is_some());
        assert!(ws.library_load_due(changes));
        // A load that fails: not tried again frame after frame, only once the files change
        // (or on Reload).
        ws.library_job = Some((changes, Job::spawn("Loading parts", ctx.clone(), |_| -> anyhow::Result<crate::parts::Library> { panic!("unreadable parts") })));
        loaded(&mut ws);
        assert_eq!(ws.library_failed, Some(changes));
        assert!(!ws.library_load_due(changes));
        assert!(ws.library_load_due(changes + 1));
        // Its next load, held, fails after Reload was clicked: tried once more.
        let (go, wait) = std::sync::mpsc::channel::<()>();
        ws.library_job = Some((changes, Job::spawn("Loading parts", ctx.clone(), move |_| -> anyhow::Result<crate::parts::Library> {
            let _ = wait.recv();
            panic!("unreadable parts")
        })));
        ws.reload_library(&ctx);
        go.send(()).expect("job waiting");
        loaded(&mut ws);
        assert!(ws.library_load_due(changes));
        // Loaded at last.
        ws.library_job = Some((changes, Job::spawn("Loading parts", ctx.clone(), |_| Ok(crate::parts::Library::default()))));
        loaded(&mut ws);
        assert_eq!(ws.library_failed, None);
        assert!(!ws.library_load_due(changes));
    }
}

#[cfg(test)]
mod browser_tests {
    use super::*;

    /// The object browser in a scroll area 300 points high scrolled to `offset`, drawn with only
    /// the rows in view or `all` of them, given `events`: the content's size, the number of
    /// widgets made (widgets out of view paint nothing, but are laid out and sensed) and the
    /// number of checkboxes in the AccessKit tree.
    fn draw(ws: &mut MeshWorkspace, ctx: &egui::Context, offset: f32, all: bool, events: Vec<egui::Event>) -> (egui::Vec2, usize, usize) {
        let mut size = egui::Vec2::ZERO;
        let input = egui::RawInput { screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(400.0, 800.0))), events, ..Default::default() };
        let out = ctx.run_ui(input, |ui| {
            let area = egui::ScrollArea::vertical().max_height(300.0).vertical_scroll_offset(offset);
            size = area.show_viewport(ui, |ui, viewport| ws.objects_panel(ui, (!all).then_some(viewport))).content_size;
        });
        let widgets = ctx.viewport(|v| v.prev_pass.widgets.layers().map(|(_, w)| w.len()).sum());
        let checkboxes = out.platform_output.accesskit_update.map_or(0, |u| u.nodes.iter().filter(|(_, n)| n.role() == egui::accesskit::Role::CheckBox).count());
        (size, widgets, checkboxes)
    }

    fn many() -> (MeshWorkspace, egui::Context) {
        let (mut ws, ctx, _) = super::command::tests::workspace();
        for k in 0..500 {
            ws.scene.add(format!("shell {k}"), mm_core::primitives::cube(1.0, 1));
        }
        ws.scene.get_mut(ws.scene.objects[3].id).expect("object").reference = true;
        ws.scene.target = Some(ws.scene.objects[7].id);
        (ws, ctx)
    }

    #[test]
    fn rows_out_of_view_take_their_room_undrawn() {
        let (mut ws, ctx) = many();
        let (full, full_widgets, _) = draw(&mut ws, &ctx, 0.0, true, Vec::new());
        assert!(ws.objects_row_h.is_some());
        let ends = [0.0, 1.0, 2000.0, 5555.5, full.y - 300.0, full.y];
        for offset in ends {
            let (size, widgets, _) = draw(&mut ws, &ctx, offset, false, Vec::new());
            assert_eq!(size, full, "offset {offset}");
            assert!(widgets * 5 < full_widgets, "offset {offset}: {widgets} of {full_widgets} widgets");
            assert_eq!(draw(&mut ws, &ctx, offset, true, Vec::new()).0, full);
        }
        // With AccessKit on (an agent reading the panels, a screen reader) every row is there,
        // wherever the list is scrolled.
        ctx.enable_accesskit();
        for offset in ends {
            let (size, widgets, checkboxes) = draw(&mut ws, &ctx, offset, false, Vec::new());
            assert_eq!((size, widgets, checkboxes), (full, full_widgets, ws.scene.objects.len()), "offset {offset}");
        }
        ctx.disable_accesskit();
        // While a name is edited every row is drawn.
        let id = ws.scene.objects[250].id;
        ctx.data_mut(|d| d.insert_temp(egui::Id::NULL.with("renaming"), (id, "shell 250".to_string(), true)));
        let (size, widgets, _) = draw(&mut ws, &ctx, 3000.0, false, Vec::new());
        assert!(widgets * 2 > full_widgets, "{widgets} of {full_widgets}");
        assert!((size.y - full.y).abs() < 30.0, "{size:?} {full:?}");
    }

    #[test]
    fn a_context_menu_stays_open_while_its_row_is_scrolled_away() {
        let (mut ws, ctx) = many();
        let (full, full_widgets, _) = draw(&mut ws, &ctx, 0.0, true, Vec::new());
        // Right-click the third row's name (just right of its checkbox).
        let step = ws.objects_row_h.expect("rows measured") + ctx.global_style().spacing.item_spacing.y;
        let at = egui::pos2(40.0, 2.5 * step);
        let button = |pressed| egui::Event::PointerButton { pos: at, button: PointerButton::Secondary, pressed, modifiers: Default::default() };
        for events in [vec![egui::Event::PointerMoved(at)], vec![button(true)], vec![button(false)]] {
            draw(&mut ws, &ctx, 0.0, false, events);
        }
        assert!(egui::Popup::is_any_open(&ctx), "the menu opened");
        for offset in [1000.0, 3000.0, full.y - 300.0] {
            let (size, widgets, _) = draw(&mut ws, &ctx, offset, false, Vec::new());
            assert!(egui::Popup::is_any_open(&ctx), "offset {offset}: the menu is still open");
            assert_eq!(size, full);
            assert!(widgets >= full_widgets, "all rows drawn: {widgets} of {full_widgets}");
        }
        // Closed (Esc): back to the rows in view.
        draw(&mut ws, &ctx, 3000.0, false, vec![egui::Event::Key { key: Key::Escape, physical_key: None, pressed: true, repeat: false, modifiers: Default::default() }]);
        assert!(!egui::Popup::is_any_open(&ctx));
        let (_, widgets, _) = draw(&mut ws, &ctx, 3000.0, false, Vec::new());
        assert!(widgets * 5 < full_widgets, "{widgets} of {full_widgets}");
    }
}
