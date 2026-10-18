//! Application shell: layout, viewport navigation, tool dispatch, file IO.

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
}

impl ColorMode {
    pub const ALL: [ColorMode; 4] = [ColorMode::Vertex, ColorMode::Group, ColorMode::Constant, ColorMode::UvChecker];
    pub fn label(self) -> &'static str {
        match self {
            ColorMode::Constant => "Constant color",
            ColorMode::Group => "Group colors",
            ColorMode::Vertex => "Vertex colors",
            ColorMode::UvChecker => "UV checker",
        }
    }
    fn key(self) -> &'static str {
        match self {
            ColorMode::Constant => "constant",
            ColorMode::Group => "group",
            ColorMode::Vertex => "vertex",
            ColorMode::UvChecker => "uv",
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
}

/// Where autosaves and the session marker live (per user).
pub(crate) fn data_dir() -> Option<PathBuf> {
    // Test runs point this elsewhere so they never touch the user's files.
    if let Some(d) = std::env::var_os("POLYMIX_DATA_DIR") {
        return Some(PathBuf::from(d));
    }
    let home = std::env::var_os("HOME").map(PathBuf::from);
    if cfg!(target_os = "macos") {
        return home.map(|h| h.join("Library/Application Support/Polymix"));
    }
    if cfg!(windows) {
        return std::env::var_os("APPDATA").map(|d| PathBuf::from(d).join("Polymix"));
    }
    std::env::var_os("XDG_DATA_HOME").map(PathBuf::from).or_else(|| home.map(|h| h.join(".local/share"))).map(|d| d.join("polymix"))
}

/// Autosave state: the files, what was saved last, and the job writing it.
struct Autosave {
    file: PathBuf,
    marker: PathBuf,
    saved_generation: u64,
    last_time: f64,
    job: Option<Job<()>>,
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
struct PendingImport {
    meshes: Vec<(String, Mesh, String)>,
    /// Append / Replace decided (or not needed).
    placement: Option<bool /*replace*/>,
    remember: bool,
}

/// Size outside the plausible range for millimetre models (bounding-box diagonal).
fn odd_size(m: &Mesh) -> bool {
    let d = m.bounds().diagonal();
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
    pub show_grid: bool,
    /// Draw the target surface see-through.
    pub transparent_target: bool,
    /// Background colour at the top of the view.
    pub background: [f32; 3],
    pub show_view_cube: bool,
    pub theme: ui_util::Theme,
    /// Draw objects' pivots (Meshmixer's Hide/Show Pivots).
    pub show_pivots: bool,
}

impl Default for ViewOptions {
    fn default() -> Self {
        Self { wireframe: false, normals: NormalMode::Group, colors: ColorMode::Vertex, boundaries: true, show_stats: true, sharp_edges: true, crease_deg: DEFAULT_CREASE_DEG, nav_mode: NavMode::Fusion360, nav_custom: NavBindings::FUSION360, brush_scroll: BrushScroll::Shift, show_grid: false, transparent_target: true, background: renderer::DEFAULT_BACKGROUND, show_view_cube: true, theme: ui_util::Theme::Dark, show_pivots: true }
    }
}

const DEFAULT_CREASE_DEG: f32 = 40.0;

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

/// Create a tool by name. Shared by the palette, hotkeys, menus and test scripts.
fn make_tool(name: &str, cx: &mut ToolCx) -> Option<Box<dyn Tool>> {
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
        "boolean" => Box::new(tools::ops::BooleanTool::new(cx)),
        "bridge" => Box::new(tools::ops::BridgeTool::new(cx)),
        "deviation" => Box::new(tools::deviation::DeviationTool::new(cx)),
        "cfdcheck" => Box::new(tools::cfd_check::CfdCheckTool::new(cx)),
        "centerline" => Box::new(tools::centerline_tool::CenterlineTool::new(cx)),
        _ => return None,
    })
}

type ImportResult = Vec<(String, Mesh, String)>;
type BoundaryKey = (ObjectId, u64, usize);

pub struct PolymixApp {
    scene: Scene,
    history: History,
    camera: Camera,
    renderer: Option<SceneRenderer>,
    gpu: HashMap<ObjectId, GpuMesh>,
    preview_gpu: Option<GpuMesh>,
    view: ViewOptions,
    /// Tool stack: the top tool is active; closing a sub-tool returns to the one below.
    tools: Vec<Box<dyn Tool>>,
    category: Option<Category>,
    /// Meshmix parts (loaded when the category opens).
    library: Option<crate::parts::Library>,
    /// Tool switch waiting for the current tool to finish computing (then it is accepted).
    pending_switch: Option<SwitchTo>,
    status: String,
    import_job: Option<Job<ImportResult>>,
    pending_import: Option<PendingImport>,
    /// Scripts normally skip the import prompts; `prompts:on` shows them (for screenshots).
    script_prompts: bool,
    export_job: Option<Job<String>>,
    /// Project being opened, and the file the scene was last saved to / opened from.
    project_job: Option<(PathBuf, Job<crate::project::Project>)>,
    project_path: Option<PathBuf>,
    autosave: Option<Autosave>,
    /// An autosave left by a session that did not exit cleanly.
    recovery: Option<PathBuf>,
    nav_drag: Option<PointerButton>,
    /// Cmd/Ctrl+middle-drag resizing the brush.
    brush_drag: bool,
    /// When the brush last took a wheel notch (its eased remainder must not zoom).
    brush_wheel_at: f64,
    /// Point the current navigation drag orbits about (picked when it started).
    orbit_pivot: Option<DVec3>,
    /// Centre of mass of an object, for the mesh version it was computed for.
    com_cache: Option<(ObjectId, u64, DVec3)>,
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
    /// 3D mouse (3Dconnexion) input and its settings window.
    space: SpaceMouse,
    space_cfg: spacemouse::Settings,
    show_space_window: bool,
    /// Space-bar hotbox position (Meshmixer-style quick view settings).
    hotbox: Option<egui::Pos2>,
    show_objects: bool,
    prefs: Prefs,
    show_prefs: bool,
    prefs_tab: usize,
    last_export_dir: Option<PathBuf>,
    export_ascii: bool,
    window_title: String,
    /// Where the floating tool panel was last drawn (keeps viewport decorations clear of it).
    tool_panel_rect: Option<egui::Rect>,
    /// Width of the floating tool panel (drag its right edge).
    tool_panel_width: f32,
    recent: Vec<PathBuf>,
    script: Option<Script>,
    /// Boundary-edge line vertices per object, keyed by (object, mesh version, mesh address).
    boundary_cache: HashMap<ObjectId, (BoundaryKey, Vec<ColorVertex>)>,
    /// Synthetic pointer events queued by scripts (consumed one per frame).
    synthetic: std::collections::VecDeque<PointerInput>,
}

/// Scripted UI actions for automated testing and screenshots (`POLYMIX_SCRIPT`), e.g.
/// `wait:20;tool:inspector;wait:5;shot:/tmp/a.png;quit`.
struct Script {
    steps: std::collections::VecDeque<String>,
    wait: u32,
    awaiting_shot: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    /// Key presses (`key:S`) for the next frame's input.
    keys: Vec<egui::Key>,
}

impl PolymixApp {
    pub fn new(cc: &eframe::CreationContext<'_>, files: Vec<PathBuf>) -> Self {
        ui_util::apply_theme(&cc.egui_ctx, ui_util::Theme::Dark);
        let renderer = cc.wgpu_render_state.as_ref().map(SceneRenderer::new);
        let mut app = Self {
            scene: Scene::default(),
            history: History::default(),
            camera: Camera::default(),
            renderer,
            gpu: HashMap::new(),
            preview_gpu: None,
            view: ViewOptions::default(),
            tools: Vec::new(),
            category: None,
            library: None,
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
            brush_wheel_at: f64::NEG_INFINITY,
            orbit_pivot: None,
            com_cache: None,
            tool_rmb: None,
            view_history: ViewHistory::default(),
            view_anim: None,
            last_wheel: 0.0,
            space_moving: false,
            last_pointer: None,
            show_help: false,
            space: SpaceMouse::start(&cc.egui_ctx),
            space_cfg: spacemouse::Settings::default(),
            show_space_window: false,
            hotbox: None,
            show_objects: true,
            prefs: Prefs::default(),
            show_prefs: false,
            prefs_tab: 0,
            last_export_dir: None,
            export_ascii: false,
            window_title: String::new(),
            tool_panel_rect: None,
            tool_panel_width: 300.0,
            recent: Vec::new(),
            synthetic: Default::default(),
            boundary_cache: HashMap::new(),
            script: std::env::var("POLYMIX_SCRIPT").ok().map(|s| Script {
                steps: s.split(';').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect(),
                wait: 0,
                awaiting_shot: None,
                keys: Vec::new(),
            }),
        };
        if let Some(st) = cc.storage.filter(|_| std::env::var_os("POLYMIX_SCRIPT").is_none()) {
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
            app.view.transparent_target = flag("transparent_target", true);
            app.view.show_view_cube = flag("view_cube", true);
            app.view.show_pivots = flag("pivots", true);
            if let Some(t) = st.get_string("theme").and_then(|k| ui_util::Theme::from_key(&k)) {
                app.view.theme = t;
                ui_util::apply_theme(&cc.egui_ctx, t);
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
            if let Some(cfg) = st.get_string("spacemouse").and_then(|v| spacemouse::Settings::from_pref(&v)) {
                app.space_cfg = cfg;
            }
            app.view.crease_deg = st.get_string("crease_deg").and_then(|v| v.parse().ok()).unwrap_or(DEFAULT_CREASE_DEG);
            app.show_objects = flag("show_objects", true);
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
        if !files.is_empty() {
            app.start_import(files, &cc.egui_ctx);
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
        let minutes = self.prefs.autosave_minutes;
        let generation = self.history.generation();
        let now = ctx.input(|i| i.time);
        let busy = self.recovery.is_some();
        let Some(auto) = &mut self.autosave else { return };
        if let Some(job) = &auto.job {
            if let Some(r) = job.poll() {
                auto.job = None;
                if let Err(e) = r {
                    log::warn!("autosave failed: {e:#}");
                }
            }
            return;
        }
        if busy || minutes == 0 || generation == auto.saved_generation || self.scene.is_empty() || now - auto.last_time < minutes as f64 * 60.0 {
            return;
        }
        auto.saved_generation = generation;
        auto.last_time = now;
        let objects = self
            .scene
            .objects
            .iter()
            .map(|o| crate::project::ProjectObject {
                name: o.name.clone(),
                mesh: (*o.mesh).clone(),
                visible: o.visible,
                color: o.color,
                material: o.material,
                frame: o.frame,
                group_names: o.group_names.iter().map(|(g, n)| (*g, n.clone())).collect(),
                uvs: o.uvs().map(|u| o.mesh.triangle_ids().map(|t| u.corners[t as usize]).collect()),
                pivots: o.pivots.clone(),
            })
            .collect();
        let project = crate::project::Project { objects, active: None, target: None, view: Some(self.camera.view_state()) };
        let file = auto.file.clone();
        auto.job = Some(Job::spawn("Autosave", ctx.clone(), move |_| crate::project::save(&file, &project)));
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
        self.import_job = Some(Job::spawn(format!("Importing {n} file(s)"), ctx.clone(), move |progress| {
            let mut out = Vec::new();
            for (i, f) in files.iter().enumerate() {
                if !progress(i as f32 / n as f32) {
                    anyhow::bail!("cancelled");
                }
                let t0 = std::time::Instant::now();
                let mut imported = mm_io::read_mesh(f).map_err(|e| anyhow::anyhow!("{}: {e:#}", f.display()))?;
                if !obj_groups && imported.format == mm_io::Format::Obj {
                    for t in imported.mesh.triangle_ids().collect::<Vec<_>>() {
                        imported.mesh.set_tri_group(t, 0);
                    }
                }
                let rep = &imported.report;
                let mut notes = format!(
                    "{}: {} triangles, {} vertices ({:.2}s)",
                    imported.name,
                    imported.mesh.triangle_count(),
                    imported.mesh.vertex_count(),
                    t0.elapsed().as_secs_f32()
                );
                if rep.nonmanifold_split + rep.degenerate + rep.duplicate > 0 {
                    notes += &format!(
                        " — fixed on load: {} non-manifold, {} degenerate, {} duplicate triangles",
                        rep.nonmanifold_split, rep.degenerate, rep.duplicate
                    );
                }
                out.push((imported.name, imported.mesh, notes));
            }
            Ok(out)
        }));
    }

    fn poll_jobs(&mut self) {
        if let Some(job) = &self.import_job {
            if let Some(res) = job.poll() {
                self.import_job = None;
                match res {
                    Ok(meshes) => {
                        let placement = match self.prefs.import_mode {
                            _ if self.scene.is_empty() => Some(false),
                            ImportMode::Ask => None,
                            ImportMode::Append => Some(false),
                            ImportMode::Replace => Some(true),
                        };
                        self.pending_import = Some(PendingImport { meshes, placement, remember: false });
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
    }

    /// Import prompts (Append / Replace, odd sizes), then add the meshes to the scene.
    fn finish_import(&mut self, ctx: &egui::Context) {
        // Test scripts never wait on prompts: they append and keep sizes.
        let scripted = self.script.is_some() && !self.script_prompts;
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
        let odd = pending.meshes.iter().any(|(_, m, _)| odd_size(m));
        let factor = match self.prefs.scale_mode {
            _ if !odd || scripted => Some(1.0),
            ScaleMode::Never => Some(1.0),
            ScaleMode::Always(k) => Some(k),
            ScaleMode::Ask => {
                let d = pending.meshes.iter().map(|(_, m, _)| m.bounds().diagonal()).fold(0.0, f64::max);
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
        for (name, mut mesh, mut note) in pending.meshes {
            if self.prefs.flip_zy {
                mesh.transform(flip_zy_in);
            }
            if factor != 1.0 && odd_size(&mesh) {
                mesh.transform(|p| p * factor);
                note += &format!(" — scaled × {factor}");
            }
            self.scene.add(name, mesh);
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
        self.project_job = Some((path, Job::spawn("Opening project", ctx.clone(), move |_| crate::project::load(&p))));
    }

    fn apply_project(&mut self, p: crate::project::Project, path: PathBuf) {
        self.history.push("Open project", &self.scene);
        self.scene.objects.clear();
        self.scene.active = None;
        self.scene.target = None;
        let mut ids = Vec::new();
        for o in p.objects {
            let id = self.scene.add(o.name, o.mesh);
            if let Some(so) = self.scene.get_mut(id) {
                so.visible = o.visible;
                so.color = o.color;
                so.material = o.material;
                so.frame = o.frame;
                so.group_names = o.group_names.into_iter().collect();
                so.uvs = o.uvs.map(|c| std::sync::Arc::new(crate::scene::UvSet::new(&so.mesh, c)));
                so.pivots = o.pivots;
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
        let index = |id: Option<ObjectId>| id.and_then(|id| self.scene.objects.iter().position(|o| o.id == id));
        let objects: Vec<crate::project::ProjectObject> = self
            .scene
            .objects
            .iter()
            .map(|o| {
                let mut groups: Vec<(i32, String)> = o.group_names.iter().map(|(g, n)| (*g, n.clone())).collect();
                groups.sort();
                crate::project::ProjectObject {
                    name: o.name.clone(),
                    mesh: (*o.mesh).clone(),
                    visible: o.visible,
                    color: o.color,
                    material: o.material,
                    frame: o.frame,
                    group_names: groups,
                    uvs: o.uvs().map(|u| o.mesh.triangle_ids().map(|t| u.corners[t as usize]).collect()),
                    pivots: o.pivots.clone(),
                }
            })
            .collect();
        let project = crate::project::Project {
            objects,
            active: index(self.scene.active),
            target: index(self.scene.target),
            view: Some(self.camera.view_state()),
        };
        self.project_path = Some(path.clone());
        self.export_job = Some(Job::spawn("Saving", ctx.clone(), move |_| {
            crate::project::save(&path, &project)?;
            Ok(format!("Saved {}", path.display()))
        }));
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
        let mesh: Mesh = match scope {
            ExportScope::AllVisible => {
                let mut m = Mesh::new();
                for o in self.scene.objects.iter().filter(|o| o.visible) {
                    m.append_mesh(&o.mesh, 0);
                }
                m
            }
            ExportScope::Active => (*active.mesh).clone(),
            ExportScope::Selection => {
                let tris: Vec<u32> = active.selection.iter().collect();
                active.mesh.submesh(&tris).0
            }
        };
        let mesh = if self.prefs.flip_zy {
            let mut m = mesh;
            m.transform(flip_zy_out);
            m
        } else {
            mesh
        };
        let p2 = path.clone();
        // UVs go out with the active object (OBJ).
        let uvs = match scope {
            ExportScope::Active => active.uvs().map(|u| active.mesh.triangle_ids().map(|t| u.corners[t as usize]).collect()),
            _ => None,
        };
        let opts = mm_io::WriteOptions { stl_ascii: self.export_ascii, obj_groups: self.prefs.obj_export_groups, uvs, ..Default::default() };
        self.export_job = Some(Job::spawn("Exporting", ctx.clone(), move |_| {
            mm_io::write_mesh(&p2, &mesh, &opts)?;
            Ok(format!("Exported {} triangles to {}", mesh.triangle_count(), p2.display()))
        }));
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
                    // `drop:<part name>` opens Drop Part with a library part.
                    let part = name.strip_prefix("drop:").and_then(|p| self.library.get_or_insert_with(crate::parts::Library::load).find(p));
                    let mut cx = self.tool_cx(ctx, vp);
                    let tool: Option<Box<dyn Tool>> = match part {
                        Some(p) => Some(Box::new(tools::drop_part::DropPartTool::new(&mut cx, p))),
                        None => make_tool(&name, &mut cx),
                    };
                    if let Some(t) = tool {
                        self.tools.push(t);
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
                    // Pick up parts saved since the last look.
                    self.library = None;
                }
                self.category = c;
            }
        }
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
                ui.menu_button("Add shape", |ui| {
                    // Meshmixer's Import Sphere / Plane (no bunny: Polymix ships no sample scans).
                    type Make = fn() -> Mesh;
                    let shapes: [(&str, Make); 4] = [
                        ("Sphere", || mm_core::primitives::icosphere(12.5, 4)),
                        ("Plane", || mm_core::primitives::plane_grid(25.0, 25.0, 32, 32)),
                        ("Cube", || mm_core::primitives::cube(25.0, 8)),
                        ("Cylinder", || {
                            let mut m = mm_core::primitives::cylinder(12.5, 25.0, 64, 16, true);
                            m.transform(|p| glam::DVec3::new(p.x, p.z - 12.5, -p.y));
                            m
                        }),
                    ];
                    for (name, make) in shapes {
                        if ui.button(name).clicked() {
                            ui.close();
                            self.history.push(format!("Add {name}"), &self.scene);
                            let first = self.scene.is_empty();
                            self.scene.add(name.to_lowercase(), make());
                            if first || self.prefs.reset_camera_on_import {
                                self.focus_scene();
                            }
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
                    self.history.push("Clear scene", &self.scene);
                    self.scene.objects.clear();
                    self.scene.active = None;
                    self.scene_changed(&ctx, vp);
                }
            });
            ui.menu_button("Edit", |ui| {
                let ul = self.history.undo_label().map(|s| format!("Undo {s}")).unwrap_or("Undo".into());
                if ui.add_enabled(self.history.can_undo(), egui::Button::new(format!("{ul}   ⌘Z"))).clicked() {
                    ui.close();
                    self.undo(&ctx);
                }
                let rl = self.history.redo_label().map(|s| format!("Redo {s}")).unwrap_or("Redo".into());
                if ui.add_enabled(self.history.can_redo(), egui::Button::new(format!("{rl}   ⇧⌘Z"))).clicked() {
                    ui.close();
                    self.redo(&ctx);
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
                    if self.accept_tools_for_command(&ctx, vp) {
                        commands::duplicate_object(&mut self.cmd_cx());
                    }
                }
                if ui.button("Combine visible objects").clicked() {
                    ui.close();
                    if self.accept_tools_for_command(&ctx, vp) {
                        commands::combine_visible(&mut self.cmd_cx());
                        self.scene_changed(&ctx, vp);
                    }
                }
                if ui.button("Separate shells").clicked() {
                    ui.close();
                    if self.accept_tools_for_command(&ctx, vp) {
                        commands::separate_shells(&mut self.cmd_cx());
                        self.scene_changed(&ctx, vp);
                    }
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
                    commands::set_target(&mut self.cmd_cx(), active);
                }
                if ui.add_enabled(self.scene.target.is_some(), egui::Button::new("Clear target")).clicked() {
                    ui.close();
                    commands::set_target(&mut self.cmd_cx(), None);
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
                if ui.button("3D mouse…").on_hover_text("3Dconnexion SpaceMouse settings").clicked() {
                    ui.close();
                    self.show_space_window = true;
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
                    self.focus_scene();
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
                ui.menu_button("Standard views", |ui| {
                    use std::f64::consts::{FRAC_PI_2, PI};
                    for (name, yaw, pitch) in [
                        ("Front", 0.0, 0.0),
                        ("Back", PI, 0.0),
                        ("Left", -FRAC_PI_2, 0.0),
                        ("Right", FRAC_PI_2, 0.0),
                        ("Top", 0.0, FRAC_PI_2),
                        ("Bottom", 0.0, -FRAC_PI_2),
                    ] {
                        if ui.button(name).clicked() {
                            ui.close();
                            self.standard_view(yaw, pitch);
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
            let _ = vp;
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
        match cat {
            Category::Meshmix => {
                let lib = self.library.get_or_insert_with(crate::parts::Library::load);
                let mut drop: Option<String> = None;
                for (title, user, kind) in [("Solid parts", false, mm_edit::PartKind::Solid), ("Open parts", false, mm_edit::PartKind::Open), ("My parts", true, mm_edit::PartKind::Solid)] {
                    let items: Vec<&crate::parts::LibPart> = lib.parts.iter().filter(|p| p.file.is_some() == user && (user || p.part.kind == kind)).collect();
                    if items.is_empty() {
                        continue;
                    }
                    ui_util::section(ui, title);
                    ui.horizontal_wrapped(|ui| {
                        for p in items {
                            let (rect, resp) = ui.allocate_exact_size(egui::vec2(72.0, 86.0), egui::Sense::click());
                            let v = ui.visuals();
                            let bg = if resp.hovered() { v.widgets.hovered.bg_fill } else { v.widgets.inactive.bg_fill };
                            ui.painter().rect_filled(rect, 4.0, bg);
                            let img = egui::Rect::from_min_size(rect.min + egui::vec2(6.0, 4.0), egui::vec2(60.0, 60.0));
                            crate::parts::paint_thumbnail(ui.painter(), img, &p.thumb);
                            ui.painter().text(rect.center_bottom() - egui::vec2(0.0, 6.0), egui::Align2::CENTER_BOTTOM, &p.part.name, egui::FontId::proportional(11.0), v.text_color());
                            let kind = if p.part.kind == mm_edit::PartKind::Open { "open part: welded into the surface" } else { "solid part" };
                            if resp.on_hover_text(format!("{} ({kind})", p.part.name)).clicked() && has_obj {
                                drop = Some(format!("drop:{}", p.part.name));
                            }
                        }
                    });
                }
                ui.small("Click a part, then press on the model where it goes and drag to size it. Make your own with Select > Convert To.");
                if let Some(name) = drop {
                    self.switch_to(SwitchTo::Tool(name), &ctx, vp);
                }
            }
            Category::Edit => {
                ui_util::section(ui, "Edit");
                ui.add_enabled_ui(has_obj, |ui| {
                    let cx = self.tool_cx(&ctx, vp);
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
                        commands::separate_shells(&mut CmdCx { scene: cx.scene, history: cx.history, status: cx.status });
                    }
                    if ui.button("Combine visible").clicked() {
                        commands::combine_visible(&mut CmdCx { scene: cx.scene, history: cx.history, status: cx.status });
                    }
                    if ui.button("Generate Complex").on_hover_text("Cells of the overlapping visible closed objects (A − B, A ∩ B, B − A, …), one face group each").clicked() {
                        commands::generate_complex(&mut CmdCx { scene: cx.scene, history: cx.history, status: cx.status });
                    }
                    if ui.button("Split Complex").on_hover_text("One object per face group").clicked() {
                        commands::split_complex(&mut CmdCx { scene: cx.scene, history: cx.history, status: cx.status });
                    }
                    if ui.button("Convex Decomposition").on_hover_text("Approximate the object by convex pieces (new object, one face group per piece)").clicked() {
                        open = Some("convex");
                    }
                    if ui.button("Duplicate").clicked() {
                        commands::duplicate_object(&mut CmdCx { scene: cx.scene, history: cx.history, status: cx.status });
                    }
                    if ui.button("Flip normals").clicked() {
                        commands::flip_selection_normals(&mut CmdCx { scene: cx.scene, history: cx.history, status: cx.status });
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
                    if ui.button("Measure diameter").clicked() {
                        open = Some("measure");
                    }
                    if ui.button("Centerline").on_hover_text("Vessel centerline tree with branch lengths and diameters; export CSV/VTK").clicked() {
                        open = Some("centerline");
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
            self.switch_to(SwitchTo::Tool(name.into()), &ctx, vp);
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

    fn objects_panel(&mut self, ui: &mut egui::Ui) {
        ui_util::section(ui, "Objects");
        let mut delete = None;
        let mut make_active = None;
        let mut solo = None;
        let mut show_all = false;
        let mut duplicate = None;
        let mut new_target: Option<Option<ObjectId>> = None;
        let target = self.scene.target;
        let rename_id = egui::Id::NULL.with("renaming");
        let mut renaming: Option<ObjectId> = ui.ctx().data(|d| d.get_temp(rename_id));
        for o in self.scene.objects.iter_mut() {
            ui.horizontal(|ui| {
                ui.checkbox(&mut o.visible, "").on_hover_text("Visible");
                let active = Some(o.id) == self.scene.active;
                if renaming == Some(o.id) {
                    let r = ui.add(egui::TextEdit::singleline(&mut o.name).desired_width(120.0));
                    r.request_focus();
                    if r.lost_focus() {
                        renaming = None;
                    }
                    return;
                }
                let name = if target == Some(o.id) { format!("{}  ◎ target", o.name) } else { o.name.clone() };
                let label = RichText::new(name).color(if active { ui_util::ACCENT } else { ui.visuals().text_color() });
                let r = ui.selectable_label(active, label).on_hover_text("Click: make active · Double-click: rename · Right-click: more");
                if r.clicked() {
                    make_active = Some(o.id);
                }
                if r.double_clicked() {
                    renaming = Some(o.id);
                }
                r.context_menu(|ui| {
                    if ui.button("Rename").clicked() {
                        renaming = Some(o.id);
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
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.small_button("x").on_hover_text("Delete object").clicked() {
                        delete = Some(o.id);
                    }
                    let mut c = [o.color.x, o.color.y, o.color.z];
                    if ui.color_edit_button_rgb(&mut c).changed() {
                        o.color = Vec4::new(c[0], c[1], c[2], o.color.w);
                    }
                });
            });
        }
        if let Some(id) = make_active {
            self.scene.active = Some(id);
        }
        ui.ctx().data_mut(|d| match renaming {
            Some(id) => {
                d.insert_temp(rename_id, id);
            }
            None => d.remove::<ObjectId>(rename_id),
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
                let b = m.bounds();
                let e = b.extents();
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
                    let open = m.boundary_edge_ids().count();
                    ui.label("Open edges");
                    ui.label(if open == 0 { RichText::new("0 (closed)").color(Color32::from_rgb(120, 200, 120)) } else { RichText::new(open.to_string()).color(Color32::from_rgb(240, 170, 90)) });
                    ui.end_row();
                    if open == 0 {
                        ui.label("Volume");
                        ui.label(format!("{:.3}", mm_core::measure::volume(m)));
                        ui.end_row();
                    }
                    ui.label("Area");
                    ui.label(format!("{:.3}", mm_core::measure::surface_area(m)));
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
                            if let Some(s) = t.brush_size() {
                                ui.add(egui::Slider::new(s, tools::BRUSH_SIZE_RANGE.0..=tools::BRUSH_SIZE_RANGE.1).logarithmic(true).text("size"));
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
                for (i, t) in ["General", "View", "File", "Dialogs"].iter().enumerate() {
                    ui.selectable_value(&mut self.prefs_tab, i, *t);
                }
            });
            ui.separator();
            match self.prefs_tab {
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

    fn focus_scene(&mut self) {
        self.remember_view();
        self.camera.focus(&self.scene.bounds());
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

    /// What an orbit started at `p` turns about: the point of the model under the cursor,
    /// or, off the model, the object's centre of mass.
    fn orbit_pivot_at(&mut self, p: Option<egui::Pos2>, vp: Viewport) -> DVec3 {
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
        }
        self.space_moving = moving;
        // The 3D mouse turns the object about its centre of mass, wherever panning has put it.
        let pivot = if motion.r.iter().any(|r| *r != 0.0) { self.center_of_mass().unwrap_or(self.camera.target) } else { self.camera.target };
        if spacemouse::apply(&mut self.camera, motion, &self.space_cfg, dt, vp.rect.height() as f64, pivot) {
            // Keep animating while the cap is deflected.
            ctx.request_repaint();
        }
        use std::f64::consts::FRAC_PI_2;
        for b in pressed {
            match spacemouse::button_action(b) {
                Some(spacemouse::ButtonAction::Menu) => {
                    self.hotbox = if self.hotbox.is_some() { None } else { Some(vp.rect.center()) };
                }
                Some(spacemouse::ButtonAction::Fit) => self.focus_scene(),
                Some(spacemouse::ButtonAction::Top) => self.standard_view(0.0, FRAC_PI_2),
                Some(spacemouse::ButtonAction::Right) => self.standard_view(FRAC_PI_2, 0.0),
                Some(spacemouse::ButtonAction::Front) => self.standard_view(0.0, 0.0),
                None => {}
            }
        }
    }

    fn space_window(&mut self, ctx: &egui::Context) {
        let mut open = self.show_space_window;
        egui::Window::new("3D mouse").open(&mut open).resizable(false).show(ctx, |ui| {
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
            egui::Grid::new("space_speeds").num_columns(2).show(ui, |ui| {
                for (label, v) in [("Speed", &mut c.speed), ("Pan", &mut c.pan_speed), ("Zoom", &mut c.zoom_speed), ("Rotate", &mut c.rotate_speed)] {
                    ui.label(label);
                    ui.add(egui::Slider::new(v, 0.1..=4.0).logarithmic(true));
                    ui.end_row();
                }
            });
            ui.horizontal(|ui| {
                ui.checkbox(&mut c.lock_rotation, "Lock rotation");
                ui.checkbox(&mut c.lock_translation, "Lock pan/zoom");
            });
            ui_util::section(ui, "Invert");
            ui.horizontal_wrapped(|ui| {
                for (i, name) in spacemouse::Settings::AXIS_NAMES.iter().enumerate() {
                    ui.checkbox(&mut c.invert[i], *name);
                }
            });
            ui_util::section(ui, "Live input");
            let (axis, buttons) = self.space.raw();
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
            ui.small("Buttons: Menu = Hotbox, Fit = focus, Top / Right / Front = views.");
        });
        if open {
            // Keep the live readout moving.
            ctx.request_repaint_after(std::time::Duration::from_millis(50));
        }
        self.show_space_window = open;
    }

    // ------------------------------------------------------------------
    // Viewport
    // ------------------------------------------------------------------

    fn viewport(&mut self, ui: &mut egui::Ui, frame: &eframe::Frame) {
        let rect = ui.available_rect_before_wrap();
        let vp = Viewport { rect };
        let response = ui.allocate_rect(rect, egui::Sense::click_and_drag());
        let ctx = ui.ctx().clone();

        // --- navigation -------------------------------------------------
        let (mods, scroll, mut zoom_delta, pointer_delta, hover) = ctx.input(|i| {
            (i.modifiers, i.smooth_scroll_delta(), i.zoom_delta(), i.pointer.delta(), i.pointer.hover_pos())
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
                self.orbit_pivot = Some(self.orbit_pivot_at(origin, vp));
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
                        let pivot = self.orbit_pivot.unwrap_or(self.camera.target);
                        self.camera.tumble_about(d.x as f64, d.y as f64, pivot);
                    }
                    Some(NavAction::Pan) => self.camera.pan(d.x as f64, d.y as f64, rect.height() as f64),
                    Some(NavAction::Zoom) => self.camera.zoom((1.0 + d.y as f64 * 0.005).max(0.05)),
                    None => {}
                }
            }
            if response.drag_stopped_by(btn) || !ctx.input(|i| i.pointer.button_down(btn)) {
                self.nav_drag = None;
                self.orbit_pivot = None;
            }
        }
        if in_rect {
            // The brush-scroll modifier (Shift by default) + wheel sets brush tools' size, from
            // the raw events (one step per notch, no easing lag); the plain wheel always zooms.
            let now = ctx.input(|i| i.time);
            let mut scroll_y = scroll.y;
            if self.view.brush_scroll.held(mods) {
                let d = ctx.input(|i| crate::navigation::brush_wheel_delta(&i.raw.events));
                if d != 0.0 && self.tool_wheel(&ctx, vp, d) {
                    self.brush_wheel_at = now;
                }
            }
            // egui eases each notch in over ~0.1 s, as a vertical scroll (Ctrl, Option) or a
            // zoom (Cmd); the rest of a notch the brush took must not zoom the view.
            if now - self.brush_wheel_at < 0.25 {
                scroll_y = 0.0;
                zoom_delta = 1.0;
            }
            let mut factor = 1.0f64;
            if scroll_y != 0.0 {
                factor *= (-scroll_y as f64 * 0.0025).exp();
            }
            if zoom_delta != 1.0 {
                factor /= zoom_delta as f64;
            }
            if factor != 1.0 {
                if now - self.last_wheel > 0.6 {
                    self.remember_view();
                }
                self.last_wheel = now;
                let pivot = hover.and_then(|p| {
                    let ray = self.camera.ray_from_ndc(vp.ndc(p), vp.aspect());
                    tools::pick_ray(&mut self.scene, &ray, None).map(|(_, h)| h.point)
                });
                match pivot {
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

        // --- boundaries overlay (cached per mesh version) ---------------------
        let mut lines: Vec<ColorVertex> = Vec::new();
        if self.view.boundaries {
            let blue = Vec4::new(0.25, 0.65, 1.0, 1.0);
            let preview = self.tools.last().and_then(|t| t.preview());
            for o in self.scene.objects.iter().filter(|o| o.visible) {
                let (key, m) = match preview.filter(|(id, _)| *id == o.id) {
                    Some((_, pm)) => ((o.id, u64::MAX - pm.shape_version(), pm as *const Mesh as usize), pm),
                    None => ((o.id, o.mesh.shape_version(), std::sync::Arc::as_ptr(&o.mesh) as usize), &*o.mesh),
                };
                let cached = self.boundary_cache.entry(o.id).or_default();
                if cached.0 != key {
                    let mut v = Vec::new();
                    for e in m.boundary_edge_ids() {
                        let [a, b] = m.edge_vertices(e);
                        v.push(ColorVertex::new(m.vertex(a).as_vec3(), blue));
                        v.push(ColorVertex::new(m.vertex(b).as_vec3(), blue));
                    }
                    *cached = (key, v);
                }
                lines.extend_from_slice(&cached.1);
            }
        }
        if self.view.show_grid {
            lines.extend(grid_lines(&self.camera, &self.scene.bounds()));
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
                    if let Some(g) = &self.preview_gpu {
                        objs.push((g, false));
                    }
                } else if let Some(g) = self.gpu.get(&o.id) {
                    let see_through = self.view.transparent_target && self.scene.target == Some(o.id);
                    objs.push((g, o.color.w < 0.999 || see_through));
                }
            }
            let input = renderer::FrameInput { globals, objects: objs, lines: &lines, overlay_tris: &overlay.tris };
            let tex = self.renderer.as_mut().unwrap().render(rs, size, &input);
            ui.painter().image(tex, rect, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), Color32::WHITE);
        } else {
            ui.painter().rect_filled(rect, 0.0, Color32::from_gray(30));
            ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, "wgpu renderer unavailable", egui::FontId::proportional(16.0), Color32::WHITE);
        }

        // --- 2D overlays ------------------------------------------------------
        let painter = ui.painter_at(rect);
        if let (Some(btn), Some(p)) = (self.nav_drag, self.orbit_pivot) {
            if nav.drag_action(btn, mods) == Some(NavAction::Tumble) {
                if let Some(n) = self.camera.project(p, vp.aspect()) {
                    let c = vp.to_screen(n);
                    painter.circle(c, 4.0, ui_util::ACCENT, egui::Stroke::new(1.5, Color32::from_gray(20)));
                }
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
                ctx.request_repaint();
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
            ctx.request_repaint();
        }
        self.axis_gizmo(&painter, rect);
        if self.view.show_view_cube {
            self.view_cube(ui, rect);
        }
        if self.nav_drag.is_some() || response.dragged() || in_rect {
            // Keep hover-driven overlays (brush cursor) responsive.
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
        let ids: Vec<ObjectId> = self.scene.objects.iter().map(|o| o.id).collect();
        self.gpu.retain(|id, _| ids.contains(id));
        let active = self.scene.active;
        let multiple = self.scene.objects.iter().filter(|o| o.visible).count() > 1;
        let preview = self.tools.last().and_then(|t| t.preview());
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
                material: crate::materials::MATERIALS[o.material.min(crate::materials::MATERIALS.len() - 1)],
            };
            if let Some((pid, pm)) = preview {
                if pid == o.id {
                    let g = self.preview_gpu.get_or_insert_with(|| renderer.create_mesh(&rs.device));
                    renderer.sync_mesh(&rs.device, &rs.queue, g, pm, None, 0, &style, (&mut no_moves, 0), None, None);
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
            renderer.sync_mesh(&rs.device, &rs.queue, g, &o.mesh, Some(&o.selection), o.selection_stamp, &style, (&mut o.moved_vertices, o.moved_from), scalars, uvs);
        }
        if preview.is_none() {
            self.preview_gpu = None;
        }
    }

    // ------------------------------------------------------------------
    // Scripting (testing)
    // ------------------------------------------------------------------

    fn run_script(&mut self, ctx: &egui::Context, vp: Viewport) {
        let busy = self.import_job.is_some() || self.tool_busy() || !self.synthetic.is_empty();
        let Some(script) = &mut self.script else { return };
        // Wait for an in-flight screenshot to be written.
        if let Some(done) = &script.awaiting_shot {
            if !done.load(std::sync::atomic::Ordering::SeqCst) {
                ctx.request_repaint();
                return;
            }
            script.awaiting_shot = None;
        }
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
            "wait" => script.wait = arg.parse().unwrap_or(1),
            "key" => {
                if let Some(k) = egui::Key::from_name(&arg) {
                    script.keys.push(k);
                }
                script.wait = 1;
            }
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
            "shot" => {
                let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
                script.awaiting_shot = Some(done.clone());
                let path = PathBuf::from(arg);
                let ctx2 = ctx.clone();
                ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::ScreenshotCallback::new(move |img| {
                    if let Err(e) = save_png(&path, &img) {
                        log::error!("screenshot failed: {e}");
                    }
                    done.store(true, std::sync::atomic::Ordering::SeqCst);
                    ctx2.request_repaint();
                })));
            }
            "quit" => ctx.send_viewport_cmd(egui::ViewportCommand::Close),
            "open" => self.start_import(vec![PathBuf::from(arg)], ctx),
            "wire" => self.view.wireframe = !self.view.wireframe,
            "flat" => self.view.normals = if self.view.normals == NormalMode::Face { NormalMode::Group } else { NormalMode::Face },
            "normals" => self.view.normals = NormalMode::from_key(arg.as_str()).unwrap_or(self.view.normals),
            "grid" => self.view.show_grid = !self.view.show_grid,
            "crease" => match arg.parse() {
                Ok(d) => (self.view.sharp_edges, self.view.crease_deg) = (true, d),
                Err(_) => self.view.sharp_edges = false,
            },
            "groups" => self.view.colors = if self.view.colors == ColorMode::Group { ColorMode::Vertex } else { ColorMode::Group },
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
            "zup" => self.camera.up_axis = UpAxis::Z,
            "tumble" => {
                // tumble:dx,dy — rotate as if dragged by (dx, dy) points.
                let v: Vec<f64> = arg.split(',').filter_map(|x| x.parse().ok()).collect();
                if v.len() == 2 {
                    for _ in 0..10 {
                        self.camera.tumble(v[0] / 10.0, v[1] / 10.0);
                    }
                }
            }
            "select_all" => commands::selection_command(&mut self.cmd_cx(), commands::SelectCmd::All),
            "duplicate" => commands::duplicate_object(&mut self.cmd_cx()),
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
            "save" => {
                self.project_path = Some(PathBuf::from(arg));
                self.save_project(ctx, false);
            }
            "prompts" => self.script_prompts = arg == "on",
            "prefs" => {
                self.show_prefs = true;
                self.prefs_tab = arg.parse().unwrap_or(0);
            }
            "hotbox" => self.hotbox = if self.hotbox.is_some() { None } else { Some(vp.rect.center()) },
            "target" => {
                let id = arg.parse::<usize>().ok().and_then(|i| self.scene.objects.get(i)).map(|o| o.id);
                commands::set_target(&mut self.cmd_cx(), id);
            }
            "activate" => {
                if let Ok(i) = arg.parse::<usize>() {
                    self.scene.active = self.scene.objects.get(i).map(|o| o.id);
                }
            }
            "hide" => {
                if let Ok(i) = arg.parse::<usize>() {
                    if let Some(o) = self.scene.objects.get_mut(i) {
                        o.visible = false;
                    }
                }
            }
            "undo" => self.undo(ctx),
            "redo" => self.redo(ctx),
            "log" => {
                // Print scene statistics to stdout (automated checks).
                for o in &self.scene.objects {
                    println!(
                        "[polymix] {} {}: tris={} verts={} open_edges={} closed={} valid={} selected={} tool={}",
                        arg,
                        o.name,
                        o.mesh.triangle_count(),
                        o.mesh.vertex_count(),
                        o.mesh.boundary_edge_ids().count(),
                        o.mesh.is_closed(),
                        o.mesh.check_validity().is_ok(),
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
            "accept" => {
                if let Some(mut t) = self.tools.pop() {
                    let mut cx = self.tool_cx(ctx, vp);
                    if !t.accept(&mut cx) {
                        self.tools.push(t);
                    }
                }
            }
            "tool" => self.switch_to(SwitchTo::Tool(arg.clone()), ctx, vp),
            other => log::warn!("unknown script step {other}"),
        }
    }

    // ------------------------------------------------------------------
    // Keyboard
    // ------------------------------------------------------------------

    fn handle_keys(&mut self, ctx: &egui::Context, vp: Viewport) {
        if ctx.egui_wants_keyboard_input() {
            return;
        }
        let i = ctx.input(|i| i.clone());
        let cmd = i.modifiers.command;
        // Undo / redo: Cmd+Z, Shift+Cmd+Z / Cmd+Y, Alt+Left / Alt+Right. Alt+Shift+arrows
        // ("skip tool changes" in Meshmixer) behave the same: tools add no history entries
        // of their own.
        if (i.key_pressed(Key::Z) && cmd && !i.modifiers.shift) || (i.key_pressed(Key::ArrowLeft) && i.modifiers.alt) {
            self.undo(ctx);
            return;
        }
        if (i.key_pressed(Key::Z) && cmd && i.modifiers.shift) || (i.key_pressed(Key::Y) && cmd) || (i.key_pressed(Key::ArrowRight) && i.modifiers.alt) {
            self.redo(ctx);
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
                    self.view.wireframe = !self.view.wireframe;
                }
                if i.key_pressed(Key::Y) && i.modifiers.shift && self.accept_tools_for_command(ctx, vp) {
                    commands::separate_shells(&mut self.cmd_cx());
                    self.scene_changed(ctx, vp);
                }
                return;
            }
        }
        if i.key_pressed(Key::W) && !cmd {
            self.view.wireframe = !self.view.wireframe;
        }
        if i.key_pressed(Key::F) && !cmd {
            self.focus_scene();
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
            } else if i.key_pressed(Key::Y) && i.modifiers.shift && self.accept_tools_for_command(ctx, vp) {
                commands::separate_shells(&mut self.cmd_cx());
                self.scene_changed(ctx, vp);
            }
        }
    }
}

impl eframe::App for PolymixApp {
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
        st.set_string("transparent_target", b(self.view.transparent_target));
        st.set_string("view_cube", b(self.view.show_view_cube));
        st.set_string("pivots", b(self.view.show_pivots));
        st.set_string("theme", self.view.theme.key().to_string());
        let bg = self.view.background;
        st.set_string("background", format!("{},{},{}", bg[0], bg[1], bg[2]));
        st.set_string("sharp_edges", b(self.view.sharp_edges));
        st.set_string("nav_mode", self.view.nav_mode.key().to_string());
        st.set_string("nav_custom", self.view.nav_custom.to_pref());
        st.set_string("brush_scroll", self.view.brush_scroll.key().to_string());
        st.set_string("spacemouse", self.space_cfg.to_pref());
        st.set_string("crease_deg", self.view.crease_deg.to_string());
        st.set_string("show_objects", b(self.show_objects));
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

    fn raw_input_hook(&mut self, _ctx: &egui::Context, raw_input: &mut egui::RawInput) {
        if let Some(script) = &mut self.script {
            for key in script.keys.drain(..) {
                for pressed in [true, false] {
                    raw_input.events.push(egui::Event::Key { key, physical_key: None, pressed, repeat: false, modifiers: Default::default() });
                }
            }
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.poll_jobs();

        let dropped: Vec<PathBuf> = ctx.input(|i| i.raw.dropped_files.iter().map(|f| f.path().to_path_buf()).collect());
        if !dropped.is_empty() {
            self.start_import(dropped, &ctx);
        }

        // The viewport rect is only known after the side panels are laid out; tools that need it
        // during panel drawing use the previous frame's value.
        let vp_prev = Viewport { rect: ctx.data(|d| d.get_temp::<egui::Rect>(egui::Id::NULL.with("vp_rect"))).unwrap_or(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0))) };

        self.space_mouse(&ctx, vp_prev);
        if let Some(to) = self.pending_switch.clone() {
            self.switch_to(to, &ctx, vp_prev);
            ctx.request_repaint();
        }

        egui::Panel::top("menu").show(ui, |ui| self.menu_bar(ui, vp_prev));
        egui::Panel::bottom("status").show(ui, |ui| {
            ui.horizontal(|ui| {
                if let Some(j) = &self.import_job {
                    ui.spinner();
                    ui.label(&j.label);
                }
                if let Some(j) = &self.export_job {
                    ui.spinner();
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
            // Under the view cube, at the right.
            let top = if self.view.show_view_cube { 118.0 } else { margin };
            egui::Area::new(egui::Id::NULL.with("floating_objects"))
                .pivot(egui::Align2::RIGHT_TOP)
                .fixed_pos(vr.right_top() + egui::vec2(-margin, top))
                .order(egui::Order::Middle)
                .show(&ctx, |ui| {
                    egui::Frame::window(ui.style()).show(ui, |ui| {
                        ui.set_width(230.0);
                        let max_h = (vr.height() - top - margin - 20.0).max(80.0);
                        ui.set_max_height(max_h);
                        egui::ScrollArea::vertical().max_height(max_h).show(ui, |ui| self.objects_panel(ui));
                    });
                });
        }
        egui::CentralPanel::default().frame(egui::Frame::NONE).show(ui, |ui| {
            let rect = ui.available_rect_before_wrap();
            ctx.data_mut(|d| d.insert_temp(egui::Id::NULL.with("vp_rect"), rect));
            self.viewport(ui, frame);
        });
        let title = match (&self.project_path, self.scene.active()) {
            (Some(p), _) => format!("Polymix — {}", p.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default()),
            (None, Some(o)) => format!("Polymix — {}", o.name),
            (None, None) => "Polymix".to_string(),
        };
        if title != self.window_title {
            ctx.send_viewport_cmd(egui::ViewportCommand::Title(title.clone()));
            self.window_title = title;
        }
        self.handle_keys(&ctx, vp_prev);
        self.help_window(&ctx);
        self.preferences_window(&ctx);
        self.finish_import(&ctx);
        self.recovery_window(&ctx);
        self.autosave_tick(&ctx);
        self.space_window(&ctx);
        self.hotbox_ui(&ctx);
        self.run_script(&ctx, vp_prev);
    }
}

fn save_png(path: &std::path::Path, img: &egui::ColorImage) -> anyhow::Result<()> {
    let f = std::io::BufWriter::new(std::fs::File::create(path)?);
    let mut enc = png::Encoder::new(f, img.size[0] as u32, img.size[1] as u32);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    let mut w = enc.write_header()?;
    let bytes: Vec<u8> = img.pixels.iter().flat_map(|c| c.to_array()).collect();
    w.write_image_data(&bytes)?;
    Ok(())
}

/// Floor grid under the scene, perpendicular to the up axis, with "nice" spacing (1, 2 or 5
/// times a power of ten) and every fifth line brighter.
fn grid_lines(camera: &Camera, b: &mm_core::Aabb) -> Vec<ColorVertex> {
    if b.is_empty() {
        return Vec::new();
    }
    let up = camera.world_up();
    let size = b.diagonal().max(1e-6);
    let raw = size / 20.0;
    let pow = 10f64.powf(raw.log10().floor());
    let step = [1.0, 2.0, 5.0, 10.0].into_iter().map(|m| m * pow).find(|s| *s >= raw).unwrap_or(10.0 * pow);
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
