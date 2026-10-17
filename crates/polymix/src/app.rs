//! Application shell: layout, viewport navigation, tool dispatch, file IO.

use crate::camera::{Camera, UpAxis, ViewHistory};
use crate::commands::{self, CmdCx};
use crate::jobs::Job;
use crate::navigation::{NavAction, NavMode};
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
    Edit,
    Vessel,
    Analysis,
    Sculpt,
}

pub struct ViewOptions {
    pub wireframe: bool,
    pub flat: bool,
    pub group_colors: bool,
    pub boundaries: bool,
    pub show_stats: bool,
    /// Keep edges sharper than `crease_deg` crisp in smooth shading (CAD parts).
    pub sharp_edges: bool,
    pub crease_deg: f32,
    /// Mouse navigation scheme (Meshmixer's Preferences → Navigation Mode).
    pub nav_mode: NavMode,
    pub show_grid: bool,
}

impl Default for ViewOptions {
    fn default() -> Self {
        Self { wireframe: false, flat: false, group_colors: false, boundaries: true, show_stats: true, sharp_edges: true, crease_deg: DEFAULT_CREASE_DEG, nav_mode: NavMode::Fusion360, show_grid: false }
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
        "units" => Box::new(tools::transform::UnitsTool::new(cx)),
        "measure" => Box::new(tools::measure::MeasureTool::new(cx)),
        "sculpt" => Box::new(tools::sculpt::SculptTool::new(cx)),
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
    /// Tool switch waiting for the current tool to finish computing (then it is accepted).
    pending_switch: Option<SwitchTo>,
    status: String,
    import_job: Option<Job<ImportResult>>,
    export_job: Option<Job<String>>,
    nav_drag: Option<PointerButton>,
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
    last_export_dir: Option<PathBuf>,
    export_ascii: bool,
    window_title: String,
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
}

impl PolymixApp {
    pub fn new(cc: &eframe::CreationContext<'_>, files: Vec<PathBuf>) -> Self {
        ui_util::apply_theme(&cc.egui_ctx);
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
            pending_switch: None,
            status: "Welcome to Polymix. Import a mesh to begin (Ctrl/Cmd+O, or drag a file onto the window).".into(),
            import_job: None,
            export_job: None,
            nav_drag: None,
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
            last_export_dir: None,
            export_ascii: false,
            window_title: String::new(),
            recent: Vec::new(),
            synthetic: Default::default(),
            boundary_cache: HashMap::new(),
            script: std::env::var("POLYMIX_SCRIPT").ok().map(|s| Script {
                steps: s.split(';').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect(),
                wait: 0,
                awaiting_shot: None,
            }),
        };
        if let Some(st) = cc.storage.filter(|_| std::env::var_os("POLYMIX_SCRIPT").is_none()) {
            let flag = |k: &str, d: bool| st.get_string(k).map(|v| v == "1").unwrap_or(d);
            app.view.wireframe = flag("wireframe", false);
            app.view.flat = flag("flat", false);
            app.view.boundaries = flag("boundaries", true);
            app.view.group_colors = flag("group_colors", false);
            app.view.show_grid = flag("grid", false);
            app.view.sharp_edges = flag("sharp_edges", true);
            app.view.nav_mode = st.get_string("nav_mode").and_then(|k| NavMode::from_key(&k)).unwrap_or(NavMode::Fusion360);
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
            app.recent = st.get_string("recent").map(|r| r.lines().filter(|l| !l.is_empty()).map(PathBuf::from).collect()).unwrap_or_default();
        }
        if !files.is_empty() {
            app.start_import(files, &cc.egui_ctx);
        }
        app
    }

    // ------------------------------------------------------------------
    // File IO
    // ------------------------------------------------------------------

    fn start_import(&mut self, files: Vec<PathBuf>, ctx: &egui::Context) {
        if files.is_empty() || self.import_job.is_some() {
            return;
        }
        for f in &files {
            self.recent.retain(|r| r != f);
            self.recent.insert(0, f.clone());
        }
        self.recent.truncate(10);
        let n = files.len();
        self.import_job = Some(Job::spawn(format!("Importing {n} file(s)"), ctx.clone(), move |progress| {
            let mut out = Vec::new();
            for (i, f) in files.iter().enumerate() {
                if !progress(i as f32 / n as f32) {
                    anyhow::bail!("cancelled");
                }
                let t0 = std::time::Instant::now();
                let imported = mm_io::read_mesh(f).map_err(|e| anyhow::anyhow!("{}: {e:#}", f.display()))?;
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
                        let first_import = self.scene.is_empty();
                        if !meshes.is_empty() {
                            self.history.push("Import", &self.scene);
                        }
                        let mut notes = Vec::new();
                        for (name, mesh, note) in meshes {
                            self.scene.add(name, mesh);
                            notes.push(note);
                        }
                        self.status = notes.join("  |  ");
                        if first_import {
                            self.camera.scene_radius = 1e-6;
                            self.camera.focus(&self.scene.bounds());
                        }
                    }
                    Err(e) => self.status = format!("Import failed: {e:#}"),
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

    fn import_dialog(&mut self, ctx: &egui::Context) {
        let files = rfd::FileDialog::new()
            .add_filter("Meshes", &["stl", "obj", "ply", "off", "3mf"])
            .add_filter("All files", &["*"])
            .pick_files();
        if let Some(files) = files {
            self.start_import(files, ctx);
        }
    }

    fn export_dialog(&mut self, ctx: &egui::Context, all_visible: bool) {
        let Some(active) = self.scene.active() else {
            self.status = "Nothing to export.".into();
            return;
        };
        let mut dlg = rfd::FileDialog::new()
            .add_filter("STL (binary)", &["stl"])
            .add_filter("OBJ", &["obj"])
            .add_filter("PLY", &["ply"])
            .add_filter("OFF", &["off"])
            .add_filter("3MF", &["3mf"])
            .set_file_name(format!("{}.stl", active.name));
        if let Some(d) = &self.last_export_dir {
            dlg = dlg.set_directory(d);
        }
        let Some(path) = dlg.save_file() else { return };
        self.last_export_dir = path.parent().map(|p| p.to_path_buf());
        let mesh: Mesh = if all_visible {
            let mut m = Mesh::new();
            for o in self.scene.objects.iter().filter(|o| o.visible) {
                m.append_mesh(&o.mesh, 0);
            }
            m
        } else {
            (*active.mesh).clone()
        };
        let p2 = path.clone();
        let opts = mm_io::WriteOptions { stl_ascii: self.export_ascii, ..Default::default() };
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
        let keep = match &to {
            SwitchTo::Tool(n) if n == "select" => self.tools.iter().position(|t| t.name() == "Select").map(|i| i + 1).unwrap_or(0),
            _ => 0,
        };
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
                    let mut cx = self.tool_cx(ctx, vp);
                    if let Some(t) = make_tool(&name, &mut cx) {
                        self.tools.push(t);
                    }
                }
            }
            SwitchTo::Category(c) => self.category = c,
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
        ToolCx { scene: &mut self.scene, history: &mut self.history, camera: &mut self.camera, status: &mut self.status, egui: ctx, vp }
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
                if ui.button("Import…   ⌘O").clicked() {
                    ui.close();
                    self.import_dialog(&ctx);
                }
                if ui.button("Export active…   ⌘E").clicked() {
                    ui.close();
                    self.export_dialog(&ctx, false);
                }
                if ui.button("Export all visible as one…").clicked() {
                    ui.close();
                    self.export_dialog(&ctx, true);
                }
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
            });
            ui.menu_button("View", |ui| {
                ui.checkbox(&mut self.view.wireframe, "Wireframe  (W)");
                ui.checkbox(&mut self.view.flat, "Flat shading");
                ui.checkbox(&mut self.view.sharp_edges, "Sharp edges")
                    .on_hover_text("Keep hard corners crisp instead of averaging normals across them (CAD parts). Edges bending more than the crease angle are shaded sharp.");
                ui.add_enabled(self.view.sharp_edges, egui::Slider::new(&mut self.view.crease_deg, 5.0..=90.0).suffix("°").text("Crease angle"));
                ui.checkbox(&mut self.view.group_colors, "Face group colors");
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
        let big = |ui: &mut egui::Ui, label: &str, active: bool| {
            let b = egui::Button::new(RichText::new(label).size(13.0)).min_size(egui::vec2(84.0, 30.0)).selected(active);
            ui.add(b)
        };
        if big(ui, "Import", false).on_hover_text("⌘O").clicked() {
            self.import_dialog(&ctx);
        }
        let sel_active = self.tools.first().map(|t| t.name() == "Select").unwrap_or(false);
        if big(ui, "Select", sel_active).on_hover_text("S").clicked() {
            self.switch_to(SwitchTo::Tool("select".into()), &ctx, vp);
        }
        for (cat, label) in [(Category::Sculpt, "Sculpt"), (Category::Edit, "Edit"), (Category::Vessel, "Vessel"), (Category::Analysis, "Analysis")] {
            if big(ui, label, self.category == Some(cat) && self.tools.is_empty()).clicked() {
                let to = if self.category == Some(cat) && self.tools.is_empty() { None } else { Some(cat) };
                self.switch_to(SwitchTo::Category(to), &ctx, vp);
            }
        }
        if big(ui, "Export", false).on_hover_text("⌘E").clicked() {
            self.export_dialog(&ctx, false);
        }
    }

    fn category_panel(&mut self, ui: &mut egui::Ui, cat: Category, vp: Viewport) {
        let ctx = ui.ctx().clone();
        let has_obj = self.scene.active().is_some();
        let mut open: Option<&str> = None;
        match cat {
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
                    if ui.button("Plane Cut").clicked() {
                        open = Some("planecut");
                    }
                    if ui.button("Mirror").clicked() {
                        open = Some("mirror");
                    }
                    if ui.button("Make Solid").clicked() {
                        open = Some("makesolid");
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
            Category::Sculpt => {
                ui_util::section(ui, "Sculpt");
                ui.add_enabled_ui(has_obj, |ui| {
                    if ui.button("Surface brushes").clicked() {
                        open = Some("sculpt");
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
                let label = RichText::new(&o.name).color(if active { ui_util::ACCENT } else { Color32::from_gray(210) });
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
                    ui.selectable_value(&mut self.view.flat, false, "Smooth");
                    ui.selectable_value(&mut self.view.flat, true, "Faceted");
                    ui.toggle_value(&mut self.view.wireframe, "Wireframe");
                });
                ui.horizontal(|ui| {
                    ui.toggle_value(&mut self.view.boundaries, "Boundaries");
                    ui.toggle_value(&mut self.view.group_colors, "Group colors");
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
                if ui.small_button("close (Space)").clicked() {
                    close = true;
                }
            });
        });
        if close {
            self.hotbox = None;
        }
    }

    fn help_window(&mut self, ctx: &egui::Context) {
        egui::Window::new("Controls & hotkeys").open(&mut self.show_help).resizable(false).show(ctx, |ui| {
            egui::Grid::new("help").num_columns(2).striped(true).show(ui, |ui| {
                ui.label(RichText::new(format!("{} navigation", self.view.nav_mode.label())).strong());
                ui.end_row();
                for (k, v) in self.view.nav_mode.help_rows().into_iter().chain([
                    ("F", "Focus on scene"),
                    ("Space", "Hotbox (view settings)"),
                    ("W", "Toggle wireframe"),
                    ("S", "Select tool"),
                    ("I", "Inspector (when not selecting)"),
                    ("T", "Transform"),
                    ("Shift+Y", "Separate shells"),
                    ("Select: drag / Shift+drag", "Paint selection / deselect"),
                    ("Select: double-click", "Select connected part"),
                    ("Select / Sculpt: [ ], wheel (Meshmixer nav) or Shift+wheel (Fusion 360 nav)", "Brush size"),
                    ("Select: right-drag (with a selection)", "Grow / shrink the selection"),
                    ("Sculpt: right-drag", "Brush strength"),
                    ("Select: E / I / G / O / < >", "Connected / Invert / Groups / Optimize / Shrink-Grow"),
                    ("Select: X, F, D, Cmd+D", "Discard, Erase & Fill, Extrude, Offset"),
                    ("Select: R, Shift+R, Cmd+F, B, Y, J, Cmd+B", "Remesh, Reduce, Smooth, Smooth Boundary, Separate, Join, Bridge"),
                    ("Remesh / Reduce: D / F, T, G", "Density ±10, adaptive, preserve groups"),
                    ("Enter or A / Esc", "Accept / Cancel tool"),
                    ("Cmd+Z / Shift+Cmd+Z (Cmd+Y) / Alt+← Alt+→", "Undo / Redo"),
                    ("Cmd+Shift+O", "Object browser"),
                ]) {
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
        if spacemouse::apply(&mut self.camera, motion, &self.space_cfg, dt, vp.rect.height() as f64) {
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
        let (mods, scroll, zoom_delta, pointer_delta, hover) = ctx.input(|i| {
            (i.modifiers, i.smooth_scroll_delta(), i.zoom_delta(), i.pointer.delta(), i.pointer.hover_pos())
        });
        // `hovered` respects layering, so popups over the viewport don't leak input into it.
        let in_rect = hover.map(|p| rect.contains(p)).unwrap_or(false) && (response.hovered() || response.dragged());
        let nav = self.view.nav_mode;
        let primary_nav = nav.starts_navigation(PointerButton::Primary, mods);
        // Tools may own right-drags (selection expand, brush strength); Alt keeps navigation.
        let tool_claims_rmb = !mods.alt && !mods.shift && self.tools.last().is_some_and(|t| t.claims_secondary_drag(&self.scene));
        if response.drag_started_by(PointerButton::Secondary) && tool_claims_rmb {
            if let Some(p) = ctx.input(|i| i.pointer.press_origin()) {
                self.tool_rmb = Some(p);
                self.tool_secondary(&ctx, vp, DragPhase::Start, p, egui::Vec2::ZERO);
            }
        }
        for btn in [PointerButton::Secondary, PointerButton::Middle, PointerButton::Primary] {
            if self.tool_rmb.is_some() && btn == PointerButton::Secondary {
                continue;
            }
            if response.drag_started_by(btn) && nav.starts_navigation(btn, mods) {
                self.nav_drag = Some(btn);
                self.remember_view();
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
                    Some(NavAction::Tumble) => self.camera.tumble(d.x as f64, d.y as f64),
                    Some(NavAction::Pan) => self.camera.pan(d.x as f64, d.y as f64, rect.height() as f64),
                    Some(NavAction::Zoom) => self.camera.zoom((1.0 + d.y as f64 * 0.005).max(0.05)),
                    None => {}
                }
            }
            if response.drag_stopped_by(btn) || !ctx.input(|i| i.pointer.button_down(btn)) {
                self.nav_drag = None;
            }
        }
        if in_rect {
            // Shift+wheel arrives as horizontal scroll from a mouse wheel on macOS.
            let wheel = if scroll.y != 0.0 { scroll.y } else { scroll.x };
            // Brush tools take the wheel for their size: the plain wheel in Meshmixer
            // navigation, Shift+wheel in Fusion 360 navigation (where the wheel zooms).
            let to_tool = match nav {
                NavMode::Meshmixer => !mods.shift && !mods.command,
                NavMode::Fusion360 => mods.shift,
            };
            let mut scroll_y = scroll.y;
            if wheel != 0.0 && to_tool && self.tool_wheel(&ctx, vp, wheel) {
                scroll_y = 0.0;
            }
            let mut factor = 1.0f64;
            if scroll_y != 0.0 {
                factor *= (-scroll_y as f64 * 0.0025).exp();
            }
            if zoom_delta != 1.0 {
                factor /= zoom_delta as f64;
            }
            if factor != 1.0 {
                let now = ctx.input(|i| i.time);
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
                    objs.push((g, o.color.w < 0.999));
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
                "Drop an STL / OBJ / PLY / OFF / 3MF file here\nor press ⌘O to import",
                egui::FontId::proportional(18.0),
                Color32::from_gray(150),
            );
        }
        if let Some(job) = &self.import_job {
            painter.text(rect.center(), egui::Align2::CENTER_CENTER, format!("{}…", job.label), egui::FontId::proportional(18.0), Color32::WHITE);
            ctx.request_repaint();
        }
        self.axis_gizmo(&painter, rect);
        if self.nav_drag.is_some() || response.dragged() || in_rect {
            // Keep hover-driven overlays (brush cursor) responsive.
            ctx.request_repaint_after(std::time::Duration::from_millis(16));
        }
    }

    fn axis_gizmo(&self, painter: &egui::Painter, rect: egui::Rect) {
        let c = rect.left_bottom() + egui::vec2(40.0, -40.0);
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
        for o in self.scene.objects.iter_mut().filter(|o| o.visible) {
            let style = ObjectStyle {
                color: o.color,
                wireframe: self.view.wireframe,
                flat: self.view.flat,
                group_colors: self.view.group_colors || self.tools.last().is_some_and(|t| t.shows_groups()),
                dim: multiple && Some(o.id) != active,
                cut_tint: matches!(cut_tint, Some(t) if t.is_none() || t == Some(o.id)),
                crease_deg: self.view.sharp_edges.then_some(self.view.crease_deg),
            };
            if let Some((pid, pm)) = preview {
                if pid == o.id {
                    let g = self.preview_gpu.get_or_insert_with(|| renderer.create_mesh(&rs.device));
                    renderer.sync_mesh(&rs.device, &rs.queue, g, pm, None, 0, &style, (&mut no_moves, 0), None);
                    continue;
                }
            }
            let g = self.gpu.entry(o.id).or_insert_with(|| renderer.create_mesh(&rs.device));
            let scalars = o.scalars.as_ref().map(|(stamp, v)| (*stamp, v.as_slice()));
            renderer.sync_mesh(&rs.device, &rs.queue, g, &o.mesh, Some(&o.selection), o.selection_stamp, &style, (&mut o.moved_vertices, o.moved_from), scalars);
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
            "flat" => self.view.flat = !self.view.flat,
            "grid" => self.view.show_grid = !self.view.show_grid,
            "crease" => match arg.parse() {
                Ok(d) => (self.view.sharp_edges, self.view.crease_deg) = (true, d),
                Err(_) => self.view.sharp_edges = false,
            },
            "groups" => self.view.group_colors = !self.view.group_colors,
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
                        "[polymix] {} {}: tris={} verts={} open_edges={} closed={} valid={}",
                        arg,
                        o.name,
                        o.mesh.triangle_count(),
                        o.mesh.vertex_count(),
                        o.mesh.boundary_edge_ids().count(),
                        o.mesh.is_closed(),
                        o.mesh.check_validity().is_ok()
                    );
                }
            }
            "category" => {
                let c = match arg.as_str() {
                    "edit" => Some(Category::Edit),
                    "vessel" => Some(Category::Vessel),
                    "analysis" => Some(Category::Analysis),
                    "sculpt" => Some(Category::Sculpt),
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
            self.export_dialog(ctx, false);
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
        // Accept: Enter, or A (Meshmixer); Sculpt will use A for attract (ROADMAP M8.3).
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
            if is_select {
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
            let to = if i.key_pressed(Key::S) {
                Some("select")
            } else if i.key_pressed(Key::I) {
                Some("inspector")
            } else if i.key_pressed(Key::T) && !i.modifiers.shift {
                Some("transform")
            } else {
                None
            };
            if let Some(name) = to {
                self.switch_to(SwitchTo::Tool(name.into()), ctx, vp);
            } else if i.key_pressed(Key::Y) && i.modifiers.shift && self.accept_tools_for_command(ctx, vp) {
                commands::separate_shells(&mut self.cmd_cx());
                self.scene_changed(ctx, vp);
            }
        }
    }
}

impl eframe::App for PolymixApp {
    fn save(&mut self, st: &mut dyn eframe::Storage) {
        // Scripted (test) sessions never overwrite the user's preferences.
        if std::env::var_os("POLYMIX_SCRIPT").is_some() {
            return;
        }
        let b = |v: bool| if v { "1".to_string() } else { "0".to_string() };
        st.set_string("wireframe", b(self.view.wireframe));
        st.set_string("flat", b(self.view.flat));
        st.set_string("boundaries", b(self.view.boundaries));
        st.set_string("group_colors", b(self.view.group_colors));
        st.set_string("grid", b(self.view.show_grid));
        st.set_string("sharp_edges", b(self.view.sharp_edges));
        st.set_string("nav_mode", self.view.nav_mode.key().to_string());
        st.set_string("spacemouse", self.space_cfg.to_pref());
        st.set_string("crease_deg", self.view.crease_deg.to_string());
        st.set_string("show_objects", b(self.show_objects));
        st.set_string("orthographic", b(self.camera.orthographic));
        st.set_string("free_orbit", b(self.camera.free_orbit));
        st.set_string("up_axis", if self.camera.up_axis == UpAxis::Z { "z".into() } else { "y".into() });
        st.set_string("export_ascii", b(self.export_ascii));
        st.set_string("recent", self.recent.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join("\n"));
        if let Some(d) = &self.last_export_dir {
            st.set_string("last_export_dir", d.display().to_string());
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
        egui::Panel::left("palette").resizable(false).exact_size(100.0).show(ui, |ui| {
            ui.vertical_centered(|ui| self.palette(ui, vp_prev));
        });
        let show_tool_panel = !self.tools.is_empty() || self.category.is_some();
        if show_tool_panel {
            egui::Panel::left("tool").resizable(true).default_size(300.0).size_range(240.0..=460.0).show(ui, |ui| {
                if !self.tools.is_empty() {
                    self.tool_panel(ui, vp_prev);
                } else if let Some(cat) = self.category {
                    self.category_panel(ui, cat, vp_prev);
                }
            });
        }
        if self.show_objects {
            egui::Panel::right("objects").resizable(true).default_size(230.0).size_range(180.0..=400.0).show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.objects_panel(ui));
            });
        }
        egui::CentralPanel::default().frame(egui::Frame::NONE).show(ui, |ui| {
            let rect = ui.available_rect_before_wrap();
            ctx.data_mut(|d| d.insert_temp(egui::Id::NULL.with("vp_rect"), rect));
            self.viewport(ui, frame);
        });
        let title = match self.scene.active() {
            Some(o) => format!("Polymix — {}", o.name),
            None => "Polymix".to_string(),
        };
        if title != self.window_title {
            ctx.send_viewport_cmd(egui::ViewportCommand::Title(title.clone()));
            self.window_title = title;
        }
        self.handle_keys(&ctx, vp_prev);
        self.help_window(&ctx);
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
