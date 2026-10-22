//! The application shell: the workspace tabs (Volume | Segmentation | Mesh) above
//! independent workspaces.
//!
//! The shell owns the window-level things every workspace shares: the tab bar and its
//! hotkeys (⌘1 / ⌘2 / ⌘3), routing of dropped files and command-line paths, the window title,
//! persistence of the chosen tab, and the test-script runner (`POLYMIX_SCRIPT`). Each
//! workspace is called through its [`Slot`], so a panic inside one stops only that one: its
//! tab shows a crash card with a Restart button while the others keep working.

use crate::agents::{self, AgentAccess};
use crate::app::MeshWorkspace;
use mm_volume::model::Catalog;
use polymix_mcp::{Call, Reply};
use polymix_segmentation::SegmentationWorkspace;
use polymix_volume::VolumeWorkspace;
use polymix_workspace::guard::{self, CardAction, Slot};
use polymix_workspace::style::{apply_theme, Theme};
use polymix_workspace::agent::{AgentReply, ToolKind};
use polymix_workspace::Workspace;
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant};

/// Agents' work waits for drawn frames; none for this long means the window is not drawn
/// (eframe draws nothing while it is covered or minimized).
const NOT_DRAWN_AFTER: Duration = Duration::from_millis(750);
/// With "Bring Polymix forward" on: bring the window forward after this long without a frame,
/// and give it this long to be drawn then.
const BRING_FORWARD_AFTER: Duration = Duration::from_millis(250);
const NOT_DRAWN_AFTER_BRINGING: Duration = Duration::from_millis(2000);
/// How long after an agent's change its work in the Mesh workspace counts as the agent's.
const AGENT_WORK: Duration = Duration::from_secs(30);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tab {
    Volume,
    Segmentation,
    Mesh,
}

impl Tab {
    const ALL: [Tab; 3] = [Tab::Volume, Tab::Segmentation, Tab::Mesh];

    fn label(self) -> &'static str {
        match self {
            Tab::Volume => "Volume",
            Tab::Segmentation => "Segmentation",
            Tab::Mesh => "Mesh",
        }
    }

    fn key(self) -> &'static str {
        match self {
            Tab::Volume => "volume",
            Tab::Segmentation => "segmentation",
            Tab::Mesh => "mesh",
        }
    }

    fn from_key(k: &str) -> Option<Tab> {
        Tab::ALL.into_iter().find(|t| t.key() == k)
    }

    fn hotkey(self) -> egui::Key {
        match self {
            Tab::Volume => egui::Key::Num1,
            Tab::Segmentation => egui::Key::Num2,
            Tab::Mesh => egui::Key::Num3,
        }
    }
}

pub struct PolymixApp {
    active: Tab,
    volume: Slot<VolumeWorkspace>,
    segmentation: Slot<SegmentationWorkspace>,
    mesh: Slot<MeshWorkspace>,
    /// Paths from the command line, opened on the first frame.
    pending_open: Vec<PathBuf>,
    /// The modifier keys the keyboard holds, and those the 3D mouse's buttons held last frame
    /// (see `spacemouse::with_held_keys`).
    keyboard_modifiers: egui::Modifiers,
    held_modifiers: egui::Modifiers,
    /// Keys 3D mouse buttons pressed last frame, to let go this one.
    keys_down: Vec<egui::Key>,
    script: Option<Script>,
    /// Frames drawn (`frames:` script step).
    frames: u64,
    /// `crash:<workspace>` (test scripts): make that workspace panic on its next frame.
    crash_next: Option<Tab>,
    /// `restart:<workspace>` (test scripts): press its crash card's Restart.
    restart_next: Option<Tab>,
    title: String,
    /// LLM clients reading (and, if allowed, changing) the session over MCP (see [`agents`]).
    agents: AgentAccess,
    /// Agents reading and working the panels (AccessKit).
    ui_access: crate::ui_access::UiAccess,
    /// `app_wait` calls, answered when every workspace is idle or their time is up.
    waits: Vec<Wait>,
    /// Screenshots asked for: the callback takes the call when the frame is drawn, or the
    /// watchdog when no frame comes.
    shots: Vec<Arc<Mutex<Option<Call>>>>,
    /// Since when agents' work has waited for a frame (cleared by every drawn frame).
    want_frame: Option<Instant>,
    /// The window was brought forward for the current wait.
    brought_forward: bool,
    /// Until when work in the Mesh workspace is an agent's (after its last change).
    agent_work_until: Option<Instant>,
    /// What Segmentation's overlay choices were listed for.
    overlay_key: Option<OverlayKey>,
}

/// The catalog, volume and *Hide identifiers* setting Segmentation's overlay choices were
/// listed for. The catalog is held weakly: that keeps its address from going to a newer
/// catalog while it is remembered, without keeping a replaced catalog's contents alive.
struct OverlayKey {
    catalog: Weak<Catalog>,
    volume: usize,
    hide: bool,
}

impl OverlayKey {
    fn new(catalog: &Arc<Catalog>, volume: usize, hide: bool) -> Self {
        Self { catalog: Arc::downgrade(catalog), volume, hide }
    }

    fn is(&self, catalog: &Arc<Catalog>, volume: usize, hide: bool) -> bool {
        std::ptr::eq(self.catalog.as_ptr(), Arc::as_ptr(catalog)) && self.volume == volume && self.hide == hide
    }
}

struct Wait {
    call: Call,
    start: Instant,
    until: Instant,
    /// Frames in a row with every workspace idle.
    quiet: u32,
}

/// The shell's part of a test script: waits, key presses, screenshots, tab switches.
/// Every other step goes to the shown workspace.
struct Script {
    steps: VecDeque<String>,
    wait: u32,
    awaiting_shot: Option<Arc<AtomicBool>>,
    /// Key presses (`key:S`) for the next frame's input.
    keys: Vec<egui::Key>,
    /// Raw mouse events (`pointer:`), one frame's worth per frame.
    pointer: VecDeque<Vec<egui::Event>>,
    /// `idle:<seconds>`: no repaints forced until then (input time), as when nobody runs a
    /// script — whatever still needs frames must ask for them itself.
    idle_until: Option<f64>,
}

impl PolymixApp {
    pub fn new(cc: &eframe::CreationContext<'_>, files: Vec<PathBuf>) -> Self {
        guard::install_panic_hook();
        let ctx = &cc.egui_ctx;
        #[cfg(target_os = "macos")]
        crate::macos_open::set_context(ctx);
        let storage = cc.storage;
        let scripted = polymix_workspace::scripted();
        // The theme is app-wide: apply it here so every workspace starts with it, even if the
        // Mesh workspace (whose preferences set it) fails to start.
        let theme = storage.filter(|_| !scripted).and_then(|s| s.get_string("theme")).and_then(|k| Theme::from_key(&k)).unwrap_or(Theme::Dark);
        apply_theme(ctx, theme);
        let mut active = storage.filter(|_| !scripted).and_then(|s| s.get_string("shell.tab")).and_then(|k| Tab::from_key(&k)).unwrap_or(Tab::Mesh);
        let script = std::env::var("POLYMIX_SCRIPT").ok().map(|s| Script {
            steps: s.split(';').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect(),
            wait: 0,
            awaiting_shot: None,
            keys: Vec::new(),
            pointer: VecDeque::new(),
            idle_until: None,
        });
        // Workspaces that fail to start are shown stopped, like a crash later on.
        let volume = slot("Volume", || VolumeWorkspace::new(ctx, storage));
        let segmentation = slot("Segmentation", || SegmentationWorkspace::new(ctx, storage));
        let mesh = slot("Mesh", || MeshWorkspace::new(ctx, cc.wgpu_render_state.as_ref(), storage));
        // A recovery prompt (or similar) must be seen: show that workspace first.
        if mesh.live().is_some_and(|w| w.needs_attention()) {
            active = Tab::Mesh;
        } else if volume.live().is_some_and(|w| w.needs_attention()) {
            active = Tab::Volume;
        }
        // Every workspace's agent tools are listed, even those of a workspace that failed to start.
        let mut agents = AgentAccess::new(storage, agents::all_tools());
        agents.start_if_enabled(ctx);
        let ui_access = crate::ui_access::UiAccess::new(ctx);
        Self { active, volume, segmentation, mesh, pending_open: files, script, frames: 0, crash_next: None, restart_next: None, title: String::new(), agents, ui_access, waits: Vec::new(), shots: Vec::new(), want_frame: None, brought_forward: false, agent_work_until: None, overlay_key: None, keyboard_modifiers: egui::Modifiers::NONE, held_modifiers: egui::Modifiers::NONE, keys_down: Vec::new() }
    }

    /// Open paths in the workspaces that take them (meshes and projects → Mesh, images and
    /// folders → Volume), showing the workspace that got them.
    fn open_paths(&mut self, paths: Vec<PathBuf>, ctx: &egui::Context) {
        let (mut mesh, mut volume) = (Vec::new(), Vec::new());
        for p in paths {
            if self.mesh.live().is_some_and(|w| w.accepts(&p)) {
                mesh.push(p);
            } else if self.volume.live().is_some_and(|w| w.accepts(&p)) {
                volume.push(p);
            } else {
                log::warn!("no workspace opens {}", p.display());
            }
        }
        if !mesh.is_empty() {
            self.mesh.call(|w| w.open(mesh, ctx));
            self.active = Tab::Mesh;
        }
        if !volume.is_empty() {
            self.volume.call(|w| w.open(volume, ctx));
            self.active = Tab::Volume;
        }
    }

    fn crashed(&self, tab: Tab) -> bool {
        match tab {
            Tab::Volume => matches!(self.volume, Slot::Crashed(_)),
            Tab::Segmentation => matches!(self.segmentation, Slot::Crashed(_)),
            Tab::Mesh => matches!(self.mesh, Slot::Crashed(_)),
        }
    }

    fn tab_bar(&mut self, ui: &mut egui::Ui) {
        egui::Panel::top("workspace_tabs").show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 2.0;
                for tab in Tab::ALL {
                    let mut text = egui::RichText::new(tab.label()).size(13.0);
                    if self.active == tab {
                        text = text.strong();
                    }
                    let label = if self.crashed(tab) { egui::RichText::new(format!("⚠ {}", tab.label())).size(13.0).color(egui::Color32::from_rgb(240, 150, 80)) } else { text };
                    let r = ui.selectable_label(self.active == tab, label).on_hover_text(format!("{} workspace (⌘{})", tab.label(), Tab::ALL.iter().position(|t| *t == tab).unwrap() + 1));
                    if r.clicked() {
                        self.active = tab;
                    }
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| self.agents.indicator(ui));
            });
        });
    }

    /// Show the active workspace, or its crash card.
    fn workspace_ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        let crash = self.crash_next == Some(self.active);
        if crash {
            self.crash_next = None;
        }
        let restart = self.restart_next.take() == Some(self.active);
        match self.active {
            Tab::Volume => match &self.volume {
                Slot::Live(_) => {
                    self.volume.call(|w| {
                        if crash {
                            panic!("crash test requested by the script");
                        }
                        w.ui(ui, frame)
                    });
                }
                Slot::Crashed(report) => {
                    if guard::crash_card(ui, report, "Restart opens an empty Volume workspace; your folders are scanned again.") == CardAction::Restart || restart {
                        self.volume = slot("Volume", || VolumeWorkspace::new(&ctx, frame.storage()));
                    }
                }
            },
            Tab::Mesh => match &self.mesh {
                Slot::Live(_) => {
                    self.mesh.call(|w| {
                        if crash {
                            panic!("crash test requested by the script");
                        }
                        w.ui(ui, frame)
                    });
                }
                Slot::Crashed(report) => {
                    if guard::crash_card(ui, report, "Restart opens a fresh Mesh workspace and offers to recover the last autosave.") == CardAction::Restart || restart {
                        self.mesh = slot("Mesh", || MeshWorkspace::new(&ctx, frame.wgpu_render_state(), frame.storage()));
                    }
                }
            },
            Tab::Segmentation => match &self.segmentation {
                Slot::Live(_) => {
                    self.segmentation.call(|w| {
                        if crash {
                            panic!("crash test requested by the script");
                        }
                        w.ui(ui, frame)
                    });
                }
                Slot::Crashed(report) => {
                    if guard::crash_card(ui, report, "Restart opens an empty Segmentation workspace; it reloads the volume selected in Volume.") == CardAction::Restart || restart {
                        self.segmentation = slot("Segmentation", || SegmentationWorkspace::new(&ctx, frame.storage()));
                    }
                }
            },
        }
    }

    fn update_title(&mut self, ctx: &egui::Context) {
        let title = match self.active {
            Tab::Volume => self.volume.live().map(|w| w.title()),
            Tab::Segmentation => self.segmentation.live().map(|w| w.title()),
            Tab::Mesh => self.mesh.live().map(|w| w.title()),
        }
        .unwrap_or_else(|| format!("Polymix — {} (stopped)", self.active.label()));
        if title != self.title {
            ctx.send_viewport_cmd(egui::ViewportCommand::Title(title.clone()));
            self.title = title;
        }
    }

    /// Answer the calls agents made since the last frame. Answers are quick here: work that
    /// takes longer was left by the workspaces to run on the server's worker threads.
    fn serve_agents(&mut self, ctx: &egui::Context) {
        let Some(server) = self.agents.server() else { return };
        let calls: Vec<Call> = std::iter::from_fn(|| server.try_recv()).take(32).collect();
        if calls.len() == 32 {
            ctx.request_repaint();
        }
        for call in calls {
            self.answer_agent(call, ctx);
        }
    }

    fn answer_agent(&mut self, call: Call, ctx: &egui::Context) {
        if let Some(why) = self.agents.refusal(&call.tool) {
            return call.answer(Reply::Error(why));
        }
        // Changes happen in front of the user: show the workspace being changed.
        if self.agents.kind(&call.tool) != ToolKind::Read {
            if let Some(tab) = call.tool.split('_').next().and_then(Tab::from_key) {
                self.active = tab;
            }
            self.agent_work_until = Some(Instant::now() + AGENT_WORK);
        }
        // Patients' names can turn up in any text (object names come from file names).
        let mask = self.volume.live().map(|w| w.agent_mask());
        let reply = match call.tool.as_str() {
            "app_status" => agents::masked(Ok(self.agent_status(ctx)), mask.as_ref()),
            "app_log" => agents::masked(Ok(agents::log(&call.args)), mask.as_ref()),
            "app_screenshot" => return self.agent_screenshot(call, ctx),
            "app_wait" => {
                let now = Instant::now();
                let seconds = polymix_workspace::agent::arg_usize(&call.args, "seconds", 10, 1, 25) as u64;
                self.waits.push(Wait { call, start: now, until: now + Duration::from_secs(seconds), quiet: 0 });
                ctx.request_repaint();
                return;
            }
            "app_ui" | "app_ui_action" => {
                if let Err(e) = self.identifiers_hidden() {
                    return call.answer(Reply::Error(e));
                }
                if call.tool == "app_ui" {
                    self.ui_access.read(call, mask, ctx);
                } else {
                    // The change comes from the shown workspace's panels.
                    match self.active {
                        Tab::Volume => self.volume.call(|w| w.agent_acting()),
                        Tab::Segmentation => self.segmentation.call(|w| w.agent_acting()),
                        Tab::Mesh => self.mesh.call(|w| w.agent_acting()),
                    };
                    self.ui_access.act(call, ctx);
                }
                return;
            }
            "app_show" => match polymix_workspace::agent::arg_str(&call.args, "workspace").and_then(Tab::from_key) {
                Some(tab) => {
                    self.active = tab;
                    Reply::Json(json!({"shown": tab.key()}))
                }
                None => Reply::Error("workspace must be volume, segmentation or mesh".into()),
            },
            tool => {
                let (name, r) = match tool.split('_').next().unwrap_or("") {
                    "volume" => ("Volume", self.volume.call(|w| agents::workspace_call(w, tool, &call.args, ctx))),
                    "segmentation" => ("Segmentation", self.segmentation.call(|w| agents::workspace_call(w, tool, &call.args, ctx))),
                    "mesh" => ("Mesh", self.mesh.call(|w| agents::workspace_call(w, tool, &call.args, ctx))),
                    _ => ("", Some(AgentReply::error(format!("unknown tool {tool}")))),
                };
                match r {
                    Some(r) => agents::reply(r, mask),
                    None => Reply::Error(format!("The {name} workspace has stopped after an internal error (see app_status); the user can restart it from its tab.")),
                }
            }
        };
        call.answer(reply);
    }

    fn agent_status(&self, ctx: &egui::Context) -> Value {
        fn state<W: Workspace>(s: &Slot<W>) -> Value {
            match s {
                Slot::Live(w) => json!({"running": true, "busy": !w.script_idle()}),
                Slot::Crashed(r) => json!({"running": false, "stopped_by": polymix_workspace::guard::scrub_paths(&r.message), "at": r.location}),
            }
        }
        let r = ctx.content_rect();
        json!({
            "app": "Polymix",
            "version": env!("CARGO_PKG_VERSION"),
            "workspace_shown": self.active.key(),
            "workspaces": {"volume": state(&self.volume), "segmentation": state(&self.segmentation), "mesh": state(&self.mesh)},
            "window": {"width": r.width().round(), "height": r.height().round(), "pixels_per_point": ctx.pixels_per_point()},
            "identifiers_hidden_in_volume": self.volume.live().is_some_and(|w| w.hides_identifiers()),
        })
    }

    /// Whether agents may see the shown tab as it looks (screenshots, panel reads): the image
    /// tabs show patient identifiers unless *Hide identifiers* is on.
    fn identifiers_hidden(&self) -> Result<(), String> {
        let hiding = self.volume.live().is_some_and(|w| w.hides_identifiers());
        if matches!(self.active, Tab::Volume | Tab::Segmentation) && !hiding {
            return Err(format!("The {} tab can show patient identifiers: agents see it only with Hide identifiers on (Volume tab toolbar). The Mesh tab is always allowed.", self.active.label()));
        }
        Ok(())
    }

    /// Answer `app_wait` calls: when every workspace has been idle for a few frames, or at
    /// their deadline.
    fn finish_waits(&mut self, ctx: &egui::Context) {
        if self.waits.is_empty() {
            return;
        }
        let now = Instant::now();
        let idle = self.all_idle();
        let mask = self.volume.live().map(|w| w.agent_mask());
        for mut w in std::mem::take(&mut self.waits) {
            w.quiet = if idle { w.quiet + 1 } else { 0 };
            if w.quiet >= 3 || now >= w.until {
                let v = json!({"idle": w.quiet >= 3, "waited_s": polymix_workspace::agent::round((now - w.start).as_secs_f64(), 2), "status": self.agent_status(ctx)});
                w.call.answer(agents::masked(Ok(v), mask.as_ref()));
            } else {
                self.waits.push(w);
            }
        }
        if !self.waits.is_empty() {
            ctx.request_repaint_after(std::time::Duration::from_millis(30));
        }
    }

    fn all_idle(&self) -> bool {
        self.volume.live().is_none_or(|w| w.script_idle()) && self.segmentation.live().is_none_or(|w| w.script_idle()) && self.mesh.live().is_none_or(|w| w.script_idle())
    }

    /// Ask for a screenshot of the next frame; the callback hands it to the server to encode.
    fn agent_screenshot(&mut self, call: Call, ctx: &egui::Context) {
        if let Err(e) = self.identifiers_hidden() {
            return call.answer(Reply::Error(e));
        }
        let max = polymix_workspace::agent::arg_usize(&call.args, "max_size", 1568, 256, 4096);
        let slot = Arc::new(Mutex::new(Some(call)));
        self.shots.push(slot.clone());
        ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::ScreenshotCallback::new(move |img| {
            if let Some(call) = slot.lock().unwrap_or_else(|e| e.into_inner()).take() {
                call.answer(Reply::Later(Box::new(move || agents::png(&img, max))));
            }
        })));
        ctx.request_repaint();
    }

    /// Agents' requests that need the window drawn (screenshots, panels, and the agent's work
    /// in the Mesh workspace: tool previews, clicks, files opening) get it drawn, or are told
    /// why it is not: eframe draws nothing while the window is covered or minimized. With
    /// "Bring Polymix forward" on, the window first comes to the front, keyboard focus staying
    /// where it is.
    fn watch_drawing(&mut self, ctx: &egui::Context) {
        self.shots.retain(|s| s.lock().unwrap_or_else(|e| e.into_inner()).is_some());
        let agent_work = self.agent_work_until.is_some_and(|t| Instant::now() < t) || !self.waits.is_empty();
        let mesh_busy = self.mesh.live().is_some_and(|w| !w.script_idle());
        if self.shots.is_empty() && !self.ui_access.is_waiting() && !(agent_work && mesh_busy) {
            self.want_frame = None;
            self.brought_forward = false;
            return;
        }
        let waited = self.want_frame.get_or_insert_with(Instant::now).elapsed();
        ctx.request_repaint_after(Duration::from_millis(100));
        let bring = self.agents.brings_forward();
        if bring && !self.brought_forward && waited > BRING_FORWARD_AFTER {
            agents::bring_to_front();
            self.brought_forward = true;
        }
        if waited < if bring { NOT_DRAWN_AFTER_BRINGING } else { NOT_DRAWN_AFTER } {
            return;
        }
        log::info!("agent work waits for frames, and the window is not drawn");
        for slot in self.shots.drain(..) {
            if let Some(call) = slot.lock().unwrap_or_else(|e| e.into_inner()).take() {
                call.answer(Reply::Error(agents::not_drawn("take a screenshot")));
            }
        }
        self.ui_access.fail_waiting(&agents::not_drawn("read or work its panels"));
        if !self.all_idle() {
            let mask = self.volume.live().map(|w| w.agent_mask());
            for w in std::mem::take(&mut self.waits) {
                let v = json!({"idle": false, "drawn": false, "note": agents::not_drawn("finish tool previews, clicks or opening files"), "status": self.agent_status(ctx)});
                w.call.answer(agents::masked(Ok(v), mask.as_ref()));
            }
        }
        // Start over: the next request gets its own chance.
        self.want_frame = None;
        self.brought_forward = false;
    }

    /// Whether the shown workspace has finished the script steps handed to it.
    fn active_idle(&self) -> bool {
        match self.active {
            Tab::Volume => self.volume.live().is_none_or(|w| w.script_idle()),
            Tab::Segmentation => self.segmentation.live().is_none_or(|w| w.script_idle()),
            Tab::Mesh => self.mesh.live().is_none_or(|w| w.script_idle()),
        }
    }

    fn run_script(&mut self, ctx: &egui::Context) {
        let idle = self.active_idle();
        let frames = self.frames;
        let Some(script) = &mut self.script else { return };
        if let Some(until) = script.idle_until {
            let now = ctx.input(|i| i.time);
            if now < until {
                ctx.request_repaint_after(std::time::Duration::from_secs_f64(until - now));
                return;
            }
            script.idle_until = None;
        }
        ctx.request_repaint();
        // Wait for an in-flight screenshot to be written.
        if let Some(done) = &script.awaiting_shot {
            if !done.load(Ordering::SeqCst) {
                return;
            }
            script.awaiting_shot = None;
        }
        if script.wait > 0 {
            script.wait -= 1;
            return;
        }
        if !script.pointer.is_empty() {
            return;
        }
        if !idle {
            return;
        }
        let Some(step) = script.steps.pop_front() else {
            self.script = None;
            return;
        };
        let (cmd, arg) = step.split_once(':').unwrap_or((step.as_str(), ""));
        match cmd {
            "wait" => script.wait = arg.parse().unwrap_or(1),
            "idle" => script.idle_until = Some(ctx.input(|i| i.time) + arg.parse::<f64>().unwrap_or(1.0)),
            // frames:<label> — how many frames have been drawn so far.
            "frames" => println!("[shell] {arg} frames={frames}"),
            "key" => {
                if let Some(k) = egui::Key::from_name(arg) {
                    script.keys.push(k);
                }
                script.wait = 1;
            }
            "shot" => {
                let done = Arc::new(AtomicBool::new(false));
                script.awaiting_shot = Some(done.clone());
                let path = PathBuf::from(arg);
                let ctx2 = ctx.clone();
                ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::ScreenshotCallback::new(move |img| {
                    if let Err(e) = save_png(&path, &img) {
                        log::error!("screenshot failed: {e}");
                    }
                    done.store(true, Ordering::SeqCst);
                    ctx2.request_repaint();
                })));
            }
            "quit" => ctx.send_viewport_cmd(egui::ViewportCommand::Close),
            // window:minimize / window:restore — e.g. to test what happens while it is not drawn.
            "window" => ctx.send_viewport_cmd(egui::ViewportCommand::Minimized(arg == "minimize")),
            // ws:volume — show a workspace.
            "ws" => {
                if let Some(t) = Tab::from_key(arg) {
                    self.active = t;
                }
                script.wait = 2;
            }
            // crash:volume / restart:volume — test the crash guard.
            "crash" => {
                self.crash_next = Tab::from_key(arg);
                script.wait = 2;
            }
            "restart" => {
                self.restart_next = Tab::from_key(arg);
                script.wait = 2;
            }
            // pointer:left+right,x0,y0,x1,y1[,steps] — press the buttons at (x0, y0) (window
            // points), move to (x1, y1) in `steps` frames, release: real mouse input for any
            // workspace. Buttons: left, right, middle, joined with `+`; `,hold` at the end keeps
            // them down until `pointer:release,x,y`.
            // wheel:x,y,lines — the mouse wheel at (x, y) (window points), one notch per line
            // (positive = up).
            "wheel" => {
                let n: Vec<f32> = arg.split(',').filter_map(|x| x.trim().parse().ok()).collect();
                if let [x, y, lines] = n[..] {
                    let at = egui::pos2(x, y);
                    let notch = egui::Event::MouseWheel { unit: egui::MouseWheelUnit::Line, delta: egui::vec2(0.0, lines.signum()), modifiers: Default::default(), phase: egui::TouchPhase::Move, source: egui::MouseWheelSource::Wheel };
                    script.pointer = std::iter::once(vec![egui::Event::PointerMoved(at)]).chain((0..lines.abs() as usize).map(|_| vec![notch.clone()])).collect();
                }
            }
            "pointer" => match pointer_frames(arg) {
                Some(frames) => script.pointer = frames,
                None => log::warn!("bad pointer step {step}"),
            },
            _ => {
                let taken = match self.active {
                    Tab::Volume => self.volume.call(|w| w.script_push(step.clone())),
                    Tab::Segmentation => self.segmentation.call(|w| w.script_push(step.clone())),
                    Tab::Mesh => self.mesh.call(|w| w.script_push(step.clone())),
                };
                if taken.is_none() {
                    log::warn!("script step {step} dropped: the {} workspace is not running", self.active.label());
                }
            }
        }
    }
}

/// Start a workspace under the crash guard.
fn slot<W: Workspace>(name: &'static str, make: impl FnOnce() -> W) -> Slot<W> {
    match guard::run_guarded(name, make) {
        Ok(w) => Slot::Live(w),
        Err(report) => Slot::Crashed(report),
    }
}

impl eframe::App for PolymixApp {
    /// Background work for every workspace, every frame (also while the window is hidden).
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let active = self.active;
        // Segmentation shows the volume selected in Volume (read only while it is shown, so
        // browsing in Volume loads nothing).
        let shown = if active == Tab::Segmentation { self.volume.call(|w| w.current_volume()).flatten() } else { None };
        if let Some((catalog, k, label)) = &shown {
            self.segmentation.call(|w| w.show_volume(&catalog.volumes[*k], label, ctx));
        }
        // Images that can be shown over the one in Segmentation (its Overlay menu): listed
        // again when asked, when another is shown, when the catalog changes, or when *Hide
        // identifiers* changes how they are labelled.
        let asked = self.segmentation.call(|w| w.wants_overlay_choices()) == Some(true);
        if active == Tab::Segmentation || asked {
            // Showing it in Segmentation changed nothing in Volume: no need to ask again.
            let current = if active == Tab::Segmentation { shown } else { self.volume.call(|w| w.current_volume()).flatten() };
            let hide = self.volume.live().is_some_and(|w| w.hides_identifiers());
            let same = match (&current, &self.overlay_key) {
                (Some((c, k, _)), Some(key)) => key.is(c, *k, hide),
                (None, None) => true,
                _ => false,
            };
            if asked || !same {
                self.overlay_key = current.as_ref().map(|(c, k, _)| OverlayKey::new(c, *k, hide));
                let choices = current.and_then(|(c, k, _)| self.volume.call(|w| w.overlay_choices(&c.volumes[k]))).unwrap_or_default();
                self.segmentation.call(|w| w.set_overlay_choices(choices));
            }
        }
        // The Segmentation tab's image, for the image slices in Mesh.
        if active == Tab::Mesh {
            let source = self.segmentation.call(|w| w.view_for_mesh()).flatten().map(|i| crate::app::slices::SliceSource { volume: i.volume, name: i.name, cursor: i.cursor, window: i.window });
            self.mesh.call(|w| w.set_image_source(source));
        }
        // Meshes sent from Mesh become segments in Segmentation, which is shown.
        if let Some(meshes) = self.mesh.call(|w| w.take_for_segmentation()).filter(|m| !m.is_empty()) {
            self.segmentation.call(|w| {
                for (name, mesh) in meshes {
                    w.import_mesh(name, mesh, ctx);
                }
            });
            self.active = Tab::Segmentation;
        }
        // Images saved in Segmentation (Mask volume) are listed in Volume.
        if let Some(paths) = self.segmentation.call(|w| w.take_volumes()).filter(|p| !p.is_empty()) {
            self.volume.call(|w| w.add_derived(paths, false, ctx));
        }
        // Meshes made in Segmentation go to Mesh, which is shown.
        if let Some((meshes, by_agent)) = self.segmentation.call(|w| w.take_meshes()).filter(|(m, _)| !m.is_empty()) {
            self.mesh.call(|w| w.add_meshes(meshes, by_agent));
            self.active = Tab::Mesh;
        }
        self.volume.call(|w| w.tick(ctx, active == Tab::Volume));
        self.segmentation.call(|w| w.tick(ctx, active == Tab::Segmentation));
        self.mesh.call(|w| w.tick(ctx, active == Tab::Mesh));
        self.serve_agents(ctx);
        self.finish_waits(ctx);
        self.ui_access.tick(ctx);
        self.watch_drawing(ctx);
    }

    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.frames += 1;
        // Drawn: whatever waited for a frame gets it now.
        self.want_frame = None;
        self.brought_forward = false;
        // Agents' panel actions, for the widgets about to be drawn.
        let actions = self.ui_access.take_actions();
        if !actions.is_empty() {
            ctx.input_mut(|i| i.events.extend(actions));
        }
        let mut open: Vec<PathBuf> = std::mem::take(&mut self.pending_open);
        #[cfg(target_os = "macos")]
        open.extend(crate::macos_open::take());
        open.extend(ctx.input(|i| i.raw.dropped_files.iter().map(|f| f.path().to_path_buf()).filter(|p| !p.as_os_str().is_empty()).collect::<Vec<_>>()));
        if !open.is_empty() {
            self.open_paths(open, &ctx);
        }
        for tab in Tab::ALL {
            if ctx.input_mut(|i| i.consume_key(egui::Modifiers::COMMAND, tab.hotkey())) {
                self.active = tab;
            }
        }
        self.tab_bar(ui);
        self.workspace_ui(ui, frame);
        self.agents.window(&ctx);
        self.update_title(&ctx);
        self.run_script(&ctx);
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        if polymix_workspace::scripted() {
            return;
        }
        storage.set_string("shell.tab", self.active.key().to_string());
        self.agents.save(storage);
        self.volume.call(|w| w.save(storage));
        self.segmentation.call(|w| w.save(storage));
        self.mesh.call(|w| w.save(storage));
    }

    fn on_exit(&mut self) {
        self.agents.on_exit();
        self.volume.call(|w| w.on_exit());
        self.segmentation.call(|w| w.on_exit());
        self.mesh.call(|w| w.on_exit());
    }

    fn raw_input_hook(&mut self, ctx: &egui::Context, raw_input: &mut egui::RawInput) {
        // 3D mouse buttons held as modifier keys (Preferences → 3D mouse → Buttons), in every
        // tab. Out of focus none counts as held: the driver would not report one let go.
        if !raw_input.focused {
            crate::spacemouse::release_buttons();
        }
        let held = crate::spacemouse::held_modifiers();
        crate::spacemouse::with_held_keys(&mut raw_input.events, &mut self.keyboard_modifiers, held, self.held_modifiers);
        self.held_modifiers = held;
        // Keys 3D mouse buttons press: down this frame, up the next.
        crate::spacemouse::key_events(&mut raw_input.events, &mut self.keys_down, crate::spacemouse::take_keys(), self.keyboard_modifiers | held);
        if !self.keys_down.is_empty() {
            ctx.request_repaint();
        }
        // Only the Mesh tab acts on button presses: elsewhere they are not kept for later.
        if self.active != Tab::Mesh {
            crate::spacemouse::drop_presses();
        }
        if let Some(script) = &mut self.script {
            for key in script.keys.drain(..) {
                for pressed in [true, false] {
                    raw_input.events.push(egui::Event::Key { key, physical_key: None, pressed, repeat: false, modifiers: Default::default() });
                }
            }
            if let Some(events) = script.pointer.pop_front() {
                raw_input.events.extend(events);
            }
        }
    }
}

/// The frames of a `pointer:` script step: move to the start, press the buttons one frame
/// apart (as hands do), move to the end, release, and one quiet frame. With `hold` last the
/// buttons stay down (for a screenshot mid-drag) until a `pointer:release,x,y,x,y` step.
fn pointer_frames(arg: &str) -> Option<VecDeque<Vec<egui::Event>>> {
    let mut parts: Vec<&str> = arg.split(',').map(str::trim).collect();
    let hold = parts.last() == Some(&"hold");
    if hold {
        parts.pop();
    }
    if parts.first() == Some(&"release") {
        let n: Vec<f32> = parts[1..].iter().map(|x| x.parse().ok()).collect::<Option<_>>()?;
        let pos = egui::pos2(*n.first()?, *n.get(1)?);
        let up = [egui::PointerButton::Primary, egui::PointerButton::Secondary, egui::PointerButton::Middle].map(|button| egui::Event::PointerButton { pos, button, pressed: false, modifiers: Default::default() });
        return Some(VecDeque::from([up.to_vec(), Vec::new()]));
    }
    let buttons: Vec<egui::PointerButton> = parts.first()?.split('+').map(|b| match b {
        "left" => Some(egui::PointerButton::Primary),
        "right" => Some(egui::PointerButton::Secondary),
        "middle" => Some(egui::PointerButton::Middle),
        _ => None,
    }).collect::<Option<_>>()?;
    let n: Vec<f32> = parts[1..].iter().map(|x| x.parse().ok()).collect::<Option<_>>()?;
    let (a, b) = (egui::pos2(*n.first()?, *n.get(1)?), egui::pos2(*n.get(2)?, *n.get(3)?));
    let steps = n.get(4).map_or(10, |s| *s as usize);
    let button = |button, pos, pressed| egui::Event::PointerButton { pos, button, pressed, modifiers: Default::default() };
    let mut frames = VecDeque::new();
    frames.push_back(vec![egui::Event::PointerMoved(a)]);
    for &bt in &buttons {
        frames.push_back(vec![button(bt, a, true)]);
    }
    for s in 1..=steps {
        frames.push_back(vec![egui::Event::PointerMoved(a.lerp(b, s as f32 / steps as f32))]);
    }
    if !hold {
        frames.push_back(buttons.iter().map(|&bt| button(bt, b, false)).collect());
    }
    frames.push_back(Vec::new());
    Some(frames)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlay_key_tells_catalogs_volumes_and_masking_apart() {
        let a = Arc::new(Catalog::default());
        let key = OverlayKey::new(&a, 3, false);
        assert!(key.is(&a, 3, false));
        assert!(!key.is(&a, 4, false));
        assert!(!key.is(&a, 3, true), "Hide identifiers relabels the choices");
        // A rescan: equal contents, another catalog.
        assert!(!key.is(&Arc::new(Catalog::default()), 3, false));
        // Catalogs made after the remembered one is dropped never take its address.
        drop(a);
        let later: Vec<Arc<Catalog>> = (0..64).map(|_| Arc::new(Catalog::default())).collect();
        assert!(later.iter().all(|c| !key.is(c, 3, false)));
    }
}
