//! The application shell: the workspace tabs (Volume | Segmentation | Mesh) above
//! independent workspaces.
//!
//! The shell owns the window-level things every workspace shares: the tab bar and its
//! hotkeys (⌘1 / ⌘2 / ⌘3), routing of dropped files and command-line paths, the window title,
//! persistence of the chosen tab, and the test-script runner (`POLYMIX_SCRIPT`). Each
//! workspace is called through its [`Slot`], so a panic inside one stops only that one: its
//! tab shows a crash card with a Restart button while the others keep working.

use crate::app::MeshWorkspace;
use polymix_segmentation::SegmentationWorkspace;
use polymix_volume::VolumeWorkspace;
use polymix_workspace::guard::{self, CardAction, Slot};
use polymix_workspace::style::{apply_theme, Theme};
use polymix_workspace::Workspace;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

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
    script: Option<Script>,
    /// `crash:<workspace>` (test scripts): make that workspace panic on its next frame.
    crash_next: Option<Tab>,
    /// `restart:<workspace>` (test scripts): press its crash card's Restart.
    restart_next: Option<Tab>,
    title: String,
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
        Self { active, volume, segmentation, mesh, pending_open: files, script, crash_next: None, restart_next: None, title: String::new() }
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
        let Some(script) = &mut self.script else { return };
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
            // workspace. Buttons: left, right, middle, joined with `+`.
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
        if active == Tab::Segmentation {
            if let Some((catalog, k, label)) = self.volume.call(|w| w.current_volume()).flatten() {
                self.segmentation.call(|w| w.show_volume(&catalog.volumes[k], &label, ctx));
            }
        }
        self.volume.call(|w| w.tick(ctx, active == Tab::Volume));
        self.segmentation.call(|w| w.tick(ctx, active == Tab::Segmentation));
        self.mesh.call(|w| w.tick(ctx, active == Tab::Mesh));
    }

    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
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
        self.update_title(&ctx);
        self.run_script(&ctx);
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        if polymix_workspace::scripted() {
            return;
        }
        storage.set_string("shell.tab", self.active.key().to_string());
        self.volume.call(|w| w.save(storage));
        self.segmentation.call(|w| w.save(storage));
        self.mesh.call(|w| w.save(storage));
    }

    fn on_exit(&mut self) {
        self.volume.call(|w| w.on_exit());
        self.segmentation.call(|w| w.on_exit());
        self.mesh.call(|w| w.on_exit());
    }

    fn raw_input_hook(&mut self, _ctx: &egui::Context, raw_input: &mut egui::RawInput) {
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
/// apart (as hands do), move to the end, release, and one quiet frame.
fn pointer_frames(arg: &str) -> Option<VecDeque<Vec<egui::Event>>> {
    let parts: Vec<&str> = arg.split(',').map(str::trim).collect();
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
    frames.push_back(buttons.iter().map(|&bt| button(bt, b, false)).collect());
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
