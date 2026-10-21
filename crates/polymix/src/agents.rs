//! Agent access, the shell's side: turning the MCP server ([`polymix_mcp`]) on and off, its
//! token, the Agents window, the app-wide tools (`app_status`, `app_screenshot`, `app_log`)
//! and the privacy pass over everything agents are sent.
//!
//! Off by default. On, it listens on 127.0.0.1 and (unless the user turns it off) needs a token
//! kept in the data folder (`agent-token`), so only programs of this user on this computer can
//! connect; web pages never can. *Add to Claude Code* registers it with the `claude` command
//! for all the user's projects. Agents read;
//! they change things only once the user also ticks *Let agents make changes*, and every
//! change is an undo step labelled "Agent: …". `POLYMIX_MCP=1` turns access on for one
//! session (scripted runs, development), `POLYMIX_MCP_WRITE=1` changes too; `POLYMIX_MCP_PORT`
//! picks the port.

use polymix_mcp::{Reply, Server, ToolSpec};
use polymix_workspace::jobs::Job;
use polymix_workspace::agent::{AgentReply, AgentTool, ToolKind};
use polymix_workspace::Workspace;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

/// How long an agent waits for the UI thread before it is told the app is busy.
const TIMEOUT: Duration = Duration::from_secs(30);
/// Masks patients' identifiers in text (see `VolumeWorkspace::agent_mask`).
pub type Mask = Arc<dyn Fn(&str) -> String + Send + Sync>;

const INSTRUCTIONS: &str = "Polymix is a desktop app for medical images and meshes: the Volume workspace finds and previews images (DICOM, NIfTI, …), Segmentation segments one in three planes, and Mesh edits surface meshes (vessel cleanup for CFD). The tools work on the session the user has open, in front of them. Start with app_status; app_screenshot shows the window as the user sees it, app_ui lists the widgets of the shown panels. Tools that change things work only if the user allowed changes (Agents window); each change is an undo step labelled \"Agent: …\" and shows the workspace it changes. Typed tools first: each workspace's state tool (mesh_scene / mesh_tool_state, volume_catalog, segmentation_state) reports everything it holds and shows, and mesh_tool run, volume_do and segmentation_do list and run every action with typed arguments; app_ui_action only for what they lack. Work that takes time (tools computing, files loading, clicks) finishes after the call returns: call app_wait, then read the result. Lengths are in millimetres. Patient identifiers are masked and dates cut to the year: do not try to recover them.";

/// Every tool agents get: the shell's and each workspace's (listed even for a workspace that
/// failed to start).
pub fn all_tools() -> Vec<AgentTool> {
    let mut all = shell_tools();
    all.extend(workspace_tools::<polymix_volume::VolumeWorkspace>("volume"));
    all.extend(workspace_tools::<polymix_segmentation::SegmentationWorkspace>("segmentation"));
    all.extend(workspace_tools::<crate::app::MeshWorkspace>("mesh"));
    all
}

/// A workspace's tools: those it lists, and `<key>_do`, made from its actions.
fn workspace_tools<W: Workspace>(key: &str) -> Vec<AgentTool> {
    let mut tools = W::agent_tools();
    let actions = W::agent_actions();
    if !actions.is_empty() {
        tools.push(polymix_workspace::agent::do_tool(key, W::agent_actions_about(), &actions));
    }
    tools
}

/// Answer a call of one of a workspace's tools (`<key>_do` runs one of its actions, checked).
pub fn workspace_call<W: Workspace>(w: &mut W, tool: &str, args: &Value, ctx: &egui::Context) -> AgentReply {
    if !tool.ends_with("_do") {
        return w.agent_call(tool, args, ctx);
    }
    match polymix_workspace::agent::find_action(&W::agent_actions(), args) {
        Ok((a, args)) => w.agent_act(a.name, &args, ctx),
        Err(e) => AgentReply::error(e),
    }
}

pub fn shell_tools() -> Vec<AgentTool> {
    vec![
        AgentTool::new("app_status", "Polymix as a whole: version, which workspace tab is shown (Volume, Segmentation or Mesh), whether each workspace is running, busy or stopped by an internal error, and the window size.", json!({})),
        AgentTool::new(
            "app_screenshot",
            "A PNG of the Polymix window as the user sees it now (the shown workspace tab only). Refused for the Volume and Segmentation tabs unless Hide identifiers is on in the Volume tab.",
            json!({"max_size": {"type": "integer", "description": "Longest side in pixels (default 1568, 256 to 4096)"}}),
        ),
        AgentTool::new(
            "app_wait",
            "Wait until Polymix has finished what was asked (tools computing, files loading or saving, clicks), up to `seconds`. Returns whether it is idle, and app_status.",
            json!({"seconds": {"type": "integer", "description": "Longest wait (default 10, up to 25)"}}),
        ),
        AgentTool::new(
            "app_ui",
            "The widgets of the panels shown now (the shown tab, its tool panel and any open windows), as indented lines: role \"label\" = value, state (checked, selected, disabled…) and an #id to act on with app_ui_action. Refused for the Volume and Segmentation tabs unless Hide identifiers is on.",
            json!({
                "contains": {"type": "string", "description": "Only lines containing this text (case-insensitive)"},
                "limit": {"type": "integer", "description": "Most lines (default 400)"}
            }),
        ),
        AgentTool::new(
            "app_ui_action",
            "Act on a widget listed by app_ui, as a screen reader would: click (buttons, checkboxes, menu entries, combo boxes and their options), set_value (`value`: a number for sliders and number fields, text for text boxes), increment, decrement, expand, collapse, focus. Changes nothing until the user allowed changes; never in the Agent access window. Buttons that open a file dialog wait for the user.",
            json!({
                "id": {"type": "string", "description": "The widget's #id from app_ui"},
                "action": {"type": "string", "enum": ["click", "set_value", "increment", "decrement", "expand", "collapse", "focus", "scroll_into_view"], "description": "Default click"},
                "value": {"type": ["number", "string"], "description": "For set_value"}
            }),
        )
        .required(&["id"])
        .kind(ToolKind::Destructive),
        AgentTool::new(
            "app_show",
            "Show a workspace tab (volume, segmentation or mesh). Tools that change a workspace show it themselves.",
            json!({"workspace": {"type": "string", "enum": ["volume", "segmentation", "mesh"]}}),
        )
        .required(&["workspace"])
        .kind(ToolKind::Change),
        AgentTool::new(
            "app_log",
            "Recent log messages: Polymix's own from info up, other libraries' warnings and errors. File paths are replaced with <path>.",
            json!({
                "lines": {"type": "integer", "description": "How many of the latest messages (default 100, up to 2000)"},
                "level": {"type": "string", "enum": ["error", "warn", "info"], "description": "Least severe level shown (default info)"}
            }),
        ),
    ]
}

pub fn spec(t: AgentTool) -> ToolSpec {
    ToolSpec { name: t.name.into(), description: t.description.into(), input_schema: t.input_schema, read_only: t.kind == ToolKind::Read, destructive: t.kind == ToolKind::Destructive }
}

/// A workspace's answer as the server takes it, masked.
pub fn reply(r: AgentReply, mask: Option<Mask>) -> Reply {
    match r {
        AgentReply::Now(r) => masked(r, mask.as_ref()),
        AgentReply::Later(work) => Reply::Later(Box::new(move || masked(work(), mask.as_ref()))),
    }
}

pub fn masked(r: Result<Value, String>, mask: Option<&Mask>) -> Reply {
    match (r, mask) {
        (Ok(mut v), Some(m)) => {
            mask_strings(&mut v, m.as_ref());
            Reply::Json(v)
        }
        (Ok(v), None) => Reply::Json(v),
        (Err(e), Some(m)) => Reply::Error(m(&e)),
        (Err(e), None) => Reply::Error(e),
    }
}

fn mask_strings(v: &mut Value, mask: &(dyn Fn(&str) -> String + Send + Sync)) {
    match v {
        Value::String(s) => *s = mask(s),
        Value::Array(a) => a.iter_mut().for_each(|x| mask_strings(x, mask)),
        Value::Object(o) => o.values_mut().for_each(|x| mask_strings(x, mask)),
        _ => {}
    }
}

pub fn log(args: &Value) -> Value {
    let n = polymix_workspace::agent::arg_usize(args, "lines", 100, 1, 2000);
    let level = match polymix_workspace::agent::arg_str(args, "level") {
        Some("error") => log::Level::Error,
        Some("warn") => log::Level::Warn,
        _ => log::Level::Info,
    };
    let now = SystemTime::now();
    let entries: Vec<Value> = crate::recent_log::recent(n, level)
        .into_iter()
        .map(|e| json!({"seconds_ago": now.duration_since(e.at).map_or(0.0, |d| polymix_workspace::agent::round(d.as_secs_f64(), 1)), "level": e.level.as_str().to_lowercase(), "target": e.target, "message": e.message}))
        .collect();
    json!({"entries": entries})
}

/// The screenshot as a PNG at most `max_side` pixels across (box-filtered down by a whole
/// factor, which keeps text legible).
pub fn png(img: &egui::ColorImage, max_side: usize) -> Reply {
    let [w, h] = img.size;
    if w == 0 || h == 0 {
        return Reply::Error("The window has no size (is it minimized?).".into());
    }
    let f = w.max(h).div_ceil(max_side.max(1)).max(1);
    let (ow, oh) = (w / f, h / f);
    let mut rgb = Vec::with_capacity(ow * oh * 3);
    for y in 0..oh {
        for x in 0..ow {
            let mut sum = [0u32; 3];
            for dy in 0..f {
                for dx in 0..f {
                    let c = img.pixels[(y * f + dy) * w + x * f + dx];
                    sum[0] += c.r() as u32;
                    sum[1] += c.g() as u32;
                    sum[2] += c.b() as u32;
                }
            }
            let n = (f * f) as u32;
            rgb.extend(sum.map(|s| ((s + n / 2) / n) as u8));
        }
    }
    let mut out = Vec::new();
    let mut enc = png::Encoder::new(&mut out, ow as u32, oh as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    let written = enc.write_header().and_then(|mut wr| wr.write_image_data(&rgb));
    match written {
        Ok(()) => Reply::Png(out),
        Err(e) => Reply::Error(format!("could not encode the screenshot: {e}")),
    }
}

/// `claude` arguments that register Polymix with Claude Code for all projects. The header
/// comes last: `--header` takes every value after it.
fn claude_args(url: &str, token: Option<&str>) -> Vec<String> {
    let mut a: Vec<String> = ["mcp", "add", "--transport", "http", "--scope", "user", "polymix", url].map(String::from).to_vec();
    if let Some(t) = token {
        a.extend(["--header".to_string(), format!("Authorization: Bearer {t}")]);
    }
    a
}

/// The same as a command to paste into a terminal.
fn claude_command(url: &str, token: Option<&str>) -> String {
    claude_args(url, token).iter().fold("claude".to_string(), |c, a| if a.contains(' ') { format!("{c} \"{a}\"") } else { format!("{c} {a}") })
}

/// Run `cmd`, giving up after `timeout` (a login shell's profile could hang).
fn run_for(cmd: &mut std::process::Command, timeout: Duration) -> anyhow::Result<std::process::Output> {
    use std::process::Stdio;
    let mut child = cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn()?;
    let start = std::time::Instant::now();
    while child.try_wait()?.is_none() {
        if start.elapsed() > timeout {
            let _ = child.kill();
            anyhow::bail!("timed out");
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    Ok(child.wait_with_output()?)
}

/// Claude Code's `claude` command. Apps opened from the Finder get a bare PATH, so its usual
/// install places and the login shell's PATH are searched too.
fn find_claude() -> Option<PathBuf> {
    let exe = if cfg!(windows) { "claude.exe" } else { "claude" };
    let mut dirs: Vec<PathBuf> = std::env::var_os("PATH").map(|p| std::env::split_paths(&p).collect()).unwrap_or_default();
    if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
        dirs.extend([".local/bin", ".claude/local", ".npm-global/bin", ".bun/bin", ".volta/bin"].map(|d| home.join(d)));
    }
    dirs.extend(["/opt/homebrew/bin", "/usr/local/bin"].map(PathBuf::from));
    if let Some(p) = dirs.iter().map(|d| d.join(exe)).find(|p| p.is_file()) {
        return Some(p);
    }
    // nvm and the like set PATH in shell profiles.
    let shell = std::env::var("SHELL").ok().filter(|_| cfg!(unix))?;
    ["-lc", "-ilc"].into_iter().find_map(|flags| {
        let out = run_for(std::process::Command::new(&shell).args([flags, "command -v claude"]), Duration::from_secs(10)).ok()?;
        String::from_utf8_lossy(&out.stdout).lines().rev().map(|l| PathBuf::from(l.trim())).find(|p| p.is_absolute() && p.is_file())
    })
}

/// Register Polymix with Claude Code for all projects, replacing an earlier registration.
fn add_to_claude(url: &str, token: Option<&str>) -> anyhow::Result<String> {
    let claude = find_claude().ok_or_else(|| anyhow::anyhow!("Claude Code's claude command was not found: install Claude Code, or copy the command below into a terminal"))?;
    // Whatever `claude` needs (node, for npm installs) is most likely next to it.
    let mut path: Vec<PathBuf> = claude.parent().map(PathBuf::from).into_iter().chain(["/opt/homebrew/bin", "/usr/local/bin"].map(PathBuf::from)).collect();
    path.extend(std::env::var_os("PATH").map(|p| std::env::split_paths(&p).collect::<Vec<_>>()).unwrap_or_default());
    let path = std::env::join_paths(path)?;
    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(std::env::temp_dir);
    let claude_cmd = || {
        let mut c = std::process::Command::new(&claude);
        c.env("PATH", &path).current_dir(&home);
        c
    };
    // Adding over an existing registration fails: remove it first (there may be none).
    let _ = run_for(claude_cmd().args(["mcp", "remove", "polymix", "--scope", "user"]), Duration::from_secs(30));
    let out = run_for(claude_cmd().args(claude_args(url, token)), Duration::from_secs(30))?;
    if !out.status.success() {
        let why = String::from_utf8_lossy(if out.stderr.is_empty() { &out.stdout } else { &out.stderr }).trim().to_string();
        anyhow::bail!("claude mcp add failed: {why}");
    }
    Ok("Added to Claude Code for all your projects. Start a new Claude Code session to use it.".into())
}

/// Why an agent's request cannot be done while the window is not drawn.
pub fn not_drawn(what: &str) -> String {
    format!("Polymix's window is covered by other windows or minimized, so it is not drawn and cannot {what} until it is. Ask the user to bring it into view (any part of it showing is enough; it does not need the keyboard focus), or to tick \"Bring Polymix forward when an agent needs it\" in its Agents window. Reading tools such as mesh_scene work meanwhile.")
}

/// Bring the app's windows to the front of other apps' windows without making Polymix the
/// active app: the keyboard stays where the user is typing. Minimized windows stay put.
#[cfg(target_os = "macos")]
pub fn bring_to_front() {
    use objc2::msg_send;
    use objc2::runtime::{AnyClass, AnyObject};
    let Some(class) = AnyClass::get(c"NSApplication") else { return };
    // SAFETY: plain AppKit calls on the main thread (the shell's logic), on objects AppKit owns.
    unsafe {
        let app: *mut AnyObject = msg_send![class, sharedApplication];
        if app.is_null() {
            return;
        }
        let windows: *mut AnyObject = msg_send![&*app, windows];
        if windows.is_null() {
            return;
        }
        let n: usize = msg_send![&*windows, count];
        for i in 0..n {
            let w: *mut AnyObject = msg_send![&*windows, objectAtIndex: i];
            if w.is_null() {
                continue;
            }
            let visible: bool = msg_send![&*w, isVisible];
            let minimized: bool = msg_send![&*w, isMiniaturized];
            if visible && !minimized {
                let _: () = msg_send![&*w, orderFrontRegardless];
            }
        }
    }
}

#[cfg(not(target_os = "macos"))]
pub fn bring_to_front() {}

/// The token file in the data folder.
fn token_path() -> Option<PathBuf> {
    polymix_workspace::data_dir().map(|d| d.join("agent-token"))
}

/// The saved token, or a new one (saved, readable by this user only).
fn load_token(fresh: bool) -> anyhow::Result<String> {
    let path = token_path();
    if !fresh {
        if let Some(t) = path.as_ref().and_then(|p| std::fs::read_to_string(p).ok()).map(|t| t.trim().to_string()).filter(|t| t.len() >= 32) {
            return Ok(t);
        }
    }
    let token = polymix_mcp::new_token()?;
    if let Some(p) = path {
        if let Some(dir) = p.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let mut opts = std::fs::OpenOptions::new();
        opts.write(true).create(true).truncate(true);
        #[cfg(unix)]
        std::os::unix::fs::OpenOptionsExt::mode(&mut opts, 0o600);
        std::io::Write::write_all(&mut opts.open(&p)?, token.as_bytes())?;
    }
    Ok(token)
}

pub struct AgentAccess {
    enabled: bool,
    /// Agents may change things (tools other than [`ToolKind::Read`]).
    allow_changes: bool,
    port: u16,
    /// Clients must send the token (otherwise any local program may connect; web pages never).
    require_token: bool,
    /// Bring the window to the front (without the keyboard focus) when an agent's work waits
    /// for it to be drawn.
    bring_forward: bool,
    /// What *Add to Claude Code* last registered ("port:token start"), to notice when Claude
    /// Code needs updating.
    claude_setup: Option<String>,
    claude_job: Option<Job<String>>,
    /// The last *Add to Claude Code* outcome, and whether it failed.
    claude_message: Option<(String, bool)>,
    /// Turned on by `POLYMIX_MCP` for this session only: never saved.
    from_env: bool,
    tools: Vec<ToolSpec>,
    kinds: HashMap<String, ToolKind>,
    server: Option<Server>,
    token: Option<String>,
    error: Option<String>,
    pub show_window: bool,
}

impl AgentAccess {
    pub fn new(storage: Option<&dyn eframe::Storage>, tools: Vec<AgentTool>) -> Self {
        let env = |k: &str| std::env::var(k).is_ok_and(|v| v == "1");
        let (env_on, env_write) = (env("POLYMIX_MCP"), env("POLYMIX_MCP_WRITE"));
        let env_port = std::env::var("POLYMIX_MCP_PORT").ok().and_then(|p| p.parse().ok());
        let stored = |k: &str| storage.filter(|_| !polymix_workspace::scripted()).and_then(|s| s.get_string(k));
        Self {
            enabled: env_on || stored("agents.enabled").is_some_and(|v| v == "1"),
            allow_changes: env_write || stored("agents.changes").is_some_and(|v| v == "1"),
            port: env_port.or_else(|| stored("agents.port").and_then(|p| p.parse().ok())).unwrap_or(polymix_mcp::DEFAULT_PORT),
            require_token: stored("agents.require_token").is_none_or(|v| v != "0"),
            bring_forward: stored("agents.bring_forward").is_some_and(|v| v == "1"),
            claude_setup: stored("agents.claude_setup").filter(|s| !s.is_empty()),
            claude_job: None,
            claude_message: None,
            from_env: env_on || env_write || env_port.is_some(),
            kinds: tools.iter().map(|t| (t.name.to_string(), t.kind)).collect(),
            tools: tools.into_iter().map(spec).collect(),
            server: None,
            token: None,
            error: None,
            show_window: false,
        }
    }

    pub fn server(&self) -> Option<&Server> {
        self.server.as_ref()
    }

    /// Why an agent may not call `tool` now, if it may not.
    pub fn refusal(&self, tool: &str) -> Option<String> {
        let kind = self.kinds.get(tool).copied().unwrap_or(ToolKind::Destructive);
        (kind != ToolKind::Read && !self.allow_changes).then(|| format!("{tool} changes the session, and the user has not allowed changes: ask them to tick \"Let agents make changes\" in Polymix's Agents window (Agents button, top right). Reading tools work meanwhile."))
    }

    pub fn kind(&self, tool: &str) -> ToolKind {
        self.kinds.get(tool).copied().unwrap_or(ToolKind::Destructive)
    }

    pub fn brings_forward(&self) -> bool {
        self.bring_forward && cfg!(target_os = "macos")
    }

    /// Start the server if agent access is on.
    pub fn start_if_enabled(&mut self, ctx: &egui::Context) {
        if self.enabled && self.server.is_none() {
            self.start(ctx, false);
        }
    }

    fn start(&mut self, ctx: &egui::Context, fresh_token: bool) {
        self.server = None;
        self.error = None;
        let token = match self.require_token.then(|| load_token(fresh_token)).transpose() {
            Ok(t) => t,
            Err(e) => {
                self.error = Some(format!("No token: {e}"));
                return;
            }
        };
        let config = polymix_mcp::Config { port: self.port, token: token.clone(), timeout: TIMEOUT, instructions: INSTRUCTIONS.into() };
        let ctx2 = ctx.clone();
        match Server::start(config, self.tools.clone(), move || ctx2.request_repaint()) {
            Ok(s) => {
                log::info!("agent access on at {}", s.url());
                self.server = Some(s);
                self.token = token;
            }
            Err(e) => self.error = Some(format!("{e:#}")),
        }
    }

    fn stop(&mut self) {
        if self.server.take().is_some() {
            log::info!("agent access off");
        }
    }

    pub fn on_exit(&mut self) {
        self.stop();
    }

    pub fn save(&self, storage: &mut dyn eframe::Storage) {
        if self.from_env {
            return;
        }
        storage.set_string("agents.enabled", if self.enabled { "1" } else { "0" }.into());
        storage.set_string("agents.changes", if self.allow_changes { "1" } else { "0" }.into());
        storage.set_string("agents.port", self.port.to_string());
        storage.set_string("agents.require_token", if self.require_token { "1" } else { "0" }.into());
        storage.set_string("agents.bring_forward", if self.bring_forward { "1" } else { "0" }.into());
        storage.set_string("agents.claude_setup", self.claude_setup.clone().unwrap_or_default());
    }

    /// The running setup as *Add to Claude Code* registers it.
    fn setup(&self) -> String {
        format!("{}:{}", self.port, self.token.as_deref().map_or("none", |t| &t[..8]))
    }

    /// Seconds since the last call, if any.
    fn last_call(&self) -> Option<f64> {
        let r = self.server.as_ref()?.recent();
        r.first().and_then(|c| c.at.elapsed().ok()).map(|d| d.as_secs_f64())
    }

    /// The Agents button at the right end of the tab bar.
    pub fn indicator(&mut self, ui: &mut egui::Ui) {
        let recent = self.last_call().filter(|s| *s < 2.0);
        let (dot, colour, tip) = match (&self.server, &self.error) {
            (Some(s), _) => ("●", if recent.is_some() { egui::Color32::from_rgb(120, 220, 255) } else { egui::Color32::from_rgb(90, 200, 120) }, format!("Agents can {} this session at {}", if self.allow_changes { "read and change" } else { "read" }, s.url())),
            (None, Some(e)) => ("⚠", egui::Color32::from_rgb(240, 150, 80), format!("Agent access failed: {e}")),
            (None, None) => ("○", ui.visuals().weak_text_color(), "Agent access is off".to_string()),
        };
        if let Some(s) = recent {
            ui.ctx().request_repaint_after(Duration::from_secs_f64(2.0 - s));
        }
        let text = egui::RichText::new(format!("{dot} Agents")).size(13.0).color(colour);
        if ui.selectable_label(self.show_window, text).on_hover_text(tip).clicked() {
            self.show_window = !self.show_window;
        }
    }

    /// The Agents window: turn access on or off, connection details, privacy, recent calls.
    pub fn window(&mut self, ctx: &egui::Context) {
        if !self.show_window {
            return;
        }
        let mut open = true;
        // Under the Agents button, clear of the workspace tabs.
        let corner = ctx.content_rect().right_top() + egui::vec2(-8.0, 34.0);
        if let Some(r) = self.claude_job.as_ref().and_then(|j| j.poll()) {
            self.claude_job = None;
            self.claude_message = Some(match r {
                Ok(msg) => {
                    self.claude_setup = Some(self.setup());
                    (msg, false)
                }
                Err(e) => (format!("{e:#}"), true),
            });
        }
        let mut restart = false;
        egui::Window::new("Agent access").open(&mut open).collapsible(false).resizable(false).default_width(460.0).pivot(egui::Align2::RIGHT_TOP).default_pos(corner).show(ctx, |ui| {
            ui.set_max_width(460.0);
            let mut on = self.enabled;
            if ui.checkbox(&mut on, "Let AI agents read this session").on_hover_text("An MCP server on this computer only, for Claude Code, Claude Desktop, Cursor and other MCP clients").changed() {
                self.enabled = on;
                if on {
                    self.start(ctx, false);
                } else {
                    self.stop();
                    self.error = None;
                }
            }
            ui.add_enabled_ui(self.enabled, |ui| {
                ui.checkbox(&mut self.allow_changes, "Let agents make changes")
                    .on_hover_text("Select, edit, run tools, open and export files, work the panels. Each change is an undo step labelled \"Agent: …\"; the workspace being changed is shown.");
                #[cfg(target_os = "macos")]
                ui.checkbox(&mut self.bring_forward, "Bring Polymix forward when an agent needs it")
                    .on_hover_text("Polymix is not drawn while other windows cover it, so screenshots, panels, tool previews and clicks wait. On: it comes to the front for the agent, while the keyboard stays where you are typing.");
                restart = ui
                    .checkbox(&mut self.require_token, "Require a token")
                    .on_hover_text("On: only programs given the token (in Polymix's data folder, readable by you only) can connect, so not other user accounts on this Mac or sandboxed apps. Off: any program on this computer can. Web pages never can.")
                    .changed()
                    && self.server.is_some();
            });
            ui.add_enabled_ui(self.server.is_none(), |ui| {
                ui.horizontal(|ui| {
                    ui.label("Port");
                    ui.add(egui::DragValue::new(&mut self.port).range(1024..=65535));
                });
            });
            if let Some(e) = &self.error {
                ui.colored_label(egui::Color32::from_rgb(240, 150, 80), format!("⚠ {e}"));
            }
            let mut new_token = false;
            if let Some(s) = &self.server {
                let url = s.url();
                let token = self.token.clone();
                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    let busy = self.claude_job.is_some();
                    let stale = self.claude_setup.as_ref().is_some_and(|c| *c != self.setup());
                    let label = if stale { "Update Claude Code" } else { "Add to Claude Code" };
                    if ui.add_enabled(!busy, egui::Button::new(label)).on_hover_text("Registers Polymix with Claude Code for all your projects (claude mcp add --scope user), replacing an earlier registration").clicked() {
                        let (url, token) = (url.clone(), token.clone());
                        self.claude_message = None;
                        self.claude_job = Some(Job::spawn("Adding to Claude Code", ctx.clone(), move |_| add_to_claude(&url, token.as_deref())));
                    }
                    if busy {
                        polymix_workspace::spinner(ui);
                    } else if stale {
                        ui.colored_label(egui::Color32::from_rgb(240, 150, 80), "Claude Code has an older port or token");
                    } else if self.claude_setup.is_some() && self.claude_message.is_none() {
                        ui.weak("Added for all your projects");
                    }
                });
                let stale = self.claude_setup.as_ref().is_some_and(|c| *c != self.setup());
                match &self.claude_message {
                    Some((msg, true)) => {
                        ui.colored_label(egui::Color32::from_rgb(240, 150, 80), format!("⚠ {msg}"));
                    }
                    // A success that no longer holds is not shown.
                    Some((msg, false)) if !stale => {
                        ui.weak(msg);
                    }
                    _ => {}
                }
                ui.add_space(4.0);
                ui.label("Other MCP clients, or by hand in a terminal:");
                // The token shortened on screen; Copy command has it whole.
                let shown = claude_command(&url, token.as_deref().map(|t| format!("{}…", &t[..8])).as_deref());
                ui.label(egui::RichText::new(shown).monospace().small());
                ui.horizontal(|ui| {
                    if ui.button("Copy command").clicked() {
                        ctx.copy_text(claude_command(&url, token.as_deref()));
                    }
                    if ui.button("Copy URL").clicked() {
                        ctx.copy_text(url.clone());
                    }
                    if let Some(t) = &token {
                        if ui.button("Copy token").on_hover_text("Other clients: send it as Authorization: Bearer <token>").clicked() {
                            ctx.copy_text(t.clone());
                        }
                        new_token = ui.button("New token").on_hover_text("Clients set up with the old token stop working until updated").clicked();
                    }
                });
            }
            if new_token {
                self.start(ctx, true);
            }
            ui.add_space(6.0);
            ui.weak(if self.allow_changes {
                "Agents can change your work: every change is an undo step labelled \"Agent: …\" (Edit → Undo). They cannot change these settings."
            } else {
                "Agents only read: nothing they do changes your work."
            });
            ui.weak("Patient names, IDs, birth dates and accession numbers are masked and dates cut to the year; screenshots and panel reads of the Volume and Segmentation tabs need Hide identifiers on. What an agent reads goes to its provider.");
            if let Some(s) = &self.server {
                let recent = s.recent();
                if !recent.is_empty() {
                    ui.add_space(6.0);
                    ui.label("Recent calls");
                    egui::Grid::new("agents.recent").num_columns(3).spacing([12.0, 2.0]).striped(true).show(ui, |ui| {
                        for c in recent.iter().take(12) {
                            let ago = c.at.elapsed().map(|d| d.as_secs()).unwrap_or(0);
                            ui.weak(if ago < 60 { format!("{ago} s ago") } else { format!("{} min ago", ago / 60) });
                            ui.label(egui::RichText::new(&c.tool).monospace());
                            match &c.error {
                                Some(e) => ui.colored_label(egui::Color32::from_rgb(240, 150, 80), e).on_hover_text(e),
                                None => ui.weak(format!("{} ms", c.duration.as_millis())),
                            };
                            ui.end_row();
                        }
                    });
                }
            }
        });
        if restart {
            self.start(ctx, false);
        }
        if !open {
            self.show_window = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn masks_every_string() {
        let mask: Mask = Arc::new(|s: &str| s.replace("DOE", "•••"));
        let r = reply(AgentReply::Now(Ok(json!({"name": "DOE_vessel", "list": ["a DOE"], "n": 3}))), Some(mask.clone()));
        let Reply::Json(v) = r else { panic!("json expected") };
        assert_eq!(v, json!({"name": "•••_vessel", "list": ["a •••"], "n": 3}));
        let Reply::Later(work) = reply(AgentReply::later(|| Err("no object DOE".into())), Some(mask)) else { panic!("later expected") };
        let Reply::Error(e) = work() else { panic!("error expected") };
        assert_eq!(e, "no object •••");
    }

    #[test]
    fn screenshots_shrink_by_whole_factors() {
        let img = egui::ColorImage::new([3000, 1000], vec![egui::Color32::from_rgb(10, 20, 30); 3_000_000]);
        let Reply::Png(bytes) = png(&img, 1568) else { panic!("png expected") };
        let info = png::Decoder::new(std::io::Cursor::new(bytes)).read_info().unwrap().info().clone();
        assert_eq!((info.width, info.height), (1500, 500));
    }
}

#[cfg(test)]
mod catalog_tests {
    use super::*;

    #[test]
    fn every_tool_has_a_unique_name_and_a_plain_object_schema() {
        let all = all_tools();
        let mut names: Vec<&str> = all.iter().map(|t| &*t.name).collect();
        names.sort();
        let n = names.len();
        names.dedup();
        assert_eq!(names.len(), n, "duplicate tool names");
        for t in &all {
            let s = &t.input_schema;
            assert_eq!(s["type"], "object", "{}", t.name);
            // The Claude API refuses oneOf / anyOf / allOf at the top of an input schema.
            assert!(["oneOf", "anyOf", "allOf"].iter().all(|k| s.get(k).is_none()), "{}", t.name);
            for r in s["required"].as_array().into_iter().flatten() {
                assert!(s["properties"].get(r.as_str().unwrap()).is_some(), "{}: required {r} is not a property", t.name);
            }
            // Reading tools are named for what they read; changing ones say what they change.
            if t.kind != ToolKind::Read {
                assert!(!t.description.is_empty());
            }
        }
        // Each image workspace offers its actions as one tool.
        for (name, kind) in [("volume_do", ToolKind::Destructive), ("segmentation_do", ToolKind::Destructive)] {
            let t = all.iter().find(|t| t.name == name).unwrap_or_else(|| panic!("{name} listed"));
            assert_eq!(t.kind, kind);
            assert!(t.description.contains("\n- "), "{name} lists its actions");
        }
        assert!(all.iter().all(|t| t.name != "volume_browse"), "volume_do replaced volume_browse");
    }

    #[test]
    fn workspace_do_tools_check_then_act() {
        let ctx = egui::Context::default();
        let mut seg = polymix_segmentation::SegmentationWorkspace::new(&ctx, None);
        let answer = |r: AgentReply| match r {
            AgentReply::Now(r) => r,
            AgentReply::Later(f) => f(),
        };
        let e = answer(workspace_call(&mut seg, "segmentation_do", &json!({"action": "paint", "args": {}}), &ctx)).unwrap_err();
        assert!(e.contains("needs `points`"), "{e}");
        let e = answer(workspace_call(&mut seg, "segmentation_do", &json!({"action": "explode"}), &ctx)).unwrap_err();
        assert!(e.contains("no action explode") && e.contains("threshold"), "{e}");
        // Checked, then the workspace's own answer.
        let e = answer(workspace_call(&mut seg, "segmentation_do", &json!({"action": "add_segment"}), &ctx)).unwrap_err();
        assert!(e.contains("No volume is loaded"), "{e}");
        assert!(answer(workspace_call(&mut seg, "segmentation_state", &json!({}), &ctx)).is_ok());
    }

    #[test]
    fn claude_commands() {
        let url = "http://127.0.0.1:47390/mcp";
        assert_eq!(claude_command(url, None), "claude mcp add --transport http --scope user polymix http://127.0.0.1:47390/mcp");
        assert_eq!(claude_command(url, Some("abc")), "claude mcp add --transport http --scope user polymix http://127.0.0.1:47390/mcp --header \"Authorization: Bearer abc\"");
        // The header goes last: --header takes every value after it.
        assert_eq!(claude_args(url, Some("abc")).last().unwrap(), "Authorization: Bearer abc");
    }

    #[test]
    fn changes_need_permission() {
        let tools = vec![AgentTool::new("mesh_scene", "", json!({})), AgentTool::new("mesh_edit", "", json!({})).kind(ToolKind::Destructive)];
        let mut a = AgentAccess::new(None, tools);
        assert!(a.refusal("mesh_scene").is_none());
        assert!(a.refusal("mesh_edit").unwrap().contains("Let agents make changes"));
        assert!(a.refusal("unknown_tool").is_some());
        a.allow_changes = true;
        assert!(a.refusal("mesh_edit").is_none());
    }
}
