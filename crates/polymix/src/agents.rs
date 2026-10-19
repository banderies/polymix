//! Agent access, the shell's side: turning the MCP server ([`polymix_mcp`]) on and off, its
//! token, the Agents window, the app-wide tools (`app_status`, `app_screenshot`, `app_log`)
//! and the privacy pass over everything agents are sent.
//!
//! Off by default. On, it listens on 127.0.0.1 and needs a token kept in the data folder
//! (`agent-token`), so only programs of this user on this computer can connect. Tools only
//! read. `POLYMIX_MCP=1` turns it on for one session (scripted runs, development);
//! `POLYMIX_MCP_PORT` picks the port.

use polymix_mcp::{Reply, Server, ToolSpec};
use polymix_workspace::agent::{AgentReply, AgentTool};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

/// How long an agent waits for the UI thread before it is told the app is busy.
const TIMEOUT: Duration = Duration::from_secs(30);
/// Masks patients' identifiers in text (see `VolumeWorkspace::agent_mask`).
pub type Mask = Arc<dyn Fn(&str) -> String + Send + Sync>;

const INSTRUCTIONS: &str = "Polymix is a desktop app for medical images and meshes: the Volume workspace finds and previews images (DICOM, NIfTI, …), Segmentation segments one in three planes, and Mesh edits surface meshes (vessel cleanup for CFD). These tools read the session the user has open; none changes anything. Start with app_status; app_screenshot shows the window as the user sees it. Lengths are in millimetres. Patient identifiers are masked and dates cut to the year: do not try to recover them.";

pub fn shell_tools() -> Vec<AgentTool> {
    vec![
        AgentTool::new("app_status", "Polymix as a whole: version, which workspace tab is shown (Volume, Segmentation or Mesh), whether each workspace is running, busy or stopped by an internal error, and the window size.", json!({})),
        AgentTool::new(
            "app_screenshot",
            "A PNG of the Polymix window as the user sees it now (the shown workspace tab only). Refused for the Volume and Segmentation tabs unless Hide identifiers is on in the Volume tab.",
            json!({"max_size": {"type": "integer", "description": "Longest side in pixels (default 1568, 256 to 4096)"}}),
        ),
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
    ToolSpec { name: t.name.into(), description: t.description.into(), input_schema: t.input_schema, read_only: true }
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
    port: u16,
    /// Turned on by `POLYMIX_MCP` for this session only: never saved.
    from_env: bool,
    tools: Vec<ToolSpec>,
    server: Option<Server>,
    token: Option<String>,
    error: Option<String>,
    pub show_window: bool,
}

impl AgentAccess {
    pub fn new(storage: Option<&dyn eframe::Storage>, tools: Vec<ToolSpec>) -> Self {
        let env_on = std::env::var("POLYMIX_MCP").is_ok_and(|v| v == "1");
        let env_port = std::env::var("POLYMIX_MCP_PORT").ok().and_then(|p| p.parse().ok());
        let stored = |k: &str| storage.filter(|_| !polymix_workspace::scripted()).and_then(|s| s.get_string(k));
        Self {
            enabled: env_on || stored("agents.enabled").is_some_and(|v| v == "1"),
            port: env_port.or_else(|| stored("agents.port").and_then(|p| p.parse().ok())).unwrap_or(polymix_mcp::DEFAULT_PORT),
            from_env: env_on || env_port.is_some(),
            tools,
            server: None,
            token: None,
            error: None,
            show_window: false,
        }
    }

    pub fn server(&self) -> Option<&Server> {
        self.server.as_ref()
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
        let token = match load_token(fresh_token) {
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
                self.token = Some(token);
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
        storage.set_string("agents.port", self.port.to_string());
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
            (Some(s), _) => ("●", if recent.is_some() { egui::Color32::from_rgb(120, 220, 255) } else { egui::Color32::from_rgb(90, 200, 120) }, format!("Agents can read this session at {}", s.url())),
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
            if let (Some(s), Some(token)) = (&self.server, self.token.clone()) {
                let url = s.url();
                let command = format!("claude mcp add --transport http polymix {url} --header \"Authorization: Bearer {token}\"");
                ui.add_space(6.0);
                ui.label("Connect Claude Code (run once in a terminal):");
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(format!("claude mcp add --transport http polymix {url} --header \"Authorization: Bearer {}…\"", &token[..8])).monospace().small());
                });
                ui.horizontal(|ui| {
                    if ui.button("Copy command").clicked() {
                        ctx.copy_text(command);
                    }
                    if ui.button("Copy URL").clicked() {
                        ctx.copy_text(url.clone());
                    }
                    if ui.button("Copy token").on_hover_text("Other clients: send it as Authorization: Bearer <token>").clicked() {
                        ctx.copy_text(token.clone());
                    }
                    new_token = ui.button("New token").on_hover_text("Clients set up with the old token stop working").clicked();
                });
            }
            if new_token {
                self.start(ctx, true);
            }
            ui.add_space(6.0);
            ui.weak("Agents only read: nothing they do changes your work. Patient names, IDs, birth dates and accession numbers are masked and dates cut to the year; screenshots of the Volume and Segmentation tabs need Hide identifiers on. What an agent reads goes to its provider.");
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
