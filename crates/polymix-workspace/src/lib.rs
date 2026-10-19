//! Shared ground for Polymix's workspaces (Volume, Segmentation, Mesh).
//!
//! Each workspace lives in its own crate and implements [`Workspace`]; the shell in the
//! `polymix` binary draws the tab bar and calls the workspaces through [`guard::run_guarded`],
//! so a panic in one workspace stops only that workspace. This crate holds what they all
//! need: the trait, the crash guard, background [`jobs`], what workspaces show [`agent`]s and
//! the per-user [`data_dir`].

pub mod agent;
pub mod guard;
pub mod jobs;
pub mod style;

use std::path::{Path, PathBuf};

/// One top-level workspace.
pub trait Workspace {
    /// Name on the tab and in crash reports ("Volume", "Mesh").
    fn name(&self) -> &'static str;
    /// Draw the workspace into the space below the tab bar (only while its tab is shown).
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame);
    /// Background work, every frame, shown or not (polling jobs, autosave).
    fn tick(&mut self, _ctx: &egui::Context, _active: bool) {}
    fn save(&mut self, _storage: &mut dyn eframe::Storage) {}
    fn on_exit(&mut self) {}
    /// Window title while this workspace is shown.
    fn title(&self) -> String {
        format!("Polymix — {}", self.name())
    }
    /// Whether this workspace opens `path` (dropped on the window or given on the command
    /// line).
    fn accepts(&self, _path: &Path) -> bool {
        false
    }
    fn open(&mut self, _paths: Vec<PathBuf>, _ctx: &egui::Context) {}
    /// Queue one step of a test script (`POLYMIX_SCRIPT`) that the shell does not handle
    /// itself.
    fn script_push(&mut self, _step: String) {}
    /// No queued script steps and nothing running that a script should wait for.
    fn script_idle(&self) -> bool {
        true
    }
    /// Something the user must answer here (e.g. a recovery prompt): the shell shows this
    /// workspace at startup.
    fn needs_attention(&self) -> bool {
        false
    }
    /// Called (under the crash guard) on a workspace that just crashed, before it is
    /// dropped: save what can be saved for the restarted workspace to recover.
    fn emergency_save(&mut self) {}
    /// Tools this workspace offers agents (see [`agent`]), the same every time (they are
    /// listed even while the workspace is stopped).
    fn agent_tools() -> Vec<agent::AgentTool>
    where
        Self: Sized,
    {
        Vec::new()
    }
    /// What agents can ask this workspace to do: the shell offers them as the agent tool
    /// `<workspace>_do` ([`agent::do_tool`]), its description listing their signatures.
    fn agent_actions() -> Vec<agent::Action>
    where
        Self: Sized,
    {
        Vec::new()
    }
    /// What `<workspace>_do` is for, and what holds for all its actions (the start of its
    /// description).
    fn agent_actions_about() -> &'static str
    where
        Self: Sized,
    {
        ""
    }
    /// Do one of [`Workspace::agent_actions`] (its arguments already checked).
    fn agent_act(&mut self, action: &str, _args: &serde_json::Value, _ctx: &egui::Context) -> agent::AgentReply {
        agent::AgentReply::error(format!("the {} workspace has no action {action}", self.name()))
    }
    /// An agent is about to work this workspace's panels (a click or value through the
    /// accessibility tree, applied next frame): label the changes that follow as the agent's.
    fn agent_acting(&mut self) {}
    /// Answer an agent's call of one of [`Workspace::agent_tools`] (on the UI thread, between
    /// frames).
    fn agent_call(&mut self, tool: &str, _args: &serde_json::Value, _ctx: &egui::Context) -> agent::AgentReply {
        agent::AgentReply::error(format!("the {} workspace has no tool {tool}", self.name()))
    }
}

/// Where Polymix keeps per-user files (autosave, parts, logs, catalogs).
pub fn data_dir() -> Option<PathBuf> {
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

/// Whether this is a scripted test run (`POLYMIX_SCRIPT`): such runs never load or save the
/// user's preferences.
pub fn scripted() -> bool {
    std::env::var_os("POLYMIX_SCRIPT").is_some()
}
