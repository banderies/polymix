//! Crash guard: a panic inside a workspace stops that workspace, not the app.
//!
//! The shell calls each workspace through [`Slot::call`]. A panic unwinds out of the
//! workspace's code, is caught there, and turns the slot into [`Slot::Crashed`]: the tab
//! shows a [`crash_card`] with the error and a Restart button while the other tabs carry on.
//! A panic hook writes every panic (with a backtrace) to `<data dir>/logs/`.
//!
//! What cannot be caught: aborts (stack overflow, out of memory, a panic while panicking).
//! Heavy work therefore runs in [`crate::jobs::Job`]s, which catch their own panics.

use crate::Workspace;
use std::cell::Cell;
use std::panic::{self, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::{Mutex, Once};

/// What stopped a workspace.
#[derive(Clone, Debug)]
pub struct CrashReport {
    pub workspace: &'static str,
    pub message: String,
    /// Source location of the panic ("file.rs:123"), when known.
    pub location: String,
    /// The log file with the backtrace, when it could be written.
    pub log: Option<PathBuf>,
}

impl CrashReport {
    /// Plain-text summary for the clipboard.
    pub fn details(&self) -> String {
        let mut s = format!("Polymix {} — the {} workspace stopped\n{}\n", env!("CARGO_PKG_VERSION"), self.workspace, self.message);
        if !self.location.is_empty() {
            s += &format!("at {}\n", self.location);
        }
        if let Some(l) = &self.log {
            s += &format!("log: {}\n", l.display());
        }
        s
    }
}

thread_local! {
    /// The workspace whose code runs on this thread right now (set by [`run_guarded`]).
    static CURRENT: Cell<Option<&'static str>> = const { Cell::new(None) };
}

/// The report of the last panic inside a guarded call, for [`run_guarded`] to pick up.
static LAST: Mutex<Option<CrashReport>> = Mutex::new(None);
static HOOK: Once = Once::new();

/// Panic locations already logged this session (one log per location).
static LOGGED: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Install the panic hook (once). It records panics in guarded workspace code for their crash
/// card and logs them (and panics in background jobs) with a backtrace, once per location;
/// panics elsewhere (e.g. a file reader failing on a corrupt file, caught by its caller) are
/// only passed on to the default hook. The hook must never call into `egui::Context`: the
/// panicking code may hold its lock.
pub fn install_panic_hook() {
    HOOK.call_once(|| {
        let default = panic::take_hook();
        panic::set_hook(Box::new(move |info| {
            let workspace = CURRENT.with(|c| c.get());
            let thread = std::thread::current().name().unwrap_or("unnamed").to_string();
            let message = payload_message(info.payload());
            let location = info.location().map(|l| format!("{}:{}", l.file(), l.line())).unwrap_or_default();
            let worth_logging = workspace.is_some() || thread.starts_with("polymix-job");
            let first_time = worth_logging && {
                let mut seen = LOGGED.lock().unwrap_or_else(|e| e.into_inner());
                let new = !seen.contains(&location);
                if new {
                    seen.push(location.clone());
                }
                new
            };
            let log = if first_time {
                let backtrace = std::backtrace::Backtrace::force_capture();
                write_log(workspace.unwrap_or("background"), &thread, &scrub_paths(&message), &location, &backtrace)
            } else {
                None
            };
            if let Some(ws) = workspace {
                *LAST.lock().unwrap_or_else(|e| e.into_inner()) = Some(CrashReport { workspace: ws, message, location, log });
            }
            default(info);
        }));
    });
}

/// Replace file paths in a message with `<path>`: file names can carry patient names, and logs
/// get shared.
pub fn scrub_paths(message: &str) -> String {
    let chars: Vec<char> = message.chars().collect();
    let mut out = String::with_capacity(message.len());
    let mut i = 0;
    while i < chars.len() {
        let at_word_start = i == 0 || matches!(chars[i - 1], ' ' | '\t' | '\n' | '"' | '\'' | '(' | '`');
        let unix = chars[i] == '/' || (chars[i] == '~' && chars.get(i + 1) == Some(&'/'));
        let windows = chars[i].is_ascii_alphabetic() && chars.get(i + 1) == Some(&':') && chars.get(i + 2) == Some(&'\\');
        if at_word_start && (unix || windows) {
            let end = (i..chars.len()).find(|&j| matches!(chars[j], ' ' | '\t' | '\n' | '"' | '\'' | ')' | '`')).unwrap_or(chars.len());
            out.push_str("<path>");
            i = end;
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

fn payload_message(p: &(dyn std::any::Any + Send)) -> String {
    p.downcast_ref::<String>().cloned().or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_else(|| "unknown panic".into())
}

/// Keep this many crash logs.
const MAX_LOGS: usize = 20;

fn write_log(workspace: &str, thread: &str, message: &str, location: &str, backtrace: &std::backtrace::Backtrace) -> Option<PathBuf> {
    let dir = crate::data_dir()?.join("logs");
    std::fs::create_dir_all(&dir).ok()?;
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let path = dir.join(format!("crash-{secs}-{}.log", workspace.to_lowercase()));
    let text = format!("Polymix {}\nworkspace: {workspace}\nthread: {thread}\nmessage: {message}\nat: {location}\n\n{backtrace}\n", env!("CARGO_PKG_VERSION"));
    std::fs::write(&path, text).ok()?;
    // Prune the oldest logs.
    if let Ok(rd) = std::fs::read_dir(&dir) {
        let mut logs: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("crash-"))).collect();
        logs.sort();
        let excess = logs.len().saturating_sub(MAX_LOGS);
        for p in &logs[..excess] {
            let _ = std::fs::remove_file(p);
        }
    }
    Some(path)
}

/// Run `f` as workspace `workspace`'s code: a panic inside it comes back as a report.
pub fn run_guarded<R>(workspace: &'static str, f: impl FnOnce() -> R) -> Result<R, CrashReport> {
    let prev = CURRENT.with(|c| c.replace(Some(workspace)));
    let result = panic::catch_unwind(AssertUnwindSafe(f));
    CURRENT.with(|c| c.set(prev));
    result.map_err(|payload| {
        let recorded = LAST.lock().unwrap_or_else(|e| e.into_inner()).take();
        // A panic on another thread (rayon) and re-raised here was not recorded as ours.
        recorded.filter(|r| r.workspace == workspace).unwrap_or_else(|| CrashReport { workspace, message: payload_message(&*payload), location: String::new(), log: None })
    })
}

/// A workspace, running or stopped by a crash.
pub enum Slot<W> {
    Live(W),
    Crashed(CrashReport),
}

impl<W: Workspace> Slot<W> {
    pub fn live(&self) -> Option<&W> {
        match self {
            Slot::Live(w) => Some(w),
            Slot::Crashed(_) => None,
        }
    }

    /// Run `f` on the live workspace under the crash guard. A panic stops the workspace:
    /// the slot becomes [`Slot::Crashed`] and the result is `None`.
    pub fn call<R>(&mut self, f: impl FnOnce(&mut W) -> R) -> Option<R> {
        let Slot::Live(w) = self else { return None };
        let name = w.name();
        match run_guarded(name, || f(w)) {
            Ok(r) => Some(r),
            Err(report) => {
                log::error!("the {name} workspace stopped: {}", report.message);
                let old = std::mem::replace(self, Slot::Crashed(report));
                // Its state may be half-updated: saving what it can and dropping it must not
                // take the app down.
                if let Slot::Live(mut w) = old {
                    let _ = run_guarded(name, || w.emergency_save());
                    let _ = panic::catch_unwind(AssertUnwindSafe(|| drop(w)));
                }
                None
            }
        }
    }
}

/// What the user chose on a crash card.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CardAction {
    None,
    Restart,
}

/// The card shown in place of a crashed workspace. `restart_note` explains what Restart
/// brings back (e.g. the Mesh workspace's autosave).
pub fn crash_card(ui: &mut egui::Ui, report: &CrashReport, restart_note: &str) -> CardAction {
    let mut action = CardAction::None;
    egui::CentralPanel::default().show(ui, |ui| {
        ui.vertical_centered(|ui| {
            ui.add_space(ui.available_height() * 0.2);
            ui.heading(format!("The {} workspace stopped", report.workspace));
            ui.add_space(6.0);
            ui.label("An internal error stopped this workspace. The other workspaces are unaffected: switch tabs to keep working.");
            ui.add_space(10.0);
            egui::Frame::group(ui.style()).show(ui, |ui| {
                ui.set_max_width(640.0);
                ui.label(egui::RichText::new(&report.message).monospace());
                if !report.location.is_empty() {
                    ui.weak(format!("at {}", report.location));
                }
            });
            ui.add_space(10.0);
            ui.horizontal(|ui| {
                // Centre the row of buttons.
                let w = 340.0;
                ui.add_space(((ui.available_width() - w) * 0.5).max(0.0));
                if ui.button("Restart workspace").on_hover_text(restart_note).clicked() {
                    action = CardAction::Restart;
                }
                if ui.button("Copy details").clicked() {
                    ui.ctx().copy_text(report.details());
                }
                if let Some(log) = &report.log {
                    if ui.button("Show log").on_hover_text(log.display().to_string()).clicked() {
                        reveal(log);
                    }
                }
            });
            if !restart_note.is_empty() {
                ui.add_space(4.0);
                ui.weak(restart_note);
            }
        });
    });
    action
}

/// Show `path` in the file manager.
pub fn reveal(path: &std::path::Path) {
    let r = if cfg!(target_os = "macos") {
        std::process::Command::new("open").arg("-R").arg(path).spawn()
    } else if cfg!(windows) {
        std::process::Command::new("explorer").arg(format!("/select,{}", path.display())).spawn()
    } else {
        std::process::Command::new("xdg-open").arg(path.parent().unwrap_or(path)).spawn()
    };
    if let Err(e) = r {
        log::warn!("could not open the file manager: {e}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fragile {
        calls: u32,
    }

    /// Crash logs from these tests go to a scratch folder, never the user's.
    fn scratch_data_dir() -> PathBuf {
        static SET: Once = Once::new();
        let dir = std::env::temp_dir().join(format!("polymix-guard-test-{}", std::process::id()));
        SET.call_once(|| std::env::set_var("POLYMIX_DATA_DIR", &dir));
        dir
    }

    impl Workspace for Fragile {
        fn name(&self) -> &'static str {
            "Fragile"
        }
        fn ui(&mut self, _ui: &mut egui::Ui, _frame: &mut eframe::Frame) {}
    }

    #[test]
    fn a_panic_stops_only_that_workspace() {
        scratch_data_dir();
        install_panic_hook();
        let mut a = Slot::Live(Fragile { calls: 0 });
        let mut b = Slot::Live(Fragile { calls: 0 });
        assert_eq!(a.call(|w| {
            w.calls += 1;
            w.calls
        }), Some(1));
        assert_eq!(a.call(|_| -> u32 { panic!("boom in a") }), None);
        let Slot::Crashed(report) = &a else { panic!("a should have stopped") };
        assert_eq!(report.workspace, "Fragile");
        assert!(report.message.contains("boom in a"));
        assert!(report.location.contains("guard.rs"));
        assert!(report.log.as_ref().is_some_and(|l| std::fs::read_to_string(l).unwrap().contains("boom in a")));
        // A stopped workspace is skipped; the other one carries on.
        assert_eq!(a.call(|w| w.calls), None);
        assert_eq!(b.call(|w| {
            w.calls += 1;
            w.calls
        }), Some(1));
    }

    #[test]
    fn logs_never_carry_paths() {
        assert_eq!(scrub_paths("cannot read \"/Users/x/MRI/DOE_JOHN.IMA\": bad"), "cannot read \"<path>\": bad");
        assert_eq!(scrub_paths("index 3 out of range"), "index 3 out of range");
    }

    #[test]
    fn panics_on_rayon_like_threads_are_still_caught() {
        scratch_data_dir();
        install_panic_hook();
        let r = run_guarded("Threads", || std::thread::scope(|s| s.spawn(|| panic!("worker failed")).join().map_err(|e| std::panic::resume_unwind(e))));
        let Err(report) = r else { panic!("the panic should have been caught") };
        assert_eq!(report.workspace, "Threads");
        assert!(report.message.contains("worker failed"));
    }
}
