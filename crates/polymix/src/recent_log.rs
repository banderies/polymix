//! The recent log, kept in memory for agents (`app_log`) beside the usual output on stderr.
//!
//! Polymix's own messages are kept from `info` up, everything else (wgpu, winit, …) from
//! `warn` up, whatever `RUST_LOG` lets through to stderr. File paths are replaced with
//! `<path>`, as in crash logs: file names can carry patient names.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::SystemTime;

const KEEP: usize = 2000;

#[derive(Clone, Debug)]
pub struct Entry {
    pub at: SystemTime,
    pub level: log::Level,
    pub target: String,
    pub message: String,
}

static RECENT: Mutex<VecDeque<Entry>> = Mutex::new(VecDeque::new());

struct Tee {
    stderr: env_logger::Logger,
}

fn kept(meta: &log::Metadata) -> bool {
    let ours = ["polymix", "mm_"].iter().any(|p| meta.target().starts_with(p));
    meta.level() <= if ours { log::Level::Info } else { log::Level::Warn }
}

impl log::Log for Tee {
    fn enabled(&self, meta: &log::Metadata) -> bool {
        self.stderr.enabled(meta) || kept(meta)
    }

    fn log(&self, record: &log::Record) {
        if self.stderr.matches(record) {
            self.stderr.log(record);
        }
        if kept(record.metadata()) {
            push(Entry { at: SystemTime::now(), level: record.level(), target: record.target().to_string(), message: polymix_workspace::guard::scrub_paths(&record.args().to_string()) });
        }
    }

    fn flush(&self) {
        self.stderr.flush();
    }
}

fn push(e: Entry) {
    let mut q = RECENT.lock().unwrap_or_else(|e| e.into_inner());
    if q.len() == KEEP {
        q.pop_front();
    }
    q.push_back(e);
}

/// Log to stderr as `env_logger` does (`RUST_LOG`, else `default_filter`) and keep the recent
/// log in memory.
pub fn install(default_filter: &str) {
    let stderr = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(default_filter)).build();
    let max = stderr.filter().max(log::LevelFilter::Info);
    if log::set_boxed_logger(Box::new(Tee { stderr })).is_ok() {
        log::set_max_level(max);
    }
}

/// The last `n` entries at `min` or more severe, oldest first.
pub fn recent(n: usize, min: log::Level) -> Vec<Entry> {
    let q = RECENT.lock().unwrap_or_else(|e| e.into_inner());
    let mut out: Vec<Entry> = q.iter().rev().filter(|e| e.level <= min).take(n).cloned().collect();
    out.reverse();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_ours_and_warnings_without_paths() {
        let meta = |target: &'static str, level| log::MetadataBuilder::new().target(target).level(level).build();
        assert!(kept(&meta("polymix::app", log::Level::Info)));
        assert!(kept(&meta("mm_volume::scan", log::Level::Info)));
        assert!(!kept(&meta("polymix::app", log::Level::Debug)));
        assert!(!kept(&meta("wgpu_core::device", log::Level::Info)));
        assert!(kept(&meta("wgpu_core::device", log::Level::Warn)));
        push(Entry { at: SystemTime::now(), level: log::Level::Warn, target: "test".into(), message: polymix_workspace::guard::scrub_paths("cannot read /scans/DOE.dcm") });
        push(Entry { at: SystemTime::now(), level: log::Level::Info, target: "test".into(), message: "opened".into() });
        let warnings = recent(10, log::Level::Warn);
        assert!(warnings.iter().any(|e| e.message == "cannot read <path>"));
        assert!(warnings.iter().all(|e| e.level <= log::Level::Warn));
        assert_eq!(recent(1, log::Level::Info).last().unwrap().message, "opened");
    }
}
