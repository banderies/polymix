//! A folder scan running in the background, with live progress and cancellation.

use mm_volume::model::Catalog;
use mm_volume::scan::{scan, FileCache, Progress, ScanOptions};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, TryRecvError};
use std::sync::{Arc, Mutex};

pub struct ScanTask {
    progress: Arc<Mutex<Option<Progress>>>,
    cancel: Arc<AtomicBool>,
    rx: Receiver<anyhow::Result<Catalog>>,
}

impl ScanTask {
    /// Scan `roots` on a background thread, reading and updating the cache at `cache_path`.
    pub fn start(roots: Vec<PathBuf>, cache_path: Option<PathBuf>, ctx: egui::Context) -> Self {
        let progress = Arc::new(Mutex::new(None));
        let cancel = Arc::new(AtomicBool::new(false));
        let (tx, rx) = channel();
        let (p2, c2) = (progress.clone(), cancel.clone());
        std::thread::Builder::new()
            .name("polymix-job-volume-scan".into())
            .spawn(move || {
                let last_repaint = Mutex::new(std::time::Instant::now());
                let report = |p: Progress| {
                    *p2.lock().unwrap_or_else(|e| e.into_inner()) = Some(p);
                    // Repaint at most ten times a second.
                    let mut t = last_repaint.lock().unwrap_or_else(|e| e.into_inner());
                    if t.elapsed().as_millis() > 100 {
                        *t = std::time::Instant::now();
                        ctx.request_repaint();
                    }
                    !c2.load(Ordering::Relaxed)
                };
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let mut cache = cache_path.as_deref().map(FileCache::load).unwrap_or_default();
                    let catalog = scan(&roots, &ScanOptions::default(), Some(&mut cache), &report)?;
                    if let Some(p) = &cache_path {
                        if let Err(e) = cache.save(p) {
                            log::warn!("could not save the scan cache: {e:#}");
                        }
                    }
                    Ok(catalog)
                }))
                .unwrap_or_else(|_| Err(anyhow::anyhow!("internal error while scanning")));
                let _ = tx.send(result);
                ctx.request_repaint();
            })
            .expect("start the scan thread");
        Self { progress, cancel, rx }
    }

    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    pub fn progress(&self) -> Option<Progress> {
        *self.progress.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// `Some(result)` once finished.
    pub fn poll(&self) -> Option<anyhow::Result<Catalog>> {
        match self.rx.try_recv() {
            Ok(r) => Some(r),
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => Some(Err(anyhow::anyhow!("the scan stopped unexpectedly"))),
        }
    }
}

impl Drop for ScanTask {
    fn drop(&mut self) {
        self.cancel();
    }
}
