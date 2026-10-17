//! Background computations with progress reporting and cancellation.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::mpsc::{channel, Receiver, TryRecvError};
use std::sync::Arc;

/// Progress callback handed to geometry operations: report a 0..1 fraction, returns
/// `false` when the user cancelled.
pub type ProgressFn<'a> = &'a dyn Fn(f32) -> bool;

pub struct Job<T> {
    pub label: String,
    progress: Arc<AtomicU32>,
    cancel: Arc<AtomicBool>,
    rx: Receiver<anyhow::Result<T>>,
}

impl<T: Send + 'static> Job<T> {
    pub fn spawn(label: impl Into<String>, ctx: egui::Context, f: impl FnOnce(ProgressFn) -> anyhow::Result<T> + Send + 'static) -> Self {
        let progress = Arc::new(AtomicU32::new(0f32.to_bits()));
        let cancel = Arc::new(AtomicBool::new(false));
        let (tx, rx) = channel();
        let (p2, c2) = (progress.clone(), cancel.clone());
        std::thread::Builder::new()
            .name("polymix-job".into())
            .spawn(move || {
                let cb = |f: f32| {
                    p2.store(f.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
                    !c2.load(Ordering::Relaxed)
                };
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(&cb)))
                    .unwrap_or_else(|p| {
                        let msg = p
                            .downcast_ref::<String>()
                            .cloned()
                            .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                            .unwrap_or_else(|| "operation panicked".into());
                        Err(anyhow::anyhow!("internal error: {msg}"))
                    });
                let _ = tx.send(r);
                ctx.request_repaint();
            })
            .expect("spawn job thread");
        Self { label: label.into(), progress, cancel, rx }
    }

    pub fn progress(&self) -> f32 {
        f32::from_bits(self.progress.load(Ordering::Relaxed))
    }

    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    /// `Some(result)` once finished.
    pub fn poll(&self) -> Option<anyhow::Result<T>> {
        match self.rx.try_recv() {
            Ok(r) => Some(r),
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => Some(Err(anyhow::anyhow!("job thread exited"))),
        }
    }
}

impl<T> Drop for Job<T> {
    /// Abandoned jobs (e.g. a tool closed mid-computation) are asked to stop early.
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}
