//! Open, save and folder dialogs that leave the frame running.
//!
//! A blocking `rfd::FileDialog` runs the panel's modal loop inside winit's event handler, i.e.
//! inside the frame that asked for it. Whatever spins the run loop in its default mode meanwhile
//! runs the events winit queued for later; AppKit does exactly that while a file or folder is
//! dragged from the Finder onto the panel. winit then finds its handler still busy with this
//! frame and panics inside an Objective-C block, where a panic cannot unwind: the app aborts.
//!
//! These start the panel as a sheet on the window and return at once, so the frame ends and
//! winit's handler is free while the panel is up. The answer comes back from [`take`] under the
//! same key in a later frame (the panel closing asks for one); call it from code that runs every
//! frame, not from inside a menu that has closed.
//!
//! ```ignore
//! if ui.button("Add folder…").clicked() {
//!     file_dialog::pick_folder(ui.ctx(), "volume.add_folder", file_dialog::FileDialog::new());
//! }
//! // …every frame:
//! if let Some(dir) = file_dialog::take(ctx, "volume.add_folder") { … }
//! ```

use std::future::Future;
use std::fmt::Debug;
use std::hash::Hash;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Wake, Waker};

pub use rfd::AsyncFileDialog as FileDialog;

/// The chosen paths; None after Cancel.
type Answer = Option<Vec<PathBuf>>;

type Panel = Pin<Box<dyn Future<Output = Answer> + Send>>;

/// A panel, kept in egui's memory under its key from when it is asked for until its answer is
/// taken.
#[derive(Clone)]
struct Open(Arc<Mutex<State>>);

enum State {
    Up(Panel),
    /// Closed, with this answer (not taken yet).
    Closed(Answer),
}

/// Wakes the UI when the panel closes, so [`take`] is called again.
struct Repaint(egui::Context);

impl Wake for Repaint {
    fn wake(self: Arc<Self>) {
        self.0.request_repaint();
    }
}

fn id(key: impl Hash + Debug) -> egui::Id {
    egui::Id::unique(("polymix.file_dialog", key))
}

fn start(ctx: &egui::Context, key: impl Hash + Debug, open: impl FnOnce() -> Panel) {
    let id = id(key);
    if ctx.data(|d| d.get_temp::<Open>(id)).is_some() {
        return;
    }
    let panel = Open(Arc::new(Mutex::new(State::Up(open()))));
    ctx.data_mut(|d| d.insert_temp(id, panel.clone()));
    // Polled once now: this shows the panel, and its closing then wakes the UI even when nothing
    // else does. (Where a sheet is not possible rfd runs a blocking panel instead, which has
    // closed by the time this returns; its answer waits for `take` all the same.)
    panel.poll(ctx);
}

impl Open {
    /// Whether the panel has closed.
    fn poll(&self, ctx: &egui::Context) -> bool {
        let mut state = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if let State::Up(panel) = &mut *state {
            let waker = Waker::from(Arc::new(Repaint(ctx.clone())));
            match panel.as_mut().poll(&mut Context::from_waker(&waker)) {
                Poll::Ready(answer) => *state = State::Closed(answer),
                Poll::Pending => return false,
            }
        }
        true
    }
}

/// Ask for one file to open.
pub fn pick_file(ctx: &egui::Context, key: impl Hash + Debug, dialog: FileDialog) {
    start(ctx, key, || Box::pin(async move { dialog.pick_file().await.map(|f| vec![f.into()]) }));
}

/// Ask for one or more files to open.
pub fn pick_files(ctx: &egui::Context, key: impl Hash + Debug, dialog: FileDialog) {
    start(ctx, key, || Box::pin(async move { dialog.pick_files().await.map(|fs| fs.into_iter().map(Into::into).collect()) }));
}

/// Ask for a folder.
pub fn pick_folder(ctx: &egui::Context, key: impl Hash + Debug, dialog: FileDialog) {
    start(ctx, key, || Box::pin(async move { dialog.pick_folder().await.map(|f| vec![f.into()]) }));
}

/// Ask where to save.
pub fn save_file(ctx: &egui::Context, key: impl Hash + Debug, dialog: FileDialog) {
    start(ctx, key, || Box::pin(async move { dialog.save_file().await.map(|f| vec![f.into()]) }));
}

/// The path chosen in the panel started under `key`, once it has closed; None while it is up,
/// after Cancel, or when none was started.
pub fn take(ctx: &egui::Context, key: impl Hash + Debug) -> Option<PathBuf> {
    take_all(ctx, key).and_then(|paths| paths.into_iter().next())
}

/// Every path chosen in the panel started under `key` ([`pick_files`]); as [`take`].
pub fn take_all(ctx: &egui::Context, key: impl Hash + Debug) -> Option<Vec<PathBuf>> {
    let id = id(key);
    let panel = ctx.data(|d| d.get_temp::<Open>(id))?;
    if !panel.poll(ctx) {
        return None;
    }
    ctx.data_mut(|d| d.remove::<Open>(id));
    let answer = std::mem::replace(&mut *panel.0.lock().unwrap_or_else(|e| e.into_inner()), State::Closed(None));
    match answer {
        State::Closed(answer) => answer,
        State::Up(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A panel that closes when the test says so.
    #[derive(Clone, Default)]
    struct TestPanel(Arc<Mutex<(Option<Answer>, Option<Waker>)>>);

    impl TestPanel {
        fn close(&self, answer: Answer) {
            let mut s = self.0.lock().unwrap();
            s.0 = Some(answer);
            if let Some(w) = s.1.take() {
                w.wake();
            }
        }
    }

    impl Future for TestPanel {
        type Output = Answer;
        fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Answer> {
            let mut s = self.0.lock().unwrap();
            match s.0.take() {
                Some(answer) => Poll::Ready(answer),
                None => {
                    s.1 = Some(cx.waker().clone());
                    Poll::Pending
                }
            }
        }
    }

    #[test]
    fn the_answer_comes_back_once_after_the_panel_closes() {
        let ctx = egui::Context::default();
        let panel = TestPanel::default();
        let opened = AtomicUsize::new(0);
        let open = || {
            opened.fetch_add(1, Ordering::Relaxed);
            Box::pin(panel.clone()) as super::Panel
        };
        start(&ctx, "k", open);
        assert_eq!(take(&ctx, "k"), None, "still up");
        // Asking again while it is up does not open a second one.
        start(&ctx, "k", || unreachable!());
        assert_eq!(take(&ctx, "other"), None);

        let _ = ctx.run_ui(Default::default(), |_| {}); // clears earlier repaint requests
        panel.close(Some(vec![PathBuf::from("/a"), PathBuf::from("/b")]));
        assert!(ctx.has_requested_repaint(), "closing the panel wakes the UI");
        assert_eq!(take_all(&ctx, "k"), Some(vec![PathBuf::from("/a"), PathBuf::from("/b")]));
        assert_eq!(take(&ctx, "k"), None, "taken once");
        assert_eq!(opened.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn cancel_gives_nothing_and_the_panel_can_be_asked_for_again() {
        let ctx = egui::Context::default();
        let panel = TestPanel::default();
        start(&ctx, "k", || Box::pin(panel.clone()));
        panel.close(None);
        assert_eq!(take(&ctx, "k"), None);
        let again = TestPanel::default();
        again.close(Some(vec![PathBuf::from("/c")]));
        start(&ctx, "k", || Box::pin(again.clone()));
        assert_eq!(take(&ctx, "k"), Some(PathBuf::from("/c")));
    }
}
