//! macOS: files opened from the Finder (double-click, Open With, dropping on the Dock icon)
//! arrive as an Apple Event that AppKit hands to the application delegate's
//! `application:openURLs:`, which winit's delegate does not implement. Just before AppKit
//! finishes launching (when the first such event is delivered), we add that method to
//! winit's delegate class; the paths it receives wait in a queue the shell opens each frame.

use objc2::runtime::{AnyClass, AnyObject, Imp, Sel};
use objc2::{msg_send, sel};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

static OPENED: Mutex<Vec<PathBuf>> = Mutex::new(Vec::new());
static CONTEXT: OnceLock<egui::Context> = OnceLock::new();
static ORIGINAL_FINISH: OnceLock<Imp> = OnceLock::new();
static INSTALLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

type FinishLaunching = unsafe extern "C-unwind" fn(*mut AnyObject, Sel);
type OpenUrls = unsafe extern "C-unwind" fn(*mut AnyObject, Sel, *mut AnyObject, *mut AnyObject);

/// Paths opened since the last call.
pub fn take() -> Vec<PathBuf> {
    std::mem::take(&mut *OPENED.lock().unwrap_or_else(|e| e.into_inner()))
}

/// Repaint this context when files arrive while the app is idle.
pub fn set_context(ctx: &egui::Context) {
    let _ = CONTEXT.set(ctx.clone());
}

unsafe extern "C-unwind" fn open_urls(_this: *mut AnyObject, _sel: Sel, _app: *mut AnyObject, urls: *mut AnyObject) {
    if urls.is_null() {
        return;
    }
    // SAFETY: `urls` is the NSArray<NSURL *> AppKit passes to `application:openURLs:`.
    let paths: Vec<PathBuf> = unsafe {
        let n: usize = msg_send![&*urls, count];
        (0..n)
            .filter_map(|i| {
                let url: *mut AnyObject = msg_send![&*urls, objectAtIndex: i];
                if url.is_null() {
                    return None;
                }
                let path: *mut AnyObject = msg_send![&*url, path];
                if path.is_null() {
                    return None;
                }
                let utf8: *const std::ffi::c_char = msg_send![&*path, UTF8String];
                (!utf8.is_null()).then(|| PathBuf::from(std::ffi::CStr::from_ptr(utf8).to_string_lossy().into_owned()))
            })
            .collect()
    };
    log::info!("opened from the Finder: {} item(s)", paths.len());
    OPENED.lock().unwrap_or_else(|e| e.into_inner()).extend(paths);
    if let Some(ctx) = CONTEXT.get() {
        ctx.request_repaint();
    }
}

unsafe extern "C-unwind" fn finish_launching(this: *mut AnyObject, sel: Sel) {
    // SAFETY: `this` is the NSApplication; by now winit has set its delegate.
    unsafe {
        let delegate: *mut AnyObject = msg_send![&*this, delegate];
        if !delegate.is_null() {
            let class = (*delegate).class() as *const AnyClass as *mut AnyClass;
            let imp: Imp = std::mem::transmute::<OpenUrls, Imp>(open_urls);
            objc2::ffi::class_addMethod(class, sel!(application:openURLs:), imp, c"v@:@@".as_ptr());
        }
        if let Some(original) = ORIGINAL_FINISH.get() {
            let original = std::mem::transmute::<Imp, FinishLaunching>(*original);
            original(this, sel);
        }
    }
}

/// Hook `-[NSApplication finishLaunching]`. Call once, before the event loop runs.
pub fn install() {
    if INSTALLED.swap(true, std::sync::atomic::Ordering::SeqCst) {
        return;
    }
    let Some(app) = AnyClass::get(c"NSApplication") else { return };
    let Some(method) = app.instance_method(sel!(finishLaunching)) else { return };
    // SAFETY: our replacement has the signature of `finishLaunching` and calls the original.
    unsafe {
        let ours = std::mem::transmute::<FinishLaunching, Imp>(finish_launching);
        let original = method.set_implementation(ours);
        let _ = ORIGINAL_FINISH.set(original);
    }
}
