//! The screen set aside for test runs (a script, or their own data folder): they open there,
//! in the background, so they never land on the screen being worked on. It is named by
//! `POLYMIX_TEST_SCREEN`, or else by the first line of `~/.config/polymix/test-screen`: part
//! of its name as macOS shows it (e.g. "U2715H (1)"), case ignored. A screen, not a desktop
//! (Space): eframe draws nothing in a window on a desktop that isn't showing.

/// The named screen's usable area, in points from the top left of the main screen (the
/// coordinates window positions use), or None when none is named or it isn't connected.
pub fn rect() -> Option<egui::Rect> {
    let wanted = std::env::var("POLYMIX_TEST_SCREEN")
        .ok()
        .or_else(|| {
            let home = std::env::var_os("HOME")?;
            let text = std::fs::read_to_string(std::path::Path::new(&home).join(".config/polymix/test-screen")).ok()?;
            text.lines().next().map(str::to_string)
        })
        .map(|s| s.trim().to_lowercase())
        .filter(|s| !s.is_empty())?;
    let screens = screens();
    let main = screens.first()?.1;
    let (_, _, v) = screens.iter().find(|(name, _, _)| name.to_lowercase().contains(&wanted))?;
    // AppKit measures up from the bottom of the main screen.
    let top = main.y + main.h - (v.y + v.h);
    Some(egui::Rect::from_min_size(egui::pos2(v.x as f32, top as f32), egui::vec2(v.w as f32, v.h as f32)))
}

/// This run's place among the test runs open at the same time (0, 1, ...; 8 at most), held
/// until it quits: windows are staggered by it, so none covers another completely (and stops
/// being drawn).
pub fn slot() -> usize {
    static HELD: std::sync::OnceLock<Option<(usize, std::fs::File)>> = std::sync::OnceLock::new();
    HELD.get_or_init(|| {
        (0..8).find_map(|k| {
            let file = std::fs::OpenOptions::new().create(true).truncate(false).write(true).open(std::env::temp_dir().join(format!("polymix-test-window-{k}.lock"))).ok()?;
            file.try_lock().ok()?;
            Some((k, file))
        })
    })
    .as_ref()
    .map_or(0, |(k, _)| *k)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Rect {
    x: f64,
    y: f64,
    w: f64,
    h: f64,
}

#[cfg(target_os = "macos")]
// SAFETY: laid out as CGRect { CGPoint origin; CGSize size; } with CGFloat = f64.
unsafe impl objc2::Encode for Rect {
    const ENCODING: objc2::Encoding = objc2::Encoding::Struct(
        "CGRect",
        &[objc2::Encoding::Struct("CGPoint", &[f64::ENCODING, f64::ENCODING]), objc2::Encoding::Struct("CGSize", &[f64::ENCODING, f64::ENCODING])],
    );
}

/// Every screen's name, frame and visible frame (without the menu bar and Dock), the main
/// screen first.
#[cfg(target_os = "macos")]
fn screens() -> Vec<(String, Rect, Rect)> {
    use objc2::msg_send;
    use objc2::runtime::{AnyClass, AnyObject};
    let Some(class) = AnyClass::get(c"NSScreen") else { return Vec::new() };
    objc2::rc::autoreleasepool(|_| {
        // SAFETY: plain AppKit reads on the main thread (before the event loop starts).
        unsafe {
            let list: *mut AnyObject = msg_send![class, screens];
            if list.is_null() {
                return Vec::new();
            }
            let n: usize = msg_send![&*list, count];
            (0..n)
                .filter_map(|i| {
                    let s: *mut AnyObject = msg_send![&*list, objectAtIndex: i];
                    if s.is_null() {
                        return None;
                    }
                    let name: *mut AnyObject = msg_send![&*s, localizedName];
                    let name = if name.is_null() {
                        String::new()
                    } else {
                        let p: *const std::ffi::c_char = msg_send![&*name, UTF8String];
                        if p.is_null() { String::new() } else { std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned() }
                    };
                    let frame: Rect = msg_send![&*s, frame];
                    let visible: Rect = msg_send![&*s, visibleFrame];
                    Some((name, frame, visible))
                })
                .collect()
        }
    })
}

#[cfg(not(target_os = "macos"))]
fn screens() -> Vec<(String, Rect, Rect)> {
    Vec::new()
}
