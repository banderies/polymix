//! 3D mouse (3Dconnexion SpaceMouse) navigation.
//!
//! macOS: the 3DxWare driver's client framework is loaded at run time, so Polymix builds and
//! runs without it. The app registers as a client for its own process name and receives the
//! cap's six axes and its buttons while it is the frontmost application. Other platforms
//! report `Unsupported` for now (spacenavd / raw HID would slot in behind the same type).
//!
//! Motion follows the usual "object mode": push the cap the way you want the model to move.

use crate::camera::Camera;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Cap deflection that counts as "full" (SpaceMouse axes report roughly ±350).
const FULL_SCALE: f32 = 350.0;
/// Axis values below this are treated as rest.
const DEAD_ZONE: i16 = 4;
/// A cap that stopped reporting for this long is treated as released (missed final event).
const STALE: Duration = Duration::from_millis(400);

/// Cap state in the *view* frame: translation (x right, y up, z toward the viewer) and
/// rotation about those axes, about -1..1 at full deflection.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Motion {
    pub t: [f32; 3],
    pub r: [f32; 3],
}

impl Motion {
    pub fn is_zero(&self) -> bool {
        self.t.iter().chain(self.r.iter()).all(|v| *v == 0.0)
    }
}

/// User settings (persisted by the app).
#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    pub enabled: bool,
    /// Overall speed multiplier.
    pub speed: f32,
    pub pan_speed: f32,
    pub zoom_speed: f32,
    pub rotate_speed: f32,
    /// Invert pan X, pan Y, zoom, tilt, spin, roll.
    pub invert: [bool; 6],
    pub lock_rotation: bool,
    pub lock_translation: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self { enabled: true, speed: 1.0, pan_speed: 1.0, zoom_speed: 1.0, rotate_speed: 1.0, invert: [false; 6], lock_rotation: false, lock_translation: false }
    }
}

impl Settings {
    pub const AXIS_NAMES: [&'static str; 6] = ["Pan left/right", "Pan up/down", "Zoom", "Tilt", "Spin", "Roll"];

    /// Compact text form for the preferences store.
    pub fn to_pref(&self) -> String {
        let inv: String = self.invert.iter().map(|b| if *b { '1' } else { '0' }).collect();
        format!(
            "{} {} {} {} {} {} {} {}",
            self.enabled as u8, self.speed, self.pan_speed, self.zoom_speed, self.rotate_speed, inv, self.lock_rotation as u8, self.lock_translation as u8
        )
    }

    pub fn from_pref(s: &str) -> Option<Self> {
        let f: Vec<&str> = s.split_whitespace().collect();
        if f.len() != 8 || f[5].len() != 6 {
            return None;
        }
        let mut invert = [false; 6];
        for (i, c) in f[5].chars().enumerate() {
            invert[i] = c == '1';
        }
        Some(Self {
            enabled: f[0] == "1",
            speed: f[1].parse().ok()?,
            pan_speed: f[2].parse().ok()?,
            zoom_speed: f[3].parse().ok()?,
            rotate_speed: f[4].parse().ok()?,
            invert,
            lock_rotation: f[6] == "1",
            lock_translation: f[7] == "1",
        })
    }
}

/// What the app should do for a device button.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ButtonAction {
    Menu,
    Fit,
    Top,
    Right,
    Front,
}

/// Default button assignments (3Dconnexion's standard button numbering: 0 Menu, 1 Fit,
/// 2 Top, 4 Right, 5 Front on models that have them).
pub fn button_action(index: u32) -> Option<ButtonAction> {
    match index {
        0 => Some(ButtonAction::Menu),
        1 => Some(ButtonAction::Fit),
        2 => Some(ButtonAction::Top),
        4 => Some(ButtonAction::Right),
        5 => Some(ButtonAction::Front),
        _ => None,
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Status {
    /// No 3D mouse support on this platform yet.
    #[cfg_attr(target_os = "macos", allow(dead_code))]
    Unsupported,
    /// The 3Dconnexion driver (3DxWare) is not installed.
    DriverMissing,
    /// Registered with the driver; `device` is the connected device's (vendor, product) ID.
    Ready { device: Option<(u16, u16)> },
    Failed(String),
}

#[derive(Default)]
struct Shared {
    axis: [i16; 6],
    last_axis: Option<Instant>,
    buttons: u32,
    pressed: Vec<u32>,
}

fn shared() -> &'static Mutex<Shared> {
    static SHARED: OnceLock<Mutex<Shared>> = OnceLock::new();
    SHARED.get_or_init(|| Mutex::new(Shared::default()))
}

/// Context to wake when the device reports (the driver calls back on its own thread).
static REPAINT: OnceLock<egui::Context> = OnceLock::new();

fn wake() {
    if let Some(ctx) = REPAINT.get() {
        ctx.request_repaint();
    }
}

/// Record a device report (called from the driver thread).
fn on_axis(axis: [i16; 6]) {
    if let Ok(mut s) = shared().lock() {
        s.axis = axis;
        s.last_axis = Some(Instant::now());
    }
    wake();
}

fn on_buttons(buttons: u32) {
    if let Ok(mut s) = shared().lock() {
        let rising = buttons & !s.buttons;
        s.pressed.extend((0..32).filter(|b| rising & (1 << b) != 0));
        s.buttons = buttons;
    }
    wake();
}


/// Device axes (as reported by the driver: x right, y toward the user, z down, rotations
/// about those) to the view frame (x right, y up, z toward the viewer).
pub fn to_view(axis: [i16; 6]) -> Motion {
    let n = |v: i16| if v.abs() < DEAD_ZONE { 0.0 } else { (v as f32 / FULL_SCALE).clamp(-1.5, 1.5) };
    Motion { t: [n(axis[0]), -n(axis[2]), n(axis[1])], r: [-n(axis[3]), n(axis[5]), -n(axis[4])] }
}

pub struct SpaceMouse {
    pub status: Status,
    /// Keeps the driver registration alive (unregistered on drop).
    #[cfg(target_os = "macos")]
    _client: Option<mac::Client>,
    last_check: Instant,
}

impl SpaceMouse {
    pub fn start(ctx: &egui::Context) -> Self {
        Self::start_with(ctx, false)
    }

    /// `background`: receive input even when this process is not the frontmost app
    /// (diagnostics); normally the driver only reports to the app in front.
    pub fn start_with(ctx: &egui::Context, background: bool) -> Self {
        let _ = REPAINT.set(ctx.clone());
        #[cfg(target_os = "macos")]
        {
            match mac::Client::register(background) {
                Ok(client) => Self { status: Status::Ready { device: client.device() }, _client: Some(client), last_check: Instant::now() },
                Err(mac::Error::DriverMissing) => Self { status: Status::DriverMissing, _client: None, last_check: Instant::now() },
                Err(mac::Error::Failed(e)) => {
                    log::warn!("3D mouse: {e}");
                    Self { status: Status::Failed(e), _client: None, last_check: Instant::now() }
                }
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = background;
            Self { status: Status::Unsupported, last_check: Instant::now() }
        }
    }

    /// Re-check which device is connected (about once a second; devices come and go).
    fn refresh_status(&mut self) {
        if self.last_check.elapsed() < Duration::from_secs(1) {
            return;
        }
        self.last_check = Instant::now();
        #[cfg(target_os = "macos")]
        if let (Status::Ready { device }, Some(c)) = (&mut self.status, &self._client) {
            *device = c.device();
        }
    }

    /// Current motion and the buttons pressed since the last call.
    pub fn poll(&mut self) -> (Motion, Vec<u32>) {
        self.refresh_status();
        let Ok(mut s) = shared().lock() else { return (Motion::default(), Vec::new()) };
        let fresh = s.last_axis.is_some_and(|t| t.elapsed() < STALE);
        if !fresh {
            s.axis = [0; 6];
        }
        (to_view(s.axis), std::mem::take(&mut s.pressed))
    }

    /// Raw axes and button bits (for the settings readout).
    pub fn raw(&self) -> ([i16; 6], u32) {
        shared().lock().map(|s| (s.axis, s.buttons)).unwrap_or(([0; 6], 0))
    }
}

/// Move the camera for `dt` seconds of cap deflection. Returns true if anything moved.
pub fn apply(camera: &mut Camera, m: Motion, s: &Settings, dt: f64, viewport_h: f64) -> bool {
    if m.is_zero() {
        return false;
    }
    let sign = |i: usize| if s.invert[i] { -1.0 } else { 1.0 };
    let k = s.speed as f64 * dt;
    if !s.lock_translation {
        // Object mode: the model follows the cap. Pan in screen points (one viewport height
        // per second at full deflection), zoom by e^2 per second.
        let pan_rate = viewport_h * s.pan_speed as f64 * k;
        let dx = m.t[0] as f64 * sign(0) * pan_rate;
        let dy = -(m.t[1] as f64) * sign(1) * pan_rate;
        if dx != 0.0 || dy != 0.0 {
            camera.pan(dx, dy, viewport_h);
        }
        let z = m.t[2] as f64 * sign(2);
        if z != 0.0 {
            camera.zoom((-2.0 * s.zoom_speed as f64 * k * z).exp());
        }
    }
    if !s.lock_rotation {
        // About 140°/s at full deflection.
        let rate = 2.5 * s.rotate_speed as f64 * k;
        camera.rotate_view(m.r[0] as f64 * sign(3) * rate, m.r[1] as f64 * sign(4) * rate, m.r[2] as f64 * sign(5) * rate);
    }
    true
}

#[cfg(target_os = "macos")]
mod mac {
    //! Minimal bindings to the 3DxWare client framework (own declarations; the framework
    //! is loaded with `dlopen` so it stays optional).
    use super::{on_axis, on_buttons};
    use libloading::{Library, Symbol};
    use std::ffi::c_void;

    const FRAMEWORK: &str = "/Library/Frameworks/3DconnexionClient.framework/3DconnexionClient";
    /// Match the client by application name rather than by a four-char signature.
    const SIGNATURE_WILDCARD: u32 = 0x2A2A_2A2A;
    /// A client that is activated explicitly instead of by being the frontmost app.
    const SIGNATURE_MANUAL: u32 = 0x2B2B_2B2B;
    const CTL_ACTIVATE_CLIENT: u32 = u32::from_be_bytes(*b"3dac");
    /// Receive all axes and buttons; the driver does not run its own assignments.
    const MODE_TAKE_OVER: u16 = 1;
    const MASK_ALL_AXES_AND_BUTTONS: u32 = 0x3FFF;
    const MASK_ALL_BUTTONS: u32 = 0xFFFF_FFFF;
    const MSG_DEVICE_STATE: u32 = u32::from_be_bytes(*b"3dSR");
    const CMD_BUTTONS: u16 = 2;
    const CMD_AXIS: u16 = 3;
    const CTL_GET_DEVICE_ID: u32 = u32::from_be_bytes(*b"3did");

    /// Device report as delivered by the driver (2-byte packed, 48 bytes).
    #[repr(C, packed(2))]
    #[derive(Clone, Copy)]
    struct DeviceState {
        version: u16,
        client: u16,
        command: u16,
        param: i16,
        value: i32,
        time: u64,
        report: [u8; 8],
        buttons8: u16,
        axis: [i16; 6],
        address: u16,
        buttons: u32,
    }
    const _: () = assert!(std::mem::size_of::<DeviceState>() == 48);

    type MessageHandler = unsafe extern "C" fn(u32, u32, *mut c_void);
    type DeviceHandler = unsafe extern "C" fn(u32);
    type SetHandlersFn = unsafe extern "C" fn(MessageHandler, DeviceHandler, DeviceHandler, bool) -> i16;
    type RegisterFn = unsafe extern "C" fn(u32, *const u8, u16, u32) -> u16;
    type ButtonMaskFn = unsafe extern "C" fn(u16, u32);
    type UnregisterFn = unsafe extern "C" fn(u16);
    type CleanupFn = unsafe extern "C" fn();
    type ControlFn = unsafe extern "C" fn(u32, i32, *mut i32) -> i16;
    type ClientControlFn = unsafe extern "C" fn(u16, u32, i32, *mut i32) -> i16;

    unsafe extern "C" fn message(_product: u32, kind: u32, arg: *mut c_void) {
        if kind != MSG_DEVICE_STATE || arg.is_null() {
            return;
        }
        // SAFETY: for device-state messages the driver passes a pointer to its report struct.
        let s = unsafe { std::ptr::read_unaligned(arg as *const DeviceState) };
        match s.command {
            CMD_AXIS => on_axis(s.axis),
            CMD_BUTTONS => on_buttons(s.buttons),
            _ => {}
        }
    }
    unsafe extern "C" fn added(_product: u32) {
        super::wake();
    }
    unsafe extern "C" fn removed(_product: u32) {
        super::wake();
    }

    pub enum Error {
        DriverMissing,
        Failed(String),
    }

    pub struct Client {
        lib: Library,
        id: u16,
    }

    impl Client {
        pub fn register(background: bool) -> Result<Self, Error> {
            if !std::path::Path::new(FRAMEWORK).exists() {
                return Err(Error::DriverMissing);
            }
            // SAFETY: loading the vendor framework runs its initialisers; it is the documented
            // client library of the installed driver.
            let lib = unsafe { Library::new(FRAMEWORK) }.map_err(|e| Error::Failed(format!("loading the 3Dconnexion framework: {e}")))?;
            // The process name, as a Pascal string, identifies this app to the driver.
            let name = std::env::current_exe()
                .ok()
                .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
                .unwrap_or_else(|| "polymix".into());
            let mut pname = vec![name.len().min(255) as u8];
            pname.extend(name.bytes().take(255));
            // SAFETY: symbol types match the framework's C API; the handlers are 'static
            // functions and stay valid until `CleanupConnexionHandlers` in `drop`.
            unsafe {
                let set: Symbol<SetHandlersFn> = lib.get(b"SetConnexionHandlers\0").map_err(|e| Error::Failed(e.to_string()))?;
                let register: Symbol<RegisterFn> = lib.get(b"RegisterConnexionClient\0").map_err(|e| Error::Failed(e.to_string()))?;
                let button_mask: Symbol<ButtonMaskFn> = lib.get(b"SetConnexionClientButtonMask\0").map_err(|e| Error::Failed(e.to_string()))?;
                let err = set(message, added, removed, true);
                if err != 0 {
                    return Err(Error::Failed(format!("SetConnexionHandlers returned {err}")));
                }
                let signature = if background { SIGNATURE_MANUAL } else { SIGNATURE_WILDCARD };
                let id = register(signature, pname.as_ptr(), MODE_TAKE_OVER, MASK_ALL_AXES_AND_BUTTONS);
                if id == 0 {
                    return Err(Error::Failed("RegisterConnexionClient failed".into()));
                }
                button_mask(id, MASK_ALL_BUTTONS);
                if background {
                    let control: Symbol<ClientControlFn> = lib.get(b"ConnexionClientControl\0").map_err(|e| Error::Failed(e.to_string()))?;
                    let mut result = 0;
                    control(id, CTL_ACTIVATE_CLIENT, 0, &mut result);
                }
                Ok(Self { lib, id })
            }
        }
    }

    impl Client {
        /// (vendor, product) ID of the connected device, if any.
        pub fn device(&self) -> Option<(u16, u16)> {
            let mut result: i32 = 0;
            // SAFETY: documented control call; `result` outlives it.
            let err = unsafe {
                let control: Symbol<ControlFn> = self.lib.get(b"ConnexionControl\0").ok()?;
                control(CTL_GET_DEVICE_ID, 0, &mut result)
            };
            (err == 0 && result != 0).then_some((((result as u32) >> 16) as u16, (result as u32 & 0xFFFF) as u16))
        }
    }

    impl Drop for Client {
        fn drop(&mut self) {
            // SAFETY: same symbols as above; unregistering stops callbacks before the
            // library is unloaded.
            unsafe {
                if let Ok(unregister) = self.lib.get::<UnregisterFn>(b"UnregisterConnexionClient\0") {
                    unregister(self.id);
                }
                if let Ok(cleanup) = self.lib.get::<CleanupFn>(b"CleanupConnexionHandlers\0") {
                    cleanup();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_axes_map_to_view_frame() {
        // Push right, pull up (device z is down), pull toward the user.
        let m = to_view([350, 0, 0, 0, 0, 0]);
        assert_eq!(m.t, [1.0, 0.0, 0.0]);
        assert_eq!(to_view([0, 0, -350, 0, 0, 0]).t, [0.0, 1.0, 0.0]);
        assert_eq!(to_view([0, 350, 0, 0, 0, 0]).t, [0.0, 0.0, 1.0]);
        // Dead zone.
        assert!(to_view([2, -3, 1, 0, 3, -2]).is_zero());
    }

    #[test]
    fn settings_round_trip() {
        let s = Settings { enabled: false, speed: 1.5, invert: [true, false, true, false, false, true], lock_rotation: true, ..Default::default() };
        assert_eq!(Settings::from_pref(&s.to_pref()), Some(s));
        assert_eq!(Settings::from_pref("garbage"), None);
    }

    #[test]
    fn motion_moves_the_camera() {
        let mut c = Camera::default();
        let s = Settings::default();
        let t0 = c.target;
        assert!(apply(&mut c, Motion { t: [1.0, 0.0, 0.0], r: [0.0; 3] }, &s, 0.1, 800.0));
        assert!(c.target.distance(t0) > 0.0);
        let d0 = c.distance;
        apply(&mut c, Motion { t: [0.0, 0.0, 1.0], r: [0.0; 3] }, &s, 0.1, 800.0);
        assert!(c.distance < d0, "pulling the cap toward you zooms in");
        let f0 = c.forward();
        apply(&mut c, Motion { t: [0.0; 3], r: [0.0, 1.0, 0.0] }, &s, 0.1, 800.0);
        assert!(c.forward().dot(f0) < 0.9999);
        let locked = Settings { lock_rotation: true, lock_translation: true, ..Default::default() };
        let before = (c.target, c.distance, c.forward());
        apply(&mut c, Motion { t: [1.0; 3], r: [1.0; 3] }, &locked, 0.1, 800.0);
        assert_eq!(before, (c.target, c.distance, c.forward()));
    }
}
