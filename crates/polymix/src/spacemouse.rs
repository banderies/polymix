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
/// Largest deflection used, in full deflections. 3Dconnexion's own sensitivity setting scales
/// what the driver reports, so a cap near 1 (it was 1.5) would undo its higher settings; this
/// one only guards against glitches.
const MAX_DEFLECTION: f32 = 6.0;
/// Axis values below this are treated as rest.
const DEAD_ZONE: i16 = 4;
/// A cap that stopped reporting for this long is treated as released (missed final event).
const STALE: Duration = Duration::from_millis(400);

/// Rates at full deflection and speed 1: orbit in radians per second (about 143°/s), pan in
/// viewport heights per second, zoom in e-folds per second (about 7.4× closer per second).
pub const ORBIT_RATE: f64 = 2.5;
pub const PAN_RATE: f64 = 1.0;
pub const ZOOM_RATE: f64 = 2.0;
/// Range of the orbit, pan and zoom speed sliders.
pub const SPEED_RANGE: std::ops::RangeInclusive<f32> = 0.1..=10.0;

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
    /// Speed multipliers (within [`SPEED_RANGE`]) for orbiting, panning and zooming.
    pub orbit_speed: f32,
    pub pan_speed: f32,
    pub zoom_speed: f32,
    /// Invert pan X, pan Y, zoom, tilt, spin, roll.
    pub invert: [bool; 6],
    pub lock_rotation: bool,
    pub lock_translation: bool,
    /// What the view turns about.
    pub centre: Centre,
    /// Show the centre of rotation while the 3D mouse moves the view.
    pub show_centre: bool,
}

/// What the 3D mouse turns the view about.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Centre {
    /// The object's centre of mass.
    Object,
    /// What is in view, chosen again at the start of every move (as Fusion 360 does): the
    /// object's centre while all of it is in view; zoomed in, the surface at the middle of the
    /// view, or the surface nearest the middle.
    View,
}

impl Centre {
    pub fn key(self) -> &'static str {
        match self {
            Centre::Object => "object",
            Centre::View => "view",
        }
    }

    fn from_key(s: &str) -> Option<Self> {
        [Centre::Object, Centre::View].into_iter().find(|c| c.key() == s)
    }
}

impl Default for Settings {
    fn default() -> Self {
        Self { enabled: true, orbit_speed: 1.0, pan_speed: 1.0, zoom_speed: 1.0, invert: [false; 6], lock_rotation: false, lock_translation: false, centre: Centre::View, show_centre: true }
    }
}

impl Settings {
    pub const AXIS_NAMES: [&'static str; 6] = ["Pan left/right", "Pan up/down", "Zoom", "Tilt", "Spin", "Roll"];

    /// Compact text form for the preferences store: enabled, orbit, pan and zoom speeds,
    /// inverted axes, locks, centre of rotation and whether it is shown.
    pub fn to_pref(&self) -> String {
        let inv: String = self.invert.iter().map(|b| if *b { '1' } else { '0' }).collect();
        format!(
            "{} {} {} {} {} {} {} {} {}",
            self.enabled as u8,
            self.orbit_speed,
            self.pan_speed,
            self.zoom_speed,
            inv,
            self.lock_rotation as u8,
            self.lock_translation as u8,
            self.centre.key(),
            self.show_centre as u8
        )
    }

    /// Reads [`Self::to_pref`]'s form, the one before it without the centre of rotation
    /// (taken as the default's), and the first one, with an overall speed in front of pan,
    /// zoom and rotate speeds (folded into each).
    pub fn from_pref(s: &str) -> Option<Self> {
        let f: Vec<&str> = s.split_whitespace().collect();
        let num = |i: usize| f[i].parse::<f32>().ok().filter(|v| v.is_finite() && *v > 0.0);
        let defaults = Self::default();
        let (orbit, pan, zoom, rest, centre, show_centre) = match f.len() {
            9 => (num(1)?, num(2)?, num(3)?, &f[4..7], Centre::from_key(f[7])?, f[8] == "1"),
            7 => (num(1)?, num(2)?, num(3)?, &f[4..], defaults.centre, defaults.show_centre),
            8 => {
                let overall = num(1)?;
                (overall * num(4)?, overall * num(2)?, overall * num(3)?, &f[5..], defaults.centre, defaults.show_centre)
            }
            _ => return None,
        };
        if rest[0].len() != 6 {
            return None;
        }
        let mut invert = [false; 6];
        for (i, c) in rest[0].chars().enumerate() {
            invert[i] = c == '1';
        }
        let clamp = |v: f32| v.clamp(*SPEED_RANGE.start(), *SPEED_RANGE.end());
        Some(Self {
            enabled: f[0] == "1",
            orbit_speed: clamp(orbit),
            pan_speed: clamp(pan),
            zoom_speed: clamp(zoom),
            invert,
            lock_rotation: rest[1] == "1",
            lock_translation: rest[2] == "1",
            centre,
            show_centre,
        })
    }
}

/// What a device button does (Preferences → 3D mouse → Buttons).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ButtonAction {
    /// The Space hotbox.
    Hotbox,
    /// Fit the view to the visible objects.
    Fit,
    /// A standard view: its index in the View menu's list (Front, Back, Left, Right, Top,
    /// Bottom).
    View(usize),
    Undo,
    Redo,
    /// Accept the open tool (as Enter does).
    Accept,
    /// Cancel the open tool (as Esc does).
    Cancel,
    /// Open a mesh tool by its name (the palette's, hotkeys' and scripts').
    Tool(String),
    /// While held, the app acts as if this modifier key were held (see [`with_held_keys`]).
    Hold(HeldKey),
    /// Press a keyboard key (with modifiers): whatever the key does where it is pressed.
    Key { key: egui::Key, modifiers: egui::Modifiers },
    /// Nothing (a button captured but not given an action yet).
    Nothing,
}

/// Whether `m` holds modifier key `k` (⌘ and Ctrl both set egui's `command` off a Mac).
pub fn holds(m: egui::Modifiers, k: HeldKey) -> bool {
    let mac = cfg!(target_os = "macos");
    match k {
        HeldKey::Command => m.command && (mac || !m.ctrl),
        HeldKey::Shift => m.shift,
        HeldKey::Option => m.alt,
        HeldKey::Control => m.ctrl,
    }
}

/// How the keyboard shows key `key` with `modifiers`: ⌘⇧I.
pub fn key_label(key: egui::Key, modifiers: egui::Modifiers) -> String {
    let symbols = [(HeldKey::Control, "⌃"), (HeldKey::Option, "⌥"), (HeldKey::Shift, "⇧"), (HeldKey::Command, "⌘")];
    symbols.iter().filter(|(k, _)| holds(modifiers, *k)).map(|(_, s)| *s).chain([key.symbol_or_name()]).collect()
}

/// A modifier key a 3D mouse button can stand for while it is held.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeldKey {
    Command,
    Shift,
    Option,
    Control,
}

impl HeldKey {
    pub const ALL: [HeldKey; 4] = [HeldKey::Command, HeldKey::Shift, HeldKey::Option, HeldKey::Control];

    pub fn key(self) -> &'static str {
        match self {
            HeldKey::Command => "command",
            HeldKey::Shift => "shift",
            HeldKey::Option => "option",
            HeldKey::Control => "control",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            HeldKey::Command => "⌘ Command",
            HeldKey::Shift => "⇧ Shift",
            HeldKey::Option => "⌥ Option",
            HeldKey::Control => "⌃ Control",
        }
    }

    /// The modifiers the keyboard reports for the key: ⌘ is egui's `command` on a Mac, Ctrl
    /// elsewhere.
    pub fn modifiers(self) -> egui::Modifiers {
        let mac = cfg!(target_os = "macos");
        match self {
            HeldKey::Command => egui::Modifiers { mac_cmd: mac, command: true, ..Default::default() },
            HeldKey::Shift => egui::Modifiers::SHIFT,
            HeldKey::Option => egui::Modifiers::ALT,
            HeldKey::Control => egui::Modifiers { ctrl: true, command: !mac, ..Default::default() },
        }
    }
}

impl ButtonAction {
    /// The form kept in the preferences.
    pub fn key(&self) -> String {
        match self {
            ButtonAction::Hotbox => "hotbox".into(),
            ButtonAction::Fit => "fit".into(),
            ButtonAction::View(k) => format!("view:{k}"),
            ButtonAction::Undo => "undo".into(),
            ButtonAction::Redo => "redo".into(),
            ButtonAction::Accept => "accept".into(),
            ButtonAction::Cancel => "cancel".into(),
            ButtonAction::Tool(name) => format!("tool:{name}"),
            ButtonAction::Hold(k) => format!("hold:{}", k.key()),
            ButtonAction::Key { key, modifiers } => {
                let held: Vec<&str> = HeldKey::ALL.into_iter().filter(|h| holds(*modifiers, *h)).map(HeldKey::key).collect();
                format!("key:{}", held.into_iter().chain([key.name()]).collect::<Vec<_>>().join("+"))
            }
            ButtonAction::Nothing => "nothing".into(),
        }
    }

    pub fn from_key(s: &str) -> Option<Self> {
        Some(match s {
            "hotbox" => ButtonAction::Hotbox,
            "fit" => ButtonAction::Fit,
            "undo" => ButtonAction::Undo,
            "redo" => ButtonAction::Redo,
            "accept" => ButtonAction::Accept,
            "cancel" => ButtonAction::Cancel,
            "nothing" => ButtonAction::Nothing,
            _ => match s.split_once(':')? {
                ("view", k) => ButtonAction::View(k.parse().ok()?),
                ("tool", name) if !name.is_empty() => ButtonAction::Tool(name.into()),
                ("hold", k) => ButtonAction::Hold(HeldKey::ALL.into_iter().find(|h| h.key() == k)?),
                ("key", k) => {
                    let mut parts: Vec<&str> = k.split('+').collect();
                    let key = egui::Key::from_name(parts.pop()?)?;
                    let modifiers = parts.into_iter().map(|p| HeldKey::ALL.into_iter().find(|h| h.key() == p).map(HeldKey::modifiers)).collect::<Option<Vec<_>>>()?;
                    ButtonAction::Key { key, modifiers: modifiers.into_iter().fold(egui::Modifiers::NONE, |a, b| a | b) }
                }
                _ => return None,
            },
        })
    }
}

/// The buttons' actions unless the user maps them: 3Dconnexion's standard numbering (0 Menu,
/// 1 Fit, 2 Top, 4 Right, 5 Front on models that have them).
pub fn default_buttons() -> Vec<(u32, ButtonAction)> {
    vec![(0, ButtonAction::Hotbox), (1, ButtonAction::Fit), (2, ButtonAction::View(4)), (4, ButtonAction::View(3)), (5, ButtonAction::View(0))]
}

/// Button mappings in the preferences' form: `button=action` pairs separated by `;`.
pub fn buttons_to_pref(buttons: &[(u32, ButtonAction)]) -> String {
    buttons.iter().map(|(b, a)| format!("{b}={}", a.key())).collect::<Vec<_>>().join(";")
}

/// The mappings of [`buttons_to_pref`]'s form; pairs it cannot read are left out.
pub fn buttons_from_pref(s: &str) -> Vec<(u32, ButtonAction)> {
    let mut v: Vec<(u32, ButtonAction)> = s.split(';').filter_map(|p| p.split_once('=')).filter_map(|(b, a)| Some((b.trim().parse().ok()?, ButtonAction::from_key(a.trim())?))).collect();
    v.sort_by_key(|(b, _)| *b);
    v.dedup_by_key(|(b, _)| *b);
    v
}

/// The buttons standing for modifier keys while held, as the settings last gave them. The
/// app's input hook reads them in every tab.
static HELD_KEYS: Mutex<Vec<(u32, HeldKey)>> = Mutex::new(Vec::new());

/// Set the buttons that stand for modifier keys (none while the 3D mouse is off).
pub fn set_held_keys(keys: Vec<(u32, HeldKey)>) {
    if let Ok(mut k) = HELD_KEYS.lock() {
        *k = keys;
    }
}

/// The modifier keys the 3D mouse's buttons hold down now.
pub fn held_modifiers() -> egui::Modifiers {
    let buttons = shared().lock().map_or(0, |s| s.buttons);
    let Ok(keys) = HELD_KEYS.lock() else { return egui::Modifiers::default() };
    keys.iter().filter(|(b, _)| *b < 32 && buttons & (1 << b) != 0).fold(egui::Modifiers::default(), |m, (_, k)| m | k.modifiers())
}

/// Keys buttons have pressed, for the app's input hook to send in (see [`key_events`]).
static PRESSED_KEYS: Mutex<Vec<(egui::Key, egui::Modifiers)>> = Mutex::new(Vec::new());

/// Press keyboard key `key` with `modifiers`, as a button does.
pub fn press_key(key: egui::Key, modifiers: egui::Modifiers) {
    if let Ok(mut k) = PRESSED_KEYS.lock() {
        k.push((key, modifiers));
    }
    wake();
}

/// The keys buttons have pressed since the last call.
pub fn take_keys() -> Vec<(egui::Key, egui::Modifiers)> {
    PRESSED_KEYS.lock().map(|mut k| std::mem::take(&mut *k)).unwrap_or_default()
}

/// Send the keys buttons pressed (`down`) into a frame's input: each goes down this frame,
/// with its own modifiers held as well as `held` (what the keyboard and the 3D mouse hold),
/// and comes up the next (`up`: the keys still down from the last frame), the modifiers back
/// to `held`. Code that asks whether Shift is held, not just which key went down, sees it.
pub fn key_events(events: &mut Vec<egui::Event>, up: &mut Vec<egui::Key>, down: Vec<(egui::Key, egui::Modifiers)>, held: egui::Modifiers) {
    use egui::Event;
    if !up.is_empty() {
        events.extend(up.drain(..).map(|key| Event::Key { key, physical_key: None, pressed: false, repeat: false, modifiers: held }));
        events.push(Event::ModifiersChanged(held));
    }
    for (key, m) in down {
        if m != egui::Modifiers::NONE {
            events.push(Event::ModifiersChanged(held | m));
        }
        events.push(Event::Key { key, physical_key: None, pressed: true, repeat: false, modifiers: held | m });
        up.push(key);
    }
}

/// Forget which buttons are down: the window lost the focus, and the driver does not report
/// a button let go meanwhile, so a held key would stick.
pub fn release_buttons() {
    if let Ok(mut s) = shared().lock() {
        s.buttons = 0;
    }
}

/// Drop the presses not yet acted on (the Mesh tab, which acts on them, is not shown).
pub fn drop_presses() {
    if let Ok(mut s) = shared().lock() {
        s.pressed.clear();
    }
}

/// Make a frame's input as if the modifier keys the 3D mouse holds (`held`) were held on the
/// keyboard too: added to every key, button and wheel event, and to the held modifiers the
/// keyboard reports (`keyboard`, kept up to date from the events). `last` is what was held
/// last frame; when it changed, the new state is announced before the frame's other events.
pub fn with_held_keys(events: &mut Vec<egui::Event>, keyboard: &mut egui::Modifiers, held: egui::Modifiers, last: egui::Modifiers) {
    use egui::Event;
    for e in events.iter_mut() {
        match e {
            Event::ModifiersChanged(m) => {
                *keyboard = *m;
                *m |= held;
            }
            Event::WindowFocused(false) => *keyboard = egui::Modifiers::default(),
            Event::Key { modifiers, .. } | Event::PointerButton { modifiers, .. } | Event::MouseWheel { modifiers, .. } => *modifiers |= held,
            _ => {}
        }
    }
    if held != last {
        events.insert(0, Event::ModifiersChanged(*keyboard | held));
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
/// about those) to the view frame (x right, y up, z toward the viewer). Both frames are
/// right-handed, so rotations map like the translations: about x as is, about view y
/// (= −device z) negated, about view z (= device y) as is. (Checked on the user's
/// SpaceMouse, 2026-10-17: the earlier, opposite signs turned the model backwards.)
pub fn to_view(axis: [i16; 6]) -> Motion {
    let n = |v: i16| if v.abs() < DEAD_ZONE { 0.0 } else { (v as f32 / FULL_SCALE).clamp(-MAX_DEFLECTION, MAX_DEFLECTION) };
    Motion { t: [n(axis[0]), -n(axis[2]), n(axis[1])], r: [n(axis[3]), -n(axis[5]), n(axis[4])] }
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

/// Move the camera for `dt` seconds of cap deflection, turning about `pivot`. Returns true if
/// anything moved.
pub fn apply(camera: &mut Camera, m: Motion, s: &Settings, dt: f64, viewport_h: f64, pivot: glam::DVec3) -> bool {
    if m.is_zero() {
        return false;
    }
    let sign = |i: usize| if s.invert[i] { -1.0 } else { 1.0 };
    if !s.lock_translation {
        // Object mode: the model follows the cap. Pan in screen points.
        let pan_rate = viewport_h * PAN_RATE * s.pan_speed as f64 * dt;
        let dx = m.t[0] as f64 * sign(0) * pan_rate;
        let dy = -(m.t[1] as f64) * sign(1) * pan_rate;
        if dx != 0.0 || dy != 0.0 {
            camera.pan(dx, dy, viewport_h);
        }
        let z = m.t[2] as f64 * sign(2);
        if z != 0.0 {
            camera.zoom((-ZOOM_RATE * s.zoom_speed as f64 * dt * z).exp());
        }
    }
    if !s.lock_rotation {
        let rate = ORBIT_RATE * s.orbit_speed as f64 * dt;
        camera.rotate_view_about(m.r[0] as f64 * sign(3) * rate, m.r[1] as f64 * sign(4) * rate, m.r[2] as f64 * sign(5) * rate, pivot);
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
        if std::env::var_os("POLYMIX_DEBUG_3D").is_some() {
            let (command, param, value, buttons, b8) = (s.command, s.param, s.value, s.buttons, s.buttons8);
            eprintln!("[3d] command {command} param {param} value {value} buttons {buttons:#x} buttons8 {b8:#x} axis {:?}", { s.axis });
        }
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
        // Rotations follow the same frame change as the translations.
        assert_eq!(to_view([0, 0, 0, 350, 0, 0]).r, [1.0, 0.0, 0.0]);
        assert_eq!(to_view([0, 0, 0, 0, 0, -350]).r, [0.0, 1.0, 0.0]);
        assert_eq!(to_view([0, 0, 0, 0, 350, 0]).r, [0.0, 0.0, 1.0]);
        // Dead zone.
        assert!(to_view([2, -3, 1, 0, 3, -2]).is_zero());
        // The driver's raised sensitivity reports past ±350: kept, up to the glitch guard.
        assert_eq!(to_view([1050, 0, 0, 0, 0, 0]).t[0], 3.0);
        assert_eq!(to_view([i16::MAX, 0, 0, 0, 0, 0]).t[0], MAX_DEFLECTION);
    }

    #[test]
    fn settings_round_trip() {
        let s = Settings { enabled: false, orbit_speed: 7.5, pan_speed: 0.25, zoom_speed: 3.0, invert: [true, false, true, false, false, true], lock_rotation: true, centre: Centre::Object, show_centre: false, ..Default::default() };
        assert_eq!(Settings::from_pref(&s.to_pref()), Some(s));
        // Saved before the centre of rotation was a setting: the default's.
        let before = Settings::from_pref("0 7.5 0.25 3 101001 1 0").unwrap();
        assert_eq!((before.orbit_speed, before.centre, before.show_centre), (7.5, Centre::View, true));
        assert_eq!(Settings::from_pref("1 1 1 1 000000 0 0 sideways 1"), None);
        assert_eq!(Settings::from_pref("garbage"), None);
        assert_eq!(Settings::from_pref("1 1 1 1 0000 0 0"), None);
        // Out of range or not a positive number.
        assert_eq!(Settings::from_pref("1 50 1 1 000000 0 0").map(|s| s.orbit_speed), Some(10.0));
        assert_eq!(Settings::from_pref("1 -2 1 1 000000 0 0"), None);
    }

    #[test]
    fn button_mappings_round_trip_and_default_to_the_standard_buttons() {
        let defaults = default_buttons();
        assert_eq!(buttons_from_pref(&buttons_to_pref(&defaults)), defaults);
        // Every kind of action, and things the preferences may hold that cannot be read.
        let all = vec![
            (0, ButtonAction::Hotbox),
            (1, ButtonAction::Fit),
            (3, ButtonAction::View(5)),
            (6, ButtonAction::Undo),
            (7, ButtonAction::Redo),
            (8, ButtonAction::Accept),
            (9, ButtonAction::Cancel),
            (12, ButtonAction::Tool("erasefill".into())),
            (21, ButtonAction::Key { key: egui::Key::E, modifiers: egui::Modifiers::NONE }),
            (22, ButtonAction::Key { key: egui::Key::I, modifiers: egui::Modifiers::SHIFT }),
            (23, ButtonAction::Key { key: egui::Key::OpenBracket, modifiers: HeldKey::Command.modifiers() | egui::Modifiers::SHIFT }),
            (25, ButtonAction::Hold(HeldKey::Command)),
            (26, ButtonAction::Hold(HeldKey::Shift)),
            (27, ButtonAction::Hold(HeldKey::Option)),
            (28, ButtonAction::Hold(HeldKey::Control)),
            (30, ButtonAction::Nothing),
        ];
        assert_eq!(buttons_from_pref(&buttons_to_pref(&all)), all);
        assert_eq!(buttons_from_pref("x=fit;4=;5=view:abc;6=tool:;7=bogus;8=hold:meta;2=redo;2=undo"), vec![(2, ButtonAction::Redo)], "unreadable pairs and repeats left out");
        assert_eq!(buttons_from_pref(""), vec![]);
        // The defaults are the buttons' old fixed actions: Menu, Fit, Top, Right, Front.
        let views: Vec<&str> = defaults.iter().filter_map(|(_, a)| if let ButtonAction::View(k) = a { Some(crate::app::STANDARD_VIEWS[*k].0) } else { None }).collect();
        assert_eq!(views, ["Top", "Right", "Front"]);
    }

    #[test]
    fn held_buttons_act_as_modifier_keys_in_every_event() {
        use egui::{Event, Key, Modifiers, PointerButton};
        let cmd = HeldKey::Command.modifiers();
        assert!(cmd.command && cmd.mac_cmd == cfg!(target_os = "macos"));
        let click = |m| Event::PointerButton { pos: egui::pos2(1.0, 2.0), button: PointerButton::Secondary, pressed: true, modifiers: m };
        let key = |m| Event::Key { key: Key::Z, physical_key: None, pressed: true, repeat: false, modifiers: m };
        let mut keyboard = Modifiers::default();
        // The button goes down: the new state comes first, then every event carries ⌘.
        let mut events = vec![click(Modifiers::NONE), key(Modifiers::NONE)];
        with_held_keys(&mut events, &mut keyboard, cmd, Modifiers::NONE);
        assert_eq!(events, vec![Event::ModifiersChanged(cmd), click(cmd), key(cmd)]);
        // Held on: Shift pressed on the keyboard meanwhile adds to it, and is remembered.
        let mut events = vec![Event::ModifiersChanged(Modifiers::SHIFT), click(Modifiers::SHIFT)];
        with_held_keys(&mut events, &mut keyboard, cmd, cmd);
        assert_eq!(events, vec![Event::ModifiersChanged(Modifiers::SHIFT | cmd), click(Modifiers::SHIFT | cmd)]);
        assert_eq!(keyboard, Modifiers::SHIFT);
        // The button let go: back to the keyboard's own.
        let mut events = Vec::new();
        with_held_keys(&mut events, &mut keyboard, Modifiers::NONE, cmd);
        assert_eq!(events, vec![Event::ModifiersChanged(Modifiers::SHIFT)]);
        // Nothing held and nothing changed: the events are left as they are.
        let mut events = vec![click(Modifiers::ALT)];
        with_held_keys(&mut events, &mut keyboard, Modifiers::NONE, Modifiers::NONE);
        assert_eq!(events, vec![click(Modifiers::ALT)]);
        // And egui then sees the key held, as the app reads it.
        let ctx = egui::Context::default();
        let mut keyboard = Modifiers::default();
        for (held, last) in [(cmd, Modifiers::NONE), (cmd, cmd)] {
            let mut events = Vec::new();
            with_held_keys(&mut events, &mut keyboard, held, last);
            let _ = ctx.run_ui(egui::RawInput { events, ..Default::default() }, |_| {});
            assert!(ctx.input(|i| i.modifiers.command), "held across frames");
        }
        let mut events = Vec::new();
        with_held_keys(&mut events, &mut keyboard, Modifiers::NONE, cmd);
        let _ = ctx.run_ui(egui::RawInput { events, ..Default::default() }, |_| {});
        assert!(!ctx.input(|i| i.modifiers.command), "let go");
    }

    #[test]
    fn a_button_presses_a_key_with_its_modifiers_for_a_frame() {
        use egui::{Event, Key, Modifiers};
        assert_eq!(key_label(Key::I, Modifiers::SHIFT | HeldKey::Command.modifiers()), "⇧⌘I");
        assert_eq!(key_label(Key::X, Modifiers::NONE), "X");
        let ctx = egui::Context::default();
        let mut up = Vec::new();
        let frame = |events: Vec<Event>| {
            let _ = ctx.run_ui(egui::RawInput { events, ..Default::default() }, |_| {});
            ctx.input(|i| (i.key_pressed(Key::I), i.key_down(Key::I), i.modifiers.shift))
        };
        // Shift + I: down with Shift held, as the app reads it (the keyboard holds nothing)…
        let mut events = Vec::new();
        key_events(&mut events, &mut up, vec![(Key::I, Modifiers::SHIFT)], Modifiers::NONE);
        assert_eq!(frame(events), (true, true, true));
        // …and up the next frame, Shift let go.
        let mut events = Vec::new();
        key_events(&mut events, &mut up, Vec::new(), Modifiers::NONE);
        assert_eq!(frame(events), (false, false, false));
        assert!(up.is_empty());
        // A plain key adds to what is held (here a 3D mouse ⌘), and changes no modifiers.
        let cmd = HeldKey::Command.modifiers();
        let mut events = Vec::new();
        key_events(&mut events, &mut up, vec![(Key::E, Modifiers::NONE)], cmd);
        assert_eq!(events, vec![Event::Key { key: Key::E, physical_key: None, pressed: true, repeat: false, modifiers: cmd }]);
        assert_eq!(up, vec![Key::E]);
    }

    #[test]
    fn earlier_settings_fold_the_overall_speed_in() {
        // enabled, overall 2, pan 1.5, zoom 0.5, rotate 3, invert, lock rotation, lock pan/zoom
        let s = Settings::from_pref("1 2 1.5 0.5 3 100001 0 1").unwrap();
        assert_eq!((s.orbit_speed, s.pan_speed, s.zoom_speed), (6.0, 3.0, 1.0));
        assert_eq!(s.invert, [true, false, false, false, false, true]);
        assert!(!s.lock_rotation && s.lock_translation);
        // The user's saved defaults read back as the defaults.
        assert_eq!(Settings::from_pref("1 1 1 1 1 000000 0 0"), Some(Settings::default()));
    }

    #[test]
    fn each_speed_scales_only_its_own_motion() {
        let pivot = glam::DVec3::ZERO;
        let run = |s: &Settings| {
            let mut c = Camera::default();
            let (t0, d0, f0) = (c.target, c.distance, c.forward());
            apply(&mut c, Motion { t: [0.5, 0.0, 0.0], r: [0.0; 3] }, s, 0.01, 800.0, pivot);
            let pan = c.target.distance(t0);
            let mut c = Camera::default();
            apply(&mut c, Motion { t: [0.0, 0.0, 0.5], r: [0.0; 3] }, s, 0.01, 800.0, pivot);
            let zoom = (d0 / c.distance).ln();
            let mut c = Camera::default();
            apply(&mut c, Motion { t: [0.0; 3], r: [0.0, 0.5, 0.0] }, s, 0.01, 800.0, pivot);
            let orbit = c.forward().angle_between(f0);
            (orbit, pan, zoom)
        };
        let (orbit, pan, zoom) = run(&Settings::default());
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-6 * b.abs().max(1e-12);
        // Half deflection for 10 ms at speed 1.
        assert!(close(orbit, ORBIT_RATE * 0.5 * 0.01), "{orbit}");
        assert!(close(zoom, ZOOM_RATE * 0.5 * 0.01), "{zoom}");
        let fast_orbit = run(&Settings { orbit_speed: 4.0, ..Default::default() });
        assert!(close(fast_orbit.0, 4.0 * orbit) && close(fast_orbit.1, pan) && close(fast_orbit.2, zoom), "{fast_orbit:?}");
        let fast_pan = run(&Settings { pan_speed: 4.0, ..Default::default() });
        assert!(close(fast_pan.0, orbit) && close(fast_pan.1, 4.0 * pan) && close(fast_pan.2, zoom), "{fast_pan:?}");
        let fast_zoom = run(&Settings { zoom_speed: 4.0, ..Default::default() });
        assert!(close(fast_zoom.0, orbit) && close(fast_zoom.1, pan) && close(fast_zoom.2, 4.0 * zoom), "{fast_zoom:?}");
    }

    #[test]
    fn motion_moves_the_camera() {
        let mut c = Camera::default();
        let s = Settings::default();
        let t0 = c.target;
        assert!(apply(&mut c, Motion { t: [1.0, 0.0, 0.0], r: [0.0; 3] }, &s, 0.1, 800.0, glam::DVec3::ZERO));
        assert!(c.target.distance(t0) > 0.0);
        let d0 = c.distance;
        apply(&mut c, Motion { t: [0.0, 0.0, 1.0], r: [0.0; 3] }, &s, 0.1, 800.0, glam::DVec3::ZERO);
        assert!(c.distance < d0, "pulling the cap toward you zooms in");
        let f0 = c.forward();
        apply(&mut c, Motion { t: [0.0; 3], r: [0.0, 1.0, 0.0] }, &s, 0.1, 800.0, glam::DVec3::ZERO);
        assert!(c.forward().dot(f0) < 0.9999);
        let locked = Settings { lock_rotation: true, lock_translation: true, ..Default::default() };
        let before = (c.target, c.distance, c.forward());
        apply(&mut c, Motion { t: [1.0; 3], r: [1.0; 3] }, &locked, 0.1, 800.0, glam::DVec3::ZERO);
        assert_eq!(before, (c.target, c.distance, c.forward()));
    }
}
