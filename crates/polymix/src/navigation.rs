//! Mouse navigation schemes, as in Meshmixer's Preferences → Navigation Mode, plus a Custom
//! scheme where the user assigns Orbit, Pan or Zoom to each middle and right drag.
//!
//! * **Fusion 360** (Meshmixer's default): middle-drag pans, Shift+middle-drag tumbles, the
//!   wheel zooms.
//! * **Meshmixer** (Maya-like): right-drag or Alt+left-drag tumbles; Alt+middle-drag,
//!   Shift+right-drag or Alt+Shift+left-drag pans; Alt+right-drag zooms.
//! * **Custom**: the user's own [`NavBindings`] table (Preferences → General).
//!
//! Every mode also accepts the pen-friendly Alt+left-drag (tumble), Alt+Shift+left-drag (pan)
//! and Alt+Cmd/Ctrl+left-drag (zoom), and a plain right-drag always navigates, so a trackpad
//! or pen can always reach every action.

use egui::{Modifiers, PointerButton};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NavMode {
    Fusion360,
    Meshmixer,
    Custom,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NavAction {
    Tumble,
    Pan,
    Zoom,
}

impl NavMode {
    pub const ALL: [NavMode; 3] = [NavMode::Fusion360, NavMode::Meshmixer, NavMode::Custom];

    pub fn label(self) -> &'static str {
        match self {
            NavMode::Fusion360 => "Fusion 360",
            NavMode::Meshmixer => "Meshmixer",
            NavMode::Custom => "Custom",
        }
    }

    pub fn key(self) -> &'static str {
        match self {
            NavMode::Fusion360 => "fusion360",
            NavMode::Meshmixer => "meshmixer",
            NavMode::Custom => "custom",
        }
    }

    pub fn from_key(k: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|m| m.key() == k)
    }

    /// The drag table this mode uses; `custom` is the user's own (used by [`NavMode::Custom`]).
    pub fn bindings(self, custom: NavBindings) -> NavBindings {
        match self {
            NavMode::Fusion360 => NavBindings::FUSION360,
            NavMode::Meshmixer => NavBindings::MESHMIXER,
            NavMode::Custom => custom,
        }
    }
}

impl NavAction {
    pub const ALL: [NavAction; 3] = [NavAction::Tumble, NavAction::Pan, NavAction::Zoom];

    pub fn label(self) -> &'static str {
        match self {
            NavAction::Tumble => "Orbit",
            NavAction::Pan => "Pan",
            NavAction::Zoom => "Zoom",
        }
    }

    fn key(self) -> &'static str {
        match self {
            NavAction::Tumble => "orbit",
            NavAction::Pan => "pan",
            NavAction::Zoom => "zoom",
        }
    }

    fn from_key(k: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|a| a.key() == k)
    }
}

/// Column headers of a [`NavBindings`] row: a plain drag, Shift+drag and Alt+drag.
pub const NAV_MODIFIERS: [&str; 3] = ["Drag", "Shift+drag", "Alt+drag"];

/// What a middle or right drag does, per modifier ([`NAV_MODIFIERS`] order). With Shift and
/// Alt both held Shift wins; Cmd/Ctrl uses the plain drag (tools take Cmd+drags first).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NavBindings {
    pub middle: [NavAction; 3],
    pub right: [NavAction; 3],
}

impl NavBindings {
    pub const FUSION360: Self = Self {
        middle: [NavAction::Pan, NavAction::Tumble, NavAction::Pan],
        right: [NavAction::Tumble, NavAction::Pan, NavAction::Tumble],
    };
    pub const MESHMIXER: Self = Self {
        middle: [NavAction::Pan, NavAction::Pan, NavAction::Pan],
        right: [NavAction::Tumble, NavAction::Pan, NavAction::Zoom],
    };

    /// Whether a drag with `button` starting with these modifiers navigates the camera
    /// (otherwise it goes to the active tool).
    pub fn starts_navigation(self, button: PointerButton, mods: Modifiers) -> bool {
        match button {
            PointerButton::Primary => mods.alt,
            PointerButton::Secondary | PointerButton::Middle => true,
            _ => false,
        }
    }

    /// The camera action for a navigation drag, from the button and the modifiers held now
    /// (changing modifiers mid-drag switches action, as in Meshmixer).
    pub fn drag_action(self, button: PointerButton, mods: Modifiers) -> Option<NavAction> {
        use NavAction::*;
        let column = if mods.shift {
            1
        } else if mods.alt {
            2
        } else {
            0
        };
        match button {
            // Pen-friendly bindings, identical in every mode.
            PointerButton::Primary if mods.alt && mods.command => Some(Zoom),
            PointerButton::Primary if mods.alt && mods.shift => Some(Pan),
            PointerButton::Primary if mods.alt => Some(Tumble),
            PointerButton::Primary => None,
            PointerButton::Middle => Some(self.middle[column]),
            PointerButton::Secondary => Some(self.right[column]),
            _ => None,
        }
    }

    /// Rows for the Controls & hotkeys window: every way to reach each action. A modified
    /// drag that does the same as the plain drag of its button is left out.
    pub fn help_rows(self) -> Vec<(String, &'static str)> {
        NavAction::ALL
            .into_iter()
            .map(|a| {
                let mut ways: Vec<String> = Vec::new();
                if a == NavAction::Zoom {
                    ways.push("scroll / pinch".into());
                }
                for (button, row) in [("middle", &self.middle), ("right", &self.right)] {
                    for (i, &b) in row.iter().enumerate() {
                        if b == a && (i == 0 || row[0] != a) {
                            ways.push(format!("{}{button}-drag", ["", "Shift+", "Alt+"][i]));
                        }
                    }
                }
                ways.push(
                    match a {
                        NavAction::Tumble => "Alt+left-drag",
                        NavAction::Pan => "Alt+Shift+left-drag",
                        NavAction::Zoom => "Alt+Cmd+left-drag",
                    }
                    .into(),
                );
                let ways = ways.join(" / ");
                let mut chars = ways.chars();
                let ways = chars.next().map(|c| c.to_uppercase().chain(chars).collect()).unwrap_or_default();
                (ways, if a == NavAction::Tumble { "Orbit (tumble)" } else { a.label() })
            })
            .collect()
    }

    /// Preference value, e.g. `pan,orbit,pan;orbit,pan,orbit` (middle row; right row).
    pub fn to_pref(self) -> String {
        let row = |r: &[NavAction; 3]| r.map(|a| a.key()).join(",");
        format!("{};{}", row(&self.middle), row(&self.right))
    }

    pub fn from_pref(s: &str) -> Option<Self> {
        let row = |r: &str| -> Option<[NavAction; 3]> {
            let v: Vec<NavAction> = r.split(',').map(NavAction::from_key).collect::<Option<_>>()?;
            v.try_into().ok()
        };
        let (middle, right) = s.split_once(';')?;
        Some(Self { middle: row(middle)?, right: row(right)? })
    }
}

/// The mouse-button table in Preferences: one dropdown per middle / right drag and modifier.
/// Editing a preset's table switches to Custom, starting from that preset.
pub fn bindings_ui(ui: &mut egui::Ui, mode: &mut NavMode, custom: &mut NavBindings) {
    let before = mode.bindings(*custom);
    let mut b = before;
    egui::Grid::new("nav_bindings").num_columns(4).spacing([8.0, 4.0]).show(ui, |ui| {
        ui.label("");
        for m in NAV_MODIFIERS {
            ui.label(m);
        }
        ui.end_row();
        for (name, row) in [("Middle", &mut b.middle), ("Right", &mut b.right)] {
            ui.label(name);
            for (i, cell) in row.iter_mut().enumerate() {
                egui::ComboBox::from_id_salt(("nav_binding", name, i)).width(64.0).selected_text(cell.label()).show_ui(ui, |ui| {
                    for a in NavAction::ALL {
                        ui.selectable_value(cell, a, a.label());
                    }
                });
            }
            ui.end_row();
        }
    });
    if b != before {
        *custom = b;
        *mode = NavMode::Custom;
    }
}

/// The modifier that turns the scroll wheel into brush size (the plain wheel always zooms).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrushScroll {
    Shift,
    Ctrl,
    Alt,
    Cmd,
}

impl BrushScroll {
    pub const ALL: [BrushScroll; 4] = [BrushScroll::Shift, BrushScroll::Ctrl, BrushScroll::Alt, BrushScroll::Cmd];

    pub fn label(self) -> &'static str {
        match self {
            BrushScroll::Shift => "Shift",
            BrushScroll::Ctrl => "Control",
            BrushScroll::Alt => "Option",
            BrushScroll::Cmd => "Command",
        }
    }

    /// How hints write it, e.g. "Option+scroll".
    pub fn scroll_label(self) -> &'static str {
        match self {
            BrushScroll::Shift => "Shift+scroll",
            BrushScroll::Ctrl => "Ctrl+scroll",
            BrushScroll::Alt => "Option+scroll",
            BrushScroll::Cmd => "Cmd+scroll",
        }
    }

    pub fn key(self) -> &'static str {
        match self {
            BrushScroll::Shift => "shift",
            BrushScroll::Ctrl => "ctrl",
            BrushScroll::Alt => "alt",
            BrushScroll::Cmd => "cmd",
        }
    }

    pub fn from_key(k: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|m| m.key() == k)
    }

    pub fn held(self, mods: Modifiers) -> bool {
        match self {
            BrushScroll::Shift => mods.shift,
            BrushScroll::Ctrl => mods.ctrl,
            BrushScroll::Alt => mods.alt,
            BrushScroll::Cmd => mods.command,
        }
    }
}

/// Brush-size change from this frame's raw wheel events, in the units tools take
/// ([`crate::tools::Tool::wheel`], points). egui eases wheel notches in over several frames
/// (which lags) and macOS accelerates them (a quick flick reports far more per notch than a
/// slow one), so for the brush each mouse-wheel notch is one fixed step, applied at once;
/// trackpad scrolling stays continuous. With Shift held macOS turns the wheel horizontal, so
/// either axis counts.
pub fn brush_wheel_delta(events: &[egui::Event]) -> f32 {
    /// One notch: about 10% of the brush size.
    const NOTCH: f32 = 25.0;
    events
        .iter()
        .map(|e| match e {
            egui::Event::MouseWheel { unit, delta, .. } => {
                let d = if delta.y != 0.0 { delta.y } else { delta.x };
                match unit {
                    egui::MouseWheelUnit::Point => d,
                    _ if d == 0.0 => 0.0,
                    _ => NOTCH * d.signum(),
                }
            }
            _ => 0.0,
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use NavAction::*;

    fn m(alt: bool, shift: bool, command: bool) -> Modifiers {
        Modifiers { alt, shift, command, mac_cmd: command, ..Default::default() }
    }

    #[test]
    fn fusion_bindings() {
        let f = NavBindings::FUSION360;
        assert_eq!(f.drag_action(PointerButton::Middle, m(false, false, false)), Some(Pan));
        assert_eq!(f.drag_action(PointerButton::Middle, m(false, true, false)), Some(Tumble));
        assert_eq!(f.drag_action(PointerButton::Secondary, m(false, false, false)), Some(Tumble));
        assert_eq!(f.drag_action(PointerButton::Primary, m(false, false, false)), None);
    }

    #[test]
    fn meshmixer_bindings() {
        let mm = NavBindings::MESHMIXER;
        assert_eq!(mm.drag_action(PointerButton::Secondary, m(false, false, false)), Some(Tumble));
        assert_eq!(mm.drag_action(PointerButton::Secondary, m(true, false, false)), Some(Zoom));
        assert_eq!(mm.drag_action(PointerButton::Secondary, m(false, true, false)), Some(Pan));
        assert_eq!(mm.drag_action(PointerButton::Middle, m(true, false, false)), Some(Pan));
        assert_eq!(mm.drag_action(PointerButton::Middle, m(false, true, false)), Some(Pan));
    }

    #[test]
    fn custom_bindings() {
        // Orbit with the middle button, pan with the right.
        let mut custom = NavBindings::FUSION360;
        custom.middle[0] = Tumble;
        custom.right[0] = Pan;
        let c = NavMode::Custom.bindings(custom);
        assert_eq!(c.drag_action(PointerButton::Middle, m(false, false, false)), Some(Tumble));
        assert_eq!(c.drag_action(PointerButton::Secondary, m(false, false, false)), Some(Pan));
        // Cmd falls back to the plain drag; Shift beats Alt.
        assert_eq!(c.drag_action(PointerButton::Secondary, m(false, false, true)), Some(Pan));
        assert_eq!(c.drag_action(PointerButton::Middle, m(true, true, false)), Some(c.middle[1]));
        assert_eq!(NavMode::Fusion360.bindings(custom), NavBindings::FUSION360);
    }

    #[test]
    fn bindings_pref_round_trip() {
        let b = NavBindings { middle: [Tumble, Zoom, Pan], right: [Pan, Tumble, Zoom] };
        assert_eq!(b.to_pref(), "orbit,zoom,pan;pan,orbit,zoom");
        assert_eq!(NavBindings::from_pref(&b.to_pref()), Some(b));
        assert_eq!(NavBindings::from_pref("orbit,pan;pan,pan,pan"), None);
        assert_eq!(NavBindings::from_pref("orbit,pan,spin;pan,pan,pan"), None);
        assert_eq!(NavBindings::from_pref(""), None);
    }

    #[test]
    fn help_rows_list_each_way() {
        let rows = NavBindings::FUSION360.help_rows();
        assert_eq!(rows[0].0, "Shift+middle-drag / right-drag / Alt+left-drag");
        assert_eq!(rows[1].0, "Middle-drag / Shift+right-drag / Alt+Shift+left-drag");
        assert_eq!(rows[2].0, "Scroll / pinch / Alt+Cmd+left-drag");
        let rows = NavBindings::MESHMIXER.help_rows();
        assert_eq!(rows[0].0, "Right-drag / Alt+left-drag");
        assert_eq!(rows[2].0, "Scroll / pinch / Alt+right-drag / Alt+Cmd+left-drag");
    }

    #[test]
    fn brush_scroll_modifiers() {
        for b in BrushScroll::ALL {
            assert_eq!(BrushScroll::from_key(b.key()), Some(b));
        }
        let ctrl = Modifiers { ctrl: true, ..Default::default() };
        assert!(BrushScroll::Ctrl.held(ctrl));
        assert!(!BrushScroll::Shift.held(ctrl));
        assert!(BrushScroll::Alt.held(m(true, false, false)));
        assert!(BrushScroll::Cmd.held(m(false, false, true)));
    }

    #[test]
    fn brush_wheel_steps_are_per_notch() {
        let wheel = |unit, x: f32, y: f32| egui::Event::MouseWheel {
            unit,
            delta: egui::vec2(x, y),
            phase: egui::TouchPhase::Move,
            source: egui::MouseWheelSource::Wheel,
            modifiers: Default::default(),
        };
        use egui::MouseWheelUnit::{Line, Point};
        // Accelerated notches (1, 4.6 lines) count the same; Shift's horizontal wheel too.
        assert_eq!(brush_wheel_delta(&[wheel(Line, 0.0, 1.0)]), brush_wheel_delta(&[wheel(Line, 0.0, 4.6)]));
        assert_eq!(brush_wheel_delta(&[wheel(Line, -2.0, 0.0)]), -brush_wheel_delta(&[wheel(Line, 0.0, 1.0)]));
        assert_eq!(brush_wheel_delta(&[wheel(Line, 0.0, 1.0), wheel(Line, 0.0, 3.0)]), 2.0 * brush_wheel_delta(&[wheel(Line, 0.0, 1.0)]));
        // Trackpads stay continuous.
        assert_eq!(brush_wheel_delta(&[wheel(Point, 0.0, 3.5)]), 3.5);
        assert_eq!(brush_wheel_delta(&[egui::Event::Copy]), 0.0);
    }

    #[test]
    fn pen_bindings_in_every_mode() {
        for mode in NavMode::ALL {
            let b = mode.bindings(NavBindings { middle: [Zoom; 3], right: [Zoom; 3] });
            assert_eq!(b.drag_action(PointerButton::Primary, m(true, false, false)), Some(Tumble));
            assert_eq!(b.drag_action(PointerButton::Primary, m(true, true, false)), Some(Pan));
            assert_eq!(b.drag_action(PointerButton::Primary, m(true, false, true)), Some(Zoom));
            assert!(b.starts_navigation(PointerButton::Primary, m(true, false, false)));
            assert!(!b.starts_navigation(PointerButton::Primary, m(false, false, false)));
            assert_eq!(NavMode::from_key(mode.key()), Some(mode));
        }
    }
}
