//! Mouse navigation schemes, as in Meshmixer's Preferences → Navigation Mode.
//!
//! * **Fusion 360** (Meshmixer's default): middle-drag pans, Shift+middle-drag tumbles, the
//!   wheel zooms.
//! * **Meshmixer** (Maya-like): right-drag or Alt+left-drag tumbles; Alt+middle-drag,
//!   Shift+right-drag or Alt+Shift+left-drag pans; Alt+right-drag zooms.
//!
//! Both modes also accept the pen-friendly Alt+left-drag (tumble), Alt+Shift+left-drag (pan)
//! and Alt+Cmd/Ctrl+left-drag (zoom), and right-drag tumbles in both, so a trackpad or pen
//! can always reach every action.

use egui::{Modifiers, PointerButton};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NavMode {
    Fusion360,
    Meshmixer,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NavAction {
    Tumble,
    Pan,
    Zoom,
}

impl NavMode {
    pub const ALL: [NavMode; 2] = [NavMode::Fusion360, NavMode::Meshmixer];

    pub fn label(self) -> &'static str {
        match self {
            NavMode::Fusion360 => "Fusion 360",
            NavMode::Meshmixer => "Meshmixer",
        }
    }

    pub fn key(self) -> &'static str {
        match self {
            NavMode::Fusion360 => "fusion360",
            NavMode::Meshmixer => "meshmixer",
        }
    }

    pub fn from_key(k: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|m| m.key() == k)
    }

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
        match button {
            // Pen-friendly bindings, identical in both modes.
            PointerButton::Primary if mods.alt && mods.command => Some(Zoom),
            PointerButton::Primary if mods.alt && mods.shift => Some(Pan),
            PointerButton::Primary if mods.alt => Some(Tumble),
            PointerButton::Primary => None,
            PointerButton::Middle => match self {
                NavMode::Fusion360 if mods.shift => Some(Tumble),
                _ => Some(Pan),
            },
            PointerButton::Secondary if mods.shift => Some(Pan),
            PointerButton::Secondary if mods.alt && self == NavMode::Meshmixer => Some(Zoom),
            PointerButton::Secondary => Some(Tumble),
            _ => None,
        }
    }

    /// Rows for the Controls & hotkeys window.
    pub fn help_rows(self) -> [(&'static str, &'static str); 3] {
        match self {
            NavMode::Fusion360 => [
                ("Shift+middle-drag / right-drag / Alt+left-drag", "Tumble (orbit)"),
                ("Middle-drag / Shift+right-drag / Alt+Shift+left-drag", "Pan"),
                ("Scroll / pinch / Alt+Cmd+left-drag", "Zoom"),
            ],
            NavMode::Meshmixer => [
                ("Right-drag / Alt+left-drag", "Tumble (orbit)"),
                ("Middle-drag / Shift+right-drag / Alt+Shift+left-drag", "Pan"),
                ("Scroll / pinch / Alt+right-drag / Alt+Cmd+left-drag", "Zoom"),
            ],
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
        let f = NavMode::Fusion360;
        assert_eq!(f.drag_action(PointerButton::Middle, m(false, false, false)), Some(Pan));
        assert_eq!(f.drag_action(PointerButton::Middle, m(false, true, false)), Some(Tumble));
        assert_eq!(f.drag_action(PointerButton::Secondary, m(false, false, false)), Some(Tumble));
        assert_eq!(f.drag_action(PointerButton::Primary, m(false, false, false)), None);
    }

    #[test]
    fn meshmixer_bindings() {
        let mm = NavMode::Meshmixer;
        assert_eq!(mm.drag_action(PointerButton::Secondary, m(false, false, false)), Some(Tumble));
        assert_eq!(mm.drag_action(PointerButton::Secondary, m(true, false, false)), Some(Zoom));
        assert_eq!(mm.drag_action(PointerButton::Secondary, m(false, true, false)), Some(Pan));
        assert_eq!(mm.drag_action(PointerButton::Middle, m(true, false, false)), Some(Pan));
        assert_eq!(mm.drag_action(PointerButton::Middle, m(false, true, false)), Some(Pan));
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
    fn pen_bindings_in_both_modes() {
        for mode in NavMode::ALL {
            assert_eq!(mode.drag_action(PointerButton::Primary, m(true, false, false)), Some(Tumble));
            assert_eq!(mode.drag_action(PointerButton::Primary, m(true, true, false)), Some(Pan));
            assert_eq!(mode.drag_action(PointerButton::Primary, m(true, false, true)), Some(Zoom));
            assert!(mode.starts_navigation(PointerButton::Primary, m(true, false, false)));
            assert!(!mode.starts_navigation(PointerButton::Primary, m(false, false, false)));
            assert_eq!(NavMode::from_key(mode.key()), Some(mode));
        }
    }
}
