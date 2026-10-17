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
