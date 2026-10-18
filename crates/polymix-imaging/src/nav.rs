//! The mouse controls every image view shares:
//!
//! | Mouse | Does |
//! |---|---|
//! | left drag | scroll through slices (up = next) |
//! | right drag | pan (the image follows the cursor) |
//! | middle drag | zoom (up = in, about where the drag started) |
//! | left + right drag | window (across) and level (up / down) |
//! | wheel | scroll through slices, one per notch; ⌘ / Ctrl + wheel or pinch zooms |
//! | left click | point at a spot (e.g. move the crosshair) |
//! | double-click | the view fills the layout, and back (a view on its own: back to the automatic view) |
//!
//! The wheel is captured while the pointer is over the view, so a scrollable panel around it
//! does not scroll too.

use egui::{PointerButton, Pos2, Rect, Response, Vec2};

/// Points of vertical drag per slice.
const DRAG_PER_SLICE: f32 = 6.0;
/// Points of smooth (trackpad) scrolling per slice.
const SCROLL_PER_SLICE: f32 = 24.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Gesture {
    Slices,
    Pan,
    Zoom,
    Window,
}

/// Zoom and pan of one image view, and the state of the gesture in progress.
#[derive(Clone, Debug)]
pub struct Nav {
    /// Magnification over fitting the whole image in the view.
    pub zoom: f32,
    /// Offset of the image centre from the view centre, in points.
    pub pan: Vec2,
    gesture: Option<Gesture>,
    /// Drag and scroll not yet turned into whole slices.
    slice_drag: f32,
    wheel: f32,
    /// Slices moved by the current left drag (undone if it turns into window / level).
    gesture_slices: i32,
}

impl Default for Nav {
    fn default() -> Self {
        Self { zoom: 1.0, pan: Vec2::ZERO, gesture: None, slice_drag: 0.0, wheel: 0.0, gesture_slices: 0 }
    }
}

/// What the user asked for this frame.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct NavEvents {
    /// Slices to move (positive = towards the next slice).
    pub slices: i32,
    /// Window / level drag in points: x widens the window, y (down) lowers the level.
    pub window: Option<Vec2>,
    /// A left click (no drag), where.
    pub clicked: Option<Pos2>,
    pub double_clicked: bool,
    /// Where the pointer is over the view.
    pub hover: Option<Pos2>,
    /// Zoom or pan changed.
    pub moved: bool,
}

impl Nav {
    /// Where an image of physical size `content` (any unit; only the aspect ratio matters)
    /// is drawn in `view`.
    pub fn image_rect(&self, view: Rect, content: Vec2) -> Rect {
        let fit = (view.width() / content.x.max(1e-6)).min(view.height() / content.y.max(1e-6));
        Rect::from_center_size(view.center() + self.pan, content * fit * self.zoom)
    }

    /// A window / level drag is in progress.
    pub fn windowing(&self) -> bool {
        self.gesture == Some(Gesture::Window)
    }

    pub fn reset(&mut self) {
        self.zoom = 1.0;
        self.pan = Vec2::ZERO;
    }

    /// Zoom by `f` keeping the point `at` (screen) fixed.
    fn zoom_about(&mut self, view: Rect, at: Pos2, f: f32) {
        let f = f.clamp(0.01, 100.0);
        let new_zoom = (self.zoom * f).clamp(0.2, 50.0);
        let f = new_zoom / self.zoom;
        let c = view.center() + self.pan;
        let c2 = at - (at - c) * f;
        self.pan = c2 - view.center();
        self.zoom = new_zoom;
    }

    /// Apply the controls to `r` (a view sensing clicks and drags). Call once per frame.
    pub fn handle(&mut self, ui: &egui::Ui, r: &Response) -> NavEvents {
        let mut ev = NavEvents { hover: r.hover_pos(), ..Default::default() };
        let view = r.rect;
        let (primary, secondary, middle, mut delta, press, pos) = ui.input(|i| (i.pointer.primary_down(), i.pointer.secondary_down(), i.pointer.middle_down(), i.pointer.delta(), i.pointer.press_origin(), i.pointer.latest_pos()));
        let active = r.is_pointer_button_down_on() || r.dragged();
        if !(primary || secondary || middle) {
            self.gesture = None;
            self.slice_drag = 0.0;
            self.gesture_slices = 0;
        } else if active {
            if primary && secondary {
                if self.gesture != Some(Gesture::Window) {
                    // Undo the slices scrolled before the second button went down.
                    ev.slices -= self.gesture_slices;
                    self.gesture_slices = 0;
                    self.gesture = Some(Gesture::Window);
                }
            } else if self.gesture.is_none() && r.dragged() {
                // The drag starts once the pointer has moved a little: count that movement
                // too, so the grabbed point follows the cursor exactly.
                if let (Some(a), Some(b)) = (press, pos) {
                    delta = b - a;
                }
                self.gesture = if r.dragged_by(PointerButton::Primary) {
                    Some(Gesture::Slices)
                } else if r.dragged_by(PointerButton::Secondary) {
                    Some(Gesture::Pan)
                } else if r.dragged_by(PointerButton::Middle) {
                    Some(Gesture::Zoom)
                } else {
                    None
                };
            }
            match self.gesture {
                Some(Gesture::Window) => {
                    if delta != Vec2::ZERO {
                        ev.window = Some(delta);
                    }
                }
                Some(Gesture::Slices) => {
                    self.slice_drag -= delta.y;
                    let n = (self.slice_drag / DRAG_PER_SLICE).trunc() as i32;
                    self.slice_drag -= n as f32 * DRAG_PER_SLICE;
                    ev.slices += n;
                    self.gesture_slices += n;
                }
                Some(Gesture::Pan) => {
                    self.pan += delta;
                    ev.moved |= delta != Vec2::ZERO;
                }
                Some(Gesture::Zoom) if delta.y != 0.0 => {
                    self.zoom_about(view, press.unwrap_or(view.center()), (-delta.y * 0.01).exp());
                    ev.moved = true;
                }
                _ => {}
            }
        }
        if r.clicked() {
            ev.clicked = r.interact_pointer_pos();
        }
        if r.double_clicked() {
            ev.double_clicked = true;
        }
        if let Some(at) = ev.hover {
            // The wheel belongs to this view while the pointer is over it.
            let (zoom, events, zoom_modifier) = ui.input(|i| (i.zoom_delta(), i.raw.events.clone(), i.modifiers.command || i.modifiers.ctrl));
            if zoom != 1.0 {
                self.zoom_about(view, at, zoom);
                ev.moved = true;
            } else if !zoom_modifier {
                for e in &events {
                    if let egui::Event::MouseWheel { unit, delta, .. } = e {
                        // Only vertical scrolling moves slices (the signum of a sideways
                        // scroll's 0 is 1).
                        if delta.y == 0.0 {
                            continue;
                        }
                        match unit {
                            egui::MouseWheelUnit::Point => self.wheel += delta.y,
                            _ => self.wheel += delta.y.signum() * SCROLL_PER_SLICE * delta.y.abs().round().max(1.0),
                        }
                    }
                }
                let n = (self.wheel / SCROLL_PER_SLICE).trunc() as i32;
                self.wheel -= n as f32 * SCROLL_PER_SLICE;
                ev.slices += n;
            }
            ui.ctx().input_mut(|i| i.smooth_scroll_delta = Vec2::ZERO);
        } else {
            self.wheel = 0.0;
        }
        ev
    }
}

/// Apply a window / level drag to (centre, width), scaled to the width.
pub fn drag_window(window: (f64, f64), d: Vec2) -> (f64, f64) {
    let (c, w) = window;
    let w2 = (w * (d.x as f64 * 0.01).exp()).max(1e-6);
    let c2 = c - d.y as f64 * w * 0.004;
    (c2, w2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zoom_keeps_the_anchor_fixed() {
        let view = Rect::from_min_size(Pos2::ZERO, Vec2::new(200.0, 100.0));
        let mut n = Nav::default();
        let content = Vec2::new(4.0, 2.0);
        let at = Pos2::new(150.0, 30.0);
        let before = n.image_rect(view, content);
        let u = (at - before.min) / before.size();
        n.zoom_about(view, at, 2.0);
        let after = n.image_rect(view, content);
        let p = after.min + u * after.size();
        assert!((p - at).length() < 1e-3, "{p:?}");
        assert!((after.width() - 2.0 * before.width()).abs() < 1e-3);
        n.reset();
        assert_eq!(n.image_rect(view, content), before);
    }

    #[test]
    fn window_drag() {
        let (c, w) = drag_window((40.0, 400.0), Vec2::new(0.0, 10.0));
        assert!(c < 40.0 && w == 400.0);
        let (_, w) = drag_window((40.0, 400.0), Vec2::new(20.0, 0.0));
        assert!(w > 400.0);
    }
}
