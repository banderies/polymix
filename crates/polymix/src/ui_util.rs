//! Small shared UI helpers and theme.

use egui::{Color32, RichText, Ui};

pub use polymix_workspace::style::{apply_theme, format_count, section, Theme, ACCENT};

/// Accept / Cancel row at the bottom of tool panels. Returns (accept, cancel).
pub fn accept_cancel(ui: &mut Ui, can_accept: bool) -> (bool, bool) {
    ui.add_space(8.0);
    ui.horizontal(|ui| {
        let a = ui
            .add_enabled(can_accept, egui::Button::new(RichText::new("Accept").strong()).fill(Color32::from_rgb(70, 110, 60)))
            .on_hover_text("Enter")
            .clicked();
        let c = ui.button("Cancel").on_hover_text("Esc").clicked();
        (a, c)
    })
    .inner
}

/// A small eye toggle: open (solid) or shut (see-through).
pub fn eye_toggle(ui: &mut Ui, open: bool) -> egui::Response {
    let (rect, r) = ui.allocate_exact_size(egui::vec2(18.0, 16.0), egui::Sense::click());
    let col = if !open {
        ACCENT
    } else if r.hovered() {
        ui.visuals().text_color()
    } else {
        ui.visuals().weak_text_color()
    };
    let p = ui.painter();
    let c = rect.center();
    let stroke = egui::Stroke::new(1.4, col);
    // The eye's outline: two arcs meeting at the corners.
    let arc = |up: f32| -> Vec<egui::Pos2> { (0..=12).map(|i| { let t = i as f32 / 12.0; let x = (t - 0.5) * 14.0; let y = up * 5.0 * (1.0 - (2.0 * t - 1.0).powi(2)); c + egui::vec2(x, -y) }).collect() };
    p.add(egui::Shape::line(arc(1.0), stroke));
    p.add(egui::Shape::line(arc(-1.0), stroke));
    if open {
        p.circle_filled(c, 2.4, col);
    } else {
        p.circle_stroke(c, 2.4, egui::Stroke::new(1.0, col));
        p.line_segment([c + egui::vec2(-7.0, 6.0), c + egui::vec2(7.0, -6.0)], stroke);
    }
    r
}
