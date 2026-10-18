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
