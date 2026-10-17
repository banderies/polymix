//! Small shared UI helpers and theme.

use egui::{Color32, RichText, Ui};

pub const ACCENT: Color32 = Color32::from_rgb(255, 140, 40);

pub fn apply_theme(ctx: &egui::Context) {
    ctx.set_theme(egui::Theme::Dark);
    let mut style = (*ctx.global_style()).clone();
    style.spacing.item_spacing = egui::vec2(6.0, 5.0);
    style.spacing.button_padding = egui::vec2(8.0, 3.0);
    style.visuals.selection.bg_fill = Color32::from_rgb(180, 90, 20);
    style.visuals.hyperlink_color = ACCENT;
    style.visuals.panel_fill = Color32::from_rgb(38, 39, 43);
    style.visuals.window_fill = Color32::from_rgb(44, 45, 50);
    ctx.set_global_style(style);
}

pub fn section(ui: &mut Ui, title: &str) {
    ui.add_space(4.0);
    ui.label(RichText::new(title).strong().color(Color32::from_gray(220)));
    ui.separator();
}

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

pub fn format_count(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, ch) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(ch);
    }
    out
}
