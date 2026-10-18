//! The look shared by every workspace: theme, accent colour, section headers, number
//! formatting.

use egui::{Color32, RichText, Ui};

pub const ACCENT: Color32 = Color32::from_rgb(255, 140, 40);

/// UI colour theme.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Theme {
    Dark,
    /// Warm light greys in the style of Meshmixer's panels.
    Light,
}

impl Theme {
    pub fn key(self) -> &'static str {
        match self {
            Theme::Dark => "dark",
            Theme::Light => "light",
        }
    }
    pub fn from_key(k: &str) -> Option<Self> {
        [Theme::Dark, Theme::Light].into_iter().find(|t| t.key() == k)
    }
}

pub fn apply_theme(ctx: &egui::Context, theme: Theme) {
    let rgb = |r, g, b| Color32::from_rgb(r, g, b);
    ctx.set_theme(match theme {
        Theme::Dark => egui::Theme::Dark,
        Theme::Light => egui::Theme::Light,
    });
    let mut style = (*ctx.global_style()).clone();
    style.spacing.item_spacing = egui::vec2(6.0, 5.0);
    style.spacing.button_padding = egui::vec2(8.0, 3.0);
    style.visuals = match theme {
        Theme::Dark => egui::Visuals::dark(),
        Theme::Light => egui::Visuals::light(),
    };
    style.visuals.hyperlink_color = ACCENT;
    match theme {
        Theme::Dark => {
            style.visuals.selection.bg_fill = rgb(180, 90, 20);
            style.visuals.panel_fill = rgb(38, 39, 43);
            style.visuals.window_fill = rgb(44, 45, 50);
        }
        Theme::Light => {
            let v = &mut style.visuals;
            v.selection.bg_fill = rgb(240, 160, 90);
            v.selection.stroke.color = rgb(68, 66, 64);
            v.panel_fill = rgb(237, 232, 228);
            v.window_fill = rgb(243, 239, 236);
            v.faint_bg_color = rgb(228, 223, 219);
            v.extreme_bg_color = rgb(252, 251, 250);
            v.window_stroke.color = rgb(208, 202, 199);
            v.widgets.noninteractive.bg_fill = rgb(237, 232, 228);
            v.widgets.noninteractive.fg_stroke.color = rgb(107, 104, 100);
            v.widgets.noninteractive.bg_stroke.color = rgb(208, 202, 199);
            v.widgets.inactive.bg_fill = rgb(222, 217, 213);
            v.widgets.inactive.weak_bg_fill = rgb(222, 217, 213);
            v.widgets.inactive.fg_stroke.color = rgb(68, 66, 64);
            v.widgets.hovered.bg_fill = rgb(179, 175, 171);
            v.widgets.hovered.weak_bg_fill = rgb(179, 175, 171);
            v.widgets.hovered.fg_stroke.color = rgb(40, 38, 36);
            v.widgets.active.bg_fill = rgb(165, 161, 157);
            v.widgets.active.weak_bg_fill = rgb(165, 161, 157);
            v.widgets.active.fg_stroke.color = rgb(30, 28, 26);
        }
    }
    ctx.set_global_style(style);
}

pub fn section(ui: &mut Ui, title: &str) {
    ui.add_space(4.0);
    let c = ui.visuals().strong_text_color();
    ui.label(RichText::new(title).strong().color(c));
    ui.separator();
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
