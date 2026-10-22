//! The card an effect's controls show in: its name and what it does, then numbered steps in
//! the order the work goes. A step whose requirement is met has a tick; the first one that
//! still needs doing is highlighted, so the card reads as "you are here".

use egui::{Color32, RichText, Ui};

/// Steps drawn so far in a card.
pub struct Card {
    n: usize,
    /// The step to do now has been drawn (later unfinished steps wait).
    current_drawn: bool,
}

/// Whether a step is done.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ready {
    Done,
    /// Still to do: the first such step is the one to do now.
    Todo,
    /// Optional, or nothing to finish (settings with a value already).
    Either,
}

impl Ready {
    pub fn when(done: bool) -> Self {
        if done {
            Ready::Done
        } else {
            Ready::Todo
        }
    }
}

const DONE: Color32 = Color32::from_rgb(90, 170, 100);

/// Draw an effect's card: `title`, `summary`, ? (shows `help`) and ✕ (sets `close`), then the
/// steps `add` draws.
pub fn show(ui: &mut Ui, id: &str, title: &str, summary: &str, help: &str, close: &mut bool, add: impl FnOnce(&mut Ui, &mut Card)) {
    let id = egui::Id::NULL.with(("segmentation.card", id));
    let accent = ui.visuals().selection.bg_fill;
    egui::Frame::group(ui.style()).fill(ui.visuals().faint_bg_color).stroke(egui::Stroke::new(1.0, accent.gamma_multiply(0.6))).corner_radius(6.0).inner_margin(egui::Margin::same(8)).show(ui, |ui| {
        ui.set_width(ui.available_width());
        let mut show_help = ui.data(|d| d.get_temp::<bool>(id).unwrap_or(false));
        ui.horizontal(|ui| {
            ui.label(RichText::new(title).size(15.0).strong());
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.small_button("×").on_hover_text("Put the effect away (Esc)").clicked() {
                    *close = true;
                }
                if ui.selectable_label(show_help, "?").on_hover_text("How it works, and its keys").clicked() {
                    show_help = !show_help;
                }
            });
        });
        ui.data_mut(|d| d.insert_temp(id, show_help));
        ui.label(summary);
        if show_help {
            egui::Frame::new().fill(ui.visuals().extreme_bg_color).corner_radius(4.0).inner_margin(egui::Margin::same(6)).show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.label(RichText::new(help).small());
            });
        }
        ui.add_space(4.0);
        let mut card = Card { n: 0, current_drawn: false };
        add(ui, &mut card);
    });
}

impl Card {
    /// A numbered step: `title`, whether it is done, and its controls.
    pub fn step(&mut self, ui: &mut Ui, title: &str, ready: Ready, add: impl FnOnce(&mut Ui)) {
        self.n += 1;
        let current = ready == Ready::Todo && !self.current_drawn;
        self.current_drawn |= current;
        let accent = ui.visuals().selection.bg_fill;
        let frame = if current { egui::Frame::new().fill(accent.gamma_multiply(0.18)) } else { egui::Frame::new() };
        frame.corner_radius(4.0).inner_margin(egui::Margin::symmetric(4, 3)).show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                // The step's number in a disc: green with a tick when done, the accent when it
                // is the one to do now.
                let (rect, _) = ui.allocate_exact_size(egui::vec2(18.0, 18.0), egui::Sense::hover());
                let (fill, text, mark) = match (ready, current) {
                    (Ready::Done, _) => (DONE, Color32::WHITE, "✓".to_string()),
                    (_, true) => (accent, ui.visuals().strong_text_color(), self.n.to_string()),
                    _ => (ui.visuals().widgets.inactive.bg_fill, ui.visuals().text_color(), self.n.to_string()),
                };
                ui.painter().circle_filled(rect.center(), 9.0, fill);
                ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, mark, egui::FontId::proportional(11.0), text);
                let title = RichText::new(title).strong();
                ui.label(if current { title.color(ui.visuals().strong_text_color()) } else { title });
            });
            ui.indent(("step", self.n), add);
        });
    }

    /// The optional Limits step (masking), folded unless opened: `summary` says what it is set
    /// to.
    pub fn limits(&mut self, ui: &mut Ui, on: bool, summary: &str, add: impl FnOnce(&mut Ui)) {
        ui.add_space(2.0);
        let title = if on { RichText::new(format!("Limits: {summary}")).strong().color(crate::WARN) } else { RichText::new("Limits: none").strong() };
        egui::CollapsingHeader::new(title).id_salt(("limits", self.n)).default_open(false).show(ui, |ui| {
            ui.weak("Where this effect may change voxels. The same limits apply to every effect.");
            add(ui);
        });
    }
}
