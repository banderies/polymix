//! A busy spinner that asks for 15 frames a second instead of every frame.
//!
//! egui's own spinner requests a repaint on every frame it is visible, so a panel showing it
//! (a long operation running) keeps the whole UI redrawing at the display's rate, taking
//! CPU and GPU time from the work it waits for. This one looks and lays out the same.

use egui::{emath::lerp, vec2, Pos2, Response, Sense, Shape, Stroke, Ui, WidgetInfo};
use std::time::Duration;

/// Time between the spinner's frames: about 15 a second, smooth enough for a busy sign.
const FRAME: Duration = Duration::from_millis(66);

/// Like `ui.spinner()`: the same size, place, accessibility role and drawing (an arc turning
/// once a second), but the next frame is asked for after 66 ms rather than at once.
pub fn spinner(ui: &mut Ui) -> Response {
    let size = ui.style().spacing.interact_size.y;
    let (rect, response) = ui.allocate_exact_size(vec2(size, size), Sense::hover());
    response.widget_info(|| WidgetInfo::new(egui::Role::ProgressIndicator));
    if ui.is_rect_visible(rect) {
        ui.ctx().request_repaint_after(FRAME);
        let color = ui.visuals().strong_text_color();
        let radius = (rect.height().min(rect.width()) * 0.5) - 2.0;
        let n_points = (radius.round() as u32).clamp(8, 128);
        let time = ui.input(|i| i.time);
        let start_angle = time * std::f64::consts::TAU;
        let end_angle = start_angle + 240f64.to_radians() * time.sin();
        let points: Vec<Pos2> = (0..n_points)
            .map(|i| {
                let angle = lerp(start_angle..=end_angle, i as f64 / n_points as f64);
                let (sin, cos) = angle.sin_cos();
                rect.center() + radius * vec2(cos as f32, sin as f32)
            })
            .collect();
        ui.painter().add(Shape::line(points, Stroke::new(3.0, color)));
    }
    response
}
