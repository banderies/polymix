//! Polymix's own icon set, drawn as vector strokes so it scales with the UI and follows the
//! theme's text colour.

use egui::{Color32, Painter, Pos2, Rect, Shape, Stroke};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Icon {
    Import,
    Select,
    Sculpt,
    Edit,
    Vessel,
    Analysis,
    Shaders,
    Export,
}

/// Draw `icon` filling `rect` (best square) in `color`.
pub fn paint(painter: &Painter, rect: Rect, icon: Icon, color: Color32) {
    let s = rect.width().min(rect.height());
    let o = rect.center() - egui::vec2(s, s) * 0.5;
    // Coordinates in a 0..1 box.
    let p = |x: f32, y: f32| Pos2::new(o.x + x * s, o.y + y * s);
    let stroke = Stroke::new((s * 0.09).max(1.2), color);
    let line = |pts: Vec<Pos2>| painter.add(Shape::line(pts, stroke));
    match icon {
        Icon::Import | Icon::Export => {
            // Tray with an arrow going in (import) or out (export).
            line(vec![p(0.12, 0.62), p(0.12, 0.9), p(0.88, 0.9), p(0.88, 0.62)]);
            let (from, to) = if icon == Icon::Import { (0.08, 0.62) } else { (0.62, 0.08) };
            line(vec![p(0.5, from), p(0.5, to)]);
            let tip = if icon == Icon::Import { 1.0 } else { -1.0 };
            line(vec![p(0.32, to - 0.18 * tip), p(0.5, to), p(0.68, to - 0.18 * tip)]);
        }
        Icon::Select => {
            let pts = vec![p(0.22, 0.08), p(0.22, 0.82), p(0.42, 0.64), p(0.56, 0.92), p(0.68, 0.86), p(0.54, 0.58), p(0.8, 0.56)];
            painter.add(Shape::convex_polygon(vec![p(0.22, 0.08), p(0.22, 0.82), p(0.8, 0.56)], color.gamma_multiply(0.25), Stroke::NONE));
            painter.add(Shape::closed_line(pts, stroke));
        }
        Icon::Sculpt => {
            // Brush: handle and bristle tip, with a stroke it painted.
            line(vec![p(0.9, 0.1), p(0.52, 0.48)]);
            painter.add(Shape::convex_polygon(vec![p(0.52, 0.42), p(0.58, 0.48), p(0.36, 0.72), p(0.24, 0.62)], color, Stroke::NONE));
            painter.add(Shape::line(vec![p(0.08, 0.92), p(0.2, 0.8), p(0.32, 0.86)], stroke));
        }
        Icon::Edit => {
            // Wireframe cube.
            let f = [p(0.1, 0.32), p(0.66, 0.32), p(0.66, 0.9), p(0.1, 0.9)];
            let b = [p(0.34, 0.1), p(0.9, 0.1), p(0.9, 0.68), p(0.34, 0.68)];
            painter.add(Shape::closed_line(f.to_vec(), stroke));
            line(vec![b[3], b[0], b[1], b[2]]);
            for i in 0..3 {
                line(vec![f[i], b[i]]);
            }
        }
        Icon::Vessel => {
            // A vessel branching in two.
            let thick = Stroke::new(stroke.width * 2.2, color);
            painter.add(Shape::line(vec![p(0.5, 0.95), p(0.5, 0.55)], thick));
            painter.add(Shape::line(vec![p(0.5, 0.55), p(0.3, 0.3), p(0.18, 0.06)], Stroke::new(stroke.width * 1.7, color)));
            painter.add(Shape::line(vec![p(0.5, 0.55), p(0.72, 0.32), p(0.9, 0.18)], Stroke::new(stroke.width * 1.4, color)));
        }
        Icon::Analysis => {
            for (i, h) in [0.35f32, 0.6, 0.85].iter().enumerate() {
                let x = 0.14 + 0.3 * i as f32;
                painter.add(Shape::rect_filled(Rect::from_min_max(p(x, 0.92 - h), p(x + 0.18, 0.92)), 1.0, color.gamma_multiply(if i == 2 { 1.0 } else { 0.7 })));
            }
        }
        Icon::Shaders => {
            painter.circle(p(0.5, 0.5), s * 0.4, color.gamma_multiply(0.3), stroke);
            painter.circle_filled(p(0.36, 0.36), s * 0.1, color);
        }
    }
}
