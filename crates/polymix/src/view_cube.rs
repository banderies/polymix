//! View cube: a small cube in the viewport corner that turns with the camera. Clicking a
//! face, edge or corner looks at the model from that direction; dragging it tumbles.
//!
//! Everything here works in the camera's canonical frame (Y up, front = +Z), the frame of
//! `Camera::orient`, so the labels match View → Standard views in both up-axis modes.

use egui::{Color32, Pos2, Stroke, Vec2};
use glam::{DQuat, DVec3};

/// Faces: outward normal, the two in-face axes, and the label.
fn faces() -> [(DVec3, DVec3, DVec3, &'static str); 6] {
    [
        (DVec3::Z, DVec3::X, DVec3::Y, "Front"),
        (-DVec3::Z, -DVec3::X, DVec3::Y, "Back"),
        (DVec3::X, -DVec3::Z, DVec3::Y, "Right"),
        (-DVec3::X, DVec3::Z, DVec3::Y, "Left"),
        (DVec3::Y, DVec3::X, -DVec3::Z, "Top"),
        (-DVec3::Y, DVec3::X, DVec3::Z, "Bottom"),
    ]
}

/// Screen position of cube point `p` (cube spans ±1) for camera orientation `orient`.
fn project(orient: DQuat, centre: Pos2, scale: f32, p: DVec3) -> (Pos2, f64) {
    let q = orient.inverse() * p;
    (centre + Vec2::new(q.x as f32, -q.y as f32) * scale, q.z)
}

/// The face, edge or corner under `p`: the sum of the axes it points along (one axis for a
/// face, two for an edge, three for a corner). Each face is split 3 × 3, like a tic-tac-toe
/// board, into its centre, edge strips and corners.
pub fn region_at(orient: DQuat, centre: Pos2, scale: f32, p: Pos2) -> Option<DVec3> {
    let mut best: Option<(f64, DVec3)> = None;
    for (n, u, v, _) in faces() {
        let (c, depth) = project(orient, centre, scale, n);
        if depth <= 1e-3 {
            continue;
        }
        // Screen-space affine map of the face: c + a·du + b·dv.
        let du = project(orient, centre, scale, n + u).0 - c;
        let dv = project(orient, centre, scale, n + v).0 - c;
        let det = du.x * dv.y - du.y * dv.x;
        if det.abs() < 1e-6 {
            continue;
        }
        let r = p - c;
        let a = ((r.x * dv.y - r.y * dv.x) / det) as f64;
        let b = ((du.x * r.y - du.y * r.x) / det) as f64;
        if a.abs() > 1.0 || b.abs() > 1.0 {
            continue;
        }
        let mut d = n;
        if a.abs() > 0.6 {
            d += u * a.signum();
        }
        if b.abs() > 0.6 {
            d += v * b.signum();
        }
        if best.is_none_or(|x| depth > x.0) {
            best = Some((depth, d));
        }
    }
    best.map(|x| x.1)
}

/// Yaw and pitch (as used by `Camera::set_view`) that look at the model from direction `d`.
pub fn yaw_pitch(d: DVec3) -> (f64, f64) {
    let d = d.normalize();
    let pitch = d.y.clamp(-1.0, 1.0).asin();
    let yaw = if d.x.abs() < 1e-9 && d.z.abs() < 1e-9 { 0.0 } else { d.x.atan2(d.z) };
    (yaw, pitch)
}

/// Draw the cube, highlighting the hovered region.
pub fn draw(painter: &egui::Painter, orient: DQuat, centre: Pos2, scale: f32, hover: Option<DVec3>, accent: Color32) {
    let face_fill = Color32::from_rgba_unmultiplied(200, 204, 210, 225);
    let edge = Stroke::new(1.0, Color32::from_gray(70));
    let mut visible: Vec<(f64, DVec3, DVec3, DVec3, &str)> = faces()
        .into_iter()
        .filter_map(|(n, u, v, label)| {
            let (_, depth) = project(orient, centre, scale, n);
            (depth > 1e-3).then_some((depth, n, u, v, label))
        })
        .collect();
    visible.sort_by(|a, b| a.0.total_cmp(&b.0));
    for (depth, n, u, v, label) in visible {
        let pt = |a: f64, b: f64| project(orient, centre, scale, n + u * a + v * b).0;
        let quad = |a0: f64, b0: f64, a1: f64, b1: f64| vec![pt(a0, b0), pt(a1, b0), pt(a1, b1), pt(a0, b1)];
        // Slightly darker the more the face turns away.
        let shade = (0.75 + 0.25 * depth) as f32;
        let fill = Color32::from_rgba_unmultiplied((face_fill.r() as f32 * shade) as u8, (face_fill.g() as f32 * shade) as u8, (face_fill.b() as f32 * shade) as u8, face_fill.a());
        painter.add(egui::Shape::convex_polygon(quad(-1.0, -1.0, 1.0, 1.0), fill, edge));
        // Highlight the hovered piece of this face (its 3 × 3 cell).
        if let Some(h) = hover.filter(|h| h.dot(n) > 0.5) {
            let cell = |axis: DVec3| {
                let s = h.dot(axis);
                if s > 0.5 {
                    (0.6, 1.0)
                } else if s < -0.5 {
                    (-1.0, -0.6)
                } else {
                    (-0.6, 0.6)
                }
            };
            let ((a0, a1), (b0, b1)) = (cell(u), cell(v));
            painter.add(egui::Shape::convex_polygon(quad(a0, b0, a1, b1), accent.gamma_multiply(0.85), Stroke::NONE));
        }
        if depth > 0.25 {
            let c = project(orient, centre, scale, n).0;
            let size = (10.0 * depth as f32).max(7.0);
            painter.text(c, egui::Align2::CENTER_CENTER, label, egui::FontId::proportional(size), Color32::from_gray(40));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regions_and_views() {
        // Looking from the front: the middle of the cube is the front face, its corner a corner.
        let c = Pos2::new(100.0, 100.0);
        assert_eq!(region_at(DQuat::IDENTITY, c, 30.0, c), Some(DVec3::Z));
        assert_eq!(region_at(DQuat::IDENTITY, c, 30.0, c + Vec2::new(28.0, -28.0)), Some(DVec3::new(1.0, 1.0, 1.0)));
        assert_eq!(region_at(DQuat::IDENTITY, c, 30.0, c + Vec2::new(28.0, 0.0)), Some(DVec3::new(1.0, 0.0, 1.0)));
        assert_eq!(region_at(DQuat::IDENTITY, c, 30.0, c + Vec2::new(60.0, 0.0)), None);
        // The view for a direction looks along it.
        for d in [DVec3::Z, DVec3::X, -DVec3::X, DVec3::Y, -DVec3::Y, DVec3::new(1.0, 1.0, 1.0)] {
            let (yaw, pitch) = yaw_pitch(d);
            let back = DQuat::from_rotation_y(yaw) * DQuat::from_rotation_x(-pitch) * DVec3::Z;
            assert!((back - d.normalize()).length() < 1e-9, "{d}");
        }
    }
}
