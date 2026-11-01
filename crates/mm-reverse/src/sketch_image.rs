// SPDX-License-Identifier: AGPL-3.0-only
//! A picture laid under a sketch to trace: a photo of the part, a page of its datasheet —
//! placed on the sheet in millimetres, brought to scale by two points on it a known length
//! apart, moved by a point on it and where that point belongs, levelled by two points that
//! should lie level. Ported from ScanRuler (`plugins/reverse/core/sketchImage.ts`),
//! © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! Plain data in sheet millimetres (u, v), the picture itself a data URL, so a sketch with
//! one saves as it is; the picture is the part model's `SketchImage`.

use glam::DVec2;

use crate::model::SketchImage;

/// The longest side a picture is kept at, pixels: past this a photo is sharper than a sketch
/// is traced, and heavier than a project should be.
pub const IMAGE_MAX_PX: f64 = 2400.0;

/// How wide a picture is laid in before it is brought to scale, mm.
const IMAGE_START_WIDTH_MM: f64 = 100.0;

/// A picture of the given size laid in `IMAGE_START_WIDTH_MM` wide, its centre at `centre`.
pub fn placed_image(src: &str, width: u32, height: u32, centre: DVec2, name: Option<&str>) -> SketchImage {
    let (w, h) = (width as f64, height as f64);
    let mm_per_px = IMAGE_START_WIDTH_MM / w.max(1.0);
    SketchImage {
        src: src.to_string(),
        width,
        height,
        origin: DVec2::new(centre.x - (w * mm_per_px) / 2.0, centre.y - (h * mm_per_px) / 2.0),
        mm_per_px,
        angle_deg: 0.0,
        opacity: 0.6,
        name: name.filter(|n| !n.is_empty()).map(str::to_string),
    }
}

fn turn(v: DVec2, deg: f64) -> DVec2 {
    let a = (deg * std::f64::consts::PI) / 180.0;
    DVec2::new(v.x * a.cos() - v.y * a.sin(), v.x * a.sin() + v.y * a.cos())
}

/// A point of the picture, in its pixels from the bottom-left, on the sheet.
pub fn image_point(img: &SketchImage, px: DVec2) -> DVec2 {
    img.origin + turn(px * img.mm_per_px, img.angle_deg)
}

/// The picture's corners on the sheet, bottom-left round counter-clockwise.
pub fn image_corners(img: &SketchImage) -> [DVec2; 4] {
    let (w, h) = (img.width as f64, img.height as f64);
    [image_point(img, DVec2::ZERO), image_point(img, DVec2::new(w, 0.0)), image_point(img, DVec2::new(w, h)), image_point(img, DVec2::new(0.0, h))]
}

/// The picture brought to scale: the two sheet points `a` and `b`, picked on it, are `length`
/// millimetres apart on the part. It grows or shrinks about `a`, which stays where it is.
/// `None` when the points are one.
pub fn scale_image(img: &SketchImage, a: DVec2, b: DVec2, length: f64) -> Option<SketchImage> {
    let now = (b - a).x.hypot((b - a).y);
    if !(now > 1e-9) || !(length > 0.0) {
        return None;
    }
    let k = length / now;
    Some(SketchImage { mm_per_px: img.mm_per_px * k, origin: a + (img.origin - a) * k, ..img.clone() })
}

/// The picture moved so the point of it at `from` lies at `to`.
pub fn move_image(img: &SketchImage, from: DVec2, to: DVec2) -> SketchImage {
    SketchImage { origin: img.origin + (to - from), ..img.clone() }
}

/// The picture turned about `a` so the line from `a` to `b` runs level — along the sheet's X,
/// the way it points nearest. `None` when the points are one.
pub fn level_image(img: &SketchImage, a: DVec2, b: DVec2) -> Option<SketchImage> {
    let d = b - a;
    if !(d.x.hypot(d.y) > 1e-9) {
        return None;
    }
    let mut deg = (-d.y.atan2(d.x) * 180.0) / std::f64::consts::PI;
    // Picked right to left, the line still lies level, not upside down.
    if deg > 90.0 {
        deg -= 180.0;
    }
    if deg < -90.0 {
        deg += 180.0;
    }
    Some(turn_image(img, deg, a))
}

/// The picture turned by `deg` counter-clockwise about the sheet point `about`.
pub fn turn_image(img: &SketchImage, deg: f64, about: DVec2) -> SketchImage {
    let angle = (((img.angle_deg + deg) % 360.0) + 540.0) % 360.0 - 180.0;
    SketchImage { angle_deg: if angle.abs() < 1e-9 { 0.0 } else { angle }, origin: about + turn(img.origin - about, deg), ..img.clone() }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn near(a: DVec2, b: DVec2, digits: i32) {
        let tol = 0.5 * 10f64.powi(-digits);
        assert!((a.x - b.x).abs() < tol && (a.y - b.y).abs() < tol, "{a} vs {b}");
    }

    #[test]
    fn is_laid_in_a_hundred_millimetres_wide_about_the_centre_given() {
        let img = placed_image("data:", 1000, 500, DVec2::new(10.0, 20.0), None);
        assert!((img.mm_per_px - 0.1).abs() < 5e-13);
        near(image_corners(&img)[0], DVec2::new(-40.0, -5.0), 9);
        near(image_corners(&img)[2], DVec2::new(60.0, 45.0), 9);
    }

    #[test]
    fn comes_to_scale_by_two_points_a_known_length_apart_the_first_staying_put() {
        let img = placed_image("data:", 1000, 500, DVec2::ZERO, None);
        // Two marks on it 30 mm apart on the sheet that are 45 mm on the part.
        let a = DVec2::new(-10.0, 0.0);
        let b = DVec2::new(20.0, 0.0);
        let scaled = scale_image(&img, a, b, 45.0).unwrap();
        assert!((scaled.mm_per_px - 0.15).abs() < 5e-13);
        // The pixel under a is under it still; the one under b is 45 mm off.
        let px_a = (a - img.origin) / img.mm_per_px;
        let px_b = (b - img.origin) / img.mm_per_px;
        near(image_point(&scaled, px_a), a, 9);
        near(image_point(&scaled, px_b), DVec2::new(35.0, 0.0), 9);
        assert!(scale_image(&img, a, a, 10.0).is_none());
    }

    #[test]
    fn moves_by_a_point_and_where_it_belongs_and_turns_level_about_the_first_of_two() {
        let img = placed_image("data:", 100, 100, DVec2::ZERO, None);
        near(move_image(&img, DVec2::new(3.0, 4.0), DVec2::ZERO).origin, DVec2::new(img.origin.x - 3.0, img.origin.y - 4.0), 9);
        // A line on the picture at 30° made level, picked either way along it.
        let a = DVec2::ZERO;
        let t = std::f64::consts::PI / 6.0;
        let b = DVec2::new(t.cos() * 10.0, t.sin() * 10.0);
        assert!((level_image(&img, a, b).unwrap().angle_deg + 30.0).abs() < 5e-10);
        assert!((level_image(&img, b, a).unwrap().angle_deg + 30.0).abs() < 5e-10);
        // The corner a quarter turn round about the origin.
        near(turn_image(&img, 90.0, DVec2::ZERO).origin, DVec2::new(-img.origin.y, img.origin.x), 9);
    }

    #[test]
    fn is_scaled_moved_and_levelled_as_the_sketch_editors_clicks_do_it() {
        // The editor's three picks, as the store makes them, on a 200 × 100 picture.
        let img = placed_image("data:image/png;base64,AAAA", 200, 100, DVec2::ZERO, Some("drawing.png"));
        let scaled = scale_image(&img, DVec2::new(-50.0, 0.0), DVec2::new(50.0, 0.0), 25.0).unwrap();
        assert!((scaled.mm_per_px - 0.125).abs() < 5e-13);
        // A point of the picture onto the origin.
        let moved = move_image(&scaled, DVec2::new(-50.0, 0.0), DVec2::ZERO);
        near(image_point(&moved, DVec2::new(0.0, 50.0)), DVec2::ZERO, 6);
        let level = level_image(&moved, DVec2::ZERO, DVec2::new(10.0, 10.0)).unwrap();
        assert!((level.angle_deg + 45.0).abs() < 5e-10);
        assert_eq!(level.name.as_deref(), Some("drawing.png"));
        let back: SketchImage = serde_json::from_str(&serde_json::to_string(&level).unwrap()).unwrap();
        assert_eq!(back, level);
    }
}
