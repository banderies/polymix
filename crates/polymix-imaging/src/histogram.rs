//! The histogram shown in the bottom left of an image view while its window / level is
//! dragged: how the image's values are spread, the window's two ends and the grey ramp
//! between them.

use egui::{Color32, FontId, Pos2, Rect, Stroke, Vec2};
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;

const BINS: usize = 256;
/// Values sampled for the histogram, at most.
const SAMPLES: usize = 1 << 20;

/// Counts of values in equal bins from `lo` to `hi` (the 0.1st to the 99.9th percentile, so a
/// few extreme voxels do not squeeze the rest into a bin or two).
#[derive(Clone, Debug, PartialEq)]
pub struct Histogram {
    pub lo: f64,
    pub hi: f64,
    pub counts: Vec<u32>,
}

impl Histogram {
    pub fn new(values: &[f32]) -> Self {
        let step = values.len().div_ceil(SAMPLES).max(1);
        let mut s: Vec<f32> = values.iter().step_by(step).copied().filter(|v| v.is_finite()).collect();
        let mut counts = vec![0u32; BINS];
        if s.is_empty() {
            return Self { lo: 0.0, hi: 1.0, counts };
        }
        let n = s.len();
        let lo = *s.select_nth_unstable_by(n / 1000, f32::total_cmp).1 as f64;
        let hi = *s.select_nth_unstable_by(n - 1 - n / 1000, f32::total_cmp).1 as f64;
        let hi = if hi > lo { hi } else { lo + 1.0 };
        let scale = BINS as f64 / (hi - lo);
        for &v in &s {
            let b = (v as f64 - lo) * scale;
            if (0.0..=BINS as f64).contains(&b) {
                counts[(b as usize).min(BINS - 1)] += 1;
            }
        }
        Self { lo, hi, counts }
    }

    /// The histogram of `values`, made once per `key` and kept in egui's memory (so the views
    /// of one image share it).
    pub fn cached(ctx: &egui::Context, key: impl Hash + Debug, values: &[f32]) -> Arc<Histogram> {
        let id = egui::Id::unique(("imaging.histogram", key));
        if let Some(h) = ctx.data(|d| d.get_temp::<Arc<Histogram>>(id)) {
            return h;
        }
        let h = Arc::new(Histogram::new(values));
        ctx.data_mut(|d| d.insert_temp(id, h.clone()));
        h
    }
}

/// While `shown` (fading out after it), draw the histogram of `values` (cached under `key`)
/// in the bottom left of `view` with its lowest edge at `bottom`, and `window` (centre,
/// width) over it: the ends of the window and the ramp from black to white between them
/// (white to black when `invert`).
#[allow(clippy::too_many_arguments)]
pub fn show(ui: &egui::Ui, id: egui::Id, view: Rect, bottom: f32, key: impl Hash + Debug, values: &[f32], window: (f64, f64), invert: bool, shown: bool) {
    // On at once, fading out after.
    let fading = ui.ctx().animate_bool_with_time(id.with("window_histogram"), shown, 0.3);
    let alpha = if shown { 1.0 } else { fading };
    if alpha <= 0.0 {
        return;
    }
    let h = Histogram::cached(ui.ctx(), key, values);
    paint(&ui.painter_at(view), view, bottom, &h, window, invert, alpha);
}

fn paint(painter: &egui::Painter, view: Rect, bottom: f32, h: &Histogram, window: (f64, f64), invert: bool, alpha: f32) {
    let (margin, pad, label_h) = (8.0, 6.0, 13.0);
    let w = (view.width() * 0.45).clamp(110.0, 240.0);
    let plot_h = (w * 0.3).clamp(32.0, 64.0);
    let panel = Rect::from_min_max(Pos2::new(view.left() + margin, bottom - plot_h - label_h - 2.0 * pad), Pos2::new(view.left() + margin + w, bottom));
    if panel.right() > view.right() - margin || panel.top() < view.top() + 24.0 {
        return;
    }
    let plot = Rect::from_min_max(panel.min + Vec2::splat(pad), Pos2::new(panel.right() - pad, panel.bottom() - pad - label_h));
    let fade = |c: Color32| c.gamma_multiply(alpha);
    painter.rect_filled(panel, 4.0, fade(Color32::from_black_alpha(190)));

    let (c, width) = window;
    let (w0, w1) = (c - width * 0.5, c + width * 0.5);
    // The values the image has and the window covers, with a little room either side.
    let (a, b) = (h.lo.min(w0), h.hi.max(w1));
    let room = (b - a) * 0.04;
    let (a, b) = (a - room, b + room);
    let x = |v: f64| plot.left() + ((v - a) / (b - a)) as f32 * plot.width();

    // Bars on a log scale (the background's bin would flatten everything else), bright
    // inside the window.
    let norm = (1.0 + h.counts.iter().copied().max().unwrap_or(0) as f32).ln().max(1e-6);
    let bin = (h.hi - h.lo) / h.counts.len() as f64;
    for (i, &n) in h.counts.iter().enumerate() {
        if n == 0 {
            continue;
        }
        let (v0, v1) = (h.lo + i as f64 * bin, h.lo + (i + 1) as f64 * bin);
        let height = (1.0 + n as f32).ln() / norm * plot.height();
        let grey = if (w0..=w1).contains(&((v0 + v1) * 0.5)) { 190 } else { 85 };
        let (x0, x1) = (x(v0), x(v1).max(x(v0) + 1.0));
        painter.rect_filled(Rect::from_min_max(Pos2::new(x0, plot.bottom() - height), Pos2::new(x1, plot.bottom())), 0.0, fade(Color32::from_gray(grey)));
    }

    // The window: a band between its ends, the ends themselves and the ramp.
    let accent = Color32::from_rgb(255, 196, 0);
    let (xa, xb) = (x(w0), x(w1));
    painter.rect_filled(Rect::from_x_y_ranges(xa..=xb, plot.y_range()), 0.0, fade(accent.gamma_multiply(0.12)));
    for xe in [xa, xb] {
        painter.line_segment([Pos2::new(xe, plot.top()), Pos2::new(xe, plot.bottom())], Stroke::new(1.0, fade(accent.gamma_multiply(0.8))));
    }
    let (dark, bright) = if invert { (plot.top(), plot.bottom()) } else { (plot.bottom(), plot.top()) };
    let ramp = vec![Pos2::new(plot.left(), dark), Pos2::new(xa, dark), Pos2::new(xb, bright), Pos2::new(plot.right(), bright)];
    painter.add(egui::Shape::line(ramp, Stroke::new(1.5, fade(accent))));

    // The values at the ends, under them (kept inside the panel).
    let font = FontId::proportional(10.5);
    let fmt = |v: f64| if width < 10.0 { format!("{v:.2}") } else { format!("{v:.0}") };
    let y = plot.bottom() + 2.0;
    let place = |text: String, at: f32| {
        let g = painter.layout_no_wrap(text, font.clone(), fade(accent));
        let x0 = (at - g.size().x * 0.5).clamp(panel.left() + pad, (panel.right() - pad - g.size().x).max(panel.left() + pad));
        (g, x0)
    };
    let (lo_text, lo_x) = place(fmt(w0), xa);
    let (hi_text, hi_x) = place(fmt(w1), xb);
    if hi_x >= lo_x + lo_text.size().x + 6.0 {
        painter.galley(Pos2::new(lo_x, y), lo_text, Color32::PLACEHOLDER);
        painter.galley(Pos2::new(hi_x, y), hi_text, Color32::PLACEHOLDER);
    } else {
        // A narrow window: one label for both ends.
        let (both, x0) = place(format!("{} – {}", fmt(w0), fmt(w1)), (xa + xb) * 0.5);
        painter.galley(Pos2::new(x0, y), both, Color32::PLACEHOLDER);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn robust_range_and_counts() {
        // 0..1000 once each, and one extreme voxel.
        let mut v: Vec<f32> = (0..1000).map(|x| x as f32).collect();
        v.push(1e6);
        let h = Histogram::new(&v);
        assert!(h.lo <= 1.0 && (998.0..1000.0).contains(&h.hi), "{} {}", h.lo, h.hi);
        let total: u32 = h.counts.iter().sum();
        assert!((995..=1000).contains(&total), "{total}");
        // Evenly spread values fill the bins evenly.
        let (min, max) = (h.counts.iter().min().unwrap(), h.counts.iter().max().unwrap());
        assert!(max - min <= 2, "{min} {max}");
        // A flat image still has a range.
        let flat = Histogram::new(&[5.0; 10]);
        assert_eq!((flat.lo, flat.hi, flat.counts[0]), (5.0, 6.0, 10));
    }
}
