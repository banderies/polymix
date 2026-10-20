//! Measurements in the slice views: a ruler (distance between two points), an angle (at the
//! middle of three points) and a region (a circle on a slice: area and the image's values in
//! it). Points are clicked in a view (they lie on its slice); each measurement is drawn on the
//! slice it was made on.

use crate::slice_view::{SliceResponse, ViewMap};
use egui::{Color32, FontId, Pos2, Stroke};
use glam::DVec3;
use mm_volume::load::LoadedVolume;
use mm_volume::slice::{plane_axes, Plane};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeasureTool {
    Ruler,
    Angle,
    Region,
}

impl MeasureTool {
    pub const ALL: [MeasureTool; 3] = [MeasureTool::Ruler, MeasureTool::Angle, MeasureTool::Region];

    pub fn label(self) -> &'static str {
        match self {
            MeasureTool::Ruler => "Ruler",
            MeasureTool::Angle => "Angle",
            MeasureTool::Region => "Region",
        }
    }

    pub fn key(self) -> &'static str {
        match self {
            MeasureTool::Ruler => "ruler",
            MeasureTool::Angle => "angle",
            MeasureTool::Region => "region",
        }
    }

    pub fn hint(self) -> &'static str {
        match self {
            MeasureTool::Ruler => "Click the two ends in a slice view",
            MeasureTool::Angle => "Click three points in a slice view: the angle is at the second",
            MeasureTool::Region => "Click the centre, then a point on the edge, of a circle in a slice view",
        }
    }

    /// Points that make one.
    pub fn points(self) -> usize {
        match self {
            MeasureTool::Angle => 3,
            _ => 2,
        }
    }
}

/// The image's values in a region.
#[derive(Clone, Debug, PartialEq)]
pub struct RegionStats {
    pub voxels: usize,
    pub area_mm2: f64,
    pub mean: f64,
    pub sd: f64,
    pub min: f32,
    pub max: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Measurement {
    pub tool: MeasureTool,
    pub plane: Plane,
    pub slice: usize,
    /// Voxel coordinates.
    pub points: Vec<DVec3>,
    /// mm (ruler), degrees (angle) or the radius in mm (region).
    pub value: f64,
    pub stats: Option<RegionStats>,
}

impl Measurement {
    /// What it measured, in words.
    pub fn text(&self) -> String {
        match (self.tool, &self.stats) {
            (MeasureTool::Ruler, _) => format!("{:.2} mm", self.value),
            (MeasureTool::Angle, _) => format!("{:.1}°", self.value),
            (MeasureTool::Region, Some(s)) => format!("{:.1} mm² · {:.4} ± {:.3} ({:.4} – {:.4}), {} voxels", s.area_mm2, s.mean, s.sd, s.min, s.max, s.voxels),
            (MeasureTool::Region, None) => "empty".into(),
        }
    }
}

/// Measure `points` (voxel coordinates, on `slice` of `plane`) with `tool`.
pub fn measure(vol: &LoadedVolume, tool: MeasureTool, plane: Plane, slice: usize, points: Vec<DVec3>) -> Measurement {
    let w = |p: DVec3| vol.geometry.world(p);
    let (value, stats) = match tool {
        MeasureTool::Ruler => (w(points[0]).distance(w(points[1])), None),
        MeasureTool::Angle => {
            let (a, b) = (w(points[0]) - w(points[1]), w(points[2]) - w(points[1]));
            (a.angle_between(b).to_degrees(), None)
        }
        MeasureTool::Region => {
            let r = w(points[0]).distance(w(points[1]));
            (r, region(vol, plane, slice, points[0], r))
        }
    };
    Measurement { tool, plane, slice, points, value, stats }
}

/// Statistics of the voxels of `slice` (of `plane`) within `r_mm` of `centre`.
fn region(vol: &LoadedVolume, plane: Plane, slice: usize, centre: DVec3, r_mm: f64) -> Option<RegionStats> {
    let axes = plane_axes(&vol.geometry, plane);
    let s = vol.geometry.spacing;
    let (step_a, step_d) = (s[axes.across], s[axes.down]);
    let (ca, cd) = (centre[axes.across], centre[axes.down]);
    let (ra, rd) = ((r_mm / step_a).ceil() as i64, (r_mm / step_d).ceil() as i64);
    let mut values = Vec::new();
    for d in (cd.round() as i64 - rd)..=(cd.round() as i64 + rd) {
        for a in (ca.round() as i64 - ra)..=(ca.round() as i64 + ra) {
            if a < 0 || d < 0 || a as usize >= vol.dims[axes.across] || d as usize >= vol.dims[axes.down] {
                continue;
            }
            if ((a as f64 - ca) * step_a).hypot((d as f64 - cd) * step_d) > r_mm {
                continue;
            }
            let mut v = [0usize; 3];
            v[axes.across] = a as usize;
            v[axes.down] = d as usize;
            v[axes.slice] = slice;
            let x = vol.get(v[0], v[1], v[2]);
            if x.is_finite() {
                values.push(x);
            }
        }
    }
    if values.is_empty() {
        return None;
    }
    let n = values.len() as f64;
    let mean = values.iter().map(|&v| v as f64).sum::<f64>() / n;
    let sd = (values.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n).sqrt();
    let (min, max) = values.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
    Some(RegionStats { voxels: values.len(), area_mm2: n * step_a * step_d, mean, sd, min, max })
}

/// The measurements of a set of views, and the tool in use.
#[derive(Clone, Debug, Default)]
pub struct Measurements {
    pub tool: Option<MeasureTool>,
    pub items: Vec<Measurement>,
    /// Points clicked so far for the next one, and the slice they are on.
    pending: Vec<DVec3>,
    pending_on: Option<(Plane, usize)>,
}

const COLOUR: Color32 = Color32::from_rgb(120, 220, 255);

impl Measurements {
    /// Clicks in the views while a tool is in use.
    pub fn input(&mut self, vol: &LoadedVolume, responses: &[(Plane, SliceResponse)]) {
        let Some(tool) = self.tool else {
            self.pending.clear();
            return;
        };
        // (On slices of the voxel grid only: regions are measured in one.)
        let Some((map, v)) = responses.iter().filter(|(_, r)| r.map.oblique.is_none()).find_map(|(_, r)| r.clicked_voxel.map(|v| (r.map, v))) else { return };
        let on = (map.plane, map.slice);
        if self.pending_on != Some(on) {
            self.pending.clear();
            self.pending_on = Some(on);
        }
        self.pending.push(v);
        if self.pending.len() >= tool.points() {
            let points = std::mem::take(&mut self.pending);
            self.items.push(measure(vol, tool, map.plane, map.slice, points));
        }
    }

    /// Drop the points of the measurement being made (Esc); `false` when there were none.
    pub fn cancel_pending(&mut self) -> bool {
        !std::mem::take(&mut self.pending).is_empty()
    }

    /// Draw the measurements on the views showing their slices, and the one being made.
    pub fn paint(&self, ui: &egui::Ui, vol: &LoadedVolume, maps: &[ViewMap], pointer: Option<Pos2>) {
        let font = FontId::proportional(12.0);
        for m in maps.iter().filter(|m| m.oblique.is_none()) {
            let painter = ui.painter_at(m.rect);
            let screen = |p: &DVec3| m.screen(p[m.axes.across], p[m.axes.down]);
            // On their slice in full; fainter within two slices of it.
            for item in self.items.iter().filter(|i| i.plane == m.plane && i.slice.abs_diff(m.slice) <= 2) {
                let colour = if item.slice == m.slice { COLOUR } else { COLOUR.gamma_multiply(0.4) };
                let stroke = Stroke::new(1.5, colour);
                let pts: Vec<Pos2> = item.points.iter().map(screen).collect();
                match item.tool {
                    MeasureTool::Ruler | MeasureTool::Angle => {
                        painter.add(egui::Shape::line(pts.clone(), stroke));
                        for &p in &pts {
                            painter.circle_filled(p, 2.5, colour);
                        }
                    }
                    MeasureTool::Region => {
                        let r_mm = item.value;
                        let ppm = m.points_per_mm(vol.geometry.spacing);
                        painter.circle_stroke(pts[0], (r_mm as f32 * ppm).max(1.0), stroke);
                    }
                }
                let at = match item.tool {
                    MeasureTool::Ruler => pts[0].lerp(pts[1], 0.5),
                    _ => pts[usize::from(item.tool == MeasureTool::Angle)],
                };
                let text = match (item.tool, &item.stats) {
                    (MeasureTool::Region, Some(s)) => format!("{:.1} mm²\n{:.4} ± {:.3}", s.area_mm2, s.mean, s.sd),
                    _ => item.text(),
                };
                let galley = painter.layout_no_wrap(text, font.clone(), colour);
                let pos = at + egui::vec2(8.0, -8.0 - galley.size().y);
                painter.rect_filled(egui::Rect::from_min_size(pos, galley.size()).expand(2.0), 2.0, Color32::from_black_alpha(160));
                painter.galley(pos, galley, colour);
            }
            // The one being made, with a rubber band to the pointer.
            let stroke = Stroke::new(1.5, COLOUR);
            if !self.pending.is_empty() && self.pending_on == Some((m.plane, m.slice)) {
                let mut pts: Vec<Pos2> = self.pending.iter().map(screen).collect();
                for &p in &pts {
                    painter.circle_filled(p, 2.5, COLOUR);
                }
                if let Some(p) = pointer.filter(|p| m.rect.contains(*p)) {
                    if self.tool == Some(MeasureTool::Region) {
                        painter.circle_stroke(pts[0], pts[0].distance(p), stroke);
                    } else {
                        pts.push(p);
                    }
                }
                painter.add(egui::Shape::line(pts, stroke));
            }
        }
    }

    /// The tool buttons, for a toolbar.
    pub fn tool_buttons(&mut self, ui: &mut egui::Ui) {
        for t in MeasureTool::ALL {
            let on = self.tool == Some(t);
            if ui.selectable_label(on, t.label()).on_hover_text(t.hint()).clicked() {
                self.tool = if on { None } else { Some(t) };
                self.pending.clear();
            }
        }
    }

    /// The list of measurements, each with a button to remove it; Copy puts them on the
    /// clipboard as text.
    pub fn list(&mut self, ui: &mut egui::Ui) {
        if self.items.is_empty() {
            ui.weak("Ruler, Angle and Region (in the toolbar) measure in the slice views.");
            return;
        }
        let mut remove = None;
        for (k, m) in self.items.iter().enumerate() {
            ui.horizontal(|ui| {
                if ui.small_button("×").clicked() {
                    remove = Some(k);
                }
                ui.label(format!("{} · {} {}: {}", m.tool.label(), m.plane.label(), m.slice + 1, m.text()));
            });
        }
        if let Some(k) = remove {
            self.items.remove(k);
        }
        ui.horizontal(|ui| {
            if ui.button("Copy").clicked() {
                let text: Vec<String> = self.items.iter().map(|m| format!("{}\t{} {}\t{}", m.tool.label(), m.plane.label(), m.slice + 1, m.text())).collect();
                ui.ctx().copy_text(text.join("\n"));
            }
            if ui.button("Clear").clicked() {
                self.items.clear();
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_volume::geometry::Geometry;

    fn vol() -> LoadedVolume {
        // 0.5 × 0.5 × 2 mm voxels; values are i.
        let g = Geometry::from_axes([20, 20, 5, 1], [0.5, 0.5, 2.0], [DVec3::X, DVec3::Y, DVec3::Z], DVec3::ZERO);
        LoadedVolume::from_values("t", [20, 20, 5], (0..2000).map(|n| (n % 20) as f32).collect(), g)
    }

    #[test]
    fn rulers_angles_and_regions() {
        let v = vol();
        let r = measure(&v, MeasureTool::Ruler, Plane::Axial, 2, vec![DVec3::new(0.0, 0.0, 2.0), DVec3::new(6.0, 8.0, 2.0)]);
        assert!((r.value - 5.0).abs() < 1e-9, "{}", r.value);
        let a = measure(&v, MeasureTool::Angle, Plane::Axial, 2, vec![DVec3::new(4.0, 0.0, 2.0), DVec3::new(0.0, 0.0, 2.0), DVec3::new(0.0, 4.0, 2.0)]);
        assert!((a.value - 90.0).abs() < 1e-9);
        // A circle of radius 1 mm (2 voxels) about voxel (10, 10): 13 voxels of 0.25 mm².
        let g = measure(&v, MeasureTool::Region, Plane::Axial, 2, vec![DVec3::new(10.0, 10.0, 2.0), DVec3::new(12.0, 10.0, 2.0)]);
        let s = g.stats.unwrap();
        assert_eq!(s.voxels, 13);
        assert!((s.area_mm2 - 13.0 * 0.25).abs() < 1e-9);
        assert!((s.mean - 10.0).abs() < 1e-9 && (s.min, s.max) == (8.0, 12.0));
    }
}
