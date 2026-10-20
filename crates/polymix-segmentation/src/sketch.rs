//! Effects drawn on a slice: Draw (click points or drag a line, then fill it), Level tracing
//! (the region under the pointer outlined; a click adds it) and Scissors (a free-form,
//! circular or rectangular shape that fills or erases inside or outside it, through a range
//! of slices), as in 3D Slicer's Segment Editor; and Threshold's local histogram (a box,
//! circle or line whose values set the range).

use crate::effects::{local_range, Effect, Histogram, LocalShape, ScissorsOp, ScissorsRange, ScissorsShape};
use crate::segmentation::Mode;
use crate::SegmentationWorkspace;
use egui::{Color32, Pos2, Stroke};
use mm_segment::shapes::{circle, polygon_prism, trace_level};
use mm_segment::Mask;
use mm_volume::load::LoadedVolume;
use mm_volume::slice::{plane_axes, Plane, PlaneAxes};
use polymix_imaging::{SliceResponse, ViewMap};

/// A shape being drawn in one view, in voxel coordinates across and down its slice.
#[derive(Clone, Debug)]
pub(crate) struct Sketch {
    pub map: ViewMap,
    pub points: Vec<(f64, f64)>,
    /// Scissors: where the drag started, and whether one is in progress.
    pub start: Option<(f64, f64)>,
    pub dragging: bool,
}

impl Sketch {
    fn new(map: ViewMap) -> Self {
        Self { map, points: Vec::new(), start: None, dragging: false }
    }

    /// Still on the view and slice it was started in.
    fn on(&self, map: &ViewMap) -> bool {
        self.map.plane == map.plane && self.map.slice == map.slice
    }
}

/// The region Level tracing outlines under the pointer, for the voxel it was traced from.
#[derive(Clone, Debug)]
pub(crate) struct Traced {
    pub map: ViewMap,
    pub seed: [usize; 3],
    pub darker: bool,
    pub voxels: Vec<usize>,
}

/// The region of Threshold's local histogram: a shape on a slice, and its values (sorted).
#[derive(Clone, Debug)]
pub(crate) struct LocalRegion {
    pub plane: Plane,
    pub slice: usize,
    /// Its outline (box, circle) or ends (line), across and down the slice in voxels.
    pub outline: Vec<(f64, f64)>,
    pub line: bool,
    pub values: Vec<f32>,
    pub hist: Histogram,
}

/// The values of `vol` in a shape on a slice: inside the polygon `outline`, or along the line
/// through its first and last points (sampled every half voxel).
pub(crate) fn region_values(vol: &LoadedVolume, axes: PlaneAxes, slice: usize, outline: &[(f64, f64)], line: bool) -> Vec<f32> {
    let mut values = Vec::new();
    if line {
        let (Some(&a), Some(&b)) = (outline.first(), outline.last()) else { return values };
        let n = ((b.0 - a.0).hypot(b.1 - a.1) * 2.0).ceil().max(1.0) as usize;
        let mut seen = std::collections::HashSet::new();
        for k in 0..=n {
            let t = k as f64 / n as f64;
            let mut v = [0usize; 3];
            let (x, y) = (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t);
            v[axes.across] = x.round().clamp(0.0, vol.dims[axes.across] as f64 - 1.0) as usize;
            v[axes.down] = y.round().clamp(0.0, vol.dims[axes.down] as f64 - 1.0) as usize;
            v[axes.slice] = slice.min(vol.dims[axes.slice] - 1);
            let i = vol.index(v[0], v[1], v[2]);
            if seen.insert(i) {
                values.push(vol.values[i]);
            }
        }
    } else if outline.len() >= 3 {
        let mask = polygon_prism(vol.dims, [axes.across, axes.down, axes.slice], outline, slice, slice);
        values.extend(mask.data.iter().zip(&vol.values).filter(|(m, _)| **m != 0).map(|(_, &v)| v));
    }
    values.retain(|v| v.is_finite());
    values.sort_by(f32::total_cmp);
    values
}

/// Across and down coordinates of a voxel position in a view.
fn on_slice(map: &ViewMap, v: glam::DVec3) -> (f64, f64) {
    (v[map.axes.across], v[map.axes.down])
}

const LINE: Color32 = Color32::from_rgb(255, 230, 120);

/// How `plane` cuts `vol`, and the voxel step along its slice axis towards the patient's
/// superior, anterior or right (as the views show it).
pub(crate) fn slicing(vol: &LoadedVolume, plane: Plane) -> (PlaneAxes, i64) {
    let axes = plane_axes(&vol.geometry, plane);
    let world = match plane {
        Plane::Sagittal => 0,
        Plane::Coronal => 1,
        Plane::Axial => 2,
    };
    (axes, if vol.geometry.directions()[axes.slice][world] >= 0.0 { 1 } else { -1 })
}

impl SegmentationWorkspace {
    /// The slice views' input for Draw, Level tracing and Scissors.
    pub(crate) fn sketch_input(&mut self, vol: &LoadedVolume, effect: Effect, responses: &[(Plane, SliceResponse)]) {
        // Shapes are drawn on slices of the voxel grid (not on oblique planes).
        let grid = |r: &&(Plane, SliceResponse)| r.1.map.oblique.is_none();
        let drag = responses.iter().filter(grid).find_map(|(_, r)| r.left_drag_voxel.map(|v| (r.map, v)));
        let click = responses.iter().filter(grid).find_map(|(_, r)| r.clicked_voxel.map(|v| (r.map, v)));
        let double = responses.iter().filter(grid).find(|(_, r)| r.double_clicked).map(|(_, r)| r.map);
        match effect {
            Effect::Draw => {
                if let Some((map, v)) = drag.or(click) {
                    let p = on_slice(&map, v);
                    let sketch = match &mut self.sketch {
                        Some(s) if s.on(&map) => s,
                        _ => self.sketch.insert(Sketch::new(map)),
                    };
                    // A dragged line keeps points half a voxel apart; a click adds one.
                    let far = sketch.points.last().is_none_or(|q| (q.0 - p.0).hypot(q.1 - p.1) >= if drag.is_some() { 0.5 } else { 0.01 });
                    if far {
                        sketch.points.push(p);
                    }
                }
                if double.is_some() {
                    self.finish_draw(vol);
                }
            }
            Effect::LevelTracing => {
                let darker = self.settings.as_ref().is_some_and(|s| s.trace_darker);
                let hover = responses.iter().filter(grid).find_map(|(_, r)| r.hover_voxel.map(|v| (r.map, v)));
                match hover {
                    Some((map, v)) => {
                        let seed = [0, 1, 2].map(|a| v[a].round().clamp(0.0, vol.dims[a] as f64 - 1.0) as usize);
                        if self.traced.as_ref().is_none_or(|t| t.seed != seed || t.darker != darker || t.map.plane != map.plane) {
                            let axes = [map.axes.across, map.axes.down, map.axes.slice];
                            self.traced = Some(Traced { map, seed, darker, voxels: trace_level(vol, axes, seed, darker) });
                        } else if let Some(t) = &mut self.traced {
                            t.map = map;
                        }
                    }
                    None => self.traced = None,
                }
                if click.is_some() {
                    if let Some(t) = self.traced.clone() {
                        let mut mask = Mask::new(vol.dims);
                        for &i in &t.voxels {
                            mask.data[i] = 1;
                        }
                        self.edit_now(vol, mask, Mode::Add, "Level tracing", true);
                    }
                }
            }
            Effect::Threshold => {
                let Some(shape) = self.settings.as_ref().map(|s| s.local_shape) else { return };
                match drag {
                    Some((map, v)) => {
                        let p = on_slice(&map, v);
                        let sketch = match &mut self.sketch {
                            Some(s) if s.on(&map) && s.dragging => s,
                            _ => {
                                let mut s = Sketch::new(map);
                                s.start = Some(p);
                                s.dragging = true;
                                self.sketch.insert(s)
                            }
                        };
                        let a = sketch.start.unwrap_or(p);
                        sketch.points = match shape {
                            LocalShape::Box => vec![a, (p.0, a.1), p, (a.0, p.1)],
                            LocalShape::Circle => {
                                let s = vol.geometry.spacing;
                                let (sa, sd) = (s[map.axes.across], s[map.axes.down]);
                                circle(a, ((p.0 - a.0) * sa).hypot((p.1 - a.1) * sd) / sa, sa / sd, 72)
                            }
                            LocalShape::Line => vec![a, p],
                        };
                    }
                    None => {
                        if self.sketch.as_ref().is_some_and(|s| s.dragging) {
                            let s = self.sketch.take().expect("checked");
                            self.set_local(vol, s.map.plane, s.map.axes, s.map.slice, s.points, shape == LocalShape::Line);
                        }
                    }
                }
            }
            Effect::Scissors => {
                let Some(st) = self.settings.as_ref() else { return };
                let shape = st.scissors_shape;
                match drag {
                    Some((map, v)) => {
                        let p = on_slice(&map, v);
                        let sketch = match &mut self.sketch {
                            Some(s) if s.on(&map) && s.dragging => s,
                            _ => {
                                let mut s = Sketch::new(map);
                                s.start = Some(p);
                                s.dragging = true;
                                self.sketch.insert(s)
                            }
                        };
                        let a = sketch.start.unwrap_or(p);
                        match shape {
                            ScissorsShape::FreeForm => {
                                if sketch.points.last().is_none_or(|q| (q.0 - p.0).hypot(q.1 - p.1) >= 0.5) {
                                    sketch.points.push(p);
                                }
                            }
                            ScissorsShape::Circle => {
                                // Round in millimetres: the radius across, scaled down by the
                                // pixels' aspect.
                                let s = vol.geometry.spacing;
                                let (sa, sd) = (s[map.axes.across], s[map.axes.down]);
                                let r_mm = ((p.0 - a.0) * sa).hypot((p.1 - a.1) * sd);
                                sketch.points = circle(a, r_mm / sa, sa / sd, 72);
                            }
                            ScissorsShape::Rectangle => sketch.points = vec![a, (p.0, a.1), p, (a.0, p.1)],
                        }
                    }
                    None => {
                        if self.sketch.as_ref().is_some_and(|s| s.dragging) {
                            let s = self.sketch.take().expect("checked");
                            self.cut(vol, s.map.axes, s.map.slice, s.map.step, &s.points);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    /// Engrave: the settings' text centred on `at` (voxel coordinates) in the slice of `axes`,
    /// reading left to right on the screen, through the depth around the slice; cut from the
    /// selected segment or raised from it.
    pub(crate) fn engrave(&mut self, vol: &LoadedVolume, axes: PlaneAxes, at: glam::DVec3) {
        let Some(st) = self.settings.as_ref() else { return };
        let shape = match mm_edit::text_stencil(&st.engrave_text) {
            Ok(s) => s,
            Err(e) => {
                self.say(format!("Engrave: {e}"), true);
                return;
            }
        };
        let sp = vol.geometry.spacing;
        // Stencil units to mm (the capitals `engrave_mm` tall), then to voxels across / down.
        let mm = st.engrave_mm as f64 / shape.cap_height.unwrap_or(1.0).max(1e-9);
        let (sa, sd) = (if axes.flip_across { -1.0 } else { 1.0 }, if axes.flip_down { -1.0 } else { 1.0 });
        let (ca, cd) = (at[axes.across], at[axes.down]);
        let n = vol.dims[axes.slice] as i64;
        let half = (st.engrave_depth_mm as f64 * 0.5 / sp[axes.slice]).round() as i64;
        let s = at[axes.slice].round() as i64;
        let (from, to) = ((s - half).clamp(0, n - 1) as usize, (s + half).clamp(0, n - 1) as usize);
        let order = [axes.across, axes.down, axes.slice];
        let mut mask = Mask::new(vol.dims);
        // Loops filled by the even-odd rule (holes in letters such as A, B, 8).
        for lp in &shape.loops {
            let poly: Vec<(f64, f64)> = lp.iter().map(|p| (ca + sa * p.x * mm / sp[axes.across], cd - sd * p.y * mm / sp[axes.down])).collect();
            let m = polygon_prism(vol.dims, order, &poly, from, to);
            for (a, b) in mask.data.iter_mut().zip(&m.data) {
                *a ^= *b;
            }
        }
        let raise = st.engrave_raise;
        let what = format!("Engrave “{}”", st.engrave_text.trim());
        self.edit_now(vol, mask, if raise { Mode::Add } else { Mode::Remove }, &what, raise);
    }

    /// Threshold's local histogram from a shape on a slice: the range is set from its values.
    pub(crate) fn set_local(&mut self, vol: &LoadedVolume, plane: Plane, axes: PlaneAxes, slice: usize, outline: Vec<(f64, f64)>, line: bool) {
        let values = region_values(vol, axes, slice, &outline, line);
        if values.is_empty() {
            self.local = None;
            return;
        }
        if let Some(st) = &mut self.settings {
            if let Some(r) = local_range(&values, st.local_lower, st.local_upper) {
                st.threshold = r;
            }
        }
        let hist = Histogram::of_values(&values, 64);
        self.local = Some(LocalRegion { plane, slice, outline, line, values, hist });
    }

    /// Fill the drawn outline into the segment, in its slice.
    pub(crate) fn finish_draw(&mut self, vol: &LoadedVolume) {
        let Some(s) = self.sketch.take() else { return };
        self.fill_outline(vol, s.map.axes, s.map.slice, &s.points);
    }

    /// Fill `polygon` (across and down on slice `slice` of the plane cut along `axes`) into
    /// the segment.
    pub(crate) fn fill_outline(&mut self, vol: &LoadedVolume, axes: PlaneAxes, slice: usize, polygon: &[(f64, f64)]) {
        if polygon.len() < 3 {
            self.say("Draw: click at least three points (or drag a line) around the region first", true);
            return;
        }
        let mask = polygon_prism(vol.dims, [axes.across, axes.down, axes.slice], polygon, slice, slice);
        self.edit_now(vol, mask, Mode::Add, "Draw", true);
    }

    /// Scissors: fill or erase inside or outside `polygon` (across and down on slice `slice`
    /// of the plane cut along `axes`, whose slices step by `step` towards the patient's
    /// superior / anterior / right), through the slices its range takes.
    pub(crate) fn cut(&mut self, vol: &LoadedVolume, axes: PlaneAxes, slice: usize, step: i64, polygon: &[(f64, f64)]) {
        let Some(st) = self.settings.as_ref() else { return };
        if polygon.len() < 3 {
            return;
        }
        let (op, range, thickness) = (st.scissors_op, st.scissors_range, st.scissors_mm as f64);
        let n = vol.dims[axes.slice];
        let s = slice as i64;
        // Slices towards the patient's superior / anterior / right are "positive" (Slicer's
        // slice normal).
        let half = (thickness * 0.5 / vol.geometry.spacing[axes.slice]).round() as i64;
        let (a, b) = match range {
            ScissorsRange::Unlimited => (0, n as i64 - 1),
            ScissorsRange::Positive => if step > 0 { (s, n as i64 - 1) } else { (0, s) },
            ScissorsRange::Negative => if step > 0 { (0, s) } else { (s, n as i64 - 1) },
            ScissorsRange::Symmetric => (s - half, s + half),
        };
        let axes = [axes.across, axes.down, axes.slice];
        let (from, to) = (a.clamp(0, n as i64 - 1) as usize, b.clamp(0, n as i64 - 1) as usize);
        let inside = polygon_prism(vol.dims, axes, polygon, from, to);
        let mask = if matches!(op, ScissorsOp::FillOutside | ScissorsOp::EraseOutside) {
            // Outside it within the same slices.
            let all = [(-1e9, -1e9), (1e9, -1e9), (1e9, 1e9), (-1e9, 1e9)];
            let mut slab = polygon_prism(vol.dims, axes, &all, from, to);
            slab.subtract(&inside);
            slab
        } else {
            inside
        };
        let (mode, fill) = match op {
            ScissorsOp::FillInside | ScissorsOp::FillOutside => (Mode::Add, true),
            ScissorsOp::EraseInside | ScissorsOp::EraseOutside => (Mode::Remove, false),
        };
        self.edit_now(vol, mask, mode, op.label(), fill);
    }

    /// Change the selected segment by `mask` now (a quick edit, not background work), with
    /// masking; `make` makes a segment when none is selected.
    pub(crate) fn edit_now(&mut self, vol: &LoadedVolume, mask: Mask, mode: Mode, what: &str, make: bool) {
        let Some(seg) = &mut self.seg else { return };
        if seg.selected.is_none() && !(make && seg.add_segment(None).is_some()) {
            self.say("Select a segment first", true);
            return;
        }
        let label = seg.selected_segment().map_or(1, |s| s.label);
        seg.checkpoint();
        let n = seg.apply(vol, &mask, label, mode, &self.masking);
        seg.changed(true);
        self.say(format!("{what}: {} voxels changed", polymix_workspace::style::format_count(n)), false);
    }

    /// Draw the shape in progress, the traced outline and the points placed over their views
    /// (`maps`: where each view drew its slice).
    pub(crate) fn paint_sketches(&self, ui: &egui::Ui, pointer: Option<Pos2>, maps: &[ViewMap]) {
        // Points on (or near) each view's slice: full on it, smaller and fainter within three
        // slices.
        for m in maps {
            let painter = ui.painter_at(m.rect);
            for p in &self.markers {
                let (at, off) = m.project(*p);
                if off.abs() <= 3.0 {
                    let (r, c) = if off.abs() <= 0.5 { (4.0, LINE) } else { (2.5, LINE.gamma_multiply(0.45)) };
                    painter.circle_filled(at, r, c);
                }
            }
        }
        if let Some(s) = &self.sketch {
            let painter = ui.painter_at(s.map.rect);
            let pts: Vec<Pos2> = s.points.iter().map(|&(a, d)| s.map.screen(a, d)).collect();
            let stroke = Stroke::new(1.5, LINE);
            if self.effect == Some(Effect::Threshold) && s.points.len() == 2 {
                painter.add(egui::Shape::line(pts, stroke));
            } else if self.effect == Some(Effect::Draw) {
                for &p in &pts {
                    painter.circle_filled(p, 2.5, LINE);
                }
                let mut line = pts.clone();
                // The rubber band to the pointer, while it is over the view.
                if let Some(p) = pointer.filter(|p| s.map.rect.contains(*p)) {
                    line.push(p);
                }
                painter.add(egui::Shape::line(line, stroke));
            } else if pts.len() >= 2 {
                painter.add(egui::Shape::closed_line(pts, stroke));
            }
        }
        // The local histogram's region, on its slice.
        if let (Some(l), Some(Effect::Threshold)) = (&self.local, self.effect) {
            for m in maps.iter().filter(|m| m.oblique.is_none() && m.plane == l.plane && m.slice == l.slice) {
                let pts: Vec<Pos2> = l.outline.iter().map(|&(a, d)| m.screen(a, d)).collect();
                let stroke = Stroke::new(1.5, LINE.gamma_multiply(0.8));
                let painter = ui.painter_at(m.rect);
                if l.line {
                    painter.add(egui::Shape::line(pts, stroke));
                } else {
                    painter.add(egui::Shape::closed_line(pts, stroke));
                }
            }
        }
        if let (Some(t), Some(Effect::LevelTracing)) = (&self.traced, self.effect) {
            let painter = ui.painter_at(t.map.rect);
            let m = &t.map;
            let (na, nd) = (self.volume.as_ref().map_or(0, |v| v.dims[m.axes.across]), self.volume.as_ref().map_or(0, |v| v.dims[m.axes.down]));
            let dims = self.volume.as_ref().map_or([0; 3], |v| v.dims);
            let stride = [1, dims[0], dims[0] * dims[1]];
            let mut inside = vec![false; na * nd];
            for &i in &t.voxels {
                let (a, d) = ((i / stride[m.axes.across]) % dims[m.axes.across], (i / stride[m.axes.down]) % dims[m.axes.down]);
                inside[d * na + a] = true;
            }
            let at = |a: i64, d: i64| a >= 0 && d >= 0 && (a as usize) < na && (d as usize) < nd && inside[d as usize * na + a as usize];
            let stroke = Stroke::new(1.5, LINE);
            // The region's outline: the edges between voxels in it and voxels not in it.
            for d in 0..nd as i64 {
                for a in 0..na as i64 {
                    if !at(a, d) {
                        continue;
                    }
                    let (x, y) = (a as f64, d as f64);
                    for (da, dd, p, q) in [(-1, 0, (x - 0.5, y - 0.5), (x - 0.5, y + 0.5)), (1, 0, (x + 0.5, y - 0.5), (x + 0.5, y + 0.5)), (0, -1, (x - 0.5, y - 0.5), (x + 0.5, y - 0.5)), (0, 1, (x - 0.5, y + 0.5), (x + 0.5, y + 0.5))] {
                        if !at(a + da, d + dd) {
                            painter.line_segment([m.screen(p.0, p.1), m.screen(q.0, q.1)], stroke);
                        }
                    }
                }
            }
        }
    }
}
