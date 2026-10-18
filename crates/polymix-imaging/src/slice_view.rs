//! One plane of a volume, for the three-plane views: the slice through the cursor in the
//! radiological convention, with the shared mouse controls ([`Nav`]), the other two planes
//! as crosshair lines in their colours, the patient directions at the edges, and labels drawn
//! over the image.
//!
//! Voxel coordinates are continuous, in the convention of the volume's affine: voxel
//! (i, j, k) is centred on (i, j, k) and covers ±0.5 around it, so rounding gives the voxel
//! and `geometry.world(cursor)` gives its position in the patient.

use crate::histogram;
use crate::nav::{drag_window, Nav};
use egui::{Align2, Color32, FontId, Pos2, Rect, Stroke, Vec2};
use glam::DVec3;
use mm_volume::load::LoadedVolume;
use mm_volume::slice::{plane_axes, Plane, PlaneAxes};

/// What the three views (and the 3D view) share: the voxel they go through and the window.
#[derive(Clone, Debug, PartialEq)]
pub struct ViewState {
    /// Voxel coordinates (i, j, k).
    pub cursor: DVec3,
    /// Window (centre, width).
    pub window: (f64, f64),
}

impl ViewState {
    /// The middle voxel of the volume, with the automatic window.
    pub fn new(vol: &LoadedVolume) -> Self {
        let mid = |a: usize| (vol.dims[a].saturating_sub(1) / 2) as f64;
        Self { cursor: DVec3::new(mid(0), mid(1), mid(2)), window: vol.auto_window }
    }
}

/// Labels drawn over the image.
pub struct LabelOverlay<'a> {
    /// One per voxel, in the volume's voxel order; 0 = no label.
    pub labels: &'a [u8],
    /// Colour of each label (index = label); label 0 is never drawn.
    pub colours: &'a [Color32],
    pub opacity: f32,
    /// Changes whenever the labels do: the view re-reads them only then.
    pub version: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SliceOptions {
    /// Left drag scrolls slices and a click moves the crosshair; off while a tool (e.g. a
    /// brush) uses the left button.
    pub left_drag_scrolls: bool,
    /// Lines where the other two planes cut this one.
    pub crosshair: bool,
    /// Interpolate between voxels (off: every voxel is a sharp square).
    pub smooth: bool,
    /// A brush of this radius (mm) follows the pointer, and Shift + wheel sizes it
    /// ([`SliceResponse::brush_wheel`]) instead of scrolling slices.
    pub brush_mm: Option<f32>,
}

impl Default for SliceOptions {
    fn default() -> Self {
        Self { left_drag_scrolls: true, crosshair: true, smooth: true, brush_mm: None }
    }
}

pub struct SliceResponse {
    pub response: egui::Response,
    /// The voxel under the pointer (over the image only).
    pub hover_voxel: Option<DVec3>,
    /// The voxel left-clicked (the cursor has moved there, unless a tool has the left
    /// button).
    pub clicked_voxel: Option<DVec3>,
    /// The voxel under a left drag, when the left button is a tool's (`left_drag_scrolls`
    /// off).
    pub left_drag_voxel: Option<DVec3>,
    /// Shift + wheel over the view while a brush is shown, in notches (positive = up).
    pub brush_wheel: f32,
    /// Double-clicked (the layout around the view decides what that does).
    pub double_clicked: bool,
}

/// View colours as in 3D Slicer: red axial, yellow sagittal, green coronal.
pub fn plane_colour(p: Plane) -> Color32 {
    match p {
        Plane::Axial => Color32::from_rgb(243, 74, 51),
        Plane::Sagittal => Color32::from_rgb(237, 213, 76),
        Plane::Coronal => Color32::from_rgb(110, 176, 75),
    }
}

/// Patient directions at the left, right, top and bottom edges of a plane's view.
fn edge_letters(p: Plane) -> [&'static str; 4] {
    match p {
        Plane::Axial => ["R", "L", "A", "P"],
        Plane::Coronal => ["R", "L", "S", "I"],
        Plane::Sagittal => ["A", "P", "S", "I"],
    }
}

/// The world axis (R, A, S) a plane's slices step along, and its letter.
fn slice_world_axis(p: Plane) -> (usize, &'static str) {
    match p {
        Plane::Sagittal => (0, "R"),
        Plane::Coronal => (1, "A"),
        Plane::Axial => (2, "S"),
    }
}

/// Voxel index nearest coordinate `x` on an axis of `n` voxels.
fn nearest(x: f64, n: usize) -> usize {
    x.round().clamp(0.0, n.saturating_sub(1) as f64) as usize
}

/// How a plane cuts a volume.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Grid {
    axes: PlaneAxes,
    /// Pixels across and down.
    size: [usize; 2],
    /// Millimetres per pixel across and down.
    mm: [f64; 2],
    /// Voxel step along the slice axis to the next slice (towards patient S, A or R): ±1.
    step: i64,
}

impl Grid {
    fn new(vol: &LoadedVolume, plane: Plane) -> Self {
        let g = &vol.geometry;
        let axes = plane_axes(g, plane);
        let step = if g.directions()[axes.slice][slice_world_axis(plane).0] >= 0.0 { 1 } else { -1 };
        Self { axes, size: [vol.dims[axes.across], vol.dims[axes.down]], mm: [g.spacing[axes.across], g.spacing[axes.down]], step }
    }

    /// Size of the plane in millimetres.
    fn content(&self) -> Vec2 {
        Vec2::new((self.size[0] as f64 * self.mm[0]) as f32, (self.size[1] as f64 * self.mm[1]) as f32)
    }

    /// Voxel coordinates across and down at image position (`u`, `v`) in pixels from the
    /// top left.
    fn voxel_at(&self, u: f64, v: f64) -> (f64, f64) {
        let a = if self.axes.flip_across { self.size[0] as f64 - u } else { u };
        let d = if self.axes.flip_down { self.size[1] as f64 - v } else { v };
        (a - 0.5, d - 0.5)
    }

    /// Image position (pixels from the left) of voxel coordinate `a` across.
    fn u_of(&self, a: f64) -> f64 {
        if self.axes.flip_across { self.size[0] as f64 - (a + 0.5) } else { a + 0.5 }
    }

    /// Image position (pixels from the top) of voxel coordinate `d` down.
    fn v_of(&self, d: f64) -> f64 {
        if self.axes.flip_down { self.size[1] as f64 - (d + 0.5) } else { d + 0.5 }
    }

    /// Values of slice `s` of a volume-ordered array, row by row from the top left.
    fn extract<T: Copy>(&self, values: &[T], dims: [usize; 3], s: usize) -> Vec<T> {
        let stride = [1, dims[0], dims[0] * dims[1]];
        let (sa, sd) = (stride[self.axes.across], stride[self.axes.down]);
        let base = s * stride[self.axes.slice];
        let [w, h] = self.size;
        let mut out = Vec::with_capacity(w * h);
        for y in 0..h {
            let d = if self.axes.flip_down { h - 1 - y } else { y };
            let row = base + d * sd;
            for x in 0..w {
                let a = if self.axes.flip_across { w - 1 - x } else { x };
                out.push(values[row + a * sa]);
            }
        }
        out
    }

    /// A full voxel position from coordinates across and down on slice `s`.
    fn voxel(&self, a: f64, d: f64, s: f64) -> DVec3 {
        let mut c = [0.0; 3];
        c[self.axes.across] = a;
        c[self.axes.down] = d;
        c[self.axes.slice] = s;
        DVec3::from(c)
    }
}

/// Take the wheel events of this frame from the input (so the slice controls do not see
/// them) and return them in notches. macOS turns Shift + wheel into a horizontal scroll, so
/// either direction counts.
fn take_wheel(ctx: &egui::Context) -> f32 {
    ctx.input_mut(|i| {
        let mut notches = 0.0;
        for e in &i.raw.events {
            if let egui::Event::MouseWheel { unit, delta, .. } = e {
                let d = if delta.y != 0.0 { delta.y } else { delta.x };
                notches += match unit {
                    egui::MouseWheelUnit::Point => d / 24.0,
                    _ => d,
                };
            }
        }
        let wheel = |e: &egui::Event| matches!(e, egui::Event::MouseWheel { .. });
        i.raw.events.retain(|e| !wheel(e));
        i.events.retain(|e| !wheel(e));
        i.smooth_scroll_delta = Vec2::ZERO;
        notches
    })
}

/// Identifies a volume (by its voxel array) so a new one is noticed.
type VolumeId = (usize, usize, [usize; 3]);

fn volume_id(vol: &LoadedVolume) -> VolumeId {
    (vol.values.as_ptr() as usize, vol.values.len(), vol.dims)
}

/// What the texture shows.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Shown {
    volume: VolumeId,
    slice: usize,
    window: (f64, f64),
    invert: bool,
    smooth: bool,
    /// Labels version and opacity.
    labels: Option<(u64, u32)>,
}

pub struct SliceView {
    pub plane: Plane,
    pub nav: Nav,
    texture: Option<egui::TextureHandle>,
    /// Slice values for (volume, slice).
    values_of: Option<(VolumeId, usize)>,
    values: Vec<f32>,
    /// Slice labels for (volume, slice, version).
    labels_of: Option<(VolumeId, usize, u64)>,
    labels: Vec<u8>,
    shown: Option<Shown>,
}

impl SliceView {
    pub fn new(plane: Plane) -> Self {
        Self { plane, nav: Nav::default(), texture: None, values_of: None, values: Vec::new(), labels_of: None, labels: Vec::new(), shown: None }
    }

    /// Show the plane through `state.cursor` in the space left in `ui`, and apply the mouse
    /// controls: they move the cursor (slices, click) and set the window in `state`.
    pub fn ui(&mut self, ui: &mut egui::Ui, vol: &LoadedVolume, state: &mut ViewState, overlay: Option<&LabelOverlay>, opts: &SliceOptions) -> SliceResponse {
        let rect = ui.available_rect_before_wrap();
        let response = ui.allocate_rect(rect, egui::Sense::click_and_drag());
        let grid = Grid::new(vol, self.plane);
        let axes = grid.axes;
        let n = vol.dims[axes.slice];
        let (primary, secondary) = ui.input(|i| (i.pointer.primary_down(), i.pointer.secondary_down()));
        let tool_drag = !opts.left_drag_scrolls && primary && !secondary && response.is_pointer_button_down_on();
        let brush_wheel = if opts.brush_mm.is_some() && response.hovered() && ui.input(|i| i.modifiers.shift) { take_wheel(ui.ctx()) } else { 0.0 };
        let ev = self.nav.handle(ui, &response);
        let img = self.nav.image_rect(rect, grid.content());
        // Pointer → voxel coordinates across and down (over the image only).
        let on_image = |p: Pos2| -> Option<(f64, f64)> {
            let u = (p.x - img.min.x) as f64 / img.width() as f64 * grid.size[0] as f64;
            let v = (p.y - img.min.y) as f64 / img.height() as f64 * grid.size[1] as f64;
            ((0.0..grid.size[0] as f64).contains(&u) && (0.0..grid.size[1] as f64).contains(&v)).then(|| grid.voxel_at(u, v))
        };

        // Slices (a tool's left drag does not scroll).
        if ev.slices != 0 && !tool_drag {
            let k = nearest(state.cursor[axes.slice], n) as i64 + ev.slices as i64 * grid.step;
            state.cursor[axes.slice] = k.clamp(0, n as i64 - 1) as f64;
        }
        if let Some(d) = ev.window {
            state.window = drag_window(state.window, d);
        }
        let slice = nearest(state.cursor[axes.slice], n);
        let at = |(a, d): (f64, f64)| grid.voxel(a, d, slice as f64);
        let clicked_voxel = ev.clicked.and_then(on_image).map(at);
        if let (Some(c), true) = (clicked_voxel, opts.left_drag_scrolls) {
            state.cursor = c;
        }
        let hover_voxel = ev.hover.and_then(on_image).map(at);
        let left_drag_voxel = if tool_drag { response.interact_pointer_pos().and_then(on_image).map(at) } else { None };

        self.update_texture(ui.ctx(), vol, &grid, slice, state.window, overlay, opts.smooth);
        self.paint(ui, rect, img, vol, &grid, state, opts, ev.hover);
        // Above the slice position in the bottom left corner.
        let key = (volume_id(vol), vol.range.0.to_bits(), vol.range.1.to_bits());
        histogram::show(ui, response.id, rect, rect.bottom() - 24.0, key, &vol.values, state.window, vol.invert, self.nav.windowing());
        SliceResponse { response, hover_voxel, clicked_voxel, left_drag_voxel, brush_wheel, double_clicked: ev.double_clicked }
    }

    /// Re-make the texture when what it shows has changed.
    #[allow(clippy::too_many_arguments)]
    fn update_texture(&mut self, ctx: &egui::Context, vol: &LoadedVolume, grid: &Grid, slice: usize, window: (f64, f64), overlay: Option<&LabelOverlay>, smooth: bool) {
        let volume = volume_id(vol);
        if self.values_of != Some((volume, slice)) {
            self.values = grid.extract(&vol.values, vol.dims, slice);
            self.values_of = Some((volume, slice));
        }
        let overlay = overlay.filter(|o| o.labels.len() == vol.values.len());
        if let Some(o) = overlay {
            if self.labels_of != Some((volume, slice, o.version)) {
                self.labels = grid.extract(o.labels, vol.dims, slice);
                self.labels_of = Some((volume, slice, o.version));
            }
        }
        let shown = Shown { volume, slice, window, invert: vol.invert, smooth, labels: overlay.map(|o| (o.version, o.opacity.to_bits())) };
        if self.shown == Some(shown) && self.texture.is_some() {
            return;
        }
        let (c, w) = window;
        let lo = c - w * 0.5;
        let scale = 255.0 / w.max(f64::MIN_POSITIVE);
        let grey = |x: f32| {
            let g = ((x as f64 - lo) * scale).round().clamp(0.0, 255.0) as u8;
            if vol.invert { 255 - g } else { g }
        };
        let pixels: Vec<Color32> = match overlay {
            Some(o) => {
                // Segments as Slicer draws them: a see-through fill and a solid outline.
                let fill = o.opacity.clamp(0.0, 1.0);
                let [w, h] = grid.size;
                let labels = &self.labels;
                (0..w * h)
                    .map(|p| {
                        let (g, l) = (grey(self.values[p]), labels[p]);
                        let Some(col) = o.colours.get(l as usize).filter(|c| l != 0 && c.a() > 0) else { return Color32::from_gray(g) };
                        let (x, y) = (p % w, p / w);
                        let edge = x == 0 || y == 0 || x + 1 == w || y + 1 == h || labels[p - 1] != l || labels[p + 1] != l || labels[p - w] != l || labels[p + w] != l;
                        let alpha = if edge { 1.0 } else { fill };
                        let mix = |c: u8| (g as f32 + (c as f32 - g as f32) * alpha).round() as u8;
                        Color32::from_rgb(mix(col.r()), mix(col.g()), mix(col.b()))
                    })
                    .collect()
            }
            None => self.values.iter().map(|&x| Color32::from_gray(grey(x))).collect(),
        };
        let image = egui::ColorImage::new(grid.size, pixels);
        let options = if smooth { egui::TextureOptions::LINEAR } else { egui::TextureOptions::NEAREST };
        match &mut self.texture {
            Some(t) => t.set(image, options),
            None => self.texture = Some(ctx.load_texture(format!("slice.{:?}", self.plane), image, options)),
        }
        self.shown = Some(shown);
    }

    #[allow(clippy::too_many_arguments)]
    fn paint(&self, ui: &egui::Ui, rect: Rect, img: Rect, vol: &LoadedVolume, grid: &Grid, state: &ViewState, opts: &SliceOptions, pointer: Option<Pos2>) {
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, Color32::BLACK);
        if let Some(t) = &self.texture {
            painter.image(t.id(), img, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);
        }
        if let (Some(r), Some(p)) = (opts.brush_mm, pointer) {
            let points_per_mm = img.width() / (grid.size[0] as f64 * grid.mm[0]) as f32;
            painter.circle_stroke(p, (r * points_per_mm).max(1.5), Stroke::new(1.5, Color32::from_rgb(255, 230, 120)));
        }
        let axes = grid.axes;
        if opts.crosshair {
            // The other planes, through the voxels they show.
            for other in Plane::ALL.into_iter().filter(|&p| p != self.plane) {
                let s = plane_axes(&vol.geometry, other).slice;
                let colour = plane_colour(other).gamma_multiply(0.7);
                let k = nearest(state.cursor[s], vol.dims[s]) as f64;
                if s == axes.across {
                    let x = img.min.x + (grid.u_of(k) / grid.size[0] as f64) as f32 * img.width();
                    painter.vline(x, rect.y_range(), Stroke::new(1.0, colour));
                } else if s == axes.down {
                    let y = img.min.y + (grid.v_of(k) / grid.size[1] as f64) as f32 * img.height();
                    painter.hline(rect.x_range(), y, Stroke::new(1.0, colour));
                }
            }
        }
        let font = FontId::proportional(12.0);
        let text = Color32::from_gray(200);
        let pad = 6.0;
        painter.text(rect.left_top() + Vec2::new(pad + 2.0, pad), Align2::LEFT_TOP, self.plane.label(), FontId::proportional(13.0), plane_colour(self.plane));
        // Patient directions at the edges.
        let [left, right, top, bottom] = edge_letters(self.plane);
        let letters = Color32::from_gray(160);
        painter.text(Pos2::new(rect.left() + pad, rect.center().y), Align2::LEFT_CENTER, left, font.clone(), letters);
        painter.text(Pos2::new(rect.right() - pad, rect.center().y), Align2::RIGHT_CENTER, right, font.clone(), letters);
        painter.text(Pos2::new(rect.center().x, rect.top() + pad), Align2::CENTER_TOP, top, font.clone(), letters);
        painter.text(Pos2::new(rect.center().x, rect.bottom() - pad), Align2::CENTER_BOTTOM, bottom, font.clone(), letters);
        // Slice position (as Slicer shows it) and number, and the window.
        let n = vol.dims[axes.slice];
        let slice = nearest(state.cursor[axes.slice], n);
        let mut voxel = state.cursor;
        voxel[axes.slice] = slice as f64;
        let (world_axis, letter) = slice_world_axis(self.plane);
        let mm = vol.geometry.world(voxel)[world_axis];
        painter.text(rect.left_bottom() + Vec2::new(pad + 2.0, -pad), Align2::LEFT_BOTTOM, format!("{letter} {mm:.1} mm   {} / {n}", slice + 1), font.clone(), text);
        let (c, w) = state.window;
        painter.text(rect.right_bottom() + Vec2::new(-pad - 2.0, -pad), Align2::RIGHT_BOTTOM, format!("W {w:.0}  L {c:.0}"), font, text);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_volume::geometry::Geometry;

    /// A world-space test object sampled on a grid with the given voxel axes (unit vectors
    /// in RAS) and spacing, centred on the origin: values tell R, A and S apart.
    fn phantom(axes: [DVec3; 3], dims: [usize; 3], spacing: [f64; 3]) -> LoadedVolume {
        let half = |a: usize| (dims[a] as f64 - 1.0) * 0.5 * spacing[a];
        let origin = -(axes[0] * half(0) + axes[1] * half(1) + axes[2] * half(2));
        let g = Geometry::from_axes([dims[0], dims[1], dims[2], 1], spacing, axes, origin);
        let mut values = Vec::with_capacity(dims.iter().product());
        for k in 0..dims[2] {
            for j in 0..dims[1] {
                for i in 0..dims[0] {
                    let p = g.world(DVec3::new(i as f64, j as f64, k as f64)).round();
                    values.push((p.x * 10_000.0 + p.y * 100.0 + p.z) as f32);
                }
            }
        }
        LoadedVolume::from_values("phantom", dims, values, g)
    }

    /// The same 4 × 6 × 8 mm object stored three ways: RAS, LPS (DICOM axial) and PIR (a
    /// sagittal acquisition).
    fn layouts() -> Vec<LoadedVolume> {
        let (x, y, z) = (DVec3::X, DVec3::Y, DVec3::Z);
        vec![phantom([x, y, z], [5, 7, 9], [1.0; 3]), phantom([-x, -y, z], [5, 7, 9], [1.0; 3]), phantom([-y, -z, x], [7, 9, 5], [1.0; 3])]
    }

    #[test]
    fn every_layout_shows_the_same_planes() {
        let vols = layouts();
        for plane in Plane::ALL {
            let shown: Vec<(Vec<f32>, [usize; 2])> = vols
                .iter()
                .map(|v| {
                    let g = Grid::new(v, plane);
                    let s = nearest(ViewState::new(v).cursor[g.axes.slice], v.dims[g.axes.slice]);
                    (g.extract(&v.values, v.dims, s), g.size)
                })
                .collect();
            assert_eq!(shown[0], shown[1], "{plane:?}: RAS and LPS differ");
            assert_eq!(shown[0], shown[2], "{plane:?}: RAS and PIR differ");
        }
    }

    #[test]
    fn edge_letters_match_the_image() {
        // Radiological convention: patient right on the left of axial and coronal views,
        // anterior on the left of sagittal ones, anterior or superior up.
        let direction = |letter: &str| match letter {
            "R" => (0, 1.0),
            "L" => (0, -1.0),
            "A" => (1, 1.0),
            "P" => (1, -1.0),
            "S" => (2, 1.0),
            _ => (2, -1.0),
        };
        for v in layouts() {
            for plane in Plane::ALL {
                let g = Grid::new(&v, plane);
                let (w, h) = (g.size[0] as f64, g.size[1] as f64);
                let world = |u: f64, y: f64| {
                    let (a, d) = g.voxel_at(u, y);
                    v.geometry.world(g.voxel(a, d, 0.0))
                };
                let [left, right, top, bottom] = edge_letters(plane);
                let (axis, sign) = direction(left);
                assert!((world(0.5, h / 2.0) - world(w - 0.5, h / 2.0))[axis] * sign > 0.0, "{plane:?} {left}");
                assert_eq!(direction(right), (axis, -sign));
                let (axis, sign) = direction(top);
                assert!((world(w / 2.0, 0.5) - world(w / 2.0, h - 0.5))[axis] * sign > 0.0, "{plane:?} {top}");
                assert_eq!(direction(bottom), (axis, -sign));
            }
        }
    }

    #[test]
    fn pixels_and_voxels_round_trip() {
        for v in layouts() {
            for plane in Plane::ALL {
                let g = Grid::new(&v, plane);
                for a in 0..g.size[0] {
                    for d in 0..g.size[1] {
                        let (u, w) = (g.u_of(a as f64), g.v_of(d as f64));
                        let (a2, d2) = g.voxel_at(u, w);
                        assert!((a2 - a as f64).abs() < 1e-9 && (d2 - d as f64).abs() < 1e-9);
                        // The pixel covering (u, w) holds voxel (a, d).
                        let img = g.extract(&v.values, v.dims, 0);
                        let p = g.voxel(a as f64, d as f64, 0.0);
                        assert_eq!(img[w.floor() as usize * g.size[0] + u.floor() as usize], v.get(p.x as usize, p.y as usize, p.z as usize));
                    }
                }
            }
        }
    }

    #[test]
    fn next_slice_goes_to_superior_anterior_right() {
        for v in layouts() {
            for plane in Plane::ALL {
                let g = Grid::new(&v, plane);
                let mut p = DVec3::new(2.0, 2.0, 2.0);
                let before = v.geometry.world(p);
                p[g.axes.slice] += g.step as f64;
                let moved = v.geometry.world(p) - before;
                assert!(moved[slice_world_axis(plane).0] > 0.0, "{plane:?}");
            }
        }
    }
}
