//! One plane of a volume, for the three-plane views: the slice through the cursor in the
//! radiological convention, with the shared mouse controls ([`Nav`]), the other two planes
//! as crosshair lines in their colours, the patient directions at the edges, and labels drawn
//! over the image; in a colour map, as a thick slab, and with a second image over it
//! ([`crate::layers`]).
//!
//! Voxel coordinates are continuous, in the convention of the volume's affine: voxel
//! (i, j, k) is centred on (i, j, k) and covers ±0.5 around it, so rounding gives the voxel
//! and `geometry.world(cursor)` gives its position in the patient.

use crate::histogram;
use crate::layers::{ColourMap, Compare, Fusion, Slab, SlabMode};
use crate::nav::{drag_window, Nav};
use crate::oblique::{direction_letter, rotated, standard_frame, Frame, ObliqueGrid};
use egui::{Align2, Color32, FontId, Pos2, Rect, Stroke, Vec2};
use glam::DVec3;
use mm_volume::load::LoadedVolume;
use mm_volume::slice::{plane_axes, Plane, PlaneAxes};

/// The three planes in the order the image tabs show them.
pub const PLANES: [Plane; 3] = [Plane::Axial, Plane::Sagittal, Plane::Coronal];

/// What the three views (and the 3D view) share: the voxel they go through, the window and
/// how values are shown.
#[derive(Clone, Debug, PartialEq)]
pub struct ViewState {
    /// Voxel coordinates (i, j, k).
    pub cursor: DVec3,
    /// Window (centre, width).
    pub window: (f64, f64),
    pub colour_map: ColourMap,
    /// Thick slabs in the slice views.
    pub slab: Slab,
    /// A second image on the same grid, over this one or compared with it.
    pub fusion: Option<Fusion>,
    /// Oblique planes (axial, sagittal, coronal views' frames in RAS); `None`: the views
    /// follow the voxel grid.
    pub frames: Option<[Frame; 3]>,
}

impl ViewState {
    /// The middle voxel of the volume, with the automatic window.
    pub fn new(vol: &LoadedVolume) -> Self {
        let mid = |a: usize| (vol.dims[a].saturating_sub(1) / 2) as f64;
        Self { cursor: DVec3::new(mid(0), mid(1), mid(2)), window: vol.auto_window, colour_map: ColourMap::Grey, slab: Slab::default(), fusion: None, frames: None }
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
    /// Where voxels of the slice shown were drawn this frame (to draw over the view).
    pub map: ViewMap,
}

/// Where a slice view drew its slice: screen positions of voxel coordinates on it, and back.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewMap {
    pub plane: Plane,
    /// The voxel axes across, down and through the slice, and whether across / down run
    /// against the screen's.
    pub axes: PlaneAxes,
    /// The slice shown (along `axes.slice`), and the voxel step along that axis towards the
    /// patient's superior, anterior or right (±1: the slice's normal).
    pub slice: usize,
    pub step: i64,
    /// The view, and the image within it.
    pub rect: Rect,
    pub image: Rect,
    size: [usize; 2],
    /// An oblique plane (the view does not follow the voxel grid: `axes`, `slice` and
    /// `screen` / `voxel` do not apply; use `project` / `voxel_at`).
    pub oblique: Option<ObliqueGrid>,
}

impl ViewMap {
    /// Screen position of voxel coordinates (across, down) on the slice.
    pub fn screen(&self, across: f64, down: f64) -> Pos2 {
        let u = if self.axes.flip_across { self.size[0] as f64 - (across + 0.5) } else { across + 0.5 };
        let v = if self.axes.flip_down { self.size[1] as f64 - (down + 0.5) } else { down + 0.5 };
        Pos2::new(self.image.min.x + (u / self.size[0] as f64) as f32 * self.image.width(), self.image.min.y + (v / self.size[1] as f64) as f32 * self.image.height())
    }

    /// Voxel coordinates (across, down) on the slice at a screen position (anywhere, not
    /// only over the image).
    pub fn voxel(&self, p: Pos2) -> (f64, f64) {
        let u = (p.x - self.image.min.x) as f64 / self.image.width() as f64 * self.size[0] as f64;
        let v = (p.y - self.image.min.y) as f64 / self.image.height() as f64 * self.size[1] as f64;
        let a = if self.axes.flip_across { self.size[0] as f64 - u } else { u };
        let d = if self.axes.flip_down { self.size[1] as f64 - v } else { v };
        (a - 0.5, d - 0.5)
    }

    /// Screen points per millimetre (across).
    pub fn points_per_mm(&self, spacing: [f64; 3]) -> f32 {
        match &self.oblique {
            Some(g) => self.image.width() / (self.size[0] as f64 * g.px) as f32,
            None => self.image.width() / (self.size[0] as f64 * spacing[self.axes.across]) as f32,
        }
    }

    /// Screen position of voxel coordinates `v`, and how far `v` lies from the plane shown
    /// (in slices: voxels along the grid, or pixels of an oblique plane).
    pub fn project(&self, v: DVec3) -> (Pos2, f64) {
        match &self.oblique {
            Some(g) => {
                let (u, w, d) = g.pixel(g.to_world.transform_point3(v));
                (Pos2::new(self.image.min.x + (u / self.size[0] as f64) as f32 * self.image.width(), self.image.min.y + (w / self.size[1] as f64) as f32 * self.image.height()), d)
            }
            None => (self.screen(v[self.axes.across], v[self.axes.down]), v[self.axes.slice] - self.slice as f64),
        }
    }

    /// Voxel coordinates of the point of the plane shown at a screen position.
    pub fn voxel_at(&self, p: Pos2) -> DVec3 {
        let u = (p.x - self.image.min.x) as f64 / self.image.width() as f64 * self.size[0] as f64;
        let v = (p.y - self.image.min.y) as f64 / self.image.height() as f64 * self.size[1] as f64;
        match &self.oblique {
            Some(g) => g.to_voxel.transform_point3(g.world(u, v)),
            None => {
                let (a, d) = self.voxel(p);
                let mut c = [0.0; 3];
                c[self.axes.across] = a;
                c[self.axes.down] = d;
                c[self.axes.slice] = self.slice as f64;
                DVec3::from(c)
            }
        }
    }

    /// Over the image (not the view's margins).
    fn on_image(&self, p: Pos2) -> bool {
        let (u, v) = ((p.x - self.image.min.x) / self.image.width(), (p.y - self.image.min.y) / self.image.height());
        (0.0..1.0).contains(&u) && (0.0..1.0).contains(&v)
    }
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
    #[cfg(test)]
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

    /// Slice `s` of volume-ordered values, or the slab of `half` slices on each side of it
    /// combined as `mode`, row by row from the top left.
    fn extract_slab(&self, values: &[f32], dims: [usize; 3], s: usize, half: usize, mode: SlabMode) -> Vec<f32> {
        let mut out = self.extract(values, dims, s);
        if half == 0 || mode == SlabMode::Off {
            return out;
        }
        let n = dims[self.axes.slice];
        let (from, to) = (s.saturating_sub(half), (s + half).min(n - 1));
        for k in (from..=to).filter(|&k| k != s) {
            let other = self.extract(values, dims, k);
            for (o, &x) in out.iter_mut().zip(&other) {
                *o = match mode {
                    SlabMode::Max => o.max(x),
                    SlabMode::Min => o.min(x),
                    _ => *o + x,
                };
            }
        }
        if mode == SlabMode::Mean {
            let count = (to - from + 1) as f32;
            out.iter_mut().for_each(|o| *o /= count);
        }
        out
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
    #[cfg(test)]
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

/// Which image of a volume a view shows: a slice of the voxel grid, or an oblique plane (its
/// frame, origin and pixel size, as bits).
#[derive(Clone, Copy, Debug, PartialEq)]
enum At {
    Slice(usize),
    Oblique([u64; 13]),
}

fn at_oblique(g: &ObliqueGrid) -> At {
    let f = g.frame;
    At::Oblique([f[0].x, f[0].y, f[0].z, f[1].x, f[1].y, f[1].z, f[2].x, f[2].y, f[2].z, g.origin.x, g.origin.y, g.origin.z, g.px].map(f64::to_bits))
}

/// What the texture shows.
#[derive(Clone, Debug, PartialEq)]
struct Shown {
    volume: VolumeId,
    at: At,
    window: (f64, f64),
    invert: bool,
    smooth: bool,
    /// Labels version and opacity.
    labels: Option<(u64, u32)>,
    colour_map: ColourMap,
    slab: Slab,
    fusion: Option<Fusion>,
}

/// Which slab of which volume a cached slice holds (and whether it was interpolated).
type SliceKey = (VolumeId, At, usize, SlabMode, bool);

pub struct SliceView {
    pub plane: Plane,
    pub nav: Nav,
    texture: Option<egui::TextureHandle>,
    /// Slice values for (volume, slice, slab).
    values_of: Option<SliceKey>,
    values: Vec<f32>,
    /// The second image's slice values, for (its values, slice, slab).
    fusion_of: Option<(usize, At, usize, SlabMode, bool)>,
    fusion_values: Vec<f32>,
    /// Slice labels for (volume, slice, version).
    labels_of: Option<(VolumeId, At, u64)>,
    labels: Vec<u8>,
    shown: Option<Shown>,
}

impl SliceView {
    pub fn new(plane: Plane) -> Self {
        Self { plane, nav: Nav::default(), texture: None, values_of: None, values: Vec::new(), fusion_of: None, fusion_values: Vec::new(), labels_of: None, labels: Vec::new(), shown: None }
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
        let index = crate::PLANES.iter().position(|p| *p == self.plane).unwrap_or(0);

        // Slices (a tool's left drag does not scroll): along the grid, or along an oblique
        // plane's normal a pixel at a time.
        if ev.slices != 0 && !tool_drag {
            match state.frames {
                Some(f) => {
                    let g = ObliqueGrid::new(vol, f[index], state.cursor);
                    let w = g.to_world.transform_point3(state.cursor) + f[index][2] * (g.px * ev.slices as f64);
                    let v = g.to_voxel.transform_point3(w);
                    state.cursor = DVec3::from([0, 1, 2].map(|a| v[a].clamp(0.0, vol.dims[a] as f64 - 1.0)));
                }
                None => {
                    let k = nearest(state.cursor[axes.slice], n) as i64 + ev.slices as i64 * grid.step;
                    state.cursor[axes.slice] = k.clamp(0, n as i64 - 1) as f64;
                }
            }
        }
        if let Some(d) = ev.window {
            state.window = drag_window(state.window, d);
        }
        let slice = nearest(state.cursor[axes.slice], n);
        let oblique = state.frames.map(|f| ObliqueGrid::new(vol, f[index], state.cursor));
        let (size, content) = match &oblique {
            Some(g) => (g.size, Vec2::new((g.size[0] as f64 * g.px) as f32, (g.size[1] as f64 * g.px) as f32)),
            None => (grid.size, grid.content()),
        };
        let img = self.nav.image_rect(rect, content);
        let map = ViewMap { plane: self.plane, axes, slice, step: grid.step, rect, image: img, size, oblique };
        // ⌥ + left drag turns the other two planes about this one's normal, around the cursor
        // (the crosshair lines follow the pointer).
        if let Some((a, b)) = ev.rotate {
            let c = map.project(state.cursor).0;
            let turn = (b - c).angle() - (a - c).angle();
            let mut f = state.frames.unwrap_or(crate::PLANES.map(standard_frame));
            let axis = f[index][2];
            for (j, frame) in f.iter_mut().enumerate().filter(|(j, _)| *j != index) {
                let _ = j;
                *frame = rotated(*frame, axis, turn as f64);
            }
            state.frames = Some(f);
        }
        // Pointer → voxel coordinates (over the image only).
        let at = |p: Pos2| map.on_image(p).then(|| map.voxel_at(p));
        let clicked_voxel = ev.clicked.and_then(at);
        if let (Some(c), true) = (clicked_voxel, opts.left_drag_scrolls) {
            state.cursor = c;
        }
        let hover_voxel = ev.hover.and_then(at);
        let left_drag_voxel = if tool_drag { response.interact_pointer_pos().and_then(at) } else { None };

        self.update_texture(ui.ctx(), vol, &grid, &map, state, overlay, opts.smooth);
        self.paint(ui, &map, vol, &grid, state, opts, ev.hover);
        // Above the slice position in the bottom left corner.
        let key = (volume_id(vol), vol.range.0.to_bits(), vol.range.1.to_bits());
        histogram::show(ui, response.id, rect, rect.bottom() - 24.0, key, &vol.values, state.window, vol.invert, self.nav.windowing());
        SliceResponse { response, hover_voxel, clicked_voxel, left_drag_voxel, brush_wheel, double_clicked: ev.double_clicked, map }
    }

    /// Re-make the texture when what it shows has changed.
    #[allow(clippy::too_many_arguments)]
    fn update_texture(&mut self, ctx: &egui::Context, vol: &LoadedVolume, grid: &Grid, map: &ViewMap, state: &ViewState, overlay: Option<&LabelOverlay>, smooth: bool) {
        let volume = volume_id(vol);
        let window = state.window;
        let slice = map.slice;
        let mode = state.slab.mode;
        // The image's values, the second image's and the labels: a slice (slab) of the grid,
        // or sampled on the oblique plane.
        let (at, half, half_mm, interpolated) = match &map.oblique {
            Some(g) => (at_oblique(g), 0, state.slab.mm * 0.5, smooth),
            None => (At::Slice(slice), state.slab.half(vol.geometry.spacing[grid.axes.slice]), 0.0, false),
        };
        let take = |values: &[f32]| match &map.oblique {
            Some(g) => g.sample(vol, values, smooth, half_mm, mode),
            None => grid.extract_slab(values, vol.dims, slice, half, mode),
        };
        if self.values_of != Some((volume, at, half, mode, interpolated)) {
            self.values = take(&vol.values);
            self.values_of = Some((volume, at, half, mode, interpolated));
        }
        let fusion = state.fusion.as_ref().filter(|f| f.values.len() == vol.values.len());
        if let Some(f) = fusion {
            let key = (f.values.as_ptr() as usize, at, half, mode, interpolated);
            if self.fusion_of != Some(key) {
                self.fusion_values = take(&f.values);
                self.fusion_of = Some(key);
            }
        }
        let overlay = overlay.filter(|o| o.labels.len() == vol.values.len());
        if let Some(o) = overlay {
            if self.labels_of != Some((volume, at, o.version)) {
                self.labels = match &map.oblique {
                    Some(g) => g.sample_labels(vol, o.labels),
                    None => grid.extract(o.labels, vol.dims, slice),
                };
                self.labels_of = Some((volume, at, o.version));
            }
        }
        let shown = Shown { volume, at, window, invert: vol.invert, smooth, labels: overlay.map(|o| (o.version, o.opacity.to_bits())), colour_map: state.colour_map, slab: state.slab, fusion: fusion.cloned() };
        if self.shown.as_ref() == Some(&shown) && self.texture.is_some() {
            return;
        }
        let level = |(c, w): (f64, f64), invert: bool| {
            let (lo, scale) = (c - w * 0.5, 255.0 / w.max(f64::MIN_POSITIVE));
            move |x: f32| {
                let g = ((x as f64 - lo) * scale).round().clamp(0.0, 255.0) as u8;
                if invert { 255 - g } else { g }
            }
        };
        let (grey, lut) = (level(window, vol.invert), state.colour_map.lut());
        // The image's colour, with the second image over it as compared.
        let [w, h] = map.size;
        let base: Vec<Color32> = match fusion {
            Some(f) => {
                let (fgrey, flut) = (level(f.window, false), f.colour_map.lut());
                let divider = (f.wipe.clamp(0.0, 1.0) * w as f32).round() as usize;
                self.values
                    .iter()
                    .zip(&self.fusion_values)
                    .enumerate()
                    .map(|(p, (&x, &y))| {
                        let (a, b) = (lut[grey(x) as usize], flut[fgrey(y) as usize]);
                        match f.compare {
                            Compare::Blend => a.lerp_to_gamma(b, f.opacity.clamp(0.0, 1.0)),
                            Compare::Wipe => {
                                if p % w < divider {
                                    a
                                } else {
                                    b
                                }
                            }
                            Compare::Toggle => {
                                if f.showing {
                                    b
                                } else {
                                    a
                                }
                            }
                        }
                    })
                    .collect()
            }
            None => self.values.iter().map(|&x| lut[grey(x) as usize]).collect(),
        };
        let pixels: Vec<Color32> = match overlay {
            Some(o) => {
                // Segments as Slicer draws them: a see-through fill and a solid outline.
                let fill = o.opacity.clamp(0.0, 1.0);
                let labels = &self.labels;
                (0..w * h)
                    .map(|p| {
                        let (g, l) = (base[p], labels[p]);
                        let Some(col) = o.colours.get(l as usize).filter(|c| l != 0 && c.a() > 0) else { return g };
                        let (x, y) = (p % w, p / w);
                        let edge = x == 0 || y == 0 || x + 1 == w || y + 1 == h || labels[p - 1] != l || labels[p + 1] != l || labels[p - w] != l || labels[p + w] != l;
                        let alpha = if edge { 1.0 } else { fill };
                        let mix = |g: u8, c: u8| (g as f32 + (c as f32 - g as f32) * alpha).round() as u8;
                        Color32::from_rgb(mix(g.r(), col.r()), mix(g.g(), col.g()), mix(g.b(), col.b()))
                    })
                    .collect()
            }
            None => base,
        };
        let image = egui::ColorImage::new(map.size, pixels);
        let options = if smooth { egui::TextureOptions::LINEAR } else { egui::TextureOptions::NEAREST };
        match &mut self.texture {
            Some(t) => t.set(image, options),
            None => self.texture = Some(ctx.load_texture(format!("slice.{:?}", self.plane), image, options)),
        }
        self.shown = Some(shown);
    }

    #[allow(clippy::too_many_arguments)]
    fn paint(&self, ui: &egui::Ui, map: &ViewMap, vol: &LoadedVolume, grid: &Grid, state: &ViewState, opts: &SliceOptions, pointer: Option<Pos2>) {
        let (rect, img) = (map.rect, map.image);
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, Color32::BLACK);
        if let Some(t) = &self.texture {
            painter.image(t.id(), img, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);
        }
        // The wipe's divider.
        if let Some(f) = state.fusion.as_ref().filter(|f| f.compare == Compare::Wipe) {
            let x = img.min.x + img.width() * ((f.wipe * map.size[0] as f32).round() / map.size[0] as f32);
            painter.vline(x, img.y_range(), Stroke::new(1.5, Color32::from_rgb(255, 230, 120)));
        }
        if let (Some(r), Some(p)) = (opts.brush_mm, pointer) {
            painter.circle_stroke(p, (r * map.points_per_mm(vol.geometry.spacing)).max(1.5), Stroke::new(1.5, Color32::from_rgb(255, 230, 120)));
        }
        let axes = grid.axes;
        if opts.crosshair {
            match (&map.oblique, &state.frames) {
                // Where the other planes cut this one: lines through the cursor.
                (Some(g), Some(frames)) => {
                    let c = map.project(state.cursor).0;
                    let reach = rect.width().max(rect.height()) * 2.0;
                    for (k, other) in crate::PLANES.into_iter().enumerate().filter(|(_, p)| *p != self.plane) {
                        let d = g.frame[2].cross(frames[k][2]);
                        let dir = Vec2::new(d.dot(g.frame[0]) as f32, d.dot(g.frame[1]) as f32);
                        if dir.length() < 1e-6 {
                            continue;
                        }
                        let dir = dir.normalized() * reach;
                        painter.line_segment([c - dir, c + dir], Stroke::new(1.0, plane_colour(other).gamma_multiply(0.7)));
                    }
                }
                // The other planes, through the voxels they show.
                _ => {
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
            }
        }
        let font = FontId::proportional(12.0);
        let text = Color32::from_gray(200);
        let pad = 6.0;
        let title = if map.oblique.is_some() { format!("{} (oblique)", self.plane.label()) } else { self.plane.label().to_string() };
        painter.text(rect.left_top() + Vec2::new(pad + 2.0, pad), Align2::LEFT_TOP, title, FontId::proportional(13.0), plane_colour(self.plane));
        // Patient directions at the edges.
        let [left, right, top, bottom] = match &map.oblique {
            Some(g) => [direction_letter(-g.frame[0]), direction_letter(g.frame[0]), direction_letter(-g.frame[1]), direction_letter(g.frame[1])],
            None => edge_letters(self.plane),
        };
        let letters = Color32::from_gray(160);
        painter.text(Pos2::new(rect.left() + pad, rect.center().y), Align2::LEFT_CENTER, left, font.clone(), letters);
        painter.text(Pos2::new(rect.right() - pad, rect.center().y), Align2::RIGHT_CENTER, right, font.clone(), letters);
        painter.text(Pos2::new(rect.center().x, rect.top() + pad), Align2::CENTER_TOP, top, font.clone(), letters);
        painter.text(Pos2::new(rect.center().x, rect.bottom() - pad), Align2::CENTER_BOTTOM, bottom, font.clone(), letters);
        // Slice position (as Slicer shows it) and number, and the window.
        let position = match &map.oblique {
            Some(g) => {
                let w = vol.geometry.world(state.cursor);
                let slab = if state.slab.mode == SlabMode::Off { String::new() } else { format!("   {} {:.0} mm", state.slab.mode.label(), state.slab.mm.max(g.px)) };
                format!("R {:.1}  A {:.1}  S {:.1} mm{slab}", w.x, w.y, w.z)
            }
            None => {
                let n = vol.dims[axes.slice];
                let slice = nearest(state.cursor[axes.slice], n);
                let mut voxel = state.cursor;
                voxel[axes.slice] = slice as f64;
                let (world_axis, letter) = slice_world_axis(self.plane);
                let mm = vol.geometry.world(voxel)[world_axis];
                let slab = match state.slab.mode {
                    SlabMode::Off => String::new(),
                    m => format!("   {} {:.0} mm", m.label(), (2 * state.slab.half(vol.geometry.spacing[axes.slice]) + 1) as f64 * vol.geometry.spacing[axes.slice]),
                };
                format!("{letter} {mm:.1} mm   {} / {n}{slab}", slice + 1)
            }
        };
        painter.text(rect.left_bottom() + Vec2::new(pad + 2.0, -pad), Align2::LEFT_BOTTOM, position, font.clone(), text);
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

    #[test]
    fn slabs_combine_the_slices_around_the_cursor() {
        // Axial slices whose values are their slice number (k).
        let g = Geometry::from_axes([3, 3, 6, 1], [1.0; 3], [DVec3::X, DVec3::Y, DVec3::Z], DVec3::ZERO);
        let vol = LoadedVolume::from_values("t", [3, 3, 6], (0..54).map(|n| (n / 9) as f32).collect(), g);
        let grid = Grid::new(&vol, Plane::Axial);
        assert!(grid.extract_slab(&vol.values, vol.dims, 2, 0, SlabMode::Max).iter().all(|&v| v == 2.0));
        assert!(grid.extract_slab(&vol.values, vol.dims, 2, 1, SlabMode::Max).iter().all(|&v| v == 3.0));
        assert!(grid.extract_slab(&vol.values, vol.dims, 2, 1, SlabMode::Min).iter().all(|&v| v == 1.0));
        assert!(grid.extract_slab(&vol.values, vol.dims, 2, 1, SlabMode::Mean).iter().all(|&v| v == 2.0));
        // At the edge the slab is cut short.
        assert!(grid.extract_slab(&vol.values, vol.dims, 5, 2, SlabMode::Mean).iter().all(|&v| v == 4.0));
    }

    #[test]
    fn oblique_maps_round_trip() {
        let g = Geometry::from_axes([20, 20, 20, 1], [1.0, 1.0, 2.0], [DVec3::X, DVec3::Y, DVec3::Z], DVec3::new(-10.0, -10.0, -20.0));
        let vol = LoadedVolume::from_values("t", [20, 20, 20], vec![0.0; 8000], g);
        let frame = crate::oblique::rotated(crate::oblique::standard_frame(Plane::Coronal), DVec3::new(1.0, 1.0, 0.0).normalize(), 0.4);
        let cursor = DVec3::new(7.0, 9.0, 11.0);
        let ob = ObliqueGrid::new(&vol, frame, cursor);
        let grid = Grid::new(&vol, Plane::Coronal);
        let rect = Rect::from_min_size(Pos2::ZERO, Vec2::new(300.0, 300.0));
        let map = ViewMap { plane: Plane::Coronal, axes: grid.axes, slice: 0, step: 1, rect, image: rect, size: ob.size, oblique: Some(ob) };
        // The cursor lies on the plane; a screen point maps to a voxel on it and back.
        let (p, d) = map.project(cursor);
        assert!(d.abs() < 1e-9);
        assert!((map.voxel_at(p) - cursor).length() < 1e-3, "through screen points (f32)");
        let q = Pos2::new(120.0, 80.0);
        let (back, d) = map.project(map.voxel_at(q));
        assert!(back.distance(q) < 1e-3 && d.abs() < 1e-6);
    }
}
