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
use rayon::prelude::*;
use std::ops::{Range, RangeInclusive};

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

/// Where the labels changed between two versions of a [`LabelOverlay`]: from version `since`
/// to version `until`, only voxels in `lo..=hi` (voxel indices, inclusive) differ, in the same
/// labels array, and nothing else drawn (the colours) changed. Given to
/// [`SliceView::set_label_dirty`], it lets a view re-read and recolour just the part of its
/// image those voxels cover. Both versions are the overlay's own, as the view saw them; a box
/// that does not lead from the view's version to the overlay's version now is not used.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LabelsDirty {
    pub since: u64,
    pub until: u64,
    pub lo: [usize; 3],
    pub hi: [usize; 3],
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
    /// ⌘ / Ctrl + right drag is a tool's ([`SliceResponse::tool_drag`]) instead of a pan.
    pub tool_secondary_drag: bool,
}

impl Default for SliceOptions {
    fn default() -> Self {
        Self { left_drag_scrolls: true, crosshair: true, smooth: true, brush_mm: None, tool_secondary_drag: false }
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
    /// A tool's ⌘ / Ctrl + right drag (`tool_secondary_drag` on), with the voxel it was
    /// pressed on (over the image only).
    pub tool_drag: Option<(crate::ToolDrag, Option<DVec3>)>,
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

/// Work (in about nanoseconds on one thread) worth splitting across threads: a job's tasks
/// wait for threads to take them, which on a busy machine takes longer than doing it all here
/// (an oblique view's colours with a second image, 3 ms here, took three times as long split)
/// until the job outlasts the time the system lets a thread run at a stretch. So only jobs of
/// several milliseconds or more are split (sagittal slabs of 30 mm or more), in tasks of half
/// a millisecond or more; every slice's colours are worked out on the calling thread.
const PARALLEL_WORK: usize = 8_000_000;
const TASK_WORK: usize = 500_000;

/// The cost of a pixel's colour (in the units of [`PARALLEL_WORK`]): its value through the
/// colour map, and a second image's.
fn colour_cost(fusion: bool) -> usize {
    if fusion {
        5
    } else {
        1
    }
}

/// `f` on each row of `out` (`w` pixels wide, each costing `cost`) with its number: in
/// parallel tasks for big jobs.
fn for_rows<T: Send>(out: &mut [T], w: usize, cost: usize, f: impl Fn(usize, &mut [T]) + Sync) {
    let w = w.max(1);
    if out.len() * cost >= PARALLEL_WORK {
        out.par_chunks_mut(w).with_min_len(TASK_WORK.div_ceil(w * cost)).enumerate().for_each(|(y, r)| f(y, r));
    } else {
        out.chunks_mut(w).enumerate().for_each(|(y, r)| f(y, r));
    }
}

/// As [`for_rows`], over the same rows of two images.
fn for_rows2<A: Send, B: Send>(a: &mut [A], b: &mut [B], w: usize, cost: usize, f: impl Fn(usize, &mut [A], &mut [B]) + Sync) {
    let w = w.max(1);
    if a.len() * cost >= PARALLEL_WORK {
        a.par_chunks_mut(w).zip(b.par_chunks_mut(w)).with_min_len(TASK_WORK.div_ceil(w * cost)).enumerate().for_each(|(y, (a, b))| f(y, a, b));
    } else {
        a.chunks_mut(w).zip(b.chunks_mut(w)).enumerate().for_each(|(y, (a, b))| f(y, a, b));
    }
}

/// A window (centre, width) as grey levels.
#[derive(Clone, Copy)]
struct Level {
    lo: f64,
    scale: f64,
    invert: bool,
}

impl Level {
    fn new((c, w): (f64, f64), invert: bool) -> Self {
        Self { lo: c - w * 0.5, scale: 255.0 / w.max(f64::MIN_POSITIVE), invert }
    }

    fn grey(&self, x: f32) -> u8 {
        let g = ((x as f64 - self.lo) * self.scale).round().clamp(0.0, 255.0) as u8;
        if self.invert {
            255 - g
        } else {
            g
        }
    }
}

/// How the slice's values become colours: the window and colour map, and the second image's
/// as compared with the first.
struct Palette<'a> {
    level: Level,
    lut: [Color32; 256],
    second: Option<Second<'a>>,
}

struct Second<'a> {
    values: &'a [f32],
    level: Level,
    lut: [Color32; 256],
    compare: Compare,
    opacity: f32,
    /// Wipe: pixels left of this column show the first image.
    divider: usize,
    showing: bool,
}

impl Palette<'_> {
    /// The colours of row `y` (its pixels' `values`) into `out`.
    fn row(&self, y: usize, values: &[f32], out: &mut [Color32]) {
        match &self.second {
            // The colour map alone: a loop the compiler can tighten.
            None => out.iter_mut().zip(values).for_each(|(c, &v)| *c = self.lut[self.level.grey(v) as usize]),
            Some(_) => out.iter_mut().zip(values).enumerate().for_each(|(x, (c, &v))| *c = self.colour(y * values.len() + x, x, v)),
        }
    }

    /// The colour of pixel `p` (in column `x`) of value `v`.
    fn colour(&self, p: usize, x: usize, v: f32) -> Color32 {
        let a = self.lut[self.level.grey(v) as usize];
        let Some(f) = &self.second else { return a };
        let b = f.lut[f.level.grey(f.values[p]) as usize];
        match f.compare {
            Compare::Blend => a.lerp_to_gamma(b, f.opacity),
            Compare::Wipe => {
                if x < f.divider {
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
    }
}

/// Segments as Slicer draws them over the image: a see-through fill and a solid outline.
struct Outline<'a> {
    /// The slice's labels, row by row.
    labels: &'a [u8],
    colours: &'a [Color32],
    fill: f32,
    size: [usize; 2],
}

impl Outline<'_> {
    /// Row `y` with the labels over its colours `base`, into `out` (most pixels have no label
    /// and keep their colour).
    fn row(&self, y: usize, base: &[Color32], out: &mut [Color32]) {
        out.copy_from_slice(base);
        let w = self.size[0];
        for (x, &l) in self.labels[y * w..][..w].iter().enumerate() {
            if l != 0 {
                out[x] = self.colour(base[x], x, y);
            }
        }
    }

    /// The colour of pixel (`x`, `y`), `g` without the labels. A pixel's colour depends on its
    /// own label and its four neighbours' alone.
    fn colour(&self, g: Color32, x: usize, y: usize) -> Color32 {
        let ([w, h], labels) = (self.size, self.labels);
        let p = y * w + x;
        let l = labels[p];
        let Some(col) = self.colours.get(l as usize).filter(|c| l != 0 && c.a() > 0) else { return g };
        let edge = x == 0 || y == 0 || x + 1 == w || y + 1 == h || labels[p - 1] != l || labels[p + 1] != l || labels[p - w] != l || labels[p + w] != l;
        let alpha = if edge { 1.0 } else { self.fill };
        let mix = |g: u8, c: u8| (g as f32 + (c as f32 - g as f32) * alpha).round() as u8;
        Color32::from_rgb(mix(g.r(), col.r()), mix(g.g(), col.g()), mix(g.b(), col.b()))
    }
}

/// Part of a slice's image: columns `x` and rows `y`.
#[derive(Clone, Debug, PartialEq)]
struct Span {
    x: Range<usize>,
    y: Range<usize>,
}

impl Span {
    /// One more pixel on every side, within an image of `size`.
    fn grown(&self, size: [usize; 2]) -> Span {
        Span { x: self.x.start.saturating_sub(1)..(self.x.end + 1).min(size[0]), y: self.y.start.saturating_sub(1)..(self.y.end + 1).min(size[1]) }
    }
}

/// What reading the labels changed: nothing, all of them, or (from a dirty box) the pixels of
/// a span or none.
#[derive(Clone, Debug, PartialEq)]
enum LabelsRead {
    Kept,
    All,
    Part(Option<Span>),
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
        if half == 0 || mode == SlabMode::Off {
            return self.extract(values, dims, s);
        }
        let n = dims[self.axes.slice];
        let (from, to) = (s.saturating_sub(half), (s + half).min(n - 1));
        match mode {
            SlabMode::Max => self.fold_slab(values, dims, s, from..=to, f32::max, None),
            SlabMode::Min => self.fold_slab(values, dims, s, from..=to, f32::min, None),
            _ => self.fold_slab(values, dims, s, from..=to, |a, b| a + b, Some((to - from + 1) as f32)),
        }
    }

    /// Each pixel of slice `s` combined with the slices `ks` (but `s`) in ascending order by
    /// `op`, then divided by `count` if given: one pass over the slab, each pixel folded in
    /// the order slice-by-slice combining used, so the values are the same to the bit.
    fn fold_slab(&self, values: &[f32], dims: [usize; 3], s: usize, ks: RangeInclusive<usize>, op: impl Fn(f32, f32) -> f32 + Sync, count: Option<f32>) -> Vec<f32> {
        let stride = [1, dims[0], dims[0] * dims[1]];
        let (sa, sd, ss) = (stride[self.axes.across], stride[self.axes.down], stride[self.axes.slice]);
        let ([w, h], (from, to)) = (self.size, (*ks.start(), *ks.end()));
        let mut out = vec![0.0; w * h];
        // A row is combined from the start of the volume's axis across, then turned round if
        // the screen runs the other way.
        let row = |y: usize, out: &mut [f32]| {
            let first = if self.axes.flip_down { h - 1 - y } else { y } * sd;
            if ss < sa {
                // A pixel's slab lies closer together than the pixels (sagittal slices of an
                // axial series): each pixel's slices in turn.
                for (a, o) in out.iter_mut().enumerate() {
                    let slab = &values[first + a * sa + from * ss..=first + a * sa + to * ss];
                    let at = |k: usize| slab[(k - from) * ss];
                    let mut acc = at(s);
                    for k in (from..s).chain(s + 1..=to) {
                        acc = op(acc, at(k));
                    }
                    *o = acc;
                }
            } else {
                // Whole rows of each slice in turn, read along the row.
                let line = |k: usize| &values[first + k * ss..=first + k * ss + (w - 1) * sa];
                match sa {
                    1 => out.copy_from_slice(line(s)),
                    _ => out.iter_mut().zip(line(s).iter().step_by(sa)).for_each(|(o, &v)| *o = v),
                }
                for k in (from..s).chain(s + 1..=to) {
                    match sa {
                        1 => out.iter_mut().zip(line(k)).for_each(|(o, &v)| *o = op(*o, v)),
                        _ => out.iter_mut().zip(line(k).iter().step_by(sa)).for_each(|(o, &v)| *o = op(*o, v)),
                    }
                }
            }
            if let Some(c) = count {
                out.iter_mut().for_each(|o| *o /= c);
            }
            if self.axes.flip_across {
                out.reverse();
            }
        };
        // A sample read along a row costs an eighth of one read across the volume.
        let n = to - from + 1;
        for_rows(&mut out, w, if ss < sa { n } else { n.div_ceil(8) }, row);
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

    /// Re-read pixels `x` × `y` of slice `s` into `out` (the whole slice, as `extract` makes
    /// it), leaving the other pixels as they are.
    fn extract_in<T: Copy>(&self, values: &[T], dims: [usize; 3], s: usize, out: &mut [T], x: Range<usize>, y: Range<usize>) {
        let stride = [1, dims[0], dims[0] * dims[1]];
        let (sa, sd) = (stride[self.axes.across], stride[self.axes.down]);
        let base = s * stride[self.axes.slice];
        let [w, h] = self.size;
        for y in y {
            let d = if self.axes.flip_down { h - 1 - y } else { y };
            let row = base + d * sd;
            for x in x.clone() {
                let a = if self.axes.flip_across { w - 1 - x } else { x };
                out[y * w + x] = values[row + a * sa];
            }
        }
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
/// frame and pixel size as bits, and its offset along its normal in nanometres).
#[derive(Clone, Copy, Debug, PartialEq)]
enum At {
    Slice(usize),
    Oblique([u64; 11]),
}

/// The origin itself is not part of the key: it is recomputed from the cursor, so a click in
/// the plane or a step along another view's normal changes its last bits without moving the
/// plane. Only its offset along the normal says which plane it is (the image's place in the
/// plane depends on the volume alone).
fn at_oblique(g: &ObliqueGrid) -> At {
    let f = g.frame;
    let offset = (g.origin.dot(f[2]) * 1e6).round() as i64;
    let b = [f[0].x, f[0].y, f[0].z, f[1].x, f[1].y, f[1].z, f[2].x, f[2].y, f[2].z, g.px].map(f64::to_bits);
    At::Oblique([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], offset as u64])
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
    /// The labels array they were read from (its address): a dirty box only tells how that
    /// one changed.
    labels_from: usize,
    /// The label colours the texture was drawn with: a dirty box recolours its part alone.
    colours_drawn: Vec<Color32>,
    /// The oblique plane the labels were sampled on, to re-sample part of them on it.
    labels_grid: Option<ObliqueGrid>,
    /// The image's colours before the labels go over them, for what is shown but the labels
    /// (kept only while labels are drawn: a dab then recolours just the labels).
    base_of: Option<Shown>,
    base: Vec<Color32>,
    /// Where the labels changed, for the next `ui` call.
    label_dirty: Option<LabelsDirty>,
    shown: Option<Shown>,
}

impl SliceView {
    pub fn new(plane: Plane) -> Self {
        Self {
            plane,
            nav: Nav::default(),
            texture: None,
            values_of: None,
            values: Vec::new(),
            fusion_of: None,
            fusion_values: Vec::new(),
            labels_of: None,
            labels: Vec::new(),
            labels_from: 0,
            colours_drawn: Vec::new(),
            labels_grid: None,
            base_of: None,
            base: Vec::new(),
            label_dirty: None,
            shown: None,
        }
    }

    /// The overlay version of the labels the view holds (`None`: none yet), which a dirty box
    /// given to [`Self::set_label_dirty`] has to start from.
    pub fn labels_version(&self) -> Option<u64> {
        self.labels_of.map(|(_, _, v)| v)
    }

    /// Where the labels changed since the version the view holds, for the next [`Self::ui`]
    /// call: the view then re-reads and recolours just the part of its image those voxels
    /// cover (nothing if its plane misses them). Without it, a new version re-reads every
    /// label shown, as it does when the box does not lead from the view's version to the
    /// overlay's version then, or the overlay's labels array or colours are not the ones the
    /// view drew. Set it (or `None`) before each `ui` call: a box is used or dropped there.
    pub fn set_label_dirty(&mut self, dirty: Option<LabelsDirty>) {
        self.label_dirty = dirty;
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
        self.nav.tool_secondary = opts.tool_secondary_drag;
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
        let tool_drag = ev.tool_drag.map(|d| (d, at(d.press)));

        self.update_texture(ui.ctx(), vol, &grid, &map, state, overlay, opts.smooth);
        self.paint(ui, &map, vol, &grid, state, opts, ev.hover);
        // Above the slice position in the bottom left corner.
        let key = (volume_id(vol), vol.range.0.to_bits(), vol.range.1.to_bits());
        histogram::show(ui, response.id, rect, rect.bottom() - 24.0, key, &vol.values, state.window, vol.invert, self.nav.windowing());
        SliceResponse { response, hover_voxel, clicked_voxel, left_drag_voxel, brush_wheel, tool_drag, double_clicked: ev.double_clicked, map }
    }

    /// Re-make the texture when what it shows has changed: only the part a dirty box covers
    /// when the labels alone moved on, only the labels over the kept image colours when the
    /// labels or their opacity changed, otherwise all of it.
    #[allow(clippy::too_many_arguments)]
    fn update_texture(&mut self, ctx: &egui::Context, vol: &LoadedVolume, grid: &Grid, map: &ViewMap, state: &ViewState, overlay: Option<&LabelOverlay>, smooth: bool) {
        let dirty = self.label_dirty.take();
        let volume = volume_id(vol);
        let window = state.window;
        let slice = map.slice;
        let mode = state.slab.mode;
        let [w, h] = map.size;
        // The image's values, the second image's and the labels: a slice (slab) of the grid,
        // or sampled on the oblique plane. `half` is the slab's slices (or samples along an
        // oblique plane's normal) on each side, which the values depend on.
        let (at, half, half_mm, interpolated) = match &map.oblique {
            Some(g) => (at_oblique(g), g.slab_steps(state.slab.mm * 0.5, mode).max(0) as usize, state.slab.mm * 0.5, smooth),
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
        let mut read = LabelsRead::Kept;
        if let Some(o) = overlay {
            if self.labels_of != Some((volume, at, o.version)) {
                // After a dirty box from the version held to this one, in the labels read
                // before, only the pixels it covers.
                let from = o.labels.as_ptr() as usize;
                let leads_here = |d: &LabelsDirty| self.labels_of == Some((volume, at, d.since)) && d.until == o.version && self.labels_from == from && self.labels.len() == w * h;
                let part = dirty.filter(leads_here).and_then(|d| self.dirty_span(vol, grid, map, &d));
                read = match part {
                    Some(span) => {
                        if let Some(s) = &span {
                            match &self.labels_grid {
                                Some(g) => g.sample_labels_in(vol, o.labels, &mut self.labels, s.x.clone(), s.y.clone()),
                                None => grid.extract_in(o.labels, vol.dims, slice, &mut self.labels, s.x.clone(), s.y.clone()),
                            }
                        }
                        LabelsRead::Part(span)
                    }
                    None => {
                        self.labels = match &map.oblique {
                            Some(g) => g.sample_labels(vol, o.labels),
                            None => grid.extract(o.labels, vol.dims, slice),
                        };
                        self.labels_grid = map.oblique;
                        self.labels_from = from;
                        LabelsRead::All
                    }
                };
                self.labels_of = Some((volume, at, o.version));
            }
        }
        let shown = Shown { volume, at, window, invert: vol.invert, smooth, labels: overlay.map(|o| (o.version, o.opacity.to_bits())), colour_map: state.colour_map, slab: state.slab, fusion: fusion.cloned() };
        if self.shown.as_ref() == Some(&shown) && self.texture.is_some() {
            return;
        }
        let options = if smooth { egui::TextureOptions::LINEAR } else { egui::TextureOptions::NEAREST };
        let base_key = Shown { labels: None, ..shown.clone() };
        let base_ok = self.base_of.as_ref() == Some(&base_key) && self.base.len() == w * h;
        let fill = overlay.map_or(0.0, |o| o.opacity.clamp(0.0, 1.0));
        let outline = overlay.map(|o| Outline { labels: &self.labels, colours: o.colours, fill, size: map.size });
        // Only the labels moved on, the texture showing everything else as it is (in the same
        // label colours): recolour the pixels they changed and the outline around them (or
        // nothing).
        let labels_only = base_ok
            && overlay.is_some_and(|o| o.colours == &self.colours_drawn[..])
            && self.shown.as_ref().is_some_and(|s| s.labels.map(|l| l.1) == shown.labels.map(|l| l.1) && Shown { labels: shown.labels, ..s.clone() } == shown);
        if let (LabelsRead::Part(span), true, Some(outline), Some(t)) = (&read, labels_only, &outline, &mut self.texture) {
            if let Some(s) = span.as_ref().map(|s| s.grown(map.size)) {
                let base = &self.base;
                let mut patch = vec![Color32::TRANSPARENT; s.x.len() * s.y.len()];
                for_rows(&mut patch, s.x.len(), 1, |r, row| {
                    let y = s.y.start + r;
                    for (x, c) in s.x.clone().zip(row) {
                        *c = outline.colour(base[y * w + x], x, y);
                    }
                });
                t.set_partial([s.x.start, s.y.start], egui::ColorImage::new([s.x.len(), s.y.len()], patch), options);
            }
            self.shown = Some(shown);
            return;
        }
        let palette = Palette {
            level: Level::new(window, vol.invert),
            lut: state.colour_map.lut(),
            second: fusion.map(|f| Second {
                values: &self.fusion_values,
                level: Level::new(f.window, false),
                lut: f.colour_map.lut(),
                compare: f.compare,
                opacity: f.opacity.clamp(0.0, 1.0),
                divider: (f.wipe.clamp(0.0, 1.0) * w as f32).round() as usize,
                showing: f.showing,
            }),
        };
        // Row by row (each pixel's colour is worked out as on its own).
        let values = &self.values;
        let mut pixels = vec![Color32::TRANSPARENT; w * h];
        let line = |y: usize| y * w..(y + 1) * w;
        match (&outline, base_ok) {
            // The image's colours are kept: the labels over them alone.
            (Some(outline), true) => {
                let base = &self.base;
                for_rows(&mut pixels, w, 1, |y, row| outline.row(y, &base[line(y)], row));
            }
            // One pass over the rows: the image's colours, kept for the next change of the
            // labels, and the labels over them.
            (Some(outline), false) => {
                self.base.resize(w * h, Color32::TRANSPARENT);
                for_rows2(&mut pixels, &mut self.base, w, colour_cost(fusion.is_some()) + 1, |y, row, base| {
                    palette.row(y, &values[line(y)], base);
                    outline.row(y, base, row);
                });
                self.base_of = Some(base_key);
            }
            // No labels (any more): the kept image's colours, or the image's colours alone
            // (not kept: nothing is drawn over them).
            (None, true) => pixels.copy_from_slice(&self.base),
            (None, false) => {
                for_rows(&mut pixels, w, colour_cost(fusion.is_some()), |y, row| palette.row(y, &values[line(y)], row));
                self.base_of = None;
                self.base = Vec::new();
            }
        }
        self.colours_drawn.clear();
        if let Some(o) = overlay {
            self.colours_drawn.extend_from_slice(o.colours);
        }
        let image = egui::ColorImage::new(map.size, pixels);
        match &mut self.texture {
            Some(t) => t.set(image, options),
            None => self.texture = Some(ctx.load_texture(format!("slice.{:?}", self.plane), image, options)),
        }
        self.shown = Some(shown);
    }

    /// The pixels whose labels can differ after the changes in dirty box `d`: `Some(None)`
    /// when the plane shown misses the box, `None` when the view cannot tell.
    fn dirty_span(&self, vol: &LoadedVolume, grid: &Grid, map: &ViewMap, d: &LabelsDirty) -> Option<Option<Span>> {
        let (lo, hi) = (d.lo, [0, 1, 2].map(|a| d.hi[a].min(vol.dims[a].saturating_sub(1))));
        if (0..3).any(|a| lo[a] > hi[a]) {
            return Some(None);
        }
        if map.oblique.is_none() {
            let (ax, [w, h]) = (grid.axes, grid.size);
            if map.slice < lo[ax.slice] || map.slice > hi[ax.slice] {
                return Some(None);
            }
            let x = if ax.flip_across { w - 1 - hi[ax.across]..w - lo[ax.across] } else { lo[ax.across]..hi[ax.across] + 1 };
            let y = if ax.flip_down { h - 1 - hi[ax.down]..h - lo[ax.down] } else { lo[ax.down]..hi[ax.down] + 1 };
            return Some(Some(Span { x, y }));
        }
        // An oblique plane (the one the labels were sampled on) samples a changed voxel only
        // at points within half a voxel of the box: where it cuts the box widened by that,
        // with a pixel to spare for rounding.
        let g = self.labels_grid.as_ref()?;
        let (mut u, mut v) = ((f64::INFINITY, f64::NEG_INFINITY), (f64::INFINITY, f64::NEG_INFINITY));
        let (mut above, mut below) = (true, true);
        for c in 0..8 {
            let corner = DVec3::from([0, 1, 2].map(|a| if c & (1 << a) > 0 { hi[a] as f64 + 0.5 } else { lo[a] as f64 - 0.5 }));
            let (cu, cv, cd) = g.pixel(g.to_world.transform_point3(corner));
            if !(cu.is_finite() && cv.is_finite() && cd.is_finite()) {
                return None;
            }
            (u, v) = ((u.0.min(cu), u.1.max(cu)), (v.0.min(cv), v.1.max(cv)));
            above &= cd > 1e-6;
            below &= cd < -1e-6;
        }
        if above || below {
            return Some(None);
        }
        let span = |(lo, hi): (f64, f64), n: usize| {
            let (a, b) = ((lo.floor() - 1.0).max(0.0), (hi.ceil() + 1.0).min(n as f64 - 1.0));
            (a <= b).then(|| a as usize..b as usize + 1)
        };
        Some(span(u, g.size[0]).zip(span(v, g.size[1])).map(|(x, y)| Span { x, y }))
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

    /// A small xorshift generator: the tests' random volumes and edits are the same each run.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn below(&mut self, n: usize) -> usize {
            (self.next() % n.max(1) as u64) as usize
        }

        fn chance(&mut self, p: f64) -> bool {
            (self.next() % 1_000_000) as f64 / 1e6 < p
        }
    }

    /// Slabs as they were combined before the one-pass fold: slice by slice (verbatim).
    fn slab_slice_by_slice(grid: &Grid, values: &[f32], dims: [usize; 3], s: usize, half: usize, mode: SlabMode) -> Vec<f32> {
        let mut out = grid.extract(values, dims, s);
        if half == 0 || mode == SlabMode::Off {
            return out;
        }
        let n = dims[grid.axes.slice];
        let (from, to) = (s.saturating_sub(half), (s + half).min(n - 1));
        for k in (from..=to).filter(|&k| k != s) {
            let other = grid.extract(values, dims, k);
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

    #[test]
    fn rows_in_parallel_are_numbered_as_in_turn() {
        // Pixels costly enough for the rows to be split.
        let (w, h) = (1000, 203);
        let cost = PARALLEL_WORK.div_ceil(w * h);
        let (mut a, mut b, mut c) = (vec![0; w * h], vec![0; w * h], vec![0; w * h]);
        for_rows2(&mut a, &mut b, w, cost, |y, a, b| {
            for (x, (a, b)) in a.iter_mut().zip(b).enumerate() {
                (*a, *b) = (y * w + x, y);
            }
        });
        for_rows(&mut c, w, cost, |y, c| c.iter_mut().enumerate().for_each(|(x, c)| *c = y * w + x));
        assert!((0..w * h).all(|p| a[p] == p && b[p] == p / w && c[p] == p));
    }

    #[test]
    fn slabs_fold_to_the_bit_as_slice_by_slice() {
        // Values that make the order of the fold matter: NaNs, signed zeros, large and small
        // magnitudes (sums round differently in another order).
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        let value = |rng: &mut Rng| match rng.below(12) {
            0 => f32::NAN,
            1 => 0.0,
            2 => -0.0,
            3 => 1e7 * (rng.below(1000) as f32 - 500.0),
            _ => (rng.next() % 100_000) as f32 * 0.013 - 600.0,
        };
        // Layouts whose slices run along each voxel axis, with rows read along each, both ways.
        let (x, y, z) = (DVec3::X, DVec3::Y, DVec3::Z);
        let layouts = [([x, y, z], [9, 7, 11]), ([-x, -y, z], [9, 7, 11]), ([-y, -z, x], [7, 11, 9]), ([z, x, y], [11, 9, 7]), ([-z, y, -x], [11, 7, 9])];
        let vols: Vec<LoadedVolume> = layouts
            .into_iter()
            .map(|(axes, dims)| {
                let mut v = phantom(axes, dims, [1.0; 3]);
                v.values.iter_mut().for_each(|x| *x = value(&mut rng));
                v
            })
            .collect();
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        // Large enough for a thick sagittal slab's rows to be combined in parallel.
        let mut big = phantom([x, y, z], [64, 360, 400], [1.0; 3]);
        let grid = Grid::new(&big, Plane::Sagittal);
        assert!(grid.size[0] * grid.size[1] * 61 >= PARALLEL_WORK);
        big.values.iter_mut().for_each(|x| *x = value(&mut rng));
        for mode in SlabMode::ALL {
            let want = slab_slice_by_slice(&grid, &big.values, big.dims, 32, 30, mode);
            assert_eq!(bits(&grid.extract_slab(&big.values, big.dims, 32, 30, mode)), bits(&want), "big {mode:?}");
        }
        for v in &vols {
            for plane in Plane::ALL {
                let grid = Grid::new(v, plane);
                let n = v.dims[grid.axes.slice];
                let slices: Vec<usize> = (0..n).collect();
                for s in slices {
                    for half in [0, 1, 2, 5, 30] {
                        for mode in SlabMode::ALL {
                            let want = slab_slice_by_slice(&grid, &v.values, v.dims, s, half, mode);
                            let got = grid.extract_slab(&v.values, v.dims, s, half, mode);
                            assert_eq!(bits(&got), bits(&want), "{plane:?} slice {s} half {half} {mode:?} dims {:?}", v.dims);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn oblique_keys_follow_the_plane_not_the_cursor() {
        let g = Geometry::from_axes([60, 50, 40, 1], [0.5, 0.5, 0.6], [DVec3::NEG_X, DVec3::NEG_Y, DVec3::Z], DVec3::new(15.0, 12.0, -10.0));
        let vol = LoadedVolume::from_values("t", [60, 50, 40], vec![0.0; 120_000], g);
        let frames = crate::oblique::along_vessel(DVec3::new(0.3, 0.4, 0.87)).0;
        let keys = |cursor: DVec3| frames.map(|f| at_oblique(&ObliqueGrid::new(&vol, f, cursor)));
        let mut cursor = DVec3::new(30.0, 25.0, 20.0);
        let mut others_moved = 0;
        for i in 0..200 {
            // A wheel step in view `k`, as SliceView::ui takes it (back and forth, so the
            // cursor stays inside).
            let k = i % 3;
            let before = keys(cursor);
            let ob = ObliqueGrid::new(&vol, frames[k], cursor);
            let w = ob.to_world.transform_point3(cursor) + frames[k][2] * (ob.px * if i % 2 == 0 { 1.0 } else { -1.0 });
            let v = ob.to_voxel.transform_point3(w);
            cursor = DVec3::from([0, 1, 2].map(|a| v[a].clamp(0.0, vol.dims[a] as f64 - 1.0)));
            let after = keys(cursor);
            assert_ne!(before[k], after[k], "the stepped view shows another plane");
            others_moved += (0..3).filter(|&j| j != k && before[j] != after[j]).count();
        }
        assert_eq!(others_moved, 0, "views the step does not move keep their image");
        // A click in a view's plane moves the other two planes, not its own.
        for k in 0..3 {
            let before = keys(cursor);
            let ob = ObliqueGrid::new(&vol, frames[k], cursor);
            cursor = ob.to_voxel.transform_point3(ob.to_world.transform_point3(cursor) + frames[k][0] * 2.3 - frames[k][1] * 1.7);
            let after = keys(cursor);
            assert_eq!(before[k], after[k]);
            assert!((0..3).filter(|&j| j != k).all(|j| before[j] != after[j]));
        }
    }

    /// A slice's colours as the views made them before the image colours were kept and
    /// coloured a row at a time (verbatim), from the same values and labels.
    #[allow(clippy::too_many_arguments)]
    fn colours_as_before(values: &[f32], fusion_values: &[f32], labels: &[u8], size: [usize; 2], window: (f64, f64), invert: bool, colour_map: ColourMap, fusion: Option<&Fusion>, overlay: Option<(&[Color32], f32)>) -> Vec<Color32> {
        let level = |(c, w): (f64, f64), invert: bool| {
            let (lo, scale) = (c - w * 0.5, 255.0 / w.max(f64::MIN_POSITIVE));
            move |x: f32| {
                let g = ((x as f64 - lo) * scale).round().clamp(0.0, 255.0) as u8;
                if invert {
                    255 - g
                } else {
                    g
                }
            }
        };
        let (grey, lut) = (level(window, invert), colour_map.lut());
        let [w, h] = size;
        let base: Vec<Color32> = match fusion {
            Some(f) => {
                let (fgrey, flut) = (level(f.window, false), f.colour_map.lut());
                let divider = (f.wipe.clamp(0.0, 1.0) * w as f32).round() as usize;
                values
                    .iter()
                    .zip(fusion_values)
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
            None => values.iter().map(|&x| lut[grey(x) as usize]).collect(),
        };
        match overlay {
            Some((colours, opacity)) => {
                let fill = opacity.clamp(0.0, 1.0);
                (0..w * h)
                    .map(|p| {
                        let (g, l) = (base[p], labels[p]);
                        let Some(col) = colours.get(l as usize).filter(|c| l != 0 && c.a() > 0) else { return g };
                        let (x, y) = (p % w, p / w);
                        let edge = x == 0 || y == 0 || x + 1 == w || y + 1 == h || labels[p - 1] != l || labels[p + 1] != l || labels[p - w] != l || labels[p + w] != l;
                        let alpha = if edge { 1.0 } else { fill };
                        let mix = |g: u8, c: u8| (g as f32 + (c as f32 - g as f32) * alpha).round() as u8;
                        Color32::from_rgb(mix(g.r(), col.r()), mix(g.g(), col.g()), mix(g.b(), col.b()))
                    })
                    .collect()
            }
            None => base,
        }
    }

    /// Slice views run in a headless egui context, their textures kept as a renderer keeps
    /// them: whole images replace, patches are written in place.
    struct Screen {
        ctx: egui::Context,
        textures: std::collections::HashMap<egui::TextureId, egui::ColorImage>,
        wholes: usize,
        patches: usize,
    }

    impl Screen {
        fn new() -> Self {
            Self { ctx: egui::Context::default(), textures: Default::default(), wholes: 0, patches: 0 }
        }

        /// One frame of `view`; its image as its texture now holds it.
        fn frame(&mut self, view: &mut SliceView, vol: &LoadedVolume, state: &ViewState, overlay: Option<&LabelOverlay>, smooth: bool) -> egui::ColorImage {
            let raw = egui::RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(300.0, 260.0))), ..Default::default() };
            let (mut state, opts) = (state.clone(), SliceOptions { smooth, ..Default::default() });
            let out = self.ctx.run_ui(raw, |ui| {
                view.ui(ui, vol, &mut state, overlay, &opts);
            });
            for (id, deltas) in &out.textures_delta.set {
                for d in deltas {
                    let egui::ImageData::Color(img) = &d.image;
                    match d.pos {
                        None => {
                            self.wholes += 1;
                            self.textures.insert(*id, (**img).clone());
                        }
                        Some([x0, y0]) => {
                            self.patches += 1;
                            let t = self.textures.get_mut(id).expect("a patch of a texture set before");
                            let [pw, ph] = img.size;
                            assert!(x0 + pw <= t.size[0] && y0 + ph <= t.size[1]);
                            for y in 0..ph {
                                t.pixels[(y0 + y) * t.size[0] + x0..][..pw].copy_from_slice(&img.pixels[y * pw..][..pw]);
                            }
                        }
                    }
                }
            }
            self.textures[&view.texture.as_ref().expect("a texture").id()].clone()
        }
    }

    #[test]
    fn dirty_boxes_and_kept_colours_draw_what_a_new_view_draws() {
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        let (x, y, z) = (DVec3::X, DVec3::Y, DVec3::Z);
        let vols = [phantom([x, y, z], [12, 10, 8], [0.8, 1.0, 1.5]), phantom([-x, -y, z], [12, 10, 8], [1.0; 3]), phantom([-y, -z, x], [10, 8, 12], [1.0; 3])];
        // Label 3's colour is see-through and 4 and 5 have none: never drawn.
        let first_colours = [Color32::TRANSPARENT, Color32::from_rgb(230, 80, 60), Color32::from_rgb(60, 200, 90), Color32::from_rgba_unmultiplied(0, 0, 255, 0)];
        let oblique = [crate::oblique::along_vessel(DVec3::new(0.3, 0.4, 0.87)).0, crate::PLANES.map(|p| rotated(standard_frame(p), DVec3::new(1.0, -0.5, 0.2).normalize(), 0.35))];
        // Patched and untouched textures, along the grid and oblique.
        let (mut fresh, mut patches, mut misses) = (Screen::new(), [0; 2], [0; 2]);
        for vol in &vols {
            let second = Fusion::new("second", vol.values.iter().map(|v| (v * 0.37).sin() * 50.0 + v * 0.01).collect(), ColourMap::Hot);
            let fusions = [None, Some(Fusion { opacity: 0.3, ..second.clone() }), Some(Fusion { compare: Compare::Wipe, wipe: 0.4, ..second.clone() }), Some(Fusion { compare: Compare::Toggle, showing: true, ..second.clone() })];
            for plane in Plane::ALL {
                for frames in [None, Some(oblique[0]), Some(oblique[1])] {
                    for fusion in &fusions {
                        let mut state = ViewState::new(vol);
                        state.frames = frames;
                        state.fusion = fusion.clone();
                        state.window = (vol.auto_window.0 + rng.below(200) as f64 - 100.0, vol.auto_window.1 * 0.7);
                        state.colour_map = ColourMap::ALL[rng.below(ColourMap::ALL.len())];
                        if rng.chance(0.3) {
                            state.slab = Slab { mode: SlabMode::Max, mm: 3.0 };
                        }
                        // The cursor in the middle or by an edge.
                        state.cursor = DVec3::from([0, 1, 2].map(|a| if rng.chance(0.3) { 0.0 } else { rng.below(vol.dims[a]) as f64 }));
                        let smooth = rng.chance(0.5);
                        let mut labels: Vec<u8> = (0..vol.values.len()).map(|_| if rng.chance(0.6) { 0 } else { rng.below(6) as u8 }).collect();
                        // Another labels array shown instead at times (a threshold preview).
                        let preview: Vec<u8> = (0..vol.values.len()).map(|_| if rng.chance(0.5) { 0 } else { 1 + rng.below(2) as u8 }).collect();
                        let (mut version, mut opacity, mut drawn, mut previewing, mut colours) = (1u64, 0.5f32, true, false, first_colours);
                        // A view left out of a frame (another one fills the layout).
                        let mut skipped = false;
                        // The voxels changed since the version the view holds.
                        let mut changed: Option<([usize; 3], [usize; 3])> = None;
                        let mut view = SliceView::new(plane);
                        let mut screen = Screen::new();
                        for step in 0..24 {
                            let mut dirty = None;
                            if step > 0 {
                                // An edit in a box (at the volume's edges at times, or a single
                                // voxel): paint, erase, or erase one label only ("inside").
                                let mut lo = [0; 3];
                                let mut hi = [0; 3];
                                for a in 0..3 {
                                    let n = vol.dims[a];
                                    let reach = if rng.chance(0.3) { n } else { 3 };
                                    let (l, size) = (rng.below(n), 1 + rng.below(reach));
                                    lo[a] = if rng.chance(0.15) { 0 } else { l };
                                    hi[a] = if rng.chance(0.15) { n - 1 } else { (lo[a] + size - 1).min(n - 1) };
                                }
                                let (op, label) = (rng.below(3), 1 + rng.below(5) as u8);
                                for k in lo[2]..=hi[2] {
                                    for j in lo[1]..=hi[1] {
                                        for i in lo[0]..=hi[0] {
                                            let l = &mut labels[vol.index(i, j, k)];
                                            *l = match op {
                                                0 => label,
                                                1 => 0,
                                                _ if *l == label => 0,
                                                _ => *l,
                                            };
                                        }
                                    }
                                }
                                version += 1;
                                let (lo, hi) = match changed {
                                    Some((a, b)) => ([0, 1, 2].map(|i| a[i].min(lo[i])), [0, 1, 2].map(|i| b[i].max(hi[i]))),
                                    None => (lo, hi),
                                };
                                changed = Some((lo, hi));
                                // Mostly a box from the view's version to this one; at times none,
                                // one that does not start from it or does not end here, or a box
                                // beyond the volume.
                                let since = view.labels_version().unwrap_or(0);
                                dirty = match rng.below(12) {
                                    0 => None,
                                    1 => Some(LabelsDirty { since: since.wrapping_sub(1), until: version, lo, hi }),
                                    2 => Some(LabelsDirty { since, until: version, lo, hi: hi.map(|h| h + 5) }),
                                    3 => Some(LabelsDirty { since, until: version - 1, lo, hi }),
                                    _ => Some(LabelsDirty { since, until: version, lo, hi }),
                                };
                                // What else changes now and then: the opacity, the window, the
                                // labels hidden and shown, the second image's opacity, a label's
                                // colour (with the box), the preview shown instead (the box tells
                                // of the labels' changes), the slab.
                                match rng.below(15) {
                                    0 => opacity = rng.below(11) as f32 * 0.1,
                                    1 => state.window.0 += 7.0,
                                    2 => drawn = !drawn,
                                    3 => {
                                        if let Some(f) = state.fusion.as_mut() {
                                            f.opacity = 0.8;
                                        }
                                    }
                                    4 => colours[1 + rng.below(2)] = Color32::from_rgb(rng.below(256) as u8, rng.below(256) as u8, rng.below(256) as u8),
                                    5 => previewing = !previewing,
                                    6 => state.slab = Slab { mode: SlabMode::ALL[rng.below(SlabMode::ALL.len())], mm: [1.0, 3.0, 6.0][rng.below(3)] },
                                    _ => drawn = true,
                                }
                            }
                            let overlay = LabelOverlay { labels: if previewing { &preview } else { &labels }, colours: &colours, opacity, version };
                            let overlay = drawn.then_some(&overlay);
                            // After a frame left out, the box given for it is at times all the
                            // view has (it no longer leads to the version shown).
                            if !skipped || rng.chance(0.5) {
                                view.set_label_dirty(dirty);
                            }
                            skipped = step > 0 && rng.chance(0.1);
                            if skipped {
                                continue;
                            }
                            let (wholes, patches_before) = (screen.wholes, screen.patches);
                            let got = screen.frame(&mut view, vol, &state, overlay, smooth);
                            if drawn {
                                changed = None;
                            }
                            patches[frames.is_some() as usize] += screen.patches - patches_before;
                            misses[frames.is_some() as usize] += (step > 0 && drawn && dirty.is_some() && screen.wholes == wholes && screen.patches == patches_before) as usize;
                            // A new view of the same.
                            let mut new = SliceView::new(plane);
                            let want = fresh.frame(&mut new, vol, &state, overlay, smooth);
                            let at = format!("{:?} {plane:?} oblique {} {:?} step {step}", vol.dims, frames.is_some(), fusion.as_ref().map(|f| f.compare));
                            if drawn {
                                assert!(view.labels == new.labels, "{at}: labels");
                            }
                            assert!(got.size == want.size && got.pixels == want.pixels, "{at}: the image differs from a new view's");
                            // And the new view's colours are those of the colouring before.
                            let before = colours_as_before(&new.values, &new.fusion_values, &new.labels, want.size, state.window, vol.invert, state.colour_map, state.fusion.as_ref(), overlay.map(|o| (o.colours, o.opacity)));
                            assert!(want.pixels == before, "{at}: the colours differ from before");
                            fresh.textures.remove(&new.texture.as_ref().expect("a texture").id());
                        }
                    }
                }
            }
        }
        assert!(patches.iter().chain(&misses).all(|&n| n > 100), "the dirty boxes were used: {patches:?} patches, {misses:?} misses");
    }

    #[test]
    fn dirty_boxes_that_do_not_lead_to_the_labels_shown_are_not_used() {
        let g = Geometry::from_axes([40, 40, 20, 1], [1.0; 3], [DVec3::X, DVec3::Y, DVec3::Z], DVec3::ZERO);
        let vol = LoadedVolume::from_values("t", [40, 40, 20], (0..32_000).map(|n| (n % 97) as f32 * 10.0).collect(), g);
        let paint = |labels: &mut [u8], lo: [usize; 3], hi: [usize; 3], l: u8| {
            for k in lo[2]..=hi[2] {
                for j in lo[1]..=hi[1] {
                    for i in lo[0]..=hi[0] {
                        labels[vol.index(i, j, k)] = l;
                    }
                }
            }
        };
        let colours = [Color32::TRANSPARENT, Color32::from_rgb(230, 80, 60), Color32::from_rgb(60, 200, 90)];
        let mut state = ViewState::new(&vol);
        state.cursor = DVec3::new(20.0, 20.0, 10.0);
        // A frame of `view`, which must draw what a new view draws.
        let check = |what: &str, view: &mut SliceView, screen: &mut Screen, o: &LabelOverlay| {
            let got = screen.frame(view, &vol, &state, Some(o), false);
            let want = Screen::new().frame(&mut SliceView::new(Plane::Axial), &vol, &state, Some(o), false);
            assert!(got.pixels == want.pixels, "{what}: the image differs from a new view's");
        };
        // A box given for a frame the view is left out of, then another edit: the next frame
        // (given no box) shows both.
        let mut labels = vec![0; 32_000];
        let (mut view, mut screen) = (SliceView::new(Plane::Axial), Screen::new());
        check("first", &mut view, &mut screen, &LabelOverlay { labels: &labels, colours: &colours, opacity: 0.5, version: 1 });
        paint(&mut labels, [5, 5, 10], [8, 8, 10], 1);
        view.set_label_dirty(Some(LabelsDirty { since: 1, until: 2, lo: [5, 5, 10], hi: [8, 8, 10] }));
        paint(&mut labels, [20, 20, 10], [25, 25, 10], 1);
        check("a box left over", &mut view, &mut screen, &LabelOverlay { labels: &labels, colours: &colours, opacity: 0.5, version: 3 });
        // To another labels array (a threshold preview) and back, with boxes of the labels'
        // own changes.
        let mut preview = vec![0; 32_000];
        paint(&mut preview, [3, 3, 10], [30, 12, 10], 2);
        view.set_label_dirty(Some(LabelsDirty { since: 3, until: 4, lo: [0, 0, 10], hi: [1, 1, 10] }));
        check("to the preview", &mut view, &mut screen, &LabelOverlay { labels: &preview, colours: &colours, opacity: 0.5, version: 4 });
        paint(&mut labels, [30, 30, 10], [31, 31, 10], 2);
        view.set_label_dirty(Some(LabelsDirty { since: 4, until: 5, lo: [30, 30, 10], hi: [31, 31, 10] }));
        check("back from the preview", &mut view, &mut screen, &LabelOverlay { labels: &labels, colours: &colours, opacity: 0.5, version: 5 });
        // A label's colour changed along with a dab.
        let mut recoloured = colours;
        recoloured[1] = Color32::from_rgb(40, 90, 250);
        paint(&mut labels, [32, 32, 10], [34, 34, 10], 2);
        view.set_label_dirty(Some(LabelsDirty { since: 5, until: 6, lo: [32, 32, 10], hi: [34, 34, 10] }));
        check("a colour changed", &mut view, &mut screen, &LabelOverlay { labels: &labels, colours: &recoloured, opacity: 0.5, version: 6 });
        // And a box that does lead here is used: a patch, no new image.
        paint(&mut labels, [10, 30, 10], [12, 31, 10], 1);
        view.set_label_dirty(Some(LabelsDirty { since: 6, until: 7, lo: [10, 30, 10], hi: [12, 31, 10] }));
        let (wholes, patches) = (screen.wholes, screen.patches);
        check("a box from the version shown", &mut view, &mut screen, &LabelOverlay { labels: &labels, colours: &recoloured, opacity: 0.5, version: 7 });
        assert_eq!((screen.wholes, screen.patches), (wholes, patches + 1));
    }

    #[test]
    fn oblique_slabs_follow_their_thickness() {
        let g = Geometry::from_axes([30, 28, 20, 1], [0.7, 0.9, 1.1], [DVec3::NEG_X, DVec3::NEG_Y, DVec3::Z], DVec3::new(10.0, -5.0, 3.0));
        let vol = LoadedVolume::from_values("t", [30, 28, 20], (0..16_800u32).map(|i| (i.wrapping_mul(2_654_435_761) >> 20) as f32).collect(), g);
        let mut state = ViewState::new(&vol);
        state.frames = Some(crate::oblique::along_vessel(DVec3::new(0.3, 0.4, 0.87)).0);
        state.fusion = Some(Fusion::new("second", vol.values.iter().map(|v| v * 0.5 - 300.0).collect(), ColourMap::Hot));
        let (mut view, mut screen) = (SliceView::new(Plane::Axial), Screen::new());
        let mut images = Vec::new();
        // Thicker and thinner with the mode kept, and thin with the thickness changed.
        for (mode, mm) in [(SlabMode::Max, 2.0), (SlabMode::Max, 12.0), (SlabMode::Mean, 12.0), (SlabMode::Mean, 3.0), (SlabMode::Off, 3.0), (SlabMode::Off, 12.0)] {
            state.slab = Slab { mode, mm };
            let got = screen.frame(&mut view, &vol, &state, None, false);
            let want = Screen::new().frame(&mut SliceView::new(Plane::Axial), &vol, &state, None, false);
            assert!(got.pixels == want.pixels, "{mode:?} {mm} mm: the image differs from a new view's");
            images.push(got.pixels);
        }
        assert!(images[0] != images[1] && images[2] != images[3], "the thickness shows");
    }
}
