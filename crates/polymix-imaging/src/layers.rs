//! How the image views show values: colour maps, window presets, thick slabs, and a second
//! image over the first (fusion, or a before / after comparison), with the controls the image
//! tabs share.

use egui::Color32;
use mm_volume::load::LoadedVolume;
use std::sync::Arc;

/// Colours from low to high values.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ColourMap {
    #[default]
    Grey,
    Hot,
    Rainbow,
    Viridis,
    Red,
    Green,
    Blue,
    Yellow,
}

impl ColourMap {
    pub const ALL: [ColourMap; 8] = [ColourMap::Grey, ColourMap::Hot, ColourMap::Rainbow, ColourMap::Viridis, ColourMap::Red, ColourMap::Green, ColourMap::Blue, ColourMap::Yellow];

    pub fn label(self) -> &'static str {
        match self {
            ColourMap::Grey => "Grey",
            ColourMap::Hot => "Hot",
            ColourMap::Rainbow => "Rainbow",
            ColourMap::Viridis => "Viridis",
            ColourMap::Red => "Red",
            ColourMap::Green => "Green",
            ColourMap::Blue => "Blue",
            ColourMap::Yellow => "Yellow",
        }
    }

    pub fn key(self) -> &'static str {
        match self {
            ColourMap::Grey => "grey",
            ColourMap::Hot => "hot",
            ColourMap::Rainbow => "rainbow",
            ColourMap::Viridis => "viridis",
            ColourMap::Red => "red",
            ColourMap::Green => "green",
            ColourMap::Blue => "blue",
            ColourMap::Yellow => "yellow",
        }
    }

    /// The colour at `t` in [0, 1].
    pub fn at(self, t: f32) -> Color32 {
        let t = t.clamp(0.0, 1.0);
        let c = |x: f32| (x.clamp(0.0, 1.0) * 255.0).round() as u8;
        let hue = |r: f32, g: f32, b: f32| Color32::from_rgb(c(r * t), c(g * t), c(b * t));
        match self {
            ColourMap::Grey => Color32::from_gray(c(t)),
            // Black → red → yellow → white.
            ColourMap::Hot => Color32::from_rgb(c(3.0 * t), c(3.0 * t - 1.0), c(3.0 * t - 2.0)),
            // Blue → cyan → green → yellow → red (hue 240° to 0°).
            ColourMap::Rainbow => {
                let h = (1.0 - t) * 4.0;
                let x = 1.0 - (h % 2.0 - 1.0).abs();
                let (r, g, b) = match h as u32 {
                    0 => (1.0, x, 0.0),
                    1 => (x, 1.0, 0.0),
                    2 => (0.0, 1.0, x),
                    _ => (0.0, x, 1.0),
                };
                Color32::from_rgb(c(r), c(g), c(b))
            }
            // Matplotlib's viridis (CC0) through five of its colours.
            ColourMap::Viridis => {
                const STOPS: [[f32; 3]; 5] = [[68.0, 1.0, 84.0], [59.0, 82.0, 139.0], [33.0, 145.0, 140.0], [94.0, 201.0, 98.0], [253.0, 231.0, 37.0]];
                let x = t * 4.0;
                let (i, f) = ((x as usize).min(3), x - (x as usize).min(3) as f32);
                let m = |k: usize| (STOPS[i][k] + (STOPS[i + 1][k] - STOPS[i][k]) * f).round() as u8;
                Color32::from_rgb(m(0), m(1), m(2))
            }
            ColourMap::Red => hue(1.0, 0.0, 0.0),
            ColourMap::Green => hue(0.0, 1.0, 0.0),
            ColourMap::Blue => hue(0.2, 0.4, 1.0),
            ColourMap::Yellow => hue(1.0, 0.9, 0.0),
        }
    }

    /// The colours of the 256 grey levels.
    pub fn lut(self) -> [Color32; 256] {
        std::array::from_fn(|g| self.at(g as f32 / 255.0))
    }
}

/// Windows (centre, width) radiologists use for CT, in Hounsfield units.
pub const CT_PRESETS: [(&str, f64, f64); 6] = [("CT brain", 40.0, 80.0), ("CT subdural", 75.0, 215.0), ("CT soft tissue", 50.0, 400.0), ("CT angio", 170.0, 600.0), ("CT bone", 400.0, 1800.0), ("CT lung", -600.0, 1500.0)];

/// How a thick slab combines the slices it spans.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SlabMode {
    #[default]
    Off,
    /// Maximum intensity projection (bright vessels in TOF / CTA).
    Max,
    /// Minimum intensity projection (airways, dark vessels).
    Min,
    Mean,
}

impl SlabMode {
    pub const ALL: [SlabMode; 4] = [SlabMode::Off, SlabMode::Max, SlabMode::Min, SlabMode::Mean];

    pub fn label(self) -> &'static str {
        match self {
            SlabMode::Off => "Thin",
            SlabMode::Max => "MIP",
            SlabMode::Min => "MinIP",
            SlabMode::Mean => "Mean",
        }
    }

    pub fn key(self) -> &'static str {
        match self {
            SlabMode::Off => "off",
            SlabMode::Max => "max",
            SlabMode::Min => "min",
            SlabMode::Mean => "mean",
        }
    }
}

/// The slice views show slabs this thick (mm) around the cursor's slice.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Slab {
    pub mode: SlabMode,
    pub mm: f64,
}

impl Default for Slab {
    fn default() -> Self {
        Self { mode: SlabMode::Off, mm: 10.0 }
    }
}

impl Slab {
    /// Slices on each side of the middle one, for voxels `spacing` mm apart along the slices.
    pub fn half(&self, spacing: f64) -> usize {
        if self.mode == SlabMode::Off {
            0
        } else {
            ((self.mm / spacing.max(1e-9) - 1.0) * 0.5).round().max(0.0) as usize
        }
    }
}

/// How the second image is shown with the first.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Compare {
    /// Over the first, see-through by `opacity` (fusion).
    #[default]
    Blend,
    /// Side by side across a divider: the first on the left, the second on the right.
    Wipe,
    /// One or the other, swapped with a key or button.
    Toggle,
}

impl Compare {
    pub const ALL: [Compare; 3] = [Compare::Blend, Compare::Wipe, Compare::Toggle];

    pub fn label(self) -> &'static str {
        match self {
            Compare::Blend => "Blend",
            Compare::Wipe => "Wipe",
            Compare::Toggle => "Toggle",
        }
    }

    pub fn key(self) -> &'static str {
        match self {
            Compare::Blend => "blend",
            Compare::Wipe => "wipe",
            Compare::Toggle => "toggle",
        }
    }
}

/// A second image on the first one's voxel grid, and how it is shown.
#[derive(Clone, Debug)]
pub struct Fusion {
    pub name: String,
    /// One per voxel of the first image, in its voxel order.
    pub values: Arc<Vec<f32>>,
    pub range: (f32, f32),
    pub auto_window: (f64, f64),
    pub window: (f64, f64),
    pub colour_map: ColourMap,
    pub opacity: f32,
    pub compare: Compare,
    /// Wipe: where the divider is, as a share of the image's width from the left.
    pub wipe: f32,
    /// Toggle: the second image is the one shown.
    pub showing: bool,
}

impl PartialEq for Fusion {
    fn eq(&self, o: &Self) -> bool {
        Arc::ptr_eq(&self.values, &o.values) && self.window == o.window && self.colour_map == o.colour_map && self.opacity == o.opacity && self.compare == o.compare && self.wipe == o.wipe && self.showing == o.showing
    }
}

impl Fusion {
    /// `values` (on the first image's grid) of the image named `name`, shown over it in
    /// `colour_map` at half opacity.
    pub fn new(name: impl Into<String>, values: Vec<f32>, colour_map: ColourMap) -> Self {
        let (range, auto_window) = value_window(&values);
        Self { name: name.into(), values: Arc::new(values), range, auto_window, window: auto_window, colour_map, opacity: 0.5, compare: Compare::Blend, wipe: 0.5, showing: false }
    }

    /// `other` on `vol`'s voxel grid, by position: copied when the grids match, otherwise
    /// sampled (trilinear) where each voxel of `vol` lies in `other` (0 outside it).
    pub fn resampled(vol: &LoadedVolume, other: &LoadedVolume) -> Vec<f32> {
        use rayon::prelude::*;
        let same = vol.dims == other.dims && (vol.geometry.matrix() - other.geometry.matrix()).to_cols_array().iter().all(|x| x.abs() < 1e-4);
        if same {
            return other.values.clone();
        }
        let to_other = other.geometry.matrix().inverse() * vol.geometry.matrix();
        let [nx, ny, _] = vol.dims;
        let d = other.dims;
        let get = |i: usize, j: usize, k: usize| other.values[(k * d[1] + j) * d[0] + i];
        (0..vol.voxel_count())
            .into_par_iter()
            .map(|n| {
                let p = to_other.transform_point3(glam::DVec3::new((n % nx) as f64, ((n / nx) % ny) as f64, (n / (nx * ny)) as f64));
                if (0..3).any(|a| p[a] < -0.5 || p[a] > d[a] as f64 - 0.5) {
                    return 0.0;
                }
                let c = [0, 1, 2].map(|a| p[a].clamp(0.0, (d[a] - 1) as f64));
                let i0 = c.map(|x| x.floor() as usize);
                let i1 = [0, 1, 2].map(|a| (i0[a] + 1).min(d[a] - 1));
                let f = [0, 1, 2].map(|a| (c[a] - i0[a] as f64) as f32);
                let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
                let x00 = lerp(get(i0[0], i0[1], i0[2]), get(i1[0], i0[1], i0[2]), f[0]);
                let x10 = lerp(get(i0[0], i1[1], i0[2]), get(i1[0], i1[1], i0[2]), f[0]);
                let x01 = lerp(get(i0[0], i0[1], i1[2]), get(i1[0], i0[1], i1[2]), f[0]);
                let x11 = lerp(get(i0[0], i1[1], i1[2]), get(i1[0], i1[1], i1[2]), f[0]);
                lerp(lerp(x00, x10, f[1]), lerp(x01, x11, f[1]), f[2])
            })
            .collect()
    }
}

/// The range of the (finite) values, and a window over their 1st to 99th percentile.
fn value_window(values: &[f32]) -> ((f32, f32), (f64, f64)) {
    let step = (values.len() / 200_000).max(1);
    let mut sample: Vec<f32> = values.iter().step_by(step).copied().filter(|v| v.is_finite()).collect();
    if sample.is_empty() {
        return ((0.0, 1.0), (0.5, 1.0));
    }
    sample.sort_by(f32::total_cmp);
    let at = |q: f64| sample[((sample.len() - 1) as f64 * q).round() as usize] as f64;
    let (lo, hi) = (at(0.01), at(0.99));
    let range = (sample[0], sample[sample.len() - 1]);
    let w = (hi - lo).max(1e-6);
    (range, ((lo + hi) * 0.5, w))
}

/// How the image views share their area: the three planes and the 3D view.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Layout {
    /// Axial and 3D above, sagittal and coronal below.
    #[default]
    FourUp,
    /// The 3D view above, the three planes in a row below.
    Conventional,
    /// The three planes side by side (no 3D view).
    Row,
}

impl Layout {
    pub const ALL: [Layout; 3] = [Layout::FourUp, Layout::Conventional, Layout::Row];

    pub fn label(self) -> &'static str {
        match self {
            Layout::FourUp => "Four-up",
            Layout::Conventional => "3D over planes",
            Layout::Row => "Planes in a row",
        }
    }

    pub fn key(self) -> &'static str {
        match self {
            Layout::FourUp => "four_up",
            Layout::Conventional => "conventional",
            Layout::Row => "row",
        }
    }

    /// Where the axial, 3D, sagittal and coronal views go in `full` (`Rect::NOTHING`: not
    /// shown), `gap` points apart.
    pub fn rects(self, full: egui::Rect, gap: f32) -> [egui::Rect; 4] {
        use egui::{Rect, Vec2};
        let row = |top: f32, h: f32| -> [Rect; 3] {
            let w = ((full.width() - 2.0 * gap) / 3.0).floor();
            std::array::from_fn(|i| Rect::from_min_size(egui::pos2(full.min.x + i as f32 * (w + gap), top), Vec2::new(w, h)))
        };
        match self {
            Layout::FourUp => {
                let size = Vec2::new(((full.width() - gap) * 0.5).floor(), ((full.height() - gap) * 0.5).floor());
                let cell = |col: f32, row: f32| Rect::from_min_size(full.min + Vec2::new(col * (size.x + gap), row * (size.y + gap)), size);
                [cell(0.0, 0.0), cell(1.0, 0.0), cell(0.0, 1.0), cell(1.0, 1.0)]
            }
            Layout::Conventional => {
                let top = ((full.height() - gap) * 0.55).floor();
                let [a, s, c] = row(full.min.y + top + gap, full.height() - top - gap);
                [a, Rect::from_min_size(full.min, Vec2::new(full.width(), top)), s, c]
            }
            Layout::Row => {
                let [a, s, c] = row(full.min.y, full.height());
                [a, Rect::NOTHING, s, c]
            }
        }
    }
}

/// The display controls the image tabs share: window presets, colour map and slab.
pub fn display_controls(ui: &mut egui::Ui, id: &str, state: &mut crate::ViewState, vol: &LoadedVolume) {
    ui.menu_button("Window ▾", |ui| {
        if ui.button("Auto").on_hover_text("1st–99th percentile of the volume").clicked() {
            state.window = vol.auto_window;
            ui.close();
        }
        if let Some(fw) = vol.file_window {
            if ui.button("From the file").clicked() {
                state.window = fw;
                ui.close();
            }
        }
        if ui.button("Full range").clicked() {
            let (lo, hi) = (vol.range.0 as f64, vol.range.1 as f64);
            state.window = ((lo + hi) * 0.5, (hi - lo).max(1e-6));
            ui.close();
        }
        ui.separator();
        for (name, c, w) in CT_PRESETS {
            if ui.button(format!("{name}  ({w:.0} / {c:.0})")).clicked() {
                state.window = (c, w);
                ui.close();
            }
        }
    });
    egui::ComboBox::from_id_salt((id, "colour_map")).selected_text(state.colour_map.label()).width(70.0).show_ui(ui, |ui| {
        for m in ColourMap::ALL {
            ui.selectable_value(&mut state.colour_map, m, m.label());
        }
    });
    egui::ComboBox::from_id_salt((id, "slab")).selected_text(state.slab.mode.label()).width(60.0).show_ui(ui, |ui| {
        for m in SlabMode::ALL {
            ui.selectable_value(&mut state.slab.mode, m, m.label());
        }
    })
    .response
    .on_hover_text("Thick slab: the slices around the cursor's combined (MIP follows bright vessels)");
    if state.slab.mode != SlabMode::Off {
        ui.add(egui::DragValue::new(&mut state.slab.mm).range(0.5..=200.0).speed(0.2).suffix(" mm"));
    }
    if state.frames.is_some() {
        if ui.button("Straighten").on_hover_text("Back to planes along the voxel grid").clicked() {
            state.frames = None;
        }
    } else {
        ui.weak("⌥ drag: turn planes").on_hover_text("⌥ / Alt + left drag in a view turns the other two planes about it (oblique planes)");
    }
}

/// The controls of a second image shown over the first: how, its window, colour map and
/// opacity; `false` when the user asked to remove it.
pub fn fusion_controls(ui: &mut egui::Ui, id: &str, f: &mut Fusion) -> bool {
    let mut keep = true;
    ui.horizontal_wrapped(|ui| {
        ui.label(egui::RichText::new(&f.name).strong());
        for c in Compare::ALL {
            ui.selectable_value(&mut f.compare, c, c.label());
        }
        match f.compare {
            Compare::Blend => {
                ui.add(egui::Slider::new(&mut f.opacity, 0.0..=1.0).text("opacity"));
            }
            Compare::Wipe => {
                ui.add(egui::Slider::new(&mut f.wipe, 0.0..=1.0).show_value(false).text("divider"));
            }
            Compare::Toggle => {
                let label = if f.showing { "Showing the second (T)" } else { "Showing the first (T)" };
                if ui.button(label).clicked() {
                    f.showing = !f.showing;
                }
            }
        }
        egui::ComboBox::from_id_salt((id, "fusion_map")).selected_text(f.colour_map.label()).width(70.0).show_ui(ui, |ui| {
            for m in ColourMap::ALL {
                ui.selectable_value(&mut f.colour_map, m, m.label());
            }
        });
        let speed = ((f.range.1 - f.range.0) as f64 / 500.0).max(1e-3);
        ui.label("window");
        ui.add(egui::DragValue::new(&mut f.window.1).speed(speed).range(1e-6..=f64::MAX));
        ui.label("level");
        ui.add(egui::DragValue::new(&mut f.window.0).speed(speed));
        if ui.small_button("Auto").clicked() {
            f.window = f.auto_window;
        }
        if ui.small_button("×").on_hover_text("Stop showing the second image").clicked() {
            keep = false;
        }
    });
    keep
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec3;
    use mm_volume::geometry::Geometry;

    #[test]
    fn colour_maps_run_from_dark_to_bright() {
        for m in ColourMap::ALL {
            let lut = m.lut();
            let lum = |c: Color32| c.r() as u32 + c.g() as u32 + c.b() as u32;
            if m != ColourMap::Rainbow {
                assert!(lum(lut[0]) < lum(lut[255]), "{m:?}");
            }
        }
        assert_eq!(ColourMap::Grey.at(0.5), Color32::from_gray(128));
        assert_eq!(ColourMap::Hot.at(1.0), Color32::WHITE);
        assert_eq!(ColourMap::Rainbow.at(0.0), Color32::from_rgb(0, 0, 255));
        assert_eq!(ColourMap::Rainbow.at(1.0), Color32::from_rgb(255, 0, 0));
        assert_eq!(ColourMap::Viridis.at(1.0), Color32::from_rgb(253, 231, 37));
    }

    #[test]
    fn slabs_span_their_thickness() {
        let s = Slab { mode: SlabMode::Max, mm: 10.0 };
        assert_eq!(s.half(1.0), 5);
        assert_eq!(s.half(2.0), 2);
        assert_eq!(Slab { mode: SlabMode::Off, mm: 10.0 }.half(1.0), 0);
    }

    #[test]
    fn a_second_image_is_resampled_by_position() {
        let g = Geometry::from_axes([4, 4, 4, 1], [1.0; 3], [DVec3::X, DVec3::Y, DVec3::Z], DVec3::ZERO);
        let a = LoadedVolume::from_values("a", [4, 4, 4], (0..64).map(|n| n as f32).collect(), g.clone());
        // The same grid: the values as they are.
        assert_eq!(Fusion::resampled(&a, &a), a.values);
        // Half the voxels (2 mm), shifted by 1 mm: voxel i of `a` lies at (i − 1) / 2 in `b`.
        let g2 = Geometry::from_axes([2, 2, 2, 1], [2.0; 3], [DVec3::X, DVec3::Y, DVec3::Z], DVec3::splat(1.0));
        let b = LoadedVolume::from_values("b", [2, 2, 2], vec![0.0, 10.0, 0.0, 10.0, 0.0, 10.0, 0.0, 10.0], g2);
        let r = Fusion::resampled(&a, &b);
        assert_eq!(r[1], 0.0);
        assert_eq!(r[2], 5.0);
        assert_eq!(r[3], 10.0);
        assert_eq!(r[0], 0.0, "voxel 0 lies half a voxel of b outside it: clamped");
    }

    #[test]
    fn layouts_place_every_view_inside() {
        let full = egui::Rect::from_min_size(egui::pos2(10.0, 20.0), egui::vec2(900.0, 600.0));
        for l in Layout::ALL {
            let r = l.rects(full, 2.0);
            for (k, v) in r.iter().enumerate() {
                if l == Layout::Row && k == 1 {
                    assert!(!v.is_positive());
                    continue;
                }
                assert!(full.contains_rect(*v) && v.width() > 100.0 && v.height() > 100.0, "{l:?} {k} {v:?}");
            }
            // No two views overlap.
            for i in 0..4 {
                for j in i + 1..4 {
                    assert!(!r[i].intersects(r[j]) || !r[i].is_positive() || !r[j].is_positive(), "{l:?}");
                }
            }
        }
    }
}
