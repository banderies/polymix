//! The editing tools ("effects", as 3D Slicer calls them): their settings, the brush, and the
//! work they do on the whole volume, which runs in the background.

use crate::segmentation::Mode;
use glam::DVec3;
use mm_core::Mesh;
use mm_segment::{mask_to_mesh, ops, Connectivity, Mask, MeshParams};
use mm_volume::geometry::Geometry;
use mm_volume::load::LoadedVolume;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Effect {
    Threshold,
    Paint,
    Erase,
    FloodFill,
    Islands,
    Margin,
    Smoothing,
    Logical,
}

impl Effect {
    pub const ALL: [Effect; 8] = [Effect::Threshold, Effect::Paint, Effect::Erase, Effect::FloodFill, Effect::Islands, Effect::Margin, Effect::Smoothing, Effect::Logical];

    pub fn label(self) -> &'static str {
        match self {
            Effect::Threshold => "Threshold",
            Effect::Paint => "Paint",
            Effect::Erase => "Erase",
            Effect::FloodFill => "Flood fill",
            Effect::Islands => "Islands",
            Effect::Margin => "Margin",
            Effect::Smoothing => "Smoothing",
            Effect::Logical => "Logical",
        }
    }

    /// Name in test scripts.
    pub fn key(self) -> &'static str {
        match self {
            Effect::Threshold => "threshold",
            Effect::Paint => "paint",
            Effect::Erase => "erase",
            Effect::FloodFill => "flood",
            Effect::Islands => "islands",
            Effect::Margin => "margin",
            Effect::Smoothing => "smoothing",
            Effect::Logical => "logical",
        }
    }

    pub fn hint(self) -> &'static str {
        match self {
            Effect::Threshold => "The segment becomes the voxels in an intensity range",
            Effect::Paint => "Paint the segment with a brush (left drag)",
            Effect::Erase => "Erase the segment with a brush (left drag)",
            Effect::FloodFill => "Click a structure: the connected voxels of similar intensity are added",
            Effect::Islands => "Keep or remove separate pieces of the segment",
            Effect::Margin => "Grow or shrink the segment by a distance",
            Effect::Smoothing => "Remove bumps and fill gaps",
            Effect::Logical => "Combine with another segment, invert, clear or fill",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IslandOp {
    KeepLargest,
    RemoveSmall,
    Split,
    KeepSelected,
    RemoveSelected,
    AddSelected,
}

impl IslandOp {
    pub const ALL: [IslandOp; 6] = [IslandOp::KeepLargest, IslandOp::RemoveSmall, IslandOp::Split, IslandOp::KeepSelected, IslandOp::RemoveSelected, IslandOp::AddSelected];

    pub fn label(self) -> &'static str {
        match self {
            IslandOp::KeepLargest => "Keep largest island",
            IslandOp::RemoveSmall => "Remove small islands",
            IslandOp::Split => "Split islands to segments",
            IslandOp::KeepSelected => "Keep selected island",
            IslandOp::RemoveSelected => "Remove selected island",
            IslandOp::AddSelected => "Add selected island",
        }
    }

    pub fn key(self) -> &'static str {
        match self {
            IslandOp::KeepLargest => "largest",
            IslandOp::RemoveSmall => "small",
            IslandOp::Split => "split",
            IslandOp::KeepSelected => "keep",
            IslandOp::RemoveSelected => "remove",
            IslandOp::AddSelected => "add",
        }
    }

    /// Chosen by clicking in a view.
    pub fn by_click(self) -> bool {
        matches!(self, IslandOp::KeepSelected | IslandOp::RemoveSelected | IslandOp::AddSelected)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SmoothMethod {
    Median,
    Opening,
    Closing,
    FillHoles,
}

impl SmoothMethod {
    pub const ALL: [SmoothMethod; 4] = [SmoothMethod::Median, SmoothMethod::Opening, SmoothMethod::Closing, SmoothMethod::FillHoles];

    pub fn label(self) -> &'static str {
        match self {
            SmoothMethod::Median => "Median",
            SmoothMethod::Opening => "Opening (remove extrusions)",
            SmoothMethod::Closing => "Closing (fill gaps)",
            SmoothMethod::FillHoles => "Fill holes",
        }
    }

    pub fn key(self) -> &'static str {
        match self {
            SmoothMethod::Median => "median",
            SmoothMethod::Opening => "open",
            SmoothMethod::Closing => "close",
            SmoothMethod::FillHoles => "fill",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogicalOp {
    Copy,
    Add,
    Subtract,
    Intersect,
    Invert,
    Clear,
    Fill,
}

impl LogicalOp {
    pub const ALL: [LogicalOp; 7] = [LogicalOp::Copy, LogicalOp::Add, LogicalOp::Subtract, LogicalOp::Intersect, LogicalOp::Invert, LogicalOp::Clear, LogicalOp::Fill];

    pub fn label(self) -> &'static str {
        match self {
            LogicalOp::Copy => "Copy",
            LogicalOp::Add => "Add",
            LogicalOp::Subtract => "Subtract",
            LogicalOp::Intersect => "Intersect",
            LogicalOp::Invert => "Invert",
            LogicalOp::Clear => "Clear",
            LogicalOp::Fill => "Fill",
        }
    }

    pub fn key(self) -> &'static str {
        match self {
            LogicalOp::Copy => "copy",
            LogicalOp::Add => "add",
            LogicalOp::Subtract => "subtract",
            LogicalOp::Intersect => "intersect",
            LogicalOp::Invert => "invert",
            LogicalOp::Clear => "clear",
            LogicalOp::Fill => "fill",
        }
    }

    /// Uses a second segment.
    pub fn needs_other(self) -> bool {
        matches!(self, LogicalOp::Copy | LogicalOp::Add | LogicalOp::Subtract | LogicalOp::Intersect)
    }
}

/// The settings of every effect, kept while switching between them.
#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    pub threshold: (f32, f32),
    pub brush_mm: f32,
    /// A ball rather than a disc in the slice.
    pub sphere: bool,
    pub islands: IslandOp,
    /// Islands smaller than this (voxels) are removed, and not split into segments.
    pub min_island: usize,
    /// Flood fill: values within this of the clicked voxel's …
    pub flood_tolerance: f32,
    /// … or in the threshold range.
    pub flood_use_threshold: bool,
    /// Flood fill does not pass through gaps narrower than this (mm; 0 = no limit).
    pub flood_min_diameter_mm: f32,
    /// Grow (> 0) or shrink (< 0).
    pub margin_mm: f32,
    pub smoothing: SmoothMethod,
    pub smoothing_mm: f32,
    pub logical: LogicalOp,
    /// The other segment of a logical operation (its label).
    pub logical_other: Option<u8>,
    pub mesh_smoothing_mm: f32,
    /// Make meshes of every visible segment rather than the selected one.
    pub mesh_all_visible: bool,
}

impl Settings {
    pub fn for_volume(vol: &LoadedVolume, hist: &Histogram) -> Self {
        let lo = hist.otsu().unwrap_or(vol.auto_window.0 as f32);
        Self {
            threshold: (lo, vol.range.1),
            brush_mm: 5.0,
            sphere: false,
            islands: IslandOp::KeepLargest,
            min_island: 1000,
            flood_tolerance: (vol.auto_window.1 * 0.1).max(1e-3) as f32,
            flood_use_threshold: false,
            flood_min_diameter_mm: 0.0,
            margin_mm: 1.0,
            smoothing: SmoothMethod::Median,
            smoothing_mm: 3.0,
            logical: LogicalOp::Add,
            logical_other: None,
            mesh_smoothing_mm: 0.5,
            mesh_all_visible: false,
        }
    }

    /// Effects that take the left button in the slice views (a click or a stroke).
    pub fn takes_left_button(&self, e: Effect) -> bool {
        match e {
            Effect::Paint | Effect::Erase | Effect::FloodFill => true,
            Effect::Islands => self.islands.by_click(),
            _ => false,
        }
    }
}

// ---------------------------------------------------------------------- histogram

/// Intensity histogram of a volume (a sample of up to a million voxels).
#[derive(Clone, Debug, PartialEq)]
pub struct Histogram {
    pub range: (f32, f32),
    pub bins: Vec<u32>,
}

impl Histogram {
    pub fn of(vol: &LoadedVolume, n: usize) -> Self {
        let (lo, hi) = vol.range;
        let mut bins = vec![0u32; n];
        let step = (vol.values.len() / 1_000_000).max(1);
        let scale = n as f32 / (hi - lo).max(f32::MIN_POSITIVE);
        for &v in vol.values.iter().step_by(step).filter(|v| v.is_finite()) {
            bins[(((v - lo) * scale) as usize).min(n - 1)] += 1;
        }
        Self { range: (lo, hi), bins }
    }

    /// Value at the lower edge of bin `b`.
    pub fn value(&self, b: f32) -> f32 {
        self.range.0 + (self.range.1 - self.range.0) * b / self.bins.len() as f32
    }

    /// Otsu's threshold: the value that best splits the histogram in two classes.
    pub fn otsu(&self) -> Option<f32> {
        let total: f64 = self.bins.iter().map(|&c| c as f64).sum();
        if total == 0.0 {
            return None;
        }
        let sum_all: f64 = self.bins.iter().enumerate().map(|(i, &c)| i as f64 * c as f64).sum();
        let (mut w0, mut sum0, mut best, mut at) = (0.0, 0.0, -1.0, None);
        for (i, &c) in self.bins.iter().enumerate().take(self.bins.len() - 1) {
            w0 += c as f64;
            sum0 += i as f64 * c as f64;
            let w1 = total - w0;
            if w0 == 0.0 || w1 == 0.0 {
                continue;
            }
            let (m0, m1) = (sum0 / w0, (sum_all - sum0) / w1);
            let between = w0 * w1 * (m0 - m1) * (m0 - m1);
            if between > best {
                best = between;
                at = Some(i + 1);
            }
        }
        at.map(|b| self.value(b as f32))
    }
}

// ---------------------------------------------------------------------- brush

/// The voxels of a brush dab of radius `r` mm centred on `c` (voxel coordinates): a ball, or
/// with `disc_axis` a disc in the slice across that voxel axis.
pub fn dab(dims: [usize; 3], spacing: [f64; 3], c: DVec3, r: f64, disc_axis: Option<usize>, out: &mut Vec<usize>) {
    let mut lo = [0usize; 3];
    let mut hi = [0usize; 3];
    for a in 0..3 {
        if disc_axis == Some(a) {
            let k = c[a].round().clamp(0.0, dims[a] as f64 - 1.0) as usize;
            (lo[a], hi[a]) = (k, k);
        } else {
            let e = (r / spacing[a]).ceil();
            lo[a] = (c[a] - e).round().max(0.0) as usize;
            hi[a] = ((c[a] + e).round() as usize).min(dims[a] - 1);
        }
    }
    let r2 = r * r;
    let start = out.len();
    for k in lo[2]..=hi[2] {
        for j in lo[1]..=hi[1] {
            for i in lo[0]..=hi[0] {
                let d = |a: usize, x: usize| if disc_axis == Some(a) { 0.0 } else { (x as f64 - c[a]) * spacing[a] };
                if d(0, i).powi(2) + d(1, j).powi(2) + d(2, k).powi(2) <= r2 {
                    out.push((k * dims[1] + j) * dims[0] + i);
                }
            }
        }
    }
    // A brush smaller than a voxel still marks the voxel under it.
    if out.len() == start {
        let v = [0, 1, 2].map(|a| c[a].round().clamp(0.0, dims[a] as f64 - 1.0) as usize);
        out.push((v[2] * dims[1] + v[1]) * dims[0] + v[0]);
    }
}

/// The voxels of dabs from `a` to `b` close enough together to leave no gaps.
pub fn stroke(dims: [usize; 3], spacing: [f64; 3], a: DVec3, b: DVec3, r: f64, disc_axis: Option<usize>) -> Vec<usize> {
    let mm = (0..3).map(|k| ((b[k] - a[k]) * spacing[k]).powi(2)).sum::<f64>().sqrt();
    let step = (r * 0.5).max(spacing.iter().cloned().fold(f64::INFINITY, f64::min) * 0.5);
    let n = (mm / step).ceil().max(1.0) as usize;
    let mut out = Vec::new();
    for s in 0..=n {
        dab(dims, spacing, a.lerp(b, s as f64 / n as f64), r, disc_axis, &mut out);
    }
    out.sort_unstable();
    out.dedup();
    out
}

// ---------------------------------------------------------------------- background work

/// What finished background work does.
pub enum Outcome {
    /// Change segment `label` by `mask`.
    Edit { label: u8, mask: Mask, mode: Mode, what: String },
    /// Segment `label` becomes `keep`; each of `islands` becomes a new segment.
    Split { label: u8, keep: Mask, islands: Vec<Mask> },
    Meshes(Vec<(String, Mesh)>),
    Saved { path: String, version: u64 },
}

/// One whole-volume operation on segment `label` (its voxels in `segment`).
pub struct Op {
    pub label: u8,
    pub segment: Mask,
    /// The other segment of a logical operation.
    pub other: Option<Mask>,
    /// The voxel clicked, for click effects.
    pub seed: Option<[usize; 3]>,
}

/// Do `effect` with `settings` on the volume.
pub fn run(effect: Effect, s: &Settings, vol: &Arc<LoadedVolume>, op: Op) -> anyhow::Result<Outcome> {
    let spacing = vol.geometry.spacing;
    let Op { label, segment, other, seed } = op;
    let edit = |mask: Mask, mode: Mode, what: &str| Ok(Outcome::Edit { label, mask, mode, what: what.to_string() });
    let conn = Connectivity::Six;
    match effect {
        Effect::Threshold => edit(ops::threshold(vol, s.threshold.0, s.threshold.1), Mode::Set, "Threshold"),
        Effect::FloodFill => {
            let seed = seed.ok_or_else(|| anyhow::anyhow!("click a voxel"))?;
            let v = vol.get(seed[0], seed[1], seed[2]);
            let (lo, hi) = if s.flood_use_threshold { s.threshold } else { (v - s.flood_tolerance, v + s.flood_tolerance) };
            if v < lo || v > hi {
                anyhow::bail!("the clicked voxel ({v}) is outside the range {lo} – {hi}");
            }
            let mut grown = ops::region_grow(vol, seed, lo, hi, None, conn);
            if s.flood_min_diameter_mm > 0.0 {
                // Cut necks narrower than the diameter, then keep the clicked piece.
                ops::open(&mut grown, s.flood_min_diameter_mm as f64 * 0.5, spacing);
                if !grown.get(seed[0], seed[1], seed[2]) {
                    anyhow::bail!("the clicked structure is narrower than the minimum diameter");
                }
                ops::island_at(&mut grown, seed, true, conn);
            }
            edit(grown, Mode::Add, "Flood fill")
        }
        Effect::Islands => {
            let mut m = segment;
            match s.islands {
                IslandOp::KeepLargest => ops::keep_largest(&mut m, conn),
                IslandOp::RemoveSmall => {
                    ops::remove_small(&mut m, s.min_island, conn);
                }
                IslandOp::Split => {
                    let (ids, sizes) = ops::components(&m, conn);
                    let mut order: Vec<usize> = (1..sizes.len()).filter(|&c| sizes[c] >= s.min_island).collect();
                    order.sort_by_key(|&c| std::cmp::Reverse(sizes[c]));
                    let piece = |c: usize| Mask { dims: m.dims, data: ids.iter().map(|&i| (i as usize == c) as u8).collect() };
                    let keep = order.first().map(|&c| piece(c)).unwrap_or_else(|| Mask::new(m.dims));
                    let islands = order.iter().skip(1).take(200).map(|&c| piece(c)).collect();
                    return Ok(Outcome::Split { label, keep, islands });
                }
                IslandOp::KeepSelected | IslandOp::RemoveSelected => {
                    let seed = seed.ok_or_else(|| anyhow::anyhow!("click an island"))?;
                    if !m.get(seed[0], seed[1], seed[2]) {
                        anyhow::bail!("there is no island of this segment there");
                    }
                    ops::island_at(&mut m, seed, s.islands == IslandOp::KeepSelected, conn);
                }
                IslandOp::AddSelected => {
                    let seed = seed.ok_or_else(|| anyhow::anyhow!("click an empty region"))?;
                    let mut empty = m.clone();
                    empty.invert();
                    if !empty.get(seed[0], seed[1], seed[2]) {
                        anyhow::bail!("that voxel is already in the segment");
                    }
                    ops::island_at(&mut empty, seed, true, conn);
                    return edit(empty, Mode::Add, "Add selected island");
                }
            }
            edit(m, Mode::Set, s.islands.label())
        }
        Effect::Margin => {
            let mut m = segment;
            if s.margin_mm >= 0.0 {
                ops::grow(&mut m, s.margin_mm as f64, spacing);
            } else {
                ops::shrink(&mut m, -s.margin_mm as f64, spacing);
            }
            edit(m, Mode::Set, "Margin")
        }
        Effect::Smoothing => {
            let mut m = segment;
            let radius = s.smoothing_mm as f64 * 0.5;
            match s.smoothing {
                SmoothMethod::Median => {
                    let min_spacing = spacing.iter().cloned().fold(f64::INFINITY, f64::min);
                    ops::majority(&mut m, (radius / min_spacing).round().max(1.0) as usize);
                }
                SmoothMethod::Opening => ops::open(&mut m, radius, spacing),
                SmoothMethod::Closing => ops::close(&mut m, radius, spacing),
                SmoothMethod::FillHoles => ops::fill_holes(&mut m),
            }
            edit(m, Mode::Set, s.smoothing.label())
        }
        Effect::Logical => {
            let mut m = segment;
            let other = || other.clone().ok_or_else(|| anyhow::anyhow!("choose the other segment"));
            match s.logical {
                LogicalOp::Copy => m = other()?,
                LogicalOp::Add => return edit(other()?, Mode::Add, "Add"),
                LogicalOp::Subtract => return edit(other()?, Mode::Remove, "Subtract"),
                LogicalOp::Intersect => m.intersect(&other()?),
                LogicalOp::Invert => m.invert(),
                LogicalOp::Clear => m = Mask::new(m.dims),
                LogicalOp::Fill => m.data.fill(1),
            }
            edit(m, Mode::Set, s.logical.label())
        }
        Effect::Paint | Effect::Erase => anyhow::bail!("{} works with the brush", effect.label()),
    }
}

/// Closed surface meshes of segments (name, voxels), in LPS millimetres as 3D Slicer
/// exports models.
pub fn meshes(segments: Vec<(String, Mask)>, geometry: &Geometry, smoothing_mm: f64, progress: &dyn Fn(f32) -> bool) -> anyhow::Result<Vec<(String, Mesh)>> {
    let n = segments.len().max(1);
    let mut out = Vec::new();
    for (k, (name, mask)) in segments.into_iter().enumerate() {
        if !progress(k as f32 / n as f32) {
            anyhow::bail!("cancelled");
        }
        if mask.count() == 0 {
            continue;
        }
        let params = MeshParams { smoothing_mm, cell_mm: None, lps: true };
        out.push((name, mask_to_mesh(&mask, geometry, &params, None)?));
    }
    if out.is_empty() {
        anyhow::bail!("the segment is empty");
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brush_dabs() {
        let dims = [9, 9, 9];
        let mut v = Vec::new();
        dab(dims, [1.0; 3], DVec3::splat(4.0), 1.5, None, &mut v);
        assert_eq!(v.len(), 19, "centre, 6 faces and 12 edges");
        v.clear();
        dab(dims, [1.0; 3], DVec3::splat(4.0), 1.5, Some(2), &mut v);
        assert_eq!(v.len(), 9, "a 3 × 3 disc in one slice");
        assert!(v.iter().all(|&i| i / 81 == 4));
        // Anisotropic voxels: 2 mm slices reach one slice less far.
        v.clear();
        dab(dims, [1.0, 1.0, 2.0], DVec3::splat(4.0), 2.0, None, &mut v);
        assert!(v.iter().all(|&i| (3..=5).contains(&(i / 81))));
        // Smaller than a voxel, near the edge.
        v.clear();
        dab(dims, [1.0; 3], DVec3::new(0.2, 8.4, 3.6), 0.1, None, &mut v);
        assert_eq!(v, vec![(4 * 9 + 8) * 9]);
    }

    #[test]
    fn strokes_leave_no_gaps() {
        let s = stroke([40, 5, 5], [1.0; 3], DVec3::new(2.0, 2.0, 2.0), DVec3::new(37.0, 2.0, 2.0), 1.0, Some(2));
        let row: Vec<usize> = s.iter().filter(|&&i| i / 40 % 5 == 2).map(|&i| i % 40).collect();
        assert_eq!(row, (1..=38).collect::<Vec<_>>());
    }

    #[test]
    fn otsu_splits_two_classes() {
        let mut bins = vec![0u32; 100];
        for b in 10..20 {
            bins[b] = 50;
        }
        for b in 70..80 {
            bins[b] = 50;
        }
        let h = Histogram { range: (0.0, 100.0), bins };
        let t = h.otsu().unwrap();
        assert!((20.0..=70.0).contains(&t), "{t}");
    }

    #[test]
    fn operations() {
        let g = Geometry::from_axes([10, 10, 10, 1], [1.0; 3], [DVec3::X, DVec3::Y, DVec3::Z], DVec3::ZERO);
        // Two bright blocks: 4 × 4 × 4 and 2 × 2 × 2.
        let mut values = vec![0.0f32; 1000];
        for k in 0..10 {
            for j in 0..10 {
                for i in 0..10 {
                    if (i < 4 && j < 4 && k < 4) || ((7..9).contains(&i) && (7..9).contains(&j) && (7..9).contains(&k)) {
                        values[(k * 10 + j) * 10 + i] = 100.0;
                    }
                }
            }
        }
        let vol = Arc::new(LoadedVolume::from_values("t", [10, 10, 10], values, g));
        let mut s = Settings::for_volume(&vol, &Histogram::of(&vol, 64));
        s.threshold = (50.0, 150.0);
        let empty = || Op { label: 1, segment: Mask::new([10, 10, 10]), other: None, seed: None };
        let Ok(Outcome::Edit { mask, mode: Mode::Set, .. }) = run(Effect::Threshold, &s, &vol, empty()) else { panic!() };
        assert_eq!(mask.count(), 64 + 8);
        let both = mask;
        // Islands: keep the largest; split.
        let Ok(Outcome::Edit { mask, .. }) = run(Effect::Islands, &s, &vol, Op { segment: both.clone(), ..empty() }) else { panic!() };
        assert_eq!(mask.count(), 64);
        s.islands = IslandOp::Split;
        s.min_island = 1;
        let Ok(Outcome::Split { keep, islands, .. }) = run(Effect::Islands, &s, &vol, Op { segment: both.clone(), ..empty() }) else { panic!() };
        assert_eq!((keep.count(), islands.len(), islands[0].count()), (64, 1, 8));
        // Flood fill from the small block only reaches it.
        s.flood_tolerance = 10.0;
        let Ok(Outcome::Edit { mask, mode: Mode::Add, .. }) = run(Effect::FloodFill, &s, &vol, Op { seed: Some([7, 7, 7]), ..empty() }) else { panic!() };
        assert_eq!(mask.count(), 8);
        // A 3 mm minimum diameter stops it from taking the 2-voxel block.
        s.flood_min_diameter_mm = 3.0;
        assert!(run(Effect::FloodFill, &s, &vol, Op { seed: Some([7, 7, 7]), ..empty() }).is_err());
        // Logical: invert, subtract.
        s.logical = LogicalOp::Invert;
        let Ok(Outcome::Edit { mask, .. }) = run(Effect::Logical, &s, &vol, Op { segment: both.clone(), ..empty() }) else { panic!() };
        assert_eq!(mask.count(), 1000 - 72);
        s.logical = LogicalOp::Subtract;
        assert!(run(Effect::Logical, &s, &vol, empty()).is_err(), "needs the other segment");
    }
}
