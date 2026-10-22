//! The editing tools ("effects", as 3D Slicer calls them): their settings, the brush, and the
//! work they do on the whole volume, which runs in the background.

use crate::segmentation::Mode;
use glam::DVec3;
use mm_core::Mesh;
use mm_segment::{mask_to_mesh, ops, AutoThreshold, Connectivity, HollowSide, Mask, MeshParams};
use mm_volume::geometry::Geometry;
use mm_volume::load::LoadedVolume;
use rayon::prelude::*;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Effect {
    Threshold,
    Paint,
    Draw,
    Erase,
    LevelTracing,
    FloodFill,
    Scissors,
    Islands,
    Margin,
    Smoothing,
    Hollow,
    Logical,
    GrowFromSeeds,
    Watershed,
    FastMarching,
    DrawTube,
    SurfaceCut,
    FillBetweenSlices,
    MaskVolume,
    SplitVolume,
    LevelSet,
    GuidedVessel,
    Engrave,
    ThresholdGrow,
    SelectPart,
}

impl Effect {
    pub const ALL: [Effect; 25] = [
        Effect::Threshold,
        Effect::Paint,
        Effect::Draw,
        Effect::Erase,
        Effect::LevelTracing,
        Effect::FloodFill,
        Effect::Scissors,
        Effect::Islands,
        Effect::Margin,
        Effect::Smoothing,
        Effect::Hollow,
        Effect::Logical,
        Effect::GrowFromSeeds,
        Effect::Watershed,
        Effect::FastMarching,
        Effect::DrawTube,
        Effect::SurfaceCut,
        Effect::FillBetweenSlices,
        Effect::MaskVolume,
        Effect::SplitVolume,
        Effect::LevelSet,
        Effect::GuidedVessel,
        Effect::Engrave,
        Effect::ThresholdGrow,
        Effect::SelectPart,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Effect::Threshold => "Threshold",
            Effect::Paint => "Paint",
            Effect::Draw => "Draw",
            Effect::Erase => "Erase",
            Effect::LevelTracing => "Level tracing",
            Effect::FloodFill => "Flood fill",
            Effect::Scissors => "Scissors",
            Effect::Islands => "Islands",
            Effect::Margin => "Margin",
            Effect::Smoothing => "Smoothing",
            Effect::Hollow => "Hollow",
            Effect::Logical => "Logical",
            Effect::GrowFromSeeds => "Grow from seeds",
            Effect::Watershed => "Watershed",
            Effect::FastMarching => "Fast marching",
            Effect::DrawTube => "Draw tube",
            Effect::SurfaceCut => "Surface cut",
            Effect::FillBetweenSlices => "Fill between slices",
            Effect::MaskVolume => "Mask volume",
            Effect::SplitVolume => "Split volume",
            Effect::LevelSet => "Level set",
            Effect::GuidedVessel => "Guided vessel",
            Effect::Engrave => "Engrave",
            Effect::ThresholdGrow => "Threshold grow",
            Effect::SelectPart => "Select part",
        }
    }

    /// Previewed in the views before Apply (from the visible segments as seeds).
    pub fn seeded(self) -> bool {
        matches!(self, Effect::GrowFromSeeds | Effect::Watershed | Effect::FillBetweenSlices)
    }

    /// Works on points clicked in the views.
    pub fn takes_points(self) -> bool {
        matches!(self, Effect::DrawTube | Effect::SurfaceCut | Effect::GuidedVessel)
    }

    /// Name in test scripts.
    pub fn key(self) -> &'static str {
        match self {
            Effect::Threshold => "threshold",
            Effect::Paint => "paint",
            Effect::Draw => "draw",
            Effect::Erase => "erase",
            Effect::LevelTracing => "trace",
            Effect::FloodFill => "flood",
            Effect::Scissors => "scissors",
            Effect::Islands => "islands",
            Effect::Margin => "margin",
            Effect::Smoothing => "smoothing",
            Effect::Hollow => "hollow",
            Effect::Logical => "logical",
            Effect::GrowFromSeeds => "grow",
            Effect::Watershed => "watershed",
            Effect::FastMarching => "march",
            Effect::DrawTube => "tube",
            Effect::SurfaceCut => "surface",
            Effect::FillBetweenSlices => "fill",
            Effect::MaskVolume => "mask",
            Effect::SplitVolume => "split",
            Effect::LevelSet => "levelset",
            Effect::GuidedVessel => "guided",
            Effect::Engrave => "engrave",
            Effect::ThresholdGrow => "tgrow",
            Effect::SelectPart => "part",
        }
    }

    pub fn hint(self) -> &'static str {
        match self {
            Effect::Threshold => "The segment becomes the voxels in an intensity range",
            Effect::Paint => "Paint the segment with a brush (left drag)",
            Effect::Draw => "Click points or drag a line around a region in a slice; double-click or Enter fills it",
            Effect::Erase => "Erase the segment with a brush (left drag)",
            Effect::LevelTracing => "The region under the pointer brighter than it is outlined in its slice; a click adds it",
            Effect::FloodFill => "Click a structure: the connected voxels of similar intensity are added",
            Effect::Scissors => "Drag a shape in a slice: fill or erase inside or outside it, through a range of slices",
            Effect::Islands => "Keep or remove separate pieces of the segment",
            Effect::Margin => "Grow or shrink the segment by a distance",
            Effect::Smoothing => "Remove bumps and fill gaps",
            Effect::Hollow => "Make the segment a shell of a thickness: a wall around a lumen, or a lining inside a structure",
            Effect::Logical => "Combine with another segment, invert, clear or fill",
            Effect::GrowFromSeeds => "Paint seeds in two or more segments (the structure and its surroundings): each grows over the voxels most like its own",
            Effect::Watershed => "Paint seeds in two or more segments: they flood the image until they meet at its edges",
            Effect::FastMarching => "Grow the segment over similar voxels, nearest first, to a size",
            Effect::DrawTube => "Click points along a vessel, catheter or nerve: a tube of a radius through them",
            Effect::SurfaceCut => "Click points on a structure's surface: the closed shape through them",
            Effect::FillBetweenSlices => "Segment every few slices; the slices in between are filled by interpolating the shapes",
            Effect::MaskVolume => "A new image: the volume with a value filled in inside or outside the segment (e.g. to blank out what is not the vessel)",
            Effect::SplitVolume => "A new image per visible segment: the volume cropped to the segment (with a margin), optionally blank outside it",
            Effect::LevelSet => "Grow (or shrink) the segment to the vessel's wall: a geodesic active contour that stays smooth, within the threshold range or following the overlay (e.g. vesselness)",
            Effect::GuidedVessel => "Click points along a vessel: it is segmented from the first to the last, following the vessel between them (within the threshold range)",
            Effect::Engrave => "Text cut into the segment or raised from it, where you click in a slice view (labels for printed models)",
            Effect::ThresholdGrow => "⌘ / Ctrl + right drag from a voxel in the threshold range: the segment grows over the connected voxels in the range, nearest first, as far as you drag right; drag back to take it back",
            Effect::SelectPart => "⌘ / Ctrl + right drag from a voxel of a segment: the part of it within reach along the segment is selected, to erase, keep, move to another segment or measure",
        }
    }
}

/// How Flood fill's region gets its edges (as 3D Slicer's Local threshold algorithms).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refine {
    /// As thresholded ("Masking").
    None,
    /// Decided by Grow from seeds within the feature size of the thresholded edge.
    GrowCut,
    /// Decided by Watershed there.
    Watershed,
}

impl Refine {
    pub const ALL: [Refine; 3] = [Refine::None, Refine::GrowCut, Refine::Watershed];

    pub fn label(self) -> &'static str {
        match self {
            Refine::None => "As thresholded",
            Refine::GrowCut => "Refined by Grow from seeds",
            Refine::Watershed => "Refined by Watershed",
        }
    }

    pub fn key(self) -> &'static str {
        match self {
            Refine::None => "none",
            Refine::GrowCut => "grow",
            Refine::Watershed => "watershed",
        }
    }
}

/// Where Mask volume fills in its values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaskFill {
    Inside,
    Outside,
    /// Both: a binary image.
    Both,
}

impl MaskFill {
    pub const ALL: [MaskFill; 3] = [MaskFill::Inside, MaskFill::Outside, MaskFill::Both];

    pub fn label(self) -> &'static str {
        match self {
            MaskFill::Inside => "Fill inside",
            MaskFill::Outside => "Fill outside",
            MaskFill::Both => "Fill inside and outside",
        }
    }

    pub fn key(self) -> &'static str {
        match self {
            MaskFill::Inside => "inside",
            MaskFill::Outside => "outside",
            MaskFill::Both => "both",
        }
    }
}

/// How an automatic threshold sets the range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AutoMode {
    /// From the threshold to the largest value (brighter structures).
    Lower,
    /// From the smallest value to the threshold (darker structures).
    Upper,
}

/// What Scissors does with its shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScissorsOp {
    EraseInside,
    EraseOutside,
    FillInside,
    FillOutside,
}

impl ScissorsOp {
    pub const ALL: [ScissorsOp; 4] = [ScissorsOp::EraseInside, ScissorsOp::EraseOutside, ScissorsOp::FillInside, ScissorsOp::FillOutside];

    pub fn label(self) -> &'static str {
        match self {
            ScissorsOp::EraseInside => "Erase inside",
            ScissorsOp::EraseOutside => "Erase outside",
            ScissorsOp::FillInside => "Fill inside",
            ScissorsOp::FillOutside => "Fill outside",
        }
    }

    pub fn key(self) -> &'static str {
        match self {
            ScissorsOp::EraseInside => "erase_inside",
            ScissorsOp::EraseOutside => "erase_outside",
            ScissorsOp::FillInside => "fill_inside",
            ScissorsOp::FillOutside => "fill_outside",
        }
    }
}

/// The shape Scissors draws.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScissorsShape {
    FreeForm,
    Circle,
    Rectangle,
}

impl ScissorsShape {
    pub const ALL: [ScissorsShape; 3] = [ScissorsShape::FreeForm, ScissorsShape::Circle, ScissorsShape::Rectangle];

    pub fn label(self) -> &'static str {
        match self {
            ScissorsShape::FreeForm => "Free-form",
            ScissorsShape::Circle => "Circle",
            ScissorsShape::Rectangle => "Rectangle",
        }
    }

    pub fn key(self) -> &'static str {
        match self {
            ScissorsShape::FreeForm => "free",
            ScissorsShape::Circle => "circle",
            ScissorsShape::Rectangle => "rectangle",
        }
    }
}

/// The slices Scissors cuts through.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScissorsRange {
    /// All of them.
    Unlimited,
    /// This one and those towards the patient's superior / anterior / right (the slice's
    /// normal), …
    Positive,
    /// … or the other way.
    Negative,
    /// A slab of a given thickness centred on this slice.
    Symmetric,
}

impl ScissorsRange {
    pub const ALL: [ScissorsRange; 4] = [ScissorsRange::Unlimited, ScissorsRange::Positive, ScissorsRange::Negative, ScissorsRange::Symmetric];

    pub fn label(self) -> &'static str {
        match self {
            ScissorsRange::Unlimited => "Unlimited",
            ScissorsRange::Positive => "Positive",
            ScissorsRange::Negative => "Negative",
            ScissorsRange::Symmetric => "Symmetric",
        }
    }

    pub fn key(self) -> &'static str {
        match self {
            ScissorsRange::Unlimited => "unlimited",
            ScissorsRange::Positive => "positive",
            ScissorsRange::Negative => "negative",
            ScissorsRange::Symmetric => "symmetric",
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
    Gaussian,
    /// Every visible segment together, so that no gaps or overlaps open between them.
    Joint,
}

impl SmoothMethod {
    pub const ALL: [SmoothMethod; 6] = [SmoothMethod::Median, SmoothMethod::Opening, SmoothMethod::Closing, SmoothMethod::FillHoles, SmoothMethod::Gaussian, SmoothMethod::Joint];

    pub fn label(self) -> &'static str {
        match self {
            SmoothMethod::Median => "Median",
            SmoothMethod::Opening => "Opening (remove extrusions)",
            SmoothMethod::Closing => "Closing (fill gaps)",
            SmoothMethod::FillHoles => "Fill holes",
            SmoothMethod::Gaussian => "Gaussian",
            SmoothMethod::Joint => "Joint (visible segments together)",
        }
    }

    pub fn key(self) -> &'static str {
        match self {
            SmoothMethod::Median => "median",
            SmoothMethod::Opening => "open",
            SmoothMethod::Closing => "close",
            SmoothMethod::FillHoles => "fill",
            SmoothMethod::Gaussian => "gaussian",
            SmoothMethod::Joint => "joint",
        }
    }

    /// The size is a Gaussian's standard deviation rather than a kernel's width.
    pub fn sigma(self) -> bool {
        matches!(self, SmoothMethod::Gaussian | SmoothMethod::Joint)
    }
}

/// The shape drawn on a slice for Threshold's local histogram.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LocalShape {
    Box,
    Circle,
    Line,
}

impl LocalShape {
    pub const ALL: [LocalShape; 3] = [LocalShape::Box, LocalShape::Circle, LocalShape::Line];

    pub fn label(self) -> &'static str {
        match self {
            LocalShape::Box => "Box",
            LocalShape::Circle => "Circle",
            LocalShape::Line => "Line",
        }
    }

    pub fn key(self) -> &'static str {
        match self {
            LocalShape::Box => "box",
            LocalShape::Circle => "circle",
            LocalShape::Line => "line",
        }
    }
}

/// Where an end of the range set from a local histogram comes from: the region's extreme
/// (minimum or maximum), a percentile (5th or 95th), or its average.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LocalPick {
    Extreme,
    Percentile,
    Average,
}

impl LocalPick {
    pub const ALL: [LocalPick; 3] = [LocalPick::Extreme, LocalPick::Percentile, LocalPick::Average];

    pub fn label(self, upper: bool) -> &'static str {
        match (self, upper) {
            (LocalPick::Extreme, false) => "Minimum",
            (LocalPick::Extreme, true) => "Maximum",
            (LocalPick::Percentile, false) => "Lower (5%)",
            (LocalPick::Percentile, true) => "Upper (95%)",
            (LocalPick::Average, _) => "Average",
        }
    }

    pub fn key(self) -> &'static str {
        match self {
            LocalPick::Extreme => "extreme",
            LocalPick::Percentile => "percentile",
            LocalPick::Average => "average",
        }
    }
}

/// The threshold range from a region's values (sorted): its lower end and upper end as picked.
pub fn local_range(sorted: &[f32], lower: LocalPick, upper: LocalPick) -> Option<(f32, f32)> {
    let (first, last) = (*sorted.first()?, *sorted.last()?);
    let mean = (sorted.iter().map(|&v| v as f64).sum::<f64>() / sorted.len() as f64) as f32;
    let at = |q: f64| sorted[((sorted.len() - 1) as f64 * q).round() as usize];
    let lo = match lower {
        LocalPick::Extreme => first,
        LocalPick::Percentile => at(0.05),
        LocalPick::Average => mean,
    };
    let hi = match upper {
        LocalPick::Extreme => last,
        LocalPick::Percentile => at(0.95),
        LocalPick::Average => mean,
    };
    Some((lo.min(hi), lo.max(hi)))
}

/// Where Hollow's shell lies: (side, label, name in scripts).
pub const HOLLOW_SIDES: [(HollowSide, &str, &str); 3] = [
    (HollowSide::Outside, "Outside the surface (the segment is the lumen)", "outside"),
    (HollowSide::Medial, "Centred on the surface", "medial"),
    (HollowSide::Inside, "Inside the surface (the segment is the outside)", "inside"),
];

pub fn hollow_side_key(side: HollowSide) -> &'static str {
    HOLLOW_SIDES.iter().find(|h| h.0 == side).map_or("outside", |h| h.2)
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
    /// The brush's size follows the view's: its diameter is `brush_percent` % of the view's
    /// smaller side (zooming changes `brush_mm`).
    pub brush_relative: bool,
    pub brush_percent: f32,
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
    /// Select part's Move: into this segment (label), or a new one.
    pub part_target: Option<u8>,
    pub smoothing: SmoothMethod,
    pub smoothing_mm: f32,
    /// Hollow's shell thickness, mm, and where it lies.
    pub hollow_mm: f32,
    pub hollow_side: HollowSide,
    pub logical: LogicalOp,
    /// The other segment of a logical operation (its label).
    pub logical_other: Option<u8>,
    pub mesh_smoothing_mm: f32,
    /// Make meshes of every visible segment rather than the selected one.
    pub mesh_all_visible: bool,
    pub scissors_op: ScissorsOp,
    pub scissors_shape: ScissorsShape,
    pub scissors_range: ScissorsRange,
    /// Thickness of a symmetric Scissors cut, mm.
    pub scissors_mm: f32,
    /// Level tracing outlines the darker region (values at most the pointer's) instead.
    pub trace_darker: bool,
    /// Margin and Smoothing change every visible segment, not only the selected one.
    pub apply_visible: bool,
    /// Logical operators ignore the masking settings.
    pub bypass_masking: bool,
    /// Draw tube's radius, mm.
    pub tube_mm: f32,
    /// Grow from seeds: how much distance counts besides likeness (0 = none).
    pub seed_locality: f32,
    /// Grow from seeds and Watershed work within this far of the seeds, mm.
    pub seed_margin_mm: f32,
    /// Grow from seeds and Watershed update their preview after every edit of the seeds.
    pub auto_update: bool,
    /// Watershed's object scale (smoothing before the edges are found), mm.
    pub watershed_mm: f32,
    /// Fast marching grows the segment to this share of the volume, %.
    pub march_percent: f32,
    /// Threshold's range from a region drawn on a slice (as 3D Slicer's local histogram):
    /// the shape, and where its lower and upper ends come from.
    pub local_histogram: bool,
    pub local_shape: LocalShape,
    pub local_lower: LocalPick,
    pub local_upper: LocalPick,
    /// Threshold's automatic method, and how it sets the range.
    pub auto_method: AutoThreshold,
    pub auto_mode: AutoMode,
    /// Fill between slices fills between slices of this plane.
    pub fill_plane: mm_volume::slice::Plane,
    /// Mask volume: where it fills, with what inside and outside, and the soft edge (mm, 0 =
    /// sharp).
    pub mask_fill: MaskFill,
    pub mask_inside: f32,
    pub mask_outside: f32,
    pub mask_soft_mm: f32,
    /// Level set: iterations, how far the front may move (mm), the curvature weight (smoothness),
    /// and whether the overlay shown (e.g. vesselness) drives it instead of the threshold range.
    pub ls_iterations: usize,
    pub ls_max_mm: f32,
    pub ls_curvature: f32,
    pub ls_overlay: bool,
    /// Engrave: the text, its letters' height and depth (mm, through the slices), and whether
    /// it is raised (added) rather than cut.
    pub engrave_text: String,
    pub engrave_mm: f32,
    pub engrave_depth_mm: f32,
    pub engrave_raise: bool,
    /// Split volume: the margin kept around each segment (mm), and whether the voxels outside
    /// the segment are filled with `mask_outside`.
    pub split_pad_mm: f32,
    pub split_blank: bool,
    /// Paint and Erase work on single voxels (the brush's size is ignored).
    pub pixel_mode: bool,
    /// Flood fill's edges, and the feature size they are refined within, mm.
    pub flood_refine: Refine,
    pub flood_feature_mm: f32,
}

impl Settings {
    /// The level set's parameters (its range is the threshold's).
    pub fn level_set_params(&self) -> mm_segment::LevelSetParams {
        mm_segment::LevelSetParams { range: self.threshold, iterations: self.ls_iterations, max_distance_mm: self.ls_max_mm as f64, curvature: self.ls_curvature as f64, ..Default::default() }
    }

    pub fn for_volume(vol: &LoadedVolume, hist: &Histogram) -> Self {
        let lo = hist.otsu().unwrap_or(vol.auto_window.0 as f32);
        Self {
            threshold: (lo, vol.range.1),
            brush_mm: 5.0,
            brush_relative: false,
            brush_percent: 5.0,
            sphere: false,
            islands: IslandOp::KeepLargest,
            min_island: 1000,
            flood_tolerance: (vol.auto_window.1 * 0.1).max(1e-3) as f32,
            flood_use_threshold: false,
            flood_min_diameter_mm: 0.0,
            margin_mm: 1.0,
            part_target: None,
            smoothing: SmoothMethod::Median,
            smoothing_mm: 3.0,
            hollow_mm: 2.0,
            hollow_side: HollowSide::Outside,
            logical: LogicalOp::Add,
            logical_other: None,
            mesh_smoothing_mm: 0.5,
            mesh_all_visible: false,
            scissors_op: ScissorsOp::EraseInside,
            scissors_shape: ScissorsShape::FreeForm,
            scissors_range: ScissorsRange::Unlimited,
            scissors_mm: 10.0,
            trace_darker: false,
            apply_visible: false,
            bypass_masking: false,
            tube_mm: 1.5,
            seed_locality: 0.0,
            seed_margin_mm: 10.0,
            auto_update: true,
            watershed_mm: 0.5,
            march_percent: 1.0,
            local_histogram: false,
            local_shape: LocalShape::Box,
            local_lower: LocalPick::Extreme,
            local_upper: LocalPick::Extreme,
            auto_method: AutoThreshold::Otsu,
            auto_mode: AutoMode::Lower,
            fill_plane: mm_volume::slice::Plane::Axial,
            mask_fill: MaskFill::Outside,
            ls_iterations: 300,
            ls_max_mm: 10.0,
            ls_curvature: 0.3,
            ls_overlay: false,
            engrave_text: "L".into(),
            engrave_mm: 8.0,
            engrave_depth_mm: 2.0,
            engrave_raise: false,
            split_pad_mm: 5.0,
            split_blank: false,
            mask_inside: 1.0,
            mask_outside: vol.range.0,
            mask_soft_mm: 0.0,
            pixel_mode: false,
            flood_refine: Refine::None,
            flood_feature_mm: 1.0,
        }
    }

    /// Effects that take the left button in the slice views (a click or a stroke).
    pub fn takes_left_button(&self, e: Effect) -> bool {
        match e {
            Effect::Paint | Effect::Erase | Effect::FloodFill | Effect::Draw | Effect::LevelTracing | Effect::Scissors | Effect::DrawTube | Effect::SurfaceCut | Effect::GuidedVessel | Effect::Engrave => true,
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
    /// Of some values (finite ones), over their own range.
    pub fn of_values(values: &[f32], n: usize) -> Self {
        let finite = values.iter().copied().filter(|v| v.is_finite());
        let (lo, hi) = finite.clone().fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), v| (a.min(v), b.max(v)));
        if lo > hi {
            return Self { range: (0.0, 1.0), bins: vec![0; n] };
        }
        let hi = if hi > lo { hi } else { lo + 1.0 };
        let mut bins = vec![0u32; n];
        let scale = n as f32 / (hi - lo);
        for v in finite {
            bins[(((v - lo) * scale) as usize).min(n - 1)] += 1;
        }
        Self { range: (lo, hi), bins }
    }

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

    /// The automatic threshold of `method`: the value splitting the histogram's two classes.
    pub fn auto(&self, method: AutoThreshold) -> Option<f32> {
        let counts: Vec<u64> = self.bins.iter().map(|&c| c as u64).collect();
        mm_segment::auto_threshold(method, &counts).map(|b| self.value(b as f32 + 1.0))
    }

    /// The range an automatic threshold gives: from it up, or up to it.
    pub fn auto_range(&self, method: AutoThreshold, mode: AutoMode) -> Option<(f32, f32)> {
        let t = self.auto(method)?;
        Some(match mode {
            AutoMode::Lower => (t, self.range.1),
            AutoMode::Upper => (self.range.0, t),
        })
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

/// The voxels of dabs from `a` to `b` close enough together to leave no gaps: one dab when
/// `a` is `b`, and none at `a` when `a_painted` (the stroke's last step ended there and
/// nothing changed since). Where dabs overlap, voxels are listed again (in no order):
/// painting a voxel twice changes nothing, and sorting a large ball's voxels took longer
/// than painting them.
pub fn stroke(dims: [usize; 3], spacing: [f64; 3], a: DVec3, b: DVec3, r: f64, disc_axis: Option<usize>, a_painted: bool) -> Vec<usize> {
    let mut out = Vec::new();
    if a == b {
        if !a_painted {
            dab(dims, spacing, a, r, disc_axis, &mut out);
        }
        return out;
    }
    let mm = (0..3).map(|k| ((b[k] - a[k]) * spacing[k]).powi(2)).sum::<f64>().sqrt();
    let step = (r * 0.5).max(spacing.iter().cloned().fold(f64::INFINITY, f64::min) * 0.5);
    let n = (mm / step).ceil().max(1.0) as usize;
    for s in (a_painted as usize)..=n {
        dab(dims, spacing, a.lerp(b, s as f64 / n as f64), r, disc_axis, &mut out);
    }
    out
}

// ---------------------------------------------------------------------- background work

/// What finished background work does.
pub enum Outcome {
    /// Change segment `label` by `mask`.
    Edit { label: u8, mask: Mask, mode: Mode, what: String },
    /// Several segments changed together (one undo step).
    Edits { edits: Vec<(u8, Mask, Mode)>, what: String },
    /// A seeded effect's result for the label map at `version`: the labels it gives (0
    /// outside its region), for the seed labels `seeds`.
    Seeded { effect: Effect, version: u64, labels: Vec<u8>, seeds: Vec<u8> },
    /// Segment `label` keeps the voxels in slot 1; the `islands` islands (slots 2, 3…) become
    /// new segments ([`crate::segmentation::Segmentation::split`]).
    Split { label: u8, slots: Vec<u8>, islands: usize },
    Meshes(Vec<(String, Mesh)>),
    /// Saved as the labels were at `version` and the segments at `style_version`.
    Saved { path: String, version: u64, style_version: u64 },
    /// A new image saved among the derived images.
    Volume { path: std::path::PathBuf },
    /// A segment's centerline and cross-sections.
    Centerline(Box<crate::vessels::VesselResult>),
    /// Measures of segments (label, statistics), for the segmentation's `version`.
    Stats { version: u64, rows: Vec<(u8, mm_segment::SegmentStats)> },
    /// New images saved among the derived images.
    Volumes { paths: Vec<std::path::PathBuf> },
    /// Segments from a file or a mesh, to add.
    Imported { imported: crate::import::Imported, what: String },
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
            if s.flood_refine != Refine::None {
                grown = refine_edges(vol, &grown, s.flood_refine, s.flood_feature_mm as f64);
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
                    // A byte per voxel for all the pieces (rather than a mask per piece): the
                    // largest is kept (slot 1), the next 200 become segments (slots 2…201).
                    let mut slot = vec![0u8; sizes.len()];
                    for (k, &c) in order.iter().take(201).enumerate() {
                        slot[c] = 1 + k as u8;
                    }
                    let slots = ids.par_iter().map(|&i| slot[i as usize]).collect();
                    return Ok(Outcome::Split { label, slots, islands: order.len().saturating_sub(1).min(200) });
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
                SmoothMethod::Gaussian => mm_segment::gaussian(&mut m, s.smoothing_mm as f64, spacing),
                SmoothMethod::Joint => anyhow::bail!("joint smoothing works on the visible segments together"),
            }
            edit(m, Mode::Set, s.smoothing.label())
        }
        Effect::Hollow => {
            if segment.count() == 0 {
                anyhow::bail!("the segment is empty");
            }
            let mut m = segment;
            mm_segment::hollow(&mut m, s.hollow_mm as f64, s.hollow_side, spacing);
            edit(m, Mode::Set, "Hollow")
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
        Effect::Draw | Effect::LevelTracing | Effect::Scissors | Effect::DrawTube | Effect::SurfaceCut => anyhow::bail!("{} works in a slice view", effect.label()),
        Effect::GrowFromSeeds | Effect::Watershed | Effect::FillBetweenSlices => anyhow::bail!("{} works from the visible segments: use Initialize", effect.label()),
        Effect::SplitVolume => anyhow::bail!("Split volume works on the visible segments"),
        Effect::LevelSet => {
            if segment.count() == 0 {
                anyhow::bail!("the segment is empty: paint or threshold a rough start inside the vessel");
            }
            let grown = mm_segment::level_set(vol, &segment, None, &s.level_set_params(), &|_| true)?;
            edit(grown, Mode::Set, "Level set")
        }
        Effect::GuidedVessel => anyhow::bail!("Guided vessel works on points clicked along the vessel"),
        Effect::Engrave => anyhow::bail!("Engrave works where you click in a slice view"),
        Effect::ThresholdGrow | Effect::SelectPart => anyhow::bail!("{} works by ⌘ / Ctrl + right drag in a slice view", effect.label()),
        Effect::MaskVolume => {
            if segment.count() == 0 {
                anyhow::bail!("the segment is empty");
            }
            let values = mask_volume(vol, &segment, s.mask_fill, s.mask_inside, s.mask_outside, s.mask_soft_mm as f64)?;
            let dir = polymix_workspace::derived_dir().ok_or_else(|| anyhow::anyhow!("no data folder to save into"))?;
            std::fs::create_dir_all(&dir)?;
            let path = polymix_workspace::derived_path(&dir, &vol.name, " masked", ".nii.gz");
            let what = format!("{} masked ({}, inside {}, outside {})", vol.name, s.mask_fill.label().to_lowercase(), s.mask_inside, s.mask_outside);
            mm_volume::write::nifti(&path, mm_volume::write::Data::F32(&values), vol.dims, &vol.geometry, &what, mm_volume::write::Intent::None)?;
            Ok(Outcome::Volume { path })
        }
        Effect::FastMarching => {
            let target = ((s.march_percent as f64 / 100.0) * vol.voxel_count() as f64).round() as usize;
            if segment.count() == 0 {
                anyhow::bail!("paint a few voxels of the structure into the segment first");
            }
            let grown = mm_segment::fast_march(vol, &segment, target.max(segment.count()));
            edit(grown, Mode::Add, "Fast marching")
        }
    }
}

/// Split volume: for each segment (name, label) of the label map `labels`, the volume cropped
/// to the segment's box grown by `pad_mm`, with `blank` (when given) outside the segment,
/// saved among the derived images. Each segment's mask is made in its turn.
pub fn split_volume(vol: &LoadedVolume, labels: &[u8], segments: &[(String, u8)], pad_mm: f64, blank: Option<f32>, progress: &dyn Fn(f32) -> bool) -> anyhow::Result<Vec<std::path::PathBuf>> {
    let dir = polymix_workspace::derived_dir().ok_or_else(|| anyhow::anyhow!("no data folder to save into"))?;
    std::fs::create_dir_all(&dir)?;
    let mut paths = Vec::new();
    for (k, (name, label)) in segments.iter().enumerate() {
        if !progress(k as f32 / segments.len().max(1) as f32) {
            anyhow::bail!("cancelled");
        }
        let mask = Mask::from_labels(vol.dims, labels, *label);
        let Some((lo, hi)) = mask.bounds() else { continue };
        let pad = vol.geometry.spacing.map(|s| (pad_mm / s).ceil().max(0.0) as usize);
        let lo = [0, 1, 2].map(|a| lo[a].saturating_sub(pad[a]));
        let hi = [0, 1, 2].map(|a| (hi[a] + pad[a]).min(vol.dims[a] - 1));
        let mut part = mm_image::grid::crop(vol, lo, hi)?;
        if let Some(b) = blank {
            for z in 0..part.dims[2] {
                for y in 0..part.dims[1] {
                    for x in 0..part.dims[0] {
                        if !mask.get(lo[0] + x, lo[1] + y, lo[2] + z) {
                            let i = part.index(x, y, z);
                            part.values[i] = b;
                        }
                    }
                }
            }
        }
        let safe: String = name.chars().map(|c| if c.is_alphanumeric() || " -_".contains(c) { c } else { '_' }).collect();
        let path = polymix_workspace::derived_path(&dir, &vol.name, &format!(" {}", safe.trim()), ".nii.gz");
        let what = format!("{} cropped to segment {name}", vol.name);
        mm_volume::write::nifti(&path, mm_volume::write::Data::F32(&part.values), part.dims, &part.geometry, &what, mm_volume::write::Intent::None)?;
        paths.push(path);
    }
    if paths.is_empty() {
        anyhow::bail!("the visible segments are empty");
    }
    Ok(paths)
}

/// `region`'s edges decided again by Grow from seeds or Watershed: the region shrunk by
/// `feature_mm` seeds the structure, everything beyond it grown by `feature_mm` seeds the
/// background, and the band between goes to whichever reaches it.
pub fn refine_edges(vol: &LoadedVolume, region: &Mask, how: Refine, feature_mm: f64) -> Mask {
    let spacing = vol.geometry.spacing;
    let f = feature_mm.max(spacing.iter().cloned().fold(f64::INFINITY, f64::min));
    let mut inner = region.clone();
    ops::shrink(&mut inner, f, spacing);
    if inner.count() == 0 {
        return region.clone();
    }
    let mut outer = region.clone();
    ops::grow(&mut outer, f, spacing);
    let mut roi = region.clone();
    ops::grow(&mut roi, 2.0 * f, spacing);
    let seeds: Vec<u8> = inner.data.par_iter().zip(&roi.data).zip(&outer.data).map(|((&i, &r), &o)| if i != 0 { 1 } else if r != 0 && o == 0 { 2 } else { 0 }).collect();
    let out = match how {
        Refine::Watershed => mm_segment::watershed(vol, &seeds, 0.5 * f, Some(&roi), 0.0),
        _ => mm_segment::grow_from_seeds(vol, &seeds, Some(&roi), 0.0, 0.0),
    };
    Mask { dims: region.dims, data: out.par_iter().map(|&l| (l == 1) as u8).collect() }
}

/// The volume's values with `inside` filled in where `segment` is (or `outside` where it is
/// not, or both), the boundary softened over a Gaussian of `soft_mm` (0: sharp).
pub fn mask_volume(vol: &LoadedVolume, segment: &Mask, fill: MaskFill, inside: f32, outside: f32, soft_mm: f64) -> anyhow::Result<Vec<f32>> {
    let w: Vec<f32> = if soft_mm > 0.0 {
        let m: Vec<f32> = segment.data.iter().map(|&x| x as f32).collect();
        let s = vol.geometry.spacing.map(|sp| soft_mm / sp.max(1e-9));
        mm_image::filter::gaussian(&m, vol.dims, s, &|_| true)?
    } else {
        segment.data.iter().map(|&x| x as f32).collect()
    };
    Ok(vol
        .values
        .iter()
        .zip(&w)
        .map(|(&v, &w)| match fill {
            MaskFill::Inside => v + (inside - v) * w,
            MaskFill::Outside => outside + (v - outside) * w,
            MaskFill::Both => outside + (inside - outside) * w,
        })
        .collect())
}

/// Grow from seeds or Watershed from `labels` (the label map with only the seed segments in
/// it), as of `version`.
pub fn run_seeded(effect: Effect, s: &Settings, vol: &Arc<LoadedVolume>, labels: Vec<u8>, version: u64) -> anyhow::Result<Outcome> {
    if effect == Effect::FillBetweenSlices {
        let segments = labels_present(&labels);
        if segments.is_empty() {
            anyhow::bail!("segment a few slices of a visible segment first");
        }
        let axis = mm_volume::slice::plane_axes(&vol.geometry, s.fill_plane).slice;
        let mut out = vec![0u8; labels.len()];
        for (l, voxels) in mm_segment::fill_between_slices(&labels, vol.dims, vol.geometry.spacing, axis, &segments) {
            for i in voxels {
                out[i] = l;
            }
        }
        return Ok(Outcome::Seeded { effect, version, labels: out, seeds: segments });
    }
    let mut seeds: Vec<u8> = Vec::new();
    for &l in &labels {
        if l != 0 && !seeds.contains(&l) {
            seeds.push(l);
            if seeds.len() > 1 {
                break;
            }
        }
    }
    if seeds.len() < 2 {
        anyhow::bail!("paint seeds in at least two visible segments: the structure, and what surrounds it");
    }
    let all = labels_present(&labels);
    let out = match effect {
        Effect::Watershed => mm_segment::watershed(vol, &labels, s.watershed_mm as f64, None, s.seed_margin_mm as f64),
        _ => mm_segment::grow_from_seeds(vol, &labels, None, s.seed_margin_mm as f64, s.seed_locality as f64),
    };
    Ok(Outcome::Seeded { effect, version, labels: out, seeds: all })
}

/// The labels other than 0 in a label map, in increasing order: a table of the labels seen,
/// filled in parallel, rather than every voxel's label collected and sorted.
fn labels_present(labels: &[u8]) -> Vec<u8> {
    let seen = labels
        .par_chunks(1 << 16)
        .fold(
            || [false; 256],
            |mut seen, chunk| {
                for &l in chunk {
                    seen[l as usize] = true;
                }
                seen
            },
        )
        .reduce(|| [false; 256], |mut a, b| {
            a.iter_mut().zip(b).for_each(|(x, y)| *x |= y);
            a
        });
    (1..=255u8).filter(|&l| seen[l as usize]).collect()
}

/// Closed surface meshes of segments (name, label) of the label map `labels` (on a `dims`
/// grid), in LPS millimetres as 3D Slicer exports models. Each segment's mask is made in its
/// turn.
pub fn meshes(labels: &[u8], dims: [usize; 3], segments: &[(String, u8)], geometry: &Geometry, smoothing_mm: f64, progress: &dyn Fn(f32) -> bool) -> anyhow::Result<Vec<(String, Mesh)>> {
    let n = segments.len().max(1);
    let mut out = Vec::new();
    for (k, (name, label)) in segments.iter().enumerate() {
        if !progress(k as f32 / n as f32) {
            anyhow::bail!("cancelled");
        }
        let mask = Mask::from_labels(dims, labels, *label);
        if mask.count() == 0 {
            continue;
        }
        let params = MeshParams { smoothing_mm, cell_mm: None, lps: true };
        out.push((name.clone(), mask_to_mesh(&mask, geometry, &params, None)?));
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
        let mut s = stroke([40, 5, 5], [1.0; 3], DVec3::new(2.0, 2.0, 2.0), DVec3::new(37.0, 2.0, 2.0), 1.0, Some(2), false);
        s.sort_unstable();
        s.dedup();
        let row: Vec<usize> = s.iter().filter(|&&i| i / 40 % 5 == 2).map(|&i| i % 40).collect();
        assert_eq!(row, (1..=38).collect::<Vec<_>>());
    }

    #[test]
    fn labels_present_as_sorted() {
        for (k, labels) in [vec![], vec![0, 0], vec![7, 0, 3, 7, 255, 1, 3], (0..300_000u32).map(|i| (i.wrapping_mul(2_654_435_761) >> 27) as u8).collect()].into_iter().enumerate() {
            let mut sorted: Vec<u8> = labels.iter().copied().filter(|&l| l != 0).collect();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(labels_present(&labels), sorted, "case {k}");
        }
    }

    /// The voxels of a stroke as the dabs give them, every dab from `a` to `b` included.
    fn all_dabs(dims: [usize; 3], spacing: [f64; 3], a: DVec3, b: DVec3, r: f64, disc_axis: Option<usize>) -> Vec<usize> {
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

    #[test]
    fn strokes_cover_every_dab() {
        // Strokes of any length, size and shape, in steps as the pointer moves: one dab where
        // the pointer stays, and each step's start left out (the step before ended there)
        // still cover what every dab of every step covers.
        let (dims, spacing) = ([30, 24, 18], [0.7, 0.45, 1.3]);
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        for case in 0..60 {
            let r = [0.0, 0.3, 1.0, 2.5, 4.0][case % 5];
            let disc = [None, Some(0), Some(1), Some(2)][case % 4];
            let mut at = DVec3::new(next() * 29.0, next() * 23.0, next() * 17.0);
            let (mut covered, mut wanted) = (Vec::new(), Vec::new());
            covered.extend(stroke(dims, spacing, at, at, r, disc, false));
            wanted.extend(all_dabs(dims, spacing, at, at, r, disc));
            for step in 0..6 {
                // Some steps do not move.
                let to = if step % 3 == 2 { at } else { (at + DVec3::new(next() - 0.5, next() - 0.5, next() - 0.5) * 8.0).clamp(DVec3::ZERO, DVec3::new(29.0, 23.0, 17.0)) };
                covered.extend(stroke(dims, spacing, at, to, r, disc, true));
                wanted.extend(all_dabs(dims, spacing, at, to, r, disc));
                at = to;
            }
            for v in [&mut covered, &mut wanted] {
                v.sort_unstable();
                v.dedup();
            }
            assert_eq!(covered, wanted, "case {case}");
        }
        // A single dab where the pointer stays at the start of a stroke.
        let p = DVec3::new(10.2, 7.7, 3.1);
        assert_eq!({ let mut v = stroke(dims, spacing, p, p, 2.0, None, false); v.sort_unstable(); v }, all_dabs(dims, spacing, p, p, 2.0, None));
        assert!(stroke(dims, spacing, p, p, 2.0, None, true).is_empty());
    }

    #[test]
    fn masking_a_volume() {
        let g = Geometry::from_axes([4, 1, 1, 1], [1.0; 3], [DVec3::X, DVec3::Y, DVec3::Z], DVec3::ZERO);
        let vol = LoadedVolume::from_values("t", [4, 1, 1], vec![10.0, 20.0, 30.0, 40.0], g);
        let seg = Mask { dims: [4, 1, 1], data: vec![0, 1, 1, 0] };
        assert_eq!(mask_volume(&vol, &seg, MaskFill::Inside, -1.0, 0.0, 0.0).unwrap(), vec![10.0, -1.0, -1.0, 40.0]);
        assert_eq!(mask_volume(&vol, &seg, MaskFill::Outside, 0.0, -5.0, 0.0).unwrap(), vec![-5.0, 20.0, 30.0, -5.0]);
        assert_eq!(mask_volume(&vol, &seg, MaskFill::Both, 1.0, 0.0, 0.0).unwrap(), vec![0.0, 1.0, 1.0, 0.0]);
        // A soft edge blends: between the two values at the boundary.
        let soft = mask_volume(&vol, &seg, MaskFill::Both, 1.0, 0.0, 1.0).unwrap();
        assert!(soft[0] > 0.0 && soft[0] < 0.5 && soft[1] > 0.5 && soft[1] < 1.0, "{soft:?}");
    }

    #[test]
    fn refined_edges_follow_the_object() {
        // A ball of radius 6 (value 100) in noise (± 30): thresholding at 50 gives ragged
        // edges; refining keeps the ball (Dice against the true ball stays high).
        let n = 24;
        let g = Geometry::from_axes([n, n, n, 1], [1.0; 3], [DVec3::X, DVec3::Y, DVec3::Z], DVec3::ZERO);
        let c = (n as f64 - 1.0) * 0.5;
        let truth: Vec<u8> = (0..n * n * n).map(|i| (DVec3::new((i % n) as f64 - c, ((i / n) % n) as f64 - c, (i / n / n) as f64 - c).length() <= 6.0) as u8).collect();
        let values: Vec<f32> = (0..n * n * n).map(|i| truth[i] as f32 * 100.0 + ((i.wrapping_mul(2654435761) >> 7) % 61) as f32 - 30.0).collect();
        let vol = LoadedVolume::from_values("t", [n, n, n], values, g);
        let region = ops::threshold(&vol, 50.0, 1e9);
        let dice = |m: &Mask| {
            let both = m.data.iter().zip(&truth).filter(|(a, b)| **a != 0 && **b != 0).count() as f64;
            2.0 * both / (m.count() as f64 + truth.iter().filter(|&&t| t != 0).count() as f64)
        };
        for how in [Refine::GrowCut, Refine::Watershed] {
            let r = refine_edges(&vol, &region, how, 1.5);
            assert!(dice(&r) > 0.9, "{how:?}: Dice {:.3}", dice(&r));
        }
        // Too small to shrink: unchanged.
        let tiny = Mask { dims: [n; 3], data: (0..n * n * n).map(|i| (i == 300) as u8).collect() };
        assert_eq!(refine_edges(&vol, &tiny, Refine::GrowCut, 2.0), tiny);
    }

    #[test]
    fn otsu_splits_two_classes() {
        let mut bins = vec![0u32; 100];
        bins[10..20].fill(50);
        bins[70..80].fill(50);
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
        let Ok(Outcome::Split { slots, islands, .. }) = run(Effect::Islands, &s, &vol, Op { segment: both.clone(), ..empty() }) else { panic!() };
        let in_slot = |k: u8| slots.iter().filter(|&&s| s == k).count();
        assert_eq!((in_slot(1), islands, in_slot(2), in_slot(0)), (64, 1, 8, 1000 - 72));
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
