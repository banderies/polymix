//! Threshold grow and Select part: seeds are clicked in the views, then every ⌘ / Ctrl + right
//! drag in a slice view, from anywhere, moves the reach out (right) or back (left) from where
//! the last one left it. The region is what lies within reach of the nearest seed along the
//! region, nearest first (as Select's drag grows a selection in the Mesh tab).
//!
//! - Threshold grow: the region is the connected voxels in the threshold range; the views show
//!   the segment as Apply would leave it, and Apply keeps it as one undo step.
//! - Select part: the region is the seeds' segment; the part shows in its own colour, to
//!   erase, keep, move to another segment or measure.

use crate::segmentation::{Masking, Segmentation};
use egui::Color32;
use glam::DVec3;
use mm_segment::DragGrow;
use mm_volume::load::LoadedVolume;
use polymix_imaging::{DragPhase, ToolDrag};
use rayon::prelude::*;

/// Colour of a selected part in the slice views.
pub const PART_COLOUR: Color32 = Color32::from_rgb(255, 214, 10);

/// The voxel index of voxel coordinates `v` (rounded, kept inside the grid).
pub fn voxel_index(dims: [usize; 3], v: DVec3) -> usize {
    let c = [0, 1, 2].map(|a| v[a].round().clamp(0.0, dims[a] as f64 - 1.0) as usize);
    c[0] + dims[0] * (c[1] + dims[1] * c[2])
}

/// What decides a region: its seeds (voxel indices) and kind, and the labels it shows over.
#[derive(Clone, PartialEq)]
struct Key {
    seeds: Vec<usize>,
    kind: Kind,
    labels_version: u64,
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    /// Threshold grow into segment `label`, over values in `range`, as `masking` allows.
    Grow { label: u8, range: (f32, f32), masking: Masking },
    /// Select part of segment `label`, shown in label `mark` (no segment's).
    Part { label: u8, mark: u8 },
}

/// A region growing from clicked seeds as far as the drags have set its reach.
pub struct Reach {
    key: Key,
    search: DragGrow,
    /// How far the region reaches from the seeds (mm, along it).
    pub mm: f64,
    /// The reach when the drag going on started.
    drag_from: Option<f64>,
    /// The region: the first so many voxels the search settled.
    shown: usize,
    /// The labels with the region in (Threshold grow: as Apply would leave them; Select part:
    /// the part in its mark): what the slice views show.
    pub overlay: Vec<u8>,
    /// Tells this region's overlay from others' ([`Reach::version`] changes with it).
    pub id: u64,
    pub version: u64,
    /// Seeds clicked that the region leaves out: outside the threshold range (Threshold grow),
    /// or not in the first seed's segment (Select part).
    pub ignored: usize,
}

impl Reach {
    /// Keep `slot` the Threshold grow region of `seeds` (voxel coordinates) into segment
    /// `label` over values in `range`: made again when any of that, the masking or the labels
    /// changed, keeping its reach. No seeds in range, or no segment: no region.
    #[allow(clippy::too_many_arguments)]
    pub fn sync_grow(slot: &mut Option<Self>, vol: &LoadedVolume, seg: &Segmentation, seeds: &[DVec3], label: Option<u8>, range: (f32, f32), masking: &Masking, ids: &mut u64) {
        let idx: Vec<usize> = seeds.iter().map(|&v| voxel_index(vol.dims, v)).collect();
        let kept: Vec<usize> = idx.iter().copied().filter(|&i| vol.values[i] >= range.0 && vol.values[i] <= range.1).collect();
        let kind = label.map(|label| Kind::Grow { label, range, masking: *masking });
        Self::sync(slot, vol, seg, kept, idx.len(), kind, ids);
    }

    /// Keep `slot` the Select part region of `seeds`: the part of the first seed's segment
    /// within reach of the seeds in it.
    pub fn sync_part(slot: &mut Option<Self>, vol: &LoadedVolume, seg: &Segmentation, seeds: &[DVec3], ids: &mut u64) {
        let idx: Vec<usize> = seeds.iter().map(|&v| voxel_index(vol.dims, v)).collect();
        let label = idx.first().map(|&i| seg.labels[i]).filter(|&l| l != 0 && seg.index_of(l).is_some());
        let kept: Vec<usize> = idx.iter().copied().filter(|&i| Some(seg.labels[i]) == label).collect();
        // The mark stays while it is still free, so the key does not change with every frame.
        let mark = slot.as_ref().and_then(|r| match r.key.kind {
            Kind::Part { mark, .. } => Some(mark),
            _ => None,
        });
        let mark = mark.filter(|&m| seg.index_of(m).is_none()).or_else(|| (1..=255u8).find(|&l| seg.index_of(l).is_none()));
        let kind = label.zip(mark).map(|(label, mark)| Kind::Part { label, mark });
        Self::sync(slot, vol, seg, kept, idx.len(), kind, ids);
    }

    fn sync(slot: &mut Option<Self>, vol: &LoadedVolume, seg: &Segmentation, seeds: Vec<usize>, clicked: usize, kind: Option<Kind>, ids: &mut u64) {
        let Some(kind) = kind.filter(|_| !seeds.is_empty()) else {
            *slot = None;
            return;
        };
        let key = Key { seeds, kind, labels_version: seg.version };
        if slot.as_ref().is_none_or(|r| r.key != key) {
            let (mm, drag_from) = slot.as_ref().map_or((0.0, None), |r| (r.mm, r.drag_from));
            *ids += 1;
            *slot = Some(Self { search: DragGrow::with_seeds(vol.dims, vol.geometry.spacing, &key.seeds), key, mm, drag_from, shown: 0, overlay: seg.labels.clone(), id: *ids, version: 0, ignored: 0 });
        }
        let r = slot.as_mut().unwrap();
        r.ignored = clicked - r.key.seeds.len();
        r.follow(vol, seg);
    }

    /// Bring the region (and the overlay) to the reach.
    fn follow(&mut self, vol: &LoadedVolume, seg: &Segmentation) {
        let k = if self.mm > 0.0 {
            match self.key.kind {
                Kind::Grow { range: (lo, hi), .. } => self.search.reach(self.mm, |i| vol.values[i] >= lo && vol.values[i] <= hi),
                Kind::Part { label, .. } => self.search.reach(self.mm, |i| seg.labels[i] == label),
            }
        } else {
            0
        };
        if k == self.shown {
            return;
        }
        let (lo, hi) = (k.min(self.shown), k.max(self.shown));
        let changed: Vec<usize> = self.search.settled()[lo..hi].iter().map(|s| s.1).collect();
        if k > self.shown {
            match self.key.kind {
                Kind::Grow { label, masking, .. } => seg.preview_paint(vol, &changed, label, &masking, &mut self.overlay),
                Kind::Part { mark, .. } => changed.iter().for_each(|&i| self.overlay[i] = mark),
            }
        } else {
            changed.iter().for_each(|&i| self.overlay[i] = seg.labels[i]);
        }
        self.shown = k;
        self.version += 1;
    }

    /// Follow a ⌘ / Ctrl + right drag: `mm_per_point` of reach per point right of its press,
    /// from the reach it started at (never below nothing).
    pub fn drag(&mut self, d: &ToolDrag, mm_per_point: f64) {
        if d.phase == DragPhase::Start || self.drag_from.is_none() {
            self.drag_from = Some(self.mm);
        }
        self.mm = (self.drag_from.unwrap_or(0.0) + d.total.x as f64 * mm_per_point).max(0.0);
        if d.phase == DragPhase::End {
            self.drag_from = None;
        }
    }

    pub fn dragging(&self) -> bool {
        self.drag_from.is_some()
    }

    /// Voxels in the region.
    pub fn count(&self) -> usize {
        self.shown
    }

    /// The region's voxels (indices).
    pub fn voxel_list(&self) -> Vec<usize> {
        self.search.settled()[..self.shown].iter().map(|s| s.1).collect()
    }

    /// The segment grown, or the part is of.
    pub fn label(&self) -> u8 {
        match self.key.kind {
            Kind::Grow { label, .. } | Kind::Part { label, .. } => label,
        }
    }

    /// Select part: the label the part shows in.
    pub fn mark(&self) -> Option<u8> {
        match self.key.kind {
            Kind::Part { mark, .. } => Some(mark),
            Kind::Grow { .. } => None,
        }
    }

    pub fn stats(&self, vol: &LoadedVolume) -> RegionStats {
        let s = vol.geometry.spacing;
        let [nx, ny, _] = vol.dims;
        let region = &self.search.settled()[..self.shown];
        let (mut lo, mut hi) = ([usize::MAX; 3], [0usize; 3]);
        for &(_, i) in region {
            let c = [i % nx, (i / nx) % ny, i / (nx * ny)];
            for a in 0..3 {
                lo[a] = lo[a].min(c[a]);
                hi[a] = hi[a].max(c[a]);
            }
        }
        RegionStats {
            voxels: region.len(),
            volume_mm3: region.len() as f64 * s[0] * s[1] * s[2],
            reach_mm: region.last().map_or(0.0, |p| p.0 as f64),
            extent_mm: [0, 1, 2].map(|a| if region.is_empty() { 0.0 } else { (hi[a] - lo[a] + 1) as f64 * s[a] }),
        }
    }

    /// Threshold grow: keep the region in the segment (one undo step). Returns what was done.
    pub fn apply_grow(&self, vol: &LoadedVolume, seg: &mut Segmentation) -> String {
        let Kind::Grow { label, masking, .. } = self.key.kind else { return String::new() };
        seg.checkpoint();
        let n = seg.paint(vol, &self.voxel_list(), label, false, &masking);
        seg.stroke_finished();
        format!("Threshold grow: {n} voxels added")
    }
}

/// What a region measures.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RegionStats {
    pub voxels: usize,
    pub volume_mm3: f64,
    /// Path length along the region from the nearest seed to its farthest voxel.
    pub reach_mm: f64,
    /// Size of the region's bounding box along the voxel axes (mm).
    pub extent_mm: [f64; 3],
}

/// What to do with a selected part.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PartAction {
    Erase,
    /// Remove the rest of the segment.
    KeepOnly,
    /// Into this segment (label), or a new one (`None`).
    MoveTo(Option<u8>),
}

/// Do `action` to the selected part `part` (as one undo step). Returns what was done.
pub fn use_part(part: &Reach, vol: &LoadedVolume, seg: &mut Segmentation, action: PartAction) -> String {
    let (Some(mark), label) = (part.mark(), part.label()) else { return String::new() };
    let stats = part.stats(vol);
    let size = format!("{} voxels, {:.1} mm³", stats.voxels, stats.volume_mm3);
    let name = |seg: &Segmentation, label: u8| seg.index_of(label).map_or_else(String::new, |k| seg.segments[k].name.clone());
    seg.checkpoint();
    match action {
        PartAction::Erase => {
            seg.relabel(&part.voxel_list(), 0);
            format!("Erased the part ({size})")
        }
        PartAction::KeepOnly => {
            let rest: Vec<usize> = seg.labels.par_iter().zip(&part.overlay).enumerate().filter(|(_, (&l, &o))| l == label && o != mark).map(|(i, _)| i).collect();
            let n = seg.relabel(&rest, 0);
            format!("Kept only the part ({size}); removed {n} voxels")
        }
        PartAction::MoveTo(to) => {
            let to = match to {
                Some(l) => l,
                None => {
                    let base = name(seg, label);
                    match seg.push_segment(Some(format!("{base}_part"))) {
                        Some(k) => seg.segments[k].label,
                        None => return "All 255 labels are in use: no new segment".into(),
                    }
                }
            };
            seg.relabel(&part.voxel_list(), to);
            seg.selected = seg.index_of(to);
            format!("Moved the part ({size}) to {}", name(seg, to))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::{pos2, vec2};
    use mm_volume::geometry::Geometry;

    /// A 30³ volume of 1 mm voxels: a bright bar along x through the middle (values 100),
    /// everything else 0.
    fn bar() -> LoadedVolume {
        let dims = [30, 30, 30];
        let g = Geometry::from_axes([30, 30, 30, 1], [1.0; 3], [DVec3::X, DVec3::Y, DVec3::Z], DVec3::ZERO);
        let values: Vec<f32> = (0..27000).map(|i| if (i / 30) % 30 >= 13 && (i / 30) % 30 <= 16 && i / 900 >= 13 && i / 900 <= 16 { 100.0 } else { 0.0 }).collect();
        LoadedVolume::from_values("bar", dims, values, g)
    }

    fn drag(r: &mut Reach, phase: DragPhase, x: f32) {
        r.drag(&ToolDrag { phase, press: pos2(0.0, 0.0), total: vec2(x, 0.0) }, 0.1);
    }

    #[test]
    fn threshold_grow_reaches_from_the_seeds_drag_after_drag_and_applies_as_one_step() {
        let vol = bar();
        let mut seg = Segmentation::new(vol.dims);
        let k = seg.add_segment(None).unwrap();
        let label = seg.segments[k].label;
        let range = (50.0, 200.0);
        let masking = Masking::default();
        let mut ids = 0;
        let mut slot = None;
        // A seed outside the range alone: no region.
        Reach::sync_grow(&mut slot, &vol, &seg, &[DVec3::new(5.0, 2.0, 2.0)], Some(label), range, &masking, &mut ids);
        assert!(slot.is_none());
        let seeds = [DVec3::new(5.0, 2.0, 2.0), DVec3::new(5.0, 14.0, 14.0)];
        Reach::sync_grow(&mut slot, &vol, &seg, &seeds, Some(label), range, &masking, &mut ids);
        let r = slot.as_mut().unwrap();
        assert_eq!((r.ignored, r.count()), (1, 0));
        // 100 points right: 10 mm along the bar from x = 5.
        drag(r, DragPhase::Start, 0.0);
        drag(r, DragPhase::Move, 100.0);
        drag(r, DragPhase::End, 100.0);
        Reach::sync_grow(&mut slot, &vol, &seg, &seeds, Some(label), range, &masking, &mut ids);
        let first = slot.as_ref().unwrap().count();
        assert!(first > 100 && first < 16 * 16, "{first}");
        let shown = |slot: &Option<Reach>| slot.as_ref().unwrap().overlay.iter().filter(|&&l| l == label).count();
        assert_eq!(shown(&slot), first, "the preview shows the region");
        assert!(seg.labels.iter().all(|&l| l == 0), "nothing is kept before Apply");
        // A second drag, from anywhere, goes on from there: 50 more points right, then back.
        let r = slot.as_mut().unwrap();
        drag(r, DragPhase::Start, 0.0);
        drag(r, DragPhase::Move, 50.0);
        assert!((r.mm - 15.0).abs() < 1e-9);
        Reach::sync_grow(&mut slot, &vol, &seg, &seeds, Some(label), range, &masking, &mut ids);
        assert!(slot.as_ref().unwrap().count() > first);
        let r = slot.as_mut().unwrap();
        drag(r, DragPhase::End, -500.0);
        assert_eq!(r.mm, 0.0, "never below nothing");
        Reach::sync_grow(&mut slot, &vol, &seg, &seeds, Some(label), range, &masking, &mut ids);
        assert_eq!(shown(&slot), 0);
        // All of it, kept: one undo step takes it back.
        slot.as_mut().unwrap().mm = 1000.0;
        Reach::sync_grow(&mut slot, &vol, &seg, &seeds, Some(label), range, &masking, &mut ids);
        assert_eq!(slot.as_ref().unwrap().count(), 16 * 30);
        slot.as_ref().unwrap().apply_grow(&vol, &mut seg);
        assert_eq!(seg.labels.iter().filter(|&&l| l == label).count(), 16 * 30);
        assert!(seg.labels.iter().zip(&vol.values).all(|(&l, &v)| l != label || v == 100.0), "only voxels in range");
        assert!(seg.undo());
        assert!(seg.labels.iter().all(|&l| l != label));
    }

    #[test]
    fn select_part_and_its_actions() {
        let vol = bar();
        let mut seg = Segmentation::new(vol.dims);
        let k = seg.add_segment(Some("vessel".into())).unwrap();
        let label = seg.segments[k].label;
        let bar_voxels: Vec<usize> = (0..vol.values.len()).filter(|&i| vol.values[i] == 100.0).collect();
        seg.relabel(&bar_voxels, label);
        let mut ids = 0;
        let mut slot = None;
        Reach::sync_part(&mut slot, &vol, &seg, &[DVec3::new(2.0, 2.0, 2.0)], &mut ids);
        assert!(slot.is_none(), "not on a segment");
        let seeds = [DVec3::new(0.0, 14.0, 14.0)];
        Reach::sync_part(&mut slot, &vol, &seg, &seeds, &mut ids);
        // 9.5 mm from the end at x = 0: the bar up to x = 9 (its far corners there are about
        // 9.4 mm away along the bar), and nothing from x = 10 on.
        drag(slot.as_mut().unwrap(), DragPhase::End, 95.0);
        Reach::sync_part(&mut slot, &vol, &seg, &seeds, &mut ids);
        let part = slot.as_ref().unwrap();
        let mark = part.mark().unwrap();
        let stats = part.stats(&vol);
        let selected = part.voxel_list();
        assert!(bar_voxels.iter().filter(|&&i| i % 30 <= 7).all(|i| selected.contains(i)), "{stats:?}");
        assert!(selected.iter().all(|&i| i % 30 <= 9 && vol.values[i] == 100.0));
        assert_eq!(stats.extent_mm, [10.0, 4.0, 4.0]);
        assert!(part.overlay.iter().filter(|&&l| l == mark).count() == stats.voxels && seg.index_of(mark).is_none());
        let n = part.count();
        let mut moved = Segmentation::new(vol.dims);
        moved.labels = seg.labels.clone();
        moved.segments = seg.segments.clone();
        let message = use_part(part, &vol, &mut moved, PartAction::MoveTo(None));
        let new = moved.segments.iter().find(|s| s.name == "vessel_part").expect(&message).label;
        assert_eq!(moved.labels.iter().filter(|&&l| l == new).count(), n);
        assert_eq!(moved.labels.iter().filter(|&&l| l == label).count(), 16 * 30 - n);
        let mut kept = Segmentation::new(vol.dims);
        kept.labels = seg.labels.clone();
        kept.segments = seg.segments.clone();
        use_part(part, &vol, &mut kept, PartAction::KeepOnly);
        assert_eq!(kept.labels.iter().filter(|&&l| l == label).count(), n);
        use_part(part, &vol, &mut seg, PartAction::Erase);
        assert_eq!(seg.labels.iter().filter(|&&l| l == label).count(), 16 * 30 - n);
        // The seed's voxel was erased with the part: no segment there now, so no region.
        Reach::sync_part(&mut slot, &vol, &seg, &seeds, &mut ids);
        assert!(slot.is_none());
        assert!(seg.undo());
        assert_eq!(seg.labels.iter().filter(|&&l| l == label).count(), 16 * 30);
    }
}
