//! Threshold grow and Select part: a ⌘ / Ctrl + right drag in a slice view grows a region
//! from the voxel pressed, nearest first along the region, as far as the drag reaches to the
//! right (as Select's drag grows a selection in the Mesh tab).
//!
//! - Threshold grow adds the connected voxels in the threshold range to the segment as the
//!   drag goes out, and takes them back as it returns. The whole drag is one undo step.
//! - Select part marks the part of a segment within reach, along the segment, to erase,
//!   keep, move to another segment or measure.

use crate::segmentation::{Masking, Segmentation};
use egui::Color32;
use mm_segment::DragGrow;
use mm_volume::load::LoadedVolume;
use rayon::prelude::*;

/// Colour of a selected part in the slice views.
pub const PART_COLOUR: Color32 = Color32::from_rgb(255, 214, 10);

/// The voxel index of voxel coordinates `v` (rounded, kept inside the grid).
pub fn voxel_index(dims: [usize; 3], v: glam::DVec3) -> usize {
    let c = [0, 1, 2].map(|a| v[a].round().clamp(0.0, dims[a] as f64 - 1.0) as usize);
    c[0] + dims[0] * (c[1] + dims[1] * c[2])
}

/// Millimetres of reach for a drag of `total_x` points (none to the left of the press).
fn reach_mm(total_x: f32, mm_per_point: f64) -> f64 {
    (total_x as f64 * mm_per_point).max(0.0)
}

/// Threshold grow's drag in progress.
pub struct Grow {
    search: DragGrow,
    mm_per_point: f64,
    label: u8,
    range: (f32, f32),
    /// The label each voxel added so far had before: the first so many the search settled.
    before: Vec<u8>,
    /// How far the drag reaches now (mm).
    pub reach: f64,
}

impl Grow {
    /// Start at voxel `seed`, which must be in `range`, growing segment `label`; makes the
    /// drag's undo step. `mm_per_point`: reach per point of drag in the view pressed.
    pub fn start(vol: &LoadedVolume, seg: &mut Segmentation, seed: usize, label: u8, range: (f32, f32), mm_per_point: f64) -> Result<Self, String> {
        let v = vol.values[seed];
        if !(v >= range.0 && v <= range.1) {
            return Err(format!("Press on a voxel in the threshold range ({:.4} – {:.4}); this one is {v:.4}", range.0, range.1));
        }
        seg.checkpoint();
        Ok(Self { search: DragGrow::new(vol.dims, vol.geometry.spacing, seed), mm_per_point, label, range, before: Vec::new(), reach: 0.0 })
    }

    /// Follow the drag (`total_x` points right of the press): add the voxels now within reach,
    /// put back those no longer within it. Returns the voxels the drag holds.
    pub fn drag(&mut self, vol: &LoadedVolume, seg: &mut Segmentation, total_x: f32, masking: &Masking) -> usize {
        self.reach = reach_mm(total_x, self.mm_per_point);
        let (lo, hi) = self.range;
        let values = &vol.values;
        let k = if self.reach > 0.0 { self.search.reach(self.reach, |i| values[i] >= lo && values[i] <= hi) } else { 0 };
        let shown = self.before.len();
        if k > shown {
            let add: Vec<usize> = self.search.settled()[shown..k].iter().map(|s| s.1).collect();
            self.before.extend(add.iter().map(|&i| seg.labels[i]));
            seg.paint(vol, &add, self.label, false, masking);
        } else if k < shown {
            let back: Vec<(usize, u8)> = self.search.settled()[k..shown].iter().zip(&self.before[k..]).map(|(s, &b)| (s.1, b)).collect();
            self.before.truncate(k);
            seg.restore_voxels(&back);
        }
        k
    }
}

/// What a selected part's measures are.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PartStats {
    pub voxels: usize,
    pub volume_mm3: f64,
    /// Path length along the segment from the voxel pressed to the part's farthest voxel.
    pub reach_mm: f64,
    /// Size of the part's bounding box along the voxel axes (mm).
    pub extent_mm: [f64; 3],
}

/// Select part's part: being dragged out, then kept until it is used or dropped.
pub struct Part {
    search: DragGrow,
    mm_per_point: f64,
    /// The segment the part is of.
    pub label: u8,
    /// A label no segment uses: the part shows in it, in [`PART_COLOUR`].
    pub mark: u8,
    /// The label map with the part's voxels in `mark`: what the slice views show meanwhile.
    pub overlay: Vec<u8>,
    /// Tells this part's overlay from others' ([`Part::version`] changes as it does).
    pub id: u64,
    pub version: u64,
    /// The segmentation's version the part was taken from: an edit since makes it stale.
    pub seg_version: u64,
    /// Voxels in the part: the first so many the search settled.
    pub voxels: usize,
    /// The drag is over.
    pub done: bool,
}

impl Part {
    /// Start at voxel `seed`, which must be in a segment.
    pub fn start(vol: &LoadedVolume, seg: &Segmentation, seed: usize, mm_per_point: f64, id: u64) -> Result<Self, String> {
        let label = seg.labels[seed];
        if label == 0 || seg.index_of(label).is_none() {
            return Err("Press on a voxel of a segment".into());
        }
        let mark = (1..=255u8).find(|&l| seg.index_of(l).is_none()).ok_or("All 255 labels are in use: none is left to show a part in")?;
        Ok(Self {
            search: DragGrow::new(vol.dims, vol.geometry.spacing, seed),
            mm_per_point,
            label,
            mark,
            overlay: seg.labels.clone(),
            id,
            version: 0,
            seg_version: seg.version,
            voxels: 0,
            done: false,
        })
    }

    /// Follow the drag (`total_x` points right of the press): the part is the segment's
    /// voxels within reach along it.
    pub fn drag(&mut self, seg: &Segmentation, total_x: f32) {
        let reach = reach_mm(total_x, self.mm_per_point);
        let (label, labels) = (self.label, &seg.labels);
        let k = if reach > 0.0 { self.search.reach(reach, |i| labels[i] == label) } else { 0 };
        if k == self.voxels {
            return;
        }
        let (lo, hi, to) = if k > self.voxels { (self.voxels, k, self.mark) } else { (k, self.voxels, label) };
        for &(_, i) in &self.search.settled()[lo..hi] {
            self.overlay[i] = to;
        }
        self.voxels = k;
        self.version += 1;
    }

    /// The part's voxels (indices).
    pub fn voxel_list(&self) -> Vec<usize> {
        self.search.settled()[..self.voxels].iter().map(|s| s.1).collect()
    }

    pub fn stats(&self, vol: &LoadedVolume) -> PartStats {
        let s = vol.geometry.spacing;
        let [nx, ny, _] = vol.dims;
        let part = &self.search.settled()[..self.voxels];
        let (mut lo, mut hi) = ([usize::MAX; 3], [0usize; 3]);
        for &(_, i) in part {
            let c = [i % nx, (i / nx) % ny, i / (nx * ny)];
            for a in 0..3 {
                lo[a] = lo[a].min(c[a]);
                hi[a] = hi[a].max(c[a]);
            }
        }
        PartStats {
            voxels: part.len(),
            volume_mm3: part.len() as f64 * s[0] * s[1] * s[2],
            reach_mm: part.last().map_or(0.0, |p| p.0 as f64),
            extent_mm: [0, 1, 2].map(|a| if part.is_empty() { 0.0 } else { (hi[a] - lo[a] + 1) as f64 * s[a] }),
        }
    }
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

/// Do `action` to `part` (as one undo step). Returns what was done, for the status line.
pub fn use_part(part: &Part, vol: &LoadedVolume, seg: &mut Segmentation, action: PartAction) -> String {
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
            let (label, mark) = (part.label, part.mark);
            let rest: Vec<usize> = seg.labels.par_iter().zip(&part.overlay).enumerate().filter(|(_, (&l, &o))| l == label && o != mark).map(|(i, _)| i).collect();
            let n = seg.relabel(&rest, 0);
            format!("Kept only the part ({size}); removed {n} voxels")
        }
        PartAction::MoveTo(to) => {
            let to = match to {
                Some(l) => l,
                None => {
                    let base = name(seg, part.label);
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
    use crate::segmentation::Segmentation;
    use glam::DVec3;
    use mm_volume::geometry::Geometry;

    /// A 30³ volume of 1 mm voxels: a bright bar along x through the middle (values 100),
    /// everything else 0.
    fn bar() -> LoadedVolume {
        let dims = [30, 30, 30];
        let g = Geometry::from_axes([30, 30, 30, 1], [1.0; 3], [DVec3::X, DVec3::Y, DVec3::Z], DVec3::ZERO);
        let values: Vec<f32> = (0..27000).map(|i| if (i / 30) % 30 >= 13 && (i / 30) % 30 <= 16 && i / 900 >= 13 && i / 900 <= 16 { 100.0 } else { 0.0 }).collect();
        LoadedVolume::from_values("bar", dims, values, g)
    }

    #[test]
    fn threshold_grow_follows_the_drag_out_and_back_as_one_undo_step() {
        let vol = bar();
        let mut seg = Segmentation::new(vol.dims);
        let k = seg.add_segment(None).unwrap();
        let label = seg.segments[k].label;
        let seed = voxel_index(vol.dims, DVec3::new(5.0, 14.0, 14.0));
        // Out of range: nothing starts.
        let version = seg.version;
        assert!(Grow::start(&vol, &mut seg, voxel_index(vol.dims, DVec3::new(5.0, 2.0, 2.0)), label, (50.0, 200.0), 0.1).is_err());
        assert_eq!(seg.version, version);
        let mut g = Grow::start(&vol, &mut seg, seed, label, (50.0, 200.0), 0.1).unwrap();
        let masking = Masking::default();
        let count = |seg: &Segmentation| seg.labels.iter().filter(|&&l| l == label).count();
        // 100 points right = 10 mm along the bar (4 × 4 voxels across), both ways from x = 5.
        let n = g.drag(&vol, &mut seg, 100.0, &masking);
        assert_eq!(count(&seg), n);
        assert!(n > 100 && n < 16 * 18, "{n}");
        assert!(seg.labels.iter().enumerate().all(|(i, &l)| l != label || vol.values[i] == 100.0), "only voxels in range");
        let far = g.drag(&vol, &mut seg, 400.0, &masking);
        assert_eq!(far, 16 * 30, "the whole bar, and nothing past its ends");
        // Back to the left of the press: everything taken back.
        g.drag(&vol, &mut seg, 30.0, &masking);
        assert!(count(&seg) < n);
        g.drag(&vol, &mut seg, -20.0, &masking);
        assert_eq!(count(&seg), 0);
        g.drag(&vol, &mut seg, 400.0, &masking);
        seg.stroke_finished();
        assert!(seg.undo());
        assert_eq!(count(&seg), 0, "one undo step takes back the whole drag");
    }

    #[test]
    fn select_part_and_its_actions() {
        let vol = bar();
        let mut seg = Segmentation::new(vol.dims);
        let k = seg.add_segment(Some("vessel".into())).unwrap();
        let label = seg.segments[k].label;
        let bar_voxels: Vec<usize> = (0..vol.values.len()).filter(|&i| vol.values[i] == 100.0).collect();
        seg.relabel(&bar_voxels, label);
        let at = |x: f64| voxel_index(vol.dims, DVec3::new(x, 14.0, 14.0));
        assert!(Part::start(&vol, &seg, voxel_index(vol.dims, DVec3::new(2.0, 2.0, 2.0)), 0.1, 1).is_err(), "not on a segment");
        let mut part = Part::start(&vol, &seg, at(0.0), 0.1, 1).unwrap();
        // 9.5 mm from the end at x = 0: the bar up to x = 9 (its far corners there are about
        // 9.4 mm away along the bar), and nothing from x = 10 on.
        part.drag(&seg, 95.0);
        let stats = part.stats(&vol);
        let x_of = |i: usize| i % 30;
        let selected: Vec<usize> = part.voxel_list();
        assert!(bar_voxels.iter().filter(|&&i| x_of(i) <= 7).all(|i| selected.contains(i)), "{stats:?}");
        assert!(selected.iter().all(|&i| x_of(i) <= 9 && vol.values[i] == 100.0));
        assert_eq!(stats.extent_mm, [10.0, 4.0, 4.0]);
        assert!(part.overlay.iter().filter(|&&l| l == part.mark).count() == stats.voxels && seg.index_of(part.mark).is_none());
        // Back a little: the overlay follows.
        part.drag(&seg, 45.0);
        assert_eq!(part.overlay.iter().filter(|&&l| l == part.mark).count(), part.voxels);
        part.drag(&seg, 95.0);
        let mut moved = Segmentation::new(vol.dims);
        moved.labels = seg.labels.clone();
        moved.segments = seg.segments.clone();
        let message = use_part(&part, &vol, &mut moved, PartAction::MoveTo(None));
        let n = part.voxels;
        let new = moved.segments.iter().find(|s| s.name == "vessel_part").expect(&message).label;
        assert_eq!(moved.labels.iter().filter(|&&l| l == new).count(), n);
        assert_eq!(moved.labels.iter().filter(|&&l| l == label).count(), 16 * 30 - n);
        let mut kept = Segmentation::new(vol.dims);
        kept.labels = seg.labels.clone();
        kept.segments = seg.segments.clone();
        use_part(&part, &vol, &mut kept, PartAction::KeepOnly);
        assert_eq!(kept.labels.iter().filter(|&&l| l == label).count(), n);
        use_part(&part, &vol, &mut seg, PartAction::Erase);
        assert_eq!(seg.labels.iter().filter(|&&l| l == label).count(), 16 * 30 - n);
        assert!(seg.undo());
        assert_eq!(seg.labels.iter().filter(|&&l| l == label).count(), 16 * 30);
    }
}
