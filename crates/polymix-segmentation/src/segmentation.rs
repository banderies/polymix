//! The segmentation being edited: a label map over the volume's voxel grid (one byte per
//! voxel: 0 = no segment, otherwise a segment's label), its segments, the masking rules that
//! limit edits (as in 3D Slicer's Segment Editor), and undo.

use egui::Color32;
use mm_segment::Mask;
use mm_volume::load::LoadedVolume;
use rayon::prelude::*;

/// Colours for new segments, in the spirit of 3D Slicer's anatomy colours.
const PALETTE: [[u8; 3]; 12] = [
    [128, 174, 128],
    [241, 214, 145],
    [177, 122, 101],
    [111, 184, 210],
    [216, 101, 79],
    [221, 130, 101],
    [144, 238, 144],
    [192, 104, 88],
    [220, 245, 20],
    [78, 63, 0],
    [255, 250, 220],
    [230, 220, 70],
];

/// Undo steps kept (as in Slicer).
const UNDO_STEPS: usize = 10;

#[derive(Clone, Debug, PartialEq)]
pub struct Segment {
    pub name: String,
    pub colour: Color32,
    pub visible: bool,
    /// Its value in the label map (1–255).
    pub label: u8,
    pub status: Status,
}

/// How far a segment's work has got (as 3D Slicer flags it).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Status {
    #[default]
    NotStarted,
    InProgress,
    Completed,
    Flagged,
}

impl Status {
    pub const ALL: [Status; 4] = [Status::NotStarted, Status::InProgress, Status::Completed, Status::Flagged];

    pub fn label(self) -> &'static str {
        match self {
            Status::NotStarted => "Not started",
            Status::InProgress => "In progress",
            Status::Completed => "Completed",
            Status::Flagged => "Flagged",
        }
    }

    /// As Slicer's `Segmentation.Status` tag has it.
    pub fn key(self) -> &'static str {
        match self {
            Status::NotStarted => "notstarted",
            Status::InProgress => "inprogress",
            Status::Completed => "completed",
            Status::Flagged => "flagged",
        }
    }

    pub fn colour(self) -> Color32 {
        match self {
            Status::NotStarted => Color32::from_gray(120),
            Status::InProgress => Color32::from_rgb(240, 180, 60),
            Status::Completed => Color32::from_rgb(110, 200, 110),
            Status::Flagged => Color32::from_rgb(235, 90, 80),
        }
    }

    /// The next one round (clicking the flag).
    pub fn next(self) -> Status {
        Status::ALL[(Status::ALL.iter().position(|s| *s == self).unwrap_or(0) + 1) % 4]
    }
}

/// Where edits may change voxels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EditableArea {
    Everywhere,
    InsideAll,
    InsideVisible,
    OutsideAll,
    OutsideVisible,
    /// Inside the segment with this label.
    Inside(u8),
    Outside(u8),
}

/// What an edit may do to voxels of other segments.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Overwrite {
    All,
    /// Only voxels of visible segments.
    Visible,
    /// None: other segments keep their voxels.
    None,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Masking {
    pub area: EditableArea,
    /// Only voxels of the volume with values in this range.
    pub intensity: Option<(f32, f32)>,
    pub overwrite: Overwrite,
}

impl Default for Masking {
    fn default() -> Self {
        Self { area: EditableArea::Everywhere, intensity: None, overwrite: Overwrite::All }
    }
}

/// How a mask changes a segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// The segment becomes the mask (where editable).
    Set,
    Add,
    Remove,
}

/// The masking rules for one edit.
struct Rules {
    masking: Masking,
    visible: [bool; 256],
}

impl Rules {
    fn in_area(&self, c: u8) -> bool {
        let v = self.visible[c as usize];
        match self.masking.area {
            EditableArea::Everywhere => true,
            EditableArea::InsideAll => c != 0,
            EditableArea::InsideVisible => c != 0 && v,
            EditableArea::OutsideAll => c == 0,
            EditableArea::OutsideVisible => c == 0 || !v,
            EditableArea::Inside(x) => c == x,
            EditableArea::Outside(x) => c != x,
        }
    }

    fn in_range(&self, value: f32) -> bool {
        self.masking.intensity.is_none_or(|(lo, hi)| value >= lo && value <= hi)
    }

    /// The new label of a voxel now labelled `c` (value `value`) that the edit wants in
    /// segment `label` (`want`) or out of it. `split`: an Erase stroke, which (as in Slicer)
    /// gives what it erases to the segment chosen as the editable area.
    fn result(&self, c: u8, value: f32, label: u8, want: bool, split: bool) -> u8 {
        if want == (c == label) || !self.in_range(value) {
            return c;
        }
        if let (false, true, EditableArea::Inside(x)) = (want, split, self.masking.area) {
            return x;
        }
        if !self.in_area(c) {
            return c;
        }
        if !want {
            return 0;
        }
        let may_take = c == 0
            || match self.masking.overwrite {
                Overwrite::All => true,
                Overwrite::Visible => self.visible[c as usize],
                Overwrite::None => false,
            };
        if may_take { label } else { c }
    }
}

/// A past state, with the label map run-length encoded.
struct Snapshot {
    runs: Vec<(u8, u32)>,
    segments: Vec<Segment>,
    selected: Option<usize>,
}

pub struct Segmentation {
    pub dims: [usize; 3],
    pub labels: Vec<u8>,
    pub segments: Vec<Segment>,
    pub selected: Option<usize>,
    /// Changes with every edit (the slice views re-read the labels).
    pub version: u64,
    /// Changes when an edit is finished, not with every brush dab (the 3D view re-uploads).
    pub version_3d: u64,
    /// Edited since it was created or saved.
    pub modified: bool,
    undo: Vec<Snapshot>,
    redo: Vec<Snapshot>,
    next_colour: usize,
    /// Voxels per label, for `version`.
    counts: Option<(u64, Vec<usize>)>,
}

impl Segmentation {
    pub fn new(dims: [usize; 3]) -> Self {
        Self { dims, labels: vec![0; dims.iter().product()], segments: Vec::new(), selected: None, version: 1, version_3d: 1, modified: false, undo: Vec::new(), redo: Vec::new(), next_colour: 0, counts: None }
    }

    /// The labels and segments without the history, to try an edit on (previews).
    pub fn scratch_copy(&self) -> Self {
        Self { dims: self.dims, labels: self.labels.clone(), segments: self.segments.clone(), selected: self.selected, version: 0, version_3d: 0, modified: false, undo: Vec::new(), redo: Vec::new(), next_colour: 0, counts: None }
    }

    pub fn selected_segment(&self) -> Option<&Segment> {
        self.selected.and_then(|k| self.segments.get(k))
    }

    pub fn index_of(&self, label: u8) -> Option<usize> {
        self.segments.iter().position(|s| s.label == label)
    }

    /// Add a segment (named `Segment_n` unless given) and select it; `None` when all 255
    /// labels are in use.
    pub fn add_segment(&mut self, name: Option<String>) -> Option<usize> {
        if self.segments.len() >= 255 {
            return None;
        }
        self.checkpoint();
        self.push_segment(name)
    }

    /// [`Segmentation::add_segment`] without an undo step of its own (part of a larger edit).
    pub fn push_segment(&mut self, name: Option<String>) -> Option<usize> {
        let label = (1..=255u8).find(|l| self.index_of(*l).is_none())?;
        let n = (1..).find(|n| !self.segments.iter().any(|s| s.name == format!("Segment_{n}"))).unwrap_or(1);
        let [r, g, b] = PALETTE[self.next_colour % PALETTE.len()];
        self.next_colour += 1;
        self.segments.push(Segment { name: name.unwrap_or_else(|| format!("Segment_{n}")), colour: Color32::from_rgb(r, g, b), visible: true, label, status: Status::NotStarted });
        self.selected = Some(self.segments.len() - 1);
        self.changed(true);
        self.selected
    }

    /// Remove a segment and its voxels.
    pub fn remove_segment(&mut self, k: usize) {
        let Some(s) = self.segments.get(k) else { return };
        let label = s.label;
        self.checkpoint();
        self.labels.par_iter_mut().for_each(|l| {
            if *l == label {
                *l = 0;
            }
        });
        self.segments.remove(k);
        self.selected = if self.segments.is_empty() { None } else { Some(k.min(self.segments.len() - 1)) };
        self.changed(true);
    }

    /// Colour of each label for drawing (transparent: no segment, or hidden).
    pub fn colours(&self) -> Vec<Color32> {
        let mut c = vec![Color32::TRANSPARENT; 256];
        for s in self.segments.iter().filter(|s| s.visible) {
            c[s.label as usize] = s.colour;
        }
        c
    }

    fn rules(&self, masking: &Masking) -> Rules {
        let mut visible = [false; 256];
        for s in self.segments.iter().filter(|s| s.visible) {
            visible[s.label as usize] = true;
        }
        Rules { masking: *masking, visible }
    }

    /// Change segment `label` by `mask` as `mode` says, where `masking` allows. Returns the
    /// voxels changed. (No undo step: call [`Segmentation::checkpoint`] first.)
    pub fn apply(&mut self, vol: &LoadedVolume, mask: &Mask, label: u8, mode: Mode, masking: &Masking) -> usize {
        assert_eq!(mask.data.len(), self.labels.len());
        let rules = self.rules(masking);
        self.labels
            .par_iter_mut()
            .zip(&mask.data)
            .zip(&vol.values)
            .map(|((c, &m), &v)| {
                let want = match mode {
                    Mode::Set => m != 0,
                    Mode::Add if m != 0 => true,
                    Mode::Remove if m != 0 => false,
                    _ => return 0,
                };
                let new = rules.result(*c, v, label, want, false);
                let changed = (new != *c) as usize;
                *c = new;
                changed
            })
            .sum()
    }

    /// Add voxels (by index) to segment `label`, or erase them from it.
    pub fn paint(&mut self, vol: &LoadedVolume, voxels: &[usize], label: u8, erase: bool, masking: &Masking) -> usize {
        let rules = self.rules(masking);
        let mut n = 0;
        for &i in voxels {
            let new = rules.result(self.labels[i], vol.values[i], label, !erase, erase);
            n += (new != self.labels[i]) as usize;
            self.labels[i] = new;
        }
        n
    }

    /// The mask of one label.
    pub fn mask_of(&self, label: u8) -> Mask {
        Mask { dims: self.dims, data: self.labels.par_iter().map(|&l| (l == label) as u8).collect() }
    }

    /// Note an edit: `finished` when it is complete (not a brush dab in a stroke).
    pub fn changed(&mut self, finished: bool) {
        self.version += 1;
        if finished {
            self.version_3d += 1;
        }
        self.modified = true;
    }

    /// Voxels per label (index = label).
    pub fn counts(&mut self) -> &[usize] {
        if self.counts.as_ref().is_none_or(|(v, _)| *v != self.version) {
            let counts = self
                .labels
                .par_chunks(1 << 16)
                .fold(
                    || vec![0usize; 256],
                    |mut h, chunk| {
                        for &l in chunk {
                            h[l as usize] += 1;
                        }
                        h
                    },
                )
                .reduce(|| vec![0usize; 256], |mut a, b| {
                    a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
                    a
                });
            self.counts = Some((self.version, counts));
        }
        &self.counts.as_ref().unwrap().1
    }

    // ------------------------------------------------------------------ undo

    /// Save the current state as an undo step (before an edit).
    pub fn checkpoint(&mut self) {
        let s = self.snapshot();
        self.undo.push(s);
        if self.undo.len() > UNDO_STEPS {
            self.undo.remove(0);
        }
        self.redo.clear();
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    pub fn undo(&mut self) -> bool {
        let Some(s) = self.undo.pop() else { return false };
        let now = self.snapshot();
        self.redo.push(now);
        self.restore(s);
        true
    }

    pub fn redo(&mut self) -> bool {
        let Some(s) = self.redo.pop() else { return false };
        let now = self.snapshot();
        self.undo.push(now);
        self.restore(s);
        true
    }

    fn snapshot(&self) -> Snapshot {
        Snapshot { runs: encode(&self.labels), segments: self.segments.clone(), selected: self.selected }
    }

    fn restore(&mut self, s: Snapshot) {
        decode(&s.runs, &mut self.labels);
        self.segments = s.segments;
        self.selected = s.selected;
        self.changed(true);
    }
}

/// Run-length encoding of a label map (mostly long runs of 0).
fn encode(labels: &[u8]) -> Vec<(u8, u32)> {
    let mut runs: Vec<(u8, u32)> = Vec::new();
    for &l in labels {
        match runs.last_mut() {
            Some((v, n)) if *v == l && *n < u32::MAX => *n += 1,
            _ => runs.push((l, 1)),
        }
    }
    runs
}

fn decode(runs: &[(u8, u32)], out: &mut [u8]) {
    let mut at = 0;
    for &(v, n) in runs {
        out[at..at + n as usize].fill(v);
        at += n as usize;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec3;
    use mm_volume::geometry::Geometry;

    /// A 4 × 1 × 1 volume with values 0, 10, 20, 30.
    fn volume() -> LoadedVolume {
        let g = Geometry::from_axes([4, 1, 1, 1], [1.0; 3], [DVec3::X, DVec3::Y, DVec3::Z], DVec3::ZERO);
        LoadedVolume::from_values("t", [4, 1, 1], vec![0.0, 10.0, 20.0, 30.0], g)
    }

    fn mask(bits: [u8; 4]) -> Mask {
        Mask { dims: [4, 1, 1], data: bits.to_vec() }
    }

    #[test]
    fn masking_rules() {
        let vol = volume();
        let mut s = Segmentation::new([4, 1, 1]);
        let (ka, kb) = (s.add_segment(None).unwrap(), s.add_segment(None).unwrap());
        let (a, b) = (s.segments[ka].label, s.segments[kb].label);
        s.labels = vec![0, a, b, b];
        // Overwrite all: b loses voxel 2 to a; Set clears a where the mask is off.
        let mut t = Segmentation { labels: s.labels.clone(), segments: s.segments.clone(), ..Segmentation::new([4, 1, 1]) };
        assert_eq!(t.apply(&vol, &mask([1, 0, 1, 0]), a, Mode::Set, &Masking::default()), 3);
        assert_eq!(t.labels, vec![a, 0, a, b]);
        // Other segments kept.
        t.labels = s.labels.clone();
        t.apply(&vol, &mask([1, 1, 1, 1]), a, Mode::Add, &Masking { overwrite: Overwrite::None, ..Default::default() });
        assert_eq!(t.labels, vec![a, a, b, b]);
        // Only visible segments overwritten.
        t.labels = s.labels.clone();
        t.segments[1].visible = false;
        t.apply(&vol, &mask([1, 1, 1, 1]), a, Mode::Add, &Masking { overwrite: Overwrite::Visible, ..Default::default() });
        assert_eq!(t.labels, vec![a, a, b, b]);
        t.segments[1].visible = true;
        // Intensity range and editable area.
        t.labels = vec![0; 4];
        t.apply(&vol, &mask([1, 1, 1, 1]), a, Mode::Add, &Masking { intensity: Some((5.0, 25.0)), ..Default::default() });
        assert_eq!(t.labels, vec![0, a, a, 0]);
        t.labels = s.labels.clone();
        t.apply(&vol, &mask([1, 1, 1, 1]), a, Mode::Add, &Masking { area: EditableArea::OutsideAll, ..Default::default() });
        assert_eq!(t.labels, vec![a, a, b, b]);
        // Erasing a with b as the editable area moves the voxels to b (splitting a segment).
        t.labels = vec![a, a, a, b];
        let n = t.paint(&vol, &[1, 2], a, true, &Masking { area: EditableArea::Inside(b), ..Default::default() });
        assert_eq!((n, t.labels.clone()), (2, vec![a, b, b, b]));
        t.labels = vec![a, a, a, b];
        t.paint(&vol, &[1, 2], a, true, &Masking::default());
        assert_eq!(t.labels, vec![a, 0, 0, b]);
        // Set does not split: inside b, a's voxels are not editable.
        t.labels = vec![a, a, a, b];
        t.apply(&vol, &mask([0, 0, 0, 1]), a, Mode::Set, &Masking { area: EditableArea::Inside(b), ..Default::default() });
        assert_eq!(t.labels, vec![a, a, a, a]);
    }

    #[test]
    fn undo_and_redo() {
        let vol = volume();
        let mut s = Segmentation::new([4, 1, 1]);
        s.add_segment(None);
        let a = s.segments[0].label;
        s.checkpoint();
        s.apply(&vol, &mask([0, 1, 1, 0]), a, Mode::Add, &Masking::default());
        s.add_segment(Some("Vessels".into()));
        assert_eq!(s.segments.len(), 2);
        assert!(s.undo());
        assert_eq!((s.segments.len(), s.labels.clone()), (1, vec![0, a, a, 0]));
        assert!(s.undo());
        assert_eq!(s.labels, vec![0; 4]);
        assert!(s.redo() && s.redo());
        assert_eq!((s.segments.len(), s.labels.clone()), (2, vec![0, a, a, 0]));
        assert_eq!(s.counts()[a as usize], 2);
        // Removing a segment clears its voxels; undo brings both back.
        s.remove_segment(0);
        assert_eq!(s.labels, vec![0; 4]);
        s.undo();
        assert_eq!((s.segments[0].label, s.labels.clone()), (a, vec![0, a, a, 0]));
    }

    #[test]
    fn run_length() {
        let labels: Vec<u8> = (0..1000).map(|i| if (300..420).contains(&i) { 3 } else { 0 }).collect();
        let runs = encode(&labels);
        assert_eq!(runs.len(), 3);
        let mut out = vec![9; 1000];
        decode(&runs, &mut out);
        assert_eq!(out, labels);
    }
}
