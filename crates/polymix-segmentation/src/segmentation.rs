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

/// A past state, with the label map compressed block by block.
struct Snapshot {
    blocks: Vec<Block>,
    segments: Vec<Segment>,
    selected: Option<usize>,
}

/// Voxels per block of a snapshot. Blocks are compressed and restored in parallel.
const BLOCK: usize = 1 << 20;

/// One block of a label map in a snapshot: never much more than a byte per voxel, however
/// noisy the labels (runs of a thresholded noisy image are a voxel or two long).
enum Block {
    /// Every voxel has this label.
    Same(u8),
    /// Runs of one label: (label, voxels).
    Runs(Vec<(u8, u16)>),
    /// The labels themselves, where runs would take more room.
    Raw(Box<[u8]>),
}

impl Block {
    fn of(labels: &[u8]) -> Self {
        let mut runs: Vec<(u8, u16)> = Vec::new();
        for (l, n) in Runs::new(labels) {
            if n == labels.len() {
                return Block::Same(l);
            }
            push_runs(&mut runs, l, n);
            if runs.len() * std::mem::size_of::<(u8, u16)>() >= labels.len() {
                return Block::Raw(labels.into());
            }
        }
        runs.shrink_to_fit();
        Block::Runs(runs)
    }

    /// Write the block's labels into `out` (as long as the block).
    fn restore(&self, out: &mut [u8]) {
        match self {
            Block::Same(l) => out.fill(*l),
            Block::Runs(runs) => decode(runs, out),
            Block::Raw(labels) => out.copy_from_slice(labels),
        }
    }
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
    /// Changes when a segment's colour, name or status does, which leaves the labels as they
    /// were (the slice views only colour them again).
    pub style_version: u64,
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
        Self { dims, labels: vec![0; dims.iter().product()], segments: Vec::new(), selected: None, version: 1, version_3d: 1, style_version: 1, modified: false, undo: Vec::new(), redo: Vec::new(), next_colour: 0, counts: None }
    }

    /// The labels and segments without the history, to try an edit on (the previews are made
    /// in one pass now; the tests check them against edits of a copy).
    #[cfg(test)]
    pub fn scratch_copy(&self) -> Self {
        Self { dims: self.dims, labels: self.labels.clone(), segments: self.segments.clone(), selected: self.selected, version: 0, version_3d: 0, style_version: 0, modified: false, undo: Vec::new(), redo: Vec::new(), next_colour: 0, counts: None }
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

    /// The labels as a Threshold of `range` into segment `label` would leave them: what
    /// [`Segmentation::apply`] of the threshold's mask (Set) does, written into `out` in one
    /// pass without the mask or a copy of the labels.
    pub fn threshold_preview(&self, vol: &LoadedVolume, range: (f32, f32), label: u8, masking: &Masking, out: &mut Vec<u8>) {
        assert_eq!(vol.values.len(), self.labels.len());
        let rules = self.rules(masking);
        let (lo, hi) = range;
        // Without masking, the rules come down to: in the range, the segment; out of it, not
        // the segment (a choice the loop can make for many voxels at once).
        let plain = *masking == Masking::default();
        out.resize(self.labels.len(), 0);
        out.par_chunks_mut(1 << 16).zip(self.labels.par_chunks(1 << 16)).zip(vol.values.par_chunks(1 << 16)).for_each(|((out, labels), values)| {
            if plain {
                for ((o, &c), &v) in out.iter_mut().zip(labels).zip(values) {
                    *o = if v >= lo && v <= hi { label } else if c == label { 0 } else { c };
                }
            } else {
                for ((o, &c), &v) in out.iter_mut().zip(labels).zip(values) {
                    *o = rules.result(c, v, label, v >= lo && v <= hi, false);
                }
            }
        });
    }

    /// The labels as keeping a seeded effect's result would leave them, written into `out`:
    /// each voxel whose `region` label is one of `seeds` added to that segment (as
    /// [`Segmentation::apply`] of each seed's mask, Add, one after another, does).
    pub fn seeded_preview(&self, vol: &LoadedVolume, region: &[u8], seeds: &[u8], masking: &Masking, out: &mut Vec<u8>) {
        out.clear();
        out.extend_from_slice(&self.labels);
        Self::add_seeded(out, &self.rules(masking), vol, region, seeds);
    }

    /// Keep a seeded effect's result (see [`Segmentation::seeded_preview`]). Returns the
    /// voxels changed. (No undo step: call [`Segmentation::checkpoint`] first.)
    pub fn apply_seeded(&mut self, vol: &LoadedVolume, region: &[u8], seeds: &[u8], masking: &Masking) -> usize {
        let rules = self.rules(masking);
        Self::add_seeded(&mut self.labels, &rules, vol, region, seeds)
    }

    /// Each seed's mask touches only the voxels with that `region` label, and adding a voxel
    /// again changes nothing, so one pass does what the seeds' edits one after another do.
    fn add_seeded(labels: &mut [u8], rules: &Rules, vol: &LoadedVolume, region: &[u8], seeds: &[u8]) -> usize {
        assert_eq!(region.len(), labels.len());
        assert_eq!(vol.values.len(), labels.len());
        let mut seed = [false; 256];
        for &l in seeds {
            seed[l as usize] = true;
        }
        labels
            .par_chunks_mut(1 << 16)
            .zip(region.par_chunks(1 << 16))
            .zip(vol.values.par_chunks(1 << 16))
            .map(|((labels, region), values)| {
                let mut n = 0;
                for ((c, &x), &v) in labels.iter_mut().zip(region).zip(values) {
                    if seed[x as usize] {
                        let new = rules.result(*c, v, x, true, false);
                        n += (new != *c) as usize;
                        *c = new;
                    }
                }
                n
            })
            .sum()
    }

    /// Islands' Split of segment `label`: it keeps the voxels in slot 1, and the voxels in slot
    /// `2 + k` become new segment `k` (named after it, `_2`, `_3`…), for `islands` islands or
    /// as many as there are labels left. Returns the new segments made. (No undo step: call
    /// [`Segmentation::checkpoint`] first.)
    pub fn split(&mut self, vol: &LoadedVolume, label: u8, slots: &[u8], islands: usize) -> usize {
        assert_eq!(slots.len(), self.labels.len());
        assert_eq!(vol.values.len(), self.labels.len());
        let base = self.index_of(label).map(|k| self.segments[k].name.clone()).unwrap_or_default();
        // The new segments first (they add no voxels), then one pass: Set of the piece kept,
        // then Add of each island, as their masks did one after another. Every masking rule
        // allows everything here, so the segments' visibility does not matter.
        let mut to = [0u8; 256];
        let mut made = 0;
        for k in 0..islands.min(254) {
            let Some(idx) = self.push_segment(Some(format!("{base}_{}", k + 2))) else { break };
            to[2 + k] = self.segments[idx].label;
            made += 1;
        }
        let rules = self.rules(&Masking::default());
        self.labels.par_chunks_mut(1 << 16).zip(slots.par_chunks(1 << 16)).zip(vol.values.par_chunks(1 << 16)).for_each(|((labels, slots), values)| {
            for ((c, &s), &v) in labels.iter_mut().zip(slots).zip(values) {
                let kept = rules.result(*c, v, label, s == 1, false);
                *c = if to[s as usize] != 0 { rules.result(kept, v, to[s as usize], true, false) } else { kept };
            }
        });
        self.selected = self.index_of(label);
        self.changed(true);
        made
    }

    /// Add voxels (by index) to segment `label`, or erase them from it. A voxel listed twice
    /// is changed once (painting it again changes nothing). A brush dab: the slice views
    /// follow it (`version`); the stroke's end is [`Segmentation::stroke_finished`].
    pub fn paint(&mut self, vol: &LoadedVolume, voxels: &[usize], label: u8, erase: bool, masking: &Masking) -> usize {
        let rules = self.rules(masking);
        // The voxels per label follow the dab, rather than being counted again over the
        // whole label map for the panel.
        let mut counts = self.counts.take().filter(|(v, _)| *v == self.version).map(|(_, c)| c);
        let mut n = 0;
        for &i in voxels {
            let old = self.labels[i];
            let new = rules.result(old, vol.values[i], label, !erase, erase);
            if new != old {
                self.labels[i] = new;
                n += 1;
                if let Some(c) = &mut counts {
                    c[old as usize] -= 1;
                    c[new as usize] += 1;
                }
            }
        }
        if n > 0 {
            self.changed(false);
        }
        self.counts = counts.map(|c| (self.version, c));
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

    /// Note the end of a brush stroke: its dabs changed the labels already, and the 3D view
    /// catches up now.
    pub fn stroke_finished(&mut self) {
        self.version_3d += 1;
        self.modified = true;
    }

    /// Note a change of segments' colours, names or status (not of the labels).
    pub fn restyled(&mut self) {
        self.style_version += 1;
        self.modified = true;
    }

    /// Voxels per label (index = label).
    pub fn counts(&mut self) -> &[usize] {
        if self.counts.as_ref().is_none_or(|(v, _)| *v != self.version) {
            self.counts = Some((self.version, histogram(&self.labels)));
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
        Snapshot { blocks: self.labels.par_chunks(BLOCK).map(Block::of).collect(), segments: self.segments.clone(), selected: self.selected }
    }

    fn restore(&mut self, s: Snapshot) {
        self.labels.par_chunks_mut(BLOCK).zip(&s.blocks).for_each(|(out, b)| b.restore(out));
        self.segments = s.segments;
        self.selected = s.selected;
        self.changed(true);
    }
}

/// Voxels per label (index = label). Each block is counted in four interleaved tables of
/// 32-bit counters, so that a run of one label does not wait on a single counter.
fn histogram(labels: &[u8]) -> Vec<usize> {
    labels
        .par_chunks(1 << 16)
        .fold(
            || vec![0usize; 256],
            |mut h, chunk| {
                let mut t = [[0u32; 256]; 4];
                let (quads, rest) = chunk.as_chunks::<4>();
                for q in quads {
                    t[0][q[0] as usize] += 1;
                    t[1][q[1] as usize] += 1;
                    t[2][q[2] as usize] += 1;
                    t[3][q[3] as usize] += 1;
                }
                for &l in rest {
                    t[0][l as usize] += 1;
                }
                for (k, x) in h.iter_mut().enumerate() {
                    *x += (t[0][k] + t[1][k] + t[2][k] + t[3][k]) as usize;
                }
                h
            },
        )
        .reduce(|| vec![0usize; 256], |mut a, b| {
            a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
            a
        })
}

/// The runs of equal labels in a label map, in order: (label, voxels).
struct Runs<'a> {
    labels: &'a [u8],
    at: usize,
}

impl<'a> Runs<'a> {
    fn new(labels: &'a [u8]) -> Self {
        Self { labels, at: 0 }
    }
}

impl Iterator for Runs<'_> {
    type Item = (u8, usize);

    fn next(&mut self) -> Option<(u8, usize)> {
        let l = *self.labels.get(self.at)?;
        let start = self.at;
        let mut i = start + 1;
        // Eight voxels at a time while they all have the label (long runs of 0 mostly), then
        // one at a time.
        let word = u64::from_ne_bytes([l; 8]);
        while let Some(w) = self.labels.get(i..i + 8) {
            if u64::from_ne_bytes(w.try_into().expect("eight bytes")) != word {
                break;
            }
            i += 8;
        }
        while self.labels.get(i) == Some(&l) {
            i += 1;
        }
        self.at = i;
        Some((l, i - start))
    }
}

/// Append a run of `n` voxels of label `l`, split into runs a u16 can count.
fn push_runs(runs: &mut Vec<(u8, u16)>, l: u8, mut n: usize) {
    while n > 0 {
        let m = n.min(u16::MAX as usize);
        runs.push((l, m as u16));
        n -= m;
    }
}

fn decode(runs: &[(u8, u16)], out: &mut [u8]) {
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
        let Block::Runs(runs) = Block::of(&labels) else { panic!("three runs") };
        assert_eq!(runs, vec![(0, 300), (3, 120), (0, 580)]);
        let mut out = vec![9; 1000];
        decode(&runs, &mut out);
        assert_eq!(out, labels);
        assert!(matches!(Block::of(&[7; 5000]), Block::Same(7)));
        // Runs longer than a u16 counts are split.
        let mut runs = Vec::new();
        push_runs(&mut runs, 4, 3 * 65_535 + 2);
        assert_eq!(runs, vec![(4, 65_535), (4, 65_535), (4, 65_535), (4, 2)]);
        // A noisy block keeps its labels.
        let noisy: Vec<u8> = (0..4096u32).map(|i| (i.wrapping_mul(2_654_435_761) >> 30) as u8).collect();
        assert!(matches!(Block::of(&noisy), Block::Raw(_)));
    }

    /// A random label map in runs of random lengths (1 to `longest` voxels) of labels below
    /// `labels`.
    fn random_labels(n: usize, labels: u16, longest: usize, seed: u64) -> Vec<u8> {
        let mut state = seed | 1;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            let (l, len) = ((next() % labels as u64) as u8, 1 + (next() % longest as u64) as usize);
            out.extend(std::iter::repeat_n(l, len.min(n - out.len())));
        }
        out
    }

    #[test]
    fn runs_are_found_eight_voxels_at_a_time_as_one_at_a_time() {
        // The runs a byte at a time, as the label map was first encoded.
        let one_by_one = |labels: &[u8]| {
            let mut runs: Vec<(u8, usize)> = Vec::new();
            for &l in labels {
                match runs.last_mut() {
                    Some((v, n)) if *v == l => *n += 1,
                    _ => runs.push((l, 1)),
                }
            }
            runs
        };
        for case in 0..400u64 {
            let n = [0, 1, 7, 8, 9, 15, 16, 17, 100, 1000, 5000][case as usize % 11];
            let labels = random_labels(n, [2, 3, 255][case as usize % 3], [1, 3, 9, 40, 700][case as usize % 5], case);
            assert_eq!(Runs::new(&labels).collect::<Vec<_>>(), one_by_one(&labels), "case {case}");
        }
    }

    #[test]
    fn snapshots_restore_the_labels() {
        // Label maps over several blocks: empty, clean, noisy, and noisy in places.
        let n = 2 * BLOCK + 12_345;
        let mut maps = vec![vec![0u8; n], random_labels(n, 4, 200_000, 1), random_labels(n, 3, 2, 2), random_labels(n, 256, 1, 3)];
        let mut mixed = random_labels(n, 2, 50_000, 4);
        mixed[BLOCK..BLOCK + 9000].copy_from_slice(&random_labels(9000, 255, 3, 5));
        maps.push(mixed);
        for (k, labels) in maps.into_iter().enumerate() {
            let mut s = Segmentation::new([n, 1, 1]);
            s.labels = labels.clone();
            let snap = s.snapshot();
            // Never much more than a byte per voxel.
            let bytes: usize = snap.blocks.iter().map(|b| match b {
                Block::Same(_) => 1,
                Block::Runs(r) => r.len() * 4,
                Block::Raw(r) => r.len(),
            }).sum();
            assert!(bytes <= n, "map {k}: {bytes} bytes");
            s.labels.iter_mut().for_each(|l| *l = l.wrapping_add(1));
            s.restore(snap);
            assert!(s.labels == labels, "map {k}");
        }
    }

    #[test]
    fn undo_and_redo_of_noisy_edits() {
        // Edits of a noisy label map, undone and redone: every state comes back exactly.
        let n = BLOCK + 777;
        let g = Geometry::from_axes([n, 1, 1, 1], [1.0; 3], [DVec3::X, DVec3::Y, DVec3::Z], DVec3::ZERO);
        let vol = LoadedVolume::from_values("t", [n, 1, 1], (0..n).map(|i| (i % 97) as f32).collect(), g);
        let mut s = Segmentation::new([n, 1, 1]);
        s.add_segment(None);
        s.add_segment(None);
        let mut states = vec![s.labels.clone()];
        for (k, (lo, hi)) in [(10.0, 50.0), (30.0, 31.0), (0.0, 96.0), (90.0, 96.0)].into_iter().enumerate() {
            s.checkpoint();
            let mask = mm_segment::ops::threshold(&vol, lo, hi);
            s.apply(&vol, &mask, 1 + (k % 2) as u8, Mode::Set, &Masking::default());
            s.changed(true);
            states.push(s.labels.clone());
        }
        for k in (0..4).rev() {
            assert!(s.undo());
            assert!(s.labels == states[k], "undo to {k}");
        }
        for (k, state) in states.iter().enumerate().skip(1) {
            assert!(s.redo());
            assert!(s.labels == *state, "redo to {k}");
        }
        assert!(!s.redo());
    }

    /// A volume of `n` voxels in a row with values 0, 1, 2… (mod 101), and a segmentation of
    /// it with three segments over random labels.
    fn random_segmentation(n: usize, seed: u64) -> (LoadedVolume, Segmentation) {
        let g = Geometry::from_axes([n, 1, 1, 1], [1.0; 3], [DVec3::X, DVec3::Y, DVec3::Z], DVec3::ZERO);
        let vol = LoadedVolume::from_values("t", [n, 1, 1], (0..n).map(|i| (i % 101) as f32).collect(), g);
        let mut s = Segmentation::new([n, 1, 1]);
        for _ in 0..3 {
            s.add_segment(None);
        }
        s.labels = random_labels(n, 5, 6, seed);
        s.segments[1].visible = false;
        s.changed(true);
        (vol, s)
    }

    /// Masking settings worth trying: each editable area, an intensity range, each overwrite.
    fn maskings() -> Vec<Masking> {
        let areas = [EditableArea::Everywhere, EditableArea::InsideAll, EditableArea::InsideVisible, EditableArea::OutsideAll, EditableArea::OutsideVisible, EditableArea::Inside(2), EditableArea::Outside(3)];
        let mut out = Vec::new();
        for (k, area) in areas.into_iter().enumerate() {
            for overwrite in [Overwrite::All, Overwrite::Visible, Overwrite::None] {
                out.push(Masking { area, intensity: (k % 2 == 1).then_some((20.0, 70.0)), overwrite });
            }
        }
        out
    }

    #[test]
    fn counts_follow_the_brush() {
        let (vol, mut s) = random_segmentation(5000, 7);
        let mut state = 99u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        s.counts();
        for (k, masking) in maskings().iter().cycle().take(200).enumerate() {
            // Dabs of random voxels, some listed twice.
            let mut voxels: Vec<usize> = (0..(next() % 300) as usize).map(|_| (next() % 5000) as usize).collect();
            voxels.extend_from_within(..voxels.len() / 3);
            let before = s.labels.clone();
            let (label, erase) = (1 + (next() % 4) as u8, k % 3 == 0);
            let version = s.version;
            let n = s.paint(&vol, &voxels, label, erase, masking);
            assert_eq!(n, before.iter().zip(&s.labels).filter(|(a, b)| a != b).count());
            assert_eq!(s.version, version + (n > 0) as u64, "a dab that changes voxels is a new version");
            let kept = s.counts.as_ref().map(|c| c.1.clone()).expect("kept up to date");
            assert_eq!(kept, histogram(&s.labels), "dab {k}");
            if k % 50 == 49 {
                // An edit counted again from scratch.
                s.changed(true);
                let fresh = histogram(&s.labels);
                assert_eq!(s.counts(), fresh.as_slice());
            }
        }
        // Counting in four tables gives what one does.
        for labels in [random_labels(70_001, 256, 3, 11), vec![5; 3], Vec::new()] {
            let mut one = vec![0usize; 256];
            labels.iter().for_each(|&l| one[l as usize] += 1);
            assert_eq!(histogram(&labels), one);
        }
    }

    #[test]
    fn previews_are_what_their_edits_leave() {
        for seed in 0..4 {
            let (vol, s) = random_segmentation(3000, seed);
            let mut buffer = Vec::new();
            for masking in maskings() {
                for (range, label) in [((10.0, 60.0), 1), ((50.0, 50.0), 2), ((-5.0, 200.0), 4), ((80.0, 20.0), 1)] {
                    // As it was made: a copy of the labels, the threshold's mask and Set.
                    let mut copy = s.scratch_copy();
                    copy.apply(&vol, &mm_segment::ops::threshold(&vol, range.0, range.1), label, Mode::Set, &masking);
                    s.threshold_preview(&vol, range, label, &masking, &mut buffer);
                    assert!(buffer == copy.labels, "{masking:?} {range:?}");
                }
                // A seeded effect's result: each seed's region added to it in turn.
                let region = random_labels(3000, 5, 4, seed + 10);
                for seeds in [vec![1, 2], vec![2, 1, 4], vec![3, 3], vec![]] {
                    let mut copy = s.scratch_copy();
                    let mut n = 0;
                    for &l in &seeds {
                        n += copy.apply(&vol, &Mask { dims: vol.dims, data: region.iter().map(|&x| (x == l) as u8).collect() }, l, Mode::Add, &masking);
                    }
                    s.seeded_preview(&vol, &region, &seeds, &masking, &mut buffer);
                    assert!(buffer == copy.labels, "{masking:?} {seeds:?}");
                    let mut kept = s.scratch_copy();
                    assert_eq!(kept.apply_seeded(&vol, &region, &seeds, &masking), n);
                    assert!(kept.labels == copy.labels);
                }
            }
        }
    }

    #[test]
    fn splitting_islands_as_their_masks_did() {
        for seed in 0..6 {
            let (vol, s) = random_segmentation(4000, seed + 20);
            let label = 2;
            let islands = [0, 1, 3, 7][seed as usize % 4];
            let slots: Vec<u8> = random_labels(4000, 2 + islands as u16, 5, seed + 30);
            // As it was: Set of the piece kept, then each island added as a new segment.
            let mut old = s.scratch_copy();
            let all = Masking::default();
            let keep = Mask { dims: vol.dims, data: slots.iter().map(|&x| (x == 1) as u8).collect() };
            old.apply(&vol, &keep, label, Mode::Set, &all);
            for k in 0..islands {
                let idx = old.push_segment(Some(format!("Segment_2_{}", k + 2))).unwrap();
                let l = old.segments[idx].label;
                old.apply(&vol, &Mask { dims: vol.dims, data: slots.iter().map(|&x| (x as usize == k + 2) as u8).collect() }, l, Mode::Add, &all);
            }
            let mut new = s.scratch_copy();
            assert_eq!(new.split(&vol, label, &slots, islands), islands);
            assert!(new.labels == old.labels, "seed {seed}");
            assert_eq!(new.segments, old.segments);
            assert_eq!(new.selected, new.index_of(label));
        }
    }

    #[test]
    fn styles_and_strokes_leave_the_labels_version() {
        let vol = volume();
        let mut s = Segmentation::new([4, 1, 1]);
        s.add_segment(None);
        let (v, v3, style) = (s.version, s.version_3d, s.style_version);
        s.modified = false;
        s.restyled();
        assert_eq!((s.version, s.version_3d, s.style_version, s.modified), (v, v3, style + 1, true));
        s.paint(&vol, &[1, 2], 1, false, &Masking::default());
        assert_eq!((s.version, s.version_3d), (v + 1, v3), "a dab: the slice views follow");
        s.paint(&vol, &[1, 2], 1, false, &Masking::default());
        assert_eq!(s.version, v + 1, "nothing changed");
        s.stroke_finished();
        assert_eq!((s.version, s.version_3d), (v + 1, v3 + 1), "the stroke's end: the 3D view follows");
    }
}
