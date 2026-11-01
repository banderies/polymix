// SPDX-License-Identifier: AGPL-3.0-only
//! What a pattern's copies copy. Ported from ScanRuler
//! (`plugins/reverse/core/patternCopies.ts`, with `patternPitch` and `fullTurn` from
//! `plugins/reverse/core/features.ts`), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! The kernel tracks no history for a pattern, so every face of a copy came out as the
//! pattern's own, and a pick by the feature that made it — the edges where a blade meets the
//! dish — found the original blade's edges alone, or every edge of every copy. Here each face
//! of the result is traced to the face it is a copy of: the faces before the pattern carried
//! through each copy's motion, turned or stepped, and found again where they land. A copy face
//! then keeps the maker of the face it copies, and is marked with the pattern and its number
//! among the copies.
//!
//! The mark rides in the body's origins map, beside the makers, under keys no face hash takes
//! (a hash is never negative): carried wherever the makers are — copied, merged, sent to
//! another kernel — with nothing else to keep.

use std::collections::{HashMap, HashSet};

use glam::{DAffine3, DMat3, DVec3};
use serde::{Deserialize, Serialize};

pub use crate::model::{full_turn, pattern_pitch, PatternMode};
use crate::topo::{face_distance, transform_fingerprint, FaceFingerprint, FaceInfo, MatchTolerance};

/// A face a pattern copied: the pattern's id, and which copy it is on — 1 the first copy, the
/// original being 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CopyMark {
    pub pattern: i64,
    pub instance: i64,
}

/// A body's origins map: by a face's hash, the id of the feature that made it — and, under
/// negative keys, the copy marks.
pub type Origins = HashMap<i64, i64>;

/// The key a face's pattern is kept under in an origins map: −1 − hash.
fn pattern_key(hash: u32) -> i64 {
    -1 - hash as i64
}

/// The key a face's copy number is kept under: −2³¹ − 1 − hash. A hash is under 2³¹, so
/// neither key meets a face's key nor the other.
fn instance_key(hash: u32) -> i64 {
    -0x8000_0001 - hash as i64
}

/// The mark of a face in an origins map, or `None` when no pattern copied it.
pub fn copy_mark_of(origins: &Origins, hash: u32) -> Option<CopyMark> {
    let pattern = *origins.get(&pattern_key(hash))?;
    let instance = *origins.get(&instance_key(hash))?;
    Some(CopyMark { pattern, instance })
}

/// A face marked in an origins map — or its mark taken off, for `None`.
pub fn mark_copy(origins: &mut Origins, hash: u32, mark: Option<CopyMark>) {
    match mark {
        None => {
            origins.remove(&pattern_key(hash));
            origins.remove(&instance_key(hash));
        }
        Some(m) => {
            origins.insert(pattern_key(hash), m.pattern);
            origins.insert(instance_key(hash), m.instance);
        }
    }
}

/// Whether a key of an origins map is a face's, not a mark's.
pub fn is_face_key(key: i64) -> bool {
    key >= 0
}

/// A grid's second direction: its way, how many rows and the step between them, mm.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PatternGrid {
    pub dir: DVec3,
    pub count: usize,
    pub step: f64,
}

/// A pattern as the builder plays it: round the axis through `origin` along `dir` (`step` the
/// angle spanned, degrees), or along `dir` (`step` the distance between copies, mm), `count`
/// in all, the original counted.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PatternMotion {
    pub mode: PatternMode,
    pub origin: DVec3,
    pub dir: DVec3,
    pub count: usize,
    pub step: f64,
    pub grid: Option<PatternGrid>,
}

/// A vector made unit, or `None` for one shorter than 1e-12.
fn normalize(a: DVec3) -> Option<DVec3> {
    let l = a.length();
    (l >= 1e-12).then(|| a / l)
}

/// A turn of `angle` radians about an axis, through the unit quaternion as the kernel's
/// rigid motions build it; none for no axis or no angle.
fn turn_about(axis: DVec3, angle: f64) -> DMat3 {
    let len = axis.length();
    if !(len > 1e-20) || !(angle.abs() > 1e-20) {
        return DMat3::IDENTITY;
    }
    let half = angle / 2.0;
    let s = half.sin() / len;
    let (mut w, mut x, mut y, mut z) = (half.cos(), axis.x * s, axis.y * s, axis.z * s);
    let n = (w * w + x * x + y * y + z * z).sqrt();
    if !(n > 1e-20) {
        return DMat3::IDENTITY;
    }
    w /= n;
    x /= n;
    y /= n;
    z /= n;
    let rows = [
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y - w * z),
        2.0 * (x * z + w * y),
        2.0 * (x * y + w * z),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z - w * x),
        2.0 * (x * z - w * y),
        2.0 * (y * z + w * x),
        1.0 - 2.0 * (x * x + y * y),
    ];
    DMat3::from_cols_array(&rows).transpose()
}

/// Each copy's motion, by its number: a row's copies first, then each row of a grid after the
/// one before. The first, the original, stays.
pub fn copy_motions(p: &PatternMotion) -> Vec<DAffine3> {
    let d = normalize(p.dir).unwrap_or(DVec3::Z);
    let mut row: Vec<DAffine3> = Vec::with_capacity(p.count);
    for i in 0..p.count {
        if p.mode == PatternMode::Linear {
            row.push(DAffine3::from_translation(d * (i as f64 * p.step)));
            continue;
        }
        // Turned about the axis through the origin: R·p + (o − R·o).
        let r = turn_about(d, (i as f64 * pattern_pitch(p.step, p.count as f64) * std::f64::consts::PI) / 180.0);
        let at = r * p.origin;
        row.push(DAffine3 { matrix3: r, translation: p.origin - at });
    }
    let Some(grid) = p.grid.filter(|g| g.count >= 2) else { return row };
    let g = normalize(grid.dir).unwrap_or(DVec3::Z);
    let mut all = Vec::with_capacity(row.len() * grid.count);
    for j in 0..grid.count {
        let shift = DAffine3::from_translation(g * (j as f64 * grid.step));
        for m in &row {
            all.push(shift * *m);
        }
    }
    all
}

/// How near a copy face must land on where its original's motion takes it: a copy is the
/// original moved exactly, so only a hair of rounding — a face a copy's neighbour cut into
/// lands nowhere near and stays the pattern's own. `diagonal`: the result's size.
fn copy_tolerance(diagonal: f64) -> MatchTolerance {
    MatchTolerance { position: (diagonal * 1e-5).max(1e-4), angle: 1e-3, size: 1e-4 }
}

/// A face as the trace compares it: its fingerprint, its maker set aside.
fn print(f: &FaceInfo) -> FaceFingerprint {
    FaceFingerprint { origin: 0, ..f.fingerprint }
}

/// A pattern's result traced to the faces before it.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TracedCopies {
    /// From each face before to the faces of the result that are it moved by a copy's motion.
    pub history: HashMap<u32, Vec<u32>>,
    /// The copies' numbers by the result's hash — the copies only, the original's faces
    /// keeping theirs.
    pub copies: HashMap<u32, usize>,
}

/// The faces of a pattern's result traced to the faces before it: a history from each face
/// before to the faces that are it moved by a copy's motion (`motions`, `copy_motions`), and
/// the copies' marks by the result's hash — for the copies only, the original's faces keeping
/// theirs. A face of the result that is no face before moved — cut where two copies overlap —
/// is in neither, and stays the pattern's own.
pub fn trace_copies(before: &[FaceInfo], after: &[FaceInfo], motions: &[DAffine3], diagonal: f64) -> TracedCopies {
    let tol = copy_tolerance(diagonal);
    let cell = tol.position * 4.0;
    let key_of = |p: DVec3| ((p.x / cell).floor() as i64, (p.y / cell).floor() as i64, (p.z / cell).floor() as i64);
    let mut grid: HashMap<(i64, i64, i64), Vec<&FaceInfo>> = HashMap::new();
    let kept: HashSet<u32> = before.iter().map(|f| f.hash).collect();
    for f in after {
        if kept.contains(&f.hash) {
            continue;
        }
        grid.entry(key_of(f.fingerprint.centroid)).or_default().push(f);
    }
    let mut out = TracedCopies::default();
    let mut taken: HashSet<u32> = HashSet::new();
    for (instance, m) in motions.iter().enumerate() {
        for b in before {
            let moved = transform_fingerprint(&print(b), m);
            let (x, y, z) = key_of(moved.centroid);
            let mut best: Option<&FaceInfo> = None;
            let mut best_d = f64::INFINITY;
            for i in -1..=1 {
                for j in -1..=1 {
                    for l in -1..=1 {
                        // Saturating: a centroid off at infinity keys to the last cell, as its
                        // neighbours do, rather than overflowing.
                        let Some(list) = grid.get(&(x.saturating_add(i), y.saturating_add(j), z.saturating_add(l))) else { continue };
                        for f in list {
                            if taken.contains(&f.hash) {
                                continue;
                            }
                            if let Some(d) = face_distance(&moved, &print(f), &tol) {
                                if d < best_d {
                                    best_d = d;
                                    best = Some(f);
                                }
                            }
                        }
                    }
                }
            }
            let Some(best) = best else { continue };
            taken.insert(best.hash);
            out.history.entry(b.hash).or_default().push(best.hash);
            if instance > 0 {
                out.copies.insert(best.hash, instance);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topo::FaceKind;

    #[test]
    fn spaces_an_open_span_first_to_last_and_a_full_turn_evenly_round() {
        assert!((pattern_pitch(90.0, 4.0) - 30.0).abs() < 5e-13);
        assert!((pattern_pitch(360.0, 14.0) - 25.714285714).abs() < 5e-9);
        assert!(full_turn(-360.0) && !full_turn(90.0));
    }

    /// Not a ScanRuler test (its pattern tests build on the kernel): a copy mark rides in the
    /// origins map under keys no face takes, and comes off again.
    #[test]
    fn marks_a_copy_beside_the_makers_and_takes_the_mark_off() {
        let mut origins = Origins::new();
        origins.insert(7, 4);
        mark_copy(&mut origins, 7, Some(CopyMark { pattern: 5, instance: 2 }));
        assert_eq!(copy_mark_of(&origins, 7), Some(CopyMark { pattern: 5, instance: 2 }));
        assert_eq!(origins.keys().filter(|k| is_face_key(**k)).count(), 1);
        mark_copy(&mut origins, 7, None);
        assert_eq!(copy_mark_of(&origins, 7), None);
        assert_eq!(origins.len(), 1);
    }

    /// Not a ScanRuler test: a blade's top face turned round Z into four copies is found again
    /// on each, and the copies numbered.
    #[test]
    fn traces_each_copy_to_the_face_it_copies() {
        let face = |hash: u32, centroid: DVec3, dir: DVec3| FaceInfo {
            hash,
            fingerprint: FaceFingerprint { kind: FaceKind::Plane, centroid, dir, area: 40.0, radius: None, center: None, origin: 4 },
            full_turn: None,
            point: centroid,
        };
        let motion = PatternMotion { mode: PatternMode::Circular, origin: DVec3::ZERO, dir: DVec3::Z, count: 4, step: 360.0, grid: None };
        let motions = copy_motions(&motion);
        assert_eq!(motions.len(), 4);
        let top = face(1, DVec3::new(25.0, 0.0, 15.0), DVec3::Z);
        let side = face(2, DVec3::new(25.0, 1.0, 10.0), DVec3::Y);
        let before = vec![top.clone(), side.clone()];
        let mut after = before.clone();
        let mut hash = 10;
        for m in motions.iter().skip(1) {
            for f in &before {
                after.push(face(hash, m.transform_point3(f.fingerprint.centroid), m.transform_vector3(f.fingerprint.dir)));
                hash += 1;
            }
        }
        let traced = trace_copies(&before, &after, &motions, 60.0);
        // The original's faces kept their hashes and are traced to nothing new.
        assert_eq!(traced.history.get(&1), Some(&vec![10, 12, 14]));
        assert_eq!(traced.history.get(&2), Some(&vec![11, 13, 15]));
        assert_eq!(traced.copies.get(&10), Some(&1));
        assert_eq!(traced.copies.get(&15), Some(&3));
        assert_eq!(traced.copies.len(), 6);
        // A grid of two rows doubles the motions, the row first.
        let grid = copy_motions(&PatternMotion { mode: PatternMode::Linear, origin: DVec3::ZERO, dir: DVec3::X, count: 3, step: 10.0, grid: Some(PatternGrid { dir: DVec3::Y, count: 2, step: 5.0 }) });
        assert_eq!(grid.len(), 6);
        assert!((grid[4].transform_point3(DVec3::ZERO) - DVec3::new(10.0, 5.0, 0.0)).length() < 1e-12);
    }
}
