// SPDX-License-Identifier: AGPL-3.0-only
//! Where a deviation map lies past its tolerance, as patches. Ported from ScanRuler
//! (`src/core/deviation/hotspots.ts`, `src/core/deviation/patchKind.ts`), © Stefan Hermann
//! (CNC Kitchen), AGPL-3.0-only.
//!
//! The vertices over tolerance are gathered into connected regions, each with its centre, its
//! box, its area, its mean and its extreme reading. Connectivity is by distance, not by the
//! scan's triangles: two vertices over tolerance within `grain` of each other, on the same side
//! of the surface, are one patch. The grain is a few point spacings, read off the points
//! themselves — the median distance to the nearest other vertex over tolerance — which gives
//! the area too, as count × spacing².
//!
//! Each patch says how deep it lies (the median reading), how much lies past the tolerance in
//! mm³, and so whether it is a feature or texture: a feature is broad and deep; texture is
//! either small or shallow.

use std::collections::HashMap;

use glam::DVec3;
use serde::Serialize;

/// A feature patch is at least this many mm² …
pub const FEATURE_MIN_AREA: f64 = 4.0;
/// … and at least this many point spacings² …
pub const FEATURE_MIN_SPACINGS: f64 = 25.0;
/// … and its median reading at least this many tolerances off.
pub const FEATURE_DEPTH: f64 = 2.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum PatchKind {
    Feature,
    Texture,
}

impl PatchKind {
    pub fn name(self) -> &'static str {
        match self {
            PatchKind::Feature => "feature",
            PatchKind::Texture => "texture",
        }
    }
}

/// Whether a patch of this area (mm²) and depth is a feature, at this point spacing and the
/// depth a feature reaches.
pub fn patch_kind(area: f64, depth: f64, spacing: f64, feature_depth: f64) -> PatchKind {
    if area >= FEATURE_MIN_AREA.max(FEATURE_MIN_SPACINGS * spacing * spacing) && depth >= feature_depth {
        PatchKind::Feature
    } else {
        PatchKind::Texture
    }
}

/// One patch.
#[derive(Debug, Clone)]
pub struct Hotspot {
    pub count: usize,
    pub centroid: DVec3,
    pub min: DVec3,
    pub max: DVec3,
    pub mean: f64,
    pub extreme: f64,
    /// Σ (|d| − tolerance) over the patch.
    pub excess: f64,
    /// mm², from the point spacing.
    pub area: f64,
    /// mm³ past the tolerance.
    pub volume: f64,
    /// The median |reading| over the patch.
    pub depth: f64,
    pub kind: PatchKind,
    /// Where the extreme reading is, and which entry it is.
    pub at: DVec3,
    pub index: usize,
    /// Every entry in the patch.
    pub members: Vec<usize>,
}

/// How the patches are asked for.
pub struct HotspotOptions<'a> {
    pub grain: Option<f64>,
    pub limit: usize,
    pub min_count: usize,
    pub only: Option<&'a dyn Fn(usize) -> bool>,
    pub group: Option<&'a dyn Fn(usize) -> u64>,
    pub feature_depth: Option<f64>,
}

impl Default for HotspotOptions<'_> {
    fn default() -> Self {
        HotspotOptions { grain: None, limit: DEFAULT_LIMIT, min_count: DEFAULT_MIN_COUNT, only: None, group: None, feature_depth: None }
    }
}

#[derive(Debug, Clone, Default)]
pub struct HotspotReport {
    pub patches: Vec<Hotspot>,
    pub found: usize,
    pub over: usize,
    pub features_count: usize,
    pub features_share_of_excess: f64,
    pub spacing: f64,
    pub grain: f64,
}

const GRAIN_SPACINGS: f64 = 3.0;
const DEFAULT_LIMIT: usize = 8;
const DEFAULT_MIN_COUNT: usize = 3;
const SPACING_SAMPLE: usize = 2000;

struct UnionFind(Vec<usize>);

impl UnionFind {
    fn find(&mut self, mut a: usize) -> usize {
        while self.0[a] != a {
            self.0[a] = self.0[self.0[a]];
            a = self.0[a];
        }
        a
    }
    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            self.0[rb] = ra;
        }
    }
}

/// A hash grid over the selected entries.
struct Grid {
    cell: f64,
    cells: HashMap<(i64, i64, i64), Vec<usize>>,
}

impl Grid {
    fn new(points: &[DVec3], sel: &[usize], cell: f64) -> Grid {
        let mut cells: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
        for (i, e) in sel.iter().enumerate() {
            let p = points[*e];
            cells.entry(((p.x / cell).floor() as i64, (p.y / cell).floor() as i64, (p.z / cell).floor() as i64)).or_default().push(i);
        }
        Grid { cell, cells }
    }

    fn near(&self, p: DVec3, mut visit: impl FnMut(usize)) {
        let (cx, cy, cz) = ((p.x / self.cell).floor() as i64, (p.y / self.cell).floor() as i64, (p.z / self.cell).floor() as i64);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(list) = self.cells.get(&(cx + dx, cy + dy, cz + dz)) {
                        for j in list {
                            visit(*j);
                        }
                    }
                }
            }
        }
    }
}

/// The median distance from a selected entry to the nearest other one, over a sample.
pub fn point_spacing(points: &[DVec3], sel: &[usize]) -> f64 {
    if sel.len() < 2 {
        return 0.0;
    }
    let mut lo = DVec3::splat(f64::INFINITY);
    let mut hi = DVec3::splat(f64::NEG_INFINITY);
    for e in sel {
        lo = lo.min(points[*e]);
        hi = hi.max(points[*e]);
    }
    let diag = (hi - lo).length();
    let cell = (diag / 4f64.max((sel.len() as f64).cbrt() * 2.0)).max(1e-9);
    let grid = Grid::new(points, sel, cell);
    let stride = (sel.len() / SPACING_SAMPLE).max(1);
    let mut nearest = Vec::new();
    let mut i = 0;
    while i < sel.len() {
        let a = points[sel[i]];
        let mut best = f64::INFINITY;
        grid.near(a, |j| {
            if j == i {
                return;
            }
            let d2 = points[sel[j]].distance_squared(a);
            if d2 < best {
                best = d2;
            }
        });
        if best.is_finite() {
            nearest.push(best.sqrt());
        }
        i += stride;
    }
    if nearest.is_empty() {
        return cell;
    }
    nearest.sort_by(f64::total_cmp);
    nearest[nearest.len() / 2]
}

/// The patches of `points` whose `values` lie past `tolerance` either way.
pub fn hotspots(points: &[DVec3], values: &[f64], tolerance: f64, options: &HotspotOptions) -> HotspotReport {
    let sel: Vec<usize> = (0..values.len()).filter(|i| values[*i].is_finite() && values[*i].abs() > tolerance && options.only.is_none_or(|f| f(*i))).collect();
    let empty = HotspotReport { over: sel.len(), grain: options.grain.unwrap_or(0.0), ..Default::default() };
    if sel.is_empty() {
        return empty;
    }
    let spacing = point_spacing(points, &sel);
    let grain = options.grain.unwrap_or((spacing * GRAIN_SPACINGS).max(1e-6));
    if !(grain > 0.0) {
        return HotspotReport { spacing, ..empty };
    }
    let m = sel.len();
    let mut uf = UnionFind((0..m).collect());
    let grid = Grid::new(points, &sel, grain);
    let g2 = grain * grain;
    for i in 0..m {
        let a = sel[i];
        let pa = points[a];
        let sign = values[a] > 0.0;
        let ga = options.group.map(|g| g(a));
        let mut joins = Vec::new();
        grid.near(pa, |j| {
            if j <= i {
                return;
            }
            let b = sel[j];
            if (values[b] > 0.0) != sign {
                return;
            }
            if let Some(g) = options.group {
                if Some(g(b)) != ga {
                    return;
                }
            }
            if points[b].distance_squared(pa) <= g2 {
                joins.push(j);
            }
        });
        for j in joins {
            uf.union(i, j);
        }
    }
    let mut by_root: Vec<(usize, Vec<usize>)> = Vec::new();
    let mut root_slot: HashMap<usize, usize> = HashMap::new();
    for i in 0..m {
        let r = uf.find(i);
        let slot = *root_slot.entry(r).or_insert_with(|| {
            by_root.push((r, Vec::new()));
            by_root.len() - 1
        });
        by_root[slot].1.push(i);
    }
    let feature_depth = options.feature_depth.unwrap_or(FEATURE_DEPTH * tolerance);
    let all_excess: f64 = sel.iter().map(|e| values[*e].abs() - tolerance).sum();
    let (mut feature_excess, mut feature_count) = (0.0, 0);
    let mut patches = Vec::new();
    for (_, list) in &by_root {
        if list.len() < options.min_count {
            continue;
        }
        let n = list.len() as f64;
        let mut min = DVec3::splat(f64::INFINITY);
        let mut max = DVec3::splat(f64::NEG_INFINITY);
        let mut centroid = DVec3::ZERO;
        let (mut sum, mut excess, mut extreme, mut index) = (0.0, 0.0, 0.0f64, 0usize);
        let mut members = Vec::with_capacity(list.len());
        let mut abs: Vec<f64> = Vec::with_capacity(list.len());
        for i in list {
            let e = sel[*i];
            members.push(e);
            let v = values[e];
            sum += v;
            abs.push(v.abs());
            excess += v.abs() - tolerance;
            if v.abs() > extreme.abs() {
                extreme = v;
                index = e;
            }
            let p = points[e];
            centroid += p / n;
            min = min.min(p);
            max = max.max(p);
        }
        abs.sort_by(f64::total_cmp);
        let half = list.len() / 2;
        let depth = if list.len() % 2 == 1 { abs[half] } else { (abs[half - 1] + abs[half]) / 2.0 };
        let area = n * spacing * spacing;
        let kind = patch_kind(area, depth, spacing, feature_depth);
        if kind == PatchKind::Feature {
            feature_count += 1;
            feature_excess += excess;
        }
        patches.push(Hotspot { count: list.len(), centroid, min, max, mean: sum / n, extreme, excess, area, volume: excess * spacing * spacing, depth, kind, at: points[index], index, members });
    }
    patches.sort_by(|a, b| b.excess.total_cmp(&a.excess));
    let found = patches.len();
    patches.truncate(options.limit);
    HotspotReport { patches, found, over: m, features_count: feature_count, features_share_of_excess: if all_excess > 0.0 { feature_excess / all_excess } else { 0.0 }, spacing, grain }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sheet of points 0.5 mm apart over `w` × `h` mm at `at`, reading `value` each.
    fn sheet(at: DVec3, w: usize, h: usize, value: f64, points: &mut Vec<DVec3>, values: &mut Vec<f64>) {
        for i in 0..w {
            for j in 0..h {
                points.push(at + DVec3::new(i as f64 * 0.5, j as f64 * 0.5, 0.0));
                values.push(value);
            }
        }
    }

    #[test]
    fn gathers_vertices_over_tolerance_into_patches_by_side_and_distance() {
        let (mut p, mut v) = (Vec::new(), Vec::new());
        sheet(DVec3::ZERO, 10, 10, 0.5, &mut p, &mut v); // a broad patch outside
        sheet(DVec3::new(20.0, 0.0, 0.0), 3, 3, -0.3, &mut p, &mut v); // a small one inside
        sheet(DVec3::new(40.0, 0.0, 0.0), 10, 10, 0.01, &mut p, &mut v); // within tolerance
        let r = hotspots(&p, &v, 0.1, &HotspotOptions::default());
        assert_eq!(r.found, 2);
        assert_eq!(r.over, 109);
        assert!((r.spacing - 0.5).abs() < 1e-9);
        let big = &r.patches[0];
        assert_eq!(big.count, 100);
        assert!((big.mean - 0.5).abs() < 1e-12);
        assert!((big.area - 25.0).abs() < 1e-9);
        assert_eq!(big.kind, PatchKind::Feature, "25 mm², 0.5 deep at 0.1: a feature");
        assert_eq!(r.patches[1].count, 9);
        assert_eq!(r.patches[1].kind, PatchKind::Texture);
        assert!(r.patches[1].extreme < 0.0);
        assert_eq!(r.features_count, 1);
    }

    #[test]
    fn a_group_keeps_entries_apart_and_small_patches_are_noise() {
        let (mut p, mut v) = (Vec::new(), Vec::new());
        sheet(DVec3::ZERO, 4, 4, 0.5, &mut p, &mut v);
        let group = |i: usize| (i % 2) as u64;
        let r = hotspots(&p, &v, 0.1, &HotspotOptions { group: Some(&group), ..Default::default() });
        assert_eq!(r.found, 2);
        let r = hotspots(&p[..2], &v[..2], 0.1, &HotspotOptions::default());
        assert_eq!(r.found, 0);
        assert_eq!(r.over, 2);
    }

    #[test]
    fn feature_or_texture() {
        assert_eq!(patch_kind(4.0, 0.2, 0.1, 0.2), PatchKind::Feature);
        assert_eq!(patch_kind(3.9, 0.2, 0.1, 0.2), PatchKind::Texture);
        assert_eq!(patch_kind(10.0, 0.19, 0.1, 0.2), PatchKind::Texture);
        // A coarse scan: 25 spacings² at 1 mm is 25 mm².
        assert_eq!(patch_kind(20.0, 1.0, 1.0, 0.2), PatchKind::Texture);
    }
}
