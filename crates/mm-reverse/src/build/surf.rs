// SPDX-License-Identifier: AGPL-3.0-only
//! What the surface features (Sheet, Split, Thicken, Fit surface, Trim, Extract of a shadow)
//! share in the build: the scan the build reads — a split and a trim test their faces against
//! it, a fitted surface is fitted through it, a shadow is taken off it — the state they keep
//! beside the bodies (a fitted surface's fit, a plane or a cylinder it was fitted as, whether a
//! sheet was trimmed or joined since), and what they say of themselves beside a body: a fit's
//! σ, a trim's faces, a split's cut face off the scan, the wall a thicken came to. Ported from
//! ScanRuler (`plugins/reverse/core/build.ts` setScan, onScanShare, nearScan and the builder's
//! `fitted` and body fields; `buildTypes.ts` SurfaceReport, TrimReport, OffScanReport,
//! SPLIT_SAMPLES, halton), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use glam::DVec3;
use mm_core::bvh::Bvh;
use mm_core::Mesh;

use super::{play_range, Builder, Built};
use crate::auto_constrain::FoundRelation;
use crate::model::{BodyId, FeatureId, Part, SurfaceShape, TrimPrimitive};
use crate::surface_fit::{Stray, SurfaceFitResult, WorstPoint};

// ---- the scan ----

/// The scan as the build reads it, in part coordinates: its vertices by the numbers a fitted
/// surface's region names them by, and its triangles. What a split and a trim test their faces
/// against, what a surface is fitted through, what a shadow is taken off. Its search structure
/// is made the first time it is asked for and kept, for every build on this scan.
pub struct BuildScan {
    positions: Vec<DVec3>,
    triangles: Vec<[u32; 3]>,
    index: OnceLock<Option<(Mesh, Bvh)>>,
    bounds: OnceLock<Option<(DVec3, DVec3)>>,
}

impl BuildScan {
    /// A scan of these vertices and triangles; a triangle naming a vertex there is none of is
    /// left out.
    pub fn new(positions: Vec<DVec3>, triangles: Vec<[u32; 3]>) -> BuildScan {
        let n = positions.len();
        let triangles = triangles.into_iter().filter(|t| t.iter().all(|v| (*v as usize) < n)).collect();
        BuildScan { positions, triangles, index: OnceLock::new(), bounds: OnceLock::new() }
    }

    /// The vertices, by number (a number no triangle names holds whatever it was given).
    pub fn positions(&self) -> &[DVec3] {
        &self.positions
    }

    pub fn triangles(&self) -> &[[u32; 3]] {
        &self.triangles
    }

    fn index(&self) -> Option<&(Mesh, Bvh)> {
        self.index
            .get_or_init(|| {
                if self.triangles.is_empty() {
                    return None;
                }
                let (mesh, _) = Mesh::from_triangles(&self.positions, &self.triangles, None);
                let bvh = Bvh::build(&mesh);
                Some((mesh, bvh))
            })
            .as_ref()
    }

    /// The scan's nearest point to `p` within `reach`, or `None` (ScanRuler's `scan.closest`).
    pub fn closest(&self, p: DVec3, reach: f64) -> Option<DVec3> {
        let (mesh, bvh) = self.index()?;
        bvh.nearest(mesh, p, reach).map(|h| h.point)
    }

    /// The box round the vertices the triangles use, or `None` for an empty scan.
    pub fn bounds(&self) -> Option<(DVec3, DVec3)> {
        *self.bounds.get_or_init(|| {
            let mut lo = DVec3::splat(f64::INFINITY);
            let mut hi = DVec3::splat(f64::NEG_INFINITY);
            for t in &self.triangles {
                for v in t {
                    if let Some(p) = self.positions.get(*v as usize).filter(|p| p.is_finite()) {
                        lo = lo.min(*p);
                        hi = hi.max(*p);
                    }
                }
            }
            lo.x.is_finite().then_some((lo, hi))
        })
    }
}

// ---- what the surface features keep beside the bodies ----

/// What became of a sheet since it was made: trimmed (split, or cut or intersected by a
/// solid), or joined to another — once joined, never again the one surface it was fitted as.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Reshaped {
    Trimmed,
    Joined,
}

/// The surface features' state in a build, beside the bodies: the scan, each fitted surface's
/// fit by the body it made (a split carries it on from it, a split within cuts inside its
/// region), the plane or the cylinder a surface was fitted as (a later fit is held to it, a
/// trim cuts with the whole of it), and the sheets reshaped since.
#[derive(Clone, Default)]
pub(crate) struct SurfState {
    pub(crate) scan: Option<Arc<BuildScan>>,
    pub(crate) fits: HashMap<BodyId, Arc<SurfaceFitResult>>,
    pub(crate) fitted: HashMap<BodyId, TrimPrimitive>,
    pub(crate) reshaped: HashMap<BodyId, Reshaped>,
    /// A sketch sheet's chains, by the body it made: what it cuts as.
    pub(crate) chains: HashMap<BodyId, Arc<super::surf_split::SheetChains>>,
}

// ---- what they say of themselves ----

/// A Fit Surface's reading of itself (ScanRuler's `SurfaceReport`): the residuals along the
/// normal and the net that was used, Auto's choice included; for a plane or a cylinder what it
/// was held to and the free fit's σ before; a freeform fit's worst points, how many points a
/// robust fit left out, its strays, and how far its margin stands off the scan.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SurfaceFitReport {
    pub rms: f64,
    pub max: f64,
    /// The net, U × V (0 × 0 for a plane or a cylinder).
    pub net: (usize, usize),
    pub points: usize,
    /// A plane or a cylinder rather than the freeform surface.
    pub shape: Option<SurfaceShape>,
    pub rms_free: Option<f64>,
    pub relations: Vec<FoundRelation>,
    pub radius: Option<f64>,
    pub worst: Vec<WorstPoint>,
    pub dropped: Option<usize>,
    pub strays: Vec<Stray>,
    /// How far the surface stands off the scan past its region, the most, and where.
    pub past_region: Option<(f64, DVec3)>,
}

/// A trim's reading of itself (ScanRuler's `TrimReport`): how many faces the surfaces cut the
/// box into, how many lie on the scan and were kept, whether they close, and the open edges
/// where they do not — drawn red on the part.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TrimReport {
    pub faces: usize,
    pub kept: usize,
    pub closed: bool,
    pub open_edges: Vec<Vec<DVec3>>,
}

/// How much of a split's cut face lies off the scan past a fitted tool's region: the tool, the
/// most, mm, the area further off than the split's tolerance, mm², and where the most is.
#[derive(Debug, Clone, PartialEq)]
pub struct OffScanReport {
    pub tool: String,
    pub max: f64,
    pub area: f64,
    pub at: DVec3,
}

/// What a surface feature says of itself beside its body, built or failed: a note on its row
/// (the wall a thicken came to), its fit, its trim, its cut face off the scan, the strays a
/// refused fit names, the lines to draw red on the part (a face that folds, a trim's open
/// edges), and how many faces a sheet has.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SurfReport {
    pub note: Option<String>,
    pub surface: Option<SurfaceFitReport>,
    pub trim: Option<TrimReport>,
    pub off_scan: Vec<OffScanReport>,
    pub strays: Vec<Stray>,
    pub where_lines: Vec<Vec<DVec3>>,
    pub sheet_faces: Option<usize>,
}

impl Builder {
    /// The scan the build reads, or why there is none for `what`.
    pub(crate) fn scan_for(&self, what: &str) -> Result<Arc<BuildScan>, String> {
        self.surf.scan.clone().ok_or_else(|| format!("The scan is not here to {what}."))
    }

    /// The step's report, made the first time it is asked for.
    pub(crate) fn report(&mut self) -> &mut SurfReport {
        self.out.surf.entry(self.step).or_default()
    }
}

// ---- building with the scan ----

/// The part built over a scan: every feature in order, up to the roll-back bar. Without one, a
/// feature that reads the scan fails, saying so.
pub fn build_scanned(part: &Part, scan: Option<Arc<BuildScan>>) -> Built {
    let upto = part.rolled_to.unwrap_or(part.features.len()).min(part.features.len());
    let mut b = Builder::new();
    b.surf.scan = scan;
    play_range(&mut b, part, 0, upto);
    b.finish(part)
}

/// As [`super::build_before`], over a scan: the part as it stood before one of its features,
/// and the whole part.
pub fn build_before_scanned(part: &Part, feature: FeatureId, scan: Option<Arc<BuildScan>>) -> (Built, Built) {
    let upto = part.rolled_to.unwrap_or(part.features.len()).min(part.features.len());
    let at = part.index_of(feature).map_or(upto, |i| i.min(upto));
    let mut b = Builder::new();
    b.surf.scan = scan;
    play_range(&mut b, part, 0, at);
    // Each body read once, for both builds.
    for (_, s) in &b.bodies.raw {
        s.topology();
    }
    // The picks are taken on the bodies with no thread's faceted cut made yet (`thread::flush`).
    let mut pick = b.clone();
    pick.threads.clear();
    let before = pick.finish(part);
    play_range(&mut b, part, at, upto);
    (before, b.finish(part))
}

// ---- sampling a shape against the scan ----

/// How many points a fragment's or a face's surface is sampled at against the scan.
pub(crate) const SPLIT_SAMPLES: usize = 1500;

/// The i-th point of the van der Corput sequence in a base — evenly spread over 0…1 for any
/// number of them, and the same every time (ScanRuler's `halton`).
pub(crate) fn halton(mut i: usize, base: usize) -> f64 {
    let mut f = 1.0;
    let mut r = 0.0;
    while i > 0 {
        f /= base as f64;
        r += f * (i % base) as f64;
        i /= base;
    }
    r
}

/// Points spread over triangles by area (ScanRuler's sampling in `onScanShare`): about
/// `SPLIT_SAMPLES` in all, each triangle given points in proportion to its area — the Halton
/// points of bases 2 and 3 folded into it — each standing for that share of the triangle. Every
/// `every`-th triangle is taken. Each point with the area it stands for and its triangle.
pub(crate) fn area_samples(positions: &[DVec3], triangles: &[[u32; 3]], every: usize) -> Vec<(DVec3, f64, usize)> {
    let every = every.max(1);
    let tri = |t: &[u32; 3]| t.map(|v| positions.get(v as usize).copied().unwrap_or_default());
    let mut areas = vec![0.0f64; triangles.len()];
    let mut whole = 0.0;
    for (k, t) in triangles.iter().enumerate().step_by(every) {
        let [a, b, c] = tri(t);
        areas[k] = (b - a).cross(c - a).length() / 2.0;
        whole += areas[k];
    }
    let mut out = Vec::new();
    if !(whole > 0.0) {
        return out;
    }
    for (k, t) in triangles.iter().enumerate().step_by(every) {
        if !(areas[k] > 0.0) {
            continue;
        }
        let [a, b, c] = tri(t);
        let count = ((SPLIT_SAMPLES as f64 * areas[k] / whole).round() as usize).max(1);
        for i in 0..count {
            let (mut u, mut v) = if count == 1 { (1.0 / 3.0, 1.0 / 3.0) } else { (halton(i + 1, 2), halton(i + 1, 3)) };
            if u + v > 1.0 {
                u = 1.0 - u;
                v = 1.0 - v;
            }
            let w = 1.0 - u - v;
            out.push((a * w + b * u + c * v, areas[k] / count as f64, k));
        }
    }
    out
}

/// The share of a mesh's surface lying within `tolerance` of the scan, by area (ScanRuler's
/// `onScanShare`): about `SPLIT_SAMPLES` points spread over it, each triangle given points in
/// proportion to its area. By area and not by triangle: a flat face is two triangles however
/// large, and the plug of a pocket — four walls on the scan, a roof and a floor where the scan
/// has nothing — counted eight triangles on and four off. What `leave_out` leaves out counts
/// neither way.
pub(crate) fn on_scan_share(scan: &BuildScan, positions: &[DVec3], triangles: &[[u32; 3]], tolerance: f64, leave_out: Option<&dyn Fn(DVec3) -> bool>) -> f64 {
    let every = (triangles.len() / SPLIT_SAMPLES).max(1);
    let (mut on, mut read) = (0.0, 0.0);
    for (p, area, _) in area_samples(positions, triangles, every) {
        if leave_out.is_some_and(|f| f(p)) {
            continue;
        }
        read += area;
        if scan.closest(p, tolerance).is_some() {
            on += area;
        }
    }
    if read > 0.0 {
        on / read
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spreads_halton_points_as_the_van_der_corput_sequence() {
        assert_eq!(halton(1, 2), 0.5);
        assert_eq!(halton(2, 2), 0.25);
        assert_eq!(halton(3, 2), 0.75);
        assert!((halton(1, 3) - 1.0 / 3.0).abs() < 1e-15);
        assert!((halton(4, 3) - (1.0 / 3.0 + 1.0 / 9.0)).abs() < 1e-15);
    }

    #[test]
    fn reads_the_share_of_a_mesh_on_the_scan_by_area() {
        // The scan: a 10 × 10 square at z = 0. The mesh: that square (on it) and one as large
        // at z = 5 (off it) — half on by area.
        let scan = BuildScan::new(vec![DVec3::new(0.0, 0.0, 0.0), DVec3::new(10.0, 0.0, 0.0), DVec3::new(10.0, 10.0, 0.0), DVec3::new(0.0, 10.0, 0.0)], vec![[0, 1, 2], [0, 2, 3]]);
        let positions = vec![
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(10.0, 0.0, 0.0),
            DVec3::new(10.0, 10.0, 0.0),
            DVec3::new(0.0, 10.0, 0.0),
            DVec3::new(0.0, 0.0, 5.0),
            DVec3::new(10.0, 0.0, 5.0),
            DVec3::new(10.0, 10.0, 5.0),
            DVec3::new(0.0, 10.0, 5.0),
        ];
        let triangles = vec![[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]];
        let share = on_scan_share(&scan, &positions, &triangles, 0.1, None);
        assert!((share - 0.5).abs() < 0.01, "{share}");
        // What is left out counts neither way: the top left out, all of it is on.
        let share = on_scan_share(&scan, &positions, &triangles, 0.1, Some(&|p: DVec3| p.z > 1.0));
        assert!((share - 1.0).abs() < 1e-9, "{share}");
        assert_eq!(scan.bounds(), Some((DVec3::ZERO, DVec3::new(10.0, 10.0, 0.0))));
        assert!(scan.closest(DVec3::new(5.0, 5.0, 0.05), 0.1).is_some());
        assert!(scan.closest(DVec3::new(5.0, 5.0, 0.5), 0.1).is_none());
    }
}
