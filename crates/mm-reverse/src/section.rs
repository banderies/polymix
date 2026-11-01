// SPDX-License-Identifier: AGPL-3.0-only
//! From a scan and a sketch plane to fitted pieces: the slice the plane cuts through the scan,
//! laid flat on the sketch's sheet, and **Fit all chains** — every chain cut into lines and
//! arcs, their corners joined, the sketch solved onto the slice points. Ported from ScanRuler
//! (`autoFit` in `plugins/reverse/state/reSketchSlice.ts`, `fitChain` and `settle` in
//! `reSketchDraft.ts`, `fitOptions.ts`), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.

use glam::{DVec2, DVec3};
use serde::{Deserialize, Serialize};

use crate::corners::{PlacedSegment, CORNER_GAP_MM};
use crate::datums::datum_traces;
use crate::frame::SectionFrame;
use crate::slice::{slice_mesh, EDGE_MIN_FEATURE_MM};
use crate::sketch::Sketch;
use crate::solver::{solve_sketch, SolveOptions, SolveReport};

/// How Fit all chains cuts — the options window's settings, remembered.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FitOptions {
    /// From 0 (coarse: few, long pieces) to 1 (fine: every small feature its own piece).
    pub detail: f64,
    /// A typed fit tolerance, mm; `None` reads it off each chain's scatter.
    pub tolerance: Option<f64>,
    /// Join the corners the fit finds.
    pub join_corners: bool,
    /// Lay lines running near the sheet's axes along them (`crate::axis_align`), within
    /// `align_degrees`, and lines near 45° too with `align_diagonals`.
    #[serde(default)]
    pub align_axes: bool,
    #[serde(default)]
    pub align_diagonals: bool,
    #[serde(default = "default_align_degrees")]
    pub align_degrees: f64,
    /// Each chain as this many arcs end to end in place of the cut — an agent's ask for one
    /// fit, never stored.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arcs: Option<usize>,
    /// Each chain as one spline through this many fit points — the same kind of ask.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spline: Option<usize>,
}

fn default_align_degrees() -> f64 {
    3.0
}

impl Default for FitOptions {
    fn default() -> Self {
        Self { detail: 0.5, tolerance: None, join_corners: true, align_axes: false, align_diagonals: false, align_degrees: 3.0, arcs: None, spline: None }
    }
}

impl FitOptions {
    /// The options made a valid set (ScanRuler's `sanitizeFitOptions`): the detail within 0–1,
    /// the alignment's reach within 0.5–10°.
    pub fn sanitized(mut self) -> Self {
        let clamp = |v: f64, lo: f64, hi: f64, fallback: f64| if v.is_finite() { v.clamp(lo, hi) } else { fallback };
        self.detail = clamp(self.detail, 0.0, 1.0, 0.5);
        self.align_degrees = clamp(self.align_degrees, 0.5, 10.0, 3.0);
        self
    }
}

/// The chains a plane cuts through a scan, laid flat on the frame's sheet, each with a point
/// at least every so often along it (see [`densify`]).
pub fn section_chains(positions: &[DVec3], triangles: &[[u32; 3]], frame: &SectionFrame) -> Vec<Vec<DVec2>> {
    slice_mesh(positions, triangles, frame.origin, frame.normal, EDGE_MIN_FEATURE_MM).project(frame).into_iter().filter(|c| !is_sliver(c)).map(|c| densify(&c)).collect()
}

/// A loop with no inside: the plane running along a face of the mesh (a mesh from CAD has
/// exact planes, and a section can land on one) cuts out along an edge and back the same way.
/// Its mean width — twice its area over its length — is under a tenth of a millimetre.
pub fn is_sliver(chain: &[DVec2]) -> bool {
    let n = chain.len();
    if n < 4 || (chain[0] - chain[n - 1]).length() > 1e-6 {
        return false;
    }
    let length: f64 = chain.windows(2).map(|w| (w[1] - w[0]).length()).sum();
    let area = chain.windows(2).map(|w| w[0].x * w[1].y - w[1].x * w[0].y).sum::<f64>().abs() / 2.0;
    length > 0.0 && 2.0 * area / length < 0.1
}

/// A chain with points added along its long stretches, so none is longer than the chain's
/// typical spacing. Not in ScanRuler, which slices dense scans with a point every fraction of
/// a millimetre: a mesh exported from CAD (or decimated) has long thin triangles, and a flat
/// wall twenty millimetres long cuts into three points — fewer than a piece needs, so the
/// segmentation folds the wall into its neighbour and a line is fitted round a corner. The
/// points added lie on the chords the mesh itself has there, so they say nothing the mesh did
/// not; a dense scan, every stretch already short, comes back as it was.
pub fn densify(chain: &[DVec2]) -> Vec<DVec2> {
    if chain.len() < 3 {
        return chain.to_vec();
    }
    let mut gaps: Vec<f64> = chain.windows(2).map(|w| (w[1] - w[0]).length()).filter(|g| *g > 0.0).collect();
    if gaps.is_empty() {
        return chain.to_vec();
    }
    gaps.sort_by(|a, b| a.total_cmp(b));
    let step = gaps[gaps.len() / 2].clamp(0.02, 0.5);
    let mut out = Vec::with_capacity(chain.len());
    for w in chain.windows(2) {
        out.push(w[0]);
        let l = (w[1] - w[0]).length();
        if l > 1.5 * step {
            let n = (l / step).round() as usize;
            for k in 1..n {
                out.push(w[0] + (w[1] - w[0]) * (k as f64 / n as f64));
            }
        }
    }
    out.push(chain[chain.len() - 1]);
    out
}

/// One chain as fitted: which pieces it became, in order — what Infer corners and Close walk
/// again later.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct FittedChain {
    pub chain: usize,
    pub order: Vec<PlacedSegment>,
    pub closed: bool,
    /// A chain fitted onto the pieces the sketch already held (`crate::sketch_fit`): its open
    /// runs, each fitted on its own, and the share it was fitted with — a refit lands on the
    /// pieces again.
    pub runs: Vec<Vec<PlacedSegment>>,
    pub share: Option<f64>,
    /// The bridges Close drew across this chain's gaps, which go with it on a refit.
    pub bridges: Vec<crate::sketch::EntityId>,
}

impl FittedChain {
    /// Every piece the fit made, the runs' and the bridges' too.
    pub fn pieces(&self) -> Vec<crate::sketch::EntityId> {
        let mut ids: Vec<crate::sketch::EntityId> = self.order.iter().map(|p| p.id).collect();
        ids.extend(self.runs.iter().flatten().map(|p| p.id));
        ids.extend(self.bridges.iter().copied());
        ids
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct FitAllReport {
    pub chains: usize,
    pub fitted: Vec<FittedChain>,
    pub lines: usize,
    pub arcs: usize,
    pub joined: usize,
    pub tangent: usize,
    pub open: usize,
    /// Chains no piece could be fitted to, with why.
    pub failed: Vec<(usize, String)>,
    pub solve: SolveReport,
}

/// Fit every chain into the sketch and solve it. Chains already fitted (by index in
/// `skip`) are left as they are.
pub fn fit_all_chains(sketch: &mut Sketch, chains: &[Vec<DVec2>], frame: &SectionFrame, options: FitOptions, skip: &[usize]) -> FitAllReport {
    let mut report = FitAllReport { chains: chains.len(), ..Default::default() };
    for (i, chain) in chains.iter().enumerate() {
        if skip.contains(&i) {
            continue;
        }
        match crate::sketch_fit::fit_chain(sketch, chain, options.tolerance, &options) {
            Ok(fit) => {
                report.lines += fit.lines;
                report.arcs += fit.arcs;
                report.joined += fit.corners.joined;
                report.tangent += fit.corners.tangent;
                report.open += fit.corners.open;
                report.fitted.push(FittedChain { chain: i, order: fit.order, closed: fit.closed, ..Default::default() });
            }
            Err(e) => report.failed.push((i, e)),
        }
    }
    let traces = datum_traces(frame);
    report.solve = solve_sketch(sketch, SolveOptions { datums: &traces, ..Default::default() });
    report
}

/// Join the corners of chains fitted with joining off, or of pieces drawn since.
pub fn infer_all_corners(sketch: &mut Sketch, fitted: &[FittedChain]) -> usize {
    fitted.iter().map(|f| crate::corners::infer_corners(sketch, &f.order, f.closed, CORNER_GAP_MM).joined).sum()
}

/// Bridge whatever gaps are left along every fitted chain.
pub fn close_all(sketch: &mut Sketch, fitted: &[FittedChain]) -> usize {
    fitted.iter().map(|f| crate::corners::close_chain(sketch, &f.order, CORNER_GAP_MM)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::WorldAxis;
    use crate::model::PlaneRef;
    use crate::regions::find_regions;
    use crate::sketch::Shape;

    /// A rounded box mesh, as a scan of one would be.
    fn rounded_box() -> (Vec<DVec3>, Vec<[u32; 3]>) {
        // A 60×40×20 box with its vertical edges rounded R5, faceted finely.
        let mut ring = Vec::new();
        let (w, h, r) = (30.0, 20.0, 5.0);
        let corners = [(w - r, -h + r, -90.0), (w - r, h - r, 0.0), (-w + r, h - r, 90.0), (-w + r, -h + r, 180.0)];
        for (cx, cy, a0) in corners {
            for k in 0..=24 {
                let a = (a0 + 90.0 * k as f64 / 24.0f64).to_radians();
                ring.push(DVec2::new(cx + r * a.cos(), cy + r * a.sin()));
            }
        }
        let n = ring.len();
        let mut pos = Vec::new();
        let levels = 10;
        for l in 0..=levels {
            let z = 20.0 * l as f64 / levels as f64;
            pos.extend(ring.iter().map(|p| DVec3::new(p.x, p.y, z)));
        }
        let mut tris = Vec::new();
        for l in 0..levels {
            for i in 0..n {
                let (a, b) = ((l * n + i) as u32, (l * n + (i + 1) % n) as u32);
                let (c, d) = (a + n as u32, b + n as u32);
                tris.push([a, b, d]);
                tris.push([a, d, c]);
            }
        }
        // Caps as fans about a centre point.
        for (z, flip) in [(0.0, true), (20.0, false)] {
            let c = pos.len() as u32;
            pos.push(DVec3::new(0.0, 0.0, z));
            let base = if z == 0.0 { 0 } else { (levels * n) as u32 };
            for i in 0..n {
                let (a, b) = (base + i as u32, base + ((i + 1) % n) as u32);
                tris.push(if flip { [c, b, a] } else { [c, a, b] });
            }
        }
        (pos, tris)
    }

    #[test]
    fn a_loop_out_and_back_along_an_edge_is_a_sliver() {
        let mut out_and_back: Vec<DVec2> = (0..=40).map(|i| DVec2::new(i as f64, 0.0)).collect();
        out_and_back.extend((0..=40).rev().map(|i| DVec2::new(i as f64, 0.03)));
        out_and_back.push(out_and_back[0]);
        assert!(is_sliver(&out_and_back));
        let mut square: Vec<DVec2> = (0..4).map(|i| [DVec2::ZERO, DVec2::new(5.0, 0.0), DVec2::new(5.0, 0.5), DVec2::new(0.0, 0.5)][i]).collect();
        square.push(square[0]);
        assert!(!is_sliver(&square), "a thin wall half a millimetre thick is no sliver");
    }

    #[test]
    fn densify_fills_long_stretches_and_leaves_dense_chains_alone() {
        let sparse = [DVec2::ZERO, DVec2::new(0.3, 0.0), DVec2::new(0.6, 0.0), DVec2::new(10.6, 0.0), DVec2::new(10.6, 0.3)];
        let d = densify(&sparse);
        assert!(d.windows(2).all(|w| (w[1] - w[0]).length() <= 0.45 + 1e-9), "{d:?}");
        assert_eq!(d.first(), sparse.first());
        assert_eq!(d.last(), sparse.last());
        let dense: Vec<DVec2> = (0..50).map(|i| DVec2::new(i as f64 * 0.2, 0.0)).collect();
        assert_eq!(densify(&dense), dense);
    }

    #[test]
    fn fits_a_section_through_a_rounded_box_into_one_closed_region() {
        let (pos, tris) = rounded_box();
        let frame = PlaneRef::world(WorldAxis::Z, 10.0).own_frame().unwrap();
        let chains = section_chains(&pos, &tris, &frame);
        assert_eq!(chains.len(), 1);
        let mut s = Sketch::default();
        let report = fit_all_chains(&mut s, &chains, &frame, FitOptions::default(), &[]);
        assert!(report.solve.ok, "{report:?}");
        assert_eq!((report.lines, report.arcs), (4, 4), "{report:?}");
        assert_eq!(report.open, 0);
        let regions = find_regions(&s);
        assert_eq!(regions.len(), 1);
        let want = 60.0 * 40.0 - (4.0 - std::f64::consts::PI) * 25.0;
        assert!((regions[0].area - want).abs() < 2.0, "area {} vs {want}", regions[0].area);
        for e in &s.entities {
            if let Shape::Arc { r, .. } = e.shape {
                assert!((r - 5.0).abs() < 0.1, "fillet radius {r}");
            }
        }
    }
}
