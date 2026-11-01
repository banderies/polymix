// SPDX-License-Identifier: AGPL-3.0-only
//! One slice chain fitted into a sketch the way the Fit options say, and a chain fitted onto
//! the pieces the sketch already holds. Ported from ScanRuler (`fitChain`, `fitChainShared` in
//! `plugins/reverse/state/reSketchDraft.ts`; `autoFit`, `refitChains` in
//! `state/reSketchSlice.ts`), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! A chain is cut into lines and arcs (`crate::segment`), or into a given number of arcs end to
//! end, or laid as one spline through a given number of fit points; lines near the sheet's axes
//! are laid along them before the corners are made (`crate::axis_align`), so the corners are
//! where the level lines cross, and the pieces of one wall are merged into the one line they
//! are; the corners are joined last. A chain fitted with a share lands on the pieces drawn
//! before it (`crate::share_fit`): what of it lies along one of them is that piece's, and the
//! rest is fitted in open runs whose ends land on the pieces they reach.

use glam::DVec2;

use crate::chain::view_chain;
use crate::corners::{infer_corners, place_segments, ChainFit, CornerReport, PlacedSegment, CORNER_GAP_MM};
use crate::section::{FitOptions, FittedChain};
use crate::segment::{arcs_of_chain, detail_levels, segment_chain, SegmentOptions};
use crate::share_fit::{land_end, share_chain, share_pieces, take_shared, Junction, RunReport, ShareReport, SharedSummary};
use crate::sketch::{Anchor, AnchorAt, Entity, Shape, Sketch};

/// One chain cut into pieces and put on the sketch the way the options say (ScanRuler's
/// `fitChain`). The caller solves afterwards. `Err` says why the chain could not be fitted.
pub fn fit_chain(sketch: &mut Sketch, chain: &[DVec2], tolerance: Option<f64>, options: &FitOptions) -> Result<ChainFit, String> {
    if let Some(count) = options.spline {
        return fit_spline_chain(sketch, chain, tolerance, count);
    }
    let cut = match options.arcs {
        Some(n) => arcs_of_chain(chain, n, tolerance),
        None => segment_chain(chain, SegmentOptions { tolerance, detail: Some(options.detail), ..Default::default() }),
    }
    .map_err(|e| e.0)?;
    let mut order = place_segments(sketch, &cut);
    // Lines are laid along the axes before the corners are made, so the corners are where the
    // level lines cross — and the pieces of one wall, level and in line, are made the one line
    // they are.
    if options.align_axes {
        let ids: Vec<_> = order.iter().map(|p| p.id).collect();
        crate::axis_align::align_to_axes(sketch, &ids, crate::axis_align::AxisAlignOptions { degrees: options.align_degrees, diagonals: options.align_diagonals });
        order = crate::axis_align::merge_aligned_lines(sketch, &order, cut.closed, 2.0 * detail_levels(cut.tolerance, 0.5).wave);
    }
    let corners = if options.join_corners { infer_corners(sketch, &order, cut.closed, CORNER_GAP_MM) } else { CornerReport::default() };
    let lines = order.iter().filter(|p| matches!(sketch.entity(p.id).map(|e| &e.shape), Some(Shape::Line { .. }))).count();
    let arcs = order.len() - lines;
    Ok(ChainFit { order, closed: cut.closed, corners, lines, arcs })
}

/// One spline along the whole chain, its points behind it: no corners to join, so the chain is
/// recorded as one open piece.
fn fit_spline_chain(sketch: &mut Sketch, chain: &[DVec2], tolerance: Option<f64>, count: usize) -> Result<ChainFit, String> {
    let view = view_chain(chain);
    let closed = view.closed;
    let points = view.points;
    let count = count.max(if closed { 3 } else { 2 });
    if points.len() < 2 * count {
        return Err(format!("The chain has {} points — too few for a spline through {count}, two a fit point at least.", points.len()));
    }
    let Some(pts) = crate::sketch_spline::fit_points_along(&points, count, closed) else {
        return Err("The chain has no length to lay a spline along.".into());
    };
    let scatter = tolerance.unwrap_or(view.scatter).max(1e-3);
    let shape = if closed { crate::sketch_spline::SplineShape::closed(pts) } else { crate::sketch_spline::SplineShape::open(pts) };
    let id = sketch.add(Entity::new(Shape::Spline(shape)).fitted(points, scatter));
    let (from, to) = if closed { (AnchorAt::P(0), AnchorAt::P(0)) } else { (AnchorAt::A, AnchorAt::B) };
    Ok(ChainFit { order: vec![PlacedSegment { id, from, to, join_next: false }], closed: false, corners: CornerReport::default(), lines: 0, arcs: 0 })
}

/// A chain fitted onto the pieces the sketch already holds (ScanRuler's `fitChainShared`): what
/// of it lies along one of `pieces` within `share` mm is that piece's — its points join a
/// fitted piece's — and the rest is fitted in open runs, each end landed on the piece it
/// reaches. A chain none of whose points are any piece's is fitted as `fit_chain` fits it.
/// What was shared and landed goes into `report`; the error of a run that did not fit is
/// returned, the others fitted all the same.
#[allow(clippy::too_many_arguments)]
pub fn fit_chain_shared(sketch: &mut Sketch, index: usize, chain: &[DVec2], tolerance: Option<f64>, options: &FitOptions, pieces: &[Entity], share: f64, report: &mut ShareReport) -> (Option<FittedChain>, Option<String>) {
    let carriers: Vec<&Entity> = pieces.iter().collect();
    let split = share_chain(chain, &carriers, share, None);
    if split.whole {
        return match fit_chain(sketch, chain, tolerance, options) {
            Ok(fit) => (Some(FittedChain { chain: index, order: fit.order, closed: fit.closed, ..Default::default() }), None),
            Err(e) => (None, Some(e)),
        };
    }
    take_shared(sketch, &split.shared);
    for s in &split.shared {
        report.shared.push(SharedSummary { piece: s.piece, points: s.points.len(), off_mean: s.off_mm.mean, off_max: s.off_mm.max });
    }
    report.corners += split.corners;
    let mut runs: Vec<Vec<PlacedSegment>> = Vec::new();
    let mut out: Vec<RunReport> = Vec::new();
    let mut error = None;
    for (k, r) in split.runs.iter().enumerate() {
        let fit = fit_chain(sketch, &r.points, tolerance, options);
        let order = match fit {
            Ok(f) if !f.order.is_empty() => f.order,
            Ok(_) => {
                out.push(RunReport { order: None, points: r.points.len(), skipped: Some("It did not fit.".into()), from: r.from, to: r.to });
                continue;
            }
            Err(why) => {
                out.push(RunReport { order: None, points: r.points.len(), skipped: Some(why.clone()), from: r.from, to: r.to });
                error = Some(why);
                continue;
            }
        };
        let (first, last) = (order[0], order[order.len() - 1]);
        for (end, on) in [(Anchor::new(first.id, first.from), r.from), (Anchor::new(last.id, last.to), r.to)] {
            let Some(on) = on else { continue };
            if let Some(landing) = land_end(sketch, end, on, share) {
                report.junctions.push(Junction { landing, chain: index, run: k });
            }
        }
        out.push(RunReport { order: Some(order.clone()), points: r.points.len(), skipped: None, from: r.from, to: r.to });
        runs.push(order);
    }
    report.chains.insert(index, out);
    (Some(FittedChain { chain: index, order: Vec::new(), closed: false, runs, share: Some(share), bridges: Vec::new() }), error)
}

/// The pieces a shared fit lands on: those the sketch held before the fit, not each other's —
/// taken as they stand.
pub fn share_carriers(sketch: &Sketch) -> Vec<Entity> {
    share_pieces(sketch).into_iter().cloned().collect()
}

/// What fitting several chains did, chain by chain.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FitChainsReport {
    pub fitted: Vec<FittedChain>,
    pub lines: usize,
    pub arcs: usize,
    pub splines: usize,
    pub joined: usize,
    pub tangent: usize,
    pub open: usize,
    /// Chains no piece could be fitted to, with why.
    pub failed: Vec<(usize, String)>,
    /// What a shared fit took and landed (only with a share).
    pub share: Option<ShareReport>,
}

/// Fit the chains listed (ScanRuler's `autoFit` over `targets`): with `share` (mm) and pieces
/// drawn before, each chain lands on them; otherwise each is fitted on its own. Chains already
/// fitted are the caller's to leave out. The caller solves, and lets go the landings the solve
/// cannot meet (`crate::share_fit::let_go`).
pub fn fit_chains(sketch: &mut Sketch, chains: &[Vec<DVec2>], targets: &[usize], options: &FitOptions, share: Option<f64>) -> FitChainsReport {
    let mut report = FitChainsReport::default();
    let pieces = match share {
        Some(s) if s > 0.0 => share_carriers(sketch),
        _ => Vec::new(),
    };
    let mut share_report = (!pieces.is_empty()).then(ShareReport::default);
    for &i in targets {
        let Some(chain) = chains.get(i) else { continue };
        let before = sketch.entities.len();
        let (fitted, error) = match (&mut share_report, share) {
            (Some(r), Some(s)) => fit_chain_shared(sketch, i, chain, options.tolerance, options, &pieces, s, r),
            _ => match fit_chain(sketch, chain, options.tolerance, options) {
                Ok(fit) => {
                    report.joined += fit.corners.joined;
                    report.tangent += fit.corners.tangent;
                    report.open += fit.corners.open;
                    (Some(FittedChain { chain: i, order: fit.order, closed: fit.closed, ..Default::default() }), None)
                }
                Err(e) => (None, Some(e)),
            },
        };
        for e in &sketch.entities[before.min(sketch.entities.len())..] {
            match e.shape {
                Shape::Line { .. } => report.lines += 1,
                Shape::Arc { .. } | Shape::Circle { .. } => report.arcs += 1,
                Shape::Spline(_) => report.splines += 1,
                Shape::Point { .. } => {}
            }
        }
        if let Some(why) = error {
            report.failed.push((i, why));
        }
        if let Some(f) = fitted {
            report.fitted.push(f);
        }
    }
    report.share = share_report;
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sketch::Rule;
    use crate::solver::{solve_sketch, SolveOptions};

    /// A closed outline through the corners, a point every `step` mm pushed off it by up to
    /// `noise` square to the side.
    fn outline(corners: &[[f64; 2]], step: f64, noise: f64) -> Vec<DVec2> {
        let mut s: i64 = 7;
        let mut rnd = move || {
            s = (s * 16807) % 2147483647;
            (s as f64 / 2147483647.0 - 0.5) * 2.0
        };
        let mut out: Vec<DVec2> = Vec::new();
        for (i, a) in corners.iter().enumerate() {
            let b = corners[(i + 1) % corners.len()];
            let len = (b[0] - a[0]).hypot(b[1] - a[1]);
            let n = [-(b[1] - a[1]) / len, (b[0] - a[0]) / len];
            let k = ((len / step) + 0.5).floor() as usize;
            for j in 0..k {
                let d = noise * rnd();
                out.push(DVec2::new(a[0] + ((b[0] - a[0]) * j as f64) / k as f64 + n[0] * d, a[1] + ((b[1] - a[1]) * j as f64) / k as f64 + n[1] * d));
            }
        }
        out.push(out[0]);
        out
    }

    #[test]
    fn a_slightly_turned_box_comes_out_level_and_upright_with_the_axes_option() {
        // Not a ScanRuler test: the options window's Align lines laid on a box 1° off the axes.
        let t = 1f64.to_radians();
        let turn = |x: f64, y: f64| [x * t.cos() - y * t.sin(), x * t.sin() + y * t.cos()];
        let chain = outline(&[turn(0.0, 0.0), turn(40.0, 0.0), turn(40.0, 20.0), turn(0.0, 20.0)], 0.25, 0.005);
        let mut s = Sketch::default();
        let options = FitOptions { align_axes: true, ..Default::default() };
        let fit = fit_chain(&mut s, &chain, None, &options).expect("fits");
        assert_eq!(fit.lines, 4);
        let held = s.constraints.iter().filter(|c| c.inferred && matches!(c.rule, Rule::Horizontal { .. } | Rule::Vertical { .. })).count();
        assert_eq!(held, 4);
        assert!(solve_sketch(&mut s, SolveOptions::default()).ok);
        // Without it, none is laid along an axis.
        let mut plain = Sketch::default();
        fit_chain(&mut plain, &chain, None, &FitOptions::default()).expect("fits");
        assert!(!plain.constraints.iter().any(|c| matches!(c.rule, Rule::Horizontal { .. } | Rule::Vertical { .. })));
    }

    #[test]
    fn a_chain_as_arcs_and_as_one_spline() {
        // Not a ScanRuler test: the agent's `arcs` and `spline` asks.
        let chain: Vec<DVec2> = (0..=400).map(|i| {
            let a = std::f64::consts::PI * i as f64 / 400.0;
            DVec2::new(20.0 * a.cos(), 10.0 * a.sin())
        }).collect();
        let mut s = Sketch::default();
        let fit = fit_chain(&mut s, &chain, None, &FitOptions { arcs: Some(3), ..Default::default() }).expect("arcs");
        assert_eq!((fit.arcs, fit.lines), (3, 0));
        let mut s = Sketch::default();
        let fit = fit_chain(&mut s, &chain, None, &FitOptions { spline: Some(6), ..Default::default() }).expect("spline");
        assert_eq!(fit.order.len(), 1);
        assert!(matches!(&s.entities[0].shape, Shape::Spline(sp) if sp.pts.len() == 6 && !sp.closed));
        let short = &chain[..8];
        let err = fit_chain(&mut Sketch::default(), short, None, &FitOptions { spline: Some(6), ..Default::default() }).unwrap_err();
        assert!(err.contains("too few for a spline through 6"), "{err}");
    }

    #[test]
    fn a_second_level_lands_on_the_walls_the_first_drew() {
        // Not a ScanRuler test: the deck and its wing, as share_fit's own test, through fit_chains.
        let deck = outline(&[[0.0, 0.0], [20.0, 0.0], [20.0, 10.0], [0.0, 10.0]], 0.2, 0.005);
        let mut s = Sketch::default();
        let first = fit_chains(&mut s, &[deck], &[0], &FitOptions::default(), None);
        assert_eq!(first.fitted.len(), 1);
        assert!(solve_sketch(&mut s, SolveOptions::default()).ok);
        let wing = outline(&[[0.03, 0.03], [30.0, 0.03], [30.0, 10.03], [0.03, 10.03]], 0.2, 0.005);
        let second = fit_chains(&mut s, &[wing], &[0], &FitOptions::default(), Some(0.1));
        let share = second.share.expect("shared");
        assert!(!share.shared.is_empty());
        assert_eq!(second.fitted.len(), 1);
        assert!(second.fitted[0].order.is_empty() && !second.fitted[0].runs.is_empty());
        assert!(!second.fitted[0].pieces().is_empty());
    }
}
