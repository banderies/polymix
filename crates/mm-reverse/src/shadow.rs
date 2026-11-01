// SPDX-License-Identifier: AGPL-3.0-only
//! The shadow of the scan on a plane: the whole part seen along the plane's normal, kept as
//! its outer boundary — the outline a straight-walled bin or a foam cutout needs, every
//! indentation filled, where a cut is only what one plane passes through. Ported from
//! ScanRuler (`src/core/section/shadow.ts`), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! The scan's triangles are projected onto the plane and rasterised into a mask (a cell is set
//! when its centre lies inside a projected triangle; a triangle seen edge-on sets nothing, so
//! a sheet edge-on has no shadow). Marching squares traces the mask's loops, which gives the
//! topology — which loops there are, which are holes — and roughly where. Every traced point
//! lies between a set centre and an unset one, and is then bisected along that step onto the
//! true boundary of the union of projected triangles: the exact loops are the mesh's own
//! silhouette to floating-point accuracy whatever the cell, and the cell only decides which
//! features are resolved (a slot narrower than a cell can close).
//!
//! The exact loops are the reference, not what the section keeps. What it keeps is a contour
//! read as lines and arcs, the way a sketch is drawn, and never inside the part: the exact
//! silhouette offset outwards by the play plus the tolerance (an exact rounded offset — the
//! distance to the exact loops, traced and bisected the same way), read with `loop_shape` at
//! the tolerance, so the reading lies between play and play + 2·tolerance outside. A
//! least-squares piece can stray past what it was asked for, so the reading is checked against
//! the exact loops, sampled piece by piece. Where it comes a little closer than the play all
//! round, the whole offset is raised by the shortfall and the loops read again — a uniform
//! margin, so lines still meet at corners — as long as its far side stays in the band. What
//! still leaves the band is a piece read wrongly (two lines meeting far past a blunt corner,
//! an arc forced through ends off its circle), and that piece alone becomes the stretch of
//! offset loop it stood for, thinned; a loop that still strays is kept as its offset loop
//! thinned to half the tolerance, which lies in the band by construction. A negative play is a
//! press fit and keeps the same guarantee against the shrunk silhouette.
//!
//! A reading that stays in the band can still carry teeth: a corner read as two with a stub of
//! a line between them, turning out and back — inside the band, and plain wrong to the eye. A
//! short line whose ends turn opposite ways is taken out where its neighbours, run on to meet,
//! or one line in their place, stay in the band.
//!
//! Last, the corners: where two lines meet turning by less than the smoothing angle (30°
//! unless told otherwise), the corner is rounded by an arc tangent to both, as large as the two
//! lines leave room for and the band allows — a faceted run of a curved side becomes one the
//! eye and a CAD kernel read as smooth; a corner turning more stays sharp.
//!
//! The pieces are what the DXF, the SVG and the STEP prism are written from; sampled finely
//! they are also an ordinary `SectionCut` — closed loops with the outward direction at every
//! point — so the sheet, the 3D view and everything that reads a cut reads a shadow the same
//! way.

use std::collections::HashMap;
use std::f64::consts::{PI, TAU};

use glam::{DVec2, DVec3};
use serde::{Deserialize, Serialize};

use crate::fit2d::{ArcFit, CircleFit, LineFit};
use crate::frame::SectionFrame;
use crate::js_fmt::{grouped, js_round, to_fixed};
use crate::outline::{loop_shape, mask_loops, LoopShape, MaskGrid, MaskLoop};
use crate::simplify::simplify_run;
use crate::slice::SectionCut;

/// The tolerance a shadow is read to unless told otherwise, mm. The raster cell is half of it.
pub const SHADOW_TOLERANCE: f64 = 0.1;

/// Corners of a shadow's outline turning by less than this are rounded unless told otherwise,
/// degrees; sharper ones stay sharp.
pub const SHADOW_SMOOTH_DEG: f64 = 30.0;

/// A corner turning by less than this is no corner to round, radians.
const MIN_TURN: f64 = 0.5 * PI / 180.0;

/// The most cells the mask may have: a bigger part widens the cell.
const MAX_CELLS: usize = 8_000_000;

/// How close the bisection gets to the boundary, mm.
const REFINE_TO: f64 = 1e-5;

/// The exact loops are thinned to this before they are measured against — well under anything
/// the reading is judged on, and enough to fold a straight run of refined points into one
/// segment.
const REFERENCE_THIN: f64 = 1e-3;

/// How many times the offset is raised when the reading comes too close.
const MAX_RAISES: usize = 3;

/// A line this many tolerances long or shorter, turning back the way it came at one end, is a
/// tooth to pull.
const TOOTH_LENGTH: f64 = 6.0;

/// The chord error an arc is sampled to, mm.
const ARC_SAGITTA: f64 = 0.002;

/// A piece of a shadow's contour on the sheet ([u, v] mm in the section's frame), in the order
/// the loop runs: the material lies to the left. An arc turns by `sweep` radians about its
/// centre, positive counter-clockwise; a circle is a whole loop on its own.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ShadowPiece {
    Line { from: DVec2, to: DVec2 },
    Arc { from: DVec2, to: DVec2, center: DVec2, radius: f64, sweep: f64 },
    Circle { center: DVec2, radius: f64 },
}

impl ShadowPiece {
    pub fn is_line(&self) -> bool {
        matches!(self, ShadowPiece::Line { .. })
    }
    pub fn is_arc(&self) -> bool {
        matches!(self, ShadowPiece::Arc { .. })
    }
    pub fn is_circle(&self) -> bool {
        matches!(self, ShadowPiece::Circle { .. })
    }
}

/// A loop of a shadow's contour.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShadowLoop {
    /// A hole in the outline (run clockwise seen from the normal's side); otherwise an outer
    /// loop (counter-clockwise).
    pub hole: bool,
    pub pieces: Vec<ShadowPiece>,
}

/// Nearest and farthest, mm.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Span {
    pub min: f64,
    pub max: f64,
}

/// What a shadow is, beside the polylines: its contour as pieces, and the figures the readout
/// and the exports give.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShadowShape {
    pub loops: Vec<ShadowLoop>,
    /// The area the contour encloses, holes taken off, mm².
    pub area: f64,
    pub play: f64,
    pub tolerance: f64,
    /// Corners turning by less than this were rounded, degrees; 0 none.
    pub smooth_deg: f64,
    /// How far the contour lies outside the exact silhouette, nearest and farthest, as
    /// measured — never less than the play.
    pub margin: Span,
    /// The raster's cell, mm — tolerance / 2, or wider for a big part.
    pub cell: f64,
    pub widened: bool,
    /// Where the part lies along the normal, from the frame's origin, mm: what an extruded
    /// shadow stands between.
    pub extent: Span,
    /// The frame the shadow was taken in, moved with the part — so a shadow slid along its
    /// normal can be shifted rather than taken again.
    pub frame: SectionFrame,
}

/// How a shadow is taken.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct ShadowOptions {
    /// mm; SHADOW_TOLERANCE by default.
    pub tolerance: Option<f64>,
    /// How far outside the part the contour lies, mm: positive is a clearance, negative a
    /// press fit. 0 by default.
    pub play: Option<f64>,
    /// Corners where the outline turns by less than this many degrees are rounded, sharper
    /// ones kept; 0 rounds none. SHADOW_SMOOTH_DEG by default.
    pub smooth_deg: Option<f64>,
    /// The most raster cells; the cell is widened past it.
    pub max_cells: Option<usize>,
}

/// A loop of the exact silhouette, in [u, v] mm.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExactLoop {
    pub points: Vec<DVec2>,
    pub hole: bool,
}

/// A shadow as a section: its contour sampled as an ordinary cut, and its shape.
#[derive(Debug, Clone)]
pub struct ShadowCut {
    pub cut: SectionCut,
    pub shadow: ShadowShape,
}

/// A shadow with the exact loops it was read from.
#[derive(Debug, Clone)]
pub struct ShadowOutline {
    /// The kernel's truth: the silhouette's loops, refined, unthinned.
    pub exact: Vec<ExactLoop>,
    /// What the section keeps.
    pub shape: ShadowShape,
    pub cut: ShadowCut,
}

/// The scan's shadow on the frame's plane, as a cut — see the top.
pub fn shadow_mesh(positions: &[DVec3], triangles: &[[u32; 3]], frame: &SectionFrame, opts: &ShadowOptions) -> ShadowCut {
    shadow_outline(positions, triangles, frame, opts).cut
}

/// The shadow with the exact loops it was read from — what the tests check the reading
/// against.
pub fn shadow_outline(positions: &[DVec3], triangles: &[[u32; 3]], frame: &SectionFrame, opts: &ShadowOptions) -> ShadowOutline {
    let tol = opts.tolerance.filter(|t| *t > 0.0).unwrap_or(SHADOW_TOLERANCE);
    let play = opts.play.filter(|p| p.is_finite()).unwrap_or(0.0);
    let smooth_deg = opts.smooth_deg.filter(|s| *s >= 0.0).map_or(SHADOW_SMOOTH_DEG, |s| s.min(180.0));

    // ---- 1. project ---------------------------------------------------------
    let nv = positions.len();
    let mut pu = vec![0.0; nv];
    let mut pv = vec![0.0; nv];
    let (mut w_min, mut w_max) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut u_min, mut u_max) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut v_min, mut v_max) = (f64::INFINITY, f64::NEG_INFINITY);
    for (i, p) in positions.iter().enumerate() {
        let d = *p - frame.origin;
        let (u, v, w) = (d.dot(frame.u), d.dot(frame.v), d.dot(frame.normal));
        pu[i] = u;
        pv[i] = v;
        u_min = u_min.min(u);
        u_max = u_max.max(u);
        v_min = v_min.min(v);
        v_max = v_max.max(v);
        w_min = w_min.min(w);
        w_max = w_max.max(w);
    }
    let blank = |cell: f64, widened: bool| -> ShadowOutline {
        let shape = ShadowShape {
            loops: vec![],
            area: 0.0,
            play,
            tolerance: tol,
            smooth_deg,
            margin: Span { min: play, max: play },
            cell,
            widened,
            extent: Span { min: if w_min.is_finite() { w_min } else { 0.0 }, max: if w_max.is_finite() { w_max } else { 0.0 } },
            frame: *frame,
        };
        ShadowOutline { exact: vec![], cut: ShadowCut { cut: SectionCut::default(), shadow: shape.clone() }, shape }
    };
    // A triangle naming a vertex that is not there is left out, as one seen edge-on is.
    let tris: Vec<[usize; 3]> = triangles.iter().map(|t| [t[0] as usize, t[1] as usize, t[2] as usize]).filter(|t| t.iter().all(|&i| i < nv)).collect();
    if nv == 0 || tris.is_empty() {
        return blank(tol / 2.0, false);
    }

    // ---- 2. rasterise -------------------------------------------------------
    let mut cell = tol / 2.0;
    let mut widened = false;
    let max_cells = opts.max_cells.unwrap_or(MAX_CELLS) as f64;
    let span_u = u_max - u_min;
    let span_v = v_max - v_min;
    if (span_u / cell + 5.0) * (span_v / cell + 5.0) > max_cells {
        cell = (((span_u + 5.0 * cell) * (span_v + 5.0 * cell)) / max_cells).sqrt() * 1.01;
        widened = true;
    }
    let grid = MaskGrid { cell, u0: u_min - 2.0 * cell, v0: v_min - 2.0 * cell, cols: (span_u / cell).ceil() as usize + 5, rows: (span_v / cell).ceil() as usize + 5 };
    let mut mask = vec![0u8; grid.cols * grid.rows];
    for t in &tris {
        fill_triangle(&mut mask, &grid, DVec2::new(pu[t[0]], pv[t[0]]), DVec2::new(pu[t[1]], pv[t[1]]), DVec2::new(pu[t[2]], pv[t[2]]));
    }

    // ---- 3. trace -----------------------------------------------------------
    let traced = mask_loops(&mask, &grid, (3.0 * cell).powi(2));
    if traced.is_empty() {
        return blank(cell, widened);
    }

    // ---- 4. refine onto the union of the projected triangles ----------------
    let union = TriangleBuckets::new(&pu, &pv, &tris, &grid, &traced);
    let exact: Vec<ExactLoop> = traced.iter().map(|l| ExactLoop { hole: l.hole, points: l.points.iter().map(|p| refine_crossing(*p, &grid, &mask, &|u, v| union.inside(u, v))).collect() }).collect();

    // ---- 5. the contour: offset, read, checked ------------------------------
    let reference = SegmentField::new(&exact, cell);
    // The band the contour must lie in, outside the exact silhouette.
    let low = play - 1e-6;
    let high = play + 2.0 * tol + 1e-6;
    let mut d = play + tol;
    let (base, base_cell) = offset_loops(&reference, &exact, d, cell);
    let base_loops: Vec<ShadowLoop> = base.iter().map(|l| ShadowLoop { hole: l.hole, pieces: read_loop(&l.points, tol, base_cell) }).collect();
    let mut loops = base_loops.clone();
    let mut margin = measure_margin(&loops, &reference, cell);
    // A reading that comes a little too close all round is raised whole, so its lines still
    // meet at their corners — as long as that keeps its far side inside the band.
    let mut round = 0;
    while round < MAX_RAISES && margin.min < low && margin.max + (play - margin.min) < high {
        d += play - margin.min + 1e-4;
        let (offset, offset_cell) = offset_loops(&reference, &exact, d, cell);
        loops = offset.iter().map(|l| ShadowLoop { hole: l.hole, pieces: read_loop(&l.points, tol, offset_cell) }).collect();
        margin = measure_margin(&loops, &reference, cell);
        round += 1;
    }
    // What still leaves the band is a piece read wrongly — two lines meeting far past a blunt
    // corner, an arc forced through ends off its circle — and only that piece is drawn instead
    // as the offset loop it stands for, thinned; a loop that still leaves the band is kept as
    // its offset loop thinned to half the tolerance, which lies within it by construction — all
    // of it on the offset of play + tolerance, not a raised one.
    let out_of_band = |m: Span| m.min < low || m.max > high;
    if out_of_band(margin) {
        loops = base_loops
            .iter()
            .enumerate()
            .map(|(i, l)| {
                let run = &base[i].points;
                let bad: Vec<bool> = l.pieces.iter().map(|pc| out_of_band(measure_margin(&[ShadowLoop { hole: l.hole, pieces: vec![*pc] }], &reference, cell))).collect();
                let n = l.pieces.len();
                let first = bad.iter().position(|b| !b);
                let mut pieces: Vec<ShadowPiece> = Vec::new();
                if let Some(first) = first {
                    // Round from a good piece, so that bad ones in a row — two lines that meet
                    // past a corner — are patched as one stretch, the corner between them gone.
                    let mut k = 0;
                    while k < n {
                        let at = (first + k) % n;
                        if !bad[at] {
                            pieces.push(l.pieces[at]);
                            k += 1;
                            continue;
                        }
                        let mut end = k;
                        while end + 1 < n && bad[(first + end + 1) % n] {
                            end += 1;
                        }
                        let from = l.pieces[at];
                        let to = l.pieces[(first + end) % n];
                        if let (Some((from_start, _)), Some((_, to_end))) = (piece_ends(&from), piece_ends(&to)) {
                            pieces.extend(patch_piece(&ShadowPiece::Line { from: from_start, to: to_end }, run, tol / 2.0));
                        }
                        k = end + 1;
                    }
                }
                let m = first.map(|_| measure_margin(&[ShadowLoop { hole: l.hole, pieces: pieces.clone() }], &reference, cell));
                if m.is_none_or(out_of_band) {
                    pieces = polygon_pieces(run, tol / 2.0);
                }
                ShadowLoop { hole: l.hole, pieces }
            })
            .collect();
    }
    loops = loops.into_iter().filter(|l| !l.pieces.is_empty()).map(orient).collect();

    // ---- 6. the teeth, then the corners ---------------------------------------
    let band = |pieces: &[ShadowPiece]| -> bool {
        let m = measure_margin(&[ShadowLoop { hole: false, pieces: pieces.to_vec() }], &reference, cell);
        m.min >= low && m.max <= high
    };
    loops = loops.into_iter().map(|l| ShadowLoop { hole: l.hole, pieces: pull_teeth(&l.pieces, TOOTH_LENGTH * tol, &band) }).collect();
    if smooth_deg > 0.0 {
        loops = loops.into_iter().map(|l| ShadowLoop { hole: l.hole, pieces: round_corners(&l.pieces, smooth_deg.to_radians(), &band) }).collect();
    }
    margin = measure_margin(&loops, &reference, cell);

    // ---- 7. the cut: the pieces sampled, with outward normals, lifted -------
    let shape = ShadowShape {
        area: loops.iter().map(|l| if l.hole { -1.0 } else { 1.0 } * loop_area(&l.pieces).abs()).sum(),
        loops,
        play,
        tolerance: tol,
        smooth_deg,
        margin,
        cell,
        widened,
        extent: Span { min: w_min, max: w_max },
        frame: *frame,
    };
    let cut = lift_loops(&shape.loops, frame, cell);
    ShadowOutline { exact, cut: ShadowCut { cut, shadow: shape.clone() }, shape }
}

/// A line's or an arc's two ends, start first; a circle has none.
fn piece_ends(p: &ShadowPiece) -> Option<(DVec2, DVec2)> {
    match *p {
        ShadowPiece::Line { from, to } | ShadowPiece::Arc { from, to, .. } => Some((from, to)),
        ShadowPiece::Circle { .. } => None,
    }
}

/// The shadow slid along its own normal: the same loops drawn on the plane at its new place.
/// `None` when the frame has turned, so the shadow has to be taken again.
pub fn shift_shadow(cut: &ShadowCut, to: &SectionFrame) -> Option<ShadowCut> {
    let sh = &cut.shadow;
    let from = &sh.frame;
    let same = |a: DVec3, b: DVec3| (a.x - b.x).abs() < 1e-9 && (a.y - b.y).abs() < 1e-9 && (a.z - b.z).abs() < 1e-9;
    if !same(from.normal, to.normal) || !same(from.u, to.u) || !same(from.v, to.v) {
        return None;
    }
    let delta = to.origin - from.origin;
    let along = delta.dot(to.normal);
    // A move that is not along the normal moves the sheet's zero: the loops would have to be
    // laid out again.
    if (delta - to.normal * along).length() > 1e-9 {
        return None;
    }
    let chains = cut.cut.chains.iter().map(|c| c.iter().map(|p| *p + delta).collect()).collect();
    Some(ShadowCut {
        cut: SectionCut { chains, normals: cut.cut.normals.clone() },
        shadow: ShadowShape { frame: *to, extent: Span { min: sh.extent.min - along, max: sh.extent.max - along }, ..sh.clone() },
    })
}

// ---- the raster ------------------------------------------------------------

/// Set every cell whose centre lies inside the triangle (edges included), row by row. A
/// triangle seen edge-on sets nothing.
fn fill_triangle(mask: &mut [u8], g: &MaskGrid, a: DVec2, b: DVec2, c: DVec2) {
    let area2 = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
    if area2 == 0.0 {
        return;
    }
    let y0 = a.y.min(b.y).min(c.y);
    let y1 = a.y.max(b.y).max(c.y);
    let r0 = ((y0 - g.v0) / g.cell - 0.5).ceil().max(0.0) as i64;
    let r1 = ((y1 - g.v0) / g.cell - 0.5).floor().min(g.rows as f64 - 1.0) as i64;
    for r in r0..=r1 {
        let y = g.v0 + (r as f64 + 0.5) * g.cell;
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        let mut edge = |p: DVec2, q: DVec2| {
            if (p.y <= y && y <= q.y) || (q.y <= y && y <= p.y) {
                if p.y == q.y {
                    lo = lo.min(p.x).min(q.x);
                    hi = hi.max(p.x).max(q.x);
                } else {
                    let x = p.x + ((y - p.y) * (q.x - p.x)) / (q.y - p.y);
                    lo = lo.min(x);
                    hi = hi.max(x);
                }
            }
        };
        edge(a, b);
        edge(b, c);
        edge(c, a);
        if !(hi >= lo) {
            continue;
        }
        let c0 = ((lo - g.u0) / g.cell - 0.5).ceil().max(0.0) as i64;
        let c1 = ((hi - g.u0) / g.cell - 0.5).floor().min(g.cols as f64 - 1.0) as i64;
        let row = r as usize * g.cols;
        for col in c0..=c1 {
            mask[row + col as usize] = 1;
        }
    }
}

/// The two cell centres a traced point lies between, the set one first — marching squares
/// puts every point halfway along such a step.
fn crossing_cells(p: DVec2, g: &MaskGrid, mask: &[u8]) -> (DVec2, DVec2) {
    let fx = (p.x - g.u0) / g.cell - 0.5;
    let fy = (p.y - g.v0) / g.cell - 0.5;
    let horizontal = (fx - js_round(fx)).abs() > 0.25;
    let c = if horizontal { fx.floor() } else { js_round(fx) } as i64;
    let r = if horizontal { js_round(fy) } else { fy.floor() } as i64;
    let c2 = if horizontal { c + 1 } else { c };
    let r2 = if horizontal { r } else { r + 1 };
    let set = |cc: i64, rr: i64| cc >= 0 && rr >= 0 && cc < g.cols as i64 && rr < g.rows as i64 && mask[(rr * g.cols as i64 + cc) as usize] != 0;
    let centre = |cc: i64, rr: i64| DVec2::new(g.u0 + (cc as f64 + 0.5) * g.cell, g.v0 + (rr as f64 + 0.5) * g.cell);
    if set(c, r) {
        (centre(c, r), centre(c2, r2))
    } else {
        (centre(c2, r2), centre(c, r))
    }
}

/// A traced point moved onto the boundary of `inside`, by bisection along the step it lies
/// on.
fn refine_crossing(p: DVec2, g: &MaskGrid, mask: &[u8], inside: &dyn Fn(f64, f64) -> bool) -> DVec2 {
    let (mut l, mut h) = crossing_cells(p, g, mask);
    let mut k = 0;
    while k < 60 && (h - l).length() > REFINE_TO {
        let m = (l + h) / 2.0;
        if inside(m.x, m.y) {
            l = m;
        } else {
            h = m;
        }
        k += 1;
    }
    (l + h) / 2.0
}

/// The projected triangles near the traced boundary, bucketed, and the test of whether a point
/// lies inside any of them. Only the buckets the boundary passes through are filled — a
/// refinement never asks elsewhere.
struct TriangleBuckets<'a> {
    pu: &'a [f64],
    pv: &'a [f64],
    tris: &'a [[usize; 3]],
    u0: f64,
    v0: f64,
    size: f64,
    bc: i64,
    br: i64,
    slot: Vec<i64>,
    count: Vec<usize>,
    items: Vec<usize>,
}

impl<'a> TriangleBuckets<'a> {
    fn new(pu: &'a [f64], pv: &'a [f64], tris: &'a [[usize; 3]], g: &MaskGrid, traced: &[MaskLoop]) -> Self {
        let w = g.cols as f64 * g.cell;
        let h = g.rows as f64 * g.cell;
        let size = 1f64.max(4.0 * g.cell).max(((w * h) / 1e6).sqrt());
        let bc = (w / size).ceil() as i64 + 1;
        let br = (h / size).ceil() as i64 + 1;
        let mut slot = vec![-1i64; (bc * br) as usize];
        let mut used = 0usize;
        let bucket_of = |u: f64, v: f64| (((u - g.u0) / size).floor() as i64, ((v - g.v0) / size).floor() as i64);
        for l in traced {
            for p in &l.points {
                let (c, r) = bucket_of(p.x, p.y);
                for dr in -1..=1 {
                    for dc in -1..=1 {
                        let (cc, rr) = (c + dc, r + dr);
                        if cc < 0 || rr < 0 || cc >= bc || rr >= br {
                            continue;
                        }
                        let k = (rr * bc + cc) as usize;
                        if slot[k] < 0 {
                            slot[k] = used as i64;
                            used += 1;
                        }
                    }
                }
            }
        }
        let each = |f: &mut dyn FnMut(usize, usize)| {
            for (t, tri) in tris.iter().enumerate() {
                let [a, b, c] = *tri;
                let c0 = ((pu[a].min(pu[b]).min(pu[c]) - g.u0) / size).floor().max(0.0) as i64;
                let c1 = ((pu[a].max(pu[b]).max(pu[c]) - g.u0) / size).floor().min(bc as f64 - 1.0) as i64;
                let r0 = ((pv[a].min(pv[b]).min(pv[c]) - g.v0) / size).floor().max(0.0) as i64;
                let r1 = ((pv[a].max(pv[b]).max(pv[c]) - g.v0) / size).floor().min(br as f64 - 1.0) as i64;
                for r in r0..=r1 {
                    for cc in c0..=c1 {
                        let s = slot[(r * bc + cc) as usize];
                        if s >= 0 {
                            f(s as usize, t);
                        }
                    }
                }
            }
        };
        let mut count = vec![0usize; used + 1];
        each(&mut |s, _| count[s + 1] += 1);
        for i in 1..=used {
            count[i] += count[i - 1];
        }
        let mut fill: Vec<usize> = count[..used].to_vec();
        let mut items = vec![0usize; count[used]];
        each(&mut |s, t| {
            items[fill[s]] = t;
            fill[s] += 1;
        });
        TriangleBuckets { pu, pv, tris, u0: g.u0, v0: g.v0, size, bc, br, slot, count, items }
    }

    fn inside(&self, u: f64, v: f64) -> bool {
        let c = ((u - self.u0) / self.size).floor() as i64;
        let r = ((v - self.v0) / self.size).floor() as i64;
        if c < 0 || r < 0 || c >= self.bc || r >= self.br {
            return false;
        }
        let s = self.slot[(r * self.bc + c) as usize];
        if s < 0 {
            return false;
        }
        let s = s as usize;
        for &t in &self.items[self.count[s]..self.count[s + 1]] {
            let [a, b, cc] = self.tris[t];
            let (ax, ay, bx, by, cx, cy) = (self.pu[a], self.pv[a], self.pu[b], self.pv[b], self.pu[cc], self.pv[cc]);
            let d1 = (bx - ax) * (v - ay) - (by - ay) * (u - ax);
            let d2 = (cx - bx) * (v - by) - (cy - by) * (u - bx);
            let d3 = (ax - cx) * (v - cy) - (ay - cy) * (u - cx);
            let area2 = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
            if area2 == 0.0 {
                continue;
            }
            if if area2 > 0.0 { d1 >= 0.0 && d2 >= 0.0 && d3 >= 0.0 } else { d1 <= 0.0 && d2 <= 0.0 && d3 <= 0.0 } {
                return true;
            }
        }
        false
    }
}

// ---- distance to the exact loops ------------------------------------------

/// The exact loops as segments, bucketed — for the distance from a point to the silhouette,
/// and (by a ray along +u, even-odd over the same buckets' rows) whether it lies inside. Long
/// segments are split so that a bucket holds only what passes through it.
struct SegmentField {
    size: f64,
    u0: f64,
    v0: f64,
    bc: i64,
    br: i64,
    /// ax, ay, bx, by per segment.
    seg: Vec<[f64; 4]>,
    start: Vec<usize>,
    items: Vec<usize>,
    /// Per bucket row, the segments crossing that row's band of v.
    row_start: Vec<usize>,
    row_items: Vec<usize>,
    /// The box round the loops: u0, v0, u1, v1.
    bx: [f64; 4],
}

impl SegmentField {
    fn new(loops: &[ExactLoop], cell: f64) -> Self {
        let mut segs: Vec<[f64; 4]> = Vec::new();
        let (mut u0, mut v0, mut u1, mut v1) = (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
        for l in loops {
            let n = l.points.len();
            if n < 2 {
                continue;
            }
            let flat: Vec<DVec2> = (0..=n).map(|i| l.points[i % n]).collect();
            let kept = simplify_run(&flat, 0, n + 1, REFERENCE_THIN);
            for k in 1..kept.len() {
                let (a, b) = (flat[kept[k - 1]], flat[kept[k]]);
                segs.push([a.x, a.y, b.x, b.y]);
                u0 = u0.min(a.x).min(b.x);
                v0 = v0.min(a.y).min(b.y);
                u1 = u1.max(a.x).max(b.x);
                v1 = v1.max(a.y).max(b.y);
            }
        }
        if segs.is_empty() {
            (u0, v0, u1, v1) = (0.0, 0.0, 0.0, 0.0);
        }
        let w = (u1 - u0).max(cell);
        let h = (v1 - v0).max(cell);
        let size = (4.0 * cell).max(0.5).max(((w * h) / 250_000.0).sqrt());
        // Split for bucketing: no piece longer than a bucket.
        let mut split: Vec<[f64; 4]> = Vec::new();
        for s in &segs {
            let [ax, ay, bx, by] = *s;
            let k = ((bx - ax).hypot(by - ay) / size).ceil().max(1.0) as usize;
            for j in 0..k {
                let (t0, t1) = (j as f64 / k as f64, (j + 1) as f64 / k as f64);
                split.push([ax + (bx - ax) * t0, ay + (by - ay) * t0, ax + (bx - ax) * t1, ay + (by - ay) * t1]);
            }
        }
        let mut field = SegmentField {
            size,
            u0: u0 - size,
            v0: v0 - size,
            bc: (w / size).ceil() as i64 + 3,
            br: (h / size).ceil() as i64 + 3,
            seg: split,
            start: vec![],
            items: vec![],
            row_start: vec![],
            row_items: vec![],
            bx: [u0, v0, u1, v1],
        };
        let n = field.seg.len();
        let cells = (field.bc * field.br) as usize;
        let cells_of = |f: &SegmentField, s: usize| -> Vec<usize> {
            let [ax, ay, bx, by] = f.seg[s];
            let (c0, c1) = (f.col(ax.min(bx)), f.col(ax.max(bx)));
            let (r0, r1) = (f.row(ay.min(by)), f.row(ay.max(by)));
            let mut out = Vec::new();
            for r in r0..=r1 {
                for c in c0..=c1 {
                    out.push((r * f.bc + c) as usize);
                }
            }
            out
        };
        let mut count = vec![0usize; cells + 1];
        let all_cells: Vec<Vec<usize>> = (0..n).map(|s| cells_of(&field, s)).collect();
        for ks in &all_cells {
            for &k in ks {
                count[k + 1] += 1;
            }
        }
        for i in 1..count.len() {
            count[i] += count[i - 1];
        }
        let mut fill: Vec<usize> = count[..cells].to_vec();
        let mut items = vec![0usize; count[cells]];
        for (s, ks) in all_cells.iter().enumerate() {
            for &k in ks {
                items[fill[k]] = s;
                fill[k] += 1;
            }
        }
        let br = field.br as usize;
        let mut rows = vec![0usize; br + 1];
        let rows_of = |f: &SegmentField, s: usize| (f.row(f.seg[s][1].min(f.seg[s][3])), f.row(f.seg[s][1].max(f.seg[s][3])));
        for s in 0..n {
            let (r0, r1) = rows_of(&field, s);
            for r in r0..=r1 {
                rows[r as usize + 1] += 1;
            }
        }
        for i in 1..rows.len() {
            rows[i] += rows[i - 1];
        }
        let mut rfill: Vec<usize> = rows[..br].to_vec();
        let mut row_items = vec![0usize; rows[br]];
        for s in 0..n {
            let (r0, r1) = rows_of(&field, s);
            for r in r0..=r1 {
                row_items[rfill[r as usize]] = s;
                rfill[r as usize] += 1;
            }
        }
        field.start = count;
        field.items = items;
        field.row_start = rows;
        field.row_items = row_items;
        field
    }

    fn col(&self, u: f64) -> i64 {
        (((u - self.u0) / self.size).floor() as i64).clamp(0, self.bc - 1)
    }

    fn row(&self, v: f64) -> i64 {
        (((v - self.v0) / self.size).floor() as i64).clamp(0, self.br - 1)
    }

    fn seg_dist(&self, s: usize, u: f64, v: f64) -> f64 {
        let [ax, ay, bx, by] = self.seg[s];
        let (dx, dy) = (bx - ax, by - ay);
        let l2 = dx * dx + dy * dy;
        let t = if l2 > 0.0 { (((u - ax) * dx + (v - ay) * dy) / l2).clamp(0.0, 1.0) } else { 0.0 };
        (u - ax - t * dx).hypot(v - ay - t * dy)
    }

    /// Inside the exact silhouette: a ray along +u crosses its loops an odd number of times.
    fn inside(&self, u: f64, v: f64) -> bool {
        let [bu0, bv0, bu1, bv1] = self.bx;
        if v < bv0 || v > bv1 || u < bu0 || u > bu1 {
            return false;
        }
        let r = self.row(v) as usize;
        let mut odd = false;
        for &s in &self.row_items[self.row_start[r]..self.row_start[r + 1]] {
            let [ax, ay, bx, by] = self.seg[s];
            if (ay > v) == (by > v) {
                continue;
            }
            if u < ax + ((v - ay) * (bx - ax)) / (by - ay) {
                odd = !odd;
            }
        }
        odd
    }

    /// Whether any of the silhouette lies within r of the point.
    fn within(&self, u: f64, v: f64, r: f64) -> bool {
        let (c0, c1) = (self.col(u - r), self.col(u + r));
        let (r0, r1) = (self.row(v - r), self.row(v + r));
        for rr in r0..=r1 {
            for c in c0..=c1 {
                let k = (rr * self.bc + c) as usize;
                if self.items[self.start[k]..self.start[k + 1]].iter().any(|&s| self.seg_dist(s, u, v) <= r) {
                    return true;
                }
            }
        }
        false
    }

    /// The distance to the silhouette, searched ring by ring outwards.
    fn distance(&self, u: f64, v: f64) -> f64 {
        let c = ((u - self.u0) / self.size).floor() as i64;
        let r = ((v - self.v0) / self.size).floor() as i64;
        let mut best = f64::INFINITY;
        let max_ring = self.bc.max(self.br) + c.abs() + r.abs();
        for ring in 0..=max_ring {
            // Everything in a farther ring is at least (ring − 1) buckets away.
            if best <= (ring - 1) as f64 * self.size {
                break;
            }
            for rr in r - ring..=r + ring {
                if rr < 0 || rr >= self.br {
                    continue;
                }
                let edge = rr == r - ring || rr == r + ring;
                let mut cc = c - ring;
                while cc <= c + ring {
                    if cc >= 0 && cc < self.bc {
                        let k = (rr * self.bc + cc) as usize;
                        for &s in &self.items[self.start[k]..self.start[k + 1]] {
                            best = best.min(self.seg_dist(s, u, v));
                        }
                    }
                    if ring == 0 {
                        break;
                    }
                    cc += if edge { 1 } else { 2 * ring };
                }
            }
        }
        best
    }

    /// The distance, negative inside.
    fn signed(&self, u: f64, v: f64) -> f64 {
        let d = self.distance(u, v);
        if self.inside(u, v) {
            -d
        } else {
            d
        }
    }
}

// ---- the offset ------------------------------------------------------------

/// The loops of the silhouette offset by d (outwards for d > 0): the points whose signed
/// distance to the exact loops is d — rounded at convex corners by d, as a real clearance is.
/// Rasterised on a grid of its own (coarser for a big offset, whose loops have no detail finer
/// than it), traced, and bisected onto that distance. With the cell it was traced at.
fn offset_loops(reference: &SegmentField, exact: &[ExactLoop], d: f64, base_cell: f64) -> (Vec<ExactLoop>, f64) {
    if d.abs() < 1e-9 {
        return (exact.to_vec(), base_cell);
    }
    let cell = base_cell.max(d.abs() / 8.0);
    let pad = d.max(0.0) + 3.0 * cell;
    let [bu0, bv0, bu1, bv1] = reference.bx;
    let g = MaskGrid { cell, u0: bu0 - pad, v0: bv0 - pad, cols: ((bu1 - bu0 + 2.0 * pad) / cell).ceil() as usize + 1, rows: ((bv1 - bv0 + 2.0 * pad) / cell).ceil() as usize + 1 };
    let mut mask = vec![0u8; g.cols * g.rows];
    // Inside the silhouette, row by row: even-odd over the exact loops.
    let mut xs: Vec<f64> = Vec::new();
    for r in 0..g.rows {
        let y = g.v0 + (r as f64 + 0.5) * cell;
        xs.clear();
        for l in exact {
            let n = l.points.len();
            let mut j = n.wrapping_sub(1);
            for i in 0..n {
                let (a, b) = (l.points[j], l.points[i]);
                if (a.y > y) != (b.y > y) {
                    xs.push(a.x + ((y - a.y) * (b.x - a.x)) / (b.y - a.y));
                }
                j = i;
            }
        }
        xs.sort_by(|p, q| p.total_cmp(q));
        let mut k = 0;
        while k + 1 < xs.len() {
            let c0 = ((xs[k] - g.u0) / cell - 0.5).ceil().max(0.0) as i64;
            let c1 = ((xs[k + 1] - g.u0) / cell - 0.5).floor().min(g.cols as f64 - 1.0) as i64;
            for c in c0..=c1 {
                mask[r * g.cols + c as usize] = 1;
            }
            k += 2;
        }
    }
    // The band within |d| of the silhouette flips: outside cells near enough join (d > 0),
    // inside cells near enough leave (d < 0).
    let reach = d.abs();
    let mut seen = vec![false; g.cols * g.rows];
    let mut flips: Vec<usize> = Vec::new();
    for l in exact {
        let n = l.points.len();
        for i in 0..n {
            let (a, b) = (l.points[i], l.points[(i + 1) % n]);
            let c0 = ((a.x.min(b.x) - reach - g.u0) / cell - 0.5).floor().max(0.0) as i64;
            let c1 = ((a.x.max(b.x) + reach - g.u0) / cell - 0.5).ceil().min(g.cols as f64 - 1.0) as i64;
            let r0 = ((a.y.min(b.y) - reach - g.v0) / cell - 0.5).floor().max(0.0) as i64;
            let r1 = ((a.y.max(b.y) + reach - g.v0) / cell - 0.5).ceil().min(g.rows as f64 - 1.0) as i64;
            for r in r0..=r1 {
                for c in c0..=c1 {
                    let k = r as usize * g.cols + c as usize;
                    if seen[k] {
                        continue;
                    }
                    seen[k] = true;
                    let set = mask[k] != 0;
                    if set == (d > 0.0) {
                        continue;
                    }
                    if reference.within(g.u0 + (c as f64 + 0.5) * cell, g.v0 + (r as f64 + 0.5) * cell, reach) {
                        flips.push(k);
                    }
                }
            }
        }
    }
    for k in flips {
        mask[k] = u8::from(d > 0.0);
    }
    let inside = |u: f64, v: f64| if d > 0.0 { reference.inside(u, v) || reference.within(u, v, reach) } else { reference.inside(u, v) && !reference.within(u, v, reach) };
    let loops = mask_loops(&mask, &g, (3.0 * cell).powi(2)).into_iter().map(|l| ExactLoop { hole: l.hole, points: l.points.iter().map(|p| refine_crossing(*p, &g, &mask, &inside)).collect() }).collect();
    (loops, cell)
}

// ---- reading a loop as pieces -----------------------------------------------

/// A loop read as lines and arcs at the tolerance — `loop_shape`'s reading made into pieces
/// that meet end to end exactly.
fn read_loop(points: &[DVec2], tolerance: f64, cell: f64) -> Vec<ShadowPiece> {
    if points.len() < 3 {
        return vec![];
    }
    let shape = loop_shape(points, tolerance, cell);
    shape_pieces(&shape, points, tolerance)
}

/// Where on the line from a to b the foot of p lies, as a share of a → b.
fn along(p: DVec2, a: DVec2, b: DVec2) -> f64 {
    let d = b - a;
    let l2 = d.dot(d);
    if l2 > 0.0 {
        (p - a).dot(d) / l2
    } else {
        0.0
    }
}

/// The arc from a to b nearest the circle fitted (centre c0, radius r0): its centre on the
/// bisector of a and b, at the fitted radius where the chord allows, so that it starts and
/// ends exactly where its neighbours do; and of the two ways round, the one that passes nearer
/// `via` — or, without one, the shorter. `None` where a and b coincide.
fn arc_through(a: DVec2, b: DVec2, c0: DVec2, r0: f64, via: Option<DVec2>) -> Option<ShadowPiece> {
    let chord = (a - b).length();
    if chord < 1e-9 {
        return None;
    }
    let r = r0.max(chord / 2.0);
    let m = (a + b) / 2.0;
    let nrm = DVec2::new(-(b.y - a.y) / chord, (b.x - a.x) / chord);
    let h = (r * r - (chord / 2.0).powi(2)).max(0.0).sqrt();
    let p = m + nrm * h;
    let q = m - nrm * h;
    let center = if (p - c0).length() <= (q - c0).length() { p } else { q };
    let a0 = (a.y - center.y).atan2(a.x - center.x);
    let a1 = (b.y - center.y).atan2(b.x - center.x);
    let mut short = a1 - a0;
    while short > PI {
        short -= TAU;
    }
    while short <= -PI {
        short += TAU;
    }
    let long = if short > 0.0 { short - TAU } else { short + TAU };
    let mut sweep = short;
    if let Some(via) = via {
        let at = |s: f64| center + DVec2::new((a0 + s / 2.0).cos(), (a0 + s / 2.0).sin()) * r;
        if (at(long) - via).length() < (at(short) - via).length() {
            sweep = long;
        }
    }
    Some(ShadowPiece::Arc { from: a, to: b, center, radius: r, sweep })
}

/// `loop_shape`'s corners, rounds and arcs as pieces end to end: a round's two lines stop
/// where they touch it, and every arc runs exactly between the ends of its neighbours.
fn shape_pieces(shape: &LoopShape, traced: &[DVec2], tolerance: f64) -> Vec<ShadowPiece> {
    if let Some(c) = shape.circle {
        return vec![ShadowPiece::Circle { center: c.center, radius: c.radius }];
    }
    let pc = shape.points.len();
    if pc < 3 {
        return vec![];
    }
    let pts = &shape.points;
    let at = |i: isize| pts[(((i % pc as isize) + pc as isize) % pc as isize) as usize];
    let round_at: HashMap<usize, (f64, DVec2)> = shape.rounds.iter().map(|r| (r.at, (r.radius, r.center))).collect();
    let arc_from: HashMap<usize, (f64, DVec2)> = shape.arcs.iter().map(|a| (a.from, (a.radius, a.center))).collect();
    // Where each corner's sides start and end: the corner, or for a round the feet of its
    // centre on the two lines.
    let mut enter: Vec<DVec2> = pts.clone();
    let mut leave: Vec<DVec2> = pts.clone();
    let mut rounded = vec![false; pc];
    let mut fillets: HashMap<usize, DVec2> = HashMap::new();
    for r in &shape.rounds {
        let i = r.at;
        // The fillet of the fitted radius between the two lines: its centre on the corner's
        // bisector, so it touches both exactly.
        let ii = i as isize;
        let c = at(ii);
        let l1 = (c - at(ii - 1)).length();
        let l2 = (at(ii + 1) - c).length();
        if !(l1 > 0.0 && l2 > 0.0) {
            continue;
        }
        let d1 = (c - at(ii - 1)) / l1;
        let d2 = (at(ii + 1) - c) / l2;
        let half = (-(d1.dot(d2))).clamp(-1.0, 1.0).acos() / 2.0;
        if !(half > 1e-3 && half < PI / 2.0 - 1e-3) {
            continue;
        }
        let back = r.radius / half.tan();
        if !(back < l1 && back < l2) {
            continue;
        }
        let bis = d2 - d1;
        let bl = bis.length();
        let reach = r.radius / half.sin();
        enter[i] = c - d1 * back;
        leave[i] = c + d2 * back;
        fillets.insert(i, c + bis / bl * reach);
        rounded[i] = true;
    }
    // Two rounds that overlap along the line between them: neither is kept.
    for i in 0..pc {
        let j = (i + 1) % pc;
        if !rounded[i] || !rounded[j] {
            continue;
        }
        if along(enter[j], at(i as isize), at(j as isize)) < along(leave[i], at(i as isize), at(j as isize)) {
            rounded[i] = false;
            rounded[j] = false;
            enter[i] = at(i as isize);
            leave[i] = at(i as isize);
            enter[j] = at(j as isize);
            leave[j] = at(j as isize);
        }
    }
    let n = traced.len();
    let nearest = |p: DVec2| {
        let mut best = 0;
        for k in 1..n {
            if (traced[k] - p).length() < (traced[best] - p).length() {
                best = k;
            }
        }
        best
    };
    let mut pieces: Vec<ShadowPiece> = Vec::new();
    for i in 0..pc {
        if rounded[i] {
            if let (Some(&(radius, _)), Some(&fillet)) = (round_at.get(&i), fillets.get(&i)) {
                if let Some(arc) = arc_through(enter[i], leave[i], fillet, radius, None) {
                    pieces.push(arc);
                }
            }
        }
        let a = leave[i];
        let b = enter[(i + 1) % pc];
        if let Some(&(radius, center)) = arc_from.get(&i) {
            let ia = nearest(a);
            let mut ib = nearest(b);
            if ib <= ia {
                ib += n;
            }
            if let Some(arc) = arc_through(a, b, center, radius, Some(traced[((ia + ib) / 2) % n])) {
                pieces.push(arc);
            }
        } else if (a - b).length() > 1e-9 {
            pieces.push(ShadowPiece::Line { from: a, to: b });
        }
    }
    merge_lines(pieces, tolerance / 4.0)
}

fn seg_dist2(p: DVec2, a: DVec2, b: DVec2) -> f64 {
    let d = b - a;
    let l2 = d.dot(d);
    let t = if l2 > 0.0 { ((p - a).dot(d) / l2).clamp(0.0, 1.0) } else { 0.0 };
    (p.x - a.x - t * d.x).hypot(p.y - a.y - t * d.y)
}

/// Two lines in a row along one straight side made one — where the loop was read from cut a
/// side in two: the joint within `slack` of the line through their far ends.
fn merge_lines(pieces: Vec<ShadowPiece>, slack: f64) -> Vec<ShadowPiece> {
    let mut out = pieces;
    let joinable = |a: &ShadowPiece, b: &ShadowPiece| -> Option<(DVec2, DVec2)> {
        let (ShadowPiece::Line { from: af, to: at }, ShadowPiece::Line { from: bf, to: bt }) = (*a, *b) else { return None };
        if seg_dist2(at, af, bt) < slack && (at - af).dot(bt - bf) > 0.0 {
            Some((af, bt))
        } else {
            None
        }
    };
    let mut i = 0;
    while out.len() > 2 && i < out.len() {
        let j = (i + 1) % out.len();
        if let Some((from, to)) = joinable(&out[i], &out[j]) {
            out[i] = ShadowPiece::Line { from, to };
            out.remove(j);
            if j < i {
                i -= 1;
            }
        } else {
            i += 1;
        }
    }
    out
}

/// A piece that left the band, drawn instead as the stretch of the offset loop it stands for —
/// from the point nearest its start to the one nearest its end, the way the loop runs —
/// thinned, its own two ends kept so its neighbours still meet it. A whole circle has no
/// stretch: it stays.
fn patch_piece(pc: &ShadowPiece, run: &[DVec2], tolerance: f64) -> Vec<ShadowPiece> {
    let Some((from, to)) = piece_ends(pc) else { return vec![*pc] };
    let n = run.len();
    if n == 0 {
        return vec![ShadowPiece::Line { from, to }];
    }
    let nearest = |p: DVec2| {
        let mut best = 0;
        for k in 1..n {
            if (run[k] - p).length() < (run[best] - p).length() {
                best = k;
            }
        }
        best
    };
    let ia = nearest(from);
    let mut ib = nearest(to);
    if ib <= ia {
        ib += n;
    }
    let mut pts = vec![from];
    for k in ia + 1..ib {
        pts.push(run[k % n]);
    }
    pts.push(to);
    let kept = simplify_run(&pts, 0, pts.len(), tolerance);
    let mut out = Vec::new();
    for k in 1..kept.len() {
        let (a, b) = (pts[kept[k - 1]], pts[kept[k]]);
        if (a - b).length() > 1e-9 {
            out.push(ShadowPiece::Line { from: a, to: b });
        }
    }
    out
}

/// Every corner where two lines meet turning by less than `max_turn` (radians) rounded by the
/// arc tangent to both: as large as half of each line allows, so neighbouring rounds never
/// overlap, and smaller — halved, a few times — where `fits` says the arc leaves the band; a
/// corner nothing fits stays sharp. Lines are cut back to where the arcs touch them, so the
/// loop runs on without a kink. Corners beside an arc are left as they are.
fn round_corners(pieces: &[ShadowPiece], max_turn: f64, fits: &dyn Fn(&[ShadowPiece]) -> bool) -> Vec<ShadowPiece> {
    let n = pieces.len();
    if n < 2 {
        return pieces.to_vec();
    }
    // Per corner k — where piece k ends and k + 1 starts — the arc, if any.
    let mut rounds: Vec<Option<(DVec2, DVec2, ShadowPiece)>> = vec![None; n];
    for k in 0..n {
        let (ShadowPiece::Line { from: af, to: at }, ShadowPiece::Line { from: bf, to: bt }) = (pieces[k], pieces[(k + 1) % n]) else { continue };
        let la = (af - at).length();
        let lb = (bf - bt).length();
        if !(la > 0.0 && lb > 0.0) {
            continue;
        }
        let d1 = (at - af) / la;
        let d2 = (bt - bf) / lb;
        let turn_sign = d1.x * d2.y - d1.y * d2.x;
        let turn = d1.dot(d2).clamp(-1.0, 1.0).acos();
        if !(turn >= MIN_TURN && turn < max_turn) {
            continue;
        }
        let c = at;
        let t = (turn / 2.0).tan();
        // Its centre to the side the loop turns to.
        let side = if turn_sign > 0.0 { DVec2::new(-d1.y, d1.x) } else { DVec2::new(d1.y, -d1.x) };
        let mut back = la.min(lb) / 2.0;
        for _ in 0..6 {
            let radius = back / t;
            let enter = c - d1 * back;
            let leave = c + d2 * back;
            let center = enter + side * radius;
            let arc = ShadowPiece::Arc { from: enter, to: leave, center, radius, sweep: if turn_sign > 0.0 { turn } else { -turn } };
            if fits(&[arc]) {
                rounds[k] = Some((enter, leave, arc));
                break;
            }
            back /= 2.0;
        }
    }
    let mut out = Vec::new();
    for k in 0..n {
        let p = pieces[k];
        let before = rounds[(k + n - 1) % n];
        let after = rounds[k];
        if let ShadowPiece::Line { from, to } = p {
            let from = before.map_or(from, |b| b.1);
            let to = after.map_or(to, |a| a.0);
            if (from - to).length() > 1e-9 {
                out.push(ShadowPiece::Line { from, to });
            }
        } else {
            out.push(p);
        }
        if let Some(a) = after {
            out.push(a.2);
        }
    }
    out
}

/// Teeth pulled: a short line between two lines whose ends turn opposite ways — a corner the
/// reading took as two with a stub between, or a wave of the scan followed — replaced by the
/// first of these that `fits` says stays in the band: its neighbours run on to where they
/// cross; the stub folded into the line before it, or after it, so one of its two turns goes;
/// one line from the first's start to the last's end. A tooth nothing replaces stays: it is
/// the part's own jog.
fn pull_teeth(pieces: &[ShadowPiece], short: f64, fits: &dyn Fn(&[ShadowPiece]) -> bool) -> Vec<ShadowPiece> {
    let mut out = pieces.to_vec();
    let cross = |p: DVec2, q: DVec2| p.x * q.y - p.y * q.x;
    let mut i = 0;
    while out.len() > 3 && i < out.len() {
        let n = out.len();
        let (a, b, c) = (out[(i + n - 1) % n], out[i], out[(i + 1) % n]);
        let mut fix: Option<Vec<ShadowPiece>> = None;
        if let (ShadowPiece::Line { from: af, to: at }, ShadowPiece::Line { from: bf, to: bt }, ShadowPiece::Line { from: cf, to: ct }) = (a, b, c) {
            if (bf - bt).length() <= short {
                let da = at - af;
                let db = bt - bf;
                let dc = ct - cf;
                if cross(da, db) * cross(db, dc) < 0.0 {
                    let mut tries: Vec<Vec<ShadowPiece>> = Vec::new();
                    let den = cross(da, dc);
                    if den.abs() > 1e-12 * da.length() * dc.length() {
                        let w = cf - af;
                        let s = cross(w, dc) / den;
                        let t = cross(w, da) / den;
                        // The crossing past a's start and short of c's end, so neither line
                        // turns back on itself.
                        if s > 0.0 && t < 1.0 {
                            let x = af + da * s;
                            tries.push(vec![ShadowPiece::Line { from: af, to: x }, ShadowPiece::Line { from: x, to: ct }]);
                        }
                    }
                    tries.push(vec![ShadowPiece::Line { from: af, to: bt }, c]);
                    tries.push(vec![a, ShadowPiece::Line { from: bf, to: ct }]);
                    tries.push(vec![ShadowPiece::Line { from: af, to: ct }]);
                    fix = tries.into_iter().find(|t| fits(t));
                }
            }
        }
        let Some(fix) = fix else {
            i += 1;
            continue;
        };
        // The three taken out, the loop turned to start where they were.
        let start = (i + n - 1) % n;
        let mut turned: Vec<ShadowPiece> = out[start..].to_vec();
        turned.extend_from_slice(&out[..start]);
        let mut next = fix;
        next.extend_from_slice(&turned[3..]);
        out = next;
        i = 0;
    }
    out
}

/// A loop kept as a run of short lines, thinned to the tolerance.
fn polygon_pieces(points: &[DVec2], tolerance: f64) -> Vec<ShadowPiece> {
    let n = points.len();
    if n < 3 {
        return vec![];
    }
    let flat: Vec<DVec2> = (0..=n).map(|i| points[i % n]).collect();
    let kept = simplify_run(&flat, 0, n + 1, tolerance);
    let mut out = Vec::new();
    for k in 1..kept.len() {
        let (a, b) = (flat[kept[k - 1]], flat[kept[k]]);
        if (a - b).length() > 1e-9 {
            out.push(ShadowPiece::Line { from: a, to: b });
        }
    }
    out
}

// ---- sampling ----------------------------------------------------------------

/// Points along a piece, its start included and its end not, `step` mm apart at most (arcs
/// also to ARC_SAGITTA), each with the outward direction — the right of the way it runs.
fn sample_piece(p: &ShadowPiece, step: f64, pts: &mut Vec<DVec2>, nrm: &mut Vec<DVec2>) {
    let (center, radius, start, sweep, from) = match *p {
        ShadowPiece::Line { from, to } => {
            let len = (from - to).length();
            let k = (len / step).ceil().max(1.0) as usize;
            let t = (to - from) / len;
            for j in 0..k {
                pts.push(from + (to - from) * (j as f64 / k as f64));
                nrm.push(DVec2::new(t.y, -t.x));
            }
            return;
        }
        ShadowPiece::Circle { center, radius } => (center, radius, 0.0, TAU, None),
        ShadowPiece::Arc { from, center, radius, sweep, .. } => (center, radius, (from.y - center.y).atan2(from.x - center.x), sweep, Some(from)),
    };
    let by_chord = if radius > ARC_SAGITTA { 2.0 * (1.0 - ARC_SAGITTA / radius).acos() } else { PI / 4.0 };
    let da = by_chord.min(step / radius.max(1e-9)).min(PI / 8.0);
    let k = ((sweep.abs() / da).ceil() as usize).max(2);
    let sign = if sweep >= 0.0 { 1.0 } else { -1.0 };
    for j in 0..k {
        let t = start + sweep * j as f64 / k as f64;
        let (cx, cy) = (t.cos(), t.sin());
        match (j, from) {
            (0, Some(from)) => pts.push(from),
            _ => pts.push(center + DVec2::new(cx, cy) * radius),
        }
        nrm.push(DVec2::new(sign * cx, sign * cy));
    }
}

fn sample_loop(pieces: &[ShadowPiece], step: f64) -> (Vec<DVec2>, Vec<DVec2>) {
    let (mut pts, mut nrm) = (Vec::new(), Vec::new());
    for p in pieces {
        sample_piece(p, step, &mut pts, &mut nrm);
    }
    (pts, nrm)
}

/// The area a loop encloses, signed: positive counter-clockwise.
pub fn loop_area(pieces: &[ShadowPiece]) -> f64 {
    // The polygon through the ends, plus each arc's segment beyond its chord.
    let mut s = 0.0;
    for p in pieces {
        match *p {
            ShadowPiece::Circle { radius, .. } => return PI * radius * radius,
            ShadowPiece::Line { from, to } => s += (from.x * to.y - to.x * from.y) / 2.0,
            ShadowPiece::Arc { from, to, radius, sweep, .. } => {
                s += (from.x * to.y - to.x * from.y) / 2.0;
                s += (radius * radius * (sweep - sweep.sin())) / 2.0;
            }
        }
    }
    s
}

/// A loop's pieces run the other way round.
pub fn reversed_pieces(pieces: &[ShadowPiece]) -> Vec<ShadowPiece> {
    pieces
        .iter()
        .rev()
        .map(|p| match *p {
            ShadowPiece::Line { from, to } => ShadowPiece::Line { from: to, to: from },
            ShadowPiece::Arc { from, to, center, radius, sweep } => ShadowPiece::Arc { from: to, to: from, center, radius, sweep: -sweep },
            c => c,
        })
        .collect()
}

/// A loop turned to run with the material on its left: outer loops counter-clockwise, holes
/// clockwise.
fn orient(l: ShadowLoop) -> ShadowLoop {
    let ccw = loop_area(&l.pieces) > 0.0;
    if ccw != l.hole {
        return l;
    }
    // A circle has no way round of its own: it runs as its loop says.
    ShadowLoop { hole: l.hole, pieces: reversed_pieces(&l.pieces) }
}

/// How far the pieces lie outside the exact silhouette, nearest and farthest, sampled finely.
fn measure_margin(loops: &[ShadowLoop], reference: &SegmentField, cell: f64) -> Span {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for l in loops {
        let (pts, _) = sample_loop(&l.pieces, cell / 2.0);
        for p in pts {
            let d = reference.signed(p.x, p.y);
            min = min.min(d);
            max = max.max(d);
        }
    }
    if min.is_finite() {
        Span { min, max }
    } else {
        Span { min: 0.0, max: 0.0 }
    }
}

/// The loops as a `SectionCut` on the frame's plane: sampled `step` apart, closed by repeating
/// the first point, with the outward direction.
fn lift_loops(loops: &[ShadowLoop], frame: &SectionFrame, step: f64) -> SectionCut {
    let mut out = SectionCut::default();
    for l in loops {
        let (mut pts, mut nrm) = sample_loop(&l.pieces, step);
        // A circle is sampled counter-clockwise; a hole runs the other way.
        if l.pieces.len() == 1 && l.pieces[0].is_circle() && l.hole {
            pts.reverse();
            nrm.reverse();
            for q in nrm.iter_mut() {
                *q = -*q;
            }
        }
        let n = pts.len();
        if n == 0 {
            continue;
        }
        let chain = (0..=n).map(|j| frame.to_world(pts[j % n])).collect();
        let normals = (0..=n).map(|j| frame.u * nrm[j % n].x + frame.v * nrm[j % n].y).collect();
        out.chains.push(chain);
        out.normals.push(normals);
    }
    out
}

/// The units a readout is given in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum DisplayUnits {
    #[default]
    Mm,
    In,
}

const MM_PER_INCH: f64 = 25.4;

impl DisplayUnits {
    fn word(self, power: i32) -> String {
        let unit = match self {
            DisplayUnits::Mm => "mm",
            DisplayUnits::In => "in",
        };
        format!("{unit}{}", if power == 2 { "²" } else if power == 3 { "³" } else { "" })
    }

    /// A figure held in millimetres (to `power`), shown in these units: `mm_decimals` in
    /// millimetres, one more and never fewer than three in inches; no "-0.00".
    fn figure(self, value: f64, mm_decimals: usize, power: i32) -> String {
        let decimals = if self == DisplayUnits::In { 3.max(mm_decimals + 1) } else { mm_decimals };
        let shown = if self == DisplayUnits::In { value / MM_PER_INCH.powi(power) } else { value };
        let text = to_fixed(shown, decimals);
        if text.parse::<f64>().is_ok_and(|x| x == 0.0) {
            to_fixed(0.0, decimals)
        } else {
            text
        }
    }
}

/// A shadow's readout in words.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShadowWords {
    pub loops: String,
    pub area: String,
    pub band: String,
}

/// "2 loops · 11,243 mm²" and the band the contour lies in — the readout's words, and the
/// row's, in `units`.
pub fn describe_shadow(sh: &ShadowShape, units: DisplayUnits) -> ShadowWords {
    let n = sh.loops.len();
    let mm = |v: f64| format!("{}{}", if v < 0.0 { "−" } else { "" }, units.figure(v.abs(), 2, 1));
    ShadowWords {
        loops: format!("{n} loop{}", if n == 1 { "" } else { "s" }),
        // A whole number of square millimetres; square inches to three decimals.
        area: if units == DisplayUnits::In { format!("{} {}", units.figure(sh.area, 0, 2), units.word(2)) } else { format!("{} {}", grouped(sh.area), units.word(2)) },
        band: format!("{} to {} {} outside the part", mm(sh.margin.min), mm(sh.margin.max), units.word(1)),
    }
}

/// A piece of a shadow's contour as one of the 2D sheet's fits, for the drawing exports.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ShadowFlatFit {
    Line(LineFit),
    Arc(ArcFit),
    Circle(CircleFit),
}

/// The contour as the 2D sheet's fits — a LINE, ARC or CIRCLE each, for the drawing exports:
/// an arc counter-clockwise from its start, as those take it, whichever way the loop runs.
/// Each with no scatter and no points behind it.
pub fn shadow_flat_fits(sh: &ShadowShape) -> Vec<ShadowFlatFit> {
    let mut out = Vec::new();
    for l in &sh.loops {
        for p in &l.pieces {
            match *p {
                ShadowPiece::Circle { center, radius } => out.push(ShadowFlatFit::Circle(CircleFit { center, radius, sigma: 0.0, used_points: 0, form_error: 0.0 })),
                ShadowPiece::Line { from, to } => {
                    let length = (from - to).length();
                    if length <= 0.0 {
                        continue;
                    }
                    out.push(ShadowFlatFit::Line(LineFit { center: (from + to) / 2.0, dir: (to - from) / length, length, sigma: 0.0, used_points: 0, form_error: 0.0 }));
                }
                ShadowPiece::Arc { from, to, center, radius, sweep } => {
                    let begin = if sweep >= 0.0 { from } else { to };
                    out.push(ShadowFlatFit::Arc(ArcFit { center, radius, start: (begin.y - center.y).atan2(begin.x - center.x), sweep: sweep.abs(), sigma: 0.0, used_points: 0, form_error: 0.0 }));
                }
            }
        }
    }
    out
}

/// An outer loop of a shadow with the holes that lie in it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShadowRegion {
    pub outer: ShadowLoop,
    pub holes: Vec<ShadowLoop>,
}

/// The outline as regions — each outer loop with the holes that lie in it, the nearest one
/// round them — as a solid made of it needs them: one body per outer loop, its holes as inner
/// bounds.
pub fn shadow_regions(sh: &ShadowShape) -> Vec<ShadowRegion> {
    let outers: Vec<&ShadowLoop> = sh.loops.iter().filter(|l| !l.hole).collect();
    let polys: Vec<Vec<DVec2>> = outers.iter().map(|l| outline_points(&l.pieces, PI / 36.0)).collect();
    let area: Vec<f64> = outers.iter().map(|l| loop_area(&l.pieces).abs()).collect();
    let mut regions: Vec<ShadowRegion> = outers.iter().map(|o| ShadowRegion { outer: (*o).clone(), holes: vec![] }).collect();
    for hole in sh.loops.iter().filter(|l| l.hole) {
        let Some(&at) = outline_points(&hole.pieces, PI / 36.0).first() else { continue };
        let mut best: Option<usize> = None;
        for i in 0..outers.len() {
            if in_polygon(at, &polys[i]) && best.is_none_or(|b| area[i] < area[b]) {
                best = Some(i);
            }
        }
        if let Some(b) = best {
            regions[b].holes.push(hole.clone());
        }
    }
    regions
}

/// A loop as a polygon for drawing: a line's ends, an arc in steps of at most `arc_step`
/// radians — closed, the first point not repeated.
pub fn outline_points(pieces: &[ShadowPiece], arc_step: f64) -> Vec<DVec2> {
    let mut out = Vec::new();
    for p in pieces {
        let (center, radius, a0, sweep, from, min_k) = match *p {
            ShadowPiece::Line { from, .. } => {
                out.push(from);
                continue;
            }
            ShadowPiece::Circle { center, radius } => (center, radius, 0.0, TAU, None, 16),
            ShadowPiece::Arc { from, center, radius, sweep, .. } => (center, radius, (from.y - center.y).atan2(from.x - center.x), sweep, Some(from), 2),
        };
        let k = ((sweep.abs() / arc_step).ceil() as usize).max(min_k);
        for j in 0..k {
            match (j, from) {
                (0, Some(from)) => out.push(from),
                _ => {
                    let t = a0 + sweep * j as f64 / k as f64;
                    out.push(center + DVec2::new(t.cos(), t.sin()) * radius);
                }
            }
        }
    }
    out
}

fn in_polygon(q: DVec2, p: &[DVec2]) -> bool {
    let mut x = false;
    let n = p.len();
    let mut j = n.wrapping_sub(1);
    for i in 0..n {
        if (p[i].y > q.y) != (p[j].y > q.y) && q.x < ((p[j].x - p[i].x) * (q.y - p[i].y)) / (p[j].y - p[i].y) + p[i].x {
            x = !x;
        }
        j = i;
    }
    x
}

#[cfg(test)]
mod tests;
