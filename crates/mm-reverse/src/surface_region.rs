// SPDX-License-Identifier: AGPL-3.0-only
//! A fitted surface against the region it was fitted through. Ported from ScanRuler
//! (`plugins/reverse/core/surfaceRegion.ts`), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only.
//!
//! The fit reaches past its region by a margin (expand_pct), its quadratics free there, and
//! a split carries it on further along the slope at its edge (`surface_fit::reaching_grid_of`):
//! past the region nothing holds it to the scan. A band whose region was cut where it still
//! curves came out 1.4 mm off the scan just past it, and the split by it cut a block's top
//! 5 mm under the scan outside the band (round 20, the mirror holder's swage). What is read
//! here, over the fit's plane:
//!  - the region's footprint (`SurfaceFitResult::footprint`) as a field of each node's
//!    distance to the nearest of its points (`region_field`);
//!  - how far the surface as fitted stands off the scan past its region, in the rectangle it
//!    spans (`past_region`);
//!  - how far a split's cut face lies off the scan past the region (`off_past_region`);
//!  - the region's outline grown by a margin, as loops (`region_outline`): what a split
//!    within the region (a split feature's `within`) cuts inside.
//!
//! "Off the scan" is where the scan lies under the surface along its height direction: a
//! point whose nearest scan point lies off to the side is past the part's edge, and not
//! read.

use std::collections::{HashMap, HashSet};

use glam::{DVec2, DVec3};

use crate::principal_frame::{js_cmp, js_hypot2, js_hypot3};
use crate::surface_fit::{eval_surface, js_round, OrderedMap, SurfaceFitResult};

/// How far past its region a surface may stand off the scan before the fit says so, mm —
/// the split's tolerance's default; a fit has none of its own.
pub const PAST_REGION_MM: f64 = 0.2;
/// The nodes a side the field over a fit's rectangle takes at most, and all of it — the cell
/// grows past the points' spacing to keep to them.
const FIELD_SIDE: f64 = 160.0;
const FIELD_NODES: f64 = 250_000.0;
/// A point of the scan lies under a point of the surface when the line between them stands
/// within 60° of the height direction (tan 60°, √3), or within a cell of it aside.
const UNDER_TAN: f64 = 1.732_050_807_568_877_2;
/// How many points a region's spacing is read at, at most.
const SPACING_SAMPLES: usize = 600;

/// A field over a rectangle of a fit's plane: each node's distance to the nearest of the
/// points, mm — infinity where none was found. Node (r, c) lies at (u0 + c·cell,
/// v0 + r·cell); `dist` is row-major, single precision as ScanRuler holds it.
#[derive(Debug, Clone, PartialEq)]
pub struct RegionField {
    pub u0: f64,
    pub v0: f64,
    pub cell: f64,
    pub cols: usize,
    pub rows: usize,
    pub dist: Vec<f32>,
}

/// The points' spacing: the median distance from one to its nearest neighbour, read at up
/// to SPACING_SAMPLES of them. Zero for fewer than two.
pub fn point_spacing(uv: &[DVec2]) -> f64 {
    let n = uv.len();
    if n < 2 {
        return 0.0;
    }
    let (mut u_lo, mut u_hi, mut v_lo, mut v_hi) = (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
    for p in uv {
        u_lo = u_lo.min(p.x);
        u_hi = u_hi.max(p.x);
        v_lo = v_lo.min(p.y);
        v_hi = v_hi.max(p.y);
    }
    // A cell no narrower than the spacing of points spread over the whole box: a point's
    // nearest neighbour lies in its cell or one beside it.
    let widest = match (u_hi - u_lo).max(v_hi - v_lo) {
        w if w == 0.0 || w.is_nan() => 1.0,
        w => w,
    };
    let cell = ((((u_hi - u_lo) * (v_hi - v_lo)).max(1e-12) / n as f64).sqrt()).max(widest / 4096.0).max(1e-6);
    let cols = ((v_hi - v_lo) / cell).floor() as i64 + 1;
    let key_of = |cu: i64, cv: i64| cu * cols + cv;
    let cell_of = |p: DVec2| (((p.x - u_lo) / cell).floor() as i64, ((p.y - v_lo) / cell).floor() as i64);
    let mut cells: HashMap<i64, Vec<usize>> = HashMap::new();
    for (i, p) in uv.iter().enumerate() {
        let (cu, cv) = cell_of(*p);
        cells.entry(key_of(cu, cv)).or_default().push(i);
    }
    let step = 1usize.max(n / SPACING_SAMPLES);
    let mut nearest = Vec::new();
    for i in (0..n).step_by(step) {
        let (cu, cv) = cell_of(uv[i]);
        let mut best = f64::INFINITY;
        for a in -1..=1 {
            for b in -1..=1 {
                if cv + b < 0 || cv + b >= cols {
                    continue;
                }
                for &j in cells.get(&key_of(cu + a, cv + b)).map(Vec::as_slice).unwrap_or(&[]) {
                    let d = js_hypot2(uv[j].x - uv[i].x, uv[j].y - uv[i].y);
                    if j != i && d > 1e-9 && d < best {
                        best = d;
                    }
                }
            }
        }
        if best.is_finite() {
            nearest.push(best);
        }
    }
    if nearest.is_empty() {
        return cell;
    }
    nearest.sort_by(|a, b| js_cmp(*a, *b));
    nearest[nearest.len() >> 1]
}

/// The field of the points' distance over the rectangle u × v, `cell` apart — no finer than
/// FIELD_NODES allows. Each node starts from the points nearest it and takes its
/// neighbours' nearest points over in two passes, forward and back (a vector distance
/// transform): within a fraction of the cell of the true distance, at the cost of two sweeps
/// of the grid.
pub fn region_field(uv: &[DVec2], u: [f64; 2], v: [f64; 2], cell: f64) -> RegionField {
    let area = ((u[1] - u[0]) * (v[1] - v[0])).max(1e-12);
    let cell = cell.max((area / FIELD_NODES).sqrt()).max(1e-6);
    let cols = (((u[1] - u[0]) / cell).floor() + 2.0).max(0.0) as usize;
    let rows = (((v[1] - v[0]) / cell).floor() + 2.0).max(0.0) as usize;
    let count = cols * rows;
    let mut su = vec![f64::NAN; count];
    let mut sv = vec![f64::NAN; count];
    let mut dist = vec![f32::INFINITY; count];
    for p in uv {
        let (pu, pv) = (p.x, p.y);
        let c = js_round((pu - u[0]) / cell);
        let r = js_round((pv - v[0]) / cell);
        if !(c >= 0.0 && c < cols as f64 && r >= 0.0 && r < rows as f64) {
            continue;
        }
        let at = r as usize * cols + c as usize;
        let d = js_hypot2(u[0] + c * cell - pu, v[0] + r * cell - pv);
        if d < dist[at] as f64 {
            dist[at] = d as f32;
            su[at] = pu;
            sv[at] = pv;
        }
    }
    let mut take = |at: usize, from: usize, uu: f64, vv: f64| {
        if su[from].is_nan() {
            return;
        }
        let d = js_hypot2(uu - su[from], vv - sv[from]);
        if d < dist[at] as f64 {
            dist[at] = d as f32;
            su[at] = su[from];
            sv[at] = sv[from];
        }
    };
    for r in 0..rows {
        for c in 0..cols {
            let at = r * cols + c;
            let uu = u[0] + c as f64 * cell;
            let vv = v[0] + r as f64 * cell;
            if r > 0 {
                if c > 0 {
                    take(at, at - cols - 1, uu, vv);
                }
                take(at, at - cols, uu, vv);
                if c + 1 < cols {
                    take(at, at - cols + 1, uu, vv);
                }
            }
            if c > 0 {
                take(at, at - 1, uu, vv);
            }
        }
    }
    for r in (0..rows).rev() {
        for c in (0..cols).rev() {
            let at = r * cols + c;
            let uu = u[0] + c as f64 * cell;
            let vv = v[0] + r as f64 * cell;
            if r + 1 < rows {
                if c + 1 < cols {
                    take(at, at + cols + 1, uu, vv);
                }
                take(at, at + cols, uu, vv);
                if c > 0 {
                    take(at, at + cols - 1, uu, vv);
                }
            }
            if c + 1 < cols {
                take(at, at + 1, uu, vv);
            }
        }
    }
    RegionField { u0: u[0], v0: v[0], cell, cols, rows, dist }
}

/// The field's distance at (u, v): its nearest node's — infinity off the field.
pub fn field_at(f: &RegionField, u: f64, v: f64) -> f64 {
    let c = js_round((u - f.u0) / f.cell);
    let r = js_round((v - f.v0) / f.cell);
    // No node at all for a point that is no number (ScanRuler reads `undefined` there).
    if c.is_nan() || r.is_nan() {
        return f64::NAN;
    }
    if !(c >= 0.0 && c < f.cols as f64 && r >= 0.0 && r < f.rows as f64) {
        return f64::INFINITY;
    }
    // A field put together by hand with fewer distances than nodes has none past them —
    // ScanRuler reads `undefined` there, which no comparison passes, as none passes NaN.
    f.dist.get(r as usize * f.cols + c as usize).map_or(f64::NAN, |&d| d as f64)
}

/// How far a point of the surface lies off the scan under it, or None where the scan's
/// nearest point lies off to the side — past the part's edge — or there is none within
/// reach. `near` is the scan's nearest point to a point, within reach.
fn off_scan(p: DVec3, normal: DVec3, near: &impl Fn(DVec3) -> Option<DVec3>, aside: f64) -> Option<f64> {
    let q = near(p)?;
    let d = DVec3::new(p.x - q.x, p.y - q.y, p.z - q.z);
    let off = js_hypot3(d);
    let along = (d.x * normal.x + d.y * normal.y + d.z * normal.z).abs();
    let across = (off * off - along * along).max(0.0).sqrt();
    if across <= aside.max(UNDER_TAN * along) {
        Some(off)
    } else {
        None
    }
}

/// Where a surface stands furthest off the scan, and how far, mm.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PastRegion {
    pub off: f64,
    pub at: DVec3,
}

/// How far the surface as fitted stands off the scan past its region: in the rectangle it
/// spans, at the nodes further than the region's spacing from every point of it — the
/// margin, and any hole the region has — the surface's distance from the scan under it, the
/// largest and where. None with no footprint, or no scan under the margin. `near` is the
/// scan's nearest point to a point, within reach (ScanRuler's `NearScan`).
pub fn past_region(fit: &SurfaceFitResult, near: impl Fn(DVec3) -> Option<DVec3>) -> Option<PastRegion> {
    let uv = fit.footprint.as_deref()?;
    if uv.len() < 4 {
        return None;
    }
    let spacing = point_spacing(uv);
    let extent = (fit.u_range[1] - fit.u_range[0]).max(fit.v_range[1] - fit.v_range[0]);
    let field = region_field(uv, fit.u_range, fit.v_range, spacing.max(extent / FIELD_SIDE));
    let mut worst: Option<PastRegion> = None;
    for r in 0..field.rows {
        let vv = field.v0 + r as f64 * field.cell;
        if vv > fit.v_range[1] {
            break;
        }
        for c in 0..field.cols {
            let uu = field.u0 + c as f64 * field.cell;
            if uu > fit.u_range[1] {
                break;
            }
            if !(field.dist[r * field.cols + c] as f64 > spacing) {
                continue;
            }
            let p = eval_surface(fit, uu, vv);
            if let Some(off) = off_scan(p, fit.normal, &near, field.cell) {
                if worst.is_none_or(|w| off > w.off) {
                    worst = Some(PastRegion { off, at: p });
                }
            }
        }
    }
    worst
}

/// How much of a split's cut face lies off the scan past a tool's region: the most, mm; the
/// area further off than the tolerance, mm²; and where the most is.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OffPastRegion {
    pub max: f64,
    pub area: f64,
    pub at: DVec3,
}

/// How far points spread over a cut face on a fitted tool (each standing for `areas` of it)
/// lie off the scan, of those past the tool's region — further than its spacing from every
/// point of it, over its plane. None where none past the region has the scan under it.
pub fn off_past_region(
    fit: &SurfaceFitResult,
    points: &[DVec3],
    areas: &[f64],
    near: impl Fn(DVec3) -> Option<DVec3>,
    tolerance: f64,
) -> Option<OffPastRegion> {
    let uv = fit.footprint.as_deref()?;
    if uv.len() < 4 {
        return None;
    }
    let spacing = point_spacing(uv);
    let (bu, bv) = box_of(uv, 2.0 * spacing);
    let field = region_field(uv, bu, bv, spacing.max((bu[1] - bu[0]).max(bv[1] - bv[0]) / FIELD_SIDE));
    let mut out: Option<OffPastRegion> = None;
    for (&p, &area) in points.iter().zip(areas) {
        let d = DVec3::new(p.x - fit.origin.x, p.y - fit.origin.y, p.z - fit.origin.z);
        let uu = d.x * fit.axis_u.x + d.y * fit.axis_u.y + d.z * fit.axis_u.z;
        let vv = d.x * fit.axis_v.x + d.y * fit.axis_v.y + d.z * fit.axis_v.z;
        if !(field_at(&field, uu, vv) > spacing) {
            continue;
        }
        let Some(off) = off_scan(p, fit.normal, &near, field.cell) else { continue };
        let o = out.get_or_insert(OffPastRegion { max: 0.0, area: 0.0, at: p });
        if off > o.max {
            o.max = off;
            o.at = p;
        }
        if off > tolerance {
            o.area += area;
        }
    }
    out
}

/// The (u, v) box round the points, `pad` wider every way.
fn box_of(uv: &[DVec2], pad: f64) -> ([f64; 2], [f64; 2]) {
    let (mut u_lo, mut u_hi, mut v_lo, mut v_hi) = (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
    for p in uv {
        u_lo = u_lo.min(p.x);
        u_hi = u_hi.max(p.x);
        v_lo = v_lo.min(p.y);
        v_hi = v_hi.max(p.y);
    }
    ([u_lo - pad, u_hi + pad], [v_lo - pad, v_hi + pad])
}

/// A region's outline grown by a margin, over its fit's plane: the outer loops,
/// counter-clockwise in (u, v), holes filled; the field they were drawn on and the distance
/// they lie at.
#[derive(Debug, Clone, PartialEq)]
pub struct RegionOutline {
    pub loops: Vec<Vec<DVec2>>,
    pub field: RegionField,
    pub level: f64,
}

/// The region's outline grown by `margin` mm: the line round its points at the margin plus
/// their spacing — the spacing so the gaps between the points close — drawn through the
/// distance field (marching squares), the holes left out, and thinned to within a twentieth
/// of that distance, or a quarter of the field's cell (Ramer–Douglas–Peucker): each side is
/// a wall where the cut within it stops. None with no footprint or no loop.
pub fn region_outline(fit: &SurfaceFitResult, margin: f64) -> Option<RegionOutline> {
    let uv = fit.footprint.as_deref()?;
    if uv.len() < 4 {
        return None;
    }
    let spacing = point_spacing(uv);
    let level = margin.max(0.0) + spacing;
    let (su, sv) = box_of(uv, 0.0);
    let extent = (su[1] - su[0]).max(sv[1] - sv[0]);
    let cell = (spacing / 2.0).max(extent / (2.0 * FIELD_SIDE));
    let (bu, bv) = box_of(uv, level + 2.0 * cell);
    let field = region_field(uv, bu, bv, cell);
    let tolerance = (field.cell / 4.0).max(level / 20.0);
    let loops: Vec<Vec<DVec2>> = level_loops(&field, level)
        .into_iter()
        .filter(|l| loop_area(l) > 0.0)
        .map(|l| thin_loop(l, tolerance))
        .filter(|l| l.len() >= 3)
        .collect();
    if loops.is_empty() {
        None
    } else {
        Some(RegionOutline { loops, field, level })
    }
}

/// The lines where the field crosses `level`, as closed loops with the inside (the field at
/// or under the level) on their left: outer loops counter-clockwise, holes clockwise. Each
/// square of four nodes adds the pieces its corners call for, a saddle's by the value at its
/// middle.
fn level_loops(f: &RegionField, level: f64) -> Vec<Vec<DVec2>> {
    let (cols, rows, cell, u0, v0) = (f.cols, f.rows, f.cell, f.u0, f.v0);
    let dist = |i: usize| f.dist[i] as f64;
    let inside = |r: usize, c: usize| dist(r * cols + c) <= level;
    // An edge between two nodes by a key: along u from (r, c), or along v.
    let h_key = |r: usize, c: usize| (r * cols + c) * 2;
    let v_key = |r: usize, c: usize| (r * cols + c) * 2 + 1;
    let point_on = |key: usize| -> DVec2 {
        let node = key >> 1;
        let (r, c) = (node / cols, node % cols);
        let along_v = key & 1 == 1;
        let (r2, c2) = if along_v { (r + 1, c) } else { (r, c + 1) };
        let (a, b) = (dist(r * cols + c), dist(r2 * cols + c2));
        let fa = if a.is_finite() { a } else { level * 4.0 + cell * 4.0 };
        let fb = if b.is_finite() { b } else { level * 4.0 + cell * 4.0 };
        let t = if (fb - fa).abs() > 1e-12 { ((level - fa) / (fb - fa)).clamp(0.0, 1.0) } else { 0.5 };
        let (dc, dr) = if along_v { (0.0, 1.0) } else { (1.0, 0.0) };
        DVec2::new(u0 + (c as f64 + dc * t) * cell, v0 + (r as f64 + dr * t) * cell)
    };
    let mut next: OrderedMap<usize, usize> = OrderedMap::new();
    let mut r = 0;
    while r + 1 < rows {
        let mut c = 0;
        while c + 1 < cols {
            // Round the square counter-clockwise: bottom, right, top, left — each edge from
            // the corner it leaves to the one it reaches.
            let corner = [inside(r, c), inside(r, c + 1), inside(r + 1, c + 1), inside(r + 1, c)];
            let edges = [h_key(r, c), v_key(r, c + 1), h_key(r + 1, c), v_key(r, c)];
            let mut leaving = Vec::with_capacity(2);
            let mut entering = Vec::with_capacity(2);
            for e in 0..4 {
                let (a, b) = (corner[e], corner[(e + 1) % 4]);
                if a && !b {
                    leaving.push(e);
                } else if !a && b {
                    entering.push(e);
                }
            }
            if leaving.len() == 1 {
                if let Some(&en) = entering.first() {
                    next.insert(edges[leaving[0]], edges[en]);
                }
            } else if !leaving.is_empty() {
                // A saddle: the inside corners joined through the middle when it is inside —
                // each leaving edge to the entering one after it, going round — and kept
                // apart when not, to the one before it.
                let middle = (dist(r * cols + c) + dist(r * cols + c + 1) + dist((r + 1) * cols + c + 1) + dist((r + 1) * cols + c)) / 4.0 <= level;
                for &e in &leaving {
                    next.insert(edges[e], edges[if middle { (e + 1) % 4 } else { (e + 3) % 4 }]);
                }
            }
            c += 1;
        }
        r += 1;
    }
    let mut loops = Vec::new();
    let mut seen: HashSet<usize> = HashSet::new();
    for start in next.keys() {
        if seen.contains(&start) {
            continue;
        }
        let mut lp = Vec::new();
        let mut at = Some(start);
        while let Some(a) = at {
            if seen.contains(&a) {
                break;
            }
            seen.insert(a);
            lp.push(point_on(a));
            at = next.get(&a).copied();
        }
        if at == Some(start) && lp.len() >= 3 {
            loops.push(lp);
        }
    }
    loops
}

/// A closed loop's signed area: positive counter-clockwise.
fn loop_area(lp: &[DVec2]) -> f64 {
    let mut a = 0.0;
    for i in 0..lp.len() {
        let (p, q) = (lp[i], lp[(i + 1) % lp.len()]);
        a += p.x * q.y - q.x * p.y;
    }
    a / 2.0
}

/// A closed loop thinned to the corners that keep it within `tolerance` of where it ran
/// (Ramer–Douglas–Peucker, from its two points furthest apart).
fn thin_loop(lp: Vec<DVec2>, tolerance: f64) -> Vec<DVec2> {
    let len = lp.len();
    if len <= 4 {
        return lp;
    }
    let (mut far, mut best) = (0, -1.0);
    for i in 1..len {
        let d = js_hypot2(lp[i].x - lp[0].x, lp[i].y - lp[0].y);
        if d > best {
            (best, far) = (d, i);
        }
    }
    let mut keep = vec![false; len];
    keep[0] = true;
    keep[far] = true;
    // Each run from … to going round, to after from; kept on a stack of its own rather than
    // the call stack, the order of the runs being of no account to what is kept.
    let mut runs = vec![(far, 0), (0, far)];
    while let Some((from, to)) = runs.pop() {
        let count = (to + len - from) % len;
        if count < 2 {
            continue;
        }
        let (a, b) = (lp[from], lp[to % len]);
        let (dx, dy) = (b.x - a.x, b.y - a.y);
        let length = js_hypot2(dx, dy);
        let (mut worst, mut at) = (-1.0, 0);
        for k in 1..count {
            let p = lp[(from + k) % len];
            let d = if length > 1e-12 { ((p.x - a.x) * dy - (p.y - a.y) * dx).abs() / length } else { js_hypot2(p.x - a.x, p.y - a.y) };
            if d > worst {
                (worst, at) = (d, (from + k) % len);
            }
        }
        if worst > tolerance {
            keep[at] = true;
            runs.push((at, to));
            runs.push((from, at));
        }
    }
    lp.into_iter().zip(keep).filter_map(|(p, k)| k.then_some(p)).collect()
}

// Ported from ScanRuler's plugins/reverse/tests/reSplitWithin.test.ts, "a region against its
// fit" (the tests there that need no kernel).
#[cfg(test)]
mod tests {
    use super::*;
    use crate::surface_fit::{footprint_of, Net, SurfacePlane};

    /// The values a JavaScript `for (let x = from; x <= to; x += step)` runs through.
    fn steps(from: f64, to: f64, step: f64) -> Vec<f64> {
        let mut out = Vec::new();
        let mut x = from;
        while x <= to {
            out.push(x);
            x += step;
        }
        out
    }

    /// The plane z = 0, u along x and v along y.
    fn plane() -> SurfacePlane {
        SurfacePlane { origin: DVec3::ZERO, axis_u: DVec3::X, axis_v: DVec3::Y, u_range: [0.0, 0.0], v_range: [0.0, 0.0] }
    }

    /// A flat fit over the plane z = 0: a cubic patch over u −14…14, v −7…7 whose heights
    /// are all zero, with the footprint given.
    fn flat_fit(footprint: Vec<DVec2>) -> SurfaceFitResult {
        SurfaceFitResult {
            origin: DVec3::ZERO,
            axis_u: DVec3::X,
            axis_v: DVec3::Y,
            normal: DVec3::Z,
            u_range: [-14.0, 14.0],
            v_range: [-7.0, 7.0],
            degree: 3,
            knots_u: vec![-14.0, -14.0, -14.0, -14.0, 14.0, 14.0, 14.0, 14.0],
            knots_v: vec![-7.0, -7.0, -7.0, -7.0, 7.0, 7.0, 7.0, 7.0],
            nu: 4,
            nv: 4,
            heights: vec![0.0; 16],
            rms: 0.0,
            max: 0.0,
            net: Net { u: 4, v: 4 },
            footprint: Some(footprint),
        }
    }

    /// A 20 × 10 grid of points 0.5 apart, over z = 0.
    fn grid() -> Vec<DVec3> {
        let mut xyz = Vec::new();
        for x in steps(-10.0, 10.0 + 1e-9, 0.5) {
            for y in steps(-5.0, 5.0 + 1e-9, 0.5) {
                xyz.push(DVec3::new(x, y, 0.0));
            }
        }
        xyz
    }

    #[test]
    fn reads_the_points_spacing() {
        assert!((point_spacing(&footprint_of(&plane(), &grid())) - 0.5).abs() < 5e-4);
    }

    #[test]
    fn draws_the_outline_grown_by_the_margin_and_the_spacing_its_corners_round() {
        let outline = region_outline(&flat_fit(footprint_of(&plane(), &grid())), 2.0).unwrap();
        assert!((outline.level - 2.5).abs() < 5e-4);
        assert_eq!(outline.loops.len(), 1);
        let lp = &outline.loops[0];
        let mut area = 0.0;
        for i in 0..lp.len() {
            let (p, q) = (lp[i], lp[(i + 1) % lp.len()]);
            area += p.x * q.y - q.x * p.y;
        }
        area /= 2.0;
        // A 20 × 10 rectangle grown by 2.5: 25 × 15 with its corners rounded.
        assert!(area > 0.0);
        assert!((area - (25.0 * 15.0 - (4.0 - std::f64::consts::PI) * 2.5f64.powi(2))).abs() < 5.0, "area {area}");
        let max_x = lp.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max);
        let min_x = lp.iter().map(|p| p.x).fold(f64::INFINITY, f64::min);
        assert!((max_x - 12.5).abs() < 0.05, "max x {max_x}");
        assert!((min_x + 12.5).abs() < 0.05, "min x {min_x}");
        // Thinned: a few corners a round, not the field's every step.
        assert!(lp.len() < 80, "{} corners", lp.len());
    }

    #[test]
    fn fills_a_hole_in_the_region_and_keeps_two_regions_apart() {
        let mut xyz = Vec::new();
        for x in steps(-10.0, 10.0 + 1e-9, 0.5) {
            for y in steps(-5.0, 5.0 + 1e-9, 0.5) {
                if !(x.abs() < 3.0 && y.abs() < 2.0) {
                    xyz.push(DVec3::new(x, y, 0.0));
                }
            }
        }
        for x in steps(30.0, 34.0 + 1e-9, 0.5) {
            for y in steps(-2.0, 2.0 + 1e-9, 0.5) {
                xyz.push(DVec3::new(x, y, 0.0));
            }
        }
        let outline = region_outline(&flat_fit(footprint_of(&plane(), &xyz)), 1.0).unwrap();
        assert_eq!(outline.loops.len(), 2);
    }

    #[test]
    fn reads_the_margin_off_the_scan_not_the_points_under_the_region() {
        // A flat region at z = 0 fitted as z = 0, the surface's margin over a scan that steps
        // down 2 mm past x = 12.
        let flat = flat_fit(footprint_of(&plane(), &grid()));
        let near = |p: DVec3| Some(DVec3::new(p.x, p.y, if p.x > 12.0 { -2.0 } else { 0.0 }));
        let past = past_region(&flat, near).unwrap();
        assert!((past.off - 2.0).abs() < 5e-4);
        assert!(past.at.x > 12.0);
        // The scan off to the side — past the part's edge — is not read.
        assert!(past_region(&flat, |p: DVec3| Some(DVec3::new(p.x + 5.0, p.y, 0.0))).is_none());
    }

    #[test]
    fn a_cut_face_past_the_region_is_read_off_the_scan() {
        // Not in ScanRuler's tests (its split tests run on the kernel): points of a cut face on
        // the flat fit, a 1 mm² each, inside the region and past it, over a scan that steps
        // down 2 mm past x = 12.
        let flat = flat_fit(footprint_of(&plane(), &grid()));
        let near = |p: DVec3| Some(DVec3::new(p.x, p.y, if p.x > 12.0 { -2.0 } else { 0.0 }));
        let points: Vec<DVec3> = [(0.0, 0.0), (11.0, 0.0), (13.0, 0.0), (13.0, 2.0)].iter().map(|&(x, y)| DVec3::new(x, y, 0.0)).collect();
        let off = off_past_region(&flat, &points, &[1.0; 4], near, PAST_REGION_MM).unwrap();
        assert!((off.max - 2.0).abs() < 1e-12);
        assert_eq!(off.area, 2.0);
        assert!(off.at.x > 12.0);
        // field_at reads the field at its nearest node, and nothing off it.
        let outline = region_outline(&flat, 0.0).unwrap();
        assert!(field_at(&outline.field, 0.0, 0.0) < 0.5);
        assert_eq!(field_at(&outline.field, 1e6, 0.0), f64::INFINITY);
    }
}
