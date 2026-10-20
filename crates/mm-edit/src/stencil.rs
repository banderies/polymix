//! The Stamp library's own shapes (Polymix designs, no Meshmixer art): geometric stencils
//! (some with holes, like a ring), icons, and a monoline font for labelling parts (digits,
//! capital letters and a few marks), plus [`stamp_loops`], which cuts a stencil made of
//! several loops into a surface.
//!
//! A stencil is a set of closed loops filled by the even-odd rule: outer loops run
//! counter-clockwise, the holes inside them clockwise. Library stencils are centred on their
//! bounding box and fit the unit circle.
//!
//! Round shapes (the letters, the check mark, the curved arrow) are drawn as round-capped
//! strokes and filled polygons; [`Sketch::outline`] traces the border of their union on a
//! fine grid (marching squares on the distance to the strokes) and simplifies it.

mod font;

pub use font::{glyph_loops, text_loops, FONT_CHARS};

use crate::stamp::{stamp_polygon, StampFrame, Stencil};
use crate::{EditError, EditResult};
use glam::DVec2;
use mm_core::Mesh;
use std::collections::{HashMap, HashSet};
use std::f64::consts::TAU;

/// A stamp outline: closed loops filled by the even-odd rule.
#[derive(Clone, Debug, PartialEq)]
pub struct StencilShape {
    pub name: String,
    /// Outer loops counter-clockwise, holes clockwise; centred, inside the unit circle.
    pub loops: Vec<Vec<DVec2>>,
    /// Text stencils: the height of the capital letters, in stencil units (so a stamp of
    /// radius `r` has letters `r · cap_height` tall).
    pub cap_height: Option<f64>,
}

impl StencilShape {
    /// A stencil from loops at any scale and orientation: oriented by nesting, centred on
    /// its bounding box and fitted to the unit circle.
    pub fn new(name: impl Into<String>, loops: Vec<Vec<DVec2>>) -> Self {
        Self { name: name.into(), loops: normalize_loops(loops).0, cap_height: None }
    }
}

// ---------------------------------------------------------------------------------------
// The library
// ---------------------------------------------------------------------------------------

fn pts(p: &[(f64, f64)]) -> Vec<DVec2> {
    p.iter().map(|&(x, y)| DVec2::new(x, y)).collect()
}

/// Counter-clockwise circle of `n` points.
fn circle(c: DVec2, r: f64, n: usize) -> Vec<DVec2> {
    (0..n).map(|i| c + DVec2::from_angle(TAU * i as f64 / n as f64) * r).collect()
}

/// Points along an elliptical arc from `a0` to `a1` (degrees; the sweep may be negative),
/// both ends included, at most 6° apart.
pub(crate) fn arc(c: DVec2, rx: f64, ry: f64, a0: f64, a1: f64) -> Vec<DVec2> {
    let n = ((a1 - a0).abs() / 6.0).ceil().max(1.0) as usize;
    (0..=n)
        .map(|i| {
            let a = (a0 + (a1 - a0) * i as f64 / n as f64).to_radians();
            c + DVec2::new(rx * a.cos(), ry * a.sin())
        })
        .collect()
}

/// Rectangle `2·hw × 2·hh` with corners rounded to radius `r`, counter-clockwise.
fn rounded_rect(hw: f64, hh: f64, r: f64) -> Vec<DVec2> {
    let r = r.min(hw).min(hh);
    let mut out = Vec::new();
    for (c, a0) in [(DVec2::new(hw - r, hh - r), 0.0), (DVec2::new(r - hw, hh - r), 90.0), (DVec2::new(r - hw, r - hh), 180.0), (DVec2::new(hw - r, r - hh), 270.0)] {
        let mut a = arc(c, r, r, a0, a0 + 90.0);
        // Where two arcs meet (r = hw or hh) keep one point.
        if out.last().is_some_and(|l: &DVec2| l.distance(a[0]) < 1e-12) {
            a.remove(0);
        }
        out.extend(a);
    }
    if out.len() > 1 && out[0].distance(out[out.len() - 1]) < 1e-12 {
        out.pop();
    }
    out
}

/// The geometric stencils: the original seven, then more plain shapes (rounded rectangles,
/// slots, rings…) that are handy for marking and keying printed parts.
pub fn shape_stencils() -> Vec<StencilShape> {
    let mut out: Vec<StencilShape> = Stencil::ALL.iter().map(|s| StencilShape::new(s.label(), vec![s.outline()])).collect();
    let ngon = |n: usize, start: f64| -> Vec<DVec2> { (0..n).map(|i| DVec2::from_angle((start + 360.0 * i as f64 / n as f64).to_radians())).collect() };
    let plus = pts(&[(0.3, 0.3), (0.3, 1.0), (-0.3, 1.0), (-0.3, 0.3), (-1.0, 0.3), (-1.0, -0.3), (-0.3, -0.3), (-0.3, -1.0), (0.3, -1.0), (0.3, -0.3), (1.0, -0.3), (1.0, 0.3)]);
    let cross: Vec<DVec2> = plus.iter().map(|p| DVec2::from_angle(TAU / 8.0).rotate(*p)).collect();
    // Circle with a flat: a keyed outline that only fits one way round.
    let flat = 0.6_f64.asin().to_degrees();
    let keyed = arc(DVec2::ZERO, 1.0, 1.0, 90.0 + flat, 450.0 - flat);
    let mut half = arc(DVec2::ZERO, 1.0, 1.0, 0.0, 180.0);
    half.iter_mut().for_each(|p| p.y -= 0.5);
    let more: Vec<(&str, Vec<Vec<DVec2>>)> = vec![
        ("Pentagon", vec![ngon(5, 90.0)]),
        ("Octagon", vec![ngon(8, 22.5)]),
        ("Diamond", vec![pts(&[(0.0, -1.0), (0.62, 0.0), (0.0, 1.0), (-0.62, 0.0)])]),
        ("Ellipse", vec![circle(DVec2::ZERO, 1.0, 64).into_iter().map(|p| DVec2::new(p.x, p.y * 0.5)).collect()]),
        ("Rounded square", vec![rounded_rect(1.0, 1.0, 0.32)]),
        ("Rounded rectangle", vec![rounded_rect(1.0, 0.5, 0.2)]),
        ("Slot", vec![rounded_rect(1.0, 0.3, 0.3)]),
        ("Half circle", vec![half]),
        ("Keyed circle", vec![keyed]),
        ("Ring", vec![circle(DVec2::ZERO, 1.0, 64), circle(DVec2::ZERO, 0.62, 48)]),
        ("Plus", vec![plus]),
        ("Cross", vec![cross]),
    ];
    out.extend(more.into_iter().map(|(n, l)| StencilShape::new(n, l)));
    out
}

/// Logo-free geometric icons: flow arrows, marks and symbols.
pub fn icon_stencils() -> Vec<StencilShape> {
    let double_arrow = pts(&[(-1.0, 0.0), (-0.5, 0.45), (-0.5, 0.17), (0.5, 0.17), (0.5, 0.45), (1.0, 0.0), (0.5, -0.45), (0.5, -0.17), (-0.5, -0.17), (-0.5, -0.45)]);
    // Curved arrow: a stroke over the top, ending in a filled head.
    let (c, r, end) = (DVec2::new(0.0, -0.25), 0.72, 28.0_f64);
    let bend = arc(c, r, r, 190.0, end);
    let tip_at = c + DVec2::from_angle(end.to_radians()) * r;
    let dir = DVec2::new(end.to_radians().sin(), -end.to_radians().cos());
    let side = dir.perp();
    let head = vec![tip_at + dir * 0.42, tip_at - dir * 0.02 + side * 0.32, tip_at - dir * 0.02 - side * 0.32];
    let curved = Sketch::default().stroke(bend, 0.24).fill(head).outline(0.012);
    let chevron = Sketch::default().stroke(pts(&[(-0.9, -0.4), (0.0, 0.45), (0.9, -0.4)]), 0.42).outline(0.012);
    let check = Sketch::default().stroke(pts(&[(-0.85, 0.05), (-0.3, -0.5), (0.85, 0.65)]), 0.34).outline(0.012);
    let target = vec![circle(DVec2::ZERO, 1.0, 64), circle(DVec2::ZERO, 0.74, 56), circle(DVec2::ZERO, 0.4, 40)];
    // Gear: twelve trapezoid teeth and a bore.
    let teeth = 12;
    let mut gear = Vec::new();
    for i in 0..teeth {
        let a = TAU * i as f64 / teeth as f64;
        let w = TAU / teeth as f64;
        for (da, r) in [(0.0, 0.8), (0.18, 0.8), (0.3, 1.0), (0.58, 1.0), (0.7, 0.8)] {
            gear.push(DVec2::from_angle(a + da * w) * r);
        }
    }
    let bore = circle(DVec2::ZERO, 0.36, 36);
    // Drop: round below, pointed above.
    let drop: Vec<DVec2> = (0..72)
        .map(|i| {
            let t = TAU * i as f64 / 72.0;
            DVec2::new(t.sin() * (t / 2.0).sin(), t.cos())
        })
        .collect();
    // Crescent: the disc minus a shifted disc.
    let (shift, inner) = (0.5_f64, 0.85_f64);
    let x = (1.0 - inner * inner + shift * shift) / (2.0 * shift);
    let y = (1.0 - x * x).sqrt();
    let a = y.atan2(x).to_degrees();
    let b = y.atan2(x - shift).to_degrees();
    let mut crescent = arc(DVec2::ZERO, 1.0, 1.0, a, 360.0 - a);
    let mut back = arc(DVec2::new(shift, 0.0), inner, inner, 360.0 - b, b);
    back.remove(0);
    back.pop();
    crescent.extend(back);
    let list: Vec<(&str, Vec<Vec<DVec2>>)> = vec![
        ("Double arrow", vec![double_arrow]),
        ("Curved arrow", curved),
        ("Chevron", chevron),
        ("Check", check),
        ("Target", target),
        ("Gear", vec![gear, bore]),
        ("Drop", vec![drop]),
        ("Crescent", vec![crescent]),
    ];
    list.into_iter().map(|(n, l)| StencilShape::new(n, l)).collect()
}

/// A text label as a stencil (see [`text_loops`]), named after the text.
pub fn text_stencil(text: &str) -> Result<StencilShape, String> {
    let loops = text_loops(text)?;
    let (loops, scale) = normalize_loops(loops);
    Ok(StencilShape { name: text.trim().to_uppercase(), loops, cap_height: Some(scale) })
}

// ---------------------------------------------------------------------------------------
// Loops: validity, orientation, normalization, simplification
// ---------------------------------------------------------------------------------------

/// Signed area of a closed polygon (positive when counter-clockwise).
pub fn signed_area(p: &[DVec2]) -> f64 {
    (0..p.len()).map(|i| p[i].perp_dot(p[(i + 1) % p.len()])).sum::<f64>() * 0.5
}

/// Whether `p` is inside the closed polygon (crossing count).
pub fn point_in_polygon(p: DVec2, poly: &[DVec2]) -> bool {
    let mut inside = false;
    for i in 0..poly.len() {
        let (a, b) = (poly[i], poly[(i + 1) % poly.len()]);
        if (a.y > p.y) != (b.y > p.y) && p.x < a.x + (p.y - a.y) / (b.y - a.y) * (b.x - a.x) {
            inside = !inside;
        }
    }
    inside
}

/// Whether `p` is inside the stencil (even-odd over all loops).
pub fn point_in_loops(p: DVec2, loops: &[Vec<DVec2>]) -> bool {
    loops.iter().filter(|l| point_in_polygon(p, l)).count() % 2 == 1
}

fn orient(a: DVec2, b: DVec2, c: DVec2) -> f64 {
    robust::orient2d(robust::Coord { x: a.x, y: a.y }, robust::Coord { x: b.x, y: b.y }, robust::Coord { x: c.x, y: c.y })
}

/// Closed segments `ab` and `cd` share a point (crossing, touching or overlapping).
fn segments_touch(a: DVec2, b: DVec2, c: DVec2, d: DVec2) -> bool {
    let within = |p: DVec2, q: DVec2, r: DVec2| r.x >= p.x.min(q.x) && r.x <= p.x.max(q.x) && r.y >= p.y.min(q.y) && r.y <= p.y.max(q.y);
    let (d1, d2, d3, d4) = (orient(c, d, a), orient(c, d, b), orient(a, b, c), orient(a, b, d));
    if d1 * d2 < 0.0 && d3 * d4 < 0.0 {
        return true;
    }
    (d1 == 0.0 && within(c, d, a)) || (d2 == 0.0 && within(c, d, b)) || (d3 == 0.0 && within(a, b, c)) || (d4 == 0.0 && within(a, b, d))
}

/// Check that a stencil is usable: every loop closed (at least three distinct points, no
/// zero-length edge, an area), no loop crossing or touching itself or another loop.
pub fn validate_loops(loops: &[Vec<DVec2>]) -> Result<(), String> {
    if loops.is_empty() {
        return Err("no loops".into());
    }
    struct Seg {
        a: DVec2,
        b: DVec2,
        lp: usize,
        i: usize,
        lo: DVec2,
        hi: DVec2,
    }
    let mut segs = Vec::new();
    for (k, l) in loops.iter().enumerate() {
        if l.len() < 3 {
            return Err(format!("loop {k} has {} points", l.len()));
        }
        if l.iter().any(|p| !p.is_finite()) {
            return Err(format!("loop {k} has a point that is not a number"));
        }
        if signed_area(l).abs() < 1e-12 {
            return Err(format!("loop {k} has no area"));
        }
        for i in 0..l.len() {
            let (a, b) = (l[i], l[(i + 1) % l.len()]);
            if a == b {
                return Err(format!("loop {k} repeats point {i}"));
            }
            segs.push(Seg { a, b, lp: k, i, lo: a.min(b), hi: a.max(b) });
        }
    }
    // Sweep along x: only segments whose x ranges overlap are compared.
    let mut order: Vec<usize> = (0..segs.len()).collect();
    order.sort_by(|&x, &y| segs[x].lo.x.total_cmp(&segs[y].lo.x));
    for (oi, &s) in order.iter().enumerate() {
        let sa = &segs[s];
        for &t in &order[oi + 1..] {
            let sb = &segs[t];
            if sb.lo.x > sa.hi.x {
                break;
            }
            if sb.lo.y > sa.hi.y || sb.hi.y < sa.lo.y {
                continue;
            }
            if sa.lp == sb.lp {
                let n = loops[sa.lp].len();
                let (i, j) = (sa.i.min(sb.i), sa.i.max(sb.i));
                if j == i + 1 || (i == 0 && j == n - 1) {
                    // Neighbours share a corner; they must not fold back over each other.
                    let (p, q, r) = if j == i + 1 { (loops[sa.lp][i], loops[sa.lp][j], loops[sa.lp][(j + 1) % n]) } else { (loops[sa.lp][n - 1], loops[sa.lp][0], loops[sa.lp][1]) };
                    if orient(p, q, r) == 0.0 && (q - p).dot(r - q) < 0.0 {
                        return Err(format!("loop {} folds back on itself at point {}", sa.lp, if j == i + 1 { j } else { 0 }));
                    }
                    continue;
                }
                if segments_touch(sa.a, sa.b, sb.a, sb.b) {
                    return Err(format!("loop {} crosses itself (edges {i} and {j})", sa.lp));
                }
            } else if segments_touch(sa.a, sa.b, sb.a, sb.b) {
                return Err(format!("loops {} and {} cross or touch", sa.lp.min(sb.lp), sa.lp.max(sb.lp)));
            }
        }
    }
    Ok(())
}

/// Orient loops for the even-odd rule: loops nested in an even number of others run
/// counter-clockwise, the others (holes) clockwise.
pub fn orient_loops(loops: &mut [Vec<DVec2>]) {
    let depth: Vec<usize> = (0..loops.len()).map(|i| (0..loops.len()).filter(|&j| j != i && point_in_polygon(loops[i][0], &loops[j])).count()).collect();
    for (l, d) in loops.iter_mut().zip(depth) {
        if (signed_area(l) > 0.0) != (d % 2 == 0) {
            l.reverse();
        }
    }
}

/// Loops oriented ([`orient_loops`]), centred on their bounding box and scaled to fit the
/// unit circle; also returns the scale applied.
pub fn normalize_loops(mut loops: Vec<Vec<DVec2>>) -> (Vec<Vec<DVec2>>, f64) {
    loops.retain(|l| l.len() >= 3);
    orient_loops(&mut loops);
    let (mut lo, mut hi) = (DVec2::splat(f64::INFINITY), DVec2::splat(f64::NEG_INFINITY));
    for p in loops.iter().flatten() {
        lo = lo.min(*p);
        hi = hi.max(*p);
    }
    let c = (lo + hi) * 0.5;
    let r = loops.iter().flatten().map(|p| (*p - c).length()).fold(0.0, f64::max);
    let s = if r > 0.0 { 1.0 / r } else { 1.0 };
    for p in loops.iter_mut().flatten() {
        *p = (*p - c) * s;
    }
    (loops, s)
}

fn seg_dist(p: DVec2, a: DVec2, b: DVec2) -> f64 {
    let ab = b - a;
    let l2 = ab.length_squared();
    let t = if l2 > 0.0 { ((p - a).dot(ab) / l2).clamp(0.0, 1.0) } else { 0.0 };
    p.distance(a + ab * t)
}

/// Douglas-Peucker simplification of a closed loop: drops points closer than `tol` to the
/// simplified outline.
pub fn simplify_loop(p: &[DVec2], tol: f64) -> Vec<DVec2> {
    let n = p.len();
    if n <= 3 {
        return p.to_vec();
    }
    // Split at the point farthest from the first, then simplify each open half.
    let far = (1..n).max_by(|&a, &b| p[a].distance_squared(p[0]).total_cmp(&p[b].distance_squared(p[0]))).unwrap();
    let mut keep = vec![false; n];
    keep[0] = true;
    keep[far] = true;
    fn rec(p: &[DVec2], idx: &dyn Fn(usize) -> usize, lo: usize, hi: usize, tol: f64, keep: &mut [bool]) {
        if hi <= lo + 1 {
            return;
        }
        let (a, b) = (p[idx(lo)], p[idx(hi)]);
        let (k, d) = (lo + 1..hi).map(|k| (k, seg_dist(p[idx(k)], a, b))).max_by(|x, y| x.1.total_cmp(&y.1)).unwrap();
        if d > tol {
            keep[idx(k)] = true;
            rec(p, idx, lo, k, tol, keep);
            rec(p, idx, k, hi, tol, keep);
        }
    }
    let id = |i: usize| i % n;
    rec(p, &id, 0, far, tol, &mut keep);
    rec(p, &id, far, n, tol, &mut keep);
    let out: Vec<DVec2> = (0..n).filter(|&i| keep[i]).map(|i| p[i]).collect();
    if out.len() >= 3 {
        out
    } else {
        p.to_vec()
    }
}

// ---------------------------------------------------------------------------------------
// Sketches: strokes and fills outlined by marching squares
// ---------------------------------------------------------------------------------------

/// Round-capped strokes and filled polygons; [`Sketch::outline`] gives the border of their
/// union as stencil loops.
#[derive(Clone, Debug, Default)]
pub struct Sketch {
    strokes: Vec<(Vec<DVec2>, f64)>,
    fills: Vec<Vec<DVec2>>,
}

impl Sketch {
    /// A polyline of the given width with round ends and joins (one point draws a dot).
    pub fn stroke(mut self, line: Vec<DVec2>, width: f64) -> Self {
        if !line.is_empty() {
            self.strokes.push((line, width));
        }
        self
    }

    /// A filled polygon.
    pub fn fill(mut self, poly: Vec<DVec2>) -> Self {
        if poly.len() >= 3 {
            self.fills.push(poly);
        }
        self
    }

    /// Signed distance-like field: negative inside the union.
    fn field(&self, p: DVec2) -> f64 {
        let mut f = f64::INFINITY;
        for (line, w) in &self.strokes {
            let d = if line.len() == 1 { p.distance(line[0]) } else { line.windows(2).map(|s| seg_dist(p, s[0], s[1])).fold(f64::INFINITY, f64::min) };
            f = f.min(d - w * 0.5);
        }
        for poly in &self.fills {
            let d = (0..poly.len()).map(|i| seg_dist(p, poly[i], poly[(i + 1) % poly.len()])).fold(f64::INFINITY, f64::min);
            f = f.min(if point_in_polygon(p, poly) { -d } else { d });
        }
        f
    }

    /// The union's border, traced on a grid of spacing `step` and simplified to about a
    /// third of it: outer loops counter-clockwise, holes clockwise.
    pub fn outline(&self, step: f64) -> Vec<Vec<DVec2>> {
        let (mut lo, mut hi) = (DVec2::splat(f64::INFINITY), DVec2::splat(f64::NEG_INFINITY));
        let mut pad: f64 = 0.0;
        for (line, w) in &self.strokes {
            for p in line {
                lo = lo.min(*p);
                hi = hi.max(*p);
            }
            pad = pad.max(w * 0.5);
        }
        for p in self.fills.iter().flatten() {
            lo = lo.min(*p);
            hi = hi.max(*p);
        }
        if !lo.is_finite() {
            return Vec::new();
        }
        let origin = lo - DVec2::splat(pad + 2.0 * step);
        let nx = ((hi.x - lo.x + 2.0 * pad + 4.0 * step) / step).ceil() as usize + 1;
        let ny = ((hi.y - lo.y + 2.0 * pad + 4.0 * step) / step).ceil() as usize + 1;
        let at = |i: usize, j: usize| origin + DVec2::new(i as f64, j as f64) * step;
        let f: Vec<f64> = (0..ny).flat_map(|j| (0..nx).map(move |i| (i, j))).map(|(i, j)| self.field(at(i, j))).collect();
        let val = |i: usize, j: usize| f[j * nx + i];
        let mut loops = march(nx, ny, &val, &at);
        for l in &mut loops {
            *l = simplify_loop(l, step * 0.3);
        }
        loops.retain(|l| l.len() >= 3 && signed_area(l).abs() > step * step);
        loops
    }
}

/// Marching squares on a grid of values (inside where negative): closed loops with the
/// inside on their left.
fn march(nx: usize, ny: usize, val: &dyn Fn(usize, usize) -> f64, at: &dyn Fn(usize, usize) -> DVec2) -> Vec<Vec<DVec2>> {
    // Edge keys: horizontal edge from (i, j) to (i + 1, j) is 2·(j·nx + i), vertical edge
    // from (i, j) to (i, j + 1) is 2·(j·nx + i) + 1.
    let h = |i: usize, j: usize| 2 * (j * nx + i);
    let v = |i: usize, j: usize| 2 * (j * nx + i) + 1;
    let point = |key: usize| -> DVec2 {
        let (cell, vertical) = (key / 2, key % 2 == 1);
        let (i, j) = (cell % nx, cell / nx);
        let (i2, j2) = if vertical { (i, j + 1) } else { (i + 1, j) };
        let (f0, f1) = (val(i, j), val(i2, j2));
        let t = (f0 / (f0 - f1)).clamp(0.0, 1.0);
        at(i, j).lerp(at(i2, j2), t)
    };
    let mut next: HashMap<usize, usize> = HashMap::new();
    for j in 0..ny - 1 {
        for i in 0..nx - 1 {
            // Corners and edges counter-clockwise: bottom, right, top, left.
            let corner = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)];
            let inside = corner.map(|(a, b)| val(a, b) < 0.0);
            if inside.iter().all(|&x| x) || inside.iter().all(|&x| !x) {
                continue;
            }
            let edge = [h(i, j), v(i + 1, j), h(i, j + 1), v(i, j)];
            // Crossings in counter-clockwise order: (edge key, leaving the inside).
            let cross: Vec<(usize, bool)> = (0..4).filter(|&k| inside[k] != inside[(k + 1) % 4]).map(|k| (edge[k], inside[k])).collect();
            if cross.len() == 2 {
                let (s, e) = if cross[0].1 { (cross[0].0, cross[1].0) } else { (cross[1].0, cross[0].0) };
                next.insert(s, e);
            } else {
                // Saddle: the cell centre decides whether the inside corners connect.
                let centre_in = corner.iter().map(|&(a, b)| val(a, b)).sum::<f64>() < 0.0;
                for k in 0..4 {
                    if cross[k].1 {
                        let partner = if centre_in { cross[(k + 1) % 4].0 } else { cross[(k + 3) % 4].0 };
                        next.insert(cross[k].0, partner);
                    }
                }
            }
        }
    }
    let mut starts: Vec<usize> = next.keys().copied().collect();
    starts.sort_unstable();
    let mut seen = HashSet::new();
    let mut loops = Vec::new();
    for s in starts {
        if seen.contains(&s) {
            continue;
        }
        let mut lp = Vec::new();
        let mut k = s;
        while seen.insert(k) {
            lp.push(point(k));
            match next.get(&k) {
                Some(&n) => k = n,
                None => break,
            }
        }
        // Drop repeated points (a crossing exactly on a grid corner).
        lp.dedup_by(|a, b| a.distance(*b) < 1e-12);
        while lp.len() > 1 && lp[0].distance(lp[lp.len() - 1]) < 1e-12 {
            lp.pop();
        }
        if lp.len() >= 3 {
            loops.push(lp);
        }
    }
    loops
}

// ---------------------------------------------------------------------------------------
// Stamping a multi-loop stencil
// ---------------------------------------------------------------------------------------

/// What [`stamp_loops`] produced.
#[derive(Clone, Debug, Default)]
pub struct LoopsStampResult {
    /// Triangles inside the stencil (inside an odd number of its loops).
    pub inside: Vec<u32>,
    /// Each loop as a closed loop of vertices (counter-clockwise seen from the frame normal).
    pub outlines: Vec<Vec<u32>>,
}

/// Cut every loop of a stencil (stencil-plane coordinates) into the surface around the
/// triangle `seed`, as [`stamp_polygon`] does for one, and return the triangles inside by
/// the even-odd rule, so letters keep their counters and a ring its middle.
pub fn stamp_loops(mesh: &mut Mesh, seed: u32, frame: &StampFrame, loops: &[Vec<DVec2>]) -> EditResult<LoopsStampResult> {
    match loops {
        [] => Err(EditError::InvalidParameter("the stencil has no outline".into())),
        [one] => {
            let r = stamp_polygon(mesh, seed, frame, one)?;
            Ok(LoopsStampResult { inside: r.inside, outlines: vec![r.outline] })
        }
        _ => {
            let reach = loops.iter().flatten().map(|p| p.length()).fold(0.0, f64::max);
            let mut outlines = Vec::with_capacity(loops.len());
            for l in loops {
                outlines.push(stamp_polygon(mesh, seed, frame, l)?.outline);
            }
            // The insides are worked out at the end: later loops may have split triangles
            // (and, where two loops came within snapping distance, edges) of earlier ones.
            let mut count: HashMap<u32, u32> = HashMap::new();
            for o in &outlines {
                for t in loop_inside(mesh, o, frame, reach)? {
                    *count.entry(t).or_default() += 1;
                }
            }
            let mut inside: Vec<u32> = count.into_iter().filter(|&(_, c)| c % 2 == 1).map(|(t, _)| t).collect();
            inside.sort_unstable();
            Ok(LoopsStampResult { inside, outlines })
        }
    }
}

/// Vertices from `a` to `b` along mesh edges lying on the straight segment between them
/// (just `[a, b]` unless the edge was split since).
fn edge_path(mesh: &Mesh, a: u32, b: u32) -> Option<Vec<u32>> {
    let (pa, pb) = (mesh.vertex(a), mesh.vertex(b));
    let len = pa.distance(pb).max(1e-300);
    let on_segment = |p: glam::DVec3| {
        let t = ((p - pa).dot(pb - pa) / (len * len)).clamp(0.0, 1.0);
        p.distance(pa.lerp(pb, t)) < 1e-9 * len
    };
    let mut path = vec![a];
    let mut cur = a;
    for _ in 0..1024 {
        if mesh.find_edge(cur, b).is_some() {
            path.push(b);
            return Some(path);
        }
        let dc = mesh.vertex(cur).distance(pb);
        cur = mesh.vertex_neighbors(cur).filter(|&m| on_segment(mesh.vertex(m)) && mesh.vertex(m).distance(pb) < dc).min_by(|&x, &y| mesh.vertex(x).distance(pb).total_cmp(&mesh.vertex(y).distance(pb)))?;
        path.push(cur);
    }
    None
}

/// Triangles on the left of a (counter-clockwise) outline.
fn loop_inside(mesh: &Mesh, outline: &[u32], frame: &StampFrame, reach: f64) -> EditResult<Vec<u32>> {
    let lost = || EditError::Failed("a stamp outline was cut by another of its loops".into());
    let mut wall = HashSet::new();
    let mut first = None;
    for i in 0..outline.len() {
        let path = edge_path(mesh, outline[i], outline[(i + 1) % outline.len()]).ok_or_else(lost)?;
        for w in path.windows(2) {
            wall.insert(mesh.find_edge(w[0], w[1]).ok_or_else(lost)?);
        }
        first.get_or_insert((path[0], path[1]));
    }
    let (a, b) = first.ok_or_else(lost)?;
    let e = mesh.find_edge(a, b).ok_or_else(lost)?;
    let start = mesh.edge_triangles(e).into_iter().find(|&t| {
        t != mm_core::INVALID && {
            let tv = mesh.triangle(t);
            (0..3).any(|k| tv[k] == a && tv[(k + 1) % 3] == b)
        }
    });
    let Some(start) = start else { return Err(EditError::Failed("stamp outline has no inside".into())) };
    let limit = reach * 1.5 + 1e-9;
    let mut seen = HashSet::from([start]);
    let mut stack = vec![start];
    while let Some(t) = stack.pop() {
        if frame.project(mesh.tri_centroid(t)).length() > limit {
            return Err(EditError::Failed("the stamp outline does not close on the surface".into()));
        }
        for e in mesh.tri_edges(t) {
            let o = mesh.edge_other_triangle(e, t);
            if !wall.contains(&e) && o != mm_core::INVALID && seen.insert(o) {
                stack.push(o);
            }
        }
    }
    Ok(seen.into_iter().collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec3;
    use mm_core::primitives;

    fn assert_valid(s: &StencilShape) {
        validate_loops(&s.loops).unwrap_or_else(|e| panic!("{}: {e}", s.name));
        let r = s.loops.iter().flatten().map(|p| p.length()).fold(0.0, f64::max);
        assert!((r - 1.0).abs() < 1e-9, "{}: radius {r}", s.name);
        // Outer loops counter-clockwise, holes clockwise.
        for (i, l) in s.loops.iter().enumerate() {
            let depth = s.loops.iter().enumerate().filter(|&(j, o)| j != i && point_in_polygon(l[0], o)).count();
            assert_eq!(signed_area(l) > 0.0, depth % 2 == 0, "{} loop {i}", s.name);
        }
    }

    #[test]
    fn library_stencils_are_valid() {
        let all: Vec<StencilShape> = shape_stencils().into_iter().chain(icon_stencils()).collect();
        let mut names = HashSet::new();
        for s in &all {
            assert!(names.insert(s.name.clone()), "duplicate {}", s.name);
            assert_valid(s);
        }
        let by = |n: &str| all.iter().find(|s| s.name == n).unwrap().clone();
        // Holes where they belong.
        assert_eq!(by("Ring").loops.len(), 2);
        assert!(!point_in_loops(DVec2::ZERO, &by("Ring").loops));
        assert!(point_in_loops(DVec2::new(0.8, 0.0), &by("Ring").loops));
        assert!(point_in_loops(DVec2::ZERO, &by("Target").loops));
        assert!(!point_in_loops(DVec2::ZERO, &by("Gear").loops));
        assert_eq!(by("Curved arrow").loops.len(), 1, "the head joins the stroke");
    }

    #[test]
    fn every_glyph_is_a_valid_stencil() {
        for c in FONT_CHARS.chars().filter(|c| *c != ' ') {
            let loops = glyph_loops(c).unwrap_or_else(|| panic!("no glyph {c}"));
            validate_loops(&loops).unwrap_or_else(|e| panic!("{c}: {e}"));
            // Letters sit on the baseline and are one unit tall (strokes add their width).
            let (lo, hi) = loops.iter().flatten().fold((f64::MAX, f64::MIN), |(lo, hi), p| (lo.min(p.y), hi.max(p.y)));
            assert!(lo > -0.2 && hi < 1.2, "{c}: {lo}..{hi}");
        }
        for c in ('0'..='9').chain('A'..='Z') {
            assert!(FONT_CHARS.contains(c), "{c}");
        }
        // Counters: letters with holes have them, the others do not.
        let holes = |c: char| glyph_loops(c).unwrap().iter().filter(|l| signed_area(l) < 0.0).count();
        for (c, n) in [('A', 1), ('B', 2), ('D', 1), ('O', 1), ('P', 1), ('Q', 1), ('R', 1), ('0', 1), ('4', 1), ('6', 1), ('8', 2), ('9', 1), ('C', 0), ('E', 0), ('1', 0), ('7', 0), ('X', 0)] {
            assert_eq!(holes(c), n, "{c}");
        }
        for c in FONT_CHARS.chars().filter(|c| *c != ' ') {
            let outer = glyph_loops(c).unwrap().iter().filter(|l| signed_area(l) > 0.0).count();
            let want = if c == ':' { 2 } else { 1 };
            assert_eq!(outer, want, "{c} is drawn in one piece");
        }
    }

    #[test]
    fn text_lays_out_left_to_right() {
        let s = text_stencil("ica-12").unwrap();
        assert_eq!(s.name, "ICA-12");
        assert_valid(&s);
        assert!(s.cap_height.unwrap() > 0.1);
        // Six characters, each one piece.
        assert_eq!(s.loops.iter().filter(|l| signed_area(l) > 0.0).count(), 6);
        // Wider than tall.
        let (lo, hi) = s.loops.iter().flatten().fold((DVec2::MAX, DVec2::MIN), |(lo, hi), p| (lo.min(*p), hi.max(*p)));
        assert!(hi.x - lo.x > 3.0 * (hi.y - lo.y));
        assert!(text_stencil("  ").is_err());
        assert!(text_stencil("Ä").is_err());
    }

    #[test]
    fn sketch_outlines_a_dot_and_a_ring() {
        let dot = Sketch::default().stroke(vec![DVec2::ZERO], 1.0).outline(0.01);
        assert_eq!(dot.len(), 1);
        assert!((signed_area(&dot[0]) - std::f64::consts::PI * 0.25).abs() < 0.01);
        let ring = Sketch::default().stroke(circle(DVec2::ZERO, 1.0, 90).into_iter().chain([DVec2::X]).collect(), 0.2).outline(0.01);
        assert_eq!(ring.len(), 2);
        validate_loops(&ring).unwrap();
        let areas: Vec<f64> = ring.iter().map(|l| signed_area(l)).collect();
        assert!(areas.iter().any(|&a| a > 0.0) && areas.iter().any(|&a| a < 0.0), "{areas:?}");
    }

    #[test]
    fn simplify_keeps_corners() {
        let mut sq = Vec::new();
        for k in 0..4 {
            let (a, b) = (DVec2::from_angle(TAU * k as f64 / 4.0), DVec2::from_angle(TAU * (k + 1) as f64 / 4.0));
            for i in 0..10 {
                sq.push(a.lerp(b, i as f64 / 10.0));
            }
        }
        let s = simplify_loop(&sq, 1e-6);
        assert_eq!(s.len(), 4);
    }

    #[test]
    fn invalid_loops_are_reported() {
        let bow = pts(&[(0.0, 0.0), (1.0, 1.0), (1.0, 0.0), (0.0, 1.0)]);
        assert!(validate_loops(&[bow]).is_err());
        let a = circle(DVec2::ZERO, 1.0, 16);
        let b = circle(DVec2::new(1.5, 0.0), 1.0, 16);
        assert!(validate_loops(&[a.clone(), b]).is_err());
        assert!(validate_loops(&[a.clone(), circle(DVec2::ZERO, 0.5, 16)]).is_ok());
        assert!(validate_loops(&[pts(&[(0.0, 0.0), (1.0, 0.0)])]).is_err());
    }

    #[test]
    fn ring_and_text_stamp_with_holes() {
        // A ring on a plane: the middle stays outside the new group.
        let mut m = primitives::plane_grid(4.0, 4.0, 24, 24);
        let frame = StampFrame::new(DVec3::new(0.01, 0.02, 0.0), DVec3::Z, DVec3::Y);
        let seed = m.triangle_ids().find(|&t| m.tri_centroid(t).truncate().length() < 0.2).unwrap();
        let ring = shape_stencils().into_iter().find(|s| s.name == "Ring").unwrap();
        let loops: Vec<Vec<DVec2>> = ring.loops.iter().map(|l| l.iter().map(|p| *p * 1.5).collect()).collect();
        let r = stamp_loops(&mut m, seed, &frame, &loops).unwrap();
        m.check_validity().unwrap();
        assert_eq!(r.outlines.len(), 2);
        let area: f64 = r.inside.iter().map(|&t| m.tri_area(t)).sum();
        let want = std::f64::consts::PI * (1.5f64.powi(2) - 0.93f64.powi(2));
        assert!((area - want).abs() < 0.06 * want, "{area} vs {want}");
        assert!(r.inside.iter().all(|&t| frame.project(m.tri_centroid(t)).length() > 0.85));

        // Text on a sphere: letters with counters.
        let mut ball = primitives::icosphere(10.0, 5);
        let frame = StampFrame::new(DVec3::Z * 10.0, DVec3::Z, DVec3::Y);
        let seed = ball.triangle_ids().max_by(|&a, &b| ball.tri_centroid(a).z.total_cmp(&ball.tri_centroid(b).z)).unwrap();
        let text = text_stencil("AB8").unwrap();
        let loops: Vec<Vec<DVec2>> = text.loops.iter().map(|l| l.iter().map(|p| *p * 5.0).collect()).collect();
        let r = stamp_loops(&mut ball, seed, &frame, &loops).unwrap();
        ball.check_validity().unwrap();
        assert!(ball.is_closed());
        let proj: f64 = r.inside.iter().map(|&t| ball.tri_normal(t).dot(DVec3::Z) * ball.tri_area(t)).sum();
        let want: f64 = loops.iter().map(|l| signed_area(l)).sum();
        assert!((proj - want).abs() < 0.1 * want, "{proj} vs {want}");
        // No inside triangle sits in a counter.
        for &t in &r.inside {
            let c = frame.project(ball.tri_centroid(t));
            let deep_out = !point_in_loops(c, &loops) && loops.iter().flat_map(|l| (0..l.len()).map(move |i| seg_dist(c, l[i], l[(i + 1) % l.len()]))).fold(f64::MAX, f64::min) > 0.1;
            assert!(!deep_out, "{c:?}");
        }
    }
}
