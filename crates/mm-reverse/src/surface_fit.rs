// SPDX-License-Identifier: AGPL-3.0-only
//! Fit Surface: a smooth B-spline through a region of the scan. Ported from ScanRuler
//! (`plugins/reverse/core/surfaceFit.ts`, and `SurfaceFitParams` from
//! `plugins/reverse/core/features.ts`), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! A scanned freeform patch is a cloud of noisy points; what the trim wants is one smooth
//! surface that reaches a little past them. The fit here is the standard one: the region's
//! principal frame is the base plane, every point becomes (u, v, height) over it, and a
//! tensor-product B-spline of degree 3 with a U × V control net is solved for its heights by
//! linear least squares — the net's u and v are fixed at the Greville abscissae, so the
//! surface's own (x, y) reproduce the plane's (u, v) exactly and only the heights are
//! unknown. A fairness term on the net keeps the surface fair where the region is thin or
//! noisy; its weight is the smoothness slider times the region's area, so one setting means
//! the same on a 10 mm boss and a 200 mm panel. Dense Cholesky is enough: the largest net in
//! use is 20 × 20, four hundred unknowns.
//!
//! The height field is the limitation: a patch that folds back over its plane — more than
//! half a cylinder — is not one surface over it. Those are primitives, and Measure fits
//! them; this is for the rest.
//!
//! The kernel does not take a control net. ScanRuler's `bsplineSurface` is Open CASCADE's
//! `GeomAPI_PointsToBSplineSurface`: a surface *through a regular grid of points*, within a
//! thousandth of a millimetre. So the fitted surface is sampled on a grid a few times denser
//! than its net (`grid_of`) and that grid is what the kernel gets — the fit does the
//! smoothing of the scan's noise, the kernel makes the exact B-rep surface through the fit.

use std::collections::{HashMap, HashSet};
use std::hash::Hash;

use glam::{DVec2, DVec3};
use serde::{Deserialize, Serialize};

use crate::fit2d::FitError;
use crate::linalg::{cholesky_solve, solve_linear};
use crate::principal_frame::{js_cmp, js_hypot3, principal_frame, symmetric_eigen3};

/// The smallest and largest net per side; Auto chooses within them.
pub const NET_MIN: usize = 3;
pub const NET_MAX: usize = 20;
/// How many grid samples per net span go to the kernel. Three per span puts the grid well
/// inside what a cubic through it can carry without wobbling between the samples.
pub const GRID_PER_SPAN: usize = 3;
/// The denser grids tried when the kernel's surface through the first does not follow it.
/// The kernel approximates the grid, up to degree 8, and on some grids that comes back
/// swinging between the samples: the water pump's underside at three a span came back with
/// twice the area and 50 mm off the fit, meshed to 780 k triangles; at two a span, 320 m
/// tall. Five a span and more followed it within a micron every time.
pub const GRID_RETRIES: [usize; 2] = [6, 8];
/// How much more area than the grid's own facets the kernel's surface through it may have
/// and still follow it: a smooth fit sampled three a span has well under 0.1 % more; a
/// surface swinging off it, from 50 % to a millionfold.
pub const GRID_AREA_SLACK: f64 = 1.02;
/// How many more triangles the kernel's mesher lays than the fewest of the edge it may lay
/// (`mesh_triangles`).
const MESH_SURPLUS: f64 = 3.0;
/// What a slider of 1 means against the data, chosen so that the default of 0.01 moves a
/// 30 mm cylinder fitted through 2,400 points by microns and still holds a 20 × 20 net over
/// 121 noisy points to a fair plane.
const SMOOTHNESS_SCALE: f64 = 1e-2;
/// The net of Auto's pilot fit, per side.
const PILOT_NET: usize = 6;
/// The pull toward the region's plane in the cells with no points within reach
/// (`fit_surface`): its weight, as a share of the points of an average cell with points; how
/// many cells off a point a cell is still within reach; and the samples a side that stand
/// in for points in a pulled cell.
pub const PULL: f64 = 0.01;
const PULL_REACH: usize = 1;
const PULL_SAMPLES: usize = 3;
/// How far the pull may move the surface at a point, as a share of the RMS the fit without
/// it leaves — never held under PULL_FLOOR mm, the noise of a clean CAD tessellation being
/// next to none — and how many weights it tries for that.
const PULL_MOVE: f64 = 0.5;
const PULL_FLOOR: f64 = 0.01;
const PULL_ROUNDS: usize = 3;

/// A control net's size, U × V.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Net {
    pub u: usize,
    pub v: usize,
}

fn is_false(b: &bool) -> bool {
    !*b
}

/// How a surface is fitted through a region (ScanRuler's `SurfaceFitParams`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SurfaceFitParams {
    /// The control net, U × V, or None for the fit to choose from the region's extent and
    /// scatter.
    pub net: Option<Net>,
    pub degree: u32,
    /// How far past the region's bounds the surface reaches, as a share of its extent — a
    /// surface is trimmed against its neighbours later and has to reach them.
    pub expand_pct: f64,
    /// The weight of the fairness term relative to the data's, so the surface stays fair
    /// where the region is thin or noisy.
    pub smoothness: f64,
    /// Fit again without the points that stand far off the first fit — a rib or a stray edge
    /// caught in the region. Absent is false.
    #[serde(default, skip_serializing_if = "is_false")]
    pub robust: bool,
    /// The height direction the surface is fitted along, any length — the way the region
    /// faces. Absent: the region's own principal normal.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normal: Option<DVec3>,
}

/// The defaults a new Fit Surface opens with — the interview's numbers.
pub const DEFAULT_SURFACE_FIT: SurfaceFitParams =
    SurfaceFitParams { net: None, degree: 3, expand_pct: 35.0, smoothness: 0.01, robust: false, normal: None };

impl Default for SurfaceFitParams {
    fn default() -> Self {
        DEFAULT_SURFACE_FIT
    }
}

/// A fitted surface.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SurfaceFitResult {
    /// The base frame: origin, the two in-plane axes and the normal.
    pub origin: DVec3,
    pub axis_u: DVec3,
    pub axis_v: DVec3,
    pub normal: DVec3,
    /// The (u, v) range the surface covers, the margin included.
    pub u_range: [f64; 2],
    pub v_range: [f64; 2],
    pub degree: usize,
    /// The clamped knot vectors and the net's heights, row-major (u rows).
    pub knots_u: Vec<f64>,
    pub knots_v: Vec<f64>,
    pub nu: usize,
    pub nv: usize,
    pub heights: Vec<f64>,
    /// The fit's residuals in the plane's normal: RMS and largest, mm.
    pub rms: f64,
    pub max: f64,
    /// The net Auto chose, or the one asked for.
    pub net: Net,
    /// The region's points over the plane, (u, v) pairs: where the surface has the scan
    /// under it as fitted — what is read past it, and what a split within the region cuts
    /// inside (`surface_region`). Set by `fit_surface_reported`; absent on a fit made
    /// otherwise. Held at single precision, as ScanRuler holds it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub footprint: Option<Vec<DVec2>>,
}

/// The plane a surface is fitted over and the (u, v) it spans: what `footprint_of` and
/// `reach_over` read of a fit (ScanRuler's `Pick<SurfaceFitResult, …>`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfacePlane {
    pub origin: DVec3,
    pub axis_u: DVec3,
    pub axis_v: DVec3,
    pub u_range: [f64; 2],
    pub v_range: [f64; 2],
}

impl SurfaceFitResult {
    /// The plane the surface stands over, and its range.
    pub fn plane(&self) -> SurfacePlane {
        SurfacePlane { origin: self.origin, axis_u: self.axis_u, axis_v: self.axis_v, u_range: self.u_range, v_range: self.v_range }
    }
}

// ---- JavaScript's number semantics, where the port has to keep them ---------------------

/// JavaScript's `Math.round`: halves go up, toward +∞.
pub(crate) fn js_round(x: f64) -> f64 {
    let f = x.floor();
    if x - f >= 0.5 {
        f + 1.0
    } else {
        f
    }
}

/// A number as JavaScript writes it in a template string: the shortest digits that read
/// back, −0 as 0, exponents past 1e21 and under 1e-6.
pub(crate) fn js_num(x: f64) -> String {
    if x.is_nan() {
        return "NaN".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    if x == 0.0 {
        return "0".into();
    }
    let a = x.abs();
    if !(1e-6..1e21).contains(&a) {
        let s = format!("{x:e}");
        return match s.split_once('e') {
            Some((m, e)) if !e.starts_with('-') => format!("{m}e+{e}"),
            _ => s,
        };
    }
    format!("{x}")
}

/// JavaScript's `toFixed(1)`: an exact tie goes to the larger digit, where Rust's formatting
/// takes the even one; and from 1e21 up the number is written as `String(x)` writes it,
/// exponent and all.
pub(crate) fn to_fixed1(x: f64) -> String {
    if !x.is_finite() {
        return js_num(x);
    }
    let neg = x < 0.0;
    let a = x.abs();
    if a >= 1e21 {
        return js_num(x);
    }
    // A double lies exactly halfway between two tenths only as an odd number of quarters
    // (k + 0.25 or k + 0.75): the tenth above is then 5q / 2 tenths rounded up, q the
    // quarters — counted in integers, which the products a·4 (exact) and q under 4e21 allow.
    let q = a * 4.0;
    let body = if q.fract() == 0.0 && (q as u128) % 2 == 1 {
        let n = (5 * q as u128).div_ceil(2);
        format!("{}.{}", n / 10, n % 10)
    } else {
        format!("{a:.1}")
    };
    if neg {
        format!("-{body}")
    } else {
        body
    }
}

/// A map that iterates in the order its keys were first set, as JavaScript's `Map` does.
#[derive(Debug, Clone)]
pub(crate) struct OrderedMap<K, V> {
    index: HashMap<K, usize>,
    entries: Vec<(K, V)>,
}

impl<K: Hash + Eq + Clone, V> OrderedMap<K, V> {
    pub(crate) fn new() -> Self {
        OrderedMap { index: HashMap::new(), entries: Vec::new() }
    }
    pub(crate) fn get(&self, k: &K) -> Option<&V> {
        self.index.get(k).map(|&i| &self.entries[i].1)
    }
    pub(crate) fn contains_key(&self, k: &K) -> bool {
        self.index.contains_key(k)
    }
    /// Sets a key's value; a key set before keeps its place.
    pub(crate) fn insert(&mut self, k: K, v: V) {
        match self.index.get(&k) {
            Some(&i) => self.entries[i].1 = v,
            None => {
                self.index.insert(k.clone(), self.entries.len());
                self.entries.push((k, v));
            }
        }
    }
    /// The value of a key, made with `make` where it has none yet.
    pub(crate) fn get_or_insert_with(&mut self, k: K, make: impl FnOnce() -> V) -> &mut V {
        let i = match self.index.get(&k) {
            Some(&i) => i,
            None => {
                self.index.insert(k.clone(), self.entries.len());
                self.entries.push((k, make()));
                self.entries.len() - 1
            }
        };
        &mut self.entries[i].1
    }
    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    pub(crate) fn keys(&self) -> Vec<K> {
        self.entries.iter().map(|(k, _)| k.clone()).collect()
    }
    pub(crate) fn iter(&self) -> impl Iterator<Item = &(K, V)> {
        self.entries.iter()
    }
    pub(crate) fn iter_mut(&mut self) -> impl Iterator<Item = &mut (K, V)> {
        self.entries.iter_mut()
    }
}

/// A vector's length as ScanRuler reads it, `Math.hypot`'s to the last bit.
fn hypot3(v: DVec3) -> f64 {
    js_hypot3(v)
}

// ---- the B-spline pieces ---------------------------------------------------------------

/// A clamped uniform knot vector of degree `p` with `n` control points over [a, b].
pub fn clamped_knots(n: usize, p: usize, a: f64, b: f64) -> Vec<f64> {
    let m = n + p + 1;
    let mut knots = vec![0.0; m];
    let inner = n as isize - p as isize - 1;
    for k in knots.iter_mut().take(p + 1) {
        *k = a;
    }
    for i in 1..=inner.max(0) as usize {
        knots[p + i] = a + ((b - a) * i as f64) / (inner + 1) as f64;
    }
    for k in knots.iter_mut().skip(m - p - 1) {
        *k = b;
    }
    knots
}

/// The Greville abscissae: where a control point "is" along the axis; the B-spline through
/// them reproduces a straight line exactly.
pub fn greville(knots: &[f64], n: usize, p: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let mut s = 0.0;
            for k in 1..=p {
                s += knots[i + k];
            }
            s / p as f64
        })
        .collect()
}

/// The knot span holding t, so that knots[span] <= t < knots[span + 1], with t at the end
/// taken into the last span.
pub fn find_span(knots: &[f64], n: usize, p: usize, t: f64) -> usize {
    if t >= knots[n] {
        return n.saturating_sub(1);
    }
    if t <= knots[p] {
        return p;
    }
    let mut lo = p;
    let mut hi = n;
    while hi - lo > 1 {
        let mid = (lo + hi) >> 1;
        if t < knots[mid] {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    lo
}

/// The p + 1 basis functions that are non-zero at t, in the span — the Cox–de Boor
/// recurrence. `out` holds at least p + 1.
pub fn basis_functions(knots: &[f64], span: usize, p: usize, t: f64, out: &mut [f64]) {
    let mut left = vec![0.0; p + 1];
    let mut right = vec![0.0; p + 1];
    out[0] = 1.0;
    for j in 1..=p {
        left[j] = t - knots[span + 1 - j];
        right[j] = knots[span + j] - t;
        let mut saved = 0.0;
        for r in 0..j {
            let temp = out[r] / (right[r + 1] + left[j - r]);
            out[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        out[j] = saved;
    }
}

/// The net Auto chooses: the region's extent over eight times its scatter, per side, within
/// the limits — enough spans to follow the shape the points resolve, not so many that the
/// net fits the noise.
pub fn auto_net(extent_u: f64, extent_v: f64, scatter: f64) -> Net {
    let per = |extent: f64| js_round(extent / (8.0 * scatter.max(1e-3))).min(NET_MAX as f64).max(NET_MIN as f64) as usize;
    Net { u: per(extent_u), v: per(extent_v) }
}

/// The plane a surface is fitted over: the region's principal frame, or — given a height
/// direction — that direction as its normal, the in-plane axes the principal ones laid
/// square to it.
pub fn height_frame(axes: &[DVec3; 3], normal: Option<DVec3>) -> [DVec3; 3] {
    let len = normal.map(hypot3).unwrap_or(0.0);
    let Some(normal) = normal.filter(|_| len > 1e-12) else {
        return *axes;
    };
    let n = DVec3::new(normal.x / len, normal.y / len, normal.z / len);
    let square = |a: DVec3| {
        let d = a.x * n.x + a.y * n.y + a.z * n.z;
        DVec3::new(a.x - d * n.x, a.y - d * n.y, a.z - d * n.z)
    };
    // The longer principal axis, unless it runs nearly along the normal.
    let mut u = square(axes[0]);
    if hypot3(u) < 0.3 {
        u = square(axes[1]);
    }
    let lu = hypot3(u);
    u = DVec3::new(u.x / lu, u.y / lu, u.z / lu);
    let v = DVec3::new(n.y * u.z - n.z * u.y, n.z * u.x - n.x * u.z, n.x * u.y - n.y * u.x);
    [u, v, n]
}

// ---- the fit ---------------------------------------------------------------------------

/// A surface fitted through the points. `scatter` is the scan's own noise in millimetres,
/// for Auto's net; with None, a pilot fit reads it (or, with a net given, the plane
/// residual's spread is used). `pull` is the pull toward the plane the cells with no points
/// near start from (`PULL`; 0 fits without it).
pub fn fit_surface(xyz: &[DVec3], params: &SurfaceFitParams, scatter: Option<f64>, pull: f64) -> Result<SurfaceFitResult, FitError> {
    let n = xyz.len();
    if n < 4 {
        return Err(FitError::new("Too few points to fit a surface through."));
    }
    let frame = principal_frame(xyz);
    let [au, av, an] = height_frame(&frame.axes, params.normal);
    let c = frame.centroid;
    let p = params.degree.clamp(1, 5) as usize;

    // Every point over the plane.
    let mut u = vec![0.0; n];
    let mut v = vec![0.0; n];
    let mut h = vec![0.0; n];
    let (mut u_min, mut u_max, mut v_min, mut v_max) = (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
    for (i, q) in xyz.iter().enumerate() {
        let (x, y, z) = (q.x - c.x, q.y - c.y, q.z - c.z);
        u[i] = x * au.x + y * au.y + z * au.z;
        v[i] = x * av.x + y * av.y + z * av.z;
        h[i] = x * an.x + y * an.y + z * an.z;
        if u[i] < u_min {
            u_min = u[i];
        }
        if u[i] > u_max {
            u_max = u[i];
        }
        if v[i] < v_min {
            v_min = v[i];
        }
        if v[i] > v_max {
            v_max = v[i];
        }
    }
    let extent_u = (u_max - u_min).max(1e-6);
    let extent_v = (v_max - v_min).max(1e-6);
    // The margin: the surface reaches past the region by a share of its extent, half on
    // each side.
    let margin = params.expand_pct.max(0.0) / 200.0;
    let u_range = [u_min - extent_u * margin, u_max + extent_u * margin];
    let v_range = [v_min - extent_v * margin, v_max + extent_v * margin];

    // Auto's net wants the scan's noise, which the plane residual's spread is not on a
    // curved patch — there it is the sag. With no scatter given, a pilot fit on a coarse net
    // reads the noise off its own residual: what a coarse smooth surface cannot follow is
    // noise, or detail Auto then gives the net enough spans to follow.
    let noise = match (scatter, params.net) {
        (Some(s), _) => s,
        (None, Some(_)) => frame.spread[2].sqrt(),
        (None, None) => {
            let pilot = SurfaceFitParams { net: Some(Net { u: PILOT_NET, v: PILOT_NET }), ..params.clone() };
            fit_surface(xyz, &pilot, Some(0.0), 0.0)?.rms
        }
    };
    let net = match params.net {
        Some(net) => Net { u: clamp_net(net.u, p), v: clamp_net(net.v, p) },
        None => auto_net(extent_u, extent_v, noise),
    };
    let nu = net.u.max(p + 1);
    let nv = net.v.max(p + 1);
    let knots_u = clamped_knots(nu, p, u_range[0], u_range[1]);
    let knots_v = clamped_knots(nv, p, v_range[0], v_range[1]);

    // Normal equations over the heights, assembled point by point from the (p + 1)² basis
    // products each point touches.
    let m = nu * nv;
    let mut ata = vec![0.0; m * m];
    let mut atb = vec![0.0; m];
    let mut bu = vec![0.0; p + 1];
    let mut bv = vec![0.0; p + 1];
    let mut cols = vec![0usize; (p + 1) * (p + 1)];
    let mut vals = vec![0.0; (p + 1) * (p + 1)];
    for i in 0..n {
        let su = find_span(&knots_u, nu, p, u[i]);
        let sv = find_span(&knots_v, nv, p, v[i]);
        basis_functions(&knots_u, su, p, u[i], &mut bu);
        basis_functions(&knots_v, sv, p, v[i], &mut bv);
        let mut k = 0;
        for a in 0..=p {
            let row = (su - p + a) * nv;
            for b in 0..=p {
                cols[k] = row + (sv - p + b);
                vals[k] = bu[a] * bv[b];
                k += 1;
            }
        }
        for a in 0..k {
            let ca = cols[a];
            let va = vals[a];
            atb[ca] += va * h[i];
            let row_a = ca * m;
            for b in 0..k {
                ata[row_a + cols[b]] += va * vals[b];
            }
        }
    }

    // The fairness term. A thin-plate energy (second differences) was tried first and
    // thrown out: it prefers a plane, so with no data in the margin it flattened the
    // extrapolation there — a 30 mm cylinder's reach was 0.7 mm off at the margin's edge,
    // where the surface has to meet its neighbour for the trim. What is penalised instead is
    // the *change* of curvature: third divided differences of the net's heights over the
    // Greville abscissae (divided, because a clamped net's ends are not evenly spaced and a
    // plain difference would bend a quadratic there), along u, along v and mixed, each
    // weighted by its cell's area. Their null space holds every quadratic, so a cylinder, a
    // sphere or a saddle extrapolates as itself, while noise and a net finer than its points
    // are still faired. The energy has the units of 1/length², so the weight is the slider
    // times the region's area squared: a slider of 0.01 means the same on a 10 mm boss and a
    // 200 mm panel.
    let lambda = params.smoothness.max(0.0) * (extent_u * extent_v).powi(2) * SMOOTHNESS_SCALE;
    let xu = greville(&knots_u, nu, p);
    let xv = greville(&knots_v, nv, p);
    let area = ((u_range[1] - u_range[0]) / (nu.max(2) - 1) as f64) * ((v_range[1] - v_range[0]) / (nv.max(2) - 1) as f64);
    let at = |i: usize, j: usize| i * nv + j;
    // A stencil that is the product of a divided difference along u and one along v, over
    // the net's block at (i, j).
    let mut cross = |i: usize, j: usize, wu: &[f64], wv: &[f64], scale: f64| {
        let mut idx = Vec::with_capacity(wu.len() * wv.len());
        let mut w = Vec::with_capacity(wu.len() * wv.len());
        for (a, &x) in wu.iter().enumerate() {
            for (b, &y) in wv.iter().enumerate() {
                idx.push(at(i + a, j + b));
                w.push(x * y);
            }
        }
        for a in 0..idx.len() {
            for b in 0..idx.len() {
                ata[idx[a] * m + idx[b]] += lambda * scale * w[a] * w[b];
            }
        }
    };
    for i in 0..nu {
        for j in 0..nv {
            if i + 3 < nu {
                cross(i, j, &divided_weights(&xu, i, 3), &[1.0], area);
            }
            if j + 3 < nv {
                cross(i, j, &[1.0], &divided_weights(&xv, j, 3), area);
            }
            if i + 2 < nu && j + 1 < nv {
                cross(i, j, &divided_weights(&xu, i, 2), &divided_weights(&xv, j, 1), 3.0 * area);
            }
            if i + 1 < nu && j + 2 < nv {
                cross(i, j, &divided_weights(&xu, i, 1), &divided_weights(&xv, j, 2), 3.0 * area);
            }
        }
    }
    // A whisper on the diagonal keeps a net span no point falls in — out in the margin —
    // from making the system singular where the bending term alone does not reach it.
    let mut diag = 0.0;
    for i in 0..m {
        diag += ata[i * m + i];
    }
    for i in 0..m {
        ata[i * m + i] += 1e-12 * (diag / m as f64).max(1e-12);
    }
    // The heights with `extra` added to the system `weight` times. The solvers work on a
    // copy: the system is solved again with the pull.
    let solve = |extra: Option<&[f64]>, weight: f64| -> Option<Vec<f64>> {
        let mut a = ata.clone();
        if let Some(extra) = extra {
            for (x, e) in a.iter_mut().zip(extra) {
                *x += weight * e;
            }
        }
        cholesky_solve(m, &a, &atb, 1e-12).or_else(|| solve_linear(m, &mut a, &atb))
    };
    let mut heights = solve(None, 0.0).ok_or_else(|| FitError::new("The surface fit is singular — the region is too thin for this net."))?;

    // The pull. The fairness term leaves every quadratic free, so a cell with no points near
    // it goes on as whatever the last points' edge bent into: a patchy region — a circle in
    // its square box, a boss the robust fit dropped — had its empty corners curl 100 mm off a
    // part 20 mm tall. The cells more than PULL_REACH cells from every cell with a point are
    // pulled toward the region's plane (height 0), by samples standing in for points: `pull`
    // times the points of an average cell with points between them to begin with, and less
    // where that moves the surface at any point by more than PULL_MOVE of the residual the
    // fit leaves anyway — the cells with points stay where the points put them, within their
    // noise, and the empty ones settle back toward the plane within a few cells. When no
    // weight tried does that, the fit goes without the pull, and `run_away` is the backstop.
    let toward_plane = if pull > 0.0 { pull_system(&u, &v, &knots_u, &knots_v, nu, nv, p) } else { None };
    if let Some((system, cells)) = toward_plane {
        let free = heights.clone();
        let mut diff = vec![0.0; m];
        let mut sq = 0.0;
        for i in 0..n {
            sq += (h[i] - eval_height(&knots_u, &knots_v, nu, nv, p, &free, u[i], v[i], &mut bu, &mut bv)).powi(2);
        }
        let allowed = (PULL_MOVE * (sq / n as f64).sqrt()).max(PULL_FLOOR);
        let mut weight = pull * (n as f64 / cells.max(1) as f64);
        for _round in 0..PULL_ROUNDS {
            let Some(held) = solve(Some(&system), weight) else { break };
            for k in 0..m {
                diff[k] = held[k] - free[k];
            }
            let mut moved = 0.0f64;
            for i in 0..n {
                moved = moved.max(eval_height(&knots_u, &knots_v, nu, nv, p, &diff, u[i], v[i], &mut bu, &mut bv).abs());
            }
            if moved <= allowed {
                heights = held;
                break;
            }
            // The surface moves far less than in proportion to the weight — about as its
            // fourth root, from 1e-4 to 0.1 on a bowl — so the weight comes down by the
            // overshoot's fourth power. (`**` is V8's fdlibm pow; libm's `powf` agrees with it
            // far more often than `powi`'s two squarings, which are off in the last bit for
            // half of all inputs.)
            weight *= ((0.8 * allowed) / moved).powf(4.0);
        }
    }

    // Residuals along the normal.
    let mut sum_sq = 0.0;
    let mut max = 0.0f64;
    for i in 0..n {
        let r = h[i] - eval_height(&knots_u, &knots_v, nu, nv, p, &heights, u[i], v[i], &mut bu, &mut bv);
        sum_sq += r * r;
        if r.abs() > max {
            max = r.abs();
        }
    }
    Ok(SurfaceFitResult {
        origin: c,
        axis_u: au,
        axis_v: av,
        normal: an,
        u_range,
        v_range,
        degree: p,
        knots_u,
        knots_v,
        nu,
        nv,
        heights,
        rms: (sum_sq / n as f64).sqrt(),
        max,
        net: Net { u: nu, v: nv },
        footprint: None,
    })
}

/// How many of a fit's worst points its report names.
const WORST_POINTS: usize = 5;
/// A robust fit leaves out the points further off than this many robust sigmas (1.4826
/// times the residuals' median absolute deviation), and never closer than ROBUST_FLOOR mm —
/// a clean CAD tessellation has next to no scatter to measure by. It fits again at most
/// ROBUST_ROUNDS times.
const ROBUST_SIGMAS: f64 = 3.0;
const ROBUST_FLOOR: f64 = 0.01;
const ROBUST_ROUNDS: usize = 3;

/// Each point's distance from the fitted surface along its normal, signed (positive on the
/// normal's side), mm.
pub fn surface_residuals(fit: &SurfaceFitResult, xyz: &[DVec3]) -> Vec<f64> {
    let mut bu = vec![0.0; fit.degree + 1];
    let mut bv = vec![0.0; fit.degree + 1];
    xyz.iter()
        .map(|q| {
            let (x, y, z) = (q.x - fit.origin.x, q.y - fit.origin.y, q.z - fit.origin.z);
            let uu = x * fit.axis_u.x + y * fit.axis_u.y + z * fit.axis_u.z;
            let vv = x * fit.axis_v.x + y * fit.axis_v.y + z * fit.axis_v.z;
            let hh = x * fit.normal.x + y * fit.normal.y + z * fit.normal.z;
            hh - eval_height(&fit.knots_u, &fit.knots_v, fit.nu, fit.nv, fit.degree, &fit.heights, uu, vv, &mut bu, &mut bv)
        })
        .collect()
}

/// The way a region was read off the scan facing (region.normal), where it was: its normals
/// lie within `within_deg` of `dir` by construction — 30° where not given, as the region's
/// query has it.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RegionFacing {
    pub dir: DVec3,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub within_deg: Option<f64>,
}
const FACING_DEG: f64 = 30.0;

/// A box in the part's frame, mm.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Bounds {
    pub min: DVec3,
    pub max: DVec3,
}

/// Points of a region the fit could not take as its surface's, named so they can be left
/// out: a second sheet over or under the one fitted along its height direction, or a group
/// of the points a robust fit left out far off it. How many of the region's points, the box
/// round them — wide enough to give back as one of region.exclude's boxes — how far off the
/// surface they lie, mm, least and most, and why, in words.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Stray {
    pub points: usize,
    #[serde(rename = "box")]
    pub bounds: Bounds,
    pub off_mm: [f64; 2],
    pub why: String,
}

/// A refusal of a fit that names its strays.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("{message}")]
pub struct StraysRefusal {
    pub message: String,
    pub strays: Vec<Stray>,
}

/// Why a reported fit was not made: the fit itself failed, or it ran away and is refused
/// with the strays it names.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum SurfaceFitError {
    #[error(transparent)]
    Fit(#[from] FitError),
    #[error(transparent)]
    Strays(#[from] StraysRefusal),
}

impl SurfaceFitError {
    /// The strays a refusal names, if it names any.
    pub fn strays(&self) -> Option<&[Stray]> {
        match self {
            SurfaceFitError::Strays(r) => Some(&r.strays),
            SurfaceFitError::Fit(_) => None,
        }
    }
}

/// One of the points a fitted surface misses most, and by how much (signed, mm).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WorstPoint {
    pub at: DVec3,
    pub off: f64,
}

/// A robust fit's last centre and limit: what its dropped points are told by.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RobustAt {
    pub med: f64,
    pub limit: f64,
}

/// A surface fitted through a region, with what an answer says of it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SurfaceReport {
    pub fit: SurfaceFitResult,
    pub worst: Vec<WorstPoint>,
    pub dropped: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strays: Option<Vec<Stray>>,
}

/// A surface fitted through the points as `fit_surface` fits it, with what an answer says
/// of it: the worst points the surface misses, and — asked for a robust fit
/// (params.robust) — how many points it left out. A robust fit fits, leaves out the points
/// past three robust sigmas, and fits the rest again, until nothing more is left out: a rib,
/// a boss or a ragged edge caught in the region does not pull the surface off the skin. A
/// fit that runs away, and a robust fit, name the strays (`stray_groups`) — a fit that runs
/// away is refused with them (`StraysRefusal`). `facing` is the way the region was read
/// facing, where it was.
pub fn fit_surface_reported(xyz: &[DVec3], params: &SurfaceFitParams, facing: Option<&RegionFacing>) -> Result<SurfaceReport, SurfaceFitError> {
    let mut points: Vec<DVec3> = xyz.to_vec();
    let mut fit = fit_surface(&points, params, None, PULL)?;
    let mut dropped = 0;
    // The last round's centre and limit: what the robust fit's dropped points are told by.
    let mut robust_at: Option<RobustAt> = None;
    for _round in 0..ROBUST_ROUNDS {
        if !params.robust {
            break;
        }
        let r = surface_residuals(&fit, &points);
        let mut sorted = r.clone();
        sorted.sort_by(|a, b| js_cmp(*a, *b));
        let med = sorted.get(sorted.len() >> 1).copied().unwrap_or(f64::NAN);
        let mut dev: Vec<f64> = r.iter().map(|x| (x - med).abs()).collect();
        dev.sort_by(|a, b| js_cmp(*a, *b));
        let limit = (ROBUST_SIGMAS * 1.4826 * dev.get(dev.len() >> 1).copied().unwrap_or(f64::NAN)).max(ROBUST_FLOOR);
        robust_at = Some(RobustAt { med, limit });
        let keep: Vec<usize> = (0..r.len()).filter(|&i| (r[i] - med).abs() <= limit).collect();
        if keep.len() == r.len() || keep.len() < 16 {
            break;
        }
        let next: Vec<DVec3> = keep.iter().map(|&i| points[i]).collect();
        dropped += r.len() - keep.len();
        points = next;
        fit = fit_surface(&points, params, None, PULL)?;
    }
    if let Some(away) = run_away(&fit, xyz, params.expand_pct, params.normal, facing) {
        let strays = stray_groups(&fit, xyz, robust_at);
        return Err(StraysRefusal { message: format!("{away}{}", strays_words(&strays)), strays }.into());
    }
    let r = surface_residuals(&fit, &points);
    let mut order: Vec<usize> = (0..r.len()).collect();
    order.sort_by(|&a, &b| js_cmp(r[b].abs(), r[a].abs()));
    let worst = order.iter().take(WORST_POINTS).map(|&i| WorstPoint { at: points[i], off: r[i] }).collect();
    let strays = if params.robust { stray_groups(&fit, xyz, robust_at) } else { Vec::new() };
    // The whole region, the points a robust fit left out among them: its outline is the
    // region's as it was marked.
    fit.footprint = Some(footprint_of(&fit.plane(), xyz));
    Ok(SurfaceReport { fit, worst, dropped, strays: if strays.is_empty() { None } else { Some(strays) } })
}

/// The points over a fit's plane, (u, v) pairs (`SurfaceFitResult::footprint`), held at
/// single precision as ScanRuler holds them.
pub fn footprint_of(plane: &SurfacePlane, xyz: &[DVec3]) -> Vec<DVec2> {
    xyz.iter()
        .map(|q| {
            let (x, y, z) = (q.x - plane.origin.x, q.y - plane.origin.y, q.z - plane.origin.z);
            let u = x * plane.axis_u.x + y * plane.axis_u.y + z * plane.axis_u.z;
            let v = x * plane.axis_v.x + y * plane.axis_v.y + z * plane.axis_v.z;
            DVec2::new(u as f32 as f64, v as f32 as f64)
        })
        .collect()
}

/// The cells a second sheet is looked for in are about this wide, mm — and never narrower
/// than twice the points' spacing, so each holds a few — and a cell holds two sheets where
/// its points' heights part by more than SHEET_GAP_MM. A cell's sheet is told from the
/// second by the heights of the cells with one sheet within SHEET_REACH cells of it.
const SHEET_CELL_MM: f64 = 2.0;
const SHEET_GAP_MM: f64 = 1.0;
const SHEET_REACH: i64 = 2;
/// A robust fit's dropped points further off its surface than this, mm, and twice its
/// limit, are strays — what lies nearer is the skin's own scatter at an edge — clustered on
/// a grid this many mm wide.
const STRAY_FAR_MM: f64 = 1.0;
const STRAY_VOXEL_MM: f64 = 5.0;
/// A group of fewer points than this is scatter, not a stray.
const STRAY_MIN_POINTS: usize = 10;
/// How many strays are named, the largest first; and the share of the region a stray's box
/// may hold.
const STRAYS_NAMED: usize = 6;
const STRAY_MOST: f64 = 0.8;
/// How far a stray's box reaches past its points, mm: the region's points are every k-th of
/// the scan's (SURFACE_REGION_LIMIT), and the box given back as an exclude has to take the
/// scan's points between them too.
const STRAY_PAD_MM: f64 = 1.0;

/// The strays of a region against the surface fitted through it, the largest first
/// (`Stray`): two cheap reads of the points, made where a fit is refused or a robust fit is
/// answered.
///  - Second sheets (`second_sheets`): the points' heights off the fit, in cells about 2 mm
///    wide over its plane, a cell whose heights part by more than a millimetre holding two
///    surfaces — the inside of a cup under its rim, a gutter beside a flange — which no
///    height field is. The cells' far sheets are flood-joined into groups.
///  - With `robust`, the robust fit's last centre and limit: the points it left out far off
///    the surface, clustered on a 5 mm grid.
pub fn stray_groups(fit: &SurfaceFitResult, xyz: &[DVec3], robust: Option<RobustAt>) -> Vec<Stray> {
    // A second sheet's points the robust fit left out too are named once, as the sheet.
    let mut named = HashSet::new();
    let sheets = second_sheets(fit, xyz, &mut named);
    let mut out = sheets;
    if let Some(robust) = robust {
        out.extend(far_points(fit, xyz, robust, &named));
    }
    // A box that would leave out most of the region is no stray's — a rim the fit does not
    // follow, all the way round — and is not named.
    let n = xyz.len();
    let most = |s: &Stray| {
        let within = xyz
            .iter()
            .filter(|p| (0..3).all(|k| p[k] >= s.bounds.min[k] && p[k] <= s.bounds.max[k]))
            .count();
        within as f64 > STRAY_MOST * n as f64
    };
    // A stable sort, the largest first, as JavaScript's sort is.
    out.sort_by_key(|s| std::cmp::Reverse(s.points));
    out.into_iter().filter(|s| !most(s)).take(STRAYS_NAMED).collect()
}

/// A tenth of a millimetre, as a stray's words and an answer round.
fn r1(x: f64) -> f64 {
    js_round(x * 10.0) / 10.0
}

/// A range of millimetres as a stray's words give it.
fn range_mm(lo: f64, hi: f64) -> String {
    if r1(lo) == r1(hi) {
        js_num(r1(lo))
    } else {
        format!("{}–{}", js_num(r1(lo)), js_num(r1(hi)))
    }
}

/// A box round some of the points, STRAY_PAD_MM wider every way, its numbers rounded
/// outward to a tenth.
fn stray_box(xyz: &[DVec3], members: &[usize]) -> Bounds {
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for &i in members {
        for k in 0..3 {
            min[k] = min[k].min(xyz[i][k]);
            max[k] = max[k].max(xyz[i][k]);
        }
    }
    let lo = |x: f64| ((x - STRAY_PAD_MM) * 10.0).floor() / 10.0 + 0.0;
    let hi = |x: f64| ((x + STRAY_PAD_MM) * 10.0).ceil() / 10.0 + 0.0;
    Bounds { min: DVec3::new(lo(min[0]), lo(min[1]), lo(min[2])), max: DVec3::new(hi(max[0]), hi(max[1]), hi(max[2])) }
}

/// The groups of keys joined by `near`, each key once: a flood fill over the cells of a
/// grid.
fn flood_groups<K: Hash + Eq + Clone>(keys: &[K], near: impl Fn(&K) -> Vec<K>) -> Vec<Vec<K>> {
    let mut left: HashSet<K> = keys.iter().cloned().collect();
    let mut groups = Vec::new();
    for start in keys {
        if !left.remove(start) {
            continue;
        }
        let mut group = vec![start.clone()];
        let mut at = 0;
        while at < group.len() {
            for n in near(&group[at]) {
                if left.remove(&n) {
                    group.push(n);
                }
            }
            at += 1;
        }
        groups.push(group);
    }
    groups
}

/// The middle of some heights, as ScanRuler's sheets take it: the upper of two middles.
fn median_of(mut s: Vec<f64>) -> f64 {
    s.sort_by(|a, b| js_cmp(*a, *b));
    s.get(s.len() >> 1).copied().unwrap_or(f64::NAN)
}

/// The second sheets along a fit's height direction (`stray_groups`).
fn second_sheets(fit: &SurfaceFitResult, xyz: &[DVec3], named: &mut HashSet<usize>) -> Vec<Stray> {
    let n = xyz.len();
    if n < 4 {
        return Vec::new();
    }
    let mut u = vec![0.0; n];
    let mut v = vec![0.0; n];
    // Each point's height off the fit, not over its plane: a steep slope across a cell is
    // no gap between its heights, where it is followed.
    let mut h = vec![0.0; n];
    let mut bu = vec![0.0; fit.degree + 1];
    let mut bv = vec![0.0; fit.degree + 1];
    let (mut u_lo, mut u_hi, mut v_lo, mut v_hi) = (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
    for (i, q) in xyz.iter().enumerate() {
        let (x, y, z) = (q.x - fit.origin.x, q.y - fit.origin.y, q.z - fit.origin.z);
        u[i] = x * fit.axis_u.x + y * fit.axis_u.y + z * fit.axis_u.z;
        v[i] = x * fit.axis_v.x + y * fit.axis_v.y + z * fit.axis_v.z;
        h[i] = x * fit.normal.x + y * fit.normal.y + z * fit.normal.z
            - eval_height(&fit.knots_u, &fit.knots_v, fit.nu, fit.nv, fit.degree, &fit.heights, u[i], v[i], &mut bu, &mut bv);
        u_lo = u_lo.min(u[i]);
        u_hi = u_hi.max(u[i]);
        v_lo = v_lo.min(v[i]);
        v_hi = v_hi.max(v[i]);
    }
    let spacing = (((u_hi - u_lo) * (v_hi - v_lo)).max(1e-12) / n as f64).sqrt();
    let size = SHEET_CELL_MM.max(2.0 * spacing);
    // A grid of more cells than a double counts exactly (2^53) has no cells to tell apart —
    // ScanRuler's keys lose their last digits there — and none past the integers' range:
    // points that far apart, or not numbers at all, hold no second sheet to name.
    let (rows_f, cols_f) = (((u_hi - u_lo) / size).floor() + 1.0, ((v_hi - v_lo) / size).floor() + 1.0);
    if !(rows_f * cols_f <= 9.0e15) {
        return Vec::new();
    }
    let cols = cols_f as i64;
    let cell_of = |i: usize| ((u[i] - u_lo) / size).floor() as i64 * cols + ((v[i] - v_lo) / size).floor() as i64;
    let mut cells: OrderedMap<i64, Vec<usize>> = OrderedMap::new();
    for i in 0..n {
        cells.get_or_insert_with(cell_of(i), Vec::new).push(i);
    }
    let median = |list: &[usize]| median_of(list.iter().map(|&i| h[i]).collect());
    // Each cell's one sheet's height, or its two sheets split at the widest gap between its
    // heights.
    let mut single: HashMap<i64, f64> = HashMap::new();
    let mut parted: OrderedMap<i64, (Vec<usize>, Vec<usize>)> = OrderedMap::new();
    for (c, list) in cells.iter_mut() {
        list.sort_by(|&a, &b| js_cmp(h[a], h[b]));
        let (mut gap, mut at) = (0.0, 0);
        for j in 1..list.len() {
            if h[list[j]] - h[list[j - 1]] > gap {
                gap = h[list[j]] - h[list[j - 1]];
                at = j;
            }
        }
        if gap > SHEET_GAP_MM {
            parted.insert(*c, (list[..at].to_vec(), list[at..].to_vec()));
        } else {
            single.insert(*c, median(list));
        }
    }
    if parted.is_empty() {
        return Vec::new();
    }
    // The far sheet of each parted cell: the one further from the heights of the cells round
    // it with one sheet — or, with none round it, from the fit, which a robust fit lays on
    // the sheet it keeps. Each far point's height off the sheet kept.
    let mut far: OrderedMap<i64, (Vec<usize>, Vec<f64>)> = OrderedMap::new();
    for (c, (under, over)) in parted.iter() {
        let (cu, cv) = (c.div_euclid(cols), c.rem_euclid(cols));
        let mut round = Vec::new();
        for a in -SHEET_REACH..=SHEET_REACH {
            for b in -SHEET_REACH..=SHEET_REACH {
                if cv + b >= 0 && cv + b < cols {
                    if let Some(&hh) = single.get(&((cu + a) * cols + cv + b)) {
                        round.push(hh);
                    }
                }
            }
        }
        let (mu, mo) = (median(under), median(over));
        let reference = if round.is_empty() { 0.0 } else { median_of(round) };
        let under_far = (mu - reference).abs() > (mo - reference).abs();
        let (members, kept) = if under_far { (under, mo) } else { (over, mu) };
        far.insert(*c, (members.clone(), members.iter().map(|&i| h[i] - kept).collect()));
    }
    // Cells side by side are one stray, whatever their heights: the inside of a cup seen
    // along a tilted height direction runs on in steps a sheet's height apart, and is one
    // box to leave out.
    let groups = flood_groups(&far.keys(), |&c| {
        let (cu, cv) = (c.div_euclid(cols), c.rem_euclid(cols));
        let mut out = Vec::new();
        for a in -1..=1i64 {
            for b in -1..=1i64 {
                let k = (cu + a) * cols + cv + b;
                if (a != 0 || b != 0) && cv + b >= 0 && cv + b < cols && far.contains_key(&k) {
                    out.push(k);
                }
            }
        }
        out
    });
    let mut strays = Vec::new();
    for group in groups {
        let members: Vec<usize> = group.iter().filter_map(|c| far.get(c)).flat_map(|f| f.0.iter().copied()).collect();
        if members.len() < STRAY_MIN_POINTS {
            continue;
        }
        named.extend(members.iter().copied());
        let offs: Vec<f64> = group.iter().filter_map(|c| far.get(c)).flat_map(|f| f.1.iter().copied()).collect();
        // units-ok: build message, also the agent's answer
        strays.push(stray_of(xyz, &members, &offs, |mm, side| {
            format!("a second sheet {mm} mm {side} the one fitted, along its height direction — no height field holds both")
        }));
    }
    strays
}

/// A stray of the points `members`, each `offs` off the surface (signed along its normal),
/// worded by `why` from the range and the side most of them lie on.
fn stray_of(xyz: &[DVec3], members: &[usize], offs: &[f64], why: impl Fn(&str, &str) -> String) -> Stray {
    let (mut lo, mut hi, mut under) = (f64::INFINITY, 0.0f64, 0usize);
    for &x in offs {
        lo = lo.min(x.abs());
        hi = hi.max(x.abs());
        if x < 0.0 {
            under += 1;
        }
    }
    let side = if under as f64 > offs.len() as f64 / 2.0 { "under" } else { "over" };
    Stray { points: members.len(), bounds: stray_box(xyz, members), off_mm: [r1(lo), r1(hi)], why: why(&range_mm(lo, hi), side) }
}

/// The points a robust fit left out far off its surface, in groups (`stray_groups`).
fn far_points(fit: &SurfaceFitResult, xyz: &[DVec3], robust: RobustAt, named: &HashSet<usize>) -> Vec<Stray> {
    let r = surface_residuals(fit, xyz);
    let beyond = STRAY_FAR_MM.max(2.0 * robust.limit);
    let mut voxels: OrderedMap<(i64, i64, i64), Vec<usize>> = OrderedMap::new();
    for i in 0..r.len() {
        if !((r[i] - robust.med).abs() > beyond) || named.contains(&i) {
            continue;
        }
        let q = xyz[i];
        let key = ((q.x / STRAY_VOXEL_MM).floor() as i64, (q.y / STRAY_VOXEL_MM).floor() as i64, (q.z / STRAY_VOXEL_MM).floor() as i64);
        voxels.get_or_insert_with(key, Vec::new).push(i);
    }
    let groups = flood_groups(&voxels.keys(), |&(x, y, z)| {
        let mut out = Vec::with_capacity(26);
        for a in -1..=1 {
            for b in -1..=1 {
                for c in -1..=1 {
                    if a != 0 || b != 0 || c != 0 {
                        // Saturating: a point far past any part (or no number) keys a voxel
                        // at the integers' end, and its neighbours stay there.
                        out.push((x.saturating_add(a), y.saturating_add(b), z.saturating_add(c)));
                    }
                }
            }
        }
        out
    });
    let mut strays = Vec::new();
    for group in groups {
        let members: Vec<usize> = group.iter().filter_map(|k| voxels.get(k)).flat_map(|l| l.iter().copied()).collect();
        if members.len() < STRAY_MIN_POINTS {
            continue;
        }
        let offs: Vec<f64> = members.iter().map(|&i| r[i]).collect();
        // units-ok: build message, also the agent's answer
        strays.push(stray_of(xyz, &members, &offs, |mm, side| format!("left out by the robust fit, {mm} mm {side} its surface")));
    }
    strays
}

/// The strays as a refusal ends on them: the largest named, the rest counted — or nothing,
/// for none.
fn strays_words(strays: &[Stray]) -> String {
    if strays.is_empty() {
        return String::new();
    }
    let v3 = |p: DVec3| format!("{}, {}, {}", js_num(p.x), js_num(p.y), js_num(p.z));
    let at = |b: &Bounds| format!("({})–({})", v3(b.min), v3(b.max));
    let named: Vec<String> = strays
        .iter()
        .take(2)
        .map(|s| {
            let why = match s.why.find(" — ") {
                Some(i) => &s.why[..i],
                None => s.why.as_str(),
            };
            format!("{} points {} in {}", s.points, why, at(&s.bounds))
        })
        .collect();
    let more = if strays.len() > 2 { format!(", and {} more", strays.len() - 2) } else { String::new() };
    format!(" Strays (strays): {}{} — give their boxes in region.exclude and fit again.", named.join("; "), more)
}

/// How far past its points' heights a fitted surface may stand, beyond its margin, as a
/// share of the region's extent (`run_away`).
const RUNAWAY_SHARE: f64 = 0.25;
/// How far a region's normals may turn from its height direction, degrees (the 95th
/// percentile), and still be one height field over it: past it, the region bends round — a
/// shell over a shoulder.
pub const HEIGHT_FIELD_DEG: f64 = 70.0;
/// How many of a region's points its normals' spread is read at, at most.
const SPREAD_POINTS: usize = 5000;

/// How far a region's normals turn from a height direction: each normal a plane through the
/// points within a few spacings of a point (their least scatter), its angle from the
/// direction either way round, degrees — the 95th percentile. Read off the points, not off a
/// fit that may have run away from them. Not the most: a plane through a rim's or a thin
/// strip's points turns any way, 89° on a region that fits at 0.06 mm.
pub fn normal_spread(xyz: &[DVec3], normal: DVec3) -> f64 {
    let n = xyz.len();
    if n < 4 {
        return 0.0;
    }
    // The points' spacing, from the area of the two longer sides of their box.
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for q in xyz {
        for k in 0..3 {
            lo[k] = lo[k].min(q[k]);
            hi[k] = hi[k].max(q[k]);
        }
    }
    let mut sides = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
    sides.sort_by(|a, b| js_cmp(*b, *a));
    let reach = 3.0 * ((sides[0] * sides[1]).max(1e-12) / n as f64).sqrt();
    let key = |q: DVec3| ((q.x / reach).floor() as i64, (q.y / reach).floor() as i64, (q.z / reach).floor() as i64);
    let mut cells: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
    for (i, q) in xyz.iter().enumerate() {
        cells.entry(key(*q)).or_default().push(i);
    }
    let len = match hypot3(normal) {
        l if l == 0.0 || l.is_nan() => 1.0,
        l => l,
    };
    let step = 1usize.max(n.div_ceil(SPREAD_POINTS));
    let mut angles = Vec::new();
    let mut near = Vec::new();
    for i in (0..n).step_by(step) {
        let p = xyz[i];
        let (cx, cy, cz) = key(p);
        near.clear();
        for a in -1..=1 {
            for b in -1..=1 {
                for d in -1..=1 {
                    if let Some(list) = cells.get(&(cx.saturating_add(a), cy.saturating_add(b), cz.saturating_add(d))) {
                        for &j in list {
                            if hypot3(xyz[j] - p) <= reach {
                                near.push(j);
                            }
                        }
                    }
                }
            }
        }
        if near.len() < 6 {
            continue;
        }
        let (mut mx, mut my, mut mz) = (0.0, 0.0, 0.0);
        for &j in &near {
            mx += xyz[j].x;
            my += xyz[j].y;
            mz += xyz[j].z;
        }
        let k = near.len() as f64;
        mx /= k;
        my /= k;
        mz /= k;
        let mut c = [0.0f64; 9];
        for &j in &near {
            let (x, y, z) = (xyz[j].x - mx, xyz[j].y - my, xyz[j].z - mz);
            c[0] += x * x;
            c[1] += x * y;
            c[2] += x * z;
            c[4] += y * y;
            c[5] += y * z;
            c[8] += z * z;
        }
        c[3] = c[1];
        c[6] = c[2];
        c[7] = c[5];
        let m = symmetric_eigen3(&c).vectors[0];
        let cos = (m.x * normal.x + m.y * normal.y + m.z * normal.z).abs() / len;
        angles.push((cos.min(1.0).acos() * 180.0) / std::f64::consts::PI);
    }
    if angles.is_empty() {
        return 0.0;
    }
    angles.sort_by(|a, b| js_cmp(*a, *b));
    let at = (angles.len() - 1).min((0.95 * angles.len() as f64).floor() as usize);
    angles[at]
}

/// Why a fitted surface is no surface of its points, or None: where it leaves the points'
/// height range over its plane by more than its margin plus a quarter of the region's extent
/// — the margin, because a surface sloping at 45° stands that far past its points at the
/// margin's edge; the quarter, for the bend a curved patch carries on with. A fit that
/// stands further off has run away where the points run out, and the answer names the spot,
/// and the last point before it. The pull toward the plane (`fit_surface`) keeps a fit from
/// it; this is the backstop, and what keeps a surface a hundred times its region's size from
/// going on to the kernel and its mesher. `asked`: the height direction the fit was asked to
/// take (params.normal). `facing`: the way the region was read facing, which bounds its
/// normals — said in place of their spread.
pub fn run_away(fit: &SurfaceFitResult, xyz: &[DVec3], expand_pct: f64, asked: Option<DVec3>, facing: Option<&RegionFacing>) -> Option<String> {
    let n = xyz.len();
    if n == 0 {
        return None;
    }
    let mut pu = vec![0.0; n];
    let mut pv = vec![0.0; n];
    let (mut h_lo, mut h_hi, mut u_lo, mut u_hi, mut v_lo, mut v_hi) =
        (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
    for (i, q) in xyz.iter().enumerate() {
        let (x, y, z) = (q.x - fit.origin.x, q.y - fit.origin.y, q.z - fit.origin.z);
        pu[i] = x * fit.axis_u.x + y * fit.axis_u.y + z * fit.axis_u.z;
        pv[i] = x * fit.axis_v.x + y * fit.axis_v.y + z * fit.axis_v.z;
        let hh = x * fit.normal.x + y * fit.normal.y + z * fit.normal.z;
        h_lo = h_lo.min(hh);
        h_hi = h_hi.max(hh);
        u_lo = u_lo.min(pu[i]);
        u_hi = u_hi.max(pu[i]);
        v_lo = v_lo.min(pv[i]);
        v_hi = v_hi.max(pv[i]);
    }
    let extent = (u_hi - u_lo).max(v_hi - v_lo);
    let allowed = (extent * expand_pct.max(0.0)) / 200.0 + RUNAWAY_SHARE * extent;
    // The surface sampled as the kernel gets it.
    let rows = 4.max((fit.nu.max(1) - 1) * GRID_PER_SPAN + 1);
    let cols = 4.max((fit.nv.max(1) - 1) * GRID_PER_SPAN + 1);
    let (mut worst, mut wu, mut wv, mut wh) = (0.0, 0.0, 0.0, 0.0);
    let mut bu = vec![0.0; fit.degree + 1];
    let mut bv = vec![0.0; fit.degree + 1];
    for r in 0..rows {
        let uu = fit.u_range[0] + ((fit.u_range[1] - fit.u_range[0]) * r as f64) / (rows - 1) as f64;
        for c in 0..cols {
            let vv = fit.v_range[0] + ((fit.v_range[1] - fit.v_range[0]) * c as f64) / (cols - 1) as f64;
            let hh = eval_height(&fit.knots_u, &fit.knots_v, fit.nu, fit.nv, fit.degree, &fit.heights, uu, vv, &mut bu, &mut bv);
            let off = (hh - h_hi).max(h_lo - hh);
            if off > worst {
                (worst, wu, wv, wh) = (off, uu, vv, hh);
            }
        }
    }
    if !(worst > allowed) {
        return None;
    }
    // The last point before the spot: the nearest over the plane.
    let (mut last, mut best) = (0, f64::INFINITY);
    for i in 0..n {
        let d = (pu[i] - wu).powi(2) + (pv[i] - wv).powi(2);
        if d < best {
            (best, last) = (d, i);
        }
    }
    let at = |uu: f64, vv: f64, hh: f64| {
        (0..3)
            .map(|k| to_fixed1(fit.origin[k] + uu * fit.axis_u[k] + vv * fit.axis_v[k] + hh * fit.normal[k]))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let last_at = (0..3).map(|k| to_fixed1(xyz[last][k])).collect::<Vec<_>>().join(", ");
    // The height direction it was fitted along, and how far the region turns from it: past
    // HEIGHT_FIELD_DEG no height field holds it. A region read facing a way has its normals
    // within the query's angle of that by construction — what is said then, not a spread
    // that reads the query's own angle back.
    let spread = normal_spread(xyz, fit.normal);
    let dir_of = |d: DVec3| format!("[{}]", (0..3).map(|k| js_num(js_round(d[k] * 1000.0) / 1000.0 + 0.0)).collect::<Vec<_>>().join(", "));
    let normal = dir_of(fit.normal);
    struct Faced {
        dir: DVec3,
        within: f64,
        apart: f64,
    }
    let faced = facing.map(|facing| {
        let len = match hypot3(facing.dir) {
            l if l == 0.0 || l.is_nan() => 1.0,
            l => l,
        };
        let dir = facing.dir / len;
        let within = facing.within_deg.unwrap_or(FACING_DEG);
        let apart = ((dir.x * fit.normal.x + dir.y * fit.normal.y + dir.z * fit.normal.z).abs().min(1.0).acos() * 180.0) / std::f64::consts::PI;
        Faced { dir, within, apart }
    });
    let asked_words = if asked.is_some() { "fit.normal" } else { "the region’s principal normal" };
    let along = match &faced {
        Some(f) => format!(
            "Its height direction was {normal} ({asked_words}); the region was read facing {} within {}°, so its normals are within {}° of that by construction{}.",
            dir_of(f.dir),
            js_num(r1(f.within)),
            js_num(r1(f.within)),
            if f.apart >= 1.0 { format!(", {}° off the height direction", js_num(js_round(f.apart))) } else { String::new() }
        ),
        None => format!(
            "Its height direction was {normal} ({asked_words}); 95 % of the region’s normals are within {}° of it.",
            js_num(js_round(spread))
        ),
    };
    let bends = spread > HEIGHT_FIELD_DEG && !faced.as_ref().is_some_and(|f| f.within + f.apart <= HEIGHT_FIELD_DEG);
    let direction = if asked.is_some() {
        String::new()
    } else {
        match &faced {
            Some(f) if f.apart >= 1.0 => format!(" — or give fit.normal: {}, the way the region was read facing, for its height direction", dir_of(f.dir)),
            _ => " — or give fit.normal, the way the region faces, for its height direction".to_string(),
        }
    };
    let advice = if bends {
        "The region bends round past what one height field holds — split it, or loft through stations.".to_string()
    } else {
        format!("Fit a coarser net, keep the region to one surface (a box closer round it, or exclude what stands off it), or take less margin{direction}.")
    };
    Some(format!(
        // units-ok: build message, also the agent's answer
        "The surface runs away from its points: it stands {} mm {} of them at ({}), {} mm past where the region's points run out at ({}) — more than the margin and a quarter of the region's {} mm allow. {along} {advice}",
        js_num(r1(worst)),
        if wh > h_hi { "above the highest" } else { "below the lowest" },
        at(wu, wv, wh),
        js_num(r1(best.sqrt())),
        last_at,
        js_num(r1(extent)),
    ))
}

/// The angle `mesh_triangles` reads the mesher at where it is given none (ScanRuler's
/// default, radians).
pub const MESH_ANGLE: f64 = 0.15;

/// About how many triangles the kernel's mesher makes of a fitted surface at a chord (and
/// its angle, as the kernel's mesh takes them; MESH_ANGLE where none is given), read off the
/// fit before the surface is made. Over a sampling a few times finer than the net, each
/// patch takes the edge the mesher may lay where it bends most — the chord's
/// sqrt(8 · chord / κ), or the angle's angle / κ where that is shorter — the same both ways,
/// as the mesher keeps its triangles from growing long and thin along the way a surface
/// does not bend, and MESH_SURPLUS times the triangles of that edge that fill the patch: the
/// mesher lays more than the fewest that would do. Within a factor of two or so of what it
/// makes (ScanRuler's tests/reSurfaceRunaway): enough to refuse a surface that bends so
/// often it would mesh for minutes.
pub fn mesh_triangles(fit: &SurfaceFitResult, chord: f64, angle: f64) -> f64 {
    let su = 4.max((fit.nu.max(1) - 1) * 4);
    let sv = 4.max((fit.nv.max(1) - 1) * 4);
    let du = (fit.u_range[1] - fit.u_range[0]) / su as f64;
    let dv = (fit.v_range[1] - fit.v_range[0]) / sv as f64;
    let whole = (fit.u_range[1] - fit.u_range[0]).min(fit.v_range[1] - fit.v_range[0]);
    let mut bu = vec![0.0; fit.degree + 1];
    let mut bv = vec![0.0; fit.degree + 1];
    let mut h = |uu: f64, vv: f64| eval_height(&fit.knots_u, &fit.knots_v, fit.nu, fit.nv, fit.degree, &fit.heights, uu, vv, &mut bu, &mut bv);
    let mut total = 0.0;
    for a in 0..su {
        let uu = fit.u_range[0] + du * (a as f64 + 0.5);
        for b in 0..sv {
            let vv = fit.v_range[0] + dv * (b as f64 + 0.5);
            let h0 = h(uu, vv);
            let (hup, hum) = (h(uu + du / 2.0, vv), h(uu - du / 2.0, vv));
            let (hvp, hvm) = (h(uu, vv + dv / 2.0), h(uu, vv - dv / 2.0));
            let hu = (hup - hum) / du;
            let hv = (hvp - hvm) / dv;
            let huu = (hup - 2.0 * h0 + hum) / ((du * du) / 4.0);
            let hvv = (hvp - 2.0 * h0 + hvm) / ((dv * dv) / 4.0);
            let w = (1.0 + hu * hu + hv * hv).sqrt();
            // The normal curvature along each parameter line; the larger rules.
            let k = (huu.abs() / (w * (1.0 + hu * hu))).max(hvv.abs() / (w * (1.0 + hv * hv)));
            let edge = whole.min(if k > 1e-12 { ((8.0 * chord) / k).sqrt().min(angle / k) } else { f64::INFINITY });
            total += (2.0 * du * dv * w) / (edge * edge);
        }
    }
    js_round(MESH_SURPLUS * total)
}

/// The pull's system (`fit_surface`): for each net cell further than PULL_REACH cells from
/// every cell with a point, PULL_SAMPLES² samples at height 0, each weighing a
/// PULL_SAMPLES²-th of one point — or None when every cell has points within reach. The
/// count is how many cells have points, for the weight of an average one.
fn pull_system(u: &[f64], v: &[f64], knots_u: &[f64], knots_v: &[f64], nu: usize, nv: usize, p: usize) -> Option<(Vec<f64>, usize)> {
    let cu = nu - p;
    let cv = nv - p;
    let mut held = vec![false; cu * cv];
    for i in 0..u.len() {
        held[(find_span(knots_u, nu, p, u[i]) - p) * cv + (find_span(knots_v, nv, p, v[i]) - p)] = true;
    }
    let cells = held.iter().filter(|&&x| x).count();
    let near = |a: usize, b: usize| {
        for i in a.saturating_sub(PULL_REACH)..=(cu - 1).min(a + PULL_REACH) {
            for j in b.saturating_sub(PULL_REACH)..=(cv - 1).min(b + PULL_REACH) {
                if held[i * cv + j] {
                    return true;
                }
            }
        }
        false
    };
    let m = nu * nv;
    let mut system = vec![0.0; m * m];
    let mut bu = vec![0.0; p + 1];
    let mut bv = vec![0.0; p + 1];
    let w = 1.0 / (PULL_SAMPLES * PULL_SAMPLES) as f64;
    let samples = PULL_SAMPLES as f64;
    let mut pulled = false;
    for a in 0..cu {
        for b in 0..cv {
            if near(a, b) {
                continue;
            }
            pulled = true;
            for s in 0..PULL_SAMPLES {
                let tu = knots_u[p + a] + ((knots_u[p + a + 1] - knots_u[p + a]) * (s as f64 + 0.5)) / samples;
                basis_functions(knots_u, p + a, p, tu, &mut bu);
                for t in 0..PULL_SAMPLES {
                    let tv = knots_v[p + b] + ((knots_v[p + b + 1] - knots_v[p + b]) * (t as f64 + 0.5)) / samples;
                    basis_functions(knots_v, p + b, p, tv, &mut bv);
                    for a1 in 0..=p {
                        for b1 in 0..=p {
                            let row = ((a + a1) * nv + b + b1) * m;
                            let va = w * bu[a1] * bv[b1];
                            for a2 in 0..=p {
                                for b2 in 0..=p {
                                    system[row + (a + a2) * nv + b + b2] += va * bu[a2] * bv[b2];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    if pulled {
        Some((system, cells))
    } else {
        None
    }
}

/// The weights of the `order`-th divided difference over the nodes x[from … from + order],
/// scaled by order! so it reads as the derivative — zero on every polynomial below that
/// order, at any spacing.
fn divided_weights(x: &[f64], from: usize, order: usize) -> Vec<f64> {
    let mut fact = 1.0;
    for k in 2..=order {
        fact *= k as f64;
    }
    (0..=order)
        .map(|k| {
            let mut denom = 1.0;
            for j in 0..=order {
                if j != k {
                    denom *= x[from + k] - x[from + j];
                }
            }
            fact / denom
        })
        .collect()
}

fn clamp_net(count: usize, p: usize) -> usize {
    NET_MIN.max(p + 1).max(NET_MAX.min(count))
}

/// The fitted height at (u, v), `bu` and `bv` the basis's scratch (p + 1 each).
#[allow(clippy::too_many_arguments)]
fn eval_height(knots_u: &[f64], knots_v: &[f64], nu: usize, nv: usize, p: usize, heights: &[f64], uu: f64, vv: f64, bu: &mut [f64], bv: &mut [f64]) -> f64 {
    // A fit whose knots, net and heights do not agree — one read back from a file — has no
    // height anywhere: ScanRuler reads `undefined` past its arrays and answers NaN, where an
    // index past a slice would panic. A fit `fit_surface` made always agrees.
    if nu <= p || nv <= p || knots_u.len() < nu + p + 1 || knots_v.len() < nv + p + 1 || heights.len() < nu * nv || bu.len() <= p || bv.len() <= p {
        return f64::NAN;
    }
    let su = find_span(knots_u, nu, p, uu);
    let sv = find_span(knots_v, nv, p, vv);
    basis_functions(knots_u, su, p, uu, bu);
    basis_functions(knots_v, sv, p, vv, bv);
    let mut s = 0.0;
    for a in 0..=p {
        let row = (su - p + a) * nv;
        for b in 0..=p {
            s += bu[a] * bv[b] * heights[row + (sv - p + b)];
        }
    }
    s
}

/// The fitted height at (u, v) of a fit.
fn height_of(fit: &SurfaceFitResult, uu: f64, vv: f64) -> f64 {
    let mut bu = vec![0.0; fit.degree + 1];
    let mut bv = vec![0.0; fit.degree + 1];
    eval_height(&fit.knots_u, &fit.knots_v, fit.nu, fit.nv, fit.degree, &fit.heights, uu, vv, &mut bu, &mut bv)
}

/// The fitted point at (u, v), in the part's frame.
pub fn eval_surface(fit: &SurfaceFitResult, uu: f64, vv: f64) -> DVec3 {
    let hh = height_of(fit, uu, vv);
    DVec3::new(
        fit.origin.x + uu * fit.axis_u.x + vv * fit.axis_v.x + hh * fit.normal.x,
        fit.origin.y + uu * fit.axis_u.y + vv * fit.axis_v.y + hh * fit.normal.y,
        fit.origin.z + uu * fit.axis_u.z + vv * fit.axis_v.z + hh * fit.normal.z,
    )
}

/// A regular grid of points for the kernel: `rows` along u, `cols` along v, row-major.
#[derive(Debug, Clone, PartialEq)]
pub struct Grid {
    pub points: Vec<DVec3>,
    pub rows: usize,
    pub cols: usize,
}

/// The fitted surface sampled on a regular grid for the kernel: `rows` along u, `cols`
/// along v, the points row-major — `per_span` samples a net span, GRID_PER_SPAN unless the
/// kernel's surface through those did not follow the fit (GRID_RETRIES).
pub fn grid_of(fit: &SurfaceFitResult, per_span: usize) -> Grid {
    let rows = 4.max((fit.nu.max(1) - 1) * per_span + 1);
    let cols = 4.max((fit.nv.max(1) - 1) * per_span + 1);
    let mut points = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        let uu = fit.u_range[0] + ((fit.u_range[1] - fit.u_range[0]) * r as f64) / (rows - 1) as f64;
        for cc in 0..cols {
            let vv = fit.v_range[0] + ((fit.v_range[1] - fit.v_range[0]) * cc as f64) / (cols - 1) as f64;
            points.push(eval_surface(fit, uu, vv));
        }
    }
    Grid { points, rows, cols }
}

/// The area of a grid's facets, two triangles a cell: what the kernel's surface through the
/// grid should have, near enough.
pub fn grid_area(grid: &Grid) -> f64 {
    let g = &grid.points;
    let mut area = 0.0;
    let mut r = 0;
    while r + 1 < grid.rows {
        let mut c = 0;
        while c + 1 < grid.cols {
            let a = r * grid.cols + c;
            let b = a + 1;
            let d = a + grid.cols;
            let e = d + 1;
            for (p, q, o) in [(a, b, e), (a, e, d)] {
                let x = g[q] - g[p];
                let y = g[o] - g[p];
                area += 0.5 * hypot3(DVec3::new(x.y * y.z - x.z * y.y, x.z * y.x - x.x * y.z, x.x * y.y - x.y * y.x));
            }
            c += 1;
        }
        r += 1;
    }
    area
}

/// The (u, v) a surface needs to span.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Reach {
    pub u: [f64; 2],
    pub v: [f64; 2],
}

/// The (u, v) a surface needs to span to reach over a body — its points (its mesh's) in the
/// surface's frame, `margin` mm wider either way, and its own range — or None where it spans
/// that already. Over the body's points, not its box's corners: a body standing askew to the
/// surface has corners of its box far past itself — on the lamp's outline prism u −120…97
/// where the prism needs −90…68 — and a freeform carried on that far came back from the
/// kernel kilometres wide.
pub fn reach_over(plane: &SurfacePlane, points: &[DVec3], margin: f64) -> Option<Reach> {
    let (mut u_lo, mut u_hi, mut v_lo, mut v_hi) = (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
    for q in points {
        let d = *q - plane.origin;
        let u = d.x * plane.axis_u.x + d.y * plane.axis_u.y + d.z * plane.axis_u.z;
        let v = d.x * plane.axis_v.x + d.y * plane.axis_v.y + d.z * plane.axis_v.z;
        u_lo = u_lo.min(u);
        u_hi = u_hi.max(u);
        v_lo = v_lo.min(v);
        v_hi = v_hi.max(v);
    }
    let u = [plane.u_range[0].min(u_lo - margin), plane.u_range[1].max(u_hi + margin)];
    let v = [plane.v_range[0].min(v_lo - margin), plane.v_range[1].max(v_hi + margin)];
    if u == plane.u_range && v == plane.v_range {
        return None;
    }
    Some(Reach { u, v })
}

/// The height of a fitted surface at (u, v) as a split carries it on (`reaching_grid_of`):
/// the fit's own within its range; past its edge, flat along the slope it has there, read
/// just inside it.
pub fn carried_height(fit: &SurfaceFitResult, uu: f64, vv: f64) -> f64 {
    let eu = (fit.u_range[1] - fit.u_range[0]) * 1e-4;
    let ev = (fit.v_range[1] - fit.v_range[0]) * 1e-4;
    let h = |a: f64, b: f64| height_of(fit, a, b);
    let cu = fit.u_range[1].min(fit.u_range[0].max(uu));
    let cv = fit.v_range[1].min(fit.v_range[0].max(vv));
    let at = h(cu, cv);
    if cu == uu && cv == vv {
        return at;
    }
    let du = if cu == uu {
        0.0
    } else {
        (if cu == fit.u_range[1] { at - h(cu - eu, cv) } else { h(cu + eu, cv) - at }) / eu
    };
    let dv = if cv == vv {
        0.0
    } else {
        (if cv == fit.v_range[1] { at - h(cu, cv - ev) } else { h(cu, cv + ev) - at }) / ev
    };
    at + du * (uu - cu) + dv * (vv - cv)
}

/// How many samples a side the grid of a surface carried on takes at most.
const REACH_SAMPLES: f64 = 96.0;

/// The fitted surface sampled for the kernel, carried on past its own range to span `u` and
/// `v`. Past its edge it does not go on as its polynomial — that curls away from the data
/// the further out it is taken — but flat, along the slope it has at the edge: the plane
/// tangent to it there, carried out. What a split cuts a body with when the surface as
/// fitted stops short of the body's sides. The samples are evenly spaced, at `grid_of`'s
/// spacing where REACH_SAMPLES allows: the kernel spaces its parameters by the samples'
/// count, not their distance, and a grid that spreads out past the edge came back from it
/// bulging between them. `per_span` as `grid_of`'s: denser where the kernel's surface
/// through the first grid did not follow it.
pub fn reaching_grid_of(fit: &SurfaceFitResult, u: [f64; 2], v: [f64; 2], per_span: usize) -> Grid {
    let inner = grid_of(fit, per_span);
    let along = |range: [f64; 2], reach: [f64; 2], n: usize| -> Vec<f64> {
        let step = (range[1] - range[0]) / (n - 1) as f64;
        let count = REACH_SAMPLES.min((n as f64).max(((reach[1] - reach[0]) / step).ceil() + 1.0)) as usize;
        (0..count).map(|i| reach[0] + ((reach[1] - reach[0]) * i as f64) / (count as f64 - 1.0)).collect()
    };
    let us = along(fit.u_range, u, inner.rows);
    let vs = along(fit.v_range, v, inner.cols);
    let mut points = Vec::with_capacity(us.len() * vs.len());
    for &uu in &us {
        for &vv in &vs {
            let hh = carried_height(fit, uu, vv);
            points.push(DVec3::new(
                fit.origin.x + uu * fit.axis_u.x + vv * fit.axis_v.x + hh * fit.normal.x,
                fit.origin.y + uu * fit.axis_u.y + vv * fit.axis_v.y + hh * fit.normal.y,
                fit.origin.z + uu * fit.axis_u.z + vv * fit.axis_v.z + hh * fit.normal.z,
            ));
        }
    }
    Grid { points, rows: us.len(), cols: vs.len() }
}

// Ported from ScanRuler's plugins/reverse/tests: reSurfaceFit.test.ts, reSurfaceNormal.test.ts,
// reSurfaceRunaway.test.ts and reSurfaceStrays.test.ts (their tests that need no kernel), and
// reSplitSides.test.ts's reachOver test; with two adapted from tests that read their region
// through the app (reSurfaceRegion.test.ts) and a patch network (reNetworkFit.test.ts).
#[cfg(test)]
mod tests {
    use super::*;
    use crate::principal_frame::js_hypot2;
    use std::f64::consts::PI;

    /// The tests' deterministic noise source, so a failing case fails the same way: a linear
    /// congruential generator, as `(s * 1664525 + 1013904223) >>> 0` steps it.
    struct Lcg(u32);
    impl Lcg {
        fn step(&mut self) -> f64 {
            self.0 = (self.0 as u64 * 1664525 + 1013904223) as u32;
            self.0 as f64 / 4294967296.0
        }
        /// Two uniforms to a roughly Gaussian value with σ ≈ 1 (reSurfaceFit, reSurfaceRunaway).
        fn gauss(&mut self) -> f64 {
            let a = self.step();
            let b = self.step();
            (-2.0 * (a + 1e-12).ln()).sqrt() * (2.0 * PI * b).cos()
        }
        /// A uniform value in [−0.5, 0.5) (reSurfaceStrays).
        fn uniform(&mut self) -> f64 {
            self.step() - 0.5
        }
    }

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

    fn net(n: usize) -> SurfaceFitParams {
        SurfaceFitParams { net: Some(Net { u: n, v: n }), ..DEFAULT_SURFACE_FIT }
    }

    /// A small matcher for the shapes the TypeScript tests' regular expressions check.
    struct Scan<'a> {
        s: &'a str,
    }
    impl<'a> Scan<'a> {
        fn lit(&mut self, lit: &str) -> bool {
            match self.s.strip_prefix(lit) {
                Some(rest) => {
                    self.s = rest;
                    true
                }
                None => false,
            }
        }
        /// `-?[\d.]+` (with `extra` characters allowed too).
        fn num_with(&mut self, extra: &str) -> Option<&'a str> {
            let start = self.s;
            let mut rest = self.s.strip_prefix('-').unwrap_or(self.s);
            let body = rest.len() - rest.trim_start_matches(|c: char| c.is_ascii_digit() || c == '.' || extra.contains(c)).len();
            if body == 0 {
                return None;
            }
            rest = &rest[body..];
            self.s = rest;
            Some(&start[..start.len() - rest.len()])
        }
        fn num(&mut self) -> Option<&'a str> {
            self.num_with("")
        }
        fn either(&mut self, options: &[&str]) -> bool {
            options.iter().any(|o| self.lit(o))
        }
    }

    // ---- reSurfaceFit.test.ts ----------------------------------------------------------

    /// A rotation about a skew axis, so no principal axis lies on x, y or z.
    fn rotation() -> [[f64; 3]; 3] {
        let a: f64 = 0.6;
        [[a.cos(), -a.sin() * 0.8, a.sin() * 0.6], [a.sin(), a.cos() * 0.8, -a.cos() * 0.6], [0.0, 0.6, 0.8]]
    }

    /// Points z = f(x, y) over a grid, jittered in the plane and noisy in z, in a frame
    /// turned away from the axes so the fit has to find its own — held at single precision,
    /// as the TypeScript test's Float32Array holds them.
    fn sample(f: impl Fn(f64, f64) -> f64, w: f64, h: f64, step: f64, sigma: f64, seed: u32) -> Vec<DVec3> {
        let mut rnd = Lcg(seed);
        let r = rotation();
        let turn = |x: f64, y: f64, z: f64| {
            DVec3::new(
                r[0][0] * x + r[0][1] * y + r[0][2] * z + 5.0,
                r[1][0] * x + r[1][1] * y + r[1][2] * z - 3.0,
                r[2][0] * x + r[2][1] * y + r[2][2] * z + 11.0,
            )
        };
        let mut pts = Vec::new();
        for x in steps(-w / 2.0, w / 2.0, step) {
            for y in steps(-h / 2.0, h / 2.0, step) {
                let jx = x + rnd.gauss() * step * 0.2;
                let jy = y + rnd.gauss() * step * 0.2;
                let p = turn(jx, jy, f(jx, jy) + rnd.gauss() * sigma);
                pts.push(DVec3::new(p.x as f32 as f64, p.y as f32 as f64, p.z as f32 as f64));
            }
        }
        pts
    }

    /// How far a point lies off the true surface, back in its own frame.
    fn truth(f: &impl Fn(f64, f64) -> f64, p: DVec3) -> f64 {
        let r = rotation();
        let (x, y, z) = (p.x - 5.0, p.y + 3.0, p.z - 11.0);
        let q = [r[0][0] * x + r[1][0] * y + r[2][0] * z, r[0][1] * x + r[1][1] * y + r[2][1] * z, r[0][2] * x + r[1][2] * y + r[2][2] * z];
        f(q[0], q[1]) - q[2]
    }

    /// The largest distance of the fitted surface from the true one, sampled over the
    /// region's own extent (the margin is extrapolation).
    fn worst_error(fit: &SurfaceFitResult, f: impl Fn(f64, f64) -> f64, shrink: f64) -> f64 {
        let [u0, u1] = fit.u_range;
        let [v0, v1] = fit.v_range;
        let (mu, mv) = ((u1 - u0) * shrink, (v1 - v0) * shrink);
        let mut worst = 0.0f64;
        for i in 0..=20 {
            for j in 0..=20 {
                let p = eval_surface(fit, u0 + mu + ((u1 - u0 - 2.0 * mu) * i as f64) / 20.0, v0 + mv + ((v1 - v0 - 2.0 * mv) * j as f64) / 20.0);
                worst = worst.max(truth(&f, p).abs());
            }
        }
        worst
    }

    #[test]
    fn clamped_knots_greville_abscissae_and_a_partition_of_unity() {
        let knots = clamped_knots(6, 3, 0.0, 10.0);
        let rounded: Vec<f64> = knots.iter().map(|k| js_round(k * 1000.0) / 1000.0).collect();
        assert_eq!(rounded, vec![0.0, 0.0, 0.0, 0.0, 3.333, 6.667, 10.0, 10.0, 10.0, 10.0]);
        let g = greville(&knots, 6, 3);
        assert_eq!(g[0], 0.0);
        assert_eq!(g[5], 10.0);
        let mut b = [0.0; 4];
        for t in [0.0, 2.5, 5.0, 7.5, 10.0] {
            basis_functions(&knots, find_span(&knots, 6, 3, t), 3, t, &mut b);
            assert!((b[0] + b[1] + b[2] + b[3] - 1.0).abs() < 5e-13);
        }
    }

    #[test]
    fn auto_sizes_the_net_to_the_extent_over_the_scatter_within_the_limits() {
        assert_eq!(auto_net(80.0, 40.0, 1.0), Net { u: 10, v: 5 });
        assert_eq!(auto_net(2.0, 2.0, 1.0), Net { u: NET_MIN, v: NET_MIN });
        assert_eq!(auto_net(1000.0, 1000.0, 1.0), Net { u: NET_MAX, v: NET_MAX });
    }

    #[test]
    fn recovers_a_tilted_plane_from_noisy_points_within_the_noise() {
        let f = |_: f64, _: f64| 0.0;
        let xyz = sample(f, 40.0, 30.0, 1.0, 0.02, 1);
        // No scatter given: Auto reads the noise off a pilot fit.
        let fit = fit_surface(&xyz, &DEFAULT_SURFACE_FIT, None, PULL).unwrap();
        assert!(fit.rms < 0.03, "rms {}", fit.rms);
        assert!(worst_error(&fit, f, 0.15) < 0.01);
        // The margin: 35 % of the extent, half each side.
        assert!(fit.u_range[1] - fit.u_range[0] > 40.0 * 1.3);
        assert!(fit.u_range[1] - fit.u_range[0] < 40.0 * 1.4);
    }

    #[test]
    fn recovers_a_cylinder_patch_and_a_saddle() {
        let r = 30.0;
        let cyl = move |x: f64, _: f64| (r * r - x * x).sqrt() - r;
        let xyz = sample(cyl, 30.0, 20.0, 0.5, 0.02, 7);
        let f1 = fit_surface(&xyz, &SurfaceFitParams { net: Some(Net { u: 8, v: 4 }), ..DEFAULT_SURFACE_FIT }, Some(0.02), PULL).unwrap();
        assert!(f1.rms < 0.03);
        assert!(worst_error(&f1, cyl, 0.15) < 0.015);

        let saddle = |x: f64, y: f64| (x * x - y * y) / 80.0;
        let xyz = sample(saddle, 40.0, 40.0, 1.0, 0.02, 3);
        let f2 = fit_surface(&xyz, &DEFAULT_SURFACE_FIT, None, PULL).unwrap();
        assert!(f2.rms < 0.03);
        // Auto puts the net at its limit here, four points a cell: the surface follows the
        // noise a little, within a sigma and a half.
        assert!(worst_error(&f2, saddle, 0.15) < 0.03);
        // Auto on a curved patch: the pilot's residual is the noise, not the sag, so the net
        // gets its spans.
        assert!(f2.net.u > NET_MIN + 2);
    }

    #[test]
    fn smoothing_fairs_a_net_too_fine_for_its_points_and_the_grid_for_the_kernel_lies_on_the_fit() {
        let f = |_: f64, _: f64| 0.0;
        let xyz = sample(f, 20.0, 20.0, 2.0, 0.05, 11);
        // Twenty by twenty over 121 points: without the bending term the system is
        // underdetermined; with it the surface stays a plane.
        let params = SurfaceFitParams { net: Some(Net { u: 20, v: 20 }), smoothness: 0.05, ..DEFAULT_SURFACE_FIT };
        let fit = fit_surface(&xyz, &params, Some(0.05), PULL).unwrap();
        // Faired to the noise: without the term this stood at 1.2 mm.
        assert!(worst_error(&fit, f, 0.15) < 0.1);
        let grid = grid_of(&fit, GRID_PER_SPAN);
        assert_eq!(grid.rows, 19 * 3 + 1);
        assert_eq!(grid.points.len(), grid.rows * grid.cols);
        let q = eval_surface(&fit, fit.u_range[0], fit.v_range[0]);
        assert_eq!(grid.points[0], q);
    }

    #[test]
    fn refuses_too_few_points() {
        let err = fit_surface(&[DVec3::ZERO; 2], &DEFAULT_SURFACE_FIT, None, PULL).unwrap_err();
        assert!(err.0.contains("Too few"));
    }

    // ---- reSurfaceNormal.test.ts -------------------------------------------------------

    /// Points on a grid w × h, z = f(x, y).
    fn grid_patch(f: impl Fn(f64, f64) -> f64, w: f64, h: f64, step: f64) -> Vec<DVec3> {
        let mut pts = Vec::new();
        for x in steps(-w / 2.0, w / 2.0, step) {
            for y in steps(-h / 2.0, h / 2.0, step) {
                pts.push(DVec3::new(x, y, f(x, y)));
            }
        }
        pts
    }

    #[test]
    fn lays_the_plane_square_to_the_direction_given_the_principal_axes_in_it() {
        let [u, v, n] = height_frame(&[DVec3::new(0.8, 0.0, 0.6), DVec3::new(0.0, 1.0, 0.0), DVec3::new(-0.6, 0.0, 0.8)], Some(DVec3::new(0.0, 0.0, 2.0)));
        assert_eq!(n, DVec3::new(0.0, 0.0, 1.0));
        assert!(u.dot(n).abs() < 5e-13);
        assert!((u.x - 1.0).abs() < 5e-13);
        assert!(v.x.abs() < 5e-13 && (v.y - 1.0).abs() < 5e-13 && v.z.abs() < 5e-13);
        // None given, or none to speak of: the principal frame as it is.
        let identity = [DVec3::X, DVec3::Y, DVec3::Z];
        assert_eq!(height_frame(&identity, None), identity);
        assert_eq!(height_frame(&identity, Some(DVec3::ZERO))[2], DVec3::Z);
    }

    #[test]
    fn fits_a_hump_on_a_slope_along_z_where_its_principal_normal_leans() {
        let hump = |x: f64, y: f64| 0.5 * x + 4.0 * (-(x * x + y * y) / 150.0).exp();
        let xyz = grid_patch(hump, 40.0, 30.0, 0.5);
        let leaning = fit_surface(&xyz, &net(12), Some(0.01), PULL).unwrap();
        assert!(leaning.normal.z.abs() < 0.95);
        let along = fit_surface(&xyz, &SurfaceFitParams { normal: Some(DVec3::new(0.0, 0.0, 5.0)), ..net(12) }, Some(0.01), PULL).unwrap();
        assert_eq!(along.normal, DVec3::new(0.0, 0.0, 1.0));
        assert!(along.rms < 0.02);
        // (The TypeScript test also checks the app's command schema takes fit.normal; that
        // schema is the app's, not this module's.)
    }

    #[test]
    fn a_runaway_names_the_direction_and_the_spread_and_on_a_gentle_region_asks_for_a_net_or_a_direction() {
        let xyz = grid_patch(|_, _| 0.0, 20.0, 20.0, 0.5);
        let mut fit = fit_surface(&xyz, &net(20), Some(0.01), 0.0).unwrap();
        // A corner of the net thrown far off, out in the margin where no point is.
        fit.heights[0] = 500.0;
        let said = run_away(&fit, &xyz, DEFAULT_SURFACE_FIT.expand_pct, None, None).unwrap();
        assert!(
            said.contains("Its height direction was [0, 0, 1] (the region’s principal normal); 95 % of the region’s normals are within 0° of it.")
                || said.contains("Its height direction was [0, 0, -1] (the region’s principal normal); 95 % of the region’s normals are within 0° of it."),
            "{said}"
        );
        assert!(said.contains("Fit a coarser net, "), "{said}");
        assert!(said.ends_with(" or take less margin — or give fit.normal, the way the region faces, for its height direction."), "{said}");
        let asked = run_away(&fit, &xyz, DEFAULT_SURFACE_FIT.expand_pct, Some(DVec3::Z), None).unwrap();
        assert!(asked.contains("(fit.normal);") && asked.ends_with(" or take less margin."), "{asked}");
    }

    #[test]
    fn says_a_region_bending_round_past_70_degrees_is_no_height_field() {
        // A shell wrapping a shoulder: 200° of a cylinder R15 round the y axis, 40 mm long, its
        // points 0.5 mm apart.
        let mut pts = Vec::new();
        for a in steps(-100.0, 100.0, 2.0) {
            for y in steps(-20.0, 20.0, 0.5) {
                pts.push(DVec3::new(15.0 * ((a * PI) / 180.0).sin(), y, 15.0 * ((a * PI) / 180.0).cos()));
            }
        }
        let mut fit = fit_surface(&pts, &net(20), Some(0.01), 0.0).unwrap();
        fit.heights[0] = 1e4;
        let said = run_away(&fit, &pts, DEFAULT_SURFACE_FIT.expand_pct, None, None).unwrap();
        let tail = "° of it. The region bends round past what one height field holds — split it, or loft through stations.";
        assert!(said.ends_with(tail), "{said}");
        let head = "95 % of the region’s normals are within ";
        let at = said.rfind(head).unwrap() + head.len();
        let deg: f64 = said[at..said.len() - tail.len()].parse().unwrap();
        assert!(deg > 80.0, "{said}");
    }

    // ---- reSurfaceRunaway.test.ts ------------------------------------------------------

    /// Points z = f(x, y) on a disc of radius r with a hole of radius `hole` in it, every
    /// `step` mm, noisy in z by sigma.
    fn disc(f: impl Fn(f64, f64) -> f64, r: f64, hole: f64, step: f64, sigma: f64, seed: u32) -> Vec<DVec3> {
        let mut rnd = Lcg(seed);
        let mut pts = Vec::new();
        for x in steps(-r, r, step) {
            for y in steps(-r, r, step) {
                let d = js_hypot2(x, y);
                if d > r || d < hole {
                    continue;
                }
                pts.push(DVec3::new(x, y, f(x, y) + rnd.gauss() * sigma));
            }
        }
        pts
    }

    /// Points on a grid w × h, z = f(x, y), a few microns off it.
    fn wavy_patch(f: impl Fn(f64, f64) -> f64, w: f64, h: f64, step: f64) -> Vec<DVec3> {
        grid_patch(|x, y| f(x, y) + 0.01 * (7.0 * x + 3.0 * y).sin(), w, h, step)
    }

    /// Stepped levels with a slight twist: a housing's underside, near enough.
    fn levels(x: f64, y: f64) -> f64 {
        (if x > 10.0 { 4.0 } else { 0.0 }) + (if y < -20.0 { -3.0 } else { 0.0 }) + (if x < -30.0 { 2.0 } else { 0.0 }) + (x * y) / 2000.0
    }
    /// The levels curling up steeply into a rim from radius c.
    fn rim(c: f64, k: f64) -> impl Fn(f64, f64) -> f64 {
        move |x, y| levels(x, y) + (js_hypot2(x, y) - c).max(0.0).powi(2) * k
    }
    fn cylinder(x: f64, _: f64) -> f64 {
        (900.0 - x * x).sqrt() - 30.0
    }

    struct Over {
        u: f64,
        v: f64,
        h: f64,
    }
    fn over_plane(fit: &SurfaceFitResult, p: DVec3) -> Over {
        let d = p - fit.origin;
        let dot = |a: DVec3| d.x * a.x + d.y * a.y + d.z * a.z;
        Over { u: dot(fit.axis_u), v: dot(fit.axis_v), h: dot(fit.normal) }
    }

    /// How far the fitted surface leaves the points' height range anywhere.
    fn overshoot(fit: &SurfaceFitResult, xyz: &[DVec3]) -> f64 {
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for p in xyz {
            let h = over_plane(fit, *p).h;
            lo = lo.min(h);
            hi = hi.max(h);
        }
        let mut worst = 0.0f64;
        for a in 0..=40 {
            for b in 0..=40 {
                let p = eval_surface(
                    fit,
                    fit.u_range[0] + ((fit.u_range[1] - fit.u_range[0]) * a as f64) / 40.0,
                    fit.v_range[0] + ((fit.v_range[1] - fit.v_range[0]) * b as f64) / 40.0,
                );
                let h = over_plane(fit, p).h;
                worst = worst.max(h - hi).max(lo - h);
            }
        }
        worst
    }

    /// How far apart two fits of the same points are at the points.
    fn moved(a: &SurfaceFitResult, b: &SurfaceFitResult, xyz: &[DVec3]) -> f64 {
        let mut worst = 0.0f64;
        for p in xyz {
            let pa = over_plane(a, *p);
            let qa = eval_surface(a, pa.u, pa.v);
            let pb = over_plane(b, *p);
            let qb = eval_surface(b, pb.u, pb.v);
            worst = worst.max(hypot3(qa - qb));
        }
        worst
    }

    #[test]
    fn the_pull_holds_the_empty_corners_of_a_patchy_region_by_its_points() {
        for (xyz, params, sigma) in [(disc(levels, 60.0, 15.0, 1.5, 1.0, 5), net(14), 1.0), (disc(levels, 60.0, 45.0, 1.5, 0.5, 5), net(20), 0.5)] {
            let free = fit_surface(&xyz, &params, Some(sigma), 0.0).unwrap();
            let held = fit_surface(&xyz, &params, Some(sigma), PULL).unwrap();
            assert!(overshoot(&free, &xyz) > 4.0, "free overshoot {}", overshoot(&free, &xyz));
            assert!(overshoot(&held, &xyz) < 0.5, "held overshoot {}", overshoot(&held, &xyz));
            // At the points it moves by under half the residual, well under the noise.
            assert!(moved(&free, &held, &xyz) < 0.5 * free.rms + 1e-9);
            assert!(moved(&free, &held, &xyz) < sigma);
        }
    }

    #[test]
    fn the_pull_moves_a_curved_patch_at_its_points_by_under_the_noise_however_low_the_noise() {
        for sigma in [0.3, 0.02] {
            let xyz = disc(|x, y| (x * x + 0.6 * y * y) / 400.0, 60.0, 15.0, 1.5, sigma, 3);
            let free = fit_surface(&xyz, &net(14), Some(sigma), 0.0).unwrap();
            let held = fit_surface(&xyz, &net(14), Some(sigma), PULL).unwrap();
            assert!(moved(&free, &held, &xyz) < sigma);
            // The bowl carries on past its points as a bowl: its corners are bent back toward
            // the plane only a little.
            assert!(overshoot(&held, &xyz) > 10.0);
        }
    }

    #[test]
    fn the_pull_leaves_a_region_with_points_near_every_cell_as_it_was() {
        let xyz = wavy_patch(cylinder, 30.0, 20.0, 0.5);
        let params = SurfaceFitParams { net: Some(Net { u: 8, v: 4 }), ..DEFAULT_SURFACE_FIT };
        let free = fit_surface(&xyz, &params, Some(0.02), 0.0).unwrap();
        let held = fit_surface(&xyz, &params, Some(0.02), PULL).unwrap();
        assert_eq!(held.heights, free.heights);
    }

    #[test]
    fn a_fit_that_runs_away_is_refused_naming_the_spot_and_where_the_points_run_out() {
        let xyz = disc(rim(56.0, 1.0), 60.0, 15.0, 1.0, 0.5, 5);
        let fit = fit_surface(&xyz, &net(14), Some(0.5), PULL).unwrap();
        assert!(overshoot(&fit, &xyz) > 0.425 * 120.0);
        let said = run_away(&fit, &xyz, DEFAULT_SURFACE_FIT.expand_pct, None, None).unwrap();
        let mut s = Scan { s: &said };
        assert!(
            s.lit("The surface runs away from its points: it stands ")
                && s.num().is_some()
                && s.lit(" mm ")
                && s.either(&["above the highest", "below the lowest"])
                && s.lit(" of them at (")
                && s.num().is_some()
                && s.lit(", ")
                && s.num().is_some()
                && s.lit(", ")
                && s.num().is_some()
                && s.lit("), ")
                && s.num().is_some()
                && s.lit(" mm past where the region's points run out at ("),
            "{said}"
        );
        let refused = fit_surface_reported(&xyz, &net(14), None).unwrap_err();
        assert!(refused.to_string().contains("runs away from its points"), "{refused}");
    }

    #[test]
    fn a_curve_carried_on_past_its_points_is_not_refused() {
        let xyz = disc(|x, y| (x * x + 0.6 * y * y) / 400.0, 60.0, 15.0, 1.5, 0.3, 3);
        assert!(run_away(&fit_surface(&xyz, &net(14), Some(0.3), PULL).unwrap(), &xyz, DEFAULT_SURFACE_FIT.expand_pct, None, None).is_none());
        let cyl = wavy_patch(cylinder, 30.0, 20.0, 0.5);
        let params = SurfaceFitParams { net: Some(Net { u: 8, v: 4 }), ..DEFAULT_SURFACE_FIT };
        assert!(run_away(&fit_surface(&cyl, &params, Some(0.02), PULL).unwrap(), &cyl, DEFAULT_SURFACE_FIT.expand_pct, None, None).is_none());
    }

    // ---- reSurfaceStrays.test.ts -------------------------------------------------------

    fn inside(p: DVec3, b: &Bounds) -> bool {
        (0..3).all(|k| p[k] >= b.min[k] && p[k] <= b.max[k])
    }

    /// Points every `step` mm over x0…x1, y0…y1 at height f, noisy by ±sigma/2, less those in
    /// `hole`.
    #[allow(clippy::too_many_arguments)]
    fn sheet(pts: &mut Vec<DVec3>, x0: f64, x1: f64, y0: f64, y1: f64, f: impl Fn(f64, f64) -> f64, rnd: &mut Lcg, hole: Option<Bounds>, step: f64, sigma: f64) {
        for x in steps(x0, x1 + 1e-9, step) {
            for y in steps(y0, y1 + 1e-9, step) {
                let p = DVec3::new(x, y, f(x, y) + sigma * rnd.uniform());
                if hole.is_none_or(|h| !inside(p, &h)) {
                    pts.push(p);
                }
            }
        }
    }

    /// A flange at z 0, 80 × 60, with the floor of a cup 6 mm under it over x 0…20,
    /// y −10…10, and a lug standing 20 mm over a hole in it.
    fn flange_over_cup() -> Vec<DVec3> {
        let mut rnd = Lcg(3);
        let mut pts = Vec::new();
        let hole = Bounds { min: DVec3::new(-36.0, -26.0, -1.0), max: DVec3::new(-24.0, -14.0, 1.0) };
        sheet(&mut pts, -40.0, 40.0, -30.0, 30.0, |_, _| 0.0, &mut rnd, Some(hole), 1.0, 0.04);
        sheet(&mut pts, 0.0, 20.0, -10.0, 10.0, |_, _| -6.0, &mut rnd, None, 1.0, 0.04);
        sheet(&mut pts, -35.0, -25.0, -25.0, -15.0, |_, _| 20.0, &mut rnd, None, 1.0, 0.04);
        pts
    }

    fn without(xyz: &[DVec3], boxes: &[Bounds]) -> Vec<DVec3> {
        xyz.iter().copied().filter(|p| !boxes.iter().any(|b| inside(*p, b))).collect()
    }

    fn robust(n: usize) -> SurfaceFitParams {
        SurfaceFitParams { robust: true, ..net(n) }
    }

    #[test]
    fn a_robust_fit_names_a_second_sheet_under_it_and_the_points_it_left_out_far_off() {
        let xyz = flange_over_cup();
        let r = fit_surface_reported(&xyz, &robust(8), None).unwrap();
        let strays = r.strays.unwrap();
        assert_eq!(strays.len(), 2, "{strays:?}");
        let cup = strays.iter().find(|s| s.why.starts_with("a second sheet")).unwrap();
        assert!(cup.why.starts_with("a second sheet 6 mm under the one fitted, along its height direction"), "{}", cup.why);
        assert_eq!(cup.points, 21 * 21);
        assert_eq!(cup.off_mm, [6.0, 6.0]);
        // The box takes the cup's floor and none of the flange over it.
        assert!(cup.bounds.min.x <= -1.0);
        assert!(cup.bounds.max.x >= 21.0);
        assert!(cup.bounds.max.z < -4.5);
        let lug = strays.iter().find(|s| s.why.starts_with("left out by the robust fit")).unwrap();
        assert_eq!(lug.why, "left out by the robust fit, 20 mm over its surface");
        assert_eq!(lug.points, 11 * 11);
        assert!(inside(DVec3::new(-35.0, -25.0, 20.0), &lug.bounds) && inside(DVec3::new(-25.0, -15.0, 20.0), &lug.bounds));
        // Given back as excludes, the region fits as one sheet, with nothing left to name.
        let boxes: Vec<Bounds> = strays.iter().map(|s| s.bounds).collect();
        let again = fit_surface_reported(&without(&xyz, &boxes), &robust(8), None).unwrap();
        assert!(again.strays.is_none());
        assert!(again.fit.rms < 0.03);
    }

    #[test]
    fn a_robust_fit_names_none_on_a_clean_region_steep_as_it_gets() {
        // Half a cylinder R30 over its axis, out to 56° from its plane: a slope across a cell
        // is no second sheet.
        let mut pts = Vec::new();
        sheet(&mut pts, -25.0, 25.0, -20.0, 20.0, cylinder, &mut Lcg(5), None, 0.5, 0.02);
        assert!(fit_surface_reported(&pts, &robust(8), None).unwrap().strays.is_none());
    }

    /// reSurfaceRunaway's disc that runs away, with a second sheet 8 mm under part of it
    /// (this file's disc draws its noise uniform, ±sigma).
    fn runaway_with_sheet() -> Vec<DVec3> {
        let mut rnd = Lcg(5);
        let f = rim(56.0, 1.0);
        let mut pts = Vec::new();
        for x in steps(-60.0, 60.0, 1.0) {
            for y in steps(-60.0, 60.0, 1.0) {
                let d = js_hypot2(x, y);
                if d > 60.0 || d < 15.0 {
                    continue;
                }
                pts.push(DVec3::new(x, y, f(x, y) + 2.0 * rnd.uniform() * 0.5));
            }
        }
        sheet(&mut pts, 20.0, 36.0, -8.0, 8.0, |x, y| levels(x, y) - 8.0, &mut Lcg(9), None, 1.0, 0.04);
        pts
    }

    #[test]
    fn a_fit_that_runs_away_is_refused_with_its_strays_and_says_what_to_do_with_them() {
        let err = fit_surface_reported(&runaway_with_sheet(), &net(14), None).unwrap_err();
        let SurfaceFitError::Strays(refused) = &err else { panic!("not a strays refusal: {err}") };
        assert!(refused.message.starts_with("The surface runs away from its points"), "{}", refused.message);
        // The sheet named first, with its box; the rim the fit does not follow all the way
        // round would leave out the whole region, and is not named.
        let tail = " — give their boxes in region.exclude and fit again.";
        assert!(refused.message.ends_with(tail), "{}", refused.message);
        let at = refused.message.find(" Strays (strays): ").unwrap();
        let mut s = Scan { s: &refused.message[at..] };
        assert!(
            s.lit(" Strays (strays): ")
                && s.num().is_some()
                && s.lit(" points a second sheet ")
                && s.num_with("–").is_some()
                && s.lit(" mm under the one fitted, along its height direction in (19, -9, ")
                && s.num().is_some()
                && s.lit(")–(37, 9, ")
                && s.num().is_some()
                && s.lit(")")
                && s.either(&["; ", ", "])
                && s.s.len() >= tail.len(),
            "{}",
            refused.message
        );
        let sheet_stray = refused.strays.iter().find(|s| s.why.starts_with("a second sheet")).unwrap();
        assert!(sheet_stray.off_mm[0] > 6.0);
        assert!(inside(DVec3::new(28.0, 0.0, levels(28.0, 0.0) - 8.0), &sheet_stray.bounds));
    }

    /// A flat region whose net was thrown off in a corner, out in the margin where no point
    /// is — reSurfaceNormal's.
    fn thrown() -> (Vec<DVec3>, SurfaceFitResult) {
        let mut pts = Vec::new();
        for x in steps(-10.0, 10.0, 0.5) {
            for y in steps(-10.0, 10.0, 0.5) {
                pts.push(DVec3::new(x, y, 0.0));
            }
        }
        let mut fit = fit_surface(&pts, &net(20), Some(0.01), 0.0).unwrap();
        fit.heights[0] = 500.0;
        (pts, fit)
    }

    #[test]
    fn a_region_read_facing_a_way_says_its_normals_are_within_the_query_angle_by_construction() {
        let (xyz, fit) = thrown();
        let said = run_away(&fit, &xyz, DEFAULT_SURFACE_FIT.expand_pct, None, Some(&RegionFacing { dir: DVec3::new(0.0, 0.0, 3.0), within_deg: Some(28.0) })).unwrap();
        assert!(
            said.contains("(the region’s principal normal); the region was read facing [0, 0, 1] within 28°, so its normals are within 28° of that by construction. Fit a coarser net"),
            "{said}"
        );
        assert!(!said.contains("95 %"));
        // 30° where the query gave no angle.
        let said = run_away(&fit, &xyz, DEFAULT_SURFACE_FIT.expand_pct, None, Some(&RegionFacing { dir: DVec3::Z, within_deg: None })).unwrap();
        assert!(said.contains("within 30°, so its normals are within 30° of that"), "{said}");
    }

    #[test]
    fn a_region_read_facing_away_from_the_height_direction_names_the_way_it_faces_for_fit_normal() {
        let (xyz, fit) = thrown();
        let tilted = DVec3::new((PI / 6.0).sin(), 0.0, (PI / 6.0).cos());
        let said = run_away(&fit, &xyz, DEFAULT_SURFACE_FIT.expand_pct, None, Some(&RegionFacing { dir: tilted, within_deg: Some(20.0) })).unwrap();
        assert!(said.contains("so its normals are within 20° of that by construction, 30° off the height direction."), "{said}");
        assert!(said.ends_with("— or give fit.normal: [0.5, 0, 0.866], the way the region was read facing, for its height direction."), "{said}");
    }

    /// Adapted from reSurfaceStrays.test.ts's "answers them on the feature's result, and a
    /// region read facing a way says its normals are within it", which builds the feature
    /// through the app's Builder: the Builder hands the region, read off the scan's
    /// Float32Array, and its regionBox.normal to fitSurfaceReported, as here. The refusal is
    /// ScanRuler's, character for character (Node, on the same points).
    #[test]
    fn adapted_a_region_read_facing_a_way_answers_its_strays_and_says_its_normals_are_within_it() {
        let xyz: Vec<DVec3> = runaway_with_sheet().iter().map(|p| DVec3::new(p.x as f32 as f64, p.y as f32 as f64, p.z as f32 as f64)).collect();
        let facing = RegionFacing { dir: DVec3::new(0.0, 0.0, 2.0), within_deg: Some(28.0) };
        let err = fit_surface_reported(&xyz, &net(14), Some(&facing)).unwrap_err();
        let SurfaceFitError::Strays(refused) = &err else { panic!("not a strays refusal: {err}") };
        assert!(refused.message.contains("the region was read facing [0, 0, 1] within 28°, so its normals are within 28° of that by construction"), "{}", refused.message);
        assert!(!refused.message.contains("turn up to"));
        assert!(refused.strays.iter().any(|s| s.why.starts_with("a second sheet")));
        assert_eq!(
            refused.message,
            "The surface runs away from its points: it stands 103.4 mm above the highest of them at (73.3, 81.0, 129.0), 54.8 mm past where the region's points run out at (36.0, 48.0, 20.8) — more than the margin and a quarter of the region's 119.9 mm allow. Its height direction was [-0.031, -0.036, 0.999] (the region’s principal normal); the region was read facing [0, 0, 1] within 28°, so its normals are within 28° of that by construction, 3° off the height direction. Fit a coarser net, keep the region to one surface (a box closer round it, or exclude what stands off it), or take less margin — or give fit.normal: [0, 0, 1], the way the region was read facing, for its height direction. Strays (strays): 289 points a second sheet 7.6–8.5 mm under the one fitted, along its height direction in (19, -9, -5.2)–(37, 9, -2.8); 32 points a second sheet 1.4–2.7 mm under the one fitted, along its height direction in (-32, -17, -1.4)–(-28, 10, 3.6), and 3 more — give their boxes in region.exclude and fit again."
        );
    }

    // ---- reSplitSides.test.ts ----------------------------------------------------------

    #[test]
    fn carries_a_fitted_surface_on_over_the_bodys_points_not_its_boxs_corners() {
        // A square 20 across, turned 45° in the plane: its box's corners lie 14.1 past its
        // middle along u, its own corners 10.
        let c = std::f64::consts::FRAC_1_SQRT_2;
        let points: Vec<DVec3> = [(10.0, 0.0), (0.0, 10.0), (-10.0, 0.0), (0.0, -10.0)].iter().map(|&(x, y)| DVec3::new(x, y, 0.0)).collect();
        let turned = SurfacePlane { origin: DVec3::ZERO, axis_u: DVec3::new(c, c, 0.0), axis_v: DVec3::new(-c, c, 0.0), u_range: [-1.0, 1.0], v_range: [-1.0, 1.0] };
        let reach = reach_over(&turned, &points, 0.0).unwrap();
        assert!((reach.u[0] + 10.0 * c).abs() < 5e-7);
        assert!((reach.u[1] - 10.0 * c).abs() < 5e-7);
        assert!((reach.v[1] - 10.0 * c).abs() < 5e-7);
        assert!(reach_over(&SurfacePlane { u_range: [-10.0, 10.0], v_range: [-10.0, 10.0], ..turned }, &points, 0.0).is_none());
    }

    // ---- adapted -----------------------------------------------------------------------

    /// Adapted from reSurfaceRegion.test.ts, whose region is read off a scanned STL by the
    /// app's region query (a box facing a way): here the region is the sheet's vertices and
    /// the boss's top, as that query reads them, given to the fit directly. The skin:
    /// z = 0.004·x² + 0.006·y² over 50 × 40 mm, 1 mm apart and a few microns off it, with a
    /// boss 6 × 6 mm standing 3 mm proud on it at (10, 8).
    #[test]
    fn adapted_a_robust_fit_leaves_a_boss_out_by_itself_and_says_how_many_points_it_dropped() {
        let skin = |x: f64, y: f64| 0.004 * x * x + 0.006 * y * y;
        let mut pts = Vec::new();
        for x in -25..=25 {
            for y in -20..=20 {
                let (x, y) = (x as f64, y as f64);
                pts.push(DVec3::new(x, y, skin(x, y) + 0.003 * (3.1 * x + 1.7 * y).sin()));
            }
        }
        let z1 = skin(10.0, 8.0) + 3.0;
        for i in 0..=6 {
            for j in 0..=6 {
                pts.push(DVec3::new(7.0 + i as f64, 5.0 + j as f64, z1));
            }
        }
        let pts: Vec<DVec3> = pts.iter().map(|p| DVec3::new(p.x as f32 as f64, p.y as f32 as f64, p.z as f32 as f64)).collect();
        let plain = fit_surface_reported(&pts, &DEFAULT_SURFACE_FIT, None).unwrap();
        let r = fit_surface_reported(&pts, &SurfaceFitParams { robust: true, ..DEFAULT_SURFACE_FIT }, None).unwrap();
        // The boss's top goes; the skin stays.
        assert!(r.dropped >= 30, "dropped {}", r.dropped);
        assert!(r.fit.rms < plain.fit.rms / 5.0, "robust {} plain {}", r.fit.rms, plain.fit.rms);
        assert!(r.fit.rms < 0.01);
        for w in &r.worst {
            assert!(w.off.abs() < 0.05);
        }
        assert_eq!(r.worst.len(), WORST_POINTS);
    }

    /// Adapted from reNetworkFit.test.ts, whose region is the vertices of a patch network's
    /// cell on a scanned cylinder: here those vertices are taken off the same grid directly
    /// — 270° of a cylinder R20 between z 5 and 35, the grid 2° and 1 mm apart, noisy along
    /// its normal by 0.02 mm. No height field holds a region that bends round 270°.
    #[test]
    fn adapted_fit_surface_refuses_a_strip_round_270_degrees_of_a_cylinder() {
        let mut rand = crate::fit2d::Mulberry32::new(5);
        let at = |u: f64, v: f64| DVec3::new(20.0 * (2.0 * PI * u).cos(), 20.0 * (2.0 * PI * u).sin(), 40.0 * v);
        let mut region = Vec::new();
        for j in 0..=40 {
            for i in 0..180 {
                let (u, v) = (i as f64 / 180.0, j as f64 / 40.0);
                let p = at(u, v);
                let e = 1e-4;
                let du = at(u + e, v) - p;
                let dv = at(u, v + e) - p;
                let n = du.cross(dv);
                let l = if hypot3(n) == 0.0 { 1.0 } else { hypot3(n) };
                let g = {
                    let a = rand.next_f64().max(1e-12);
                    (-2.0 * a.ln()).sqrt() * (2.0 * PI * rand.next_f64()).cos()
                } * 0.02;
                let q = p + n / l * g;
                if i <= 135 && (5..=35).contains(&j) {
                    region.push(DVec3::new(q.x as f32 as f64, q.y as f32 as f64, q.z as f32 as f64));
                }
            }
        }
        let err = fit_surface_reported(&region, &DEFAULT_SURFACE_FIT, None).unwrap_err();
        assert!(err.to_string().contains("bends round past what one height field holds"), "{err}");
    }

    // ---- not in ScanRuler's tests: the pieces its kernel tests exercise ------------------

    #[test]
    fn grid_area_carried_height_and_the_reaching_grid() {
        let xyz = grid_patch(|x, y| 0.1 * x + 0.05 * y, 20.0, 10.0, 0.5);
        let fit = fit_surface(&xyz, &net(6), Some(0.01), PULL).unwrap();
        // A plane's grid has the plane's area over its range.
        let grid = grid_of(&fit, GRID_PER_SPAN);
        let span = (fit.u_range[1] - fit.u_range[0]) * (fit.v_range[1] - fit.v_range[0]);
        assert!((grid_area(&grid) - span).abs() < 1e-6 * span);
        // Carried on past its edge, a plane goes on as itself.
        for (u, v) in [(fit.u_range[1] + 30.0, 0.0), (fit.u_range[0] - 5.0, fit.v_range[1] + 7.0)] {
            assert!(carried_height(&fit, u, v).abs() < 1e-6);
        }
        let reach = reaching_grid_of(&fit, [fit.u_range[0] - 100.0, fit.u_range[1] + 100.0], fit.v_range, GRID_PER_SPAN);
        assert_eq!(reach.rows, 96);
        assert_eq!(reach.cols, grid.cols);
    }

    #[test]
    fn mesh_triangles_grow_with_the_bending() {
        let gentle = fit_surface(&wavy_patch(|x, y| (x * x + y * y) / 4000.0, 100.0, 100.0, 2.0), &net(10), Some(0.01), PULL).unwrap();
        let egg = fit_surface(&wavy_patch(|x, y| 30.0 * (x / 5.1).sin() * (y / 5.1).sin(), 100.0, 100.0, 1.0), &SurfaceFitParams { smoothness: 0.0, expand_pct: 0.0, ..net(20) }, Some(0.01), PULL).unwrap();
        let a = mesh_triangles(&gentle, 0.05, MESH_ANGLE);
        let b = mesh_triangles(&egg, 0.05, MESH_ANGLE);
        assert!(a > 0.0 && b > 10.0 * a, "{a} {b}");
        assert!(mesh_triangles(&gentle, 0.02, MESH_ANGLE) >= a);
    }

    #[test]
    fn reads_scanruler_json_and_answers_in_its_names() {
        // A feature's fit as ScanRuler saves it, and the defaults it opens with.
        let saved: SurfaceFitParams = serde_json::from_str(r#"{"net":{"u":8,"v":6},"degree":3,"expandPct":20,"smoothness":0.01,"robust":true,"normal":[0,0,1]}"#).unwrap();
        assert_eq!(saved, SurfaceFitParams { net: Some(Net { u: 8, v: 6 }), expand_pct: 20.0, robust: true, normal: Some(DVec3::Z), ..DEFAULT_SURFACE_FIT });
        let fresh: SurfaceFitParams = serde_json::from_str(r#"{"net":null,"degree":3,"expandPct":35,"smoothness":0.01}"#).unwrap();
        assert_eq!(fresh, DEFAULT_SURFACE_FIT);
        let facing: RegionFacing = serde_json::from_str(r#"{"dir":[0,0,2],"withinDeg":28}"#).unwrap();
        assert_eq!(facing.within_deg, Some(28.0));
        // A fit and a stray answer as ScanRuler's do.
        let fit = fit_surface(&grid_patch(|x, y| 0.1 * x - 0.05 * y, 10.0, 10.0, 1.0), &net(4), Some(0.01), PULL).unwrap();
        let json = serde_json::to_value(&fit).unwrap();
        for key in ["axisU", "axisV", "uRange", "vRange", "knotsU", "knotsV", "heights", "nu", "nv"] {
            assert!(json.get(key).is_some(), "{key} in {json}");
        }
        assert!(json.get("footprint").is_none());
        let back: SurfaceFitResult = serde_json::from_value(json).unwrap();
        assert_eq!(back, fit);
        let stray = Stray { points: 12, bounds: Bounds { min: DVec3::ZERO, max: DVec3::ONE }, off_mm: [1.0, 2.0], why: String::new() };
        let json = serde_json::to_value(&stray).unwrap();
        assert!(json.get("offMm").is_some() && json.get("box").is_some(), "{json}");
    }

    #[test]
    fn a_fit_whose_arrays_disagree_has_no_height_rather_than_a_panic() {
        // Not in ScanRuler's tests: a fit read back with its heights cut short, or its knots,
        // answers NaN where ScanRuler reads `undefined` past its arrays.
        let mut fit = fit_surface(&grid_patch(|_, _| 0.0, 10.0, 10.0, 1.0), &net(4), Some(0.01), PULL).unwrap();
        assert!(eval_surface(&fit, 0.0, 0.0).is_finite());
        fit.heights.truncate(3);
        assert!(eval_surface(&fit, 0.0, 0.0).z.is_nan());
        assert_eq!(grid_of(&fit, GRID_PER_SPAN).points.len(), 10 * 10);
        fit.knots_u.clear();
        assert!(carried_height(&fit, 100.0, 0.0).is_nan());
        assert!(mesh_triangles(&fit, 0.05, MESH_ANGLE).is_nan());
        // Points no number, or out past any part, name no strays and turn no normals.
        let mut xyz = grid_patch(|_, _| 0.0, 10.0, 10.0, 1.0);
        let good = fit_surface(&xyz, &net(4), Some(0.01), PULL).unwrap();
        xyz.push(DVec3::new(f64::INFINITY, 0.0, 1e300));
        xyz.push(DVec3::NAN);
        let _ = stray_groups(&good, &xyz, Some(RobustAt { med: 0.0, limit: 0.01 }));
        let _ = normal_spread(&xyz, DVec3::Z);
        let _ = run_away(&good, &xyz, 35.0, None, None);
    }

    #[test]
    fn javascript_numbers_as_the_messages_write_them() {
        assert_eq!(js_round(2.5), 3.0);
        assert_eq!(js_round(-2.5), -2.0);
        assert_eq!(js_round(0.49999999999999994), 0.0);
        assert_eq!(js_num(-0.0), "0");
        assert_eq!(js_num(0.866), "0.866");
        assert_eq!(js_num(1e21), "1e+21");
        assert_eq!(js_num(f64::INFINITY), "Infinity");
        assert_eq!(to_fixed1(0.25), "0.3");
        assert_eq!(to_fixed1(-0.25), "-0.3");
        assert_eq!(to_fixed1(0.35), "0.3");
        assert_eq!(to_fixed1(-0.04), "-0.0");
        assert_eq!(to_fixed1(12.0), "12.0");
        // As Node writes them: ties far from zero, and from 1e21 up the exponent form.
        // 2^50 + 0.25, a tie at the tenths.
        assert_eq!(to_fixed1(1125899906842624.2), "1125899906842624.3");
        assert_eq!(to_fixed1(2251799813685248.5), "2251799813685248.5");
        assert_eq!(to_fixed1(123456789.75), "123456789.8");
        assert_eq!(to_fixed1(-0.75), "-0.8");
        assert_eq!(to_fixed1(0.05), "0.1");
        assert_eq!(to_fixed1(1.45), "1.4");
        assert_eq!(to_fixed1(999999999999999900000.0), "999999999999999868928.0");
        assert_eq!(to_fixed1(9.158655086897401e21), "9.158655086897401e+21");
        assert_eq!(to_fixed1(-4.578656688994482e23), "-4.578656688994482e+23");
        assert_eq!(to_fixed1(1e21), "1e+21");
        assert_eq!(to_fixed1(5e-7), "0.0");
    }
}
