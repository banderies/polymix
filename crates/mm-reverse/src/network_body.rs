// SPDX-License-Identifier: AGPL-3.0-only
//! A patch network made a body (PN9): the patches' grids and control nets for the kernel,
//! the faces sewn through it, the network's report, and its ghost while it is drawn.
//!
//! Ported from ScanRuler (`plugins/reverse/core/networkBody.ts`, with `gridArea`,
//! `GRID_RETRIES` and `GRID_AREA_SLACK` of `plugins/reverse/core/surfaceFit.ts` and
//! `meshDeflection` of `plugins/reverse/core/buildTypes.ts`), © Stefan Hermann
//! (CNC Kitchen), AGPL-3.0-only.
//!
//! The kernel takes no control net: its B-spline surface is a surface through a grid of
//! points, within a micron of them (the kernel's `surfaceThrough`). So each patch is sampled
//! on a grid a few times finer than its net and that grid goes to the kernel, with Fit
//! Surface's retries at denser grids and its check on the area. A shared curve's samples are
//! the curve's own, taken once and handed to both patches either side of it — the same points
//! on both grids — so the two faces' edges lie within the kernel's micron of each other along
//! it, and sew. The faces are sewn without being unified: each patch stays one face, traced
//! back to its cell. A network that closes sews into a solid; an open one into a sheet. The
//! edges left open must be the network's own outer boundary; one that lies along a shared
//! curve is named, with the gap the faces left there.
//!
//! ScanRuler's kernel is Open CASCADE behind its `ReKernel`. Here the kernel is
//! [`NetworkKernel`], which the build implements: [`network_body`] is ScanRuler's
//! `networkBody` with every call into the kernel made through it. A kernel that can make a
//! face from a control net directly takes [`patch_surfaces`] instead of the grids: each
//! patch's poles and clamped knots, a shared curve's poles the boundary row of both patches
//! beside it, so their edges are the same curve exactly.

use std::collections::HashMap;
use std::time::Instant;

use glam::DVec3;
use serde::{Deserialize, Serialize};

use crate::fit2d::FitError;
use crate::network::{CellOutcome, NetworkReading};
use crate::network_cells::{fixed1, Unfilled};
use crate::network_fit::{on_side, BodyReading, FittedPatch, IrregularCorner, NetworkFitResult, Strip, WorstPoint};
use crate::network_spline::{curve_point, knots_of, patch_point, DEGREE};

/// Grid samples per net span; denser on a retry (GRID_RETRIES).
pub const GRID_PER_SPAN: usize = 3;
/// The ghost's samples a patch side at the fewest: a net of a few spans at three samples a
/// span — 25 across a sixth of a sphere — meshed it 0.1 % short of the solid made of the same
/// patches.
const GHOST_SIDE: usize = 48;
/// The sewing tolerance, mm: the grid route's edges lie within the kernel's micron of each
/// other, and sewing at a thousandth of a millimetre has taken them; more where the gaps read
/// more.
pub const SEW_TOLERANCE: f64 = 1e-3;
/// How many points along each shared curve its gap is read at.
const GAP_SAMPLES: usize = 8;
/// A patch's samples a side in its report.
pub const PATCH_GRID: usize = 9;
/// The denser grids tried when the kernel's surface through the first does not follow it
/// (Fit Surface's): the kernel approximates the grid, and on some grids that comes back
/// swinging between the samples. Five a span and more followed it within a micron every
/// time.
const GRID_RETRIES: [usize; 2] = [6, 8];
/// How much more area than the grid's own facets the kernel's surface through it may have
/// and still follow it (Fit Surface's): a smooth fit sampled three a span has well under
/// 0.1 % more; a surface swinging off it, from 50 % to a millionfold.
const GRID_AREA_SLACK: f64 = 1.02;

// ---------------------------------------------------------------------------------------
// The kernel.

/// A regular grid of points for the kernel: `rows` along u, `cols` along v, row-major — the
/// shape of Fit Surface's grid.
#[derive(Debug, Clone, PartialEq)]
pub struct PatchGrid {
    pub points: Vec<DVec3>,
    pub rows: usize,
    pub cols: usize,
}

/// An edge of a shape as the kernel reads it: its middle, and how many faces meet at it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KernelEdge {
    pub mid: DVec3,
    pub faces: usize,
}

/// A face of a shape as the kernel reads it: its hash, and the point at the middle of its
/// parameter box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FacePoint {
    pub hash: u64,
    pub at: DVec3,
}

/// What [`network_body`] asks of the kernel: ScanRuler's `ReKernel` calls it makes, each
/// failing with the kernel's own message.
pub trait NetworkKernel {
    type Shape;
    /// A face through a grid of points, within a micron of them (`surfaceThrough`).
    fn surface_through(&mut self, grid: &PatchGrid) -> Result<Self::Shape, FitError>;
    /// A shape's surface area, mm².
    fn surface_area(&mut self, shape: &Self::Shape) -> Result<f64, FitError>;
    /// How far a point is from a shape, mm.
    fn distance_to_point(&mut self, shape: &Self::Shape, point: DVec3) -> Result<f64, FitError>;
    fn copy(&mut self, shape: &Self::Shape) -> Result<Self::Shape, FitError>;
    /// The faces sewn at `tolerance`, and made a solid where `closed` asks and they close.
    fn sew_patches(&mut self, faces: &[Self::Shape], tolerance: f64, closed: bool) -> Result<Self::Shape, FitError>;
    /// Whether a shape is a valid closed solid with a volume.
    fn is_closed_solid(&mut self, shape: &Self::Shape) -> Result<bool, FitError>;
    /// A shape's size: the diagonal of its box (`diagonalOf`).
    fn diagonal(&mut self, shape: &Self::Shape) -> Result<f64, FitError>;
    /// A shape's edges, read at a chord tolerance.
    fn edges(&mut self, shape: &Self::Shape, deflection: f64) -> Result<Vec<KernelEdge>, FitError>;
    /// A shape's faces, each with its hash and the point at the middle of its parameter box.
    fn face_points(&mut self, shape: &Self::Shape) -> Result<Vec<FacePoint>, FitError>;
    /// A shape let go; dropping it, unless the kernel says otherwise.
    fn release(&mut self, shape: Self::Shape) {
        drop(shape);
    }
    /// A face made from a patch's control net directly ([`PatchSurface`]), for a kernel that
    /// can: its edge along a shared curve is then the curve exactly, and the grids and their
    /// retries are not needed. `None` (the default) takes the grid route, as ScanRuler's
    /// kernel does.
    fn surface_from_net(&mut self, net: &PatchSurface) -> Option<Result<Self::Shape, FitError>> {
        let _ = net;
        None
    }
}

/// The chord tolerance a body is meshed to, off its size — capped at 0.05 mm, so a large part
/// zoomed in on is not drawn in facets (the build's `mesh_deflection`).
fn mesh_deflection(diagonal: f64) -> f64 {
    0.05f64.min(0.005f64.max(diagonal * 5e-4))
}

/// The area of a grid's facets, two triangles a cell: what the kernel's surface through the
/// grid should have, near enough (Fit Surface's `gridArea`).
fn grid_area(grid: &PatchGrid) -> f64 {
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
                let n = DVec3::new(x.y * y.z - x.z * y.y, x.z * y.x - x.x * y.z, x.x * y.y - x.y * y.x);
                area += 0.5 * (n.x * n.x + n.y * n.y + n.z * n.z).sqrt();
            }
            c += 1;
        }
        r += 1;
    }
    area
}

// ---------------------------------------------------------------------------------------
// The report.

/// A patch's net, its span counts.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ReportNet {
    pub u: usize,
    pub v: usize,
}

/// A patch in a network's report: how it sits on its points, its net, the curves of its four
/// sides and its corners' nodes, and the patch sampled PATCH_GRID × PATCH_GRID — what the
/// viewport picks and outlines it by.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReportPatch {
    pub seed: DVec3,
    pub rms: f64,
    pub max: f64,
    pub misfit: f64,
    pub net: ReportNet,
    pub points: usize,
    pub worst: Vec<WorstPoint>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dropped: Option<usize>,
    pub sides: Vec<Vec<u32>>,
    pub corners: Vec<u32>,
    /// x, y, z per sample, row-major along u.
    pub grid: Vec<f32>,
}

/// A shared curve in a network's report: the gap between the two faces along it (mm, as the
/// kernel's faces lie — on the ghost, which makes none, as the patches do), and the largest
/// angle between the patches' normals along it (deg).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ReportCurve {
    pub curve: u32,
    pub gap: f64,
    pub angle: f64,
    pub crease: bool,
}

/// A spoke of a cell split from its centre: the feature's cell and the spoke's path.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReportSpoke {
    pub cell: usize,
    pub points: Vec<f32>,
}

/// A split cell's centre, by the feature's cell.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ReportCentre {
    pub cell: usize,
    pub at: DVec3,
}

/// How long each stage of the read and the body took, ms.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct ReportTimes {
    pub trace: f64,
    pub cells: f64,
    pub map: f64,
    pub fit: f64,
    pub faces: f64,
}

/// A network's reading of itself, as its feature's row and an agent read it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NetworkReport {
    pub patches: Vec<ReportPatch>,
    pub cells: Vec<CellOutcome>,
    /// Closed regions of three to six sides no seed fills — Fill all's.
    pub unfilled: Vec<Unfilled>,
    /// Each shared curve: its gap and angle (see [`ReportCurve`]).
    pub curves: Vec<ReportCurve>,
    pub corners: Vec<IrregularCorner>,
    pub strips: Vec<Strip>,
    /// Curves left out: a node gone, a trace empty.
    pub lost: Vec<u32>,
    /// The cells split from their centres: each spoke's path, and each centre, by the
    /// feature's cell — what the viewport draws and drags.
    pub spokes: Vec<ReportSpoke>,
    pub centres: Vec<ReportCentre>,
    /// Each patch side beside a body's face: the angle to the face along it (deg) and the
    /// curve's stand-off from the body's edge (mm).
    pub bodies: Vec<BodyReading>,
    pub closed: bool,
    /// The worst angle across a tangent curve and the worst gap.
    pub worst_angle: f64,
    pub worst_gap: f64,
    pub ms: ReportTimes,
}

/// A fitted patch as its report row has it.
fn report_patch(p: &FittedPatch) -> ReportPatch {
    ReportPatch {
        seed: p.seed,
        rms: p.rms,
        max: p.max,
        misfit: p.misfit,
        net: ReportNet { u: p.net.nu, v: p.net.nv },
        points: p.points,
        worst: p.worst.clone(),
        dropped: p.dropped,
        sides: p.sides.iter().map(|s| s.edges.iter().map(|h| h.curve).collect()).collect(),
        corners: p.corners.clone(),
        grid: sampled(p),
    }
}

/// A reading's spokes and centres for its report.
fn split_of<F>(reading: &NetworkReading<F>) -> (Vec<ReportSpoke>, Vec<ReportCentre>) {
    match &reading.split {
        Some(s) => (s.spokes.iter().map(|x| ReportSpoke { cell: x.cell, points: x.points.clone() }).collect(), s.centres.iter().map(|c| ReportCentre { cell: c.cell, at: c.at }).collect()),
        None => (Vec::new(), Vec::new()),
    }
}

fn times<F>(reading: &NetworkReading<F>, faces: f64) -> ReportTimes {
    ReportTimes { trace: reading.ms.trace, cells: reading.ms.cells, map: reading.ms.map, fit: reading.ms.fit, faces }
}

/// The report of a reading with no body: its cells and — where the patches were fitted and
/// the body could not be made of them — the fit's own readings, the gaps the patches' (none
/// the kernel's faces').
pub fn empty_report(reading: &NetworkReading<NetworkFitResult>) -> NetworkReport {
    let (spokes, centres) = split_of(reading);
    let Some(fit) = &reading.fit else {
        return NetworkReport {
            patches: Vec::new(),
            cells: reading.outcomes.clone(),
            unfilled: reading.cells.unfilled.clone(),
            curves: Vec::new(),
            corners: Vec::new(),
            strips: Vec::new(),
            lost: reading.traced.lost.clone(),
            spokes,
            centres,
            bodies: Vec::new(),
            closed: false,
            worst_angle: 0.0,
            worst_gap: 0.0,
            ms: times(reading, 0.0),
        };
    };
    let curves: Vec<ReportCurve> = fit.shared.iter().map(|s| ReportCurve { curve: s.curve, gap: s.gap, angle: s.angle, crease: s.crease }).collect();
    NetworkReport {
        patches: fit.patches.iter().map(report_patch).collect(),
        cells: reading.outcomes.clone(),
        unfilled: reading.cells.unfilled.clone(),
        worst_angle: curves.iter().filter(|s| !s.crease).fold(0.0, |m, s| m.max(s.angle)),
        worst_gap: curves.iter().fold(0.0, |m, s| m.max(s.gap)),
        curves,
        corners: fit.corners.clone(),
        strips: fit.strips.clone(),
        lost: reading.traced.lost.clone(),
        spokes,
        centres,
        bodies: fit.bodies.clone(),
        closed: false,
        ms: times(reading, 0.0),
    }
}

// ---------------------------------------------------------------------------------------
// The grids.

/// Each patch's side samples at a count: a curve sampled once at each count, forwards, and a
/// side backwards on it given the samples in the other order — the same points on both grids
/// either side of it.
pub struct SideSampler<'a> {
    fit: &'a NetworkFitResult,
    samples: HashMap<(u32, usize), Vec<DVec3>>,
}

impl<'a> SideSampler<'a> {
    pub fn new(fit: &'a NetworkFitResult) -> Self {
        SideSampler { fit, samples: HashMap::new() }
    }

    /// A curve sampled at `count` evenly spaced parameters, once.
    fn curve_samples(&mut self, curve: u32, count: usize) -> Option<&Vec<DVec3>> {
        let fit = self.fit;
        if let std::collections::hash_map::Entry::Vacant(e) = self.samples.entry((curve, count)) {
            let f = fit.curves.get(&curve)?;
            e.insert((0..count).map(|i| curve_point(&f.poles, f.spans, param(i, count))).collect());
        }
        self.samples.get(&(curve, count))
    }

    /// Side k of a patch sampled at `count` points, from its start.
    pub fn side_samples(&mut self, patch: &FittedPatch, k: usize, count: usize) -> Vec<DVec3> {
        let Some(side) = patch.sides.get(k) else {
            // No such side: the patch's own edge there.
            return (0..count)
                .map(|i| {
                    let (u, v) = on_side(k, param(i, count));
                    patch_point(&patch.net, u, v).0
                })
                .collect();
        };
        if let (true, Some(&edge)) = (side.single, side.edges.first()) {
            if let Some(s) = self.curve_samples(edge.curve, count) {
                return if edge.forward { s.clone() } else { s.iter().rev().copied().collect() };
            }
        }
        // A side of more than one curve — a T-junction's — has a row of its own; its
        // neighbours' edges are not its samples.
        let spans = if k % 2 == 0 { patch.net.nu } else { patch.net.nv };
        (0..count).map(|i| curve_point(&side.poles, spans, param(i, count))).collect()
    }
}

/// Sample i of `count` evenly spaced over [0, 1].
fn param(i: usize, count: usize) -> f64 {
    if count > 1 {
        i as f64 / (count - 1) as f64
    } else {
        0.0
    }
}

/// A patch's grid for the kernel: `per_span` samples a span — more, to make `fewest` a side at
/// least, by its span count alone, so two patches either side of a curve still take the same
/// samples on it — row-major with u along the rows, its boundary rows the shared samples of
/// its sides.
pub fn patch_grid(patch: &FittedPatch, per_span: usize, sampler: &mut SideSampler<'_>, fewest: usize) -> PatchGrid {
    let nu = patch.net.nu.max(1);
    let nv = patch.net.nv.max(1);
    let rows = nu * per_span.max(fewest.div_ceil(nu)) + 1;
    let cols = nv * per_span.max(fewest.div_ceil(nv)) + 1;
    let mut points = vec![DVec3::ZERO; rows * cols];
    for i in 1..rows - 1 {
        for j in 1..cols - 1 {
            points[i * cols + j] = patch_point(&patch.net, i as f64 / (rows - 1) as f64, j as f64 / (cols - 1) as f64).0;
        }
    }
    // Side 0 runs along v = 0 from u = 0, side 1 along u = 1 from v = 0, side 2 along v = 1
    // from u = 1, side 3 along u = 0 from v = 1.
    let s0 = sampler.side_samples(patch, 0, rows);
    let s1 = sampler.side_samples(patch, 1, cols);
    let s2 = sampler.side_samples(patch, 2, rows);
    let s3 = sampler.side_samples(patch, 3, cols);
    for i in 0..rows {
        points[i * cols] = s0[i];
        points[i * cols + cols - 1] = s2[rows - 1 - i];
    }
    for j in 0..cols {
        points[(rows - 1) * cols + j] = s1[j];
        points[j] = s3[cols - 1 - j];
    }
    PatchGrid { points, rows, cols }
}

/// A patch as a clamped cubic B-spline surface, for a kernel that makes a face from a control
/// net: `rows` × `cols` poles row-major with u along the rows, and the full knot vectors each
/// way. A single side's boundary row is its curve's poles, so the two patches beside a shared
/// curve have the same edge exactly.
#[derive(Debug, Clone, PartialEq)]
pub struct PatchSurface {
    pub degree: usize,
    pub rows: usize,
    pub cols: usize,
    pub poles: Vec<DVec3>,
    pub knots_u: Vec<f64>,
    pub knots_v: Vec<f64>,
    /// The cell's seed, where the face is traced back to.
    pub seed: DVec3,
}

/// Every fitted patch as a B-spline surface (see [`PatchSurface`]).
pub fn patch_surfaces(fit: &NetworkFitResult) -> Vec<PatchSurface> {
    fit.patches
        .iter()
        .map(|p| PatchSurface {
            degree: DEGREE,
            rows: p.net.nu + DEGREE,
            cols: p.net.nv + DEGREE,
            poles: p.net.poles.chunks_exact(3).map(|c| DVec3::new(c[0], c[1], c[2])).collect(),
            knots_u: knots_of(p.net.nu),
            knots_v: knots_of(p.net.nv),
            seed: p.seed,
        })
        .collect()
}

// ---------------------------------------------------------------------------------------
// The body.

/// The network's faces sewn into a body: the shape, its report, and each face of the body (by
/// the kernel's hash) to its patch.
#[derive(Debug)]
pub struct NetworkBody<S> {
    pub shape: S,
    pub report: NetworkReport,
    pub face_of: HashMap<u64, usize>,
}

/// The network's faces sewn into a body, and its report — or why not.
pub fn network_body<K: NetworkKernel>(k: &mut K, reading: &NetworkReading<NetworkFitResult>) -> Result<NetworkBody<K::Shape>, FitError> {
    let Some(fit) = &reading.fit else {
        return Err(FitError::new("No cell of the network is filled: there are no patches to make a body of."));
    };
    let began = Instant::now();
    let mut faces: Vec<K::Shape> = Vec::new();
    let out = sew(k, reading, fit, &mut faces, began);
    for f in faces {
        k.release(f);
    }
    out
}

/// [`network_body`]'s work, the faces it makes left in `faces` for it to let go of.
fn sew<K: NetworkKernel>(k: &mut K, reading: &NetworkReading<NetworkFitResult>, fit: &NetworkFitResult, faces: &mut Vec<K::Shape>, began: Instant) -> Result<NetworkBody<K::Shape>, FitError> {
    let mut sampler = SideSampler::new(fit);
    let nets = patch_surfaces(fit);
    for (index, patch) in fit.patches.iter().enumerate() {
        // A kernel that makes a face from the net takes it as fitted.
        if let Some(face) = nets.get(index).and_then(|n| k.surface_from_net(n)) {
            faces.push(face?);
            continue;
        }
        let mut made: Option<K::Shape> = None;
        let mut off = 0.0;
        for per_span in [GRID_PER_SPAN, GRID_RETRIES[0], GRID_RETRIES[1]] {
            let grid = patch_grid(patch, per_span, &mut sampler, 0);
            let face = k.surface_through(&grid)?;
            off = k.surface_area(&face)? / grid_area(&grid);
            if !(off > GRID_AREA_SLACK) {
                made = Some(face);
                break;
            }
            k.release(face);
        }
        let Some(made) = made else {
            let by = if off > 100.0 { format!("{} times", js_round(off)) } else { format!("{} % more than", js_round((off - 1.0) * 100.0)) };
            return Err(FitError::new(format!("The kernel's surface through the patch at {} does not follow it: {by} its area at every grid it was given. Give its strips fewer spans.", where_(patch.seed))));
        };
        faces.push(made);
    }
    // The gaps along the shared curves, as the kernel's faces lie: a point of the curve
    // between two grid samples, its distance from each face.
    let mut shared: Vec<ReportCurve> = Vec::with_capacity(fit.shared.len());
    for s in &fit.shared {
        let mut gap = 0.0f64;
        if let (Some(f), Some(a), Some(b)) = (fit.curves.get(&s.curve), faces.get(s.patches[0]), faces.get(s.patches[1])) {
            for i in 0..GAP_SAMPLES {
                let q = curve_point(&f.poles, f.spans, (i as f64 + 0.5) / GAP_SAMPLES as f64);
                gap = gap.max(k.distance_to_point(a, q)? + k.distance_to_point(b, q)?);
            }
        }
        shared.push(ReportCurve { curve: s.curve, gap: gap.max(s.gap), angle: s.angle, crease: s.crease });
    }
    let worst_gap = shared.iter().fold(0.0f64, |m, s| m.max(s.gap));
    // Closed: every side of every patch is shared.
    let sides: usize = fit.patches.iter().map(|p| p.sides.len()).sum();
    let closed = fit.shared.len() * 2 == sides;
    let tolerance = SEW_TOLERANCE.max(2.0 * worst_gap);
    let mut shape = if faces.len() == 1 { k.copy(&faces[0])? } else { k.sew_patches(faces, tolerance, closed)? };
    if closed && !k.is_closed_solid(&shape)? {
        k.release(shape);
        shape = k.sew_patches(faces, tolerance, false)?;
    }
    let solid = closed && k.is_closed_solid(&shape)?;
    // The edges left open: the outer boundary's, and any along a shared curve, which did not
    // sew.
    let deflection = mesh_deflection(k.diagonal(&shape)?);
    let open: Vec<KernelEdge> = k.edges(&shape, deflection)?.into_iter().filter(|e| e.faces < 2).collect();
    let mut not_sewn: Vec<u32> = Vec::new();
    for e in &open {
        for s in &fit.shared {
            let Some(f) = fit.curves.get(&s.curve) else { continue };
            let mut near = f64::INFINITY;
            for i in 0..=16 {
                let q = curve_point(&f.poles, f.spans, i as f64 / 16.0);
                let d = q - e.mid;
                near = near.min((d.x * d.x + d.y * d.y + d.z * d.z).sqrt());
            }
            if near < 10.0 * tolerance + 1e-6 && !not_sewn.contains(&s.curve) {
                not_sewn.push(s.curve);
            }
        }
    }
    if !not_sewn.is_empty() {
        k.release(shape);
        let named: Vec<String> = not_sewn
            .iter()
            .map(|&c| {
                let gap = shared.iter().find(|s| s.curve == c).map(|s| s.gap).unwrap_or(0.0);
                format!("curve {c} (gap {} µm)", fixed1(gap * 1000.0))
            })
            .collect();
        return Err(FitError::new(format!("The patches did not sew along {} — their faces stand apart there; give the strip across it more spans, or draw the curve again.", named.join(", "))));
    }
    // Each face of the body to its patch, by where its middle is. Sewing keeps each face on
    // its own surface, so the middle of a face's parameter box is where its patch's is.
    let mut middles: Vec<DVec3> = Vec::with_capacity(faces.len());
    for f in faces.iter() {
        middles.push(k.face_points(f)?.first().map(|p| p.at).unwrap_or(DVec3::ZERO));
    }
    let mut face_of: HashMap<u64, usize> = HashMap::new();
    for f in k.face_points(&shape)? {
        let mut best = 0;
        let mut best_d = f64::INFINITY;
        for (i, m) in middles.iter().enumerate() {
            let d = *m - f.at;
            let d = (d.x * d.x + d.y * d.y + d.z * d.z).sqrt();
            if d < best_d {
                best_d = d;
                best = i;
            }
        }
        face_of.insert(f.hash, best);
    }
    let (spokes, centres) = split_of(reading);
    let report = NetworkReport {
        patches: fit.patches.iter().map(report_patch).collect(),
        cells: reading.outcomes.clone(),
        unfilled: reading.cells.unfilled.clone(),
        worst_angle: shared.iter().filter(|s| !s.crease).fold(0.0, |m, s| m.max(s.angle)),
        curves: shared,
        corners: fit.corners.clone(),
        strips: fit.strips.clone(),
        lost: reading.traced.lost.clone(),
        spokes,
        centres,
        bodies: fit.bodies.clone(),
        closed: solid,
        worst_gap,
        ms: times(reading, began.elapsed().as_secs_f64() * 1e3),
    };
    Ok(NetworkBody { shape, report, face_of })
}

// ---------------------------------------------------------------------------------------
// The ghost.

/// A mesh as the viewport draws it: x, y, z per vertex, a unit normal per vertex, three
/// vertices a triangle.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct GhostMesh {
    pub positions: Vec<f32>,
    pub normals: Vec<f32>,
    pub indices: Vec<u32>,
}

/// A network's ghost: its patches meshed, its report, and the volume the mesh closes round
/// (0 where it does not close).
#[derive(Debug, Clone, PartialEq)]
pub struct NetworkGhost {
    pub mesh: GhostMesh,
    pub report: NetworkReport,
    pub volume: f64,
}

/// A network's ghost while it is being drawn: each patch meshed from its own grid, without
/// the kernel — no faces made, none sewn — and the report of the fit, the gaps the patches'
/// own. The grids are the ones the faces are made of, a shared curve's samples on both sides
/// of it, so the mesh closes where the network does and its volume is read off it. Making and
/// sewing the faces was most of a preview's time; Create makes them, and says there what they
/// come to. `None` where no patch was fitted.
pub fn network_ghost(reading: &NetworkReading<NetworkFitResult>) -> Option<NetworkGhost> {
    let fit = reading.fit.as_ref()?;
    let began = Instant::now();
    let mut sampler = SideSampler::new(fit);
    let mut positions: Vec<f64> = Vec::new();
    let mut normals: Vec<f32> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    for patch in &fit.patches {
        let grid = patch_grid(patch, GRID_PER_SPAN, &mut sampler, GHOST_SIDE);
        let base = (positions.len() / 3) as u32;
        for i in 0..grid.rows {
            for j in 0..grid.cols {
                let at = grid.points[i * grid.cols + j];
                positions.extend_from_slice(&[at.x, at.y, at.z]);
                let (_, du, dv) = patch_point(&patch.net, i as f64 / (grid.rows - 1) as f64, j as f64 / (grid.cols - 1) as f64);
                let n = DVec3::new(du.y * dv.z - du.z * dv.y, du.z * dv.x - du.x * dv.z, du.x * dv.y - du.y * dv.x);
                let l = (n.x * n.x + n.y * n.y + n.z * n.z).sqrt();
                let l = if l != 0.0 && !l.is_nan() { l } else { 1.0 };
                normals.extend_from_slice(&[(n.x / l) as f32, (n.y / l) as f32, (n.z / l) as f32]);
            }
        }
        let cols = grid.cols as u32;
        for i in 0..grid.rows as u32 - 1 {
            for j in 0..cols - 1 {
                let a = base + i * cols + j;
                let b = a + cols;
                indices.extend_from_slice(&[a, b, b + 1, a, b + 1, a + 1]);
            }
        }
    }
    // Closed: every side of every patch shared, as the body would be.
    let sides: usize = fit.patches.iter().map(|p| p.sides.len()).sum();
    let closed = fit.shared.len() * 2 == sides;
    let mut volume = 0.0;
    if closed {
        let p = &positions;
        for t in indices.chunks_exact(3) {
            let (a, b, c) = (t[0] as usize * 3, t[1] as usize * 3, t[2] as usize * 3);
            volume += (p[a] * (p[b + 1] * p[c + 2] - p[b + 2] * p[c + 1]) + p[a + 1] * (p[b + 2] * p[c] - p[b] * p[c + 2]) + p[a + 2] * (p[b] * p[c + 1] - p[b + 1] * p[c])) / 6.0;
        }
    }
    let mut report = empty_report(reading);
    report.closed = closed;
    report.ms.faces = began.elapsed().as_secs_f64() * 1e3;
    Some(NetworkGhost { mesh: GhostMesh { positions: positions.into_iter().map(|x| x as f32).collect(), normals, indices }, report, volume: volume.abs() })
}

/// A patch sampled PATCH_GRID × PATCH_GRID, row-major along u.
fn sampled(p: &FittedPatch) -> Vec<f32> {
    let mut out = Vec::with_capacity(PATCH_GRID * PATCH_GRID * 3);
    for i in 0..PATCH_GRID {
        for j in 0..PATCH_GRID {
            let q = patch_point(&p.net, i as f64 / (PATCH_GRID - 1) as f64, j as f64 / (PATCH_GRID - 1) as f64).0;
            out.extend_from_slice(&[q.x as f32, q.y as f32, q.z as f32]);
        }
    }
    out
}

/// JavaScript's `Math.round`: the nearest whole number, a half rounded up.
fn js_round(x: f64) -> f64 {
    let f = x.floor();
    if x - f >= 0.5 {
        f + 1.0
    } else {
        f
    }
}

fn where_(p: DVec3) -> String {
    format!("({}, {}, {})", fixed1(p.x), fixed1(p.y), fixed1(p.z))
}

#[cfg(test)]
mod tests {
    //! ScanRuler's `plugins/reverse/tests/reNetworkBody.test.ts` (PN9) and
    //! `reNetworkBench.test.ts` (PN14), as far as they run without Open CASCADE: the kernel
    //! here is [`GridKernel`], a stand-in that keeps each face as its grid's facets and sews
    //! the sides whose samples coincide — so what is checked is this module's own work: the
    //! grids, the shared samples, the sewing's flow, the report, the ghost and its volume.
    //! The STEP round trip and the thickening are the builder's and the kernel's, left to
    //! integration.

    use std::f64::consts::PI;
    use std::sync::OnceLock;

    use super::*;
    use crate::network_feature::NetworkFeature;
    use crate::network_fit::read_network_fitted;
    use crate::network_meshes::{curve, grid_network, grid_scan, network, sphere_scan, GridOptions};
    use crate::scan_path::ScanPaths;

    const R: f64 = 40.0;
    const NOISE: f64 = 0.02;

    fn on_sphere(u: f64, v: f64) -> DVec3 {
        let a = (u - 0.5) * 1.4;
        let b = (v - 0.5) * 1.4;
        let l = (1.0 + a * a + b * b).sqrt();
        DVec3::new(R / l, (R * a) / l, (R * b) / l)
    }

    fn sphere() -> &'static ScanPaths {
        static SPHERE: OnceLock<ScanPaths> = OnceLock::new();
        SPHERE.get_or_init(|| {
            let m = sphere_scan(R, NOISE, 6);
            ScanPaths::new(m.positions, m.indices)
        })
    }

    fn read(scan: &ScanPaths, f: &NetworkFeature) -> NetworkReading<NetworkFitResult> {
        match read_network_fitted(scan, f, None) {
            Ok(r) => r,
            Err(e) => panic!("{}", e.0),
        }
    }

    /// A cube's corners on the sphere, its twelve edges the curves, a seed in each face.
    fn cube() -> NetworkFeature {
        let s = R / 3f64.sqrt();
        let corner = |i: u32| DVec3::new(if i & 1 != 0 { s } else { -s }, if i & 2 != 0 { s } else { -s }, if i & 4 != 0 { s } else { -s });
        let nodes: Vec<(u32, DVec3)> = (0..8).map(|i| (i + 1, corner(i))).collect();
        let mut curves = Vec::new();
        for i in 0..8u32 {
            for bit in [1, 2, 4] {
                if i & bit == 0 {
                    curves.push(curve(i + 1, (i | bit) + 1));
                }
            }
        }
        let seeds = [DVec3::new(R, 0.0, 0.0), DVec3::new(-R, 0.0, 0.0), DVec3::new(0.0, R, 0.0), DVec3::new(0.0, -R, 0.0), DVec3::new(0.0, 0.0, R), DVec3::new(0.0, 0.0, -R)];
        network(&nodes, &curves, &seeds)
    }

    // ---- a kernel of grids ---------------------------------------------------------

    #[derive(Debug, Clone)]
    enum GridShape {
        Face(PatchGrid),
        Sewn { faces: Vec<PatchGrid>, solid: bool },
    }

    impl GridShape {
        fn grids(&self) -> Vec<&PatchGrid> {
            match self {
                GridShape::Face(g) => vec![g],
                GridShape::Sewn { faces, .. } => faces.iter().collect(),
            }
        }
    }

    /// A kernel whose face through a grid is the grid's facets, and whose sewing joins the
    /// sides of two faces whose samples coincide within the tolerance.
    #[derive(Default)]
    struct GridKernel {
        made: usize,
        released: usize,
    }

    /// A grid's four sides as its samples run, in the patch's order.
    fn grid_sides(g: &PatchGrid) -> [Vec<DVec3>; 4] {
        let p = &g.points;
        [
            (0..g.rows).map(|i| p[i * g.cols]).collect(),
            (0..g.cols).map(|j| p[(g.rows - 1) * g.cols + j]).collect(),
            (0..g.rows).rev().map(|i| p[i * g.cols + g.cols - 1]).collect(),
            (0..g.cols).rev().map(|j| p[j]).collect(),
        ]
    }

    fn coincide(a: &[DVec3], b: &[DVec3], tolerance: f64) -> bool {
        a.len() == b.len() && (a.iter().zip(b).all(|(x, y)| (*x - *y).length() <= tolerance) || a.iter().zip(b.iter().rev()).all(|(x, y)| (*x - *y).length() <= tolerance))
    }

    /// Each face's sides, and whether another face has a side on the same samples.
    fn sides_sewn(grids: &[&PatchGrid], tolerance: f64) -> Vec<(DVec3, bool)> {
        let sides: Vec<[Vec<DVec3>; 4]> = grids.iter().map(|g| grid_sides(g)).collect();
        let mut out = Vec::new();
        for (f, own) in sides.iter().enumerate() {
            for side in own {
                let sewn = sides.iter().enumerate().any(|(g, other)| g != f && other.iter().any(|o| coincide(side, o, tolerance)));
                out.push((side[side.len() / 2], sewn));
            }
        }
        out
    }

    /// The nearest point of a triangle to p (Ericson, Real-Time Collision Detection 5.1.5).
    fn closest_on_triangle(p: DVec3, a: DVec3, b: DVec3, c: DVec3) -> DVec3 {
        let (ab, ac, ap) = (b - a, c - a, p - a);
        let (d1, d2) = (ab.dot(ap), ac.dot(ap));
        if d1 <= 0.0 && d2 <= 0.0 {
            return a;
        }
        let bp = p - b;
        let (d3, d4) = (ab.dot(bp), ac.dot(bp));
        if d3 >= 0.0 && d4 <= d3 {
            return b;
        }
        let vc = d1 * d4 - d3 * d2;
        if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
            return a + ab * (d1 / (d1 - d3));
        }
        let cp = p - c;
        let (d5, d6) = (ab.dot(cp), ac.dot(cp));
        if d6 >= 0.0 && d5 <= d6 {
            return c;
        }
        let vb = d5 * d2 - d1 * d6;
        if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
            return a + ac * (d2 / (d2 - d6));
        }
        let va = d3 * d6 - d5 * d4;
        if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
            return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
        }
        let denom = 1.0 / (va + vb + vc);
        a + ab * (vb * denom) + ac * (vc * denom)
    }

    impl NetworkKernel for GridKernel {
        type Shape = GridShape;

        fn surface_through(&mut self, grid: &PatchGrid) -> Result<GridShape, FitError> {
            if grid.rows < 2 || grid.cols < 2 || grid.points.len() != grid.rows * grid.cols {
                return Err(FitError::new("the surface grid is malformed."));
            }
            self.made += 1;
            Ok(GridShape::Face(grid.clone()))
        }

        fn surface_area(&mut self, shape: &GridShape) -> Result<f64, FitError> {
            Ok(shape.grids().iter().map(|g| grid_area(g)).sum())
        }

        fn distance_to_point(&mut self, shape: &GridShape, p: DVec3) -> Result<f64, FitError> {
            let mut best = f64::INFINITY;
            for g in shape.grids() {
                for r in 0..g.rows - 1 {
                    for c in 0..g.cols - 1 {
                        let a = r * g.cols + c;
                        let (b, d) = (a + 1, a + g.cols);
                        let e = d + 1;
                        for (x, y, z) in [(a, b, e), (a, e, d)] {
                            best = best.min((closest_on_triangle(p, g.points[x], g.points[y], g.points[z]) - p).length());
                        }
                    }
                }
            }
            Ok(best)
        }

        fn copy(&mut self, shape: &GridShape) -> Result<GridShape, FitError> {
            Ok(shape.clone())
        }

        fn sew_patches(&mut self, faces: &[GridShape], tolerance: f64, closed: bool) -> Result<GridShape, FitError> {
            if faces.is_empty() {
                return Err(FitError::new("no faces to sew."));
            }
            let grids: Vec<PatchGrid> = faces.iter().flat_map(|f| f.grids().into_iter().cloned()).collect();
            let solid = closed && sides_sewn(&grids.iter().collect::<Vec<_>>(), tolerance).iter().all(|s| s.1);
            Ok(GridShape::Sewn { faces: grids, solid })
        }

        fn is_closed_solid(&mut self, shape: &GridShape) -> Result<bool, FitError> {
            Ok(matches!(shape, GridShape::Sewn { solid: true, .. }))
        }

        fn diagonal(&mut self, shape: &GridShape) -> Result<f64, FitError> {
            let (mut lo, mut hi) = (DVec3::INFINITY, DVec3::NEG_INFINITY);
            for g in shape.grids() {
                for &p in &g.points {
                    lo = lo.min(p);
                    hi = hi.max(p);
                }
            }
            Ok((hi - lo).length())
        }

        fn edges(&mut self, shape: &GridShape, _deflection: f64) -> Result<Vec<KernelEdge>, FitError> {
            Ok(sides_sewn(&shape.grids(), SEW_TOLERANCE).into_iter().map(|(mid, sewn)| KernelEdge { mid, faces: if sewn { 2 } else { 1 } }).collect())
        }

        fn face_points(&mut self, shape: &GridShape) -> Result<Vec<FacePoint>, FitError> {
            Ok(shape.grids().iter().enumerate().map(|(i, g)| FacePoint { hash: 100 + i as u64, at: g.points[(g.rows / 2) * g.cols + g.cols / 2] }).collect())
        }

        fn release(&mut self, _shape: GridShape) {
            self.released += 1;
        }
    }

    // ---- the tests --------------------------------------------------------------------

    #[test]
    fn makes_a_2x2_network_on_a_sphere_a_sheet() {
        let r = read(sphere(), &grid_network(2, 2, &on_sphere, 1, [0.0, 1.0, 0.0, 1.0]));
        let mut k = GridKernel::default();
        let body = match network_body(&mut k, &r) {
            Ok(b) => b,
            Err(e) => panic!("{}", e.0),
        };
        let report = &body.report;
        assert_eq!(report.patches.len(), 4);
        assert!(!report.closed);
        assert!(report.worst_angle < 0.5, "angle {}", report.worst_angle);
        // The kernel's faces here are the grids' facets, a chord's sag off the curve between
        // two samples — not the micron a kernel's surface through them keeps (ScanRuler's
        // test reads 0.005 mm there).
        assert!(report.worst_gap < 0.1, "gap {}", report.worst_gap);
        for p in &report.patches {
            assert!(p.rms < 1.4 * NOISE, "rms {}", p.rms);
        }
        // Four faces, each traced back to its own patch; every face made let go of but the
        // sewn shape's own.
        let mut patches: Vec<usize> = body.face_of.values().copied().collect();
        patches.sort_unstable();
        assert_eq!(patches, vec![0, 1, 2, 3]);
        assert_eq!(k.made, k.released);
        // Its report as JSON, as the feature's row carries it.
        let json = serde_json::to_value(report).map_err(|e| e.to_string()).expect("json");
        assert!(json.get("worstAngle").is_some() && json.get("worstGap").is_some());
        assert_eq!(json["patches"][0]["grid"].as_array().map(|a| a.len()), Some(PATCH_GRID * PATCH_GRID * 3));
    }

    #[test]
    fn sews_a_network_closed_round_a_sphere_into_one_solid_and_its_ghost_closes_round_the_same_volume() {
        let f = cube();
        let r = read(sphere(), &f);
        let mut k = GridKernel::default();
        let body = match network_body(&mut k, &r) {
            Ok(b) => b,
            Err(e) => panic!("{}", e.0),
        };
        let report = &body.report;
        assert_eq!(report.patches.len(), 6);
        assert!(report.closed);
        // Every corner is one where three patches meet, reported.
        assert_eq!(report.corners.len(), 8);
        assert_eq!(body.face_of.len(), 6);
        // The ghost: the patches meshed as fitted, without the kernel's faces — closed round
        // the sphere's volume, the same patches read again.
        let again = read(sphere(), &f);
        let ghost = network_ghost(&again).expect("a ghost");
        assert!(ghost.report.closed);
        let ball = (4.0 / 3.0) * PI * R.powi(3);
        assert!((ghost.volume / ball - 1.0).abs() < 0.005, "volume {} of {ball}", ghost.volume);
        assert_eq!(ghost.report.patches.iter().map(|p| p.rms).collect::<Vec<_>>(), report.patches.iter().map(|p| p.rms).collect::<Vec<_>>());
        assert!(ghost.mesh.indices.len() > 6 * 2 * 3);
        assert_eq!(ghost.mesh.positions.len(), ghost.mesh.normals.len());
        assert!(ghost.report.worst_gap < report.worst_gap + 1e-9);
    }

    #[test]
    fn the_grids_either_side_of_a_shared_curve_take_the_same_samples() {
        let r = read(sphere(), &cube());
        let fit = r.fit.as_ref().expect("a fit");
        let mut sampler = SideSampler::new(fit);
        let grids: Vec<PatchGrid> = fit.patches.iter().map(|p| patch_grid(p, GRID_PER_SPAN, &mut sampler, 0)).collect();
        let surfaces = patch_surfaces(fit);
        assert_eq!(fit.shared.len(), 12);
        for s in &fit.shared {
            let [a, b] = s.patches;
            let side = |i: usize| fit.patches[i].sides.iter().position(|x| x.single && x.edges[0].curve == s.curve).expect("the side");
            let (ka, kb) = (side(a), side(b));
            // The samples inside the side, the same to the bit, one the other's reverse; the
            // ends are the nodes, which the other sides' samples write.
            let sa = &grid_sides(&grids[a])[ka];
            let sb = &grid_sides(&grids[b])[kb];
            assert_eq!(sa.len(), sb.len());
            let n = sa.len();
            assert!((1..n - 1).all(|i| sa[i] == sb[n - 1 - i]), "curve {}", s.curve);
            assert!((sa[0] - sb[n - 1]).length() < 1e-9 && (sa[n - 1] - sb[0]).length() < 1e-9);
            // The control nets' boundary rows: the curve's poles, both ways.
            let row = |surface: &PatchSurface, k: usize| -> Vec<DVec3> {
                let (rows, cols) = (surface.rows, surface.cols);
                match k {
                    0 => (0..rows).map(|i| surface.poles[i * cols]).collect(),
                    1 => (0..cols).map(|j| surface.poles[(rows - 1) * cols + j]).collect(),
                    2 => (0..rows).rev().map(|i| surface.poles[i * cols + cols - 1]).collect(),
                    _ => (0..cols).rev().map(|j| surface.poles[j]).collect(),
                }
            };
            let (ra, rb) = (row(&surfaces[a], ka), row(&surfaces[b], kb));
            assert_eq!(ra, rb.iter().rev().copied().collect::<Vec<_>>());
        }
        assert_eq!(surfaces[0].knots_u.len(), surfaces[0].rows + DEGREE + 1);
    }

    #[test]
    fn refuses_a_reading_with_no_patches() {
        let scan = sphere();
        // No seeds: nothing filled, no fit.
        let mut f = cube();
        f.cells.clear();
        let r = read(scan, &f);
        assert!(r.fit.is_none());
        let mut k = GridKernel::default();
        assert!(network_body(&mut k, &r).is_err());
        assert!(network_ghost(&r).is_none());
        let empty = empty_report(&r);
        assert!(empty.patches.is_empty() && !empty.closed);
    }

    /// A gently waved panel, 300 × 200 mm.
    fn panel(u: f64, v: f64) -> DVec3 {
        let x = 300.0 * u - 150.0;
        let y = 200.0 * v - 100.0;
        DVec3::new(x, y, 12.0 * (x / 60.0).sin() * (y / 50.0).cos() + 0.0004 * x * x)
    }

    /// The patch network's timing bench (PN14): thirty patches at Auto — a 5 × 6 network over
    /// a 300 × 200 mm freeform sheet scanned at a 0.5 mm edge, half a million triangles with
    /// 0.02 mm of noise — traced, read into cells, mapped, fitted together and made a body,
    /// under ten seconds. What each stage took is printed. Like ScanRuler's benches it runs
    /// only when asked: `RE_NETWORK_BENCH=1 cargo test --release bench -- --nocapture`. The
    /// body is the grid kernel's here, not Open CASCADE's.
    #[test]
    fn fits_and_builds_thirty_patches_at_auto_in_under_ten_seconds() {
        if std::env::var_os("RE_NETWORK_BENCH").is_none() {
            return;
        }
        let m = grid_scan(600, 400, &panel, &GridOptions { noise: 0.02, ..Default::default() });
        let triangles = m.indices.len() / 3;
        let began = Instant::now();
        let scan = ScanPaths::new(m.positions, m.indices);
        let f = grid_network(5, 6, &panel, 1, [0.02, 0.98, 0.03, 0.97]);
        let r = read(&scan, &f);
        let mut k = GridKernel::default();
        let body = match network_body(&mut k, &r) {
            Ok(b) => b,
            Err(e) => panic!("{}", e.0),
        };
        let ms = began.elapsed().as_secs_f64() * 1e3;
        let report = &body.report;
        let rms = report.patches.iter().map(|p| p.rms);
        eprintln!(
            "30 patches, {triangles} triangles: {:.2} s in all; trace {:.2} s, cells {:.2} s, map {:.2} s, fit {:.2} s, faces {:.2} s; nets {} × {}; worst edge {:.3}°, worst gap {:.2} µm, RMS {:.4}–{:.4} mm",
            ms / 1000.0,
            report.ms.trace / 1000.0,
            report.ms.cells / 1000.0,
            report.ms.map / 1000.0,
            report.ms.fit / 1000.0,
            report.ms.faces / 1000.0,
            report.patches[0].net.u,
            report.patches[0].net.v,
            report.worst_angle,
            report.worst_gap * 1000.0,
            rms.clone().fold(f64::INFINITY, f64::min),
            rms.fold(0.0, f64::max)
        );
        assert_eq!(report.patches.len(), 30);
        assert!(report.worst_angle < 0.5, "angle {}", report.worst_angle);
        assert!(ms < 10_000.0, "{ms} ms");
    }
}
