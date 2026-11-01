// SPDX-License-Identifier: AGPL-3.0-only
//! The Patch network feature's build: curves on the scan, the cells they close filled with
//! patches fitted to the scan together, and the patches made one body — a sheet, or a solid
//! where the network closes. Ported from ScanRuler (`plugins/reverse/core/build.ts`:
//! `networkReading`, `networkOf`, `networkPreview`), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only, on the SolidCraft kernel in place of Open CASCADE.
//!
//! The reading — the curves traced, the cells, the maps, the joint fit — is
//! [`crate::network_fit::read_network_fitted`]; the body is [`crate::network_body::network_body`]
//! with the SolidCraft kernel behind its [`NetworkKernel`]. SolidCraft makes a face from a
//! patch's control net directly (a clamped cubic B-spline surface, as a regular T-spline sheet),
//! so a shared curve is the same edge on both faces beside it and the faces sew without the
//! grids Open CASCADE's surface through points needs.
//!
//! A build reads the scan, which the part does not hold: the caller hands it in for the
//! build's thread with [`NetworkScope::enter`], with the feature whose box is open. That one is
//! built as ScanRuler previews it — its patches meshed as fitted, no kernel faces made, the
//! mesh standing as its body — and every network built says what it came to through the
//! scope's sink: its report (the box's rows, the layer's corners and refusals), or why there
//! is no body.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use glam::DVec3;
use solidcraft_geom::bspline::NurbsSurface;
use solidcraft_geom::tspline::TSpline;
use solidcraft_kernel::history::FaceOrigin;
use solidcraft_kernel::{self as kernel, Body};

use super::{diagonal_of, dv3, kerr, sv3, BodyState, Builder, FaceMesh};
use crate::fit2d::FitError;
use crate::model::{self, BodyId, Feature, FeatureId, NetworkFeature, Part};
use crate::network::{CellOutcome, NetworkBodies, NetworkReading};
use crate::network_body::{empty_report, network_body, network_ghost, FacePoint, KernelEdge, NetworkKernel, NetworkReport, PatchGrid, PatchSurface};
use crate::network_feature as nf;
use crate::network_fit::{read_network_fitted, NetworkFitResult};
use crate::network_spline::{patch_point, PatchNet, DEGREE};
use crate::scan_path::ScanPaths;
use crate::topo::{match_edge, match_face, match_tolerance};

// ---- what a build of a network is handed, and what it hands back ----

/// What a network's build said of it: its report — the cells, the patches, the strips, the
/// curves' gaps and angles — or why there is none, and the body it made.
#[derive(Debug, Clone)]
pub struct NetworkOutcome {
    pub feature: FeatureId,
    /// Built as the open box's preview: the patches meshed, no kernel faces.
    pub preview: bool,
    /// What the build read; `None` where there was nothing to read (no curve yet, as
    /// ScanRuler's preview stands idle) or the reading itself failed.
    pub report: Option<NetworkReport>,
    /// Why no body was made (ScanRuler's `failure`): no cell filled, none could be, the fit
    /// or the kernel refused.
    pub failure: Option<String>,
    /// The body's volume where the network closes, mm³; 0 for a sheet.
    pub volume: f64,
    pub closed: bool,
}

/// What a build reads besides the part, for its networks: the scan, the open box's feature,
/// and where each network's outcome goes.
pub struct NetworkContext {
    /// The scan in the part's frame, prepared for paths along it — asked for the first time a
    /// network is built, so a part without one never makes it.
    pub scan: Box<dyn Fn() -> Option<Arc<ScanPaths>> + Send>,
    /// The feature whose box is open: built as its preview (see the module documentation).
    pub preview: Option<FeatureId>,
    /// Where each network built says what it came to.
    pub sink: Option<Box<dyn Fn(NetworkOutcome) + Send>>,
}

thread_local! {
    static CONTEXT: RefCell<Option<Rc<NetworkContext>>> = const { RefCell::new(None) };
}

/// A build's networks reading the scan and saying what they came to, for as long as the scope
/// stands, on this thread: [`NetworkScope::enter`] before the build, dropped after it.
pub struct NetworkScope {
    previous: Option<Rc<NetworkContext>>,
}

impl NetworkScope {
    pub fn enter(context: NetworkContext) -> NetworkScope {
        let previous = CONTEXT.with(|c| c.borrow_mut().replace(Rc::new(context)));
        NetworkScope { previous }
    }
}

impl Drop for NetworkScope {
    fn drop(&mut self) {
        let previous = self.previous.take();
        CONTEXT.with(|c| *c.borrow_mut() = previous);
    }
}

fn context() -> Option<Rc<NetworkContext>> {
    CONTEXT.with(|c| c.borrow().clone())
}

fn tell(ctx: Option<&NetworkContext>, outcome: NetworkOutcome) {
    if let Some(sink) = ctx.and_then(|c| c.sink.as_ref()) {
        sink(outcome);
    }
}

// ---- the build ----

pub(crate) fn build(b: &mut Builder, _part: &Part, f: &Feature, x: &NetworkFeature) -> Result<(), String> {
    let ctx = context();
    let ctx = ctx.as_deref();
    let preview = ctx.is_some_and(|c| c.preview == Some(f.id));
    let outcome = |report: Option<NetworkReport>, failure: Option<String>, volume: f64, closed: bool| NetworkOutcome { feature: f.id, preview, report, failure, volume, closed };
    // A network with no curve is nothing to read: its box's preview stands idle.
    if preview && x.curves.is_empty() {
        tell(ctx, outcome(None, None, 0.0, false));
        return Ok(());
    }
    let Some(paths) = ctx.and_then(|c| (c.scan)()) else {
        let why = "The scan is not here to draw a patch network on.".to_string();
        tell(ctx, outcome(None, Some(why.clone()), 0.0, false));
        return if preview { Ok(()) } else { Err(why) };
    };
    let (feature, ids) = reading_feature(x);
    let reading = {
        let rests = Rests { b, ids: &ids, faces: RefCell::new(HashMap::new()) };
        read_network_fitted(&paths, &feature, Some(&rests))
    };
    let reading = match reading {
        Ok(r) => r,
        Err(e) => {
            tell(ctx, outcome(None, Some(e.0.clone()), 0.0, false));
            return if preview { Ok(()) } else { Err(e.0) };
        }
    };
    // No cell filled, or none could be: why, with what the cells came to — the regions the
    // curves close are what a click fills and Fill all fills.
    if reading.fit.is_none() {
        let refused = reading.outcomes.iter().find_map(|o| match o {
            CellOutcome::Refused { refused, .. } => Some(refused.clone()),
            CellOutcome::Filled { .. } => None,
        });
        let why = if x.cells.is_empty() {
            "No cell is filled yet — click inside a region the curves close, or Fill all.".to_string()
        } else {
            format!("No cell could be filled{}.", refused.map(|r| format!(": {r}")).unwrap_or_default())
        };
        tell(ctx, outcome(Some(empty_report(&reading)), Some(why.clone()), 0.0, false));
        return if preview { Ok(()) } else { Err(why) };
    }
    if preview {
        return preview_body(b, ctx, &reading, outcome);
    }
    let made = network_body(&mut SolidNet, &reading);
    let made = match made {
        Ok(m) => m,
        Err(e) => {
            tell(ctx, outcome(Some(empty_report(&reading)), Some(e.0.clone()), 0.0, false));
            return Err(e.0);
        }
    };
    let report = made.report;
    let shape = made.shape.body;
    let id = b.fresh_id()?;
    let mut state = BodyState::new(shape.clone(), b.own_origins(&shape));
    state.surface = !report.closed;
    b.set_body(id, state);
    b.touched(vec![id]);
    let refused: Vec<String> = report.cells.iter().filter_map(|c| match c {
        CellOutcome::Refused { refused, .. } => Some(refused.clone()),
        CellOutcome::Filled { .. } => None,
    }).collect();
    if !refused.is_empty() {
        b.warn(format!("{} cell{} not filled: {}.", refused.len(), if refused.len() == 1 { "" } else { "s" }, refused.join("; ")));
    }
    let volume = if report.closed { super::volume(&shape).unwrap_or(0.0) } else { 0.0 };
    let closed = report.closed;
    tell(ctx, outcome(Some(report), None, volume, closed));
    Ok(())
}

/// The open box's network as ScanRuler previews it: its patches meshed as fitted, no faces
/// made — the mesh standing as its body, a sheet or, where it closes, a solid — and the fit's
/// report.
fn preview_body(b: &mut Builder, ctx: Option<&NetworkContext>, reading: &NetworkReading<NetworkFitResult>, outcome: impl Fn(Option<NetworkReport>, Option<String>, f64, bool) -> NetworkOutcome) -> Result<(), String> {
    let Some(ghost) = network_ghost(reading) else {
        tell(ctx, outcome(Some(empty_report(reading)), None, 0.0, false));
        return Ok(());
    };
    let positions: Vec<solidcraft_geom::Vec3> = ghost.mesh.positions.chunks_exact(3).map(|p| solidcraft_geom::Vec3::new(p[0] as f64, p[1] as f64, p[2] as f64)).collect();
    let triangles: Vec<[u32; 3]> = ghost.mesh.indices.chunks_exact(3).map(|t| [t[0], t[1], t[2]]).collect();
    let closed = ghost.report.closed;
    let volume = ghost.volume;
    match kernel::mesh_body(&positions, &triangles) {
        Ok(shape) => {
            let id = b.fresh_id()?;
            let mut state = BodyState::new(shape.clone(), b.own_origins(&shape));
            state.surface = !closed;
            b.set_body(id, state);
            b.touched(vec![id]);
            tell(ctx, outcome(Some(ghost.report), None, volume, closed));
        }
        Err(e) => tell(ctx, outcome(Some(ghost.report), Some(kerr(e)), 0.0, false)),
    }
    Ok(())
}

// ---- the feature as the reading takes it ----

/// The network as its reading takes it: the bodies it rests on numbered in the order it names
/// them (their ids, by number, alongside), the strips' span counts as numbers.
pub fn reading_feature(x: &NetworkFeature) -> (nf::NetworkFeature, Vec<BodyId>) {
    let mut ids: Vec<BodyId> = Vec::new();
    let mut number = |id: BodyId| -> u32 {
        match ids.iter().position(|b| *b == id) {
            Some(i) => i as u32,
            None => {
                ids.push(id);
                (ids.len() - 1) as u32
            }
        }
    };
    let nodes = x
        .nodes
        .iter()
        .map(|n| nf::NetworkNode {
            id: n.id,
            at: n.at,
            on: n.on.as_ref().map(|on| match on {
                model::NodeOn::Edge { body, edge, t } => nf::NodeOn::Edge { body: number(*body), edge: edge.clone(), t: *t },
                model::NodeOn::Vertex { body, vertex, end } => nf::NodeOn::Vertex { body: number(*body), vertex: vertex.clone(), end: *end },
            }),
            corner: n.corner,
        })
        .collect();
    let curves = x
        .curves
        .iter()
        .map(|c| nf::NetworkCurve {
            id: c.id,
            from: c.from,
            to: c.to,
            via: c.via.clone(),
            crease: c.crease,
            along: c.along.as_ref().map(|a| nf::CurveAlong { body: number(a.body), edge: a.edge.clone() }),
            side: c.side.as_ref().map(|s| nf::CurveSide {
                body: number(s.body),
                face: s.face,
                mode: match s.mode {
                    model::SideMode::Contact => nf::SideMode::Contact,
                    model::SideMode::Tangent => nf::SideMode::Tangent,
                },
            }),
        })
        .collect();
    let cells = x.cells.iter().map(|c| nf::NetworkCell { seed: c.seed, centre: c.centre, corners: c.corners.clone() }).collect();
    let spans = x.spans.iter().map(|(k, v)| (k.clone(), *v as f64)).collect();
    let fit = nf::NetworkFitParams { smoothness: x.fit.smoothness, robust: x.fit.robust, tolerance: x.fit.tolerance };
    (nf::NetworkFeature { nodes, curves, cells, spans, fit }, ids)
}

/// The bodies a network rests on, as they stand before it (PN11): an edge found by its
/// fingerprint, as its points; a face's normals off one mesh of its body.
struct Rests<'a> {
    b: &'a Builder,
    ids: &'a [BodyId],
    faces: RefCell<HashMap<BodyId, Option<Rc<FaceMesh>>>>,
}

impl NetworkBodies for Rests<'_> {
    fn edge(&self, body: u32, edge: &nf::EdgeRef) -> Option<Vec<f32>> {
        let id = *self.ids.get(body as usize)?;
        let state = self.b.body(id)?;
        let t = state.topology();
        let i = match_edge(edge, &t.edge_prints(), &match_tolerance(state.diagonal()))?;
        Some(t.edges.get(i)?.polyline.iter().flat_map(|p| [p.x as f32, p.y as f32, p.z as f32]).collect())
    }

    fn normals(&self, body: u32, face: &nf::FaceRef, points: &[DVec3]) -> Option<Vec<DVec3>> {
        let id = *self.ids.get(body as usize)?;
        let state = self.b.body(id)?;
        let t = state.topology();
        let i = match_face(face, &t.face_prints(), &match_tolerance(state.diagonal()))?;
        let mesh = self.faces.borrow_mut().entry(id).or_insert_with(|| FaceMesh::of(&state.shape).map(Rc::new)).clone()?;
        mesh.normals(i, points).into_iter().collect()
    }
}

// ---- the kernel ----

/// A face or a body of faces the network made: the kernel's body, the patch nets its faces
/// are (exact, for the distance from a point), and the middle of each face's parameter box,
/// in the body's face order.
#[derive(Clone)]
pub(crate) struct NetShape {
    pub(crate) body: Body,
    nets: Vec<PatchNet>,
    middles: Vec<DVec3>,
}

/// SolidCraft as the network's kernel (ScanRuler's `ReKernel` calls `networkBody` makes): a
/// patch's face from its control net, the faces sewn by the kernel's stitch.
pub(crate) struct SolidNet;

fn fit_err(e: impl std::fmt::Display) -> FitError {
    FitError::new(e.to_string())
}

/// A patch as a face of the kernel: its clamped cubic net as a regular T-spline sheet, one
/// face bounded by the net's four boundary curves.
pub(crate) fn net_face(s: &PatchSurface) -> Result<Body, String> {
    if s.rows * s.cols != s.poles.len() || s.rows <= s.degree || s.cols <= s.degree {
        return Err("A patch's net does not match its size.".into());
    }
    // The kernel's surfaces take their points u fastest; a net's rows run along u.
    let mut points = Vec::with_capacity(s.poles.len());
    for j in 0..s.cols {
        for i in 0..s.rows {
            points.push(sv3(s.poles[i * s.cols + j]));
        }
    }
    let weights = vec![1.0; points.len()];
    let surface = NurbsSurface::new([s.degree, s.degree], [s.knots_u.clone(), s.knots_v.clone()], &points, &weights).map_err(|e| e.to_string())?;
    let t = TSpline::from_nurbs(&surface).map_err(|e| e.to_string())?;
    kernel::tspline_sheet(&t).map_err(kerr)
}

/// How far a point is from a patch, by its net: the nearest of a sampling, then Gauss–Newton
/// steps held to the patch's square.
fn distance_to_net(net: &PatchNet, q: DVec3) -> f64 {
    const SAMPLES: usize = 16;
    let mut best = (f64::INFINITY, 0.0, 0.0);
    for i in 0..=SAMPLES {
        for j in 0..=SAMPLES {
            let (u, v) = (i as f64 / SAMPLES as f64, j as f64 / SAMPLES as f64);
            let d = (patch_point(net, u, v).0 - q).length_squared();
            if d < best.0 {
                best = (d, u, v);
            }
        }
    }
    let (_, mut u, mut v) = best;
    for _ in 0..24 {
        let (p, du, dv) = patch_point(net, u, v);
        let r = p - q;
        let (a, b, c) = (du.dot(du), du.dot(dv), dv.dot(dv));
        let det = a * c - b * b;
        if !(det.abs() > 1e-30) {
            break;
        }
        let (gu, gv) = (du.dot(r), dv.dot(r));
        let su = (c * gu - b * gv) / det;
        let sv = (a * gv - b * gu) / det;
        let (nu, nv) = ((u - su).clamp(0.0, 1.0), (v - sv).clamp(0.0, 1.0));
        let moved = (nu - u).abs() + (nv - v).abs();
        u = nu;
        v = nv;
        if moved < 1e-14 {
            break;
        }
    }
    (patch_point(net, u, v).0 - q).length()
}

impl NetworkKernel for SolidNet {
    type Shape = NetShape;

    fn surface_from_net(&mut self, net: &PatchSurface) -> Option<Result<NetShape, FitError>> {
        let poles: Vec<f64> = net.poles.iter().flat_map(|p| [p.x, p.y, p.z]).collect();
        let patch = PatchNet { nu: net.rows.saturating_sub(DEGREE), nv: net.cols.saturating_sub(DEGREE), poles };
        let middle = patch_point(&patch, 0.5, 0.5).0;
        Some(net_face(net).map(|body| NetShape { body, nets: vec![patch], middles: vec![middle] }).map_err(FitError::new))
    }

    fn surface_through(&mut self, _grid: &PatchGrid) -> Result<NetShape, FitError> {
        Err(FitError::new("The kernel makes a patch's face from its control net, not through a grid of points."))
    }

    fn surface_area(&mut self, shape: &NetShape) -> Result<f64, FitError> {
        kernel::measure(&shape.body).map(|m| m.area).map_err(fit_err)
    }

    fn distance_to_point(&mut self, shape: &NetShape, point: DVec3) -> Result<f64, FitError> {
        if !shape.nets.is_empty() {
            return Ok(shape.nets.iter().map(|n| distance_to_net(n, point)).fold(f64::INFINITY, f64::min));
        }
        // A shape of no net: by its display mesh's facets.
        let mesh = shape.body.display_mesh(super::mesh_deflection(diagonal_of(&shape.body))).map_err(fit_err)?;
        let mut best = f64::INFINITY;
        for t in &mesh.triangles {
            let [a, b, c] = [t[0], t[1], t[2]].map(|i| mesh.positions.get(i as usize).copied().map(dv3).unwrap_or_default());
            best = best.min((crate::scan_path::closest_on_triangle(point, a, b, c) - point).length());
        }
        Ok(best)
    }

    fn copy(&mut self, shape: &NetShape) -> Result<NetShape, FitError> {
        Ok(shape.clone())
    }

    fn sew_patches(&mut self, faces: &[NetShape], tolerance: f64, _closed: bool) -> Result<NetShape, FitError> {
        let bodies: Vec<&Body> = faces.iter().map(|f| &f.body).collect();
        let body = kernel::stitch(&bodies, tolerance).map_err(fit_err)?;
        // Each face of the sewn body is the face it was sewn from: its middle is that one's.
        let count = body.face_count();
        let mut middles = vec![DVec3::ZERO; count];
        match body.history().filter(|h| h.operands == faces.len() && h.faces.len() == count) {
            Some(h) => {
                for (i, prov) in h.faces.iter().enumerate() {
                    let from = prov.origins.iter().find_map(|o| match o {
                        FaceOrigin::Face { operand, .. } => Some(*operand),
                        _ => None,
                    });
                    if let Some(m) = from.and_then(|k| faces.get(k)).and_then(|f| f.middles.first()) {
                        middles[i] = *m;
                    }
                }
            }
            None => {
                // No history: each face by where its centroid lies, nearest a patch's middle.
                for f in shape_faces(&body)? {
                    if let Some(slot) = middles.get_mut(f.0) {
                        *slot = f.1;
                    }
                }
            }
        }
        Ok(NetShape { body, nets: faces.iter().flat_map(|f| f.nets.iter().cloned()).collect(), middles })
    }

    fn is_closed_solid(&mut self, shape: &NetShape) -> Result<bool, FitError> {
        if shape.body.is_surface() || shape.body.face_count() == 0 {
            return Ok(false);
        }
        Ok(super::volume(&shape.body).is_some_and(|v| v > 0.0))
    }

    fn diagonal(&mut self, shape: &NetShape) -> Result<f64, FitError> {
        Ok(diagonal_of(&shape.body))
    }

    fn edges(&mut self, shape: &NetShape, deflection: f64) -> Result<Vec<KernelEdge>, FitError> {
        let mesh = shape.body.display_mesh(deflection).map_err(fit_err)?;
        Ok(mesh
            .edges
            .iter()
            .enumerate()
            .map(|(i, pts)| {
                let polyline: Vec<DVec3> = pts.iter().map(|p| dv3(*p)).collect();
                let mut faces: Vec<u32> = mesh.edge_faces.get(i).cloned().unwrap_or_default();
                faces.dedup();
                let seam = mesh.seams.get(i).copied().unwrap_or(false);
                KernelEdge { mid: super::polyline_mid(&polyline), faces: if seam { 2 } else { faces.len() } }
            })
            .collect())
    }

    fn face_points(&mut self, shape: &NetShape) -> Result<Vec<FacePoint>, FitError> {
        Ok(shape.middles.iter().enumerate().map(|(i, at)| FacePoint { hash: i as u64, at: *at }).collect())
    }
}

/// Each face of a body with its centroid, by its index.
fn shape_faces(body: &Body) -> Result<Vec<(usize, DVec3)>, FitError> {
    let faces = body.faces(super::mesh_deflection(diagonal_of(body))).map_err(fit_err)?;
    Ok(faces.into_iter().map(|f| (f.index, dv3(f.centroid))).collect())
}

#[cfg(test)]
mod tests {
    //! ScanRuler's `plugins/reverse/tests/reNetworkBody.test.ts` (PN9) through the builder and
    //! the SolidCraft kernel: a 2 × 2 network on a noisy sphere is a sheet of four faces,
    //! tangent across its curves within half a degree, its shared edges within a micron, and it
    //! writes to STEP; a network closed round the sphere sews into one solid of the sphere's
    //! volume, and its preview — no faces made — closes round the same volume and reads the
    //! same patches.

    use std::f64::consts::PI;
    use std::sync::{Mutex, OnceLock};

    use super::*;
    use crate::model::{FeatureKind, NetworkCell, NetworkCurve, NetworkFitParams, NetworkNode};
    use crate::network_meshes::{grid_network, sphere_scan};

    const R: f64 = 40.0;
    const NOISE: f64 = 0.02;

    fn on_sphere(u: f64, v: f64) -> DVec3 {
        let a = (u - 0.5) * 1.4;
        let b = (v - 0.5) * 1.4;
        let l = (1.0 + a * a + b * b).sqrt();
        DVec3::new(R / l, (R * a) / l, (R * b) / l)
    }

    fn sphere() -> Arc<ScanPaths> {
        static SPHERE: OnceLock<Arc<ScanPaths>> = OnceLock::new();
        SPHERE
            .get_or_init(|| {
                let m = sphere_scan(R, NOISE, 6);
                Arc::new(ScanPaths::new(m.positions, m.indices))
            })
            .clone()
    }

    /// The reading's feature in the part's form.
    fn model_feature(f: &nf::NetworkFeature) -> NetworkFeature {
        NetworkFeature {
            nodes: f.nodes.iter().map(|n| NetworkNode { id: n.id, at: n.at, on: None, corner: n.corner }).collect(),
            curves: f.curves.iter().map(|c| NetworkCurve { id: c.id, from: c.from, to: c.to, via: c.via.clone(), crease: c.crease, along: None, side: None }).collect(),
            cells: f.cells.iter().map(|c| NetworkCell { seed: c.seed, centre: c.centre, corners: c.corners.clone() }).collect(),
            spans: f.spans.iter().map(|(k, v)| (k.clone(), *v as u32)).collect(),
            fit: NetworkFitParams { smoothness: f.fit.smoothness, robust: f.fit.robust, tolerance: f.fit.tolerance },
        }
    }

    /// The part built over the sphere, `preview` its open box; what each network said.
    fn built(part: &Part, preview: Option<FeatureId>) -> (super::super::Built, Vec<NetworkOutcome>) {
        let said: Arc<Mutex<Vec<NetworkOutcome>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = said.clone();
        let scan = sphere();
        let scope = NetworkScope::enter(NetworkContext {
            scan: Box::new(move || Some(scan.clone())),
            preview,
            sink: Some(Box::new(move |o| {
                if let Ok(mut s) = sink.lock() {
                    s.push(o);
                }
            })),
        });
        let out = super::super::build(part);
        drop(scope);
        let said = said.lock().map(|s| s.clone()).unwrap_or_default();
        (out, said)
    }

    fn cube() -> NetworkFeature {
        let s = R / 3f64.sqrt();
        let corner = |i: u32| DVec3::new(if i & 1 != 0 { s } else { -s }, if i & 2 != 0 { s } else { -s }, if i & 4 != 0 { s } else { -s });
        let nodes: Vec<NetworkNode> = (0..8).map(|i| NetworkNode { id: i + 1, at: corner(i), on: None, corner: None }).collect();
        let mut curves = Vec::new();
        let mut id = 100;
        for i in 0..8u32 {
            for bit in [1, 2, 4] {
                if i & bit == 0 {
                    id += 1;
                    curves.push(NetworkCurve { id, from: i + 1, to: (i | bit) + 1, via: Vec::new(), crease: None, along: None, side: None });
                }
            }
        }
        let seeds = [DVec3::new(R, 0.0, 0.0), DVec3::new(-R, 0.0, 0.0), DVec3::new(0.0, R, 0.0), DVec3::new(0.0, -R, 0.0), DVec3::new(0.0, 0.0, R), DVec3::new(0.0, 0.0, -R)];
        NetworkFeature { nodes, curves, cells: seeds.iter().map(|s| NetworkCell { seed: *s, centre: None, corners: None }).collect(), spans: Default::default(), fit: NetworkFitParams { smoothness: 0.001, robust: None, tolerance: None } }
    }

    #[test]
    fn makes_a_2x2_network_on_a_sphere_a_sheet_that_writes_to_step() {
        let mut part = Part::default();
        let id = part.add(FeatureKind::Network(model_feature(&grid_network(2, 2, &on_sphere, 1, [0.0, 1.0, 0.0, 1.0]))));
        let (out, said) = built(&part, None);
        assert!(out.errors.get(&id).is_none(), "{:?}", out.errors.get(&id));
        let o = said.iter().find(|o| o.feature == id).expect("an outcome");
        let report = o.report.as_ref().expect("a report");
        assert_eq!(report.patches.len(), 4);
        assert!(!report.closed);
        assert!(report.worst_angle < 0.5, "angle {}", report.worst_angle);
        // The faces are the patches' nets: their edges along a shared curve are the curve.
        assert!(report.worst_gap < 0.005, "gap {}", report.worst_gap);
        for p in &report.patches {
            assert!(p.rms < 1.4 * NOISE, "rms {}", p.rms);
        }
        let body = out.bodies.iter().find(|b| b.id.feature == id).expect("a body");
        assert!(body.surface);
        assert_eq!(body.body.face_count(), 4);
        let text = super::super::export_step(&[("Sheet", &body.body)]).expect("STEP");
        assert!(text.contains("B_SPLINE_SURFACE"), "the patches go to STEP as B-spline surfaces");
    }

    #[test]
    fn sews_a_network_closed_round_a_sphere_into_one_solid_and_previews_the_same_volume() {
        let mut part = Part::default();
        let id = part.add(FeatureKind::Network(cube()));
        let (out, said) = built(&part, None);
        assert!(out.errors.get(&id).is_none(), "{:?}", out.errors.get(&id));
        let o = said.iter().find(|o| o.feature == id).expect("an outcome");
        let report = o.report.as_ref().expect("a report");
        assert_eq!(report.patches.len(), 6);
        assert!(report.closed);
        assert_eq!(report.corners.len(), 8);
        let body = out.bodies.iter().find(|b| b.id.feature == id).expect("a body");
        assert!(!body.surface);
        assert_eq!(body.body.face_count(), 6);
        let volume = super::super::volume(&body.body).expect("a volume");
        let ball = (4.0 / 3.0) * PI * R.powi(3);
        assert!((volume / ball - 1.0).abs() < 0.005, "volume {volume} of {ball}");
        // The preview: no faces made, closed round the same volume, the same patches read.
        let (ghost, said) = built(&part, Some(id));
        assert!(ghost.errors.get(&id).is_none());
        let g = said.iter().find(|o| o.feature == id).expect("an outcome");
        assert!(g.preview && g.closed);
        assert!((g.volume / volume - 1.0).abs() < 0.0005, "ghost {} body {volume}", g.volume);
        let gr = g.report.as_ref().expect("a report");
        assert_eq!(gr.patches.iter().map(|p| p.rms).collect::<Vec<_>>(), report.patches.iter().map(|p| p.rms).collect::<Vec<_>>());
        assert!(gr.worst_gap < report.worst_gap + 1e-9);
        let gb = ghost.bodies.iter().find(|b| b.id.feature == id).expect("the ghost stands as a body");
        assert!(gb.body.is_mesh());
    }

    #[test]
    fn says_why_where_no_cell_is_filled_and_keeps_the_regions_to_fill() {
        let mut f = model_feature(&grid_network(2, 2, &on_sphere, 1, [0.0, 1.0, 0.0, 1.0]));
        f.cells.clear();
        let mut part = Part::default();
        let id = part.add(FeatureKind::Network(f));
        let (out, said) = built(&part, None);
        assert_eq!(out.errors.get(&id).map(String::as_str), Some("No cell is filled yet — click inside a region the curves close, or Fill all."));
        let o = said.iter().find(|o| o.feature == id).expect("an outcome");
        assert!(o.report.as_ref().is_some_and(|r| !r.unfilled.is_empty()), "the regions the curves close, to fill");
        // Previewed, it is no error: the box says it.
        let (out, said) = built(&part, Some(id));
        assert!(out.errors.get(&id).is_none());
        assert!(said.iter().any(|o| o.feature == id && o.failure.is_some()));
    }

    #[test]
    fn fails_by_name_without_a_scan() {
        let mut part = Part::default();
        let id = part.add(FeatureKind::Network(cube()));
        let out = super::super::build(&part);
        assert_eq!(out.errors.get(&id).map(String::as_str), Some("The scan is not here to draw a patch network on."));
    }

    /// A 40 × 20 × 10 block, built: its body's id and the builder.
    fn block() -> (BodyId, Builder) {
        let mut part = Part::default();
        let mut s = crate::sketch::Sketch::default();
        s.add_rectangle(glam::DVec2::new(0.0, 0.0), glam::DVec2::new(40.0, 20.0));
        let regions = super::super::default_regions(&crate::regions::find_regions(&s));
        let sketch = part.add(FeatureKind::Sketch(model::SketchFeature::on_world(crate::frame::WorldAxis::Z, 0.0, s)));
        let extrude = part.add(FeatureKind::Extrude(model::ExtrudeFeature::new(sketch, regions, model::ExtrudeExtent::Distance { from: 0.0, to: 10.0 }, model::BodyOp::New)));
        (BodyId::new(extrude), super::super::play(&part, 2))
    }

    #[test]
    fn reads_a_bodys_edge_as_its_points_and_its_faces_normals_for_the_curves_resting_on_it() {
        let (id, b) = block();
        let t = b.topology_of(id).expect("the block");
        let edge = t.edges.iter().find(|e| (e.fingerprint.mid - DVec3::new(20.0, 0.0, 10.0)).length() < 1e-6).expect("its top front edge");
        let top = t.faces.iter().find(|f| f.fingerprint.dir.z > 0.99 && (f.fingerprint.centroid.z - 10.0).abs() < 1e-6).expect("its top").fingerprint;
        let ids = vec![id];
        let rests = Rests { b: &b, ids: &ids, faces: RefCell::new(HashMap::new()) };
        let points = rests.edge(0, &edge.fingerprint).expect("the edge's points");
        let ends: Vec<f32> = [&points[..3], &points[points.len() - 3..]].concat();
        assert!(ends.chunks(3).all(|p| p[1].abs() < 1e-4 && (p[2] - 10.0).abs() < 1e-4 && (p[0].abs() < 1e-4 || (p[0] - 40.0).abs() < 1e-4)), "{ends:?}");
        let normals = rests.normals(0, &top, &[DVec3::new(20.0, 5.0, 10.0), DVec3::new(2.0, 1.0, 10.0)]).expect("the top's normals");
        assert!(normals.iter().all(|n| (*n - DVec3::Z).length() < 1e-6), "{normals:?}");
        // A body the network does not name, or one not built, reads nothing.
        assert!(rests.edge(1, &edge.fingerprint).is_none());
        let gone = vec![BodyId::new(99)];
        assert!(Rests { b: &b, ids: &gone, faces: RefCell::new(HashMap::new()) }.edge(0, &edge.fingerprint).is_none());
    }

    #[test]
    fn numbers_the_bodies_a_network_rests_on_in_the_order_it_names_them() {
        let (id, b) = block();
        let t = b.topology_of(id).expect("the block");
        let edge = t.edges[0].fingerprint.clone();
        let face = t.faces[0].fingerprint;
        let other = model::BodyId { feature: 7, island: 1 };
        let mut f = cube();
        f.nodes[0].on = Some(model::NodeOn::Vertex { body: other, vertex: edge.clone(), end: 1 });
        f.curves[0].along = Some(model::CurveAlong { body: id, edge: edge.clone() });
        f.curves[0].side = Some(model::CurveSide { body: id, face, mode: model::SideMode::Tangent });
        f.spans.insert("101".into(), 6);
        let (read, ids) = reading_feature(&f);
        assert_eq!(ids, vec![other, id]);
        assert!(matches!(read.nodes[0].on, Some(nf::NodeOn::Vertex { body: 0, end: 1, .. })));
        assert_eq!(read.curves[0].along.as_ref().map(|a| a.body), Some(1));
        assert_eq!(read.curves[0].side.as_ref().map(|s| (s.body, s.mode)), Some((1, nf::SideMode::Tangent)));
        assert_eq!(read.spans.get("101"), Some(&6.0));
        assert_eq!(read.cells.len(), 6);
    }

    #[test]
    fn a_net_face_follows_its_patch_and_reads_its_distance_exactly() {
        let net = PatchNet { nu: 1, nv: 1, poles: (0..16).flat_map(|k| { let (i, j) = (k / 4, k % 4); [i as f64 * 10.0, j as f64 * 10.0, if (i == 1 || i == 2) && (j == 1 || j == 2) { 5.0 } else { 0.0 }] }).collect() };
        let s = PatchSurface { degree: 3, rows: 4, cols: 4, poles: net.poles.chunks_exact(3).map(|c| DVec3::new(c[0], c[1], c[2])).collect(), knots_u: crate::network_spline::knots_of(1), knots_v: crate::network_spline::knots_of(1), seed: DVec3::ZERO };
        let face = net_face(&s).expect("a face");
        assert_eq!(face.face_count(), 1);
        assert!(face.is_surface());
        let p = patch_point(&net, 0.3, 0.7).0;
        assert!(distance_to_net(&net, p) < 1e-9);
        assert!((distance_to_net(&net, p + DVec3::Z * 0.5) - 0.5).abs() < 0.05);
        // The kernel's face lies where the net does.
        let m = face.display_mesh(0.01).expect("a mesh");
        for q in m.positions.iter().take(50) {
            assert!(distance_to_net(&net, dv3(*q)) < 0.02);
        }
    }
}
