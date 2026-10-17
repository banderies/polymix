//! The Inspector: defect analysis ([`inspect`]), one-click repair ([`auto_repair`]) and
//! repair of a single clicked defect ([`repair_single`]).

use crate::fill::{fill_hole, FillMode, FillParams, FillResult};
use crate::ops::{self, ComponentInfo};
use crate::{RepairError, Result};
use glam::DVec3;
use mm_core::mesh::Mesh;
use mm_core::topology::{self, EdgeLoop};
use std::time::Instant;

/// Parameters of [`inspect`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InspectParams {
    /// A component (other than the largest) is "small" if its area is below this fraction
    /// of the total surface area...
    pub small_component_area_fraction: f64,
    /// ...or if it has fewer triangles than this.
    pub small_component_tri_count: usize,
    /// A triangle is degenerate if `area ≤ degenerate_area_eps · (longest edge)²`
    /// (scale-independent; `1e-6` means its height is below ~2e-6 of its longest edge).
    pub degenerate_area_eps: f64,
    /// Boundary vertices closer than this are reported as cracks. Default (`None`):
    /// `1e-6 ×` bounding-box diagonal.
    pub crack_tolerance: Option<f64>,
}

impl Default for InspectParams {
    fn default() -> Self {
        Self { small_component_area_fraction: 0.01, small_component_tri_count: 10, degenerate_area_eps: 1e-6, crack_tolerance: None }
    }
}

/// One boundary loop.
#[derive(Clone, Debug, Default)]
pub struct HoleInfo {
    /// The loop (pass it to [`fill_hole`] or [`Defect::Hole`]).
    pub edge_loop: EdgeLoop,
    /// Mean of the loop vertices.
    pub centroid: DVec3,
    /// Sum of loop edge lengths.
    pub perimeter: f64,
    /// Number of loop edges.
    pub num_edges: usize,
    /// The loop passes through a bowtie vertex or visits a vertex twice.
    pub touches_bowtie: bool,
}

/// Result of [`inspect`].
#[derive(Clone, Debug, Default)]
pub struct InspectionReport {
    /// Boundary loops (found independently of triangle winding).
    pub holes: Vec<HoleInfo>,
    /// Non-manifold vertices (sorted).
    pub bowtie_vertices: Vec<u32>,
    /// Small floating components (the largest component is never listed).
    pub small_components: Vec<ComponentInfo>,
    /// Zero-area triangles (see [`InspectParams::degenerate_area_eps`]).
    pub degenerate_triangles: Vec<u32>,
    /// Interior edges whose two triangles have opposite winding.
    pub inconsistent_orientation_edges: usize,
    /// Boundary vertices with a coincident boundary vertex (duplicated STL seams / cracks).
    pub crack_vertices: usize,
    /// Edge-connected components.
    pub num_components: usize,
    /// Edges with a single triangle.
    pub boundary_edges: usize,
    /// Surface area.
    pub total_area: f64,
    /// Triangle count.
    pub triangle_count: usize,
    /// Vertex count.
    pub vertex_count: usize,
}

impl InspectionReport {
    /// No holes, bowties, small components, degenerate triangles or orientation errors.
    pub fn is_clean(&self) -> bool {
        self.holes.is_empty()
            && self.bowtie_vertices.is_empty()
            && self.small_components.is_empty()
            && self.degenerate_triangles.is_empty()
            && self.inconsistent_orientation_edges == 0
    }
    /// Closed (no boundary edges).
    pub fn is_closed(&self) -> bool {
        self.boundary_edges == 0
    }
    /// All defects as a flat list, e.g. for the GUI's clickable markers.
    pub fn defects(&self) -> Vec<Defect> {
        let mut out = Vec::new();
        out.extend(self.holes.iter().map(|h| Defect::Hole(h.edge_loop.clone())));
        out.extend(self.bowtie_vertices.iter().map(|&v| Defect::BowtieVertex(v)));
        out.extend(self.small_components.iter().map(|c| Defect::SmallComponent(c.triangles.clone())));
        out.extend(self.degenerate_triangles.iter().map(|&t| Defect::DegenerateTriangle(t)));
        out
    }
}

fn default_tolerance(mesh: &Mesh) -> f64 {
    1e-6 * mesh.bounds().diagonal()
}

/// Analyse `mesh` for the defects the Inspector shows.
pub fn inspect(mesh: &Mesh, params: &InspectParams) -> InspectionReport {
    let ((loops, bowtie_vertices), (infos, degenerate_triangles)) = rayon::join(
        || rayon::join(|| ops::boundary_loops(mesh), || ops::find_bowtie_vertices(mesh)),
        || rayon::join(|| ops::component_infos(mesh), || ops::find_degenerate_triangles(mesh, params.degenerate_area_eps)),
    );
    let inconsistent_orientation_edges = ops::count_inconsistent_edges(mesh);
    let tol = params.crack_tolerance.unwrap_or_else(|| default_tolerance(mesh));
    let crack_vertices = ops::count_coincident_boundary_vertices(mesh, tol);
    let holes = loops
        .into_iter()
        .map(|lp| {
            let mut vs = lp.vertices.clone();
            vs.sort_unstable();
            let repeated = vs.windows(2).any(|w| w[0] == w[1]);
            let touches_bowtie = repeated || lp.vertices.iter().any(|&v| bowtie_vertices.binary_search(&v).is_ok());
            HoleInfo {
                centroid: lp.centroid(mesh),
                perimeter: lp.perimeter(mesh),
                num_edges: lp.edges.len(),
                touches_bowtie,
                edge_loop: lp,
            }
        })
        .collect();
    let small_idx = ops::small_components(&infos, params.small_component_area_fraction, params.small_component_tri_count);
    let total_area = infos.iter().map(|c| c.area).sum();
    let num_components = infos.len();
    let small_components = small_idx.iter().map(|&i| infos[i].clone()).collect();
    InspectionReport {
        holes,
        bowtie_vertices,
        small_components,
        degenerate_triangles,
        inconsistent_orientation_edges,
        crack_vertices,
        num_components,
        boundary_edges: mesh.boundary_edge_ids().count(),
        total_area,
        triangle_count: mesh.triangle_count(),
        vertex_count: mesh.vertex_count(),
    }
}

/// Parameters of [`auto_repair`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RepairParams {
    /// How holes are filled.
    pub fill_mode: FillMode,
    /// Target edge length for Flat/Smooth fills (default: each loop's mean edge length).
    pub fill_target_edge_length: Option<f64>,
    /// Face group for all fills; `None` gives every fill a new group.
    pub fill_group: Option<i32>,
    /// Holes with more edges than this are left open (`None`: fill all).
    pub max_hole_edges: Option<usize>,
    /// Merge coincident boundary vertices (zip cracks / duplicated STL seams).
    pub weld_cracks: bool,
    /// Boundary vertices closer than this are welded. Default: `1e-6 ×` bbox diagonal.
    pub weld_tolerance: Option<f64>,
    /// Collapse / flip away zero-area triangles.
    pub remove_degenerate: bool,
    /// See [`InspectParams::degenerate_area_eps`].
    pub degenerate_area_eps: f64,
    /// Give each triangle fan of a bowtie vertex its own vertex.
    pub split_bowties: bool,
    /// Make winding consistent and closed components outward-facing.
    pub fix_orientation: bool,
    /// Delete small floating components.
    pub remove_small_components: bool,
    /// See [`InspectParams::small_component_area_fraction`].
    pub small_component_area_fraction: f64,
    /// See [`InspectParams::small_component_tri_count`].
    pub small_component_tri_count: usize,
    /// Fill all holes (subject to `max_hole_edges`).
    pub fill_holes: bool,
}

impl Default for RepairParams {
    fn default() -> Self {
        let ip = InspectParams::default();
        Self {
            fill_mode: FillMode::Smooth,
            fill_target_edge_length: None,
            fill_group: None,
            max_hole_edges: None,
            weld_cracks: true,
            weld_tolerance: None,
            remove_degenerate: true,
            degenerate_area_eps: ip.degenerate_area_eps,
            split_bowties: true,
            fix_orientation: true,
            remove_small_components: true,
            small_component_area_fraction: ip.small_component_area_fraction,
            small_component_tri_count: ip.small_component_tri_count,
            fill_holes: true,
        }
    }
}

/// What [`auto_repair`] did.
#[derive(Clone, Debug, Default)]
pub struct RepairReport {
    /// Boundary vertices merged into a coincident partner.
    pub welded_vertices: usize,
    /// Coincident pairs left unmerged because merging would be non-manifold.
    pub weld_rejected: usize,
    /// Degenerate triangles fixed (collapses, flips, removals).
    pub degenerate_fixed: usize,
    /// Degenerate triangles still present at the end.
    pub degenerate_remaining: usize,
    /// Bowtie vertices split (those not already resolved by filling their loop).
    pub bowties_split: usize,
    /// Triangles whose winding was reversed.
    pub triangles_flipped: usize,
    /// Small components deleted, and their triangle count.
    pub components_removed: usize,
    /// Triangles deleted with small components.
    pub component_triangles_removed: usize,
    /// Boundary loops filled.
    pub holes_filled: usize,
    /// Loops skipped because of `max_hole_edges`.
    pub holes_skipped: usize,
    /// Loops whose fill failed (see `messages`).
    pub holes_failed: usize,
    /// Triangles created by fills.
    pub fill_triangles_added: usize,
    /// Vertices created by fills.
    pub fill_vertices_added: usize,
    /// Fill errors (one line per failed hole).
    pub messages: Vec<String>,
    /// Boundary loops left at the end.
    pub remaining_holes: usize,
    /// Edge-connected components at the end.
    pub num_components: usize,
    /// No boundary edges at the end.
    pub is_closed: bool,
    /// Consistent winding at the end.
    pub is_consistently_oriented: bool,
    /// The progress callback stopped the repair early (the mesh is valid).
    pub cancelled: bool,
    /// Wall-clock time.
    pub seconds: f64,
    /// Wall-clock time per phase.
    pub timings: Vec<(&'static str, f64)>,
}

impl RepairReport {
    /// Closed, consistently oriented and a single component.
    pub fn is_watertight(&self) -> bool {
        self.is_closed && self.is_consistently_oriented && self.num_components == 1
    }
}

/// Automatic repair, in this order: weld cracks → remove degenerate triangles → remove
/// small components → orient consistently → fill holes (loops through bowtie vertices are
/// split into simple sub-loops) → split remaining bowtie vertices → remove degenerate
/// triangles left by fills → orient outward. The mesh stays valid if the progress callback
/// cancels.
pub fn auto_repair(mesh: &mut Mesh, params: &RepairParams, progress: Option<&dyn Fn(f32) -> bool>) -> RepairReport {
    let start = Instant::now();
    let mut rep = RepairReport::default();
    let report = |f: f32| progress.is_none_or(|p| p(f));
    let mut clock = Instant::now();
    let mut lap = |rep: &mut RepairReport, name: &'static str| {
        rep.timings.push((name, clock.elapsed().as_secs_f64()));
        clock = Instant::now();
    };
    let finish = |mesh: &Mesh, mut rep: RepairReport| -> RepairReport {
        let t = Instant::now();
        rep.remaining_holes = ops::boundary_loops(mesh).len();
        rep.num_components = topology::connected_components(mesh).len();
        rep.is_closed = rep.remaining_holes == 0 && mesh.is_closed();
        rep.is_consistently_oriented = ops::count_inconsistent_edges(mesh) == 0;
        rep.timings.push(("final checks", t.elapsed().as_secs_f64()));
        rep.seconds = start.elapsed().as_secs_f64();
        rep
    };
    macro_rules! step {
        ($f:expr) => {
            if !report($f) {
                rep.cancelled = true;
                return finish(mesh, rep);
            }
        };
    }
    step!(0.0);
    if params.weld_cracks {
        let tol = params.weld_tolerance.unwrap_or_else(|| default_tolerance(mesh));
        let w = ops::weld_boundary_vertices(mesh, tol);
        rep.welded_vertices = w.welded;
        rep.weld_rejected = w.rejected;
        lap(&mut rep, "weld");
    }
    step!(0.1);
    if params.remove_degenerate {
        let d = ops::remove_degenerate_triangles(mesh, params.degenerate_area_eps);
        rep.degenerate_fixed = d.collapsed + d.flipped + d.removed;
        rep.degenerate_remaining = d.remaining;
        lap(&mut rep, "degenerate");
    }
    step!(0.2);
    if params.remove_small_components {
        let (c, t) = ops::remove_small_components(mesh, params.small_component_area_fraction, params.small_component_tri_count);
        rep.components_removed = c;
        rep.component_triangles_removed = t;
        lap(&mut rep, "small components");
    }
    step!(0.25);
    // Consistent winding before filling, so every loop has one orientation (outward-ness is
    // settled at the end).
    if params.fix_orientation && ops::count_inconsistent_edges(mesh) > 0 {
        rep.triangles_flipped += topology::orient_consistently(mesh);
        lap(&mut rep, "orient");
    }
    step!(0.3);
    if params.fill_holes {
        // Loops through bowtie vertices are split into simple sub-loops by `fill_hole`,
        // which resolves boundary bowties without creating coincident vertices.
        let loops = ops::boundary_loops(mesh);
        let mut next_group = mesh.max_group_id() + 1;
        let n = loops.len().max(1);
        for (i, lp) in loops.iter().enumerate() {
            step!(0.3 + 0.6 * i as f32 / n as f32);
            if params.max_hole_edges.is_some_and(|m| lp.len() > m) {
                rep.holes_skipped += 1;
                continue;
            }
            let group = params.fill_group.unwrap_or_else(|| {
                next_group += 1;
                next_group - 1
            });
            let fp = FillParams {
                mode: params.fill_mode,
                target_edge_length: params.fill_target_edge_length,
                group: Some(group),
                ..Default::default()
            };
            match fill_hole(mesh, lp, &fp) {
                Ok(r) => {
                    rep.holes_filled += 1;
                    rep.fill_triangles_added += r.new_triangles.len();
                    rep.fill_vertices_added += r.new_vertices.len();
                }
                Err(e) => {
                    rep.holes_failed += 1;
                    rep.messages.push(format!("hole {i} ({} edges): {e}", lp.len()));
                }
            }
        }
        lap(&mut rep, "fill");
    }
    step!(0.9);
    // Bowties left now join closed fans (or open fans of unfilled holes): give each fan its
    // own vertex.
    if params.split_bowties {
        rep.bowties_split = ops::split_all_bowties(mesh).0;
        lap(&mut rep, "bowties");
    }
    if params.remove_degenerate {
        let d = ops::remove_degenerate_triangles(mesh, params.degenerate_area_eps);
        rep.degenerate_fixed += d.collapsed + d.flipped + d.removed;
        rep.degenerate_remaining = d.remaining;
        lap(&mut rep, "degenerate (after fill)");
    }
    step!(0.95);
    if params.fix_orientation {
        rep.triangles_flipped += ops::orient_outward(mesh);
        lap(&mut rep, "orient outward");
    }
    let rep = finish(mesh, rep);
    if let Some(p) = progress {
        p(1.0);
    }
    rep
}

/// A single defect, as listed by [`InspectionReport::defects`].
#[derive(Clone, Debug)]
pub enum Defect {
    /// A boundary loop.
    Hole(EdgeLoop),
    /// A non-manifold vertex joining several triangle fans.
    BowtieVertex(u32),
    /// The triangles of a floating component.
    SmallComponent(Vec<u32>),
    /// A zero-area triangle.
    DegenerateTriangle(u32),
}

/// What [`repair_single`] changed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SingleRepair {
    /// Triangles/vertices created by a hole fill.
    pub fill: FillResult,
    /// Vertices created by splitting a bowtie.
    pub new_vertices: Vec<u32>,
    /// Triangles deleted (small component, degenerate fix).
    pub removed_triangles: usize,
}

/// Fix one defect (e.g. the hole the user clicked). Holes are filled with `fill`.
/// Returns [`RepairError::StaleDefect`] if the defect no longer exists in the mesh.
pub fn repair_single(mesh: &mut Mesh, defect: &Defect, fill: &FillParams) -> Result<SingleRepair> {
    match defect {
        Defect::Hole(lp) => Ok(SingleRepair { fill: fill_hole(mesh, lp, fill)?, ..Default::default() }),
        Defect::BowtieVertex(v) => {
            if !mesh.is_vertex(*v) || !mesh.is_bowtie_vertex(*v) {
                return Err(RepairError::StaleDefect(format!("vertex {v} is not a bowtie")));
            }
            Ok(SingleRepair { new_vertices: ops::split_bowtie_vertex(mesh, *v), ..Default::default() })
        }
        Defect::SmallComponent(tris) => {
            let live: Vec<u32> = tris.iter().copied().filter(|&t| mesh.is_triangle(t)).collect();
            if live.is_empty() {
                return Err(RepairError::StaleDefect("component no longer exists".into()));
            }
            let n = live.len();
            mesh.remove_triangles(live);
            Ok(SingleRepair { removed_triangles: n, ..Default::default() })
        }
        Defect::DegenerateTriangle(t) => {
            if !mesh.is_triangle(*t) {
                return Err(RepairError::StaleDefect(format!("triangle {t} no longer exists")));
            }
            let before = mesh.triangle_count();
            let ok = ops::fix_degenerate_triangle(mesh, *t);
            if !ok {
                return Err(RepairError::Mesh(mm_core::MeshError::InvalidNeighbourhood));
            }
            Ok(SingleRepair { removed_triangles: before.saturating_sub(mesh.triangle_count()), ..Default::default() })
        }
    }
}
