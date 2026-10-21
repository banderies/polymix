//! Make Solid and the SDF-based offset / hollow / boolean operations.

use crate::cleanup;
use crate::extract::{build_mesh, marching_tetrahedra_with};
use crate::field::{BoolField, HollowField, ImplicitField, MeshSdf, DEFAULT_BAND_CELLS, FIELD_BETA};
pub use crate::field::BoolOp;
use crate::grid::{build_field_with, GridSpec, SparseField};
use crate::progress::Prog;
use crate::winding::WindingNumber;
use crate::{Progress, Result, SolidError};
use glam::DVec3;
use mm_core::{geom::Ray, measure, Aabb, Bvh, Mesh};
use rayon::prelude::*;
use std::borrow::Cow;
use std::sync::Arc;

/// Output style of [`make_solid`] (Meshmixer's "Solid Type").
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SolidMode {
    /// Marching-tetrahedra surface of the signed distance field, cleaned of slivers.
    #[default]
    Fast,
    /// As `Fast`, then vertices are projected onto the nearest input-surface point when it
    /// is within one cell, lies in the interior of the input (not on a hole rim) and the
    /// input normal agrees with the output normal. Topology is untouched and moves that
    /// would flip a triangle are rejected, so manifoldness is preserved.
    Accurate,
    /// Boundary faces of the voxels whose centres are inside (made well-composed by
    /// filling critical voxel configurations, so the result is still a 2-manifold).
    Blocky,
    /// As `Accurate`, then vertices near the input's creases and corners move onto them and
    /// edges are flipped to follow the creases, so CAD-like parts keep crisp edges.
    SharpEdges,
}

/// Parameters for [`make_solid`], mirroring Meshmixer's Make Solid panel.
#[derive(Clone, Debug, PartialEq)]
pub struct MakeSolidParams {
    /// Output style.
    pub mode: SolidMode,
    /// Grid spacing in model units ("Solid Accuracy"). Values `<= 0` select 128 cells
    /// along the largest bounding-box dimension. See [`MakeSolidParams::with_resolution`].
    pub cell_size: f64,
    /// Grow (> 0) or shrink (< 0) the solid by this distance ("Offset Distance").
    pub offset: f64,
    /// Minimum wall thickness ("Min Thickness"). Open sheets become slabs of this
    /// thickness, and closed walls / tubes thinner than this (measured by an inward ray
    /// from each triangle) are thickened symmetrically to roughly this thickness. Regions
    /// that are already thick enough are unchanged.
    pub min_thickness: Option<f64>,
    /// A point is inside when its generalized winding number satisfies `|w| ≥` this
    /// (0.5 by default; lower values close larger gaps more eagerly, higher values drop
    /// weakly enclosed regions). `|w|` is used so inverted shells also count as solid.
    pub winding_threshold: f64,
    /// `true` (default): holes are capped by the smooth `|w| = threshold` surface, so the
    /// result is a closed solid. `false` (Meshmixer's "preserve boundaries" behaviour, a.k.a.
    /// *preserve open boundaries*): the solid is computed the same way, then the cap
    /// patches spanning the input's holes are cut away so the input's open boundaries
    /// (e.g. vessel inlets/outlets) stay open. The result is then *not* closed, but is
    /// still edge-manifold, bowtie-free and consistently oriented.
    pub close_holes: bool,
    /// Remove disconnected shells whose |enclosed volume| is below this (model units³).
    /// `None` = automatic (8 cell volumes), `Some(0.0)` keeps everything. The largest shell
    /// is always kept.
    pub min_shell_volume: Option<f64>,
    /// Collapse output edges shorter than this ("Mesh Density"). `None` = 0.3 cells, which
    /// removes marching-tetrahedra slivers without visibly changing the shape.
    pub min_edge_length: Option<f64>,
    /// Run quality-improving edge flips on nearly flat regions (default `true`).
    pub improve_quality: bool,
    /// Give each output triangle the face group of the nearest input triangle ("Transfer
    /// Groups"). Otherwise the output is a single group.
    pub transfer_groups: bool,
    /// Crease angle (degrees) that [`SolidMode::SharpEdges`] treats as a sharp edge.
    pub sharp_angle_deg: f64,
    /// Give the output the input's vertex colours (sampled at the nearest input point),
    /// when the input has any ("Color Transfer Mode: Vertex-to-Vertex").
    pub transfer_colors: bool,
}

impl Default for MakeSolidParams {
    fn default() -> Self {
        Self {
            mode: SolidMode::Fast,
            cell_size: 0.0,
            offset: 0.0,
            min_thickness: None,
            winding_threshold: 0.5,
            close_holes: true,
            min_shell_volume: None,
            min_edge_length: None,
            improve_quality: true,
            transfer_groups: false,
            sharp_angle_deg: 30.0,
            transfer_colors: true,
        }
    }
}

impl MakeSolidParams {
    /// Default parameters with the grid sized to `cells` cells along the largest
    /// bounding-box dimension of `mesh` (Meshmixer's accuracy range is roughly 64–1024).
    pub fn with_resolution(mesh: &Mesh, cells: u32) -> Self {
        Self { cell_size: cell_size_for_resolution(&mesh.bounds(), cells), ..Default::default() }
    }

    /// Map Meshmixer-style sliders in `[0, 1]` to parameters: `solid_accuracy` sets the grid
    /// to 64·16^accuracy cells (64..1024) along the largest dimension; `mesh_density` sets
    /// the collapse length from 1.5 cells (0, coarse) down to 0.3 cells (1, finest).
    pub fn from_sliders(mesh: &Mesh, solid_accuracy: f64, mesh_density: f64) -> Self {
        let cells = (64.0 * 16f64.powf(solid_accuracy.clamp(0.0, 1.0))).round() as u32;
        let h = cell_size_for_resolution(&mesh.bounds(), cells);
        let d = mesh_density.clamp(0.0, 1.0);
        Self { cell_size: h, min_edge_length: Some(h * (1.5 + (0.3 - 1.5) * d)), ..Default::default() }
    }
}

/// Cell size giving `cells` cells along the largest dimension of `bounds`.
pub fn cell_size_for_resolution(bounds: &Aabb, cells: u32) -> f64 {
    let ext = bounds.extents();
    let m = ext.x.max(ext.y).max(ext.z);
    if m > 0.0 {
        m / cells.max(1) as f64
    } else {
        1.0
    }
}

/// Drop triangles with non-finite vertices and validate what remains.
fn prepare(mesh: &Mesh) -> Result<(Cow<'_, Mesh>, Aabb)> {
    let m: Cow<Mesh> = match finite_part(mesh) {
        None => Cow::Borrowed(mesh),
        Some(m) => Cow::Owned(m),
    };
    let b = check_prepared(&m)?;
    Ok((m, b))
}

/// `mesh` without its triangles that have a non-finite corner, or `None` when there are none.
fn finite_part(mesh: &Mesh) -> Option<Mesh> {
    let finite = |t: u32| mesh.tri_positions(t).iter().all(|p| p.is_finite());
    if mesh.triangle_ids().all(finite) {
        None
    } else {
        let keep: Vec<u32> = mesh.triangle_ids().filter(|&t| finite(t)).collect();
        Some(mesh.submesh(&keep).0)
    }
}

/// The bounds of a prepared mesh, which must enclose something.
fn check_prepared(m: &Mesh) -> Result<Aabb> {
    if m.triangle_count() == 0 {
        return Err(SolidError::EmptyInput);
    }
    let b = m.bounds();
    if b.is_empty() || b.diagonal() <= 0.0 {
        return Err(SolidError::EmptyInput);
    }
    Ok(b)
}

fn check_cell(h: f64) -> Result<f64> {
    if h > 0.0 && h.is_finite() {
        Ok(h)
    } else {
        Err(SolidError::InvalidParameter(format!("cell size must be positive, got {h}")))
    }
}

/// Shared tail of every pipeline: extraction, mesh build, cleanup.
pub(crate) fn extract_and_clean(field: &SparseField, blocky: bool, min_edge: f64, flips: bool, shell_vol: f64, prog: &Prog) -> Result<Mesh> {
    let (pos, tris) = if blocky {
        crate::blocky::voxel_faces_with(field, &prog.sub(0.0, 0.5))?
    } else {
        marching_tetrahedra_with(field, &prog.sub(0.0, 0.5))?
    };
    let mut m = build_mesh(&pos, &tris)?;
    drop((pos, tris));
    prog.report(0.6)?;
    if !blocky {
        cleanup::collapse_short_edges(&mut m, min_edge);
        prog.report(0.8)?;
        if flips {
            cleanup::flip_to_improve(&mut m, 3);
        }
    }
    prog.report(0.9)?;
    cleanup::remove_small_shells(&mut m, shell_vol);
    prog.report(1.0)?;
    Ok(m)
}

/// The search trees the field needs: the BVH and the winding number (with [`FIELD_BETA`]),
/// built side by side.
fn build_trees(mesh: &Mesh) -> (Bvh, WindingNumber) {
    rayon::join(|| Bvh::build(mesh), || WindingNumber::with_beta(mesh, FIELD_BETA))
}

/// Make Solid's input made ready once ([`prepare_input`]): the mesh without the triangles
/// that have a non-finite corner, its bounds and its search trees. Make Solid runs on the
/// same mesh with different settings (a preview being adjusted) can share it through
/// [`make_solid_with_prepared`] and, while the field's own settings stay the same,
/// [`build_solid_field`] and [`make_solid_from_field`].
pub struct Prepared {
    mesh: Arc<Mesh>,
    is_source: bool,
    bounds: Aabb,
    bvh: Bvh,
    wn: WindingNumber,
}

impl Prepared {
    /// The mesh the solid is made from.
    pub fn mesh(&self) -> &Mesh {
        &self.mesh
    }
    /// The mesh as shared.
    pub fn mesh_arc(&self) -> &Arc<Mesh> {
        &self.mesh
    }
    /// True when no triangle had to be dropped, so the mesh is the caller's own and the
    /// trees also serve it (for instance to transfer groups or colours after a remesh).
    pub fn is_source(&self) -> bool {
        self.is_source
    }
    pub fn bounds(&self) -> Aabb {
        self.bounds
    }
    /// BVH over the mesh's triangles.
    pub fn bvh(&self) -> &Bvh {
        &self.bvh
    }
    /// Winding number of the mesh, with the field's accuracy ([`FIELD_BETA`]).
    pub fn winding(&self) -> &WindingNumber {
        &self.wn
    }
    /// The grid spacing `params` asks for on this mesh (resolving `cell_size <= 0`).
    pub fn cell_size_for(&self, params: &MakeSolidParams) -> f64 {
        if params.cell_size > 0.0 {
            params.cell_size
        } else {
            cell_size_for_resolution(&self.bounds, 128)
        }
    }
}

/// Prepare `mesh` for [`make_solid_with_prepared`]: drop the triangles with a non-finite
/// corner (keeping the shared mesh itself when there are none), and build its bounds and
/// search trees. `progress` may cancel; a cancelled build returns
/// [`SolidError::Cancelled`] and should not be kept.
pub fn prepare_input(mesh: Arc<Mesh>, progress: Progress) -> Result<Prepared> {
    let prog = Prog::new(progress);
    let (mesh, is_source) = match finite_part(&mesh) {
        None => (mesh, true),
        Some(m) => (Arc::new(m), false),
    };
    let bounds = check_prepared(&mesh)?;
    prog.report(0.1)?;
    let (bvh, wn) = build_trees(&mesh);
    prog.report(1.0)?;
    Ok(Prepared { mesh, is_source, bounds, bvh, wn })
}

/// The distance field Make Solid extracts its surface from ([`build_solid_field`]). It
/// depends on the prepared mesh, the grid spacing, the offset, the winding threshold and
/// the minimum thickness only, so the steps after it ([`make_solid_from_field`]) can run
/// again with other output settings (mode, edge collapse, flips, small shells, holes,
/// transfers) without rebuilding it.
pub struct SolidField {
    mesh: Arc<Mesh>,
    field: SparseField,
    /// Triangles the minimum thickness thickened, by id.
    thickened: Option<Vec<bool>>,
    flip_input: bool,
    /// Bits of the grid spacing, offset, winding threshold and minimum thickness.
    key: [u64; 4],
}

impl SolidField {
    /// The sampled field.
    pub fn field(&self) -> &SparseField {
        &self.field
    }
    /// The grid spacing it was sampled with.
    pub fn cell_size(&self) -> f64 {
        f64::from_bits(self.key[0])
    }
    /// Whether this field is the one `params` would build on `prepared`.
    pub fn matches(&self, prepared: &Prepared, params: &MakeSolidParams) -> bool {
        Arc::ptr_eq(&self.mesh, &prepared.mesh) && self.key == field_key(prepared.cell_size_for(params), params)
    }
}

/// What the field depends on besides the mesh, as bits.
fn field_key(h: f64, params: &MakeSolidParams) -> [u64; 4] {
    let thick = params.min_thickness.filter(|&t| t > 0.0).map_or(0, f64::to_bits);
    [h.to_bits(), params.offset.to_bits(), params.winding_threshold.to_bits(), thick]
}

/// Check the parameters the field depends on; the grid spacing.
fn check_field_params(params: &MakeSolidParams, bounds: &Aabb) -> Result<f64> {
    let h = if params.cell_size > 0.0 { params.cell_size } else { cell_size_for_resolution(bounds, 128) };
    let h = check_cell(h)?;
    let tau = params.winding_threshold;
    if !(tau > 0.0) || !tau.is_finite() {
        return Err(SolidError::InvalidParameter(format!("winding threshold {tau}")));
    }
    if !params.offset.is_finite() {
        return Err(SolidError::InvalidParameter("offset must be finite".into()));
    }
    Ok(h)
}

/// The field with what the steps after extraction need from its build.
struct FieldParts {
    field: SparseField,
    thickened: Option<Vec<bool>>,
    flip_input: bool,
}

/// Sample the field of `mesh` (steps up to the grid; `prog` from 0.05 to 0.75).
fn solid_field(mesh: &Mesh, bounds: &Aabb, bvh: &Bvh, wn: &WindingNumber, params: &MakeSolidParams, h: f64, prog: &Prog) -> Result<FieldParts> {
    let band = params.offset.abs() + DEFAULT_BAND_CELLS * h;
    let mut sdf = MeshSdf::from_refs(mesh, bvh, wn, params.offset, params.winding_threshold, band);
    let mut thickened: Option<Vec<bool>> = None;
    if let Some(t) = params.min_thickness.filter(|&t| t > 0.0) {
        let (tris, radii) = thin_triangles(&sdf, t, h);
        let mut flags = vec![false; mesh.max_tid()];
        for &t in &tris {
            flags[t as usize] = true;
        }
        thickened = Some(flags);
        sdf.set_thin_shell(&tris, radii);
    }
    prog.report(0.05)?;
    let grid = GridSpec::covering(bounds, h, sdf.band() + 2.0 * h)?;
    let field = build_field_with(&sdf, grid, &prog.sub(0.05, 0.75))?;
    let flip_input = measure::volume(mesh) < 0.0;
    Ok(FieldParts { field, thickened, flip_input })
}

/// The surface of the field, cleaned (`prog` from 0.75 to 0.92).
fn extract_solid(field: &SparseField, params: &MakeSolidParams, h: f64, prog: &Prog) -> Result<Mesh> {
    let blocky = params.mode == SolidMode::Blocky;
    let min_edge = params.min_edge_length.unwrap_or(0.3 * h);
    let shell_vol = params.min_shell_volume.unwrap_or(8.0 * h * h * h);
    extract_and_clean(field, blocky, min_edge, params.improve_quality, shell_vol, &prog.sub(0.75, 0.92))
}

/// The steps after extraction: projection and sharpening, hole caps, transfers (`prog`
/// from 0.92 to 1).
#[allow(clippy::too_many_arguments)]
fn finish_solid(mut out: Mesh, mesh: &Mesh, bvh: &Bvh, thickened: Option<&[bool]>, flip_input: bool, params: &MakeSolidParams, h: f64, prog: &Prog) -> Result<Mesh> {
    let min_edge = params.min_edge_length.unwrap_or(0.3 * h);
    if matches!(params.mode, SolidMode::Accurate | SolidMode::SharpEdges) {
        cleanup::project_to_input(&mut out, mesh, bvh, h, params.offset, flip_input, thickened);
        if params.mode == SolidMode::SharpEdges {
            cleanup::sharpen_features(&mut out, mesh, bvh, h, params.offset, flip_input, thickened, params.sharp_angle_deg);
        }
        cleanup::collapse_short_edges(&mut out, 0.5 * min_edge);
        if params.improve_quality {
            cleanup::flip_to_improve(&mut out, 2);
        }
    }
    prog.report(0.96)?;
    if !params.close_holes {
        let extra = params.min_thickness.filter(|&t| t > 0.0).map_or(0.0, |t| 0.5 * t);
        remove_hole_caps(&mut out, mesh, bvh, h, params.offset, extra);
    }
    if params.transfer_groups {
        transfer_face_groups_with(&mut out, mesh, bvh);
    }
    let mut out = compact(out);
    if params.transfer_colors && mesh.has_vertex_colors() {
        transfer_vertex_colors_with(&mut out, mesh, bvh);
    }
    prog.report(1.0)?;
    Ok(out)
}

/// The field, the surface and the steps after it, for a prepared mesh and its trees.
#[allow(clippy::too_many_arguments)]
fn solid_pipeline(mesh: &Mesh, bounds: &Aabb, bvh: &Bvh, wn: &WindingNumber, params: &MakeSolidParams, h: f64, prog: &Prog) -> Result<Mesh> {
    let FieldParts { field, thickened, flip_input } = solid_field(mesh, bounds, bvh, wn, params, h, prog)?;
    let out = extract_solid(&field, params, h, prog)?;
    // The field is no longer needed: free it before the steps that copy the mesh.
    drop(field);
    finish_solid(out, mesh, bvh, thickened.as_deref(), flip_input, params, h, prog)
}

/// Turn an arbitrary triangle mesh (holes, self-intersections, overlapping shells,
/// inconsistent orientation) into a closed, 2-manifold, outward-oriented solid.
///
/// Pipeline: narrow-band signed distance field on a sparse 8³-brick grid (exact BVH
/// distances, fast-winding-number sign) → marching tetrahedra on the Kuhn triangulation
/// (manifold by construction) → sliver collapse / flips → small-shell removal → optional
/// projection ([`SolidMode::Accurate`]) or hole-cap removal (`close_holes == false`).
/// Internal voids (inner shells) of the input are preserved as inward-facing shells.
///
/// The result may be empty when nothing encloses volume (e.g. a single open sheet without
/// `min_thickness`). Inputs are first stripped of triangles with non-finite vertices.
pub fn make_solid(mesh: &Mesh, params: &MakeSolidParams, progress: Progress) -> Result<Mesh> {
    make_solid_with(mesh, params, None, progress)
}

/// [`make_solid`] with the search trees of `mesh` built beforehand: its BVH and its winding
/// number with the field's accuracy ([`FIELD_BETA`]), both over all its triangles (as
/// `Bvh::build` and `WindingNumber::with_beta` build them). The result is the same. The
/// trees are used only when every triangle is kept (a mesh with non-finite corners is
/// stripped of them first, and the trees are then built for what remains) and when the
/// winding number's β is the field's.
pub fn make_solid_with(mesh: &Mesh, params: &MakeSolidParams, trees: Option<(&Bvh, &WindingNumber)>, progress: Progress) -> Result<Mesh> {
    let prog = Prog::new(progress);
    let (mesh, bounds) = prepare(mesh)?;
    let given = trees.filter(|(_, wn)| matches!(mesh, Cow::Borrowed(_)) && wn.beta() == FIELD_BETA);
    let mesh: &Mesh = &mesh;
    let h = check_field_params(params, &bounds)?;
    let built;
    let (bvh, wn) = match given {
        Some(t) => t,
        None => {
            built = build_trees(mesh);
            (&built.0, &built.1)
        }
    };
    prog.report(0.03)?;
    solid_pipeline(mesh, &bounds, bvh, wn, params, h, &prog)
}

/// [`make_solid`] of a prepared mesh: the same result, without stripping the mesh or
/// building its trees again.
pub fn make_solid_with_prepared(prepared: &Prepared, params: &MakeSolidParams, progress: Progress) -> Result<Mesh> {
    let prog = Prog::new(progress);
    let h = check_field_params(params, &prepared.bounds)?;
    prog.report(0.03)?;
    solid_pipeline(&prepared.mesh, &prepared.bounds, &prepared.bvh, &prepared.wn, params, h, &prog)
}

/// The distance field [`make_solid_with_prepared`] would extract its surface from, to keep
/// for [`make_solid_from_field`].
pub fn build_solid_field(prepared: &Prepared, params: &MakeSolidParams, progress: Progress) -> Result<SolidField> {
    let h = check_field_params(params, &prepared.bounds)?;
    // The field's part of the whole run (0.05 to 0.75) spread over the full range.
    let prog = Prog::new(progress).sub(-0.05 / 0.7, 0.95 / 0.7);
    let FieldParts { field, thickened, flip_input } = solid_field(&prepared.mesh, &prepared.bounds, &prepared.bvh, &prepared.wn, params, h, &prog)?;
    Ok(SolidField { mesh: prepared.mesh.clone(), field, thickened, flip_input, key: field_key(h, params) })
}

/// The steps of Make Solid after the field (extraction, cleanup, projection, sharpening,
/// hole caps, transfers) on a field built by [`build_solid_field`] for the same prepared
/// mesh and the same grid spacing, offset, winding threshold and minimum thickness
/// ([`SolidField::matches`]): the same result as [`make_solid_with_prepared`] with `params`.
pub fn make_solid_from_field(prepared: &Prepared, field: &SolidField, params: &MakeSolidParams, progress: Progress) -> Result<Mesh> {
    if !field.matches(prepared, params) {
        return Err(SolidError::InvalidParameter("the field was built for another mesh or other field settings".into()));
    }
    let h = field.cell_size();
    // The steps after the field (0.75 to 1 of the whole run) spread over the full range.
    let prog = Prog::new(progress).sub(-3.0, 1.0);
    let out = extract_solid(&field.field, params, h, &prog)?;
    finish_solid(out, &prepared.mesh, &prepared.bvh, field.thickened.as_deref(), field.flip_input, params, h, &prog)
}

/// Give each vertex of `out` the colour of `input` at its nearest point (interpolated from
/// the input's vertex colours). Does nothing when the input has no colours.
pub fn transfer_vertex_colors(out: &mut Mesh, input: &Mesh) {
    if input.has_vertex_colors() {
        transfer_vertex_colors_with(out, input, &Bvh::build(input));
    }
}

/// [`transfer_vertex_colors`] with the BVH of `input` built beforehand (it must be current).
/// When the input has no colours it still walks every vertex, so check
/// [`Mesh::has_vertex_colors`] first.
pub fn transfer_vertex_colors_with(out: &mut Mesh, input: &Mesh, bvh: &Bvh) {
    let ids: Vec<u32> = out.vertex_ids().collect();
    let snapshot: &Mesh = out;
    let cols: Vec<Option<[f32; 3]>> = ids
        .par_iter()
        .map(|&v| bvh.nearest(input, snapshot.vertex(v), f64::INFINITY).and_then(|h| input.color_at(h.tid, h.bary)))
        .collect();
    for (&v, c) in ids.iter().zip(cols) {
        if let Some(c) = c {
            out.set_vertex_color(v, c);
        }
    }
}

/// Give each triangle of `out` the face group of the nearest triangle of `input` (measured
/// from its centroid), then let single triangles stranded inside another group join it.
pub fn transfer_face_groups(out: &mut Mesh, input: &Mesh) {
    transfer_face_groups_with(out, input, &Bvh::build(input));
}

/// [`transfer_face_groups`] with the BVH of `input` built beforehand (it must be current).
pub fn transfer_face_groups_with(out: &mut Mesh, input: &Mesh, bvh: &Bvh) {
    let ids: Vec<u32> = out.triangle_ids().collect();
    let snapshot: &Mesh = out;
    let groups: Vec<Option<i32>> = ids
        .par_iter()
        .map(|&t| bvh.nearest(input, snapshot.tri_centroid(t), f64::INFINITY).map(|hit| input.tri_group(hit.tid)))
        .collect();
    for (&t, g) in ids.iter().zip(groups) {
        if let Some(g) = g {
            out.set_tri_group(t, g);
        }
    }
    let islands: Vec<(u32, i32)> = ids
        .iter()
        .filter_map(|&t| {
            let nb: Vec<i32> = out
                .tri_edges(t)
                .iter()
                .map(|&e| out.edge_other_triangle(e, t))
                .filter(|&o| o != mm_core::INVALID)
                .map(|o| out.tri_group(o))
                .collect();
            let g = *nb.first()?;
            (nb.len() == 3 && nb.iter().all(|&x| x == g) && g != out.tri_group(t)).then_some((t, g))
        })
        .collect();
    for (t, g) in islands {
        out.set_tri_group(t, g);
    }
}

/// Compact ids by round-tripping through flat arrays (lower peak memory than
/// `Mesh::compacted`, which keeps both copies alive).
pub(crate) fn compact(m: Mesh) -> Mesh {
    let (p, t, g) = m.to_arrays();
    drop(m);
    Mesh::from_triangles(&p, &t, Some(&g)).0
}

/// Robust offset surface: the solid of `mesh` grown (`distance > 0`) or shrunk
/// (`distance < 0`), computed on a sparse grid with spacing `cell_size`.
pub fn offset_solid(mesh: &Mesh, distance: f64, cell_size: f64, progress: Progress) -> Result<Mesh> {
    let params = MakeSolidParams { cell_size: check_cell(cell_size)?, offset: distance, ..Default::default() };
    make_solid(mesh, &params, progress)
}

/// Hollow the solid of `mesh`, keeping a wall of `wall_thickness` inside the original
/// surface. The result has the outer surface plus an inward-facing inner shell.
pub fn hollow(mesh: &Mesh, wall_thickness: f64, cell_size: f64, progress: Progress) -> Result<Mesh> {
    let prog = Prog::new(progress);
    let (mesh, bounds) = prepare(mesh)?;
    let mesh: &Mesh = &mesh;
    let h = check_cell(cell_size)?;
    if !(wall_thickness > 0.0) || !wall_thickness.is_finite() {
        return Err(SolidError::InvalidParameter(format!("wall thickness {wall_thickness}")));
    }
    let margin = DEFAULT_BAND_CELLS * h;
    let base = MeshSdf::with_band(mesh, 0.0, 0.5, wall_thickness + 2.0 * margin);
    let src = HollowField { base, thickness: wall_thickness, far: margin };
    let grid = GridSpec::covering(&bounds, h, src.base.band() + 2.0 * h)?;
    let field = build_field_with(&src, grid, &prog.sub(0.0, 0.8))?;
    let out = extract_and_clean(&field, false, 0.3 * h, true, 8.0 * h * h * h, &prog.sub(0.8, 1.0))?;
    Ok(compact(out))
}

/// Approximate voxel boolean of the solids of `a` and `b` (each made solid by winding
/// number, so they may be open or self-intersecting). Useful to merge overlapping vessel
/// pieces into one watertight surface.
pub fn sdf_boolean(a: &Mesh, b: &Mesh, op: BoolOp, cell_size: f64, progress: Progress) -> Result<Mesh> {
    let prog = Prog::new(progress);
    let (a, ba) = prepare(a)?;
    let (b, bb) = prepare(b)?;
    let (a, b): (&Mesh, &Mesh) = (&a, &b);
    let h = check_cell(cell_size)?;
    let fa = MeshSdf::new(a, h, 0.0, 0.5);
    let fb = MeshSdf::new(b, h, 0.0, 0.5);
    let pad = fa.band().max(fb.band()) + 2.0 * h;
    let bounds = match op {
        BoolOp::Union => ba.union(&bb),
        BoolOp::Difference => ba,
        BoolOp::Intersection => {
            let i = Aabb { min: ba.min.max(bb.min), max: ba.max.min(bb.max) };
            if i.min.cmpgt(i.max).any() {
                return Ok(Mesh::new());
            }
            i
        }
    };
    let src = BoolField { a: fa, b: fb, op };
    let grid = GridSpec::covering(&bounds, h, pad)?;
    let field = build_field_with(&src, grid, &prog.sub(0.0, 0.8))?;
    let out = extract_and_clean(&field, false, 0.3 * h, true, 8.0 * h * h * h, &prog.sub(0.8, 1.0))?;
    Ok(compact(out))
}

/// Mesh the region `{p : f(p) < 0}` inside `bounds` with a grid of spacing `cell_size`.
/// `lipschitz` must bound |∇f| (1 for exact distance fields) so that empty space can be
/// skipped. The result is closed (it is clipped by the grid boundary if needed),
/// 2-manifold and outward oriented.
pub fn mesh_implicit<F: Fn(DVec3) -> f64 + Sync>(
    f: F,
    lipschitz: f64,
    bounds: &Aabb,
    cell_size: f64,
    progress: Progress,
) -> Result<Mesh> {
    let prog = Prog::new(progress);
    let h = check_cell(cell_size)?;
    let src = ImplicitField::new(f, lipschitz, 3.0 * h * lipschitz.max(1e-6));
    let grid = GridSpec::covering(bounds, h, 2.0 * h)?;
    let field = build_field_with(&src, grid, &prog.sub(0.0, 0.7))?;
    let out = extract_and_clean(&field, false, 0.3 * h, true, 0.0, &prog.sub(0.7, 1.0))?;
    Ok(compact(out))
}

/// Triangles that need thickening for `min_thickness = t`, with per-triangle radius.
/// Sheets (both sides outside, i.e. open single-layer surfaces) get `t/2`; triangles of
/// closed walls whose inward ray hits the opposite wall at distance `s < t` get `(t−s)/2`
/// (which, applied to both walls, makes the wall about `t` thick).
fn thin_triangles(sdf: &MeshSdf, t: f64, h: f64) -> (Vec<u32>, Vec<f64>) {
    let mesh = sdf.mesh();
    let eps = (0.05 * h).min(0.05 * t);
    let ids: Vec<u32> = mesh.triangle_ids().collect();
    let res: Vec<Option<(u32, f64)>> = ids
        .par_iter()
        .map(|&tid| {
            let n = mesh.tri_normal(tid);
            if n == DVec3::ZERO {
                return None;
            }
            let c = mesh.tri_centroid(tid);
            let front = sdf.indicator(c + n * eps) < 0.0;
            let back = sdf.indicator(c - n * eps) < 0.0;
            if front == back {
                // Sheet (or internal wall): thicken sheets only.
                return (!front).then_some((tid, 0.5 * t));
            }
            // Proper boundary triangle: measure the wall thickness along the inward normal.
            let inward = if back { -n } else { n };
            let ray = Ray::new(c + inward * eps, inward);
            let hit = sdf.bvh().raycast_filtered(mesh, &ray, t, |x| x != tid)?;
            let s = hit.t + eps;
            (s < t).then_some((tid, 0.5 * (t - s)))
        })
        .collect();
    let mut tris = Vec::new();
    let mut radii = Vec::new();
    for (tid, r) in res.into_iter().flatten() {
        if r > 0.0 {
            tris.push(tid);
            radii.push(r);
        }
    }
    (tris, radii)
}

/// Remove the parts of a solidified surface that cap the input's holes (for
/// `close_holes == false`), then clean up so the result stays manifold.
///
/// Every output vertex on the (offset) input surface satisfies `d ≈ |offset|`, whereas
/// vertices on hole caps lie on the `|w| = τ` surface where the modulated field
/// `g(w)·d = offset` forces `d > |offset|`. Triangles whose three vertices are farther than
/// `|offset| + 0.5 h + extra` from the input are therefore cap triangles.
fn remove_hole_caps(out: &mut Mesh, input: &Mesh, bvh: &Bvh, h: f64, offset: f64, extra: f64) -> usize {
    let ids: Vec<u32> = out.vertex_ids().collect();
    let lim = offset.abs() + 0.5 * h + extra;
    let flags: Vec<bool> = ids.par_iter().map(|&v| bvh.nearest(input, out.vertex(v), lim).is_none()).collect();
    let mut is_cap = vec![false; out.max_vid()];
    for (&v, f) in ids.iter().zip(flags) {
        is_cap[v as usize] = f;
    }
    let caps: Vec<u32> = out.triangle_ids().filter(|&t| out.triangle(t).iter().all(|&v| is_cap[v as usize])).collect();
    let n = caps.len();
    out.remove_triangles(caps);
    make_manifold_after_removal(out);
    n
}

/// Remove triangles until no bowtie vertices remain (keeping the largest fan at each).
fn make_manifold_after_removal(m: &mut Mesh) {
    for _ in 0..64 {
        let bowties: Vec<u32> = m.vertex_ids().filter(|&v| m.is_bowtie_vertex(v)).collect();
        if bowties.is_empty() {
            break;
        }
        for v in bowties {
            if !m.is_vertex(v) || !m.is_bowtie_vertex(v) {
                continue;
            }
            let fans = vertex_fans(m, v);
            let keep = fans.iter().enumerate().max_by_key(|(_, f)| f.len()).map(|(i, _)| i).unwrap_or(0);
            let remove: Vec<u32> = fans.iter().enumerate().filter(|(i, _)| *i != keep).flat_map(|(_, f)| f.iter().copied()).collect();
            m.remove_triangles(remove);
        }
    }
    // Drop tiny fragments left behind (fewer than 4 triangles).
    let comps = mm_core::topology::connected_components(m);
    for c in comps {
        if c.len() < 4 {
            m.remove_triangles(c);
        }
    }
}

fn vertex_fans(m: &Mesh, v: u32) -> Vec<Vec<u32>> {
    let tris = m.vertex_triangles(v);
    let mut seen: Vec<u32> = Vec::new();
    let mut fans = Vec::new();
    for &t0 in tris.iter() {
        if seen.contains(&t0) {
            continue;
        }
        let mut fan = vec![t0];
        seen.push(t0);
        let mut stack = vec![t0];
        while let Some(t) = stack.pop() {
            for e in m.tri_edges(t) {
                let [a, b] = m.edge_vertices(e);
                if a != v && b != v {
                    continue;
                }
                let o = m.edge_other_triangle(e, t);
                if o != mm_core::INVALID && !seen.contains(&o) {
                    seen.push(o);
                    fan.push(o);
                    stack.push(o);
                }
            }
        }
        fans.push(fan);
    }
    fans
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_core::primitives;

    fn same(a: &Mesh, b: &Mesh) -> bool {
        a.max_vid() == b.max_vid()
            && a.max_tid() == b.max_tid()
            && a.vertex_ids().eq(b.vertex_ids())
            && a.vertex_ids().all(|v| a.vertex(v).to_array().map(f64::to_bits) == b.vertex(v).to_array().map(f64::to_bits) && a.vertex_color(v) == b.vertex_color(v))
            && a.triangle_ids().eq(b.triangle_ids())
            && a.triangle_ids().all(|t| a.triangle(t) == b.triangle(t) && a.tri_group(t) == b.tri_group(t))
    }

    /// A bent open tube with coloured vertices and two face groups.
    fn tube() -> Mesh {
        let path: Vec<DVec3> = (0..=60).map(|i| DVec3::new((i as f64 * 0.1).sin() * 2.0, i as f64 * 0.15, 0.0)).collect();
        let mut m = primitives::tube_along(&path, &vec![0.8; path.len()], 24);
        for v in m.vertex_ids().collect::<Vec<_>>() {
            let p = m.vertex(v);
            m.set_vertex_color(v, [p.x as f32 * 0.1, p.y as f32 * 0.05, 0.5]);
        }
        for t in m.triangle_ids().collect::<Vec<_>>() {
            if m.tri_centroid(t).y > 4.0 {
                m.set_tri_group(t, 3);
            }
        }
        m
    }

    #[test]
    fn every_route_gives_make_solids_result() {
        let mut nan = primitives::icosphere(2.0, 3);
        let v = nan.vertex_ids().nth(5).unwrap();
        nan.set_vertex(v, DVec3::NAN);
        let base = MakeSolidParams { cell_size: 0.12, ..Default::default() };
        let variants = [
            base.clone(),
            MakeSolidParams { mode: SolidMode::Accurate, transfer_groups: true, ..base.clone() },
            MakeSolidParams { mode: SolidMode::SharpEdges, offset: 0.2, ..base.clone() },
            MakeSolidParams { mode: SolidMode::Blocky, ..base.clone() },
            MakeSolidParams { close_holes: false, min_thickness: Some(0.5), ..base.clone() },
            MakeSolidParams { offset: -0.25, min_edge_length: Some(0.05), improve_quality: false, ..base.clone() },
        ];
        for (name, m) in [("tube", tube()), ("sphere with a NaN vertex", nan), ("sphere", primitives::icosphere(2.0, 3))] {
            let shared = Arc::new(m.clone());
            let prepared = prepare_input(shared.clone(), None).unwrap();
            assert_eq!(prepared.is_source(), Arc::ptr_eq(prepared.mesh_arc(), &shared), "{name}");
            let (bvh, wn) = build_trees(&m);
            let other_beta = WindingNumber::new(&m);
            for p in &variants {
                let want = make_solid(&m, p, None).unwrap();
                assert!(same(&want, &make_solid_with(&m, p, Some((&bvh, &wn)), None).unwrap()), "{name} {p:?}: given trees");
                assert!(same(&want, &make_solid_with(&m, p, Some((&bvh, &other_beta)), None).unwrap()), "{name} {p:?}: another beta");
                assert!(same(&want, &make_solid_with_prepared(&prepared, p, None).unwrap()), "{name} {p:?}: prepared");
                let field = build_solid_field(&prepared, p, None).unwrap();
                assert!(field.matches(&prepared, p));
                assert!(same(&want, &make_solid_from_field(&prepared, &field, p, None).unwrap()), "{name} {p:?}: from the field");
            }
            // A field serves every setting it does not depend on.
            let field = build_solid_field(&prepared, &base, None).unwrap();
            for p in [&variants[1], &variants[3]] {
                assert!(same(&make_solid(&m, p, None).unwrap(), &make_solid_from_field(&prepared, &field, p, None).unwrap()), "{name} {p:?}: shared field");
            }
            assert!(!field.matches(&prepared, &variants[2]));
            assert!(make_solid_from_field(&prepared, &field, &variants[2], None).is_err());
        }
    }

    #[test]
    fn prepared_input_cancels() {
        let m = Arc::new(primitives::icosphere(1.0, 3));
        assert_eq!(prepare_input(m.clone(), Some(&|_| false)).err(), Some(SolidError::Cancelled));
        let prepared = prepare_input(m, None).unwrap();
        let p = MakeSolidParams { cell_size: 0.1, ..Default::default() };
        assert_eq!(make_solid_with_prepared(&prepared, &p, Some(&|_| false)).err(), Some(SolidError::Cancelled));
        assert_eq!(build_solid_field(&prepared, &p, Some(&|_| false)).err(), Some(SolidError::Cancelled));
    }
}
