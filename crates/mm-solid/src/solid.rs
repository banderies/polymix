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
    let finite = |t: u32| mesh.tri_positions(t).iter().all(|p| p.is_finite());
    let m: Cow<Mesh> = if mesh.triangle_ids().all(finite) {
        Cow::Borrowed(mesh)
    } else {
        let keep: Vec<u32> = mesh.triangle_ids().filter(|&t| finite(t)).collect();
        Cow::Owned(mesh.submesh(&keep).0)
    };
    if m.triangle_count() == 0 {
        return Err(SolidError::EmptyInput);
    }
    let b = m.bounds();
    if b.is_empty() || b.diagonal() <= 0.0 {
        return Err(SolidError::EmptyInput);
    }
    Ok((m, b))
}

fn check_cell(h: f64) -> Result<f64> {
    if h > 0.0 && h.is_finite() {
        Ok(h)
    } else {
        Err(SolidError::InvalidParameter(format!("cell size must be positive, got {h}")))
    }
}

/// Shared tail of every pipeline: extraction, mesh build, cleanup.
fn extract_and_clean(field: &SparseField, blocky: bool, min_edge: f64, flips: bool, shell_vol: f64, prog: &Prog) -> Result<Mesh> {
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
    let prog = Prog::new(progress);
    let (mesh, bounds) = prepare(mesh)?;
    let mesh: &Mesh = &mesh;
    let h = if params.cell_size > 0.0 { params.cell_size } else { cell_size_for_resolution(&bounds, 128) };
    let h = check_cell(h)?;
    let tau = params.winding_threshold;
    if !(tau > 0.0) || !tau.is_finite() {
        return Err(SolidError::InvalidParameter(format!("winding threshold {tau}")));
    }
    if !params.offset.is_finite() {
        return Err(SolidError::InvalidParameter("offset must be finite".into()));
    }
    let band = params.offset.abs() + DEFAULT_BAND_CELLS * h;
    let bvh = Bvh::build(mesh);
    let wn = WindingNumber::with_beta(mesh, FIELD_BETA);
    prog.report(0.03)?;
    let mut sdf = MeshSdf::from_parts(mesh, bvh, wn, params.offset, tau, band);
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
    let grid = GridSpec::covering(&bounds, h, sdf.band() + 2.0 * h)?;
    let field = build_field_with(&sdf, grid, &prog.sub(0.05, 0.75))?;
    let blocky = params.mode == SolidMode::Blocky;
    let min_edge = params.min_edge_length.unwrap_or(0.3 * h);
    let shell_vol = params.min_shell_volume.unwrap_or(8.0 * h * h * h);
    let mut out = extract_and_clean(&field, blocky, min_edge, params.improve_quality, shell_vol, &prog.sub(0.75, 0.92))?;
    drop(field);
    let flip_input = measure::volume(mesh) < 0.0;
    if matches!(params.mode, SolidMode::Accurate | SolidMode::SharpEdges) {
        cleanup::project_to_input(&mut out, mesh, sdf.bvh(), h, params.offset, flip_input, thickened.as_deref());
        if params.mode == SolidMode::SharpEdges {
            let exclude = thickened.as_deref();
            cleanup::sharpen_features(&mut out, mesh, sdf.bvh(), h, params.offset, flip_input, exclude, params.sharp_angle_deg);
        }
        cleanup::collapse_short_edges(&mut out, 0.5 * min_edge);
        if params.improve_quality {
            cleanup::flip_to_improve(&mut out, 2);
        }
    }
    prog.report(0.96)?;
    if !params.close_holes {
        let extra = params.min_thickness.filter(|&t| t > 0.0).map_or(0.0, |t| 0.5 * t);
        remove_hole_caps(&mut out, mesh, sdf.bvh(), h, params.offset, extra);
    }
    if params.transfer_groups {
        transfer_face_groups_with(&mut out, mesh, sdf.bvh());
    }
    let out = compact(out);
    prog.report(1.0)?;
    Ok(out)
}

/// Give each triangle of `out` the face group of the nearest triangle of `input` (measured
/// from its centroid), then let single triangles stranded inside another group join it.
pub fn transfer_face_groups(out: &mut Mesh, input: &Mesh) {
    transfer_face_groups_with(out, input, &Bvh::build(input));
}

fn transfer_face_groups_with(out: &mut Mesh, input: &Mesh, bvh: &Bvh) {
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
fn compact(m: Mesh) -> Mesh {
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
