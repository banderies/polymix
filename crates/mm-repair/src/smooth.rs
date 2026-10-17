//! Smoothing and fairing (Meshmixer *Deform > Smooth* and selection smoothing).
//!
//! * [`smooth_vertices`] — explicit Laplacian or Taubin λ|μ smoothing (Taubin, "A Signal
//!   Processing Approach to Fair Surface Design", SIGGRAPH 1995), Jacobi-style updates
//!   computed in parallel.
//! * [`fair_region`] — implicit membrane (order 1) or thin-plate (order 2) fairing of a set
//!   of vertices with every other vertex held fixed, so the faired region blends smoothly
//!   (C⁰ resp. C¹) into its surroundings ("select a bump and smooth it away").

use crate::laplacian::{directed_weight, fair_vertices, LaplacianWeights};
use crate::sparse::{PcgParams, PcgStats};
use crate::util::LocalIndex;
use crate::{RepairError, Result};
use glam::DVec3;
use mm_core::mesh::Mesh;
use rayon::prelude::*;
use smallvec::SmallVec;

/// Explicit smoothing scheme.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SmoothMethod {
    /// Plain Laplacian steps `x += λ Δx` (shrinks).
    Laplacian,
    /// Taubin λ|μ: each iteration is a λ step followed by an inflating μ step. `mu` should
    /// be negative with `|mu|` slightly larger than the strength λ (e.g. λ = 0.5,
    /// μ = −0.53); a positive value is interpreted as its negation.
    Taubin {
        /// Inflation factor μ (< 0).
        mu: f64,
    },
}

/// Parameters of [`smooth_vertices`].
#[derive(Clone, Copy, Debug)]
pub struct SmoothParams {
    /// Number of iterations (a Taubin iteration is a λ and a μ step).
    pub iterations: usize,
    /// Step size λ in (0, 1].
    pub strength: f64,
    /// Laplacian weights.
    pub weights: LaplacianWeights,
    /// Keep mesh-boundary vertices fixed. If false they are smoothed along the boundary
    /// curve (using only their boundary neighbours) so open edges do not pull inwards.
    pub preserve_boundary: bool,
    /// Plain Laplacian or Taubin λ|μ.
    pub method: SmoothMethod,
    /// For closed meshes: after smoothing, offset the moved vertices along their normals so
    /// that the enclosed volume is restored.
    pub preserve_volume: bool,
}

impl Default for SmoothParams {
    fn default() -> Self {
        Self {
            iterations: 10,
            strength: 0.5,
            weights: LaplacianWeights::Uniform,
            preserve_boundary: true,
            method: SmoothMethod::Taubin { mu: -0.53 },
            preserve_volume: false,
        }
    }
}

/// Precomputed one-ring of a vertex being smoothed.
struct Stencil {
    v: u32,
    /// (edge, neighbour) pairs used for the average.
    ring: SmallVec<[(u32, u32); 8]>,
    /// Boundary vertex smoothed along its boundary curve (always uniform weights).
    curve: bool,
}

/// Smooth `vertices` (all vertices if `None`) in place. Returns the number of vertices that
/// were allowed to move.
pub fn smooth_vertices(mesh: &mut Mesh, vertices: Option<&[u32]>, params: &SmoothParams) -> usize {
    smooth_vertices_with_progress(mesh, vertices, params, None).unwrap_or(0)
}

/// [`smooth_vertices`] with a progress callback (checked once per iteration; returning
/// `false` stops early with [`RepairError::Cancelled`], leaving the partially smoothed mesh).
pub fn smooth_vertices_with_progress(
    mesh: &mut Mesh,
    vertices: Option<&[u32]>,
    params: &SmoothParams,
    progress: Option<&dyn Fn(f32) -> bool>,
) -> Result<usize> {
    if !(params.strength.is_finite()) {
        return Err(RepairError::InvalidParameter("strength must be finite".into()));
    }
    let lambda = params.strength.clamp(0.0, 1.0);
    let candidates: Vec<u32> = match vertices {
        Some(vs) => {
            let mut seen = LocalIndex::new(mesh.max_vid(), vs.len());
            let mut out = Vec::with_capacity(vs.len());
            for &v in vs {
                if mesh.is_vertex(v) && seen.get(v).is_none() {
                    seen.insert(v, 0);
                    out.push(v);
                }
            }
            out
        }
        None => mesh.vertex_ids().collect(),
    };
    let m: &Mesh = mesh;
    let make = |&v: &u32| -> Option<Stencil> {
        if m.vertex_valence(v) == 0 {
            return None;
        }
        let boundary = m.is_boundary_vertex(v);
        if boundary && params.preserve_boundary {
            return None;
        }
        let mut ring = SmallVec::new();
        for &e in m.vertex_edges(v) {
            if !boundary || m.is_boundary_edge(e) {
                ring.push((e, m.edge_other_vertex(e, v)));
            }
        }
        if ring.is_empty() || (boundary && ring.len() != 2) {
            return None;
        }
        Some(Stencil { v, ring, curve: boundary })
    };
    let stencils: Vec<Stencil> = if candidates.len() >= 2048 {
        candidates.par_iter().filter_map(make).collect()
    } else {
        candidates.iter().filter_map(make).collect()
    };
    if stencils.is_empty() || params.iterations == 0 {
        return Ok(stencils.len());
    }
    let closed_volume = if params.preserve_volume && mesh.is_closed() { Some(mm_core::measure::volume(mesh)) } else { None };

    let mu = match params.method {
        SmoothMethod::Laplacian => None,
        SmoothMethod::Taubin { mu } => Some(-mu.abs()),
    };
    let mut newpos = vec![DVec3::ZERO; stencils.len()];
    for it in 0..params.iterations {
        if let Some(cb) = progress {
            if !cb(it as f32 / params.iterations as f32) {
                return Err(RepairError::Cancelled);
            }
        }
        laplacian_step(mesh, &stencils, params.weights, lambda, &mut newpos);
        if let Some(mu) = mu {
            laplacian_step(mesh, &stencils, params.weights, mu, &mut newpos);
        }
    }
    if let Some(v0) = closed_volume {
        let moved: Vec<u32> = stencils.iter().map(|s| s.v).collect();
        restore_volume(mesh, &moved, v0);
    }
    mesh.bump_shape_version();
    Ok(stencils.len())
}

fn laplacian_step(mesh: &mut Mesh, stencils: &[Stencil], kind: LaplacianWeights, step: f64, newpos: &mut [DVec3]) {
    let m: &Mesh = mesh;
    let compute = |(s, out): (&Stencil, &mut DVec3)| {
        let p = m.vertex(s.v);
        let mut acc = DVec3::ZERO;
        let mut wsum = 0.0;
        for &(e, nb) in &s.ring {
            let w = if s.curve { 1.0 } else { directed_weight(m, e, s.v, kind) };
            acc += (m.vertex(nb) - p) * w;
            wsum += w;
        }
        *out = if wsum > 0.0 { p + acc * (step / wsum) } else { p };
    };
    if stencils.len() >= 2048 {
        stencils.par_iter().zip(newpos.par_iter_mut()).for_each(compute);
    } else {
        stencils.iter().zip(newpos.iter_mut()).for_each(compute);
    }
    let pos = mesh.positions_mut();
    for (s, p) in stencils.iter().zip(newpos.iter()) {
        pos[s.v as usize] = *p;
    }
}

/// Offset `moved` vertices along their normals until the closed mesh's volume is `target`.
pub(crate) fn restore_volume(mesh: &mut Mesh, moved: &[u32], target: f64) {
    for _ in 0..4 {
        let vol = mm_core::measure::volume(mesh);
        let normals: Vec<(u32, DVec3, f64)> = moved
            .iter()
            .map(|&v| {
                let tris = mesh.vertex_triangles(v);
                let area: f64 = tris.iter().map(|&t| mesh.tri_area(t)).sum::<f64>() / 3.0;
                (v, mesh.vertex_normal(v), area)
            })
            .collect();
        let area: f64 = normals.iter().map(|x| x.2).sum();
        if area <= 0.0 {
            return;
        }
        let d = (target - vol) / area;
        if !d.is_finite() || d.abs() < 1e-15 * (1.0 + target.abs().cbrt()) {
            return;
        }
        let pos = mesh.positions_mut();
        for (v, n, _) in normals {
            pos[v as usize] += n * d;
        }
    }
    mesh.bump_shape_version();
}

/// Parameters of [`fair_region_with`].
#[derive(Clone, Copy, Debug)]
pub struct FairParams {
    /// 1 = membrane (Laplace, C⁰ blend), 2 = thin plate (bilaplacian, C¹ blend).
    pub order: u32,
    /// Laplacian weights (cotangent recommended).
    pub weights: LaplacianWeights,
    /// Relative residual tolerance of the conjugate-gradient solve.
    pub tolerance: f64,
    /// Conjugate-gradient iteration limit.
    pub max_iterations: usize,
    /// Recompute geometric weights on the result and solve again this many extra times
    /// (only meaningful for cotangent / mean-value weights).
    pub reweight_passes: usize,
}

impl Default for FairParams {
    fn default() -> Self {
        Self { order: 2, weights: LaplacianWeights::Cotangent, tolerance: 1e-7, max_iterations: 10_000, reweight_passes: 0 }
    }
}

/// Implicitly fair `vertices_to_move` (order 1 = Laplacian, 2 = bilaplacian) with every
/// other vertex fixed. Vertices whose connected free region has no fixed neighbour are left
/// unchanged.
pub fn fair_region(mesh: &mut Mesh, vertices_to_move: &[u32], order: u32, weights: LaplacianWeights) -> Result<PcgStats> {
    fair_region_with(mesh, vertices_to_move, &FairParams { order, weights, ..Default::default() }, None)
}

/// [`fair_region`] with full parameters and a progress callback (reported per solver
/// iteration batch; returning `false` cancels and leaves the mesh unchanged).
pub fn fair_region_with(
    mesh: &mut Mesh,
    vertices_to_move: &[u32],
    params: &FairParams,
    progress: Option<&dyn Fn(f32) -> bool>,
) -> Result<PcgStats> {
    let pcg = PcgParams { tolerance: params.tolerance, max_iterations: params.max_iterations };
    let passes = 1 + if params.weights == LaplacianWeights::Uniform { 0 } else { params.reweight_passes };
    let backup: Vec<(u32, DVec3)> = vertices_to_move.iter().filter(|&&v| mesh.is_vertex(v)).map(|&v| (v, mesh.vertex(v))).collect();
    let mut stats = PcgStats::default();
    for pass in 0..passes {
        let cb = progress.map(|p| {
            move |it: usize, _res: f64| -> bool {
                let f = (pass as f32 + (it as f32 / pcg.max_iterations.max(1) as f32).min(1.0)) / passes as f32;
                p(f)
            }
        });
        let cbref: Option<&dyn Fn(usize, f64) -> bool> = cb.as_ref().map(|c| c as &dyn Fn(usize, f64) -> bool);
        // Solve without touching the mesh if cancelled: fair_vertices writes only on success.
        let st = fair_vertices(mesh, vertices_to_move, params.order, params.weights, &pcg, cbref);
        match st {
            Ok(s) if s.cancelled => {
                let pos = mesh.positions_mut();
                for &(v, p) in &backup {
                    pos[v as usize] = p;
                }
                mesh.bump_shape_version();
                return Err(RepairError::Cancelled);
            }
            Ok(s) => stats = s,
            Err(e) => {
                let pos = mesh.positions_mut();
                for &(v, p) in &backup {
                    pos[v as usize] = p;
                }
                mesh.bump_shape_version();
                return Err(e);
            }
        }
    }
    Ok(stats)
}
